/* Copyright 2025 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "legion/views/fill.h"
#include "legion/analysis/aggregator.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/region.h"
#include "legion/utilities/collectives.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // FillView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillView::FillView(
        DistributedID did,
#ifdef LEGION_SPY
        UniqueID op_uid,
#endif
        bool register_now, CollectiveMapping* map)
      : DeferredView(encode_fill_did(did), register_now, map),
#ifdef LEGION_SPY
        fill_op_uid(op_uid),
#endif
        value(nullptr), value_size(0),
        collective_first_active((map != nullptr) && map->contains(local_space))
    //--------------------------------------------------------------------------
    {
      // Add an extra reference here until we receive the value update
      add_base_resource_ref(PENDING_UNBOUND_REF);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Fill View %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FillView::FillView(
        DistributedID did,
#ifdef LEGION_SPY
        UniqueID op_uid,
#endif
        const void* val, size_t size, bool register_now, CollectiveMapping* map)
      : DeferredView(encode_fill_did(did), register_now, map),
#ifdef LEGION_SPY
        fill_op_uid(op_uid),
#endif
        value(malloc(size)), value_size(size),
        collective_first_active((map != nullptr) && map->contains(local_space))
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(value_size > 0);
#endif
      memcpy(value.load(), val, size);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Fill View %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FillView::~FillView(void)
    //--------------------------------------------------------------------------
    {
      if (value.load() != nullptr)
        free(value.load());
    }

    //--------------------------------------------------------------------------
    void FillView::pack_valid_ref(void)
    //--------------------------------------------------------------------------
    {
      pack_global_ref();
    }

    //--------------------------------------------------------------------------
    void FillView::unpack_valid_ref(void)
    //--------------------------------------------------------------------------
    {
      unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    void FillView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(collective_mapping == nullptr);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
#ifdef LEGION_SPY
        rez.serialize(fill_op_uid);
#endif
        AutoLock v_lock(view_lock);
        size_t size = value_size.load();
        rez.serialize<size_t>(size);
        if (size > 0)
          rez.serialize(value.load(), size);
        // Update the remote instances while holding the lock
        update_remote_instances(target);
      }
      runtime->send_fill_view(target, rez);
    }

    //--------------------------------------------------------------------------
    void FillView::flatten(
        CopyFillAggregator& aggregator, InstanceView* dst_view,
        const FieldMask& src_mask, IndexSpaceExpression* expr,
        PredEvent pred_guard, const PhysicalTraceInfo& trace_info,
        EquivalenceSet* tracing_eq, CopyAcrossHelper* helper)
    //--------------------------------------------------------------------------
    {
      aggregator.record_fill(
          dst_view, this, src_mask, expr, pred_guard, tracing_eq, helper);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FillView::handle_send_fill_view(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
#ifdef LEGION_SPY
      UniqueID op_uid;
      derez.deserialize(op_uid);
#endif
      size_t value_size;
      derez.deserialize(value_size);

      void* location =
          runtime->find_or_create_pending_collectable_location<FillView>(did);
      FillView* view = nullptr;
      if (value_size > 0)
      {
        const void* value = derez.get_current_pointer();
        view = new (location) FillView(
            did,
#ifdef LEGION_SPY
            op_uid,
#endif
            value, value_size, false /*register now*/);
        derez.advance_pointer(value_size);
      } else
        view = new (location) FillView(
            did,
#ifdef LEGION_SPY
            op_uid,
#endif
            false /*register now*/);
      view->register_with_runtime();
    }

    //--------------------------------------------------------------------------
    bool FillView::matches(FillView* other)
    //--------------------------------------------------------------------------
    {
      if (value == nullptr)
      {
        RtEvent wait_on;
        {
          AutoLock v_lock(view_lock);
          if (value == nullptr)
          {
            value_ready = Runtime::create_rt_user_event();
            wait_on = value_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
#ifdef DEBUG_LEGION
      assert(value != nullptr);
#endif
      return other->matches(value, value_size);
    }

    //--------------------------------------------------------------------------
    bool FillView::matches(const void* other, size_t size)
    //--------------------------------------------------------------------------
    {
      if (value == nullptr)
      {
        RtEvent wait_on;
        {
          AutoLock v_lock(view_lock);
          if (value == nullptr)
          {
            value_ready = Runtime::create_rt_user_event();
            wait_on = value_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
#ifdef DEBUG_LEGION
      assert(value != nullptr);
#endif
      if (value_size != size)
        return false;
      return (memcmp(value, other, value_size) == 0);
    }

    //--------------------------------------------------------------------------
    bool FillView::set_value(const void* val, size_t size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(size > 0);
      assert(val != nullptr);
      assert(value.load() == nullptr);
      assert(value_size.load() == 0);
#endif
      void* result = malloc(size);
      memcpy(result, val, size);
      // Take the lock and sent out any notifications
      AutoLock v_lock(view_lock);
      value_size.store(size);
      value.store(result);
      if (value_ready.exists())
        Runtime::trigger_event(value_ready);
      if (is_owner() && has_remote_instances())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(size);
          rez.serialize(val, size);
        }
        struct ValueFunctor {
          ValueFunctor(Serializer& z) : rez(z) { }
          inline void apply(AddressSpaceID target)
          {
            if (target == runtime->address_space)
              return;
            runtime->send_fill_view_value(target, rez);
          }
          Serializer& rez;
        };
        ValueFunctor functor(rez);
        map_over_remote_instances(functor);
      }
      return remove_base_resource_ref(PENDING_UNBOUND_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FillView::handle_send_fill_view_value(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      size_t size;
      derez.deserialize(size);
      const void* value = derez.get_current_pointer();
      derez.advance_pointer(size);

      // This message can arrive out-of-order with respect to the creation
      // of the fill view on the remote node, so do the normal steps
      RtEvent ready;
      FillView* view = static_cast<FillView*>(
          runtime->find_or_request_logical_view(did, ready));
      if (ready.exists() && !ready.has_triggered())
        ready.wait();

      if (view->set_value(value, size))
        delete view;
    }

    //--------------------------------------------------------------------------
    ApEvent FillView::issue_fill(
        Operation* op, IndexSpaceExpression* fill_expr,
        IndividualView* dst_view, const FieldMask& fill_mask,
        const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& dst_fields,
        std::set<RtEvent>& applied_events, PhysicalManager* manager,
        ApEvent precondition, PredEvent pred_guard,
        CollectiveKind collective_kind, bool fill_restricted)
    //--------------------------------------------------------------------------
    {
      if (value_size.load() == 0)
      {
        // We don't know the value yet so we need to launch a task to
        // actually issue the fill once we know the value
        AutoLock v_lock(view_lock);
        if (value_size.load() == 0)
        {
          if (!value_ready.exists())
            value_ready = Runtime::create_rt_user_event();
          DeferIssueFill args(
              this, op, fill_expr, dst_view, fill_mask, trace_info, dst_fields,
              manager, precondition, pred_guard, collective_kind,
              fill_restricted, applied_events);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_DEFERRED_PRIORITY, value_ready);
          return args.done;
        }
      }
      // If we get here the that means we have a value and can issue
      // the fill from this fill view
      ApEvent result = fill_expr->issue_fill(
          op, trace_info, dst_fields, value.load(), value_size.load(),
#ifdef LEGION_SPY
          fill_op_uid, manager->field_space_node->handle, manager->tree_id,
#endif
          precondition, pred_guard, manager->get_unique_event(),
          collective_kind, fill_restricted);
      if (trace_info.recording)
      {
        const UniqueInst dst_inst(dst_view);
        trace_info.record_fill_inst(
            result, fill_expr, dst_inst, fill_mask, applied_events,
            (dst_view->get_redop() > 0));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FillView::DeferIssueFill::DeferIssueFill(
        FillView* v, Operation* o, IndexSpaceExpression* expr,
        IndividualView* dst_v, const FieldMask& mask,
        const PhysicalTraceInfo& info, const std::vector<CopySrcDstField>& dst,
        PhysicalManager* man, ApEvent pre, PredEvent guard,
        CollectiveKind collect, bool fill_restrict,
        std::set<RtEvent>& applied_events)
      : LgTaskArgs<DeferIssueFill>(o->get_unique_op_id()), view(v), op(o),
        fill_expr(expr), dst_view(dst_v), fill_mask(new FieldMask(mask)),
        trace_info(new PhysicalTraceInfo(info)),
        dst_fields(new std::vector<CopySrcDstField>(dst)), manager(man),
        precondition(pre), pred_guard(guard), collective(collect),
        applied(Runtime::create_rt_user_event()),
        done(Runtime::create_ap_user_event(&info)),
        fill_restricted(fill_restrict)
    //--------------------------------------------------------------------------
    {
      view->add_base_resource_ref(META_TASK_REF);
      dst_view->add_base_resource_ref(META_TASK_REF);
      fill_expr->add_base_expression_reference(META_TASK_REF);
      manager->add_base_resource_ref(META_TASK_REF);
      applied_events.insert(applied);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FillView::handle_defer_issue_fill(const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferIssueFill* dargs = (const DeferIssueFill*)args;
      std::set<RtEvent> applied_events;
      const ApEvent result = dargs->view->issue_fill(
          dargs->op, dargs->fill_expr, dargs->dst_view, *(dargs->fill_mask),
          *(dargs->trace_info), *(dargs->dst_fields), applied_events,
          dargs->manager, dargs->precondition, dargs->pred_guard,
          dargs->collective, dargs->fill_restricted);
      Runtime::trigger_event(
          dargs->done, result, *(dargs->trace_info), applied_events);
      Runtime::trigger_event(
          dargs->applied, Runtime::merge_events(applied_events));
      delete dargs->fill_mask;
      delete dargs->trace_info;
      delete dargs->dst_fields;
      if (dargs->view->remove_base_resource_ref(META_TASK_REF))
        delete dargs->view;
      if (dargs->dst_view->remove_base_resource_ref(META_TASK_REF))
        delete dargs->dst_view;
      if (dargs->fill_expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->fill_expr;
      if (dargs->manager->remove_base_resource_ref(META_TASK_REF))
        delete dargs->manager;
    }

  }  // namespace Internal
}  // namespace Legion
