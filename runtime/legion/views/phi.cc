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

#include "legion/views/phi.h"
#include "legion/kernel/runtime.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // PhiView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhiView::PhiView(
        DistributedID did, PredEvent tguard, PredEvent fguard,
        FieldMaskSet<DeferredView>&& true_vws,
        FieldMaskSet<DeferredView>&& false_vws, bool register_now)
      : DeferredView(encode_phi_did(did), register_now), true_guard(tguard),
        false_guard(fguard), true_views(true_vws), false_views(false_vws)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(true_guard.exists());
      assert(false_guard.exists());
      assert(true_views.get_valid_mask() == false_views.get_valid_mask());
#endif
      if (register_now)
        add_initial_references(false /*unpack references*/);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Phi View %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    PhiView::~PhiView(void)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<DeferredView>::const_iterator it = true_views.begin();
           it != true_views.end(); it++)
        if (it->first->remove_nested_resource_ref(did))
          delete it->first;
      for (FieldMaskSet<DeferredView>::const_iterator it = false_views.begin();
           it != false_views.end(); it++)
        if (it->first->remove_nested_resource_ref(did))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_local(void)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<DeferredView>::const_iterator it = true_views.begin();
           it != true_views.end(); it++)
        it->first->remove_nested_gc_ref(did);
      for (FieldMaskSet<DeferredView>::const_iterator it = false_views.begin();
           it != false_views.end(); it++)
        it->first->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void PhiView::pack_valid_ref(void)
    //--------------------------------------------------------------------------
    {
      pack_global_ref();
      for (FieldMaskSet<DeferredView>::const_iterator it = true_views.begin();
           it != true_views.end(); it++)
        it->first->pack_valid_ref();
      for (FieldMaskSet<DeferredView>::const_iterator it = false_views.begin();
           it != false_views.end(); it++)
        it->first->pack_valid_ref();
    }

    //--------------------------------------------------------------------------
    void PhiView::unpack_valid_ref(void)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<DeferredView>::const_iterator it = true_views.begin();
           it != true_views.end(); it++)
        it->first->unpack_valid_ref();
      for (FieldMaskSet<DeferredView>::const_iterator it = false_views.begin();
           it != false_views.end(); it++)
        it->first->unpack_valid_ref();
      unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    void PhiView::add_initial_references(bool unpack_references)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<DeferredView>::const_iterator it = true_views.begin();
           it != true_views.end(); it++)
      {
        it->first->add_nested_resource_ref(did);
        it->first->add_nested_gc_ref(did);
        if (unpack_references)
          it->first->unpack_global_ref();
      }
      for (FieldMaskSet<DeferredView>::const_iterator it = false_views.begin();
           it != false_views.end(); it++)
      {
        it->first->add_nested_resource_ref(did);
        it->first->add_nested_gc_ref(did);
        if (unpack_references)
          it->first->unpack_global_ref();
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::send_view(AddressSpaceID target)
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
        rez.serialize(true_guard);
        rez.serialize(false_guard);
        rez.serialize<size_t>(true_views.size());
        for (FieldMaskSet<DeferredView>::const_iterator it = true_views.begin();
             it != true_views.end(); it++)
        {
          it->first->pack_global_ref();
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
        rez.serialize<size_t>(false_views.size());
        for (FieldMaskSet<DeferredView>::const_iterator it =
                 false_views.begin();
             it != false_views.end(); it++)
        {
          it->first->pack_global_ref();
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
      }
      runtime->send_phi_view(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    void PhiView::flatten(
        CopyFillAggregator& aggregator, InstanceView* dst_view,
        const FieldMask& src_mask, IndexSpaceExpression* expr,
        PredEvent pred_guard, const PhysicalTraceInfo& trace_info,
        EquivalenceSet* tracing_eq, CopyAcrossHelper* helper)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!(src_mask - true_views.get_valid_mask()));
      assert(!(src_mask - false_views.get_valid_mask()));
#endif
      const PredEvent next_true =
          !pred_guard.exists() ?
              true_guard :
              Runtime::merge_events(&trace_info, pred_guard, true_guard);
      for (FieldMaskSet<DeferredView>::const_iterator it = true_views.begin();
           it != true_views.end(); it++)
      {
        const FieldMask overlap = src_mask & it->second;
        if (!overlap)
          continue;
        it->first->flatten(
            aggregator, dst_view, overlap, expr, next_true, trace_info,
            tracing_eq, helper);
      }
      const PredEvent next_false =
          !pred_guard.exists() ?
              false_guard :
              Runtime::merge_events(&trace_info, pred_guard, false_guard);
      for (FieldMaskSet<DeferredView>::const_iterator it = false_views.begin();
           it != false_views.end(); it++)
      {
        const FieldMask overlap = src_mask & it->second;
        if (!overlap)
          continue;
        it->first->flatten(
            aggregator, dst_view, overlap, expr, next_false, trace_info,
            tracing_eq, helper);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhiView::handle_send_phi_view(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      PredEvent true_guard, false_guard;
      derez.deserialize(true_guard);
      derez.deserialize(false_guard);
      std::set<RtEvent> ready_events;
      FieldMaskSet<DeferredView> true_views, false_views;
      size_t num_true_views;
      derez.deserialize(num_true_views);
      for (unsigned idx = 0; idx < num_true_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        DeferredView* view = static_cast<DeferredView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        FieldMask mask;
        derez.deserialize(mask);
        true_views.insert(view, mask);
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
      }
      size_t num_false_views;
      derez.deserialize(num_false_views);
      for (unsigned idx = 0; idx < num_false_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        DeferredView* view = static_cast<DeferredView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        FieldMask mask;
        derez.deserialize(mask);
        false_views.insert(view, mask);
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
      }
      // Make the phi view but don't register it yet
      void* location =
          runtime->find_or_create_pending_collectable_location<PhiView>(did);
      PhiView* view = new (location) PhiView(
          did, true_guard, false_guard, std::move(true_views),
          std::move(false_views), false /*register_now*/);
      if (!ready_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(ready_events);
        DeferPhiViewRegistrationArgs args(view);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, wait_on);
      }
      else
      {
        // Add the resource references
        view->add_initial_references(true /*unpack references*/);
        view->register_with_runtime();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhiView::handle_deferred_view_registration(const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferPhiViewRegistrationArgs* pargs =
          (const DeferPhiViewRegistrationArgs*)args;
      pargs->view->add_initial_references(true /*unpack references*/);
      pargs->view->register_with_runtime();
    }

  }  // namespace Internal
}  // namespace Legion
