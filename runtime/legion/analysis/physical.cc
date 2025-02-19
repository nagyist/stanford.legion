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

#include "legion/analysis/physical.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/analysis/versioning.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/expression.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Physical Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalAnalysis::PhysicalAnalysis(
        Operation* o, unsigned idx, IndexSpaceExpression* e, bool h, bool im,
        bool ex, CollectiveMapping* m, bool first)
      : previous(runtime->address_space),
        original_source(runtime->address_space), analysis_expr(e), op(o),
        index(idx), owns_op(false), on_heap(h), exclusive(ex), immutable(im),
        collective_first_local(first), parallel_traversals(false),
        restricted(false), recorded_instances(nullptr), collective_mapping(m)
    //--------------------------------------------------------------------------
    {
      analysis_expr->add_base_expression_reference(PHYSICAL_ANALYSIS_REF);
      if (collective_mapping != nullptr)
        collective_mapping->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::PhysicalAnalysis(
        AddressSpaceID source, AddressSpaceID prev, Operation* o, unsigned idx,
        IndexSpaceExpression* e, bool h, bool im, CollectiveMapping* mapping,
        bool ex, bool first)
      : previous(prev), original_source(source), analysis_expr(e), op(o),
        index(idx), owns_op(true), on_heap(h), exclusive(ex), immutable(im),
        collective_first_local(first), parallel_traversals(false),
        restricted(false), recorded_instances(nullptr),
        collective_mapping(mapping)
    //--------------------------------------------------------------------------
    {
      analysis_expr->add_base_expression_reference(PHYSICAL_ANALYSIS_REF);
      if (collective_mapping != nullptr)
        collective_mapping->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::~PhysicalAnalysis(void)
    //--------------------------------------------------------------------------
    {
      if (deferred_applied_event.exists())
      {
        if (!deferred_applied_events.empty())
        {
          Runtime::trigger_event(
              deferred_applied_event,
              Runtime::merge_events(deferred_applied_events));
#ifdef DEBUG_LEGION
          deferred_applied_events.clear();
#endif
        }
        else
          Runtime::trigger_event(deferred_applied_event);
      }
#ifdef DEBUG_LEGION
      assert(deferred_applied_events.empty());
#endif
      if (analysis_expr->remove_base_expression_reference(
              PHYSICAL_ANALYSIS_REF))
        delete analysis_expr;
      if ((collective_mapping != nullptr) &&
          collective_mapping->remove_reference())
        delete collective_mapping;
      if (recorded_instances != nullptr)
        delete recorded_instances;
      if (owns_op && (op != nullptr))
        delete op;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::analyze(
        EquivalenceSet* set, const FieldMask& mask,
        std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
        RtEvent precondition /*= NO_EVENT*/,
        const bool already_deferred /* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!precondition.exists() || precondition.has_triggered())
      {
#ifdef LEGION_SPY_EQUIVALENCE_SETS
        LegionSpy::log_equivalence_set_use(
            set->did, op->get_unique_op_id(), index);
#endif
        if (set->set_expr == analysis_expr)
          set->analyze(
              *this, analysis_expr, true /*covers*/, mask, deferral_events,
              applied_events, already_deferred);
        else if (!set->set_expr->is_empty())
        {
          IndexSpaceExpression* expr =
              runtime->intersect_index_spaces(set->set_expr, analysis_expr);
          if (expr->is_empty())
            return;
          // Check to see this expression covers the equivalence set
          // If it does then we can use original set expression
          if (expr->get_volume() == set->set_expr->get_volume())
            set->analyze(
                *this, set->set_expr, true /*covers*/, mask, deferral_events,
                applied_events, already_deferred);
          else
            set->analyze(
                *this, expr, false /*covers*/, mask, deferral_events,
                applied_events, already_deferred);
        }
        else
          set->analyze(
              *this, set->set_expr, true /*covers*/, mask, deferral_events,
              applied_events, already_deferred);
      }
      else
        // This has to be the first time through and isn't really
        // a deferral of an the traversal since we haven't even
        // started the traversal yet
        defer_analysis(
            precondition, set, mask, deferral_events, applied_events,
            RtUserEvent::NO_RT_USER_EVENT, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::defer_traversal(
        RtEvent precondition, const VersionInfo& version_info,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have a deferred applied event yet
      if (!deferred_applied_event.exists())
      {
        deferred_applied_event = Runtime::create_rt_user_event();
        applied_events.insert(deferred_applied_event);
      }
      const DeferPerformTraversalArgs args(this, version_info);
      runtime->issue_runtime_meta_task(
          args, LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      return args.done_event;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::defer_analysis(
        RtEvent precondition, EquivalenceSet* set, const FieldMask& mask,
        std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
        RtUserEvent deferral_event, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have a deferred applied event yet
      if (!deferred_applied_event.exists())
      {
        deferred_applied_event = Runtime::create_rt_user_event();
        applied_events.insert(deferred_applied_event);
      }
      const DeferPerformAnalysisArgs args(
          this, set, mask, deferral_event, already_deferred);
      runtime->issue_runtime_meta_task(
          args, LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      deferral_events.insert(args.done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::defer_remote(
        RtEvent precondition, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have a deferred applied event yet
      if (!deferred_applied_event.exists())
      {
        deferred_applied_event = Runtime::create_rt_user_event();
        applied_events.insert(deferred_applied_event);
      }
      const DeferPerformRemoteArgs args(this);
      runtime->issue_runtime_meta_task(
          args, LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      return args.done_event;
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::defer_updates(
        RtEvent precondition, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have a deferred applied event yet
      if (!deferred_applied_event.exists())
      {
        deferred_applied_event = Runtime::create_rt_user_event();
        applied_events.insert(deferred_applied_event);
      }
      const DeferPerformUpdateArgs args(this);
      runtime->issue_runtime_meta_task(
          args, LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      return args.done_event;
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::defer_registration(
        RtEvent precondition, const RegionUsage& usage,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info,
        ApEvent init_precondition, ApEvent termination,
        ApEvent& instances_ready, bool symbolic)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have a deferred applied event yet
      if (!deferred_applied_event.exists())
      {
        deferred_applied_event = Runtime::create_rt_user_event();
        applied_events.insert(deferred_applied_event);
      }
      const DeferPerformRegistrationArgs args(
          this, usage, trace_info, init_precondition, termination, symbolic);
      runtime->issue_runtime_meta_task(
          args, LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      instances_ready = args.instances_ready;
      return args.done_event;
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalAnalysis::defer_output(
        RtEvent precondition, const PhysicalTraceInfo& trace_info,
        bool track_effects, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have a deferred applied event yet
      if (!deferred_applied_event.exists())
      {
        deferred_applied_event = Runtime::create_rt_user_event();
        applied_events.insert(deferred_applied_event);
      }
      const DeferPerformOutputArgs args(this, track_effects, trace_info);
      runtime->issue_runtime_meta_task(
          args, LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      return args.effects_event;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::record_deferred_applied_events(
        std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(*this);
      if (!deferred_applied_events.empty())
        deferred_applied_events.insert(applied.begin(), applied.end());
      else
        deferred_applied_events.swap(applied);
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::perform_traversal(
        RtEvent precondition, const VersionInfo& info,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (precondition.exists() && !precondition.has_triggered())
        return defer_traversal(precondition, info, applied_events);
      std::set<RtEvent> deferral_events;
      const FieldMaskSet<EquivalenceSet>& eq_sets = info.get_equivalence_sets();
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = eq_sets.begin();
           it != eq_sets.end(); it++)
        analyze(it->first, it->second, deferral_events, applied_events);
      if (!deferral_events.empty())
        return Runtime::merge_events(deferral_events);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::perform_remote(
        RtEvent precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // only called by derived classes
      std::abort();
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::perform_updates(
        RtEvent precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // only called by derived classes
      std::abort();
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::perform_registration(
        RtEvent precondition, const RegionUsage& usage,
        std::set<RtEvent>& applied_events, ApEvent init_precondition,
        ApEvent termination, ApEvent& instances_ready, bool symbolic)
    //--------------------------------------------------------------------------
    {
      // only called by derived classes
      std::abort();
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalAnalysis::perform_output(
        RtEvent precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // only called by derived classes
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::process_remote_instances(
        Deserializer& derez, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      size_t num_views;
      derez.deserialize(num_views);
      AutoLock a_lock(*this);
      if (recorded_instances == nullptr)
        recorded_instances = new FieldMaskSet<LogicalView>();
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        LogicalView* view =
            runtime->find_or_request_logical_view(view_did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        recorded_instances->insert(view, mask);
      }
      bool remote_restrict;
      derez.deserialize(remote_restrict);
      if (remote_restrict)
        restricted = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::process_local_instances(
        const FieldMaskSet<LogicalView>& views, const bool local_restricted)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(*this);
      if (recorded_instances == nullptr)
        recorded_instances = new FieldMaskSet<LogicalView>();
      for (FieldMaskSet<LogicalView>::const_iterator it = views.begin();
           it != views.end(); it++)
        if (it->first->is_instance_view())
          recorded_instances->insert(it->first, it->second);
      if (local_restricted)
        restricted = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::filter_remote_expressions(
        FieldMaskSet<IndexSpaceExpression>& exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      FieldMaskSet<IndexSpaceExpression> remote_exprs;
      for (LegionMap<AddressSpaceID, FieldMaskSet<EquivalenceSet> >::
               const_iterator rit = remote_sets.begin();
           rit != remote_sets.end(); rit++)
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
                 rit->second.begin();
             it != rit->second.end(); it++)
          remote_exprs.insert(it->first->set_expr, it->second);
      FieldMaskSet<IndexSpaceExpression> to_add;
      std::vector<IndexSpaceExpression*> to_remove;
      if (remote_exprs.size() > 1)
      {
        LegionList<FieldSet<IndexSpaceExpression*> > field_sets;
        remote_exprs.compute_field_sets(FieldMask(), field_sets);
        for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator fit =
                 field_sets.begin();
             fit != field_sets.end(); fit++)
        {
          IndexSpaceExpression* remote_expr =
              (fit->elements.size() == 1) ?
                  *(fit->elements.begin()) :
                  runtime->union_index_spaces(fit->elements);
          for (FieldMaskSet<IndexSpaceExpression>::iterator it = exprs.begin();
               it != exprs.end(); it++)
          {
            const FieldMask overlap = it->second & fit->set_mask;
            if (!overlap)
              continue;
            IndexSpaceExpression* diff =
                runtime->subtract_index_spaces(it->first, remote_expr);
            if (!diff->is_empty())
              to_add.insert(diff, overlap);
            it.filter(overlap);
            if (!it->second)
              to_remove.push_back(it->first);
          }
        }
      }
      else
      {
        FieldMaskSet<IndexSpaceExpression>::const_iterator first =
            remote_exprs.begin();

        for (FieldMaskSet<IndexSpaceExpression>::iterator it = exprs.begin();
             it != exprs.end(); it++)
        {
          const FieldMask overlap = it->second & first->second;
          if (!overlap)
            continue;
          IndexSpaceExpression* diff =
              runtime->subtract_index_spaces(it->first, first->first);
          if (!diff->is_empty())
            to_add.insert(diff, overlap);
          it.filter(overlap);
          if (!it->second)
            to_remove.push_back(it->first);
        }
      }
      if (!to_remove.empty())
      {
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
                 to_remove.begin();
             it != to_remove.end(); it++)
          exprs.erase(*it);
      }
      if (!to_add.empty())
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                 to_add.begin();
             it != to_add.end(); it++)
          exprs.insert(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalAnalysis::report_instances(FieldMaskSet<LogicalView>& insts)
    //--------------------------------------------------------------------------
    {
      // No need for the lock since we shouldn't be mutating anything at
      // this point anyway
      if (recorded_instances != nullptr)
        recorded_instances->swap(insts);
      return restricted;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::record_remote(
        EquivalenceSet* set, const FieldMask& mask, const AddressSpaceID owner)
    //--------------------------------------------------------------------------
    {
      if (parallel_traversals)
      {
        AutoLock a_lock(*this);
        remote_sets[owner].insert(set, mask);
      }
      else
        // No lock needed if we're the only one
        remote_sets[owner].insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::record_instance(
        LogicalView* view, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      // Lock held from caller
      if (recorded_instances == nullptr)
        recorded_instances = new FieldMaskSet<LogicalView>();
      recorded_instances->insert(view, mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_remote_instances(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      PhysicalAnalysis* target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      std::set<RtEvent> ready_events;
      target->process_remote_instances(derez, ready_events);
      if (!ready_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformTraversalArgs::DeferPerformTraversalArgs(
        PhysicalAnalysis* ana, const VersionInfo& info)
      : LgTaskArgs<DeferPerformTraversalArgs>(ana->op->get_unique_op_id()),
        analysis(ana), version_info(&info),
        done_event(Runtime::create_rt_user_event())
    //--------------------------------------------------------------------------
    {
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_traversal(
        const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformTraversalArgs* dargs =
          (const DeferPerformTraversalArgs*)args;
      std::set<RtEvent> applied_events;
      Runtime::trigger_event(
          dargs->done_event,
          dargs->analysis->perform_traversal(
              RtEvent::NO_RT_EVENT, *(dargs->version_info), applied_events));
      if (!applied_events.empty())
        dargs->analysis->record_deferred_applied_events(applied_events);
      if (dargs->analysis->on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformAnalysisArgs::DeferPerformAnalysisArgs(
        PhysicalAnalysis* ana, EquivalenceSet* s, const FieldMask& m,
        RtUserEvent done, bool def)
      : LgTaskArgs<DeferPerformAnalysisArgs>(ana->op->get_unique_op_id()),
        analysis(ana), set(s), mask(new FieldMask(m)),
        done_event(done.exists() ? done : Runtime::create_rt_user_event()),
        already_deferred(def)
    //--------------------------------------------------------------------------
    {
      analysis->record_parallel_traversals();
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_analysis(const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformAnalysisArgs* dargs =
          (const DeferPerformAnalysisArgs*)args;
      // Get this before doing anything
      const bool on_heap = dargs->analysis->on_heap;
      std::set<RtEvent> deferral_events, applied_events;
      dargs->analysis->analyze(
          dargs->set, *(dargs->mask), deferral_events, applied_events,
          RtEvent::NO_RT_EVENT, dargs->already_deferred);
      if (!deferral_events.empty())
        Runtime::trigger_event(
            dargs->done_event, Runtime::merge_events(deferral_events));
      else
        Runtime::trigger_event(dargs->done_event);
      if (!applied_events.empty())
        dargs->analysis->record_deferred_applied_events(applied_events);
      if (on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
      delete dargs->mask;
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformRemoteArgs::DeferPerformRemoteArgs(
        PhysicalAnalysis* ana)
      : LgTaskArgs<DeferPerformRemoteArgs>(ana->op->get_unique_op_id()),
        analysis(ana), done_event(Runtime::create_rt_user_event())
    //--------------------------------------------------------------------------
    {
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_remote(const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformRemoteArgs* dargs = (const DeferPerformRemoteArgs*)args;
      std::set<RtEvent> applied_events;
      // Get this before doing anything
      const bool on_heap = dargs->analysis->on_heap;
      const RtEvent done = dargs->analysis->perform_remote(
          RtEvent::NO_RT_EVENT, applied_events, true /*already deferred*/);
      Runtime::trigger_event(dargs->done_event, done);
      if (!applied_events.empty())
        dargs->analysis->record_deferred_applied_events(applied_events);
      if (on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformUpdateArgs::DeferPerformUpdateArgs(
        PhysicalAnalysis* ana)
      : LgTaskArgs<DeferPerformUpdateArgs>(ana->op->get_unique_op_id()),
        analysis(ana), done_event(Runtime::create_rt_user_event())
    //--------------------------------------------------------------------------
    {
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_update(const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformUpdateArgs* dargs = (const DeferPerformUpdateArgs*)args;
      std::set<RtEvent> applied_events;
      // Get this before doing anything
      const bool on_heap = dargs->analysis->on_heap;
      const RtEvent done = dargs->analysis->perform_updates(
          RtEvent::NO_RT_EVENT, applied_events, true /*already deferred*/);
      Runtime::trigger_event(dargs->done_event, done);
      if (!applied_events.empty())
        dargs->analysis->record_deferred_applied_events(applied_events);
      if (on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformRegistrationArgs::
        DeferPerformRegistrationArgs(
            PhysicalAnalysis* ana, const RegionUsage& use,
            const PhysicalTraceInfo& info, ApEvent pre, ApEvent term, bool symb)
      : LgTaskArgs<DeferPerformRegistrationArgs>(ana->op->get_unique_op_id()),
        analysis(ana), usage(use), trace_info(&info), precondition(pre),
        termination(term),
        instances_ready(Runtime::create_ap_user_event(&info)),
        done_event(Runtime::create_rt_user_event()), symbolic(symb)
    //--------------------------------------------------------------------------
    {
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_registration(
        const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformRegistrationArgs* dargs =
          (const DeferPerformRegistrationArgs*)args;
      ApEvent insts_ready;
      std::set<RtEvent> applied_events;
      const RtEvent done = dargs->analysis->perform_registration(
          RtEvent::NO_RT_EVENT, dargs->usage, applied_events,
          dargs->precondition, dargs->termination, insts_ready,
          dargs->symbolic);
      Runtime::trigger_event(
          dargs->instances_ready, insts_ready, *(dargs->trace_info),
          applied_events);
      Runtime::trigger_event(dargs->done_event, done);
      if (!applied_events.empty())
        dargs->analysis->record_deferred_applied_events(applied_events);
      if (dargs->analysis->on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformOutputArgs::DeferPerformOutputArgs(
        PhysicalAnalysis* ana, bool track, const PhysicalTraceInfo& info)
      : LgTaskArgs<DeferPerformOutputArgs>(ana->op->get_unique_op_id()),
        analysis(ana), trace_info(&info),
        effects_event(
            track ? Runtime::create_ap_user_event(&info) :
                    ApUserEvent::NO_AP_USER_EVENT)
    //--------------------------------------------------------------------------
    {
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_output(const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformOutputArgs* dargs = (const DeferPerformOutputArgs*)args;
      std::set<RtEvent> applied_events;
      const bool on_heap = dargs->analysis->on_heap;
      const ApEvent result = dargs->analysis->perform_output(
          RtEvent::NO_RT_EVENT, applied_events, true /*already deferred*/);
      if (dargs->effects_event.exists())
        Runtime::trigger_event(
            dargs->effects_event, result, *(dargs->trace_info), applied_events);
      if (!applied_events.empty())
        dargs->analysis->record_deferred_applied_events(applied_events);
      if (on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
    }

  }  // namespace Internal
}  // namespace Legion
