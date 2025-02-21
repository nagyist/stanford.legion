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

#include "legion/analysis/aggregator.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/analysis/physical.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/across.h"
#include "legion/nodes/expression.h"
#include "legion/views/fill.h"
#include "legion/views/individual.h"
#include "legion/views/collective.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Copy Fill Guard
    /////////////////////////////////////////////////////////////

#ifndef NON_AGGRESSIVE_AGGREGATORS
    //--------------------------------------------------------------------------
    CopyFillGuard::CopyFillGuard(RtUserEvent post, RtUserEvent applied)
      : guard_postcondition(post), effects_applied(applied),
        releasing_guards(false), read_only_guard(false)
    //--------------------------------------------------------------------------
    { }
#else
    //--------------------------------------------------------------------------
    CopyFillGuard::CopyFillGuard(RtUserEvent applied)
      : effects_applied(applied), releasing_guards(false),
        read_only_guard(false)
    //--------------------------------------------------------------------------
    { }
#endif

    //--------------------------------------------------------------------------
    CopyFillGuard::~CopyFillGuard(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(releasing_guards);  // should have done a release
      assert(guarded_sets.empty());
      assert(remote_release_events.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void CopyFillGuard::pack_guard(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(guard_lock);
      // If we're already releasing a guard then there is no point in sending it
      if (releasing_guards)
      {
        rez.serialize(RtUserEvent::NO_RT_USER_EVENT);
        return;
      }
#ifdef DEBUG_LEGION
      assert(!guarded_sets.empty());
      assert(effects_applied.exists());
#endif
      // We only ever pack the effects applied event here because once a
      // guard is on a remote node then the guard postcondition is no longer
      // useful since all remote copy fill operations will need to key off
      // the effects applied event to be correct
      rez.serialize(effects_applied);
      rez.serialize<bool>(read_only_guard);
      // Make an event for recording when all the remote events are applied
      RtUserEvent remote_release = Runtime::create_rt_user_event();
      rez.serialize(remote_release);
      remote_release_events.push_back(remote_release);
    }

    //--------------------------------------------------------------------------
    /*static*/ CopyFillGuard* CopyFillGuard::unpack_guard(
        Deserializer& derez, EquivalenceSet* set)
    //--------------------------------------------------------------------------
    {
      RtUserEvent effects_applied;
      derez.deserialize(effects_applied);
      if (!effects_applied.exists())
        return nullptr;
#ifndef NON_AGGRESSIVE_AGGREGATORS
      // Note we use the effects applied event here twice because all
      // copy-fill aggregators on this node will need to wait for the
      // full effects to be applied of any guards on a remote node
      CopyFillGuard* result =
          new CopyFillGuard(effects_applied, effects_applied);
#else
      CopyFillGuard* result = new CopyFillGuard(effects_applied);
#endif
      bool read_only_guard;
      derez.deserialize(read_only_guard);
#ifdef DEBUG_LEGION
      if (!result->record_guard_set(set, read_only_guard))
        std::abort();
#else
      result->record_guard_set(set, read_only_guard);
#endif
      RtUserEvent remote_release;
      derez.deserialize(remote_release);
      std::set<RtEvent> release_preconditions;
      result->release_guards(release_preconditions, true /*defer*/);
      if (!release_preconditions.empty())
        Runtime::trigger_event(
            remote_release, Runtime::merge_events(release_preconditions));
      else
        Runtime::trigger_event(remote_release);
      return result;
    }

    //--------------------------------------------------------------------------
    bool CopyFillGuard::record_guard_set(EquivalenceSet* set, bool read_only)
    //--------------------------------------------------------------------------
    {
      if (releasing_guards)
        return false;
      AutoLock g_lock(guard_lock);
      // Check again after getting the lock to avoid the race
      if (releasing_guards)
        return false;
#ifdef DEBUG_LEGION
      assert(guarded_sets.empty() || (read_only_guard == read_only));
#endif
      guarded_sets.insert(set);
      read_only_guard = read_only;
      return true;
    }

    //--------------------------------------------------------------------------
    bool CopyFillGuard::release_guards(
        std::set<RtEvent>& applied, bool force_deferral /*=false*/)
    //--------------------------------------------------------------------------
    {
      if (force_deferral || !effects_applied.has_triggered())
      {
        RtUserEvent released = Runtime::create_rt_user_event();
        // Meta-task will take responsibility for deletion
        CopyFillDeletion args(this, implicit_provenance, released);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, effects_applied);
        applied.insert(released);
        return false;
      }
      else
        release_guarded_sets(applied);
      return true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyFillGuard::handle_deletion(const void* args)
    //--------------------------------------------------------------------------
    {
      const CopyFillDeletion* dargs = (const CopyFillDeletion*)args;
      std::set<RtEvent> released_preconditions;
      dargs->guard->release_guarded_sets(released_preconditions);
      if (!released_preconditions.empty())
        Runtime::trigger_event(
            dargs->released, Runtime::merge_events(released_preconditions));
      else
        Runtime::trigger_event(dargs->released);
      delete dargs->guard;
    }

    //--------------------------------------------------------------------------
    void CopyFillGuard::release_guarded_sets(std::set<RtEvent>& released)
    //--------------------------------------------------------------------------
    {
      std::set<EquivalenceSet*> to_remove;
      {
        AutoLock g_lock(guard_lock);
#ifdef DEBUG_LEGION
        assert(!releasing_guards);
#endif
        releasing_guards = true;
        to_remove.swap(guarded_sets);
        if (!remote_release_events.empty())
        {
          released.insert(
              remote_release_events.begin(), remote_release_events.end());
          remote_release_events.clear();
        }
      }
      if (!to_remove.empty())
      {
        if (read_only_guard)
        {
          for (std::set<EquivalenceSet*>::const_iterator it = to_remove.begin();
               it != to_remove.end(); it++)
            (*it)->remove_read_only_guard(this);
        }
        else
        {
          for (std::set<EquivalenceSet*>::const_iterator it = to_remove.begin();
               it != to_remove.end(); it++)
            (*it)->remove_reduction_fill_guard(this);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Copy Fill Aggregator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(
        PhysicalAnalysis* a, CopyFillGuard* previous, bool t, PredEvent p)
#ifndef NON_AGGRESSIVE_AGGREGATORS
      : CopyFillGuard(
            Runtime::create_rt_user_event(), Runtime::create_rt_user_event()),
#else
      : CopyFillGuard(Runtime::create_rt_user_event()),
#endif
        local_space(runtime->address_space), analysis(a),
        collective_mapping(analysis->get_replicated_mapping()),
        src_index(analysis->index), dst_index(analysis->index),
#ifndef NON_AGGRESSIVE_AGGREGATORS
        guard_precondition(
            (previous == nullptr) ? RtEvent::NO_RT_EVENT :
                                    RtEvent(previous->guard_postcondition)),
#else
        guard_precondition(
            (previous == nullptr) ? RtEvent::NO_RT_EVENT :
                                    RtEvent(previous->effects_applied)),
#endif
        predicate_guard(p), track_events(t)
    //--------------------------------------------------------------------------
    {
      analysis->add_reference();
      // Need to transitively chain effects across aggregators since they
      // each need to summarize all the ones that came before
      if (previous != nullptr)
        effects.insert(previous->effects_applied);
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(
        PhysicalAnalysis* a, unsigned src_idx, unsigned dst_idx,
        CopyFillGuard* previous, bool t, PredEvent p,
        RtEvent alternative_precondition)
#ifndef NON_AGGRESSIVE_AGGREGATORS
      : CopyFillGuard(
            Runtime::create_rt_user_event(), Runtime::create_rt_user_event()),
#else
      : CopyFillGuard(Runtime::create_rt_user_event()),
#endif
        local_space(runtime->address_space), analysis(a),
        collective_mapping(analysis->get_replicated_mapping()),
        src_index(src_idx), dst_index(dst_idx),
#ifndef NON_AGGRESSIVE_AGGREGATORS
        guard_precondition(
            (previous == nullptr) ? alternative_precondition :
                                    RtEvent(previous->guard_postcondition)),
#else
        guard_precondition(
            (previous == nullptr) ? alternative_precondition :
                                    RtEvent(previous->effects_applied)),
#endif
        predicate_guard(p), track_events(t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((previous == nullptr) || !alternative_precondition.exists());
#endif
      analysis->add_reference();
      // Need to transitively chain effects across aggregators since they
      // each need to summarize all the ones that came before
      if (previous != nullptr)
        effects.insert(previous->effects_applied);
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::~CopyFillAggregator(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
#ifndef NON_AGGRESSIVE_AGGREGATORS
      assert(guard_postcondition.has_triggered());
#endif
      assert(effects_applied.has_triggered());
#endif
      if (analysis->remove_reference())
        delete analysis;
      // Remove references from any views that we have
      for (std::set<LogicalView*>::const_iterator it = all_views.begin();
           it != all_views.end(); it++)
        if ((*it)->remove_base_valid_ref(AGGREGATOR_REF))
          delete (*it);
      all_views.clear();
      // Delete all our copy updates
      for (LegionMap<InstanceView*, FieldMaskSet<Update> >::const_iterator mit =
               sources.begin();
           mit != sources.end(); mit++)
      {
        for (FieldMaskSet<Update>::const_iterator it = mit->second.begin();
             it != mit->second.end(); it++)
          delete it->first;
      }
      for (std::vector<LegionMap<InstanceView*, FieldMaskSet<Update> > >::
               const_iterator rit = reductions.begin();
           rit != reductions.end(); rit++)
      {
        for (LegionMap<InstanceView*, FieldMaskSet<Update> >::const_iterator
                 mit = rit->begin();
             mit != rit->end(); mit++)
        {
          for (FieldMaskSet<Update>::const_iterator it = mit->second.begin();
               it != mit->second.end(); it++)
            delete it->first;
        }
      }
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::Update::Update(
        IndexSpaceExpression* exp, const FieldMask& mask,
        CopyAcrossHelper* helper)
      : expr(exp), src_mask(mask), across_helper(helper)
    //--------------------------------------------------------------------------
    {
      expr->add_base_expression_reference(AGGREGATOR_REF);
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::Update::~Update(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_base_expression_reference(AGGREGATOR_REF))
        delete expr;
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::CopyUpdate::record_source_expressions(
        InstanceFieldExprs& src_exprs) const
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<IndexSpaceExpression>& exprs = src_exprs[source];
      FieldMaskSet<IndexSpaceExpression>::iterator finder = exprs.find(expr);
      if (finder == exprs.end())
        exprs.insert(expr, src_mask);
      else
        finder.merge(src_mask);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::CopyUpdate::sort_updates(
        std::map<InstanceView*, std::vector<CopyUpdate*> >& copies,
        std::vector<FillUpdate*>& fills)
    //--------------------------------------------------------------------------
    {
      copies[source].push_back(this);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::FillUpdate::record_source_expressions(
        InstanceFieldExprs& src_exprs) const
    //--------------------------------------------------------------------------
    {
      // Do nothing, we have no source expressions
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::FillUpdate::sort_updates(
        std::map<InstanceView*, std::vector<CopyUpdate*> >& copies,
        std::vector<FillUpdate*>& fills)
    //--------------------------------------------------------------------------
    {
      fills.push_back(this);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_update(
        InstanceView* dst_view, PhysicalManager* dst_man, LogicalView* src_view,
        const FieldMask& src_mask, IndexSpaceExpression* expr,
        const PhysicalTraceInfo& trace_info, EquivalenceSet* tracing_eq,
        ReductionOpID redop /*=0*/, CopyAcrossHelper* helper /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!src_mask);
      assert(!expr->is_empty());
#endif
      if (src_view->is_deferred_view())
      {
#ifdef DEBUG_LEGION
        assert(redop == 0);
#endif
        DeferredView* def_view = src_view->as_deferred_view();
        def_view->flatten(
            *this, dst_view, src_mask, expr, predicate_guard, trace_info,
            tracing_eq, helper);
      }
      else
      {
        InstanceView* inst_view = src_view->as_instance_view();
        PhysicalManager* src_man = nullptr;
        if (inst_view->is_collective_view())
        {
          if (dst_man != nullptr)
          {
            std::vector<InstanceView*> src_views(1, inst_view);
            const SelectSourcesResult& result =
                select_sources(dst_view, dst_man, src_views);
#ifdef DEBUG_LEGION
            assert(result.ranking.size() == 1);
#endif
            std::map<unsigned, PhysicalManager*>::const_iterator finder =
                result.points.find(0);
            if (finder != result.points.end())
              src_man = finder->second;
          }
        }
        else
          src_man = inst_view->as_individual_view()->get_manager();
        record_instance_update(
            dst_view, inst_view, src_man, src_mask, expr, tracing_eq, redop,
            helper);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_instance_update(
        InstanceView* dst_view, InstanceView* src_view,
        PhysicalManager* src_man, const FieldMask& src_mask,
        IndexSpaceExpression* expr, EquivalenceSet* tracing_eq,
        ReductionOpID redop, CopyAcrossHelper* helper)
    //--------------------------------------------------------------------------
    {
      update_fields |= src_mask;
      record_view(dst_view);
      record_view(src_view);
      CopyUpdate* update =
          new CopyUpdate(src_view, src_man, src_mask, expr, redop, helper);
      FieldMaskSet<Update>& updates = sources[dst_view];
      if (helper == nullptr)
        updates.insert(update, src_mask);
      else
        updates.insert(update, helper->convert_src_to_dst(src_mask));
      if (tracing_eq != nullptr)
        update_tracing_valid_views(
            tracing_eq, src_view, dst_view, src_mask, expr, redop);
    }

    //--------------------------------------------------------------------------
    const CopyFillAggregator::SelectSourcesResult&
        CopyFillAggregator::select_sources(
            InstanceView* dst_view, PhysicalManager* dst_man,
            const std::vector<InstanceView*>& src_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dst_view != nullptr);
      assert(dst_man != nullptr);
      assert(!src_views.empty());
#endif
      const std::pair<InstanceView*, PhysicalManager*> key(dst_view, dst_man);
      std::map<
          std::pair<InstanceView*, PhysicalManager*>,
          std::vector<SelectSourcesResult> >::iterator finder =
          mapper_queries.find(key);
      if (finder != mapper_queries.end())
      {
        for (std::vector<SelectSourcesResult>::const_iterator it =
                 finder->second.begin();
             it != finder->second.end(); it++)
          if (it->matches(src_views))
            return *it;
      }
      else
        finder =
            mapper_queries
                .insert(std::make_pair(key, std::vector<SelectSourcesResult>()))
                .first;
      // If we didn't find the query result we need to do it for ourself
      std::vector<unsigned> ranking;
      std::map<unsigned, PhysicalManager*> points;
      // Always use the source index for selecting sources
      analysis->op->select_sources(
          src_index, dst_man, src_views, ranking, points);
      // Check to make sure that the ranking has sound output
      unsigned count = 0;
      std::vector<bool> unique_indexes(src_views.size(), false);
      for (std::vector<unsigned>::iterator it = ranking.begin();
           it != ranking.end();
           /*nothing*/)
      {
        if (((*it) < unique_indexes.size()) && !unique_indexes[*it])
        {
          unique_indexes[*it] = true;
          count++;
          it++;
        }
        else  // remove duplicates and out of bound entries
          it = ranking.erase(it);
      }
      if (count < unique_indexes.size())
      {
        for (unsigned idx = 0; idx < unique_indexes.size(); idx++)
          if (!unique_indexes[idx])
            ranking.push_back(idx);
      }
      // Save the result for the future
      finder->second.emplace_back(SelectSourcesResult(
          std::vector<InstanceView*>(src_views) /*make a copy*/,
          std::move(ranking), std::move(points)));
      return finder->second.back();
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_updates(
        InstanceView* dst_view, PhysicalManager* dst_man,
        const FieldMaskSet<LogicalView>& src_views, const FieldMask& src_mask,
        IndexSpaceExpression* expr, const PhysicalTraceInfo& trace_info,
        EquivalenceSet* tracing_eq, ReductionOpID redop /*=0*/,
        CopyAcrossHelper* helper /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!src_mask);
      assert(!src_views.empty());
      assert(!expr->is_empty());
#endif
      if (src_views.size() == 1)
      {
        LogicalView* src_view = src_views.begin()->first;
        const FieldMask record_mask = src_views.get_valid_mask() & src_mask;
        record_update(
            dst_view, dst_man, src_view, record_mask, expr, trace_info,
            tracing_eq, redop, helper);
      }
      else
      {
        // We have multiple views, so let's sort them
        local::list<FieldSet<LogicalView*> > view_sets;
        src_views.compute_field_sets(src_mask, view_sets);
        for (local::list<FieldSet<LogicalView*> >::const_iterator vit =
                 view_sets.begin();
             vit != view_sets.end(); vit++)
        {
          if (vit->elements.empty())
            continue;
          if (vit->elements.size() == 1)
          {
            // Easy case, just one view so do it
            LogicalView* src_view = *(vit->elements.begin());
            const FieldMask& record_mask = vit->set_mask;
            record_update(
                dst_view, dst_man, src_view, record_mask, expr, trace_info,
                tracing_eq, redop, helper);
          }
          else
          {
            // Sort the views, prefer deferred  then instances
            DeferredView* deferred = nullptr;
            std::vector<InstanceView*> instances;
            for (std::set<LogicalView*>::const_iterator it =
                     vit->elements.begin();
                 it != vit->elements.end(); it++)
            {
              if (!(*it)->is_instance_view())
              {
                deferred = (*it)->as_deferred_view();
                // Break out since we found what we're looking for
                break;
              }
              else
                instances.push_back((*it)->as_instance_view());
            }
            if (deferred != nullptr)
              deferred->flatten(
                  *this, dst_view, vit->set_mask, expr, predicate_guard,
                  trace_info, tracing_eq, helper);
            else if (!instances.empty())
            {
              if (instances.size() == 1)
              {
                // Easy, just one instance to use and no collective instances
                InstanceView* src_view = instances.back();
                record_update(
                    dst_view, dst_man, src_view, vit->set_mask, expr,
                    trace_info, tracing_eq, redop, helper);
              }
              else
              {
                // Hard, multiple potential sources,
                // ask the mapper which one to use
                const SelectSourcesResult& result =
                    select_sources(dst_view, dst_man, instances);
#ifdef DEBUG_LEGION
                assert(result.ranking.size() == instances.size());
#endif
                const unsigned first = result.ranking.front();
                InstanceView* src_view = instances[first];
                PhysicalManager* src_man = nullptr;
                // Find the source point if it is a collective view
                if (src_view->is_collective_view())
                {
                  std::map<unsigned, PhysicalManager*>::const_iterator
                      point_finder = result.points.find(first);
                  if (point_finder != result.points.end())
                    src_man = point_finder->second;
                }
                else
                  src_man = src_view->as_individual_view()->get_manager();
                record_instance_update(
                    dst_view, src_view, src_man, vit->set_mask, expr,
                    tracing_eq, redop, helper);
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_partial_updates(
        InstanceView* dst_view, PhysicalManager* dst_man,
        const LegionMap<LogicalView*, FieldMaskSet<IndexSpaceExpression> >&
            src_views,
        const FieldMask& src_mask, IndexSpaceExpression* expr,
        const PhysicalTraceInfo& trace_info, EquivalenceSet* tracing_eq,
        ReductionOpID redop, CopyAcrossHelper* across_helper)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!src_mask);
      assert(!src_views.empty());
      assert(!expr->is_empty());
#endif
      update_fields |= src_mask;
      record_view(dst_view);
      std::vector<InstanceView*> instances;
      FieldMaskSet<IndexSpaceExpression> remainders;
      remainders.insert(expr, src_mask);
      // Issue deferred immediately, otherwise record instances so that
      // we can ask the mapper what order it wants us to issue copies from
      for (LegionMap<LogicalView*, FieldMaskSet<IndexSpaceExpression> >::
               const_iterator vit = src_views.begin();
           vit != src_views.end(); vit++)
      {
        FieldMask view_overlap =
            vit->second.get_valid_mask() & remainders.get_valid_mask();
        ;
        if (!view_overlap)
          continue;
        if (vit->first->is_deferred_view())
        {
          DeferredView* deferred = vit->first->as_deferred_view();
          // Skip any deferred if we're doing a reduction, we only care
          // about valid instances here
          if (redop > 0)
            continue;
          // Join in the fields to see what overlaps
          LegionMap<
              std::pair<IndexSpaceExpression*, IndexSpaceExpression*>,
              FieldMask>
              deferred_exprs;
          unique_join_on_field_mask_sets(
              remainders, vit->second, deferred_exprs);
          bool need_tighten = false;
          for (LegionMap<
                   std::pair<IndexSpaceExpression*, IndexSpaceExpression*>,
                   FieldMask>::const_iterator it = deferred_exprs.begin();
               it != deferred_exprs.end(); it++)
          {
            IndexSpaceExpression* overlap = runtime->intersect_index_spaces(
                it->first.first, it->first.second);
            const size_t overlap_size = overlap->get_volume();
            if (overlap_size == 0)
              continue;
            FieldMaskSet<IndexSpaceExpression>::iterator finder =
                remainders.find(it->first.first);
#ifdef DEBUG_LEGION
            assert(finder != remainders.end());
#endif
            finder.filter(it->second);
            if (!finder->second)
              remainders.erase(finder);
            if (overlap_size < it->first.first->get_volume())
            {
              if (overlap_size == it->first.second->get_volume())
                deferred->flatten(
                    *this, dst_view, it->second, it->first.second,
                    predicate_guard, trace_info, tracing_eq, across_helper);
              else
                deferred->flatten(
                    *this, dst_view, it->second, overlap, predicate_guard,
                    trace_info, tracing_eq, across_helper);
              // Compute the difference
              IndexSpaceExpression* diff_expr =
                  runtime->subtract_index_spaces(it->first.first, overlap);
              remainders.insert(diff_expr, it->second);
            }
            else  // completely covers remainder expression
            {
              deferred->flatten(
                  *this, dst_view, it->second, it->first.first, predicate_guard,
                  trace_info, tracing_eq, across_helper);
              if (remainders.empty())
                return;
              need_tighten = true;
            }
          }
          if (need_tighten)
            remainders.tighten_valid_mask();
        }
        else
          instances.push_back(vit->first->as_instance_view());
      }
      // If we get here, next try to sort the instances into whatever order
      // the mapper wants us to try to issue copies from them
      if (!instances.empty())
      {
        std::vector<unsigned> ranking;
        std::map<unsigned, PhysicalManager*> points;
        // Need to ask the mapper which instances it prefers if there are
        // multiple choices or we have a collective instance to pick from
        if ((instances.size() > 1) || instances.back()->is_collective_view())
        {
          const SelectSourcesResult& result =
              select_sources(dst_view, dst_man, instances);
          ranking = result.ranking;
          points = result.points;
        }
        else
          ranking.push_back(0);
        for (unsigned idx = 0; idx < ranking.size(); idx++)
        {
          InstanceView* src_view = instances[ranking[idx]];
          PhysicalManager* src_man = nullptr;
          // Find the source key if this is a collective instance
          if (src_view->is_collective_view())
          {
            std::map<unsigned, PhysicalManager*>::const_iterator point_finder =
                points.find(ranking[idx]);
            if (point_finder != points.end())
              src_man = point_finder->second;
          }
          else
            src_man = src_view->as_individual_view()->get_manager();
          LegionMap<LogicalView*, FieldMaskSet<IndexSpaceExpression> >::
              const_iterator finder = src_views.find(src_view);
#ifdef DEBUG_LEGION
          assert(finder != src_views.end());
#endif
          LegionMap<
              std::pair<IndexSpaceExpression*, IndexSpaceExpression*>,
              FieldMask>
              src_expressions;
          unique_join_on_field_mask_sets(
              remainders, finder->second, src_expressions);
          bool need_tighten = false;
          for (LegionMap<
                   std::pair<IndexSpaceExpression*, IndexSpaceExpression*>,
                   FieldMask>::const_iterator it = src_expressions.begin();
               it != src_expressions.end(); it++)
          {
            IndexSpaceExpression* overlap = runtime->intersect_index_spaces(
                it->first.first, it->first.second);
            const size_t overlap_size = overlap->get_volume();
            if (overlap_size == 0)
              continue;
            FieldMaskSet<IndexSpaceExpression>::iterator finder =
                remainders.find(it->first.first);
#ifdef DEBUG_LEGION
            assert(finder != remainders.end());
#endif
            finder.filter(it->second);
            if (!finder->second)
              remainders.erase(finder);
            if (overlap_size < it->first.first->get_volume())
            {
              if (overlap_size == it->first.second->get_volume())
                record_instance_update(
                    dst_view, src_view, src_man, it->second, it->first.second,
                    tracing_eq, redop, across_helper);
              else
                record_instance_update(
                    dst_view, src_view, src_man, it->second, overlap,
                    tracing_eq, redop, across_helper);
              // Compute the difference
              IndexSpaceExpression* diff_expr =
                  runtime->subtract_index_spaces(it->first.first, overlap);
              remainders.insert(diff_expr, it->second);
            }
            else  // completely covers remainder expression
            {
              record_instance_update(
                  dst_view, src_view, src_man, it->second, it->first.first,
                  tracing_eq, redop, across_helper);
              if (remainders.empty())
                return;
              need_tighten = true;
            }
          }
          if (need_tighten)
            remainders.tighten_valid_mask();
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_fill(
        InstanceView* dst_view, FillView* src_view, const FieldMask& fill_mask,
        IndexSpaceExpression* expr, const PredEvent fill_guard,
        EquivalenceSet* tracing_eq, CopyAcrossHelper* helper /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!fill_mask);
      assert(!expr->is_empty());
#endif
      update_fields |= fill_mask;
      record_view(src_view);
      record_view(dst_view);
      FillUpdate* update =
          new FillUpdate(src_view, fill_mask, expr, fill_guard, helper);
      if (helper == nullptr)
        sources[dst_view].insert(update, fill_mask);
      else
        sources[dst_view].insert(update, helper->convert_src_to_dst(fill_mask));
      if (tracing_eq != nullptr)
        tracing_eq->update_tracing_fill_views(
            src_view, dst_view, expr, fill_mask, (src_index != dst_index));
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_reductions(
        InstanceView* dst_view, PhysicalManager* dst_man,
        const std::list<std::pair<InstanceView*, IndexSpaceExpression*> >&
            src_views,
        const unsigned src_fidx, const unsigned dst_fidx,
        EquivalenceSet* tracing_eq, CopyAcrossHelper* across_helper)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!src_views.empty());
#endif
      update_fields.set_bit(src_fidx);
      record_view(dst_view);
      const std::pair<InstanceView*, unsigned> dst_key(dst_view, dst_fidx);
      std::vector<ReductionOpID>& redop_epochs = reduction_epochs[dst_key];
      FieldMask src_mask, dst_mask;
      src_mask.set_bit(src_fidx);
      dst_mask.set_bit(dst_fidx);
      // Always start scanning from the first redop index
      unsigned redop_index = 0;
      for (std::list<std::pair<InstanceView*, IndexSpaceExpression*> >::
               const_iterator it = src_views.begin();
           it != src_views.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(!it->second->is_empty());
#endif
        record_view(it->first);
        const ReductionOpID redop = it->first->get_redop();
#ifdef DEBUG_LEGION
        assert(redop > 0);
#endif
        PhysicalManager* src_man = nullptr;
        if (it->first->is_collective_view())
        {
          if (dst_man != nullptr)
          {
            std::vector<InstanceView*> src_views(1, it->first);
            const SelectSourcesResult& result =
                select_sources(dst_view, dst_man, src_views);
#ifdef DEBUG_LEGION
            assert(result.ranking.size() == 1);
#endif
            std::map<unsigned, PhysicalManager*>::const_iterator finder =
                result.points.find(0);
            if (finder != result.points.end())
              src_man = finder->second;
          }
        }
        else
          src_man = it->first->as_individual_view()->get_manager();
        CopyUpdate* update = new CopyUpdate(
            it->first, src_man, src_mask, it->second, redop, across_helper);
        // Ignore shadows when tracing, we only care about the normal
        // preconditions and postconditions for the copies
        if (tracing_eq != nullptr)
          update_tracing_valid_views(
              tracing_eq, it->first, dst_view, src_mask, it->second, redop);
        // Scan along looking for a reduction op epoch that matches
        while ((redop_index < redop_epochs.size()) &&
               (redop_epochs[redop_index] != redop))
          redop_index++;
        if (redop_index == redop_epochs.size())
        {
#ifdef DEBUG_LEGION
          assert(redop_index <= reductions.size());
#endif
          // Start a new redop epoch if necessary
          redop_epochs.push_back(redop);
          if (reductions.size() == redop_index)
            resize_reductions(redop_index + 1);
        }
        reductions[redop_index][dst_view].insert(update, dst_mask);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::resize_reductions(size_t new_size)
    //--------------------------------------------------------------------------
    {
      std::vector<LegionMap<InstanceView*, FieldMaskSet<Update> > >
          new_reductions(new_size);
      for (unsigned idx = 0; idx < reductions.size(); idx++)
        new_reductions[idx].swap(reductions[idx]);
      reductions.swap(new_reductions);
    }

    //--------------------------------------------------------------------------
    ApEvent CopyFillAggregator::issue_updates(
        const PhysicalTraceInfo& trace_info, ApEvent precondition,
        const bool restricted_output, const bool manage_dst_events,
        std::map<InstanceView*, std::vector<ApEvent> >* dst_events, int stage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!sources.empty() || !reductions.empty());
#endif
      if (guard_precondition.exists() && !guard_precondition.has_triggered())
      {
        if (track_events && !summary_event.exists())
          summary_event = Runtime::create_ap_user_event(&trace_info);
        CopyFillAggregation args(
            this, trace_info, precondition, manage_dst_events,
            restricted_output, analysis->op->get_unique_op_id(), stage,
            dst_events);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_DEFERRED_PRIORITY, guard_precondition);
        return summary_event;
      }
#ifdef DEBUG_LEGION
      assert(
          !guard_precondition.exists() || guard_precondition.has_triggered());
#endif
#ifndef NON_AGGRESSIVE_AGGREGATORS
      std::set<RtEvent> recorded_events;
#endif
      // This is really subtle so pay attention:
      // Copy across aggregators issue copies and fills to destination
      // instances. In some cases those destination instances are filled
      // and copied to multiple times across the initial sources and then
      // reductions of this copy fill aggregator. We need a way to make
      // sure that all the updates to the views for those destination
      // instances for the copy users are recorded in the right order.
      // To facilitate that we have a fast path and slow path. In the
      // fast path, if all the updates are coming from this node
      // (in the case with no collective analysis) or from all the nodes
      // in the collective analysis then we can issue all those copies
      // back-to-back without interruption because we know that the
      // messages being sent are all pipelined and run through ordered
      // virtual channels. However, if any copies need to be issued
      // to a destination that does not share the collective mapping
      // then we don't have that pipelining guarantee so we need to
      // block the issuing of the copies between those stages. Whether
      // we can do this pipelining or not after each stage is tracked
      // by the 'pipeline' variable. This method will defer itself and
      // restart when this pipelining is not possible.
      bool pipeline = true;
      // Perform updates from any sources first
      if (stage < 0)
      {
#ifdef DEBUG_LEGION
        assert(stage == -1);
#endif
        if (!sources.empty())
        {
          pipeline = perform_updates(
              sources, trace_info, precondition,
#ifdef NON_AGGRESSIVE_AGGREGATORS
              effects,
#else
              recorded_events,
#endif
              stage, manage_dst_events, restricted_output, dst_events);
        }
        stage = 0;
      }
      // Then apply any reductions that we might have
      if (!reductions.empty())
      {
        // Skip any passes that we might have already done
        for (unsigned idx = stage; idx < reductions.size(); idx++)
        {
          if (!pipeline)
          {
#ifdef NON_AGGRESSIVE_AGGREGATORS
            const RtEvent stage_pre = Runtime::merge_events(effects);
            effects.clear();
#else
            const RtEvent stage_pre = Runtime::merge_events(recorded_events);
            recorded_events.clear();
#endif
            if (stage_pre.exists() && !stage_pre.has_triggered())
            {
              // If it's not safe to pipeline the launching of these copies
              // with the copies from the previous stage then we need to
              // defer launching those copies until the previous ones have
              // run and registered all of their users
              CopyFillAggregation args(
                  this, trace_info, precondition, manage_dst_events,
                  restricted_output, analysis->op->get_unique_op_id(), idx,
                  dst_events);
              runtime->issue_runtime_meta_task(
                  args, LG_THROUGHPUT_DEFERRED_PRIORITY, stage_pre);
              return summary_event;
            }
          }
          perform_updates(
              reductions[idx], trace_info, precondition,
#ifdef NON_AGGRESSIVE_AGGREGATORS
              effects,
#else
              recorded_events,
#endif
              idx /*redop index*/, manage_dst_events, restricted_output,
              dst_events);
        }
      }
      // Make sure we do this before we trigger the effects_applied
      // event as it could result in the deletion of this object
      ApEvent summary;
      if (track_events)
      {
        summary = Runtime::merge_events(&trace_info, events);
        if (summary_event.exists())
        {
          Runtime::trigger_event(summary_event, summary, trace_info, effects);
          // Pull this onto the stack in case the object is deleted
          summary = summary_event;
        }
      }
#ifndef NON_AGGRESSIVE_AGGREGATORS
      if (!recorded_events.empty())
        Runtime::trigger_event(
            guard_postcondition, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(guard_postcondition);
      // Make sure the guard postcondition is chained on the deletion
      if (!effects.empty())
      {
        effects.insert(guard_postcondition);
        Runtime::trigger_event(effects_applied, Runtime::merge_events(effects));
      }
      else
        Runtime::trigger_event(effects_applied, guard_postcondition);
#else
      // We can also trigger our effects event once the effects are applied
      if (!effects.empty())
        Runtime::trigger_event(effects_applied, Runtime::merge_events(effects));
      else
        Runtime::trigger_event(effects_applied);
#endif
      return summary;
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_view(LogicalView* new_view)
    //--------------------------------------------------------------------------
    {
      std::pair<std::set<LogicalView*>::iterator, bool> result =
          all_views.insert(new_view);
      if (result.second)
        new_view->add_base_valid_ref(AGGREGATOR_REF);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::update_tracing_valid_views(
        EquivalenceSet* tracing_eq, InstanceView* src, InstanceView* dst,
        const FieldMask& mask, IndexSpaceExpression* expr,
        ReductionOpID redop) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing_eq != nullptr);
#endif
      if (redop > 0)
        tracing_eq->update_tracing_reduction_views(
            src, dst, expr, mask, (src_index != dst_index));
      else
        tracing_eq->update_tracing_copy_views(
            src, dst, expr, mask, (src_index != dst_index));
    }

    //--------------------------------------------------------------------------
    bool CopyFillAggregator::perform_updates(
        const LegionMap<InstanceView*, FieldMaskSet<Update> >& updates,
        const PhysicalTraceInfo& trace_info, const ApEvent precondition,
        std::set<RtEvent>& recorded_events, const int redop_index,
        const bool manage_dst_events, const bool restricted_output,
        std::map<InstanceView*, std::vector<ApEvent> >* dst_events)
    //--------------------------------------------------------------------------
    {
      bool pipelined = true;
      std::vector<ApEvent>* target_events = nullptr;
      for (LegionMap<InstanceView*, FieldMaskSet<Update> >::const_iterator uit =
               updates.begin();
           uit != updates.end(); uit++)
      {
        ApEvent dst_precondition = precondition;
        // In the case where we're not managing destination events
        // then we need to incorporate any event postconditions from
        // previous passes as part of the preconditions for this pass
        if (!manage_dst_events)
        {
#ifdef DEBUG_LEGION
          assert(dst_events != nullptr);
#endif
          // This only happens in the case of across copies
          std::map<InstanceView*, std::vector<ApEvent> >::iterator finder =
              dst_events->find(uit->first);
#ifdef DEBUG_LEGION
          assert(finder != dst_events->end());
#endif
          if (!finder->second.empty())
          {
            // Update our precondition to incude the copies from
            // any previous passes that we performed
            finder->second.push_back(precondition);
            dst_precondition =
                Runtime::merge_events(&trace_info, finder->second);
            // Clear this for the next iteration
            // It's not obvious why this safe, but it is
            // We are guaranteed to issue at least one fill/copy that
            // will depend on this and therefore either test that it
            // has triggered or record itself back in the set of events
            // which gives us a transitive precondition
            finder->second.clear();
          }
          target_events = &finder->second;
        }
        // Group by fields first
        local::list<FieldSet<Update*> > field_groups;
        uit->second.compute_field_sets(FieldMask(), field_groups);
        for (local::list<FieldSet<Update*> >::const_iterator fit =
                 field_groups.begin();
             fit != field_groups.end(); fit++)
        {
          const FieldMask& dst_mask = fit->set_mask;
          // Now that we have the src mask for these operations group
          // them into fills and copies
          std::vector<FillUpdate*> fills;
          std::map<InstanceView* /*src*/, std::vector<CopyUpdate*> > copies;
          for (std::set<Update*>::const_iterator it = fit->elements.begin();
               it != fit->elements.end(); it++)
            (*it)->sort_updates(copies, fills);
          // Issue the copies and fills
          if (!fills.empty())
            issue_fills(
                uit->first, fills, recorded_events, dst_precondition, dst_mask,
                trace_info, manage_dst_events, restricted_output,
                target_events);
          if (!copies.empty())
            issue_copies(
                uit->first, copies, recorded_events, dst_precondition, dst_mask,
                trace_info, manage_dst_events, restricted_output,
                target_events);
        }
        // Check whether later stages can be pipelined with respect to this
        // current stage. The requirement for pipelining is that the target
        // view have the same collective
        if (!pipelined ||
            (collective_mapping == uit->first->collective_mapping))
          continue;
        // If the collective mapping of the target view does not match the
        // collective mapping of the analysis then we cannot pipeline
        // the view analysis for issuing of copies
        if ((collective_mapping == nullptr) ||
            (uit->first->collective_mapping == nullptr) ||
            (*collective_mapping != *(uit->first->collective_mapping)))
          pipelined = false;
      }
      return pipelined;
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_fills(
        InstanceView* target, const std::vector<FillUpdate*>& fills,
        std::set<RtEvent>& recorded_events, const ApEvent precondition,
        const FieldMask& fill_mask, const PhysicalTraceInfo& trace_info,
        const bool manage_dst_events, const bool restricted_output,
        std::vector<ApEvent>* dst_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fills.empty());
      assert(!!fill_mask);
      // Should only have across helper on across copies
      assert((fills[0]->across_helper == nullptr) || !manage_dst_events);
      assert((dst_events == nullptr) || track_events);
#endif
      const IndexSpaceID match_space = analysis->get_collective_match_space();
      if (fills.size() == 1)
      {
        FillUpdate* update = fills[0];
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        // Should cover all the fields
        if (fills[0]->across_helper != nullptr)
        {
          const FieldMask src_mask =
              fills[0]->across_helper->convert_dst_to_src(fill_mask);
          assert(!(src_mask - update->src_mask));
        }
        else
        {
          assert(!(fill_mask - update->src_mask));
        }
#endif
#endif
        IndexSpaceExpression* fill_expr = update->expr;
        FillView* fill_view = update->source;
        const ApEvent result = target->fill_from(
            fill_view, precondition, update->fill_guard, fill_expr,
            analysis->op, dst_index, match_space, fill_mask, trace_info,
            recorded_events, effects, fills[0]->across_helper,
            manage_dst_events, restricted_output, track_events);
        if (result.exists())
        {
          if (track_events)
            events.push_back(result);
          if (dst_events != nullptr)
            dst_events->push_back(result);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        FieldMask src_mask;
        if (fills[0]->across_helper != nullptr)
          src_mask = fills[0]->across_helper->convert_dst_to_src(fill_mask);
        else
          src_mask = fill_mask;
        // These should all have had the same across helper
        for (unsigned idx = 1; idx < fills.size(); idx++)
          assert(fills[idx]->across_helper == fills[0]->across_helper);
#endif
#endif
        // Fills can have different predicates because of nested predicated
        // fills so we can only merge across fills with the same predicates
        std::map<
            std::pair<FillView*, PredEvent>, std::set<IndexSpaceExpression*> >
            exprs;
        for (std::vector<FillUpdate*>::const_iterator it = fills.begin();
             it != fills.end(); it++)
        {
#ifdef DEBUG_LEGION
          // Should cover all the fields
          assert(!(src_mask - (*it)->src_mask));
          // Should also have the same across helper as the first one
          assert(fills[0]->across_helper == (*it)->across_helper);
#endif
          std::pair<FillView*, PredEvent> key((*it)->source, (*it)->fill_guard);
          exprs[key].insert((*it)->expr);
        }
        for (std::map<
                 std::pair<FillView*, PredEvent>,
                 std::set<IndexSpaceExpression*> >::const_iterator it =
                 exprs.begin();
             it != exprs.end(); it++)
        {
          IndexSpaceExpression* fill_expr =
              (it->second.size() == 1) ?
                  *(it->second.begin()) :
                  runtime->union_index_spaces(it->second);
          // See if we have any work to do for tracing
          const ApEvent result = target->fill_from(
              it->first.first, precondition, it->first.second, fill_expr,
              analysis->op, dst_index, match_space, fill_mask, trace_info,
              recorded_events, effects, fills[0]->across_helper,
              manage_dst_events, restricted_output, track_events);
          if (result.exists())
          {
            if (track_events)
              events.push_back(result);
            if (dst_events != nullptr)
              dst_events->push_back(result);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_copies(
        InstanceView* target,
        std::map<InstanceView*, std::vector<CopyUpdate*> >& copies,
        std::set<RtEvent>& recorded_events, const ApEvent precondition,
        const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
        const bool manage_dst_events, const bool restricted_output,
        std::vector<ApEvent>* dst_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!copies.empty());
      assert(!!copy_mask);
      assert((src_index == dst_index) || !manage_dst_events);
      assert((dst_events == nullptr) || track_events);
#endif
      const IndexSpaceID match_space = analysis->get_collective_match_space();
      // We'll also look for an interesting optimization case here
      // that was identified by cuNumeric tensor contractions, in
      // some cases we'll have a group of individual views in the
      // source that are all going to be copied into the same
      // collective destination view. In this case, what we can
      // copy all the sources to one instance in the collective
      // view and then we can fuse the broadcast result.
      if (target->is_collective_view())
      {
        std::map<IndividualView*, std::vector<CopyUpdate*> >
            fused_gather_copies;
        for (std::map<InstanceView*, std::vector<CopyUpdate*> >::iterator cit =
                 copies.begin();
             cit != copies.end();
             /*nothing*/)
        {
          if (!cit->first->is_individual_view())
          {
            cit++;
            continue;
          }
          IndividualView* source = cit->first->as_individual_view();
          for (std::vector<CopyUpdate*>::iterator it = cit->second.begin();
               it != cit->second.end();
               /*nothing*/)
          {
#ifdef DEBUG_LEGION
            assert((*it)->across_helper == nullptr);
            assert(!(copy_mask - (*it)->src_mask));
#endif
            if ((*it)->redop == 0)
            {
              fused_gather_copies[source].push_back(*it);
              it = cit->second.erase(it);
            }
            else
              it++;
          }
          if (cit->second.empty())
          {
            std::map<InstanceView*, std::vector<CopyUpdate*> >::iterator
                to_delete = cit++;
            copies.erase(to_delete);
          }
          else
            cit++;
        }
        if (fused_gather_copies.size() > 1)
        {
#ifdef DEBUG_LEGION
          assert(manage_dst_events);
#endif
          CollectiveView* target_collective = target->as_collective_view();
          std::map<IndividualView*, IndexSpaceExpression*> view_exprs;
          for (std::map<IndividualView*, std::vector<CopyUpdate*> >::iterator
                   cit = fused_gather_copies.begin();
               cit != fused_gather_copies.end(); cit++)
          {
            if (cit->second.size() > 1)
            {
              std::set<IndexSpaceExpression*> union_exprs;
              for (std::vector<CopyUpdate*>::const_iterator it =
                       cit->second.begin();
                   it != cit->second.end(); it++)
                union_exprs.insert((*it)->expr);
              view_exprs[cit->first] = runtime->union_index_spaces(union_exprs);
            }
            else
              view_exprs[cit->first] = (*(cit->second.begin()))->expr;
          }
          const ApEvent result = target_collective->collective_fuse_gather(
              view_exprs, precondition, predicate_guard, analysis->op,
              dst_index, match_space, copy_mask, trace_info, recorded_events,
              effects, restricted_output, track_events);
          if (result.exists())
          {
            if (track_events)
              events.push_back(result);
            if (dst_events != nullptr)
              dst_events->push_back(result);
          }
        }
        else if (!fused_gather_copies.empty())
        {
          // Only one view so we can put them back onto the original set
          // of copies since we're just going to perform a normal fusion
          std::map<IndividualView*, std::vector<CopyUpdate*> >::iterator next =
              fused_gather_copies.begin();
          std::vector<CopyUpdate*>& original = copies[next->first];
          if (original.empty())
            original.swap(next->second);
          else
            original.insert(
                original.end(), next->second.begin(), next->second.end());
        }
      }
      for (std::map<InstanceView*, std::vector<CopyUpdate*> >::const_iterator
               cit = copies.begin();
           cit != copies.end(); cit++)
      {
#ifdef DEBUG_LEGION
        assert(!cit->second.empty());
        // Should only have across helpers for across copies
        assert(
            (cit->second[0]->across_helper == nullptr) || !manage_dst_events);
#endif
        if (cit->second.size() == 1)
        {
          // Easy case of a single update copy
          CopyUpdate* update = cit->second[0];
#ifdef DEBUG_LEGION
#ifndef NDEBUG
          if (cit->second[0]->across_helper != nullptr)
          {
            const FieldMask src_mask =
                cit->second[0]->across_helper->convert_dst_to_src(copy_mask);
            assert(!(src_mask - update->src_mask));
          }
          else
          {
            // Should cover all the fields
            assert(!(copy_mask - update->src_mask));
          }
#endif
#endif
          InstanceView* source = update->source;
          IndexSpaceExpression* copy_expr = update->expr;
          const ApEvent result = target->copy_from(
              source, precondition, predicate_guard, update->redop, copy_expr,
              analysis->op, manage_dst_events ? dst_index : src_index,
              match_space, copy_mask, update->src_man, trace_info,
              recorded_events, effects, cit->second[0]->across_helper,
              manage_dst_events, restricted_output, track_events);
          if (result.exists())
          {
            if (track_events)
              events.push_back(result);
            if (dst_events != nullptr)
              dst_events->push_back(result);
          }
        }
        else
        {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
          FieldMask src_mask;
          if (cit->second[0]->across_helper != nullptr)
            src_mask =
                cit->second[0]->across_helper->convert_dst_to_src(copy_mask);
          else
            src_mask = copy_mask;
#endif
#endif
          // Have to group by source instances in order to merge together
          // different index space expressions for the same copy
          // For collective instances we also need to group by the source
          // point of the collective instance to use
          std::map<
              std::pair<InstanceView*, PhysicalManager*>,
              std::set<IndexSpaceExpression*> >
              fused_exprs;
          const ReductionOpID redop = cit->second[0]->redop;
          std::map<IndividualView*, IndexSpaceExpression*> collective_gather;
          for (std::vector<CopyUpdate*>::const_iterator it =
                   cit->second.begin();
               it != cit->second.end(); it++)
          {
#ifdef DEBUG_LEGION
            // Should cover all the fields
            assert(!(src_mask - (*it)->src_mask));
            // Should have the same redop
            assert(redop == (*it)->redop);
            // Should also have the same across helper as the first one
            assert(cit->second[0]->across_helper == (*it)->across_helper);
#endif
            const std::pair<InstanceView*, PhysicalManager*> key(
                (*it)->source, (*it)->src_man);
            fused_exprs[key].insert((*it)->expr);
          }
          for (std::map<
                   std::pair<InstanceView*, PhysicalManager*>,
                   std::set<IndexSpaceExpression*> >::const_iterator it =
                   fused_exprs.begin();
               it != fused_exprs.end(); it++)
          {
            IndexSpaceExpression* copy_expr =
                (it->second.size() == 1) ?
                    *(it->second.begin()) :
                    runtime->union_index_spaces(it->second);
            const ApEvent result = target->copy_from(
                it->first.first, precondition, predicate_guard, redop,
                copy_expr, analysis->op,
                manage_dst_events ? dst_index : src_index, match_space,
                copy_mask, it->first.second, trace_info, recorded_events,
                effects, cit->second[0]->across_helper, manage_dst_events,
                restricted_output, track_events);
            if (result.exists())
            {
              if (track_events)
                events.push_back(result);
              if (dst_events != nullptr)
                dst_events->push_back(result);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyFillAggregator::handle_aggregation(const void* args)
    //--------------------------------------------------------------------------
    {
      const CopyFillAggregation* cfargs = (const CopyFillAggregation*)args;
      cfargs->aggregator->issue_updates(
          *cfargs, cfargs->pre, cfargs->restricted_output,
          cfargs->manage_dst_events, cfargs->dst_events, cfargs->stage);
      cfargs->remove_recorder_reference();
      if (cfargs->dst_events != nullptr)
        delete cfargs->dst_events;
    }

  }  // namespace Internal
}  // namespace Legion
