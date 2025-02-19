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

#include "legion/analysis/logical.h"
#include "legion/analysis/projection.h"
#include "legion/contexts/inner.h"
#include "legion/api/functors_impl.h"
#include "legion/nodes/region.h"
#include "legion/operations/close.h"
#include "legion/operations/refinement.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // LogicalUser
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(
        Operation* o, unsigned id, const RegionUsage& u, ProjectionSummary* p,
        unsigned internal)
      : Collectable(), usage(u), op(o), ctx_index(op->get_context_index()),
        uid(o->get_unique_op_id()), internal_idx(internal), idx(id),
        gen(o->get_generation()), shard_proj(p),
        pointwise_analyzable(
            op->is_pointwise_analyzable() && (shard_proj != nullptr) &&
            ((shard_proj->projection->projection_id == 0) ||
             shard_proj->projection->is_invertible))
    //--------------------------------------------------------------------------
    {
      if (op != nullptr)
        op->add_mapping_reference(gen);
      if (shard_proj != nullptr)
        shard_proj->add_reference();
    }

    //--------------------------------------------------------------------------
    LogicalUser::~LogicalUser(void)
    //--------------------------------------------------------------------------
    {
      if (op != nullptr)
        op->remove_mapping_reference(gen);
      if ((shard_proj != nullptr) && shard_proj->remove_reference())
        delete shard_proj;
    }

    /////////////////////////////////////////////////////////////
    // FieldState
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldState::FieldState(void) : open_state(NOT_OPEN), redop(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FieldState::FieldState(
        OpenState state, const FieldMask& m, RegionTreeNode* child)
      : open_state(state), redop(0)
    //--------------------------------------------------------------------------
    {
      if (open_children.insert(child, m))
        child->add_base_gc_ref(FIELD_STATE_REF);
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(
        const RegionUsage& usage, const FieldMask& m, RegionTreeNode* child)
      : redop(0)
    //--------------------------------------------------------------------------
    {
      if (IS_READ_ONLY(usage))
        open_state = OPEN_READ_ONLY;
      else if (IS_WRITE(usage))
        open_state = OPEN_READ_WRITE;
      else if (IS_REDUCE(usage))
      {
        open_state = OPEN_REDUCE;
        redop = usage.redop;
      }
      if (open_children.insert(child, m))
        child->add_base_gc_ref(FIELD_STATE_REF);
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const FieldState& rhs)
      : open_state(rhs.open_state), redop(rhs.redop)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs.open_children.empty());
#endif
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(FieldState&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      open_children.swap(rhs.open_children);
      open_state = rhs.open_state;
      redop = rhs.redop;
    }

    //--------------------------------------------------------------------------
    FieldState::~FieldState(void)
    //--------------------------------------------------------------------------
    {
      for (OrderedFieldMaskChildren::const_iterator it = open_children.begin();
           it != open_children.end(); it++)
        if (it->first->remove_base_gc_ref(FIELD_STATE_REF))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    FieldState& FieldState::operator=(const FieldState& rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(open_children.empty());
      assert(rhs.open_children.empty());
#endif
      open_state = rhs.open_state;
      redop = rhs.redop;
      return *this;
    }

    //--------------------------------------------------------------------------
    FieldState& FieldState::operator=(FieldState&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      open_children.swap(rhs.open_children);
      open_state = rhs.open_state;
      redop = rhs.redop;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool FieldState::overlaps(const FieldState& rhs) const
    //--------------------------------------------------------------------------
    {
      if (redop != rhs.redop)
        return false;
      // Now check the privilege states
      if (redop == 0)
        return (open_state == rhs.open_state);
      else
      {
#ifdef DEBUG_LEGION
        assert(open_state == OPEN_REDUCE);
        assert(rhs.open_state == OPEN_REDUCE);
#endif
        // Only support merging reduction fields with exactly the
        // same mask which should be single fields for reductions
        return (valid_fields() == rhs.valid_fields());
      }
    }

    //--------------------------------------------------------------------------
    void FieldState::merge(FieldState& rhs, RegionTreeNode* node)
    //--------------------------------------------------------------------------
    {
      if (!rhs.open_children.empty())
      {
        for (OrderedFieldMaskChildren::const_iterator it =
                 rhs.open_children.begin();
             it != rhs.open_children.end(); it++)
          // Remove duplicate references if we already had it
          if (!open_children.insert(it->first, it->second))
            it->first->remove_base_gc_ref(FIELD_STATE_REF);
        rhs.open_children.clear();
      }
      else
        open_children.relax_valid_mask(rhs.open_children.get_valid_mask());
#ifdef DEBUG_LEGION
      assert(redop == rhs.redop);
#endif
      if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(!open_children.empty());
#endif
        // For the reductions, handle the case where we need to merge
        // reduction modes, if they are all disjoint, we don't need
        // to distinguish between single and multi reduce
        if (node->are_all_children_disjoint())
        {
          open_state = OPEN_READ_WRITE;
          redop = 0;
        }
        else
          open_state = OPEN_REDUCE;
      }
    }

    //--------------------------------------------------------------------------
    bool FieldState::filter(const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      std::vector<RegionTreeNode*> to_delete;
      for (OrderedFieldMaskChildren::iterator it = open_children.begin();
           it != open_children.end(); it++)
      {
        it.filter(mask);
        if (!it->second)
          to_delete.push_back(it->first);
      }
      if (to_delete.size() < open_children.size())
      {
        for (std::vector<RegionTreeNode*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
        {
          open_children.erase(*it);
          if ((*it)->remove_base_gc_ref(FIELD_STATE_REF))
            delete (*it);
        }
      }
      else
      {
        open_children.clear();
        for (std::vector<RegionTreeNode*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
          if ((*it)->remove_base_gc_ref(FIELD_STATE_REF))
            delete (*it);
      }
      open_children.tighten_valid_mask();
      return open_children.empty();
    }

    //--------------------------------------------------------------------------
    void FieldState::add_child(RegionTreeNode* child, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      if (open_children.insert(child, mask))
        child->add_base_gc_ref(FIELD_STATE_REF);
    }

    //--------------------------------------------------------------------------
    void FieldState::remove_child(RegionTreeNode* child)
    //--------------------------------------------------------------------------
    {
      OrderedFieldMaskChildren::iterator finder = open_children.find(child);
#ifdef DEBUG_LEGION
      assert(finder != open_children.end());
      assert(!finder->second);
#endif
      open_children.erase(finder);
      if (child->remove_base_gc_ref(FIELD_STATE_REF))
        delete child;
    }

    /////////////////////////////////////////////////////////////
    // LogicalState
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalState::LogicalState(RegionTreeNode* node, ContextID c)
      : owner(node), total_timeout_check_iterations(MIN_TIMEOUT_CHECK_SIZE),
        remaining_timeout_check_iterations(total_timeout_check_iterations),
        timeout_exchange(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LogicalState::~LogicalState(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void LogicalState::check_init(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_states.empty());
      assert(curr_epoch_users.empty());
      assert(prev_epoch_users.empty());
      assert(timeout_exchange == nullptr);
      assert(refinement_trackers.empty());
      assert(projection_summary_cache.empty());
      assert(interfering_shards.empty());
      assert(pointwise_dependences.empty());
#endif
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void LogicalState::sanity_check(void) const
    //--------------------------------------------------------------------------
    {
      // For every child and every field, it should only be open in one mode
      FieldMaskSet<RegionTreeNode> previous_children;
      for (std::list<FieldState>::const_iterator fit = field_states.begin();
           fit != field_states.end(); fit++)
      {
        FieldMask actually_valid;
        for (FieldState::OrderedFieldMaskChildren::const_iterator it =
                 fit->open_children.begin();
             it != fit->open_children.end(); it++)
        {
          actually_valid |= it->second;
          FieldMaskSet<RegionTreeNode>::iterator finder =
              previous_children.find(it->first);
          if (finder != previous_children.end())
          {
            assert(!(finder->second & it->second));
            finder.merge(it->second);
          }
          else
            previous_children.insert(it->first, it->second);
        }
        // Actually valid should be greater than or equal
        assert(!(actually_valid - fit->valid_fields()));
      }
      // Make sure that each refinement has a disjoint set of fields
      if (!refinement_trackers.empty())
      {
        FieldMask disjoint_refinements;
        for (FieldMaskSet<RefinementTracker>::const_iterator it =
                 refinement_trackers.begin();
             it != refinement_trackers.end(); it++)
        {
          assert(disjoint_refinements * it->second);
          disjoint_refinements |= it->second;
        }
      }
    }
#endif

    //--------------------------------------------------------------------------
    void LogicalState::clear(void)
    //--------------------------------------------------------------------------
    {
      field_states.clear();
      if (!curr_epoch_users.empty())
      {
        for (OrderedFieldMaskUsers::const_iterator it =
                 curr_epoch_users.begin();
             it != curr_epoch_users.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        curr_epoch_users.clear();
      }
      if (!prev_epoch_users.empty())
      {
        for (OrderedFieldMaskUsers::const_iterator it =
                 prev_epoch_users.begin();
             it != prev_epoch_users.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        prev_epoch_users.clear();
      }
      total_timeout_check_iterations = MIN_TIMEOUT_CHECK_SIZE;
      remaining_timeout_check_iterations = total_timeout_check_iterations;
      if (timeout_exchange != nullptr)
      {
        timeout_exchange->perform_collective_wait();
        delete timeout_exchange;
        timeout_exchange = nullptr;
      }
      if (!refinement_trackers.empty())
      {
        for (FieldMaskSet<RefinementTracker>::const_iterator it =
                 refinement_trackers.begin();
             it != refinement_trackers.end(); it++)
          delete it->first;
        refinement_trackers.clear();
      }
      while (!projection_summary_cache.empty())
      {
        ProjectionSummary* summary = projection_summary_cache.back();
        projection_summary_cache.pop_back();
        if ((projection_summary_cache.size() < PROJECTION_CACHE_SIZE) &&
            summary->remove_reference())
          delete summary;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::clear_deleted_state(
        ContextID ctx, const FieldMask& deleted_mask)
    //--------------------------------------------------------------------------
    {
      for (LegionList<FieldState>::iterator it = field_states.begin();
           it != field_states.end();
           /*nothing*/)
      {
        if (it->filter(deleted_mask))
          it = field_states.erase(it);
        else
          it++;
      }
      if (!curr_epoch_users.empty() &&
          !(deleted_mask * curr_epoch_users.get_valid_mask()))
      {
        std::vector<LogicalUser*> to_delete;
        for (OrderedFieldMaskUsers::iterator it = curr_epoch_users.begin();
             it != curr_epoch_users.end(); it++)
        {
          it.filter(deleted_mask);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<LogicalUser*>::const_iterator it = to_delete.begin();
             it != to_delete.end(); it++)
        {
          curr_epoch_users.erase(*it);
          if ((*it)->remove_reference())
            delete (*it);
        }
        curr_epoch_users.tighten_valid_mask();
      }
      if (!prev_epoch_users.empty() &&
          !(deleted_mask * prev_epoch_users.get_valid_mask()))
      {
        std::vector<LogicalUser*> to_delete;
        for (OrderedFieldMaskUsers::iterator it = prev_epoch_users.begin();
             it != prev_epoch_users.end(); it++)
        {
          it.filter(deleted_mask);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<LogicalUser*>::const_iterator it = to_delete.begin();
             it != to_delete.end(); it++)
        {
          prev_epoch_users.erase(*it);
          if ((*it)->remove_reference())
            delete (*it);
        }
        prev_epoch_users.tighten_valid_mask();
      }
      if (!refinement_trackers.empty() &&
          !(deleted_mask * refinement_trackers.get_valid_mask()))
      {
        std::vector<RefinementTracker*> to_delete;
        for (FieldMaskSet<RefinementTracker>::iterator it =
                 refinement_trackers.begin();
             it != refinement_trackers.end(); it++)
        {
          const FieldMask overlap = deleted_mask & it->second;
          if (!it->second)
            continue;
          it->first->invalidate_refinement(ctx, overlap);
          it.filter(overlap);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<RefinementTracker*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
        {
          refinement_trackers.erase(*it);
          delete (*it);
        }
        refinement_trackers.tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    ProjectionSummary* LogicalState::find_or_create_projection_summary(
        Operation* op, unsigned index, const RegionRequirement& req,
        LogicalAnalysis& analysis, const ProjectionInfo& proj_info)
    //--------------------------------------------------------------------------
    {
      // Check to see if the projection functor is functional
      if (proj_info.projection->is_functional)
      {
        // Check to see if we can find this in the cache
        unsigned index = 0;
        ProjectionSummary* invalidated = nullptr;
        for (std::list<ProjectionSummary*>::iterator it =
                 projection_summary_cache.begin();
             it != projection_summary_cache.end(); it++, index++)
        {
          if ((*it)->matches(proj_info, req))
          {
            ProjectionSummary* result = *it;
            // Move it to the front of the list if it wasn't already there
            if (it != projection_summary_cache.begin())
            {
              projection_summary_cache.splice(
                  projection_summary_cache.begin(), projection_summary_cache,
                  it);
              // If this wasn't already in the cache range with a reference
              // then we need to update the reference counts
              if (PROJECTION_CACHE_SIZE <= index)
              {
                // Add a reference to the result
                result->add_reference();
#ifdef DEBUG_LEGION
                assert(invalidated != nullptr);
#endif
                if (invalidated->remove_reference())
                  delete invalidated;
              }
            }
            return result;
          }
          if (index == (PROJECTION_CACHE_SIZE - 1))
          {
            // If we make it here then this is the last entry in the
            // cache currenty and we're either going to match on an
            // entry outside the cache and move it into the cache
            // or we're going to make a new entry so this summary
            // is going to be invalidated either way, so save it
            // so we can do the invalidation after we're done
            // traversing the list when we won't mess with the iterator
            invalidated = (*it);
          }
        }
        if ((invalidated != nullptr) && invalidated->remove_reference())
          delete invalidated;
      }
      ProjectionSummary* result =
          analysis.context->construct_projection_summary(
              op, index, req, this, proj_info);
      // If the projection functor is functional then we can save it for
      // the future and evict the least recently used projection
      if (proj_info.projection->is_functional)
      {
        result->add_reference();
        projection_summary_cache.push_front(result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void LogicalState::remove_projection_summary(ProjectionSummary* summary)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(summary->owner == this);
#endif
      if (summary->projection->is_functional &&
          (PROJECTION_CACHE_SIZE <= projection_summary_cache.size()))
      {
        // Only need to filter from things not in the cache so start
        // from the back and only go up to the last element not in the cache
        // Note: handle the off-by-one case where we're deleting the last
        // thing in the cache because we're about to add a new summary
        const unsigned stop =
            (projection_summary_cache.size() - PROJECTION_CACHE_SIZE) + 1;
        std::list<ProjectionSummary*>::reverse_iterator finder =
            projection_summary_cache.rbegin();
        for (unsigned idx = 0; idx < stop; idx++)
        {
          if ((*finder) == summary)
          {
            // Reverse iterators are stupid
            projection_summary_cache.erase(std::next(finder).base());
            break;
          }
          finder++;
        }
      }
      if (summary->can_perform_name_based_self_analysis())
      {
        std::unordered_map<
            ProjectionSummary*,
            std::unordered_map<ProjectionSummary*, std::pair<bool, bool> > >::
            iterator finder = pointwise_dependences.find(summary);
        if (finder != pointwise_dependences.end())
        {
          for (std::unordered_map<ProjectionSummary*, std::pair<bool, bool> >::
                   const_iterator it = finder->second.begin();
               it != finder->second.end(); it++)
            pointwise_dependences[it->first].erase(summary);
          pointwise_dependences.erase(finder);
        }
      }
      std::unordered_map<
          ProjectionSummary*,
          std::unordered_map<ProjectionSummary*, std::pair<bool, bool> > >::
          iterator finder = interfering_shards.find(summary);
      if (finder != interfering_shards.end())
      {
        for (std::unordered_map<ProjectionSummary*, std::pair<bool, bool> >::
                 const_iterator it = finder->second.begin();
             it != finder->second.end(); it++)
          interfering_shards[it->first].erase(summary);
        interfering_shards.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    bool LogicalState::has_interfering_shards(
        LogicalAnalysis& analysis, ProjectionSummary* one,
        ProjectionSummary* two, bool& dominates)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dominates);
      assert(one->owner == this);
      assert(two->owner == this);
#endif
      if (one == two)
        // We can elide the close operation here if we can prove that
        // all the regions used a disjoint from each other and they all
        // have exactly one kind of shard user
        return (
            !one->can_perform_name_based_self_analysis() ||
            !one->has_unique_shard_users());
      std::unordered_map<
          ProjectionSummary*,
          std::unordered_map<ProjectionSummary*, std::pair<bool, bool> > >::
          const_iterator one_finder = interfering_shards.find(one);
      if (one_finder != interfering_shards.end())
      {
        std::unordered_map<ProjectionSummary*, std::pair<bool, bool> >::
            const_iterator two_finder = one_finder->second.find(two);
        if (two_finder != one_finder->second.end())
        {
          dominates = two_finder->second.second;
          return two_finder->second.first;
        }
      }
      // Do the test and save the results for later
      const bool result =
          analysis.context->has_interfering_shards(one, two, dominates);
      interfering_shards[one][two] = std::make_pair(result, dominates);
      interfering_shards[two][one] = std::make_pair(result, dominates);
      return result;
    }

    //--------------------------------------------------------------------------
    bool LogicalState::record_pointwise_dependence(
        LogicalAnalysis& analysis, const LogicalUser& prev,
        const LogicalUser& next, bool& dominates)
    //--------------------------------------------------------------------------
    {
      if (!prev.pointwise_analyzable)
        return false;
      if (!next.pointwise_analyzable)
        return false;
      ProjectionSummary* one = prev.shard_proj;
      ProjectionSummary* two = next.shard_proj;
#ifdef DEBUG_LEGION
      assert(one != nullptr);
      assert(two != nullptr);
      assert(one->owner == this);
      assert(two->owner == this);
#endif
      // In order to do pointwise analysis then each of them have to support
      // name based self-analysis meaning all the points are accessing
      // disjoint data and all the accesses are at the leaves
      if (!one->can_perform_name_based_self_analysis())
        return false;
      if (!two->can_perform_name_based_self_analysis())
        return false;
      // If they're the same summary then we can do pointiwse analysis
      if (one == two)
      {
        dominates = true;
        return true;
      }
      // Before we do the local analysis, see if we can find the result
      // in the cache from a prior computation
      std::unordered_map<
          ProjectionSummary*,
          std::unordered_map<ProjectionSummary*, std::pair<bool, bool> > >::
          const_iterator one_finder = pointwise_dependences.find(one);
      if (one_finder != pointwise_dependences.end())
      {
        std::unordered_map<ProjectionSummary*, std::pair<bool, bool> >::
            const_iterator two_finder = one_finder->second.find(two);
        if (two_finder != one_finder->second.end())
        {
          dominates = two_finder->second.second;
          return two_finder->second.first;
        }
      }
      // If they are different summaries, but are using the same projection
      // function and either one's launch domain is a (non-strict) subset of
      // the other's launch domain then we can do pointwise dependence
      // analysis. This follows from the fact that we know that both launches
      // can do intra-space named based dependence analysis so their is no
      // aliasing between the subregions of either projection. Therefore
      // whichever points exist in one launch domain but not in the other
      // cannot alias with the ones that overlap (or alias in a way that
      // supports name-based analysis with other points). However, if both
      // launch domains are just overlapping with neither dominating then
      // the non-overlapping points in each projection could map to aliasing
      // regions and therefore not be safe for name-based analysis.
      if ((one->projection == two->projection) &&
          ((one->args == two->args) ||
           ((one->arglen == two->arglen) &&
            (std::memcmp(one->args, two->args, one->arglen) == 0))))
      {
        if (one->domain == two->domain)
        {
          // If they both have the same domain then this is easy
          pointwise_dependences[one][two] = std::make_pair(true, true);
          pointwise_dependences[two][one] = std::make_pair(true, true);
          dominates = true;
          return true;
        }
        else
        {
          const bool one_dominates = one->domain->dominates(two->domain);
          dominates = two->domain->dominates(one->domain);
          const bool result = one_dominates || dominates;
          pointwise_dependences[one][two] = std::make_pair(result, dominates);
          // Keep the data structure symmetric for when we go to
          // prune out projection summaries
          pointwise_dependences[two][one] =
              std::make_pair(result, one_dominates);
          return result;
        }
      }
      // If we get here, do the more expensive check to see if we can do
      // point-wise analysis locally between the two summaries in our
      // local address space and then all-reduce the result between any
      // shards to see if they are all local and then save the result
      // in the cache for future cases.
      const std::pair<bool, bool> dominance =
          analysis.context->has_pointwise_dominance(one, two);
      // Same logic applies here as above: if either dominates then you
      // can do pointwise dependence analysis
      const bool result = dominance.first || dominance.second;
      pointwise_dependences[one][two] =
          std::make_pair(result, dominance.second);
      pointwise_dependences[two][one] = std::make_pair(result, dominance.first);
      dominates = dominance.second;
      return result;
    }

    //--------------------------------------------------------------------------
    void LogicalState::initialize_no_refine_fields(const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner->is_region());
      assert(mask * refinement_trackers.get_valid_mask());
#endif
      RefinementTracker* new_tracker = owner->create_refinement_tracker();
      new_tracker->initialize_no_refine();
      refinement_trackers.insert(new_tracker, mask);
    }

    //--------------------------------------------------------------------------
    void LogicalState::update_refinement_child(
        ContextID ctx, RegionTreeNode* child, const RegionUsage& usage,
        FieldMask& refinement_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!refinement_mask);
#endif
      // This function filters through the refinement trackers and updates
      // them as necessary. Since the refinement trackers are not field aware,
      // this function will clone them and delete them as necessary to make
      // sure that each field is accurately represented
      FieldMask need_tracker = refinement_mask;
      if (!(refinement_mask * refinement_trackers.get_valid_mask()))
      {
        FieldMaskSet<RefinementTracker> to_add;
        std::vector<RefinementTracker*> to_delete;
        for (FieldMaskSet<RefinementTracker>::iterator it =
                 refinement_trackers.begin();
             it != refinement_trackers.end(); it++)
        {
          const FieldMask overlap = refinement_mask & it->second;
          if (!overlap)
            continue;
          if (overlap != it->second)
          {
            RefinementTracker* diff = it->first->clone();
            FieldMask diff_mask = it->second - overlap;
            to_add.insert(diff, diff_mask);
            it.filter(diff_mask);
          }
          bool allow_refinement = false;
          if (it->first->update_child(child, usage, allow_refinement))
          {
            it->first->invalidate_refinement(ctx, overlap);
            to_delete.push_back(it->first);
          }
          else
          {
            if (!allow_refinement)
              refinement_mask -= overlap;
            need_tracker -= overlap;
            if (!need_tracker)
              break;
          }
        }
        // Remove old entries
        for (std::vector<RefinementTracker*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
        {
          refinement_trackers.erase(*it);
          delete (*it);
        }
        // Add new entries
        for (FieldMaskSet<RefinementTracker>::const_iterator it =
                 to_add.begin();
             it != to_add.end(); it++)
          refinement_trackers.insert(it->first, it->second);
      }
      if (!!need_tracker)
      {
        RefinementTracker* new_tracker = owner->create_refinement_tracker();
        bool allow_refinement = false;
        if (new_tracker->update_child(child, usage, allow_refinement))
          std::abort();  // should never get here
        if (!allow_refinement)
          refinement_mask -= need_tracker;
        refinement_trackers.insert(new_tracker, need_tracker);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::update_refinement_projection(
        ContextID ctx, ProjectionSummary* summary, const RegionUsage& usage,
        FieldMask& refinement_mask)
    //--------------------------------------------------------------------------
    {
      // This function filters through the refinement trackers and updates
      // them as necessary. Since the refinement trackers are not field aware,
      // this function will clone them and delete them as necessary to make
      // sure that each field is accurately represented
      FieldMask need_tracker = refinement_mask;
      if (!(refinement_mask * refinement_trackers.get_valid_mask()))
      {
        FieldMaskSet<RefinementTracker> to_add;
        std::vector<RefinementTracker*> to_delete;
        for (FieldMaskSet<RefinementTracker>::iterator it =
                 refinement_trackers.begin();
             it != refinement_trackers.end(); it++)
        {
          const FieldMask overlap = refinement_mask & it->second;
          if (!overlap)
            continue;
          if (overlap != it->second)
          {
            RefinementTracker* diff = it->first->clone();
            FieldMask diff_mask = it->second - overlap;
            to_add.insert(diff, diff_mask);
            it.filter(diff_mask);
          }
          bool allow_refinement = false;
          if (it->first->update_projection(summary, usage, allow_refinement))
          {
            it->first->invalidate_refinement(ctx, overlap);
            to_delete.push_back(it->first);
          }
          else
          {
            if (!allow_refinement)
              refinement_mask -= overlap;
            need_tracker -= overlap;
            if (!need_tracker)
              break;
          }
        }
        // Remove old entries
        for (std::vector<RefinementTracker*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
        {
          refinement_trackers.erase(*it);
          delete (*it);
        }
        // Add new entries
        for (FieldMaskSet<RefinementTracker>::const_iterator it =
                 to_add.begin();
             it != to_add.end(); it++)
          refinement_trackers.insert(it->first, it->second);
      }
      if (!!need_tracker)
      {
        RefinementTracker* new_tracker = owner->create_refinement_tracker();
        bool allow_refinement = false;
        if (new_tracker->update_projection(summary, usage, allow_refinement))
          std::abort();  // should never get here
        if (!allow_refinement)
          refinement_mask -= need_tracker;
        refinement_trackers.insert(new_tracker, need_tracker);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::update_refinement_arrival(
        ContextID ctx, const RegionUsage& usage, FieldMask& refinement_mask)
    //--------------------------------------------------------------------------
    {
      // This function filters through the refinement trackers and updates
      // them as necessary. Since the refinement trackers are not field aware,
      // this function will clone them and delete them as necessary to make
      // sure that each field is accurately represented
      if (!(refinement_mask * refinement_trackers.get_valid_mask()))
      {
        FieldMaskSet<RefinementTracker> to_add;
        std::vector<RefinementTracker*> to_delete;
        for (FieldMaskSet<RefinementTracker>::iterator it =
                 refinement_trackers.begin();
             it != refinement_trackers.end(); it++)
        {
          const FieldMask overlap = refinement_mask & it->second;
          if (!overlap)
            continue;
          if (overlap != it->second)
          {
            RefinementTracker* diff = it->first->clone();
            FieldMask diff_mask = it->second - overlap;
            to_add.insert(diff, diff_mask);
            it.filter(diff_mask);
          }
          if (it->first->update_arrival(usage))
          {
            it->first->invalidate_refinement(ctx, overlap);
            to_delete.push_back(it->first);
          }
          else
          {
            refinement_mask -= overlap;
            if (!refinement_mask)
              break;
          }
        }
        // Remove old entries
        for (std::vector<RefinementTracker*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
        {
          refinement_trackers.erase(*it);
          delete (*it);
        }
        // Add new entries
        for (FieldMaskSet<RefinementTracker>::const_iterator it =
                 to_add.begin();
             it != to_add.end(); it++)
          refinement_trackers.insert(it->first, it->second);
      }
      if (!!refinement_mask)
      {
        RefinementTracker* new_tracker = owner->create_refinement_tracker();
        if (new_tracker->update_arrival(usage))
          std::abort();  // should never get here
        refinement_trackers.insert(new_tracker, refinement_mask);
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::invalidate_refinements(
        ContextID ctx, FieldMask invalidation_mask)
    //--------------------------------------------------------------------------
    {
      std::vector<RefinementTracker*> to_delete;
      for (FieldMaskSet<RefinementTracker>::iterator it =
               refinement_trackers.begin();
           it != refinement_trackers.end(); it++)
      {
        const FieldMask overlap = invalidation_mask & it->second;
        if (!overlap)
          continue;
        it->first->invalidate_refinement(ctx, overlap);
        it.filter(overlap);
        if (!it->second)
          to_delete.push_back(it->first);
        invalidation_mask -= overlap;
        if (!invalidation_mask)
          break;
      }
#ifdef DEBUG_LEGION
      assert(!invalidation_mask);  // should have seen all the fields
#endif
      for (std::vector<RefinementTracker*>::const_iterator it =
               to_delete.begin();
           it != to_delete.end(); it++)
      {
        refinement_trackers.erase(*it);
        delete (*it);
      }
      refinement_trackers.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void LogicalState::record_refinement_dependences(
        ContextID ctx, const LogicalUser& refinement_user,
        const FieldMask& refinement_mask, const ProjectionInfo& no_proj_info,
        RegionTreeNode* previous_child, LogicalRegion privilege_root,
        LogicalAnalysis& logical_analysis)
    //--------------------------------------------------------------------------
    {
      FieldMask dummy_open_below;
      // First register dependences on any local users, we can't dominate
      // anything here since we need to leave things here for later
      owner->perform_dependence_checks<false /*track dom*/>(
          privilege_root, refinement_user, curr_epoch_users, refinement_mask,
          dummy_open_below, false /*arrived*/, no_proj_info, *this,
          logical_analysis);
      owner->perform_dependence_checks<false /*track dom*/>(
          privilege_root, refinement_user, prev_epoch_users, refinement_mask,
          dummy_open_below, false /*arrived*/, no_proj_info, *this,
          logical_analysis);
      // If we have a previous child and all the children are independent
      // then we know we don't need to traverse anything else
      if ((previous_child == nullptr) || !owner->are_all_children_disjoint())
      {
        // Now traverse any open children and record dependences on them as well
        for (LegionList<FieldState, SHORT_BOUNDED_LIFETIME>::const_iterator
                 fit = field_states.begin();
             fit != field_states.end(); fit++)
        {
          const FieldMask field_overlap = fit->valid_fields() & refinement_mask;
          if (!field_overlap)
            continue;
          for (FieldState::OrderedFieldMaskChildren::const_iterator it =
                   fit->open_children.begin();
               it != fit->open_children.end(); it++)
          {
            // Can skip the previous child if we've already done it
            if (it->first == previous_child)
              continue;
            const FieldMask overlap = refinement_mask & it->second;
            if (!overlap)
              continue;
            if ((previous_child != nullptr) &&
                owner->are_children_disjoint(
                    previous_child->get_color(), it->first->get_color()))
              continue;
            it->first->record_refinement_dependences(
                ctx, refinement_user, overlap, no_proj_info,
                nullptr /*previous child*/, privilege_root, logical_analysis);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::filter_previous_epoch_users(const FieldMask& field_mask)
    //--------------------------------------------------------------------------
    {
      std::vector<LogicalUser*> to_delete;
      for (OrderedFieldMaskUsers::iterator it = prev_epoch_users.begin();
           it != prev_epoch_users.end(); it++)
      {
        it.filter(field_mask);
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<LogicalUser*>::const_iterator it = to_delete.begin();
           it != to_delete.end(); it++)
      {
        prev_epoch_users.erase(*it);
        if ((*it)->remove_reference())
          delete (*it);
      }
      prev_epoch_users.filter_valid_mask(field_mask);
    }

    //--------------------------------------------------------------------------
    void LogicalState::register_local_user(
        LogicalUser& user, const FieldMask& user_mask)
    //--------------------------------------------------------------------------
    {
      if (curr_epoch_users.insert(&user, user_mask))
        user.add_reference();
    }

    //--------------------------------------------------------------------------
    void LogicalState::filter_current_epoch_users(const FieldMask& field_mask)
    //--------------------------------------------------------------------------
    {
      std::vector<LogicalUser*> to_delete;
      for (OrderedFieldMaskUsers::iterator it = curr_epoch_users.begin();
           it != curr_epoch_users.end(); it++)
      {
        const FieldMask local_dom = it->second & field_mask;
        if (!local_dom)
          continue;
        if (prev_epoch_users.insert(it->first, local_dom))
          it->first->add_reference();
        it.filter(local_dom);
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<LogicalUser*>::const_iterator it = to_delete.begin();
           it != to_delete.end(); it++)
      {
        curr_epoch_users.erase(*it);
        if ((*it)->remove_reference())
          delete (*it);
      }
      curr_epoch_users.filter_valid_mask(field_mask);
    }

    //--------------------------------------------------------------------------
    void LogicalState::filter_timeout_users(LogicalAnalysis& analysis)
    //--------------------------------------------------------------------------
    {
      // We only run this function if we're not doing legion spy verification
      // as it will mess up the logical dependence analysis verification
#ifndef LEGION_SPY
      // Skip this if we're tracing
      if (analysis.op->is_tracing())
        return;
      // First check to see if we have more users than the maximum allowed
      if ((curr_epoch_users.size() + prev_epoch_users.size()) <=
          MIN_TIMEOUT_CHECK_SIZE)
        return;
      // Next check to see if we've hit the right number of remaining
      // iterations to perform this check
      if (--remaining_timeout_check_iterations > 0)
        return;
      // Go through and filter any current or previous epoch users that have
      // been committed and therefore can no longer be rolled back
      std::vector<LogicalUser*> timeout_users;
      for (OrderedFieldMaskUsers::const_iterator it = curr_epoch_users.begin();
           it != curr_epoch_users.end(); it++)
      {
        if (!it->first->op->is_operation_committed(it->first->gen))
          continue;
        it->first->add_reference();
        timeout_users.push_back(it->first);
      }
      const size_t prev_size = timeout_users.size();
      for (OrderedFieldMaskUsers::const_iterator it = prev_epoch_users.begin();
           it != prev_epoch_users.end(); it++)
      {
        if (!it->first->op->is_operation_committed(it->first->gen))
          continue;
        if (std::binary_search(
                timeout_users.begin(), timeout_users.begin() + prev_size,
                it->first))
          continue;
        it->first->add_reference();
        timeout_users.push_back(it->first);
      }
      // Now we do the exchange and record whether we need to double
      // the timeout check iterations
      std::vector<LogicalUser*> to_delete;
      if (analysis.context->match_timeouts(
              timeout_users, to_delete, timeout_exchange))
        total_timeout_check_iterations *= 2;
      remaining_timeout_check_iterations = total_timeout_check_iterations;
      bool tighten_current = false;
      bool tighten_previous = false;
      for (std::vector<LogicalUser*>::const_iterator it = to_delete.begin();
           it != to_delete.end(); it++)
      {
        // One reference from when we were added in the
        // perform dependence checks function
        unsigned references_to_remove = 1;
        OrderedFieldMaskUsers::iterator finder = curr_epoch_users.find(*it);
        if (finder != curr_epoch_users.end())
        {
          curr_epoch_users.erase(finder);
          references_to_remove++;
          tighten_current = true;
        }
        finder = prev_epoch_users.find(*it);
        if (finder != prev_epoch_users.end())
        {
          prev_epoch_users.erase(finder);
          references_to_remove++;
          tighten_previous = true;
        }
        if ((*it)->remove_reference(references_to_remove))
          delete (*it);
      }
      if (tighten_current)
        curr_epoch_users.tighten_valid_mask();
      if (tighten_previous)
        prev_epoch_users.tighten_valid_mask();
#endif
    }

    //--------------------------------------------------------------------------
    void LogicalState::promote_next_child(RegionTreeNode* child, FieldMask mask)
    //--------------------------------------------------------------------------
    {
      // This will promote the child up to a read-write field state
      for (LegionList<FieldState, SHORT_BOUNDED_LIFETIME>::iterator it =
               field_states.begin();
           it != field_states.end(); it++)
      {
        const FieldMask overlap = mask & it->valid_fields();
        if (!overlap)
          continue;
        FieldState::OrderedFieldMaskChildren::iterator finder =
            it->open_children.find(child);
        if (finder == it->open_children.end())
          continue;
        if ((it->open_children.size() == 1) && (overlap == finder->second))
        {
          // We can just update this state directly
          it->open_state = OPEN_READ_WRITE;
          it->redop = 0;
          mask -= overlap;
          if (!mask)
            return;
        }
        else
        {
          // Remove the child from the state
          finder.filter(overlap);
          if (!finder->second)
            it->remove_child(child);
          it->open_children.tighten_valid_mask();
        }
      }
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      // If we get here we still have fields so we need to introduce a new
      // field state here for this child in read-write mode
      field_states.emplace_back(FieldState(OPEN_READ_WRITE, mask, child));
    }

    /////////////////////////////////////////////////////////////
    // Logical Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalAnalysis::LogicalAnalysis(Operation* o, unsigned out_off)
      : op(o), context(op->get_context()), output_region_offset(out_off)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LogicalAnalysis::~LogicalAnalysis(void)
    //--------------------------------------------------------------------------
    {
      // If we have any pending refinements, have them record dependences
      // on any pending closes that were done along their path and then
      // issue the refinements
      unsigned internal_index = 0;
      for (OrderedRefinements::const_iterator it = pending_refinements.begin();
           it != pending_refinements.end(); it++)
      {
        RegionTreeNode* node = it->first->get_refinement_node();
        RegionTreeNode* path_node = node;
        while (path_node != nullptr)
        {
          std::map<RegionTreeNode*, MergeCloseOp*>::const_iterator finder =
              pending_closes.find(path_node);
          if (finder != pending_closes.end())
          {
            FieldMask overlap = it->second & finder->second->get_close_mask();
            if (!!overlap)
            {
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  context->get_unique_id(), finder->second->get_unique_op_id(),
                  0 /*index*/, it->first->get_unique_op_id(), 0 /*index*/,
                  LEGION_TRUE_DEPENDENCE);
#endif
              it->first->register_region_dependence(
                  0 /*index*/, finder->second, finder->second->get_generation(),
                  0 /*index*/, LEGION_TRUE_DEPENDENCE, overlap);
            }
          }
          path_node = path_node->get_parent();
        }
        it->first->record_refinement_mask(internal_index, it->second);
        issue_internal_operation(node, it->first, it->second, internal_index++);
      }
      // Issue the pending closes
      if (!pending_closes.empty())
      {
        // Need to issue these close operations in order in case we are
        // control replicated and therefore all the shards need to see
        // the closes in the same order for things to work correctly
        std::map<LogicalRegion, RegionTreeNode*> ordered_region_closes;
        std::map<LogicalPartition, RegionTreeNode*> ordered_partition_closes;
        for (std::map<RegionTreeNode*, MergeCloseOp*>::const_iterator it =
                 pending_closes.begin();
             it != pending_closes.end(); it++)
        {
          if (it->first->is_region())
          {
            RegionNode* region = it->first->as_region_node();
            ordered_region_closes[region->handle] = it->first;
          }
          else
          {
            PartitionNode* partition = it->first->as_partition_node();
            ordered_partition_closes[partition->handle] = it->first;
          }
        }
        for (std::map<LogicalRegion, RegionTreeNode*>::const_iterator it =
                 ordered_region_closes.begin();
             it != ordered_region_closes.end(); it++)
        {
          MergeCloseOp* close = pending_closes[it->second];
          issue_internal_operation(
              it->second, close, close->get_close_mask(), internal_index++);
        }
        for (std::map<LogicalPartition, RegionTreeNode*>::const_iterator it =
                 ordered_partition_closes.begin();
             it != ordered_partition_closes.end(); it++)
        {
          MergeCloseOp* close = pending_closes[it->second];
          issue_internal_operation(
              it->second, close, close->get_close_mask(), internal_index++);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalAnalysis::record_pending_refinement(
        LogicalRegion privilege, unsigned req_index, unsigned parent_req_index,
        RegionTreeNode* refinement_node, const FieldMask& refinement_mask,
        OrderedRefinements& pending)
    //--------------------------------------------------------------------------
    {
      // Ignore any requests for refinements for output region requirmeents
      if (output_region_offset <= req_index)
        return;
      // See if we already have a refinement for handling this node
      for (OrderedRefinements::iterator it = pending_refinements.begin();
           it != pending_refinements.end(); it++)
      {
        if (it->first->get_refinement_node() != refinement_node)
          continue;
        it.merge(refinement_mask);
        pending.insert(it->first, refinement_mask);
        return;
      }
#ifdef DEBUG_LEGION_COLLECTIVES
      RefinementOp* refinement_op =
          context->get_refinement_op(op, refinement_node);
#else
      RefinementOp* refinement_op = context->get_refinement_op();
#endif
      refinement_op->initialize(
          op, req_index, privilege, refinement_node, parent_req_index);
      // Start the dependence analysis for this refinement now
      // We'll finish the dependence analysis in the destructor
      refinement_op->begin_dependence_analysis();
      pending_refinements.insert(refinement_op, refinement_mask);
      pending.insert(refinement_op, refinement_mask);
    }

    //--------------------------------------------------------------------------
    void LogicalAnalysis::record_close_dependence(
        LogicalRegion parent, unsigned req_index, RegionTreeNode* path_node,
        const LogicalUser* user, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeNode*, MergeCloseOp*>::const_iterator finder =
          pending_closes.find(path_node);
      if (finder == pending_closes.end())
      {
        // Start a new close operation for this node in the region tree
        // Construct a reigon requirement for this operation
        // All privileges are based on the parent logical region
        RegionRequirement req;
        if (path_node->is_region())
          req = RegionRequirement(
              path_node->as_region_node()->handle, LEGION_READ_WRITE,
              LEGION_EXCLUSIVE, parent);
        else
          req = RegionRequirement(
              path_node->as_partition_node()->handle, 0, LEGION_READ_WRITE,
              LEGION_EXCLUSIVE, parent);
#ifdef DEBUG_LEGION_COLLECTIVES
        MergeCloseOp* close_op = context->get_merge_close_op(op, path_node);
#else
        MergeCloseOp* close_op = context->get_merge_close_op();
#endif
        close_op->initialize(context, req, req_index, op);
        // Mark that we are starting our dependence analysis
        close_op->begin_dependence_analysis();
        finder =
            pending_closes.insert(std::make_pair(path_node, close_op)).first;
      }
#ifdef LEGION_SPY
      LegionSpy::log_mapping_dependence(
          context->get_unique_id(), user->uid, user->idx,
          finder->second->get_unique_op_id(), 0 /*index*/,
          LEGION_TRUE_DEPENDENCE);
#endif
      finder->second->register_region_dependence(
          0 /*index*/, user->op, user->gen, user->idx, LEGION_TRUE_DEPENDENCE,
          mask);
      finder->second->update_close_mask(mask);
    }

    //--------------------------------------------------------------------------
    void LogicalAnalysis::issue_internal_operation(
        RegionTreeNode* node, InternalOp* internal_op,
        const FieldMask& internal_mask, const unsigned internal_index) const
    //--------------------------------------------------------------------------
    {
      // Do any other work for the dependence analysis
      internal_op->trigger_dependence_analysis();
      // Record a user for this internal operation in the region tree
      LogicalUser* user = new LogicalUser(
          internal_op, 0 /*region index*/,
          RegionUsage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0 /*redop*/),
          nullptr /*projection*/, internal_index);
      LogicalState& state =
          node->get_logical_state(context->get_logical_tree_context());
      // This will take ownership of the user
      state.register_local_user(*user, internal_mask);
      // Record a dependence on the internal operation for ourself
      op->register_region_dependence(
          internal_op->get_internal_index(), internal_op,
          internal_op->get_generation(), 0 /*internal idx*/,
          LEGION_TRUE_DEPENDENCE, internal_mask);
#ifdef LEGION_SPY
      LegionSpy::log_mapping_dependence(
          context->get_unique_id(), internal_op->get_unique_op_id(),
          0 /*index*/, op->get_unique_op_id(),
          internal_op->get_internal_index(), LEGION_TRUE_DEPENDENCE);
#endif
      // Mark that we are done, this puts the op in the pipeline!
      internal_op->end_dependence_analysis();
    }

    /////////////////////////////////////////////////////////////
    // PathTraverser
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PathTraverser::PathTraverser(RegionTreePath& p) : path(p)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PathTraverser::~PathTraverser(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool PathTraverser::traverse(RegionTreeNode* node)
    //--------------------------------------------------------------------------
    {
      // Continue visiting nodes and then finding their children
      // until we have traversed the entire path.
      while (true)
      {
#ifdef DEBUG_LEGION
        assert(node != nullptr);
#endif
        depth = node->get_depth();
        has_child = path.has_child(depth);
        if (has_child)
          next_child = path.get_child(depth);
        bool continue_traversal = node->visit_node(this);
        if (!continue_traversal)
          return false;
        if (!has_child)
          break;
        node = node->get_tree_child(next_child);
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // CurrentInitializer
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CurrentInitializer::CurrentInitializer(ContextID c) : ctx(c)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CurrentInitializer::~CurrentInitializer(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_region(RegionNode* node)
    //--------------------------------------------------------------------------
    {
      node->initialize_current_state(ctx);
      return true;
    }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_partition(PartitionNode* node)
    //--------------------------------------------------------------------------
    {
      node->initialize_current_state(ctx);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // CurrentInvalidator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CurrentInvalidator::CurrentInvalidator(ContextID c) : ctx(c)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CurrentInvalidator::~CurrentInvalidator(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_region(RegionNode* node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_current_state(ctx);
      return true;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_partition(PartitionNode* node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_current_state(ctx);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // DeletionInvalidator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeletionInvalidator::DeletionInvalidator(ContextID c, const FieldMask& dm)
      : ctx(c), deletion_mask(dm)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    DeletionInvalidator::~DeletionInvalidator(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_region(RegionNode* node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_deleted_state(ctx, deletion_mask);
      return true;
    }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_partition(PartitionNode* node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_deleted_state(ctx, deletion_mask);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // RegionTreePath
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreePath::RegionTreePath(void) : min_depth(0), max_depth(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void RegionTreePath::initialize(unsigned min, unsigned max)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min <= max);
#endif
      min_depth = min;
      max_depth = max;
      path.resize(max_depth + 1, INVALID_COLOR);
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::register_child(unsigned depth, const LegionColor color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      path[depth] = color;
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::clear(void)
    //--------------------------------------------------------------------------
    {
      path.clear();
      min_depth = 0;
      max_depth = 0;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    bool RegionTreePath::has_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      return (path[depth] != INVALID_COLOR);
    }

    //--------------------------------------------------------------------------
    LegionColor RegionTreePath::get_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      assert(has_child(depth));
      return path[depth];
    }
#endif

  }  // namespace Internal
}  // namespace Legion
