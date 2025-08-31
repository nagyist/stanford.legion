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

#include <cmath>

#include "legion/analysis/refinement.h"
#include "legion/analysis/projection.h"
#include "legion/api/functors_impl.h"
#include "legion/nodes/region.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // RegionRefinementTracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionRefinementTracker::RegionRefinementTracker(RegionNode* node)
      : region(node), refinement_state(UNREFINED_STATE), refined_child(nullptr),
        refined_projection(nullptr), total_traversals(0), return_timeout(0)
    //--------------------------------------------------------------------------
    {
      region->add_base_resource_ref(REFINEMENT_REF);
    }

    //--------------------------------------------------------------------------
    RegionRefinementTracker::~RegionRefinementTracker(void)
    //--------------------------------------------------------------------------
    {
      if ((refined_child != nullptr) &&
          refined_child->remove_base_resource_ref(REFINEMENT_REF))
        delete refined_child;
      if ((refined_projection != nullptr) &&
          refined_projection->remove_reference())
        delete refined_projection;
      for (std::unordered_map<PartitionNode*, std::pair<double, uint64_t> >::
               const_iterator it = candidate_partitions.begin();
           it != candidate_partitions.end(); it++)
        if (it->first->remove_base_resource_ref(REFINEMENT_REF))
          delete it->first;
      for (std::unordered_map<ProjectionRegion*, std::pair<double, uint64_t> >::
               const_iterator it = candidate_projections.begin();
           it != candidate_projections.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      if (region->remove_base_resource_ref(REFINEMENT_REF))
        delete region;
    }

    //--------------------------------------------------------------------------
    RefinementTracker* RegionRefinementTracker::clone(void) const
    //--------------------------------------------------------------------------
    {
      RegionRefinementTracker* tracker = new RegionRefinementTracker(region);
      tracker->refinement_state = refinement_state;
      if (refined_child != nullptr)
      {
        tracker->refined_child = refined_child;
        refined_child->add_base_resource_ref(REFINEMENT_REF);
      }
      if (refined_projection != nullptr)
      {
        tracker->refined_projection = refined_projection;
        refined_projection->add_reference();
      }
      for (std::unordered_map<PartitionNode*, std::pair<double, uint64_t> >::
               const_iterator it = candidate_partitions.begin();
           it != candidate_partitions.end(); it++)
      {
        it->first->add_base_resource_ref(REFINEMENT_REF);
        tracker->candidate_partitions.insert(*it);
      }
      for (std::unordered_map<ProjectionRegion*, std::pair<double, uint64_t> >::
               const_iterator it = candidate_projections.begin();
           it != candidate_projections.end(); it++)
      {
        it->first->add_reference();
        tracker->candidate_projections.insert(*it);
      }
      tracker->total_traversals = total_traversals;
      tracker->return_timeout = return_timeout;
      return tracker;
    }

    //--------------------------------------------------------------------------
    void RegionRefinementTracker::initialize_no_refine(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(refined_child == nullptr);
      legion_assert(refinement_state == UNREFINED_STATE);
      refinement_state = NO_REFINEMENT_STATE;
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::is_child_current_refinement(
        RegionTreeNode* child) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!child->is_region());
      return (child == refined_child);
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::update_child(
        RegionTreeNode* node, const RegionUsage& usage, ContextID ctx,
        const FieldMask& current_mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(!node->is_region());
      PartitionNode* child = node->as_partition_node();
      // First see if we need to change states
      switch (refinement_state)
      {
        case UNREFINED_STATE:
          {
            legion_assert(refined_child == nullptr);
            legion_assert(refined_projection == nullptr);
            // If we don't have any refinements, we'll always allow them
            if (child->row_source->is_complete())
              refinement_state = IS_WRITE(usage) ?
                                     COMPLETE_WRITE_REFINED_STATE :
                                     COMPLETE_NONWRITE_REFINED_STATE;
            else
              refinement_state = IS_WRITE(usage) ?
                                     INCOMPLETE_WRITE_REFINED_STATE :
                                     INCOMPLETE_NONWRITE_REFINED_STATE;
            return update_refinement(child, ctx, current_mask);
          }
        case INCOMPLETE_NONWRITE_REFINED_STATE:
          {
            // See if we need to change states
            if (child->row_source->is_complete())
            {
              refinement_state = IS_WRITE(usage) ?
                                     COMPLETE_WRITE_REFINED_STATE :
                                     COMPLETE_NONWRITE_REFINED_STATE;
              return update_refinement(child, ctx, current_mask);
            }
            else if (IS_WRITE(usage))
            {
              refinement_state = INCOMPLETE_WRITE_REFINED_STATE;
              return update_refinement(child, ctx, current_mask);
            }
            break;
          }
        case INCOMPLETE_WRITE_REFINED_STATE:
          {
            if (child->row_source->is_complete())
            {
              refinement_state = IS_WRITE(usage) ?
                                     COMPLETE_WRITE_REFINED_STATE :
                                     COMPLETE_NONWRITE_REFINED_STATE;
              return update_refinement(child, ctx, current_mask);
            }
            else if (!IS_WRITE(usage))
              return false;
            break;
          }
        case COMPLETE_NONWRITE_REFINED_STATE:
          {
            if (!child->row_source->is_complete())
              return false;
            if (IS_WRITE(usage))
            {
              refinement_state = COMPLETE_WRITE_REFINED_STATE;
              return update_refinement(child, ctx, current_mask);
            }
            break;
          }
        case COMPLETE_WRITE_REFINED_STATE:
          {
            if (!IS_WRITE(usage))
              return false;
            if (!child->row_source->is_complete())
              return false;
            break;
          }
        case NO_REFINEMENT_STATE:
          return false;
        default:
          std::abort();
      }
      // If we get here then we haven't changed state so check to
      // see if this partition makes us want to change state
      // Step the clock for a new traversal
      total_traversals++;
      // Check to see if we've observed this refinement before
      std::unordered_map<PartitionNode*, std::pair<double, uint64_t> >::iterator
          finder = candidate_partitions.find(child);
      if (finder != candidate_partitions.end())
      {
        // Update the score using exponentially weighted moving average
        finder->second.first = std::pow(
                                   CHANGE_REFINEMENT_RETURN_WEIGHT,
                                   (total_traversals - finder->second.second)) *
                                   finder->second.first +
                               1.0;
        // Since CHANGE_REFINEMENT_RETURN_COUNT is a power of 2 the
        // compiler should be able to optimize integer division to a
        // basic bit shift
        const uint64_t previous_epoch =
            finder->second.second / CHANGE_REFINEMENT_RETURN_COUNT;
        const uint64_t current_epoch =
            total_traversals / CHANGE_REFINEMENT_RETURN_COUNT;
        finder->second.second = total_traversals;
        // Reset the timeout since we saw a return
        return_timeout = 0;
        // If the last time we saw this child was in a previous epoch
        // then we can check to see if this is now the dominant child
        if ((previous_epoch != current_epoch) &&
            is_dominant_candidate(
                finder->second.first, (child == refined_child)))
        {
          // If we're current refinement we don't switch but just end
          // invalidating all the other candidates so they can start
          // again
          if (child != refined_child)
            return update_refinement(child, ctx, current_mask);
          else
            invalidate_unused_candidates();
        }
      }
      else if (child == refined_child)
      {
        // This counts as a return too since we're refined this way
        candidate_partitions[child] =
            std::pair<double, uint64_t>(1.0, total_traversals);
        child->add_base_resource_ref(REFINEMENT_REF);
        // Reset the timeout since we saw a return
        return_timeout = 0;
        // No need to switch, we're already the refinement
      }
      else
      {
        if (++return_timeout == CHANGE_REFINEMENT_TIMEOUT)
        {
          invalidate_unused_candidates();
          return_timeout = 0;
        }
        child->add_base_resource_ref(REFINEMENT_REF);
        candidate_partitions[child] =
            std::pair<double, uint64_t>(0.0, total_traversals);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::update_projection(
        ProjectionSummary* summary, const RegionUsage& usage, ContextID ctx,
        const FieldMask& current_mask)
    //--------------------------------------------------------------------------
    {
      // If this is a projection with the identity projection on a region
      // then we don't want to consider this like a projection and instead
      // want to treat it like we're using this region directly
      if (summary->projection->projection_id == 0)
        return false;
      switch (refinement_state)
      {
        case UNREFINED_STATE:
          {
            legion_assert(refined_child == nullptr);
            legion_assert(refined_projection == nullptr);
            if (summary->is_complete())
              refinement_state = IS_WRITE(usage) ?
                                     COMPLETE_WRITE_REFINED_STATE :
                                     COMPLETE_NONWRITE_REFINED_STATE;
            else
              refinement_state = IS_WRITE(usage) ?
                                     INCOMPLETE_WRITE_REFINED_STATE :
                                     INCOMPLETE_NONWRITE_REFINED_STATE;
            return update_refinement(summary, ctx, current_mask);
          }
        case INCOMPLETE_NONWRITE_REFINED_STATE:
          {
            // See if we need to change states
            if (summary->is_complete())
            {
              refinement_state = IS_WRITE(usage) ?
                                     COMPLETE_WRITE_REFINED_STATE :
                                     COMPLETE_NONWRITE_REFINED_STATE;
              return update_refinement(summary, ctx, current_mask);
            }
            else if (IS_WRITE(usage))
            {
              refinement_state = INCOMPLETE_WRITE_REFINED_STATE;
              return update_refinement(summary, ctx, current_mask);
            }
            break;
          }
        case INCOMPLETE_WRITE_REFINED_STATE:
          {
            if (summary->is_complete())
            {
              refinement_state = IS_WRITE(usage) ?
                                     COMPLETE_WRITE_REFINED_STATE :
                                     COMPLETE_NONWRITE_REFINED_STATE;
              return update_refinement(summary, ctx, current_mask);
            }
            else if (!IS_WRITE(usage))
              return false;
            break;
          }
        case COMPLETE_NONWRITE_REFINED_STATE:
          {
            if (!summary->is_complete())
              return false;
            if (IS_WRITE(usage))
            {
              refinement_state = COMPLETE_WRITE_REFINED_STATE;
              return update_refinement(summary, ctx, current_mask);
            }
            break;
          }
        case COMPLETE_WRITE_REFINED_STATE:
          {
            if (!IS_WRITE(usage))
              return false;
            if (!summary->is_complete())
              return false;
            break;
          }
        case NO_REFINEMENT_STATE:
          return false;
        default:
          std::abort();
      }
      ProjectionRegion* projection =
          summary->get_tree()->as_region_projection();
      // Step the clock for a new traversal
      total_traversals++;
      // Check to see if we observed this refinement before
      std::unordered_map<
          ProjectionRegion*, std::pair<double, uint64_t> >::iterator finder =
          candidate_projections.find(projection);
      if (finder != candidate_projections.end())
      {
        // Update the score using exponentially weighted moving average
        finder->second.first = std::pow(
                                   CHANGE_REFINEMENT_RETURN_WEIGHT,
                                   (total_traversals - finder->second.second)) *
                                   finder->second.first +
                               1.0;
        const uint64_t previous_epoch =
            finder->second.second / CHANGE_REFINEMENT_RETURN_COUNT;
        const uint64_t current_epoch =
            total_traversals / CHANGE_REFINEMENT_RETURN_COUNT;
        finder->second.second = total_traversals;
        // Reset the timeout since we saw a return
        return_timeout = 0;
        // If the last time we saw this projection was in a previous
        // epoch then we can check to see if this is now the dominant
        // child
        if ((previous_epoch != current_epoch) &&
            is_dominant_candidate(
                finder->second.first, (projection == refined_projection)))
        {
          // If we're current refinement we don't switch but just end
          // invalidating all the other candidates so they can start
          // again
          if (projection != refined_projection)
            return update_refinement(summary, ctx, current_mask);
          else
            invalidate_unused_candidates();
        }
      }
      else if (projection == refined_projection)
      {
        // This counts as a return too since we're refined this way
        candidate_projections[projection] =
            std::pair<double, uint64_t>(1.0, total_traversals);
        projection->add_reference();
        // Reset the timeout since we saw a return
        return_timeout = 0;
        // No need to switch, we're already using the refinement
      }
      else
      {
        if (++return_timeout == CHANGE_REFINEMENT_TIMEOUT)
        {
          invalidate_unused_candidates();
          return_timeout = 0;
        }
        candidate_projections[projection] =
            std::pair<double, uint64_t>(0.0, total_traversals);
        projection->add_reference();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void RegionRefinementTracker::invalidate_refinement(
        ContextID ctx, const FieldMask& invalidation_mask)
    //--------------------------------------------------------------------------
    {
      if (refined_child != nullptr)
        refined_child->invalidate_logical_refinement(ctx, invalidation_mask);
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::merge_refinement(
        const RefinementTracker* rhs) const
    //--------------------------------------------------------------------------
    {
      const RegionRefinementTracker* other =
          legion_safe_cast<const RegionRefinementTracker*>(rhs);
      if (refinement_state != other->refinement_state)
        return false;
      if (refined_child != other->refined_child)
        return false;
      if (refined_projection != other->refined_projection)
        return false;
      // Ignore the candidates, we don't actually care if they're the same
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::is_dominant_candidate(
        double score, bool is_current)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (refined_child != nullptr) || (refined_projection != nullptr));
      bool is_dominant = true;
      // Has to have the most returns
      for (std::unordered_map<
               PartitionNode*, std::pair<double, uint64_t> >::iterator it =
               candidate_partitions.begin();
           it != candidate_partitions.end(); it++)
      {
        // Skip ourselves
        if (it->second.second == total_traversals)
          continue;
        // Recompute the score before comparing
        it->second.first = std::pow(
                               CHANGE_REFINEMENT_RETURN_WEIGHT,
                               (total_traversals - it->second.second)) *
                           it->second.first;
        it->second.second = total_traversals;
        if (score < it->second.first)
          is_dominant = false;
      }
      for (std::unordered_map<
               ProjectionRegion*, std::pair<double, uint64_t> >::iterator it =
               candidate_projections.begin();
           it != candidate_projections.end(); it++)
      {
        // Skip ourselves
        if (it->second.second == total_traversals)
          continue;
        // Recompute the score before comparing
        it->second.first = std::pow(
                               CHANGE_REFINEMENT_RETURN_WEIGHT,
                               (total_traversals - it->second.second)) *
                           it->second.first;
        it->second.second = total_traversals;
        if (score < it->second.first)
          is_dominant = false;
      }
      if (!is_dominant)
        return false;
      // If we're the current refinement then just being the largest is enough
      // to indicate that we're the dominant candidate
      if (!is_current)
      {
        // If we're not the current refinement, then we want to make sure its
        // obvious that we should switch and not just ping-pong so we need to
        // have a score that is at least sqrt(total_candidates) more than the
        // current refinement's score
        if (refined_child != nullptr)
        {
          std::unordered_map<PartitionNode*, std::pair<double, uint64_t> >::
              const_iterator finder = candidate_partitions.find(refined_child);
          // Current refinement is never observed
          if (finder == candidate_partitions.end())
            return true;
          double total_candidates =
              candidate_partitions.size() + candidate_projections.size();
          double hysteresis = std::sqrt(total_candidates);
          return ((finder->second.first * hysteresis) < score);
        }
        else
        {
          legion_assert(refined_projection != nullptr);
          std::unordered_map<ProjectionRegion*, std::pair<double, uint64_t> >::
              const_iterator finder =
                  candidate_projections.find(refined_projection);
          // Current refinement is never observed
          if (finder == candidate_projections.end())
            return true;
          double total_candidates =
              candidate_partitions.size() + candidate_projections.size();
          double hysteresis = std::sqrt(total_candidates);
          return ((finder->second.first * hysteresis) < score);
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::update_refinement(
        PartitionNode* child, ContextID ctx, const FieldMask& current_mask)
    //--------------------------------------------------------------------------
    {
      if (refined_projection != nullptr)
      {
        if (refined_projection->remove_reference())
          delete refined_projection;
        refined_projection = nullptr;
      }
      const bool changed = (child != refined_child);
      if (changed)
      {
        if (refined_child != nullptr)
        {
          refined_child->invalidate_logical_refinement(ctx, current_mask);
          if (refined_child->remove_base_resource_ref(REFINEMENT_REF))
            delete refined_child;
        }
        refined_child = child;
        refined_child->add_base_resource_ref(REFINEMENT_REF);
      }
      invalidate_unused_candidates();
      return changed;
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::update_refinement(
        ProjectionSummary* summary, ContextID ctx,
        const FieldMask& current_mask)
    //--------------------------------------------------------------------------
    {
      ProjectionRegion* projection =
          summary->get_tree()->as_region_projection();
      if (refined_child != nullptr)
      {
        refined_child->invalidate_logical_refinement(ctx, current_mask);
        if (refined_child->remove_base_resource_ref(REFINEMENT_REF))
          delete refined_child;
        refined_child = nullptr;
      }
      const bool changed = (projection != refined_projection);
      if (changed)
      {
        if ((refined_projection != nullptr) &&
            refined_projection->remove_reference())
          delete refined_projection;
        refined_projection = projection;
        refined_projection->add_reference();
      }
      invalidate_unused_candidates();
      return changed;
    }

    //--------------------------------------------------------------------------
    void RegionRefinementTracker::invalidate_unused_candidates(void)
    //--------------------------------------------------------------------------
    {
      // Remove any candidates that have gone too long without being observed
      if (!candidate_partitions.empty())
      {
        for (std::unordered_map<
                 PartitionNode*, std::pair<double, uint64_t> >::iterator it =
                 candidate_partitions.begin();
             it != candidate_partitions.end();
             /*nothing*/)
        {
          uint64_t last_observation = total_traversals - it->second.second;
          if (CHANGE_REFINEMENT_TIMEOUT < last_observation)
          {
            if (it->first->remove_base_resource_ref(REFINEMENT_REF))
              delete it->first;
            std::unordered_map<PartitionNode*, std::pair<double, uint64_t> >::
                iterator to_delete = it++;
            candidate_partitions.erase(to_delete);
          }
          else
            it++;
        }
      }
      if (!candidate_projections.empty())
      {
        for (std::unordered_map<
                 ProjectionRegion*, std::pair<double, uint64_t> >::iterator it =
                 candidate_projections.begin();
             it != candidate_projections.end();
             /*nothing*/)
        {
          uint64_t last_observation = total_traversals - it->second.second;
          if (CHANGE_REFINEMENT_TIMEOUT < last_observation)
          {
            if (it->first->remove_reference())
              delete it->first;
            std::unordered_map<
                ProjectionRegion*, std::pair<double, uint64_t> >::iterator
                to_delete = it++;
            candidate_projections.erase(to_delete);
          }
          else
            it++;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // PartitionRefinementTracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionRefinementTracker::PartitionRefinementTracker(PartitionNode* node)
      : partition(node), refined_projection(nullptr), children_score(-1.0),
        children_last(0), total_traversals(0), return_timeout(0)
    //--------------------------------------------------------------------------
    {
      partition->add_base_resource_ref(REFINEMENT_REF);
    }

    //--------------------------------------------------------------------------
    PartitionRefinementTracker::~PartitionRefinementTracker(void)
    //--------------------------------------------------------------------------
    {
      if ((refined_projection != nullptr) &&
          refined_projection->remove_reference())
        delete refined_projection;
      for (std::vector<RegionNode*>::const_iterator it = children.begin();
           it != children.end(); it++)
        if ((*it)->remove_base_resource_ref(REFINEMENT_REF))
          delete (*it);
      for (std::unordered_map<
               ProjectionPartition*,
               std::pair<double, uint64_t> >::const_iterator it =
               candidate_projections.begin();
           it != candidate_projections.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      if (partition->remove_base_resource_ref(REFINEMENT_REF))
        delete partition;
    }

    //--------------------------------------------------------------------------
    RefinementTracker* PartitionRefinementTracker::clone(void) const
    //--------------------------------------------------------------------------
    {
      PartitionRefinementTracker* tracker =
          new PartitionRefinementTracker(partition);
      if (refined_projection != nullptr)
      {
        tracker->refined_projection = refined_projection;
        refined_projection->add_reference();
      }
      if (!children.empty())
      {
        tracker->children = children;
        for (std::vector<RegionNode*>::const_iterator it = children.begin();
             it != children.end(); it++)
          (*it)->add_base_resource_ref(REFINEMENT_REF);
      }
      for (std::unordered_map<
               ProjectionPartition*,
               std::pair<double, uint64_t> >::const_iterator it =
               candidate_projections.begin();
           it != candidate_projections.end(); it++)
      {
        it->first->add_reference();
        tracker->candidate_projections.insert(*it);
      }
      tracker->children_score = children_score;
      tracker->children_last = children_last;
      tracker->total_traversals = total_traversals;
      tracker->return_timeout = return_timeout;
      return tracker;
    }

    //--------------------------------------------------------------------------
    void PartitionRefinementTracker::initialize_no_refine(void)
    //--------------------------------------------------------------------------
    {
      std::abort();  // this should never be called
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::is_child_current_refinement(
        RegionTreeNode* child) const
    //--------------------------------------------------------------------------
    {
      legion_assert(child->is_region());
      if (refined_projection != nullptr)
        return false;
      return std::binary_search(children.begin(), children.end(), child);
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::update_child(
        RegionTreeNode* node, const RegionUsage& usage, ContextID ctx,
        const FieldMask& current_mask)
    //--------------------------------------------------------------------------
    {
      // No refinement tracking through aliased partitions
      if (!partition->row_source->is_disjoint())
        return false;
      legion_assert(node->is_region());
      RegionNode* child = node->as_region_node();
      total_traversals++;
      if (children.empty())
      {
        child->add_base_resource_ref(REFINEMENT_REF);
        children.emplace_back(child);
        if (refined_projection == nullptr)
          // Children become the initial refinement
          return true;
        if (++return_timeout == CHANGE_REFINEMENT_TIMEOUT)
        {
          invalidate_unused_candidates();
          return_timeout = 0;
        }
        children_score = 0.0;
        children_last = total_traversals;
      }
      else
      {
        // Returned to the children so reset the timeout
        return_timeout = 0;
        if (!std::binary_search(children.begin(), children.end(), child))
        {
          // Add the child to the list of children
          child->add_base_resource_ref(REFINEMENT_REF);
          children.emplace_back(child);
          std::sort(children.begin(), children.end());
        }
        if (refined_projection == nullptr)
        {
          // Children are still the refinement
          children_score = 1.0;
          children_last = total_traversals;
        }
        else
        {
          // See if we want to change refinement to the children
          children_score = std::pow(
                               CHANGE_REFINEMENT_RETURN_WEIGHT,
                               (total_traversals - children_last)) *
                           children_score;
          // Check to see if we've changed epochs
          const uint64_t previous_epoch =
              children_last / CHANGE_REFINEMENT_RETURN_COUNT;
          const uint64_t current_epoch =
              total_traversals / CHANGE_REFINEMENT_RETURN_COUNT;
          children_last = total_traversals;
          if ((previous_epoch != current_epoch) &&
              is_dominant_candidate(children_score, false))
          {
            if (refined_projection->remove_reference())
              delete refined_projection;
            refined_projection = nullptr;
            invalidate_unused_candidates();
            return true;
          }
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::update_projection(
        ProjectionSummary* summary, const RegionUsage& usage, ContextID ctx,
        const FieldMask& current_mask)
    //--------------------------------------------------------------------------
    {
      ProjectionPartition* projection =
          summary->get_tree()->as_partition_projection();
      // Special case if we have no refinements
      if ((refined_projection == nullptr) && children.empty())
      {
        // We'll just become the refinement right away
        refined_projection = projection;
        refined_projection->add_reference();
        return true;
      }
      // Update the score and see if we want to change
      total_traversals++;
      // Check to see if we observed this refinement before
      std::unordered_map<
          ProjectionPartition*, std::pair<double, uint64_t> >::iterator finder =
          candidate_projections.find(projection);
      if (finder != candidate_projections.end())
      {
        // Update the score using exponentially weighted moving average
        finder->second.first = std::pow(
                                   CHANGE_REFINEMENT_RETURN_WEIGHT,
                                   (total_traversals - finder->second.second)) *
                                   finder->second.first +
                               1.0;
        const uint64_t previous_epoch =
            finder->second.second / CHANGE_REFINEMENT_RETURN_COUNT;
        const uint64_t current_epoch =
            total_traversals / CHANGE_REFINEMENT_RETURN_COUNT;
        finder->second.second = total_traversals;
        // Reset the timeout since we saw a return
        return_timeout = 0;
        // If the last time we saw this projection was in a previous
        // epoch then we can check to see if this is now the dominant
        // child
        if ((previous_epoch != current_epoch) &&
            is_dominant_candidate(
                finder->second.first, (projection == refined_projection)))
        {
          // We've found a new best refinement
          if (projection != refined_projection)
          {
            if (refined_projection == nullptr)
            {
              // Invalidate the refined children
              for (std::vector<RegionNode*>::const_iterator it =
                       children.begin();
                   it != children.end(); it++)
              {
                (*it)->invalidate_logical_refinement(ctx, current_mask);
                if ((*it)->remove_base_resource_ref(REFINEMENT_REF))
                  delete (*it);
              }
              children.clear();
            }
            else if (refined_projection->remove_reference())
              delete refined_projection;
            refined_projection = projection;
            refined_projection->add_reference();
            invalidate_unused_candidates();
            return true;
          }
          invalidate_unused_candidates();
        }
      }
      else if (projection == refined_projection)
      {
        // This counts as a return too since we're refined this way
        candidate_projections[projection] =
            std::pair<double, uint64_t>(1.0, total_traversals);
        projection->add_reference();
        // Reset the timeout since we saw a return
        return_timeout = 0;
        // No need to switch, we're already using the refinement
      }
      else
      {
        if (++return_timeout == CHANGE_REFINEMENT_TIMEOUT)
        {
          invalidate_unused_candidates();
          return_timeout = 0;
        }
        candidate_projections[projection] =
            std::pair<double, uint64_t>(0.0, total_traversals);
        projection->add_reference();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void PartitionRefinementTracker::invalidate_refinement(
        ContextID ctx, const FieldMask& invalidation_mask)
    //--------------------------------------------------------------------------
    {
      for (std::vector<RegionNode*>::const_iterator it = children.begin();
           it != children.end(); it++)
        (*it)->invalidate_logical_refinement(ctx, invalidation_mask);
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::merge_refinement(
        const RefinementTracker* rhs) const
    //--------------------------------------------------------------------------
    {
      const PartitionRefinementTracker* other =
          legion_safe_cast<const PartitionRefinementTracker*>(rhs);
      if (refined_projection != other->refined_projection)
        return false;
      if (children.size() != other->children.size())
        return false;
      for (unsigned idx1 = 0; idx1 < children.size(); idx1++)
      {
        bool found = false;
        for (unsigned idx2 = 0; idx2 < other->children.size(); idx2++)
        {
          if (children[idx1] != other->children[idx2])
            continue;
          found = true;
          break;
        }
        if (!found)
          return false;
      }
      // Ignore the candidates, we don't actually care if they're the same
      return true;
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::is_dominant_candidate(
        double score, bool is_current)
    //--------------------------------------------------------------------------
    {
      bool is_dominant = true;
      if (children_last != total_traversals)
      {
        // Recompute the children score and compare it
        children_score = std::pow(
                             CHANGE_REFINEMENT_RETURN_WEIGHT,
                             (total_traversals - children_last)) *
                         children_score;
        children_last = total_traversals;
        if (score < children_score)
          is_dominant = false;
      }
      for (std::unordered_map<
               ProjectionPartition*, std::pair<double, uint64_t> >::iterator
               it = candidate_projections.begin();
           it != candidate_projections.end(); it++)
      {
        // Skip ourselves
        if (it->second.second == total_traversals)
          continue;
        // Recompute the score before comparing
        it->second.first = std::pow(
                               CHANGE_REFINEMENT_RETURN_WEIGHT,
                               (total_traversals - it->second.second)) *
                           it->second.first;
        it->second.second = total_traversals;
        if (score < it->second.first)
          is_dominant = false;
      }
      if (!is_dominant)
        return false;
      if (!is_current)
      {
        if (refined_projection != nullptr)
        {
          // If we're not the current refinement, then we want to make sure its
          // obvious that we should switch and not just ping-pong so we need to
          // have a score that is at least sqrt(total_candidates) more than the
          // current refinement's score
          std::unordered_map<
              ProjectionPartition*,
              std::pair<double, uint64_t> >::const_iterator finder =
              candidate_projections.find(refined_projection);
          // Current refinement is never observed
          if (finder == candidate_projections.end())
            return true;
          double total_candidates =
              (children_score >= 0.0) ? 1 : 0 + candidate_projections.size();
          double hysteresis = std::sqrt(total_candidates);
          return ((finder->second.first * hysteresis) < score);
        }
        else
        {
          // The children are refined so compute the total candidates
          if (children_score < 0.0)
            return true;
          double total_candidates = candidate_projections.size() + 1;
          double hysteresis = std::sqrt(total_candidates);
          return ((children_score * hysteresis) < score);
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PartitionRefinementTracker::invalidate_unused_candidates(void)
    //--------------------------------------------------------------------------
    {
      if (refined_projection != nullptr)
      {
        for (std::vector<RegionNode*>::const_iterator it = children.begin();
             it != children.end(); it++)
          if ((*it)->remove_base_resource_ref(REFINEMENT_REF))
            delete (*it);
        children.clear();
      }
      if (!candidate_projections.empty())
      {
        for (std::unordered_map<
                 ProjectionPartition*, std::pair<double, uint64_t> >::iterator
                 it = candidate_projections.begin();
             it != candidate_projections.end();
             /*nothing*/)
        {
          uint64_t last_observation = total_traversals - it->second.second;
          if (CHANGE_REFINEMENT_TIMEOUT < last_observation)
          {
            if (it->first->remove_reference())
              delete it->first;
            std::unordered_map<
                ProjectionPartition*, std::pair<double, uint64_t> >::iterator
                to_delete = it++;
            candidate_projections.erase(to_delete);
          }
          else
            it++;
        }
      }
    }

  }  // namespace Internal
}  // namespace Legion
