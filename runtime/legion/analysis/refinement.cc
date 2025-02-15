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
#include "legion/interface/functors_impl.h"
#include "legion/nodes/region.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // RegionRefinementTracker
    /////////////////////////////////////////////////////////////

    /*static*/ constexpr uint64_t RefinementTracker::MAX_INCOMPLETE_WRITES;

    //--------------------------------------------------------------------------
    RegionRefinementTracker::RegionRefinementTracker(RegionNode *node)
      : region(node), refinement_state(UNREFINED_STATE), refined_child(NULL),
        refined_projection(NULL), total_traversals(0), return_timeout(0)
    //--------------------------------------------------------------------------
    {
      region->add_base_resource_ref(REFINEMENT_REF);
    }

    //--------------------------------------------------------------------------
    RegionRefinementTracker::~RegionRefinementTracker(void)
    //--------------------------------------------------------------------------
    {
      if ((refined_child != NULL) && 
          refined_child->remove_base_resource_ref(REFINEMENT_REF))
        delete refined_child;
      if ((refined_projection != NULL) && 
            refined_projection->remove_reference())
        delete refined_projection;
      for (std::unordered_map<PartitionNode*,
                std::pair<double,uint64_t> >::const_iterator it =
            candidate_partitions.begin(); it != 
            candidate_partitions.end(); it++)
        if (it->first->remove_base_resource_ref(REFINEMENT_REF))
          delete it->first;
      for (std::unordered_map<ProjectionRegion*,
                std::pair<double,uint64_t> >::const_iterator it =
            candidate_projections.begin(); it != 
            candidate_projections.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      if (region->remove_base_resource_ref(REFINEMENT_REF))
        delete region;
    }

    //--------------------------------------------------------------------------
    RefinementTracker* RegionRefinementTracker::clone(void) const
    //--------------------------------------------------------------------------
    {
      RegionRefinementTracker *tracker = new RegionRefinementTracker(region);
      tracker->refinement_state = refinement_state;
      if (refined_child != NULL)
      {
        tracker->refined_child = refined_child;
        refined_child->add_base_resource_ref(REFINEMENT_REF);
      }
      if (refined_projection != NULL)
      {
        tracker->refined_projection = refined_projection;
        refined_projection->add_reference();
      }
      for (std::unordered_map<PartitionNode*,
                std::pair<double,uint64_t> >::const_iterator it =
            candidate_partitions.begin(); it != 
            candidate_partitions.end(); it++)
      {
        it->first->add_base_resource_ref(REFINEMENT_REF);
        tracker->candidate_partitions.insert(*it);
      }
      for (std::unordered_map<ProjectionRegion*,
                std::pair<double,uint64_t> >::const_iterator it =
            candidate_projections.begin(); it != 
            candidate_projections.end(); it++)
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
#ifdef DEBUG_LEGION
      assert(refined_child == NULL);
      assert(refinement_state == UNREFINED_STATE);
#endif
      refinement_state = NO_REFINEMENT_STATE;
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::update_child(RegionTreeNode *node,
                               const RegionUsage &usage, bool &allow_refinement)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!allow_refinement);
      assert(!node->is_region());
#endif
      PartitionNode *child = node->as_partition_node();
      switch (refinement_state)
      {
        case UNREFINED_STATE:
          {
#ifdef DEBUG_LEGION
            assert(refined_child == NULL);
            assert(refined_projection == NULL);
#endif
            // If we don't have any refinements, we'll always allow them
            if (child->row_source->is_complete())
              refinement_state = IS_WRITE(usage) ? 
                COMPLETE_WRITE_REFINED_STATE : COMPLETE_NONWRITE_REFINED_STATE;
            else
              refinement_state = IS_WRITE(usage) ?
                INCOMPLETE_WRITE_REFINED_STATE : 
                INCOMPLETE_NONWRITE_REFINED_STATE;
            child->add_base_resource_ref(REFINEMENT_REF);
            refined_child = child;
            break;
          }
        case COMPLETE_NONWRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Switch over to write refinements as soon as we see one
              // We can delete this tracker and make a new one to get
              // any kind of field coalescing happening
              return true;
            }
            // Don't bother refining for other kinds of partitions 
            break;
          }
        case INCOMPLETE_NONWRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Switch over to write refinements as soon as we see one
              // We can delete this tracker and make a new one to get
              // any kind of field coalescing happening
              return true;
            }
            if (child->row_source->is_complete())
            {
              // Swith over to non-write complete
              // We can delete this tracker and make a new one to
              // get any kind of field coalescing happening
              return true;
            }
            break;
          }
        case COMPLETE_WRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage) && child->row_source->is_complete())
            {
              // Comparing complete writes against each other
              // Step the clock for a new traversal
              total_traversals++;
              // Check to see if we've observed this refinement before
              std::unordered_map<PartitionNode*,
                std::pair<double,uint64_t> >::iterator finder =
                  candidate_partitions.find(child);
              if (finder != candidate_partitions.end())
              {
                // Update the score using exponentially weighted moving average
                finder->second.first = std::pow(CHANGE_REFINEMENT_RETURN_WEIGHT,
                    (total_traversals - finder->second.second)) *
                      finder->second.first + 1.0;
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
                    is_dominant_candidate(finder->second.first,
                                          (child == refined_child)))
                {
                  // If we're current refinement we don't switch but just end
                  // invalidating all the other candidates so they can start again
                  if (child == refined_child)
                    invalidate_unused_candidates();
                  else
                    return true;
                }
              }
              else if (child == refined_child)
              {
                // This counts as a return too since we're refined this way
                candidate_partitions[child] = 
                  std::pair<double,uint64_t>(1.0, total_traversals);
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
                  std::pair<double,uint64_t>(0.0, total_traversals);
              }
            }
            break;
          }
        case INCOMPLETE_WRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              if (child->row_source->is_complete())
              {
                // Swith over to write complete
                // We can delete this tracker and make a new one to
                // get any kind of field coalescing happening
                return true;
              }
              else
              {
                // Always allow refinements down writing incomplete children
                allow_refinement = true;
                // Don't both deleting this, we can stay in this mode as long
                // as there are more incomplete writes
                return false;
              }
            }
            break;
          }
        case NO_REFINEMENT_STATE:
          // Don't allow anything to happen here
          break;
        default:
          assert(false); // should never hit this
      }
      // Allow alternative refinements of our current child
      allow_refinement = (child == refined_child);
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::update_projection(ProjectionSummary *summary,
                               const RegionUsage &usage, bool &allow_refinement)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!allow_refinement);
#endif
      // If this is a projection with the identity projection on a region
      // then we don't want to consider this like a projection and instead
      // want to treat it like we're using this region directly
      if (summary->projection->projection_id == 0)
        return update_arrival(usage);
      switch (refinement_state)
      {
        case UNREFINED_STATE:
          {
#ifdef DEBUG_LEGION
            assert(refined_child == NULL);
            assert(refined_projection == NULL);
#endif
            // If we don't have any refinements, we'll always allow them
            allow_refinement = true;
            if (summary->is_complete())
              refinement_state = IS_WRITE(usage) ? 
                COMPLETE_WRITE_REFINED_STATE : COMPLETE_NONWRITE_REFINED_STATE;
            else
              refinement_state = IS_WRITE(usage) ?
                INCOMPLETE_WRITE_REFINED_STATE : 
                INCOMPLETE_NONWRITE_REFINED_STATE;
            refined_projection = summary->get_tree()->as_region_projection();
            refined_projection->add_reference();
            break;
          }
        case COMPLETE_NONWRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Switch over to write refinements as soon as we see one
              // We can delete this tracker and make a new one to get
              // any kind of field coalescing happening
              return true;
            }
            // Don't bother refining for other kinds of partitions 
            break;
          }
        case INCOMPLETE_NONWRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Switch over to write refinements as soon as we see one
              // We can delete this tracker and make a new one to get
              // any kind of field coalescing happening
              return true;
            }
            if (summary->is_complete())
            {
              // Swith over to non-write complete
              // We can delete this tracker and make a new one to
              // get any kind of field coalescing happening
              return true;
            }
            break;
          }
        case COMPLETE_WRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage) && summary->is_complete())
            {
              ProjectionRegion *projection = 
                summary->get_tree()->as_region_projection();
              // Step the clock for a new traversal
              total_traversals++;
              // Check to see if we observed this refinement before
              std::unordered_map<ProjectionRegion*,
                std::pair<double,uint64_t> >::iterator finder =
                  candidate_projections.find(projection);
              if (finder != candidate_projections.end())
              {
                // Update the score using exponentially weighted moving average
                finder->second.first = std::pow(CHANGE_REFINEMENT_RETURN_WEIGHT,
                    (total_traversals - finder->second.second)) *
                  finder->second.first + 1.0;
                const uint64_t previous_epoch = 
                  finder->second.second / CHANGE_REFINEMENT_RETURN_COUNT;
                const uint64_t current_epoch = 
                  total_traversals / CHANGE_REFINEMENT_RETURN_COUNT;
                finder->second.second = total_traversals;
                // Reset the timeout since we saw a return
                return_timeout = 0;
                // If the last time we saw this projection was in a previous epoch 
                // then we can check to see if this is now the dominant child
                if ((previous_epoch != current_epoch) &&
                    is_dominant_candidate(finder->second.first, 
                                          (projection == refined_projection)))
                {
                  // If we're current refinement we don't switch but just end
                  // invalidating all the other candidates so they can start again
                  if (projection == refined_projection)
                    invalidate_unused_candidates();
                  else
                    return true;
                }
              }
              else if (projection == refined_projection)
              {
                // This counts as a return too since we're refined this way
                candidate_projections[projection] =
                  std::pair<double,uint64_t>(1.0, total_traversals);
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
                  std::pair<double,uint64_t>(0.0, total_traversals);
                projection->add_reference();
              }
            }
            break;
          }
        case INCOMPLETE_WRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              if (summary->is_complete())
              {
                // If this is a complete write then we're going to switch to it
                return true;
              }
              // Otherwise we're only going to do a refinement for this 
              // projection if it is the "first" time we've seen it. In 
              // practice it's unreasonable to keep a memory of all prior
              // disjoint and complete partitions, so we track a set of 
              // the most recent MAX_INCOMPLETE_WRITES
              ProjectionRegion *projection = 
                summary->get_tree()->as_region_projection();
              std::unordered_map<ProjectionRegion*,
                std::pair<double,uint64_t> >::iterator finder =
                  candidate_projections.find(projection);
              if (finder == candidate_projections.end())
              {
                // Didn't find it so decrement all the prior entries TTLs 
                for (std::unordered_map<ProjectionRegion*,
                      std::pair<double,uint64_t> >::iterator it =
                      candidate_projections.begin(); it != 
                      candidate_projections.end(); /*nothing*/)
                {
                  if (--it->second.second == 0)
                  {
                    std::unordered_map<ProjectionRegion*,
                      std::pair<double,uint64_t> >::iterator delete_it = it++;
                    candidate_projections.erase(delete_it);
                  }
                  else
                    it++;
                }
                projection->add_reference();
                candidate_projections.emplace(std::make_pair(projection,
                  std::pair<double,uint64_t>(0.0, MAX_INCOMPLETE_WRITES)));
                allow_refinement = true;
              }
              else
              {
                // We already had it, so decrement all TTLs of everything that
                // came before it and then add it back with a new TTL
                for (std::unordered_map<ProjectionRegion*,
                      std::pair<double,uint64_t> >::iterator it =
                      candidate_projections.begin(); it != 
                      candidate_projections.end(); it++)
                  if (finder->second.second < it->second.second)
                    it->second.second--;
                finder->second.second = MAX_INCOMPLETE_WRITES;
              }
            }
            break;
          }
        case NO_REFINEMENT_STATE:
          // Don't allow anything to happen here
          break;
        default:
          assert(false); // should never hit this
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::update_arrival(const RegionUsage &usage)
    //--------------------------------------------------------------------------
    {
      // We don't change the state because of an arrival right now
      return false;
    }

    //--------------------------------------------------------------------------
    void RegionRefinementTracker::invalidate_refinement(ContextID ctx,
                                             const FieldMask &invalidation_mask)
    //--------------------------------------------------------------------------
    {
      if (refined_child != NULL)
        refined_child->invalidate_logical_refinement(ctx, invalidation_mask);
    }

    //--------------------------------------------------------------------------
    bool RegionRefinementTracker::is_dominant_candidate(double score, 
                                                        bool is_current)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((refined_child != NULL) || (refined_projection != NULL));
#endif
      bool is_dominant = true;
      // Has to have the most returns
      for (std::unordered_map<PartitionNode*,
                              std::pair<double,uint64_t> >::iterator it =
            candidate_partitions.begin(); it !=
            candidate_partitions.end(); it++)
      {
        // Skip ourselves
        if (it->second.second == total_traversals)
          continue;
        // Recompute the score before comparing
        it->second.first = std::pow(CHANGE_REFINEMENT_RETURN_WEIGHT,
            (total_traversals - it->second.second)) * it->second.first;
        it->second.second = total_traversals;
        if (score < it->second.first)
          is_dominant = false;
      }
      for (std::unordered_map<ProjectionRegion*,
                              std::pair<double,uint64_t> >::iterator it =
            candidate_projections.begin(); it !=
            candidate_projections.end(); it++)
      {
        // Skip ourselves
        if (it->second.second == total_traversals)
          continue;
        // Recompute the score before comparing
        it->second.first = std::pow(CHANGE_REFINEMENT_RETURN_WEIGHT,
            (total_traversals - it->second.second)) * it->second.first;
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
        if (refined_child != NULL)
        {
          std::unordered_map<PartitionNode*,
            std::pair<double,uint64_t> >::const_iterator finder =
              candidate_partitions.find(refined_child);
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
#ifdef DEBUG_LEGION
          assert(refined_projection != NULL);
#endif
          std::unordered_map<ProjectionRegion*,
            std::pair<double,uint64_t> >::const_iterator
              finder = candidate_projections.find(refined_projection);
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
    void RegionRefinementTracker::invalidate_unused_candidates(void)
    //--------------------------------------------------------------------------
    {
      // Remove any candidates that have gone too long without being observed
      if (!candidate_partitions.empty())
      {
        for (std::unordered_map<PartitionNode*,
                            std::pair<double,uint64_t> >::iterator it =
              candidate_partitions.begin(); it !=
              candidate_partitions.end(); /*nothing*/)
        {
          uint64_t last_observation = total_traversals - it->second.second;
          if (CHANGE_REFINEMENT_TIMEOUT < last_observation)
          {
            if (it->first->remove_base_resource_ref(REFINEMENT_REF))
              delete it->first;
            std::unordered_map<PartitionNode*,
              std::pair<double,uint64_t> >::iterator to_delete = it++;
            candidate_partitions.erase(to_delete);
          }
          else
            it++;
        }
      }
      if (!candidate_projections.empty())
      {
        for (std::unordered_map<ProjectionRegion*,
                            std::pair<double,uint64_t> >::iterator it =
              candidate_projections.begin(); it !=
              candidate_projections.end(); /*nothing*/)
        {
          uint64_t last_observation = total_traversals - it->second.second;
          if (CHANGE_REFINEMENT_TIMEOUT < last_observation)
          {
            if (it->first->remove_reference())
              delete it->first;
            std::unordered_map<ProjectionRegion*,
              std::pair<double,uint64_t> >::iterator to_delete = it++;
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
    PartitionRefinementTracker::PartitionRefinementTracker(PartitionNode *node)
      : partition(node), refined_projection(NULL),
        refinement_state(UNREFINED_STATE), children_score(-1.0),
        children_last(0), total_traversals(0), return_timeout(0)
    //--------------------------------------------------------------------------
    {
      partition->add_base_resource_ref(REFINEMENT_REF);
    }

    //--------------------------------------------------------------------------
    PartitionRefinementTracker::~PartitionRefinementTracker(void)
    //--------------------------------------------------------------------------
    {
      if ((refined_projection != NULL) && 
          refined_projection->remove_reference())
        delete refined_projection;
      for (std::vector<RegionNode*>::const_iterator it =
            children.begin(); it != children.end(); it++)
        if ((*it)->remove_base_resource_ref(REFINEMENT_REF))
          delete (*it);
      for (std::unordered_map<ProjectionPartition*,
                std::pair<double,uint64_t> >::const_iterator it =
            candidate_projections.begin(); it != 
            candidate_projections.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      if (partition->remove_base_resource_ref(REFINEMENT_REF))
        delete partition;
    }

    //--------------------------------------------------------------------------
    RefinementTracker* PartitionRefinementTracker::clone(void) const
    //--------------------------------------------------------------------------
    {
      PartitionRefinementTracker *tracker = 
        new PartitionRefinementTracker(partition);
      if (refined_projection != NULL)
      {
        tracker->refined_projection = refined_projection;
        refined_projection->add_reference();
      }
      tracker->refinement_state = refinement_state; 
      if (!children.empty())
      {
        tracker->children = children;
        for (std::vector<RegionNode*>::const_iterator it =
            children.begin(); it != children.end(); it++)
          (*it)->add_base_resource_ref(REFINEMENT_REF);
      }
      for (std::unordered_map<ProjectionPartition*,
                std::pair<double,uint64_t> >::const_iterator it =
            candidate_projections.begin(); it != 
            candidate_projections.end(); it++)
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
      assert(false); // this should never be called
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::update_child(RegionTreeNode *node,
                               const RegionUsage &usage, bool &allow_refinement)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node->is_region());
      assert(!allow_refinement);
#endif
      RegionNode *child = node->as_region_node();
      switch (refinement_state)
      {
        case UNREFINED_STATE:
          {
#ifdef DEBUG_LEGION
            assert(children.empty());
            assert(refined_projection == NULL);
#endif
            allow_refinement = true;
            child->add_base_resource_ref(REFINEMENT_REF);
            children.push_back(child);
            if (((uint64_t)partition->row_source->total_children) <= 
                CHANGE_REFINEMENT_PARTITION_FRACTION)
              refinement_state = IS_WRITE(usage) ?
                COMPLETE_WRITE_REFINED_STATE : COMPLETE_NONWRITE_REFINED_STATE;
            else
              refinement_state = IS_WRITE(usage) ?
                INCOMPLETE_WRITE_REFINED_STATE :
                INCOMPLETE_NONWRITE_REFINED_STATE;
            break;
          }
        case COMPLETE_NONWRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Switch over to write refinements as soon as we see one
              // We can delete this tracker and make a new one to get
              // any kind of field coalescing happening
              return true;
            }
            // Only track non-write children if we're complete because of
            // children and not because of a complete projection refinement
            if (refined_projection == NULL)
            {
              allow_refinement = true;
              if (!std::binary_search(children.begin(), children.end(), child))
              {
                child->add_base_resource_ref(REFINEMENT_REF);
                children.push_back(child);
                std::sort(children.begin(), children.end());
              }
            }
            break;
          }
        case INCOMPLETE_NONWRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Switch over to write refinements as soon as we see one
              // We can delete this tracker and make a new one to get
              // any kind of field coalescing happening
              return true;
            }
            allow_refinement = true;
            if (!std::binary_search(children.begin(), children.end(), child))
            {
              child->add_base_resource_ref(REFINEMENT_REF);
              children.push_back(child);
              std::sort(children.begin(), children.end());
              if (((uint64_t)partition->row_source->total_children) <= 
                  (children.size() * CHANGE_REFINEMENT_PARTITION_FRACTION))
              {
                refinement_state = COMPLETE_NONWRITE_REFINED_STATE;
                // Remove any refined projections that we've done
                if (refined_projection != NULL)
                {
                  if (refined_projection->remove_reference())
                    delete refined_projection;
                  refined_projection = NULL;
                }
              }
            }
            break;
          }
        case COMPLETE_WRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              allow_refinement = true;
              // Increment the count for total traversals
              children_last = ++total_traversals;
              if (!std::binary_search(children.begin(), children.end(), child))
              {
                child->add_base_resource_ref(REFINEMENT_REF);
                children.push_back(child);
                std::sort(children.begin(), children.end());
                // See if we dominate the projection at this point
                if ((refined_projection != NULL) && 
                    (((uint64_t)partition->row_source->total_children) <=
                     (children.size() * CHANGE_REFINEMENT_PARTITION_FRACTION)))
                {
                  if (refined_projection->remove_reference())
                    delete refined_projection;
                  refined_projection = NULL;
                }
              }
            }
            break;
          }
        case INCOMPLETE_WRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              allow_refinement = true;
              if (!std::binary_search(children.begin(), children.end(), child))
              {
                child->add_base_resource_ref(REFINEMENT_REF);
                children.push_back(child);
                std::sort(children.begin(), children.end());
                if (((uint64_t)partition->row_source->total_children) <=
                    (children.size() * CHANGE_REFINEMENT_PARTITION_FRACTION))
                {
                  refinement_state = COMPLETE_WRITE_REFINED_STATE;
                  // Remove any refined projections that we've done
                  if (refined_projection != NULL)
                  {
                    if (refined_projection->remove_reference())
                      delete refined_projection;
                    refined_projection = NULL;
                  }
                }
              }
            }
            break;
          }
        default:
          assert(false);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::update_projection(
                                                    ProjectionSummary *summary, 
                                                    const RegionUsage &usage, 
                                                    bool &allow_refinement)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!allow_refinement);
#endif
      switch (refinement_state)
      {
        case UNREFINED_STATE:
          {
#ifdef DEBUG_LEGION
            assert(children.empty());
            assert(refined_projection == NULL);
#endif
            allow_refinement = true;
            refined_projection = summary->get_tree()->as_partition_projection();
            refined_projection->add_reference();
            if (summary->is_complete())
              refinement_state = IS_WRITE(usage) ?
                COMPLETE_WRITE_REFINED_STATE : COMPLETE_NONWRITE_REFINED_STATE;
            else
              refinement_state = IS_WRITE(usage) ?
                INCOMPLETE_WRITE_REFINED_STATE :
                INCOMPLETE_NONWRITE_REFINED_STATE;
            break;
          }
        case COMPLETE_NONWRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Switch over to write refinements as soon as we see one
              // We can delete this tracker and make a new one to get
              // any kind of field coalescing happening
              return true;
            }
            // If we already have a non-write complete refinement
            // then there is nothing more for us to do here
            break;
          }
        case INCOMPLETE_NONWRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Switch over to write refinements as soon as we see one
              // We can delete this tracker and make a new one to get
              // any kind of field coalescing happening
              return true;
            }
            // Check to see if we're a complete projection
            if (summary->is_complete())
            {
              // We're a new complete non-write projection so 
              // swtich to that
              return true;
            }
            // Otherwise we don't bother switching
            break;
          }
        case COMPLETE_WRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage) && summary->is_complete())
            {
              // Testing against other writing-complete projections
              // and/or the group of direct children
              // Step the clock for a new traversal
              total_traversals++;
              ProjectionPartition *projection =
                summary->get_tree()->as_partition_projection();
              std::unordered_map<ProjectionPartition*,
                std::pair<double,uint64_t> >::iterator finder =
                  candidate_projections.find(projection);
              if (finder != candidate_projections.end())
              {
                // Update the score using exponentially weighted moving average
                finder->second.first = std::pow(CHANGE_REFINEMENT_RETURN_WEIGHT,
                    (total_traversals - finder->second.second)) *
                  finder->second.first + 1.0;
                // Since CHANGE_REFINEMENT_RETURN_COUNT is a power of 2 the compiler
                // should be able to optimize integer division to a basic bit shift
                const uint64_t previous_epoch = 
                  finder->second.second / CHANGE_REFINEMENT_RETURN_COUNT;
                const uint64_t current_epoch = 
                  total_traversals / CHANGE_REFINEMENT_RETURN_COUNT;
                finder->second.second = total_traversals;
                // Reset the timeout since we saw a return
                return_timeout = 0;
                // If the last time we saw this projection was in a previous epoch 
                // then we can check to see if this is now the dominant child
                if ((previous_epoch != current_epoch) &&
                    is_dominant_candidate(finder->second.first, 
                                          (projection == refined_projection)))
                {
                  // If we're current refinement we don't switch but just end
                  // invalidating all the other candidates so they can start again
                  if (projection == refined_projection)
                    invalidate_unused_candidates();
                  else
                    return true;
                }
              }
              else if (projection == refined_projection)
              {
                // This counts as a return too since we're refined this way
                candidate_projections[projection] = 
                  std::pair<double,uint64_t>(1.0, total_traversals);
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
                  std::pair<double,uint64_t>(0.0, total_traversals);
                projection->add_reference();
              }
            }
            break;
          }
        case INCOMPLETE_WRITE_REFINED_STATE:
          {
            if (IS_WRITE(usage))
            {
              // Check to see if we're a complete projection
              if (summary->is_complete())
              {
                // We're a new complete writing projection so 
                // swtich to that
                return true; 
              }
              // Otherwise we're only going to do a refinement for this 
              // projection if it is the "first" time we've seen it. In 
              // practice it's unreasonable to keep a memory of all prior
              // disjoint and complete partitions, so we track a set of 
              // the most recent MAX_INCOMPLETE_WRITES
              ProjectionPartition *projection = 
                summary->get_tree()->as_partition_projection();
              std::unordered_map<ProjectionPartition*,
                std::pair<double,uint64_t> >::iterator finder =
                  candidate_projections.find(projection);
              if (finder == candidate_projections.end())
              {
                // Didn't find it so decrement all the prior entries TTLs 
                for (std::unordered_map<ProjectionPartition*,
                      std::pair<double,uint64_t> >::iterator it =
                      candidate_projections.begin(); it != 
                      candidate_projections.end(); /*nothing*/)
                {
                  if (--it->second.second == 0)
                  {
                    std::unordered_map<ProjectionPartition*,
                      std::pair<double,uint64_t> >::iterator delete_it = it++;
                    candidate_projections.erase(delete_it);
                  }
                  else
                    it++;
                }
                projection->add_reference();
                candidate_projections.emplace(std::make_pair(projection,
                  std::pair<double,uint64_t>(0.0, MAX_INCOMPLETE_WRITES)));
                allow_refinement = true;
              }
              else
              {
                // We already had it, so decrement all TTLs of everything that
                // came before it and then add it back with a new TTL
                for (std::unordered_map<ProjectionPartition*,
                      std::pair<double,uint64_t> >::iterator it =
                      candidate_projections.begin(); it != 
                      candidate_projections.end(); it++)
                  if (finder->second.second < it->second.second)
                    it->second.second--;
                finder->second.second = MAX_INCOMPLETE_WRITES;
              }
            }
            break;
          }
        default:
          assert(false);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::update_arrival(const RegionUsage &usage)
    //--------------------------------------------------------------------------
    {
      assert(false); // should never arrive on a partition without a projection 
      return false;
    }

    //--------------------------------------------------------------------------
    void PartitionRefinementTracker::invalidate_refinement(ContextID ctx,
                                             const FieldMask &invalidation_mask)
    //--------------------------------------------------------------------------
    {
      for (std::vector<RegionNode*>::const_iterator it =
            children.begin(); it != children.end(); it++)
        (*it)->invalidate_logical_refinement(ctx, invalidation_mask);
    }

    //--------------------------------------------------------------------------
    bool PartitionRefinementTracker::is_dominant_candidate(double score,
                                                          bool is_current)
    //--------------------------------------------------------------------------
    {
      bool is_dominant = true;
      if (((uint64_t)partition->row_source->total_children) <=
          (children.size() * CHANGE_REFINEMENT_PARTITION_FRACTION))
      {
        // Recompute the children score and compare it
        children_score = std::pow(CHANGE_REFINEMENT_RETURN_WEIGHT,
            (total_traversals - children_last)) * children_score;
        children_last = total_traversals;
        if (score < children_score)
          is_dominant = false;
      }
      for (std::unordered_map<ProjectionPartition*,
                              std::pair<double,uint64_t> >::iterator it =
            candidate_projections.begin(); it !=
            candidate_projections.end(); it++) 
      {
        // Skip ourselves
        if (it->second.second == total_traversals)
          continue;
        // Recompute the score before comparing
        it->second.first = std::pow(CHANGE_REFINEMENT_RETURN_WEIGHT,
            (total_traversals - it->second.second)) * it->second.first;
        it->second.second = total_traversals;
        if (score < it->second.first)
          is_dominant = false;
      }
      if (!is_dominant)
        return false;
      if (!is_current)
      {
        if (refined_projection != NULL)
        {
          // If we're not the current refinement, then we want to make sure its
          // obvious that we should switch and not just ping-pong so we need to
          // have a score that is at least sqrt(total_candidates) more than the
          // current refinement's score
          std::unordered_map<ProjectionPartition*,
              std::pair<double,uint64_t> >::const_iterator 
                finder = candidate_projections.find(refined_projection);
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
      if (children_score >= 0.0)
      {
        uint64_t last_observation = total_traversals - children_last;
        if (CHANGE_REFINEMENT_TIMEOUT < last_observation)
          children_score = -1.0;
      }
      if (!candidate_projections.empty())
      {
        for (std::unordered_map<ProjectionPartition*,
                            std::pair<double,uint64_t> >::iterator it =
              candidate_projections.begin(); it !=
              candidate_projections.end(); /*nothing*/)
        {
          uint64_t last_observation = total_traversals - it->second.second;
          if (CHANGE_REFINEMENT_TIMEOUT < last_observation)
          {
            if (it->first->remove_reference())
              delete it->first;
            std::unordered_map<ProjectionPartition*,
              std::pair<double,uint64_t> >::iterator to_delete = it++;
            candidate_projections.erase(to_delete);
          }
          else
            it++;
        }
      }
    }

  } // namespace Internal
} // namespace Legion
