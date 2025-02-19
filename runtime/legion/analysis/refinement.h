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

#ifndef __LEGION_REFINEMENT_H__
#define __LEGION_REFINEMENT_H__

#include "legion/api/types.h"
#include "legion/utilities/bitmask.h"

namespace Legion {
  namespace Internal {

    /**
     * \class RefinementTracker
     * This class provides a generic interface for deciding when to
     * perform or change refinements of the region tree
     */
    class RefinementTracker {
    public:
      virtual ~RefinementTracker(void) { };
    public:
      virtual RegionRefinementTracker* as_region_tracker(void)
      {
        return nullptr;
      }
      virtual PartitionRefinementTracker* as_partition_tracker(void)
      {
        return nullptr;
      }
      virtual RefinementTracker* clone(void) const = 0;
      virtual void initialize_no_refine(void) = 0;
      virtual bool update_child(
          RegionTreeNode* child, const RegionUsage& usage,
          bool& allow_refinement) = 0;
      virtual bool update_projection(
          ProjectionSummary* summary, const RegionUsage& usage,
          bool& allow_refinement) = 0;
      virtual bool update_arrival(const RegionUsage& usage) = 0;
      virtual void invalidate_refinement(
          ContextID ctx, const FieldMask& invalidation_mask) = 0;
    public:
      // This is the number of return children or projections we need
      // to observe in total before we consider a change to a refinement
      static constexpr uint64_t CHANGE_REFINEMENT_RETURN_COUNT = 256;
      // Check that this is a power of 2 for fast integer division
      static_assert(
          (CHANGE_REFINEMENT_RETURN_COUNT &
           (CHANGE_REFINEMENT_RETURN_COUNT - 1)) == 0,
          "must be power of two");
      // This is the weight for how degrading scores over time in our
      // exponentially weighted moving average computation
      static constexpr double CHANGE_REFINEMENT_RETURN_WEIGHT = 0.99;
      // This is the timeout for refinements where we will clear out all
      // candidate refinements and reset the state to look again
      static constexpr uint64_t CHANGE_REFINEMENT_TIMEOUT = 4096;
      // The maximum number of incomplete projection writes that we're
      // willing to remember at any particular node in the tree
      static constexpr uint64_t MAX_INCOMPLETE_WRITES = 32;
    protected:
      enum RefinementState {
        UNREFINED_STATE,
        COMPLETE_NONWRITE_REFINED_STATE,
        INCOMPLETE_NONWRITE_REFINED_STATE,
        COMPLETE_WRITE_REFINED_STATE,
        INCOMPLETE_WRITE_REFINED_STATE,
        NO_REFINEMENT_STATE,
      };
    };

    /**
     * \class RegionRefinementTracker
     * This class tracks the refinements (both partitions and projections)
     * on a region node in the region tree.
     */
    class RegionRefinementTracker
      : public RefinementTracker,
        public Heapify<RegionRefinementTracker, CONTEXT_LIFETIME> {
    public:
      RegionRefinementTracker(RegionNode* node);
      RegionRefinementTracker(const RegionRefinementTracker& rhs) = delete;
      virtual ~RegionRefinementTracker(void);
    public:
      RegionRefinementTracker& operator=(const RegionRefinementTracker& rhs) =
          delete;
    public:
      virtual RegionRefinementTracker* as_region_tracker(void) { return this; }
      virtual RefinementTracker* clone(void) const;
      virtual void initialize_no_refine(void);
      virtual bool update_child(
          RegionTreeNode* child, const RegionUsage& usage,
          bool& allow_refinement);
      virtual bool update_projection(
          ProjectionSummary* summary, const RegionUsage& usage,
          bool& allow_refinement);
      virtual bool update_arrival(const RegionUsage& usage);
      virtual void invalidate_refinement(
          ContextID ctx, const FieldMask& invalidation_mask);
    protected:
      bool is_dominant_candidate(double score, bool is_current);
      void invalidate_unused_candidates(void);
    public:
      RegionNode* const region;
    protected:
      RefinementState refinement_state;
      PartitionNode* refined_child;
      ProjectionRegion* refined_projection;
    protected:
      // Track the candidate children and projections and how often
      // we have observed a return back to them
      // <current score,timestamp of last observed return>
      std::unordered_map<PartitionNode*, std::pair<double, uint64_t> >
          candidate_partitions;
      std::unordered_map<ProjectionRegion*, std::pair<double, uint64_t> >
          candidate_projections;
      // Monotonically increasing clock counting total number of traversals
      uint64_t total_traversals;
      // The timeout tracks how long we've gone without seeing a return
      // If we go for too long without seeing a return, we timeout and
      // clear out all the candidates so we can try again
      uint64_t return_timeout;
    };

    /**
     * \class PartitionRefinementTracker
     * This class tracks the refinements (both sub-regions and projections)
     * on a region node in the region tree.
     */
    class PartitionRefinementTracker
      : public RefinementTracker,
        public Heapify<PartitionRefinementTracker, CONTEXT_LIFETIME> {
    public:
      PartitionRefinementTracker(PartitionNode* node);
      PartitionRefinementTracker(const PartitionRefinementTracker& rhs) =
          delete;
      virtual ~PartitionRefinementTracker(void);
    public:
      PartitionRefinementTracker& operator=(
          const PartitionRefinementTracker& rhs) = delete;
    public:
      virtual PartitionRefinementTracker* as_partition_tracker(void)
      {
        return this;
      }
      virtual RefinementTracker* clone(void) const;
      virtual void initialize_no_refine(void);
      virtual bool update_child(
          RegionTreeNode* child, const RegionUsage& usage,
          bool& allow_refinement);
      virtual bool update_projection(
          ProjectionSummary* summary, const RegionUsage& usage,
          bool& allow_refinement);
      virtual bool update_arrival(const RegionUsage& usage);
      virtual void invalidate_refinement(
          ContextID ctx, const FieldMask& invalidation_mask);
    protected:
      bool is_dominant_candidate(double score, bool is_current);
      void invalidate_unused_candidates(void);
    public:
      PartitionNode* const partition;
    protected:
      ProjectionPartition* refined_projection;
      RefinementState refinement_state;
      // These are children which are disjoint and complete
      // Note we don't need to hold references on them as they are kept alive
      // by the reference we are holding on their partition
      std::vector<RegionNode*> children;
      std::unordered_map<ProjectionPartition*, std::pair<double, uint64_t> >
          candidate_projections;
      // The individual score and last traversals of all the children
      double children_score;
      uint64_t children_last;
      // Monotonically increasing clock counting total number of traversals
      uint64_t total_traversals;
      // The timeout tracks how long we've gone without seeing a return
      // If we go for too long without seeing a return, we timeout and
      // clear out all the candidates so we can try again
      uint64_t return_timeout;
      // What fraction of children need to observed before we condider
      // this partition as being disjoint and complete, 1 means all the
      // children need to be observed to be considered complete, 2 means
      // that half the children need to be observed before being complete,
      // 3 means a third need to be observed before being complete, etc
      static constexpr uint64_t CHANGE_REFINEMENT_PARTITION_FRACTION = 2;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REFINEMENT_H__
