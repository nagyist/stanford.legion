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

#include "legion/tasks/index.h"
#include "legion/contexts/replicate.h"
#include "legion/api/argument_map_impl.h"
#include "legion/api/functors_impl.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/nodes/index.h"
#include "legion/operations/mustepoch.h"
#include "legion/tasks/individual.h"
#include "legion/tasks/slice.h"
#include "legion/tracing/recognizer.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Index Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(void) : MultiTask()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexTask::~IndexTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexTask::activate(void)
    //--------------------------------------------------------------------------
    {
      MultiTask::activate();
      serdez_redop_fns = nullptr;
      total_points = 0;
      mapped_points = 0;
      completed_points = 0;
      committed_points = 0;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      copy_fill_priority = 0;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
    }

    //--------------------------------------------------------------------------
    void IndexTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      reduction_instance = nullptr;  // we don't own this so clear it
      for (std::map<Color, ConcurrentGroup>::iterator it =
               concurrent_groups.begin();
           it != concurrent_groups.end(); it++)
        if (it->second.task_barrier.exists())
          it->second.task_barrier.destroy_barrier();
      MultiTask::deactivate(false /*free*/);
      if (!origin_mapped_slices.empty())
      {
        for (std::vector<SliceTask*>::const_iterator it =
                 origin_mapped_slices.begin();
             it != origin_mapped_slices.end(); it++)
          (*it)->deactivate();
        origin_mapped_slices.clear();
      }
      if (!reduction_instances.empty())
      {
        for (std::vector<FutureInstance*>::const_iterator it =
                 reduction_instances.begin();
             it != reduction_instances.end(); it++)
          delete (*it);
        reduction_instances.clear();
      }
      serdez_redop_targets.clear();
      // Remove our reference to the reduction future
      reduction_future = Future();
      reduction_future_size.reset();
      map_applied_conditions.clear();
      output_preconditions.clear();
      commit_preconditions.clear();
      version_infos.clear();
      interfering_requirements.clear();
      point_requirements.clear();
#ifdef DEBUG_LEGION
      assert(pending_pointwise_dependences.empty());
#endif
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void IndexTask::validate_output_extents(
        unsigned index, const OutputRequirement& req,
        const OutputExtentMap& output_extents) const
    //--------------------------------------------------------------------------
    {
      size_t num_tasks = 0;
      if (sharding_space.exists())
        num_tasks = runtime->get_domain_volume(sharding_space);
      else
        num_tasks = launch_space->get_volume();

      if (output_extents.size() == num_tasks)
        return;

      REPORT_LEGION_ERROR(
          ERROR_INVALID_OUTPUT_REGION_PROJECTION,
          "A projection functor for every output requirement must be "
          "bijective, but projection functor %u for output requirement %u "
          "in task %s (UID: %lld) mapped more than one point in the launch "
          "domain to the same subregion.",
          req.projection, index, get_task_name(), get_unique_op_id());
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_output_extents(
        std::vector<OutputExtentMap>& output_extents)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(output_region_extents.size() == output_extents.size());
#endif
      {
        AutoLock o_lock(op_lock);
        for (unsigned idx = 0; idx < output_extents.size(); idx++)
        {
          OutputExtentMap& target = output_region_extents[idx];
          if (!target.empty())
          {
            // Merge the new extents in
            OutputExtentMap& extents = output_extents[idx];
            for (OutputExtentMap::const_iterator it = extents.begin();
                 it != extents.end(); it++)
            {
              if (target.find(it->first) != target.end())
              {
                const DomainPoint& color = it->first;
                const OutputRequirement& req = output_regions[idx];
                std::stringstream ss;
                ss << "(" << color[0];
                for (int dim = 1; dim < color.dim; ++dim)
                  ss << "," << color[dim];
                ss << ")";
                REPORT_LEGION_ERROR(
                    ERROR_INVALID_OUTPUT_REGION_PROJECTION,
                    "A projection functor for every output requirement must be "
                    "bijective, but projection functor %u for output "
                    "requirement "
                    "%u in task %s (UID: %lld) mapped more than one point "
                    "in the launch domain to the same subregion of color %s.",
                    req.projection, idx, get_task_name(), get_unique_op_id(),
                    ss.str().c_str());
              }
              target.insert(*it);
            }
          }
          else
            target.swap(output_extents[idx]);
        }
        // Now Check to see if we've received all the extents
        for (unsigned idx = 0; idx < output_region_extents.size(); idx++)
        {
          if (is_output_valid(idx))
            continue;
#ifdef DEBUG_LEGION
          assert(output_region_extents[idx].size() <= total_points);
#endif
          if (output_region_extents[idx].size() < total_points)
            return;
        }
      }
      // If we get here then we can finalize our output regions
      finalize_output_regions(true /*first invocation*/);
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_output_registered(RtEvent registered)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(registered.exists());
      assert(!output_regions.empty());
#endif
      // Record it in the set of output events and if we've seen all of them
      // then we can launch the meta-task to do the final regisration
      AutoLock o_lock(op_lock);
      output_preconditions.emplace_back(registered);
#ifdef DEBUG_LEGION
      assert(output_preconditions.size() <= total_points);
#endif
      if (output_preconditions.size() == total_points)
      {
        // Can only mark the EqKDTree ready once all the points are registered
        FinalizeOutputEqKDTreeArgs args(this);
        registered = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY,
            Runtime::merge_events(output_preconditions));
        commit_preconditions.insert(registered);
      }
    }

    //--------------------------------------------------------------------------
    Domain IndexTask::compute_global_output_ranges(
        IndexSpaceNode* parent, IndexPartNode* part,
        const OutputExtentMap& output_extents,
        const OutputExtentMap& local_extents)
    //--------------------------------------------------------------------------
    {
      // First, we collect all the extents of local outputs.
      // While doing this, we also check the alignment.
      Domain color_space = part->color_space->get_tight_domain();
#ifdef DEBUG_LEGION
      assert(color_space.dense());
#endif
      int32_t ndim = color_space.dim;
      DomainPoint color_extents = color_space.hi() - color_space.lo() + 1;

#ifdef DEBUG_LEGION
      // Check alignments between tiles
      for (OutputExtentMap::const_iterator it = output_extents.begin();
           it != output_extents.end(); ++it)
      {
        const DomainPoint& color = it->first;
        const DomainPoint& extent = it->second;

        for (int32_t dim = 0; dim < ndim; ++dim)
        {
          if (color[dim] == 0)
            continue;
          DomainPoint neighbor = color;
          --neighbor[dim];
          auto finder = output_extents.find(neighbor);
          assert(finder != output_extents.end());

          const DomainPoint& neighbor_extent = it->second;
          if (extent[dim] != neighbor_extent[dim])
          {
            std::stringstream ss;
            ss << "Point task " << color << " returned an output of extent "
               << extent[dim] << " for dimension " << dim
               << ", but an adjacent point task returned an output of extent "
               << neighbor_extent[dim] << ". "
               << "Please make sure the outputs from point tasks are aligned.";
            REPORT_LEGION_ERROR(
                ERROR_UNALIGNED_OUTPUT_REGION, "%s", ss.str().c_str());
          }
        }
      }
#endif

      // Initialize the vectors of extents with 0
      std::vector<std::vector<coord_t>> all_extents(ndim);
      for (int32_t dim = 0; dim < ndim; ++dim)
        all_extents[dim].resize(color_extents[dim] + 1, 0);

      // Populate the extent vectors
      for (OutputExtentMap::const_iterator it = output_extents.begin();
           it != output_extents.end(); ++it)
      {
        const DomainPoint& color = it->first;
        const DomainPoint& extent = it->second;
        for (int32_t dim = 0; dim < ndim; ++dim)
        {
          coord_t c = color[dim];
          coord_t ext = extent[dim];
          coord_t& to_update = all_extents[dim][c];
          // Ignore all zero extents when populating the extent vector
          if (to_update == 0 && ext > 0)
            to_update = ext;
        }
      }

      // Prefix sum the extents to get sub-ranges for each dimension
      for (int32_t dim = 0; dim < ndim; ++dim)
      {
        std::vector<coord_t>& extents = all_extents[dim];
        coord_t sum = 0;
        for (size_t idx = 0; idx < extents.size() - 1; ++idx)
        {
          coord_t ext = extents[idx];
          extents[idx] = sum;
          sum += ext;
        }
        extents.back() = sum;
      }

      // Initialize the subspaces using the compute sub-ranges
      for (OutputExtentMap::const_iterator it = output_extents.begin();
           it != output_extents.end(); ++it)
      {
        const DomainPoint& color = it->first;

        // If this subspace isn't local to us, we are not allowed to
        // set its range.
        if (local_extents.find(color) == local_extents.end())
          continue;

        IndexSpaceNode* child =
            part->get_child(part->color_space->linearize_color(color));

        DomainPoint lo;
        lo.dim = ndim;
        DomainPoint hi;
        hi.dim = ndim;
        for (int32_t dim = 0; dim < ndim; ++dim)
        {
          std::vector<coord_t>& extents = all_extents[dim];
          coord_t c = color[dim];
          lo[dim] = extents[c];
          hi[dim] = extents[c + 1] - 1;
        }
        if (child->set_domain(
                Domain(lo, hi), ApEvent::NO_AP_EVENT, false /*take ownership*/,
                true /*broadcast*/))
          delete child;
      }

      // Finally, compute the extents of the root index space and return it
      DomainPoint lo;
      lo.dim = ndim;
      DomainPoint hi;
      hi.dim = ndim;
      for (int32_t dim = 0; dim < ndim; ++dim)
        hi[dim] = all_extents[dim].back() - 1;

      return Domain(lo, hi);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finalize_output_regions(bool first_invocation)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < output_regions.size(); ++idx)
      {
        const OutputOptions& options = output_region_options[idx];
        if (options.valid_requirement())
          continue;
        IndexSpaceNode* parent =
            runtime->get_node(output_regions[idx].parent.get_index_space());
#ifdef DEBUG_LEGION
        validate_output_extents(
            idx, output_regions[idx], output_region_extents[idx]);
#endif
        if (options.global_indexing())
        {
          // For globally indexed output regions, we need to check
          // the alignment between outputs from adjacent point tasks
          // and compute the ranges of subregions via prefix sum.

          IndexPartNode* part = runtime->get_node(
              output_regions[idx].partition.get_index_partition());
          Domain root_domain = compute_global_output_ranges(
              parent, part, output_region_extents[idx],
              output_region_extents[idx]);

          if (parent->set_domain(
                  root_domain, ApEvent::NO_AP_EVENT, false /*take ownership*/))
            delete parent;
        }
        // For locally indexed output regions, sizes of subregions are already
        // set when they are fianlized by the point tasks. So we only need to
        // initialize the root index space by taking a union of subspaces.
        else if (parent->set_output_union(output_region_extents[idx]))
          delete parent;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::initialize_task(
        InnerContext* ctx, const IndexTaskLauncher& launcher,
        IndexSpace launch_sp, Provenance* provenance, bool track /*= true*/,
        std::vector<OutputRequirement>* outputs /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      initialize_regions(launcher.region_requirements);
      futures = launcher.futures;
      // If the task has any output requirements, we create fresh region and
      // partition names and return them back to the user
      if (outputs != nullptr)
        create_output_regions(*outputs, launch_sp);
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);

      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
        arg_manager.save_buffer(launcher.global_arg.get_ptr(), arglen);
        args = arg_manager.get_buffer();
      }
      // Very important that these freezes occur before we initialize
      // this operation because they can launch creation operations to
      // make the future maps
      point_arguments =
          launcher.argument_map.impl->freeze(parent_ctx, provenance);
      const size_t num_point_futures = launcher.point_futures.size();
      if (num_point_futures > 0)
      {
        point_futures.resize(num_point_futures);
        for (unsigned idx = 0; idx < num_point_futures; idx++)
          point_futures[idx] =
              launcher.point_futures[idx].impl->freeze(parent_ctx, provenance);
      }
      concurrent_task = launcher.concurrent;
      concurrent_functor = launcher.concurrent_functor;
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == nullptr);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      is_index_space = true;
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
      assert(launch_space == nullptr);
#endif
      launch_space = runtime->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        index_domain = launch_space->get_tight_domain();
      else
        index_domain = launcher.launch_domain;
      internal_space = launch_space->handle;
      sharding_space = launcher.sharding_space;
      initialize_base_task(ctx, launcher.predicate, task_id, provenance);
      if (outputs != nullptr)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
          REPORT_LEGION_ERROR(
              ERROR_OUTPUT_REGIONS_IN_PREDICATED_TASK,
              "Output requirements are disallowed for tasks launched with "
              "predicates, but preidcated task launch for task %s (%lld) in "
              "parent task %s (UID %lld) is used with output requirements.",
              get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
        if (get_trace() != nullptr)
          REPORT_LEGION_ERROR(
              ERROR_OUTPUT_REGIONS_IN_TRACE,
              "Output requirements are disallowed for tasks launched inside "
              "traces. Task %s (UID %lld) in parent task %s (UID %lld) has "
              "output requirements in trace %d.",
              get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id(), get_trace()->get_trace_id())
      }
      if (!launcher.elide_future_return)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
          initialize_predicate(
              launcher.predicate_false_future, launcher.predicate_false_result);
        future_map = create_future_map(
            ctx, launch_space->handle, launcher.sharding_space);
        future_return_size = launcher.future_return_size;
      }
      else
        elide_future_return = true;
      // Make sure you do this after making the output regions
      compute_parent_indexes(false /*force*/);
      if (concurrent_task && parent_ctx->is_concurrent_context())
        REPORT_LEGION_ERROR(
            ERROR_ILLEGAL_CONCURRENT_EXECUTION,
            "Illegal nested concurrent index space task launch %s (UID %lld) "
            "inside task %s (UID %lld) which has a concurrent ancesstor (must "
            "epoch or index task). Nested concurrency is not supported.",
            get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
            parent_ctx->get_unique_id())
      if (runtime->legion_spy_enabled)
      {
        // Don't log this yet if we're part of a must epoch operation
        if (track)
          LegionSpy::log_index_task(
              parent_ctx->get_unique_id(), unique_op_id, task_id,
              get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it =
                 launcher.wait_barriers.begin();
             it != launcher.wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
      }
      return future_map;
    }

    //--------------------------------------------------------------------------
    Future IndexTask::initialize_task(
        InnerContext* ctx, const IndexTaskLauncher& launcher,
        IndexSpace launch_sp, Provenance* provenance, ReductionOpID redop_id,
        bool deterministic, bool track /*= true*/,
        std::vector<OutputRequirement>* outputs /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
      if (launcher.elide_future_return)
      {
        initialize_task(ctx, launcher, launch_sp, provenance, track, outputs);
        return Future();
      }
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      initialize_regions(launcher.region_requirements);
      futures = launcher.futures;
      // If the task has any output requirements, we create fresh region and
      // partition names and return them back to the user
      if (outputs != nullptr)
        create_output_regions(*outputs, launch_sp);
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);

      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
        arg_manager.save_buffer(launcher.global_arg.get_ptr(), arglen);
        args = arg_manager.get_buffer();
      }
      // Very important that these freezes occur before we initialize
      // this operation because they can launch creation operations to
      // make the future maps
      point_arguments =
          launcher.argument_map.impl->freeze(parent_ctx, provenance);
      const size_t num_point_futures = launcher.point_futures.size();
      if (num_point_futures > 0)
      {
        point_futures.resize(num_point_futures);
        for (unsigned idx = 0; idx < num_point_futures; idx++)
          point_futures[idx] =
              launcher.point_futures[idx].impl->freeze(parent_ctx, provenance);
      }
      concurrent_task = launcher.concurrent;
      concurrent_functor = launcher.concurrent_functor;
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == nullptr);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      is_index_space = true;
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
      assert(launch_space == nullptr);
#endif
      launch_space = runtime->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        index_domain = launch_space->get_tight_domain();
      else
        index_domain = launcher.launch_domain;
      internal_space = launch_space->handle;
      sharding_space = launcher.sharding_space;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      redop_initial_value = launcher.initial_value;
      deterministic_redop = deterministic;
      serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
      if (!reduction_op->identity)
        REPORT_LEGION_ERROR(
            ERROR_REDUCTION_OPERATION_INDEX,
            "Reduction operation %d for index task launch %s "
            "(ID %lld) is not foldable.",
            redop, get_task_name(), get_unique_id())
      initialize_base_task(ctx, launcher.predicate, task_id, provenance);
      if (outputs != nullptr)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
          REPORT_LEGION_ERROR(
              ERROR_OUTPUT_REGIONS_IN_PREDICATED_TASK,
              "Output requirements are disallowed for tasks launched with "
              "predicates, but preidcated task launch for task %s (%lld) in "
              "parent task %s (UID %lld) is used with output requirements.",
              get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
        if (get_trace() != nullptr)
          REPORT_LEGION_ERROR(
              ERROR_OUTPUT_REGIONS_IN_TRACE,
              "Output requirements are disallowed for tasks launched inside "
              "traces. Task %s (UID %lld) in parent task %s (UID %lld) has "
              "output requirements in trace %d.",
              get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id(), get_trace()->get_trace_id())
      }
      if (launcher.predicate != Predicate::TRUE_PRED)
        initialize_predicate(
            launcher.predicate_false_future, launcher.predicate_false_result);
      reduction_future = Future(new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), provenance, this));
      if (serdez_redop_fns == nullptr)
      {
        reduction_future_size = reduction_op->sizeof_rhs;
        reduction_future.impl->set_future_result_size(
            reduction_op->sizeof_rhs, runtime->address_space);
      }
      else if (launcher.future_return_size)
      {
        reduction_future_size = launcher.future_return_size;
        reduction_future.impl->set_future_result_size(
            *reduction_future_size, runtime->address_space);
      }
      // Make sure you do this after making the output regions
      compute_parent_indexes(false /*force*/);
      if (concurrent_task && parent_ctx->is_concurrent_context())
        REPORT_LEGION_ERROR(
            ERROR_ILLEGAL_CONCURRENT_EXECUTION,
            "Illegal nested concurrent index space task launch %s (UID %lld) "
            "inside task %s (UID %lld) which has a concurrent ancesstor (must "
            "epoch or index task). Nested concurrency is not supported.",
            get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
            parent_ctx->get_unique_id())
      if (runtime->legion_spy_enabled && track)
      {
        LegionSpy::log_index_task(
            parent_ctx->get_unique_id(), unique_op_id, task_id,
            get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it =
                 launcher.wait_barriers.begin();
             it != launcher.wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
        LegionSpy::log_future_creation(
            unique_op_id, reduction_future.impl->did, index_point);
      }
      return reduction_future;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_regions(const std::vector<RegionRequirement>& rs)
    //--------------------------------------------------------------------------
    {
      regions = rs;
      // Rewrite any singular region requirements to projections
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionRequirement& req = regions[idx];
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
        {
          req.handle_type = LEGION_REGION_PROJECTION;
          req.projection = 0;  // identity
        }
        // These are some checks for sanity if the user is using the default
        // projection functor from an upper bound region to make sure they
        // know what they are doing
        if (IS_WRITE(req) && (req.projection == 0) &&
            (req.handle_type == LEGION_REGION_PROJECTION))
        {
          if (IS_WRITE_DISCARD(req))
          {
            if (!IS_COLLECTIVE(req))
              REPORT_LEGION_ERROR(
                  ERROR_ALIASED_INTERFERING_REGION,
                  "Parent task %s (UID %lld) issued index space task %s "
                  "(UID %lld) with interfering region requirement %d that "
                  "requested write-discard privileges for all point tasks "
                  "on the same logical region without indicating that they "
                  "should be performed concurrently. If you intend for all "
                  "the point tasks to perform independent writes to the same "
                  "logical region then you must mark the region requirement "
                  "as being a collective write.",
                  parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                  get_task_name(), get_unique_op_id(), idx)
          }
          else if (runtime->runtime_warnings)
            REPORT_LEGION_WARNING(
                LEGION_WARNING_NON_SCALABLE_IDENTITY_PROJECTION,
                "Parent task %s (UID %lld) issued index space task %s "
                "(UID %lld) with non-scalable projection region requirement %d "
                "that ensures all point tasks will be reading and writing to "
                "the same logical region. This implies there will be no task "
                "parallelism in this index space task launch.",
                parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                get_task_name(), get_unique_op_id(), idx)
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_predicate(
        const Future& pred_future, const UntypedBuffer& pred_arg)
    //--------------------------------------------------------------------------
    {
      if (pred_future.impl != nullptr)
        predicate_false_future = pred_future;
      else if (pred_arg.get_size() > 0)
        predicate_false_result.save_buffer(
            pred_arg.get_ptr(), pred_arg.get_size());
    }

    //--------------------------------------------------------------------------
    void IndexTask::prepare_map_must_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch != nullptr);
#endif
      set_origin_mapped(true);
      total_points = launch_space->get_volume();
      if (!elide_future_return)
      {
        future_map = must_epoch->get_future_map();
        enumerate_futures(index_domain);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Initialize the privilege paths
      if (!options_selected)
      {
        const bool inline_task = select_task_options(false /*prioritize*/);
        if (inline_task)
        {
          REPORT_LEGION_WARNING(
              LEGION_WARNING_MAPPER_REQUESTED_INLINE,
              "Mapper %s requested to inline task %s "
              "(UID %lld) but the 'enable_inlining' option was "
              "not set on the task launcher so the request is "
              "being ignored",
              mapper->get_mapper_name(), get_task_name(), get_unique_id());
        }
      }
      if (runtime->legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          TaskOp::log_requirement(unique_op_id, idx, logical_regions[idx]);
        log_launch_space(launch_space->handle);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      analyze_region_requirements(launch_space);
    }

    //--------------------------------------------------------------------------
    void IndexTask::create_output_regions(
        std::vector<OutputRequirement>& outputs, IndexSpace launch_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      size_t num_tasks = runtime->get_domain_volume(launch_space);
#endif
      Provenance* provenance = get_provenance();
      output_region_options.resize(outputs.size());
      output_region_extents.resize(outputs.size());
      for (unsigned idx = 0; idx < outputs.size(); idx++)
      {
        OutputRequirement& req = outputs[idx];
        output_region_options[idx] = OutputOptions(
            req.global_indexing, req.valid_requirement, false /*grouped*/);

        IndexSpace color_space = launch_space;
        if (req.projection != 0)
        {
          color_space = req.color_space;

          if (!color_space.exists())
            REPORT_LEGION_ERROR(
                ERROR_INVALID_OUTPUT_REGION_PROJECTION,
                "Output region %u of task %s (UID: %lld) requests projection "
                "of ID %u but no color space is specified. "
                "Every output requirement with a non-identity projection must "
                "have a color space set.",
                idx, get_task_name(), get_unique_op_id(), req.projection);

#ifdef DEBUG_LEGION
          IndexSpaceNode* node = runtime->get_node(color_space);
          Domain color_domain = node->get_tight_domain();
          if (req.global_indexing && !color_domain.dense())
            REPORT_LEGION_ERROR(
                ERROR_INVALID_OUTPUT_REGION_PROJECTION,
                "The global indexing mode requires the color space of an "
                "output requirement to be dense, but a sparse color space is "
                "assigned to output requirement %u of task %s (UID: %lld).",
                idx, get_task_name(), get_unique_op_id());

          if (color_domain.get_volume() != num_tasks)
            REPORT_LEGION_ERROR(
                ERROR_INVALID_OUTPUT_REGION_PROJECTION,
                "Output region %u of task %s (UID: %lld) requests projection "
                "but the volume of the color space is different from the total "
                "number of point tasks. "
                "The mapping between the launch domain and the subregions must "
                "be bijective.",
                idx, get_task_name(), get_unique_op_id());
#endif
        }
        int color_dim = color_space.get_dim();

        if (!req.valid_requirement)
        {
          TypeTag type_tag;
          int requested_dim =
              Internal::NT_TemplateHelper::get_dim(req.type_tag);
          if (req.global_indexing)
          {
            if (color_dim != requested_dim)
              REPORT_LEGION_ERROR(
                  ERROR_INVALID_OUTPUT_REGION_DOMAIN,
                  "Output region %u of task %s (UID: %lld) is requested to "
                  "have "
                  "%d dimensions, but the color space has %d dimensions. "
                  "Dimensionalities of output regions must be the same as the "
                  "color space's in global indexing mode.",
                  idx, get_task_name(), get_unique_op_id(), requested_dim,
                  launch_space.get_dim());

            type_tag = req.type_tag;
          }
          else
          {
            // When local indexing is used for the output region,
            // we create an (N+1)-D index space when the color domain is N-D.

            // Before creating the index space, we make sure that
            // the dimensionality (N+1) does not exceed LEGION_MAX_DIM.
            if (color_dim + requested_dim > LEGION_MAX_DIM)
              REPORT_LEGION_ERROR(
                  ERROR_INVALID_OUTPUT_REGION_DOMAIN,
                  "Dimensionality of output region %u of task %s (UID: %lld) "
                  "exceeded LEGION_MAX_DIM. You may rebuild your code with a "
                  "bigger LEGION_MAX_DIM value or reduce dimensionality of "
                  "either the color space or the output region.",
                  idx, get_task_name(), get_unique_op_id());

            OutputRegionTagCreator creator(&type_tag, color_dim);
            Internal::NT_TemplateHelper::demux<OutputRegionTagCreator>(
                req.type_tag, &creator);
          }

          // Create a deferred index space
          IndexSpace index_space =
              parent_ctx->create_unbound_index_space(type_tag, provenance);

          // Create a pending partition using the launch domain as the color
          // space
          IndexPartition pid = parent_ctx->create_pending_partition(
              index_space, color_space, LEGION_DISJOINT_COMPLETE_KIND,
              LEGION_AUTO_GENERATE_ID, provenance, true /*trust partitioning*/);

          // Instantiate all local children to ensure that others can refer
          // to them even before they've been computed if necessary
          IndexPartNode* index_part = runtime->get_node(pid);
          for (ColorSpaceIterator itr(index_part, true /*local*/); itr; itr++)
          {
            // This will force the instantiation of the child without blocking
            RtEvent instantiated;
            index_part->get_child(*itr, &instantiated);
            if (instantiated.exists())
              commit_preconditions.insert(instantiated);
          }

          // Create an output region and a partition
          LogicalRegion region = parent_ctx->create_logical_region(
              index_space, req.field_space, false /*local region*/, provenance,
              true /*output region*/);

          LogicalPartition partition =
              runtime->get_logical_partition(region, pid);

          // Set the region and partition back to the output requirement
          // so the caller can use it for downstream tasks
          req.partition = partition;
          req.parent = region;
          req.handle_type = LEGION_PARTITION_PROJECTION;
          req.flags |= LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG;
        }

        req.privilege = LEGION_WRITE_DISCARD;

        // Store the output requirement in the task
        output_regions.emplace_back(req);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_base_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // To be correct with the new scheduler we also have to
      // register mapping dependences on futures
      for (std::vector<Future>::const_iterator it = futures.begin();
           it != futures.end(); it++)
        if (it->impl != nullptr)
          it->impl->register_dependence(this);
      if (predicate_false_future.impl != nullptr)
        predicate_false_future.impl->register_dependence(this);
      // Always have to register a full dependence on this since we need
      // to have the producer mapped by the time we're enumerating points
      if (point_arguments.impl != nullptr)
        point_arguments.impl->register_dependence(this);
      // Register mapping dependences on any future maps also
      // if we're not pointwise analyzable. If we are then we'll
      // do pointwise analysis on these when we enumerate the points
      if (!is_pointwise_analyzable())
      {
        for (std::vector<FutureMap>::const_iterator it = point_futures.begin();
             it != point_futures.end(); it++)
          it->impl->register_dependence(this);
      }
#ifdef LEGION_SPY
      else
      {
        // Record pointwise dependences on the point futures
        for (std::vector<FutureMap>::const_iterator it = point_futures.begin();
             it != point_futures.end(); it++)
        {
          if (!it->impl->context_index)
            continue;
          LegionSpy::log_future_dependence(
              parent_ctx->get_unique_id(), it->impl->op_uid, unique_op_id,
              true /*pointwise*/);
        }
      }
#endif
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(
            this, wait_barriers, arrive_barriers, must_epoch);
      version_infos.resize(logical_regions.size());
    }

    //--------------------------------------------------------------------------
    void IndexTask::report_interfering_requirements(
        unsigned idx1, unsigned idx2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx1 < idx2);
#endif
      // The logical dependence analysis can report this because there are
      // interfering fields and regions, check to make sure there are alos
      // interfering privileges and index spaces
      const RegionRequirement& req1 = get_requirement(idx1);
      const RegionRequirement& req2 = get_requirement(idx2);
      if (IS_READ_ONLY(req1) && IS_READ_ONLY(req2))
        return;
      if (IS_REDUCE(req1) && IS_REDUCE(req2) && (req1.redop == req2.redop))
        return;
      interfering_requirements.insert(
          std::pair<unsigned, unsigned>(idx1, idx2));
    }

    //--------------------------------------------------------------------------
    bool IndexTask::record_trace_hash(
        TraceRecognizer& recognizer, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      if (output_regions.size() > 0)
        return recognizer.record_operation_untraceable(opidx);
      Murmur3Hasher hasher;
      hasher.hash(get_operation_kind());
      hasher.hash(task_id);
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
           it != regions.end(); it++)
        hash_requirement(hasher, *it);
      hasher.hash(concurrent_functor);
      hasher.hash<bool>(concurrent_task);
      hasher.hash<bool>(must_epoch_task);
      hasher.hash(index_domain);
      if (future_return_size)
        hasher.hash(*future_return_size);
      if (reduction_future_size)
        hasher.hash(*reduction_future_size);
      return recognizer.record_operation_hash(this, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_model)
        start_check_point_requirements();
      // Do a quick test for empty index space launches
      total_points = launch_space->get_volume();
      if (total_points == 0)
      {
        // Clean up this task execution if there are no points
        complete_mapping();
        complete_execution();
        trigger_children_committed();
      }
      else
      {
        // If we had a trace then we might need to recompute our parent
        // region indexes now since we might have skipped it earlier
        if (trace != nullptr)
          compute_parent_indexes(true /*force*/);
        // Enumerate the futures in the future map
        if ((redop == 0) && !elide_future_return)
          enumerate_futures(index_domain);
        if (concurrent_task)
        {
          // Instantiate preconditions for each of the colors
          ConcurrentColoringFunctor* functor =
              runtime->find_concurrent_coloring_functor(concurrent_functor);
          if (functor->supports_max_color())
          {
            Color max_color = functor->max_color(index_domain);
            for (Color color = 0; color <= max_color; color++)
              concurrent_groups[color].precondition.interpreted =
                  Runtime::create_rt_user_event();
          }
          else
          {
            // If we're recording we need to iterate all the points to
            // count how many points are associated with each color so
            // we can make the barrier to be used later
            for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
            {
              Color color = functor->color(*itr, index_domain);
              std::map<Color, ConcurrentGroup>::iterator finder =
                  concurrent_groups.find(color);
              if (finder == concurrent_groups.end())
              {
                finder = concurrent_groups
                             .emplace(std::make_pair(color, ConcurrentGroup()))
                             .first;
                finder->second.precondition.interpreted =
                    Runtime::create_rt_user_event();
              }
            }
          }
        }
        Operation::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    size_t IndexTask::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return launch_space->get_volume();
    }

    //--------------------------------------------------------------------------
    void IndexTask::enumerate_futures(const Domain& domain)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!elide_future_return);
      assert(future_handles == nullptr);
#endif
      future_handles = new FutureHandles;
      future_handles->add_reference();
      std::map<DomainPoint, DistributedID>& handles = future_handles->handles;
      for (Domain::DomainPointIterator itr(domain); itr; itr++)
      {
        Future f = future_map.impl->get_future(itr.p, true /*internal only*/);
        handles[itr.p] = f.impl->did;
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::rendezvous_concurrent_mapped(
        const DomainPoint& point, Processor target, Color color,
        RtEvent precondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(concurrent_task);
#endif
      bool done = false;
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
#ifdef DEBUG_LEGION
      assert(finder != concurrent_groups.end());
#endif
      {
        AutoLock o_lock(op_lock);
        if (precondition.exists())
          finder->second.preconditions.emplace_back(precondition);
        std::map<Processor, DomainPoint>::const_iterator proc_finder =
            finder->second.processors.find(target);
        if (proc_finder != finder->second.processors.end())
          report_concurrent_mapping_failure(target, point, proc_finder->second);
        finder->second.processors[target] = point;
        finder->second.group_points++;
#ifdef DEBUG_LEGION
        assert(concurrent_points < total_points);
#endif
        done = (++concurrent_points == total_points);
      }
      if (done)
        finalize_concurrent_mapped();
    }

    //--------------------------------------------------------------------------
    void IndexTask::rendezvous_concurrent_mapped(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(concurrent_task);
#endif
      bool done = false;
      {
        size_t num_colors;
        derez.deserialize(num_colors);
        AutoLock o_lock(op_lock);
        for (unsigned idx1 = 0; idx1 < num_colors; idx1++)
        {
          Color color;
          derez.deserialize(color);
          std::map<Color, ConcurrentGroup>::iterator finder =
              concurrent_groups.find(color);
#ifdef DEBUG_LEGION
          assert(finder != concurrent_groups.end());
#endif
          RtEvent precondition;
          derez.deserialize(precondition);
          if (precondition.exists())
            finder->second.preconditions.emplace_back(precondition);
          size_t num_points;
          derez.deserialize(num_points);
          for (unsigned idx2 = 0; idx2 < num_points; idx2++)
          {
            Processor target;
            derez.deserialize(target);
            DomainPoint point;
            derez.deserialize(point);
            std::map<Processor, DomainPoint>::const_iterator proc_finder =
                finder->second.processors.find(target);
            if (proc_finder != finder->second.processors.end())
              report_concurrent_mapping_failure(
                  target, point, proc_finder->second);
            finder->second.processors[target] = point;
          }
          finder->second.group_points += num_points;
          concurrent_points += num_points;
#ifdef DEBUG_LEGION
          assert(concurrent_points <= total_points);
#endif
        }
        done = (concurrent_points == total_points);
      }
      if (done)
        finalize_concurrent_mapped();
    }

    //--------------------------------------------------------------------------
    void IndexTask::finalize_concurrent_mapped(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<Color, ConcurrentGroup>::iterator it =
               concurrent_groups.begin();
           it != concurrent_groups.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->second.precondition.interpreted.exists());
#endif
        it->second.color_points = it->second.group_points;
        if (is_recording())
        {
          // Create and record the barrier with the right number of arrivals
          // for this concurrent group for later iterations
          const RtBarrier barrier =
              runtime->create_rt_barrier(it->second.color_points);
          it->second.shards.resize(1, 0);
          tpl->record_concurrent_group(
              this, it->first, it->second.group_points, it->second.color_points,
              barrier, it->second.shards);
        }
        if (!it->second.preconditions.empty())
          Runtime::trigger_event(
              it->second.precondition.interpreted,
              Runtime::merge_events(it->second.preconditions));
        else
          Runtime::trigger_event(it->second.precondition.interpreted);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_concurrent_group(
        Color color, size_t local, size_t global, RtBarrier barrier,
        const std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(concurrent_groups.find(color) == concurrent_groups.end());
#endif
      ConcurrentGroup& group = concurrent_groups[color];
      group.group_points = local;
      group.color_points = global;
      group.precondition.traced = barrier;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_must_epoch_concurrent_group(
        Color color, RtUserEvent precondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch_task);
      assert(concurrent_groups.find(color) == concurrent_groups.end());
#endif
      ConcurrentGroup& group = concurrent_groups[color];
      group.precondition.interpreted = precondition;
    }

    //--------------------------------------------------------------------------
    void IndexTask::concurrent_allreduce(
        Color color, SliceTask* slice, AddressSpaceID slice_space,
        size_t points, uint64_t lamport_clock, VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      if (must_epoch_task)
      {
#ifdef DEBUG_LEGION
        assert(color == 0);
#endif
        must_epoch->concurrent_allreduce(
            slice, slice_space, points, lamport_clock, poisoned);
        return;
      }
      bool done = false;
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
#ifdef DEBUG_LEGION
      assert(finder != concurrent_groups.end());
#endif
      {
        AutoLock o_lock(op_lock);
        if (finder->second.lamport_clock < lamport_clock)
          finder->second.lamport_clock = lamport_clock;
        if (poisoned)
          finder->second.poisoned = true;
        if (finder->second.slice_tasks.empty())
          finder->second.variant = vid;
        else if (finder->second.variant != vid)
          finder->second.variant = std::min(finder->second.variant, vid);
        finder->second.slice_tasks.emplace_back(
            std::make_pair(slice, slice_space));
#ifdef DEBUG_LEGION
        assert(points <= finder->second.group_points);
#endif
        finder->second.group_points -= points;
        done = (finder->second.group_points == 0);
      }
      if (done)
      {
        if (finder->second.variant > 0)
        {
          // Check to see if this variant needs a task barrier
          VariantImpl* variant =
              runtime->find_variant_impl(task_id, finder->second.variant);
          if (variant->needs_barrier())
            finder->second.task_barrier =
                runtime->create_rt_barrier(finder->second.color_points);
        }
        // Swap this vector onto the stack in case the slice task gets deleted
        // out from under us while we are finalizing things
        std::vector<std::pair<SliceTask*, AddressSpaceID>> local_copy;
        local_copy.swap(finder->second.slice_tasks);
        for (std::vector<std::pair<SliceTask*, AddressSpaceID>>::const_iterator
                 it = local_copy.begin();
             it != local_copy.end(); it++)
        {
          if (it->second != runtime->address_space)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(it->first);
              rez.serialize(color);
              rez.serialize(finder->second.task_barrier);
              rez.serialize(finder->second.lamport_clock);
              rez.serialize(finder->second.variant);
              rez.serialize(finder->second.poisoned);
            }
            runtime->send_slice_concurrent_allreduce_response(it->second, rez);
          }
          else
            it->first->finish_concurrent_allreduce(
                color, finder->second.lamport_clock, finder->second.poisoned,
                finder->second.variant, finder->second.task_barrier);
        }
      }
    }

    //--------------------------------------------------------------------------
    uint64_t IndexTask::collective_lamport_allreduce(
        uint64_t lamport_clock, size_t points, bool need_result)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      if (collective_lamport_clock < lamport_clock)
        collective_lamport_clock = lamport_clock;
      collective_unbounded_points += points;
#ifdef DEBUG_LEGION
      assert(collective_unbounded_points <= total_points);
#endif
      if (collective_unbounded_points < total_points)
      {
        if (need_result)
        {
          if (!collective_lamport_clock_ready.exists())
            collective_lamport_clock_ready = Runtime::create_rt_user_event();
          o_lock.release();
          collective_lamport_clock_ready.wait();
        }
      }
      else if (collective_lamport_clock_ready.exists())
        Runtime::trigger_event(collective_lamport_clock_ready);
      return collective_lamport_clock;
    }

    //--------------------------------------------------------------------------
    void IndexTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      RtEvent execution_condition;
      // Fill in the index task map with the default future value
      if (!elide_future_return)
      {
        if (redop == 0)
        {
          // Only need to do this if the internal domain exists, it
          // might not in a control replication context
          if (internal_space.exists())
          {
            // Get the domain that we will have to iterate over
            Domain local_domain;
            runtime->find_domain(internal_space, local_domain);
            // Handling the future map case
            if (predicate_false_future.impl != nullptr)
            {
              for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
              {
                Future f =
                    future_map.impl->get_future(itr.p, true /*internal*/);
                // Safe to block indefinitely waiting for unbounded pools
                f.impl->set_result(
                    this, predicate_false_future.impl,
                    nullptr /*safe_for_unbounded_pools*/);
              }
            }
            else
            {
              for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
              {
                Future f =
                    future_map.impl->get_future(itr.p, true /*internal*/);
                if (predicate_false_result.get_size() > 0)
                  f.impl->set_local(
                      predicate_false_result.get_buffer(),
                      predicate_false_result.get_size(), false /*own*/);
                else
                  f.impl->set_result(ApEvent::NO_AP_EVENT, nullptr);
              }
            }
          }
        }
        else
        {
          // Handling a reduction case
          if (redop_initial_value.impl != nullptr)
          {
            // Safe to block here indefinitely waiting for unbounded pools
            reduction_future.impl->set_result(
                this, redop_initial_value.impl,
                nullptr /*safe_for_unbounded_pools*/);
          }
          else
            reduction_future.impl->set_local(
                &reduction_op->identity, reduction_op->sizeof_rhs);
        }
      }
      // Can check this without the lock since we know the predication state
      // has been marked correctly while holding the lock
      if (!pending_pointwise_dependences.empty())
      {
        // Just trigger these since the points won't be mapped anyway
        for (std::map<DomainPoint, RtUserEvent>::const_iterator it =
                 pending_pointwise_dependences.begin();
             it != pending_pointwise_dependences.end(); it++)
          Runtime::trigger_event(it->second);
        pending_pointwise_dependences.clear();
      }
      // Then clean up this task execution
      complete_mapping();
      complete_execution(execution_condition);
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
      // We only need to premap the task if it has a reduction down
      // to individual futures so that we can map the futures
      if (redop == 0)
        return;
      // Call premap task here to see if there are any future destinations
      Mapper::PremapTaskInput input;
      Mapper::PremapTaskOutput output;
      // Initialize this to not have a new target processor
      output.new_target_proc = Processor::NO_PROC;
      // Now invoke the mapper call
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_premap_task(this, input, output);
      // See if we need to update the new target processor
      if (output.new_target_proc.exists())
        this->target_proc = output.new_target_proc;
      create_future_instances(output.reduction_futures);
      // If we're recording this trace then we need to remember this
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((tpl != nullptr) && tpl->is_recording());
#endif
        tpl->record_premap_output(this, output, map_applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::create_future_instances(std::vector<Memory>& target_mems)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_instances.empty());
#endif
      if (!target_mems.empty())
      {
        if (target_mems.size() > 1)
        {
          std::set<Memory> unique_mems;
          for (std::vector<Memory>::iterator it = target_mems.begin();
               it != target_mems.end();
               /*nothing*/)
          {
            if (!it->exists())
              REPORT_LEGION_ERROR(
                  ERROR_INVALID_MAPPER_OUTPUT,
                  "Invalid mapper output. Mapper %s requested index task "
                  "reduction future be mapped to a NO_MEMORY for task %s "
                  "(UID %lld) which is illegal. All requests for mapping "
                  "output futures must be mapped to actual memories.",
                  mapper->get_mapper_name(), get_task_name(), unique_op_id)
            if (unique_mems.find(*it) == unique_mems.end())
            {
              unique_mems.insert(*it);
              it++;
            }
            else
              it = target_mems.erase(it);
          }
        }
        else if (!(target_mems.begin()->exists()))
          REPORT_LEGION_ERROR(
              ERROR_INVALID_MAPPER_OUTPUT,
              "Invalid mapper output. Mapper %s requested index task "
              "reduction future be mapped to a NO_MEMORY for task %s "
              "(UID %lld) which is illegal. All requests for mapping "
              "output futures must be mapped to actual memories.",
              mapper->get_mapper_name(), get_task_name(), unique_op_id)
      }
      else
        target_mems.emplace_back(runtime->runtime_system_memory);
      // If we've got a serdez redop function then we don't know how big
      // the output is going to be until later, otherwise we know the
      // output size from the reduction operator
      if (serdez_redop_fns == nullptr)
      {
        reduction_instances.reserve(target_mems.size());
        TaskTreeCoordinates coordinates;
        compute_task_tree_coordinates(coordinates);
        int runtime_visible_index = -1;
        for (std::vector<Memory>::const_iterator it = target_mems.begin();
             it != target_mems.end(); it++)
        {
          if ((runtime_visible_index < 0) &&
              FutureInstance::check_meta_visible(*it))
            runtime_visible_index = reduction_instances.size();
          MemoryManager* manager = runtime->find_memory_manager(*it);
          // Safe to block here indefinitely waiting for unbounded pools
          reduction_instances.emplace_back(manager->create_future_instance(
              unique_op_id, coordinates, reduction_op->sizeof_rhs,
              nullptr /*safe_for_unbounded_pools*/));
        }
        // This is an important optimization: if we're doing a small
        // reduction value we always want the reduction instance to
        // be somewhere meta visible for performance reasons, so we
        // make a meta-visible instance if we don't have one
        if ((runtime_visible_index < 0) &&
            (reduction_op->sizeof_rhs <= LEGION_MAX_RETURN_SIZE))
        {
          runtime_visible_index = reduction_instances.size();
          MemoryManager* manager =
              runtime->find_memory_manager(runtime->runtime_system_memory);
          // Safe to block here indefinitely waiting for unbounded pools
          reduction_instances.emplace_back(manager->create_future_instance(
              unique_op_id, coordinates, reduction_op->sizeof_rhs,
              nullptr /*safe_for_unbounded_pools*/));
        }
        if (runtime_visible_index > 0)
          std::swap(
              reduction_instances.front(),
              reduction_instances[runtime_visible_index]);
#ifdef DEBUG_LEGION
        assert(reduction_instance == nullptr);
#endif
        reduction_instance = reduction_instances.front();
        // Need to initialize this with the reduction value
        if ((redop_initial_value.impl != nullptr) &&
            (parent_ctx->get_task()->get_shard_id() == 0))
          reduction_instance_precondition = redop_initial_value.impl->copy_to(
              reduction_instance, this, execution_fence_event);
        else
          reduction_instance_precondition =
              reduction_instance.load()->initialize(
                  reduction_op, this, execution_fence_event);
      }
      else
      {
        if ((redop_initial_value.impl != nullptr) &&
            (parent_ctx->get_task()->get_shard_id() == 0))
        {
          redop_initial_value.impl->request_runtime_instance(this);
          const void* value = redop_initial_value.impl->find_runtime_buffer(
              parent_ctx, serdez_redop_state_size);
          serdez_redop_state = malloc(serdez_redop_state_size);
          memcpy(serdez_redop_state, value, serdez_redop_state_size);
        }
        serdez_redop_targets.swap(target_mems);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      if (is_origin_mapped())
      {
        // This will only get called if we had slices that couldn't map, but
        // they have now all mapped
#ifdef DEBUG_LEGION
        assert(slices.empty());
#endif
        // We're never actually run
        return false;
      }
      else
      {
        if (!is_sliced() && target_proc.exists() &&
            !runtime->is_local(target_proc))
        {
          // Make a slice copy and send it away
          SliceTask* clone = clone_as_slice_task(
              internal_space, target_proc, true /*needs slice*/, stealable);
          runtime->send_task(clone);
          return false;  // We have now been sent away
        }
        else
        {
          set_current_proc(target_proc);
          return true;  // Still local so we can be sliced
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::perform_mapping(
        MustEpochOp* owner /*=nullptr*/,
        const DeferMappingArgs* args /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      // This will only get called if we had slices that failed to origin map
#ifdef DEBUG_LEGION
      assert(!slices.empty());
      // Should never get duplicate invocations here
      assert(args == nullptr);
#endif
      while (!slices.empty())
      {
        SliceTask* slice = slices.front();
        slices.pop_front();
        slice->trigger_mapping();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      // Index space tasks are never stealable, they must first be
      // split into slices which can then be stolen.  Note that slicing
      // always happens after premapping so we know stealing is safe.
      return false;
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind IndexTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDEX_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(
          unique_op_id, ApEvent::NO_AP_EVENT, effects);
#endif
      // Set the future if we actually ran the task or we speculated
      if ((redop > 0) && (predication_state != PREDICATED_FALSE_STATE))
      {
        if (reduction_future_size &&
            (*reduction_future_size < reduction_instances.front()->size))
        {
#ifdef DEBUG_LEGION
          // This failure mode should only happen with serdez redops fns
          // since we should do the other reductions correctly ourself
          assert(serdez_redop_fns != nullptr);
#endif
          Provenance* provenance = get_provenance();
          if (provenance != nullptr)
            REPORT_LEGION_ERROR(
                ERROR_FUTURE_SIZE_BOUNDS_EXCEEDED,
                "Index Task %s (UID %lld, provenance: %.*s) produced a "
                "reduced future value of %zd bytes which is larger than "
                "the dynamically specified bounds of %zd bytes.",
                get_task_name(), get_unique_id(),
                int(provenance->human.length()), provenance->human.data(),
                reduction_instances.front()->size, *reduction_future_size)
          else
            REPORT_LEGION_ERROR(
                ERROR_FUTURE_SIZE_BOUNDS_EXCEEDED,
                "Index Task %s (UID %lld) produced a reduced future value "
                "of %zd bytes which is larger than the dynamically "
                "specified bounds of %zd bytes.",
                get_task_name(), get_unique_id(),
                reduction_instances.front()->size, *reduction_future_size)
        }
        reduction_future.impl->set_results(
            effects, reduction_instances, reduction_metadata,
            reduction_metasize);
        // Clear this since we no longer own the buffer
        reduction_metadata = nullptr;
        reduction_instances.clear();
      }
      if (must_epoch != nullptr)
      {
        must_epoch->notify_subop_complete(this, effects);
        complete_operation(effects);
      }
      else
        complete_operation(effects);
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
      {
        if (outstanding_profiling_requests == 0)
        {
          // We're not expecting any profiling callbacks so we need to
          // do one ourself to inform the mapper that there won't be any
          Mapping::Mapper::TaskProfilingInfo info;
          info.total_reports = 0;
          info.task_response = true;
          info.region_requirement_index = 0;
          info.fill_response = false;  // make valgrind happy
          mapper->invoke_task_report_profiling(this, info);
          Runtime::trigger_event(profiling_reported);
        }
        else
          commit_preconditions.insert(profiling_reported);
      }
      if (must_epoch != nullptr)
      {
        RtEvent commit_precondition;
        if (!commit_preconditions.empty())
          commit_precondition = Runtime::merge_events(commit_preconditions);
        must_epoch->notify_subop_commit(this, commit_precondition);
        commit_operation(true /*deactivate*/, commit_precondition);
      }
      else
      {
        // Mark that this operation is now committed
        if (!commit_preconditions.empty())
          commit_operation(
              true /*deactivate*/, Runtime::merge_events(commit_preconditions));
        else
          commit_operation(true /*deactivate*/);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::pack_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool IndexTask::unpack_task(
        Deserializer& derez, Processor current, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_inlining(
        VariantImpl* variant, const std::deque<InstanceSet>& parent_regions)
    //--------------------------------------------------------------------------
    {
      total_points = launch_space->get_volume();
      if ((redop == 0) && !elide_future_return)
        enumerate_futures(index_domain);
      SliceTask* slice = clone_as_slice_task(
          launch_space->handle, current_proc, false /*recurse*/,
          false /*stealable*/);
      slice->enumerate_points(true /*inlining*/);
      slice->perform_inlining(variant, parent_regions);
    }

    //--------------------------------------------------------------------------
    SliceTask* IndexTask::clone_as_slice_task(
        IndexSpace is, Processor p, bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      SliceTask* result = runtime->get_operation<SliceTask>();
      result->initialize_base_task(
          parent_ctx, Predicate::TRUE_PRED, this->task_id, get_provenance());
      result->clone_multi_from(this, is, p, recurse, stealable);
      result->index_owner = this;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_index_slice(get_unique_id(), result->get_unique_id());
      if (implicit_profiler != nullptr)
        implicit_profiler->register_slice_owner(
            get_unique_op_id(), result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::reduce_future(
        const DomainPoint& point, FutureInstance* inst, ApEvent effects)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_op != nullptr);
#endif
      // If we're doing a deterministic reduction then we need to
      // buffer up these future values until we get all of them so
      // that we can fold them in a deterministic way
      if (deterministic_redop)
      {
        // Store it in our temporary futures for later
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(temporary_futures.find(point) == temporary_futures.end());
#endif
        temporary_futures[point] = std::make_pair(inst, effects);
      }
      else
      {
        if (!fold_reduction_future(inst, effects))
        {
          // save it to delete later
          AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
          assert(temporary_futures.find(point) == temporary_futures.end());
#endif
          temporary_futures[point] = std::make_pair(inst, effects);
        }
        else
          delete inst;
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::pack_profiling_requests(
        Serializer& rez, std::set<RtEvent>& applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(copy_fill_priority);
      rez.serialize<size_t>(copy_profiling_requests.size());
      if (!copy_profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < copy_profiling_requests.size(); idx++)
          rez.serialize(copy_profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Send a message to the owner with an update for the extra counts
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        rez.serialize<RtEvent>(done_event);
        applied.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    int IndexTask::add_copy_profiling_request(
        const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
        bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any copy profiling requests
      if (copy_profiling_requests.empty())
        return copy_fill_priority;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest& request = requests.add_request(
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, &response,
          sizeof(response));
      bool has_finish = false;
      for (std::vector<ProfilingMeasurementID>::const_iterator it =
               copy_profiling_requests.begin();
           it != copy_profiling_requests.end(); it++)
      {
        const Realm::ProfilingMeasurementID measurement =
            (Realm::ProfilingMeasurementID)*it;
        request.add_measurement(measurement);
        if (measurement == Realm::PMID_OP_FINISH_EVENT)
          has_finish = true;
      }
      // Need thetimeline for the operation to know how to profile this
      // profiling response
      if (!has_finish && (runtime->profiler != nullptr))
        request.add_measurement(Realm::PMID_OP_FINISH_EVENT);
      handle_profiling_update(count);
      return copy_fill_priority;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      const OpProfilingResponse* task_prof =
          static_cast<const OpProfilingResponse*>(response.user_data());
      Realm::ProfilingMeasurements::OperationFinishEvent finish_event;
      if (response.get_measurement(finish_event))
        fevent = LgEvent(finish_event.finish_event);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many reports to expect
      const RtEvent mapped = get_mapped_event();
      if (!mapped.has_triggered())
        mapped.wait();
      // If we get here then we can handle the response now
      Mapping::Mapper::TaskProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.task_response = task_prof->task;
      info.region_requirement_index = task_prof->src;
      info.total_reports = outstanding_profiling_requests;
      info.fill_response = task_prof->fill;
      mapper->invoke_task_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests);
#endif
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void IndexTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::create_future_map(
        TaskContext* ctx, IndexSpace launch_space, IndexSpace sharding_space)
    //--------------------------------------------------------------------------
    {
      FutureMapImpl* result = new FutureMapImpl(
          ctx, this, this->launch_space,
          runtime->get_available_distributed_id(), get_provenance());
      future_map_coordinate = result->blocking_index;
      return FutureMap(result);
    }

    //--------------------------------------------------------------------------
    RtEvent IndexTask::find_intra_space_dependence(
        const DomainPoint& point, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      // We're not control replicated so this is just a pointwise dependence
      return find_pointwise_dependence(point, get_generation(), to_trigger);
    }

    //--------------------------------------------------------------------------
    RtEvent IndexTask::find_pointwise_dependence(
        const DomainPoint& point, GenerationID needed_gen,
        RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(needed_gen <= gen);
#endif
      if ((needed_gen < gen) || mapped ||
          (predication_state == PREDICATED_FALSE_STATE))
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return RtEvent::NO_RT_EVENT;
      }
      // See if we can find this in the point mapped events data structure
      std::map<DomainPoint, RtEvent>::const_iterator finder =
          point_mapped_events.find(point);
      if (finder != point_mapped_events.end())
      {
        if (to_trigger.exists())
        {
          Runtime::trigger_event(to_trigger, finder->second);
          return to_trigger;
        }
        else
          return finder->second;
      }
      else
      {
        // Create a pending pointwise dependence for this point
        std::map<DomainPoint, RtUserEvent>::const_iterator pending_finder =
            pending_pointwise_dependences.find(point);
        if (pending_finder != pending_pointwise_dependences.end())
        {
          if (to_trigger.exists())
          {
            Runtime::trigger_event(to_trigger, pending_finder->second);
            return to_trigger;
          }
          else
            return pending_finder->second;
        }
        if (!to_trigger.exists())
          to_trigger = Runtime::create_rt_user_event();
        pending_pointwise_dependences.emplace(
            std::make_pair(point, to_trigger));
        return to_trigger;
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_origin_mapped_slice(SliceTask* local_slice)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      origin_mapped_slices.emplace_back(local_slice);
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_point_mapped(
        const DomainPoint& point, RtEvent mapped_event)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      bool trigger_children_commit = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(point_mapped_events.find(point) == point_mapped_events.end());
#endif
        point_mapped_events.emplace(std::make_pair(point, mapped_event));
        std::map<DomainPoint, RtUserEvent>::iterator finder =
            pending_pointwise_dependences.find(point);
        if (finder != pending_pointwise_dependences.end())
        {
          Runtime::trigger_event(finder->second, mapped_event);
          pending_pointwise_dependences.erase(finder);
        }
        if (mapped_event.exists())
          map_applied_conditions.insert(mapped_event);
#ifdef DEBUG_LEGION
        assert(mapped_points < total_points);
#endif
        if (++mapped_points == total_points)
        {
          // Don't complete this yet if we have redop serdez fns because
          // we still need to map the output future instance before we
          // can consider ourselves mapped and we can't do that until we
          // get the final future value
          if (serdez_redop_fns == nullptr)
            need_trigger = true;
          if ((committed_points == total_points) && !children_commit_invoked)
          {
            trigger_children_commit = true;
            children_commit_invoked = true;
          }
        }
      }
      if (need_trigger)
      {
        // Get the mapped precondition note we can now access this
        // without holding the lock because we know we've seen
        // all the responses so no one else will be mutating it.
        if (!map_applied_conditions.empty())
        {
          RtEvent map_condition = Runtime::merge_events(map_applied_conditions);
          complete_mapping(map_condition);
        }
        else
          complete_mapping();
      }
      if (trigger_children_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_complete(
        unsigned points, ApEvent slice_effect, void* metadata /*= nullptr*/,
        size_t metasize /*= 0*/)
    //--------------------------------------------------------------------------
    {
      if (slice_effect.exists())
        record_completion_effect(slice_effect);
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        completed_points += points;
#ifdef DEBUG_LEGION
        assert(completed_points <= total_points);
#endif
        need_trigger = (completed_points == total_points);
        if (metadata != nullptr)
        {
#ifdef DEBUG_LEGION
          assert(redop > 0);
#endif
          if (reduction_metadata == nullptr)
          {
            reduction_metadata = metadata;
            reduction_metasize = metasize;
            metadata = nullptr;  // mark that we grabbed it
          }
        }
      }
      if (need_trigger)
      {
        // If we are reducing to a single value we need to finish that now
        if ((redop > 0) && (predication_state != PREDICATED_FALSE_STATE))
        {
#ifdef DEBUG_LEGION
          assert((serdez_redop_fns != nullptr) || !reduction_instances.empty());
          assert(
              (serdez_redop_fns != nullptr) ||
              (reduction_instance == reduction_instances.front()));
#endif
          // First finish applying any deterministic reductions
          if (deterministic_redop)
          {
            // Fold any temporary future for deterministic reduction
            for (std::map<DomainPoint, std::pair<FutureInstance*, ApEvent>>::
                     iterator it = temporary_futures.begin();
                 it != temporary_futures.end();
                 /*nothing*/)
            {
              if (fold_reduction_future(it->second.first, it->second.second))
              {
                delete it->second.first;
                std::map<DomainPoint, std::pair<FutureInstance*, ApEvent>>::
                    iterator to_delete = it++;
                temporary_futures.erase(to_delete);
              }
              else
                it++;
            }
          }
          else if (serdez_redop_fns == nullptr)
          {
            // Merge any reduction fold events back into the
            // reduction_instance_precondition to know when the
            // reduction instance is safe to use
            // Note all these events dominate the reduction fold precondition
            // so there is no need to include and we can just overwrite it
            if (!reduction_fold_effects.empty())
            {
              reduction_instance_precondition =
                  Runtime::merge_events(nullptr, reduction_fold_effects);
              reduction_fold_effects.clear();
            }
          }
#ifdef DEBUG_LEGION
          assert(reduction_fold_effects.empty());
#endif
          // Finish the index task reduction
          finish_index_task_reduction();
        }
        // Forward completion effects for any local-mapped slices
        for (std::vector<SliceTask*>::const_iterator it =
                 origin_mapped_slices.begin();
             it != origin_mapped_slices.end(); it++)
          (*it)->forward_completion_effects();
        complete_execution();
      }
      // If we didn't grab ownership then free this now
      if (metadata != nullptr)
        free(metadata);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finish_index_task_reduction(void)
    //--------------------------------------------------------------------------
    {
      // If we have serdez redop fns, we now know how big the output
      // is so we can make our target instances and complete the mapping
      if (serdez_redop_fns != nullptr)
      {
#ifdef DEBUG_LEGION
        assert(reduction_instances.empty());
        assert(!serdez_redop_targets.empty());
#endif
        reduction_instances.reserve(serdez_redop_targets.size());
        int runtime_visible_index = -1;
        TaskTreeCoordinates coordinates;
        for (std::vector<Memory>::const_iterator it =
                 serdez_redop_targets.begin();
             it != serdez_redop_targets.end(); it++)
        {
          if ((runtime_visible_index == -1) &&
              ((*it) == runtime->runtime_system_memory))
          {
            runtime_visible_index = reduction_instances.size();
            reduction_instances.emplace_back(FutureInstance::create_local(
                serdez_redop_state, serdez_redop_state_size, false /*own*/));
          }
          else
          {
            if (coordinates.empty())
              compute_task_tree_coordinates(coordinates);
            MemoryManager* manager = runtime->find_memory_manager(*it);
            // Safe to block here indefinitely waiting for unbounded pools
            reduction_instances.emplace_back(manager->create_future_instance(
                unique_op_id, coordinates, serdez_redop_state_size,
                nullptr /*safe_for_unbounded_pools*/));
          }
        }
        if (runtime_visible_index < 0)
        {
          runtime_visible_index = reduction_instances.size();
          reduction_instances.emplace_back(FutureInstance::create_local(
              serdez_redop_state, serdez_redop_state_size, false /*own*/));
        }
        // Make sure the instance with the data is at the front
        if (runtime_visible_index > 0)
          std::swap(
              reduction_instances.front(),
              reduction_instances[runtime_visible_index]);
        reduction_instance = reduction_instances.front();
        // Get the mapped precondition note we can now access this
        // without holding the lock because we know we've seen
        // all the responses so no one else will be mutating it.
        if (!map_applied_conditions.empty())
        {
          const RtEvent map_condition =
              Runtime::merge_events(map_applied_conditions);
          complete_mapping(map_condition);
        }
        else
          complete_mapping();
      }
#ifdef DEBUG_LEGION
      assert(!reduction_instances.empty());
      assert(reduction_instance == reduction_instances.front());
      assert(reduction_fold_effects.empty());
#endif
      // Now do the copy out from the reduction_instance to any other
      // target futures that we have, we'll do this with a broadcast tree
      if (reduction_instances.size() > 1)
      {
        std::vector<ApEvent> reduction_instances_ready(
            reduction_instances.size(), reduction_instance_precondition);
        // Do the copy from 0 to 1 first
        reduction_instances_ready[1] = reduction_instances[1]->copy_from(
            reduction_instance, this, reduction_instances_ready[0]);
        for (unsigned idx = 1; idx < reduction_instances.size(); idx++)
        {
          if (reduction_instances.size() <= (2 * idx))
            break;
          reduction_instances_ready[2 * idx] =
              reduction_instances[2 * idx]->copy_from(
                  reduction_instances[idx], this,
                  reduction_instances_ready[idx]);
          if (reduction_instances.size() <= (2 * idx + 1))
            break;
          reduction_instances_ready[2 * idx + 1] =
              reduction_instances[2 * idx + 1]->copy_from(
                  reduction_instances[idx], this,
                  reduction_instances_ready[idx]);
        }
        record_completion_effects(reduction_instances_ready);
      }
      else
        record_completion_effect(reduction_instance_precondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_commit(
        unsigned points, RtEvent commit_precondition)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (commit_precondition.exists())
          commit_preconditions.insert(commit_precondition);
        committed_points += points;
#ifdef DEBUG_LEGION
        assert(committed_points <= total_points);
#endif
        if ((committed_points == total_points) && !children_commit_invoked)
        {
          need_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (need_trigger)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_point_mapped(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DomainPoint point;
      derez.deserialize(point);
      RtEvent mapped_event;
      derez.deserialize(mapped_event);
      return_point_mapped(point, mapped_event);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_complete(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      ApEvent completion_effect;
      derez.deserialize(completion_effect);
      if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(reduction_op != nullptr);
#endif
        if (deterministic_redop)
        {
          size_t num_futures;
          derez.deserialize(num_futures);
          for (unsigned idx = 0; idx < num_futures; idx++)
          {
            DomainPoint point;
            derez.deserialize(point);
            FutureInstance* instance = FutureInstance::unpack_instance(derez);
            ApEvent effects;
            if (!instance->is_meta_visible)
              derez.deserialize(effects);
            reduce_future(point, instance, effects);
          }
        }
        else
        {
          if (serdez_redop_fns != nullptr)
          {
            size_t reduc_size;
            derez.deserialize(reduc_size);
            if (reduc_size > 0)
            {
              const void* reduc_ptr = derez.get_current_pointer();
              FutureInstance instance(
                  reduc_ptr, reduc_size, true /*external*/,
                  false /*own allocation*/);
              fold_reduction_future(&instance, ApEvent::NO_AP_EVENT);
              // Advance the pointer on the deserializer
              derez.advance_pointer(reduc_size);
            }
          }
          else
          {
            DomainPoint point;
            derez.deserialize(point);
            if (point.get_dim() > 0)
            {
              FutureInstance* instance = FutureInstance::unpack_instance(derez);
              ApEvent effects;
              if (!instance->is_meta_visible)
                derez.deserialize(effects);
              reduce_future(point, instance, effects);
            }
          }
        }
        size_t metasize;
        derez.deserialize(metasize);
        if (metasize > 0)
        {
          AutoLock o_lock(op_lock);
          if (reduction_metadata == nullptr)
          {
            reduction_metadata = malloc(metasize);
            memcpy(reduction_metadata, derez.get_current_pointer(), metasize);
            reduction_metasize = metasize;
          }
          derez.advance_pointer(metasize);
        }
      }
      return_slice_complete(points, completion_effect);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_commit(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      RtEvent commit_precondition;
      derez.deserialize(commit_precondition);
      const RtEvent resources_returned =
          (must_epoch == nullptr) ?
              ResourceTracker::unpack_resources_return(derez, parent_ctx) :
              ResourceTracker::unpack_resources_return(derez, must_epoch);
      if (resources_returned.exists())
      {
        if (commit_precondition.exists())
          return_slice_commit(
              points,
              Runtime::merge_events(commit_precondition, resources_returned));
        else
          return_slice_commit(points, resources_returned);
      }
      else
        return_slice_commit(points, commit_precondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_collective_versioning_rendezvous(
        Deserializer& derez, unsigned index, size_t total_points)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      op::map<LogicalRegion, RegionVersioning> to_perform;
      {
        size_t num_regions;
        derez.deserialize(num_regions);
        AutoLock o_lock(op_lock);
        std::map<unsigned, PendingVersioning>::iterator finder =
            pending_versioning.find(index);
        if (finder == pending_versioning.end())
        {
          finder = pending_versioning
                       .insert(std::make_pair(index, PendingVersioning()))
                       .first;
          finder->second.remaining_arrivals = this->get_collective_points();
        }
        for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
        {
          LogicalRegion region;
          derez.deserialize(region);
          RtUserEvent ready_event;
          derez.deserialize(ready_event);
          op::map<LogicalRegion, RegionVersioning>::iterator region_finder =
              finder->second.region_versioning.find(region);
          if (region_finder == finder->second.region_versioning.end())
          {
            region_finder =
                finder->second.region_versioning
                    .emplace(std::make_pair(region, RegionVersioning()))
                    .first;
            region_finder->second.ready_event = ready_event;
          }
          else
            Runtime::trigger_event(
                ready_event, region_finder->second.ready_event);
          size_t num_trackers;
          derez.deserialize(num_trackers);
          for (unsigned idx2 = 0; idx2 < num_trackers; idx2++)
          {
            std::pair<AddressSpaceID, EqSetTracker*> key;
            derez.deserialize(key.first);
            derez.deserialize(key.second);
#ifdef DEBUG_LEGION
            assert(
                region_finder->second.trackers.find(key) ==
                region_finder->second.trackers.end());
#endif
            derez.deserialize(region_finder->second.trackers[key]);
          }
        }
#ifdef DEBUG_LEGION
        assert(finder->second.remaining_arrivals >= total_points);
#endif
        finder->second.remaining_arrivals -= total_points;
        if (finder->second.remaining_arrivals == 0)
        {
          done = true;
          to_perform.swap(finder->second.region_versioning);
          pending_versioning.erase(finder);
        }
        if (num_regions == 0)
        {
          RtUserEvent done_event;
          derez.deserialize(done_event);
          Runtime::trigger_event(done_event);
        }
      }
      if (done)
      {
        const unsigned parent_req_index = find_parent_index(index);
        finalize_collective_versioning_analysis(
            index, parent_req_index, to_perform);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_replaying());
      assert(current_proc.exists());
#endif
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      // If we're going to be doing an output reduction do that now
      if (redop > 0)
      {
        std::vector<Memory> reduction_futures;
        tpl->get_premap_output(this, reduction_futures);
        create_future_instances(reduction_futures);
      }
      else if (!elide_future_return)
      {
        Domain internal_domain;
        runtime->find_domain(internal_space, internal_domain);
        enumerate_futures(internal_domain);
      }
      if (concurrent_task)
        tpl->initialize_concurrent_groups(this);
      // Mark that this is origin mapped effectively in case we
      // have any remote tasks, do this before we clone it
      map_origin = true;
      SliceTask* new_slice =
          this->clone_as_slice_task(internal_space, current_proc, false, false);
      // Count how many total points we need for this index space task
      total_points = new_slice->enumerate_points(false /*inline*/);
      // We need to make one slice per point here in case we need to move
      // points to remote nodes. The way we do slicing right now prevents
      // us from knowing which point tasks are going remote until later in
      // the replay so we have to be pessimistic here
      new_slice->expand_replay_slices(slices);
      // Then do the replay on all the slices
      for (std::list<SliceTask*>::const_iterator it = slices.begin();
           it != slices.end(); it++)
        (*it)->trigger_replay();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_mapped(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexTask* task;
      derez.deserialize(task);
      task->unpack_point_mapped(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_complete(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      IndexTask* task;
      derez.deserialize(task);
      task->unpack_slice_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_commit(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      IndexTask* task;
      derez.deserialize(task);
      task->unpack_slice_commit(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_find_intra_dependence(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* task;
      derez.deserialize(task);
      DomainPoint point;
      derez.deserialize(point);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      task->find_intra_space_dependence(point, to_trigger);
    }

    //--------------------------------------------------------------------------
    void IndexTask::start_check_point_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement& req = regions[idx];
        if (!IS_WRITE(req) || IS_COLLECTIVE(req))
          continue;
        // If the projection functions are invertible then we don't have to
        // worry about interference because the runtime knows how to hook
        // up those kinds of dependences
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
        {
          if (req.projection == 0)
          {
            if (req.handle_type == LEGION_PARTITION_PROJECTION)
            {
              IndexPartNode* partition =
                  runtime->get_node(req.partition.get_index_partition());
              if (partition->is_disjoint())
                continue;
            }
            else  // Identity functor is invertible
              continue;
          }
          else
          {
            ProjectionFunction* func =
                runtime->find_projection_function(req.projection);
            if (func->is_invertible)
              continue;
          }
        }
        interfering_requirements.insert(
            std::pair<unsigned, unsigned>(idx, idx));
      }
      if (interfering_requirements.empty() || launch_space->is_empty())
        return;
      Domain internal_domain;
      if (internal_space.exists())
        runtime->find_domain(internal_space, internal_domain);
      Domain launch_domain = launch_space->get_tight_domain();
      // Exchange all the domains for the interfering requirements
      std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>
          point_domains;
      for (std::set<std::pair<unsigned, unsigned>>::const_iterator rit =
               interfering_requirements.begin();
           rit != interfering_requirements.end(); rit++)
      {
        std::vector<std::pair<DomainPoint, Domain>>& domains =
            point_domains[rit->first];
        // Already found it for this region requirements
        if (!domains.empty())
          continue;
        domains.reserve(internal_domain.get_volume());
        const RegionRequirement& req = regions[rit->first];
        ProjectionFunction* function = nullptr;
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
          function = runtime->find_projection_function(req.projection);
        for (Domain::DomainPointIterator itr(internal_domain); itr; itr++)
        {
          LogicalRegion point_region =
              (function == nullptr) ?
                  req.region :
                  function->project_point(
                      this, rit->first, launch_domain, *itr);
          if (!point_region.exists())
            continue;
          Domain point_domain;
          runtime->find_domain(point_region.get_index_space(), point_domain);
          domains.emplace_back(std::make_pair(*itr, point_domain));
        }
      }
      finish_check_point_requirements(point_domains);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finish_check_point_requirements(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>&
            point_domains)
    //--------------------------------------------------------------------------
    {
      Domain internal_domain;
      if (internal_space.exists())
        runtime->find_domain(internal_space, internal_domain);
      Domain launch_domain = launch_space->get_tight_domain();
      // Iterate our local points and check their first region requirements
      // against all the points in the second region requirements
      for (std::set<std::pair<unsigned, unsigned>>::const_iterator rit =
               interfering_requirements.begin();
           rit != interfering_requirements.end(); rit++)
      {
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>::
            const_iterator finder = point_domains.find(rit->first);
#ifdef DEBUG_LEGION
        assert(finder != point_domains.end());
#endif
        const RegionRequirement& req = get_requirement(rit->second);
        ProjectionFunction* function = nullptr;
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
          function = runtime->find_projection_function(req.projection);
        for (Domain::DomainPointIterator itr(internal_domain); itr; itr++)
        {
          LogicalRegion point_region =
              (function == nullptr) ?
                  req.region :
                  function->project_point(
                      this, rit->second, launch_domain, *itr);
          IndexSpaceNode* node =
              runtime->get_node(point_region.get_index_space());
          DomainPoint interfering;
          if (node->has_interfering_point(
                  finder->second, interfering,
                  (rit->first == rit->second) ? *itr : DomainPoint()))
            Exception(PROGRAMMING_MODEL_EXCEPTION, this)
                << "Index " << *this
                << " has interfering region requirements between "
                << "region requirements " << rit->first << " of point task "
                << interfering << " and " << rit->second << " of point task "
                << *itr
                << ". Interfering region requirements are not permitted for "
                   "index tasks.";
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // OutputExtentExchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OutputExtentExchange::OutputExtentExchange(
        ReplicateContext* ctx, ReplIndexTask* own, CollectiveIndexLocation loc,
        std::vector<OutputExtentMap>& all_extents)
      : AllGatherCollective<false>(loc, ctx), owner(own),
        all_output_extents(all_extents)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    OutputExtentExchange::~OutputExtentExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void OutputExtentExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      rez.serialize(all_output_extents.size());
#endif
      for (std::vector<OutputExtentMap>::const_iterator it =
               all_output_extents.begin();
           it != all_output_extents.end(); ++it)
      {
        rez.serialize(it->size());
        for (OutputExtentMap::const_iterator sit = it->begin();
             sit != it->end(); ++sit)
        {
          rez.serialize(sit->first);
          rez.serialize(sit->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void OutputExtentExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      size_t num_sizes;
      derez.deserialize(num_sizes);
      assert(all_output_extents.size() == num_sizes);
#endif
      for (unsigned idx = 0; idx < all_output_extents.size(); idx++)
      {
        OutputExtentMap& extents = all_output_extents[idx];
        size_t num_entries;
        derez.deserialize(num_entries);
        for (unsigned eidx = 0; eidx < num_entries; eidx++)
        {
          DomainPoint point;
          derez.deserialize(point);
#ifdef DEBUG_LEGION
          DomainPoint size;
          derez.deserialize(size);
          assert(
              (extents.find(point) == extents.end()) ||
              (extents.find(point)->second == size));
          extents[point] = size;
#else
          derez.deserialize(extents[point]);
#endif
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent OutputExtentExchange::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      owner->finalize_output_regions(false /*first invocation*/);
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Concurrent Mapping Rendezvous
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConcurrentMappingRendezvous::ConcurrentMappingRendezvous(
        ReplIndexTask* own, CollectiveIndexLocation loc, ReplicateContext* ctx,
        std::map<Color, MultiTask::ConcurrentGroup>& g)
      : AllGatherCollective<true>(loc, ctx), owner(own), groups(g)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ConcurrentMappingRendezvous::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(groups.size());
      for (std::map<Color, MultiTask::ConcurrentGroup>::iterator git =
               groups.begin();
           git != groups.end(); git++)
      {
        rez.serialize(git->first);
        if (git->second.preconditions.empty())
          rez.serialize(RtEvent::NO_RT_EVENT);
        else if (git->second.preconditions.size() == 1)
          rez.serialize(git->second.preconditions.front());
        else
        {
          RtEvent pre = Runtime::merge_events(git->second.preconditions);
          rez.serialize(pre);
          git->second.preconditions.resize(1);
          git->second.preconditions[0] = pre;
        }
        rez.serialize<size_t>(git->second.shards.size());
        for (std::vector<ShardID>::const_iterator it =
                 git->second.shards.begin();
             it != git->second.shards.end(); it++)
          rez.serialize(*it);
        rez.serialize<size_t>(git->second.processors.size());
        for (std::map<Processor, DomainPoint>::const_iterator it =
                 git->second.processors.begin();
             it != git->second.processors.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        rez.serialize(git->second.color_points);
      }
      rez.serialize<size_t>(trace_barriers.size());
      for (std::map<Color, std::pair<RtBarrier, size_t>>::const_iterator it =
               trace_barriers.begin();
           it != trace_barriers.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second.first);
        rez.serialize(it->second.second);
      }
    }

    //--------------------------------------------------------------------------
    void ConcurrentMappingRendezvous::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_groups;
      derez.deserialize(num_groups);
      for (unsigned idx1 = 0; idx1 < num_groups; idx1++)
      {
        Color color;
        derez.deserialize(color);
        MultiTask::ConcurrentGroup& group = groups[color];
        RtEvent precondition;
        derez.deserialize(precondition);
        if (precondition.exists())
          group.preconditions.emplace_back(precondition);
        size_t num_shards;
        derez.deserialize(num_shards);
        for (unsigned idx2 = 0; idx2 < num_shards; idx2++)
        {
          ShardID shard;
          derez.deserialize(shard);
          if (!std::binary_search(
                  group.shards.begin(), group.shards.end(), shard))
          {
            group.shards.emplace_back(shard);
            std::sort(group.shards.begin(), group.shards.end());
          }
        }
        size_t num_processors;
        derez.deserialize(num_processors);
        for (unsigned idx2 = 0; idx2 < num_processors; idx2++)
        {
          Processor proc;
          derez.deserialize(proc);
          DomainPoint point;
          derez.deserialize(point);
          std::map<Processor, DomainPoint>::const_iterator finder =
              group.processors.find(proc);
          if (finder != group.processors.end())
          {
            // If we are a non-participating shard we might get our
            // own points sent back to us so we need to detect that
            // case and not report an error for it
#ifdef DEBUG_LEGION
            assert(
                (finder->second != point) || (!participating && (stage == -1)));
#endif
            if (finder->second != point)
              owner->report_concurrent_mapping_failure(
                  proc, point, finder->second);
          }
          else
            group.processors.emplace(std::make_pair(proc, point));
        }
        size_t points;
        derez.deserialize(points);
        if (!participating)
        {
#ifdef DEBUG_LEGION
          assert(stage == -1);
#endif
          group.color_points = points;
        }
        else
          group.color_points += points;
      }
      size_t num_barriers;
      derez.deserialize(num_barriers);
      for (unsigned idx = 0; idx < num_barriers; idx++)
      {
        Color color;
        derez.deserialize(color);
#ifdef DEBUG_LEGION
        assert(trace_barriers.find(color) == trace_barriers.end());
#endif
        std::pair<RtBarrier, size_t>& barrier = trace_barriers[color];
        derez.deserialize(barrier.first);
        derez.deserialize(barrier.second);
      }
    }

    //--------------------------------------------------------------------------
    void ConcurrentMappingRendezvous::set_trace_barrier(
        Color color, RtBarrier bar, size_t arrivals)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bar.exists());
      assert(trace_barriers.find(color) == trace_barriers.end());
#endif
      trace_barriers.emplace(
          std::make_pair(color, std::make_pair(bar, arrivals)));
    }

    //--------------------------------------------------------------------------
    void ConcurrentMappingRendezvous::perform_rendezvous(void)
    //--------------------------------------------------------------------------
    {
      // Record our local shard as one of the shards for each group
      // In some cases we might already have computed it so we don't need
      // to add ourselves in that case as we'll already be represented
      for (std::map<Color, MultiTask::ConcurrentGroup>::iterator it =
               groups.begin();
           it != groups.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->second.group_points > 0);
        assert(
            it->second.shards.empty() ||
            std::binary_search(
                it->second.shards.begin(), it->second.shards.end(),
                local_shard));
#endif
        it->second.color_points = it->second.group_points;
        if (it->second.shards.empty())
          it->second.shards.emplace_back(local_shard);
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    RtEvent ConcurrentMappingRendezvous::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      owner->finish_concurrent_mapped(trace_barriers);
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Concurrent Allreduce
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConcurrentAllreduce::ConcurrentAllreduce(
        ReplicateContext* ctx, CollectiveID id, Color c,
        const std::vector<ShardID>& shards)
      : AllGatherCollective<false>(ctx, id, shards), color(c)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ConcurrentAllreduce::ConcurrentAllreduce(
        CollectiveIndexLocation loc, ReplicateContext* ctx)
      : AllGatherCollective<false>(loc, ctx), color(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ConcurrentAllreduce::~ConcurrentAllreduce(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ConcurrentAllreduce::perform_concurrent_allreduce(
        MultiTask::ConcurrentGroup& group)
    //--------------------------------------------------------------------------
    {
      slice_tasks.swap(group.slice_tasks);
      task_barrier = group.task_barrier;
      lamport_clock = group.lamport_clock;
      variant = group.variant;
      poisoned = group.poisoned;
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void ConcurrentAllreduce::perform_concurrent_allreduce(
        std::vector<std::pair<IndividualTask*, AddressSpaceID>>& single,
        std::vector<std::pair<SliceTask*, AddressSpace>>& slices,
        uint64_t clock, bool poison)
    //--------------------------------------------------------------------------
    {
      single_tasks.swap(single);
      slice_tasks.swap(slices);
      lamport_clock = clock;
      variant = 0;
      poisoned = poison;
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void ConcurrentAllreduce::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(task_barrier);
      rez.serialize(lamport_clock);
      rez.serialize(variant);
      rez.serialize<bool>(poisoned);
    }

    //--------------------------------------------------------------------------
    void ConcurrentAllreduce::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      RtBarrier barrier;
      derez.deserialize(barrier);
      if (!task_barrier.exists() && barrier.exists())
        task_barrier = barrier;
      uint64_t clock;
      derez.deserialize(clock);
      if (lamport_clock < clock)
        lamport_clock = clock;
      VariantID vid;
      derez.deserialize(vid);
      if (variant != vid)
        variant = std::min(variant, vid);
      bool poison;
      derez.deserialize<bool>(poison);
      if (poison)
        poisoned = true;
    }

    //--------------------------------------------------------------------------
    RtEvent ConcurrentAllreduce::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      // Pull these onto the stack in order to avoid a race with the caller
      // cleaning up this object
      std::vector<std::pair<IndividualTask*, AddressSpaceID>> local_tasks;
      local_tasks.swap(single_tasks);
      std::vector<std::pair<SliceTask*, AddressSpace>> local_slices;
      local_slices.swap(slice_tasks);
      for (std::vector<
               std::pair<IndividualTask*, AddressSpaceID>>::const_iterator it =
               local_tasks.begin();
           it != local_tasks.end(); it++)
      {
        if (it->second != runtime->address_space)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(it->first);
            rez.serialize(lamport_clock);
            rez.serialize(poisoned);
          }
          runtime->send_individual_concurrent_allreduce_response(
              it->second, rez);
        }
        else
          it->first->finish_concurrent_allreduce(lamport_clock, poisoned);
      }
      for (std::vector<std::pair<SliceTask*, AddressSpaceID>>::const_iterator
               it = local_slices.begin();
           it != local_slices.end(); it++)
      {
        if (it->second != runtime->address_space)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(it->first);
            rez.serialize(color);
            rez.serialize(task_barrier);
            rez.serialize(lamport_clock);
            rez.serialize(variant);
            rez.serialize(poisoned);
          }
          runtime->send_slice_concurrent_allreduce_response(it->second, rez);
        }
        else
          it->first->finish_concurrent_allreduce(
              color, lamport_clock, poisoned, variant, task_barrier);
      }
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexTask::ReplIndexTask(void) : ReplCollectiveViewCreator<IndexTask>()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplIndexTask::~ReplIndexTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplIndexTask::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<IndexTask>::activate();
      sharding_functor = UINT_MAX;
      sharding_function = nullptr;
      serdez_redop_collective = nullptr;
      all_reduce_collective = nullptr;
      reduction_collective = nullptr;
      broadcast_collective = nullptr;
      output_size_collective = nullptr;
      collective_check_id = 0;
      interfering_check_id = 0;
      slice_sharding_output = false;
      output_bar = RtBarrier::NO_RT_BARRIER;
      concurrent_mapping_rendezvous = nullptr;
      interfering_check_id = 0;
      interfering_exchange = nullptr;
      collective_exchange_id = 0;
      collective_exchange = nullptr;
#ifdef DEBUG_LEGION
      sharding_collective = nullptr;
#endif
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      for (std::map<Color, ConcurrentGroup>::iterator it =
               concurrent_groups.begin();
           it != concurrent_groups.end(); it++)
        if (it->second.exchange == nullptr)
          delete it->second.exchange;
      ReplCollectiveViewCreator<IndexTask>::deactivate(false /*free*/);
      if (serdez_redop_collective != nullptr)
        delete serdez_redop_collective;
      if (all_reduce_collective != nullptr)
        delete all_reduce_collective;
      if (reduction_collective != nullptr)
        delete reduction_collective;
      if (broadcast_collective != nullptr)
        delete broadcast_collective;
      if (output_size_collective != nullptr)
        delete output_size_collective;
      if (concurrent_mapping_rendezvous != nullptr)
        delete concurrent_mapping_rendezvous;
      if (interfering_exchange != nullptr)
        delete interfering_exchange;
      concurrent_exchange_ids.clear();
      if (collective_exchange != nullptr)
        delete collective_exchange;
#ifdef DEBUG_LEGION
      if (sharding_collective != nullptr)
        delete sharding_collective;
#endif
      unique_intra_space_deps.clear();
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::prepare_map_must_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
      assert(must_epoch != nullptr);
      assert(sharding_function != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      set_origin_mapped(true);
      total_points = launch_space->get_volume();
      if (!elide_future_return)
      {
        future_map = must_epoch->get_future_map();
        const IndexSpace local_space =
            sharding_space.exists() ?
                sharding_function->find_shard_space(
                    repl_ctx->owner_shard->shard_id, launch_space,
                    sharding_space, get_provenance()) :
                sharding_function->find_shard_space(
                    repl_ctx->owner_shard->shard_id, launch_space,
                    launch_space->handle, get_provenance());
        if (local_space.exists())
        {
          Domain local_domain;
          runtime->find_domain(local_space, local_domain);
          enumerate_futures(local_domain);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif

      // We might be able to skip this if the sharding function was already
      // picked for us which occurs when we're part of a must-epoch launch
      if (sharding_function == nullptr)
        select_sharding_function(repl_ctx);
#ifdef DEBUG_LEGION
      assert(sharding_function != nullptr);
      assert(sharding_collective != nullptr);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
        REPORT_LEGION_ERROR(
            ERROR_INVALID_MAPPER_OUTPUT,
            "Mapper %s chose different sharding functions "
            "for index task %s (UID %lld) in %s (UID %lld)",
            mapper->get_mapper_name(), get_task_name(), get_unique_id(),
            parent_ctx->get_task_name(), parent_ctx->get_unique_id())
#endif
      // Now we can do the normal prepipeline stage
      IndexTask::trigger_prepipeline_stage();
      if (runtime->safe_mapper)
      {
        // Check that all the mappers agreed on the set of
        // collective view region requirements
        if (repl_ctx->owner_shard->shard_id == 0)
        {
          Serializer rez;
          rez.serialize<size_t>(check_collective_regions.size());
          for (std::vector<unsigned>::const_iterator it =
                   check_collective_regions.begin();
               it != check_collective_regions.end(); it++)
            rez.serialize(*it);
          BufferBroadcast collective(collective_check_id, repl_ctx);
          collective.broadcast(
              const_cast<void*>(rez.get_buffer()), rez.get_used_bytes(),
              false /*copy*/);
        }
        else
        {
          BufferBroadcast collective(
              collective_check_id, 0 /*owner*/, repl_ctx);
          size_t size;
          const void* buffer = collective.get_buffer(size);
          Deserializer derez(buffer, size);
          size_t num_regions;
          derez.deserialize(num_regions);
          if (num_regions != check_collective_regions.size())
            REPORT_LEGION_ERROR(
                ERROR_INVALID_MAPPER_OUTPUT,
                "Mapper %s provided different number of logical regions to "
                "check for collective views on shards 0 and %d of task %s "
                "(UID %lld). Shard 0 provided %zd regions while Shard %d "
                "provided %zd regions. All shards must provide the same "
                "logical regions to check for the collective view creation.",
                mapper->get_mapper_name(), repl_ctx->owner_shard->shard_id,
                get_task_name(), get_unique_id(), num_regions,
                repl_ctx->owner_shard->shard_id,
                check_collective_regions.size())
          for (unsigned idx = 0; idx < num_regions; idx++)
          {
            unsigned index;
            derez.deserialize(index);
            if (!std::binary_search(
                    check_collective_regions.begin(),
                    check_collective_regions.end(), index))
              REPORT_LEGION_ERROR(
                  ERROR_INVALID_MAPPER_OUTPUT,
                  "Mapper %s provided different logical regions to check for "
                  "collective views on shards 0 and %d of task %s (UID %lld). "
                  "Shard 0 provided region %d while Shard %d did not. All "
                  "shards must provide the same logical regions to check for "
                  "the collective view creation.",
                  mapper->get_mapper_name(), repl_ctx->owner_shard->shard_id,
                  get_task_name(), get_unique_id(), index,
                  repl_ctx->owner_shard->shard_id)
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::select_sharding_function(ReplicateContext* repl_ctx)
    //--------------------------------------------------------------------------
    {
      // Do the mapper call to get the sharding function to use
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output = {
          std::numeric_limits<ShardingID>::max(), true};
      mapper->invoke_task_select_sharding_functor(this, *input, output);
      if (output.chosen_functor == UINT_MAX)
        REPORT_LEGION_ERROR(
            ERROR_INVALID_MAPPER_OUTPUT,
            "Mapper %s failed to pick a valid sharding functor for "
            "task %s (UID %lld)",
            mapper->get_mapper_name(), get_task_name(), get_unique_id())
      this->sharding_functor = output.chosen_functor;
      sharding_function =
          repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      slice_sharding_output = output.slice_recurse;
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finish_check_point_requirements(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>&
            point_domains)
    //--------------------------------------------------------------------------
    {
      // See if this is the first time through or not
      if (interfering_exchange == nullptr)
      {
        // First time through, make the exchange and kick it off
#ifdef DEBUG_LEGION
        assert(interfering_check_id > 0);
        ReplicateContext* repl_ctx =
            dynamic_cast<ReplicateContext*>(parent_ctx);
        assert(repl_ctx != nullptr);
#else
        ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
        interfering_exchange = new InterferingPointExchange<ReplIndexTask>(
            repl_ctx, interfering_check_id, this);
        // Record a dependence on this to make sure it is done before we
        // clean up the operation
        commit_preconditions.insert(interfering_exchange->get_done_event());
        interfering_exchange->exchange_domain_points(point_domains);
      }
      else  // Second time through call the base class since we have the
            // results
        IndexTask::finish_check_point_requirements(point_domains);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // If we have a future map then set the sharding function
      if ((redop == 0) && !elide_future_return && (must_epoch == nullptr))
      {
#ifdef DEBUG_LEGION
        assert(future_map.impl != nullptr);
        ReplFutureMapImpl* impl =
            dynamic_cast<ReplFutureMapImpl*>(future_map.impl);
        assert(impl != nullptr);
#else
        ReplFutureMapImpl* impl =
            static_cast<ReplFutureMapImpl*>(future_map.impl);
#endif
        impl->set_sharding_function(sharding_function);
      }
      // Compute the local index space of points for this shard
      if (sharding_space.exists())
        internal_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, sharding_space,
            get_provenance());
      else
        internal_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, launch_space->handle,
            get_provenance());
      if (runtime->safe_model)
        start_check_point_requirements();
      // If we're recording then record the local_space
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert(!is_remote());
        assert((tpl != nullptr) && tpl->is_recording());
#endif
        tpl->record_local_space(trace_local_id, internal_space);
        // Record the sharding function if needed for the future map
        if (redop == 0)
          tpl->record_sharding_function(trace_local_id, sharding_function);
      }
      // If it's empty we're done, otherwise we go back on the queue
      if (!internal_space.exists())
      {
        // Check to see if we still need to participate in the premap_task call
        if (must_epoch == nullptr)
          premap_task();
        // Still need to participate in any collective mappings
        if (concurrent_task || !check_collective_regions.empty())
        {
          collective_exchange =
              new AllReduceCollective<MaxReduction<uint64_t>, false>(
                  repl_ctx, collective_exchange_id);
          collective_exchange->async_all_reduce(collective_lamport_clock);
          AutoLock o_lock(op_lock);
          commit_preconditions.insert(collective_exchange->get_done_event());
        }
        // Still need to participate in any collective view rendezvous
        if (!collective_view_rendezvous.empty())
          shard_off_collective_rendezvous(commit_preconditions);
#ifdef LEGION_SPY
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // Finalize any output regions
        if (output_size_collective != nullptr)
        {
          finalize_output_regions(true /*first invocation*/);
          record_output_registered(RtEvent::NO_RT_EVENT);
        }
        // We have no local points, so we can just trigger
        if (serdez_redop_fns == nullptr)
        {
          if (!map_applied_conditions.empty())
            complete_mapping(Runtime::merge_events(map_applied_conditions));
          else
            complete_mapping();
        }
        if (concurrent_task)
          concurrent_mapping_rendezvous->perform_rendezvous();
        if (redop > 0)
          finish_index_task_reduction();
        complete_execution();
        trigger_children_committed();
      }
      else  // We have valid points, so it goes on the ready queue
      {
        // If we had a trace then we might need to recompute our parent
        // region indexes now since we might have skipped it earlier
        if (trace != nullptr)
          compute_parent_indexes(true /*force*/);
        // Update the total number of points we're actually repsonsible
        // for now with this shard
        IndexSpaceNode* node = runtime->get_node(internal_space);
        total_points = node->get_volume();
#ifdef DEBUG_LEGION
        assert(total_points > 0);
#endif
        if ((redop == 0) && !elide_future_return)
        {
          Domain shard_domain = node->get_tight_domain();
          enumerate_futures(shard_domain);
        }
        if (concurrent_task)
        {
          if (concurrent_functor == 0)
          {
            // The built-in concurrent coloring functor makes this easy
            // since it always maps all the points to the same color
            ConcurrentGroup& group = concurrent_groups[0];
            group.precondition.interpreted = Runtime::create_rt_user_event();
            group.color_points = launch_space->get_volume();
          }
          else
          {
            // Not the built-in functor so we actually need to some work
            ConcurrentColoringFunctor* functor =
                runtime->find_concurrent_coloring_functor(concurrent_functor);
            Domain shard_domain = node->get_tight_domain();
            for (Domain::DomainPointIterator itr(shard_domain); itr; itr++)
            {
              Color color = functor->color(*itr, index_domain);
              if (concurrent_groups.find(color) == concurrent_groups.end())
                concurrent_groups[color].precondition.interpreted =
                    Runtime::create_rt_user_event();
            }
          }
        }
        // If we still need to slice the task then we can run it
        // through the normal path, otherwise we can simply make
        // the slice task for these points and put it in the queue
        if (!slice_sharding_output)
        {
          if (must_epoch == nullptr)
            premap_task();
          SliceTask* new_slice = this->clone_as_slice_task(
              internal_space, target_proc, false /*recurse*/,
              !runtime->stealing_disabled);
          slices.emplace_back(new_slice);
          trigger_slices();
        }
        else
          enqueue_ready_operation();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl != nullptr);
#endif
      elide_collective_rendezvous();
      internal_space = tpl->find_local_space(trace_local_id);
      if ((redop == 0) && !elide_future_return)
      {
        sharding_function = tpl->find_sharding_function(trace_local_id);
#ifdef DEBUG_LEGION
        assert(future_map.impl != nullptr);
        ReplFutureMapImpl* impl =
            dynamic_cast<ReplFutureMapImpl*>(future_map.impl);
        assert(impl != nullptr);
#else
        ReplFutureMapImpl* impl =
            static_cast<ReplFutureMapImpl*>(future_map.impl);
#endif
        impl->set_sharding_function(sharding_function);
      }
      // We know all the points are going to be issues so no need for this
      if (concurrent_mapping_rendezvous != nullptr)
        concurrent_mapping_rendezvous->elide_collective();
      // If it's empty we're done, otherwise we do the replay
      if (!internal_space.exists())
      {
#ifdef LEGION_SPY
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We have no local points, so we can just trigger
        if (serdez_redop_fns == nullptr)
        {
          if (!map_applied_conditions.empty())
            complete_mapping(Runtime::merge_events(map_applied_conditions));
          else
            complete_mapping();
        }
        if (redop > 0)
        {
          std::vector<Memory> reduction_futures;
          tpl->get_premap_output(this, reduction_futures);
          create_future_instances(reduction_futures);
          finish_index_task_reduction();
        }
        complete_execution();
        trigger_children_committed();
      }
      else
        IndexTask::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      perform_base_dependence_analysis();
      ShardingFunction* analysis_sharding_function = sharding_function;
      if (must_epoch_task)
      {
        // Note we use a special
        // projection function for must epoch launches that maps all the
        // tasks to the special shard UINT_MAX so that they appear to be
        // on a different shard than any other tasks, but on the same shard
        // for all the tasks in the must epoch launch.
        analysis_sharding_function =
            repl_ctx->get_universal_sharding_function();
      }
      analyze_region_requirements(
          launch_space, analysis_sharding_function, sharding_space);
      if ((concurrent_task && !must_epoch_task) ||
          !check_collective_regions.empty())
      {
        // Create the collective exchange ID in case we need it
        collective_exchange_id = repl_ctx->get_next_collective_index(
            COLLECTIVE_LOC_68, true /*logical*/);
        // Generate any collective view rendezvous that we will need
        for (std::vector<unsigned>::const_iterator it =
                 check_collective_regions.begin();
             it != check_collective_regions.end(); it++)
          create_collective_rendezvous(
              logical_regions[*it].parent.get_tree_id(), *it);
      }
      if (concurrent_task && !must_epoch_task)
      {
#ifdef DEBUG_LEGION
        ReplicateContext* repl_ctx =
            dynamic_cast<ReplicateContext*>(parent_ctx);
        assert(repl_ctx != nullptr);
#else
        ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
        // If this is a concurrent task we need to allocated collective IDs
        // for all colors that the task might need for when it goes to
        // perform its max allreduce. We need to allocate IDs for all colors
        // even if we might only use some of them so that we guarantee that
        // all the shards stay with the same set of collective IDs
        // If the concurrent coloring functor is the built-in one then we
        // know that we only need one ID in that case
        if (concurrent_functor > 0)
        {
          ConcurrentColoringFunctor* functor =
              runtime->find_concurrent_coloring_functor(concurrent_functor);
          if (functor->supports_max_color())
          {
            Color max_color = functor->max_color(index_domain);
            for (Color color = 0; color <= max_color; color++)
              concurrent_exchange_ids[color] =
                  repl_ctx->get_next_collective_index(
                      COLLECTIVE_LOC_79, true /*logical*/);
          }
          else
          {
            // If we have a trace we can try to look these up
            bool found = false;
            if (trace != nullptr)
              found =
                  trace->find_concurrent_colors(this, concurrent_exchange_ids);
            if (!found)
            {
              // Iterate all the points and save the unique colors
              for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
              {
                Color color = functor->color(*itr, index_domain);
                if (concurrent_exchange_ids.find(color) ==
                    concurrent_exchange_ids.end())
                  concurrent_exchange_ids.emplace(
                      std::pair<Color, CollectiveID>(color, 0));
              }
#ifdef DEBUG_LEGION
              assert(!concurrent_exchange_ids.empty());
#endif
              if (trace != nullptr)
                trace->record_concurrent_colors(this, concurrent_exchange_ids);
            }
            for (std::map<Color, CollectiveID>::iterator it =
                     concurrent_exchange_ids.begin();
                 it != concurrent_exchange_ids.end(); it++)
            {
#ifdef DEBUG_LEGION
              assert(it->second == 0);
#endif
              it->second = repl_ctx->get_next_collective_index(
                  COLLECTIVE_LOC_79, true /*logical*/);
            }
          }
        }
        else
          concurrent_exchange_ids[0] = repl_ctx->get_next_collective_index(
              COLLECTIVE_LOC_79, true /*logical*/);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::create_future_instances(
        std::vector<Memory>& target_memories)
    //--------------------------------------------------------------------------
    {
      // Do the base call first
      IndexTask::create_future_instances(target_memories);
      // Now check to see if we need to make a shadow instance for our
      // future all reduce collective
      if (all_reduce_collective != nullptr)
      {
#ifdef DEBUG_LEGION
        assert(!reduction_instances.empty());
        assert(reduction_instance != nullptr);
#endif
        // If the instance is in a memory we cannot see or is "too big"
        // then we need to make the shadow instance for the future
        // all-reduce collective to use now while still in the mapping stage
        if ((!reduction_instance.load()->is_meta_visible) ||
            (reduction_instance.load()->size > LEGION_MAX_RETURN_SIZE))
        {
          MemoryManager* manager =
              runtime->find_memory_manager(reduction_instance.load()->memory);
          TaskTreeCoordinates coordinates;
          compute_task_tree_coordinates(coordinates);
          // Safe to block indefinitely here waiting for unbounded pools
          FutureInstance* shadow_instance = manager->create_future_instance(
              unique_op_id, coordinates, reduction_op->sizeof_rhs,
              nullptr /*safe_for_unbounded_pools*/);
          all_reduce_collective->set_shadow_instance(shadow_instance);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finish_index_task_reduction(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop != 0);
#endif
      // Set the future if we actually ran the task or we speculated
      if (predication_state == PREDICATED_FALSE_STATE)
        return;
      if (serdez_redop_fns != nullptr)
      {
#ifdef DEBUG_LEGION
        assert(serdez_redop_collective != nullptr);
#endif
        const std::map<ShardID, std::pair<void*, size_t>>& remote_buffers =
            serdez_redop_collective->exchange_buffers(
                serdez_redop_state, serdez_redop_state_size,
                deterministic_redop);
        if (deterministic_redop)
        {
          // Reset this back to empty so we can reduce in order across shards
          // Note the serdez_redop_collective took ownership of deleting
          // the buffer in this case so we know that it is not leaking
          serdez_redop_state = nullptr;
          for (std::map<ShardID, std::pair<void*, size_t>>::const_iterator it =
                   remote_buffers.begin();
               it != remote_buffers.end(); it++)
          {
            if (serdez_redop_state == nullptr)
            {
              serdez_redop_state_size = it->second.second;
              serdez_redop_state = malloc(serdez_redop_state_size);
              memcpy(
                  serdez_redop_state, it->second.first,
                  serdez_redop_state_size);
            }
            else
              (*(serdez_redop_fns->fold_fn))(
                  reduction_op, serdez_redop_state, serdez_redop_state_size,
                  it->second.first);
          }
        }
        else
        {
          for (std::map<ShardID, std::pair<void*, size_t>>::const_iterator it =
                   remote_buffers.begin();
               it != remote_buffers.end(); it++)
          {
#ifdef DEBUG_LEGION
            assert(it->first != serdez_redop_collective->local_shard);
#endif
            (*(serdez_redop_fns->fold_fn))(
                reduction_op, serdez_redop_state, serdez_redop_state_size,
                it->second.first);
          }
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(
            (all_reduce_collective != nullptr) ||
            ((reduction_collective != nullptr) &&
             (broadcast_collective != nullptr)));
        assert(!reduction_instances.empty());
        assert(reduction_instance == reduction_instances.front());
#endif
        RtEvent collective_done;
        if (all_reduce_collective == nullptr)
        {
          reduction_collective->async_reduce(
              reduction_instance, reduction_instance_precondition);
          reduction_instance_precondition = broadcast_collective->finished;
          if (broadcast_collective->is_origin())
            collective_done = reduction_collective->get_done_event();
          else
            collective_done = broadcast_collective->async_broadcast(
                reduction_instance, ApEvent::NO_AP_EVENT,
                reduction_collective->get_done_event());
        }
        else
          collective_done = all_reduce_collective->async_reduce(
              reduction_instance, reduction_instance_precondition);
        // No need to do anything with the output local precondition
        // We already added it to the complete_effects when we made
        // the collective at the beginning
        if (collective_done.exists() && !collective_done.has_triggered())
        {
          AutoLock o_lock(op_lock);
          commit_preconditions.insert(collective_done);
        }
      }
      // Now call the base version of this to finish making
      // the instances for the future results
      IndexTask::finish_index_task_reduction();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Otherwise, we need to update the internal space so we only set
      // our local points with the predicate false result
      if (!elide_future_return)
      {
        if (redop == 0)
        {
#ifdef DEBUG_LEGION
          ReplicateContext* repl_ctx =
              dynamic_cast<ReplicateContext*>(parent_ctx);
          assert(repl_ctx != nullptr);
#else
          ReplicateContext* repl_ctx =
              static_cast<ReplicateContext*>(parent_ctx);
#endif
#ifdef DEBUG_LEGION
          assert(sharding_function != nullptr);
          assert(future_map.impl != nullptr);
          ReplFutureMapImpl* impl =
              dynamic_cast<ReplFutureMapImpl*>(future_map.impl);
          assert(impl != nullptr);
#else
          ReplFutureMapImpl* impl =
              static_cast<ReplFutureMapImpl*>(future_map.impl);
#endif
          impl->set_sharding_function(sharding_function);
          // Compute the local index space of points for this shard
          if (sharding_space.exists())
            internal_space = sharding_function->find_shard_space(
                repl_ctx->owner_shard->shard_id, launch_space, sharding_space,
                get_provenance());
          else
            internal_space = sharding_function->find_shard_space(
                repl_ctx->owner_shard->shard_id, launch_space,
                launch_space->handle, get_provenance());
        }
        else
        {
          if (serdez_redop_collective != nullptr)
            serdez_redop_collective->elide_collective();
          if (all_reduce_collective != nullptr)
            all_reduce_collective->elide_collective();
          if (reduction_collective != nullptr)
            reduction_collective->elide_collective();
          if (broadcast_collective != nullptr)
            broadcast_collective->elide_collective();
        }
      }
      if (output_size_collective != nullptr)
      {
        output_size_collective->elide_collective();
        runtime->phase_barrier_arrive(output_bar, 1 /*count*/);
      }
      elide_collective_rendezvous();
      // Now continue through and do the base case
      IndexTask::predicate_false();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(serdez_redop_collective == nullptr);
      assert(all_reduce_collective == nullptr);
      assert(reduction_collective == nullptr);
      assert(broadcast_collective == nullptr);
#endif
      // If we have a reduction op then we need an exchange
      if (!elide_future_return && (redop > 0))
      {
        if (serdez_redop_fns == nullptr)
        {
          if (deterministic_redop)
          {
            broadcast_collective = new FutureBroadcastCollective(
                ctx, COLLECTIVE_LOC_63, 0 /*origin shard*/, this);
            reduction_collective = new FutureReductionCollective(
                ctx, COLLECTIVE_LOC_64, 0 /*origin shard*/, this,
                broadcast_collective, reduction_op, redop);
          }
          else
            all_reduce_collective = new FutureAllReduceCollective(
                this, COLLECTIVE_LOC_53, ctx, redop, reduction_op);
        }
        else
          serdez_redop_collective = new BufferExchange(ctx, COLLECTIVE_LOC_53);
      }
      if (!output_regions.empty())
      {
        bool has_output_region = false;
        for (unsigned idx = 0; idx < output_regions.size(); ++idx)
          if (!output_region_options[idx].valid_requirement())
          {
            has_output_region = true;
            break;
          }
        if (has_output_region)
        {
          output_size_collective = new OutputExtentExchange(
              ctx, this, COLLECTIVE_LOC_29, output_region_extents);
          output_bar = ctx->get_next_output_regions_barrier();
        }
      }
      if (runtime->safe_mapper)
        collective_check_id = ctx->get_next_collective_index(COLLECTIVE_LOC_76);
      if (runtime->safe_model)
        interfering_check_id =
            ctx->get_next_collective_index(COLLECTIVE_LOC_69);
      if (concurrent_task && !must_epoch_task)
      {
        concurrent_mapping_rendezvous = new ConcurrentMappingRendezvous(
            this, COLLECTIVE_LOC_104, ctx, concurrent_groups);
        commit_preconditions.insert(
            concurrent_mapping_rendezvous->get_done_event());
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::set_sharding_function(
        ShardingID functor, ShardingFunction* function)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch != nullptr);
      assert(sharding_function == nullptr);
#endif
      sharding_functor = functor;
      sharding_function = function;
    }

    //--------------------------------------------------------------------------
    FutureMap ReplIndexTask::create_future_map(
        TaskContext* ctx, IndexSpace launch_space, IndexSpace shard_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(ctx);
#endif
      IndexSpaceNode* launch_node = runtime->get_node(launch_space);
      IndexSpaceNode* shard_node =
          ((launch_space == shard_space) || !shard_space.exists()) ?
              launch_node :
              runtime->get_node(shard_space);
      const DistributedID future_map_did = repl_ctx->get_next_distributed_id();
      // Make a replicate future map
      return repl_ctx->shard_manager->deduplicate_future_map_creation(
          repl_ctx, this, launch_node, shard_node, future_map_did,
          get_provenance());
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finalize_concurrent_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(concurrent_task);
      assert(!must_epoch_task);
      assert(concurrent_mapping_rendezvous != nullptr);
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      if (is_recording())
      {
        if (concurrent_functor == 0)
        {
          // The built-in concurrent coloring functor makes this easy
          // since it always maps all the points to the same color
          // See if we're the shard that owns the first point in the
          // launch, if we are then we're the one to make the barrier
          Domain launch_domain = launch_space->get_tight_domain();
          Domain sharding_domain;
          if (sharding_space.exists())
            runtime->find_domain(sharding_space, sharding_domain);
          else
            sharding_domain = launch_domain;
          Domain::DomainPointIterator itr(launch_domain);
          ShardID first_nonempty_shard =
              sharding_function->find_owner(*itr, sharding_domain);
          if (first_nonempty_shard == repl_ctx->owner_shard->shard_id)
          {
            ConcurrentGroup& group = concurrent_groups[0];
            const RtBarrier barrier =
                runtime->create_rt_barrier(group.color_points);
            concurrent_mapping_rendezvous->set_trace_barrier(
                0, barrier, group.color_points);
          }
        }
        else
        {
          // Not the built-in functor so we actually need to some work
          ConcurrentColoringFunctor* functor =
              runtime->find_concurrent_coloring_functor(concurrent_functor);
          // Iterate the whole index domain and find all the points
          // that map to colors that we have on this shard so we can
          // count how many arrivals there are and determine if we're
          // the lowest shard to have a point with this color in which
          // case we're also going to make the barrier for this color
          Domain launch_domain = launch_space->get_tight_domain();
          Domain sharding_domain;
          if (sharding_space.exists())
            runtime->find_domain(sharding_space, sharding_domain);
          else
            sharding_domain = launch_domain;
          for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
          {
            Color color = functor->color(*itr, index_domain);
            std::map<Color, ConcurrentGroup>::iterator finder =
                concurrent_groups.find(color);
            // Not one of our local colors then we don't care about it
            if (finder == concurrent_groups.end())
              continue;
            finder->second.color_points++;
            // Compute the shard for this point
            ShardID color_shard =
                sharding_function->find_owner(*itr, sharding_domain);
            if (!std::binary_search(
                    finder->second.shards.begin(), finder->second.shards.end(),
                    color_shard))
            {
              finder->second.shards.emplace_back(color_shard);
              std::sort(
                  finder->second.shards.begin(), finder->second.shards.end());
            }
          }
          // See if our local shards is the smallest shard and make and
          // record the barrier if we are
          for (std::map<Color, ConcurrentGroup>::const_iterator it =
                   concurrent_groups.begin();
               it != concurrent_groups.end(); it++)
          {
#ifdef DEBUG_LEGION
            assert(!it->second.shards.empty());
#endif
            if (it->second.shards.front() == repl_ctx->owner_shard->shard_id)
            {
              const RtBarrier barrier =
                  runtime->create_rt_barrier(it->second.color_points);
              concurrent_mapping_rendezvous->set_trace_barrier(
                  it->first, barrier, it->second.color_points);
            }
          }
        }
      }
      concurrent_mapping_rendezvous->perform_rendezvous();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finish_concurrent_mapped(
        const std::map<Color, std::pair<RtBarrier, size_t>>& trace_barriers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      for (std::map<Color, ConcurrentGroup>::iterator it =
               concurrent_groups.begin();
           it != concurrent_groups.end(); it++)
      {
        // Skip groups that we don't have any local shards for
        if (it->second.group_points == 0)
          continue;
#ifdef DEBUG_LEGION
        assert(it->second.precondition.interpreted.exists());
        assert(it->second.exchange == nullptr);
#endif
        if (is_recording())
        {
          std::map<Color, std::pair<RtBarrier, size_t>>::const_iterator finder =
              trace_barriers.find(it->first);
#ifdef DEBUG_LEGION
          assert(finder != trace_barriers.end());
          assert(finder->second.second == it->second.color_points);
#endif
          tpl->record_concurrent_group(
              this, it->first, it->second.group_points, it->second.color_points,
              finder->second.first, it->second.shards);
        }
        std::map<Color, CollectiveID>::const_iterator finder =
            concurrent_exchange_ids.find(it->first);
#ifdef DEBUG_LEGION
        assert(finder != concurrent_exchange_ids.end());
#endif
        // Create the max allreduce collective
        it->second.exchange = new ConcurrentAllreduce(
            repl_ctx, finder->second, it->first, it->second.shards);
        if (!it->second.preconditions.empty())
          Runtime::trigger_event(
              it->second.precondition.interpreted,
              Runtime::merge_events(it->second.preconditions));
        else
          Runtime::trigger_event(it->second.precondition.interpreted);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::initialize_concurrent_group(
        Color color, size_t local, size_t global, RtBarrier barrier,
        const std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      IndexTask::initialize_concurrent_group(
          color, local, global, barrier, shards);
      // Create a concurrent exchange for this color
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
      std::map<Color, CollectiveID>::const_iterator id_finder =
          concurrent_exchange_ids.find(color);
#ifdef DEBUG_LEGION
      assert(finder != concurrent_groups.end());
      assert(finder->second.exchange == nullptr);
      assert(id_finder != concurrent_exchange_ids.end());
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      finder->second.shards = shards;
      finder->second.exchange = new ConcurrentAllreduce(
          repl_ctx, id_finder->second, color, finder->second.shards);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::concurrent_allreduce(
        Color color, SliceTask* slice, AddressSpaceID slice_space,
        size_t points, uint64_t lamport_clock, VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      if (must_epoch_task)
      {
#ifdef DEBUG_LEGION
        assert(color == 0);
#endif
        must_epoch->concurrent_allreduce(
            slice, slice_space, points, lamport_clock, poisoned);
        return;
      }
      bool done = false;
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
#ifdef DEBUG_LEGION
      assert(finder != concurrent_groups.end());
#endif
      {
        AutoLock o_lock(op_lock);
        if (finder->second.lamport_clock < lamport_clock)
          finder->second.lamport_clock = lamport_clock;
        if (poisoned)
          finder->second.poisoned = true;
        if (finder->second.slice_tasks.empty())
          finder->second.variant = vid;
        else if (finder->second.variant != vid)
          finder->second.variant = std::min(finder->second.variant, vid);
        finder->second.slice_tasks.emplace_back(
            std::make_pair(slice, slice_space));
#ifdef DEBUG_LEGION
        assert(points <= finder->second.group_points);
#endif
        finder->second.group_points -= points;
        done = (finder->second.group_points == 0);
      }
      if (done)
      {
        if (finder->second.variant > 0)
        {
          VariantImpl* variant =
              runtime->find_variant_impl(task_id, finder->second.variant);
          if (variant->needs_barrier())
          {
#ifdef DEBUG_LEGION
            ReplicateContext* repl_ctx =
                dynamic_cast<ReplicateContext*>(parent_ctx);
            assert(repl_ctx != nullptr);
            assert(!finder->second.shards.empty());
#else
            ReplicateContext* repl_ctx =
                static_cast<ReplicateContext*>(parent_ctx);
#endif
            if (finder->second.shards.front() ==
                repl_ctx->owner_shard->shard_id)
              finder->second.task_barrier =
                  runtime->create_rt_barrier(finder->second.color_points);
          }
        }
        finder->second.exchange->perform_concurrent_allreduce(finder->second);
      }
    }

    //--------------------------------------------------------------------------
    uint64_t ReplIndexTask::collective_lamport_allreduce(
        uint64_t lamport_clock, size_t points, bool need_result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      {
        AutoLock o_lock(op_lock);
        if (collective_lamport_clock < lamport_clock)
          collective_lamport_clock = lamport_clock;
        collective_unbounded_points += points;
#ifdef DEBUG_LEGION
        assert(collective_unbounded_points <= total_points);
#endif
        if (collective_unbounded_points < total_points)
        {
          if (need_result)
          {
            if (collective_exchange == nullptr)
              collective_exchange =
                  new AllReduceCollective<MaxReduction<uint64_t>, false>(
                      repl_ctx, collective_exchange_id);
            o_lock.release();
            collective_exchange->get_done_event().wait();
            return collective_exchange->get_result();
          }
          else
            return collective_lamport_clock;
        }
        // Otherwise we're going to fall through and do the allreduce
        if (collective_exchange == nullptr)
          collective_exchange =
              new AllReduceCollective<MaxReduction<uint64_t>, false>(
                  repl_ctx, collective_exchange_id);
        if (!need_result)
          commit_preconditions.insert(collective_exchange->get_done_event());
      }
      collective_exchange->async_all_reduce(collective_lamport_clock);
      if (need_result)
      {
        collective_exchange->get_done_event().wait();
        return collective_exchange->get_result();
      }
      else
        return collective_lamport_clock;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplIndexTask::find_intra_space_dependence(
        const DomainPoint& point, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sharding_function != nullptr);
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      Domain launch_domain;
      if (sharding_space.exists())
        runtime->find_domain(sharding_space, launch_domain);
      else
        launch_domain = launch_space->get_tight_domain();
      const ShardID point_shard =
          sharding_function->find_owner(point, launch_domain);
      if (point_shard == repl_ctx->owner_shard->shard_id)
      {
        // Sharded locally to this shard
        AutoLock o_lock(op_lock);
        std::map<DomainPoint, RtEvent>::const_iterator finder =
            point_mapped_events.find(point);
        if (finder != point_mapped_events.end())
        {
          if (to_trigger.exists())
          {
            Runtime::trigger_event(to_trigger, finder->second);
            return to_trigger;
          }
          else
            return finder->second;
        }
        else
        {
          std::map<DomainPoint, RtUserEvent>::const_iterator pending_finder =
              pending_pointwise_dependences.find(point);
          if (pending_finder == pending_pointwise_dependences.end())
          {
            if (!to_trigger.exists())
              to_trigger = Runtime::create_rt_user_event();
            pending_pointwise_dependences.emplace(
                std::make_pair(point, to_trigger));
            return to_trigger;
          }
          else
          {
            if (to_trigger.exists())
            {
              Runtime::trigger_event(to_trigger, pending_finder->second);
              return to_trigger;
            }
            else
              return pending_finder->second;
          }
        }
      }
      else  // Send it off to the remote shard to find it
        return repl_ctx->find_pointwise_dependence(
            context_index, point, point_shard, to_trigger);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::record_output_registered(RtEvent registered)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(output_bar.exists());
      assert(!output_regions.empty());
#endif
      // Record it in the set of output events and if we've seen all of them
      // then we can launch the meta-task to do the final regisration
      AutoLock o_lock(op_lock);
      // Can be a no-event if we've been shared off
      if (registered.exists())
        output_preconditions.emplace_back(registered);
#ifdef DEBUG_LEGION
      assert(output_preconditions.size() <= total_points);
#endif
      if (output_preconditions.size() == total_points)
      {
        // Can only do the registration once all registrations are done
        runtime->phase_barrier_arrive(
            output_bar, 1 /*count*/,
            Runtime::merge_events(output_preconditions));
        // Can only mark the EqKDTree ready once all the points are registered
        FinalizeOutputEqKDTreeArgs args(this);
        registered = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, output_bar);
        commit_preconditions.insert(registered);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finalize_output_regions(bool first_invocation)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have an exchange to perform
      if (first_invocation && (output_size_collective != nullptr))
      {
        local_output_extents = output_region_extents;
        // We need to gather output region sizes from all the other shards
        // to determine the sizes of globally indexed output regions
        output_size_collective->perform_collective_async();
        // The collective will call us when it is ready but we still need
        // to make sure that we fold the completion event for the
        // collective back into the completion events
        const RtEvent done_event = output_size_collective->get_done_event();
        if (done_event.exists() && !done_event.has_triggered())
        {
          AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
          // We should still not be complete if we're here
          assert((completed_points < total_points) || (total_points == 0));
#endif
          commit_preconditions.insert(done_event);
        }
        return;
      }
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      if (!repl_ctx->shard_manager->is_first_local_shard(repl_ctx->owner_shard))
        return;
      for (unsigned idx = 0; idx < output_regions.size(); ++idx)
      {
        const OutputOptions& options = output_region_options[idx];
        if (options.valid_requirement())
          continue;
        IndexSpaceNode* parent =
            runtime->get_node(output_regions[idx].parent.get_index_space());
#ifdef DEBUG_LEGION
        validate_output_extents(
            idx, output_regions[idx], output_region_extents[idx]);
#endif
        if (options.global_indexing())
        {
          // For globally indexed output regions, we need to check
          // the alignment between outputs from adjacent point tasks
          // and compute the ranges of subregions via prefix sum.
          IndexPartNode* part = runtime->get_node(
              output_regions[idx].partition.get_index_partition());
          Domain root_domain = compute_global_output_ranges(
              parent, part, output_region_extents[idx],
              local_output_extents[idx]);

          if (parent->set_domain(
                  root_domain, ApEvent::NO_AP_EVENT, false /*take ownership*/))
            delete parent;
        }
        // For locally indexed output regions, sizes of subregions are already
        // set when they are fianlized by the point tasks. So we only need to
        // initialize the root index space by taking a union of subspaces.
        else if (parent->set_output_union(output_region_extents[idx]))
          delete parent;
      }
    }

    //--------------------------------------------------------------------------
    size_t ReplIndexTask::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_node(internal_space)->get_volume();
    }

    //--------------------------------------------------------------------------
    bool ReplIndexTask::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sharding_function != nullptr);
#endif
      if (sharding_space.exists())
        return sharding_function->find_shard_participants(
            launch_space, sharding_space, shards);
      else
        return sharding_function->find_shard_participants(
            launch_space, launch_space->handle, shards);
    }

  }  // namespace Internal
}  // namespace Legion
