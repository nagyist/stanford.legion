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

#include "legion/operations/discard.h"
#include "legion/analysis/filter.h"
#include "legion/contexts/replicate.h"
#include "legion/managers/shard.h"
#include "legion/tracing/recognizer.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Discard Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DiscardOp::DiscardOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    DiscardOp::~DiscardOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void DiscardOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      parent_req_index = 0;
    }

    //--------------------------------------------------------------------------
    void DiscardOp::deactivate(bool free)
    //--------------------------------------------------------------------------
    {
      Operation::deactivate(false /*free*/);
      requirement.privilege_fields.clear();
      version_info.clear();
      map_applied_conditions.clear();
      if (free)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* DiscardOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DISCARD_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind DiscardOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DISCARD_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t DiscardOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    unsigned DiscardOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx == 0);
      assert(parent_req_index != TRACED_PARENT_INDEX);
#endif
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void DiscardOp::initialize(
        InnerContext* ctx, const DiscardLauncher& launcher,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      requirement.region = launcher.handle;
      requirement.parent = launcher.parent;
      requirement.privilege = LEGION_WRITE_DISCARD;
      requirement.prop = LEGION_EXCLUSIVE;
      requirement.handle_type = LEGION_SINGULAR_PROJECTION;
      requirement.privilege_fields = launcher.fields;
      if (runtime->safe_model)
        verify_requirement(requirement);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_discard_operation(
            parent_ctx->get_unique_id(), unique_op_id);
    }

    //--------------------------------------------------------------------------
    void DiscardOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled)
        log_requirement();
    }

    //--------------------------------------------------------------------------
    void DiscardOp::log_requirement(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_logical_requirement(
          unique_op_id, 0 /*index*/, true /*region*/,
          requirement.region.index_space.get_id(),
          requirement.region.field_space.get_id(),
          requirement.region.get_tree_id(), requirement.privilege,
          requirement.prop, requirement.redop,
          requirement.parent.index_space.get_id());
      LegionSpy::log_requirement_fields(
          unique_op_id, 0 /*index*/, requirement.privilege_fields);
    }

    //--------------------------------------------------------------------------
    void DiscardOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void DiscardOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void DiscardOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const PhysicalTraceInfo trace_info(this, 0 /*idx*/);
      discard_fields(trace_info);
      if (!map_applied_conditions.empty())
        complete_mapping(finalize_complete_mapping(
            Runtime::merge_events(map_applied_conditions)));
      else
        complete_mapping(finalize_complete_mapping(RtEvent::NO_RT_EVENT));
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void DiscardOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    //--------------------------------------------------------------------------
    bool DiscardOp::record_trace_hash(
        TraceRecognizer& recognizer, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return recognizer.record_operation_noop(this);
    }

    //--------------------------------------------------------------------------
    void DiscardOp::discard_fields(const PhysicalTraceInfo& trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      RegionNode* region = runtime->get_node(requirement.region);
      FilterAnalysis* analysis =
          new FilterAnalysis(this, 0 /*index*/, region, trace_info);
      analysis->add_reference();
      // Still need to pretend to convert an empty set of views to get
      // the collective mapping initialized properly
      const InstanceSet empty_instances;
      const RtEvent views_ready = analysis->convert_views(
          requirement.region, empty_instances, nullptr /*sources*/,
          nullptr /*usage*/, false /*rendezvous*/);
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_conditions);
      // Send out any remote updates
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_conditions);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Repl Discard Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplDiscardOp::ReplDiscardOp(void)
      : ReplCollectiveVersioning<CollectiveVersioning<DiscardOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplDiscardOp::~ReplDiscardOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplDiscardOp::initialize_replication(
        ReplicateContext* ctx, bool is_first_local)
    //--------------------------------------------------------------------------
    {
      is_first_local_shard = is_first_local;
    }

    //--------------------------------------------------------------------------
    void ReplDiscardOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveVersioning<CollectiveVersioning<DiscardOp> >::activate();
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      is_first_local_shard = false;
    }

    //--------------------------------------------------------------------------
    void ReplDiscardOp::deactivate(bool free)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Make sure we didn't leak our barrier
      assert(!collective_map_barrier.exists());
#endif
      ReplCollectiveVersioning<CollectiveVersioning<DiscardOp> >::deactivate(
          false /*free*/);
      if (free)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplDiscardOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      collective_map_barrier = repl_ctx->get_next_collective_map_barriers();
      create_collective_rendezvous(0 /*requirement index*/);
      // Then do the base class analysis
      DiscardOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplDiscardOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_map_barrier.exists());
#endif
      // Signal that all of our mapping dependences are satisfied
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions,
          nullptr /*output region*/, true /*rendezvous*/);
      if (!collective_map_barrier.has_triggered())
        preconditions.insert(collective_map_barrier);
      Runtime::advance_barrier(collective_map_barrier);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    RtEvent ReplDiscardOp::finalize_complete_mapping(RtEvent pre)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_map_barrier.exists());
#endif
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/, pre);
#ifdef DEBUG_LEGION
      const RtEvent result = collective_map_barrier;
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      return result;
#else
      return collective_map_barrier;
#endif
    }

    //--------------------------------------------------------------------------
    bool ReplDiscardOp::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext* repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != nullptr);
#else
      ReplicateContext* repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      mapping = &(repl_ctx->shard_manager->get_collective_mapping());
      mapping->add_reference();
      first_local = is_first_local_shard;
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplDiscardOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Remote Discard Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteDiscardOp::RemoteDiscardOp(Operation* ptr, AddressSpaceID src)
      : RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteDiscardOp::~RemoteDiscardOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteDiscardOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteDiscardOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteDiscardOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteDiscardOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteDiscardOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DISCARD_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteDiscardOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DISCARD_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteDiscardOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
    }

    //--------------------------------------------------------------------------
    void RemoteDiscardOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Nothing for the moment
    }

  }  // namespace Internal
}  // namespace Legion
