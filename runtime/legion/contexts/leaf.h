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

#ifndef __LEGION_LEAF_CONTEXT_H__
#define __LEGION_LEAF_CONTEXT_H__

#include "legion/contexts/context.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LeafContext
     * A context for the execution of a leaf task
     */
    class LeafContext : public TaskContext,
                        public Heapify<LeafContext, CONTEXT_LIFETIME> {
    public:
      LeafContext(
          SingleTask* owner, std::map<Memory, MemoryPool*>&& pools,
          bool inline_task = false);
      LeafContext(const LeafContext& rhs) = delete;
      virtual ~LeafContext(void);
    public:
      LeafContext& operator=(const LeafContext& rhs) = delete;
    public:  // Garbage collection methods
      virtual void notify_local(void) { /* nothing to do */ }
    public:
      // Interface for task contexts
      virtual ContextID get_logical_tree_context(void) const;
      virtual ContextID get_physical_tree_context(void) const;
      virtual void compute_task_tree_coordinates(
          TaskTreeCoordinates& coordinatess) const;
      void inline_child_task(TaskOp* child);
      virtual VariantImpl* select_inline_variant(
          TaskOp* child, const std::vector<PhysicalRegion>& parent_regions,
          std::deque<InstanceSet>& physical_instances);
      virtual bool is_leaf_context(void) const;
      virtual RtEvent find_pointwise_dependence(
          uint64_t context_index, const DomainPoint& point, ShardID shard,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT);
      virtual void return_resources(
          ResourceTracker* target, uint64_t return_index,
          std::set<RtEvent>& preconditions);
      virtual void pack_return_resources(
          Serializer& rez, uint64_t return_index);
      virtual void log_created_requirements(void);
      virtual void report_leaks_and_duplicates(
          std::set<RtEvent>& preconditions);
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(
          const Domain& bounds, bool take_ownership, TypeTag type_tag,
          Provenance* provenance);
      virtual IndexSpace create_index_space(
          const Future& future, TypeTag type_tag, Provenance* provenance);
      virtual IndexSpace create_index_space(
          const std::vector<DomainPoint>& points, Provenance* provenance);
      virtual IndexSpace create_index_space(
          const std::vector<Domain>& rects, Provenance* provenance);
      virtual IndexSpace create_unbound_index_space(
          TypeTag type_tag, Provenance* provenance);
      virtual IndexSpace union_index_spaces(
          const std::vector<IndexSpace>& spaces, Provenance* provenance);
      virtual IndexSpace intersect_index_spaces(
          const std::vector<IndexSpace>& spaces, Provenance* provenance);
      virtual IndexSpace subtract_index_spaces(
          IndexSpace left, IndexSpace right, Provenance* provenance);
      virtual void create_shared_ownership(IndexSpace handle);
      virtual void destroy_index_space(
          IndexSpace handle, const bool unordered, const bool recurse,
          Provenance* provenance);
      virtual void create_shared_ownership(IndexPartition handle);
      virtual void destroy_index_partition(
          IndexPartition handle, const bool unordered, const bool recurse,
          Provenance* provenance);
      virtual IndexPartition create_equal_partition(
          IndexSpace parent, IndexSpace color_space, size_t granularity,
          Color color, Provenance* provenance);
      virtual IndexPartition create_partition_by_weights(
          IndexSpace parent, const FutureMap& weights, IndexSpace color_space,
          size_t granularity, Color color, Provenance* provenance);
      virtual IndexPartition create_partition_by_union(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance);
      virtual IndexPartition create_partition_by_intersection(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance);
      virtual IndexPartition create_partition_by_intersection(
          IndexSpace parent, IndexPartition partition, PartitionKind kind,
          Color color, bool dominates, Provenance* provenance);
      virtual IndexPartition create_partition_by_difference(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance);
      virtual Color create_cross_product_partitions(
          IndexPartition handle1, IndexPartition handle2,
          std::map<IndexSpace, IndexPartition>& handles, PartitionKind kind,
          Color color, Provenance* provenance);
      virtual void create_association(
          LogicalRegion domain, LogicalRegion domain_parent, FieldID domain_fid,
          IndexSpace range, MapperID id, MappingTagID tag,
          const UntypedBuffer& marg, Provenance* prov);
      virtual IndexPartition create_restricted_partition(
          IndexSpace parent, IndexSpace color_space, const void* transform,
          size_t transform_size, const void* extent, size_t extent_size,
          PartitionKind part_kind, Color color, Provenance* provenance);
      virtual IndexPartition create_partition_by_domain(
          IndexSpace parent, const FutureMap& domains, IndexSpace color_space,
          bool perform_intersections, PartitionKind part_kind, Color color,
          Provenance* provenance, bool skip_check = false);
      virtual IndexPartition create_partition_by_field(
          LogicalRegion handle, LogicalRegion parent_priv, FieldID fid,
          IndexSpace color_space, Color color, MapperID id, MappingTagID tag,
          PartitionKind part_kind, const UntypedBuffer& marg, Provenance* prov);
      virtual IndexPartition create_partition_by_image(
          IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* prov);
      virtual IndexPartition create_partition_by_image_range(
          IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* prov);
      virtual IndexPartition create_partition_by_preimage(
          IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* prov);
      virtual IndexPartition create_partition_by_preimage_range(
          IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* prov);
      virtual IndexPartition create_pending_partition(
          IndexSpace parent, IndexSpace color_space, PartitionKind part_kind,
          Color color, Provenance* provenance, bool trust = false);
      virtual IndexSpace create_index_space_union(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, const std::vector<IndexSpace>& handles,
          Provenance* provenance);
      virtual IndexSpace create_index_space_union(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexPartition handle, Provenance* provenance);
      virtual IndexSpace create_index_space_intersection(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, const std::vector<IndexSpace>& handles,
          Provenance* provenance);
      virtual IndexSpace create_index_space_intersection(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexPartition handle, Provenance* provenance);
      virtual IndexSpace create_index_space_difference(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexSpace initial,
          const std::vector<IndexSpace>& handles, Provenance* provenance);
      virtual FieldSpace create_field_space(Provenance* provenance);
      virtual FieldSpace create_field_space(
          const std::vector<size_t>& sizes,
          std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
          Provenance* provenance);
      virtual FieldSpace create_field_space(
          const std::vector<Future>& sizes,
          std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
          Provenance* provenance);
      virtual void create_shared_ownership(FieldSpace handle);
      virtual void destroy_field_space(
          FieldSpace handle, const bool unordered, Provenance* provenance);
      virtual FieldAllocatorImpl* create_field_allocator(
          FieldSpace handle, bool unordered);
      virtual void destroy_field_allocator(FieldSpaceNode* node);
      virtual FieldID allocate_field(
          FieldSpace space, size_t field_size, FieldID fid, bool local,
          CustomSerdezID serdez_id, Provenance* provenance);
      virtual FieldID allocate_field(
          FieldSpace space, const Future& field_size, FieldID fid, bool local,
          CustomSerdezID serdez_id, Provenance* provenance);
      virtual void allocate_local_field(
          FieldSpace space, size_t field_size, FieldID fid,
          CustomSerdezID serdez_id, std::set<RtEvent>& done_events,
          Provenance* provenance);
      virtual void free_field(
          FieldAllocatorImpl* allocator, FieldSpace space, FieldID fid,
          const bool unordered, Provenance* provenance);
      virtual void allocate_fields(
          FieldSpace space, const std::vector<size_t>& sizes,
          std::vector<FieldID>& resuling_fields, bool local,
          CustomSerdezID serdez_id, Provenance* provenance);
      virtual void allocate_fields(
          FieldSpace space, const std::vector<Future>& sizes,
          std::vector<FieldID>& resuling_fields, bool local,
          CustomSerdezID serdez_id, Provenance* provenance);
      virtual void allocate_local_fields(
          FieldSpace space, const std::vector<size_t>& sizes,
          const std::vector<FieldID>& resuling_fields, CustomSerdezID serdez_id,
          std::set<RtEvent>& done_events, Provenance* provenance);
      virtual void free_fields(
          FieldAllocatorImpl* allocator, FieldSpace space,
          const std::set<FieldID>& to_free, const bool unordered,
          Provenance* provenance);
      virtual LogicalRegion create_logical_region(
          IndexSpace index_space, FieldSpace field_space, bool task_local,
          Provenance* provenance, const bool output_region = false);
      virtual void create_shared_ownership(LogicalRegion handle);
      virtual void destroy_logical_region(
          LogicalRegion handle, const bool unordered, Provenance* provenance);
      virtual void reset_equivalence_sets(
          LogicalRegion parent, LogicalRegion region,
          const std::set<FieldID>& fields);
      virtual void get_local_field_set(
          const FieldSpace handle, const std::set<unsigned>& indexes,
          std::set<FieldID>& to_set) const;
      virtual void get_local_field_set(
          const FieldSpace handle, const std::set<unsigned>& indexes,
          std::vector<FieldID>& to_set) const;
    public:
      virtual void add_physical_region(
          const RegionRequirement& req, bool mapped, MapperID mid,
          MappingTagID tag, ApUserEvent& unmap_event, bool virtual_mapped,
          const InstanceSet& physical_instances);
      virtual Future execute_task(
          const TaskLauncher& launcher, std::vector<OutputRequirement>* outputs,
          Provenance* provenance);
      virtual FutureMap execute_index_space(
          const IndexTaskLauncher& launcher,
          std::vector<OutputRequirement>* outputs, Provenance* provenance);
      virtual Future execute_index_space(
          const IndexTaskLauncher& launcher, ReductionOpID redop,
          bool deterministic, std::vector<OutputRequirement>* outputs,
          Provenance* provenance);
      virtual Future reduce_future_map(
          const FutureMap& future_map, ReductionOpID redop, bool deterministic,
          MapperID map_id, MappingTagID tag, Provenance* provenance,
          Future initial_value);
      virtual FutureMap construct_future_map(
          IndexSpace domain, const std::map<DomainPoint, UntypedBuffer>& data,
          Provenance* provenance, bool collective = false, ShardingID sid = 0,
          bool implicit = false, bool check_space = true);
      virtual FutureMap construct_future_map(
          const Domain& domain,
          const std::map<DomainPoint, UntypedBuffer>& data,
          bool collective = false, ShardingID sid = 0, bool implicit = false);
      virtual FutureMap construct_future_map(
          IndexSpace domain, const std::map<DomainPoint, Future>& futures,
          Provenance* provenance, bool collective = false, ShardingID sid = 0,
          bool implicit = false, bool check_space = true);
      virtual FutureMap construct_future_map(
          const Domain& domain, const std::map<DomainPoint, Future>& futures,
          bool collective = false, ShardingID sid = 0, bool implicit = false);
      virtual FutureMap transform_future_map(
          const FutureMap& fm, IndexSpace new_domain,
          TransformFutureMapImpl::PointTransformFnptr fnptr,
          Provenance* provenance);
      virtual FutureMap transform_future_map(
          const FutureMap& fm, IndexSpace new_domain,
          PointTransformFunctor* functor, bool own_functor,
          Provenance* provenance);
      virtual PhysicalRegion map_region(
          const InlineLauncher& launcher, Provenance* provenance);
      virtual ApEvent remap_region(
          const PhysicalRegion& region, Provenance* provenance,
          bool internal = false);
      virtual void unmap_region(PhysicalRegion region);
      virtual void unmap_all_regions(bool external = true);
      virtual void fill_fields(
          const FillLauncher& launcher, Provenance* provenance);
      virtual void fill_fields(
          const IndexFillLauncher& launcher, Provenance* provenance);
      virtual void discard_fields(
          const DiscardLauncher& launcher, Provenance* provenance);
      virtual void issue_copy(
          const CopyLauncher& launcher, Provenance* provenance);
      virtual void issue_copy(
          const IndexCopyLauncher& launcher, Provenance* provenance);
      virtual void issue_acquire(
          const AcquireLauncher& launcher, Provenance* provenance);
      virtual void issue_release(
          const ReleaseLauncher& launcher, Provenance* provenance);
      virtual PhysicalRegion attach_resource(
          const AttachLauncher& launcher, Provenance* provenance);
      virtual ExternalResources attach_resources(
          const IndexAttachLauncher& launcher, Provenance* provenance);
      virtual Future detach_resource(
          PhysicalRegion region, const bool flush, const bool unordered,
          Provenance* provenance = nullptr);
      virtual Future detach_resources(
          ExternalResources resources, const bool flush, const bool unordered,
          Provenance* provenance);
      virtual void progress_unordered_operations(bool end_task = false);
      virtual FutureMap execute_must_epoch(
          const MustEpochLauncher& launcher, Provenance* provenance);
      virtual Future issue_timing_measurement(
          const TimingLauncher& launcher, Provenance* provenance);
      virtual Future select_tunable_value(
          const TunableLauncher& launcher, Provenance* provenance);
      virtual Future issue_mapping_fence(Provenance* provenance);
      virtual Future issue_execution_fence(Provenance* provenance);
      virtual void complete_frame(Provenance* provenance);
      virtual Predicate create_predicate(
          const Future& f, Provenance* provenance);
      virtual Predicate predicate_not(
          const Predicate& p, Provenance* provenance);
      virtual Predicate create_predicate(
          const PredicateLauncher& launcher, Provenance* provenance);
      virtual Future get_predicate_future(
          const Predicate& p, Provenance* provenance);
    public:
      virtual void begin_trace(
          TraceID tid, bool logical_only, bool static_trace,
          const std::set<RegionTreeID>* managed, bool dep,
          Provenance* provenance);
      virtual void end_trace(
          TraceID tid, bool deprecated, Provenance* provenance);
      virtual void record_blocking_call(
          uint64_t future_coordinate, bool invalidate_trace = true);
      virtual void wait_on_future(FutureImpl* future, RtEvent ready);
      virtual void wait_on_future_map(FutureMapImpl* map, RtEvent ready);
    public:
      virtual InnerContext* find_top_context(InnerContext* previous = nullptr);
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement>& clone_requirements,
          const std::vector<ApUserEvent>& unmap_events);
      virtual void invalidate_logical_context(void);
      virtual void invalidate_region_tree_contexts(
          const bool is_top_level_task, std::set<RtEvent>& applied,
          const ShardMapping* mapping = nullptr, ShardID source_shard = 0);
    public:
      virtual FutureInstance* create_task_local_future(
          Memory memory, size_t size, bool silence_warnings = false,
          const char* warning_string = nullptr);
      virtual PhysicalInstance create_task_local_instance(
          Memory memory, Realm::InstanceLayoutGeneric* layout);
      virtual void destroy_task_local_instance(
          PhysicalInstance instance, RtEvent precondition);
      virtual size_t query_available_memory(Memory target);
      virtual void release_memory_pool(Memory target);
    public:
      virtual void end_task(
          const void* res, size_t res_size, bool owned, PhysicalInstance inst,
          FutureFunctor* callback_functor,
          const Realm::ExternalInstanceResource* resource,
          void (*freefunc)(const Realm::ExternalInstanceResource&),
          const void* metadataptr, size_t metadatasize, ApEvent effects);
      virtual void post_end_task(void);
      virtual RtEvent escape_task_local_instance(
          PhysicalInstance instance, RtEvent effects, size_t num_results,
          PhysicalInstance* results, LgEvent* unique_events,
          const Realm::InstanceLayoutGeneric** layouts = nullptr);
      virtual void release_task_local_instances(
          ApEvent effects, RtEvent safe_effects);
      virtual void handle_mispredication(void);
    public:
      virtual void destroy_lock(Lock l);
      virtual Grant acquire_grant(const std::vector<LockRequest>& requests);
      virtual void release_grant(Grant grant);
    public:
      virtual void destroy_phase_barrier(PhaseBarrier pb);
    public:
      virtual DynamicCollective create_dynamic_collective(
          unsigned arrivals, ReductionOpID redop, const void* init_value,
          size_t init_size);
      virtual void destroy_dynamic_collective(DynamicCollective dc);
      virtual void arrive_dynamic_collective(
          DynamicCollective dc, const void* buffer, size_t size,
          unsigned count);
      virtual void defer_dynamic_collective_arrival(
          DynamicCollective dc, const Future& future, unsigned count);
      virtual Future get_dynamic_collective_result(
          DynamicCollective dc, Provenance* provenance);
      virtual DynamicCollective advance_dynamic_collective(
          DynamicCollective dc);
    public:
      virtual TaskPriority get_current_priority(void) const;
      virtual void set_current_priority(TaskPriority priority);
    protected:
      // These are the memory pools for doing immediate allocations since
      // this leaf task has already considered itself as mapped and therefore
      // cannot use the normal allocation pathway
#ifndef LEGION_DEBUG
      const
#endif
          std::map<Memory, MemoryPool*>
              memory_pools;
    protected:
      mutable LocalLock leaf_lock;
      size_t inlined_tasks;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_LEAF_CONTEXT_H__
