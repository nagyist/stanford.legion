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

#ifndef __LEGION_INNER_CONTEXT_H__
#define __LEGION_INNER_CONTEXT_H__

#include "legion/contexts/context.h"
#include "legion/instances/physical.h"
#include "legion/operations/collective.h"
#include "legion/tasks/single.h"
#include "legion/utilities/fieldmask_set.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    class InnerContext : public TaskContext,
                         public ResourceTracker,
                         public InstanceDeletionSubscriber,
                         public Heapify<InnerContext, CONTEXT_LIFETIME> {
    public:
      enum ReplayStatus {
        TRACE_NOT_REPLAYING = 0,
        TRACE_REPLAYING = 1,
      };
      struct ReorderBufferEntry {
      public:
        inline ReorderBufferEntry(Operation* op, size_t index)
          : operation(op), operation_index(index), complete(false),
            child_complete(false)
        { }
      public:
        Operation* operation;
        uint64_t operation_index;
        ApEvent complete_event;
        bool complete;
        bool child_complete;
      };
    public:
      // Prepipeline stages need to hold a reference since the
      // logical analysis could clean the context up before it runs
      struct PrepipelineArgs : public LgTaskArgs<PrepipelineArgs> {
      public:
        static const LgTaskID TASK_ID = LG_PRE_PIPELINE_ID;
      public:
        PrepipelineArgs(Operation* op, InnerContext* ctx)
          : LgTaskArgs<PrepipelineArgs>(op->get_unique_op_id()), context(ctx)
        { }
      public:
        InnerContext* const context;
      };
      struct DependenceArgs : public LgTaskArgs<DependenceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_DEPENDENCE_ID;
      public:
        DependenceArgs(Operation* op, InnerContext* ctx)
          : LgTaskArgs<DependenceArgs>(op->get_unique_op_id()), context(ctx)
        { }
      public:
        InnerContext* const context;
      };
      struct TriggerReadyArgs : public LgTaskArgs<TriggerReadyArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_READY_ID;
      public:
        TriggerReadyArgs(Operation* op, InnerContext* ctx)
          : LgTaskArgs<TriggerReadyArgs>(op->get_unique_op_id()), context(ctx)
        { }
      public:
        InnerContext* const context;
      };
      struct DeferredEnqueueTaskArgs
        : public LgTaskArgs<DeferredEnqueueTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_ENQUEUE_TASK_ID;
      public:
        DeferredEnqueueTaskArgs(
            SingleTask* t, InnerContext* ctx, RtEvent pre, long long perf)
          : LgTaskArgs<DeferredEnqueueTaskArgs>(t->get_unique_op_id()),
            context(ctx), precondition(pre), previous_fevent(implicit_fevent),
            performed(perf)
        { }
      public:
        InnerContext* const context;
        const RtEvent precondition;
        const LgEvent previous_fevent;
        const long long performed;
      };
      struct TriggerExecutionArgs : public LgTaskArgs<TriggerExecutionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_EXECUTION_ID;
      public:
        TriggerExecutionArgs(
            Operation* op, InnerContext* ctx, RtEvent pre, long long perf)
          : LgTaskArgs<TriggerExecutionArgs>(op->get_unique_op_id()),
            context(ctx), precondition(pre), previous_fevent(implicit_fevent),
            performed(perf)
        { }
      public:
        InnerContext* const context;
        const RtEvent precondition;
        const LgEvent previous_fevent;
        const long long performed;
      };
      struct DeferredExecutionArgs : public LgTaskArgs<DeferredExecutionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_EXECUTION_ID;
      public:
        DeferredExecutionArgs(
            Operation* op, InnerContext* ctx, RtEvent pre, long long perf)
          : LgTaskArgs<DeferredExecutionArgs>(op->get_unique_op_id()),
            context(ctx), precondition(pre), previous_fevent(implicit_fevent),
            performed(perf)
        { }
      public:
        InnerContext* const context;
        const RtEvent precondition;
        const LgEvent previous_fevent;
        const long long performed;
      };
      struct DeferredMappedArgs : public LgTaskArgs<DeferredMappedArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_MAPPED_ID;
      public:
        DeferredMappedArgs(
            Operation* op, InnerContext* ctx, RtEvent pre, long long perf)
          : LgTaskArgs<DeferredMappedArgs>(op->get_unique_op_id()),
            context(ctx), precondition(pre), previous_fevent(implicit_fevent),
            performed(perf)
        { }
      public:
        InnerContext* const context;
        const RtEvent precondition;
        const LgEvent previous_fevent;
        const long long performed;
      };
      struct DeferredCompletionArgs
        : public LgTaskArgs<DeferredCompletionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COMPLETION_ID;
      public:
        DeferredCompletionArgs(
            Operation* op, InnerContext* ctx, RtEvent pre, long long perf)
          : LgTaskArgs<DeferredCompletionArgs>(op->get_unique_op_id()),
            context(ctx), precondition(pre), previous_fevent(implicit_fevent),
            performed(perf)
        { }
      public:
        InnerContext* const context;
        const RtEvent precondition;
        const LgEvent previous_fevent;
        const long long performed;
      };
      struct TriggerCommitArgs : public LgTaskArgs<TriggerCommitArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_COMMIT_ID;
      public:
        TriggerCommitArgs(Operation* op, InnerContext* ctx)
          : LgTaskArgs<TriggerCommitArgs>(op->get_unique_op_id()), context(ctx)
        { }
      public:
        InnerContext* const context;
      };
      struct DeferredCommitArgs : public LgTaskArgs<DeferredCommitArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COMMIT_ID;
      public:
        DeferredCommitArgs(
            const std::pair<Operation*, bool>& op, InnerContext* ctx,
            RtEvent pre, long long perf)
          : LgTaskArgs<DeferredCommitArgs>(op.first->get_unique_op_id()),
            context(ctx), precondition(pre), previous_fevent(implicit_fevent),
            performed(perf)
        { }
      public:
        InnerContext* const context;
        const RtEvent precondition;
        const LgEvent previous_fevent;
        const long long performed;
      };
      struct VerifyPartitionArgs : public LgTaskArgs<VerifyPartitionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_VERIFY_PARTITION_TASK_ID;
      public:
        VerifyPartitionArgs(
            InnerContext* proxy, IndexPartition p, PartitionKind k,
            const char* f)
          : LgTaskArgs<VerifyPartitionArgs>(proxy->get_unique_id()),
            proxy_this(proxy), pid(p), kind(k), func(f)
        { }
      public:
        InnerContext* const proxy_this;
        const IndexPartition pid;
        const PartitionKind kind;
        const char* const func;
      };
      template<typename T>
      struct QueueEntry {
      public:
        QueueEntry(void) { op = {}; }
        QueueEntry(T o, RtEvent r) : op(o), ready(r) { }
      public:
        T op;
        RtEvent ready;
      };
      struct CompletionEntry {
      public:
        CompletionEntry(void) : op(nullptr) { }
        CompletionEntry(Operation* o, ApEvent e) : op(o), effects(e) { }
      public:
        Operation* op;
        ApEvent effects;
      };
      struct LocalFieldInfo {
      public:
        LocalFieldInfo(void)
          : fid(0), size(0), serdez(0), index(0), ancestor(false)
        { }
        LocalFieldInfo(
            FieldID f, size_t s, CustomSerdezID z, unsigned idx, bool a)
          : fid(f), size(s), serdez(z), index(idx), ancestor(a)
        { }
      public:
        FieldID fid;
        size_t size;
        CustomSerdezID serdez;
        unsigned index;
        bool ancestor;
      };
      class AttachProjectionFunctor : public ProjectionFunctor {
      public:
        AttachProjectionFunctor(
            ProjectionID pid, std::vector<IndexSpace>&& spaces);
        virtual ~AttachProjectionFunctor(void) { }
      public:
        using ProjectionFunctor::project;
        virtual LogicalRegion project(
            LogicalRegion upper_bound, const DomainPoint& point,
            const Domain& launch_domain);
        virtual LogicalRegion project(
            LogicalPartition upper_bound, const DomainPoint& point,
            const Domain& launch_domain);
      public:
        virtual bool is_functional(void) const { return true; }
        // Some depth >0 means the runtime can't analyze it
        virtual unsigned get_depth(void) const { return UINT_MAX; }
      public:
        static unsigned compute_offset(
            const DomainPoint& point, const Domain& launch);
      public:
        const std::vector<IndexSpace> handles;
        const ProjectionID pid;
      };
      typedef CollectiveViewCreatorBase::CollectiveResult CollectiveResult;
    public:
      InnerContext(
          const Mapper::ContextConfigOutput& config, SingleTask* owner,
          int depth, bool full_inner,
          const std::vector<RegionRequirement>& reqs,
          const std::vector<OutputRequirement>& output_reqs,
          const std::vector<unsigned>& parent_indexes,
          const std::vector<bool>& virt_mapped, TaskPriority priority,
          ApEvent execution_fence, DistributedID did = 0,
          bool inline_task = false, bool implicit_task = false,
          bool concurrent_task = false, CollectiveMapping* mapping = nullptr);
      InnerContext(const InnerContext& rhs) = delete;
      virtual ~InnerContext(void);
    public:
      InnerContext& operator=(const InnerContext& rhs) = delete;
    public:
      inline unsigned get_max_trace_templates(void) const
      {
        return context_configuration.max_templates_per_trace;
      }
      void record_physical_trace_replay(RtEvent ready, bool replay);
      bool is_replaying_physical_trace(void);
      inline bool is_concurrent_context(void) const
      {
        return concurrent_context;
      }
    public:  // Garbage collection methods
      virtual void notify_local(void);
    public:  // Privilege tracker methods
      virtual void receive_resources(
          uint64_t return_index,
          std::map<LogicalRegion, unsigned>& created_regions,
          std::vector<DeletedRegion>& deleted_regions,
          std::set<std::pair<FieldSpace, FieldID> >& created_fields,
          std::vector<DeletedField>& deleted_fields,
          std::map<FieldSpace, unsigned>& created_field_spaces,
          std::map<FieldSpace, std::set<LogicalRegion> >& latent_spaces,
          std::vector<DeletedFieldSpace>& deleted_field_spaces,
          std::map<IndexSpace, unsigned>& created_index_spaces,
          std::vector<DeletedIndexSpace>& deleted_index_spaces,
          std::map<IndexPartition, unsigned>& created_partitions,
          std::vector<DeletedPartition>& deleted_partitions,
          std::set<RtEvent>& preconditions);
    public:
      LogicalRegion find_logical_region(unsigned index);
      int find_parent_region_req(
          const RegionRequirement& req, bool check_privilege = true);
      unsigned find_parent_region_index(
          Operation* op, const RegionRequirement& req, unsigned index = 0,
          bool skip_privileges = false, bool force_compute = false);
      LegionErrorType check_privilege(
          const RegionRequirement& req, FieldID& bad_field, int& bad_index,
          bool skip_privileges = false) const;
    public:
      unsigned add_created_region(
          LogicalRegion handle, bool task_local, bool output_region = false);
      // for logging created region requirements
      virtual void log_created_requirements(void);
      virtual void report_leaks_and_duplicates(
          std::set<RtEvent>& preconditions);
    public:
      unsigned register_region_creation(
          LogicalRegion handle, bool task_local, bool output_region);
    public:
      void register_field_creation(FieldSpace space, FieldID fid, bool local);
      void register_all_field_creations(
          FieldSpace space, bool local, const std::vector<FieldID>& fields);
    public:
      void register_field_space_creation(FieldSpace space);
    public:
      bool has_created_index_space(IndexSpace space) const;
      void register_index_space_creation(IndexSpace space);
    public:
      void register_index_partition_creation(IndexPartition handle);
    public:
      void analyze_destroy_fields(
          FieldSpace handle, const std::set<FieldID>& to_delete,
          std::vector<RegionRequirement>& delete_reqs,
          std::vector<unsigned>& parent_req_indexes,
          std::vector<FieldID>& global_to_free,
          std::vector<FieldID>& local_to_free,
          std::vector<FieldID>& local_field_indexes,
          std::vector<unsigned>& deletion_req_indexes);
      void analyze_destroy_logical_region(
          LogicalRegion handle, std::vector<RegionRequirement>& delete_reqs,
          std::vector<unsigned>& parent_req_indexes,
          std::vector<bool>& returnable_privileges);
      void analyze_free_local_fields(
          FieldSpace handle, const std::vector<FieldID>& local_to_free,
          std::vector<unsigned>& local_field_indexes);
      void remove_deleted_local_fields(
          FieldSpace space, const std::vector<FieldID>& to_remove);
    protected:
      void register_region_creations(
          std::map<LogicalRegion, unsigned>& regions);
      void register_region_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedRegion>& regions,
          std::set<RtEvent>& preconditions);
      void register_field_creations(
          std::set<std::pair<FieldSpace, FieldID> >& fields);
      void register_field_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedField>& fields, std::set<RtEvent>& preconditions);
      void register_field_space_creations(
          std::map<FieldSpace, unsigned>& spaces);
      void register_latent_field_spaces(
          std::map<FieldSpace, std::set<LogicalRegion> >& spaces);
      void register_field_space_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedFieldSpace>& spaces,
          std::set<RtEvent>& preconditions);
      void register_index_space_creations(
          std::map<IndexSpace, unsigned>& spaces);
      void register_index_space_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedIndexSpace>& spaces,
          std::set<RtEvent>& preconditions);
      void register_index_partition_creations(
          std::map<IndexPartition, unsigned>& parts);
      void register_index_partition_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedPartition>& parts,
          std::set<RtEvent>& preconditions);
      void compute_return_deletion_dependences(
          uint64_t return_index,
          std::map<Operation*, GenerationID>& dependences);
    public:
      int has_conflicting_regions(
          MapOp* map, bool& parent_conflict, bool& inline_conflict);
      int has_conflicting_regions(
          AttachOp* attach, bool& parent_conflict, bool& inline_conflict);
      int has_conflicting_internal(
          const RegionRequirement& req, bool& parent_conflict,
          bool& inline_conflict);
      void find_conflicting_regions(
          TaskOp* task, std::vector<PhysicalRegion>& conflicting);
      void find_conflicting_regions(
          CopyOp* copy, std::vector<PhysicalRegion>& conflicting);
      void find_conflicting_regions(
          AcquireOp* acquire, std::vector<PhysicalRegion>& conflicting);
      void find_conflicting_regions(
          ReleaseOp* release, std::vector<PhysicalRegion>& conflicting);
      void find_conflicting_regions(
          DependentPartitionOp* partition,
          std::vector<PhysicalRegion>& conflicting);
      void find_conflicting_internal(
          const RegionRequirement& req,
          std::vector<PhysicalRegion>& conflicting);
      void find_conflicting_regions(
          FillOp* fill, std::vector<PhysicalRegion>& conflicting);
      void find_conflicting_regions(
          DiscardOp* fill, std::vector<PhysicalRegion>& conflicting);
      void register_inline_mapped_region(const PhysicalRegion& region);
      void unregister_inline_mapped_region(const PhysicalRegion& region);
    public:
      void print_children(void);
    public:
      // Interface for task contexts
      virtual ContextID get_logical_tree_context(void) const;
      virtual ContextID get_physical_tree_context(void) const;
      virtual bool is_inner_context(void) const;
      virtual void pack_remote_context(
          Serializer& rez, AddressSpaceID target, bool replicate = false);
      virtual void compute_task_tree_coordinates(
          TaskTreeCoordinates& coordinates) const;
      virtual RtEvent compute_equivalence_sets(
          unsigned req_index, const std::vector<EqSetTracker*>& targets,
          const std::vector<AddressSpaceID>& target_spaces,
          AddressSpaceID creation_target_space, IndexSpaceExpression* expr,
          const FieldMask& mask);
      virtual RtEvent record_output_equivalence_set(
          EqSetTracker* source, AddressSpaceID source_space, unsigned req_index,
          EquivalenceSet* set, const FieldMask& mask);
      EqKDTree* find_equivalence_set_kd_tree(
          unsigned req_index, LocalLock*& tree_lock,
          bool return_null_if_doesnt_exist = false);
      EqKDTree* find_or_create_output_set_kd_tree(
          unsigned req_index, LocalLock*& tree_lock);
      void finalize_output_eqkd_tree(unsigned req_index);
      // This method must be called while holding the privilege lock
      IndexSpace find_root_index_space(unsigned req_index);
      RtEvent report_equivalence_sets(
          unsigned req_index, const CollectiveMapping& target_mapping,
          const std::vector<EqSetTracker*>& targets,
          const AddressSpaceID creation_target_space, const FieldMask& mask,
          std::vector<unsigned>& new_target_references,
          FieldMaskSet<EquivalenceSet>& eq_sets,
          FieldMaskSet<EqKDTree>& new_subscriptions,
          FieldMaskSet<EqKDTree>& to_create,
          op::map<EqKDTree*, Domain>& creation_rects,
          op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
          size_t expected_responses, std::vector<RtEvent>& ready_events);
      RtEvent report_output_registrations(
          EqSetTracker* target, AddressSpaceID target_space,
          unsigned references, FieldMaskSet<EqKDTree>& new_subscriptions);
      virtual EqKDTree* create_equivalence_set_kd_tree(IndexSpaceNode* node);
    public:
      bool inline_child_task(TaskOp* child);
      virtual void return_resources(
          ResourceTracker* target, uint64_t return_index,
          std::set<RtEvent>& preconditions);
      virtual void pack_return_resources(
          Serializer& rez, uint64_t return_index);
    protected:
      IndexSpace create_index_space_internal(
          const Domain& bounds, TypeTag type_tag, Provenance* provenance,
          bool take_ownership);
      RtEvent create_pending_partition_internal(
          IndexPartition pid, IndexSpace parent, IndexSpace color_space,
          LegionColor& partition_color, PartitionKind part_kind,
          Provenance* provenance, CollectiveMapping* mapping = nullptr,
          RtEvent initialized = RtEvent::NO_RT_EVENT);
      void create_pending_cross_product_internal(
          IndexPartition handle1, IndexPartition handle2,
          std::map<IndexSpace, IndexPartition>& user_handles,
          PartitionKind kind, Provenance* provenance, LegionColor& part_color,
          std::set<RtEvent>& safe_events, ShardID shard = 0,
          const ShardMapping* mapping = nullptr,
          ValueBroadcast<LegionColor>* color_broadcast = nullptr);
      IndexSpace instantiate_subspace(
          IndexPartition parent, const void* realm_color, TypeTag type_tag);
    public:
      // Find an index space name for a concrete launch domain
      IndexSpace find_index_launch_space(
          const Domain& domain, Provenance* provenance,
          bool take_ownership = false);
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
      virtual void verify_partition(
          IndexPartition pid, PartitionKind kind, const char* function_name);
      static void handle_partition_verification(const void* args);
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
      virtual FieldAllocatorImpl* create_field_allocator(
          FieldSpace handle, bool unordered);
      virtual void destroy_field_allocator(FieldSpaceNode* node);
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
      virtual RegionTreeNode* compute_index_attach_upper_bound(
          const IndexAttachLauncher& launcher,
          const std::vector<unsigned>& indexes);
      ProjectionID compute_index_attach_projection(
          IndexTreeNode* node, IndexAttachOp* op, unsigned local_start,
          size_t local_size, std::vector<IndexSpace>& spaces,
          const bool can_use_identity = false);
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
      virtual PredicateImpl* create_predicate_impl(Operation* op);
    public:
      // Must be called while holding the dependence lock
      virtual void insert_unordered_ops(AutoLock& d_lock);
      void issue_unordered_operations(
          AutoLock& d_lock, std::vector<Operation*>& ready_operations);
      virtual unsigned minimize_repeat_results(
          unsigned ready, bool& double_wait_interval);
    public:
      void add_to_prepipeline_queue(Operation* op);
      bool process_prepipeline_stage(void);
    public:
      virtual bool add_to_dependence_queue(
          Operation* op,
          const std::vector<StaticDependence>* dependences = nullptr,
          bool unordered = false, bool outermost = true);
      virtual FenceOp* initialize_trace_completion(Provenance* prov);
      void process_dependence_stage(void);
    public:
      template<typename T, typename ARGS, bool HAS_BOUNDS>
      void add_to_queue(
          QueueEntry<T> entry, LocalLock& lock,
          std::list<QueueEntry<T> >& queue, CompletionQueue& comp_queue);
      template<typename T>
      T process_queue(
          LocalLock& lock, RtEvent& precondition,
          std::list<QueueEntry<T> >& queue, CompletionQueue& comp_queue,
          std::vector<T>& to_perform, LgEvent previous_fevent,
          long long& performed) const;
    public:
      void add_to_ready_queue(Operation* op);
      bool process_ready_queue(void);
    public:
      void add_to_task_queue(SingleTask* task, RtEvent ready);
      bool process_enqueue_task_queue(
          RtEvent precondition, LgEvent fevent, long long performed);
    public:
      void add_to_trigger_execution_queue(Operation* op, RtEvent ready);
      bool process_trigger_execution_queue(
          RtEvent precondition, LgEvent fevent, long long performed);
    public:
      void add_to_deferred_execution_queue(Operation* op, RtEvent ready);
      bool process_deferred_execution_queue(
          RtEvent precondition, LgEvent fevent, long long performed);
    public:
      void add_to_deferred_mapped_queue(Operation* op, RtEvent ready);
      bool process_deferred_mapped_queue(
          RtEvent precondition, LgEvent fevent, long long performed);
    public:
      void add_to_deferred_completion_queue(
          Operation* op, ApEvent effects, bool tracked);
      bool process_deferred_completion_queue(
          RtEvent precondition, LgEvent fevent, long long performed);
    public:
      void add_to_deferred_commit_queue(
          Operation* op, RtEvent ready, bool deactivate);
      bool process_deferred_commit_queue(
          RtEvent precondition, LgEvent fevent, long long performed);
      bool process_trigger_commit_queue(void);
    public:
      void register_executing_child(Operation* op);
      void register_child_complete(Operation* op);
      void register_child_commit(Operation* op);
      ReorderBufferEntry& find_rob_entry(Operation* op);
      void register_implicit_dependences(Operation* op);
    public:
      ApEvent get_current_execution_fence_event(void);
      // Break this into two pieces since we know that there are some
      // kinds of operations (like deletions) that want to act like
      // one-sided fences (e.g. waiting on everything before) but not
      // preventing re-ordering for things afterwards
      void perform_mapping_fence_analysis(Operation* op);
      void update_current_mapping_fence(FenceOp* op);
      void perform_execution_fence_analysis(
          Operation* op, std::set<ApEvent>& preconditions);
      void update_current_execution_fence(FenceOp* op, ApEvent fence_event);
      void update_current_implicit_creation(Operation* op);
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
      void increment_outstanding(void);
      void decrement_outstanding(void);
      void increment_pending(void);
      void decrement_pending(TaskOp* child);
      void decrement_pending(bool need_deferral);
      void increment_frame(void);
      void decrement_frame(void);
      void finish_frame(FrameOp* frame);
    public:
#ifdef DEBUG_LEGION_COLLECTIVES
      virtual MergeCloseOp* get_merge_close_op(
          Operation* op, RegionTreeNode* node);
      virtual RefinementOp* get_refinement_op(
          Operation* op, RegionTreeNode* node);
#else
      virtual MergeCloseOp* get_merge_close_op(void);
      virtual RefinementOp* get_refinement_op(void);
#endif
    public:
      virtual void pack_inner_context(Serializer& rez) const;
      static InnerContext* unpack_inner_context(Deserializer& derez);
    public:
      virtual InnerContext* find_parent_physical_context(unsigned index);
    public:
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_top_context(InnerContext* previous = nullptr);
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement>& clone_requirements,
          const std::vector<ApUserEvent>& unmap_events);
      virtual EquivalenceSet* create_initial_equivalence_set(
          unsigned idx1, const RegionRequirement& req);
      virtual void refine_equivalence_sets(
          unsigned req_index, IndexSpaceNode* node,
          const FieldMask& refinement_mask,
          std::vector<RtEvent>& applied_events, bool sharded = false,
          bool first = true, const CollectiveMapping* mapping = nullptr);
      virtual void find_trace_local_sets(
          unsigned req_index, const FieldMask& mask,
          std::map<EquivalenceSet*, unsigned>& current_sets,
          IndexSpaceNode* node = nullptr,
          const CollectiveMapping* mapping = nullptr);
      virtual void invalidate_logical_context(void);
      virtual void invalidate_region_tree_contexts(
          const bool is_top_level_task, std::set<RtEvent>& applied,
          const ShardMapping* mapping = nullptr, ShardID source_shard = 0);
      void invalidate_created_requirement_contexts(
          const bool is_top_level_task, std::set<RtEvent>& applied,
          const ShardMapping* mapping, ShardID source_shard);
      virtual void receive_created_region_contexts(
          const std::vector<RegionNode*>& created_regions,
          const std::vector<EqKDTree*>& created_trees,
          std::set<RtEvent>& applied_events, const ShardMapping* mapping,
          ShardID source_shard);
      void invalidate_region_tree_context(
          const RegionRequirement& req, unsigned req_index,
          std::set<RtEvent>& applied_events, bool filter_specific_fields);
    public:
      virtual ProjectionSummary* construct_projection_summary(
          Operation* op, unsigned index, const RegionRequirement& req,
          LogicalState* owner, const ProjectionInfo& proj_info);
      virtual bool has_interfering_shards(
          ProjectionSummary* one, ProjectionSummary* two, bool& dominates);
      virtual bool match_timeouts(
          std::vector<LogicalUser*>& timeouts,
          std::vector<LogicalUser*>& to_delete,
          TimeoutMatchExchange*& exchange);
    public:
      virtual std::pair<bool, bool> has_pointwise_dominance(
          ProjectionSummary* one, ProjectionSummary* two);
      virtual RtEvent find_pointwise_dependence(
          uint64_t context_index, const DomainPoint& point, ShardID shard,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT);
    public:
      void record_fill_view_creation(FillView* view);
      void record_fill_view_creation(DistributedID future_did, FillView* view);
      FillView* find_or_create_fill_view(
          FillOp* op, const void* value, size_t value_size);
      FillView* find_or_create_fill_view(
          FillOp* op, const Future& future, bool& set_value);
      FillView* find_fill_view(const void* value, size_t value_size);
      FillView* find_fill_view(const Future& future);
    public:
      virtual void notify_instance_deletion(PhysicalManager* deleted);
      virtual void add_subscriber_reference(PhysicalManager* manager)
      {
        add_nested_resource_ref(manager->did);
      }
      virtual bool remove_subscriber_reference(PhysicalManager* manager)
      {
        return remove_nested_resource_ref(manager->did);
      }
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
      virtual void handle_mispredication(void);
    public:
      virtual void destroy_lock(Lock l);
      virtual Grant acquire_grant(const std::vector<LockRequest>& requests);
      virtual void release_grant(Grant grant);
    public:
      virtual void destroy_phase_barrier(PhaseBarrier pb);
    public:
      void perform_barrier_dependence_analysis(
          Operation* op, const std::vector<PhaseBarrier>& wait_barriers,
          const std::vector<PhaseBarrier>& arrive_barriers,
          MustEpochOp* must_epoch = nullptr);
    protected:
      void analyze_barrier_dependences(
          Operation* op, const std::vector<PhaseBarrier>& barriers,
          MustEpochOp* must_epoch, bool previous_gen);
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
    public:
      static void handle_compute_equivalence_sets_request(
          Deserializer& derez, AddressSpaceID source);
      static void handle_compute_equivalence_sets_response(Deserializer& derez);
      static void handle_output_equivalence_set_request(Deserializer& derez);
      static void handle_output_equivalence_set_response(
          Deserializer& derez, AddressSpaceID source);
    public:
      static void handle_prepipeline_stage(const void* args);
      static void handle_dependence_stage(const void* args);
      static void handle_ready_queue(const void* args);
      static void handle_enqueue_task_queue(const void* args);
      static void handle_launch_task_queue(const void* args);
      static void handle_trigger_execution_queue(const void* args);
      static void handle_deferred_execution_queue(const void* args);
      static void handle_deferred_mapped_queue(const void* args);
      static void handle_deferred_completion_queue(const void* args);
      static void handle_trigger_commit_queue(const void* args);
      static void handle_deferred_commit_queue(const void* args);
    public:
      void send_context(AddressSpaceID source);
    public:
      // These three methods guard all access to the creation of views onto
      // physical instances within a parent task context. This is important
      // because we need to guarantee the invariant that for every given
      // physical instance in a context it has at most one logical view
      // that represents its state in the physical analysis.
      // Be careful here! These methods should be called on a context
      // that is the result of find_parent_physical_context to account
      // for virtual mappings
      void convert_individual_views(
          const std::vector<PhysicalManager*>& srcs,
          std::vector<IndividualView*>& views,
          CollectiveMapping* mapping = nullptr);
      void convert_individual_views(
          const InstanceSet& sources, std::vector<IndividualView*>& views,
          CollectiveMapping* mapping = nullptr);
      void convert_analysis_views(
          const InstanceSet& targets,
          op::vector<FieldMaskSet<InstanceView> >& target_views);
      IndividualView* create_instance_top_view(
          PhysicalManager* manager, AddressSpaceID source,
          CollectiveMapping* mapping = nullptr);
      virtual CollectiveResult* find_or_create_collective_view(
          RegionTreeID tid, const std::vector<DistributedID>& instances,
          RtEvent& ready);
      void notify_collective_deletion(RegionTreeID tid, DistributedID did);
    protected:
      RtEvent dispatch_collective_invalidation(
          const CollectiveResult* collective, const FieldMask& invalid_mask,
          const FieldMaskSet<CollectiveResult>& replacements);
      CollectiveResult* find_or_create_collective_view(
          RegionTreeID tid, const std::vector<DistributedID>& instances);
      RtEvent create_collective_view(
          DistributedID creator_did, DistributedID collective_did,
          CollectiveMapping* mapping,
          const std::vector<DistributedID>& individual_dids);
      static void release_collective_view(
          DistributedID context_did, DistributedID collective_did);
    public:
      static void handle_create_collective_view(Deserializer& derez);
      static void handle_delete_collective_view(Deserializer& derez);
      static void handle_release_collective_view(Deserializer& derez);
    protected:
      void execute_task_launch(
          TaskOp* task, bool index,
          const std::vector<StaticDependence>* dependences,
          Provenance* provenance, bool silence_warnings, bool inlining_enabled);
      void remap_unmapped_regions(
          LogicalTrace* current_trace,
          const std::vector<PhysicalRegion>& unmapped_regions,
          Provenance* provenance);
    public:
      static constexpr uint64_t NO_BLOCKING_INDEX =
          std::numeric_limits<uint64_t>::max();
      uint64_t get_next_blocking_index(void);
    public:
      void clone_local_fields(
          std::map<FieldSpace, std::vector<LocalFieldInfo> >& child_local)
          const;
#ifdef DEBUG_LEGION
      // This is a helpful debug method that can be useful when called from
      // a debugger to find the earliest operation that hasn't mapped yet
      // which is especially useful when debugging scheduler hangs
      Operation* get_earliest(void) const;
#endif
#ifdef LEGION_SPY
      void register_implicit_replay_dependence(Operation* op);
#endif
    public:
      const ContextID tree_context;
      const bool full_inner_context;
    protected:
      // This is immutable except for remote contexts which unpack it
      // after the object has already been created
      bool concurrent_context;
      bool finished_execution;
      bool has_inline_accessor;
    protected:
      mutable LocalLock privilege_lock;
      unsigned next_created_index;
      // Application tasks can manipulate these next two data
      // structure by creating regions and fields, make sure you are
      // holding the operation lock when you are accessing them
      // We use a region requirement with an empty privilege_fields
      // set to indicate regions on which we have privileges for
      // all fields because this is a created region instead of
      // a created field.
      std::map<unsigned, RegionRequirement> created_requirements;
      std::map<unsigned, bool> returnable_privileges;
      // Number of outstanding deletions using this created requirement
      // The last one to send the count to zero actually gets to delete
      // the requirement and the logical region
      std::map<unsigned, unsigned> deletion_counts;
    protected:
      // Equivalence set trees are used for finding the equivalence sets
      // for a given parent region requirement. Note that each of these
      // trees comes with an associated tree lock that guarantees that
      // invalidation are exclusive with respect to all other kinds of
      // operations that traverse the equivalence set trees
      class EqKDRoot {
      public:
        EqKDRoot(void);
        EqKDRoot(EqKDTree* tree);
        EqKDRoot(const EqKDRoot& rhs) = delete;
        EqKDRoot(EqKDRoot&& rhs) noexcept;
        ~EqKDRoot(void);
      public:
        EqKDRoot& operator=(const EqKDRoot& rhs) = delete;
        EqKDRoot& operator=(EqKDRoot&& rhs) noexcept;
      public:
        EqKDTree* tree;
        LocalLock* lock;
      };
      std::map<unsigned, EqKDRoot> equivalence_set_trees;
      // Pending computations for equivalence set trees
      std::map<unsigned, RtUserEvent> pending_equivalence_set_trees;
    protected:
      const Mapper::ContextConfigOutput context_configuration;
      TaskTreeCoordinates context_coordinates;
    protected:
      // TODO: In the future convert these into std::span so that they
      // are bounded from above by regions.size(), in practice they might
      // actually be bigger than that since the owner task might have
      // output regions which means these will be as big as
      // regions.size() + output_regions.size(), but we don't actually
      // want to think of them that way since the output regions this
      // task is producing don't have any bearing on the sub-tasks that
      // we are launching in this context.
      const std::vector<unsigned>& parent_req_indexes;
      const std::vector<bool>& virtual_mapped;
      // Keep track of inline mapping regions for this task
      // so we can see when there are conflicts, note that accessing
      // this data structure requires the inline lock because
      // unordered detach operations can touch it without synchronizing
      // with the executing task
      ctx::list<PhysicalRegion> inline_regions;
    protected:
      mutable LocalLock child_op_lock;
      // Track whether this task has finished executing
      uint64_t total_children_count;  // total number of sub-operations
      uint64_t next_blocking_index;
      std::deque<ReorderBufferEntry> reorder_buffer;
      // For tracking any operations that come from outside the
      // task like a garbage collector that need to be inserted
      // into the stream of operations from the task
      std::vector<Operation*> unordered_ops;
#ifdef LEGION_SPY
      // Some help for Legion Spy for validating fences
      std::deque<UniqueID> ops_since_last_fence;
      std::set<ApEvent> previous_completion_events;
      // And for verifying the cummulativity property of task
      // (e.g. that they are not complete until all their children are)
      std::vector<ApEvent> cummulative_child_completion_events;
#endif
    protected:  // Queues for fusing together small meta-tasks
      mutable LocalLock prepipeline_lock;
      std::deque<std::pair<Operation*, GenerationID> > prepipeline_queue;
      unsigned outstanding_prepipeline_tasks;
    protected:
      mutable LocalLock dependence_lock;
      std::deque<Operation*> dependence_queue;
      RtEvent dependence_precondition;
    protected:
      mutable LocalLock ready_lock;
      std::deque<Operation*> ready_queue;
    protected:
      mutable LocalLock enqueue_task_lock;
      std::list<QueueEntry<SingleTask*> > enqueue_task_queue;
      CompletionQueue enqueue_task_comp_queue;
    protected:
      mutable LocalLock trigger_execution_lock;
      std::list<QueueEntry<Operation*> > trigger_execution_queue;
      CompletionQueue trigger_execution_comp_queue;
    protected:
      mutable LocalLock deferred_execution_lock;
      std::list<QueueEntry<Operation*> > deferred_execution_queue;
      CompletionQueue deferred_execution_comp_queue;
    protected:
      mutable LocalLock deferred_mapped_lock;
      std::list<QueueEntry<Operation*> > deferred_mapped_queue;
      CompletionQueue deferred_mapped_comp_queue;
    protected:
      // Uses the child op lock
      std::list<CompletionEntry> deferred_completion_queue;
      CompletionQueue deferred_completion_comp_queue;
    protected:
      mutable LocalLock deferred_commit_lock;
      std::list<QueueEntry<std::pair<Operation*, bool> > >
          deferred_commit_queue;
      CompletionQueue deferred_commit_comp_queue;
    protected:
      // Traces for this task's execution
      LegionMap<TraceID, LogicalTrace*, CONTEXT_LIFETIME> traces;
      LogicalTrace* current_trace;
      LogicalTrace* previous_trace;
      uint64_t current_trace_blocking_index;
      // ID is either 0 for not replaying, 1 for replaying not idempotent,
      // 2 for replaying idempotent or the event id for signaling that
      // the status isn't ready
      std::atomic<realm_id_t> physical_trace_replay_status;
      RtUserEvent window_wait;
      std::deque<FrameOp*> frame_ops;
    protected:
      // Number of sub-tasks ready to map
      unsigned outstanding_subtasks;
      // Number of mapped sub-tasks that are yet to run
      unsigned pending_subtasks;
      // Number of pending_frames
      unsigned pending_frames;
      // Track whether this context is current active for scheduling
      // indicating that it is no longer far enough ahead
      bool currently_active_context;
      // Whether we have an outstanding commit task in flight
      bool outstanding_commit_task;
    protected:
#ifdef LEGION_SPY
      UniqueID current_fence_uid;
#endif
      FenceOp* current_mapping_fence;
      GenerationID current_mapping_fence_gen;
      uint64_t current_mapping_fence_index;
      ApEvent current_execution_fence_event;
      uint64_t current_execution_fence_index;
      // We currently do not track dependences for dependent partitioning
      // operations on index partitions and their subspaces directly, so
      // we instead use this to ensure mapping dependence ordering with
      // any operations which might need downstream information about
      // partitions or subspaces. Note that this means that all dependent
      // partitioning operations are guaranteed to map in order currently
      // We've now extended this to include creation operations and pending
      // partition operations as well for similar reasons, so now this
      // is a general operation class
      Operation* last_implicit_creation;
      GenerationID last_implicit_creation_gen;
    protected:
      // For managing changing task priorities
      TaskPriority current_priority;
    protected:  // Instance top view data structures
      mutable LocalLock instance_view_lock;
      std::map<PhysicalManager*, IndividualView*> instance_top_views;
      std::map<PhysicalManager*, RtUserEvent> pending_top_views;
    protected:
      // Field allocation data
      std::map<FieldSpace, FieldAllocatorImpl*> field_allocators;
    protected:
      // Our cached set of index spaces for immediate domains
      std::map<Domain, IndexSpace> index_launch_spaces;
    protected:
      std::map<uint64_t, std::map<DomainPoint, RtUserEvent> >
          pending_pointwise_dependences;
    protected:
      // Dependence tracking information for phase barriers
      mutable LocalLock phase_barrier_lock;
      struct BarrierContribution {
      public:
        BarrierContribution(void) : op(nullptr), gen(0), uid(0), muid(0) { }
        BarrierContribution(
            Operation* o, GenerationID g, UniqueID u, UniqueID m, size_t bg)
          : op(o), gen(g), uid(u), muid(m), bargen(bg)
        { }
      public:
        Operation* op;
        GenerationID gen;
        UniqueID uid;
        UniqueID muid;  // must epoch uid
        size_t bargen;  // the barrier generation
      };
      std::map<size_t, std::list<BarrierContribution> > barrier_contributions;
    protected:
      // Track information for locally allocated fields
      mutable LocalLock local_field_lock;
      std::map<FieldSpace, std::vector<LocalFieldInfo> > local_field_infos;
    protected:
      // Cache for fill views
      mutable LocalLock fill_view_lock;
      std::list<FillView*> value_fill_view_cache;
      std::list<std::pair<FillView*, DistributedID> > future_fill_view_cache;
      static const size_t MAX_FILL_VIEW_CACHE_SIZE = 64;
    protected:
      // This data structure should only be accessed during the logical
      // analysis stage of the pipeline and therefore no lock is needed
      std::map<IndexTreeNode*, std::vector<AttachProjectionFunctor*> >
          attach_functions;
    protected:
      // Resources that can build up over a task's lifetime
      ctx::deque<Reservation> context_locks;
      ctx::deque<ApBarrier> context_barriers;
    protected:
      // Collective instance rendezvous data structures
      mutable LocalLock collective_lock;
      // Only valid on the onwer context node
      std::map<RegionTreeID, std::vector<CollectiveResult*> >
          collective_results;
    };

    /**
     * \class TimeoutMatchExchange
     * This class helps perform all all-reduce exchange between the shards
     * to see which logical users that have timed out on their analyses
     * can be collected across all the shards. To be collected all the
     * shards must agree on what they are pruning.
     */
    class TimeoutMatchExchange : public AllGatherCollective<false> {
    public:
      TimeoutMatchExchange(ReplicateContext* ctx, CollectiveIndexLocation loc);
      TimeoutMatchExchange(const TimeoutMatchExchange& rhs) = delete;
      virtual ~TimeoutMatchExchange(void);
    public:
      TimeoutMatchExchange& operator=(const TimeoutMatchExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
      {
        return SEND_CONTROL_REPLICATION_TIMEOUT_MATCH_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage);
      virtual void unpack_collective_stage(Deserializer& derez, int stage);
    public:
      void perform_exchange(std::vector<LogicalUser*>& timeouts, bool ready);
      bool complete_exchange(std::vector<LogicalUser*>& to_delete);
    protected:
      std::vector<LogicalUser*> timeout_users;
      // Pair represents <context index,region requirement index> for each user
      std::vector<std::pair<size_t, unsigned> > all_timeouts;
      bool double_latency;
    };

    /**
     * \class CrossProductExchange
     * This all-gather exchanges IDs for the creation of replicated
     * partitions when performing a cross-product partition
     */
    class CrossProductExchange : public AllGatherCollective<false> {
    public:
      CrossProductExchange(ReplicateContext* ctx, CollectiveIndexLocation loc);
      CrossProductExchange(const CrossProductExchange& rhs) = delete;
      virtual ~CrossProductExchange(void) { }
    public:
      CrossProductExchange& operator=(const CrossProductExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const
      {
        return SEND_CONTROL_REPLICATION_CROSS_PRODUCT_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage);
      virtual void unpack_collective_stage(Deserializer& derez, int stage);
    public:
      void exchange_ids(LegionColor color, IndexPartition pid);
      void sync_child_ids(LegionColor color, IndexPartition& pid);
    protected:
      std::map<LegionColor, IndexPartition> child_ids;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INNER_CONTEXT_H__
