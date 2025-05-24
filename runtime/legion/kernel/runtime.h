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

#ifndef __LEGION_CORE_RUNTIME_H__
#define __LEGION_CORE_RUNTIME_H__

#include "legion/kernel/metatask.h"
#include "legion/api/exception.h"
#include "legion/api/interop_impl.h"
#include "legion/api/registrars.h"
#include "legion/api/runtime.h"
#include "legion/managers/message.h"
#include "legion/managers/shutdown.h"
#include "legion/operations/factory.h"
#include "legion/tools/profiler.h"
#include "legion/tools/spy.h"
#include "legion/tracing/recording.h"
#include "legion/utilities/small_vector.h"

namespace Legion {
  namespace Internal {

    // Runtime task numbering
    enum {
      LG_STARTUP_TASK_ID = Realm::Processor::TASK_ID_PROCESSOR_INIT,
      LG_SHUTDOWN_TASK_ID = Realm::Processor::TASK_ID_PROCESSOR_SHUTDOWN,
      LG_TASK_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE,
#ifdef LEGION_SEPARATE_META_TASKS
      LG_LEGION_PROFILING_ID = LG_TASK_ID + LG_LAST_TASK_ID + LAST_SEND_KIND,
      LG_ENDPOINT_TASK_ID = LG_TASK_ID + LG_LAST_TASK_ID + LAST_SEND_KIND + 1,
      LG_APP_PROC_TASK_ID = LG_TASK_ID + LG_LAST_TASK_ID + LAST_SEND_KIND + 2,
      LG_TASK_ID_AVAILABLE = LG_APP_PROC_TASK_ID + LG_LAST_TASK_ID,
#else
      LG_LEGION_PROFILING_ID = LG_TASK_ID + 1,
      LG_ENDPOINT_TASK_ID = LG_TASK_ID + 2,
      LG_APP_PROC_TASK_ID = LG_TASK_ID + 3,
      LG_TASK_ID_AVAILABLE = LG_TASK_ID + 4,
#endif
    };

    /**
     * \class OperationCreator
     * A base class for handling the creation of index space operations
     */
    class OperationCreator {
    public:
      OperationCreator(void);
      virtual ~OperationCreator(void);
    public:
      void produce(IndexSpaceOperation* op);
      IndexSpaceExpression* consume(void);
    public:
      virtual void create_operation(void) = 0;
    protected:
      IndexSpaceOperation* result;
    };

    /**
     * \class PendingVariantRegistration
     * A small helper class for deferring the restration of task
     * variants until the runtime is started.
     */
    class PendingVariantRegistration {
    public:
      PendingVariantRegistration(
          VariantID vid, size_t return_type_size, bool has_return_type_size,
          const TaskVariantRegistrar& registrar, const void* user_data,
          size_t user_data_size, const CodeDescriptor& realm_desc,
          const char* task_name);
      PendingVariantRegistration(const PendingVariantRegistration& rhs) =
          delete;
      ~PendingVariantRegistration(void);
    public:
      PendingVariantRegistration& operator=(
          const PendingVariantRegistration& rhs) = delete;
    public:
      void perform_registration(void);
    private:
      VariantID vid;
      size_t return_type_size;
      bool has_return_type_size;
      TaskVariantRegistrar registrar;
      void* user_data;
      size_t user_data_size;
      CodeDescriptor realm_desc;
      char* logical_task_name;  // optional semantic info to attach to the task
    };

    struct PendingRegistrationCallback {
    public:
      PendingRegistrationCallback(
          RegistrationCallback call, bool dedup, size_t tag);
      PendingRegistrationCallback(
          RegistrationWithArgsCallback call, const UntypedBuffer& buf,
          bool dedup, size_t tag);
      PendingRegistrationCallback(const PendingRegistrationCallback&) = delete;
      PendingRegistrationCallback(PendingRegistrationCallback&& rhs);
      ~PendingRegistrationCallback(void);
    public:
      PendingRegistrationCallback& operator=(
          const PendingRegistrationCallback&) = delete;
      PendingRegistrationCallback& operator=(PendingRegistrationCallback&&) =
          delete;
    public:
      union {
        RegistrationCallback withoutargs;
        RegistrationWithArgsCallback withargs;
      };
      UntypedBuffer buffer;
      size_t dedup_tag;
      bool deduplicate;
      bool has_args;
    };

    /**
     * \class Runtime
     * This is the actual implementation of the Legion runtime functionality
     * that implements the underlying interface for the Runtime
     * objects.  Most of the calls in the Runtime class translate
     * directly to calls to this interface.  Unfortunately this adds
     * an extra function call overhead to every runtime call because C++
     * is terrible and doesn't have mix-in classes.
     */
    class Runtime : public Heapify<Runtime, RUNTIME_LIFETIME> {
    public:
      struct LegionConfiguration {
      public:
        LegionConfiguration(void)
          : delay_start(0), legion_collective_radix(LEGION_COLLECTIVE_RADIX),
            initial_task_window_size(LEGION_DEFAULT_MAX_TASK_WINDOW),
            initial_task_window_hysteresis(
                LEGION_DEFAULT_TASK_WINDOW_HYSTERESIS),
            initial_tasks_to_schedule(LEGION_DEFAULT_MIN_TASKS_TO_SCHEDULE),
            initial_meta_task_vector_width(
                LEGION_DEFAULT_META_TASK_VECTOR_WIDTH),
            max_message_size(LEGION_DEFAULT_MAX_MESSAGE_SIZE),
            gc_epoch_size(LEGION_DEFAULT_GC_EPOCH_SIZE),
            max_control_replication_contexts(
                LEGION_DEFAULT_MAX_CONTROL_REPLICATION_CONTEXTS),
            max_local_fields(LEGION_DEFAULT_LOCAL_FIELDS),
            max_replay_parallelism(LEGION_DEFAULT_MAX_REPLAY_PARALLELISM),
            safe_control_replication(0), program_order_execution(false),
            dump_physical_traces(false), enable_automatic_tracing(false),
            no_tracing(false), no_physical_tracing(false),
            no_auto_tracing(false), no_trace_optimization(false),
            no_fence_elision(false), no_transitive_reduction(false),
            inline_transitive_reduction(false), replay_on_cpus(false),
            verify_partitions(false), runtime_warnings(false),
            warnings_backtrace(false), warnings_are_errors(false),
            report_leaks(false), record_registration(false),
            stealing_disabled(false), resilient_mode(false),
            unsafe_launch(false), unsafe_mapper(false), safe_mapper(false),
#ifdef LEGION_DEBUG
            safe_model(true),
#else
            safe_model(false),
#endif
            safe_tracing(false), disable_independence_tests(false),
            enable_pointwise_analysis(false), enable_test_mapper(false),
            slow_config_ok(false), verbose_logging(false),
            check_privileges(true), dump_free_ranges(false),
            num_profiling_nodes(0), serializer_type("binary"),
            prof_footprint_threshold(128 << 20), prof_target_latency(100),
            prof_call_threshold(0), prof_self_profile(false),
            prof_no_critical_paths(false), prof_all_critical_arrivals(false)
        { }
      public:
        int delay_start;
        int legion_collective_radix;
        int initial_task_window_size;
        unsigned initial_task_window_hysteresis;
        unsigned initial_tasks_to_schedule;
        unsigned initial_meta_task_vector_width;
        unsigned max_message_size;
        unsigned gc_epoch_size;
        unsigned max_control_replication_contexts;
        unsigned max_local_fields;
        unsigned max_replay_parallelism;
        unsigned safe_control_replication;
      public:
        bool program_order_execution;
        bool dump_physical_traces;
        bool enable_automatic_tracing;
        bool no_tracing;
        bool no_physical_tracing;
        bool no_auto_tracing;
        bool no_trace_optimization;
        bool no_fence_elision;
        bool no_transitive_reduction;
        bool inline_transitive_reduction;
        bool replay_on_cpus;
        bool verify_partitions;
        bool runtime_warnings;
        bool warnings_backtrace;
        bool warnings_are_errors;
        bool report_leaks;
        bool record_registration;
        bool stealing_disabled;
        bool resilient_mode;
        bool unsafe_launch;
        bool unsafe_mapper;
        bool safe_mapper;
        bool safe_model;
        bool safe_tracing;
        bool disable_independence_tests;
        bool enable_pointwise_analysis;
        bool enable_test_mapper;
        std::string replay_file;
        std::string ldb_file;
        bool slow_config_ok;
        bool verbose_logging;
        bool check_privileges;
        bool dump_free_ranges;
      public:
        unsigned num_profiling_nodes;
        std::string serializer_type;
        std::string prof_logfile;
        size_t prof_footprint_threshold;
        size_t prof_target_latency;
        size_t prof_call_threshold;
        bool prof_self_profile;
        bool prof_no_critical_paths;
        bool prof_all_critical_arrivals;
      };
    public:
      struct TopFinishArgs : public LgTaskArgs<TopFinishArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_TOP_FINISH_TASK_ID;
      public:
        TopFinishArgs(void) = default;
        TopFinishArgs(TopLevelContext* c) : LgTaskArgs<TopFinishArgs>(0), ctx(c)
        { }
        void execute(void) const;
      public:
        TopLevelContext* ctx;
      };
      struct MapperTaskArgs : public LgTaskArgs<MapperTaskArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_MAPPER_TASK_ID;
      public:
        MapperTaskArgs(void) = default;
        MapperTaskArgs(
            FutureImpl* f, MapperID mid, Processor p, TopLevelContext* c)
          : LgTaskArgs<MapperTaskArgs>(implicit_provenance), future(f),
            map_id(mid), proc(p), ctx(c)
        { }
        void execute(void) const;
      public:
        FutureImpl* future;
        MapperID map_id;
        Processor proc;
        TopLevelContext* ctx;
      };
    public:
      struct ProcessorGroupInfo {
      public:
        ProcessorGroupInfo(void)
          : processor_group(ProcessorGroup::NO_PROC_GROUP)
        { }
        ProcessorGroupInfo(ProcessorGroup p, const ProcessorMask& m)
          : processor_group(p), processor_mask(m)
        { }
      public:
        ProcessorGroup processor_group;
        ProcessorMask processor_mask;
      };
    public:
      Runtime(
          Machine m, const LegionConfiguration& config, bool background,
          InputArgs input_args, AddressSpaceID space_id, Memory sysmem,
          const std::set<Processor>& local_procs,
          const std::set<Processor>& local_util_procs,
          const std::set<AddressSpaceID>& address_spaces,
          bool supply_default_mapper);
      Runtime(const Runtime& rhs) = delete;
      ~Runtime(void);
    public:
      Runtime& operator=(const Runtime& rhs) = delete;
    public:
      // The Runtime wrapper for this class
      Legion::Runtime* const external;
      // The Mapper Runtime for this class
      Legion::Mapping::MapperRuntime* const mapper_runtime;
      // The machine object for this runtime
      const Machine machine;
      const Memory runtime_system_memory;
      const AddressSpaceID address_space;
      const unsigned total_address_spaces;
      // stride for uniqueness, may or may not be the same depending
      // on the number of available control replication contexts
      const unsigned runtime_stride;  // stride for uniqueness
      LegionProfiler* profiler;
      VirtualManager* virtual_manager;
      Processor local_group;    // all local processors
      Processor utility_group;  // all utility processors
      const size_t num_utility_procs;
    public:
      const InputArgs input_args;
      const int initial_task_window_size;
      const unsigned initial_task_window_hysteresis;
      const unsigned initial_tasks_to_schedule;
      const unsigned initial_meta_task_vector_width;
      const unsigned max_message_size;
      const unsigned gc_epoch_size;
      const unsigned max_control_replication_contexts;
      const unsigned max_local_fields;
      const unsigned max_replay_parallelism;
      const unsigned safe_control_replication;
    public:
      const bool program_order_execution;
      const bool dump_physical_traces;
      const bool no_tracing;
      const bool no_physical_tracing;
      const bool no_auto_tracing;
      const bool no_trace_optimization;
      const bool no_fence_elision;
      const bool no_transitive_reduction;
      const bool inline_transitive_reduction;
      const bool replay_on_cpus;
      const bool verify_partitions;
      const bool runtime_warnings;
      const bool warnings_backtrace;
      const bool warnings_are_errors;
      const bool report_leaks;
      const bool record_registration;
      const bool stealing_disabled;
      const bool resilient_mode;
      const bool unsafe_launch;
      const bool safe_mapper;
      const bool safe_model;
      const bool safe_tracing;
      const bool disable_independence_tests;
      const bool enable_pointwise_analysis;
      const bool supply_default_mapper;
      const bool enable_test_mapper;
      const bool legion_ldb_enabled;
      const std::string replay_file;
      const bool verbose_logging;
      const bool check_privileges;
      const bool dump_free_ranges;
    public:
      const int legion_collective_radix;
      MPIRankTable* const mpi_rank_table;
    public:
      void register_static_variants(void);
      CollectiveMapping* register_static_constraints(
          uint64_t& next_static_did, LayoutConstraintID& virtual_layout_id);
      void register_static_projections(void);
      void register_static_sharding_functors(void);
      void register_static_concurrent_functors(void);
      void initialize_legion_prof(const LegionConfiguration& config);
      void log_local_machine(void) const;
      void initialize_mappers(void);
      void initialize_virtual_manager(
          uint64_t& next_static_did, LayoutConstraintID virtual_layout_id,
          CollectiveMapping* mapping);
      TopLevelContext* initialize_runtime(Processor local_proc);
#ifdef LEGION_USE_LIBDL
      void send_registration_callback(
          AddressSpaceID space, Realm::DSOReferenceImplementation* impl,
          RtEvent done, std::set<RtEvent>& applied, const void* buffer,
          size_t buffer_size, bool withargs, bool deduplicate,
          size_t dedup_tag);
#endif
      RtEvent perform_registration_callback(
          const PendingRegistrationCallback& callback, bool global,
          bool preregistered);
      void broadcast_startup_barrier(RtBarrier startup_barrier);
      void finalize_runtime(std::vector<Realm::Event>& shutdown_events);
      ApEvent launch_mapper_task(
          Mapper* mapper, Processor proc, TaskID tid, const UntypedBuffer& arg,
          MapperID map_id);
      void process_mapper_task_result(const MapperTaskArgs* args);
    public:
      void create_shared_ownership(
          IndexSpace handle, const bool total_sharding_collective = false,
          const bool unpack_reference = false);
      void create_shared_ownership(
          IndexPartition handle, const bool total_sharding_collective = false,
          const bool unpack_reference = false);
      void create_shared_ownership(
          FieldSpace handle, const bool total_sharding_collective = false,
          const bool unpack_reference = false);
      void create_shared_ownership(
          LogicalRegion handle, const bool total_sharding_collective = false,
          const bool unpack_reference = false);
    public:
      size_t get_domain_volume(IndexSpace handle);
      void find_domain(IndexSpace handle, Domain& launch_domain);
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      bool has_index_partition(IndexSpace parent, Color color);
      IndexSpace get_index_subspace(
          IndexPartition p, const void* realm_color, TypeTag type_tag);
      bool has_index_subspace(
          IndexPartition p, const void* realm_color, TypeTag type_tag);
      void get_index_space_domain(
          IndexSpace handle, void* realm_is, TypeTag type_tag);
      Domain get_index_partition_color_space(IndexPartition p);
      void get_index_partition_color_space(
          IndexPartition p, void* realm_is, TypeTag type_tag);
      IndexSpace get_index_partition_color_space_name(IndexPartition p);
      void get_index_space_partition_colors(
          IndexSpace handle, std::set<Color>& colors);
      bool is_index_partition_disjoint(IndexPartition p);
      bool is_index_partition_complete(IndexPartition p);
      void get_index_space_color_point(
          IndexSpace handle, void* realm_color, TypeTag type_tag);
      DomainPoint get_index_space_color_point(IndexSpace handle);
      Color get_index_partition_color(IndexPartition handle);
      IndexSpace get_parent_index_space(IndexPartition handle);
      bool has_parent_index_partition(IndexSpace handle);
      IndexPartition get_parent_index_partition(IndexSpace handle);
      unsigned get_index_space_depth(IndexSpace handle);
      unsigned get_index_partition_depth(IndexPartition handle);
      bool has_index_path(IndexSpace parent, IndexSpace child);
      bool has_partition_path(IndexSpace parent, IndexPartition child);
      unsigned get_projection_depth(LogicalRegion result, LogicalRegion upper);
      unsigned get_projection_depth(
          LogicalRegion result, LogicalPartition upper);
      bool is_top_level_index_space(IndexSpace handle);
    public:
      void destroy_index_space(
          IndexSpace handle, AddressSpaceID source,
          std::set<RtEvent>& applied_events,
          const CollectiveMapping* mapping = nullptr);
      void destroy_index_partition(
          IndexPartition handle, std::set<RtEvent>& applied,
          const CollectiveMapping* mapping = nullptr);
      void destroy_field_space(
          FieldSpace handle, std::set<RtEvent>& applied,
          const CollectiveMapping* mapping = nullptr);
      void destroy_logical_region(
          LogicalRegion handle, std::set<RtEvent>& applied,
          const CollectiveMapping* mapping = nullptr);
      static void send_index_space_destruction(
          IndexSpace handle, AddressSpaceID target, std::set<RtEvent>& applied);
      static void send_index_partition_destruction(
          IndexPartition handle, AddressSpaceID target,
          std::set<RtEvent>& applied);
      static void send_field_space_destruction(
          FieldSpace handle, AddressSpaceID target, std::set<RtEvent>& applied);
      static void send_logical_region_destruction(
          LogicalRegion handle, AddressSpaceID target,
          std::set<RtEvent>& applied);
    public:
      void get_all_fields(FieldSpace handle, std::set<FieldID>& fields);
      void get_all_regions(FieldSpace handle, std::set<LogicalRegion>& regions);
      size_t get_coordinate_size(IndexSpace handle, bool range);
      size_t get_field_size(FieldSpace handle, FieldID fid);
      CustomSerdezID get_field_serdez(FieldSpace handle, FieldID fid);
      void get_field_space_fields(
          FieldSpace handle, std::vector<FieldID>& fields);
    public:
      // Return true if local is set to true and we actually performed the
      // allocation.  It is an error if the field already existed and the
      // allocation was not local.
      RtEvent allocate_field(
          FieldSpace handle, size_t field_size, FieldID fid,
          CustomSerdezID serdez_id, Provenance* provenance,
          bool sharded_non_owner = false);
      FieldSpaceNode* allocate_field(
          FieldSpace handle, ApEvent ready, FieldID fid,
          CustomSerdezID serdez_id, Provenance* provenance,
          RtEvent& precondition, bool sharded_non_owner = false);
      void free_field(
          FieldSpace handle, FieldID fid, std::set<RtEvent>& applied,
          bool sharded_non_owner = false);
      RtEvent allocate_fields(
          FieldSpace handle, const std::vector<size_t>& sizes,
          const std::vector<FieldID>& resulting_fields,
          CustomSerdezID serdez_id, Provenance* provenance,
          bool sharded_non_owner = false);
      FieldSpaceNode* allocate_fields(
          FieldSpace handle, ApEvent ready,
          const std::vector<FieldID>& resulting_fields,
          CustomSerdezID serdez_id, Provenance* provenance,
          RtEvent& precondition, bool sharded_non_owner = false);
      void free_fields(
          FieldSpace handle, const std::vector<FieldID>& to_free,
          std::set<RtEvent>& applied, bool sharded_non_owner = false);
      void free_field_indexes(
          FieldSpace handle, const std::vector<FieldID>& to_free, RtEvent freed,
          bool sharded_non_owner = false);
    public:
      bool allocate_local_fields(
          FieldSpace handle, const std::vector<FieldID>& resulting_fields,
          const std::vector<size_t>& sizes, CustomSerdezID serdez_id,
          const std::set<unsigned>& allocated_indexes,
          std::vector<unsigned>& new_indexes, Provenance* provenance);
      void free_local_fields(
          FieldSpace handle, const std::vector<FieldID>& to_free,
          const std::vector<unsigned>& indexes,
          const CollectiveMapping* mapping = nullptr);
      void update_local_fields(
          FieldSpace handle, const std::vector<FieldID>& fields,
          const std::vector<size_t>& sizes,
          const std::vector<CustomSerdezID>& serdez_ids,
          const std::vector<unsigned>& indexes, Provenance* provenance);
      void remove_local_fields(
          FieldSpace handle, const std::vector<FieldID>& to_remove);
    public:
      bool is_subregion(LogicalRegion child, LogicalRegion parent);
      bool is_subregion(LogicalRegion child, LogicalPartition parent);
      bool is_disjoint(IndexPartition handle);
      bool is_disjoint(LogicalPartition handle);
    public:
      bool are_disjoint(IndexSpace one, IndexSpace two);
      bool are_disjoint(IndexSpace one, IndexPartition two);
      bool are_disjoint(IndexPartition one, IndexPartition two);
      // Can only use the region tree for proving disjointness here
      bool are_disjoint_tree_only(
          IndexTreeNode* one, IndexTreeNode* two,
          IndexTreeNode*& common_ancestor);
    public:
      bool check_types(TypeTag t1, TypeTag t2, bool& diff_dims);
      bool is_dominated(IndexSpace src, IndexSpace dst);
      bool is_dominated_tree_only(IndexSpace test, IndexPartition dominator);
      bool is_dominated_tree_only(IndexPartition test, IndexSpace dominator);
      bool is_dominated_tree_only(
          IndexPartition test, IndexPartition dominator);
    public:
      LogicalPartition get_logical_partition(
          LogicalRegion parent, IndexPartition handle);
      LogicalPartition get_logical_partition_by_color(
          LogicalRegion parent, Color c);
      bool has_logical_partition_by_color(LogicalRegion parent, Color c);
      LogicalPartition get_logical_partition_by_tree(
          IndexPartition handle, FieldSpace fspace, RegionTreeID tid);
      LogicalRegion get_logical_subregion(
          LogicalPartition parent, IndexSpace handle);
      LogicalRegion get_logical_subregion_by_color(
          LogicalPartition parent, const void* realm_color, TypeTag type_tag);
      bool has_logical_subregion_by_color(
          LogicalPartition parent, const void* realm_color, TypeTag type_tag);
      LogicalRegion get_logical_subregion_by_tree(
          IndexSpace handle, FieldSpace fspace, RegionTreeID tid);
      void get_logical_region_color(
          LogicalRegion handle, void* realm_color, TypeTag type_tag);
      DomainPoint get_logical_region_color_point(LogicalRegion handle);
      Color get_logical_partition_color(LogicalPartition handle);
      LogicalRegion get_parent_logical_region(LogicalPartition handle);
      bool has_parent_logical_partition(LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(LogicalRegion handle);
      bool is_top_level_region(LogicalRegion handle);
    public:
      IndexSpaceNode* create_index_space(
          IndexSpace handle, const Domain& domain, bool take_ownership,
          Provenance* provenance, CollectiveMapping* mapping = nullptr,
          IndexSpaceExprID expr_id = 0, ApEvent ready = ApEvent::NO_AP_EVENT,
          RtEvent initialized = RtEvent::NO_RT_EVENT);
      IndexSpaceNode* create_union_space(
          IndexSpace handle, Provenance* provenance,
          const std::vector<IndexSpace>& sources,
          RtEvent initialized = RtEvent::NO_RT_EVENT,
          CollectiveMapping* mapping = nullptr, IndexSpaceExprID expr_id = 0);
      IndexSpaceNode* create_intersection_space(
          IndexSpace handle, Provenance* provenance,
          const std::vector<IndexSpace>& sources,
          RtEvent initialized = RtEvent::NO_RT_EVENT,
          CollectiveMapping* mapping = nullptr, IndexSpaceExprID expr_id = 0);
      IndexSpaceNode* create_difference_space(
          IndexSpace handle, Provenance* provenance, IndexSpace left,
          IndexSpace right, RtEvent initialized = RtEvent::NO_RT_EVENT,
          CollectiveMapping* mapping = nullptr, IndexSpaceExprID expr_id = 0);
    public:
      // We know the domain of the index space
      IndexSpaceNode* create_node(
          IndexSpace is, const Domain& bounds, bool take_ownership,
          IndexPartNode* par, LegionColor color, RtEvent initialized,
          Provenance* provenance, ApEvent is_ready = ApEvent::NO_AP_EVENT,
          IndexSpaceExprID expr_id = 0, CollectiveMapping* mapping = nullptr,
          const bool add_root_reference = false,
          unsigned depth = std::numeric_limits<unsigned>::max(),
          const bool tree_valid = true);
      IndexSpaceNode* create_node(
          IndexSpace is, IndexPartNode& par, LegionColor color,
          RtEvent initialized, Provenance* provenance,
          IndexSpaceExprID expr_id = 0, CollectiveMapping* mapping = nullptr,
          unsigned depth = std::numeric_limits<unsigned>::max());
      // We know the disjointness of the index partition
      IndexPartNode* create_node(
          IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* color_space,
          LegionColor color, bool disjoint, int complete,
          Provenance* provenance, RtEvent init,
          CollectiveMapping* mapping = nullptr);
      // Give the event for when the disjointness information is ready
      IndexPartNode* create_node(
          IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* color_space,
          LegionColor color, int complete, Provenance* provenance, RtEvent init,
          CollectiveMapping* mapping = nullptr);
      FieldSpaceNode* create_node(
          FieldSpace space, RtEvent init, Provenance* provenance,
          CollectiveMapping* mapping = nullptr);
      FieldSpaceNode* create_node(
          FieldSpace space, RtEvent initialized, Provenance* provenance,
          CollectiveMapping* mapping, Deserializer& derez);
      RegionNode* create_node(
          LogicalRegion r, PartitionNode* par, RtEvent initialized,
          DistributedID did, Provenance* provenance = nullptr,
          CollectiveMapping* mapping = nullptr);
      PartitionNode* create_node(LogicalPartition p, RegionNode* par);
    public:
      void record_pending_index_space(DistributedID space);
      void record_pending_partition(DistributedID pid);
      void record_pending_field_space(DistributedID space);
      void record_pending_region_tree(RegionTreeID tree);
    public:
      void revoke_pending_index_space(DistributedID space);
      void revoke_pending_partition(DistributedID pid);
      void revoke_pending_field_space(DistributedID space);
      void revoke_pending_region_tree(RegionTreeID tree);
    public:
      IndexSpaceNode* get_node(
          IndexSpace space, RtEvent* defer = nullptr,
          const bool can_fail = false, const bool first = true);
      IndexPartNode* get_node(
          IndexPartition part, RtEvent* defer = nullptr,
          const bool can_fail = false, const bool first = true,
          const bool local_only = false);
      FieldSpaceNode* get_node(
          FieldSpace space, RtEvent* defer = nullptr, bool first = true);
      RegionNode* get_node(
          LogicalRegion handle, bool need_check = true, bool first = true);
      PartitionNode* get_node(LogicalPartition handle, bool need_check = true);
      RegionNode* get_tree(
          RegionTreeID tid, bool can_fail = false, bool first = true);
      // Request but don't block
      RtEvent find_or_request_node(IndexSpace space, AddressSpaceID target);
    public:
      bool has_node(IndexSpace space);
      bool has_node(IndexPartition part);
      bool has_node(FieldSpace space);
      bool has_node(LogicalRegion handle);
      bool has_node(LogicalPartition handle);
      bool has_tree(RegionTreeID tid);
      bool has_field(FieldSpace space, FieldID fid);
    public:
      void remove_node(IndexSpace space);
      void remove_node(IndexPartition part);
      void remove_node(FieldSpace space);
      void remove_node(LogicalRegion handle, bool top);
      void remove_node(LogicalPartition handle);
    public:
      // These three methods a something pretty awesome and crazy
      // We want to do common sub-expression elimination on index space
      // unions, intersections, and difference operations to avoid repeating
      // expensive Realm dependent partition calls where possible, by
      // running everything through this interface we first check to see
      // if these operations have been requested before and if so will
      // return the common sub-expression, if not we will actually do
      // the computation and memoize it for the future
      //
      // Note that you do not need to worry about reference counting
      // expressions returned from these methods inside of tasks because
      // we implicitly add references to them and store them in the
      // implicit_live_expression data structure and then remove the
      // references after the meta-task or runtime call is done executing.

      IndexSpaceExpression* union_index_spaces(
          IndexSpaceExpression* lhs, IndexSpaceExpression* rhs);
      IndexSpaceExpression* union_index_spaces(
          const SetView<IndexSpaceExpression*>& exprs);
    protected:
      // Internal version
      IndexSpaceExpression* union_index_spaces(
          const std::vector<IndexSpaceExpression*>& exprs,
          OperationCreator* creator = nullptr);
    public:
      IndexSpaceExpression* intersect_index_spaces(
          IndexSpaceExpression* lhs, IndexSpaceExpression* rhs);
      IndexSpaceExpression* intersect_index_spaces(
          const SetView<IndexSpaceExpression*>& exprs);
    protected:
      IndexSpaceExpression* intersect_index_spaces(
          const std::vector<IndexSpaceExpression*>& exprs,
          OperationCreator* creator = nullptr);
    public:
      IndexSpaceExpression* subtract_index_spaces(
          IndexSpaceExpression* lhs, IndexSpaceExpression* rhs,
          OperationCreator* creator = nullptr);
    protected:
      // You don't call this method directly, call
      // IndexSpaceExpression::get_canonical_expression instead
      friend class IndexSpaceExpression;
      IndexSpaceExpression* find_canonical_expression(IndexSpaceExpression* ex);
    public:
      void remove_canonical_expression(IndexSpaceExpression* expr);
      void record_empty_expression(IndexSpaceExpression* expr);
    public:
      // Methods for removing index space expression when they are done
      void remove_union_operation(
          IndexSpaceOperation* expr,
          const std::vector<IndexSpaceExpression*>& exprs);
      void remove_intersection_operation(
          IndexSpaceOperation* expr,
          const std::vector<IndexSpaceExpression*>& exprs);
      void remove_subtraction_operation(
          IndexSpaceOperation* expr, IndexSpaceExpression* lhs,
          IndexSpaceExpression* rhs);
    public:
      // Remote expression methods
      IndexSpaceExpression* find_or_create_remote_expression(
          IndexSpaceExprID remote_expr_id, Deserializer& derez, bool& created);
      void unregister_remote_expression(IndexSpaceExprID remote_expr_id);
    public:
      TraceID generate_dynamic_trace_id(bool check_context = true);
      TraceID generate_library_trace_ids(const char* name, size_t count);
      static TraceID& get_current_static_trace_id(void);
      static TraceID generate_static_trace_id(void);
    public:
      Mapper* get_mapper(MapperID id, Processor target);
      MappingCallInfo* begin_mapper_call(
          MapperID id, Processor target, Operation* op);
      void end_mapper_call(MappingCallInfo* info);
    public:
      bool is_MPI_interop_configured(void);
      const std::map<int, AddressSpace>& find_forward_MPI_mapping(void);
      const std::map<AddressSpace, int>& find_reverse_MPI_mapping(void);
      int find_local_MPI_rank(void);
    public:
      Mapping::MapperRuntime* get_mapper_runtime(void);
      MapperID generate_dynamic_mapper_id(bool check_context = true);
      MapperID generate_library_mapper_ids(const char* name, size_t count);
      static MapperID& get_current_static_mapper_id(void);
      static MapperID generate_static_mapper_id(void);
      void add_mapper(MapperID map_id, Mapper* mapper, Processor proc);
      void replace_default_mapper(Mapper* mapper, Processor proc);
      MapperManager* find_mapper(MapperID map_id);
      MapperManager* find_mapper(Processor target, MapperID map_id);
      bool has_non_default_mapper(void) const;
      static MapperManager* wrap_mapper(
          Mapper* mapper, MapperID map_id, Processor proc,
          bool is_default = false);
    public:
      ProjectionID generate_dynamic_projection_id(bool check_context = true);
      ProjectionID generate_library_projection_ids(
          const char* name, size_t cnt);
      static ProjectionID& get_current_static_projection_id(void);
      static ProjectionID generate_static_projection_id(void);
      void register_projection_functor(
          ProjectionID pid, ProjectionFunctor* func,
          bool need_zero_check = true, bool silence_warnings = false,
          const char* warning_string = nullptr, bool preregistered = false);
      static void preregister_projection_functor(
          ProjectionID pid, ProjectionFunctor* func);
      ProjectionFunction* find_projection_function(
          ProjectionID pid, bool can_fail = false);
      static ProjectionFunctor* get_projection_functor(ProjectionID pid);
      void unregister_projection_functor(ProjectionID pid);
    public:
      ShardingID generate_dynamic_sharding_id(bool check_context = true);
      ShardingID generate_library_sharding_ids(const char* name, size_t count);
      static ShardingID& get_current_static_sharding_id(void);
      static ShardingID generate_static_sharding_id(void);
      void register_sharding_functor(
          ShardingID sid, ShardingFunctor* func, bool need_zero_check = true,
          bool silence_warnings = false, const char* warning_string = nullptr,
          bool preregistered = false);
      static void preregister_sharding_functor(
          ShardingID sid, ShardingFunctor* func);
      ShardingFunctor* find_sharding_functor(
          ShardingID sid, bool can_fail = false);
      static ShardingFunctor* get_sharding_functor(ShardingID sid);
    public:
      ConcurrentID generate_dynamic_concurrent_id(bool check_context = true);
      ConcurrentID generate_library_concurrent_ids(
          const char* name, size_t count);
      static ConcurrentID& get_current_static_concurrent_id(void);
      static ConcurrentID generate_static_concurrent_id(void);
      void register_concurrent_functor(
          ConcurrentID cid, ConcurrentColoringFunctor* functor,
          bool need_zero_check = true, bool silence_warnings = false,
          const char* warning_string = nullptr, bool preregistered = false);
      static void preregister_concurrent_functor(
          ConcurrentID cid, ConcurrentColoringFunctor* functor);
      ConcurrentColoringFunctor* find_concurrent_coloring_functor(
          ConcurrentID cid, bool can_fail = false);
      static ConcurrentColoringFunctor* get_concurrent_functor(ConcurrentID id);
    public:
      void register_reduction(
          ReductionOpID redop_id, ReductionOp* redop, SerdezInitFunc init_func,
          SerdezFoldFunc fold_func, bool permit_duplicates, bool preregistered);
      void register_serdez(
          CustomSerdezID serdez_id, SerdezOp* serdez_op, bool permit_duplicates,
          bool preregistered);
      const ReductionOp* get_reduction(ReductionOpID redop_id);
      FillView* find_or_create_reduction_fill_view(ReductionOpID redop_id);
      const SerdezOp* get_serdez(CustomSerdezID serdez_id);
      const SerdezRedopFns* get_serdez_redop(ReductionOpID redop_id);
    public:
      void attach_semantic_information(
          TaskID task_id, SemanticTag, const void* buffer, size_t size,
          bool is_mutable, bool send_to_owner = true);
      void attach_semantic_information(
          IndexSpace handle, SemanticTag tag, const void* buffer, size_t size,
          bool is_mutable);
      void attach_semantic_information(
          IndexPartition handle, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable);
      void attach_semantic_information(
          FieldSpace handle, SemanticTag tag, const void* buffer, size_t size,
          bool is_mutable);
      void attach_semantic_information(
          FieldSpace handle, FieldID fid, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable);
      void attach_semantic_information(
          LogicalRegion handle, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable);
      void attach_semantic_information(
          LogicalPartition handle, SemanticTag tag, const void* buffer,
          size_t size, bool is_mutable);
    public:
      bool retrieve_semantic_information(
          TaskID task_id, SemanticTag tag, const void*& result, size_t& size,
          bool can_fail, bool wait_until);
      bool retrieve_semantic_information(
          IndexSpace handle, SemanticTag tag, const void*& result, size_t& size,
          bool can_fail, bool wait_until);
      bool retrieve_semantic_information(
          IndexPartition handle, SemanticTag tag, const void*& result,
          size_t& size, bool can_fail, bool wait_until);
      bool retrieve_semantic_information(
          FieldSpace handle, SemanticTag tag, const void*& result, size_t& size,
          bool can_fail, bool wait_until);
      bool retrieve_semantic_information(
          FieldSpace handle, FieldID fid, SemanticTag tag, const void*& result,
          size_t& size, bool can_fail, bool wait_until);
      bool retrieve_semantic_information(
          LogicalRegion handle, SemanticTag tag, const void*& result,
          size_t& size, bool can_fail, bool wait_until);
      bool retrieve_semantic_information(
          LogicalPartition part, SemanticTag tag, const void*& result,
          size_t& size, bool can_fail, bool wait_until);
    public:
      TaskID generate_dynamic_task_id(bool check_context = true);
      TaskID generate_library_task_ids(const char* name, size_t count);
      VariantID register_variant(
          const TaskVariantRegistrar& registrar, const void* user_data,
          size_t user_data_size, const CodeDescriptor& realm_desc,
          size_t return_type_size, bool has_return_type_size,
          VariantID vid = LEGION_AUTO_GENERATE_ID, bool check_task_id = true,
          bool check_context = true, bool preregistered = false);
      TaskImpl* find_or_create_task_impl(TaskID task_id);
      TaskImpl* find_task_impl(TaskID task_id);
      VariantImpl* find_variant_impl(
          TaskID task_id, VariantID variant_id, bool can_fail = false);
    public:
      ReductionOpID generate_dynamic_reduction_id(bool check_context = true);
      ReductionOpID generate_library_reduction_ids(
          const char* name, size_t count);
    public:
      CustomSerdezID generate_dynamic_serdez_id(bool check_context = true);
      CustomSerdezID generate_library_serdez_ids(
          const char* name, size_t count);
    public:
      // Memory manager functions
      MemoryManager* find_memory_manager(Memory mem);
    public:
      // Messaging functions
      MessageManager* find_messenger(AddressSpaceID sid);
      void handle_endpoint_creation(Deserializer& derez);
    public:
      void process_mapper_message(
          Processor target, MapperID map_id, Processor source,
          const void* message, size_t message_size, unsigned message_kind);
      void process_mapper_broadcast(
          MapperID map_id, Processor source, const void* message,
          size_t message_size, unsigned message_kind, int radix, int index);
    public:
      void send_task(IndividualTask* task);
      void send_task(SliceTask* task);
      void send_tasks(Processor target, std::vector<SingleTask*>& tasks);
      void send_steal_request(
          const std::multimap<Processor, MapperID>& targets, Processor thief);
      void send_advertisements(
          const std::set<Processor>& targets, MapperID map_id,
          Processor source);
    public:
      void handle_steal(Deserializer& derez);
      void handle_advertisement(Deserializer& derez);
#ifdef LEGION_USE_LIBDL
      void handle_registration_callback(Deserializer& derez);
#endif
      void handle_library_mapper_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_library_mapper_response(Deserializer& derez);
      void handle_library_trace_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_library_trace_response(Deserializer& derez);
      void handle_library_projection_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_library_projection_response(Deserializer& derez);
      void handle_library_sharding_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_library_sharding_response(Deserializer& derez);
      void handle_library_concurrent_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_library_concurrent_response(Deserializer& derez);
      void handle_library_task_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_library_task_response(Deserializer& derez);
      void handle_library_redop_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_library_redop_response(Deserializer& derez);
      void handle_library_serdez_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_library_serdez_response(Deserializer& derez);
#if 0
      void send_remote_task_replay(AddressSpaceID target, Serializer& rez);
      void send_remote_task_profiling_response(Processor tar, Serializer& rez);
      void send_shared_ownership(AddressSpaceID target, Serializer& rez);
      void send_index_space_request(AddressSpaceID target, Serializer& rez);
      void send_index_space_response(AddressSpaceID target, Serializer& rez);
      void send_index_space_return(AddressSpaceID target, Serializer& rez);
      void send_index_space_set(AddressSpaceID target, Serializer& rez);
      void send_index_space_child_request(
          AddressSpaceID target, Serializer& rez);
      void send_index_space_child_response(
          AddressSpaceID target, Serializer& rez);
      void send_index_space_colors_request(
          AddressSpaceID target, Serializer& rez);
      void send_index_space_colors_response(
          AddressSpaceID target, Serializer& rez);
      void send_index_space_generate_color_request(
          AddressSpaceID target, Serializer& rez);
      void send_index_space_generate_color_response(
          AddressSpaceID target, Serializer& rez);
      void send_index_space_release_color(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_notification(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_request(AddressSpaceID target, Serializer& rez);
      void send_index_partition_response(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_return(AddressSpaceID target, Serializer& rez);
      void send_index_partition_child_request(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_child_response(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_child_replication(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_disjoint_update(
          AddressSpaceID target, Serializer& rez,
          RtEvent pre = RtEvent::NO_RT_EVENT);
      void send_index_partition_shard_rects_request(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_shard_rects_response(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_remote_interference_request(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_remote_interference_response(
          AddressSpaceID target, Serializer& rez);
      void send_field_space_node(AddressSpaceID target, Serializer& rez);
      void send_field_space_request(AddressSpaceID target, Serializer& rez);
      void send_field_space_return(AddressSpaceID target, Serializer& rez);
      void send_field_space_allocator_request(
          AddressSpaceID target, Serializer& rez);
      void send_field_space_allocator_response(
          AddressSpaceID target, Serializer& rez);
      void send_field_space_allocator_invalidation(
          AddressSpaceID, Serializer& rez);
      void send_field_space_allocator_flush(
          AddressSpaceID target, Serializer& rez);
      void send_field_space_allocator_free(
          AddressSpaceID target, Serializer& rez);
      void send_field_space_infos_request(AddressSpaceID, Serializer& rez);
      void send_field_space_infos_response(AddressSpaceID, Serializer& rez);
      void send_field_alloc_request(AddressSpaceID target, Serializer& rez);
      void send_field_size_update(AddressSpaceID target, Serializer& rez);
      void send_field_free(AddressSpaceID target, Serializer& rez);
      void send_field_free_indexes(AddressSpaceID target, Serializer& rez);
      void send_field_space_layout_invalidation(
          AddressSpaceID target, Serializer& rez);
      void send_local_field_alloc_request(
          AddressSpaceID target, Serializer& rez);
      void send_local_field_alloc_response(
          AddressSpaceID target, Serializer& rez);
      void send_local_field_free(AddressSpaceID target, Serializer& rez);
      void send_local_field_update(AddressSpaceID target, Serializer& rez);
      void send_top_level_region_request(
          AddressSpaceID target, Serializer& rez);
      void send_top_level_region_return(AddressSpaceID target, Serializer& rez);
      void send_index_space_destruction(
          IndexSpace handle, AddressSpaceID target, std::set<RtEvent>& applied);
      void send_index_partition_destruction(
          IndexPartition handle, AddressSpaceID target,
          std::set<RtEvent>& applied);
      void send_field_space_destruction(
          FieldSpace handle, AddressSpaceID target, std::set<RtEvent>& applied);
      void send_logical_region_destruction(
          LogicalRegion handle, AddressSpaceID target,
          std::set<RtEvent>& applied);
      void send_individual_remote_future_size(
          Processor target, Serializer& rez);
      void send_individual_remote_output_registration(
          Processor target, Serializer& rez);
      void send_individual_remote_mapped(Processor target, Serializer& rez);
      void send_individual_remote_complete(Processor target, Serializer& rez);
      void send_individual_remote_commit(Processor target, Serializer& rez);
      void send_individual_concurrent_allreduce_request(
          Processor target, Serializer& rez);
      void send_individual_concurrent_allreduce_response(
          AddressSpaceID target, Serializer& rez);
      void send_slice_remote_mapped(Processor target, Serializer& rez);
      void send_slice_remote_complete(Processor target, Serializer& rez);
      void send_slice_remote_commit(Processor target, Serializer& rez);
      void send_slice_rendezvous_concurrent_mapped(
          Processor target, Serializer& rez);
      void send_slice_collective_allreduce_request(
          Processor target, Serializer& rez);
      void send_slice_collective_allreduce_response(
          AddressSpaceID target, Serializer& rez);
      void send_slice_concurrent_allreduce_request(
          Processor target, Serializer& rez);
      void send_slice_concurrent_allreduce_response(
          AddressSpaceID target, Serializer& rez);
      void send_slice_find_intra_space_dependence(
          Processor target, Serializer& rez);
      void send_slice_remote_rendezvous(Processor target, Serializer& rez);
      void send_slice_remote_versioning_rendezvous(
          Processor target_proc, Serializer& rez);
      void send_slice_remote_output_extents(Processor target, Serializer& rez);
      void send_slice_remote_output_registration(
          Processor target, Serializer& rez);
      void send_did_remote_registration(AddressSpaceID target, Serializer& rez);
      void send_did_downgrade_request(AddressSpaceID target, Serializer& rez);
      void send_did_downgrade_response(AddressSpaceID target, Serializer& rez);
      void send_did_downgrade_success(AddressSpaceID target, Serializer& rez);
      void send_did_downgrade_update(AddressSpaceID target, Serializer& rez);
      void send_did_downgrade_restart(AddressSpaceID target, Serializer& rez);
      void send_did_acquire_global_request(
          AddressSpaceID target, Serializer& rez);
      void send_did_acquire_global_response(
          AddressSpaceID target, Serializer& rez);
      void send_did_acquire_valid_request(
          AddressSpaceID target, Serializer& rez);
      void send_did_acquire_valid_response(
          AddressSpaceID target, Serializer& rez);
      void send_created_region_contexts(AddressSpaceID target, Serializer& rez);
      void send_back_atomic(AddressSpaceID target, Serializer& rez);
      void send_atomic_reservation_request(
          AddressSpaceID target, Serializer& rez);
      void send_atomic_reservation_response(
          AddressSpaceID target, Serializer& rez);
      void send_padded_reservation_request(
          AddressSpaceID target, Serializer& rez);
      void send_padded_reservation_response(
          AddressSpaceID target, Serializer& rez);
      void send_materialized_view(AddressSpaceID target, Serializer& rez);
      void send_fill_view(AddressSpaceID target, Serializer& rez);
      void send_fill_view_value(AddressSpaceID target, Serializer& rez);
      void send_phi_view(AddressSpaceID target, Serializer& rez);
      void send_reduction_view(AddressSpaceID target, Serializer& rez);
      void send_replicated_view(AddressSpaceID target, Serializer& rez);
      void send_allreduce_view(AddressSpaceID target, Serializer& rez);
      void send_instance_manager(AddressSpaceID target, Serializer& rez);
      void send_manager_update(AddressSpaceID target, Serializer& rez);
      void send_collective_distribute_fill(
          AddressSpaceID target, Serializer& rez);
      void send_collective_distribute_point(
          AddressSpaceID target, Serializer& rez);
      void send_collective_distribute_pointwise(
          AddressSpaceID target, Serializer& rez);
      void send_collective_distribute_reduction(
          AddressSpaceID target, Serializer& rez);
      void send_collective_distribute_broadcast(
          AddressSpaceID target, Serializer& rez);
      void send_collective_distribute_reducecast(
          AddressSpaceID target, Serializer& rez);
      void send_collective_distribute_hourglass(
          AddressSpaceID target, Serializer& rez);
      void send_collective_distribute_allreduce(
          AddressSpaceID target, Serializer& rez);
      void send_collective_hammer_reduction(
          AddressSpaceID target, Serializer& rez);
      void send_collective_fuse_gather(AddressSpaceID target, Serializer& rez);
      void send_collective_register_user_request(
          AddressSpaceID target, Serializer& rez);
      void send_collective_register_user_response(
          AddressSpaceID target, Serializer& rez);
      void send_collective_individual_register_user(
          AddressSpaceID target, Serializer& rez);
      void send_collective_point_request(
          AddressSpaceID target, Serializer& rez);
      void send_collective_point_response(
          AddressSpaceID target, Serializer& rez);
      void send_collective_remote_instances_request(
          AddressSpaceID target, Serializer& rez);
      void send_collective_remote_instances_response(
          AddressSpaceID target, Serializer& rez);
      void send_collective_nearest_instances_request(
          AddressSpaceID target, Serializer& rez);
      void send_collective_nearest_instances_response(
          AddressSpaceID target, Serializer& rez);
      void send_collective_remote_registration(
          AddressSpaceID target, Serializer& rez);
      void send_collective_deletion(AddressSpaceID target, Serializer& rez);
      void send_collective_finalize_mapping(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_creation(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_deletion(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_release(AddressSpaceID target, Serializer& rez);
      void send_collective_view_notification(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_make_valid(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_make_invalid(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_invalidate_request(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_invalidate_response(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_add_remote_reference(
          AddressSpaceID target, Serializer& rez);
      void send_collective_view_remove_remote_reference(
          AddressSpaceID target, Serializer& rez);
      void send_create_top_view_request(AddressSpaceID target, Serializer& rez);
      void send_create_top_view_response(
          AddressSpaceID target, Serializer& rez);
      void send_view_request(AddressSpaceID target, Serializer& rez);
      void send_view_register_user(AddressSpaceID target, Serializer& rez);
      void send_view_find_copy_preconditions_request(
          AddressSpaceID target, Serializer& rez);
      void send_view_add_copy_user(AddressSpaceID target, Serializer& rez);
      void send_view_find_last_users_request(
          AddressSpaceID target, Serializer& rez);
      void send_view_find_last_users_response(
          AddressSpaceID target, Serializer& rez);
      void send_future_result(AddressSpaceID target, Serializer& rez);
      void send_future_result_size(AddressSpaceID target, Serializer& rez);
      void send_future_subscription(AddressSpaceID target, Serializer& rez);
      void send_future_create_instance_request(
          AddressSpaceID target, Serializer& rez);
      void send_future_create_instance_response(
          AddressSpaceID target, Serializer& rez);
      void send_future_map_request_future(
          AddressSpaceID target, Serializer& rez);
      void send_future_map_response_future(
          AddressSpaceID target, Serializer& rez);
      void send_future_map_find_pointwise(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_compute_equivalence_sets(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_output_equivalence_set(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_refine_equivalence_sets(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_equivalence_set_notification(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_broadcast_update(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_created_regions(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_trace_event_request(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_trace_event_response(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_trace_event_trigger(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_trace_frontier_request(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_trace_frontier_response(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_trace_update(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_find_trace_local_sets(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_implicit_rendezvous(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_find_collective_view(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_pointwise_dependence(
          AddressSpaceID target, Serializer& rez);
      void send_mapper_message(AddressSpaceID target, Serializer& rez);
      void send_mapper_broadcast(AddressSpaceID target, Serializer& rez);
      void send_task_impl_semantic_request(
          AddressSpaceID target, Serializer& rez);
      void send_index_space_semantic_request(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_semantic_request(
          AddressSpaceID target, Serializer& rez);
      void send_field_space_semantic_request(
          AddressSpaceID target, Serializer& rez);
      void send_field_semantic_request(AddressSpaceID target, Serializer& rez);
      void send_logical_region_semantic_request(
          AddressSpaceID target, Serializer& rez);
      void send_logical_partition_semantic_request(
          AddressSpaceID target, Serializer& rez);
      void send_task_impl_semantic_info(AddressSpaceID target, Serializer& rez);
      void send_index_space_semantic_info(
          AddressSpaceID target, Serializer& rez);
      void send_index_partition_semantic_info(
          AddressSpaceID target, Serializer& rez);
      void send_field_space_semantic_info(
          AddressSpaceID target, Serializer& rez);
      void send_field_semantic_info(AddressSpaceID target, Serializer& rez);
      void send_logical_region_semantic_info(
          AddressSpaceID target, Serializer& rez);
      void send_logical_partition_semantic_info(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_request(AddressSpaceID target, Serializer& rez);
      void send_remote_context_response(AddressSpaceID target, Serializer& rez);
      void send_remote_context_physical_request(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_physical_response(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_find_collective_view_request(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_find_collective_view_response(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_collective_rendezvous(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_refine_equivalence_sets(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_pointwise_dependence(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_find_trace_local_sets_request(
          AddressSpaceID target, Serializer& rez);
      void send_remote_context_find_trace_local_sets_response(
          AddressSpaceID target, Serializer& rez);
      void send_compute_equivalence_sets_request(
          AddressSpaceID target, Serializer& rez);
      void send_compute_equivalence_sets_response(
          AddressSpaceID target, Serializer& rez);
      void send_compute_equivalence_sets_pending(
          AddressSpaceID target, Serializer& rez);
      void send_output_equivalence_set_request(
          AddressSpaceID target, Serializer& rez);
      void send_output_equivalence_set_response(
          AddressSpaceID target, Serializer& rez);
      void send_cancel_equivalence_sets_subscription(
          AddressSpaceID target, Serializer& rez);
      void send_invalidate_equivalence_sets_subscription(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_creation(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_reuse(AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_response(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_replication_request(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_replication_response(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_migration(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_owner_update(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_clone_request(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_clone_response(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_capture_request(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_capture_response(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_request_instances(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_request_invalid(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_request_antivalid(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_updates(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_acquires(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_releases(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_copies_across(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_overwrites(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_filters(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_remote_instances(
          AddressSpaceID target, Serializer& rez);
      void send_equivalence_set_filter_invalidations(
          AddressSpaceID target, Serializer& rez);
      void send_instance_request(AddressSpaceID target, Serializer& rez);
      void send_instance_response(AddressSpaceID target, Serializer& rez);
      void send_external_create_request(AddressSpaceID target, Serializer& rez);
      void send_external_create_response(
          AddressSpaceID target, Serializer& rez);
      void send_external_attach(AddressSpaceID target, Serializer& rez);
      void send_external_detach(AddressSpaceID target, Serializer& rez);
      void send_gc_priority_update(AddressSpaceID target, Serializer& rez);
      void send_gc_request(AddressSpaceID target, Serializer& rez);
      void send_gc_response(AddressSpaceID target, Serializer& rez);
      void send_gc_acquire(AddressSpaceID target, Serializer& rez);
      void send_gc_failed(AddressSpaceID target, Serializer& rez);
      void send_gc_mismatch(AddressSpaceID target, Serializer& rez);
      void send_gc_notify(AddressSpaceID target, Serializer& rez);
      void send_gc_debug_request(AddressSpaceID target, Serializer& rez);
      void send_gc_debug_response(AddressSpaceID target, Serializer& rez);
      void send_gc_record_event(AddressSpaceID target, Serializer& rez);
      void send_acquire_request(AddressSpaceID target, Serializer& rez);
      void send_acquire_response(AddressSpaceID target, Serializer& rez);
      void send_variant_broadcast(AddressSpaceID target, Serializer& rez);
      void send_constraint_request(AddressSpaceID target, Serializer& rez);
      void send_constraint_response(AddressSpaceID target, Serializer& rez);
      void send_constraint_release(AddressSpaceID target, Serializer& rez);
      void send_mpi_rank_exchange(AddressSpaceID target, Serializer& rez);
      void send_replicate_distribution(AddressSpaceID target, Serializer& rez);
      void send_replicate_collective_versioning(
          AddressSpaceID target, Serializer& rez);
      void send_replicate_collective_mapping(
          AddressSpaceID target, Serializer& rez);
      void send_replicate_rendezvous_virtual_mappings(
          AddressSpaceID target, Serializer& rez);
      void send_replicate_startup_complete(
          AddressSpaceID target, Serializer& rez);
      void send_replicate_post_mapped(AddressSpaceID target, Serializer& rez);
      void send_replicate_trigger_complete(
          AddressSpaceID target, Serializer& rez);
      void send_replicate_trigger_commit(
          AddressSpaceID target, Serializer& rez);
      void send_control_replicate_rendezvous_message(
          AddressSpaceID target, Serializer& rez);
      void send_library_mapper_request(AddressSpaceID target, Serializer& rez);
      void send_library_mapper_response(AddressSpaceID target, Serializer& rez);
      void send_library_trace_request(AddressSpaceID target, Serializer& rez);
      void send_library_trace_response(AddressSpaceID target, Serializer& rez);
      void send_library_projection_request(
          AddressSpaceID target, Serializer& rez);
      void send_library_projection_response(
          AddressSpaceID target, Serializer& rez);
      void send_library_sharding_request(
          AddressSpaceID target, Serializer& rez);
      void send_library_sharding_response(
          AddressSpaceID target, Serializer& rez);
      void send_library_concurrent_request(
          AddressSpaceID target, Serializer& rez);
      void send_library_concurrent_response(
          AddressSpaceID target, Serializer& rez);
      void send_library_task_request(AddressSpaceID target, Serializer& rez);
      void send_library_task_response(AddressSpaceID target, Serializer& rez);
      void send_library_redop_request(AddressSpaceID target, Serializer& rez);
      void send_library_redop_response(AddressSpaceID target, Serializer& rez);
      void send_library_serdez_request(AddressSpaceID target, Serializer& rez);
      void send_library_serdez_response(AddressSpaceID target, Serializer& rez);
      void send_remote_op_report_uninitialized(
          AddressSpaceID target, Serializer& rez);
      void send_remote_op_profiling_count_update(
          AddressSpaceID target, Serializer& rez);
      void send_remote_op_completion_effect(
          AddressSpaceID target, Serializer& rez);
      void send_remote_trace_update(AddressSpaceID target, Serializer& rez);
      void send_remote_trace_response(AddressSpaceID target, Serializer& rez);
      void send_free_external_allocation(
          AddressSpaceID target, Serializer& rez);
      void send_notify_collected_instances(
          AddressSpaceID target, Serializer& rez);
      void send_create_memory_pool_request(
          AddressSpaceID target, Serializer& rez);
      void send_create_memory_pool_response(
          AddressSpaceID target, Serializer& rez);
      void send_create_future_instance_request(
          AddressSpaceID target, Serializer& rez);
      void send_create_future_instance_response(
          AddressSpaceID target, Serializer& rez);
      void send_free_future_instance(AddressSpaceID target, Serializer& rez);
      void send_profiler_event_trigger(AddressSpaceID target, Serializer& rez);
      void send_profiler_event_poison(AddressSpaceID target, Serializer& rez);
      void send_shutdown_notification(AddressSpaceID target, Serializer& rez);
      void send_shutdown_response(AddressSpaceID target, Serializer& rez);
    public:
      // Complementary tasks for handling messages
      void handle_startup_barrier(Deserializer& derez);
      void handle_task(Deserializer& derez); 
      void handle_remote_task_replay(Deserializer& derez);
      void handle_remote_task_profiling_response(Deserializer& derez);
      void handle_shared_ownership(Deserializer& derez);
      void handle_index_space_request(Deserializer& derez);
      void handle_index_space_response(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_space_return(Deserializer& derez);
      void handle_index_space_set(Deserializer& derez, AddressSpaceID source);
      void handle_index_space_child_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_space_child_response(Deserializer& derez);
      void handle_index_space_colors_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_space_colors_response(Deserializer& derez);
      void handle_index_space_generate_color_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_space_generate_color_response(Deserializer& derez);
      void handle_index_space_release_color(Deserializer& derez);
      void handle_index_partition_notification(Deserializer& derez);
      void handle_index_partition_request(Deserializer& derez);
      void handle_index_partition_response(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_partition_return(Deserializer& derez);
      void handle_index_partition_child_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_partition_child_response(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_partition_child_replication(Deserializer& derez);
      void handle_index_partition_disjoint_update(Deserializer& derez);
      void handle_index_partition_shard_rects_request(Deserializer& derez);
      void handle_index_partition_shard_rects_response(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_partition_remote_interference_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_partition_remote_interference_response(
          Deserializer& derez);
      void handle_field_space_node(Deserializer& derez, AddressSpaceID source);
      void handle_field_space_request(Deserializer& derez);
      void handle_field_space_return(Deserializer& derez);
      void handle_field_space_allocator_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_field_space_allocator_response(Deserializer& derez);
      void handle_field_space_allocator_invalidation(Deserializer& derez);
      void handle_field_space_allocator_flush(Deserializer& derez);
      void handle_field_space_allocator_free(
          Deserializer& derez, AddressSpaceID source);
      void handle_field_space_infos_request(Deserializer& derez);
      void handle_field_space_infos_response(Deserializer& derez);
      void handle_field_alloc_request(Deserializer& derez);
      void handle_field_size_update(Deserializer& derez, AddressSpaceID source);
      void handle_field_free(Deserializer& derez, AddressSpaceID source);
      void handle_field_free_indexes(Deserializer& derez);
      void handle_field_space_layout_invalidation(
          Deserializer& derez, AddressSpaceID source);
      void handle_local_field_alloc_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_local_field_alloc_response(Deserializer& derez);
      void handle_local_field_free(Deserializer& derez);
      void handle_local_field_update(Deserializer& derez);
      void handle_top_level_region_request(Deserializer& derez);
      void handle_top_level_region_return(
          Deserializer& derez, AddressSpaceID source); 
      void handle_individual_remote_future_size(Deserializer& derez);
      void handle_individual_remote_output_registration(Deserializer& derez);
      void handle_individual_remote_mapped(Deserializer& derez);
      void handle_individual_remote_complete(Deserializer& derez);
      void handle_individual_remote_commit(Deserializer& derez);
      void handle_individual_concurrent_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_individual_concurrent_response(Deserializer& derez);
      void handle_slice_remote_mapped(
          Deserializer& derez, AddressSpaceID source);
      void handle_slice_remote_complete(Deserializer& derez);
      void handle_slice_remote_commit(Deserializer& derez);
      void handle_slice_rendezvous_concurrent_mapped(Deserializer& derez);
      void handle_slice_collective_allreduce_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_slice_collective_allreduce_response(Deserializer& derez);
      void handle_slice_concurrent_allreduce_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_slice_concurrent_allreduce_response(Deserializer& derez);
      void handle_slice_find_intra_dependence(Deserializer& derez);
      void handle_slice_remote_collective_rendezvous(
          Deserializer& derez, AddressSpaceID source);
      void handle_slice_remote_collective_versioning_rendezvous(
          Deserializer& derez);
      void handle_slice_remote_output_extents(Deserializer& derez);
      void handle_slice_remote_output_registration(Deserializer& derez);
      void handle_did_remote_registration(
          Deserializer& derez, AddressSpaceID source);
      void handle_did_downgrade_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_did_downgrade_response(Deserializer& derez);
      void handle_did_downgrade_success(Deserializer& derez);
      void handle_did_downgrade_update(Deserializer& derez);
      void handle_did_downgrade_restart(
          Deserializer& derez, AddressSpaceID source);
      void handle_did_global_acquire_request(Deserializer& derez);
      void handle_did_global_acquire_response(Deserializer& derez);
      void handle_did_valid_acquire_request(Deserializer& derez);
      void handle_did_valid_acquire_response(Deserializer& derez);
      void handle_created_region_contexts(Deserializer& derez);
      void handle_send_atomic_reservation_request(Deserializer& derez);
      void handle_send_atomic_reservation_response(Deserializer& derez);
      void handle_send_padded_reservation_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_send_padded_reservation_response(Deserializer& derez);
      void handle_send_materialized_view(Deserializer& derez);
      void handle_send_fill_view(Deserializer& derez);
      void handle_send_fill_view_value(Deserializer& derez);
      void handle_send_phi_view(Deserializer& derez);
      void handle_send_reduction_view(Deserializer& derez);
      void handle_send_replicated_view(Deserializer& derez);
      void handle_send_allreduce_view(Deserializer& derez);
      void handle_send_instance_manager(
          Deserializer& derez, AddressSpaceID source);
      void handle_send_manager_update(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_distribute_fill(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_distribute_point(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_distribute_pointwise(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_distribute_reduction(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_distribute_broadcast(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_distribute_reducecast(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_distribute_hourglass(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_distribute_allreduce(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_hammer_reduction(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_fuse_gather(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_user_request(Deserializer& derez);
      void handle_collective_user_response(Deserializer& derez);
      void handle_collective_user_registration(Deserializer& derez);
      void handle_collective_remote_instances_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_remote_instances_response(
          Deserializer& derez, AddressSpaceID source);
      void handle_collective_nearest_instances_request(Deserializer& derez);
      void handle_collective_nearest_instances_response(Deserializer& derez);
      void handle_collective_remote_registration(Deserializer& derez);
      void handle_collective_finalize_mapping(Deserializer& derez);
      void handle_collective_view_creation(Deserializer& derez);
      void handle_collective_view_deletion(Deserializer& derez);
      void handle_collective_view_release(Deserializer& derez);
      void handle_collective_view_notification(Deserializer& derez);
      void handle_collective_view_make_valid(Deserializer& derez);
      void handle_collective_view_make_invalid(Deserializer& derez);
      void handle_collective_view_invalidate_request(Deserializer& derez);
      void handle_collective_view_invalidate_response(Deserializer& derez);
      void handle_collective_view_add_remote_reference(Deserializer& derez);
      void handle_collective_view_remove_remote_reference(Deserializer& derez);
      void handle_create_top_view_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_create_top_view_response(Deserializer& derez);
      void handle_view_request(Deserializer& derez);
      void handle_view_register_user(
          Deserializer& derez, AddressSpaceID source);
      void handle_view_copy_pre_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_view_add_copy_user(
          Deserializer& derez, AddressSpaceID source);
      void handle_view_find_last_users_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_view_find_last_users_response(Deserializer& derez);
      void handle_manager_request(Deserializer& derez);
      void handle_future_result(Deserializer& derez);
      void handle_future_result_size(
          Deserializer& derez, AddressSpaceID source);
      void handle_future_subscription(
          Deserializer& derez, AddressSpaceID source);
      void handle_future_create_instance_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_future_create_instance_response(Deserializer& derez);
      void handle_future_map_future_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_future_map_future_response(Deserializer& derez);
      void handle_future_map_find_pointwise(Deserializer& derez);
      void handle_mapper_message(Deserializer& derez);
      void handle_mapper_broadcast(Deserializer& derez);
      void handle_task_impl_semantic_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_space_semantic_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_partition_semantic_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_field_space_semantic_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_field_semantic_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_logical_region_semantic_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_logical_partition_semantic_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_task_impl_semantic_info(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_space_semantic_info(
          Deserializer& derez, AddressSpaceID source);
      void handle_index_partition_semantic_info(
          Deserializer& derez, AddressSpaceID source);
      void handle_field_space_semantic_info(
          Deserializer& derez, AddressSpaceID source);
      void handle_field_semantic_info(
          Deserializer& derez, AddressSpaceID source);
      void handle_logical_region_semantic_info(
          Deserializer& derez, AddressSpaceID source);
      void handle_logical_partition_semantic_info(
          Deserializer& derez, AddressSpaceID source);
      void handle_remote_context_request(Deserializer& derez);
      void handle_remote_context_response(Deserializer& derez);
      void handle_remote_context_physical_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_remote_context_physical_response(Deserializer& derez);
      void handle_remote_context_find_collective_view_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_remote_context_find_collective_view_response(
          Deserializer& derez);
      void handle_remote_context_refine_equivalence_sets(Deserializer& derez);
      void handle_remote_context_pointwise_dependence(Deserializer& derez);
      void handle_remote_context_find_trace_local_sets_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_remote_context_find_trace_local_sets_response(
          Deserializer& derez);
      void handle_compute_equivalence_sets_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_compute_equivalence_sets_response(Deserializer& derez);
      void handle_compute_equivalence_sets_pending(Deserializer& derez);
      void handle_output_equivalence_set_request(Deserializer& derez);
      void handle_output_equivalence_set_response(
          Deserializer& derez, AddressSpaceID source);
      void handle_cancel_equivalence_sets_subscription(
          Deserializer& derez, AddressSpaceID source);
      void handle_invalidate_equivalence_sets_subscription(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_creation(Deserializer& derez);
      void handle_equivalence_set_reuse(Deserializer& derez);
      void handle_equivalence_set_request(Deserializer& derez);
      void handle_equivalence_set_response(Deserializer& derez);
      void handle_equivalence_set_invalidate_trackers(Deserializer& derez);
      void handle_equivalence_set_replication_request(Deserializer& derez);
      void handle_equivalence_set_replication_response(Deserializer& derez);
      void handle_equivalence_set_migration(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_owner_update(Deserializer& derez);
      void handle_equivalence_set_clone_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_clone_response(Deserializer& derez);
      void handle_equivalence_set_capture_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_capture_response(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_remote_request_instances(
          Deserializer& derez, AddressSpaceID srouce);
      void handle_equivalence_set_remote_request_invalid(
          Deserializer& derez, AddressSpaceID srouce);
      void handle_equivalence_set_remote_request_antivalid(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_remote_updates(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_remote_acquires(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_remote_releases(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_remote_copies_across(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_remote_overwrites(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_remote_filters(
          Deserializer& derez, AddressSpaceID source);
      void handle_equivalence_set_remote_instances(Deserializer& derez);
      void handle_equivalence_set_filter_invalidations(Deserializer& derez);
      void handle_instance_request(Deserializer& derez, AddressSpaceID source);
      void handle_instance_response(Deserializer& derez, AddressSpaceID source);
      void handle_external_create_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_external_create_response(Deserializer& derez);
      void handle_external_attach(Deserializer& derez);
      void handle_external_detach(Deserializer& derez);
      void handle_gc_priority_update(
          Deserializer& derez, AddressSpaceID source);
      void handle_gc_request(Deserializer& derez, AddressSpaceID source);
      void handle_gc_response(Deserializer& derez);
      void handle_gc_acquire(Deserializer& derez);
      void handle_gc_failed(Deserializer& derez);
      void handle_gc_mismatch(Deserializer& derez);
      void handle_gc_notify(Deserializer& derez);
      void handle_gc_debug_request(Deserializer& derez, AddressSpaceID source);
      void handle_gc_debug_response(Deserializer& derez);
      void handle_gc_record_event(Deserializer& derez);
      void handle_acquire_request(Deserializer& derez, AddressSpaceID source);
      void handle_acquire_response(Deserializer& derez, AddressSpaceID source);
      void handle_variant_request(Deserializer& derez, AddressSpaceID source);
      void handle_variant_response(Deserializer& derez);
      void handle_variant_broadcast(Deserializer& derez);
      void handle_constraint_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_constraint_response(Deserializer& derez, AddressSpaceID src);
      void handle_top_level_task_complete(Deserializer& derez);
      void handle_mpi_rank_exchange(Deserializer& derez);
      void handle_replicate_distribution(Deserializer& derez);
      void handle_replicate_collective_versioning(Deserializer& derez);
      void handle_replicate_collective_mapping(Deserializer& derez);
      void handle_replicate_virtual_rendezvous(Deserializer& derez);
      void handle_replicate_startup_complete(Deserializer& derez);
      void handle_replicate_post_mapped(Deserializer& derez);
      void handle_replicate_trigger_complete(Deserializer& derez);
      void handle_replicate_trigger_commit(Deserializer& derez);
      void handle_control_replicate_rendezvous_message(Deserializer& derez);
      void handle_control_replicate_compute_equivalence_sets(
          Deserializer& derez);
      void handle_control_replicate_output_equivalence_set(Deserializer& derez);
      void handle_control_replicate_refine_equivalence_sets(
          Deserializer& derez);
      void handle_control_replicate_equivalence_set_notification(
          Deserializer& derez);
      void handle_control_replicate_broadcast_update(Deserializer& derez);
      void handle_control_replicate_created_regions(Deserializer& derez);
      void handle_control_replicate_trace_event_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_control_replicate_trace_event_response(Deserializer& derez);
      void handle_control_replicate_trace_event_trigger(Deserializer& derez);
      void handle_control_replicate_trace_frontier_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_control_replicate_trace_frontier_response(
          Deserializer& derez);
      void handle_control_replicate_trace_update(
          Deserializer& derez, AddressSpaceID source);
      void handle_control_replicate_find_trace_local_sets(
          Deserializer& derez, AddressSpaceID source);
      void handle_control_replicate_implicit_rendezvous(Deserializer& derez);
      void handle_control_replicate_find_collective_view(Deserializer& derez);
      void handle_control_replicate_pointwise_dependence(Deserializer& derez); 
      void handle_remote_op_report_uninitialized(Deserializer& derez);
      void handle_remote_op_profiling_count_update(Deserializer& derez);
      void handle_remote_op_completion_effect(Deserializer& derez);
      void handle_remote_tracing_update(
          Deserializer& derez, AddressSpaceID source);
      void handle_remote_tracing_response(Deserializer& derez);
      void handle_free_external_allocation(Deserializer& derez);
      void handle_notify_collected_instances(Deserializer& derez);
      void handle_create_memory_pool_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_create_memory_pool_response(Deserializer& derez);
      void handle_create_future_instance_request(
          Deserializer& derez, AddressSpaceID source);
      void handle_create_future_instance_response(Deserializer& derez);
      void handle_free_future_instance(Deserializer& derez);
      void handle_shutdown_notification(
          Deserializer& derez, AddressSpaceID source);
      void handle_shutdown_response(Deserializer& derez);
#endif
    public:  // Calls to handle mapper requests
      bool create_physical_instance(
          Memory target_memory, const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions,
          const TaskTreeCoordinates& coordinates, MappingInstance& result,
          Processor processor, bool acquire, GCPriority priority,
          bool tight_bounds, const LayoutConstraint** unsat, size_t* footprint,
          UniqueID creator_id, RtEvent* safe_for_unbounded_pools);
      bool create_physical_instance(
          Memory target_memory, LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions,
          const TaskTreeCoordinates& coordinates, MappingInstance& result,
          Processor processor, bool acquire, GCPriority priority,
          bool tight_bounds, const LayoutConstraint** unsat, size_t* footprint,
          UniqueID creator_id, RtEvent* safe_for_unbounded_pools);
      bool find_or_create_physical_instance(
          Memory target_memory, const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions,
          const TaskTreeCoordinates& coordinates, MappingInstance& result,
          bool& created, Processor processor, bool acquire, GCPriority priority,
          bool tight_bounds, const LayoutConstraint** unsat, size_t* footprint,
          UniqueID creator_id, RtEvent* safe_for_unbounded_pools);
      bool find_or_create_physical_instance(
          Memory target_memory, LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions,
          const TaskTreeCoordinates& coordinates, MappingInstance& result,
          bool& created, Processor processor, bool acquire, GCPriority priority,
          bool tight_bounds, const LayoutConstraint** unsat, size_t* footprint,
          UniqueID creator_id, RtEvent* safe_for_unbounded_pools);
      bool find_physical_instance(
          Memory target_memory, const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions, MappingInstance& result,
          bool acquire, bool tight_region_bounds);
      bool find_physical_instance(
          Memory target_memory, LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions, MappingInstance& result,
          bool acquire, bool tight_region_bounds);
      void find_physical_instances(
          Memory target_memory, const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions,
          std::vector<MappingInstance>& results, bool acquire,
          bool tight_region_bounds);
      void find_physical_instances(
          Memory target_memory, LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions,
          std::vector<MappingInstance>& result, bool acquire,
          bool tight_region_bounds);
      void release_tree_instances(RegionTreeID tid);
    public:
      // Manage the execution of tasks within a context
      void activate_context(InnerContext* context);
      void deactivate_context(InnerContext* context);
    public:
      void add_to_ready_queue(Processor p, SingleTask* task);
    public:
      inline Processor find_local_group(void) { return local_group; }
      inline Processor find_utility_group(void) { return utility_group; }
      Processor find_processor_group(const std::vector<Processor>& procs);
      ProcessorMask find_processor_mask(const std::vector<Processor>& procs);
      template<typename T>
      inline RtEvent issue_runtime_meta_task(
          const LgTaskArgs<T>& args, LgPriority lg_priority,
          RtEvent precondition = RtEvent::NO_RT_EVENT,
          Processor proc = Processor::NO_PROC);
      template<typename T>
      inline RtEvent issue_application_processor_task(
          const LgTaskArgs<T>& args, LgPriority lg_priority,
          const Processor proc, RtEvent precondition = RtEvent::NO_RT_EVENT);
    public:
      // Support for concurrent index task execution
      void order_concurrent_task_launch(
          Processor proc, SingleTask* task, ApEvent precondition,
          ApUserEvent ready, VariantID vid);
      void end_concurrent_task(Processor proc);
    public:
      DistributedID get_next_static_distributed_id(uint64_t& next_did);
      DistributedID get_available_distributed_id(void);
      DistributedID get_remote_distributed_id(AddressSpaceID from);
      AddressSpaceID determine_owner(DistributedID did) const;
      size_t find_distance(AddressSpaceID src, AddressSpaceID dst) const;
    public:
      void register_distributed_collectable(
          DistributedID did, DistributedCollectable* dc);
      void unregister_distributed_collectable(DistributedID did);
      bool has_distributed_collectable(DistributedID did);
      DistributedCollectable* find_distributed_collectable(DistributedID did);
      DistributedCollectable* weak_find_distributed_collectable(
          DistributedID did);
      template<typename T>
      void* find_or_create_pending_collectable_location(DistributedID did);
    public:
      LogicalView* find_or_request_logical_view(
          DistributedID did, RtEvent& ready);
      PhysicalManager* find_or_request_instance_manager(
          DistributedID did, RtEvent& ready);
      EquivalenceSet* find_or_request_equivalence_set(
          DistributedID did, RtEvent& ready);
      InnerContext* find_or_request_inner_context(DistributedID did);
      ShardManager* find_shard_manager(
          DistributedID did, bool can_fail = false);
    protected:
      template<typename T, typename AM>
      DistributedCollectable* find_or_request_distributed_collectable(
          DistributedID did, RtEvent& ready);
    public:
      FutureImpl* find_or_create_future(
          DistributedID did, DistributedID ctx_did,
          const ContextCoordinate& coordinate, Provenance* provenance,
          bool has_global_reference,
          // Can be ignored with global ref
          RtEvent& registered, Operation* op = nullptr, GenerationID op_gen = 0,
          UniqueID op_uid = 0, int op_depth = 0,
          CollectiveMapping* mapping = nullptr);
      FutureMapImpl* find_or_create_future_map(
          DistributedID did, TaskContext* ctx, uint64_t coord,
          IndexSpace domain, Provenance* provenance,
          const std::optional<uint64_t>& ctx_index);
      IndexSpace find_or_create_index_slice_space(
          const Domain& launch_domain, bool take_ownership, TypeTag type_tag,
          Provenance* provenance);
    public:
      void increment_outstanding_top_level_tasks(void);
      void decrement_outstanding_top_level_tasks(void);
    public:
      void issue_runtime_shutdown_attempt(void);
      void initiate_runtime_shutdown(
          AddressSpaceID source, ShutdownManager::ShutdownPhase phase,
          ShutdownManager* owner = nullptr);
      void confirm_runtime_shutdown(
          ShutdownManager* shutdown_manager, bool phase_one);
      void prepare_runtime_shutdown(void);
    public:
      bool has_outstanding_tasks(void);
      void increment_total_outstanding_tasks(unsigned tid, bool meta);
      void decrement_total_outstanding_tasks(unsigned tid, bool meta);
    public:
      template<typename OP>
      inline OP* get_operation(void)
      {
        OP* result = nullptr;
        {
          AutoLock op_lock(operation_lock);
          operation_industry.create(result);
#ifdef LEGION_DEBUG
          legion_assert(
              outstanding_operations.find(result) ==
              outstanding_operations.end());
          outstanding_operations.insert(result);
#endif
        }
        result->activate();
        return result;
      }
      template<typename OP>
      inline void free_operation(OP* op)
      {
        AutoLock op_lock(operation_lock);
#ifdef LEGION_DEBUG
        std::set<Operation*>::iterator finder = outstanding_operations.find(op);
        legion_assert(finder != outstanding_operations.end());
        outstanding_operations.erase(finder);
#endif
        operation_industry.recycle(op);
      }
    public:
      ContextID allocate_region_tree_context(void);
      void invalidate_region_tree_context(
          ContextID ctx, const RegionRequirement& req,
          bool filter_specific_fields);
      void check_region_tree_context(ContextID ctx);
      void free_region_tree_context(ContextID tree_ctx);
      inline AddressSpaceID get_runtime_owner(UniqueID uid) const
      {
        return (uid % total_address_spaces);
      }
    public:
      bool is_local(Processor proc) const;
      ProcessorManager* find_processor_manager(Processor proc) const;
      bool is_visible_memory(Processor proc, Memory mem);
      void find_visible_memories(Processor proc, std::set<Memory>& visible);
      Memory find_local_memory(Processor proc, Memory::Kind mem_kind);
    public:
      DistributedID get_unique_index_space_id(void);
      DistributedID get_unique_index_partition_id(void);
      DistributedID get_unique_field_space_id(void);
      IndexTreeID get_unique_index_tree_id(void);
      RegionTreeID get_unique_region_tree_id(void);
      UniqueID get_unique_operation_id(void);
      FieldID get_unique_field_id(void);
      CodeDescriptorID get_unique_code_descriptor_id(void);
      LayoutConstraintID get_unique_constraint_id(void);
      IndexSpaceExprID get_unique_index_space_expr_id(void);
      uint64_t get_unique_top_level_task_id(void);
      uint64_t get_unique_implicit_top_level_task_id(void);
      unsigned get_unique_indirections_id(void);
    public:
      Provenance* find_or_create_provenance(const char* prov, size_t length);
    public:
      // Methods for helping with dumb nested class scoping problems
      IndexSpace help_create_index_space_handle(TypeTag type_tag);
    public:
      unsigned generate_random_integer(void);
#ifdef LEGION_TRACE_ALLOCATION
    public:
      void trace_allocation(const std::type_info& info, size_t size, int elems);
      void trace_free(const std::type_info& info, size_t size, int elems);
      void dump_allocation_info(void);
#endif
    public:
      // These are the static methods that become the meta-tasks
      // for performing all the needed runtime operations
      static void startup_runtime_task(
          const void* args, size_t arglen, const void* userdata, size_t userlen,
          Processor p);
      static void shutdown_runtime_task(
          const void* args, size_t arglen, const void* userdata, size_t userlen,
          Processor p);
      static void legion_runtime_task(
          const void* args, size_t arglen, const void* userdata, size_t userlen,
          Processor p);
      static void profiling_runtime_task(
          const void* args, size_t arglen, const void* userdata, size_t userlen,
          Processor p);
      static void endpoint_runtime_task(
          const void* args, size_t arglen, const void* userdata, size_t userlen,
          Processor p);
      static void application_processor_runtime_task(
          const void* args, size_t arglen, const void* userdata, size_t userlen,
          Processor p);
    protected:
      static RtBarrier find_or_wait_for_startup_barrier(void);
    protected:
      bool prepared_for_shutdown;
    protected:
#ifdef LEGION_DEBUG
      mutable LocalLock outstanding_task_lock;
      std::map<std::pair<unsigned, bool>, unsigned> outstanding_task_counts;
      unsigned total_outstanding_tasks;
#else
      std::atomic<unsigned> total_outstanding_tasks;
#endif
      std::atomic<unsigned> outstanding_top_level_tasks;
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
    public:
      std::vector<std::atomic<int> > outstanding_counts;
#endif
    public:
      // Internal runtime state
      // The local processor managed by this runtime
      const std::set<Processor> local_procs;
    protected:
      // The local utility processors owned by this runtime
      const std::set<Processor> local_utils;
      // Processor managers for each of the local processors
      std::map<Processor, ProcessorManager*> proc_managers;
      // Lock for looking up memory managers
      mutable LocalLock memory_manager_lock;
      // Lock for initializing message managers
      mutable LocalLock message_manager_lock;
      // Memory managers for all the memories we know about
      std::map<Memory, MemoryManager*> memory_managers;
      // Message managers for each of the other runtimes
      std::atomic<MessageManager*> message_managers[LEGION_MAX_NUM_NODES];
      // Pending message manager requests
      std::map<AddressSpaceID, RtUserEvent> pending_endpoint_requests;
    protected:
      // The task table
      mutable LocalLock task_variant_lock;
      std::map<TaskID, TaskImpl*> task_table;
      std::deque<VariantImpl*> variant_table;
    protected:
      // Constraint sets
      mutable LocalLock layout_constraints_lock;
      std::map<LayoutConstraintID, LayoutConstraints*> layout_constraints_table;
      std::map<LayoutConstraintID, RtEvent> pending_constraint_requests;
    protected:
      struct MapperInfo {
        MapperInfo(void) : proc(Processor::NO_PROC), map_id(0) { }
        MapperInfo(Processor p, MapperID mid) : proc(p), map_id(mid) { }
      public:
        Processor proc;
        MapperID map_id;
      };
      mutable LocalLock mapper_info_lock;
      // For every mapper remember its mapper ID and processor
      std::map<Mapper*, MapperInfo> mapper_infos;
    protected:
      std::atomic<unsigned> unique_index_tree_id;
      std::atomic<unsigned> unique_field_id;
      std::atomic<unsigned long long> unique_operation_id;
      std::atomic<unsigned long long> unique_code_descriptor_id;
      std::atomic<unsigned long long> unique_constraint_id;
      std::atomic<unsigned long long> unique_is_expr_id;
      std::atomic<uint64_t> unique_top_level_task_id;
      uint64_t unique_implicit_top_level_task_id;
      std::atomic<unsigned> unique_indirections_id;
      std::atomic<unsigned> unique_task_id;
      std::atomic<unsigned> unique_mapper_id;
      std::atomic<unsigned> unique_trace_id;
      std::atomic<unsigned> unique_projection_id;
      std::atomic<unsigned> unique_sharding_id;
      std::atomic<unsigned> unique_concurrent_id;
      std::atomic<unsigned> unique_redop_id;
      std::atomic<unsigned> unique_serdez_id;
    protected:
      mutable LocalLock lookup_lock;
      mutable LocalLock lookup_is_op_lock;
      mutable LocalLock congruence_lock;
    private:
      // The lookup lock must be held when accessing these
      // data structures
      std::map<IndexSpace, IndexSpaceNode*> index_nodes;
      std::map<IndexPartition, IndexPartNode*> index_parts;
      std::map<FieldSpace, FieldSpaceNode*> field_nodes;
      std::map<LogicalRegion, RegionNode*> region_nodes;
      std::map<LogicalPartition, PartitionNode*> part_nodes;
      std::map<RegionTreeID, RegionNode*> tree_nodes;
    private:
      // pending events for requested nodes
      std::map<IndexSpace, RtEvent> index_space_requests;
      std::map<IndexPartition, RtEvent> index_part_requests;
      std::map<FieldSpace, RtEvent> field_space_requests;
      std::map<RegionTreeID, RtEvent> region_tree_requests;
    private:
      std::map<DistributedID, RtUserEvent> pending_index_spaces;
      std::map<DistributedID, RtUserEvent> pending_partitions;
      std::map<DistributedID, RtUserEvent> pending_field_spaces;
      std::map<RegionTreeID, RtUserEvent> pending_region_trees;
    private:
      // Index space operations
      std::map<IndexSpaceExprID /*first*/, ExpressionTrieNode*> union_ops;
      std::map<IndexSpaceExprID /*first*/, ExpressionTrieNode*>
          intersection_ops;
      std::map<IndexSpaceExprID /*lhs*/, ExpressionTrieNode*> difference_ops;
      // Remote expressions
      std::map<IndexSpaceExprID, IndexSpaceExpression*> remote_expressions;
      std::map<IndexSpaceExprID, RtEvent> pending_remote_expressions;
      std::vector<IndexSpaceExpression*> empty_expressions;
      static constexpr unsigned MAX_EXPRESSION_FANOUT = 32;
    private:
      // In order for the symbolic analysis to work, we need to know that
      // we don't have multiple symbols for congruent expressions. This data
      // structure is used to find congruent expressions where they exist
      typedef SmallPointerVector<IndexSpaceExpression, true> CanonicalSet;
      std::unordered_map<uint64_t /*hash*/, CanonicalSet> canonical_expressions;
    private:
      mutable LocalLock provenance_lock;
      std::unordered_map<size_t, Provenance*> provenances;
    protected:
      mutable LocalLock library_lock;
      struct LibraryMapperIDs {
      public:
        MapperID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string, LibraryMapperIDs> library_mapper_ids;
      // This is only valid on node 0
      unsigned unique_library_mapper_id;
    protected:
      struct LibraryTraceIDs {
        TraceID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string, LibraryTraceIDs> library_trace_ids;
      // This is only valid on node 0
      unsigned unique_library_trace_id;
    protected:
      struct LibraryProjectionIDs {
      public:
        ProjectionID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string, LibraryProjectionIDs> library_projection_ids;
      // This is only valid on node 0
      unsigned unique_library_projection_id;
    protected:
      struct LibraryShardingIDs {
      public:
        ShardingID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string, LibraryShardingIDs> library_sharding_ids;
      // This is only valid on node 0
      unsigned unique_library_sharding_id;
    protected:
      struct LibraryConcurrentIDs {
      public:
        ConcurrentID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string, LibraryConcurrentIDs> library_concurrent_ids;
      // This is only valid on node 0
      unsigned unique_library_concurrent_id;
    protected:
      struct LibraryTaskIDs {
      public:
        TaskID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string, LibraryTaskIDs> library_task_ids;
      // This is only valid on node 0
      unsigned unique_library_task_id;
    protected:
      struct LibraryRedopIDs {
      public:
        ReductionOpID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string, LibraryRedopIDs> library_redop_ids;
      // This is only valid on node 0
      unsigned unique_library_redop_id;
    protected:
      struct LibrarySerdezIDs {
      public:
        CustomSerdezID result;
        size_t count;
        RtEvent ready;
        bool result_set;
      };
      std::map<std::string, LibrarySerdezIDs> library_serdez_ids;
      // This is only valid on node 0
      unsigned unique_library_serdez_id;
    protected:
      mutable LocalLock callback_lock;
#ifdef LEGION_USE_LIBDL
      // Have this be a member variable so that it keeps references
      // to all the dynamic objects that we load
      Realm::DSOCodeTranslator callback_translator;
#endif
      std::map<void*, RtEvent> local_callbacks_done;
    public:
      struct RegistrationKey {
        inline RegistrationKey(void) : tag(0) { }
        inline RegistrationKey(
            size_t t, const std::string& dso, const std::string& symbol)
          : tag(t), dso_name(dso), symbol_name(symbol)
        { }
        inline bool operator<(const RegistrationKey& rhs) const
        {
          if (tag < rhs.tag)
            return true;
          if (tag > rhs.tag)
            return false;
          if (dso_name < rhs.dso_name)
            return true;
          if (dso_name > rhs.dso_name)
            return false;
          return symbol_name < rhs.symbol_name;
        }
        size_t tag;
        std::string dso_name;
        std::string symbol_name;
      };
    protected:
      std::map<RegistrationKey, RtEvent> global_callbacks_done;
      std::map<RegistrationKey, RtEvent> global_local_done;
      std::map<RegistrationKey, std::set<RtUserEvent> >
          pending_remote_callbacks;
    protected:
      mutable LocalLock redop_lock;
      std::map<ReductionOpID, FillView*> redop_fill_views;
      mutable LocalLock serdez_lock;
    protected:
      mutable LocalLock projection_lock;
      std::map<ProjectionID, ProjectionFunction*> projection_functions;
    protected:
      mutable LocalLock sharding_lock;
      std::map<ShardingID, ShardingFunctor*> sharding_functors;
    protected:
      mutable LocalLock concurrent_lock;
      std::map<ConcurrentID, ConcurrentColoringFunctor*> concurrent_functors;
    protected:
      mutable LocalLock group_lock;
      rt::map<uint64_t, rt::deque<ProcessorGroupInfo> > processor_groups;
    protected:
      mutable LocalLock processor_mapping_lock;
      std::map<Processor, unsigned> processor_mapping;
    protected:
      std::atomic<DistributedID> unique_distributed_id;
    protected:
      mutable LocalLock distributed_collectable_lock;
      lng::map<DistributedID, DistributedCollectable*> dist_collectables;
      std::map<DistributedID, std::pair<DistributedCollectable*, RtUserEvent> >
          pending_collectables;
    protected:
      mutable LocalLock is_slice_lock;
      std::map<std::pair<Domain, TypeTag>, std::pair<IndexSpace, RtUserEvent> >
          index_slice_spaces;
    protected:
      // The runtime keeps track of remote contexts so they
      // can be re-used by multiple tasks that get sent remotely
      mutable LocalLock context_lock;
      unsigned total_contexts;
      std::vector<ContextID> available_contexts;
    protected:
      // Keep track of managers for control replication execution
      mutable LocalLock shard_lock;
      std::map<TaskID, ImplicitShardManager*> implicit_shard_managers;
    protected:
      // For generating random numbers
      mutable LocalLock random_lock;
      unsigned short random_state[3];
#ifdef LEGION_TRACE_ALLOCATION
    public:
      struct AllocationTracker {
      public:
        AllocationTracker(const char* n)
          : name(n), total_allocations(0), total_bytes(0), diff_allocations(0),
            diff_bytes(0)
        { }
      public:
        const char* const name;
      public:
        unsigned total_allocations;
        size_t total_bytes;
        int diff_allocations;
        off_t diff_bytes;
      };
      mutable LocalLock allocation_lock;  // leak this lock intentionally
      std::unordered_map<std::size_t, AllocationTracker> allocation_manager;
      std::atomic<unsigned long long> allocation_tracing_count;
#endif
    private:
      mutable LocalLock operation_lock;
      /**
       * Make a type that provides overloads for making each of the
       * different kinds of operations that we might want to use
       */
      template<typename... Factories>
      class OperationIndustry : public Factories... {
      public:
        using Factories::create...;
        using Factories::recycle...;
      };
      OperationIndustry<
          OperationFactory<IndividualTask, Predicated<IndividualTask> >,
          OperationFactory<PointTask, Memoizable<PointTask>, true>,
          OperationFactory<IndexTask, Predicated<IndexTask> >,
          OperationFactory<SliceTask, Memoizable<SliceTask>, true>,
          OperationFactory<MapOp>,
          OperationFactory<CopyOp, Predicated<CopyOp> >,
          OperationFactory<IndexCopyOp, Predicated<IndexCopyOp> >,
          OperationFactory<PointCopyOp, Memoizable<PointCopyOp> >,
          OperationFactory<FenceOp, Memoizable<FenceOp> >,
          OperationFactory<FrameOp>, OperationFactory<CreationOp>,
          OperationFactory<DeletionOp>, OperationFactory<MergeCloseOp>,
          OperationFactory<PostCloseOp>, OperationFactory<RefinementOp>,
          OperationFactory<ResetOp>,
          OperationFactory<
              DynamicCollectiveOp, Memoizable<DynamicCollectiveOp> >,
          OperationFactory<FuturePredOp>, OperationFactory<NotPredOp>,
          OperationFactory<AndPredOp>, OperationFactory<OrPredOp>,
          OperationFactory<AcquireOp, Predicated<AcquireOp> >,
          OperationFactory<ReleaseOp, Predicated<ReleaseOp> >,
          OperationFactory<TraceBeginOp>, OperationFactory<TraceRecurrentOp>,
          OperationFactory<TraceCompleteOp>, OperationFactory<MustEpochOp>,
          OperationFactory<PendingPartitionOp>,
          OperationFactory<DependentPartitionOp>,
          OperationFactory<PointDepPartOp, PointDepPartOp, true>,
          OperationFactory<FillOp, Predicated<FillOp> >,
          OperationFactory<IndexFillOp, Predicated<IndexFillOp> >,
          OperationFactory<PointFillOp, Memoizable<PointFillOp>, true>,
          OperationFactory<DiscardOp>, OperationFactory<AttachOp>,
          OperationFactory<IndexAttachOp>,
          OperationFactory<PointAttachOp, PointAttachOp, true>,
          OperationFactory<DetachOp>, OperationFactory<IndexDetachOp>,
          OperationFactory<PointDetachOp, PointDetachOp, true>,
          OperationFactory<TimingOp>, OperationFactory<TunableOp>,
          OperationFactory<AllReduceOp, Memoizable<AllReduceOp> >,
          OperationFactory<ReplIndividualTask, Predicated<ReplIndividualTask> >,
          OperationFactory<ReplIndexTask, Predicated<ReplIndexTask> >,
          OperationFactory<ReplMergeCloseOp>,
          OperationFactory<ReplRefinementOp>, OperationFactory<ReplResetOp>,
          OperationFactory<ReplFillOp, Predicated<ReplFillOp> >,
          OperationFactory<ReplIndexFillOp, Predicated<ReplIndexFillOp> >,
          OperationFactory<ReplCopyOp, Predicated<ReplCopyOp> >,
          OperationFactory<ReplIndexCopyOp, Predicated<ReplIndexCopyOp> >,
          OperationFactory<ReplDeletionOp>,
          OperationFactory<ReplPendingPartitionOp>,
          OperationFactory<ReplDependentPartitionOp>,
          OperationFactory<ReplMustEpochOp>, OperationFactory<ReplTimingOp>,
          OperationFactory<ReplTunableOp>,
          OperationFactory<ReplAllReduceOp, Memoizable<ReplAllReduceOp> >,
          OperationFactory<ReplFenceOp, Memoizable<ReplFenceOp> >,
          OperationFactory<ReplMapOp>, OperationFactory<ReplDiscardOp>,
          OperationFactory<ReplAttachOp>, OperationFactory<ReplIndexAttachOp>,
          OperationFactory<ReplDetachOp>, OperationFactory<ReplIndexDetachOp>,
          OperationFactory<ReplAcquireOp, Predicated<ReplAcquireOp> >,
          OperationFactory<ReplReleaseOp, Predicated<ReplReleaseOp> >,
          OperationFactory<ReplTraceBeginOp>,
          OperationFactory<ReplTraceRecurrentOp>,
          OperationFactory<ReplTraceCompleteOp> >
          operation_industry;
#ifdef LEGION_DEBUG
      std::set<Operation*> outstanding_operations;
#endif
    public:
      LayoutConstraintID register_layout(
          const LayoutConstraintRegistrar& registrar, LayoutConstraintID id,
          DistributedID did = 0,
          CollectiveMapping* collective_mapping = nullptr);
      LayoutConstraints* register_layout(
          FieldSpace handle, const LayoutConstraintSet& cons, bool internal);
      bool register_layout(LayoutConstraints* new_constraints);
      void release_layout(LayoutConstraintID layout_id);
      void unregister_layout(LayoutConstraintID layout_id);
      static LayoutConstraintID preregister_layout(
          const LayoutConstraintRegistrar& registrar,
          LayoutConstraintID layout_id);
      FieldSpace get_layout_constraint_field_space(LayoutConstraintID id);
      void get_layout_constraints(
          LayoutConstraintID layout_id,
          LayoutConstraintSet& layout_constraints);
      const char* get_layout_constraints_name(LayoutConstraintID layout_id);
      LayoutConstraints* find_layout_constraints(
          LayoutConstraintID layout_id, bool can_fail = false,
          RtEvent* wait_for = nullptr);
    public:
      // Static methods for start-up and callback phases
      static int start(
          int argc, char** argv, bool background, bool def_mapper, bool filter);
      static void register_builtin_reduction_operators(void);
      static const LegionConfiguration& initialize(
          int* argc, char*** argv, bool parse, bool filter);
      static unsigned initialize_outstanding_top_level_tasks(
          AddressSpaceID local_space, size_t total_spaces, unsigned radix);
      static void perform_slow_config_checks(const LegionConfiguration& config);
      static void configure_interoperability(void);
      static Processor configure_runtime(
          int argc, char** argv, const LegionConfiguration& config,
          RealmRuntime& realm, std::set<Processor>& local_procs,
          bool background, bool default_mapper);
      static int wait_for_shutdown(void);
      static void set_return_code(int return_code);
      Future launch_top_level_task(
          const TaskLauncher& launcher, TopLevelContext* context = nullptr);
      IndividualTask* create_implicit_top_level(
          TaskID top_task_id, MapperID top_mapper_id, Processor proxy,
          const char* task_name, CollectiveMapping* mapping = nullptr);
      ImplicitShardManager* find_implicit_shard_manager(
          TaskID top_task_id, MapperID top_mapper_id, Processor::Kind kind,
          unsigned shards_per_space);
      void unregister_implicit_shard_manager(TaskID top_task_id);
      Context begin_implicit_task(
          TaskID top_task_id, MapperID top_mapper_id, Processor::Kind proc_kind,
          const char* task_name, bool control_replicable,
          unsigned shard_per_address_space, int shard_id,
          const DomainPoint& point);
      void unbind_implicit_task_from_external_thread(Context ctx);
      void bind_implicit_task_to_external_thread(Context ctx);
      void finish_implicit_task(Context ctx, ApEvent effects);
      static void set_top_level_task_id(TaskID top_id);
      static void set_top_level_task_mapper_id(MapperID mapper_id);
      static void configure_MPI_interoperability(int rank);
      static void register_handshake(LegionHandshake& handshake);
      static const ReductionOp* get_reduction_op(
          ReductionOpID redop_id, bool has_lock = false);
      static const SerdezOp* get_serdez_op(
          CustomSerdezID serdez_id, bool has_lock = false);
      static const SerdezRedopFns* get_serdez_redop_fns(
          ReductionOpID redop_id, bool has_lock = false);
      static void add_registration_callback(
          RegistrationCallback callback, bool dedup, size_t dedup_tag);
      static void add_registration_callback(
          RegistrationWithArgsCallback callback, const UntypedBuffer& buffer,
          bool dedup, size_t dedup_tag);
      static void perform_dynamic_registration_callback(
          RegistrationCallback callback, bool global, bool deduplicate,
          size_t dedup_tag);
      static void perform_dynamic_registration_callback(
          RegistrationWithArgsCallback callback, const UntypedBuffer& buffer,
          bool global, bool deduplicate, size_t dedup_tag);
      static ReductionOpTable& get_reduction_table(bool safe);
      static SerdezOpTable& get_serdez_table(bool safe);
      static SerdezRedopTable& get_serdez_redop_table(bool safe);
      static void register_reduction_op(
          ReductionOpID redop_id, ReductionOp* redop, SerdezInitFunc init_func,
          SerdezFoldFunc fold_func, bool permit_duplicates,
          bool has_lock = false);
      static void register_serdez_op(
          CustomSerdezID serdez_id, SerdezOp* serdez_op, bool permit_duplicates,
          bool has_lock = false);
      static std::deque<PendingVariantRegistration*>& get_pending_variant_table(
          void);
      static std::map<LayoutConstraintID, LayoutConstraintRegistrar>&
          get_pending_constraint_table(void);
      static std::map<ProjectionID, ProjectionFunctor*>&
          get_pending_projection_table(void);
      static std::map<ShardingID, ShardingFunctor*>& get_pending_sharding_table(
          void);
      static std::map<ConcurrentID, ConcurrentColoringFunctor*>&
          get_pending_concurrent_table(void);
      static std::vector<LegionHandshake>& get_pending_handshake_table(void);
      static std::vector<PendingRegistrationCallback>&
          get_pending_registration_callbacks(void);
      static TaskID& get_current_static_task_id(void);
      static TaskID generate_static_task_id(void);
      static VariantID preregister_variant(
          const TaskVariantRegistrar& registrar, const void* user_data,
          size_t user_data_size, const CodeDescriptor& realm_desc,
          size_t return_type_size, bool has_return_type_size,
          const char* task_name, VariantID vid, bool check_id = true);
    public:
      static ReductionOpID& get_current_static_reduction_id(void);
      static ReductionOpID generate_static_reduction_id(void);
      static CustomSerdezID& get_current_static_serdez_id(void);
      static CustomSerdezID generate_static_serdez_id(void);
    public:
      static void report_fatal_message(
          int code, const char* file_name, const int line_number,
          const char* message);
      static void report_error_message(
          int code, const char* file_name, const int line_number,
          const char* message);
      static void report_warning_message(
          int code, const char* file_name, const int line_number,
          const char* message);
      static void raise_warning(
          Exception&& exception, const Realm::Backtrace& backtrace);
      [[noreturn]] static void raise_exception(
          Exception&& exception, const Realm::Backtrace& backtrace);
    public:
      // Static member variables
      static inline TaskID legion_main_id = 0;
      static inline MapperID legion_main_mapper_id = 0;
      static inline bool legion_main_set = false;
      static inline bool runtime_initialized = false;
      static inline bool runtime_cmdline_parsed = false;
      static inline bool runtime_started = false;
      static inline bool runtime_backgrounded = false;
      static inline std::atomic<Realm::Event::id_t> startup_event = {0};
      static inline Realm::Barrier::timestamp_t startup_timestamp = 0;
      static inline std::atomic<bool> background_wait = {0};
      static inline void (*meta_task_table[LG_LAST_TASK_ID])(
          const void*, size_t) = {0};
      // Shutdown error condition
      static inline int return_code = 0;
      // Static member variables for MPI interop
      static inline int mpi_rank = -1;
    public:
      static inline ApEvent merge_events(
          const TraceInfo* info, ApEvent e1, ApEvent e2);
      static inline ApEvent merge_events(
          const TraceInfo* info, ApEvent e1, ApEvent e2, ApEvent e3);
      static inline ApEvent merge_events(
          const TraceInfo* info, const std::set<ApEvent>& events);
      static inline ApEvent merge_events(
          const TraceInfo* info, const std::vector<ApEvent>& events);
    public:
      static inline RtEvent merge_events(RtEvent e1, RtEvent e2);
      static inline RtEvent merge_events(RtEvent e1, RtEvent e2, RtEvent e3);
      static inline RtEvent merge_events(const std::set<RtEvent>& events);
      static inline RtEvent merge_events(const std::vector<RtEvent>& events);
    public:
      static inline ApUserEvent create_ap_user_event(const TraceInfo* info);
      static inline void trigger_event(
          ApUserEvent to_trigger, ApEvent precondition, const TraceInfo& info,
          std::set<RtEvent>& applied_events);
      static inline void trigger_event_untraced(
          ApUserEvent to_trigger, ApEvent precondition = ApEvent::NO_AP_EVENT);
      static inline void poison_event(ApUserEvent to_poison);
    public:
      static inline RtUserEvent create_rt_user_event(void);
      static inline void trigger_event(
          RtUserEvent to_trigger, RtEvent precondition = RtEvent::NO_RT_EVENT);
      static inline void poison_event(RtUserEvent to_poison);
    public:
      static inline PredUserEvent create_pred_event(void);
      static inline void trigger_event(PredUserEvent to_trigger);
      static inline void poison_event(PredUserEvent to_poison);
      static inline PredEvent merge_events(
          const TraceInfo* info, PredEvent e1, PredEvent e2);
    public:
      static inline ApEvent ignorefaults(ApEvent e);
      static inline RtEvent protect_event(ApEvent to_protect);
      static inline RtEvent protect_merge_events(
          const std::set<ApEvent>& events);
    public:
      static inline ApBarrier get_previous_phase(const PhaseBarrier& bar);
      inline void phase_barrier_arrive(
          const PhaseBarrier& bar, unsigned cnt,
          ApEvent precondition = ApEvent::NO_AP_EVENT,
          const void* reduce_value = nullptr, size_t reduce_value_size = 0);
      static inline void advance_barrier(PhaseBarrier& bar);
      static inline void alter_arrival_count(PhaseBarrier& bar, int delta);
    public:
      inline ApBarrier create_ap_barrier(size_t arrivals);
      static inline ApBarrier get_previous_phase(const ApBarrier& bar);
      inline void phase_barrier_arrive(
          const ApBarrier& bar, unsigned cnt,
          ApEvent precondition = ApEvent::NO_AP_EVENT);
      static inline void advance_barrier(ApBarrier& bar);
      static inline bool get_barrier_result(
          ApBarrier bar, void* result, size_t result_size);
    public:
      inline RtBarrier create_rt_barrier(size_t arrivals);
      static inline RtBarrier get_previous_phase(const RtBarrier& bar);
#ifdef LEGION_DEBUG_COLLECTIVES
      inline void phase_barrier_arrive(
          const RtBarrier& bar, unsigned cnt,
          RtEvent precondition = RtEvent::NO_RT_EVENT,
          const void* reduce_value = nullptr, size_t reduce_value_size = 0);
#else
      inline void phase_barrier_arrive(
          const RtBarrier& bar, unsigned cnt,
          RtEvent precondition = RtEvent::NO_RT_EVENT);
#endif
      static inline void advance_barrier(RtBarrier& bar);
      static inline bool get_barrier_result(
          RtBarrier bar, void* result, size_t result_size);
      static inline void alter_arrival_count(RtBarrier& bar, int delta);
    public:
      static inline ApEvent acquire_ap_reservation(
          Reservation r, bool exclusive,
          ApEvent precondition = ApEvent::NO_AP_EVENT);
      static inline RtEvent acquire_rt_reservation(
          Reservation r, bool exclusive,
          RtEvent precondition = RtEvent::NO_RT_EVENT);
      static inline void release_reservation(
          Reservation r, LgEvent precondition = LgEvent::NO_LG_EVENT);
      static inline void rename_event(LgEvent& to_rename);
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/kernel/runtime.inl"

#endif  // __LEGION_CORE_RUNTIME_H__
