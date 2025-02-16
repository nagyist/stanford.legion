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

#ifndef __LEGION_MESSAGE_MANAGER_H__
#define __LEGION_MESSAGE_MANAGER_H__

#include "legion/kernel/metatask.h"

namespace Legion {
  namespace Internal {

    enum VirtualChannelKind {
      // The default and work virtual channels are unordered
      DEFAULT_VIRTUAL_CHANNEL = 0, // latency priority
      THROUGHPUT_VIRTUAL_CHANNEL = 1, // throughput priority
      LAST_UNORDERED_VIRTUAL_CHANNEL = THROUGHPUT_VIRTUAL_CHANNEL,
      // All the rest of these are ordered (latency-priority) channels
      MAPPER_VIRTUAL_CHANNEL = 1, 
      TASK_VIRTUAL_CHANNEL = 2,
      INDEX_SPACE_VIRTUAL_CHANNEL = 3,
      FIELD_SPACE_VIRTUAL_CHANNEL = 4,
      REFERENCE_VIRTUAL_CHANNEL = 6,
      UPDATE_VIRTUAL_CHANNEL = 7, // deferred-priority
      SUBSET_VIRTUAL_CHANNEL = 8,
      COLLECTIVE_VIRTUAL_CHANNEL = 9,
      LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL = 10,
      EXPRESSION_VIRTUAL_CHANNEL = 11,
      MIGRATION_VIRTUAL_CHANNEL = 12,
      TRACING_VIRTUAL_CHANNEL = 13,
      RENDEZVOUS_VIRTUAL_CHANNEL = 14,
      PROFILING_VIRTUAL_CHANNEL = 15,
      MAX_NUM_VIRTUAL_CHANNELS = 16, // this one must be last
    };

    enum MessageKind {
      SEND_STARTUP_BARRIER,
      TASK_MESSAGE,
      STEAL_MESSAGE,
      ADVERTISEMENT_MESSAGE,
      SEND_REGISTRATION_CALLBACK,
      SEND_REMOTE_TASK_REPLAY,
      SEND_REMOTE_TASK_PROFILING_RESPONSE,
      SEND_SHARED_OWNERSHIP,
      SEND_INDEX_SPACE_REQUEST,
      SEND_INDEX_SPACE_RESPONSE,
      SEND_INDEX_SPACE_RETURN,
      SEND_INDEX_SPACE_SET,
      SEND_INDEX_SPACE_CHILD_REQUEST,
      SEND_INDEX_SPACE_CHILD_RESPONSE,
      SEND_INDEX_SPACE_COLORS_REQUEST,
      SEND_INDEX_SPACE_COLORS_RESPONSE,
      SEND_INDEX_SPACE_GENERATE_COLOR_REQUEST,
      SEND_INDEX_SPACE_GENERATE_COLOR_RESPONSE,
      SEND_INDEX_SPACE_RELEASE_COLOR,
      SEND_INDEX_PARTITION_NOTIFICATION,
      SEND_INDEX_PARTITION_REQUEST,
      SEND_INDEX_PARTITION_RESPONSE,
      SEND_INDEX_PARTITION_RETURN,
      SEND_INDEX_PARTITION_CHILD_REQUEST,
      SEND_INDEX_PARTITION_CHILD_RESPONSE,
      SEND_INDEX_PARTITION_CHILD_REPLICATION,
      SEND_INDEX_PARTITION_DISJOINT_UPDATE,
      SEND_INDEX_PARTITION_SHARD_RECTS_REQUEST,
      SEND_INDEX_PARTITION_SHARD_RECTS_RESPONSE,
      SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_REQUEST,
      SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_RESPONSE,
      SEND_FIELD_SPACE_NODE,
      SEND_FIELD_SPACE_REQUEST,
      SEND_FIELD_SPACE_RETURN,
      SEND_FIELD_SPACE_ALLOCATOR_REQUEST,
      SEND_FIELD_SPACE_ALLOCATOR_RESPONSE,
      SEND_FIELD_SPACE_ALLOCATOR_INVALIDATION,
      SEND_FIELD_SPACE_ALLOCATOR_FLUSH,
      SEND_FIELD_SPACE_ALLOCATOR_FREE,
      SEND_FIELD_SPACE_INFOS_REQUEST,
      SEND_FIELD_SPACE_INFOS_RESPONSE,
      SEND_FIELD_ALLOC_REQUEST,
      SEND_FIELD_SIZE_UPDATE,
      SEND_FIELD_FREE,
      SEND_FIELD_FREE_INDEXES,
      SEND_FIELD_SPACE_LAYOUT_INVALIDATION,
      SEND_LOCAL_FIELD_ALLOC_REQUEST,
      SEND_LOCAL_FIELD_ALLOC_RESPONSE,
      SEND_LOCAL_FIELD_FREE,
      SEND_LOCAL_FIELD_UPDATE,
      SEND_TOP_LEVEL_REGION_REQUEST,
      SEND_TOP_LEVEL_REGION_RETURN,
      INDEX_SPACE_DESTRUCTION_MESSAGE,
      INDEX_PARTITION_DESTRUCTION_MESSAGE,
      FIELD_SPACE_DESTRUCTION_MESSAGE,
      LOGICAL_REGION_DESTRUCTION_MESSAGE,
      INDIVIDUAL_REMOTE_FUTURE_SIZE,
      INDIVIDUAL_REMOTE_OUTPUT_REGISTRATION,
      INDIVIDUAL_REMOTE_MAPPED,
      INDIVIDUAL_REMOTE_COMPLETE,
      INDIVIDUAL_REMOTE_COMMIT,
      INDIVIDUAL_CONCURRENT_REQUEST,
      INDIVIDUAL_CONCURRENT_RESPONSE,
      SLICE_REMOTE_MAPPED,
      SLICE_REMOTE_COMPLETE,
      SLICE_REMOTE_COMMIT,
      SLICE_RENDEZVOUS_CONCURRENT_MAPPED,
      SLICE_COLLECTIVE_ALLREDUCE_REQUEST,
      SLICE_COLLECTIVE_ALLREDUCE_RESPONSE,
      SLICE_CONCURRENT_ALLREDUCE_REQUEST,
      SLICE_CONCURRENT_ALLREDUCE_RESPONSE,
      SLICE_FIND_INTRA_DEP,
      SLICE_REMOTE_COLLECTIVE_RENDEZVOUS,
      SLICE_REMOTE_VERSIONING_COLLECTIVE_RENDEZVOUS,
      SLICE_REMOTE_OUTPUT_EXTENTS,
      SLICE_REMOTE_OUTPUT_REGISTRATION,
      DISTRIBUTED_REMOTE_REGISTRATION,
      DISTRIBUTED_DOWNGRADE_REQUEST,
      DISTRIBUTED_DOWNGRADE_RESPONSE,
      DISTRIBUTED_DOWNGRADE_SUCCESS,
      DISTRIBUTED_DOWNGRADE_UPDATE,
      DISTRIBUTED_DOWNGRADE_RESTART,
      DISTRIBUTED_GLOBAL_ACQUIRE_REQUEST,
      DISTRIBUTED_GLOBAL_ACQUIRE_RESPONSE,
      DISTRIBUTED_VALID_ACQUIRE_REQUEST,
      DISTRIBUTED_VALID_ACQUIRE_RESPONSE,
      SEND_ATOMIC_RESERVATION_REQUEST,
      SEND_ATOMIC_RESERVATION_RESPONSE,
      SEND_PADDED_RESERVATION_REQUEST,
      SEND_PADDED_RESERVATION_RESPONSE,
      SEND_CREATED_REGION_CONTEXTS,
      SEND_MATERIALIZED_VIEW,
      SEND_FILL_VIEW,
      SEND_FILL_VIEW_VALUE,
      SEND_PHI_VIEW,
      SEND_REDUCTION_VIEW,
      SEND_REPLICATED_VIEW,
      SEND_ALLREDUCE_VIEW,
      SEND_INSTANCE_MANAGER,
      SEND_MANAGER_UPDATE,
      SEND_COLLECTIVE_DISTRIBUTE_FILL,
      SEND_COLLECTIVE_DISTRIBUTE_POINT,
      SEND_COLLECTIVE_DISTRIBUTE_POINTWISE,
      SEND_COLLECTIVE_DISTRIBUTE_REDUCTION,
      SEND_COLLECTIVE_DISTRIBUTE_BROADCAST,
      SEND_COLLECTIVE_DISTRIBUTE_REDUCECAST,
      SEND_COLLECTIVE_DISTRIBUTE_HOURGLASS,
      SEND_COLLECTIVE_DISTRIBUTE_ALLREDUCE,
      SEND_COLLECTIVE_HAMMER_REDUCTION,
      SEND_COLLECTIVE_FUSE_GATHER,
      SEND_COLLECTIVE_USER_REQUEST,
      SEND_COLLECTIVE_USER_RESPONSE,
      SEND_COLLECTIVE_REGISTER_USER,
      SEND_COLLECTIVE_REMOTE_INSTANCES_REQUEST,
      SEND_COLLECTIVE_REMOTE_INSTANCES_RESPONSE,
      SEND_COLLECTIVE_NEAREST_INSTANCES_REQUEST,
      SEND_COLLECTIVE_NEAREST_INSTANCES_RESPONSE,
      SEND_COLLECTIVE_REMOTE_REGISTRATION,
      SEND_COLLECTIVE_FINALIZE_MAPPING,
      SEND_COLLECTIVE_VIEW_CREATION,
      SEND_COLLECTIVE_VIEW_DELETION,
      SEND_COLLECTIVE_VIEW_RELEASE,
      SEND_COLLECTIVE_VIEW_NOTIFICATION,
      SEND_COLLECTIVE_VIEW_MAKE_VALID,
      SEND_COLLECTIVE_VIEW_MAKE_INVALID,
      SEND_COLLECTIVE_VIEW_INVALIDATE_REQUEST,
      SEND_COLLECTIVE_VIEW_INVALIDATE_RESPONSE,
      SEND_COLLECTIVE_VIEW_ADD_REMOTE_REFERENCE,
      SEND_COLLECTIVE_VIEW_REMOVE_REMOTE_REFERENCE,
      SEND_CREATE_TOP_VIEW_REQUEST,
      SEND_CREATE_TOP_VIEW_RESPONSE,
      SEND_VIEW_REQUEST,
      SEND_VIEW_REGISTER_USER,
      SEND_VIEW_FIND_COPY_PRE_REQUEST,
      SEND_VIEW_ADD_COPY_USER,
      SEND_VIEW_FIND_LAST_USERS_REQUEST,
      SEND_VIEW_FIND_LAST_USERS_RESPONSE,
      SEND_MANAGER_REQUEST,
      SEND_FUTURE_RESULT,
      SEND_FUTURE_RESULT_SIZE,
      SEND_FUTURE_SUBSCRIPTION,
      SEND_FUTURE_CREATE_INSTANCE_REQUEST,
      SEND_FUTURE_CREATE_INSTANCE_RESPONSE,
      SEND_FUTURE_MAP_REQUEST,
      SEND_FUTURE_MAP_RESPONSE,
      SEND_FUTURE_MAP_POINTWISE,
      SEND_REPL_COMPUTE_EQUIVALENCE_SETS,
      SEND_REPL_OUTPUT_EQUIVALENCE_SET,
      SEND_REPL_REFINE_EQUIVALENCE_SETS,
      SEND_REPL_EQUIVALENCE_SET_NOTIFICATION,
      SEND_REPL_BROADCAST_UPDATE,
      SEND_REPL_CREATED_REGIONS,
      SEND_REPL_TRACE_EVENT_REQUEST,
      SEND_REPL_TRACE_EVENT_RESPONSE,
      SEND_REPL_TRACE_EVENT_TRIGGER,
      SEND_REPL_TRACE_FRONTIER_REQUEST,
      SEND_REPL_TRACE_FRONTIER_RESPONSE,
      SEND_REPL_TRACE_UPDATE,
      SEND_REPL_FIND_TRACE_SETS,
      SEND_REPL_IMPLICIT_RENDEZVOUS,
      SEND_REPL_FIND_COLLECTIVE_VIEW,
      SEND_REPL_POINTWISE_DEPENDENCE,
      SEND_MAPPER_MESSAGE,
      SEND_MAPPER_BROADCAST,
      SEND_TASK_IMPL_SEMANTIC_REQ,
      SEND_INDEX_SPACE_SEMANTIC_REQ,
      SEND_INDEX_PARTITION_SEMANTIC_REQ,
      SEND_FIELD_SPACE_SEMANTIC_REQ,
      SEND_FIELD_SEMANTIC_REQ,
      SEND_LOGICAL_REGION_SEMANTIC_REQ,
      SEND_LOGICAL_PARTITION_SEMANTIC_REQ,
      SEND_TASK_IMPL_SEMANTIC_INFO,
      SEND_INDEX_SPACE_SEMANTIC_INFO,
      SEND_INDEX_PARTITION_SEMANTIC_INFO,
      SEND_FIELD_SPACE_SEMANTIC_INFO,
      SEND_FIELD_SEMANTIC_INFO,
      SEND_LOGICAL_REGION_SEMANTIC_INFO,
      SEND_LOGICAL_PARTITION_SEMANTIC_INFO,
      SEND_REMOTE_CONTEXT_REQUEST,
      SEND_REMOTE_CONTEXT_RESPONSE,
      SEND_REMOTE_CONTEXT_PHYSICAL_REQUEST,
      SEND_REMOTE_CONTEXT_PHYSICAL_RESPONSE,
      SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_REQUEST,
      SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_RESPONSE,
      SEND_REMOTE_CONTEXT_REFINE_EQUIVALENCE_SETS,
      SEND_REMOTE_CONTEXT_POINTWISE_DEPENDENCE,
      SEND_REMOTE_CONTEXT_FIND_TRACE_LOCAL_SETS_REQUEST,
      SEND_REMOTE_CONTEXT_FIND_TRACE_LOCAL_SETS_RESPONSE,
      SEND_COMPUTE_EQUIVALENCE_SETS_REQUEST,
      SEND_COMPUTE_EQUIVALENCE_SETS_RESPONSE,
      SEND_COMPUTE_EQUIVALENCE_SETS_PENDING,
      SEND_OUTPUT_EQUIVALENCE_SET_REQUEST,
      SEND_OUTPUT_EQUIVALENCE_SET_RESPONSE,
      SEND_CANCEL_EQUIVALENCE_SETS_SUBSCRIPTION,
      SEND_INVALIDATE_EQUIVALENCE_SETS_SUBSCRIPTION,
      SEND_EQUIVALENCE_SET_CREATION,
      SEND_EQUIVALENCE_SET_REUSE,
      SEND_EQUIVALENCE_SET_REQUEST,
      SEND_EQUIVALENCE_SET_RESPONSE,
      SEND_EQUIVALENCE_SET_REPLICATION_REQUEST,
      SEND_EQUIVALENCE_SET_REPLICATION_RESPONSE,
      SEND_EQUIVALENCE_SET_MIGRATION,
      SEND_EQUIVALENCE_SET_OWNER_UPDATE,
      SEND_EQUIVALENCE_SET_CLONE_REQUEST,
      SEND_EQUIVALENCE_SET_CLONE_RESPONSE,
      SEND_EQUIVALENCE_SET_CAPTURE_REQUEST,
      SEND_EQUIVALENCE_SET_CAPTURE_RESPONSE,
      SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INSTANCES,
      SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INVALID,
      SEND_EQUIVALENCE_SET_REMOTE_REQUEST_ANTIVALID,
      SEND_EQUIVALENCE_SET_REMOTE_UPDATES,
      SEND_EQUIVALENCE_SET_REMOTE_ACQUIRES,
      SEND_EQUIVALENCE_SET_REMOTE_RELEASES,
      SEND_EQUIVALENCE_SET_REMOTE_COPIES_ACROSS,
      SEND_EQUIVALENCE_SET_REMOTE_OVERWRITES,
      SEND_EQUIVALENCE_SET_REMOTE_FILTERS,
      SEND_EQUIVALENCE_SET_REMOTE_INSTANCES,
      SEND_EQUIVALENCE_SET_FILTER_INVALIDATIONS,
      SEND_INSTANCE_REQUEST,
      SEND_INSTANCE_RESPONSE,
      SEND_EXTERNAL_CREATE_REQUEST,
      SEND_EXTERNAL_CREATE_RESPONSE,
      SEND_EXTERNAL_ATTACH,
      SEND_EXTERNAL_DETACH,
      SEND_GC_PRIORITY_UPDATE,
      SEND_GC_REQUEST,
      SEND_GC_RESPONSE,
      SEND_GC_ACQUIRE,
      SEND_GC_FAILED,
      SEND_GC_MISMATCH,
      SEND_GC_NOTIFY,
      SEND_GC_DEBUG_REQUEST,
      SEND_GC_DEBUG_RESPONSE,
      SEND_GC_RECORD_EVENT,
      SEND_ACQUIRE_REQUEST,
      SEND_ACQUIRE_RESPONSE,
      SEND_VARIANT_BROADCAST,
      SEND_CONSTRAINT_REQUEST,
      SEND_CONSTRAINT_RESPONSE,
      SEND_CONSTRAINT_RELEASE,
      SEND_TOP_LEVEL_TASK_COMPLETE,
      SEND_MPI_RANK_EXCHANGE,
      SEND_REPLICATE_DISTRIBUTION,
      SEND_REPLICATE_COLLECTIVE_VERSIONING,
      SEND_REPLICATE_COLLECTIVE_MAPPING,
      SEND_REPLICATE_VIRTUAL_RENDEZVOUS,
      SEND_REPLICATE_STARTUP_COMPLETE,
      SEND_REPLICATE_POST_MAPPED,
      SEND_REPLICATE_TRIGGER_COMPLETE,
      SEND_REPLICATE_TRIGGER_COMMIT,
      SEND_CONTROL_REPLICATE_RENDEZVOUS_MESSAGE,
      SEND_LIBRARY_MAPPER_REQUEST,
      SEND_LIBRARY_MAPPER_RESPONSE,
      SEND_LIBRARY_TRACE_REQUEST,
      SEND_LIBRARY_TRACE_RESPONSE,
      SEND_LIBRARY_PROJECTION_REQUEST,
      SEND_LIBRARY_PROJECTION_RESPONSE,
      SEND_LIBRARY_SHARDING_REQUEST,
      SEND_LIBRARY_SHARDING_RESPONSE,
      SEND_LIBRARY_CONCURRENT_REQUEST,
      SEND_LIBRARY_CONCURRENT_RESPONSE,
      SEND_LIBRARY_TASK_REQUEST,
      SEND_LIBRARY_TASK_RESPONSE,
      SEND_LIBRARY_REDOP_REQUEST,
      SEND_LIBRARY_REDOP_RESPONSE,
      SEND_LIBRARY_SERDEZ_REQUEST,
      SEND_LIBRARY_SERDEZ_RESPONSE,
      SEND_REMOTE_OP_REPORT_UNINIT,
      SEND_REMOTE_OP_PROFILING_COUNT_UPDATE,
      SEND_REMOTE_OP_COMPLETION_EFFECT,
      SEND_REMOTE_TRACE_UPDATE,
      SEND_REMOTE_TRACE_RESPONSE,
      SEND_FREE_EXTERNAL_ALLOCATION,
      SEND_NOTIFY_COLLECTED_INSTANCES,
      SEND_CREATE_MEMORY_POOL_REQUEST,
      SEND_CREATE_MEMORY_POOL_RESPONSE,
      SEND_CREATE_FUTURE_INSTANCE_REQUEST,
      SEND_CREATE_FUTURE_INSTANCE_RESPONSE,
      SEND_FREE_FUTURE_INSTANCE,
      SEND_REMOTE_DISTRIBUTED_ID_REQUEST,
      SEND_REMOTE_DISTRIBUTED_ID_RESPONSE,
      SEND_CONTROL_REPLICATION_FUTURE_ALLREDUCE,
      SEND_CONTROL_REPLICATION_FUTURE_BROADCAST,
      SEND_CONTROL_REPLICATION_FUTURE_REDUCTION,
      SEND_CONTROL_REPLICATION_VALUE_ALLREDUCE,
      SEND_CONTROL_REPLICATION_VALUE_BROADCAST,
      SEND_CONTROL_REPLICATION_VALUE_EXCHANGE,
      SEND_CONTROL_REPLICATION_BUFFER_BROADCAST,
      SEND_CONTROL_REPLICATION_SHARD_SYNC_TREE,
      SEND_CONTROL_REPLICATION_SHARD_EVENT_TREE,
      SEND_CONTROL_REPLICATION_SINGLE_TASK_TREE,
      SEND_CONTROL_REPLICATION_CROSS_PRODUCT_PARTITION,
      SEND_CONTROL_REPLICATION_SHARDING_GATHER_COLLECTIVE,
      SEND_CONTROL_REPLICATION_INDIRECT_COPY_EXCHANGE,
      SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_EXCHANGE,
      SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_GATHER,
      SEND_CONTROL_REPLICATION_DEPPART_RESULT_SCATTER,
      SEND_CONTROL_REPLICATION_BUFFER_EXCHANGE,
      SEND_CONTROL_REPLICATION_FUTURE_NAME_EXCHANGE,
      SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_BROADCAST,
      SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_EXCHANGE,
      SEND_CONTROL_REPLICATION_MUST_EPOCH_DEPENDENCE_EXCHANGE,
      SEND_CONTROL_REPLICATION_MUST_EPOCH_COMPLETION_EXCHANGE,
      SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_MAPPING,
      SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_SOURCES,
      SEND_CONTROL_REPLICATION_TEMPLATE_INDEX_EXCHANGE,
      SEND_CONTROL_REPLICATION_UNORDERED_EXCHANGE,
      SEND_CONTROL_REPLICATION_CONSENSUS_MATCH,
      SEND_CONTROL_REPLICATION_VERIFY_CONTROL_REPLICATION_EXCHANGE,
      SEND_CONTROL_REPLICATION_OUTPUT_SIZE_EXCHANGE,
      SEND_CONTROL_REPLICATION_INDEX_ATTACH_LAUNCH_SPACE,
      SEND_CONTROL_REPLICATION_INDEX_ATTACH_UPPER_BOUND,
      SEND_CONTROL_REPLICATION_INDEX_ATTACH_EXCHANGE,
      SEND_CONTROL_REPLICATION_SHARD_PARTICIPANTS_EXCHANGE,
      SEND_CONTROL_REPLICATION_IMPLICIT_SHARDING_FUNCTOR,
      SEND_CONTROL_REPLICATION_CREATE_FILL_VIEW,
      SEND_CONTROL_REPLICATION_VERSIONING_RENDEZVOUS,
      SEND_CONTROL_REPLICATION_VIEW_RENDEZVOUS,
      SEND_CONTROL_REPLICATION_CONCURRENT_MAPPING_RENDEZVOUS,
      SEND_CONTROL_REPLICATION_CONCURRENT_ALLREDUCE,
      SEND_CONTROL_REPLICATION_PROJECTION_TREE_EXCHANGE,
      SEND_CONTROL_REPLICATION_TIMEOUT_MATCH_EXCHANGE,
      SEND_CONTROL_REPLICATION_MASK_EXCHANGE,
      SEND_CONTROL_REPLICATION_PREDICATE_EXCHANGE,
      SEND_CONTROL_REPLICATION_CROSS_PRODUCT_EXCHANGE,
      SEND_CONTROL_REPLICATION_TRACING_SET_DEDUPLICATION,
      SEND_CONTROL_REPLICATION_POINTWISE_ALLREDUCE,
      SEND_CONTROL_REPLICATION_INTERFERING_POINT_EXCHANGE,
      SEND_CONTROL_REPLICATION_SLOW_BARRIER,
      SEND_PROFILER_EVENT_TRIGGER,
      SEND_PROFILER_EVENT_POISON,
      SEND_SHUTDOWN_NOTIFICATION,
      SEND_SHUTDOWN_RESPONSE,
      LAST_SEND_KIND, // This one must be last
    };

#define LG_MESSAGE_DESCRIPTIONS(name)                                 \
      const char *name[LAST_SEND_KIND] = {                            \
        "Send Startup Barrier",                                       \
        "Task Message",                                               \
        "Steal Message",                                              \
        "Advertisement Message",                                      \
        "Send Registration Callback",                                 \
        "Send Remote Task Replay",                                    \
        "Send Remote Task Profiling Response",                        \
        "Send Shared Ownership",                                      \
        "Send Index Space Request",                                   \
        "Send Index Space Response",                                  \
        "Send Index Space Return",                                    \
        "Send Index Space Set",                                       \
        "Send Index Space Child Request",                             \
        "Send Index Space Child Response",                            \
        "Send Index Space Colors Request",                            \
        "Send Index Space Colors Response",                           \
        "Send Index Space Generate Color Request",                    \
        "Send Index Space Generate Color Response",                   \
        "Send Index Space Release Color",                             \
        "Send Index Partition Notification",                          \
        "Send Index Partition Request",                               \
        "Send Index Partition Response",                              \
        "Send Index Partition Return",                                \
        "Send Index Partition Child Request",                         \
        "Send Index Partition Child Response",                        \
        "Send Index Partition Child Replication",                     \
        "Send Index Partition Disjoint Update",                       \
        "Send Index Partition Shard Rects Request",                   \
        "Send Index Partition Shard Rects Response",                  \
        "Send Index Partition Remote Interference Request",           \
        "Send Index Partition Remote Interference Response",          \
        "Send Field Space Node",                                      \
        "Send Field Space Request",                                   \
        "Send Field Space Return",                                    \
        "Send Field Space Allocator Request",                         \
        "Send Field Space Allocator Response",                        \
        "Send Field Space Allocator Invalidation",                    \
        "Send Field Space Allocator Flush",                           \
        "Send Field Space Allocator Free",                            \
        "Send Field Space Infos Request",                             \
        "Send Field Space Infos Response",                            \
        "Send Field Alloc Request",                                   \
        "Send Field Size Update",                                     \
        "Send Field Free",                                            \
        "Send Field Free Indexes",                                    \
        "Send Field Space Layout Invalidation",                       \
        "Send Local Field Alloc Request",                             \
        "Send Local Field Alloc Response",                            \
        "Send Local Field Free",                                      \
        "Send Local Field Update",                                    \
        "Send Top Level Region Request",                              \
        "Send Top Level Region Return",                               \
        "Index Space Destruction",                                    \
        "Index Partition Destruction",                                \
        "Field Space Destruction",                                    \
        "Logical Region Destruction",                                 \
        "Individual Remote Future Size",                              \
        "Individual Remote Output Region Registration",               \
        "Individual Remote Mapped",                                   \
        "Individual Remote Complete",                                 \
        "Individual Remote Commit",                                   \
        "Individual Concurrent Request",                              \
        "Individual Concurrent Response",                             \
        "Slice Remote Mapped",                                        \
        "Slice Remote Complete",                                      \
        "Slice Remote Commit",                                        \
        "Slice Rendezvous Concurrent Mapped",                         \
        "Slice Collective Unbounded Pools Allreduce Request",         \
        "Slice Collective Unbounded Pools Allreduce Response",        \
        "Slice Concurrent Allreduce Request",                         \
        "Slice Concurrent Allreduce Response",                        \
        "Slice Find Intra-Space Dependence",                          \
        "Slice Remote Collective Rendezvous",                         \
        "Slice Remote Collective Versioning Rendezvous",              \
        "Slice Remote Output Region Extents",                         \
        "Slice Remote Output Region Registration",                    \
        "Distributed Remote Registration",                            \
        "Distributed Downgrade Request",                              \
        "Distributed Downgrade Response",                             \
        "Distributed Downgrade Success",                              \
        "Distributed Downgrade Update",                               \
        "Distributed Downgrade Restart",                              \
        "Distributed Global Acquire Request",                         \
        "Distributed Global Acquire Response",                        \
        "Distributed Valid Acquire Request",                          \
        "Distributed Valid Acquire Response",                         \
        "Send Atomic Reservation Request",                            \
        "Send Atomic Reservation Response",                           \
        "Send Padded Reservation Request",                            \
        "Send Padded Reservation Response",                           \
        "Send Created Region Contexts",                               \
        "Send Materialized View",                                     \
        "Send Fill View",                                             \
        "Send Fill View Value",                                       \
        "Send Phi View",                                              \
        "Send Reduction View",                                        \
        "Send Replicated View",                                       \
        "Send Allreduce View",                                        \
        "Send Instance Manager",                                      \
        "Send Manager Update",                                        \
        "Send Collective Distribute Fill",                            \
        "Send Collective Distribute Point",                           \
        "Send Collective Distribute Pointwise",                       \
        "Send Collective Distribute Reduction",                       \
        "Send Collective Distribute Broadcast",                       \
        "Send Collective Distribute Reducecast",                      \
        "Send Collective Distribute Hourglass",                       \
        "Send Collective Distribute Allreduce",                       \
        "Send Collective Hammer Reduction",                           \
        "Send Collective Fuse Gather",                                \
        "Send Collective User Request",                               \
        "Send Collective User Response",                              \
        "Send Collective Individual Register User",                   \
        "Send Collective Remote Instances Request",                   \
        "Send Collective Remote Instances Response",                  \
        "Send Collective Nearest Instances Request",                  \
        "Send Collective Nearest Instances Response",                 \
        "Send Collective Remote Registration",                        \
        "Send Collective Finalize Mapping",                           \
        "Send Collective View Creation",                              \
        "Send Collective View Deletion",                              \
        "Send Collective View Release",                               \
        "Send Collective View Deletion Notification",                 \
        "Send Collective View Make Valid",                            \
        "Send Collective View Make Invalid",                          \
        "Send Collective View Invalidate Request",                    \
        "Send Collective View Invalidate Response",                   \
        "Send Collective View Add Remote Reference",                  \
        "Send Collective View Remove Remote Reference",               \
        "Send Create Top View Request",                               \
        "Send Create Top View Response",                              \
        "Send View Request",                                          \
        "Send View Register User",                                    \
        "Send View Find Copy Preconditions Request",                  \
        "Send View Add Copy User",                                    \
        "Send View Find Last Users Request",                          \
        "Send View Find Last Users Response",                         \
        "Send Manager Request",                                       \
        "Send Future Result",                                         \
        "Send Future Result Size",                                    \
        "Send Future Subscription",                                   \
        "Send Future Create Instance Request",                        \
        "Send Future Create Instance Response",                       \
        "Send Future Map Future Request",                             \
        "Send Future Map Future Response",                            \
        "Send Future Map Find Pointwise Dependence",                  \
        "Send Replicate Compute Equivalence Sets",                    \
        "Send Replicate Register Output Equivalence Set",             \
        "Send Replicate Refine Equivalence Sets",                     \
        "Send Replicate Equivalence Set Notification",                \
        "Send Replicate Broadcast Update",                            \
        "Send Replicate Created Regions Return",                      \
        "Send Replicate Trace Event Request",                         \
        "Send Replicate Trace Event Response",                        \
        "Send Replicate Trace Event Trigger",                         \
        "Send Replicate Trace Frontier Request",                      \
        "Send Replicate Trace Frontier Response",                     \
        "Send Replicate Trace Update",                                \
        "Send Replicate Find Trace Local Sets",                       \
        "Send Replicate Implicit Rendezvous",                         \
        "Send Replicate Find or Create Collective View",              \
        "Send Replicate Find Pointwise Dependence",                   \
        "Send Mapper Message",                                        \
        "Send Mapper Broadcast",                                      \
        "Send Task Impl Semantic Req",                                \
        "Send Index Space Semantic Req",                              \
        "Send Index Partition Semantic Req",                          \
        "Send Field Space Semantic Req",                              \
        "Send Field Semantic Req",                                    \
        "Send Logical Region Semantic Req",                           \
        "Send Logical Partition Semantic Req",                        \
        "Send Task Impl Semantic Info",                               \
        "Send Index Space Semantic Info",                             \
        "Send Index Partition Semantic Info",                         \
        "Send Field Space Semantic Info",                             \
        "Send Field Semantic Info",                                   \
        "Send Logical Region Semantic Info",                          \
        "Send Logical Partition Semantic Info",                       \
        "Send Remote Context Request",                                \
        "Send Remote Context Response",                               \
        "Send Remote Context Physical Request",                       \
        "Send Remote Context Physical Response",                      \
        "Send Remote Context Find Collective View Request",           \
        "Send Remote Context Find Collective View Response",          \
        "Send Remote Context Refine Equivalence Sets",                \
        "Send Remote Context Pointwise Dependence",                   \
        "Send Remote Context Find Trace Local Sets Request",          \
        "Send Remote Context Find Trace Local Sets Response",         \
        "Send Compute Equivalence Sets Request",                      \
        "Send Compute Equivalence Sets Response",                     \
        "Send Compute Equivalence Sets Pending",                      \
        "Send Register Output Equivalence Set Request",               \
        "Send Register Output Equivalence Set Response",              \
        "Send Cancel Equivalence Sets Subscription",                  \
        "Send Invalidate Equivalence Sets Subscription",              \
        "Send Equivalence Set Creation",                              \
        "Send Equivalence Set Reuse",                                 \
        "Send Equivalence Set Request",                               \
        "Send Equivalence Set Response",                              \
        "Send Equivalence Set Replication Request",                   \
        "Send Equivalence Set Replication Response",                  \
        "Send Equivalence Set Migration",                             \
        "Send Equivalence Set Owner Update",                          \
        "Send Equivalence Set Clone Request",                         \
        "Send Equivalence Set Clone Response",                        \
        "Send Equivalence Set Tracing Capture Request",               \
        "Send Equivalence Set Tracing Capture Response",              \
        "Send Equivalence Set Remote Request Instances",              \
        "Send Equivalence Set Remote Request Invalid",                \
        "Send Equivalence Set Remote Request Antivalid",              \
        "Send Equivalence Set Remote Updates",                        \
        "Send Equivalence Set Remote Acquires",                       \
        "Send Equivalence Set Remote Releases",                       \
        "Send Equivalence Set Remote Copies Across",                  \
        "Send Equivalence Set Remote Overwrites",                     \
        "Send Equivalence Set Remote Filters",                        \
        "Send Equivalence Set Remote Instances",                      \
        "Send Equivalence Set Filter Invalidations",                  \
        "Send Instance Request",                                      \
        "Send Instance Response",                                     \
        "Send External Create Request",                               \
        "Send External Create Response",                              \
        "Send External Attach",                                       \
        "Send External Detach",                                       \
        "Send GC Priority Update",                                    \
        "Send GC Request",                                            \
        "Send GC Response",                                           \
        "Send GC Acquire Request",                                    \
        "Send GC Acquire Failed",                                     \
        "Send GC Packed Reference Mismatch",                          \
        "Send GC Notify Collected",                                   \
        "Send GC Debug Request",                                      \
        "Send GC Debug Response",                                     \
        "Send GC Record Event",                                       \
        "Send Acquire Request",                                       \
        "Send Acquire Response",                                      \
        "Send Task Variant Broadcast",                                \
        "Send Constraint Request",                                    \
        "Send Constraint Response",                                   \
        "Send Constraint Release",                                    \
        "Top Level Task Complete",                                    \
        "Send MPI Rank Exchange",                                     \
        "Send Replication Distribution",                              \
        "Send Replication Collective Versioning",                     \
        "Send Replication Collective Mapping",                        \
        "Send Replication Virtual Mapping Rendezvous",                \
        "Send Replication Startup Complete",                          \
        "Send Replication Post Mapped",                               \
        "Send Replication Trigger Complete",                          \
        "Send Replication Trigger Commit",                            \
        "Send Control Replication Rendezvous Message",                \
        "Send Library Mapper Request",                                \
        "Send Library Mapper Response",                               \
        "Send Library Trace Request",                                 \
        "Send Library Trace Response",                                \
        "Send Library Projection Request",                            \
        "Send Library Projection Response",                           \
        "Send Library Sharding Request",                              \
        "Send Library Sharding Response",                             \
        "Send Library Concurrent Request",                            \
        "Send Library Concurrent Response",                           \
        "Send Library Task Request",                                  \
        "Send Library Task Response",                                 \
        "Send Library Redop Request",                                 \
        "Send Library Redop Response",                                \
        "Send Library Serdez Request",                                \
        "Send Library Serdez Response",                               \
        "Remote Op Report Uninitialized",                             \
        "Remote Op Profiling Count Update",                           \
        "Remote Op Completion Effect",                                \
        "Send Remote Trace Update",                                   \
        "Send Remote Trace Response",                                 \
        "Send Free External Allocation",                              \
        "Send Notify Collected Instances",                            \
        "Send Create Memory Pool Request",                            \
        "Send Create Memory Pool Response",                           \
        "Send Create Future Instance Request",                        \
        "Send Create Future Instance Response",                       \
        "Send Free Future Instance",                                  \
        "Send Remote Distributed ID Request",                         \
        "Send Remote Distributed ID Response",                        \
        "Control Replication Collective Future All-Reduce",           \
        "Control Replication Collective Future Broadcast",            \
        "Control Replication Collective Future Reduction",            \
        "Control Replication Collective Value All-Reduce",            \
        "Control Replication Collective Value Broadcast",             \
        "Control Replication Collective Value Exchange",              \
        "Control Replication Collective Buffer Broadcast",            \
        "Control Replication Collective Shard Sync Tree",             \
        "Control Replication Collective Shard Event Tree",            \
        "Control Replication Collective Single Task Tree",            \
        "Control Replication Collective Cross Product Partition",     \
        "Control Replication Collective Sharding Gather Collective",  \
        "Control Replication Collective Indirect Copy Exchange",      \
        "Control Replication Collective Field Descriptor Exchange",   \
        "Control Replication Collective Field Descriptor Gather",     \
        "Control Replication Collective Deppart Result Scatter",      \
        "Control Replication Collective Buffer Exchange",             \
        "Control Replication Collective Future Name Exchange",        \
        "Control Replication Collective Must Epoch Mapping Broadcast",\
        "Control Replication Collective Must Epoch Mapping Exchange", \
        "Control Replication Collective Must Epoch Dependence Exchange",\
        "Control Replication Collective Must Epoch Completion Exchange",\
        "Control Replication Collective Check Mapping",               \
        "Control Replication Collective Check Sources",               \
        "Control Replication Collective Template Index Exchange",     \
        "Control Replication Collective Unordered Exchange",          \
        "Control Replication Collective Consensus Match",             \
        "Control Replication Collective Verify Control Replication Exchange",\
        "Control Replication Collective Output Size Exchange",        \
        "Control Replication Collective Index Attach Launch Space",   \
        "Control Replication Collective Index Attach Upper Bound",    \
        "Control Replication Collective Index Attach Exchange",       \
        "Control Replication Collective Shard Participants Exchange", \
        "Control Replication Collective Implicit Sharding Functor",   \
        "Control Replication Collective Create Fill View",            \
        "Control Replication Collective Versioning Rendezvous",       \
        "Control Replication Collective View Rendezvous",             \
        "Control Replication Collective Concurrent Mapping Rendezvous",\
        "Control Replication Collective Concurrent Allreduce",        \
        "Control Replication Collective Projection Tree Exchange",    \
        "Control Replication Collective Timeout Match Exchange",      \
        "Control Replication Collective Mask Exchange",               \
        "Control Replication Collective Predicate Exchange",          \
        "Control Replication Collective Cross Product Exchange",      \
        "Control Replication Collective Tracing Set Deduplication",   \
        "Control Replication Collective Pointwise Allreduce",         \
        "Control Replication Collective Interering Points Check",     \
        "Control Replication Collective Slow Barrier",                \
        "Send Profiler Event Trigger",                                \
        "Send Profiler Event Poison",                                 \
        "Send Shutdown Notification",                                 \
        "Send Shutdown Response",                                     \
      };

    /**
     * \class VirtualChannel
     * This class provides the basic support for sending and receiving
     * messages for a single virtual channel.
     */
    class VirtualChannel {
    public:
      // Implement a three-state state-machine for sending
      // messages.  Either fully self-contained messages
      // or chains of partial messages followed by a final
      // message.
      enum MessageHeader {
        FULL_MESSAGE = 0x1,
        PARTIAL_MESSAGE = 0x2,
        FINAL_MESSAGE = 0x3,
      };
      struct PartialMessage {
      public:
        PartialMessage(void)
          : buffer(nullptr), size(0), index(0), messages(0), total(0) { }
      public:
        uint8_t *buffer;
        size_t size;
        size_t index;
        unsigned messages;
        unsigned total;
      };
    public:
      VirtualChannel(VirtualChannelKind kind,AddressSpaceID local_address_space,
               size_t max_message_size, bool profile);
      VirtualChannel(const VirtualChannel &rhs);
      ~VirtualChannel(void);
    public:
      VirtualChannel& operator=(const VirtualChannel &rhs);
    public:
      void package_message(Serializer &rez, MessageKind k, bool flush,
                           RtEvent flush_precondition,
                           Processor target, bool response);
      void process_message(const void *args, size_t arglen, 
                           AddressSpaceID remote_address_space);
      void confirm_shutdown(ShutdownManager *shutdown_manager, bool phase_one,
          Processor target, bool profiling_virtual_channel);
    private:
      void send_message(bool complete, Processor target, 
                        MessageKind kind, bool response,
                        RtEvent send_precondition);
      void handle_messages(unsigned num_messages,
                           AddressSpaceID remote_address_space,
                           const uint8_t *args, size_t arglen) const;
      static void buffer_messages(unsigned num_messages,
                                  const void *args, size_t arglen,
                                  uint8_t *&receiving_buffer,
                                  size_t &receiving_buffer_size,
                                  size_t &receiving_index,
                                  unsigned &received_messages,
                                  unsigned &partial_messages);
      void filter_unordered_events(void);
    private:
      mutable LocalLock channel_lock;
      uint8_t *const sending_buffer;
      unsigned sending_index;
      const size_t sending_buffer_size;
      RtEvent last_message_event;
      MessageHeader header;
      unsigned packaged_messages;
      // For unordered channels so we can group partial
      // messages from remote nodes
      unsigned partial_message_id;
      bool partial;
    private:
      const bool ordered_channel;
      const bool profile_outgoing_messages;
      const LgPriority request_priority;
      const LgPriority response_priority;
      static const unsigned MAX_UNORDERED_EVENTS = 32;
      std::set<RtEvent> unordered_events;
    private:
      // State for receiving messages
      // No lock for receiving messages since we know
      // that they are ordered for ordered virtual
      // channels, for un-ordered virtual channels then
      // we know that we do need the lock
      uint8_t *receiving_buffer;
      size_t receiving_buffer_size;
      size_t receiving_index;
      unsigned received_messages;
      unsigned partial_messages;
      std::map<unsigned/*message id*/,PartialMessage> *partial_assembly;
      mutable bool observed_recent;
    };

    /**
     * \class MessageManager
     * This class manages sending and receiving of message between
     * instances of the Internal runtime residing on different nodes.
     * The manager also abstracts some of the details of sending these
     * messages.  Messages can be accumulated together in bulk messages
     * for performance reason.  The runtime can also place an upper
     * bound on the size of the data communicated between runtimes in
     * an active message, which the message manager then uses to
     * break down larger messages into smaller active messages.
     *
     * On the receiving side, the message manager unpacks the messages
     * that have been sent and then call the appropriate runtime
     * methods for handling the messages.  In cases where larger
     * messages were broken down into smaller messages, then message
     * manager waits until it has received all the active messages
     * before handling the message.
     */
    class MessageManager { 
    public:
      MessageManager(AddressSpaceID remote, size_t max,
                     const Processor remote_util_group);
      MessageManager(const MessageManager &rhs) = delete;
      ~MessageManager(void);
    public:
      MessageManager& operator=(const MessageManager &rhs) = delete;
    public:
      void send_message(MessageKind message, Serializer &rez, bool flush,
                        bool response = false,
                        RtEvent flush_precondition = RtEvent::NO_RT_EVENT);
      void receive_message(const void *args, size_t arglen);
      void confirm_shutdown(ShutdownManager *shutdown_manager,
                            bool phase_one);
      // Maintain a static-mapping between message kinds and virtual channels
      static inline VirtualChannelKind find_message_vc(MessageKind kind);
    private:
      VirtualChannel *const channels;
    public:
      // State for sending messages
      const AddressSpaceID remote_address_space;
      const Processor target;
    };

  } // namespace Internal
} // namespace Legion

#include "legion/managers/message.inl"

#endif // __LEGION_MESSAGE_MANAGER_H__
