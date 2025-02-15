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

#ifndef __LEGION_METATASK_H__
#define __LEGION_METATASK_H__

#include "legion/api/types.h"

namespace Legion {
  namespace Internal {

    // Enumeration of Legion runtime tasks
    enum LgTaskID {
      LG_SCHEDULER_ID,
      LG_TRIGGER_READY_ID,
      LG_TRIGGER_EXECUTION_ID,
      LG_TRIGGER_COMMIT_ID,
      LG_DEFERRED_EXECUTION_ID,
      LG_DEFERRED_COMPLETION_ID,
      LG_DEFERRED_COMMIT_ID,
      LG_PRE_PIPELINE_ID,
      LG_TRIGGER_DEPENDENCE_ID,
      LG_DEFERRED_MAPPED_ID,
      LG_TRIGGER_OP_ID,
      LG_TRIGGER_TASK_ID,
      LG_DEFER_MAPPER_SCHEDULER_TASK_ID,
      LG_CONTRIBUTE_COLLECTIVE_ID,
      LG_FUTURE_CALLBACK_TASK_ID,
      LG_CALLBACK_RELEASE_TASK_ID,
      LG_FUTURE_BROADCAST_TASK_ID,
      LG_TOP_FINISH_TASK_ID,
      LG_MAPPER_TASK_ID,
      LG_DISJOINTNESS_TASK_ID,
      LG_DEFER_TIMING_MEASUREMENT_TASK_ID,
      LG_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID,
      LG_INDEX_SPACE_SEMANTIC_INFO_REQ_TASK_ID,
      LG_INDEX_PART_SEMANTIC_INFO_REQ_TASK_ID,
      LG_FIELD_SPACE_SEMANTIC_INFO_REQ_TASK_ID,
      LG_FIELD_SEMANTIC_INFO_REQ_TASK_ID,
      LG_DEFER_FIELD_INFOS_TASK_ID,
      LG_REGION_SEMANTIC_INFO_REQ_TASK_ID,
      LG_PARTITION_SEMANTIC_INFO_REQ_TASK_ID,
      LG_INDEX_SPACE_DEFER_CHILD_TASK_ID,
      LG_INDEX_PART_DEFER_CHILD_TASK_ID,
      LG_INDEX_PART_DEFER_SHARD_RECTS_TASK_ID,
      LG_DEFERRED_ENQUEUE_TASK_ID,
      LG_DEFER_MAPPER_MESSAGE_TASK_ID,
      LG_DEFER_MAPPER_COLLECTION_TASK_ID,
      LG_REMOTE_VIEW_CREATION_TASK_ID,
      LG_DEFER_PERFORM_MAPPING_TASK_ID,
      LG_FINALIZE_OUTPUT_TREE_TASK_ID,
      LG_MISPREDICATION_TASK_ID,
      LG_DEFER_TRIGGER_CHILDREN_COMMIT_TASK_ID,
      LG_ORDER_CONCURRENT_LAUNCH_TASK_ID,
      LG_DEFER_MATERIALIZED_VIEW_TASK_ID,
      LG_DEFER_REDUCTION_VIEW_TASK_ID,
      LG_DEFER_PHI_VIEW_REGISTRATION_TASK_ID,
      LG_DEFER_COMPOSITE_COPY_TASK_ID,
      LG_TIGHTEN_INDEX_SPACE_TASK_ID,
      LG_REPLAY_SLICE_TASK_ID,
      LG_TRANSITIVE_REDUCTION_TASK_ID,
      LG_DELETE_TEMPLATE_TASK_ID,
      LG_DEFER_MAKE_OWNER_TASK_ID,
      LG_DEFER_APPLY_STATE_TASK_ID,
      LG_COPY_FILL_AGGREGATION_TASK_ID,
      LG_COPY_FILL_DELETION_TASK_ID,
      LG_FINALIZE_EQ_SETS_TASK_ID,
      LG_FINALIZE_OUTPUT_EQ_SET_TASK_ID,
      LG_DEFERRED_COPY_ACROSS_TASK_ID,
      LG_DEFER_REMOTE_OP_DELETION_TASK_ID,
      LG_DEFER_REMOTE_INSTANCE_TASK_ID,
      LG_DEFER_REMOTE_REDUCTION_TASK_ID,
      LG_DEFER_REMOTE_UPDATE_TASK_ID,
      LG_DEFER_REMOTE_ACQUIRE_TASK_ID,
      LG_DEFER_REMOTE_RELEASE_TASK_ID,
      LG_DEFER_REMOTE_COPIES_ACROSS_TASK_ID,
      LG_DEFER_REMOTE_OVERWRITE_TASK_ID,
      LG_DEFER_REMOTE_FILTER_TASK_ID,
      LG_DEFER_PERFORM_TRAVERSAL_TASK_ID,
      LG_DEFER_PERFORM_ANALYSIS_TASK_ID,
      LG_DEFER_PERFORM_REMOTE_TASK_ID,
      LG_DEFER_PERFORM_UPDATE_TASK_ID,
      LG_DEFER_PERFORM_REGISTRATION_TASK_ID,
      LG_DEFER_PERFORM_OUTPUT_TASK_ID,
      LG_DEFER_PHYSICAL_MANAGER_TASK_ID,
      LG_DEFER_DELETE_PHYSICAL_MANAGER_TASK_ID,
      LG_DEFER_VERIFY_PARTITION_TASK_ID,
      LG_DEFER_RELEASE_ACQUIRED_TASK_ID,
      LG_DEFER_COPY_ACROSS_TASK_ID,
      LG_DEFER_COLLECTIVE_MESSAGE_TASK_ID,
      LG_MALLOC_INSTANCE_TASK_ID,
      LG_FREE_INSTANCE_TASK_ID,
      LG_DEFER_TRACE_UPDATE_TASK_ID,
      LG_DEFER_DELETE_FUTURE_INSTANCE_TASK_ID,
      LG_FREE_EXTERNAL_TASK_ID,
      LG_DEFER_CONSENSUS_MATCH_TASK_ID,
      LG_DEFER_COLLECTIVE_TASK_ID,
      LG_DEFER_ISSUE_FILL_TASK_ID,
      LG_DEFER_MUST_EPOCH_RETURN_TASK_ID,
      LG_DEFER_DELETION_COMMIT_TASK_ID,
      LG_YIELD_TASK_ID,
      LG_AUTO_TRACE_PROCESS_REPEATS_TASK_ID,
      // this marks the beginning of task IDs tracked by the shutdown algorithm
      LG_BEGIN_SHUTDOWN_TASK_IDS,
      LG_RETRY_SHUTDOWN_TASK_ID = LG_BEGIN_SHUTDOWN_TASK_IDS,
      // Message ID goes at the end so we can append additional 
      // message IDs here for the profiler and separate meta-tasks
      LG_MESSAGE_ID,
      LG_LAST_TASK_ID, // This one should always be last
    };  

    // Make this a macro so we can keep it close to 
    // declaration of the task IDs themselves
#define LG_TASK_DESCRIPTIONS(name)                               \
      const char *name[LG_LAST_TASK_ID] = {                      \
        "Scheduler",                                              \
        "Trigger Ready",                                          \
        "Trigger Execution",                                      \
        "Trigger Commit",                                         \
        "Deferred Execution",                                     \
        "Deferred Completion",                                    \
        "Deferred Commit",                                        \
        "Prepipeline Stage",                                      \
        "Logical Dependence Analysis",                            \
        "Deferred Mapped",                                        \
        "Trigger Operation Mapping",                              \
        "Trigger Task Mapping",                                   \
        "Defer Mapper Scheduler",                                 \
        "Contribute Collective",                                  \
        "Future Callback",                                        \
        "Future Callback Release",                                \
        "Future Broadcast",                                       \
        "Top Finish",                                             \
        "Mapper Task",                                            \
        "Disjointness Test",                                      \
        "Defer Timing Measurement",                               \
        "Task Impl Semantic Request",                             \
        "Index Space Semantic Request",                           \
        "Index Partition Semantic Request",                       \
        "Field Space Semantic Request",                           \
        "Field Semantic Request",                                 \
        "Defer Field Infos Request",                              \
        "Region Semantic Request",                                \
        "Partition Semantic Request",                             \
        "Defer Index Space Child Request",                        \
        "Defer Index Partition Child Request",                    \
        "Defer Index Partition Find Shard Rects",                 \
        "Deferred Enqueue Task",                                  \
        "Deferred Mapper Message",                                \
        "Deferred Mapper Instance Collective",                    \
        "Remote View Creation",                                   \
        "Defer Task Perform Mapping",                             \
        "Finalize Output Regions Eq KD Tree",                     \
        "Handle Mapping Mispredication",                          \
        "Defer Trigger Children Commit",                          \
        "Order Concurrent Launch",                                \
        "Defer Materialized View Registration",                   \
        "Defer Reduction View Registration",                      \
        "Defer Phi View Registration",                            \
        "Defer Composite Copy",                                   \
        "Tighten Index Space",                                    \
        "Replay Physical Trace",                                  \
        "Template Transitive Reduction",                          \
        "Delete Physical Template",                               \
        "Defer Equivalence Set Make Owner",                       \
        "Defer Equivalence Set Apply State",                      \
        "Copy Fill Aggregation",                                  \
        "Copy Fill Deletion",                                     \
        "Finalize Equivalence Sets",                              \
        "Finalize Output Equivalence Set",                        \
        "Deferred Copy Across",                                   \
        "Defer Remote Op Deletion",                               \
        "Defer Remote Instance Request",                          \
        "Defer Remote Reduction Request",                         \
        "Defer Remote Update Equivalence Set",                    \
        "Defer Remote Acquire",                                   \
        "Defer Remote Release",                                   \
        "Defer Remote Copy Across",                               \
        "Defer Remote Overwrite Equivalence Set",                 \
        "Defer Remote Filter Equivalence Set",                    \
        "Defer Physical Analysis Traversal Stage",                \
        "Defer Physical Analysis Analyze Equivalence Set Stage",  \
        "Defer Physical Analysis Remote Stage",                   \
        "Defer Physical Analysis Update Stage",                   \
        "Defer Physical Analysis Registration Stage",             \
        "Defer Physical Analysis Output Stage",                   \
        "Defer Physical Manager Registration",                    \
        "Defer Physical Manager Deletion",                        \
        "Defer Verify Partition",                                 \
        "Defer Release Acquired Instances",                       \
        "Defer Copy-Across Execution for Preimages",              \
        "Defer Collective Instance Message",                      \
        "Malloc Instance",                                        \
        "Free Instance",                                          \
        "Defer Trace Update",                                     \
        "Defer Delete Future Instance",                           \
        "Free External Allocation",                               \
        "Defer Consensus Match",                                  \
        "Defer Collective Async",                                 \
        "Defer Issue Fill",                                       \
        "Defer Must Epoch Return Resources",                      \
        "Defer Deletion Commit",                                  \
        "Yield",                                                  \
        "Auto Trace Find Repeats",                                \
        "Retry Shutdown",                                         \
        "Remote Message",                                         \
      };

    // Methodology for assigning priorities to meta-tasks:
    // Minimum and low priority are for things like profiling
    // that we don't want to interfere with normal execution.
    // Resource priority is reserved for tasks that have been 
    // granted resources like reservations. Running priority
    // is the highest and guarantees that we drain out any 
    // previously running tasks over starting new ones. The rest
    // of the priorities are classified as either 'throughput'
    // or 'latency' sensitive. Under each of these two major
    // categories there are four sub-priorities:
    //  - work: general work to be done
    //  - deferred: work that was already scheduled but 
    //              for which a continuation had to be 
    //              made so we don't want to wait behind
    //              work that hasn't started yet
    //  - messsage: a message from a remote node that we
    //              should handle sooner than our own
    //              work since work on the other node is
    //              blocked waiting on our response
    //  - response: a response message from a remote node
    //              that we should handle to unblock work
    //              on our own node
    enum LgPriority {
      LG_MIN_PRIORITY = INT_MIN,
      LG_LOW_PRIORITY = -1,
      // Throughput priorities
      LG_THROUGHPUT_WORK_PRIORITY = 0,
      LG_THROUGHPUT_DEFERRED_PRIORITY = 1,
      LG_THROUGHPUT_MESSAGE_PRIORITY = 2,
      LG_THROUGHPUT_RESPONSE_PRIORITY = 3,
      // Latency priorities
      LG_LATENCY_WORK_PRIORITY = 4,
      LG_LATENCY_DEFERRED_PRIORITY = 5,
      LG_LATENCY_MESSAGE_PRIORITY = 6,
      LG_LATENCY_RESPONSE_PRIORITY = 7,
      // Resource priorities
      LG_RESOURCE_PRIORITY = 8,
      // Running priorities
      LG_RUNNING_PRIORITY = 9,
    };

    /**
     * \class LgTaskArgs
     * The base class for all Legion Task arguments
     */
    template<typename T>
    struct LgTaskArgs {
    public:
      LgTaskArgs(::legion_unique_id_t uid)
        : provenance(uid),
#ifdef DEBUG_LEGION_CALLERS
          lg_call_id(implicit_task_kind),
#endif
          lg_task_id(T::TASK_ID) { }
    public:
      // In this order for alignment reasons
      const ::legion_unique_id_t provenance;
#ifdef DEBUG_LEGION_CALLERS
      const LgTaskID lg_call_id;
#endif
      const LgTaskID lg_task_id;
    };

  } // namespace Internal
} // namespace Legion

#endif // __LEGION_METATASK_H__
