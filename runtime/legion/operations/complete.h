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

#ifndef __LEGION_COMPLETE_OPERATION_H__
#define __LEGION_COMPLETE_OPERATION_H__

#include "legion/api/redop.h"
#include "legion/operations/trace.h"
#include "legion/tracing/physical.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class CompleteOp
     * A pure virtual interface for completion ops to implement
     * to help with completing captures and replays of templates
     */
    class CompleteOp {
    public:
      virtual FenceOp* get_complete_operation(void) = 0;
      virtual void begin_replayable_exchange(ReplayableStatus status) { }
      virtual void end_replayable_exchange(ReplayableStatus &status) { }
      virtual void begin_idempotent_exchange(IdempotencyStatus idempotent) { }
      virtual void end_idempotent_exchange(IdempotencyStatus &idempotent) { }
      virtual void sync_compute_frontiers(RtEvent event) { assert(false); }
      virtual void deduplicate_condition_sets(
          std::map<EquivalenceSet*,unsigned> &condition_sets) { }
    };

    /**
     * \class TraceCompleteOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when the execution
     * of a trace has been completed.  This fence operation
     * then registers dependences on all operations in the trace
     * and becomes the new current fence.
     */
    class TraceCompleteOp : public TraceOp, public CompleteOp {
    public:
      TraceCompleteOp(void);
      TraceCompleteOp(const TraceCompleteOp &rhs) = delete;
      virtual ~TraceCompleteOp(void);
    public:
      TraceCompleteOp& operator=(const TraceCompleteOp &rhs) = delete;
    public:
      void initialize_complete(InnerContext *ctx, LogicalTrace *trace,
                               Provenance *provenance, bool remove_reference);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void); 
      virtual bool record_trace_hash(TraceRecognizer &identifier, uint64_t idx);
    protected:
      virtual FenceOp* get_complete_operation(void) { return this; }
    protected:
      bool has_blocking_call;
      bool remove_trace_reference;
    };

    class ReplCompleteOp : public ReplTraceOp, public CompleteOp {
    public:
      ReplCompleteOp(void) : ReplTraceOp() { }
      virtual ~ReplCompleteOp(void) { }
    };

    // Mixin class for adding support for replicated complete interfaces
    template<typename OP>
    class ReplTraceComplete : public OP {
    public:
      ReplTraceComplete(void);
      virtual ~ReplTraceComplete(void) { }
    protected:
      void initialize_complete(ReplicateContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual FenceOp* get_complete_operation(void) { return this; }
      virtual void begin_replayable_exchange(ReplayableStatus status);
      virtual void end_replayable_exchange(ReplayableStatus &status);
      virtual void begin_idempotent_exchange(IdempotencyStatus idempotent);
      virtual void end_idempotent_exchange(IdempotencyStatus &idempotent);
      virtual void sync_compute_frontiers(RtEvent event);
      virtual void deduplicate_condition_sets(
          std::map<EquivalenceSet*,unsigned> &condition_sets);
    private:
      CollectiveID replayable_collective_id;
      CollectiveID idempotent_collective_id;
      CollectiveID sync_compute_frontiers_collective_id;
      CollectiveID deduplication_collective_id;
      AllReduceCollective<ProdReduction<bool>,false > *replayable_collective;
      AllReduceCollective<ProdReduction<bool>,false > *idempotent_collective;
    };

    /**
     * \class ReplTraceCompleteOp
     * Control replicated version of TraceCompleteOp
     */
    class ReplTraceCompleteOp : public ReplTraceComplete<ReplCompleteOp> {
    public:
      ReplTraceCompleteOp(void);
      ReplTraceCompleteOp(const ReplTraceCompleteOp &rhs) = delete;
      virtual ~ReplTraceCompleteOp(void);
    public:
      ReplTraceCompleteOp& operator=(const ReplTraceCompleteOp &rhs) = delete;
    public:
      void initialize_complete(ReplicateContext *ctx, LogicalTrace *trace,
                               Provenance *provenance, bool remove_reference);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual bool record_trace_hash(TraceRecognizer &recognizer, uint64_t idx);
    protected:
      bool has_blocking_call;
      bool remove_trace_reference;
    };

    /**
     * \class TracingSetDeduplication
     * This class performs an all-gather on the names of equivalence sets
     * which might be replicated across the shards and therefore we want 
     * to efficiently deduplicate which of them will be captured by 
     * different nodes for establishing tracing pre/post-conditions
     */
    class TracingSetDeduplication : public AllGatherCollective<false> {
    public:
      TracingSetDeduplication(ReplicateContext *ctx, CollectiveID id);
      TracingSetDeduplication(const TracingSetDeduplication &rhs) = delete;
      virtual ~TracingSetDeduplication(void);
    public:
      TracingSetDeduplication& operator=(
          const TracingSetDeduplication &rhs) = delete;
    public:
       virtual MessageKind get_message_kind(void) const
        { return SEND_CONTROL_REPLICATION_TRACING_SET_DEDUPLICATION; }
      virtual void pack_collective_stage(ShardID target,
                                         Serializer &rez, int stage);
      virtual void unpack_collective_stage(Deserializer &derez, int stage);
    public:
      void record_set(DistributedID did, unsigned parent_req_index);
      const std::map<DistributedID,unsigned>& all_gather_collective_sets(void);
    private:
      std::map<DistributedID,unsigned> collective_sets;
    };

  } // namespace Internal
} // namespace Legion

#endif // __LEGION_COMPLETE_OPERATION_H__
