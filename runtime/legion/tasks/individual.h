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

#ifndef __LEGION_INDIVIDUAL_TASK_H__
#define __LEGION_INDIVIDUAL_TASK_H__

#include "legion/tasks/single.h"

namespace Legion {
  namespace Internal {

    /**
     * \class IndividualTask
     * This class serves as the basis for all individual task
     * launch calls performed by the runtime.
     */
    class IndividualTask : public SingleTask,
                           public Heapify<IndividualTask, RUNTIME_LIFETIME> {
    public:
      IndividualTask(void);
      IndividualTask(const IndividualTask& rhs) = delete;
      virtual ~IndividualTask(void);
    public:
      IndividualTask& operator=(const IndividualTask& rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    protected:
      virtual SingleTask* get_origin_task(void) const { return orig_task; }
      virtual Domain get_slice_domain(void) const { return Domain::NO_DOMAIN; }
      virtual ShardID get_shard_id(void) const { return 0; }
      virtual size_t get_total_shards(void) const { return 1; }
      virtual DomainPoint get_shard_point(void) const { return DomainPoint(0); }
      virtual Domain get_shard_domain(void) const
      {
        return Domain(DomainPoint(0), DomainPoint(0));
      }
      virtual Operation* get_origin_operation(void)
      {
        return is_remote() ? orig_task : this;
      }
    public:
      Future initialize_task(
          InnerContext* ctx, const TaskLauncher& launcher,
          Provenance* provenance, bool top_level = false,
          bool must_epoch_launch = false,
          std::vector<OutputRequirement>* outputs = nullptr);
      void perform_base_dependence_analysis(void);
    protected:
      void create_output_regions(std::vector<OutputRequirement>& outputs);
    public:
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void report_interfering_requirements(
          unsigned idx1, unsigned idx2);
      virtual bool record_trace_hash(TraceHashRecorder& recorder, uint64_t idx);
      // Virtual method for creating the future for this task so that
      // we can overload for control replication
      virtual Future create_future(void);
    public:
      virtual void predicate_false(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(
          MustEpochOp* owner = nullptr, const DeferMappingArgs* args = nullptr);
      virtual bool finalize_map_task_output(
          Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
          MustEpochOp* must_epoch_owner);
      virtual void handle_future_size(
          size_t return_type_size, std::set<RtEvent>& applied_events);
      virtual void record_output_registered(
          RtEvent registered, std::set<RtEvent>& applied_events);
      virtual bool is_stealable(void) const;
      virtual bool replicate_task(void);
    public:
      virtual bool is_output_valid(unsigned idx) const;
      virtual bool is_output_grouped(unsigned idx) const;
    public:
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual void trigger_complete(ApEvent effects);
      virtual void trigger_task_commit(void);
    public:
      virtual void handle_future(
          ApEvent effects, FutureInstance* instance, const void* metadata,
          size_t metasize, FutureFunctor* functor, Processor future_proc,
          bool own_functor);
      virtual void handle_mispredication(void);
      virtual void prepare_map_must_epoch(void);
    public:
      virtual bool send_task(
          Processor target, std::vector<SingleTask*>& others);
      virtual bool pack_task(Serializer& rez, AddressSpaceID target);
      virtual bool unpack_task(
          Deserializer& derez, Processor current,
          std::set<RtEvent>& ready_events);
      virtual bool is_top_level_task(void) const { return top_level_task; }
    public:
      void set_concurrent_postcondition(RtEvent postcondition);
      virtual uint64_t order_collectively_mapped_unbounded_pools(
          uint64_t lamport_clock, bool need_result);
      virtual ApEvent order_concurrent_launch(ApEvent start, VariantImpl* impl);
      virtual void concurrent_allreduce(
          ProcessorManager* manager, uint64_t lamport_clock, VariantID vid,
          bool poisoned);
      virtual void perform_concurrent_task_barrier(void);
      void finish_concurrent_allreduce(uint64_t lamport_clock, bool poisoned);
    public:
      void pack_remote_complete(Serializer& rez, ApEvent effect);
      void pack_remote_commit(Serializer& rez, RtEvent precondition);
      void unpack_remote_complete(Deserializer& derez);
      void unpack_remote_commit(Deserializer& derez);
    public:
      // From MemoizableOp
      virtual void complete_replay(ApEvent pre);
    protected:
      Future result;
    protected:
      std::vector<OutputOptions> output_region_options;
      // Event for when the output regions are registered with the context
      RtEvent output_regions_registered;
      RtEvent remote_commit_precondition;
    protected:
      // Events for concurrent task launches, only used for when this task
      // is part of a must-epoch launch
      RtUserEvent concurrent_precondition;
      RtEvent concurrent_postcondition;
    protected:
      // Information for remotely executing task
      IndividualTask* orig_task;  // Not a valid pointer when remote
      UniqueID remote_unique_id;
    protected:
      Future predicate_false_future;
      BufferManager<IndividualTask, OPERATION_LIFETIME> predicate_false_result;
    protected:
      bool sent_remotely;
    protected:
      friend class Internal;
      // Special field for the top level task
      bool top_level_task;
    };

    /**
     * \class ReplIndividualTask
     * An individual task that is aware that it is
     * being executed in a control replication context.
     */
    class ReplIndividualTask : public IndividualTask {
    public:
      ReplIndividualTask(void);
      ReplIndividualTask(const ReplIndividualTask& rhs) = delete;
      virtual ~ReplIndividualTask(void);
    public:
      ReplIndividualTask& operator=(const ReplIndividualTask& rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void predicate_false(void);
      virtual void shard_off(RtEvent mapped_precondition);
      virtual void prepare_map_must_epoch(void);
    public:
      // Override this so it can broadcast the future result
      virtual Future create_future(void);
    public:
      // Override for saying when it is safe to use output region trees
      virtual void record_output_registered(
          RtEvent registered, std::set<RtEvent>& applied_events);
    public:
      void initialize_replication(ReplicateContext* ctx);
      void set_sharding_function(
          ShardingID functor, ShardingFunction* function);
    protected:
      ShardID owner_shard;
      IndexSpaceNode* launch_space;
      ShardingID sharding_functor;
      ShardingFunction* sharding_function;
      RtBarrier output_bar;
    public:
      inline void set_sharding_collective(ShardingGatherCollective* collective)
      {
        sharding_collective = collective;
      }
    protected:
      ShardingGatherCollective* sharding_collective;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INDVIDUAL_TASK_H__
