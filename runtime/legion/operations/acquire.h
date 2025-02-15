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

#ifndef __LEGION_ACQUIRE_H__
#define __LEGION_ACQUIRE_H__

#include "legion/analysis/versioning.h"
#include "legion/operations/collective.h"
#include "legion/operations/predicate.h"
#include "legion/operations/remote.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalAcquire
     * An extension of the external-facing Acquire to help 
     * with packing and unpacking them
     */
    class ExternalAcquire : public Acquire, public ExternalMappable {
    public:
      ExternalAcquire(void);
    public:
      virtual void set_context_index(uint64_t index) = 0;
    public:
      void pack_external_acquire(Serializer &rez, AddressSpaceID target) const;
      void unpack_external_acquire(Deserializer &derez);
    };

    /**
     * \class AcquireOp
     * Acquire operations are used for performing
     * user-level software coherence when tasks own
     * regions with simultaneous coherence.
     */
    class AcquireOp : public ExternalAcquire, public PredicatedOp {
    public:
      AcquireOp(void);
      AcquireOp(const AcquireOp &rhs) = delete;
      virtual ~AcquireOp(void);
    public:
      AcquireOp& operator=(const AcquireOp &rhs) = delete;
    public:
      void initialize(InnerContext *ctx, const AcquireLauncher &launcher,
                      Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const; 
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_complete(ApEvent complete);
      virtual bool record_trace_hash(TraceRecognizer &recognizer, uint64_t idx);
    public:
      virtual void predicate_false(void);
    public:
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual std::map<PhysicalManager*,unsigned>*
                   get_acquired_instances_ref(void);
    public: 
      virtual UniqueID get_unique_id(void) const;
      virtual uint64_t get_context_index(void) const;
      virtual void set_context_index(uint64_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const;
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      // From Memoizable
      virtual void complete_replay(ApEvent acquire_complete_event);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual const RegionRequirement& get_requirement(unsigned idx = 0) const;
    public:
      // These are helper methods for ReplAcquireOp
      virtual RtEvent finalize_complete_mapping(RtEvent event) { return event; }
    protected:
      void invoke_mapper(void);
      void log_acquire_requirement(void);
      virtual int add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse &response, const void *orig,
          size_t orig_length, LgEvent &fevent, bool &failed_alloc);
      virtual void handle_profiling_update(int count);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      ApEvent acquire_restrictions(const RegionRequirement &req,
                                   const VersionInfo &version_info,
                                   unsigned index,
                                   ApEvent precondition, ApEvent term_event,
                                   InstanceSet &restricted_instances,
                                   const PhysicalTraceInfo &trace_info,
                                   std::set<RtEvent> &map_applied_events
#ifdef DEBUG_LEGION
                                   , const char *log_name
                                   , UniqueID uid
#endif
                                   );
    protected:
      RegionRequirement requirement;
      PhysicalRegion    restricted_region;
      VersionInfo       version_info;
      unsigned          parent_req_index;
      std::map<PhysicalManager*,unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager*    mapper;
    protected:
      std::vector<ProfilingMeasurementID>            profiling_requests;
      RtUserEvent                                    profiling_reported;
      int                                            profiling_priority;
      int                                            copy_fill_priority;
      std::atomic<int>                   outstanding_profiling_requests;
      std::atomic<int>                   outstanding_profiling_reported;
    };

    /**
     * \class ReplAcquireOp
     * An acquire op that is aware that it is
     * executing in a control replicated context
     */
    class ReplAcquireOp : 
      public ReplCollectiveViewCreator<CollectiveViewCreator<AcquireOp> > {
    public:
      ReplAcquireOp(void);
      ReplAcquireOp(const ReplAcquireOp &rhs) = delete;
      virtual ~ReplAcquireOp(void);
    public:
      ReplAcquireOp& operator=(const ReplAcquireOp &rhs) = delete;
    public:
      void initialize_replication(ReplicateContext *context,
                                  bool first_local_shard);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void predicate_false(void);
      virtual RtEvent finalize_complete_mapping(RtEvent precondition);
      virtual bool perform_collective_analysis(CollectiveMapping *&mapping,
                                               bool &first_local);
      virtual RtEvent perform_collective_versioning_analysis(unsigned index,
                       LogicalRegion handle, EqSetTracker *tracker,
                       const FieldMask &mask, unsigned parent_req_index);
    protected:
      RtBarrier collective_map_barrier;
      bool is_first_local_shard;
    };

    /**
     * \class RemoteAcquireOp
     * This is a remote copy of a AcquireOp to be used
     * for mapper calls and other operations
     */
    class RemoteAcquireOp : public ExternalAcquire, public RemoteOp,
      public Heapify<RemoteAcquireOp,OPERATION_LIFETIME> {
    public:
      RemoteAcquireOp(Operation *ptr, AddressSpaceID src);
      RemoteAcquireOp(const RemoteAcquireOp &rhs) = delete;
      virtual ~RemoteAcquireOp(void);
    public:
      RemoteAcquireOp& operator=(const RemoteAcquireOp &rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual uint64_t get_context_index(void) const;
      virtual void set_context_index(uint64_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez);
    };

  } // namespace Internal
} // namespace Legion

#endif // __LEGION_ACQUIRE_H__
