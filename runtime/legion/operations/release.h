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

#ifndef __LEGION_RELEASE_OPERATION_H__
#define __LEGION_RELEASE_OPERATION_H__

#include "legion/analysis/versioning.h"
#include "legion/operations/collective.h"
#include "legion/operations/predicate.h"
#include "legion/operations/remote.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalRelease
     * An extension of the external-facing Release to help
     * with packing and unpacking them
     */
    class ExternalRelease : public Release,
                            public ExternalMappable {
    public:
      ExternalRelease(void);
    public:
      virtual void set_context_index(uint64_t index) = 0;
    public:
      void pack_external_release(Serializer& rez, AddressSpaceID target) const;
      void unpack_external_release(Deserializer& derez);
    };

    /**
     * \class ReleaseOp
     * Release operations are used for performing
     * user-level software coherence when tasks own
     * regions with simultaneous coherence.
     */
    class ReleaseOp : public ExternalRelease,
                      public PredicatedOp {
    public:
      ReleaseOp(void);
      ReleaseOp(const ReleaseOp& rhs) = delete;
      virtual ~ReleaseOp(void);
    public:
      ReleaseOp& operator=(const ReleaseOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, const ReleaseLauncher& launcher,
          Provenance* provenance);
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
      virtual bool record_trace_hash(TraceRecognizer& recognizer, uint64_t idx);
    public:
      virtual void predicate_false(void);
    public:
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points);
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void);
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
      virtual void complete_replay(ApEvent release_complete_event);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual const RegionRequirement& get_requirement(unsigned idx = 0) const;
    public:
      // These are helper methods for ReplReleaseOp
      virtual RtEvent finalize_complete_mapping(RtEvent event) { return event; }
      virtual void invoke_mapper(std::vector<PhysicalManager*>& src_instances);
    protected:
      void log_release_requirement(void);
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1);
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc);
      virtual void handle_profiling_update(int count);
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const;
    protected:
      ApEvent release_restrictions(
          const RegionRequirement& req, const VersionInfo& version_info,
          unsigned index, ApEvent precondition, ApEvent term_event,
          InstanceSet& restricted_instances,
          const std::vector<PhysicalManager*>& sources,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& map_applied_events
#ifdef DEBUG_LEGION
          ,
          const char* log_name, UniqueID uid
#endif
      );
    protected:
      RegionRequirement requirement;
      PhysicalRegion restricted_region;
      VersionInfo version_info;
      unsigned parent_req_index;
      std::map<PhysicalManager*, unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager* mapper;
    protected:
      struct ReleaseProfilingInfo
        : public Mapping::Mapper::ReleaseProfilingInfo {
      public:
        void* buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID> profiling_requests;
      std::vector<ReleaseProfilingInfo> profiling_info;
      RtUserEvent profiling_reported;
      int profiling_priority;
      int copy_fill_priority;
      std::atomic<int> outstanding_profiling_requests;
      std::atomic<int> outstanding_profiling_reported;
    };

    /**
     * \class ReplReleaseOp
     * A release op that is aware that it
     */
    class ReplReleaseOp
      : public ReplCollectiveViewCreator<CollectiveViewCreator<ReleaseOp> > {
    public:
      ReplReleaseOp(void);
      ReplReleaseOp(const ReplReleaseOp& rhs) = delete;
      virtual ~ReplReleaseOp(void);
    public:
      ReplReleaseOp& operator=(const ReplReleaseOp& rhs) = delete;
    public:
      void initialize_replication(
          ReplicateContext* context, bool first_local_shard);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_replay(void);
      virtual void predicate_false(void);
      virtual RtEvent finalize_complete_mapping(RtEvent event);
      virtual void invoke_mapper(std::vector<PhysicalManager*>& src_instances);
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local);
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    protected:
      CollectiveID sources_check;
      RtBarrier collective_map_barrier;
      bool is_first_local_shard;
    };

    /**
     * \class RemoteReleaseOp
     * This is a remote copy of a ReleaseOp to be used
     * for mapper calls and other operations
     */
    class RemoteReleaseOp
      : public ExternalRelease,
        public RemoteOp,
        public Heapify<RemoteReleaseOp, OPERATION_LIFETIME> {
    public:
      RemoteReleaseOp(Operation* ptr, AddressSpaceID src);
      RemoteReleaseOp(const RemoteReleaseOp& rhs) = delete;
      virtual ~RemoteReleaseOp(void);
    public:
      RemoteReleaseOp& operator=(const RemoteReleaseOp& rhs) = delete;
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
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points);
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const;
      virtual void unpack(Deserializer& derez);
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_RELEASE_OPERATION_H__
