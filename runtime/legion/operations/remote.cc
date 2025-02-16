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

#include "legion/operations/remote.h"
#include "legion/contexts/inner.h"
#include "legion/operations/acquire.h"
#include "legion/operations/attach.h"
#include "legion/operations/close.h"
#include "legion/operations/copy.h"
#include "legion/operations/deletion.h"
#include "legion/operations/dependent.h"
#include "legion/operations/detach.h"
#include "legion/operations/discard.h"
#include "legion/operations/fill.h"
#include "legion/operations/mapping.h"
#include "legion/operations/release.h"
#include "legion/operations/trace.h"
#include "legion/tasks/remote.h"
#include "legion/utilities/provenance.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    ///////////////////////////////////////////////////////////// 
    // Remote Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteOp::RemoteOp(Operation *ptr, AddressSpaceID src)
      : Operation(), remote_ptr(ptr), source(src), mapper(nullptr),
        profiling_reports(0)
    //--------------------------------------------------------------------------
    {
      set_provenance(nullptr);
    }

    //--------------------------------------------------------------------------
    RemoteOp::~RemoteOp(void)
    //--------------------------------------------------------------------------
    {
      if (!profiling_requests.empty())
      {
#ifdef DEBUG_LEGION
        assert(profiling_response.exists());
#endif
        if (profiling_reports.load() > 0)
        {
          Serializer rez;
          rez.serialize(remote_ptr);
          rez.serialize(profiling_reports.load());
          rez.serialize(profiling_response);
          runtime->send_remote_op_profiling_count_update(source, rez);
        }
        else
          Runtime::trigger_event(profiling_response);
      }
      Provenance *provenance = get_provenance();
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    void RemoteOp::defer_deletion(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      DeferRemoteOpDeletionArgs args(this);
      runtime->issue_runtime_meta_task(args, 
          LG_THROUGHPUT_WORK_PRIORITY, precondition);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::pack_remote_base(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent_ctx != nullptr);
#endif
      rez.serialize(get_operation_kind());
      rez.serialize(remote_ptr);
      rez.serialize(source);
      rez.serialize(unique_op_id);
      parent_ctx->pack_inner_context(rez);
      Provenance *provenance = get_provenance();
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      rez.serialize<bool>(tracing);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::unpack_remote_base(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(unique_op_id);
      parent_ctx = InnerContext::unpack_inner_context(derez);
      set_provenance(Provenance::deserialize(derez));
      derez.deserialize<bool>(tracing);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::pack_profiling_requests(Serializer &rez,
                                           std::set<RtEvent> &applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(copy_fill_priority);
      rez.serialize<size_t>(profiling_requests.size());
      if (profiling_requests.empty())
        return;
      for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
        rez.serialize(profiling_requests[idx]);
      rez.serialize(profiling_priority);
      rez.serialize(profiling_target);
      // Send a message to the owner with an update for the extra counts
      const RtUserEvent done_event = Runtime::create_rt_user_event();
      rez.serialize<RtEvent>(done_event);
      applied.insert(done_event);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::unpack_profiling_requests(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(copy_fill_priority);
      size_t num_requests;
      derez.deserialize(num_requests);
      if (num_requests == 0)
        return;
      profiling_requests.resize(num_requests);
      for (unsigned idx = 0; idx < num_requests; idx++)
        derez.deserialize(profiling_requests[idx]);
      derez.deserialize(profiling_priority);
      derez.deserialize(profiling_target);
      derez.deserialize(profiling_response);
#ifdef DEBUG_LEGION
      assert(profiling_response.exists());
#endif
    }

    //--------------------------------------------------------------------------
    void RemoteOp::activate(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void RemoteOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void RemoteOp::report_uninitialized_usage(const unsigned index,
                                              const char *field_string,
                                              RtUserEvent reported)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->report_uninitialized_usage(index, field_string, reported);
        return;
      }
      // Ship this back to the owner node to report it there 
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ptr);
        rez.serialize(reported);
        rez.serialize(index);
        // Include the null terminator character
        const size_t length = strlen(field_string) + 1;
        rez.serialize<size_t>(length);
        rez.serialize(field_string, length);
      }
      // Send the message and wait for it to be received
      runtime->send_remote_op_report_uninitialized(source, rez);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>*
                                      RemoteOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      // We shouldn't actually be acquiring anything here so we just
      // need to make sure that we don't assert
      return nullptr;
    }

    //--------------------------------------------------------------------------
    int RemoteOp::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return copy_fill_priority;
      OpProfilingResponse response(remote_ptr, info.index, info.dst_index,fill);
      // Send the result back to the owner node
      Realm::ProfilingRequest &request = requests.add_request( 
          profiling_target, LG_LEGION_PROFILING_ID, 
          &response, sizeof(response), profiling_priority);
      bool has_timeline = false;
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            profiling_requests.begin(); it != profiling_requests.end(); it++)
      {
        const Realm::ProfilingMeasurementID measurement = 
          (Realm::ProfilingMeasurementID)*it;
        request.add_measurement(measurement);
        if (measurement == Realm::PMID_OP_TIMELINE)
          has_timeline = true;
      }
      // Need thetimeline for the operation to know how to profile this
      // profiling response
      if (!has_timeline && (runtime->profiler != nullptr))
        request.add_measurement(Realm::PMID_OP_TIMELINE);
      profiling_reports.fetch_add(count);
      return copy_fill_priority;
    }

    //--------------------------------------------------------------------------
    void RemoteOp::record_completion_effect(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (source != runtime->address_space)
      {
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_ptr);
          rez.serialize(effect);
          rez.serialize(applied);
        }
        runtime->send_remote_op_completion_effect(source, rez);
        applied.wait();
      }
      else
        remote_ptr->record_completion_effect(effect);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::record_completion_effect(ApEvent effect,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (source != runtime->address_space)
      {
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_ptr);
          rez.serialize(effect);
          rez.serialize(applied);
        }
        runtime->send_remote_op_completion_effect(source, rez);
        applied_events.insert(applied);
      }
      else
        remote_ptr->record_completion_effect(effect, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteOp::record_completion_effects(const std::set<ApEvent> &effects)
    //--------------------------------------------------------------------------
    {
      // should never be called without map applied events
      std::abort();
    }

    //--------------------------------------------------------------------------
    void RemoteOp::record_completion_effects(
                                            const std::vector<ApEvent> &effects)
    //--------------------------------------------------------------------------
    {
      // should never be called without map applied events
      std::abort();
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteOp::handle_deferred_deletion(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRemoteOpDeletionArgs *dargs = 
        (const DeferRemoteOpDeletionArgs*)args;
      delete dargs->op;
    }

    //--------------------------------------------------------------------------
    /*static*/ RemoteOp* RemoteOp::unpack_remote_operation(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      OpKind kind;
      derez.deserialize(kind);
      Operation *remote_ptr;
      derez.deserialize(remote_ptr);
      AddressSpaceID source;
      derez.deserialize(source);
      RemoteOp *result = nullptr;
      switch (kind)
      {
        case TASK_OP_KIND:
          {
            result = new RemoteTaskOp(remote_ptr, source);
            break;
          }
        case MAP_OP_KIND:
          {
            result = new RemoteMapOp(remote_ptr, source);
            break;
          }
        case COPY_OP_KIND:
          {
            result = new RemoteCopyOp(remote_ptr, source);
            break;
          }
        case POST_CLOSE_OP_KIND:
          {
            result = new RemoteCloseOp(remote_ptr, source);
            break;
          }
        case ACQUIRE_OP_KIND:
          {
            result = new RemoteAcquireOp(remote_ptr, source);
            break;
          }
        case RELEASE_OP_KIND:
          {
            result = new RemoteReleaseOp(remote_ptr, source);
            break;
          }
        case DEPENDENT_PARTITION_OP_KIND:
          {
            result = new RemotePartitionOp(remote_ptr, source);
            break;
          }
        case FILL_OP_KIND:
          {
            result = new RemoteFillOp(remote_ptr, source);
            break;
          }
        case DISCARD_OP_KIND:
          {
            result = new RemoteDiscardOp(remote_ptr, source);
            break;
          }
        case ATTACH_OP_KIND:
          {
            result = new RemoteAttachOp(remote_ptr, source);
            break;
          }
        case DETACH_OP_KIND:
          {
            result = new RemoteDetachOp(remote_ptr, source);
            break;
          }
        case DELETION_OP_KIND:
          {
            result = new RemoteDeletionOp(remote_ptr, source);
            break;
          }
        case TRACE_BEGIN_OP_KIND:
        case TRACE_RECURRENT_OP_KIND:
	case TRACE_COMPLETE_OP_KIND:
          {
            result = new RemoteTraceOp(remote_ptr, source, kind);
	    break;
          }
        default:
          std::abort();
      }
      // Do the rest of the unpack
      result->unpack_remote_base(derez);
      result->unpack(derez);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteOp::handle_report_uninitialized(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Operation *op;
      derez.deserialize(op);
      RtUserEvent reported;
      derez.deserialize(reported);
      unsigned index;
      derez.deserialize(index);
      size_t length;
      derez.deserialize(length);
      const char *field_string = (const char*)derez.get_current_pointer();
      derez.advance_pointer(length);
      op->report_uninitialized_usage(index, field_string, reported);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteOp::handle_report_profiling_count_update(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Operation *op;
      derez.deserialize(op);
      int update_count;
      derez.deserialize(update_count);
      RtUserEvent done_event;
      derez.deserialize(done_event);
#ifdef DEBUG_LEGION
      assert(done_event.exists());
#endif
      op->handle_profiling_update(update_count);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteOp::handle_completion_effect(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Operation *op;
      derez.deserialize(op);
      ApEvent effect;
      derez.deserialize(effect);
      RtUserEvent done;
      derez.deserialize(done);

      std::set<RtEvent> applied_events;
      op->record_completion_effect(effect, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(done, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done);
    }

  } // namespace Internal
} // namespace Legion
