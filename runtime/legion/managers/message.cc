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

#include <unistd.h>  // usleep

#include "legion/managers/message.h"
#include "legion/kernel/runtime.h"
#include "legion/managers/shard.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    MessageHeader::MessageHeader(MessageKind k, VirtualChannelKind vc)
      : LgTaskArgs<MessageHeader>(implicit_provenance), kind(k), channel(vc),
        sender(runtime->address_space)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    /*static*/ void MessageHeader::handle(const void* data, size_t size)
    //--------------------------------------------------------------------------
    {
      legion_assert(sizeof(MessageHeader) <= size);
      const MessageHeader* header = static_cast<const MessageHeader*>(data);
      legion_assert(header->lg_task_id == TASK_ID);
#ifdef LEGION_DEBUG_CALLERS
      implicit_task_caller = header->lg_call_id;
#endif
      implicit_provenance = header->provenance;
      // Find the handler for this message
      legion_assert(header->kind < LAST_SEND_KIND);
      void (*handler)(Deserializer&, AddressSpaceID) =
          MessageManager::message_handler_table[header->kind];
      assert(handler != nullptr);
      Deserializer derez(header + 1, size - sizeof(MessageHeader));
      (*handler)(derez, header->sender);
      // Record that we've seen this message
      MessageManager* manager = runtime->find_messenger(header->sender);
      manager->find_channel(header->channel).record_seen(header->kind);
    }

    /////////////////////////////////////////////////////////////
    // Virtual Channel
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualChannel::VirtualChannel(
        VirtualChannelKind kind, AddressSpaceID local_address_space,
        size_t max_message_size, bool profile_outgoing)
      : ordered_channel(
            (kind != DEFAULT_VIRTUAL_CHANNEL) &&
            (kind != THROUGHPUT_VIRTUAL_CHANNEL)),
        profile_outgoing_messages(profile_outgoing),
        request_priority(
            (kind == THROUGHPUT_VIRTUAL_CHANNEL) ?
                LG_THROUGHPUT_MESSAGE_PRIORITY :
            (kind == UPDATE_VIRTUAL_CHANNEL) ? LG_LATENCY_DEFERRED_PRIORITY :
                                               LG_LATENCY_MESSAGE_PRIORITY),
        response_priority(
            (kind == THROUGHPUT_VIRTUAL_CHANNEL) ?
                LG_THROUGHPUT_RESPONSE_PRIORITY :
            (kind == UPDATE_VIRTUAL_CHANNEL) ? LG_LATENCY_MESSAGE_PRIORITY :
                                               LG_LATENCY_RESPONSE_PRIORITY),
        observed_recent(true)
    //--------------------------------------------------------------------------
    //
    { }

    //--------------------------------------------------------------------------
    VirtualChannel::~VirtualChannel(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void VirtualChannel::send_message(
        MessageKind kind, const Serializer& rez, RtEvent send_precondition,
        Processor target, bool response)
    //--------------------------------------------------------------------------
    {
      // Need an excluisve lock if this an ordered channel, otherwise this
      // is just a formality and many messages can be sent in parallel
      // and just can't race with shutdown tests
      AutoLock c_lock(channel_lock, ordered_channel ? 0 : 1, ordered_channel);
      // Send the message directly there, don't go through the
      // runtime interface to avoid being counted, still include
      // a profiling request though if necessary in order to
      // see waits on message handlers
      if (profile_outgoing_messages)
      {
        Realm::ProfilingRequestSet requests;
        const RtEvent precondition =
            ordered_channel ? (send_precondition.exists() ?
                                   Runtime::merge_events(
                                       send_precondition, last_message_event) :
                                   last_message_event) :
                              send_precondition;
        LegionProfiler::add_message_request(
            requests, kind, target, precondition);
        const RtEvent message_done(target.spawn(
#ifdef LEGION_SEPARATE_META_TASKS
            LG_TASK_ID + LG_MESSAGE_ID + kind,
#else
            LG_TASK_ID,
#endif
            rez.get_buffer(), rez.get_used_bytes(), requests, precondition,
            response ? response_priority : request_priority));
        if (!ordered_channel)
        {
          last_message_event = message_done;
          unordered_events.insert(last_message_event);
          if (unordered_events.size() >= MAX_UNORDERED_EVENTS)
            filter_unordered_events();
        }
      }
      else
      {
        RtEvent message_done(target.spawn(
#ifdef LEGION_SEPARATE_META_TASKS
            LG_TASK_ID + LG_MESSAGE_ID + kind,
#else
            LG_TASK_ID,
#endif
            rez.get_buffer(), rez.get_used_bytes(),
            ordered_channel ? (send_precondition.exists() ?
                                   Runtime::merge_events(
                                       send_precondition, last_message_event) :
                                   last_message_event) :
                              send_precondition,
            response ? response_priority : request_priority));
        if (!ordered_channel)
        {
          last_message_event = message_done;
          unordered_events.insert(last_message_event);
          if (unordered_events.size() >= MAX_UNORDERED_EVENTS)
            filter_unordered_events();
        }
      }
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::filter_unordered_events(void)
    //--------------------------------------------------------------------------
    {
      // Lock held from caller
      legion_assert(!ordered_channel);
      legion_assert(unordered_events.size() >= MAX_UNORDERED_EVENTS);
      // Prune out any triggered events
      for (std::set<RtEvent>::iterator it = unordered_events.begin();
           it != unordered_events.end();
           /*nothing*/)
      {
        if (it->has_triggered())
        {
          std::set<RtEvent>::iterator to_delete = it++;
          unordered_events.erase(to_delete);
        }
        else
          it++;
      }
      // If we still have too many events, collapse them down
      if (unordered_events.size() >= MAX_UNORDERED_EVENTS)
      {
        const RtEvent summary = Runtime::merge_events(unordered_events);
        unordered_events.clear();
        unordered_events.insert(summary);
      }
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::record_seen(MessageKind kind)
    //--------------------------------------------------------------------------
    {
      // Any message that is not a shutdown message needs to be recorded
      if ((kind != SEND_SHUTDOWN_NOTIFICATION) &&
          (kind != SEND_SHUTDOWN_RESPONSE))
      {
        AutoLock c_lock(channel_lock);
        observed_recent = true;
      }
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::confirm_shutdown(
        ShutdownManager* shutdown_manager, bool phase_one, Processor target,
        bool profiling_virtual_channel)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(channel_lock);
      if (phase_one)
      {
        if (ordered_channel)
        {
          if (!last_message_event.has_triggered())
          {
            // Subscribe to make sure we see this trigger
            last_message_event.subscribe();
            // A little hack here for slow gasnet conduits
            // If the event didn't trigger yet, make sure its just
            // because we haven't gotten the return message yet
            usleep(1000);
            if (!last_message_event.has_triggered())
              shutdown_manager->record_pending_message(last_message_event);
            else
              observed_recent = false;
          }
          else
            observed_recent = false;
        }
        else
        {
          observed_recent = false;
          for (std::set<RtEvent>::const_iterator it = unordered_events.begin();
               it != unordered_events.end(); it++)
          {
            if (!it->has_triggered())
            {
              // Subscribe to make sure we see this trigger
              it->subscribe();
              // A little hack here for slow gasnet conduits
              // If the event didn't trigger yet, make sure its just
              // because we haven't gotten the return message yet
              usleep(1000);
              if (!it->has_triggered())
              {
                shutdown_manager->record_pending_message(*it);
                observed_recent = true;
                break;
              }
            }
          }
        }
      }
      else
      {
        if (observed_recent)
        {
          shutdown_manager->record_recent_message();
        }
        else
        {
          if (ordered_channel)
          {
            if (!last_message_event.has_triggered())
            {
              // Subscribe to make sure we see this trigger
              last_message_event.subscribe();
              // A little hack here for slow gasnet conduits
              // If the event didn't trigger yet, make sure its just
              // because we haven't gotten the return message yet
              usleep(1000);
              if (!last_message_event.has_triggered())
                shutdown_manager->record_recent_message();
            }
          }
          else
          {
            for (std::set<RtEvent>::const_iterator it =
                     unordered_events.begin();
                 it != unordered_events.end(); it++)
            {
              if (!it->has_triggered())
              {
                // Subscribe to make sure we see this trigger
                it->subscribe();
                // A little hack here for slow gasnet conduits
                // If the event didn't trigger yet, make sure its just
                // because we haven't gotten the return message yet
                usleep(1000);
                if (!it->has_triggered())
                {
                  shutdown_manager->record_recent_message();
                  break;
                }
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ MessageManager* MessageManager::find_manager(
        AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      return runtime->find_messenger(target);
    }

#if 0
    //--------------------------------------------------------------------------
    void VirtualChannel::handle_message(MessageKind kind
        AddressSpaceID remote_address_space, const void* args, size_t arglen) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < num_messages; idx++)
      {
        // Pull off the message kind and the size of the message
        legion_assert(arglen >= (sizeof(MessageKind) + sizeof(size_t)));
        MessageKind kind;
        memcpy(&kind, args, sizeof(kind));
        // Any message that is not a shutdown message needs to be recorded
        if (!observed_recent && (kind != SEND_SHUTDOWN_NOTIFICATION) &&
            (kind != SEND_SHUTDOWN_RESPONSE))
          observed_recent = true;
        args += sizeof(kind);
        arglen -= sizeof(kind);
        memcpy(&implicit_provenance, args, sizeof(implicit_provenance));
        args += sizeof(implicit_provenance);
        arglen -= sizeof(implicit_provenance);
#ifdef LEGION_DEBUG_CALLERS
        implicit_task_kind = (LgTaskID)(LG_MESSAGE_ID + kind);
        memcpy(&implicit_task_caller, args, sizeof(implicit_task_caller));
        args += sizeof(implicit_task_caller);
        arglen -= sizeof(implicit_task_caller);
#endif
        size_t message_size;
        memcpy(&message_size, args, sizeof(message_size));
        args += sizeof(message_size);
        arglen -= sizeof(message_size);
        legion_assert((idx != (num_messages - 1)) || (message_size == arglen));
        // Build the deserializer
        Deserializer derez(args, message_size);
        switch (kind)
        {
          case SEND_STARTUP_BARRIER:
            {
              runtime->handle_startup_barrier(derez);
              break;
            }
          case TASK_MESSAGE:
            {
              runtime->handle_task(derez);
              break;
            }
          case STEAL_MESSAGE:
            {
              runtime->handle_steal(derez);
              break;
            }
          case ADVERTISEMENT_MESSAGE:
            {
              runtime->handle_advertisement(derez);
              break;
            }
#ifdef LEGION_USE_LIBDL
          case SEND_REGISTRATION_CALLBACK:
            {
              runtime->handle_registration_callback(derez);
              break;
            }
#endif
          case SEND_REMOTE_TASK_REPLAY:
            {
              runtime->handle_remote_task_replay(derez);
              break;
            }
          case SEND_REMOTE_TASK_PROFILING_RESPONSE:
            {
              runtime->handle_remote_task_profiling_response(derez);
              break;
            }
          case SEND_SHARED_OWNERSHIP:
            {
              runtime->handle_shared_ownership(derez);
              break;
            }
          case SEND_INDEX_SPACE_REQUEST:
            {
              runtime->handle_index_space_request(derez);
              break;
            }
          case SEND_INDEX_SPACE_RESPONSE:
            {
              runtime->handle_index_space_response(derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_RETURN:
            {
              runtime->handle_index_space_return(derez);
              break;
            }
          case SEND_INDEX_SPACE_SET:
            {
              runtime->handle_index_space_set(derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_CHILD_REQUEST:
            {
              runtime->handle_index_space_child_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_CHILD_RESPONSE:
            {
              runtime->handle_index_space_child_response(derez);
              break;
            }
          case SEND_INDEX_SPACE_COLORS_REQUEST:
            {
              runtime->handle_index_space_colors_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_COLORS_RESPONSE:
            {
              runtime->handle_index_space_colors_response(derez);
              break;
            }
          case SEND_INDEX_SPACE_GENERATE_COLOR_REQUEST:
            {
              runtime->handle_index_space_generate_color_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_GENERATE_COLOR_RESPONSE:
            {
              runtime->handle_index_space_generate_color_response(derez);
              break;
            }
          case SEND_INDEX_SPACE_RELEASE_COLOR:
            {
              runtime->handle_index_space_release_color(derez);
              break;
            }
          case SEND_INDEX_PARTITION_NOTIFICATION:
            {
              runtime->handle_index_partition_notification(derez);
              break;
            }
          case SEND_INDEX_PARTITION_REQUEST:
            {
              runtime->handle_index_partition_request(derez);
              break;
            }
          case SEND_INDEX_PARTITION_RESPONSE:
            {
              runtime->handle_index_partition_response(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_RETURN:
            {
              runtime->handle_index_partition_return(derez);
              break;
            }
          case SEND_INDEX_PARTITION_CHILD_REQUEST:
            {
              runtime->handle_index_partition_child_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_CHILD_RESPONSE:
            {
              runtime->handle_index_partition_child_response(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_CHILD_REPLICATION:
            {
              runtime->handle_index_partition_child_replication(derez);
              break;
            }
          case SEND_INDEX_PARTITION_DISJOINT_UPDATE:
            {
              runtime->handle_index_partition_disjoint_update(derez);
              break;
            }
          case SEND_INDEX_PARTITION_SHARD_RECTS_REQUEST:
            {
              runtime->handle_index_partition_shard_rects_request(derez);
              break;
            }
          case SEND_INDEX_PARTITION_SHARD_RECTS_RESPONSE:
            {
              runtime->handle_index_partition_shard_rects_response(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_REQUEST:
            {
              runtime->handle_index_partition_remote_interference_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_RESPONSE:
            {
              runtime->handle_index_partition_remote_interference_response(
                  derez);
              break;
            }
          case SEND_FIELD_SPACE_NODE:
            {
              runtime->handle_field_space_node(derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_REQUEST:
            {
              runtime->handle_field_space_request(derez);
              break;
            }
          case SEND_FIELD_SPACE_RETURN:
            {
              runtime->handle_field_space_return(derez);
              break;
            }
          case SEND_FIELD_SPACE_ALLOCATOR_REQUEST:
            {
              runtime->handle_field_space_allocator_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_ALLOCATOR_RESPONSE:
            {
              runtime->handle_field_space_allocator_response(derez);
              break;
            }
          case SEND_FIELD_SPACE_ALLOCATOR_INVALIDATION:
            {
              runtime->handle_field_space_allocator_invalidation(derez);
              break;
            }
          case SEND_FIELD_SPACE_ALLOCATOR_FLUSH:
            {
              runtime->handle_field_space_allocator_flush(derez);
              break;
            }
          case SEND_FIELD_SPACE_ALLOCATOR_FREE:
            {
              runtime->handle_field_space_allocator_free(
                  derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_INFOS_REQUEST:
            {
              runtime->handle_field_space_infos_request(derez);
              break;
            }
          case SEND_FIELD_SPACE_INFOS_RESPONSE:
            {
              runtime->handle_field_space_infos_response(derez);
              break;
            }
          case SEND_FIELD_ALLOC_REQUEST:
            {
              runtime->handle_field_alloc_request(derez);
              break;
            }
          case SEND_FIELD_SIZE_UPDATE:
            {
              runtime->handle_field_size_update(derez, remote_address_space);
              break;
            }
          case SEND_FIELD_FREE:
            {
              runtime->handle_field_free(derez, remote_address_space);
              break;
            }
          case SEND_FIELD_FREE_INDEXES:
            {
              runtime->handle_field_free_indexes(derez);
              break;
            }
          case SEND_FIELD_SPACE_LAYOUT_INVALIDATION:
            {
              runtime->handle_field_space_layout_invalidation(
                  derez, remote_address_space);
              break;
            }
          case SEND_LOCAL_FIELD_ALLOC_REQUEST:
            {
              runtime->handle_local_field_alloc_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LOCAL_FIELD_ALLOC_RESPONSE:
            {
              runtime->handle_local_field_alloc_response(derez);
              break;
            }
          case SEND_LOCAL_FIELD_FREE:
            {
              runtime->handle_local_field_free(derez);
              break;
            }
          case SEND_LOCAL_FIELD_UPDATE:
            {
              runtime->handle_local_field_update(derez);
              break;
            }
          case SEND_TOP_LEVEL_REGION_REQUEST:
            {
              runtime->handle_top_level_region_request(derez);
              break;
            }
          case SEND_TOP_LEVEL_REGION_RETURN:
            {
              runtime->handle_top_level_region_return(
                  derez, remote_address_space);
              break;
            }
          case INDEX_SPACE_DESTRUCTION_MESSAGE:
            {
              runtime->handle_index_space_destruction(
                  derez, remote_address_space);
              break;
            }
          case INDEX_PARTITION_DESTRUCTION_MESSAGE:
            {
              runtime->handle_index_partition_destruction(derez);
              break;
            }
          case FIELD_SPACE_DESTRUCTION_MESSAGE:
            {
              runtime->handle_field_space_destruction(derez);
              break;
            }
          case LOGICAL_REGION_DESTRUCTION_MESSAGE:
            {
              runtime->handle_logical_region_destruction(derez);
              break;
            }
          case INDIVIDUAL_REMOTE_FUTURE_SIZE:
            {
              runtime->handle_individual_remote_future_size(derez);
              break;
            }
          case INDIVIDUAL_REMOTE_OUTPUT_REGISTRATION:
            {
              runtime->handle_individual_remote_output_registration(derez);
              break;
            }
          case INDIVIDUAL_REMOTE_MAPPED:
            {
              runtime->handle_individual_remote_mapped(derez);
              break;
            }
          case INDIVIDUAL_REMOTE_COMPLETE:
            {
              runtime->handle_individual_remote_complete(derez);
              break;
            }
          case INDIVIDUAL_REMOTE_COMMIT:
            {
              runtime->handle_individual_remote_commit(derez);
              break;
            }
          case INDIVIDUAL_CONCURRENT_REQUEST:
            {
              runtime->handle_individual_concurrent_request(
                  derez, remote_address_space);
              break;
            }
          case INDIVIDUAL_CONCURRENT_RESPONSE:
            {
              runtime->handle_individual_concurrent_response(derez);
              break;
            }
          case SLICE_REMOTE_MAPPED:
            {
              runtime->handle_slice_remote_mapped(derez, remote_address_space);
              break;
            }
          case SLICE_REMOTE_COMPLETE:
            {
              runtime->handle_slice_remote_complete(derez);
              break;
            }
          case SLICE_REMOTE_COMMIT:
            {
              runtime->handle_slice_remote_commit(derez);
              break;
            }
          case SLICE_RENDEZVOUS_CONCURRENT_MAPPED:
            {
              runtime->handle_slice_rendezvous_concurrent_mapped(derez);
              break;
            }
          case SLICE_COLLECTIVE_ALLREDUCE_REQUEST:
            {
              runtime->handle_slice_collective_allreduce_request(
                  derez, remote_address_space);
              break;
            }
          case SLICE_COLLECTIVE_ALLREDUCE_RESPONSE:
            {
              runtime->handle_slice_collective_allreduce_response(derez);
              break;
            }
          case SLICE_CONCURRENT_ALLREDUCE_REQUEST:
            {
              runtime->handle_slice_concurrent_allreduce_request(
                  derez, remote_address_space);
              break;
            }
          case SLICE_CONCURRENT_ALLREDUCE_RESPONSE:
            {
              runtime->handle_slice_concurrent_allreduce_response(derez);
              break;
            }
          case SLICE_FIND_INTRA_DEP:
            {
              runtime->handle_slice_find_intra_dependence(derez);
              break;
            }
          case SLICE_REMOTE_COLLECTIVE_RENDEZVOUS:
            {
              runtime->handle_slice_remote_collective_rendezvous(
                  derez, remote_address_space);
              break;
            }
          case SLICE_REMOTE_VERSIONING_COLLECTIVE_RENDEZVOUS:
            {
              runtime->handle_slice_remote_collective_versioning_rendezvous(
                  derez);
              break;
            }
          case SLICE_REMOTE_OUTPUT_EXTENTS:
            {
              runtime->handle_slice_remote_output_extents(derez);
              break;
            }
          case SLICE_REMOTE_OUTPUT_REGISTRATION:
            {
              runtime->handle_slice_remote_output_registration(derez);
              break;
            }
          case DISTRIBUTED_REMOTE_REGISTRATION:
            {
              runtime->handle_did_remote_registration(
                  derez, remote_address_space);
              break;
            }
          case DISTRIBUTED_DOWNGRADE_REQUEST:
            {
              runtime->handle_did_downgrade_request(
                  derez, remote_address_space);
              break;
            }
          case DISTRIBUTED_DOWNGRADE_RESPONSE:
            {
              runtime->handle_did_downgrade_response(derez);
              break;
            }
          case DISTRIBUTED_DOWNGRADE_SUCCESS:
            {
              runtime->handle_did_downgrade_success(derez);
              break;
            }
          case DISTRIBUTED_DOWNGRADE_UPDATE:
            {
              runtime->handle_did_downgrade_update(derez);
              break;
            }
          case DISTRIBUTED_DOWNGRADE_RESTART:
            {
              runtime->handle_did_downgrade_restart(
                  derez, remote_address_space);
              break;
            }
          case DISTRIBUTED_GLOBAL_ACQUIRE_REQUEST:
            {
              runtime->handle_did_global_acquire_request(derez);
              break;
            }
          case DISTRIBUTED_GLOBAL_ACQUIRE_RESPONSE:
            {
              runtime->handle_did_global_acquire_response(derez);
              break;
            }
          case DISTRIBUTED_VALID_ACQUIRE_REQUEST:
            {
              runtime->handle_did_valid_acquire_request(derez);
              break;
            }
          case DISTRIBUTED_VALID_ACQUIRE_RESPONSE:
            {
              runtime->handle_did_valid_acquire_response(derez);
              break;
            }
          case SEND_ATOMIC_RESERVATION_REQUEST:
            {
              runtime->handle_send_atomic_reservation_request(derez);
              break;
            }
          case SEND_ATOMIC_RESERVATION_RESPONSE:
            {
              runtime->handle_send_atomic_reservation_response(derez);
              break;
            }
          case SEND_PADDED_RESERVATION_REQUEST:
            {
              runtime->handle_send_padded_reservation_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_PADDED_RESERVATION_RESPONSE:
            {
              runtime->handle_send_padded_reservation_response(derez);
              break;
            }
          case SEND_CREATED_REGION_CONTEXTS:
            {
              runtime->handle_created_region_contexts(derez);
              break;
            }
          case SEND_MATERIALIZED_VIEW:
            {
              runtime->handle_send_materialized_view(derez);
              break;
            }
          case SEND_FILL_VIEW:
            {
              runtime->handle_send_fill_view(derez);
              break;
            }
          case SEND_FILL_VIEW_VALUE:
            {
              runtime->handle_send_fill_view_value(derez);
              break;
            }
          case SEND_PHI_VIEW:
            {
              runtime->handle_send_phi_view(derez);
              break;
            }
          case SEND_REDUCTION_VIEW:
            {
              runtime->handle_send_reduction_view(derez);
              break;
            }
          case SEND_REPLICATED_VIEW:
            {
              runtime->handle_send_replicated_view(derez);
              break;
            }
          case SEND_ALLREDUCE_VIEW:
            {
              runtime->handle_send_allreduce_view(derez);
              break;
            }
          case SEND_INSTANCE_MANAGER:
            {
              runtime->handle_send_instance_manager(
                  derez, remote_address_space);
              break;
            }
          case SEND_MANAGER_UPDATE:
            {
              runtime->handle_send_manager_update(derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_DISTRIBUTE_FILL:
            {
              runtime->handle_collective_distribute_fill(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_DISTRIBUTE_POINT:
            {
              runtime->handle_collective_distribute_point(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_DISTRIBUTE_POINTWISE:
            {
              runtime->handle_collective_distribute_pointwise(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_DISTRIBUTE_REDUCTION:
            {
              runtime->handle_collective_distribute_reduction(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_DISTRIBUTE_BROADCAST:
            {
              runtime->handle_collective_distribute_broadcast(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_DISTRIBUTE_REDUCECAST:
            {
              runtime->handle_collective_distribute_reducecast(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_DISTRIBUTE_HOURGLASS:
            {
              runtime->handle_collective_distribute_hourglass(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_DISTRIBUTE_ALLREDUCE:
            {
              runtime->handle_collective_distribute_allreduce(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_HAMMER_REDUCTION:
            {
              runtime->handle_collective_hammer_reduction(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_FUSE_GATHER:
            {
              runtime->handle_collective_fuse_gather(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_USER_REQUEST:
            {
              runtime->handle_collective_user_request(derez);
              break;
            }
          case SEND_COLLECTIVE_USER_RESPONSE:
            {
              runtime->handle_collective_user_response(derez);
              break;
            }
          case SEND_COLLECTIVE_REGISTER_USER:
            {
              runtime->handle_collective_user_registration(derez);
              break;
            }
          case SEND_COLLECTIVE_REMOTE_INSTANCES_REQUEST:
            {
              runtime->handle_collective_remote_instances_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_REMOTE_INSTANCES_RESPONSE:
            {
              runtime->handle_collective_remote_instances_response(
                  derez, remote_address_space);
              break;
            }
          case SEND_COLLECTIVE_NEAREST_INSTANCES_REQUEST:
            {
              runtime->handle_collective_nearest_instances_request(derez);
              break;
            }
          case SEND_COLLECTIVE_NEAREST_INSTANCES_RESPONSE:
            {
              runtime->handle_collective_nearest_instances_response(derez);
              break;
            }
          case SEND_COLLECTIVE_REMOTE_REGISTRATION:
            {
              runtime->handle_collective_remote_registration(derez);
              break;
            }
          case SEND_COLLECTIVE_FINALIZE_MAPPING:
            {
              runtime->handle_collective_finalize_mapping(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_CREATION:
            {
              runtime->handle_collective_view_creation(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_DELETION:
            {
              runtime->handle_collective_view_deletion(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_RELEASE:
            {
              runtime->handle_collective_view_release(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_NOTIFICATION:
            {
              runtime->handle_collective_view_notification(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_MAKE_VALID:
            {
              runtime->handle_collective_view_make_valid(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_MAKE_INVALID:
            {
              runtime->handle_collective_view_make_invalid(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_INVALIDATE_REQUEST:
            {
              runtime->handle_collective_view_invalidate_request(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_INVALIDATE_RESPONSE:
            {
              runtime->handle_collective_view_invalidate_response(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_ADD_REMOTE_REFERENCE:
            {
              runtime->handle_collective_view_add_remote_reference(derez);
              break;
            }
          case SEND_COLLECTIVE_VIEW_REMOVE_REMOTE_REFERENCE:
            {
              runtime->handle_collective_view_remove_remote_reference(derez);
              break;
            }
          case SEND_CREATE_TOP_VIEW_REQUEST:
            {
              runtime->handle_create_top_view_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_CREATE_TOP_VIEW_RESPONSE:
            {
              runtime->handle_create_top_view_response(derez);
              break;
            }
          case SEND_VIEW_REQUEST:
            {
              runtime->handle_view_request(derez);
              break;
            }
          case SEND_VIEW_REGISTER_USER:
            {
              runtime->handle_view_register_user(derez, remote_address_space);
              break;
            }
          case SEND_VIEW_FIND_COPY_PRE_REQUEST:
            {
              runtime->handle_view_copy_pre_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_VIEW_ADD_COPY_USER:
            {
              runtime->handle_view_add_copy_user(derez, remote_address_space);
              break;
            }
          case SEND_VIEW_FIND_LAST_USERS_REQUEST:
            {
              runtime->handle_view_find_last_users_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_VIEW_FIND_LAST_USERS_RESPONSE:
            {
              runtime->handle_view_find_last_users_response(derez);
              break;
            }
          case SEND_MANAGER_REQUEST:
            {
              runtime->handle_manager_request(derez);
              break;
            }
          case SEND_FUTURE_RESULT:
            {
              runtime->handle_future_result(derez);
              break;
            }
          case SEND_FUTURE_RESULT_SIZE:
            {
              runtime->handle_future_result_size(derez, remote_address_space);
              break;
            }
          case SEND_FUTURE_SUBSCRIPTION:
            {
              runtime->handle_future_subscription(derez, remote_address_space);
              break;
            }
          case SEND_FUTURE_CREATE_INSTANCE_REQUEST:
            {
              runtime->handle_future_create_instance_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_FUTURE_CREATE_INSTANCE_RESPONSE:
            {
              runtime->handle_future_create_instance_response(derez);
              break;
            }
          case SEND_FUTURE_MAP_REQUEST:
            {
              runtime->handle_future_map_future_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_FUTURE_MAP_RESPONSE:
            {
              runtime->handle_future_map_future_response(derez);
              break;
            }
          case SEND_FUTURE_MAP_POINTWISE:
            {
              runtime->handle_future_map_find_pointwise(derez);
              break;
            }
          case SEND_REPL_COMPUTE_EQUIVALENCE_SETS:
            {
              runtime->handle_control_replicate_compute_equivalence_sets(derez);
              break;
            }
          case SEND_REPL_OUTPUT_EQUIVALENCE_SET:
            {
              runtime->handle_control_replicate_output_equivalence_set(derez);
              break;
            }
          case SEND_REPL_REFINE_EQUIVALENCE_SETS:
            {
              runtime->handle_control_replicate_refine_equivalence_sets(derez);
              break;
            }
          case SEND_REPL_EQUIVALENCE_SET_NOTIFICATION:
            {
              runtime->handle_control_replicate_equivalence_set_notification(
                  derez);
              break;
            }
          case SEND_REPL_BROADCAST_UPDATE:
            {
              runtime->handle_control_replicate_broadcast_update(derez);
              break;
            }
          case SEND_REPL_CREATED_REGIONS:
            {
              runtime->handle_control_replicate_created_regions(derez);
              break;
            }
          case SEND_REPL_TRACE_EVENT_REQUEST:
            {
              runtime->handle_control_replicate_trace_event_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_REPL_TRACE_EVENT_RESPONSE:
            {
              runtime->handle_control_replicate_trace_event_response(derez);
              break;
            }
          case SEND_REPL_TRACE_EVENT_TRIGGER:
            {
              runtime->handle_control_replicate_trace_event_trigger(derez);
              break;
            }
          case SEND_REPL_TRACE_FRONTIER_REQUEST:
            {
              runtime->handle_control_replicate_trace_frontier_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_REPL_TRACE_FRONTIER_RESPONSE:
            {
              runtime->handle_control_replicate_trace_frontier_response(derez);
              break;
            }
          case SEND_REPL_TRACE_UPDATE:
            {
              runtime->handle_control_replicate_trace_update(
                  derez, remote_address_space);
              break;
            }
          case SEND_REPL_FIND_TRACE_SETS:
            {
              runtime->handle_control_replicate_find_trace_local_sets(
                  derez, remote_address_space);
              break;
            }
          case SEND_REPL_IMPLICIT_RENDEZVOUS:
            {
              runtime->handle_control_replicate_implicit_rendezvous(derez);
              break;
            }
          case SEND_REPL_FIND_COLLECTIVE_VIEW:
            {
              runtime->handle_control_replicate_find_collective_view(derez);
              break;
            }
          case SEND_REPL_POINTWISE_DEPENDENCE:
            {
              runtime->handle_control_replicate_pointwise_dependence(derez);
              break;
            }
          case SEND_MAPPER_MESSAGE:
            {
              runtime->handle_mapper_message(derez);
              break;
            }
          case SEND_MAPPER_BROADCAST:
            {
              runtime->handle_mapper_broadcast(derez);
              break;
            }
          case SEND_TASK_IMPL_SEMANTIC_REQ:
            {
              runtime->handle_task_impl_semantic_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_SEMANTIC_REQ:
            {
              runtime->handle_index_space_semantic_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_SEMANTIC_REQ:
            {
              runtime->handle_index_partition_semantic_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_SEMANTIC_REQ:
            {
              runtime->handle_field_space_semantic_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SEMANTIC_REQ:
            {
              runtime->handle_field_semantic_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LOGICAL_REGION_SEMANTIC_REQ:
            {
              runtime->handle_logical_region_semantic_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LOGICAL_PARTITION_SEMANTIC_REQ:
            {
              runtime->handle_logical_partition_semantic_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_TASK_IMPL_SEMANTIC_INFO:
            {
              runtime->handle_task_impl_semantic_info(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_SEMANTIC_INFO:
            {
              runtime->handle_index_space_semantic_info(
                  derez, remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_SEMANTIC_INFO:
            {
              runtime->handle_index_partition_semantic_info(
                  derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_SEMANTIC_INFO:
            {
              runtime->handle_field_space_semantic_info(
                  derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SEMANTIC_INFO:
            {
              runtime->handle_field_semantic_info(derez, remote_address_space);
              break;
            }
          case SEND_LOGICAL_REGION_SEMANTIC_INFO:
            {
              runtime->handle_logical_region_semantic_info(
                  derez, remote_address_space);
              break;
            }
          case SEND_LOGICAL_PARTITION_SEMANTIC_INFO:
            {
              runtime->handle_logical_partition_semantic_info(
                  derez, remote_address_space);
              break;
            }
          case SEND_REMOTE_CONTEXT_REQUEST:
            {
              runtime->handle_remote_context_request(derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_RESPONSE:
            {
              runtime->handle_remote_context_response(derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_PHYSICAL_REQUEST:
            {
              runtime->handle_remote_context_physical_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_REMOTE_CONTEXT_PHYSICAL_RESPONSE:
            {
              runtime->handle_remote_context_physical_response(derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_REQUEST:
            {
              runtime->handle_remote_context_find_collective_view_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_RESPONSE:
            {
              runtime->handle_remote_context_find_collective_view_response(
                  derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_REFINE_EQUIVALENCE_SETS:
            {
              runtime->handle_remote_context_refine_equivalence_sets(derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_POINTWISE_DEPENDENCE:
            {
              runtime->handle_remote_context_pointwise_dependence(derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_FIND_TRACE_LOCAL_SETS_REQUEST:
            {
              runtime->handle_remote_context_find_trace_local_sets_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_REMOTE_CONTEXT_FIND_TRACE_LOCAL_SETS_RESPONSE:
            {
              runtime->handle_remote_context_find_trace_local_sets_response(
                  derez);
              break;
            }
          case SEND_COMPUTE_EQUIVALENCE_SETS_REQUEST:
            {
              runtime->handle_compute_equivalence_sets_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_COMPUTE_EQUIVALENCE_SETS_RESPONSE:
            {
              runtime->handle_compute_equivalence_sets_response(derez);
              break;
            }
          case SEND_COMPUTE_EQUIVALENCE_SETS_PENDING:
            {
              runtime->handle_compute_equivalence_sets_pending(derez);
              break;
            }
          case SEND_OUTPUT_EQUIVALENCE_SET_REQUEST:
            {
              runtime->handle_output_equivalence_set_request(derez);
              break;
            }
          case SEND_OUTPUT_EQUIVALENCE_SET_RESPONSE:
            {
              runtime->handle_output_equivalence_set_response(
                  derez, remote_address_space);
              break;
            }
          case SEND_CANCEL_EQUIVALENCE_SETS_SUBSCRIPTION:
            {
              runtime->handle_cancel_equivalence_sets_subscription(
                  derez, remote_address_space);
              break;
            }
          case SEND_INVALIDATE_EQUIVALENCE_SETS_SUBSCRIPTION:
            {
              runtime->handle_invalidate_equivalence_sets_subscription(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_CREATION:
            {
              runtime->handle_equivalence_set_creation(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_REUSE:
            {
              runtime->handle_equivalence_set_reuse(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_REQUEST:
            {
              runtime->handle_equivalence_set_request(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_RESPONSE:
            {
              runtime->handle_equivalence_set_response(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_REPLICATION_REQUEST:
            {
              runtime->handle_equivalence_set_replication_request(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_REPLICATION_RESPONSE:
            {
              runtime->handle_equivalence_set_replication_response(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_MIGRATION:
            {
              runtime->handle_equivalence_set_migration(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_OWNER_UPDATE:
            {
              runtime->handle_equivalence_set_owner_update(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_CLONE_REQUEST:
            {
              runtime->handle_equivalence_set_clone_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_CLONE_RESPONSE:
            {
              runtime->handle_equivalence_set_clone_response(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_CAPTURE_REQUEST:
            {
              runtime->handle_equivalence_set_capture_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_CAPTURE_RESPONSE:
            {
              runtime->handle_equivalence_set_capture_response(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INSTANCES:
            {
              runtime->handle_equivalence_set_remote_request_instances(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INVALID:
            {
              runtime->handle_equivalence_set_remote_request_invalid(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_ANTIVALID:
            {
              runtime->handle_equivalence_set_remote_request_antivalid(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_UPDATES:
            {
              runtime->handle_equivalence_set_remote_updates(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_ACQUIRES:
            {
              runtime->handle_equivalence_set_remote_acquires(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_RELEASES:
            {
              runtime->handle_equivalence_set_remote_releases(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_COPIES_ACROSS:
            {
              runtime->handle_equivalence_set_remote_copies_across(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_OVERWRITES:
            {
              runtime->handle_equivalence_set_remote_overwrites(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_FILTERS:
            {
              runtime->handle_equivalence_set_remote_filters(
                  derez, remote_address_space);
              break;
            }
          case SEND_EQUIVALENCE_SET_REMOTE_INSTANCES:
            {
              runtime->handle_equivalence_set_remote_instances(derez);
              break;
            }
          case SEND_EQUIVALENCE_SET_FILTER_INVALIDATIONS:
            {
              runtime->handle_equivalence_set_filter_invalidations(derez);
              break;
            }
          case SEND_INSTANCE_REQUEST:
            {
              runtime->handle_instance_request(derez, remote_address_space);
              break;
            }
          case SEND_INSTANCE_RESPONSE:
            {
              runtime->handle_instance_response(derez, remote_address_space);
              break;
            }
          case SEND_EXTERNAL_CREATE_REQUEST:
            {
              runtime->handle_external_create_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_EXTERNAL_CREATE_RESPONSE:
            {
              runtime->handle_external_create_response(derez);
              break;
            }
          case SEND_EXTERNAL_ATTACH:
            {
              runtime->handle_external_attach(derez);
              break;
            }
          case SEND_EXTERNAL_DETACH:
            {
              runtime->handle_external_detach(derez);
              break;
            }
          case SEND_GC_PRIORITY_UPDATE:
            {
              runtime->handle_gc_priority_update(derez, remote_address_space);
              break;
            }
          case SEND_GC_REQUEST:
            {
              runtime->handle_gc_request(derez, remote_address_space);
              break;
            }
          case SEND_GC_RESPONSE:
            {
              runtime->handle_gc_response(derez);
              break;
            }
          case SEND_GC_ACQUIRE:
            {
              runtime->handle_gc_acquire(derez);
              break;
            }
          case SEND_GC_FAILED:
            {
              runtime->handle_gc_failed(derez);
              break;
            }
          case SEND_GC_MISMATCH:
            {
              runtime->handle_gc_mismatch(derez);
              break;
            }
          case SEND_GC_NOTIFY:
            {
              runtime->handle_gc_notify(derez);
              break;
            }
          case SEND_GC_DEBUG_REQUEST:
            {
              runtime->handle_gc_debug_request(derez, remote_address_space);
              break;
            }
          case SEND_GC_DEBUG_RESPONSE:
            {
              runtime->handle_gc_debug_response(derez);
              break;
            }
          case SEND_GC_RECORD_EVENT:
            {
              runtime->handle_gc_record_event(derez);
              break;
            }
          case SEND_ACQUIRE_REQUEST:
            {
              runtime->handle_acquire_request(derez, remote_address_space);
              break;
            }
          case SEND_ACQUIRE_RESPONSE:
            {
              runtime->handle_acquire_response(derez, remote_address_space);
              break;
            }
          case SEND_VARIANT_BROADCAST:
            {
              runtime->handle_variant_broadcast(derez);
              break;
            }
          case SEND_CONSTRAINT_REQUEST:
            {
              runtime->handle_constraint_request(derez, remote_address_space);
              break;
            }
          case SEND_CONSTRAINT_RESPONSE:
            {
              runtime->handle_constraint_response(derez, remote_address_space);
              break;
            }
          case SEND_CONSTRAINT_RELEASE:
            {
              runtime->handle_constraint_release(derez);
              break;
            }
          case SEND_TOP_LEVEL_TASK_COMPLETE:
            {
              runtime->handle_top_level_task_complete(derez);
              break;
            }
          case SEND_MPI_RANK_EXCHANGE:
            {
              runtime->handle_mpi_rank_exchange(derez);
              break;
            }
          case SEND_REPLICATE_DISTRIBUTION:
            {
              runtime->handle_replicate_distribution(derez);
              break;
            }
          case SEND_REPLICATE_COLLECTIVE_VERSIONING:
            {
              runtime->handle_replicate_collective_versioning(derez);
              break;
            }
          case SEND_REPLICATE_COLLECTIVE_MAPPING:
            {
              runtime->handle_replicate_collective_mapping(derez);
              break;
            }
          case SEND_REPLICATE_VIRTUAL_RENDEZVOUS:
            {
              runtime->handle_replicate_virtual_rendezvous(derez);
              break;
            }
          case SEND_REPLICATE_STARTUP_COMPLETE:
            {
              runtime->handle_replicate_startup_complete(derez);
              break;
            }
          case SEND_REPLICATE_POST_MAPPED:
            {
              runtime->handle_replicate_post_mapped(derez);
              break;
            }
          case SEND_REPLICATE_TRIGGER_COMPLETE:
            {
              runtime->handle_replicate_trigger_complete(derez);
              break;
            }
          case SEND_REPLICATE_TRIGGER_COMMIT:
            {
              runtime->handle_replicate_trigger_commit(derez);
              break;
            }
          case SEND_CONTROL_REPLICATE_RENDEZVOUS_MESSAGE:
            {
              runtime->handle_control_replicate_rendezvous_message(derez);
              break;
            }
          case SEND_LIBRARY_MAPPER_REQUEST:
            {
              runtime->handle_library_mapper_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LIBRARY_MAPPER_RESPONSE:
            {
              runtime->handle_library_mapper_response(derez);
              break;
            }
          case SEND_LIBRARY_TRACE_REQUEST:
            {
              runtime->handle_library_trace_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LIBRARY_TRACE_RESPONSE:
            {
              runtime->handle_library_trace_response(derez);
              break;
            }
          case SEND_LIBRARY_PROJECTION_REQUEST:
            {
              runtime->handle_library_projection_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LIBRARY_PROJECTION_RESPONSE:
            {
              runtime->handle_library_projection_response(derez);
              break;
            }
          case SEND_LIBRARY_SHARDING_REQUEST:
            {
              runtime->handle_library_sharding_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LIBRARY_SHARDING_RESPONSE:
            {
              runtime->handle_library_sharding_response(derez);
              break;
            }
          case SEND_LIBRARY_CONCURRENT_REQUEST:
            {
              runtime->handle_library_concurrent_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LIBRARY_CONCURRENT_RESPONSE:
            {
              runtime->handle_library_concurrent_response(derez);
              break;
            }
          case SEND_LIBRARY_TASK_REQUEST:
            {
              runtime->handle_library_task_request(derez, remote_address_space);
              break;
            }
          case SEND_LIBRARY_TASK_RESPONSE:
            {
              runtime->handle_library_task_response(derez);
              break;
            }
          case SEND_LIBRARY_REDOP_REQUEST:
            {
              runtime->handle_library_redop_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LIBRARY_REDOP_RESPONSE:
            {
              runtime->handle_library_redop_response(derez);
              break;
            }
          case SEND_LIBRARY_SERDEZ_REQUEST:
            {
              runtime->handle_library_serdez_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_LIBRARY_SERDEZ_RESPONSE:
            {
              runtime->handle_library_serdez_response(derez);
              break;
            }
          case SEND_REMOTE_OP_REPORT_UNINIT:
            {
              runtime->handle_remote_op_report_uninitialized(derez);
              break;
            }
          case SEND_REMOTE_OP_PROFILING_COUNT_UPDATE:
            {
              runtime->handle_remote_op_profiling_count_update(derez);
              break;
            }
          case SEND_REMOTE_OP_COMPLETION_EFFECT:
            {
              runtime->handle_remote_op_completion_effect(derez);
              break;
            }
          case SEND_REMOTE_TRACE_UPDATE:
            {
              runtime->handle_remote_tracing_update(
                  derez, remote_address_space);
              break;
            }
          case SEND_REMOTE_TRACE_RESPONSE:
            {
              runtime->handle_remote_tracing_response(derez);
              break;
            }
          case SEND_FREE_EXTERNAL_ALLOCATION:
            {
              runtime->handle_free_external_allocation(derez);
              break;
            }
          case SEND_NOTIFY_COLLECTED_INSTANCES:
            {
              runtime->handle_notify_collected_instances(derez);
              break;
            }
          case SEND_CREATE_MEMORY_POOL_REQUEST:
            {
              runtime->handle_create_memory_pool_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_CREATE_MEMORY_POOL_RESPONSE:
            {
              runtime->handle_create_memory_pool_response(derez);
              break;
            }
          case SEND_CREATE_FUTURE_INSTANCE_REQUEST:
            {
              runtime->handle_create_future_instance_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_CREATE_FUTURE_INSTANCE_RESPONSE:
            {
              runtime->handle_create_future_instance_response(derez);
              break;
            }
          case SEND_FREE_FUTURE_INSTANCE:
            {
              runtime->handle_free_future_instance(derez);
              break;
            }
          case SEND_REMOTE_DISTRIBUTED_ID_REQUEST:
            {
              runtime->handle_remote_distributed_id_request(
                  derez, remote_address_space);
              break;
            }
          case SEND_REMOTE_DISTRIBUTED_ID_RESPONSE:
            {
              runtime->handle_remote_distributed_id_response(derez);
              break;
            }
          case SEND_CONTROL_REPLICATION_FUTURE_ALLREDUCE:
          case SEND_CONTROL_REPLICATION_FUTURE_BROADCAST:
          case SEND_CONTROL_REPLICATION_FUTURE_REDUCTION:
          case SEND_CONTROL_REPLICATION_VALUE_ALLREDUCE:
          case SEND_CONTROL_REPLICATION_VALUE_BROADCAST:
          case SEND_CONTROL_REPLICATION_VALUE_EXCHANGE:
          case SEND_CONTROL_REPLICATION_BUFFER_BROADCAST:
          case SEND_CONTROL_REPLICATION_SHARD_SYNC_TREE:
          case SEND_CONTROL_REPLICATION_SHARD_EVENT_TREE:
          case SEND_CONTROL_REPLICATION_SINGLE_TASK_TREE:
          case SEND_CONTROL_REPLICATION_CROSS_PRODUCT_PARTITION:
          case SEND_CONTROL_REPLICATION_SHARDING_GATHER_COLLECTIVE:
          case SEND_CONTROL_REPLICATION_INDIRECT_COPY_EXCHANGE:
          case SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_EXCHANGE:
          case SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_GATHER:
          case SEND_CONTROL_REPLICATION_DEPPART_RESULT_SCATTER:
          case SEND_CONTROL_REPLICATION_BUFFER_EXCHANGE:
          case SEND_CONTROL_REPLICATION_FUTURE_NAME_EXCHANGE:
          case SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_BROADCAST:
          case SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_EXCHANGE:
          case SEND_CONTROL_REPLICATION_MUST_EPOCH_DEPENDENCE_EXCHANGE:
          case SEND_CONTROL_REPLICATION_MUST_EPOCH_COMPLETION_EXCHANGE:
          case SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_MAPPING:
          case SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_SOURCES:
          case SEND_CONTROL_REPLICATION_TEMPLATE_INDEX_EXCHANGE:
          case SEND_CONTROL_REPLICATION_UNORDERED_EXCHANGE:
          case SEND_CONTROL_REPLICATION_CONSENSUS_MATCH:
          case SEND_CONTROL_REPLICATION_VERIFY_CONTROL_REPLICATION_EXCHANGE:
          case SEND_CONTROL_REPLICATION_OUTPUT_SIZE_EXCHANGE:
          case SEND_CONTROL_REPLICATION_INDEX_ATTACH_LAUNCH_SPACE:
          case SEND_CONTROL_REPLICATION_INDEX_ATTACH_UPPER_BOUND:
          case SEND_CONTROL_REPLICATION_INDEX_ATTACH_EXCHANGE:
          case SEND_CONTROL_REPLICATION_SHARD_PARTICIPANTS_EXCHANGE:
          case SEND_CONTROL_REPLICATION_IMPLICIT_SHARDING_FUNCTOR:
          case SEND_CONTROL_REPLICATION_CREATE_FILL_VIEW:
          case SEND_CONTROL_REPLICATION_VERSIONING_RENDEZVOUS:
          case SEND_CONTROL_REPLICATION_VIEW_RENDEZVOUS:
          case SEND_CONTROL_REPLICATION_CONCURRENT_MAPPING_RENDEZVOUS:
          case SEND_CONTROL_REPLICATION_CONCURRENT_ALLREDUCE:
          case SEND_CONTROL_REPLICATION_PROJECTION_TREE_EXCHANGE:
          case SEND_CONTROL_REPLICATION_TIMEOUT_MATCH_EXCHANGE:
          case SEND_CONTROL_REPLICATION_MASK_EXCHANGE:
          case SEND_CONTROL_REPLICATION_PREDICATE_EXCHANGE:
          case SEND_CONTROL_REPLICATION_CROSS_PRODUCT_EXCHANGE:
          case SEND_CONTROL_REPLICATION_TRACING_SET_DEDUPLICATION:
          case SEND_CONTROL_REPLICATION_POINTWISE_ALLREDUCE:
          case SEND_CONTROL_REPLICATION_INTERFERING_POINT_EXCHANGE:
          case SEND_CONTROL_REPLICATION_SLOW_BARRIER:
            {
              ShardManager::process_collective_message(derez);
              break;
            }
          case SEND_PROFILER_EVENT_TRIGGER:
            {
              implicit_profiler->process_event_trigger(derez);
              break;
            }
          case SEND_PROFILER_EVENT_POISON:
            {
              implicit_profiler->process_event_poison(derez);
              break;
            }
          case SEND_SHUTDOWN_NOTIFICATION:
            {
              runtime->handle_shutdown_notification(
                  derez, remote_address_space);
              break;
            }
          case SEND_SHUTDOWN_RESPONSE:
            {
              runtime->handle_shutdown_response(derez);
              break;
            }
          default:
            std::abort();  // should never get here
        }
        // Update the args and arglen
        args += message_size;
        arglen -= message_size;
      }
      legion_assert(arglen == 0);  // make sure we processed everything
    }
#endif

    /////////////////////////////////////////////////////////////
    // Message Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MessageManager::MessageManager(
        AddressSpaceID remote, size_t max_message_size,
        const Processor remote_util_group)
      : channels((VirtualChannel*)malloc(
            MAX_NUM_VIRTUAL_CHANNELS * sizeof(VirtualChannel))),
        remote_address_space(remote), target(remote_util_group)
    //--------------------------------------------------------------------------
    {
      legion_assert(remote != runtime->address_space);
      const bool has_profiler = (runtime->profiler != nullptr);
      // Initialize our virtual channels
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
      {
        new (channels + idx) VirtualChannel(
            (VirtualChannelKind)idx, runtime->address_space, max_message_size,
            has_profiler);
      }
    }

    //--------------------------------------------------------------------------
    MessageManager::~MessageManager(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
        channels[idx].~VirtualChannel();
      free(channels);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_message(
        MessageKind kind, VirtualChannelKind channel, const Serializer& rez,
        bool response, RtEvent flush_precondition)
    //--------------------------------------------------------------------------
    {
      channels[channel].send_message(
          kind, rez, flush_precondition, target, response);
    }

    //--------------------------------------------------------------------------
    VirtualChannel& MessageManager::find_channel(VirtualChannelKind vc)
    //--------------------------------------------------------------------------
    {
      legion_assert(vc < MAX_NUM_VIRTUAL_CHANNELS);
      return channels[vc];
    }

    //--------------------------------------------------------------------------
    void MessageManager::confirm_shutdown(
        ShutdownManager* shutdown_manager, bool phase_one)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
        channels[idx].confirm_shutdown(
            shutdown_manager, phase_one, target,
            (idx == PROFILING_VIRTUAL_CHANNEL));
    }

    //--------------------------------------------------------------------------
    /*static*/ void MessageManager::register_handlers(void)
    //--------------------------------------------------------------------------
    {
#define REGISTER_HANDLERS(kind, type, name, response)    \
  legion_assert(message_handler_table[kind] == nullptr); \
  message_handler_table[kind] = type::handle;
      LEGION_ACTIVE_MESSAGES(REGISTER_HANDLERS)
#undef REGISTER_HANDLERS
#define REGISTER_SHARD_COLLECTIVES(kind, name)           \
  legion_assert(message_handler_table[kind] == nullptr); \
  message_handler_table[kind] = ShardCollectiveMessage::handle;
      LEGION_SHARD_COLLECTIVE_ACTIVE_MESSAGES(REGISTER_SHARD_COLLECTIVES)
#undef REGISTER_SHARD_COLLECTIVES
    }

  }  // namespace Internal
}  // namespace Legion
