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

#include "legion/managers/shutdown.h"
#include "legion/kernel/runtime.h"
#include "legion/api/future_impl.h"
#include "legion/managers/memory.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Shutdown Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShutdownManager::ShutdownManager(ShutdownPhase p,
                                     AddressSpaceID s, unsigned r, 
                                     ShutdownManager *own)
      : phase(p), source(s), radix(r), owner(own),
        needed_responses(0), return_code(runtime->return_code), result(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShutdownManager::~ShutdownManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool ShutdownManager::attempt_shutdown(void)
    //--------------------------------------------------------------------------
    {
      // Do the broadcast tree to the other nodes
      // Figure out who we have to send messages to
      std::vector<AddressSpaceID> targets;
      const AddressSpaceID local_space = runtime->address_space;
      const AddressSpaceID start = local_space * radix + 1;
      for (unsigned idx = 0; idx < radix; idx++)
      {
        AddressSpaceID next = start+idx;
        if (next < runtime->total_address_spaces)
          targets.push_back(next);
        else
          break;
      }
      
      if (!targets.empty())
      {
        // Set the number of needed_responses
        needed_responses = targets.size();
        Serializer rez;
        rez.serialize(this);
        rez.serialize(phase);
        for (std::vector<AddressSpaceID>::const_iterator it = 
              targets.begin(); it != targets.end(); it++)
          runtime->send_shutdown_notification(*it, rez); 
        return false;
      }
      else // no messages means we can finalize right now
      {
        finalize();
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool ShutdownManager::handle_response(int code, bool success,
                                          const std::set<RtEvent> &to_add)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      {
        AutoLock s_lock(shutdown_lock);
        if ((return_code == 0) && (code != 0))
          return_code = code;
        if (result && !success)
          result = false;
        wait_for.insert(to_add.begin(), to_add.end()); 
#ifdef DEBUG_LEGION
        assert(needed_responses > 0);
#endif
        needed_responses--;
        done = (needed_responses == 0);
      }
      if (done)
      {
        finalize();
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::finalize(void)
    //--------------------------------------------------------------------------
    {
      // Do our local check
      runtime->confirm_runtime_shutdown(this, 
          (phase == CHECK_TERMINATION) || (phase == CHECK_SHUTDOWN));
#ifdef DEBUG_SHUTDOWN_HANG
      if (!result)
      {
        LG_TASK_DESCRIPTIONS(task_descs);
        // Only need to see tasks less than this 
        for (unsigned idx = 0; idx < LG_BEGIN_SHUTDOWN_TASK_IDS; idx++)
        {
          if (runtime->outstanding_counts[idx].load() == 0)
            continue;
          log_shutdown.info("Meta-Task %s: %d outstanding",
                task_descs[idx], runtime->outstanding_counts[idx].load());
        }
      }
#endif
      if (result && (runtime->address_space == source))
      {
        log_shutdown.info("SHUTDOWN PHASE %d SUCCESS!", phase);
        if (phase != CONFIRM_SHUTDOWN)
        {
          if (phase == CONFIRM_TERMINATION)
            runtime->prepare_runtime_shutdown();
          // Do the next phase
          runtime->initiate_runtime_shutdown(source, (ShutdownPhase)(phase+1));
        }
        else
        {
          log_shutdown.info("SHUTDOWN SUCCEEDED!");
          std::vector<RtEvent> shutdown_events;
          Realm::ProfilingRequestSet empty_requests;
          const Processor utility_group = runtime->find_utility_group();
          shutdown_events.push_back(RtEvent(utility_group.spawn(
                  LG_SHUTDOWN_TASK_ID, nullptr, 0, empty_requests)));
          // One last really crazy precondition on shutdown, we actually need to
          // make sure that this task itself is done executing before trying to
          // shutdown so add our own completion event as a precondition
          shutdown_events.push_back(
              RtEvent(Processor::get_current_finish_event()));
          // Then tell Realm to shutdown when they are all done
          RealmRuntime realm = RealmRuntime::get_runtime();
          realm.shutdown(Runtime::merge_events(shutdown_events), return_code);
        }
      }
      else if (runtime->address_space != source)
      {
#ifdef DEBUG_LEGION
        assert(owner != nullptr);
#endif
        // Send the message back
        Serializer rez;
        rez.serialize(owner);
        rez.serialize(return_code);
        rez.serialize<bool>(result);
        rez.serialize<size_t>(wait_for.size());
        for (std::set<RtEvent>::const_iterator it = 
              wait_for.begin(); it != wait_for.end(); it++)
          rez.serialize(*it);
        runtime->send_shutdown_response(source, rez);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!result);
#endif
        log_shutdown.info("FAILED SHUTDOWN PHASE %d! Trying again...", phase);
        RtEvent precondition;
        if (!wait_for.empty())
          precondition = Runtime::merge_events(wait_for);
        // If we failed an even phase we go back to the one before it
        RetryShutdownArgs args(((phase % 2) == 0) ?
            (ShutdownPhase)(phase-1) : phase);
        runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY,
                                         precondition);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_remote_tracing_update(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RemoteTraceRecorder::handle_remote_update(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_remote_tracing_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RemoteTraceRecorder::handle_remote_response(derez);
    }
  
    //--------------------------------------------------------------------------
    void Runtime::handle_free_external_allocation(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FutureInstance::handle_free_external(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_notify_collected_instances(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      MemoryManager::handle_notify_collected_instances(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_create_memory_pool_request(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MemoryManager::handle_create_memory_pool_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_create_memory_pool_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      MemoryManager::handle_create_memory_pool_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_create_future_instance_request(Deserializer &derez,
                                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory memory;
      derez.deserialize(memory);
      std::atomic<FutureInstance*> *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);
      UniqueID uid;
      derez.deserialize(uid);
      TaskTreeCoordinates coordinates;
      coordinates.deserialize(derez);
      size_t size;
      derez.deserialize(size);
      RtEvent *remote_safe_for_unbounded_pools;
      derez.deserialize(remote_safe_for_unbounded_pools);

      MemoryManager *manager = find_memory_manager(memory);
      RtEvent safe_for_unbounded_pools;
      FutureInstance *result = manager->create_future_instance(uid,
          coordinates, size, (remote_safe_for_unbounded_pools == nullptr) ?
          nullptr : &safe_for_unbounded_pools);
      if ((result != nullptr) || ((remote_safe_for_unbounded_pools != nullptr) &&
            safe_for_unbounded_pools.exists()))
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          if (result != nullptr)
            result->pack_instance(rez, ApEvent::NO_AP_EVENT,
                true/*pack ownership*/, false/*allow by value*/);
          else
            FutureInstance::pack_null(rez);
          rez.serialize(remote_safe_for_unbounded_pools);
          if (remote_safe_for_unbounded_pools != nullptr)
            rez.serialize(safe_for_unbounded_pools);
          rez.serialize(done);
        }
        send_create_future_instance_response(source, rez);
        delete result;
      }
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_create_future_instance_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<FutureInstance*> *target;
      derez.deserialize(target);
      target->store(FutureInstance::unpack_instance(derez));
      RtEvent *safe_for_unbounded_pools;
      derez.deserialize(safe_for_unbounded_pools);
      if (safe_for_unbounded_pools != nullptr)
        derez.deserialize(*safe_for_unbounded_pools);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_free_future_instance(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory memory;
      derez.deserialize(memory);
      PhysicalInstance instance;
      derez.deserialize(instance);
      size_t size;
      derez.deserialize(size);
      RtEvent free_event;
      derez.deserialize(free_event);
      MemoryManager *manager = find_memory_manager(memory);
      manager->free_future_instance(instance, size, free_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShutdownManager::handle_shutdown_notification(
                   Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ShutdownManager *owner;
      derez.deserialize(owner);
      ShutdownPhase phase;
      derez.deserialize(phase);
      runtime->initiate_runtime_shutdown(source, phase, owner);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShutdownManager::handle_shutdown_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      ShutdownManager *shutdown_manager;
      derez.deserialize(shutdown_manager);
      int return_code;
      derez.deserialize(return_code);
      bool success;
      derez.deserialize(success);
      size_t num_events;
      derez.deserialize(num_events);
      std::set<RtEvent> wait_for;
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        RtEvent event;
        derez.deserialize(event);
        wait_for.insert(event);
      }
      if (shutdown_manager->handle_response(return_code, success, wait_for))
        delete shutdown_manager;
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_outstanding_tasks(void)
    //--------------------------------------------------------------------------
    {
      // Instant death
      result = false;
      log_shutdown.info("Outstanding tasks on node %d", runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_recent_message(void)
    //--------------------------------------------------------------------------
    {
      // Instant death
      result = false;
      log_shutdown.info("Outstanding message on node %d", 
                        runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_pending_message(RtEvent pending_event)
    //--------------------------------------------------------------------------
    {
      // Instant death
      result = false;
      wait_for.insert(pending_event);
      log_shutdown.info("Pending message on node %d", runtime->address_space);
    }

  } // namespace Internal
} // namespace Legion
