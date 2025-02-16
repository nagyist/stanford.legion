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

#include "legion/managers/processor.h"
#include "legion/contexts/inner.h"
#include "legion/kernel/runtime.h"
#include "legion/managers/mapper.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Processor Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(Processor proc, Processor::Kind kind,
                                       unsigned def_mappers,
                                       bool no_steal, bool replay)
      : local_proc(proc), proc_kind(kind), 
        stealing_disabled(no_steal), replay_execution(replay), 
        next_local_index(0), task_scheduler_enabled(false), 
        outstanding_task_scheduler(false),
        total_active_contexts(0), total_active_mappers(0), 
        total_progress_tasks(0), concurrent_lamport_clock(0),
        ready_concurrent_tasks(0), outstanding_concurrent_task(false)
    //--------------------------------------------------------------------------
    {
      context_states.resize(LEGION_DEFAULT_CONTEXTS);
      // Find our set of visible memories
      Machine::MemoryQuery vis_mems(runtime->machine);
      vis_mems.has_affinity_to(proc);
      vis_mems.has_capacity(1/*at least one byte*/);
      for (Machine::MemoryQuery::iterator it = vis_mems.begin();
            it != vis_mems.end(); it++)
      {
        Realm::Machine::AffinityDetails affinity;
        runtime->machine.has_affinity(proc, *it, &affinity);
        visible_memories[*it] = affinity.bandwidth;
      }
    }

    //--------------------------------------------------------------------------
    ProcessorManager::~ProcessorManager(void)
    //--------------------------------------------------------------------------
    {
      mapper_states.clear();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::prepare_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<MapperID,std::pair<MapperManager*,bool> >::iterator it = 
            mappers.begin(); it != mappers.end(); it++)
      {
        if (it->second.second)
          delete it->second.first;
      }
      mappers.clear();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_mapper(MapperID mid, MapperManager *m, 
                                      bool check, bool own, bool skip_replay)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are doing replay execution
      if (!skip_replay && replay_execution)
        return;
      log_run.spew("Adding mapper %d on processor " IDFMT "", 
                          mid, local_proc.id);
      if (check && (mid == 0))
        REPORT_LEGION_ERROR(ERROR_RESERVED_MAPPING_ID, 
                            "Invalid mapping ID. ID 0 is reserved.");
      if (check && !inside_registration_callback)
          REPORT_LEGION_WARNING(LEGION_WARNING_NON_CALLBACK_REGISTRATION,
            "Mapper %s (ID %d) was dynamically registered outside of a "
            "registration callback invocation. In the near future this will " 
            "become an error in order to support task subprocesses. Please "
            "use 'perform_registration_callback' to generate a callback "
            "where it will be safe to perform dynamic registrations.", 
            m->get_mapper_name(), mid)
      AutoLock m_lock(mapper_lock);
      std::map<MapperID,std::pair<MapperManager*,bool> >::iterator finder = 
        mappers.find(mid);
      if (finder != mappers.end())
      {
        if (finder->second.second)
          delete finder->second.first;
        finder->second = std::pair<MapperManager*,bool>(m, own);
      }
      else
      {
        mappers[mid] = std::pair<MapperManager*,bool>(m, own); 
        AutoLock q_lock(queue_lock);
        mapper_states[mid] = MapperState();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::replace_default_mapper(MapperManager *m, bool own)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are doing replay execution
      if (replay_execution)
        return;
      if (!inside_registration_callback)
          REPORT_LEGION_WARNING(LEGION_WARNING_NON_CALLBACK_REGISTRATION,
            "Replacing default mapper with %s was dynamically performed "
            "outside of a registration callback invocation. In the near "
            "future this will become an error in order to support task "
            "subprocesses. Please use 'perform_registration_callback' to "
            "generate a callback where it will be safe to perform dynamic " 
            "registrations.", m->get_mapper_name())
      AutoLock m_lock(mapper_lock);
      std::map<MapperID,std::pair<MapperManager*,bool> >::iterator finder = 
        mappers.find(0);
      if (finder != mappers.end())
      {
        if (finder->second.second)
          delete finder->second.first;
        finder->second = std::pair<MapperManager*,bool>(m, own);
      }
      else
      {
        mappers[0] = std::pair<MapperManager*,bool>(m, own);
        AutoLock q_lock(queue_lock);
        mapper_states[0] = MapperState();
      }
    }

    //--------------------------------------------------------------------------
    MapperManager* ProcessorManager::find_mapper(MapperID mid) const 
    //--------------------------------------------------------------------------
    {
      // Easy case if we are doing replay execution
      if (replay_execution)
      {
        std::map<MapperID,std::pair<MapperManager*,bool> >::const_iterator
          finder = mappers.find(0);
#ifdef DEBUG_LEGION
        assert(finder != mappers.end());
#endif
        return finder->second.first;
      }
      AutoLock m_lock(mapper_lock, 0/*mode*/, false/*exclusive*/);
      MapperManager *result = nullptr;
      // We've got the lock, so do the operation
      std::map<MapperID,std::pair<MapperManager*,bool> >::const_iterator
        finder = mappers.find(mid);
      if (finder != mappers.end())
        result = finder->second.first;
      return result;
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::has_non_default_mapper(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock, 0/*mode*/, false/*exclusive*/);
      for (std::map<MapperID,std::pair<MapperManager*,bool> >::const_iterator
            it = mappers.begin(); it != mappers.end(); it++)
        if (!it->second.first->is_default_mapper)
          return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_scheduling(void)
    //--------------------------------------------------------------------------
    {
      perform_mapping_operations(); 
      // Now re-take the lock and re-check the condition to see 
      // if the next scheduling task should be launched
      AutoLock q_lock(queue_lock);
#ifdef DEBUG_LEGION
      assert(outstanding_task_scheduler);
#endif
      // If the task scheduler is enabled launch ourselves again
      if (task_scheduler_enabled)
      {
        SchedulerArgs sched_args(local_proc);
        // If we need to recursively run the scheduler then we do so with
        // a lower priority than other meta-tasks to ensure that those other
        // meta tasks can continue to make forward progress and the scheduler
        // cannot starve other tasks.
        runtime->issue_runtime_meta_task(sched_args,
            LG_THROUGHPUT_WORK_PRIORITY);
      }
      else
        outstanding_task_scheduler = false;
    } 

    //--------------------------------------------------------------------------
    void ProcessorManager::launch_task_scheduler(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!outstanding_task_scheduler);
#endif
      outstanding_task_scheduler = true;
      SchedulerArgs sched_args(local_proc);
      // This is waking the scheduler up so give it higher priority in
      // order to ensure that we can get tasks mapped and running sooner
      runtime->issue_runtime_meta_task(sched_args, LG_LATENCY_WORK_PRIORITY);
    } 

    //--------------------------------------------------------------------------
    void ProcessorManager::notify_deferred_mapper(MapperID map_id,
                                                  RtEvent deferred_event)
    //--------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      MapperState &state = mapper_states[map_id];
      // Check to see if the deferral event matches the one that we have
      if (state.deferral_event == deferred_event)
      {
        // Now we can clear it
        state.deferral_event = RtEvent::NO_RT_EVENT;
        // And if we still have tasks, reactivate the mapper
        if (!state.ready_queue.empty())
          increment_active_mappers();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ProcessorManager::handle_defer_mapper(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMapperSchedulerArgs *dargs = 
        (const DeferMapperSchedulerArgs*)args; 
      dargs->proxy_this->notify_deferred_mapper(dargs->map_id, 
                                                dargs->deferral_event);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::activate_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      ContextID ctx_id = context->get_logical_tree_context();
      AutoLock q_lock(queue_lock); 
      ContextState &state = context_states[ctx_id];
#ifdef DEBUG_LEGION
      assert(!state.active);
#endif
      state.active = true;
      if (state.owned_tasks > 0)
        increment_active_contexts();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::deactivate_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      ContextID ctx_id = context->get_logical_tree_context();
      // We can do this without holding the lock because we know
      // the size of this vector is fixed
      AutoLock q_lock(queue_lock); 
      ContextState &state = context_states[ctx_id];
#ifdef DEBUG_LEGION
      assert(state.active);
#endif
      state.active = false;
      if (state.owned_tasks > 0)
        decrement_active_contexts();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::update_max_context_count(unsigned max_contexts)
    //--------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      context_states.resize(max_contexts);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_active_contexts(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      if (!task_scheduler_enabled && (total_active_contexts == 0) &&
          (total_progress_tasks == 0) && (total_active_mappers > 0))
      {
        task_scheduler_enabled = true;
        if (!outstanding_task_scheduler)
          launch_task_scheduler();
      }
      total_active_contexts++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_active_contexts(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
#ifdef DEBUG_LEGION
      assert(total_active_contexts > 0);
#endif
      total_active_contexts--;
      if ((total_active_contexts == 0) && (total_progress_tasks == 0))
        task_scheduler_enabled = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_active_mappers(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      if (!task_scheduler_enabled && (total_active_mappers == 0) &&
          ((total_active_contexts > 0) || (total_progress_tasks > 0)))
      {
        task_scheduler_enabled = true;
        if (!outstanding_task_scheduler)
          launch_task_scheduler();
      }
      total_active_mappers++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_active_mappers(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
#ifdef DEBUG_LEGION
      assert(total_active_mappers > 0);
#endif
      total_active_mappers--;
      if (total_active_mappers == 0)
        task_scheduler_enabled = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_progress_tasks(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      if (!task_scheduler_enabled && (total_active_contexts == 0) &&
          (total_progress_tasks == 0) && (total_active_mappers > 0))
      {
        task_scheduler_enabled = true;
        if (!outstanding_task_scheduler)
          launch_task_scheduler();
      }
      total_progress_tasks++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_progress_tasks(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(total_progress_tasks > 0);
#endif
      total_progress_tasks--;
      if ((total_active_contexts == 0) && (total_progress_tasks == 0))
        task_scheduler_enabled = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::process_steal_request(Processor thief,
                                           const std::vector<MapperID> &thieves)
    //--------------------------------------------------------------------------
    {
      log_run.spew("handling a steal request on processor " IDFMT " "
                         "from processor " IDFMT "", local_proc.id,thief.id);
      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal the task
      std::vector<SingleTask*> stolen;
      std::vector<MapperID> successful_thiefs;
      for (std::vector<MapperID>::const_iterator steal_it = thieves.begin();
            steal_it != thieves.end(); steal_it++)
      {
        const MapperID stealer = *steal_it;
        // Handle a race condition here where some processors can 
        // issue steal requests to another processor before the mappers 
        // have been initialized on that processor.  There's no 
        // correctness problem for ignoring a steal request so just do that.
        MapperManager *mapper = find_mapper(stealer);
        if (mapper == nullptr)
          continue;
        Mapper::StealRequestInput input;
        {
          // Wait until we can exclusive access to the ready queue
          RtEvent queue_copy_ready;
          // Pull out the current tasks for this mapping operation
          // Need to iterate until we get access to the queue
          do
          {
            if (queue_copy_ready.exists() && !queue_copy_ready.has_triggered())
            {
              queue_copy_ready.wait();
              queue_copy_ready = RtEvent::NO_RT_EVENT;
            }
            AutoLock q_lock(queue_lock);
            MapperState &map_state = mapper_states[*steal_it];
            if (!map_state.queue_guard)
            {
              // If we don't have a deferral event then grab our
              // ready queue of tasks so we can try to map them
              // this will also prevent them from being stolen
              if (!map_state.ready_queue.empty())
              {
                for (std::list<SingleTask*>::const_iterator it =
                      map_state.ready_queue.begin(); it !=
                      map_state.ready_queue.end(); it++)
                  if ((*it)->is_stealable() && !(*it)->is_origin_mapped())
                    input.stealable_tasks.push_back(*it);
                // Set the queue guard so no one else tries to
                // read the ready queue while we've checked it out
                if (!input.stealable_tasks.empty())
                  map_state.queue_guard = true;
              }
            }
            else
            {
              // Make an event if necessary
              if (!map_state.queue_waiter.exists())
                map_state.queue_waiter = Runtime::create_rt_user_event();
              // Record that we need to wait on it
              queue_copy_ready = map_state.queue_waiter;
            }
          } while (queue_copy_ready.exists());
        }
        if (input.stealable_tasks.empty())
          continue;
        input.thief_proc = thief;
        Mapper::StealRequestOutput output;
        // Ask the mapper what it wants to allow be stolen
        if (!input.stealable_tasks.empty())
          mapper->invoke_permit_steal_request(input, output);
        // See which tasks we can succesfully steal
        std::vector<SingleTask*> local_stolen;
        {
          // Retake the lock, put any tasks still in the ready queue
          // back into the queue and remove the queue guard
          AutoLock q_lock(queue_lock);
          MapperState &map_state = mapper_states[*steal_it];
#ifdef DEBUG_LEGION
          assert(map_state.queue_guard);
#endif
          std::list<SingleTask*> &rqueue = map_state.ready_queue;
          for (std::list<SingleTask*>::iterator it =
                rqueue.begin(); it != rqueue.end(); /*nothing*/)
          {
            if (output.stolen_tasks.find(*it) != output.stolen_tasks.end())
            {
              const ContextID ctx_id = 
                (*it)->get_context()->get_logical_tree_context();
              ContextState &state = context_states[ctx_id];
#ifdef DEBUG_LEGION
              assert(state.owned_tasks > 0);
#endif
              state.owned_tasks--;
              if (state.active && (state.owned_tasks == 0))
                decrement_active_contexts();
              if ((*it)->is_forward_progress_task())
                decrement_progress_tasks();
              (*it)->mark_stolen();
              local_stolen.push_back(*it);
              it = rqueue.erase(it);
            }
            else
              it++;
          }
          if (rqueue.empty())
          {
            if (map_state.deferral_event.exists())
              map_state.deferral_event = RtEvent::NO_RT_EVENT;
            else
              decrement_active_mappers();
          }
          // Remove the queue guard
          map_state.queue_guard = false;
          if (map_state.queue_waiter.exists())
          {
            Runtime::trigger_event(map_state.queue_waiter);
            map_state.queue_waiter = RtUserEvent::NO_RT_USER_EVENT;
          }
        }
        if (!local_stolen.empty())
        {
          successful_thiefs.push_back(stealer);
          for (std::vector<SingleTask*>::const_iterator it = 
                local_stolen.begin(); it != local_stolen.end(); it++)
            (*it)->deactivate_outstanding_task();
          if (stolen.empty())
            stolen.swap(local_stolen);
          else
            stolen.insert(stolen.end(),local_stolen.begin(),local_stolen.end());
        }
        else
          mapper->process_failed_steal(thief);
      }
      if (!stolen.empty())
      {
#ifdef DEBUG_LEGION
        for (std::vector<SingleTask*>::const_iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
          log_task.debug("task %s (ID %lld) stolen from processor " IDFMT
                         " by processor " IDFMT "", (*it)->get_task_name(), 
                         (*it)->get_unique_id(), local_proc.id, thief.id);
        }
#endif
        runtime->send_tasks(thief, stolen);
        // Also have to send advertisements to the mappers that 
        // successfully stole so they know that they can try again
        std::set<Processor> thief_set;
        thief_set.insert(thief);
        for (std::vector<MapperID>::const_iterator it = 
              successful_thiefs.begin(); it != successful_thiefs.end(); it++)
          runtime->send_advertisements(thief_set, *it, local_proc);
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::process_advertisement(Processor advertiser,
                                                 MapperID mid)
    //--------------------------------------------------------------------------
    {
      MapperManager *mapper = find_mapper(mid);
      mapper->process_advertisement(advertiser);
      // See if this mapper would like to try stealing again
      std::multimap<Processor,MapperID> stealing_targets;
      mapper->perform_stealing(stealing_targets);
      if (!stealing_targets.empty())
        runtime->send_steal_request(stealing_targets, local_proc);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_ready_queue(SingleTask *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(task != nullptr);
#endif
      // have to do this when we are not holding the lock
      task->activate_outstanding_task();
      // Check to see if this task is a task that must map in order to
      // guarantee forward progress
      const bool forward_progress_task = task->is_forward_progress_task();
      // We can do this without holding the lock because the
      // vector is of a fixed size
      ContextID ctx_id = task->get_context()->get_logical_tree_context();
      AutoLock q_lock(queue_lock);
#ifdef DEBUG_LEGION
      assert(mapper_states.find(task->map_id) != mapper_states.end());
#endif
      // Update the state for the context
      ContextState &state = context_states[ctx_id];
      if (state.active && (state.owned_tasks == 0))
        increment_active_contexts();
      state.owned_tasks++;
      // Also update the queue for the mapper
      MapperState &map_state = mapper_states[task->map_id];
      if (map_state.ready_queue.empty() || map_state.deferral_event.exists())
      {
        // Clear our deferral event since we are changing state
        map_state.deferral_event = RtEvent::NO_RT_EVENT;
        increment_active_mappers();
      }
      map_state.ready_queue.push_back(task);
      // Finally if this is a progress task increment it
      if (forward_progress_task)
        increment_progress_tasks();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::find_visible_memories(std::set<Memory> &visible)const
    //--------------------------------------------------------------------------
    {
      for (std::map<Memory,size_t>::const_iterator it =
            visible_memories.begin(); it != visible_memories.end(); it++)
        visible.insert(it->first);
    }

    //--------------------------------------------------------------------------
    Memory ProcessorManager::find_best_visible_memory(Memory::Kind kind) const
    //--------------------------------------------------------------------------
    {
      size_t affinity = 0;
      Memory result = Memory::NO_MEMORY;
      for (std::map<Memory,size_t>::const_iterator it =
            visible_memories.begin(); it != visible_memories.end(); it++)
      {
        if (it->first.kind() != kind)
          continue;
        if (it->second < affinity)
          continue;
        result = it->first;
        affinity = it->second;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::order_concurrent_task_launch(SingleTask *task,
                    ApEvent precondition, ApUserEvent ready, VariantID vid)
    //--------------------------------------------------------------------------
    {
      uint64_t lamport_clock = 0;
      {
        AutoLock c_lock(concurrent_lock);
#ifdef DEBUG_LEGION
        assert(concurrent_tasks.find(task) == concurrent_tasks.end());
#endif
        lamport_clock = concurrent_lamport_clock++;
        concurrent_tasks.insert(std::make_pair(task,
              ConcurrentState(lamport_clock, precondition, ready)));
      }
      // Check to see if the precondition event was poisoned
      bool poisoned = false;
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      bool triggered =
#endif
#endif
        precondition.has_triggered_faultaware(poisoned);
#ifdef DEBUG_LEGION
      assert(triggered);
#endif
      // Tell the task to compute the max all-reduce of lamport clocks
      task->concurrent_allreduce(this, lamport_clock, vid, poisoned);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::finalize_concurrent_task_order(SingleTask *task,
                                          uint64_t lamport_clock, bool poisoned)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(concurrent_lock);
      std::map<SingleTask*,ConcurrentState>::iterator finder = 
        concurrent_tasks.find(task);
#ifdef DEBUG_LEGION
      assert(finder != concurrent_tasks.end());
      assert(!finder->second.max);
      assert(finder->second.lamport_clock <= lamport_clock);
#endif
      if (concurrent_lamport_clock <= lamport_clock)
        concurrent_lamport_clock = lamport_clock + 1;
      if (poisoned)
      {
        Runtime::poison_event(finder->second.ready);
        concurrent_tasks.erase(finder);
      }
      else
      {
        finder->second.lamport_clock = lamport_clock;
        finder->second.max = true;
        ready_concurrent_tasks++;
        if (!outstanding_concurrent_task)
          start_next_concurrent_task();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::end_concurrent_task(void)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(concurrent_lock);
#ifdef DEBUG_LEGION
      assert(outstanding_concurrent_task);
#endif
      outstanding_concurrent_task = false;
      if (ready_concurrent_tasks > 0)
        start_next_concurrent_task();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::start_next_concurrent_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!concurrent_tasks.empty());
      assert(!outstanding_concurrent_task);
      assert(ready_concurrent_tasks > 0);
#endif
      // See if we can prove that there is a task that is safe to start
      uint64_t min_next = std::numeric_limits<uint64_t>::max();
      uint64_t min_pending = std::numeric_limits<uint64_t>::max();
      SingleTask *next = nullptr;
      TaskTreeCoordinates next_coords;
      for (std::map<SingleTask*,ConcurrentState>::const_iterator it =
            concurrent_tasks.begin(); it != concurrent_tasks.end(); it++)
      {
        if (it->second.max)
        {
          if (next != nullptr)
          {
            // Compare the lamport clocks
            if (it->second.lamport_clock < min_next)
            {
              next = it->first;
              next_coords.clear();
              min_next = it->second.lamport_clock;
            }
            else if (min_next == it->second.lamport_clock)
            {
              // Very bad case, same min of max all-reduce of clocks
              // Resolve this conflict based on task tree coordinates
              TaskTreeCoordinates it_coords;
              if (next_coords.empty())
                next->compute_task_tree_coordinates(next_coords);
              it->first->compute_task_tree_coordinates(it_coords);
              const size_t lower_bound =
                std::min(next_coords.size(), it_coords.size());
              bool equal = true;
              for (unsigned idx = 0; idx < lower_bound; idx++)
              {
                const ContextCoordinate &c1 = next_coords[idx];
                const ContextCoordinate &c2 = it_coords[idx];
                if (c1.context_index == c2.context_index)
                {
                  if (c2.index_point < c1.index_point)
                  {
                    next = it->first;
                    next_coords.swap(it_coords);
                  }
                  else if (c1.index_point == c2.index_point)
                    continue;
                }
                else if (c2.context_index < c1.context_index)
                {
                  next = it->first;
                  next_coords.swap(it_coords);
                }
                equal = false;
                break;
              }
              if (equal)
              {
#ifdef DEBUG_LEGION
                assert(next_coords.size() != it_coords.size());
#endif
                if (it_coords.size() < next_coords.size())
                {
                  next = it->first;
                  next_coords.swap(it_coords);
                }
              }
            }
          }
          else
          {
            next = it->first;
            min_next = it->second.lamport_clock;
          }
        }
        else if (it->second.lamport_clock < min_pending)
          min_pending = it->second.lamport_clock;
      }
      // If all the pending tasks with lamport clocks are
      // larger than our max lamport clock of the next task
      // to launch then we know they won't ever come before it
      // so we can issue our next task now, otherwise we'll need
      // to wait until those pending lamport clocks are done
      if (min_next < min_pending)
      {
        std::map<SingleTask*,ConcurrentState>::iterator finder =
          concurrent_tasks.find(next);
#ifdef DEBUG_LEGION
        assert(finder != concurrent_tasks.end());
#endif
        // Trigger the ready event with the precondition to keep
        // tools like Legion Spy happy even though we know that
        // the precondition event has already triggered
        Runtime::trigger_event_untraced(finder->second.ready, 
            finder->second.precondition);
        concurrent_tasks.erase(finder);
        ready_concurrent_tasks--;
        outstanding_concurrent_task = true;
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_mapping_operations(void)
    //--------------------------------------------------------------------------
    {
      std::multimap<Processor,MapperID> stealing_targets;
      std::vector<MapperID> mappers_with_stealable_work;
      std::vector<std::pair<MapperID,MapperManager*> > current_mappers;
      // Take a snapshot of our current mappers
      {
        AutoLock m_lock(mapper_lock,1,false/*exclusive*/);
        // Fast path for no deferred mappers
        current_mappers.resize(mappers.size());
        unsigned idx = 0;
        for (std::map<MapperID,std::pair<MapperManager*,bool> >::
              const_iterator it = mappers.begin(); it != 
              mappers.end(); it++, idx++)
          current_mappers[idx] = 
            std::pair<MapperID,MapperManager*>(it->first, it->second.first);
      }
      for (std::vector<std::pair<MapperID,MapperManager*> >::const_iterator
            it = current_mappers.begin(); it != current_mappers.end(); it++)
      {
        const MapperID map_id = it->first;
        MapperManager *const mapper = it->second;
        Mapper::SelectMappingInput input;
        {
          RtEvent input_ready;
          // Pull out the current tasks for this mapping operation
          // Need to iterate until we get access to the queue
          do
          {
            if (input_ready.exists() && !input_ready.has_triggered())
            {
              input_ready.wait();
              input_ready = RtEvent::NO_RT_EVENT;
            }
            AutoLock q_lock(queue_lock);
            MapperState &map_state = mapper_states[map_id];
            if (!map_state.queue_guard)
            {
              // If we don't have a deferral event then grab our
              // ready queue of tasks so we can try to map them
              // this will also prevent them from being stolen
              if (!map_state.deferral_event.exists() &&
                  !map_state.ready_queue.empty())
              {
                // Only ask the mapper about ready tasks that have
                // active contexts that we should keep mapping
                for (std::list<SingleTask*>::const_iterator it =
                      map_state.ready_queue.begin(); it != 
                      map_state.ready_queue.end(); it++)
                {
                  const ContextID ctx =
                    (*it)->get_context()->get_logical_tree_context();
                  const ContextState &ctx_state = context_states[ctx];
                  if (ctx_state.active || (*it)->is_forward_progress_task())
                    input.ready_tasks.push_back(*it);
                }
                // Set the queue guard so no one else tries to
                // read the ready queue while we've checked it out
                if (!input.ready_tasks.empty())
                  map_state.queue_guard = true;
              }
            }
            else
            {
              // Make an event if necessary
              if (!map_state.queue_waiter.exists())
                map_state.queue_waiter = Runtime::create_rt_user_event();
              // Record that we need to wait on it
              input_ready = map_state.queue_waiter;
            }
          } while (input_ready.exists());
        }
        // Do this before anything else in case we don't have any tasks
        if (!stealing_disabled)
          mapper->perform_stealing(stealing_targets);
        // Nothing to do if there are no tasks on the queue
        if (input.ready_tasks.empty())
          continue;
        // Ask the mapper which tasks it would like to schedule
        Mapper::SelectMappingOutput output;
        mapper->invoke_select_tasks_to_map(input, output);
        // If we had no entry then we better have gotten a mapper event
        if (output.map_tasks.empty() && output.relocate_tasks.empty())
        {
          const RtEvent wait_on = output.deferral_event.impl;
          if (wait_on.exists())
          {
            // Put this on the list of the deferred mappers
            AutoLock q_lock(queue_lock);
            MapperState &map_state = mapper_states[map_id];
            // We have to check to see if any new tasks were added to 
            // the ready queue while we were doing our mapper call, if 
            // they were then we need to invoke select_tasks_to_map again
            if (map_state.ready_queue.empty())
            {
#ifdef DEBUG_LEGION
              assert(!map_state.deferral_event.exists());
              assert(map_state.queue_guard);
#endif
              map_state.deferral_event = wait_on;
              // Decrement the number of active mappers
              decrement_active_mappers();
              // Clear the queue guard
              map_state.queue_guard = false;
              if (map_state.queue_waiter.exists())
              {
                Runtime::trigger_event(map_state.queue_waiter);
                map_state.queue_waiter = RtUserEvent::NO_RT_USER_EVENT;
              }
              // Launch a task to remove the deferred mapper 
              // event when it triggers
              DeferMapperSchedulerArgs args(this, map_id, wait_on);
              // If we need to recursively run the scheduler then we do so with
              // a lower priority than other meta-tasks to ensure that those 
              // other meta tasks can continue to make forward progress and the
              // scheduler cannot starve other tasks
              runtime->issue_runtime_meta_task(args,
                  LG_THROUGHPUT_WORK_PRIORITY, wait_on);
              // We can continue because there is nothing 
              // left to do for this mapper
              continue;
            }
            // Otherwise we fall through to put our tasks back on the queue 
            // which will lead to select_tasks_to_map being called again
          }
          else // Very bad, error message
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Mapper %s failed to specify an output MapperEvent "
                          "when returning from a call to 'select_tasks_to_map' "
                          "that performed no other actions. Specifying a "
                          "MapperEvent in such situation is necessary to avoid "
                          "livelock conditions. Please return a "
                          "'deferral_event' in the 'output' struct.",
                          mapper->get_mapper_name())
        }
        else if (!output.relocate_tasks.empty())
        {
          for (std::map<const Task*,Processor>::const_iterator it = 
                output.relocate_tasks.begin(); it != 
                output.relocate_tasks.end(); it++)
            if (it->second.kind() == Processor::UTIL_PROC)
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                  "Invalid mapper output. Mapper %s requested that task %s "
                  "(UID %lld) be relocated to a utility processor in "
                  "'select_tasks_to_map.' Only application processor kinds "
                  "are permitted to be the target processor for tasks.",
                  mapper->get_mapper_name(), it->first->get_task_name(),
                  it->first->get_unique_id())
        }
        // Figure out which tasks are to be triggered
        std::vector<SingleTask*> to_trigger;
        {
          // Retake the lock, put any tasks that the mapper didn't select
          // back on the queue and update the context states for any
          // that were selected 
          AutoLock q_lock(queue_lock);
          MapperState &map_state = mapper_states[map_id];
#ifdef DEBUG_LEGION
          assert(map_state.queue_guard);
#endif
          std::list<SingleTask*> &rqueue = map_state.ready_queue;
          // Iterate over the list and find any items to remove
          for (std::list<SingleTask*>::iterator it =
                rqueue.begin(); it != rqueue.end(); /*nothing*/)
          {
            if ((output.map_tasks.find(*it) != output.map_tasks.end()) ||
                (output.relocate_tasks.find(*it) != 
                 output.relocate_tasks.end()))
            {
              // Remove it from our set of local tasks
              const ContextID ctx_id = 
                (*it)->get_context()->get_logical_tree_context(); 
              ContextState &state = context_states[ctx_id];
#ifdef DEBUG_LEGION
              assert(state.owned_tasks > 0);
#endif
              state.owned_tasks--;
              if (state.active && (state.owned_tasks == 0))
                decrement_active_contexts();
              if ((*it)->is_forward_progress_task())
                decrement_progress_tasks();
              to_trigger.push_back(*it);
              it = rqueue.erase(it);
            }
            else
              it++;
          }
          if (rqueue.empty())
          {
            if (map_state.deferral_event.exists())
              map_state.deferral_event = RtEvent::NO_RT_EVENT;
            else
              decrement_active_mappers();
          }
          else if (!stealing_disabled)
          {
            for (std::list<SingleTask*>::const_iterator it =
                  rqueue.begin(); it != rqueue.end(); it++)
            {
              if ((*it)->is_stealable())
              {
                mappers_with_stealable_work.push_back(map_id);
                break;
              }
            }
          }
          // Remove the queue guard
          map_state.queue_guard = false;
          if (map_state.queue_waiter.exists())
          {
            Runtime::trigger_event(map_state.queue_waiter);
            map_state.queue_waiter = RtUserEvent::NO_RT_USER_EVENT;
          }
        }
        // Now we can trigger our tasks that the mapper selected
        std::map<Processor,std::vector<SingleTask*> > to_send;
        for (std::vector<SingleTask*>::const_iterator it = 
              to_trigger.begin(); it != to_trigger.end(); it++)
        {
          // Mark that this task is no longer outstanding
          (*it)->deactivate_outstanding_task();
          // Update the target processor for this task if necessary
          std::map<const Task*,Processor>::const_iterator finder = 
            output.relocate_tasks.find(*it);
          if (finder != output.relocate_tasks.end())
          {
            (*it)->set_target_proc(finder->second);
            // See if the target processor is local
            if (!runtime->is_local(finder->second))
            {
              // This is the tricky case, we need to actually send this
              // remotely, which is hard if it is a point task that is
              // owned by a slice task, if it is just a normal indvidual
              // task then we can just ship it remotely immediately
              to_send[finder->second].push_back(*it);
            }
            else
              (*it)->enqueue_ready_task(true/*use target processor*/);
          }
          else
          {
            TaskOp::TriggerTaskArgs trigger_args(*it);
            runtime->issue_runtime_meta_task(trigger_args,
                                             LG_THROUGHPUT_WORK_PRIORITY);
          }
        }
        if (!to_send.empty())
        {
          for (std::map<Processor,std::vector<SingleTask*> >::iterator
                it = to_send.begin(); it != to_send.end(); it++)
            runtime->send_tasks(it->first, it->second);
        }
      }

      // Advertise any work that we have
      if (!stealing_disabled && !mappers_with_stealable_work.empty())
      {
        for (std::vector<MapperID>::const_iterator it = 
              mappers_with_stealable_work.begin(); it !=
              mappers_with_stealable_work.end(); it++)
          issue_advertisements(*it);
      }

      // Finally issue any steal requeusts
      if (!stealing_disabled && !stealing_targets.empty())
        runtime->send_steal_request(stealing_targets, local_proc);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::issue_advertisements(MapperID map_id)
    //--------------------------------------------------------------------------
    {
      // Create a clone of the processors we want to advertise so that
      // we don't call into the high level runtime holding a lock
      std::set<Processor> failed_waiters;
      MapperManager *mapper = find_mapper(map_id);
      mapper->perform_advertisements(failed_waiters);
      if (!failed_waiters.empty())
        runtime->send_advertisements(failed_waiters, map_id, local_proc);
    }

  } // namespace Internal
} // namespace Legion
