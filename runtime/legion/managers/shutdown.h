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

#ifndef __LEGION_SHUTDOWN_MANAGER_H__
#define __LEGION_SHUTDOWN_MANAGER_H__

#include "legion/core/metatask.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ShutdownManager
     * A class for helping to manage the shutdown of the 
     * runtime after the application has finished
     */
    class ShutdownManager {
    public:
      enum ShutdownPhase {
        CHECK_TERMINATION = 1,
        CONFIRM_TERMINATION = 2,
        CHECK_SHUTDOWN = 3,
        CONFIRM_SHUTDOWN = 4,
      };
    public:
      struct RetryShutdownArgs : public LgTaskArgs<RetryShutdownArgs> {
      public:
        static const LgTaskID TASK_ID = LG_RETRY_SHUTDOWN_TASK_ID;
      public:
        RetryShutdownArgs(ShutdownPhase p)
          : LgTaskArgs<RetryShutdownArgs>(0), phase(p) { }
      public:
        const ShutdownPhase phase;
      };
    public:
      ShutdownManager(ShutdownPhase phase, AddressSpaceID source,
                      unsigned radix, ShutdownManager *owner = NULL);
      ShutdownManager(const ShutdownManager &rhs) = delete;
      ~ShutdownManager(void);
    public:
      ShutdownManager& operator=(const ShutdownManager &rhs) = delete;
    public:
      bool attempt_shutdown(void);
      bool handle_response(int code, bool success, 
                           const std::set<RtEvent> &to_add);
    protected:
      void finalize(void);
    public:
      static void handle_shutdown_notification(Deserializer &derez, 
                          AddressSpaceID source);
      static void handle_shutdown_response(Deserializer &derez);
    public:
      void record_outstanding_tasks(void);
      void record_recent_message(void);
      void record_pending_message(RtEvent pending_event);
    public:
      const ShutdownPhase phase;
      const AddressSpaceID source; 
      const unsigned radix;
      ShutdownManager *const owner;
    protected:
      mutable LocalLock shutdown_lock;
      unsigned needed_responses;
      std::set<RtEvent> wait_for;
      int return_code;
      bool result;
    };

  } // namespace Internal
} // namespace Legion

#endif // __LEGION_SHUTDOWN_MANAGER_H__
