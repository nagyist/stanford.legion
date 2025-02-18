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

#include "legion/kernel/exception.h"
#include "legion/kernel/runtime.h"
#include "legion/contexts/inner.h"
#include "legion/operations/operation.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Exception
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Exception::Exception(ExceptionType t, const Operation* o)
      : Realm::LoggerMessage(
            (t == APPLICATION_EXCEPTION) ? log_legion.print() :
            ((t == WARNING_EXCEPTION) &&
             ((runtime == nullptr) || !runtime->warnings_are_errors)) ?
                                           log_legion.warning() :
                                           log_legion.error()),
        op(o), type(t)
    //--------------------------------------------------------------------------
    {
      (*this)
          << "\n------------------------------------------------------------\n";
      switch (type)
      {
        case APPLICATION_EXCEPTION:
          {
            (*this) << "LEGION ENCOUNTERED AN APPLICATION EXCEPTION";
            break;
          }
        case INTERFACE_EXCEPTION:
          {
            (*this) << "LEGION ENCOUNTERED AN API USAGE ERROR";
            break;
          }
        case DYNAMIC_TYPE_EXCEPTION:
          {
            (*this) << "LEGION ENCOUNTERED A DYNAMIC TYPE ERROR";
            break;
          }
        case PROGRAMMING_MODEL_EXCEPTION:
          {
            (*this) << "LEGION ENCOUNTERED A PROGRAMMING MODEL ERROR";
            break;
          }
        case MAPPER_EXCEPTION:
          {
            (*this) << "LEGION ENCOUNTERED A MAPPER ERROR";
            break;
          }
        case STARTUP_EXCEPTION:
          {
            (*this) << "LEGION ENCOUNTERED A MAPPER ERROR";
            break;
          }
        case FATAL_EXCEPTION:
          {
            (*this) << "LEGION ENCOUNTERED AN INTERNAL FATAL ERROR";
            break;
          }
        case WARNING_EXCEPTION:
          {
            if ((runtime == nullptr) || !runtime->warnings_are_errors)
              (*this) << "LEGION_ENCOUNTERED A WARNING";
            else
              (*this)
                  << "LEGION ENCOUNTERED A WARNING BEING TREATED AS AN ERROR";
            break;
          }
        default:
          std::abort();
      }
      // Check to see if we can report where this error occurred
      if (op != nullptr)
        (*this) << " IN " << op->get_logging_name() << " ("
                << op->get_unique_op_id() << "):\n";
      else if (implicit_context != nullptr)
        (*this) << " IN " << implicit_context->get_task_name() << "("
                << implicit_context->get_unique_id() << "):\n";
      else if (implicit_provenance > 0)
        (*this) << " IN META-TASK FOR UKNOWN OPERATION " << implicit_provenance
                << ":\n";
      else
        (*this) << ":\n";
    }

    //--------------------------------------------------------------------------
    Exception::~Exception(void)
    //--------------------------------------------------------------------------
    {
      // If this is a warning, check to see if we're just going to report
      // that and be done with it, otherwise we do extra work
      if ((type == WARNING_EXCEPTION) &&
          ((runtime == nullptr) || !runtime->warnings_are_errors))
      {
        if (runtime->warnings_backtrace)
        {
          (*this) << "\n-----------------------------------\n";
          (*this) << "Backtrace:\n\n";
          Realm::Backtrace bt;
          bt.capture_backtrace();
          (*this) << bt;
        }
        (*this)
            << "\n------------------------------------------------------------";
        // Return which will just result in the message being printed
        // when the base class destructor is called
        return;
      }
      if (op != nullptr)
      {
        Provenance* prov = op->get_provenance();
        if (prov != nullptr)
        {
          (*this) << "\n-----------------------------------\n";
          (*this) << "Provenance:\n\n" << prov->human;
        }
        (*this) << "\n-----------------------------------\n";
        (*this) << "Task Tree Trace:\n\n";
        (*this) << op->get_logging_name() << "(" << op->get_unique_op_id()
                << ")";
        InnerContext* context = op->get_context();
        while (context->owner_task != nullptr)
        {
          (*this) << "\n"
                  << context->get_task_name() << "(" << context->get_unique_id()
                  << ")";
          context = context->owner_task->get_context();
        }
      } else if (implicit_context != nullptr)
      {
        Provenance* prov = implicit_context->owner_task->get_provenance();
        if (prov != nullptr)
        {
          (*this) << "\n-----------------------------------\n";
          (*this) << "Provenance:\n\n" << prov->human;
        }
        (*this) << "\n-----------------------------------\n";
        (*this) << "Task Tree Trace:\n";
        TaskContext* context = implicit_context;
        while (context->owner_task != nullptr)
        {
          (*this) << "\n"
                  << context->get_task_name() << "(" << context->get_unique_id()
                  << ")";
          context = context->owner_task->get_context();
        }
      }
      (*this) << "\n-----------------------------------\n";
      (*this) << "Backtrace:\n\n";
      Realm::Backtrace bt;
      bt.capture_backtrace();
      (*this) << bt;
      (*this)
          << "\n------------------------------------------------------------";
      // Now flush the message by explicitly calling the base destructor
      // which is safe because we're about to abort
      this->Realm::LoggerMessage::~LoggerMessage();
      std::abort();
    }

    //--------------------------------------------------------------------------
    Exception& Exception::operator<<(Memory memory)
    //--------------------------------------------------------------------------
    {
      (*this) << std::hex << memory.id << std::dec;
      return *this;
    }

    //--------------------------------------------------------------------------
    Exception& Exception::operator<<(Processor proc)
    //--------------------------------------------------------------------------
    {
      (*this) << std::hex << proc.id << std::dec;
      return *this;
    }

    //--------------------------------------------------------------------------
    Exception& Exception::operator<<(Memory::Kind kind)
    //--------------------------------------------------------------------------
    {
      static const char* memory_names[] = {
#define MEMORY_NAMES(name, desc) #name,
          REALM_MEMORY_KINDS(MEMORY_NAMES)
#undef MEMORY_NAMES
      };
      (*this) << memory_names[kind];
      return *this;
    }

    //--------------------------------------------------------------------------
    Exception& Exception::operator<<(Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      static const char* proc_names[] = {
#define PROCESSOR_NAMES(name, desc) #name,
          REALM_PROCESSOR_KINDS(PROCESSOR_NAMES)
#undef PROCESSOR_NAMES
      };
      (*this) << proc_names[kind];
      return *this;
    }

    //--------------------------------------------------------------------------
    Exception& Exception::operator<<(PhysicalInstance inst)
    //--------------------------------------------------------------------------
    {
      (*this) << std::hex << inst.id << std::dec;
      return *this;
    }

    //--------------------------------------------------------------------------
    Exception& Exception::operator<<(LayoutConstraintKind kind)
    //--------------------------------------------------------------------------
    {
      static const char* constraint_names[] = {
#define CONSTRAINT_NAMES(name, desc) desc,
          LEGION_LAYOUT_CONSTRAINT_KINDS(CONSTRAINT_NAMES)
#undef CONSTRAINT_NAMES
      };
      (*this) << constraint_names[kind];
      return *this;
    }

  }  // namespace Internal
}  // namespace Legion
