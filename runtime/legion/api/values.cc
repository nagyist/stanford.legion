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

#include "legion/api/values.h"
#include "legion/api/runtime.h"
#include "legion/contexts/context.h"
#include "legion/kernel/exception.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // UntypedDeferredValue
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  UntypedDeferredValue::UntypedDeferredValue(void)
    : instance(Realm::RegionInstance::NO_INST)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  UntypedDeferredValue::UntypedDeferredValue(
      size_t field_size, Memory memory, const void* initial_value,
      size_t alignment)
  //--------------------------------------------------------------------------
  {
    initialize(memory, field_size, initial_value, alignment);
  }

  //--------------------------------------------------------------------------
  UntypedDeferredValue::UntypedDeferredValue(
      size_t field_size, Memory::Kind memkind, const void* initial_value,
      size_t alignment)
  //--------------------------------------------------------------------------
  {
    const Memory memory = find_memory_by_kind(memkind, true /*value*/);
    initialize(memory, field_size, initial_value, alignment);
  }

  //--------------------------------------------------------------------------
  void UntypedDeferredValue::initialize(
      Memory memory, size_t field_size, const void* initial_value,
      size_t alignment)
  //--------------------------------------------------------------------------
  {
    const Realm::Point<1, coord_t> zero(0);
    Realm::IndexSpace<1, coord_t> bounds = Realm::Rect<1, coord_t>(zero, zero);
    const std::vector<size_t> field_sizes(1, field_size);
    Realm::InstanceLayoutConstraints constraints(field_sizes, 0 /*blocking*/);
    int dim_order[1];
    dim_order[0] = 0;
    Realm::InstanceLayoutGeneric* layout =
        Realm::InstanceLayoutGeneric::choose_instance_layout(
            bounds, constraints, dim_order);
    layout->alignment_reqd = alignment;
    instance = allocate_instance(memory, layout);
    if (initial_value != nullptr)
    {
      // Check to see if we can write to it directly
      Runtime* runtime = Runtime::get_runtime();
      Context ctx = Runtime::get_context();
      const Processor exec_proc = runtime->get_executing_processor(ctx);
      Machine machine = Realm::Machine::get_machine();
      if (machine.has_affinity(exec_proc, memory))
      {
        // Has affinity so we shold jsut be able to memcpy this
        void* ptr = instance.pointer_untyped(0 /*offset*/, field_size);
        std::memcpy(ptr, initial_value, field_size);
      }
      else
      {
        Realm::ProfilingRequestSet no_requests;
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0 /*field id*/, field_size);
        const Internal::LgEvent wait_on(
            bounds.fill(dsts, no_requests, initial_value, field_size));
        if (wait_on.exists())
          wait_on.wait();
      }
    }
  }

  //--------------------------------------------------------------------------
  /*static*/ Memory UntypedDeferredValue::find_memory_by_kind(
      Memory::Kind kind, bool value)
  //--------------------------------------------------------------------------
  {
    Machine machine = Realm::Machine::get_machine();
    Machine::MemoryQuery finder(machine);
    Runtime* runtime = Runtime::get_runtime();
    Context ctx = Runtime::get_context();
    const Processor exec_proc = runtime->get_executing_processor(ctx);
    finder.best_affinity_to(exec_proc);
    finder.only_kind(kind);
    if (finder.count() == 0)
    {
      finder = Machine::MemoryQuery(machine);
      finder.has_affinity_to(exec_proc);
      finder.only_kind(kind);
    }
    if (finder.count() == 0)
      Internal::Exception(Internal::INTERFACE_EXCEPTION)
          << "Unable to find associated " << kind << " memory kind for "
          << exec_proc << " when performed an (Untyped)Deferred"
          << (value ? "Value" : "Buffer") << " creation";
    return finder.first();
  }

  //--------------------------------------------------------------------------
  UntypedDeferredValue::UntypedDeferredValue(const UntypedDeferredValue& rhs)
    : instance(rhs.instance)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  void UntypedDeferredValue::finalize(Context ctx) const
  //--------------------------------------------------------------------------
  {
    const size_t size = field_size();
    Runtime::legion_task_postamble(
        ctx, instance.pointer_untyped(0, size), size, true /*owner*/, instance);
  }

  //--------------------------------------------------------------------------
  Realm::RegionInstance UntypedDeferredValue::get_instance() const
  //--------------------------------------------------------------------------
  {
    return instance;
  }

  //--------------------------------------------------------------------------
  /*static*/ Realm::RegionInstance UntypedDeferredValue::allocate_instance(
      Memory memory, Realm::InstanceLayoutGeneric* layout)
  //--------------------------------------------------------------------------
  {
    if (Internal::implicit_context == nullptr)
      Internal::Exception(Internal::INTERFACE_EXCEPTION)
          << "Illegal request to create a DeferredBuffer, DeferredValue, "
          << "or a DeferredReduction outside of a Legion task.";
    return Internal::implicit_context->create_task_local_instance(
        memory, layout);
  }

  //--------------------------------------------------------------------------
  /*static*/ void UntypedDeferredValue::destroy_instance(
      Realm::RegionInstance instance, Realm::Event precondition)
  //--------------------------------------------------------------------------
  {
    if (Internal::implicit_context == nullptr)
      Internal::Exception(Internal::INTERFACE_EXCEPTION)
          << "Illegal request to destroy a DeferredBuffer, DeferredValue, "
          << "or a DeferredReduction outside of a Legion task.";
    // Don't trust events passed in by users to be safe from poison
    if (precondition.exists())
      return Internal::implicit_context->destroy_task_local_instance(
          instance,
          Internal::RtEvent(Realm::Event::ignorefaults(precondition)));
    else
      return Internal::implicit_context->destroy_task_local_instance(
          instance, Internal::RtEvent::NO_RT_EVENT);
  }

  //--------------------------------------------------------------------------
  /*static*/ Domain UntypedDeferredValue::get_index_space_bounds(
      IndexSpace space)
  //--------------------------------------------------------------------------
  {
    return Runtime::get_runtime()->get_index_space_domain(space);
  }

  //--------------------------------------------------------------------------
  /*static*/ void UntypedDeferredValue::report_incompatible_accessor(
      const char* accessor_kind, bool buffer)
  //--------------------------------------------------------------------------
  {
    if (buffer)
      Internal::Exception(Internal::INTERFACE_EXCEPTION)
          << "Incompatible " << accessor_kind << " for (Untyped)DeferredBuffer";
    else
      Internal::Exception(Internal::INTERFACE_EXCEPTION)
          << "Incompatible " << accessor_kind << " for (Untyped)DeferredValue";
  }

  //--------------------------------------------------------------------------
  /*static*/ void UntypedDeferredValue::report_nondense_domain(void)
  //--------------------------------------------------------------------------
  {
    Internal::Exception(Internal::INTERFACE_EXCEPTION)
        << "DeferredBuffer only supporst dense domains. Make sure your "
        << "domain for a DeferredBuffer does not have a sparsity map.";
  }

  //--------------------------------------------------------------------------
  /*static*/ void UntypedDeferredValue::report_nondense_rect(void)
  //--------------------------------------------------------------------------
  {
    Internal::Exception(Internal::INTERFACE_EXCEPTION)
        << "Illegal request for point of non-dense rectangle in a "
        << "DeferredBuffer. Make sure that you only ask for rectangles "
        << "that are dense in the layout of the deferred buffer or use "
        << "the version that passes back strides.";
  }

}  // namespace Legion
