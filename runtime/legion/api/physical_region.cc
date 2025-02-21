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

#include "legion/api/physical_region_impl.h"
#include "legion/contexts/inner.h"
#include "legion/kernel/exception.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/region.h"
#include "legion/operations/detach.h"
#include "legion/tasks/single.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Physical Region
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  PhysicalRegion::PhysicalRegion(void) : impl(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  PhysicalRegion::PhysicalRegion(const PhysicalRegion& rhs) : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  PhysicalRegion::PhysicalRegion(PhysicalRegion&& rhs) noexcept : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    rhs.impl = nullptr;
  }

  //--------------------------------------------------------------------------
  PhysicalRegion::PhysicalRegion(Internal::PhysicalRegionImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  PhysicalRegion::~PhysicalRegion(void)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
      impl = nullptr;
    }
  }

  //--------------------------------------------------------------------------
  PhysicalRegion& PhysicalRegion::operator=(const PhysicalRegion& rhs)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
    }
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_reference();
    return *this;
  }

  //--------------------------------------------------------------------------
  PhysicalRegion& PhysicalRegion::operator=(PhysicalRegion&& rhs) noexcept
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
    impl = rhs.impl;
    rhs.impl = nullptr;
    return *this;
  }

  //--------------------------------------------------------------------------
  std::size_t PhysicalRegion::hash(void) const
  //--------------------------------------------------------------------------
  {
    return std::hash<const void*>{}(impl);
  }

  //--------------------------------------------------------------------------
  bool PhysicalRegion::is_mapped(void) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
      return false;
    return impl->is_mapped();
  }

  //--------------------------------------------------------------------------
  void PhysicalRegion::wait_until_valid(
      bool silence_warnings, const char* warning_string)
  //--------------------------------------------------------------------------
  {
#ifdef DEBUG_LEGION
    assert(impl != nullptr);
#endif
    impl->wait_until_valid(silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  bool PhysicalRegion::is_valid(void) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      return impl->is_valid();
    else
      return false;
  }

  //--------------------------------------------------------------------------
  LogicalRegion PhysicalRegion::get_logical_region(void) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      return impl->get_logical_region();
    else
      return LogicalRegion::NO_REGION;
  }

  //--------------------------------------------------------------------------
  PrivilegeMode PhysicalRegion::get_privilege(void) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      return impl->get_privilege();
    else
      return LEGION_NO_ACCESS;
  }

  //--------------------------------------------------------------------------
  void PhysicalRegion::get_memories(
      std::set<Memory>& memories, bool silence_warnings,
      const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    impl->get_memories(memories, silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  void PhysicalRegion::get_fields(std::vector<FieldID>& fields) const
  //--------------------------------------------------------------------------
  {
    impl->get_fields(fields);
  }

  //--------------------------------------------------------------------------
  void PhysicalRegion::get_bounds(void* realm_is, TypeTag type_tag) const
  //--------------------------------------------------------------------------
  {
    impl->get_bounds(realm_is, type_tag);
  }

  //--------------------------------------------------------------------------
  Realm::RegionInstance PhysicalRegion::get_instance_info(
      PrivilegeMode mode, FieldID fid, size_t field_size, void* realm_is,
      TypeTag type_tag, const char* warning_string, bool silence_warnings,
      bool generic_accessor, bool check_field_size, ReductionOpID redop) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
      Internal::Exception(Internal::INTERFACE_EXCEPTION)
          << "Illegal request to create an accessor on null physical region";
    return impl->get_instance_info(
        mode, fid, field_size, realm_is, type_tag, warning_string,
        silence_warnings, generic_accessor, check_field_size, redop);
  }

  //--------------------------------------------------------------------------
  Realm::RegionInstance PhysicalRegion::get_padding_info(
      FieldID fid, size_t field_size, Domain* inner, Domain& outer,
      const char* warning_string, bool silence_warnings, bool generic_accessor,
      bool check_field_size) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
      Internal::Exception(Internal::INTERFACE_EXCEPTION)
          << "Illegal request to create a padded accessor on null physical "
             "region";
    return impl->get_padding_info(
        fid, field_size, inner, outer, warning_string, silence_warnings,
        generic_accessor, check_field_size);
  }

  //--------------------------------------------------------------------------
  void PhysicalRegion::report_incompatible_accessor(
      const char* accessor_kind, Realm::RegionInstance instance,
      FieldID fid) const
  //--------------------------------------------------------------------------
  {
    impl->report_incompatible_accessor(accessor_kind, instance, fid);
  }

  //--------------------------------------------------------------------------
  void PhysicalRegion::report_incompatible_multi_accessor(
      unsigned index, FieldID fid, Realm::RegionInstance inst1,
      Realm::RegionInstance inst2) const
  //--------------------------------------------------------------------------
  {
    impl->report_incompatible_multi_accessor(index, fid, inst1, inst2);
  }

  //--------------------------------------------------------------------------
  void PhysicalRegion::report_colocation_violation(
      const char* accessor_kind, FieldID fid, Realm::RegionInstance inst1,
      Realm::RegionInstance inst2, const PhysicalRegion& other,
      bool reduction) const
  //--------------------------------------------------------------------------
  {
    impl->report_colocation_violation(
        accessor_kind, fid, inst1, inst2, other, reduction);
  }

  //--------------------------------------------------------------------------
  /*static*/ void PhysicalRegion::empty_colocation_regions(
      const char* accessor_kind, FieldID fid, bool reduction)
  //--------------------------------------------------------------------------
  {
    Internal::PhysicalRegionImpl::empty_colocation_regions(
        accessor_kind, fid, reduction);
  }

  //--------------------------------------------------------------------------
  /*static*/ void PhysicalRegion::fail_bounds_check(
      DomainPoint p, FieldID fid, PrivilegeMode mode, bool multi)
  //--------------------------------------------------------------------------
  {
    Internal::PhysicalRegionImpl::fail_bounds_check(p, fid, mode, multi);
  }

  //--------------------------------------------------------------------------
  /*static*/ void PhysicalRegion::fail_bounds_check(
      Domain d, FieldID fid, PrivilegeMode mode, bool multi)
  //--------------------------------------------------------------------------
  {
    Internal::PhysicalRegionImpl::fail_bounds_check(d, fid, mode, multi);
  }

  //--------------------------------------------------------------------------
  /*static*/ void PhysicalRegion::fail_privilege_check(
      DomainPoint p, FieldID fid, PrivilegeMode mode)
  //--------------------------------------------------------------------------
  {
    Internal::PhysicalRegionImpl::fail_privilege_check(p, fid, mode);
  }

  //--------------------------------------------------------------------------
  /*static*/ void PhysicalRegion::fail_privilege_check(
      Domain d, FieldID fid, PrivilegeMode mode)
  //--------------------------------------------------------------------------
  {
    Internal::PhysicalRegionImpl::fail_privilege_check(d, fid, mode);
  }

  //--------------------------------------------------------------------------
  /*static*/ void PhysicalRegion::fail_padding_check(DomainPoint p, FieldID fid)
  //--------------------------------------------------------------------------
  {
    Internal::PhysicalRegionImpl::fail_padding_check(p, fid);
  }

  //--------------------------------------------------------------------------
  /*static*/ void PhysicalRegion::fail_nondense_rect(void)
  //--------------------------------------------------------------------------
  {
    Internal::Exception(Internal::INTERFACE_EXCEPTION)
        << "Illegal request for non-dense rectangle pointer. Use the "
        << "version of 'ptr' for a rectangle that also returns strides.";
  }

  //--------------------------------------------------------------------------
  /*static*/ void PhysicalRegion::fail_rect_piece(void)
  //--------------------------------------------------------------------------
  {
    Internal::Exception(Internal::INTERFACE_EXCEPTION)
        << "Illegal request for pointer of a rectangle not contained "
        << "within the bounds of any piece in the instance.";
  }

  /////////////////////////////////////////////////////////////
  // ExternalResources
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  ExternalResources::ExternalResources(void) : impl(nullptr)
  //--------------------------------------------------------------------------
  { }

  ExternalResources::ExternalResources(const ExternalResources& rhs)
    : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  ExternalResources::ExternalResources(Internal::ExternalResourcesImpl* i)
    : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  ExternalResources::ExternalResources(ExternalResources&& rhs) noexcept
    : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    rhs.impl = nullptr;
  }

  //--------------------------------------------------------------------------
  ExternalResources::~ExternalResources(void)
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
  }

  //--------------------------------------------------------------------------
  ExternalResources& ExternalResources::operator=(const ExternalResources& rhs)
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_reference();
    return *this;
  }

  //--------------------------------------------------------------------------
  ExternalResources& ExternalResources::operator=(
      ExternalResources&& rhs) noexcept
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
    impl = rhs.impl;
    rhs.impl = nullptr;
    return *this;
  }

  //--------------------------------------------------------------------------
  size_t ExternalResources::size(void) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
      return 0;
    return impl->size();
  }

  //--------------------------------------------------------------------------
  PhysicalRegion ExternalResources::operator[](unsigned index) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
      return PhysicalRegion();
    return impl->get_region(index);
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Physical Region Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(
        const RegionRequirement& r, RtEvent mapped, ApEvent ready,
        ApUserEvent term, bool m, TaskContext* ctx, MapperID mid,
        MappingTagID t, bool leaf, bool virt, bool col, uint64_t blocking)
      : Collectable(), context(ctx), map_id(mid), tag(t), leaf_region(leaf),
        virtual_mapped(virt), collective(col),
        replaying((ctx != nullptr) ? ctx->owner_task->is_replaying() : false),
        req(r), mapped_event(mapped), ready_event(ready),
        termination_event(term), blocking_index(blocking), mapped(m),
        valid(false), made_accessor(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::~PhysicalRegionImpl(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!termination_event.exists());
#endif
      if (!references.empty() && !replaying)
      {
        if (leaf_region)
          references.remove_resource_references(PHYSICAL_REGION_REF);
        else
          references.remove_valid_references(PHYSICAL_REGION_REF);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::wait_until_valid(
        bool silence_warnings, const char* warning_string, bool warn,
        const char* source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(implicit_context != nullptr);
      assert(implicit_context == context);
#endif
      context->record_blocking_call(blocking_index);
      if (runtime->runtime_warnings && !silence_warnings &&
          (context != nullptr) && !context->is_leaf_context())
      {
        if (source != nullptr)
          REPORT_LEGION_WARNING(
              LEGION_WARNING_WAITING_REGION,
              "Waiting for a physical region to be valid "
              "for call %s in non-leaf task %s (UID %lld) is a violation of "
              "Legion's deferred execution model best practices. You may "
              "notice a severe performance degradation. Warning string: %s",
              source, context->get_task_name(), context->get_unique_id(),
              (warning_string == nullptr) ? "" : warning_string)
        else
          REPORT_LEGION_WARNING(
              LEGION_WARNING_WAITING_REGION,
              "Waiting for a physical region to be valid "
              "in non-leaf task %s (UID %lld) is a violation of Legion's "
              "deferred execution model best practices. You may notice a "
              "severe performance degradation. Warning string: %s",
              context->get_task_name(), context->get_unique_id(),
              (warning_string == nullptr) ? "" : warning_string)
      }
      if (mapped_event.exists() && !mapped_event.has_triggered())
      {
        if (warn && !silence_warnings && (source != nullptr))
          REPORT_LEGION_WARNING(
              LEGION_WARNING_MISSING_REGION_WAIT,
              "Request for %s was performed on a "
              "physical region in task %s (ID %lld) without first waiting "
              "for the physical region to be valid. Legion is performing "
              "the wait for you. Warning string: %s",
              source, context->get_task_name(), context->get_unique_id(),
              (warning_string == nullptr) ? "" : warning_string)
        mapped_event.wait();
      }
      // If we've already gone through this process we're good
      if (valid)
        return;
      // Now wait for the reference to be ready
      bool poisoned = false;
      if (!ready_event.has_triggered_faultaware(poisoned))
      {
        if (!poisoned)
          ready_event.wait_faultaware(poisoned);
      }
      if (poisoned)
        context->raise_poison_exception();
      valid = true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      if (valid)
        return true;
      if (!mapped_event.exists() || mapped_event.has_triggered())
      {
        bool poisoned = false;
        if (ready_event.has_triggered_faultaware(poisoned))
          return true;
        if (poisoned)
          implicit_context->raise_poison_exception();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::is_mapped(void) const
    //--------------------------------------------------------------------------
    {
      return mapped;
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegionImpl::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
      return req.region;
    }

    //--------------------------------------------------------------------------
    PrivilegeMode PhysicalRegionImpl::get_privilege(void) const
    //--------------------------------------------------------------------------
    {
      return req.privilege;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::unmap_region(void)
    //--------------------------------------------------------------------------
    {
      if (!mapped)
        return;
#ifdef DEBUG_LEGION
      assert(termination_event.exists());
#endif
      // trigger the termination event conditional upon the ready event
      Runtime::trigger_event_untraced(termination_event, ready_event);
#ifdef LEGION_SPY
      // This is a really mind-bending corner case so be prepared
      // If we're doing a trace replay and we actually end up replaying a
      // physical template, we need to make it look to Legion Spy like the
      // fence at the beginning of the trace depends on any mapped physical
      // regions in the context that will be unmapped during the execution
      // of the trace otherwise Legion Spy will be unhappy with its validation.
      // We can fake this because it is already safe by definition, but just
      // in a way that Legion Spy can't actually see with its analysis. There
      // are two different sceanrios in the case where unmapping of physical
      // region occurs inside of the trace:
      // 1. the application doesn't unmap before launching a task or other
      //    operation that uses an interfering region requirement and the
      //    runtime has to insert unmap/remap operations which by definition
      //    will prevent a physical template from being captured because
      //    inline mapping operations are not (and never will be) memoizable
      //    so on all future traces we can only do logical analysis
      // 2. the application does its own unmap before a conflicting use of
      //    the logical region which creates an implicit happens-before
      //    relationship between the unmap and any uses by the template
      //    because operations can't be replayed before they are launched
      // There we can see this is trivially safe, but we need to create this
      // explicit event relationship for Legion Spy here to keep it happy
      const ApEvent tracing_replay_event = context->get_tracing_replay_event();
      if (tracing_replay_event.exists())
        LegionSpy::log_event_dependence(
            termination_event, tracing_replay_event);
#endif
#ifdef DEBUG_LEGION
      termination_event = ApUserEvent::NO_AP_USER_EVENT;
#endif
      mapped = false;
      valid = false;
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalRegionImpl::remap_region(
        ApEvent new_ready, uint64_t blocking)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!mapped);
      assert(!termination_event.exists());
#endif
      blocking_index = blocking;
      termination_event = Runtime::create_ap_user_event(nullptr);
      ready_event = new_ready;
      mapped = true;
      return termination_event;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& PhysicalRegionImpl::get_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return req;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::add_padded_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      padded_fields.emplace_back(fid);
      // Resort to keep things in order
      if (padded_fields.size() > 1)
        std::sort(padded_fields.begin(), padded_fields.end());
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::set_reference(const InstanceRef& ref, bool safe)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ref.has_ref());
      assert(references.empty());
      assert(safe || (mapped_event.exists() && !mapped_event.has_triggered()));
#endif
      references.add_instance(ref);
      if (!replaying)
      {
        if (leaf_region)
          ref.add_resource_reference(PHYSICAL_REGION_REF);
        else
          ref.add_valid_reference(PHYSICAL_REGION_REF);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::set_references(const InstanceSet& refs, bool safe)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(references.empty());
      assert(safe || (mapped_event.exists() && !mapped_event.has_triggered()));
#endif
      references = refs;
      if (!references.empty() && !replaying)
      {
        if (leaf_region)
          references.add_resource_references(PHYSICAL_REGION_REF);
        else
          references.add_valid_references(PHYSICAL_REGION_REF);
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::has_references(void) const
    //--------------------------------------------------------------------------
    {
      return !references.empty();
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::get_references(InstanceSet& instances) const
    //--------------------------------------------------------------------------
    {
      if (mapped_event.exists() && !mapped_event.has_triggered())
        mapped_event.wait();
      instances = references;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::get_memories(
        std::set<Memory>& memories, bool silence_warnings,
        const char* warning_string) const
    //--------------------------------------------------------------------------
    {
      if (mapped_event.exists() && !mapped_event.has_triggered())
      {
        if (runtime->runtime_warnings && !silence_warnings)
          REPORT_LEGION_WARNING(
              LEGION_WARNING_MISSING_REGION_WAIT,
              "Request for 'get_memories' was performed on a "
              "physical region in task %s (ID %lld) without first waiting "
              "for the physical region to be valid. Legion is performing "
              "the wait for you. Warning string: %s",
              context->get_task_name(), context->get_unique_id(),
              (warning_string == nullptr) ? "" : warning_string)
        mapped_event.wait();
      }
      const InstanceSet& instances = references;
      for (unsigned idx = 0; idx < instances.size(); idx++)
        memories.insert(instances[idx].get_memory());
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::get_fields(std::vector<FieldID>& fields) const
    //--------------------------------------------------------------------------
    {
      // Just get these from the region requirement
      fields.insert(
          fields.end(), req.privilege_fields.begin(),
          req.privilege_fields.end());
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::get_bounds(void* realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_domain(
          req.region.get_index_space(), realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    PieceIteratorImpl* PhysicalRegionImpl::get_piece_iterator(
        FieldID fid, bool privilege_only, bool silence_warnings,
        const char* warning_string)
    //--------------------------------------------------------------------------
    {
      if (req.privilege_fields.find(fid) == req.privilege_fields.end())
        REPORT_LEGION_ERROR(
            ERROR_INVALID_FIELD_PRIVILEGES,
            "Piece iterator construction in task %s on "
            "PhysicalRegion that does not contain field %d!",
            context->get_task_name(), fid)
      if (mapped_event.exists() && !mapped_event.has_triggered())
      {
        if (runtime->runtime_warnings && !silence_warnings)
          REPORT_LEGION_WARNING(
              LEGION_WARNING_MISSING_REGION_WAIT,
              "Request for 'get_piece_iterator' was performed on a "
              "physical region in task %s (ID %lld) without first waiting "
              "for the physical region to be valid. Legion is performing "
              "the wait for you. Warning string: %s",
              context->get_task_name(), context->get_unique_id(),
              (warning_string == nullptr) ? "" : warning_string)
        mapped_event.wait();
      }
      const InstanceSet& instances = references;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const InstanceRef& ref = instances[idx];
        if (ref.is_field_set(fid))
        {
          PhysicalManager* manager = ref.get_physical_manager();
          if (privilege_only)
          {
            IndexSpaceNode* privilege_node =
                runtime->get_node(req.region.get_index_space());
            return manager->create_piece_iterator(privilege_node);
          }
          else
            return manager->create_piece_iterator(nullptr);
        }
      }
      std::abort();
    }

    //--------------------------------------------------------------------------
    PhysicalInstance PhysicalRegionImpl::get_instance_info(
        PrivilegeMode mode, FieldID fid, size_t field_size, void* realm_is,
        TypeTag type_tag, const char* warning_string, bool silence_warnings,
        bool generic_accessor, bool check_field_size, ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      // Check the privilege mode first
      switch (mode)
      {
        case LEGION_READ_ONLY:
          {
            if (!(LEGION_READ_ONLY & req.privilege))
              REPORT_LEGION_ERROR(
                  ERROR_ACCESSOR_PRIVILEGE_CHECK,
                  "Error creating read-only field accessor without "
                  "read-only privileges on field %d in task %s",
                  fid, context->get_task_name())
            break;
          }
        case LEGION_READ_WRITE:
          {
            if (req.privilege == LEGION_WRITE_DISCARD)
            {
              if (!silence_warnings)
                REPORT_LEGION_WARNING(
                    LEGION_WARNING_READ_DISCARD,
                    "creating read-write accessor for "
                    "field %d in task %s which only has "
                    "WRITE_DISCARD privileges. You may be "
                    "accessing uninitialized data. "
                    "Warning string: %s",
                    fid, context->get_task_name(),
                    (warning_string == nullptr) ? "" : warning_string)
            }
            else if (req.privilege != LEGION_READ_WRITE)
              REPORT_LEGION_ERROR(
                  ERROR_ACCESSOR_PRIVILEGE_CHECK,
                  "Error creating read-write field accessor without "
                  "read-write privileges on field %d in task %s",
                  fid, context->get_task_name())
            break;
          }
        case LEGION_WRITE_ONLY:
        case LEGION_WRITE_DISCARD:
          {
            if (!(LEGION_WRITE_ONLY & req.privilege))
              REPORT_LEGION_ERROR(
                  ERROR_ACCESSOR_PRIVILEGE_CHECK,
                  "Error creating write-discard field accessor "
                  "without write privileges on field %d in task %s",
                  fid, context->get_task_name())
            break;
          }
        case LEGION_REDUCE:
          {
            if ((LEGION_REDUCE != req.privilege) || (redop != req.redop))
            {
              if (!(LEGION_REDUCE & req.privilege))
                REPORT_LEGION_ERROR(
                    ERROR_ACCESSOR_PRIVILEGE_CHECK,
                    "Error creating reduction field accessor "
                    "without reduction privileges on field %d in "
                    "task %s",
                    fid, context->get_task_name())
              else if (
                  (redop != req.redop) && (req.privilege != LEGION_READ_WRITE))
                REPORT_LEGION_ERROR(
                    ERROR_ACCESSOR_PRIVILEGE_CHECK,
                    "Error creating reduction field accessor "
                    "with mismatched reduction operators %d and %d "
                    "on field %d in task %s",
                    redop, req.redop, fid, context->get_task_name())
#ifdef DEBUG_LEGION
              assert(req.privilege == LEGION_READ_WRITE);
#endif
            }
            break;
          }
        default:  // rest of the privileges don't matter
          break;
      }
      if (context != nullptr)
      {
        if (context->is_inner_context())
          REPORT_LEGION_ERROR(
              ERROR_INNER_TASK_VIOLATION,
              "Illegal accessor construction inside "
              "task %s (UID %lld) for a variant that was labeled as an 'inner' "
              "variant.",
              context->get_task_name(), context->get_unique_id())
        else if (
            runtime->runtime_warnings && !silence_warnings &&
            !context->is_leaf_context())
          REPORT_LEGION_WARNING(
              LEGION_WARNING_NONLEAF_ACCESSOR,
              "Accessor construction in non-leaf "
              "task %s (UID %lld) is a blocking operation in violation of "
              "Legion's deferred execution model best practices. You may "
              "notice a severe performance degradation. Warning string: %s",
              context->get_task_name(), context->get_unique_id(),
              (warning_string == nullptr) ? "" : warning_string)
      }
      // If this physical region isn't mapped, then we have to
      // map it before we can return an accessor
      if (!mapped)
      {
        if (virtual_mapped)
          REPORT_LEGION_ERROR(
              ERROR_ILLEGAL_IMPLICIT_MAPPING,
              "Illegal implicit mapping of a virtual mapped region "
              "in task %s (UID %lld)",
              context->get_task_name(), context->get_unique_id())
        if (runtime->runtime_warnings && !silence_warnings)
          REPORT_LEGION_WARNING(
              LEGION_WARNING_UNMAPPED_ACCESSOR,
              "Accessor construction was "
              "performed on an unmapped region in task %s "
              "(UID %lld). Legion is mapping it for you. "
              "Please try to be more careful. Warning string: %s",
              context->get_task_name(), context->get_unique_id(),
              (warning_string == nullptr) ? "" : warning_string)
        context->remap_region(
            PhysicalRegion(this), nullptr /*prov*/, true /*internal*/);
        // At this point we should have a new ready event
        // and be mapped
#ifdef DEBUG_LEGION
        assert(mapped);
#endif
      }
      if (req.privilege_fields.find(fid) == req.privilege_fields.end())
        REPORT_LEGION_ERROR(
            ERROR_INVALID_FIELD_PRIVILEGES,
            "Accessor construction for field %d in task %s "
            "without privileges!",
            fid, context->get_task_name())
      if (generic_accessor && runtime->runtime_warnings && !silence_warnings)
        REPORT_LEGION_WARNING(
            LEGION_WARNING_GENERIC_ACCESSOR,
            "Using a generic accessor for accessing a "
            "physical instance of task %s (UID %lld). "
            "Generic accessors are very slow and are "
            "strongly discouraged for use in high "
            "performance code. Warning string: %s",
            context->get_task_name(), context->get_unique_id(),
            (warning_string == nullptr) ? "" : warning_string)
      // Get the index space to use for the accessor
      IndexSpaceNode* bounds = runtime->get_node(req.region.get_index_space());
      // Check to see if this is a padded field, if it is then we need to
      // merge the padding into the resulting domain that is allowed
      // to be accessed by the accessor for bounds checks
      bool need_padded_bounds = false;
      if (!std::binary_search(padded_fields.begin(), padded_fields.end(), fid))
        bounds->get_index_space_domain(realm_is, type_tag);
      else
        need_padded_bounds = true;
      // Wait until we are valid before returning the accessor
      wait_until_valid(
          silence_warnings, warning_string, runtime->runtime_warnings,
          "Accessor Construction");
      made_accessor = true;
      const InstanceSet& instances = references;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const InstanceRef& ref = instances[idx];
        if (ref.is_field_set(fid))
        {
          PhysicalManager* manager = ref.get_physical_manager();
          if (check_field_size)
          {
            const size_t actual_size =
                manager->field_space_node->get_field_size(fid);
            if (actual_size != field_size)
              REPORT_LEGION_ERROR(
                  ERROR_ACCESSOR_FIELD_SIZE_CHECK,
                  "Error creating accessor for field %d with a "
                  "type of size %zd bytes when the field was "
                  "originally allocated with a size of %zd bytes "
                  "in task %s (UID %lld)",
                  fid, field_size, actual_size, context->get_task_name(),
                  context->get_unique_id())
          }
          if (need_padded_bounds)
          {
            Domain domain = bounds->get_tight_domain();
#ifdef DEBUG_LEGION
            assert(domain.dense());
#endif
            // Do not add padding to empty domains
            if (!domain.empty())
            {
              // Now we can compute the bounds on this instance
              const Domain& delta =
                  manager->layout->constraints->padding_constraint.delta;
#ifdef DEBUG_LEGION
              assert(domain.get_dim() == delta.get_dim());
#endif
              const Domain padded_bounds =
                  Domain(domain.lo() - delta.lo(), domain.hi() + delta.hi());
              switch (domain.get_dim())
              {
#define DIMFUNC(DIM)                                               \
  case DIM:                                                        \
    {                                                              \
      RealmSpaceConverter<DIM, Realm::DIMTYPES>::convert_to(       \
          padded_bounds, realm_is, type_tag, "get_instance_info"); \
      break;                                                       \
    }
                LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
                default:
                  std::abort();
              }
            }
            else
              bounds->get_index_space_domain(realm_is, type_tag);
          }
          return manager->get_instance();
        }
      }
      // should never get here at worst there should have been an
      // error raised earlier in this function
      std::abort();
    }

    //--------------------------------------------------------------------------
    PhysicalInstance PhysicalRegionImpl::get_padding_info(
        FieldID fid, size_t field_size, Domain* inner, Domain& outer,
        const char* warning_string, bool silence_warnings,
        bool generic_accessor, bool check_field_size)
    //--------------------------------------------------------------------------
    {
      if (!std::binary_search(padded_fields.begin(), padded_fields.end(), fid))
        REPORT_LEGION_ERROR(
            ERROR_INVALID_PADDED_ACCESSOR,
            "Illegal request to create a padded accessor for field %d in "
            "parent task %s (UID %lld) which does not have padded privileges. "
            "You must record a layout constraint with an explicit for padding "
            "constraint when registering this task variant in order to be able "
            "to access the padded space on this instance.",
            fid, context->get_task_name(), context->get_unique_id())
      if (context != nullptr)
      {
        if (context->is_inner_context())
          REPORT_LEGION_ERROR(
              ERROR_INNER_TASK_VIOLATION,
              "Illegal padding accessor construction inside "
              "task %s (UID %lld) for a variant that was labeled as an 'inner' "
              "variant.",
              context->get_task_name(), context->get_unique_id())
        else if (
            runtime->runtime_warnings && !silence_warnings &&
            !context->is_leaf_context())
          REPORT_LEGION_WARNING(
              LEGION_WARNING_NONLEAF_ACCESSOR,
              "Padding ccessor construction in non-leaf "
              "task %s (UID %lld) is a blocking operation in violation of "
              "Legion's deferred execution model best practices. You may "
              "notice a severe performance degradation. Warning string: %s",
              context->get_task_name(), context->get_unique_id(),
              (warning_string == nullptr) ? "" : warning_string)
      }
      if (req.privilege_fields.find(fid) == req.privilege_fields.end())
        REPORT_LEGION_ERROR(
            ERROR_INVALID_FIELD_PRIVILEGES,
            "Padding accessor construction for field %d in task %s "
            "without privileges!",
            fid, context->get_task_name())
      if (generic_accessor && runtime->runtime_warnings && !silence_warnings)
        REPORT_LEGION_WARNING(
            LEGION_WARNING_GENERIC_ACCESSOR,
            "Using a generic accessor for accessing a "
            "physical instance of task %s (UID %lld). "
            "Generic accessors are very slow and are "
            "strongly discouraged for use in high "
            "performance code. Warning string: %s",
            context->get_task_name(), context->get_unique_id(),
            (warning_string == nullptr) ? "" : warning_string)
      const InstanceSet& instances = references;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const InstanceRef& ref = instances[idx];
        if (ref.is_field_set(fid))
        {
          PhysicalManager* manager = ref.get_physical_manager();
          if (check_field_size)
          {
            const size_t actual_size =
                manager->field_space_node->get_field_size(fid);
            if (actual_size != field_size)
              REPORT_LEGION_ERROR(
                  ERROR_ACCESSOR_FIELD_SIZE_CHECK,
                  "Error creating accessor for field %d with a "
                  "type of size %zd bytes when the field was "
                  "originally allocated with a size of %zd bytes "
                  "in task %s (UID %lld)",
                  fid, field_size, actual_size, context->get_task_name(),
                  context->get_unique_id())
          }
          // If this is a padded instance, then we know that this is an affine
          // instance so we can get it's index space expression and it should
          // be dense so then we can just add the offsets
          Domain bounds = manager->instance_domain->get_tight_domain();
#ifdef DEBUG_LEGION
          assert(bounds.dense());
#endif
          if (inner != nullptr)
            *inner = bounds;
          if (!bounds.empty())
          {
            // Now we can compute the bounds on this instance
            const Domain& delta =
                manager->layout->constraints->padding_constraint.delta;
#ifdef DEBUG_LEGION
            assert(bounds.get_dim() == delta.get_dim());
#endif
            outer = Domain(bounds.lo() - delta.lo(), bounds.hi() + delta.hi());
          }
          else
            outer = bounds;
          return manager->get_instance();
        }
      }
      // should never get here at worst there should have been an
      // error raised earlier in this function
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::report_incompatible_accessor(
        const char* accessor_kind, PhysicalInstance instance, FieldID fid)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(
          ERROR_ACCESSOR_COMPATIBILITY_CHECK,
          "Unable to create Realm %s for field %d of instance %llx in task %s",
          accessor_kind, fid, instance.id, context->get_task_name())
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::report_incompatible_multi_accessor(
        unsigned index, FieldID fid, PhysicalInstance inst1,
        PhysicalInstance inst2)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(
          ERROR_ACCESSOR_COMPATIBILITY_CHECK,
          "Unable to create multi-region accessor for field %d because "
          "instances " IDFMT " (index 0) and " IDFMT
          " (index %d) are "
          "differnt. Multi-region accessors must always be for region "
          "requirements with the same physical instance.",
          fid, inst1.id, inst2.id, index)
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::report_colocation_violation(
        const char* accessor_kind, FieldID fid, PhysicalInstance inst1,
        PhysicalInstance inst2, const PhysicalRegion& other, bool reduction)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(
          ERROR_COLOCATION_VIOLATION,
          "Unable to create co-location %s<%s> from multiple physical regions "
          "for field %d in task %s because regions have different physical "
          "instances " IDFMT " and  " IDFMT,
          reduction ? "ReductionAccessor" : "FieldAccessor", accessor_kind, fid,
          context->get_task_name(), inst1.id, inst2.id)
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalRegionImpl::empty_colocation_regions(
        const char* accessor_kind, FieldID fid, bool reduction)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(
          ERROR_COLOCATION_VIOLATION,
          "Attempt to create colocation %s<%s> with no physical regions for "
          "field %d task %s. Must provide a non-empty set of regions.",
          reduction ? "ReductionAccessor" : "FieldAccessor", accessor_kind, fid,
          implicit_context->get_task_name())
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalRegionImpl::fail_bounds_check(
        DomainPoint p, FieldID fid, PrivilegeMode mode, bool multi)
    //--------------------------------------------------------------------------
    {
      char point_string[128];
      snprintf(point_string, sizeof point_string, " (");
      for (int d = 0; d < p.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          snprintf(buffer, sizeof buffer, "%lld", p[0]);
        else
          snprintf(buffer, sizeof buffer, ",%lld", p[d]);
        strcat(point_string, buffer);
      }
      strcat(point_string, ")");
      switch (mode)
      {
        case LEGION_READ_ONLY:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_BOUNDS_CHECK,
                "Bounds check failure reading point %s from "
                "field %d in task %s%s\n",
                point_string, fid, implicit_context->get_task_name(),
                multi ? " for multi-region accessor" : "")
            break;
          }
        case LEGION_READ_WRITE:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_BOUNDS_CHECK,
                "Bounds check failure geting a reference to point %s "
                "from field %d in task %s%s\n",
                point_string, fid, implicit_context->get_task_name(),
                multi ? " for multi-region accessor" : "")
            break;
          }
        case LEGION_WRITE_ONLY:
        case LEGION_WRITE_DISCARD:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_BOUNDS_CHECK,
                "Bounds check failure writing to point %s in "
                "field %d in task %s%s\n",
                point_string, fid, implicit_context->get_task_name(),
                multi ? " for multi-region accessor" : "")
            break;
          }
        case LEGION_REDUCE:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_BOUNDS_CHECK,
                "Bounds check failure reducing to point %s in "
                "field %d in task %s%s\n",
                point_string, fid, implicit_context->get_task_name(),
                multi ? " for multi-region accessor" : "")
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalRegionImpl::fail_bounds_check(
        Domain dom, FieldID fid, PrivilegeMode mode, bool multi)
    //--------------------------------------------------------------------------
    {
      char rect_string[256];
      snprintf(rect_string, sizeof rect_string, " (");
      for (int d = 0; d < dom.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          snprintf(buffer, sizeof buffer, "%lld", dom.lo()[0]);
        else
          snprintf(buffer, sizeof buffer, ",%lld", dom.lo()[d]);
        strcat(rect_string, buffer);
      }
      strcat(rect_string, ") - (");
      for (int d = 0; d < dom.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          snprintf(buffer, sizeof buffer, "%lld", dom.hi()[0]);
        else
          snprintf(buffer, sizeof buffer, ",%lld", dom.hi()[d]);
        strcat(rect_string, buffer);
      }
      strcat(rect_string, ")");
      switch (mode)
      {
        case LEGION_READ_ONLY:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_BOUNDS_CHECK,
                "Bounds check failure getting a read-only reference "
                "to rect %s from field %d in task %s%s\n",
                rect_string, fid, implicit_context->get_task_name(),
                multi ? " for multi-region accessor" : "")
            break;
          }
        case LEGION_READ_WRITE:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_BOUNDS_CHECK,
                "Bounds check failure geting a reference to rect %s "
                "from field %d in task %s%s\n",
                rect_string, fid, implicit_context->get_task_name(),
                multi ? " for multi-region accessor" : "")
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalRegionImpl::fail_privilege_check(
        DomainPoint p, FieldID fid, PrivilegeMode mode)
    //--------------------------------------------------------------------------
    {
      char point_string[128];
      snprintf(point_string, sizeof point_string, " (");
      for (int d = 0; d < p.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          snprintf(buffer, sizeof buffer, "%lld", p[0]);
        else
          snprintf(buffer, sizeof buffer, ",%lld", p[d]);
        strcat(point_string, buffer);
      }
      strcat(point_string, ")");
      switch (mode)
      {
        case LEGION_READ_ONLY:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_PRIVILEGE_CHECK,
                "Privilege check failure reading point %s from "
                "field %d in task %s\n",
                point_string, fid, implicit_context->get_task_name())
            break;
          }
        case LEGION_READ_WRITE:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_PRIVILEGE_CHECK,
                "Privilege check failure geting a reference to point "
                "%s from field %d in task %s\n",
                point_string, fid, implicit_context->get_task_name())
            break;
          }
        case LEGION_WRITE_ONLY:
        case LEGION_WRITE_DISCARD:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_PRIVILEGE_CHECK,
                "Privilege check failure writing to point %s in "
                "field %d in task %s\n",
                point_string, fid, implicit_context->get_task_name())
            break;
          }
        case LEGION_REDUCE:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_PRIVILEGE_CHECK,
                "Privilege check failure reducing to point %s in "
                "field %d in task %s\n",
                point_string, fid, implicit_context->get_task_name())
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalRegionImpl::fail_privilege_check(
        Domain dom, FieldID fid, PrivilegeMode mode)
    //--------------------------------------------------------------------------
    {
      char rect_string[256];
      snprintf(rect_string, sizeof rect_string, " (");
      for (int d = 0; d < dom.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          snprintf(buffer, sizeof buffer, "%lld", dom.lo()[0]);
        else
          snprintf(buffer, sizeof buffer, ",%lld", dom.lo()[d]);
        strcat(rect_string, buffer);
      }
      strcat(rect_string, ") - (");
      for (int d = 0; d < dom.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          snprintf(buffer, sizeof buffer, "%lld", dom.hi()[0]);
        else
          snprintf(buffer, sizeof buffer, ",%lld", dom.hi()[d]);
        strcat(rect_string, buffer);
      }
      strcat(rect_string, ")");
      switch (mode)
      {
        case LEGION_READ_ONLY:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_PRIVILEGE_CHECK,
                "Privilege check failure getting a read-only "
                "reference to rect %s from field %d in task %s\n",
                rect_string, fid, implicit_context->get_task_name())
            break;
          }
        case LEGION_READ_WRITE:
          {
            REPORT_LEGION_ERROR(
                ERROR_ACCESSOR_PRIVILEGE_CHECK,
                "Privilege check failure geting a reference to rect "
                "%s from field %d in task %s\n",
                rect_string, fid, implicit_context->get_task_name())
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalRegionImpl::fail_padding_check(
        DomainPoint p, FieldID fid)
    //--------------------------------------------------------------------------
    {
      char point_string[128];
      snprintf(point_string, sizeof point_string, " (");
      for (int d = 0; d < p.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          snprintf(buffer, sizeof buffer, "%lld", p[0]);
        else
          snprintf(buffer, sizeof buffer, ",%lld", p[d]);
        strcat(point_string, buffer);
      }
      strcat(point_string, ")");
      REPORT_LEGION_ERROR(
          ERROR_ACCESSOR_BOUNDS_CHECK,
          "Bounds check failure accessing padded point %s from "
          "field %d in task %s\n",
          point_string, fid, implicit_context->get_task_name())
    }

    /////////////////////////////////////////////////////////////
    // ExternalResourcesImpl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalResourcesImpl::ExternalResourcesImpl(
        InnerContext* ctx, size_t num_regions, RegionTreeNode* upper,
        IndexSpaceNode* launch, LogicalRegion par,
        const std::set<FieldID>& fields)
      : context(ctx), upper_bound(upper), launch_bounds(launch),
        privilege_fields(fields.begin(), fields.end()), parent(par), pid(0),
        detached(false)
    //--------------------------------------------------------------------------
    {
      regions.resize(num_regions);
      upper_bound->add_base_resource_ref(PHYSICAL_REGION_REF);
      launch_bounds->add_base_resource_ref(PHYSICAL_REGION_REF);
    }

    //--------------------------------------------------------------------------
    ExternalResourcesImpl::~ExternalResourcesImpl(void)
    //--------------------------------------------------------------------------
    {
      if (upper_bound->remove_base_resource_ref(PHYSICAL_REGION_REF))
        delete upper_bound;
      if (launch_bounds->remove_base_resource_ref(PHYSICAL_REGION_REF))
        delete launch_bounds;
    }

    //--------------------------------------------------------------------------
    size_t ExternalResourcesImpl::size(void) const
    //--------------------------------------------------------------------------
    {
      return regions.size();
    }

    //--------------------------------------------------------------------------
    void ExternalResourcesImpl::set_region(
        unsigned index, PhysicalRegionImpl* region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < regions.size());
      assert(regions[index].impl == nullptr);
#endif
      regions[index] = PhysicalRegion(region);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion ExternalResourcesImpl::get_region(unsigned index) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < regions.size());
#endif
      return regions[index];
    }

    //--------------------------------------------------------------------------
    void ExternalResourcesImpl::set_projection(ProjectionID id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(pid == 0);
#endif
      pid = id;
    }

    //--------------------------------------------------------------------------
    Future ExternalResourcesImpl::detach(
        InnerContext* ctx, IndexDetachOp* op, const bool flush,
        const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (ctx != context)
        REPORT_LEGION_ERROR(
            ERROR_INDEX_SPACE_DETACH,
            "Attempted detach of external resources in context of task %s "
            "(UID %lld). Detach of external resources must always be performed "
            "in the the context of the task in which they are attached.",
            ctx->get_task_name(), ctx->get_unique_id())
      if (detached)
        REPORT_LEGION_ERROR(
            ERROR_INDEX_SPACE_DETACH,
            "Duplicate detach of external resources performed in task %s "
            "(UID %lld). External resources should only be detached once.",
            ctx->get_task_name(), ctx->get_unique_id())
      detached = true;
      // Unmap any mapped regions
      for (std::vector<PhysicalRegion>::iterator it = regions.begin();
           it != regions.end(); it++)
      {
        if (!it->impl->is_mapped())
          continue;
        it->impl->unmap_region();
        ctx->unregister_inline_mapped_region(*it);
      }
      // Now initialize the detach operation
      return op->initialize_detach(
          ctx, parent, upper_bound, launch_bounds, this, privilege_fields,
          regions, flush, unordered, provenance);
    }

  }  // namespace Internal
}  // namespace Legion
