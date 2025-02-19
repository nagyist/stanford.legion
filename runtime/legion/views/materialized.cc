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

#include "legion/views/materialized.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // MaterializedView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(
        DistributedID did, AddressSpaceID log_own, PhysicalManager* man,
        bool register_now, CollectiveMapping* mapping)
      : IndividualView(
            encode_materialized_did(did), man, log_own, register_now, mapping)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info(
          "GC Materialized View %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space,
          LEGION_DISTRIBUTED_ID_FILTER(manager->did));
#endif
    }

    //--------------------------------------------------------------------------
    MaterializedView::~MaterializedView(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    const FieldMask& MaterializedView::get_physical_mask(void) const
    //--------------------------------------------------------------------------
    {
      return manager->layout->allocated_fields;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::has_space(const FieldMask& space_mask) const
    //--------------------------------------------------------------------------
    {
      return !(space_mask - manager->layout->allocated_fields);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(manager->did);
        rez.serialize(owner_space);
        rez.serialize(logical_owner);
      }
      runtime->send_materialized_view(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_materialized_view(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      RtEvent man_ready, ctx_ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(manager_did, man_ready);
      if (man_ready.exists() && !man_ready.has_triggered())
      {
        // Defer this until the manager is ready
        DeferMaterializedViewArgs args(did, manager, logical_owner);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_RESPONSE_PRIORITY, man_ready);
      }
      else
        create_remote_view(did, manager, logical_owner);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_defer_materialized_view(
        const void* args)
    //--------------------------------------------------------------------------
    {
      const DeferMaterializedViewArgs* dargs =
          (const DeferMaterializedViewArgs*)args;
      create_remote_view(dargs->did, dargs->manager, dargs->logical_owner);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::create_remote_view(
        DistributedID did, PhysicalManager* manager,
        AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager->is_physical_manager());
#endif
      PhysicalManager* inst_manager = manager->as_physical_manager();
      void* location =
          runtime
              ->find_or_create_pending_collectable_location<MaterializedView>(
                  did);
      MaterializedView* view = new (location) MaterializedView(
          did, logical_owner, inst_manager, false /*register now*/);
      // Register only after construction
      view->register_with_runtime();
    }

  }  // namespace Internal
}  // namespace Legion
