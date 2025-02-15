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

#include "legion/views/logical.h"
#include "legion/kernel/runtime.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // LogicalView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalView::LogicalView(DistributedID did,
                             bool register_now, CollectiveMapping *map)
      : DistributedCollectable(did, register_now, map),
        valid_references(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalView::~LogicalView(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(valid_references == 0);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalView::handle_view_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      LogicalView *view = dynamic_cast<LogicalView*>(dc);
      assert(view != NULL);
#else
      LogicalView *view = static_cast<LogicalView*>(dc);
#endif
      if (view->collective_mapping != NULL)
      {
        // Check to see if this is a collective view, if the target
        // is in the replicated set, then there's nothing we need to do
        // We can just ignore this and the registration will be done later
        if (view->collective_mapping->contains(source))
          return;
        AddressSpaceID nearest = view->collective_mapping->find_nearest(source);
        if (nearest != runtime->address_space)
        {
          // Forward this on to the nearest space in the collective mapping
          Serializer rez;
          {
            RezCheck z2(rez);
            rez.serialize(did);
            rez.serialize(source);
          }
          runtime->send_view_request(nearest, rez);
          return;
        }
      }
      view->send_view(source);
    }

#ifdef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void LogicalView::add_base_valid_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      valid_references += cnt;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_valid_references.find(source);
      if (finder == detailed_base_valid_references.end())
        detailed_base_valid_references[source] = cnt;
      else
        finder->second += cnt;
      if (valid_references == cnt)
        notify_valid();
    }

    //--------------------------------------------------------------------------
    void LogicalView::add_nested_valid_ref_internal(
                                                  DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      valid_references += cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_valid_references.find(source);
      if (finder == detailed_nested_valid_references.end())
        detailed_nested_valid_references[source] = cnt;
      else
        finder->second += cnt;
      if (valid_references == cnt)
        notify_valid();
    }

    //--------------------------------------------------------------------------
    bool LogicalView::remove_base_valid_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
#ifdef DEBUG_LEGION
      assert(valid_references >= cnt);
#endif
      valid_references -= cnt;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_valid_references.find(source);
      assert(finder != detailed_base_valid_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_base_valid_references.erase(finder);
      if (valid_references == 0)
        return notify_invalid();
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool LogicalView::remove_nested_valid_ref_internal(
                                                  DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
#ifdef DEBUG_LEGION
      assert(valid_references >= cnt);
#endif
      valid_references -= cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_valid_references.find(source);
      assert(finder != detailed_nested_valid_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_nested_valid_references.erase(finder);
      if (valid_references == 0)
        return notify_invalid();
      else
        return false;
    }
#else // DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void LogicalView::add_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      if (valid_references.fetch_add(cnt) == 0)
        notify_valid();
    }

    //--------------------------------------------------------------------------
    bool LogicalView::remove_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
#ifdef DEBUG_LEGION
      assert(valid_references.load() >= cnt);
#endif
      if (valid_references.fetch_sub(cnt) == cnt)
        return notify_invalid();
      else
        return false;
    }
#endif // DEBUG_LEGION_GC

  } // namespace Internal
} // namespace Legion
