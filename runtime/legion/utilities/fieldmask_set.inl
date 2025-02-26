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

// Included from fieldmask_set.h - do not include this directly

// Useful for IDEs
#include "legion/utilities/fieldmask_set.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    template<typename T>
    inline void compute_field_sets(
        FieldMask universe_mask, const MapView<T, FieldMask>& inputs,
        local::list<FieldSet<T> >& output_sets)
    //--------------------------------------------------------------------------
    {
      // Special cases for empty and size 1 sets
      if (inputs.empty())
      {
        if (!!universe_mask)
          output_sets.emplace_back(FieldSet<T>(universe_mask));
        return;
      }
      else if (inputs.size() == 1)
      {
        typename MapView<T, FieldMask>::const_iterator first = inputs.begin();
        output_sets.emplace_back(FieldSet<T>(first->second));
        FieldSet<T>& last = output_sets.back();
        last.elements.insert(first->first);
        if (!!universe_mask)
        {
          universe_mask -= first->second;
          if (!!universe_mask)
            output_sets.emplace_back(FieldSet<T>(universe_mask));
        }
        return;
      }
      for (typename MapView<T, FieldMask>::const_iterator pit = inputs.begin();
           pit != inputs.end(); pit++)
      {
        bool inserted = false;
        // Also keep track of which fields have updates
        // but don't have any members
        if (!!universe_mask)
          universe_mask -= pit->second;
        FieldMask remaining = pit->second;
        // Insert this event into the precondition sets
        for (typename local::list<FieldSet<T> >::iterator it =
                 output_sets.begin();
             it != output_sets.end(); it++)
        {
          // Easy case, check for equality
          if (remaining == it->set_mask)
          {
            it->elements.insert(pit->first);
            inserted = true;
            break;
          }
          FieldMask overlap = remaining & it->set_mask;
          // Easy case, they are disjoint so keep going
          if (!overlap)
            continue;
          // Moderate case, we are dominated, split into two sets
          // reusing existing set and making a new set
          if (overlap == remaining)
          {
            // Leave the existing set and make it the difference
            it->set_mask -= overlap;
            output_sets.emplace_back(FieldSet<T>(overlap));
            FieldSet<T>& last = output_sets.back();
            last.elements = it->elements;
            last.elements.insert(pit->first);
            inserted = true;
            break;
          }
          // Moderate case, we dominate the existing set
          if (overlap == it->set_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->elements.insert(pit->first);
            remaining -= overlap;
            // Can't consider ourselves added yet
            continue;
          }
          // Hard case, neither dominates, compute three
          // distinct sets of fields, keep left one in
          // place and reduce scope, add new one at the
          // end for overlap, continue iterating for right one
          it->set_mask -= overlap;
          const local::set<T>& temp_elements = it->elements;
          it = output_sets.insert(it, FieldSet<T>(overlap));
          it->elements = temp_elements;
          it->elements.insert(pit->first);
          remaining -= overlap;
          continue;
        }
        if (!inserted)
        {
          output_sets.emplace_back(FieldSet<T>(remaining));
          FieldSet<T>& last = output_sets.back();
          last.elements.insert(pit->first);
        }
      }
      // For any fields which need copies but don't have
      // any elements, but them in their own set.
      // Put it on the front because it is the copy with
      // no elements so it can start right away!
      if (!!universe_mask)
        output_sets.emplace_front(FieldSet<T>(universe_mask));
    }

    // This is a generalization of the above method but takes a list of
    // anything that has the same members as a FieldSet
    //--------------------------------------------------------------------------
    template<typename T, typename CT>
    inline void compute_field_sets(
        FieldMask universe_mask, const MapView<T, FieldMask>& inputs,
        local::list<CT>& output_sets)
    //--------------------------------------------------------------------------
    {
      // Special cases for empty and size 1 sets
      if (inputs.empty())
      {
        if (!!universe_mask)
          output_sets.emplace_back(CT(universe_mask));
        return;
      }
      else if (inputs.size() == 1)
      {
        typename MapView<T, FieldMask>::const_iterator first = inputs.begin();
        output_sets.emplace_back(CT(first->second));
        CT& last = output_sets.back();
        last.elements.insert(first->first);
        if (!!universe_mask)
        {
          universe_mask -= first->second;
          if (!!universe_mask)
            output_sets.emplace_back(CT(universe_mask));
        }
        return;
      }
      for (typename MapView<T, FieldMask>::const_iterator pit = inputs.begin();
           pit != inputs.end(); pit++)
      {
        bool inserted = false;
        // Also keep track of which fields have updates
        // but don't have any members
        if (!!universe_mask)
          universe_mask -= pit->second;
        FieldMask remaining = pit->second;
        // Insert this event into the precondition sets
        for (typename local::list<CT>::iterator it = output_sets.begin();
             it != output_sets.end(); it++)
        {
          // Easy case, check for equality
          if (remaining == it->set_mask)
          {
            it->elements.insert(pit->first);
            inserted = true;
            break;
          }
          FieldMask overlap = remaining & it->set_mask;
          // Easy case, they are disjoint so keep going
          if (!overlap)
            continue;
          // Moderate case, we are dominated, split into two sets
          // reusing existing set and making a new set
          if (overlap == remaining)
          {
            // Leave the existing set and make it the difference
            it->set_mask -= overlap;
            output_sets.emplace_back(CT(overlap));
            CT& last = output_sets.back();
            last.elements = it->elements;
            last.elements.insert(pit->first);
            inserted = true;
            break;
          }
          // Moderate case, we dominate the existing set
          if (overlap == it->set_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->elements.insert(pit->first);
            remaining -= overlap;
            // Can't consider ourselves added yet
            continue;
          }
          // Hard case, neither dominates, compute three
          // distinct sets of fields, keep left one in
          // place and reduce scope, add new one at the
          // end for overlap, continue iterating for right one
          it->set_mask -= overlap;
          const local::set<T>& temp_elements = it->elements;
          it = output_sets.insert(it, CT(overlap));
          it->elements = temp_elements;
          it->elements.insert(pit->first);
          remaining -= overlap;
          continue;
        }
        if (!inserted)
        {
          output_sets.emplace_back(CT(remaining));
          CT& last = output_sets.back();
          last.elements.insert(pit->first);
        }
      }
      // For any fields which need copies but don't have
      // any elements, but them in their own set.
      // Put it on the front because it is the copy with
      // no elements so it can start right away!
      if (!!universe_mask)
        output_sets.emplace_front(CT(universe_mask));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline FieldMaskSet<T, L, D>::FieldMaskSet(
        T* init, const FieldMask& mask, bool no_null)
      : single(true)
    //--------------------------------------------------------------------------
    {
      if (!no_null || (init != nullptr))
      {
        entries.single_entry = init;
        valid_fields = mask;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline FieldMaskSet<T, L, D>::FieldMaskSet(const FieldMaskSet<T, L, D>& rhs)
      : valid_fields(rhs.valid_fields), single(rhs.single)
    //--------------------------------------------------------------------------
    {
      if (single)
        entries.single_entry = rhs.entries.single_entry;
      else
        entries.multi_entries = new FSMap(
            rhs.entries.multi_entries->begin(),
            rhs.entries.multi_entries->end());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline FieldMaskSet<T, L, D>::FieldMaskSet(
        FieldMaskSet<T, L, D>&& rhs) noexcept
      : valid_fields(rhs.valid_fields), single(rhs.single)
    //--------------------------------------------------------------------------
    {
      if (single)
        entries.single_entry = rhs.entries.single_entry;
      else
        entries.multi_entries = rhs.entries.multi_entries;
      rhs.valid_fields.clear();
      rhs.single = true;
      rhs.entries.single_entry = nullptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline FieldMaskSet<T, L, D>::FieldMaskSet(
        FieldMaskSet<T, L, D>& rhs, bool copy)
      : valid_fields(rhs.valid_fields), single(rhs.single)
    //--------------------------------------------------------------------------
    {
      if (copy)
      {
        if (single)
          entries.single_entry = rhs.entries.single_entry;
        else
          entries.multi_entries = new FSMap(
              rhs.entries.multi_entries->begin(),
              rhs.entries.multi_entries->end());
      }
      else
      {
        if (single)
          entries.single_entry = rhs.entries.single_entry;
        else
          entries.multi_entries = rhs.entries.multi_entries;
        rhs.entries.single_entry = nullptr;
        rhs.valid_fields.clear();
        rhs.single = true;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline FieldMaskSet<T, L, D>& FieldMaskSet<T, L, D>::operator=(
        const FieldMaskSet<T, L, D>& rhs)
    //--------------------------------------------------------------------------
    {
      // Check our current state
      if (single != rhs.single)
      {
        // Different data structures
        if (single)
        {
          entries.multi_entries = new FSMap(
              rhs.entries.multi_entries->begin(),
              rhs.entries.multi_entries->end());
        }
        else
        {
          // Free our map
          delete entries.multi_entries;
          entries.single_entry = rhs.entries.single_entry;
        }
        single = rhs.single;
      }
      else
      {
        // Same data structures so we can just copy things over
        if (single)
          entries.single_entry = rhs.entries.single_entry;
        else
        {
          entries.multi_entries->clear();
          entries.multi_entries->insert(
              rhs.entries.multi_entries->begin(),
              rhs.entries.multi_entries->end());
        }
      }
      valid_fields = rhs.valid_fields;
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline FieldMaskSet<T, L, D>& FieldMaskSet<T, L, D>::operator=(
        FieldMaskSet<T, L, D>&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      // Check our current state
      if (single != rhs.single)
      {
        // Different data structures
        if (single)
        {
          entries.multi_entries = rhs.entries.multi_entries;
        }
        else
        {
          // Free our map
          delete entries.multi_entries;
          entries.single_entry = rhs.entries.single_entry;
        }
        single = rhs.single;
      }
      else
      {
        // Same data structures so we can just copy things over
        if (single)
        {
          entries.single_entry = rhs.entries.single_entry;
        }
        else
        {
          delete entries.multi_entries;
          entries.multi_entries = rhs.entries.multi_entries;
        }
      }
      valid_fields = rhs.valid_fields;
      rhs.valid_fields.clear();
      rhs.single = true;
      rhs.entries.single_entry = nullptr;
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline const FieldMask& FieldMaskSet<T, L, D>::tighten_valid_mask(void)
    //--------------------------------------------------------------------------
    {
      // If we're single then there is nothing to do as we're already tight
      if (single)
        return valid_fields;
      valid_fields.clear();
      for (typename FSMap::const_iterator it = entries.multi_entries->begin();
           it != entries.multi_entries->end(); it++)
        valid_fields |= it->second;
      return valid_fields;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::relax_valid_mask(const FieldMask& m)
    //--------------------------------------------------------------------------
    {
      if (single && (entries.single_entry != nullptr))
      {
        if (!(m - valid_fields))
          return;
        // have to avoid the aliasing case
        T* entry = entries.single_entry;
        entries.multi_entries = new FSMap();
        entries.multi_entries->insert(std::make_pair(entry, valid_fields));
        single = false;
      }
      valid_fields |= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::filter_valid_mask(const FieldMask& m)
    //--------------------------------------------------------------------------
    {
      valid_fields -= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::restrict_valid_mask(const FieldMask& m)
    //--------------------------------------------------------------------------
    {
      valid_fields &= m;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline const FieldMask& FieldMaskSet<T, L, D>::operator[](T* entry) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(entry == entries.single_entry);
#endif
        return valid_fields;
      }
      else
      {
        typename FSMap::const_iterator finder =
            entries.multi_entries->find(entry);
#ifdef DEBUG_LEGION
        assert(finder != entries.multi_entries->end());
#endif
        return finder->second;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline bool FieldMaskSet<T, L, D>::insert(T* entry, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      if (single)
      {
        if (entries.single_entry == nullptr)
        {
          entries.single_entry = entry;
          valid_fields |= mask;
        }
        else if (entries.single_entry == entry)
        {
          valid_fields |= mask;
          result = false;
        }
        else
        {
          // Go to multi
          FSMap* multi = new FSMap();
          (*multi)[entries.single_entry] = valid_fields;
          (*multi)[entry] = mask;
          entries.multi_entries = multi;
          single = false;
          valid_fields |= mask;
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(entries.multi_entries != nullptr);
#endif
        typename FSMap::iterator finder = entries.multi_entries->find(entry);
        if (finder == entries.multi_entries->end())
          (*entries.multi_entries)[entry] = mask;
        else
        {
          finder->second |= mask;
          result = false;
        }
        valid_fields |= mask;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::filter(
        const FieldMask& filter, bool tighten)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (entries.single_entry != nullptr)
        {
          if (tighten)
            valid_fields -= filter;
          if (!valid_fields)
            entries.single_entry = nullptr;
        }
      }
      else
      {
        if (tighten)
          valid_fields -= filter;
        if (!valid_fields || (!tighten && (filter == valid_fields)))
        {
          // No fields left so just clean everything up
          delete entries.multi_entries;
          entries.multi_entries = nullptr;
          single = true;
        }
        else
        {
          // Manually remove entries
          typename std::vector<T*> to_delete;
          for (typename FSMap::iterator it = entries.multi_entries->begin();
               it != entries.multi_entries->end(); it++)
          {
            it->second -= filter;
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          if (!to_delete.empty())
          {
            if (to_delete.size() < entries.multi_entries->size())
            {
              for (typename std::vector<T*>::const_iterator it =
                       to_delete.begin();
                   it != to_delete.end(); it++)
                entries.multi_entries->erase(*it);
              if (entries.multi_entries->empty())
              {
                delete entries.multi_entries;
                entries.multi_entries = nullptr;
                single = true;
              }
              else if (
                  (entries.multi_entries->size() == 1) &&
                  (entries.multi_entries->begin()->second == valid_fields))
              {
                typename FSMap::iterator last = entries.multi_entries->begin();
                T* temp = last->first;
                delete entries.multi_entries;
                entries.single_entry = temp;
                single = true;
              }
            }
            else
            {
              delete entries.multi_entries;
              entries.multi_entries = nullptr;
              single = true;
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::erase(T* to_erase)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(entries.single_entry == to_erase);
#endif
        entries.single_entry = nullptr;
        valid_fields.clear();
      }
      else
      {
        typename FSMap::iterator finder = entries.multi_entries->find(to_erase);
#ifdef DEBUG_LEGION
        assert(finder != entries.multi_entries->end());
#endif
        entries.multi_entries->erase(finder);
        if (entries.multi_entries->size() == 1)
        {
          // go back to single
          finder = entries.multi_entries->begin();
          valid_fields = finder->second;
          T* first = finder->first;
          delete entries.multi_entries;
          entries.single_entry = first;
          single = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::clear(void)
    //--------------------------------------------------------------------------
    {
      if (single)
        entries.single_entry = nullptr;
      else
      {
#ifdef DEBUG_LEGION
        assert(entries.multi_entries != nullptr);
#endif
        delete entries.multi_entries;
        entries.multi_entries = nullptr;
        single = true;
      }
      valid_fields.clear();
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline size_t FieldMaskSet<T, L, D>::size(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (entries.single_entry == nullptr)
          return 0;
        else
          return 1;
      }
      else
        return entries.multi_entries->size();
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::swap(FieldMaskSet& other)
    //--------------------------------------------------------------------------
    {
      // Just use single, doesn't matter for swap
      T* temp_entry = other.entries.single_entry;
      other.entries.single_entry = entries.single_entry;
      entries.single_entry = temp_entry;

      bool temp_single = other.single;
      other.single = single;
      single = temp_single;

      FieldMask temp_valid_fields = other.valid_fields;
      other.valid_fields = valid_fields;
      valid_fields = temp_valid_fields;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline typename FieldMaskSet<T, L, D>::iterator
        FieldMaskSet<T, L, D>::begin(void)
    //--------------------------------------------------------------------------
    {
      // Scariness!
      if (single)
      {
        // If we're empty return end
        if (entries.single_entry == nullptr)
          return end();
        FieldMaskSet<T, L, D>* ptr = this;
        std::pair<T* const, FieldMask>* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return iterator(this, result);
      }
      else
        return iterator(this, entries.multi_entries->begin());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline typename FieldMaskSet<T, L, D>::iterator FieldMaskSet<T, L, D>::find(
        T* e)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((entries.single_entry == nullptr) || (entries.single_entry != e))
          return end();
        FieldMaskSet<T, L, D>* ptr = this;
        std::pair<T* const, FieldMask>* result = nullptr;
        static_assert(sizeof(result) == sizeof(ptr));
        memcpy(&result, &ptr, sizeof(result));
        return iterator(this, result);
      }
      else
      {
        typename FSMap::iterator finder = entries.multi_entries->find(e);
        if (finder == entries.multi_entries->end())
          return end();
        return iterator(this, finder);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::erase(iterator& it)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(it != end());
#endif
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(entries.single_entry == it->first);
#endif
        entries.single_entry = nullptr;
        valid_fields.clear();
      }
      else
      {
        it.erase(*(entries.multi_entries));
        if (entries.multi_entries->size() == 1)
        {
          // go back to single
          typename FSMap::iterator finder = entries.multi_entries->begin();
          valid_fields = finder->second;
          T* first = finder->first;
          delete entries.multi_entries;
          entries.single_entry = first;
          single = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline typename FieldMaskSet<T, L, D>::iterator FieldMaskSet<T, L, D>::end(
        void)
    //--------------------------------------------------------------------------
    {
      if (single)
        return iterator(this, nullptr);
      else
        return iterator(this, entries.multi_entries->end(), true /*end*/);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline typename FieldMaskSet<T, L, D>::const_iterator
        FieldMaskSet<T, L, D>::begin(void) const
    //--------------------------------------------------------------------------
    {
      // Scariness!
      if (single)
      {
        // If we're empty return end
        if (entries.single_entry == nullptr)
          return end();
        FieldMaskSet<T, L, D>* ptr = const_cast<FieldMaskSet<T, L, D>*>(this);
        std::pair<T* const, FieldMask>* result = nullptr;
        static_assert(sizeof(ptr) == sizeof(result));
        memcpy(&result, &ptr, sizeof(result));
        return const_iterator(this, result);
      }
      else
        return const_iterator(this, entries.multi_entries->begin());
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline typename FieldMaskSet<T, L, D>::const_iterator
        FieldMaskSet<T, L, D>::find(T* e) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((entries.single_entry == nullptr) || (entries.single_entry != e))
          return end();
        FieldMaskSet<T, L, D>* ptr = const_cast<FieldMaskSet<T, L, D>*>(this);
        std::pair<T* const, FieldMask>* result = nullptr;
        static_assert(sizeof(ptr) == sizeof(result));
        memcpy(&result, &ptr, sizeof(result));
        return const_iterator(this, result);
      }
      else
      {
        typename FSMap::const_iterator finder = entries.multi_entries->find(e);
        if (finder == entries.multi_entries->end())
          return end();
        return const_iterator(this, finder);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline typename FieldMaskSet<T, L, D>::const_iterator
        FieldMaskSet<T, L, D>::end(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
        return const_iterator(this, nullptr);
      else
        return const_iterator(this, entries.multi_entries->end(), true /*end*/);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L, bool D>
    inline void FieldMaskSet<T, L, D>::compute_field_sets(
        FieldMask universe_mask, local::list<FieldSet<T*> >& output_sets) const
    //--------------------------------------------------------------------------
    {
      // Handle special cases for single entry and single fields
      if (empty())
      {
        if (!!universe_mask)
          output_sets.emplace_back(FieldSet<T*>(universe_mask));
        return;
      }
      else if (single)
      {
        output_sets.emplace_back(FieldSet<T*>(valid_fields));
        FieldSet<T*>& last = output_sets.back();
        last.elements.insert(entries.single_entry);
        if (!!universe_mask)
        {
          universe_mask -= valid_fields;
          if (!!universe_mask)
            output_sets.emplace_back(FieldSet<T*>(universe_mask));
        }
        return;
      }
      else if (valid_fields.pop_count() == 1)
      {
        output_sets.emplace_back(FieldSet<T*>(valid_fields));
        FieldSet<T*>& last = output_sets.back();
        bool has_empty = false;
        for (const_iterator pit = this->begin(); pit != this->end(); pit++)
        {
          if (!!pit->second)
            last.elements.insert(pit->first);
          else
            has_empty = true;
        }
        if (has_empty)
        {
          output_sets.emplace_back(FieldSet<T*>(FieldMask()));
          last = output_sets.back();
          for (const_iterator pit = this->begin(); pit != this->end(); pit++)
            if (!pit->second)
              last.elements.insert(pit->first);
        }
        if (!!universe_mask)
        {
          universe_mask -= valid_fields;
          if (!!universe_mask)
            output_sets.emplace_back(FieldSet<T*>(universe_mask));
        }
        return;
      }
      // Otherwise we fall through and do the full thing
      for (const_iterator pit = this->begin(); pit != this->end(); pit++)
      {
        bool inserted = false;
        // Also keep track of which fields have updates
        // but don't have any members
        if (!!universe_mask)
          universe_mask -= pit->second;
        FieldMask remaining = pit->second;
        // Insert this event into the precondition sets
        for (typename local::list<FieldSet<T*> >::iterator it =
                 output_sets.begin();
             it != output_sets.end(); it++)
        {
          // Easy case, check for equality
          if (remaining == it->set_mask)
          {
            it->elements.insert(pit->first);
            inserted = true;
            break;
          }
          FieldMask overlap = remaining & it->set_mask;
          // Easy case, they are disjoint so keep going
          if (!overlap)
            continue;
          // Moderate case, we are dominated, split into two sets
          // reusing existing set and making a new set
          if (overlap == remaining)
          {
            // Leave the existing set and make it the difference
            it->set_mask -= overlap;
            output_sets.emplace_back(FieldSet<T*>(overlap));
            FieldSet<T*>& last = output_sets.back();
            last.elements = it->elements;
            last.elements.insert(pit->first);
            inserted = true;
            break;
          }
          // Moderate case, we dominate the existing set
          if (overlap == it->set_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->elements.insert(pit->first);
            remaining -= overlap;
            // Can't consider ourselves added yet
            continue;
          }
          // Hard case, neither dominates, compute three
          // distinct sets of fields, keep left one in
          // place and reduce scope, add new one at the
          // end for overlap, continue iterating for right one
          it->set_mask -= overlap;
          const local::set<T*>& temp_elements = it->elements;
          it = output_sets.insert(it, FieldSet<T*>(overlap));
          it->elements = temp_elements;
          it->elements.insert(pit->first);
          remaining -= overlap;
          continue;
        }
        if (!inserted)
        {
          output_sets.emplace_back(FieldSet<T*>(remaining));
          FieldSet<T*>& last = output_sets.back();
          last.elements.insert(pit->first);
        }
      }
      // For any fields which need copies but don't have
      // any elements, but them in their own set.
      // Put it on the front because it is the copy with
      // no elements so it can start right away!
      if (!!universe_mask)
        output_sets.emplace_front(FieldSet<T*>(universe_mask));
    }

    //--------------------------------------------------------------------------
    template<typename T1, typename T2>
    inline void unique_join_on_field_mask_sets(
        const FieldMaskSet<T1>& left, const FieldMaskSet<T2>& right,
        local::map<std::pair<T1*, T2*>, FieldMask>& results)
    //--------------------------------------------------------------------------
    {
      if (left.empty() || right.empty())
        return;
      if (left.get_valid_mask() * right.get_valid_mask())
        return;
#ifdef DEBUG_LEGION
      FieldMask unique_test;
#endif
      if (left.size() == 1)
      {
        typename FieldMaskSet<T1>::const_iterator first = left.begin();
        for (typename FieldMaskSet<T2>::const_iterator it = right.begin();
             it != right.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          const FieldMask overlap = first->second & it->second;
          if (!overlap)
            continue;
          const std::pair<T1*, T2*> key(first->first, it->first);
          results[key] = overlap;
        }
        return;
      }
      if (right.size() == 1)
      {
        typename FieldMaskSet<T2>::const_iterator first = right.begin();
        for (typename FieldMaskSet<T1>::const_iterator it = left.begin();
             it != left.end(); it++)
        {
          const FieldMask overlap = first->second & it->second;
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          if (!overlap)
            continue;
          const std::pair<T1*, T2*> key(it->first, first->first);
          results[key] = overlap;
        }
        return;
      }
      // Build the lookup table for the one with fewer fields
      // since it is probably more costly to allocate memory
      if (left.get_valid_mask().pop_count() <
          right.get_valid_mask().pop_count())
      {
        // Build the hash table for left
        std::map<unsigned, T1*> hash_table;
        for (typename FieldMaskSet<T1>::const_iterator it = left.begin();
             it != left.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          int fidx = it->second.find_first_set();
          while (fidx >= 0)
          {
            hash_table[fidx] = it->first;
            fidx = it->second.find_next_set(fidx + 1);
          }
        }
#ifdef DEBUG_LEGION
        unique_test.clear();
#endif
        for (typename FieldMaskSet<T2>::const_iterator it = right.begin();
             it != right.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          int fidx = it->second.find_first_set();
          while (fidx >= 0)
          {
            typename std::map<unsigned, T1*>::const_iterator finder =
                hash_table.find(fidx);
            if (finder != hash_table.end())
            {
              const std::pair<T1*, T2*> key(finder->second, it->first);
              results[key].set_bit(fidx);
            }
            fidx = it->second.find_next_set(fidx + 1);
          }
        }
      }
      else
      {
        // Build the hash table for the right
        std::map<unsigned, T2*> hash_table;
        for (typename FieldMaskSet<T2>::const_iterator it = right.begin();
             it != right.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          int fidx = it->second.find_first_set();
          while (fidx >= 0)
          {
            hash_table[fidx] = it->first;
            fidx = it->second.find_next_set(fidx + 1);
          }
        }
#ifdef DEBUG_LEGION
        unique_test.clear();
#endif
        for (typename FieldMaskSet<T1>::const_iterator it = left.begin();
             it != left.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * unique_test);
          unique_test |= it->second;
#endif
          int fidx = it->second.find_first_set();
          while (fidx >= 0)
          {
            typename std::map<unsigned, T2*>::const_iterator finder =
                hash_table.find(fidx);
            if (finder != hash_table.end())
            {
              const std::pair<T1*, T2*> key(it->first, finder->second);
              results[key].set_bit(fidx);
            }
            fidx = it->second.find_next_set(fidx + 1);
          }
        }
      }
    }

  }  // namespace Internal
}  // namespace Legion
