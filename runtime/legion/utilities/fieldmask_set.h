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

#ifndef __LEGION_FIELDMASK_SET_H__
#define __LEGION_FIELDMASK_SET_H__

#include "legion/utilities/bitmask.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct FieldSet
     * A helper template class for the method below for describing
     * sets of members that all contain the same fields
     */
    template<typename T>
    struct FieldSet {
    public:
      FieldSet(void) { }
      FieldSet(const FieldMask& m) : set_mask(m) { }
    public:
      FieldMask set_mask;
      local::set<T> elements;
    };

    /**
     * \class FieldMaskSet
     * A template helper class for tracking collections of
     * objects associated with different sets of fields
     */
    template<
        typename T, AllocationLifetime L = TASK_LOCAL_LIFETIME,
        bool DETERMINISTIC = false>
    class FieldMaskSet : public Heapify<FieldMaskSet<T>, L> {
    private:
      // Call the deterministic pointer less method for
      // any types that have asked for deterministic sets
      template<typename U>
      struct DeterministicComparator {
      public:
        inline bool operator()(const U* one, const U* two) const
        {
          return one->deterministic_pointer_less(two);
        }
      };
      using Comparator = typename std::conditional<
          DETERMINISTIC, DeterministicComparator<T>,
          std::less<const T*> >::type;
      using FSMap = std::map<
          T*, FieldMask, Comparator,
          LegionAllocator<std::pair<T* const, FieldMask>, L> >;
    public:
      // forward declaration
      class const_iterator;
      class iterator {
      public:
        // explicitly set iterator traits
        typedef std::input_iterator_tag iterator_category;
        typedef std::pair<T* const, FieldMask> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::pair<T* const, FieldMask>* pointer;
        typedef std::pair<T* const, FieldMask>& reference;

        iterator(FieldMaskSet* _set, std::pair<T* const, FieldMask>* _result)
          : set(_set), result(_result), single(true)
        { }
        iterator(
            FieldMaskSet* _set, typename FSMap::iterator _it, bool end = false)
          : set(_set), result(end ? nullptr : &(*_it)), it(_it), single(false)
        { }
      public:
        iterator(const iterator& rhs)
          : set(rhs.set), result(rhs.result), it(rhs.it), single(rhs.single)
        { }
        ~iterator(void) { }
      public:
        inline iterator& operator=(const iterator& rhs)
        {
          set = rhs.set;
          result = rhs.result;
          it = rhs.it;
          single = rhs.single;
          return *this;
        }
      public:
        inline bool operator==(const iterator& rhs) const
        {
          if (set != rhs.set)
            return false;
          if (single)
            return (result == rhs.result);
          else
            return (it == rhs.it);
        }
        inline bool operator!=(const iterator& rhs) const
        {
          if (set != rhs.set)
            return true;
          if (single)
            return (result != rhs.result);
          else
            return (it != rhs.it);
        }
      public:
        inline const std::pair<T* const, FieldMask> operator*(void)
        {
          return *result;
        }
        inline const std::pair<T* const, FieldMask>* operator->(void)
        {
          return result;
        }
        inline iterator& operator++(/*prefix*/ void)
        {
          if (!single)
          {
            ++it;
            if ((*this) != set->end())
              result = &(*it);
            else
              result = nullptr;
          }
          else
            result = nullptr;
          return *this;
        }
        inline iterator operator++(/*postfix*/ int)
        {
          iterator copy(*this);
          if (!single)
          {
            ++it;
            if ((*this) != set->end())
              result = &(*it);
            else
              result = nullptr;
          }
          else
            result = nullptr;
          return copy;
        }
      public:
        inline operator bool(void) const { return (result != nullptr); }
      public:
        inline void merge(const FieldMask& mask)
        {
          result->second |= mask;
          if (!single)
            set->valid_fields |= mask;
        }
        inline void filter(const FieldMask& mask)
        {
          result->second -= mask;
          // Don't filter valid fields since its unsound
        }
        inline void clear(void) { result->second.clear(); }
      public:
        inline void erase(FSMap& target)
        {
#ifdef DEBUG_LEGION
          assert(!single);
#endif
          // Erase it from the target
          target.erase(it);
          // Invalidate the iterator
          it = target.end();
          result = nullptr;
        }
      private:
        friend class const_iterator;
        FieldMaskSet* set;
        std::pair<T* const, FieldMask>* result;
        typename FSMap::iterator it;
        bool single;
      };
    public:
      class const_iterator {
      public:
        // explicitly set iterator traits
        typedef std::input_iterator_tag iterator_category;
        typedef std::pair<T* const, FieldMask> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::pair<T* const, FieldMask>* pointer;
        typedef std::pair<T* const, FieldMask>& reference;

        const_iterator(
            const FieldMaskSet* _set,
            const std::pair<T* const, FieldMask>* _result)
          : set(_set), result(_result), single(true)
        { }
        const_iterator(
            const FieldMaskSet* _set, typename FSMap::const_iterator _it,
            bool end = false)
          : set(_set), result(end ? nullptr : &(*_it)), it(_it), single(false)
        { }
      public:
        const_iterator(const const_iterator& rhs)
          : set(rhs.set), result(rhs.result), it(rhs.it), single(rhs.single)
        { }
        // We can also make a const_iterator from a normal iterator
        const_iterator(const iterator& rhs)
          : set(rhs.set), result(rhs.result), it(rhs.it), single(rhs.single)
        { }
        ~const_iterator(void) { }
      public:
        inline const_iterator& operator=(const const_iterator& rhs)
        {
          set = rhs.set;
          result = rhs.result;
          it = rhs.it;
          single = rhs.single;
          return *this;
        }
        inline const_iterator& operator=(const iterator& rhs)
        {
          set = rhs.set;
          result = rhs.result;
          it = rhs.it;
          single = rhs.single;
          return *this;
        }
      public:
        inline bool operator==(const const_iterator& rhs) const
        {
          if (set != rhs.set)
            return false;
          if (single)
            return (result == rhs.result);
          else
            return (it == rhs.it);
        }
        inline bool operator!=(const const_iterator& rhs) const
        {
          if (set != rhs.set)
            return true;
          if (single)
            return (result != rhs.result);
          else
            return (it != rhs.it);
        }
      public:
        inline const std::pair<T* const, FieldMask> operator*(void)
        {
          return *result;
        }
        inline const std::pair<T* const, FieldMask>* operator->(void)
        {
          return result;
        }
        inline const_iterator& operator++(/*prefix*/ void)
        {
          if (!single)
          {
            ++it;
            if ((*this) != set->end())
              result = &(*it);
            else
              result = nullptr;
          }
          else
            result = nullptr;
          return *this;
        }
        inline const_iterator operator++(/*postfix*/ int)
        {
          const_iterator copy(*this);
          if (!single)
          {
            ++it;
            if ((*this) != set->end())
              result = &(*it);
            else
              result = nullptr;
          }
          else
            result = nullptr;
          return copy;
        }
      public:
        inline operator bool(void) const { return (result != nullptr); }
      private:
        const FieldMaskSet* set;
        const std::pair<T* const, FieldMask>* result;
        typename FSMap::const_iterator it;
        bool single;
      };
    public:
      FieldMaskSet(void) : single(true) { entries.single_entry = nullptr; }
      inline FieldMaskSet(T* init, const FieldMask& m, bool no_null = true);
      inline FieldMaskSet(const FieldMaskSet<T, L, DETERMINISTIC>& rhs);
      inline FieldMaskSet(FieldMaskSet<T, L, DETERMINISTIC>&& rhs) noexcept;
      // If copy is set to false then this is a move constructor
      inline FieldMaskSet(FieldMaskSet<T, L, DETERMINISTIC>& rhs, bool copy);
      ~FieldMaskSet(void) { clear(); }
    public:
      inline FieldMaskSet& operator=(
          const FieldMaskSet<T, L, DETERMINISTIC>& rh);
      inline FieldMaskSet& operator=(
          FieldMaskSet<T, L, DETERMINISTIC>&& rhs) noexcept;
    public:
      inline bool empty(void) const
      {
        return single && (entries.single_entry == nullptr);
      }
      inline const FieldMask& get_valid_mask(void) const
      {
        return valid_fields;
      }
      inline const FieldMask& tighten_valid_mask(void);
      inline void relax_valid_mask(const FieldMask& m);
      inline void filter_valid_mask(const FieldMask& m);
      inline void restrict_valid_mask(const FieldMask& m);
    public:
      inline const FieldMask& operator[](T* entry) const;
    public:
      // Return true if we actually added the entry, false if it already existed
      inline bool insert(T* entry, const FieldMask& mask);
      inline void filter(const FieldMask& filter, bool tighten = true);
      inline void erase(T* to_erase);
      inline void clear(void);
      inline size_t size(void) const;
    public:
      inline void swap(FieldMaskSet& other);
    public:
      inline iterator begin(void);
      inline iterator find(T* entry);
      inline void erase(iterator& it);
      inline iterator end(void);
    public:
      inline const_iterator begin(void) const;
      inline const_iterator find(T* entry) const;
      inline const_iterator end(void) const;
    public:
      inline void compute_field_sets(
          FieldMask universe_mask,
          local::list<FieldSet<T*> >& output_sets) const;
    protected:
      template<typename T2, AllocationLifetime L2, bool D2>
      friend class FieldMaskSet;

      // Fun with C, keep these two fields first and in this order
      // so that a FieldMaskSet of size 1 looks the same as an entry
      // in the STL Map in the multi-entries case,
      // provides goodness for the iterator
      union {
        T* single_entry;
        FSMap* multi_entries;
      } entries;
      // This can be an overapproximation if we have multiple entries
      FieldMask valid_fields;
      bool single;
    };

    // Create some insantiations of these templates in the lifetime namespaces
    namespace local {
      template<typename T, bool DETERMINISTIC = false>
      using FieldMaskSet =
          Legion::Internal::FieldMaskSet<T, TASK_LOCAL_LIFETIME, DETERMINISTIC>;
    }  // namespace local
    namespace op {
      template<typename T, bool DETERMINISTIC = false>
      using FieldMaskSet =
          Legion::Internal::FieldMaskSet<T, OPERATION_LIFETIME, DETERMINISTIC>;
    }  // namespace op
    namespace ctx {
      template<typename T, bool DETERMINISTIC = false>
      using FieldMaskSet =
          Legion::Internal::FieldMaskSet<T, CONTEXT_LIFETIME, DETERMINISTIC>;
    }  // namespace ctx
    namespace shrt {
      template<typename T, bool DETERMINISTIC = false>
      using FieldMaskSet =
          Legion::Internal::FieldMaskSet<T, SHORT_LIFETIME, DETERMINISTIC>;
    }  // namespace shrt
    namespace lng {
      template<typename T, bool DETERMINISTIC = false>
      using FieldMaskSet =
          Legion::Internal::FieldMaskSet<T, LONG_LIFETIME, DETERMINISTIC>;
    }  // namespace lng
    namespace rt {
      template<typename T, bool DETERMINISTIC = false>
      using FieldMaskSet =
          Legion::Internal::FieldMaskSet<T, RUNTIME_LIFETIME, DETERMINISTIC>;
    }  // namespace rt

  }  // namespace Internal
}  // namespace Legion

#include "legion/utilities/fieldmask_set.inl"

#endif  // __LEGION_FIELDMASK_SET_H__
