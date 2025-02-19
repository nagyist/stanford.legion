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

#ifndef __LEGION_ALLOCATION__
#define __LEGION_ALLOCATION__

#include <set>
#include <map>
#include <new>
#include <list>
#include <queue>
#include <deque>
#include <vector>
#include <limits>
#include <cstring>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#ifdef LEGION_TRACE_ALLOCATION
#include <typeinfo>
#endif
#include "legion/api/config.h"

namespace Legion {
  namespace Internal {

    enum AllocationLifetime {
      TASK_LOCAL_LIFETIME,     // limited to the life of this Realm task
      OPERATION_LIFETIME,      // lifetime of the operation but across multiple
                               // tasks
      CONTEXT_LIFETIME,        // lifetime of the enclosing task context
      SHORT_BOUNDED_LIFETIME,  // lives for a few operations but not many
      LONG_BOUNDED_LIFETIME,   // lives for potentially a long set of operations
      RUNTIME_LIFETIME,        // lives for the duration of the Legion runtime
      TODO_LIFETIME,           // still need to figure out the lifetime
    };

#ifdef LEGION_TRACE_ALLOCATION
    // Implementations in runtime.cc
    struct LegionAllocation {
    public:
      static void trace_allocation(
          const std::type_info& info, size_t size, int elems = 1);
      static void trace_free(
          const std::type_info& info, size_t size, int elems = 1);
    };
#endif

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime LIFETIME>
    inline T* legion_malloc(std::size_t size, std::size_t alignment)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      LegionAllocation::trace_allocation(typeid(T), size);
#endif
      if (alignment <= alignof(std::max_align_t))
        return static_cast<T*>(std::malloc(size));
      else
        return static_cast<T*>(std::aligned_alloc(alignment, size));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime LIFETIME>
    inline T* legion_calloc(std::size_t num)
    //--------------------------------------------------------------------------
    {
      if (num == 0)
        return nullptr;
      constexpr std::size_t SIZE = sizeof(T);
      constexpr std::size_t ALIGNMENT = alignof(T);
      if (num == 1)
      {
        // Need to zero-initialize this to comply with semantics of calloc
        void* ptr =
            static_cast<void*>(legion_malloc<T, LIFETIME>(SIZE, ALIGNMENT));
        std::memset(ptr, 0 /*value*/, SIZE);
        return static_cast<T*>(ptr);
      }
      // Compute the padding required between the elements
      constexpr std::size_t PADDING =
          (ALIGNMENT - (SIZE % ALIGNMENT)) % ALIGNMENT;
      // Has to hold for aligned alloc to work
      static_assert(((SIZE + PADDING) % ALIGNMENT) == 0);
      // Can subtract any padding off the last element
      const std::size_t bytes = num * (SIZE + PADDING);
#ifdef LEGION_TRACE_ALLOCATION
      LegionAllocation::trace_allocation(typeid(T), bytes);
#endif
      void* ptr = std::aligned_alloc(ALIGNMENT, bytes);
      // Need to zero-initialize this to comply with semantics of calloc
      std::memset(ptr, 0 /*value*/, bytes);
      return static_cast<T*>(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime LIFETIME>
    inline T* legion_realloc(T* ptr, std::size_t old_size, std::size_t new_size)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      const std::type_info& info = typeid(T);
      LegionAllocation::trace_free(info, old_size);
      LegionAllocation::trace_allocation(info, new_size);
#endif
      return static_cast<T*>(std::realloc(static_cast<void*>(ptr), new_size));
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void legion_free(T* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      LegionAllocation::trace_free(typeid(T), size);
#endif
      std::free(ptr);
    }

    // A class for Legion objects to inherit from to have their dynamic
    // memory allocations managed for alignment and tracing
    template<typename T, AllocationLifetime L>
    class Heapify {
    public:
      static inline void* operator new(std::size_t size);
      static inline void* operator new[](std::size_t size);
      static inline void* operator new(
          std::size_t size, std::align_val_t alignment);
      static inline void* operator new[](
          std::size_t size, std::align_val_t alignment);
    public:
      static inline void* operator new(std::size_t size, void* ptr);
      static inline void* operator new[](std::size_t size, void* ptr);
    public:
      static inline void operator delete(void* ptr, std::size_t size);
      static inline void operator delete[](void* ptr, std::size_t size);
      static inline void operator delete(void* ptr, void* place);
      static inline void operator delete[](void* ptr, void* place);
    };

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new(std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new[](std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new(
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new[](
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new(
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new[](
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void Heapify<T, L>::operator delete(
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void Heapify<T, L>::operator delete[](
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void Heapify<T, L>::operator delete(
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void Heapify<T, L>::operator delete[](
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    // Same as Heapify but for overriding a base class definitions to keep
    // the compiler happy so it knows which definitions of operator new/delete
    // to use without getting confused
    template<typename T, typename B, AllocationLifetime L>
    class HeapifyMixin : public B {
    public:
      template<typename... Args>
      HeapifyMixin(Args&&... args) : B(std::forward<Args>(args)...)
      { }
    public:
      static inline void* operator new(std::size_t size);
      static inline void* operator new[](std::size_t size);
      static inline void* operator new(
          std::size_t size, std::align_val_t alignment);
      static inline void* operator new[](
          std::size_t size, std::align_val_t alignment);
    public:
      static inline void* operator new(std::size_t size, void* ptr);
      static inline void* operator new[](std::size_t size, void* ptr);
    public:
      static inline void operator delete(void* ptr, std::size_t size);
      static inline void operator delete[](void* ptr, std::size_t size);
      static inline void operator delete(void* ptr, void* place);
      static inline void operator delete[](void* ptr, void* place);
    };

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new(
        std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new[](
        std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new(
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new[](
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new(
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new[](
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void HeapifyMixin<T, B, L>::operator delete(
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void HeapifyMixin<T, B, L>::operator delete[](
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void HeapifyMixin<T, B, L>::operator delete(
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void HeapifyMixin<T, B, L>::operator delete[](
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    // A class to ensure that a type is never dynamically allocated
    class NoHeapify {
    public:
      static inline void* operator new(std::size_t) = delete;
      static inline void* operator new[](std::size_t) = delete;
      static inline void* operator new(std::size_t, std::align_val_t) = delete;
      static inline void* operator new[](std::size_t, std::align_val_t) =
          delete;
    public:
      static inline void* operator new(std::size_t, void*) = delete;
      static inline void* operator new[](std::size_t, void*) = delete;
      static inline void* operator new(std::size_t, std::align_val_t, void*) =
          delete;
      static inline void* operator new[](std::size_t, std::align_val_t, void*) =
          delete;
    };

    /**
     * \class LegionAllocator
     * A custom Legion allocator for tracing memory usage in STL
     * data structures. When tracing is disabled, it defaults back
     * to using the standard malloc/free and new/delete operations.
     */
    template<typename T, AllocationLifetime L>
    class LegionAllocator {
    public:
      typedef size_t size_type;
      typedef ptrdiff_t difference_type;
      typedef T* pointer;
      typedef const T* const_pointer;
      typedef T& reference;
      typedef const T& const_reference;
      typedef T value_type;
    public:
      template<typename U>
      struct rebind {
        typedef LegionAllocator<U, L> other;
      };
    public:
      inline explicit LegionAllocator(void) { }
      inline ~LegionAllocator(void) { }
      inline LegionAllocator(const LegionAllocator<T, L>& rhs) { }
      template<typename U>
      inline LegionAllocator(const LegionAllocator<U, L>& rhs)
      { }
    public:
      inline pointer address(reference r) { return &r; }
      inline const_pointer address(const_reference r) { return &r; }
    public:
      inline T* allocate(std::size_t num)
      {
        return static_cast<T*>(legion_calloc<T, L>(num));
      }
      inline void deallocate(T* ptr, std::size_t num)
      {
#ifdef LEGION_TRACE_ALLOCATION
        constexpr std::size_t SIZE = sizeof(T);
        if (num == 1)
        {
          legion_free<T>(ptr, SIZE);
        } else if (num > 1)
        {
          constexpr std::size_t ALIGNMENT = alignof(T);
          // Compute the padding required between the elements
          constexpr std::size_t PADDING =
              (ALIGNMENT - (SIZE % ALIGNMENT)) % ALIGNMENT;
          // Can subtract any padding off the last element
          const std::size_t bytes = num * (SIZE + PADDING);
          legion_free<T>(ptr, bytes);
        }
#else
        legion_free<T>(ptr, 0 /*bogus size*/);
#endif
      }
    public:
      inline size_type max_size(void) const
      {
        return std::numeric_limits<size_type>::max() / sizeof(T);
      }
    public:
#if __cplusplus > 201703L
      template<class U, class... Args>
      inline constexpr U* construct_at(U* p, Args&&... args)
      {
        return ::new (const_cast<void*>(static_cast<const void volatile *>(p)))
            U(std::forward<Args>(args)...);
      }

#else
      template<class U, class... Args>
      inline void construct(U* p, Args&&... args)
      {
        ::new ((void*)p) U(std::forward<Args>(args)...);
      }
#endif
#if __cplusplus > 201703L
      template<class U>
      inline constexpr void destroy_at(U* p)
      {
        p->~U();
      }
#else
      template<class U>
      inline void destroy_at(U* p)
      {
        p->~U();
      }
#endif
    public:
      inline bool operator==(const LegionAllocator&) const { return true; }
      inline bool operator!=(const LegionAllocator& a) const
      {
        return !operator==(a);
      }
    };

    template<
        typename T, AllocationLifetime L = TASK_LOCAL_LIFETIME,
        typename COMPARATOR = std::less<T> >
    using LegionSet = std::set<T, COMPARATOR, LegionAllocator<T, L> >;

    template<typename T, AllocationLifetime L = TASK_LOCAL_LIFETIME>
    using LegionList = std::list<T, LegionAllocator<T, L> >;

    template<typename T, AllocationLifetime L = TASK_LOCAL_LIFETIME>
    using LegionDeque = std::deque<T, LegionAllocator<T, L> >;

    template<typename T, AllocationLifetime L = TASK_LOCAL_LIFETIME>
    using LegionVector = std::vector<T, LegionAllocator<T, L> >;

    template<
        typename T1, typename T2, AllocationLifetime L = TASK_LOCAL_LIFETIME,
        typename COMPARATOR = std::less<T1> >
    using LegionMap = std::map<
        T1, T2, COMPARATOR, LegionAllocator<std::pair<const T1, T2>, L> >;
  };  // namespace Internal
};  // namespace Legion

#endif  // __LEGION_ALLOCATION__
