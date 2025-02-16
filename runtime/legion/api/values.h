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

#ifndef __LEGION_DEFERRED_VALUES_H__
#define __LEGION_DEFERRED_VALUES_H__

#include "legion/api/types.h"
#include "legion/api/geometry.h"

namespace Legion {

  /**
   * \class UntypedDeferredValue
   * This is a type-erased deferred value with the type of the field.
   */
  class UntypedDeferredValue {
  public:
    UntypedDeferredValue(void);
    UntypedDeferredValue(size_t field_size, Memory target_memory,
                         const void *initial_value = nullptr,
                         size_t alignment = 16);
    UntypedDeferredValue(size_t field_size,
                         Memory::Kind memory_kind = Memory::Z_COPY_MEM,
                         const void *initial_value = nullptr,
                         size_t alignment = 16);
    UntypedDeferredValue(const UntypedDeferredValue &rhs);
  public:
    template<typename T>
    inline operator DeferredValue<T>(void) const;
    template<typename REDOP, bool EXCLUSIVE>
    inline operator DeferredReduction<REDOP,EXCLUSIVE>(void) const;
    inline size_t field_size(void) const;
  public:
    void finalize(Context ctx) const;
    Realm::RegionInstance get_instance(void) const;
    static void report_incompatible_accessor(const char *accessor_kind, bool buffer = false);
    static void report_nondense_domain(void);
    static void report_nondense_rect(void);
  protected:
    Realm::RegionInstance instance;
  protected:
    void initialize(Memory memory, size_t field_size,
        const void *initial_value, size_t alignment);
    // Helper methods for DeferredValues and DeferredBuffers
    static Memory find_memory_by_kind(Memory::Kind kind, bool value);
    static Realm::RegionInstance allocate_instance(Memory memory,
        Realm::InstanceLayoutGeneric *layout);
    static void destroy_instance(Realm::RegionInstance,
        Realm::Event precondition);
    static Domain get_index_space_bounds(IndexSpace space);
    template<PrivilegeMode,typename,int,typename,typename,bool>
    friend class FieldAccessor;
    template<typename,bool,int,typename,typename,bool>
    friend class ReductionAccessor;
    template<typename>
    friend class UntypedDeferredBuffer;
    template<typename,int,typename,bool>
    friend class DeferredBuffer;
  };

  /**
   * \class DeferredValue
   * A deferred value is a special helper class for handling return values 
   * for tasks that do asynchronous operations (e.g. GPU kernel launches), 
   * but we don't want to wait for the asynchronous operations to be returned. 
   * This object should be returned directly as the result of a Legion task, 
   * but its value will not be read until all of the "effects" of the task 
   * are done. The following methods are supported during task execution:
   *  - T read(void) const
   *  - void write(T val) const
   *  - T* ptr(void) const
   *  - T& operator(void) const
   */
  template<typename T>
  class DeferredValue : public UntypedDeferredValue {
  public:
    DeferredValue(T initial_value,
                  size_t alignment = std::alignment_of<T>(),
                  Memory::Kind memory_kind = Memory::Z_COPY_MEM);
    DeferredValue(T initial_value, Memory target_memory,
                  size_t alignment = std::alignment_of<T>());
  public:
    __LEGION_CUDA_HD__
    inline T read(void) const;
    __LEGION_CUDA_HD__
    inline void write(T value) const;
    __LEGION_CUDA_HD__
    inline T* ptr(void) const;
    __LEGION_CUDA_HD__
    inline T& ref(void) const;
    __LEGION_CUDA_HD__
    inline operator T(void) const;
    __LEGION_CUDA_HD__
    inline DeferredValue<T>& operator=(T value);
  public:
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference; 
  protected:
    DeferredValue(void);
    Realm::AffineAccessor<T,1,coord_t> accessor;
  };

  /**
   * \class DeferredReduction 
   * This is a special case of a DeferredValue that also supports
   * a reduction operator. It supports all the same methods
   * as the DeferredValue as well as an additional method for
   * doing reductions using a reduction operator.
   *  - void reduce(REDOP::RHS val)
   *  - void <<=(REDOP::RHS val)
   */
  template<typename REDOP, bool EXCLUSIVE=false>
  class DeferredReduction: public DeferredValue<typename REDOP::RHS> {
  public:
    DeferredReduction(
        size_t alignment = std::alignment_of<typename REDOP::RHS>());
  public:
    __LEGION_CUDA_HD__
    inline void reduce(typename REDOP::RHS val) const;
    __LEGION_CUDA_HD__
    inline void operator<<=(typename REDOP::RHS val) const;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
  };

} // namespace Legion

#include "legion/api/values.inl"

#endif // __LEGION_DEFERRED_VALUES_H__
