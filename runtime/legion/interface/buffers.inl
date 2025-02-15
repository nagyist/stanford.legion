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

// Included from buffers.h - do not include this directly

// Useful for IDEs
#include "legion/interface/buffers.h"

namespace Legion {

    // Some helper methods for accessors and deferred buffers
    namespace Internal {
    template<int N, typename T> __LEGION_CUDA_HD__
    static inline bool is_dense_layout(const Rect<N,T> &bounds,
                              const size_t strides[N], size_t field_size)
    {
      ptrdiff_t exp_offset = field_size;
      int used_mask = 0; // keep track of the dimensions we've already matched
      static_assert((N <= (8*sizeof(used_mask))), "Mask dim exceeded");
      for (int i = 0; i < N; i++) {
        bool found = false;
        for (int j = 0; j < N; j++) {
          if ((used_mask >> j) & 1) continue;
          if (strides[j] != exp_offset) 
          {
            // Mask off any dimensions with stride 0
            if (strides[j] == 0)
            {
              if (bounds.lo[j] != bounds.hi[j])
                return false;
              used_mask |= (1 << j);
              if (++i == N) 
              {
                found = true;
                break;
              }
            }
            continue;
          }
          found = true;
          // It's possible other dimensions can have the same strides if
          // there are multiple dimensions with extents of size 1. At most
          // one dimension can have an extent >1 though
          int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
          for (int k = j+1; k < N; k++) {
            if ((used_mask >> k) & 1) continue;
            if (strides[k] == exp_offset) {
              if (bounds.lo[k] < bounds.hi[k]) {
                // if we already saw a non-trivial dimension this is bad
                if (nontrivial >= 0)
                  return false;
                else
                  nontrivial = k;
              }
              used_mask |= (1 << k);
              i++;
            }
          }
          used_mask |= (1 << j);
          if (nontrivial >= 0)
            exp_offset *= (bounds.hi[nontrivial] - bounds.lo[nontrivial] + 1);
          break;
        }
        if (!found)
          return false;
      }
      return true;
    }

    // Same method as above but for realm points from affine accessors
    template<int N, typename T> __LEGION_CUDA_HD__
    static inline bool is_dense_layout(const Rect<N,T> &bounds,
                const Realm::Point<N,size_t> &strides, size_t field_size)
    {
      size_t exp_offset = field_size;
      int used_mask = 0; // keep track of the dimensions we've already matched
      static_assert((N <= (8*sizeof(used_mask))), "Mask dim exceeded");
      for (int i = 0; i < N; i++) {
        bool found = false;
        for (int j = 0; j < N; j++) {
          if ((used_mask >> j) & 1) continue;
          if (strides[j] != exp_offset) 
          {
            // Mask off any dimensions with stride 0
            if (strides[j] == 0) 
            {
              if (bounds.lo[j] != bounds.hi[j])
                return false;
              used_mask |= (1 << j);
              if (++i == N) 
              {
                found = true;
                break;
              }
            }
            continue;
          }
          found = true;
          // It's possible other dimensions can have the same strides if
          // there are multiple dimensions with extents of size 1. At most
          // one dimension can have an extent >1 though
          int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
          for (int k = j+1; k < N; k++) {
            if ((used_mask >> k) & 1) continue;
            if (strides[k] == exp_offset) {
              if (bounds.lo[k] < bounds.hi[k]) {
                // if we already saw a non-trivial dimension this is bad
                if (nontrivial >= 0)
                  return false;
                else
                  nontrivial = k;
              }
              used_mask |= (1 << k);
              i++;
            }
          }
          used_mask |= (1 << j);
          if (nontrivial >= 0)
            exp_offset *= (bounds.hi[nontrivial] - bounds.lo[nontrivial] + 1);
          break;
        }
        if (!found)
          return false;
      }
      return true;
    }
  }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(void)
      : instance(Realm::RegionInstance::NO_INST)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(
                             Memory::Kind kind, const Domain &space,
                             const FT *initial_value/* = NULL*/,
                             size_t alignment/* = alignof(FT)*/,
                             bool fortran_order_dims/* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
        UntypedDeferredValue::report_nondense_domain();
      const Realm::Memory memory =
        UntypedDeferredValue::find_memory_by_kind(kind, false);
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(
                             const Rect<N,T> &rect, Memory::Kind kind,
                             const FT *initial_value /*= NULL*/,
                             size_t alignment/* = alignof(FT)*/,
                             bool fortran_order_dims /*= false*/)
    //--------------------------------------------------------------------------
    {
      const Realm::Memory memory = 
        UntypedDeferredValue::find_memory_by_kind(kind, false);
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(
                             Memory memory, const Domain &space,
                             const FT *initial_value/* = NULL*/,
                             size_t alignment/* = alignof(FT)*/,
                             bool fortran_order_dims/* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
        UntypedDeferredValue::report_nondense_domain();
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(
                             const Rect<N,T> &rect, Memory memory,
                             const FT *initial_value /*= NULL*/,
                             size_t alignment/* = alignof(FT)*/,
                             bool fortran_order_dims /*= false*/)
    //--------------------------------------------------------------------------
    {
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(
                             Memory::Kind kind, const Domain &space,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value/* = NULL*/,
                             size_t _alignment/* = alignof(FT)*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
        UntypedDeferredValue::report_nondense_domain();
      const Realm::Memory memory = 
        UntypedDeferredValue::find_memory_by_kind(kind, false);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(
                             const Rect<N,T> &rect, Memory::Kind kind,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value /*= NULL*/,
                             size_t _alignment/* = alignof(FT)*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      const Realm::Memory memory = 
        UntypedDeferredValue::find_memory_by_kind(kind, false);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(
                             Memory memory, const Domain &space,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value/* = NULL*/,
                             size_t _alignment/* = alignof(FT)*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
        UntypedDeferredValue::report_nondense_domain();
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    inline DeferredBuffer<FT,N,T,CB>::DeferredBuffer(
                             const Rect<N,T> &rect, Memory memory,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value /*= NULL*/,
                             size_t _alignment/* = alignof(FT)*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    void DeferredBuffer<FT,N,T,CB>::initialize_layout(size_t _alignment,
                                                      bool fortran_order_dims)
    //--------------------------------------------------------------------------
    {
      if (fortran_order_dims)
      {
        for (int i = 0; i < N; i++)
          ordering[i] =
            static_cast<DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
      }
      else
      {
        for (int i = 0; i < N; i++)
          ordering[i] =
            static_cast<DimensionKind>(
                static_cast<int>(LEGION_DIM_X) + N - (i + 1));
      }

      alignment = _alignment;
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB>
    void DeferredBuffer<FT,N,T,CB>::initialize(Memory memory,
                                      DomainT<N,T> space,
                                      const FT *initial_value)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(space.dense());
#endif
      const std::vector<size_t> field_sizes(1,sizeof(FT));
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      int dim_order[N];
      for (int i = 0; i < N; ++i)
        dim_order[i] =
          static_cast<int>(ordering[i]) - static_cast<int>(LEGION_DIM_X);
      Realm::InstanceLayoutGeneric *layout = 
        Realm::InstanceLayoutGeneric::choose_instance_layout(
          space, constraints, dim_order);
      layout->alignment_reqd = alignment;
      instance = UntypedDeferredValue::allocate_instance(memory, layout);
      bounds = space.bounds;
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests;
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, sizeof(FT));
        const Internal::LgEvent wait_on(
            bounds.fill(dsts, no_requests, initial_value, sizeof(FT)));
        if (wait_on.exists())
          wait_on.wait();
      }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool is_compatible =
        Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
                                                     0/*fid*/,
                                                     bounds);
#endif
      assert(is_compatible);
#endif
      // We can make the accessor
      accessor = Realm::AffineAccessor<FT,N,T>(instance,
                                               0/*field id*/,
                                               bounds);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB> __LEGION_CUDA_HD__
    inline FT DeferredBuffer<FT,N,T,CB>::read(const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      assert(!CB || bounds.contains(p));
      return accessor.read(p);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB> __LEGION_CUDA_HD__
    inline void DeferredBuffer<FT,N,T,CB>::write(const Point<N,T> &p,
                                                    FT value) const
    //--------------------------------------------------------------------------
    {
      assert(!CB || bounds.contains(p));
      accessor.write(p, value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB > __LEGION_CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,CB>::ptr(const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      assert(!CB || bounds.contains(p));
      return accessor.ptr(p);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB> __LEGION_CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,CB>::ptr(const Rect<N,T> &r) const
    //--------------------------------------------------------------------------
    {
      assert(!CB || bounds.contains(r));
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, sizeof(FT)));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, sizeof(FT)))
        UntypedDeferredValue::report_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T,bool CB> __LEGION_CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,CB>::ptr(
                                    const Rect<N,T> &r, size_t strides[N]) const
    //--------------------------------------------------------------------------
    {
      assert(!CB || bounds.contains(r));
      for (int i = 0; i < N; i++)
        strides[i] = accessor.strides[i] / sizeof(FT);
      return accessor.ptr(r.lo);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB> __LEGION_CUDA_HD__
    inline FT& DeferredBuffer<FT,N,T,CB>::operator[](const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      assert(!CB || bounds.contains(p));
      return accessor[p];
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T,bool CB>
    inline void DeferredBuffer<FT,N,T,CB>::destroy(Realm::Event precondition)
    //--------------------------------------------------------------------------
    {
      UntypedDeferredValue::destroy_instance(instance, precondition);
      instance = Realm::RegionInstance::NO_INST;
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB> __LEGION_CUDA_HD__
    inline Realm::RegionInstance DeferredBuffer<FT,N,T,CB>::get_instance(
                                                                     void) const
    //--------------------------------------------------------------------------
    {
      return instance;
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T, bool CB> __LEGION_CUDA_HD__
    inline Rect<N,T> DeferredBuffer<FT,N,T,CB>::get_bounds(void) const
    //--------------------------------------------------------------------------
    {
      return bounds;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(void)
      : instance(Realm::RegionInstance::NO_INST), field_size(0), dims(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(size_t fs, int d,
                                                    Memory::Kind memkind,
                                                    IndexSpace space,
                                                    const void *initial_value,
                                                    size_t alignment,
                                                    bool fortran_order_dims)
      : field_size(fs), dims(d)
    //--------------------------------------------------------------------------
    {
      assert(dims > 0);
      assert(dims <= LEGION_MAX_DIM);
      const Memory memory =
        UntypedDeferredValue::find_memory_by_kind(memkind, false);
      const std::vector<size_t> field_sizes(1, field_size);
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      Realm::InstanceLayoutGeneric *layout = NULL;
      const Domain domain = UntypedDeferredValue::get_index_space_bounds(space);
      switch (dims)
      {
#define DIMFUNC(DIM)                                                        \
        case DIM:                                                           \
          {                                                                 \
            const DomainT<DIM,T> bounds = domain;                           \
            if (!bounds.dense())                                            \
            {                                                               \
              fprintf(stderr, "DeferredBuffer only allows a dense domain\n");\
              assert(false);                                                \
            }                                                               \
            int dim_order[DIM];                                             \
            if (fortran_order_dims)                                         \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = i;                                           \
            }                                                               \
            else                                                            \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = DIM - (i+1);                                 \
            }                                                               \
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout(  \
                bounds, constraints, dim_order);                            \
            break;                                                          \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      layout->alignment_reqd = alignment;
      instance = UntypedDeferredValue::allocate_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, field_size);
        Internal::LgEvent wait_on;
        switch (dims)
        {
#define DIMFUNC(DIM)                                                      \
          case DIM:                                                       \
            {                                                             \
              const DomainT<DIM,T> bounds = domain;                       \
              wait_on = Internal::LgEvent(                                \
              bounds.fill(dsts, no_requests, initial_value, field_size)); \
              break;                                                      \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            std::abort();
        }
        if (wait_on.exists())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(size_t fs, int d,
                                                    Memory::Kind memkind,
                                                    const Domain &space,
                                                    const void *initial_value,
                                                    size_t alignment,
                                                    bool fortran_order_dims)
      : field_size(fs), dims(d)
    //--------------------------------------------------------------------------
    {
      assert(dims > 0);
      assert(dims <= LEGION_MAX_DIM);
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      const Memory memory =
        UntypedDeferredValue::find_memory_by_kind(memkind, false);
      const std::vector<size_t> field_sizes(1, field_size);
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      Realm::InstanceLayoutGeneric *layout = NULL;
      switch (dims)
      {
#define DIMFUNC(DIM)                                                        \
        case DIM:                                                           \
          {                                                                 \
            const DomainT<DIM,T> bounds = space;                            \
            int dim_order[DIM];                                             \
            if (fortran_order_dims)                                         \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = i;                                           \
            }                                                               \
            else                                                            \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = DIM - (i+1);                                 \
            }                                                               \
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout(  \
                bounds, constraints, dim_order);                            \
            break;                                                          \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      layout->alignment_reqd = alignment;
      instance = UntypedDeferredValue::allocate_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, field_size);
        Internal::LgEvent wait_on;
        switch (dims)
        {
#define DIMFUNC(DIM)                                                      \
          case DIM:                                                       \
            {                                                             \
              const DomainT<DIM,T> bounds = space;                        \
              wait_on = Internal::LgEvent(                                \
              bounds.fill(dsts, no_requests, initial_value, field_size)); \
              break;                                                      \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            std::abort();
        }
        if (wait_on.exists())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(size_t fs, int d,
                                                    Memory memory,
                                                    IndexSpace space,
                                                    const void *initial_value,
                                                    size_t alignment,
                                                    bool fortran_order_dims)
      : field_size(fs), dims(d)
    //--------------------------------------------------------------------------
    {
      assert(dims > 0);
      assert(dims <= LEGION_MAX_DIM);
      const std::vector<size_t> field_sizes(1, field_size);
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      Realm::InstanceLayoutGeneric *layout = NULL;
      const Domain domain = UntypedDeferredValue::get_index_space_bounds(space);
      switch (dims)
      {
#define DIMFUNC(DIM)                                                        \
        case DIM:                                                           \
          {                                                                 \
            const DomainT<DIM,T> bounds = domain;                           \
            if (!bounds.dense())                                            \
            {                                                               \
              fprintf(stderr, "DeferredBuffer only allows a dense domain\n");\
              assert(false);                                                \
            }                                                               \
            int dim_order[DIM];                                             \
            if (fortran_order_dims)                                         \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = i;                                           \
            }                                                               \
            else                                                            \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = DIM - (i+1);                                 \
            }                                                               \
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout(  \
                bounds, constraints, dim_order);                            \
            break;                                                          \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      layout->alignment_reqd = alignment;
      instance = UntypedDeferredValue::allocate_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, field_size);
        Internal::LgEvent wait_on;
        switch (dims)
        {
#define DIMFUNC(DIM)                                                      \
          case DIM:                                                       \
            {                                                             \
              const DomainT<DIM,T> bounds = domain;                       \
              wait_on = Internal::LgEvent(                                \
              bounds.fill(dsts, no_requests, initial_value, field_size)); \
              break;                                                      \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            std::abort();
        }
        if (wait_on.exists())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(size_t fs, int d,
                                                    Memory memory,
                                                    const Domain &space,
                                                    const void *initial_value,
                                                    size_t alignment,
                                                    bool fortran_order_dims)
      : field_size(fs), dims(d)
    //--------------------------------------------------------------------------
    {
      assert(dims > 0);
      assert(dims <= LEGION_MAX_DIM);
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      const std::vector<size_t> field_sizes(1, field_size);
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      Realm::InstanceLayoutGeneric *layout = NULL;
      switch (dims)
      {
#define DIMFUNC(DIM)                                                        \
        case DIM:                                                           \
          {                                                                 \
            const DomainT<DIM,T> bounds = space;                            \
            int dim_order[DIM];                                             \
            if (fortran_order_dims)                                         \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = i;                                           \
            }                                                               \
            else                                                            \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = DIM - (i+1);                                 \
            }                                                               \
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout(  \
                bounds, constraints, dim_order);                            \
            break;                                                          \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      layout->alignment_reqd = alignment;
      instance = UntypedDeferredValue::allocate_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, field_size);
        Internal::LgEvent wait_on;
        switch (dims)
        {
#define DIMFUNC(DIM)                                                      \
          case DIM:                                                       \
            {                                                             \
              const DomainT<DIM,T> bounds = space;                        \
              wait_on = Internal::LgEvent(                                \
              bounds.fill(dsts, no_requests, initial_value, field_size)); \
              break;                                                      \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            std::abort();
        }
        if (wait_on.exists())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T> template<typename FT, int DIM>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(
                                            const DeferredBuffer<FT,DIM,T> &rhs)
      : instance(rhs.instance), field_size(sizeof(FT)), dims(DIM)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T> template<typename FT, int DIM, bool BC>
    inline UntypedDeferredBuffer<T>::operator 
                                         DeferredBuffer<FT,DIM,T,BC>(void) const
    //--------------------------------------------------------------------------
    {
      static_assert(0 < DIM, "Only positive dimensions allowed");
      static_assert(DIM <= LEGION_MAX_DIM, "Exceeded LEGION_MAX_DIM");
      assert(field_size == sizeof(FT));
      assert(dims == DIM);
      DeferredBuffer<FT,DIM,T> result;
      result.instance = instance;
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool is_compatible = 
        Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance, 0/*field id*/);
#endif
      assert(is_compatible);
#endif
      // We can make the accessor
      result.accessor = Realm::AffineAccessor<FT,DIM,T>(instance,0/*field id*/);
      result.bounds = instance.template get_indexspace<DIM,T>().bounds;
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void UntypedDeferredBuffer<T>::destroy(Realm::Event precondition)
    //--------------------------------------------------------------------------
    {
      UntypedDeferredValue::destroy_instance(instance, precondition);
      instance = Realm::RegionInstance::NO_INST;
      field_size = 0;
      dims = 0;
    }

} // namespace Legion
