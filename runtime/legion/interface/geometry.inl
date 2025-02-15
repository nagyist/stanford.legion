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

// Included from geometry.h - do not include this directly

// Useful for IDEs
#include "legion/interface/geometry.h"

namespace Legion {

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint::DomainPoint(void)
    : dim(0)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < MAX_POINT_DIM; i++)
      point_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint::DomainPoint(coord_t index)
    : dim(1)
  //----------------------------------------------------------------------------
  {
    point_data[0] = index;
    for (int i = 1; i < MAX_POINT_DIM; i++)
      point_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint::DomainPoint(const DomainPoint &rhs)
    : dim(rhs.dim)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < MAX_POINT_DIM; i++)
      point_data[i] = rhs.point_data[i];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline DomainPoint::DomainPoint(const Point<DIM,T> &rhs)
    : dim(DIM)
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    for (int i = 0; i < DIM; i++)
      point_data[i] = check_for_overflow<T>(rhs[i]);
    // Zero out the rest of the buffer to avoid uninitialized warnings
    for (int i = DIM; i < MAX_POINT_DIM; i++)
      point_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  template<typename T> __CUDA_HD__
  /*static*/ inline coord_t DomainPoint::check_for_overflow(const T &value)
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_same<coord_t,long long>::value,"coord_t changed");
#ifdef DEBUG_LEGION
#ifndef NDEBUG
    constexpr bool CHECK =
      std::is_unsigned<T>::value && (sizeof(T) >= sizeof(coord_t));
    assert(!CHECK || 
        (((unsigned long long)value) <= ((unsigned long long)LLONG_MAX)));
#endif
#endif
    return coord_t(value);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline DomainPoint::operator Point<DIM,T>(void) const
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    assert(DIM == dim);
    Point<DIM,T> result;
    for (int i = 0; i < DIM; i++)
      result[i] = point_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint& DomainPoint::operator=(const DomainPoint &rhs)
  //----------------------------------------------------------------------------
  {
    dim = rhs.dim;
    for (int i = 0; i < MAX_POINT_DIM; i++)
      point_data[i] = rhs.point_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline DomainPoint& DomainPoint::operator=(const Point<DIM,T> &rhs)
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    dim = DIM;
    for (int i = 0; i < DIM; i++)
      point_data[i] = check_for_overflow<T>(rhs[i]);
    for (int i = DIM; i < MAX_POINT_DIM; i++)
      point_data[i] = 0;
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool DomainPoint::operator==(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    if(dim != rhs.dim) return false;
    for(int i = 0; (i == 0) || (i < dim); i++)
      if(point_data[i] != rhs.point_data[i]) return false;
    return true;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool DomainPoint::operator!=(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    return !((*this) == rhs);
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool DomainPoint::operator<(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    if (dim < rhs.dim) return true;
    if (dim > rhs.dim) return false;
    for (int i = 0; (i == 0) || (i < dim); i++) {
      if (point_data[i] < rhs.point_data[i]) return true;
      if (point_data[i] > rhs.point_data[i]) return false;
    }
    return false;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint DomainPoint::operator+(coord_t scalar) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] + scalar;
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ 
  inline DomainPoint DomainPoint::operator+(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] + rhs.point_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint& DomainPoint::operator+=(coord_t scalar)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < dim; i++)
      point_data[i] += scalar;
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__
  inline DomainPoint& DomainPoint::operator+=(const DomainPoint &rhs) 
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    for (int i = 0; i < dim; i++)
      point_data[i] += rhs.point_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint DomainPoint::operator-(coord_t scalar) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] - scalar;
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ 
  inline DomainPoint DomainPoint::operator-(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] - rhs.point_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint& DomainPoint::operator-=(coord_t scalar)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < dim; i++)
      point_data[i] -= scalar;
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__
  inline DomainPoint& DomainPoint::operator-=(const DomainPoint &rhs) 
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    for (int i = 0; i < dim; i++)
      point_data[i] -= rhs.point_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint DomainPoint::operator*(coord_t scalar) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] * scalar;
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ 
  inline DomainPoint DomainPoint::operator*(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] * rhs.point_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint& DomainPoint::operator*=(coord_t scalar)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < dim; i++)
      point_data[i] *= scalar;
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__
  inline DomainPoint& DomainPoint::operator*=(const DomainPoint &rhs) 
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    for (int i = 0; i < dim; i++)
      point_data[i] *= rhs.point_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint DomainPoint::operator/(coord_t scalar) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] / scalar;
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ 
  inline DomainPoint DomainPoint::operator/(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] / rhs.point_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint& DomainPoint::operator/=(coord_t scalar)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < dim; i++)
      point_data[i] /= scalar;
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__
  inline DomainPoint& DomainPoint::operator/=(const DomainPoint &rhs) 
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    for (int i = 0; i < dim; i++)
      point_data[i] /= rhs.point_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint DomainPoint::operator%(coord_t scalar) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] % scalar;
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ 
  inline DomainPoint DomainPoint::operator%(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = point_data[i] % rhs.point_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint& DomainPoint::operator%=(coord_t scalar)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < dim; i++)
      point_data[i] %= scalar;
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__
  inline DomainPoint& DomainPoint::operator%=(const DomainPoint &rhs) 
  //----------------------------------------------------------------------------
  {
    assert(get_dim() == rhs.get_dim());
    for (int i = 0; i < dim; i++)
      point_data[i] %= rhs.point_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline coord_t& DomainPoint::operator[](unsigned index)
  //----------------------------------------------------------------------------
  {
    assert(index < MAX_POINT_DIM);
    return point_data[index];
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline const coord_t& DomainPoint::operator[](unsigned index)const
  //----------------------------------------------------------------------------
  {
    assert(index < MAX_POINT_DIM);
    return point_data[index];
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Color DomainPoint::get_color(void) const
  //----------------------------------------------------------------------------
  {
    assert(dim == 1);
    return point_data[0];
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline coord_t DomainPoint::get_index(void) const
  //----------------------------------------------------------------------------
  {
    assert(dim == 1);
    return point_data[0];
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline int DomainPoint::get_dim(void) const
  //----------------------------------------------------------------------------
  {
    return dim;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool DomainPoint::is_null(void) const
  //----------------------------------------------------------------------------
  {
    return (dim == -1);
  }

  //----------------------------------------------------------------------------
  /*static*/ __CUDA_HD__ inline DomainPoint DomainPoint::nil(void)
  //----------------------------------------------------------------------------
  {
    DomainPoint p;
    p.dim = -1;
    return p;
  }

  //----------------------------------------------------------------------------
  inline /*friend */std::ostream& operator<<(std::ostream& os,
					     const DomainPoint& dp)
  //----------------------------------------------------------------------------
  {
    if (dp.dim > 0)
    {
      os << '(';
      for (int d = 0; d < dp.dim; d++)
      {
        if (d > 0)
          os << ',';
        os << dp.point_data[d];
      }
      os << ')';
    }
    else
      os << '[' << dp.point_data[0] << ']';
    return os;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain::Domain(void)
    : is_id(0), is_type(0), dim(0)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < 2*MAX_RECT_DIM; i++)
      rect_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain::Domain(const Domain &other)
    : is_id(other.is_id), is_type(is_id > 0 ? other.is_type : 0), dim(other.dim)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < 2*MAX_RECT_DIM; i++)
      rect_data[i] = other.rect_data[i];
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain::Domain(Domain &&other) noexcept
    : is_id(other.is_id), is_type(is_id > 0 ? other.is_type : 0), dim(other.dim)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < 2*MAX_RECT_DIM; i++)
      rect_data[i] = other.rect_data[i];
    other.is_id = 0;
    other.is_type = 0;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain::Domain(const DomainPoint &lo,const DomainPoint &hi)
    : is_id(0), is_type(0), dim(lo.dim)
  //----------------------------------------------------------------------------
  {
    assert(lo.dim == hi.dim);
    for (int i = 0; i < dim; i++)
      rect_data[i] = lo[i];
    for (int i = 0; i < dim; i++)
      rect_data[i+dim] = hi[i];
    // Zero-out the other values in rect data to keep compilers happy
    for (int i = 2*dim; i < 2*MAX_RECT_DIM; i++)
      rect_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Domain::Domain(const Rect<DIM,T> &other)
    : is_id(0), is_type(0), dim(DIM)
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    for (int i = 0; i < DIM; i++)
      rect_data[i] = check_for_overflow<T>(other.lo[i]);
    for (int i = 0; i < DIM; i++)
      rect_data[DIM+i] = check_for_overflow<T>(other.hi[i]);
    // Zero-out the other values in rect data to keep compilers happy
    for (int i = 2*DIM; i < 2*MAX_RECT_DIM; i++)
      rect_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  template<typename T> __CUDA_HD__
  /*static*/ inline coord_t Domain::check_for_overflow(const T &value)
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_same<coord_t,long long>::value, "coord_t changed");
#ifdef DEBUG_LEGION
#ifndef NDEBUG
    constexpr bool CHECK =
      std::is_unsigned<T>::value && (sizeof(T) >= sizeof(coord_t));
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    constexpr uint64_t MAX = std::numeric_limits<coord_t>::max();
#else
    const uint64_t MAX = LLONG_MAX;
#endif
    assert(!CHECK || (((uint64_t)value) <= MAX));
#endif
#endif
    return coord_t(value);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Domain::Domain(const DomainT<DIM,T> &other)
    : is_id(other.sparsity.id),
      is_type((is_id > 0) ? 
          Internal::NT_TemplateHelper::template encode_tag<DIM,T>() : 0),
      dim(DIM)
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    for (int i = 0; i < DIM; i++)
      rect_data[i] = check_for_overflow<T>(other.bounds.lo[i]);
    for (int i = 0; i < DIM; i++)
      rect_data[DIM+i] = check_for_overflow<T>(other.bounds.hi[i]);
    // Zero-out the other values in rect data to keep compilers happy
    for (int i = 2*DIM; i < 2*MAX_RECT_DIM; i++)
      rect_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain& Domain::operator=(const Domain &other)
  //----------------------------------------------------------------------------
  {
    is_id = other.is_id;
    // Like this for backwards compatibility
    is_type = (is_id > 0) ? other.is_type : 0;
    dim = other.dim;
    for(int i = 0; i < 2*dim; i++)
      rect_data[i] = other.rect_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain& Domain::operator=(Domain &&other) noexcept
  //----------------------------------------------------------------------------
  {
    is_id = other.is_id;
    other.is_id = 0;
    // Like this for backwards compatibility
    is_type = (is_id > 0) ? other.is_type : 0;
    other.is_type = 0;
    dim = other.dim;
    for(int i = 0; i < 2*dim; i++)
      rect_data[i] = other.rect_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Domain& Domain::operator=(const Rect<DIM,T> &other)
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    is_id = 0;
    is_type = 0;
    dim = DIM;
    for (int i = 0; i < DIM; i++)
      rect_data[i] = check_for_overflow<T>(other.lo[i]);
    for (int i = 0; i < DIM; i++)
      rect_data[DIM+i] = check_for_overflow<T>(other.hi[i]);
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Domain& Domain::operator=(const DomainT<DIM,T> &other)
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    dim = DIM;
    is_id = other.sparsity.id;
    if (is_id > 0)
      is_type = Internal::NT_TemplateHelper::template encode_tag<DIM,T>();
    else
      is_type = 0;
    for (int i = 0; i < DIM; i++)
      rect_data[i] = check_for_overflow<T>(other.bounds.lo[i]);
    for (int i = 0; i < DIM; i++)
      rect_data[DIM+i] = check_for_overflow<T>(other.bounds.hi[i]);
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool Domain::operator==(const Domain &rhs) const
  //----------------------------------------------------------------------------
  {
    if(is_id != rhs.is_id) return false;
    // No need to check type tag, equivalence subsumed by sparsity id test
    if(dim != rhs.dim) return false;
    for(int i = 0; i < dim; i++) {
      if(rect_data[i*2] != rhs.rect_data[i*2]) return false;
      if(rect_data[i*2+1] != rhs.rect_data[i*2+1]) return false;
    }
    return true;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool Domain::operator!=(const Domain &rhs) const
  //----------------------------------------------------------------------------
  {
    return !(*this == rhs);
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool Domain::operator<(const Domain &rhs) const
  //----------------------------------------------------------------------------
  {
    if(is_id < rhs.is_id) return true;
    if(is_id > rhs.is_id) return false;
    // No need to check type tag, subsumed by sparsity id test
    if(dim < rhs.dim) return true;
    if(dim > rhs.dim) return false;
    for(int i = 0; i < 2*dim; i++) {
      if(rect_data[i] < rhs.rect_data[i]) return true;
      if(rect_data[i] > rhs.rect_data[i]) return false;
    }
    return false; // otherwise they are equal
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain Domain::operator+(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    assert(dense());
    DomainPoint lo = this->lo() + rhs;
    DomainPoint hi = this->hi() + rhs;
    return Domain(lo, hi);
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain& Domain::operator+=(const DomainPoint &rhs)
  //----------------------------------------------------------------------------
  {
    assert(dense());
    assert(get_dim() == rhs.get_dim());
    for (int i = 0; i < dim; i++)
    {
      rect_data[i] += rhs[i];
      rect_data[dim+i] += rhs[i];
    }
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain Domain::operator-(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    assert(dense());
    DomainPoint lo = this->lo() - rhs;
    DomainPoint hi = this->hi() - rhs;
    return Domain(lo, hi);
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline Domain& Domain::operator-=(const DomainPoint &rhs)
  //----------------------------------------------------------------------------
  {
    assert(dense());
    assert(get_dim() == rhs.get_dim());
    for (int i = 0; i < dim; i++)
    {
      rect_data[i] -= rhs[i];
      rect_data[dim+i] -= rhs[i];
    }
    return *this;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool Domain::exists(void) const
  //----------------------------------------------------------------------------
  {
    return (dim > 0);
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool Domain::dense(void) const
  //----------------------------------------------------------------------------
  {
    return (is_id == 0);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Rect<DIM,T> Domain::bounds(void) const
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    assert(DIM == dim);
    Rect<DIM,T> result;
    for (int i = 0; i < DIM; i++)
      result.lo[i] = rect_data[i];
    for (int i = 0; i < DIM; i++)
      result.hi[i] = rect_data[DIM+i];
    return result;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Domain::operator Rect<DIM,T>(void) const
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    assert(DIM == dim);
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    if (is_id != 0)
      fprintf(stderr,"ERROR: Cannot implicitly convert sparse Domain to Rect");
#endif
    assert(is_id == 0); // better not be one of these
    Rect<DIM,T> result;
    for (int i = 0; i < DIM; i++)
      result.lo[i] = rect_data[i];
    for (int i = 0; i < DIM; i++)
      result.hi[i] = rect_data[DIM+i];
    return result;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline Domain::operator DomainT<DIM,T>(void) const
  //----------------------------------------------------------------------------
  {
    static_assert(std::is_integral<T>::value, "must be integral type");
    assert(DIM == dim);
    DomainT<DIM,T> result;
    if (is_id > 0)
    {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      TypeTag type = Internal::NT_TemplateHelper::template encode_tag<DIM,T>();
      assert(is_type == type); 
#endif
#endif
      result.sparsity.id = is_id;
    }
    else
      result.sparsity.id = 0;
    for (int i = 0; i < DIM; i++)
      result.bounds.lo[i] = rect_data[i];
    for (int i = 0; i < DIM; i++)
      result.bounds.hi[i] = rect_data[DIM+i];
    return result;
  }

  //----------------------------------------------------------------------------
  /*static*/ inline Domain Domain::from_domain_point(const DomainPoint &p)
  //----------------------------------------------------------------------------
  {
    switch (p.dim) {
      case 0:
        assert(false);
#define DIMFUNC(DIM) \
      case DIM: \
        return Domain(p, p);
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }
    return Domain::NO_DOMAIN;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool Domain::is_valid(void) const
  //----------------------------------------------------------------------------
  {
    return exists();
  }

  //----------------------------------------------------------------------------
  inline bool Domain::contains(const DomainPoint &point) const
  //----------------------------------------------------------------------------
  {
    assert(point.get_dim() == dim);
    bool result = false;
    if (is_id > 0)
    {
      ContainsFunctor functor(*this, point, result);
      Internal::NT_TemplateHelper::demux<ContainsFunctor>(is_type, &functor);
      return result;
    }
    else
    {
      switch (dim)
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            Point<DIM,coord_t> p1 = point; \
            Rect<DIM,coord_t> rect = *this; \
            result = rect.contains(p1); \
            break; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
    }
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline bool Domain::contains_bounds_only(
                                                 const DomainPoint &point) const
  //----------------------------------------------------------------------------
  {
    assert(point.get_dim() == dim);
    for (int i = 0; i < dim; i++)
      if (point[i] < rect_data[i])
        return false;
      else if (point[i] > rect_data[dim+i])
        return false;
    return true;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline int Domain::get_dim(void) const
  //----------------------------------------------------------------------------
  {
    return dim;
  }

  //----------------------------------------------------------------------------
  inline bool Domain::empty(void) const
  //----------------------------------------------------------------------------
  {
    return (get_volume() == 0);
  }

  //----------------------------------------------------------------------------
  inline void Domain::destroy(Realm::Event wait_on)
  //----------------------------------------------------------------------------
  {
    if (!dense())
    {
      DestroyFunctor functor(*this, wait_on);
      Internal::NT_TemplateHelper::demux<DestroyFunctor>(is_type, &functor);
    }
    is_type = 0;
    is_id = 0;
    dim = 0;
  }

  //----------------------------------------------------------------------------
  inline size_t Domain::get_volume(void) const
  //----------------------------------------------------------------------------
  {
    if (dense())
    {
      switch (dim)
      {
        case 0:
          return 0;
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            Rect<DIM,coord_t> rect = *this; \
            return rect.volume(); \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      return 0;
    }
    else
    {
      size_t result = 0;
      VolumeFunctor functor(*this, result);
      Internal::NT_TemplateHelper::demux<VolumeFunctor>(is_type, &functor);
      return result;
    }
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint Domain::lo(void) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = rect_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  __CUDA_HD__ inline DomainPoint Domain::hi(void) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = rect_data[dim+i];
    return result;
  }

  //----------------------------------------------------------------------------
  inline Domain Domain::intersection(const Domain &other) const
  //----------------------------------------------------------------------------
  {
    assert(dim == other.dim);
    // Only allow for one of these to have a sparsity map currently
    assert(dense() || other.dense());
    assert((is_type == other.is_type) ||
              (is_type == 0) || (other.is_type == 0));
    if (!dense() || !other.dense())
    {
      Domain result;
      IntersectionFunctor functor(*this, other, result);
      Internal::NT_TemplateHelper::demux<IntersectionFunctor>(
          (is_id > 0) ? is_type : other.is_type, &functor); 
      return result;
    }
    else
    {
      switch (dim)
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            Rect<DIM,coord_t> rect1 = *this; \
            Rect<DIM,coord_t> rect2 = other; \
            Rect<DIM,coord_t> result = rect1.intersection(rect2); \
            return Domain(result); \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
    }
    return Domain::NO_DOMAIN;
  }

  //----------------------------------------------------------------------------
  inline Domain Domain::convex_hull(const DomainPoint &p) const
  //----------------------------------------------------------------------------
  {
    assert(dim == p.dim);
    Realm::ProfilingRequestSet dummy_requests;
    switch (dim)
    {
#define DIMFUNC(DIM) \
      case DIM: \
        { \
          Rect<DIM,coord_t> rect1 = *this; \
          Rect<DIM,coord_t> rect2(p, p); \
          Rect<DIM,coord_t> result = rect1.union_bbox(rect2); \
          return Domain(result); \
        }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }
    return Domain::NO_DOMAIN;
  }

  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator::DomainPointIterator(const Domain &d)
    : is_type(d.is_type)
  //----------------------------------------------------------------------------
  {
    p.dim = d.get_dim();
    if (d.dense())
    {
      // We've just got a rect so we can do the dumb thing
      switch (p.get_dim()) {
      case 0:
        {
          is_valid = false;
          break;
        }
#define DIMFUNC(DIM) \
      case DIM: \
        { \
          Rect<DIM,coord_t> rect = d; \
          Realm::PointInRectIterator<DIM,coord_t> rect_itr(rect); \
          static_assert(sizeof(rect_itr) <= sizeof(rect_iterator)); \
          rect_valid = rect_itr.valid; \
          if (rect_valid) { \
            is_valid = true; \
            p = rect_itr.p; \
            memcpy(rect_iterator, &rect_itr, sizeof(rect_itr)); \
          } else { \
            is_valid = false; \
          } \
          break; \
        }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(0);
      };
    }
    else
    {
      IteratorInitFunctor functor(d, *this);
      Internal::NT_TemplateHelper::demux<IteratorInitFunctor>(d.is_type, 
                                                              &functor);
    }
  }

  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator::DomainPointIterator(
                                                const DomainPointIterator &rhs)
    : p(rhs.p), is_type(rhs.is_type), is_valid(rhs.is_valid),
      rect_valid(rhs.rect_valid)
  //----------------------------------------------------------------------------
  {
    memcpy(is_iterator, rhs.is_iterator,
      sizeof(Realm::IndexSpaceIterator<MAX_RECT_DIM,coord_t>));
    memcpy(rect_iterator, rhs.rect_iterator,
      sizeof(Realm::PointInRectIterator<MAX_RECT_DIM,coord_t>));
  }

  //----------------------------------------------------------------------------
  inline bool Domain::DomainPointIterator::step(void)
  //----------------------------------------------------------------------------
  {
    assert(is_valid && rect_valid);
    // Step the rect iterator first and see if we can just get a new point
    // from the rect iterator without needing to demux
    switch (p.get_dim()) 
    {
#define DIMFUNC(DIM) \
      case DIM: \
        { \
          Realm::PointInRectIterator<DIM,coord_t> rect_itr; \
          memcpy(&rect_itr, rect_iterator, sizeof(rect_itr)); \
          rect_itr.step(); \
          rect_valid = rect_itr.valid; \
          if (rect_valid) { \
            p = rect_itr.p; \
            memcpy(rect_iterator, &rect_itr, sizeof(rect_itr)); \
          } \
          break; \
        }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(0);
    }
    if (!rect_valid && (is_type > 0))
    {
      // If we had a sparsity map, try to step the index space iterator
      // to the next rectangle using a demux
      IteratorStepFunctor functor(*this);
      Internal::NT_TemplateHelper::demux<IteratorStepFunctor>(is_type,
                                                              &functor);
    }
    return is_valid && rect_valid;
  }

  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator::operator bool(void) const
  //----------------------------------------------------------------------------
  {
    return is_valid && rect_valid;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint& Domain::DomainPointIterator::operator*(void)
  //----------------------------------------------------------------------------
  {
    return p;
  }
  
  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator& Domain::DomainPointIterator::operator=(
                                                 const DomainPointIterator &rhs)
  //----------------------------------------------------------------------------
  {
    p = rhs.p;
    memcpy(is_iterator, rhs.is_iterator, 
      sizeof(Realm::IndexSpaceIterator<MAX_RECT_DIM,coord_t>));
    memcpy(rect_iterator, rhs.rect_iterator,
      sizeof(Realm::PointInRectIterator<MAX_RECT_DIM,coord_t>));
    is_type = rhs.is_type;
    is_valid = rhs.is_valid;
    rect_valid = rhs.rect_valid;
    return *this;
  }

  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator& Domain::DomainPointIterator::operator++(
                                                                          void)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator Domain::DomainPointIterator::operator++(
                                                                int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    Domain::DomainPointIterator result(*this);
    step();
    return result;
  }

  //----------------------------------------------------------------------------
  inline std::ostream& operator<<(std::ostream &os, const Domain &d)
  //----------------------------------------------------------------------------
  {
    os << "{ bounds = ";
    switch(d.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: { os << d.bounds<DIM,coord_t>(); break; }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: assert(0);
    }
    if (d.is_id != 0)
      os << ", sparse = " << std::hex << d.is_id << std::dec;
    else
      os << ", dense";
    os << " }";
    return os;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline PointInRectIterator<DIM,COORD_T>::PointInRectIterator(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline PointInRectIterator<DIM,COORD_T>::PointInRectIterator(
             const Rect<DIM,COORD_T> &r, bool column_major_order)
    : itr(Realm::PointInRectIterator<DIM,COORD_T>(r, column_major_order))
  //----------------------------------------------------------------------------
  {
    assert(valid());
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline bool PointInRectIterator<DIM,COORD_T>::valid(void) const
  //----------------------------------------------------------------------------
  {
    return itr.valid;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline bool PointInRectIterator<DIM,COORD_T>::step(void)
  //----------------------------------------------------------------------------
  {
    assert(valid());
    itr.step();
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline bool PointInRectIterator<DIM,COORD_T>::operator()(void) const
  //----------------------------------------------------------------------------
  {
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline Point<DIM,COORD_T> 
                         PointInRectIterator<DIM,COORD_T>::operator*(void) const
  //----------------------------------------------------------------------------
  {
    return itr.p;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline COORD_T 
              PointInRectIterator<DIM,COORD_T>::operator[](unsigned index) const
  //----------------------------------------------------------------------------
  {
    return itr.p[index];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline const Point<DIM,COORD_T>* 
                        PointInRectIterator<DIM,COORD_T>::operator->(void) const
  //----------------------------------------------------------------------------
  {
    return &(itr.p);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline PointInRectIterator<DIM,COORD_T>&
                              PointInRectIterator<DIM,COORD_T>::operator++(void)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> __CUDA_HD__
  inline PointInRectIterator<DIM,COORD_T>
                    PointInRectIterator<DIM,COORD_T>::operator++(int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    PointInRectIterator<DIM,COORD_T> result(*this);
    step();
    return result;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline RectInDomainIterator<DIM,COORD_T>::RectInDomainIterator(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline RectInDomainIterator<DIM,COORD_T>::RectInDomainIterator(
                                       const DomainT<DIM,COORD_T> &d)
    : itr(Realm::IndexSpaceIterator<DIM,COORD_T>(d))
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool RectInDomainIterator<DIM,COORD_T>::valid(void) const
  //----------------------------------------------------------------------------
  {
    return itr.valid;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool RectInDomainIterator<DIM,COORD_T>::step(void)
  //----------------------------------------------------------------------------
  {
    assert(valid());
    itr.step();
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool RectInDomainIterator<DIM,COORD_T>::operator()(void) const
  //----------------------------------------------------------------------------
  {
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline Rect<DIM,COORD_T>
                        RectInDomainIterator<DIM,COORD_T>::operator*(void) const
  //----------------------------------------------------------------------------
  {
    return itr.rect;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline const Rect<DIM,COORD_T>*
                       RectInDomainIterator<DIM,COORD_T>::operator->(void) const
  //----------------------------------------------------------------------------
  {
    return &(itr.rect);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline RectInDomainIterator<DIM,COORD_T>& 
                             RectInDomainIterator<DIM,COORD_T>::operator++(void)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline RectInDomainIterator<DIM,COORD_T>
                   RectInDomainIterator<DIM,COORD_T>::operator++(int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    RectInDomainIterator<DIM,COORD_T> result(*this);
    step();
    return result;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInDomainIterator<DIM,COORD_T>::PointInDomainIterator(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInDomainIterator<DIM,COORD_T>::PointInDomainIterator(
              const DomainT<DIM,COORD_T> &d, bool column_major_order)
    : rect_itr(RectInDomainIterator<DIM,COORD_T>(d)),
      column_major(column_major_order)
  //----------------------------------------------------------------------------
  {
    if (rect_itr())
      point_itr = PointInRectIterator<DIM,COORD_T>(*rect_itr, column_major);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInDomainIterator<DIM,COORD_T>::valid(void) const
  //----------------------------------------------------------------------------
  {
    return point_itr();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInDomainIterator<DIM,COORD_T>::step(void)
  //----------------------------------------------------------------------------
  {
    assert(valid()); 
    point_itr++;
    if (!point_itr())
    {
      rect_itr++;
      if (rect_itr())
        point_itr = PointInRectIterator<DIM,COORD_T>(*rect_itr, column_major);
    }
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInDomainIterator<DIM,COORD_T>::operator()(void) const
  //----------------------------------------------------------------------------
  {
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline Point<DIM,COORD_T> 
                       PointInDomainIterator<DIM,COORD_T>::operator*(void) const
  //----------------------------------------------------------------------------
  {
    return *point_itr;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline const Point<DIM,COORD_T>* 
                      PointInDomainIterator<DIM,COORD_T>::operator->(void) const
  //----------------------------------------------------------------------------
  {
    return &(*point_itr);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline COORD_T 
            PointInDomainIterator<DIM,COORD_T>::operator[](unsigned index) const
  //----------------------------------------------------------------------------
  {
    return point_itr[index];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInDomainIterator<DIM,COORD_T>&
                            PointInDomainIterator<DIM,COORD_T>::operator++(void)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInDomainIterator<DIM,COORD_T>
                  PointInDomainIterator<DIM,COORD_T>::operator++(int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    PointInDomainIterator<DIM,COORD_T> result(*this);
    step();
    return result;
  }

} // namespace Legion
