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

#ifndef __LEGION_SERDEZ_H__
#define __LEGION_SERDEZ_H__

#include "legion/api/geometry.h"
#include "legion/utilities/bitmask.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Serializer
  /////////////////////////////////////////////////////////////
  class Serializer {
  public:
    Serializer(size_t base_bytes = 4096)
      : total_bytes(base_bytes), buffer((uint8_t*)malloc(base_bytes)), index(0)
#ifdef DEBUG_LEGION
        ,
        context_bytes(0)
#endif
    { }
    Serializer(const Serializer& rhs) = delete;
  public:
    ~Serializer(void) { free(buffer); }
  public:
    Serializer& operator=(const Serializer& rhs) = delete;
  public:
    template<typename T>
    inline void serialize(const T& element);
    // we need special serializers for bit masks
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void serialize(const Internal::BitMask<T, MAX, SHIFT, MASK>& mask);
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void serialize(const Internal::TLBitMask<T, MAX, SHIFT, MASK>& mask);
#ifdef __SSE2__
    template<unsigned int MAX>
    inline void serialize(const Internal::SSEBitMask<MAX>& mask);
    template<unsigned int MAX>
    inline void serialize(const Internal::SSETLBitMask<MAX>& mask);
#endif
#ifdef __AVX__
    template<unsigned int MAX>
    inline void serialize(const Internal::AVXBitMask<MAX>& mask);
    template<unsigned int MAX>
    inline void serialize(const Internal::AVXTLBitMask<MAX>& mask);
#endif
#ifdef __ALTIVEC__
    template<unsigned int MAX>
    inline void serialize(const Internal::PPCBitMask<MAX>& mask);
    template<unsigned int MAX>
    inline void serialize(const Internal::PPCTLBitMask<MAX>& mask);
#endif
#ifdef __ARM_NEON
    template<unsigned int MAX>
    inline void serialize(const Internal::NeonBitMask<MAX>& mask);
    template<unsigned int MAX>
    inline void serialize(const Internal::NeonTLBitMask<MAX>& mask);
#endif
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void serialize(
        const Internal::CompoundBitMask<DT, BLOAT, BIDIR>& mask);
    inline void serialize(const Domain& domain);
    inline void serialize(const DomainPoint& dp);
    inline void serialize(const Internal::CopySrcDstField& field);
    inline void serialize(const void* src, size_t bytes);
  public:
    inline void begin_context(void);
    inline void end_context(void);
  public:
    inline size_t get_index(void) const { return index; }
    inline const void* get_buffer(void) const { return buffer; }
    inline size_t get_buffer_size(void) const { return total_bytes; }
    inline size_t get_used_bytes(void) const { return index; }
    inline void* reserve_bytes(size_t size);
    inline void reset(void);
  private:
    inline void resize(void);
  private:
    size_t total_bytes;
    uint8_t* buffer;
    size_t index;
#ifdef DEBUG_LEGION
    size_t context_bytes;
#endif
  };

  /////////////////////////////////////////////////////////////
  // Deserializer
  /////////////////////////////////////////////////////////////
  class Deserializer {
  public:
    Deserializer(
        const void* buf, size_t buffer_size
#ifdef DEBUG_LEGION
        ,
        size_t ctx_bytes = 0
#endif
        )
      : total_bytes(buffer_size), buffer((const uint8_t*)buf), index(0)
#ifdef DEBUG_LEGION
        ,
        context_bytes(ctx_bytes)
#endif
    { }
    Deserializer(const Deserializer& rhs) = delete;
  public:
    ~Deserializer(void)
    {
#ifdef DEBUG_LEGION
      // should have used the whole buffer
      assert(index == total_bytes);
#endif
    }
  public:
    Deserializer& operator=(const Deserializer& rhs) = delete;
  public:
    template<typename T>
    inline void deserialize(T& element);
    // We need specialized deserializers for bit masks
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void deserialize(Internal::BitMask<T, MAX, SHIFT, MASK>& mask);
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void deserialize(Internal::TLBitMask<T, MAX, SHIFT, MASK>& mask);
#ifdef __SSE2__
    template<unsigned int MAX>
    inline void deserialize(Internal::SSEBitMask<MAX>& mask);
    template<unsigned int MAX>
    inline void deserialize(Internal::SSETLBitMask<MAX>& mask);
#endif
#ifdef __AVX__
    template<unsigned int MAX>
    inline void deserialize(Internal::AVXBitMask<MAX>& mask);
    template<unsigned int MAX>
    inline void deserialize(Internal::AVXTLBitMask<MAX>& mask);
#endif
#ifdef __ALTIVEC__
    template<unsigned int MAX>
    inline void deserialize(Internal::PPCBitMask<MAX>& mask);
    template<unsigned int MAX>
    inline void deserialize(Internal::PPCTLBitMask<MAX>& mask);
#endif
#ifdef __ARM_NEON
    template<unsigned int MAX>
    inline void deserialize(Internal::NeonBitMask<MAX>& mask);
    template<unsigned int MAX>
    inline void deserialize(Internal::NeonTLBitMask<MAX>& mask);
#endif
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void deserialize(Internal::CompoundBitMask<DT, BLOAT, BIDIR>& mask);
    inline void deserialize(Domain& domain);
    inline void deserialize(DomainPoint& dp);
    inline void deserialize(Internal::CopySrcDstField& field);
    inline void deserialize(void* dst, size_t bytes);
  public:
    inline void begin_context(void);
    inline void end_context(void);
  public:
    inline size_t get_remaining_bytes(void) const;
    inline const void* get_current_pointer(void) const;
    inline void advance_pointer(size_t bytes);
  private:
    const size_t total_bytes;
    const uint8_t* buffer;
    size_t index;
#ifdef DEBUG_LEGION
    size_t context_bytes;
  public:
    inline size_t get_context_bytes(void) const { return context_bytes; }
#endif
  };

  /////////////////////////////////////////////////////////////
  // Rez Checker
  /////////////////////////////////////////////////////////////
  /*
   * Helps in making the calls to begin and end context for
   * both Serializer and Deserializer classes.
   */
  class RezCheck {
  public:
    RezCheck(Serializer& r) : rez(r) { rez.begin_context(); }
    RezCheck(RezCheck& rhs) = delete;
    ~RezCheck(void) { rez.end_context(); }
  public:
    RezCheck& operator=(const RezCheck& rhs) = delete;
  private:
    Serializer& rez;
  };
  // Same thing except for deserializers, yes we could template
  // it, but then we have to type out to explicitly instantiate
  // the template on the constructor call and that is a lot of
  // unnecessary typing.
  class DerezCheck {
  public:
    DerezCheck(Deserializer& r) : derez(r) { derez.begin_context(); }
    DerezCheck(DerezCheck& rhs) = delete;
    ~DerezCheck(void) { derez.end_context(); }
  public:
    DerezCheck& operator=(const DerezCheck& rhs) = delete;
  private:
    Deserializer& derez;
  };

}  // namespace Legion

#include "legion/utilities/serdez.inl"

#endif  // __LEGION_SERDEZ_H__
