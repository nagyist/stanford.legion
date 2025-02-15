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

// Included from serdez.h - do not include this directly

// Useful for IDEs
#include "legion/utilities/serdez.h"

namespace Legion {

    //--------------------------------------------------------------------------
    inline Serializer& Serializer::operator=(const Serializer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Serializer::serialize(const T &element)
    //--------------------------------------------------------------------------
    {
      static_assert(std::is_trivially_copyable<T>::value);
      while ((index + sizeof(T)) > total_bytes)
        resize();
      memcpy(buffer+index, (const void*)&element, sizeof(T));
      index += sizeof(T);
#ifdef DEBUG_LEGION
      context_bytes += sizeof(T);
#endif
    }

    //--------------------------------------------------------------------------
    template<>
    inline void Serializer::serialize<bool>(const bool &element)
    //--------------------------------------------------------------------------
    {
      const uint32_t flag = element ? 1 : 0;
      serialize<uint32_t>(flag);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Serializer::serialize(const Internal::BitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Serializer::serialize(const Internal::TLBitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

#ifdef __SSE2__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::SSEBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::SSETLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

#ifdef __AVX__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::AVXBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::AVXTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

#ifdef __ALTIVEC__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::PPCBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::PPCTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

#ifdef __ARM_NEON
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::NeonBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Serializer::serialize(const Internal::NeonTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.serialize(*this);
    }
#endif

    //--------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void Serializer::serialize(const Internal::CompoundBitMask<DT,BLOAT,BIDIR> &m)
    //--------------------------------------------------------------------------
    {
      m.serialize(*this);
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const Domain &dom)
    //--------------------------------------------------------------------------
    {
      serialize(dom.is_id);
      if (dom.is_id > 0)
        serialize(dom.is_type);
      serialize(dom.dim);
      for (int i = 0; i < 2*dom.dim; i++)
        serialize(dom.rect_data[i]);
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      serialize(dp.dim);
      if (dp.dim == 0)
        serialize(dp.point_data[0]);
      else
      {
        for (int idx = 0; idx < dp.dim; idx++)
          serialize(dp.point_data[idx]);
      }
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const Internal::CopySrcDstField &field)
    //--------------------------------------------------------------------------
    {
      serialize(field.inst);
      serialize(field.field_id);
      serialize(field.redop_id);
      if (field.redop_id > 0)
      {
        serialize<bool>(field.red_fold);
        serialize<bool>(field.red_exclusive);
      }
      serialize(field.serdez_id);
      serialize(field.subfield_offset);
      serialize(field.indirect_index);
      serialize(field.size);
      // we know if there's a fill value if the field ID is -1
      if (field.field_id == (Realm::FieldID)-1)
      {
        if (field.size <= Internal::CopySrcDstField::MAX_DIRECT_SIZE)
          serialize(field.fill_data.direct, field.size);
        else
          serialize(field.fill_data.indirect, field.size);
      }
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const void *src, size_t bytes)
    //--------------------------------------------------------------------------
    {
      while ((index + bytes) > total_bytes)
        resize();
      memcpy(buffer+index,src,bytes);
      index += bytes;
#ifdef DEBUG_LEGION
      context_bytes += bytes;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Serializer::begin_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      while ((index + sizeof(context_bytes)) > total_bytes)
        resize();
      memcpy(buffer+index, &context_bytes, sizeof(context_bytes));
      index += sizeof(context_bytes);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Serializer::end_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Save the size into the buffer
      while ((index + sizeof(context_bytes)) > total_bytes)
        resize();
      memcpy(buffer+index, &context_bytes, sizeof(context_bytes));
      index += sizeof(context_bytes);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline void* Serializer::reserve_bytes(size_t bytes)
    //--------------------------------------------------------------------------
    {
      while ((index + bytes) > total_bytes)
        resize();
      void *result = buffer+index;
      index += bytes;
#ifdef DEBUG_LEGION
      context_bytes += bytes;
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline void Serializer::reset(void)
    //--------------------------------------------------------------------------
    {
      index = 0;
#ifdef DEBUG_LEGION
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Serializer::resize(void)
    //--------------------------------------------------------------------------
    {
      // Double the buffer size
      total_bytes *= 2;
#ifdef DEBUG_LEGION
      assert(total_bytes != 0); // this would cause deallocation
#endif
      uint8_t *next = (uint8_t*)realloc(buffer,total_bytes);
#ifdef DEBUG_LEGION
      assert(next != NULL);
#endif
      buffer = next;
    }

    //--------------------------------------------------------------------------
    inline Deserializer& Deserializer::operator=(const Deserializer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Deserializer::deserialize(T &element)
    //--------------------------------------------------------------------------
    {
      static_assert(std::is_trivially_copyable<T>::value);
#ifdef DEBUG_LEGION
      // Check to make sure we don't read past the end
      assert((index+sizeof(T)) <= total_bytes);
#endif
      memcpy(&element, buffer+index, sizeof(T));
      index += sizeof(T);
#ifdef DEBUG_LEGION
      context_bytes += sizeof(T);
#endif
    }

    //--------------------------------------------------------------------------
    template<>
    inline void Deserializer::deserialize<bool>(bool &element)
    //--------------------------------------------------------------------------
    {
      uint32_t flag;
      deserialize<uint32_t>(flag);
      element = (flag != 0);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Deserializer::deserialize(Internal::BitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<typename T, unsigned int MAX, unsigned SHIFT, unsigned MASK>
    inline void Deserializer::deserialize(Internal::TLBitMask<T,MAX,SHIFT,MASK> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

#ifdef __SSE2__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::SSEBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::SSETLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

#ifdef __AVX__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::AVXBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::AVXTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

#ifdef __ALTIVEC__
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::PPCBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::PPCTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

#ifdef __ARM_NEON
    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::NeonBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    template<unsigned int MAX>
    inline void Deserializer::deserialize(Internal::NeonTLBitMask<MAX> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }
#endif

    //--------------------------------------------------------------------------
    template<typename DT, unsigned BLOAT, bool BIDIR>
    inline void Deserializer::deserialize(Internal::CompoundBitMask<DT,BLOAT,BIDIR> &mask)
    //--------------------------------------------------------------------------
    {
      mask.deserialize(*this);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(Domain &dom)
    //--------------------------------------------------------------------------
    {
      deserialize(dom.is_id);
      if (dom.is_id > 0)
        deserialize(dom.is_type);
      deserialize(dom.dim);
      for (int i = 0; i < 2*dom.dim; i++)
        deserialize(dom.rect_data[i]);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      deserialize(dp.dim);
      if (dp.dim == 0)
        deserialize(dp.point_data[0]);
      else
      {
        for (int idx = 0; idx < dp.dim; idx++)
          deserialize(dp.point_data[idx]);
      }
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(Internal::CopySrcDstField &field)
    //--------------------------------------------------------------------------
    {
      deserialize(field.inst);
      deserialize(field.field_id);
      deserialize(field.redop_id);
      if (field.redop_id > 0)
      {
        deserialize<bool>(field.red_fold);
        deserialize<bool>(field.red_exclusive);
      }
      deserialize(field.serdez_id);
      deserialize(field.subfield_offset);
      deserialize(field.indirect_index);
      if (field.size > Internal::CopySrcDstField::MAX_DIRECT_SIZE)
      {
        free(field.fill_data.indirect);
        field.fill_data.indirect = NULL;
      }
      deserialize(field.size);
      // we know if there's a fill value if the field ID is -1
      if (field.field_id == (Realm::FieldID)-1)
      {
        if (field.size > Internal::CopySrcDstField::MAX_DIRECT_SIZE)
        {
          field.fill_data.indirect = malloc(field.size);
          deserialize(field.fill_data.indirect, field.size);
        }
        else
          deserialize(field.fill_data.direct, field.size);
      }
    }
      
    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(void *dst, size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((index + bytes) <= total_bytes);
#endif
      memcpy(dst,buffer+index,bytes);
      index += bytes;
#ifdef DEBUG_LEGION
      context_bytes += bytes;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::begin_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Save our enclosing context on the stack
#ifndef NDEBUG
      decltype(context_bytes) sent_context = 0;
      memcpy(&sent_context, buffer+index, sizeof(sent_context));
#endif
      index += sizeof(context_bytes);
      // Check to make sure that they match
      assert(sent_context == context_bytes);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::end_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Read the send context size out of the buffer      
#ifndef NDEBUG
      decltype(context_bytes) sent_context = 0;
      memcpy(&sent_context, buffer+index, sizeof(sent_context));
#endif
      index += sizeof(context_bytes);
      // Check to make sure that they match
      assert(sent_context == context_bytes);
      context_bytes = 0;
#endif
    }

    //--------------------------------------------------------------------------
    inline size_t Deserializer::get_remaining_bytes(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index <= total_bytes);
#endif
      return total_bytes - index;
    }

    //--------------------------------------------------------------------------
    inline const void* Deserializer::get_current_pointer(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index <= total_bytes);
#endif
      return (const void*)(buffer+index);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::advance_pointer(size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((index+bytes) <= total_bytes);
      context_bytes += bytes;
#endif
      index += bytes;
    }

} // namespace Legion
