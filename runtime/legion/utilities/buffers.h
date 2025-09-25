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

#ifndef __LEGION_BUFFERS_H__
#define __LEGION_BUFFERS_H__

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Buffer Manager
    /////////////////////////////////////////////////////////////

    /**
     * A class that is helpful in keeping track of owned buffer
     */
    template<typename T, AllocationLifetime L>
    class BufferManager : public NoHeapify {
    public:
      BufferManager(void) : buffer(nullptr), size(0) { }
      BufferManager(const BufferManager& rhs) = delete;
      BufferManager(BufferManager&& rhs) noexcept
        : buffer(rhs.buffer), size(rhs.size)
      {
        rhs.buffer = nullptr;
        rhs.size = 0;
      }
      ~BufferManager(void) { clear(); }
    public:
      BufferManager& operator=(const BufferManager& rhs) = delete;
      BufferManager& operator=(BufferManager&& rhs) noexcept
      {
        save_buffer(rhs.buffer, rhs.size);
        rhs.buffer = nullptr;
        rhs.size = 0;
        return *this;
      }
    public:
      inline void clear(void)
      {
        if (buffer != nullptr)
        {
          legion_free<void, BufferManager<T, L> >(buffer, size);
          buffer = nullptr;
          size = 0;
        }
      }
      inline void save_buffer(const void* src, size_t sz)
      {
        if (buffer != nullptr)
          legion_free<void, BufferManager<T, L> >(buffer, size);
        size = sz;
        if (size > 0)
        {
          buffer = legion_malloc<void, L, BufferManager<T, L> >(
              size, alignof(uint8_t));
          std::memcpy(buffer, src, size);
        }
        else
        {
          buffer = nullptr;
        }
      }
      inline void* get_buffer(void) const { return buffer; }
      inline size_t get_size(void) const { return size; }
    private:
      void* buffer;
      size_t size;
    };

    /////////////////////////////////////////////////////////////
    // Semantic Info
    /////////////////////////////////////////////////////////////

    /**
     * \struct SemanticInfo
     * A struct for storing semantic information for various things
     */
    struct SemanticInfo {
    public:
      SemanticInfo(void) : is_mutable(false) { }
      SemanticInfo(const void* buf, size_t s, bool is_mut = true)
        : is_mutable(is_mut)
      {
        buffer.save_buffer(buf, s);
      }
      SemanticInfo(RtUserEvent ready) : ready_event(ready), is_mutable(true) { }
    public:
      inline bool is_valid(void) const { return ready_event.has_triggered(); }
    public:
      BufferManager<SemanticInfo, LONG_LIFETIME> buffer;
      RtUserEvent ready_event;
      bool is_mutable;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_BUFFERS_H__
