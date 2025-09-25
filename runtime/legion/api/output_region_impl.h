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

#ifndef __LEGION_OUTPUT_REGION_IMPL_H__
#define __LEGION_OUTPUT_REGION_IMPL_H__

#include "legion/api/output_region.h"
#include "legion/kernel/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \class OutputRegionImpl
     * The base implementation of an output region object.
     *
     * Just like physical region impls, we don't need to make
     * output region impls thread safe, because they are accessed
     * exclusively by a single task.
     */
    class OutputRegionImpl
      : public Collectable,
        public Heapify<OutputRegionImpl, OPERATION_LIFETIME> {
    private:
      struct LayoutCreator {
      public:
        LayoutCreator(
            Realm::InstanceLayoutGeneric*& l, const Domain& d,
            const Realm::InstanceLayoutConstraints& c,
            const std::vector<int32_t>& d_order)
          : layout(l), domain(d), constraints(c), dim_order(d_order)
        { }
        template<typename DIM, typename COLOR_T>
        static inline void demux(LayoutCreator* creator)
        {
          legion_assert(creator->dim_order.size() == DIM::N);
          const DomainT<DIM::N, COLOR_T> bounds =
              Rect<DIM::N, COLOR_T>(creator->domain);
          creator->layout =
              Realm::InstanceLayoutGeneric::choose_instance_layout(
                  bounds, creator->constraints, creator->dim_order.data());
        }
      private:
        Realm::InstanceLayoutGeneric*& layout;
        const Domain& domain;
        const Realm::InstanceLayoutConstraints& constraints;
        const std::vector<int32_t>& dim_order;
      };
    public:
      OutputRegionImpl(
          unsigned index, const OutputRequirement& req,
          const InstanceSet& instance_set, TaskContext* ctx,
          const bool global_indexing, const bool valid,
          const bool grouped_fields);
      OutputRegionImpl(const OutputRegionImpl& rhs) = delete;
      ~OutputRegionImpl(void);
    public:
      OutputRegionImpl& operator=(const OutputRegionImpl& rhs) = delete;
    public:
      Memory target_memory(void) const;
    public:
      LogicalRegion get_logical_region(void) const;
      bool is_valid_output_region(void) const;
    public:
      void check_type_tag(TypeTag type_tag) const;
      void check_field_size(FieldID field_id, size_t field_size) const;
      void get_layout(
          FieldID field_id, std::vector<DimensionKind>& ordering,
          size_t& alignment) const;
      size_t get_field_size(FieldID field_id) const;
    public:
      void return_data(
          const DomainPoint& extents, FieldID field_id,
          PhysicalInstance instance, const LayoutConstraintSet* constraints,
          bool check_constraints);
    public:
      void finalize(RtEvent safe_effects);
    public:
      bool is_complete(FieldID& unbound_field) const;
    public:
      const OutputRequirement& get_requirement(void) const { return req; }
      DomainPoint get_extents(void) const { return extents; }
    protected:
      PhysicalManager* get_manager(FieldID field_id) const;
    public:
      TaskContext* const context;
      const OutputRequirement& req;
      RegionNode* const region;
      const unsigned index;
      const bool created_region;
      const bool global_indexing;
      // Either AOS or hybrid or contiguous SOA
      const bool grouped_fields;
    private:
      // Output data batched during task execution
      std::map<FieldID, PhysicalInstance> returned_instances;
      std::vector<PhysicalManager*> managers;
      DomainPoint extents;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_OUTPUT_REGION_IMPL_H__
