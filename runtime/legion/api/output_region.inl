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

// Included from output_region.h - do not include this directly

// Useful for IDEs
#include "legion/api/output_region.h"

namespace Legion {

    //--------------------------------------------------------------------------
    template<typename T, int DIM, typename COORD_T, bool CHECK_BOUNDS>
    DeferredBuffer<T,DIM,COORD_T,CHECK_BOUNDS> OutputRegion::create_buffer(
                                              const Point<DIM,COORD_T> &extents,
                                              FieldID field_id,
                                              const T *initial_value /*= NULL*/,
                                              bool return_buffer /*= false*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      check_type_tag(
        Internal::NT_TemplateHelper::encode_tag<DIM, COORD_T>());
#endif

      Rect<DIM> bounds(Point<DIM>::ZEROES(), extents - Point<DIM>::ONES());

      std::vector<DimensionKind> ordering;
      size_t alignment;
      get_layout(field_id, ordering, alignment);
      std::array<DimensionKind, DIM> ord;
#ifdef DEBUG_LEGION
      assert(ordering.size() == DIM);
#endif
      std::copy(ordering.begin(), ordering.end(), ord.begin());

      DeferredBuffer<T,DIM,COORD_T,CHECK_BOUNDS> buffer(
        bounds, target_memory(), ord, initial_value, alignment);
      if (return_buffer)
        return_data(extents, field_id, buffer.instance, false);
      return buffer;
    }

    //--------------------------------------------------------------------------
    template<typename T, int DIM, typename COORD_T, bool CHECK_BOUNDS>
    void OutputRegion::return_data(
                             const Point<DIM,COORD_T> &extents,
                             FieldID field_id,
                             DeferredBuffer<T,DIM,COORD_T,CHECK_BOUNDS> &buffer)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      check_type_tag(
        Internal::NT_TemplateHelper::encode_tag<DIM, COORD_T>());
      check_field_size(field_id, sizeof(T));
#endif
      // Populate the layout constraints for the returned buffer
      // for the constraint checks.
      LayoutConstraintSet constraints;
      std::vector<DimensionKind> ordering(DIM + 1);
      for (int32_t i = 0; i < DIM; ++i) ordering[i] = buffer.ordering[i];
      ordering[DIM] = LEGION_DIM_F;
      constraints.ordering_constraint = OrderingConstraint(ordering, false);
      constraints.alignment_constraints.push_back(
        AlignmentConstraint(field_id, LEGION_EQ_EK, buffer.alignment));

      return_data(extents, field_id, buffer.instance, &constraints, true);
    }

} // namespace Legion
