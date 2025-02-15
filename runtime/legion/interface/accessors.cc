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

#include "legion/interface/accessors.h"
#include "legion/interface/physical_region_impl.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Piece Iterator
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  PieceIterator::PieceIterator(void)
    : impl(NULL), index(-1)
  //--------------------------------------------------------------------------
  {
  }

  //--------------------------------------------------------------------------
  PieceIterator::PieceIterator(const PhysicalRegion &region, FieldID fid,
                               bool privilege_only, bool silence_warnings,
                               const char *warning_string)
    : impl(NULL), index(-1)
  //--------------------------------------------------------------------------
  {
    if (region.impl != NULL)
      impl = region.impl->get_piece_iterator(fid, privilege_only,
                                silence_warnings, warning_string);
    if (impl != NULL)
    {
      impl->add_reference();
      index = impl->get_next(index, current_piece);
    }
  }

  //--------------------------------------------------------------------------
  PieceIterator::PieceIterator(const PieceIterator &rhs)
    : impl(rhs.impl), index(rhs.index), current_piece(rhs.current_piece)
  //--------------------------------------------------------------------------
  {
    if (impl != NULL)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  PieceIterator::PieceIterator(PieceIterator &&rhs) noexcept
    : impl(rhs.impl), index(rhs.index), current_piece(rhs.current_piece)
  //--------------------------------------------------------------------------
  {
    rhs.impl = NULL;
  }

  //--------------------------------------------------------------------------
  PieceIterator::~PieceIterator(void)
  //--------------------------------------------------------------------------
  {
    if ((impl != NULL) && impl->remove_reference())
      delete impl;
  }

  //--------------------------------------------------------------------------
  PieceIterator& PieceIterator::operator=(const PieceIterator &rhs)
  //--------------------------------------------------------------------------
  {
    if ((impl != NULL) && impl->remove_reference())
      delete impl;
    impl = rhs.impl;
    index = rhs.index;
    current_piece = rhs.current_piece;
    if (impl != NULL)
      impl->add_reference();
    return *this;
  }

  //--------------------------------------------------------------------------
  PieceIterator& PieceIterator::operator=(PieceIterator &&rhs) noexcept
  //--------------------------------------------------------------------------
  {
    if ((impl != NULL) && impl->remove_reference())
      delete impl;
    impl = rhs.impl;
    rhs.impl = NULL;
    index = rhs.index;
    current_piece = rhs.current_piece;
    return *this;
  }

  //--------------------------------------------------------------------------
  bool PieceIterator::step(void)
  //--------------------------------------------------------------------------
  {
    if ((impl != NULL) && (index >= 0))
      index = impl->get_next(index, current_piece);
    return valid();
  }

} // namespace Legion
