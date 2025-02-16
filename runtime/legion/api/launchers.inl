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

// Included from launchers.h - do not include this directly

// Useful for IDEs
#include "legion/api/launchers.h"

namespace Legion {

    //--------------------------------------------------------------------------
    inline void StaticDependence::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      dependent_fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline IndexSpaceRequirement& TaskLauncher::add_index_requirement(
                                              const IndexSpaceRequirement &req)
    //--------------------------------------------------------------------------
    {
      index_requirements.push_back(req);
      return index_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& TaskLauncher::add_region_requirement(
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      region_requirements.push_back(req);
      return region_requirements.back();
    } 

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_field(unsigned idx, FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < region_requirements.size());
#endif
      region_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_future(Future f)
    //--------------------------------------------------------------------------
    {
      futures.push_back(f);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_arrival_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_predicate_false_future(Future f)
    //--------------------------------------------------------------------------
    {
      predicate_false_future = f;
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_predicate_false_result(UntypedBuffer arg)
    //--------------------------------------------------------------------------
    {
      predicate_false_result = arg;
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_independent_requirements(bool independent)
    //--------------------------------------------------------------------------
    {
      independent_requirements = independent;
    }

    //--------------------------------------------------------------------------
    inline IndexSpaceRequirement& IndexTaskLauncher::add_index_requirement(
                                              const IndexSpaceRequirement &req)
    //--------------------------------------------------------------------------
    {
      index_requirements.push_back(req);
      return index_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& IndexTaskLauncher::add_region_requirement(
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      region_requirements.push_back(req);
      return region_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < region_requirements.size());
#endif
      region_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_future(Future f)
    //--------------------------------------------------------------------------
    {
      futures.push_back(f);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_predicate_false_future(Future f)
    //--------------------------------------------------------------------------
    {
      predicate_false_future = f;
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_predicate_false_result(UntypedBuffer arg)
    //--------------------------------------------------------------------------
    {
      predicate_false_result = arg;
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_independent_requirements(
                                                               bool independent)
    //--------------------------------------------------------------------------
    {
      independent_requirements = independent;
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_field(FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
      requirement.add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_arrival_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline unsigned CopyLauncher::add_copy_requirements(
                     const RegionRequirement &src, const RegionRequirement &dst)
    //--------------------------------------------------------------------------
    {
      unsigned result = src_requirements.size();
#ifdef DEBUG_LEGION
      assert(result == dst_requirements.size());
#endif
      src_requirements.push_back(src);
      dst_requirements.push_back(dst);
      return result;
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_src_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < src_requirements.size());
#endif
      src_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_dst_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < dst_requirements.size());
#endif
      dst_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_src_indirect_field(FieldID src_idx_field,
                            const RegionRequirement &req, bool range, bool inst)
    //--------------------------------------------------------------------------
    {
      src_indirect_requirements.push_back(req);
      src_indirect_requirements.back().add_field(src_idx_field, inst);
      src_indirect_is_range.push_back(range);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_dst_indirect_field(FieldID dst_idx_field,
                            const RegionRequirement &req, bool range, bool inst)
    //--------------------------------------------------------------------------
    {
      dst_indirect_requirements.push_back(req);
      dst_indirect_requirements.back().add_field(dst_idx_field, inst);
      dst_indirect_is_range.push_back(range);
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& CopyLauncher::add_src_indirect_field(
                                       const RegionRequirement &req, bool range)
    //--------------------------------------------------------------------------
    {
      src_indirect_requirements.push_back(req);
      src_indirect_is_range.push_back(range);
      return src_indirect_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& CopyLauncher::add_dst_indirect_field(
                                       const RegionRequirement &req, bool range)
    //--------------------------------------------------------------------------
    {
      dst_indirect_requirements.push_back(req);
      dst_indirect_is_range.push_back(range);
      return dst_indirect_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_arrival_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline unsigned IndexCopyLauncher::add_copy_requirements(
                     const RegionRequirement &src, const RegionRequirement &dst)
    //--------------------------------------------------------------------------
    {
      unsigned result = src_requirements.size();
#ifdef DEBUG_LEGION
      assert(result == dst_requirements.size());
#endif
      src_requirements.push_back(src);
      dst_requirements.push_back(dst);
      return result;
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_src_field(unsigned idx,
                                                 FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < src_requirements.size());
#endif
      src_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_dst_field(unsigned idx,
                                                 FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < dst_requirements.size());
#endif
      dst_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_src_indirect_field(FieldID src_idx_field,
                              const RegionRequirement &r, bool range, bool inst)
    //--------------------------------------------------------------------------
    {
      src_indirect_requirements.push_back(r);
      src_indirect_requirements.back().add_field(src_idx_field, inst);
      src_indirect_is_range.push_back(range);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_dst_indirect_field(FieldID dst_idx_field,
                              const RegionRequirement &r, bool range, bool inst)
    //--------------------------------------------------------------------------
    {
      dst_indirect_requirements.push_back(r);
      dst_indirect_requirements.back().add_field(dst_idx_field, inst);
      dst_indirect_is_range.push_back(range);
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& IndexCopyLauncher::add_src_indirect_field(
                                       const RegionRequirement &req, bool range)
    //--------------------------------------------------------------------------
    {
      src_indirect_requirements.push_back(req);
      src_indirect_is_range.push_back(range);
      return src_indirect_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& IndexCopyLauncher::add_dst_indirect_field(
                                       const RegionRequirement &req, bool range)
    //--------------------------------------------------------------------------
    {
      dst_indirect_requirements.push_back(req);
      dst_indirect_is_range.push_back(range);
      return dst_indirect_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::set_argument(UntypedBuffer arg)
    //--------------------------------------------------------------------------
    {
      argument = arg;
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::set_future(Future f)
    //--------------------------------------------------------------------------
    {
      future = f;
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_wait_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      wait_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_arrival_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      arrive_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_arrival_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::set_argument(UntypedBuffer arg)
    //--------------------------------------------------------------------------
    {
      argument = arg;
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::set_future(Future f)
    //--------------------------------------------------------------------------
    {
      future = f;
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_wait_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      wait_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_arrival_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      arrive_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    LEGION_DISABLE_DEPRECATED_WARNINGS

    //--------------------------------------------------------------------------
    inline void AttachLauncher::initialize_constraints(bool column_major, 
                          bool soa, const std::vector<FieldID> &fields,
                          const std::map<FieldID,size_t> *alignments /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
      constraints.add_constraint(
          FieldConstraint(fields, true/*contiugous*/, true/*inorder*/));
      const int dims = handle.get_index_space().get_dim();
      std::vector<DimensionKind> dim_order(dims+1);
      // Field dimension first for AOS
      dim_order[soa ? dims : 0] = LEGION_DIM_F;
      if (column_major)
      {
        for (int idx = 0; idx < dims; idx++)
          dim_order[idx+(soa ? 0 : 1)] = (DimensionKind)(LEGION_DIM_X + idx); 
      }
      else
      {
        for (int idx = 0; idx < dims; idx++)
          dim_order[idx+(soa ? 0 : 1)] = 
            (DimensionKind)(LEGION_DIM_X + (dims-1) - idx);
      }
      constraints.add_constraint(
          OrderingConstraint(dim_order, false/*contiguous*/));
      if (alignments != nullptr)
        for (std::map<FieldID,size_t>::const_iterator it = alignments->begin();
             it != alignments->end(); it++)
          constraints.add_constraint(
              AlignmentConstraint(it->first, LEGION_GE_EK, it->second));
    }

    //--------------------------------------------------------------------------
    inline void DiscardLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_file(const char *name,
                                            const std::vector<FieldID> &fields,
                                            LegionFileMode m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource == LEGION_EXTERNAL_POSIX_FILE);
#endif
      file_name = name;
      mode = m;
      file_fields = fields;
      initialize_constraints(true/*column major*/, true/*soa*/, fields);
      privilege_fields.insert(fields.begin(), fields.end());
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_hdf5(const char *name,
                                const std::map<FieldID,const char*> &field_map,
                                LegionFileMode m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource == LEGION_EXTERNAL_HDF5_FILE);
#endif
      file_name = name;
      mode = m;
      field_files = field_map;
      std::vector<FieldID> fields;
      fields.reserve(field_map.size());
      for (std::map<FieldID,const char*>::const_iterator it =
            field_map.begin(); it != field_map.end(); it++)
        fields.push_back(it->first);
      initialize_constraints(true/*column major*/, true/*soa*/, fields);
      privilege_fields.insert(fields.begin(), fields.end());
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_array_aos(void *base, bool column_major,
                          const std::vector<FieldID> &fields, Memory mem,
                          const std::map<FieldID,size_t> *alignments /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
      assert(resource == LEGION_EXTERNAL_INSTANCE);
#endif
      constraints.add_constraint(PointerConstraint(mem, uintptr_t(base)));
      if (mem.exists())
        constraints.add_constraint(MemoryConstraint(mem.kind()));
      constraints.add_constraint(
          FieldConstraint(fields, true/*contiugous*/, true/*inorder*/));
      initialize_constraints(column_major, false/*soa*/, fields, alignments);
      privilege_fields.insert(fields.begin(), fields.end());
    }
    
    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_array_soa(void *base, bool column_major,
                          const std::vector<FieldID> &fields, Memory mem,
                          const std::map<FieldID,size_t> *alignments /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
      assert(resource == LEGION_EXTERNAL_INSTANCE);
#endif
      constraints.add_constraint(PointerConstraint(mem, uintptr_t(base)));
      if (mem.exists())
        constraints.add_constraint(MemoryConstraint(mem.kind()));
      constraints.add_constraint(
          FieldConstraint(fields, true/*contiguous*/, true/*inorder*/));
      initialize_constraints(column_major, true/*soa*/, fields, alignments);
      privilege_fields.insert(fields.begin(), fields.end());
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::initialize_constraints(bool column_major, 
                          bool soa, const std::vector<FieldID> &fields,
                          const std::map<FieldID,size_t> *alignments /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
      constraints.add_constraint(
          FieldConstraint(fields, true/*contiugous*/, true/*inorder*/));
      const int dims = parent.get_index_space().get_dim();
      std::vector<DimensionKind> dim_order(dims+1);
      // Field dimension first for AOS
      dim_order[soa ? dims : 0] = LEGION_DIM_F;
      if (column_major)
      {
        for (int idx = 0; idx < dims; idx++)
          dim_order[idx+(soa ? 0 : 1)] = (DimensionKind)(LEGION_DIM_X + idx); 
      }
      else
      {
        for (int idx = 0; idx < dims; idx++)
          dim_order[idx+(soa ? 0 : 1)] = 
            (DimensionKind)(LEGION_DIM_X + (dims-1) - idx);
      }
      constraints.add_constraint(
          OrderingConstraint(dim_order, false/*contiguous*/));
      if (alignments != nullptr)
        for (std::map<FieldID,size_t>::const_iterator it = alignments->begin();
             it != alignments->end(); it++)
          constraints.add_constraint(
              AlignmentConstraint(it->first, LEGION_GE_EK, it->second));
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::add_external_resource(LogicalRegion handle,
                                const Realm::ExternalInstanceResource *resource)
    //--------------------------------------------------------------------------
    {
      handles.push_back(handle);
      external_resources.push_back(resource);
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::attach_file(LogicalRegion handle,
                                             const char *file_name,
                                             const std::vector<FieldID> &fields,
                                             LegionFileMode m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource == LEGION_EXTERNAL_POSIX_FILE);
#endif
      if (handles.empty())
      {
        mode = m;
        initialize_constraints(true/*column major*/, true/*soa*/, fields);
        privilege_fields.insert(fields.begin(), fields.end());
      }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
        assert(mode == m);
#endif
#endif
      handles.push_back(handle);
      file_names.push_back(file_name);
      if (!file_fields.empty())
      {
#ifdef DEBUG_LEGION
        assert(fields.size() == file_fields.size());
#ifndef NDEBUG
        for (unsigned idx = 0; idx < fields.size(); idx++)
          assert(file_fields[idx] == fields[idx]);
#endif
#endif
      }
      else
        file_fields = fields;
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::attach_hdf5(LogicalRegion handle,
                                 const char *file_name,
                                 const std::map<FieldID,const char*> &field_map,
                                 LegionFileMode m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource == LEGION_EXTERNAL_HDF5_FILE);
#endif
      if (handles.empty())
      {
        mode = m;
        std::vector<FieldID> fields;
        fields.reserve(field_map.size());
        for (std::map<FieldID,const char*>::const_iterator it =
              field_map.begin(); it != field_map.end(); it++)
          fields.push_back(it->first);
        initialize_constraints(true/*column major*/, true/*soa*/, fields);
        privilege_fields.insert(fields.begin(), fields.end());
      }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
        assert(mode == m);
#endif
#endif
      handles.push_back(handle);
      file_names.push_back(file_name);
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool first = field_files.empty();
#endif
#endif
      for (std::map<FieldID,const char*>::const_iterator it =
            field_map.begin(); it != field_map.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(first || (field_files.find(it->first) != field_files.end()));
#endif
        field_files[it->first].push_back(it->second);
      }
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::attach_array_aos(LogicalRegion handle,
              void *base, bool column_major, const std::vector<FieldID> &fields,
              Memory mem, const std::map<FieldID,size_t> *alignments)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
      assert(resource == LEGION_EXTERNAL_INSTANCE);
#endif
      if (handles.empty())
      {
        initialize_constraints(column_major, false/*soa*/, fields, alignments);
        privilege_fields.insert(fields.begin(), fields.end());
      }
#ifdef DEBUG_LEGION
      else
      {
        // Check that the fields are the same
        assert(fields.size() == privilege_fields.size());
        for (std::vector<FieldID>::const_iterator it =
              fields.begin(); it != fields.end(); it++)
          assert(privilege_fields.find(*it) != privilege_fields.end());
        // Check that the layouts are the same
        const OrderingConstraint &order = constraints.ordering_constraint;
        assert(order.ordering.front() == LEGION_DIM_F);
        const int dims = handle.get_index_space().get_dim();
        assert(dims == handles.back().get_index_space().get_dim());
        if (column_major)
        {
          for (int idx = 0; idx < dims; idx++)
            assert(order.ordering[idx+1] == ((DimensionKind)LEGION_DIM_X+idx));
        }
        else
        {
          for (int idx = 0; idx < dims; idx++)
            assert(order.ordering[idx+1] == 
                ((DimensionKind)(LEGION_DIM_X + (dims-1) - idx)));
        }
        // Check that the alignments are the same
        if (alignments != nullptr)
        {
          assert(alignments->size() == 
                  constraints.alignment_constraints.size());
          unsigned index = 0;
          for (std::map<FieldID,size_t>::const_iterator it =
                alignments->begin(); it != alignments->end(); it++, index++)
          {
            const AlignmentConstraint &alignment = 
              constraints.alignment_constraints[index];
            assert(alignment.fid == it->first);
            assert(alignment.alignment == it->second);
          }
        }
      }
#endif
      handles.push_back(handle);
      pointers.emplace_back(PointerConstraint(mem, uintptr_t(base)));
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::attach_array_soa(LogicalRegion handle,
              void *base, bool column_major, const std::vector<FieldID> &fields,
              Memory mem, const std::map<FieldID,size_t> *alignments)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
      assert(resource == LEGION_EXTERNAL_INSTANCE);
#endif
      if (handles.empty())
      {
        initialize_constraints(column_major, true/*soa*/, fields, alignments); 
        privilege_fields.insert(fields.begin(), fields.end());
      }
#ifdef DEBUG_LEGION
      else
      {
        // Check that the fields are the same
        assert(fields.size() == privilege_fields.size());
        for (std::vector<FieldID>::const_iterator it =
              fields.begin(); it != fields.end(); it++)
          assert(privilege_fields.find(*it) != privilege_fields.end());
        // Check that the layouts are the same
        const OrderingConstraint &order = constraints.ordering_constraint;
        const int dims = handle.get_index_space().get_dim();
        assert(dims == handles.back().get_index_space().get_dim());
        if (column_major)
        {
          for (int idx = 0; idx < dims; idx++)
            assert(order.ordering[idx] == ((DimensionKind)LEGION_DIM_X+idx));
        }
        else
        {
          for (int idx = 0; idx < dims; idx++)
            assert(order.ordering[idx] == 
                ((DimensionKind)(LEGION_DIM_X + (dims-1) - idx)));
        }
        assert(order.ordering.back() == LEGION_DIM_F);
        // Check that the alignments are the same
        if (alignments != nullptr)
        {
          assert(alignments->size() == 
                  constraints.alignment_constraints.size());
          unsigned index = 0;
          for (std::map<FieldID,size_t>::const_iterator it =
                alignments->begin(); it != alignments->end(); it++, index++)
          {
            const AlignmentConstraint &alignment = 
              constraints.alignment_constraints[index];
            assert(alignment.fid == it->first);
            assert(alignment.alignment == it->second);
          }
        }
      }
#endif
      handles.push_back(handle);
      pointers.emplace_back(PointerConstraint(mem, uintptr_t(base)));
    }

    LEGION_REENABLE_DEPRECATED_WARNINGS

    //--------------------------------------------------------------------------
    inline void PredicateLauncher::add_predicate(const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      predicates.push_back(pred);
    }

    //--------------------------------------------------------------------------
    inline void TimingLauncher::add_precondition(const Future &f)
    //--------------------------------------------------------------------------
    {
      preconditions.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void MustEpochLauncher::add_single_task(const DomainPoint &point,
                                                   const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      single_tasks.push_back(launcher);
      single_tasks.back().point = point;
    }

    //--------------------------------------------------------------------------
    inline void MustEpochLauncher::add_index_task(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      index_tasks.push_back(launcher);
    }

} // namespace Legion
