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

#include "legion/tracing/logical.h"
#include "legion/contexts/inner.h"
#include "legion/nodes/region.h"
#include "legion/operations/close.h"
#include "legion/operations/fence.h"
#include "legion/operations/pointwise.h"
#include "legion/tasks/index.h"
#include "legion/tracing/physical.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // LogicalTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalTrace::LogicalTrace(
        InnerContext* c, TraceID t, bool logical_only, bool static_trace,
        Provenance* p, const std::set<RegionTreeID>* trees)
      : context(c), tid(t), begin_provenance(p), end_provenance(nullptr),
        physical_trace(logical_only ? nullptr : new PhysicalTrace(this)),
        verification_index(0), blocking_call_observed(false), fixed(false),
        intermediate_fence(false), recording(true), trace_fence(nullptr),
        static_translator(static_trace ? new StaticTranslator(trees) : nullptr)
    //--------------------------------------------------------------------------
    {
      if (begin_provenance != nullptr)
        begin_provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    LogicalTrace::~LogicalTrace(void)
    //--------------------------------------------------------------------------
    {
      if (physical_trace != nullptr)
        delete physical_trace;
      if ((begin_provenance != nullptr) && begin_provenance->remove_reference())
        delete begin_provenance;
      if ((end_provenance != nullptr) && end_provenance->remove_reference())
        delete end_provenance;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::fix_trace(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fixed);
      assert(end_provenance == nullptr);
#endif
      fixed = true;
      end_provenance = provenance;
      if (end_provenance != nullptr)
        end_provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::initialize_op_tracing(
        Operation* op, const std::vector<StaticDependence>* dependences)
    //--------------------------------------------------------------------------
    {
      if (op->is_internal_op())
      {
        if (!recording)
          return false;
      }
      else
      {
        if (has_physical_trace() && (op->get_memoizable() == nullptr))
          REPORT_LEGION_ERROR(
              ERROR_PHYSICAL_TRACING_UNSUPPORTED_OP,
              "Illegal operation in physical trace. The application launched "
              "a %s operation inside of physical trace %d of parent task %s "
              "(UID %lld) but this kind of operation is not supported for "
              "physical traces at the moment. You can request support but "
              "we can guarantee support for all kinds of operations in "
              "physical traces.",
              op->get_logging_name(), tid, context->get_task_name(),
              context->get_unique_id())
        // Check to see if we are doing safe tracing checks or not
        if (runtime->safe_tracing)
        {
          // Compute the hash for this operation
          Murmur3Hasher hasher;
          const OpKind kind = op->get_operation_kind();
          hasher.hash(kind);
          TaskID task_id = 0;
          if (kind == TASK_OP_KIND)
          {
#ifdef DEBUG_LEGION
            TaskOp* task = dynamic_cast<TaskOp*>(op);
            assert(task != nullptr);
#else
            TaskOp* task = dynamic_cast<TaskOp*>(op);
#endif
            task_id = task->task_id;
            hasher.hash(task_id);
          }
          const unsigned num_regions = op->get_region_count();
          for (unsigned idx = 0; idx < num_regions; idx++)
          {
            const RegionRequirement& req = op->get_requirement(idx);
            hasher.hash(req.parent);
            hasher.hash(req.handle_type);
            if (req.handle_type == LEGION_PARTITION_PROJECTION)
              hasher.hash(req.partition);
            else
              hasher.hash(req.region);
            for (std::set<FieldID>::const_iterator it =
                     req.privilege_fields.begin();
                 it != req.privilege_fields.end(); it++)
              hasher.hash(*it);
            for (std::vector<FieldID>::const_iterator it =
                     req.instance_fields.begin();
                 it != req.instance_fields.end(); it++)
              hasher.hash(*it);
            hasher.hash(req.privilege);
            hasher.hash(req.prop);
            hasher.hash(req.redop);
            hasher.hash(req.tag);
            hasher.hash(req.flags);
            if (req.handle_type != LEGION_SINGULAR_PROJECTION)
              hasher.hash(req.projection);
            size_t projection_size = 0;
            const void* projection_args =
                req.get_projection_args(&projection_size);
            if (projection_size > 0)
              hasher.hash(projection_args, projection_size);
          }
          uint64_t hash[2];
          hasher.finalize(hash);
          if (fixed)
          {
            if (verification_infos.size() <= verification_index)
              REPORT_LEGION_ERROR(
                  ERROR_TRACE_VIOLATION_OPERATION,
                  "Detected %d operations in trace %d of parent task %s "
                  "(UID %lld) which differs from the %zd operations that "
                  "where recorded in the first execution of the trace. "
                  "The number of operations in the trace must always "
                  "be the same across all executions of the trace.",
                  verification_index, tid, context->get_task_name(),
                  context->get_unique_id(), verification_infos.size())
            const VerificationInfo& info =
                verification_infos[verification_index++];
            if (info.kind != kind)
              REPORT_LEGION_ERROR(
                  ERROR_TRACE_VIOLATION_OPERATION,
                  "Operation %s does match the recorded operation kind %s "
                  "for the %d operation in trace %d of parent task %s "
                  "(UID %lld). The same order of operations must be "
                  "issued every time a trace is executed.",
                  Operation::get_string_rep(kind), op->get_logging_name(),
                  verification_index - 1, tid, context->get_task_name(),
                  context->get_unique_id())
            if (info.task_id != task_id)
              REPORT_LEGION_ERROR(
                  ERROR_TRACE_VIOLATION_OPERATION,
                  "Task %d does match the recorded task %d for the %d task "
                  "in trace %d of parent task %s (UID %lld). The same order "
                  "of operations must be issued every time a trace is "
                  "executed.",
                  task_id, info.task_id, verification_index - 1, tid,
                  context->get_task_name(), context->get_unique_id())
            if (info.regions != num_regions)
            {
              if (kind == TASK_OP_KIND)
                REPORT_LEGION_ERROR(
                    ERROR_TRACE_VIOLATION_OPERATION,
                    "Task %s recorded %d region requirements for trace "
                    "%d in parent task %s (UID %lld) but was re-executed with "
                    "%d region requirements. The number of region requirements"
                    " recorded must always match the number re-executed for "
                    "each corresponding operation in the trace.",
                    op->get_logging_name(), info.regions, tid,
                    context->get_task_name(), context->get_unique_id(),
                    num_regions)
              else
                REPORT_LEGION_ERROR(
                    ERROR_TRACE_VIOLATION_OPERATION,
                    "Operation %s recorded %d region requirements for trace "
                    "%d in parent task %s (UID %lld) but was re-executed with "
                    "%d region requirements. The number of region requirements"
                    " must always match the number re-executed for each "
                    "corresponding operation in the trace.",
                    op->get_logging_name(), info.regions, tid,
                    context->get_task_name(), context->get_unique_id(),
                    num_regions)
            }
            if ((info.hash[0] != hash[0]) || (info.hash[1] != hash[1]))
            {
              if (kind == TASK_OP_KIND)
                REPORT_LEGION_ERROR(
                    ERROR_TRACE_VIOLATION_OPERATION,
                    "Task %s was replayed with different region requirements "
                    "for trace %d in parent task %s (UID %lld) than what it "
                    "had when it was recorded. Region requirement arguments "
                    "must match exactly every time a trace is executed.",
                    op->get_logging_name(), tid, context->get_task_name(),
                    context->get_unique_id())
              else
                REPORT_LEGION_ERROR(
                    ERROR_TRACE_VIOLATION_OPERATION,
                    "Operation %s was replayed with different region "
                    "requirements for trace %d in parent task %s (UID %lld) "
                    "than waht it had when it was recorded. Region "
                    "requirement arguments must match exactly every time a "
                    "trace is executed.",
                    op->get_logging_name(), tid, context->get_task_name(),
                    context->get_unique_id())
            }
          }
          else
            verification_infos.emplace_back(
                VerificationInfo(kind, task_id, num_regions, hash));
        }
        if (fixed)
          return false;
      }
      if (static_translator != nullptr)
        static_translator->push_dependences(dependences);
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::check_operation_count(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(verification_index <= verification_infos.size());
#endif
      if (verification_index < verification_infos.size())
        REPORT_LEGION_ERROR(
            ERROR_TRACE_VIOLATION_OPERATION,
            "Detected %d operations in trace %d of parent task %s "
            "(UID %lld) which differs from the %zd operations that "
            "where recorded in the first execution of the trace. "
            "The number of operations in the trace must always "
            "be the same across all executions of the trace.",
            verification_index, tid, context->get_task_name(),
            context->get_unique_id(), verification_infos.size())
      verification_index = 0;
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::skip_analysis(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
#endif
      if (static_translator == nullptr)
        return false;
      return static_translator->skip_analysis(tid);
    }

    //--------------------------------------------------------------------------
    size_t LogicalTrace::register_operation(Operation* op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      const std::pair<Operation*, GenerationID> key(op, gen);
#ifdef LEGION_SPY
      current_uids[key] = op->get_unique_op_id();
      num_regions[key] = op->get_region_count();
#endif
      if (recording)
      {
        // Recording
        if (op->is_internal_op())
          // We don't need to register internal operations
          return std::numeric_limits<size_t>::max();
        const size_t index = replay_info.size();
        const size_t op_index = operations.size();
        op_map[key] = std::make_pair(op_index, index);
        operations.emplace_back(OpInfo(op));
        replay_info.emplace_back(OperationInfo());
        if (static_translator != nullptr)
        {
          // Add a mapping reference since we might need to refer to it later
          op->add_mapping_reference(gen);
          // Recording a static trace so see if we have
          // dependences to translate
          std::vector<StaticDependence> to_translate;
          static_translator->pop_dependences(to_translate);
          translate_dependence_records(op, op_index, to_translate);
        }
        return index;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!op->is_internal_op());
#endif
        // Replaying
        const size_t index = replay_index++;
        if (index >= replay_info.size())
          REPORT_LEGION_ERROR(
              ERROR_TRACE_VIOLATION_RECORDED,
              "Trace violation! Recorded %zd operations in trace "
              "%d in task %s (UID %lld) but %zd operations have "
              "now been issued!",
              replay_info.size(), tid, context->get_task_name(),
              context->get_unique_id(), index + 1)

        // Check to see if the meta-data alignes
        OperationInfo& info = replay_info[index];
        // Add a mapping reference since ops will be registering dependences
        op->add_mapping_reference(gen);
        operations.emplace_back(OpInfo(op));
        frontiers.insert(key);
        // First make any close operations needed for this operation and
        // register their dependences
        for (LegionVector<CloseInfo>::const_iterator cit = info.closes.begin();
             cit != info.closes.end(); cit++)
        {
#ifdef DEBUG_LEGION_COLLECTIVES
          MergeCloseOp* close_op = context->get_merge_close_op(op, cit->node);
#else
          MergeCloseOp* close_op = context->get_merge_close_op();
#endif
          close_op->initialize(context, cit->requirement, cit->creator_idx, op);
          close_op->update_close_mask(cit->close_mask);
          const GenerationID close_gen = close_op->get_generation();
          const std::pair<Operation*, GenerationID> close_key(
              close_op, close_gen);
          close_op->add_mapping_reference(close_gen);
          operations.emplace_back(OpInfo(close_op));
#ifdef LEGION_SPY
          current_uids[close_key] = close_op->get_unique_op_id();
          num_regions[close_key] = close_op->get_region_count();
#endif
          close_op->begin_dependence_analysis();
          close_op->trigger_dependence_analysis();
          replay_operation_dependences(close_op, cit->dependences);
          close_op->end_dependence_analysis();
        }
        if (!info.pointwise_dependences.empty())
          replay_pointwise_dependences(op, info.pointwise_dependences);
        // Then register the dependences for this operation
        if (!info.dependences.empty())
          replay_operation_dependences(op, info.dependences);
        else  // need to at least record a dependence on the fence event
          op->register_dependence(trace_fence, trace_fence_gen);
        return index;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::register_internal(InternalOp* op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
#endif
      const std::pair<Operation*, GenerationID> key(op, op->get_generation());
      // Note that we don't record the index of the operation here since
      // it has no index, instead we record the index of the replay_info
      // that is associated with this internal operation
      op_map[key] = std::make_pair(-1, replay_info.size() - 1);
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::register_close(
        MergeCloseOp* op, unsigned creator_idx,
#ifdef DEBUG_LEGION_COLLECTIVES
        RegionTreeNode* node,
#endif
        const RegionRequirement& req)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(!replay_info.empty());
#endif
      std::pair<Operation*, GenerationID> key(op, op->get_generation());
      const size_t index = operations.size();
      operations.emplace_back(OpInfo(op));
      op_map[key] = std::make_pair(index, replay_info.size());
      OperationInfo& info = replay_info.back();
      info.closes.emplace_back(CloseInfo(
          op, creator_idx,
#ifdef DEBUG_LEGION_COLLECTIVES
          node,
#endif
          req));
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::replay_operation_dependences(
        Operation* op, const LegionVector<DependenceRecord>& dependences)
    //--------------------------------------------------------------------------
    {
      for (LegionVector<DependenceRecord>::const_iterator it =
               dependences.begin();
           it != dependences.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->operation_idx >= 0);
        assert(((size_t)it->operation_idx) < operations.size());
        assert(it->dtype != LEGION_NO_DEPENDENCE);
#endif
        std::pair<Operation*, GenerationID> target = std::make_pair(
            operations[it->operation_idx].op,
            operations[it->operation_idx].gen);
        std::set<std::pair<Operation*, GenerationID>>::iterator finder =
            frontiers.find(target);
        if (finder != frontiers.end())
        {
          finder->first->remove_mapping_reference(finder->second);
          frontiers.erase(finder);
        }
        if ((it->prev_idx == -1) || (it->next_idx == -1))
        {
          op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(),
              get_current_uid_by_index(it->operation_idx),
              (it->prev_idx == -1) ? 0 : it->prev_idx, op->get_unique_op_id(),
              (it->next_idx == -1) ? 0 : it->next_idx, LEGION_TRUE_DEPENDENCE);
#endif
        }
        else
        {
          op->register_region_dependence(
              it->next_idx, target.first, target.second, it->prev_idx,
              it->dtype, it->dependent_mask);
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(),
              get_current_uid_by_index(it->operation_idx), it->prev_idx,
              op->get_unique_op_id(), it->next_idx, it->dtype);
#endif
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::replay_pointwise_dependences(
        Operation* op,
        const std::map<unsigned, std::vector<PointwiseDependence>>& dependences)
    //--------------------------------------------------------------------------
    {
      // Make a copy of the dependences and then update the context index
      // with the right entry from the list of operations
      std::map<unsigned, std::vector<PointwiseDependence>> copy = dependences;
      for (std::map<unsigned, std::vector<PointwiseDependence>>::iterator cit =
               copy.begin();
           cit != copy.end(); cit++)
      {
        for (std::vector<PointwiseDependence>::iterator it =
                 cit->second.begin();
             it != cit->second.end(); it++)
        {
          const OpInfo& info = operations[it->context_index];
          it->context_index = info.context_index;
          it->unique_id = info.unique_id;
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(), info.unique_id,
              it->region_index, op->get_unique_op_id(), cit->first,
              LEGION_TRUE_DEPENDENCE, true /*pointwise*/);
#endif
        }
      }
      op->replay_pointwise_dependences(copy);
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_dependence(
        Operation* target, GenerationID target_gen, Operation* source,
        GenerationID source_gen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(!target->is_internal_op());
      assert(!source->is_internal_op());
#endif
      const std::pair<Operation*, GenerationID> target_key(target, target_gen);
      std::map<
          std::pair<Operation*, GenerationID>,
          std::pair<unsigned, unsigned>>::const_iterator target_finder =
          op_map.find(target_key);
      // The target is not part of the trace so there's no need to record it
      if (target_finder == op_map.end())
        return false;
#ifdef DEBUG_LEGION
      assert(!replay_info.empty());
#endif
      OperationInfo& info = replay_info.back();
      DependenceRecord record(target_finder->second.first);
      for (LegionVector<DependenceRecord>::iterator it =
               info.dependences.begin();
           it != info.dependences.end(); it++)
        if (it->merge(record))
          return true;
      info.dependences.emplace_back(std::move(record));
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::record_region_dependence(
        Operation* target, GenerationID target_gen, Operation* source,
        GenerationID source_gen, unsigned target_idx, unsigned source_idx,
        DependenceType dtype, const FieldMask& dep_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
#endif
      const std::pair<Operation*, GenerationID> target_key(target, target_gen);
      std::map<
          std::pair<Operation*, GenerationID>,
          std::pair<unsigned, unsigned>>::const_iterator target_finder =
          op_map.find(target_key);
      // The target is not part of the trace so there's no need to record it
      if (target_finder == op_map.end())
      {
        // If this is a close operation then we still need to update the mask
        if (source->get_operation_kind() == MERGE_CLOSE_OP_KIND)
        {
#ifdef DEBUG_LEGION
          assert(!replay_info.empty());
          assert(
              op_map.find(std::make_pair(source, source_gen)) != op_map.end());
#endif
          OperationInfo& info = replay_info.back();
#ifdef DEBUG_LEGION
          bool found = false;
          assert(!info.closes.empty());
#endif
          // Find the right close info and record the dependence
          for (unsigned idx = 0; idx < info.closes.size(); idx++)
          {
            CloseInfo& close = info.closes[idx];
            if (close.close_op != source)
              continue;
#ifdef DEBUG_LEGION
            found = true;
#endif
            close.close_mask |= dep_mask;
            break;
          }
#ifdef DEBUG_LEGION
          assert(found);
#endif
        }
        return false;
      }
#ifdef DEBUG_LEGION
      assert(!replay_info.empty());
#endif
      OperationInfo& info = replay_info.back();
      if (source->get_operation_kind() == MERGE_CLOSE_OP_KIND)
      {
#ifdef DEBUG_LEGION
        bool found = false;
        assert(!info.closes.empty());
#endif
        // Find the right close info and record the dependence
        for (unsigned idx = 0; idx < info.closes.size(); idx++)
        {
          CloseInfo& close = info.closes[idx];
          if (close.close_op != source)
            continue;
#ifdef DEBUG_LEGION
          found = true;
#endif
          close.close_mask |= dep_mask;
          if (target->is_internal_op() &&
              (target->get_operation_kind() != MERGE_CLOSE_OP_KIND))
          {
#ifdef DEBUG_LEGION
            assert(target_finder->second.second < replay_info.size());
#endif
            const OperationInfo& target_info =
                replay_info[target_finder->second.second];
            std::map<unsigned, LegionVector<DependenceRecord>>::const_iterator
                finder = target_info.internal_dependences.find(target_idx);
            if (finder != target_info.internal_dependences.end())
            {
              for (LegionVector<DependenceRecord>::const_iterator rit =
                       finder->second.begin();
                   rit != finder->second.end(); rit++)
              {
                const FieldMask overlap = dep_mask & rit->dependent_mask;
                if (!overlap)
                  continue;
                DependenceRecord record(
                    rit->operation_idx, rit->prev_idx, source_idx, dtype,
                    overlap);
                bool found2 = false;
                for (LegionVector<DependenceRecord>::iterator it =
                         close.dependences.begin();
                     it != close.dependences.end(); it++)
                {
                  if (!it->merge(record))
                    continue;
                  found2 = true;
                  break;
                }
                if (!found2)
                  close.dependences.emplace_back(std::move(record));
              }
            }
          }
          else
          {
            DependenceRecord record(
                target_finder->second.first, target_idx, source_idx, dtype,
                dep_mask);
            for (LegionVector<DependenceRecord>::iterator it =
                     close.dependences.begin();
                 it != close.dependences.end(); it++)
              if (it->merge(record))
                return true;
            close.dependences.emplace_back(std::move(record));
          }
          break;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
      else if (source->is_internal_op())
      {
        // Record the dependence on the correct region requirement
        // of the internal operations
        LegionVector<DependenceRecord>& internal_dependences =
            info.internal_dependences[source_idx];
        if (target->is_internal_op() &&
            (target->get_operation_kind() != MERGE_CLOSE_OP_KIND))
        {
#ifdef DEBUG_LEGION
          assert(target_finder->second.second < replay_info.size());
#endif
          const OperationInfo& target_info =
              replay_info[target_finder->second.second];
          std::map<unsigned, LegionVector<DependenceRecord>>::const_iterator
              finder = target_info.internal_dependences.find(target_idx);
          if (finder != target_info.internal_dependences.end())
          {
            for (LegionVector<DependenceRecord>::const_iterator rit =
                     finder->second.begin();
                 rit != finder->second.end(); rit++)
            {
              const FieldMask overlap = dep_mask & rit->dependent_mask;
              if (!overlap)
                continue;
              DependenceRecord record(
                  rit->operation_idx, rit->prev_idx, source_idx, dtype,
                  overlap);
              bool found = false;
              for (LegionVector<DependenceRecord>::iterator it =
                       internal_dependences.begin();
                   it != internal_dependences.end(); it++)
              {
                if (!it->merge(record))
                  continue;
                found = true;
                break;
              }
              if (!found)
                internal_dependences.emplace_back(std::move(record));
            }
          }
        }
        else
        {
          DependenceRecord record(
              target_finder->second.first, target_idx, source_idx, dtype,
              dep_mask);
          for (LegionVector<DependenceRecord>::iterator it =
                   internal_dependences.begin();
               it != internal_dependences.end(); it++)
            if (it->merge(record))
              return true;
          internal_dependences.emplace_back(std::move(record));
        }
      }
      else if (
          target->is_internal_op() &&
          (target->get_operation_kind() != MERGE_CLOSE_OP_KIND))
      {
#ifdef DEBUG_LEGION
        assert(target_finder->second.second < replay_info.size());
#endif
        // Figure out which region requirement it was for and look for
        // overlapping dependences on that region requirement and record
        // any of those instead
        const OperationInfo& target_info =
            replay_info[target_finder->second.second];
        std::map<unsigned, LegionVector<DependenceRecord>>::const_iterator
            finder = target_info.internal_dependences.find(target_idx);
        if (finder != target_info.internal_dependences.end())
        {
          for (LegionVector<DependenceRecord>::const_iterator rit =
                   finder->second.begin();
               rit != finder->second.end(); rit++)
          {
            const FieldMask overlap = dep_mask & rit->dependent_mask;
            if (!overlap)
              continue;
            DependenceRecord record(
                rit->operation_idx, rit->prev_idx, source_idx, dtype, overlap);
            bool found = false;
            for (LegionVector<DependenceRecord>::iterator it =
                     info.dependences.begin();
                 it != info.dependences.end(); it++)
            {
              if (!it->merge(record))
                continue;
              found = true;
              break;
            }
            if (!found)
              info.dependences.emplace_back(std::move(record));
          }
        }
      }
      else
      {
        DependenceRecord record(
            target_finder->second.first, target_idx, source_idx, dtype,
            dep_mask);
        for (LegionVector<DependenceRecord>::iterator it =
                 info.dependences.begin();
             it != info.dependences.end(); it++)
          if (it->merge(record))
            return true;
        info.dependences.emplace_back(std::move(record));
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::record_pointwise_dependence(
        Operation* target, GenerationID target_gen, Operation* source,
        GenerationID source_gen, unsigned idx,
        const PointwiseDependence& dependence)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!target->is_internal_op());
      assert(!source->is_internal_op());
#endif
      const std::pair<Operation*, GenerationID> target_key(target, target_gen);
      std::map<
          std::pair<Operation*, GenerationID>,
          std::pair<unsigned, unsigned>>::const_iterator target_finder =
          op_map.find(target_key);
      // If the target is not part of the trace then there's nothing to do
      if (target_finder == op_map.end())
        return;
#ifdef DEBUG_LEGION
      assert(!replay_info.empty());
#endif
      OperationInfo& info = replay_info.back();
      // Append the pointwise record to the
      PointwiseDependence& last =
          info.pointwise_dependences[idx].emplace_back(dependence);
      // Update the context index with the relative context index
      last.context_index = target_finder->second.first;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::begin_logical_trace(FenceOp* fence_op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace_fence == nullptr);
#endif
      if (!recording)
      {
        trace_fence = fence_op;
        trace_fence_gen = fence_op->get_generation();
        fence_op->add_mapping_reference(trace_fence_gen);
        replay_index = 0;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::end_logical_trace(FenceOp* op)
    //--------------------------------------------------------------------------
    {
      if (!recording)
      {
#ifdef DEBUG_LEGION
        assert(trace_fence != nullptr);
#endif
        if (replay_index != replay_info.size())
          REPORT_LEGION_ERROR(
              ERROR_TRACE_VIOLATION_RECORDED,
              "Trace violation! Recorded %zd operations in trace "
              "%d in task %s (UID %lld) but only %zd operations "
              "have been issued at the end of the trace!",
              replay_info.size(), tid, context->get_task_name(),
              context->get_unique_id(), replay_index)
        op->register_dependence(trace_fence, trace_fence_gen);
        trace_fence->remove_mapping_reference(trace_fence_gen);
        trace_fence = nullptr;
        // Register for this fence on every one of the operations in
        // the trace and then clear out the operations data structure
        for (std::set<std::pair<Operation*, GenerationID>>::iterator it =
                 frontiers.begin();
             it != frontiers.end(); ++it)
        {
          const std::pair<Operation*, GenerationID>& target = *it;
#ifdef DEBUG_LEGION
          assert(!target.first->is_internal_op());
#endif
          op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
          for (unsigned req_idx = 0; req_idx < num_regions[target]; req_idx++)
          {
            LegionSpy::log_mapping_dependence(
                op->get_context()->get_unique_id(), current_uids[target],
                req_idx, op->get_unique_op_id(), 0, LEGION_TRUE_DEPENDENCE);
          }
#endif
          // Remove any mapping references that we hold
          target.first->remove_mapping_reference(target.second);
        }
      }
      else  // Finished the recording so we are done
      {
        recording = false;
        op_map.clear();
        for (unsigned idx = 0; idx < replay_info.size(); idx++)
          replay_info[idx].internal_dependences.clear();
        if (static_translator != nullptr)
        {
#ifdef DEBUG_LEGION
          assert(static_translator->dependences.empty());
#endif
          delete static_translator;
          static_translator = nullptr;
          // Also remove the mapping references from all the operations
          for (std::vector<OpInfo>::const_iterator it = operations.begin();
               it != operations.end(); it++)
            it->op->remove_mapping_reference(it->gen);
          // Remove mapping fences on the frontiers which haven't been removed
          for (std::set<std::pair<Operation*, GenerationID>>::const_iterator
                   it = frontiers.begin();
               it != frontiers.end(); it++)
            it->first->remove_mapping_reference(it->second);
        }
      }
      operations.clear();
      frontiers.clear();
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    }

    //--------------------------------------------------------------------------
    bool LogicalTrace::find_concurrent_colors(
        ReplIndexTask* task,
        std::map<Color, CollectiveID>& concurrent_exchange_colors)
    //--------------------------------------------------------------------------
    {
      std::map<TraceLocalID, std::vector<Color>>::const_iterator finder =
          concurrent_colors.find(task->get_trace_local_id());
      if (finder == concurrent_colors.end())
        return false;
      for (std::vector<Color>::const_iterator it = finder->second.begin();
           it != finder->second.end(); it++)
        concurrent_exchange_colors.emplace(
            std::pair<Color, CollectiveID>(*it, 0));
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::record_concurrent_colors(
        ReplIndexTask* task,
        const std::map<Color, CollectiveID>& concurrent_exchange_colors)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID tlid = task->get_trace_local_id();
      std::vector<Color>& colors = concurrent_colors[tlid];
#ifdef DEBUG_LEGION
      assert(colors.empty());
#endif
      colors.reserve(concurrent_exchange_colors.size());
      for (std::map<Color, CollectiveID>::const_iterator it =
               concurrent_exchange_colors.begin();
           it != concurrent_exchange_colors.end(); it++)
        colors.emplace_back(it->first);
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    UniqueID LogicalTrace::get_current_uid_by_index(unsigned op_idx) const
    //--------------------------------------------------------------------------
    {
      assert(op_idx < operations.size());
      const std::pair<Operation*, GenerationID> key =
          std::make_pair(operations[op_idx].op, operations[op_idx].gen);
      std::map<std::pair<Operation*, GenerationID>, UniqueID>::const_iterator
          finder = current_uids.find(key);
      assert(finder != current_uids.end());
      return finder->second;
    }
#endif

    //--------------------------------------------------------------------------
    void LogicalTrace::translate_dependence_records(
        Operation* op, const unsigned index,
        const std::vector<StaticDependence>& dependences)
    //--------------------------------------------------------------------------
    {
      const bool is_replicated = (context->get_replication_id() > 0);
      for (std::vector<StaticDependence>::const_iterator it =
               dependences.begin();
           it != dependences.end(); it++)
      {
        if (it->dependence_type == LEGION_NO_DEPENDENCE)
          continue;
#ifdef DEBUG_LEGION
        assert(it->previous_offset <= index);
#endif
        // const std::pair<Operation*,GenerationID> &prev =
        std::pair<Operation*, GenerationID> prev = std::make_pair(
            operations[index - it->previous_offset].op,
            operations[index - it->previous_offset].gen);
        unsigned parent_index = op->find_parent_index(it->current_req_index);
        LogicalRegion root_region = context->find_logical_region(parent_index);
        FieldSpaceNode* fs = runtime->get_node(root_region.get_field_space());
        const FieldMask mask = fs->get_field_mask(it->dependent_fields);
        if (is_replicated && !it->shard_only)
        {
          // Need a merge close op to mediate the dependence
          RegionRequirement req(
              root_region, LEGION_READ_WRITE, LEGION_EXCLUSIVE, root_region);
          req.privilege_fields = it->dependent_fields;
#ifdef DEBUG_LEGION_COLLECTIVES
          MergeCloseOp* close_op =
              context->get_merge_close_op(op, runtime->get_node(root_region));
#else
          MergeCloseOp* close_op = context->get_merge_close_op();
#endif
          close_op->initialize(context, req, it->current_req_index, op);
          close_op->update_close_mask(mask);
          register_close(
              close_op, it->current_req_index,
#ifdef DEBUG_LEGION_COLLECTIVES
              runtime->get_node(root_region),
#endif
              req);
          // Mark that we are starting our dependence analysis
          close_op->begin_dependence_analysis();
          // Do any other work for the dependence analysis
          close_op->trigger_dependence_analysis();
          // Record the dependence of the close on the previous op
          close_op->register_region_dependence(
              0 /*close index*/, prev.first, prev.second,
              it->previous_req_index, LEGION_TRUE_DEPENDENCE, mask);
          // Then record our dependence on the close operation
          op->register_region_dependence(
              it->current_req_index, close_op, close_op->get_generation(),
              0 /*close index*/, LEGION_TRUE_DEPENDENCE, mask);
          // Dispatch this close op
          close_op->end_dependence_analysis();
        }
        else
        {
          // Can just record a normal dependence
          op->register_region_dependence(
              it->current_req_index, prev.first, prev.second,
              it->previous_req_index, it->dependence_type, mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalTrace::invalidate_equivalence_sets(void) const
    //--------------------------------------------------------------------------
    {
      if (physical_trace != nullptr)
        physical_trace->invalidate_equivalence_sets();
    }

  }  // namespace Internal
}  // namespace Legion
