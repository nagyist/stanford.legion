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

#include "legion/tracing/viewset.h"
#include "legion/contexts/inner.h"
#include "legion/instances/physical.h"
#include "legion/nodes/region.h"
#include "legion/views/collective.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // TraceViewSet
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    std::string TraceViewSet::FailedPrecondition::to_string(
                                                         TaskContext *ctx) const
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      char *m = mask.to_string();
      if (view->is_fill_view())
      {
        ss << "fill view: " << std::hex << view->did << std::dec
           << ", Index expr: " << expr->expr_id
           << ", Field Mask: " << m;
      }
      else if (view->is_collective_view())
      {
        CollectiveView *collective = view->as_collective_view();
        ss << "collective view: " << std::hex << view->did << std::dec
           << ", Index expr: " << expr->expr_id
           << ", Field Mask: " << m;
        const char *mem_names[] = {
#define MEM_NAMES(name, desc) #name,
            REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
          };
        bool first = true;
        for (std::vector<DistributedID>::const_iterator it =
              collective->instances.begin(); it != 
              collective->instances.end(); it++)
        {
          RtEvent ready;
          PhysicalManager *manager = 
            runtime->find_or_request_instance_manager(*it, ready);
          if (ready.exists())
            ready.wait();
          if (first)
          {
            ss << ", Fields: ";
            FieldSpaceNode *field_space = manager->field_space_node;
            std::vector<FieldID> fields;
            field_space->get_field_set(mask, ctx, fields);
            for (std::vector<FieldID>::const_iterator fit =
                  fields.begin(); fit != fields.end(); fit++)
            {
              if (fit != fields.begin())
                ss << ", ";
              const void *name = nullptr;
              size_t name_size = 0;
              if (field_space->retrieve_semantic_information(
                    LEGION_NAME_SEMANTIC_TAG, name, name_size,
                    true/*can fail*/, false/*wait until*/))
                ss << ((const char*)name) << " (" << *fit << ")";
              else
                ss << *fit;
            }
            ss << ", Instances: ";
            first = false;
          }
          Memory memory = manager->memory_manager->memory;
          ss << "Instance " << std::hex << *it << std::dec
             << " (" << std::hex << manager->get_instance().id 
             << std::dec << ")"
             << " in " << mem_names[memory.kind()]
             << " Memory " << std::hex << memory.id << std::dec;
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(view->is_individual_view());
#endif
        const char *mem_names[] = {
#define MEM_NAMES(name, desc) #name,
            REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
          };
        PhysicalManager *manager =
          view->as_individual_view()->get_manager();
        FieldSpaceNode *field_space = manager->field_space_node;
        Memory memory = manager->memory_manager->memory;

        std::vector<FieldID> fields;
        field_space->get_field_set(mask, ctx, fields);

        ss << "Instance " << std::hex << manager->did << std::dec
           << " (" << std::hex << manager->get_instance().id << std::dec << ")"
           << " in " << mem_names[memory.kind()]
           << " Memory " << std::hex << memory.id << std::dec
           << ", Index expr: " << expr->expr_id
           << ", Field Mask: " << m << ", Fields: ";
        for (std::vector<FieldID>::const_iterator it =
              fields.begin(); it != fields.end(); it++)
        {
          if (it != fields.begin())
            ss << ", ";
          const void *name = nullptr;
          size_t name_size = 0;
          if (field_space->retrieve_semantic_information(
                LEGION_NAME_SEMANTIC_TAG, name, name_size,
                true/*can fail*/, false/*wait until*/))
            ss << ((const char*)name) << " (" << *it << ")";
          else
            ss << *it;
        }
      }
      return ss.str();
    }

    //--------------------------------------------------------------------------
    TraceViewSet::TraceViewSet(InnerContext *ctx, DistributedID own_did, 
                               IndexSpaceExpression *expr, RegionTreeID tid)
      : context(ctx), expression(expr), tree_id(tid), owner_did(
          (own_did > 0) ? own_did : ctx->did), has_collective_views(false)
    //--------------------------------------------------------------------------
    {
      expression->add_nested_expression_reference(owner_did);
      if (owner_did == ctx->did)
        context->add_base_resource_ref(TRACE_REF);
      else
        context->add_nested_resource_ref(owner_did);
    }

    //--------------------------------------------------------------------------
    TraceViewSet::~TraceViewSet(void)
    //--------------------------------------------------------------------------
    {
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); vit++)
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
          if (it->first->remove_nested_expression_reference(owner_did))
            delete it->first;
        if (vit->first->remove_nested_gc_ref(owner_did))
          delete vit->first;
      }
      if (owner_did == context->did)
      {
        if (context->remove_base_resource_ref(TRACE_REF))
          delete context;
      }
      else
      {
        if (context->remove_nested_resource_ref(owner_did))
          delete context;
      }
      if (expression->remove_nested_expression_reference(owner_did))
        delete expression;
      conditions.clear();
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::insert(LogicalView *view, IndexSpaceExpression *expr, 
                              const FieldMask &mask, bool antialiased)
    //--------------------------------------------------------------------------
    {
      ViewExprs::iterator finder = conditions.find(view);
      IndexSpaceExpression *const total_expr = expression; 
      const size_t expr_volume = expr->get_volume();
      if (expr != total_expr)
      {
#ifdef DEBUG_LEGION
        // This is a necessary but not sufficient condition for dominance
        // If we need to we can put in the full intersection test later
        assert(expr_volume <= total_expr->get_volume());
#endif
        // Recognize total expressions when they get here
        if (expr_volume == total_expr->get_volume())
          expr = total_expr;
      }
      // We need to enforce the invariant that there is at most one 
      // expression for field in this function
      if (finder != conditions.end())
      {
        FieldMask set_overlap = mask & finder->second.get_valid_mask();
        if (!!set_overlap)
        {
          if (set_overlap != mask)
          {
            // Handle the difference fields first before we mutate set_overlap
            FieldMask diff = mask - set_overlap;
            if (finder->second.insert(expr, mask))
              expr->add_nested_expression_reference(owner_did);
          }
          FieldMaskSet<IndexSpaceExpression> to_add;
          std::vector<IndexSpaceExpression*> to_delete;
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            const FieldMask overlap = set_overlap & it->second;
            if (!overlap)
              continue;
            if (it->first != total_expr)
            {
              if (it->first != expr)
              {
                // Not the same expression, so compute the union
                IndexSpaceExpression *union_expr = 
                  runtime->union_index_spaces(it->first, expr);
                const size_t union_volume = union_expr->get_volume();
                if (it->first->get_volume() < union_volume)
                {
                  if (expr_volume < union_volume)
                    to_add.insert(union_expr, overlap);
                  else
                    to_add.insert(expr, overlap);
                  it.filter(overlap);
                  if (!it->second)
                    to_delete.push_back(it->first);
                }
                else
                  it.merge(overlap);
              }
              else
                it.merge(overlap);
            }
            set_overlap -= overlap;
            if (!set_overlap)
              break;
          }
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                to_add.begin(); it != to_add.end(); it++)
            if (finder->second.insert(it->first, it->second))
              it->first->add_nested_expression_reference(owner_did);
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            if (to_add.find(*it) != to_add.end())
              continue;
            finder->second.erase(*it);
            if ((*it)->remove_nested_expression_reference(owner_did))
              delete (*it);
          }
        }
        else if (finder->second.insert(expr, mask))
          expr->add_nested_expression_reference(owner_did);
      }
      else
      {
        if (!antialiased)
        {
          if (view->is_collective_view())
          {
            FieldMaskSet<InstanceView> antialiased_views;
            antialias_collective_view(view->as_collective_view(), mask, 
                                      antialiased_views);
            // Now we can insert all the antialiased 
            for (FieldMaskSet<InstanceView>::const_iterator it =
                 antialiased_views.begin(); it != antialiased_views.end(); it++)
              insert(it->first, expr, it->second, true/*antialiased*/);
            return;
          }
          else if (has_collective_views && view->is_instance_view())
            antialias_individual_view(view->as_individual_view(), mask);
        }
        view->add_nested_gc_ref(owner_did);
        expr->add_nested_expression_reference(owner_did);
        conditions[view].insert(expr, mask);
        if (view->is_collective_view())
          has_collective_views = true;
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::insert(LegionMap<LogicalView*,
                  FieldMaskSet<IndexSpaceExpression> > &views, bool antialiased)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMaskSet<IndexSpaceExpression> >::
            const_iterator vit = views.begin(); vit != views.end(); vit++)
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
          insert(vit->first, it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::invalidate(
       LogicalView *view, IndexSpaceExpression *expr, const FieldMask &mask,
       std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove,
       std::map<LogicalView*,unsigned> *view_refs_to_remove, bool antialiased)
    //--------------------------------------------------------------------------
    {
      ViewExprs::iterator finder = conditions.find(view);
      if ((finder == conditions.end()) || 
          (finder->second.get_valid_mask() * mask))
      {
        if (!antialiased)
        {
          if (view->is_collective_view())
          {
            FieldMaskSet<InstanceView> antialiased_views;
            antialias_collective_view(view->as_collective_view(), mask, 
                                      antialiased_views);
            // Now we can insert all the antialiased 
            for (FieldMaskSet<InstanceView>::const_iterator it =
                 antialiased_views.begin(); it != antialiased_views.end(); it++)
              invalidate(it->first, expr, it->second, expr_refs_to_remove,
                  view_refs_to_remove, true/*antialiased*/);
          }
          else if (has_collective_views && view->is_instance_view())
          {
            antialias_individual_view(view->as_individual_view(), mask);
            invalidate(view, expr, mask, expr_refs_to_remove, 
                view_refs_to_remove, true/*antialiased*/);
          }
        }
        return;
      }
      const size_t expr_volume = expr->get_volume();
      IndexSpaceExpression *const total_expr = expression; 
#ifdef DEBUG_LEGION
      // This is a necessary but not sufficient condition for dominance
      // If we need to we can put in the full intersection test later
      assert(expr_volume <= total_expr->get_volume());
#endif
      if ((expr == total_expr) || (expr_volume == total_expr->get_volume()))
      {
        // Expr covers the whole instance so no need to do intersections
        if (!(finder->second.get_valid_mask() - mask))
        {
          // Dominate all fields so just filter everything
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            if (expr_refs_to_remove != nullptr)
            {
              std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                expr_refs_to_remove->find(it->first);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[it->first] = 1;
              else
                finder->second += 1;
            }
            else if (it->first->remove_nested_expression_reference(owner_did))
              delete it->first;
          }
          if (view_refs_to_remove != nullptr)
          {
            std::map<LogicalView*,unsigned>::iterator finder = 
              view_refs_to_remove->find(view);
            if (finder == view_refs_to_remove->end())
              (*view_refs_to_remove)[view] = 1;
            else
              finder->second += 1;
          }
          else if (view->remove_nested_gc_ref(owner_did))
            delete view;
          conditions.erase(finder);
        }
        else
        {
          // Filter on fields
          std::vector<IndexSpaceExpression*> to_delete;
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            it.filter(mask);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            finder->second.erase(*it);
            if (expr_refs_to_remove != nullptr)
            {
              std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                expr_refs_to_remove->find(*it);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[*it] = 1;
              else
                finder->second += 1;
            }
            else if ((*it)->remove_nested_expression_reference(owner_did))
              delete (*it);
          }
          if (finder->second.empty())
          {
            if (view_refs_to_remove != nullptr)
            {
              std::map<LogicalView*,unsigned>::iterator finder = 
                view_refs_to_remove->find(view);
              if (finder == view_refs_to_remove->end())
                (*view_refs_to_remove)[view] = 1;
              else
                finder->second += 1;
            }
            else if (view->remove_nested_gc_ref(owner_did))
              delete view;
            conditions.erase(finder);
          }
          else
            finder->second.tighten_valid_mask();
        }
      }
      else
      {
        // We need intersection tests as part of filtering
        FieldMaskSet<IndexSpaceExpression> to_add;
        std::vector<IndexSpaceExpression*> to_delete;
        for (FieldMaskSet<IndexSpaceExpression>::iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *intersection = expr;
          if (it->first != total_expr)
          {
            intersection = runtime->intersect_index_spaces(it->first, expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            if (volume == expr_volume)
              intersection = expr;
            else if (volume == it->first->get_volume())
              intersection = it->first;
          }
          if (intersection->get_volume() < it->first->get_volume())
          {
            // Only dominated part of it so compute the difference
            IndexSpaceExpression *diff = 
              runtime->subtract_index_spaces(it->first, intersection);
            to_add.insert(diff, overlap);
          }
          // No matter what we're removing these fields for this expr
          it.filter(overlap);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              to_add.begin(); it != to_add.end(); it++)
          if (finder->second.insert(it->first, it->second))
            it->first->add_nested_expression_reference(owner_did);
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          if (to_add.find(*it) != to_add.end())
            continue;
          finder->second.erase(*it);
          if (expr_refs_to_remove != nullptr)
          {
            std::map<IndexSpaceExpression*,unsigned>::iterator finder =
              expr_refs_to_remove->find(*it);
            if (finder == expr_refs_to_remove->end())
              (*expr_refs_to_remove)[*it] = 1;
            else
              finder->second += 1;
          }
          else if ((*it)->remove_nested_expression_reference(owner_did))
            delete (*it);
        }
        if (finder->second.empty())
        {
          if (view_refs_to_remove != nullptr)
          {
            std::map<LogicalView*,unsigned>::iterator finder = 
              view_refs_to_remove->find(view);
            if (finder == view_refs_to_remove->end())
              (*view_refs_to_remove)[view] = 1;
            else
              finder->second += 1;
          }
          else if (view->remove_nested_gc_ref(owner_did))
            delete view;
          conditions.erase(finder);
        }
        else
          finder->second.tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::invalidate_all_but(LogicalView *except,
                              IndexSpaceExpression *expr, const FieldMask &mask,
         std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove,
         std::map<LogicalView*,unsigned> *view_refs_to_remove, bool antialiased)
    //--------------------------------------------------------------------------
    {
      if (!antialiased && (except != nullptr))
      {
        if (except->is_collective_view())
        {
          FieldMaskSet<InstanceView> antialiased_views;
          antialias_collective_view(except->as_collective_view(), mask, 
                                    antialiased_views);
          // Now we can insert all the antialiased 
          for (FieldMaskSet<InstanceView>::const_iterator it =
               antialiased_views.begin(); it != antialiased_views.end(); it++)
            invalidate_all_but(it->first, expr, it->second, expr_refs_to_remove,
                view_refs_to_remove, true/*antialiased*/);
          return;
        }
        else if (has_collective_views && except->is_instance_view())
          antialias_individual_view(except->as_individual_view(), mask);
      }
      std::vector<LogicalView*> to_invalidate;
      for (ViewExprs::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
      {
        if (it->first == except)
          continue;
        if (it->second.get_valid_mask() * mask)
          continue;
        to_invalidate.push_back(it->first);
      }
      for (std::vector<LogicalView*>::const_iterator it = 
            to_invalidate.begin(); it != to_invalidate.end(); it++)
        invalidate(*it, expr, mask, expr_refs_to_remove, 
                   view_refs_to_remove, true/*antialiased*/);
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::dominates(LogicalView *view,
                     IndexSpaceExpression *expr, FieldMask &non_dominated) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!non_dominated);
#endif
      // If this is for an empty equivalence set then it doesn't matter
      if (expr->is_empty())
        return true;
      const size_t expr_volume = expr->get_volume();
      IndexSpaceExpression *const total_expr = expression;
#ifdef DEBUG_LEGION
      // This is a necessary but not sufficient condition for dominance
      // If we need to we can put in the full intersection test later
      assert(expr_volume <= total_expr->get_volume());
#endif
      if (expr_volume == total_expr->get_volume())
        expr = total_expr;
      ViewExprs::const_iterator finder = conditions.find(view);
      if (finder != conditions.end() && 
          !(finder->second.get_valid_mask() * non_dominated))
      {
        if ((expr == total_expr) || (expr_volume == total_expr->get_volume()))
        {
          // Expression is for the whole view, so will only be dominated
          // by the expression for the full view
          FieldMaskSet<IndexSpaceExpression>::const_iterator expr_finder =
            finder->second.find(total_expr);
          if (expr_finder != finder->second.end())
          {
            non_dominated -= expr_finder->second;
            if (!non_dominated)
              return true;
          }
        }
        // There is at most one expression per field so just iterate and compare
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          const FieldMask overlap = non_dominated & it->second;
          if (!overlap)
            continue;
          if ((it->first != total_expr) && (it->first != expr))
          {
            IndexSpaceExpression *intersection = 
              runtime->intersect_index_spaces(it->first, expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            // Can only dominate if we have enough points
            if (volume < expr->get_volume())
              continue;
          }
          // If we get here we were dominated
          non_dominated -= overlap;
          if (!non_dominated)
            return true;
        }
      }
#ifdef DEBUG_LEGION
      assert(!!non_dominated);
#endif
      // If we couldn't find it directly then we need to deal with aliasing
      if (view->is_collective_view())
      {
        CollectiveAntiAlias alias_analysis(view->as_collective_view());
        for (ViewExprs::const_iterator vit =
              conditions.begin(); vit != conditions.end(); vit++)
        {
          if (!vit->first->is_instance_view())
            continue;
          if (vit->second.get_valid_mask() * non_dominated)
            continue;
          InstanceView *inst_view = vit->first->as_instance_view();
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                vit->second.begin(); it != vit->second.end(); it++)
          {
            const FieldMask overlap = it->second & non_dominated;
            if (!overlap)
              continue;
            // No need to be precise here since the resulting analysis
            // on the leaves is filtering and not computing a union
            alias_analysis.traverse(inst_view, overlap, it->first);
          }
        }
        FieldMask dominated = non_dominated;
        FieldMaskSet<IndexSpaceExpression> empty_exprs;
        alias_analysis.visit_leaves(non_dominated, dominated,
                                    expr, empty_exprs);
        if (!!dominated)
          non_dominated -= dominated;
      }
      else if (has_collective_views && view->is_instance_view())
      {
        IndividualView *individual_view = view->as_individual_view();
        for (ViewExprs::const_iterator vit =
              conditions.begin(); vit != conditions.end(); vit++)
        {
          if (!vit->first->is_collective_view())
            continue;
          if (vit->second.get_valid_mask() * non_dominated)
            continue;
          if (!individual_view->aliases(vit->first->as_collective_view()))
            continue;
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                vit->second.begin(); it != vit->second.end(); it++)
          {
            const FieldMask overlap = non_dominated & it->second;
            if (!overlap)
              continue;
            if ((it->first != total_expr) && (it->first != expr))
            {
              IndexSpaceExpression *intersection = 
                runtime->intersect_index_spaces(it->first, expr);
              const size_t volume = intersection->get_volume();
              if (volume == 0)
                continue;
              // Can only dominate if we have enough points
              if (volume < expr->get_volume())
                continue;
            }
            // If we get here we were dominated
            non_dominated -= overlap;
            if (!non_dominated)
              return true;
          }
        }
      }
      // If there are no fields left then we dominated
      return !non_dominated;
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::dominates(LogicalView *view, 
                    IndexSpaceExpression *expr, FieldMask mask,
                    LegionMap<LogicalView*,
                      FieldMaskSet<IndexSpaceExpression> > &non_dominated) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(non_dominated.empty());
#endif
      // If this is for an empty equivalence set then it doesn't matter
      if (expr->is_empty())
        return;
      const size_t expr_volume = expr->get_volume();
      IndexSpaceExpression *const total_expr = expression;
#ifdef DEBUG_LEGION
      // This is a necessary but not sufficient condition for dominance
      // If we need to we can put in the full intersection test later
      assert(expr_volume <= total_expr->get_volume());
#endif
      if (expr_volume == total_expr->get_volume())
        expr = total_expr;
      ViewExprs::const_iterator finder = conditions.find(view);
      if (finder != conditions.end() && 
          !(finder->second.get_valid_mask() * mask))
      {
        if ((expr == total_expr) || (expr_volume == total_expr->get_volume()))
        {
          // Expression is for the whole view, so will only be dominated
          // for the full view
          FieldMaskSet<IndexSpaceExpression>::const_iterator expr_finder =
            finder->second.find(total_expr);
          if (expr_finder != finder->second.end())
          {
            const FieldMask overlap = mask & expr_finder->second;
            if (!!overlap)
            {
              mask -= overlap;
              if (!mask)
                return;
            }
          }
        }
        // There is at most one expression per field so just iterate and compare
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          if ((it->first != total_expr) && (it->first != expr))
          {
            IndexSpaceExpression *intersection = 
              runtime->intersect_index_spaces(it->first, expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            // Can only dominate if we have enough points
            if (volume < expr->get_volume())
            {
              IndexSpaceExpression *diff = 
                runtime->subtract_index_spaces(expr, intersection);
              non_dominated[view].insert(diff, overlap);
            }
          } 
          mask -= overlap;
          // Make sure we keep going if we have non-dominated because
          // we need to check it against any collective aliasing
          if (!mask)
          {
            if (non_dominated.empty() ||
                (!has_collective_views && !view->is_collective_view()))
              return;
            else
              break;
          }
        }
        if (!!mask)
          non_dominated[view].insert(expr, mask);
      }
      else
        non_dominated[view].insert(expr, mask);
#ifdef DEBUG_LEGION
      assert(!non_dominated.empty());
#endif
      FieldMaskSet<IndexSpaceExpression> &non_view = non_dominated[view];
      // Now do the checks for any aliasing with collective views 
      if (view->is_collective_view())
      {
        CollectiveView *collective_view = view->as_collective_view();
        CollectiveAntiAlias alias_analysis(collective_view);
        for (ViewExprs::const_iterator vit =
              conditions.begin(); vit != conditions.end(); vit++)
        {
          if (!vit->first->is_instance_view())
            continue;
          if (vit->second.get_valid_mask() * non_view.get_valid_mask())
            continue;
          InstanceView *inst_view = vit->first->as_instance_view();
          if (!collective_view->aliases(inst_view))
            continue;
          // Only record expressions that are relevant
          LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
            FieldMask> join;
          unique_join_on_field_mask_sets(non_view, vit->second, join);
          for (LegionMap<std::pair<IndexSpaceExpression*,
                IndexSpaceExpression*>,FieldMask>::const_iterator it =
                join.begin(); it != join.end(); it++)
          {
            if (it->first.first != it->first.second)
            {
              IndexSpaceExpression *overlap_expr = 
                runtime->intersect_index_spaces(it->first.first,
                                               it->first.second);
              if (overlap_expr->is_empty())
                continue;
              if (it->first.first->get_volume() == overlap_expr->get_volume())
                alias_analysis.traverse(inst_view, it->second, it->first.first);
              else if (it->first.second->get_volume() == 
                        overlap_expr->get_volume())
                alias_analysis.traverse(inst_view, it->second,it->first.second);
              else
                alias_analysis.traverse(inst_view, it->second, overlap_expr);
            }
            else
              alias_analysis.traverse(inst_view, it->second, it->first.first);
          }
        }
        // For each of the non-dominated expressions go through the
        // alias analysis and get new expressions that are still not
        // dominated even after the alias analysis
        std::vector<IndexSpaceExpression*> to_remove;
        for (FieldMaskSet<IndexSpaceExpression>::iterator it =
              non_view.begin(); it != non_view.end(); it++)
        {
          FieldMask dominated_mask; 
          alias_analysis.visit_leaves(it->second, dominated_mask,
              context, tree_id, collective_view, non_dominated, 
              it->first);
          // Remove any fields that were diffed
          if (!!dominated_mask)
          {
            it.filter(dominated_mask);
            if (!it->second)
              to_remove.push_back(it->first);
          }
        }
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              to_remove.begin(); it != to_remove.end(); it++)
          non_view.erase(*it);
        if (non_view.empty())
          non_dominated.erase(view);
      }
      else if (has_collective_views && view->is_instance_view())
      {
        IndividualView *individual_view = view->as_individual_view();
        for (ViewExprs::const_iterator vit =
              conditions.begin(); vit != conditions.end(); vit++)
        {
          if (!vit->first->is_collective_view())
            continue;
          if (vit->second.get_valid_mask() * non_view.get_valid_mask())
            continue;
          if (!individual_view->aliases(vit->first->as_collective_view()))
            continue;
          // Join on the fields to find expressions that match
          LegionMap<std::pair<IndexSpaceExpression*,
            IndexSpaceExpression*>,FieldMask> join;
          unique_join_on_field_mask_sets(non_view, vit->second, join);
          for (LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                FieldMask>::const_iterator it = join.begin(); 
                it != join.end(); it++)
          {
            IndexSpaceExpression *difference = 
              runtime->subtract_index_spaces(it->first.first, it->first.second);
            if (difference->get_volume() < it->first.first->get_volume())
            {
              FieldMaskSet<IndexSpaceExpression>::iterator finder =
                non_view.find(it->first.first);
              finder.filter(it->second);
              if (!finder->second)
                non_view.erase(finder);
              if (!difference->is_empty())
                non_view.insert(difference, it->second);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::filter_independent_fields(IndexSpaceExpression *expr,
                                                 FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      FieldMask independent = mask;
      for (ViewExprs::const_iterator vit =
            conditions.begin(); vit != conditions.end(); vit++)
      {
        if (independent * vit->second.get_valid_mask())
          continue;
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
        {
          const FieldMask overlap = it->second & independent;
          if (!overlap)
            continue;
          IndexSpaceExpression *overlap_expr = 
            runtime->intersect_index_spaces(it->first, expr);
          if (!overlap_expr->is_empty())
          {
            independent -= overlap;
            if (!independent)
              break;
          }
        }
        if (!independent)
          break;
      }
      if (!!independent)
        mask -= independent;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::subsumed_by(TraceViewSet &set, 
        const FieldMaskSet<IndexSpaceExpression> &unique_dirty_exprs,
        FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      bool subsumed = true;
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator eit =
              vit->second.begin(); eit != vit->second.end(); ++eit)
        {
          // First check to see what fields and expressions are not dominated
          LegionMap<LogicalView*,
                    FieldMaskSet<IndexSpaceExpression> > non_dominated;
          set.dominates(vit->first, eit->first, eit->second, non_dominated);
          if (non_dominated.empty())
            continue;
          // For all the non-dominated fields and expressions we need to
          // check to see if they are dirty or not. If there are any dirty
          // expression-fields that are not dominated then set is no longer
          // subsumed. If the non-dominated fields are not dirty, then it's
          // ok for them to not be subsumed as that means they are just
          // read-only and any additional copies in the postconditions.
          for (LegionMap<LogicalView*,
                FieldMaskSet<IndexSpaceExpression> >::iterator dit =
                non_dominated.begin(); dit != non_dominated.end(); /*nothing*/)
          {
            FieldMaskSet<IndexSpaceExpression> to_add;
            std::vector<IndexSpaceExpression*> to_delete;
            for (FieldMaskSet<IndexSpaceExpression>::iterator nit =
                  dit->second.begin(); nit != dit->second.end(); nit++)
            {
              // Check to see if it interferes with the dirty expressions
              if (nit->second * unique_dirty_exprs.get_valid_mask())
                continue;
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                    unique_dirty_exprs.begin(); it != 
                    unique_dirty_exprs.end(); it++)
              {
                const FieldMask overlap = nit->second & it->second;
                if (!overlap)
                  continue;
                IndexSpaceExpression *expr_overlap = 
                  runtime->intersect_index_spaces(nit->first, it->first);
                if (expr_overlap->is_empty())
                  continue;
                // These are dirty expr-fields which are not subsumed
                subsumed = false;
                if (condition != nullptr)
                {
                  condition->view = vit->first;
                  condition->expr = expr_overlap;
                  condition->mask = overlap;
                }
                if (expr_overlap->get_volume() < nit->first->get_volume())
                {
                  // Not everything is dominated so we need to record it
                  IndexSpaceExpression *non_dirty =
                    runtime->subtract_index_spaces(nit->first, it->first);
                  to_add.insert(non_dirty, overlap);
                }
                nit.filter(overlap);
                if (!nit->second)
                {
                  to_delete.push_back(nit->first);
                  break;
                }
              }
            }
            // Update the non-dominated expressions
            for (std::vector<IndexSpaceExpression*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              if (to_add.find(*it) != to_add.end())
                continue;
              dit->second.erase(*it);
            }
            if (!to_add.empty())
            {
              if (!dit->second.empty())
              {
                for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                      to_add.begin(); it != to_add.end(); it++)
                  dit->second.insert(it->first, it->second);
              }
              else
                dit->second.swap(to_add);
            }
            if (dit->second.empty())
            {
              LegionMap<LogicalView*,
                FieldMaskSet<IndexSpaceExpression> >::iterator 
                  delete_it = dit++;
              non_dominated.erase(delete_it);
            }
            else
              dit++;
          }
          // If there are any remanining non-dominated fields then we
          // add them to the postconditions because views that are both
          // non-dirty and non-dominated need to be in the postconditions
          // so we don't invalidate them when we do the overwriting
          for (LegionMap<LogicalView*,
                FieldMaskSet<IndexSpaceExpression> >::iterator dit =
                non_dominated.begin(); dit != non_dominated.end(); dit++)
          {
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator nit =
                  dit->second.begin(); nit != dit->second.end(); nit++)
            {
              // This is a small optimization to see if there are any
              // fields which are independent for this view against all
              // the other views in the postcondition set. If there are
              // then we don't need to record this view at all since 
              // there won't be any postcondition to overwrite it.
              FieldMask mask = nit->second;
              set.filter_independent_fields(nit->first, mask);
              if (!mask)
                continue;
              set.insert(dit->first, nit->first, mask);
            }
          }
        }
      }
      return subsumed;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::independent_of(const TraceViewSet &set,
                                      FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      if (conditions.size() > set.conditions.size())
        return set.independent_of(*this, condition);
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
      {
        ViewExprs::const_iterator finder = set.conditions.find(vit->first);
        if (finder == set.conditions.end())
        {
          if (vit->first->is_collective_view())
          {
            CollectiveView *collective = vit->first->as_collective_view();
            for (ViewExprs::const_iterator sit = 
                  set.conditions.begin(); sit != set.conditions.end(); sit++)
            {
              if (!sit->first->is_instance_view())
                continue;
              if (vit->second.get_valid_mask() * sit->second.get_valid_mask())
                continue;
              if (!collective->aliases(sit->first->as_instance_view()))
                continue;
              if (has_overlapping_expressions(collective, vit->second, 
                                              sit->second, condition))
                return false;
            }
          }
          else if (set.has_collective_views && vit->first->is_instance_view())
          {
            IndividualView *view = vit->first->as_individual_view();
            for (ViewExprs::const_iterator sit =
                  set.conditions.begin(); sit != set.conditions.end(); sit++)
            {
              if (!sit->first->is_collective_view())
                continue;
              if (vit->second.get_valid_mask() * sit->second.get_valid_mask())
                continue;
              if (!view->aliases(sit->first->as_collective_view()))
                continue;
              if (has_overlapping_expressions(view, vit->second, 
                                              sit->second, condition))
                return false;
            }
          }
          continue;
        }
        if (vit->second.get_valid_mask() * finder->second.get_valid_mask())
          continue;
        if (has_overlapping_expressions(vit->first, vit->second, 
                                        finder->second, condition))
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::has_overlapping_expressions(LogicalView *view,
        const FieldMaskSet<IndexSpaceExpression> &left_exprs,
        const FieldMaskSet<IndexSpaceExpression> &right_exprs,
        FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                FieldMask> overlaps;
      unique_join_on_field_mask_sets(left_exprs, right_exprs, overlaps);
      for (LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                     FieldMask>::const_iterator it = 
            overlaps.begin(); it != overlaps.end(); it++)
      {
        IndexSpaceExpression *overlap = 
          runtime->intersect_index_spaces(it->first.first, it->first.second);
        if (!overlap->is_empty())
        {
          if (condition != nullptr)
          {
            condition->view = view;
            condition->expr = overlap;
            condition->mask = it->second;
          }
          return true;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::record_first_failed(FailedPrecondition *condition) const
    //--------------------------------------------------------------------------
    {
      ViewExprs::const_iterator vit = conditions.begin();
      FieldMaskSet<IndexSpaceExpression>::const_iterator it =
        vit->second.begin();
      condition->view = vit->first;
      condition->expr = it->first;
      condition->mask = it->second;
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::transpose_uniquely(
      LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > &target) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target.empty());
#endif
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
          target[it->first].insert(vit->first, it->second);
      if (target.size() == 1)
        return;
      // Now for the hard part, we need to compare any expresions that overlap
      // and have overlapping fields so we can uniquify them, this reduces the
      // number of analyses in the precondition/anticondition cases, and is 
      // necessary for correctness in the postcondition case where we cannot
      // have multiple overwrites for the same fields and index expressions
      FieldMaskSet<IndexSpaceExpression> expr_fields;
      LegionMap<IndexSpaceExpression*,
                FieldMaskSet<LogicalView> > intermediate;
      intermediate.swap(target);
      for (LegionMap<IndexSpaceExpression*,
            FieldMaskSet<LogicalView> >::const_iterator it =
            intermediate.begin(); it != intermediate.end(); it++)
        expr_fields.insert(it->first, it->second.get_valid_mask());
      LegionList<FieldSet<IndexSpaceExpression*> > field_exprs;
      expr_fields.compute_field_sets(FieldMask(), field_exprs);
      for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator
            eit = field_exprs.begin(); eit != field_exprs.end(); eit++)
      {
        if (eit->elements.size() == 1)
        {
          IndexSpaceExpression *expr = *(eit->elements.begin());
          FieldMaskSet<LogicalView> &src_views = intermediate[expr];
          FieldMaskSet<LogicalView> &dst_views = target[expr];
          // No chance of overlapping so just move everything over
          if (eit->set_mask != src_views.get_valid_mask())
          {
            // Move over the relevant expressions
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  src_views.begin(); it != src_views.end(); it++)
            {
              const FieldMask overlap = eit->set_mask & it->second;
              if (!overlap)
                continue;
              dst_views.insert(it->first, overlap);
            }
          }
          else if (!dst_views.empty())
          {
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  src_views.begin(); it != src_views.end(); it++)
              dst_views.insert(it->first, it->second);
          }
          else
            dst_views.swap(src_views);
          continue;
        }
        // Do pair-wise intersection tests for overlapping of the expressions
        std::vector<IndexSpaceExpression*> disjoint_expressions;
        std::vector<std::vector<IndexSpaceExpression*> > disjoint_components;
        for (std::set<IndexSpaceExpression*>::const_iterator isit = 
              eit->elements.begin(); isit != eit->elements.end(); isit++)
        {
          IndexSpaceExpression *current = *isit;
          const size_t num_expressions = disjoint_expressions.size();
          for (unsigned idx = 0; idx < num_expressions; idx++)
          {
            IndexSpaceExpression *expr = disjoint_expressions[idx];
            // Compute the intersection
            IndexSpaceExpression *intersection =
              runtime->intersect_index_spaces(expr, current);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            if (volume == current->get_volume())
            {
              // this one dominates us, see if we need to split ourself off
              if (volume < expr->get_volume())
              {
                disjoint_expressions.push_back(intersection);
                disjoint_components.resize(disjoint_components.size() + 1);
                std::vector<IndexSpaceExpression*> &components =
                  disjoint_components.back();
                components.insert(components.end(),
                    disjoint_components[idx].begin(), 
                    disjoint_components[idx].end());
                components.push_back(*isit);
                disjoint_expressions[idx] =
                  runtime->subtract_index_spaces(expr, intersection);
              }
              else // Congruent so we are done
                disjoint_components[idx].push_back(*isit);
              current = nullptr;
              break;
            }
            else if (volume == expr->get_volume())
            {
              // We dominate the expression so add ourselves and compute diff
              disjoint_components[idx].push_back(*isit); 
              current = runtime->subtract_index_spaces(current, intersection);
#ifdef DEBUG_LEGION
              assert(!current->is_empty());
#endif
            }
            else
            {
              // Split into the three parts and keep going
              disjoint_expressions.push_back(intersection);
              disjoint_components.resize(disjoint_components.size() + 1);
              std::vector<IndexSpaceExpression*> &components = 
                disjoint_components.back();
              components.insert(components.end(),
                  disjoint_components[idx].begin(), 
                  disjoint_components[idx].end());
              components.push_back(*isit);
              disjoint_expressions[idx] =
                runtime->subtract_index_spaces(expr, intersection);
              current = runtime->subtract_index_spaces(current, intersection);
#ifdef DEBUG_LEGION
              assert(!current->is_empty());
#endif
            }
          }
          if (current != nullptr)
          {
            disjoint_expressions.push_back(current);
            disjoint_components.resize(disjoint_components.size() + 1);
            disjoint_components.back().push_back(*isit);
          }
        }
        // Now we have overlapping expressions and constituents for
        // each of what used to be the old equivalence sets, so we
        // can now build the actual output target
        for (unsigned idx = 0; idx < disjoint_expressions.size(); idx++)
        {
          FieldMaskSet<LogicalView> &dst_views =
            target[disjoint_expressions[idx]];
          for (std::vector<IndexSpaceExpression*>::const_iterator sit =
                disjoint_components[idx].begin(); sit !=
                disjoint_components[idx].end(); sit++)
          {
#ifdef DEBUG_LEGION
            assert(intermediate.find(*sit) != intermediate.end());
#endif
            const FieldMaskSet<LogicalView> &src_views = intermediate[*sit];
            for (FieldMaskSet<LogicalView>::const_iterator it =
                  src_views.begin(); it != src_views.end(); it++)
            {
              const FieldMask overlap = it->second & eit->set_mask;
              if (!overlap)
                continue;
              dst_views.insert(it->first, overlap);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::find_overlaps(TraceViewSet &target, 
                                     IndexSpaceExpression *expr, 
                                     const bool expr_covers, 
                                     const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      if (expr_covers)
      {
        for (ViewExprs::const_iterator vit = 
              conditions.begin(); vit != conditions.end(); vit++)
        {
          if (!(vit->second.get_valid_mask() - mask))
          {
            // sending everything
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
              target.insert(vit->first, it->first, it->second);
          }
          else
          {
            // filtering on fields
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
            {
              const FieldMask overlap = mask & it->second;
              if (!overlap)
                continue;
              target.insert(vit->first, it->first, overlap);
            }
          }
        }
      }
      else
      {
        for (ViewExprs::const_iterator vit = 
              conditions.begin(); vit != conditions.end(); vit++)
        {
          FieldMask view_overlap = vit->second.get_valid_mask() & mask;
          if (!view_overlap)
            continue;
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                vit->second.begin(); it != vit->second.end(); it++)
          {
            const FieldMask overlap = it->second & view_overlap;
            if (!overlap)
              continue;
            IndexSpaceExpression *expr_overlap = 
              runtime->intersect_index_spaces(it->first, expr); 
            const size_t volume = expr_overlap->get_volume();
            if (volume > 0)
            {
              if (volume == expr->get_volume())
                target.insert(vit->first, expr, overlap);
              else if (volume == it->first->get_volume())
                target.insert(vit->first, it->first, overlap);
              else
                target.insert(vit->first, expr_overlap, overlap);
            }
            view_overlap -= overlap;
            if (!view_overlap)
              break;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool TraceViewSet::empty(void) const
    //--------------------------------------------------------------------------
    {
      return conditions.empty();
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::merge(TraceViewSet &target) const
    //--------------------------------------------------------------------------
    {
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
          target.insert(vit->first, it->first, it->second);
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::pack(Serializer &rez, AddressSpaceID target,
                            const bool pack_references) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(conditions.size());
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
      {
        rez.serialize(vit->first->did);
        rez.serialize<size_t>(vit->second.size());
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
        {
          it->first->pack_expression(rez, target);
          rez.serialize(it->second);
        }
        if (pack_references)
          vit->first->pack_global_ref();
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::unpack(Deserializer &derez, size_t num_views,
                         AddressSpaceID source, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx1 = 0; idx1 < num_views; idx1++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view =
          runtime->find_or_request_logical_view(did, ready);
        size_t num_exprs;
        derez.deserialize(num_exprs);
        FieldMaskSet<IndexSpaceExpression> &exprs = conditions[view];
        for (unsigned idx2 = 0; idx2 < num_exprs; idx2++)
        {
          IndexSpaceExpression *expr = 
            IndexSpaceExpression::unpack_expression(derez, source);
          FieldMask mask;
          derez.deserialize(mask);
          if (exprs.insert(expr, mask))
            expr->add_nested_expression_reference(owner_did);
        }
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
        if (LogicalView::is_collective_did(did))
          has_collective_views = true;
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::unpack_references(void) const
    //--------------------------------------------------------------------------
    {
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); vit++)
      {
        vit->first->add_nested_gc_ref(owner_did);
        vit->first->unpack_global_ref();
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::dump(void) const
    //--------------------------------------------------------------------------
    {
      RegionNode *region = runtime->get_tree(tree_id);
      for (ViewExprs::const_iterator vit = 
            conditions.begin(); vit != conditions.end(); ++vit)
      {
        LogicalView *view = vit->first;
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); ++it)
        {
          char *mask = region->column_source->to_string(it->second, context);
          if (view->is_fill_view())
          {
            log_tracing.info() << "  "
                      << "Fill View: " << std::hex << view->did << std::dec
                      << ", Index expr: " << it->first->expr_id
                      << ", Fields: " << mask;
          }
          else if (view->is_collective_view())
          {
            CollectiveView *collective = view->as_collective_view();
            std::stringstream ss;
            for (std::vector<DistributedID>::const_iterator cit =
                  collective->instances.begin(); cit != 
                  collective->instances.end(); cit++)
            {
              RtEvent ready;
              PhysicalManager *manager = 
                runtime->find_or_request_instance_manager(*cit, ready);
              if (ready.exists())
                ready.wait();
              ss << " Instance " << std::hex << manager->did << std::dec
                 << "(" << std::hex << manager->get_instance().id 
                 << std::dec << "),";
            }
            log_tracing.info() << "  Collective "
                      << (view->is_reduction_kind() ? "Reduction " : "")
                      << "View: " << std::hex << view->did << std::dec
                      << ", Index expr: " << it->first->expr_id
                      << ", Fields: " << mask
                      << ", Instances:" << ss.str();
          }
          else
          {
            PhysicalManager *manager = 
              view->as_individual_view()->get_manager();
            log_tracing.info() << "  "
                      << (view->is_reduction_view() ? 
                          "Reduction" : "Normal")
                      << " Instance " << std::hex << manager->did << std::dec
                      << "(" << std::hex << manager->get_instance().id 
                      << std::dec << ")"
                      << ", Index expr: " << it->first->expr_id
                      << ", Fields: " << mask;
          }
          free(mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::antialias_individual_view(IndividualView *view,
                                                 FieldMask mask)
    //--------------------------------------------------------------------------
    {
      if (!has_collective_views)
        return;
      // See if we can find it in which case we know that it doesn't alias
      // with anything else so there is nothing to split
      ViewExprs::const_iterator finder = conditions.find(view);
      if (finder != conditions.end())
      {
        mask -= finder->second.get_valid_mask();
        if (!mask)
          return;
      }
      FieldMaskSet<CollectiveView> to_refine;
      for (ViewExprs::const_iterator it = 
            conditions.begin(); it != conditions.end(); it++)
      {
        if (!it->first->is_collective_view())
          continue;
        const FieldMask overlap = mask & it->second.get_valid_mask();
        if (!overlap)
          continue;
        CollectiveView *collective = it->first->as_collective_view();
        if (!collective->aliases(view))
          continue;
        to_refine.insert(collective, overlap);  
        mask -= overlap;
        if (!mask)
          break;
      }
      // We've got the names of any collective views that need to be
      // refined to not include this individual view, so go ahead and
      // ask the context to make that collective view for us
      std::vector<RtEvent> views_ready;
      std::map<CollectiveView*,PhysicalManager*> individual_results;
      std::map<CollectiveView*,InnerContext::CollectiveResult*> results;
      for (FieldMaskSet<CollectiveView>::const_iterator it = 
            to_refine.begin(); it != to_refine.end(); it++)
      {
        std::vector<DistributedID> dids = it->first->instances;
        std::vector<DistributedID>::iterator finder = 
          std::find(dids.begin(), dids.end(), view->manager->did);
#ifdef DEBUG_LEGION
        assert(finder != dids.end());
#endif
        dids.erase(finder);
        RtEvent ready;
        if (dids.size() > 1)
        {
          InnerContext::CollectiveResult *result =
            context->find_or_create_collective_view(tree_id, dids, ready);
          results[it->first] = result;
        }
        else
        {
          // Just making a single view at this point
          PhysicalManager *manager = 
            runtime->find_or_request_instance_manager(dids.back(), ready);
          individual_results[it->first] = manager;
        }
        if (ready.exists())
          views_ready.push_back(ready);
      }
      if (!views_ready.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(views_ready);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      for (FieldMaskSet<CollectiveView>::const_iterator rit =
            to_refine.begin(); rit != to_refine.end(); rit++)
      {
        RtEvent ready;
        InstanceView *view = nullptr;
        std::map<CollectiveView*,PhysicalManager*>::const_iterator
          individual_finder = individual_results.find(rit->first);
        if (individual_finder == individual_results.end())
        {
#ifdef DEBUG_LEGION
          assert(results.find(rit->first) != results.end());
#endif
          // Common case
          InnerContext::CollectiveResult *result = results[rit->first];
          // Then wait for the collective view to be registered
          if (result->ready_event.exists() && 
              !result->ready_event.has_triggered())
            result->ready_event.wait();
          view = static_cast<InstanceView*>(
              runtime->find_or_request_logical_view(
                result->collective_did, ready));
          if (result->remove_reference())
            delete result;
        }
        else // Unusual case of an downgrading to an individual view
          view = context->create_instance_top_view(individual_finder->second,
                                                   runtime->address_space);
        ViewExprs::iterator finder = conditions.find(rit->first);
        if (finder->second.get_valid_mask() == rit->second)
        {
          // Can just swap expressions over in this particular case
          conditions[view].swap(finder->second);
          // Remove the reference if we have one
          if (finder->first->remove_nested_gc_ref(owner_did))
            delete finder->first;
          conditions.erase(finder);
        }
        else
        {
          // Need to filter over specific expression in this case
          FieldMaskSet<IndexSpaceExpression> &to_add = conditions[view];
          std::vector<IndexSpaceExpression*> to_delete; 
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            const FieldMask overlap = rit->second & it->second;
            if (!overlap)
              continue;
            to_add.insert(it->first, overlap);
            it.filter(overlap);
            if (!it->second) // reference flows back
              to_delete.push_back(it->first);
            else
              it->first->add_nested_expression_reference(owner_did);
          }
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
            finder->second.erase(*it);
          finder->second.tighten_valid_mask();
        }
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        view->add_nested_gc_ref(owner_did);
      }
    }

    //--------------------------------------------------------------------------
    void TraceViewSet::antialias_collective_view(CollectiveView *collective,
                  FieldMask mask, FieldMaskSet<InstanceView> &alternative_views)
    //--------------------------------------------------------------------------
    {
      ViewExprs::const_iterator collective_finder = conditions.find(collective);
      if (collective_finder != conditions.end())
      {
        // If we can already find it then it is already anti-aliased so
        // there's no need to do the rest of this work for those fields
        FieldMask overlap = mask & collective_finder->second.get_valid_mask();
        if (!!overlap)
        {
          alternative_views.insert(collective, overlap);
          mask -= overlap;
          if (!mask)
            return;
        }
      }
      ViewExprs to_add;
      CollectiveAntiAlias alias_analysis(collective);
      for (ViewExprs::iterator vit = conditions.begin(); 
            vit != conditions.end(); /*nothing*/)
      {
        if (!vit->first->is_instance_view())
        {
          vit++;
          continue;
        }
        const FieldMask view_overlap = mask & vit->second.get_valid_mask();
        if (!view_overlap)
        {
          vit++;
          continue;
        }
        if (vit->first->is_collective_view())
        {
          CollectiveView *current = vit->first->as_collective_view();
          // See how the instances overlap
          // First get the intersection
          std::vector<DistributedID> intersection;
          if (current->instances.size() < collective->instances.size())
          {
            for (std::vector<DistributedID>::const_iterator it =
                  current->instances.begin(); it !=
                  current->instances.end(); it++)
              if (std::binary_search(collective->instances.begin(),
                    collective->instances.end(), *it))
                intersection.push_back(*it);
          }
          else
          {
            for (std::vector<DistributedID>::const_iterator it =
                  collective->instances.begin(); it !=
                  collective->instances.end(); it++)
              if (std::binary_search(current->instances.begin(),
                    current->instances.end(), *it))
                intersection.push_back(*it);
          }
          // If they don't overlap at all then there's nothing to do
          if (intersection.empty())
          {
            vit++;
            continue;
          }
          // Don't care about expressions for this analysis
          // but we're reusing an exisint alias so we have to
          // conform to get the linker to work
          IndexSpaceExpression *null_expr = nullptr;
          alias_analysis.traverse(current, view_overlap, null_expr);
          if (intersection.size() == current->instances.size())
          {
#ifdef DEBUG_LEGION
            assert(intersection.size() < collective->instances.size());
#endif
            vit++;
          }
          else
          {
            // Otherwise, if vit->first is not covered by the intersection
            // then we need to do two things
            // 1. Create a new instance for the difference and record
            //    any overlapping expressions for that in to_add
            std::vector<DistributedID> difference;
            for (std::vector<DistributedID>::const_iterator it =
                  current->instances.begin(); it != 
                  current->instances.end(); it++)
              if (!std::binary_search(collective->instances.begin(),
                    collective->instances.end(), *it))
                difference.push_back(*it);
            InstanceView *diff_view = find_instance_view(difference);
            if (to_add.find(diff_view) == to_add.end())
              diff_view->add_nested_gc_ref(owner_did);
            // 2. Make a new instance for the intersection, analyze it
            //    and record any overlapping expressions in to_add
            InstanceView *inter_view = 
              (intersection.size() == collective->instances.size()) ?
              collective : find_instance_view(intersection);
            if (to_add.find(inter_view) == to_add.end())
              inter_view->add_nested_gc_ref(owner_did);
            std::vector<IndexSpaceExpression*> to_delete;
            for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
            {
              const FieldMask overlap = view_overlap & it->second;
              if (!overlap)
                continue;
              if (to_add[diff_view].insert(it->first, overlap))
                it->first->add_nested_expression_reference(owner_did);
              to_add[inter_view].insert(it->first, overlap);
              it.filter(overlap);
              if (!it->second) // reference flows back
                to_delete.push_back(it->first);
              else
                it->first->add_nested_expression_reference(owner_did);
            }
            if (to_delete.size() < vit->second.size())
            {
              for (std::vector<IndexSpaceExpression*>::const_iterator it =
                    to_delete.begin(); it != to_delete.end(); it++)
                vit->second.erase(*it);
              vit->second.tighten_valid_mask();
              vit++;
            }
            else
            {
              vit->second.clear();
              if (vit->first->remove_nested_gc_ref(owner_did))
                delete vit->first;
              ViewExprs::iterator to_delete = vit++;
              conditions.erase(to_delete);
            }
          }
        }
        else // just an individual view, so we can just traverse it
        {
          IndividualView *individual = vit->first->as_individual_view();
          // Check to see if it they alias
          if (std::binary_search(collective->instances.begin(),
                collective->instances.end(), individual->manager->did))
          {
            // Don't care about expressions for this analysis
            // but we're reusing an exisint alias so we have to
            // conform to get the linker to work
            IndexSpaceExpression *null_expr = nullptr;
            alias_analysis.traverse(individual, view_overlap, null_expr);
          }
          vit++;
        }
      }
      // Now traverse the alias analysis and record the alternate views
      // and their index space expressions in to_add
      FieldMask allvalid_mask = mask;
      alias_analysis.visit_leaves(mask, allvalid_mask, 
                                  *this, alternative_views);
      if (!!allvalid_mask)
        alternative_views.insert(collective, allvalid_mask);
      if (!to_add.empty())
      {
        for (ViewExprs::iterator vit = to_add.begin(); 
              vit != to_add.end(); vit++)
        {
          ViewExprs::iterator finder = conditions.find(vit->first);
          if (finder != conditions.end())
          {
            // Remove duplicate view reference
            vit->first->remove_nested_gc_ref(owner_did);
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  vit->second.begin(); it != vit->second.end(); it++)
              // Remove duplicate references
              if (!finder->second.insert(it->first, it->second))
                it->first->remove_nested_expression_reference(owner_did);
          }
          else
          {
            // Already have a reference to the view so pass it here
            // Also have references on the expression so the swap is enough
            conditions[vit->first].swap(vit->second);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* TraceViewSet::find_instance_view(
                                        const std::vector<DistributedID> &dids)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!dids.empty());
#endif
      if (dids.size() > 1)
      {
        RtEvent ready;
        InnerContext::CollectiveResult *result =
          context->find_or_create_collective_view(tree_id, dids, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        // Then wait for the collective view to be registered
        if (result->ready_event.exists() && 
            !result->ready_event.has_triggered())
          result->ready_event.wait();
        InstanceView *view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(
            result->collective_did, ready));
        if (result->remove_reference())
          delete result;
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        return view;
      }
      else
      {
        RtEvent ready;
        PhysicalManager *manager =
          runtime->find_or_request_instance_manager(
              dids.back(), ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        return context->create_instance_top_view(manager,
                        runtime->address_space);
      }
    }

  } // namespace Internal
} // namespace Legion
