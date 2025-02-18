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

#include "legion/views/individual.h"
#include "legion/analysis/collective.h"
#include "legion/instances/physical.h"
#include "legion/nodes/across.h"
#include "legion/nodes/index.h"
#include "legion/views/allreduce.h"
#include "legion/views/collective.h"
#include "legion/views/fill.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Physical User
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(
        const RegionUsage& u, IndexSpaceExpression* e, ApEvent term,
        UniqueID id, unsigned x, bool cpy, bool cov)
      : usage(u), expr(e), term_event(term), op_id(id), index(x),
        copy_user(cpy), covers(cov)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != nullptr);
#endif
      expr->add_base_expression_reference(PHYSICAL_USER_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalUser::~PhysicalUser(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != nullptr);
#endif
      if (expr->remove_base_expression_reference(PHYSICAL_USER_REF))
        delete expr;
    }

    /////////////////////////////////////////////////////////////
    // ExprView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExprView::ExprView(
        DistributedID did, IndexSpaceExpression* exp, bool unbound)
      : view_expr(unbound ? exp : exp->get_canonical_expression()),
        view_volume(std::numeric_limits<size_t>::max()), view_did(did),
        invalid_fields(FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES))
    //--------------------------------------------------------------------------
    {
      view_expr->add_nested_expression_reference(view_did);
    }

    //--------------------------------------------------------------------------
    ExprView::~ExprView(void)
    //--------------------------------------------------------------------------
    {
      if (view_expr->remove_nested_expression_reference(view_did))
        delete view_expr;
      if (!subviews.empty())
      {
        for (FieldMaskSet<ExprView>::iterator it = subviews.begin();
             it != subviews.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
      }
      // If we have any current or previous users filter them out now
      if (!current_epoch_users.empty())
      {
        for (FieldMaskSet<PhysicalUser>::const_iterator it =
                 current_epoch_users.begin();
             it != current_epoch_users.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        current_epoch_users.clear();
      }
      if (!previous_epoch_users.empty())
      {
        for (FieldMaskSet<PhysicalUser>::const_iterator it =
                 previous_epoch_users.begin();
             it != previous_epoch_users.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        previous_epoch_users.clear();
      }
    }

    //--------------------------------------------------------------------------
    size_t ExprView::get_view_volume(void)
    //--------------------------------------------------------------------------
    {
      size_t result = view_volume.load();
      if (result != std::numeric_limits<size_t>::max())
        return result;
      result = view_expr->get_volume();
#ifdef DEBUG_LEGION
      assert(result != std::numeric_limits<size_t>::max());
#endif
      view_volume.store(result);
      return result;
    }

    //--------------------------------------------------------------------------
    void ExprView::find_all_done_events(std::set<ApEvent>& all_done) const
    //--------------------------------------------------------------------------
    {
      // No need for any locks here since we're in the view destructor
      // and there should be no more races between anything
      for (FieldMaskSet<PhysicalUser>::const_iterator it =
               current_epoch_users.begin();
           it != current_epoch_users.end(); it++)
        all_done.insert(it->first->term_event);
      for (FieldMaskSet<PhysicalUser>::const_iterator it =
               previous_epoch_users.begin();
           it != previous_epoch_users.end(); it++)
        all_done.insert(it->first->term_event);
      for (FieldMaskSet<ExprView>::const_iterator it = subviews.begin();
           it != subviews.end(); it++)
        it->first->find_all_done_events(all_done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExprView::verify_current_to_filter(
        const FieldMask& dominated,
        FieldMaskSet<PhysicalUser>& current_to_filter)
    //--------------------------------------------------------------------------
    {
      if (!!dominated)
      {
        const FieldMask non_dominated =
            current_to_filter.get_valid_mask() - dominated;
        if (!non_dominated)
          return;
        if (non_dominated != current_to_filter.get_valid_mask())
        {
          // Selectively filter
          std::vector<PhysicalUser*> to_delete;
          for (FieldMaskSet<PhysicalUser>::iterator it =
                   current_to_filter.begin();
               it != current_to_filter.end(); it++)
          {
            it.filter(non_dominated);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          for (std::vector<PhysicalUser*>::const_iterator it =
                   to_delete.begin();
               it != to_delete.end(); it++)
          {
            current_to_filter.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
          }
          current_to_filter.tighten_valid_mask();
          return;
        }
      }
      // Otherwise we fall through here and clean out all the users
      for (FieldMaskSet<PhysicalUser>::const_iterator it =
               current_to_filter.begin();
           it != current_to_filter.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      current_to_filter.clear();
    }

    //--------------------------------------------------------------------------
    void ExprView::find_user_preconditions(
        const RegionUsage& usage, IndexSpaceExpression* user_expr,
        const bool user_dominates, const FieldMask& user_mask,
        ApEvent term_event, UniqueID op_id, unsigned index,
        std::set<ApEvent>& preconditions, const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      FieldMask dominated;
      std::set<PhysicalUser*> dead_users;
      FieldMaskSet<PhysicalUser> current_to_filter, previous_to_filter;
      // Perform the analysis with a read-only lock
      {
        AutoLock v_lock(view_lock, 1, false /*exclusive*/);
        // Check to see if we dominate when doing this analysis and
        // can therefore filter or whether we are just intersecting
        // Do the local analysis
        if (user_dominates)
        {
          // We dominate in this case so we can do filtering
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(
                usage, user_mask, user_expr, term_event, op_id, index,
                user_dominates, preconditions, dead_users, current_to_filter,
                observed, non_dominated, trace_recording, false /*copy*/);
            if (!!observed)
              dominated = observed - non_dominated;
          }
          if (!previous_epoch_users.empty())
          {
            if (!!dominated)
              find_previous_filter_users(dominated, previous_to_filter);
            const FieldMask previous_mask = user_mask - dominated;
            if (!!previous_mask)
              find_previous_preconditions(
                  usage, previous_mask, user_expr, term_event, op_id, index,
                  user_dominates, preconditions, dead_users, trace_recording,
                  false /*copy*/);
          }
        } else
        {
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(
                usage, user_mask, user_expr, term_event, op_id, index,
                user_dominates, preconditions, dead_users, current_to_filter,
                observed, non_dominated, trace_recording, false /*copy*/);
#ifdef DEBUG_LEGION
            assert(!observed);
            assert(current_to_filter.empty());
#endif
          }
          if (!previous_epoch_users.empty())
            find_previous_preconditions(
                usage, user_mask, user_expr, term_event, op_id, index,
                user_dominates, preconditions, dead_users, trace_recording,
                false /*copy*/);
        }
      }
      // It's possible that we recorded some users for fields which
      // are not actually fully dominated, if so we need to prune them
      // otherwise we can get into issues of soundness
      if (!current_to_filter.empty())
        verify_current_to_filter(dominated, current_to_filter);
      if (!dead_users.empty() || !previous_to_filter.empty() ||
          !current_to_filter.empty())
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_users.empty())
          filter_dead_users(dead_users);
        if (!previous_to_filter.empty())
          filter_previous_users(previous_to_filter);
        if (!current_to_filter.empty())
          filter_current_users(current_to_filter);
      }
      // Then see if there are any users below that we need to traverse
      if (!subviews.empty() && !(subviews.get_valid_mask() * user_mask))
      {
        FieldMaskSet<ExprView> to_traverse;
        std::map<ExprView*, IndexSpaceExpression*> traverse_exprs;
        for (FieldMaskSet<ExprView>::const_iterator it = subviews.begin();
             it != subviews.end(); it++)
        {
          FieldMask overlap = it->second & user_mask;
          if (!overlap)
            continue;
          // If we've already determined the user dominates
          // then we don't even have to do this test
          if (user_dominates)
          {
            to_traverse.insert(it->first, overlap);
            continue;
          }
          if (it->first->view_expr == user_expr)
          {
            to_traverse.insert(it->first, overlap);
            traverse_exprs[it->first] = user_expr;
            continue;
          }
          IndexSpaceExpression* expr_overlap =
              runtime->intersect_index_spaces(user_expr, it->first->view_expr);
          if (!expr_overlap->is_empty())
          {
            to_traverse.insert(it->first, overlap);
            traverse_exprs[it->first] = expr_overlap;
          }
        }
        if (!to_traverse.empty())
        {
          if (user_dominates)
          {
            for (FieldMaskSet<ExprView>::const_iterator it =
                     to_traverse.begin();
                 it != to_traverse.end(); it++)
              it->first->find_user_preconditions(
                  usage, it->first->view_expr, true /*dominate*/, it->second,
                  term_event, op_id, index, preconditions, trace_recording);
          } else
          {
            for (FieldMaskSet<ExprView>::const_iterator it =
                     to_traverse.begin();
                 it != to_traverse.end(); it++)
            {
              IndexSpaceExpression* intersect = traverse_exprs[it->first];
              const bool user_dominates =
                  (intersect->expr_id == it->first->view_expr->expr_id) ||
                  (intersect->get_volume() == it->first->get_view_volume());
              it->first->find_user_preconditions(
                  usage, intersect, user_dominates, it->second, term_event,
                  op_id, index, preconditions, trace_recording);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_copy_preconditions(
        const RegionUsage& usage, IndexSpaceExpression* copy_expr,
        const bool copy_dominates, const FieldMask& copy_mask, UniqueID op_id,
        unsigned index, std::set<ApEvent>& preconditions,
        const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      FieldMask dominated;
      std::set<PhysicalUser*> dead_users;
      FieldMaskSet<PhysicalUser> current_to_filter, previous_to_filter;
      // Do the first pass with a read-only lock on the events
      {
        AutoLock v_lock(view_lock, 1, false /*exclusive*/);
        // Check to see if we dominate when doing this analysis and
        // can therefore filter or whether we are just intersecting
        // Do the local analysis
        if (copy_dominates)
        {
          // We dominate in this case so we can do filtering
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(
                usage, copy_mask, copy_expr, ApEvent::NO_AP_EVENT, op_id, index,
                copy_dominates, preconditions, dead_users, current_to_filter,
                observed, non_dominated, trace_recording, true /*copy user*/);
            if (!!observed)
              dominated = observed - non_dominated;
          }
          if (!previous_epoch_users.empty())
          {
            if (!!dominated)
              find_previous_filter_users(dominated, previous_to_filter);
            const FieldMask previous_mask = copy_mask - dominated;
            if (!!previous_mask)
              find_previous_preconditions(
                  usage, previous_mask, copy_expr, ApEvent::NO_AP_EVENT, op_id,
                  index, copy_dominates, preconditions, dead_users,
                  trace_recording, true /*copy user*/);
          }
        } else
        {
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(
                usage, copy_mask, copy_expr, ApEvent::NO_AP_EVENT, op_id, index,
                copy_dominates, preconditions, dead_users, current_to_filter,
                observed, non_dominated, trace_recording, true /*copy user*/);
#ifdef DEBUG_LEGION
            assert(!observed);
            assert(current_to_filter.empty());
#endif
          }
          if (!previous_epoch_users.empty())
            find_previous_preconditions(
                usage, copy_mask, copy_expr, ApEvent::NO_AP_EVENT, op_id, index,
                copy_dominates, preconditions, dead_users, trace_recording,
                true /*copy user*/);
        }
      }
      // It's possible that we recorded some users for fields which
      // are not actually fully dominated, if so we need to prune them
      // otherwise we can get into issues of soundness
      if (!current_to_filter.empty())
        verify_current_to_filter(dominated, current_to_filter);
      if (!dead_users.empty() || !previous_to_filter.empty() ||
          !current_to_filter.empty())
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_users.empty())
          filter_dead_users(dead_users);
        if (!previous_to_filter.empty())
          filter_previous_users(previous_to_filter);
        if (!current_to_filter.empty())
          filter_current_users(current_to_filter);
      }
      // Then see if there are any users below that we need to traverse
      if (!subviews.empty() && !(subviews.get_valid_mask() * copy_mask))
      {
        for (FieldMaskSet<ExprView>::const_iterator it = subviews.begin();
             it != subviews.end(); it++)
        {
          FieldMask overlap = it->second & copy_mask;
          if (!overlap)
            continue;
          // If the copy dominates then we don't even have
          // to do the intersection test
          if (copy_dominates)
          {
            it->first->find_copy_preconditions(
                usage, it->first->view_expr, true /*dominate*/, overlap, op_id,
                index, preconditions, trace_recording);
            continue;
          }
          if (it->first->view_expr == copy_expr)
          {
            it->first->find_copy_preconditions(
                usage, copy_expr, true /*dominate*/, overlap, op_id, index,
                preconditions, trace_recording);
            continue;
          }
          IndexSpaceExpression* expr_overlap =
              runtime->intersect_index_spaces(it->first->view_expr, copy_expr);
          if (!expr_overlap->is_empty())
          {
            const bool copy_dominates =
                (expr_overlap->expr_id == it->first->view_expr->expr_id) ||
                (expr_overlap->get_volume() == it->first->get_view_volume());
            it->first->find_copy_preconditions(
                usage, expr_overlap, copy_dominates, overlap, op_id, index,
                preconditions, trace_recording);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_last_users(
        const RegionUsage& usage, IndexSpaceExpression* expr,
        const bool expr_dominates, const FieldMask& mask,
        std::set<ApEvent>& last_events) const
    //--------------------------------------------------------------------------
    {
      // See if there are any users below that we need to traverse
      if (!subviews.empty() && !(subviews.get_valid_mask() * mask))
      {
        for (FieldMaskSet<ExprView>::const_iterator it = subviews.begin();
             it != subviews.end(); it++)
        {
          FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          // If the expr dominates then we don't even have
          // to do the intersection test
          if (expr_dominates)
          {
            it->first->find_last_users(
                usage, it->first->view_expr, true /*dominate*/, overlap,
                last_events);
            continue;
          }
          if (it->first->view_expr == expr)
          {
            it->first->find_last_users(
                usage, expr, true /*dominate*/, overlap, last_events);
            continue;
          }
          IndexSpaceExpression* expr_overlap =
              runtime->intersect_index_spaces(it->first->view_expr, expr);
          if (!expr_overlap->is_empty())
          {
            const bool dominates =
                (expr_overlap->expr_id == it->first->view_expr->expr_id) ||
                (expr_overlap->get_volume() == it->first->get_view_volume());
            it->first->find_last_users(
                usage, expr_overlap, dominates, overlap, last_events);
          }
        }
      }
      FieldMask dominated;
      // Now we can traverse at this level
      AutoLock v_lock(view_lock, 1, false /*exclusive*/);
      // We dominate in this case so we can do filtering
      if (!current_epoch_users.empty())
      {
        FieldMask observed, non_dominated;
        find_current_preconditions(
            usage, mask, expr, expr_dominates, last_events, observed,
            non_dominated);
        if (!!observed)
          dominated = observed - non_dominated;
      }
      if (!previous_epoch_users.empty())
      {
        const FieldMask previous_mask = mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(
              usage, previous_mask, expr, expr_dominates, last_events);
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::insert_subview(ExprView* subview, FieldMask& subview_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this != subview);
#endif
      // Iterate over all subviews and see which ones we dominate and which
      // ones dominate the subview
      if (!subviews.empty() && !(subviews.get_valid_mask() * subview_mask))
      {
        bool need_tighten = true;
        std::vector<ExprView*> to_delete;
        FieldMaskSet<ExprView> dominating_subviews;

        for (FieldMaskSet<ExprView>::iterator it = subviews.begin();
             it != subviews.end(); it++)
        {
          // See if we intersect on fields
          FieldMask overlap_mask = it->second & subview_mask;
          if (!overlap_mask)
            continue;
          IndexSpaceExpression* overlap = runtime->intersect_index_spaces(
              subview->view_expr, it->first->view_expr);
          const size_t overlap_volume = overlap->get_volume();
          if (overlap_volume == 0)
            continue;
          // See if we dominate or just intersect
          if (overlap_volume == subview->get_view_volume())
          {
#ifdef DEBUG_LEGION
            // Should only strictly dominate if they were congruent
            // then we wouldn't be inserting in the first place
            assert(overlap_volume < it->first->get_view_volume());
#endif
            // Dominator so we can just continue traversing
            dominating_subviews.insert(it->first, overlap_mask);
          } else if (overlap_volume == it->first->get_view_volume())
          {
#ifdef DEBUG_LEGION
            assert(overlap_mask * dominating_subviews.get_valid_mask());
#endif
            // We dominate this view so we can just pull it
            // in underneath of us now
            it.filter(overlap_mask);
            subview->insert_subview(it->first, overlap_mask);
            need_tighten = true;
            // See if we need to remove this subview
            if (!it->second)
              to_delete.push_back(it->first);
          }
          // Otherwise it's just a normal intersection
        }
        // See if we had any dominators
        if (!dominating_subviews.empty())
        {
          if (dominating_subviews.size() > 1)
          {
            // We need to deduplicate finding or making the new ExprView
            // First check to see if we have it already in one sub-tree
            // If not, we'll pick the one with the smallest bounding volume
            LegionMap<std::pair<size_t /*volume*/, ExprView*>, FieldMask>
                sorted_subviews;
            for (FieldMaskSet<ExprView>::const_iterator it =
                     dominating_subviews.begin();
                 it != dominating_subviews.end(); it++)
            {
              FieldMask overlap = it->second;
              // Channeling Tuco here
              it->first->find_tightest_subviews(
                  subview->view_expr, overlap, sorted_subviews);
            }
            for (LegionMap<std::pair<size_t, ExprView*>, FieldMask>::
                     const_iterator it = sorted_subviews.begin();
                 it != sorted_subviews.end(); it++)
            {
              FieldMask overlap = it->second & subview_mask;
              if (!overlap)
                continue;
              subview_mask -= overlap;
              it->first.second->insert_subview(subview, overlap);
              if (!subview_mask ||
                  (subview_mask * dominating_subviews.get_valid_mask()))
                break;
            }
#ifdef DEBUG_LEGION
            assert(subview_mask * dominating_subviews.get_valid_mask());
#endif
          } else
          {
            FieldMaskSet<ExprView>::const_iterator first =
                dominating_subviews.begin();
            FieldMask dominated_mask = first->second;
            subview_mask -= dominated_mask;
            first->first->insert_subview(subview, dominated_mask);
          }
        }
        if (!to_delete.empty())
        {
          for (std::vector<ExprView*>::const_iterator it = to_delete.begin();
               it != to_delete.end(); it++)
          {
            subviews.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
          }
        }
        if (need_tighten)
          subviews.tighten_valid_mask();
      }
      // If we make it here and there are still fields then we need to
      // add it locally
      if (!!subview_mask && subviews.insert(subview, subview_mask))
        subview->add_reference();
    }

    //--------------------------------------------------------------------------
    void ExprView::find_tightest_subviews(
        IndexSpaceExpression* expr, FieldMask& expr_mask,
        LegionMap<std::pair<size_t, ExprView*>, FieldMask>& bounding_views)
    //--------------------------------------------------------------------------
    {
      if (!subviews.empty() && !(expr_mask * subviews.get_valid_mask()))
      {
        FieldMask dominated_mask;
        for (FieldMaskSet<ExprView>::iterator it = subviews.begin();
             it != subviews.end(); it++)
        {
          // See if we intersect on fields
          FieldMask overlap_mask = it->second & expr_mask;
          if (!overlap_mask)
            continue;
          IndexSpaceExpression* overlap =
              runtime->intersect_index_spaces(expr, it->first->view_expr);
          const size_t overlap_volume = overlap->get_volume();
          if (overlap_volume == 0)
            continue;
          // See if we dominate or just intersect
          if (overlap_volume == expr->get_volume())
          {
#ifdef DEBUG_LEGION
            // Should strictly dominate otherwise we'd be congruent
            assert(overlap_volume < it->first->get_view_volume());
#endif
            dominated_mask |= overlap_mask;
            // Continute the traversal
            it->first->find_tightest_subviews(
                expr, overlap_mask, bounding_views);
          }
        }
        // Remove any dominated fields from below
        if (!!dominated_mask)
          expr_mask -= dominated_mask;
      }
      // If we still have fields then record ourself
      if (!!expr_mask)
      {
        std::pair<size_t, ExprView*> key(get_view_volume(), this);
        bounding_views[key] |= expr_mask;
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::add_partial_user(
        const RegionUsage& usage, UniqueID op_id, unsigned index,
        FieldMask user_mask, const ApEvent term_event,
        IndexSpaceExpression* user_expr, const size_t user_volume,
        PhysicalUser*& covered_user, PhysicalUser*& uncovered_user)
    //--------------------------------------------------------------------------
    {
      // We're going to try to put this user as far down the ExprView tree
      // as we can in order to avoid doing unnecessary intersection tests later
      {
        // Find all the intersecting subviews to see if we can
        // continue the traversal
        // No need for the view lock anymore since we're protected
        // by the expr_lock at the top of the tree
        // AutoLock v_lock(view_lock,1,false/*exclusive*/);
        for (FieldMaskSet<ExprView>::const_iterator it = subviews.begin();
             it != subviews.end(); it++)
        {
          // If the fields don't overlap then we don't care
          const FieldMask overlap_mask = it->second & user_mask;
          if (!overlap_mask)
            continue;
          IndexSpaceExpression* overlap =
              runtime->intersect_index_spaces(user_expr, it->first->view_expr);
          const size_t overlap_volume = overlap->get_volume();
          if (overlap_volume == user_volume)
          {
            // Check for the cases where we dominated perfectly
            if (overlap_volume == it->first->get_view_volume())
            {
              if (covered_user == nullptr)
              {
                covered_user = new PhysicalUser(
                    usage, user_expr, term_event, op_id, index, true /*copy*/,
                    true /*covers*/);
                covered_user->add_reference();
              }
              it->first->add_current_user(covered_user, overlap_mask);
            } else
            {
              // Continue the traversal on this node
              it->first->add_partial_user(
                  usage, op_id, index, overlap_mask, term_event, user_expr,
                  user_volume, covered_user, uncovered_user);
            }
            // We only need to record the partial user in one sub-tree
            // where it is dominated in order to be sound
            user_mask -= overlap_mask;
            if (!user_mask)
              break;
          }
          // Otherwise for all other cases we're going to record it here
          // because they don't dominate the user to be recorded
        }
      }
      // If we still have local fields, make a user and record it here
      if (!!user_mask)
      {
        if (uncovered_user == nullptr)
        {
          uncovered_user = new PhysicalUser(
              usage, user_expr, term_event, op_id, index, true /*copy*/,
              false /*covers*/);
          uncovered_user->add_reference();
        }
        add_current_user(uncovered_user, user_mask);
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::add_current_user(PhysicalUser* user, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      if (current_epoch_users.insert(user, mask))
        user->add_reference();
    }

    //--------------------------------------------------------------------------
    void ExprView::clean_views(
        FieldMask& valid_mask, FieldMaskSet<ExprView>& clean_set)
    //--------------------------------------------------------------------------
    {
      // Handle the base case if we already did it
      FieldMaskSet<ExprView>::const_iterator finder = clean_set.find(this);
      if (finder != clean_set.end())
      {
        valid_mask = finder->second;
        return;
      }
      // No need to hold the lock for this part we know that no one
      // is going to be modifying this data structure at the same time
      FieldMaskSet<ExprView> new_subviews;
      std::vector<ExprView*> to_delete;
      for (FieldMaskSet<ExprView>::iterator it = subviews.begin();
           it != subviews.end(); it++)
      {
        FieldMask new_mask;
        it->first->clean_views(new_mask, clean_set);
        // Save this as part of the valid mask without filtering
        valid_mask |= new_mask;
        // Have to make sure to filter this by the previous set of fields
        // since we could get more than we initially had
        // We also need update the invalid fields if we remove a path
        // to the subview
        if (!!new_mask)
        {
          new_mask &= it->second;
          const FieldMask new_invalid = it->second - new_mask;
          if (!!new_invalid)
          {
#ifdef DEBUG_LEGION
            // Should only have been one path here
            assert(it->first->invalid_fields * new_invalid);
#endif
            it->first->invalid_fields |= new_invalid;
          }
        } else
        {
#ifdef DEBUG_LEGION
          // Should only have been one path here
          assert(it->first->invalid_fields * it->second);
#endif
          it->first->invalid_fields |= it->second;
        }
        if (!!new_mask)
          new_subviews.insert(it->first, new_mask);
        else
          to_delete.push_back(it->first);
      }
      subviews.swap(new_subviews);
      if (!to_delete.empty())
      {
        for (std::vector<ExprView*>::const_iterator it = to_delete.begin();
             it != to_delete.end(); it++)
          if ((*it)->remove_reference())
            delete (*it);
      }
      AutoLock v_lock(view_lock);
      if (!current_epoch_users.empty())
        valid_mask |= current_epoch_users.get_valid_mask();
      if (!previous_epoch_users.empty())
        valid_mask |= previous_epoch_users.get_valid_mask();
      // Save this for the future so we don't need to compute it again
      if (clean_set.insert(this, valid_mask))
        add_reference();
    }

    //--------------------------------------------------------------------------
    void ExprView::filter_dead_users(const std::set<PhysicalUser*>& dead_users)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are in Legion Spy since we want to see
      // all of the dependences on an instance
      for (std::set<PhysicalUser*>::const_iterator it = dead_users.begin();
           it != dead_users.end(); it++)
      {
        unsigned refs_to_remove = 1;
#ifndef LEGION_DISABLE_EVENT_PRUNING
        FieldMaskSet<PhysicalUser>::iterator finder =
            current_epoch_users.find(*it);
        if (finder != current_epoch_users.end())
        {
          current_epoch_users.erase(finder);
          refs_to_remove++;
        }
        finder = previous_epoch_users.find(*it);
        if (finder != previous_epoch_users.end())
        {
          previous_epoch_users.erase(finder);
          refs_to_remove++;
        }
#endif
        if ((*it)->remove_reference(refs_to_remove))
          delete (*it);
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::filter_current_users(
        const FieldMaskSet<PhysicalUser>& to_filter)
    //--------------------------------------------------------------------------
    {
      // Lock needs to be held by caller
      for (FieldMaskSet<PhysicalUser>::const_iterator it = to_filter.begin();
           it != to_filter.end(); it++)
      {
        unsigned refs_to_remove = 1;
        FieldMaskSet<PhysicalUser>::iterator finder =
            current_epoch_users.find(it->first);
        if (finder != current_epoch_users.end())
        {
          const FieldMask overlap = it->second & finder->second;
          if (!overlap)
          {
            finder.filter(overlap);
            if (!finder->second)
            {
              current_epoch_users.erase(finder);
              refs_to_remove++;
            }
            // Add this into the previous epoch users
            // and pass along a reference if necessary
            if (previous_epoch_users.insert(it->first, overlap))
              refs_to_remove--;
          }
        }
        if ((refs_to_remove > 0) && it->first->remove_reference(refs_to_remove))
          delete it->first;
      }
      current_epoch_users.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void ExprView::filter_previous_users(
        const FieldMaskSet<PhysicalUser>& to_filter)
    //--------------------------------------------------------------------------
    {
      // Lock needs to be held by caller
      for (FieldMaskSet<PhysicalUser>::const_iterator it = to_filter.begin();
           it != to_filter.end(); it++)
      {
        unsigned refs_to_remove = 1;
        FieldMaskSet<PhysicalUser>::iterator finder =
            previous_epoch_users.find(it->first);
        if (finder != previous_epoch_users.end())
        {
          finder.filter(it->second);
          if (!finder->second)
          {
            previous_epoch_users.erase(finder);
            refs_to_remove++;
          }
        }
        // Remove the reference we were holding
        if (it->first->remove_reference(refs_to_remove))
          delete it->first;
      }
      previous_epoch_users.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void ExprView::find_current_preconditions(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceExpression* user_expr, ApEvent term_event,
        const UniqueID op_id, const unsigned index, const bool user_covers,
        std::set<ApEvent>& preconditions, std::set<PhysicalUser*>& dead_users,
        FieldMaskSet<PhysicalUser>& filter_users, FieldMask& observed,
        FieldMask& non_dominated, const bool trace_recording,
        const bool copy_user)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      if (user_mask * current_epoch_users.get_valid_mask())
        return;
      for (FieldMaskSet<PhysicalUser>::const_iterator it =
               current_epoch_users.begin();
           it != current_epoch_users.end(); it++)
      {
        if (it->first->term_event == term_event)
          continue;
#ifndef LEGION_DISABLE_EVENT_PRUNING
        // We're about to do a bunch of expensive tests,
        // so first do something cheap to see if we can
        // skip all the tests.
        if (!trace_recording &&
            it->first->term_event.has_triggered_faultignorant())
        {
          if (dead_users.insert(it->first).second)
            it->first->add_reference();
          continue;
        }
#if 0
        // You might think you can optimize things like this, but you can't
        // because we still need the correct epoch users for every ExprView
        // when we go to add our user later
        if (!trace_recording &&
            preconditions.find(it->first->term_event) != preconditions.end())
          continue;
#endif
#endif
        const FieldMask overlap = user_mask & it->second;
        if (!overlap)
          continue;
        bool dominates = true;
        if (has_local_precondition(
                it->first, usage, user_expr, op_id, index, user_covers,
                copy_user, &dominates))
        {
          preconditions.insert(it->first->term_event);
          if (dominates)
          {
            observed |= overlap;
            if (filter_users.insert(it->first, overlap))
              it->first->add_reference();
          } else
            non_dominated |= overlap;
        } else
          non_dominated |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_previous_preconditions(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceExpression* user_expr, ApEvent term_event,
        const UniqueID op_id, const unsigned index, const bool user_covers,
        std::set<ApEvent>& preconditions, std::set<PhysicalUser*>& dead_users,
        const bool trace_recording, const bool copy_user)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      if (user_mask * previous_epoch_users.get_valid_mask())
        return;
      for (FieldMaskSet<PhysicalUser>::const_iterator it =
               previous_epoch_users.begin();
           it != previous_epoch_users.end(); it++)
      {
        if (it->first->term_event == term_event)
          continue;
#ifndef LEGION_DISABLE_EVENT_PRUNING
        // We're about to do a bunch of expensive tests,
        // so first do something cheap to see if we can
        // skip all the tests.
        if (!trace_recording &&
            it->first->term_event.has_triggered_faultignorant())
        {
          if (dead_users.insert(it->first).second)
            it->first->add_reference();
          continue;
        }
#if 0
        // You might think you can optimize things like this, but you can't
        // because we still need the correct epoch users for every ExprView
        // when we go to add our user later
        if (!trace_recording &&
            preconditions.find(it->first->term_event) != preconditions.end())
          continue;
#endif
#endif
        const FieldMask overlap = user_mask & it->second;
        if (!overlap)
          continue;
        if (has_local_precondition(
                it->first, usage, user_expr, op_id, index, user_covers,
                copy_user))
          preconditions.insert(it->first->term_event);
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_current_preconditions(
        const RegionUsage& usage, const FieldMask& mask,
        IndexSpaceExpression* expr, const bool expr_covers,
        std::set<ApEvent>& last_events, FieldMask& observed,
        FieldMask& non_dominated) const
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      if (mask * current_epoch_users.get_valid_mask())
        return;
      for (FieldMaskSet<PhysicalUser>::const_iterator it =
               current_epoch_users.begin();
           it != current_epoch_users.end(); it++)
      {
        const FieldMask overlap = mask & it->second;
        if (!overlap)
          continue;
        bool dominated = true;
        // We're just reading these and we want to see all prior
        // dependences so just give dummy opid and index
        if (has_local_precondition(
                it->first, usage, expr, 0 /*opid*/, 0 /*index*/, expr_covers,
                true /*copy user*/, &dominated))
        {
          last_events.insert(it->first->term_event);
          if (dominated)
            observed |= overlap;
          else
            non_dominated |= overlap;
        } else
          non_dominated |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_previous_preconditions(
        const RegionUsage& usage, const FieldMask& mask,
        IndexSpaceExpression* expr, const bool expr_covers,
        std::set<ApEvent>& last_users) const
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      if (mask * previous_epoch_users.get_valid_mask())
        return;
      for (FieldMaskSet<PhysicalUser>::const_iterator it =
               previous_epoch_users.begin();
           it != previous_epoch_users.end(); it++)
      {
        const FieldMask overlap = mask & it->second;
        if (!overlap)
          continue;
        // We're just reading these and we want to see all prior
        // dependences so just give dummy opid and index
        if (has_local_precondition(
                it->first, usage, expr, 0 /*opid*/, 0 /*index*/, expr_covers,
                true /*copy user*/))
          last_users.insert(it->first->term_event);
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_previous_filter_users(
        const FieldMask& dom_mask, FieldMaskSet<PhysicalUser>& filter_users)
    //--------------------------------------------------------------------------
    {
      // Lock better be held by caller
      if (dom_mask * previous_epoch_users.get_valid_mask())
        return;
      for (FieldMaskSet<PhysicalUser>::const_iterator it =
               previous_epoch_users.begin();
           it != previous_epoch_users.end(); it++)
      {
        const FieldMask overlap = dom_mask & it->second;
        if (!overlap)
          continue;
        if (filter_users.insert(it->first, overlap))
          it->first->add_reference();
      }
    }

    /////////////////////////////////////////////////////////////
    // IndividualView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualView::IndividualView(
        DistributedID did, PhysicalManager* man, AddressSpaceID log_owner,
        bool register_now, CollectiveMapping* mapping)
      : InstanceView(did, register_now, mapping), manager(man),
        logical_owner(log_owner),
        current_users(
            (log_owner == local_space) ?
                new ExprView(
                    this->did, manager->instance_domain,
                    manager->is_unbound()) :
                nullptr),
        expr_cache_uses(0), outstanding_additions(0)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != nullptr);
#endif
      // Keep the manager from being collected
      manager->add_nested_resource_ref(did);
      manager->add_nested_gc_ref(did);
      if (current_users != nullptr)
        current_users->add_reference();
    }

    //--------------------------------------------------------------------------
    IndividualView::~IndividualView(void)
    //--------------------------------------------------------------------------
    {
      if (is_logical_owner() && !view_reservations.empty())
      {
        std::set<ApEvent> done_events;
        current_users->find_all_done_events(done_events);
        const ApEvent all_done = Runtime::merge_events(nullptr, done_events);
        // No need for the lock here since we should be in a destructor
        // and there should be no more races
        for (std::map<unsigned, Reservation>::iterator it =
                 view_reservations.begin();
             it != view_reservations.end(); it++)
          it->second.destroy_reservation(all_done);
      }
      if ((current_users != nullptr) && current_users->remove_reference())
        delete current_users;
      if (manager->remove_nested_resource_ref(did))
        delete manager;
    }

    //--------------------------------------------------------------------------
    void IndividualView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      bool result =
#endif
          manager->acquire_instance(did);
      assert(result);
#else
      manager->add_nested_valid_ref(did);
#endif
      add_base_gc_ref(INTERNAL_VALID_REF);
    }

    //--------------------------------------------------------------------------
    bool IndividualView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      manager->remove_nested_valid_ref(did);
      return remove_base_gc_ref(INTERNAL_VALID_REF);
    }

    //--------------------------------------------------------------------------
    void IndividualView::notify_local(void)
    //--------------------------------------------------------------------------
    {
      manager->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void IndividualView::pack_valid_ref(void)
    //--------------------------------------------------------------------------
    {
      pack_global_ref();
      manager->pack_valid_ref();
    }

    //--------------------------------------------------------------------------
    void IndividualView::unpack_valid_ref(void)
    //--------------------------------------------------------------------------
    {
      manager->unpack_valid_ref();
      unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    AddressSpaceID IndividualView::get_analysis_space(
        PhysicalManager* instance) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance == manager);
#endif
      return logical_owner;
    }

    //--------------------------------------------------------------------------
    bool IndividualView::aliases(InstanceView* other) const
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_collective_view())
      {
        CollectiveView* collective = other->as_collective_view();
        return std::binary_search(
            collective->instances.begin(), collective->instances.end(),
            manager->did);
      } else
        return (this == other);
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::fill_from(
        FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* fill_expression, Operation* op,
        const unsigned index, const IndexSpaceID collective_match_space,
        const FieldMask& fill_mask, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        CopyAcrossHelper* across_helper, const bool manage_dst_events,
        const bool fill_restricted, const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((across_helper == nullptr) || !manage_dst_events);
#endif
      // Compute the precondition first
      if (manage_dst_events)
      {
        ApEvent dst_precondition = find_copy_preconditions(
            false /*reading*/, 0 /*redop*/, fill_mask, fill_expression,
            op->get_unique_op_id(), index, applied_events, trace_info);
        if (dst_precondition.exists())
        {
          if (precondition.exists())
            precondition = Runtime::merge_events(
                &trace_info, precondition, dst_precondition);
          else
            precondition = dst_precondition;
        }
      }
      std::vector<CopySrcDstField> dst_fields;
      if (across_helper != nullptr)
      {
        const FieldMask src_mask = across_helper->convert_dst_to_src(fill_mask);
        across_helper->compute_across_offsets(src_mask, dst_fields);
      } else
        manager->compute_copy_offsets(fill_mask, dst_fields);
      const ApEvent result = fill_view->issue_fill(
          op, fill_expression, this, fill_mask, trace_info, dst_fields,
          applied_events, manager, precondition, predicate_guard,
          COLLECTIVE_NONE, fill_restricted);
      // Save the result
      if (manage_dst_events && result.exists())
        add_copy_user(
            false /*reading*/, 0 /*redop*/, result, fill_mask, fill_expression,
            op->get_unique_op_id(), index, recorded_events,
            trace_info.recording, runtime->address_space);
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::copy_from(
        InstanceView* src_view, ApEvent precondition, PredEvent predicate_guard,
        ReductionOpID reduction_op_id, IndexSpaceExpression* copy_expression,
        Operation* op, const unsigned index,
        const IndexSpaceID collective_match_space, const FieldMask& copy_mask,
        PhysicalManager* src_point, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        CopyAcrossHelper* across_helper, const bool manage_dst_events,
        const bool copy_restricted, const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((across_helper == nullptr) || !manage_dst_events);
#endif
      // Compute the preconditions first
      const UniqueID op_id = op->get_unique_op_id();
      // We'll need to compute our destination precondition no matter what
      if (manage_dst_events)
      {
        const ApEvent dst_pre = find_copy_preconditions(
            false /*reading*/, reduction_op_id, copy_mask, copy_expression,
            op_id, index, applied_events, trace_info);
        if (dst_pre.exists())
        {
          if (precondition.exists())
            precondition =
                Runtime::merge_events(&trace_info, precondition, dst_pre);
          else
            precondition = dst_pre;
        }
      }
      const FieldMask* src_mask =
          (across_helper == nullptr) ?
              &copy_mask :
              new FieldMask(across_helper->convert_dst_to_src(copy_mask));
      // Several cases here:
      // 1. The source is another individual manager - just straight up
      //    compute the dependences and do the copy or reduction
      // 2. The source is a normal collective manager - issue a copy from
      //    an instance close to the destination instance
      // 3. The source is a reduction collective manager - build a reduction
      //    tree down to a source instance close to the destination instance
      ApEvent result;
      if (src_view->is_individual_view())
      {
        IndividualView* source_view = src_view->as_individual_view();
        // Case 1: Source manager is another instance manager
        const ApEvent src_pre = source_view->find_copy_preconditions(
            true /*reading*/, 0 /*redop*/, *src_mask, copy_expression, op_id,
            index, applied_events, trace_info);
        if (src_pre.exists())
        {
          if (precondition.exists())
            precondition =
                Runtime::merge_events(&trace_info, precondition, src_pre);
          else
            precondition = src_pre;
        }
        // Compute the field offsets
        std::vector<CopySrcDstField> dst_fields, src_fields;
        if (across_helper == nullptr)
          manager->compute_copy_offsets(copy_mask, dst_fields);
        else
          across_helper->compute_across_offsets(*src_mask, dst_fields);
        PhysicalManager* source_manager = source_view->get_manager();
        assert(manager->instance.id != source_manager->instance.id);
        source_manager->compute_copy_offsets(*src_mask, src_fields);
        std::vector<Reservation> reservations;
        // If we're doing a reduction operation then set the reduction
        // information on the source-dst fields
        if (reduction_op_id > 0)
        {
#ifdef DEBUG_LEGION
          assert((get_redop() == 0) || (get_redop() == reduction_op_id));
#endif
          // Get the reservations
          find_field_reservations(copy_mask, reservations);
          // Set the redop on the destination fields
          // Note that we can mark these as exclusive copies since
          // we are protecting them with the reservations
          for (unsigned idx = 0; idx < dst_fields.size(); idx++)
            dst_fields[idx].set_redop(
                reduction_op_id, (get_redop() > 0) /*fold*/,
                true /*exclusive*/);
        }
        result = copy_expression->issue_copy(
            op, trace_info, dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
            source_manager->tree_id, manager->tree_id,
#endif
            precondition, predicate_guard, source_manager->get_unique_event(),
            manager->get_unique_event(), COLLECTIVE_NONE, copy_restricted);
        if (result.exists())
        {
          source_view->add_copy_user(
              true /*reading*/, 0 /*redop*/, result, *src_mask, copy_expression,
              op_id, index, recorded_events, trace_info.recording,
              runtime->address_space);
          if (manage_dst_events)
            add_copy_user(
                false /*reading*/, reduction_op_id, result, copy_mask,
                copy_expression, op_id, index, recorded_events,
                trace_info.recording, runtime->address_space);
        }
        if (trace_info.recording)
        {
          const UniqueInst src_inst(source_view);
          const UniqueInst dst_inst(this);
          trace_info.record_copy_insts(
              result, copy_expression, src_inst, dst_inst, *src_mask, copy_mask,
              reduction_op_id, applied_events);
        }
      } else
      {
        CollectiveView* collective = src_view->as_collective_view();
        std::vector<CopySrcDstField> dst_fields;
        if (across_helper == nullptr)
          manager->compute_copy_offsets(copy_mask, dst_fields);
        else
          across_helper->compute_across_offsets(*src_mask, dst_fields);
        std::vector<Reservation> reservations;
        if (reduction_op_id > 0)
        {
#ifdef DEBUG_LEGION
          assert((get_redop() == 0) || (get_redop() == reduction_op_id));
#endif
          find_field_reservations(copy_mask, reservations);
          // Set the redop on the destination fields
          // Note that we can mark these as exclusive copies since
          // we are protecting them with the reservations
          for (unsigned idx = 0; idx < dst_fields.size(); idx++)
            dst_fields[idx].set_redop(
                reduction_op_id, (get_redop() > 0) /*fold*/,
                true /*exclusive*/);
        }
        if (collective->is_allreduce_view())
        {
#ifdef DEBUG_LEGION
          assert(reduction_op_id == collective->get_redop());
#endif
          AllreduceView* allreduce = collective->as_allreduce_view();
          // Case 3
          // This is subtle as fuck
          // In the normal case where we're doing a reduction from a
          // collective instance to a normal instance then we can get
          // away with just building the reduction tree.
          //
          // An important note here: we only need to build a reduction tree
          // and not do an all-reduce for the collective reduction instance
          // because we know the equivalence set code above will only ever
          // issue a single copy from a reduction instance into another
          // instance before that reduction instance is refreshed, so it
          // is safe to break the invariant that all instances in the
          // collective manager have the same data.
          //
          // However, in the case where we are doing a copy-across, then we
          // might still be asked to do an intra-region reduction later so
          // it's unsafe to do the partial accumulations into our own
          // instances. Therefore for now we will hammer all the source
          // instances into the destination instance without any
          // intermediate reductions.
          const UniqueInst dst_inst(this);
          if (manage_dst_events)
          {
            // Reduction-tree case
            const AddressSpaceID origin =
                (src_point != nullptr) ?
                    src_point->owner_space :
                    collective->select_source_space(owner_space);
            // There will always be a single result for this copy
            if (origin != local_space)
            {
              const RtUserEvent recorded = Runtime::create_rt_user_event();
              const RtUserEvent applied = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(allreduce->did);
                pack_fields(rez, dst_fields);
                rez.serialize<size_t>(reservations.size());
                for (unsigned idx = 0; idx < reservations.size(); idx++)
                  rez.serialize(reservations[idx]);
                rez.serialize(precondition);
                rez.serialize(predicate_guard);
                copy_expression->pack_expression(rez, origin);
                op->pack_remote_operation(rez, origin, applied_events);
                rez.serialize(index);
                rez.serialize(*src_mask);
                rez.serialize(copy_mask);
                if (src_point != nullptr)
                  rez.serialize(src_point->did);
                else
                  rez.serialize<DistributedID>(0);
                dst_inst.serialize(rez);
                rez.serialize(manager->get_unique_event());
                trace_info.pack_trace_info(rez);
                rez.serialize(recorded);
                rez.serialize(applied);
                if (trace_info.recording)
                {
                  ApBarrier bar;
                  ShardID sid =
                      trace_info.record_barrier_creation(bar, 1 /*arrivals*/);
                  rez.serialize(bar);
                  rez.serialize(sid);
                  result = bar;
                } else
                {
                  const ApUserEvent to_trigger =
                      Runtime::create_ap_user_event(&trace_info);
                  result = to_trigger;
                  rez.serialize(to_trigger);
                }
                rez.serialize(origin);
                rez.serialize(COLLECTIVE_REDUCTION);
              }
              runtime->send_collective_distribute_reduction(origin, rez);
              recorded_events.insert(recorded);
              applied_events.insert(applied);
            } else
            {
              const ApUserEvent to_trigger =
                  Runtime::create_ap_user_event(&trace_info);
              result = to_trigger;
              allreduce->perform_collective_reduction(
                  dst_fields, reservations, precondition, predicate_guard,
                  copy_expression, op, index, *src_mask, copy_mask,
                  (src_point != nullptr) ? src_point->did : 0, dst_inst,
                  manager->get_unique_event(), trace_info, COLLECTIVE_REDUCTION,
                  recorded_events, applied_events, to_trigger, origin);
            }
          } else
          {
            // Hammer reduction case
            // Issue a performance warning if we're ever going to
            // be doing this case and the number of instance is large
            if (collective->instances.size() > LEGION_COLLECTIVE_RADIX)
              REPORT_LEGION_WARNING(
                  LEGION_WARNING_COLLECTIVE_HAMMER_REDUCTION,
                  "WARNING: Performing copy-across reduction hammer with %zd "
                  "instances into a single instance from collective manager "
                  "%llx to normal manager %llx. Please report this use case "
                  "to the Legion developers' mailing list.",
                  collective->instances.size(), collective->did, did)
            const AddressSpaceID origin =
                collective->select_source_space(owner_space);
            if (origin != local_space)
            {
              const RtUserEvent recorded = Runtime::create_rt_user_event();
              const RtUserEvent applied = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(allreduce->did);
                pack_fields(rez, dst_fields);
                rez.serialize<size_t>(reservations.size());
                for (unsigned idx = 0; idx < reservations.size(); idx++)
                  rez.serialize(reservations[idx]);
                rez.serialize(precondition);
                rez.serialize(predicate_guard);
                copy_expression->pack_expression(rez, origin);
                op->pack_remote_operation(rez, origin, applied_events);
                rez.serialize(index);
                rez.serialize(*src_mask);
                rez.serialize(copy_mask);
                dst_inst.serialize(rez);
                rez.serialize(manager->get_unique_event());
                trace_info.pack_trace_info(rez);
                rez.serialize(recorded);
                rez.serialize(applied);
                if (trace_info.recording)
                {
                  ApBarrier bar;
                  ShardID sid =
                      trace_info.record_barrier_creation(bar, 1 /*arrivals*/);
                  rez.serialize(bar);
                  rez.serialize(sid);
                  result = bar;
                } else
                {
                  const ApUserEvent to_trigger =
                      Runtime::create_ap_user_event(&trace_info);
                  rez.serialize(to_trigger);
                  result = to_trigger;
                }
                rez.serialize(origin);
              }
              runtime->send_collective_hammer_reduction(origin, rez);
              recorded_events.insert(recorded);
              applied_events.insert(applied);
            } else
              result = allreduce->perform_hammer_reduction(
                  dst_fields, reservations, precondition, predicate_guard,
                  copy_expression, op, index, *src_mask, copy_mask, dst_inst,
                  manager->get_unique_event(), trace_info, recorded_events,
                  applied_events, origin);
          }
        } else
        {
          // Case 2
          // We can issue the copy from an instance in the source
          const Memory location = manager->memory_manager->memory;
          const DomainPoint no_point;
          const AddressSpaceID origin =
              (src_point != nullptr) ?
                  src_point->owner_space :
                  collective->select_source_space(owner_space);
          const UniqueInst dst_inst(this);
          if (origin != local_space)
          {
            const RtUserEvent recorded = Runtime::create_rt_user_event();
            const RtUserEvent applied = Runtime::create_rt_user_event();
            ApUserEvent to_trigger = Runtime::create_ap_user_event(&trace_info);
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(collective->did);
              pack_fields(rez, dst_fields);
              rez.serialize<size_t>(reservations.size());
              for (unsigned idx = 0; idx < reservations.size(); idx++)
                rez.serialize(reservations[idx]);
              rez.serialize(precondition);
              rez.serialize(predicate_guard);
              copy_expression->pack_expression(rez, origin);
              op->pack_remote_operation(rez, origin, applied_events);
              rez.serialize(index);
              rez.serialize(*src_mask);
              rez.serialize(copy_mask);
              rez.serialize(location);
              dst_inst.serialize(rez);
              rez.serialize(manager->get_unique_event());
              if (src_point != nullptr)
                rez.serialize(src_point->did);
              else
                rez.serialize<DistributedID>(0);
              trace_info.pack_trace_info(rez);
              rez.serialize(COLLECTIVE_NONE);
              rez.serialize(recorded);
              rez.serialize(applied);
              rez.serialize(to_trigger);
            }
            runtime->send_collective_distribute_point(origin, rez);
            recorded_events.insert(recorded);
            applied_events.insert(applied);
            result = to_trigger;
          } else
            result = collective->perform_collective_point(
                dst_fields, reservations, precondition, predicate_guard,
                copy_expression, op, index, *src_mask, copy_mask, location,
                dst_inst, manager->get_unique_event(),
                (src_point != nullptr) ? src_point->did : 0, trace_info,
                recorded_events, applied_events);
        }
        if (result.exists() && manage_dst_events)
          add_copy_user(
              false /*reading*/, reduction_op_id, result, copy_mask,
              copy_expression, op_id, index, recorded_events,
              trace_info.recording, runtime->address_space);
      }
      if (across_helper != nullptr)
        delete src_mask;
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualView::add_initial_user(
        ApEvent term_event, const RegionUsage& usage,
        const FieldMask& user_mask, IndexSpaceExpression* user_expr,
        const UniqueID op_id, const unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      add_internal_task_user(
          usage, user_expr, user_mask, term_event, op_id, index);
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::register_user(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceNode* user_expr, const UniqueID op_id,
        const size_t op_ctx_index, const unsigned index,
        const IndexSpaceID match_space, ApEvent term_event,
        PhysicalManager* target, CollectiveMapping* analysis_mapping,
        size_t local_collective_arrivals, std::vector<RtEvent>& registered,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info,
        const AddressSpaceID source, const bool symbolic /*=false*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target == manager);
#endif
      // Handle the collective rendezvous if necessary
      if (local_collective_arrivals > 0)
        return register_collective_user(
            usage, user_mask, user_expr, op_id, op_ctx_index, index,
            match_space, term_event, target, analysis_mapping,
            local_collective_arrivals, registered, applied_events, trace_info,
            symbolic);
      // Quick test for empty index space expressions
      if (!symbolic && user_expr->is_empty())
      {
        manager->record_instance_user(term_event, applied_events);
        return manager->get_use_event(term_event);
      }
      if (!is_logical_owner())
      {
        ApUserEvent ready_event;
        // Check to see if this user came from somewhere that wasn't
        // the logical owner, if so we need to send the update back
        // to the owner to be handled
        if (source != logical_owner)
        {
          // If we're not the logical owner send a message there
          // to do the analysis and provide a user event to trigger
          // with the precondition
          ready_event = Runtime::create_ap_user_event(&trace_info);
          RtUserEvent registered_event = Runtime::create_rt_user_event();
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(target->did);
            rez.serialize(usage);
            rez.serialize(user_mask);
            rez.serialize(user_expr->handle);
            rez.serialize(op_id);
            rez.serialize(op_ctx_index);
            rez.serialize(index);
            rez.serialize(match_space);
            rez.serialize(term_event);
            rez.serialize(local_collective_arrivals);
            rez.serialize(ready_event);
            rez.serialize(registered_event);
            rez.serialize(applied_event);
            trace_info.pack_trace_info(rez);
          }
          runtime->send_view_register_user(logical_owner, rez);
          registered.push_back(registered_event);
          applied_events.insert(applied_event);
        }
        return ready_event;
      } else
      {
        // Now we can do our local analysis
        std::set<ApEvent> wait_on_events;
        ApEvent start_use_event = manager->get_use_event(term_event);
        if (start_use_event.exists())
          wait_on_events.insert(start_use_event);
        // Find the preconditions
        const bool user_dominates =
            (user_expr->expr_id == current_users->view_expr->expr_id) ||
            (user_expr->get_volume() == current_users->get_view_volume());
        {
          // Traversing the tree so need the expr_view lock
          AutoLock e_lock(expr_lock, 1, false /*exclusive*/);
          current_users->find_user_preconditions(
              usage, user_expr, user_dominates, user_mask, term_event, op_id,
              index, wait_on_events, trace_info.recording);
        }
        // Add our local user
        add_internal_task_user(
            usage, user_expr, user_mask, term_event, op_id, index);
        manager->record_instance_user(term_event, applied_events);
        // At this point tasks shouldn't be allowed to wait on themselves
#ifdef DEBUG_LEGION
        if (term_event.exists())
          assert(wait_on_events.find(term_event) == wait_on_events.end());
#endif
        // Return the merge of the events
        if (!wait_on_events.empty())
          return Runtime::merge_events(&trace_info, wait_on_events);
        else
          return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::find_copy_preconditions(
        bool reading, ReductionOpID redop, const FieldMask& copy_mask,
        IndexSpaceExpression* copy_expr, UniqueID op_id, unsigned index,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
        // Check to see if there are any replicated fields here which we
        // can handle locally so we don't have to send a message to the owner
        ApEvent result_event;
        FieldMask request_mask(copy_mask);
        {
          // All the fields are not local, first send the request to
          // the owner to do the analysis since we're going to need
          // to do that anyway, then issue any request for replicated
          // fields to be moved to this node and record it as a
          // precondition for the mapping
          ApUserEvent ready_event = Runtime::create_ap_user_event(&trace_info);
          RtUserEvent applied = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<bool>(reading);
            rez.serialize(redop);
            rez.serialize(copy_mask);
            copy_expr->pack_expression(rez, logical_owner);
            rez.serialize(op_id);
            rez.serialize(index);
            rez.serialize(ready_event);
            rez.serialize(applied);
            trace_info.pack_trace_info(rez);
          }
          runtime->send_view_find_copy_preconditions_request(
              logical_owner, rez);
          applied_events.insert(applied);
          result_event = ready_event;
        }
        return result_event;
      } else
      {
        // In the case where we're the owner we can just handle
        // this without needing to do anything
        std::set<ApEvent> preconditions;
        const ApEvent start_use_event = manager->get_use_event();
        if (start_use_event.exists())
          preconditions.insert(start_use_event);
        const RegionUsage usage(
            reading     ? LEGION_READ_ONLY :
            (redop > 0) ? LEGION_REDUCE :
                          LEGION_READ_WRITE,
            LEGION_EXCLUSIVE, redop);
        const bool copy_dominates =
            (copy_expr->expr_id == current_users->view_expr->expr_id) ||
            (copy_expr->get_volume() == current_users->get_view_volume());
        {
          // Need a read-only copy of the expr_lock to traverse the tree
          AutoLock e_lock(expr_lock, 1, false /*exclusive*/);
          current_users->find_copy_preconditions(
              usage, copy_expr, copy_dominates, copy_mask, op_id, index,
              preconditions, trace_info.recording);
        }
        if (preconditions.empty())
          return ApEvent::NO_AP_EVENT;
        return Runtime::merge_events(&trace_info, preconditions);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::add_copy_user(
        bool reading, ReductionOpID redop, ApEvent term_event,
        const FieldMask& copy_mask, IndexSpaceExpression* copy_expr,
        UniqueID op_id, unsigned index, std::set<RtEvent>& applied_events,
        const bool trace_recording, const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
        // Check to see if this update came from some place other than the
        // source in which case we need to send it back to the source
        if (source != logical_owner)
        {
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<bool>(reading);
            rez.serialize(redop);
            rez.serialize(term_event);
            rez.serialize(copy_mask);
            copy_expr->pack_expression(rez, logical_owner);
            rez.serialize(op_id);
            rez.serialize(index);
            rez.serialize(applied_event);
            rez.serialize<bool>(trace_recording);
          }
          runtime->send_view_add_copy_user(logical_owner, rez);
          applied_events.insert(applied_event);
        }
      } else
      {
        // Now we can do our local analysis
        const RegionUsage usage(
            reading     ? LEGION_READ_ONLY :
            (redop > 0) ? LEGION_REDUCE :
                          LEGION_READ_WRITE,
            (redop > 0) ? LEGION_ATOMIC : LEGION_EXCLUSIVE, redop);
        add_internal_copy_user(
            usage, copy_expr, copy_mask, term_event, op_id, index);
        manager->record_instance_user(term_event, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::find_last_users(
        PhysicalManager* instance, std::set<ApEvent>& events,
        const RegionUsage& usage, const FieldMask& mask,
        IndexSpaceExpression* expr, std::vector<RtEvent>& ready_events) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance == manager);
#endif
      // Check to see if we're on the right node to perform this analysis
      if (logical_owner != local_space)
      {
        const RtUserEvent ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(manager->did);
          rez.serialize(&events);
          rez.serialize(usage);
          rez.serialize(mask);
          expr->pack_expression(rez, logical_owner);
          rez.serialize(ready);
        }
        runtime->send_view_find_last_users_request(logical_owner, rez);
        ready_events.push_back(ready);
      } else
      {
        const bool expr_dominates =
            (expr->expr_id == current_users->view_expr->expr_id) ||
            (expr->get_volume() == current_users->get_view_volume());
        {
          // Need a read-only copy of the expr_lock to traverse the tree
          AutoLock e_lock(expr_lock, 1, false /*exclusive*/);
          current_users->find_last_users(
              usage, expr, expr_dominates, mask, events);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::add_internal_task_user(
        const RegionUsage& usage, IndexSpaceExpression* user_expr,
        const FieldMask& user_mask, ApEvent term_event, UniqueID op_id,
        const unsigned index)
    //--------------------------------------------------------------------------
    {
      // Convert to the canonical expression if it is not the root expr
      if (user_expr != current_users->view_expr)
      {
        // Handle the dumb case of output region expr views
        // Since we cannot make the root expr view have a canonical
        // expression then we need to check to make sure we're just not
        // finding it because it is congruent to the output region
        const size_t user_volume = user_expr->get_volume();
        const size_t root_volume = current_users->view_expr->get_volume();
#ifdef DEBUG_LEGION
        assert(user_volume <= root_volume);
#endif
        if (user_volume < root_volume)
          user_expr = user_expr->get_canonical_expression();
        else
          user_expr = current_users->view_expr;
      }
      PhysicalUser* user = new PhysicalUser(
          usage, user_expr, term_event, op_id, index, false /*copy user*/,
          true /*covers*/);
      // Hold a reference to this in case it finishes before we're done
      // with the analysis and its get pruned/deleted
      user->add_reference();
      ExprView* target_view = nullptr;
      bool has_target_view = false;
      // Handle an easy case first, if the user_expr is the same as the
      // view_expr for the root then this is easy
      bool update_count = true;
      if (user_expr == current_users->view_expr)
      {
        // This is just going to add at the top so never needs to wait
        target_view = current_users;
        update_count = false;
        has_target_view = true;
      } else if (expr_cache_uses.fetch_add(1) < USER_CACHE_TIMEOUT)
      {
        // Hard case where we will have subviews
        AutoLock e_lock(expr_lock, 1, false /*exclusive*/);
        // See if we can find the entry in the cache and it's valid
        // for all of our fields
        LegionMap<IndexSpaceExprID, ExprView*>::const_iterator finder =
            expr_cache.find(user_expr->expr_id);
        if (finder != expr_cache.end())
        {
          target_view = finder->second;
          if (finder->second->invalid_fields * user_mask)
            has_target_view = true;
        }
        // increment the number of outstanding additions
        outstanding_additions.fetch_add(1);
      } else
      {
        // This is the path where we clean the cache, multiple threads
        // can race to get here
        AutoLock e_lock(expr_lock);
        // Block waiting for the prior additions to drain
        while (USER_CACHE_TIMEOUT <= expr_cache_uses.load())
        {
          // Wait for the prior outstanding additions to drain
          if (outstanding_additions.load() > 0)
          {
            if (!clean_waiting.exists())
              clean_waiting = Runtime::create_rt_user_event();
            const RtEvent wait_on = clean_waiting;
            e_lock.release();
            wait_on.wait();
            e_lock.reacquire();
          } else  // We won the race to wake up and clean the cache
            clean_cache();
        }
        // Now we can do the normal lookup
        // Have the lock in exclusive mode so can make nodes if needed
        LegionMap<IndexSpaceExprID, ExprView*>::const_iterator finder =
            expr_cache.find(user_expr->expr_id);
        if (finder == expr_cache.end())
        {
          target_view = new ExprView(this->did, user_expr);
          expr_cache[user_expr->expr_id] = target_view;
        } else
          target_view = finder->second;
        if (target_view != current_users)
        {
          // Now see if we need to insert it
          FieldMask insert_mask = user_mask & target_view->invalid_fields;
          if (!!insert_mask)
          {
            // Remove these fields from being invalid before we
            // destroy the insert mask
            target_view->invalid_fields -= insert_mask;
            // Do the insertion into the tree
            current_users->insert_subview(target_view, insert_mask);
          }
        }
        has_target_view = true;
        // increment the number of outstanding additions
        outstanding_additions.fetch_add(1);
      }
      if (!has_target_view)
      {
        // This could change the shape of the view tree so we need
        // exclusive privileges on the expr lock to serialize it
        // with everything else traversing the tree
        AutoLock e_lock(expr_lock);
        // If we don't have a target view see if there is a
        // congruent one already in the tree
        if (target_view == nullptr)
        {
          // Check to see if someone else made it when we released the lock
          LegionMap<IndexSpaceExprID, ExprView*>::const_iterator finder =
              expr_cache.find(user_expr->expr_id);
          if (finder == expr_cache.end())
          {
            target_view = new ExprView(this->did, user_expr);
            expr_cache[user_expr->expr_id] = target_view;
          } else
            target_view = finder->second;
        }
        if (target_view != current_users)
        {
          // Now see if we need to insert it
          FieldMask insert_mask = user_mask & target_view->invalid_fields;
          if (!!insert_mask)
          {
            // Remove these fields from being invalid before we
            // destroy the insert mask
            target_view->invalid_fields -= insert_mask;
            // Do the insertion into the tree
            current_users->insert_subview(target_view, insert_mask);
          }
        }
      }
      // Now we know the target view and it's valid for all fields
      // so we can add it to the expr view
      target_view->add_current_user(user, user_mask);
      if (user->remove_reference())
        delete user;
      if (update_count && (outstanding_additions.fetch_sub(1) == 1) &&
          (USER_CACHE_TIMEOUT <= expr_cache_uses.load()))
      {
        AutoLock e_lock(expr_lock);
        if (clean_waiting.exists())
        {
          // Wake up the clean waiter
          Runtime::trigger_event(clean_waiting);
          clean_waiting = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::add_internal_copy_user(
        const RegionUsage& usage, IndexSpaceExpression* user_expr,
        const FieldMask& user_mask, ApEvent term_event, UniqueID op_id,
        const unsigned index)
    //--------------------------------------------------------------------------
    {
      // Convert to the canonical expression if it is not the root
      if (user_expr != current_users->view_expr)
      {
        // Handle the dumb case of output region expr views
        // Since we cannot make the root expr view have a canonical
        // expression then we need to check to make sure we're just not
        // finding it because it is congruent to the output region
        const size_t user_volume = user_expr->get_volume();
        const size_t root_volume = current_users->view_expr->get_volume();
#ifdef DEBUG_LEGION
        assert(user_volume <= root_volume);
#endif
        if (user_volume < root_volume)
          user_expr = user_expr->get_canonical_expression();
        else
          user_expr = current_users->view_expr;
      }
      // First we're going to check to see if we can add this directly to
      // an existing ExprView with the same expresssion in which case
      // we'll be able to mark this user as being precise
      ExprView* target_view = nullptr;
      bool has_target_view = false;
      bool skip_check = false;
      // Handle an easy case first, if the user_expr is the same as the
      // view_expr for the root then this is easy
      bool update_count = true;
      if (user_expr == current_users->view_expr)
      {
        // This is just going to add at the top so never needs to wait
        target_view = current_users;
        update_count = false;
        has_target_view = true;
      } else if (expr_cache_uses.fetch_add(1) < USER_CACHE_TIMEOUT)
      {
        // Hard case where we will have subviews
        AutoLock e_lock(expr_lock, 1, false /*exclusive*/);
        // See if we can find the entry in the cache and it's valid
        // for all of our fields
        LegionMap<IndexSpaceExprID, ExprView*>::const_iterator finder =
            expr_cache.find(user_expr->expr_id);
        if (finder != expr_cache.end())
        {
          target_view = finder->second;
          if (finder->second->invalid_fields * user_mask)
            has_target_view = true;
        }
        // increment the number of outstanding additions
        outstanding_additions.fetch_add(1);
      } else
      {
        // This is the path where we clean the cache, multiple threads
        // can race to get here
        AutoLock e_lock(expr_lock);
        // Block waiting for the prior additions to drain
        while (USER_CACHE_TIMEOUT <= expr_cache_uses.load())
        {
          // Wait for the prior outstanding additions to drain
          if (outstanding_additions.load() > 0)
          {
            if (!clean_waiting.exists())
              clean_waiting = Runtime::create_rt_user_event();
            const RtEvent wait_on = clean_waiting;
            e_lock.release();
            wait_on.wait();
            e_lock.reacquire();
          } else  // We won the race to wake up and clean the cache
            clean_cache();
        }
        // Now we can do the normal lookup
        LegionMap<IndexSpaceExprID, ExprView*>::const_iterator finder =
            expr_cache.find(user_expr->expr_id);
        if (finder != expr_cache.end())
        {
          target_view = finder->second;
          // No need to insert this if it's the root
          if (target_view != current_users)
          {
            FieldMask insert_mask = target_view->invalid_fields & user_mask;
            if (!!insert_mask)
            {
              target_view->invalid_fields -= insert_mask;
              current_users->insert_subview(target_view, insert_mask);
            }
          }
          has_target_view = true;
        } else
          skip_check = true;  // No point in checking again
        // increment the number of outstanding additions
        outstanding_additions.fetch_add(1);
      }
      if (!has_target_view && !skip_check)
      {
        // This could change the shape of the view tree so we need
        // exclusive privileges on the expr lock to serialize it
        // with everything else traversing the tree
        AutoLock e_lock(expr_lock);
        // If we don't have a target view see if there is a
        // congruent one already in the tree
        if (target_view == nullptr)
        {
          // Check to see if someone else made it when we released the lock
          LegionMap<IndexSpaceExprID, ExprView*>::const_iterator finder =
              expr_cache.find(user_expr->expr_id);
          if (finder != expr_cache.end())
            target_view = finder->second;
        }
        // Don't make it though if we don't already have it
        if (target_view != nullptr)
        {
          // No need to insert this if it's the root
          if (target_view != current_users)
          {
            FieldMask insert_mask = target_view->invalid_fields & user_mask;
            if (!!insert_mask)
            {
              target_view->invalid_fields -= insert_mask;
              current_users->insert_subview(target_view, insert_mask);
            }
          }
          has_target_view = true;
        }
      }
      if (has_target_view)
      {
        // If we have a target view, then we know we cover it because
        // the expressions match directly
        PhysicalUser* user = new PhysicalUser(
            usage, user_expr, term_event, op_id, index, true /*copy user*/,
            true /*covers*/);
        // Hold a reference to this in case it finishes before we're done
        // with the analysis and its get pruned/deleted
        user->add_reference();
        // We already know the view so we can just add the user directly
        // there and then do any updates that we need to
        target_view->add_current_user(user, user_mask);
        if (user->remove_reference())
          delete user;
      } else
      {
        // We're traversing the view tree but not modifying it so
        // we need a read-only copy of the expr_lock
        AutoLock e_lock(expr_lock, 1, false /*exclusive*/);
        PhysicalUser* covered_user = nullptr;
        PhysicalUser* uncovered_user = nullptr;
        current_users->add_partial_user(
            usage, op_id, index, user_mask, term_event, user_expr,
            user_expr->get_volume(), covered_user, uncovered_user);
        // Remove the reference that was added when this was made
        if ((covered_user != nullptr) && covered_user->remove_reference())
          delete covered_user;
        if ((uncovered_user != nullptr) && uncovered_user->remove_reference())
          delete uncovered_user;
      }
      if (update_count && (outstanding_additions.fetch_sub(1) == 1) &&
          (USER_CACHE_TIMEOUT <= expr_cache_uses.load()))
      {
        AutoLock e_lock(expr_lock);
        if (clean_waiting.exists())
        {
          // Wake up the clean waiter
          Runtime::trigger_event(clean_waiting);
          clean_waiting = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::clean_cache(void)
    //--------------------------------------------------------------------------
    {
      // Clear the cache
      expr_cache.clear();
      // Reset the cache use counter
      expr_cache_uses.store(0);
      // Anytime we clean the cache, we also traverse the
      // view tree and see if there are any views we can
      // remove because they no longer have live users
      FieldMask dummy_mask;
      FieldMaskSet<ExprView> clean_set;
      current_users->clean_views(dummy_mask, clean_set);
      // We can safely repopulate the cache with any view expressions which
      // are still valid, remove all references for views in the clean set
      for (FieldMaskSet<ExprView>::const_iterator it = clean_set.begin();
           it != clean_set.end(); it++)
      {
        if (!!(~(it->first->invalid_fields)))
          expr_cache[it->first->view_expr->expr_id] = it->first;
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::register_collective_analysis(
        const CollectiveView* source, CollectiveAnalysis* analysis)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      const RendezvousKey key(
          analysis->get_context_index(), analysis->get_requirement_index(),
          analysis->get_match_space());
      RtUserEvent to_trigger;
      {
        AutoLock v_lock(view_lock);
        std::map<RendezvousKey, RegisteredAnalysis>::iterator finder =
            collective_analyses.find(key);
        if (finder != collective_analyses.end())
        {
          // Note that this will deduplicate multiple registrations
          if (finder->second.analysis == nullptr)
          {
#ifdef DEBUG_LEGION
            assert(finder->second.ready.exists());
#endif
            analysis->add_analysis_reference();
            finder->second.analysis = analysis;
            to_trigger = finder->second.ready;
#ifdef DEBUG_LEGION
            finder->second.ready = RtUserEvent::NO_RT_USER_EVENT;
#endif
          }
          // Record the view so we know how many registrations to expect
          // We'll see one unregister for each source view
          finder->second.views.insert(source->did);
        } else
        {
          analysis->add_analysis_reference();
          RegisteredAnalysis& registration = collective_analyses[key];
          registration.analysis = analysis;
          registration.views.insert(source->did);
        }
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    CollectiveAnalysis* IndividualView::find_collective_analysis(
        size_t context_index, unsigned region_index, IndexSpaceID match_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      RtEvent wait_on;
      const RendezvousKey key(context_index, region_index, match_space);
      {
        AutoLock v_lock(view_lock);
        std::map<RendezvousKey, RegisteredAnalysis>::iterator finder =
            collective_analyses.find(key);
        if (finder != collective_analyses.end())
        {
          if (finder->second.analysis == nullptr)
          {
#ifdef DEBUG_LEGION
            assert(finder->second.ready.exists());
#endif
            wait_on = finder->second.ready;
          } else
            return finder->second.analysis;
        } else
        {
          RegisteredAnalysis& registration = collective_analyses[key];
          registration.analysis = nullptr;
          registration.ready = Runtime::create_rt_user_event();
          wait_on = registration.ready;
        }
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
      AutoLock v_lock(view_lock);
      std::map<RendezvousKey, RegisteredAnalysis>::iterator finder =
          collective_analyses.find(key);
#ifdef DEBUG_LEGION
      assert(finder != collective_analyses.end());
      assert(finder->second.analysis != nullptr);
#endif
      return finder->second.analysis;
    }

    //--------------------------------------------------------------------------
    void IndividualView::unregister_collective_analysis(
        const CollectiveView* source, size_t context_index,
        unsigned region_index, IndexSpaceID match_space)
    //--------------------------------------------------------------------------
    {
      CollectiveAnalysis* removed = nullptr;
      const RendezvousKey key(context_index, region_index, match_space);
      {
        AutoLock v_lock(view_lock);
        std::map<RendezvousKey, RegisteredAnalysis>::iterator finder =
            collective_analyses.find(key);
        // Not all collective user registrations record an analysis with
        // the instance for performing copies. Specifically this is the
        // case with OverwriteAnalysis for attach operations, and
        // FilterAnalysis in the case where there is no flush
        if (finder == collective_analyses.end())
          return;
#ifdef DEBUG_LEGION
        assert(finder->second.analysis != nullptr);
        assert(!finder->second.ready.exists());
#endif
        std::set<DistributedID>::iterator view_finder =
            finder->second.views.find(source->did);
#ifdef DEBUG_LEGION
        assert(view_finder != finder->second.views.end());
#endif
        finder->second.views.erase(view_finder);
        // If we're not the last view that needs to unregister this
        // analysis then we don't remove it yet
        if (!finder->second.views.empty())
          return;
        removed = finder->second.analysis;
        collective_analyses.erase(finder);
      }
      if (removed->remove_analysis_reference())
        delete removed;
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::register_collective_user(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
        const unsigned index, const IndexSpaceID match_space,
        ApEvent term_event, PhysicalManager* target,
        CollectiveMapping* analysis_mapping, size_t local_collective_arrivals,
        std::vector<RtEvent>& registered_events,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info,
        const bool symbolic)
    //--------------------------------------------------------------------------
    {
      // This case occurs when all the points mapping to the same logical
      // region also map to the same physical instance. Most commonly this
      // will occur with control replication doing attach operations on
      // file instances, but can occur outside of control replication as
      // well, especially in intra-node cases
#ifdef DEBUG_LEGION
      assert(local_collective_arrivals > 0);
      assert((analysis_mapping != nullptr) || (local_collective_arrivals > 1));
#endif
      // First we need to decide which node is going to be the owner node
      // We'll prefer it to be the logical view owner since that is where
      // the event will be produced, otherwise, we'll just pick whichever
      // is closest to the logical view node
      const AddressSpaceID origin =
          (analysis_mapping == nullptr) ?
              local_space :
          analysis_mapping->contains(logical_owner) ?
              logical_owner :
              analysis_mapping->find_nearest(logical_owner);
      ApUserEvent result;
      RtUserEvent applied, registered;
      std::vector<ApEvent> term_events;
      PhysicalTraceInfo* result_info = nullptr;
      const RendezvousKey key(op_ctx_index, index, match_space);
      {
        AutoLock v_lock(view_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey, UserRendezvous>::iterator finder =
            rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder =
              rendezvous_users.insert(std::make_pair(key, UserRendezvous()))
                  .first;
          UserRendezvous& rendezvous = finder->second;
          rendezvous.remaining_local_arrivals = local_collective_arrivals;
          rendezvous.local_initialized = true;
          rendezvous.remaining_remote_arrivals =
              (analysis_mapping == nullptr) ?
                  0 :
                  analysis_mapping->count_children(origin, local_space);
          rendezvous.ready_event = Runtime::create_ap_user_event(&trace_info);
          rendezvous.trace_info = new PhysicalTraceInfo(trace_info);
          if (analysis_mapping != nullptr)
          {
            rendezvous.analysis_mapping = analysis_mapping;
            analysis_mapping->add_reference();
          }
          rendezvous.expr = expr;
          expr->add_nested_expression_reference(did);
          rendezvous.registered = Runtime::create_rt_user_event();
          rendezvous.applied = Runtime::create_rt_user_event();
        } else if (!finder->second.local_initialized)
        {
#ifdef DEBUG_LEGION
          assert(!finder->second.ready_event.exists());
          assert(finder->second.trace_info == nullptr);
          assert(finder->second.analysis_mapping != nullptr);
#endif
          // First local arrival
          finder->second.remaining_local_arrivals = local_collective_arrivals;
          finder->second.local_initialized = true;
          finder->second.ready_event =
              Runtime::create_ap_user_event(&trace_info);
          finder->second.trace_info = new PhysicalTraceInfo(trace_info);
          finder->second.expr = expr;
          expr->add_nested_expression_reference(did);
          if (!finder->second.remote_ready_events.empty())
          {
            for (std::map<ApUserEvent, PhysicalTraceInfo*>::const_iterator it =
                     finder->second.remote_ready_events.begin();
                 it != finder->second.remote_ready_events.end(); it++)
            {
              Runtime::trigger_event(
                  it->first, finder->second.ready_event, *it->second,
                  applied_events);
              delete it->second;
            }
            finder->second.remote_ready_events.clear();
          }
        }
        result = finder->second.ready_event;
        result_info = finder->second.trace_info;
        analysis_mapping = finder->second.analysis_mapping;
        registered = finder->second.registered;
        registered_events.push_back(registered);
        applied = finder->second.applied;
        applied_events.insert(applied);
        if (term_event.exists())
          finder->second.term_events.push_back(term_event);
#ifdef DEBUG_LEGION
        assert(finder->second.local_initialized);
        assert(finder->second.remaining_local_arrivals > 0);
#endif
        // If we're still expecting arrivals then nothing to do yet
        if ((--finder->second.remaining_local_arrivals > 0) ||
            (finder->second.remaining_remote_arrivals > 0))
        {
          // We need to save the trace info no matter what
          if (finder->second.mask == nullptr)
          {
            if (local_space == origin)
            {
              // Save our state for performing the registration later
              finder->second.usage = usage;
              finder->second.mask = new FieldMask(user_mask);
              finder->second.op_id = op_id;
              finder->second.symbolic = symbolic;
            }
          }
          return result;
        }
        term_events.swap(finder->second.term_events);
        expr = finder->second.expr;
#ifdef DEBUG_LEGION
        assert(finder->second.remote_ready_events.empty());
#endif
        // We're done with our entry after this so no need to keep it
        rendezvous_users.erase(finder);
      }
      if (!term_events.empty())
        term_event = Runtime::merge_events(&trace_info, term_events);
      if (local_space != origin)
      {
#ifdef DEBUG_LEGION
        assert(analysis_mapping != nullptr);
#endif
        const AddressSpaceID parent =
            analysis_mapping->get_parent(origin, local_space);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(op_ctx_index);
          rez.serialize(index);
          rez.serialize(match_space);
          rez.serialize(origin);
          result_info->pack_trace_info(rez);
          analysis_mapping->pack(rez);
          rez.serialize(term_event);
          rez.serialize(result);
          rez.serialize(registered);
          rez.serialize(applied);
        }
        runtime->send_collective_individual_register_user(parent, rez);
      } else
      {
        std::vector<RtEvent> local_registered;
        std::set<RtEvent> local_applied;
        const ApEvent ready = register_user(
            usage, user_mask, expr, op_id, op_ctx_index, index, match_space,
            term_event, target, nullptr /*analysis*/,
            0 /*no collective arrivals*/, local_registered, local_applied,
            *result_info, runtime->address_space, symbolic);
        Runtime::trigger_event(result, ready, *result_info, local_applied);
        if (!local_registered.empty())
          Runtime::trigger_event(
              registered, Runtime::merge_events(local_registered));
        else
          Runtime::trigger_event(registered);
        if (!local_applied.empty())
          Runtime::trigger_event(applied, Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(applied);
      }
      if (expr->remove_nested_expression_reference(did))
        delete expr;
      if ((analysis_mapping != nullptr) && analysis_mapping->remove_reference())
        delete analysis_mapping;
      delete result_info;
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualView::process_collective_user_registration(
        const size_t op_ctx_index, const unsigned index,
        const IndexSpaceID match_space, const AddressSpaceID origin,
        const PhysicalTraceInfo& trace_info,
        CollectiveMapping* analysis_mapping, ApEvent remote_term_event,
        ApUserEvent remote_ready_event, RtUserEvent remote_registered,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(analysis_mapping != nullptr);
#endif
      UserRendezvous to_perform;
      const RendezvousKey key(op_ctx_index, index, match_space);
      {
        AutoLock v_lock(view_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey, UserRendezvous>::iterator finder =
            rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder =
              rendezvous_users.insert(std::make_pair(key, UserRendezvous()))
                  .first;
          UserRendezvous& rendezvous = finder->second;
          rendezvous.local_initialized = false;
          rendezvous.analysis_mapping = analysis_mapping;
          analysis_mapping->add_reference();
          rendezvous.remaining_remote_arrivals =
              analysis_mapping->count_children(origin, local_space);
          // Don't make the ready event, that needs to be done with a
          // local trace_info
          rendezvous.registered = Runtime::create_rt_user_event();
          rendezvous.applied = Runtime::create_rt_user_event();
        }
        if (remote_term_event.exists())
          finder->second.term_events.push_back(remote_term_event);
        Runtime::trigger_event(remote_registered, finder->second.registered);
        if (finder->second.applied.exists())
          applied_events.insert(finder->second.applied);
        if (!finder->second.ready_event.exists())
          finder->second.remote_ready_events[remote_ready_event] =
              new PhysicalTraceInfo(trace_info);
        else
          Runtime::trigger_event(
              remote_ready_event, finder->second.ready_event, trace_info,
              applied_events);
#ifdef DEBUG_LEGION
        assert(finder->second.remaining_remote_arrivals > 0);
#endif
        // Check to see if we've done all the arrivals
        if ((--finder->second.remaining_remote_arrivals > 0) ||
            !finder->second.local_initialized ||
            (finder->second.remaining_local_arrivals > 0))
          return;
#ifdef DEBUG_LEGION
        assert(finder->second.remote_ready_events.empty());
        assert(finder->second.trace_info != nullptr);
#endif
        // Last needed arrival, see if we're the origin or not
        to_perform = std::move(finder->second);
        rendezvous_users.erase(finder);
      }
      ApEvent term_event;
      if (!to_perform.term_events.empty())
        term_event = Runtime::merge_events(
            to_perform.trace_info, to_perform.term_events);
      if (local_space != origin)
      {
#ifdef DEBUG_LEGION
        assert(to_perform.applied.exists());
        assert(to_perform.analysis_mapping != nullptr);
#endif
        // Send the message to the parent
        const AddressSpaceID parent =
            to_perform.analysis_mapping->get_parent(origin, local_space);
        std::set<RtEvent> applied_events;
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(op_ctx_index);
          rez.serialize(index);
          rez.serialize(match_space);
          rez.serialize(origin);
          to_perform.trace_info->pack_trace_info(rez);
          rez.serialize(term_event);
          rez.serialize(to_perform.ready_event);
          rez.serialize(to_perform.registered);
          rez.serialize(to_perform.applied);
        }
        runtime->send_collective_individual_register_user(parent, rez);
        if (!applied_events.empty())
          Runtime::trigger_event(
              to_perform.applied, Runtime::merge_events(applied_events));
        else
          Runtime::trigger_event(to_perform.applied);
      } else
      {
#ifdef DEBUG_LEGION
        assert(to_perform.applied.exists());
#endif
        std::vector<RtEvent> registered_events;
        std::set<RtEvent> applied_events;
        const ApEvent ready = register_user(
            to_perform.usage, *to_perform.mask, to_perform.expr,
            to_perform.op_id, op_ctx_index, index, match_space, term_event,
            manager, nullptr /*no analysis mapping*/,
            0 /*no collective arrivals*/, registered_events, applied_events,
            *to_perform.trace_info, runtime->address_space,
            to_perform.symbolic);
        Runtime::trigger_event(
            to_perform.ready_event, ready, *to_perform.trace_info,
            applied_events);
        if (!registered_events.empty())
          Runtime::trigger_event(
              to_perform.registered, Runtime::merge_events(registered_events));
        else
          Runtime::trigger_event(to_perform.registered);
        if (!applied_events.empty())
          Runtime::trigger_event(
              to_perform.applied, Runtime::merge_events(applied_events));
        else
          Runtime::trigger_event(to_perform.applied);
        delete to_perform.mask;
      }
      if (to_perform.expr->remove_nested_expression_reference(did))
        delete to_perform.expr;
      if ((to_perform.analysis_mapping != nullptr) &&
          to_perform.analysis_mapping->remove_reference())
        delete to_perform.analysis_mapping;
      delete to_perform.trace_info;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualView::handle_collective_user_registration(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      IndividualView* view = static_cast<IndividualView*>(
          runtime->find_or_request_logical_view(did, ready));
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      IndexSpaceID match_space;
      derez.deserialize(match_space);
      AddressSpaceID origin;
      derez.deserialize(origin);
      std::set<RtEvent> applied_events;
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      size_t num_spaces;
      derez.deserialize(num_spaces);
#ifdef DEBUG_LEGION
      assert(num_spaces > 0);
#endif
      CollectiveMapping* mapping = new CollectiveMapping(derez, num_spaces);
      mapping->add_reference();
      ApEvent term_event;
      derez.deserialize(term_event);
      ApUserEvent ready_event;
      derez.deserialize(ready_event);
      RtUserEvent registered_event, applied_event;
      derez.deserialize(registered_event);
      derez.deserialize(applied_event);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();

      view->process_collective_user_registration(
          op_ctx_index, index, match_space, origin, trace_info, mapping,
          term_event, ready_event, registered_event, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(
            applied_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
      if (mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    void IndividualView::pack_fields(
        Serializer& rez, const std::vector<CopySrcDstField>& fields) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(fields.size());
      for (unsigned idx = 0; idx < fields.size(); idx++)
        rez.serialize(fields[idx]);
      if (runtime->legion_spy_enabled)
      {
        rez.serialize<size_t>(0);  // not part of the collective
        rez.serialize(did);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::find_atomic_reservations(
        const FieldMask& mask, Operation* op, unsigned index, bool excl)
    //--------------------------------------------------------------------------
    {
      std::vector<Reservation> reservations;
      find_field_reservations(mask, reservations);
      for (unsigned idx = 0; idx < reservations.size(); idx++)
        op->update_atomic_locks(index, reservations[idx], excl);
    }

    //--------------------------------------------------------------------------
    void IndividualView::find_field_reservations(
        const FieldMask& mask, std::vector<Reservation>& reservations)
    //--------------------------------------------------------------------------
    {
      const RtEvent ready =
          find_field_reservations(mask, &reservations, runtime->address_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      // Sort them into order if necessary
      if (reservations.size() > 1)
        std::sort(reservations.begin(), reservations.end());
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualView::find_field_reservations(
        const FieldMask& mask, std::vector<Reservation>* reservations,
        AddressSpaceID source, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      std::vector<Reservation> results;
      if (is_logical_owner())
      {
        results.reserve(mask.pop_count());
        // We're the owner so we can make all the fields
        AutoLock v_lock(view_lock);
        for (int idx = mask.find_first_set(); idx >= 0;
             idx = mask.find_next_set(idx + 1))
        {
          std::map<unsigned, Reservation>::const_iterator finder =
              view_reservations.find(idx);
          if (finder == view_reservations.end())
          {
            // Make a new reservation and add it to the set
            Reservation handle = Reservation::create_reservation();
            view_reservations[idx] = handle;
            results.push_back(handle);
          } else
            results.push_back(finder->second);
        }
      } else
      {
        // See if we can find them all locally
        {
          AutoLock v_lock(view_lock, 1, false /*exclusive*/);
          for (int idx = mask.find_first_set(); idx >= 0;
               idx = mask.find_next_set(idx + 1))
          {
            std::map<unsigned, Reservation>::const_iterator finder =
                view_reservations.find(idx);
            if (finder != view_reservations.end())
              results.push_back(finder->second);
            else
              break;
          }
        }
        if (results.size() < mask.pop_count())
        {
          // Couldn't find them all so send the request to the owner
          if (!to_trigger.exists())
            to_trigger = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(mask);
            rez.serialize(reservations);
            rez.serialize(source);
            rez.serialize(to_trigger);
          }
          runtime->send_atomic_reservation_request(logical_owner, rez);
          return to_trigger;
        }
      }
      if (source != local_space)
      {
#ifdef DEBUG_LEGION
        assert(to_trigger.exists());
#endif
        // Send the result back to the source
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(mask);
          rez.serialize(reservations);
          rez.serialize<size_t>(results.size());
          for (std::vector<Reservation>::const_iterator it = results.begin();
               it != results.end(); it++)
            rez.serialize(*it);
          rez.serialize(to_trigger);
        }
        runtime->send_atomic_reservation_response(source, rez);
      } else
      {
        reservations->swap(results);
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
      }
      return to_trigger;
    }

    //--------------------------------------------------------------------------
    void IndividualView::update_field_reservations(
        const FieldMask& mask, const std::vector<Reservation>& reservations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
#endif
      AutoLock v_lock(view_lock);
      unsigned offset = 0;
      for (int idx = mask.find_first_set(); idx >= 0;
           idx = mask.find_next_set(idx + 1))
        view_reservations[idx] = reservations[offset++];
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualView::handle_atomic_reservation_request(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      IndividualView* view = static_cast<IndividualView*>(
          runtime->find_or_request_logical_view(did, ready));
      FieldMask mask;
      derez.deserialize(mask);
      std::vector<Reservation>* target;
      derez.deserialize(target);
      AddressSpaceID source;
      derez.deserialize(source);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      view->find_field_reservations(mask, target, source, to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualView::handle_atomic_reservation_response(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      IndividualView* view = static_cast<IndividualView*>(
          runtime->find_or_request_logical_view(did, ready));
      FieldMask mask;
      derez.deserialize(mask);
      std::vector<Reservation>* target;
      derez.deserialize(target);
      size_t num_reservations;
      derez.deserialize(num_reservations);
      target->resize(num_reservations);
      for (unsigned idx = 0; idx < num_reservations; idx++)
        derez.deserialize((*target)[idx]);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      view->update_field_reservations(mask, *target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualView::handle_view_find_copy_pre_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView* view = runtime->find_or_request_logical_view(did, ready);

      bool reading;
      derez.deserialize<bool>(reading);
      ReductionOpID redop;
      derez.deserialize(redop);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      IndexSpaceExpression* copy_expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      ApUserEvent to_trigger;
      derez.deserialize(to_trigger);
      RtUserEvent applied;
      derez.deserialize(applied);
      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);

      // This blocks the virtual channel, but keeps queries in-order
      // with respect to updates from the same node which is necessary
      // for preventing cycles in the realm event graph
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      IndividualView* inst_view = view->as_individual_view();
      const ApEvent pre = inst_view->find_copy_preconditions(
          reading, redop, copy_mask, copy_expr, op_id, index, applied_events,
          trace_info);
      Runtime::trigger_event(to_trigger, pre, trace_info, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualView::handle_view_add_copy_user(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView* view = runtime->find_or_request_logical_view(did, ready);

      bool reading;
      derez.deserialize(reading);
      ReductionOpID redop;
      derez.deserialize(redop);
      ApEvent term_event;
      derez.deserialize(term_event);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      IndexSpaceExpression* copy_expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);
      bool trace_recording;
      derez.deserialize(trace_recording);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_individual_view());
#endif
      IndividualView* inst_view = view->as_individual_view();

      std::set<RtEvent> applied_events;
      inst_view->add_copy_user(
          reading, redop, term_event, copy_mask, copy_expr, op_id, index,
          applied_events, trace_recording, source);
      if (!applied_events.empty())
        Runtime::trigger_event(
            applied_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }

    //--------------------------------------------------------------------------
    void IndividualView::handle_view_find_last_users_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      LogicalView* view = runtime->find_or_request_logical_view(did, ready);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      RtEvent manager_ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(manager_did, manager_ready);

      std::vector<ApEvent>* target;
      derez.deserialize(target);
      RegionUsage usage;
      derez.deserialize(usage);
      FieldMask mask;
      derez.deserialize(mask);
      IndexSpaceExpression* expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      RtUserEvent done;
      derez.deserialize(done);

      std::set<ApEvent> result;
      std::vector<RtEvent> applied;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      if (manager_ready.exists() && !manager_ready.has_triggered())
        manager_ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_individual_view());
#endif
      IndividualView* inst_view = view->as_individual_view();
      inst_view->find_last_users(manager, result, usage, mask, expr, applied);
      if (!result.empty())
      {
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize<size_t>(result.size());
          for (std::set<ApEvent>::const_iterator it = result.begin();
               it != result.end(); it++)
            rez.serialize(*it);
          rez.serialize(done);
          if (!applied.empty())
            rez.serialize(Runtime::merge_events(applied));
          else
            rez.serialize(RtEvent::NO_RT_EVENT);
        }
        runtime->send_view_find_last_users_response(source, rez);
      } else
      {
        if (!applied.empty())
          Runtime::trigger_event(done, Runtime::merge_events(applied));
        else
          Runtime::trigger_event(done);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::handle_view_find_last_users_response(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::set<ApEvent>* target;
      derez.deserialize(target);
      size_t num_events;
      derez.deserialize(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        ApEvent event;
        derez.deserialize(event);
        target->insert(event);
      }
      RtUserEvent done;
      derez.deserialize(done);
      RtEvent pre;
      derez.deserialize(pre);
      Runtime::trigger_event(done, pre);
    }

  }  // namespace Internal
}  // namespace Legion
