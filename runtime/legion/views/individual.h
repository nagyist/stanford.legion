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

#ifndef __LEGION_INDIVIDUAL_VIEW_H__
#define __LEGION_INDIVIDUAL_VIEW_H__

#include "legion/views/instance.h"
#include "legion/nodes/expression.h"
#include "legion/tracing/recording.h"
#include "legion/utilities/privileges.h"

namespace Legion {
  namespace Internal {

    /**
     * \class IndividualView
     * This class provides an abstract base class for any kind of view
     * that only represents an individual physical instance.
     */
    class IndividualView : public InstanceView {
    public:
      IndividualView(
          DistributedID did, PhysicalManager* man, AddressSpaceID logical_owner,
          bool register_now, CollectiveMapping* mapping);
      virtual ~IndividualView(void);
    public:
      inline bool is_logical_owner(void) const
      {
        return (local_space == logical_owner);
      }
      inline PhysicalManager* get_manager(void) const { return manager; }
    public:
      virtual AddressSpaceID get_analysis_space(
          PhysicalManager* inst) const override;
      virtual bool aliases(InstanceView* other) const override;
    public:
      // Reference counting state change functions
      virtual void notify_local(void) override;
      virtual void notify_valid(void) override;
      virtual bool notify_invalid(void) override;
    public:
      virtual void pack_valid_ref(void) override;
      virtual void unpack_valid_ref(void) override;
    public:
      virtual ApEvent fill_from(
          FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpaceID collective_match_space, const FieldMask& fill_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CopyAcrossHelper* across_helper, const bool manage_dst_events,
          const bool fill_restricted, const bool need_valid_return) override;
      virtual ApEvent copy_from(
          InstanceView* src_view, ApEvent precondition,
          PredEvent predicate_guard, ReductionOpID redop,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpaceID collective_match_space, const FieldMask& copy_mask,
          PhysicalManager* src_point, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CopyAcrossHelper* across_helper, const bool manage_dst_events,
          const bool copy_restricted, const bool need_valid_return) override;
      virtual ApEvent register_user(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
          const unsigned index, const IndexSpaceID collective_match_space,
          ApEvent term_event, PhysicalManager* target,
          CollectiveMapping* collective_mapping,
          size_t local_collective_arrivals,
          std::vector<RtEvent>& registered_events,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info, const AddressSpaceID source,
          const bool symbolic = false) override;
    public:
      void add_initial_user(
          ApEvent term_event, const RegionUsage& usage,
          const FieldMask& user_mask, IndexSpaceExpression* expr,
          const UniqueID op_id, const unsigned index);
      ApEvent find_copy_preconditions(
          bool reading, ReductionOpID redop, const FieldMask& copy_mask,
          IndexSpaceExpression* copy_expr, UniqueID op_id, unsigned index,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info);
      void add_copy_user(
          bool reading, ReductionOpID redop, ApEvent done_event,
          const FieldMask& copy_mask, IndexSpaceExpression* copy_expr,
          UniqueID op_id, unsigned index, std::set<RtEvent>& applied_events,
          const bool trace_recording, const AddressSpaceID source);
      void find_last_users(
          PhysicalManager* target, std::set<ApEvent>& events,
          const RegionUsage& usage, const FieldMask& mask,
          IndexSpaceExpression* user_expr, std::vector<RtEvent>& applied) const;
    public:
      void pack_fields(
          Serializer& rez, const std::vector<CopySrcDstField>& fields) const;
      void find_atomic_reservations(
          const FieldMask& mask, Operation* op, const unsigned index,
          bool exclusive);
      void find_field_reservations(
          const FieldMask& mask, std::vector<Reservation>& results);
      RtEvent find_field_reservations(
          const FieldMask& mask, std::vector<Reservation>* results,
          AddressSpaceID source,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT);
      void update_field_reservations(
          const FieldMask& mask, const std::vector<Reservation>& rsrvs);
    public:
      void register_collective_analysis(
          const CollectiveView* source, CollectiveAnalysis* analysis);
      CollectiveAnalysis* find_collective_analysis(
          size_t context_index, unsigned region_index,
          IndexSpaceID match_space);
      void unregister_collective_analysis(
          const CollectiveView* source, size_t context_index,
          unsigned region_index, IndexSpaceID match_space);
    protected:
      void add_internal_task_user(
          const RegionUsage& usage, IndexSpaceExpression* user_expr,
          const FieldMask& user_mask, ApEvent term_event, UniqueID op_id,
          const unsigned index);
      void add_internal_copy_user(
          const RegionUsage& usage, IndexSpaceExpression* user_expr,
          const FieldMask& user_mask, ApEvent term_event, UniqueID op_id,
          const unsigned index);
      void clean_cache(void);
      ApEvent register_collective_user(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
          const unsigned index, const IndexSpaceID match_space,
          ApEvent term_event, PhysicalManager* target,
          CollectiveMapping* analysis_mapping, size_t local_collective_arrivals,
          std::vector<RtEvent>& registered_events,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info, const bool symbolic);
    public:
      void process_collective_user_registration(
          const size_t op_ctx_index, const unsigned index,
          const IndexSpaceID match_space, const AddressSpaceID origin,
          const PhysicalTraceInfo& trace_info,
          CollectiveMapping* analysis_mapping, ApEvent remote_term_event,
          ApUserEvent remote_ready_event, RtUserEvent remote_registered,
          std::set<RtEvent>& applied_events);
    public:
      PhysicalManager* const manager;
      // This is the owner space for the purpose of logical analysis
      // If you ever make this non-const then be sure to update the
      // code in register_collective_user
      const AddressSpaceID logical_owner;
    protected:
      // Use a ExprView DAG to track the current users of this instance
      ExprView* const current_users;
      // Lock for serializing creation of ExprView objects
      mutable LocalLock expr_lock;
      // Mapping from user expressions to ExprViews to attach to
      lng::map<IndexSpaceExprID, ExprView*> expr_cache;
      // Number of users to be added between cache invalidations
      static constexpr unsigned USER_CACHE_TIMEOUT = 1024;
      // A timeout counter for the cache so we don't permanently keep growing
      // in the case where the sets of expressions we use change over time
      std::atomic<unsigned> expr_cache_uses;
      // Helping with making sure that there are no outstanding users being
      // added for when we go to invalidate the cache and clean the views
      std::atomic<unsigned> outstanding_additions;
      RtUserEvent clean_waiting;
      std::map<unsigned, Reservation> view_reservations;
    protected:
      // This is an infrequently used data structure for handling collective
      // register user calls on individual managers that occurs with certain
      // operation in control replicated contexts
      struct UserRendezvous {
        UserRendezvous(void)
          : remaining_local_arrivals(0), remaining_remote_arrivals(0),
            trace_info(nullptr), analysis_mapping(nullptr), mask(nullptr),
            expr(nullptr), op_id(0), symbolic(false), local_initialized(false)
        { }
        // event for when local instances can be used
        ApUserEvent ready_event;
        // remote ready events to trigger
        std::map<ApUserEvent, PhysicalTraceInfo*> remote_ready_events;
        // all the local term events
        std::vector<ApEvent> term_events;
        // event that marks when all registrations are done
        RtUserEvent registered;
        // event for when any local effects are applied
        RtUserEvent applied;
        // Counts of remaining notficiations before registration
        unsigned remaining_local_arrivals;
        unsigned remaining_remote_arrivals;
        // PhysicalTraceInfo that made the ready_event and should trigger it
        PhysicalTraceInfo* trace_info;
        CollectiveMapping* analysis_mapping;
        // Arguments for performing the local registration
        RegionUsage usage;
        HeapifyBox<FieldMask, OPERATION_LIFETIME>* mask;
        IndexSpaceNode* expr;
        UniqueID op_id;
        bool symbolic;
        bool local_initialized;
      };
      std::map<RendezvousKey, UserRendezvous> rendezvous_users;
    protected:
      // This is actually quite important!
      // Normally each collective analysis is associated with a specific
      // collective view. However the copies done by that analysis might
      // only be occurring on collective views that are a subset of the
      // collective view for the analysis. Therefore we register the analyses
      // with the individual views so that they can be found by any copies
      struct RegisteredAnalysis {
      public:
        CollectiveAnalysis* analysis;
        RtUserEvent ready;
        // We need to deduplicate across views that are performing
        // registrations on this instance. With multiple fields we
        // can get multiple different views using the same instance
        // and each doing their own registration
        std::set<DistributedID> views;
      };
      std::map<RendezvousKey, RegisteredAnalysis> collective_analyses;
    };

    /**
     * \struct PhysicalUser
     * A class for representing physical users of a logical
     * region including necessary information to
     * register execution dependences on the user.
     */
    struct PhysicalUser : public Collectable {
    public:
      PhysicalUser(
          const RegionUsage& u, IndexSpaceExpression* expr, ApEvent term,
          UniqueID op_id, unsigned index, bool copy, bool covers);
      PhysicalUser(const PhysicalUser& rhs) = delete;
      ~PhysicalUser(void);
    public:
      PhysicalUser& operator=(const PhysicalUser& rhs) = delete;
    public:
      const RegionUsage usage;
      IndexSpaceExpression* const expr;
      const ApEvent term_event;
      const UniqueID op_id;
      const unsigned index;  // region requirement index
      const bool copy_user;  // is this from a copy or an operation
      const bool covers;     // whether the expr covers the ExprView its in
    };

    /**
     * \class ExprView
     * A ExprView is a node in a tree of ExprViews for capturing users of a
     * physical instance. At each node it tracks the users of a specific
     * index space expression for the physical instance. It also knows about
     * the subviews which are any expressions that are dominated by the
     * current node and which may overlap but no subview can dominate another.
     * Finding the interfering users then just requires traversing the top
     * node and any overlapping sub nodes and then doing this recursively.
     */
    class ExprView : public Heapify<ExprView, CONTEXT_LIFETIME>,
                     public Collectable {
    public:
      ExprView(
          DistributedID view_did, IndexSpaceExpression* expr,
          bool unbound = false);
      ExprView(const ExprView& rhs) = delete;
      virtual ~ExprView(void);
    public:
      ExprView& operator=(const ExprView& rhs) = delete;
    public:
      void find_user_preconditions(
          const RegionUsage& usage, IndexSpaceExpression* user_expr,
          const bool user_dominates, const FieldMask& user_mask,
          ApEvent term_event, UniqueID op_id, unsigned index,
          std::set<ApEvent>& preconditions, const bool trace_recording);
      void find_copy_preconditions(
          const RegionUsage& usage, IndexSpaceExpression* copy_expr,
          const bool copy_dominates, const FieldMask& copy_mask, UniqueID op_id,
          unsigned index, std::set<ApEvent>& preconditions,
          const bool trace_recording);
      void find_last_users(
          const RegionUsage& usage, IndexSpaceExpression* expr,
          const bool expr_dominates, const FieldMask& mask,
          std::set<ApEvent>& last_events) const;
      // Add a new subview with fields into the tree
      void insert_subview(ExprView* subview, FieldMask& subview_mask);
      void find_tightest_subviews(
          IndexSpaceExpression* expr, FieldMask& expr_mask,
          local::map<std::pair<size_t, ExprView*>, FieldMask>& bounding_views);
      void add_partial_user(
          const RegionUsage& usage, UniqueID op_id, unsigned index,
          FieldMask user_mask, const ApEvent term_event,
          IndexSpaceExpression* user_expr, const size_t user_volume,
          PhysicalUser*& covered_user, PhysicalUser*& uncovered_user);
      void add_current_user(PhysicalUser* user, const FieldMask& user_mask);
      // TODO: Optimize this so that we prune out intermediate nodes in
      // the tree while still allowing precondition searches to proceed
      // in parallel. Right now we stop the world to prune out such nodes
      void clean_views(
          FieldMask& valid_mask, local::FieldMaskMap<ExprView>& clean_set);
    protected:
      void find_current_preconditions(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceExpression* user_expr, ApEvent term_event,
          const UniqueID op_id, const unsigned index, const bool user_covers,
          std::set<ApEvent>& preconditions, std::set<PhysicalUser*>& dead_users,
          local::FieldMaskMap<PhysicalUser>& filter_users, FieldMask& observed,
          FieldMask& non_dominated, const bool trace_recording,
          const bool copy_user);
      void find_previous_preconditions(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceExpression* user_expr, ApEvent term_event,
          const UniqueID op_id, const unsigned index, const bool user_covers,
          std::set<ApEvent>& preconditions, std::set<PhysicalUser*>& dead_users,
          const bool trace_recording, const bool copy_user);
      void find_previous_filter_users(
          const FieldMask& dominated_mask,
          local::FieldMaskMap<PhysicalUser>& filter_users);
      // Overloads for find_last_users
      void find_current_preconditions(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceExpression* expr, const bool expr_covers,
          std::set<ApEvent>& last_events, FieldMask& observed,
          FieldMask& non_dominated) const;
      void find_previous_preconditions(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceExpression* expr, const bool expr_covers,
          std::set<ApEvent>& last_events) const;
      inline bool has_local_precondition(
          PhysicalUser* prev_user, const RegionUsage& next_user,
          IndexSpaceExpression* user_expr, const UniqueID op_id,
          const unsigned index, const bool user_covers, const bool copy_user,
          bool* dominates = nullptr) const;
    public:
      size_t get_view_volume(void);
      void find_all_done_events(std::set<ApEvent>& all_done) const;
    protected:
      void filter_dead_users(const std::set<PhysicalUser*>& dead_users);
      void filter_current_users(const FieldMapView<PhysicalUser>& to_filter);
      void filter_previous_users(const FieldMapView<PhysicalUser>& to_filter);
      static void verify_current_to_filter(
          const FieldMask& dominated,
          local::FieldMaskMap<PhysicalUser>& current_to_filter);
    public:
      IndexSpaceExpression* const view_expr;
      std::atomic<size_t> view_volume;
      const DistributedID view_did;
      // This is publicly mutable and protected by expr_lock from
      // the owner inst_view
      FieldMask invalid_fields;
    protected:
      mutable LocalLock view_lock;
    protected:
      // There are three operations that are done on materialized views
      // 1. iterate over all the users for use analysis
      // 2. garbage collection to remove old users for an event
      // 3. send updates for a certain set of fields
      // The first and last both iterate over the current and previous
      // user sets, while the second one needs to find specific events.
      // Therefore we store the current and previous sets as maps to
      // users indexed by events. Iterating over the maps are no worse
      // than iterating over lists (for arbitrary insertion and deletion)
      // and will provide fast indexing for removing items. We used to
      // store users in current and previous epochs similar to logical
      // analysis, but have since switched over to storing readers and
      // writers that are not filtered as part of analysis. This let's
      // us perform more analysis in parallel since we'll only need to
      // hold locks in read-only mode prevent user fragmentation. It also
      // deals better with the common case which are higher views in
      // the view tree that less frequently filter their sub-users.
      shrt::FieldMaskMap<PhysicalUser> current_epoch_users;
      shrt::FieldMaskMap<PhysicalUser> previous_epoch_users;
    protected:
      // Subviews for fields that have users in subexpressions
      lng::FieldMaskMap<ExprView> subviews;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/views/individual.inl"

#endif  // __LEGION_INDIVIDUAL_VIEW_H__
