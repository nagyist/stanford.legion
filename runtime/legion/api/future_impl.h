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

#ifndef __LEGION_FUTURE_IMPL_H__
#define __LEGION_FUTURE_IMPL_H__

#include "legion/api/functors.h"
#include "legion/api/future.h"
#include "legion/api/sync.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/kernel/metatask.h"
#include "legion/utilities/coordinates.h"

namespace Legion {

  // Help for adding and removing references to sparsity maps 
  struct SparsityReferenceHelper {
  public:
    SparsityReferenceHelper(const Domain &d) : domain(d) { }
    template<typename N, typename T>
    static inline void demux(SparsityReferenceHelper *functor)
    {
      DomainT<N::N,T> is = functor->domain;
      Internal::RtEvent wait_on(is.sparsity.add_reference());
      wait_on.wait();
    }
    static void deletion_function(const Realm::ExternalInstanceResource &r);
  public:
    const Domain &domain;
  };

  namespace Internal {

    /**
     * \class FutureImpl
     * The base implementation of a future object.  The runtime
     * manages future implementation objects and knows how to
     * copy them from one node to another.  Future implementations
     * are always made first on the owner node and then moved
     * remotely.  We use the distributed collectable scheme
     * to manage garbage collection of distributed futures
     */
    class FutureImpl : public DistributedCollectable,
                       public Heapify<FutureImpl,SHORT_BOUNDED_LIFETIME> {
    public:
      struct ContributeCollectiveArgs : 
        public LgTaskArgs<ContributeCollectiveArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CONTRIBUTE_COLLECTIVE_ID;
      public:
        ContributeCollectiveArgs(FutureImpl *i, DynamicCollective d, unsigned c)
          : LgTaskArgs<ContributeCollectiveArgs>(implicit_provenance),
            impl(i), dc(d), count(c) { }
      public:
        FutureImpl *const impl;
        const DynamicCollective dc;
        const unsigned count;
      };
      struct FutureCallbackArgs : public LgTaskArgs<FutureCallbackArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FUTURE_CALLBACK_TASK_ID;
      public:
        FutureCallbackArgs(FutureImpl *i);
      public:
        FutureImpl *const impl;
      };
      struct CallbackReleaseArgs : public LgTaskArgs<CallbackReleaseArgs> {
      public:
        static const LgTaskID TASK_ID = LG_CALLBACK_RELEASE_TASK_ID;
      public:
        CallbackReleaseArgs(FutureFunctor *functor, bool own_functor);
      public:
        FutureFunctor *const functor;
        const bool own_functor;
      };
      struct FutureBroadcastArgs : public LgTaskArgs<FutureBroadcastArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FUTURE_BROADCAST_TASK_ID;
      public:
        FutureBroadcastArgs(FutureImpl *i);
      public:
        FutureImpl *const impl;
      };
      struct PendingInstance {
      public:
        PendingInstance(void)
          : instance(NULL), creator_uid(0) { }
        PendingInstance(FutureInstance *i, UniqueID u)
          : instance(i), creator_uid(u) { }
      public:
        FutureInstance *instance;
        UniqueID creator_uid;
        ApUserEvent inst_ready;
        RtEvent safe_inst_ready;
      };
      struct FutureInstanceTracker {
      public:
        FutureInstanceTracker(void) : instance(NULL) { }
        FutureInstanceTracker(FutureInstance *inst, ApEvent ready,
            ApUserEvent remote = ApUserEvent::NO_AP_USER_EVENT)
          : instance(inst), ready_event(ready), remote_postcondition(remote) { }
      public:
        FutureInstance *const instance;
        const ApEvent ready_event;
        RtEvent safe_ready_event;
        ApUserEvent remote_postcondition;
        std::vector<ApEvent> read_events;
      };
    public:
      // This constructor provides the complete size and effects event
      // at the point the future is constructed so they don't need to
      // be provided later with set_future_result_size 
      FutureImpl(TaskContext *ctx, bool register_future,
                 DistributedID did, Provenance *provenance,
                 Operation *op = NULL);
      // This constructor is for futures made by tasks or other operations
      // which do not know the size or effects for the operation until later
      FutureImpl(TaskContext *ctx, bool register_future, 
                 DistributedID did, Operation *op, GenerationID gen,
                 const ContextCoordinate &coordinate,
                 UniqueID op_uid, int op_depth, Provenance *provenance,
                 CollectiveMapping *mapping = NULL);
      FutureImpl(const FutureImpl &rhs) = delete;
      virtual ~FutureImpl(void);
    public:
      FutureImpl& operator=(const FutureImpl &rhs) = delete;
    public:
      // Finalize the future before everything shuts down
      void prepare_for_shutdown(void);
      ApEvent get_complete_event(void);
      bool is_ready(bool subscribe);
      // Wait without subscribing to the payload
      void wait(bool silence_warnings, const char *warning_string);
      const void* get_buffer(Processor proc, Memory::Kind memory,
                             size_t *extent_in_bytes = NULL, 
                             bool check_extent = false,
                             bool silence_warnings = false, 
                             const char *warning_string = NULL);
      const void* get_buffer(Memory memory,
                             size_t *extent_in_bytes = NULL, 
                             bool check_extent = false,
                             bool silence_warnings = false, 
                             const char *warning_string = NULL);
      void get_memories(std::set<Memory> &memories,
                        bool silence_warnings, const char *warning_string);
      PhysicalInstance get_instance(Memory::Kind kind,
                             size_t extent_in_bytes, bool check_extent,
                             bool silence_warnings, const char *warning_string);
      void report_incompatible_accessor(const char *accessor_kind,
                                        PhysicalInstance instance);
      bool request_application_instance(Memory target, SingleTask *task,
          RtEvent *safe_for_unbounded_pools, bool can_fail = false,
          size_t known_upper_bound_size = SIZE_MAX);
      bool find_or_create_application_instance(Memory target, 
          size_t known_upper_bound_size, UniqueID task_uid,
          const TaskTreeCoordinates &coordinates,
          RtEvent *safe_for_unbounded_pools);
      ApEvent find_application_instance_ready(Memory target, SingleTask *task);
      void request_runtime_instance(Operation *op);
      RtEvent find_runtime_instance_ready(void);
      const void *find_runtime_buffer(TaskContext *ctx, size_t &expected_size);
      ApEvent copy_to(FutureInstance *target, Operation *op,
                      ApEvent precondition);
      ApEvent reduce_to(FutureInstance *target, AllReduceOp *op,
                        const ReductionOpID redop_id, const ReductionOp *redop,
                        bool exclusive, ApEvent precondition);
      bool is_empty(bool block, bool silence_warnings = true,
                    const char *warning_string = NULL,
                    bool internal = false);
      size_t get_untyped_size(void);
      const void *get_metadata(size_t *metasize);
      // A special function for predicates to peek
      // at the boolean value of a future if it is set
      // Must have called request internal buffer first and event must trigger
      bool get_boolean_value(TaskContext *ctx);
    public:
      // This will simply save the value of the future
      void set_result(ApEvent complete, FutureInstance *instance = NULL, 
                      const void *metadata = NULL, size_t metasize = 0);
      void set_results(ApEvent complete,
                      const std::vector<FutureInstance*> &instances,
                      const void *metadata = NULL, size_t metasize = 0);
      void set_result(ApEvent complete, FutureFunctor *callback_functor,
                      bool own, Processor functor_proc);
      void set_result(Operation *op, FutureImpl *previous,
                      RtEvent *safe_for_unbounded_pools);
      void set_result(TaskContext *ctx, FutureImpl *previous);
      // This is the same as above but for data that we know is visible
      // in the system memory and should always make a local FutureInstance
      // and for which we know that there is no completion effects
      void set_local(const void *value, size_t size, bool own = false);
      // This will save the value of the future locally
      void unpack_future_result(Deserializer &derez);
      void save_metadata(const void *meta, size_t size);
      // Reset the future in case we need to restart the
      // computation for resiliency reasons
      bool reset_future(void);
      // Request that we get meta data for the future on this node
      // The return event here will indicate when we have local data
      // that is valid to access for this particular future
      RtEvent subscribe(bool need_lock = true);
      size_t get_upper_bound_size(void);
      bool get_context_coordinate(const TaskContext *ctx,
                                  ContextCoordinate &coordinate) const;
      void pack_future(Serializer &rez, AddressSpaceID target);
      static Future unpack_future(
          Deserializer &derez, Operation *op = NULL, GenerationID op_gen = 0,
          UniqueID op_uid = 0, int op_depth = 0);
    public:
      virtual void notify_local(void);
    public:
      void register_dependence(Operation *consumer_op);
      void register_remote(AddressSpaceID sid);
      void set_future_result_size(size_t size, AddressSpaceID source);
    protected:
      void finish_set_future(ApEvent complete); // must be holding lock
      void create_pending_instances(void); // must be holding lock
      FutureInstance* find_or_create_instance(Memory memory,ApEvent &inst_ready,
          bool silence_warnings, const char *warning_string);
      FutureInstance* create_instance(Operation *op, Memory memory,
          size_t size, RtEvent *safe_for_unbounded_pools);
      // Must be holding the lock when calling initialize_instance
      ApEvent record_instance(FutureInstance *instance, UniqueID creator_uid);
      Memory find_best_source(Memory target) const;
      void mark_sampled(void);
      void broadcast_result(void); // must be holding lock
      void record_subscription(AddressSpaceID subscriber, bool need_lock);
    protected:
      RtEvent invoke_callback(void); // must be holding lock
      void perform_callback(void);
      void perform_broadcast(void);
      // must be holding lock
      void pack_future_result(Serializer &rez, AddressSpaceID target);
    public:
      RtEvent record_future_registered(bool has_global_reference);
      static void handle_future_result(Deserializer &derez);
      static void handle_future_result_size(Deserializer &derez,
                                  AddressSpaceID source);
      static void handle_future_subscription(Deserializer &derez,
                                             AddressSpaceID source);
      static void handle_future_create_instance_request(Deserializer &derez,
                                                        AddressSpaceID source);
      static void handle_future_create_instance_response(Deserializer &derez);
    public:
      void contribute_to_collective(const DynamicCollective &dc,unsigned count);
      static void handle_contribute_to_collective(const void *args);
      static void handle_callback(const void *args);
      static void handle_release(const void *args);
      static void handle_broadcast(const void *args);
    public:
      TaskContext *const context;
      // These three fields are only valid on the owner node
      Operation *const producer_op;
      const GenerationID op_gen;
      // The depth of the context in which this was made
      const int producer_depth;
      const UniqueID producer_uid;
      // Note this is a blocking coordinate and not a task tree coordinate!
      const ContextCoordinate coordinate;
      Provenance *const provenance;
    private:
      mutable LocalLock future_lock;
      RtUserEvent subscription_event;
      AddressSpaceID result_set_space;
      // On the owner node, keep track of the registered waiters
      std::set<AddressSpaceID> subscribers; 
      std::map<Memory,FutureInstanceTracker> instances;
      Memory local_visible_memory;
    private:
      void *metadata;
      size_t metasize;
    private:
      // The determined size of this future to this point
      // This is only an upper bound until it is solidifed
      size_t future_size;
      // This is the upper bound size prior to being refined
      // down to a precise size when the future is finally set
      size_t upper_bound_size; 
      // The event denoting when all the effects represented by
      // this future are actually complete
      ApEvent future_complete;
      // Event for when the future size is set if needed
      RtUserEvent future_size_ready;
    private:
      // Instances that need to be made once we set the future
      std::map<Memory,PendingInstance> pending_instances;
    private:
      Processor callback_proc;
      FutureFunctor *callback_functor;
      bool own_callback_functor;
    private:
      // Whether this future has a size set yet
      bool future_size_set;
    private:
      std::atomic<bool> empty;
      std::atomic<bool> sampled;
    };

    /**
     * \class FutureInstance
     * A future instance represents the data for a single copy of
     * the future in a memory somewhere. It has a duality to it that
     * is likely confusing at first. It can either be an external 
     * allocation which may or may not have an external realm instance
     * associated with it. Or it could be a normal realm instance for
     * which we have extracted the pointer and size for it. Furthermore
     * when moving these from one node to another, sometimes we pass
     * them by-value if they can be cheaply copied, other times we
     * will move the just the references to the instances and allocations.
     * You'll have to look into the implementation to discover which
     * is happening, but when you get an unpacked copy on the remote
     * side it is a valid future instance that can you use regardless.
     * Each future instance has a concept of instance ownership which 
     * exists with exactly one copy of each future instance. If a future
     * instance is packed and moved to a remote node then it can only be
     * read from so we can track the appropriate read effects.
     * Current future instances are immutable after they are initially 
     * written, but are designed so that we might easily be able to relax
     * that later so we can support mutable future values.
     * Note that none of the methods in this class are thread safe so
     * atomicity needs to come from the caller.
     */
    class FutureInstance {
    public:
      struct DeferDeleteFutureInstanceArgs :
        public LgTaskArgs<DeferDeleteFutureInstanceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_DELETE_FUTURE_INSTANCE_TASK_ID;
      public:
        DeferDeleteFutureInstanceArgs(FutureInstance *inst)
          : LgTaskArgs<DeferDeleteFutureInstanceArgs>(implicit_provenance),
            instance(inst) { }
      public:
        FutureInstance *const instance;
      };
      struct FreeExternalArgs : public LgTaskArgs<FreeExternalArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FREE_EXTERNAL_TASK_ID;
      public:
        FreeExternalArgs(const Realm::ExternalInstanceResource *r,
            void (*func)(const Realm::ExternalInstanceResource&),
            PhysicalInstance inst);
      public:
        const Realm::ExternalInstanceResource *const resource;
        void (*const freefunc)(const Realm::ExternalInstanceResource&);
        const PhysicalInstance instance;
      };
    public:
      FutureInstance(const void *data, size_t size,
                     bool external, bool own_allocation = true,
                     LgEvent unique_event = LgEvent::NO_LG_EVENT,
                     PhysicalInstance inst = PhysicalInstance::NO_INST,
                     Processor free_proc = Processor::NO_PROC,
                     RtEvent use_event = RtEvent::NO_RT_EVENT);
      FutureInstance(const void *data, size_t size, bool own,
                     const Realm::ExternalInstanceResource *allocation,
                     void (*freefunc)(
                       const Realm::ExternalInstanceResource&) = NULL,
                     Processor free_proc = Processor::NO_PROC,
                     LgEvent unique_event = LgEvent::NO_LG_EVENT,
                     PhysicalInstance inst = PhysicalInstance::NO_INST,
                     RtEvent use_event = RtEvent::NO_RT_EVENT);
      FutureInstance(const FutureInstance &rhs) = delete;
      ~FutureInstance(void);
    public:
      FutureInstance& operator=(const FutureInstance &rhs) = delete;
    public:
      ApEvent initialize(const ReductionOp *redop, Operation *op,
                         ApEvent precondition);
      ApEvent copy_from(FutureInstance *source, Operation *op,
                        ApEvent precondition);
      ApEvent copy_from(FutureInstance *source, UniqueID uid,
                        ApEvent precondition);
      ApEvent reduce_from(FutureInstance *source, Operation *op,
                          const ReductionOpID redop_id,
                          const ReductionOp *redop, bool exclusive,
                          ApEvent precondition);
    public:
      // This method can be called concurrently from different threads
      const void* get_data(void);
      // This method will return an instance that represents the
      // data for this future instance of a given size, if the needed size
      // does not match the base size then a fresh instance will be returned
      // which will be the responsibility of the caller to destroy
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
      PhysicalInstance get_instance(size_t needed_size, 
          LgEvent &inst_event, bool &own_inst);
#else
      PhysicalInstance get_instance(size_t needed_size, bool &own_inst);
#endif
      bool defer_deletion(ApEvent precondition);
    public:
      bool is_immediate(void) const; 
      bool can_pack_by_value(void) const;
      // You only need to check the return value if you set pack_ownership=false
      // as that is when the you need to make sure the instance isn't deleted
      // remotely, whereas in all other cases it is safe to delete locally
      bool pack_instance(Serializer &rez, ApEvent ready_event,
          bool pack_ownership, bool allow_by_value = true); 
      static void pack_null(Serializer &rez);
      static FutureInstance* unpack_instance(Deserializer &derez);
    public:
      static bool check_meta_visible(Memory memory);
      static FutureInstance* create_local(const void *value, size_t size, 
                                          bool own);
      static void handle_free_external(Deserializer &derez);
      static void handle_free_external(const void *args);
      static void free_host_memory(const Realm::ExternalInstanceResource &mem);
      static void handle_defer_deletion(const void *args);
    public:
      const size_t size;
      const Memory memory;
      const Realm::ExternalInstanceResource *const resource;
      void (*const freefunc)(const Realm::ExternalInstanceResource&);
      const Processor freeproc;
      const bool external_allocation;
      const bool is_meta_visible;
    protected:
      bool own_allocation;
      std::atomic<const void*> data;
      // This instance always has a domain of [0,0] and a field
      // size == `size` for the future instance
      PhysicalInstance instance;
      // Event for when it is safe to use the instance
      RtEvent use_event;
      // Unique event to identiy the instance for profiling
      LgEvent unique_event;
      // Whether we own this instance
      // Note if we own the allocation then we must own the instance as well
      // We can own the instance without owning the allocation in the case
      // of external allocations that we don't own but make an instance later
      bool own_instance;
    }; 

    // Small helper functor for adding references to sparsity maps
    

  } // namespace Internal
} // namespace Legion

#endif // __LEGION_FUTURE_IMPL_H__
