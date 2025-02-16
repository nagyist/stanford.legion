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

#ifndef __LEGION_TYPES_H__
#define __LEGION_TYPES_H__

#include <limits>
#include <optional>

#include "legion/api/config.h"
#include "realm.h"
#include "realm/id.h"
#include "realm/dynamic_templates.h"

#ifndef LEGION_DEPRECATED
// This is from before Legion required c++17
#define LEGION_DEPRECATED(x) [[deprecated(x)]]
#endif

// Macros for disabling and re-enabling deprecated warnings
#if defined(__PGIC__)
// PGI has to go first because it also responds to GCC defines
#define LEGION_DISABLE_DEPRECATED_WARNINGS \
  _Pragma("warning (push)") \
  _Pragma("diag_suppress 1445")
#define LEGION_REENABLE_DEPRECATED_WARNINGS \
  _Pragma("warning (pop)")
#elif defined(__GNUC__)
#define LEGION_DISABLE_DEPRECATED_WARNINGS \
  _Pragma("GCC diagnostic push") \
  _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define LEGION_REENABLE_DEPRECATED_WARNINGS \
  _Pragma("GCC diagnostic pop")
#elif defined(__clang__)
#define LEGION_DISABLE_DEPRECATED_WARNINGS \
  _Pragma("clang diagnostic push") \
  _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#define LEGION_REENABLE_DEPRECATED_WARNINGS \
  _Pragma("clang diagnostic pop")
#elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define LEGION_DISABLE_DEPRECATED_WARNINGS \
  _Pragma("warning push") \
  _Pragma("warning disable 1478")
#define LEGION_REENABLE_DEPRECATED_WARNINGS \
  _Pragma("warning pop")
#else
#warning "Don't know how to suppress deprecated warnings for this compiler"
#define LEGION_DISABLE_DEPRECATED_WARNINGS
#define LEGION_REENABLE_DEPRECATED_WARNINGS
#endif

// If we're doing full LEGION_SPY then turn off event pruning
#ifdef LEGION_SPY
#ifndef LEGION_DISABLE_EVENT_PRUNING
#define LEGION_DISABLE_EVENT_PRUNING
#endif
#endif

namespace Legion {

  // Pull C types into the C++ namespace
  typedef ::realm_id_t IDType;
  typedef ::legion_privilege_mode_t PrivilegeMode;
  typedef ::legion_allocate_mode_t AllocateMode;
  typedef ::legion_coherence_property_t CoherenceProperty;
  typedef ::legion_region_flags_t RegionFlags;
  typedef ::legion_projection_type_t ProjectionType;
  typedef ::legion_partition_kind_t PartitionKind;
  typedef ::legion_external_resource_t ExternalResource;
  typedef ::legion_timing_measurement_t TimingMeasurement;
  typedef ::legion_dependence_type_t DependenceType;
  typedef ::legion_mappable_type_id_t MappableType;
  typedef ::legion_file_mode_t LegionFileMode;
  typedef ::legion_execution_constraint_t ExecutionConstraintKind;
  typedef ::legion_layout_constraint_t LayoutConstraintKind;
  typedef ::legion_equality_kind_t EqualityKind;
  typedef ::legion_dimension_kind_t DimensionKind;
  typedef ::legion_isa_kind_t ISAKind;
  typedef ::legion_resource_constraint_t ResourceKind;
  typedef ::legion_launch_constraint_t LaunchKind;
  typedef ::legion_specialized_constraint_t SpecializedKind;
  typedef ::legion_unbounded_pool_scope_t UnboundPoolScope;
  typedef ::legion_projection_type_t HandleType;
  typedef ::legion_address_space_t AddressSpace;
  typedef ::legion_task_priority_t TaskPriority;
  typedef ::legion_task_priority_t RealmPriority;
  typedef ::legion_garbage_collection_priority_t GCPriority;
  typedef ::legion_color_t Color;
  typedef ::legion_field_id_t FieldID;
  typedef ::legion_trace_id_t TraceID;
  typedef ::legion_mapper_id_t MapperID;
  typedef ::legion_context_id_t ContextID;
  typedef ::legion_instance_id_t InstanceID;
  typedef ::legion_index_tree_id_t IndexTreeID;
  typedef ::legion_generation_id_t GenerationID;
  typedef ::legion_type_handle TypeHandle;
  typedef ::legion_projection_id_t ProjectionID;
  typedef ::legion_sharding_id_t ShardingID;
  typedef ::legion_concurrent_id_t ConcurrentID;
  typedef ::legion_region_tree_id_t RegionTreeID;
  typedef ::legion_distributed_id_t DistributedID;
  typedef ::legion_distributed_id_t IndexSpaceID;
  typedef ::legion_address_space_t AddressSpaceID;
  typedef ::legion_tunable_id_t TunableID;
  typedef ::legion_local_variable_id_t LocalVariableID;
  typedef ::legion_mapping_tag_id_t MappingTagID;
  typedef ::legion_semantic_tag_t SemanticTag;
  typedef ::legion_variant_id_t VariantID;
  typedef ::legion_code_descriptor_id_t CodeDescriptorID;
  typedef ::legion_unique_id_t UniqueID;
  typedef ::legion_version_id_t VersionID;
  typedef ::legion_projection_epoch_id_t ProjectionEpochID;
  typedef ::legion_task_id_t TaskID;
  typedef ::legion_layout_constraint_id_t LayoutConstraintID;
  typedef ::legion_shard_id_t ShardID;
  typedef ::legion_provenance_id_t ProvenanceID;
  typedef ::legion_internal_color_t LegionColor;
  typedef ::legion_reduction_op_id_t ReductionOpID;
  typedef ::legion_custom_serdez_id_t CustomSerdezID;
  typedef ::legion_coord_t coord_t;

  // Pull Realm types into the Legion namespace
  typedef Realm::Runtime RealmRuntime;
  typedef Realm::Machine Machine;
  typedef Realm::Memory Memory;
  typedef Realm::Processor Processor;
  typedef Realm::ProcessorGroup ProcessorGroup;
  typedef Realm::CodeDescriptor CodeDescriptor;
  typedef Realm::Reservation Reservation;
  typedef Realm::CompletionQueue CompletionQueue;
  typedef Realm::ReductionOpUntyped ReductionOp;
  typedef Realm::CustomSerdezUntyped SerdezOp;
  typedef Realm::Machine::ProcessorMemoryAffinity ProcessorMemoryAffinity;
  typedef Realm::Machine::MemoryMemoryAffinity MemoryMemoryAffinity;
  typedef Realm::DynamicTemplates::TagType TypeTag;
  typedef Realm::Logger Logger;
  template<int DIM, typename T = coord_t>
  using Point = Realm::Point<DIM,T>;
  template<int DIM, typename T = coord_t>
  using Rect = Realm::Rect<DIM,T>;
  template<int DIM, typename T = coord_t>
  using DomainT = Realm::IndexSpace<DIM,T>;
  template<int M, int N, typename T = coord_t>
  using Transform = Realm::Matrix<M,N,T>;

  // Forward declarations of types in the public interface
  class IndexSpace;
  template<int DIM, typename T> class IndexSpaceT;
  class IndexPartition;
  template<int DIM, typename T> class IndexPartitionT;
  class FieldSpace;
  class LogicalRegion;
  template<int DIM, typename T> class LogicalRegionT;
  class LogicalPartition;
  template<int DIM, typename T> class LogicalPartitionT;
  class IndexAllocator;
  class FieldAllocator;
  class UntypedBuffer;
  class ArgumentMap;
  class Lock;
  struct LockRequest;
  class Grant;
  class PhaseBarrier;
  struct RegionRequirement;
  struct OutputRequirement;
  struct IndexSpaceRequirement;
  struct FieldSpaceRequirement;
  struct TaskLauncher;
  struct IndexTaskLauncher;
  typedef IndexTaskLauncher IndexLauncher; // for backwards compatibility
  struct InlineLauncher;
  struct CopyLauncher;
  struct AcquireLauncher;
  struct ReleaseLauncher;
  struct FillLauncher;
  struct LayoutConstraintRegistrar;
  struct TaskVariantRegistrar;
  struct PoolBounds;
  class Future;
  class FutureMap;
  class Predicate;
  class PhysicalRegion;
  class OutputRegion;
  class ExternalResources;
  class UntypedDeferredValue;
  template<typename>
    class DeferredValue;
  template<typename, bool>
    class DeferredReduction;
  template<typename, int, typename, bool>
    class DeferredBuffer;
  template<typename COORD_T>
    class UntypedDeferredBuffer;
  template<PrivilegeMode,typename,int,typename,typename,bool> 
    class FieldAccessor;
  template<typename, bool, int, typename, typename, bool>
    class ReductionAccessor;
  template<typename, int, typename, typename, bool>
    class PaddingAccessor;
#ifdef LEGION_MULTI_REGION_ACCESSOR
  template<typename, int,typename,typename,bool,bool,int>
    class MultiRegionAccessor;
#endif
  template<typename,int,typename,typename>
    class UnsafeFieldAccessor;
  namespace ArraySyntax {
    template<typename, PrivilegeMode> class AccessorRefHelper;
    template<typename> class AffineRefHelper;
  }
  class PieceIterator;
  template<int,typename>
  class PieceIteratorT;
  template<PrivilegeMode,typename,int,typename>
  class SpanIterator;
  struct InputArgs;
  struct RegistrationCallbackArgs;
  class ProjectionFunctor;
  class ShardingFunctor;
  class Task;
  class Copy;
  class InlineMapping;
  class Acquire;
  class Release;
  class Close;
  class Fill;
  class Partition;
  class MustEpoch;
  class PointTransformFunctor;
  class Runtime;
  class LegionHandshake;
  class MPILegionHandshake;
  // Helper for saving instantiated template functions
  struct SerdezRedopFns;
  // Some typedefs for making things nicer for users with C++11 support
  template<typename FT, int N, typename T = ::legion_coord_t>
  using GenericAccessor = Realm::GenericAccessor<FT,N,T>;
  template<typename FT, int N, typename T = ::legion_coord_t>
  using AffineAccessor = Realm::AffineAccessor<FT,N,T>;
  template<typename FT, int N, typename T = ::legion_coord_t>
  using MultiAffineAccessor = Realm::MultiAffineAccessor<FT,N,T>;
  class LegionTaskWrapper;
  class LegionSerialization;
  class CObjectWrapper;
  class DomainPoint;
  class Domain;
  class ISAConstraint;
  class ProcessorConstraint;
  class ResourceConstraint;
  class LaunchConstraint;
  class ColocationConstraint;
  class ExecutionConstraintSet;
  class SpecializedConstraint;
  class MemoryConstraint;
  class FieldConstraint;
  class PaddingConstraint;
  class OrderingConstraint;
  class TilingConstraint;
  class DimensionConstraint;
  class AlignmentConstraint;
  class OffsetConstraint;
  class PointerConstraint;
  class LayoutConstraintSet;
  class TaskLayoutConstraintSet;
  class Mappable;
  class Task;
  class Copy;
  class Fill;
  class InlineMapping;
  class Acquire;
  class Release;
  class Partition;
  class Close;
  class MustEpoch;

  // A class for preventing serialization of Legion objects
  // which cannot be serialized
  class Unserializable { };
  class Serializer;
  class Deserializer;

  // Typedefs that are needed everywhere
  typedef std::map<CustomSerdezID, 
                   const Realm::CustomSerdezUntyped *> SerdezOpTable;
  typedef std::map<Realm::ReductionOpID, 
                   Realm::ReductionOpUntyped *> ReductionOpTable;
  typedef void (*SerdezInitFnptr)(const ReductionOp*, void *&, size_t&);
  typedef void (*SerdezFoldFnptr)(const ReductionOp*, void *&, 
                                  size_t&, const void*);
  typedef std::map<Realm::ReductionOpID, SerdezRedopFns> SerdezRedopTable;
  
  typedef void (*RegistrationCallbackFnptr)(Machine machine, 
                Runtime *rt, const std::set<Processor> &local_procs);
  typedef void (*RegistrationWithArgsCallbackFnptr)(
                const RegistrationCallbackArgs &args);
  typedef LogicalRegion (*RegionProjectionFnptr)(LogicalRegion parent, 
      const DomainPoint&, Runtime *rt);
  typedef LogicalRegion (*PartitionProjectionFnptr)(LogicalPartition parent, 
      const DomainPoint&, Runtime *rt);
  typedef bool (*PredicateFnptr)(const void*, size_t, 
      const std::vector<Future> futures);
  typedef void (*RealmFnptr)(const void*,size_t,
                             const void*,size_t,Processor);

  // Forward declarations for the mapping namespace
  namespace Mapping {
    class PhysicalInstance;
    class CollectiveView;
    class MapperEvent;
    class ProfilingRequestSet;
    class Mapper;
    class MapperRuntime;
    class AutoLock;
    class DefaultMapper;
    class ShimMapper;
    class TestMapper;
    class DebugMapper;
    class ReplayMapper;

    // The following types are effectively overlaid on the Realm versions
    // to allow for Legion-specific profiling measurements
    enum ProfilingMeasurementID {
      PMID_LEGION_FIRST = Realm::PMID_REALM_LAST,
      PMID_RUNTIME_OVERHEAD,
    };
  } // namespace Mapping

  namespace Internal {
    class LocalLock;
    class AutoLock;
    class AutoTryLock;
    class LgEvent; // base event type for legion
    class ApEvent; // application event
    class ApUserEvent; // application user event
    class ApBarrier; // application barrier
    class RtEvent; // runtime event
    class RtUserEvent; // runtime user event
    class RtBarrier;

    struct RegionUsage; 
    class Collectable;
    class FieldAllocatorImpl;
    class ArgumentMapImpl;
    class FutureImpl;
    class FutureInstance;
    class FutureMapImpl;
    class ReplFutureMapImpl;
    class PhysicalRegionImpl;
    class OutputRegionImpl;
    class ExternalResourcesImpl;
    class PieceIteratorImpl;
    class GrantImpl;
    class PredicateImpl;
    class HandshakeImpl;
    typedef HandshakeImpl LegionHandshakeImpl;
    class MessageManager;
    class ShutdownManager;
    class ProcessorManager;
    class MemoryManager;
    class MemoryPool;
    class VirtualChannel;
    class MessageManager;
    class ShutdownManager;
    class TaskImpl;
    class VariantImpl;
    class LayoutConstraints;
    class ProjectionFunction;
    class ShardingFunction;
    class Runtime; 

    class Provenance;
    class Operation;
    class MemoizableOp;
    class PredicatedOp;
    class MapOp;
    class CopyOp;
    class IndexCopyOp;
    class PointCopyOp;
    class FenceOp;
    class FrameOp;
    class CreationOp;
    class DeletionOp;
    class InternalOp;
    class CloseOp;
    class MergeCloseOp;
    class PostCloseOp;
    class RefinementOp;
    class ResetOp;
    class AcquireOp;
    class ReleaseOp;
    class DynamicCollectiveOp;
    class FuturePredOp;
    class NotPredOp;
    class AndPredOp;
    class OrPredOp;
    class MustEpochOp;
    class PendingPartitionOp;
    class DependentPartitionOp;
    class PointDepPartOp;
    class FillOp;
    class IndexFillOp;
    class PointFillOp;
    class DiscardOp;
    class AttachOp;
    class IndexAttachOp;
    class PointAttachOp;
    class DetachOp;
    class IndexDetachOp;
    class PointDetachOp;
    class TimingOp;
    class TunableOp;
    class AllReduceOp;
    class BeginOp;
    class CompleteOp;
    class RecurrentOp;
    class ExternalMappable;
    class RemoteOp;
    class RemoteMapOp;
    class RemoteCopyOp;
    class RemoteCloseOp;
    class RemoteAcquireOp;
    class RemoteReleaseOp;
    class RemoteFillOp;
    class RemotePartitionOp;
    class RemoteReplayOp;
    class RemoteSummaryOp;
    template<typename OP>
    class Memoizable;
    template<typename OP>
    class Predicated;
    struct PointwiseDependence;

    class ExternalTask;
    class TaskOp;
    class RemoteTaskOp;
    class SingleTask;
    class MultiTask;
    class IndividualTask;
    class PointTask;
    class ShardTask;
    class IndexTask;
    class SliceTask;
    class RemoteTask;

    class TaskContext;
    class InnerContext;
    class TopLevelContext;
    class ReplicateContext;
    class RemoteContext;
    class LeafContext;

    class LogicalTrace;
    class TraceBeginOp;
    class TraceRecurrentOp;
    class TraceCompleteOp;
    class PhysicalTrace;
    class TraceViewSet;
    class TraceConditionSet;
    class PhysicalTemplate;
    class ShardedPhysicalTemplate;
    class Instruction;
    class GetTermEvent;
    class ReplayMapping;
    class CreateApUserEvent;
    class TriggerEvent;
    class MergeEvent;
    class AssignFenceCompletion;
    class IssueCopy;
    class IssueFill;
    class IssueAcross;
    class GetOpTermEvent;
    class SetOpSyncEvent;
    class SetEffects;
    class CompleteReplay;
    class AcquireReplay;
    class ReleaseReplay;
    class BarrierArrival;
    class BarrierAdvance;
    class TraceRecognizer;

    class CopyAcrossExecutor;
    class CopyAcrossUnstructured;
    class IndexSpaceExpression;
    class IndexSpaceExprRef;
    class IndexSpaceOperation;
    template<int DIM, typename T> class IndexSpaceOperationT;
    template<int DIM, typename T> class IndexSpaceUnion;
    template<int DIM, typename T> class IndexSpaceIntersection;
    template<int DIM, typename T> class IndexSpaceDifference;
    class ExpressionTrieNode;
    class IndexTreeNode;
    class IndexSpaceNode;
    template<int DIM, typename T> class IndexSpaceNodeT;
    class IndexPartNode;
    template<int DIM, typename T> class IndexPartNodeT;
    class FieldSpaceNode;
    class RegionTreeNode;
    class RegionNode;
    class PartitionNode;
    class ColorSpaceIterator;
    template<int DIM, typename T> class ColorSpaceLinearizationT;
    class KDTree;
    template<int DIM, typename T, typename RT = void> class KDNode;
    class EqKDTree;
    template<int DIM, typename T> class EqKDTreeT;

    class RegionTreePath;
    class PathTraverser;
    class NodeTraverser;

    class LogicalState;
    class LogicalAnalysis;
    class PhysicalAnalysis;
    class ValidInstAnalysis;
    class InvalidInstAnalysis;
    class AntivalidInstAnalysis;
    class RegistrationAnalysis;
    class CollectiveAnalysis;
    class UpdateAnalysis;
    class CopyAcrossAnalysis;
    class AcquireAnalysis;
    class ReleaseAnalysis;
    class OverwriteAnalysis;
    class FilterAnalysis;
    class EquivalenceSet;
    class EqSetTracker;
    class VersionManager;
    class VersionInfo;
    class ProjectionNode;
    class ProjectionRegion;
    class ProjectionPartition;
    class RefinementTracker;
    class RegionRefinementTracker;
    class PartitionRefinementTracker;
    class CopyFillGuard;
    class CopyFillAggregator;

    class Collectable;
    class Notifiable;
    class ImplicitReferenceTracker;
    class DistributedCollectable;
    class LayoutDescription;
    class InstanceManager; // base class for all instances
    class CopyAcrossHelper;
    class LogicalView; // base class for instance and reduction
    class InstanceKey;
    class InstanceView;
    class ExprView;
    class CollectableView; // pure virtual class
    class IndividualView;
    class CollectiveView;
    class MaterializedView;
    class ReplicatedView;
    class ReductionView;
    class AllreduceView;
    class DeferredView;
    class FillView;
    class PhiView;
    class MappingRef;
    class InstanceRef;
    class InstanceSet;
    class InnerTaskView;
    class VirtualManager;
    class PhysicalManager;
    class InstanceBuilder;

    class RegionAnalyzer;
    class RegionMapper;

    struct LogicalUser;
    struct PhysicalUser;
    struct LogicalTraceInfo;
    struct TraceInfo;
    struct PhysicalTraceInfo;
    struct UniqueInst;
    class TreeCloseImpl;
    class TreeClose;
    struct CloseInfo; 
    struct FieldDataDescriptor;
    class ProjectionSummary;
    class ProjectionInfo;

    class LegionProfiler;
    class LegionProfInstance;

    class MappingCallInfo;
    class MapperManager;
    class SerializingManager;
    class ConcurrentManager; 

    class ShardedMapping;
    class ReplIndividualTask;
    class ReplIndexTask;
    class ReplMergeCloseOp;
    class ReplRefinementOp;
    class ReplResetOp;
    class ReplFillOp;
    class ReplIndexFillOp;
    class ReplDiscardOp;
    class ReplCopyOp;
    class ReplIndexCopyOp;
    class ReplDeletionOp;
    class ReplPendingPartitionOp;
    class ReplDependentPartitionOp;
    class ReplPredicateImpl;
    class ReplMustEpochOp;
    class ReplTimingOp;
    class ReplTunableOp;
    class ReplAllReduceOp;
    class ReplFenceOp;
    class ReplMapOp;
    class ReplAttachOp;
    class ReplIndexAttachOp;
    class ReplDetachOp;
    class ReplIndexDetachOp;
    class ReplAcquireOp;
    class ReplReleaseOp;
    class ReplTraceOp;
    class ReplTraceBeginOp;
    class ReplTraceRecurrentOp;
    class ReplTraceCompleteOp;
    class ShardMapping;
    class CollectiveMapping;
    class ShardManager;
    class ImplicitShardManager;
    class ShardCollective;
    class GatherCollective;
    template<bool>
    class AllGatherCollective;
    template<typename T> class BarrierExchangeCollective;
    template<typename T> class ValueBroadcast;
    template<typename T, bool> class AllReduceCollective;
    class CrossProductCollective;
    class ShardingGatherCollective;
    class FieldDescriptorExchange;
    class FieldDescriptorGather;
    class FutureBroadcast;
    class FutureExchange;
    class FutureNameExchange;
    class MustEpochMappingBroadcast;
    class MustEpochMappingExchange;
    class PredicateCollective;
    class UnorderedExchange;
    class ShardRendezvous;
    class ProjectionTreeExchange;
    class TimeoutMatchExchange;
    class ConcurrentAllreduce;
    class BufferBroadcast;

    // The invalid color
    constexpr LegionColor INVALID_COLOR = std::numeric_limits<LegionColor>::max();
    // This is only needed internally
    typedef Realm::RegionInstance PhysicalInstance;
    typedef Realm::CopySrcDstField CopySrcDstField;
    typedef unsigned long long CollectiveID;
    typedef unsigned long long IndexSpaceExprID;
    struct ContextCoordinate;
    typedef ContextCoordinate TraceLocalID;
    class TaskTreeCoordinates;
    // Helper for encoding templates
    struct NT_TemplateHelper : 
      public Realm::DynamicTemplates::ListProduct2<Realm::DIMCOUNTS, 
                                                   Realm::DIMTYPES> {
    typedef Realm::DynamicTemplates::ListProduct2<Realm::DIMCOUNTS, 
                                                  Realm::DIMTYPES> SUPER;
    public:
      template<int N, typename T> __LEGION_CUDA_HD__
      static inline constexpr TypeTag encode_tag(void) {
#if __cplusplus >= 201402L
        constexpr TypeTag type =
          SUPER::template encode_tag<Realm::DynamicTemplates::Int<N>, T>();
        static_assert(type != 0, "All types should be non-zero for Legion");
        return type;
#else
        return SUPER::template encode_tag<Realm::DynamicTemplates::Int<N>, T>();
#endif
      }
      template<int N, typename T>
      static inline void check_type(const TypeTag t) {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        const TypeTag t1 = encode_tag<N,T>();
#endif
        assert(t1 == t);
#endif
      }
      struct DimHelper {
      public:
        template<typename N, typename T>
        static inline void demux(int *result) { *result = N::N; }
      };
      static inline int get_dim(const TypeTag t) {
        int result = 0;
        SUPER::demux<DimHelper>(t, &result);
        return result; 
      }
    };
    // Pull some of the mapper types into the internal space
    typedef Mapping::Mapper Mapper;
    typedef Mapping::MapperEvent MapperEvent;
    typedef Mapping::PhysicalInstance MappingInstance;
    typedef Mapping::CollectiveView MappingCollective;
    typedef Mapping::ProfilingMeasurementID ProfilingMeasurementID;

    // External global variable references
    // With the exception fo the runtime singleton all of these should
    // only be thread-local or logggers
    // This is the pointer to the runtime singleton
    extern Runtime *runtime;
    // Nasty global variable for TLS support of figuring out
    // our context implicitly
    extern thread_local TaskContext *implicit_context;
    // Mapper context if we're inside of a mapper call
    extern thread_local MappingCallInfo *implicit_mapper_call;
    // Implicit thread-local profiler
    extern thread_local LegionProfInstance *implicit_profiler;
    // Another nasty global variable for tracking the fast
    // reservations that we are holding
    extern thread_local AutoLock *local_lock_list;
    // One more nasty global variable that we use for tracking
    // the provenance of meta-task operations for profiling
    // purposes, this has no bearing on correctness
    extern thread_local ::legion_unique_id_t implicit_provenance;
    // Use this global variable to track name of the "finish event"
    // for whatever (meta-)task we're running on at the moment.
    // It should always be the case that the owner node of the
    // "finish" event" is the same as the node we're on.
    extern thread_local LgEvent implicit_fevent;
    // Use this to track if we're inside of a registration 
    // callback function which we know to be deduplicated
    enum RegistrationCallbackMode {
      NO_REGISTRATION_CALLBACK = 0,
      LOCAL_REGISTRATION_CALLBACK = 1,
      GLOBAL_REGISTRATION_CALLBACK = 2,
    };
    extern thread_local RegistrationCallbackMode inside_registration_callback;
    // This data structure tracks references to any live
    // temporary index space expressions that have been
    // handed back by the region tree inside the execution
    // of a meta-task or a runtime API call. It also tracks
    // changes to remote distributed collectable that can be
    // delayed and batched together.
    extern thread_local ImplicitReferenceTracker *implicit_reference_tracker; 
#ifdef DEBUG_LEGION_CALLERS
    extern thread_local LgTaskID implicit_task_kind;
    extern thread_local LgTaskID implicit_task_caller;
#endif
    extern Realm::Logger log_run;
    extern Realm::Logger log_task;
    extern Realm::Logger log_index;
    extern Realm::Logger log_field;
    extern Realm::Logger log_region;
    extern Realm::Logger log_inst;
    extern Realm::Logger log_variant;
    extern Realm::Logger log_allocation;
    extern Realm::Logger log_migration;
    extern Realm::Logger log_prof;
    extern Realm::Logger log_garbage;
    extern Realm::Logger log_spy;
    extern Realm::Logger log_shutdown;
    extern Realm::Logger log_tracing;
    extern Realm::Logger log_auto_trace;
    extern Realm::Logger log_legion;
  } // namespace Internal

  // Magical typedefs 
  typedef Internal::TaskContext* Context;
  // More magical typedefs for the mapping namespace
  namespace Mapping {
    typedef Internal::MappingCallInfo* MapperContext;
    typedef Internal::InstanceManager* PhysicalInstanceImpl;
    typedef Internal::CollectiveView*  CollectiveViewImpl;
    // This type import is experimental to facilitate coordination and
    // synchronization between different mappers and may be revoked later
    // as we develop new abstractions for mappers to interact
    typedef Internal::LocalLock LocalLock;
  };

  // Events are locks are important enough that we want them inlined everywhere
  // so we pull these classes and implementations into this header file so 
  // that everyone will have access to them.
  namespace Internal {

    // Legion derived event types
    class LgEvent : public Realm::Event {
    public:
      static const LgEvent NO_LG_EVENT;
    public:
      LgEvent(void) noexcept { id = 0; }
      LgEvent(const LgEvent &rhs) = default;
      explicit LgEvent(const Realm::Event e) { id = e.id; }
    public:
      inline LgEvent& operator=(const LgEvent &rhs) = default;
    public:
      // Override the wait method so we can have our own implementation
      inline void wait(void) const;
      inline void wait_faultaware(bool &poisoned, bool from_application) const;
      inline bool is_barrier(void) const;
    protected:
      void begin_context_wait(Context ctx, bool from_application) const;
      void end_context_wait(Context ctx, bool from_application) const;
      void begin_mapper_call_wait(MappingCallInfo *call) const;
      void record_event_wait(LegionProfInstance *profiler, 
                             Realm::Backtrace &bt) const;
      void record_event_trigger(LgEvent precondition) const;
    };

    class PredEvent : public LgEvent {
    public:
      static const PredEvent NO_PRED_EVENT;
    public:
      PredEvent(void) noexcept : LgEvent() { } 
      PredEvent(const PredEvent &rhs) = default;
      explicit PredEvent(const Realm::Event &e) : LgEvent(e) { }
    public:
      inline PredEvent& operator=(const PredEvent &rhs) = default;
    };

    class PredUserEvent : public PredEvent {
    public:
      static const PredUserEvent NO_PRED_USER_EVENT;
    public:
      PredUserEvent(void) noexcept : PredEvent() { }
      PredUserEvent(const PredUserEvent &rhs) = default;
      explicit PredUserEvent(const Realm::UserEvent &e) : PredEvent(e) { }
    public:
      inline PredUserEvent& operator=(const PredUserEvent &rhs) = default;
      inline operator Realm::UserEvent() const
        { Realm::UserEvent e; e.id = id; return e; }
    };

    class ApEvent : public LgEvent {
    public:
      static const ApEvent NO_AP_EVENT;
    public:
      ApEvent(void) noexcept : LgEvent() { }
      ApEvent(const ApEvent &rhs) = default;
      explicit ApEvent(const Realm::Event &e) : LgEvent(e) { }
      explicit ApEvent(const PredEvent &e) { id = e.id; }
    public:
      inline ApEvent& operator=(const ApEvent &rhs) = default;
      inline bool has_triggered_faultignorant(void) const
        { bool poisoned = false; 
          return has_triggered_faultaware(poisoned); }
      inline void wait_faultaware(bool &poisoned) const
        { return LgEvent::wait_faultaware(poisoned, true/*application*/); }
      inline void wait_faultignorant(void) const
        { bool poisoned = false; 
          LgEvent::wait_faultaware(poisoned, true/*application*/); }
    private:
      // Make these private because we always want to be conscious of faults
      // when testing or waiting on application events
      inline bool has_triggered(void) const { return LgEvent::has_triggered(); }
      inline void wait(void) const { LgEvent::wait(); }
    };

    class ApUserEvent : public ApEvent {
    public:
      static const ApUserEvent NO_AP_USER_EVENT;
    public:
      ApUserEvent(void) noexcept : ApEvent() { }
      ApUserEvent(const ApUserEvent &rhs) = default;
      explicit ApUserEvent(const Realm::UserEvent &e) : ApEvent(e) { }
    public:
      inline ApUserEvent& operator=(const ApUserEvent &rhs) = default;
      inline operator Realm::UserEvent() const
        { Realm::UserEvent e; e.id = id; return e; }
    };

    class ApBarrier : public ApEvent {
    public:
      static const ApBarrier NO_AP_BARRIER;
    public:
      ApBarrier(void) noexcept : ApEvent(), timestamp(0) { }
      ApBarrier(const ApBarrier &rhs) = default; 
      explicit ApBarrier(const Realm::Barrier &b) 
        : ApEvent(b), timestamp(b.timestamp) { }
    public:
      inline ApBarrier& operator=(const ApBarrier &rhs) = default;
      inline operator Realm::Barrier() const
        { Realm::Barrier b; b.id = id; 
          b.timestamp = timestamp; return b; }
    public:
      inline bool get_result(void *value, size_t value_size) const
        { Realm::Barrier b; b.id = id;
          b.timestamp = timestamp; return b.get_result(value, value_size); }
      inline void destroy_barrier(void)
        { Realm::Barrier b; b.id = id;
          b.timestamp = timestamp; b.destroy_barrier(); }
    public:
      Realm::Barrier::timestamp_t timestamp;
    };

    class RtEvent : public LgEvent {
    public:
      static const RtEvent NO_RT_EVENT;
    public:
      RtEvent(void) noexcept : LgEvent() { }
      RtEvent(const RtEvent &rhs) = default;
      explicit RtEvent(const Realm::Event &e) : LgEvent(e) { }
      explicit RtEvent(const PredEvent &e) { id = e.id; }
    public:
      inline RtEvent& operator=(const RtEvent &rhs) = default;
    };

    class RtUserEvent : public RtEvent {
    public:
      static const RtUserEvent NO_RT_USER_EVENT;
    public:
      RtUserEvent(void) noexcept : RtEvent() { }
      RtUserEvent(const RtUserEvent &rhs) = default;
      explicit RtUserEvent(const Realm::UserEvent &e) : RtEvent(e) { }
    public:
      inline RtUserEvent& operator=(const RtUserEvent &rhs) = default;
      inline operator Realm::UserEvent() const
        { Realm::UserEvent e; e.id = id; return e; }
    };

    class RtBarrier : public RtEvent {
    public:
      static const RtBarrier NO_RT_BARRIER;
    public:
      RtBarrier(void) noexcept : RtEvent(), timestamp(0) { }
      RtBarrier(const RtBarrier &rhs) = default;
      explicit RtBarrier(const Realm::Barrier &b)
        : RtEvent(b), timestamp(b.timestamp) { }
    public:
      inline RtBarrier& operator=(const RtBarrier &rhs) = default;
      inline operator Realm::Barrier() const
        { Realm::Barrier b; b.id = id; 
          b.timestamp = timestamp; return b; } 
    public:
      inline bool get_result(void *value, size_t value_size) const
        { Realm::Barrier b; b.id = id;
          b.timestamp = timestamp; return b.get_result(value, value_size); }
      inline RtBarrier get_previous_phase(void)
        { Realm::Barrier b; b.id = id;
          return RtBarrier(b.get_previous_phase()); }
      inline void destroy_barrier(void)
        { Realm::Barrier b; b.id = id;
          b.timestamp = timestamp; b.destroy_barrier(); }
    public:
      Realm::Barrier::timestamp_t timestamp;
    }; 

    // Local lock for accelerating lock taking
    class LocalLock {
    public:
      inline LocalLock(void) { } 
    public:
      LocalLock(const LocalLock &rhs) = delete;
      inline ~LocalLock(void) { }
    public:
      LocalLock& operator=(const LocalLock &rhs) = delete;
    private:
      // These are only accessible via AutoLock
      friend class AutoLock;
      friend class AutoTryLock;
      friend class Mapping::AutoLock;
      inline RtEvent lock(void)   { return RtEvent(wrlock()); }
      inline RtEvent wrlock(void) { return RtEvent(reservation.wrlock()); }
      inline RtEvent rdlock(void) { return RtEvent(reservation.rdlock()); }
      inline bool trylock(void) { return reservation.trylock(); }
      inline bool trywrlock(void) { return reservation.trywrlock(); }
      inline bool tryrdlock(void) { return reservation.tryrdlock(); }
      inline void unlock(void) { reservation.unlock(); }
    private:
      inline void advise_sleep_entry(Realm::UserEvent guard)
        { reservation.advise_sleep_entry(guard); }
      inline void advise_sleep_exit(void)
        { reservation.advise_sleep_exit(); }
    protected:
      Realm::FastReservation reservation;
    };

    /////////////////////////////////////////////////////////////
    // AutoLock 
    /////////////////////////////////////////////////////////////
    // An auto locking class for taking a lock and releasing it when
    // the object goes out of scope
    class AutoLock { 
    public:
      inline AutoLock(LocalLock &r, int mode = 0, bool excl = true)
        : local_lock(r), previous(local_lock_list), 
          exclusive(excl), held(true)
      {
#ifdef DEBUG_REENTRANT_LOCKS
        if (previous != NULL)
          previous->check_for_reentrant_locks(&local_lock);
#endif
        if (exclusive)
        {
          RtEvent ready = local_lock.wrlock();
          while (ready.exists())
          {
            ready.wait();
            ready = local_lock.wrlock();
          }
        }
        else
        {
          RtEvent ready = local_lock.rdlock();
          while (ready.exists())
          {
            ready.wait();
            ready = local_lock.rdlock();
          }
        }
        local_lock_list = this;
      }
    protected:
      // Helper constructor for AutoTryLock and Mapping::AutoLock
      inline AutoLock(int mode, bool excl, LocalLock &r)
        : local_lock(r), previous(local_lock_list), 
          exclusive(excl), held(false)
      {
#ifdef DEBUG_REENTRANT_LOCKS
        if (previous != NULL)
          previous->check_for_reentrant_locks(&local_lock);
#endif
      }
    public:
      AutoLock(AutoLock &&rhs) = delete;
      AutoLock(const AutoLock &rhs) = delete;
      inline ~AutoLock(void)
      {
        if (held)
        {
#ifdef DEBUG_LEGION
          assert(local_lock_list == this);
#endif
          local_lock.unlock();
          local_lock_list = previous;
        }
        else
          assert(local_lock_list == previous);
      }
    public:
      AutoLock& operator=(AutoLock &&rhs) = delete;
      AutoLock& operator=(const AutoLock &rhs) = delete;
    public:
      inline void release(void) 
      { 
#ifdef DEBUG_LEGION
        assert(held);
        assert(local_lock_list == this);
#endif
        local_lock.unlock(); 
        local_lock_list = previous;
        held = false; 
      }
      inline void reacquire(void)
      {
#ifdef DEBUG_LEGION
        assert(!held);
        assert(local_lock_list == previous);
#endif
#ifdef DEBUG_REENTRANT_LOCKS
        if (previous != NULL)
          previous->check_for_reentrant_locks(&local_lock);
#endif
        if (exclusive)
        {
          RtEvent ready = local_lock.wrlock();
          while (ready.exists())
          {
            ready.wait();
            ready = local_lock.wrlock();
          }
        }
        else
        {
          RtEvent ready = local_lock.rdlock();
          while (ready.exists())
          {
            ready.wait();
            ready = local_lock.rdlock();
          }
        }
        local_lock_list = this;
        held = true;
      }
    public:
      inline void advise_sleep_entry(Realm::UserEvent guard) const
      {
        if (held)
          local_lock.advise_sleep_entry(guard);
        if (previous != NULL)
          previous->advise_sleep_entry(guard);
      }
      inline void advise_sleep_exit(void) const
      {
        if (held)
          local_lock.advise_sleep_exit();
        if (previous != NULL)
          previous->advise_sleep_exit();
      }
#ifdef DEBUG_REENTRANT_LOCKS
      inline void check_for_reentrant_locks(LocalLock *to_acquire) const
      {
        assert(to_acquire != &local_lock);
        if (previous != NULL)
          previous->check_for_reentrant_locks(to_acquire);
      }
#endif
    protected:
      LocalLock &local_lock;
      AutoLock *const previous;
      const bool exclusive;
      bool held;
    };

    // AutoTryLock is an extension of AutoLock that supports try lock
    class AutoTryLock : public AutoLock {
    public:
      inline AutoTryLock(LocalLock &r, int mode = 0, bool excl = true)
        : AutoLock(mode, excl, r) 
      {
        if (exclusive)
          ready = local_lock.wrlock();
        else
          ready = local_lock.rdlock();
        held = !ready.exists();
        if (held)
          local_lock_list = this;
      }
      AutoTryLock(const AutoTryLock &rhs) = delete;
    public:
      AutoTryLock& operator=(const AutoTryLock &rhs) = delete;
    public:
      // Allow an easy test for whether we got the lock or not
      inline bool has_lock(void) const { return held; }
      inline RtEvent try_next(void) const { return ready; }
    protected:
      RtEvent ready;
    };

    //--------------------------------------------------------------------------
    inline void LgEvent::wait(void) const
    //--------------------------------------------------------------------------
    {
      if (!exists())
        return;
#ifdef DEBUG_LEGION_CALLERS
      LgTaskID local_kind = implicit_task_kind;
      LgTaskID local_caller = implicit_task_caller;
#endif
      // Save the mapper call locally
      MappingCallInfo *local_call = implicit_mapper_call;
      // If we're in a mapper call, notify the mapper that we're waiting
      // Do this first in case we get a re-entrant wait (e.g. because
      // SerializingManager::pause_mapper_call takes and lock and we might
      // end up coming back around here to wait on that event too)
      if (local_call != NULL)
      {
        implicit_mapper_call = NULL;
        begin_mapper_call_wait(local_call);
      }
      // Save whether we are in a registration callback
      RegistrationCallbackMode local_callback = inside_registration_callback;
      // Save the reference tracker that we have
      ImplicitReferenceTracker *local_tracker = implicit_reference_tracker;
      implicit_reference_tracker = NULL;
      LegionProfInstance *local_profiler = implicit_profiler;
      if (local_profiler != NULL)
      {
        // Cannot call back into wait while recording a wait
        implicit_profiler = NULL;
        Realm::Backtrace bt;
        bt.capture_backtrace();
        record_event_wait(local_profiler, bt);
      }
      // Save the context locally
      TaskContext *local_ctx = implicit_context; 
      implicit_context = NULL; 
      // Save the implicit fevent
      LgEvent local_fevent = implicit_fevent;
      implicit_fevent = LgEvent::NO_LG_EVENT;
      // Save the task provenance information
      UniqueID local_provenance = implicit_provenance;
      implicit_provenance = 0;
      // Check to see if we have any local locks to notify
      if (local_lock_list != NULL)
      {
        // Make a copy of the local locks here
        AutoLock *local_lock_list_copy = local_lock_list;
        // Set this back to NULL until we are done waiting
        local_lock_list = NULL;
        // Make a user event and notify all the thread locks
        const Realm::UserEvent done = Realm::UserEvent::create_user_event();
        local_lock_list_copy->advise_sleep_entry(done);
        if (local_ctx != NULL)
          begin_context_wait(local_ctx, false/*from application*/); 
        // Now we can do the wait
        if (!Processor::get_executing_processor().exists())
          Realm::Event::external_wait();
        else
          Realm::Event::wait();
        if (local_ctx != NULL)
          end_context_wait(local_ctx, false/*from application*/);
        // When we wake up, notify that we are done and exited the wait
        local_lock_list_copy->advise_sleep_exit();
        // If we're profiling we need to record that we triggered this
        // event as it will help us hook up the critical path for
        // local lock acquires
        if (local_profiler != NULL)
        {
          // Make sure to restore this before recording
          implicit_fevent = local_fevent;
          implicit_profiler = local_profiler;
          const LgEvent to_trigger(done);
          to_trigger.record_event_trigger(LgEvent::NO_LG_EVENT);
        }
        // Trigger the user-event
        done.trigger();
        // Restore our local lock list
        local_lock_list = local_lock_list_copy; 
      }
      else // Just do the normal wait
      {
        if (local_ctx != NULL)
          begin_context_wait(local_ctx, false/*from application*/);
        if (!Processor::get_executing_processor().exists())
          Realm::Event::external_wait();
        else
          Realm::Event::wait();
        if (local_ctx != NULL)
          end_context_wait(local_ctx, false/*from application*/);
      }
      // Write the context back
      implicit_context = local_ctx;
      // Write the mapper call back
      implicit_mapper_call = local_call;
      // Write the implicit fevent back
      implicit_fevent = local_fevent;
      // Write the provenance information back
      implicit_provenance = local_provenance;
      // Restore the local profiler
      implicit_profiler = local_profiler;
#ifdef DEBUG_LEGION_CALLERS
      implicit_task_kind = local_kind;
      implicit_task_caller = local_caller;
#endif
      // Write the registration callback information back
      inside_registration_callback = local_callback;
#ifdef DEBUG_LEGION
      assert(implicit_reference_tracker == NULL);
#endif
      // Write the local reference tracker back
      implicit_reference_tracker = local_tracker;
    }

    //--------------------------------------------------------------------------
    inline void LgEvent::wait_faultaware(bool &poisoned, bool from_app) const
    //--------------------------------------------------------------------------
    {
      if (!exists())
        return;
      if (has_triggered_faultaware(poisoned))
        return;
#ifdef DEBUG_LEGION_CALLERS
      LgTaskID local_kind = implicit_task_kind;
      LgTaskID local_caller = implicit_task_caller;
#endif
      // Save the mapper call locally
      MappingCallInfo *local_call = implicit_mapper_call;
      // If we're in a mapper call, notify the mapper that we're waiting
      // Do this first in case we get a re-entrant wait (e.g. because
      // SerializingManager::pause_mapper_call takes and lock and we might
      // end up coming back around here to wait on that event too)
      if (local_call != NULL)
      {
        implicit_mapper_call = NULL;
        begin_mapper_call_wait(local_call);
      }
      // Save whether we are in a registration callback
      RegistrationCallbackMode local_callback = inside_registration_callback;
      // Save the reference tracker that we have
      ImplicitReferenceTracker *local_tracker = implicit_reference_tracker;
      implicit_reference_tracker = NULL;
      LegionProfInstance *local_profiler = implicit_profiler;
      if (local_profiler != NULL)
      {
        // Cannot call back into wait while recording a wait
        implicit_profiler = NULL;
        Realm::Backtrace bt;
        bt.capture_backtrace();
        record_event_wait(local_profiler, bt);
      }
      // Save the context locally
      TaskContext *local_ctx = implicit_context; 
      implicit_context = NULL;
      // Save the fevent
      LgEvent local_fevent = implicit_fevent;
      implicit_fevent = LgEvent::NO_LG_EVENT;
      // Save the task provenance information
      UniqueID local_provenance = implicit_provenance;
      implicit_provenance = 0;
      // Check to see if we have any local locks to notify
      if (local_lock_list != NULL)
      {
        // Make a copy of the local locks here
        AutoLock *local_lock_list_copy = local_lock_list;
        // Set this back to NULL until we are done waiting
        local_lock_list = NULL;
        // Make a user event and notify all the thread locks
        const Realm::UserEvent done = Realm::UserEvent::create_user_event();
        local_lock_list_copy->advise_sleep_entry(done);
        if (local_ctx != NULL)
          begin_context_wait(local_ctx, from_app);
        // Now we can do the wait
        if (!Processor::get_executing_processor().exists())
          Realm::Event::external_wait_faultaware(poisoned);
        else
          Realm::Event::wait_faultaware(poisoned);
        if (local_ctx != NULL)
          end_context_wait(local_ctx, from_app);
        // When we wake up, notify that we are done and exited the wait
        local_lock_list_copy->advise_sleep_exit();
        // If we're profiling we need to record that we triggered this
        // event as it will help us hook up the critical path for
        // local lock acquires
        if (local_profiler != NULL)
        {
          // Make sure to restore this before recording
          implicit_fevent = local_fevent;
          implicit_profiler = local_profiler;
          const LgEvent to_trigger(done);
          to_trigger.record_event_trigger(LgEvent::NO_LG_EVENT);
        }
        // Trigger the user-event
        done.trigger();
        // Restore our local lock list
        local_lock_list = local_lock_list_copy; 
      }
      else // Just do the normal wait
      {
        if (local_ctx != NULL)
          begin_context_wait(local_ctx, from_app);
        if (!Processor::get_executing_processor().exists())
          Realm::Event::external_wait_faultaware(poisoned);
        else
          Realm::Event::wait_faultaware(poisoned);
        if (local_ctx != NULL)
          end_context_wait(local_ctx, from_app);
      }
      // Write the context back
      implicit_context = local_ctx;
      // Write the mapper call back
      implicit_mapper_call = local_call;
      // Write the implicit fevent back
      implicit_fevent = local_fevent;
      // Write the provenance information back
      implicit_provenance = local_provenance;
      // Restore the local profiler
      implicit_profiler = local_profiler;
#ifdef DEBUG_LEGION_CALLERS
      implicit_task_kind = local_kind;
      implicit_task_caller = local_caller;
#endif
      // Write the registration callback information back
      inside_registration_callback = local_callback;
#ifdef DEBUG_LEGION
      assert(implicit_reference_tracker == NULL);
#endif
      // Write the local reference tracker back
      implicit_reference_tracker = local_tracker;
    }

    //--------------------------------------------------------------------------
    inline bool LgEvent::is_barrier(void) const
    //--------------------------------------------------------------------------
    {
      const Realm::ID identity(id);
      return identity.is_barrier();
    }

  } // namespace Internal
} // namespace Legion

#define FRIEND_ALL_RUNTIME_CLASSES                          \
    friend class Legion::Runtime;                           \
    friend class Legion::Mapping::MapperRuntime;            \
    friend class Internal::Runtime;                         \
    friend class Internal::FutureImpl;                      \
    friend class Internal::FutureMapImpl;                   \
    friend class Internal::PhysicalRegionImpl;              \
    friend class Internal::ExternalResourcesImpl;           \
    friend class Internal::TaskImpl;                        \
    friend class Internal::VariantImpl;                     \
    friend class Internal::ProcessorManager;                \
    friend class Internal::MemoryManager;                   \
    friend class Internal::Operation;                       \
    friend class Internal::PredicatedOp;                    \
    friend class Internal::MapOp;                           \
    friend class Internal::CopyOp;                          \
    friend class Internal::IndexCopyOp;                     \
    friend class Internal::PointCopyOp;                     \
    friend class Internal::FenceOp;                         \
    friend class Internal::DynamicCollectiveOp;             \
    friend class Internal::FuturePredOp;                    \
    friend class Internal::CreationOp;                      \
    friend class Internal::DeletionOp;                      \
    friend class Internal::CloseOp;                         \
    friend class Internal::MergeCloseOp;                    \
    friend class Internal::PostCloseOp;                     \
    friend class Internal::RefinementOp;                    \
    friend class Internal::ResetOp;                         \
    friend class Internal::AcquireOp;                       \
    friend class Internal::ReleaseOp;                       \
    friend class Internal::PredicateImpl;                   \
    friend class Internal::NotPredOp;                       \
    friend class Internal::AndPredOp;                       \
    friend class Internal::OrPredOp;                        \
    friend class Internal::MustEpochOp;                     \
    friend class Internal::PendingPartitionOp;              \
    friend class Internal::DependentPartitionOp;            \
    friend class Internal::PointDepPartOp;                  \
    friend class Internal::FillOp;                          \
    friend class Internal::IndexFillOp;                     \
    friend class Internal::PointFillOp;                     \
    friend class Internal::DiscardOp;                       \
    friend class Internal::AttachOp;                        \
    friend class Internal::IndexAttachOp;                   \
    friend class Internal::ReplIndexAttachOp;               \
    friend class Internal::PointAttachOp;                   \
    friend class Internal::DetachOp;                        \
    friend class Internal::IndexDetachOp;                   \
    friend class Internal::ReplIndexDetachOp;               \
    friend class Internal::PointDetachOp;                   \
    friend class Internal::TimingOp;                        \
    friend class Internal::TunableOp;                       \
    friend class Internal::AllReduceOp;                     \
    friend class Internal::TraceRecurrentOp;                \
    friend class Internal::ExternalMappable;                \
    friend class Internal::ExternalTask;                    \
    friend class Internal::TaskOp;                          \
    friend class Internal::SingleTask;                      \
    friend class Internal::MultiTask;                       \
    friend class Internal::IndividualTask;                  \
    friend class Internal::PointTask;                       \
    friend class Internal::IndexTask;                       \
    friend class Internal::SliceTask;                       \
    friend class Internal::ReplIndividualTask;              \
    friend class Internal::ReplIndexTask;                   \
    friend class Internal::ReplFillOp;                      \
    friend class Internal::ReplIndexFillOp;                 \
    friend class Internal::ReplDiscardOp;                   \
    friend class Internal::ReplCopyOp;                      \
    friend class Internal::ReplIndexCopyOp;                 \
    friend class Internal::ReplDeletionOp;                  \
    friend class Internal::ReplPendingPartitionOp;          \
    friend class Internal::ReplDependentPartitionOp;        \
    friend class Internal::ReplMustEpochOp;                 \
    friend class Internal::ReplMapOp;                       \
    friend class Internal::ReplTimingOp;                    \
    friend class Internal::ReplTunableOp;                   \
    friend class Internal::ReplAllReduceOp;                 \
    friend class Internal::ReplFenceOp;                     \
    friend class Internal::ReplAttachOp;                    \
    friend class Internal::ReplDetachOp;                    \
    friend class Internal::ReplAcquireOp;                   \
    friend class Internal::ReplReleaseOp;                   \
    friend class Internal::MemoizableOp;                    \
    template<typename OP>                                   \
    friend class Internal::Memoizable;                      \
    friend class Internal::ShardManager;                    \
    friend class Internal::IndexSpaceNode;                  \
    template<int, typename>                                 \
    friend class Internal::IndexSpaceNodeT;                 \
    friend class Internal::IndexPartNode;                   \
    friend class Internal::FieldSpaceNode;                  \
    friend class Internal::RegionTreeNode;                  \
    friend class Internal::RegionNode;                      \
    friend class Internal::PartitionNode;                   \
    friend class Internal::LogicalView;                     \
    friend class Internal::InstanceView;                    \
    friend class Internal::DeferredView;                    \
    friend class Internal::ReductionView;                   \
    friend class Internal::MaterializedView;                \
    friend class Internal::FillView;                        \
    friend class Internal::LayoutDescription;               \
    friend class Internal::InstanceManager;                 \
    friend class Internal::PhysicalManager;                 \
    friend class Internal::MapperManager;                   \
    friend class Internal::InstanceRef;                     \
    friend class Internal::HandshakeImpl;                   \
    friend class Internal::ArgumentMapImpl;                 \
    friend class Internal::FutureMapImpl;                   \
    friend class Internal::ReplFutureMapImpl;               \
    friend class Internal::TaskContext;                     \
    friend class Internal::InnerContext;                    \
    friend class Internal::TopLevelContext;                 \
    friend class Internal::RemoteContext;                   \
    friend class Internal::LeafContext;                     \
    friend class Internal::ReplicateContext;                \
    friend class Internal::InstanceBuilder;                 \
    friend class Internal::FutureNameExchange;              \
    friend class Internal::MustEpochMappingExchange;        \
    friend class Internal::MustEpochMappingBroadcast;       \
    friend class Internal::MappingCallInfo;                 \
    friend class CObjectWrapper;

#endif // __LEGION_TYPES_H__
