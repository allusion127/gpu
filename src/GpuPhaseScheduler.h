#pragma once

// Minimal case-phase queue + classify/compact core -- Rev.7.1 Task 3, W0-scoped.
//
// WHAT W0 CUT, AND WHY THIS FILE IS SMALL.  The offline scheduler replay
// (Sec 8.8, promoted to W0) FAILED at fleet = 64: the projected idle/tail win
// did not survive at the width the campaign actually runs.  So the two-level
// scheduler is built for the part that DID hold -- immediate slot refill
// (Sec 8.2) and the M1 flux-segment boundary -- and nothing else.  Concretely,
// what is deliberately NOT here:
//
//   * no fairness machinery.  No max-age promotion, no cost-class scoring, no
//     starvation credit.  Phase selection is a fixed-order scan picking the
//     single largest ready queue.  The extension point is marked below; do not
//     grow it before a replay says the width is worth it.
//   * no persistent / cooperative anything.  W0 measured c_barrier = 0.78 us
//     against the 0.384 us kill threshold (constraint 17), so the persistent
//     case-phase scheduler of Sec 5.7 is permanently closed.  There is no
//     cooperative-groups helper here and there must never be one.
//
// WHAT LEVEL-1 IS ALLOWED TO TOUCH.  `DeviceSlotPhase` and nothing else.  Not
// DeviceSlotState, not DeviceSearchState, not DeviceScheduleParams, and
// obviously no bulk array.  At 64 slots that is 2 KiB, which stays in L1 across
// the ~68k epochs a single deck spends in the scheduler; the 128-byte packet
// Rev.7 proposed would be 8 KiB and would re-fetch through the same L2 that
// Sec 3.6 already names as the real risk.  The contract test enforces this by
// grep, because "it only reads the hot struct" is not visible in any timing.
//
// ORDERING IS PART OF THE CONTRACT.  A queue holds ascending physical slot ids,
// padded with -1.  Ascending is not cosmetic: a phase kernel launched over a
// queue reads slot arrays in queue order, so a queue whose order depended on
// warp scheduling would make the memory access pattern -- and therefore any
// reduction written in queue order -- nondeterministic between runs.  The
// device kernel's warp-ballot compaction and the serial reference below are
// required to produce the IDENTICAL array; that equivalence is what
// test/gpu_phase_compaction.cpp checks, on the reference, without a GPU.
//
// DISPATCH COST.  W0: c_dispatch = 0.783 us per launch, conditional-graph
// switch evaluation = 5.13 us.  Those two numbers are why the quantum table
// below is what it is; see kQuantumCmfdEscalationNote.

#include "GpuSlotControl.h"

#include <cstddef>
#include <cstdint>

namespace rasbery::gpu {

/// Sec 5.9: physical slots are a fixed, dense range.  The cap is defined once,
/// in GpuSlotControl.h, so the arena's control block and the scheduler's queues
/// cannot disagree about how wide the fleet may be.
inline constexpr int kMaxSchedulerSlots = kMaxDeviceSlots;

/// Padding value in every queue.  A lane that reads one has no work; it must
/// return, never dereference.
inline constexpr int kQueueEmptySlot = -1;

// ---------------------------------------------------------------------------
// Sec 5.5  Conditional-graph bucket set
// ---------------------------------------------------------------------------

/// The widths a conditional-graph body may be instantiated at.  A phase runs at
/// the smallest bucket that covers its ready count, and the lanes between the
/// count and the bucket are padding (gpuDispatchIsPadding below).
///
/// Buckets exist because a graph bakes its launch dimensions: without them
/// every distinct ready count would need its own captured body.  Nine buckets
/// cover 1..64 with at most 50% padding waste (24 -> 32 is the worst case).
inline constexpr int kDispatchBuckets[]   = {1, 2, 4, 8, 16, 24, 32, 48, 64};
inline constexpr int kDispatchBucketCount = 9;

/// Smallest configured bucket >= count.  Returns 0 for an empty queue and -1
/// when the count exceeds the widest bucket, which is a configuration error
/// (kMaxSchedulerSlots and the last bucket must agree).
///
/// Written as a ladder rather than a loop over kDispatchBuckets because DEVICE
/// CODE CANNOT TAKE THE ADDRESS OF A NAMESPACE-SCOPE CONSTEXPR ARRAY -- the same
/// nvcc constraint XsReconKernel.h documents for ACTIVE_XT.  The array above is
/// the readable, host-iterable spelling of this ladder, and bucketLadderAgrees()
/// asserts the two are the same ladder at compile time, so the duplication is
/// checked rather than merely commented.
RASBERY_GPU_HD constexpr int gpuSelectBucket(int count) {
    if (count <= 0) return 0;
    if (count <= 1) return 1;
    if (count <= 2) return 2;
    if (count <= 4) return 4;
    if (count <= 8) return 8;
    if (count <= 16) return 16;
    if (count <= 24) return 24;
    if (count <= 32) return 32;
    if (count <= 48) return 48;
    if (count <= 64) return 64;
    return -1;
}

namespace detail {
/// Every table entry is a fixed point of the ladder, every value one past a
/// bucket rounds up to the next, and one past the widest is the error code.
constexpr bool bucketLadderAgrees() {
    for (int i = 0; i < kDispatchBucketCount; ++i) {
        if (gpuSelectBucket(kDispatchBuckets[i]) != kDispatchBuckets[i]) return false;
        if (i > 0 && gpuSelectBucket(kDispatchBuckets[i - 1] + 1) != kDispatchBuckets[i])
            return false;
    }
    return gpuSelectBucket(kDispatchBuckets[kDispatchBucketCount - 1] + 1) == -1;
}
} // namespace detail

static_assert(detail::bucketLadderAgrees(),
              "kDispatchBuckets and gpuSelectBucket describe different ladders");
static_assert(kDispatchBuckets[kDispatchBucketCount - 1] == kMaxSchedulerSlots,
              "the widest bucket must cover every slot");

/// Graph-path padding guard.  A conditional body is launched at the BUCKET
/// width, so every phase kernel begins with:
///
///     if (gpuDispatchIsPadding(logical, q.count)) return;
///
/// `logical` is the lane's index into the queue, NOT a slot id.  The check is
/// `logical >= active_count` and nothing more -- a padding lane must not read
/// the queue at all, because the value there is kQueueEmptySlot.
RASBERY_GPU_HD inline bool gpuDispatchIsPadding(int logical, int active_count) {
    return logical >= active_count;
}

// ---------------------------------------------------------------------------
// Sec 8.3  Queue shape
// ---------------------------------------------------------------------------

/// One phase's ready set.  `slots[0..count)` are ascending physical slot ids;
/// `slots[count..bucket)` are kQueueEmptySlot; the rest is untouched.
struct DevicePhaseQueue {
    int slots[kMaxSchedulerSlots];
    int count;  ///< logical ready count
    int bucket; ///< dispatched width, gpuSelectBucket(count)
};

/// Fault bits.  Sec 5.2 makes two of these FATAL rather than recoverable: a
/// duplicate insertion means two graph bodies would drive one slot's state, and
/// a requeue of an in-flight slot means one is already driving it.  Neither can
/// be retried into correctness, so the scheduler records and stops.
inline constexpr std::uint32_t kSchedFaultNone            = 0u;
inline constexpr std::uint32_t kSchedFaultDuplicateQueue  = 0x1u;
inline constexpr std::uint32_t kSchedFaultInFlightRequeue = 0x2u;
inline constexpr std::uint32_t kSchedFaultSlotOverflow    = 0x4u;
inline constexpr std::uint32_t kSchedFaultBadPhase        = 0x8u;

/// `fault_slot` when nothing faulted.  Not 0 -- slot 0 is a real slot, and a
/// sentinel that collides with a valid id is the kind of bug that reports the
/// wrong slot exactly when it matters.  Both the serial reference and the
/// device kernel take the SMALLEST offending slot (atomicMin on the device),
/// so the receipt names the first failure in slot order, not in warp order.
inline constexpr std::uint32_t kSchedNoFaultSlot = 0xFFFFFFFFu;

/// Every fault this scheduler can raise is fatal -- there is no recoverable
/// class.  A duplicate insert or an in-flight requeue means two bodies would
/// drive one slot; an overflow means work was dropped; a bad phase word means
/// the slot's state is not interpretable.  None of those becomes correct on a
/// retry, so the consumer below fails the slot rather than continuing.
RASBERY_GPU_HD inline bool gpuSchedulerFaultIsFatal(std::uint32_t flags) {
    return (flags & (kSchedFaultDuplicateQueue | kSchedFaultInFlightRequeue |
                     kSchedFaultSlotOverflow | kSchedFaultBadPhase)) != 0u;
}

/// THE CONSUMER.  Without this the fault bits were written to a struct nobody
/// read: classify would flag a duplicate insertion and the offending slot would
/// keep running, which is exactly the corruption the check exists to stop.
///
/// Failing the slot -- not the fleet -- is the Sec 9.2 policy: one case dies,
/// the other 63 keep going, and the receipt names what happened.  Bumping
/// state_epoch is what invalidates whatever stale queue entry caused the fault.
RASBERY_GPU_HD inline void gpuMarkSlotFailed(DeviceSlotPhase& p, std::uint32_t flags) {
    p.error_code = flags;
    p.flags      = static_cast<std::uint8_t>((p.flags | kSlotFlagFatal) & ~kSlotFlagInFlight);
    p.phase      = static_cast<std::uint8_t>(DevicePhase::Failed);
    ++p.state_epoch;
}

/// Everything one classify epoch produces.  Written once per epoch; read by the
/// dispatch path and by the receipt.
struct DevicePhaseQueues {
    DevicePhaseQueue queue[kDevicePhaseCount];
    int              phase_count[kDevicePhaseCount];

    int selected_phase;  ///< DevicePhase chosen for this epoch, or -1 when idle
    int selected_count;
    int selected_bucket;
    int active_count;    ///< slots carrying kSlotFlagActive
    int free_count;      ///< Done/Failed/Empty slots, the refill kernel's input

    std::uint32_t fault_flags;
    std::uint32_t fault_slot; ///< first offending slot, for the receipt
};

static_assert(std::is_trivially_copyable_v<DevicePhaseQueue>);
static_assert(std::is_trivially_copyable_v<DevicePhaseQueues>);

// ---------------------------------------------------------------------------
// Sec 5.4  Phase selection -- FIXED ORDER ONLY (fairness deferred, see below)
// ---------------------------------------------------------------------------

/// Which phases are schedulable at all.  Empty / Done / Failed are not: a slot
/// in one of those is the REFILL kernel's business (Sec 8.2), not a queue's.
RASBERY_GPU_HD inline bool gpuPhaseIsSchedulable(DevicePhase p) {
    return p != DevicePhase::Empty && p != DevicePhase::Done && p != DevicePhase::Failed;
}

/// The scan order, which is a TIE-BREAK and nothing else: selection picks the
/// largest ready queue and consults this order only when two queues tie.
///
/// Rev.7.1 Sec 3.1 has no separate CMFD and Nodal phases -- one `Outer` covers
/// the CMFD sweep and the nodal correction, which is exactly the quantum
/// (kOuterQuantumSteps below).  So the W0-scoped order
/// "CMFD -> Nodal -> Material/Xe -> TH/Search -> Depletion -> Output" maps onto
/// the enum as:
///
///   Outer                      CMFD sweep + nodal correction, one outer
///   Material                   XS rebuild after a committed perturbation
///   Xenon, ThermalHydraulics, Search
///   NormalizeFluxSign, Derivative, RodOp     statepoint-level host remnants
///   DepletionPredictor, DepletionCorrector
///   Ppr, ResultAggregate, OutputPack
///   Import                     LAST on purpose: a tie between Import and Outer
///                              should go to Outer, which keeps the pipeline
///                              full rather than admitting more work into it.
inline constexpr DevicePhase kPhaseScanOrder[] = {
    DevicePhase::Outer,
    DevicePhase::Material,
    DevicePhase::Xenon,
    DevicePhase::ThermalHydraulics,
    DevicePhase::Search,
    DevicePhase::NormalizeFluxSign,
    DevicePhase::Derivative,
    DevicePhase::RodOp,
    DevicePhase::DepletionPredictor,
    DevicePhase::DepletionCorrector,
    DevicePhase::Ppr,
    DevicePhase::ResultAggregate,
    DevicePhase::OutputPack,
    DevicePhase::Import,
};
inline constexpr int kPhaseScanOrderCount =
    static_cast<int>(sizeof(kPhaseScanOrder) / sizeof(kPhaseScanOrder[0]));

/// The same order as a function, for the same nvcc reason as gpuSelectBucket:
/// device code cannot take the address of kPhaseScanOrder.  scanOrderAgrees()
/// asserts the two spellings match at compile time.
RASBERY_GPU_HD constexpr DevicePhase gpuPhaseScanAt(int i) {
    switch (i) {
        case 0:  return DevicePhase::Outer;
        case 1:  return DevicePhase::Material;
        case 2:  return DevicePhase::Xenon;
        case 3:  return DevicePhase::ThermalHydraulics;
        case 4:  return DevicePhase::Search;
        case 5:  return DevicePhase::NormalizeFluxSign;
        case 6:  return DevicePhase::Derivative;
        case 7:  return DevicePhase::RodOp;
        case 8:  return DevicePhase::DepletionPredictor;
        case 9:  return DevicePhase::DepletionCorrector;
        case 10: return DevicePhase::Ppr;
        case 11: return DevicePhase::ResultAggregate;
        case 12: return DevicePhase::OutputPack;
        case 13: return DevicePhase::Import;
        default: break;
    }
    return DevicePhase::Empty; // never schedulable, so an out-of-range index is inert
}

namespace detail {
constexpr bool scanOrderAgrees() {
    for (int i = 0; i < kPhaseScanOrderCount; ++i)
        if (gpuPhaseScanAt(i) != kPhaseScanOrder[i]) return false;
    return gpuPhaseScanAt(kPhaseScanOrderCount) == DevicePhase::Empty;
}
} // namespace detail

static_assert(detail::scanOrderAgrees(),
              "kPhaseScanOrder and gpuPhaseScanAt describe different orders");

// === FAIRNESS EXTENSION POINT ==============================================
//
// DEFERRED BY THE W0 VERDICT.  The replay failed at fleet = 64, so max-age
// promotion and cost-class scoring buy nothing measurable today and would add
// per-slot state to the hot 32-byte struct to pay for it.  `phase_age` is
// already maintained in DeviceSlotPhase, so the inputs exist when they are
// wanted.
//
// To add fairness, replace gpuSelectPhase's body -- and ONLY its body.  Nothing
// else in this header, in the kernels, or in the queue shape depends on how the
// phase is chosen; selection is a pure function of the phase counts.  The
// order it must beat is: largest ready queue, ties by kPhaseScanOrder.
//
// Do not add a fairness field to DeviceSlotPhase without re-checking the 32-byte
// assert: `reserved` is the only spare word.
// ===========================================================================

/// Largest ready queue; ties resolved by kPhaseScanOrder position.  Returns -1
/// when nothing is ready.
RASBERY_GPU_HD inline int gpuSelectPhase(const int* phase_count) {
    int best_phase = -1;
    int best_count = 0;
    for (int i = 0; i < kPhaseScanOrderCount; ++i) {
        const int p = static_cast<int>(gpuPhaseScanAt(i));
        if (phase_count[p] > best_count) {
            best_count = phase_count[p];
            best_phase = p;
        }
    }
    return best_phase;
}

// ---------------------------------------------------------------------------
// Task 3 Step 6  Phase quantum table
// ---------------------------------------------------------------------------

/// How much work one dispatch of a phase does.  This is the unit the scheduler
/// epoch is measured in, so it decides how many epochs a deck costs.
enum class PhaseQuantum : std::uint32_t {
    None = 0,
    OneOuter,          ///< CMFD sweep + nodal correction, once (Sec 5.1 default)
    RemainingBudget,   ///< CMFD escalation: the whole remaining sweep budget
    FullPhaseChain,    ///< nodal trl0 -> trl12 -> MatEven -> Jnet
    OneStep,           ///< Xe, TH
    OneTrial,          ///< critical search
    OneStage,          ///< depletion predictor or corrector
    OneCornerIteration ///< PPR corner balance
};

/// CMFD quantum modes, both kept so the escalation is a flag flip and not a
/// rewrite (constraint 15 rollback preservation).  Selected by
/// RASBERY_GPU_PHASE_QUANTUM_CMFD = "outer" (default) | "budget".
enum class CmfdQuantumMode : std::uint32_t { OneOuter = 0, RemainingBudget = 1 };

/// Sec 5.1(2) escalation clause, with the W0 numbers substituted.
///
///     epochs per deck        ~= 68,000     (one epoch per CMFD outer)
///     conditional switch eval = 5.13 us    (W0 spike (3), measured)
///     control cost            = 68,000 x 5.13 us = 0.349 s
///     v2 baseline single wall = 55.35 s
///     -> 0.63% of wall, against the constraint-32 ceiling of 3%
///
/// So the DEFAULT SURVIVES: CMFD quantum stays at one outer.  The escalation to
/// "remaining sweep budget" is retained behind the flag because the margin is a
/// factor of ~4.7, not a factor of 100 -- a wider fleet or a cheaper solver
/// moves the ratio, and the promotion additionally requires tail_efficiency to
/// regress by no more than 3 percentage points.
inline constexpr double kW0SwitchEvalMicros   = 5.13;
inline constexpr double kW0DispatchMicros     = 0.783;
inline constexpr double kQuantumEpochBudgetFraction = 0.03;
inline constexpr const char* kQuantumCmfdEscalationNote =
    "CMFD quantum = one outer; escalate to remaining-budget only when "
    "68000 * switch_eval(5.13us) exceeds 3% of solver wall AND tail_efficiency "
    "regresses <= 3 percentage points";

/// The escalation predicate itself, so the contract test and the runtime read
/// the same arithmetic rather than two copies of it.
inline constexpr bool gpuCmfdQuantumShouldEscalate(double epochs, double epoch_cost_seconds,
                                                   double solver_wall_seconds) {
    return epochs * epoch_cost_seconds > kQuantumEpochBudgetFraction * solver_wall_seconds;
}

RASBERY_GPU_HD inline PhaseQuantum gpuPhaseQuantum(DevicePhase p, CmfdQuantumMode cmfd) {
    switch (p) {
        case DevicePhase::Outer:
            return cmfd == CmfdQuantumMode::RemainingBudget ? PhaseQuantum::RemainingBudget
                                                            : PhaseQuantum::OneOuter;
        case DevicePhase::Material:           return PhaseQuantum::FullPhaseChain;
        case DevicePhase::Xenon:              return PhaseQuantum::OneStep;
        case DevicePhase::ThermalHydraulics:  return PhaseQuantum::OneStep;
        case DevicePhase::Search:             return PhaseQuantum::OneTrial;
        case DevicePhase::DepletionPredictor: return PhaseQuantum::OneStage;
        case DevicePhase::DepletionCorrector: return PhaseQuantum::OneStage;
        case DevicePhase::Ppr:                return PhaseQuantum::OneCornerIteration;
        case DevicePhase::Import:
        case DevicePhase::NormalizeFluxSign:
        case DevicePhase::Derivative:
        case DevicePhase::RodOp:
        case DevicePhase::ResultAggregate:
        case DevicePhase::OutputPack:         return PhaseQuantum::FullPhaseChain;
        default:                              return PhaseQuantum::None;
    }
}

// ---------------------------------------------------------------------------
// The Outer quantum, spelled out against the host
// ---------------------------------------------------------------------------

/// "One outer" is not a slogan: it is these seven steps, in this order, which
/// is what Driver.h's SolveLoop does per iteration (sptelem::Phase, Driver.h:85-93).
/// A device outer that reorders them is a different fixed point.
///
/// `host_anchor` is a literal that occurs in src/Driver.h; the contract test
/// mines the file for each one and asserts the LINE NUMBERS ascend in this
/// order.  Anchoring on text rather than a line number means the check survives
/// edits above it and still fails on a reordering.
struct OuterQuantumStep {
    const char* name;
    const char* host_anchor;
};

inline constexpr OuterQuantumStep kOuterQuantumSteps[] = {
    {"updpsi",     "ctx.cmfd_solver.updpsi(ctx.geometry.Phif());"},
    {"setls",      "ctx.cmfd_solver.setls(eigv);"},
    {"drive",      "ctx.cmfd_solver.drive(eigv, ctx.geometry.Phif(), residual);"},
    {"conv_check", "const bool flux_converged ="},
    {"updjnet",    "ctx.cmfd_solver.updjnet(ctx.geometry.Phif(), ctx.geometry.Jnet());"},
    {"nodal",      "ctx.nodal_solver.drive();"},
    {"cusping",    "ctx.cross_sections.ApplyRodCusping(eigv, ctx.nodal_solver.axialTransverseLeakage())"},
    {"upddhat",    "ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());"},
};
inline constexpr int kOuterQuantumStepCount =
    static_cast<int>(sizeof(kOuterQuantumSteps) / sizeof(kOuterQuantumSteps[0]));

// ---------------------------------------------------------------------------
// Sec 6.21  Function-to-queue transition map, AS DATA
// ---------------------------------------------------------------------------

/// Sec 5.4's caveat is the reason this table is checked and not just written:
/// queue execution order between cases does not change results, but the phase
/// ORDER WITHIN a case does.  Xe -> TH -> Search, predictor -> corrector, and
/// the DERIVATIVE re-converge retry are physics, not scheduling.  If the table
/// disagrees with Driver.h the answers move, and no amount of fairness tuning
/// will show it.
enum class PhaseEdgeGuard : std::uint32_t {
    Always = 0,
    FluxConverged,     ///< the outer's convergence test passed
    FluxNotConverged,  ///< requeue the same phase
    XePending,
    ThPending,
    SearchPending,
    /// Depletion has TWO edges out of NormalizeFluxSign and they are not
    /// interchangeable: Driver.h normalises after the pre-solve and then runs
    /// PredictorStep, normalises after the predictor solve and then runs
    /// CorrectorStep.  One shared "schedule is depletion" guard made which edge
    /// fires depend on evaluation order.  The distinguishing state already
    /// exists -- DeviceSlotState::solve_call_kind (Sec 3.2(B)).
    DepletionPredictorPending, ///< solve_call_kind == pre-solve
    DepletionCorrectorPending, ///< solve_call_kind == predictor
    ScheduleIsDerivative,
    ScheduleIsRod,
    /// The COMPLEMENT of the three above: no statepoint pre-work is pending, so
    /// the converged solve is the final one and PPR follows.  Spelled as its own
    /// guard rather than `Always` because an `Always` edge sitting beside four
    /// conditional siblings is not a state machine -- it is an ambiguity, and
    /// which edge fires would depend on evaluation order.
    StatepointSolveComplete,
    DerivativeCollapsed, ///< k_eff halved/doubled or non-finite: one retry only
    DerivativeOk,
    IterationRemaining,  ///< PPR corner-balance budget not exhausted
    IterationComplete,
    Fatal
};

struct PhaseEdge {
    DevicePhase    from;
    DevicePhase    to;
    PhaseEdgeGuard guard;
    const char*    host_anchor; ///< literal text in src/Driver.h, or "" when structural
};

/// The edges. Ordered so that the anchored ones ascend through Driver.h's
/// statepoint loop, which is what the contract test checks.
inline constexpr PhaseEdge kPhaseTransitions[] = {
    // --- admission -------------------------------------------------------
    {DevicePhase::Empty, DevicePhase::Import, PhaseEdgeGuard::Always, ""},
    {DevicePhase::Import, DevicePhase::Material, PhaseEdgeGuard::Always, ""},
    {DevicePhase::Material, DevicePhase::Outer, PhaseEdgeGuard::Always, ""},

    // --- the flux segment: one outer, requeued until converged ------------
    {DevicePhase::Outer, DevicePhase::Outer, PhaseEdgeGuard::FluxNotConverged,
     "ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());"},
    // Sec 5.4: the perturbation order inside a converged outer is Xe, then TH,
    // then Search.  Driver.h checks search before TH but PERTURBS TH first
    // (Driver.h "if (has_th && !th_converged)" precedes the search commit), and
    // it is the perturbation that moves the physics.
    {DevicePhase::Outer, DevicePhase::Xenon, PhaseEdgeGuard::XePending,
     "// 3. Equilibrium xenon feedback."},
    {DevicePhase::Xenon, DevicePhase::Outer, PhaseEdgeGuard::Always, ""},
    {DevicePhase::Outer, DevicePhase::ThermalHydraulics, PhaseEdgeGuard::ThPending,
     "if (has_th && !th_converged) {"},
    {DevicePhase::ThermalHydraulics, DevicePhase::Material, PhaseEdgeGuard::Always, ""},
    {DevicePhase::Outer, DevicePhase::Search, PhaseEdgeGuard::SearchPending,
     "if (has_search && !search_converged) {"},
    {DevicePhase::Search, DevicePhase::Material, PhaseEdgeGuard::Always, ""},

    // --- statepoint level (Sec 6.21 additions) ---------------------------
    {DevicePhase::Outer, DevicePhase::NormalizeFluxSign, PhaseEdgeGuard::FluxConverged,
     "cross_sections.NormalizeFluxSign();"},
    {DevicePhase::NormalizeFluxSign, DevicePhase::DepletionPredictor,
     PhaseEdgeGuard::DepletionPredictorPending,
     "cross_sections.PredictorStep(sub_dt, thermal_power,"},
    {DevicePhase::DepletionPredictor, DevicePhase::Outer, PhaseEdgeGuard::Always, ""},
    {DevicePhase::NormalizeFluxSign, DevicePhase::DepletionCorrector,
     PhaseEdgeGuard::DepletionCorrectorPending,
     "cross_sections.CorrectorStep(sub_dt, thermal_power,"},
    {DevicePhase::DepletionCorrector, DevicePhase::Outer, PhaseEdgeGuard::Always, ""},
    {DevicePhase::NormalizeFluxSign, DevicePhase::Derivative,
     PhaseEdgeGuard::ScheduleIsDerivative, "cross_sections.UpdateDerivative(schedule.delta_bppm,"},
    {DevicePhase::NormalizeFluxSign, DevicePhase::RodOp, PhaseEdgeGuard::ScheduleIsRod,
     "cross_sections.SetRod(schedule.rod_insertions);"},
    {DevicePhase::RodOp, DevicePhase::Material, PhaseEdgeGuard::Always,
     "cmfd_solver.resetDhat();"},
    // One retry only: Driver.h resets flux + d-hat and re-solves exactly once.
    {DevicePhase::Derivative, DevicePhase::RodOp, PhaseEdgeGuard::DerivativeCollapsed,
     "schedule.type == ScheduleType::DERIVATIVE &&"},
    {DevicePhase::Derivative, DevicePhase::Ppr, PhaseEdgeGuard::DerivativeOk, ""},
    {DevicePhase::NormalizeFluxSign, DevicePhase::Ppr, PhaseEdgeGuard::StatepointSolveComplete,
     "pin_power_reconstruction.reset(1.0 / eigv, geometry.Jnet(), geometry.Phif(), geometry.Phis());"},
    {DevicePhase::Ppr, DevicePhase::Ppr, PhaseEdgeGuard::IterationRemaining,
     "pin_power_reconstruction.drive(ppr_iters);"},
    {DevicePhase::Ppr, DevicePhase::ResultAggregate, PhaseEdgeGuard::IterationComplete,
     "pin_power_reconstruction.reconstructPinPower(true, schedule.print_opt.pin_flux);"},
    {DevicePhase::ResultAggregate, DevicePhase::OutputPack, PhaseEdgeGuard::Always,
     "input_output.AddResult(geometry, eigv, step_index, step_number, efpd);"},
    {DevicePhase::OutputPack, DevicePhase::Done, PhaseEdgeGuard::Always, ""},

    // --- failure ----------------------------------------------------------
    {DevicePhase::Outer, DevicePhase::Failed, PhaseEdgeGuard::Fatal, ""},
    {DevicePhase::DepletionPredictor, DevicePhase::Failed, PhaseEdgeGuard::Fatal, ""},
    {DevicePhase::DepletionCorrector, DevicePhase::Failed, PhaseEdgeGuard::Fatal, ""},

    // --- refill (Sec 8.2) -------------------------------------------------
    {DevicePhase::Done, DevicePhase::Empty, PhaseEdgeGuard::Always, ""},
    {DevicePhase::Failed, DevicePhase::Empty, PhaseEdgeGuard::Always, ""},
};
inline constexpr int kPhaseTransitionCount =
    static_cast<int>(sizeof(kPhaseTransitions) / sizeof(kPhaseTransitions[0]));

// ---------------------------------------------------------------------------
// Sec 8.2  Refill inputs
// ---------------------------------------------------------------------------

/// One pending case.  Deliberately tiny and trivially copyable: the refill
/// kernel claims one of these with an atomic and copies its ids into the reset
/// slot.  Everything else about the deck arrives later, in the Import phase.
struct DeviceInputDescriptor {
    std::uint32_t input_id;
    std::uint32_t job_id;
    std::uint32_t schedule_index;
    std::uint32_t statepoint;
};

static_assert(std::is_trivially_copyable_v<DeviceInputDescriptor>);

struct GpuRefillCounters {
    std::uint32_t refilled;             ///< slots that claimed an input this pass
    std::uint32_t exhausted;            ///< free slots with no input left to claim
    std::uint32_t stale_tenant_errors;  ///< reserved for the Task 20 audit kernel
};

// ---------------------------------------------------------------------------
// Serial reference classification
//
// This is the DEFINITION the warp-ballot kernel has to match.  Keeping it here,
// pure and CPU-runnable, is what lets test/gpu_phase_compaction.cpp check the
// ordering, the padding, the bucket choice and both fatal faults without a GPU.
// ---------------------------------------------------------------------------

/// Reset a queue set to empty.  Padding is written across the whole array so a
/// stale entry from a previous epoch can never be read as a slot id.
RASBERY_GPU_HD inline void gpuQueuesClear(DevicePhaseQueues& q) {
    for (int p = 0; p < kDevicePhaseCount; ++p) {
        for (int i = 0; i < kMaxSchedulerSlots; ++i) q.queue[p].slots[i] = kQueueEmptySlot;
        q.queue[p].count  = 0;
        q.queue[p].bucket = 0;
        q.phase_count[p]  = 0;
    }
    q.selected_phase  = -1;
    q.selected_count  = 0;
    q.selected_bucket = 0;
    q.active_count    = 0;
    q.free_count      = 0;
    q.fault_flags     = kSchedFaultNone;
    q.fault_slot      = kSchedNoFaultSlot;
}

/// The serial reference. One pass in ascending slot order, so the queues come
/// out ascending by construction.
///
/// Sec 5.2 ownership, enforced here and in the kernel identically:
///   * a slot whose captured entry is still current (slotAlreadyQueued) is
///     ALREADY in a queue -- inserting it again would let two graph bodies
///     drive one slot, so it is a fatal duplicate;
///   * a slot with in_flight set is being driven right now -- queueing it
///     anywhere is a fatal scheduler error, not a retry;
///   * inserting captures the epoch (queued_epoch = state_epoch,
///     queued_phase = phase), which is what makes the entry go stale by itself
///     at the next transition without walking any queue.
RASBERY_GPU_HD inline void gpuClassifySerial(DeviceSlotPhase* phases, int slot_count,
                                             DevicePhaseQueues& out) {
    gpuQueuesClear(out);
    if (slot_count > kMaxSchedulerSlots) {
        out.fault_flags |= kSchedFaultSlotOverflow;
        out.fault_slot = static_cast<std::uint32_t>(kMaxSchedulerSlots);
        return;
    }

    for (int s = 0; s < slot_count; ++s) {
        DeviceSlotPhase& p = phases[s];
        if (p.phase >= kDevicePhaseCount) {
            out.fault_flags |= kSchedFaultBadPhase;
            if (static_cast<std::uint32_t>(s) < out.fault_slot)
                out.fault_slot = static_cast<std::uint32_t>(s);
            continue;
        }
        const DevicePhase phase = static_cast<DevicePhase>(p.phase);

        if (!slotActive(p) || !gpuPhaseIsSchedulable(phase)) {
            if (!gpuPhaseIsSchedulable(phase)) ++out.free_count;
            continue;
        }
        ++out.active_count;

        if (slotInFlight(p)) {
            out.fault_flags |= kSchedFaultInFlightRequeue;
            if (static_cast<std::uint32_t>(s) < out.fault_slot)
                out.fault_slot = static_cast<std::uint32_t>(s);
            continue;
        }
        if (slotAlreadyQueued(p)) {
            out.fault_flags |= kSchedFaultDuplicateQueue;
            if (static_cast<std::uint32_t>(s) < out.fault_slot)
                out.fault_slot = static_cast<std::uint32_t>(s);
            continue;
        }

        const int         pi = static_cast<int>(phase);
        DevicePhaseQueue& q  = out.queue[pi];
        q.slots[q.count]     = s;
        ++q.count;
        ++out.phase_count[pi];

        // Capture the epoch: from here the entry is valid only while the slot
        // has not transitioned.
        p.queued_phase = p.phase;
        p.queued_epoch = p.state_epoch;
    }

    for (int p = 0; p < kDevicePhaseCount; ++p) out.queue[p].bucket = gpuSelectBucket(out.queue[p].count);

    out.selected_phase = gpuSelectPhase(out.phase_count);
    if (out.selected_phase >= 0) {
        out.selected_count  = out.queue[out.selected_phase].count;
        out.selected_bucket = out.queue[out.selected_phase].bucket;
    }

    if (gpuSchedulerFaultIsFatal(out.fault_flags) && out.fault_slot < static_cast<std::uint32_t>(slot_count))
        gpuMarkSlotFailed(phases[out.fault_slot], out.fault_flags);
}

// ---------------------------------------------------------------------------
// Host launchers.  Defined by the CUDA arm (GpuPhaseScheduler.cu) and by the
// no-CUDA stub, so call sites never need an #ifdef.
// ---------------------------------------------------------------------------

/// Backend-neutral stream handle; see GpuPhysicsArena.h for why void* is enough.
using GpuSchedulerStream = void*;

/// One classify/compact epoch.  Level-1: ONE CTA of 128 threads, one thread per
/// slot, reading only `phases`.
///
/// ORDERING: classify must observe the refill's writes.  Both kernels write the
/// same `phases` array -- refill stamps a recycled slot Active/Import, classify
/// queues it -- so issuing them on different streams without an event races,
/// and the symptom is a slot that sits Empty for an epoch or is queued before
/// its control structs are reset.  Use gpuLaunchRefillThenClassify below, which
/// puts both on one stream; if you must call these separately, they have to be
/// on the same stream or separated by an event.
bool gpuLaunchClassify(DeviceSlotPhase* phases, int slot_count, DevicePhaseQueues* queues,
                       GpuSchedulerStream stream);

struct GpuRefillArgs {
    DeviceSlotPhase*             phases;
    DeviceSlotState*             states;
    DeviceSearchState*           searches;
    DeviceScheduleParams*        params;
    int                          slot_count;
    const DeviceInputDescriptor* inputs;
    int                          input_count;
    int*                         next_input; ///< device atomic cursor
    GpuRefillCounters*           counters;
};

/// Sec 8.2 immediate refill: one thread per slot, free slots only.
bool gpuLaunchRefill(const GpuRefillArgs& args, GpuSchedulerStream stream);

/// One scheduler epoch: refill, then classify, ON THE SAME STREAM.  This is the
/// supported entry point precisely because the ordering is not optional -- a
/// function that takes one stream and issues both in order cannot be called
/// with them on different streams.
bool gpuLaunchRefillThenClassify(const GpuRefillArgs& args, DevicePhaseQueues* queues,
                                 GpuSchedulerStream stream);

/// Human-readable fault decode, for receipts and test failures.
inline const char* gpuSchedulerFaultName(std::uint32_t bit) {
    switch (bit) {
        case kSchedFaultDuplicateQueue:  return "duplicate_queue_insert";
        case kSchedFaultInFlightRequeue: return "in_flight_requeue";
        case kSchedFaultSlotOverflow:    return "slot_overflow";
        case kSchedFaultBadPhase:        return "bad_phase";
        default:                         break;
    }
    return "none";
}

} // namespace rasbery::gpu
