#pragma once

// Device outer control state machine -- Rev.7.1 plan Task 9 / Sec 3.1, 6.13, 6.21.
//
// ---------------------------------------------------------------------------
// THE HOLE THIS CLOSES, IN THE NUMBER THAT MEASURED IT
// ---------------------------------------------------------------------------
//
// W2 put every piece of one outer on the device -- updpsi, updjnet, upddhat,
// the convergence state machine, the nodal constants, the resident CMFD sweep
// -- and the wall did not move the way the kernel times said it should.  The
// M64 dispatch campaign found out why: the bottleneck was never the kernels, it
// was the ARRIVAL WIDTH AT THE HOST RENDEZVOUS.  Compaction of the active set
// came out exactly neutral, and widening the Anderson batch made it WORSE
// (mean_width 2.86, -8.7%), because a rendezvous cannot be wider than the
// slowest arriving slot and every outer had one.
//
// So the fix is not a wider batch and not a cheaper kernel.  It is to stop
// returning to the host between outers at all.  That is this file: the host
// launches ONE segment, the device runs up to `budget` outers inside it, and the
// host learns nothing until the segment exits.
//
// ---------------------------------------------------------------------------
// WHAT ONE SEGMENT IS
// ---------------------------------------------------------------------------
//
// kOuterSegmentPlan below, which is kOuterQuantumSteps (GpuPhaseScheduler.h)
// with an issuer named against each step.  The steps are Driver.h's, in
// Driver.h's order, and the contract test asserts the two lists agree name for
// name -- a device outer that reorders them is a different fixed point, not a
// faster one.
//
// TWO STEPS ARE HOOKS, AND THAT IS A SCOPE DECISION, NOT AN OVERSIGHT.  The
// resident CMFD sweep (CudaBatchArena::solveSweeps) and the nodal drive
// (XsReconBackend::solveNodal) are today HOST-DRIVEN calls: each one
// rendezvouses, launches, drains the stream and copies results back.  Calling
// either from inside a segment would put the per-outer round trip straight back
// in, and the segment would still be correct -- just pointless.  So they are
// declared as stream-ordered enqueue hooks with an explicit contract
// (OuterSegmentHooks), and a segment REFUSES TO RUN until both are supplied.
// Wiring them is Task 10/18's business; refusing is this file's.
//
// THE CONVERGENCE DECISION IS EVALUATED LAST, AND THAT IS EXACT.  Driver.h
// computes `flux_converged` at :1562, immediately after drive() and before
// updjnet -- but nothing between :1562 and the control block at :1601 touches
// eigv, residual or prev_inner.  So evaluating cmfdOuterConvergence after
// upddhat, on the SAME three inputs, produces the same decision from the same
// state.  What may NOT move is that prev_inner is read before it is overwritten,
// and cmfdOuterConvergence does that internally (CmfdOuterKernel.h:477-479).
// Deferring it is what lets one kernel see both the flux decision and the
// device-side signals (negative flux, Rayleigh handover) that the transition
// has to rank above it.
//
// ---------------------------------------------------------------------------
// THE HALT GATE, AND WHY IT IS NOT A CONDITIONAL GRAPH YET
// ---------------------------------------------------------------------------
//
// A host-driven loop that never observes cannot stop early: by the time the
// host could see "converged at outer 3" it has already enqueued outers 4..8.
// Running them would move the trajectory, and Sec 9.1 Class B0 on trajectory
// says the ON arm must take the SAME outers with the SAME decisions as v2.
//
// So the segment carries a per-slot halt word.  Every kernel in the body reads
// it after resolving its slot and returns if it is set; the transition kernel is
// what sets it.  This is the tree's own idiom -- CudaBICGBackend.cu's resident
// sweep gates every kernel on `sweep_halt[m]` for exactly this reason -- and it
// is deliberately the SAME SHAPE the Task 10 conditional WHILE needs: the halt
// word becomes the WHILE predicate, the enqueue sequence below becomes the
// captured body, and nothing else has to move.
//
// The overrun is real and it is counted (`halted_outer_launches`): a segment
// that exits at outer 3 of 8 still pays 5 no-op launches at c_dispatch = 0.783us
// each, i.e. ~3.9us.  That is the price of not having the conditional graph, it
// is bounded by the budget, and it is the number Task 10 has to beat.
//
// ---------------------------------------------------------------------------
// CUSPING LEAVES THE SEGMENT (the plan's Stage A)
// ---------------------------------------------------------------------------
//
// Driver.h:1579-1580 runs ApplyRodCusping between the nodal drive and upddhat,
// and a cusping that fired re-runs upddtil -- so it changes the operator the
// NEXT step reads, inside the outer.  Porting that is Task 11.  Until then a
// slot whose deck has fractional rods is simply not eligible: it would have to
// escape with MaterialChanged on the very first outer, which is a segment of
// length one and buys nothing.
//
// The eligibility predicate is mined from XSSet::ApplyRodCusping (XSSet.cpp:
// 3242-3282) rather than guessed: that function returns false immediately when
// `_axial_rod_division <= 0`, and otherwise only does work for nodes with
// `EPS < rod_fraction(l) < 1 - EPS`.  A deck with neither can never cusp, so the
// step the device skips is a step the host would also have skipped.
//
// ONE ASYMMETRY IS DELIBERATE AND HARMLESS.  ApplyRodCusping bumps
// `_hoststate_generation` unconditionally (XSSet.cpp:3280), even on the
// early-out path where it changed nothing.  A segment does not call it, so that
// counter does not advance -- but it gates UPLOADS of data that provably did not
// change, so suppressing the bump suppresses a redundant copy and no more.
//
// ---------------------------------------------------------------------------
// LAYERING
// ---------------------------------------------------------------------------
//
// Everything above the `#if defined(__CUDACC__)` line is CUDA-free and pure, so
// test/outer_state_replay.cpp drives the SAME state machine the kernel drives,
// on the host, with no device.  The host-facing runner is declared here and
// defined in CudaOuterGraph.cu (CUDA arm) and CudaOuterGraphStub.cpp (no-CUDA
// arm), the same split GpuPhaseScheduler.h/.cu uses.
//
// ---------------------------------------------------------------------------
// READING THE `Driver.h:NNNN` CITATIONS
// ---------------------------------------------------------------------------
//
// Every one of them names a line in the Driver.h this task was written against
// (8be6bee), which is the numbering EVERY other Driver.h citation in this tree
// uses -- CmfdOuterKernel.h, GpuPhaseScheduler.h, GpuPhysicsTypes.h,
// CudaBICGBackend.cu, cmfd_outer_replay.cpp and the two scheduler contract
// tests among them.  Task 9's own delegation block adds 131 lines to
// ReconvergeFlux, which sits ABOVE SolveLoop, so a citation into SolveLoop
// resolves at NNNN + 131 in the working tree.  Rebasing the citations here and
// nowhere else would leave two numberings in one tree, which is worse than one
// stale one; the offset is stated once, here, instead.

#include "CudaCmfdOuterKernels.h"
#include "CudaNodalConstantKernel.h"
#include "GpuCanonicalState.h"
#include "GpuPhaseScheduler.h"
#include "GpuPhysicsTypes.h"
#include "HostOuterBodyCounters.h"

#include <cstdint>
#include <iosfwd>
#include <string>

namespace rasbery::gpu {

// ---------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------

/// Sec 10: RASBERY_GPU_OUTER, default OFF.  Read once per process.
///
/// DEFAULT OFF IS THE WHOLE SAFETY ARGUMENT.  With it unset the delegation
/// branch in Driver.h::SolveLoop is a `const bool` that is false, the host loop
/// runs exactly the code it ran before, and the ON/OFF comparison is
/// byte-identical by construction rather than by measurement.
[[nodiscard]] bool outerGpuEnabled();

/// The W0-informed default segment length.
///
/// EIGHT, FOR TWO REASONS THAT AGREE.  (1) kOuterQuantumSteps has eight steps,
/// so a budget of 8 makes the segment one full "outer quantum square" and keeps
/// the overrun cost (budget - exit_index launches at c_dispatch = 0.783us)
/// under 6us, which is inside the constraint-32 3% control ceiling at any
/// realistic outer cost.  (2) W0's replay put the useful uninterrupted run
/// length between 4 and 16 outers -- below 4 the launch amortisation does not
/// pay, above 16 the overrun on an early convergence starts to dominate.
inline constexpr unsigned int kOuterSegmentBudgetDefault = 8u;

/// RASBERY_GPU_OUTER_SEGMENT_MAX, default kOuterSegmentBudgetDefault.  Clamped
/// to [1, kOuterSegmentBudgetMax]: a budget of 0 would be a segment that runs no
/// outers and returns true, which is an infinite loop at the call site.
inline constexpr unsigned int kOuterSegmentBudgetMax = 64u;
[[nodiscard]] unsigned int outerSegmentBudget();

// ---------------------------------------------------------------------------
// The segment plan, AS DATA
// ---------------------------------------------------------------------------

/// One step of the segment body.
///
/// `quantum_name` must equal kOuterQuantumSteps[i].name for the same i -- the
/// contract test compares the two lists element by element, so a step that is
/// added here without a host anchor, or reordered against Driver.h, fails at
/// source level rather than in a 500-statepoint A/B.
///
/// `issuer` is the function this step enqueues.  A step whose issuer is a HOOK
/// name is one the segment cannot issue itself yet; `hook` marks those so the
/// test can check that every hook is refused when absent instead of silently
/// skipped.
struct OuterSegmentStep {
    const char* quantum_name; ///< kOuterQuantumSteps[i].name
    const char* issuer;       ///< enqueue function, or the hook that owns it
    int         hook;         ///< 1 = supplied by the caller, not by this file
    int         prologue;     ///< 1 = issued once per segment, not once per outer
};

/// The body, in Driver.h's order.  Read it against Driver.h:1544-1585.
inline constexpr OuterSegmentStep kOuterSegmentPlan[] = {
    // Driver.h:1547.
    {"updpsi", "enqueueUpdPsi", 0, 0},
    // Driver.h:1551-1555.  setls and drive are ONE hook because the resident
    // sweep graph does both plus the Wielandt update and the negative-flux
    // check (CudaBICGBackend.cu, cmfd_sweep_* ); splitting them here would
    // describe a decomposition that does not exist on the device.
    {"setls", "OuterSegmentHooks::enqueue_cmfd_sweep", 1, 0},
    {"drive", "OuterSegmentHooks::enqueue_cmfd_sweep", 1, 0},
    // Driver.h:1562.  The INPUTS are formed here; the decision is published at
    // the end of the body (see the header note).  This step refreshes
    // CmfdOuterInputs::eigv/residual from the sweep's own device scalars, which
    // is what makes the decision kernel runnable without a host round trip.
    {"conv_check", "enqueueOuterRefreshInputs", 0, 0},
    // Driver.h:1569.
    {"updjnet", "enqueueUpdJnet", 0, 0},
    // Driver.h:1573-1575.
    {"nodal", "OuterSegmentHooks::enqueue_nodal_drive", 1, 0},
    // Driver.h:1579-1580.  Stage A: eligibility guarantees this is a no-op, and
    // a slot for which it would NOT be a no-op escapes with MaterialChanged.
    {"cusping", "OuterSegmentEligibility::fractional_rods", 0, 0},
    // Driver.h:1584.
    {"upddhat", "enqueueUpdDhat", 0, 0},
};
inline constexpr int kOuterSegmentPlanCount =
    static_cast<int>(sizeof(kOuterSegmentPlan) / sizeof(kOuterSegmentPlan[0]));

static_assert(kOuterSegmentPlanCount == kOuterQuantumStepCount,
              "the segment body must have exactly the steps of one outer quantum");

/// updateConstant is a PROLOGUE, not a per-outer step, and that is Driver.h's
/// shape too: Nodal::updateConstant re-runs when the macro-XS move, and inside a
/// flux-reconvergence segment they do not (the only in-outer XS write is rod
/// cusping, which an eligible slot cannot do).  Issuing it once per outer would
/// be eight times the work for a bit-identical result.
inline constexpr OuterSegmentStep kOuterSegmentPrologue[] = {
    {"nodal_constants", "enqueueNodalUpdateConstant", 0, 1},
};
inline constexpr int kOuterSegmentPrologueCount =
    static_cast<int>(sizeof(kOuterSegmentPrologue) / sizeof(kOuterSegmentPrologue[0]));

// ---------------------------------------------------------------------------
// Sec 3.1  The device-side signals the transition ranks
// ---------------------------------------------------------------------------

/// What the hooks publish about ONE outer, per slot, in device memory.
///
/// WHY THE PROBE EXISTS AT ALL.  k_cmfd_outer_convergence reads eigv and
/// residual out of a CmfdOuterInputs array the HOST fills.  Inside a segment the
/// host cannot refill it -- that is the round trip being removed -- and eigv and
/// residual are the two fields that change every outer.  Everything else in
/// CmfdOuterInputs is constant across a flux-reconvergence segment: the
/// tolerances are the schedule's, and xe_pending / th_pending / search_pending
/// only move when one of those steps FIRES, which ends the segment.  So the
/// probe carries exactly the two that move, plus the four signals that only the
/// device can see.
///
/// `rayleigh` is mined from the existing latch, not invented: cmfd_wiel_finalize
/// (CudaBICGBackend.cu:1725) writes `sm[kSweepState] = 2.0` and halts the slot
/// when the Wielandt gamma degenerates, and CmfdSweepIO documents state 2 as
/// "the host must finish the current sweep with the Rayleigh branch".  That is a
/// HANDOVER, not a failure: the segment exits, the host finishes the sweep, and
/// the slot resumes in Outer.
struct DeviceOuterProbe {
    double eigv;     ///< the sweep's eigenvalue for this outer
    double residual; ///< the sweep's L2 flux residual for this outer

    std::uint32_t negative_flux;    ///< CmfdSweepIO::negative_last
    std::uint32_t rayleigh;         ///< CmfdSweepIO::state == 2
    std::uint32_t nonfinite;        ///< eigv or residual is not finite
    std::uint32_t material_changed; ///< cusping would have fired (Stage A escape)
};

static_assert(std::is_trivially_copyable_v<DeviceOuterProbe>);

/// One slot's segment bookkeeping, in device memory.
///
/// THE HALT WORD IS NOT IN HERE, and that is a layering decision rather than an
/// omission.  Every kernel of the segment body has to read it, including the
/// five in CudaCmfdOuterKernels.h -- and that header must not include this one
/// (this one already includes it).  So the halt lives in a DENSE
/// `std::uint32_t[slot_count]` array the runner owns, which those kernels take
/// as a plain pointer with no new type dependency, and there is exactly one
/// source of truth for it.  `exit` here is the answer the HOST reads back once.
struct DeviceOuterSegmentState {
    std::uint32_t exit;             ///< 1 = the segment has decided; read once
    std::uint32_t outer_in_segment; ///< outers COMMITTED in this segment
    std::uint32_t budget;           ///< RASBERY_GPU_OUTER_SEGMENT_MAX for this segment
    std::uint32_t next_phase;       ///< DevicePhase written at exit

    std::uint32_t escape; ///< DeviceEscape written at exit
    std::uint32_t reserved0;
    std::uint32_t reserved1;
    std::uint32_t reserved2;
};

static_assert(std::is_trivially_copyable_v<DeviceOuterSegmentState>);
static_assert(sizeof(DeviceOuterSegmentState) == 32);

RASBERY_GPU_HD inline void deviceOuterSegmentReset(DeviceOuterSegmentState& s,
                                                   unsigned int budget) {
    s.exit             = 0u;
    s.outer_in_segment = 0u;
    s.budget           = budget;
    s.next_phase       = static_cast<std::uint32_t>(DevicePhase::Outer);
    s.escape           = static_cast<std::uint32_t>(DeviceEscape::None);
    s.reserved0        = 0u;
    s.reserved1        = 0u;
    s.reserved2        = 0u;
}

// ---------------------------------------------------------------------------
// THE STATE MACHINE
// ---------------------------------------------------------------------------

/// What one outer's transition decided.
struct OuterTransition {
    unsigned int next_phase;   ///< DevicePhase
    unsigned int escape;       ///< DeviceEscape
    int          exit_segment; ///< 1 = the host must observe before the next outer
};

/// Rank the signals and the CMFD decision into ONE transition.
///
/// THE ORDER IS THE CONTRACT.  Six rules, and every one of them can be got
/// backwards in a way no converged answer would reveal:
///
///  (1) NON-FINITE FIRST.  A NaN eigenvalue makes `|prev_inner - eigv| < tol`
///      false and `residual < tol` false, so the CMFD body would report a
///      perfectly ordinary "not converged, requeue" and the segment would spin
///      its whole budget on garbage.  Sec 9.2's fail-closed rule is that a
///      non-finite is a FAILED SLOT, so it outranks everything.
///
///  (2) NEGATIVE FLUX SECOND.  Same class -- the iterate is not a flux, so no
///      test computed from it means anything -- but ranked below non-finite so a
///      run that produced both reports the more fundamental one.
///
///  (3) RAYLEIGH HANDOVER THIRD, AND IT IS NOT A FAILURE.  The sweep halted
///      itself with the sums exported and asked the host to finish this sweep on
///      the Rayleigh branch.  The slot stays in Outer and the host re-enters, so
///      this is an EXIT WITHOUT A PHASE CHANGE.  Ranking it below the two fatal
///      signals matters: a degenerate gamma that also produced a NaN is a failure
///      first and a handover second.
///
///  (4) MATERIAL CHANGED FOURTH.  Cusping fired (or would have).  Also an exit
///      without a phase change: the host runs ApplyRodCusping and re-enters.
///      Below Rayleigh because a sweep that already handed itself back has not
///      produced the flux the cusping blend would read.
///
///  (5) THE CMFD DECISION FIFTH.  Anything that is not Outer -> Outer ends the
///      segment, because the host owns Xe, T/H, Search and the statepoint tail.
///      Its escape is carried through verbatim -- FluxConverged,
///      FluxLimitCycleSample, FluxStallFatal -- because those are the CPU's own
///      words for what happened and the search consumes them.
///
///  (6) THE BUDGET LAST, AND THAT IS THE ONE THAT LOOKS WRONG.  An outer that
///      CONVERGED on the budget-th iteration must publish FluxConverged, not
///      SegmentBudget: the budget is a scheduling artefact and the convergence is
///      physics.  Checking the budget first would relabel a converged solve as a
///      truncated one, the host would re-enter, run one more outer, and converge
///      again -- a different outer count for the same answer, which is exactly
///      the Class-B0-on-trajectory failure this task is gated on.
///
/// `outer_in_segment` is the count BEFORE this outer is committed, so the budget
/// test is on `outer_in_segment + 1`.
RASBERY_GPU_HD inline OuterTransition deviceOuterTransition(const CmfdOuterDecision& d,
                                                            const DeviceOuterProbe& probe,
                                                            unsigned int outer_in_segment,
                                                            unsigned int budget) {
    OuterTransition t{};

    if (probe.nonfinite != 0u) {
        t.next_phase   = static_cast<unsigned int>(DevicePhase::Failed);
        t.escape       = static_cast<unsigned int>(DeviceEscape::NonFinite);
        t.exit_segment = 1;
        return t;
    }
    if (probe.negative_flux != 0u) {
        t.next_phase   = static_cast<unsigned int>(DevicePhase::Failed);
        t.escape       = static_cast<unsigned int>(DeviceEscape::NegativeFlux);
        t.exit_segment = 1;
        return t;
    }
    if (probe.rayleigh != 0u) {
        t.next_phase   = static_cast<unsigned int>(DevicePhase::Outer);
        t.escape       = static_cast<unsigned int>(DeviceEscape::RayleighFallback);
        t.exit_segment = 1;
        return t;
    }
    if (probe.material_changed != 0u) {
        t.next_phase   = static_cast<unsigned int>(DevicePhase::Outer);
        t.escape       = static_cast<unsigned int>(DeviceEscape::MaterialChanged);
        t.exit_segment = 1;
        return t;
    }

    if (d.next_phase != static_cast<unsigned int>(DevicePhase::Outer)) {
        t.next_phase   = d.next_phase;
        t.escape       = d.escape;
        t.exit_segment = 1;
        return t;
    }

    if (outer_in_segment + 1u >= budget) {
        t.next_phase   = static_cast<unsigned int>(DevicePhase::Outer);
        t.escape       = static_cast<unsigned int>(DeviceEscape::SegmentBudget);
        t.exit_segment = 1;
        return t;
    }

    t.next_phase   = static_cast<unsigned int>(DevicePhase::Outer);
    t.escape       = static_cast<unsigned int>(DeviceEscape::None);
    t.exit_segment = 0;
    return t;
}

/// Publish one transition into the hot 32-byte control packet.
///
/// WHAT IT WRITES: phase, escape, state_epoch, and the fatal flag when the phase
/// is Failed.  WHAT IT MUST NEVER WRITE: queued_phase and queued_epoch.  Those
/// are captured by CLASSIFY at queue insertion (Sec 5.2), and a phase kernel that
/// stamps them re-validates the very queue entry it is being run from -- the slot
/// then looks "already queued" forever and classify raises a duplicate-insert
/// fault against it.  The contract test greps for exactly that.
///
/// state_epoch IS BUMPED EVEN ON Outer -> Outer.  A requeue is a transition:
/// without the bump `slotAlreadyQueued` (queued_epoch == state_epoch &&
/// queued_phase == phase) stays true and classify refuses to re-queue the slot.
/// The `{Outer, Outer, FluxNotConverged}` edge in kPhaseTransitions is precisely
/// this case, so it is not an optimisation opportunity.
///
/// error_code IS NOT WRITTEN, deliberately.  It carries the SCHEDULER fault bits
/// (kSchedFault*, 0x1..0x8), and a DeviceEscape is a small ordinal from the same
/// numeric range -- writing one into the other would make gpuSchedulerFaultName
/// decode a physics escape as a queue fault.  The escape word already says why.
RASBERY_GPU_HD inline void outerApplyTransition(DeviceSlotPhase& p,
                                                const OuterTransition& t) {
    p.phase  = static_cast<std::uint8_t>(t.next_phase);
    p.escape = static_cast<std::uint8_t>(t.escape);
    if (t.next_phase == static_cast<unsigned int>(DevicePhase::Failed))
        p.flags = static_cast<std::uint8_t>((p.flags | kSlotFlagFatal) & ~kSlotFlagInFlight);
    ++p.state_epoch;
}

/// Every (from, to) pair this machine can emit, as data, so the replay and the
/// contract test score the SAME set against kPhaseTransitions rather than two
/// hand-maintained copies of it.
///
/// All six exist as edges out of Outer in kPhaseTransitions: Outer (requeue),
/// Xenon, ThermalHydraulics, Search, NormalizeFluxSign (converged), Failed.
RASBERY_GPU_HD constexpr DevicePhase outerEmittedPhaseAt(int i) {
    switch (i) {
        case 0: return DevicePhase::Outer;
        case 1: return DevicePhase::Xenon;
        case 2: return DevicePhase::ThermalHydraulics;
        case 3: return DevicePhase::Search;
        case 4: return DevicePhase::NormalizeFluxSign;
        case 5: return DevicePhase::Failed;
        default: break;
    }
    return DevicePhase::Empty;
}
inline constexpr int kOuterEmittedPhaseCount = 6;

/// Human-readable escape decode for the receipt and for test failures.
inline const char* outerEscapeName(DeviceEscape e) {
    switch (e) {
        case DeviceEscape::None:                 return "none";
        case DeviceEscape::FluxConverged:        return "flux_converged";
        case DeviceEscape::FluxLimitCycleSample: return "flux_limit_cycle_sample";
        case DeviceEscape::FluxStallFatal:       return "flux_stall_fatal";
        case DeviceEscape::NegativeFlux:         return "negative_flux";
        case DeviceEscape::RayleighFallback:     return "rayleigh_fallback";
        case DeviceEscape::SegmentBudget:        return "segment_budget";
        case DeviceEscape::MaterialChanged:      return "material_changed";
        case DeviceEscape::NonFinite:            return "nonfinite";
        case DeviceEscape::MaxIteration:         return "max_iteration";
        case DeviceEscape::CramZeroDiagonal:     return "cram_zero_diagonal";
        case DeviceEscape::CramNotConverged:     return "cram_not_converged";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Eligibility
// ---------------------------------------------------------------------------

/// Why a segment did not run.  Every refusal is COUNTED and printed, because the
/// failure mode of an opt-in fast path is that it silently never engages and the
/// A/B measures the slow path twice (the G0 rule the XSRECON receipt exists for).
enum class OuterSegmentRefusal : int {
    None = 0,
    FeatureOff,      ///< RASBERY_GPU_OUTER unset
    NoRunner,        ///< no CUDA arm, or the runner failed to initialise
    NoArena,         ///< GpuPhysicsArena is not reserved, so no fixed addresses
    Unbound,         ///< nobody handed the runner the arena-derived views
    BatchMode,       ///< the run's batch is wider than the physics arena stood up
    GeometryMismatch,///< this Driver's deck is not the deck the arena was stood on
    FractionalRods,  ///< Stage A: cusping would have to escape on outer 1
    CriticalSearch,  ///< see the note on OuterSegmentEligibility::critical_search
    NoSweepHook,     ///< the resident CMFD sweep enqueue was not supplied
    NoNodalHook,     ///< the nodal drive enqueue was not supplied
    SweepNotResident,///< RASBERY_GPU_CMFD_SWEEP/ASSEMBLY off: no device flux to read
    NoResidency,     ///< the sweep arena's buffers were never handed over
    LaunchFailed,    ///< a CUDA call in the segment failed; the host path takes over
    Count
};

inline const char* outerRefusalName(OuterSegmentRefusal r) {
    switch (r) {
        case OuterSegmentRefusal::None:            return "none";
        case OuterSegmentRefusal::FeatureOff:      return "feature_off";
        case OuterSegmentRefusal::NoRunner:        return "no_runner";
        case OuterSegmentRefusal::NoArena:         return "no_arena";
        case OuterSegmentRefusal::Unbound:         return "unbound";
        case OuterSegmentRefusal::BatchMode:       return "batch_mode";
        case OuterSegmentRefusal::GeometryMismatch: return "geometry_mismatch";
        case OuterSegmentRefusal::FractionalRods:  return "fractional_rods";
        case OuterSegmentRefusal::CriticalSearch:  return "critical_search";
        case OuterSegmentRefusal::NoSweepHook:     return "no_sweep_hook";
        case OuterSegmentRefusal::NoNodalHook:     return "no_nodal_hook";
        case OuterSegmentRefusal::SweepNotResident: return "sweep_not_resident";
        case OuterSegmentRefusal::NoResidency:     return "no_residency";
        case OuterSegmentRefusal::LaunchFailed:    return "launch_failed";
        case OuterSegmentRefusal::Count:           break;
    }
    return "?";
}

/// What the caller knows about the deck, reduced to the bits eligibility turns
/// on.  Pure, so the predicate is testable with no device.
struct OuterSegmentEligibility {
    int runner_available; ///< a CUDA arm exists and initialised
    int arena_reserved;   ///< GpuPhysicsArena::available()
    int bound;            ///< bind() was called with the arena-derived views

    /// Rev.7.1 Task 18-lite: the two halves of "is there a seat for this deck".
    ///
    /// WHAT REPLACED `batch_width > 1`.  Task 9 refused every batch outright
    /// because the physics arena was stood up at width ONE and there was one
    /// process-wide runner, so an M-wide batch had M Drivers reaching for one
    /// slot's jnet/phis/flux.  The arena is now stood up at the RUN's width and
    /// there is one runner per slot, so the honest question is no longer "is
    /// this a batch" but "is this batch inside the arena that was stood up":
    ///
    ///   batch_width > arena_slots -- the arena is narrower than the run.  This
    ///       is what a VRAM admission that shrank the arena, or a width past
    ///       kMaxDeviceSlots, looks like from here.  Still `batch_mode`, because
    ///       that is still the true reason and the receipt vocabulary does not
    ///       need a second word for it.
    ///
    ///   slot_admitted -- this Driver's slot is inside the arena AND the arena
    ///       holds THIS deck's shape.  The arena is ONE allocation with one
    ///       layout for every slot (arenaComputeLayout replicates a single slot
    ///       block), so two decks with different nxyz/nsurf/nxy/n_fuel/ng cannot
    ///       share it; the first Driver to arrive stands it up and a later one
    ///       with a different shape is refused by name rather than served a slot
    ///       whose strides are somebody else's.  See rasberyOuterSlotAdmitted.
    int batch_width;      ///< rasberyBatchWidth()
    int arena_slots;      ///< the width the physics arena was actually stood at
    int slot_admitted;    ///< rasberyOuterSlotAdmitted(slot, this deck's shape)

    int fractional_rods;  ///< outerDeckHasFractionalRods()

    /// A CRITICAL SEARCH IS REFUSED ONLY WHERE THE DEVICE DECISION IS
    /// AUTHORITATIVE -- which, since Task 10, is ReconvergeFlux and not
    /// SolveLoop.
    ///
    /// Both reasons below are about the DECISION, and SolveLoop stopped
    /// consuming it: its delegation takes flux_converged and the three carried
    /// scalars, and flux_converged is computed at CmfdOuterKernel.h:476-479
    /// from prev_inner, eigv, residual and the two tolerances -- before any
    /// search term is read.  The host ladder still evaluates
    /// schedule.searchResidual(eigv) from the eigenvalue the segment returned,
    /// still owns clean_iters and the settle gate, and still decides
    /// FluxLimitCycleSample; the device copies of those drift and nothing reads
    /// them.  So on that path neither reason can reach the answer.
    ///
    /// What the search DOES do to the segment is move the macro-XS when it
    /// commits a boron trial, and that is handled by the per-outer syncs rather
    /// than by a refusal: xsnf and dtil are uploaded at the top of every outer,
    /// and CMFD::resetDhat is called only from Drive() (Driver.h:2718, 2735),
    /// outside SolveLoop, so the arm-time dhat seed is never stale within one.
    ///
    /// THE TWO REASONS, KEPT because ReconvergeFlux still consumes the whole
    /// decision and Tasks 13/14/17 will bring SolveLoop back to consuming it:
    ///
    /// (1) Driver.h:1864-1871 evaluates `schedule.searchResidual(eigv)` per outer
    ///     from the eigenvalue THAT outer produced, and there is no device Search
    ///     phase to do it with (Sec 6.17 / Task 17).  A segment would have to
    ///     carry a `search_pending` computed from the PREVIOUS outer's eigv, and
    ///     that is a different trajectory, not a stale field.
    ///
    /// (2) CmfdOuterInputs::search_pending is used for two different host terms.
    ///     At CmfdOuterKernel.h:505 and :540 it stands in for Driver.h's
    ///     `has_search` (:1652 `|| !has_search`, :1852 `if (has_search && ...)`),
    ///     which is a SCHEDULE property; at :555 it stands in for
    ///     `has_search && !search_converged` (:1901).  Those two are not the same
    ///     predicate, and on a solve where the search has converged but the flux
    ///     limit-cycles they disagree: the host takes the noisy sample and breaks
    ///     CONVERGED, the body takes the Fatal branch.  That is a W2 modelling
    ///     gap this task inherits rather than creates -- refusing search decks is
    ///     how Task 9 avoids depending on its resolution.
    int critical_search;

    /// Link 2: has anyone handed this runner the sweep arena's buffers?
    ///
    /// ONE QUESTION, NOT TWO.  It is tempting to ask `is the run configured for
    /// the resident sweep` separately, but the receipt is printed from a
    /// process-wide singleton that has no BICGCMFD to ask -- and a second
    /// spelling of the gate would be free to disagree with the one that
    /// actually decided.  So the runner reports what it can see: whether the
    /// handover happened.  WHY it did not is recorded by the arming site
    /// itself, which does have the solver, and shows up in `refusals{}`.
    int residency_bound;
    int have_sweep_hook;
    int have_nodal_hook;
    /// Can this runner honour step 7 at all?
    ///
    /// WITHOUT IT, A FRACTIONAL ROD IS STILL A REFUSAL, and that pairing is the
    /// point: `the deck cusps` is only a problem while the segment cannot cusp.
    /// Stage A refused the deck because the capability was missing; now that
    /// outerCuspingHook supplies it the same deck is eligible, and the refusal
    /// survives exactly where it is still true -- a runner with no cusping hook.
    int have_cusping_hook;
};

/// The predicate, in one place, ranked so the FIRST reason a reader would ask
/// about is the one reported.
[[nodiscard]] RASBERY_GPU_HD inline OuterSegmentRefusal
outerSegmentRefusal(const OuterSegmentEligibility& e) {
    if (!e.runner_available) return OuterSegmentRefusal::NoRunner;
    if (!e.arena_reserved) return OuterSegmentRefusal::NoArena;
    if (!e.bound) return OuterSegmentRefusal::Unbound;
    if (e.batch_width > e.arena_slots) return OuterSegmentRefusal::BatchMode;
    if (!e.slot_admitted) return OuterSegmentRefusal::GeometryMismatch;
    if (e.fractional_rods && !e.have_cusping_hook)
        return OuterSegmentRefusal::FractionalRods;
    if (e.critical_search) return OuterSegmentRefusal::CriticalSearch;
    if (!e.residency_bound) return OuterSegmentRefusal::NoResidency;
    if (!e.have_sweep_hook) return OuterSegmentRefusal::NoSweepHook;
    if (!e.have_nodal_hook) return OuterSegmentRefusal::NoNodalHook;
    return OuterSegmentRefusal::None;
}

/// The half of the ladder a caller can decide BEFORE it arms.
///
/// WHY THIS EXISTS -- THE BATCH PERTURBATION.  Arming is not a query.  It binds
/// residency (which patches the shared slot table, synchronises the device and
/// seeds dhat/psi H2D over live arena memory) and it adopts the segment's
/// jnet/phis/flux as the nodal backend's CANONICAL set.  Those buffers come from
/// ONE process-wide arena slot, so in `--batch-mode` every concurrent Driver
/// adopted the SAME three device pointers and the batched nodal drive stopped
/// having a per-deck jnet, phis and flux.  The segment itself refused every time
/// -- `refusals{"batch_mode":N}`, `segment_launches:0` -- and the answers moved
/// anyway: ~622 of 708 datasets differed, ppm by ~1.3 of 1285, k_eff by 8e-6.
///
/// So the refusal has to be asked BEFORE the arm, not after it, and this is the
/// part of it that can be: every reason ranked above `residency_bound` is a
/// property of the RUN (is there a runner, is it bound, is this a batch, does
/// the deck cusp, is there a search) rather than of the arming.  The reasons
/// below it are exactly the ones arming exists to satisfy, so they are answered
/// `yes` here and left to the post-arm `refusal()` -- which still runs, still
/// reports, and still ranks `batch_mode` above them, so the receipt is
/// unchanged.
///
/// The ladder is not restated: this builds an eligibility whose post-arm fields
/// are all satisfied and asks the SAME function, so the two cannot disagree.
///
/// Rev.7.1 Task 18-lite: `arena_slots` and `slot_admitted` join it, because the
/// batch reason stopped being "this is a batch" and became "this batch does not
/// fit the arena that was stood up".  Both are properties of the RUN, both are
/// knowable before the arm, and both are asked through the same ladder.
[[nodiscard]] inline OuterSegmentRefusal
outerSegmentPreArmRefusal(int batch_width, bool fractional_rods, bool critical_search,
                          int arena_slots, bool slot_admitted) {
    if (!outerGpuEnabled()) return OuterSegmentRefusal::FeatureOff;
    OuterSegmentEligibility e{};
    e.runner_available  = 1;
    e.arena_reserved    = 1;
    e.bound             = 1;
    e.batch_width       = batch_width;
    e.arena_slots       = arena_slots;
    e.slot_admitted     = slot_admitted ? 1 : 0;
    e.fractional_rods   = fractional_rods ? 1 : 0;
    e.critical_search   = critical_search ? 1 : 0;
    e.residency_bound   = 1;
    e.have_sweep_hook   = 1;
    e.have_nodal_hook   = 1;
    e.have_cusping_hook = 1;
    return outerSegmentRefusal(e);
}

/// The Stage-A cusping predicate, mined from XSSet::ApplyRodCusping
/// (XSSet.cpp:3242-3268).  `division` is XSSet::axial_rod_division(),
/// `fraction` is Geometry::rod_fraction over the node list, `eps` is pch.h's EPS.
///
/// Kept as a free function over raw arrays rather than a method on XSSet so the
/// replay can drive it without dragging Geometry, std::map and std::string into
/// the test -- the same reason DeviceSearchExit restates SearchExit.
[[nodiscard]] inline bool outerDeckHasFractionalRods(int division, const double* fraction,
                                                     int nxyz, double eps) {
    if (division <= 0) return false;
    if (fraction == nullptr) return false;
    for (int l = 0; l < nxyz; ++l)
        if (fraction[l] > eps && fraction[l] < 1.0 - eps) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Sec 9.3  Receipt
// ---------------------------------------------------------------------------

/// Counters only -- Task 9's telemetry budget is one increment per event, no
/// timers and no per-outer records.  A segment is at most 8 outers, so a
/// microsecond of instrumentation per outer would be a measurable tax on the
/// thing being measured.
struct OuterSegmentCounters {
    std::uint64_t segment_launches         = 0; ///< segments that actually ran
    std::uint64_t device_outers            = 0; ///< outers executed inside segments
    std::uint64_t host_outer_observations  = 0; ///< segment exits the host had to read
    std::uint64_t budget_exits             = 0; ///< exits whose escape was SegmentBudget
    std::uint64_t halted_outer_launches    = 0; ///< no-op launches after the halt latched
    /// Bytes moved bridging jnet around the HOST nodal drive, both ways.
    ///
    /// THE HONEST REMAINING COST OF LINK 2, and the number that says when it is
    /// paid off.  The dhat H2D it removed is nsurf*ng doubles per outer; this
    /// bridge is twice that, because the nodal drive is still a host call that
    /// reads and writes Geometry::Jnet.  Link 2 is therefore a wash on bytes and
    /// a win on host arithmetic; the bytes come back when the nodal drive becomes
    /// a stream-ordered enqueue and the bridge disappears with it.
    ///
    /// Rev.7.1 Task 18-lite MADE THAT SENTENCE COME TRUE WITHOUT THE ENQUEUE.
    /// The bridge exists because the nodal drive reads a HOST array; when the
    /// drive is the DEVICE nodal and it has adopted the segment's own jnet as
    /// its canonical buffer, it reads the buffer updjnet just wrote and writes
    /// the buffer upddhat is about to read -- so there is nothing to bridge and
    /// this counter is zero.  It stays non-zero exactly where the binding cannot
    /// be taken: a host nodal drive, or a deck Nodal::TryDriveGpu refuses
    /// (i-SMR CY02's fractional rods).  Zero here and non-zero
    /// canonical_nodal_outers is the receipt that the binding is live.
    std::uint64_t jnet_bridge_bytes        = 0;
    /// Outers whose updjnet had to be RE-ISSUED after the host finished the
    /// sweep.
    ///
    /// WHAT IT COUNTS, AND WHY IT IS NOT ZERO.  cmfd_sweep_verdict raises the
    /// segment's halt when the drive did not finish on the device (sweep state
    /// 0, the launch's slot budget ran out; state 2, the Wielandt gamma
    /// degenerated), so that the rest of the body does not read a half sweep --
    /// and updjnet is already enqueued BEHIND that verdict on the same stream,
    /// so it becomes a no-op.  The host then finishes the drive at the
    /// observation and republishAfterHostSweep takes the halt off, but nothing
    /// re-ran the step the halt had swallowed: upddhat corrected the current
    /// against the PREVIOUS outer's jnet, and the segment's trajectory left the
    /// host's.  Measured on kngr_238: three outers out of 11,993 -- the first at
    /// statepoint 23 -- and those three were the whole remaining ON-vs-OFF
    /// divergence.
    ///
    /// A NON-ZERO VALUE IS HEALTHY, not a warning: it is the count of outers the
    /// repair covered.  A run with sweep escapes and a ZERO here is the shape to
    /// distrust.
    std::uint64_t updjnet_reissued         = 0;
    /// Bytes uploaded to guarantee the device flux matches Geometry::Phif.
    std::uint64_t flux_sync_bytes          = 0;
    /// Bytes returned to the HOST dhat/psi so the host drive path stays correct.
    ///
    /// This is what replaced the 416 KiB/outer dhat H2D rather than removing it
    /// outright: the copy changed direction and left the PCIe budget roughly
    /// where it was, while the arithmetic moved to the device.  It disappears
    /// when every drive is device-resident and nothing on the host reads _dhat
    /// or _psi any more -- which is a separate audit, not an assumption.
    std::uint64_t host_mirror_bytes        = 0;
    /// Outers on which ApplyRodCusping fired inside a segment.
    ///
    /// The number that says whether a deck cusps at all, and the one that makes
    /// the i-SMR CY02 failure visible in a receipt rather than only in a k_eff:
    /// before this the segment SKIPPED step 7, so a cusping deck ran a different
    /// outer from the host's and said nothing about it.
    std::uint64_t cusping_fired            = 0;
    /// d-tilde bytes re-uploaded because cusping rebuilt it mid-outer.
    std::uint64_t cusping_dtil_bytes       = 0;
    /// Outers whose drive left the device flux authoritative.
    ///
    /// The ceiling on what the flux elision can save: an outer whose drive fell
    /// back to the host loop -- the Wielandt warm-up, a declined enqueue -- has
    /// to be re-uploaded whatever the generations say.  Reported so the answer
    /// to `how much of this deck is warm-up` is a number and not an estimate.
    std::uint64_t device_flux_outers       = 0;
    /// Uploads the generations cancelled, by array.
    /// PER OUTER.  Geometry::Phif moves inside a segment on every outer whose
    /// drive takes the host CMFD loop, so this decision cannot be staged.
    std::uint64_t flux_uploads_elided      = 0;
    /// PER SEGMENT since W3 item 4, plus once per cusping that fired -- those are
    /// the only two points at which _xs and _dtil can move inside a segment, and
    /// the proof is in runSegment's staging block.  Compare against
    /// `segment_launches` and `cusping_fired`, NOT against `device_outers`.
    std::uint64_t xsnf_uploads_elided      = 0;
    std::uint64_t dtil_uploads_elided      = 0;
    /// Segment exits that mirrored psi and dhat back.  The receipt arithmetic:
    /// host_mirror_bytes should be about (mirror_exits + host-loop outers) times
    /// one psi+dhat pair, not device_outers times it.
    std::uint64_t mirror_exits             = 0;
    /// Rev.7.1 Task 18-lite.  Outers whose nodal drive ran with the canonical
    /// binding live -- the ones that paid no jnet bridge and no flux upload
    /// inside the nodal call.
    std::uint64_t canonical_nodal_outers   = 0;

    /// Rev.7.1 W3 item 2: outers whose nodal drive handed back an EVENT instead
    /// of having drained itself, so the segment stream waited on it and the host
    /// did not block.
    ///
    /// A DIFFERENCE, NOT A TOTAL.  device_outers - nodal_event_waits is the
    /// number of outers that still paid a host round trip for the nodal
    /// handover: the Wielandt warm-up, a drive that fell back to the CPU body,
    /// and any outer a consumer had asked to materialise.  Zero on a run with the
    /// feature wired means the backend never deferred, which is what a
    /// half-installed hook table looks like.
    std::uint64_t nodal_event_waits        = 0;

    /// Rev.7.1 W3 item 3: outers whose 1/eigv was written by k_outer_publish_reigv
    /// and consumed straight out of device memory by the nodal drive -- i.e. the
    /// outers that no longer carry the eigenvalue through the host to get it
    /// there.  Re-issues after a host-finished sweep are counted separately, so
    /// `reigv_device_outers + reigv_reissued` is the number of launches and
    /// `reigv_device_outers` is the number of outers.
    std::uint64_t reigv_device_outers      = 0;
    std::uint64_t reigv_reissued           = 0;

    /// Rev.7.1 W3 item 4: HOST RENDEZVOUS INSIDE THE BODY, COUNTED.
    ///
    /// THE NUMBER TASK 10 HAS TO DRIVE TO ZERO, and the reason it is a counter
    /// rather than a claim: every other receipt here says what the DEVICE did,
    /// and none of them can say how often the host still had to look.  Divide by
    /// `device_outers` for the per-outer figure.
    ///
    /// The four sites, in body order:
    ///   1. the exit observation at the top of every pass but the first -- the
    ///      halt gate makes an overrun outer a no-op KERNEL, but the nodal drive
    ///      and a declining sweep are host CALLS and a host call cannot read a
    ///      device word.
    ///   2. the mirror drain, on the outers whose drive takes the host CMFD loop
    ///      (invariant 6).  ~1% of outers on kngr_238.
    ///   3. the pre-nodal drain, which since W3 item 3 exists only for the sweep
    ///      OBSERVATION -- BICGCMFD::finishDrive reads the sweep's scalars out of
    ///      a pinned block the stream filled.  This is the one that has to move.
    ///   4. the re-issue drain, on the three-in-twelve-thousand outers where the
    ///      device could not finish the sweep.
    std::uint64_t in_body_host_syncs       = 0;
    /// The same total, split by site, because the four have different fixes.
    /// sync_pre_nodal is the one Task 10 has to remove and the only one that
    /// fires on EVERY outer; sync_exit_observation is what a wide segment adds
    /// back, so a budget of 8 pays MORE rendezvous per outer than a budget of 1
    /// and its advantage today is transfer amortisation, not fewer round trips.
    std::uint64_t sync_exit_observation    = 0;
    std::uint64_t sync_mirror_drain        = 0;
    std::uint64_t sync_pre_nodal           = 0;
    std::uint64_t sync_sweep_reissue       = 0;
    /// Not in the total: this one is the segment EXIT, which the host is
    /// entitled to and Task 10 does not remove.  Carried so the per-outer
    /// rendezvous arithmetic can be done from the receipt alone.
    std::uint64_t sync_segment_exit        = 0;
    /// Bytes returned to Geometry::Phis at a segment exit.
    ///
    /// THE PRICE OF THE ELIDED DOWNLOAD, and it is charged per EXIT rather than
    /// per outer, which is the whole trade: the device nodal writes phis into
    /// the arena every outer and nothing on the host looks at it until the
    /// statepoint (PPR, and XSSet::NormalizeFluxSign just before it).  The
    /// reader list is in GpuCanonicalState.h's CanonicalConsumer table and it is
    /// pinned by tools/test_segment_canonical_nodal_contract.py -- a host reader
    /// that is not covered here is a stale read that produces plausible numbers.
    std::uint64_t phis_mirror_bytes        = 0;
    /// Bytes returned to Geometry::Jnet at a segment exit, same rule and the
    /// same reader list.  Counted apart from jnet_bridge_bytes so the bridge's
    /// disappearance stays legible: the bridge was 2 x nsurf*ng per OUTER, this
    /// is 1 x nsurf*ng per EXIT.
    std::uint64_t jnet_mirror_bytes        = 0;
    std::uint64_t refusals[static_cast<int>(OuterSegmentRefusal::Count)] = {};
    std::uint64_t escapes[kDeviceEscapeCount]                           = {};
};

/// Process-wide counters.  One deck per process on the single-run path, which is
/// the only path Task 9 is eligible on, so a plain object is enough; when Task 10
/// admits a batch this becomes per-slot and the receipt gains a slot axis.
[[nodiscard]] OuterSegmentCounters outerSegmentCounters();

/// The Sec 9.3 payload: `{segment_launches, device_outers,
/// host_outer_observations, budget_exits, halted_outer_launches, escapes{...},
/// refusals{...}, host_body_calls{...}}`.
[[nodiscard]] std::string outerSegmentReceiptJson();

/// THE OTHER HALF OF THE CLAIM, and the half an audit of 8be6bee found missing.
///
/// Every counter above says what the DEVICE did.  None of them can say whether
/// the host did it too -- and at 8be6bee it did: the Task 4/5/7 kernels had no
/// production caller at all, so a run with every device feature on still
/// executed BICGCMFD::updpsi/updjnet/upddhat and Nodal::updateConstant on the
/// CPU and every receipt in the tree looked healthy.
///
/// So the receipt carries the host counters beside the device ones.  A device
/// outer's claim is `host_body_calls` flat across the segment; printing them
/// together is what makes "the feature is on" and "the host still did the
/// arithmetic" two different numbers instead of one silent one.
///
/// Defined here rather than in either arm so the two cannot format it
/// differently, which is the same reason outerEscapeName lives in the header.
/// Why nothing ran, asked at PRINT time rather than remembered from a call site.
///
/// THE HOLE THIS CLOSES.  `refusals{}` is only populated by a call site that
/// actually reached the delegation -- ReconvergeFlux, which a deck that never
/// falls back to a previous trial point never enters.  So a perfectly ordinary
/// run with the feature ON printed `device_outers:0, refusals:{}`, which says
/// "nothing happened" and not one word about WHY, and is byte-for-byte what a
/// run whose every segment succeeded-at-zero-outers would print.
///
/// Asking the runner at print time costs one pure predicate and always answers.
/// `fractional_rods` and `critical_search` are passed false deliberately: they
/// are DECK properties the receipt has no access to here, and they are ranked
/// BELOW every structural refusal (no runner, no arena, unbound, no hooks), so
/// with any of those present the answer is exact.  When none of them is, the
/// segment is running and this field is not printed at all.
[[nodiscard]] inline std::string outerIdleReasonJson(OuterSegmentRefusal why) {
    return std::string("\"idle_reason\":\"") + outerRefusalName(why) + "\"";
}

[[nodiscard]] inline std::string outerHostBodyFields(const hostouter::Snapshot& h) {
    return "{\"updpsi\":" + std::to_string(h.updpsi) +
           ",\"updjnet\":" + std::to_string(h.updjnet) +
           ",\"upddhat\":" + std::to_string(h.upddhat) +
           ",\"upddtil\":" + std::to_string(h.upddtil) +
           ",\"nodal_constants\":" + std::to_string(h.nodal_constants) + "}";
}

/// BOTH SCOPES, because they answer different questions and only one of them is
/// the claim.  `host_body_calls` is the whole process -- it is what catches a
/// device kernel with no production caller (8be6bee).  `host_body_calls_in_segment`
/// is what a device outer actually asserts: zero host arithmetic BETWEEN the
/// segment's entry and its exit.  The run-wide number cannot be zero on any real
/// deck (SolveLoop builds d-tilde before it delegates, and a material change
/// rebuilds the nodal constants before the next segment starts), so a gate held
/// to the run-wide number would be held to something unreachable.
[[nodiscard]] inline std::string outerHostBodyJson() {
    return "\"host_body_calls\":" + outerHostBodyFields(hostouter::snapshot()) +
           ",\"host_body_calls_in_segment\":" +
           outerHostBodyFields(hostouter::snapshotOf(hostouter::segmentCounters()));
}

/// Writes one `[RASBERY][OUTER_GPU] {...}` line.  Emitted whenever the feature
/// was ENABLED, including when every segment was refused -- a receipt that only
/// appears on success cannot tell the difference between "off" and "on but never
/// engaged", and that difference is the whole G0 argument.
void reportOuterSegment(std::ostream& os);

/// Record one refusal without attempting a segment.
///
/// FOR A CALL SITE THAT HOISTS THE DECISION OUT OF ITS LOOP.  refusal() is a
/// pure query and records nothing, so a caller that asks it once and then never
/// calls runSegment() would leave the receipt all zeros -- indistinguishable
/// from a run with the feature off, which is precisely the state the receipt
/// exists to expose.  Calling this once, with the reason, restores that.
void noteOuterSegmentRefusal(OuterSegmentRefusal why);

// ---------------------------------------------------------------------------
// The runner
// ---------------------------------------------------------------------------

/// Backend-neutral stream handle, same convention as GpuPhysicsArena.
using OuterSegmentStream = void*;

/// A stream-ordered enqueue supplied by whoever owns a device solver.
///
/// THE CONTRACT, AND IT IS THE WHOLE POINT OF THIS TASK: enqueue work on
/// `stream` and RETURN.  No cudaStreamSynchronize, no cudaDeviceSynchronize, no
/// D2H the host waits on, no rendezvous.  A hook that drains the stream puts the
/// per-outer host round trip straight back in, and the segment is then correct
/// but worthless.
///
/// The hook must also publish this outer's DeviceOuterProbe for `slot` on the
/// same stream -- eigv, residual, and the four signals -- because the
/// convergence kernel that runs after it has no other way to see them.
///
/// `outer_index` is the 0-based position in the segment, for a hook that needs to
/// do something once (a first-outer upload, say).  Returning false aborts the
/// segment; the host path takes over and `LaunchFailed` is counted.
/// Step 7's hook.  Returns TRUE when ApplyRodCusping fired, i.e. when the
/// macro-XS moved and d-tilde was rebuilt on the host.
///
/// No stream argument, deliberately: it is pure host work over host arrays and
/// it must not enqueue anything.  The runner owns the one device consequence.
/// What can change BETWEEN the outers of one segment.
///
/// WHY THIS IS NOT IN OuterSegmentScalars.  That struct is read ONCE, when
/// runSegment is called.  Everything in it is stale for outers 2..N of a
/// segment whose budget is above one -- and an upload elision decided from a
/// stale generation skips a copy that was needed.  That is not hypothetical:
/// the first attempt at this elision put the generations in OuterSegmentScalars
/// and i-SMR CY02 failed at b8 and b16 while PASSING at b1, because a segment
/// of one cannot have a stale second outer.
///
/// So the runner asks the host again at the top of every outer.  Three loads
/// and a bool, against 681 KiB of copies they can cancel.
struct OuterSegmentLiveState {
    /// Geometry::fluxGeneration() -- bumped by every HOST writer of Phif.
    unsigned long long flux_generation = 0;
    /// XSSet::hoststateGeneration().
    ///
    /// READ, AND DELIBERATELY NOT THE xsnf GATE.  It means "the device mirror of
    /// _xs is stale", not "the host bytes of _xs changed": XSSet does not bump
    /// it when the GPU XS arm writes a freshly reconstructed _xs into host
    /// memory (XSSet.cpp:2772, and UpdateEquilibriumXenon's device arm), because
    /// after such a write the host and THAT device mirror agree.  The segment's
    /// device xsnf is the CMFD arena's, a different buffer, so those writes are
    /// exactly the ones it must not miss -- and it did miss them until the
    /// upload moved to a byte-exact shadow (CudaOuterGraph.cu, Impl::resident_xsnf).
    ///
    /// Kept because it is one host load and it is the honest name for "the XS
    /// moved at all", which the receipt and the state-machine test read.
    unsigned long long xs_generation = 0;
    /// BICGCMFD::dtilGeneration() -- bumped by upddtil(), its only writer.
    unsigned long long dtil_generation = 0;

    /// Did the drive that just returned leave the device flux equal to the host
    /// one?  Only meaningful immediately after the sweep hook.
    ///
    /// BOTH THIS AND THE GENERATION ARE NEEDED, and the first attempt failed
    /// for using only this one.  It answers `the device downloaded the flux`;
    /// the question the elision asks is `has the host written it since`, and
    /// the ladder between two outers -- normalisation, an Xe commit -- writes
    /// it without the device seeing anything.  The generation catches those,
    /// this catches the drive that never downloaded at all.
    bool device_owns_flux = false;

    /// Will the drive about to run take the DEVICE sweep?
    ///
    /// BICGCMFD::canEnqueueDrive() -- the same gate drive() applies.  When it
    /// is true nothing on the host reads _psi or _dhat during that drive: the
    /// sweep works from the device buffers, its uploads are elided, and the two
    /// exceptional states download for themselves
    /// (issueExceptionalOperatorDownloads).  When it is false the host loop
    /// runs and reads _psi through wiel and _dhat through the host assembly,
    /// and the mirrors are what make those reads correct.
    bool sweep_will_enqueue = false;
};

/// Re-read the live state.  Pure host reads; it must not enqueue anything.
using OuterLiveStateHook = void (*)(void* ctx, OuterSegmentLiveState& out);

using OuterCuspingHook = bool (*)(void* ctx, int slot, unsigned int outer_index);

using OuterSegmentHook = bool (*)(void* ctx, OuterSegmentStream stream, int slot,
                                  unsigned int outer_index);

/// Rev.7.1 W3 item 2: how the drive that just ran is ordered against the
/// segment's stream.
///
/// Returns the nodal backend's completion event (an opaque cudaEvent_t) when
/// that drive DEFERRED its own drain, and nullptr when it drained itself.
/// Asked after every nodal hook, because the answer changes per outer: a drive
/// that fell back to the CPU body, or one that materialised jnet/phis for a host
/// reader, blocks and needs no event; a drive whose downloads were all elided
/// left its work in flight and the segment must wait on it before upddhat reads
/// the jnet.
///
/// NULLPTR IS A LEGAL ANSWER AND MEANS `ALREADY ORDERED`, never `do not bother`.
using OuterNodalCompletionHook = void* (*)(void* ctx);

struct OuterSegmentHooks {
    OuterSegmentHook enqueue_cmfd_sweep = nullptr; ///< setls + drive (+ probe)
    OuterSegmentHook enqueue_nodal_drive = nullptr; ///< nodal reset + drive
    /// The other half of the nodal handover; see OuterNodalCompletionHook.
    OuterNodalCompletionHook nodal_completion_event = nullptr;

    /// Step 7: ApplyRodCusping + the upddtil it forces.  Returns whether the
    /// macro-XS actually moved.
    ///
    /// A HOST HOOK, AND IT COSTS NOTHING EXTRA.  The nodal drive immediately
    /// above it is already a host call in an already-synchronised window, so
    /// cusping runs in the same window at the same point of the outer the host
    /// loop runs it (Driver.h, between the nodal drive and upddhat).  That is
    /// the whole design: not a device port of cusping, and not an escape that
    /// hands the outer back -- just the host's own call, where the host makes
    /// it.
    ///
    /// WHY IT HAS ITS OWN SIGNATURE.  Every other hook returns `did it work`.
    /// This one returns `did it fire`, because the answer changes what the
    /// runner must do next: a cusping that fired rewrote the host d-tilde, and
    /// the device upddhat two steps later reads the DEVICE one.
    OuterCuspingHook apply_cusping = nullptr;

    /// Re-read the generations at the top of every outer, and again after the
    /// sweep.  Without it the segment cannot elide a single upload safely.
    OuterLiveStateHook read_live_state = nullptr;

    /// Rev.7.1 Task 10 part 2: the post-synchronise half of the sweep.
    ///
    /// A STREAM-ORDERED SWEEP HAS TWO HALVES AND ONLY ONE OF THEM IS AN
    /// ENQUEUE.  enqueue_cmfd_sweep issues the launch and returns; the flux it
    /// produced still has to be adopted into Geometry::Phif and the eigenvalue
    /// into the host's `eigv` before the NODAL hook -- which is host arithmetic
    /// over host arrays -- can run.  Both are reads of memory the runner's own
    /// per-outer synchronise has already made valid, so this hook costs no
    /// transfer and no extra sync; it exists so the runner does not have to know
    /// what a CMFD drive's observation consists of.
    ///
    /// Supplying it is what makes `sweep_synchronizes` false legal.
    OuterSegmentHook finish_cmfd_sweep = nullptr;

    /// TRUE while the sweep hook is a HOST call that rendezvouses.
    ///
    /// BICGCMFD::drive drains its stream and copies the flux back, so a segment
    /// containing it cannot enqueue outer i+1 before observing outer i.  The
    /// runner therefore forces the budget to 1: a longer budget would enqueue
    /// outers whose inputs the halt gate has not yet been told about, which is a
    /// different trajectory, not a faster one.
    ///
    /// It goes FALSE when the caller supplies finish_cmfd_sweep, i.e. when the
    /// sweep is BICGCMFD::enqueueDrive rather than BICGCMFD::drive.  The segment
    /// still synchronises once per outer -- the nodal drive is host arithmetic
    /// and nothing can remove that until Task 18 -- but the sweep no longer
    /// rendezvouses, no longer drains a second time, and no longer hands its
    /// verdict back through the host, so outer i+1 may be enqueued behind outer
    /// i's kernels under the halt gate.
    bool sweep_synchronizes = false;

    /// Rev.7.1 Task 18-lite: which side owns the canonical nodal regions NOW.
    ///
    /// Called with 1 immediately before the nodal hook of every outer, and with
    /// 0 exactly once at the segment exit (and on the failure paths, after the
    /// runner has brought the host arrays back).  The runner does not know what
    /// a canonical buffer is; the hook does, because it is the same call site
    /// that adopted them.
    ///
    /// WHY A HOOK AND NOT A CALL FROM Driver.h AROUND runSegment.  The exit is
    /// not where the caller regains control: ReconvergeFlux `return`s straight
    /// out of the loop on FluxConverged, and SolveLoop takes four different
    /// paths out of the same call.  A release that has to be written at every
    /// one of those is a release that will be missing from the fifth -- and a
    /// missing release is an elided upload against a host array somebody has
    /// since rewritten, which is the exact failure mode this whole mechanism
    /// exists to make impossible.
    ///
    /// Null means the segment has no canonical nodal binding, which is the
    /// default and the feature-off shape.
    void (*canonical_nodal_mode)(void* ctx, int in_segment) = nullptr;

    /// Will the drive THIS outer is about to take run on the device against the
    /// canonical buffers?  Non-zero yes.
    ///
    /// ASKED BEFORE THE BRIDGE IS DECIDED, and asked every outer, because the
    /// answer changes inside a segment.  Nodal::TryDriveGpu falls back to the
    /// CPU body on any deck with a fractional rod -- and the CPU body reads
    /// Geometry::Jnet, which is stale the moment the bridge stops running.  A
    /// rod SEARCH moves the bank inside SolveLoop, so a segment armed with every
    /// rod integral meets fractional ones a few outers later; i-SMR CY02 is that
    /// deck, and an arm-scope answer converged it somewhere else entirely.
    ///
    /// It must be TryDriveGpu's own predicate rather than a copy of it -- the
    /// same relationship BICGCMFD::canEnqueueDrive() has to drive() -- so the
    /// bridge and the drive cannot disagree about one outer.
    int (*canonical_nodal_eligible)(void* ctx) = nullptr;

    /// Rev.7.1 W3 item 3: WHERE THE NODAL DRIVE READS 1/eigv.
    ///
    /// Returns the nodal backend's device reigv slot, or nullptr when there is
    /// none right now -- before the first drive, on the hybrid arm, or with the
    /// batch arena driving.  A null answer is not a failure: it means the host
    /// still owns the reciprocal and the segment leaves the transfer alone.
    ///
    /// ASKED EVERY OUTER, NEVER CACHED.  The backend frees and re-lays-out that
    /// block when nsurf changes, and the runner writing a remembered address
    /// after that is a write into a freed allocation.  It is here rather than in
    /// the runner for outerCanonicalNodalHook's reason: the backend is reached
    /// through XSSet, which is Driver.h's object.
    void* (*nodal_reigv_slot)(void* ctx) = nullptr;

    /// Rev.7.1 W3 item 3: tell the drive who wrote that slot.
    ///
    /// Called with 1 before the nodal hook of an outer whose reciprocal this
    /// segment published on the device, and with 0 before every other drive and
    /// at the segment exit.  The pairing with `canonical_nodal_mode` is exact and
    /// deliberate: both are per-drive declarations of what the runner has already
    /// done to device memory, and a drive that gets one without the other either
    /// uploads over the device's reciprocal (harmless, just the old cost) or
    /// reads a slot nobody wrote (not harmless) -- so they are set together, from
    /// one predicate, at one point of the outer.
    void (*nodal_reigv_mode)(void* ctx, int device_resident) = nullptr;

    void*            ctx                 = nullptr;
};

/// The arena-derived half of a segment, set ONCE by whoever owns the arena.
///
/// THE SPLIT IS THE LAYERING, and it is what keeps Driver.h honest.  Driver.h
/// has a Geometry, an XSSet and a BICGCMFD; it has no DeviceArenaView, no
/// CmfdOuterSlotTable and no contraction mask, and a call site that had to build
/// them would be reaching through three abstraction layers to fill a struct it
/// cannot validate.  So the arena-shaped fields live here, are bound once by the
/// component that reserved the arena, and Driver.h passes only the fifteen plain
/// scalars its own loop already holds (OuterSegmentScalars).
///
/// Nothing binds this today -- GpuPhysicsArena::reserve() has no production call
/// site yet -- which is why `Unbound` exists as a named refusal instead of a
/// crash on a null table.
struct OuterSegmentBinding {
    cmfd::CmfdGeometryView geom{};     ///< the CMFD bodies' cached geometry
    DeviceGeometryView     geometry{}; ///< the arena's geometry, for the prologue
    CmfdOuterSlotTable     table{};
    unsigned long long     forms         = 0; ///< cmfdOuterFormsRuntime()
    bool                   dhat_clamp    = false; ///< RASBERY_DHAT_CLAMP
    CmfdOuterCounters*     dhat_counters = nullptr;
    /// Geometry::Jnet.  The runner bridges the device jnet to it around the
    /// nodal hook; see OuterSegmentResidency::host_jnet.
    double*                host_jnet     = nullptr;
    /// The physics arena's jnet for the bound slot, the other end of the bridge.
    double*                device_jnet   = nullptr;
    /// Geometry::Phif; see OuterSegmentResidency::host_flux for why the segment
    /// uploads it rather than trusting the device copy.
    const double*          host_flux     = nullptr;
    const double*          host_xsnf     = nullptr;
    const double*          host_dtil     = nullptr;
    double*                device_dtil   = nullptr;
    double*                device_xsnf   = nullptr;
    double*                host_dhat     = nullptr;
    double*                host_psi      = nullptr;
    double*                device_dhat   = nullptr;
    double*                device_psi    = nullptr;

    // --- Rev.7.1 Task 18-lite: the canonical nodal set ----------------------
    //
    // THE THREE BUFFERS Nodal::drive READS AND WRITES, as device addresses.
    // Two of them the segment already had under other names -- `device_jnet` is
    // the physics arena's jnet, and the sweep's phi arrives through
    // OuterSegmentResidency::flux -- and the third is the arena's own phis
    // region, which no other device body touches.  They are named here so the
    // ADOPTION has one source of truth: the same three pointers go to
    // XsReconBackend::adoptCanonicalBuffers and to the exit mirror, and a
    // reader can see that they do.
    //
    // LAYOUT, because a canonical buffer that is indexed two ways is worse than
    // no sharing at all.  NodalKernel.h indexes flux as [lk*NG + ig] and
    // CmfdOuterKernel.h indexes it as [l*ng + ig]; jnet and phis are both
    // [ls*ng + ig] on both sides and both are the same LR*ng*NDIRMAX*nxyz
    // allocation Geometry gives them.  No transpose, no rebase.
    double*                device_flux   = nullptr; ///< the sweep's phi
    double*                device_phis   = nullptr; ///< the arena's phis
    double*                host_phis     = nullptr; ///< Geometry::Phis
    /// True once a backend has actually adopted the three above.  Set by
    /// setCanonicalNodalBound(), never inferred from the pointers: they are
    /// non-null whenever the arena exists, and the binding is only LIVE when
    /// the nodal drive that will run inside the segment is the device one.
    bool                   canonical_nodal = false;

    int                    nxyz          = 0;
    int                    ng            = 0;
};

/// What the HOST loop knows and the device does not.
///
/// Every field is a variable Driver.h already has in scope at the top of an
/// outer.  Nothing here is derived, so a caller cannot get it subtly wrong; what
/// it CAN get wrong is passing a field that is not yet knowable, which is why
/// there is no `search_pending` (see OuterSegmentEligibility::critical_search).
struct OuterSegmentScalars {
    int slot = 0;

    /// Rev.7.1 Task 18-lite: the caller's answer to rasberyOuterSlotAdmitted.
    ///
    /// CARRIED RATHER THAN RECOMPUTED, so the ladder the body asks and the
    /// ladder the caller armed on cannot disagree about one outer.  The runner
    /// has no Geometry to compare shapes with; the Driver does, and it asked
    /// before it armed.  Defaults to admitted, which is what a single run is.
    int slot_admitted = 1;

    double eigv       = 0.0; ///< at segment entry
    double residual   = 0.0; ///< at segment entry
    double prev_inner = 0.0; ///< Driver.h's `prev_inner` local

    double       keff_tol       = 0.0; ///< schedule.tolerance_keff
    double       flux_tol       = 0.0; ///< max(keff_tol, CMFD_FLUX_L2_TOLERANCE)
    unsigned int max_outer_iter = 0;   ///< schedule.max_outer_iter (the stall ladder)

    // Sec 6.13's perturbation gates.  Constant across a flux-reconvergence
    // segment: each of them only moves when its own step FIRES, and a step that
    // fires ends the segment.
    int          xe_pending      = 0;
    double       xe_interim_l2   = 0.0; ///< RASBERY_XE_INTERIM_L2, 0 = off
    int          xe_once_mode    = 0;
    unsigned int xe_budget_probe = 0;
    int          th_pending      = 0;

    /// Driver.h's `flux_stall` at segment entry -- outers since the flux last
    /// converged, the counter the limit-cycle test in SolveLoop's ladder reads.
    ///
    /// UPLOADED FOR THE SAME REASON prev_inner IS.  The device machine advances
    /// its own copy inside cmfdOuterConvergence, and a copy that started from
    /// the previous segment's leftovers cannot end the segment at the outer the
    /// host's ladder would have ended it at.  Seeded, `++st.flux_stall >
    /// max_outer_iter` fires on exactly that outer; the host then adds
    /// `device_outers` to its own counter and reaches the same verdict.
    ///
    /// ReconvergeFlux leaves it 0: that loop has no limit-cycle handling at all
    /// and hands the device its own iteration bound as `max_outer_iter`, so the
    /// ladder is deliberately unreachable there.
    unsigned int flux_stall = 0;

    /// Issue the updateConstant prologue before the first outer.  False when the
    /// caller knows the constants are current (nodal_constant_generation
    /// unchanged since the last drive).
    bool run_nodal_constants = false;
};

/// What the host adopts when a segment exits.
///
/// EXACTLY THE SolveLoop LOCALS THE DEVICE CARRIED, and nothing else.  Every one
/// of them is a variable Driver.h keeps across outers (Driver.h:1391-1400,
/// 1486-1488), so adopting them is not a translation -- it is handing the host
/// back its own state.  A field missing here is a trajectory that diverges
/// silently at the next outer, which is why cmfd_outer_replay.cpp already checks
/// the DeviceSlotState round trip for the same seven fields.
struct OuterSegmentResume {
    unsigned int device_outers = 0; ///< outers COMMITTED; always >= 1 on success
    unsigned int next_phase    = 0; ///< DevicePhase
    unsigned int escape        = 0; ///< DeviceEscape

    /// Driver.h's `flux_converged`, computed on the device from the SAME three
    /// inputs (Driver.h:1818 |prev_inner - eigv| < keff_tol && residual < flux_tol).
    ///
    /// THE ONE FIELD SolveLoop's LADDER TAKES FROM THE DEVICE, and it is exact
    /// for a reason worth stating: cmfdOuterConvergence computes it from
    /// st.prev_inner -- which the host uploads at segment entry -- and from the
    /// eigv/residual the sweep just produced.  None of the other state the
    /// device machine carries (flux_stall, stall_events, clean_iters,
    /// xe_interim_count) participates in it, so those may drift on the device
    /// without touching this answer.
    int flux_converged = 0;

    double eigv       = 0.0;
    double residual   = 0.0;
    double prev_inner = 0.0;

    unsigned int flux_stall         = 0;
    unsigned int stall_events       = 0;
    unsigned int stall_sample_taken = 0;
    unsigned int clean_iters        = 0;
    unsigned int xe_interim_count   = 0;
    unsigned int total_outer        = 0;
};

// ---------------------------------------------------------------------------
// Link 2: the sweep's buffers ARE the segment's buffers
// ---------------------------------------------------------------------------
//
// WHAT MAKES THIS A HANDOFF.  CudaBatchArena::residentView() hands out the
// device addresses of one slot's phi/psi/dtil/dhat/xsnf, and every one of them
// already has the layout CmfdOuterView wants -- node-major flux [l*ng+ig],
// [ls*ng+ig] operators, group-major xsnf.  So binding is a pointer write: the
// segment's updpsi writes the psi the sweep reads, its upddhat writes the dhat
// cmfd_assemble_operator_2g reads, and the 416 KiB/outer dhat H2D disappears
// because there is nothing left to copy.
//
// WHY THE SEGMENT REFUSES WITHOUT THE RESIDENT SWEEP.  The segment's kernels
// read the DEVICE flux.  Only the resident sweep maintains it: with
// RASBERY_GPU_CMFD_SWEEP off, BICGCMFD::drive runs the host CMFD loop, the
// device phi is never written, and updpsi/updjnet/upddhat would compute this
// outer from whatever flux was last uploaded.  That is not a slow path, it is a
// wrong one, so `SweepNotResident` refuses it by name.  The same applies to the
// device assembly: without it the sweep builds diag/cc from the HOST dhat, and
// the host dhat is exactly what the segment stopped writing.
struct OuterSegmentResidency {
    double*       flux = nullptr; ///< the sweep's phi   [l*ng + ig]
    double*       psi  = nullptr; ///< the sweep's psi   [l]
    double*       dtil = nullptr; ///< the sweep's dtil  [ls*ng + ig]
    double*       dhat = nullptr; ///< the sweep's dhat  [ls*ng + ig]
    const double* xsnf = nullptr; ///< the sweep's xsnf  [ig*nxyz + l]

    /// The HOST jnet the nodal drive reads and writes (Geometry::Jnet).
    ///
    /// The sweep arena has no jnet -- jnet is not a CMFD input -- so the segment
    /// keeps it in the physics arena and bridges it around the nodal hook, which
    /// is still a host call.  That bridge is the honest remaining cost of link 2
    /// and it is counted (`jnet_bridge_bytes`); it goes away when the nodal drive
    /// becomes a stream-ordered enqueue.
    double* host_jnet = nullptr;

    /// Geometry::Phif -- the flux the segment's updpsi is about to read.
    ///
    /// WHY IT IS UPLOADED AND NOT ASSUMED CURRENT.  The device phi is only
    /// refreshed when the resident sweep launches, and drive() falls back to the
    /// HOST loop for the Wielandt warm-up and whenever the device sweep declines
    /// (BICGCMFD.cpp:558-565).  After such a drive the host flux has moved and
    /// the device copy has not, so a segment that trusted the device buffer
    /// would compute this outer from the flux of some earlier one.  One
    /// ngxyz-double H2D per outer buys the guarantee; it is counted, and it is
    /// still less than the dhat H2D the handoff removed.
    const double* host_flux = nullptr;

    /// CMFD::dhatData() / CMFD::psiData() -- the host twins of the two arrays
    /// the segment writes.  See CMFD.h: the host drive path still reads them
    /// during the Wielandt warm-up and whenever the device sweep declines.
    /// XSSet::xsnfData() -- the fission cross section updpsi is about to read.
    ///
    /// SAME ORDERING PROBLEM AS THE FLUX, and it bites harder.  The sweep
    /// refreshes xs_xsnf from the host INSIDE drive(), which is step 2 of the
    /// segment -- but updpsi is step 1.  So without this the segment's fission
    /// source would be built from the PREVIOUS outer's cross sections, and
    /// every Xe or T/H step between outers would be one outer late.
    const double* host_xsnf = nullptr;

    /// CMFD::dtilData() -- the d-tilde updjnet and upddhat read.
    ///
    /// THE THIRD BUFFER WITH A CONDITIONAL REFRESH, and the one that cost a
    /// whole outer.  issueSweepUploads pushes dtil ONLY inside its
    /// `if (sl.device_assembly)` branch, and device assembly is off for the
    /// entire Wielandt warm-up -- so on the first outers dtil_dev holds
    /// whatever was in the arena while the host had a freshly computed _dtil.
    /// The device updjnet then produced a different jnet, upddhat a different
    /// dhat, and i-SMR CY01 took one extra outer per statepoint with the same
    /// converged k_eff.
    const double* host_dtil = nullptr;

    double* host_dhat = nullptr;
    double* host_psi  = nullptr;

    /// Geometry::Phis -- the other half of the canonical nodal set's host end.
    ///
    /// It arrives with the residency and not with the stand-up binding because
    /// it is a GEOMETRY pointer, and the stand-up half of the binding is built
    /// from the arena alone (see the split note on OuterSegmentBinding).
    double* host_phis = nullptr;

    int  arena_slot = -1;
    bool valid      = false;
};

/// Rebind the bound slot table onto the sweep arena's buffers.
///
/// Called once, after the sweep arena exists and before the first segment.
/// Returns false and leaves the segment refusing when the view is not usable.
bool rasberyBindOuterResidency(const OuterSegmentResidency& residency);

/// Publish one outer's observation from inside the sweep hook.
///
/// The hook has just returned from a HOST call that synchronised, so this is a
/// plain synchronous H2D of six words -- there is no stream to be ordered
/// against any more.  It is the runner's, not the hook's, so the probe layout
/// stays private to this file.
bool rasberyPublishOuterProbe(int slot, double eigv, double residual, bool negative_flux,
                              bool rayleigh);

/// The five integers that decide an arena LAYOUT, and nothing else.
///
/// Rev.7.1 Task 18-lite.  The arena is one allocation whose slot block is
/// replicated `slots` times from ONE ArenaDims (arenaComputeLayout), so every
/// slot has the same strides.  A batch may therefore share it only while every
/// deck agrees on these five; a sixth Driver with a different mesh has to be
/// told no by name.  Split out of OuterSegmentDeck so the comparison can be made
/// at the ARM site -- which holds a Geometry and no deck -- against the shape the
/// stand-up recorded.
struct OuterSegmentDeckShape {
    int nxyz   = 0;
    int nsurf  = 0;
    int nxy    = 0;
    int n_fuel = 0;
    int ng     = 0;

    [[nodiscard]] bool operator==(const OuterSegmentDeckShape& o) const {
        return nxyz == o.nxyz && nsurf == o.nsurf && nxy == o.nxy &&
               n_fuel == o.n_fuel && ng == o.ng;
    }
    [[nodiscard]] bool operator!=(const OuterSegmentDeckShape& o) const {
        return !(*this == o);
    }
};

/// The one object that runs a segment.
///
/// Defined by the CUDA arm (CudaOuterGraph.cu) and by the no-CUDA stub, so call
/// sites never need an #ifdef -- the same split GpuPhaseScheduler.h/.cu and
/// GpuPhysicsArena.h/.cu use.
class CudaOuterSegment {
public:
    CudaOuterSegment();
    ~CudaOuterSegment();

    CudaOuterSegment(const CudaOuterSegment&)            = delete;
    CudaOuterSegment& operator=(const CudaOuterSegment&) = delete;

    /// Take the device scratch this runner needs (probe, segment state, refreshed
    /// inputs) and the stream it will issue on.  Returns false and leaves the
    /// runner unavailable on any CUDA error, with the reason in status().
    ///
    /// Rev.7.1 Task 18-lite: `slot` says which arena slot THIS runner serves.
    /// There is one runner per slot now, and the device scratch below is still
    /// sized [slot_count] because the probe and halt arrays are addressed by the
    /// slot index the CMFD sweep's verdict kernel is given -- a runner that
    /// packed its own slot at index 0 would need a second index space, and the
    /// two would be free to disagree.
    bool initialize(const DeviceArenaView& arena, int slot_count, int slot);
    void release();

    [[nodiscard]] bool               available() const;
    [[nodiscard]] const std::string& status() const;

    /// Hand over the arena-derived views.  Until this is called every segment
    /// refuses with `Unbound`.
    void bind(const OuterSegmentBinding& binding);
    [[nodiscard]] bool bound() const;

    void setHooks(const OuterSegmentHooks& hooks);

    /// Link 2: point the CMFD slot table at the sweep arena's buffers.
    bool bindResidency(const OuterSegmentResidency& residency);
    [[nodiscard]] bool residencyBound() const;

    /// Rev.7.1 Task 18-lite: the three device buffers a nodal backend must
    /// adopt to run inside this segment without a single per-outer transfer.
    ///
    /// All-null until the residency is bound, because the flux half of the set
    /// is the SWEEP's phi and only the residency knows it.  Coherent by
    /// construction -- all three or none -- which is what
    /// gpu::canonicalNodalSetIsCoherent demands and what stops a caller pairing
    /// the canonical jnet with the nodal arena's own flux.
    [[nodiscard]] CanonicalSlotBuffers canonicalNodalSet() const;

    /// Say whether a backend actually took the set above.  Until it is true the
    /// runner bridges jnet around the nodal hook exactly as it did before, which
    /// is what makes a refused adoption a slow path rather than a wrong one.
    void setCanonicalNodalBound(bool bound);
    [[nodiscard]] bool canonicalNodalBound() const;

    /// Write one outer's observation into the probe the decision kernel reads.
    bool publishProbe(int slot, double eigv, double residual, bool negative_flux,
                      bool rayleigh);

    /// Rev.7.1 Task 10 part 2: run the segment's kernels on SOMEBODY ELSE'S
    /// stream.
    ///
    /// ONE STREAM IS THE WHOLE ORDERING ARGUMENT.  The sweep is enqueued by the
    /// arena on the arena's stream, and everything around it -- updpsi before,
    /// updjnet and upddhat after -- reads and writes the buffers that sweep
    /// touches.  Two streams would need an event pair per outer to say so;
    /// adopting the arena's makes the ordering the stream's own and costs
    /// nothing.  Passing null restores the runner's private stream.
    ///
    /// Refused while a segment is in flight, and refused when the runner is not
    /// initialised, because either would retarget work already queued.
    bool useStream(void* stream);

    /// The device addresses the sweep's verdict kernel publishes into.
    ///
    /// Handed out rather than passed through the hook signature because the
    /// probe LAYOUT is this file's (DeviceOuterProbe), and a hook that built the
    /// four addresses itself would be a second place that knows it.
    struct ProbeAddresses {
        double*        eigv     = nullptr;
        double*        residual = nullptr;
        std::uint32_t* negative = nullptr;
        std::uint32_t* rayleigh  = nullptr;
        std::uint32_t* nonfinite = nullptr;
        std::uint32_t* halt      = nullptr;
        bool           valid     = false;
    };
    [[nodiscard]] ProbeAddresses probeAddresses(int slot) const;

    /// Undo a device verdict the host had to overrule.
    ///
    /// The verdict kernel published the eigenvalue of a HALF drive and latched
    /// the segment's halt on it (sweep state 0 or 2); the host has since
    /// finished that drive, so both have to be replaced by what actually
    /// happened.  Same four values publishProbe takes, plus the halt release.
    bool republishAfterHostSweep(int slot, double eigv, double residual,
                                 bool negative_flux, bool rayleigh);

    [[nodiscard]] OuterSegmentHooks hooks() const;

    /// Would a segment run right now?  Pure query, no CUDA call, so the caller
    /// can hoist it out of the outer loop.  Records nothing.
    ///
    /// Rev.7.1 Task 18-lite: `slot_admitted` comes from the caller and
    /// `arena_slots` from this runner's own slot count, so the pre-arm and
    /// post-arm spellings of the ladder are asked with the same two facts.
    [[nodiscard]] OuterSegmentRefusal refusal(int batch_width, bool fractional_rods,
                                              bool critical_search,
                                              bool slot_admitted) const;

    /// Which arena slot this runner serves; -1 until initialize().
    [[nodiscard]] int slot() const;

    /// Run ONE segment.  Returns true when at least one outer was committed on
    /// the device, with `resume` filled; false when the segment was refused or a
    /// launch failed, in which case `resume` is untouched and the caller must run
    /// the host loop for this outer exactly as before.
    ///
    /// Issues the whole body for `budget` outers on ONE stream and synchronises
    /// ONCE, at the end.  Outers past the halt are launched and return
    /// immediately (see the header note on the halt gate); they are counted.
    bool runSegment(const OuterSegmentScalars& scalars, int batch_width,
                    bool fractional_rods, bool critical_search,
                    OuterSegmentResume& resume);

private:
    struct Impl;
    Impl* _impl;
};


/// THE RUNNER FOR ONE ARENA SLOT.
///
/// Rev.7.1 Task 18-lite: THIS USED TO BE ONE OBJECT FOR THE WHOLE PROCESS, and
/// that is what refused `--batch-mode` outright.  A CudaOuterSegment holds a
/// residency, a hook set, a binding, a canonical-nodal latch and a reigv latch,
/// and every one of them is a property of ONE Driver's deck; M Drivers on M host
/// threads sharing them meant the last arm won and the other M-1 ran against a
/// residency that was not theirs.  One runner per slot makes every one of those
/// fields per-slot by construction, which is cheaper to be sure of than M
/// parallel state tables inside one object.
///
/// SLOT 0 IS THE SINGLE-RUN PATH, UNCHANGED.  A run with no batch stands the
/// arena up at width 1, acquires CMFD slot 0 and therefore uses runner 0 -- the
/// same object, initialised with the same slot_count of 1, doing the same work.
/// That is what makes the single-path gate a byte comparison rather than an
/// argument.
///
/// An out-of-range index answers with a runner that was never initialised, so it
/// refuses `no_runner` instead of aliasing somebody else's slot.  It is an
/// exported function and not an inline static because the table has to be the
/// SAME table in the .cu and in whatever else links the header.
[[nodiscard]] CudaOuterSegment& rasberyOuterSegment(int slot);

/// The single-run spelling.  Slot 0, by definition.
[[nodiscard]] inline CudaOuterSegment& rasberyOuterSegment() {
    return rasberyOuterSegment(0);
}

/// The width the physics arena was actually stood up at; 0 before the stand-up.
///
/// Not `rasberyBatchWidth()`: the run ASKS for a width and the arena's Sec 4.4
/// admission may refuse it, and the refusal ladder has to compare the two.
[[nodiscard]] int rasberyOuterArenaSlots();

/// Is there a seat in the stood-up arena for this Driver?
///
/// True when the slot is inside the arena AND the arena was stood up on a deck
/// of this shape.  Pure query, no CUDA call, safe to ask before arming -- which
/// is the whole point: the arm binds residency and adopts a canonical nodal set,
/// so a Driver that is going to be refused must find out first.
///
/// TODAY THIS RUNG IS UNREACHABLE, AND IT IS KEPT ANYWAY.  rasberyBatchArena
/// throws on the second geometry -- "batch mode requires every instance to share
/// one geometry" -- inside BICGSolver's constructor, which runs BEFORE
/// Driver::Run reaches the stand-up, so a mixed-geometry batch dies with that
/// message and this is never asked.  MEASURED: kngr3 + i-SMR CY01 at
/// --batch-mode 2 ran i-SMR to completion with the segment engaged and refused
/// the kngr3 deck by name.  The rung stays because the assumption it guards is
/// the PHYSICS arena's own -- one ArenaDims lays out every slot -- and it must
/// not be inherited silently from a check in another arena that a future
/// heterogeneous batch could relax.
[[nodiscard]] bool rasberyOuterSlotAdmitted(int slot, const OuterSegmentDeckShape& shape);

/// Tell this thread which slot its refusals and counters belong to.
///
/// THE COUNTERS ARE PER SLOT AND THE BUMP SITES DO NOT KNOW IT.  There are ~50
/// `bump(counters().x)` sites inside the body and threading a slot index through
/// all of them would be a large diff whose only purpose is bookkeeping.  A
/// Driver owns one host thread and one slot for its whole life, so the thread IS
/// the index: this sets it once per solve loop, `counters()` reads it, and the
/// [OUTER_GPU] receipt sums the slots back into the run-wide totals it always
/// printed.  Unset means slot 0, which is what a single run is.
void outerSetThreadSlot(int slot);

// ---------------------------------------------------------------------------
// Sec 4  Standing the segment up in production  (Rev.7.1 Task 9, link 1)
// ---------------------------------------------------------------------------
//
// THE STATE THIS REPLACES.  GpuPhysicsArena::reserve() had zero callers in the
// whole tree, so nothing ever built a DeviceSlotView, nothing published a
// DeviceArenaView, and the runner refused every segment with `no_runner`.  The
// Task 4/5/7 kernels were therefore dead code with a passing replay behind
// them.  This is the one call that makes the runner exist.
//
// ONE SLOT, DELIBERATELY.  Task 9 is eligible on the single-run path only
// (OuterSegmentEligibility::batch_width), so reserving 64 slots would take
// 64x the VRAM to leave 63 of them permanently empty.  Sec 4.4 admission is
// run either way and the refusal is loud: a deck that does not fit is refused,
// never silently shrunk.

/// What the production owner hands over.  Every field is something Driver.h
/// already holds at the top of Drive(); nothing here is derived, so a caller
/// cannot get it subtly wrong.
struct OuterSegmentDeck {
    // --- cohort shape (ArenaDims) ---
    int nxyz   = 0;
    int nsurf  = 0;
    int nxy    = 0;
    int n_fuel = 0;
    int ng     = 0;

    [[nodiscard]] OuterSegmentDeckShape shape() const {
        OuterSegmentDeckShape s;
        s.nxyz   = nxyz;
        s.nsurf  = nsurf;
        s.nxy    = nxy;
        s.n_fuel = n_fuel;
        s.ng     = ng;
        return s;
    }

    // --- the CMFD topology cache, uploaded ONCE (CMFD.h base pointers) ---
    //
    // These are the five arrays cmfd::CmfdGeometryView is the device twin of,
    // in the same layout, so the upload is a byte copy and not a translation.
    const int*    surface_node    = nullptr; ///< [ls*LR + side]
    const int*    surface_dir     = nullptr; ///< [ls*LR + side]
    const double* node_hmesh      = nullptr; ///< [l*NDIRMAX + dir]
    const double* node_volume     = nullptr; ///< [nxyz]
    const double* boundary_albedo = nullptr; ///< [dir*LR + side]

    bool dhat_clamp = false; ///< RASBERY_DHAT_CLAMP, as CMFD resolved it

    /// Sec 3.5 link 5: the host counter behind DeviceSlotState::material_generation.
    ///
    /// XSSet::hoststateGeneration(), and NOT a new counter.  Every host site
    /// that writes `_xs` already bumps it -- Reconstruct (XSSet.cpp:1060), the
    /// reference rebuild (:1474), both UpdateFlatXS arms (:2774, :2801),
    /// ResetCuspingNodesToBase (:3180), ApplyRodCusping (:3280), the Xe
    /// reconstruct (:3759), equilibrium Xe (:3955) and depletion (:4043) -- and
    /// SetBoron/SetRod reach it through UpdateFlatXS.  Inventing a second
    /// counter beside a correct one is how two counters disagree.
    unsigned long long material_generation = 0;
};

/// Reserve the arena, upload the geometry, publish the views, bind the runner.
///
/// Emits the Sec 4.4 admission receipt (`[RASBERY][GPU_ARENA]`) on `receipt`
/// whatever the outcome, because a refusal that prints nothing is the failure
/// mode Sec 4.4 exists to prevent.  Returns true when the runner is available
/// and bound afterwards.
///
/// IT DOES NOT MAKE SEGMENTS RUN.  With the runner standing, the refusal moves
/// from `no_runner` to `no_sweep_hook`, which is the honest next blocker: the
/// resident CMFD sweep still has no stream-ordered enqueue (link 2).  Moving a
/// refusal one step down the list IS the deliverable -- the list is now the
/// real one rather than one that stopped at the first missing thing.
bool rasberyStandUpOuterSegment(const OuterSegmentDeck& deck, std::ostream& receipt);

/// Tear it down.  Safe to call when nothing was ever stood up.
void rasberyTearDownOuterSegment();


#if defined(__CUDACC__)

// ---------------------------------------------------------------------------
// Kernels (nvcc only)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The slot table IS the arena  (Rev.7.1 Task 9, links 3 and 4)
// ---------------------------------------------------------------------------
//
// ONE SOURCE OF TRUTH, DERIVED ON THE DEVICE.  A CmfdOuterSlotTable built by
// hand beside a reserved arena is a SECOND set of pointers to the same slot,
// and the two would be free to disagree -- a table whose `dhat` is not the
// arena`s `dhat` fails nowhere and produces a wrong answer everywhere.  So the
// table is computed FROM arena.slotView(), by this kernel, and the only place
// a binding decision is written down is the eight lines below.
//
// LINK 3, AND WHY THERE IS NO TRANSPOSE HERE.  Every field is a pointer
// REBASE, not a copy and not a re-layout, because the arena deliberately
// adopted the host`s addressing for exactly these arrays:
//
//   CmfdOuterView::flux [l*ng + ig]   <- DeviceSlotView::phif, documented as
//                                        "AoS, matching Geometry::Phif"
//                                        (GpuPhysicsTypes.h:250)
//   jnet/dtil/dhat      [ls*ng + ig]  <- the same, and the note at
//                                        GpuPhysicsTypes.h:258-270 says why:
//                                        the Class B0 bodies are scored against
//                                        the CPU loops, so the canonical device
//                                        buffers are the SAME BYTES as the host
//                                        arrays -- no transpose kernel, no
//                                        second layout to keep in step.
//   xsdf/xsnf [ig*nxyz + l]           <- two of the NXS packed scalar slots of
//                                        DeviceSlotView::xs, addressed with the
//                                        shared macroXsIndex.
//
// The group-major/node-major mismatch that does exist is against
// BatchCore::phi [ig*nxyz + l], which is the SWEEP`s buffer, not the arena`s.
// It is therefore a link-2 boundary question and adding a transpose here would
// be transposing something that already matches.
//
// LINK 4: `psi` IS `cmfd_psi`.  DeviceSlotView carries two [nxyz] arrays --
// `psi` (SlotRegion::Psi) and `cmfd_psi` (SlotRegion::CmfdPsi, documented as
// "the CMFD fission source, distinct from psi").  CmfdOuterView::psi is the
// CMFD fission source (CmfdOuterKernel.h:211), so it binds to the second.
// Binding the first is silent: both are [nxyz] doubles, both are writable, and
// updpsi would happily fill the array the nodal path reads.  This kernel is
// the ONLY place that choice is expressed, and the contract test pins it.
__global__ void k_cmfd_build_slot_table(DeviceArenaView arena, cmfd::CmfdOuterView* views,
                                        int slot_count, int ng, int nxyz) {
    const int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slot_count) return;

    const DeviceSlotView& v = arena.slotView(slot);

    cmfd::CmfdOuterView o;
    o.xsdf = v.xs + macroXsIndex(kXtXsdf, 0, 0, ng, nxyz);
    o.xsnf = v.xs + macroXsIndex(kXtXsnf, 0, 0, ng, nxyz);
    o.flux = v.phif;
    o.jnet = v.jnet;
    o.dtil = v.dtil;
    o.dhat = v.dhat;
    o.psi  = v.cmfd_psi;
    views[slot] = o;
}

/// Seed one slot`s control packet so the segment has somewhere to resume from.
///
/// `material_generation` is the host`s XSSet::hoststateGeneration() (link 5).
/// Stamping it here rather than leaving the reset`s zero is what makes
/// nodalConstantSlotIsCurrent truthful: with both counters at zero the gate
/// reads "current" and the constants phase would return without computing
/// anything, for the whole run.
__global__ void k_outer_seed_slot(DeviceArenaView arena, int slot,
                                  unsigned long long material_generation) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    DeviceSlotState& st = arena.states[slot];

    // RESET THE WHOLE CONTROL PACKET FIRST.  The arena's ONE allocation is a
    // cudaMallocFromPoolAsync block: the control structs come back with
    // whatever was in those pages, and GpuPhysicsArena::clearSlotAsync
    // deliberately cannot reach them (they live below slot_base, outside every
    // slot's stride).  cmfdLoadOuterState reads flux_stall, stall_events,
    // clean_iters and xe_interim_count out of this struct, and
    // cmfdOuterConvergence BRANCHES on them -- so garbage here sends the
    // decision down the xe-interim or boron-settling path, which publishes
    // `prev_inner = eigv + 1.0` instead of `eigv`.  SolveLoop then adopts that
    // as its own prev_inner and its convergence test is wrong.
    //
    // The symptom was run-to-run variation with the same inputs: i-SMR CY01
    // statepoint 1 came out at 1.028112, 1.028550 and 1.044908 across five
    // runs of one binary, with one run crashing outright.  Uninitialised
    // memory reads as noise, and noise in a branch condition reads as
    // non-determinism.
    deviceSlotStateReset(st);
    deviceSlotPhaseReset(*arena.slotView(slot).phase, 1u);

    st.material_generation = material_generation;
    // Deliberately NOT equal to it: the constants have never been built on the
    // device, so the gate must read "stale" the first time it is asked.
    st.nodal_constant_generation = material_generation - 1ull;
}

/// Link 2: patch ONE slot of the table onto the sweep arena's buffers.
///
/// A PATCH AND NOT A REBUILD, deliberately.  jnet has no twin in the sweep
/// arena -- jnet is not a CMFD input -- so it must keep the physics-arena
/// pointer k_cmfd_build_slot_table gave it.  Rebuilding the whole view here
/// would mean this kernel owning both bindings, and the two would be free to
/// disagree about jnet.  It writes the five fields the sweep owns and nothing
/// else.
///
/// xsdf is NOT among them and that is not an omission: the only body that reads
/// it is upddtil, which the segment does not run (an eligible deck cannot cusp,
/// so nothing re-runs upddtil inside an outer).  Pointing it at a buffer the
/// sweep does not have would be inventing an address for a reader that does not
/// exist.
__global__ void k_cmfd_bind_resident(cmfd::CmfdOuterView* views, int slot, double* flux,
                                     double* psi, double* dtil, double* dhat,
                                     const double* xsnf) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    cmfd::CmfdOuterView v = views[slot];
    v.flux = flux;
    v.psi  = psi;
    v.dtil = dtil;
    v.dhat = dhat;
    v.xsnf = xsnf;
    views[slot] = v;
}

inline cudaError_t enqueueBuildCmfdSlotTable(const DeviceArenaView& arena,
                                             cmfd::CmfdOuterView* views, int slot_count,
                                             int ng, int nxyz, cudaStream_t stream) {
    if (slot_count <= 0) return cudaSuccess;
    const int block = 64;
    const int grid  = (slot_count + block - 1) / block;
    k_cmfd_build_slot_table<<<grid, block, 0, stream>>>(arena, views, slot_count, ng, nxyz);
    return cudaGetLastError();
}

/// The plan`s name for the nodal constants phase (Sec 6.1 / Task 9).
///
/// A thin, deliberate alias: the plan and the campaign notes both call this
/// `enqueueNodalConstants`, the implementation is called
/// enqueueNodalUpdateConstant, and a reader who greps the plan`s name found
/// nothing.  Naming it here rather than renaming the original keeps
/// CudaNodalConstantKernel.h`s three-launch contract and its own tests intact.
inline cudaError_t enqueueNodalConstants(const DeviceArenaView& arena,
                                         const DevicePhaseQueue& queue,
                                         const DeviceGeometryView& geom, int nxyz, int ng,
                                         cudaStream_t stream) {
    return enqueueNodalUpdateConstant(arena, queue, geom, nxyz, ng, stream);
}

/// Refresh the two CmfdOuterInputs fields that move between outers.
///
/// ONE THREAD PER SLOT, like the convergence kernel and for the same reason
/// (Sec 6.17): it is two loads and two stores of per-slot scalars.
///
/// It also raises `nonfinite` here rather than in the transition, because this
/// is the one place that holds both values at once and a NaN must be caught
/// BEFORE cmfdOuterConvergence turns it into an ordinary "not converged".
__global__ void k_outer_refresh_inputs(DevicePhaseQueue queue, DeviceOuterProbe* probes,
                                       cmfd::CmfdOuterInputs* inputs,
                                       const std::uint32_t* halt) {
    const int logical = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    const int slot = queue.slots[logical];
    if (halt != nullptr && halt[slot] != 0u) return;

    DeviceOuterProbe& probe = probes[slot];
    const bool finite = isfinite(probe.eigv) && isfinite(probe.residual);
    if (!finite) probe.nonfinite = 1u;

    inputs[slot].eigv     = probe.eigv;
    inputs[slot].residual = probe.residual;
}

/// Rev.7.1 W3 item 3: publish 1/eigv where the nodal drive reads it.
///
/// ONE THREAD, ONE DIVIDE, ONE STORE, and it removes two hops of a four-hop
/// journey.  The eigenvalue is produced on the device by the sweep's verdict
/// kernel and lands in DeviceOuterProbe; the nodal drive consumes its reciprocal
/// from a device double (`NodalView::reigv_dev`).  Between those two device
/// facts sat the host: finishDrive copied eigv up, Driver.h's nodal hook divided
/// in double precision, and a captured H2D carried the quotient back down from a
/// pinned slot the host had to rewrite before every launch.  This kernel does
/// the divide where both endpoints already are.
///
/// IT IS BIT-EXACT AND NOT APPROXIMATELY SO.  IEEE-754 makes division a
/// correctly-rounded operation, so `1.0 / e` has exactly one right answer for a
/// given `e` and round-to-nearest -- the host's `1.0 / eigv` and this kernel's
/// must agree bit for bit or one of them is not IEEE.  __ddiv_rn is spelled out
/// rather than `1.0 / e` so that a build which ever acquires -use_fast_math
/// cannot quietly substitute the approximate reciprocal.
///
/// IT IS DELIBERATELY NOT HALT-GATED, which is the opposite of every other
/// kernel in the body.  The halt exists to stop a step from ADVANCING state past
/// a decided exit; this step advances nothing -- it is a pure function of the
/// probe, recomputable at any time, and its consumer (the nodal drive) is a HOST
/// call the halt cannot stop anyway.  Gating it would leave the slot holding the
/// PREVIOUS outer's reciprocal on exactly the outers where the sweep halted
/// itself mid-flight, and the host drive that then finishes those outers would
/// solve the nodal problem at the wrong eigenvalue.  The runner re-issues it
/// after republishAfterHostSweep for the same reason it re-issues updjnet, and
/// the two together are what keep the exceptional path exact.
__global__ void k_outer_publish_reigv(const DeviceOuterProbe* probes, double* reigv,
                                      int slot) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    reigv[0] = __ddiv_rn(1.0, probes[slot].eigv);
}

/// Rank the decision and the signals, publish the phase, latch the halt.
///
/// ONE THREAD PER SLOT.  It is the only kernel in the segment that writes
/// DeviceSlotPhase, which is deliberate: Sec 5.2 gives the transition to exactly
/// one owner, and CudaCmfdOuterKernels.h's convergence kernel publishes a
/// DECISION precisely so that this kernel can be that owner.
__global__ void k_outer_transition(DeviceArenaView arena, DevicePhaseQueue queue,
                                   const CmfdOuterDecision* decisions,
                                   const DeviceOuterProbe* probes,
                                   DeviceOuterSegmentState* segments, std::uint32_t* halt,
                                   unsigned long long* halted_launches) {
    const int logical = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    const int slot = queue.slots[logical];

    if (halt != nullptr && halt[slot] != 0u) {
        // The body kernels of this outer all returned without doing anything;
        // count the wasted dispatch so the Task 10 conditional graph has a
        // number to beat rather than an assertion that it will help.
        if (halted_launches != nullptr) atomicAdd(halted_launches, 1ull);
        return;
    }

    DeviceOuterSegmentState& seg = segments[slot];
    const OuterTransition    t =
        deviceOuterTransition(decisions[slot], probes[slot], seg.outer_in_segment, seg.budget);

    ++seg.outer_in_segment;
    outerApplyTransition(*arena.slotView(slot).phase, t);

    if (t.exit_segment) {
        seg.next_phase = t.next_phase;
        seg.escape     = t.escape;
        seg.exit       = 1u;
        if (halt != nullptr) halt[slot] = 1u;
    }
}

// ---------------------------------------------------------------------------
// Host enqueue
// ---------------------------------------------------------------------------

inline cudaError_t enqueueOuterRefreshInputs(const DevicePhaseQueue& queue,
                                             DeviceOuterProbe* probes,
                                             cmfd::CmfdOuterInputs* inputs,
                                             const std::uint32_t* halt, cudaStream_t stream) {
    if (queue.count <= 0) return cudaSuccess;
    const int grid = (queue.bucket + kCmfdOuterBlock - 1) / kCmfdOuterBlock;
    k_outer_refresh_inputs<<<grid, kCmfdOuterBlock, 0, stream>>>(queue, probes, inputs, halt);
    return cudaGetLastError();
}

inline cudaError_t enqueueOuterPublishReigv(const DeviceOuterProbe* probes, double* reigv,
                                            int slot, cudaStream_t stream) {
    // A null slot is the ordinary answer before the nodal backend has allocated
    // its device block, and on the hybrid arm; it means "the host still owns the
    // reciprocal", which is a legal state and not a failure.
    if (probes == nullptr || reigv == nullptr || slot < 0) return cudaSuccess;
    k_outer_publish_reigv<<<1, 1, 0, stream>>>(probes, reigv, slot);
    return cudaGetLastError();
}

inline cudaError_t enqueueOuterTransition(const DeviceArenaView& arena,
                                          const DevicePhaseQueue& queue,
                                          const CmfdOuterDecision* decisions,
                                          const DeviceOuterProbe* probes,
                                          DeviceOuterSegmentState* segments,
                                          std::uint32_t* halt,
                                          unsigned long long* halted_launches,
                                          cudaStream_t stream) {
    if (queue.count <= 0) return cudaSuccess;
    const int grid = (queue.bucket + kCmfdOuterBlock - 1) / kCmfdOuterBlock;
    k_outer_transition<<<grid, kCmfdOuterBlock, 0, stream>>>(arena, queue, decisions, probes,
                                                             segments, halt, halted_launches);
    return cudaGetLastError();
}

#endif // __CUDACC__

} // namespace rasbery::gpu
