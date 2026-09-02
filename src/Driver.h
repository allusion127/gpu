#pragma once
#include "BICGCMFD.h"
#include "BatchLightResult.h"
#include "CaseFidelity.h"
#include "CaseKey.h"
#include "CudaOuterGraph.h"
#include "EvaluatorContext.h"
#include "GpuFullContract.h"
#include "IO.h"
#include "Nodal.h"
#include "OuterTrace.h"
#include "PPR.h"
#include "Scheduler.h"
#include "WarmState.h"
#include "XSTiming.h"
#include "XeFormAudit.h"
#include "XeFormMask.h"
#include "XeGpuReceipt.h"
#include "XeKernel.h"
#include "XsLibrary.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <filesystem>
#include "CompatFormat.h"
#include <stdexcept>
#include <string>
#include <vector>

// A FUNCTION BODY THAT MAY NOT JOIN ITS CALLER, AND WHY THAT IS A CONTRACT.
//
// Everything in this header is an implicitly-inline member of one class, and
// the whole solver reaches g++ as ONE translation unit (src/main.cpp is the
// only .cpp that includes Driver.h; there is no LTO).  SolveLoop's Xe step is
// a chain of static member functions with EXACTLY ONE CALL SITE EACH --
// SolveLoop -> TryAndersonXeStep -> TryAndersonXeStepGpu -- so
// -finline-functions-called-once folds every one of them into SolveLoop.  A new
// sibling with one call site therefore does not sit "next to" the production
// arm: its body is spliced INTO the hottest function in the tree, and the
// inlining, scheduling and -ffp-contract=fast decisions of everything already
// there are re-made around it.  That is not a hypothesis; it is what moved the
// flag-off trajectory in `71092e2` (docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md
// Sec 7).
//
// So a default-off arm that lives in this header is written the way
// xe::auditAndersonFit is written -- a cached bool and an OPAQUE CALL -- and
// when it cannot live in its own translation unit, this attribute is what makes
// the call opaque.  It is a correctness contract for the OFF arm, not a
// performance hint for the ON one.
#if !defined(RASBERY_NEVER_INLINE)
#  if defined(__GNUC__)
#    define RASBERY_NEVER_INLINE __attribute__((noinline, cold))
#  elif defined(_MSC_VER)
#    define RASBERY_NEVER_INLINE __declspec(noinline)
#  else
#    define RASBERY_NEVER_INLINE
#  endif
#endif

namespace rasbery {

// Per-statepoint decomposition telemetry (plan Rev.4 Sec 8), behind
// RASBERY_STATEPOINT_TELEMETRY=1.
//
// WHY.  Every phase of this campaign is scored against the Sec 1 Amdahl model
//
//   T_total = T_fixed + N_outer*T_outer + N_xe*T_xe + N_search*T_search
//             + T_depletion + T_IO + T_startup
//
// and none of its coefficients are recoverable from the receipts we already
// print: [RASBERY][OUTER][PHASE] is one process-wide sum over the whole run,
// and the `NO.= .. outer=` line gives N_outer per statepoint but not what
// CAUSED those outers.  This splits the ~600 outers of a statepoint into the
// driver causes that produced them, so Phase 3 (warm-start audit, adaptive
// inner tolerance), Phase 4 (Anderson) and Phase 5 (persistent kernel) are
// ordered by measured headroom instead of by guess.
//
// ATTRIBUTION.  Every outer belongs to exactly ONE cause, decided by SEGMENT
// BOUNDARY rather than by inspecting the outer itself: an outer is charged to
// the most recent state perturbation before it, and a perturbation opens a new
// segment at the point in SolveLoop where it commits.  The rules, in the order
// SolveLoop applies them:
//
//   INITIAL   from SolveLoop entry until the first perturbation of that call.
//             This is the cost of re-converging the flux on the state the
//             previous statepoint (or the predictor/corrector step) handed
//             over -- exactly the quantity Phase 3's warm-start audit needs.
//   XE        after every XSSet::UpdateEquilibriumXenon call, settled-flux or
//             interim.  An Xe step is a perturbation whether or not its
//             returned change clears XE_EQUILIBRIUM_TOLERANCE, so the outers
//             after it are XE until something else perturbs.
//   TH        after XSSet::UpdateTH.
//   SEARCH    after a trial point is committed AND applied (SetBoron/SetRod).
//             A search step that exits without committing (SEARCH_EXHAUSTED,
//             NO_PROPOSAL) opens no segment, because no state moved.
//   SETTLE    after the SEARCH_SETTLE_ITERS gate defers an otherwise usable
//             k_eff sample.  These are Sec 8's "settling gate extra outers":
//             outers that exist only because the gate refused the sample.
//   FALLBACK  the bounded ReconvergeFlux() run on the best observed trial point
//             after a non-converged search.  Measured as the delta of
//             total_outer across that call -- no cause can change inside it.
//
// TIE-BREAK.  T/H and the search can both perturb inside one outer (SolveLoop
// runs T/H first, then the search).  The following segment is charged to
// SEARCH: a trial point moves boron/rod macro-XS core-wide, whereas the T/H
// step is a bounded temperature correction.  How often the ambiguity arises is
// published as `th_search_coincident`, so an analysis can bound the error the
// tie-break can introduce instead of having to trust it.
//
// INVARIANT.  sum(outers_by_cause) equals the `outer=` count of the NO.= line,
// which is republished here as `outers` for exactly that cross-check.
//
// COST (Sec 8.1 contract).  The integer tallies are plain non-atomic members of
// the per-Driver SolverContext -- no atomics, no thread-local lookup, no
// allocation -- and each is one increment in a loop body whose iteration
// already costs milliseconds, so they are carried unconditionally, the same way
// `total_outer` and BICGCMFD::iter already are.  Everything with real cost is
// gated on enabled(): the per-phase clock reads (outer_timing::Scope), the
// backend counter snapshots, and every string.  Formatting happens once per
// statepoint (35-51 lines/run) and once at end of run, never inside a loop.
namespace sptelem {

enum Phase : int {
    PH_UPDPSI = 0,
    PH_SETLS,
    PH_DRIVE,
    PH_UPDJNET,
    PH_NODAL,
    PH_CUSPING,
    PH_UPDDHAT,
    // --- SolveLoop phases OUTSIDE the outer body (WP9 stage A) --------------
    // GA evaluator plan Sec 3.2 measured the gap these close.  The regression
    // slope of the statepoint cost model is d = 4.805 ms/outer; the seven outer
    // body phases above sum to 4.428 ms/outer, and the document names the
    // missing 0.377 ms/outer as "the things outside the Scopes (Xe update,
    // convergence decision, search arithmetic)" without an instrument behind
    // that sentence.  These four are that instrument.  They live INSIDE
    // SolveLoop, so they belong to `d` and not to the statepoint floor `c` --
    // which is why they get their own `loop_wall` object and are neither folded
    // into `phase_wall` (that object is calibrated on by
    // tools/scheduler_trace_replay.py) nor into `floor_wall` (that object is
    // the statepoint boundary, and these are not on it).
    PH_TH_UPDATE,       ///< XSSet::UpdateTH, whole call (incl. its UpdateFlatXS)
    PH_XE_STEP,         ///< XSSet::UpdateEquilibriumXenon, the production Picard step
    PH_SEARCH_PROPOSE,  ///< ProposeNextSearchPoint + CommitSearchPoint: the secant arithmetic
    PH_SEARCH_APPLY,    ///< SetBoron/SetRod for a committed trial (incl. its UpdateFlatXS)
    // --- statepoint FLOOR phases (outside the outer body) -------------------
    // The regression of GPU_RASBERY_GA_EVALUATOR_PLAN Sec 2.2 splits a
    // statepoint into `c + d x outer`.  The seven phases above are `d` -- the
    // outer body, and the OUTER][PHASE] receipt already sums to it.  `c` was
    // only ever a residual: 0.474 s of host work per statepoint that the plan
    // attributes to PPR / depletion / FlatXS / T-H / result packing from
    // reading the code, with no instrument behind the split.  These buckets are
    // that instrument.  They are deliberately NOT emitted inside `phase_wall`
    // -- tools/scheduler_trace_replay.py derives its per-statepoint boundary
    // work as `wall - sum(phase_wall)`, and folding the floor into that object
    // would silently move a number that model is calibrated on.  They get their
    // own `floor_wall` object instead, and the residual stays readable as
    // `wall - sum(phase_wall) - sum(floor_wall)`.
    // NO EXPLICIT INITIALISER.  It used to read `PH_PPR_RESET = PH_UPDDHAT + 1`,
    // which is 7 -- the value PH_TH_UPDATE already has, because the four loop
    // phases were added between PH_UPDDHAT and here after that line was written.
    // The floor phases therefore ALIASED the loop phases, `phaseName`'s switch
    // had four duplicate case labels (ill-formed: every translation unit that
    // includes Driver.h fails to compile), and PH_COUNT was four short of the
    // number of phases it counts.  Following on from PH_SEARCH_APPLY is what
    // the block comment above has always said this is.
    PH_PPR_RESET,                  ///< PPR::reset (buckling, corner flux, fitting, leakage, source)
    PH_PPR_DRIVE,                  ///< PPR::drive(100) corner-balance iteration
    PH_PPR_RECON,                  ///< PPR::reconstructPinPower (+ normalisation, Fq/FdH)
    PH_DEPL_PRED,                  ///< XSSet::PredictorStep (CRAM Bateman, BOS)
    PH_DEPL_CORR,                  ///< XSSet::CorrectorStep (CRAM Bateman, EOS)
    PH_RESULT_ADD,                 ///< IO::AddResult (result packing), the head of io_wall
    PH_RESULT_WRITE,               ///< BatchLightResult::Write | IO::WriteStepToResult
    PH_COUNT,
    PH_LOOP_FIRST  = PH_TH_UPDATE,
    PH_FLOOR_FIRST = PH_PPR_RESET,
};

inline const char* phaseName(int phase) {
    switch (phase) {
        case PH_UPDPSI:  return "updpsi";
        case PH_SETLS:   return "setls";
        case PH_DRIVE:   return "drive";
        case PH_UPDJNET: return "updjnet";
        case PH_NODAL:   return "nodal";
        case PH_CUSPING: return "cusping";
        case PH_UPDDHAT: return "upddhat";
        case PH_TH_UPDATE:      return "th_update";
        case PH_XE_STEP:        return "xe_step";
        case PH_SEARCH_PROPOSE: return "search_propose";
        case PH_SEARCH_APPLY:   return "search_apply";
        case PH_PPR_RESET:  return "ppr_reset";
        case PH_PPR_DRIVE:  return "ppr_drive";
        case PH_PPR_RECON:  return "ppr_recon";
        case PH_DEPL_PRED:  return "depl_predictor";
        case PH_DEPL_CORR:  return "depl_corrector";
        case PH_RESULT_ADD: return "result_add";
        case PH_RESULT_WRITE: return "result_write";
        default:         break;
    }
    return "unknown";
}

enum Cause : int {
    CAUSE_INITIAL = 0,
    CAUSE_XE,
    CAUSE_TH,
    CAUSE_SEARCH,
    CAUSE_SETTLE,
    CAUSE_FALLBACK,
    CAUSE_COUNT
};

inline bool enabled() {
    static const bool on = [] {
        const char* value = std::getenv("RASBERY_STATEPOINT_TELEMETRY");
        if (value == nullptr) return false;
        const std::string s(value);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" ||
                 s == "false" || s == "FALSE");
    }();
    return on;
}

/// Per-thread phase wall in seconds, written by outer_timing::Scope.  One
/// Driver owns one host thread (batch mode gives every deck its own), so this
/// is per-Driver by construction and needs no atomics; Drive() snapshots it at
/// the statepoint boundary and publishes the difference.
inline double* phaseWall() {
    static thread_local double wall[PH_COUNT] = {};
    return wall;
}

/// One statepoint's tallies plus the baselines its deltas are measured from.
/// The same type doubles as the run-total accumulator, through accumulate().
struct Counters {
    long long outers_by_cause[CAUSE_COUNT]{};
    long long outers_driver        = 0;  ///< Driver's own total_outer (cross-check)
    long long solve_loops          = 0;  ///< SolveLoop calls (BOS/predictor/final/...)
    long long xe_updates           = 0;  ///< settled-flux equilibrium-Xe steps
    long long xe_interim_updates   = 0;  ///< RASBERY_XE_INTERIM_L2 loose-flux steps
    /// Xe re-convergence cascade STARTS: one per SolveLoop entry plus one per
    /// committed perturbation (search trial, T/H update) that moves the macro-XS
    /// and therefore the Xe fixed point.  xe_updates/xe_cascades is the steps-per-
    /// cascade the outer budget is actually being spent on; a statepoint whose
    /// ratio collapses toward 1 is one whose late cascades are being starved.
    long long xe_cascades          = 0;
    /// Cascades that spent their step budget with the Xe change still above
    /// XE_EQUILIBRIUM_TOLERANCE -- i.e. published a truncated Xe inventory.
    long long xe_budget_exhausted  = 0;
    /// RASBERY_XE_MODE=once only.  Steps beyond the cascade's first, i.e. the damped
    /// trust-region steps bought by a step that overshot XE_ONCE_TRUST; zero on a run
    /// whose cascades all land inside the trust region on their first step.
    long long xe_once_extra_steps  = 0;
    /// RASBERY_XE_MODE=once only.  Steps whose RAW relative change exceeded
    /// XE_ONCE_TRUST.  Counted at the cap too, so a cascade that ran out of steps
    /// while still outside the region is visible as trips == XE_ONCE_MAX_STEPS.
    long long xe_once_trust_trips  = 0;
    /// RASBERY_XE_ANDERSON only (plan Rev.4 Sec 10.5).  A cascade step where the
    /// history held at least one residual difference and the residual was still
    /// above tolerance, so an extrapolated candidate was actually built:
    /// proposed == accepted + rejected, exactly.
    long long xe_aa_proposed       = 0;
    /// Candidates that passed every safeguard and were committed.
    long long xe_aa_accepted       = 0;
    /// Candidates that failed one, and fell back to the production Picard step.
    /// ONE counter with the reason folded in; RASBERY_XE_ANDERSON_DEBUG prints
    /// the per-rejection reason so the distribution is recoverable when wanted.
    long long xe_aa_rejected       = 0;
    /// History discards: the cascade re-arm after a committed T/H or search
    /// perturbation, and damper activation.  A reset with nothing stored is not
    /// charged, so this counts real discards rather than outers.
    long long xe_aa_history_resets = 0;
    /// A2 staged tolerance (RASBERY_STAGED_FLUX_TOL / _XE_TOL) only.  Times the
    /// loose stage's "everything converged" verdict did NOT survive being re-asked
    /// at the production tolerance, so the loop dropped back to loose and committed
    /// another trial.  Zero with the feature off.  A count comparable to
    /// search_trials is the signature of a multiplier loose enough that the loose
    /// stage is locating a different root than the one being published.
    long long staged_relapses      = 0;
    long long search_trials        = 0;  ///< committed AND applied trial points
    // WP9-D.  The search's convergence history, copied from the Schedule at the
    // statepoint boundary (Scheduler.h owns the classification, because `method`
    // is decided there).  These are how the 238 runner prices what the boron
    // search costs per statepoint before any trial-reduction option is built:
    // 137 trials over a run is one number, but 137 trials that are mostly
    // BOOTSTRAP PROBES and 137 that are mostly bisections inside a collapsing
    // bracket are two different problems with two different fixes.
    long long search_proposals     = 0;  ///< proposal attempts, accepted or not
    long long search_refused       = 0;  ///< attempts that produced no next point
    long long search_secant        = 0;
    long long search_carry         = 0;  ///< first steps on a slope CARRIED from before
    long long search_probe         = 0;  ///< bootstrap steps: no slope was available
    long long search_bisect        = 0;
    /// WP9-D stage D.  Carry steps taken on the burnup-EXTRAPOLATED slope
    /// (RASBERY_SEARCH_CARRY_SLOPE).  `search_carry` counts the steps the lever
    /// could have moved and this counts the ones it did; the difference is how
    /// often the correction's guards refused.  Zero with the knob unset.
    long long search_extrap        = 0;
    long long search_iterations    = 0;  ///< Schedule::search_iteration at the close
    long long th_updates           = 0;
    long long flux_limit_retries   = 0;  ///< flux limit-cycle events ([WARN][flux])
    long long th_search_coincident = 0;  ///< outers where T/H and search both fired
    long long cmfd_sweeps          = 0;
    long long bicg_iters           = 0;
    double    wall                 = 0.0;  ///< solve wall of the statepoint
    double    io_wall              = 0.0;
    double    phase[PH_COUNT]{};

    /// WP9-A `nested_wall`: XS phases that run INSIDE the buckets above and are
    /// therefore published beside them rather than summed with them.
    double    xs_wall[xsphase::LB_COUNT]{};
    long long xs_calls[xsphase::LB_COUNT]{};

    /// WP9-A `floor_transfer`: the host copies charged to the STATEPOINT
    /// BOUNDARY, i.e. everything after the statepoint's last SolveLoop
    /// returned -- PPR's inputs, the result packing, the restart save.  The
    /// existing `*_delta` fields below span the WHOLE statepoint, so
    /// `delta - floor` is the in-iteration half and no field has to be
    /// re-interpreted to get either.
    std::uint64_t floor_h2d_bytes = 0, floor_h2d_calls = 0;
    std::uint64_t floor_d2h_bytes = 0, floor_d2h_calls = 0;

    double        phase0[PH_COUNT]{};
    double        xs_wall0[xsphase::LB_COUNT]{};
    long long     xs_calls0[xsphase::LB_COUNT]{};
    std::uint64_t graph0 = 0, h2d0 = 0, d2h0 = 0, d2h_calls0 = 0;
    /// The floor baseline, armed by armFloor() after the last SolveLoop.  A
    /// statepoint that never armed it reports a zero floor rather than the
    /// difference against an unrelated baseline.
    bool          floor_armed = false;
    std::uint64_t fh2d0 = 0, fh2d_calls0 = 0, fd2h0 = 0, fd2h_calls0 = 0;
    std::uint64_t graph_delta = 0, h2d_delta = 0, d2h_delta = 0, d2h_calls_delta = 0;

    [[nodiscard]] long long outers() const {
        long long sum = 0;
        for (long long v : outers_by_cause) sum += v;
        return sum;
    }

    /// Clear the tallies and arm the baselines for a new statepoint.
    void begin(std::uint64_t graph, std::uint64_t h2d, std::uint64_t d2h,
               std::uint64_t d2h_calls) {
        *this                 = Counters{};
        const double* current = phaseWall();
        for (int p = 0; p < PH_COUNT; ++p) phase0[p] = current[p];
        const xsphase::LocalWall& xs = xsphase::localWall();
        for (int b = 0; b < xsphase::LB_COUNT; ++b) {
            xs_wall0[b]  = xs.seconds[b];
            xs_calls0[b] = xs.calls[b];
        }
        graph0     = graph;
        h2d0       = h2d;
        d2h0       = d2h;
        d2h_calls0 = d2h_calls;
    }

    /// Arm the STATEPOINT-BOUNDARY transfer baseline.  Called once, after the
    /// statepoint's last SolveLoop has returned and before PPR: everything
    /// after this point is floor work, so everything it copies is a boundary
    /// copy.  Idempotent by the `floor_armed` latch -- a second call would move
    /// the baseline forward and silently shrink the floor.
    void armFloor(std::uint64_t h2d, std::uint64_t h2d_calls, std::uint64_t d2h,
                  std::uint64_t d2h_calls) {
        if (floor_armed) return;
        floor_armed = true;
        fh2d0       = h2d;
        fh2d_calls0 = h2d_calls;
        fd2h0       = d2h;
        fd2h_calls0 = d2h_calls;
    }

    /// Close the statepoint: resolve every delta against the armed baselines.
    void end(std::uint64_t graph, std::uint64_t h2d, std::uint64_t h2d_calls,
             std::uint64_t d2h, std::uint64_t d2h_calls) {
        const double* current = phaseWall();
        for (int p = 0; p < PH_COUNT; ++p) phase[p] = current[p] - phase0[p];
        const xsphase::LocalWall& xs = xsphase::localWall();
        for (int b = 0; b < xsphase::LB_COUNT; ++b) {
            xs_wall[b]  = xs.seconds[b] - xs_wall0[b];
            xs_calls[b] = xs.calls[b] - xs_calls0[b];
        }
        graph_delta     = graph - graph0;
        h2d_delta       = h2d - h2d0;
        d2h_delta       = d2h - d2h0;
        d2h_calls_delta = d2h_calls - d2h_calls0;
        if (floor_armed) {
            floor_h2d_bytes = h2d - fh2d0;
            floor_h2d_calls = h2d_calls - fh2d_calls0;
            floor_d2h_bytes = d2h - fd2h0;
            floor_d2h_calls = d2h_calls - fd2h_calls0;
        }
    }

    /// Fold one closed statepoint into a run-total accumulator.
    void accumulate(const Counters& step) {
        for (int c = 0; c < CAUSE_COUNT; ++c)
            outers_by_cause[c] += step.outers_by_cause[c];
        for (int p = 0; p < PH_COUNT; ++p) phase[p] += step.phase[p];
        for (int b = 0; b < xsphase::LB_COUNT; ++b) {
            xs_wall[b]  += step.xs_wall[b];
            xs_calls[b] += step.xs_calls[b];
        }
        floor_h2d_bytes      += step.floor_h2d_bytes;
        floor_h2d_calls      += step.floor_h2d_calls;
        floor_d2h_bytes      += step.floor_d2h_bytes;
        floor_d2h_calls      += step.floor_d2h_calls;
        outers_driver        += step.outers_driver;
        solve_loops          += step.solve_loops;
        xe_updates           += step.xe_updates;
        xe_interim_updates   += step.xe_interim_updates;
        xe_cascades          += step.xe_cascades;
        xe_budget_exhausted  += step.xe_budget_exhausted;
        xe_once_extra_steps  += step.xe_once_extra_steps;
        xe_once_trust_trips  += step.xe_once_trust_trips;
        xe_aa_proposed       += step.xe_aa_proposed;
        xe_aa_accepted       += step.xe_aa_accepted;
        xe_aa_rejected       += step.xe_aa_rejected;
        xe_aa_history_resets += step.xe_aa_history_resets;
        staged_relapses      += step.staged_relapses;
        search_trials        += step.search_trials;
        search_proposals     += step.search_proposals;
        search_refused       += step.search_refused;
        search_secant        += step.search_secant;
        search_carry         += step.search_carry;
        search_probe         += step.search_probe;
        search_bisect        += step.search_bisect;
        search_extrap        += step.search_extrap;
        search_iterations    += step.search_iterations;
        th_updates           += step.th_updates;
        flux_limit_retries   += step.flux_limit_retries;
        th_search_coincident += step.th_search_coincident;
        cmfd_sweeps          += step.cmfd_sweeps;
        bicg_iters           += step.bicg_iters;
        wall                 += step.wall;
        io_wall              += step.io_wall;
        graph_delta          += step.graph_delta;
        h2d_delta            += step.h2d_delta;
        d2h_delta            += step.d2h_delta;
        d2h_calls_delta      += step.d2h_calls_delta;
    }
};

} // namespace sptelem

// Wall-clock attribution of the outer iteration's serial phases, process-wide
// across batch instances (RASBERY_OUTER_TIMING=1; zero cost when unset).
// Same design rules as xsphase (XSTiming.h): atomics, one JSON line at exit.
//
// The same Scope feeds sptelem's per-thread, per-statepoint phase wall, so the
// two receipts can never disagree about where a phase boundary is.  When both
// variables are unset the scope stores nothing and never reads the clock.
namespace outer_timing {
struct Buckets {
    /// Indexed by sptelem::Phase: updpsi, setls (host linear-system assembly),
    /// drive (BiCG outer incl. arena wait + sync), updjnet, nodal (reset +
    /// drive), cusping (ApplyRodCusping (+ upddtil on change)), upddhat.
    std::atomic<double>    phase[sptelem::PH_COUNT];
    std::atomic<long long> outers{0};
    Buckets() {
        for (auto& bucket : phase) bucket.store(0.0, std::memory_order_relaxed);
    }
};
inline Buckets& buckets() { static Buckets b; return b; }
inline bool enabled() {
    static const bool on = std::getenv("RASBERY_OUTER_TIMING") != nullptr;
    return on;
}
class Scope {
    int  _phase  = -1;
    bool _global = false;
    bool _local  = false;
    std::chrono::steady_clock::time_point _t0;
public:
    explicit Scope(sptelem::Phase phase) {
        _global = enabled();
        _local  = sptelem::enabled();
        if (!_global && !_local) return;
        _phase = static_cast<int>(phase);
        _t0    = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (_phase < 0) return;
        const double dt = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - _t0).count();
        if (_global) {
            std::atomic<double>& acc = buckets().phase[_phase];
            double cur = acc.load(std::memory_order_relaxed);
            while (!acc.compare_exchange_weak(cur, cur + dt)) {}
        }
        if (_local) sptelem::phaseWall()[_phase] += dt;
    }
};
inline void report(std::ostream& out) {
    if (!enabled()) return;
    Buckets& b = buckets();
    out << "[RASBERY][OUTER][PHASE] {\"outers\":" << b.outers.load();
    for (int p = 0; p < sptelem::PH_COUNT; ++p)
        out << ",\"" << sptelem::phaseName(p) << "\":" << b.phase[p].load();
    out << "}" << std::endl;
}
} // namespace outer_timing

// ---------------------------------------------------------------------------
// The trajectory receipt -- ALWAYS ON, and that is the whole point.
// ---------------------------------------------------------------------------
//
// WHY THIS EXISTS.  Instrumentation must never move the iteration.  On 238 an
// A2 candidate was reported at 3,114 outers plain and 4,393 with
// RASBERY_STATEPOINT_TELEMETRY=1, and there was no way to tell, from the two
// logs, whether the telemetry had perturbed the solve or whether the two runs
// had simply been given different environments -- because the only per-run
// record of "which arm was this" is scattered across receipts that each feature
// prints only when it is ON, and the per-statepoint line carries a wall time
// that makes a plain `diff` of two logs useless.
//
// So this receipt answers exactly that question, in one line, on every run:
//
//   * `digest` is a fold of the per-statepoint (step, outers, T/H steps, and the
//     BIT PATTERNS of efpd, k_eff and boron) -- i.e. the trajectory, not a
//     summary of it.  Two runs of the same arm agree on it exactly; a run whose
//     iteration moved by one outer, or whose k_eff moved in the last bit, does
//     not.
//   * `env` is the raw, UNPARSED value of every knob that can move a
//     trajectory.  Raw on purpose: a receipt that re-derived `staged_flux_mult`
//     would be a second interpretation of the variable that could drift from
//     SolveLoop's, and the one thing this must not do is disagree with the
//     solver about what the run was asked for.  Unset prints as null.
//   * `telemetry` is reported BESIDE the digest and is deliberately NOT folded
//     into it, so "telemetry is trajectory-neutral" is a mechanical test:
//     two runs whose `env` and `digest` agree while `telemetry` differs.
//
// COST.  Five integer folds per statepoint (35-51 of them) and one formatted
// line per run.  There is no gate because a receipt nobody can be sure was
// enabled is a receipt nobody can quote.
namespace trajectory {

/// The knobs that can move an iteration, reported raw so the receipt cannot
/// disagree with the solver's own reading of them.  RASBERY_STATEPOINT_TELEMETRY
/// is deliberately absent: it is the thing under test, and it rides in its own
/// field so an arm comparison can hold every OTHER knob equal.
///
/// RASBERY_GPU_CRAM is deliberately PRESENT, and it is the counterexample that
/// makes the PPR rule below mean something.  The device depletion arm (Task 16)
/// writes the isotope inventory the NEXT statepoint reconstructs its cross
/// sections from, so the knob moves keff, the critical boron and the axial
/// offset of every statepoint after the first.  It belongs in the list on the
/// list's own terms; tools/test_cram_gpu_contract.py asserts the presence, the
/// way test_ppr_gpu_contract.py asserts the absence.
///
/// RASBERY_GPU_PPR is deliberately absent for a DIFFERENT reason, and the
/// difference is the whole point of the arm.  Pin-power reconstruction is
/// strictly downstream of the iteration: it runs after the statepoint's final
/// SolveLoop, reads Jnet/Phif/Phis, and writes only Geometry's PPR coefficient
/// arrays and the pin map -- nothing SolveLoop, the boron search, the T/H
/// update or the depletion will read again.  So the knob cannot move a
/// trajectory, and listing it here would say it could.  Its own
/// `[RASBERY][PPR_GPU]` line answers "which arm was this" for the one thing it
/// does change.  If a future change ever lets PPR feed back, this list is where
/// that change has to be declared -- tools/test_ppr_gpu_contract.py asserts the
/// absence so the declaration cannot be forgotten.
///
/// SET-VS-UNSET, AND WHAT THE v5 DEFAULT FLIP DID NOT CHANGE.  This list is
/// reported RAW: `armEnvJson()` prints the string the shell exported, or `null`
/// when the variable is unset.  So from 2026-08-30, when RASBERY_GPU_FLATXS_CTA
/// and RASBERY_GPU_XE_TXN became DEFAULT ON, two runs of the SAME arm can print
/// two different `env` objects -- `"1"` if the launcher exported the knob,
/// `null` if it relied on the default.  That is the receipt doing its job: it
/// reports what it was asked for, not what it resolved to, precisely so it can
/// never disagree with the solver's own reading.
///
/// THE `digest` IS UNAFFECTED, and this is a statement about the features, not
/// about the fold.  `digest` folds statepoints, outers, T/H steps and the bit
/// patterns of efpd / k_eff / boron -- `env` is not mixed into it at all.  Both
/// flipped knobs are B0: 238 measured TXN=0 and TXN=1 at the same
/// 0d15abf29d222a02 / 4382 and CTA=0 and CTA=1 at the same 1f36e75dc00ed2b4 /
/// 4377, and 181 measured the same identity on its own digests.  So a v5 run
/// with an EMPTY env prints the SAME digest as the old explicit-ON env, and
/// tools/test_v5_defaults_contract.py holds the four defaults in place so that
/// stays true.
///
/// THE `case_key` IS AFFECTED, and it is the one place this costs something.
/// src/CaseKey.h's env half digests the same raw strings in this same order, so
/// an unset knob (`~`) and an explicit `1` are two different payloads and two
/// different keys for what is now one arm.  That is conservative in the safe
/// direction -- a cache MISS, never a wrong hit -- but it means a GA controller
/// that wants cache continuity across the flip must keep exporting the knobs it
/// exported before.  The v5 manifest therefore lists all four flipped knobs
/// EXPLICITLY in its env, even though unset would resolve identically.
inline constexpr const char* kArmEnv[] = {
    "RASBERY_GPU",
    "RASBERY_GPU_CMFD_SWEEP",
    "RASBERY_GPU_CMFD_RESIDENT_SINGLE",
    "RASBERY_GPU_CMFD_FP32",
    // WP20.  The device-wide single-precision arm and its two extensions
    // (src/GpuFp32Arm.h).  These belong here on the list's own terms and more
    // plainly than anything else on it: they select the ROUNDING of every
    // device kernel in the iteration, which is the most trajectory-moving thing
    // a knob in this binary can do.  Gate class A2 -- validated by Gate A
    // against the FP64 trajectory and Gate B against MASTER, never by the
    // bit-golden gate.  Being here is also what folds them into the WP10.1 case
    // key, so an FP64 answer can never be served to an FP32 request.
    "RASBERY_GPU_FP32",
    "RASBERY_GPU_FP32_STRICT",
    "RASBERY_GPU_FP32_CRAM",
    // WP20.2.  RASBERY_GPU_FP32_PPR is deliberately ABSENT from this list, on
    // the same footing RASBERY_GPU_PPR is: PPR is downstream of the statepoint
    // and cannot move a trajectory, so two runs that differ only in it are the
    // same arm and must compare as one.  It appears in the [RASBERY][FP32]
    // receipt instead, which is where a VRAM item belongs.
    // WP20.2.  The refinement ROUND CAP, and it belongs here for a sharper
    // reason than the other three: it does not merely change the rounding of a
    // kernel, it changes HOW MANY TIMES the inner solve runs and what the outer
    // loop accepts, so two runs that differ only in this knob are two different
    // trajectories with the same digest inputs.  Default 2 under the arm and 1
    // without it, so an unset value and an explicit "0" are two different
    // payloads -- deliberately, exactly as for the other knobs on this list.
    "RASBERY_GPU_FP32_REFINE",
    "RASBERY_GPU_NODAL",
    "RASBERY_GPU_NODAL_FULL",
    "RASBERY_GPU_XSRECON",
    "RASBERY_GPU_FLATXS",
    "RASBERY_GPU_FLATXS_CTA",
    "RASBERY_GPU_FLATXS_CTA_THREADS",
    "RASBERY_GPU_CRAM",
    "RASBERY_GPU_XE",
    "RASBERY_GPU_XE_DOT_PARTITIONS",
    "RASBERY_GPU_XE_TXN",
    "RASBERY_GPU_OUTER",
    "RASBERY_GPU_OUTER_SEGMENT_MAX",
    "RASBERY_GPU_OUTER_BATCH_STREAM_SWEEP",
    "RASBERY_GPU_WIEL_FOLD",
    "RASBERY_CMFD_OUTER_FORMS",
    "RASBERY_XE_FORMS",
    // The host spelling of the Anderson normal equations (XeFormMask.h,
    // XE_HOST_FORMS_DEFAULT).  It belongs here more plainly than any other
    // knob on the list: it selects the rounding of four expressions on the
    // RASBERY_GPU_XE arm's own critical path, so two runs that disagree about
    // it are two different trajectories -- and the 238 sweep that pins its
    // default varies exactly this string.  A case key that did not fold it
    // would serve one sweep point's answer to another's request.
    "RASBERY_XE_HOST_FORMS",
    "RASBERY_XE_MODE",
    "RASBERY_XE_ANDERSON",
    "RASBERY_XE_ANDERSON_MAX_STEP",
    "RASBERY_XE_CASCADE_BUDGET",
    "RASBERY_XE_INTERIM_L2",
    "RASBERY_STAGED_FLUX_TOL",
    "RASBERY_STAGED_XE_TOL",
    "RASBERY_STAGED_LOOSE_SETTLE",
    // WP9-D stage D.  Five search-policy levers (Scheduler.h SearchPolicy).
    // Four of them move the PROPOSALS, which moves the trial sequence, which
    // moves the iteration; the fifth scales the loose-stage sample tolerance.
    // All five therefore belong here on the list's own terms, and being here is
    // also what folds them into the WP10.1 case key -- a cached answer produced
    // under one search policy must never be served to a request made under
    // another.
    "RASBERY_SEARCH_CARRY_SLOPE",
    "RASBERY_SEARCH_WARM_BORON",
    "RASBERY_SEARCH_BORON_BRACKET",
    "RASBERY_SEARCH_MAX_TRIALS",
    "RASBERY_SEARCH_STAGED_MARGIN",
    "RASBERY_GA_FEEDBACK_PASSES",
    "RASBERY_ALLOW_SCREENING",
};

/// FNV-1a, one 64-bit word at a time.  Chosen because it is eight lines, has no
/// table, and this is an identity check between two runs of the same binary --
/// not a cryptographic claim.
inline void mix(unsigned long long& h, unsigned long long word) {
    for (int b = 0; b < 8; ++b) {
        h ^= (word >> (b * 8)) & 0xffull;
        h *= 1099511628211ull;
    }
}

/// The BIT PATTERN of a double.  A digest that folded the printed decimals
/// would agree across a trajectory change too small to print, which is the one
/// case it exists to catch.
inline unsigned long long bits(double v) {
    unsigned long long u = 0;
    static_assert(sizeof(u) == sizeof(v), "a double is not eight bytes here");
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

struct Digest {
    unsigned long long h           = 14695981039346656037ull; ///< FNV-1a offset basis
    int                statepoints = 0;
    long long          outers      = 0;
    long long          th          = 0;

    /// One statepoint, folded at the moment the solver publishes it.
    void step(int step_number, int outer, int th_steps, double efpd, double eigv,
              double ppm) {
        ++statepoints;
        outers += outer;
        th     += th_steps;
        mix(h, static_cast<unsigned long long>(step_number));
        mix(h, static_cast<unsigned long long>(outer));
        mix(h, static_cast<unsigned long long>(th_steps));
        mix(h, bits(efpd));
        mix(h, bits(eigv));
        mix(h, bits(ppm));
    }
};

/// `"NAME":"value"` for every arm knob, `null` where the variable is unset.
inline std::string armEnvJson() {
    std::string out = "{";
    bool        first = true;
    for (const char* name : kArmEnv) {
        const char* v = std::getenv(name);
        if (!first) out += ",";
        first = false;
        out += "\"";
        out += name;
        out += "\":";
        if (v == nullptr) {
            out += "null";
        } else {
            // The values are short flag-like strings; quote them and drop the
            // two characters that could break the line, rather than pull in a
            // JSON escaper for a receipt.
            out += "\"";
            for (const char* p = v; *p != '\0'; ++p)
                if (*p != '"' && *p != '\\' && *p != '\n') out += *p;
            out += "\"";
        }
    }
    out += "}";
    return out;
}

} // namespace trajectory

// In-core equilibrium-xenon mode (RASBERY_XE_MODE), resolved once per process.
//
//   equilibrium  (default, and what an unset variable means)  The production
//                solver: SolveLoop drives the Xe<->flux fixed point down to
//                XE_EQUILIBRIUM_TOLERANCE at every cascade, and the damper,
//                the interim-flux probe (RASBERY_XE_INTERIM_*) and the
//                per-cascade budget (RASBERY_XE_CASCADE_BUDGET) all hang off
//                that iteration.
//   frozen       The in-core iteration is not run at all.  The I-135/Xe-135/
//                Xe-135m rows of XSSet::_iden keep whatever the deck handed
//                over -- the library initialization, the restart file, or the
//                previous statepoint's depletion -- and the XS reconstruction
//                uses them as they stand.
//   once         The middle mode.  The iteration still runs, but every cascade
//                is capped at ONE step: the flux converges, the closed-form
//                equilibrium is applied exactly once (undamped), the flux
//                re-converges, and Xe does not fire again until the next
//                cascade -- a T/H commit, a search commit, or the next
//                SolveLoop entry, the same three sites the cascade counters
//                and RASBERY_XE_CASCADE_BUDGET already open a segment at.
//                There is no convergence requirement on the Xe residual: the
//                single Picard step is accepted as it stands.
//
// WHY "once" EXISTS.  Frozen mode's two measured failures are the same defect,
// that the held Xe is not consistent with the statepoint's own flux: a fresh
// core never equilibrates at all (Xe sits at the library's fresh-fuel trace,
// worth ~250 ppm of critical boron against MASTER), and a restart deck whose
// later statepoints move far from the restart state can drive the macro-XS
// reconstruction out of the fitted range (the Xe-135 density is a
// spectral-history coordinate) until BiCGSTAB returns a non-finite iterate.
// One step per segment costs ~6-12 Xe updates per statepoint instead of ~100
// and leaves Xe equilibrated against the most recent converged flux, so both
// failures are structurally out of reach while nearly all of the cascade
// multiplicity -- which is where the outer budget goes -- is still retired.
//
// BOUNDED STEPS, NOT ONE STEP.  The first cut of the mode took literally one
// UNDAMPED step per segment, and that was measured to be unsafe on exactly the
// decks it was built for: a restart deck arrives far from its current-flux
// equilibrium, so the single step covered the WHOLE distance at once and handed
// the converged flux an absorption-XS jolt it could not absorb -- 17 of 64
// i-SMR CY02 restart jobs at M64 died in the following re-convergence with a
// non-finite BiCGSTAB iterate, each at a different statepoint.  Equilibrium
// mode survives the same distance because it crosses it in many small damped
// steps.  So what a segment is capped at is the step SIZE, not just the count:
// a step whose raw relative change exceeds XE_ONCE_TRUST buys the cascade
// another, damped step (XE_ONCE_MAX_STEPS total), each followed by the usual
// flux re-convergence, and the first cascade of a restart's first statepoint --
// the one with no earlier step to measure -- is damped up front.  A mid-cycle
// cascade lands inside the trust region on its first step and still costs
// exactly one.
//
// WHAT once IS NOT.  It is still not the equilibrium fixed point: one to three
// Picard steps from a moved macro-XS state land near it, not on it, so results
// move at the pcm level and this is a campaign mode compared against the exact
// run, never an acceptance path.  The oscillation damper and the interim-flux
// probe are both bypassed -- the mode sizes its own steps, so the damper has
// nothing left to decide, and the interim probe exists to spend cascade steps
// on a loose flux, which is the opposite of what this mode does.
//
// WHY.  Measured on KNGR CY1: 17,564 of 21,271 outer iterations (82.6 %) were
// Xe-update reconvergence cascades.  Retiring them is the single largest lever
// on N_outer in the Sec 1 Amdahl model, projected at ~3.5-4x on a single run.
//
// WHAT IT IS NOT.  Frozen mode changes the physics -- the published Xe is no
// longer in equilibrium with the published flux -- so it is a deliberately
// selected comparison mode (MASTER is run in its matching mode and the two are
// compared same-mode), never an acceptance path.  It does NOT touch depletion:
// Deplete / PredictorStep / CorrectorStep still run their Bateman/CRAM advance
// and still apply their own equilibrium overwrite at the depletion rates, so
// the I/Xe/Sm inventory bookkeeping between statepoints is unchanged.  Only the
// in-SolveLoop refresh stops.
//
// NO SAMARIUM COUNTERPART.  There is nothing symmetric to switch: RASBERY has
// no in-core Sm equilibrium iteration.  Sm-149 and Pm-149 move only through the
// depletion Bateman/CRAM solve (Chiffon::Isotope::iSm149 appears in no
// equilibrium formula -- ApplyXeEquilibrium writes the three Xe-chain rows and
// nothing else), and frozen mode leaves depletion running.  MASTER's
// SAMARIUM=2 deck flag therefore has no RASBERY analogue to hold.
enum class XeMode { EQUILIBRIUM, FROZEN, ONCE };

inline XeMode xeMode() {
    static const XeMode mode = [] {
        const char* value = std::getenv("RASBERY_XE_MODE");
        if (value == nullptr)
            return XeMode::EQUILIBRIUM;
        std::string s(value);
        for (char& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s.empty() || s == "equilibrium")
            return XeMode::EQUILIBRIUM;
        if (s == "frozen")
            return XeMode::FROZEN;
        if (s == "once")
            return XeMode::ONCE;
        // A typo must not silently buy the default: this is a campaign switch
        // whose arms produce different physics.
        std::cerr << "[RASBERY][WARN][xe] RASBERY_XE_MODE=\"" << value
                  << "\" is not a mode (equilibrium|frozen|once); using equilibrium\n";
        return XeMode::EQUILIBRIUM;
    }();
    return mode;
}

/// True when the in-core Xe<->flux fixed-point iteration is switched off.
inline bool xeFrozen() { return xeMode() == XeMode::FROZEN; }

/// True when the iteration still runs but every cascade is bounded: at most
/// XE_ONCE_MAX_STEPS steps, sized so no one of them moves the inventory by more
/// than the trust region.  Deliberately NOT folded into xeFrozen(): frozen mode's
/// short-circuit (has_eq_xe) must stay a frozen-only property, because once
/// mode needs every arm of the machinery that gate feeds.
inline bool xeOnce() { return xeMode() == XeMode::ONCE; }

/// The mode as the [RASBERY][PHYSICS_MODE] and [RASBERY][SPTELEM] receipts
/// publish it, so a run can be sorted by mode without re-reading the env.
inline const char* xeModeName() {
    switch (xeMode()) {
        case XeMode::FROZEN: return "frozen";
        case XeMode::ONCE:   return "once";
        case XeMode::EQUILIBRIUM: break;
    }
    return "equilibrium";
}

// ---------------------------------------------------------------------------
// Execution mode: one deck at a time, or --batch-mode.
//
// WHY THIS EXISTS.  The Anderson adoption default (below) is mode-dependent,
// and the decision point has to know which kind of run it is in.  Nothing in
// Driver could answer that: `--batch-mode M` is an argv flag, the batch width
// lives behind the CUDA backend (rasberyBatchWidth(), absent in a stub build),
// and neither is a legal dependency for a header the CPU-only build compiles.
//
// SO main() DECLARES IT.  One latch, set once from the SAME predicate that
// selects the batch branch (`batch_width > 0 && !rasbery_inputs.empty()`), and
// set BEFORE the first receipt is emitted -- which is before any Driver exists
// and before any mode-dependent gate can resolve.  The default is Single, so a
// unit test, a tool, or a direct Driver construction is a single run by
// construction rather than by remembering to declare it.
//
// A declaration that arrives after somebody already read the mode is a
// programming error, not a runtime condition: the gate it was supposed to feed
// has already cached the wrong answer.  It says so on stderr rather than
// pretending the late value took effect.
// ---------------------------------------------------------------------------
enum class ExecutionMode { Single, Batch };

namespace exec_detail {
inline std::atomic<ExecutionMode>& modeCell() {
    static std::atomic<ExecutionMode> cell{ExecutionMode::Single};
    return cell;
}
/// Latched by the first READ, so a late declaration can be detected.
inline std::atomic<bool>& observed() {
    static std::atomic<bool> seen{false};
    return seen;
}
} // namespace exec_detail

/// The mode this process runs in.  Reading it latches it.
inline ExecutionMode executionMode() {
    exec_detail::observed().store(true, std::memory_order_relaxed);
    return exec_detail::modeCell().load(std::memory_order_relaxed);
}

/// True under `--batch-mode M` with decks to run; false for a single run.
inline bool batchExecution() { return executionMode() == ExecutionMode::Batch; }

/// The mode as the receipts publish it.
inline const char* executionModeName() { return batchExecution() ? "batch" : "single"; }

/// main() declares this once, from the batch-branch predicate, before anything
/// reads it.
inline void declareExecutionMode(ExecutionMode requested) {
    if (exec_detail::observed().load(std::memory_order_relaxed) &&
        exec_detail::modeCell().load(std::memory_order_relaxed) != requested) {
        // REFUSE the late change rather than record it.  A mode-dependent
        // default has already cached against the old value, so storing the new
        // one would leave the cell disagreeing with the state the run is
        // actually in -- and the receipt reads the cell.  Keeping the old value
        // means what ran and what is reported stay the same thing.
        std::cerr << "[RASBERY][WARN][exec] execution mode declared after it was already "
                     "read; a mode-dependent default has resolved against the previous "
                     "value, so the declaration is refused and the mode stays as it was\n";
        return;
    }
    exec_detail::modeCell().store(requested, std::memory_order_relaxed);
}

// Safeguarded Anderson acceleration of the in-core Xe fixed point
// (RASBERY_XE_ANDERSON, plan Rev.4 Sec 10).  OFF means the plain Picard cascade
// byte for byte: the gate is one cached read, the solver holds one extra bool,
// and the only new call in the step is short-circuited by it.
//
// ADOPTION (2026-08-27): THE DEFAULT IS MODE-DEPENDENT.
//
//   single run (no --batch-mode)  -> ON  by default
//   --batch-mode                  -> OFF by default
//   RASBERY_XE_ANDERSON           -> overrides the default in BOTH modes:
//                                    0|off|false|no -> OFF, 1|on|true|yes -> ON
//                                    (trimmed and case-folded), anything else
//                                    warns and falls back to the mode default
//
// NOTE, THE EMPTY VALUE CHANGED MEANING.  Before the adoption an empty
// RASBERY_XE_ANDERSON meant OFF; it now means "no request", so a single run
// takes the default and gets ON.  A script that relied on an inherited empty
// variable to disable the arm has to say RASBERY_XE_ANDERSON=0 out loud.
//
// ON for a single deck because the 238 validation measured 1.69x (93.6 -> 55.5
// s, outers -51 %, 95 % acceptance) with the MASTER agreement IMPROVING, not
// degrading: reactivity 1.970 -> 1.905 pcm, AO 0.022 -> 0.013.  That direction
// is the whole argument -- the v1 baseline carried a ~2-3 pcm artifact of a
// TRUNCATED (unconverged) Xe fixed point, and converging it properly moves
// RASBERY toward MASTER.  So this is adopted as a CORRECTNESS improvement that
// happens to be faster, and the Gate A delta against the v1 baseline is the old
// baseline's defect being exposed (hence the v2 re-freeze).
//
// OFF in batch because it measured NET-NEGATIVE there: 202 vs 216 cases/h at
// M64.  The cause is not I/O (the writer thread removed that term) and not the
// algorithm (batch acceptance 95.6 % matches solo 95.0 %) -- it is ARRIVAL-WIDTH
// STARVATION.  Anderson makes each job need ~38 % fewer outers, so a job spends
// less wall in the solve, so fewer jobs are concurrently inside the batched CMFD
// rendezvous at any instant; the measured mean width falls and every batched
// kernel pays for slots nobody is using.  The per-job win is real and the
// aggregate loses it.  The designed fix is slot compaction (Phase 5 plan), which
// decouples grid cost from declared width; until that lands the batch default
// stays OFF, and RASBERY_XE_ANDERSON=1 keeps the batch A/B experiments running.
//
// The env override is deliberately symmetric: batch runs can force it ON to
// re-measure, and a single deck can force it OFF to reproduce a legacy (v1
// baseline) trajectory.
//
// WHAT IT IS.  Anderson does NOT change the map, the acceptance semantics or
// the convergence test.  The same F(x) is evaluated at the same points, the
// same XE_EQUILIBRIUM_TOLERANCE decides when the cascade is done, and the
// same damping machinery is in charge whenever it engages.  What changes is
// the ITERATE: instead of x_{k+1} = F(x_k) it takes the extrapolation the
// least-squares fit of the last two residual differences predicts, which on a
// contraction of rho 0.4-0.75 (measured: fresh-core cascades of 10-15 Picard
// steps, restart decks ~97 Xe updates per statepoint) reaches the SAME
// tolerance in 2-4 steps.  83.5 % of outers on both fleets are Xe
// re-convergence, so the cascade multiplicity is the whole lever.
//
// WHY IT IS NOT THE STEP HEURISTICS.  RASBERY_XE_MODE=frozen/once cap the
// cascade -- they publish an inventory that is NOT the fixed point, which is
// why both failed on the restart fleet (shock, then trust-loop overhead).
// Anderson converges the same fixed point to the same tolerance; the published
// state is the one the production test accepted.
//
// EQUILIBRIUM MODE ONLY.  frozen has no cascade to accelerate and once is
// deliberately not converging one, so asking for Anderson in either is a
// misconfiguration and is reported rather than silently honoured.

/// Where the effective Anderson state came from: the mode-dependent adoption
/// default, or an explicit RASBERY_XE_ANDERSON.  Published in the
/// [RASBERY][PHYSICS_MODE] receipt, because "AA was on" and "AA was on because
/// somebody asked for it" are different facts about a measurement.
enum class XeAndersonSource { Default, Env };

struct XeAndersonGate {
    bool             on;
    XeAndersonSource source;
};

inline const XeAndersonGate& xeAndersonGate() {
    static const XeAndersonGate resolved = [] {
        const char* value = std::getenv("RASBERY_XE_ANDERSON");
        std::string requested = (value != nullptr) ? std::string(value) : std::string();

        // TRIM, THEN CASE-FOLD, before anything looks at the value.  This tree
        // has been bitten by CRLF-terminated env files more than once, and a
        // trailing '\r' turning "0" into an unrecognised word would buy the
        // OPPOSITE state of the one the operator wrote.  Same folding rule as
        // xeMode(), which is the precedent for every RASBERY mode switch.
        constexpr const char*        kBlank = " \t\r\n\v\f";
        const std::string::size_type first  = requested.find_first_not_of(kBlank);
        if (first == std::string::npos)
            requested.clear();
        else
            requested =
                requested.substr(first, requested.find_last_not_of(kBlank) - first + 1);
        for (char& c : requested)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        // Unset, empty and all-whitespace are the same thing: no request.  An
        // inherited empty variable must not read as an explicit "off" and
        // suppress the default.
        bool             want   = false;
        XeAndersonSource source = XeAndersonSource::Default;
        if (!requested.empty()) {
            if (requested == "0" || requested == "off" || requested == "false" ||
                requested == "no") {
                want   = false;
                source = XeAndersonSource::Env;
            } else if (requested == "1" || requested == "on" || requested == "true" ||
                       requested == "yes") {
                want   = true;
                source = XeAndersonSource::Env;
            } else {
                // A typo is not a state.  Name the value that was rejected --
                // that is the difference between a two-second fix and an A/B
                // whose two arms were secretly the same run -- and fall through
                // to the mode default rather than guessing at an intent.
                std::cerr << "[RASBERY][WARN][xe] RASBERY_XE_ANDERSON=\"" << value
                          << "\" is not a state (0|off|false|no or 1|on|true|yes); using "
                             "the mode default\n";
            }
        }
        if (source == XeAndersonSource::Default) {
            // The adoption default: ON for a single run, OFF under --batch-mode
            // (arrival-width starvation -- see the block above).
            want = (executionMode() == ExecutionMode::Single);
        }

        // EQUILIBRIUM MODE ONLY, in both provenances.  frozen has no cascade to
        // accelerate and once is deliberately not converging one.  An explicit
        // request in those modes is a misconfiguration and is reported; the
        // default silently declines, because nobody asked for anything.
        if (want && xeMode() != XeMode::EQUILIBRIUM) {
            if (source == XeAndersonSource::Env)
                std::cerr << "[RASBERY][WARN][xe] RASBERY_XE_ANDERSON is set with "
                             "RASBERY_XE_MODE="
                          << xeModeName()
                          << "; Anderson accelerates the equilibrium cascade and has nothing "
                             "to accelerate in that mode -- staying off\n";
            return XeAndersonGate{false, source};
        }
        return XeAndersonGate{want, source};
    }();
    return resolved;
}

/// The effective state, and the ONLY thing the solve path consults.
inline bool xeAnderson() { return xeAndersonGate().on; }

/// "default" or "env" -- the provenance of the state above.
inline const char* xeAndersonSourceName() {
    return xeAndersonGate().source == XeAndersonSource::Env ? "env" : "default";
}

/// Per-rejection reason trace for the Anderson arm (RASBERY_XE_ANDERSON_DEBUG).
/// Off by default; the counters in the SPTELEM receipt carry the totals, and
/// this is the only thing that resolves them into a reason distribution.
inline bool xeAndersonDebug() {
    static const bool on = std::getenv("RASBERY_XE_ANDERSON_DEBUG") != nullptr;
    return on;
}

class Driver {
public:
    /// The trajectory receipt's payload, as a value.
    ///
    /// WHY IT EXISTS.  The trajectory digest is the campaign's identity check
    /// between two runs of the same binary, and until now the only way to read
    /// it was to grep this process's own stdout.  That is fine for a launcher
    /// and impossible for WP8's `--evaluator`: the per-case receipt has to
    /// carry the digest of the case it is reporting, on the thread that ran it,
    /// interleaved with sixty-three siblings' output.  So the same numbers the
    /// receipt prints are also kept as a value.
    ///
    /// `complete` is false when Drive() threw before the fold closed, which is
    /// how a caller tells "this case has no digest" from "this case digested to
    /// zero".  Nothing here is read by the solver; it is a pure output, and it
    /// is deliberately NOT folded into the digest itself.
    struct CaseReceipt {
        unsigned long long digest      = 0;
        int                statepoints = 0;
        long long          outers      = 0;
        long long          th_updates  = 0;
        int                slot        = -1;
        bool               complete    = false;
        /// WP10.1: the canonical duplicate key of the case that was just run.
        /// Carried in the receipt as well as printed, so `--evaluator` can hand
        /// it back to the controller without anyone grepping stdout.
        std::string        case_key;
        /// WP10.3: WHAT THIS CASE SOLVED AT, as the case itself resolved it.
        ///
        /// The process-level [PHYSICS_MODE] receipt answers "what was this
        /// PROCESS configured for", and once fidelity is per case that is no
        /// longer the same question as "what did THIS case solve at".  A
        /// mixed-fidelity wave has ONE [PHYSICS_MODE] line and sixty-four
        /// answers, so the per-case audit (tools/exact_audit.py
        /// audit_case_fidelity) keys on these, and they are filled even when
        /// nothing per-case was requested -- a field that only some cases
        /// carried would be a field no audit could require.
        std::string        policy;            ///< strict | A2 | L3coarse | ...
        std::string        physics_fidelity;  ///< the plan Sec 6.2 spelling
        std::string        statepoint_grid;   ///< "full" unless the deck was coarsened
        std::string        fidelity_declared; ///< the request's word, raw, or empty
        /// The screening case_key this one was promoted FROM, or empty.
        std::string        promoted_from;
        bool               acceptance_eligible = false;
    };

private:
    std::string _input;
    std::string _result_output;
    /// What this case is asked to produce.  Per DRIVER, not per process: one
    /// --batch-mode wave may carry light screening cases and full elite cases
    /// side by side (see ResultMode in BatchLightResult.h).
    ResultMode  _result_mode = ResultMode::Full;
    /// WP10.2 warm start.  Empty means off, on both halves.
    std::string _warm_start_from;
    std::string _warm_state_out;
    /// WP10.3.  HOW this case solves, per Driver for the reason ResultMode and
    /// the warm start are: one wave carries a screening population and a
    /// promoted elite, and they do not converge the same way.  Defaulted from
    /// the environment, so an unconfigured Driver is the pre-WP10.3 Driver.
    CaseFidelity _fidelity = processCaseFidelity();
    /// The trajectory fold of the last Drive(), for a caller that cannot grep
    /// stdout (WP8's --evaluator).  Written once, at the same place the
    /// trajectory receipt is printed; never read by the solver.
    CaseReceipt _case_receipt{};

    static constexpr double CMFD_FLUX_L2_TOLERANCE = 1.0e-6;
    // Search / T-H feedbacks engage once the CMFD flux L2 residual drops below this, so they
    // act on a meaningful (not rough-early) flux while still co-converging in the single loop.
    static constexpr double FEEDBACK_TRIGGER = 1.0e-3;
    // T/H feedback convergence on the relative Doppler (fuel) temperature change (PARCS eps_Dop).
    static constexpr double TH_DOPPLER_TOLERANCE = 1.0e-2;
    // Equilibrium Xe is a flux-dependent material feedback.  It must be
    // initialized at BOC and reconverged after boron/T-H changes.
    static constexpr double XE_EQUILIBRIUM_TOLERANCE = 1.0e-6;
    static constexpr int    XE_EQUILIBRIUM_MAX_ITER  = 100;
    // Damping for the equilibrium-Xe fixed-point iteration.  The damped map
    // x <- x + relax*(F(x) - x) has the same fixed point, so only the path changes; relax
    // = 0.5 turns a contraction factor near -1 (a two-state limit cycle) into one near 0.
    //
    // The trigger has to be strict, on two axes, and both were found the hard way.
    //
    // Damping is not free: it halves the step of a healthy iteration too, so a monotone
    // contraction that took 60 iterations takes ~120 and can run into the cap.  A first
    // trigger of "failed to shrink by 10%" engaged 45 times on i-SMR CY02; the run got
    // faster, but four statepoints stopped at ~1e-5 instead of the 1e-6 tolerance and keff
    // moved up to 8.9 pcm.  Hence XE_OSCILLATION_STREAK: the step must fail to shrink AT ALL,
    // this many times in a row.  A contracting iteration never does that; a limit cycle does
    // it immediately.
    //
    // That alone still misfires.  Near the tolerance the residual jitters in the arithmetic
    // noise: i-SMR CY03 contracted cleanly to 1.46e-6 and then read 1.65, 1.92, 3.64e-6 --
    // three "non-contractions" at the noise floor of an iteration that was already finished.
    // Damping there changed CY03 (and, through the restart chain, CY04) by several pcm for
    // nothing.  XE_OSCILLATION_FLOOR keeps the detector out of that regime: only an iteration
    // still two orders of magnitude from its tolerance is stuck rather than finishing.  The
    // real pathology sits far above it (APR1400 cy01 oscillates at ~1e-3 for 80+ steps).
    // Failing this guard costs nothing beyond the pre-existing behaviour.
    static constexpr int    XE_OSCILLATION_STREAK = 3;
    static constexpr double XE_OSCILLATION_FLOOR  = 100.0 * XE_EQUILIBRIUM_TOLERANCE;
    static constexpr double XE_DAMPED_RELAX       = 0.5;
    // RASBERY_XE_MODE=once caps the step COUNT per cascade; these two cap the step
    // SIZE, which is what actually has to be bounded.
    //
    // The count cap alone was measured to be unsafe.  A restart deck arrives with an
    // inventory equilibrated against a DIFFERENT flux, so the distance to the
    // current-flux equilibrium is large; one undamped step covers all of it at once
    // and hands a converged flux a correspondingly large absorption-XS jolt.  On the
    // i-SMR CY02 restart set at M64, 17 of 64 jobs then died in the following flux
    // re-convergence with a non-finite BiCGSTAB residual, at a different statepoint in
    // each deck.  Equilibrium mode covered the same distance safely because it covered
    // it in many small damped steps.
    //
    // So: a step whose RAW (pre-damping, see UpdateEquilibriumXenon) relative change
    // exceeds XE_ONCE_TRUST is a step the flux is not trusted to absorb, and the
    // cascade is allowed up to XE_ONCE_MAX_STEPS total, the extras damped, each
    // followed by the usual flux re-convergence.  The stopping test is the trust
    // threshold, NOT XE_EQUILIBRIUM_TOLERANCE: the mode is bounding shock, not
    // converging the fixed point -- that is still equilibrium mode's job.  A typical
    // mid-cycle cascade lands well inside the trust region on its first step and the
    // one-step fast path is exactly what it was.
    static constexpr int    XE_ONCE_MAX_STEPS = 3;
    static constexpr double XE_ONCE_TRUST     = 0.10;
    // --- Safeguarded Anderson (RASBERY_XE_ANDERSON, plan Rev.4 Sec 10) -------
    //
    // Window depth.  Sec 10.5 says start at 2 and try 3 only after the Gate:
    // two residual differences is the smallest window that can see curvature,
    // and every extra column is another chance for an ill-conditioned normal
    // matrix on a map whose residual is already near the noise floor.
    static constexpr int    XE_ANDERSON_DEPTH = 2;
    // Trust region, as a MULTIPLE of the Picard step this candidate replaces,
    // in the same raw |dXe|/|Xe| metric UpdateEquilibriumXenon returns
    // (RASBERY_XE_ANDERSON_MAX_STEP overrides).  Anderson's whole failure mode
    // is a long extrapolation off a nearly-singular fit, and the measured
    // consequence of an oversized Xe step on this solver is not a wrong answer
    // but a dead run: an absorption-XS jolt the converged flux cannot absorb,
    // which killed 17 of 64 i-SMR CY02 restart jobs at M64 with a non-finite
    // BiCGSTAB iterate.  1.25 lets the extrapolation overshoot the Picard step
    // enough to be worth taking -- an AA(1) secant step on a rho=0.5 contraction
    // is ~2x the Picard step in RESIDUAL but well inside this in step norm --
    // while leaving the shock bounded by something the flux already survives.
    static constexpr double XE_ANDERSON_MAX_STEP = 1.25;
    // Conditioning floor for the least-squares fit, as a normalized Gram
    // determinant: a two-column solve needs det(G) > this * G00 * G11, i.e. the
    // two residual differences must not be within ~1e-4 rad of parallel, and a
    // one-column solve needs <d,d> > this * <g,g>, i.e. the residual must
    // actually have MOVED relative to its own size.  Both are scale-free, so
    // the same number works at 1e-1 and at 1e-6 residual.  Failing the
    // two-column test drops to the newest column alone rather than rejecting;
    // failing the one-column test rejects.
    static constexpr double XE_ANDERSON_MIN_GRAM = 1.0e-8;
    // A damped iteration legitimately needs about twice the steps for the same contraction,
    // so it gets about twice the budget.  Without this the damping trades a limit cycle for
    // a cap-exhausted, still-unconverged Xe state -- a different way to publish the wrong
    // inventory.
    static constexpr int    XE_EQUILIBRIUM_MAX_ITER_DAMPED = 200;
    // XE_EQUILIBRIUM_MAX_ITER is spent by ONE counter shared by every Xe cascade in a
    // SolveLoop, but a cascade restarts at every committed perturbation (a search trial
    // point, a T/H update): the macro-XS moves, so the Xe fixed point moves with it and
    // the iteration has to contract to 1e-6 all over again.  On a search-heavy statepoint
    // the early cascades therefore eat the budget and the late ones are cut off
    // mid-contraction -- and the inventory that gets PUBLISHED belongs to a late cascade.
    // With RASBERY_XE_CASCADE_BUDGET set the counter is re-armed at each cascade start,
    // and this multiple of the per-cascade budget becomes the SolveLoop-wide ceiling so a
    // pathological deck still cannot spin forever.  Default off; the reset changes the
    // iteration path, so it is a gated correctness fix pending Gate A/B validation.
    static constexpr int    XE_CASCADE_TOTAL_MULTIPLIER = 10;
    // A trial point whose flux never satisfies the joint (delta-k, L2) test inside one full
    // outer budget is almost always a rod-cusping limit cycle: the fractional fine-cell blend
    // flips between two states, so the CMFD matrix alternates and the L2 residual parks at
    // ~1e-4 while k_eff oscillates with an amplitude of a few pcm.  That is a degraded but
    // still usable observation of k_eff(x) -- it is not a reason to abandon the whole solve.
    // Let the search step away from such a point, and only give up after this many events.
    static constexpr int    MAX_FLUX_STALL_EVENTS = 3;
    // Bound on the flux re-convergence performed after falling back to the best trial point.
    static constexpr int    FALLBACK_RECONVERGE_ITER = 200;
    // A k_eff sample taken within a couple of outers of a state change still carries the
    // dhat/Xe transient.  Measured on APR1400 cy01: at 240 EFPD the search commits every 4-6
    // outers and its samples scatter +-5 pcm around a settled k(x) curve that is smooth to
    // 0.5 ppm steps (slope -5.4 pcm/ppm); at BOC the search published k=1.000000 at 2346.04
    // ppm where a settled solve gives 0.998882, a +112 pcm (~ +21 ppm CBC) contamination.
    // With search_tol = 1e-5 the secant then runs on noise, and when two consecutive samples
    // agree to within min_secant_denom it exits no-next-trial-point: 26 of cy01's 47 boron
    // searches ended that way.  Require this many consecutive converged-flux iterations since
    // the last perturbation before the search is allowed to look at k_eff.
    static constexpr int    SEARCH_SETTLE_ITERS = 2;

    enum class SolveExit {
        CONVERGED,        // search (and T/H) tolerances met
        FLUX_STALL,       // flux limit-cycled on too many trial points
        SEARCH_EXHAUSTED, // ran out of critical-search iterations
        NO_PROPOSAL,      // the search could not propose a next trial point
        ITER_EXHAUSTED    // hit the hard outer-iteration safety bound
    };

    static const char* SolveExitName(SolveExit reason) {
        switch (reason) {
            case SolveExit::CONVERGED:        return "converged";
            case SolveExit::FLUX_STALL:       return "flux-limit-cycle";
            case SolveExit::SEARCH_EXHAUSTED: return "search-iterations-exhausted";
            case SolveExit::NO_PROPOSAL:      return "no-next-trial-point";
            case SolveExit::ITER_EXHAUSTED:   return "outer-iterations-exhausted";
        }
        return "unknown";
    }

    struct SolverContext {
        Geometry&    geometry;
        XSSet&       cross_sections;
        BICGCMFD&    cmfd_solver;
        Nodal&       nodal_solver;
        SearchMemory search_memory;
        /// Phase 2 statepoint telemetry (plan Rev.4 Sec 8).  Per-Driver, plain
        /// members, never shared -- see the sptelem comment block.  Drive()
        /// re-arms it at every statepoint boundary; SolveLoop only increments.
        sptelem::Counters telemetry{};
        /// Which statepoint SolveLoop is currently serving, for diagnostics only
        /// (the [WARN][xe] cascade-starvation line).  Drive() stamps both at the
        /// top of the schedule loop; nothing in the solve path reads them.
        int    statepoint = 0;
        double efpd       = 0.0;
        /// One-shot latch for the frozen-Xe startup guard, so the warning is a
        /// property of the run rather than of every statepoint in it.
        bool   xe_frozen_checked = false;
        /// WP10.3.  HOW THIS CASE SOLVES, as a value the case owns.
        ///
        /// SolveLoop is static and takes SolverContext, so this is the carrier
        /// that already existed: the three staged-tolerance knobs used to be
        /// function-local statics inside SolveLoop, which latched the FIRST
        /// case of the process and handed its convergence policy to every later
        /// one.  In an evaluator that answers a mixed wave, that is sixty-three
        /// cases running a fidelity nobody asked them for and a receipt for each
        /// saying otherwise.  Default-constructed it is the process environment
        /// (processCaseFidelity()), so a Driver nobody configured behaves
        /// exactly as this tree did before.
        CaseFidelity fidelity = processCaseFidelity();
        /// WP9-D stage D.  HOW THIS RUN'S CRITICAL SEARCH PROPOSES.  Resolved
        /// once from the environment (Scheduler.h processSearchPolicy) and
        /// carried here for the same reason `fidelity` is: SolveLoop is static,
        /// so a policy that is not on the context is a policy that has to be a
        /// static inside it -- and that is the latch WP10.3 spent a commit
        /// removing.  Default-constructed it is the process environment, and
        /// with every knob unset every field is the built-in default.
        SearchPolicy search_policy = processSearchPolicy();
        /// D1's cross-statepoint slope history and D2's parent boron.  Driver
        /// lifetime, never shared: a carried boron worth belongs to one core,
        /// and a slot refill must not inherit a neighbour's.
        SearchCarry  search_carry{};
    };


    // =======================================================================
    // Rev.7.1 Task 9 link 2: the two hooks a real device outer needs
    // =======================================================================
    //
    // WHAT A HOOK IS ALLOWED TO BE TODAY.  The contract
    // (CudaOuterGraph.h::OuterSegmentHook) asks for a stream-ordered enqueue
    // that returns without draining.  Neither of these can be that yet:
    // BICGCMFD::drive rendezvouses and copies the flux back, and Nodal::drive
    // is host arithmetic over host arrays.  So the hooks are HONEST about it --
    // OuterSegmentHooks::sweep_synchronizes is set, the runner forces a segment
    // of one, and `device_outers` therefore equals the host outer count instead
    // of exceeding `segment_launches` by the budget.
    //
    // WHAT THEY BUY ANYWAY.  Everything between them is now device work on
    // device-resident buffers: updpsi writes the psi the sweep reads, updjnet
    // and upddhat write the jnet and dhat the next step reads, and the 416
    // KiB/outer dhat H2D is gone because the sweep reads the buffer the segment
    // wrote.  BICGCMFD::updpsi / updjnet / upddhat are not called at all, which
    // is what host_body_calls reports.
    struct OuterHookCtx {
        SolverContext* ctx      = nullptr;
        double*        eigv     = nullptr;
        double*        residual = nullptr;
        /// Rev.7.1 Task 10 part 2: did THIS outer's sweep get enqueued, or did
        /// the enqueue hook have to fall back to the blocking drive?
        ///
        /// The choice is per OUTER and not per segment, because the gate it
        /// turns on is the Wielandt warm-up -- `_wiel_sweep >=
        /// WIELANDT_WARMUP_SWEEPS`, which is false at the top of every SolveLoop
        /// and becomes true a few outers in.  Deciding it once at arm time would
        /// therefore arm the blocking path for the whole loop, which is the same
        /// trap deviceSweepResident() documents for the residency gate.
        bool           enqueued = false;
    };
    /// ONE CONTEXT PER ARENA SLOT.
    ///
    /// Rev.7.1 Task 18-lite: THIS WAS PROCESS-WIDE AND IT WAS ONE OF THE TWO
    /// THINGS THAT MADE `--batch-mode` UNSERVEABLE.  The context holds a
    /// SolverContext* and pointers to the solve loop's own `eigv` and `residual`
    /// locals; `--batch-mode M` runs M Drivers on M host threads, so a single
    /// static meant every hook the runner called reached whichever Driver armed
    /// last -- its geometry, its CMFD solver, its stack.  Keyed by the CMFD slot
    /// the Driver already owns, each Driver gets its own and nothing is shared.
    ///
    /// A THREAD-LOCAL WOULD ALSO WORK AND IS THE WRONG SHAPE.  The slot is what
    /// the runner, the residency, the counters and the receipt are all keyed on;
    /// making this the one thing keyed on the thread instead would leave two
    /// index spaces that agree only by accident.  An out-of-range index answers
    /// with slot 0's, which is unreachable: every caller has already been
    /// through the ladder's `slot_admitted`.
    static OuterHookCtx& outerHookCtx(int slot) {
        static OuterHookCtx c[gpu::kMaxDeviceSlots];
        const int           i = (slot >= 0 && slot < gpu::kMaxDeviceSlots) ? slot : 0;
        return c[i];
    }

    /// setls + drive, then publish what the sweep observed.
    ///
    /// THE PROBE IS THE WHOLE REASON THIS IS A HOOK and not a plain call: the
    /// convergence kernel that runs after it reads eigv and residual out of
    /// device memory, and nothing else in the segment can see them.  The two
    /// device-only signals come from the sweep's own status -- a negative flux
    /// iterate, and the degenerate-gamma hand-back cmfd_wiel_finalize latches as
    /// sweep_state == 2.
    static bool outerSweepHook(void* raw, gpu::OuterSegmentStream, int slot, unsigned int) {
        OuterHookCtx& h = *static_cast<OuterHookCtx*>(raw);

        // THE RESIDENCY IS SCOPED TO THIS ONE drive(), and that scope is the
        // bug fix.  It used to be latched true at arm time and never cleared,
        // so when the segment stopped running -- a launch failure, an escape,
        // or a deck where SolveLoop refuses and only ReconvergeFlux delegates
        // -- the sweep went on eliding the dhat and psi H2D for the rest of the
        // run while the HOST body wrote the host arrays.  Every subsequent host
        // outer then drove from a device dhat nobody was updating.
        //
        // On kngr_238 that is one device outer, an escape, and then 169 host
        // outers assembling the CMFD operator from a dhat frozen at outer 1:
        // k_eff 1.135 -> 0.678 locally, and a non-finite abort on the server.
        //
        // Scoped this way the flag says exactly what the sweep needs to know --
        // 'the caller of THIS drive owns dhat and psi on the device' -- and it
        // cannot outlive the caller that made it true.
        h.ctx->cmfd_solver.setOuterSegmentResident(true);
        h.ctx->cmfd_solver.setls(*h.eigv);
        h.ctx->cmfd_solver.drive(*h.eigv, h.ctx->geometry.PhifMutable(), *h.residual);
        h.ctx->cmfd_solver.setOuterSegmentResident(false);

        // NO DEVICE-ONLY SIGNALS FROM A HOST-DRIVEN DRIVE.
        //
        // The probe's negative_flux and rayleigh fields mean 'the sweep halted
        // mid-flight and handed back'.  driveDeviceSweeps never does that: it
        // owns the negative-flux retry and it finishes the degenerate-gamma
        // sweep on the Rayleigh branch itself, then keeps looping, and only
        // returns once neither is outstanding.  Publishing them here reported a
        // condition the host had already resolved -- and worse, it reported a
        // STALE one, because the latches are only written when a device sweep
        // actually ran and drive() takes the host loop for the whole Wielandt
        // warm-up.  That is the negative_flux escape kngr_238 raised on its
        // very first device outer.
        //
        // A genuinely bad iterate is still caught: k_outer_refresh_inputs tests
        // eigv and residual for finiteness, which is the check that has meaning
        // on this path.  The two signals come back when the sweep itself is
        // stream-ordered and can hand back mid-segment (Task 10 part 2).
        return gpu::rasberyPublishOuterProbe(slot, *h.eigv, *h.residual, false, false);
    }

    // =======================================================================
    // Rev.7.1 Task 10 part 2: the same sweep, in two stream-ordered halves
    // =======================================================================
    //
    // WHY IT IS TWO HOOKS AND NOT ONE.  A drive is an enqueue AND an
    // observation, and only the first of them is stream-ordered.  Splitting them
    // is what lets the segment put its own work -- updjnet, the jnet download --
    // BETWEEN the launch and the observation, so the one synchronise the nodal
    // drive forces covers the sweep as well.  A single hook would have to
    // synchronise inside itself to return an answer, which is the round trip the
    // whole task is removing.
    //
    // THE PROBE IS PUBLISHED BY A DEVICE KERNEL on the normal path
    // (cmfd_sweep_verdict, CudaBICGBackend.cu): eigv, residual, the negative
    // census and the Rayleigh latch are already in device memory when the graph
    // ends, and the convergence kernel reads them from device memory, so the
    // host was only ever carrying them from one device buffer to another.
    static bool outerSweepEnqueueHook(void* raw, gpu::OuterSegmentStream stream, int slot,
                                      unsigned int outer_index) {
        OuterHookCtx& h = *static_cast<OuterHookCtx*>(raw);
        const gpu::CudaOuterSegment::ProbeAddresses a =
            gpu::rasberyOuterSegment(slot).probeAddresses(slot);
        h.enqueued = false;
        if (!a.valid) return false;
        CudaBatchArena::CmfdSweepProbeSink sink;
        sink.eigv      = a.eigv;
        sink.residual  = a.residual;
        sink.negative  = a.negative;
        sink.rayleigh  = a.rayleigh;
        sink.nonfinite = a.nonfinite;
        sink.halt      = a.halt;
        sink.halt_slot = slot;
        // Rev.7.1 Task 10 part 3: THE TWO FIELDS THAT MAKE A DRIVE UNOBSERVED.
        //
        // `accum` is non-null exactly while a HOST-FREE segment is running (the
        // runner scopes it, not the arm), and it says `nobody will look at this
        // launch on its own -- keep the summary yourself`.
        //
        // `patch_from_probe` is why outer 0 is different.  setls is a no-op on
        // the device-assembly arm, so of everything staged for a launch only the
        // eigenvalue and its two reciprocals carry a value the previous outer's
        // OBSERVATION would have produced -- and on outer 0 there was no
        // previous outer in this segment, so `*h.eigv` is the current one and
        // the probe may still describe a drive several host outers old.  From
        // outer 1 the probe holds outer i-1's verdict, which is exactly what
        // finishDrive would have written into `*h.eigv`.
        sink.accum =
            static_cast<CudaBatchArena::CmfdSweepProbeSink::Accum*>(a.accum);
        sink.patch_from_probe = sink.accum != nullptr && outer_index > 0;
        // RAISED HERE, LOWERED IN THE FINISH HALF.  stageSweepIO reads the
        // residency, and on this path it runs inside enqueueDrive AND again
        // inside finishDrive's exceptional blocking launches -- so the flag has
        // to span both halves of one drive, not just the enqueue.  The finish
        // hook lowers it on every exit; the fallback below lowers it itself,
        // because a refused enqueue has no finish half.
        h.ctx->cmfd_solver.setOuterSegmentResident(true);
        h.ctx->cmfd_solver.setls(*h.eigv);
        // Rev.7.1 Task 18: THE RUNNER'S STREAM IS AN ARGUMENT NOW.  When the
        // segment bound the arena's own stream (one deck) the drive sees the
        // same handle it would have used and nothing is joined -- byte for byte
        // the Task 10 path.  In `--batch-mode` the segment keeps a private
        // stream, because a graph capture swallows every enqueue on the stream
        // it is open on and M segments on one stream capture each other; the
        // drive then joins the two with an event pair instead.
        if (h.ctx->cmfd_solver.enqueueDrive(*h.eigv, h.ctx->geometry.PhifMutable(), *h.residual,
                                            sink, stream)) {
            h.enqueued = true;
            return true;
        }
        // THE WIELANDT WARM-UP, AND THE ONLY OTHER WAY OUT.  enqueueDrive
        // refuses exactly when drive() would not have taken the device sweep, so
        // this is the Task 9 hook verbatim -- with one addition: the segment has
        // stream-ordered work in flight (updpsi and the psi mirror the host loop
        // is about to read), and a blocking drive would read those arrays while
        // the copies were still filling them.
        // TWO STREAMS TO SETTLE, NOT ONE.  The blocking drive is about to read
        // and write host arrays that both streams have async copies in flight
        // into -- the arena's for the sweep staging, the runner's for the psi
        // and dhat mirrors this outer issued.  They are the same stream for a
        // single deck and different ones in a batch, and draining a stream
        // twice is free.
        // WP1 (plan Sec 6.3), and the WP1 FOLLOW-UP that gave it teeth.
        //
        // WHY IT STILL CANNOT THROW HERE.  This runs INSIDE a live device outer
        // segment, and a throw would unwind past the segment's stream -- and, on
        // the WHILE arm, past an open graph capture -- with nothing written to
        // clean either up.
        //
        // WHAT CHANGED.  "Cannot throw here" used to mean "counts and the case
        // runs anyway", which on host 181 was 71 host outer bodies, a
        // `contract_pass:false` receipt, and exit 0.  It now means DEFERRED: the
        // violation is latched and raised the moment runSegment() returns, where
        // unwinding is safe.  And the decision is made on the REASON rather than
        // on the fact of a refusal, because one of the reasons -- the Rayleigh
        // warm-up -- is host BY DESIGN and is already excluded by the CMFD seam
        // (see kGpuFullAllowedOuterRefusals for the evidence and for the reasons
        // that were considered and refused).
        //
        // THE REASON COMES FROM THE CALL THAT ACTUALLY DECIDED, not from
        // re-asking the ladder: enqueueDrive() records it at both of its refusal
        // points, so a per-drive state that moved in between cannot make the
        // receipt name a reason that did not apply.
        RASBERY_GPU_FULL_DEFER_ALLOWED(
            Outer, "Driver::outerSweepEnqueueHook",
            BICGCMFD::enqueueRefusalName(h.ctx->cmfd_solver.lastEnqueueRefusal()));
        h.ctx->cmfd_solver.syncSweepStream();
        gpu::rasberySyncSegmentStream(stream);
        h.ctx->cmfd_solver.drive(*h.eigv, h.ctx->geometry.PhifMutable(), *h.residual);
        h.ctx->cmfd_solver.setOuterSegmentResident(false);
        // The blocking drive resolved the retry and the Rayleigh hand-back
        // before returning, so these two are what THIS drive observed rather
        // than a latch from an older one -- prepareDeviceSweeps cleared them on
        // entry.
        return gpu::rasberyPublishOuterProbe(slot, *h.eigv, *h.residual,
                                             h.ctx->cmfd_solver.lastSweepNegativeFlux(),
                                             h.ctx->cmfd_solver.lastSweepRayleigh());
    }

    /// The observation half.  Runs on the segment's own synchronise, so it costs
    /// no transfer -- except on the exceptional launches (sweep state 0 or 2)
    /// that the device could not finish, where it runs the remaining blocking
    /// launches and then has to overrule the verdict kernel's half-drive probe.
    static bool outerSweepFinishHook(void* raw, gpu::OuterSegmentStream, int slot,
                                     unsigned int) {
        OuterHookCtx& h              = *static_cast<OuterHookCtx*>(raw);
        // Nothing was enqueued: the enqueue hook took the blocking drive and has
        // already published this outer's probe.
        if (!h.enqueued) return true;
        h.enqueued                   = false;
        bool          host_continued = false;
        const bool    ok =
            h.ctx->cmfd_solver.finishDrive(*h.eigv, h.ctx->geometry.PhifMutable(), *h.residual,
                                           host_continued);
        // LOWERED ON EVERY EXIT, including the failure.  finishDrive is the end
        // of the drive the enqueue half raised the flag for; leaving it up on a
        // failure path hands the next HOST outer a sweep that elides the dhat
        // and psi H2D, which is exactly the sticky-flag bug in a rarer costume.
        h.ctx->cmfd_solver.setOuterSegmentResident(false);
        if (!ok) return false;
        if (!host_continued) return true;
        return gpu::rasberyOuterSegment(slot).republishAfterHostSweep(
            slot, *h.eigv, *h.residual, h.ctx->cmfd_solver.lastSweepNegativeFlux(),
            h.ctx->cmfd_solver.lastSweepRayleigh());
    }

    // =======================================================================
    // Rev.7.1 Task 10 part 3: the observation half, ONCE PER SEGMENT
    // =======================================================================
    //
    // THE SAME TWO THINGS outerSweepFinishHook DOES, over N drives instead of
    // one: absorb what the sweep did, and -- when the device abandoned a drive
    // -- finish it on the host and overrule the verdict kernel's half-drive
    // probe.  What it does NOT do is read a scalar block: the block belongs to
    // whichever launch was enqueued last, and on a segment that halted that is
    // one whose every kernel was masked.  Everything comes from @p raw_accum,
    // the summary the verdict kernel kept as the segment ran.
    //
    // THE RESIDENCY FLAG SPANS THE WHOLE SEGMENT HERE, for the reason it spans
    // one drive on the per-outer arm: stageSweepIO reads it, and on the
    // exceptional path finishDeferredDrives stages further blocking launches of
    // its own.  Raised by the enqueue half of every outer, lowered here, on
    // every exit including the failure -- leaving it up would hand the next HOST
    // outer a sweep that elides the dhat and psi H2D, which is the sticky-flag
    // failure that cost 169 kngr_238 outers in its original costume.
    static bool outerSweepFinishDeferredHook(void* raw, const void* raw_accum, int slot) {
        OuterHookCtx& h = *static_cast<OuterHookCtx*>(raw);
        h.enqueued      = false;
        if (raw_accum == nullptr) return false;
        const auto& acc =
            *static_cast<const CudaBatchArena::CmfdSweepProbeSink::Accum*>(raw_accum);
        bool       host_continued = false;
        const bool ok             = h.ctx->cmfd_solver.finishDeferredDrives(
            acc, *h.eigv, h.ctx->geometry.PhifMutable(), *h.residual, host_continued);
        h.ctx->cmfd_solver.setOuterSegmentResident(false);
        if (!ok) return false;
        if (!host_continued) return true;
        return gpu::rasberyOuterSegment(slot).republishAfterHostSweep(
            slot, *h.eigv, *h.residual, h.ctx->cmfd_solver.lastSweepNegativeFlux(),
            h.ctx->cmfd_solver.lastSweepRayleigh());
    }

    /// Rev.7.1 Task 10 part 3: may the segment stop asking about cusping?
    ///
    /// A PURE QUERY ON XSSet's OWN STATE, and it is the same state
    /// ApplyRodCusping consults on its first three lines -- which is what makes
    /// `skip it` an identity rather than an optimisation.  See
    /// XSSet::RodCuspingQuiescent for why a segment-scope answer is sound.
    static int outerCuspingQuiescentHook(void* raw) {
        OuterHookCtx& h = *static_cast<OuterHookCtx*>(raw);
        return h.ctx->cross_sections.RodCuspingQuiescent() ? 1 : 0;
    }

    /// Rev.7.1 Task 10 part 3: hand the nodal backend the segment's halt word.
    ///
    /// HERE AND NOT IN THE RUNNER for outerNodalCompletionHook's reason: the
    /// backend is reached through XSSet, which is Driver.h's object.  The runner
    /// owns the halt table and knows nothing about a nodal graph key; the
    /// backend owns the key and knows nothing about a segment.
    static void outerNodalHaltGateHook(void* raw, const void* halt, int slot) {
        OuterHookCtx&   h  = *static_cast<OuterHookCtx*>(raw);
        XsReconBackend* be = h.ctx->cross_sections.EnsureBackend();
        if (be != nullptr) be->setNodalHaltGate(halt, slot);
    }

    /// Rev.7.1 Task 10 part 3: make the nodal stream wait for the segment's.
    ///
    /// The mirror image of outerNodalCompletionHook, which orders the OTHER
    /// direction.  Both halves of the outer's cross-stream pair are events now,
    /// on the host-free arm; the per-outer arm still uses its drain for this one
    /// and never calls this.
    static bool outerNodalWaitHook(void* raw, void* event) {
        OuterHookCtx&   h  = *static_cast<OuterHookCtx*>(raw);
        XsReconBackend* be = h.ctx->cross_sections.EnsureBackend();
        return be == nullptr || be->waitOnSegmentEvent(event);
    }

    /// The nodal drive, exactly as SolveLoop runs it.
    ///
    /// The runner has already brought the device jnet down into Geometry::Jnet
    /// and will take the result back up, so this is the host body verbatim --
    /// which is the point: a hook that reimplemented it would be a second nodal
    /// solver to keep in step.
    ///
    /// Rev.7.1 W3 item 3: `1.0 / *h.eigv` IS STILL COMPUTED AND IS NOW DEAD ON
    /// THE ARM THAT MATTERS.  Nodal::_reigv is read by the CPU body
    /// (Nodal.cpp:290) and by the hybrid arm's by-value scalar, and both still
    /// need it -- so the divide stays.  On the FULL device arm inside a segment
    /// the kernel reads NodalView::reigv_dev instead (NodalKernel.h:415), which
    /// the segment's own k_outer_publish_reigv wrote from the sweep's device
    /// eigenvalue; this quotient is then computed and discarded.  What the
    /// change removed is not the arithmetic, it is the DATA DEPENDENCY: the
    /// eigenvalue no longer has to reach the host for the drive to run.
    static bool outerNodalHook(void* raw, gpu::OuterSegmentStream, int, unsigned int) {
        OuterHookCtx& h = *static_cast<OuterHookCtx*>(raw);
        h.ctx->nodal_solver.reset(1.0 / *h.eigv, h.ctx->geometry.Jnet(),
                                  h.ctx->geometry.Phif(), h.ctx->geometry.Phis());
        h.ctx->nodal_solver.drive();
        return true;
    }

    /// Rev.7.1 W3 item 2: the event that orders the drive that just ran against
    /// the segment's stream, or nullptr when the drive drained itself.
    ///
    /// ASKED AFTER EVERY DRIVE AND NOT CACHED, because the answer is a property
    /// of that drive: the same backend defers on an in-segment canonical outer
    /// and blocks on the Wielandt warm-up outer three lines later, where the
    /// downloads are live and a host reader is next.  It is here rather than in
    /// the runner for outerCanonicalNodalHook's reason -- the backend is reached
    /// through XSSet, which is Driver.h's object.
    static void* outerNodalCompletionHook(void* raw) {
        OuterHookCtx&   h  = *static_cast<OuterHookCtx*>(raw);
        XsReconBackend* be = h.ctx->cross_sections.EnsureBackend();
        return be != nullptr ? be->nodalCompletionEvent() : nullptr;
    }

    /// Rev.7.1 W3 item 3: where the FULL nodal drive reads 1/eigv, or nullptr.
    ///
    /// A PURE QUERY WITH NO SIDE EFFECT, asked at the top of the sweep so the
    /// runner can enqueue its publish kernel behind the sweep that produced the
    /// eigenvalue.  It is here rather than in the runner for the same reason
    /// outerNodalCompletionHook is: the backend is reached through XSSet, which
    /// is Driver.h's object and not the runner's.
    static void* outerNodalReigvSlotHook(void* raw) {
        OuterHookCtx&   h  = *static_cast<OuterHookCtx*>(raw);
        XsReconBackend* be = h.ctx->cross_sections.EnsureBackend();
        return be != nullptr ? be->nodalReigvDeviceSlot() : nullptr;
    }

    /// Rev.7.1 W3 item 3: the matching declaration.
    ///
    /// Set with 1 only for an outer whose drive is the FULL device pipeline and
    /// whose reciprocal this segment has already written on the device; cleared
    /// for every other drive and at every exit.  The runner latches it, so the
    /// steady state is one call per segment rather than one per outer.
    static void outerNodalReigvModeHook(void* raw, int device_resident) {
        OuterHookCtx&   h  = *static_cast<OuterHookCtx*>(raw);
        XsReconBackend* be = h.ctx->cross_sections.EnsureBackend();
        if (be != nullptr) be->setNodalReigvDeviceResident(device_resident != 0);
    }

    /// Rev.7.1 Task 18-lite: which side owns the canonical nodal regions.
    ///
    /// Called by the runner with 1 before every in-segment drive and with 0 at
    /// the segment exit.  It is here and not in the runner because the backend
    /// that adopted the buffers is reached through XSSet, which is Driver.h's
    /// object and not the runner's -- the same layering that keeps the nodal
    /// hook a plain host call.
    ///
    /// lastDriveLeftDeviceFlux() is the SAME source outerLiveStateHook reads for
    /// `device_owns_flux`, so the flux upload the nodal call elides and the flux
    /// upload the segment itself elides can never disagree about one outer.
    static void outerCanonicalNodalHook(void* raw, int in_segment) {
        OuterHookCtx&   h  = *static_cast<OuterHookCtx*>(raw);
        XsReconBackend* be = h.ctx->cross_sections.EnsureBackend();
        if (be == nullptr) return;
        be->setCanonicalNodalSegmentMode(in_segment != 0,
                                         h.ctx->cmfd_solver.lastDriveLeftDeviceFlux());
    }

    /// Rev.7.1 Task 18-lite: will THIS outer's drive take the device path?
    ///
    /// Nodal::DeviceDriveEligible is TryDriveGpu's own refusal test, so the
    /// runner's decision to drop the bridge and the drive's decision to run on
    /// the device are the same decision.  The backend check is the second half
    /// of TryDriveGpu's refusal and it belongs here for the same reason: a
    /// backend that went unavailable mid-run sends the drive to the CPU body,
    /// which reads Geometry::Jnet.
    ///
    /// The hybrid arm is excluded because it downloads trlcff/matM and runs
    /// calculateEven on the host, then finishes in solveNodalPost -- a drive
    /// with a host half in the middle is not one the segment can hold ownership
    /// across.
    static int outerCanonicalNodalEligibleHook(void* raw) {
        OuterHookCtx& h = *static_cast<OuterHookCtx*>(raw);
        if (!rasberyGpuNodalFullEnabled()) return 0;
        if (!h.ctx->nodal_solver.DeviceDriveEligible()) return 0;
        XsReconBackend* be = h.ctx->cross_sections.EnsureBackend();
        return (be != nullptr && be->available()) ? 1 : 0;
    }

    /// Step 7, verbatim from the host loop.
    ///
    /// THIS IS THE WHOLE FIX FOR i-SMR CY02.  The segment used to skip cusping
    /// on the strength of a Stage A eligibility check -- `does any node have a
    /// fractional rod right now` -- evaluated ONCE per SolveLoop entry.  The
    /// host asks a different question, once per OUTER, and ApplyRodCusping can
    /// answer yes from its own prev_scratch with no node fractional at all
    /// (XSSet.cpp:3282).  So on a cusping deck the segment ran a different outer
    /// from the host's and converged somewhere else: 298 outers and k_eff
    /// 1.000003 where the host took 707 and got 0.999975.
    ///
    /// Calling it here is the same call, on the same leakage, at the same point
    /// of the outer.  The runner handles the one device consequence -- a fired
    /// cusping rebuilt _dtil and the device upddhat reads the device copy.
    static bool outerCuspingHook(void* raw, int, unsigned int) {
        OuterHookCtx& h = *static_cast<OuterHookCtx*>(raw);
        if (!h.ctx->cross_sections.ApplyRodCusping(
                *h.eigv, h.ctx->nodal_solver.axialTransverseLeakage()))
            return false;
        h.ctx->cmfd_solver.upddtil();
        return true;
    }

    /// The generations the segment's upload elisions are decided from.
    ///
    /// Four host reads.  Called at the TOP of every outer, again after the
    /// sweep observation, and again after a cusping that fired -- the three
    /// points at which one of them can have moved without the runner looking.
    static void outerLiveStateHook(void* raw, gpu::OuterSegmentLiveState& out) {
        OuterHookCtx& h   = *static_cast<OuterHookCtx*>(raw);
        out.flux_generation = h.ctx->geometry.fluxGeneration();
        out.xs_generation   = h.ctx->cross_sections.hoststateGeneration();
        out.dtil_generation = h.ctx->cmfd_solver.dtilGeneration();
        out.device_owns_flux = h.ctx->cmfd_solver.lastDriveLeftDeviceFlux();
        out.sweep_will_enqueue = h.ctx->cmfd_solver.canEnqueueDrive();
    }

    // =======================================================================
    // WP10.7: THE ADMISSION DOOR -- device residency, established per case
    // =======================================================================
    //
    // WHAT WAS MISSING, AND THE RUN THAT SHOWED IT.  238 GPU1, build 0054838,
    // 20 generations x width 16, PROD + RASBERY_GPU_FULL=1, arm A (no
    // RASBERY_ARENA_PERSIST).  Four cases died at ZERO statepoints in
    // 0.18-0.43 s -- generations 6, 9 and 12's once-per-generation `promote`
    // step and generation 19's case 14 -- every one of them with
    //
    //   [RASBERY][GPU_FULL][VIOLATION] subsystem=flatxs site=XSSet::UpdateFlatXS
    //   reason=the FlatXS device arm declined; the reference reconstruction
    //   loop runs
    //
    // The timing is the whole diagnosis: 0.2 s is a deck parse and nothing
    // else, so the case died at its FIRST UpdateFlatXS -- the one InitXS makes
    // (:InitXS below) -- and never ran a statepoint.
    //
    // THE SKIPPED STEP IS THAT THERE WAS NO STEP.  Run() establishes the OUTER
    // segment's residency at the stand-up call below, per admission, and has
    // since Task 9 link 1.  The FlatXS arm has no such door at all: the
    // XsReconBackend is constructed on FIRST TOUCH, inside
    // XSSet::TryUpdateFlatXSGpu (src/XSSet.cpp:3262-3271), which is inside the
    // fail-closed seam's own call.  So the very first thing that ever asks
    // whether this case's flat-XS device residency exists is the guard that
    // kills the case for it not existing, and the guard cannot say why: the
    // backend's own reason (XsReconBackend::status() -- "no CUDA device: ...",
    // "stream: ...", "scalar buffer allocation failed") goes to a
    // PROCESS-WIDE std::call_once warn.  In a resident evaluator process
    // spanning twenty generations that means the first case to hit it printed
    // a reason and the other three printed nothing at all, which is exactly
    // what the 16k-line arm-A log contains.
    //
    // WHICH HYPOTHESIS THE EVIDENCE SUPPORTS.  (c): a per-case, first-touch
    // establishment with no door, no reason and no receipt -- and, in the
    // no-persist arm, one full free/re-lay-out of the flat-XS blocks per case
    // (`arena_rebuilds` deltas +17/generation = 16 cases + 1 promote, i.e. one
    // per case executed) feeding it.  The other three are refused HERE, from
    // this tree, not from the log:
    //
    //   (a) "the promote case carries a different CaseFidelity/StatepointGrid
    //       and the residency plan is not rebuilt for it" -- REFUSED.  No
    //       device arm reads CaseFidelity: `_fidelity` reaches ReadInput (the
    //       burnup grid) and SolverContext (the tolerances) and nothing else,
    //       and neither TryUpdateFlatXSGpu nor armOuterSegment can see it.  The
    //       fourth death was g0019c0014, a REGULAR strict case with the same
    //       grid as its fifteen siblings, so the grid cannot be the
    //       discriminator either.
    //
    //   (b) "the WP18 slot reset on recycle clears residency flags only the
    //       first admission sets" -- REFUSED.  The FlatXS backend is per XSSet,
    //       per Driver, per case: it has no per-slot flag for a reset to clear.
    //       And the outer half re-binds on EVERY SolveLoop and ReconvergeFlux
    //       entry (armOuterSegment -> bindResidency, which re-patches the slot
    //       table and re-seeds dhat/psi), not on the first only.
    //
    //   (d) "the nine outer/no_residency events are the same root" -- SAME
    //       CLASS, DIFFERENT MECHANISM, and the second half of this work
    //       package.  armOuterSegment's return value was DISCARDED at both of
    //       its call sites, so a bind that failed -- clearing residency_bound
    //       on its way out (CudaOuterGraph.cu:1118) and leaving its reason in
    //       CudaOuterSegment::status() -- was re-derived one line later by the
    //       post-arm ladder as the generic `no_residency`.  The receipt named
    //       the symptom and threw the cause away.  Both call sites now read the
    //       return and name the step.
    //
    // WHAT THIS DOOR DOES.  It runs UNCONDITIONALLY, once per admission,
    // BEFORE any physics -- there is no first-admission special case, and a
    // recycled slot goes through the identical call -- and it asks each
    // REQUESTED arm whether its residency is established, in the layer that
    // knows.  EnsureBackend() is idempotent by construction (XSSet.h:705), so
    // "establish" and "re-establish" are one call and a second admission on the
    // same lane cannot skip it.  When an arm was asked for and cannot be
    // established the contract still raises -- fail-closed is unchanged -- but
    // now at the door, with the establishing layer's OWN reason on the line
    // instead of the seam describing itself.
    //
    // WITH EVERY ARM OFF THIS IS TWO CACHED getenv READS AND A RETURN.
    static void establishDeviceResidency(XSSet& cross_sections, bool outer_stood_up,
                                         std::ostream& receipt) {
        const bool want_outer  = gpu::outerGpuEnabled();
        const bool want_flatxs = rasberyGpuFlatXsEnabled();
        if (!want_outer && !want_flatxs) return;

        // --- the FlatXS arm: construct-or-reuse, and ASK ---------------------
        //
        // Called whether or not the arm is on, because EnsureBackend() is the
        // one place the backend is built and a door that only built it for the
        // armed run would make the two arms differ in construction order.  The
        // backend refuses cheaply with every arm unset (its constructor returns
        // early on exactly that predicate), so this costs one allocation of an
        // empty Impl on the OFF path.
        XsReconBackend* flatxs = cross_sections.EnsureBackend();
        const bool      flatxs_ready = flatxs != nullptr && flatxs->available();
        const std::string flatxs_status =
            flatxs != nullptr ? flatxs->status() : std::string("no backend");

        // --- the receipt, printed on both outcomes --------------------------
        //
        // THE SAME G0 RULE THE REST OF THIS TREE FOLLOWS.  A door that only
        // spoke on failure would leave "the arm was on and established" and
        // "the arm was never asked" looking identical, which is the reading
        // that cost the arm-A investigation its first day.  One line per
        // admission, and it names the case's own status string, so the
        // resident process's Nth case is as diagnosable as its first -- the
        // property the process-wide std::call_once warn does not have.
        receipt << "[RASBERY][RESIDENCY] {\"outer\":" << (want_outer ? "true" : "false")
                << ",\"outer_stood_up\":" << (outer_stood_up ? "true" : "false")
                << ",\"flatxs\":" << (want_flatxs ? "true" : "false")
                << ",\"flatxs_ready\":" << (flatxs_ready ? "true" : "false")
                << ",\"flatxs_status\":\"" << flatxs_status << "\"}" << std::endl;

        // --- fail-closed, at the door, with the refusal NAMED ---------------
        if (want_flatxs && !flatxs_ready)
            RASBERY_GPU_FULL_REQUIRE_RESIDENCY(FlatXs, "Driver: admission residency",
                                               flatxs_status.c_str());
        if (want_outer && !outer_stood_up)
            RASBERY_GPU_FULL_REQUIRE_RESIDENCY(
                Outer, "Driver: admission residency",
                gpu::rasberyOuterSegment().status().c_str());
    }

    /// Everything the refusal ladder needs that only a Driver can see.
    ///
    /// Rev.7.1 Task 18-lite.  Two of the ladder's reasons became per-Driver when
    /// the arena stopped being width 1: WHICH slot this deck holds, and whether
    /// the arena that was stood up has this deck's shape.  Both are asked here,
    /// once per solve-loop entry, and the answer is carried into the pre-arm
    /// gate, the post-arm refusal and every segment this loop runs -- so the
    /// three cannot disagree about one outer.
    ///
    /// It also stamps the thread's slot, which is what makes the [OUTER_GPU]
    /// receipt's per-slot lines say what each deck did rather than what the sum
    /// of the batch did.
    struct OuterSlotClaim {
        int  slot        = -1;
        int  arena_slots = 0;
        bool admitted    = false;

        /// Which runner to ASK.  `slot` is -1 when this run has no resident CMFD
        /// arena at all, and that is a reason the ladder already has a better
        /// word for -- `no_arena`, recorded by armOuterSegment, which is the
        /// only place that can see it.  Asking runner 0 there keeps the receipt
        /// saying what it said before this task: `no_arena` then
        /// `no_residency`, not `no_runner` from an out-of-range index.
        [[nodiscard]] int query() const { return slot >= 0 ? slot : 0; }
    };

    static OuterSlotClaim outerSlotClaim(SolverContext& ctx) {
        OuterSlotClaim c;
        c.slot        = ctx.cmfd_solver.residentSlot();
        c.arena_slots = gpu::rasberyOuterArenaSlots();
        gpu::OuterSegmentDeckShape shape;
        shape.nxyz  = ctx.geometry.nxyz();
        shape.nsurf = ctx.geometry.nsurf();
        shape.nxy   = ctx.geometry.nxy();
        shape.ng    = ctx.geometry.ng();
        // THE FUEL COUNT IS SCANNED, exactly as the stand-up scans it
        // (Driver::Run), because Geometry keeps the flag per node and no count
        // beside it.  Two spellings of `how many fuel nodes` that could disagree
        // is how a deck gets admitted to an arena laid out for a different one.
        int n_fuel = 0;
        for (int l = 0; l < ctx.geometry.nxyz(); ++l)
            if (ctx.geometry.IsFuel(l)) ++n_fuel;
        shape.n_fuel = n_fuel;
        // NO SLOT IS NOT A MISMATCH.  A run with no resident CMFD arena has
        // residentSlot() == -1, and refusing it `geometry_mismatch` here would
        // skip the arm -- and the arm is the only place that can see, and
        // record, that there is no arena.  Admitting it lets armOuterSegment
        // say `no_arena` and the post-arm ladder say `no_residency`, which is
        // the receipt this configuration produced before the slot existed.
        c.admitted = c.slot < 0 || gpu::rasberyOuterSlotAdmitted(c.slot, shape);
        gpu::outerSetThreadSlot(c.slot);
        return c;
    }

    /// Hand the runner the sweep arena's buffers and install the hooks.
    ///
    /// Called once per SolveLoop/ReconvergeFlux entry, because the arena slot is
    /// a property of the BICGCMFD instance and the eigv/residual it must publish
    /// are that loop's locals.  Returns false when anything is missing, in which
    /// case the segment refuses by name and the host loop runs unchanged.
    static bool armOuterSegment(SolverContext& ctx, double& eigv, double& residual) {
        if (!gpu::outerGpuEnabled()) return false;
        const CudaBatchArena* arena = ctx.cmfd_solver.residentArena();
        // Rev.7.1 Task 18-lite: THE CMFD SLOT IS THE PHYSICS-ARENA SLOT.
        //
        // It used to be hard-coded 0 with the comment "the physics arena is
        // width 1 (link 1)", which was true and is what made a batch unservable:
        // M Drivers bound their residency into one slot table entry and adopted
        // one slot's jnet/phis as their canonical nodal set.  The arena is now
        // stood up at the run's width with the SAME index space the CMFD and
        // nodal arenas use, so this is the slot -- no mapping, no second
        // allocator, and no way for the three to disagree about which deck slot
        // m belongs to.
        const int             slot  = ctx.cmfd_solver.residentSlot();
        if (arena == nullptr || slot < 0) {
            gpu::noteOuterSegmentRefusal(gpu::OuterSegmentRefusal::NoArena);
            return false;
        }
        // The reason lives HERE because this is the only place that can see it:
        // the receipt is printed from a process-wide singleton with no solver to
        // ask.  Recording it makes `sweep_not_resident` appear in refusals{}
        // instead of the run looking like nobody ever tried.
        if (!ctx.cmfd_solver.deviceSweepResident()) {
            gpu::noteOuterSegmentRefusal(gpu::OuterSegmentRefusal::SweepNotResident);
            return false;
        }

        const CudaBatchArena::CmfdResidentView view = arena->residentView(slot);
        if (!view.valid) {
            // F11 (review doc Sec 3).  This used to be a bare `return false`.
            // The arena existed, the slot was ours and the sweep was resident,
            // so every earlier refusal had already been ruled out -- and then
            // the segment declined with NOTHING in `[OUTER_GPU].refusals`.  A
            // run that lost the outer here looked exactly like a run that never
            // reached armOuterSegment at all, which is the one thing this
            // receipt exists to tell apart.
            gpu::noteOuterSegmentRefusal(gpu::OuterSegmentRefusal::NoResidency);
            return false;
        }

        gpu::OuterSegmentResidency residency;
        residency.flux       = view.phi;
        residency.psi        = view.psi;
        residency.dtil       = view.dtil;
        residency.dhat       = view.dhat;
        residency.xsnf       = view.xsnf;
        residency.host_jnet  = ctx.geometry.Jnet();
        residency.host_flux  = ctx.geometry.Phif();
        residency.host_dhat  = ctx.cmfd_solver.dhatData();
        residency.host_psi   = ctx.cmfd_solver.psiData();
        residency.host_xsnf  = ctx.cross_sections.xsnfData();
        residency.host_dtil  = ctx.cmfd_solver.dtilData();
        // Rev.7.1 Task 18-lite: the host end of the canonical nodal set's phis.
        residency.host_phis  = ctx.geometry.Phis();
        residency.arena_slot = slot;
        residency.valid      = true;

        OuterHookCtx& hc = outerHookCtx(slot);
        hc.ctx           = &ctx;
        hc.eigv          = &eigv;
        hc.residual      = &residual;

        // Rev.7.1 Task 10 part 2: take the stream-ordered sweep when the drive
        // can actually be enqueued, and the blocking one otherwise.
        //
        // canEnqueueDrive() is drive()'s OWN gate asked without running
        // anything, so this cannot arm an arm drive() would not have taken --
        // and when it is false (the Wielandt warm-up, no arena, the form-probe
        // capture) the hooks are exactly Task 9's and the budget goes back to
        // one.  Both arms bind the SAME stream, because the segment's kernels
        // read and write the buffers the sweep touches and stream order is what
        // says so.
        // canEnqueueDrive() is NOT asked here, deliberately: it carries the
        // per-drive Wielandt warm-up, which is false at the top of every
        // SolveLoop, so arming on it would arm the blocking path for the whole
        // loop and never engage.  What is asked here is the RUN's shape -- is
        // there an arena stream to share -- and the per-outer question is asked
        // per outer, inside the enqueue hook.
        //
        // ============================================================
        // A BATCH KEEPS ITS OWN STREAM AND STILL GETS THE STREAM-ORDERED SWEEP
        // ============================================================
        //
        // Rev.7.1 Task 18-lite found the constraint; Task 18 removes half of it.
        //
        // THE ARENA STREAM IS UNDER GRAPH CAPTURE, and a batch has other threads
        // doing the capturing.  CudaBatchArena captures the CMFD sweep on
        // `core.stream` -- ONE stream for every slot -- and a capture swallows
        // everything enqueued on that stream by anybody.  With M Drivers,
        // worker A's segment enqueued updpsi while worker B was capturing, and
        // both died:
        //
        //   [RASBERY][FAIL] cudaMemcpyAsync(d_slot_map, ...): operation failed
        //   due to a previous error during capture
        //
        // -- measured on the 4-deck local batch, four failed decks in 1.2 s.
        //
        // So SHARING the stream stays a single-run arrangement.  What Task
        // 18-lite concluded from that -- "and therefore a batch takes the
        // blocking sweep hook, and therefore a budget of one" -- was a property
        // of the enqueue path, not of the batch: `enqueueSweeps` refused more
        // than one live instance because it took no claim on the stream and
        // staged the fleet masks into one buffer per arena.
        //
        // Both are fixed where they live (CudaBICGBackend.cu): the enqueue path
        // takes the arena's `stream_mutex` for the duration of its enqueue, so
        // two decks' launches never interleave on the host and the capture is
        // exclusive; the fleet masks got one staging lane per slot, so a second
        // launcher's host writes cannot reach the first one's in-flight DMA;
        // and a caller on a DIFFERENT stream is joined to the arena's by an
        // event pair, which is what orders a segment's updpsi against the sweep
        // that reads its psi without either of them owning the other's stream.
        //
        // So the arm is now about the STREAM only.  One deck binds the arena
        // stream and pays nothing; a batch keeps its private stream and pays
        // two events an outer.  Both get `finish_cmfd_sweep`, both leave
        // `sweep_synchronizes` false, and both therefore run the segment at its
        // configured budget.
        //
        // useStream(nullptr) IS NOT REDUNDANT.  The runner is process-lived and
        // arms many times; a previous arm may have pointed it at an arena
        // stream, and this restores its own.
        const bool solo = rasberyBatchWidth() <= 1;
        const bool have_sweep_stream = ctx.cmfd_solver.sweepStream() != nullptr;
        // Only the SOLO arm adopts the arena stream as the segment's own.
        const bool shared_stream =
            solo && have_sweep_stream &&
            gpu::rasberyOuterSegment(slot).useStream(ctx.cmfd_solver.sweepStream());
        if (!shared_stream) gpu::rasberyOuterSegment(slot).useStream(nullptr);
        // ===================================================================
        // Rev.7.1 Task 18c: A BATCH TAKES THE RENDEZVOUS BACK, BY DEFAULT
        // ===================================================================
        //
        // Task 18 made `stream_sweep` unconditional -- `have_sweep_stream`
        // alone -- on the reasoning that the only thing standing between a
        // batch and a budget of 8 was the enqueue path's refusal to serve a
        // second live instance.  That reasoning was right about the REFUSAL and
        // wrong about the COST, and the cost is the whole point of batch mode.
        //
        // WHAT THE TWO ARMS ACTUALLY ARE.  The blocking hook routes the
        // segment's CMFD sweep through CudaBatchArena::solveCommon(kind=1) --
        // the RENDEZVOUS -- which collects every deck that has reached this
        // outer and launches ONE batched sweep of width M.  That width is the
        // entire reason `--batch-mode` is faster than M processes.  The
        // stream-ordered hook routes it through enqueueSweeps, which stages ONE
        // slot (`issueSweepUploads(&m, 1, unroll)`, CudaBICGBackend.cu) onto
        // the arena's ONE stream under `stream_mutex`.  So a batch on the
        // stream-ordered arm does not run M sweeps concurrently; it runs M
        // sweeps of width 1, serialised on the host by the claim and on the
        // device by the stream.
        //
        // MEASURED, 238 / RTX PRO 6000 / M64, arm X + RASBERY_GPU_OUTER=1
        // SEGMENT_MAX=8: 224.5 c/h at Task 18-lite (rendezvous, budget 1)
        // against a run still unfinished at 2x the wall with the stream-ordered
        // arm -- at 93 % GPU utilisation, which is what 64 narrow launches back
        // to back look like.  Locally, 8 decks on a 1080 Ti reproduce the same
        // sign.
        //
        // WHY THE TRADE CANNOT BE SPLIT.  Budget > 1 means the segment runs
        // outer i+1 on the device without returning to the host, and a HOST
        // rendezvous is a host return by definition.  So the two are exclusive:
        // width M at budget 1, or width 1 at budget 8.  At M=64 the width is
        // worth far more than the budget, and it is worth more the wider the
        // batch gets -- which is the direction this campaign is going.
        //
        // WHAT IS KEPT.  Everything in Task 18 except this one choice: the
        // claim, the per-slot staging lanes and the event join stay (they are
        // what make the opt-in arm CORRECT, and they cost nothing when it is
        // off), and the batched nodal arena still honours the canonical binding
        // -- so a batch keeps `jnet_bridge_bytes` 0 on either arm.  A solo run
        // is untouched: it has no rendezvous to lose.
        //
        // THE OPT-IN EXISTS because the trade reverses at small M and on a
        // device that cannot fill itself from one deck.  RASBERY_GPU_OUTER_
        // BATCH_STREAM_SWEEP=1 restores Task 18's arm exactly.
        static const bool batch_stream_sweep = [] {
            const char* v = std::getenv("RASBERY_GPU_OUTER_BATCH_STREAM_SWEEP");
            return v != nullptr && std::string(v) != "0";
        }();
        const bool stream_sweep = have_sweep_stream && (solo || batch_stream_sweep);

        gpu::OuterSegmentHooks hooks;
        hooks.enqueue_cmfd_sweep =
            stream_sweep ? &outerSweepEnqueueHook : &outerSweepHook;
        hooks.finish_cmfd_sweep   = stream_sweep ? &outerSweepFinishHook : nullptr;
        hooks.enqueue_nodal_drive = &outerNodalHook;
        hooks.nodal_completion_event = &outerNodalCompletionHook;
        hooks.nodal_reigv_slot       = &outerNodalReigvSlotHook;
        hooks.nodal_reigv_mode       = &outerNodalReigvModeHook;
        hooks.apply_cusping       = &outerCuspingHook;
        hooks.read_live_state     = &outerLiveStateHook;
        hooks.canonical_nodal_mode     = &outerCanonicalNodalHook;
        hooks.canonical_nodal_eligible = &outerCanonicalNodalEligibleHook;
        // Rev.7.1 Task 10 part 3.  Supplied unconditionally: WHETHER a segment
        // runs host-free is the runner's per-segment decision, and it needs all
        // three of these to be able to make it.  Installing them on the
        // rendezvous arm too costs nothing -- that arm fails the
        // `not_stream_sweep` term of the ladder and never calls them.
        hooks.finish_cmfd_sweep_deferred = &outerSweepFinishDeferredHook;
        hooks.cusping_quiescent          = &outerCuspingQuiescentHook;
        hooks.nodal_halt_gate            = &outerNodalHaltGateHook;
        hooks.nodal_wait_event           = &outerNodalWaitHook;
        hooks.ctx                 = &hc;
        hooks.sweep_synchronizes  = !stream_sweep;
        gpu::rasberyOuterSegment(slot).setHooks(hooks);

        if (!gpu::rasberyBindOuterResidency(residency)) {
            // F11 (review doc Sec 3).  The second nameless refusal: the hooks
            // were installed and the bind is the LAST thing between this and an
            // armed segment, so a silent false here is the most expensive one
            // to diagnose -- everything upstream reports success.  `unbound` is
            // the enum's own word for "nobody handed the runner the
            // arena-derived views", which is precisely what just failed.
            gpu::noteOuterSegmentRefusal(gpu::OuterSegmentRefusal::Unbound);
            return false;
        }

        // ===================================================================
        // Rev.7.1 Task 18-lite: CANONICAL NODAL BINDING
        // ===================================================================
        //
        // WHAT IT REMOVES.  The nodal drive inside a segment is a host CALL, but
        // with RASBERY_GPU_NODAL_FULL it is not host ARITHMETIC -- the whole
        // drive runs on the device.  It was nevertheless reading and writing
        // Geometry::Jnet, so the runner had to bring the device jnet down and
        // push it back around every outer: 2 x nsurf*ng doubles that existed
        // only because the two device buffers had different addresses.  Adopting
        // the segment's own jnet/flux/phis as the backend's canonical set makes
        // them the same address, and the bridge has nothing left to carry.
        //
        // THE SEGMENT DOES NOT FORCE `FULL`, AND THAT IS A DECISION.  It could:
        // the flag is process-wide and the segment could flip it while it owns
        // the outer.  It does not, for two reasons.  First, forcing would make
        // RASBERY_GPU_OUTER silently change WHICH nodal solver runs, so an
        // ON-vs-OFF comparison at the same environment would stop comparing the
        // same physics -- and that comparison is the gate this whole campaign is
        // held to.  Second, the two arms are not certified equal by anything in
        // this tree: the device arm is measured clean on these decks, and
        // measured is not proven.  The binding therefore engages only when the
        // operator has already asked for the device nodal, and says so in the
        // receipt when it does.
        //
        // THE FRACTIONAL-ROD REFUSAL IS NOT DECIDED HERE, AND THAT COST A DECK.
        // Nodal::TryDriveGpu drops to the CPU body on any deck with a fractional
        // rod (rod cusping reads the host trlcff arrays, which FULL leaves
        // device-only), and the CPU body reads Geometry::Jnet -- which the
        // dropped bridge has left several outers stale.  The obvious place to
        // refuse is here, once per arm, and it is the wrong place: a rod SEARCH
        // moves the bank INSIDE SolveLoop, so a loop that armed with every rod
        // integral meets fractional ones a few outers later.  i-SMR CY02 armed
        // clean, ran 639 outers whose drive had quietly become a CPU body, and
        // converged at k_eff 1.000043 against the host's 0.999975.
        //
        // So the arm only decides whether the buffers are ADOPTED; whether the
        // binding is LIVE is asked per outer, by the runner, through
        // OuterSegmentHooks::canonical_nodal_eligible -- and an outer that says
        // no keeps its bridge and gets its ownership handed back before the
        // drive runs.
        //
        // AND THE DRIVE HAS TO HONOUR THE BINDING, NOT JUST TAKE IT.  Rev.7.1
        // Task 18-lite found that the BATCHED nodal arena accepted an adopted
        // set into its view table and then uploaded Geometry::Jnet over it on
        // every drive.  The segment, seeing the binding live, stops filling
        // Geometry::Jnet -- so the arena uploaded an array one outer stale and
        // the deck converged somewhere else (kngr3 statepoint 1: 800.33 ppm in
        // 290 outers against 770.15 in 263).  The workaround was a third term
        // here -- a static predicate that answered `will the drive that
        // consumes an adopted set honour it` -- which kept the bridge for the
        // whole run whenever the arena was engaged.
        //
        // Rev.7.1 Task 18 RETIRED THAT PREDICATE BY FIXING THE ARENA.  launchBatch
        // now addresses jnet/flux/phis through the per-slot view table and
        // consults gpu::canonicalElidesUpload/Download with the ownership each
        // slot staged, which is the same predicate the per-instance arm has
        // used since Task 7.  Both arms honour the binding, so the question has
        // one answer and does not need asking.
        const bool nodal_on_device =
            rasberyGpuNodalEnabled() && rasberyGpuNodalFullEnabled();
        bool canonical_nodal = false;
        if (nodal_on_device) {
            const gpu::CanonicalSlotBuffers set =
                gpu::rasberyOuterSegment(slot).canonicalNodalSet();
            XsReconBackend* backend = ctx.cross_sections.EnsureBackend();
            if (set.shared() && backend != nullptr && backend->available()) {
                // Re-adopting the same three pointers is a no-op inside the
                // backend (it compares before it drops the captured graph), so
                // arming once per SolveLoop entry costs nothing after the first.
                backend->adoptCanonicalBuffers(set);
                // START OUT OF SEGMENT.  Arming is not running: between here and
                // the first outer the host may still take a whole outer of its
                // own, and that outer's drive must transfer exactly as it always
                // did.  The runner raises the claim per outer.
                backend->setCanonicalNodalSegmentMode(false, false);
                canonical_nodal = backend->canonicalBuffers().shared();
            }
        }
        gpu::rasberyOuterSegment(slot).setCanonicalNodalBound(canonical_nodal);
        // ONE LINE PER PROCESS, on the first arm, because a receipt that says
        // nothing about a binding that did not engage is the reason a reader
        // cannot tell `off` from `on and refused`.
        //
        // ATOMIC, because `--batch-mode M` reaches this from M threads at once
        // and a plain bool would print the line twice or not at all.  It also
        // carries the slot now: one line per process was enough when there was
        // one deck, and in a batch the interesting failure is exactly the deck
        // whose binding did NOT engage.
        static std::atomic<bool> canonical_said[gpu::kMaxDeviceSlots] = {};
        const int                said_i =
            (slot >= 0 && slot < gpu::kMaxDeviceSlots) ? slot : 0;
        if (!canonical_said[said_i].exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr,
                         "[RASBERY][OUTER_GPU] slot=%d canonical_nodal=%d nodal_on_device=%d "
                         "rod_fallback_at_arm=%d sweep_arm=%s\n",
                         slot, canonical_nodal ? 1 : 0, nodal_on_device ? 1 : 0,
                         ctx.nodal_solver.DeviceDriveEligible() ? 0 : 1,
                         // Rev.7.1 Task 18c: WHICH SWEEP ARM THIS DECK GOT, on
                         // the line that already reports what the arm decided.
                         // A batch on `stream` has traded one rendezvous launch
                         // of width M for M launches of width 1, and that trade
                         // is the difference between 224.5 c/h at M64 and less
                         // than half of it.  Naming it here means an operator
                         // never has to infer the arm from a wall time.
                         stream_sweep ? "stream" : "rendezvous");
        }
        // The residency flag is NOT set here.  It belongs to the segment's own
        // drive() and is raised and lowered around it in outerSweepHook; a flag
        // set at arm time outlives the segment and starves the host path.
        ctx.cmfd_solver.setOuterSegmentResident(false);
        return true;
    }

    /// WHY THERE IS NO LOOP-SCOPE OWNERSHIP SETTER.
    ///
    /// Two fixes for the sticky residency flag met here.  One set it from the
    /// ARMED decision at loop scope; this tree raises and lowers it around the
    /// drive itself (outerSweepHook / outerSweepEnqueueHook +
    /// outerSweepFinishHook).  Keeping both would leave the flag TRUE while the
    /// HOST body runs inside an armed loop -- which happens on a launch failure
    /// and on every `!outer_on_device` pass -- and that is the same bug the two
    /// fixes were written for, one call later.
    ///
    /// Drive scope is the tighter of the two and it subsumes the looser: the
    /// flag says `the caller of THIS drive owns dhat and psi on the device`,
    /// which cannot outlive that caller however the loop is armed.

    // Flux-only re-convergence (CMFD/BiCGSTAB + nodal/CNCC + cusping), with every feedback
    // (search, T/H, Xe) held fixed.  Used after the search falls back to a previously observed
    // trial point so that the published k_eff belongs to the published rod position / boron.
    static void ReconvergeFlux(SolverContext& ctx, double& eigv, int max_iter,
                               double keff_tol, double flux_tol, int& total_outer) {
        double residual   = 1.0;
        double prev_inner = eigv + 1.0;
        ctx.cmfd_solver.upddtil();

        // ===================================================================
        // Rev.7.1 Task 9: DEVICE OUTER SEGMENT  (RASBERY_GPU_OUTER, default OFF)
        // ===================================================================
        //
        // THIS LOOP IS THE ELIGIBLE FLUX-RECONVERGENCE SEGMENT, and it is the
        // only one in Driver.h that is.  Every feedback is held fixed here --
        // no Xe, no T/H, no search -- so the escape set is closed and every
        // member of it has an exact host resume:
        //
        //     FluxConverged     -> return, which is `if (converged) return;`
        //     SegmentBudget     -> continue, which is the loop's own next pass
        //     RayleighFallback  -> continue; the host finishes that sweep
        //     MaterialChanged   -> unreachable (fractional rods are refused)
        //     NonFinite /
        //     NegativeFlux      -> stop delegating; the host loop is the
        //                          reference and it decides what happens next
        //
        // ------------------------------------------------------------------
        // WHY THE CALL SITE IS HERE AND NOT IN SolveLoop -- A NAMED DEVIATION
        // ------------------------------------------------------------------
        //
        // The plan's Task 9 Step 5b asks for `host_numeric_calls == 0` across
        // SolveLoop's entry/exit (the M1 half of the W3 gate), and the segment
        // BODY below is modelled on SolveLoop's outer, step for step.  The
        // delegating CALL is nevertheless in ReconvergeFlux, and the reason is
        // structural rather than an oversight.
        //
        // A SolveLoop segment exits on the outer whose decision was not a
        // requeue -- and that outer's BODY has already advanced flux, jnet and
        // d-hat, because the host's own loop also runs the body before it
        // decides.  Resuming therefore means re-entering SolveLoop's loop body
        // PAST the control ladder, at the starvation probe.  There is no
        // formulation that avoids this: the device cannot know an outer's
        // decision without running its body, and it cannot un-run one.
        // Manufacturing the entry point means hoisting flux_converged,
        // xe_interim, stall_sample, xe_starved and xe_pending out of the body
        // and guarding 120 lines of the most B0-sensitive loop in the tree.
        //
        // That restructure is exactly what Task 10's conditional WHILE removes
        // the need for: the predicate is evaluated on the device BEFORE the next
        // body runs, so the graph's exit lands where the host already is and the
        // resume is a no-op.  Step 5b is therefore a Task 10 gate in practice,
        // and doing the restructure here -- for a path that cannot execute at
        // all until the arena is reserved and both hooks exist -- would be
        // paying the risk before any of the benefit.
        //
        // WITH THE FEATURE OFF `gpu_outer_armed` is a false const and every
        // statement below folds away, so the loop is byte-for-byte the code it
        // was.  With it ON and no arena bound, the runner refuses and the
        // [RASBERY][OUTER_GPU] receipt names why.
        const bool gpu_outer_enabled = gpu::outerGpuEnabled();
        // Stage A eligibility: a deck that can cusp would have to escape with
        // MaterialChanged on outer 1, which is a segment of length one.  The
        // predicate is XSSet::ApplyRodCusping's own (XSSet.cpp:3243, 3266).
        //
        // ASKED BEFORE THE ARM, because the arm consumes it (below).
        const bool gpu_outer_rods =
            gpu_outer_enabled &&
            gpu::outerDeckHasFractionalRods(ctx.cross_sections.axial_rod_division(),
                                            ctx.geometry.nxyz() > 0
                                                ? &ctx.geometry.rod_fraction(0)
                                                : nullptr,
                                            ctx.geometry.nxyz(), EPS);
        // Link 2: hand the runner the sweep arena's buffers and install the two
        // hooks.  Done HERE rather than at stand-up because the arena slot
        // belongs to this BICGCMFD and the eigv/residual the sweep hook has to
        // publish are this loop's own locals.
        //
        // GATED ON THE PRE-ARM REFUSAL, because arming is not free and not a
        // query: it binds residency (a shared-slot kernel patch, a device
        // synchronise and two H2D seeds over live arena memory) and it adopts
        // the segment's ONE jnet/phis/flux as this backend's canonical nodal
        // set.  In `--batch-mode` every concurrent Driver adopted the same three
        // process-wide pointers, so the batched nodal drive lost its per-deck
        // buffers -- while the segment refused every outer and the receipt said
        // so.  A refusal the caller can see coming must not be armed for.  See
        // gpu::outerSegmentPreArmRefusal.
        const OuterSlotClaim gpu_outer_claim =
            gpu_outer_enabled ? outerSlotClaim(ctx) : OuterSlotClaim{};
        const bool gpu_outer_may_arm =
            gpu_outer_enabled &&
            gpu::outerSegmentPreArmRefusal(rasberyBatchWidth(), gpu_outer_rods, false,
                                           gpu_outer_claim.arena_slots,
                                           gpu_outer_claim.admitted) ==
                gpu::OuterSegmentRefusal::None;
        // WP10.7.  THE RETURN IS READ.  It was discarded here, and that is
        // how nine cases in the 238 arm-A soak came to die naming
        // `no_residency`: armOuterSegment binds residency, bindResidency
        // clears `residency_bound` on its way in (CudaOuterGraph.cu:1118) and
        // leaves its real reason in CudaOuterSegment::status(), and the
        // post-arm ladder one line down then re-derived the failure as the
        // generic "nobody handed this runner its buffers".  The ladder still
        // decides -- the throw below is unchanged and so are the counters --
        // but the step that actually failed is named first, so the receipt
        // carries the cause and not only the symptom.
        const bool gpu_outer_arm_ok =
            gpu_outer_may_arm && armOuterSegment(ctx, eigv, residual);
        if (gpu_outer_may_arm && !gpu_outer_arm_ok)
            gpufull::nameFirstFallback(gpufull::Subsystem::Outer,
                                       "Driver: outer segment arm",
                                       gpu::rasberyOuterSegment(gpu_outer_claim.slot)
                                           .status().c_str());

        // ReconvergeFlux runs no critical search by construction, so the
        // search refusal cannot apply here.
        const gpu::OuterSegmentRefusal gpu_outer_why =
            gpu_outer_enabled ? gpu::rasberyOuterSegment(gpu_outer_claim.query())
                                    .refusal(rasberyBatchWidth(), gpu_outer_rods, false,
                                             gpu_outer_claim.admitted)
                              : gpu::OuterSegmentRefusal::FeatureOff;
        bool gpu_outer_armed = (gpu_outer_why == gpu::OuterSegmentRefusal::None);
        // The decision is hoisted out of the loop, so nothing below would ever
        // record it; say it once here or the receipt cannot tell "off" from "on
        // and refused every time".
        if (gpu_outer_enabled && !gpu_outer_armed) {
            gpu::noteOuterSegmentRefusal(gpu_outer_why);
            // WP1 (plan Sec 6.3).  The segment never armed, so this whole loop
            // is the host outer body; `gpu_outer_enabled` is the arm's own
            // predicate, so a FeatureOff run never reaches here.  The reason is
            // matched against kGpuFullAllowedOuterRefusals, which no
            // OuterSegmentRefusal is on -- `batch_mode` least of all; the
            // header says why each candidate was refused.
            RASBERY_GPU_FULL_GUARD_ALLOWED(Outer, "Driver: outer segment pre-arm",
                                           gpu::outerRefusalName(gpu_outer_why));
        }

        for (int i = 0; i < max_iter; ++i) {
            if (gpu_outer_armed) {
                gpu::OuterSegmentScalars s{};
                s.slot          = gpu_outer_claim.slot;
                s.slot_admitted = gpu_outer_claim.admitted ? 1 : 0;
                s.eigv       = eigv;
                s.residual   = residual;
                s.prev_inner = prev_inner;
                s.keff_tol   = keff_tol;
                s.flux_tol   = flux_tol;
                // The stall ladder must not fire inside this loop: ReconvergeFlux
                // has no limit-cycle handling of its own, it simply runs out of
                // iterations.  Handing the device the SAME bound makes the two
                // agree -- the ladder would trip one outer past the last one this
                // loop can take.
                s.max_outer_iter = static_cast<unsigned int>(max_iter);
                // Every feedback is held fixed here; that is what this function is.
                s.xe_pending = 0;
                s.th_pending = 0;

                gpu::OuterSegmentResume seg{};
                if (gpu::rasberyOuterSegment(gpu_outer_claim.query())
                        .runSegment(s, rasberyBatchWidth(), gpu_outer_rods, false, seg)) {
                    // WP1 follow-up: THE SEGMENT IS OVER -- its stream is
                    // drained and any graph capture is closed -- so this is the
                    // first point at which the enqueue seam's DEFERRED violation
                    // can be thrown without leaving CUDA state nothing is
                    // written to clean up.  A no-op unless one was latched,
                    // which is every call with the gate off.
                    RASBERY_GPU_FULL_RAISE_PENDING();
                    eigv        = seg.eigv;
                    residual    = seg.residual;
                    prev_inner  = seg.prev_inner;
                    total_outer += static_cast<int>(seg.device_outers);
                    // CHARGE THE LOOP BOUND BEFORE ANY EXIT IS TAKEN.  `max_iter`
                    // is a hard safety bound on OUTERS, and the device just took
                    // `device_outers` of them; a path that skipped this would
                    // leave the ON arm with a larger remaining budget than the
                    // OFF arm at the same point in the solve, which is a
                    // trajectory difference even when no iterate moved.  The
                    // loop's own ++ supplies the last one.
                    i += static_cast<int>(seg.device_outers) - 1;
                    if (seg.escape ==
                        static_cast<unsigned int>(gpu::DeviceEscape::FluxConverged))
                        return;
                    if (seg.next_phase ==
                        static_cast<unsigned int>(gpu::DevicePhase::Failed)) {
                        // A non-finite iterate.  Hand the rest of this
                        // re-convergence back to the host path rather than
                        // inventing an exit ReconvergeFlux never had.
                        //
                        // AND DO NOT KEEP THE DEVICE'S prev_inner.  The three
                        // scalars adopted above came through the sweep hook,
                        // which wrote eigv and residual in place, so they are
                        // the host's own -- but prev_inner came out of a
                        // DeviceSlotState whose decision reached Failed, and on
                        // a non-finite the value that produced it is not a
                        // number this loop should carry.  The host body's own
                        // rule is `prev_inner = eigv`, so use that.
                        prev_inner      = eigv;
                        gpu_outer_armed = false;
                        // WP1 (plan Sec 6.3).  The device reached a non-finite
                        // iterate and handed the rest of the re-convergence
                        // back; under the gate that is a case failure, not a
                        // quiet change of executor.
                        RASBERY_GPU_FULL_GUARD(
                            Outer, "Driver::ReconvergeFlux",
                            "the device segment escaped with a non-finite iterate; "
                            "the host outer takes the rest of the re-convergence");
                    }
                    continue;
                }
                // Refused or failed to launch: fall through to the host outer,
                // which is the reference path and always correct.
                //
                // AND STOP DELEGATING.  Every ELIGIBILITY refusal was decided
                // once, above the loop, so a false here can only be a launch or
                // hook failure -- and one that will recur identically on the next
                // outer.  Re-arming would pay a failed launch, a warning line and
                // a refusal count per outer for the rest of the re-convergence
                // and still run the host body every time.
                // Same safe point as the delegated branch: the segment has
                // returned, so a deferred violation from one of its outers is
                // raised here, BEFORE any of this loop's own bookkeeping.
                RASBERY_GPU_FULL_RAISE_PENDING();
                gpu_outer_armed = false;
                // WP1 (plan Sec 6.3).  A launch or hook failure inside an armed
                // segment; the refusal is already counted in [OUTER_GPU], but
                // nothing made it fatal.
                RASBERY_GPU_FULL_GUARD(Outer, "Driver::ReconvergeFlux",
                                       "runSegment refused or failed to launch; the "
                                       "host outer body takes over");
            }
            ctx.cmfd_solver.updpsi(ctx.geometry.Phif());
            ctx.cmfd_solver.setls(eigv);
            ctx.cmfd_solver.drive(eigv, ctx.geometry.PhifMutable(), residual);
            ++total_outer;
            const bool converged = std::abs(prev_inner - eigv) < keff_tol && residual < flux_tol;
            prev_inner           = eigv;
            ctx.cmfd_solver.updjnet(ctx.geometry.Phif(), ctx.geometry.Jnet());
            ctx.nodal_solver.reset(1.0 / eigv, ctx.geometry.Jnet(),
                                   ctx.geometry.Phif(), ctx.geometry.Phis());
            ctx.nodal_solver.drive();
            if (ctx.cross_sections.ApplyRodCusping(eigv, ctx.nodal_solver.axialTransverseLeakage()))
                ctx.cmfd_solver.upddtil();
            ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());
            if (converged)
                return;
        }
    }

    // =====================================================================
    // Safeguarded Anderson acceleration of the in-core Xe fixed point
    // (RASBERY_XE_ANDERSON, plan Rev.4 Sec 10).
    //
    // Everything from here to TryAndersonXeStep is unreachable with the gate
    // unset: SolveLoop's single entry point short-circuits on a cached bool
    // before the call is formed.  The one cost the default path pays is
    // constructing an empty XeAndersonState -- nine empty std::vectors, no
    // allocation, no clock read, no branch inside the outer loop.
    //
    // THE FORMULA, EXACTLY AS IMPLEMENTED.  Type-II Anderson with window
    // m = XE_ANDERSON_DEPTH and mixing beta = 1 -- no extra relaxation on top,
    // because Sec 10.5 forbids damping an Anderson candidate and the damper's
    // own engagement switches the whole arm off instead.
    //
    // At step k the map has been evaluated at the current iterate x_k:
    //
    //     F_k = F(x_k)        g_k = F_k - x_k
    //
    // and the history holds the last n <= m difference columns, taken between
    // CONSECUTIVE evaluations (j = n-1 is the newest):
    //
    //     dF_j = F_{j+1} - F_j        dG_j = g_{j+1} - g_j
    //
    // Solve the small least squares in residual space,
    //
    //     min_gamma || g_k - sum_j gamma_j dG_j ||_2
    //
    // through its explicit normal equations.  For n = 2, with
    //
    //     a = <dG_0,dG_0>   b = <dG_0,dG_1>   c = <dG_1,dG_1>
    //     p = <dG_0,g_k>    q = <dG_1,g_k>    det = a*c - b*b
    //
    //     gamma_0 = (c*p - b*q) / det        gamma_1 = (a*q - b*p) / det
    //
    // and for n = 1 -- also the fallback when the two-column Gram matrix is
    // ill-conditioned -- with d the NEWEST column,
    //
    //     gamma = <d,g_k> / <d,d>
    //
    // which is the secant / Aitken extrapolation written as AA(1).  The
    // candidate is
    //
    //     x_{k+1} = F_k - sum_j gamma_j dF_j
    //
    // and the same normal equations give the model's predicted squared residual
    //
    //     ||g_pred||^2 = <g_k,g_k> - sum_j gamma_j <dG_j,g_k>
    //
    // which is what the predicted-decrease safeguard tests.  Setting every
    // gamma to zero reproduces x_{k+1} = F_k, the plain undamped Picard step,
    // so the acceleration is a strict generalization of the map it replaces and
    // never a different map.
    //
    // TWO DEVIATIONS FROM SEC 10, BOTH DELIBERATE AND BOTH DOCUMENTED HERE.
    //
    // (a) NO FULL-EXACT TRUE-RESIDUAL ACCEPTANCE.  Sec 10.4 wants a candidate
    //     accepted only after a subsequent full-exact evaluation confirms the
    //     true residual fell, and Sec 10.2/10.3 want that trial to run on a
    //     side-effect-free coupled snapshot with transactional rollback.  That
    //     is a large, separate piece of machinery -- CoupledStateSnapshot has
    //     to carry flux, fission source, eigenvalue, current, d-hat, cusping
    //     state and the XS generation -- and it is deferred to a later
    //     hardening pass.  What stands in for it here is the trust cap: a
    //     candidate may not move the inventory further than
    //     XE_ANDERSON_MAX_STEP times the Picard step it replaces, in the same
    //     raw metric, so the worst an accepted candidate can do is the shock a
    //     1.25x Picard step delivers -- and the cascade then re-converges the
    //     flux and re-measures the residual at the very next outer, which is
    //     the production convergence test doing the checking one step late.
    //
    // (b) NO AXIAL BRANCH GUARD.  Sec 10.4's |AO_candidate - AO_picard| test
    //     needs the candidate's converged axial power, i.e. exactly the trial
    //     solve (a) defers.  Until then the axial-branch property is a
    //     VALIDATION-level obligation, not a runtime one: Gate A/B compare AO
    //     against the frozen exact baseline, and the arm is not adoptable on a
    //     run whose AO moved.  The damper interlock below is the runtime half
    //     of the same concern -- when the damper engages it is SELECTING a root
    //     (see the xe_relax initializer's cy02 case), and an extrapolation
    //     built from the pre-damping map would fight that choice, so Anderson
    //     stands down for the rest of the solve and its history goes with it.
    // =====================================================================

    /// One (I-135, Xe-135, Xe-135m) field over the fuel nodes, indexed by
    /// fuel-node ORDINAL (XSSet::fuel_nodes()).  The Anderson algebra treats
    /// the three rows as one vector of length 3*n_fuel; keeping them as three
    /// arrays is what lets Evaluate/Commit take them with no repack.
    struct XeTriple {
        std::vector<double> i135;
        std::vector<double> xe135;
        std::vector<double> xe135m;
        [[nodiscard]] size_t size() const { return xe135.size(); }
        void resize(size_t n) {
            i135.resize(n);
            xe135.resize(n);
            xe135m.resize(n);
        }
    };

    /// <a,b> over the concatenated 3*n_fuel vector.
    static double XeDot(const XeTriple& a, const XeTriple& b) {
        const size_t n   = a.size();
        double       sum = 0.0;
        for (size_t i = 0; i < n; ++i)
            sum += a.i135[i] * b.i135[i] + a.xe135[i] * b.xe135[i] +
                   a.xe135m[i] * b.xe135m[i];
        return sum;
    }

    /// out = a - b over all three rows.
    static void XeSub(const XeTriple& a, const XeTriple& b, XeTriple& out) {
        const size_t n = a.size();
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            out.i135[i]   = a.i135[i] - b.i135[i];
            out.xe135[i]  = a.xe135[i] - b.xe135[i];
            out.xe135m[i] = a.xe135m[i] - b.xe135m[i];
        }
    }

    static void XeSwap(XeTriple& a, XeTriple& b) {
        a.i135.swap(b.i135);
        a.xe135.swap(b.xe135);
        a.xe135m.swap(b.xe135m);
    }

    /// The raw relative Xe-135 change of `cand` against `base`, in EXACTLY the
    /// metric UpdateEquilibriumXenon returns -- max over fuel nodes of
    /// |new - old| / max(|new|, 1e-30) -- so the trust region compares the
    /// candidate against the Picard step in one and the same measurement.
    static double XeRelativeChange(const XeTriple& cand, const XeTriple& base) {
        const size_t n     = cand.size();
        double       worst = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double scale = std::max(std::abs(cand.xe135[i]), 1.0e-30);
            worst = std::max(worst, std::abs(cand.xe135[i] - base.xe135[i]) / scale);
        }
        return worst;
    }

    /// Anderson history for ONE Xe cascade.  Declared as a SolveLoop local, so
    /// it is destroyed at every exit and no history can outlive a depletion
    /// step (Drive() runs PredictorStep/CorrectorStep BETWEEN SolveLoop calls).
    struct XeAndersonState {
        XeTriple x;      ///< x_k, the iterate the map was evaluated at
        XeTriple f;      ///< F(x_k)
        XeTriple g;      ///< g_k = F(x_k) - x_k
        XeTriple f_prev; ///< F at the previous evaluation
        XeTriple g_prev; ///< g at the previous evaluation
        XeTriple df[XE_ANDERSON_DEPTH];
        XeTriple dg[XE_ANDERSON_DEPTH];
        XeTriple cand; ///< the proposed x_{k+1}
        int      ncol      = 0;     ///< usable difference columns, newest last
        bool     have_prev = false; ///< an evaluation is on record to difference against

        /// Is there anything to discard?  So that a reset at a cascade boundary
        /// the arm never stepped in is not charged to the telemetry.
        [[nodiscard]] bool holds_history() const { return have_prev || ncol > 0; }
        /// Drop the history, keep the buffers: the next cascade has the same
        /// fuel-node count, so freeing them would be pure allocator churn.
        void forget() {
            have_prev = false;
            ncol      = 0;
        }
    };

    /// Discard the Anderson history (Sec 10.5).  The history is only meaningful
    /// while the MAP is fixed, so every event that moves the macro-XS -- a
    /// committed T/H step, a committed search trial -- invalidates it, and so
    /// does damper activation, which changes which root the cascade is heading
    /// for.  Cheap and idempotent; charges the counter only for a real discard.
    static void ResetXeAndersonHistory(SolverContext& ctx, XeAndersonState& aa) {
        if (!aa.holds_history())
            return;
        aa.forget();
        ++ctx.telemetry.xe_aa_history_resets;
        // The same event in the run-total receipt.  The DEVICE history needs no
        // separate discard: aa.ncol and aa.have_prev are the only things that
        // decide whether a column is read, they live here on the host for both
        // arms, and forget() has just cleared them -- so a device column can no
        // longer be reached until an evaluation overwrites it.  One place
        // decides, which is what keeps the two arms' reset counts identical.
        xe::xeGpuTally().reset_edges.fetch_add(1, std::memory_order_relaxed);
    }

    /// Charge one rejection and, only under RASBERY_XE_ANDERSON_DEBUG, say why.
    /// ONE counter with the reason folded in (Sec 10.5 lists the reasons
    /// separately; the receipt stays one field and this trace resolves it).
    static void RejectXeAnderson(SolverContext& ctx, const char* reason, int cols,
                                 double picard, double value) {
        ++ctx.telemetry.xe_aa_rejected;
        if (xeAndersonDebug())
            std::cerr << std::format(
                "[RASBERY][DEBUG][xe-aa] rejected ({}) cols={} picard={:.3e} value={:.3e}\n",
                reason, cols, picard, value);
    }

    /// WP7 stage C -- the same step as ONE DEVICE TRANSACTION
    /// (RASBERY_GPU_XE_TXN, default off).
    ///
    /// ---------------------------------------------------------------------
    /// WHAT IT RETURNS, AND WHY THAT IS NOT WHAT ITS SIBLING RETURNS
    /// ---------------------------------------------------------------------
    ///
    /// TryAndersonXeStepGpu returns false to mean "I wrote nothing, run the
    /// Picard step yourself".  The transaction cannot say that: it has already
    /// committed by the time anyone on the host knows whether the candidate
    /// passed, and on a rejection it commits THE IMAGE THE PICARD FALLBACK
    /// WOULD HAVE COMMITTED -- the F it already holds, undamped, with the
    /// zero-flux skip.  So `true` here means "a step was taken", accepted or
    /// not, and the caller must not take another.
    ///
    /// That is the same commit, bit for bit, and the argument is short: the
    /// fallback's UpdateEquilibriumXenon re-evaluates the map at a state
    /// NOTHING HAS WRITTEN since this evaluation, through the same kernel, so
    /// it can only reproduce the same F; and relax is 1.0 on this path because
    /// SolveLoop arms the attempt with `xe_relax == 1.0` and hands the fallback
    /// that same xe_relax.  The backend refuses any other relax rather than
    /// assume it.
    ///
    /// `false` means the transaction DECLINED -- an unavailable arm, a deck the
    /// history block was not sized for, zero power -- having written nothing,
    /// and the round-tripping arm below runs the step instead.  That path is
    /// counted (`txn_declined`) because a run where it fires is a run whose
    /// census is a mixture of two arms.
    ///
    /// EVERY COUNTER IS THE ONE THE OTHER PAIR OF ARMS CHARGES.  On a rejected
    /// step, TXN=0 charges aa_proposed here and xe_updates/device_updates in
    /// UpdateEquilibriumXenon; there is one call here instead of two, so both
    /// are charged from the downloaded reason.  A receipt that changed under a
    /// flag claiming bit-identity would be the first thing to disbelieve.
    ///
    /// ---------------------------------------------------------------------
    /// RASBERY_NEVER_INLINE, AND IT IS THE OFF ARM THAT NEEDS IT
    /// ---------------------------------------------------------------------
    ///
    /// This body has ONE call site, in TryAndersonXeStepGpu, which itself has
    /// one call site in TryAndersonXeStep, which has one in SolveLoop.  Without
    /// the attribute -finline-functions-called-once splices all ~110 lines of it
    /// into SolveLoop, where the four normal-equation expressions of the SPLIT
    /// arm (`det = a*c - b*b` and the three below it, further down this file)
    /// are compiled -- unbarriered, at -O3 -march=native with gcc's default
    /// -ffp-contract=fast.  Re-making SolveLoop's inlining and contraction
    /// decisions around a body nobody runs is exactly what a default-off flag
    /// may not do, and it is what `71092e2` did: flag-off digest
    /// 22b9a3187bfb4beb / 4566 outers became c1a5d9116df9edb3 / 4601 with
    /// RASBERY_GPU_XE_TXN unset.
    ///
    /// The shape that is PROVEN neutral on the same host is the audit hook a
    /// few hundred lines below -- a cached `static const bool audit` and, under
    /// it, one call to xe::auditAndersonFit -- and what makes it neutral is that
    /// auditAndersonFit is defined in src/XeFormAudit.cpp, so the call is
    /// opaque and nothing joins the caller (`8919331` added it and the
    /// trajectory did not move).  This attribute buys the same opacity for a
    /// body that cannot leave the header, because it reaches SolverContext,
    /// XeAndersonState and RejectXeAnderson, all of which are declared here.
    /// tools/test_xe_split_arm_sequence_contract.py pins it.
    static RASBERY_NEVER_INLINE bool TryAndersonXeStepGpuTxn(SolverContext& ctx,
                                                             XeAndersonState& aa,
                                                             double power, double max_step,
                                                             double& xe_change) {
        XSSet& xs = ctx.cross_sections;
        if (xs.fuel_nodes().empty())
            return false;

        // The window bookkeeping the device cannot do: aa is HOST state, and
        // these three numbers are exactly what steps 2 and 3 of the sibling
        // compute before it knows anything about this step's outcome.
        XsReconBackend::XeTxnRequest req;
        int                          ncol_after = aa.ncol;
        if (aa.have_prev) {
            if (aa.ncol == XE_ANDERSON_DEPTH) {
                req.hist_rotate = true;
                --ncol_after; // the oldest column falls out of the window
            }
            req.hist_col = ncol_after;
            ++ncol_after;
        }
        req.ncol     = ncol_after;
        req.eq_tol   = XE_EQUILIBRIUM_TOLERANCE;
        req.min_gram = XE_ANDERSON_MIN_GRAM;
        req.max_step = max_step;
        req.relax    = 1.0; // see the header: SolveLoop guarantees it

        xe::XeTxnControl out{};
        if (!xs.XeGpuTransaction(power, req, out)) {
            xe::xeGpuTally().txn_declined.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        aa.ncol      = ncol_after;
        aa.have_prev = true;
        xe_change    = out.picard;

        // The step happened on the device however it ended, so the split arm's
        // sum-to-whole triple is charged once, here, instead of once in
        // UpdateEquilibriumXenon.
        xe::XeGpuTally& tally = xe::xeGpuTally();
        tally.xe_updates.fetch_add(1, std::memory_order_relaxed);
        tally.device_updates.fetch_add(1, std::memory_order_relaxed);

        if (out.reason == xe::XE_TXN_NOT_ARMED)
            return true; // arming is not a rejection: neither counter moves

        ++ctx.telemetry.xe_aa_proposed;
        tally.aa_proposed.fetch_add(1, std::memory_order_relaxed);

        if (out.reason == xe::XE_TXN_ACCEPTED) {
            ++ctx.telemetry.xe_aa_accepted;
            tally.aa_accepted.fetch_add(1, std::memory_order_relaxed);
            return true;
        }

        // The same four reasons, with the same reported value, as the sibling's
        // four RejectXeAnderson call sites.
        const char*  reason = "condition";
        double       value  = out.gg;
        if (out.reason == xe::XE_TXN_RESIDUAL) {
            reason = "residual";
            value  = out.gg - out.proj;
        } else if (out.reason == xe::XE_TXN_PHYSICS) {
            reason = "physics";
            value  = out.step;
        } else if (out.reason == xe::XE_TXN_STEP) {
            reason = "step";
            value  = out.step;
        }
        RejectXeAnderson(ctx, reason, out.ncol, out.picard, value);
        return true;
    }

    /// The DEVICE arm of the safeguarded Anderson step (RASBERY_GPU_XE,
    /// Rev.7.1 Task 13).
    ///
    /// ---------------------------------------------------------------------
    /// WHY THIS IS A SIBLING AND NOT A PARAMETER OF THE HOST FUNCTION
    /// ---------------------------------------------------------------------
    ///
    /// The obvious move is to lift the two arms behind a small interface --
    /// evaluate / roll / dots / candidate / commit -- and share the safeguards.
    /// It is the wrong move HERE, and the reason is measurable: the host arm's
    /// XeDot and candidate loop are currently INLINED into the host arm, and
    /// what gcc contracts inside an inlined body is not what it contracts
    /// behind an indirect call.  Task 13's first gate is that RASBERY_GPU_XE
    /// unset reproduces the frozen baseline byte for byte, and a refactor that
    /// moves the host's arithmetic into a different function body cannot
    /// promise that.  So the host arm is not touched at all, and the price is
    /// this: the eight safeguard decisions appear twice.
    ///
    /// THEY ARE THE SAME EIGHT, IN THE SAME ORDER, WITH THE SAME CONSTANTS, and
    /// tools/test_xe_gpu_contract.py reads both functions and fails when they
    /// stop being so.  What differs -- the ONLY thing that differs -- is where
    /// the six inner products are computed:
    ///
    ///     host arm    XeDot, one serial fold over ~3*n_fuel terms
    ///     this arm    k_xe_dot_reduce, a fixed partition of the same range
    ///
    /// which associates the additions differently and is therefore a trajectory
    /// change (N1, Gate A/B).  Everything else -- the map, the window, the
    /// normal equations, the four safeguards, the acceptance semantics, the
    /// history reset edges -- is bit-for-bit the host's.
    ///
    /// THE REJECTION PATH IS THE HOST'S, NOT AN OPTIMISED ONE.  The plan's
    /// Rev.7.1 Step 5 has the device commit a damped Picard image from the F it
    /// already holds, saving one evaluation.  This returns FALSE instead and
    /// lets the caller run UpdateEquilibriumXenon, which re-evaluates the map at
    /// the same untouched state and therefore commits THE SAME IMAGE -- with
    /// the same xe_relax the plan asks for, because that call takes the
    /// damper's relax.  Identical result, on ~5 % of steps, and the caller's
    /// control flow stays the one the host arm's caller already has: the
    /// fallback is literally the production path rather than a second near-copy
    /// of it.  That is the same trade the host arm's own comment defends.
    static bool TryAndersonXeStepGpu(SolverContext& ctx, XeAndersonState& aa,
                                     double power, double max_step, double& xe_change) {
        // WP7-C.  ONE LINE, AT THE TOP, and the body below is untouched -- the
        // same shape Task 13's own dispatch takes.  With the flag unset this is
        // a load of a cached bool and the round-tripping arm runs exactly as it
        // always has, which is what makes the flag's A/B an A/B.
        //
        // "ONE LINE" IS ONLY TRUE BECAUSE THE CALLEE IS RASBERY_NEVER_INLINE.
        // It was not true when this shipped: the callee had one call site, so
        // gcc inlined its whole body here and from here into SolveLoop, and the
        // flag-off trajectory moved (22b9a3187bfb4beb/4566 -> c1a5d911.../4601).
        // See TryAndersonXeStepGpuTxn's header and
        // docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md Sec 7.
        static const bool txn = rasberyGpuXeTxnEnabled();
        if (txn && TryAndersonXeStepGpuTxn(ctx, aa, power, max_step, xe_change))
            return true;

        XSSet& xs = ctx.cross_sections;

        // 1. Evaluate the map without applying it (plan Sec 10.1) -- x, F(x)
        //    and g land in the DEVICE history and never cross the bus.
        double picard = 0.0;
        if (!xs.XeGpuEvaluate(power, picard))
            return false;
        const size_t n = xs.fuel_nodes().size();
        if (n == 0)
            return false;

        // 2. Roll the history forward BEFORE deciding anything.  A refused step
        //    still contributes its pair: the Picard step the caller falls back
        //    to advances the SAME iteration on the SAME map, so the next
        //    evaluation's difference column is only meaningful if this one was
        //    recorded.  Sec 10.5: raw, undamped (x, F(x)) pairs only.
        if (aa.have_prev) {
            if (aa.ncol == XE_ANDERSON_DEPTH) {
                if (!xs.XeGpuRotateHistory())
                    return false;
                --aa.ncol; // the oldest column falls out of the window
            }
            if (!xs.XeGpuRecordColumn(aa.ncol))
                return false;
            ++aa.ncol;
        }
        if (!xs.XeGpuSaveEvaluation())
            return false;
        aa.have_prev = true;
        // The host arm's length check has no device counterpart and needs none:
        // the history block is one allocation sized on n_fuel, which is
        // geometry-fixed, and the backend refuses every call whose n_fuel does
        // not match what the block was sized for.  A column of the wrong length
        // is unrepresentable rather than guarded against.

        // 3. Arming -- NOT a rejection, so neither counter moves.  Same two
        //    terms, same reasons, as the host arm.
        if (aa.ncol == 0 || picard < XE_EQUILIBRIUM_TOLERANCE)
            return false;

        ++ctx.telemetry.xe_aa_proposed;
        xe::xeGpuTally().aa_proposed.fetch_add(1, std::memory_order_relaxed);

        // 4. The least squares, through explicit normal equations.  THE HOST
        //    COMPUTES THESE, from six device-reduced scalars: it is eight
        //    doubles of arithmetic, it is the part a reader checks against
        //    Sec 10, and doing it here means the two arms differ in exactly one
        //    identified place (the inner products) rather than in "the algebra".
        double dots[xe::XE_DOT_COUNT] = {};
        if (!xs.XeGpuDots(aa.ncol, dots))
            return false;

        static_assert(XE_ANDERSON_DEPTH == 2,
                      "the normal equations below are written out for a two-column window; "
                      "the device dot slots (xe::XeDotSlot) are too, and a depth-3 "
                      "experiment has to widen both together");
        const double gg = dots[xe::XE_DOT_GG];
        double       gamma[XE_ANDERSON_DEPTH] = {};
        double       proj                     = 0.0; // sum_j gamma_j <dG_j, g_k>
        bool         solved                   = false;
        // ===================================================================
        // WHY THESE FOUR EXPRESSIONS ARE NO LONGER SPELLED `a * c - b * b`
        // ===================================================================
        //
        // THE INVERTED A/B, 238, 2026-08-31.  `048c6c1` marked the default-off
        // transaction arm RASBERY_NEVER_INLINE so it could not be folded into
        // SolveLoop, and the FLAG-OFF trajectory moved AGAIN -- to
        // c1a5d9116df9edb3 / 4601 outers, where the same tree with only that
        // token removed gives 22b9a3187bfb4beb / 4566, the `7cfe3a4` value.
        // The attribute did not restore the baseline; `d85984e` had landed on
        // it by luck.
        //
        // The class named in docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md §7
        // was right and its remedy was a coin toss.  These four expressions
        // were UNBARRIERED host arithmetic in a single translation unit at
        // -O3 -ffp-contract=fast with no LTO, so which of each pair's two
        // multiplies gcc folds into the add is a decision it re-makes for
        // every inlining context this block lands in.  ANY edit near SolveLoop
        // -- a new sibling, an attribute, a header, a static -- re-rolls it.
        // No attribute can fix that, because the attribute is itself one of
        // the things that re-rolls it.
        //
        // So the decision is REMOVED rather than steered.  Each site now goes
        // through xe::xeSiteSub / xe::xeSiteAdd -- the same bodies the device
        // arm runs, whose three states are all written with xsr::xsrFma and
        // xsr::xsrMul, and xsrMul carries an `asm volatile` barrier on gcc so
        // no surrounding context can re-fuse it.  Every FP operation on this
        // path is now either barriered or unfusable (a division, a compare, or
        // a multiply chain with no add), and INLINING CANNOT CHANGE THE
        // RESULT.  The arm's trajectory is a property of `host_forms` and of
        // nothing else about the call graph.
        //
        // WHICH combination reproduces 22b9a3187bfb4beb is a MEASUREMENT, not
        // a derivation: XE_HOST_FORMS_DEFAULT carries the current pin and the
        // 238 sweep over RASBERY_XE_HOST_FORMS is what set it (§8.4 of the
        // regression doc).  The mask is NOT xeFormMask(): that one is mined
        // against a quotation in another translation unit and answers the
        // DEVICE's question -- src/XeFormAudit.h is the whole argument for why
        // the two are separate numbers.
        //
        // Read once, cached, exactly like the `audit` bool below: a getenv per
        // Anderson step would be both a cost and a second opinion.
        static const unsigned long long host_forms = xe::xeHostFormMask();
        if (aa.ncol == XE_ANDERSON_DEPTH) {
            const double a   = dots[xe::XE_DOT_A];
            const double b   = dots[xe::XE_DOT_B];
            const double c   = dots[xe::XE_DOT_C];
            const double p   = dots[xe::XE_DOT_P];
            const double q   = dots[xe::XE_DOT_Q];
            const double det = xe::xeSiteSub(
                a, c, b, b, xe::xeSiteState(host_forms, xe::XE_TXN_DET_BIT));
            // NOT A SITE: a multiply chain with no add cannot be contracted on
            // either compiler, and the device arm spells it the same way.
            if (a > 0.0 && c > 0.0 && std::isfinite(det) && std::isfinite(p) &&
                std::isfinite(q) && det > XE_ANDERSON_MIN_GRAM * a * c) {
                gamma[0] = xe::xeSiteSub(
                               c, p, b, q,
                               xe::xeSiteState(host_forms, xe::XE_TXN_G0_BIT)) /
                           det;
                gamma[1] = xe::xeSiteSub(
                               a, q, b, p,
                               xe::xeSiteState(host_forms, xe::XE_TXN_G1_BIT)) /
                           det;
                proj     = xe::xeSiteAdd(
                    gamma[0], p, gamma[1], q,
                    xe::xeSiteState(host_forms, xe::XE_TXN_PROJ_BIT));
                solved   = true;
            }
        }
        if (!solved) {
            // The newest column.  At ncol == 2 that is dg[1] -- slots C and Q --
            // and at ncol == 1 it is dg[0], which is slots A and P; the backend
            // fills exactly those for a one-column window.
            const int    j = aa.ncol - 1;
            const double a = (j == 1) ? dots[xe::XE_DOT_C] : dots[xe::XE_DOT_A];
            const double p = (j == 1) ? dots[xe::XE_DOT_Q] : dots[xe::XE_DOT_P];
            if (a > 0.0 && std::isfinite(a) && std::isfinite(p) &&
                a > XE_ANDERSON_MIN_GRAM * gg) {
                for (double& gj : gamma)
                    gj = 0.0;
                gamma[j] = p / a;
                // ONE multiply, no add -- so no site, and none is mined for it.
                // It is still BARRIERED, and that is not decoration: `pred2`
                // below is `gg - proj`, and an unbarriered product feeding a
                // subtraction is precisely the shape -ffp-contract=fast fuses.
                // xsrMul is what stops `gg - gamma[j] * p` from becoming an
                // fnma in some inlining contexts and not others.  The device
                // arm spells this same line xsr::xsrMul(gj, p).
                proj     = xsrecon::xsrMul(gamma[j], p);
                solved   = true;
            }
        }
        // WP7-C.  THE ONE MEASUREMENT THE MINING CANNOT MAKE, and it is made
        // here because HERE is the call site whose codegen is in question.
        //
        // The eight doubles above are what RASBERY_GPU_XE_TXN=1 moves onto the
        // device, where they are re-spelled with explicit fma/mul under the
        // mined XE_FORMS mask.  Whether that re-spelling is the same bits is a
        // property of what g++ did to THESE FOUR EXPRESSIONS, INLINED HERE --
        // and XeFormMine.h can only score a quotation of them, in another
        // translation unit, with different operand provenance.  Host 181
        // (2026-08-30) is the case where the two part company: the mask mined
        // clean (0xd3d, no [WARN][FORMS], every site decisive) and the
        // full-deck TXN A/B still diverged.  So the audit compares the shipped
        // body against this block, on this run's own operands, and the
        // [RASBERY][XE_GPU] receipt carries the count.
        //
        // OFF BY DEFAULT AND OUT OF LINE.  `audit` is a cached bool and
        // xe::auditAndersonFit lives in its own translation unit
        // (src/XeFormAudit.h says why an inline one would be a false negative),
        // so with the knob unset this is one predicted branch and the arm is
        // byte-for-byte the arm it was.
        {
            static const bool audit = xe::xeFormAuditEnabled();
            if (audit)
                xe::auditAndersonFit(dots, aa.ncol, XE_ANDERSON_MIN_GRAM, solved,
                                     gamma[0], gamma[1], proj);
        }
        if (!solved) {
            RejectXeAnderson(ctx, "condition", aa.ncol, picard, gg);
            return false;
        }

        // SAFEGUARD 2/4: the fit must predict a residual DECREASE.
        const double pred2 = gg - proj;
        if (!(std::isfinite(pred2) && pred2 >= 0.0 && pred2 < gg)) {
            RejectXeAnderson(ctx, "residual", aa.ncol, picard, pred2);
            return false;
        }

        // x_{k+1} = F_k - sum_j gamma_j dF_j, and with it SAFEGUARD 3/4 (every
        // density finite and non-negative) and the trust-region metric, all in
        // one device pass over the fuel nodes.  Two scalars come back.
        double step        = 0.0;
        bool   physics_ok  = false;
        if (!xs.XeGpuCandidate(gamma, aa.ncol, step, physics_ok))
            return false;
        if (!physics_ok) {
            // The host arm names the offending Xe-135 density here; the device
            // arm knows only that one node failed, and the step is what it can
            // honestly report.  The counter and the reason are the same.
            RejectXeAnderson(ctx, "physics", aa.ncol, picard, step);
            return false;
        }

        // SAFEGUARD 4/4: the trust region.  Written as !(<=) so a NaN rejects.
        if (!(step <= max_step * picard)) {
            RejectXeAnderson(ctx, "step", aa.ncol, picard, step);
            return false;
        }

        // Accepted.  The commit writes exactly the three Xe-chain rows,
        // reconstructs the fuel nodes and downloads both back into the host
        // arrays -- so the host is authoritative again and, exactly like the
        // fused device arm, the host-state generation is NOT bumped.
        if (!xs.XeGpuCommitCandidate(power))
            return false;
        ++ctx.telemetry.xe_aa_accepted;
        {
            xe::XeGpuTally& t = xe::xeGpuTally();
            t.aa_accepted.fetch_add(1, std::memory_order_relaxed);
            t.xe_updates.fetch_add(1, std::memory_order_relaxed);
            t.device_updates.fetch_add(1, std::memory_order_relaxed);
        }
        xe_change = picard;
        return true;
    }

    /// One safeguarded Anderson step on the Xe fixed point.
    ///
    /// Returns TRUE only when a candidate passed every safeguard and was
    /// committed; `xe_change` then carries the RAW residual measured at x_k,
    /// which is the same number UpdateEquilibriumXenon would have returned, so
    /// every downstream consumer of it (the convergence test, the damper's
    /// contraction streak, the trace line) reads exactly what it always read.
    ///
    /// Returns FALSE having written NOTHING -- no _iden row, no reconstruction,
    /// no generation bump.  The caller then runs the production Picard step on
    /// a solver state that is byte-for-byte what that step expects.  A refusal
    /// costs one extra HOST map evaluation and no flux solve; buying the
    /// property that the fallback is literally the production path, rather than
    /// a second near-copy of it, is worth that on a cascade whose every step
    /// otherwise pays a full flux re-convergence.
    static bool TryAndersonXeStep(SolverContext& ctx, XeAndersonState& aa, double power,
                                  double max_step, double& xe_change) {
        // Rev.7.1 Task 13.  ONE LINE, AT THE TOP, and the body below is
        // untouched -- the same shape the fused equilibrium-Xe entry point in
        // XSSet.cpp gives its device branch, and for the same reason: with
        // RASBERY_GPU_XE unset this is a load of a cached bool and the host arm
        // runs exactly the expressions it always ran, in the same function,
        // with the same inlining.  A restructuring that hoisted the two arms
        // behind a common interface would move the host's arithmetic into a
        // different function body, and a bit-identity claim does not survive
        // that.
        if (rasberyGpuXeEnabled())
            return TryAndersonXeStepGpu(ctx, aa, power, max_step, xe_change);

        XSSet& xs = ctx.cross_sections;

        // 1. Evaluate the map without applying it (plan Sec 10.1).
        xs.SnapshotXenon(aa.x.i135, aa.x.xe135, aa.x.xe135m);
        const double picard =
            xs.EvaluateEquilibriumXenon(power, aa.f.i135, aa.f.xe135, aa.f.xe135m);
        const size_t n = aa.x.size();
        if (n == 0 || aa.f.size() != n)
            return false;
        XeSub(aa.f, aa.x, aa.g);

        // 2. Roll the history forward BEFORE deciding anything.  A refused step
        //    still contributes its pair: the Picard step the caller falls back
        //    to advances the SAME iteration on the SAME map, so the next
        //    evaluation's difference column is only meaningful if this one was
        //    recorded.  Sec 10.5: raw, undamped (x, F(x)) pairs only.
        if (aa.have_prev) {
            if (aa.ncol == XE_ANDERSON_DEPTH) {
                for (int j = 0; j + 1 < XE_ANDERSON_DEPTH; ++j) {
                    XeSwap(aa.df[j], aa.df[j + 1]);
                    XeSwap(aa.dg[j], aa.dg[j + 1]);
                }
                --aa.ncol; // the oldest column falls out of the window
            }
            XeSub(aa.f, aa.f_prev, aa.df[aa.ncol]);
            XeSub(aa.g, aa.g_prev, aa.dg[aa.ncol]);
            ++aa.ncol;
        }
        aa.f_prev    = aa.f;
        aa.g_prev    = aa.g;
        aa.have_prev = true;

        // The fuel-node count is geometry-fixed for the whole run, so a column
        // of the wrong length can only be a bug -- but the dot products below
        // index one vector by the other's length, so start the window over
        // rather than run off the end of it.
        for (int j = 0; j < aa.ncol; ++j)
            if (aa.dg[j].size() != n || aa.df[j].size() != n) {
                aa.forget();
                return false;
            }

        // 3. Arming -- NOT a rejection, so neither counter moves.  There is
        //    nothing to extrapolate from on a cascade's first step; and below
        //    the convergence tolerance the cascade is finished, so an
        //    extrapolation buys no steps and would put a state the production
        //    test never measured into the published inventory.  With this test
        //    in place, what a cascade PUBLISHES is always a plain Picard image.
        if (aa.ncol == 0 || picard < XE_EQUILIBRIUM_TOLERANCE)
            return false;

        ++ctx.telemetry.xe_aa_proposed;

        // 4. The least squares, through explicit normal equations.
        //    SAFEGUARD 1/4: conditioning.  A two-column Gram matrix within
        //    ~1e-4 rad of singular drops to the newest column alone; a
        //    one-column fit whose residual barely moved is refused outright,
        //    because gamma = <d,g>/<d,d> is then an arbitrarily long lever.
        static_assert(XE_ANDERSON_DEPTH == 2,
                      "the normal equations below are written out for a two-column window; "
                      "Sec 10.5 allows trying depth 3 after the Gate, and that needs a "
                      "general small least-squares solve, not just a wider array");
        const double gg = XeDot(aa.g, aa.g);
        double       gamma[XE_ANDERSON_DEPTH] = {};
        double       proj                     = 0.0; // sum_j gamma_j <dG_j, g_k>
        bool         solved                   = false;
        if (aa.ncol == XE_ANDERSON_DEPTH) {
            const double a   = XeDot(aa.dg[0], aa.dg[0]);
            const double b   = XeDot(aa.dg[0], aa.dg[1]);
            const double c   = XeDot(aa.dg[1], aa.dg[1]);
            const double p   = XeDot(aa.dg[0], aa.g);
            const double q   = XeDot(aa.dg[1], aa.g);
            const double det = a * c - b * b;
            if (a > 0.0 && c > 0.0 && std::isfinite(det) && std::isfinite(p) &&
                std::isfinite(q) && det > XE_ANDERSON_MIN_GRAM * a * c) {
                gamma[0] = (c * p - b * q) / det;
                gamma[1] = (a * q - b * p) / det;
                proj     = gamma[0] * p + gamma[1] * q;
                solved   = true;
            }
        }
        if (!solved) {
            const int    j = aa.ncol - 1; // the newest column
            const double a = XeDot(aa.dg[j], aa.dg[j]);
            const double p = XeDot(aa.dg[j], aa.g);
            if (a > 0.0 && std::isfinite(a) && std::isfinite(p) &&
                a > XE_ANDERSON_MIN_GRAM * gg) {
                for (double& gj : gamma)
                    gj = 0.0;
                gamma[j] = p / a;
                proj     = gamma[j] * p;
                solved   = true;
            }
        }
        if (!solved) {
            RejectXeAnderson(ctx, "condition", aa.ncol, picard, gg);
            return false;
        }

        // SAFEGUARD 2/4: the fit must predict a residual DECREASE.  With an
        // exact least squares this holds unless gamma is zero, but the
        // one-column fallback and finite precision can both break it, and a fit
        // that predicts no progress has no business moving the iterate.
        const double pred2 = gg - proj;
        if (!(std::isfinite(pred2) && pred2 >= 0.0 && pred2 < gg)) {
            RejectXeAnderson(ctx, "residual", aa.ncol, picard, pred2);
            return false;
        }

        // x_{k+1} = F_k - sum_j gamma_j dF_j.
        aa.cand.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double vi = aa.f.i135[i];
            double vx = aa.f.xe135[i];
            double vm = aa.f.xe135m[i];
            for (int j = 0; j < aa.ncol; ++j) {
                vi -= gamma[j] * aa.df[j].i135[i];
                vx -= gamma[j] * aa.df[j].xe135[i];
                vm -= gamma[j] * aa.df[j].xe135m[i];
            }
            aa.cand.i135[i]   = vi;
            aa.cand.xe135[i]  = vx;
            aa.cand.xe135m[i] = vm;
        }

        // SAFEGUARD 3/4: physics.  Every density finite and non-negative.  A
        // negative Xe-135 is not a slightly wrong inventory -- it is a negative
        // absorption cross section handed to the flux solve.
        for (size_t i = 0; i < n; ++i) {
            const double vi = aa.cand.i135[i];
            const double vx = aa.cand.xe135[i];
            const double vm = aa.cand.xe135m[i];
            if (!(std::isfinite(vi) && std::isfinite(vx) && std::isfinite(vm)) ||
                vi < 0.0 || vx < 0.0 || vm < 0.0) {
                RejectXeAnderson(ctx, "physics", aa.ncol, picard, vx);
                return false;
            }
        }

        // SAFEGUARD 4/4: the trust region.  The candidate may not move the
        // inventory further than max_step times the Picard step it replaces,
        // measured the same way.  This is what stands in for the deferred
        // full-exact true-residual acceptance (see the deviation note above):
        // it bounds the shock the converged flux is asked to absorb by a factor
        // of a step it already survives.  Written as !(<=) so a NaN rejects.
        const double step = XeRelativeChange(aa.cand, aa.x);
        if (!(step <= max_step * picard)) {
            RejectXeAnderson(ctx, "step", aa.ncol, picard, step);
            return false;
        }

        // Accepted.  Commit writes exactly the three Xe-chain rows, reconstructs
        // the fuel nodes and bumps the host-state generation, which is what the
        // device arms re-upload on.
        xs.CommitXenon(aa.cand.i135, aa.cand.xe135, aa.cand.xe135m);
        ++ctx.telemetry.xe_aa_accepted;
        xe_change = picard;
        return true;
    }

    // Frozen-Xe startup guard (RASBERY_XE_MODE=frozen), one line, once per run.
    //
    // Frozen mode publishes whatever Xe-135 the deck handed over.  A restart or
    // a shuffle hands over a real, once-equilibrated inventory, so the mode does
    // what it says.  A run that starts from the library hands over no in-core
    // inventory at all -- the fresh-fuel depletion point of an XSLIB carries no
    // Xe worth the name -- and a deck that asked for equilibrium Xe then runs
    // its first statepoint (and, until depletion makes some, the ones after it)
    // effectively Xe-free.  That is a misconfigured run far more often than it
    // is an experiment, and it is invisible in the results: a Xe-free core
    // simply reads a few hundred pcm reactive (measured: ~250 ppm of critical
    // boron against MASTER on a fresh core).
    //
    // TWO THINGS WENT WRONG IN THE FIRST VERSION, and both are fixed here.
    //
    // (1) WHAT IT TESTED.  `peak_xe == 0.0` is a bit-exact equality on the MAX
    //     over every fuel node, so one node holding any nonzero value at all --
    //     including the trace the library's own fresh-fuel point hands over --
    //     suppressed the warning for the whole core.  That is exactly the case
    //     the guard exists for, and it is the case that got past it.  The
    //     emptiness test is now structural instead: a run that did not inherit
    //     state from another run has, by construction, never equilibrated Xe
    //     against a core flux, whatever number happens to sit in the row.  The
    //     bit-zero test is kept as the second arm, for an inherited inventory
    //     that really is empty, and the measured peak is printed either way so
    //     the operator can see which arm fired.
    //
    // (2) WHEN IT LOOKED.  It was called from SolveLoop and latched on the
    //     FIRST call, whatever that call happened to be.  On a deck whose first
    //     schedule entry is a depletion step, the first SolveLoop runs AFTER
    //     PredictorStep, and PredictorStep's own Deplete -> ApplyXeEquilibrium
    //     has already written the Xe-chain rows at the depletion rates.  The
    //     guard then measured a full inventory, burned its one-shot latch on
    //     it, and never looked again.  It now runs once in Drive(), straight
    //     after InitXS, which is the only moment the "incoming" inventory means
    //     anything: the library values are in, the restart/shuffle values are
    //     in, and no schedule entry has touched them yet.
    //
    // Deliberately a warning and not a refusal: a zero-Xe frozen run is a
    // legitimate, if unusual, thing to ask for, and the mode switch is supposed
    // to be reversible without also being opinionated.
    //
    // FROZEN ONLY, by design.  RASBERY_XE_MODE=once computes the equilibrium
    // itself at every segment, so it cannot inherit an empty inventory and has
    // nothing to warn about; xeFrozen() is false there and this returns.
    static void WarnFrozenXeIfEmpty(SolverContext& ctx, const Schedule& schedule,
                                    bool inherited_inventory) {
        if (!xeFrozen() || schedule.xenon_transient || ctx.xe_frozen_checked)
            return;
        ctx.xe_frozen_checked = true;
        const int nxyz       = ctx.geometry.nxyz();
        int       fuel_nodes = 0;
        double    peak_xe    = 0.0;
        for (int l = 0; l < nxyz; ++l) {
            if (!ctx.geometry.IsFuel(l))
                continue;
            ++fuel_nodes;
            peak_xe = std::max(
                peak_xe,
                std::abs(ctx.cross_sections.iden(Chiffon::Isotope::iXe135, l)));
        }
        if (fuel_nodes == 0)
            return;
        const bool empty_inventory = (peak_xe == 0.0);
        if (!empty_inventory && inherited_inventory)
            return;
        std::cerr << std::format(
            "[RASBERY][WARN][xe] RASBERY_XE_MODE=frozen with no in-core Xe-135 to "
            "hold: {} over {} fuel nodes (peak incoming Xe-135 {:.3e} /barn-cm). "
            "The in-core equilibrium iteration is off, so nothing will equilibrate "
            "the Xe chain against this core's flux and the run stays effectively "
            "Xe-free until depletion makes some\n",
            empty_inventory ? "the incoming inventory is identically zero"
                            : "this run starts from the library (no restart, no "
                              "shuffle), so no inventory was ever equilibrated",
            fuel_nodes, peak_xe);
    }

    // Single-loop eigenvalue solve following PARCS Fig 10.1. One outer iteration performs a
    // CMFD/BiCGSTAB flux update (Wielandt inside drive), then the nodal correction + CNCC
    // (d-hat) + rod cusping, then — once the flux is meaningful — one damped critical-search
    // step and one T/H feedback update. The sub-problems co-converge and are checked together
    // ("All Converged?"), instead of fully converging each in a nested inner loop.
    // primeXeDamping: start the equilibrium-Xe iteration already damped instead of
    // waiting for XE_OSCILLATION_STREAK non-contractions.  See the xe_relax initializer.
    static void SolveLoop(SolverContext& ctx, double& eigv, Schedule& schedule,
                          int& total_outer, int& total_th, bool keepSearch = false,
                          bool primeXeDamping = false) {
        const double power_fraction = schedule.powerFraction();
        const bool   has_th         = schedule.usesTHFeedback();
        const bool   has_search     = schedule.hasCriticalSearch();
        // RASBERY_XE_MODE=frozen retires the in-core Xe<->flux fixed point here
        // and nowhere else.  has_eq_xe is the single gate the whole machinery
        // hangs off -- the cascade counters, xe_pending and its interim probe,
        // the starvation charge, the UpdateEquilibriumXenon call itself, and
        // the cascade re-arm after a committed perturbation -- so clearing it
        // bypasses all of them at once, with no second branch that could ever
        // disagree with this one, and no reachable path left to the equilibrium
        // update.  xeFrozen() is a cached read of an env var resolved once per
        // process, so an unset variable leaves this expression exactly what it
        // was: !schedule.xenon_transient.
        const bool   has_eq_xe      = !schedule.xenon_transient && !xeFrozen();
        // RASBERY_XE_MODE=once keeps has_eq_xe -- it needs the cascade
        // counters, the step itself and the cascade re-arm -- and bounds each
        // cascade instead: at most XE_ONCE_MAX_STEPS steps, and the extras
        // damped, so no single step can hand the converged flux a shock bigger
        // than the trust region.  This is the only place the mode is read
        // (xeOnce() is a cached lookup of an env var resolved once per
        // process), and every once-specific term below short-circuits on it, so
        // an unset variable leaves all of them exactly what they were.
        const bool   xe_once_mode   = xeOnce();
        // Safeguarded Anderson acceleration of the SAME cascade
        // (RASBERY_XE_ANDERSON, plan Rev.4 Sec 10; see the comment block above
        // TryAndersonXeStep for the formula and the two deviations).  Read once
        // here, exactly like the two mode terms beside it, and decided before
        // anything in the solve moves.  xeAnderson() is a cached lookup of an
        // env var resolved once per process and already refuses every mode but
        // equilibrium, so an unset variable leaves this false and the single
        // consumer below short-circuits to the pre-Anderson expression.
        const bool   xe_anderson    = xeAnderson();
        // Optional multi-fidelity GA screening mode.  A positive value limits
        // the expensive Xe and T/H fixed-point feedback passes, while the
        // default (0/unset) preserves the production solver exactly.  Screened
        // candidates must be rerun through the unlimited exact path before
        // acceptance.
        static const int ga_feedback_passes = [] {
            const char* value = std::getenv("RASBERY_GA_FEEDBACK_PASSES");
            return value == nullptr ? 0 : std::max(0, std::atoi(value));
        }();
        const bool   trace_search   = (std::getenv("RASBERY_SEARCH_TRACE") != nullptr);
        const bool   trace_sl       = (std::getenv("RASBERY_SL_TRACE") != nullptr);
        const int    sl_outer0      = total_outer;
        const int    sl_th0         = total_th;

        if (has_search) {
            if (keepSearch) {
                // Keep the trial point and the carried secant slope, but discard the trial
                // history: it was measured before the material state moved.
                schedule.ResetSearchTrials();
            } else {
                schedule.ResetSearchState();
            }
            // First entry into this schedule step: pick an initial guess and apply it.
            if (!schedule.search_initialized) {
                // Charged to search_apply like every other trial: it IS the
                // first trial point, and leaving it out would make the
                // per-trial cost of a statepoint look cheaper by one.
                outer_timing::Scope apply_scope(sptelem::PH_SEARCH_APPLY);
                schedule.StartCriticalSearch(ctx.search_memory, ctx.geometry.bppm(0),
                                             ctx.cross_sections.rod_max_step(),
                                             ctx.search_policy, ctx.search_carry,
                                             ctx.efpd);
                if (schedule.searchType == SearchType::BORON)
                    ctx.cross_sections.SetBoron(schedule.search_current_x);
                else {
                    ctx.cross_sections.SetRod(schedule.search_current_x);
                    schedule.rod_step = schedule.search_current_x;
                }
            }
        }

        // Rod-crit search over a cusping-enabled core: the fractional fine-cell stencil smooths
        // the keff(rod) staircase but residual kinks at fine-cell boundaries leave a ~3e-5 keff
        // noise floor. A 1e-5 search tolerance sits inside that noise, so the secant bounces
        // (8-13 trials/step). Raise the floor above the noise so it converges in ~half as many.
        if (has_search && schedule.searchType == SearchType::RODCRIT)
            schedule.rodcrit_search_floor = (ctx.cross_sections.axial_rod_division() > 0) ? 5.0e-5 : 0.0;

        const double keff_tol   = schedule.tolerance_keff;
        const double search_tol = schedule.criticalSearchTolerance();
        const double flux_tol   = std::max(keff_tol, CMFD_FLUX_L2_TOLERANCE);

        // =================================================================
        // A2: STAGED TOLERANCE -- loose while the statepoint is still moving,
        // production tolerance for the answer it publishes.  Default OFF.
        // =================================================================
        //
        // WHAT THE PROFILE SAYS (tools/outer_profile.py on kngr_238, S2 +
        // device outer segment, Anderson Xe on): 12,017 outers over 35
        // statepoints, of which 8,579 (71.4 %) are charged to the
        // equilibrium-Xe cascade and 1,508 (12.5 %) to the critical search.
        // The structure behind those two numbers is one thing, not two: 228 Xe
        // cascades, 10.65 settled Xe steps each, 3.53 outers of flux
        // re-convergence per step -- and a cascade is re-armed by EVERY
        // committed search trial and T/H update.  So a single boron trial point
        // costs ~38 outers of Xe re-equilibration, and 137 of them are taken per
        // run.  MASTER needs ~59 outers per statepoint in total.
        //
        // WHY THE TOLERANCES ARE THE LEVER.  Every one of those 8,579 outers is
        // converging the flux to |dk| < 1e-6 (1 pcm) and L2 < 1e-6 so that a Xe
        // step can be taken, and re-converging the Xe inventory to a 1e-6
        // relative change, at a TRIAL BORON CONCENTRATION THAT IS ABOUT TO BE
        // THROWN AWAY.  The consumer of a trial point is the secant search,
        // whose own tolerance is 2e-5 -- twenty times looser.  Nothing needs
        // that precision until the point is accepted.
        //
        // THE SHAPE OF THE FIX, AND WHY IT CANNOT SILENTLY LOSE ACCURACY.  The
        // loop runs in one of two stages.  In the LOOSE stage the flux and Xe
        // tolerances are multiplied out; when every feedback reports converged
        // there, that is NOT an exit -- it is the trigger to switch to the
        // POLISH stage, where the production tolerances are restored, the Xe
        // cascade is re-armed against the production-converged flux, and the
        // SAME convergence test is asked again.  Only a convergence reached at
        // production tolerance can end the solve.  If the loose stage put the
        // search on the wrong boron, the polish stage sees |dk| > search_tol,
        // commits another trial, and drops back to loose -- the loop
        // self-corrects rather than publishing the loose answer.  What the
        // feature can cost is outers (a thrash between the stages); what it
        // cannot do is publish a state that never met the production tolerance.
        //
        // WHY IT IS NOT THE INTERIM-Xe KNOB.  RASBERY_XE_INTERIM_L2 fires the Xe
        // step on an UNCONVERGED flux, which is not a point of the composite map
        // Anderson extrapolates, so every interim step is a plain Picard step
        // taken outside the history.  Measured on this deck it costs rather than
        // saves: 12,017 -> 14,332 outers at 1e-4 and 17,755 at 1e-5, with
        // Anderson proposals collapsing 1,472 -> 664.  Staging keeps every Xe
        // step on a CONVERGED flux -- converged to a looser tolerance, but
        // consistently so, which is still a fixed point of a well-defined map --
        // so the Anderson history stays valid and stays engaged.
        //
        // WP10.3: PER CASE, NOT PER PROCESS.  These two were `static const`
        // lambdas reading RASBERY_STAGED_FLUX_TOL / _XE_TOL, so the first case
        // a process ran latched the convergence policy for every case after it.
        // The evaluator (EvaluatorServer.h) answers waves whose cases may
        // legitimately differ -- a screening population beside a promoted elite
        // -- and there the latch is not a cache, it is sixty-three cases running
        // a policy nobody asked them for under receipts that claim otherwise.
        // ctx.fidelity is default-constructed FROM THE ENVIRONMENT
        // (processCaseFidelity()), so a Driver nobody configured reads exactly
        // the same two numbers this code read before, once per process, from a
        // static that is now one level up.
        const double staged_flux_mult = ctx.fidelity.staged_flux_mult;
        const double staged_xe_mult   = ctx.fidelity.staged_xe_mult;
        const bool   staged_tol       = ctx.fidelity.staged();
        // THE SEARCH SAMPLE'S OWN FLOOR.  A secant search reading k_eff off a
        // flux converged no better than its own tolerance samples noise, and the
        // rod-crit case (rodcrit_search_floor, above) is this code base's
        // existing record of what that costs: the search bounces and spends more
        // outers than the loosening saved.  So the loose keff tolerance is
        // capped a factor STAGED_SEARCH_MARGIN below search_tol whenever a
        // search is running -- the loosening is allowed to be large where
        // nothing reads the digits and is not allowed to reach the digits the
        // search reads.  With no search there is no such consumer and the
        // multiplier stands as given.
        constexpr double STAGED_SEARCH_MARGIN = 4.0;
        // WP9-D stage D, candidate D3 (gate A2).  The margin was a literal
        // nobody could sweep, and it is the one number that decides how much of
        // the loosening a search TRIAL is allowed to keep.  The knob may only
        // replace it: unset, `stagedMargin` answers with the built-in and the
        // expression below is the one this tree had, to the bit.  It is inert
        // without staging, because with a single stage `polishing` is true
        // throughout and no loose tolerance is ever read.
        const double staged_search_margin =
            ctx.search_policy.stagedMargin(STAGED_SEARCH_MARGIN);
        const double loose_keff_tol =
            has_search ? std::min(keff_tol * staged_flux_mult, search_tol / staged_search_margin)
                       : keff_tol * staged_flux_mult;
        const double loose_flux_tol = std::max(loose_keff_tol, flux_tol * staged_flux_mult);
        const double loose_xe_tol   = XE_EQUILIBRIUM_TOLERANCE * staged_xe_mult;
        // FALSE means the loose stage is in force.  Latched true the first time
        // every feedback agrees at the loose tolerance, and cleared again by any
        // committed perturbation, because a perturbation invalidates the
        // agreement that set it.  With the feature off it is true from the start
        // and never read, so every tolerance below is the production one and the
        // convergence break is exactly the pre-existing one.
        bool polishing = !staged_tol;
        // Telemetry only: how many times the loose stage's agreement failed to
        // survive the polish pass.  A candidate whose thrash count is comparable
        // to its search-trial count is one whose loosening is too aggressive.
        int  polish_relapses = 0;

        ctx.cmfd_solver.resetIteration();
        ctx.cmfd_solver.upddtil();
        double residual   = 1.0;
        double prev_inner = eigv + 1.0;
        double th_dop     = 1.0; // last Doppler-temperature change from UpdateTH
        int    th_count   = 0;
        int    xe_count   = 0;
        int    flux_stall   = 0;
        int    stall_events = 0;
        // Consecutive settled iterations since the last state perturbation (search commit,
        // T/H update, Xe step).  Gates the search sampling only; see SEARCH_SETTLE_ITERS.
        int    clean_iters  = 0;
        // Equilibrium-Xe damping, engaged only when the fixed-point iteration stops
        // contracting.  Starts undamped so a well-behaved solve keeps its exact old path.
        //
        // Except on the first statepoint of a restart+shuffle cycle (primeXeDamping),
        // where waiting is not safe.  There the Xe field starts at ~0 (the pool cooling
        // has decayed it) and the Xe<->flux map is not merely slow but has MORE THAN ONE
        // fixed point: on APR1400 cy02 the undamped iteration limit-cycles with GROWING
        // amplitude (rel 0.185 -> 0.233 -> 0.271 -> ... -> 0.389, eigv oscillating with
        // period 2) and then settles, after damping finally engages, on an axial mode
        // that is upside down -- AO +0.314 where MASTER has -0.054, equilibrium Xe 12.6 %
        // short, critical boron 65.5 ppm high.  Priming the damping instead lands the same
        // solve on the physical branch: AO -0.004, Xe within 0.3 %, boron +7.3 ppm.
        //
        // The physical branch is identifiable independently of either code.  At hot full
        // power with a near-symmetric axial burnup the coolant is denser at the bottom, so
        // moderation is stronger there and the power must skew DOWN; MASTER's AO is -0.054
        // and, unlike ours, is continuous across the restart (-0.0543 -> -0.0537 -> ...).
        // A top-peaked first statepoint contradicts the density gradient, so this is not a
        // matter of matching the reference -- the undamped path was landing on the wrong
        // root.  Damping cannot move a fixed point (it only rescales the applied step), so
        // this changes which root is reached, never where the roots are.
        // Priming does not damp from the start; it makes the existing trigger a hair
        // trigger.  The iteration still begins undamped and still has to FAIL to
        // contract before anything happens -- it just does not get three chances.
        // Both forms were measured on cy02 sp1: engaging at the first failure
        // (it = 5) and engaging from it = 1 reach the same critical boron, 1319.97
        // ppm, the same axial mode, and agree to the printed digit at all 51
        // statepoints.  The trigger form is preferable because an iteration that
        // contracts never trips it, so a healthy restart cycle keeps its old path.
        const int xe_streak_limit = primeXeDamping ? 1 : XE_OSCILLATION_STREAK;
        // Companion experiment knob: start the Xe iteration damped (same
        // fixed points, contraction from step one) so interim stepping cannot
        // kick the axial Xe map into another basin.  Default off.
        static const bool xe_interim_damp =
            std::getenv("RASBERY_XE_INTERIM_DAMP") != nullptr;
        // ONCE mode sizes every step itself, at the step (see xe_once_full), so
        // neither this companion knob nor the oscillation damper below may touch
        // its relax: the knob would damp a step the trust region has already
        // decided should be full, and the damper's streak is assembled out of
        // single steps taken against DIFFERENT macro-XS states, which is not a
        // limit cycle.  Both are switched off in the mode.
        double    xe_relax        = (xe_interim_damp && !xe_once_mode)
                                        ? XE_DAMPED_RELAX
                                        : 1.0;
        // Per-cascade Xe budget (RASBERY_XE_CASCADE_BUDGET, default off = exact old
        // path).  See XE_CASCADE_TOTAL_MULTIPLIER for why one shared counter starves
        // the late cascades of a search-heavy statepoint.
        static const bool xe_cascade_budget = [] {
            const char* value = std::getenv("RASBERY_XE_CASCADE_BUDGET");
            if (value == nullptr) return false;
            const std::string s(value);
            return !(s.empty() || s == "0" || s == "off" || s == "OFF" ||
                     s == "false" || s == "FALSE");
        }();
        // ONCE mode's trust region (RASBERY_XE_ONCE_TRUST, default XE_ONCE_TRUST):
        // the largest RAW relative Xe change a single step is trusted to hand the
        // converged flux.  Cached in a function-local static like every other knob
        // here, so the loop never reaches getenv, and read only under xe_once_mode.
        // A non-positive or unparsable override would make every step a trip and
        // spend the cap on every cascade, so it falls back to the default instead.
        static const double xe_once_trust = [] {
            const char*  value = std::getenv("RASBERY_XE_ONCE_TRUST");
            const double t     = (value != nullptr) ? std::atof(value) : XE_ONCE_TRUST;
            return (t > 0.0) ? t : XE_ONCE_TRUST;
        }();
        // Anderson trust region (RASBERY_XE_ANDERSON_MAX_STEP, default
        // XE_ANDERSON_MAX_STEP): the largest step, as a multiple of the Picard
        // step it replaces, an extrapolated candidate is allowed to take.
        // Cached in a function-local static like every other knob here, so the
        // loop never reaches getenv, and read only under xe_anderson.  A
        // non-positive or unparsable override would refuse every candidate and
        // silently turn the feature off, so it falls back to the default.
        static const double xe_aa_max_step = [] {
            const char*  value = std::getenv("RASBERY_XE_ANDERSON_MAX_STEP");
            const double t     = (value != nullptr) ? std::atof(value) : XE_ANDERSON_MAX_STEP;
            return (t > 0.0) ? t : XE_ANDERSON_MAX_STEP;
        }();
        // Anderson history for the cascade (plan Rev.4 Sec 10.5).  Declared
        // HERE, which IS cascade start 1 -- the same site the cascade counter
        // and the once-mode allowance are armed at -- and, being a SolveLoop
        // local, destroyed at every exit, so no history can survive a depletion
        // step: Drive() runs PredictorStep/CorrectorStep BETWEEN SolveLoop
        // calls.  Constructed unconditionally (nine empty vectors, no
        // allocation) and touched only under xe_anderson.
        XeAndersonState xe_aa{};
        double prev_xe_change = std::numeric_limits<double>::infinity();
        int    xe_no_progress = 0;   // consecutive Xe steps that did not shrink
        int    xe_interim_count = 0; // loose-flux Xe steps (RASBERY_XE_INTERIM_L2)
        // Settled Xe steps over the WHOLE SolveLoop.  xe_count is per-cascade once the
        // gate is set, so this is what the safety ceiling is measured against; with the
        // gate unset it is written but never read, and the old bound stands unchanged.
        int    xe_total       = 0;
        // One starvation charge per cascade.  With the gate unset there is exactly one
        // cascade per SolveLoop, so this latches after the first event -- which is the
        // pre-existing behaviour, just now counted.
        bool   xe_cap_charged = false;
        // ONCE mode, per-cascade state.  All three are cleared at the three cascade
        // starts -- SolveLoop entry (these initializers) and the T/H and search
        // commits, the same sites the cascade counter and the
        // RASBERY_XE_CASCADE_BUDGET re-arm use, because they are the same events.
        // Written unconditionally and read only under xe_once_mode, so the default
        // path gains three dead stores and nothing else.
        //
        // xe_once_steps: Xe steps this cascade has spent (cap XE_ONCE_MAX_STEPS).
        // xe_once_done:  the cascade's allowance is closed -- either the last step
        //                landed inside the trust region, or the cap is spent.  This
        //                is the term on the step gate, exactly where the one-step
        //                cap used to sit.
        int    xe_once_steps  = 0;
        bool   xe_once_done   = false;
        // The FIRST cascade of a primed statepoint (the first statepoint of a
        // restart/shuffle cycle, primeXeDamping) is the one that arrives farthest
        // from the current-flux equilibrium -- the pool has decayed the inventory,
        // or the shuffle has moved it to a different flux.  There is no earlier step
        // to measure that distance, so the trust region cannot bound step one after
        // the fact; damp it up front instead, the same intent primeXeDamping already
        // carries for the equilibrium iteration.  Cleared at the cascade re-arm, so
        // every later cascade starts at a full step again.
        bool   xe_once_prime  = primeXeDamping;
        SolveExit exit_reason = SolveExit::ITER_EXHAUSTED;
        // Telemetry only (plan Rev.4 Sec 8).  The cause of the segment the NEXT
        // outer belongs to; see the attribution rules in the sptelem comment.
        sptelem::Cause sp_cause = sptelem::CAUSE_INITIAL;
        ++ctx.telemetry.solve_loops;
        // Entering the loop is itself a Xe cascade start: the inventory carried in was
        // equilibrated against a different flux, so it has to re-converge from scratch.
        if (has_eq_xe) ++ctx.telemetry.xe_cascades;

        if (has_search)
            schedule.ResetSearchExitStatus();

        // Hard safety bound. Each feedback (search/T-H) step is followed by a bounded flux
        // re-convergence (flux_stall guard), and the search/T-H step counts are bounded too.
        const int max_iter = 50 * std::max({schedule.max_outer_iter, schedule.max_th_iter,
                                            has_search ? schedule.max_search_iter : 0});
        // ===================================================================
        // Rev.7.1 Task 10: device outer eligibility for SolveLoop
        // ===================================================================
        //
        // Hoisted, exactly as ReconvergeFlux hoists it, for the same reason: a
        // per-outer eligibility query would ask a pure predicate 600 times to
        // get the same answer.  With the feature off this is a false const and
        // every statement in the delegation branch folds away.
        const bool gpu_outer_enabled = gpu::outerGpuEnabled();
        const bool gpu_outer_rods =
            gpu_outer_enabled &&
            gpu::outerDeckHasFractionalRods(ctx.cross_sections.axial_rod_division(),
                                            ctx.geometry.nxyz() > 0
                                                ? &ctx.geometry.rod_fraction(0)
                                                : nullptr,
                                            ctx.geometry.nxyz(), EPS);
        // GATED ON THE PRE-ARM REFUSAL.  Same reason as ReconvergeFlux's copy:
        // arming binds residency and adopts the process-wide canonical nodal
        // set, so a segment that is going to be refused for a reason already
        // visible here -- above all `batch_mode` -- must not be armed for.  With
        // the gate the OUTER=1 arm is byte-identical to OUTER unset whenever the
        // receipt says the segment never engaged.
        const OuterSlotClaim gpu_outer_claim =
            gpu_outer_enabled ? outerSlotClaim(ctx) : OuterSlotClaim{};
        const bool gpu_outer_may_arm =
            gpu_outer_enabled &&
            gpu::outerSegmentPreArmRefusal(rasberyBatchWidth(), gpu_outer_rods, false,
                                           gpu_outer_claim.arena_slots,
                                           gpu_outer_claim.admitted) ==
                gpu::OuterSegmentRefusal::None;
        // WP10.7.  THE RETURN IS READ.  It was discarded here, and that is
        // how nine cases in the 238 arm-A soak came to die naming
        // `no_residency`: armOuterSegment binds residency, bindResidency
        // clears `residency_bound` on its way in (CudaOuterGraph.cu:1118) and
        // leaves its real reason in CudaOuterSegment::status(), and the
        // post-arm ladder one line down then re-derived the failure as the
        // generic "nobody handed this runner its buffers".  The ladder still
        // decides -- the throw below is unchanged and so are the counters --
        // but the step that actually failed is named first, so the receipt
        // carries the cause and not only the symptom.
        const bool gpu_outer_arm_ok =
            gpu_outer_may_arm && armOuterSegment(ctx, eigv, residual);
        if (gpu_outer_may_arm && !gpu_outer_arm_ok)
            gpufull::nameFirstFallback(gpufull::Subsystem::Outer,
                                       "Driver: outer segment arm",
                                       gpu::rasberyOuterSegment(gpu_outer_claim.slot)
                                           .status().c_str());

        // A CRITICAL SEARCH IS NO LONGER REFUSED HERE.
        //
        // Task 9 refused it because the DEVICE decision stood in for Driver.h's
        // search terms and the two spellings disagree
        // (OuterSegmentEligibility::critical_search).  SolveLoop stopped
        // consuming that decision in Task 10: it takes flux_converged and the
        // three carried scalars, and flux_converged is computed before any
        // search term is read.  The host ladder still evaluates
        // schedule.searchResidual(eigv) from the eigenvalue the segment
        // returned, still owns clean_iters and the settle gate, and still
        // decides FluxLimitCycleSample -- exactly the scalars it read before.
        //
        // The one thing the search does to the segment is move the macro-XS
        // when it commits a trial, and that is covered: xsnf and dtil are
        // uploaded at the top of every outer, and resetDhat() is called only
        // from Drive() (2718, 2735), outside this loop, so the arm-time dhat
        // seed cannot go stale within one SolveLoop.
        //
        // This matters because the production workload is a boron-search deck
        // from statepoint 1: with the refusal in place APR1400/kngr_238 never
        // saw a device outer at all.
        const gpu::OuterSegmentRefusal gpu_outer_why =
            gpu_outer_enabled ? gpu::rasberyOuterSegment(gpu_outer_claim.query())
                                    .refusal(rasberyBatchWidth(), gpu_outer_rods, false,
                                             gpu_outer_claim.admitted)
                              : gpu::OuterSegmentRefusal::FeatureOff;
        bool gpu_outer_armed = (gpu_outer_why == gpu::OuterSegmentRefusal::None);
        if (gpu_outer_enabled && !gpu_outer_armed) {
            gpu::noteOuterSegmentRefusal(gpu_outer_why);
            // WP1 (plan Sec 6.3).  The segment never armed, so this whole loop
            // is the host outer body; `gpu_outer_enabled` is the arm's own
            // predicate, so a FeatureOff run never reaches here.  The reason is
            // matched against kGpuFullAllowedOuterRefusals, which no
            // OuterSegmentRefusal is on -- `batch_mode` least of all; the
            // header says why each candidate was refused.
            RASBERY_GPU_FULL_GUARD_ALLOWED(Outer, "Driver: outer segment pre-arm",
                                           gpu::outerRefusalName(gpu_outer_why));
        }

        for (int iout = 0; iout < max_iter; ++iout) {
            // HOW MANY OUTERS THIS PASS OF THE LOOP ACTUALLY RAN.
            //
            // One, on the host body.  `seg.device_outers` on the delegated arm,
            // which is what the stall ladder below has to count: `flux_stall` is
            // Driver.h's "outers since the flux last converged", and a ladder
            // that counted PASSES let a budget-8 segment run eight times as far
            // before the limit cycle was declared.  On kngr_238 statepoint 12 --
            // the first statepoint whose boron trial point limit-cycles -- that
            // is 3619 outers against the host's 680, and the search then samples
            // a different iterate and the whole depletion diverges.
            int  outers_this_pass = 1;
            // WHERE THE TRACER IS.  Set before anything of this outer runs, so
            // the device runner's step lines carry the same (statepoint, outer)
            // the host arm's do and a single grep aligns the two arms.
            if (outertrace::enabled()) outertrace::setContext(ctx.statepoint, iout);
            bool stall_sample = false; // limit-cycle fall-through this outer
            bool th_fired     = false; // telemetry: T/H perturbed inside this outer
            bool xe_restart   = false; // a commit below re-fires the Xe cascade
            const int xe_budget_probe = ga_feedback_passes > 0
                                            ? ga_feedback_passes
                                            : ((xe_relax < 1.0) ? XE_EQUILIBRIUM_MAX_ITER_DAMPED
                                                                : XE_EQUILIBRIUM_MAX_ITER);
            // THIS OUTER'S TOLERANCES.  Resolved once, at the top, so the device
            // segment and the host ladder cannot be asked at different stages
            // within one outer -- the segment is handed these and decides
            // flux_converged with them, and the ladder below re-derives the same
            // verdict from the same pair.  With the feature off `polishing` is
            // true from construction and all three are the production values, so
            // these are three copies and no behaviour.
            const double keff_tol_now = polishing ? keff_tol : loose_keff_tol;
            const double flux_tol_now = polishing ? flux_tol : loose_flux_tol;
            const double xe_tol_now   = polishing ? XE_EQUILIBRIUM_TOLERANCE : loose_xe_tol;
            // ===============================================================
            // Rev.7.1 Task 10: the SolveLoop delegation
            // ===============================================================
            //
            // WHAT MADE THIS POSSIBLE.  Task 9 argued that a SolveLoop segment
            // could not resume, because the outer whose decision ends the
            // segment has already had its BODY run and SolveLoop had no entry
            // point past the body.  That was a statement about the loop, not
            // about the physics: the only thing the ladder below needs from the
            // body is `flux_converged` and the three scalars it is computed
            // from.  Hoisting the declaration out of the body is the entry
            // point, and it costs one `const` and nothing else.
            //
            // THE HOST LADDER STAYS AUTHORITATIVE, deliberately.  The device
            // machine also advances flux_stall, stall_events, clean_iters and
            // xe_interim_count inside cmfdOuterConvergence, and the ladder below
            // advances its own.  Consuming both would double-count, so this
            // branch takes ONLY flux_converged and the three carried scalars and
            // ignores seg.escape entirely.  The device copies of those counters
            // drift and are never read here; they matter on the ReconvergeFlux
            // path, where the device machine IS the authority, and they will
            // matter again when Tasks 13/14/17 put the ladder itself on the
            // device.  flux_converged is exact regardless: it is computed from
            // st.prev_inner, which this loop uploads at segment entry.
            bool flux_converged = false;
            bool outer_on_device = false;
            if (gpu_outer_armed) {
                gpu::OuterSegmentScalars s{};
                s.slot           = gpu_outer_claim.slot;
                s.slot_admitted  = gpu_outer_claim.admitted ? 1 : 0;
                s.eigv           = eigv;
                s.residual       = residual;
                s.prev_inner     = prev_inner;
                s.keff_tol       = keff_tol_now;
                s.flux_tol       = flux_tol_now;
                s.max_outer_iter = static_cast<unsigned int>(schedule.max_outer_iter);
                // The gates the decision reads.  They are this outer`s, not the
                // previous one`s, because the ladder recomputes them below from
                // the same locals -- so the device sees what the host would.
                s.xe_pending      = 0;
                s.th_pending      = 0;
                s.xe_interim_l2   = 0.0;
                s.xe_once_mode    = 0;
                s.xe_budget_probe = static_cast<unsigned int>(xe_budget_probe);
                // THE STALL LADDER'S CARRIED COUNT, UPLOADED LIKE prev_inner.
                //
                // Without it the device machine counts from whatever the
                // previous segment left, and -- worse -- it cannot stop at the
                // outer this loop's own limit-cycle test would have stopped at.
                // Seeded, the device's `++st.flux_stall > max_outer_iter` fires
                // on exactly that outer, so the segment ends there and the
                // ladder below re-derives the same verdict from
                // `flux_stall + device_outers`.
                s.flux_stall      = static_cast<unsigned int>(flux_stall);

                gpu::OuterSegmentResume seg{};
                if (gpu::rasberyOuterSegment(gpu_outer_claim.query())
                        .runSegment(s, rasberyBatchWidth(), gpu_outer_rods, false, seg)) {
                    // WP1 follow-up: THE SEGMENT IS OVER -- its stream is
                    // drained and any graph capture is closed -- so this is the
                    // first point at which the enqueue seam's DEFERRED violation
                    // can be thrown without leaving CUDA state nothing is
                    // written to clean up.  A no-op unless one was latched,
                    // which is every call with the gate off.
                    RASBERY_GPU_FULL_RAISE_PENDING();
                    eigv           = seg.eigv;
                    residual       = seg.residual;
                    prev_inner     = seg.prev_inner;
                    flux_converged = seg.flux_converged != 0;
                    total_outer += static_cast<int>(seg.device_outers);
                    // CHARGE THE LOOP BOUND IN OUTERS, NOT IN PASSES.
                    //
                    // ReconvergeFlux has done this since Task 9 (Driver.h:1402)
                    // and SolveLoop did not, which made `max_iter` mean
                    // `max_iter SEGMENTS` on the ON arm -- up to eight times the
                    // outers the OFF arm is allowed at the same point of the
                    // solve.  The loop's own ++ supplies the last one.
                    iout += static_cast<int>(seg.device_outers) - 1;
                    outers_this_pass = static_cast<int>(seg.device_outers);
                    ctx.telemetry.outers_by_cause[sp_cause] +=
                        static_cast<long long>(seg.device_outers);
                    outer_timing::buckets().outers.fetch_add(seg.device_outers,
                                                             std::memory_order_relaxed);
                    outer_on_device = true;
                } else {
                    // A launch or hook failure.  Every ELIGIBILITY refusal was
                    // decided once above the loop, so this recurs; stop paying
                    // for it and let the host body below be the whole outer.
                    // Same safe point on the refusal path: see ReconvergeFlux.
                    RASBERY_GPU_FULL_RAISE_PENDING();
                    gpu_outer_armed = false;
                    // WP1 (plan Sec 6.3), same seam as ReconvergeFlux's.
                    RASBERY_GPU_FULL_GUARD(Outer, "Driver::SolveLoop",
                                           "runSegment refused or failed to launch; the "
                                           "host outer body takes over");
                }
            }

            if (!outer_on_device) {
            // THE PER-STEP TRACE, host arm.  Five hashes at the five points the
            // device runner also hashes, so a divergence names a STEP and not
            // just an outer.  hashDoubles over the host arrays is right HERE
            // because this arm computes into them; the device arm hashes device
            // memory, for the reason OuterTrace.h states.
            const bool     tr_step = outertrace::active();
            const int      tr_nxyz = ctx.geometry.nxyz();
            const int      tr_ng   = ctx.geometry.ng();
            const size_t   tr_nsg  = static_cast<size_t>(ctx.geometry.nsurf()) * tr_ng;
            const size_t   tr_nn   = static_cast<size_t>(tr_nxyz) * tr_ng;
            // 1. Flux: CMFD BiCGSTAB iterations + Wielandt shift.
            {
                outer_timing::Scope t(sptelem::PH_UPDPSI);
                ctx.cmfd_solver.updpsi(ctx.geometry.Phif());
            }
            if (tr_step)
                outertrace::emitStep("host", "updpsi", "psi",
                                     outertrace::hashDoubles(ctx.cmfd_solver.psiData(),
                                                             static_cast<size_t>(tr_nxyz)),
                                     nullptr, 0);
            {
                outer_timing::Scope t(sptelem::PH_SETLS);
                ctx.cmfd_solver.setls(eigv);
            }
            {
                outer_timing::Scope t(sptelem::PH_DRIVE);
                ctx.cmfd_solver.drive(eigv, ctx.geometry.PhifMutable(), residual);
            }
            if (tr_step)
                outertrace::emitStepEigv("host", "sweep",
                                         outertrace::hashDoubles(ctx.geometry.Phif(), tr_nn),
                                         eigv);
            ++total_outer;
            // Exactly one cause bucket per outer, charged to the segment this
            // outer belongs to (plan Rev.4 Sec 8 attribution rules).
            ++ctx.telemetry.outers_by_cause[sp_cause];
            outer_timing::buckets().outers.fetch_add(1, std::memory_order_relaxed);
            flux_converged = std::abs(prev_inner - eigv) < keff_tol_now && residual < flux_tol_now;
            prev_inner     = eigv;

            // 2. Nodal correction -> CNCC (d-hat) + rod cusping macro-XS update. The cusping blend
            //    co-converges with the flux, so its settledness is implied by flux_converged.
            {
                outer_timing::Scope t(sptelem::PH_UPDJNET);
                ctx.cmfd_solver.updjnet(ctx.geometry.Phif(), ctx.geometry.Jnet());
            }
            if (tr_step)
                outertrace::emitStep("host", "updjnet", "jnet",
                                     outertrace::hashDoubles(ctx.geometry.Jnet(), tr_nsg),
                                     nullptr, 0);
            {
                outer_timing::Scope t(sptelem::PH_NODAL);
                ctx.nodal_solver.reset(1.0 / eigv, ctx.geometry.Jnet(),
                                       ctx.geometry.Phif(), ctx.geometry.Phis());
                ctx.nodal_solver.drive();
            }
            if (tr_step)
                outertrace::emitStep("host", "nodal", "jnet",
                                     outertrace::hashDoubles(ctx.geometry.Jnet(), tr_nsg),
                                     "phis",
                                     outertrace::hashDoubles(ctx.geometry.Phis(), tr_nsg));
            {
                outer_timing::Scope t(sptelem::PH_CUSPING);
                if (ctx.cross_sections.ApplyRodCusping(eigv, ctx.nodal_solver.axialTransverseLeakage()))
                    ctx.cmfd_solver.upddtil();
            }
            {
                outer_timing::Scope t(sptelem::PH_UPDDHAT);
                ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());
            }
            if (tr_step)
                outertrace::emitStep("host", "upddhat", "dhat",
                                     outertrace::hashDoubles(ctx.cmfd_solver.dhatData(), tr_nsg),
                                     nullptr, 0);
            }

            if (outertrace::active()) {
                const int    nxyz  = ctx.geometry.nxyz();
                const int    ngg   = ctx.geometry.ng();
                const size_t nsg   = static_cast<size_t>(ctx.geometry.nsurf()) * ngg;
                outertrace::emit(
                    ctx.statepoint, iout, outer_on_device ? "dev" : "host", eigv, residual,
                    prev_inner,
                    outertrace::hashDoubles(ctx.cmfd_solver.psiData(),
                                            static_cast<size_t>(nxyz)),
                    outertrace::hashDoubles(ctx.geometry.Jnet(), nsg),
                    outertrace::hashDoubles(ctx.cmfd_solver.dhatData(), nsg),
                    outertrace::hashDoubles(ctx.geometry.Phif(),
                                            static_cast<size_t>(nxyz) * ngg));
            }

            // Keep iterating flux + nodal/cusping until the flux is converged; the feedbacks
            // (search, T/H) are root-finds on k_eff / power and must act on a clean flux.
            //
            // EXPERIMENT (RASBERY_XE_INTERIM_L2=<residual>, default off = exact
            // old path): the equilibrium-Xe cascade re-converges the flux to
            // full tolerance between EVERY Xe update, and that cascade is where
            // the outer budget goes (CY02: 622 outers/state vs MASTER's ~59).
            // With the gate set, a pending Xe update may fire as soon as the
            // flux residual is below the interim threshold; search and T/H
            // still act only on a fully-converged flux, and final convergence
            // still requires the full tolerance with Xe settled, so the
            // converged fixed point is the same to within the tolerances --
            // the PATH differs, so results shift at the pcm level and the
            // accuracy gate is keff/CBC/AO/Fq against the canon run.
            static const double xe_interim_l2 = [] {
                const char* v = std::getenv("RASBERY_XE_INTERIM_L2");
                return v ? std::atof(v) : 0.0;
            }();
            // Interim steps do not consume the settled-flux Xe budget: that
            // budget exists to break converged-level limit cycles, and with
            // interim stepping Xe fires nearly every outer, so counting those
            // would freeze Xe long before it settles (measured: node power off
            // by 11% at tol=1e-4 with the shared counter).  Interim spins are
            // bounded separately at 10x the budget, under the global max_iter.
            //
            // "Out of budget", in one place so the interim probe and the Xe step
            // below can never disagree about it.  xe_count is per-cascade once
            // RASBERY_XE_CASCADE_BUDGET is set, so the SolveLoop-wide ceiling on
            // xe_total takes over as the safety bound there; with the gate unset the
            // second term is not evaluated and this is exactly xe_count < xe_budget.
            const int  xe_budget  = xe_budget_probe;
            const bool xe_starved = xe_count >= xe_budget ||
                                    (xe_cascade_budget &&
                                     xe_total >= XE_CASCADE_TOTAL_MULTIPLIER * xe_budget);
            const bool xe_pending = has_eq_xe && !xe_starved &&
                                    (xe_count + xe_interim_count == 0 ||
                                     prev_xe_change >= xe_tol_now);
            // ONCE mode takes its one step on a CONVERGED flux -- that is the
            // whole point of the mode, an inventory consistent with the flux the
            // segment actually publishes -- so the interim probe is excluded
            // there rather than left to be spent by the single step.
            const bool xe_interim = xe_interim_l2 > 0.0 && !xe_once_mode && xe_pending &&
                                    xe_interim_count < 10 * xe_budget_probe &&
                                    !flux_converged && residual < xe_interim_l2;
            if (xe_interim)
                flux_stall = 0; // the Xe step below changes the problem; not a stall
            if (!flux_converged && !xe_interim) {
                flux_stall += outers_this_pass;
                if (flux_stall <= schedule.max_outer_iter)
                    continue;

                // Flux limit cycle on this trial point.  Previously this break abandoned the
                // solve and published the limit-cycling k_eff as the step answer with no
                // diagnostic at all (CY02 step 1: k=0.999562, |dk| = 43.8 pcm = 8.8x the
                // 5e-5 rod-crit tolerance).  Record it, warn, and carry on with the search so
                // the root find can step off the pathological point.
                ++stall_events;
                ++ctx.telemetry.flux_limit_retries;
                if (has_search)
                    schedule.search_stall_count = stall_events;
                std::cerr << std::format(
                    "[RASBERY][WARN][flux] limit cycle at trial point (event {}/{}): "
                    "|dk_inner|={:.3e} L2={:.3e} k_eff={:.8f} x={:.6f}\n",
                    stall_events, MAX_FLUX_STALL_EVENTS, std::abs(prev_inner - eigv), residual,
                    eigv, has_search ? schedule.search_current_x : 0.0);
                flux_stall = 0;
                if (stall_events > MAX_FLUX_STALL_EVENTS || !has_search) {
                    exit_reason = SolveExit::FLUX_STALL;
                    break;
                }
                stall_sample = true;
                // fall through: treat the limit-cycle k_eff as a (noisy) observation.
                // The settling gate below must not hold here: the flux never converges on
                // this trial point, so waiting for settled iterations would spin until the
                // outer bound.  Take the sample as-is, exactly as before this gate existed.
                clean_iters = SEARCH_SETTLE_ITERS;
            } else {
                flux_stall = 0;
            }

            // Starvation probe (telemetry, always on).  The cascade still has a change
            // above tolerance but has no steps left, so every inventory published from
            // here on -- including the one this statepoint writes out -- is a truncated,
            // unconverged one.  Read at the top of the outer rather than at the step
            // that spent the last unit, so a budget that has just doubled because the
            // damper engaged is already reflected and cannot raise a false alarm.
            if (has_eq_xe && !xe_cap_charged && xe_starved &&
                prev_xe_change >= xe_tol_now) {
                xe_cap_charged = true;
                ++ctx.telemetry.xe_budget_exhausted;
                // GA screening deliberately truncates the feedback passes, so the
                // event is expected there and only the counter is wanted.
                if (xe_cascade_budget && ga_feedback_passes == 0)
                    std::cerr << std::format(
                        "[RASBERY][WARN][xe] cascade budget exhausted at statepoint {} "
                        "(EFPD {:.3f}): {} steps, last rel={:.3e} > tol {:.1e}; published "
                        "Xe inventory is NOT converged\n",
                        ctx.statepoint, ctx.efpd, xe_count, prev_xe_change,
                        XE_EQUILIBRIUM_TOLERANCE);
            }

            // 3. Equilibrium xenon feedback.  Previously equilibrium Xe was
            // only overwritten inside depletion, so a BOC STANDARD step
            // silently ran with zero Xe despite "xenon":"equilibrium".
            // ONCE mode's cap lives here, as one term on the gate the step
            // already hangs off: `(!xe_once_mode || !xe_once_done)` is
            // identically true with the mode unset, and in the mode it lets a
            // cascade through until the trust-region block below closes it --
            // one step when that step lands inside the region, at most
            // XE_ONCE_MAX_STEPS when it does not.  The gate carries no test on
            // the Xe residual: XE_EQUILIBRIUM_TOLERANCE is not what this mode
            // is chasing.
            if (has_eq_xe && !xe_starved && (!xe_once_mode || !xe_once_done) &&
                (flux_converged || xe_interim || stall_sample)) {
                // WP9-A `loop_wall.xe_step`: THE WHOLE Xe step region -- the
                // Anderson attempt, the production Picard step it falls back
                // to, and the trust-region / damper bookkeeping that follows.
                // Whole, and not just the production call, for two reasons.
                // The bucket then has no hole: `xe_updates +
                // xe_interim_updates` is exactly its call count whichever arm
                // took the step, and the Anderson/Picard split stays readable
                // from `xe_aa_accepted` in the same receipt.  And it keeps the
                // production call site textually untouched, which is what
                // tools/test_xe_anderson.py pins.
                outer_timing::Scope xe_step_scope(sptelem::PH_XE_STEP);
                // ONCE sizes THIS step, here, because the size is per-step and
                // not per-iteration: a full step unless the trust region has
                // already been breached this cascade (xe_once_steps > 0) or the
                // cascade is the primed one that cannot be measured first.
                // Both writes are unreachable with the mode unset, so xe_relax
                // reaches the call exactly as the damper left it.  The one other
                // reader of xe_relax is xe_budget_probe, which hands a damped
                // solve the larger XE_EQUILIBRIUM_MAX_ITER_DAMPED budget: a once
                // cascade that had to damp buys the same loosening, which is the
                // safe direction and is nowhere near binding at a cap of three.
                const bool xe_once_full = xe_once_mode && xe_once_steps == 0 && !xe_once_prime;
                if (xe_once_mode) xe_relax = xe_once_full ? 1.0 : XE_DAMPED_RELAX;
                // Safeguarded Anderson attempt on the SAME map, at the SAME
                // point, with the SAME acceptance semantics -- only the iterate
                // differs (see TryAndersonXeStep).  Three terms guard it, and
                // all three short-circuit, so with the feature unset the
                // initializer below is exactly the pre-Anderson one and nothing
                // else is evaluated:
                //
                //   xe_anderson       the cached feature gate.
                //   xe_relax == 1.0   the oscillation damper is not engaged.
                //                     Once it is, it is SELECTING a root, and an
                //                     extrapolation built off the pre-damping
                //                     map must not fight that choice.
                //   flux_converged    the step is being taken on the converged
                //                     flux.  What Anderson iterates is the
                //                     COMPOSITE map x -> (converge the flux at
                //                     x) -> equilibrium Xe, and only a
                //                     converged-flux evaluation is a point of
                //                     that map.  The other two ways into this
                //                     block -- the RASBERY_XE_INTERIM_L2 probe's
                //                     loose flux and the flux-limit-cycle
                //                     fall-through -- evaluate something else,
                //                     so they take the plain step and are never
                //                     recorded in the history.  On the default
                //                     configuration this term is already true
                //                     wherever the step fires.
                //
                // The attempt writes nothing unless it accepts, so the fallback
                // is the production step running on an untouched solver state.
                double     xe_change   = 0.0;
                const bool xe_aa_taken = xe_anderson && xe_relax == 1.0 && flux_converged &&
                                         TryAndersonXeStep(ctx, xe_aa, schedule.thermalPower(),
                                                           xe_aa_max_step, xe_change);
                if (!xe_aa_taken)
                    xe_change =
                        ctx.cross_sections.UpdateEquilibriumXenon(schedule.thermalPower(), xe_relax);
                // An Xe step moves the macro-XS whatever it returns, so it opens
                // an XE segment even when the change is under tolerance and the
                // loop falls through to the search/T-H checks below.
                sp_cause = sptelem::CAUSE_XE;
                if (xe_interim && !flux_converged) {
                    ++xe_interim_count;
                    ++ctx.telemetry.xe_interim_updates;
                } else {
                    ++xe_count;
                    ++xe_total;
                    ++ctx.telemetry.xe_updates;
                }
                // ONCE mode's trust region.  UpdateEquilibriumXenon returns the RAW,
                // PRE-damping relative change (XSSet.cpp measures |new-old|/|new| off
                // the undamped Picard image and only then blends by relax; the GPU
                // kernel in XsReconKernel.h does the same, in the same order), so
                // xe_change is the distance to the current-flux equilibrium whatever
                // relax this step was applied with -- which is exactly the shock a
                // full step would have delivered.  Bound THAT, not the residual:
                // stop the cascade as soon as a step lands inside the region, and
                // otherwise buy one more damped step, up to the hard cap.
                if (xe_once_mode) {
                    const bool xe_once_extra = xe_once_steps > 0;
                    ++xe_once_steps;
                    if (xe_once_extra) ++ctx.telemetry.xe_once_extra_steps;
                    const bool xe_once_trip = xe_change > xe_once_trust;
                    if (xe_once_trip) ++ctx.telemetry.xe_once_trust_trips;
                    xe_once_done = !xe_once_trip || xe_once_steps >= XE_ONCE_MAX_STEPS;
                }
                // Not contracting?  The undamped Xe<->flux map is limit-cycling rather than
                // converging.  Measured on APR1400 cy01 at 195-225 EFPD: [XE] rel bounces
                // between 8e-x and 11e-x for 80+ iterations, eigv swinging +-30 pcm, until
                // the 100-iteration cap is spent -- five statepoints (17-21) burned 7,016 of
                // the run's 13,383 outers that way.  Halving the applied step leaves the
                // fixed point alone and turns the oscillation into a contraction.  Once
                // engaged it stays engaged for this solve: re-raising it re-enters the cycle.
                if (xe_change >= XE_OSCILLATION_FLOOR && xe_change >= prev_xe_change)
                    ++xe_no_progress;
                else
                    xe_no_progress = 0;
                // Unreachable in ONCE mode: xe_no_progress compares consecutive
                // steps of one iteration, and there is no iteration to compare
                // -- the streak would be assembled out of single steps taken
                // against DIFFERENT macro-XS states, which is not a limit cycle.
                if (!xe_once_mode && xe_relax == 1.0 && xe_no_progress >= xe_streak_limit) {
                    xe_relax = XE_DAMPED_RELAX;
                    // Anderson stands down here for the rest of the solve --
                    // xe_relax never goes back up -- because the damper is a
                    // ROOT SELECTOR, not a speed knob (see the xe_relax
                    // initializer's cy02 case): it is choosing which fixed point
                    // the cascade lands on, and residual differences collected
                    // off the undamped map would pull straight back toward the
                    // branch it just walked away from.  Sec 10.5 asks for the
                    // history to go with the relaxation change, so it does.
                    ResetXeAndersonHistory(ctx, xe_aa);
                    if (trace_sl)
                        std::cout << std::format(
                            "        [XE] no contraction for {} steps ({:.3e} -> {:.3e}); "
                            "damping to relax={:.2f}, budget {}\n",
                            xe_no_progress, prev_xe_change, xe_change, xe_relax,
                            XE_EQUILIBRIUM_MAX_ITER_DAMPED);
                }
                prev_xe_change = xe_change;
                if (trace_sl)
                    std::cout << std::format("        [XE] it={} rel={:.3e} eigv={:.6f}\n",
                                             xe_count, xe_change, eigv);
                // The second term is ONCE mode's: a cascade the trust region has
                // left open owes its next step a re-converged flux, and that has
                // to hold even for a RASBERY_XE_ONCE_TRUST set below the
                // equilibrium tolerance, where the first term would not fire.
                // It short-circuits on xe_once_mode, so with the mode unset this
                // is exactly the first term and only that is evaluated -- and
                // with A2 staging also unset, xe_tol_now IS
                // XE_EQUILIBRIUM_TOLERANCE, so the whole line is what it was.
                if (xe_change >= xe_tol_now || (xe_once_mode && !xe_once_done)) {
                    // Cross sections changed; re-converge the flux before
                    // taking a search or T/H feedback step.
                    prev_inner  = eigv + 1.0;
                    clean_iters = 0;
                    continue;
                }
            }

            // An interim Xe step ran on a loosely-converged flux; whatever it
            // returned, search/T-H may only act on a fully-converged flux.
            // Scoped to the interim path only: the flux-stall fall-through
            // above also reaches here unconverged, and its noisy-sample
            // behavior must stay exactly as it was.
            if (xe_interim && !flux_converged) {
                prev_inner = eigv + 1.0;
                continue;
            }

            // Settling gate.  The flux is converged and Xe has settled; give the nodal d-hat
            // the same chance before the search reads k_eff off it.  Search solves only --
            // a no-search solve keeps its exact old iteration path, hence its exact answer.
            //
            // BORON only, deliberately.  The gate buys settled samples by spending extra
            // outers, and on a rod search with cusping active those extra outers are what
            // feeds the pathology: each one re-blends the fractional fine-cell macro-XS, the
            // CMFD matrix alternates, and the L2 residual parks at ~1e-4 so the flux never
            // re-converges at all (see MAX_FLUX_STALL_EVENTS).  Measured: applying the gate
            // to RODCRIT drove i-SMR CY03/CY04 into flux limit cycles they had never hit,
            // moving keff by up to 8.2 pcm and the rod step by 0.014.  The rod search already
            // has its own answer to sampling noise -- rodcrit_search_floor raises the search
            // tolerance to 5e-5, above the cusping noise floor -- and that one is validated.
            // A2 (RASBERY_STAGED_LOOSE_SETTLE=1, default off, staged tolerance
            // only): the LOOSE stage does not pay for the gate.
            //
            // The gate exists so the search reads k_eff off a d-hat that has had
            // the same chance to settle the flux has.  That argument is about
            // the sample the search TRUSTS.  Under staging there are two kinds
            // of sample: the loose ones, which only steer the secant toward the
            // neighbourhood and are all re-tested at production tolerance before
            // anything is published, and the polish one, which is the sample the
            // exit test and the published k_eff are taken from.  The gate is
            // worth its two outers on the second and is spending them on the
            // first: measured at RASBERY_STAGED_FLUX_TOL=50 / _XE_TOL=1000 the
            // settle bucket is 1,199 outers, 21.1 % of the run, having been
            // 8.5 % before staging -- staging shrank everything else around it.
            //
            // The risk it takes is a noisier secant, so it is its own flag and
            // its own arm: a loosening that costs more search trials than it
            // saves settle outers is a loss, and that has to be measured rather
            // than assumed.  It is inert unless staging is on, because with a
            // single stage `polishing` is true throughout and this is the
            // original gate exactly.
            // WP10.3: per case, for the reason the two multipliers above are.
            // The truthiness test itself did not move -- it is
            // parseStagedLooseSettle (CaseFidelity.h), which is this lambda
            // verbatim, so that the receipt and the solver read the knob the
            // same way.
            const bool staged_loose_settle = ctx.fidelity.loose_settle;
            if (has_search && schedule.searchType == SearchType::BORON &&
                !(staged_loose_settle && !polishing) &&
                clean_iters < SEARCH_SETTLE_ITERS) {
                ++clean_iters;
                prev_inner = eigv + 1.0;   // force a real re-drive before the next check
                // The next outer exists only because the gate refused this
                // sample: that is Sec 8's "settling gate extra outer".
                sp_cause = sptelem::CAUSE_SETTLE;
                continue;
            }

            // 4. Critical search ("CBC Search?").
            bool         search_converged = !has_search;
            const double k_residual       = has_search ? schedule.searchResidual(eigv) : 0.0;
            if (has_search) {
                schedule.UpdateBestSearchPoint(k_residual);
                if (schedule.searchType == SearchType::RODCRIT) {
                    schedule.rod_step = schedule.search_current_x;
                    schedule.UpdateRodBracket(k_residual);
                } else if (ctx.search_policy.boron_bracket) {
                    // WP9-D stage D.  The boron search gets the sign-change
                    // bracket RODCRIT has, and only behind its own flag: the
                    // bracket is what turns a wandering secant into a bounded
                    // one, and it is the fallback the trial cap lands on.
                    schedule.UpdateSearchBracket(k_residual);
                }
                search_converged = std::abs(k_residual) < search_tol;
            }

            // 5. T/H feedback ("Need T/H?"): converged on the Doppler (fuel) temperature change
            //    (PARCS delta_Dop), which is physical and far above the cusping k_eff noise floor.
            const bool th_converged = !has_th ||
                                      (ga_feedback_passes > 0 && th_count >= ga_feedback_passes) ||
                                      (th_count > 0 && th_dop < TH_DOPPLER_TOLERANCE) ||
                                      th_count >= schedule.max_th_iter;

            // 6. All converged?
            if (search_converged && th_converged) {
                // A2 STAGED TOLERANCE: agreement reached at the LOOSE tolerance
                // is not an exit, it is the trigger for the polish pass.  Three
                // things are restored here and re-tested, in the order the loop
                // will meet them again:
                //
                //   the flux -- prev_inner is poisoned so the next outer is a
                //   real re-drive rather than a re-read of the loose iterate,
                //   and keff_tol_now/flux_tol_now become the production pair at
                //   the top of that outer;
                //
                //   the Xe inventory -- prev_xe_change is re-armed to infinity
                //   so xe_pending fires once more.  Without this the published
                //   inventory would be the one equilibrated against the LOOSE
                //   flux, and the whole point of the polish pass is that
                //   everything it publishes met the production tolerance.  A
                //   cascade that was already converged answers in one step;
                //
                //   the settling gate -- clean_iters is cleared so the search
                //   re-samples k_eff off the polished flux rather than trusting
                //   the loose sample it just accepted.
                //
                // If the search then disagrees at production tolerance it
                // commits another trial and the block below drops back to loose,
                // which is the self-correction this design rests on; the relapse
                // is counted so an over-loose multiplier is visible as thrash
                // rather than as an unexplained outer count.
                if (!polishing) {
                    polishing   = true;
                    prev_inner  = eigv + 1.0;
                    clean_iters = 0;
                    if (has_eq_xe)
                        prev_xe_change = std::numeric_limits<double>::infinity();
                    if (trace_sl)
                        std::cout << std::format(
                            "        [STAGE] loose agreement at outer {}; polishing at "
                            "keff_tol={:.2e} flux_tol={:.2e} xe_tol={:.2e}\n",
                            iout, keff_tol, flux_tol, XE_EQUILIBRIUM_TOLERANCE);
                    continue;
                }
                exit_reason = SolveExit::CONVERGED;
                break;
            }
            // Production tolerance failed to hold what the loose stage agreed
            // on.  Back to loose for the trial the perturbation blocks below are
            // about to commit -- the point being converged to is about to move,
            // so there is nothing left to polish.
            if (staged_tol && polishing) {
                polishing = false;
                ++polish_relapses;
                ++ctx.telemetry.staged_relapses;
            }

            // Otherwise perturb the unconverged feedbacks, then re-converge the flux.
            if (has_th && !th_converged) {
                {
                    // WP9-A `loop_wall.th_update`: the whole call, including
                    // the UpdateFlatXS it ends with.  `nested_wall.flatxs`
                    // says how much of it that was.
                    outer_timing::Scope th_scope(sptelem::PH_TH_UPDATE);
                    th_dop = ctx.cross_sections.UpdateTH(power_fraction);
                }
                ++total_th;
                ++th_count;
                ++ctx.telemetry.th_updates;
                th_fired    = true;
                xe_restart  = true; // new temperatures -> new Xe fixed point
                sp_cause    = sptelem::CAUSE_TH;
                clean_iters = 0;   // the cross sections just moved
                if (trace_sl)
                    std::cout << std::format("        [TH] it={} th_dop={:.3e} eigv={:.6f}\n",
                                             th_count, th_dop, eigv);
            }
            if (has_search && !search_converged) {
                // WP9-D stage D, candidate D5 (gate N1).  The cap may only take
                // trials away (trialCap never returns more than the deck's own
                // limit), and it exits through the deck limit's OWN path -- so
                // the deterministic best-fallback below re-converges the best
                // observed point at PRODUCTION tolerance and search_exit_status
                // publishes that the statepoint did not converge.  Acceptance
                // is therefore unchanged by construction.  With the knob unset
                // trialCap answers `max_search_iter` and this is the same test.
                if (schedule.search_iteration >=
                    ctx.search_policy.trialCap(schedule.max_search_iter)) {
                    // The best-observed point is re-applied deterministically after the loop
                    // (see the fallback block below), so there is nothing to salvage here.
                    exit_reason = SolveExit::SEARCH_EXHAUSTED;
                    break;
                }
                {
                    // WP9-A `loop_wall.search_propose`: the secant/bracket
                    // arithmetic and the commit.  No cross sections move in
                    // here -- that is the next bucket -- so the two together
                    // are what one search TRIAL costs on the host, which is
                    // the number WP9-D's trial-reduction options are priced
                    // against.
                    outer_timing::Scope propose_scope(sptelem::PH_SEARCH_PROPOSE);
                    double      next_x = schedule.search_current_x;
                    std::string method;
                    bool        bracket_not_found = false;
                    if (!schedule.ProposeNextSearchPoint(eigv, ctx.search_memory,
                                                         ctx.cross_sections.rod_max_step(),
                                                         ctx.search_policy, ctx.search_carry,
                                                         next_x, method, bracket_not_found)) {
                        exit_reason = SolveExit::NO_PROPOSAL;
                        break;
                    }
                    schedule.CommitSearchPoint(eigv, next_x, ctx.search_memory);
                }
                if (trace_search) {
                    const char* nm = (schedule.searchType == SearchType::BORON) ? "BORON" : "ROD";
                    std::cout << std::format(
                        "        [SEARCH] {} it={} x={:.6f} k={:.8f} dk={:+.3e} outer={} bracket={}\n",
                        nm, schedule.search_iteration, schedule.search_current_x, eigv, k_residual,
                        total_outer, schedule.hasRodBracket()
                                         ? std::format("[{:.6f},{:.6f}]", schedule.search_bracket_lo_x,
                                                       schedule.search_bracket_hi_x)
                                         : std::string("none"));
                }
                {
                    // WP9-A `loop_wall.search_apply`: the trial point reaching
                    // the macro cross sections.  SetBoron/SetRod end in
                    // UpdateFlatXS, so this is where a boron trial's real host
                    // cost is, and `nested_wall.flatxs` is how much of it.
                    outer_timing::Scope apply_scope(sptelem::PH_SEARCH_APPLY);
                    if (schedule.searchType == SearchType::BORON)
                        ctx.cross_sections.SetBoron(schedule.search_current_x);
                    else {
                        ctx.cross_sections.SetRod(schedule.search_current_x);
                        schedule.rod_step = schedule.search_current_x;
                    }
                }
                // Committed AND applied: this is the trial the segment below
                // pays for.  A T/H step in the same outer loses the tie-break
                // (see the sptelem comment) but is counted as ambiguous.
                ++ctx.telemetry.search_trials;
                if (th_fired) ++ctx.telemetry.th_search_coincident;
                xe_restart  = true; // new boron / rod position -> new Xe fixed point
                sp_cause    = sptelem::CAUSE_SEARCH;
                clean_iters = 0;   // new trial point: the next sample must settle first
            }

            // Cascade boundary.  A committed AND APPLIED perturbation moved the
            // macro-XS, so the Xe iteration that just converged is finished and a new
            // one starts here -- the same two sites the SEARCH/TH attribution segments
            // open at, because they are the same events.  A T/H step and a search step
            // in the same outer are one restart, not two (the tie-break the sptelem
            // comment describes).  Counting is unconditional; re-arming the budget is
            // gated, since that is what changes the trajectory.
            if (xe_restart && has_eq_xe) {
                ++ctx.telemetry.xe_cascades;
                // The fresh cascade gets its own ONCE allowance: the macro-XS just
                // moved, so the inventory equilibrated against the previous state is
                // stale and a step against the new one is what this mode owes the
                // segment.  The step count and the trust latch reset together -- a
                // cascade that hit XE_ONCE_MAX_STEPS must not carry the cap into the
                // next one -- and the prime flag is spent here, because after a
                // committed perturbation the inventory is no longer the far-off one
                // the deck handed over and the trust region can measure the distance
                // for itself.  Unconditional; only read under xe_once_mode, so the
                // default path is unchanged.
                xe_once_steps = 0;
                xe_once_done  = false;
                xe_once_prime = false;
                // The macro-XS just moved, so the map whose residual differences
                // the Anderson history was built from no longer exists (Sec
                // 10.5: boron / rod / T-H change -> history reset).  Extrapolating
                // across that boundary would fit the OLD map's curvature onto the
                // new one.  Unconditional and idempotent; it charges the counter
                // only when there was something to discard, and with the feature
                // off there never is.
                ResetXeAndersonHistory(ctx, xe_aa);
                // xe_interim_count is re-armed with it so the fresh cascade is allowed
                // its first interim step: xe_pending's "nothing has fired yet" term
                // reads the pair.  Interim spins stay bounded by max_iter.
                if (xe_cascade_budget) {
                    xe_count         = 0;
                    xe_interim_count = 0;
                    xe_cap_charged   = false;
                }
            }
        }

        // Deterministic acceptance.  Every exit above now lands here instead of silently
        // publishing whatever point the loop happened to stop on.  If the search tolerance was
        // not met, re-apply the best observed trial point (minimum |k_eff - target| over the
        // whole search), re-converge the flux on it so the published k_eff belongs to the
        // published rod position / boron, and warn.
        if (has_search) {
            schedule.search_exit_tol = search_tol;
            double k_res             = schedule.searchResidual(eigv);
            schedule.UpdateBestSearchPoint(k_res);

            if (std::abs(k_res) < search_tol) {
                schedule.search_exit_status = static_cast<int>(SearchExit::CONVERGED);
            } else if (schedule.search_has_best &&
                       std::abs(schedule.search_best_residual) < std::abs(k_res)) {
                const double from_x = schedule.search_current_x;
                schedule.search_current_x = schedule.search_best_x;
                {
                    outer_timing::Scope apply_scope(sptelem::PH_SEARCH_APPLY);
                    if (schedule.searchType == SearchType::BORON)
                        ctx.cross_sections.SetBoron(schedule.search_current_x);
                    else {
                        ctx.cross_sections.SetRod(schedule.search_current_x);
                        schedule.rod_step = schedule.search_current_x;
                    }
                }
                const int fallback_outer0 = total_outer;
                ReconvergeFlux(ctx, eigv, FALLBACK_RECONVERGE_ITER, keff_tol, flux_tol,
                               total_outer);
                // ReconvergeFlux runs its own loop with every feedback frozen, so
                // no cause can change inside it: charge the whole delta at once.
                ctx.telemetry.outers_by_cause[sptelem::CAUSE_FALLBACK] +=
                    total_outer - fallback_outer0;
                k_res                       = schedule.searchResidual(eigv);
                schedule.search_exit_status = static_cast<int>(SearchExit::BEST_FALLBACK);
                std::cerr << std::format(
                    "[RASBERY][WARN][search] fell back to best trial point: x {:.6f} -> {:.6f} "
                    "(observed dk {:+.3e} -> re-converged dk {:+.3e})\n",
                    from_x, schedule.search_current_x, schedule.search_best_residual, k_res);
            } else {
                schedule.search_exit_status = static_cast<int>(SearchExit::UNCONVERGED);
            }

            schedule.search_exit_dk = k_res;
            if (schedule.search_exit_status != static_cast<int>(SearchExit::CONVERGED))
                std::cerr << std::format(
                    "[RASBERY][WARN][search] {} search NOT converged: exit={} |dk|={:.3e} "
                    "({:.1f} pcm) tol={:.3e} ({:.1f}x) x={:.6f} iters={} stalls={}\n",
                    (schedule.searchType == SearchType::BORON) ? "boron" : "rod",
                    SolveExitName(exit_reason), std::abs(k_res), std::abs(k_res) * 1.0e5,
                    search_tol, std::abs(k_res) / std::max(search_tol, 1.0e-30),
                    schedule.search_current_x, schedule.search_iteration, stall_events);
        }

        if (has_search && schedule.searchType == SearchType::RODCRIT)
            schedule.rod_step = schedule.search_current_x;

        // BICGCMFD zeroes both counters in the resetIteration() at the top of
        // this function, so what they hold now is this SolveLoop's own total --
        // ReconvergeFlux's sweeps included, since it ran above.
        ctx.telemetry.cmfd_sweeps += ctx.cmfd_solver.innerIterations();
        ctx.telemetry.bicg_iters  += ctx.cmfd_solver.bicgIterations();

        if (trace_sl)
            std::cout << std::format(
                "      [SL] outer+={} th+={} (search={} relax={:.3f} exit={} stalls={} "
                "relapses={})\n",
                total_outer - sl_outer0, total_th - sl_th0, has_search ? 1 : 0,
                ctx.cross_sections.rod_cusping_relaxation(), SolveExitName(exit_reason),
                stall_events, polish_relapses);
    }

    /// Where this run's restart_<step>.h5 goes (plan Rev.4 Sec 7).
    ///
    /// The historical location is the INPUT directory, which is a namespace
    /// collision the moment a batch runs several decks off one input file --
    /// the policy explicitly allows that ("same input file: allowed") and
    /// equally explicitly forbids sharing a restart namespace.  Every deck
    /// would write the same restart_1.h5, and with N host workers they would
    /// interleave inside a single HDF5 file.
    ///
    /// In batch mode the default therefore keys off the OUTPUT path, which the
    /// launcher and main() both guarantee is unique per deck.  Both halves of
    /// that path are used -- directory AND stem -- because Sec 7 also allows
    /// two jobs to share an output PARENT directory, so
    /// `<dir>/<stem>_restart_<step>.h5` is what makes "distinct --raso implies
    /// distinct restart namespace" true.  It is the same derivation
    /// IO::OpenResult already uses for `<stem>_pinpower.csv`.
    /// RASBERY_RESTART_AT_INPUT=1 restores the legacy location for anything
    /// that reads restarts back by their old path.
    /// WP10.1: everything the canonical case key needs beyond the deck.
    ///
    /// THE ENVIRONMENT LIST IS trajectory::kArmEnv, AND THAT IS THE POINT.  It
    /// is already the campaign's answer to "which knobs can move an iteration",
    /// maintained by the trajectory receipt and pinned by its contracts.  A
    /// second list here would be a second answer, and the failure it would
    /// cause is the expensive one: two runs with different physics sharing one
    /// cache entry.  Values are RAW and unparsed, for the same reason the
    /// trajectory receipt reports them raw.
    /// A string that is safe inside a receipt's JSON.  Paths reach the
    /// warm-start receipt, and a Windows path is full of backslashes: a receipt
    /// that emitted them raw would be a receipt no parser could read, which is
    /// the same as no receipt.
    static std::string jsonString(const std::string& text) {
        std::string out;
        out.reserve(text.size());
        for (const char c : text) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) out += ' ';
                    else out += c;
            }
        }
        return out;
    }

    /// WP10.3.  The staged knobs are no longer environment facts, so the key
    /// may not read them from the environment.
    ///
    /// THE DEFECT THIS AVOIDS, stated plainly: `trajectory::kArmEnv` is walked
    /// with getenv() below, and once one process can run a strict case and an
    /// A2 case back to back, both would fold the SAME `RASBERY_STAGED_FLUX_TOL`
    /// string into their key -- so the promoted strict re-run of a screened
    /// candidate would collide with the screening result it exists to replace.
    /// A cache keyed on that would serve the approximation as the acceptance
    /// answer, which is the one thing WP1's contract exists to prevent, reached
    /// through a new door.  So the three per-case knobs are spelled from the
    /// CaseFidelity, in the same `%.17g`-free integer/decimal shape an operator
    /// would have exported, and everything else still comes from the
    /// environment because everything else still IS the environment.
    ///
    /// AND WHY THE RAW STRING IS KEPT WHENEVER NOTHING WAS OVERRIDDEN.  The
    /// key has a second implementation (tools/case_key.py), which reads these
    /// three out of a resolved child environment as RAW TEXT, and
    /// tools/test_case_key_contract.py holds the two to one fixture.  If this
    /// side normalised `50.0` to `50` for every run, the two would part company
    /// on a deck nobody changed.  So: identical to the process default means
    /// identical bytes -- every key this tree has ever computed is unchanged --
    /// and only a case that actually asked for something else gets the
    /// canonical spelling of what it asked for.  A per-case knob has no
    /// environment string to be raw about; %.17g is the same spelling
    /// casekey::appendValue gives a double, so the two languages still agree.
    static std::string armEnvValue(const char* name, const CaseFidelity& fidelity) {
        const CaseFidelity& base = processCaseFidelity();
        const char*         raw  = std::getenv(name);
        const std::string   text = raw != nullptr ? std::string(raw) : std::string();
        const auto spell = [](double value) { return std::format("{:.17g}", value); };
        if (std::strcmp(name, "RASBERY_STAGED_FLUX_TOL") == 0) {
            if (fidelity.staged_flux_mult == base.staged_flux_mult) return text;
            return fidelity.staged_flux_mult > 1.0 ? spell(fidelity.staged_flux_mult)
                                                   : std::string();
        }
        if (std::strcmp(name, "RASBERY_STAGED_XE_TOL") == 0) {
            if (fidelity.staged_xe_mult == base.staged_xe_mult) return text;
            return fidelity.staged_xe_mult > 1.0 ? spell(fidelity.staged_xe_mult)
                                                 : std::string();
        }
        if (std::strcmp(name, "RASBERY_STAGED_LOOSE_SETTLE") == 0) {
            if (fidelity.loose_settle == base.loose_settle) return text;
            return fidelity.loose_settle ? std::string("1") : std::string();
        }
        return text;
    }

    static casekey::Provenance caseKeyProvenance(const IO& input_output,
                                                 const std::string& warm_provenance,
                                                 const CaseFidelity& fidelity) {
        casekey::Provenance p;
        p.deck_digest = input_output.deck_key_digest();
        // WP10.2.  A warm start is N1 -- it can select a root where the
        // Xe<->flux map has more than one -- so a warm-started answer is not
        // interchangeable with a cold one and must not share its cache entry.
        // The token is the parent state's CONTENT digest, and it is empty
        // whenever the warm start was refused, because a refused warm start
        // produced a cold run and should key like one.
        p.warm_start = warm_provenance;
        // WP10.3: the CASE's fidelity, not the process's.  With no per-case
        // request these are the same two strings effectivePhysicsFidelity()
        // produced, because CaseFidelity defaults from the same environment.
        p.fidelity = fidelity.physicsFidelity();
        p.policy   = fidelity.policy();
        // The library's CONTENT, not its path: two decks naming the same file
        // through different mount points are the same case, and two files at
        // one path across a library update are not.
        //
        // XsLibraryContentDigest, NOT BatchLightResult::Sha256FileCached.  Both
        // stream the same bytes through the same Sha256, so the key bytes are
        // identical while the file is; they differ only in WHEN they stop
        // looking.  Sha256FileCached memoises by PATH ALONE and never expires,
        // and the evaluator server (WP8.1.5) is now a process that outlives a
        // library rebuild -- it would have gone on serving the digest of a file
        // that is no longer there, and stamped a stale key onto answers computed
        // from the new one.  XsLibraryContentDigest memoises by
        // (path, size, mtime) and is the SAME digest the library cache keys on,
        // so the case key and the cache it addresses cannot disagree.
        if (!input_output.xs_path().empty())
            p.xslib_digest = XsLibraryContentDigest(input_output.xs_path());
        for (const char* name : trajectory::kArmEnv)
            p.env.emplace_back(name, armEnvValue(name, fidelity));
        return p;
    }

    static std::string RestartPath(const IO& input_output, int step_number) {
        const char* at_input = std::getenv("RASBERY_RESTART_AT_INPUT");
        const bool  legacy =
            (at_input != nullptr && *at_input != '\0' && std::string(at_input) != "0");
        const std::string result_dir  = input_output.result_dir();
        const std::string result_stem = input_output.result_stem();
        if (legacy || rasberyBatchWidth() <= 0 || result_dir.empty() || result_stem.empty())
            return input_output.input_dir() + std::format("restart_{}.h5", step_number);
        return result_dir + std::format("{}_restart_{}.h5", result_stem, step_number);
    }

public:
    explicit Driver(const std::string& input, const std::string& result_output = "",
                    ResultMode result_mode = BatchLightResult::DefaultMode())
        : _input(input),
          _result_output(result_output),
          _result_mode(result_mode) {
    }

    /// WP10.2.  `from` seeds this case's BOC flux, boron and k_eff from a
    /// parent's saved warm state; `save_to` writes this case's own BOC state
    /// there for a child.  Either may be empty, and with both empty nothing in
    /// the warm-start path runs at all -- feature-off is byte identity.
    ///
    /// PER DRIVER, not per process, and for the same reason ResultMode is: one
    /// wave carries many candidates, and each has its own parent.
    void setWarmStart(std::string from, std::string save_to) {
        _warm_start_from = std::move(from);
        _warm_state_out  = std::move(save_to);
    }

    /// WP10.3.  The resolved per-case fidelity -- staged tolerances, the loose
    /// settle gate and the burnup grid.  Set it BEFORE Drive(): the grid is
    /// applied inside ReadInput and the tolerances are copied into
    /// SolverContext at the top of the solve, so a later change would be a
    /// change the receipt already contradicted.
    ///
    /// The caller is expected to have run it through resolveCaseFidelity(),
    /// which is where the declared-vs-solved equality lives; this setter takes
    /// a resolved value and does not re-judge it, because two judges would
    /// eventually disagree and the receipt would then be a coin toss.
    void setCaseFidelity(CaseFidelity fidelity) { _fidelity = std::move(fidelity); }

    [[nodiscard]] const CaseFidelity& caseFidelity() const { return _fidelity; }

    /// What the last Drive() folded.  Valid only after Drive() returns.
    [[nodiscard]] const CaseReceipt& caseReceipt() const { return _case_receipt; }

    int Drive() {
        const auto driver_start = std::chrono::steady_clock::now();
        // Phase 2 statepoint telemetry (plan Rev.4 Sec 8).  One static-local
        // bool read; every cost beyond the plain counters hangs off it.
        const bool sp_telem = sptelem::enabled();
        // WP9-A: the XS phase mirror (XSTiming.h) is armed from THIS gate and
        // nowhere else, so RASBERY_STATEPOINT_TELEMETRY keeps exactly one
        // reader in the tree and XSSet cannot form a second opinion about what
        // the run was asked for.
        xsphase::armLocalWall(sp_telem);
        // Inner GA evaluations only need scalar fitness/safety receipts.  Full
        // HDF5/restart/pin output remains the default and is used for selected
        // exact cases; light mode avoids queueing mutable Geometry/Schedule
        // references while preserving every scalar computed below.
        const bool light_result = (_result_mode == ResultMode::Light);
        // pin-off drops the ~119 MB/case pin-power CSV and the fmap flux
        // reconstruction.  It does NOT skip PPR: Fq and FdH are PPR outputs and
        // the GA reads both, so the reconstruction runs and only the printing
        // stops.  Applied after ReadInput, over every schedule entry, because
        // the deck states it per entry ("pin-wise information").
        const bool pin_off = (_result_mode == ResultMode::PinOff);

        // 1. Build solver objects and read input deck
        //
        // The four objects that ARE the case, in one named lifetime.  They were
        // four stack locals here; CaseContext is the same four, in the same
        // order, with the boundary that `--evaluator` has to respect written
        // down beside them (EvaluatorContext.h).  The references below keep the
        // rest of this function reading exactly as it did.
        CaseContext case_state;
        Geometry&   geometry       = case_state.geometry;
        Scheduler&  scheduler      = case_state.scheduler;
        XSSet&      cross_sections = case_state.cross_sections;
        IO&         input_output   = case_state.input_output;
        // Deck + XSLIB parse, split out of Init+IO so the Amdahl model's T_fixed
        // can be separated from the XSLIB-cache track (plan Rev.4 Sec 14).
        const auto library_start = std::chrono::steady_clock::now();
        // WP10.3.  THE BURNUP GRID IS APPLIED HERE AND NOWHERE ELSE, because
        // ReadInput folds the deck's canonical key digest immediately after the
        // parse and the coarse deck must be the deck that digest is taken of --
        // otherwise a ten-statepoint screening answer and a thirty-five
        // statepoint acceptance answer share a case key.  Empty is the deck as
        // written, and is one string compare (StatepointGrid.h::isFullGrid).
        input_output.ReadInput(_input, _fidelity.statepoint_grid);
        if (pin_off) {
            for (auto& entry : scheduler.schedule()) {
                entry.print_opt.pin_info = false;
                entry.print_opt.pin_flux = false;
            }
        }
        const double library_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - library_start).count();

        BICGCMFD cmfd_solver(geometry, cross_sections);
        cmfd_solver.setNcmfd(5);
        cmfd_solver.setEpsl2(1.0e-6);
        cmfd_solver.setEshift(0.04);

        Nodal nodal_solver(geometry, cross_sections);
        PPR   pin_power_reconstruction(geometry, cross_sections);

        SolverContext ctx{geometry, cross_sections, cmfd_solver, nodal_solver, SearchMemory{}};
        // WP10.3.  ONE copy, here, before any solve: SolveLoop is static and
        // reads its convergence policy off the context it is handed, so this
        // assignment is the whole of "this case converges the way this case was
        // asked to".  It is a struct of two doubles, a bool and three short
        // strings -- there is no path where copying it per case is measurable
        // beside a deck parse.
        ctx.fidelity = _fidelity;

        // ===================================================================
        // Rev.7.1 Task 9 link 1: stand the device outer segment up, ONCE.
        // ===================================================================
        //
        // HERE, AND NOT EARLIER, because this is the first point at which the
        // deck is fully known: ReadInput has run, so the geometry dimensions and
        // the CMFD topology cache are final, and the arena's ONE allocation can
        // be sized without guessing.  It is also before any solve, which is the
        // other half of the arena's contract -- every address it hands out is
        // fixed for the run, so nothing may allocate after a graph could exist.
        //
        // WITH THE FEATURE OFF THIS IS ONE PREDICATE AND A RETURN.  The gate is
        // inside rasberyStandUpOuterSegment (a cached env read), so the OFF path
        // pays a call that returns false and nothing else -- no geometry scan,
        // no VRAM query, and above all no allocation.
        //
        // THE FUEL-NODE COUNT IS SCANNED, not read, because Geometry keeps the
        // flag per node and no count beside it.  nxyz loads once per run, next
        // to an HDF5 parse; a cached count that could disagree with the flags
        // would be the more expensive mistake.
        // WP10.7: witnessed by the admission door below, so a stand-up that
        // refused cannot look like one that never ran.  True with the arm off
        // -- nothing was promised, so nothing failed to be established.
        bool outer_stood_up = !gpu::outerGpuEnabled();
        if (gpu::outerGpuEnabled()) {
            gpu::OuterSegmentDeck deck;
            deck.nxyz  = geometry.nxyz();
            deck.nsurf = geometry.nsurf();
            deck.nxy   = geometry.nxy();
            deck.ng    = geometry.ng();
            int n_fuel = 0;
            for (int l = 0; l < geometry.nxyz(); ++l)
                if (geometry.IsFuel(l)) ++n_fuel;
            deck.n_fuel = n_fuel;

            deck.surface_node    = cmfd_solver.surfaceNodeData();
            deck.surface_dir     = cmfd_solver.surfaceDirData();
            deck.node_hmesh      = cmfd_solver.nodeHmeshData();
            deck.node_volume     = cmfd_solver.nodeVolumeData();
            deck.boundary_albedo = cmfd_solver.boundaryAlbedoData();

            deck.dhat_clamp = cmfd_solver.dhatClampEnabled();
            // Link 5: the counter that already moves on every host `_xs` write.
            deck.material_generation = cross_sections.hoststateGeneration();

            // The arena receipt goes to stdout beside every other [RASBERY]
            // receipt; a VRAM admission that only appeared on failure would be
            // the one number nobody could quote when a run did fit.
            //
            // WP10.7.  THE RETURN IS NO LONGER DISCARDED.  A stand-up that
            // refuses -- an arena Sec 4.4 admission that did not fit, a runner
            // that would not initialise, a deck whose shape is not the one the
            // arena was stood on -- used to be answered later and generically
            // by the post-arm ladder, and the reason it printed to stderr on
            // the way out was the only copy.
            outer_stood_up = gpu::rasberyStandUpOuterSegment(deck, std::cout);
        }

        // WP10.7.  THE ADMISSION DOOR.  Unconditional, once per admission, and
        // BEFORE InitXS -- which is the call whose UpdateFlatXS was the first
        // and only thing that ever asked whether this case's flat-XS device
        // residency existed.  See establishDeviceResidency above for the run
        // that made this necessary and for why the promote step was three of
        // its four deaths.
        establishDeviceResidency(cross_sections, outer_stood_up, std::cout);

        // 2. Initialize run state
        const bool is_restart_run = input_output.has_restart() && !input_output.has_shuffle();
        const bool is_shuffle_run = input_output.has_shuffle();

        double eigv = 1.0;
        double efpd = 0.0;
        if (input_output.has_restart()) efpd = input_output.restart_efpd();

        auto& initial_schedule = scheduler.schedule(0);
        cross_sections.InitXS(initial_schedule.bppm,
                              initial_schedule.tful,
                              initial_schedule.tmod,
                              initial_schedule.pressure,
                              0.0, !is_restart_run);

        // Frozen mode + a deck carrying no in-core Xe: warn once, here and
        // nowhere else.  This is the only moment "the incoming inventory" is
        // well defined -- the library values are in, a restart's or a shuffle's
        // values have replaced them where they apply, and no schedule entry has
        // run yet, so nothing has overwritten the Xe rows.  See
        // WarnFrozenXeIfEmpty for why it used to be called from SolveLoop and
        // why that was the wrong place.
        WarnFrozenXeIfEmpty(ctx, initial_schedule, is_restart_run || is_shuffle_run);

        if (!is_restart_run)
            cross_sections.ResetFluxAndCurrents(1.0);

        // ===================================================================
        // WP10.2 -- the warm start, applied HERE and nowhere else.
        // ===================================================================
        //
        // AFTER ResetFluxAndCurrents, because that is what writes the cold
        // guess this replaces; BEFORE the first SolveLoop, because the bucket
        // it aims at (`initial`) is that solve.  A restart run is left alone:
        // it already carries a converged flux for THIS core, which is a better
        // seed than any sibling's and is an input rather than a guess.
        //
        // EVERY REFUSAL DEGRADES TO A COLD START.  A missing file, a foreign
        // magic, a version bump, a geometry that does not match -- each returns
        // a reason, the flux keeps the cold guess, and the receipt says which.
        // A warm start that cannot be honoured must never become a wrong one.
        std::string      warm_provenance;
        std::string      warm_status      = "off";
        std::string      warm_reason;
        std::string      warm_save_status = _warm_state_out.empty() ? "off" : "pending";
        std::string      warm_save_reason;
        bool             warm_saved       = false;
        long long        warm_initial_outers = 0;
        // WP9-D stage D.  Run totals of the search classification, folded
        // UNCONDITIONALLY for exactly the reason `warm_initial_outers` is: the
        // wall-timing arm runs with RASBERY_STATEPOINT_TELEMETRY unset (plan
        // Sec 6.4 keeps the timing arm and the telemetry arm apart), and a
        // lever whose before/after can only be read on the OTHER arm is a lever
        // nobody can price.  Nine integer adds per statepoint; read by one
        // receipt and by nothing in the solve.
        struct SearchLedger {
            long long trials = 0, proposals = 0, refused = 0, probe = 0, carry = 0;
            long long extrap = 0, secant = 0, bisect = 0, outers = 0;
        } sp_search{};
        warmstate::State warm{};
        if (!_warm_start_from.empty()) {
            warm_status = "cold_fallback";
            warmstate::State parent{};
            warm_reason = warmstate::load(_warm_start_from, parent);
            if (warm_reason.empty()) {
                warmstate::State here{};
                here.ng   = static_cast<std::uint32_t>(geometry.ng());
                here.nxyz = static_cast<std::uint32_t>(geometry.nxyz());
                here.nx   = static_cast<std::uint32_t>(geometry.nx());
                here.ny   = static_cast<std::uint32_t>(geometry.ny());
                here.nz   = static_cast<std::uint32_t>(geometry.nz());
                if (!parent.shapeMatches(here)) {
                    warm_reason = std::format(
                        "geometry mismatch: parent ng={} nxyz={} ({}x{}x{}) vs "
                        "this case ng={} nxyz={} ({}x{}x{})",
                        parent.ng, parent.nxyz, parent.nx, parent.ny, parent.nz,
                        here.ng, here.nxyz, here.nx, here.ny, here.nz);
                } else if (!std::isfinite(parent.keff) || parent.keff <= 0.1 ||
                           parent.keff >= 3.0 || !std::isfinite(parent.boron) ||
                           parent.boron < 0.0) {
                    // A non-physical seed is worse than no seed: it would send
                    // the first search trial somewhere the secant has to walk
                    // back from.  Sec 10.2 asks for exactly this refusal.
                    warm_reason = std::format("implausible seed: keff={} boron={}",
                                              parent.keff, parent.boron);
                } else {
                    std::copy(parent.flux.begin(), parent.flux.end(),
                              geometry.PhifMutable());
                    // The boron seed reaches the search through the same door
                    // the deck's does: StartCriticalSearch reads bppm(0).
                    cross_sections.SetBoron(parent.boron);
                    // WP9-D stage D, candidate D2's solver half.  The door
                    // above is the only one a deck that names no
                    // `search_boron_ppm` needs; a deck that DOES name one
                    // overrides the parent, and RASBERY_SEARCH_WARM_BORON is
                    // what lets the measurement beat the campaign default.
                    // Gated on the flag so an unset knob stores nothing.
                    if (ctx.search_policy.warm_boron) {
                        ctx.search_carry.has_warm_boron = true;
                        ctx.search_carry.warm_boron     = parent.boron;
                    }
                    eigv        = parent.keff;
                    warm_status = "applied";
                    warm        = parent;
                }
            }
            // CONTENT, not path: two runs can name one path and mean different
            // files, and the case key has to tell them apart.
            warm_provenance = warmstate::digest(_warm_start_from);
            if (warm_status != "applied") warm_provenance.clear();
        }

        if (is_restart_run)
            std::cout << std::format("  [RESTART] Continuing from '{}'\n", input_output.restart_path());
        else if (is_shuffle_run)
            std::cout << std::format("  [SHUFFLE] New cycle, {} shuffle spec(s) applied\n", input_output.restart_files().size());

        std::string result_path = _result_output;
        if (result_path.empty())
            result_path = input_output.input_dir() + "result.h5";

        const std::filesystem::path result_file_path(result_path);
        if (result_file_path.has_parent_path())
            std::filesystem::create_directories(result_file_path.parent_path());

        if (!light_result)
            input_output.OpenResult(result_path);

        const double init_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - driver_start).count();
        std::cout << std::format("  [TIMING] Init+IO={:.3f} s\n", init_seconds);

        // ===================================================================
        // WP10.1 -- the canonical duplicate key, and the receipt that names it.
        // ===================================================================
        //
        // HERE, because this is the first point where all four halves exist:
        // the deck digest (IO folded it at parse time, with the loading pattern
        // canonicalised under the core's symmetry), the effective fidelity, the
        // arm environment, and the cross-section library's CONTENT digest.
        //
        // UNCONDITIONAL, like the trajectory receipt and for the same reason: a
        // key that only some runs computed is a key no cache could trust, and
        // "was this case the one that produced that answer" is a question a run
        // has to be able to answer without having been asked in advance.
        //
        // COST.  One SHA-256 of a ~1 kB payload, plus the XS library digest --
        // which is memoised per process by (path, size, mtime)
        // (XsLibraryContentDigest), so a 64-case batch
        // pays the 34 MB read once and every later case reads the cache.  The
        // deck digest itself was already folded during the parse.
        //
        // THE PROVENANCE IS A LOCAL NOW, AND THE RECEIPT NAMES ITS COMPONENTS.
        // WP10.1's first live gate (host 181, 2026-08-30) failed on
        // kngr_238.json with the solver and tools/case_key.py disagreeing about
        // the key -- and there was NOTHING TO READ.  One digest on each side and
        // six inputs behind it: the deck half, the fidelity pair, the env half,
        // the library's content, the warm-start token and the build identity.
        // The deck half already had its own digest in this line, which is why
        // it was the only component the gate could clear; the other four were
        // guesses.  So each of them gets a component of its own here, and
        // `tools/case_key.py --components` prints the same five keys with the
        // same spellings.  A mismatch is then one `diff` and not a bisect.
        const casekey::Provenance case_provenance =
            caseKeyProvenance(input_output, warm_provenance, _fidelity);
        const std::string case_key         = casekey::keyOf(case_provenance);
        const std::string case_env_digest  = casekey::envDigest(case_provenance);
        _case_receipt.case_key            = case_key;
        // WP10.3.  The per-case fidelity, carried as a value for the same
        // reason the digest is: a mixed wave's sixty-four answers interleave on
        // stdout and the evaluator's per-case receipt has to name its OWN.
        _case_receipt.policy              = _fidelity.policy();
        _case_receipt.physics_fidelity    = _fidelity.physicsFidelity();
        _case_receipt.statepoint_grid     = _fidelity.gridToken();
        _case_receipt.fidelity_declared   = _fidelity.declared;
        _case_receipt.promoted_from       = _fidelity.promoted_from;
        _case_receipt.acceptance_eligible = _fidelity.acceptanceEligible();
        // WP10.4.  `physics_fidelity` IS `fidelity`, PRINTED TWICE ON PURPOSE.
        // The evaluator's [RASBERY][EVALUATOR][CASE] line spells this value
        // `physics_fidelity` (plan Sec 6.2); this line spelled it `fidelity`
        // (the campaign shorthand tools/case_key.py keys the case on), and
        // tools/exact_audit.py CASE_REQUIRED_FIELDS audits BOTH tags for the
        // Sec 6.2 spelling.  So every Driver-side receipt was reported as
        // "this binary predates WP10.3" -- 83 of the 86 findings in the host
        // 181 soak at 91004f7, on a binary that had carried the field all
        // along under the other name.  The old key STAYS: the case key is
        // computed from it on both sides (tools/case_key.py COMPONENT_FIELDS)
        // and dropping it would invalidate every manifest on disk.  The new
        // one is the name the audit reads.
        //
        // WP10.5.  `output` IS THE PER-CASE IDENTIFIER, and this line had none.
        // It carried `case_key`, which is the CANONICAL DUPLICATE key -- and in
        // a width-16 cold wave of one deck at one fidelity every case has the
        // SAME one, by design.  So a mixed-fidelity audit, which resolves a
        // receipt to the word its case was declared at, could not resolve this
        // tag at all: the 55c0dce soak on 181 reported 82 cases as "no fidelity
        // was declared for it" purely because the Driver's receipt named no
        // case a declarer could have keyed on.  The `--raso` path is that name.
        // It is unique per case by the evaluator's own wave namespace rule, it
        // exists in every mode (argv, `--jobs` manifest, evaluator) -- unlike
        // the evaluator's optional client `key` -- and the Driver already has
        // it, so nothing is plumbed to print it.  schema_version 6.
        std::cout << std::format(
            "  [RASBERY][CASE] {{\"schema_version\":6,\"case_key\":\"{}\",\"key_schema\":\"{}\","
            "\"core_op\":\"{}\",\"deck_digest\":\"{}\",\"env_digest\":\"{}\","
            "\"env_set\":\"{}\","
            "\"xslib_digest\":\"{}\",\"xslib_policy\":\"{}\",\"warm_start_token\":\"{}\","
            "\"code_sha\":\"{}\",\"fidelity\":\"{}\",\"physics_fidelity\":\"{}\","
            "\"policy\":\"{}\","
            "\"result_mode\":\"{}\",\"output\":\"{}\","
            "\"warm_start\":\"{}\",\"statepoint_grid\":\"{}\","
            "\"acceptance_eligible\":{},\"fidelity_declared\":{},\"promoted_from\":{}}}\n",
            case_key, casekey::kSchema, input_output.deck_key_core_op(),
            input_output.deck_key_digest(), case_env_digest,
            casekey::envSetToken(case_provenance),
            casekey::tokenOrTilde(case_provenance.xslib_digest),
            XsLibraryDigestPolicyName(),
            jsonString(casekey::tokenOrTilde(case_provenance.warm_start)),
            jsonString(casekey::codeShaToken()),
            _case_receipt.physics_fidelity, _case_receipt.physics_fidelity,
            _case_receipt.policy,
            ResultModeName(_result_mode), jsonString(result_path),
            warm_status, _case_receipt.statepoint_grid,
            _case_receipt.acceptance_eligible ? "true" : "false",
            _fidelity.declared.empty()
                ? std::string("null")
                : "\"" + jsonString(_fidelity.declared) + "\"",
            _fidelity.promoted_from.empty()
                ? std::string("null")
                : "\"" + jsonString(_fidelity.promoted_from) + "\"");

        // Receipt keys (plan Rev.4 Sec 8.1).  result_stem() is empty until
        // OpenResult(), and light-result runs never call it, so fall back to the
        // stem of the path this deck was told to write -- the same string that
        // makes a job's output namespace unique (Sec 7).  Built once, and only
        // when the telemetry is on: nothing is allocated on the default path.
        std::string sp_job_id;
        int         sp_slot = -1;
        sptelem::Counters sp_run;
        // The trajectory fold (always on; see the namespace comment).  Declared
        // beside the telemetry accumulator and armed by nothing: a receipt whose
        // presence depended on a flag could not answer a question ABOUT flags.
        trajectory::Digest sp_traj;
        if (sp_telem) {
            sp_job_id = input_output.result_stem();
            if (sp_job_id.empty())
                sp_job_id = std::filesystem::path(result_path).stem().string();
            sp_slot = cmfd_solver.batchSlot();
        }

        // 3. Main schedule loop.  The bound is re-read every iteration because a
        // depletion entry carrying until_boron_ppm re-queues itself (natural EOC);
        // decks without that key never grow the vector, so their path is unchanged.
        double    total_io_seconds     = 0.0;
        int       natural_eoc_inserts  = 0;
        // Statepoints whose PPR ran on the host.  With RASBERY_GPU_PPR unset
        // that is all of them and the [PPR_GPU] receipt is not printed; with it
        // set, any non-zero value is a fallback and the receipt says so.
        long long ppr_host_statepoints = 0;

        for (int step_index = 0; step_index < static_cast<int>(scheduler.schedule().size()); ++step_index) {
            auto& schedule = scheduler.schedule(step_index);
            const auto step_start = std::chrono::steady_clock::now();
            // WP9-D: re-arm the search history HERE, at the statepoint boundary
            // and not at SolveLoop entry.  A depletion statepoint with substeps
            // enters SolveLoop several times, and clearing per entry would
            // report the last substep's search as the whole statepoint's.  The
            // natural-EOC block re-queues a COPY of a schedule entry, so a
            // re-queued statepoint would otherwise inherit its parent's tallies.
            schedule.ResetSearchTelemetry();

            schedule.PrepareForStep(cross_sections.CoreHeavyMetalMassKg());
            schedule.ApplyToGeometry(geometry);

            const double power_fraction = schedule.powerFraction();
            const double thermal_power  = schedule.thermalPower();
            const double step_dt        = schedule.time * 86400.0;
            cross_sections.SetPowerRate(power_fraction);
            efpd += schedule.time * power_fraction;
            // Diagnostics only; see SolverContext.
            ctx.statepoint = step_index + 1;
            ctx.efpd       = efpd;

            int        total_outer = 0;
            int        total_th    = 0;
            // Arm this statepoint's telemetry.  The backend tallies are
            // cumulative, so everything device-side is published as a delta.
            {
                const BackendCounters base =
                    sp_telem ? cmfd_solver.backendCounters() : BackendCounters{};
                ctx.telemetry.begin(base.graph_launches,
                                    base.bulk_h2d_bytes_during_iteration,
                                    base.bulk_d2h_bytes_during_iteration,
                                    base.bulk_d2h_calls_during_iteration);
            }
            const bool keep_search = schedule.keepSearchBetweenSolves();
            // Only the first statepoint of a cycle that inherited its state from another
            // run starts with a decayed Xe field, and only there does the Xe<->flux map
            // admit the spurious axial root (see SolveLoop's xe_relax initializer).  A
            // run that starts cold is untouched, so its path stays bit-for-bit as before.
            const bool prime_xe = (step_index == 0) && (is_restart_run || is_shuffle_run);

            // Pre-work by schedule type
            if (schedule.type == ScheduleType::DEPLETION) {
                const int    nsub   = std::max(1, schedule.substep);
                const double sub_dt = step_dt / nsub;
                for (int isub = 0; isub < nsub; ++isub) {
                    // Predictor (BOS) solve. The first substep reuses the flux carried from the
                    // previous step's final solve, which already converged this exact composition
                    // (BOS_k == EOS_{k-1}); only later substeps need a fresh BOS re-solve.
                    if (isub > 0) {
                        SolveLoop(ctx, eigv, schedule, total_outer, total_th, keep_search, prime_xe);
                        cross_sections.NormalizeFluxSign();
                    }
                    {
                        outer_timing::Scope pred(sptelem::PH_DEPL_PRED);
                        cross_sections.PredictorStep(sub_dt, thermal_power, schedule.xenon_transient);
                    }
                    SolveLoop(ctx, eigv, schedule, total_outer, total_th, keep_search, prime_xe);
                    cross_sections.NormalizeFluxSign();
                    {
                        outer_timing::Scope corr(sptelem::PH_DEPL_CORR);
                        cross_sections.CorrectorStep(sub_dt, thermal_power, schedule.xenon_transient);
                    }
                }
            }

            if (schedule.type == ScheduleType::DERIVATIVE) {
                cross_sections.UpdateDerivative(schedule.delta_bppm,
                                                schedule.delta_tful,
                                                schedule.delta_tmod,
                                                schedule.delta_dmod);
            }

            if (schedule.type == ScheduleType::ROD) {
                cross_sections.SetRod(schedule.rod_insertions);
                cross_sections.ResetFluxAndCurrents(1.0);
                cmfd_solver.resetDhat();
                eigv = 1.0;
            }

            // Final solve
            const double eigv_before = eigv;
            SolveLoop(ctx, eigv, schedule, total_outer, total_th, keep_search, prime_xe);
            cross_sections.NormalizeFluxSign();

            // Zero-time DERIVATIVE solves reuse the carried flux; on rare states the eigen
            // iteration collapses (k -> ~0 -> NaN) and, with no reset in the chain, poisons
            // every later zero-time solve until the next rod op. A zero-time state delta
            // cannot halve/double k, so treat that (or a non-finite k) as numerical collapse:
            // reset flux + d-hat and re-solve once from a clean state.
            if (schedule.type == ScheduleType::DERIVATIVE &&
                (!std::isfinite(eigv) || eigv < 0.5 * eigv_before || eigv > 2.0 * eigv_before)) {
                cross_sections.ResetFluxAndCurrents(1.0);
                cmfd_solver.resetDhat();
                eigv = 1.0;
                SolveLoop(ctx, eigv, schedule, total_outer, total_th, keep_search, prime_xe);
                cross_sections.NormalizeFluxSign();
            }

            // WP9-A: the statepoint's iteration is over.  Everything from here
            // to the receipt below is FLOOR work -- PPR, result packing, the
            // restart save -- so the transfer counters are re-baselined here
            // and `floor_transfer` is the host-copy bill of the boundary alone.
            // One struct read per statepoint, behind the telemetry gate.
            if (sp_telem) {
                const BackendCounters at_floor = cmfd_solver.backendCounters();
                ctx.telemetry.armFloor(at_floor.bulk_h2d_bytes_during_iteration,
                                       at_floor.bulk_h2d_calls_during_iteration,
                                       at_floor.bulk_d2h_bytes_during_iteration,
                                       at_floor.bulk_d2h_calls_during_iteration);
            }

            // PPR
            // Corner-balance iteration cap.  The loop exits early on its own
            // corner-flux tolerance; measured on KNGR CY1 it needs ~50 rounds,
            // and the historical cap of 5 shipped an unconverged reconstruction
            // (pin power vs MASTER: 4.76% rms at cap 5 -> 0.84% converged, with
            // identical results from 50 to 200).  The cap is a safety bound, so
            // set it far above the measured need; cost is ~0.1 s per printed
            // statepoint.  RASBERY_PPR_ITERS overrides for studies.
            static const int ppr_iters = []() {
                const char* e = std::getenv("RASBERY_PPR_ITERS");
                const int   v = e ? std::atoi(e) : 100;
                return v > 0 ? v : 100;
            }();
            // GA evaluator plan Sec 6.3 Task 10.  RASBERY_GPU_PPR=1 runs
            // reset()+drive() as one device sequence; anything else -- arm off,
            // no CUDA, ng != 2, a CUDA failure -- returns false having
            // touched nothing (naming which, in the receipt), and the two host calls run
            // exactly as they did before.  The fused device call is charged to
            // ppr_drive, so `ppr_reset + ppr_drive` is the like-for-like
            // comparison between the two arms and `ppr_reset == 0` is how the
            // receipt says which one ran.
            //
            // WP6 stage C.  THE OFFER IS MADE PER STATEPOINT, AND WITHDRAWN THE
            // SAME WAY.  The arena's per-slot addresses are fixed for the run,
            // but whether the SEGMENT is still the thing that wrote them is not:
            // a statepoint whose outer ran on the host leaves the device buffers
            // an outer stale, and PPR reading them would blend two iterations
            // with every value finite.  `canonicalNodalBound()` is the segment's
            // own answer to "did a backend adopt this set and is it live", which
            // is why it -- and not the mere existence of the pointers -- is the
            // condition.  A default-constructed offer WITHDRAWS: silence would
            // leave the previous statepoint's standing.
            {
                PprCanonicalInputs       ppr_canon;
                const ppr::CanonicalMode ppr_canon_mode = ppr::canonicalModeFromEnv();
                if (ppr_canon_mode != ppr::CanonicalMode::Off) {
                    // OuterSlotClaim::query()'s rule, spelled the same way: -1
                    // (no resident CMFD) asks segment 0, which is the segment the
                    // single-instance path binds.
                    const int resident = cmfd_solver.residentSlot();
                    auto&     seg = gpu::rasberyOuterSegment(resident >= 0 ? resident : 0);
                    if (seg.canonicalNodalBound()) {
                        const gpu::CanonicalSlotBuffers set = seg.canonicalNodalSet();
                        ppr_canon.mode = ppr_canon_mode;
                        ppr_canon.phif = set.flux;
                        ppr_canon.phis = set.phis;
                        ppr_canon.jnet = set.jnet;
                    }
                }
                pin_power_reconstruction.adoptCanonicalDeviceInputs(ppr_canon);
            }
            bool ppr_on_device = false;
            {
                outer_timing::Scope ppr_drive_scope(sptelem::PH_PPR_DRIVE);
                ppr_on_device = pin_power_reconstruction.resetAndDriveGpu(
                    1.0 / eigv, geometry.Jnet(), geometry.Phif(),
                    geometry.Phis(), ppr_iters);
            }
            // WP1 (plan Sec 6.3).  THE PPR FAIL-OPEN SEAM.  Every reason
            // resetAndDriveGpu can decline -- arm off, no CUDA, ng != 2, a
            // CUDA failure -- lands below, and the two host calls there are the
            // CPU pin-power reconstruction.  The guard is conditioned on the
            // arm so an unset RASBERY_GPU_PPR promises nothing.
            //
            // WP6 stage F.  THE GUARD IS HANDED THE LADDER'S OWN NAME, not a
            // fixed sentence.  `first_violation` used to read "the device PPR
            // arm declined" for every one of the seven reasons, which is the
            // same defect as `host_fallbacks:35` with no reason beside it: it
            // told a reader that something refused and never which one.  The
            // string is the enum's own name(), so GpuFullContract's receipt and
            // the [RASBERY][PPR_GPU] ladder cannot drift apart.
            if (!ppr_on_device) {
                ++ppr_host_statepoints;
                RASBERY_GPU_FULL_GUARD_IF(
                    rasbery::gpufull::armRequested("RASBERY_GPU_PPR"), Ppr,
                    "Driver: statepoint PPR",
                    pin_power_reconstruction.gpu().lastRefusalName());
                {
                    outer_timing::Scope ppr_reset_scope(sptelem::PH_PPR_RESET);
                    pin_power_reconstruction.reset(1.0 / eigv, geometry.Jnet(),
                                                   geometry.Phif(), geometry.Phis());
                }
                {
                    outer_timing::Scope ppr_drive_scope(sptelem::PH_PPR_DRIVE);
                    pin_power_reconstruction.drive(ppr_iters);
                }
            }
            // MASTER reports pin-volume-averaged reconstructed power.  A pin-centre
            // sample biases Fq high once intra-node curvature grows during burnup,
            // so use the precomputed 3x3 Gauss-Legendre pin-area integration.
            {
                outer_timing::Scope ppr_recon_scope(sptelem::PH_PPR_RECON);
                // WP6 stage D's third argument: WILL THE HOST READ THE PIN MAP.
                // Geometry::PinPower() has exactly one reader -- IO.cpp's
                // `if (d.print_opt.pin_info)` block -- so the flag that decides
                // whether the map is written to HDF5 is the same flag that
                // decides whether it has to leave the device.  Passing anything
                // else here (a constant `true`, or a different predicate) would
                // either ship 5.4 MB per statepoint for nobody or write a step
                // out of a stale array; there is no third correct value.  The
                // HOST path ignores it -- it computes in place.
                pin_power_reconstruction.reconstructPinPower(
                    true, schedule.print_opt.pin_flux, schedule.print_opt.pin_info);
            }

            // Output
            const int step_number = step_index + 1;
            schedule.eigv         = eigv;
            schedule.rho          = (eigv > 1.0e-12) ? (eigv - 1.0) / eigv : 0.0;
            schedule.ppm          = geometry.bppm(0);
            const double step_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
            std::cout << std::format("  NO.={:4d}  EFPD={:10.3f}  K-EFF={:.6f}  PPM={:8.2f}  outer={:3d}  TH={:2d}  t={:5.2f}s\n",
                                     step_number, efpd, eigv, geometry.bppm(0), total_outer, total_th, step_seconds);
            // The same five numbers the line above prints, folded at full
            // precision.  HERE and not later because this is where they are the
            // published values of this statepoint; `step_seconds` is the one
            // field of that line deliberately left out, being the only one that
            // is allowed to differ between two runs of the same arm.
            sp_traj.step(step_number, total_outer, total_th, efpd, eigv, geometry.bppm(0));

            // WP10.2 receipts.  `initial` is the bucket the warm start aims at
            // (GA evaluator plan Sec 2.2: 347 of 4,609 outers), so it is tallied
            // UNCONDITIONALLY -- one add per statepoint -- rather than only
            // under RASBERY_STATEPOINT_TELEMETRY.  A lever whose before/after
            // can only be read on a telemetry run is a lever that cannot be
            // measured on the wall-timing run it is supposed to shorten.
            warm_initial_outers += ctx.telemetry.outers_by_cause[sptelem::CAUSE_INITIAL];

            // WP9-D stage D: the same argument, for the search.  The Schedule
            // owns the classification (that is where `method` is decided) and
            // ctx.telemetry owns the committed-trial count, so the ledger reads
            // each from its owner rather than re-deriving either.
            sp_search.trials    += ctx.telemetry.search_trials;
            sp_search.proposals += schedule.search_n_proposals;
            sp_search.refused   += schedule.search_n_refused;
            sp_search.probe     += schedule.search_n_probe;
            sp_search.carry     += schedule.search_n_carry;
            sp_search.extrap    += schedule.search_n_extrap;
            sp_search.secant    += schedule.search_n_secant;
            sp_search.bisect    += schedule.search_n_bisect;
            sp_search.outers    += ctx.telemetry.outers_by_cause[sptelem::CAUSE_SEARCH];

            // The BOC state, for a child.  FIRST STATEPOINT ONLY: every later
            // one already starts from the previous statepoint's converged flux,
            // which is the best warm start there is, so 34 more files would
            // seed nothing.  Written after the statepoint publishes, so what is
            // saved is what the receipt reported.
            if (!_warm_state_out.empty() && !warm_saved) {
                warm_saved = true;
                warmstate::State out{};
                out.ng   = static_cast<std::uint32_t>(geometry.ng());
                out.nxyz = static_cast<std::uint32_t>(geometry.nxyz());
                out.nx   = static_cast<std::uint32_t>(geometry.nx());
                out.ny   = static_cast<std::uint32_t>(geometry.ny());
                out.nz   = static_cast<std::uint32_t>(geometry.nz());
                out.keff  = eigv;
                out.boron = geometry.bppm(0);
                out.efpd  = efpd;
                out.flux.assign(geometry.Phif(),
                                geometry.Phif() +
                                    static_cast<std::size_t>(out.ng) * out.nxyz);
                const std::filesystem::path save_path(_warm_state_out);
                if (save_path.has_parent_path())
                    std::filesystem::create_directories(save_path.parent_path());
                warm_save_reason = warmstate::save(_warm_state_out, out);
                warm_save_status = warm_save_reason.empty() ? "saved" : "save_failed";
            }

            const auto io_start = std::chrono::steady_clock::now();
            {
                outer_timing::Scope add(sptelem::PH_RESULT_ADD);
                input_output.AddResult(geometry, eigv, step_index, step_number, efpd);
            }

            if (!light_result && schedule.print_opt.save) {
                input_output.SaveRestart(RestartPath(input_output, step_number),
                                         geometry, cross_sections, eigv, efpd, step_number);
            }

            {
                // WP9-A `floor_wall.result_write`: the scalar JSONL line or the
                // HDF5 step write.  Separate from `result_add` because the two
                // answer different questions -- AddResult is the packing every
                // mode pays, this is what the chosen output mode costs on top.
                outer_timing::Scope write_scope(sptelem::PH_RESULT_WRITE);
                if (light_result) {
                    BatchLightResult::Fidelity light_fidelity;
                    light_fidelity.policy              = _case_receipt.policy;
                    light_fidelity.physics_fidelity    = _case_receipt.physics_fidelity;
                    light_fidelity.statepoint_grid     = _case_receipt.statepoint_grid;
                    light_fidelity.declared            = _case_receipt.fidelity_declared;
                    light_fidelity.promoted_from       = _case_receipt.promoted_from;
                    light_fidelity.acceptance_eligible = _case_receipt.acceptance_eligible;
                    BatchLightResult::Write(_input, input_output.xs_path(), case_key,
                                            light_fidelity,
                                            warm_saved ? _warm_state_out : std::string(),
                                            schedule.step, schedule.substep,
                                            schedule.efpd, schedule.bu_avg,
                                            schedule.eigv, schedule.ppm,
                                            schedule.ao, schedule.fqp, schedule.frp,
                                            schedule.search_exit_status,
                                            schedule.search_exit_dk,
                                            schedule.search_exit_tol);
                } else {
                    input_output.WriteStepToResult(geometry, cross_sections, step_index);
                }
            }
            const double step_io_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
            total_io_seconds += step_io_seconds;

            // Statepoint boundary: close the deltas and emit ONE machine-readable
            // line (plan Rev.4 Sec 8.1).  This is the only place the telemetry
            // formats anything, and it runs 35-51 times per run.
            if (sp_telem) {
                const BackendCounters now = cmfd_solver.backendCounters();
                ctx.telemetry.end(now.graph_launches,
                                  now.bulk_h2d_bytes_during_iteration,
                                  now.bulk_h2d_calls_during_iteration,
                                  now.bulk_d2h_bytes_during_iteration,
                                  now.bulk_d2h_calls_during_iteration);
                ctx.telemetry.outers_driver = total_outer;
                // WP9-D: the search history, carried across from the Schedule
                // that owns it.  At the close, so it is this statepoint's whole
                // search including every substep's.
                ctx.telemetry.search_proposals  = schedule.search_n_proposals;
                ctx.telemetry.search_refused    = schedule.search_n_refused;
                ctx.telemetry.search_secant     = schedule.search_n_secant;
                ctx.telemetry.search_carry      = schedule.search_n_carry;
                ctx.telemetry.search_probe      = schedule.search_n_probe;
                ctx.telemetry.search_bisect     = schedule.search_n_bisect;
                ctx.telemetry.search_extrap     = schedule.search_n_extrap;
                ctx.telemetry.search_iterations = schedule.search_iteration;
                ctx.telemetry.wall          = step_seconds;
                ctx.telemetry.io_wall       = step_io_seconds;
                const sptelem::Counters& c  = ctx.telemetry;
                // Batch mode shares ONE arena, and therefore one set of device
                // tallies, across every deck in the process: a delta taken here
                // then includes whatever the other slots did meanwhile.  Say so
                // rather than let the number be read as this deck's alone.
                const bool shared = (sp_slot >= 0);
                // Derived at print, never carried: the cascade resolution of the Xe
                // budget.  Additive fields only, so schema_version stays 1.
                const double xe_per_cascade =
                    c.xe_cascades > 0
                        ? static_cast<double>(c.xe_updates) / static_cast<double>(c.xe_cascades)
                        : 0.0;
                // Through the buffered line sink, not std::cout directly: this
                // is 35-51 lines per deck x 64 decks through a stdio-synced
                // stream, i.e. a lock (and often a write(2)) per line on the
                // very threads the writer thread was freed from.  The sink
                // appends under one mutex, so a line still cannot be split, and
                // flushes per line in inline mode -- same bytes, same order.
                iowriter::appendLine(std::format(
                    "[RASBERY][SPTELEM] {{\"schema_version\":1,\"job_id\":\"{}\",\"slot\":{},"
                    "\"statepoint\":{},\"efpd\":{:.4f},\"outers\":{},\"outers_attributed\":{},"
                    "\"outers_initial\":{},\"xe_mode\":\"{}\","
                    "\"xe_updates\":{},\"xe_interim_updates\":{},"
                    "\"xe_cascades\":{},\"xe_steps_per_cascade\":{:.3f},"
                    "\"xe_budget_exhausted\":{},"
                    "\"xe_once_extra_steps\":{},\"xe_once_trust_trips\":{},"
                    "\"xe_aa_proposed\":{},\"xe_aa_accepted\":{},"
                    "\"xe_aa_rejected\":{},\"xe_aa_history_resets\":{},"
                    "\"xe_outers\":{},\"search_trials\":{},\"search_outers\":{},"
                    "\"th_updates\":{},\"th_outers\":{},\"settle_outers\":{},"
                    "\"fallback_outers\":{},\"staged_relapses\":{},\"th_search_coincident\":{},"
                    "\"flux_limit_retries\":{},\"solve_loops\":{},\"cmfd_sweeps\":{},"
                    "\"bicg_iters\":{},\"graph_launches_delta\":{},\"h2d_bytes_delta\":{},"
                    "\"d2h_bytes_delta\":{},\"d2h_calls_delta\":{},\"counters_shared\":{},"
                    "\"wall\":{:.6f},\"io_wall\":{:.6f},\"phase_wall\":{{\"updpsi\":{:.6f},"
                    "\"setls\":{:.6f},\"drive\":{:.6f},\"updjnet\":{:.6f},\"nodal\":{:.6f},"
                    "\"cusping\":{:.6f},\"upddhat\":{:.6f}}},\"loop_wall\":{{"
                    "\"th_update\":{:.6f},\"xe_step\":{:.6f},"
                    "\"search_propose\":{:.6f},\"search_apply\":{:.6f}}},"
                    "\"floor_wall\":{{"
                    "\"ppr_reset\":{:.6f},\"ppr_drive\":{:.6f},\"ppr_recon\":{:.6f},"
                    "\"depl_predictor\":{:.6f},\"depl_corrector\":{:.6f},"
                    "\"result_add\":{:.6f},\"result_write\":{:.6f}}},"
                    "\"nested_wall\":{{\"flatxs\":{:.6f},\"flatxs_calls\":{}}},"
                    "\"floor_transfer\":{{\"h2d_bytes\":{},\"h2d_calls\":{},"
                    "\"d2h_bytes\":{},\"d2h_calls\":{}}},"
                    "\"search\":{{\"trials\":{},\"proposals\":{},\"refused\":{},"
                    "\"secant\":{},\"carry_secant\":{},\"probe\":{},\"bisect\":{},"
                    "\"extrap\":{},"
                    "\"iterations\":{},\"exit\":{},\"tol\":{:.3e},\"dk\":{:.6e},"
                    "\"x_first\":{:.6f},\"x_final\":{:.6f},\"dx_last\":{:.6f}}}}}\n",
                    sp_job_id, sp_slot, step_number, efpd, total_outer, c.outers(),
                    c.outers_by_cause[sptelem::CAUSE_INITIAL], xeModeName(),
                    c.xe_updates, c.xe_interim_updates, c.xe_cascades, xe_per_cascade,
                    c.xe_budget_exhausted, c.xe_once_extra_steps, c.xe_once_trust_trips,
                    c.xe_aa_proposed, c.xe_aa_accepted, c.xe_aa_rejected,
                    c.xe_aa_history_resets,
                    c.outers_by_cause[sptelem::CAUSE_XE],
                    c.search_trials, c.outers_by_cause[sptelem::CAUSE_SEARCH],
                    c.th_updates, c.outers_by_cause[sptelem::CAUSE_TH],
                    c.outers_by_cause[sptelem::CAUSE_SETTLE],
                    c.outers_by_cause[sptelem::CAUSE_FALLBACK], c.staged_relapses,
                    c.th_search_coincident,
                    c.flux_limit_retries, c.solve_loops, c.cmfd_sweeps, c.bicg_iters,
                    c.graph_delta, c.h2d_delta, c.d2h_delta, c.d2h_calls_delta, shared,
                    c.wall, c.io_wall, c.phase[sptelem::PH_UPDPSI],
                    c.phase[sptelem::PH_SETLS], c.phase[sptelem::PH_DRIVE],
                    c.phase[sptelem::PH_UPDJNET], c.phase[sptelem::PH_NODAL],
                    c.phase[sptelem::PH_CUSPING], c.phase[sptelem::PH_UPDDHAT],
                    c.phase[sptelem::PH_TH_UPDATE], c.phase[sptelem::PH_XE_STEP],
                    c.phase[sptelem::PH_SEARCH_PROPOSE], c.phase[sptelem::PH_SEARCH_APPLY],
                    c.phase[sptelem::PH_PPR_RESET], c.phase[sptelem::PH_PPR_DRIVE],
                    c.phase[sptelem::PH_PPR_RECON], c.phase[sptelem::PH_DEPL_PRED],
                    c.phase[sptelem::PH_DEPL_CORR], c.phase[sptelem::PH_RESULT_ADD],
                    c.phase[sptelem::PH_RESULT_WRITE],
                    c.xs_wall[xsphase::LB_FLATXS], c.xs_calls[xsphase::LB_FLATXS],
                    c.floor_h2d_bytes, c.floor_h2d_calls,
                    c.floor_d2h_bytes, c.floor_d2h_calls,
                    c.search_trials, c.search_proposals, c.search_refused,
                    c.search_secant, c.search_carry, c.search_probe, c.search_bisect,
                    c.search_extrap,
                    c.search_iterations, schedule.search_exit_status,
                    schedule.search_exit_tol, schedule.search_exit_dk,
                    schedule.search_first_x, schedule.search_current_x,
                    schedule.search_last_dx));
                sp_run.accumulate(c);
            }

            // Natural EOC (MASTER %EXE_DEP tgobj boron): while the converged critical
            // boron is still above the target, re-queue a copy of this entry right
            // after it.  When the letdown slope predicts the target falls inside the
            // next full step, the copy gets the predicted partial time and drops the
            // flag, so it is FINAL and accepted wherever it lands -- the same one-shot
            // prediction MASTER uses (600 d @15.11 ppm -> +1.482 d, lands 10.15 ppm).
            {
                const Schedule& cur = scheduler.schedule(step_index);
                if (cur.type == ScheduleType::DEPLETION && cur.until_boron_ppm > 0.0 &&
                    cur.ppm > cur.until_boron_ppm) {
                    if (++natural_eoc_inserts > 500)
                        throw std::runtime_error("natural EOC: boron target not reached within 500 extra steps");
                    Schedule next = cur;
                    next.ResetSearchState();
                    if (step_index > 0 && cur.time > 0.0) {
                        const double slope = (scheduler.schedule(step_index - 1).ppm - cur.ppm) / cur.time;
                        if (slope > 1.0e-6) {
                            const double t_hit = (cur.ppm - cur.until_boron_ppm) / slope;
                            if (t_hit < cur.time) {
                                next.time            = std::max(t_hit, 1.0e-3);
                                next.until_boron_ppm = 0.0;
                            }
                        }
                    }
                    scheduler.schedule().insert(scheduler.schedule().begin() + step_index + 1,
                                                std::move(next));
                }
            }
        }

        if (!light_result)
            input_output.CloseResult();

        // One-line accounting of how often the nonlinear correction had to be
        // guarded/damped over the whole run. Silent when nothing fired.
        cmfd_solver.reportDhatGuardStats(_input.c_str());

        // GA evaluator plan Task 10 receipt.  Printed only when the arm is on,
        // and it carries the two numbers that decide whether the arm actually
        // ran: `statepoints` (device) against `host_fallbacks`.  `iterations`
        // is the corner-balance count the device loop actually spent, so the
        // "did the break test move?" question is answered by the receipt rather
        // than assumed -- it is directly comparable to the host arm's, which is
        // 100 x statepoints minus whatever the tolerance saved.
        {
            const PprBackend& g = pin_power_reconstruction.gpu();
            // Printed when the arm is on (or ran at all), and additionally under
            // RASBERY_STATEPOINT_TELEMETRY so the HOST arm's iteration count is
            // obtainable at all -- without that the "did the break test move?"
            // comparison would have only one side.  A run with neither is
            // byte-identical on stdout to the binary before this change.
            if (g.available() || g.statepoints() > 0 || sp_telem) {
                // WP6 raised schema_version to 2.  The stage B/C/E numbers
                // are the whole point of the work package and none of them
                // is derivable from the four that were here before:
                // `host_syncs_per_statepoint` says whether the per-round
                // synchronise actually went away, `h2d_bytes` /
                // `h2d_bytes_elided` price the borrow, `canonical_mismatch`
                // says whether the borrow was SOUND, and `allocations` /
                // `reallocations` are stage E's "once per slot, not once
                // per statepoint".
                //
                // WP6 stage F added `refusal` and `refusals`.  A receipt that
                // prints `host_fallbacks:35` and nothing else cannot tell a
                // deck this build does not serve from a MODE this build did
                // not port, and the campaign paid for that: RASBERY_PPR_MODE
                // =master refused every statepoint of every production run and
                // the number looked exactly like "no CUDA device".  `refusal`
                // is the last reason, `refusals` the whole tally, both from
                // ppr::refusalName -- the same strings the RASBERY_GPU_FULL
                // seam reports as `first_violation`.
                std::cout << std::format(
                    "  [RASBERY][PPR_GPU] {{\"schema_version\":2,\"slot\":{},"
                    "\"statepoints\":{},\"device\":{},\"host_fallbacks\":{},"
                    "\"iterations\":{},\"host_iterations\":{},"
                    "\"loop_arm\":\"{}\",\"host_syncs\":{},"
                    "\"host_syncs_per_statepoint\":{:.3f},"
                    "\"graph_launches\":{},\"graph_builds\":{},"
                    "\"graph_refusal\":\"{}\",\"canonical_mode\":\"{}\","
                    "\"canonical_statepoints\":{},\"canonical_mismatch\":{},"
                    "\"h2d_bytes\":{},\"h2d_bytes_elided\":{},\"d2h_bytes\":{},"
                    "\"recon_statepoints\":{},\"pin_materializations\":{},"
                    "\"recon_repairs\":{},"
                    "\"recon_refusal\":\"{}\",\"allocations\":{},\"reallocations\":{},"
                    "\"refusal\":\"{}\",\"refusals\":{},"
                    "\"wall_ms\":{:.3f},\"status\":\"{}\"}}\n",
                    cmfd_solver.batchSlot(), g.statepoints(), g.deviceOrdinal(),
                    ppr_host_statepoints, g.iterations(),
                    pin_power_reconstruction.hostIterations(), g.loopArm(),
                    g.hostSyncs(), g.hostSyncsPerStatepoint(), g.graphLaunches(),
                    g.graphBuilds(), g.graphRefusal(),
                    ppr::canonicalModeName(ppr::canonicalModeFromEnv()),
                    g.canonicalStatepoints(), g.canonicalMismatch(), g.h2dBytes(),
                    g.h2dBytesElided(), g.d2hBytes(), g.reconStatepoints(),
                    g.pinMaterializations(), g.reconRepairs(), g.reconRefusal(),
                    g.allocations(),
                    g.reallocations(), g.lastRefusalName(), g.refusalJson(),
                    g.wallMs(), g.status());
            }
        }

        // GA evaluator plan Task 16 receipt.  Printed when the arm is on (or ran
        // at all), and under RASBERY_STATEPOINT_TELEMETRY so an A/B has a line
        // from both sides.  The three numbers that decide whether the arm ran
        // are `statepoints` (device predictor calls), `host_fallbacks` (every
        // predictor or corrector half that ran on the CPU) and `gs_iters_mean`
        // -- the mean Gauss-Seidel sweep count per (node, pole), which is the
        // observable that says the device solved the SAME iteration the host
        // does rather than a differently-converged one.
        //
        // WP21-D ADDED THE MAPPING, AND THE MAPPING IS NOT THE ARM.  The
        // pole-parallel kernel (`kernel_variant`:"pole4", `lanes_per_node`:4)
        // is B0 against the serial one -- same additions in the same order --
        // so RASBERY_GPU_CRAM_PARALLEL is deliberately NOT in kArmEnv: it does
        // not move the trajectory the way RASBERY_GPU_CRAM does.  But it does
        // move the TIME, and a per-launch figure quoted without the mapping it
        // was measured under is unattributable, which is why both fields are
        // here and why `gs_iters_mean` sitting next to them is the check that
        // the wider kernel solved the same iteration count as the narrow one.
        // `launch_us_mean` is -1 unless RASBERY_GPU_CRAM_TIMING was set: it is
        // the cudaEvent time of the KERNELS ALONE, which is what ncu's block 39
        // reports, whereas `wall_ms` is and has always been the whole call.
        {
            const CramBackend& c = cross_sections.cram();
            if (c.available() || c.predictorCalls() > 0 || c.correctorCalls() > 0 ||
                sp_telem) {
                const double gs_mean =
                    (c.gsSolves() > 0)
                        ? static_cast<double>(c.gsIterations()) /
                              static_cast<double>(c.gsSolves())
                        : 0.0;
                std::cout << std::format(
                    "  [RASBERY][CRAM_GPU] {{\"schema_version\":1,\"slot\":{},"
                    "\"statepoints\":{},\"predictor_calls\":{},"
                    "\"corrector_calls\":{},\"nodes\":{},\"device\":{},"
                    "\"host_fallbacks\":{},\"gs_iters_mean\":{:.3f},"
                    "\"gs_solves\":{},\"micx_h2d_mb\":{:.1f},"
                    "\"kernel_variant\":\"{}\",\"lanes_per_node\":{},"
                    "\"launches\":{},\"launch_us_mean\":{:.1f},"
                    "\"bos_reuse\":{},\"pole_sum\":\"{}\",\"wall_ms\":{:.3f},\"status\":\"{}\"}}\n",
                    cmfd_solver.batchSlot(), c.predictorCalls(), c.predictorCalls(),
                    c.correctorCalls(), c.nodesSolved(), c.deviceOrdinal(),
                    cross_sections.cramHostFallbacks(), gs_mean, c.gsSolves(),
                    static_cast<double>(c.micxH2dBytes()) / (1024.0 * 1024.0),
                    c.kernelVariant(), c.lanesPerNode(), c.launches(),
                    c.launchUsMean(),
                    c.bosReuses(), c.poleSumPrecision(), c.wallMs(), c.status());
            }
        }

        // WP15 receipt -- the micx/lmpx residency (RASBERY_GPU_MICX_RESIDENT).
        //
        // PRINTED ONLY WHEN THE ARM WAS ASKED FOR OR ACTUALLY FIRED, so a
        // flag-off log is the log of a build without this feature.  That is what
        // "feature-off identity" has to mean for a receipt as well as a result.
        //
        // `resident_hits` IS THE G0 CHECK: the arm on with 0 hits means the flag
        // never reached a flat-XS solve and any saving quoted from this run is
        // void.  `lazy_downloads + slice_downloads` is what the host actually
        // asked for back -- their ratio to `resident_hits` is the measurement,
        // because a feature that defers 384 downloads and then materialises 384
        // times has moved the copies, not removed them.  `nodal_const_*` is the
        // §2 census this work measured and did not change.
        //
        // WP15.1 ADDED THREE PAIRS.  `cram_micx_h2d_mb` against
        // `cram_micx_d2d_mb` is the depletion round trip: with the arm on and
        // the generations agreeing the first should be ~0 and the second should
        // carry what it used to.  `nodal_jnet_*` is the BATCH arena's upload
        // shadow, and its TESTS matter as much as its bytes -- 0 tests means
        // the shadow was never consulted (a single-deck run never enters the
        // arena at all), which reads very differently from consulted and always
        // missed.  A hit rate of -1 is "never asked", not "never hit".
        if (rasberyGpuMicxResidentEnabled() ||
            XsReconBackend::micxResidentHits() > 0) {
            const unsigned long long hits  = XsReconBackend::micxResidentHits();
            const unsigned long long lazy  = XsReconBackend::micxLazyDownloads();
            const unsigned long long slice = XsReconBackend::micxSliceDownloads();
            const unsigned long long jtest = XsReconBackend::nodalJnetElisionTests();
            std::cout << std::format(
                "  [RASBERY][MICX] {{\"schema_version\":2,\"slot\":{},"
                "\"arm\":{},\"resident_hits\":{},\"lazy_downloads\":{},"
                "\"slice_downloads\":{},\"materialised_per_hit\":{:.3f},"
                "\"bytes_saved\":{},\"mb_saved\":{:.1f},"
                "\"cram_micx_h2d_mb\":{:.1f},\"cram_micx_d2d_mb\":{:.1f},"
                "\"nodal_jnet_elided_mb\":{:.1f},\"nodal_jnet_elision_tests\":{},"
                "\"nodal_jnet_hit_rate\":{:.3f},"
                "\"nodal_const_uploads\":{},\"nodal_const_mb\":{:.1f}}}\n",
                cmfd_solver.batchSlot(), rasberyGpuMicxResidentEnabled() ? 1 : 0,
                hits, lazy, slice,
                hits > 0 ? static_cast<double>(lazy + slice) /
                               static_cast<double>(hits)
                         : 0.0,
                XsReconBackend::micxBytesSaved(),
                static_cast<double>(XsReconBackend::micxBytesSaved()) /
                    (1024.0 * 1024.0),
                static_cast<double>(cross_sections.cram().micxH2dBytes()) /
                    (1024.0 * 1024.0),
                static_cast<double>(cross_sections.cram().micxD2dBytes()) /
                    (1024.0 * 1024.0),
                static_cast<double>(XsReconBackend::nodalJnetElidedBytes()) /
                    (1024.0 * 1024.0),
                jtest,
                jtest > 0
                    ? static_cast<double>(XsReconBackend::nodalJnetElisionHits()) /
                          static_cast<double>(jtest)
                    : -1.0,
                XsReconBackend::nodalConstUploads(),
                static_cast<double>(XsReconBackend::nodalConstBytes()) /
                    (1024.0 * 1024.0));
        }

        // WP10.2 receipt.  Printed ONLY when a warm start was asked for or
        // saved: with both halves off nothing here runs and the log is the log
        // of a build without this feature, which is what "feature-off identity"
        // has to mean for a receipt as well as for a result.
        //
        // `initial_outers` is the number the A/B is decided on -- run the same
        // deck cold and warm and compare it, then compare `outers` and wall.
        // `keff_seed` / `boron_seed` are what the parent handed over, so a run
        // that converged somewhere else can be asked whether the seed was the
        // reason.
        if (warm_status != "off" || warm_save_status != "off") {
            std::cout << std::format(
                "  [RASBERY][WARMSTART] {{\"schema_version\":1,\"slot\":{},"
                "\"load\":\"{}\",\"load_path\":\"{}\",\"save\":\"{}\","
                "\"save_path\":\"{}\",\"gate\":\"N1\",\"initial_outers\":{},"
                "\"outers\":{},\"statepoints\":{},\"keff_seed\":{:.8f},"
                "\"boron_seed\":{:.4f},\"efpd_seed\":{:.4f},\"provenance\":\"{}\","
                "\"reason\":\"{}\"}}\n",
                cmfd_solver.batchSlot(), warm_status, jsonString(_warm_start_from),
                warm_save_status, jsonString(_warm_state_out), warm_initial_outers,
                sp_traj.outers, sp_traj.statepoints, warm.keff, warm.boron, warm.efpd,
                warm_provenance,
                jsonString(warm_reason.empty() ? warm_save_reason : warm_reason));
        }

        // WP9-D stage D receipt.  Printed ONLY when at least one lever is
        // armed: with every knob unset nothing here runs and this build's log
        // is the log of a build that does not have the feature, which is what
        // feature-off identity has to mean for a receipt as well as a result.
        //
        // WHAT IT IS FOR.  The adoption bar is "total outers down 10 % or more
        // with Gate B inside the v2 envelope", and the doc's revert conditions
        // are stated per candidate against the CLASSIFICATION, not against the
        // wall: D1 is discarded if `trials` rises against `probe+carry`, D3 if
        // `staged_relapses` reaches the same order as `trials`.  So the line
        // carries the arm AND the distribution it produced, on the same line,
        // for both the telemetry arm and the wall arm.
        //
        // `gate` is A2 exactly when a knob that relaxes a CONVERGENCE CRITERION
        // is set, and RASBERY_SEARCH_STAGED_MARGIN is the only one that does.
        // The other four move the starting point and the trial sequence -- the
        // final acceptance test is untouched production tolerance -- so they
        // are N1, and saying so here is what stops an N1 arm being filed as an
        // A2 one.  Note that the margin is itself inert without staging, so an
        // A2 gate word here without RASBERY_STAGED_FLUX_TOL in the arm env
        // means a knob that did nothing.
        if (ctx.search_policy.any()) {
            const rasbery::SearchPolicy& sp = ctx.search_policy;
            std::cout << std::format(
                "  [RASBERY][SEARCH_POLICY] {{\"schema_version\":1,\"slot\":{},"
                "\"gate\":\"{}\",\"carry_slope\":{},\"warm_boron\":{},"
                "\"boron_bracket\":{},\"max_trials\":{},\"staged_margin\":{:.4f},"
                "\"trials\":{},\"proposals\":{},\"refused\":{},\"probe\":{},"
                "\"carry_secant\":{},\"extrap\":{},\"secant\":{},\"bisect\":{},"
                "\"search_outers\":{},\"outers\":{},\"statepoints\":{}}}\n",
                cmfd_solver.batchSlot(), sp.staged_margin > 0.0 ? "A2" : "N1",
                sp.carry_slope, sp.warm_boron, sp.boron_bracket, sp.max_trials,
                sp.stagedMargin(0.0), sp_search.trials, sp_search.proposals,
                sp_search.refused, sp_search.probe, sp_search.carry, sp_search.extrap,
                sp_search.secant, sp_search.bisect, sp_search.outers,
                sp_traj.outers, sp_traj.statepoints);
        }

        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - driver_start).count();
        std::cout << std::format("  [TIMING] IO write={:.3f} s\n", total_io_seconds);
        std::cout << std::format("  TOTAL DRIVER TIME={:10.3f} s\n", total_seconds);

        // Run summary (plan Rev.4 Sec 8.1): the run totals of every per-statepoint
        // field, plus the measured T_fixed of the Sec 1 Amdahl model.  T_fixed is
        // reported as the Init+IO block (deck parse, XSLIB load, geometry build,
        // result-file open) with its library-parse half broken out, because the
        // XSLIB cache track (Sec 14) attacks exactly that half.  Process startup
        // before Drive() is NOT included -- this Driver cannot observe it, so
        // T_startup stays a harness-level measurement.
        if (sp_telem) {
            const sptelem::Counters& c = sp_run;
            const double xe_per_cascade =
                c.xe_cascades > 0
                    ? static_cast<double>(c.xe_updates) / static_cast<double>(c.xe_cascades)
                    : 0.0;
            iowriter::appendLine(std::format(
                "[RASBERY][SPTELEM][SUMMARY] {{\"schema_version\":1,\"job_id\":\"{}\","
                "\"slot\":{},\"statepoints\":{},\"outers\":{},\"outers_attributed\":{},"
                "\"outers_initial\":{},\"xe_mode\":\"{}\","
                "\"xe_updates\":{},\"xe_interim_updates\":{},"
                "\"xe_cascades\":{},\"xe_steps_per_cascade\":{:.3f},"
                "\"xe_budget_exhausted\":{},"
                "\"xe_once_extra_steps\":{},\"xe_once_trust_trips\":{},"
                "\"xe_aa_proposed\":{},\"xe_aa_accepted\":{},"
                "\"xe_aa_rejected\":{},\"xe_aa_history_resets\":{},"
                "\"xe_outers\":{},\"search_trials\":{},\"search_outers\":{},"
                "\"th_updates\":{},\"th_outers\":{},\"settle_outers\":{},"
                "\"fallback_outers\":{},\"staged_relapses\":{},\"th_search_coincident\":{},"
                "\"flux_limit_retries\":{},\"solve_loops\":{},\"cmfd_sweeps\":{},"
                "\"bicg_iters\":{},\"graph_launches_delta\":{},\"h2d_bytes_delta\":{},"
                "\"d2h_bytes_delta\":{},\"d2h_calls_delta\":{},\"counters_shared\":{},"
                "\"t_fixed\":{:.6f},\"init_seconds\":{:.6f},\"library_seconds\":{:.6f},"
                "\"solve_wall\":{:.6f},\"io_wall\":{:.6f},\"total_seconds\":{:.6f},"
                "\"phase_wall\":{{\"updpsi\":{:.6f},\"setls\":{:.6f},\"drive\":{:.6f},"
                "\"updjnet\":{:.6f},\"nodal\":{:.6f},\"cusping\":{:.6f},"
                "\"upddhat\":{:.6f}}},\"loop_wall\":{{"
                "\"th_update\":{:.6f},\"xe_step\":{:.6f},"
                "\"search_propose\":{:.6f},\"search_apply\":{:.6f}}},"
                "\"floor_wall\":{{"
                "\"ppr_reset\":{:.6f},\"ppr_drive\":{:.6f},\"ppr_recon\":{:.6f},"
                "\"depl_predictor\":{:.6f},\"depl_corrector\":{:.6f},"
                "\"result_add\":{:.6f},\"result_write\":{:.6f}}},"
                "\"nested_wall\":{{\"flatxs\":{:.6f},\"flatxs_calls\":{}}},"
                "\"floor_transfer\":{{\"h2d_bytes\":{},\"h2d_calls\":{},"
                "\"d2h_bytes\":{},\"d2h_calls\":{}}},"
                "\"search\":{{\"trials\":{},\"proposals\":{},\"refused\":{},"
                "\"secant\":{},\"carry_secant\":{},\"probe\":{},\"bisect\":{},"
                "\"extrap\":{},"
                "\"iterations\":{}}}}}\n",
                sp_job_id, sp_slot, static_cast<int>(scheduler.schedule().size()),
                c.outers_driver, c.outers(), c.outers_by_cause[sptelem::CAUSE_INITIAL],
                xeModeName(), c.xe_updates, c.xe_interim_updates, c.xe_cascades,
                xe_per_cascade,
                c.xe_budget_exhausted, c.xe_once_extra_steps, c.xe_once_trust_trips,
                c.xe_aa_proposed, c.xe_aa_accepted, c.xe_aa_rejected,
                c.xe_aa_history_resets,
                c.outers_by_cause[sptelem::CAUSE_XE],
                c.search_trials, c.outers_by_cause[sptelem::CAUSE_SEARCH],
                c.th_updates, c.outers_by_cause[sptelem::CAUSE_TH],
                c.outers_by_cause[sptelem::CAUSE_SETTLE],
                c.outers_by_cause[sptelem::CAUSE_FALLBACK], c.staged_relapses,
                c.th_search_coincident,
                c.flux_limit_retries, c.solve_loops, c.cmfd_sweeps, c.bicg_iters,
                c.graph_delta, c.h2d_delta, c.d2h_delta, c.d2h_calls_delta,
                sp_slot >= 0, init_seconds, init_seconds, library_seconds,
                c.wall, total_io_seconds, total_seconds,
                c.phase[sptelem::PH_UPDPSI], c.phase[sptelem::PH_SETLS],
                c.phase[sptelem::PH_DRIVE], c.phase[sptelem::PH_UPDJNET],
                c.phase[sptelem::PH_NODAL], c.phase[sptelem::PH_CUSPING],
                c.phase[sptelem::PH_UPDDHAT],
                c.phase[sptelem::PH_TH_UPDATE], c.phase[sptelem::PH_XE_STEP],
                c.phase[sptelem::PH_SEARCH_PROPOSE], c.phase[sptelem::PH_SEARCH_APPLY],
                c.phase[sptelem::PH_PPR_RESET], c.phase[sptelem::PH_PPR_DRIVE],
                c.phase[sptelem::PH_PPR_RECON], c.phase[sptelem::PH_DEPL_PRED],
                c.phase[sptelem::PH_DEPL_CORR], c.phase[sptelem::PH_RESULT_ADD],
                c.phase[sptelem::PH_RESULT_WRITE],
                c.xs_wall[xsphase::LB_FLATXS], c.xs_calls[xsphase::LB_FLATXS],
                c.floor_h2d_bytes, c.floor_h2d_calls,
                c.floor_d2h_bytes, c.floor_d2h_calls,
                c.search_trials, c.search_proposals, c.search_refused,
                c.search_secant, c.search_carry, c.search_probe, c.search_bisect,
                c.search_extrap,
                c.search_iterations));
            // The summary is the last line this deck emits, so flush here: an
            // abnormal exit then loses at most the lines of the decks still
            // running, never a finished deck's telemetry.
            iowriter::flushLines();
        }

        // The same fold, as a value, for a caller that cannot grep this
        // process's stdout (WP8 --evaluator; see CaseReceipt).  Stamped from
        // the SAME variables the trajectory line below prints, immediately
        // before it, so the two can never disagree about what this case did.
        _case_receipt.digest      = sp_traj.h;
        _case_receipt.statepoints = sp_traj.statepoints;
        _case_receipt.outers      = sp_traj.outers;
        _case_receipt.th_updates  = sp_traj.th;
        _case_receipt.slot        = cmfd_solver.batchSlot();
        _case_receipt.complete    = true;

        // The trajectory receipt, unconditionally, as the last line of the run.
        //
        // AFTER the telemetry summary and outside its gate, so that the two runs
        // being compared emit this line in the same place whether the telemetry
        // is on or off.  `telemetry` is printed and NOT folded into `digest`:
        // that separation is what makes "instrumentation moved the iteration" a
        // question two greps can answer.
        std::cout << std::format(
            "[RASBERY][TRAJECTORY] {{\"schema_version\":1,\"slot\":{},\"statepoints\":{},"
            "\"outers\":{},\"th_updates\":{},\"digest\":\"{:016x}\",\"telemetry\":{},"
            "\"env\":{}}}\n",
            cmfd_solver.batchSlot(), sp_traj.statepoints, sp_traj.outers, sp_traj.th,
            sp_traj.h, sp_telem ? 1 : 0, trajectory::armEnvJson());
        return 0;
    }
};

} // namespace rasbery
