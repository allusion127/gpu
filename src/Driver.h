#pragma once
#include "BICGCMFD.h"
#include "BatchLightResult.h"
#include "IO.h"
#include "Nodal.h"
#include "PPR.h"
#include "Scheduler.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include "CompatFormat.h"
#include <stdexcept>
#include <string>

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
    PH_COUNT
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
    long long search_trials        = 0;  ///< committed AND applied trial points
    long long th_updates           = 0;
    long long flux_limit_retries   = 0;  ///< flux limit-cycle events ([WARN][flux])
    long long th_search_coincident = 0;  ///< outers where T/H and search both fired
    long long cmfd_sweeps          = 0;
    long long bicg_iters           = 0;
    double    wall                 = 0.0;  ///< solve wall of the statepoint
    double    io_wall              = 0.0;
    double    phase[PH_COUNT]{};

    double        phase0[PH_COUNT]{};
    std::uint64_t graph0 = 0, h2d0 = 0, d2h0 = 0, d2h_calls0 = 0;
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
        graph0     = graph;
        h2d0       = h2d;
        d2h0       = d2h;
        d2h_calls0 = d2h_calls;
    }

    /// Close the statepoint: resolve every delta against the armed baselines.
    void end(std::uint64_t graph, std::uint64_t h2d, std::uint64_t d2h,
             std::uint64_t d2h_calls) {
        const double* current = phaseWall();
        for (int p = 0; p < PH_COUNT; ++p) phase[p] = current[p] - phase0[p];
        graph_delta     = graph - graph0;
        h2d_delta       = h2d - h2d0;
        d2h_delta       = d2h - d2h0;
        d2h_calls_delta = d2h_calls - d2h_calls0;
    }

    /// Fold one closed statepoint into a run-total accumulator.
    void accumulate(const Counters& step) {
        for (int c = 0; c < CAUSE_COUNT; ++c)
            outers_by_cause[c] += step.outers_by_cause[c];
        for (int p = 0; p < PH_COUNT; ++p) phase[p] += step.phase[p];
        outers_driver        += step.outers_driver;
        solve_loops          += step.solve_loops;
        xe_updates           += step.xe_updates;
        xe_interim_updates   += step.xe_interim_updates;
        xe_cascades          += step.xe_cascades;
        xe_budget_exhausted  += step.xe_budget_exhausted;
        search_trials        += step.search_trials;
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
// WHAT once IS NOT.  It is still not the equilibrium fixed point: one Picard
// step from a moved macro-XS state lands near it, not on it, so results move
// at the pcm level and this is a campaign mode compared against the exact run,
// never an acceptance path.  The damper and the interim-flux probe are both
// bypassed -- a single undamped step per segment cannot limit-cycle, so there
// is nothing for the oscillation detector to detect, and the interim probe
// exists to spend cascade steps on a loose flux, which is the opposite of what
// this mode does.
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

/// True when the iteration still runs but every cascade is capped at one
/// undamped step.  Deliberately NOT folded into xeFrozen(): frozen mode's
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

class Driver {
private:
    std::string _input;
    std::string _result_output;

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
    };

    // Flux-only re-convergence (CMFD/BiCGSTAB + nodal/CNCC + cusping), with every feedback
    // (search, T/H, Xe) held fixed.  Used after the search falls back to a previously observed
    // trial point so that the published k_eff belongs to the published rod position / boron.
    static void ReconvergeFlux(SolverContext& ctx, double& eigv, int max_iter,
                               double keff_tol, double flux_tol, int& total_outer) {
        double residual   = 1.0;
        double prev_inner = eigv + 1.0;
        ctx.cmfd_solver.upddtil();
        for (int i = 0; i < max_iter; ++i) {
            ctx.cmfd_solver.updpsi(ctx.geometry.Phif());
            ctx.cmfd_solver.setls(eigv);
            ctx.cmfd_solver.drive(eigv, ctx.geometry.Phif(), residual);
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
        // counters, the step itself and the cascade re-arm -- and caps each
        // cascade at one undamped step instead.  This is the only place the
        // mode is read (xeOnce() is a cached lookup of an env var resolved once
        // per process), and every once-specific term below short-circuits on
        // it, so an unset variable leaves all of them exactly what they were.
        const bool   xe_once_mode   = xeOnce();
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
                schedule.StartCriticalSearch(ctx.search_memory, ctx.geometry.bppm(0),
                                             ctx.cross_sections.rod_max_step());
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
        // ONCE mode is undamped by definition: relax rescales the applied step
        // of an ITERATION, and one step per segment is not an iteration.  A
        // damped single step would simply publish half of the equilibrium move
        // and call it the segment's answer, so the companion knob is ignored
        // there and the damper below is switched off with it.
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
        // ONCE mode: has this cascade already spent its single Xe step?  Cleared
        // at the three cascade starts -- SolveLoop entry (this initializer) and
        // the T/H and search commits, the same sites the cascade counter and the
        // RASBERY_XE_CASCADE_BUDGET re-arm use, because they are the same events.
        // Written unconditionally and read only under xe_once_mode, so the
        // default path gains a dead store and nothing else.
        bool   xe_once_fired  = false;
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
        for (int iout = 0; iout < max_iter; ++iout) {
            bool stall_sample = false; // limit-cycle fall-through this outer
            bool th_fired     = false; // telemetry: T/H perturbed inside this outer
            bool xe_restart   = false; // a commit below re-fires the Xe cascade
            const int xe_budget_probe = ga_feedback_passes > 0
                                            ? ga_feedback_passes
                                            : ((xe_relax < 1.0) ? XE_EQUILIBRIUM_MAX_ITER_DAMPED
                                                                : XE_EQUILIBRIUM_MAX_ITER);
            // 1. Flux: CMFD BiCGSTAB iterations + Wielandt shift.
            {
                outer_timing::Scope t(sptelem::PH_UPDPSI);
                ctx.cmfd_solver.updpsi(ctx.geometry.Phif());
            }
            {
                outer_timing::Scope t(sptelem::PH_SETLS);
                ctx.cmfd_solver.setls(eigv);
            }
            {
                outer_timing::Scope t(sptelem::PH_DRIVE);
                ctx.cmfd_solver.drive(eigv, ctx.geometry.Phif(), residual);
            }
            ++total_outer;
            // Exactly one cause bucket per outer, charged to the segment this
            // outer belongs to (plan Rev.4 Sec 8 attribution rules).
            ++ctx.telemetry.outers_by_cause[sp_cause];
            outer_timing::buckets().outers.fetch_add(1, std::memory_order_relaxed);
            const bool flux_converged = std::abs(prev_inner - eigv) < keff_tol && residual < flux_tol;
            prev_inner                = eigv;

            // 2. Nodal correction -> CNCC (d-hat) + rod cusping macro-XS update. The cusping blend
            //    co-converges with the flux, so its settledness is implied by flux_converged.
            {
                outer_timing::Scope t(sptelem::PH_UPDJNET);
                ctx.cmfd_solver.updjnet(ctx.geometry.Phif(), ctx.geometry.Jnet());
            }
            {
                outer_timing::Scope t(sptelem::PH_NODAL);
                ctx.nodal_solver.reset(1.0 / eigv, ctx.geometry.Jnet(),
                                       ctx.geometry.Phif(), ctx.geometry.Phis());
                ctx.nodal_solver.drive();
            }
            {
                outer_timing::Scope t(sptelem::PH_CUSPING);
                if (ctx.cross_sections.ApplyRodCusping(eigv, ctx.nodal_solver.axialTransverseLeakage()))
                    ctx.cmfd_solver.upddtil();
            }
            {
                outer_timing::Scope t(sptelem::PH_UPDDHAT);
                ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());
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
                                     prev_xe_change >= XE_EQUILIBRIUM_TOLERANCE);
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
                if (++flux_stall <= schedule.max_outer_iter)
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
                prev_xe_change >= XE_EQUILIBRIUM_TOLERANCE) {
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
            // already hangs off: `(!xe_once_mode || !xe_once_fired)` is
            // identically true with the mode unset, and in the mode it lets
            // exactly one step through per cascade, with no test on the Xe
            // residual -- the single Picard step is the segment's answer.
            if (has_eq_xe && !xe_starved && (!xe_once_mode || !xe_once_fired) &&
                (flux_converged || xe_interim || stall_sample)) {
                const double xe_change =
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
                // This cascade has now spent its ONCE step.  See xe_once_fired.
                xe_once_fired = true;
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
                if (xe_change >= XE_EQUILIBRIUM_TOLERANCE) {
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
            if (has_search && schedule.searchType == SearchType::BORON &&
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
                exit_reason = SolveExit::CONVERGED;
                break;
            }

            // Otherwise perturb the unconverged feedbacks, then re-converge the flux.
            if (has_th && !th_converged) {
                th_dop = ctx.cross_sections.UpdateTH(power_fraction);
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
                if (schedule.search_iteration >= schedule.max_search_iter) {
                    // The best-observed point is re-applied deterministically after the loop
                    // (see the fallback block below), so there is nothing to salvage here.
                    exit_reason = SolveExit::SEARCH_EXHAUSTED;
                    break;
                }
                {
                    double      next_x = schedule.search_current_x;
                    std::string method;
                    bool        bracket_not_found = false;
                    if (!schedule.ProposeNextSearchPoint(eigv, ctx.search_memory,
                                                         ctx.cross_sections.rod_max_step(),
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
                if (schedule.searchType == SearchType::BORON)
                    ctx.cross_sections.SetBoron(schedule.search_current_x);
                else {
                    ctx.cross_sections.SetRod(schedule.search_current_x);
                    schedule.rod_step = schedule.search_current_x;
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
                // The fresh cascade gets its ONCE step: the macro-XS just moved,
                // so the inventory equilibrated against the previous state is
                // stale and one step against the new one is exactly what this
                // mode owes the segment.  Unconditional; only read under
                // xe_once_mode, so the default path is unchanged.
                xe_once_fired = false;
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
                if (schedule.searchType == SearchType::BORON)
                    ctx.cross_sections.SetBoron(schedule.search_current_x);
                else {
                    ctx.cross_sections.SetRod(schedule.search_current_x);
                    schedule.rod_step = schedule.search_current_x;
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
                "      [SL] outer+={} th+={} (search={} relax={:.3f} exit={} stalls={})\n",
                total_outer - sl_outer0, total_th - sl_th0, has_search ? 1 : 0,
                ctx.cross_sections.rod_cusping_relaxation(), SolveExitName(exit_reason),
                stall_events);
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
    explicit Driver(const std::string& input, const std::string& result_output = "")
        : _input(input),
          _result_output(result_output) {
    }

    int Drive() {
        const auto driver_start = std::chrono::steady_clock::now();
        // Phase 2 statepoint telemetry (plan Rev.4 Sec 8).  One static-local
        // bool read; every cost beyond the plain counters hangs off it.
        const bool sp_telem = sptelem::enabled();
        // Inner GA evaluations only need scalar fitness/safety receipts.  Full
        // HDF5/restart/pin output remains the default and is used for selected
        // exact cases; light mode avoids queueing mutable Geometry/Schedule
        // references while preserving every scalar computed below.
        const bool light_result = BatchLightResult::Enabled();

        // 1. Build solver objects and read input deck
        Geometry  geometry;
        Scheduler scheduler;
        XSSet     cross_sections(geometry);

        IO input_output(geometry, cross_sections, scheduler);
        // Deck + XSLIB parse, split out of Init+IO so the Amdahl model's T_fixed
        // can be separated from the XSLIB-cache track (plan Rev.4 Sec 14).
        const auto library_start = std::chrono::steady_clock::now();
        input_output.ReadInput(_input);
        const double library_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - library_start).count();

        BICGCMFD cmfd_solver(geometry, cross_sections);
        cmfd_solver.setNcmfd(5);
        cmfd_solver.setEpsl2(1.0e-6);
        cmfd_solver.setEshift(0.04);

        Nodal nodal_solver(geometry, cross_sections);
        PPR   pin_power_reconstruction(geometry, cross_sections);

        SolverContext ctx{geometry, cross_sections, cmfd_solver, nodal_solver, SearchMemory{}};

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

        // Receipt keys (plan Rev.4 Sec 8.1).  result_stem() is empty until
        // OpenResult(), and light-result runs never call it, so fall back to the
        // stem of the path this deck was told to write -- the same string that
        // makes a job's output namespace unique (Sec 7).  Built once, and only
        // when the telemetry is on: nothing is allocated on the default path.
        std::string sp_job_id;
        int         sp_slot = -1;
        sptelem::Counters sp_run;
        if (sp_telem) {
            sp_job_id = input_output.result_stem();
            if (sp_job_id.empty())
                sp_job_id = std::filesystem::path(result_path).stem().string();
            sp_slot = cmfd_solver.batchSlot();
        }

        // 3. Main schedule loop.  The bound is re-read every iteration because a
        // depletion entry carrying until_boron_ppm re-queues itself (natural EOC);
        // decks without that key never grow the vector, so their path is unchanged.
        double total_io_seconds    = 0.0;
        int    natural_eoc_inserts = 0;

        for (int step_index = 0; step_index < static_cast<int>(scheduler.schedule().size()); ++step_index) {
            auto& schedule = scheduler.schedule(step_index);
            const auto step_start = std::chrono::steady_clock::now();

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
                    cross_sections.PredictorStep(sub_dt, thermal_power, schedule.xenon_transient);
                    SolveLoop(ctx, eigv, schedule, total_outer, total_th, keep_search, prime_xe);
                    cross_sections.NormalizeFluxSign();
                    cross_sections.CorrectorStep(sub_dt, thermal_power, schedule.xenon_transient);
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

            // PPR
            pin_power_reconstruction.reset(1.0 / eigv, geometry.Jnet(), geometry.Phif(), geometry.Phis());
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
            pin_power_reconstruction.drive(ppr_iters);
            // MASTER reports pin-volume-averaged reconstructed power.  A pin-centre
            // sample biases Fq high once intra-node curvature grows during burnup,
            // so use the precomputed 3x3 Gauss-Legendre pin-area integration.
            pin_power_reconstruction.reconstructPinPower(true, schedule.print_opt.pin_flux);

            // Output
            const int step_number = step_index + 1;
            schedule.eigv         = eigv;
            schedule.rho          = (eigv > 1.0e-12) ? (eigv - 1.0) / eigv : 0.0;
            schedule.ppm          = geometry.bppm(0);
            const double step_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
            std::cout << std::format("  NO.={:4d}  EFPD={:10.3f}  K-EFF={:.6f}  PPM={:8.2f}  outer={:3d}  TH={:2d}  t={:5.2f}s\n",
                                     step_number, efpd, eigv, geometry.bppm(0), total_outer, total_th, step_seconds);

            const auto io_start = std::chrono::steady_clock::now();
            input_output.AddResult(geometry, eigv, step_index, step_number, efpd);

            if (!light_result && schedule.print_opt.save) {
                input_output.SaveRestart(RestartPath(input_output, step_number),
                                         geometry, cross_sections, eigv, efpd, step_number);
            }

            if (light_result) {
                BatchLightResult::Write(_input, input_output.xs_path(),
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
                                  now.bulk_d2h_bytes_during_iteration,
                                  now.bulk_d2h_calls_during_iteration);
                ctx.telemetry.outers_driver = total_outer;
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
                std::cout << std::format(
                    "[RASBERY][SPTELEM] {{\"schema_version\":1,\"job_id\":\"{}\",\"slot\":{},"
                    "\"statepoint\":{},\"efpd\":{:.4f},\"outers\":{},\"outers_attributed\":{},"
                    "\"outers_initial\":{},\"xe_mode\":\"{}\","
                    "\"xe_updates\":{},\"xe_interim_updates\":{},"
                    "\"xe_cascades\":{},\"xe_steps_per_cascade\":{:.3f},"
                    "\"xe_budget_exhausted\":{},"
                    "\"xe_outers\":{},\"search_trials\":{},\"search_outers\":{},"
                    "\"th_updates\":{},\"th_outers\":{},\"settle_outers\":{},"
                    "\"fallback_outers\":{},\"th_search_coincident\":{},"
                    "\"flux_limit_retries\":{},\"solve_loops\":{},\"cmfd_sweeps\":{},"
                    "\"bicg_iters\":{},\"graph_launches_delta\":{},\"h2d_bytes_delta\":{},"
                    "\"d2h_bytes_delta\":{},\"d2h_calls_delta\":{},\"counters_shared\":{},"
                    "\"wall\":{:.6f},\"io_wall\":{:.6f},\"phase_wall\":{{\"updpsi\":{:.6f},"
                    "\"setls\":{:.6f},\"drive\":{:.6f},\"updjnet\":{:.6f},\"nodal\":{:.6f},"
                    "\"cusping\":{:.6f},\"upddhat\":{:.6f}}}}}\n",
                    sp_job_id, sp_slot, step_number, efpd, total_outer, c.outers(),
                    c.outers_by_cause[sptelem::CAUSE_INITIAL], xeModeName(),
                    c.xe_updates, c.xe_interim_updates, c.xe_cascades, xe_per_cascade,
                    c.xe_budget_exhausted, c.outers_by_cause[sptelem::CAUSE_XE],
                    c.search_trials, c.outers_by_cause[sptelem::CAUSE_SEARCH],
                    c.th_updates, c.outers_by_cause[sptelem::CAUSE_TH],
                    c.outers_by_cause[sptelem::CAUSE_SETTLE],
                    c.outers_by_cause[sptelem::CAUSE_FALLBACK], c.th_search_coincident,
                    c.flux_limit_retries, c.solve_loops, c.cmfd_sweeps, c.bicg_iters,
                    c.graph_delta, c.h2d_delta, c.d2h_delta, c.d2h_calls_delta, shared,
                    c.wall, c.io_wall, c.phase[sptelem::PH_UPDPSI],
                    c.phase[sptelem::PH_SETLS], c.phase[sptelem::PH_DRIVE],
                    c.phase[sptelem::PH_UPDJNET], c.phase[sptelem::PH_NODAL],
                    c.phase[sptelem::PH_CUSPING], c.phase[sptelem::PH_UPDDHAT]);
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
            std::cout << std::format(
                "[RASBERY][SPTELEM][SUMMARY] {{\"schema_version\":1,\"job_id\":\"{}\","
                "\"slot\":{},\"statepoints\":{},\"outers\":{},\"outers_attributed\":{},"
                "\"outers_initial\":{},\"xe_mode\":\"{}\","
                "\"xe_updates\":{},\"xe_interim_updates\":{},"
                "\"xe_cascades\":{},\"xe_steps_per_cascade\":{:.3f},"
                "\"xe_budget_exhausted\":{},"
                "\"xe_outers\":{},\"search_trials\":{},\"search_outers\":{},"
                "\"th_updates\":{},\"th_outers\":{},\"settle_outers\":{},"
                "\"fallback_outers\":{},\"th_search_coincident\":{},"
                "\"flux_limit_retries\":{},\"solve_loops\":{},\"cmfd_sweeps\":{},"
                "\"bicg_iters\":{},\"graph_launches_delta\":{},\"h2d_bytes_delta\":{},"
                "\"d2h_bytes_delta\":{},\"d2h_calls_delta\":{},\"counters_shared\":{},"
                "\"t_fixed\":{:.6f},\"init_seconds\":{:.6f},\"library_seconds\":{:.6f},"
                "\"solve_wall\":{:.6f},\"io_wall\":{:.6f},\"total_seconds\":{:.6f},"
                "\"phase_wall\":{{\"updpsi\":{:.6f},\"setls\":{:.6f},\"drive\":{:.6f},"
                "\"updjnet\":{:.6f},\"nodal\":{:.6f},\"cusping\":{:.6f},"
                "\"upddhat\":{:.6f}}}}}\n",
                sp_job_id, sp_slot, static_cast<int>(scheduler.schedule().size()),
                c.outers_driver, c.outers(), c.outers_by_cause[sptelem::CAUSE_INITIAL],
                xeModeName(), c.xe_updates, c.xe_interim_updates, c.xe_cascades,
                xe_per_cascade,
                c.xe_budget_exhausted, c.outers_by_cause[sptelem::CAUSE_XE],
                c.search_trials, c.outers_by_cause[sptelem::CAUSE_SEARCH],
                c.th_updates, c.outers_by_cause[sptelem::CAUSE_TH],
                c.outers_by_cause[sptelem::CAUSE_SETTLE],
                c.outers_by_cause[sptelem::CAUSE_FALLBACK], c.th_search_coincident,
                c.flux_limit_retries, c.solve_loops, c.cmfd_sweeps, c.bicg_iters,
                c.graph_delta, c.h2d_delta, c.d2h_delta, c.d2h_calls_delta,
                sp_slot >= 0, init_seconds, init_seconds, library_seconds,
                c.wall, total_io_seconds, total_seconds,
                c.phase[sptelem::PH_UPDPSI], c.phase[sptelem::PH_SETLS],
                c.phase[sptelem::PH_DRIVE], c.phase[sptelem::PH_UPDJNET],
                c.phase[sptelem::PH_NODAL], c.phase[sptelem::PH_CUSPING],
                c.phase[sptelem::PH_UPDDHAT]);
        }
        return 0;
    }
};

} // namespace rasbery
