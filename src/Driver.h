#pragma once
#include "BICGCMFD.h"
#include "BatchLightResult.h"
#include "IO.h"
#include "Nodal.h"
#include "PPR.h"
#include "Scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include "CompatFormat.h"
#include <stdexcept>
#include <string>

namespace rasbery {

// Wall-clock attribution of the outer iteration's serial phases, process-wide
// across batch instances (RASBERY_OUTER_TIMING=1; zero cost when unset).
// Same design rules as xsphase (XSTiming.h): atomics, one JSON line at exit.
namespace outer_timing {
struct Buckets {
    std::atomic<double> updpsi{0.0};
    std::atomic<double> setls{0.0};    // host linear-system assembly
    std::atomic<double> drive{0.0};    // BiCG outer incl. arena wait + sync
    std::atomic<double> updjnet{0.0};
    std::atomic<double> nodal{0.0};    // reset + drive
    std::atomic<double> cusping{0.0};  // ApplyRodCusping (+ upddtil on change)
    std::atomic<double> upddhat{0.0};
    std::atomic<long long> outers{0};
};
inline Buckets& buckets() { static Buckets b; return b; }
inline bool enabled() {
    static const bool on = std::getenv("RASBERY_OUTER_TIMING") != nullptr;
    return on;
}
class Scope {
    std::atomic<double>* _acc = nullptr;
    std::chrono::steady_clock::time_point _t0;
public:
    explicit Scope(std::atomic<double>& acc) {
        if (!enabled()) return;
        _acc = &acc;
        _t0  = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (_acc == nullptr) return;
        const double dt = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - _t0).count();
        double cur = _acc->load(std::memory_order_relaxed);
        while (!_acc->compare_exchange_weak(cur, cur + dt)) {}
    }
};
inline void report(std::ostream& out) {
    if (!enabled()) return;
    Buckets& b = buckets();
    out << "[RASBERY][OUTER][PHASE] {\"outers\":" << b.outers.load()
        << ",\"updpsi\":" << b.updpsi.load()
        << ",\"setls\":" << b.setls.load()
        << ",\"drive\":" << b.drive.load()
        << ",\"updjnet\":" << b.updjnet.load()
        << ",\"nodal\":" << b.nodal.load()
        << ",\"cusping\":" << b.cusping.load()
        << ",\"upddhat\":" << b.upddhat.load() << "}" << std::endl;
}
} // namespace outer_timing

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
        const bool   has_eq_xe      = !schedule.xenon_transient;
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
        double    xe_relax        = 1.0;
        double prev_xe_change = std::numeric_limits<double>::infinity();
        int    xe_no_progress = 0;   // consecutive Xe steps that did not shrink
        SolveExit exit_reason = SolveExit::ITER_EXHAUSTED;

        if (has_search)
            schedule.ResetSearchExitStatus();

        // Hard safety bound. Each feedback (search/T-H) step is followed by a bounded flux
        // re-convergence (flux_stall guard), and the search/T-H step counts are bounded too.
        const int max_iter = 50 * std::max({schedule.max_outer_iter, schedule.max_th_iter,
                                            has_search ? schedule.max_search_iter : 0});
        for (int iout = 0; iout < max_iter; ++iout) {
            // 1. Flux: CMFD BiCGSTAB iterations + Wielandt shift.
            {
                outer_timing::Scope t(outer_timing::buckets().updpsi);
                ctx.cmfd_solver.updpsi(ctx.geometry.Phif());
            }
            {
                outer_timing::Scope t(outer_timing::buckets().setls);
                ctx.cmfd_solver.setls(eigv);
            }
            {
                outer_timing::Scope t(outer_timing::buckets().drive);
                ctx.cmfd_solver.drive(eigv, ctx.geometry.Phif(), residual);
            }
            ++total_outer;
            outer_timing::buckets().outers.fetch_add(1, std::memory_order_relaxed);
            const bool flux_converged = std::abs(prev_inner - eigv) < keff_tol && residual < flux_tol;
            prev_inner                = eigv;

            // 2. Nodal correction -> CNCC (d-hat) + rod cusping macro-XS update. The cusping blend
            //    co-converges with the flux, so its settledness is implied by flux_converged.
            {
                outer_timing::Scope t(outer_timing::buckets().updjnet);
                ctx.cmfd_solver.updjnet(ctx.geometry.Phif(), ctx.geometry.Jnet());
            }
            {
                outer_timing::Scope t(outer_timing::buckets().nodal);
                ctx.nodal_solver.reset(1.0 / eigv, ctx.geometry.Jnet(),
                                       ctx.geometry.Phif(), ctx.geometry.Phis());
                ctx.nodal_solver.drive();
            }
            {
                outer_timing::Scope t(outer_timing::buckets().cusping);
                if (ctx.cross_sections.ApplyRodCusping(eigv, ctx.nodal_solver.axialTransverseLeakage()))
                    ctx.cmfd_solver.upddtil();
            }
            {
                outer_timing::Scope t(outer_timing::buckets().upddhat);
                ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());
            }

            // Keep iterating flux + nodal/cusping until the flux is converged; the feedbacks
            // (search, T/H) are root-finds on k_eff / power and must act on a clean flux.
            if (!flux_converged) {
                if (++flux_stall <= schedule.max_outer_iter)
                    continue;

                // Flux limit cycle on this trial point.  Previously this break abandoned the
                // solve and published the limit-cycling k_eff as the step answer with no
                // diagnostic at all (CY02 step 1: k=0.999562, |dk| = 43.8 pcm = 8.8x the
                // 5e-5 rod-crit tolerance).  Record it, warn, and carry on with the search so
                // the root find can step off the pathological point.
                ++stall_events;
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
                // fall through: treat the limit-cycle k_eff as a (noisy) observation.
                // The settling gate below must not hold here: the flux never converges on
                // this trial point, so waiting for settled iterations would spin until the
                // outer bound.  Take the sample as-is, exactly as before this gate existed.
                clean_iters = SEARCH_SETTLE_ITERS;
            } else {
                flux_stall = 0;
            }

            // 3. Equilibrium xenon feedback.  Previously equilibrium Xe was
            // only overwritten inside depletion, so a BOC STANDARD step
            // silently ran with zero Xe despite "xenon":"equilibrium".
            const int xe_budget = ga_feedback_passes > 0
                                      ? ga_feedback_passes
                                      : ((xe_relax < 1.0) ? XE_EQUILIBRIUM_MAX_ITER_DAMPED
                                                          : XE_EQUILIBRIUM_MAX_ITER);
            if (has_eq_xe && xe_count < xe_budget) {
                const double xe_change =
                    ctx.cross_sections.UpdateEquilibriumXenon(schedule.thermalPower(), xe_relax);
                ++xe_count;
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
                if (xe_relax == 1.0 && xe_no_progress >= xe_streak_limit) {
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
                clean_iters = 0;   // new trial point: the next sample must settle first
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
                ReconvergeFlux(ctx, eigv, FALLBACK_RECONVERGE_ITER, keff_tol, flux_tol,
                               total_outer);
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

        if (trace_sl)
            std::cout << std::format(
                "      [SL] outer+={} th+={} (search={} relax={:.3f} exit={} stalls={})\n",
                total_outer - sl_outer0, total_th - sl_th0, has_search ? 1 : 0,
                ctx.cross_sections.rod_cusping_relaxation(), SolveExitName(exit_reason),
                stall_events);
    }

public:
    explicit Driver(const std::string& input, const std::string& result_output = "")
        : _input(input),
          _result_output(result_output) {
    }

    int Drive() {
        const auto driver_start = std::chrono::steady_clock::now();
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
        input_output.ReadInput(_input);

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

        // 3. Main schedule loop
        double    total_io_seconds = 0.0;
        const int schedule_count   = static_cast<int>(scheduler.schedule().size());

        for (int step_index = 0; step_index < schedule_count; ++step_index) {
            auto& schedule = scheduler.schedule(step_index);
            const auto step_start = std::chrono::steady_clock::now();

            schedule.PrepareForStep(cross_sections.CoreHeavyMetalMassKg());
            schedule.ApplyToGeometry(geometry);

            const double power_fraction = schedule.powerFraction();
            const double thermal_power  = schedule.thermalPower();
            const double step_dt        = schedule.time * 86400.0;
            cross_sections.SetPowerRate(power_fraction);
            efpd += schedule.time * power_fraction;

            int        total_outer = 0;
            int        total_th    = 0;
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
            pin_power_reconstruction.drive(5);
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
                input_output.SaveRestart(input_output.input_dir() + std::format("restart_{}.h5", step_number),
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
            total_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
        }

        if (!light_result)
            input_output.CloseResult();

        // One-line accounting of how often the nonlinear correction had to be
        // guarded/damped over the whole run. Silent when nothing fired.
        cmfd_solver.reportDhatGuardStats(_input.c_str());

        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - driver_start).count();
        std::cout << std::format("  [TIMING] IO write={:.3f} s\n", total_io_seconds);
        std::cout << std::format("  TOTAL DRIVER TIME={:10.3f} s\n", total_seconds);
        return 0;
    }
};

} // namespace rasbery
