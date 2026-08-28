// Task 9 gate: the DEVICE OUTER SEGMENT state machine must reproduce Driver.h's
// segment boundaries exactly, and must never emit a phase edge the W1 transition
// table does not contain.
//
//   ./rasbery_outer_state_replay
//
// WHAT THIS CHECKS THAT NOTHING ELSE CAN.  cmfd_outer_replay.cpp already scores
// cmfdOuterConvergence against Driver.h's branches, one outer at a time.  What it
// cannot see is the SEGMENT: how many outers run before the host is allowed to
// look, which escape is reported when two conditions are true at once, and
// whether a budget of 8 takes the same decisions as a budget of 1.  Those are
// properties of a SEQUENCE, and they are exactly where a device state machine
// diverges from a host loop without any single outer being wrong.
//
// THE REFERENCE IS DRIVER.H, QUOTED.  hostSegmentReference() below is
// Driver.h:1562-1664 written out as a segment rule -- flux_converged from
// :1562, the stall ladder from :1633-1664, the interim probe from :1628-1632 --
// and each site names its line.  It is deliberately NOT a call into
// cmfdOuterConvergence: a test that drove the shipped body and compared it to
// itself would pass on any body at all.
//
// FIVE SCRIPTED SEQUENCES, one per escape the plan names for this task:
// converged, budget, stall-fatal, limit-cycle, negative-flux -- plus the two
// device-only handovers (Rayleigh, cusping) and the non-finite guard.
//
// Builds anywhere: the state machine is pure and CUDA-free by construction, so
// this needs no device and no capture.
//
// THE `Driver.h:NNNN` CITATIONS are against the Driver.h this task was written
// against (8be6bee), the same numbering the rest of the tree cites; Task 9's
// delegation block adds 131 lines above SolveLoop, so a SolveLoop citation
// resolves at NNNN + 131 in the working tree.  See CudaOuterGraph.h's note.

#include "CudaOuterGraph.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace co  = rasbery::cmfd;
namespace ng_ = rasbery::gpu;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL %s\n", what.c_str());
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// One scripted outer's observable inputs
// ---------------------------------------------------------------------------

/// What the sweep produced for one outer, plus the device-only signals.  This is
/// the "recorded host state sequence": a replay feeds a list of these through
/// both machines and compares the transition sequences.
struct ScriptedOuter {
    double eigv;
    double residual;
    int    negative_flux    = 0;
    int    rayleigh         = 0;
    int    material_changed = 0;
};

struct SegmentInputs {
    double       keff_tol       = 1.0e-6;
    double       flux_tol       = 1.0e-6;
    unsigned int max_outer_iter = 3; ///< small, so the stall ladder is reachable
    int          xe_pending     = 0;
    int          th_pending     = 0;
    int          search_pending = 0;
    int          search_is_boron = 0;
    double       xe_interim_l2  = 0.0;
    int          xe_once_mode   = 0;
    unsigned int xe_budget_probe = 5;
    double       prev_inner     = 0.0;
};

/// One step of a transition sequence, as the comparison sees it.
struct Step {
    unsigned int phase;
    unsigned int escape;
    int          exit_segment;

    bool operator!=(const Step& o) const {
        return phase != o.phase || escape != o.escape || exit_segment != o.exit_segment;
    }
};

std::string stepText(const Step& s) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "(phase=%u escape=%s exit=%d)", s.phase,
                  ng_::outerEscapeName(static_cast<ng_::DeviceEscape>(s.escape)),
                  s.exit_segment);
    return buf;
}

// ---------------------------------------------------------------------------
// THE HOST REFERENCE -- Driver.h, quoted
// ---------------------------------------------------------------------------
//
// Written from src/Driver.h's SolveLoop, with the sites named.  It answers ONE
// question per outer: does the host go straight to the next outer, or does it
// stop and do something?  A segment is a maximal run of "straight to the next
// outer", which is precisely Driver.h's `continue` at :1635.

struct HostState {
    double       prev_inner;
    unsigned int flux_stall;
    unsigned int stall_events;
    unsigned int clean_iters;
    unsigned int xe_interim_count;
};

Step hostSegmentReference(const SegmentInputs& in, const ScriptedOuter& o, HostState& st,
                          unsigned int outer_in_segment, unsigned int budget) {
    Step s{};

    // The device-only signals come first, because the host has no equivalent of
    // them at all: a NaN, a negative flux or a degenerate Wielandt gamma is
    // something the CPU path would have discovered inside drive() and thrown or
    // fallen back on (CudaBICGBackend.cu:1725 latches sweep_state = 2 for
    // exactly the last one).  Their RANK is the contract being checked here.
    if (!std::isfinite(o.eigv) || !std::isfinite(o.residual)) {
        s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Failed);
        s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::NonFinite);
        s.exit_segment = 1;
        return s;
    }
    if (o.negative_flux) {
        s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Failed);
        s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::NegativeFlux);
        s.exit_segment = 1;
        return s;
    }
    if (o.rayleigh) {
        s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Outer);
        s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::RayleighFallback);
        s.exit_segment = 1;
        return s;
    }
    if (o.material_changed) {
        s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Outer);
        s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::MaterialChanged);
        s.exit_segment = 1;
        return s;
    }

    // Driver.h:1562-1563.
    const bool flux_converged =
        std::fabs(st.prev_inner - o.eigv) < in.keff_tol && o.residual < in.flux_tol;
    st.prev_inner = o.eigv;

    // Driver.h:1628-1632.
    const bool xe_interim = in.xe_interim_l2 > 0.0 && !in.xe_once_mode && in.xe_pending &&
                            st.xe_interim_count < 10u * in.xe_budget_probe && !flux_converged &&
                            o.residual < in.xe_interim_l2;
    if (xe_interim) {
        ++st.xe_interim_count;
        st.flux_stall = 0;
    }

    bool stall_sample = false;
    if (!flux_converged && !xe_interim) {
        // Driver.h:1634-1635 -- THE segment continue.
        if (++st.flux_stall <= in.max_outer_iter) {
            if (outer_in_segment + 1u >= budget) {
                s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Outer);
                s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::SegmentBudget);
                s.exit_segment = 1;
                return s;
            }
            s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Outer);
            s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::None);
            s.exit_segment = 0;
            return s;
        }
        // Driver.h:1642-1661.
        ++st.stall_events;
        st.flux_stall = 0;
        if (st.stall_events > static_cast<unsigned int>(co::MAX_FLUX_STALL_EVENTS) ||
            !in.search_pending) {
            s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Failed);
            s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::FluxStallFatal);
            s.exit_segment = 1;
            return s;
        }
        stall_sample   = true;
        st.clean_iters = static_cast<unsigned int>(co::SEARCH_SETTLE_ITERS);
    } else {
        st.flux_stall = 0;
    }

    // Driver.h:1698-1699.
    if (in.xe_pending && (flux_converged || xe_interim || stall_sample)) {
        s.phase  = static_cast<unsigned int>(ng_::DevicePhase::Xenon);
        s.escape = stall_sample
                       ? static_cast<unsigned int>(ng_::DeviceEscape::FluxLimitCycleSample)
                       : static_cast<unsigned int>(ng_::DeviceEscape::None);
        s.exit_segment = 1;
        return s;
    }

    // Driver.h:1834-1837.
    if (xe_interim && !flux_converged) {
        st.prev_inner = o.eigv + 1.0;
        if (outer_in_segment + 1u >= budget) {
            s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Outer);
            s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::SegmentBudget);
            s.exit_segment = 1;
            return s;
        }
        s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Outer);
        s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::None);
        s.exit_segment = 0;
        return s;
    }

    // Driver.h:1852-1859.  BORON only.
    if (in.search_pending && in.search_is_boron &&
        st.clean_iters < static_cast<unsigned int>(co::SEARCH_SETTLE_ITERS)) {
        ++st.clean_iters;
        st.prev_inner = o.eigv + 1.0;
        if (outer_in_segment + 1u >= budget) {
            s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Outer);
            s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::SegmentBudget);
            s.exit_segment = 1;
            return s;
        }
        s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Outer);
        s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::None);
        s.exit_segment = 0;
        return s;
    }

    // Driver.h:1888 / :1901 -- T/H is perturbed BEFORE the search commit.
    if (in.th_pending) {
        s.phase        = static_cast<unsigned int>(ng_::DevicePhase::ThermalHydraulics);
        s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::None);
        s.exit_segment = 1;
        return s;
    }
    if (in.search_pending) {
        s.phase        = static_cast<unsigned int>(ng_::DevicePhase::Search);
        s.escape       = static_cast<unsigned int>(ng_::DeviceEscape::None);
        s.exit_segment = 1;
        return s;
    }

    // Driver.h:1882-1885.
    s.phase  = static_cast<unsigned int>(ng_::DevicePhase::NormalizeFluxSign);
    s.escape = stall_sample
                   ? static_cast<unsigned int>(ng_::DeviceEscape::FluxLimitCycleSample)
                   : static_cast<unsigned int>(ng_::DeviceEscape::FluxConverged);
    s.exit_segment = 1;
    return s;
}

// ---------------------------------------------------------------------------
// The device machine, driven over the same script
// ---------------------------------------------------------------------------

std::vector<Step> runDevice(const SegmentInputs& in, const std::vector<ScriptedOuter>& script,
                            unsigned int budget) {
    co::CmfdOuterState st{};
    st.prev_inner = in.prev_inner;

    ng_::DeviceOuterSegmentState seg{};
    ng_::deviceOuterSegmentReset(seg, budget);

    std::vector<Step> out;
    for (const ScriptedOuter& o : script) {
        ng_::DeviceOuterProbe probe{};
        probe.eigv             = o.eigv;
        probe.residual         = o.residual;
        probe.negative_flux    = static_cast<std::uint32_t>(o.negative_flux);
        probe.rayleigh         = static_cast<std::uint32_t>(o.rayleigh);
        probe.material_changed = static_cast<std::uint32_t>(o.material_changed);
        // k_outer_refresh_inputs raises this before the decision runs, which is
        // the whole reason it is a separate kernel; reproduce that here.
        probe.nonfinite =
            (std::isfinite(o.eigv) && std::isfinite(o.residual)) ? 0u : 1u;

        co::CmfdOuterInputs ci{};
        ci.eigv            = o.eigv;
        ci.residual        = o.residual;
        ci.keff_tol        = in.keff_tol;
        ci.flux_tol        = in.flux_tol;
        ci.max_outer_iter  = in.max_outer_iter;
        ci.xe_pending      = in.xe_pending;
        ci.xe_interim_l2   = in.xe_interim_l2;
        ci.xe_once_mode    = in.xe_once_mode;
        ci.xe_budget_probe = in.xe_budget_probe;
        ci.th_pending      = in.th_pending;
        ci.search_pending  = in.search_pending;
        ci.search_is_boron = in.search_is_boron;

        const co::CmfdOuterResult r = co::cmfdOuterConvergence(ci, st);
        const ng_::CmfdOuterDecision d = ng_::cmfdPackDecision(r);
        const ng_::OuterTransition   t =
            ng_::deviceOuterTransition(d, probe, seg.outer_in_segment, seg.budget);

        ++seg.outer_in_segment;
        out.push_back(Step{t.next_phase, t.escape, t.exit_segment});
        if (t.exit_segment) break;
    }
    return out;
}

std::vector<Step> runHost(const SegmentInputs& in, const std::vector<ScriptedOuter>& script,
                          unsigned int budget) {
    HostState st{};
    st.prev_inner = in.prev_inner;

    std::vector<Step> out;
    unsigned int      committed = 0;
    for (const ScriptedOuter& o : script) {
        const Step s = hostSegmentReference(in, o, st, committed, budget);
        ++committed;
        out.push_back(s);
        if (s.exit_segment) break;
    }
    return out;
}

void compare(const char* name, const SegmentInputs& in, const std::vector<ScriptedOuter>& script,
             unsigned int budget, ng_::DeviceEscape expect_escape,
             unsigned int expect_outers) {
    const std::vector<Step> host = runHost(in, script, budget);
    const std::vector<Step> dev  = runDevice(in, script, budget);

    std::printf("    %-22s budget=%u host=%zu dev=%zu escape=%s\n", name, budget, host.size(),
                dev.size(),
                dev.empty() ? "-"
                            : ng_::outerEscapeName(
                                  static_cast<ng_::DeviceEscape>(dev.back().escape)));

    check(host.size() == dev.size(),
          std::string(name) + ": the segment lengths differ (host " +
              std::to_string(host.size()) + ", device " + std::to_string(dev.size()) +
              ") -- a segment that stops at a different outer is a different trajectory");
    const std::size_t n = host.size() < dev.size() ? host.size() : dev.size();
    for (std::size_t i = 0; i < n; ++i)
        check(!(host[i] != dev[i]), std::string(name) + ": outer " + std::to_string(i) +
                                        " host " + stepText(host[i]) + " != device " +
                                        stepText(dev[i]));

    check(!dev.empty() && dev.back().escape == static_cast<unsigned int>(expect_escape),
          std::string(name) + ": expected escape " +
              ng_::outerEscapeName(expect_escape) + ", got " +
              (dev.empty() ? "none"
                           : ng_::outerEscapeName(
                                 static_cast<ng_::DeviceEscape>(dev.back().escape))));
    check(dev.size() == expect_outers,
          std::string(name) + ": expected " + std::to_string(expect_outers) +
              " committed outers, got " + std::to_string(dev.size()));
}

// ---------------------------------------------------------------------------
// (1) the five scripted sequences
// ---------------------------------------------------------------------------

void replaySequences() {
    // A converging run: the eigenvalue settles and the residual falls under
    // tolerance at outer 3.
    {
        SegmentInputs in{};
        in.prev_inner = 1.10;
        std::vector<ScriptedOuter> script = {
            {1.05, 1.0e-2, 0, 0, 0},
            {1.02, 1.0e-4, 0, 0, 0},
            {1.02, 1.0e-9, 0, 0, 0}, // |dk| = 0 and residual under tol
            {1.02, 1.0e-9, 0, 0, 0},
        };
        compare("converged", in, script, 8, ng_::DeviceEscape::FluxConverged, 3);
    }

    // The same physics with a budget of 2: the segment must stop at 2 with
    // SegmentBudget and take the SAME decisions on the outers it did run.  This
    // is the "budget does not move the trajectory" invariant.
    {
        SegmentInputs in{};
        in.prev_inner = 1.10;
        std::vector<ScriptedOuter> script = {
            {1.05, 1.0e-2, 0, 0, 0},
            {1.02, 1.0e-4, 0, 0, 0},
            {1.02, 1.0e-9, 0, 0, 0},
        };
        compare("budget", in, script, 2, ng_::DeviceEscape::SegmentBudget, 2);
    }

    // A run that never converges and has no search to fall back on: the stall
    // ladder exhausts max_outer_iter (3) and then goes fatal on the FIRST event,
    // because Driver.h:1652's `|| !has_search` short-circuits there.
    {
        SegmentInputs in{};
        in.prev_inner     = 1.0;
        in.max_outer_iter = 2;
        in.search_pending = 0;
        std::vector<ScriptedOuter> script = {
            {1.20, 1.0e-1, 0, 0, 0}, {1.10, 1.0e-1, 0, 0, 0}, {1.20, 1.0e-1, 0, 0, 0},
            {1.10, 1.0e-1, 0, 0, 0}, {1.20, 1.0e-1, 0, 0, 0},
        };
        compare("stall_fatal", in, script, 8, ng_::DeviceEscape::FluxStallFatal, 3);
    }

    // The same limit cycle WITH a search to fall back on: the ladder falls
    // through to a noisy sample instead of dying, which is the whole point of
    // Driver.h:1656-1661.
    //
    // THE ESCAPE IS PUBLISHED ON THE XENON BRANCH, not the Search one.
    // CmfdOuterKernel.h:525-528 sets FluxLimitCycleSample where Driver.h:1698
    // fires the Xe step on a noisy observation; the Search branch at :555 leaves
    // the escape None and carries the sample in DeviceSlotState::
    // stall_sample_taken instead (:524).  Both fixtures are driven, because the
    // difference is exactly the kind of thing a "the escape is always set" guess
    // gets wrong.
    {
        SegmentInputs in{};
        in.prev_inner      = 1.0;
        in.max_outer_iter  = 2;
        in.search_pending  = 1;
        in.search_is_boron = 1;
        in.xe_pending      = 1;
        std::vector<ScriptedOuter> script = {
            {1.20, 1.0e-1, 0, 0, 0}, {1.10, 1.0e-1, 0, 0, 0}, {1.20, 1.0e-1, 0, 0, 0},
            {1.10, 1.0e-1, 0, 0, 0},
        };
        compare("limit_cycle_xe", in, script, 8, ng_::DeviceEscape::FluxLimitCycleSample, 3);
        const std::vector<Step> dev = runDevice(in, script, 8);
        check(dev.back().phase == static_cast<unsigned int>(ng_::DevicePhase::Xenon),
              "limit_cycle_xe: Driver.h:1698 fires the Xe step on a noisy observation");
    }
    {
        SegmentInputs in{};
        in.prev_inner      = 1.0;
        in.max_outer_iter  = 2;
        in.search_pending  = 1;
        in.search_is_boron = 1;
        std::vector<ScriptedOuter> script = {
            {1.20, 1.0e-1, 0, 0, 0}, {1.10, 1.0e-1, 0, 0, 0}, {1.20, 1.0e-1, 0, 0, 0},
            {1.10, 1.0e-1, 0, 0, 0},
        };
        compare("limit_cycle_search", in, script, 8, ng_::DeviceEscape::None, 3);
        const std::vector<Step> dev = runDevice(in, script, 8);
        check(dev.back().phase == static_cast<unsigned int>(ng_::DevicePhase::Search),
              "limit_cycle_search: the settling gate must NOT hold a limit-cycle sample "
              "back -- CmfdOuterKernel.h:515 sets clean_iters to SEARCH_SETTLE_ITERS so "
              "the gate at :540 cannot spin on a point whose flux never converges");
    }

    // A negative flux iterate on outer 2.  It must outrank the ordinary
    // convergence decision even though that outer LOOKS converged.
    {
        SegmentInputs in{};
        in.prev_inner = 1.0;
        std::vector<ScriptedOuter> script = {
            {1.00, 1.0e-2, 0, 0, 0},
            {1.00, 1.0e-9, 1, 0, 0}, // converged-looking AND negative
            {1.00, 1.0e-9, 0, 0, 0},
        };
        compare("negative_flux", in, script, 8, ng_::DeviceEscape::NegativeFlux, 2);
    }

    // The Rayleigh handover: sweep_state == 2.  Not a failure -- the slot stays
    // in Outer and the host finishes that sweep.
    {
        SegmentInputs in{};
        in.prev_inner = 1.0;
        std::vector<ScriptedOuter> script = {
            {1.00, 1.0e-2, 0, 0, 0},
            {1.00, 1.0e-3, 0, 1, 0},
            {1.00, 1.0e-9, 0, 0, 0},
        };
        compare("rayleigh", in, script, 8, ng_::DeviceEscape::RayleighFallback, 2);
        const std::vector<Step> dev = runDevice(in, script, 8);
        check(dev.back().phase == static_cast<unsigned int>(ng_::DevicePhase::Outer),
              "rayleigh: the handover must leave the slot in Outer -- it is the host "
              "finishing one sweep, not a failed slot");
    }

    // Cusping: the Stage A host escape.
    {
        SegmentInputs in{};
        in.prev_inner = 1.0;
        std::vector<ScriptedOuter> script = {
            {1.00, 1.0e-2, 0, 0, 0},
            {1.00, 1.0e-3, 0, 0, 1},
        };
        compare("material_changed", in, script, 8, ng_::DeviceEscape::MaterialChanged, 2);
    }

    // A non-finite eigenvalue.  It must outrank everything, including a negative
    // flux raised on the same outer.
    {
        SegmentInputs in{};
        in.prev_inner = 1.0;
        std::vector<ScriptedOuter> script = {
            {1.00, 1.0e-2, 0, 0, 0},
            {std::nan(""), 1.0e-9, 1, 1, 1},
        };
        compare("nonfinite", in, script, 8, ng_::DeviceEscape::NonFinite, 2);
    }
}

// ---------------------------------------------------------------------------
// (2) the budget must not move the trajectory
// ---------------------------------------------------------------------------
//
// THE Sec 9.1 CLASS-B0-ON-TRAJECTORY PROPERTY, as a test.  A run split into
// segments of 1, 2, 3, ... must take the same DECISIONS in the same ORDER as a
// run in one segment; only the number of host observations changes.  If the
// budget check were ranked above the convergence check (see the note on
// deviceOuterTransition), this is what would catch it: a solve converging on the
// budget-th outer would report SegmentBudget under one budget and FluxConverged
// under another.

void budgetIsNeutral() {
    SegmentInputs in{};
    in.prev_inner     = 1.10;
    in.max_outer_iter = 20;
    std::vector<ScriptedOuter> script = {
        {1.0500, 1.0e-2, 0, 0, 0}, {1.0200, 1.0e-3, 0, 0, 0}, {1.0100, 1.0e-4, 0, 0, 0},
        {1.0050, 1.0e-5, 0, 0, 0}, {1.0025, 1.0e-6, 0, 0, 0}, {1.0025, 1.0e-9, 0, 0, 0},
        {1.0025, 1.0e-9, 0, 0, 0},
    };

    // The reference: one segment wide enough to hold the whole solve.
    const std::vector<Step> whole = runDevice(in, script, 32);
    check(whole.back().escape == static_cast<unsigned int>(ng_::DeviceEscape::FluxConverged),
          "budget neutrality: the reference run did not converge, so it proves nothing");

    for (unsigned int budget = 1; budget <= 8; ++budget) {
        // Re-drive the whole solve as a chain of segments of `budget`, exactly as
        // the host loop would: adopt the carried state, continue.
        co::CmfdOuterState st{};
        st.prev_inner = in.prev_inner;

        std::vector<Step> chained;
        unsigned int      segments = 0;
        std::size_t       i        = 0;
        while (i < script.size()) {
            ng_::DeviceOuterSegmentState seg{};
            ng_::deviceOuterSegmentReset(seg, budget);
            ++segments;
            // `i` is advanced ONCE, inside the body, so a segment boundary does
            // not silently skip a scripted outer.
            while (i < script.size()) {
                const ScriptedOuter&  o = script[i];
                ng_::DeviceOuterProbe probe{};
                probe.eigv     = o.eigv;
                probe.residual = o.residual;

                co::CmfdOuterInputs ci{};
                ci.eigv            = o.eigv;
                ci.residual        = o.residual;
                ci.keff_tol        = in.keff_tol;
                ci.flux_tol        = in.flux_tol;
                ci.max_outer_iter  = in.max_outer_iter;
                ci.xe_budget_probe = in.xe_budget_probe;

                const co::CmfdOuterResult    r = co::cmfdOuterConvergence(ci, st);
                const ng_::CmfdOuterDecision d = ng_::cmfdPackDecision(r);
                const ng_::OuterTransition   t =
                    ng_::deviceOuterTransition(d, probe, seg.outer_in_segment, seg.budget);
                ++seg.outer_in_segment;
                ++i;
                chained.push_back(Step{t.next_phase, t.escape, t.exit_segment});
                if (t.exit_segment) break;
            }
            if (!chained.empty() &&
                chained.back().escape ==
                    static_cast<unsigned int>(ng_::DeviceEscape::FluxConverged))
                break;
        }

        check(chained.size() == whole.size(),
              "budget neutrality: budget " + std::to_string(budget) + " took " +
                  std::to_string(chained.size()) + " outers, the single segment took " +
                  std::to_string(whole.size()) +
                  " -- the segment length changed the outer count");
        const std::size_t n = chained.size() < whole.size() ? chained.size() : whole.size();
        for (std::size_t k = 0; k < n; ++k) {
            // Only the SegmentBudget escapes may differ: those are the extra host
            // observations a shorter budget buys, and they carry no phase change.
            const bool budget_exit =
                chained[k].escape == static_cast<unsigned int>(ng_::DeviceEscape::SegmentBudget);
            if (budget_exit) {
                check(chained[k].phase == static_cast<unsigned int>(ng_::DevicePhase::Outer),
                      "budget neutrality: a SegmentBudget exit changed the phase");
                continue;
            }
            check(chained[k].phase == whole[k].phase && chained[k].escape == whole[k].escape,
                  "budget neutrality: budget " + std::to_string(budget) + " outer " +
                      std::to_string(k) + " decided " + stepText(chained[k]) + ", the single "
                      "segment decided " + stepText(whole[k]));
        }
        std::printf("    budget=%u  outers=%zu segments=%u\n", budget, chained.size(),
                    segments);
    }
}

// ---------------------------------------------------------------------------
// (3) every emitted edge exists in kPhaseTransitions
// ---------------------------------------------------------------------------

bool edgeExists(ng_::DevicePhase to) {
    for (int i = 0; i < ng_::kPhaseTransitionCount; ++i)
        if (ng_::kPhaseTransitions[i].from == ng_::DevicePhase::Outer &&
            ng_::kPhaseTransitions[i].to == to)
            return true;
    return false;
}

void edgesAreInTheTable() {
    for (int i = 0; i < ng_::kOuterEmittedPhaseCount; ++i) {
        const ng_::DevicePhase p = ng_::outerEmittedPhaseAt(i);
        check(edgeExists(p),
              "the transition kernel can emit Outer -> phase " +
                  std::to_string(static_cast<int>(p)) +
                  ", which kPhaseTransitions does not contain; the scheduler cannot "
                  "execute that trajectory (Sec 5.4)");
    }
    check(ng_::outerEmittedPhaseAt(ng_::kOuterEmittedPhaseCount) == ng_::DevicePhase::Empty,
          "outerEmittedPhaseAt does not terminate at kOuterEmittedPhaseCount");

    // And the converse: every phase the machine CAN reach must be listed, or the
    // list above is checking a subset of itself.  Drive the machine over every
    // (decision phase, signal) combination and confirm the emitted phase is in
    // the list.
    for (int dp = 0; dp < ng_::kDevicePhaseCount; ++dp) {
        for (int sig = 0; sig < 5; ++sig) {
            ng_::CmfdOuterDecision d{};
            d.next_phase = static_cast<unsigned int>(dp);
            ng_::DeviceOuterProbe probe{};
            if (sig == 1) probe.nonfinite = 1;
            if (sig == 2) probe.negative_flux = 1;
            if (sig == 3) probe.rayleigh = 1;
            if (sig == 4) probe.material_changed = 1;
            const ng_::OuterTransition t = ng_::deviceOuterTransition(d, probe, 0, 8);
            bool                       listed = false;
            for (int i = 0; i < ng_::kOuterEmittedPhaseCount; ++i)
                if (static_cast<unsigned int>(ng_::outerEmittedPhaseAt(i)) == t.next_phase)
                    listed = true;
            // A decision phase the CMFD body cannot produce is not a real input;
            // only score the ones cmfdOuterActionPhase can emit.
            bool producible = false;
            for (int a = 0; a <= static_cast<int>(co::CmfdOuterAction::Fatal); ++a)
                if (static_cast<unsigned int>(ng_::cmfdOuterActionPhase(
                        static_cast<co::CmfdOuterAction>(a))) == static_cast<unsigned int>(dp))
                    producible = true;
            if (!producible) continue;
            check(listed, "deviceOuterTransition emitted phase " +
                              std::to_string(t.next_phase) +
                              " which outerEmittedPhaseAt does not list");
        }
    }
}

// ---------------------------------------------------------------------------
// (4) the transition writes what it may and nothing else
// ---------------------------------------------------------------------------

void transitionWritesOnlyItsOwnFields() {
    ng_::DeviceSlotPhase p{};
    ng_::deviceSlotPhaseReset(p, 7u);
    p.phase        = static_cast<std::uint8_t>(ng_::DevicePhase::Outer);
    p.queued_phase = static_cast<std::uint8_t>(ng_::DevicePhase::Outer);
    p.queued_epoch = p.state_epoch;
    p.input_id     = 42u;
    p.job_id       = 4242u;
    p.error_code   = 0x1u; // a scheduler fault bit already recorded
    p.phase_age    = 3u;

    const ng_::DeviceSlotPhase before = p;

    ng_::OuterTransition t{};
    t.next_phase = static_cast<unsigned int>(ng_::DevicePhase::Outer);
    t.escape     = static_cast<unsigned int>(ng_::DeviceEscape::SegmentBudget);
    ng_::outerApplyTransition(p, t);

    check(p.queued_phase == before.queued_phase && p.queued_epoch == before.queued_epoch,
          "outerApplyTransition wrote queued_phase / queued_epoch -- classify captures "
          "those (Sec 5.2), and a phase kernel that stamps them re-validates the queue "
          "entry it is being run from, so the slot looks queued forever");
    check(p.state_epoch == before.state_epoch + 1u,
          "outerApplyTransition did not bump state_epoch on an Outer -> Outer requeue.  "
          "Without the bump slotAlreadyQueued() stays true and classify refuses to "
          "re-queue the slot, which is the {Outer, Outer, FluxNotConverged} edge");
    check(!ng_::slotAlreadyQueued(p),
          "after a transition the captured queue entry must be stale");
    check(p.error_code == before.error_code,
          "outerApplyTransition wrote error_code, which carries the SCHEDULER fault bits "
          "(kSchedFault*, 0x1..0x8).  A DeviceEscape is a small ordinal from the same "
          "range, so gpuSchedulerFaultName would decode a physics escape as a queue fault");
    check(p.input_id == before.input_id && p.job_id == before.job_id,
          "outerApplyTransition touched the tenant identity");
    check(p.escape == static_cast<std::uint8_t>(ng_::DeviceEscape::SegmentBudget),
          "the escape was not published");
    check((p.flags & ng_::kSlotFlagFatal) == 0u,
          "a SegmentBudget exit is not a failure and must not set the fatal flag");

    // A fatal transition must set the fatal bit and clear in-flight.
    ng_::DeviceSlotPhase q{};
    ng_::deviceSlotPhaseReset(q, 3u);
    q.flags = static_cast<std::uint8_t>(ng_::kSlotFlagActive | ng_::kSlotFlagInFlight);
    ng_::OuterTransition f{};
    f.next_phase = static_cast<unsigned int>(ng_::DevicePhase::Failed);
    f.escape     = static_cast<unsigned int>(ng_::DeviceEscape::NonFinite);
    ng_::outerApplyTransition(q, f);
    check((q.flags & ng_::kSlotFlagFatal) != 0u, "a Failed transition must set the fatal flag");
    check((q.flags & ng_::kSlotFlagInFlight) == 0u,
          "a Failed transition must clear in_flight, or the scheduler treats the dead slot "
          "as one a graph body is still driving");
}

// ---------------------------------------------------------------------------
// (5) the segment plan matches the outer quantum
// ---------------------------------------------------------------------------

void planMatchesQuantum() {
    check(ng_::kOuterSegmentPlanCount == ng_::kOuterQuantumStepCount,
          "the segment plan and the outer quantum have different lengths");
    const int n = ng_::kOuterSegmentPlanCount < ng_::kOuterQuantumStepCount
                      ? ng_::kOuterSegmentPlanCount
                      : ng_::kOuterQuantumStepCount;
    for (int i = 0; i < n; ++i)
        check(std::strcmp(ng_::kOuterSegmentPlan[i].quantum_name,
                          ng_::kOuterQuantumSteps[i].name) == 0,
              std::string("segment plan step ") + std::to_string(i) + " is '" +
                  ng_::kOuterSegmentPlan[i].quantum_name + "' but the outer quantum step is '" +
                  ng_::kOuterQuantumSteps[i].name +
                  "' -- a device outer that reorders Driver.h's steps is a different "
                  "fixed point, not a faster one");
}

// ---------------------------------------------------------------------------
// (6) the Stage A cusping predicate, mined from XSSet::ApplyRodCusping
// ---------------------------------------------------------------------------

void cuspingPredicate() {
    const double eps = 1.0e-10; // pch.h EPS
    std::vector<double> none = {0.0, 0.0, 1.0, 1.0};
    std::vector<double> some = {0.0, 0.5, 1.0};

    check(!ng_::outerDeckHasFractionalRods(0, some.data(), 3, eps),
          "axial_rod_division <= 0 makes ApplyRodCusping return false at its first line "
          "(XSSet.cpp:3243), so such a deck can never cusp whatever the fractions say");
    check(!ng_::outerDeckHasFractionalRods(10, none.data(), 4, eps),
          "fully in or fully out is not fractional (XSSet.cpp:3266)");
    check(ng_::outerDeckHasFractionalRods(10, some.data(), 3, eps),
          "a node at 0.5 is exactly the cusped case");
    check(!ng_::outerDeckHasFractionalRods(10, nullptr, 3, eps),
          "a null fraction array must refuse rather than dereference");

    // The boundaries are STRICT on both sides in XSSet.cpp:3266
    // (`frac > EPS && frac < 1.0 - EPS`), so a node sitting exactly on either
    // one is NOT cusped.  Getting this backwards would make every fully-inserted
    // deck ineligible for no reason.
    std::vector<double> edge = {eps, 1.0 - eps};
    check(!ng_::outerDeckHasFractionalRods(10, edge.data(), 2, eps),
          "the EPS boundaries are strict in XSSet.cpp:3266");
}

} // namespace

int main() {
    std::printf("outer state replay: budget default=%u, plan steps=%d\n",
                rasbery::gpu::kOuterSegmentBudgetDefault, ng_::kOuterSegmentPlanCount);

    std::printf(" [1] scripted sequences vs the Driver.h reference\n");
    replaySequences();

    std::printf(" [2] the budget does not move the trajectory\n");
    budgetIsNeutral();

    std::printf(" [3] every emitted edge is in kPhaseTransitions\n");
    edgesAreInTheTable();

    std::printf(" [4] the transition writes only its own fields\n");
    transitionWritesOnlyItsOwnFields();

    std::printf(" [5] the segment plan matches the outer quantum\n");
    planMatchesQuantum();

    std::printf(" [6] the Stage A cusping predicate\n");
    cuspingPredicate();

    if (failures != 0) {
        std::fprintf(stderr, "outer state replay: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("outer state replay: PASS\n");
    return 0;
}
