// Task 5 B0 gate: the shared CMFD outer bodies must be BIT-IDENTICAL to the CPU
// loops they mirror, and the convergence/stall state machine must reproduce
// Driver.h's decisions and stay inside the W1 phase transition table.
//
//   ./rasbery_cmfd_outer_replay [nxyz]
//
// TWO HALVES, TWO KINDS OF EVIDENCE.
//
//   (1) ARITHMETIC.  upddtil / updpsi / updjnet / upddhat, scored elementwise
//       against src/CmfdOuterReference.cpp -- a verbatim quotation of the CPU
//       loops compiled in a translation unit of its own (see the note there for
//       why that separation is load-bearing).  Class B0 means BIT-identical:
//       there is no transcendental in any of these bodies, so "close" is not a
//       result, it is a bug.  The dhat counters are checked too, including the
//       ratio maximum, because Sec 6.12 replaces the host's serial accumulation
//       with a block reduction and an atomicMax over bit patterns.
//
//   (2) CONTROL.  cmfdOuterConvergence is not floating-point work, it is
//       Driver.h's outer state machine, and what can go wrong with it is a
//       dropped branch rather than a rounding.  So it is driven through the
//       cases that separate the branches -- the stall ladder, the limit-cycle
//       fall-through, the fatal exit, the interim-Xe path, the BORON-only
//       settling gate, and the Xe/TH/Search perturbation ORDER -- and each
//       assertion names the Driver.h behaviour it is protecting.  Every phase
//       edge the machine can emit is then cross-checked against
//       kPhaseTransitions: an edge the W1 table does not contain is a
//       trajectory the scheduler cannot execute, and Sec 5.4 is explicit that
//       phase order within a case IS the physics.

#include "CmfdOuterKernel.h"
#include "CudaCmfdOuterKernels.h"

#include "CmfdOuterFormMine.h"
#include "CmfdOuterReference.h"

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

std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

co::CmfdGeometryView geomOf(const cmfdref::Mesh& m) {
    co::CmfdGeometryView g{};
    g.surface_node    = m.surface_node;
    g.surface_dir     = m.surface_dir;
    g.node_hmesh      = m.node_hmesh;
    g.node_volume     = m.node_volume;
    g.boundary_albedo = m.boundary_albedo;
    g.nxyz            = m.nxyz;
    g.nsurf           = m.nsurf;
    g.ng              = m.ng;
    return g;
}

// ---------------------------------------------------------------------------
// (1) arithmetic
// ---------------------------------------------------------------------------

void replayArithmetic(const cmfdref::Fixture& f, bool clamp_enabled,
                      unsigned long long forms) {
    const cmfdref::Mesh m = f.mesh();

    std::vector<double>   ref_dtil(f.dtil.size()), ref_jnet(f.jnet.size()),
        ref_dhat(f.dhat.size()), ref_psi(static_cast<size_t>(f.nxyz));
    cmfdref::DhatCounters ref_counters;
    cmfdref::refUpdDtil(m, ref_dtil.data());
    cmfdref::refUpdPsi(m, ref_psi.data());
    cmfdref::refUpdJnet(m, ref_jnet.data());
    cmfdref::refUpdDhat(m, clamp_enabled, ref_dhat.data(), &ref_counters);

    std::vector<double> dtil = f.dtil, jnet = f.jnet, dhat = f.dhat;
    std::vector<double> psi(static_cast<size_t>(f.nxyz), 0.0);
    co::CmfdOuterView   v{};
    v.xsdf = m.xsdf;
    v.xsnf = m.xsnf;
    v.flux = m.flux;
    v.jnet = jnet.data();
    v.dtil = dtil.data();
    v.dhat = dhat.data();
    v.psi  = psi.data();
    const co::CmfdGeometryView g     = geomOf(m);

    long bad_dtil = 0, bad_psi = 0, bad_jnet = 0, bad_dhat = 0;

    // The four bodies are scored on the INPUT state, exactly as the host loops
    // read it: upddtil does not consume its own output, updjnet reads the dtil
    // and dhat that were current when the host ran, and so on.  Writing them
    // into the same arrays in one pass would silently make each body's input
    // depend on the previous body's output and turn a bit comparison into a
    // comparison of two different sequences.
    for (int ls = 0; ls < m.nsurf; ++ls)
        for (int ig = 0; ig < m.ng; ++ig) {
            const size_t k = static_cast<size_t>(ls) * m.ng + ig;
            if (bits(co::cmfdUpdDtilSurface(g, v, ls, ig)) != bits(ref_dtil[k])) ++bad_dtil;
            if (bits(co::cmfdUpdJnetSurface(g, v, ls, ig, forms)) != bits(ref_jnet[k]))
                ++bad_jnet;
        }
    for (int l = 0; l < m.nxyz; ++l)
        if (bits(co::cmfdUpdPsiNode(g, v, l, forms)) != bits(ref_psi[static_cast<size_t>(l)]))
            ++bad_psi;

    // upddhat plus the Sec 6.12 counter accumulation, in the reduction shape the
    // kernel uses: independent per-site contributions summed, and the maximum
    // taken over the non-negative bit patterns.
    unsigned long long total = 0, guard = 0, clamped = 0, ratio_bits = 0;
    for (int ls = 0; ls < m.nsurf; ++ls)
        for (int ig = 0; ig < m.ng; ++ig) {
            const size_t k = static_cast<size_t>(ls) * m.ng + ig;
            const co::CmfdDhatContribution c =
                co::cmfdUpdDhatSurface(g, v, ls, ig, clamp_enabled, forms);
            if (bits(c.dhat) != bits(ref_dhat[k])) ++bad_dhat;
            total += static_cast<unsigned long long>(c.counted);
            guard += static_cast<unsigned long long>(c.fsum_guard);
            clamped += static_cast<unsigned long long>(c.clamped);
            if (c.ratio >= 0.0) {
                const unsigned long long b = ng_::cmfdRatioToBits(c.ratio);
                if (b > ratio_bits) ratio_bits = b;
            }
        }

    std::printf("  clamp=%-3s  dtil_bad=%ld psi_bad=%ld jnet_bad=%ld dhat_bad=%ld\n",
                clamp_enabled ? "on" : "off", bad_dtil, bad_psi, bad_jnet, bad_dhat);
    check(bad_dtil == 0, "upddtil is not bit-identical to the CPU loop");
    check(bad_psi == 0, "updpsi is not bit-identical to the CPU loop");
    check(bad_jnet == 0, "updjnet is not bit-identical to the CPU loop");
    check(bad_dhat == 0, "upddhat is not bit-identical to the CPU loop");

    std::printf("  counters   total=%llu/%lld guard=%llu/%lld clamped=%llu/%lld "
                "ratio_max=%a/%a\n",
                total, ref_counters.total, guard, ref_counters.fsum_guard, clamped,
                ref_counters.clamped, ng_::cmfdRatioFromBits(ratio_bits),
                ref_counters.ratio_max);
    check(total == static_cast<unsigned long long>(ref_counters.total),
          "_dhat_total disagrees with the CPU count");
    check(guard == static_cast<unsigned long long>(ref_counters.fsum_guard),
          "_dhat_fsum_guard disagrees -- both early exits must count, and they count "
          "into the SAME counter (CMFD.cpp:152, 159)");
    check(clamped == static_cast<unsigned long long>(ref_counters.clamped),
          "_dhat_clamped disagrees");
    check(bits(ng_::cmfdRatioFromBits(ratio_bits)) == bits(ref_counters.ratio_max),
          "max|dhat/dtil| disagrees -- the atomicMax-over-bit-patterns reduction (Sec "
          "6.12) must give the same double the host's serial maximum does");
}

/// The bit-pattern maximum is only valid for NON-NEGATIVE doubles: a negative
/// double's sign bit makes it the largest unsigned value there is.  Sec 6.12
/// relies on that precondition, so it is checked rather than assumed.
void checkRatioBitMonotonicity() {
    const double probes[] = {0.0,
                             4.9406564584124654e-324,
                             1e-300,
                             1e-12,
                             0.5,
                             1.0,
                             1.0000000000000002,
                             2.72,
                             1e12,
                             1.7976931348623157e308};
    const int n = static_cast<int>(sizeof(probes) / sizeof(probes[0]));
    bool      ok = true;
    for (int i = 1; i < n; ++i)
        if (!(ng_::cmfdRatioToBits(probes[i - 1]) < ng_::cmfdRatioToBits(probes[i]))) ok = false;
    check(ok, "the double->bit-pattern map is not monotone on non-negative values, so "
              "atomicMax over bit patterns would not compute the maximum ratio");
    for (int i = 0; i < n; ++i)
        check(bits(ng_::cmfdRatioFromBits(ng_::cmfdRatioToBits(probes[i]))) == bits(probes[i]),
              "cmfdRatioToBits/FromBits do not round-trip");
}

// ---------------------------------------------------------------------------
// (2) control -- Driver.h's outer state machine
// ---------------------------------------------------------------------------

co::CmfdOuterInputs baseInputs() {
    co::CmfdOuterInputs in{};
    in.eigv            = 1.0;
    in.residual        = 1.0e-9;
    in.keff_tol        = 1.0e-6;
    in.flux_tol        = 1.0e-6;
    in.max_outer_iter  = 4;
    in.xe_pending      = 0;
    in.xe_interim_l2   = 0.0;
    in.xe_once_mode    = 0;
    in.xe_budget_probe = 10;
    in.th_pending      = 0;
    in.search_pending  = 0;
    in.search_is_boron = 0;
    return in;
}

co::CmfdOuterState freshState() {
    co::CmfdOuterState st{};
    st.prev_inner = 1.0; // |prev_inner - eigv| = 0 -> converged when residual is small
    return st;
}

const char* actionName(co::CmfdOuterAction a) {
    switch (a) {
        case co::CmfdOuterAction::RequeueOuter:      return "Outer";
        case co::CmfdOuterAction::Xenon:             return "Xenon";
        case co::CmfdOuterAction::ThermalHydraulics: return "ThermalHydraulics";
        case co::CmfdOuterAction::Search:            return "Search";
        case co::CmfdOuterAction::Converged:         return "NormalizeFluxSign";
        case co::CmfdOuterAction::Fatal:             return "Failed";
    }
    return "?";
}

/// Is `Outer -> to` an edge the W1 table actually contains?  If it is not, the
/// scheduler cannot execute the trajectory this machine just asked for, and no
/// amount of numerical agreement will show it (Sec 5.4).
bool edgeExists(ng_::DevicePhase to) {
    for (int i = 0; i < ng_::kPhaseTransitionCount; ++i)
        if (ng_::kPhaseTransitions[i].from == ng_::DevicePhase::Outer &&
            ng_::kPhaseTransitions[i].to == to)
            return true;
    return false;
}

void replayControl() {
    std::vector<ng_::DevicePhase> seen;

    auto record = [&](const co::CmfdOuterResult& r) {
        const ng_::DevicePhase p = ng_::cmfdOuterActionPhase(r.action);
        for (ng_::DevicePhase q : seen)
            if (q == p) return;
        seen.push_back(p);
    };

    // --- converged, nothing pending: the statepoint's solve is done ----------
    {
        co::CmfdOuterInputs in = baseInputs();
        co::CmfdOuterState  st = freshState();
        const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.flux_converged == 1, "converged outer not reported converged");
        check(r.action == co::CmfdOuterAction::Converged,
              "a converged outer with nothing pending must go to NormalizeFluxSign");
        check(r.escape == co::CmfdOuterEscape::FluxConverged, "escape should be FluxConverged");
        check(st.flux_stall == 0, "a converged outer must clear flux_stall (Driver.h:1663)");
        check(st.total_outer == 1, "total_outer must count every outer (Driver.h:1557)");
    }

    // --- the stall ladder: max_outer_iter requeues, then a limit-cycle event -
    {
        co::CmfdOuterInputs in = baseInputs();
        in.residual            = 1.0;   // never converges
        in.search_pending      = 1;     // has_search, so the fall-through is allowed
        in.search_is_boron     = 0;     // keep the settling gate out of this case
        co::CmfdOuterState st  = freshState();

        for (unsigned i = 0; i < in.max_outer_iter; ++i) {
            st.prev_inner = in.eigv + 1.0; // force "not converged" every time
            const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
            record(r);
            check(r.action == co::CmfdOuterAction::RequeueOuter,
                  "within max_outer_iter the outer must requeue (Driver.h:1634)");
            check(st.flux_stall == i + 1, "flux_stall must increment once per stalled outer");
            check(st.stall_events == 0, "no stall event until the budget is spent");
        }
        // One more: the budget is spent, so this is a limit cycle.
        st.prev_inner = in.eigv + 1.0;
        const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.warn_limit_cycle == 1, "the limit-cycle event must be reported");
        check(st.stall_events == 1, "stall_events must increment (Driver.h:1642)");
        check(st.flux_stall == 0, "flux_stall must reset after a limit-cycle event "
                                  "(Driver.h:1651)");
        check(st.clean_iters == static_cast<unsigned>(co::SEARCH_SETTLE_ITERS),
              "the fall-through must open the settling gate (Driver.h:1661) -- otherwise "
              "the solve spins to the outer bound on a point whose flux never converges");
        check(r.stall_sample == 1, "the limit-cycle observation is a (noisy) sample");
        check(st.stall_sample_taken == 1u,
              "stall_sample is a host LOCAL; on the device it must survive the phase "
              "boundary or the Xe step never sees it");
    }

    // --- no search: a limit cycle is fatal immediately ----------------------
    {
        co::CmfdOuterInputs in = baseInputs();
        in.residual            = 1.0;
        in.search_pending      = 0;
        co::CmfdOuterState st  = freshState();
        for (unsigned i = 0; i < in.max_outer_iter; ++i) {
            st.prev_inner = in.eigv + 1.0;
            co::cmfdOuterConvergence(in, st);
        }
        st.prev_inner               = in.eigv + 1.0;
        const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.action == co::CmfdOuterAction::Fatal,
              "without a search there is nothing to step off the pathological point, so "
              "the limit cycle is fatal (Driver.h:1652)");
        check(r.escape == co::CmfdOuterEscape::FluxStallFatal, "escape must be FluxStallFatal");
    }

    // --- the stall-event budget itself is fatal on the 4th event -------------
    {
        co::CmfdOuterInputs in = baseInputs();
        in.residual            = 1.0;
        in.search_pending      = 1;
        co::CmfdOuterState st  = freshState();
        int                fatal_at = -1;
        for (int event = 0; event < co::MAX_FLUX_STALL_EVENTS + 2 && fatal_at < 0; ++event) {
            for (unsigned i = 0; i <= in.max_outer_iter; ++i) {
                st.prev_inner               = in.eigv + 1.0;
                const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
                if (r.action == co::CmfdOuterAction::Fatal) {
                    fatal_at = event;
                    break;
                }
            }
        }
        check(fatal_at == co::MAX_FLUX_STALL_EVENTS,
              "the fatal exit must fire on the event AFTER MAX_FLUX_STALL_EVENTS "
              "(Driver.h:776, 1652: `stall_events > MAX_FLUX_STALL_EVENTS`)");
    }

    // --- the interim-Xe path -------------------------------------------------
    {
        co::CmfdOuterInputs in = baseInputs();
        in.residual            = 1.0e-5; // below the interim gate, above flux_tol
        in.xe_interim_l2       = 1.0e-4;
        in.xe_pending          = 1;
        co::CmfdOuterState st  = freshState();
        st.prev_inner          = in.eigv + 1.0;
        st.flux_stall          = 3;

        const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.xe_interim == 1, "the interim probe should fire here");
        check(st.flux_stall == 0,
              "an interim step changes the problem, so it is not a stall (Driver.h:1632)");
        check(st.xe_interim_count == 1,
              "interim steps are budgeted separately from the settled-flux Xe budget "
              "(Driver.h:1605-1610)");
        check(r.action == co::CmfdOuterAction::Xenon,
              "a pending Xe step fires on an interim flux (Driver.h:1699)");
    }
    {
        // Interim fired but Xe is NOT pending any more: search/TH may only act
        // on a fully converged flux, so the outer requeues (Driver.h:1834).
        co::CmfdOuterInputs in = baseInputs();
        in.residual            = 1.0e-5;
        in.xe_interim_l2       = 1.0e-4;
        in.xe_pending          = 1;
        in.search_pending      = 1;
        co::CmfdOuterState st  = freshState();
        st.prev_inner          = in.eigv + 1.0;
        in.xe_pending          = 1;
        co::CmfdOuterResult r  = co::cmfdOuterConvergence(in, st);
        check(r.action == co::CmfdOuterAction::Xenon, "setup: expected the Xe edge");

        in.xe_pending = 0; // the Xe phase reports itself settled
        st.prev_inner = in.eigv + 1.0;
        r             = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.xe_interim == 0,
              "with no Xe pending the interim probe must not fire (Driver.h:1628)");
    }

    // --- the settling gate is BORON-only ------------------------------------
    {
        co::CmfdOuterInputs in = baseInputs();
        in.search_pending      = 1;
        in.search_is_boron     = 1;
        co::CmfdOuterState st  = freshState();

        for (int i = 0; i < co::SEARCH_SETTLE_ITERS; ++i) {
            st.prev_inner               = in.eigv; // converged
            const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
            record(r);
            check(r.action == co::CmfdOuterAction::RequeueOuter,
                  "a BORON search must spend SEARCH_SETTLE_ITERS settled outers before it "
                  "samples k_eff (Driver.h:1852)");
            check(st.clean_iters == static_cast<unsigned>(i + 1), "clean_iters must advance");
        }
        st.prev_inner               = in.eigv;
        const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.action == co::CmfdOuterAction::Search, "after settling, the search samples");
    }
    {
        // RODCRIT: the gate must NOT apply.  Driver.h:1843-1851 records that
        // applying it drove i-SMR CY03/CY04 into flux limit cycles they had
        // never hit, moving k_eff by up to 8.2 pcm and the rod step by 0.014.
        co::CmfdOuterInputs in = baseInputs();
        in.search_pending      = 1;
        in.search_is_boron     = 0;
        co::CmfdOuterState st  = freshState();
        const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.action == co::CmfdOuterAction::Search,
              "a ROD search must sample immediately -- the settling gate is BORON-only "
              "(Driver.h:1852)");
        check(st.clean_iters == 0, "the gate must not advance clean_iters for a rod search");
    }

    // --- the perturbation ORDER: Xe, then TH, then Search --------------------
    {
        co::CmfdOuterInputs in = baseInputs();
        in.xe_pending          = 1;
        in.th_pending          = 1;
        in.search_pending      = 1;
        in.search_is_boron     = 0;
        co::CmfdOuterState st  = freshState();
        const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.action == co::CmfdOuterAction::Xenon,
              "with all three pending, Xe goes first (Sec 5.4 / the W1 edge order)");
    }
    {
        co::CmfdOuterInputs in = baseInputs();
        in.th_pending          = 1;
        in.search_pending      = 1;
        in.search_is_boron     = 0;
        co::CmfdOuterState st  = freshState();
        const co::CmfdOuterResult r = co::cmfdOuterConvergence(in, st);
        record(r);
        check(r.action == co::CmfdOuterAction::ThermalHydraulics,
              "Driver.h checks the search first but PERTURBS T/H first, and it is the "
              "perturbation that moves the physics (Driver.h:1888 before 1901)");
    }

    // --- every emitted edge must exist in the W1 table -----------------------
    std::printf("  phases emitted:");
    for (ng_::DevicePhase p : seen) {
        std::printf(" %d", static_cast<int>(p));
        check(edgeExists(p),
              std::string("the convergence kernel emits Outer -> phase ") +
                  std::to_string(static_cast<int>(p)) +
                  ", which kPhaseTransitions does not contain");
    }
    std::printf("  (%zu distinct)\n", seen.size());
    check(seen.size() >= 5,
          "the control replay did not exercise at least five distinct outgoing edges");

    // --- the escape codes must line up with DeviceEscape ---------------------
    check(static_cast<unsigned>(ng_::cmfdOuterEscapeCode(co::CmfdOuterEscape::FluxConverged)) ==
              static_cast<unsigned>(ng_::DeviceEscape::FluxConverged),
          "escape mapping: FluxConverged");
    check(static_cast<unsigned>(
              ng_::cmfdOuterEscapeCode(co::CmfdOuterEscape::FluxLimitCycleSample)) ==
              static_cast<unsigned>(ng_::DeviceEscape::FluxLimitCycleSample),
          "escape mapping: FluxLimitCycleSample");
    check(static_cast<unsigned>(ng_::cmfdOuterEscapeCode(co::CmfdOuterEscape::FluxStallFatal)) ==
              static_cast<unsigned>(ng_::DeviceEscape::FluxStallFatal),
          "escape mapping: FluxStallFatal");

    // --- the carried state must survive a DeviceSlotState round trip ---------
    {
        co::CmfdOuterState a{};
        a.prev_inner         = 1.00123;
        a.flux_stall         = 3;
        a.stall_events       = 2;
        a.stall_sample_taken = 1;
        a.clean_iters        = 1;
        a.xe_interim_count   = 17;
        a.total_outer        = 4210;

        ng_::DeviceSlotState s{};
        ng_::deviceSlotStateReset(s);
        ng_::cmfdStoreOuterState(a, s);
        const co::CmfdOuterState b = ng_::cmfdLoadOuterState(s);

        check(bits(a.prev_inner) == bits(b.prev_inner) && a.flux_stall == b.flux_stall &&
                  a.stall_events == b.stall_events &&
                  a.stall_sample_taken == b.stall_sample_taken &&
                  a.clean_iters == b.clean_iters && a.xe_interim_count == b.xe_interim_count &&
                  a.total_outer == b.total_outer,
              "the Sec 6.13 carried state does not survive a DeviceSlotState round trip -- "
              "a field dropped here is a trajectory that diverges silently");
    }

    // --- a refilled slot must not inherit any of it --------------------------
    {
        ng_::DeviceSlotState s{};
        co::CmfdOuterState   dirty{};
        dirty.flux_stall         = 9;
        dirty.stall_events       = 3;
        dirty.stall_sample_taken = 1;
        dirty.clean_iters        = 2;
        dirty.xe_interim_count   = 44;
        ng_::cmfdStoreOuterState(dirty, s);
        ng_::deviceSlotStateReset(s);
        const co::CmfdOuterState clean = ng_::cmfdLoadOuterState(s);
        check(clean.flux_stall == 0 && clean.stall_events == 0 &&
                  clean.stall_sample_taken == 0 && clean.clean_iters == 0 &&
                  clean.xe_interim_count == 0,
              "a refilled slot inherits the previous tenant's stall state -- the exact "
              "class of bug the four-struct reset exists to prevent");
    }
}

} // namespace

int main(int argc, char** argv) {
    const int nxyz = argc > 1 ? std::atoi(argv[1]) : 1024;
    const cmfdref::Fixture f = cmfdref::buildFixture(nxyz);

    // SELF-CALIBRATING (see CmfdOuterFormMine.h): the mask records which
    // multiply-adds THIS HOST's compiler fused -- measured 0x6 on the authoring
    // box and 0x7 on 238's Xeon Gold 5317 -- so the Class B0 comparison must be
    // made against the host's own mask, not a literal from another machine.
    // What is still asserted is that the DERIVATION is stable and that the mined
    // mask reproduces the reference exactly.
    bool                     mine_sound = true;
    const unsigned long long forms       = cmfdmine::mineStable(f, mine_sound);
    std::printf("cmfd outer replay: nxyz=%d nsurf=%d ng=%d mined=0x%llX (build default "
                "0x%llX)\n",
                f.nxyz, f.nsurf, cmfdref::NG, forms, co::CMFD_OUTER_FORMS);
    check(mine_sound,
          "a search seed failed to reach zero mismatches -- this fixture does not "
          "determine the mask, which is a defect on any host");

    std::printf(" [1] arithmetic (Class B0, bit-identical)\n");
    replayArithmetic(f, false, forms);
    replayArithmetic(f, true, forms);
    checkRatioBitMonotonicity();

    std::printf(" [2] convergence / stall state machine (Sec 6.13)\n");
    replayControl();

    if (failures) {
        std::printf("cmfd outer replay: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("cmfd outer replay: PASS\n");
    return 0;
}
