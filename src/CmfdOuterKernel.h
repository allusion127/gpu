#pragma once

// Shared host/device bodies of the CMFD outer boundary work -- Rev.7.1 Sec 6.2,
// 6.3, 6.6, 6.12, 6.13.  Four per-element bodies plus the flux-convergence /
// stall state machine:
//
//   cmfdUpdDtilSurface   CMFD.cpp:103-124   surface diffusion coupling
//   cmfdUpdPsiNode       CMFD.cpp:246-254   CMFD fission source
//   cmfdUpdJnetSurface   CMFD.h:240-256     net current from the CMFD operator
//   cmfdUpdDhatSurface   CMFD.cpp:126-190   CNCC correction + its guards
//   cmfdOuterConvergence Driver.h:1562,1601-1705,1834-1860
//
// CLASS B0 (Sec 9.1): every one of these must produce BIT-IDENTICAL results to
// the CPU loop it mirrors.  There is no transcendental here -- no exp, no sqrt,
// nothing but IEEE add/sub/mul/div and comparisons -- so unlike Task 4 there is
// nothing to excuse a deviation with.  A difference is a bug.
//
// ---------------------------------------------------------------------------
// CONTRACTION, MINED FROM THE PRODUCTION ASSEMBLY
// ---------------------------------------------------------------------------
//
// gcc at -O3 -march=native (-ffp-contract=fast, the C++ default) does NOT make
// the same choice at every multiply-add here, and guessing "it fuses everything"
// would be wrong in the one place it matters most.  The choices below were read
// out of the release-flag assembly for CMFD.cpp and BICGCMFD.cpp, instruction by
// instruction, and each is quoted at its site:
//
//   upddtil   NO multiply-add site at all: 2*bl*br/(bl+br) is products and one
//             divide.  Nothing to pin.
//   updpsi    NOT fused.  `_psi[l] += flux*xsnf` writes through memory every
//             iteration, so gcc emits vmulsd then vaddsd and keeps the product
//             separately rounded.  This is the site a "surely it fuses" guess
//             gets wrong.
//   updjnet   internal surfaces: FUSED, with the SECOND product rounded first.
//             gcc emits vmulpd (dhat*(fr+fl)) then vfnmsub231pd, i.e.
//             fma(-dtil, fr-fl, -round(dhat*(fr+fl))).  The two boundary
//             branches have no multiply-add at all.
//   upddhat   FUSED: vfnmsub213sd computes -(dtil*fdiff) - jnet in one rounding.
//
// The probe (test/cmfd_outer_form_probe.cpp) re-derives all of it and the replay
// (test/cmfd_outer_replay.cpp) scores the bodies against verbatim quotations of
// the CPU loops.
//
// ---------------------------------------------------------------------------
// LAYOUTS ARE THE HOST'S, DELIBERATELY
// ---------------------------------------------------------------------------
//
// dtil/dhat/jnet are [ls*ng + ig] and flux is [l*ng + ig], which is what
// CMFD.h:172-186 and CMFD.cpp:5-6 say.  Note that DeviceSlotView (Sec 3.5)
// annotates `dtil` as [ig*nsurf + ls] -- the transpose.  Either addressing gives
// the same arithmetic, so this is not a correctness fork, but it IS a decision
// Task 6/7 has to make deliberately when it wires the arena in: transposing
// changes the coalescing of every surface kernel, and doing it by accident
// (because two headers disagreed) is the worst of both.  Until then these bodies
// use the host's layout so the replay is a true bit comparison and not a
// comparison of two different index calculations.
//
// Must stay compilable by g++ and nvcc: no STL, no allocation, no exceptions.

#include "GpuFormMask.h"    // the runtime override + receipt (host only)
#include "XsReconKernel.h"  // xsrFma / xsrMul, the two rounding primitives

#include <cmath>

#if defined(__CUDACC__)
    #define RASBERY_CMFD_HD __host__ __device__
#else
    #define RASBERY_CMFD_HD
#endif

namespace rasbery::cmfd {

constexpr int NG      = 2; ///< every accepted deck is 2-group
constexpr int NDIR    = 3; ///< pch.h NDIRMAX
constexpr int NLR     = 2; ///< pch.h LR
constexpr int C_LEFT  = 0; ///< pch.h LEFT
constexpr int C_RIGHT = 1; ///< pch.h RIGHT

/// CMFD::upddhat's guard constants, verbatim (CMFD.cpp:150, 179-186).
constexpr double DHAT_FSUM_FLOOR_SCALE = 1.0e-12;
constexpr double DHAT_RATIO_LIMIT      = 1.0;

// ---------------------------------------------------------------------------
// Contraction policy
// ---------------------------------------------------------------------------

// ===========================================================================
// THE MASK IS PER BUILD HOST.  IT IS NOT A UNIVERSAL CONSTANT.
// ===========================================================================
//
// This mask records which multiply-adds THE HOST COMPILER ON THIS MACHINE
// fused, so that the device build can reproduce them.  Both halves of that
// sentence are host-specific: a different CPU (different -march=native ISA), a
// different gcc, or different flags can move a site, and then a fixed constant
// is wrong on one of the two machines.
//
// Measured, not hypothesised:
//
//     WSL2 / g++ 13.3 / Xeon-class dev box   ->  0x6   (CO_PSI_ACC NOT fused)
//     238 / Xeon Gold 5317                   ->  0x7   (CO_PSI_ACC     fused)
//
// So the value below is a DEFAULT FOR THE BUILD IT SHIPPED WITH, and the tests
// that police it MINE IT FIRST rather than asserting this literal -- see
// test/cmfd_outer_form_probe.cpp.  RASBERY_CMFD_OUTER_FORMS overrides it at run
// time (hex or decimal) for the case where a binary is built on one host and
// run against a reference produced on another.
//
// WHAT DOES NOT CHANGE: on any single host, the device build must reproduce
// that host's contraction exactly.  Class B0 is a per-build-host contract, and
// the mask is how it is kept.  Task 22's v3 freeze pins the 238 value.
//
// 1-bit sites: 1 = single-rounding fma.  2-bit sites over A*B + C*D:
// 0 = both products rounded, 1 = fma(A, B, round(C*D)), 2 = fma(C, D,
// round(A*B)).
//
// How 0x6 was established on the authoring host -- the method, which travels
// even though the value does not: the assembly quotations in the file header
// were read first, from CMFD.cpp / BICGCMFD.cpp at the release flags; then
// test/cmfd_outer_form_probe.cpp --mine coordinate-descended the mask against a
// separately compiled verbatim reference and landed on the same value with ZERO
// mismatching words.  The probe also checks every site is DECISIVE on its
// fixture, so no bit is a guess that happened to go unpunished.
inline constexpr unsigned long long CMFD_OUTER_FORMS = 0x6ull;

/// The BUILD DEFAULT as a function, for the nvcc reason XsReconKernel.h
/// documents for ACTIVE_XT: device code cannot always address a namespace-scope
/// constant.  Device code that needs a mask should take it as a kernel argument
/// (see cmfdOuterFormsRuntime); this is the compile-time fallback.
RASBERY_CMFD_HD constexpr unsigned long long cmfdOuterForms() {
    return 0x6ull;
}

/// Mine THIS BINARY's contraction on THIS HOST.  Defined in
/// CmfdOuterFormMiner.cpp, which is the only translation unit that may see both
/// the shipped bodies and the verbatim CPU quotation; `sound` comes back false
/// when the coordinate descent could not reach a bit-exact mask, which is the
/// only honest reason to fall back to the baked constant.
///
/// HOST ONLY.  Device code never calls it: every enqueue in
/// CudaCmfdOuterKernels.h takes `forms` as a kernel ARGUMENT.
unsigned long long mineCmfdOuterFormsOnThisHost(bool& sound);

/// HOST-SIDE resolved mask, MINED rather than assumed.  Read once, receipt
/// logged once.
///
/// WHAT CHANGED AND WHY.  This used to return CMFD_OUTER_FORMS unless
/// RASBERY_CMFD_OUTER_FORMS was set -- the production binary trusted a constant
/// measured on the AUTHORING host, while the gates around it had already been
/// taught to mine the host's own (see the note above CMFD_OUTER_FORMS).  On both
/// machines this campaign runs on, the two disagree: default 0x6, mined 0x7,
/// because the Xeon Gold 5317 and the current WSL toolchain both fuse
/// CO_PSI_ACC.  So the device updpsi rounded `psi += flux*xsnf` in two steps
/// where the host loop fused it, and RASBERY_GPU_OUTER=1 diverged from the host
/// outer in the last bits of psi at the FIRST device outer, on every deck,
/// deterministically.  On kngr_238 that alone was 421 of 644 datasets.
///
/// The mask asserts "the shipped host bodies and the verbatim CPU quotation
/// agree bit for bit in THIS binary", which is a question this binary can
/// answer.  So it answers it, once, and says so in the receipt.  See
/// CmfdOuterFormMiner.cpp for the cost and the layering.
inline unsigned long long cmfdOuterFormsRuntime() {
    static const unsigned long long value = [] {
        bool                     sound = false;
        const unsigned long long mined = mineCmfdOuterFormsOnThisHost(sound);
        return gpu::resolveCalibratedFormMask("RASBERY_CMFD_OUTER_FORMS", CMFD_OUTER_FORMS,
                                              mined, sound, "cmfd_outer");
    }();
    return value;
}

enum : int {
    // --- 1-bit sites ---
    CO_PSI_ACC = 0,  ///< updpsi:  psi += flux*xsnf          (MEASURED: NOT fused)
    CO_DHAT_NUM,     ///< upddhat: -dtil*fdiff - jnet        (MEASURED: fused)
    CO_ONE_BIT_COUNT,

    // --- 2-bit sites ---
    /// updjnet internal: -dtil*(fr-fl) - dhat*(fr+fl)
    /// (MEASURED: state 1 -- the dhat product is rounded, the dtil one is fused)
    CO2_JNET_INTERNAL = CO_ONE_BIT_COUNT,
    CO_BIT_COUNT      = CO2_JNET_INTERNAL + 2
};

static_assert(CO_BIT_COUNT <= 64, "the form mask is one uint64");

RASBERY_CMFD_HD inline double coMa1(unsigned long long m, int bit, double a, double b,
                                    double c) {
#if defined(__CUDA_ARCH__)
    return ((m >> bit) & 1ull) ? fma(a, b, c) : a * b + c;
#else
    return ((m >> bit) & 1ull) ? xsrecon::xsrFma(a, b, c) : xsrecon::xsrMul(a, b) + c;
#endif
}

RASBERY_CMFD_HD inline double coMa2(unsigned long long m, int bit, double a, double b,
                                    double c, double d) {
    const unsigned st = static_cast<unsigned>((m >> bit) & 3ull);
#if defined(__CUDA_ARCH__)
    if (st == 1) return fma(a, b, c * d);
    if (st == 2) return fma(c, d, a * b);
    return a * b + c * d;
#else
    if (st == 1) return xsrecon::xsrFma(a, b, xsrecon::xsrMul(c, d));
    if (st == 2) return xsrecon::xsrFma(c, d, xsrecon::xsrMul(a, b));
    return xsrecon::xsrMul(a, b) + xsrecon::xsrMul(c, d);
#endif
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// The geometry the outer bodies read, in CMFD's own cached layout (CMFD.cpp
/// builds these once per Driver, CMFD.cpp:31-73).  Immutable for a cohort, so
/// one copy is shared by every slot (Sec 3.3).
struct CmfdGeometryView {
    const int*    surface_node;    ///< [ls*LR + side], CMFD::_surface_node (-1 = boundary)
    const int*    surface_dir;     ///< [ls*LR + side], CMFD::_surface_dir
    const double* node_hmesh;      ///< [l*NDIRMAX + dir]
    const double* node_volume;     ///< [l]
    const double* boundary_albedo; ///< [dir*LR + side]

    int nxyz;
    int nsurf;
    int ng;
};

/// One slot's outer state.  Layouts as documented in the file header.
struct CmfdOuterView {
    const double* xsdf; ///< [ig*nxyz + l]  (XSSet.h:554)
    const double* xsnf; ///< [ig*nxyz + l]  (XSSet.h:553)
    const double* flux; ///< [l*ng + ig]    (Geometry::Phif)

    double* jnet; ///< [ls*ng + ig]  (Geometry::Jnet)
    double* dtil; ///< [ls*ng + ig]
    double* dhat; ///< [ls*ng + ig]
    double* psi;  ///< [l], the CMFD fission source
};

// ---------------------------------------------------------------------------
// Sec 6.2  upddtil -- one surface, one group
// ---------------------------------------------------------------------------

/// CMFD::upddtil(ls), CMFD.cpp:103-124, for one group.
///
/// No multiply-add site: 2*betal*betar/(betal+betar) is two products and a
/// divide.  Written verbatim, no policy, on purpose -- a policied site here
/// would suggest there is a choice to get wrong.
RASBERY_CMFD_HD inline double cmfdUpdDtilSurface(const CmfdGeometryView& g,
                                                 const CmfdOuterView& v, int ls, int ig) {
    const int ll    = g.surface_node[ls * NLR + C_LEFT];
    const int lr    = g.surface_node[ls * NLR + C_RIGHT];
    const int idirl = g.surface_dir[ls * NLR + C_LEFT];
    const int idirr = g.surface_dir[ls * NLR + C_RIGHT];

    double betal, betar;
    if (ll < 0) {
        betal = g.boundary_albedo[idirl * NLR + C_LEFT] * 0.5;
    } else {
        betal = v.xsdf[ig * g.nxyz + ll] / g.node_hmesh[ll * NDIR + idirl];
    }
    if (lr < 0) {
        betar = g.boundary_albedo[idirr * NLR + C_RIGHT] * 0.5;
    } else {
        betar = v.xsdf[ig * g.nxyz + lr] / g.node_hmesh[lr * NDIR + idirr];
    }
    return 2 * betal * betar / (betal + betar);
}

// ---------------------------------------------------------------------------
// Sec 6.3  updpsi -- one node
// ---------------------------------------------------------------------------

/// CMFD::updpsi(l, flux), CMFD.cpp:246-254.
///
/// THE ACCUMULATION IS NOT FUSED, and that is measured, not assumed: the host
/// loop accumulates through `_psi[l]` in memory, so gcc emits vmulsd + vaddsd
/// and the product is separately rounded.  The group loop runs ascending, one
/// group at a time -- no tree reduction, which at ng = 2 would be a different
/// association for no gain anyway.
RASBERY_CMFD_HD inline double cmfdUpdPsiNode(const CmfdGeometryView& g,
                                             const CmfdOuterView& v, int l,
                                             unsigned long long forms) {
    double psi = 0.0;
    for (int ig = 0; ig < g.ng; ++ig)
        psi = coMa1(forms, CO_PSI_ACC, v.flux[l * g.ng + ig], v.xsnf[ig * g.nxyz + l], psi);
    return psi * g.node_volume[l];
}

// ---------------------------------------------------------------------------
// Sec 6.6  updjnet -- one surface, one group
// ---------------------------------------------------------------------------

/// CMFD::updjnet(ls, flux, jnet), CMFD.h:240-256, for one group.
///
/// Three branches.  The two boundary branches are a sum and a product with no
/// multiply-add between them; the internal branch is the mined two-product site.
RASBERY_CMFD_HD inline double cmfdUpdJnetSurface(const CmfdGeometryView& g,
                                                 const CmfdOuterView& v, int ls, int ig,
                                                 unsigned long long forms) {
    const int    ll   = g.surface_node[ls * NLR + C_LEFT];
    const int    lr   = g.surface_node[ls * NLR + C_RIGHT];
    const double dtil = v.dtil[ls * g.ng + ig];
    const double dhat = v.dhat[ls * g.ng + ig];

    if (ll < 0) return -(dtil + dhat) * v.flux[lr * g.ng + ig];
    if (lr < 0) return (dtil - dhat) * v.flux[ll * g.ng + ig];

    const double fl = v.flux[ll * g.ng + ig];
    const double fr = v.flux[lr * g.ng + ig];
    // -dtil*(fr - fl) - dhat*(fr + fl)
    return coMa2(forms, CO2_JNET_INTERNAL, -dtil, fr - fl, -dhat, fr + fl);
}

// ---------------------------------------------------------------------------
// Sec 6.6 / 6.12  upddhat -- one surface, one group, and its counters
// ---------------------------------------------------------------------------

/// What one upddhat evaluation contributes to CMFD's diagnostic counters.
///
/// Sec 6.12: the three integer totals are order-insensitive sums, so the kernel
/// block-reduces them and does ONE atomic per block; `ratio` feeds an atomicMax
/// over the non-negative double bit pattern.  Both of those are only sound
/// because every field here is produced independently per (surface, group) --
/// which is why the body RETURNS them instead of incrementing shared state.
struct CmfdDhatContribution {
    double dhat;        ///< the value to store
    int    counted;     ///< 1 -- every evaluation bumps _dhat_total (CMFD.cpp:148)
    int    fsum_guard;  ///< 1 when the fsum/finiteness guard fired
    int    clamped;     ///< 1 when |dhat| > |dtil| (counted even when not enforced)
    double ratio;       ///< |dhat|/|dtil| when |dtil| > 0, else -1 (no contribution)
};

/// CMFD::upddhat(ls, flux, jnet), CMFD.cpp:126-190, for one group.
///
/// The guard structure is reproduced exactly, including the order of the two
/// early exits and the fact that BOTH of them increment `_dhat_fsum_guard`
/// (CMFD.cpp:152, 159) -- the second is a non-finite quotient, not an fsum
/// problem, and merging the two counters would lose that.
///
/// `clamp_enabled` is RASBERY_DHAT_CLAMP.  It is OFF by default and must stay a
/// parameter rather than a compile-time constant: CMFD.cpp:164-178 records that
/// enforcing the envelope biases i-SMR CY01 BOC by ~+100 pcm, so the flag is a
/// diagnostic that must remain flippable without a rebuild of this header's
/// meaning.
RASBERY_CMFD_HD inline CmfdDhatContribution
cmfdUpdDhatSurface(const CmfdGeometryView& g, const CmfdOuterView& v, int ls, int ig,
                   bool clamp_enabled, unsigned long long forms) {
    const int ll = g.surface_node[ls * NLR + C_LEFT];
    const int lr = g.surface_node[ls * NLR + C_RIGHT];

    double fdiff, fsum;
    if (ll < 0) {
        fdiff = v.flux[lr * g.ng + ig];
        fsum  = v.flux[lr * g.ng + ig];
    } else if (lr < 0) {
        fdiff = -v.flux[ll * g.ng + ig];
        fsum  = v.flux[ll * g.ng + ig];
    } else {
        fdiff = v.flux[lr * g.ng + ig] - v.flux[ll * g.ng + ig];
        fsum  = v.flux[lr * g.ng + ig] + v.flux[ll * g.ng + ig];
    }

    CmfdDhatContribution out{};
    out.counted = 1;
    out.ratio   = -1.0;

    const double dtl = v.dtil[ls * g.ng + ig];
    const double floor_ =
        DHAT_FSUM_FLOOR_SCALE * (1.0 > std::fabs(dtl) ? 1.0 : std::fabs(dtl));
    if (!(std::fabs(fsum) > floor_) || !std::isfinite(fsum)) {
        out.fsum_guard = 1;
        out.dhat       = 0.0;
        return out;
    }

    // jnet_fdm - jnet(ig, ls), where jnet_fdm = -dtil*fdiff.  MEASURED FUSED:
    // gcc emits one vfnmsub213sd for the whole numerator.
    const double num = coMa1(forms, CO_DHAT_NUM, -dtl, fdiff, -v.jnet[ls * g.ng + ig]);
    double       dh  = num / fsum;
    if (!std::isfinite(dh)) {
        out.fsum_guard = 1;
        out.dhat       = 0.0;
        return out;
    }

    const double cap = std::fabs(dtl);
    if (cap > 0.0) {
        const double ratio = std::fabs(dh) / cap;
        out.ratio          = ratio;
        if (ratio > DHAT_RATIO_LIMIT) {
            out.clamped = 1;
            if (clamp_enabled) dh = (dh > 0.0 ? cap : -cap);
        }
    }
    out.dhat = dh;
    return out;
}

// ---------------------------------------------------------------------------
// Sec 6.13  Flux convergence / stall state machine  -- [Rev.7.1 수정]
// ---------------------------------------------------------------------------
//
// Driver.h:1562 and 1601-1705 plus the two `continue`s at 1834-1860.  Rev.7.1
// names five fields the Rev.7 plan omitted -- flux_stall, stall_events,
// stall_sample_taken, clean_iters, xe_interim_count -- and they are omitted for
// a reason worth restating: on the host, `stall_sample` is a LOCAL bool of one
// outer iteration.  On the device the Xe step is a different phase in a
// different launch, so the bool has to survive the phase boundary or the
// limit-cycle sample is silently dropped and the search never sees the noisy
// observation the host feeds it.
//
// Driver.h's own constants, restated (Driver.h:776, 788).

constexpr int MAX_FLUX_STALL_EVENTS = 3;
constexpr int SEARCH_SETTLE_ITERS   = 2;

/// Everything one outer's convergence decision reads that is NOT already in the
/// slot's carried state.  Grouped so the call site reads like Driver.h's outer.
struct CmfdOuterInputs {
    double eigv;     ///< after cmfd drive()
    double residual; ///< L2 flux residual from drive()
    double keff_tol;
    double flux_tol;

    unsigned int max_outer_iter; ///< schedule.max_outer_iter

    /// Sec 6.15: whether an equilibrium-Xe step is pending, already resolved by
    /// the Xe phase's own budget/starvation rules.  Not recomputed here.
    int xe_pending;
    /// RASBERY_XE_INTERIM_L2, 0.0 = off (the default, the exact old path).
    double xe_interim_l2;
    int    xe_once_mode;
    unsigned int xe_budget_probe;

    int th_pending;
    int search_pending;
    int search_is_boron; ///< the settling gate is BORON-only (Driver.h:1852)
};

/// The carried state this body reads and writes.  Mirrors the DeviceSlotState
/// fields of the same names (Sec 3.2 B); kept as a small struct so the body is
/// testable without the whole control packet.
struct CmfdOuterState {
    double       prev_inner;
    unsigned int flux_stall;
    unsigned int stall_events;
    unsigned int stall_sample_taken;
    unsigned int clean_iters;
    unsigned int xe_interim_count;
    unsigned int total_outer;
};

/// What the outer decided.  `next_phase` is a DevicePhase value; it is an int
/// here so this header does not have to include the scheduler (which would make
/// a pure numerical body depend on the queue shape).  The replay cross-checks
/// every (from, to) pair against kPhaseTransitions.
enum class CmfdOuterAction : unsigned int {
    RequeueOuter = 0,   ///< Outer -> Outer, FluxNotConverged
    Xenon,              ///< Outer -> Xenon, XePending
    ThermalHydraulics,  ///< Outer -> ThermalHydraulics, ThPending
    Search,             ///< Outer -> Search, SearchPending
    Converged,          ///< Outer -> NormalizeFluxSign, FluxConverged
    Fatal               ///< Outer -> Failed, Fatal (FluxStallFatal)
};

/// Escape codes this body can raise, mirroring DeviceEscape by value.  Spelled
/// out rather than included for the same reason as the phase.
enum class CmfdOuterEscape : unsigned int {
    None                 = 0,
    FluxConverged        = 1,
    FluxLimitCycleSample = 2,
    FluxStallFatal       = 3
};

struct CmfdOuterResult {
    CmfdOuterAction action;
    CmfdOuterEscape escape;
    int             flux_converged;
    int             xe_interim;   ///< this outer took an interim (loose-flux) Xe step
    int             stall_sample; ///< limit-cycle fall-through happened this outer
    int             warn_limit_cycle; ///< the host prints a WARN here; the device counts it
};

/// One outer's convergence decision, Driver.h line for line.
///
/// ORDER MATTERS AND IS NOT NEGOTIABLE.  `flux_converged` is computed BEFORE
/// prev_inner is overwritten (Driver.h:1562-1563); the interim probe clears
/// flux_stall before the stall counter is touched (1631-1632); the limit-cycle
/// fall-through sets clean_iters to SEARCH_SETTLE_ITERS so the settling gate
/// below cannot hold on a point whose flux never converges (1661); and the
/// settling gate is BORON-only because applying it to RODCRIT drove i-SMR
/// CY03/CY04 into limit cycles they had never hit (1843-1851).
RASBERY_CMFD_HD inline CmfdOuterResult
cmfdOuterConvergence(const CmfdOuterInputs& in, CmfdOuterState& st) {
    CmfdOuterResult r{};
    r.action = CmfdOuterAction::RequeueOuter;
    r.escape = CmfdOuterEscape::None;

    ++st.total_outer;

    // Driver.h:1562-1563
    const bool flux_converged =
        std::fabs(st.prev_inner - in.eigv) < in.keff_tol && in.residual < in.flux_tol;
    st.prev_inner   = in.eigv;
    r.flux_converged = flux_converged ? 1 : 0;

    // Driver.h:1617-1630.  The xe_starved / xe_pending arithmetic belongs to the
    // Xe phase (Sec 6.15) and arrives resolved; what stays here is the interim
    // probe, which is a FLUX decision.
    const bool xe_interim = in.xe_interim_l2 > 0.0 && !in.xe_once_mode && in.xe_pending &&
                            st.xe_interim_count < 10u * in.xe_budget_probe &&
                            !flux_converged && in.residual < in.xe_interim_l2;
    r.xe_interim = xe_interim ? 1 : 0;
    if (xe_interim) {
        ++st.xe_interim_count;
        st.flux_stall = 0; // the Xe step changes the problem; not a stall
    }

    bool stall_sample = false;
    if (!flux_converged && !xe_interim) {
        // Driver.h:1634-1635
        if (++st.flux_stall <= in.max_outer_iter) {
            r.action = CmfdOuterAction::RequeueOuter;
            return r;
        }
        // Driver.h:1642-1661.  Flux limit cycle on this trial point.
        ++st.stall_events;
        r.warn_limit_cycle = 1;
        st.flux_stall      = 0;
        if (st.stall_events > static_cast<unsigned int>(MAX_FLUX_STALL_EVENTS) ||
            !in.search_pending) {
            r.action = CmfdOuterAction::Fatal;
            r.escape = CmfdOuterEscape::FluxStallFatal;
            return r;
        }
        stall_sample = true;
        // The settling gate must NOT hold here: the flux never converges on this
        // trial point, so waiting for settled iterations would spin to the outer
        // bound.  Take the sample as-is.
        st.clean_iters = static_cast<unsigned int>(SEARCH_SETTLE_ITERS);
    } else {
        st.flux_stall = 0;
    }
    r.stall_sample = stall_sample ? 1 : 0;

    // Driver.h:1698-1699.  The Xe step fires on a converged flux, on an interim
    // step, or on a limit-cycle sample.  `stall_sample` is a host LOCAL; on the
    // device the Xe step is a separate phase, so it is published here.
    if (stall_sample) st.stall_sample_taken = 1u;
    if (in.xe_pending && (flux_converged || xe_interim || stall_sample)) {
        r.action = CmfdOuterAction::Xenon;
        if (stall_sample) r.escape = CmfdOuterEscape::FluxLimitCycleSample;
        return r;
    }

    // Driver.h:1834-1837.  An interim Xe step ran on a loosely converged flux;
    // search and T/H may only act on a fully converged one.
    if (xe_interim && !flux_converged) {
        st.prev_inner = in.eigv + 1.0;
        r.action      = CmfdOuterAction::RequeueOuter;
        return r;
    }

    // Driver.h:1852-1859.  Settling gate, BORON searches only.
    if (in.search_pending && in.search_is_boron &&
        st.clean_iters < static_cast<unsigned int>(SEARCH_SETTLE_ITERS)) {
        ++st.clean_iters;
        st.prev_inner = in.eigv + 1.0; // force a real re-drive before the next check
        r.action      = CmfdOuterAction::RequeueOuter;
        return r;
    }

    // Driver.h:1888-1943.  T/H is perturbed BEFORE the search commit, and it is
    // the perturbation that moves the physics -- Sec 5.4's caveat, and why the
    // W1 transition table orders Xe -> TH -> Search.
    if (in.th_pending) {
        r.action = CmfdOuterAction::ThermalHydraulics;
        return r;
    }
    if (in.search_pending) {
        r.action = CmfdOuterAction::Search;
        return r;
    }

    // Driver.h:1882-1885.  Everything converged: the statepoint's solve is done.
    if (stall_sample) {
        r.escape = CmfdOuterEscape::FluxLimitCycleSample;
    } else {
        r.escape = CmfdOuterEscape::FluxConverged;
    }
    r.action = CmfdOuterAction::Converged;
    return r;
}

} // namespace rasbery::cmfd
