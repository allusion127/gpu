#pragma once

// Shared host/device body of the per-outer SENM nodal phases:
//   caltrlcff0 -> caltrlcff12 -> updateMatrix -> calculateEven -> calculateJnet
// (Nodal.cpp), one function per phase, one thread per node (or surface).
//
// updateConstant is NOT here by design: it is the only transcendental site
// (exp/sqrt -- glibc exp is not correctly rounded and does not match CUDA)
// and it already recomputes only when xsrf/xsdf change, so it stays on the
// host and its nine coefficient arrays ride a generation counter.  Same
// resolve/apply split as FlatXsKernel.h.
//
// Determinism contract (same rules as XsReconKernel.h / FlatXsKernel.h):
//  - statement-for-statement transcription of Nodal.cpp, same evaluation
//    order, no reassociation;
//  - every a*b+c-shaped site goes through the policy's ma(); two-product
//    sums a*b (+|-) c*d follow the CMFD form finding (gcc evaluates the
//    SECOND product first, rounded, and fuses the FIRST into an fma), spelled
//    out below via ma(bit, a, b, +-xsrMul(c, d));
//  - the production mask is mined from a RASBERY_NODAL_DUMP capture by
//    test/nodal_replay.cpp, per phase.
//
// Must stay compilable by g++ and nvcc: no STL, no allocation.

#include "XsReconKernel.h" // xsrFma / xsrMul / RASBERY_XSR_HD

namespace rasbery::nodal {

using xsrecon::xsrFma;
using xsrecon::xsrMul;

constexpr int NG    = 2;
constexpr int NG2   = 4;
constexpr int NDIR  = 3; // NDIRMAX
constexpr int NLR   = 2; // LR
constexpr int NEWSB = 6; // NEWSBT (dir*LR+side stride of Geometry::_neib)
constexpr int C_LEFT = 0, C_RIGHT = 1, C_CENTER = 2;
constexpr int C_XDIR = 0, C_YDIR = 1, C_ZDIR = 2;

// Verbatim from Nodal.h's #defines.
constexpr double M011  = 0.666666667;
constexpr double M022  = 0.4;
constexpr double M033  = 0.285714286;
constexpr double M044  = 0.222222222;
constexpr double M220  = 6.;
constexpr double RM220 = 0.166666667;
constexpr double M240  = 20.;
constexpr double M231  = 10.;
constexpr double M242  = 14.;

/// Per-phase contraction masks.  Bit=1 means the site is a single-rounding
/// fma.  Phases: 0=trlcff12, 1=updateMatrix, 2=calculateEven, 3=jnet1n,
/// 4=jnet2n.  (caltrlcff0 has no multiply-add sites: products and divides
/// only.)  Production values mined by test/nodal_replay.cpp --sweep.
/// Mined masks (test/nodal_replay.cpp --sweep against the APR1400 profile
/// capture).  1-bit sites: 1 = single-rounding fma.  2-bit (ma2) sites over
/// A*B + C*D: 0 = both products rounded; 1 = fma(A,B, round(C*D));
/// 2 = fma(C,D, round(A*B)).  trlcff12 and updateMatrix mined exact on the
/// first pass (all their sites fused).
///
/// Phase 2 (calculateEven) -- MINED EXACT 2026-08-24, mask 0x2D555D57F55.
///   provenance: KNGR deck (~/kngr_238/kngr_238.json, RASBERY_PPR_MODE=master
///     RASBERY_PC_MODE=decart, pure-CPU drive), captures cpu3/cpu9/cpu17.bin
///     from RASBERY_NODAL_DUMP with RASBERY_NODAL_DUMP_CALL=3/9/17
///     (nxyz=8451, nsurf=26692);
///   tool: build/rasbery_nodal_mine_device cpu3.bin cpu9.bin cpu17.bin
///     (CUDA_VISIBLE_DEVICES=0, sm_120, --fmad=false) -> bad=0 EXACT, and
///     build/rasbery_nodal_device_replay <capture> -> c2/c4/c6 bad=0.
///   what was wrong before: the mask 0xD555D57F55 was never mined, it was the
///     uniform "everything fused" seed, and it left 1154 of 152118 c2/c4/c6
///     elements 1-2 ULP off.  Coordinate descent AND the pairwise stage both
///     stalled there because the residual was not representable by the site
///     set: `at2[igd][igd] += m022*rm220*m240` (Nodal.cpp:332) carried no
///     site at all.  Reading gcc-14.3 -O3 -march=native asm for
///     rasbery::Nodal::calculateEven settled it: gcc fuses that add into the
///     diagonal product for igd=1 (`vfmadd132sd (%rsi,%r12,8), %xmm7, %xmm0`)
///     but NOT for igd=0 (`vaddsd %xmm7, %xmm2, %xmm2`) -- the igd=0 product
///     has two uses across the `_ng == 1` versioning branch, so gcc's
///     widening_mul cannot consume it.  Sites 40/41 below spell that asymmetry
///     out; every other phase-2 site is fused, exactly as the old seed assumed.
constexpr unsigned long long NODAL_FORMS[5] = {
    0x3Full, 0xFull, 0x2D555D57F55ull, 0x7DD555Bull, 0xFBAAB56F79ull};

/// Device code cannot address the namespace-scope array's storage (same
/// finding as xsrecon's ACTIVE_XT); this constexpr function is the mask
/// source both compilers can fold.  Keep it identical to NODAL_FORMS.
RASBERY_XSR_HD constexpr unsigned long long nodalFormsOf(int phase) {
    return phase == 0   ? 0x3Full
           : phase == 1 ? 0xFull
           : phase == 2 ? 0x2D555D57F55ull
           : phase == 3 ? 0x7DD555Bull
                        : 0xFBAAB56F79ull;
}

RASBERY_XSR_HD inline double nodalMa1(unsigned long long m, int bit, double a,
                                      double b, double c) {
#if defined(__CUDA_ARCH__)
    return ((m >> bit) & 1ull) ? fma(a, b, c) : a * b + c;
#else
    return ((m >> bit) & 1ull) ? xsrFma(a, b, c) : xsrMul(a, b) + c;
#endif
}

RASBERY_XSR_HD inline double nodalMa2(unsigned long long m, int bit, double a,
                                      double b, double c, double d) {
    const unsigned st = static_cast<unsigned>((m >> bit) & 3ull);
#if defined(__CUDA_ARCH__)
    if (st == 1) return fma(a, b, c * d);
    if (st == 2) return fma(c, d, a * b);
    return a * b + c * d;
#else
    if (st == 1) return xsrFma(a, b, xsrMul(c, d));
    if (st == 2) return xsrFma(c, d, xsrMul(a, b));
    return xsrMul(a, b) + xsrMul(c, d);
#endif
}

struct StaticForms {
    RASBERY_XSR_HD double ma(int phase, int bit, double a, double b, double c) const {
        return nodalMa1(nodalFormsOf(phase), bit, a, b, c);
    }
    RASBERY_XSR_HD double ma2(int phase, int bit, double a, double b, double c,
                              double d) const {
        return nodalMa2(nodalFormsOf(phase), bit, a, b, c, d);
    }
};

/// Runtime-mask policy for the sweep (host-only test tool).
struct RuntimeForms {
    unsigned long long mask[5] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
    RASBERY_XSR_HD double ma(int phase, int bit, double a, double b, double c) const {
        return nodalMa1(mask[phase], bit, a, b, c);
    }
    RASBERY_XSR_HD double ma2(int phase, int bit, double a, double b, double c,
                              double d) const {
        return nodalMa2(mask[phase], bit, a, b, c, d);
    }
};

/// Pointer view of one instance's nodal state.  Indexing mirrors Nodal.cpp's
/// macros exactly:
///   per-(node,dir,group):  arr[(lk*NDIR + idir)*NG + ig]
///   per-(node,dir) matrix: arr[(lk*NDIR + idir)*NG2 + j*NG + i]
///   per-node matrix:       arr[lk*NG2 + j*NG + i]
///   flux:                  flux[lk*NG + ig]
///   jnet/phis:             arr[ls*NG + ig]
///   xs arrays:             arr[ig*nxyz + lk]  (SoA, shared with the xs arms)
///   xssm:                  arr[(igs*NG + ige)*nxyz + lk]
/// WP20.1: `ValueT` IS THE NARROWED HALF OF THIS VIEW, AND ONLY THAT HALF.
///
/// Under RASBERY_GPU_FP32 (src/GpuFp32Arm.h) the nodal drive own state -- the
/// nine updateConstant products, the four cross-section inputs, and the twelve
/// private working arrays -- is float.  Everything else here stays `double`
/// and each exclusion is a REASON, not an oversight:
///
///   hmesh / albedo    geometry, `nxyz*NDIR + NDIR*NLR` elements, uploaded
///                     ONCE per run.  Narrowing them saves 200 KB and buys a
///                     conversion of a table nothing re-sends.
///   flux / jnet / phis
///                     THE CANONICAL STATE (src/GpuCanonicalState.h).  In
///                     shared mode these three pointers ARE the CMFD backend
///                     buffers, and Geometry::Jnet / Phif are the D2H
///                     destinations.  Narrowing them would break the
///                     byte-sharing that makes canonical mode a pointer swap
///                     instead of a copy, and would trip HostPinRegistry rule
///                     that one host base is pinned at one width by every arm.
///   reigv / reigv_dev the eigenvalue.  FP64 for the reason CMFD one is
///                     (src/GpuFp32Arm.h): k_eff is judged at pcm and one ULP
///                     of a bare float near 1.0 is ~12 pcm.
///
/// So the split is the one the rest of this arm makes -- narrow the STATE,
/// keep the OPERATIONS -- and it is why every body below still computes in
/// `double` and every mined NODAL_FORMS contraction site still sees double
/// operands.  A float load widens at the read; a double result rounds at the
/// write, once, exactly where CtaWorkspaceF32 rounds.
template <class ValueT>
struct NodalViewT {
    // geometry (immutable; device copy uploaded once)
    const double* hmesh;   // [lk*NDIR + dir]
    const int*    lktosfc; // [(lk*NDIR + dir)*NLR + side]
    const int*    neib;    // [lk*NEWSB + dir*NLR + side]
    const int*    lklr;    // [ls*NLR + side]
    const int*    idirlr;  // [ls*NLR + side]
    const int*    sgnlr;   // [ls*NLR + side]
    const double* albedo;  // [dir*NLR + side]

    // xs inputs.  On the FP64 arm these alias the flat-XS / CMFD device block
    // directly.  Under RASBERY_GPU_FP32 they are the drive OWN narrowed copy,
    // written once per drive by the boundary kernel in
    // CudaXsReconBackend.cu -- because the producer of those bytes (the
    // macroscopic rebuild) stays FP64 and its block may not be narrowed under
    // it.
    const ValueT* xsrf; // [ig*nxyz + lk]
    const ValueT* xsnf;
    const ValueT* xssm; // [(igs*NG+ige)*nxyz + lk]
    const ValueT* chif; // [ig*nxyz + lk]; ignored when chif_empty
    int           chif_empty;

    // updateConstant products (host-computed, uploaded on generation change)
    const ValueT* eta1;
    const ValueT* eta2;
    const ValueT* m260;
    const ValueT* m251;
    const ValueT* m253;
    const ValueT* m262;
    const ValueT* m264;
    const ValueT* diagD;
    const ValueT* diagDI;

    // device-resident working arrays
    ValueT* trlcff0;
    ValueT* trlcff1;
    ValueT* trlcff2;
    ValueT* mu;
    ValueT* tau;
    ValueT* matM;
    ValueT* matMI;
    ValueT* matMs;
    ValueT* matMf;
    ValueT* dsncff2;
    ValueT* dsncff4;
    ValueT* dsncff6;

    // per-call inputs/outputs
    const double* flux; // phif
    double*       jnet;
    double*       phis;
    double        reigv;
    /// Optional indirection for `reigv`, the ONLY per-drive scalar in this
    /// view.  A CUDA graph bakes kernel arguments, so the device arm parks
    /// reigv in a device slot and points here; nullptr (host, replay tools,
    /// hybrid arm) keeps the by-value `reigv` above.  Both spellings deliver
    /// the identical double to the identical arithmetic site.
    const double* reigv_dev;
    int           nxyz;
    int           nsurf;

    // =======================================================================
    // Rev.7.1 Task 10 part 3: THE SEGMENT'S HALT, CARRIED INTO THE NODAL DRIVE
    // =======================================================================
    //
    // WHY A DEVICE OUTER NEEDS IT.  A host-free segment enqueues its whole
    // budget back to back and never looks at the exit word in between, so the
    // outers past the exit are already in flight when the transition latches.
    // Every kernel of the CMFD body tests that halt and returns
    // (CudaCmfdOuterKernels.h); the nodal drive did not, because it is a HOST
    // call and a host call cannot read a device word.  Its kernels can.
    //
    // AND IT IS NOT A TIDY-UP.  The drive is NOT idempotent: nodalTrlcff0Group
    // builds the transverse leakage FROM jnet and nodalCalculateJnet writes
    // jnet, so a drive re-run on a halted outer computes its leakage from its
    // own previous output.  Left ungated, a budget-8 segment that exited at
    // outer 3 would run four more nodal drives over the answer it had already
    // decided -- finite, plausible, and not the host's.
    //
    // NULL IS THE DEFAULT AND THE FEATURE-OFF SHAPE.  Every arm that is not a
    // host-free segment leaves it null: the host outer body, the hybrid drive,
    // the replay tools, the batch arena.  `halt_slot` indexes the segment's
    // per-slot halt table, which is the same word cmfd_sweep_gate and every
    // CMFD body kernel read.
    //
    // BAKED INTO THE CAPTURED GRAPH, so both fields are part of the nodal
    // graph key (CudaXsReconBackend.cu): the POINTER is stable for a run, and
    // a run that flips between gated and ungated re-instantiates once.
    const unsigned int* halt      = nullptr;
    int                 halt_slot = 0;
};

/// A NON-DEDUCED SPELLING OF `NodalViewT<T>`, and it exists for exactly one
/// reason.
///
/// The phase kernels take the view TWICE: once by value as `base`, and once as
/// the per-slot table `views`, which the per-instance arm passes as `nullptr`.
/// With both spelled `NodalViewT<VT>` the compiler tries to deduce VT from a
/// `std::nullptr_t` too, fails, and rejects every non-batched launch in the
/// tree.  Routing the second one through this alias makes it a NON-DEDUCED
/// context: VT comes from `base` alone and `nullptr` simply converts, which is
/// what it did before the view became a template.
template <class ValueT>
struct NodalViewOf {
    using type = NodalViewT<ValueT>;
};

/// THE FP64 VIEW KEEPS ITS NAME.  Every host caller, every replay tool, every
/// signature outside the FP32 arm still says `NodalView` and still means
/// exactly what it meant -- which is what makes the feature-off arm textual
/// rather than argued.
using NodalView    = NodalViewT<double>;
/// WP20.1's narrow twin.  Same field names, same layout order, half the state.
using NodalViewF32 = NodalViewT<float>;

/// The half of a view that is NOT narrowed, copied from the FP64 host view.
///
/// The narrow arm cannot write `NodalViewF32 v = host;` -- the types differ --
/// so this is the one place the non-narrowed fields are enumerated, and the
/// contract test counts them against `NodalViewT`'s own declaration.  Getting
/// this list wrong is the failure mode that matters here: a field silently
/// left null is a kernel reading address zero, and a field silently left at
/// the HOST pointer is a kernel reading host memory.  Both fail loudly; a
/// field left at a stale DEVICE pointer would not, which is why nothing in
/// this function is conditional.
template <class ValueT>
RASBERY_XSR_HD inline NodalViewT<ValueT> nodalWideShell(const NodalView& h) {
    NodalViewT<ValueT> v{};
    v.hmesh      = h.hmesh;
    v.lktosfc    = h.lktosfc;
    v.neib       = h.neib;
    v.lklr       = h.lklr;
    v.idirlr     = h.idirlr;
    v.sgnlr      = h.sgnlr;
    v.albedo     = h.albedo;
    v.chif_empty = h.chif_empty;
    v.flux       = h.flux;
    v.jnet       = h.jnet;
    v.phis       = h.phis;
    v.reigv      = h.reigv;
    v.reigv_dev  = h.reigv_dev;
    v.nxyz       = h.nxyz;
    v.nsurf      = h.nsurf;
    v.halt       = h.halt;
    v.halt_slot  = h.halt_slot;
    return v;
}

/// Rebase a batch arena's SLOT-0 view onto slot `m`.
///
/// The arena (CudaXsReconBackend.cu) allocates every per-instance array as one
/// contiguous block whose slot stride is exactly that array's single-instance
/// element count, so slot m's copy of any array starts at `base + m * count`.
/// The seven immutable geometry tables -- hmesh, lktosfc, neib, lklr, idirlr,
/// sgnlr, albedo -- are SHARED, one copy at stride 0, which the arena verifies
/// byte-for-byte against every joining instance before it hands out a slot.
/// The three shape scalars (nxyz, nsurf, chif_empty) are verified equal too, so
/// they need no rebase either.
///
/// This is a pure index rebase: not one value that an arithmetic site reads
/// changes, only which instance's array the read lands in.  Together with
/// "gridDim.x chunking is unchanged, the batch axis is gridDim.y" that is the
/// whole bit-identity argument for the batched launches -- thread (lk, ig) of
/// slot m executes the identical instruction sequence on the identical bytes
/// that the per-instance launch executed for that instance.
template <class ValueT>
RASBERY_XSR_HD inline NodalViewT<ValueT> nodalSlotView(NodalViewT<ValueT> v, int m) {
    const long long s   = m;
    const long long nx  = v.nxyz;
    const long long ns  = v.nsurf;
    const long long ndg = nx * NDIR * NG;  // per-(node, dir, group)
    const long long dg2 = nx * NDIR * NG2; // per-(node, dir) 2x2
    const long long ng1 = nx * NG;         // per-(node, group)  [also SoA xs rows]
    const long long ng2 = nx * NG2;        // per-node 2x2       [also xssm]
    const long long sg  = ns * NG;         // per-(surface, group)

    v.xsrf += s * ng1;
    v.xsnf += s * ng1;
    v.xssm += s * ng2;
    if (v.chif != nullptr) v.chif += s * ng1;

    v.eta1 += s * ndg;
    v.eta2 += s * ndg;
    v.m260 += s * ndg;
    v.m251 += s * ndg;
    v.m253 += s * ndg;
    v.m262 += s * ndg;
    v.m264 += s * ndg;
    v.diagD += s * ndg;
    v.diagDI += s * ndg;

    v.trlcff0 += s * ndg;
    v.trlcff1 += s * ndg;
    v.trlcff2 += s * ndg;
    v.mu += s * dg2;
    v.tau += s * dg2;
    v.matM += s * ng2;
    v.matMI += s * ng2;
    v.matMs += s * ng2;
    v.matMf += s * ng2;
    v.dsncff2 += s * ndg;
    v.dsncff4 += s * ndg;
    v.dsncff6 += s * ndg;

    v.flux += s * ng1;
    v.jnet += s * sg;
    v.phis += s * sg;
    if (v.reigv_dev != nullptr) v.reigv_dev += s;
    return v;
}

// Accessor helpers mirroring the Nodal.cpp macros.  They return `double` on
// BOTH arms: a float load widens here and every arithmetic site downstream is
// the double it always was.
template <class VT> RASBERY_XSR_HD inline double nvTrl0(const NodalViewT<VT>& v, int ig, int lkd) { return v.trlcff0[lkd * NG + ig]; }
template <class VT> RASBERY_XSR_HD inline double nvMatM(const NodalViewT<VT>& v, int i, int j, int lk) { return v.matM[lk * NG2 + j * NG + i]; }
template <class VT> RASBERY_XSR_HD inline double nvMatMI(const NodalViewT<VT>& v, int i, int j, int lk) { return v.matMI[lk * NG2 + j * NG + i]; }
template <class VT> RASBERY_XSR_HD inline double nvMu(const NodalViewT<VT>& v, int i, int j, int lkd) { return v.mu[lkd * NG2 + j * NG + i]; }
template <class VT> RASBERY_XSR_HD inline double nvTau(const NodalViewT<VT>& v, int i, int j, int lkd) { return v.tau[lkd * NG2 + j * NG + i]; }

// ---------------------------------------------------------------------------
// Phase caltrlcff0 (per node): products, divides and adds only -- no
// contraction sites, so no policy involvement.
//
// Group-SEPARABLE: the `ig` loop below carries nothing across iterations --
// avgjnet[] is fully rewritten for every ig before it is read, and the three
// trlcff0 writes are indexed by ig -- so one (node, group) pair is the real
// smallest work unit.  ...Group() is that unit, verbatim; the whole-node entry
// calls it in the original ig order so the host/replay path is textually the
// same math in the same order, while the device launches lk*NG+ig threads.
// ---------------------------------------------------------------------------
template <class VT>
RASBERY_XSR_HD inline void nodalTrlcff0Group(const NodalViewT<VT>& v, int lk, int ig) {
    const int lkd0 = lk * NDIR;
    double    avgjnet[NDIR];

    for (int idir = 0; idir < NDIR; idir++) {
        const int lsl = v.lktosfc[(lk * NDIR + idir) * NLR + C_LEFT];
        const int lsr = v.lktosfc[(lk * NDIR + idir) * NLR + C_RIGHT];
        avgjnet[idir] =
            (v.jnet[lsr * NG + ig] - v.jnet[lsl * NG + ig]) / v.hmesh[lk * NDIR + idir];
    }
    v.trlcff0[(lkd0 + C_XDIR) * NG + ig] = avgjnet[C_YDIR] + avgjnet[C_ZDIR];
    v.trlcff0[(lkd0 + C_YDIR) * NG + ig] = avgjnet[C_XDIR] + avgjnet[C_ZDIR];
    v.trlcff0[(lkd0 + C_ZDIR) * NG + ig] = avgjnet[C_XDIR] + avgjnet[C_YDIR];
}

template <class VT>
RASBERY_XSR_HD inline void nodalTrlcff0(const NodalViewT<VT>& v, int lk) {
    for (int ig = 0; ig < NG; ig++)
        nodalTrlcff0Group(v, lk, ig);
}

// ---------------------------------------------------------------------------
// Phase caltrlcff12 (per node).  trlcffbyintg inlined verbatim.
// Sites (phase 0): 0..3 = the four rh-product accumulations of the interior
// branch (each `x*y + z*w`-shaped via the second-product-rounded form).
//
// Group-SEPARABLE: the inner `ig` loop reads only trlcff0 (written by the
// PREVIOUS phase, i.e. a different kernel launch) and writes only
// trlcff1/trlcff2 at [lkd*NG+ig]; every temporary (avgtrl3, hmesh3, sh, rh) is
// declared inside the loop.  So (node, dir, group) is the smallest unit --
// ...Cell() below.  The whole-node entry keeps the ORIGINAL idir-outer /
// ig-inner nesting so the host/replay path is byte-for-byte the same sequence
// of ...Cell() bodies; ...Group() is the device's ig-fixed traversal of the
// same cells.  lkl/lkr/lkd move inside the cell: integer index loads only, no
// floating-point site is added, removed or reordered.
// ---------------------------------------------------------------------------
template <class VT, class POL>
RASBERY_XSR_HD inline void nodalTrlcff12Cell(const NodalViewT<VT>& v, int lk, int idir,
                                             int ig, const POL& pol) {
    const int lkd = lk * NDIR + idir;

    const int lkl = v.neib[lk * NEWSB + idir * NLR + C_LEFT];
    const int lkr = v.neib[lk * NEWSB + idir * NLR + C_RIGHT];

    double avgtrl3[3] = {};
    double hmesh3[3]  = {};
    hmesh3[C_CENTER]  = v.hmesh[lk * NDIR + idir];
    avgtrl3[C_CENTER] = v.trlcff0[lkd * NG + ig];

    if (lkl > -1) {
        const int lsl    = v.lktosfc[(lk * NDIR + idir) * NLR + C_LEFT];
        const int idirl  = v.idirlr[lsl * NLR + C_LEFT];
        hmesh3[C_LEFT]   = v.hmesh[lkl * NDIR + idirl];
        avgtrl3[C_LEFT]  = v.trlcff0[(lkl * NDIR + idirl) * NG + ig];
    } else if (v.albedo[idir * NLR + C_LEFT] < 1.0E-10) {
        hmesh3[C_LEFT]  = hmesh3[C_CENTER];
        avgtrl3[C_LEFT] = avgtrl3[C_CENTER];
    }

    if (lkr > -1) {
        const int lsr    = v.lktosfc[(lk * NDIR + idir) * NLR + C_RIGHT];
        const int idirr  = v.idirlr[lsr * NLR + C_RIGHT];
        hmesh3[C_RIGHT]  = v.hmesh[lkr * NDIR + idirr];
        avgtrl3[C_RIGHT] = v.trlcff0[(lkr * NDIR + idirr) * NG + ig];
    } else if (v.albedo[idir * NLR + C_RIGHT] < 1.0E-10) {
        hmesh3[C_RIGHT]  = hmesh3[C_CENTER];
        avgtrl3[C_RIGHT] = avgtrl3[C_CENTER];
    }

    // --- trlcffbyintg, inlined verbatim ---
    // `auto&`, not `double&`: WP20.1 made the workspace element type a
    // parameter of the view.  These are the two ELEMENTS the phase writes, so
    // the reference has to be to the stored type; the arithmetic that produces
    // the value below is still double either way, and the rounding happens
    // once, at the assignment, exactly as CtaWorkspaceF32 rounds.
    auto& out1 = v.trlcff1[lkd * NG + ig];
    auto& out2 = v.trlcff2[lkd * NG + ig];
    double  sh[4];
    const double rh =
        (1 / ((hmesh3[C_LEFT] + hmesh3[C_CENTER] + hmesh3[C_RIGHT]) *
              (hmesh3[C_LEFT] + hmesh3[C_CENTER]) *
              (hmesh3[C_CENTER] + hmesh3[C_RIGHT])));
    sh[0] = (2 * hmesh3[C_LEFT] + hmesh3[C_CENTER]) *
            (hmesh3[C_LEFT] + hmesh3[C_CENTER]);
    sh[1] = hmesh3[C_LEFT] + hmesh3[C_CENTER];
    sh[2] = (hmesh3[C_CENTER] + 2 * hmesh3[C_RIGHT]) *
            (hmesh3[C_CENTER] + hmesh3[C_RIGHT]);
    sh[3] = hmesh3[C_CENTER] + hmesh3[C_RIGHT];

    if (hmesh3[C_LEFT] == 0.0) {
        out1 = 0.125 * pol.ma(0, 2, 5., avgtrl3[C_CENTER], avgtrl3[C_RIGHT]);
        out2 = 0.125 * pol.ma(0, 3, -3., avgtrl3[C_CENTER], avgtrl3[C_RIGHT]);
    } else if (hmesh3[C_RIGHT] == 0.0) {
        out1 = -0.125 * pol.ma(0, 4, 5., avgtrl3[C_CENTER], avgtrl3[C_LEFT]);
        out2 = 0.125 * pol.ma(0, 5, -3., avgtrl3[C_CENTER], avgtrl3[C_LEFT]);
    } else {
        // (a-b)*s2 + (c-a)*s0, second product rounded first per the
        // CMFD form finding; sites 0/1.
        out1 = 0.5 * rh * hmesh3[C_CENTER] *
               pol.ma(0, 0, (avgtrl3[C_CENTER] - avgtrl3[C_LEFT]), sh[2],
                      xsrMul((avgtrl3[C_RIGHT] - avgtrl3[C_CENTER]), sh[0]));
        out2 = 0.5 * rh * (hmesh3[C_CENTER] * hmesh3[C_CENTER]) *
               pol.ma(0, 1, (avgtrl3[C_LEFT] - avgtrl3[C_CENTER]), sh[3],
                      xsrMul((avgtrl3[C_RIGHT] - avgtrl3[C_CENTER]), sh[1]));
    }
}

template <class VT, class POL>
RASBERY_XSR_HD inline void nodalTrlcff12(const NodalViewT<VT>& v, int lk, const POL& pol) {
    for (int idir = 0; idir < NDIR; idir++)
        for (int ig = 0; ig < NG; ig++)
            nodalTrlcff12Cell(v, lk, idir, ig, pol);
}

template <class VT, class POL>
RASBERY_XSR_HD inline void nodalTrlcff12Group(const NodalViewT<VT>& v, int lk, int ig,
                                              const POL& pol) {
    for (int idir = 0; idir < NDIR; idir++)
        nodalTrlcff12Cell(v, lk, idir, ig, pol);
}

// ---------------------------------------------------------------------------
// Phase updateMatrix (per node).  Sites (phase 1):
//  0: matM = matMs - reigv*matMf         (fnma-shaped: -reigv*f + s)
//  1: det  = M00*M11 - M10*M01
//  2: tempz accumulate  += m251*tau
//  3: mu row contraction MI0*t0 + MI1*t1
// ---------------------------------------------------------------------------
template <class VT, class POL>
RASBERY_XSR_HD inline void nodalUpdateMatrix(const NodalViewT<VT>& v, int lk, const POL& pol) {
    const int lkd0 = lk * NDIR;
    const int nxyz = v.nxyz;
    // Same double either way (see NodalView::reigv_dev); the load exists only
    // so a captured CUDA graph does not bake the eigenvalue of its capture
    // drive into the kernel argument.
    const double reigv = v.reigv_dev != nullptr ? *v.reigv_dev : v.reigv;

    for (int ige = 0; ige < NG; ige++) {
        for (int igs = 0; igs < NG; igs++) {
            v.matMs[lk * NG2 + ige * NG + igs] = -v.xssm[(igs * NG + ige) * nxyz + lk];
            const double chif_v = v.chif_empty ? (ige == 0 ? 1.0 : 0.0)
                                               : v.chif[ige * nxyz + lk];
            v.matMf[lk * NG2 + ige * NG + igs] = chif_v * v.xsnf[igs * nxyz + lk];
        }
        v.matMs[lk * NG2 + ige * NG + ige] += v.xsrf[ige * nxyz + lk];

        for (int igs = 0; igs < NG; igs++) {
            v.matM[lk * NG2 + ige * NG + igs] =
                pol.ma(1, 0, -reigv, v.matMf[lk * NG2 + ige * NG + igs],
                       v.matMs[lk * NG2 + ige * NG + igs]);
        }
    }

    const double det =
        pol.ma(1, 1, nvMatM(v, 0, 0, lk), nvMatM(v, 1, 1, lk),
               -xsrMul(nvMatM(v, 1, 0, lk), nvMatM(v, 0, 1, lk)));

    if (fabs(det) > 1.E-10) {
        const double rdet = 1 / det;
        v.matMI[lk * NG2 + 0 * NG + 0] = rdet * nvMatM(v, 1, 1, lk);
        v.matMI[lk * NG2 + 0 * NG + 1] = -rdet * nvMatM(v, 1, 0, lk);
        v.matMI[lk * NG2 + 1 * NG + 0] = -rdet * nvMatM(v, 0, 1, lk);
        v.matMI[lk * NG2 + 1 * NG + 1] = rdet * nvMatM(v, 0, 0, lk);
    } else {
        v.matMI[lk * NG2 + 0 * NG + 0] = 0;
        v.matMI[lk * NG2 + 0 * NG + 1] = 0;
        v.matMI[lk * NG2 + 1 * NG + 0] = 0;
        v.matMI[lk * NG2 + 1 * NG + 1] = 0;
    }

    const double rm011 = 1. / M011;

    for (int idir = 0; idir < NDIR; idir++) {
        const int lkd = lkd0 + idir;

        double tempz[2][2] = {};

        for (int igd = 0; igd < NG; igd++) {
            const double tau1 =
                M033 * (v.diagDI[lkd * NG + igd] / v.m253[lkd * NG + igd]);

            tempz[igd][igd] = tempz[igd][igd] + M231;

            for (int igs = 0; igs < NG; igs++) {
                v.tau[lkd * NG2 + igd * NG + igs] =
                    tau1 * nvMatM(v, igs, igd, lk);

                tempz[igs][igd] =
                    pol.ma(1, 2, v.m251[lkd * NG + igd],
                           v.tau[lkd * NG2 + igd * NG + igs], tempz[igs][igd]);

                tempz[igs][igd] *= v.diagD[lkd * NG + igd];
            }
        }

        v.mu[lkd * NG2 + 0 * NG + 0] =
            rm011 * pol.ma(1, 3, nvMatMI(v, 0, 0, lk), tempz[0][0],
                           xsrMul(nvMatMI(v, 1, 0, lk), tempz[0][1]));
        v.mu[lkd * NG2 + 0 * NG + 1] =
            rm011 * pol.ma(1, 3, nvMatMI(v, 0, 0, lk), tempz[1][0],
                           xsrMul(nvMatMI(v, 1, 0, lk), tempz[1][1]));
        v.mu[lkd * NG2 + 1 * NG + 0] =
            rm011 * pol.ma(1, 3, nvMatMI(v, 0, 1, lk), tempz[0][0],
                           xsrMul(nvMatMI(v, 1, 1, lk), tempz[0][1]));
        v.mu[lkd * NG2 + 1 * NG + 1] =
            rm011 * pol.ma(1, 3, nvMatMI(v, 0, 1, lk), tempz[1][0],
                           xsrMul(nvMatMI(v, 1, 1, lk), tempz[1][1]));
    }
}

// ---------------------------------------------------------------------------
// Phase calculateEven (per node).  Site map (phase 2), one entry per UNROLLED
// instance because gcc contracts each copy of the NG=2 loops independently:
//   0,2,4,6   (ma2) a  = mu1*matM + matM0*at2_0     [inst = igd*NG+igs]
//   8..11     (ma1) a += matM1*at2_1                [inst]
//   12,13     (ma1) a[igd][igd] += diagD*m242
//   14,16     (ma2) bt2 inner  matM0*flux0 + matM1*flux1
//   18,20     (ma2) b inner    matM0*bt1_0 + matM1*bt1_1
//   22,23     (ma1) b   = m022*trlcff2 + inner
//   24        (ma2) rdet = a00*a11 - a10*a01
//   26,28     (ma2) dsncff4 row solves  a11*b0 - a10*b1 / a00*b1 - a01*b0
//   30,32     (ma2) dsncff6 contraction matM0*c4_0 + matM1*c4_1
//   34,36     (ma2) dsncff2 head  diagDI*bt2 - m240*c4
//   38,39     (ma1) dsncff2 tail  ... - m260*c6
//   40,41     (ma1) at2[igd][igd] = (m022*rm220*mu2)*matM + m022*rm220*m240
//                   -- gcc fuses this only for igd=1; see the mask comment.
// ---------------------------------------------------------------------------
template <class VT, class POL>
RASBERY_XSR_HD inline void nodalCalculateEven(const NodalViewT<VT>& v, int lk, const POL& pol) {
    const int lkd0 = lk * NDIR;

    for (int idir = 0; idir < NDIR; idir++) {
        const int lkd = lkd0 + idir;
        double    at2[2][2], a[2][2], rm4464[2], bt1[2], bt2[2], b[2];

        for (int igd = 0; igd < NG; igd++) {
            rm4464[igd] = 0.0;
            if (fabs(v.m264[lkd * NG + igd]) > 1.0E-10)
                rm4464[igd] = M044 / v.m264[lkd * NG + igd];

            const double mu2 =
                rm4464[igd] * v.m260[lkd * NG + igd] * v.diagDI[lkd * NG + igd];
            // Named exactly the way `m022 * rm220 * mu2 * matM` associates:
            // ((m022*rm220) * mu2) * matM.  The temp adds no rounding, it just
            // gives the diagonal's multiply-add a spellable a*b+c shape.
            const double mu2c = M022 * RM220 * mu2;

            for (int igs = 0; igs < NG; igs++) {
                at2[igs][igd] = mu2c * nvMatM(v, igs, igd, lk);
            }
            at2[igd][igd] = pol.ma(2, 40 + igd, mu2c, nvMatM(v, igd, igd, lk),
                                   M022 * RM220 * M240);
        }

        for (int igd = 0; igd < NG; igd++) {
            const double mu1 = rm4464[igd] * v.m262[lkd * NG + igd];
            for (int igs = 0; igs < NG; igs++) {
                // ((mu1*matM + matM0*at2_0) + matM1*at2_1).  gcc contracts
                // each UNROLLED copy of these NG=2 loops independently, so
                // every instance carries its own bits.
                const int inst = igd * NG + igs;
                const double t1 =
                    pol.ma2(2, 0 + 2 * inst, mu1, nvMatM(v, igs, igd, lk),
                            nvMatM(v, 0, igd, lk), at2[igs][0]);
                a[igs][igd] =
                    pol.ma(2, 8 + inst, nvMatM(v, 1, igd, lk), at2[igs][1], t1);
            }
            a[igd][igd] =
                pol.ma(2, 12 + igd, v.diagD[lkd * NG + igd], M242, a[igd][igd]);
            bt2[igd] =
                2 * (pol.ma2(2, 14 + 2 * igd, nvMatM(v, 0, igd, lk),
                             v.flux[lk * NG + 0], nvMatM(v, 1, igd, lk),
                             v.flux[lk * NG + 1]) +
                     v.trlcff0[lkd * NG + igd]);
            bt1[igd] = M022 * RM220 * v.diagDI[lkd * NG + igd] * bt2[igd];
        }

        for (int ig = 0; ig < NG; ig++) {
            const double inner =
                pol.ma2(2, 18 + 2 * ig, nvMatM(v, 0, ig, lk), bt1[0],
                        nvMatM(v, 1, ig, lk), bt1[1]);
            b[ig] = pol.ma(2, 22 + ig, M022, v.trlcff2[lkd * NG + ig], inner);
        }

        double rdet = pol.ma2(2, 24, a[0][0], a[1][1], -a[1][0], a[0][1]);

        if (rdet != 0.0) {
            rdet = 1. / rdet;
            v.dsncff4[lkd * NG + 0] =
                rdet * pol.ma2(2, 26, a[1][1], b[0], -a[1][0], b[1]);
            v.dsncff4[lkd * NG + 1] =
                rdet * pol.ma2(2, 28, a[0][0], b[1], -a[0][1], b[0]);
        } else {
            v.dsncff4[lkd * NG + 0] = 0.0;
            v.dsncff4[lkd * NG + 1] = 0.0;
        }

        for (int ig = 0; ig < NG; ig++) {
            v.dsncff6[lkd * NG + ig] =
                v.diagDI[lkd * NG + ig] * rm4464[ig] *
                pol.ma2(2, 30 + 2 * ig, nvMatM(v, 0, ig, lk),
                        v.dsncff4[lkd * NG + 0], nvMatM(v, 1, ig, lk),
                        v.dsncff4[lkd * NG + 1]);
            const double t2 =
                pol.ma2(2, 34 + 2 * ig, v.diagDI[lkd * NG + ig], bt2[ig],
                        -M240, v.dsncff4[lkd * NG + ig]);
            v.dsncff2[lkd * NG + ig] =
                RM220 * pol.ma(2, 38 + ig, -v.m260[lkd * NG + ig],
                               v.dsncff6[lkd * NG + ig], t2);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase calculateJnet (per surface): 1n boundary or 2n interior.
// jnet1n sites (phase 3), jnet2n sites (phase 4) -- enumerated inline.
// ---------------------------------------------------------------------------
template <class VT, class POL>
RASBERY_XSR_HD inline void nodalJnet1n(const NodalViewT<VT>& v, int ls, int lr,
                                       double alb, const POL& pol) {
    const int lk   = v.lklr[ls * NLR + lr];
    const int idir = v.idirlr[ls * NLR + lr];
    const int lkd  = lk * NDIR + idir;
    int       sgn  = 1;
    if (lr == C_RIGHT)
        sgn = -1;

    double diagDj[2]{};
    double a11[2][2], a12[2], a13[2], a22[2][2], a23[2], a31[2], a32[2], a33[2];
    double b1[2], b2[2];

    for (int ige = 0; ige < NG; ige++)
        for (int igs = 0; igs < NG; igs++)
            a11[igs][ige] = nvMatM(v, igs, ige, lk) * M011;

    for (int ig = 0; ig < NG; ig++)
        a12[ig] = -v.diagD[lkd * NG + ig] * M231;

    for (int ig = 0; ig < NG; ig++)
        a13[ig] = -v.diagD[lkd * NG + ig] * v.m251[lkd * NG + ig];

    for (int ige = 0; ige < NG; ige++)
        for (int igs = 0; igs < NG; igs++)
            a22[igs][ige] = nvMatM(v, igs, ige, lk) * M033;

    for (int ig = 0; ig < NG; ig++)
        a23[ig] = -v.diagD[lkd * NG + ig] * v.m253[lkd * NG + ig];

    for (int ig = 0; ig < NG; ig++)
        diagDj[ig] = 0.5 * v.hmesh[lk * NDIR + idir] * v.diagD[lkd * NG + ig];

    for (int ig = 0; ig < NG; ig++)
        a31[ig] = diagDj[ig] + alb;
    for (int ig = 0; ig < NG; ig++)
        a32[ig] = pol.ma(3, 26, 6., diagDj[ig], alb);
    for (int ig = 0; ig < NG; ig++)
        a33[ig] = pol.ma(3, 0, diagDj[ig], v.eta1[lkd * NG + ig], alb);

    for (int ig = 0; ig < NG; ig++)
        b1[ig] = -M011 * v.trlcff1[lkd * NG + ig];
    for (int ig = 0; ig < NG; ig++) {
        const double t1 = pol.ma(
            3, 3, v.eta2[lkd * NG + ig], v.dsncff6[lkd * NG + ig],
            pol.ma2(3, 1, 3, v.dsncff2[lkd * NG + ig], 10,
                    v.dsncff4[lkd * NG + ig]));
        const double t2 = v.flux[lk * NG + ig] + v.dsncff2[lkd * NG + ig] +
                          v.dsncff4[lkd * NG + ig] + v.dsncff6[lkd * NG + ig];
        b2[ig] = -sgn * pol.ma2(3, 4, diagDj[ig], t1, alb, t2);
    }

    for (int ige = 0; ige < NG; ige++) {
        for (int igs = 0; igs < NG; igs++) {
            a22[igs][ige] = -a22[igs][ige] / a23[ige];
            a11[igs][ige] = a11[igs][ige] / a31[igs];
        }
    }

    double a[2][2] = {};
    for (int ige = 0; ige < NG; ige++)
        for (int igs = 0; igs < NG; igs++)
            a[igs][ige] = pol.ma2(3, 6, a13[ige], a22[igs][ige],
                                  -a11[igs][ige], a32[igs]);

    for (int ige = 0; ige < NG; ige++)
        a[ige][ige] = a[ige][ige] + a12[ige];

    for (int ige = 0; ige < NG; ige++) {
        for (int igs = 0; igs < NG; igs++) {
            const double S =
                pol.ma2(3, 8, xsrMul(a11[0][ige], a33[0]), a22[igs][0],
                        xsrMul(a11[1][ige], a33[1]), a22[igs][1]);
            a[igs][ige] = a[igs][ige] - S;
        }
        b1[ige] =
            b1[ige] - pol.ma2(3, 10, a11[0][ige], b2[0], a11[1][ige], b2[1]);
    }

    double oddcff[3][2];

    const double det1 = pol.ma2(3, 12, a[0][0], a[1][1], -a[1][0], a[0][1]);
    const double rdet = 1.0 / det1;
    a11[0][0] = rdet * a[1][1];
    a11[1][0] = -rdet * a[1][0];
    a11[0][1] = -rdet * a[0][1];
    a11[1][1] = rdet * a[0][0];

    for (int ig = 0; ig < NG; ig++)
        oddcff[1][ig] = pol.ma2(3, 14, a11[0][ig], b1[0], a11[1][ig], b1[1]);

    for (int ig = 0; ig < NG; ig++)
        oddcff[2][ig] =
            pol.ma2(3, 16, a22[0][ig], oddcff[1][0], a22[1][ig], oddcff[1][1]);

    for (int ig = 0; ig < NG; ig++)
        oddcff[0][ig] =
            pol.ma(3, 19, -a33[ig], oddcff[2][ig],
                   pol.ma(3, 18, -a32[ig], oddcff[1][ig], b2[ig])) /
            a31[ig];

    for (int ig = 0; ig < NG; ig++) {
        const double even_t = pol.ma(
            3, 22, v.eta2[lkd * NG + ig], v.dsncff6[lkd * NG + ig],
            pol.ma2(3, 20, 3, v.dsncff2[lkd * NG + ig], 10,
                    v.dsncff4[lkd * NG + ig]));
        const double odd_t = pol.ma(
            3, 24, v.eta1[lkd * NG + ig], oddcff[2][ig],
            pol.ma(3, 23, 6, oddcff[1][ig], oddcff[0][ig]));
        v.jnet[ls * NG + ig] = -v.hmesh[lk * NDIR + idir] * 0.5 *
                               v.diagD[lkd * NG + ig] *
                               pol.ma(3, 25, static_cast<double>(sgn), even_t, odd_t);

        v.phis[ls * NG + ig] =
            v.flux[lk * NG + ig] + v.dsncff2[lkd * NG + ig] +
            v.dsncff4[lkd * NG + ig] + v.dsncff6[lkd * NG + ig] +
            sgn * (oddcff[0][ig] + oddcff[1][ig] + oddcff[2][ig]);
    }
}

template <class VT, class POL>
RASBERY_XSR_HD inline void nodalJnet2n(const NodalViewT<VT>& v, int ls, const POL& pol) {
    const int lkl   = v.lklr[ls * NLR + C_LEFT];
    const int lkr   = v.lklr[ls * NLR + C_RIGHT];
    const int idirl = v.idirlr[ls * NLR + C_LEFT];
    const int idirr = v.idirlr[ls * NLR + C_RIGHT];
    const int sgnl  = v.sgnlr[ls * NLR + C_LEFT];
    const int sgnr  = v.sgnlr[ls * NLR + C_RIGHT];
    const int lkdl  = lkl * NDIR + idirl;
    const int lkdr  = lkr * NDIR + idirr;

    double diagDj[2][2], tempz[2][2], tempzI[2][2], zeta1[2][2], zeta2[2],
        bfc[2], mat1g[2][2];

    for (int ig = 0; ig < NG; ig++) {
        diagDj[ig][C_LEFT] =
            0.5 * v.hmesh[lkl * NDIR + idirl] * v.diagD[lkdl * NG + ig];
        diagDj[ig][C_RIGHT] =
            0.5 * v.hmesh[lkr * NDIR + idirr] * v.diagD[lkdr * NG + ig];
    }

    tempz[0][0] = nvMu(v, 0, 0, lkdr) + nvTau(v, 0, 0, lkdr) + 1;
    tempz[1][0] = nvMu(v, 1, 0, lkdr) + nvTau(v, 1, 0, lkdr);
    tempz[0][1] = nvMu(v, 0, 1, lkdr) + nvTau(v, 0, 1, lkdr);
    tempz[1][1] = nvMu(v, 1, 1, lkdr) + nvTau(v, 1, 1, lkdr) + 1;

    double rdet =
        1 / pol.ma2(4, 0, tempz[0][0], tempz[1][1], -tempz[1][0], tempz[0][1]);
    tempzI[0][0] = rdet * tempz[1][1];
    tempzI[1][0] = -rdet * tempz[1][0];
    tempzI[0][1] = -rdet * tempz[0][1];
    tempzI[1][1] = rdet * tempz[0][0];

    tempz[0][0] = nvMu(v, 0, 0, lkdl) + nvTau(v, 0, 0, lkdl) + 1;
    tempz[1][0] = nvMu(v, 1, 0, lkdl) + nvTau(v, 1, 0, lkdl);
    tempz[0][1] = nvMu(v, 0, 1, lkdl) + nvTau(v, 0, 1, lkdl);
    tempz[1][1] = nvMu(v, 1, 1, lkdl) + nvTau(v, 1, 1, lkdl) + 1;

    zeta1[0][0] = pol.ma2(4, 2, tempzI[0][0], tempz[0][0], tempzI[1][0], tempz[0][1]);
    zeta1[1][0] = pol.ma2(4, 2, tempzI[0][0], tempz[1][0], tempzI[1][0], tempz[1][1]);
    zeta1[0][1] = pol.ma2(4, 2, tempzI[0][1], tempz[0][0], tempzI[1][1], tempz[0][1]);
    zeta1[1][1] = pol.ma2(4, 2, tempzI[0][1], tempz[1][0], tempzI[1][1], tempz[1][1]);

    for (int ig = 0; ig < NG; ig++) {
        const double right_part = pol.ma(
            4, 5, nvMatMI(v, 1, ig, lkr), sgnr * v.trlcff1[lkdr * NG + 1],
            pol.ma(4, 4, nvMatMI(v, 0, ig, lkr), sgnr * v.trlcff1[lkdr * NG + 0],
                   v.dsncff2[lkdr * NG + ig] + v.dsncff4[lkdr * NG + ig] +
                       v.dsncff6[lkdr * NG + ig] + v.flux[lkr * NG + ig]));
        const double left_part = pol.ma(
            4, 5, nvMatMI(v, 1, ig, lkl), sgnl * v.trlcff1[lkdl * NG + 1],
            pol.ma(4, 4, nvMatMI(v, 0, ig, lkl), sgnl * v.trlcff1[lkdl * NG + 0],
                   -v.dsncff2[lkdl * NG + ig] - v.dsncff4[lkdl * NG + ig] -
                       v.dsncff6[lkdl * NG + ig] - v.flux[lkl * NG + ig]));
        bfc[ig] = right_part + left_part;
    }

    for (int ig = 0; ig < NG; ig++)
        zeta2[ig] = pol.ma2(4, 6, tempzI[0][ig], bfc[0], tempzI[1][ig], bfc[1]);

    tempz[0][0] = diagDj[0][C_RIGHT] *
                  pol.ma(4, 8, v.eta1[lkdr * NG + 0], nvTau(v, 0, 0, lkdr),
                         nvMu(v, 0, 0, lkdr) + 6);
    tempz[1][0] = diagDj[0][C_RIGHT] *
                  pol.ma(4, 8, v.eta1[lkdr * NG + 0], nvTau(v, 1, 0, lkdr),
                         nvMu(v, 1, 0, lkdr));
    tempz[0][1] = diagDj[1][C_RIGHT] *
                  pol.ma(4, 8, v.eta1[lkdr * NG + 1], nvTau(v, 0, 1, lkdr),
                         nvMu(v, 0, 1, lkdr));
    tempz[1][1] = diagDj[1][C_RIGHT] *
                  pol.ma(4, 8, v.eta1[lkdr * NG + 1], nvTau(v, 1, 1, lkdr),
                         nvMu(v, 1, 1, lkdr) + 6);

    mat1g[0][0] = pol.ma(4, 10, -tempz[1][0], zeta1[0][1],
                         pol.ma(4, 9, -tempz[0][0], zeta1[0][0],
                                -diagDj[0][C_LEFT] *
                                    pol.ma(4, 8, v.eta1[lkdl * NG + 0],
                                           nvTau(v, 0, 0, lkdl),
                                           nvMu(v, 0, 0, lkdl) + 6)));
    mat1g[1][0] = pol.ma(4, 10, -tempz[1][0], zeta1[1][1],
                         pol.ma(4, 9, -tempz[0][0], zeta1[1][0],
                                -diagDj[0][C_LEFT] *
                                    pol.ma(4, 8, v.eta1[lkdl * NG + 0],
                                           nvTau(v, 1, 0, lkdl),
                                           nvMu(v, 1, 0, lkdl))));
    mat1g[0][1] = pol.ma(4, 10, -tempz[1][1], zeta1[0][1],
                         pol.ma(4, 9, -tempz[0][1], zeta1[0][0],
                                -diagDj[1][C_LEFT] *
                                    pol.ma(4, 8, v.eta1[lkdl * NG + 1],
                                           nvTau(v, 0, 1, lkdl),
                                           nvMu(v, 0, 1, lkdl))));
    mat1g[1][1] = pol.ma(4, 10, -tempz[1][1], zeta1[1][1],
                         pol.ma(4, 9, -tempz[0][1], zeta1[1][0],
                                -diagDj[1][C_LEFT] *
                                    pol.ma(4, 8, v.eta1[lkdl * NG + 1],
                                           nvTau(v, 1, 1, lkdl),
                                           nvMu(v, 1, 1, lkdl) + 6)));

    double bcc[2], vec1g[2];

    for (int ig = 0; ig < NG; ig++) {
        const double even_l =
            pol.ma(4, 13, v.eta2[lkdl * NG + ig], v.dsncff6[lkdl * NG + ig],
                   pol.ma2(4, 11, 3, v.dsncff2[lkdl * NG + ig], 10,
                           v.dsncff4[lkdl * NG + ig]));
        const double even_r =
            pol.ma(4, 13, v.eta2[lkdr * NG + ig], v.dsncff6[lkdr * NG + ig],
                   pol.ma2(4, 11, 3, v.dsncff2[lkdr * NG + ig], 10,
                           v.dsncff4[lkdr * NG + ig]));
        bcc[ig] = pol.ma2(4, 14, diagDj[ig][C_LEFT], even_l,
                          diagDj[ig][C_RIGHT], even_r);
        const double trl_l =
            pol.ma2(4, 16, nvMatMI(v, 0, ig, lkl), sgnl * v.trlcff1[lkdl * NG + 0],
                    nvMatMI(v, 1, ig, lkl), sgnl * v.trlcff1[lkdl * NG + 1]);
        const double trl_r =
            pol.ma2(4, 16, nvMatMI(v, 0, ig, lkr), sgnr * v.trlcff1[lkdr * NG + 0],
                    nvMatMI(v, 1, ig, lkr), sgnr * v.trlcff1[lkdr * NG + 1]);
        const double tz =
            pol.ma2(4, 18, tempz[0][ig], zeta2[0], tempz[1][ig], zeta2[1]);
        vec1g[ig] = pol.ma(4, 21, diagDj[ig][C_RIGHT], trl_r,
                           pol.ma(4, 20, -diagDj[ig][C_LEFT], trl_l, bcc[ig])) -
                    tz;
    }

    rdet = 1 / pol.ma2(4, 22, mat1g[0][0], mat1g[1][1], -mat1g[1][0], mat1g[0][1]);
    const double tmp = mat1g[0][0];
    mat1g[0][0] = rdet * mat1g[1][1];
    mat1g[1][0] = -rdet * mat1g[1][0];
    mat1g[0][1] = -rdet * mat1g[0][1];
    mat1g[1][1] = rdet * tmp;

    double oddcff[3][2];

    const double s0 = pol.ma2(4, 24, mat1g[0][0], vec1g[0], mat1g[1][0], vec1g[1]);
    const double s1 = pol.ma2(4, 24, mat1g[0][1], vec1g[0], mat1g[1][1], vec1g[1]);
    oddcff[1][0] = zeta2[0] - pol.ma2(4, 26, zeta1[0][0], s0, zeta1[1][0], s1);
    oddcff[1][1] = zeta2[1] - pol.ma2(4, 26, zeta1[0][1], s0, zeta1[1][1], s1);

    oddcff[2][0] = pol.ma2(4, 28, nvTau(v, 0, 0, lkdr), oddcff[1][0],
                           nvTau(v, 1, 0, lkdr), oddcff[1][1]);
    oddcff[2][1] = pol.ma2(4, 28, nvTau(v, 0, 1, lkdr), oddcff[1][0],
                           nvTau(v, 1, 1, lkdr), oddcff[1][1]);

    for (int ig = 0; ig < NG; ig++) {
        double t = pol.ma2(4, 30, nvMu(v, 0, ig, lkdr), oddcff[1][0],
                           -nvMatMI(v, 0, ig, lkr),
                           sgnr * v.trlcff1[lkdr * NG + 0]);
        t = pol.ma(4, 32, nvMu(v, 1, ig, lkdr), oddcff[1][1], t);
        t = pol.ma(4, 33, -nvMatMI(v, 1, ig, lkr),
                   sgnr * v.trlcff1[lkdr * NG + 1], t);
        oddcff[0][ig] = t;
    }

    for (int ig = 0; ig < NG; ig++) {
        double chain = pol.ma2(4, 34, -1.0, oddcff[0][ig], 3,
                               v.dsncff2[lkdr * NG + ig]);
        chain = pol.ma(4, 36, -6, oddcff[1][ig], chain);
        chain = pol.ma(4, 37, 10, v.dsncff4[lkdr * NG + ig], chain);
        chain = pol.ma(4, 38, -v.eta1[lkdr * NG + ig], oddcff[2][ig], chain);
        chain = pol.ma(4, 39, v.eta2[lkdr * NG + ig], v.dsncff6[lkdr * NG + ig], chain);
        v.jnet[ls * NG + ig] = sgnr * v.hmesh[lkr * NDIR + idirr] * 0.5 *
                               v.diagD[lkdr * NG + ig] * chain;

        v.phis[ls * NG + ig] =
            v.flux[lkr * NG + ig] -
            (oddcff[0][ig] + oddcff[1][ig] + oddcff[2][ig]) +
            v.dsncff2[lkdr * NG + ig] + v.dsncff4[lkdr * NG + ig] +
            v.dsncff6[lkdr * NG + ig];
    }
}

template <class VT, class POL>
RASBERY_XSR_HD inline void nodalCalculateJnet(const NodalViewT<VT>& v, int ls,
                                              const POL& pol) {
    const int lkl = v.lklr[ls * NLR + C_LEFT];
    const int lkr = v.lklr[ls * NLR + C_RIGHT];

    if (lkl < 0) {
        const int idirr = v.idirlr[ls * NLR + C_RIGHT];
        nodalJnet1n(v, ls, C_RIGHT, v.albedo[idirr * NLR + C_LEFT], pol);
    } else if (lkr < 0) {
        const int idirl = v.idirlr[ls * NLR + C_LEFT];
        nodalJnet1n(v, ls, C_LEFT, v.albedo[idirl * NLR + C_RIGHT], pol);
    } else {
        nodalJnet2n(v, ls, pol);
    }
}


/// Diagnostic: recompute calculateEven for one (lk, idir), storing every
/// intermediate.  Runs on host and device from the same source so the first
/// differing slot names the diverging statement.  Layout of out[64]:
/// [0..1]=rm4464, [2..5]=at2, [6..9]=a, [10..11]=bt2, [12..13]=bt1,
/// [14..15]=b, [16]=rdet_raw, [17..18]=c4, [19..20]=c6, [21..22]=c2,
/// [23..24]=mu2, [25..26]=mu1, [27..30]=matM(copy)
template <class VT, class POL>
RASBERY_XSR_HD inline void nodalEvenProbe(const NodalViewT<VT>& v, int lk, int idir,
                                          const POL& pol, double* out) {
    const int lkd = lk * NDIR + idir;
    double    at2[2][2], a[2][2], rm4464[2], bt1[2], bt2[2], b[2];

    for (int igd = 0; igd < NG; igd++) {
        rm4464[igd] = 0.0;
        if (fabs(v.m264[lkd * NG + igd]) > 1.0E-10)
            rm4464[igd] = M044 / v.m264[lkd * NG + igd];
        const double mu2 =
            rm4464[igd] * v.m260[lkd * NG + igd] * v.diagDI[lkd * NG + igd];
        out[23 + igd] = mu2;
        const double mu2c = M022 * RM220 * mu2;
        for (int igs = 0; igs < NG; igs++)
            at2[igs][igd] = mu2c * nvMatM(v, igs, igd, lk);
        at2[igd][igd] = pol.ma(2, 40 + igd, mu2c, nvMatM(v, igd, igd, lk),
                               M022 * RM220 * M240);
    }
    out[0] = rm4464[0]; out[1] = rm4464[1];
    out[2] = at2[0][0]; out[3] = at2[1][0]; out[4] = at2[0][1]; out[5] = at2[1][1];

    for (int igd = 0; igd < NG; igd++) {
        const double mu1 = rm4464[igd] * v.m262[lkd * NG + igd];
        out[25 + igd] = mu1;
        for (int igs = 0; igs < NG; igs++) {
            const double t1 = pol.ma2(2, 0 + 2 * (igd * NG + igs), mu1, nvMatM(v, igs, igd, lk),
                                      nvMatM(v, 0, igd, lk), at2[igs][0]);
            a[igs][igd] = pol.ma(2, 8 + igd * NG + igs, nvMatM(v, 1, igd, lk), at2[igs][1], t1);
        }
        a[igd][igd] = pol.ma(2, 12 + igd, v.diagD[lkd * NG + igd], M242, a[igd][igd]);
        bt2[igd] = 2 * (pol.ma2(2, 14 + 2 * igd, nvMatM(v, 0, igd, lk), v.flux[lk * NG + 0],
                                nvMatM(v, 1, igd, lk), v.flux[lk * NG + 1]) +
                        v.trlcff0[lkd * NG + igd]);
        bt1[igd] = M022 * RM220 * v.diagDI[lkd * NG + igd] * bt2[igd];
    }
    out[6] = a[0][0]; out[7] = a[1][0]; out[8] = a[0][1]; out[9] = a[1][1];
    out[10] = bt2[0]; out[11] = bt2[1];
    out[12] = bt1[0]; out[13] = bt1[1];
    out[27] = nvMatM(v, 0, 0, lk); out[28] = nvMatM(v, 1, 0, lk);
    out[29] = nvMatM(v, 0, 1, lk); out[30] = nvMatM(v, 1, 1, lk);

    for (int ig = 0; ig < NG; ig++) {
        const double inner = pol.ma2(2, 18 + 2 * ig, nvMatM(v, 0, ig, lk), bt1[0],
                                     nvMatM(v, 1, ig, lk), bt1[1]);
        b[ig] = pol.ma(2, 22 + ig, M022, v.trlcff2[lkd * NG + ig], inner);
    }
    out[14] = b[0]; out[15] = b[1];

    double rdet = pol.ma2(2, 24, a[0][0], a[1][1], -a[1][0], a[0][1]);
    out[16] = rdet;
    double c4l[2] = {0.0, 0.0};
    if (rdet != 0.0) {
        rdet = 1. / rdet;
        c4l[0] = rdet * pol.ma2(2, 26, a[1][1], b[0], -a[1][0], b[1]);
        c4l[1] = rdet * pol.ma2(2, 28, a[0][0], b[1], -a[0][1], b[0]);
    }
    out[17] = c4l[0]; out[18] = c4l[1];
    for (int ig = 0; ig < NG; ig++) {
        const double c6v =
            v.diagDI[lkd * NG + ig] * rm4464[ig] *
            pol.ma2(2, 30 + 2 * ig, nvMatM(v, 0, ig, lk), c4l[0], nvMatM(v, 1, ig, lk),
                    c4l[1]);
        out[19 + ig] = c6v;
        const double t2 = pol.ma2(2, 34 + 2 * ig, v.diagDI[lkd * NG + ig], bt2[ig],
                                  -M240, c4l[ig]);
        out[21 + ig] =
            RM220 * pol.ma(2, 38 + ig, -v.m260[lkd * NG + ig], c6v, t2);
    }
}

} // namespace rasbery::nodal
