#pragma once

// Shared host/device bodies of the SPLIT Xe fixed-point step -- Rev.7.1 Task 13
// (GPU Xe), behind RASBERY_GPU_XE (default off).
//
// ---------------------------------------------------------------------------
// WHAT THIS IS, AND WHY IT IS NOT XsReconKernel.h
// ---------------------------------------------------------------------------
//
// XsReconKernel.h carries ONE FUSED operation: evaluate the equilibrium map,
// damp it, write the three Xe-chain rows and reconstruct, all in one node body.
// That is exactly XSSet::UpdateEquilibriumXenon, and it is the right shape for
// the plain Picard cascade.
//
// The safeguarded Anderson arm (Driver.h, plan Rev.4 Sec 10) cannot use it: it
// has to look at F(x) BEFORE it commits anything, so on the host it drives
// SnapshotXenon / EvaluateEquilibriumXenon / CommitXenon instead.  This header
// is the device half of that split, and it is assembled out of the SAME pieces
// the fused body is made of -- xsreconImageNode for (a)-(d) and
// xsreconReconstructNode for (f) -- rather than out of a second copy of them.
// A second copy is the failure this campaign has already paid for once
// (CmfdOuterFormMiner.cpp's header): two bodies that are "obviously the same"
// diverge in the last bits and nothing says so.
//
// Everything genuinely NEW lives here, and it is small: the Anderson algebra
// (three inner products' worth of accumulation, and the candidate combination)
// plus the fixed-partition reduction that replaces the host's serial fold.
//
// ---------------------------------------------------------------------------
// TWO CLASSES, AND THE LINE BETWEEN THEM
// ---------------------------------------------------------------------------
//
//   PICARD PATH  (RASBERY_XE_ANDERSON=0)          B0, bit-gated
//       evaluate -> damp -> commit -> reconstruct.  Every expression is the
//       fused body's, split at a point where only DOUBLES cross (the image
//       triple), so the split cannot change a rounding.  It is bit-identical
//       to the host loop for the same reason the fused kernel is.
//
//   ANDERSON PATH (RASBERY_XE_ANDERSON=1)         N1, Gate A/B
//       the inner products are the reason, and the only reason.  The host
//       XeDot accumulates ~3*n_fuel terms (about 15,000 on kngr_238) into ONE
//       running sum, in ordinal order; a device reduction that reproduces that
//       exactly is a single serial thread, which is measurably slower than the
//       host it replaces.  So the device uses a FIXED PARTITION -- a partition
//       count that depends on nothing but a compile-time constant and the fuel
//       count -- which is deterministic run to run but associates the additions
//       differently from the host.  Different association, different last bits,
//       different trajectory: that is an N1 change and it is gated as one.
//
//       RASBERY_GPU_XE_DOT_PARTITIONS=1 collapses the partition to the host's
//       single fold and IS bit-exact.  It exists so the N1 claim can be split
//       in two -- "the algebra is right" (partitions=1, bit-exact) and "the
//       partition is what moved it" (partitions>1) -- instead of being one
//       undifferentiated difference.
//
// ---------------------------------------------------------------------------
// THE CONTRACTION MASK
// ---------------------------------------------------------------------------
//
// The Anderson algebra is new arithmetic, so it needs the same treatment
// CmfdOuterKernel.h's mask gets: which multiply-adds THE HOST COMPILER fused is
// a property of the build machine, and a baked constant is right on one host
// and wrong on the next.  XeFormMiner.cpp mines it from this binary at startup
// and RASBERY_XE_FORMS overrides it; see XeFormMine.h for the sites.
//
// This header must stay compilable by both g++ and nvcc: no STL containers, no
// exceptions, no allocation.  Same rule and same reason as XsReconKernel.h.

#include "XsReconKernel.h"

namespace rasbery::xe {

namespace xsr = rasbery::xsrecon;

#if defined(__CUDACC__)
    #define RASBERY_XE_HD __host__ __device__
#else
    #define RASBERY_XE_HD
#endif

// ---------------------------------------------------------------------------
// Contraction sites (XeFormMine.h mines these; RASBERY_XE_FORMS overrides)
// ---------------------------------------------------------------------------
//
// Host expression, Driver.h XeDot:
//     sum += a.i135[i]*b.i135[i] + a.xe135[i]*b.xe135[i] + a.xe135m[i]*b.xe135m[i];
// which parses as sum + ((p1 + p2) + p3).  -ffp-contract=fast may fuse ONE of
// p1/p2 into the first add and p3 into the second; the final `sum + t` has no
// multiply left to fuse.  Three states for the first site, two for the second.
//
// Host expression, TryAndersonXeStep's candidate loop:
//     vi -= gamma[j]*df[j].i135[i];
// TWO SITES, ONE PER WINDOW WIDTH, and that is a measurement rather than a
// guess: on the authoring host gcc emits DIFFERENT code for the one-column and
// two-column trip counts of that inner loop -- it fuses at ncol == 1 and does
// not at ncol == 2 -- so a single shared bit cannot be right for both, and the
// mining says so by never reaching zero.  A one-column step is not a rare
// corner either: it is the secant fallback the host takes whenever the
// two-column Gram matrix is ill-conditioned.
constexpr int XE_DOT_FIRST_BIT = 0; ///< 2 bits, 3 states
constexpr int XE_DOT_THIRD_BIT = 2; ///< 1 bit
constexpr int XE_CAND1_BIT     = 3; ///< 1 bit, the one-column window
constexpr int XE_CAND2_BIT     = 4; ///< 1 bit, the two-column window

// WP7 stage C (RASBERY_GPU_XE_TXN).  FOUR MORE SITES, AND THEY ARE NEW ONLY IN
// WHERE THEY RUN.  The transaction moves Driver.h's normal equations onto the
// device so a step needs no host observation between the dots and the commit.
// The arithmetic is not allowed to move with them: TXN=1 must reproduce TXN=1's
// own predecessor -- the current RASBERY_GPU_XE arm -- bit for bit, and that
// arm computes these four expressions with g++ at -O3, where -ffp-contract=fast
// may fuse either product of each into the add.  The device TU is built
// --fmad=false, so nothing fuses there unless it is written as an fma; which
// one to write is a property of the BUILD MACHINE and therefore mined, exactly
// like the dot's and the candidate's sites.
//
// The four host expressions, from TryAndersonXeStepGpu:
//
//     det      = a * c - b * b;
//     gamma[0] = (c * p - b * q) / det;
//     gamma[1] = (a * q - b * p) / det;
//     proj     = gamma[0] * p + gamma[1] * q;
//
// Each is one add with two multiplies feeding it, so each has three states --
// neither product fused, the first fused, the second fused -- and takes two
// bits.  What is NOT a site: `XE_ANDERSON_MIN_GRAM * a * c` and
// `max_step * picard` are multiply chains with no add; `p / a` is a division;
// `gg - proj` and the one-column `gamma[j] * p` have one multiply or none.
constexpr int XE_TXN_DET_BIT   = 5;  ///< 2 bits, 3 states: a*c - b*b
constexpr int XE_TXN_G0_BIT    = 7;  ///< 2 bits, 3 states: c*p - b*q
constexpr int XE_TXN_G1_BIT    = 9;  ///< 2 bits, 3 states: a*q - b*p
constexpr int XE_TXN_PROJ_BIT  = 11; ///< 2 bits, 3 states: g0*p + g1*q
constexpr int XE_BIT_COUNT     = 13;

/// THE TWO SUB-MASKS, AND WHY THE SPLIT IS A CONSTANT AND NOT A CONVENTION.
///
/// Bits 0..4 are the SHIPPED channel: the fixed-partition dot and the candidate
/// loop, the two things the production RASBERY_GPU_XE arm launches on every Xe
/// step whatever RASBERY_GPU_XE_TXN says.  Bits 5..12 are the ALGEBRA channel:
/// WP7-C's normal equations, which only the transaction evaluates.
///
/// 8919331 already stopped the algebra channel's SOUNDNESS from demoting the
/// shipped channel's VALUE.  It did not stop the algebra channel's BITS from
/// travelling into the shipped kernels' `forms` argument: on 238 the mined mask
/// went 0xd -> 0xd2d and the split arm's kernels were handed 0xd2d, relying on
/// every consumer inside them to extract its own two-bit field and ignore the
/// rest.  They do (xeDotChunk masks &3 and &1, xeCandidateOrdinal masks &1), so
/// the arithmetic did not move -- but "relies on every future reader to mask"
/// is not a contract, it is a hope, and the WP7-C episode is what a broken hope
/// costs.  The shipped kernels are now handed `resolved & XE_SHIPPED_FORMS`
/// and the algebra bits are unrepresentable in that argument.
///
/// tools/test_xe_forms_shipped_split_contract.py pins both halves.
constexpr unsigned long long XE_SHIPPED_FORMS =
    (1ull << XE_TXN_DET_BIT) - 1ull; ///< bits 0..4 -- dot + candidate, 0x1f
constexpr unsigned long long XE_ALGEBRA_FORMS =
    (((1ull << XE_BIT_COUNT) - 1ull) & ~XE_SHIPPED_FORMS); ///< bits 5..12, 0x1fe0

/// States of the first dot site.
constexpr unsigned XE_DOT_FIRST_NONE = 0u; ///< both products rounded, plain add
constexpr unsigned XE_DOT_FIRST_P1   = 1u; ///< fma(a1,b1, mul(a2,b2))
constexpr unsigned XE_DOT_FIRST_P2   = 2u; ///< fma(a2,b2, mul(a1,b1))

/// Per-build default.  A RECORD OF THE MACHINE IT WAS MEASURED ON, not a value
/// the run depends on: XeFormMiner mines this host's answer at startup and the
/// mined value wins (rasbery::gpu::resolveCalibratedFormMask).  It is kept so
/// the receipt can say when a host disagrees with the shipped record instead of
/// that being discovered three campaigns later as an unexplained ULP.
///
///     WSL2 / g++ 13.3 dev box, g++ -O3 -march=native    XE_FORMS = 0xd
///
/// Reading it: the dot's first site is state 1 (the FIRST product fused into
/// the add), the dot's third product is fused, the one-column candidate is
/// fused and the two-column one is not.  All four sites were measured decisive
/// on this host by test/xe_form_probe.cpp -- none of them is a coin toss.
/// WP7-C adds bits 5..12 and they are ZERO here, which is not a measurement --
/// it is the absence of one.  Nothing was ever measured for those four sites on
/// the authoring host because the authoring host has no nvcc, so the shipped
/// record says "unfused" and the MINING is what the run actually uses.  The
/// [RASBERY][FORMS] receipt is where a host that disagrees with this record
/// announces itself, and on the first 238 build it is expected to.
constexpr unsigned long long XE_FORMS_DEFAULT = 0xdull;

/// States of a two-product site.  ONE ENCODING FOR ALL FOUR of the WP7-C sites,
/// so a reader checks the meaning once: 0 is both products separately rounded,
/// 1 fuses the FIRST product into the add, 2 fuses the SECOND.  Deliberately
/// the same numbering XE_DOT_FIRST_* uses.
constexpr unsigned XE_SITE_NONE = 0u;
constexpr unsigned XE_SITE_P1   = 1u;
constexpr unsigned XE_SITE_P2   = 2u;

/// THE HOST MASK -- the same four sites, the same two-bit fields, the same
/// XE_SITE_* encoding, applied to `Driver.h::TryAndersonXeStepGpu`'s OWN copy
/// of the normal equations rather than to the device's.
///
/// ---------------------------------------------------------------------------
/// WHY A HOST MASK EXISTS AT ALL -- THE INVERTED A/B, 238, 2026-08-31
/// ---------------------------------------------------------------------------
///
/// `048c6c1` put RASBERY_NEVER_INLINE on TryAndersonXeStepGpuTxn to stop a
/// default-off arm from being folded into SolveLoop, and on 238 the flag-off
/// trajectory MOVED AGAIN, the other way: digest c1a5d9116df9edb3 / 4601
/// outers, where the same tree with only that token removed gives
/// 22b9a3187bfb4beb / 4566 -- the `7cfe3a4` value.  Every other observation
/// points the same way (47161ed clean, 47161ed+71092e2 drifted, d7b81af and
/// 8919331 and 32ac308 drifted, the kernel-side hunk drifted).
///
/// So the class in `docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md` §7 was
/// right and its REMEDY was a coin: the flag-off trajectory flips with ANY
/// change of gcc's inlining and codegen context around this arm, because the
/// four expressions
///
///     det      = a * c - b * b;
///     gamma[0] = (c * p - b * q) / det;
///     gamma[1] = (a * q - b * p) / det;
///     proj     = gamma[0] * p + gamma[1] * q;
///
/// were UNBARRIERED host arithmetic compiled at -O3 -ffp-contract=fast in a
/// single translation unit with no LTO, and which multiply gcc folds into each
/// add is a decision it re-makes per inlining context.  `d85984e` did not fix
/// anything; it landed on the `7cfe3a4` form by luck, and the next edit near
/// SolveLoop would have moved it off again.
///
/// The fix is not another attribute.  It is to REMOVE THE DECISION: every one
/// of the four sites now goes through xeSiteSub/xeSiteAdd, whose three states
/// are all written with xsr::xsrFma and xsr::xsrMul -- and xsrMul carries an
/// `asm volatile` barrier on gcc, so no surrounding context can re-fuse it.
/// Once the whole path is barriered, inlining cannot change the result, and
/// the arm's trajectory becomes a property of THIS CONSTANT instead of a
/// property of the call graph.
///
/// ---------------------------------------------------------------------------
/// WHY IT IS A CONSTANT AND NOT MINED
/// ---------------------------------------------------------------------------
///
/// Mining answers "what does gcc do to a QUOTATION of these expressions, in
/// XeAlgebraReference.cpp, with that TU's operand provenance" -- which is
/// exactly the question `src/XeFormAudit.h` documents as the wrong one.  There
/// is no fixture that can reach the production call site.  So the value is
/// PINNED BY MEASUREMENT INSTEAD: the 238 runner sweeps the 81 combinations
/// with RASBERY_XE_HOST_FORMS and keeps the one whose `[RASBERY][TRAJECTORY]`
/// digest is 22b9a3187bfb4beb / 4566 outers.  See §8.4 of the regression doc
/// for the loop.
///
///     PROVISIONAL: 0xaa0 -- all four sites XE_SITE_P1.
///
/// That is gcc's usual `convert_mult_to_fma` outcome (the pass reaches the
/// FIRST multiply feeding the add and fuses it), and it is a PREDICTION, not a
/// measurement.  It is written here rather than left at zero because zero is
/// also a guess and a wrong non-zero guess announces itself in the sweep,
/// where an all-none default would look like "the sweep was never run".  The
/// moment 238 reports the winning combination this line is re-pinned to it and
/// the receipt's `source` field says `build_default` again.
///
/// Bits outside XE_ALGEBRA_FORMS are not representable in this mask: the
/// resolver masks them off.  The dot and the candidate are DEVICE sites and
/// keep being decided by XE_FORMS.
constexpr unsigned long long XE_HOST_FORMS_DEFAULT =
    (static_cast<unsigned long long>(XE_SITE_P1) << XE_TXN_DET_BIT) |
    (static_cast<unsigned long long>(XE_SITE_P1) << XE_TXN_G0_BIT) |
    (static_cast<unsigned long long>(XE_SITE_P1) << XE_TXN_G1_BIT) |
    (static_cast<unsigned long long>(XE_SITE_P1) << XE_TXN_PROJ_BIT);

static_assert((XE_HOST_FORMS_DEFAULT & ~XE_ALGEBRA_FORMS) == 0ull,
              "the host form mask lives in the algebra channel, bits 5..12; a bit "
              "outside it would be a dot/candidate decision made by the wrong knob");

/// The number of contiguous partitions the device inner product is cut into.
/// FIXED -- it depends on no launch parameter, no occupancy, no device -- which
/// is what makes the reduction reproducible run to run.  Overridable with
/// RASBERY_GPU_XE_DOT_PARTITIONS; 1 reproduces the host fold exactly.
constexpr int XE_DOT_PARTITIONS_DEFAULT = 1024;

/// Hard ceiling on the partition count, so a typo in the override cannot ask
/// for an allocation that dwarfs the vectors being reduced.
constexpr int XE_DOT_PARTITIONS_MAX = 65536;

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// One (I-135, Xe-135, Xe-135m) field over the fuel nodes, indexed by fuel-node
/// ORDINAL -- position in XSSet::fuel_nodes(), NOT node index.  Deliberately
/// the same indexing and the same three-array shape the host XeTriple uses, so
/// a host/device A/B compares like with like and neither half has to know the
/// SoA stride.
struct XeTriple {
    double* i135;
    double* xe135;
    double* xe135m;
};

/// Read-only twin of XeTriple.
struct XeTripleConst {
    const double* i135;
    const double* xe135;
    const double* xe135m;
};

/// Slot ids of the device-resident Anderson history.  The order is the
/// declaration order of Driver.h's XeAndersonState, so the two are readable
/// side by side; XE_TRIPLE_COUNT sizes the device block.
enum XeTripleId {
    XE_T_X      = 0, ///< x_k, the iterate the map was evaluated at
    XE_T_F      = 1, ///< F(x_k)
    XE_T_G      = 2, ///< g_k = F(x_k) - x_k
    XE_T_F_PREV = 3,
    XE_T_G_PREV = 4,
    XE_T_CAND   = 5, ///< the proposed x_{k+1}
    XE_T_DF0    = 6,
    XE_T_DF1    = 7,
    XE_T_DG0    = 8,
    XE_T_DG1    = 9,
    XE_TRIPLE_COUNT = 10
};

/// Anderson window depth, mirroring Driver.h's XE_ANDERSON_DEPTH.  The df/dg
/// slot ids above are written out for depth 2 exactly as the host normal
/// equations are; a depth-3 experiment needs both changed together.
constexpr int XE_DEPTH = 2;

static_assert(XE_T_DF0 + XE_DEPTH == XE_T_DG0, "df slots must be contiguous");
static_assert(XE_T_DG0 + XE_DEPTH == XE_TRIPLE_COUNT, "dg slots must be contiguous");

// ---------------------------------------------------------------------------
// Node bodies
// ---------------------------------------------------------------------------

/// Evaluate the map at ONE fuel ordinal without applying it: x (the snapshot),
/// F(x) and g = F(x) - x, plus the RAW relative Xe-135 step.
///
/// THE SKIP IS THE HOST'S SKIP.  XSSet::EvaluateEquilibriumXenon seeds its
/// outputs with the snapshot and `continue`s on a node whose normalized flux is
/// not positive, so F(x) = x there and the node contributes nothing to the
/// residual.  Written out here as data -- f = x, g = 0, change = 0 -- because a
/// device kernel has no "leave the output alone" option: every ordinal is
/// written by exactly one thread, every launch.
///
/// `processed` receives 1/0 so the Picard commit can reproduce the FUSED
/// update's skip too: that path does not write and does not reconstruct a
/// skipped node, where CommitXenon reconstructs every fuel node it is given.
/// The two are different functions with different contracts and the device has
/// to honour both.
RASBERY_XE_HD inline void xeEvaluateOrdinal(const xsr::BatchView& v, int k,
                                            XeTriple x, XeTriple f, XeTriple g,
                                            unsigned char* processed,
                                            double* change_out) {
    const int    l    = v.fuel[k];
    const int    nxyz = v.nxyz;
    const double xi   = v.iden[xsr::I135 * nxyz + l];
    const double xx   = v.iden[xsr::XE135 * nxyz + l];
    const double xm   = v.iden[xsr::XE135M * nxyz + l];
    x.i135[k]         = xi;
    x.xe135[k]        = xx;
    x.xe135m[k]       = xm;

    double           iden[xsr::NISO];
    xsr::XeNodeImage img;
    double           change = 0.0;
    if (!xsr::xsreconImageNode(v, l, iden, &img, &change)) {
        f.i135[k]   = xi;
        f.xe135[k]  = xx;
        f.xe135m[k] = xm;
        g.i135[k]   = 0.0;
        g.xe135[k]  = 0.0;
        g.xe135m[k] = 0.0;
        processed[k] = 0u;
        *change_out  = 0.0;
        return;
    }

    f.i135[k]    = img.i135;
    f.xe135[k]   = img.xe135;
    f.xe135m[k]  = img.xe135m;
    // g = F(x) - x, elementwise, exactly as the host XeSub forms it.
    g.i135[k]    = img.i135 - xi;
    g.xe135[k]   = img.xe135 - xx;
    g.xe135m[k]  = img.xe135m - xm;
    processed[k] = 1u;
    *change_out  = change;
}

/// out = a - b at one ordinal, over all three rows (host XeSub).
RASBERY_XE_HD inline void xeSubOrdinal(XeTripleConst a, XeTripleConst b, XeTriple out,
                                       int k) {
    out.i135[k]   = a.i135[k] - b.i135[k];
    out.xe135[k]  = a.xe135[k] - b.xe135[k];
    out.xe135m[k] = a.xe135m[k] - b.xe135m[k];
}

/// The candidate x_{k+1} = F_k - sum_j gamma_j dF_j at one ordinal, together
/// with the two things the safeguards need from it:
///
///   `physics_bad` set when any component is non-finite or negative -- a
///   negative Xe-135 is not a slightly wrong inventory, it is a negative
///   absorption cross section handed to the flux solve (host SAFEGUARD 3/4);
///   `step_out`    |cand - x| / max(|cand|, 1e-30) on Xe-135, the same raw
///                 metric UpdateEquilibriumXenon returns, for the trust region
///                 (host XeRelativeChange).
///
/// Both reduce with OR and MAX, which are order-insensitive and therefore say
/// the same thing whatever order the blocks retire in.
RASBERY_XE_HD inline void xeCandidateOrdinal(XeTripleConst f, XeTripleConst x,
                                             const double* df_i, const double* df_x,
                                             const double* df_m, int ncol,
                                             const double* gamma, XeTriple cand, int k,
                                             int n_fuel, unsigned long long forms,
                                             int* physics_bad, double* step_out) {
    double vi = f.i135[k];
    double vx = f.xe135[k];
    double vm = f.xe135m[k];
    // One bit per window width: see the XE_CAND1_BIT / XE_CAND2_BIT comment.
    const int  cand_bit = (ncol <= 1) ? XE_CAND1_BIT : XE_CAND2_BIT;
    const bool fused    = ((forms >> cand_bit) & 1ull) != 0ull;
    for (int j = 0; j < ncol; ++j) {
        // ROW-MAJOR OVER TRIPLES: the history block is laid out [row][triple]
        // [ordinal], so the SAME row of consecutive window columns is n_fuel
        // apart and column j is just `+ j*n_fuel` from the df[0] row pointer.
        // The host indexes an array of structs and the device indexes one
        // block; the values are the same, and this is the one place the two
        // layouts have to be reconciled.
        const double dj_i = df_i[static_cast<long long>(j) * n_fuel + k];
        const double dj_x = df_x[static_cast<long long>(j) * n_fuel + k];
        const double dj_m = df_m[static_cast<long long>(j) * n_fuel + k];
        if (fused) {
            vi = xsr::xsrFma(-gamma[j], dj_i, vi);
            vx = xsr::xsrFma(-gamma[j], dj_x, vx);
            vm = xsr::xsrFma(-gamma[j], dj_m, vm);
        } else {
            vi = vi - xsr::xsrMul(gamma[j], dj_i);
            vx = vx - xsr::xsrMul(gamma[j], dj_x);
            vm = vm - xsr::xsrMul(gamma[j], dj_m);
        }
    }
    cand.i135[k]   = vi;
    cand.xe135[k]  = vx;
    cand.xe135m[k] = vm;

    // finite AND non-negative, in one pair of comparisons per row.  A NaN
    // fails both (every comparison against it is false), -inf fails the lower
    // bound and +inf the upper, so this is `std::isfinite(v) && v >= 0.0`
    // without reaching for a classifier whose host and device spellings differ
    // -- and without `v - v == 0.0`, which a compiler is entitled to fold.
    const bool ok = vi >= 0.0 && vi <= 1.0e308 && vx >= 0.0 && vx <= 1.0e308 &&
                    vm >= 0.0 && vm <= 1.0e308;
    *physics_bad  = ok ? 0 : 1;

    const double abs_cand = fabs(vx);
    const double scale    = (abs_cand < 1.0e-30) ? 1.0e-30 : abs_cand;
    *step_out             = fabs(vx - x.xe135[k]) / scale;
}

/// Commit one ordinal: write the three Xe-chain rows of node `l` and
/// reconstruct it.  EXACTLY the three rows ApplyXeEquilibrium owns, and nothing
/// else -- the rest of the isotope vector belongs to depletion (host
/// XSSet::CommitXenon).
RASBERY_XE_HD inline void xeCommitOrdinal(const xsr::BatchView& v, int k, double vi,
                                          double vx, double vm) {
    const int l    = v.fuel[k];
    const int nxyz = v.nxyz;

    v.iden[xsr::I135 * nxyz + l]   = vi;
    v.iden[xsr::XE135 * nxyz + l]  = vx;
    v.iden[xsr::XE135M * nxyz + l] = vm;

    // ReconstructNode re-reads the density vector from memory, so the local
    // copy is filled from v.iden AFTER the three rows above were stored -- the
    // operands are then the host's operands, including the rows just written.
    double iden[xsr::NISO];
    for (int iso = 0; iso < xsr::NISO; ++iso)
        iden[iso] = v.iden[iso * nxyz + l];
    xsr::xsreconReconstructNode(v, l, iden);
}

/// The damped Picard image at one ordinal: x + relax*(F(x) - x).
///
/// The `relax < 1.0` guard is the fused body's, kept for the same reason: it
/// makes the undamped case arithmetically identical to F(x) rather than relying
/// on 1.0*(a-b)+b == a.
RASBERY_XE_HD inline void xeBlendOrdinal(XeTripleConst f, XeTripleConst x, double relax,
                                         int k, double* vi, double* vx, double* vm) {
    const double fi = f.i135[k], fx = f.xe135[k], fm = f.xe135m[k];
    if (relax < 1.0) {
        const double oi = x.i135[k], ox = x.xe135[k], om = x.xe135m[k];
        *vi = xsr::xsrFma(relax, fi - oi, oi);
        *vx = xsr::xsrFma(relax, fx - ox, ox);
        *vm = xsr::xsrFma(relax, fm - om, om);
    } else {
        *vi = fi;
        *vx = fx;
        *vm = fm;
    }
}

// ---------------------------------------------------------------------------
// The inner product
// ---------------------------------------------------------------------------

/// <a,b> over the ordinals [i0, i1) of the concatenated 3*n_fuel vector, in
/// EXACTLY the host XeDot's per-ordinal form and in ascending ordinal order.
///
/// One thread owns one partition, start to finish, so the accumulation inside a
/// partition is serial and the partition boundaries are the only place the
/// association differs from the host's single fold.  With one partition there
/// is no such place and the two agree bit for bit.
RASBERY_XE_HD inline double xeDotChunk(XeTripleConst a, XeTripleConst b, int i0, int i1,
                                       unsigned long long forms) {
    const unsigned site0 =
        static_cast<unsigned>((forms >> XE_DOT_FIRST_BIT) & 3ull);
    const bool third = ((forms >> XE_DOT_THIRD_BIT) & 1ull) != 0ull;

    double sum = 0.0;
    for (int i = i0; i < i1; ++i) {
        double t;
        if (site0 == XE_DOT_FIRST_P1)
            t = xsr::xsrFma(a.i135[i], b.i135[i], xsr::xsrMul(a.xe135[i], b.xe135[i]));
        else if (site0 == XE_DOT_FIRST_P2)
            t = xsr::xsrFma(a.xe135[i], b.xe135[i], xsr::xsrMul(a.i135[i], b.i135[i]));
        else
            t = xsr::xsrMul(a.i135[i], b.i135[i]) + xsr::xsrMul(a.xe135[i], b.xe135[i]);

        t = third ? xsr::xsrFma(a.xe135m[i], b.xe135m[i], t)
                  : t + xsr::xsrMul(a.xe135m[i], b.xe135m[i]);

        sum += t;
    }
    return sum;
}

/// Partition p of [0, n) when the range is cut into `parts` contiguous pieces.
/// The remainder goes to the LOW partitions, one element each, so the boundary
/// depends on nothing but (n, parts) -- no launch shape, no device, no order of
/// arrival.  Both halves of an A/B must compute it the same way, so it is one
/// function and not two matching expressions.
RASBERY_XE_HD inline void xeDotPartitionRange(int n, int parts, int p, int* i0, int* i1) {
    const int base = n / parts;
    const int rem  = n - base * parts;
    const int lo   = p * base + (p < rem ? p : rem);
    const int hi   = lo + base + (p < rem ? 1 : 0);
    *i0 = lo;
    *i1 = hi;
}

/// Fold the partials in ascending partition order -- strict, serial, one
/// thread.  A tree here would put the association back in the scheduler's
/// hands, which is the whole thing the fixed partition exists to take out of
/// them.
RASBERY_XE_HD inline double xeDotFold(const double* partials, int parts) {
    double sum = 0.0;
    for (int p = 0; p < parts; ++p)
        sum += partials[p];
    return sum;
}

/// The six inner products the depth-2 normal equations need, in the order the
/// host reads them.  Pair `slot` -> (left triple id, right triple id) is a
/// table rather than a switch so the device kernel and the host harness cannot
/// disagree about which product landed in which slot.

enum XeDotSlot {
    XE_DOT_GG = 0, ///< <g,g>
    XE_DOT_A  = 1, ///< <dg0,dg0>
    XE_DOT_B  = 2, ///< <dg0,dg1>
    XE_DOT_C  = 3, ///< <dg1,dg1>
    XE_DOT_P  = 4, ///< <dg0,g>
    XE_DOT_Q  = 5, ///< <dg1,g>
    XE_DOT_COUNT = 6
};

// ---------------------------------------------------------------------------
// WP7 stage C -- the Xe device transaction
// ---------------------------------------------------------------------------
//
// WHAT MOVES, AND WHAT DOES NOT.  Today an Anderson Xe step is five device
// launches with FOUR host stream synchronisations threaded between them,
// because the host owns three decisions: is the step armed, did the 2x2 solve
// condition, did the candidate pass the safeguards.  Each decision reads a
// handful of doubles the device just computed, so each costs a full round trip.
// The arithmetic of those decisions is EIGHT DOUBLES WIDE.  It is not on the
// host because it is expensive; it is on the host because that is where it was
// written.
//
// So it moves, unchanged, into one single-thread kernel, and the ONLY thing
// that changes about a step is that nobody observes it in the middle.  The
// dots are consumed in the same slot order, the safeguards fire in the same
// sequence with the same constants, the same candidate is committed.  That is
// why WP7-C is a B0 claim against the current device arm and not an N1 one --
// the N1 line was crossed by the fixed partition, once, and it stays crossed
// exactly where it was.
//
// THE REJECTED STEP IS COMMITTED HERE TOO, and that is the part that has to be
// argued rather than asserted.  Today a refusal returns false and the caller
// runs UpdateEquilibriumXenon, which -- on this arm -- re-evaluates the map at
// a state nothing has written and commits x + relax*(F - x).  The transaction
// instead commits the F IT ALREADY HOLDS, from the same evaluation, with the
// same blend.  Those are the same bits: the map is a deterministic kernel over
// unchanged operands, so the second evaluation could only ever return the
// first's answer.  What it saves is that second evaluation -- one full
// 39-isotope condensation over every fuel node, on every rejected step.
//
// AND relax IS 1.0 ON THIS PATH, always.  Driver.h arms the Anderson attempt
// with `xe_anderson && xe_relax == 1.0 && flux_converged`, and the fallback it
// guards uses that same xe_relax.  The transaction therefore commits F itself
// (xeBlendOrdinal's undamped arm) rather than a blend, and the contract test
// asserts the guard that makes this true is still in the caller.

/// Why a step ended, in the order the safeguards are tested.  Downloaded once
/// per step so the host telemetry counts exactly what it counted before.
enum XeTxnReason {
    XE_TXN_ACCEPTED  = 0, ///< candidate passed every safeguard and was committed
    XE_TXN_NOT_ARMED = 1, ///< ncol == 0 or the residual is already under tolerance
    XE_TXN_CONDITION = 2, ///< SAFEGUARD 1/4: neither window width conditioned
    XE_TXN_RESIDUAL  = 3, ///< SAFEGUARD 2/4: the fit predicts no decrease
    XE_TXN_PHYSICS   = 4, ///< SAFEGUARD 3/4: a density went negative or non-finite
    XE_TXN_STEP      = 5  ///< SAFEGUARD 4/4: outside the trust region
};

/// The whole of one step's host-visible state, device-resident, written by the
/// two control kernels and read by the commit kernel.  ONE STRUCT AND ONE
/// DOWNLOAD: the four scalars the host still needs (the residual, the outcome,
/// the two counter increments) ride back on the transfer the commit's drain was
/// already paying for, so a transaction step adds no synchronisation of its own.
///
/// Plain old data, doubles first: it is cudaMemcpy'd as one block and the host
/// reads the same struct out of the same header.
struct XeTxnControl {
    double gamma[2];    ///< the fit coefficients, read by the candidate kernel
    double picard;      ///< the raw residual at x_k -- the caller's xe_change
    double step;        ///< the candidate's trust-region metric
    double proj;        ///< sum_j gamma_j <dG_j, g_k>
    double gg;          ///< <g,g>, kept so a rejection is readable in the receipt
    int    solved;      ///< the fit conditioned; the candidate kernel runs on this
    int    accept;      ///< 1 = commit XE_T_CAND, 0 = commit XE_T_F
    int    proposed;    ///< the step got past arming, i.e. it is a real proposal
    int    reason;      ///< XeTxnReason
    int    ncol;        ///< the window width this step used
    int    physics_bad; ///< the candidate kernel's OR-reduction
    int    pad[2];
};

/// isfinite() without reaching for a classifier whose host and device spellings
/// differ, and without `v - v == 0.0`, which a compiler is entitled to fold.
/// A NaN fails the comparison (every comparison against it is false) and either
/// infinity exceeds the largest finite double, so this IS std::isfinite.
RASBERY_XE_HD inline bool xeFinite(double v) {
    return fabs(v) <= 1.7976931348623157e308;
}

/// `u*v - w*z` under one mined site.  See XE_SITE_* for the encoding.
RASBERY_XE_HD inline double xeSiteSub(double u, double v, double w, double z,
                                      unsigned state) {
    if (state == XE_SITE_P1) return xsr::xsrFma(u, v, -xsr::xsrMul(w, z));
    if (state == XE_SITE_P2) return xsr::xsrFma(-w, z, xsr::xsrMul(u, v));
    return xsr::xsrMul(u, v) - xsr::xsrMul(w, z);
}

/// `u*v + w*z` under one mined site.
RASBERY_XE_HD inline double xeSiteAdd(double u, double v, double w, double z,
                                      unsigned state) {
    if (state == XE_SITE_P1) return xsr::xsrFma(u, v, xsr::xsrMul(w, z));
    if (state == XE_SITE_P2) return xsr::xsrFma(w, z, xsr::xsrMul(u, v));
    return xsr::xsrMul(u, v) + xsr::xsrMul(w, z);
}

RASBERY_XE_HD inline unsigned xeSiteState(unsigned long long forms, int bit) {
    return static_cast<unsigned>((forms >> bit) & 3ull);
}

/// THE LEAST SQUARES AND SAFEGUARD 1/4 (conditioning), and NOTHING ELSE.
///
/// SEPARATE FROM THE CONTROL BODY BECAUSE THE MINING NEEDS IT SEPARATE.  The
/// four contraction sites live in here, and XeFormMine.h scores them by running
/// this against XeAndersonReference.cpp's verbatim quotation of the same block
/// in Driver.h.  A scorer that had to go through xeAndersonSolveControl would
/// get no gamma at all whenever SAFEGUARD 2/4 rejected -- it writes them only on
/// the accepting path -- and would then mine a site out of the cases where it
/// happened not to fire.  Which is a statement about the fixture and not about
/// the compiler; CmfdOuterFormMiner.cpp names that failure by name.
///
/// Returns Driver.h's `solved`.  On false the outputs are untouched.
RASBERY_XE_HD inline bool xeAndersonFit(const double* dots, int ncol, double min_gram,
                                        unsigned long long forms, double* gamma0_out,
                                        double* gamma1_out, double* proj_out) {
    const double gg = dots[XE_DOT_GG];
    if (ncol == XE_DEPTH) {
        const double a   = dots[XE_DOT_A];
        const double b   = dots[XE_DOT_B];
        const double c   = dots[XE_DOT_C];
        const double p   = dots[XE_DOT_P];
        const double q   = dots[XE_DOT_Q];
        const double det = xeSiteSub(a, c, b, b, xeSiteState(forms, XE_TXN_DET_BIT));
        if (a > 0.0 && c > 0.0 && xeFinite(det) && xeFinite(p) && xeFinite(q) &&
            det > min_gram * a * c) {
            const double g0 =
                xeSiteSub(c, p, b, q, xeSiteState(forms, XE_TXN_G0_BIT)) / det;
            const double g1 =
                xeSiteSub(a, q, b, p, xeSiteState(forms, XE_TXN_G1_BIT)) / det;
            *gamma0_out = g0;
            *gamma1_out = g1;
            *proj_out   = xeSiteAdd(g0, p, g1, q, xeSiteState(forms, XE_TXN_PROJ_BIT));
            return true;
        }
    }
    // The newest column.  At ncol == 2 that is dg[1] -- slots C and Q -- and at
    // ncol == 1 it is dg[0], which is slots A and P.
    const int    j = ncol - 1;
    const double a = (j == 1) ? dots[XE_DOT_C] : dots[XE_DOT_A];
    const double p = (j == 1) ? dots[XE_DOT_Q] : dots[XE_DOT_P];
    if (a > 0.0 && xeFinite(a) && xeFinite(p) && a > min_gram * gg) {
        const double gj = p / a;
        *gamma0_out     = (j == 1) ? 0.0 : gj;
        *gamma1_out     = (j == 1) ? gj : 0.0;
        // ONE multiply, no add: no site here, and none is mined for it.
        *proj_out       = xsr::xsrMul(gj, p);
        return true;
    }
    return false;
}

/// THE NORMAL EQUATIONS AND SAFEGUARDS 1/4 AND 2/4, in Driver.h's order.
///
/// Reads `dots` in XeDotSlot order and nothing else; `eq_tol` and `min_gram`
/// are PARAMETERS rather than constants of this header on purpose -- Driver.h
/// owns XE_EQUILIBRIUM_TOLERANCE and XE_ANDERSON_MIN_GRAM, and a second
/// declaration of either is a second opinion waiting to drift.
///
/// Writes `ctl` completely: on every exit path every field it owns is assigned,
/// because the commit kernel reads them whatever happened.
RASBERY_XE_HD inline void xeAndersonSolveControl(const double* dots, int ncol,
                                                 double picard, double eq_tol,
                                                 double min_gram,
                                                 unsigned long long forms,
                                                 XeTxnControl* ctl) {
    ctl->gamma[0]    = 0.0;
    ctl->gamma[1]    = 0.0;
    ctl->picard      = picard;
    ctl->step        = 0.0;
    ctl->proj        = 0.0;
    ctl->gg          = dots[XE_DOT_GG];
    ctl->solved      = 0;
    ctl->accept      = 0;
    ctl->proposed    = 0;
    ctl->ncol        = ncol;
    ctl->physics_bad = 0;

    // 3. Arming -- NOT a rejection, so neither counter moves.
    if (ncol == 0 || picard < eq_tol) {
        ctl->reason = XE_TXN_NOT_ARMED;
        return;
    }
    ctl->proposed = 1;

    const double gg     = dots[XE_DOT_GG];
    double       proj   = 0.0;
    double       gamma0 = 0.0, gamma1 = 0.0;
    const bool   solved =
        xeAndersonFit(dots, ncol, min_gram, forms, &gamma0, &gamma1, &proj);
    if (!solved) {
        ctl->reason = XE_TXN_CONDITION;
        return;
    }

    // SAFEGUARD 2/4: the fit must predict a residual DECREASE.
    const double pred2 = gg - proj;
    if (!(xeFinite(pred2) && pred2 >= 0.0 && pred2 < gg)) {
        ctl->proj   = proj;
        ctl->reason = XE_TXN_RESIDUAL;
        return;
    }

    ctl->gamma[0] = gamma0;
    ctl->gamma[1] = gamma1;
    ctl->proj     = proj;
    ctl->solved   = 1;
    ctl->reason   = XE_TXN_ACCEPTED; // provisional; the gate below can overturn it
}

/// SAFEGUARDS 3/4 AND 4/4, after the candidate kernel's two reductions.  Split
/// from the solve for the one reason a split is ever justified here: the
/// candidate's grid-wide OR and MAX are not readable until its grid retires,
/// and a kernel boundary is the only barrier that says so.
RASBERY_XE_HD inline void xeAndersonGateControl(int physics_bad, double step,
                                                double max_step, XeTxnControl* ctl) {
    ctl->physics_bad = physics_bad;
    ctl->step        = step;
    if (!ctl->solved) return; // the reason is already recorded
    if (physics_bad) {
        ctl->accept = 0;
        ctl->reason = XE_TXN_PHYSICS;
        return;
    }
    // SAFEGUARD 4/4: written as !(<=) so a NaN rejects.
    if (!(step <= max_step * ctl->picard)) {
        ctl->accept = 0;
        ctl->reason = XE_TXN_STEP;
        return;
    }
    ctl->accept = 1;
    ctl->reason = XE_TXN_ACCEPTED;
}

/// THE FUSED HISTORY MAINTENANCE (WP7-C step b).  One thread per fuel ordinal
/// performs, IN THIS ORDER, what fourteen device-to-device copies and two kXeSub
/// launches performed before:
///
///     rotate   df[0] <- df[1], dg[0] <- dg[1]        (only at a full window)
///     record   df[col] <- F - F_prev, dg[col] <- G - G_prev
///     save     F_prev <- F, G_prev <- G
///
/// ORDER-PRESERVATION NOTE.  Every one of these is ELEMENTWISE at a single
/// ordinal: no reduction, no neighbour, no accumulation.  Ordinal k's outputs
/// are a function of ordinal k's inputs alone, so no thread can observe another
/// thread's write and the grid may retire in any order.  What the fusion HAS to
/// preserve is the sequence WITHIN one ordinal -- the record must read F_prev
/// before the save overwrites it, and the rotate must read df[1] before the
/// record overwrites it -- and it does, by reading all six operands into
/// registers first.  No arithmetic changes: the two subtractions are
/// xeSubOrdinal's own, unrounded and unfused, and the copies are copies.
RASBERY_XE_HD inline void xeHistoryOrdinal(const double* f, const double* f_prev,
                                           const double* g, const double* g_prev,
                                           double* df0, double* df1, double* dg0,
                                           double* dg1, double* f_prev_out,
                                           double* g_prev_out, int k, int col,
                                           int rotate) {
    const double fv = f[k], fp = f_prev[k], gv = g[k], gp = g_prev[k];
    if (rotate) {
        df0[k] = df1[k];
        dg0[k] = dg1[k];
    }
    double* dfc = (col == 0) ? df0 : df1;
    double* dgc = (col == 0) ? dg0 : dg1;
    if (col >= 0) {
        dfc[k] = fv - fp;
        dgc[k] = gv - gp;
    }
    f_prev_out[k] = fv;
    g_prev_out[k] = gv;
}


} // namespace rasbery::xe
