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
constexpr int XE_BIT_COUNT     = 5;

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
constexpr unsigned long long XE_FORMS_DEFAULT = 0xdull;

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

} // namespace rasbery::xe
