#pragma once

// WP5 stage B -- the CTA-per-node arm of the flat-XS node update.
//
// DEVICE ONLY.  This header uses __shared__ and __syncthreads(), so it is
// included from .cu translation units and nothing else.  Those TUs build with
// --fmad=false, exactly like the thread-per-node arm, and the contraction
// policy object is literally the SAME type (fxs::StaticForms over the SAME
// FLATXS_FORMS mask) -- there is one definition of every fused/unfused choice
// in the tree and both arms call it.
//
// -------------------------------------------------------------------------
// WHY THIS ARM EXISTS
// -------------------------------------------------------------------------
// kernelFlatXs runs ONE THREAD PER NODE, and flatxsSolveNode's workspace is a
// per-thread local array set:
//
//     bl   [N_ACTIVE*NG  =   18 doubles]      144 B
//     bls  [NLSM         =    4 doubles]       32 B
//     bm   [N_ACTIVE*NMIC=  702 doubles]    5,616 B
//     bms  [NMSM         =  156 doubles]    1,248 B
//     iden [NISO         =   39 doubles]      312 B
//     active_xt[9] ints                        36 B
//                                          --------
//                                          7,388 B / thread   (~7.2 KiB)
//
// No GPU has 7 KiB of registers per thread, so ptxas puts that in LOCAL
// memory -- per-thread global-backed storage.  Stage A of WP5 measures how
// much of it actually spills and what the achieved occupancy is
// (tools/flatxs_resource_report.py); this kernel is the answer IF stage A
// shows the local traffic the static count predicts.
//
// The fix is to make the workspace a per-CTA object instead of a per-thread
// one: 919 doubles = 7,352 B of __shared__ per BLOCK rather than per thread.
// At 128 threads/CTA that is 57 B/thread of workspace instead of 7,388.
//
// -------------------------------------------------------------------------
// WHY IT IS BIT-IDENTICAL (the B0 argument -- read this before touching it)
// -------------------------------------------------------------------------
// The claim is NOT "the same math in a different order rounds the same".  It
// is the stronger and checkable claim: **every floating-point operation of
// flatxsSolveNode is executed here, on the same operands, in the same
// per-value order.  Only the identity of the thread executing it changes.**
// That holds because of exactly two structural properties, and both of them
// are asserted by tools/test_flatxs_cta_contract.py:
//
// (P1) FIXED LANE OWNERSHIP OVER AN ELEMENT ORDINAL SPACE.
//      Every workspace element has a flat ordinal q, and every phase of this
//      kernel walks its ordinal space with the SAME grid-stride form
//
//          for (int q = tid; q < N; q += T)
//
//      so the thread that gathers element q is the thread that applies every
//      delta to element q and the thread that scatters element q.  Element
//      q's accumulation chain therefore lives entirely inside one thread and
//      is never split, reassociated, or re-ordered.  The reference's chain
//      for element q is
//
//          bx[q] = ref[q];
//          for s in stream order:  bx[q] = ma(ACC, scale_s, horner_s(q), bx[q])
//
//      and that is exactly the sequence this kernel executes for q.  Changing
//      T (the block size) changes WHICH thread owns q and changes NOTHING
//      about q's chain -- which is why the block size is a tunable and not a
//      numerics knob.
//
// (P2) THE ISOTOPE FOLD IS NEVER PARALLELISED.
//      The macro-XS rebuild is
//
//          val = bl[t*NG+ig];
//          for (iso = 0; iso < NISO; ++iso)          // ASCENDING, SEQUENTIAL
//              val = ma(F_MACRO_SCAL, mt[iso*NG+ig], iden[iso], val);
//
//      a 39-long dependent FMA chain.  fma(a,b,c) is a SINGLE-ROUNDING
//      operation, so this chain is not associative in any sense -- a tree
//      reduction over iso does not merely re-round, it computes different
//      products (it would have to sum a*b terms that the reference never
//      materialises at all).  So there is NO parallel reduction here and no
//      atomicAdd.  Instead the PARALLELISM IS OVER THE CHAINS: there are
//      N_ACTIVE*NG = 18 independent scalar chains and NLSM = 4 independent
//      scatter chains, each one owned whole by a single lane, each one folded
//      left-to-right iso = 0..NISO-1.  No lane ever folds part of another
//      lane's chain, so no "fixed-order tree" is needed -- and none is used.
//      Same rule as XsReconKernel.h's determinism contract, same reason.
//
//      The XSRF pass (rf += xs_ssm[...] over ige) is likewise a per-ig
//      sequential chain owned by one lane; NG = 2, it is not worth splitting
//      and splitting it would be wrong for the same reason.
//
// (P3) UNIFORM CONTROL FLOW IS RECOMPUTED, NOT BROADCAST.
//      did/x/scale, the DeltaMeta, the mode-1 interval search, xloc and base
//      are computed REDUNDANTLY BY EVERY LANE from the same global bytes.
//      They are integer work plus one exact subtraction (xloc = x - knot), so
//      every lane gets bit-identical values; broadcasting them through shared
//      memory would need a __syncthreads() per stream entry and buy nothing.
//      This is the one place this kernel deliberately departs from the WP5
//      plan's pseudo-code, which syncs once per delta: with (P1) there is no
//      cross-lane dependency inside the stream loop at all, so the loop runs
//      with ZERO barriers regardless of how long the stream is.
//
// The only barriers in the kernel are the three that publish one phase's
// writes to lanes that did not perform them:
//   * after the light-isotope refresh (all lanes read sh_iden),
//   * after the workspace is final          (all lanes read sh_bl/sh_bm/...),
//   * after the macro pass                  (the XSDF/XSRF pass re-reads
//     xs[XSTF]/xs[XSAF]/xs_ssm from GLOBAL, exactly like the reference does,
//     and __syncthreads() is what makes another lane's global store visible).
//
// FMA NOTE.  Nothing in this file writes `a * b + c`.  Every multiply-add
// goes through pol.ma(<bit>, ...) with the same FormBit the reference uses at
// that site, and the TU is compiled --fmad=false so nvcc cannot fuse the
// unfused arms on its own.  If a site here were spelled with a different bit
// than the reference spells it, the replay gate
// (test/flatxs_device_replay.cu --cta) fails loudly instead of drifting.
//
// -------------------------------------------------------------------------
// WHAT IS *NOT* CLAIMED
// -------------------------------------------------------------------------
// Nothing about speed.  Stage A decides whether local traffic is the real
// bottleneck and the 238 runbook in docs/WP5_FLATXS_CTA_20260831_KO.md
// decides whether this arm is adopted.  The flag defaults OFF.

#include "FlatXsKernel.h"

#include <cuda_runtime.h>

#if !defined(__CUDACC__)
    #error "FlatXsCtaKernel.cuh is device-only; include it from a .cu TU"
#endif

namespace rasbery::flatxs {

/// Shared workspace: the exact same five arrays flatxsSolveNode declares as
/// thread locals, sized from the SAME constants (N_ACTIVE/NG/NLSM/NMIC/NMSM/
/// NISO from FlatXsKernel.h and XsReconKernel.h).  Nothing here is a
/// hand-copied magic number -- if the isotope registry or the group count
/// moves, this struct moves with it, and the contract test refuses any
/// literal size.
struct CtaWorkspace {
    double bl[N_ACTIVE * NG];   ///< lmpx scalars   [t*NG + ig]
    double bls[NLSM];           ///< lmpx scatter   [igs*NG + ige]
    double bm[N_ACTIVE * NMIC]; ///< micx scalars   [t*NMIC + iso*NG + ig]
    double bms[NMSM];           ///< micx scatter   [iso*NG*NG + igs*NG + ige]
    double iden[NISO];          ///< node densities after RefreshLightIsotopes
};

/// WP20's NARROWED TWIN of the workspace above (RASBERY_GPU_FP32, default OFF).
///
/// Same five arrays, sized from the SAME expressions -- the contract test
/// asserts the two structs field-for-field, so the isotope registry cannot move
/// under one of them and not the other -- and half the bytes: **3,676 B/CTA
/// against 7,352**.
///
/// WHAT THIS BUYS, precisely.  WP20 landed it as a SHARED-MEMORY OCCUPANCY
/// saving and said out loud that it saved no transfer, because the node's
/// inputs still arrived as doubles in global memory and its outputs still left
/// as doubles.  **WP20.1 removed that caveat for the four micx/lmpx blocks**:
/// `FlatXsView::narrow_blocks` makes `ref_lmp/ref_lsm/ref_mic/ref_msm` and
/// `lmp/lsm/mic/msm` float on the same arm, so the gather below reads half the
/// bytes and the scatter writes half.  `v.xs` / `v.xs_ssm` / `v.iden` are
/// still double and still the FP64 authority -- they are the macroscopic
/// answer the nodal drive, the CMFD operator and the host read.  What follows
/// is therefore the occupancy half of the story; the bandwidth half is in
/// src/FlatXsKernel.h's accessor note.  The arithmetic is worth doing HONESTLY
/// because the tempting version of it is wrong.  At 7,352 B/CTA a 100 KiB
/// shared budget seats floor(102400/7352) = 13 blocks per SM; at 3,676 it seats
/// 27.  It does NOT get 27: at T=128 that would be 108 warps and the SM caps at
/// 64 (2,048 threads), so the resident count goes 13 -> 16 and shared stops
/// being the binding constraint at all.  **+23 % blocks, not +108 %.**  The
/// second saving is the one with no such ceiling: the shared BANK TRAFFIC
/// itself halves, and the Horner accumulation reads and writes `w.bm` once per
/// delta entry per element, which is the innermost loop of the whole kernel.
/// Anybody quoting a 2x from this comment has read the first sentence of the
/// arithmetic and not the second.
///
/// WHY IT IS A2 AND NOT B0, said out loud because the FP64 twin above carries a
/// bit-identity argument and a reader will arrive here expecting one.  Every
/// floating-point operation is still executed on the same operands in the same
/// per-value order -- structural properties (P1)/(P3) are untouched, and the
/// same contract test still holds them.  What changes is the ROUNDING of the
/// accumulator between operations: `dst[e] = scale*val + dst[e]` now rounds to
/// float after each delta entry instead of to double.  A node's delta stream is
/// short (`node_cnt`, single digits on the KNGR deck), so the accumulated
/// rounding is a few ULPs of float on a cross-section -- but a few ULPs of
/// float is ~1e-7 relative, and it enters the trajectory through the macro XS,
/// which is why this arm is gated and measured rather than adopted.
struct CtaWorkspaceF32 {
    float bl[N_ACTIVE * NG];   ///< lmpx scalars   [t*NG + ig]
    float bls[NLSM];           ///< lmpx scatter   [igs*NG + ige]
    float bm[N_ACTIVE * NMIC]; ///< micx scalars   [t*NMIC + iso*NG + ig]
    float bms[NMSM];           ///< micx scatter   [iso*NG*NG + igs*NG + ige]
    float iden[NISO];          ///< node densities after RefreshLightIsotopes
};

/// Ordinal-space extents.  These name the loop bounds used by (P1) so the
/// gather / apply / scatter phases cannot drift apart.
constexpr int Q_LMP = N_ACTIVE * NG;   // 18
constexpr int Q_LSM = NLSM;            // 4
constexpr int Q_MIC = N_ACTIVE * NMIC; // 702
constexpr int Q_MSM = NMSM;            // 156

/// One CTA, one unrodded node.  `T` is blockDim.x as a compile-time constant
/// so the strides fold; it is a PERFORMANCE parameter only (see P1).
/// WP20 made the workspace a TEMPLATE PARAMETER rather than editing `double` to
/// `float` in the body: the FP64 and FP32 arms are then compiled from ONE text,
/// so the narrow arm cannot drift away from the reference under maintenance,
/// and every structural property the contract test checks (the lane-owned
/// ordinal loops, the FormBit census, the barrier placement) is checked once and
/// holds for both.  Every operand crossing the workspace boundary converts
/// implicitly -- double in, float stored; float out, double arithmetic -- which
/// is exactly the "narrow the STATE, keep the OPERATIONS" split the arm claims.
template <int T, class POL, class WS>
__device__ inline void flatxsSolveNodeCta(const FlatXsView& v, int i,
                                          const POL& pol, WS& w) {
    const int nxyz = v.nxyz;
    const int l    = v.nodes[i];
    const int tid  = static_cast<int>(threadIdx.x);

    // --- 1. Gather the reference state (plain copies, no arithmetic) -------
    // Same ordinal mapping as every later phase: q -> lane q % T.
    for (int q = tid; q < Q_LMP; q += T) {
        const int t  = q / NG;
        const int ig = q - t * NG;
        w.bl[q]      = fxsRefLmp(v, t, block_layout::lmp(nxyz, l, ig));
    }
    for (int q = tid; q < Q_LSM; q += T)
        w.bls[q] = fxsRefLsm(v, block_layout::lsm(nxyz, l, q));
    for (int q = tid; q < Q_MIC; q += T) {
        const int t = q / NMIC;
        const int e = q - t * NMIC;
        w.bm[q]     = fxsRefMic(v, t, block_layout::mic(nxyz, l, e));
    }
    for (int q = tid; q < Q_MSM; q += T)
        w.bms[q] = fxsRefMsm(v, block_layout::msm(nxyz, l, q));

    // NO BARRIER HERE, AND THAT IS DELIBERATE: by (P1) the lane that wrote
    // element q is the lane that reads it below.  A barrier would be free
    // insurance against a bug that (P1) makes impossible and that the contract
    // test checks for.

    // --- 2. Apply the resolved delta stream in the caller's order ----------
    const int s0 = v.node_off[i];
    const int s1 = s0 + v.node_cnt[i];
    for (int s = s0; s < s1; ++s) {
        const int    did   = v.stream_did[s];
        const double x     = v.stream_x[s];
        const double scale = v.stream_scale[s];
        // Same defensive guard as the reference; uniform across the block, so
        // it costs no divergence.
        if (did < 0 || scale == 0.0) continue;

        // (P3): every lane recomputes this, nobody broadcasts it.
        const DeltaMeta dm   = v.deltas[did];
        int             base = dm.coeff_base;
        int             nord = dm.nord;
        double          xloc = x;
        if (dm.mode == 1) {
            const int nintervals = dm.nord / dm.ncoeff;
            int       interval   = nintervals - 1;
            for (int k = 0; k < nintervals - 1; ++k) {
                if (x < v.knots[dm.knot_offset + k + 1]) {
                    interval = k;
                    break;
                }
            }
            xloc = x - v.knots[dm.knot_offset + interval];
            base += interval * dm.ncoeff;
            nord = dm.ncoeff;
        }

        // lmpx scalars.  Reference: for t, for e -> Horner down p, then
        // dst[e] = ma(F_ACC_LMP, scale, val, dst[e]).  Here the (t,e) pair is
        // the ordinal q and the chain is identical.
        for (int q = tid; q < Q_LMP; q += T) {
            const int     t     = q / NG;
            const int     e     = q - t * NG;
            const double* cdata = v.coeff_lmp[t];
            double        val   = cdata[(base + nord - 1) * NG + e];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_LMP, val, xloc, cdata[(base + p) * NG + e]);
            w.bl[q] = pol.ma(F_ACC_LMP, scale, val, w.bl[q]);
        }
        // lmpx scatter.
        for (int q = tid; q < Q_LSM; q += T) {
            double val = v.coeff_lsm[(base + nord - 1) * NLSM + q];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_LSM, val, xloc,
                             v.coeff_lsm[(base + p) * NLSM + q]);
            w.bls[q] = pol.ma(F_ACC_LSM, scale, val, w.bls[q]);
        }

        if (!v.has_coeff_micx) continue;
        // micx scalars.
        for (int q = tid; q < Q_MIC; q += T) {
            const int     t     = q / NMIC;
            const int     e     = q - t * NMIC;
            const double* cdata = v.coeff_mic[t];
            double        val   = cdata[(base + nord - 1) * NMIC + e];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_MIC, val, xloc, cdata[(base + p) * NMIC + e]);
            w.bm[q] = pol.ma(F_ACC_MIC, scale, val, w.bm[q]);
        }
        // micx scatter.
        for (int q = tid; q < Q_MSM; q += T) {
            double val = v.coeff_msm[(base + nord - 1) * NMSM + q];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_MSM, val, xloc,
                             v.coeff_msm[(base + p) * NMSM + q]);
            w.bms[q] = pol.ma(F_ACC_MSM, scale, val, w.bms[q]);
        }
    }

    // --- 3. Densities + RefreshLightIsotopes ------------------------------
    // Rows IH1/IB10/IO16 (0/1/2, contiguous by registry design) are the ONLY
    // rows the refresh rewrites, so lanes load rows 3.. from memory and lane 0
    // computes rows 0..2.  Splitting it this way means no lane ever reads a
    // row another lane is about to overwrite, so one barrier suffices.
    for (int iso = tid; iso < NISO; iso += T)
        if (iso > IO16) w.iden[iso] = v.iden[iso * nxyz + l];
    if (tid == 0) {
        const double nH2O       = v.dmod[l] * v.wvfr[l] * WATER_NUMBER_DENSITY;
        const double boron_dmod = fxsBoronDmod(v, l);
        w.iden[IH1]             = 2.0 * nH2O;
        w.iden[IO16]            = nH2O;
        w.iden[IB10] = boron_dmod * v.wvfr[l] * v.bppm[l] * BORON_DENSITY_FACTOR;
        v.iden[IH1 * nxyz + l]  = w.iden[IH1];
        v.iden[IB10 * nxyz + l] = w.iden[IB10];
        v.iden[IO16 * nxyz + l] = w.iden[IO16];
    }

    // --- 4. Scatter the workspace back to the SoA arrays (plain copies) ----
    // Still lane-owned (P1), so this needs no barrier of its own.
    for (int q = tid; q < Q_LMP; q += T) {
        const int t             = q / NG;
        const int ig            = q - t * NG;
        fxsStoreLmp(v, t, block_layout::lmp(nxyz, l, ig), w.bl[q]);
    }
    for (int q = tid; q < Q_LSM; q += T)
        fxsStoreLsm(v, block_layout::lsm(nxyz, l, q), w.bls[q]);
    for (int q = tid; q < Q_MIC; q += T) {
        const int t            = q / NMIC;
        const int e            = q - t * NMIC;
        fxsStoreMic(v, t, block_layout::mic(nxyz, l, e), w.bm[q]);
    }
    for (int q = tid; q < Q_MSM; q += T)
        fxsStoreMsm(v, block_layout::msm(nxyz, l, q), w.bms[q]);

    // Publish the workspace and sh_iden: from here on lanes read elements they
    // did not write.
    __syncthreads();

    // --- 5. Rebuild this node's macroscopic XS ----------------------------
    // (P2): one lane per output chain, isotope fold strictly ascending.
    const int active_xt[N_ACTIVE] = {
        xsrecon::T_XSTF, xsrecon::T_XSAF, xsrecon::T_XSFF,
        xsrecon::T_XSNF, xsrecon::T_XSKF, xsrecon::T_XSSF,
        xsrecon::T_FYLD, xsrecon::T_XS2N, xsrecon::T_XS3N};
    for (int q = tid; q < Q_LMP; q += T) {
        const int   t  = q / NG;
        const int   ig = q - t * NG;
        // `auto`, not `const double*`: this is the one place the body names the
        // workspace's ELEMENT type, and WP20 made that type a parameter.  The
        // arithmetic below is still double either way -- `mt[...]` promotes.
        const auto* mt = w.bm + t * NMIC;
        double      val = w.bl[q];
        for (int iso = 0; iso < NISO; ++iso)
            val = pol.ma(F_MACRO_SCAL, mt[iso * NG + ig], w.iden[iso], val);
        v.xs[active_xt[t]][ig * nxyz + l] = val;
    }
    for (int q = tid; q < Q_LSM; q += T) {
        // q == igs*NG + ige, and bms is [iso*NLSM + q] -- same expression the
        // reference writes as bms[iso*NLSM + igs*NG + ige].
        double val = w.bls[q];
        for (int iso = 0; iso < NISO; ++iso)
            val = pol.ma(F_MACRO_SSM, w.bms[iso * NLSM + q], w.iden[iso], val);
        v.xs_ssm[q * nxyz + l] = val;
    }

    // The XSDF/XSRF pass re-reads xs[XSTF]/xs[XSAF]/xs_ssm from GLOBAL, exactly
    // as the reference does, and those stores came from other lanes.
    __syncthreads();

    for (int ig = tid; ig < NG; ig += T) {
        const double tr = v.xs[xsrecon::T_XSTF][ig * nxyz + l];
        v.xs[xsrecon::T_XSDF][ig * nxyz + l] =
            (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;

        double rf = v.xs[xsrecon::T_XSAF][ig * nxyz + l];
        for (int ige = 0; ige < NG; ++ige)
            rf += v.xs_ssm[(ig * NG + ige) * nxyz + l];
        v.xs[xsrecon::T_XSRF][ig * nxyz + l] = rf;
    }
}

/// One CTA per node.  Static __shared__ (7,352 B/CTA): no dynamic shared-mem
/// opt-in is needed and the occupancy is legible from the launch bounds.
template <int T>
__global__ void __launch_bounds__(T) kernelFlatXsCta(FlatXsView v) {
    __shared__ CtaWorkspace w;
    const int i = static_cast<int>(blockIdx.x);
    // Block-uniform, so the whole CTA leaves together and no lane can be
    // stranded at a barrier the others already passed.
    if (i >= v.n_nodes) return;
    flatxsSolveNodeCta<T>(v, i, StaticForms{}, w);
}

/// WP20's narrow twin (RASBERY_GPU_FP32).  Static __shared__ (3,676 B/CTA).
///
/// TEXTUALLY IDENTICAL to kernelFlatXsCta except for the workspace type, and
/// that is the point: the body it calls is the same instantiated template, so
/// the two arms cannot diverge in anything but precision.  It is a SEPARATE
/// __global__ rather than a runtime branch because the shared allocation is a
/// compile-time property of the kernel -- a branch would have to reserve the
/// wide workspace and would save nothing at all.
template <int T>
__global__ void __launch_bounds__(T) kernelFlatXsCtaF32(FlatXsView v) {
    __shared__ CtaWorkspaceF32 w;
    const int i = static_cast<int>(blockIdx.x);
    if (i >= v.n_nodes) return;
    flatxsSolveNodeCta<T>(v, i, StaticForms{}, w);
}

// ===========================================================================
// WP21-B2 -- THE TILED ARM: SAME ARITHMETIC, TRANSPOSED GATHER AND SCATTER
// ===========================================================================
//
// THE OBSERVATION (238 ncu, block 39).  `kernelFlatXsCta<128>` is the only
// bandwidth-leaning kernel of the whole solve (dram 23 % of peak) and its
// stores measure **25.2 sectors per request** against an ideal of 2.  The
// cause is NOT the block layout: the block is already component-major /
// node-innermost (`block_layout::elem`, `c*nxyz + l`) and that order is the
// right one for every OTHER consumer -- kernelFlatXs, the CRAM D2D fill, the
// Xe commit, the host accessors -- all of which are parallel over `l`.
// docs/WP21_BC_FLATXS_NODAL_COALESCING_20260831_KO.md proves the permutation
// is exactly zero-sum and must not be done.
//
// The cause is the PARALLELISATION AXIS.  One CTA owns one node, so `l` is
// block-uniform and the 32 lanes of a warp hold 32 different component
// ordinals `q`.  Under `c*nxyz + l` consecutive lanes are `nxyz *
// elem_bytes` apart -- one sector each, and the scatter phase is 880 such
// stores per node with nothing to dilute them.
//
// THE FIX IS THE STANDARD SHARED-MEMORY TRANSPOSE, and it is available here
// for free because the workspace is ALREADY in shared memory.  One CTA takes a
// TILE of `TILE` consecutive entries of `v.nodes`, keeps one workspace per
// tile slot, and uses TWO lane mappings inside one kernel:
//
//   * the DELTA-STREAM phase keeps today's mapping -- lanes walk the ordinal
//     space of ONE node (`for (q = lane; q < Q_x; q += T)`, group `j` owning
//     lanes `[j*T, (j+1)*T)`).  That mapping is not an accident: the
//     coefficient reads `cdata[(base+p)*NMIC + e]` have `e == q` and are
//     STRIDE 1 across lanes.  They are the dominant load count of the kernel
//     and the reason this arm exists at all.  A node-innermost mapping here
//     would give every lane a different `base`, scatter the coefficient
//     stream, and buy the store back out of the load.
//
//   * the GATHER, SCATTER, DENSITY and MACRO phases use the NODE-INNERMOST
//     mapping `p -> (q = p / TILE, j = p % TILE)`, so consecutive lanes hold
//     consecutive `l` at a FIXED component ordinal and the address is
//     `c*nxyz + l0 + j` -- contiguous.
//
// WHY OCCUPANCY DOES NOT COLLAPSE (the objection this WP had to answer).  The
// tile multiplies the shared workspace by TILE, but it multiplies the THREAD
// COUNT by TILE as well: `blockDim.x = T * TILE`.  Shared bytes PER THREAD are
// therefore unchanged -- 7,352/128 = 57.4 B on the FP64 arm and 3,676/128 =
// 28.7 B on the FP32 one, exactly as before -- and shared memory constrains
// resident THREADS, not resident blocks.  What the tile costs is block
// granularity, not per-thread footprint.  The budget table is in
// docs/WP21_B2C2_COALESCING_20260831_KO.md.
//
// WHY IT IS STILL B0.  Structural properties (P1)/(P2)/(P3) are untouched:
//
//   (P1) every element ordinal of every tile slot is still owned WHOLE by one
//        lane through gather -> apply -> scatter.  The lane IDENTITY changes
//        between phases (that is the transpose), so the two phase boundaries
//        that cross the change carry a `__syncthreads()` -- publication, not
//        arithmetic.  A barrier moves no value.
//   (P2) both isotope folds are still `for (iso = 0; iso < NISO; ++iso)`,
//        ascending, sequential, one lane per output chain.
//   (P3) did/x/scale/xloc/base are still recomputed redundantly per lane; the
//        delta-stream loop still carries NO barrier, which is what keeps its
//        cost independent of the stream length even though the tile slots have
//        DIFFERENT stream lengths (each group of T lanes walks its own node
//        stream; with T >= 32 no warp is ever split across two slots).
//
// So the same double is read from the same address, multiplied by the same
// double, and written to the same address.  Only which thread does it moves.
//
// THE TAIL.  The grid is `n_nodes / TILE` full tiles; the remaining
// `n_nodes % TILE` nodes run on `kernelFlatXsCtaAt`, which is
// `kernelFlatXsCta` with a base offset -- the untiled path, kept live both as
// the tail handler and as the `RASBERY_GPU_FLATXS_CTA_TILE=1` A/B arm.
//
// WHY THIS IS A SECOND BODY AND NOT A TEMPLATE PARAMETER ON THE FIRST.
// `flatxsSolveNodeCta` above is the arm the 238 gate certified and the arm
// `RASBERY_GPU_FLATXS_CTA_TILE=1` still runs; generalising it would mean the
// reference and the candidate were the same text and an A/B could no longer
// bisect them.  The cost is duplicated arithmetic, and the cost is PAID BY THE
// CONTRACT TEST: tools/test_flatxs_cta_contract.py runs the same FormBit
// census, the same isotope-fold rule, the same bare-multiply-add rule and the
// same barrier rules over THIS body as it runs over that one, against the same
// reference `flatxsSolveNode`.  A site that drifts in one body and not the
// other fails the census before it reaches a GPU.
// ---------------------------------------------------------------------------

/// One CTA, `TILE` consecutive entries of `v.nodes`, `T` lanes per node.
template <int T, int TILE, class POL, class WS>
__device__ inline void flatxsSolveTileCta(const FlatXsView& v, int i0,
                                          const int* sh_l, const POL& pol,
                                          WS* w) {
    const int     nxyz = v.nxyz;
    const int     tid  = static_cast<int>(threadIdx.x); // 0 .. NT-1
    constexpr int NT   = T * TILE;
    const int     jown = tid / T;        // tile slot this lane group owns
    const int     lane = tid - jown * T; // lane inside that group

    // --- 1. Gather the reference state (plain copies, no arithmetic) -------
    // NODE-INNERMOST: consecutive lanes hold consecutive `l` at one ordinal.
    for (int p = tid; p < Q_LMP * TILE; p += NT) {
        const int q  = p / TILE;
        const int j  = p - q * TILE;
        const int t  = q / NG;
        const int ig = q - t * NG;
        w[j].bl[q]   = fxsRefLmp(v, t, block_layout::lmp(nxyz, sh_l[j], ig));
    }
    for (int p = tid; p < Q_LSM * TILE; p += NT) {
        const int q = p / TILE;
        const int j = p - q * TILE;
        w[j].bls[q] = fxsRefLsm(v, block_layout::lsm(nxyz, sh_l[j], q));
    }
    for (int p = tid; p < Q_MIC * TILE; p += NT) {
        const int q = p / TILE;
        const int j = p - q * TILE;
        const int t = q / NMIC;
        const int e = q - t * NMIC;
        w[j].bm[q]  = fxsRefMic(v, t, block_layout::mic(nxyz, sh_l[j], e));
    }
    for (int p = tid; p < Q_MSM * TILE; p += NT) {
        const int q = p / TILE;
        const int j = p - q * TILE;
        w[j].bms[q] = fxsRefMsm(v, block_layout::msm(nxyz, sh_l[j], q));
    }

    // Publish the gather: the transpose means the lane that WROTE ordinal q of
    // slot j is not the lane that APPLIES deltas to it.  This barrier is the
    // whole price of the transpose and it is paid ONCE per tile, not once per
    // delta -- (P3) is about the stream loop below and is untouched.
    __syncthreads();

    // --- 2. Apply the resolved delta stream in the caller order ------------
    // LANE-OWNED, one node per group of T lanes: the phase of
    // flatxsSolveNodeCta with `tid` read as `lane` and `w` as `w[jown]`.  The
    // coefficient reads stay stride-1 across lanes because `q` still varies
    // with the lane and `base` is still uniform inside the group.
    const int i  = i0 + jown;
    const int s0 = v.node_off[i];
    const int s1 = s0 + v.node_cnt[i];
    for (int s = s0; s < s1; ++s) {
        const int    did   = v.stream_did[s];
        const double x     = v.stream_x[s];
        const double scale = v.stream_scale[s];
        if (did < 0 || scale == 0.0) continue;

        // (P3): every lane recomputes this, nobody broadcasts it.
        const DeltaMeta dm   = v.deltas[did];
        int             base = dm.coeff_base;
        int             nord = dm.nord;
        double          xloc = x;
        if (dm.mode == 1) {
            const int nintervals = dm.nord / dm.ncoeff;
            int       interval   = nintervals - 1;
            for (int k = 0; k < nintervals - 1; ++k) {
                if (x < v.knots[dm.knot_offset + k + 1]) {
                    interval = k;
                    break;
                }
            }
            xloc = x - v.knots[dm.knot_offset + interval];
            base += interval * dm.ncoeff;
            nord = dm.ncoeff;
        }

        for (int q = lane; q < Q_LMP; q += T) {
            const int     t     = q / NG;
            const int     e     = q - t * NG;
            const double* cdata = v.coeff_lmp[t];
            double        val   = cdata[(base + nord - 1) * NG + e];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_LMP, val, xloc, cdata[(base + p) * NG + e]);
            w[jown].bl[q] = pol.ma(F_ACC_LMP, scale, val, w[jown].bl[q]);
        }
        for (int q = lane; q < Q_LSM; q += T) {
            double val = v.coeff_lsm[(base + nord - 1) * NLSM + q];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_LSM, val, xloc,
                             v.coeff_lsm[(base + p) * NLSM + q]);
            w[jown].bls[q] = pol.ma(F_ACC_LSM, scale, val, w[jown].bls[q]);
        }

        if (!v.has_coeff_micx) continue;
        for (int q = lane; q < Q_MIC; q += T) {
            const int     t     = q / NMIC;
            const int     e     = q - t * NMIC;
            const double* cdata = v.coeff_mic[t];
            double        val   = cdata[(base + nord - 1) * NMIC + e];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_MIC, val, xloc, cdata[(base + p) * NMIC + e]);
            w[jown].bm[q] = pol.ma(F_ACC_MIC, scale, val, w[jown].bm[q]);
        }
        for (int q = lane; q < Q_MSM; q += T) {
            double val = v.coeff_msm[(base + nord - 1) * NMSM + q];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_MSM, val, xloc,
                             v.coeff_msm[(base + p) * NMSM + q]);
            w[jown].bms[q] = pol.ma(F_ACC_MSM, scale, val, w[jown].bms[q]);
        }
    }

    // --- 3. Densities + RefreshLightIsotopes ------------------------------
    // Same split as the untiled body -- rows 0..2 belong to ONE lane per tile
    // slot, rows 3.. are strided -- so no lane reads a row another lane is
    // about to overwrite and no barrier is owed here either.  Both halves are
    // node-innermost now: `v.iden[iso*nxyz + l]` and `v.dmod/wvfr/bppm[l]` are
    // read by consecutive lanes at consecutive `l`.
    for (int p = tid; p < NISO * TILE; p += NT) {
        const int iso = p / TILE;
        const int j   = p - iso * TILE;
        const int l   = sh_l[j];
        if (iso > IO16) w[j].iden[iso] = v.iden[iso * nxyz + l];
    }
    if (tid < TILE) {
        const int    j          = tid;
        const int    l          = sh_l[j];
        const double nH2O       = v.dmod[l] * v.wvfr[l] * WATER_NUMBER_DENSITY;
        const double boron_dmod = fxsBoronDmod(v, l);
        w[j].iden[IH1]          = 2.0 * nH2O;
        w[j].iden[IO16]         = nH2O;
        w[j].iden[IB10] = boron_dmod * v.wvfr[l] * v.bppm[l] * BORON_DENSITY_FACTOR;
        v.iden[IH1 * nxyz + l]  = w[j].iden[IH1];
        v.iden[IB10 * nxyz + l] = w[j].iden[IB10];
        v.iden[IO16 * nxyz + l] = w[j].iden[IO16];
    }

    // Publish the workspace and sh_iden: the scatter and the macro rebuild
    // below both read slots this lane did not compute.
    __syncthreads();

    // --- 4. Scatter the workspace back to the SoA arrays (plain copies) ----
    // THIS IS THE 25.2 -> ~3 STORE.  Node-innermost: for a fixed component the
    // TILE lanes write `c*nxyz + l0 .. c*nxyz + l0 + TILE-1`.
    for (int p = tid; p < Q_LMP * TILE; p += NT) {
        const int q  = p / TILE;
        const int j  = p - q * TILE;
        const int t  = q / NG;
        const int ig = q - t * NG;
        fxsStoreLmp(v, t, block_layout::lmp(nxyz, sh_l[j], ig), w[j].bl[q]);
    }
    for (int p = tid; p < Q_LSM * TILE; p += NT) {
        const int q = p / TILE;
        const int j = p - q * TILE;
        fxsStoreLsm(v, block_layout::lsm(nxyz, sh_l[j], q), w[j].bls[q]);
    }
    for (int p = tid; p < Q_MIC * TILE; p += NT) {
        const int q = p / TILE;
        const int j = p - q * TILE;
        const int t = q / NMIC;
        const int e = q - t * NMIC;
        fxsStoreMic(v, t, block_layout::mic(nxyz, sh_l[j], e), w[j].bm[q]);
    }
    for (int p = tid; p < Q_MSM * TILE; p += NT) {
        const int q = p / TILE;
        const int j = p - q * TILE;
        fxsStoreMsm(v, block_layout::msm(nxyz, sh_l[j], q), w[j].bms[q]);
    }

    // --- 5. Rebuild this tile macroscopic XS ------------------------------
    // (P2): one lane per output chain, isotope fold strictly ascending.  The
    // chain is identical to the untiled body; only the lane holding it and the
    // store neighbours change.
    const int active_xt[N_ACTIVE] = {
        xsrecon::T_XSTF, xsrecon::T_XSAF, xsrecon::T_XSFF,
        xsrecon::T_XSNF, xsrecon::T_XSKF, xsrecon::T_XSSF,
        xsrecon::T_FYLD, xsrecon::T_XS2N, xsrecon::T_XS3N};
    for (int p = tid; p < Q_LMP * TILE; p += NT) {
        const int   q   = p / TILE;
        const int   j   = p - q * TILE;
        const int   l   = sh_l[j];
        const int   t   = q / NG;
        const int   ig  = q - t * NG;
        const auto* mt  = w[j].bm + t * NMIC;
        double      val = w[j].bl[q];
        for (int iso = 0; iso < NISO; ++iso)
            val = pol.ma(F_MACRO_SCAL, mt[iso * NG + ig], w[j].iden[iso], val);
        v.xs[active_xt[t]][ig * nxyz + l] = val;
    }
    for (int p = tid; p < Q_LSM * TILE; p += NT) {
        const int q   = p / TILE;
        const int j   = p - q * TILE;
        const int l   = sh_l[j];
        double    val = w[j].bls[q];
        for (int iso = 0; iso < NISO; ++iso)
            val = pol.ma(F_MACRO_SSM, w[j].bms[iso * NLSM + q], w[j].iden[iso], val);
        v.xs_ssm[q * nxyz + l] = val;
    }

    // The XSDF/XSRF pass re-reads xs[XSTF]/xs[XSAF]/xs_ssm from GLOBAL, exactly
    // as the reference does, and those stores came from other lanes.
    __syncthreads();

    for (int p = tid; p < NG * TILE; p += NT) {
        const int    ig = p / TILE;
        const int    j  = p - ig * TILE;
        const int    l  = sh_l[j];
        const double tr = v.xs[xsrecon::T_XSTF][ig * nxyz + l];
        v.xs[xsrecon::T_XSDF][ig * nxyz + l] =
            (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;

        double rf = v.xs[xsrecon::T_XSAF][ig * nxyz + l];
        for (int ige = 0; ige < NG; ++ige)
            rf += v.xs_ssm[(ig * NG + ige) * nxyz + l];
        v.xs[xsrecon::T_XSRF][ig * nxyz + l] = rf;
    }
}

/// The static-__shared__ ceiling every architecture in
/// RASBERY_CUDA_ARCHITECTURES (80/86/89/100/120) honours WITHOUT a
/// cudaFuncSetAttribute opt-in.  Those parts allow more DYNAMIC shared memory
/// per block, but taking it would make this launch helper stateful (an
/// attribute call per kernel per device) and the tile that fits inside 48 KiB
/// already reaches the ideal sector count.
constexpr int CTA_SMEM_STATIC_MAX = 48 * 1024;

/// The hardware cap on one CUDA block.
constexpr int CTA_MAX_THREADS = 1024;

/// Is `<T, TILE>` a legal instantiation for workspace `WS`?  BOTH bounds are
/// compile-time errors when violated (`__launch_bounds__` above 1024,
/// `__shared__` above the static ceiling), so every dispatch site guards with
/// `if constexpr` on this predicate rather than trusting a runtime clamp.
///
/// On the KNGR deck this evaluates to: FP64 (7,352 B/slot) TILE <= 6, FP32
/// (3,676 B/slot) TILE <= 13 -- and the ladder below rounds those down to the
/// powers of two 4 and 8, which is where both arms land on 29,408 B/CTA.
template <int T, int TILE, class WS>
constexpr bool ctaTileFits() {
    return TILE >= 1 && T * TILE <= CTA_MAX_THREADS
           && TILE * static_cast<int>(sizeof(WS)) <= CTA_SMEM_STATIC_MAX;
}

/// One CTA, `TILE` consecutive entries of `v.nodes`, `T` lanes per node.
/// `WS` is the workspace type -- WP20 two arms compile from ONE text here for
/// the same reason `flatxsSolveNodeCta` takes it as a parameter.
template <int T, int TILE, class WS>
__global__ void __launch_bounds__(T * TILE) kernelFlatXsCtaTile(FlatXsView v) {
    __shared__ WS  w[TILE];
    __shared__ int sh_l[TILE];
    const int      i0 = static_cast<int>(blockIdx.x) * TILE;
    // Block-uniform: the launcher only ever issues FULL tiles (the remainder
    // goes to kernelFlatXsCtaAt), so this is a guard, not a tail path.
    if (i0 + TILE > v.n_nodes) return;
    if (static_cast<int>(threadIdx.x) < TILE)
        sh_l[threadIdx.x] = v.nodes[i0 + static_cast<int>(threadIdx.x)];
    __syncthreads();
    flatxsSolveTileCta<T, TILE>(v, i0, sh_l, StaticForms{}, w);
}

/// `kernelFlatXsCta` with a base offset: the tail of a tiled launch, and the
/// only new text is `node_base +`.  It calls the SAME `flatxsSolveNodeCta` the
/// untiled arm calls, so the tail nodes are computed by the certified body.
template <int T, class WS>
__global__ void __launch_bounds__(T) kernelFlatXsCtaAt(FlatXsView v, int node_base) {
    __shared__ WS w;
    const int     i = node_base + static_cast<int>(blockIdx.x);
    if (i >= v.n_nodes) return;
    flatxsSolveNodeCta<T>(v, i, StaticForms{}, w);
}

/// Block sizes the arm accepts.  The list is a PERFORMANCE ladder, not a
/// numerics one (P1): every entry produces the same bytes.  Anything else the
/// caller asks for is clamped to CTA_THREADS_DEFAULT.
constexpr int CTA_THREADS_DEFAULT = 128;

/// WP21-B2 tile ladder.  `1` is the untiled arm above -- the certified,
/// bisectable reference; `CTA_TILE_DEFAULT` / `CTA_TILE_DEFAULT_F32` are what
/// the backend asks for when RASBERY_GPU_FLATXS_CTA_TILE is unset.
///
/// THE TWO DEFAULTS ARE THE SAME SHARED BUDGET, not two different bets:
/// 4 x 7,352 = 8 x 3,676 = 29,408 B/CTA.  They differ because the FP64 slot is
/// twice the FP32 one, and they are the largest powers of two that fit the
/// 48 KiB static ceiling (FP64 admits 6, FP32 admits 13).  Under both, shared
/// bytes per THREAD are exactly what the untiled arm spends, because
/// blockDim.x scales with the tile.
constexpr int CTA_TILE_DEFAULT     = 4;
constexpr int CTA_TILE_DEFAULT_F32 = 8;
constexpr int CTA_TILE_MAX         = 8;

/// A tiled launch: `n_nodes / TILE` full tiles plus a `n_nodes % TILE` tail on
/// the untiled body.  TWO kernels, ONE stream, in order -- they write disjoint
/// nodes, so the order is bookkeeping and not a dependency.
template <int T, int TILE, class WS>
inline void flatxsCtaTiledLaunch(const FlatXsView& v, cudaStream_t stream) {
    const int n_tiles = v.n_nodes / TILE;
    const int tail    = v.n_nodes - n_tiles * TILE;
    if (n_tiles > 0)
        kernelFlatXsCtaTile<T, TILE, WS><<<n_tiles, T * TILE, 0, stream>>>(v);
    if (tail > 0)
        kernelFlatXsCtaAt<T, WS><<<tail, T, 0, stream>>>(v, n_tiles * TILE);
}

/// Resolve the requested tile DOWN the ladder to the largest legal entry.  The
/// `if constexpr` is load-bearing: `kernelFlatXsCtaTile<256, 8, CtaWorkspace>`
/// is not a slow kernel, it is a COMPILE ERROR (2,048 threads, 58 KiB shared),
/// so the illegal combinations must never be instantiated.
template <int T, class WS>
inline void flatxsCtaDispatchTile(const FlatXsView& v, int tile,
                                  cudaStream_t stream) {
    if constexpr (ctaTileFits<T, 8, WS>()) {
        if (tile >= 8) { flatxsCtaTiledLaunch<T, 8, WS>(v, stream); return; }
    }
    if constexpr (ctaTileFits<T, 4, WS>()) {
        if (tile >= 4) { flatxsCtaTiledLaunch<T, 4, WS>(v, stream); return; }
    }
    // The bottom of the ladder.  It is a static_assert and not a runtime
    // fallback ON PURPOSE: a fallback would be a FOURTH launch site of the
    // untiled kernels, and "kernelFlatXsCtaF32 is launched from exactly one
    // place" is an invariant tools/test_gpu_fp32_contract.py holds.  A tile of
    // two costs 14,704 B of shared memory and 2*T threads, so it fits on every
    // architecture in RASBERY_CUDA_ARCHITECTURES; if that ever stops being
    // true the build says so here rather than silently launching a fifth arm.
    static_assert(ctaTileFits<T, 2, WS>(),
                  "TILE=2 must fit: it is the bottom of the tile ladder");
    flatxsCtaTiledLaunch<T, 2, WS>(v, stream);
}

/// The largest tile the ladder will actually run for `<threads, narrow>`.  The
/// runtime twin of `ctaTileFits`, so the backend receipt can state the tile
/// that RAN rather than the tile that was asked for.
inline int flatxsCtaTileResolved(int threads, bool narrow, int tile) {
    if (tile < 1) tile = 1;
    if (tile > CTA_TILE_MAX) tile = CTA_TILE_MAX;
    const int slot = narrow ? static_cast<int>(sizeof(CtaWorkspaceF32))
                            : static_cast<int>(sizeof(CtaWorkspace));
    int       t    = threads;
    if (t != 64 && t != 128 && t != 256) t = CTA_THREADS_DEFAULT;
    // The ladder only carries powers of two; round a stray 3/5/6/7 DOWN FIRST,
    // so the halving below cannot turn a 6 into a 2 while an 8 becomes a 4.
    while (tile > 1 && (tile & (tile - 1)) != 0) --tile;
    while (tile > 1
           && (t * tile > CTA_MAX_THREADS || tile * slot > CTA_SMEM_STATIC_MAX))
        tile /= 2;
    return tile;
}

/// Launch helper: the one place the block-size ladder is spelled, so the
/// production backend and the replay gate cannot pick different ladders.
/// `narrow` selects WP20's FP32 workspace.  DEFAULTED TO FALSE so every caller
/// that predates the arm -- the replay gate included -- keeps launching exactly
/// the kernel it launched before, which is what the feature-off byte-identity
/// claim rests on.  The ladder is the same on both arms, deliberately: block
/// size stays a performance parameter (P1) and precision stays a numerics one,
/// and mixing the two would make an A/B unattributable.
///
/// WP21-B2 ADDS `tile` ON THE SAME TERMS.  It DEFAULTS TO 1, which is the
/// pre-WP21-B2 text below, unchanged and still the arm the replay gate
/// certified.  By (P1) the tile is a performance parameter too: every entry
/// produces the same bytes, so a bad value is clamped down the ladder and
/// never fails a run.
inline void flatxsCtaLaunch(const FlatXsView& v, int threads, cudaStream_t stream,
                            bool narrow = false, int tile = 1) {
    const int grid = v.n_nodes;
    if (grid <= 0) return;
    if (tile > 1) {
        const int t = flatxsCtaTileResolved(threads, narrow, tile);
        switch (threads) {
            case 64:
                if (narrow) flatxsCtaDispatchTile<64, CtaWorkspaceF32>(v, t, stream);
                else        flatxsCtaDispatchTile<64, CtaWorkspace>(v, t, stream);
                return;
            case 256:
                if (narrow) flatxsCtaDispatchTile<256, CtaWorkspaceF32>(v, t, stream);
                else        flatxsCtaDispatchTile<256, CtaWorkspace>(v, t, stream);
                return;
            case 128:
            default:
                if (narrow)
                    flatxsCtaDispatchTile<CTA_THREADS_DEFAULT, CtaWorkspaceF32>(
                        v, t, stream);
                else
                    flatxsCtaDispatchTile<CTA_THREADS_DEFAULT, CtaWorkspace>(
                        v, t, stream);
                return;
        }
    }
    if (narrow) {
        switch (threads) {
            case 64:  kernelFlatXsCtaF32<64><<<grid, 64, 0, stream>>>(v); break;
            case 256: kernelFlatXsCtaF32<256><<<grid, 256, 0, stream>>>(v); break;
            case 128:
            default:
                kernelFlatXsCtaF32<CTA_THREADS_DEFAULT>
                    <<<grid, CTA_THREADS_DEFAULT, 0, stream>>>(v);
                break;
        }
        return;
    }
    switch (threads) {
        case 64:  kernelFlatXsCta<64><<<grid, 64, 0, stream>>>(v); break;
        case 256: kernelFlatXsCta<256><<<grid, 256, 0, stream>>>(v); break;
        case 128:
        default:
            kernelFlatXsCta<CTA_THREADS_DEFAULT>
                <<<grid, CTA_THREADS_DEFAULT, 0, stream>>>(v);
            break;
    }
}

} // namespace rasbery::flatxs
