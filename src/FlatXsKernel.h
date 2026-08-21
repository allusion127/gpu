#pragma once

// Shared host/device body of the unrodded flat-XS node update:
// gather the burnup-interpolated reference -> apply the resolved branch/history
// delta stream (Horner per element) -> refresh the light isotopes -> scatter
// back to the SoA arrays -> rebuild the node's macroscopic XS.  One call of
// flatxsSolveNode() reproduces one iteration of the unrodded branch of
// XSSet::UpdateFlatXS (XSSet::UpdateUnroddedNodeXS, src/XSSet.cpp) for one
// node, in the same arithmetic order, so the host build of this header is the
// reference the device build is scored against.
//
// Split of responsibilities (the reason this kernel can be bit-identical at
// all): everything transcendental or library-shaped -- the spectral-history
// coordinate math with its std::log calls, the burnup bracket lookups, the
// per-node branch interpolation weights -- happens on the HOST, in
// XSSet::BuildFlatXsStream, which reuses the very same
// ResolveSpectralHistoryDeltas the CPU arm calls.  What crosses to the device
// is a flat stream of (did, x, scale) applications per node, and the device
// applies them with the same Horner/accumulate arithmetic the CPU loop uses.
// glibc's log() is not correctly rounded and does not match CUDA's, so any
// design that evaluates coordinates on the device is unverifiable; this one
// never has to.
//
// Determinism contract (same rules as XsReconKernel.h):
//  - the delta stream is applied in the caller's order, which is the CPU
//    loop's applyDelta call order; no reordering, no batching by did;
//  - every isotope accumulation runs iso = 0..NISO-1 ascending;
//  - contraction is EXPLICIT via fxsMa<> below.  gcc -O3 -ffp-contract=fast
//    chooses per-statement whether to fuse a*b+c, and the xsrecon campaign
//    proved the choice cannot be predicted from the source text -- it must be
//    mined from a production capture (test/flatxs_replay.cpp sweeps every
//    combination against a RASBERY_FLATXS_DUMP capture and reports the
//    winning mask).  FLATXS_FORMS below is that mined mask.
//
// This header must stay compilable by both g++ and nvcc: no STL containers,
// no exceptions, no allocation.

#include "XsReconKernel.h" // xsrFma / xsrMul / NISO / NG / RASBERY_XSR_HD

namespace rasbery::flatxs {

using xsrecon::NG;
using xsrecon::NISO;
using xsrecon::xsrFma;
using xsrecon::xsrMul;

constexpr int N_ACTIVE = 9; // == N_ACTIVE_XT (XSTF,XSAF,XSFF,XSNF,XSKF,XSSF,FYLD,XS2N,XS3N)
constexpr int NMIC     = NISO * NG;      // 78 micro scalars per xt slot
constexpr int NLSM     = NG * NG;        // 4 lumped scatter elements
constexpr int NMSM     = NISO * NG * NG; // 156 micro scatter elements

// Isotope rows RefreshLightIsotopes rewrites (contiguous by design of the
// Chiffon isotope registry: H-1, B-10, O-16).
constexpr int IH1  = 0;
constexpr int IB10 = 1;
constexpr int IO16 = 2;

// Verbatim from src/XSSet.cpp (anonymous namespace constants).
constexpr double WATER_NUMBER_DENSITY = 0.033427699;
constexpr double BORON_DENSITY_FACTOR = 5.5707678E-8;

/// Contraction-site bits for the forms mask.  One bit per statement whose
/// fused/unfused choice gcc makes independently (xsrecon finding: the SAME
/// expression contracts differently per loop).  fused=1 means single-rounding
/// fma; unfused=0 means the product rounds before the add.
enum FormBit : unsigned {
    F_HORNER_LMP  = 1u << 0, ///< applyDelta lmpx scalar: val = val*xloc + c
    F_ACC_LMP     = 1u << 1, ///< applyDelta lmpx scalar: dst[e] += scale*val
    F_HORNER_LSM  = 1u << 2, ///< applyDelta lmpx scatter Horner
    F_ACC_LSM     = 1u << 3, ///< applyDelta lmpx scatter accumulate
    F_HORNER_MIC  = 1u << 4, ///< applyDelta micx scalar Horner (omp simd loop)
    F_ACC_MIC     = 1u << 5, ///< applyDelta micx scalar accumulate
    F_HORNER_MSM  = 1u << 6, ///< applyDelta micx scatter Horner (omp simd loop)
    F_ACC_MSM     = 1u << 7, ///< applyDelta micx scatter accumulate
    F_MACRO_SCAL  = 1u << 8, ///< rebuild: val += mt[iso*ng+ig] * iden[iso]
    F_MACRO_SSM   = 1u << 9, ///< rebuild scatter: val += bms[..] * iden[iso]
};

/// Production forms mask, mined by test/flatxs_replay.cpp --sweep against the
/// APR1400 profile-deck capture (2026-08-21): every exercised site FUSED.
/// The sweep's exact-match set was {0x3F1..0x3FF step 2} -- bits 1
/// (F_ACC_LMP), 2 (F_HORNER_LSM), 3 (F_ACC_LSM) are NOT constrained by that
/// capture (the lumped-scatter delta surfaces of this library leave them
/// unexercised), so a library that does exercise them must re-run the sweep.
/// Both replay gates (host and --fmad=false device) PASS 0-ULP at this mask,
/// and the full-deck M1 A/B is bit-identical.  Re-run the sweep after any
/// XSSet.cpp change that touches UpdateUnroddedNodeXS.
constexpr unsigned FLATXS_FORMS = 0x3FF;

/// Contraction policy with the mask baked in at compile time: `bit` is a
/// literal at every call site, so the ternary folds and the device code is
/// a bare DFMA (or separately-rounded multiply; the TU builds with
/// --fmad=false so nvcc cannot re-fuse the unfused arm).
struct StaticForms {
    RASBERY_XSR_HD double ma(unsigned bit, double a, double b, double c) const {
#if defined(__CUDA_ARCH__)
        return (FLATXS_FORMS & bit) ? fma(a, b, c) : a * b + c;
#else
        return (FLATXS_FORMS & bit) ? xsrFma(a, b, c) : xsrMul(a, b) + c;
#endif
    }
};

/// Runtime-mask policy for the form sweep (host-only test tool; never on the
/// production path).  Same rounding guarantees per arm as StaticForms.
struct RuntimeForms {
    unsigned mask = 0;
    double   ma(unsigned bit, double a, double b, double c) const {
        return (mask & bit) ? xsrFma(a, b, c) : xsrMul(a, b) + c;
    }
};

/// One fitted delta surface's metadata, flattened from XSSet::DeltaInfo.
/// knot_count is not needed by the apply side (the interval search only walks
/// nintervals-1 knots ahead of knot_offset).
struct DeltaMeta {
    int nord;
    int mode;
    int ncoeff;
    int coeff_base;
    int knot_offset;
};

/// Pointer view of one instance's flat-XS state.  Same philosophy as
/// xsrecon::BatchView: the body only sees pointers, so arrays may live on the
/// host or the device without touching the body.
///
/// Indexing (all SoA over nxyz):
///  - ref_lmp[t]/lmp[t]:   [ig*nxyz + l], t indexes the ACTIVE_XT list
///  - ref_mic[t]/mic[t]:   [(iso*NG+ig)*nxyz + l]
///  - *_lsm:               [(igs*NG+ige)*nxyz + l]
///  - *_msm:               [(iso*NG*NG+igs*NG+ige)*nxyz + l]
///  - xs[xt]:              [ig*nxyz + l], xt indexes ALL 11 Chiffon slots
///  - iden:                [iso*nxyz + l]
///  - coeff_lmp[t]:        [row*NG + e],   e < NG
///  - coeff_lsm:           [row*NLSM + e], e < NLSM
///  - coeff_mic[t]:        [row*NMIC + e], e < NMIC
///  - coeff_msm:           [row*NMSM + e], e < NMSM
/// The delta stream is the concatenation of every node's applyDelta calls in
/// CPU order: node v.nodes[i] owns stream entries [node_off[i],
/// node_off[i]+node_cnt[i]).
struct FlatXsView {
    // library coefficient tables (immutable after load; device copy shared
    // across instances)
    const double*    coeff_lmp[N_ACTIVE];
    const double*    coeff_lsm;
    const double*    coeff_mic[N_ACTIVE];
    const double*    coeff_msm;
    const double*    knots;
    const DeltaMeta* deltas;
    int              has_coeff_micx;

    // burnup-interpolated reference state (per instance, rebuilt by
    // PrecomputeBranchCoefficients; device copy keyed by ref generation)
    const double* ref_lmp[N_ACTIVE];
    const double* ref_lsm;
    const double* ref_mic[N_ACTIVE];
    const double* ref_msm;

    // live outputs
    double* lmp[N_ACTIVE];
    double* lsm;
    double* mic[N_ACTIVE];
    double* msm;
    double* xs[xsrecon::NXS]; ///< all 11 slots; kernel writes ACTIVE + XSDF/XSRF
    double* xs_ssm;
    double* iden;

    // per-node inputs
    const double* wvfr; ///< _node_wvfr, already refreshed to _ref_wvfr by the host
    const double* dmod;
    const double* bppm;

    // Host-only staging pointers (never dereferenced by the body): the full
    // 11-slot live micx/lmpx arrays, so the backend can re-upload the whole
    // resident block on a generation mismatch.  ACTIVE-slot pointers above
    // alias 9 of these.
    const double* mic_all[xsrecon::NXS];
    const double* lmp_all[xsrecon::NXS];

    // resolved application stream
    const int*    stream_did;
    const double* stream_x;
    const double* stream_scale;
    const int*    node_off;
    const int*    node_cnt;
    const int*    nodes; ///< target (unrodded) node list
    int           n_nodes;
    int           nxyz;

    double boron_dmod_average; ///< FuelVolumeAverageDmod, computed on the host
    int    use_average_dmod;   ///< USE_AVERAGE_DMOD_FOR_BORON (constexpr false today)
};

/// BoronDmod(src/XSSet.cpp): average toggle is a compile-time constant there;
/// carried as data here so the body needs no XSSet include.
RASBERY_XSR_HD inline double fxsBoronDmod(const FlatXsView& v, int l) {
    return v.use_average_dmod ? v.boron_dmod_average : v.dmod[l];
}

/// One unrodded node: the full UpdateUnroddedNodeXS body minus what the host
/// already did (wvfr refresh, delta resolution).  Workspace layout and pass
/// order match the CPU function statement for statement.
template <class POL>
RASBERY_XSR_HD inline void flatxsSolveNode(const FlatXsView& v, int i,
                                           const POL& pol) {
    const int nxyz = v.nxyz;
    const int l    = v.nodes[i];

    double bl[N_ACTIVE * NG];   // lmpx scalars [t*NG + ig]
    double bls[NLSM];           // lmpx scatter [igs*NG + ige]
    double bm[N_ACTIVE * NMIC]; // micx scalars [t*NMIC + iso*NG + ig]
    double bms[NMSM];           // micx scatter [iso*NG*NG + igs*NG + ige]

    // 1. Gather the reference state into the workspace (plain copies).
    for (int t = 0; t < N_ACTIVE; ++t)
        for (int ig = 0; ig < NG; ++ig)
            bl[t * NG + ig] = v.ref_lmp[t][ig * nxyz + l];
    for (int sm = 0; sm < NLSM; ++sm)
        bls[sm] = v.ref_lsm[sm * nxyz + l];
    for (int t = 0; t < N_ACTIVE; ++t)
        for (int e = 0; e < NMIC; ++e)
            bm[t * NMIC + e] = v.ref_mic[t][e * nxyz + l];
    for (int e = 0; e < NMSM; ++e)
        bms[e] = v.ref_msm[e * nxyz + l];

    // 2. Apply the resolved delta stream in CPU call order.
    const int s0 = v.node_off[i];
    const int s1 = s0 + v.node_cnt[i];
    for (int s = s0; s < s1; ++s) {
        const int    did   = v.stream_did[s];
        const double x     = v.stream_x[s];
        const double scale = v.stream_scale[s];
        // The host only streams live applications, but keep the CPU guard so
        // a defensive caller cannot change the arithmetic.
        if (did < 0 || scale == 0.0) continue;

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

        for (int t = 0; t < N_ACTIVE; ++t) {
            const double* cdata = v.coeff_lmp[t];
            double*       dst   = bl + t * NG;
            for (int e = 0; e < NG; ++e) {
                double val = cdata[(base + nord - 1) * NG + e];
                for (int p = nord - 2; p >= 0; --p)
                    val = pol.ma(F_HORNER_LMP, val, xloc, cdata[(base + p) * NG + e]);
                dst[e] = pol.ma(F_ACC_LMP, scale, val, dst[e]);
            }
        }
        for (int e = 0; e < NLSM; ++e) {
            double val = v.coeff_lsm[(base + nord - 1) * NLSM + e];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_LSM, val, xloc, v.coeff_lsm[(base + p) * NLSM + e]);
            bls[e] = pol.ma(F_ACC_LSM, scale, val, bls[e]);
        }

        if (!v.has_coeff_micx) continue;
        for (int t = 0; t < N_ACTIVE; ++t) {
            const double* cdata = v.coeff_mic[t];
            double*       dst   = bm + t * NMIC;
            for (int e = 0; e < NMIC; ++e) {
                double val = cdata[(base + nord - 1) * NMIC + e];
                for (int p = nord - 2; p >= 0; --p)
                    val = pol.ma(F_HORNER_MIC, val, xloc, cdata[(base + p) * NMIC + e]);
                dst[e] = pol.ma(F_ACC_MIC, scale, val, dst[e]);
            }
        }
        for (int e = 0; e < NMSM; ++e) {
            double val = v.coeff_msm[(base + nord - 1) * NMSM + e];
            for (int p = nord - 2; p >= 0; --p)
                val = pol.ma(F_HORNER_MSM, val, xloc, v.coeff_msm[(base + p) * NMSM + e]);
            bms[e] = pol.ma(F_ACC_MSM, scale, val, bms[e]);
        }
    }

    // 3. Snapshot the node's densities, then RefreshLightIsotopes: pure
    //    products, no contraction opportunity, exact on both compilers.
    double iden[NISO];
    for (int iso = 0; iso < NISO; ++iso)
        iden[iso] = v.iden[iso * nxyz + l];

    const double nH2O       = v.dmod[l] * v.wvfr[l] * WATER_NUMBER_DENSITY;
    const double boron_dmod = fxsBoronDmod(v, l);
    iden[IH1]  = 2.0 * nH2O;
    iden[IO16] = nH2O;
    iden[IB10] = boron_dmod * v.wvfr[l] * v.bppm[l] * BORON_DENSITY_FACTOR;
    v.iden[IH1 * nxyz + l]  = iden[IH1];
    v.iden[IB10 * nxyz + l] = iden[IB10];
    v.iden[IO16 * nxyz + l] = iden[IO16];

    // 4. Scatter the workspace back to the SoA arrays (plain copies).
    for (int t = 0; t < N_ACTIVE; ++t)
        for (int ig = 0; ig < NG; ++ig)
            v.lmp[t][ig * nxyz + l] = bl[t * NG + ig];
    for (int sm = 0; sm < NLSM; ++sm)
        v.lsm[sm * nxyz + l] = bls[sm];
    for (int t = 0; t < N_ACTIVE; ++t)
        for (int e = 0; e < NMIC; ++e)
            v.mic[t][e * nxyz + l] = bm[t * NMIC + e];
    for (int e = 0; e < NMSM; ++e)
        v.msm[e * nxyz + l] = bms[e];

    // 5. Rebuild this node's macroscopic XS from the workspace.  The CPU
    //    reads _iden from memory here; rows 0..2 of the local copy hold
    //    exactly the values just stored, so the operands are identical.
    const int active_xt[N_ACTIVE] = {
        xsrecon::T_XSTF, xsrecon::T_XSAF, xsrecon::T_XSFF,
        xsrecon::T_XSNF, xsrecon::T_XSKF, xsrecon::T_XSSF,
        xsrecon::T_FYLD, xsrecon::T_XS2N, xsrecon::T_XS3N};
    for (int t = 0; t < N_ACTIVE; ++t) {
        const double* mt = bm + t * NMIC;
        for (int ig = 0; ig < NG; ++ig) {
            double val = bl[t * NG + ig];
            for (int iso = 0; iso < NISO; ++iso)
                val = pol.ma(F_MACRO_SCAL, mt[iso * NG + ig], iden[iso], val);
            v.xs[active_xt[t]][ig * nxyz + l] = val;
        }
    }
    for (int igs = 0; igs < NG; ++igs) {
        for (int ige = 0; ige < NG; ++ige) {
            double val = bls[igs * NG + ige];
            for (int iso = 0; iso < NISO; ++iso)
                val = pol.ma(F_MACRO_SSM, bms[iso * NLSM + igs * NG + ige],
                             iden[iso], val);
            v.xs_ssm[(igs * NG + ige) * nxyz + l] = val;
        }
    }
    for (int ig = 0; ig < NG; ++ig) {
        const double tr = v.xs[xsrecon::T_XSTF][ig * nxyz + l];
        v.xs[xsrecon::T_XSDF][ig * nxyz + l] =
            (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;

        double rf = v.xs[xsrecon::T_XSAF][ig * nxyz + l];
        for (int ige = 0; ige < NG; ++ige)
            rf += v.xs_ssm[(ig * NG + ige) * nxyz + l];
        v.xs[xsrecon::T_XSRF][ig * nxyz + l] = rf;
    }
}

/// Host-side helper for the NodeSpectralIndex workspace probe: the value of
/// ONE micx scalar element after a prefix of the node's delta stream, using
/// the same forms the full loop uses.  NodeSpectralIndex reads exactly two
/// such elements (XSAF Pu-239/B-10 thermal), and the history resolution needs
/// them before the kernel runs.
template <class POL>
inline double flatxsProbeMicElement(const FlatXsView& v, int l, int t, int e,
                                    const int* dids, const double* xs_,
                                    const double* scales, int count,
                                    const POL& pol) {
    double acc = v.ref_mic[t][e * v.nxyz + l];
    if (!v.has_coeff_micx) return acc;
    for (int s = 0; s < count; ++s) {
        const int    did   = dids[s];
        const double scale = scales[s];
        if (did < 0 || scale == 0.0) continue;
        const DeltaMeta dm   = v.deltas[did];
        int             base = dm.coeff_base;
        int             nord = dm.nord;
        double          xloc = xs_[s];
        if (dm.mode == 1) {
            const int nintervals = dm.nord / dm.ncoeff;
            int       interval   = nintervals - 1;
            for (int k = 0; k < nintervals - 1; ++k) {
                if (xs_[s] < v.knots[dm.knot_offset + k + 1]) {
                    interval = k;
                    break;
                }
            }
            xloc = xs_[s] - v.knots[dm.knot_offset + interval];
            base += interval * dm.ncoeff;
            nord = dm.ncoeff;
        }
        const double* cdata = v.coeff_mic[t];
        double        val   = cdata[(base + nord - 1) * NMIC + e];
        for (int p = nord - 2; p >= 0; --p)
            val = pol.ma(F_HORNER_MIC, val, xloc, cdata[(base + p) * NMIC + e]);
        acc = pol.ma(F_ACC_MIC, scale, val, acc);
    }
    return acc;
}

} // namespace rasbery::flatxs
