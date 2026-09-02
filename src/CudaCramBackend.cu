// GPU CRAM Bateman depletion: XSSet::Deplete + XSSet::CorrectorStep node loops
// on the device.  Contract, cost ledger and gate class: CudaCramBackend.h.
//
// The arithmetic below is a line-for-line transcription of XSSet.cpp's
// DepleteNode / CorrectorStep node bodies and milk.h's solveBatemanCRAM, in the
// same statement order, with the same intermediate temporaries, the same
// per-node `value == 0.0` row compression and the same break tests.
// --fmad=false is set on this file in CMakeLists.txt for exactly the reason
// CudaXsReconBackend.cu and CudaPprBackend.cu have it: nvcc must not contract
// a*b+c into an fma where the host compiler did not, or the deviation stops
// being attributable to the one place the header documents.

#include "CudaCramBackend.h"
#include "GpuCaptureArbiter.h"
#include "XferLedger.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace rasbery {

namespace {

// ---------------------------------------------------------------------------
// Compile-time shape
// ---------------------------------------------------------------------------
//
// NISO and the pattern bound are compile-time so the per-node working set can
// be a plain local array (CUDA interleaves per-thread locals across the warp,
// so a 36-unknown Gauss-Seidel gets coalesced loads without a hand-written
// node-strided scratch block).  Both are checked at runtime against the deck:
// the backend declines rather than overruns.

constexpr int kNiso   = 39;  ///< Chiffon::Isotope::niso
constexpr int kMaxNnz = 192; ///< measured union pattern is 124 off-diagonals

/// N_XS_SCALAR slot ids (Chiffon::XSTYPE order), restated here so this TU does
/// not drag Model.h through nvcc.  XSSet.cpp already static_asserts that order
/// against XsReconKernel.h; tools/test_cram_gpu_contract.py asserts these four.
constexpr int kXSAF = 2;
constexpr int kXSFF = 3;
constexpr int kXS2N = 9;
constexpr int kXS3N = 10;

/// The four condensed slots this kernel reads, in the order the host's
/// `condensed` array holds them.  BuildTransitionMatrix reads XSAF/XSFF/XS2N/
/// XS3N and ComputeXeEquilibrium reads XSFF/XSAF; the other seven of the
/// eleven the host condenses are never read downstream, so they are never
/// condensed here and their micro-XS blocks are never uploaded.  That is a
/// 58 MB -> 21 MB cut in the per-statepoint H2D, and it is exact because a
/// value nothing reads cannot change a result.
constexpr int kSlot[4]  = {kXSAF, kXSFF, kXS2N, kXS3N};
constexpr int kCondXSAF = 0;
constexpr int kCondXSFF = 1;
constexpr int kCondXS2N = 2;
constexpr int kCondXS3N = 3;

/// WP20.1: the narrow-block handover, D2D.
///
/// Under RASBERY_GPU_FP32 the flat-XS micx block is float and CRAM's own state
/// is not (src/GpuFp32Arm.h: the partial-fraction sum cancels, and float has no
/// headroom for that cancellation).  So the four condensation slots arrive
/// through a widening copy instead of a memcpy.  It is a plain elementwise
/// convert -- no arithmetic, no reduction, no order to argue about -- and it
/// still moves only `n * sizeof(float)` off DRAM, which is the whole point.
__global__ void kCramWidenMic(double* __restrict__ dst,
                              const float* __restrict__ src, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<double>(src[i]);
}

// ---------------------------------------------------------------------------
// CRAM order-8 constants -- milk.h solveBatemanCRAM, `order == 8` branch
// ---------------------------------------------------------------------------

constexpr double kAlpha0    = 1.1722341374385704e-08;
constexpr int    kMaxIter   = 64;
constexpr double kRelTol    = 1.0e-13;
constexpr double kMatrixSgn = -1.0;
constexpr int    kPoleCount = 4;
constexpr double kAbsTol    = 1.0e-28;
constexpr double kDiagTol   = 1.0e-30;

// ---------------------------------------------------------------------------
// WP21-D: where the four poles live
// ---------------------------------------------------------------------------
//
// The serial kernel walks pole 0..3 inside ONE thread, which is why block 39 of
// the 238 ncu profile shows 133 blocks x 64 threads, 4.38/4.49 ms per launch,
// 4.15 % warps active and 0.45 % of DRAM peak: the kernel is neither bandwidth-
// nor compute-bound, it is a long serial latency chain running on 2 % of the
// card.  The four poles are four INDEPENDENT complex linear solves against the
// same real matrix -- they share `mdiag` and the compressed off-diagonals and
// nothing else -- so the only thing that couples them is the closing sum
// `accum[row] += x_pole[row]`, and that sum is a fixed left-to-right walk over
// pole = 0, 1, 2, 3.  Moving each pole to its own LANE and re-forming that walk
// with __shfl_sync IN POLE ORDER gives the identical sequence of double
// additions, so the pole-parallel arm is B0 against the serial one, not N1.
//
// FOUR LANES AND NOT EIGHT.  A real/imaginary split would need a shuffle inside
// the Gauss-Seidel inner product, where the host's arithmetic is the complex
// `sum -= vals[i] * x[cols[i]]`; that reassociates, and the class would drop to
// N1 for a factor the occupancy does not need.  The sweep itself stays serial in
// `row` and serial in `i` -- WP21-D widened the node, not the sweep.
constexpr int kLanesPerNode = kPoleCount; ///< pole-parallel arm: lane == pole

/// Which node body the launch selects.  RASBERY_GPU_CRAM_PARALLEL.
enum class Variant : int {
    kSerial = 0, ///< one thread per node, poles in a loop (the WP21-D baseline)
    kPole4  = 1, ///< default: kLanesPerNode lanes per node, lane == pole
};

__constant__ double kAlphaRe[4] = {
    +1.83174069610856716e+00, -2.43619809577363400e+00,
    +6.32575834187860564e-01, -2.81291599910903876e-02};
__constant__ double kAlphaIm[4] = {
    -9.52542527224556679e+00, +3.71667983752542863e+00,
    -4.43912790240850230e-01, +1.15770931709880849e-02};
__constant__ double kThetaRe[4] = {
    -3.22092672186933981e+00, -2.29222964471934798e+00,
    -2.69470045809068803e-01, +3.40856168532368731e+00};
__constant__ double kThetaIm[4] = {
    +1.19361884200025181e+00, +3.60076959131180274e+00,
    +6.08203046216700294e+00, +8.77303318542488775e+00};

// Per-node status codes.  Non-zero anywhere makes the WHOLE call fail open.
constexpr unsigned int kOk         = 0u;
constexpr unsigned int kZeroDiag   = 1u; ///< milk: CRAM Gauss-Seidel zero diagonal
constexpr unsigned int kNoConverge = 2u; ///< milk: CRAM Gauss-Seidel did not converge
constexpr unsigned int kNonFinite  = 4u; ///< a density came out NaN/Inf

bool truthy(const char* v) {
    if (v == nullptr) return false;
    const std::string s(v);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
             s == "FALSE");
}

// ---------------------------------------------------------------------------
// Complex arithmetic, in the shapes the host compiler actually emits
// ---------------------------------------------------------------------------

__device__ inline double devInf() {
    return __longlong_as_double(0x7ff0000000000000LL);
}

/// gcc's inline `__complex__ double` multiply.  Written out rather than left to
/// a library so the operand order is the host's; the NaN-recovery call gcc
/// appends (__muldc3) is unreachable for finite operands, and a node that
/// reaches non-finite operands is failed open by the status byte instead.
__device__ inline void cmul(double ar, double ai, double br, double bi,
                            double* xr, double* xi) {
    *xr = ar * br - ai * bi;
    *xi = ar * bi + ai * br;
}

/// libgcc's __divdc3, transcribed.  libstdc++ lowers `complex<double> /
/// complex<double>` to this, so a device path that used the naive formula --
/// or CUDA's own cuCdiv, which is Smith's method WITHOUT the logb rescale --
/// would not be running the host's division.  `logb`, `scalbn` and the divides
/// are the device libm's, which is the declared N1 seam.
__device__ inline void cdiv(double a, double b, double c, double d,
                            double* xo, double* yo) {
    int          ilogbw = 0;
    const double logbw  = logb(fmax(fabs(c), fabs(d)));
    if (isfinite(logbw)) {
        ilogbw = static_cast<int>(logbw);
        c      = scalbn(c, -ilogbw);
        d      = scalbn(d, -ilogbw);
    }
    const double denom = c * c + d * d;
    double       x     = scalbn((a * c + b * d) / denom, -ilogbw);
    double       y     = scalbn((b * c - a * d) / denom, -ilogbw);
    if (isnan(x) && isnan(y)) {
        const double inf = devInf();
        if (denom == 0.0 && (!isnan(a) || !isnan(b))) {
            x = copysign(inf, c) * a;
            y = copysign(inf, c) * b;
        } else if ((isinf(a) || isinf(b)) && isfinite(c) && isfinite(d)) {
            a = copysign(isinf(a) ? 1.0 : 0.0, a);
            b = copysign(isinf(b) ? 1.0 : 0.0, b);
            x = inf * (a * c + b * d);
            y = inf * (b * c - a * d);
        } else if (isinf(logbw) && isfinite(a) && isfinite(b)) {
            c = copysign(isinf(c) ? 1.0 : 0.0, c);
            d = copysign(isinf(d) ? 1.0 : 0.0, d);
            x = 0.0 * (a * c + b * d);
            y = 0.0 * (b * c - a * d);
        }
    }
    *xo = x;
    *yo = y;
}

/// milk::detail::magnitude(std::complex<double>), branch for branch.  The
/// out-of-range arm is libstdc++'s std::abs(complex) == hypot.
__device__ inline double cmagnitude(double re, double im) {
    const double ar = fabs(re);
    const double ai = fabs(im);
    const double mx = (ar < ai) ? ai : ar; // std::max(ar, ai)
    return (mx < 1.0e150 && mx > 1.0e-150) ? sqrt(re * re + im * im)
                                           : hypot(re, im);
}

/// std::max for doubles: `(a < b) ? b : a`.  fmax is NOT the same function.
__device__ inline double dmax(double a, double b) { return (a < b) ? b : a; }

// ---------------------------------------------------------------------------
// Device context
// ---------------------------------------------------------------------------

struct DevCtx {
    // shape / isotope registry
    int niso;
    int first;
    int ac_first, ac_last;
    int i135, xe135, xe135m;
    int u234, u235, u238, np237;
    int ng;
    int nxyz;

    // step scalars
    double dt;
    int    xe_transient;
    double norm_factor;        ///< predictor
    double bos_norm, eos_norm; ///< corrector
    int    density_average;
    int    xe_equilibrium_fix;

    // node-invariant tables
    const double*        dep_decay; ///< [niso*niso], row-major
    const double*        dep_trans; ///< [niso*niso], row-major
    const int*           row_ptr;   ///< [niso-first+1], off-diagonal pattern
    const unsigned char* col_idx;   ///< [nnz], absolute isotope index, ascending
    const double*        dfac;      ///< [nxyz]
    const double*        vol;       ///< [nxyz]

    // per-step blocks
    const double* flux;     ///< [nxyz*ng] node-major; EOS flux in the corrector
    const double* flux_bos; ///< [nxyz*ng]
    const double* xskf_bos; ///< [ng*nxyz]
    const double* xskf_eos; ///< [ng*nxyz]
    const double* mic[4];     ///< EOS condensation inputs, kSlot order
    const double* mic_bos[4]; ///< BOS condensation inputs (corrector only)
    const double* iden_in;    ///< [niso*nxyz]: predictor _iden / corrector _iden_bos
    const double* iden_pred;  ///< [niso*nxyz]: corrector's N^P, decart mode only
    const int*    burn_bos;   ///< [nxyz]

    // outputs (device staging; nothing reaches the host until the status is clean)
    double*        iden_out; ///< [niso*nxyz]
    int*           burn_out; ///< [nxyz]
    unsigned long long* stats; ///< [0]=status OR, [1]=gs iterations, [2]=gs solves
};

// ---------------------------------------------------------------------------
// The node body
// ---------------------------------------------------------------------------

/// XSSet::BuildTransitionMatrix's value at one (row, col), assembled in the
/// host's update order: depDecay, then the p-loop's single touch at p == col,
/// then the two (n,2n) specials.  Each entry is touched at most once by the
/// p-loop, so there is no accumulation order to preserve beyond this.
__device__ inline double transEntry(const DevCtx& x, const double* cond4,
                                    double sumflux, int r, int c) {
    double v = x.dep_decay[r * x.niso + c];

    const double xsaf_val = cond4[c * 4 + kCondXSAF];
    if (r == c) {
        v -= xsaf_val * sumflux;
    } else {
        const double topo = x.dep_trans[r * x.niso + c];
        if (topo != 0.0) {
            const double xsff_val = cond4[c * 4 + kCondXSFF];
            const double xs2n_val = cond4[c * 4 + kCondXS2N];
            const double xs3n_val = cond4[c * 4 + kCondXS3N];
            const double xscf     = xsaf_val - xsff_val + xs2n_val + 2.0 * xs3n_val;

            const bool isActinide = (c >= x.ac_first && c <= x.ac_last);
            const bool dIsNonAc   = (r < x.ac_first || r > x.ac_last);
            if (isActinide && dIsNonAc)
                v += topo * xsff_val * sumflux;
            else
                v += topo * xscf * sumflux;
        }
    }

    if (r == x.u234 && c == x.u235) v += cond4[x.u235 * 4 + kCondXS2N] * sumflux;
    if (r == x.np237 && c == x.u238) v += cond4[x.u238 * 4 + kCondXS2N] * sumflux;
    return v;
}

/// The split, once per node: the host's row scan of `A(row, col)` over
/// `col >= first`, with its `value == 0.0` compression, in ascending row and
/// ascending column order.  POLE-INVARIANT -- nothing in it reads theta or
/// alpha -- which is what lets the pole-parallel arm hand each pole its own lane
/// and simply have all four lanes rebuild this identically.  (Sharing it through
/// shared memory was costed and rejected: 3.7 KB per node caps a 64-thread block
/// at eight nodes, and that costs more occupancy than the redundancy costs
/// instructions -- the build is ~200 transEntry evaluations against the sweeps'
/// ~25,000 complex operations.)
__device__ inline void cramBuildSplit(const DevCtx& x, const double* cond4,
                                      double sumflux, double dt, double* cval,
                                      unsigned char* ccol, int* cend,
                                      double* mdiag) {
    const int n     = x.niso;
    const int first = x.first;

    int fill = 0;
    for (int row = first; row < n; ++row) {
        const int k0 = x.row_ptr[row - first];
        const int k1 = x.row_ptr[row - first + 1];
        for (int k = k0; k < k1; ++k) {
            const int    col   = static_cast<int>(x.col_idx[k]);
            const double value = transEntry(x, cond4, sumflux, row, col);
            if (value == 0.0) continue; // the host's `if (value == 0.0) continue;`
            cval[fill] = kMatrixSgn * value * dt;
            ccol[fill] = static_cast<unsigned char>(col);
            ++fill;
        }
        cend[row] = fill;

        const double dvalue = transEntry(x, cond4, sumflux, row, row);
        mdiag[row]          = (dvalue == 0.0) ? 0.0 : (kMatrixSgn * dvalue * dt);
    }
}

/// ONE pole of milk.h's partial-fraction sum: the body of its
/// `for (pole = 0; pole < pole_count; ++pole)` loop, statement for statement.
/// `xr`/`xi` receive that pole's x; `*iters_out` the Gauss-Seidel sweeps it
/// actually ran (0 when the diagonal test refused before the first one).
__device__ unsigned int cramPole(const DevCtx& x, const double* iden,
                                 const double* cval, const unsigned char* ccol,
                                 const int* cend, const double* mdiag, int pole,
                                 double* xr, double* xi, int* iters_out) {
    const int n     = x.niso;
    const int first = x.first;

    *iters_out = 0;

    const double th_r = kThetaRe[pole];
    const double th_i = kThetaIm[pole];
    const double al_r = kAlphaRe[pole];
    const double al_i = kAlphaIm[pole];

    double rhs_norm = 0.0;
    for (int row = first; row < n; ++row) {
        const double dg_r = mdiag[row] - th_r;
        const double dg_i = 0.0 - th_i;
        if (cmagnitude(dg_r, dg_i) <= kDiagTol) return kZeroDiag;

        // rhs[row] = complex(N[row], 0) * alpha[pole]
        double rr, ri;
        cmul(iden[row], 0.0, al_r, al_i, &rr, &ri);
        cdiv(rr, ri, dg_r, dg_i, &xr[row], &xi[row]);
        rhs_norm = dmax(rhs_norm, cmagnitude(rr, ri));
    }
    rhs_norm = dmax(rhs_norm, 1.0e-30);

    bool converged = false;
    int  iters     = 0;
    for (int iter = 0; iter < kMaxIter; ++iter) {
        // The compacted lists are laid out row by row in ascending row and
        // ascending column order, so one running cursor walks them in the
        // host's `for (i = 0; i < cols.size(); ++i)` order.
        int k = 0;
        for (int row = first; row < n; ++row) {
            // sum = rhs[row]
            double sr, si;
            cmul(iden[row], 0.0, al_r, al_i, &sr, &si);
            for (; k < cend[row]; ++k) {
                const int c = static_cast<int>(ccol[k]);
                double    pr, pi;
                cmul(cval[k], 0.0, xr[c], xi[c], &pr, &pi);
                sr -= pr;
                si -= pi;
            }
            const double dg_r = mdiag[row] - th_r;
            const double dg_i = 0.0 - th_i;
            cdiv(sr, si, dg_r, dg_i, &xr[row], &xi[row]);
        }

        double max_residual = 0.0;
        k                   = 0;
        for (int row = first; row < n; ++row) {
            const double dg_r = mdiag[row] - th_r;
            const double dg_i = 0.0 - th_i;
            double       axr, axi;
            cmul(dg_r, dg_i, xr[row], xi[row], &axr, &axi);
            for (; k < cend[row]; ++k) {
                const int c = static_cast<int>(ccol[k]);
                double    pr, pi;
                cmul(cval[k], 0.0, xr[c], xi[c], &pr, &pi);
                axr += pr;
                axi += pi;
            }
            double rr, ri;
            cmul(iden[row], 0.0, al_r, al_i, &rr, &ri);
            max_residual = dmax(max_residual, cmagnitude(rr - axr, ri - axi));
        }

        iters = iter + 1;
        if (max_residual <= kAbsTol + kRelTol * rhs_norm) {
            converged = true;
            break;
        }
    }

    *iters_out = iters;
    return converged ? kOk : kNoConverge;
}

/// milk.h's closing loop: `alpha0 * N[row] + 2 * accum[row].real()`, the
/// negative clamp, and the finite test the host's `static_cast<T>` cannot do.
/// The IMAGINARY accumulator milk.h carries is never read by that expression --
/// `.real()` is the only projection taken -- so it is not carried here either;
/// dropping a sum nothing reads changes no result and saves 312 B of local
/// memory per thread on a kernel whose problem is per-thread state.
__device__ inline unsigned int cramFinish(const DevCtx& x, double* iden,
                                          const double* accr) {
    unsigned int status = kOk;
    for (int row = x.first; row < x.niso; ++row) {
        double value = kAlpha0 * iden[row] + 2.0 * accr[row];
        if (value < 0.0 && fabs(value) < 1.0e-12) value = 0.0;
        if (!isfinite(value)) status |= kNonFinite;
        iden[row] = value;
    }
    return status;
}

/// milk::Solver<double>::solveBatemanCRAM(A, N, dt, out=N, ws, 8, first), the
/// SERIAL arm: one thread walks all four poles.  `iden` is both N and out,
/// exactly as every host call site passes it.  Returns a status mask; on
/// anything non-zero `iden` is left in whatever state the partial solve reached,
/// which is why the caller writes to staging.
__device__ unsigned int cramSolveNode(const DevCtx& x, double* iden,
                                      const double* cond4, double sumflux,
                                      double dt, unsigned long long* gs_iters) {
    const int n     = x.niso;
    const int first = x.first;

    double        cval[kMaxNnz];
    unsigned char ccol[kMaxNnz];
    int           cend[kNiso];
    double        mdiag[kNiso];
    cramBuildSplit(x, cond4, sumflux, dt, cval, ccol, cend, mdiag);

    double accr[kNiso];
    for (int row = 0; row < n; ++row) accr[row] = 0.0;

    double xr[kNiso], xi[kNiso];

    for (int pole = 0; pole < kPoleCount; ++pole) {
        int                iters = 0;
        const unsigned int st =
            cramPole(x, iden, cval, ccol, cend, mdiag, pole, xr, xi, &iters);
        // A zero diagonal refuses BEFORE the first sweep, so it contributes no
        // sweeps; a non-convergence contributes the 64 it ran.  gs_iters_mean is
        // the receipt observable that says the device solved the same iteration
        // the host does, so which of the two it is has to survive.
        if (st == kZeroDiag) return kZeroDiag;
        *gs_iters += static_cast<unsigned long long>(iters);
        if (st != kOk) return st;

        for (int row = first; row < n; ++row) accr[row] += xr[row];
    }

    return cramFinish(x, iden, accr);
}

/// The SAME solve with the four poles on four lanes of one warp (WP21-D).
///
/// `pole` is the lane's index inside its group and IS the pole it solves; `base`
/// is the group's first lane in the warp and `mask` its four lanes.  Every lane
/// rebuilds the pole-invariant split, solves its own pole, and then the group
/// re-forms the two things the serial loop did ACROSS poles:
///
///   * the STATUS, which is the FIRST failing pole's -- __ffs over the group's
///     ballot -- so a pole-3 non-convergence behind a pole-0 zero diagonal still
///     declines with the message the serial kernel prints, and the sweep count
///     still stops at that pole;
///   * the ACCUMULATION, `accr[row] += x_pole[row]` walked p = 0, 1, 2, 3 with
///     __shfl_sync supplying pole p's value.  That is the serial loop nest with
///     ONE substitution, so it is the same sequence of double additions and not
///     merely an equivalent one -- which is what makes this arm B0, not N1.
__device__ unsigned int cramSolveNodePole4(const DevCtx& x, double* iden,
                                           const double* cond4, double sumflux,
                                           double dt, unsigned long long* gs_iters,
                                           int pole, unsigned int mask, int base) {
    const int n     = x.niso;
    const int first = x.first;

    double        cval[kMaxNnz];
    unsigned char ccol[kMaxNnz];
    int           cend[kNiso];
    double        mdiag[kNiso];
    cramBuildSplit(x, cond4, sumflux, dt, cval, ccol, cend, mdiag);

    double             xr[kNiso], xi[kNiso];
    int                iters = 0;
    const unsigned int st =
        cramPole(x, iden, cval, ccol, cend, mdiag, pole, xr, xi, &iters);

    // Every lane of the group reaches this point -- cramPole returns, it does
    // not exit -- so the whole `mask` is converged here and at every shuffle
    // below.  A lane that failed still participates; only its VALUES are unused.
    const unsigned int failed      = __ballot_sync(mask, st != kOk) & mask;
    unsigned int       node_status = kOk;
    int                last_run    = kPoleCount - 1;
    if (failed != 0u) {
        const int lane = __ffs(static_cast<int>(failed)) - 1;
        node_status    = __shfl_sync(mask, st, lane);
        last_run       = lane - base;
    }

    unsigned long long swept = 0;
    for (int p = 0; p < kPoleCount; ++p) {
        const int          it = __shfl_sync(mask, iters, base + p);
        const unsigned int sp = __shfl_sync(mask, st, base + p);
        if (p < last_run || (p == last_run && sp != kZeroDiag))
            swept += static_cast<unsigned long long>(it);
    }
    *gs_iters += swept;
    // `node_status` came off a ballot, so it is uniform over the group: all four
    // lanes take this return or none does, and no shuffle below is left orphaned.
    if (node_status != kOk) return node_status;

    // THE POLE-SUM ORDER.  This is the serial driver's
    // `for (pole) for (row) accr[row] += xr[row]` with pole p's `xr[row]` fetched
    // from the lane that owns pole p.  Same nesting, same direction, same four
    // addends in the same order into the same zero-initialised accumulator.
    double accr[kNiso];
    for (int row = 0; row < n; ++row) accr[row] = 0.0;
    for (int p = 0; p < kPoleCount; ++p) {
        for (int row = first; row < n; ++row)
            accr[row] += __shfl_sync(mask, xr[row], base + p);
    }

    return cramFinish(x, iden, accr);
}

/// ComputeXeEquilibrium + ApplyXeEquilibrium, transcribed.
__device__ inline void applyXeEquilibrium(const DevCtx& x, double* iden,
                                          const double* cond4, double sumflux) {
    constexpr double lambdaI     = 2.930607e-05;
    constexpr double lambdaXe    = 2.106574e-05;
    constexpr double lambdaXem   = 7.555561e-04;
    constexpr double brItoXe135m = 1.650900e-01;

    double fissSourceI = 0.0, fissSourceXe = 0.0;
    for (int j = x.ac_first; j <= x.ac_last; ++j) {
        const double xsff  = cond4[j * 4 + kCondXSFF];
        const double fRate = iden[j] * xsff * sumflux;
        fissSourceI += fRate * x.dep_trans[x.i135 * x.niso + j];
        fissSourceXe += fRate * x.dep_trans[x.xe135 * x.niso + j];
    }

    const double sigaXe = cond4[x.xe135 * 4 + kCondXSAF] * sumflux;
    const double Ieq    = fissSourceI / lambdaI;
    const double Xeeq   = (lambdaI * Ieq + fissSourceXe) / (lambdaXe + sigaXe);

    iden[x.i135]   = Ieq;
    iden[x.xe135]  = Xeeq;
    iden[x.xe135m] = brItoXe135m * lambdaI * Ieq / lambdaXem;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// The node body's two halves are factored so the serial and the pole-parallel
// kernel run the SAME prologue and the SAME epilogue.  An A/B whose two arms
// differ in the condensation as well as in the solve measures two things at once.

/// The part of DepleteNode that runs before solveBatemanCRAM: the 2-group flux
/// normalisation, the four-slot condensation, and the inventory load.
__device__ inline void predictorPrologue(const DevCtx& x, int l, double* cond4,
                                         double* iden, double* sumflux_out) {
    const int    ng   = x.ng;
    const size_t nxyz = static_cast<size_t>(x.nxyz);

    double abs_flux[2];
    double raw_sumflux = 0.0;
    for (int ig = 0; ig < ng; ++ig) {
        abs_flux[ig] = x.flux[static_cast<size_t>(l) * ng + ig] * x.norm_factor;
        raw_sumflux += abs_flux[ig];
    }
    const double invflux = (raw_sumflux > 0.0) ? 1.0 / raw_sumflux : 0.0;

    const double f0 = abs_flux[0] * invflux;
    const double f1 = abs_flux[1] * invflux;
    for (int iso = 0; iso < x.niso; ++iso) {
        const size_t base0 = (static_cast<size_t>(iso) * 2) * nxyz + l;
        const size_t base1 = (static_cast<size_t>(iso) * 2 + 1) * nxyz + l;
        for (int s = 0; s < 4; ++s)
            cond4[iso * 4 + s] = x.mic[s][base0] * f0 + x.mic[s][base1] * f1;
    }

    double s = 0.0;
    for (int ig = 0; ig < ng; ++ig) s += abs_flux[ig];
    *sumflux_out = s * 1.0e-24; // FluxScale

    for (int i = 0; i < x.niso; ++i)
        iden[i] = x.iden_in[static_cast<size_t>(i) * nxyz + l];
}

/// DepleteNode's tail: the Xe equilibrium overwrite, the publish, the counters.
///
/// THE STORE IS ALREADY NODE-INNERMOST, AND THAT IS THE FINDING.
/// `iden_out[iso * nxyz + l]` puts the node index in the fastest-varying
/// position, so a warp writing one isotope covers 32 consecutive doubles -- 256
/// bytes, which IS the 8-sectors-per-request floor for fp64.  The 8.7
/// sectors/request ncu reports for block 39 is therefore NOT this store: it is
/// the LOCAL-memory traffic of `cval`/`ccol`, whose `fill` cursor diverges
/// across a warp because the host's `value == 0.0` compression is per node.
/// Grouping four lanes per node makes those cursors agree inside a group -- 8
/// distinct cursors per warp instead of 32 -- which is where WP21-D's store-side
/// win comes from.  There is no SoA transpose to do, and none is done; nothing
/// downstream (WP15.1's FillCramMicDevice, the XSSet host readers, the D2H of
/// rows [first, niso) as one contiguous copy) sees a layout change at all.
__device__ inline void predictorEpilogue(const DevCtx& x, int l, double* iden,
                                         const double* cond4, double sumflux,
                                         unsigned int status,
                                         unsigned long long gs) {
    const size_t nxyz = static_cast<size_t>(x.nxyz);

    if (status == kOk && !x.xe_transient)
        applyXeEquilibrium(x, iden, cond4, sumflux);

    for (int i = x.first; i < x.niso; ++i)
        x.iden_out[static_cast<size_t>(i) * nxyz + l] = iden[i];

    if (status != kOk) atomicOr(&x.stats[0], static_cast<unsigned long long>(status));
    atomicAdd(&x.stats[1], gs);
    atomicAdd(&x.stats[2], static_cast<unsigned long long>(kPoleCount));
}

/// XSSet::Deplete's per-node body (DepleteNode, ngrp == 2 branch), serial arm.
__global__ void kPredictor(DevCtx x) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= x.nxyz) return;

    double cond4[kNiso * 4];
    double iden[kNiso];
    double sumflux = 0.0;
    predictorPrologue(x, l, cond4, iden, &sumflux);

    unsigned long long gs     = 0;
    const unsigned int status = cramSolveNode(x, iden, cond4, sumflux, x.dt, &gs);

    predictorEpilogue(x, l, iden, cond4, sumflux, status, gs);
}

/// The same body with kLanesPerNode lanes per node, lane == pole (WP21-D).
__global__ void kPredictorP4(DevCtx x) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int l    = gtid / kLanesPerNode;
    const int pole = gtid % kLanesPerNode;
    if (l >= x.nxyz) return;

    // blockDim.x is one of 32/64/128/256 and kLanesPerNode divides 32, so a
    // node's group never straddles a warp boundary and `mask` is always four
    // lanes of ONE warp -- the precondition every __shfl_sync downstream needs.
    // A tail group is out of range as a whole (the group IS the node), so the
    // mask never names a lane that already returned.
    const int          base = (threadIdx.x & 31) & ~(kLanesPerNode - 1);
    const unsigned int mask = ((1u << kLanesPerNode) - 1u) << base;

    double cond4[kNiso * 4];
    double iden[kNiso];
    double sumflux = 0.0;
    predictorPrologue(x, l, cond4, iden, &sumflux);

    unsigned long long gs = 0;
    const unsigned int status =
        cramSolveNodePole4(x, iden, cond4, sumflux, x.dt, &gs, pole, mask, base);

    if (pole == 0) predictorEpilogue(x, l, iden, cond4, sumflux, status, gs);
}

/// The part of CorrectorStep that runs before solveBatemanCRAM: the BOS/EOS
/// flux and kappa-fission average, the burnup integrand, the inventory load and
/// the four-slot condensation on averaged micro XS.
__device__ inline void correctorPrologue(const DevCtx& x, int l, double* cond4,
                                         double* iden, double* sumflux_out,
                                         double* burn_out) {
    const int    ng   = x.ng;
    const size_t nxyz = static_cast<size_t>(x.nxyz);
    const size_t node = static_cast<size_t>(l);

    double burn = 0.0;
    double corrected_flux[2];
    for (int ig = 0; ig < ng; ++ig)
        corrected_flux[ig] =
            x.density_average
                ? x.flux[node * ng + ig] * x.eos_norm
                : 0.5 * (x.flux_bos[node * ng + ig] * x.bos_norm +
                         x.flux[node * ng + ig] * x.eos_norm);

    for (int ig = 0; ig < ng; ++ig) {
        const double sigma_corrected =
            x.density_average
                ? x.xskf_eos[static_cast<size_t>(ig) * nxyz + node]
                : 0.5 * (x.xskf_bos[static_cast<size_t>(ig) * nxyz + node] +
                         x.xskf_eos[static_cast<size_t>(ig) * nxyz + node]);
        burn += sigma_corrected * corrected_flux[ig] * x.vol[node] * x.dt;
    }
    *burn_out = burn;

    for (int i = 0; i < x.niso; ++i)
        iden[i] = x.iden_in[static_cast<size_t>(i) * nxyz + node];

    double raw_sumflux = 0.0;
    for (int ig = 0; ig < ng; ++ig) raw_sumflux += corrected_flux[ig];
    const double invflux = (raw_sumflux > 0.0) ? 1.0 / raw_sumflux : 0.0;

    double sfl = 0.0;
    for (int ig = 0; ig < ng; ++ig) sfl += corrected_flux[ig];
    // The Xe equilibrium overwrite uses this same value; the host's
    // `xe_sumflux` was never a second quantity, only a second name.
    *sumflux_out = sfl * 1.0e-24; // FluxScale

    for (int iso = 0; iso < x.niso; ++iso) {
        for (int s = 0; s < 4; ++s) {
            double sum = 0.0;
            for (int ig = 0; ig < ng; ++ig) {
                const size_t off =
                    (static_cast<size_t>(iso) * ng + static_cast<size_t>(ig)) * nxyz + node;
                const double sigma_corrected =
                    x.density_average ? x.mic[s][off]
                                      : 0.5 * (x.mic_bos[s][off] + x.mic[s][off]);
                sum += sigma_corrected * corrected_flux[ig];
            }
            cond4[iso * 4 + s] = sum * invflux;
        }
    }
}

/// CorrectorStep's tail: the Xe equilibrium overwrite, the Eq. (6.20) density
/// average, the optional post-average Xe fix, the burnup key, the counters.
/// The store is `iden_out[iso * nxyz + node]` -- see predictorEpilogue for why
/// that is already the coalesced layout and why nothing is transposed.
__device__ inline void correctorEpilogue(const DevCtx& x, int l, double* iden,
                                         const double* cond4, double sumflux,
                                         double burn, unsigned int status,
                                         unsigned long long gs) {
    const size_t nxyz = static_cast<size_t>(x.nxyz);
    const size_t node = static_cast<size_t>(l);

    if (status == kOk && !x.xe_transient)
        applyXeEquilibrium(x, iden, cond4, sumflux);

    // N^C published, or the Eq. (6.20) average against the predictor inventory.
    for (int i = x.first; i < x.niso; ++i) {
        const double n_corr = iden[i];
        const double out =
            x.density_average
                ? 0.5 * (x.iden_pred[static_cast<size_t>(i) * nxyz + node] + n_corr)
                : n_corr;
        x.iden_out[static_cast<size_t>(i) * nxyz + node] = out;
        iden[i]                                          = out;
    }

    if (x.xe_equilibrium_fix && x.density_average && !x.xe_transient &&
        status == kOk) {
        applyXeEquilibrium(x, iden, cond4, sumflux);
        x.iden_out[static_cast<size_t>(x.i135) * nxyz + node]   = iden[x.i135];
        x.iden_out[static_cast<size_t>(x.xe135) * nxyz + node]  = iden[x.xe135];
        x.iden_out[static_cast<size_t>(x.xe135m) * nxyz + node] = iden[x.xe135m];
    }

    int burn_key = x.burn_bos[node];
    if (burn >= 1.0e-10) {
        const double burn_key_increment = burn / x.dfac[node] * 1000.0;
        // The host's `_burn[l] += static_cast<int>(inc + 0.5)`.  A non-finite
        // increment is UB in that cast on both sides, so it is caught here and
        // the whole call fails open rather than publishing a burnup key nobody
        // can attribute.  See the NaN-mask note in the Task 16 doc.
        if (!isfinite(burn_key_increment))
            status |= kNonFinite;
        else
            burn_key += static_cast<int>(burn_key_increment + 0.5);
    }
    x.burn_out[node] = burn_key;

    if (status != kOk) atomicOr(&x.stats[0], static_cast<unsigned long long>(status));
    atomicAdd(&x.stats[1], gs);
    atomicAdd(&x.stats[2], static_cast<unsigned long long>(kPoleCount));
}

/// XSSet::CorrectorStep's per-node body, pcSubsteps == 1, serial arm.
__global__ void kCorrector(DevCtx x) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= x.nxyz) return;

    double cond4[kNiso * 4];
    double iden[kNiso];
    double sumflux = 0.0;
    double burn    = 0.0;
    correctorPrologue(x, l, cond4, iden, &sumflux, &burn);

    unsigned long long gs     = 0;
    const unsigned int status = cramSolveNode(x, iden, cond4, sumflux, x.dt, &gs);

    correctorEpilogue(x, l, iden, cond4, sumflux, burn, status, gs);
}

/// The same body with kLanesPerNode lanes per node, lane == pole (WP21-D).
__global__ void kCorrectorP4(DevCtx x) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int l    = gtid / kLanesPerNode;
    const int pole = gtid % kLanesPerNode;
    if (l >= x.nxyz) return;

    const int          base = (threadIdx.x & 31) & ~(kLanesPerNode - 1);
    const unsigned int mask = ((1u << kLanesPerNode) - 1u) << base;

    double cond4[kNiso * 4];
    double iden[kNiso];
    double sumflux = 0.0;
    double burn    = 0.0;
    correctorPrologue(x, l, cond4, iden, &sumflux, &burn);

    unsigned long long gs = 0;
    const unsigned int status =
        cramSolveNodePole4(x, iden, cond4, sumflux, x.dt, &gs, pole, mask, base);

    if (pole == 0) correctorEpilogue(x, l, iden, cond4, sumflux, burn, status, gs);
}

} // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct CramBackend::Impl {
    bool        enabled = false;
    bool        failed  = false;
    std::string status_text;
    int         device = -1;
    int         block  = 64;

    /// WP21-D.  kPole4 is the DEFAULT and is B0 against kSerial; kSerial is kept
    /// so the A/B has a same-binary control arm and so a regression can be
    /// bisected to the mapping rather than to the build.
    Variant     variant      = Variant::kPole4;
    std::string variant_text = "pole4";
    /// RASBERY_GPU_CRAM_TIMING: bracket the KERNEL with a second event pair.
    bool        timing = false;

    cudaStream_t stream = nullptr;
    cudaEvent_t  ev_start = nullptr;
    cudaEvent_t  ev_stop  = nullptr;
    cudaEvent_t  ev_k0    = nullptr; ///< kernel only, under `timing`
    cudaEvent_t  ev_k1    = nullptr;

    // Shape this instance is sized for; a change re-allocates.
    int niso = 0, nxyz = 0, ng = 0, first = 0, nnz = 0;

    // node-invariant
    double*        d_dep_decay = nullptr;
    double*        d_dep_trans = nullptr;
    int*           d_row_ptr   = nullptr;
    unsigned char* d_col_idx   = nullptr;
    double*        d_dfac      = nullptr;
    double*        d_vol       = nullptr;
    unsigned long long lib_generation = 0;
    bool               lib_uploaded   = false;

    // per-step
    double* d_flux      = nullptr;
    double* d_flux_bos  = nullptr;
    double* d_xskf_bos  = nullptr;
    double* d_xskf_eos  = nullptr;
    double* d_mic[4]    = {nullptr, nullptr, nullptr, nullptr};
    double* d_mic_bos[4] = {nullptr, nullptr, nullptr, nullptr};
    double* d_iden_in   = nullptr;
    double* d_iden_pred = nullptr;
    double* d_iden_out  = nullptr;
    int*    d_burn_bos  = nullptr;
    int*    d_burn_out  = nullptr;

    unsigned long long* d_stats = nullptr;
    unsigned long long* h_stats = nullptr; ///< pinned, 3

    /// Residency key of d_mic, mirroring XsReconBackend's micx_generation
    /// contract.  0 means "nothing resident".
    unsigned long long micx_resident = 0;
    /// The predictor call that filled d_mic_bos.  A corrector must present it.
    unsigned long long bos_token = 0;
    bool               bos_valid = false;

    unsigned long long n_predictor  = 0;
    unsigned long long n_corrector  = 0;
    unsigned long long n_nodes      = 0;
    unsigned long long n_gs_iters   = 0;
    unsigned long long n_gs_solves  = 0;
    unsigned long long n_micx_bytes = 0;
    /// WP15.1: bytes the four condensation slots came in DEVICE-TO-DEVICE.
    unsigned long long n_micx_d2d_bytes = 0;
    unsigned long long n_bos_reuse  = 0;
    double             wall_ms      = 0.0;
    /// WP21-D: kernel launches (predictor + corrector), and the cudaEvent time
    /// of the kernels ALONE.  `wall_ms` has always been the whole call --
    /// transfers, status drain and all -- which is not the number ncu prints.
    unsigned long long n_launches = 0;
    double             kernel_ms  = 0.0;

    ~Impl() { release(); }

    void release() {
        // WP19.  Same rule, same reason, same defect as CudaPprBackend.cu:
        // a deck's teardown frees device memory, unregisters pinned host
        // memory and destroys a stream on ITS OWN thread while fifteen
        // sibling lanes are running -- one of which may have a graph
        // capture open.  See GpuCaptureArbiter.h.
        rasbery::AllocWindow _alloc_window("cram.release");
        auto f = [](void* p) { if (p) cudaFree(p); };
        f(d_dep_decay); f(d_dep_trans); f(d_row_ptr); f(d_col_idx);
        f(d_dfac); f(d_vol);
        f(d_flux); f(d_flux_bos); f(d_xskf_bos); f(d_xskf_eos);
        for (int s = 0; s < 4; ++s) { f(d_mic[s]); f(d_mic_bos[s]); }
        f(d_iden_in); f(d_iden_pred); f(d_iden_out);
        f(d_burn_bos); f(d_burn_out); f(d_stats);
        if (h_stats) cudaFreeHost(h_stats);
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_stop) cudaEventDestroy(ev_stop);
        if (ev_k0) cudaEventDestroy(ev_k0);
        if (ev_k1) cudaEventDestroy(ev_k1);
        if (stream) cudaStreamDestroy(stream);
        d_dep_decay = d_dep_trans = nullptr;
        d_row_ptr = nullptr; d_col_idx = nullptr;
        d_dfac = d_vol = nullptr;
        d_flux = d_flux_bos = d_xskf_bos = d_xskf_eos = nullptr;
        for (int s = 0; s < 4; ++s) { d_mic[s] = nullptr; d_mic_bos[s] = nullptr; }
        d_iden_in = d_iden_pred = d_iden_out = nullptr;
        d_burn_bos = d_burn_out = nullptr;
        d_stats = nullptr; h_stats = nullptr;
        ev_start = ev_stop = nullptr; ev_k0 = ev_k1 = nullptr; stream = nullptr;
        lib_uploaded = false; micx_resident = 0; bos_valid = false;
    }

    bool fail(const char* what, cudaError_t rc) {
        failed      = true;
        status_text = std::string("disabled after CUDA failure in ") + what + ": " +
                      cudaGetErrorString(rc);
        std::fprintf(stderr,
                     "[RASBERY][CRAM_GPU][WARN] %s -- falling back to host depletion\n",
                     status_text.c_str());
        release();
        return false;
    }

    /// Refuse rather than run: a shape this kernel's compile-time arrays cannot
    /// hold, or a deck whose registry does not match the one the pattern was
    /// mined from.
    bool decline(const char* why) {
        status_text = std::string("declined: ") + why;
        return false;
    }

    /// WP21-D receipt: one launch, and -- only when RASBERY_GPU_CRAM_TIMING is
    /// set -- the microseconds it took.  Two extra event records per launch are
    /// two extra ordering points in a stream a production batch drives 102 times
    /// per case, so they are behind the flag rather than always on.  Called
    /// AFTER the stats drain, which is the sync that makes ev_k1 readable, and
    /// on the decline path too: a launch that produced a bad node still ran.
    void noteLaunch() {
        ++n_launches;
        if (!timing) return;
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, ev_k0, ev_k1) == cudaSuccess)
            kernel_ms += static_cast<double>(ms);
    }

    /// Threads for `nxyz` nodes under the selected mapping, and the grid to
    /// cover them.  The serial arm is one thread per node; the pole-parallel arm
    /// is kLanesPerNode.
    int gridFor(int nxyz_in) const {
        const long long threads =
            static_cast<long long>(nxyz_in) * lanesPerNode();
        return static_cast<int>((threads + block - 1) / block);
    }

    int lanesPerNode() const {
        return (variant == Variant::kPole4) ? kLanesPerNode : 1;
    }

    bool ensureShape(const cram::LibView& lib, int ng_in, int nxyz_in, int nnz_in) {
        if (niso == lib.niso && nxyz == nxyz_in && ng == ng_in &&
            first == lib.first && nnz == nnz_in && stream != nullptr)
            return true;
        release();
        niso  = lib.niso;
        nxyz  = nxyz_in;
        ng    = ng_in;
        first = lib.first;
        nnz   = nnz_in;

        // The ordinal the RECEIPT reports is the one these buffers land on, and
        // that is decided here, not in the constructor: --batch-mode selects a
        // slot's device on the worker thread, after the Driver (hence the XSSet,
        // hence this backend) already exists.
        cudaGetDevice(&device);

        // WP19.  Everything from here to the end of this function --
        // cudaStreamCreate, two cudaEventCreate, twenty cudaMalloc and a
        // cudaMallocHost -- is "potentially unsafe" during a capture, and
        // it runs on the FIRST depletion step of every deck.  With
        // RASBERY_GPU_CRAM=1 in the production arm this is the second
        // instance of the CudaPprBackend.cu gap, found by the same scan.
        rasbery::AllocWindow _alloc_window("cram.shape.standup");
        // WP19.2: cudaStreamNonBlocking.  This stream is never captured, but a
        // LEGACY-BLOCKING stream is joined by -- and joins -- the NULL stream
        // process-wide, which is precisely the coupling that invalidates a
        // sibling lane's in-flight capture.  Two of the tree's three blocking
        // streams were the PPR WHILE's own; this was the third, and leaving it
        // behind would leave the contract test a permanent exception.
        cudaError_t rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (rc != cudaSuccess) return fail("cudaStreamCreateWithFlags", rc);
        if ((rc = cudaEventCreate(&ev_start)) != cudaSuccess) return fail("cudaEventCreate", rc);
        if ((rc = cudaEventCreate(&ev_stop)) != cudaSuccess) return fail("cudaEventCreate", rc);
        if ((rc = cudaEventCreate(&ev_k0)) != cudaSuccess) return fail("cudaEventCreate", rc);
        if ((rc = cudaEventCreate(&ev_k1)) != cudaSuccess) return fail("cudaEventCreate", rc);

        const size_t nn   = static_cast<size_t>(nxyz);
        const size_t nng  = nn * ng;
        const size_t nis  = static_cast<size_t>(niso) * nn;
        const size_t nmic = static_cast<size_t>(niso) * ng * nn;

        struct Alloc { void** p; size_t bytes; const char* name; };
        std::vector<Alloc> allocs = {
            {(void**)&d_dep_decay, static_cast<size_t>(niso) * niso * sizeof(double), "dep_decay"},
            {(void**)&d_dep_trans, static_cast<size_t>(niso) * niso * sizeof(double), "dep_trans"},
            {(void**)&d_row_ptr,   static_cast<size_t>(niso - first + 1) * sizeof(int), "row_ptr"},
            {(void**)&d_col_idx,   static_cast<size_t>(nnz > 0 ? nnz : 1), "col_idx"},
            {(void**)&d_dfac,      nn * sizeof(double), "dfac"},
            {(void**)&d_vol,       nn * sizeof(double), "vol"},
            {(void**)&d_flux,      nng * sizeof(double), "flux"},
            {(void**)&d_flux_bos,  nng * sizeof(double), "flux_bos"},
            {(void**)&d_xskf_bos,  nng * sizeof(double), "xskf_bos"},
            {(void**)&d_xskf_eos,  nng * sizeof(double), "xskf_eos"},
            {(void**)&d_iden_in,   nis * sizeof(double), "iden_in"},
            {(void**)&d_iden_pred, nis * sizeof(double), "iden_pred"},
            {(void**)&d_iden_out,  nis * sizeof(double), "iden_out"},
            {(void**)&d_burn_bos,  nn * sizeof(int), "burn_bos"},
            {(void**)&d_burn_out,  nn * sizeof(int), "burn_out"},
            {(void**)&d_stats,     3 * sizeof(unsigned long long), "stats"},
        };
        for (int s = 0; s < 4; ++s) {
            allocs.push_back({(void**)&d_mic[s], nmic * sizeof(double), "mic"});
            allocs.push_back({(void**)&d_mic_bos[s], nmic * sizeof(double), "mic_bos"});
        }
        for (const Alloc& a : allocs) {
            rc = cudaMalloc(a.p, a.bytes);
            if (rc != cudaSuccess) return fail(a.name, rc);
        }
        rc = cudaMallocHost((void**)&h_stats, 3 * sizeof(unsigned long long));
        if (rc != cudaSuccess) return fail("cudaMallocHost(stats)", rc);
        return true;
    }

    /// WP13.1: `name` was already the site's own label for the failure
    /// message; it is now also its ledger leaf, so every one of the
    /// seventeen callers gets its own row without a second argument.
    bool h2d(void* d, const void* h, size_t bytes, const char* name) {
        const cudaError_t rc = rasbery::xfer::memcpyAsync(
            "CudaCramBackend.cu:h2d", name, d, h, bytes, cudaMemcpyHostToDevice, stream);
        if (rc != cudaSuccess) return fail(name, rc);
        return true;
    }

    /// WP15.1: the four condensation slots, from wherever they already are.
    ///
    /// kSlot SEMANTICS ARE UNTOUCHED -- it still indexes the caller's ELEVEN-slot
    /// array, and the four blocks this backend uploads are still exactly
    /// {XSAF, XSFF, XS2N, XS3N}.  What changes is the SOURCE of those four: the
    /// flat-XS backend's resident device block when the caller offered it (the
    /// caller having already checked that its generation matches), the host
    /// array otherwise.
    ///
    /// ALL FOUR OR NONE.  A mixed fill would put two epochs in one condensation,
    /// and the caller's own generation check cannot see that; `mic_device[0]`
    /// standing for all four is what makes the decision atomic here.
    ///
    /// THE EVENT IS NOT OPTIONAL WHEN THE D2D RUNS.  The two backends are on
    /// different streams, so without the wait this stream could read the block
    /// while the flat-XS kernel is still writing it -- a race that produces
    /// finite, plausible, wrong inventories.  No event offered, no D2D.
    ///
    /// WP20.1: THE SOURCE MAY NOW BE FLOAT AND THE DESTINATION MAY NOT.
    ///
    /// RASBERY_GPU_FP32 narrows the flat-XS micx/lmpx blocks to float, but
    /// CRAM stays FP64 on purpose (src/GpuFp32Arm.h: the partial-fraction sum
    /// alternates in sign over nuclide fields spanning ~20 decades and cancels
    /// catastrophically).  So when the offered block is four bytes wide the
    /// handover is a WIDENING KERNEL rather than a memcpy -- same stream, same
    /// event wait, same all-or-nothing rule.  A memcpy would have copied
    /// `nmic * sizeof(double)` bytes out of a `nmic * sizeof(float)` block:
    /// half of it the next slot's data, and the inventories finite, plausible
    /// and wrong.  `elem_bytes` is the caller's word for the width and the
    /// only thing this function is allowed to believe about it.
    bool fillMic(const double* const* host_mic, const void* const* dev_mic,
                 void* ready, size_t nmic, int elem_bytes, const char* who) {
        const bool d2d = dev_mic != nullptr && dev_mic[kSlot[0]] != nullptr &&
                         ready != nullptr;
        if (d2d) {
            const cudaError_t we =
                cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(ready), 0);
            if (we != cudaSuccess) return fail("wait micx event", we);
        }
        const bool narrow = d2d && elem_bytes == static_cast<int>(sizeof(float));
        for (int k = 0; k < 4; ++k) {
            const size_t bytes = nmic * sizeof(double);
            if (narrow) {
                const size_t moved = nmic * sizeof(float);
                const int    block = 256;
                const int    grid =
                    static_cast<int>((nmic + static_cast<size_t>(block) - 1) /
                                     static_cast<size_t>(block));
                kCramWidenMic<<<grid, block, 0, stream>>>(
                    d_mic[k], static_cast<const float*>(dev_mic[kSlot[k]]), nmic);
                const cudaError_t rc = cudaGetLastError();
                if (rc != cudaSuccess) return fail("D2D mic (widen)", rc);
                n_micx_d2d_bytes += moved;
            } else if (d2d) {
                const cudaError_t rc = rasbery::xfer::memcpyAsync(
                    "CudaCramBackend.cu:fillMic", "D2D mic", d_mic[k],
                    static_cast<const double*>(dev_mic[kSlot[k]]), bytes,
                    cudaMemcpyDeviceToDevice, stream);
                if (rc != cudaSuccess) return fail("D2D mic", rc);
                n_micx_d2d_bytes += bytes;
            } else {
                const std::string name = std::string("H2D mic (") + who + ")";
                if (!h2d(d_mic[k], host_mic[kSlot[k]], bytes, name.c_str())) return false;
                n_micx_bytes += bytes;
            }
        }
        return true;
    }

    /// Fill the DevCtx fields that are the same for both entry points.
    void fillCommon(DevCtx& x, const cram::LibView& lib, int ng_in, int nxyz_in) {
        x.niso     = lib.niso;
        x.first    = lib.first;
        x.ac_first = lib.ac_first;
        x.ac_last  = lib.ac_last;
        x.i135     = lib.i135;
        x.xe135    = lib.xe135;
        x.xe135m   = lib.xe135m;
        x.u234     = lib.u234;
        x.u235     = lib.u235;
        x.u238     = lib.u238;
        x.np237    = lib.np237;
        x.ng       = ng_in;
        x.nxyz     = nxyz_in;

        x.dep_decay = d_dep_decay;
        x.dep_trans = d_dep_trans;
        x.row_ptr   = d_row_ptr;
        x.col_idx   = d_col_idx;
        x.dfac      = d_dfac;
        x.vol       = d_vol;

        x.flux     = d_flux;
        x.flux_bos = d_flux_bos;
        x.xskf_bos = d_xskf_bos;
        x.xskf_eos = d_xskf_eos;
        for (int s = 0; s < 4; ++s) {
            x.mic[s]     = d_mic[s];
            x.mic_bos[s] = d_mic_bos[s];
        }
        x.iden_in   = d_iden_in;
        x.iden_pred = d_iden_pred;
        x.burn_bos  = d_burn_bos;
        x.iden_out  = d_iden_out;
        x.burn_out  = d_burn_out;
        x.stats     = d_stats;
    }

};

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

CramBackend::CramBackend() : _impl(new Impl) {
    _impl->enabled = truthy(std::getenv("RASBERY_GPU_CRAM"));
    if (!_impl->enabled) {
        _impl->status_text = "off (RASBERY_GPU_CRAM unset)";
        return;
    }
    int         count = 0;
    cudaError_t rc    = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess || count <= 0) {
        _impl->enabled     = false;
        _impl->status_text = "no CUDA device";
        return;
    }
    cudaGetDevice(&_impl->device);
    if (const char* b = std::getenv("RASBERY_GPU_CRAM_BLOCK")) {
        const int v = std::atoi(b);
        if (v == 32 || v == 64 || v == 128 || v == 256) _impl->block = v;
    }
    // WP21-D.  The DEFAULT is the pole-parallel mapping, which is B0 against the
    // serial one; `serial` selects the pre-WP21-D body for a same-binary A/B.
    // An unrecognised spelling keeps the default AND SAYS SO in the status --
    // silently running a different arithmetic than the one the operator typed is
    // the failure mode this knob exists to avoid.
    _impl->status_text = "on";
    if (const char* p = std::getenv("RASBERY_GPU_CRAM_PARALLEL")) {
        const std::string want(p);
        if (want == "serial") {
            _impl->variant      = Variant::kSerial;
            _impl->variant_text = "serial";
        } else if (want == "pole" || want == "pole4") {
            _impl->variant      = Variant::kPole4;
            _impl->variant_text = "pole4";
        } else {
            _impl->status_text =
                "on (unknown RASBERY_GPU_CRAM_PARALLEL=" + want + ", using pole4)";
        }
    }
    _impl->timing = truthy(std::getenv("RASBERY_GPU_CRAM_TIMING"));
}

CramBackend::~CramBackend() = default;

bool CramBackend::available() const { return _impl->enabled && !_impl->failed; }
const std::string& CramBackend::status() const { return _impl->status_text; }
unsigned long long CramBackend::predictorCalls() const { return _impl->n_predictor; }
unsigned long long CramBackend::correctorCalls() const { return _impl->n_corrector; }
unsigned long long CramBackend::nodesSolved() const { return _impl->n_nodes; }
unsigned long long CramBackend::gsIterations() const { return _impl->n_gs_iters; }
unsigned long long CramBackend::gsSolves() const { return _impl->n_gs_solves; }
double             CramBackend::wallMs() const { return _impl->wall_ms; }
int                CramBackend::deviceOrdinal() const { return _impl->device; }
unsigned long long CramBackend::micxH2dBytes() const { return _impl->n_micx_bytes; }

unsigned long long CramBackend::micxD2dBytes() const { return _impl->n_micx_d2d_bytes; }
unsigned long long CramBackend::bosReuses() const { return _impl->n_bos_reuse; }
const std::string& CramBackend::kernelVariant() const { return _impl->variant_text; }
int                CramBackend::lanesPerNode() const { return _impl->lanesPerNode(); }
unsigned long long CramBackend::launches() const { return _impl->n_launches; }

double CramBackend::launchUsMean() const {
    // -1 is "never measured", which reads very differently from "measured 0".
    if (!_impl->timing || _impl->n_launches == 0) return -1.0;
    return _impl->kernel_ms * 1000.0 / static_cast<double>(_impl->n_launches);
}

namespace {

/// Mine the node-independent union pattern: nonzeros(depDecay) U the diagonal U
/// nonzeros(depTrans) U the two (n,2n) specials, restricted to rows and columns
/// >= first, off-diagonals only, columns ascending within a row.  Exactly the
/// set the host's per-node scan can ever produce; the kernel re-runs the host's
/// `value == 0.0` test so a node whose value vanishes is compressed the same way.
bool minePattern(const cram::LibView& lib, std::vector<int>& row_ptr,
                 std::vector<unsigned char>& col_idx) {
    const int n     = lib.niso;
    const int first = lib.first;
    row_ptr.assign(static_cast<size_t>(n - first + 1), 0);
    col_idx.clear();

    for (int r = first; r < n; ++r) {
        for (int c = first; c < n; ++c) {
            if (c == r) continue;
            const bool decay = lib.dep_decay[static_cast<size_t>(r) * n + c] != 0.0;
            const bool trans = lib.dep_trans[static_cast<size_t>(r) * n + c] != 0.0;
            const bool extra = (r == lib.u234 && c == lib.u235) ||
                               (r == lib.np237 && c == lib.u238);
            if (!decay && !trans && !extra) continue;
            if (c > 255) return false;
            col_idx.push_back(static_cast<unsigned char>(c));
        }
        row_ptr[static_cast<size_t>(r - first + 1)] = static_cast<int>(col_idx.size());
    }
    return true;
}

} // namespace

bool CramBackend::predictor(const cram::LibView& lib, const cram::PredictorView& v,
                            unsigned long long* token_out) {
    Impl& s = *_impl;
    // The device BOS snapshot belongs to the LAST SUCCESSFUL predictor.  Retiring
    // it here, before anything can go wrong, is what stops a predictor that
    // declines half way from leaving the previous statepoint's token still valid
    // -- which would let this statepoint's corrector match it and read rates that
    // are one statepoint stale.  (XSSet::DepleteGpu clears its own copy of the
    // token for the same reason; either guard alone would do, and both together
    // mean neither side has to trust the other.)
    s.bos_valid = false;
    if (!s.enabled || s.failed) return false;
    if (v.ng != 2) return s.decline("ng != 2 (the condensation body is 2-group)");
    if (lib.niso != kNiso) return s.decline("niso != 39 (pattern shape is compile-time)");
    if (lib.dep_decay == nullptr || lib.dep_trans == nullptr)
        return s.decline("no depletion data");

    std::vector<int>           row_ptr;
    std::vector<unsigned char> col_idx;
    if (!minePattern(lib, row_ptr, col_idx)) return s.decline("isotope index > 255");
    if (static_cast<int>(col_idx.size()) > kMaxNnz)
        return s.decline("union pattern exceeds kMaxNnz");

    if (!s.ensureShape(lib, v.ng, v.nxyz, static_cast<int>(col_idx.size()))) return false;

    const size_t nn   = static_cast<size_t>(v.nxyz);
    const size_t nng  = nn * v.ng;
    const size_t nis  = static_cast<size_t>(lib.niso) * nn;
    const size_t nmic = static_cast<size_t>(lib.niso) * v.ng * nn;

    if (!s.lib_uploaded || s.lib_generation != lib.generation) {
        if (!s.h2d(s.d_dep_decay, lib.dep_decay,
                   static_cast<size_t>(lib.niso) * lib.niso * sizeof(double), "H2D dep_decay"))
            return false;
        if (!s.h2d(s.d_dep_trans, lib.dep_trans,
                   static_cast<size_t>(lib.niso) * lib.niso * sizeof(double), "H2D dep_trans"))
            return false;
        if (!s.h2d(s.d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), "H2D row_ptr"))
            return false;
        if (!col_idx.empty() &&
            !s.h2d(s.d_col_idx, col_idx.data(), col_idx.size(), "H2D col_idx"))
            return false;
        if (!s.h2d(s.d_dfac, lib.dfac, nn * sizeof(double), "H2D dfac")) return false;
        if (!s.h2d(s.d_vol, lib.vol, nn * sizeof(double), "H2D vol")) return false;
        s.lib_uploaded   = true;
        s.lib_generation = lib.generation;
    }

    if (!s.h2d(s.d_flux, v.phif, nng * sizeof(double), "H2D phif")) return false;
    if (!s.h2d(s.d_iden_in, v.iden, nis * sizeof(double), "H2D iden")) return false;

    if (s.micx_resident != v.micx_generation) {
        if (!s.fillMic(v.mic, v.mic_device, v.mic_device_ready, nmic,
                       v.mic_device_elem_bytes, "predictor"))
            return false;
        s.micx_resident = v.micx_generation;
    }

    // The BOS snapshot the matching corrector will read: a device-to-device copy
    // of the block that is already here, not a second 21 MB trip over the bus.
    for (int k = 0; k < 4; ++k) {
        const cudaError_t rc = rasbery::xfer::memcpyAsync(
            "CudaCramBackend.cu:predictor", "mic_bos (D2D)", s.d_mic_bos[k], s.d_mic[k],
            nmic * sizeof(double), cudaMemcpyDeviceToDevice, s.stream);
        if (rc != cudaSuccess) return s.fail("D2D mic_bos", rc);
    }

    DevCtx x{};
    s.fillCommon(x, lib, v.ng, v.nxyz);
    x.dt           = v.dt;
    x.xe_transient = v.xe_transient;
    x.norm_factor  = v.norm_factor;

    cudaError_t rc = cudaMemsetAsync(s.d_stats, 0, 3 * sizeof(unsigned long long), s.stream);
    if (rc != cudaSuccess) return s.fail("memset stats", rc);

    const int blocks = s.gridFor(v.nxyz);
    cudaEventRecord(s.ev_start, s.stream);
    if (s.timing) cudaEventRecord(s.ev_k0, s.stream);
    if (s.variant == Variant::kPole4)
        kPredictorP4<<<blocks, s.block, 0, s.stream>>>(x);
    else
        kPredictor<<<blocks, s.block, 0, s.stream>>>(x);
    if (s.timing) cudaEventRecord(s.ev_k1, s.stream);
    if ((rc = cudaGetLastError()) != cudaSuccess) return s.fail("kPredictor launch", rc);

    rc = rasbery::xfer::memcpyAsync("CudaCramBackend.cu:predictor", "stats", s.h_stats,
                                    s.d_stats, 3 * sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost, s.stream);
    if (rc != cudaSuccess) return s.fail("D2H stats", rc);
    if ((rc = rasbery::xfer::streamSync("CudaCramBackend.cu:predictor", "stats drain",
                                        s.stream)) != cudaSuccess)
        return s.fail("predictor sync", rc);
    s.noteLaunch();

    // NOTHING has touched a host array yet.  A node that hit either of milk.h's
    // throw conditions, or produced a non-finite density, makes the entire call
    // decline; the host loop then runs from the untouched inventory and throws
    // the same std::runtime_error at the same node it always would.
    if (s.h_stats[0] != 0) {
        s.status_text = "declined: node status " + std::to_string(s.h_stats[0]) +
                        " (zero diagonal / no convergence / non-finite)";
        return false;
    }

    // Rows [first, niso) are contiguous in the [iso * nxyz + l] layout, so the
    // exact set DepleteNode writes is one copy.
    const size_t off  = static_cast<size_t>(lib.first) * nn;
    const size_t rows = static_cast<size_t>(lib.niso - lib.first) * nn;
    rc = rasbery::xfer::memcpyAsync("CudaCramBackend.cu:predictor", "iden out",
                                    v.iden + off, s.d_iden_out + off,
                                    rows * sizeof(double), cudaMemcpyDeviceToHost,
                                    s.stream);
    if (rc != cudaSuccess) return s.fail("D2H iden", rc);

    cudaEventRecord(s.ev_stop, s.stream);
    if ((rc = rasbery::xfer::streamSync("CudaCramBackend.cu:predictor", "final drain",
                                        s.stream)) != cudaSuccess)
        return s.fail("predictor final sync", rc);

    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, s.ev_start, s.ev_stop) == cudaSuccess)
        s.wall_ms += static_cast<double>(ms);

    ++s.n_predictor;
    s.n_nodes += static_cast<unsigned long long>(v.nxyz);
    s.n_gs_iters += s.h_stats[1];
    s.n_gs_solves += s.h_stats[2];
    s.bos_valid = true;
    ++s.bos_token;
    if (token_out) *token_out = s.bos_token;
    return true;
}

bool CramBackend::corrector(const cram::LibView& lib, const cram::CorrectorView& v) {
    Impl& s = *_impl;
    if (!s.enabled || s.failed) return false;
    if (v.ng != 2) return s.decline("ng != 2 (the condensation body is 2-group)");
    if (lib.niso != kNiso) return s.decline("niso != 39 (pattern shape is compile-time)");
    if (lib.dep_decay == nullptr || lib.dep_trans == nullptr)
        return s.decline("no depletion data");
    // The BOS micro-XS block on the device belongs to a specific predictor call.
    // A corrector whose predictor fell back to the host would read some other
    // statepoint's rates and produce a plausible, wrong inventory.
    if (!s.bos_valid || v.bos_token != s.bos_token)
        return s.decline("BOS snapshot does not match this statepoint's predictor");
    if (s.niso != lib.niso || s.nxyz != v.nxyz || s.ng != v.ng)
        return s.decline("shape changed between predictor and corrector");

    const size_t nn   = static_cast<size_t>(v.nxyz);
    const size_t nng  = nn * v.ng;
    const size_t nis  = static_cast<size_t>(lib.niso) * nn;
    const size_t nmic = static_cast<size_t>(lib.niso) * v.ng * nn;

    if (!s.h2d(s.d_flux, v.flux_eos, nng * sizeof(double), "H2D flux_eos")) return false;
    if (!s.h2d(s.d_flux_bos, v.flux_bos, nng * sizeof(double), "H2D flux_bos")) return false;
    if (!s.h2d(s.d_xskf_eos, v.xskf_eos, nng * sizeof(double), "H2D xskf_eos")) return false;
    if (!s.h2d(s.d_xskf_bos, v.xskf_bos, nng * sizeof(double), "H2D xskf_bos")) return false;
    if (!s.h2d(s.d_iden_in, v.iden_bos, nis * sizeof(double), "H2D iden_bos")) return false;
    if (!s.h2d(s.d_burn_bos, v.burn_bos, nn * sizeof(int), "H2D burn_bos")) return false;
    // The predictor inventory N^P is only read by the Eq. (6.20) average.
    if (v.density_average &&
        !s.h2d(s.d_iden_pred, v.iden, nis * sizeof(double), "H2D iden_pred"))
        return false;

    if (s.micx_resident != v.micx_generation) {
        if (!s.fillMic(v.mic, v.mic_device, v.mic_device_ready, nmic,
                       v.mic_device_elem_bytes, "corrector"))
            return false;
        s.micx_resident = v.micx_generation;
    }
    ++s.n_bos_reuse;

    DevCtx x{};
    s.fillCommon(x, lib, v.ng, v.nxyz);
    x.dt                 = v.dt;
    x.xe_transient       = v.xe_transient;
    x.bos_norm           = v.bos_norm;
    x.eos_norm           = v.eos_norm;
    x.density_average    = v.density_average;
    x.xe_equilibrium_fix = v.xe_equilibrium_fix;

    cudaError_t rc = cudaMemsetAsync(s.d_stats, 0, 3 * sizeof(unsigned long long), s.stream);
    if (rc != cudaSuccess) return s.fail("memset stats", rc);

    const int blocks = s.gridFor(v.nxyz);
    cudaEventRecord(s.ev_start, s.stream);
    if (s.timing) cudaEventRecord(s.ev_k0, s.stream);
    if (s.variant == Variant::kPole4)
        kCorrectorP4<<<blocks, s.block, 0, s.stream>>>(x);
    else
        kCorrector<<<blocks, s.block, 0, s.stream>>>(x);
    if (s.timing) cudaEventRecord(s.ev_k1, s.stream);
    if ((rc = cudaGetLastError()) != cudaSuccess) return s.fail("kCorrector launch", rc);

    rc = rasbery::xfer::memcpyAsync("CudaCramBackend.cu:corrector", "stats", s.h_stats,
                                    s.d_stats, 3 * sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost, s.stream);
    if (rc != cudaSuccess) return s.fail("D2H stats", rc);
    if ((rc = rasbery::xfer::streamSync("CudaCramBackend.cu:corrector", "stats drain",
                                        s.stream)) != cudaSuccess)
        return s.fail("corrector sync", rc);
    s.noteLaunch();

    if (s.h_stats[0] != 0) {
        s.status_text = "declined: node status " + std::to_string(s.h_stats[0]) +
                        " (zero diagonal / no convergence / non-finite)";
        return false;
    }

    const size_t off  = static_cast<size_t>(lib.first) * nn;
    const size_t rows = static_cast<size_t>(lib.niso - lib.first) * nn;
    rc = rasbery::xfer::memcpyAsync("CudaCramBackend.cu:corrector", "iden out",
                                    v.iden + off, s.d_iden_out + off,
                                    rows * sizeof(double), cudaMemcpyDeviceToHost,
                                    s.stream);
    if (rc != cudaSuccess) return s.fail("D2H iden", rc);
    rc = rasbery::xfer::memcpyAsync("CudaCramBackend.cu:corrector", "burn out", v.burn,
                                    s.d_burn_out, nn * sizeof(int),
                                    cudaMemcpyDeviceToHost, s.stream);
    if (rc != cudaSuccess) return s.fail("D2H burn", rc);

    cudaEventRecord(s.ev_stop, s.stream);
    if ((rc = rasbery::xfer::streamSync("CudaCramBackend.cu:corrector", "final drain",
                                        s.stream)) != cudaSuccess)
        return s.fail("corrector final sync", rc);

    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, s.ev_start, s.ev_stop) == cudaSuccess)
        s.wall_ms += static_cast<double>(ms);

    ++s.n_corrector;
    s.n_nodes += static_cast<unsigned long long>(v.nxyz);
    s.n_gs_iters += s.h_stats[1];
    s.n_gs_solves += s.h_stats[2];
    // One corrector consumes one predictor's snapshot.
    s.bos_valid = false;
    return true;
}

} // namespace rasbery
