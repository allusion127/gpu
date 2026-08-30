// GPU pin-power reconstruction: PPR::reset + PPR::drive on the device.
// Contract, cost ledger and gate class: CudaPprBackend.h.
//
// The arithmetic below is a line-for-line transcription of PPR.cpp's
// reset()/updateAxialLeakage()/updateSource()/updateFused()/updateCorner(),
// in the same statement order and with the same intermediate temporaries, so
// the only sources of difference against the host are the device libm's `exp`
// and the corner-sum reduction's chunking.  --fmad=false is set on this file in
// CMakeLists.txt for exactly the reason CudaXsReconBackend.cu has it: nvcc must
// not contract a*b+c where the host compiler did not.

#include "CudaPprBackend.h"
#include "GpuCaptureArbiter.h"
#include "PprReconstructionKernel.cuh"
#include "XferLedger.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rasbery {

// WP19.  GpuCaptureArbiter.h spells the capture-illegal error class as numbers
// because it must compile in the no-CUDA stub build.  This is the one TU that
// can check the numbers against the enum, so it does -- if a future toolkit
// renumbers the block, the build stops here instead of the retry quietly
// deciding that nothing is ever retryable.
static_assert(static_cast<int>(cudaErrorStreamCaptureUnsupported) == kCaptureErrFirst,
              "GpuCaptureArbiter.h kCaptureErrFirst no longer matches CUDA");
static_assert(static_cast<int>(cudaErrorStreamCaptureWrongThread) == kCaptureErrLast,
              "GpuCaptureArbiter.h kCaptureErrLast no longer matches CUDA");

namespace {

// pch.h constants, re-declared here so this translation unit does not drag the
// host headers (and their `using namespace std`) into nvcc.
constexpr int kXDIR = 0, kYDIR = 1, kZDIR = 2, kNDIRMAX = 3;
constexpr int kLEFT = 0, kRIGHT = 1, kLR = 2;
constexpr int kNW = 0, kNE = 1, kSW = 2, kSE = 3;
constexpr int kWEST = 0, kEAST = 1, kNORTH = 2, kSOUTH = 3, kNEWS = 4;

constexpr double kSq2  = 1.414213562373;
constexpr double kRsq2 = 0.707106781186;

// PPR.cpp anonymous namespace
constexpr int    kSourceSweepsPerIteration = 3;
constexpr double kCornerFluxTolerance      = 1.0E-5;

/// Geometry.h rasbery_det_chunks / rasbery_det_chunk_begin, verbatim.  The
/// corner-sum partition is a property of the reduction, not of the host, so it
/// is spelled the same way on both sides.
constexpr int kDetChunkTarget = 256;

int detChunks(int n) {
    if (n <= 1) return 1;
    return (n < kDetChunkTarget) ? n : kDetChunkTarget;
}

__host__ __device__ inline int detChunkBegin(int n, int nchunk, int c) {
    if (c >= nchunk) return n;
    const long long nn = n;
    return static_cast<int>((nn * c) / nchunk);
}

bool truthy(const char* v) {
    if (v == nullptr) return false;
    const std::string s(v);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
             s == "FALSE");
}

/// WP6 stage B.  The Picard loop's whole host-visible state, in device memory.
///
/// WHY `reigv` LIVES HERE and not in DevCtx-by-value.  The conditional WHILE
/// captures the body ONCE per (deck, shape) and replays it at every statepoint,
/// and a kernel node's parameters are baked at capture time.  A `reigv` passed
/// by value would therefore be the FIRST statepoint's eigenvalue forever --
/// finite, plausible, and wrong.  Every per-statepoint scalar the body reads is
/// in this block, which the host refreshes with one 64-byte H2D and the body
/// reads through a pointer that never moves.
struct PprLoopState {
    double reigv;      ///< 1/k_eff for THIS statepoint
    double prev[4];    ///< the previous round's fuel-only corner sums
    double cur[4];     ///< this round's
    int    iters;      ///< rounds actually executed == the host's `citer + 1`
    int    converged;  ///< the break test fired; every guarded kernel no-ops
    int    niter;      ///< the cap the caller passed
    int    pad;
};

/// Everything a kernel needs, by value.  Pointers are device pointers.
struct DevCtx {
    int ng;
    int nxyz;
    int nxy;
    int nsurf;
    int has_chif;
    int nchunk;

    /// Per-statepoint scalars and the loop's own state.  NEVER null once
    /// ensureShape has run.
    PprLoopState* loop;
    /// The corner-sum partials the fold kernel reads.  Also never null.
    double* partials;

    const double*        hmesh;
    const int*           lktosfc;
    const int*           neibrb;
    const unsigned char* is_fuel;

    const double* phif;
    const double* phis;
    const double* jnet;
    const double* xsdf;
    const double* xsrf;
    const double* xsnf;
    const double* xssm;
    const double* chif;
    const double* crdf;

    double* phic;
    double* p;
    double* a;
    double* c;
    double* q;
    double* l;
    double* bt;

    /// WP6 stage F, MASTER mode only.  The CPB solve is Jacobi on the device --
    /// the host's Gauss-Seidel writes `phic` in place while later nodes read
    /// it, which is a serial dependence over the whole mesh -- so it needs the
    /// NEXT iterate to write into, and a per-(node, group) relative-change
    /// scratch for the break test.  Allocated with everything else and left
    /// untouched by the SENM arm, whose corner update is already a pure
    /// scatter-free read of `c`.
    double* phic_next;
    double* mrel;
    /// 1 = the master body was enqueued.  Carried in the ctx (and therefore in
    /// the graph key) so a capture cannot be replayed under the other scheme.
    int mode_master;

    /// Read-only alias of `c`.  updateCorner and updateSource only read the
    /// fitting coefficients, and they read them from OTHER nodes -- 64 scattered
    /// loads per thread in updateCorner.  Reading them through a
    /// const __restrict__ pointer lets the compiler route those through the
    /// read-only data cache, which is what makes the 3x3 stencil's sharing
    /// between neighbouring threads pay.
    ///
    /// The `__restrict__` lives on the LOCAL alias each reader takes, not here.
    /// This member aliases `c`, and kUpdateFused writes `c`: a no-alias promise
    /// carried on the member would be a promise that kernel does not keep.  The
    /// two kernels that read it (kUpdateSource, jnetDirD) never write `c`, so
    /// their local restrict aliases are true, and the read-only cache path is
    /// reached exactly the same way.
    const double* c_ro;
};

// --- element accessors, mirroring the PPR.h macros -------------------------

__device__ inline double dXsdf(const DevCtx& x, int g, int lk) { return x.xsdf[g * x.nxyz + lk]; }
__device__ inline double dXsrf(const DevCtx& x, int g, int lk) { return x.xsrf[g * x.nxyz + lk]; }
__device__ inline double dXsnf(const DevCtx& x, int g, int lk) { return x.xsnf[g * x.nxyz + lk]; }
__device__ inline double dXssm(const DevCtx& x, int igs, int ige, int lk) {
    return x.xssm[(igs * x.ng + ige) * x.nxyz + lk];
}
__device__ inline double dChif(const DevCtx& x, int ig, int lk) {
    return x.has_chif ? x.chif[ig * x.nxyz + lk] : (ig == 0 ? 1.0 : 0.0);
}
__device__ inline double dHmesh(const DevCtx& x, int dir, int lk) {
    return x.hmesh[lk * kNDIRMAX + dir];
}
__device__ inline int dSfc(const DevCtx& x, int side, int dir, int lk) {
    return x.lktosfc[(lk * kNDIRMAX + dir) * kLR + side];
}
__device__ inline int dNeib(const DevCtx& x, int dir, int l) { return x.neibrb[l * kNEWS + dir]; }

__device__ inline double dCrdf(const DevCtx& x, int lk, int g) {
    return x.crdf[static_cast<size_t>(lk) * x.ng + g];
}

/// WP6 stage B.  THE ONE READ THAT MAKES AN ENQUEUED BODY A NO-OP.
///
/// Every kernel of the Picard body starts with this, and the `device_stream`
/// arm relies on nothing else: the fold kernel raises `converged` at the end of
/// round i, and every kernel of rounds i+1..niter-1 -- already enqueued, and
/// nobody is going to wait to find out -- returns before writing anything.  The
/// state is therefore frozen at round i, which is exactly where the host's
/// `break` left it.
///
/// It is a template parameter and not a runtime `if` because the reset() half
/// launches the SAME kUpdateSource, where there is no loop and no flag to read.
__device__ inline bool pprHalted(const DevCtx& x) { return x.loop->converged != 0; }

/// Upper-triangular slot of the 15-term expansion: 5i - i(i-1)/2 + j.
__device__ inline int triIdx(int i, int j) { return 5 * i - (i * (i - 1)) / 2 + j; }

__device__ inline double* pPtr(const DevCtx& x, int lk, int g) { return &x.p[(lk * 15 * x.ng) + g * 15]; }
__device__ inline double* aPtr(const DevCtx& x, int lk, int g) { return &x.a[(lk * 8 * x.ng) + g * 8]; }
__device__ inline double* cPtr(const DevCtx& x, int lk, int g) { return &x.c[(lk * 15 * x.ng) + g * 15]; }
__device__ inline const double* cRoPtr(const DevCtx& x, int lk, int g) {
    return &x.c_ro[(lk * 15 * x.ng) + g * 15];
}
__device__ inline double* qPtr(const DevCtx& x, int lk, int g) { return &x.q[(lk * 15 * x.ng) + g * 15]; }
__device__ inline double* lPtr(const DevCtx& x, int lk, int g) { return &x.l[(lk * 9 * x.ng) + g * 9]; }
__device__ inline double* phicPtr(const DevCtx& x, int lk, int g) {
    return &x.phic[(lk * 4 * x.ng) + g * 4];
}

__device__ inline double dPhis(const DevCtx& x, int dir, int lr, int lk, int g) {
    return x.phis[dSfc(x, lr, dir, lk) * x.ng + g];
}
__device__ inline double dJnet(const DevCtx& x, int dir, int lr, int lk, int g) {
    return x.jnet[dSfc(x, lr, dir, lk) * x.ng + g];
}

/// PPR::getPhis -- bounds-checked, returns 0 outside the mesh.
__device__ inline double getPhisD(const DevCtx& x, int lr, int dir, int lk, int g) {
    if (lk < 0 || lk >= x.nxyz) return 0.0;
    return x.phis[dSfc(x, lr, dir, lk) * x.ng + g];
}

/// PPR::getJoutRed -- the outward net current at surface (lr, dir) of node lk,
/// reduced by 2D/h, which is the current unit of MASTER MM Eq. 6.6/6.8.  Host
/// expression, host order, host guards (out of mesh and D <= 0 both give 0.0).
__device__ inline double getJoutRedD(const DevCtx& x, int lr, int dir, int lk, int g) {
    if (lk < 0 || lk >= x.nxyz) return 0.0;
    const double D = dXsdf(x, g, lk);
    if (D <= 0.0) return 0.0;
    const double h   = dHmesh(x, dir, lk);
    const double jn  = x.jnet[dSfc(x, lr, dir, lk) * x.ng + g];
    const double sgn = (lr == kRIGHT) ? 1.0 : -1.0;
    return sgn * jn * h / (2.0 * D);
}

__device__ inline double* phicNextPtr(const DevCtx& x, int lk, int g) {
    return &x.phic_next[(lk * 4 * x.ng) + g * 4];
}

/// PPR::buildStencil.
__device__ inline void buildStencilD(const DevCtx& x, int lk, int idx[3][3],
                                     bool xrev[3][3], bool yrev[3][3]) {
    const int nxy = x.nxy;
    const int l2d = lk % nxy;
    const int k   = lk / nxy;

    int nb[3][3];
    nb[1][1] = l2d;
    nb[0][1] = dNeib(x, kWEST, l2d);
    nb[2][1] = dNeib(x, kEAST, l2d);
    nb[1][0] = dNeib(x, kNORTH, l2d);
    nb[1][2] = dNeib(x, kSOUTH, l2d);
    nb[0][0] = (nb[1][0] >= 0) ? dNeib(x, kWEST, nb[1][0]) : -1;
    nb[2][0] = (nb[1][0] >= 0) ? dNeib(x, kEAST, nb[1][0]) : -1;
    nb[0][2] = (nb[1][2] >= 0) ? dNeib(x, kWEST, nb[1][2]) : -1;
    nb[2][2] = (nb[1][2] >= 0) ? dNeib(x, kEAST, nb[1][2]) : -1;

    const bool xr_w = (dNeib(x, kWEST, l2d) == l2d);
    const bool xr_e = (dNeib(x, kEAST, l2d) == l2d);
    const bool yr_n = (dNeib(x, kNORTH, l2d) == l2d);
    const bool yr_s = (dNeib(x, kSOUTH, l2d) == l2d);

    const bool x_reflection[3] = {xr_w, false, xr_e};
    const bool y_reflection[3] = {yr_n, false, yr_s};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            idx[i][j]  = (nb[i][j] >= 0) ? nb[i][j] + k * nxy : -1;
            xrev[i][j] = x_reflection[i];
            yrev[i][j] = y_reflection[j];
        }
    }
}

// ---------------------------------------------------------------------------
// reset()
// ---------------------------------------------------------------------------

/// reset() step 1: transverse buckling.
__global__ void kBuckling(DevCtx x) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    const double hmesh = dHmesh(x, kXDIR, lk);
    x.bt[lk * x.ng + g] =
        sqrt((hmesh * hmesh * dXsrf(x, g, lk)) / (4.0 * dXsdf(x, g, lk)));
}

/// reset() step 2: corner flux from surface/average flux.
__global__ void kCornerInit(DevCtx x) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    int  idx[3][3];
    bool xrev[3][3], yrev[3][3];
    buildStencilD(x, lk, idx, xrev, yrev);

    double* phic = phicPtr(x, lk, g);

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            double    sur_flux = 0.0;
            double    avg_flux = 0.0;
            const int dir      = j * 2 + i;

            sur_flux += (getPhisD(x, kRIGHT ^ xrev[i + 0][j + 0], kXDIR, idx[i + 0][j + 0], g) +
                         getPhisD(x, kRIGHT ^ xrev[i + 0][j + 1], kXDIR, idx[i + 0][j + 1], g)) *
                        0.5;

            sur_flux += (getPhisD(x, kRIGHT ^ yrev[i + 0][j + 0], kYDIR, idx[i + 0][j + 0], g) +
                         getPhisD(x, kRIGHT ^ yrev[i + 1][j + 0], kYDIR, idx[i + 1][j + 0], g)) *
                        0.5;

            avg_flux += (idx[i + 0][j + 0] < 0 || idx[i + 0][j + 0] >= x.nxyz) ? 0.0 : x.phif[idx[i + 0][j + 0] * x.ng + g];
            avg_flux += (idx[i + 1][j + 0] < 0 || idx[i + 1][j + 0] >= x.nxyz) ? 0.0 : x.phif[idx[i + 1][j + 0] * x.ng + g];
            avg_flux += (idx[i + 0][j + 1] < 0 || idx[i + 0][j + 1] >= x.nxyz) ? 0.0 : x.phif[idx[i + 0][j + 1] * x.ng + g];
            avg_flux += (idx[i + 1][j + 1] < 0 || idx[i + 1][j + 1] >= x.nxyz) ? 0.0 : x.phif[idx[i + 1][j + 1] * x.ng + g];

            phic[dir] = sur_flux - avg_flux * 0.25;
        }
    }
}

/// reset() step 3: flux fitting from the nodal constraints.
__global__ void kFit(DevCtx x) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    double*      c    = cPtr(x, lk, g);
    const double hm   = dHmesh(x, kXDIR, lk);
    const double invD = 1.0 / dXsdf(x, g, lk);

    const double aflux   = x.phif[lk * x.ng + g];
    const double phis_xl = dPhis(x, kXDIR, kLEFT, lk, g);
    const double phis_xr = dPhis(x, kXDIR, kRIGHT, lk, g);
    const double phis_yl = dPhis(x, kYDIR, kLEFT, lk, g);
    const double phis_yr = dPhis(x, kYDIR, kRIGHT, lk, g);
    const double jnet_xl = dJnet(x, kXDIR, kLEFT, lk, g);
    const double jnet_xr = dJnet(x, kXDIR, kRIGHT, lk, g);
    const double jnet_yl = dJnet(x, kYDIR, kLEFT, lk, g);
    const double jnet_yr = dJnet(x, kYDIR, kRIGHT, lk, g);

    const double* phic = phicPtr(x, lk, g);

    c[triIdx(4, 0)] = 0.2142857143 * (2.0 * aflux - phis_xl - phis_xr) + (0.03571428571 * hm * invD) * (jnet_xr - jnet_xl);
    c[triIdx(0, 4)] = 0.2142857143 * (2.0 * aflux - phis_yl - phis_yr) + (0.03571428571 * hm * invD) * (jnet_yr - jnet_yl);

    c[triIdx(3, 1)] = 0.0;
    c[triIdx(1, 3)] = 0.0;

    c[triIdx(3, 0)] = 0.1 * (phis_xl - phis_xr) - (0.05 * hm * invD) * (jnet_xr + jnet_xl);
    c[triIdx(0, 3)] = 0.1 * (phis_yl - phis_yr) - (0.05 * hm * invD) * (jnet_yr + jnet_yl);

    const double rc = dCrdf(x, lk, g);

    c[triIdx(2, 2)] = aflux + 0.25 * rc * (phic[kNE] + phic[kNW] + phic[kSE] + phic[kSW]) - 0.5 * (phis_xr + phis_xl + phis_yr + phis_yl);

    c[triIdx(2, 1)] = 0.25 * rc * (phic[kSE] + phic[kSW] - phic[kNE] - phic[kNW]) + 0.5 * (phis_yr - phis_yl);
    c[triIdx(1, 2)] = 0.25 * rc * (phic[kSE] - phic[kSW] + phic[kNE] - phic[kNW]) + 0.5 * (phis_xr - phis_xl);

    c[triIdx(2, 0)] = 0.7142857143 * (-2 * aflux + phis_xl + phis_xr) + (0.03571428571 * hm * invD) * (jnet_xl - jnet_xr);
    c[triIdx(0, 2)] = 0.7142857143 * (-2 * aflux + phis_yl + phis_yr) + (0.03571428571 * hm * invD) * (jnet_yl - jnet_yr);

    c[triIdx(1, 1)] = 0.25 * rc * (phic[kSE] - phic[kSW] - phic[kNE] + phic[kNW]);

    c[triIdx(1, 0)] = 0.6 * (phis_xl - phis_xr) + (0.05 * hm * invD) * (jnet_xr + jnet_xl);
    c[triIdx(0, 1)] = 0.6 * (phis_yl - phis_yr) + (0.05 * hm * invD) * (jnet_yr + jnet_yl);

    c[triIdx(0, 0)] = aflux;
}

__device__ inline double zleakD(const DevCtx& x, int lk, int g) {
    return (x.jnet[dSfc(x, kRIGHT, kZDIR, lk) * x.ng + g] -
            x.jnet[dSfc(x, kLEFT, kZDIR, lk) * x.ng + g]) /
           dHmesh(x, kZDIR, lk);
}

__device__ inline double getLeakageD(const DevCtx& x, int lk, int g) {
    return (lk >= 0 && lk < x.nxyz) ? zleakD(x, lk, g) : 0.0;
}

__device__ inline int to3D(int neighbor2d, int axialPlane, int nxy) {
    return (neighbor2d >= 0) ? neighbor2d + axialPlane * nxy : -1;
}

/// updateAxialLeakage().
__global__ void kAxialLeakage(DevCtx x) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    const int nxy = x.nxy;
    const int l2d = lk % nxy;
    const int k   = lk / nxy;

    const int west2d  = dNeib(x, kWEST, l2d);
    const int east2d  = dNeib(x, kEAST, l2d);
    const int north2d = dNeib(x, kNORTH, l2d);
    const int south2d = dNeib(x, kSOUTH, l2d);

    const int northWest2d = (north2d >= 0) ? dNeib(x, kWEST, north2d) : -1;
    const int northEast2d = (north2d >= 0) ? dNeib(x, kEAST, north2d) : -1;
    const int southWest2d = (south2d >= 0) ? dNeib(x, kWEST, south2d) : -1;
    const int southEast2d = (south2d >= 0) ? dNeib(x, kEAST, south2d) : -1;

    const int west      = to3D(west2d, k, nxy);
    const int east      = to3D(east2d, k, nxy);
    const int north     = to3D(north2d, k, nxy);
    const int south     = to3D(south2d, k, nxy);
    const int northWest = to3D(northWest2d, k, nxy);
    const int northEast = to3D(northEast2d, k, nxy);
    const int southWest = to3D(southWest2d, k, nxy);
    const int southEast = to3D(southEast2d, k, nxy);

    const double leak11 = zleakD(x, lk, g);
    const double leak01 = getLeakageD(x, west, g);
    const double leak21 = getLeakageD(x, east, g);
    const double leak10 = getLeakageD(x, north, g);
    const double leak12 = getLeakageD(x, south, g);
    const double leak00 = getLeakageD(x, northWest, g);
    const double leak20 = getLeakageD(x, northEast, g);
    const double leak02 = getLeakageD(x, southWest, g);
    const double leak22 = getLeakageD(x, southEast, g);

    double* l = lPtr(x, lk, g);
    l[0 * 3 + 0] = leak11;
    l[1 * 3 + 0] = (0.25) * (leak21 - leak01);
    l[0 * 3 + 1] = (0.25) * (leak12 - leak10);
    l[2 * 3 + 0] = 0.0833333333 * (leak01 - 2 * leak11 + leak21);
    l[0 * 3 + 2] = 0.0833333333 * (leak10 - 2 * leak11 + leak12);
    l[1 * 3 + 1] = 0.0625000000 * (leak00 - leak02 - leak20 + leak22);
    l[1 * 3 + 2] = 0.0208333333 * (-leak00 + 2 * leak01 - leak02 + leak20 - 2 * leak21 + leak22);
    l[2 * 3 + 1] = 0.0208333333 * (-leak00 + 2 * leak10 + leak02 - leak20 - 2 * leak12 + leak22);
    l[2 * 3 + 2] = 0.0069444444 * (leak00 - 2 * leak01 + leak02 - 2 * leak10 + 4 * leak11 - 2 * leak12 + leak20 - 2 * leak21 + leak22);
}

/// updateSource().  ng == 2 by construction (the host body hardcodes it too --
/// coeff[2][2], c0/c1, q0/q1 -- and the caller refuses any other deck).
template <bool kGuarded>
__global__ void kUpdateSource(DevCtx x) {
    if (kGuarded && pprHalted(x)) return;
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk >= x.nxyz) return;
    const int    ng    = x.ng;
    const double reigv = x.loop->reigv;

    double coeff[2][2];
    for (int g = 0; g < ng; ++g)
        for (int to_g = 0; to_g < ng; ++to_g)
            coeff[g][to_g] = dXssm(x, g, to_g, lk) + dChif(x, to_g, lk) * dXsnf(x, g, lk) * reigv;

    const double* __restrict__ c0 = &x.c_ro[(lk * 15 * ng) + (0 * 15)];
    const double* __restrict__ c1 = &x.c_ro[(lk * 15 * ng) + (1 * 15)];
    double*       q0 = &x.q[(lk * 15 * ng) + (0 * 15)];
    double*       q1 = &x.q[(lk * 15 * ng) + (1 * 15)];

    const double c00 = coeff[0][0], c10 = coeff[1][0];
    const double c01 = coeff[0][1], c11 = coeff[1][1];

    for (int k = 0; k < 15; ++k) {
        q0[k] = c00 * c0[k] + c10 * c1[k];
        q1[k] = c01 * c0[k] + c11 * c1[k];
    }

    const double* l0 = &x.l[(lk * 9 * ng) + (0 * 9)];
    const double* l1 = &x.l[(lk * 9 * ng) + (1 * 9)];

    const int qidx[9] = {0, 1, 2, 5, 6, 7, 9, 10, 11};
    for (int k = 0; k < 9; ++k) {
        q0[qidx[k]] -= l0[k];
        q1[qidx[k]] -= l1[k];
    }
}

// ---------------------------------------------------------------------------
// drive()
// ---------------------------------------------------------------------------

/// updateFused(): particular + homogeneous + projectFlux, one thread per (node,
/// group).  Purely node-local -- it reads q/phic of its own node only -- so the
/// host's `for g { for lk { ... } }` sweep order is not a dependence.
template <bool kGuarded>
__global__ void kUpdateFused(DevCtx x) {
    if (kGuarded && pprHalted(x)) return;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    // The host body keeps p/a/c in an array it re-reads dozens of times; on the
    // device those would be global-memory round trips (the four hFlux_* lines
    // alone read `p` sixty times).  Held in registers here and written back
    // once.  That is a placement change, not an arithmetic one: every
    // expression below is still evaluated in the host's order on the host's
    // operands, which is what keeps the deviation attributable to `exp`.
    double pl[15];
    double al[8];
    double cl[15];
    double qv[15];
    double phv[4];

    {
        const double* qg = qPtr(x, lk, g);
        const double* pc = phicPtr(x, lk, g);
#pragma unroll
        for (int i = 0; i < 15; ++i) qv[i] = qg[i];
#pragma unroll
        for (int i = 0; i < 4; ++i) phv[i] = pc[i];
    }

    const double D  = dXsdf(x, g, lk);
    const double rr = 1.0 / dXsrf(x, g, lk);
    const double rh = 1.0 / dHmesh(x, kXDIR, lk);

    const double Drh2r2  = D * (rr * rr) * (rh * rh);
    const double D2rh4r3 = (D * D) * (rr * rr * rr) * (rh * rh * rh * rh);

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
        for (int j = 0; j < 5 - i; j++)
            pl[triIdx(i, j)] = qv[triIdx(i, j)] * rr;

    pl[triIdx(0, 2)] += Drh2r2 * (140.0 * qv[triIdx(0, 4)] + 12.0 * qv[triIdx(2, 2)]);
    pl[triIdx(2, 0)] += Drh2r2 * (140.0 * qv[triIdx(4, 0)] + 12.0 * qv[triIdx(2, 2)]);
    pl[triIdx(1, 1)] += 60.0 * Drh2r2 * (qv[triIdx(1, 3)] + qv[triIdx(3, 1)]);
    pl[triIdx(0, 1)] += Drh2r2 * (60.0 * qv[triIdx(0, 3)] + 12.0 * qv[triIdx(2, 1)]);
    pl[triIdx(1, 0)] += Drh2r2 * (60.0 * qv[triIdx(3, 0)] + 12.0 * qv[triIdx(1, 2)]);
    pl[triIdx(0, 0)] += Drh2r2 * (12.0 * qv[triIdx(0, 2)] + 12.0 * qv[triIdx(2, 0)] + 40.0 * qv[triIdx(0, 4)] + 40.0 * qv[triIdx(4, 0)]);
    pl[triIdx(0, 0)] += D2rh4r3 * (1680.0 * qv[triIdx(0, 4)] + 288.0 * qv[triIdx(2, 2)] + 1680.0 * qv[triIdx(4, 0)]);

    const double hmesh  = dHmesh(x, kXDIR, lk);
    const double rhmesh = rh;
    const double _2DrH  = 2.0 * D * rhmesh;
    const double bt     = x.bt[lk * x.ng + g];

    const double e1 = exp(bt), re1 = 1.0 / e1;
    const double e2 = exp(bt * kRsq2), re2 = 1.0 / e2;
    const double S   = 0.5 * (e1 - re1);
    const double C   = 0.5 * (e1 + re1);
    const double Sp  = 0.5 * (e2 - re2);
    const double Cp  = 0.5 * (e2 + re2);
    const double Sp2 = Sp * Sp;
    const double Cp2 = Cp * Cp;

    const double rc = dCrdf(x, lk, g);
    const double hFlux_SW = rc * phv[kSW] - (pl[triIdx(0, 0)] + pl[triIdx(0, 1)] + pl[triIdx(0, 2)] + pl[triIdx(0, 3)] + pl[triIdx(0, 4)] - pl[triIdx(1, 0)] - pl[triIdx(1, 1)] - pl[triIdx(1, 2)] - pl[triIdx(1, 3)] + pl[triIdx(2, 0)] + pl[triIdx(2, 1)] + pl[triIdx(2, 2)] - pl[triIdx(3, 0)] - pl[triIdx(3, 1)] + pl[triIdx(4, 0)]);
    const double hFlux_SE = rc * phv[kSE] - (pl[triIdx(0, 0)] + pl[triIdx(0, 1)] + pl[triIdx(0, 2)] + pl[triIdx(0, 3)] + pl[triIdx(0, 4)] + pl[triIdx(1, 0)] + pl[triIdx(1, 1)] + pl[triIdx(1, 2)] + pl[triIdx(1, 3)] + pl[triIdx(2, 0)] + pl[triIdx(2, 1)] + pl[triIdx(2, 2)] + pl[triIdx(3, 0)] + pl[triIdx(3, 1)] + pl[triIdx(4, 0)]);
    const double hFlux_NW = rc * phv[kNW] - (pl[triIdx(0, 0)] - pl[triIdx(0, 1)] + pl[triIdx(0, 2)] - pl[triIdx(0, 3)] + pl[triIdx(0, 4)] - pl[triIdx(1, 0)] + pl[triIdx(1, 1)] - pl[triIdx(1, 2)] + pl[triIdx(1, 3)] + pl[triIdx(2, 0)] - pl[triIdx(2, 1)] + pl[triIdx(2, 2)] - pl[triIdx(3, 0)] + pl[triIdx(3, 1)] + pl[triIdx(4, 0)]);
    const double hFlux_NE = rc * phv[kNE] - (pl[triIdx(0, 0)] - pl[triIdx(0, 1)] + pl[triIdx(0, 2)] - pl[triIdx(0, 3)] + pl[triIdx(0, 4)] + pl[triIdx(1, 0)] - pl[triIdx(1, 1)] + pl[triIdx(1, 2)] - pl[triIdx(1, 3)] + pl[triIdx(2, 0)] - pl[triIdx(2, 1)] + pl[triIdx(2, 2)] + pl[triIdx(3, 0)] - pl[triIdx(3, 1)] + pl[triIdx(4, 0)]);

    const double hCurr_xl = dJnet(x, kXDIR, kLEFT, lk, g) + _2DrH * (pl[triIdx(1, 0)] - 3. * pl[triIdx(2, 0)] + 6. * pl[triIdx(3, 0)] - 10. * pl[triIdx(4, 0)]);
    const double hCurr_xr = dJnet(x, kXDIR, kRIGHT, lk, g) + _2DrH * (pl[triIdx(1, 0)] + 3. * pl[triIdx(2, 0)] + 6. * pl[triIdx(3, 0)] + 10. * pl[triIdx(4, 0)]);
    const double hCurr_yl = dJnet(x, kYDIR, kLEFT, lk, g) + _2DrH * (pl[triIdx(0, 1)] - 3. * pl[triIdx(0, 2)] + 6. * pl[triIdx(0, 3)] - 10. * pl[triIdx(0, 4)]);
    const double hCurr_yr = dJnet(x, kYDIR, kRIGHT, lk, g) + _2DrH * (pl[triIdx(0, 1)] + 3. * pl[triIdx(0, 2)] + 6. * pl[triIdx(0, 3)] + 10. * pl[triIdx(0, 4)]);

    const double Dph_1 = D * (hFlux_SE - hFlux_SW + hFlux_NE - hFlux_NW);
    const double Dph_2 = D * (hFlux_SE - hFlux_SW - hFlux_NE + hFlux_NW);
    const double Dph_3 = D * (hFlux_SE + hFlux_SW - hFlux_NE - hFlux_NW);
    const double Dph_4 = D * (hFlux_SE + hFlux_SW + hFlux_NE + hFlux_NW);

    const double hj_xs = hmesh * (hCurr_xr + hCurr_xl);
    const double hj_xd = hmesh * (hCurr_xr - hCurr_xl);
    const double hj_ys = hmesh * (hCurr_yr + hCurr_yl);
    const double hj_yd = hmesh * (hCurr_yr - hCurr_yl);

    const double denom1 = 1.0 / (4.0 * D * (S - bt * C));
    al[0]               = (Dph_1 + hj_xs) * denom1; // a(1)
    al[2]               = (Dph_3 + hj_ys) * denom1; // a(3)

    const double denom2 = 1.0 / (4.0 * D * bt * S * (bt * S * Cp2 - 2 * C * Sp2));
    al[1]               = (C * Sp2 * (hj_xd - hj_yd) - bt * S * (Cp2 * hj_xd + Sp2 * Dph_4)) * denom2; // a(2)
    al[3]               = (C * Sp2 * (hj_yd - hj_xd) - bt * S * (Cp2 * hj_yd + Sp2 * Dph_4)) * denom2; // a(4)

    const double denom3 = 1.0 / (4.0 * D * Cp * Sp * (bt * C - S));
    al[4]               = (bt * C * Dph_1 + S * hj_xs) * denom3; // a(5)
    al[6]               = (bt * C * Dph_3 + S * hj_ys) * denom3; // a(7)

    al[5] = Dph_2 / (4.0 * D * Sp2); // a(6)

    const double denom4 = 1.0 / (4.0 * D * (bt * S * Cp2 - 2 * C * Sp2));
    al[7]               = denom4 * (bt * S * Dph_4 + C * (hj_xd + hj_yd)); // a(8)

    // projectFlux
    const double bt2 = bt * bt;
    const double bt3 = bt * bt2;
    const double bt4 = bt2 * bt2;

    const double rbt  = 1.0 / bt;
    const double rbt2 = rbt * rbt;
    const double rbt3 = rbt * rbt2;
    const double rbt4 = rbt2 * rbt2;
    const double rbt5 = rbt3 * rbt2;
    const double rbt6 = rbt3 * rbt3;

    const double CpSp   = Cp * Sp;
    const double BtCpSp = bt * CpSp;

    cl[triIdx(4, 0)] = 9.0 * rbt6 * (al[1] * bt * (S * (105 + 45 * bt2 + bt4) - 5 * C * bt * (21 + 2 * bt2)) + 2.0 * al[7] * Sp * ((420 + 90 * bt2 + bt4) * Sp - (10 * kSq2 * bt * (21 + bt2)) * Cp));
    cl[triIdx(0, 4)] = 9.0 * rbt6 * (al[3] * bt * (S * (105 + 45 * bt2 + bt4) - 5 * C * bt * (21 + 2 * bt2)) + 2.0 * al[7] * Sp * ((420 + 90 * bt2 + bt4) * Sp - (10 * kSq2 * bt * (21 + bt2)) * Cp));

    cl[triIdx(3, 1)] = 42 * rbt6 * al[5] * (30 * bt2 + bt4 - 60 * kSq2 * BtCpSp - 7 * kSq2 * bt2 * BtCpSp + 60 * Sp2 + 42 * bt2 * Sp2 + bt4 * Sp2);
    cl[triIdx(1, 3)] = cl[triIdx(3, 1)];

    cl[triIdx(3, 0)] = 7 * rbt5 * (al[0] * bt * (C * bt3 - 6 * S * bt2 + 15 * C * bt - 15 * S) + 2 * al[4] * Sp * (bt * Cp * (30 + bt2) - (6 * (5 + bt2) * kSq2 * Sp)));
    cl[triIdx(0, 3)] = 7 * rbt5 * (al[2] * bt * (C * bt3 - 6 * S * bt2 + 15 * C * bt - 15 * S) + 2 * al[6] * Sp * (bt * Cp * (30 + bt2) - (6 * (5 + bt2) * kSq2 * Sp)));

    cl[triIdx(2, 2)] = 50 * rbt6 * al[7] * (18 * bt2 - 36 * kSq2 * BtCpSp - 6 * kSq2 * bt2 * BtCpSp + 36 * Sp2 + 30 * bt2 * Sp2 + bt4 * Sp2);

    cl[triIdx(2, 1)] = 30 * rbt5 * al[6] * (-3 * kSq2 * bt2 + bt * (12 + bt2) * CpSp - 2 * kSq2 * (3 + 2 * bt2) * Sp2);
    cl[triIdx(1, 2)] = 30 * rbt5 * al[4] * (-3 * kSq2 * bt2 + bt * (12 + bt2) * CpSp - 2 * kSq2 * (3 + 2 * bt2) * Sp2);

    cl[triIdx(2, 0)] = 5 * rbt4 * (al[1] * bt * (3 * S - 3 * C * bt + S * bt2) + 2 * Sp * al[7] * (Sp * bt2 + 6 * Sp - 3 * kSq2 * bt * Cp));
    cl[triIdx(0, 2)] = 5 * rbt4 * (al[3] * bt * (3 * S - 3 * C * bt + S * bt2) + 2 * Sp * al[7] * (Sp * bt2 + 6 * Sp - 3 * kSq2 * bt * Cp));

    cl[triIdx(1, 1)] = 18 * rbt4 * al[5] * (2 * Sp2 + bt2 * (1 + Sp2) - 2 * kSq2 * bt * CpSp);

    cl[triIdx(1, 0)] = 3 * rbt3 * (al[0] * bt * (C * bt - S) + 2 * Sp * al[4] * (bt * Cp - kSq2 * Sp));
    cl[triIdx(0, 1)] = 3 * rbt3 * (al[2] * bt * (C * bt - S) + 2 * Sp * al[6] * (bt * Cp - kSq2 * Sp));

    cl[triIdx(0, 0)] = rbt2 * (S * bt * (al[1] + al[3]) + 2 * al[7] * Sp2);

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
        for (int j = 0; j < 5 - i; j++)
            cl[triIdx(i, j)] += pl[triIdx(i, j)];

    double* pg = pPtr(x, lk, g);
    double* ag = aPtr(x, lk, g);
    double* cg = cPtr(x, lk, g);
#pragma unroll
    for (int i = 0; i < 15; ++i) pg[i] = pl[i];
#pragma unroll
    for (int i = 0; i < 8; ++i) ag[i] = al[i];
#pragma unroll
    for (int i = 0; i < 15; ++i) cg[i] = cl[i];
}

/// PPR::jnetDir, reading the fitting coefficients of node `lk`.
__device__ inline double jnetDirD(const DevCtx& x, int dir, int lk, int g,
                                  double xx, double yy, bool xrev, bool yrev) {
    if (lk < 0 || lk >= x.nxyz) return 0.0;
    if (xrev) xx = -xx;
    if (yrev) yy = -yy;
    const double* __restrict__ c = cRoPtr(x, lk, g);
    const double  j =
        (dir == kXDIR)
            ? c[triIdx(1, 0)] + yy * c[triIdx(1, 1)] + 0.5 * (3 * yy * yy - 1) * c[triIdx(1, 2)] + 3 * xx * c[triIdx(2, 0)] + 3 * xx * yy * c[triIdx(2, 1)] + 1.5 * xx * (3 * yy * yy - 1) * c[triIdx(2, 2)] + 0.5 * (15 * xx * xx - 3) * c[triIdx(3, 0)] + 0.125 * (-60 * xx + 140 * xx * xx * xx) * c[triIdx(4, 0)]
            : c[triIdx(0, 1)] + 3 * yy * c[triIdx(0, 2)] + 0.5 * (15 * yy * yy - 3) * c[triIdx(0, 3)] + 0.125 * (140 * yy * yy * yy - 60 * yy) * c[triIdx(0, 4)] + xx * c[triIdx(1, 1)] + 3 * xx * yy * c[triIdx(1, 2)] + 0.5 * (3 * xx * xx - 1) * c[triIdx(2, 1)] + 1.5 * (3 * xx * xx - 1) * yy * c[triIdx(2, 2)];
    return j * -2.0 * dXsdf(x, g, lk) / dHmesh(x, dir, lk);
}

/// updateCorner().  Reads the 3x3 stencil's fitting coefficients, writes only
/// its own node's corner fluxes -- no write conflict, and no dependence on the
/// host loop's node order.
template <bool kGuarded>
__global__ void kUpdateCorner(DevCtx x) {
    if (kGuarded && pprHalted(x)) return;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    int  idx[3][3];
    bool xrev[3][3], yrev[3][3];
    buildStencilD(x, lk, idx, xrev, yrev);

    double* phic = phicPtr(x, lk, g);

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            const int dir = j * 2 + i;

            const double jnet_x11 = jnetDirD(x, kXDIR, idx[i + 0][j + 0], g, 1, 1, xrev[i + 0][j + 0], yrev[i + 0][j + 0]);
            const double jnet_x12 = jnetDirD(x, kXDIR, idx[i + 0][j + 1], g, 1, -1, xrev[i + 0][j + 1], yrev[i + 0][j + 1]);
            const double jnet_x21 = jnetDirD(x, kXDIR, idx[i + 1][j + 0], g, -1, 1, xrev[i + 1][j + 0], yrev[i + 1][j + 0]);
            const double jnet_x22 = jnetDirD(x, kXDIR, idx[i + 1][j + 1], g, -1, -1, xrev[i + 1][j + 1], yrev[i + 1][j + 1]);

            const double jnet_y11 = jnetDirD(x, kYDIR, idx[i + 0][j + 0], g, 1, 1, xrev[i + 0][j + 0], yrev[i + 0][j + 0]);
            const double jnet_y12 = jnetDirD(x, kYDIR, idx[i + 0][j + 1], g, 1, -1, xrev[i + 0][j + 1], yrev[i + 0][j + 1]);
            const double jnet_y21 = jnetDirD(x, kYDIR, idx[i + 1][j + 0], g, -1, 1, xrev[i + 1][j + 0], yrev[i + 1][j + 0]);
            const double jnet_y22 = jnetDirD(x, kYDIR, idx[i + 1][j + 1], g, -1, -1, xrev[i + 1][j + 1], yrev[i + 1][j + 1]);

            const double xdiff = jnet_x11 - jnet_x21 + jnet_x12 - jnet_x22;
            const double ydiff = jnet_y11 - jnet_y12 + jnet_y21 - jnet_y22;

            phic[dir] = phic[dir] + 0.25 * (xdiff + ydiff);
        }
    }
}

/// The four fuel-only corner-flux sums, on the fixed 256-chunk partition.  One
/// thread per chunk, each summing ITS range in ascending (node, group) order,
/// so the value depends on the partition and not on the launch.  The host then
/// folds the partials in ascending chunk index.
template <bool kGuarded>
__global__ void kCornerPartials(DevCtx x) {
    if (kGuarded && pprHalted(x)) return;
    const int nchunk   = x.nchunk;
    double*   partials = x.partials;
    const int c        = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nchunk) return;

    const int lb = detChunkBegin(x.nxyz, nchunk, c);
    const int le = detChunkBegin(x.nxyz, nchunk, c + 1);

    double nw = 0.0, sw = 0.0, ne = 0.0, se = 0.0;
    for (int lk = lb; lk < le; ++lk) {
        if (!x.is_fuel[lk]) continue;
        for (int g = 0; g < x.ng; ++g) {
            const double* phic = phicPtr(x, lk, g);
            nw += phic[kNW];
            sw += phic[kSW];
            ne += phic[kNE];
            se += phic[kSE];
        }
    }
    partials[0 * nchunk + c] = nw;
    partials[1 * nchunk + c] = sw;
    partials[2 * nchunk + c] = ne;
    partials[3 * nchunk + c] = se;
}

__host__ __device__ inline double relativeChange(double current, double previous) {
    return (previous != 0.0) ? fabs((current - previous) / previous) : 1.0;
}

/// WP6 stage B.  THE FOLD AND THE BREAK TEST, ON THE DEVICE.
///
/// `<<<1, 4>>>` -- one thread per corner, and each thread walks chunk 0 to
/// nchunk-1 ASCENDING into a double it started at 0.0.  That is not "a
/// deterministic reduction" in the general sense; it is byte for byte the loop
/// the host ran over the D2H'd partials, which is what makes the arm's result
/// bit-identical to the c502856 host-sync arm rather than merely close to it.
/// A tree reduction over 256 chunks would have been faster and would have moved
/// the sum, so it is not used.
///
/// Thread 0 then applies PPR::drive's own test to PPR::drive's own tolerance,
/// increments the round counter, and either raises the flag or rolls `cur` into
/// `prev`.  `__threadfence()` before the flag: the guarded kernels of the NEXT
/// body read `converged` and must not see it raised before `cur`/`prev`/`iters`
/// are visible.  (Stream order already sequences the kernels; the fence is what
/// makes the WRITES ordered rather than only the launches.)
__global__ void kCornerFoldAndCheck(DevCtx x) {
    if (pprHalted(x)) return;

    __shared__ double err[4];

    const int          t        = static_cast<int>(threadIdx.x);
    const int          nchunk   = x.nchunk;
    const double*      partials = x.partials;
    PprLoopState*      st       = x.loop;

    if (t < 4) {
        double s = 0.0;
        for (int c = 0; c < nchunk; ++c) s += partials[t * nchunk + c];
        st->cur[t] = s;
        err[t]     = relativeChange(s, st->prev[t]);
    }
    __syncthreads();

    if (t != 0) return;

    const double err_nw = err[0];
    const double err_sw = err[1];
    const double err_ne = err[2];
    const double err_se = err[3];

    st->iters = st->iters + 1;

    if (err_nw < kCornerFluxTolerance && err_sw < kCornerFluxTolerance &&
        err_ne < kCornerFluxTolerance && err_se < kCornerFluxTolerance) {
        __threadfence();
        st->converged = 1;
        return;
    }

    st->prev[0] = st->cur[0];
    st->prev[1] = st->cur[1];
    st->prev[2] = st->cur[2];
    st->prev[3] = st->cur[3];
}

// ---------------------------------------------------------------------------
// WP6 stage F: RASBERY_PPR_MODE=master -- MASTER 4.0 MM section 6.1
// ---------------------------------------------------------------------------
//
// PPR::driveMaster, in three kernels and one loop body.  The scheme is NOT the
// SENM Picard iteration above and shares nothing with it but `reset()`: no
// particular/homogeneous split, no source sweeps, no exponentials.  A 13-term
// Legendre interpolant lands straight in `c`, its nine even-parity terms read
// off the nodal solution and its four cross terms read off corner fluxes that
// solve the corner-point-balance system.
//
// THE ONE PLACE THE DEVICE CANNOT BE THE HOST.  driveMaster's CPB sweep is
// GAUSS-SEIDEL: node lk's corner reads `_phic` of the 2x2 block around that
// corner while nodes below lk in the same sweep have already overwritten
// theirs.  That is a serial dependence over the whole mesh.  The device runs
// the same balance as JACOBI -- read `phic`, write `phic_next`, commit -- with
// the SAME 1e-5 relative-change break, the SAME cap, and the same skip for a
// corner whose 2x2 block contributes no weight.  Both are contractions on one
// diagonally dominant system (per node: diagonal 4w, off-diagonal 2w) and reach
// the SAME fixed point; they stop at different distances from it, ~1e-5
// relative on the corner fluxes.  That is the whole of this arm's N1, and it is
// four orders below the Gate B pin-power envelope.

/// driveMaster step 1: the even-parity coefficients (MM Eq. 6.2 and the first
/// eight of Eq. 6.6).  Elementwise in (node, group) -- B0.
__global__ void kMasterEven(DevCtx x) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    // PPR.cpp's `constexpr double r10 / r14`, and they are written as the same
    // quotients rather than as decimal literals: 1/14 is not representable and
    // 0.07142857142857142 is a different double from the one the host divides.
    const double r10 = 1.0 / 10.0;
    const double r14 = 1.0 / 14.0;

    double* c = cPtr(x, lk, g);

    const double pb  = x.phif[lk * x.ng + g];
    const double pxr = dPhis(x, kXDIR, kRIGHT, lk, g), pxl = dPhis(x, kXDIR, kLEFT, lk, g);
    const double pyr = dPhis(x, kYDIR, kRIGHT, lk, g), pyl = dPhis(x, kYDIR, kLEFT, lk, g);
    const double jxr = getJoutRedD(x, kRIGHT, kXDIR, lk, g);
    const double jxl = getJoutRedD(x, kLEFT, kXDIR, lk, g);
    const double jyr = getJoutRedD(x, kRIGHT, kYDIR, lk, g);
    const double jyl = getJoutRedD(x, kLEFT, kYDIR, lk, g);

    c[triIdx(0, 0)] = pb;
    c[triIdx(1, 0)] = r10 * (6.0 * (pxr - pxl) + (jxr - jxl));
    c[triIdx(2, 0)] = r14 * (10.0 * (pxr + pxl) + (jxr + jxl) - 20.0 * pb);
    c[triIdx(3, 0)] = -r10 * ((pxr - pxl) + (jxr - jxl));
    c[triIdx(4, 0)] = -r14 * (3.0 * (pxr + pxl) + (jxr + jxl) - 6.0 * pb);
    c[triIdx(0, 1)] = r10 * (6.0 * (pyr - pyl) + (jyr - jyl));
    c[triIdx(0, 2)] = r14 * (10.0 * (pyr + pyl) + (jyr + jyl) - 20.0 * pb);
    c[triIdx(0, 3)] = -r10 * ((pyr - pyl) + (jyr - jyl));
    c[triIdx(0, 4)] = -r14 * (3.0 * (pyr + pyl) + (jyr + jyl) - 6.0 * pb);
    c[triIdx(1, 3)] = 0.0;
    c[triIdx(3, 1)] = 0.0;
    c[triIdx(1, 1)] = 0.0;
    c[triIdx(1, 2)] = 0.0;
    c[triIdx(2, 1)] = 0.0;
    c[triIdx(2, 2)] = 0.0;
}

/// driveMaster step 2, the sweep: one thread per (node, group), all four of its
/// corner copies, reading the PREVIOUS iterate and writing the next.  The
/// per-thread relative change goes to `mrel` so the break test can be a
/// reduction rather than a shared accumulator.
template <bool kGuarded>
__global__ void kMasterCpb(DevCtx x) {
    if (kGuarded && pprHalted(x)) return;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    int  idx[3][3];
    bool xrev[3][3], yrev[3][3];
    buildStencilD(x, lk, idx, xrev, yrev);

    const double* phic_in  = phicPtr(x, lk, g);
    double*       phic_out = phicNextPtr(x, lk, g);

    double node_max = 0.0;

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            const int dir = j * 2 + i;
            double    num = 0.0;
            double    den = 0.0;

            for (int dj = 0; dj < 2; dj++) {
                for (int di = 0; di < 2; di++) {
                    const int m = idx[i + di][j + dj];
                    if (m < 0 || m >= x.nxyz) continue;

                    const bool xr = xrev[i + di][j + dj];
                    const bool yr = yrev[i + di][j + dj];

                    const int out_x = ((di == 0) ? kRIGHT : kLEFT) ^ (xr ? 1 : 0);
                    const int out_y = ((dj == 0) ? kRIGHT : kLEFT) ^ (yr ? 1 : 0);

                    const double pox = getPhisD(x, out_x, kXDIR, m, g);
                    const double pix = getPhisD(x, out_x ^ 1, kXDIR, m, g);
                    const double poy = getPhisD(x, out_y, kYDIR, m, g);
                    const double piy = getPhisD(x, out_y ^ 1, kYDIR, m, g);
                    const double jox = getJoutRedD(x, out_x, kXDIR, m, g);
                    const double joy = getJoutRedD(x, out_y, kYDIR, m, g);
                    const double pbm = x.phif[m * x.ng + g];

                    const int    icl   = ((di == 0) ? 1 : 0) ^ (xr ? 1 : 0);
                    const int    jcl   = ((dj == 0) ? 1 : 0) ^ (yr ? 1 : 0);
                    const int    adj1  = jcl * 2 + (1 - icl);
                    const int    adj2  = (1 - jcl) * 2 + icl;
                    const double fadj1 = x.phic[(m * 4 * x.ng) + (g * 4) + adj1];
                    const double fadj2 = x.phic[(m * 4 * x.ng) + (g * 4) + adj2];

                    const double w = dXsdf(x, g, m) / dHmesh(x, kXDIR, m);
                    num += w * (5.0 * (pox + poy) + (pix + piy) + jox + joy -
                                6.0 * pbm - fadj1 - fadj2);
                    den += 4.0 * w;
                }
            }

            const double fold = phic_in[dir];
            if (den <= 0.0) {
                // The host `continue`s: the corner keeps its value and takes no
                // part in the break test.  Copying it forward is that, exactly,
                // and leaving `node_max` alone is the other half -- writing a
                // relativeChange of a value against itself would report 1.0 for
                // an untouched zero corner and hold the loop open forever.
                phic_out[dir] = fold;
                continue;
            }

            const double fnew = num / den;
            if (fold != 0.0) node_max = fmax(node_max, fabs((fnew - fold) / fold));
            phic_out[dir] = fnew;
        }
    }

    x.mrel[lk * x.ng + g] = node_max;
}

/// driveMaster step 2, the commit: `phic_next` becomes `phic`, and the same
/// deterministic partition the corner sums use folds `mrel` into one max per
/// chunk.  Both in one pass because both walk the same node range, and a
/// separate copy kernel would read 4x the traffic for nothing.
///
/// THE PARTITION DOES NOT MOVE THIS ANSWER.  `max` is exactly associative and
/// commutative in floating point -- unlike the SENM corner SUMS, which is why
/// that reduction is pinned to the host's 256-chunk association and this one
/// would give the same bits under any.
template <bool kGuarded>
__global__ void kMasterCommit(DevCtx x) {
    if (kGuarded && pprHalted(x)) return;
    const int nchunk = x.nchunk;
    const int c      = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nchunk) return;

    const int lb = detChunkBegin(x.nxyz, nchunk, c);
    const int le = detChunkBegin(x.nxyz, nchunk, c + 1);

    double m = 0.0;
    for (int lk = lb; lk < le; ++lk) {
        for (int g = 0; g < x.ng; ++g) {
            double*       dst = phicPtr(x, lk, g);
            const double* src = phicNextPtr(x, lk, g);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            m = fmax(m, x.mrel[lk * x.ng + g]);
        }
    }
    x.partials[c] = m;
}

/// driveMaster step 2, the break test: PPR.cpp's `if (maxrel <
/// kCornerFluxTolerance) break;` after the sweep, on the device, so the loop
/// needs no per-round synchronise.  Same shape as kCornerFoldAndCheck -- one
/// thread, ascending chunks, `iters` counted only while the flag is down, a
/// `__threadfence()` before raising it.
__global__ void kMasterFoldAndCheck(DevCtx x) {
    if (pprHalted(x)) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double maxrel = 0.0;
    for (int c = 0; c < x.nchunk; ++c) maxrel = fmax(maxrel, x.partials[c]);

    PprLoopState* st = x.loop;
    st->cur[0]       = maxrel;
    st->iters        = st->iters + 1;

    if (maxrel < kCornerFluxTolerance) {
        __threadfence();
        st->converged = 1;
    }
}

/// driveMaster step 3: the four cross terms from the converged corners (the
/// last four of MM Eq. 6.6).  MM axes put xi=+1 east and eta=+1 south, so
/// phi1=SE, phi2=SW, phi3=NW, phi4=NE.  Elementwise -- B0 given its input.
__global__ void kMasterCross(DevCtx x) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= x.nxyz * x.ng) return;
    const int lk = t / x.ng;
    const int g  = t % x.ng;

    double*       c    = cPtr(x, lk, g);
    const double* phic = phicPtr(x, lk, g);

    const double f1 = phic[kSE], f2 = phic[kSW], f3 = phic[kNW], f4 = phic[kNE];
    const double pxr = dPhis(x, kXDIR, kRIGHT, lk, g), pxl = dPhis(x, kXDIR, kLEFT, lk, g);
    const double pyr = dPhis(x, kYDIR, kRIGHT, lk, g), pyl = dPhis(x, kYDIR, kLEFT, lk, g);
    const double aflux = x.phif[lk * x.ng + g];

    c[triIdx(1, 1)] = 0.25 * (f1 - f2 + f3 - f4);
    c[triIdx(1, 2)] = 0.25 * (f1 - f2 - f3 + f4 - 2.0 * (pxr - pxl));
    c[triIdx(2, 1)] = 0.25 * (f1 + f2 - f3 - f4 - 2.0 * (pyr - pyl));
    c[triIdx(2, 2)] =
        0.25 * (f1 + f2 + f3 + f4 - 2.0 * (pxr + pxl + pyr + pyl) + 4.0 * aflux);
}

// ---------------------------------------------------------------------------
// WP6 stage C: the borrow, checked rather than asserted
// ---------------------------------------------------------------------------

/// Elementwise BITWISE comparison of a borrowed device buffer against the host
/// array it replaces (uploaded into this instance's own block for the purpose).
///
/// BITWISE, not `!=`.  The question is "is the borrow the same bytes", and two
/// NaNs compare unequal under `!=` while being the same value for every
/// consumer here; a tolerance would answer a different and easier question.
///
/// NO ATOMIC.  One thread per deterministic chunk writing its own count, folded
/// by kFoldMismatch -- the same partition the corner sums use, for the same
/// reason: this file may not contain a launch-order-dependent reduction.
__global__ void kCanonicalCompare(const double* borrowed, const double* uploaded,
                                  long long n, int nchunk, int slot,
                                  unsigned long long* counts) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nchunk) return;
    const long long    lb  = (n * c) / nchunk;
    const long long    le  = (n * (c + 1)) / nchunk;
    unsigned long long bad = 0;
    for (long long i = lb; i < le; ++i)
        if (__double_as_longlong(borrowed[i]) != __double_as_longlong(uploaded[i])) ++bad;
    counts[static_cast<long long>(slot) * nchunk + c] = bad;
}

__global__ void kFoldMismatch(unsigned long long* counts, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned long long s = 0;
    for (int i = 0; i < n; ++i) s += counts[i];
    counts[n] = s;
}

// ---------------------------------------------------------------------------
// WP6 stage B: the conditional WHILE
// ---------------------------------------------------------------------------
//
// A LOCAL COPY OF GpuOuterWhile.h's SEVEN-CALL SEQUENCE, deliberately.  That
// header's `buildOuterWhile` is bound to DeviceOuterSegmentState and its
// predicate kernel is a non-static `__global__` already defined in
// CudaOuterGraph.cu -- including it here would be a duplicate definition at
// link time.  What is shared is the ORDER OF THE CALLS and the CUDA-13
// signature split, and tools/test_ppr_device_loop_contract.py checks that the
// two files still spell both the same way.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define RASBERY_HAS_PPR_WHILE 1
#else
#define RASBERY_HAS_PPR_WHILE 0
#endif

#if RASBERY_HAS_PPR_WHILE

/// The stop rule, in one kernel, used TWICE -- as the root graph's arm node and
/// as the body's last node.  It is PPR::drive's loop header and nothing else:
/// `citer < niter` and "the break test has not fired".  Two spellings of a stop
/// rule are two chances to spell it differently, so there is one.
__global__ void kPprWhileCond(cudaGraphConditionalHandle handle,
                              const PprLoopState* st) {
    cudaGraphSetConditional(
        handle, (st->converged == 0 && st->iters < st->niter) ? 1u : 0u);
}

/// cudaGraphAddNode's signature changed in CUDA 13.0 (an edge-data pointer was
/// inserted before numDependencies).  Same split, same reason, same wording as
/// GpuOuterWhile.h::addWhileNode.
inline cudaError_t addPprWhileNode(cudaGraphNode_t* out_node, cudaGraph_t parent,
                                   const cudaGraphNode_t* deps, std::size_t ndeps,
                                   cudaGraphConditionalHandle handle,
                                   cudaGraph_t* body_out) {
    cudaGraphNodeParams np{};
    std::memset(&np, 0, sizeof(np));
    np.type               = cudaGraphNodeTypeConditional;
    np.conditional.handle = handle;
    np.conditional.type   = cudaGraphCondTypeWhile;
    np.conditional.size   = 1;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const cudaError_t e = cudaGraphAddNode(out_node, parent, deps, nullptr, ndeps, &np);
#else
    const cudaError_t e = cudaGraphAddNode(out_node, parent, deps, ndeps, &np);
#endif
    if (e != cudaSuccess) return e;
    *body_out = np.conditional.phGraph_out[0];
    return cudaSuccess;
}

/// Build the WHILE.  `record` is handed the body stream and must enqueue
/// EXACTLY ONE Picard round on it.  Every early return ends both captures, so a
/// refusal leaves neither stream in capture mode; `stage` names the call that
/// refused and becomes the receipt's `graph_refusal`.
template <typename RecordBody>
inline cudaError_t buildPprWhile(cudaStream_t root_stream, cudaStream_t body_stream,
                                 const PprLoopState* d_loop, const char** stage,
                                 RecordBody record, cudaGraph_t* root_out,
                                 cudaGraphExec_t* exec_out) {
    *stage         = "BeginCapture(root)";
    cudaError_t rc = cudaStreamBeginCapture(root_stream, cudaStreamCaptureModeRelaxed);
    if (rc != cudaSuccess) return rc;

    auto abandon_root = [&](cudaError_t err) {
        cudaGraph_t dead = nullptr;
        cudaStreamEndCapture(root_stream, &dead);
        if (dead != nullptr) cudaGraphDestroy(dead);
        cudaGetLastError();
        return err;
    };

    cudaStreamCaptureStatus st    = cudaStreamCaptureStatusNone;
    unsigned long long      id    = 0;
    cudaGraph_t             g     = nullptr;
    const cudaGraphNode_t*  deps  = nullptr;
    std::size_t             ndeps = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const cudaGraphEdgeData* edge_data = nullptr;
#endif

    *stage = "GetCaptureInfo(root)";
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    rc = cudaStreamGetCaptureInfo(root_stream, &st, &id, &g, &deps, &edge_data, &ndeps);
#else
    rc = cudaStreamGetCaptureInfo(root_stream, &st, &id, &g, &deps, &ndeps);
#endif
    if (rc != cudaSuccess) return abandon_root(rc);
    if (g == nullptr) return abandon_root(cudaErrorStreamCaptureUnsupported);

    *stage = "ConditionalHandleCreate";
    cudaGraphConditionalHandle handle{};
    // Default 0 and NO cudaGraphCondAssignDefault: the arm kernel is the only
    // thing entitled to open the loop, and a default of 1 would run one round
    // on a statepoint whose cap is zero.
    rc = cudaGraphConditionalHandleCreate(&handle, g, 0, 0);
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "arm";
    kPprWhileCond<<<1, 1, 0, root_stream>>>(handle, d_loop);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "GetCaptureInfo(arm)";
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    rc = cudaStreamGetCaptureInfo(root_stream, &st, &id, &g, &deps, &edge_data, &ndeps);
#else
    rc = cudaStreamGetCaptureInfo(root_stream, &st, &id, &g, &deps, &ndeps);
#endif
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "AddNode(while)";
    cudaGraphNode_t cond = nullptr;
    cudaGraph_t     body = nullptr;
    rc                   = addPprWhileNode(&cond, g, deps, ndeps, handle, &body);
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "UpdateCaptureDependencies";
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    rc = cudaStreamUpdateCaptureDependencies(root_stream, &cond, nullptr, 1,
                                             cudaStreamSetCaptureDependencies);
#else
    rc = cudaStreamUpdateCaptureDependencies(root_stream, &cond, 1,
                                             cudaStreamSetCaptureDependencies);
#endif
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "BeginCaptureToGraph(body)";
    rc     = cudaStreamBeginCaptureToGraph(body_stream, body, nullptr, nullptr, 0,
                                           cudaStreamCaptureModeRelaxed);
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage              = "record(body)";
    const bool recorded = record(body_stream);

    // The body's LAST node is the stop rule, and it is the same kernel the arm
    // is.  Issued even when `recorded` is false: the body capture has to be
    // ended either way, and a body without it would be an infinite loop if it
    // ever reached an exec.
    kPprWhileCond<<<1, 1, 0, body_stream>>>(handle, d_loop);
    const cudaError_t cond_rc = cudaGetLastError();

    *stage                    = "EndCapture(body)";
    cudaGraph_t       body_out = nullptr;
    const cudaError_t body_rc  = cudaStreamEndCapture(body_stream, &body_out);

    *stage                   = "EndCapture(root)";
    cudaGraph_t       root    = nullptr;
    const cudaError_t root_rc = cudaStreamEndCapture(root_stream, &root);

    if (!recorded || cond_rc != cudaSuccess || body_rc != cudaSuccess ||
        root_rc != cudaSuccess) {
        if (root != nullptr) cudaGraphDestroy(root);
        cudaGetLastError();
        if (body_rc != cudaSuccess) { *stage = "EndCapture(body)"; return body_rc; }
        if (root_rc != cudaSuccess) { *stage = "EndCapture(root)"; return root_rc; }
        if (cond_rc != cudaSuccess) { *stage = "cond"; return cond_rc; }
        *stage = "record(body)";
        return cudaErrorStreamCaptureInvalidated;
    }

    *stage = "Instantiate";
    cudaGraphExec_t exec = nullptr;
    // 3-argument form: the legacy (errorNode, logBuffer, size) overload is gone
    // in CUDA 13, which the 238 server builds with.
    rc = cudaGraphInstantiate(&exec, root, 0ull);
    if (rc != cudaSuccess) {
        cudaGraphDestroy(root);
        cudaGetLastError();
        return rc;
    }
    *root_out = root;
    *exec_out = exec;
    *stage    = "ok";
    return cudaSuccess;
}

#endif // RASBERY_HAS_PPR_WHILE

/// Which loop drives the Picard iteration.  RASBERY_GPU_PPR_DEVICE_LOOP=0 puts
/// the c502856 host-sync loop back for the A/B; RASBERY_GPU_PPR_GRAPH=1 asks
/// for the WHILE on top of it.
enum class LoopArm : int { HostSync = 0, DeviceStream, DeviceGraph };

inline const char* loopArmName(LoopArm a) {
    switch (a) {
        case LoopArm::HostSync:     return "host_sync";
        case LoopArm::DeviceStream: return "device_stream";
        case LoopArm::DeviceGraph:  return "device_graph";
    }
    return "?";
}

} // namespace
// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct PprBackend::Impl {
    bool        enabled = false;
    bool        failed  = false;
    std::string status_text;
    int         device = -1;

    cudaStream_t stream = nullptr;

    // WP6 stage B.  The arm the caller asked for, and the arm actually in use
    // (they differ only when the graph refused and fell back to the stream).
    LoopArm     arm_requested = LoopArm::DeviceStream;
    LoopArm     arm           = LoopArm::DeviceStream;
    std::string graph_refusal;

    // Shape this instance is sized for; a change re-allocates.
    int ng = 0, nxyz = 0, nxy = 0, nsurf = 0;
    int nchunk = 0;

    DevCtx ctx{};

    // Device allocations, all owned here.
    double*        d_hmesh   = nullptr;
    int*           d_lktosfc = nullptr;
    int*           d_neibrb  = nullptr;
    unsigned char* d_is_fuel = nullptr;

    double* d_phif = nullptr;
    double* d_phis = nullptr;
    double* d_jnet = nullptr;
    double* d_xsdf = nullptr;
    double* d_xsrf = nullptr;
    double* d_xsnf = nullptr;
    double* d_xssm = nullptr;
    double* d_chif = nullptr;
    double* d_crdf = nullptr;

    double* d_phic = nullptr;
    double* d_p    = nullptr;
    double* d_a    = nullptr;
    double* d_c    = nullptr;
    double* d_q    = nullptr;
    double* d_l    = nullptr;
    double* d_bt   = nullptr;

    // WP6 stage F.  MASTER mode's Jacobi CPB needs the next iterate and a
    // per-(node, group) relative-change scratch.  Allocated unconditionally --
    // 5 * nxyz * ng doubles, 0.7 MB at KNGR size -- because ensureShape is
    // reached before the mode is known and making the shape depend on the mode
    // would turn a mode switch into a re-shape of every buffer and every graph.
    double* d_phic_next = nullptr;
    double* d_mrel      = nullptr;

    double* d_partials = nullptr;
    double* h_partials = nullptr; ///< pinned, 4 * nchunk

    // --- WP6 stage D: the reconstruction's own block -----------------------
    //
    // SEPARATE FROM ensureShape's, because its shape is a different tuple
    // (nxya, nz, kbc, kec, ndiv, npins, overlaps, form slots) and because the
    // arm is opt-in: a run with RASBERY_GPU_PPR_RECON unset allocates none of
    // it.  Freed by release() along with everything else, so a re-shape of the
    // NODE mesh drops it too -- the coefficients it reads would be gone.
    bool recon_shaped          = false;
    bool recon_static_uploaded = false;
    bool fmap_uploaded         = false;
    int  r_nxya = 0, r_nz = 0, r_kbc = 0, r_kec = 0, r_ndiv = 0, r_npins = 0;
    int  r_overlaps = 0, r_slots = 0;
    unsigned long long recon_static_bytes = 0;
    unsigned long long recon_shapes       = 0;
    std::string        recon_refusal;
    /// Did the LAST resetAndDrive leave device coefficients?  The reconstruction
    /// reads `p`/`a`/`bt` as that call left them, so a statepoint that fell back
    /// to the host must not reconstruct on the device from a stale set.
    bool last_drive_ok = false;

    int*           d_latol   = nullptr;
    double*        d_vol     = nullptr;
    double*        d_hz      = nullptr;
    int*           d_pin_off = nullptr;
    int*           d_ovl_di  = nullptr;
    int*           d_ovl_dj  = nullptr;
    double*        d_ovl_dxh = nullptr;
    double*        d_ovl_dyh = nullptr;
    double*        d_q_xq    = nullptr;
    double*        d_q_yq    = nullptr;
    double*        d_q_wt    = nullptr;
    double*        d_q_leg   = nullptr;
    double*        d_xskf    = nullptr;
    double*        d_gmap    = nullptr;
    double*        d_fmap    = nullptr;
    int*           d_plane_lo    = nullptr;
    int*           d_plane_hi    = nullptr;
    double*        d_plane_alpha = nullptr;
    double*        d_pin_power   = nullptr;
    double*        d_pin_flux    = nullptr;
    double*        d_norm_partial = nullptr;
    double*        d_peak_partial = nullptr;
    double*        d_radial_power = nullptr;
    double*        d_radial_hz    = nullptr;
    double*        d_scalars      = nullptr;
    double*        h_scalars      = nullptr; ///< pinned, 4

    // WP6 stage B.  The Picard loop's device-side state, and its pinned mirror.
    // The mirror is written before the statepoint (reigv, the cap, the zeroed
    // sums) and read after it (the round count) -- one 64-byte transfer each
    // way, both async on the same stream, no extra synchronise.
    PprLoopState* d_loop = nullptr;
    PprLoopState* h_loop = nullptr;

    // WP6 stage C.  3*nchunk per-chunk mismatch counts plus one folded total,
    // only ever written under CanonicalMode::Verify.
    unsigned long long* d_mismatch = nullptr;
    unsigned long long* h_mismatch = nullptr; ///< pinned, 1

    bool geometry_uploaded = false;

    // WP6 stage C.  The generations the device copies of chif / crdf were built
    // from.  0 is "nothing uploaded yet", which always uploads.
    unsigned long long chif_gen_seen = 0;
    unsigned long long crdf_gen_seen = 0;

#if RASBERY_HAS_PPR_WHILE
    // WP6 stage B.  ONE instantiation per (deck, shape, bound pointer set).
    // The key is the DevCtx the body was captured with, compared bytewise: the
    // captured kernel nodes bake every pointer and every scalar in it, so a
    // DevCtx that differs by one field is a graph that would replay the wrong
    // operand.  Padding can only make the comparison say "changed", which costs
    // an instantiation and cannot cost correctness.
    cudaStream_t    graph_root_stream = nullptr;
    cudaGraph_t     graph_root        = nullptr;
    cudaGraphExec_t graph_exec        = nullptr;
    DevCtx          graph_ctx{};
    bool            graph_valid = false;
#endif

    unsigned long long n_statepoints = 0;
    unsigned long long n_iterations  = 0;
    double             wall_ms       = 0.0;

    unsigned long long n_host_syncs      = 0;
    unsigned long long n_graph_launches  = 0;
    unsigned long long n_graph_builds    = 0;
    unsigned long long n_h2d_bytes       = 0;
    unsigned long long n_h2d_elided      = 0;
    unsigned long long n_d2h_bytes       = 0;
    unsigned long long n_canonical_sp    = 0;
    unsigned long long n_canonical_bad   = 0;
    unsigned long long n_allocations     = 0;
    unsigned long long n_reallocations   = 0;
    unsigned long long n_shapes          = 0;
    unsigned long long n_d2h_elided      = 0;
    unsigned long long n_recon_statepoints    = 0;
    unsigned long long n_pin_materializations = 0;
    unsigned long long n_recon_repairs        = 0;

    /// WP6 stage F.  The refusal ladder: why the LAST statepoint fell back, and
    /// how many times each reason has.  Written by noteRefusal() at every
    /// `return false` on the way into the device, including the ones
    /// PPR::resetAndDriveGpu takes before this class is ever asked.
    ppr::Refusal       last_refusal = ppr::Refusal::None;
    unsigned long long n_refusals[static_cast<int>(ppr::Refusal::Count)] = {};

    void noteRefusal(ppr::Refusal r) {
        const int i = static_cast<int>(r);
        if (i < 0 || i >= static_cast<int>(ppr::Refusal::Count)) return;
        last_refusal = r;
        ++n_refusals[i];
    }

    /// The last drive elided the coefficient D2H on the promise that the
    /// device reconstruction would consume them.  Cleared by the reconstruction
    /// that does; read by noteReconRepair() when one does not.
    bool coeffs_device_only = false;

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop  = nullptr;

    ~Impl() { release(); }

    void releaseRecon() {
        // WP19.  Every free here is "potentially unsafe" in CUDA's capture
        // vocabulary and a re-shape runs it mid-run, on one lane's thread,
        // while another lane may be capturing.  See GpuCaptureArbiter.h.
        rasbery::AllocWindow _alloc_window("ppr.recon.release");
        auto f = [](void* p) { if (p) cudaFree(p); };
        f(d_latol); f(d_vol); f(d_hz);
        f(d_pin_off); f(d_ovl_di); f(d_ovl_dj); f(d_ovl_dxh); f(d_ovl_dyh);
        f(d_q_xq); f(d_q_yq); f(d_q_wt); f(d_q_leg);
        f(d_xskf); f(d_gmap); f(d_fmap);
        f(d_plane_lo); f(d_plane_hi); f(d_plane_alpha);
        f(d_pin_power); f(d_pin_flux);
        f(d_norm_partial); f(d_peak_partial); f(d_radial_power); f(d_radial_hz);
        f(d_scalars);
        if (h_scalars) cudaFreeHost(h_scalars);
        d_latol = nullptr; d_vol = nullptr; d_hz = nullptr;
        d_pin_off = nullptr; d_ovl_di = nullptr; d_ovl_dj = nullptr;
        d_ovl_dxh = nullptr; d_ovl_dyh = nullptr;
        d_q_xq = d_q_yq = d_q_wt = d_q_leg = nullptr;
        d_xskf = d_gmap = d_fmap = nullptr;
        d_plane_lo = nullptr; d_plane_hi = nullptr; d_plane_alpha = nullptr;
        d_pin_power = d_pin_flux = nullptr;
        d_norm_partial = d_peak_partial = d_radial_power = d_radial_hz = nullptr;
        d_scalars = nullptr; h_scalars = nullptr;
        recon_shaped          = false;
        recon_static_uploaded = false;
        fmap_uploaded         = false;
        recon_static_bytes    = 0;
    }

    void releaseGraph() {
#if RASBERY_HAS_PPR_WHILE
        rasbery::AllocWindow _alloc_window("ppr.graph.release");
        if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
        if (graph_root != nullptr) cudaGraphDestroy(graph_root);
        graph_exec  = nullptr;
        graph_root  = nullptr;
        graph_valid = false;
#endif
    }

    void release() {
        releaseGraph();
        // WP19.  Held across the whole teardown -- including releaseRecon()'s
        // own nested window, which the arbiter's thread-local depth counter
        // makes free.  A deck's teardown is exactly as unsafe to run inside a
        // sibling's capture as its stand-up is.
        rasbery::AllocWindow _alloc_window("ppr.release");
#if RASBERY_HAS_PPR_WHILE
        if (graph_root_stream != nullptr) cudaStreamDestroy(graph_root_stream);
        graph_root_stream = nullptr;
#endif
        auto f  = [](void* p) { if (p) cudaFree(p); };
        f(d_hmesh);  f(d_lktosfc); f(d_neibrb); f(d_is_fuel);
        f(d_phif);   f(d_phis);    f(d_jnet);
        f(d_xsdf);   f(d_xsrf);    f(d_xsnf);   f(d_xssm);  f(d_chif); f(d_crdf);
        f(d_phic);   f(d_p);       f(d_a);      f(d_c);     f(d_q);    f(d_l);  f(d_bt);
        f(d_phic_next); f(d_mrel);
        f(d_partials); f(d_loop);  f(d_mismatch);
        releaseRecon();
        if (h_partials) cudaFreeHost(h_partials);
        if (h_loop) cudaFreeHost(h_loop);
        if (h_mismatch) cudaFreeHost(h_mismatch);
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_stop) cudaEventDestroy(ev_stop);
        if (stream) cudaStreamDestroy(stream);
        d_hmesh = nullptr; d_lktosfc = nullptr; d_neibrb = nullptr; d_is_fuel = nullptr;
        d_phif = d_phis = d_jnet = nullptr;
        d_xsdf = d_xsrf = d_xsnf = d_xssm = d_chif = d_crdf = nullptr;
        d_phic = d_p = d_a = d_c = d_q = d_l = d_bt = nullptr;
        d_phic_next = nullptr; d_mrel = nullptr;
        d_partials = nullptr; h_partials = nullptr;
        d_loop = nullptr; h_loop = nullptr;
        d_mismatch = nullptr; h_mismatch = nullptr;
        ev_start = nullptr; ev_stop = nullptr; stream = nullptr;
        geometry_uploaded = false;
        chif_gen_seen = 0;
        crdf_gen_seen = 0;
    }

    /// THE ONLY cudaStreamSynchronize IN THIS FILE, and the only place
    /// n_host_syncs moves.  Both entry points call it, so
    /// host_syncs_per_statepoint stays the truth when stage D is on.
    bool syncStream(const char* what) {
        const cudaError_t rc =
            rasbery::xfer::streamSync("CudaPprBackend.cu:syncStream", what, stream);
        ++n_host_syncs;
        if (rc != cudaSuccess) return fail(what, rc);
        return true;
    }

    bool fail(const char* what, cudaError_t rc) {
        failed      = true;
        // EVERY CUDA failure is one rung of the ladder, and it is set HERE
        // rather than at the dozens of `if (!h2d(...)) return false;` call
        // sites -- one writer, so the reason cannot be forgotten at a seam
        // somebody adds later.
        noteRefusal(ppr::Refusal::CudaFailure);
        status_text = std::string("disabled after CUDA failure in ") + what + ": " +
                      cudaGetErrorString(rc);
        std::fprintf(stderr, "[RASBERY][PPR_GPU][WARN] %s -- falling back to host PPR\n",
                     status_text.c_str());
        release();
        return false;
    }

    bool ensureShape(const ppr::GeomView& g) {
        if (ng == g.ng && nxyz == g.nxyz && nxy == g.nxy && nsurf == g.nsurf &&
            stream != nullptr)
            return true;
        // WP6 stage E.  A RE-shape is the thing a per-slot backend must not do
        // per statepoint, so it is counted rather than assumed away: a run whose
        // receipt says `reallocations > 0` allocated inside the statepoint loop.
        if (n_shapes > 0) ++n_reallocations;
        ++n_shapes;
        release();
        // WP19, AND THIS IS THE ONE THAT KILLED THE BATCH.  Everything below --
        // cudaStreamCreate, two cudaEventCreate, 25 cudaMalloc, 3
        // cudaMallocHost -- is a "potentially unsafe" API, and it runs on the
        // FIRST statepoint of every deck, i.e. exactly when a sibling lane that
        // started a few milliseconds earlier is capturing its CMFD/outer/PPR
        // graph.  ThreadLocal capture does not stop a sibling thread; the
        // arbiter does.  See GpuCaptureArbiter.h.
        rasbery::AllocWindow _alloc_window("ppr.shape.standup");
        ng     = g.ng;
        nxyz   = g.nxyz;
        nxy    = g.nxy;
        nsurf  = g.nsurf;
        nchunk = detChunks(nxyz);

        // The ordinal the RECEIPT reports is the one these buffers land on, and
        // that is decided here, not in the constructor: --batch-mode selects a
        // slot's device on the worker thread, after the Driver (hence the PPR,
        // hence this backend) already exists.  A constructor-time ordinal would
        // name GPU 0 for every slot in a multi-GPU batch.
        cudaGetDevice(&device);

        cudaError_t rc = cudaStreamCreate(&stream);
        if (rc != cudaSuccess) return fail("cudaStreamCreate", rc);
        if ((rc = cudaEventCreate(&ev_start)) != cudaSuccess) return fail("cudaEventCreate", rc);
        if ((rc = cudaEventCreate(&ev_stop)) != cudaSuccess) return fail("cudaEventCreate", rc);

        const size_t nn  = static_cast<size_t>(nxyz);
        const size_t nng = nn * ng;
        const size_t nsg = static_cast<size_t>(nsurf) * ng;

        struct Alloc { void** p; size_t bytes; const char* name; };
        const Alloc allocs[] = {
            {(void**)&d_hmesh,   nn * kNDIRMAX * sizeof(double),      "hmesh"},
            {(void**)&d_lktosfc, nn * kNDIRMAX * kLR * sizeof(int),   "lktosfc"},
            {(void**)&d_neibrb,  static_cast<size_t>(nxy) * kNEWS * sizeof(int), "neibrb"},
            {(void**)&d_is_fuel, nn * sizeof(unsigned char),          "is_fuel"},
            {(void**)&d_phif,    nng * sizeof(double),                "phif"},
            {(void**)&d_phis,    nsg * sizeof(double),                "phis"},
            {(void**)&d_jnet,    nsg * sizeof(double),                "jnet"},
            {(void**)&d_xsdf,    nng * sizeof(double),                "xsdf"},
            {(void**)&d_xsrf,    nng * sizeof(double),                "xsrf"},
            {(void**)&d_xsnf,    nng * sizeof(double),                "xsnf"},
            {(void**)&d_xssm,    nn * ng * ng * sizeof(double),       "xssm"},
            {(void**)&d_chif,    nng * sizeof(double),                "chif"},
            {(void**)&d_crdf,    nng * sizeof(double),                "crdf"},
            {(void**)&d_phic,    nng * 4 * sizeof(double),            "phic"},
            {(void**)&d_p,       nng * 15 * sizeof(double),           "p"},
            {(void**)&d_a,       nng * 8 * sizeof(double),            "a"},
            {(void**)&d_c,       nng * 15 * sizeof(double),           "c"},
            {(void**)&d_q,       nng * 15 * sizeof(double),           "q"},
            {(void**)&d_l,       nng * 9 * sizeof(double),            "l"},
            {(void**)&d_bt,      nng * sizeof(double),                "bt"},
            {(void**)&d_phic_next, nng * 4 * sizeof(double),          "phic_next"},
            {(void**)&d_mrel,    nng * sizeof(double),                "mrel"},
            {(void**)&d_partials, static_cast<size_t>(4 * nchunk) * sizeof(double), "partials"},
            {(void**)&d_loop,    sizeof(PprLoopState),                "loop_state"},
            {(void**)&d_mismatch,
             (static_cast<size_t>(3 * nchunk) + 1) * sizeof(unsigned long long), "mismatch"},
        };
        for (const Alloc& a : allocs) {
            rc = cudaMalloc(a.p, a.bytes);
            if (rc != cudaSuccess) return fail(a.name, rc);
            ++n_allocations;
        }
        rc = cudaMallocHost((void**)&h_partials,
                            static_cast<size_t>(4 * nchunk) * sizeof(double));
        if (rc != cudaSuccess) return fail("cudaMallocHost(partials)", rc);
        ++n_allocations;
        rc = cudaMallocHost((void**)&h_loop, sizeof(PprLoopState));
        if (rc != cudaSuccess) return fail("cudaMallocHost(loop_state)", rc);
        ++n_allocations;
        rc = cudaMallocHost((void**)&h_mismatch, sizeof(unsigned long long));
        if (rc != cudaSuccess) return fail("cudaMallocHost(mismatch)", rc);
        ++n_allocations;
        return true;
    }

    /// WP6 stage D.  `fmap` and the pin FLUX map are NOT part of the shape:
    /// `print_opt.pin_flux` can turn on at one statepoint out of thirty-five,
    /// and making that a re-shape would drop every buffer and every graph for a
    /// print option.  They are allocated on first demand and kept.
    bool ensureReconShape(const ppr::ReconGeomView& g, const ppr::ReconStepView& st) {
        const int npina = g.npins * g.npins;
        if (recon_shaped && r_nxya == g.nxya && r_nz == g.nz && r_kbc == g.kbc &&
            r_kec == g.kec && r_ndiv == g.ndiv && r_npins == g.npins &&
            r_overlaps == g.n_overlaps && r_slots == st.n_form_slots) {
            return ensureReconFlux(g, st);
        }
        if (recon_shapes > 0) ++n_reallocations;
        ++recon_shapes;
        releaseRecon();
        // WP19.  Same rule as ensureShape: 23 cudaMalloc and a cudaMallocHost,
        // on a lane's thread, at a moment another lane chooses.
        rasbery::AllocWindow _alloc_window("ppr.recon.standup");
        r_nxya = g.nxya; r_nz = g.nz; r_kbc = g.kbc; r_kec = g.kec;
        r_ndiv = g.ndiv; r_npins = g.npins;
        r_overlaps = g.n_overlaps; r_slots = st.n_form_slots;

        const int    ndiv2 = g.ndiv * g.ndiv;
        const size_t nn    = static_cast<size_t>(nxyz);
        const size_t no    = static_cast<size_t>(g.n_overlaps);
        const size_t nplan = static_cast<size_t>(g.nz) * g.nxya;
        const size_t nmap  = nplan * npina;
        const size_t nrad  = static_cast<size_t>(g.nxya) * npina;

        struct Alloc { void** p; size_t bytes; const char* name; bool once; };
        const Alloc allocs[] = {
            {(void**)&d_latol,   static_cast<size_t>(g.nxya) * ndiv2 * sizeof(int), "latol", true},
            {(void**)&d_vol,     nn * sizeof(double),                 "vol",     true},
            {(void**)&d_hz,      static_cast<size_t>(g.nz) * sizeof(double), "hz", true},
            {(void**)&d_pin_off, (static_cast<size_t>(npina) + 1) * sizeof(int), "pin_off", true},
            {(void**)&d_ovl_di,  no * sizeof(int),                    "ovl_di",  true},
            {(void**)&d_ovl_dj,  no * sizeof(int),                    "ovl_dj",  true},
            {(void**)&d_ovl_dxh, no * sizeof(double),                 "ovl_dxh", true},
            {(void**)&d_ovl_dyh, no * sizeof(double),                 "ovl_dyh", true},
            {(void**)&d_q_xq,    no * 9 * sizeof(double),             "q_xq",    true},
            {(void**)&d_q_yq,    no * 9 * sizeof(double),             "q_yq",    true},
            {(void**)&d_q_wt,    no * 9 * sizeof(double),             "q_wt",    true},
            {(void**)&d_q_leg,   no * 9 * 15 * sizeof(double),        "q_leg",   true},
            {(void**)&d_gmap,    static_cast<size_t>(st.n_form_slots) * npina * sizeof(double),
             "gmap", true},
            {(void**)&d_xskf,    nn * ng * sizeof(double),            "xskf",    false},
            {(void**)&d_plane_lo,    nplan * sizeof(int),             "plane_lo",    false},
            {(void**)&d_plane_hi,    nplan * sizeof(int),             "plane_hi",    false},
            {(void**)&d_plane_alpha, nplan * sizeof(double),          "plane_alpha", false},
            {(void**)&d_pin_power,   nmap * sizeof(double),           "pin_power",   false},
            {(void**)&d_norm_partial, static_cast<size_t>(2 * nchunk) * sizeof(double),
             "norm_partial", false},
            {(void**)&d_peak_partial, static_cast<size_t>(2 * nchunk) * sizeof(double),
             "peak_partial", false},
            {(void**)&d_radial_power, nrad * sizeof(double),          "radial_power", false},
            {(void**)&d_radial_hz,    static_cast<size_t>(g.nxya) * sizeof(double),
             "radial_hz", false},
            {(void**)&d_scalars,      4 * sizeof(double),             "recon_scalars", false},
        };
        recon_static_bytes = 0;
        for (const Alloc& a : allocs) {
            const cudaError_t rc = cudaMalloc(a.p, a.bytes);
            if (rc != cudaSuccess) return fail(a.name, rc);
            ++n_allocations;
            if (a.once) recon_static_bytes += a.bytes;
        }
        const cudaError_t rc = cudaMallocHost((void**)&h_scalars, 4 * sizeof(double));
        if (rc != cudaSuccess) return fail("cudaMallocHost(recon_scalars)", rc);
        ++n_allocations;
        recon_shaped = true;
        return ensureReconFlux(g, st);
    }

    bool ensureReconFlux(const ppr::ReconGeomView& g, const ppr::ReconStepView& st) {
        if (!st.reconstruct_flux || d_fmap != nullptr) return true;
        // WP19.  Late and rare -- `print_opt.pin_flux` at one statepoint out of
        // thirty-five -- which makes it the WORST kind of unguarded allocation:
        // one that fires deep into a run, when every lane is capturing.
        rasbery::AllocWindow _alloc_window("ppr.recon.flux");
        const int    npina = g.npins * g.npins;
        const size_t nmap  = static_cast<size_t>(g.nz) * g.nxya * npina;
        cudaError_t  rc    = cudaMalloc((void**)&d_fmap,
                                        static_cast<size_t>(r_slots) * ng * npina * sizeof(double));
        if (rc != cudaSuccess) return fail("fmap", rc);
        ++n_allocations;
        rc = cudaMalloc((void**)&d_pin_flux, nmap * ng * sizeof(double));
        if (rc != cudaSuccess) return fail("pin_flux", rc);
        ++n_allocations;
        return true;
    }
};

PprBackend::PprBackend() : _impl(new Impl) {
    _impl->enabled = truthy(std::getenv("RASBERY_GPU_PPR"));
    // WP6 stage B.  The device loop is the DEFAULT inside an arm that is itself
    // default off; RASBERY_GPU_PPR_DEVICE_LOOP=0 restores c502856's per-round
    // host synchronise, which is the A/B control and nothing else.  The WHILE is
    // opt-in on top of it, for the same reason RASBERY_GPU_OUTER_GRAPH is: a
    // capture freezes the body's operands, which is a different KIND of claim
    // from "the same kernels in the same order".
    {
        const char* e = std::getenv("RASBERY_GPU_PPR_DEVICE_LOOP");
        const bool  device_loop = (e == nullptr) ? true : truthy(e);
        const bool  want_graph  = truthy(std::getenv("RASBERY_GPU_PPR_GRAPH"));
        _impl->arm_requested =
            !device_loop ? LoopArm::HostSync
                         : (want_graph ? LoopArm::DeviceGraph : LoopArm::DeviceStream);
        _impl->arm = _impl->arm_requested;
#if !RASBERY_HAS_PPR_WHILE
        if (_impl->arm == LoopArm::DeviceGraph) {
            _impl->arm           = LoopArm::DeviceStream;
            _impl->graph_refusal = "toolkit < 12.3 (no conditional nodes)";
        }
#endif
    }
    if (!_impl->enabled) {
        _impl->status_text = "off (RASBERY_GPU_PPR unset)";
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
    _impl->status_text = "on";
}

PprBackend::~PprBackend() = default;

bool PprBackend::available() const { return _impl->enabled && !_impl->failed; }

const std::string& PprBackend::status() const { return _impl->status_text; }

unsigned long long PprBackend::statepoints() const { return _impl->n_statepoints; }
unsigned long long PprBackend::iterations() const { return _impl->n_iterations; }
double             PprBackend::wallMs() const { return _impl->wall_ms; }
int                PprBackend::deviceOrdinal() const { return _impl->device; }

const char*        PprBackend::loopArm() const { return loopArmName(_impl->arm); }
unsigned long long PprBackend::hostSyncs() const { return _impl->n_host_syncs; }
double             PprBackend::hostSyncsPerStatepoint() const {
    return (_impl->n_statepoints == 0)
               ? 0.0
               : static_cast<double>(_impl->n_host_syncs) /
                     static_cast<double>(_impl->n_statepoints);
}
unsigned long long PprBackend::graphLaunches() const { return _impl->n_graph_launches; }
unsigned long long PprBackend::graphBuilds() const { return _impl->n_graph_builds; }
const std::string& PprBackend::graphRefusal() const { return _impl->graph_refusal; }
unsigned long long PprBackend::h2dBytes() const { return _impl->n_h2d_bytes; }
unsigned long long PprBackend::h2dBytesElided() const { return _impl->n_h2d_elided; }
unsigned long long PprBackend::d2hBytes() const { return _impl->n_d2h_bytes; }
unsigned long long PprBackend::canonicalStatepoints() const { return _impl->n_canonical_sp; }
unsigned long long PprBackend::canonicalMismatch() const { return _impl->n_canonical_bad; }
void PprBackend::noteReconRepair() {
    if (_impl->coeffs_device_only) ++_impl->n_recon_repairs;
    _impl->coeffs_device_only = false;
}
unsigned long long PprBackend::reconRepairs() const { return _impl->n_recon_repairs; }
unsigned long long PprBackend::reconStatepoints() const { return _impl->n_recon_statepoints; }
unsigned long long PprBackend::pinMaterializations() const {
    return _impl->n_pin_materializations;
}
const std::string& PprBackend::reconRefusal() const { return _impl->recon_refusal; }
unsigned long long PprBackend::allocations() const { return _impl->n_allocations; }
unsigned long long PprBackend::reallocations() const { return _impl->n_reallocations; }

// --- WP6 stage F: the refusal ladder ---------------------------------------

void PprBackend::noteHostFallback(ppr::Refusal reason) { _impl->noteRefusal(reason); }

ppr::Refusal PprBackend::lastRefusal() const { return _impl->last_refusal; }

const char* PprBackend::lastRefusalName() const {
    return ppr::refusalName(_impl->last_refusal);
}

unsigned long long PprBackend::refusalCount(ppr::Refusal reason) const {
    const int i = static_cast<int>(reason);
    if (i < 0 || i >= static_cast<int>(ppr::Refusal::Count)) return 0;
    return _impl->n_refusals[i];
}

std::string PprBackend::refusalJson() const {
    // ONLY THE RUNGS THAT FIRED.  A fixed seven-key object would put six zeros
    // in every receipt and make the one number that matters harder to find;
    // `{}` is the shape of a run that never fell back, which is the shape the
    // campaign is trying to reach.
    std::string out = "{";
    bool        first = true;
    for (int i = 1; i < static_cast<int>(ppr::Refusal::Count); ++i) {
        if (_impl->n_refusals[i] == 0) continue;
        if (!first) out += ",";
        first = false;
        out += "\"";
        out += ppr::refusalName(static_cast<ppr::Refusal>(i));
        out += "\":";
        out += std::to_string(_impl->n_refusals[i]);
    }
    out += "}";
    return out;
}


bool PprBackend::resetAndDrive(const ppr::GeomView& geom, const ppr::StepView& step,
                               int niter, int* iters) {
    Impl& s = *_impl;
    // THE LADDER IS EXHAUSTIVE.  Every `return false` from here to the success
    // return names its rung, either directly or through fail(), so
    // `host_fallbacks:35` in the receipt is always accompanied by which of the
    // seven reasons produced it.
    if (!s.enabled) {
        s.noteRefusal(ppr::Refusal::ArmOff);
        return false;
    }
    if (s.failed) {
        s.noteRefusal(ppr::Refusal::BackendDisabled);
        return false;
    }
    // Lowered FIRST, raised only on the success return: every early exit below
    // therefore leaves the reconstruction arm refusing, which is what stops it
    // reading coefficients this statepoint did not write.
    s.last_drive_ok = false;
    if (geom.ng != 2) {
        s.status_text = "declined: ng != 2 (updateSource body is 2-group)";
        s.noteRefusal(ppr::Refusal::NotTwoGroup);
        return false;
    }
    if (niter <= 0) {
        s.noteRefusal(ppr::Refusal::NonPositiveIter);
        return false;
    }
    if (!s.ensureShape(geom)) {
        // ensureShape's own failures go through fail(), which already recorded
        // CudaFailure; this rung is the one it cannot reach -- a shape that
        // stood nothing up without a cudaError_t to blame.
        if (s.last_refusal != ppr::Refusal::CudaFailure)
            s.noteRefusal(ppr::Refusal::ShapeAllocFail);
        return false;
    }

    const size_t nn  = static_cast<size_t>(geom.nxyz);
    const size_t nng = nn * geom.ng;
    const size_t nsg = static_cast<size_t>(geom.nsurf) * geom.ng;

    cudaError_t rc = cudaSuccess;
    auto        h2d = [&](void* d, const void* h, size_t bytes, const char* name) -> bool {
        rc = rasbery::xfer::memcpyAsync("CudaPprBackend.cu:drive", name, d, h, bytes,
                                        cudaMemcpyHostToDevice, s.stream);
        if (rc != cudaSuccess) return s.fail(name, rc);
        s.n_h2d_bytes += bytes;
        return true;
    };

    if (!s.geometry_uploaded) {
        if (!h2d(s.d_hmesh, geom.hmesh, nn * kNDIRMAX * sizeof(double), "H2D hmesh")) return false;
        if (!h2d(s.d_lktosfc, geom.lktosfc, nn * kNDIRMAX * kLR * sizeof(int), "H2D lktosfc")) return false;
        if (!h2d(s.d_neibrb, geom.neibrb, static_cast<size_t>(geom.nxy) * kNEWS * sizeof(int), "H2D neibrb")) return false;
        if (!h2d(s.d_is_fuel, geom.is_fuel, nn * sizeof(unsigned char), "H2D is_fuel")) return false;
        s.geometry_uploaded = true;
    }

    // --- WP6 stage C: which of the three nodal arrays are borrowed ----------
    //
    // ALL THREE OR NONE.  A borrowed jnet paired with an uploaded phif is two
    // different outer iterations blended into one reconstruction, which is the
    // shape gpu::canonicalNodalSetIsCoherent refuses one layer up; refusing it
    // again here means the rule holds even if a future caller forgets it.
    const bool have_set = step.dev_phif != nullptr && step.dev_phis != nullptr &&
                          step.dev_jnet != nullptr;
    const bool borrow = step.canonical != ppr::CanonicalMode::Off && have_set;
    const bool verify = borrow && step.canonical == ppr::CanonicalMode::Verify;

    // WP6 stage F.  WHICH SCHEME, decided ONCE and read by everything below:
    // the reset half's tail (kMasterEven), the round body, the post-loop cross
    // terms and the D2H list.  A per-kernel branch would be four chances to
    // disagree about what this statepoint is.
    const bool master = step.mode_master;

    const size_t nodal_bytes = (nng + 2 * nsg) * sizeof(double);
    if (!borrow || verify) {
        if (!h2d(s.d_phif, step.phif, nng * sizeof(double), "H2D phif")) return false;
        if (!h2d(s.d_phis, step.phis, nsg * sizeof(double), "H2D phis")) return false;
        if (!h2d(s.d_jnet, step.jnet, nsg * sizeof(double), "H2D jnet")) return false;
    } else {
        s.n_h2d_elided += nodal_bytes;
    }

    if (!h2d(s.d_xsdf, step.xsdf, nng * sizeof(double), "H2D xsdf")) return false;
    if (!h2d(s.d_xsrf, step.xsrf, nng * sizeof(double), "H2D xsrf")) return false;
    if (!h2d(s.d_xsnf, step.xsnf, nng * sizeof(double), "H2D xsnf")) return false;
    if (!h2d(s.d_xssm, step.xssm, nn * geom.ng * geom.ng * sizeof(double), "H2D xssm")) return false;

    // chif and crdf do NOT move every statepoint (the fission spectrum moves
    // when the library reference blocks are rebuilt; crdf is all-ones unless
    // RASBERY_PPR_CRDF).  A generation of 0 means the caller cannot vouch for
    // it, which uploads -- the conservative direction.
    if (step.chif != nullptr) {
        const bool stale = (step.chif_generation == 0) || (s.chif_gen_seen == 0) ||
                           (step.chif_generation != s.chif_gen_seen);
        if (stale) {
            if (!h2d(s.d_chif, step.chif, nng * sizeof(double), "H2D chif")) return false;
            s.chif_gen_seen = step.chif_generation;
        } else {
            s.n_h2d_elided += nng * sizeof(double);
        }
    }
    {
        const bool stale = (step.crdf_generation == 0) || (s.crdf_gen_seen == 0) ||
                           (step.crdf_generation != s.crdf_gen_seen);
        if (stale) {
            if (!h2d(s.d_crdf, step.crdf, nng * sizeof(double), "H2D crdf")) return false;
            s.crdf_gen_seen = step.crdf_generation;
        } else {
            s.n_h2d_elided += nng * sizeof(double);
        }
    }

    // --- the per-statepoint scalars, and the loop's own state ---------------
    s.h_loop->reigv     = step.reigv;
    s.h_loop->iters     = 0;
    s.h_loop->converged = 0;
    s.h_loop->niter     = niter;
    s.h_loop->pad       = 0;
    for (int i = 0; i < 4; ++i) {
        s.h_loop->prev[i] = 0.0;
        s.h_loop->cur[i]  = 0.0;
    }
    if (!h2d(s.d_loop, s.h_loop, sizeof(PprLoopState), "H2D loop state")) return false;

    DevCtx& x   = s.ctx;
    x.ng        = geom.ng;
    x.nxyz      = geom.nxyz;
    x.nxy       = geom.nxy;
    x.nsurf     = geom.nsurf;
    x.has_chif  = (step.chif != nullptr) ? 1 : 0;
    x.nchunk    = s.nchunk;
    x.loop      = s.d_loop;
    x.partials  = s.d_partials;
    x.hmesh     = s.d_hmesh;
    x.lktosfc   = s.d_lktosfc;
    x.neibrb    = s.d_neibrb;
    x.is_fuel   = s.d_is_fuel;
    x.phif      = borrow ? step.dev_phif : s.d_phif;
    x.phis      = borrow ? step.dev_phis : s.d_phis;
    x.jnet      = borrow ? step.dev_jnet : s.d_jnet;
    x.xsdf      = s.d_xsdf;
    x.xsrf      = s.d_xsrf;
    x.xsnf      = s.d_xsnf;
    x.xssm      = s.d_xssm;
    x.chif      = s.d_chif;
    x.crdf      = s.d_crdf;
    x.phic      = s.d_phic;
    x.p         = s.d_p;
    x.a         = s.d_a;
    x.c         = s.d_c;
    x.q         = s.d_q;
    x.l         = s.d_l;
    x.bt        = s.d_bt;
    x.phic_next = s.d_phic_next;
    x.mrel      = s.d_mrel;
    x.mode_master = step.mode_master ? 1 : 0;
    x.c_ro      = s.d_c;

    // 128, not 256: the whole grid is nxyz*ng = ~17k threads, so at 256 the
    // 1080 Ti's 28 SMs get 2.4 blocks each and the tail wastes a third of the
    // launch.  Halving the block halves the tail, and the register-heavy fused
    // kernel prefers the smaller block anyway.
    const int threads   = 128;
    const int blocks_ng = static_cast<int>((nng + threads - 1) / threads);
    const int blocks_n  = static_cast<int>((nn + threads - 1) / threads);
    const int blocks_ch = (s.nchunk + threads - 1) / threads;

    cudaEventRecord(s.ev_start, s.stream);

    // WP6 stage C, verify mode: compare the borrowed buffers against the host
    // arrays (already uploaded into this instance's own block above) BEFORE
    // anything reads them, so a run that reports a mismatch also reports the
    // physics that mismatch produced rather than hiding one behind the other.
    if (verify) {
        kCanonicalCompare<<<blocks_ch, threads, 0, s.stream>>>(
            step.dev_phif, s.d_phif, static_cast<long long>(nng), s.nchunk, 0, s.d_mismatch);
        kCanonicalCompare<<<blocks_ch, threads, 0, s.stream>>>(
            step.dev_phis, s.d_phis, static_cast<long long>(nsg), s.nchunk, 1, s.d_mismatch);
        kCanonicalCompare<<<blocks_ch, threads, 0, s.stream>>>(
            step.dev_jnet, s.d_jnet, static_cast<long long>(nsg), s.nchunk, 2, s.d_mismatch);
        kFoldMismatch<<<1, 1, 0, s.stream>>>(s.d_mismatch, 3 * s.nchunk);
        if ((rc = cudaGetLastError()) != cudaSuccess) return s.fail("canonical compare", rc);
    }

    // reset()
    kBuckling<<<blocks_ng, threads, 0, s.stream>>>(x);
    kCornerInit<<<blocks_ng, threads, 0, s.stream>>>(x);
    kFit<<<blocks_ng, threads, 0, s.stream>>>(x);
    kAxialLeakage<<<blocks_ng, threads, 0, s.stream>>>(x);
    kUpdateSource<false><<<blocks_n, threads, 0, s.stream>>>(x);

    // MASTER mode still runs the WHOLE of reset() -- PPR::reset has no mode
    // branch, and driveMaster consumes the corner flux kCornerInit seeds --
    // then overwrites every one of the 15 `c` slots kFit just wrote.  Doing
    // less would be a different starting iterate than the host's.
    if (master) kMasterEven<<<blocks_ng, threads, 0, s.stream>>>(x);

    if ((rc = cudaGetLastError()) != cudaSuccess) return s.fail("reset kernels", rc);

    // --- drive(niter) -------------------------------------------------------
    //
    // ONE Picard round, spelled once.  All three arms enqueue exactly this and
    // differ only in WHO decides to run it again: the host after a synchronise,
    // the enqueue count with the device flag making the surplus rounds no-ops,
    // or the WHILE's predicate.
    //
    // WP6 stage F.  MASTER mode's round is a DIFFERENT body -- the CPB sweep,
    // its commit and its max-fold -- selected by `master` above and never
    // inside a kernel, so neither scheme pays for the other's branch and a
    // captured graph can only ever replay the one it was captured for.
    auto enqueue_round = [&](cudaStream_t st, bool guarded) {
        if (master) {
            if (guarded) {
                kMasterCpb<true><<<blocks_ng, threads, 0, st>>>(x);
                kMasterCommit<true><<<blocks_ch, threads, 0, st>>>(x);
                kMasterFoldAndCheck<<<1, 1, 0, st>>>(x);
            } else {
                kMasterCpb<false><<<blocks_ng, threads, 0, st>>>(x);
                kMasterCommit<false><<<blocks_ch, threads, 0, st>>>(x);
            }
            return;
        }
        if (guarded) {
            for (int f = 0; f < kSourceSweepsPerIteration; ++f) {
                kUpdateFused<true><<<blocks_ng, threads, 0, st>>>(x);
                kUpdateSource<true><<<blocks_n, threads, 0, st>>>(x);
            }
            kUpdateCorner<true><<<blocks_ng, threads, 0, st>>>(x);
            kCornerPartials<true><<<blocks_ch, threads, 0, st>>>(x);
            kCornerFoldAndCheck<<<1, 4, 0, st>>>(x);
        } else {
            for (int f = 0; f < kSourceSweepsPerIteration; ++f) {
                kUpdateFused<false><<<blocks_ng, threads, 0, st>>>(x);
                kUpdateSource<false><<<blocks_n, threads, 0, st>>>(x);
            }
            kUpdateCorner<false><<<blocks_ng, threads, 0, st>>>(x);
            kCornerPartials<false><<<blocks_ch, threads, 0, st>>>(x);
        }
    };

    int iters_done = 0;

    if (s.arm == LoopArm::HostSync && master) {
        // The A/B control for the master arm: the same round, with the host
        // folding the per-chunk maxima and applying driveMaster's own test.
        for (int citer = 0; citer < niter; ++citer) {
            enqueue_round(s.stream, false);

            rc = rasbery::xfer::memcpyAsync(
                "CudaPprBackend.cu:drive", "master partials", s.h_partials, s.d_partials,
                static_cast<size_t>(s.nchunk) * sizeof(double), cudaMemcpyDeviceToHost,
                s.stream);
            if (rc != cudaSuccess) return s.fail("D2H master partials", rc);
            s.n_d2h_bytes += static_cast<size_t>(s.nchunk) * sizeof(double);
            if (!s.syncStream("master drive sync")) return false;

            double maxrel = 0.0;
            for (int c = 0; c < s.nchunk; ++c)
                maxrel = (s.h_partials[c] > maxrel) ? s.h_partials[c] : maxrel;

            iters_done = citer + 1;
            if (maxrel < kCornerFluxTolerance) break;
        }
    } else if (s.arm == LoopArm::HostSync) {
        // c502856, verbatim: the host folds the partials and applies the test.
        double prev_nw = 0.0, prev_sw = 0.0, prev_ne = 0.0, prev_se = 0.0;
        for (int citer = 0; citer < niter; ++citer) {
            enqueue_round(s.stream, false);

            rc = rasbery::xfer::memcpyAsync(
                "CudaPprBackend.cu:drive", "corner partials", s.h_partials, s.d_partials,
                static_cast<size_t>(4 * s.nchunk) * sizeof(double), cudaMemcpyDeviceToHost,
                s.stream);
            if (rc != cudaSuccess) return s.fail("D2H corner partials", rc);
            s.n_d2h_bytes += static_cast<size_t>(4 * s.nchunk) * sizeof(double);
            if (!s.syncStream("drive sync")) return false;

            double nw = 0.0, sw = 0.0, ne = 0.0, se = 0.0;
            for (int c = 0; c < s.nchunk; ++c) nw += s.h_partials[0 * s.nchunk + c];
            for (int c = 0; c < s.nchunk; ++c) sw += s.h_partials[1 * s.nchunk + c];
            for (int c = 0; c < s.nchunk; ++c) ne += s.h_partials[2 * s.nchunk + c];
            for (int c = 0; c < s.nchunk; ++c) se += s.h_partials[3 * s.nchunk + c];

            const double err_nw = relativeChange(nw, prev_nw);
            const double err_sw = relativeChange(sw, prev_sw);
            const double err_ne = relativeChange(ne, prev_ne);
            const double err_se = relativeChange(se, prev_se);

            iters_done = citer + 1;
            if (err_nw < kCornerFluxTolerance && err_sw < kCornerFluxTolerance &&
                err_ne < kCornerFluxTolerance && err_se < kCornerFluxTolerance)
                break;

            prev_nw = nw;
            prev_sw = sw;
            prev_ne = ne;
            prev_se = se;
        }
    } else {
#if RASBERY_HAS_PPR_WHILE
        if (s.arm == LoopArm::DeviceGraph) {
            // The key is the body's baked operands.  A borrow that engaged (or
            // stopped engaging) between statepoints moves x.phif/phis/jnet, and
            // replaying a graph captured with the old pointers would read the
            // wrong buffer with every value finite.
            if (s.graph_valid && std::memcmp(&s.graph_ctx, &x, sizeof(DevCtx)) != 0)
                s.releaseGraph();
            if (!s.graph_valid) {
                if (s.graph_root_stream == nullptr) {
                    // WP19.  Stream creation is a stand-up call like any other
                    // and it happens once, on the first statepoint -- the same
                    // window the PPR buffers are allocated in.
                    rasbery::AllocWindow _alloc_window("ppr.graph.root_stream");
                    if ((rc = cudaStreamCreate(&s.graph_root_stream)) != cudaSuccess)
                        return s.fail("cudaStreamCreate(graph root)", rc);
                }
                const char* stage = "?";
                cudaGraph_t root  = nullptr;
                cudaGraphExec_t exec = nullptr;
                // WP19, THE GAP ITSELF.  This was the ONE capture in the tree
                // outside the arbiter: two BeginCaptures (root, and
                // BeginCaptureToGraph on the segment stream) with nothing
                // holding the process quiet around them, in the arm the
                // production v6 env turns on.  The window below is what every
                // other capture site has had since Rev.7.1 Task 18d.
                auto build = [&]() {
                    rasbery::CaptureWindow _capture_window(s.stream, "ppr.while");
                    return buildPprWhile(
                        s.graph_root_stream, s.stream, s.d_loop, &stage,
                        [&](cudaStream_t bs) {
                            enqueue_round(bs, true);
                            return cudaGetLastError() == cudaSuccess;
                        },
                        &root, &exec);
                };
                cudaError_t brc = build();
                // THE RETRY, AND WHY IT IS ONE AND NOT A LOOP.  A capture-
                // illegal code says another thread was in the window, not that
                // this graph is unbuildable, so the same build under a quiet
                // process is a different experiment -- worth exactly one run.
                // A second failure is not a race any more and is reported as
                // such rather than retried into a hang.  cudaGetLastError()
                // between the two clears the sticky error the first attempt
                // left, or the retry inherits the corpse.
                //
                // WP19.1, AND WHY THIS RETRY STAYS UNCONDITIONAL WHILE THE
                // OUTER WHILE'S DOES NOT.  The retry is sound exactly when
                // re-running the body's recorder moves no host state.  Here the
                // recorder is `enqueue_round`, which is nothing but kernel
                // launches on the body stream: every H2D this statepoint needs
                // (phif/phis/jnet, the four XS blocks, chif/crdf and the loop
                // scalars) was issued and its generation memo taken BEFORE the
                // build, outside the capture, so a second record cannot elide
                // an upload that never landed.  CudaOuterGraph.cu's body is
                // `runOneOuter`, which does move shadows -- hence the stage gate
                // there and none here.  If enqueue_round ever grows an upload,
                // this retry needs that gate too.
                if (rasbery::captureIllegal(static_cast<int>(brc))) {
                    // `refusals[CaptureRaceRetry]` is the COUNTER the receipt
                    // reports; `last_refusal` is why the LAST statepoint fell
                    // back, and a retry that worked did not fall back -- so the
                    // rung is counted and the ladder's tip is put back.
                    const ppr::Refusal prev = s.last_refusal;
                    cudaGetLastError();
                    // WP19.1: with the site, the stage and the slot on it, so a
                    // process-wide count of 1 is no longer the only evidence.
                    // -1 for the slot: a PPR backend is one per Driver and does
                    // not carry the arena slot id (the receipt's `slot` is
                    // stamped by the caller), so the honest value is "unknown"
                    // rather than a number this object does not own.
                    rasbery::noteCaptureRaceRetry("ppr.while", stage,
                                                  static_cast<int>(brc), -1);
                    s.noteRefusal(ppr::Refusal::CaptureRaceRetry);
                    root = nullptr;
                    exec = nullptr;
                    brc  = build();
                    if (brc == cudaSuccess) {
                        s.last_refusal = prev;
                    } else if (rasbery::captureIllegal(static_cast<int>(brc))) {
                        rasbery::noteCaptureRaceUnrecovered("ppr.while",
                                                            static_cast<int>(brc),
                                                            cudaGetErrorString(brc));
                    }
                }
                if (brc == cudaSuccess) {
                    s.graph_root  = root;
                    s.graph_exec  = exec;
                    s.graph_ctx   = x;
                    s.graph_valid = true;
                    ++s.n_graph_builds;
                } else {
                    // A REFUSAL IS NOT A FAILURE.  Capture is the only claim in
                    // this arm that the stream arm does not also make, so the
                    // stream arm is what a refusal falls back to -- for the rest
                    // of the run, named, and never to the host.
                    s.graph_refusal = std::string(stage) + ": " + cudaGetErrorString(brc);
                    s.arm           = LoopArm::DeviceStream;
                    cudaGetLastError();
                }
            }
            if (s.graph_valid) {
                rc = cudaGraphLaunch(s.graph_exec, s.stream);
                if (rc != cudaSuccess) return s.fail("cudaGraphLaunch(ppr while)", rc);
                ++s.n_graph_launches;
            }
        }
#endif
        if (s.arm == LoopArm::DeviceStream) {
            for (int citer = 0; citer < niter; ++citer) enqueue_round(s.stream, true);
        }
        if ((rc = cudaGetLastError()) != cudaSuccess) return s.fail("drive kernels", rc);
    }

    // driveMaster step 3, OUTSIDE the loop and therefore unguarded: it must run
    // on the converged corners whether the break fired at round 3 or the cap
    // was reached, which is exactly what the host does after its `break`.
    if (master) {
        kMasterCross<<<blocks_ng, threads, 0, s.stream>>>(x);
        if ((rc = cudaGetLastError()) != cudaSuccess) return s.fail("master cross terms", rc);
    }

    // Only what the host reconstruction reads comes back; phic follows so the
    // host arrays describe one consistent state if anything ever inspects them.
    //
    // A failure PART WAY THROUGH these copies leaves the host arrays half
    // device / half stale -- and that is safe, because the caller's response to
    // `false` is the untouched host reset() + drive() pair, which rewrites every
    // one of the seven (reset: bt, phic, c, l, q; drive: p, a, c) before
    // reconstructPinPower reads any of them.  There is no partial state to
    // repair, so there is no repair path to get wrong.
    auto d2h = [&](void* h, const void* d, size_t bytes, const char* name) -> bool {
        rc = rasbery::xfer::memcpyAsync("CudaPprBackend.cu:drive", name, h, d, bytes,
                                        cudaMemcpyDeviceToHost, s.stream);
        if (rc != cudaSuccess) return s.fail(name, rc);
        s.n_d2h_bytes += bytes;
        return true;
    };
    // MASTER mode NEVER WRITES p OR a, on either arm: driveMaster has no
    // particular/homogeneous split and reset() does not touch them.  The host
    // arrays therefore hold whatever Geometry stood up, and the device buffers
    // hold nothing at all -- copying them back would replace one with the
    // other and both are equally unread.  So the master D2H is five arrays,
    // not seven, and that is the host's own state, not an optimisation.
    const size_t coeff_terms = master ? (15 + 1 + 4 + 15 + 9) : (15 + 8 + 15 + 1 + 4 + 15 + 9);
    const size_t coeff_bytes = nng * coeff_terms * sizeof(double);
    s.coeffs_device_only = step.coefficients_stay_on_device;
    if (!s.coeffs_device_only) {
        if (!master) {
            if (!d2h(step.p, s.d_p, nng * 15 * sizeof(double), "D2H p")) return false;
            if (!d2h(step.a, s.d_a, nng * 8 * sizeof(double), "D2H a")) return false;
        }
        if (!d2h(step.c, s.d_c, nng * 15 * sizeof(double), "D2H c")) return false;
        if (!d2h(step.bt, s.d_bt, nng * sizeof(double), "D2H bt")) return false;
        if (!d2h(step.phic, s.d_phic, nng * 4 * sizeof(double), "D2H phic")) return false;
        if (!d2h(step.q, s.d_q, nng * 15 * sizeof(double), "D2H q")) return false;
        if (!d2h(step.l, s.d_l, nng * 9 * sizeof(double), "D2H l")) return false;
    } else {
        s.n_d2h_elided += coeff_bytes;
    }
    // The round count rides the same batch: on the device arms it is the only
    // way the host learns how many rounds ran, and it costs no extra sync.
    if (!d2h(s.h_loop, s.d_loop, sizeof(PprLoopState), "D2H loop state")) return false;
    if (verify &&
        !d2h(s.h_mismatch, s.d_mismatch + 3 * s.nchunk, sizeof(unsigned long long),
             "D2H canonical mismatch"))
        return false;

    cudaEventRecord(s.ev_stop, s.stream);
    if (!s.syncStream("final sync")) return false;

    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, s.ev_start, s.ev_stop) == cudaSuccess)
        s.wall_ms += static_cast<double>(ms);

    if (s.arm != LoopArm::HostSync) iters_done = s.h_loop->iters;
    if (borrow) ++s.n_canonical_sp;
    if (verify) s.n_canonical_bad += *s.h_mismatch;

    ++s.n_statepoints;
    s.n_iterations += static_cast<unsigned long long>(iters_done);
    if (iters) *iters = iters_done;
    s.last_drive_ok = true;
    return true;
}

// ---------------------------------------------------------------------------
// WP6 stage D: reconstructPinPower on the device
// ---------------------------------------------------------------------------

bool PprBackend::reconstructPinPower(const ppr::ReconGeomView& geom,
                                     const ppr::ReconStepView& step) {
    Impl& s = *_impl;
    if (!s.enabled) {
        s.recon_refusal = "off (RASBERY_GPU_PPR unset)";
        return false;
    }
    if (s.failed) {
        s.recon_refusal = "backend disabled after an earlier CUDA failure";
        return false;
    }
    // ONLY AFTER A DEVICE DRIVE.  The kernels read `p`, `a` and `bt` as the
    // previous call left them; a statepoint whose drive fell back to the host
    // has host coefficients and device coefficients one statepoint stale, and
    // reconstructing from the stale ones is the failure this guard exists for.
    if (!s.last_drive_ok) {
        s.recon_refusal = "previous drive did not run on the device";
        return false;
    }
    if (!ppr::reconEnabledFromEnv()) {
        s.recon_refusal = "off (RASBERY_GPU_PPR_RECON unset)";
        return false;
    }
    if (!s.ensureReconShape(geom, step)) {
        if (s.recon_refusal.empty())
            s.recon_refusal = "reconstruction buffers could not be stood up";
        return false;
    }

    cudaError_t rc = cudaSuccess;
    auto        h2d = [&](void* d, const void* h, size_t bytes, const char* name) -> bool {
        rc = rasbery::xfer::memcpyAsync("CudaPprBackend.cu:reconstructPinPower", name, d,
                                        h, bytes, cudaMemcpyHostToDevice, s.stream);
        if (rc != cudaSuccess) return s.fail(name, rc);
        s.n_h2d_bytes += bytes;
        return true;
    };

    const size_t nn    = static_cast<size_t>(s.nxyz);
    const int    ndiv2 = geom.ndiv * geom.ndiv;
    const int    npina = geom.npins * geom.npins;
    const size_t nplan = static_cast<size_t>(geom.nz) * geom.nxya;
    const size_t nmap  = nplan * npina;

    if (!s.recon_static_uploaded) {
        if (!h2d(s.d_latol, geom.latol,
                 static_cast<size_t>(geom.nxya) * ndiv2 * sizeof(int), "H2D latol")) return false;
        if (!h2d(s.d_vol, geom.vol, nn * sizeof(double), "H2D vol")) return false;
        if (!h2d(s.d_hz, geom.hz, static_cast<size_t>(geom.nz) * sizeof(double), "H2D hz"))
            return false;
        const size_t no = static_cast<size_t>(geom.n_overlaps);
        if (!h2d(s.d_pin_off, geom.pin_off, (static_cast<size_t>(npina) + 1) * sizeof(int),
                 "H2D pin_off")) return false;
        if (!h2d(s.d_ovl_di, geom.ovl_di, no * sizeof(int), "H2D ovl_di")) return false;
        if (!h2d(s.d_ovl_dj, geom.ovl_dj, no * sizeof(int), "H2D ovl_dj")) return false;
        if (!h2d(s.d_ovl_dxh, geom.ovl_dxh, no * sizeof(double), "H2D ovl_dxh")) return false;
        if (!h2d(s.d_ovl_dyh, geom.ovl_dyh, no * sizeof(double), "H2D ovl_dyh")) return false;
        if (!h2d(s.d_q_xq, geom.q_xq, no * 9 * sizeof(double), "H2D q_xq")) return false;
        if (!h2d(s.d_q_yq, geom.q_yq, no * 9 * sizeof(double), "H2D q_yq")) return false;
        if (!h2d(s.d_q_wt, geom.q_wt, no * 9 * sizeof(double), "H2D q_wt")) return false;
        if (!h2d(s.d_q_leg, geom.q_leg, no * 9 * 15 * sizeof(double), "H2D q_leg")) return false;
        if (!h2d(s.d_gmap, step.gmap,
                 static_cast<size_t>(step.n_form_slots) * npina * sizeof(double), "H2D gmap"))
            return false;
        if (s.d_fmap != nullptr && step.fmap != nullptr &&
            !h2d(s.d_fmap, step.fmap,
                 static_cast<size_t>(step.n_form_slots) * s.ng * npina * sizeof(double),
                 "H2D fmap"))
            return false;
        s.recon_static_uploaded = true;
    } else {
        // The table, the map and the registry are library/geometry data: after
        // the first statepoint every byte of them is elided, for the life of the
        // run and of the slot.
        s.n_h2d_elided += s.recon_static_bytes;
    }

    if (!h2d(s.d_xskf, step.xskf, nn * s.ng * sizeof(double), "H2D xskf")) return false;
    if (!h2d(s.d_plane_lo, step.plane_lo, nplan * sizeof(int), "H2D plane_lo")) return false;
    if (!h2d(s.d_plane_hi, step.plane_hi, nplan * sizeof(int), "H2D plane_hi")) return false;
    if (!h2d(s.d_plane_alpha, step.plane_alpha, nplan * sizeof(double), "H2D plane_alpha"))
        return false;

    ppr::ReconCtx r{};
    r.ng               = s.ng;
    r.nxyz             = s.nxyz;
    r.nxy              = s.nxy;
    r.nxya             = geom.nxya;
    r.nz               = geom.nz;
    r.kbc              = geom.kbc;
    r.kec              = geom.kec;
    r.ndiv             = geom.ndiv;
    r.ndiv2            = ndiv2;
    r.npins            = geom.npins;
    r.npina            = npina;
    r.nchunk           = s.nchunk;
    r.reconstruct_flux = step.reconstruct_flux ? 1 : 0;
    // WP6 stage F.  MASTER mode reads `c` where SENM reads p/a/bt, and reads
    // nothing else differently -- same overlaps, same quadrature, same
    // normalisation, same peaks.
    r.mode_master      = step.mode_master ? 1 : 0;
    r.c                = s.d_c;
    r.latol            = s.d_latol;
    r.vol              = s.d_vol;
    r.hz               = s.d_hz;
    r.is_fuel          = s.d_is_fuel;
    r.pin_off          = s.d_pin_off;
    r.ovl_di           = s.d_ovl_di;
    r.ovl_dj           = s.d_ovl_dj;
    r.ovl_dxh          = s.d_ovl_dxh;
    r.ovl_dyh          = s.d_ovl_dyh;
    r.q_xq             = s.d_q_xq;
    r.q_yq             = s.d_q_yq;
    r.q_wt             = s.d_q_wt;
    r.q_leg            = s.d_q_leg;
    r.p                = s.d_p;
    r.a                = s.d_a;
    r.bt               = s.d_bt;
    // The SAME flux the drive read -- borrowed or uploaded.  Reading it out of
    // the DevCtx rather than re-deriving it is what keeps the normalisation's
    // `_phif` and the expansion's the same array under either stage-C mode.
    r.phif             = s.ctx.phif;
    r.xskf             = s.d_xskf;
    r.gmap             = s.d_gmap;
    r.fmap             = step.reconstruct_flux ? s.d_fmap : nullptr;
    r.plane_lo         = s.d_plane_lo;
    r.plane_hi         = s.d_plane_hi;
    r.plane_alpha      = s.d_plane_alpha;
    r.pin_power        = s.d_pin_power;
    r.pin_flux         = step.reconstruct_flux ? s.d_pin_flux : nullptr;
    r.norm_partial     = s.d_norm_partial;
    r.peak_partial     = s.d_peak_partial;
    r.radial_power     = s.d_radial_power;
    r.radial_hz        = s.d_radial_hz;
    r.scalars          = s.d_scalars;

    // The host's `std::fill_n(ppower, ..., 0.0)` over the WHOLE map, including
    // the planes outside [kbc, kec) that no kernel below touches.  0.0 is
    // all-zero bytes, so a memset is that fill exactly.
    rc = cudaMemsetAsync(s.d_pin_power, 0, nmap * sizeof(double), s.stream);
    if (rc != cudaSuccess) return s.fail("memset pin_power", rc);
    if (step.reconstruct_flux) {
        rc = cudaMemsetAsync(s.d_pin_flux, 0, nmap * s.ng * sizeof(double), s.stream);
        if (rc != cudaSuccess) return s.fail("memset pin_flux", rc);
    }

    const int       threads = 128;
    const long long npin_t  = static_cast<long long>(geom.kec - geom.kbc) * geom.nxya * npina;
    const int       b_pin   = static_cast<int>((npin_t + threads - 1) / threads);
    const int       b_ch    = (s.nchunk + threads - 1) / threads;
    const int       b_asm   = (geom.nxya + threads - 1) / threads;
    const long long nrad    = static_cast<long long>(geom.nxya) * npina;
    const int       b_rad   = static_cast<int>((nrad + threads - 1) / threads);

    ppr::kReconPins<<<b_pin, threads, 0, s.stream>>>(r);
    ppr::kReconNormPartials<<<b_ch, threads, 0, s.stream>>>(r);
    ppr::kReconNormFold<<<1, 1, 0, s.stream>>>(r);
    ppr::kReconScale<<<b_pin, threads, 0, s.stream>>>(r);
    ppr::kReconRadialHz<<<b_asm, threads, 0, s.stream>>>(r);
    ppr::kReconRadial<<<b_rad, threads, 0, s.stream>>>(r);
    ppr::kReconPeakPartials<<<b_ch, threads, 0, s.stream>>>(r);
    ppr::kReconPeakFold<<<1, 1, 0, s.stream>>>(r);
    if ((rc = cudaGetLastError()) != cudaSuccess) return s.fail("recon kernels", rc);

    auto d2h = [&](void* h, const void* d, size_t bytes, const char* name) -> bool {
        rc = rasbery::xfer::memcpyAsync("CudaPprBackend.cu:reconstructPinPower", name, h,
                                        d, bytes, cudaMemcpyDeviceToHost, s.stream);
        if (rc != cudaSuccess) return s.fail(name, rc);
        s.n_d2h_bytes += bytes;
        return true;
    };
    if (!d2h(s.h_scalars, s.d_scalars, 4 * sizeof(double), "D2H recon scalars")) return false;
    if (step.materialize_pin) {
        if (!d2h(step.pin_power, s.d_pin_power, nmap * sizeof(double), "D2H pin_power"))
            return false;
        if (step.reconstruct_flux && step.pin_flux != nullptr &&
            !d2h(step.pin_flux, s.d_pin_flux, nmap * s.ng * sizeof(double), "D2H pin_flux"))
            return false;
        ++s.n_pin_materializations;
    } else {
        // THE HOST ARRAY IS NOW STALE, AND THAT IS THE DESIGN.  Geometry's pin
        // map has exactly one reader (IO.cpp, under print_opt.pin_info) and the
        // caller passes that flag straight through; a statepoint that does not
        // print does not pay 5.4 MB to leave a copy nobody opens.
        s.n_d2h_elided += nmap * sizeof(double);
        if (step.reconstruct_flux) s.n_d2h_elided += nmap * s.ng * sizeof(double);
    }

    if (!s.syncStream("recon sync")) return false;

    if (step.frp != nullptr) *step.frp = s.h_scalars[1];
    if (step.fqp != nullptr) *step.fqp = s.h_scalars[2];
    ++s.n_recon_statepoints;
    s.coeffs_device_only = false;
    s.recon_refusal.clear();
    return true;
}

} // namespace rasbery
