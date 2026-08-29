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

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace rasbery {

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

/// Everything a kernel needs, by value.  Pointers are device pointers.
struct DevCtx {
    int ng;
    int nxyz;
    int nxy;
    int nsurf;
    int has_chif;
    double reigv;

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
__global__ void kUpdateSource(DevCtx x) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk >= x.nxyz) return;
    const int    ng    = x.ng;
    const double reigv = x.reigv;

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
__global__ void kUpdateFused(DevCtx x) {
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
__global__ void kUpdateCorner(DevCtx x) {
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
__global__ void kCornerPartials(DevCtx x, int nchunk, double* partials) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
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

double relativeChange(double current, double previous) {
    return (previous != 0.0) ? fabs((current - previous) / previous) : 1.0;
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

    double* d_partials = nullptr;
    double* h_partials = nullptr; ///< pinned, 4 * nchunk

    bool geometry_uploaded = false;

    unsigned long long n_statepoints = 0;
    unsigned long long n_iterations  = 0;
    double             wall_ms       = 0.0;

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop  = nullptr;

    ~Impl() { release(); }

    void release() {
        auto f  = [](void* p) { if (p) cudaFree(p); };
        f(d_hmesh);  f(d_lktosfc); f(d_neibrb); f(d_is_fuel);
        f(d_phif);   f(d_phis);    f(d_jnet);
        f(d_xsdf);   f(d_xsrf);    f(d_xsnf);   f(d_xssm);  f(d_chif); f(d_crdf);
        f(d_phic);   f(d_p);       f(d_a);      f(d_c);     f(d_q);    f(d_l);  f(d_bt);
        f(d_partials);
        if (h_partials) cudaFreeHost(h_partials);
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_stop) cudaEventDestroy(ev_stop);
        if (stream) cudaStreamDestroy(stream);
        d_hmesh = nullptr; d_lktosfc = nullptr; d_neibrb = nullptr; d_is_fuel = nullptr;
        d_phif = d_phis = d_jnet = nullptr;
        d_xsdf = d_xsrf = d_xsnf = d_xssm = d_chif = d_crdf = nullptr;
        d_phic = d_p = d_a = d_c = d_q = d_l = d_bt = nullptr;
        d_partials = nullptr; h_partials = nullptr;
        ev_start = nullptr; ev_stop = nullptr; stream = nullptr;
        geometry_uploaded = false;
    }

    bool fail(const char* what, cudaError_t rc) {
        failed      = true;
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
        release();
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
            {(void**)&d_partials, static_cast<size_t>(4 * nchunk) * sizeof(double), "partials"},
        };
        for (const Alloc& a : allocs) {
            rc = cudaMalloc(a.p, a.bytes);
            if (rc != cudaSuccess) return fail(a.name, rc);
        }
        rc = cudaMallocHost((void**)&h_partials,
                            static_cast<size_t>(4 * nchunk) * sizeof(double));
        if (rc != cudaSuccess) return fail("cudaMallocHost(partials)", rc);
        return true;
    }
};

PprBackend::PprBackend() : _impl(new Impl) {
    _impl->enabled = truthy(std::getenv("RASBERY_GPU_PPR"));
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


bool PprBackend::resetAndDrive(const ppr::GeomView& geom, const ppr::StepView& step,
                               int niter, int* iters) {
    Impl& s = *_impl;
    if (!s.enabled || s.failed) return false;
    if (geom.ng != 2) {
        s.status_text = "declined: ng != 2 (updateSource body is 2-group)";
        return false;
    }
    if (!s.ensureShape(geom)) return false;

    const size_t nn  = static_cast<size_t>(geom.nxyz);
    const size_t nng = nn * geom.ng;
    const size_t nsg = static_cast<size_t>(geom.nsurf) * geom.ng;

    cudaError_t rc = cudaSuccess;
    auto        h2d = [&](void* d, const void* h, size_t bytes, const char* name) -> bool {
        rc = cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s.stream);
        if (rc != cudaSuccess) return s.fail(name, rc);
        return true;
    };

    if (!s.geometry_uploaded) {
        if (!h2d(s.d_hmesh, geom.hmesh, nn * kNDIRMAX * sizeof(double), "H2D hmesh")) return false;
        if (!h2d(s.d_lktosfc, geom.lktosfc, nn * kNDIRMAX * kLR * sizeof(int), "H2D lktosfc")) return false;
        if (!h2d(s.d_neibrb, geom.neibrb, static_cast<size_t>(geom.nxy) * kNEWS * sizeof(int), "H2D neibrb")) return false;
        if (!h2d(s.d_is_fuel, geom.is_fuel, nn * sizeof(unsigned char), "H2D is_fuel")) return false;
        s.geometry_uploaded = true;
    }

    if (!h2d(s.d_phif, step.phif, nng * sizeof(double), "H2D phif")) return false;
    if (!h2d(s.d_phis, step.phis, nsg * sizeof(double), "H2D phis")) return false;
    if (!h2d(s.d_jnet, step.jnet, nsg * sizeof(double), "H2D jnet")) return false;
    if (!h2d(s.d_xsdf, step.xsdf, nng * sizeof(double), "H2D xsdf")) return false;
    if (!h2d(s.d_xsrf, step.xsrf, nng * sizeof(double), "H2D xsrf")) return false;
    if (!h2d(s.d_xsnf, step.xsnf, nng * sizeof(double), "H2D xsnf")) return false;
    if (!h2d(s.d_xssm, step.xssm, nn * geom.ng * geom.ng * sizeof(double), "H2D xssm")) return false;
    if (step.chif != nullptr &&
        !h2d(s.d_chif, step.chif, nng * sizeof(double), "H2D chif"))
        return false;
    if (!h2d(s.d_crdf, step.crdf, nng * sizeof(double), "H2D crdf")) return false;

    DevCtx& x   = s.ctx;
    x.ng        = geom.ng;
    x.nxyz      = geom.nxyz;
    x.nxy       = geom.nxy;
    x.nsurf     = geom.nsurf;
    x.has_chif  = (step.chif != nullptr) ? 1 : 0;
    x.reigv     = step.reigv;
    x.hmesh     = s.d_hmesh;
    x.lktosfc   = s.d_lktosfc;
    x.neibrb    = s.d_neibrb;
    x.is_fuel   = s.d_is_fuel;
    x.phif      = s.d_phif;
    x.phis      = s.d_phis;
    x.jnet      = s.d_jnet;
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

    // reset()
    kBuckling<<<blocks_ng, threads, 0, s.stream>>>(x);
    kCornerInit<<<blocks_ng, threads, 0, s.stream>>>(x);
    kFit<<<blocks_ng, threads, 0, s.stream>>>(x);
    kAxialLeakage<<<blocks_ng, threads, 0, s.stream>>>(x);
    kUpdateSource<<<blocks_n, threads, 0, s.stream>>>(x);

    if ((rc = cudaGetLastError()) != cudaSuccess) return s.fail("reset kernels", rc);

    // drive(niter): the host's loop, kernel for kernel, break test for break test.
    double prev_nw = 0.0, prev_sw = 0.0, prev_ne = 0.0, prev_se = 0.0;
    int    iters_done = 0;
    for (int citer = 0; citer < niter; ++citer) {
        for (int f = 0; f < kSourceSweepsPerIteration; ++f) {
            kUpdateFused<<<blocks_ng, threads, 0, s.stream>>>(x);
            kUpdateSource<<<blocks_n, threads, 0, s.stream>>>(x);
        }
        kUpdateCorner<<<blocks_ng, threads, 0, s.stream>>>(x);
        kCornerPartials<<<blocks_ch, threads, 0, s.stream>>>(x, s.nchunk, s.d_partials);

        rc = cudaMemcpyAsync(s.h_partials, s.d_partials,
                             static_cast<size_t>(4 * s.nchunk) * sizeof(double),
                             cudaMemcpyDeviceToHost, s.stream);
        if (rc != cudaSuccess) return s.fail("D2H corner partials", rc);
        if ((rc = cudaStreamSynchronize(s.stream)) != cudaSuccess)
            return s.fail("drive sync", rc);

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
        rc = cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s.stream);
        if (rc != cudaSuccess) return s.fail(name, rc);
        return true;
    };
    if (!d2h(step.p, s.d_p, nng * 15 * sizeof(double), "D2H p")) return false;
    if (!d2h(step.a, s.d_a, nng * 8 * sizeof(double), "D2H a")) return false;
    if (!d2h(step.c, s.d_c, nng * 15 * sizeof(double), "D2H c")) return false;
    if (!d2h(step.bt, s.d_bt, nng * sizeof(double), "D2H bt")) return false;
    if (!d2h(step.phic, s.d_phic, nng * 4 * sizeof(double), "D2H phic")) return false;
    if (!d2h(step.q, s.d_q, nng * 15 * sizeof(double), "D2H q")) return false;
    if (!d2h(step.l, s.d_l, nng * 9 * sizeof(double), "D2H l")) return false;

    cudaEventRecord(s.ev_stop, s.stream);
    if ((rc = cudaStreamSynchronize(s.stream)) != cudaSuccess) return s.fail("final sync", rc);

    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, s.ev_start, s.ev_stop) == cudaSuccess)
        s.wall_ms += static_cast<double>(ms);

    ++s.n_statepoints;
    s.n_iterations += static_cast<unsigned long long>(iters_done);
    if (iters) *iters = iters_done;
    return true;
}

} // namespace rasbery
