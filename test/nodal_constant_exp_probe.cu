// Task 4 Step 0 -- the B0 RESCUE SPIKE (Rev.7.1 Sec 6.1).
//
// THE QUESTION.  nodalConstantCoefficients() (src/NodalConstantKernel.h:31,39)
// calls exactly two library functions: sqrt(kp2) and exp(kp).  Everything else
// in the body is IEEE basic arithmetic, which is bit-reproducible on any
// conforming device once contraction is pinned (--fmad=false).  So the whole
// classification of Task 4 -- B0 (bit-preserving, no Gate A/B, no v3 freeze) or
// N1 (trajectory-changing, Gate A/B deferred to Task 22) -- reduces to: do CUDA
// device sqrt and exp return the same doubles as the glibc that builds RASBERY,
// on the arguments this body actually evaluates?
//
// THE ANSWER, MEASURED.  sm_61 (GTX 1080 Ti), CUDA 12.6, glibc 2.39, 4,000,000
// log-uniform kp2 in [1e-8, 1e4]:
//
//     sqrt   0 mismatches            -> would have been B0 on its own
//     exp    133,547 mismatches      3.3387%, max 1 ulp
//            in kp2 in [1e-3,1e2]     5.3388% of 1,666,666
//
// sqrt is IEEE-754 correctly rounded and NVIDIA guarantees it (0 ulp); the
// measurement agrees.  exp is documented at <=1 ulp, glibc's is a different
// algorithm at roughly half an ulp, and on about one argument in nineteen inside
// the band the nodal body actually visits, the two round to different doubles.
//
// VERDICT: N1.  The rescue FAILED.  Recorded in NodalConstantKernel.h and
// src/CudaNodalConstantKernel.h; Gate A/B and the v3 freeze stay on Task 22.
//
// ---------------------------------------------------------------------------
// WHY THIS FILE STAYS IN THE TREE AFTER THE SPIKE ANSWERED
// ---------------------------------------------------------------------------
//
// It is the gate that keeps the recorded verdict honest, in three directions:
//
//   1. sqrt REGRESSION.  If a future device, driver or compiler flag stops
//      returning correctly-rounded sqrt, the N1 deviation stops being confined
//      to exp and the replay gate's error attribution silently becomes wrong.
//      Any sqrt mismatch fails this probe.
//   2. glibc DRIFT.  The 142 baked anchors below carry the bit patterns glibc
//      2.39 produced at authoring time.  If the host libm that builds RASBERY
//      changes its exp, the CPU reference itself moved and the N1 receipt is
//      about a different reference.  A baked-anchor mismatch on the HOST side
//      says so explicitly, and says which anchor.
//   3. THE RESCUE BECOMING AVAILABLE.  If a future toolchain does make device
//      exp bit-match glibc over the swept range, the N1 classification is now
//      too conservative and Task 4 should be re-run as B0 -- so "no exp
//      mismatches anywhere" is ALSO a failure of this gate.  A spike whose
//      verdict nothing re-checks is a comment, not a receipt.
//
// ---------------------------------------------------------------------------
// USAGE
// ---------------------------------------------------------------------------
//
//   ./rasbery_nodal_constant_exp_probe [sweep_points]
//   ./rasbery_nodal_constant_exp_probe --capture <RASBERY_NODAL_DUMP file>
//
// The default sweep is a log-uniform envelope, deliberately WIDER than the
// physics: kp2 = xsrf*h*h/(4*xsdf) with 2-group PWR/SMR data (xsrf 7e-3..0.15,
// xsdf 0.26..1.45, hmesh 1.26..30.5 cm, reflector rows included) lands in
// roughly [5e-3, 90]; the sweep runs [1e-8, 1e4] and the baked anchors add
// zero, the subnormals, DBL_MIN, the exp-overflow edge at kp ~ 709.78, and
// +inf.  --capture reads a production nodal dump and mines the deck's REAL kp2
// population instead (xsrf and hmesh are in the capture verbatim; xsdf is
// recovered from diagD = 4*xsdf/(h*h), so a captured kp2 carries ~2 ulp of
// recovery error -- fine for a range, not a bit contract, and the anchors are
// the bit contract).
//
// Build: nvcc --fmad=false (the flag is irrelevant to library sqrt/exp but the
// probe must be built exactly like the production TU, or it is measuring a
// different compilation than the one whose verdict it records).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct GoldenAnchor {
    unsigned long long kp2;  ///< bit pattern of the argument
    unsigned long long sqrt; ///< glibc 2.39 sqrt(kp2)
    unsigned long long exp;  ///< glibc 2.39 exp(sqrt(kp2))
};

// Generated on Ubuntu 24.04 / glibc 2.39 with the recipe documented above:
// 126 realistic (xsrf, xsdf, hmesh) triples on a fixed stride, then 16
// extremes.  Regenerate ONLY together with a new receipt -- these bits ARE the
// recorded reference.
#include "nodal_constant_exp_golden.h"

inline constexpr int kAnchorCount =
    static_cast<int>(sizeof(kExpGolden) / sizeof(kExpGolden[0]));

__host__ __device__ inline double fromBits(unsigned long long b) {
    double d;
    memcpy(&d, &b, sizeof d);
    return d;
}

inline unsigned long long toBits(double d) {
    unsigned long long b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

/// Distance in representable doubles.  Only ever reported, never a pass
/// criterion: this gate is about bit equality, and "close enough" is exactly
/// the judgement a B0/N1 classification is not allowed to make.
long long ulpDistance(double a, double b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return -1;
    long long x = static_cast<long long>(toBits(a));
    long long y = static_cast<long long>(toBits(b));
    if (x < 0) x = static_cast<long long>(0x8000000000000000ull) - x;
    if (y < 0) y = static_cast<long long>(0x8000000000000000ull) - y;
    const long long d = x - y;
    return d < 0 ? -d : d;
}

__global__ void k_sqrt_exp(const double* __restrict__ kp2, double* __restrict__ out_sqrt,
                           double* __restrict__ out_exp, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double kp = sqrt(kp2[i]);
    out_sqrt[i]     = kp;
    out_exp[i]      = exp(kp);
}

#define TRY(expr)                                                                 \
    do {                                                                          \
        const cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                                  \
            std::fprintf(stderr, "%s -> %s\n", #expr, cudaGetErrorString(_e));    \
            return 3;                                                             \
        }                                                                         \
    } while (0)

struct Score {
    long long n         = 0;
    long long sqrt_bad  = 0;
    long long exp_bad   = 0;
    long long exp_maxulp = 0;
    double    worst_kp  = 0.0;
};

/// Run the device kernel over `kp2` and score it against this machine's glibc.
int scoreAgainstHostLibm(const std::vector<double>& kp2, Score& s) {
    const int n = static_cast<int>(kp2.size());
    double *d_in = nullptr, *d_s = nullptr, *d_e = nullptr;
    TRY(cudaMalloc(&d_in, sizeof(double) * n));
    TRY(cudaMalloc(&d_s, sizeof(double) * n));
    TRY(cudaMalloc(&d_e, sizeof(double) * n));
    TRY(cudaMemcpy(d_in, kp2.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    k_sqrt_exp<<<(n + 255) / 256, 256>>>(d_in, d_s, d_e, n);
    TRY(cudaGetLastError());
    TRY(cudaDeviceSynchronize());

    std::vector<double> hs(n), he(n);
    TRY(cudaMemcpy(hs.data(), d_s, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(he.data(), d_e, sizeof(double) * n, cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_s);
    cudaFree(d_e);

    for (int i = 0; i < n; ++i) {
        const double gs = std::sqrt(kp2[i]);
        const double ge = std::exp(gs);
        ++s.n;
        if (toBits(gs) != toBits(hs[i])) ++s.sqrt_bad;
        if (toBits(ge) != toBits(he[i])) {
            ++s.exp_bad;
            const long long u = ulpDistance(ge, he[i]);
            if (u > s.exp_maxulp) {
                s.exp_maxulp = u;
                s.worst_kp   = gs;
            }
        }
    }
    return 0;
}

/// Mine the deck's real kp2 population out of a RASBERY_NODAL_DUMP capture.
/// Layout is Nodal.cpp's nodalDumpState(), read only as far as diagD.
bool loadCaptureKp2(const char* path, std::vector<double>& out) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::int64_t hdr[8];
    if (std::fread(hdr, sizeof hdr[0], 8, f) != 8) { std::fclose(f); return false; }
    const std::int64_t nxyz = hdr[0], nsurf = hdr[1], ndir = hdr[2], ng = hdr[3];
    const std::int64_t chif_empty = hdr[5];
    if (nxyz <= 0 || ndir != 3 || ng != 2) { std::fclose(f); return false; }

    auto skip = [&](long long bytes) { std::fseek(f, static_cast<long>(bytes), SEEK_CUR); };
    auto rd   = [&](std::vector<double>& v, size_t n) {
        v.resize(n);
        return std::fread(v.data(), sizeof(double), n, f) == n;
    };

    skip(sizeof(double));                          // reigv
    skip(sizeof(int) * nxyz * 3 * 2);              // lktosfc
    skip(sizeof(int) * nxyz * 6);                  // neib
    skip(sizeof(int) * nsurf * 2 * 3);             // lklr, idirlr, sgnlr

    std::vector<double> hmesh, albedo, xsrf, tmp;
    if (!rd(hmesh, static_cast<size_t>(nxyz) * 3)) { std::fclose(f); return false; }
    if (!rd(albedo, 6)) { std::fclose(f); return false; }
    if (!rd(xsrf, static_cast<size_t>(ng) * nxyz)) { std::fclose(f); return false; }
    skip(sizeof(double) * ng * nxyz);              // xsnf
    skip(sizeof(double) * ng * ng * nxyz);         // xssm
    if (!chif_empty) skip(sizeof(double) * ng * nxyz);
    // consts[]: eta1 eta2 m260 m251 m253 m262 m264 diagD diagDI -- DUMP order,
    // which is Nodal.cpp's local `consts[9]` and NOT the arena's packing.
    const long long one = static_cast<long long>(nxyz) * 3 * ng;
    skip(sizeof(double) * one * 7);
    std::vector<double> diagD;
    if (!rd(diagD, static_cast<size_t>(one))) { std::fclose(f); return false; }
    std::fclose(f);

    out.clear();
    out.reserve(static_cast<size_t>(nxyz) * 3 * ng);
    for (std::int64_t lk = 0; lk < nxyz; ++lk)
        for (int idir = 0; idir < 3; ++idir)
            for (std::int64_t ig = 0; ig < ng; ++ig) {
                const double h  = hmesh[static_cast<size_t>(lk) * 3 + idir];
                const double dD = diagD[static_cast<size_t>((lk * 3 + idir) * ng + ig)];
                if (!(dD > 0.0) || !(h > 0.0)) continue;
                // diagD = 4*xsdf/(h*h)  ->  xsdf = diagD*h*h/4 (1 ulp of recovery)
                const double xsdf = dD * h * h / 4;
                const double r    = xsrf[static_cast<size_t>(ig) * nxyz + lk];
                out.push_back(r * h * h / (4 * xsdf));
            }
    return !out.empty();
}

} // namespace

int main(int argc, char** argv) {
    std::vector<double> capture;
    long                sweep_n = 4000000;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--capture") == 0 && i + 1 < argc) {
            if (!loadCaptureKp2(argv[i + 1], capture)) {
                std::fprintf(stderr, "cannot read capture %s\n", argv[i + 1]);
                return 2;
            }
            ++i;
        } else {
            sweep_n = std::strtol(argv[i], nullptr, 10);
        }
    }
    if (sweep_n < 2) sweep_n = 2;

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("nodal constant exp probe: SKIP (no CUDA device)\n");
        return 0;
    }

    int failures = 0;

    // --- (A) the baked glibc anchors, on the HOST -------------------------
    // Does the libm that would build RASBERY here still agree with the libm the
    // recorded verdict was measured against?
    int host_drift = 0;
    for (int i = 0; i < kAnchorCount; ++i) {
        const double kp2 = fromBits(kExpGolden[i].kp2);
        const double gs  = std::sqrt(kp2);
        const double ge  = std::exp(gs);
        if (toBits(gs) != kExpGolden[i].sqrt || toBits(ge) != kExpGolden[i].exp) {
            if (host_drift < 5)
                std::fprintf(stderr,
                             "  host libm drift at anchor %d: kp2=%.17g "
                             "sqrt %016llx vs baked %016llx, exp %016llx vs baked %016llx\n",
                             i, kp2, toBits(gs), kExpGolden[i].sqrt, toBits(ge),
                             kExpGolden[i].exp);
            ++host_drift;
        }
    }
    if (host_drift) {
        std::fprintf(stderr,
                     "nodal constant exp probe: FAIL host libm no longer reproduces %d/%d "
                     "baked anchors -- the CPU reference the N1 receipt is written against "
                     "has MOVED; re-run the spike and re-record before trusting Task 4\n",
                     host_drift, kAnchorCount);
        ++failures;
    } else {
        std::printf("baked glibc anchors: %d/%d reproduced on this host\n", kAnchorCount,
                    kAnchorCount);
    }

    // --- (B) the anchors, on the DEVICE ------------------------------------
    std::vector<double> anchor_kp2(kAnchorCount);
    for (int i = 0; i < kAnchorCount; ++i) anchor_kp2[i] = fromBits(kExpGolden[i].kp2);
    Score anchors;
    if (const int rc = scoreAgainstHostLibm(anchor_kp2, anchors); rc) return rc;

    // --- (C) the dense sweep, or the capture -------------------------------
    std::vector<double> points;
    const char*         source = "sweep";
    if (!capture.empty()) {
        points = capture;
        source = "capture";
    } else {
        const double lo = 1e-8, hi = 1e4;
        const double llo = std::log(lo), lhi = std::log(hi);
        points.resize(static_cast<size_t>(sweep_n));
        for (long i = 0; i < sweep_n; ++i)
            points[static_cast<size_t>(i)] =
                std::exp(llo + (lhi - llo) * static_cast<double>(i) /
                                   static_cast<double>(sweep_n - 1));
    }
    Score bulk;
    if (const int rc = scoreAgainstHostLibm(points, bulk); rc) return rc;

    // --- (D) the physical band, reported separately ------------------------
    // The band is where the classification actually bites; the wide sweep only
    // proves nothing hides outside it.
    std::vector<double> band;
    for (double k : points)
        if (k >= 1e-3 && k <= 1e2) band.push_back(k);
    Score in_band;
    if (!band.empty())
        if (const int rc = scoreAgainstHostLibm(band, in_band); rc) return rc;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s sm_%d%d\n", prop.name, prop.major, prop.minor);
    std::printf("anchors  n=%lld  sqrt_bad=%lld  exp_bad=%lld  max_ulp=%lld\n", anchors.n,
                anchors.sqrt_bad, anchors.exp_bad, anchors.exp_maxulp);
    std::printf("%s   n=%lld  sqrt_bad=%lld  exp_bad=%lld (%.4f%%)  max_ulp=%lld at kp=%.17g\n",
                source, bulk.n, bulk.sqrt_bad, bulk.exp_bad,
                100.0 * static_cast<double>(bulk.exp_bad) / static_cast<double>(bulk.n ? bulk.n : 1),
                bulk.exp_maxulp, bulk.worst_kp);
    if (in_band.n)
        std::printf("band kp2 in [1e-3,1e2]  n=%lld  sqrt_bad=%lld  exp_bad=%lld (%.4f%%)\n",
                    in_band.n, in_band.sqrt_bad, in_band.exp_bad,
                    100.0 * static_cast<double>(in_band.exp_bad) /
                        static_cast<double>(in_band.n));

    // --- verdict -----------------------------------------------------------
    const long long sqrt_bad = anchors.sqrt_bad + bulk.sqrt_bad;
    const long long exp_bad  = anchors.exp_bad + bulk.exp_bad;

    if (sqrt_bad != 0) {
        std::fprintf(stderr,
                     "nodal constant exp probe: FAIL device sqrt is no longer correctly "
                     "rounded (%lld mismatches).  The recorded N1 deviation is 'exp only'; "
                     "with sqrt moving too, the replay gate's attribution is wrong.\n",
                     sqrt_bad);
        ++failures;
    }
    if (exp_bad == 0) {
        std::fprintf(stderr,
                     "nodal constant exp probe: FAIL device exp now matches glibc on every "
                     "swept argument.  The B0 rescue that FAILED at authoring time would "
                     "now SUCCEED -- re-run Task 4 Step 0 and reclassify (B0 drops Gate "
                     "A/B and the v3 freeze).\n");
        ++failures;
    }

    if (failures) {
        std::printf("VERDICT: recorded classification N1 is NO LONGER SUPPORTED here\n");
        return 1;
    }
    std::printf("VERDICT: N1 (sqrt bit-exact, exp differs by <=%lld ulp on %.4f%% of "
                "arguments) -- B0 rescue failed, as recorded\n",
                bulk.exp_maxulp,
                100.0 * static_cast<double>(exp_bad) /
                    static_cast<double>(anchors.n + bulk.n));
    std::printf("nodal constant exp probe: PASS\n");
    return 0;
}
