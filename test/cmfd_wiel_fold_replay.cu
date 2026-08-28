// Device replay gate for the cmfd_wiel_finalize Wielandt fold.
//
// The fold in src/CudaBICGBackend.cu is a hard serial dependency: it has to
// reproduce, bit for bit, the l-ascending `err`/`gammad`/`gamman` accumulation
// of BICGCMFD::wiel, which is already baked into the frozen reference output.
// The only thing the shipped kernel is allowed to do differently from the
// naive loop is hoist the LOADS -- never an add, never an operand pairing.
//
// This harness holds the two forms side by side on the device and demands that
// they agree in every bit, on the real mesh width and on a spread of widths
// that exercise every remainder class of kWielFoldBatch (including nxyz below
// one batch, exactly one batch, and one short of a batch).  It also prints the
// two timings, which is the whole reason the batched form exists.
//
// Build (no CMake target; tools/test_cmfd_wiel_fold_contract.py drives it):
//   nvcc -O3 -std=c++17 -arch=sm_61 --fmad=false test/cmfd_wiel_fold_replay.cu \
//        -o cmfd_wiel_fold_replay && ./cmfd_wiel_fold_replay
//
// Exit code 0 = every width bit-identical.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_OK(call)                                                              \
    do {                                                                           \
        const cudaError_t rc_ = (call);                                            \
        if (rc_ != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(rc_),    \
                         __FILE__, __LINE__);                                      \
            std::exit(2);                                                          \
        }                                                                          \
    } while (0)

// Must match src/CudaBICGBackend.cu.
constexpr int kWielFoldBatch = 64;
constexpr int kReduceThreads   = 256;
constexpr int kMaxReduceBlocks = 256;

__host__ __device__ inline int reduce_blocks_for(const int n) {
    const int per_block = kReduceThreads * 4;
    int       blocks    = (n + per_block - 1) / per_block;
    if (blocks < 1) blocks = 1;
    if (blocks > kMaxReduceBlocks) blocks = kMaxReduceBlocks;
    return blocks;
}

// ---------------------------------------------------------------------------
// REFERENCE: the frozen flat fold, exactly as BICGCMFD::wiel accumulates and
// exactly as cmfd_wiel_finalize did before the batching.  Never change this.
// ---------------------------------------------------------------------------
__global__ void fold_reference(const int nxyz, const long long vec_stride,
                               const double* __restrict__ terms_ab,
                               const double* __restrict__ terms_c,
                               double* out) {
    const int     m    = static_cast<int>(blockIdx.y);
    const int     lane = static_cast<int>(threadIdx.x);
    const double* ta   = terms_ab + m * vec_stride;
    const double* tc   = terms_c + m * vec_stride;
    __shared__ double lane_sum[3];
    if (lane < 3) {
        const double* values = lane == 0 ? ta : (lane == 1 ? ta + nxyz : tc);
        double        sum    = 0.0;
        for (int l = 0; l < nxyz; ++l) sum = sum + values[l];
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;
    out[m * 3 + 0] = lane_sum[0];
    out[m * 3 + 1] = lane_sum[1];
    out[m * 3 + 2] = lane_sum[2];
}

// ---------------------------------------------------------------------------
// SHIPPED: the marked fold body of cmfd_wiel_finalize in
// src/CudaBICGBackend.cu, copied verbatim.  The contract test compares the two
// texts, so an edit there that is not mirrored here fails before this runs.
// ---------------------------------------------------------------------------
__global__ void fold_shipped(const int nxyz, const long long vec_stride,
                             const double* __restrict__ terms_ab,
                             const double* __restrict__ terms_c,
                             double* out) {
    const int     m    = static_cast<int>(blockIdx.y);
    const int     lane = static_cast<int>(threadIdx.x);
    const double* ta   = terms_ab + m * vec_stride;
    const double* tc   = terms_c + m * vec_stride;
    __shared__ double lane_sum[3];
    if (lane < 3) {
        const double* values = lane == 0 ? ta : (lane == 1 ? ta + nxyz : tc);
        double sum = 0.0;
        // RASBERY_CMFD_WIEL_FOLD_BEGIN -- mirrored verbatim by
        // test/cmfd_wiel_fold_replay.cu; tools/test_cmfd_wiel_fold_contract.py
        // fails if the two texts drift apart.
        int       l    = 0;
        const int tail = nxyz - (nxyz % kWielFoldBatch);
        for (; l < tail; l += kWielFoldBatch) {
            double batch[kWielFoldBatch];
#pragma unroll
            for (int j = 0; j < kWielFoldBatch; ++j) batch[j] = __ldg(values + l + j);
#pragma unroll
            for (int j = 0; j < kWielFoldBatch; ++j) sum = sum + batch[j];
        }
        for (; l < nxyz; ++l) sum = sum + values[l];
        // RASBERY_CMFD_WIEL_FOLD_END
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;
    out[m * 3 + 0] = lane_sum[0];
    out[m * 3 + 1] = lane_sum[1];
    out[m * 3 + 2] = lane_sum[2];
}

// ---------------------------------------------------------------------------
// CHUNKED (RASBERY_GPU_WIEL_FOLD=chunked).  Class N1: deterministic on a given
// arch, NOT bit-identical to serial.  Copied from src/CudaBICGBackend.cu; the
// shape invariants the determinism argument rests on -- the (nxyz, gridDim.x)
// chunk, the fixed tree, the strict stage-2 walk, the absence of atomics --
// are what tools/test_cmfd_wiel_fold_contract.py pins in both files.
// ---------------------------------------------------------------------------
__global__ void chunked_stage1(const int nxyz, const long long vec_stride,
                               const double* __restrict__ terms_ab,
                               const double* __restrict__ terms_c,
                               double* __restrict__ partial) {
    const int m = static_cast<int>(blockIdx.y);
    __shared__ double sh_err[kReduceThreads];
    __shared__ double sh_gd[kReduceThreads];
    __shared__ double sh_gn[kReduceThreads];

    const double* ta = terms_ab + m * vec_stride;
    const double* tc = terms_c + m * vec_stride;
    double*       pm = partial + static_cast<long long>(m) * (3 * kMaxReduceBlocks);

    const int chunk = (nxyz + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, nxyz);

    double s_err = 0.0, s_gd = 0.0, s_gn = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x)) {
        s_err += ta[i];
        s_gd += ta[nxyz + i];
        s_gn += tc[i];
    }
    sh_err[threadIdx.x] = s_err;
    sh_gd[threadIdx.x]  = s_gd;
    sh_gn[threadIdx.x]  = s_gn;
    __syncthreads();

    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            sh_err[threadIdx.x] += sh_err[threadIdx.x + stride];
            sh_gd[threadIdx.x] += sh_gd[threadIdx.x + stride];
            sh_gn[threadIdx.x] += sh_gn[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pm[0 * kMaxReduceBlocks + blockIdx.x] = sh_err[0];
        pm[1 * kMaxReduceBlocks + blockIdx.x] = sh_gd[0];
        pm[2 * kMaxReduceBlocks + blockIdx.x] = sh_gn[0];
    }
}

__global__ void chunked_stage2(const int blocks, const double* __restrict__ partial,
                               double* out) {
    const int     m    = static_cast<int>(blockIdx.y);
    const int     lane = static_cast<int>(threadIdx.x);
    const double* pm   = partial + static_cast<long long>(m) * (3 * kMaxReduceBlocks);
    __shared__ double lane_sum[3];
    if (lane < 3) {
        const double* values = pm + lane * kMaxReduceBlocks;
        double        sum    = 0.0;
        for (int i = 0; i < blocks; ++i) sum = sum + values[i]; // strict index order
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;
    out[m * 3 + 0] = lane_sum[0];
    out[m * 3 + 1] = lane_sum[1];
    out[m * 3 + 2] = lane_sum[2];
}

namespace {

int  g_failures = 0;
long g_widths   = 0;
long g_chunked_widths = 0;
long g_ulp_max        = 0;

std::uint64_t asBits(double v) {
    std::uint64_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

/// Addends shaped like the real ones: err1*err1 and pv*pv are non-negative and
/// span several decades, psid*pv changes sign.  Magnitude spread is what makes
/// a reassociated sum differ, so a fixture without it would pass vacuously.
void fillFixture(std::vector<double>& ab, std::vector<double>& c, int nxyz, unsigned seed) {
    std::uint64_t s = 0x9E3779B97F4A7C15ull ^ seed;
    auto          next = [&s] {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    };
    for (int l = 0; l < nxyz; ++l) {
        const double scale = std::pow(10.0, -6.0 + 12.0 * next());
        const double pv    = scale * (0.5 + next());
        const double err1  = scale * 1e-3 * (next() - 0.5);
        ab[static_cast<size_t>(l)]                          = err1 * err1;
        ab[static_cast<size_t>(nxyz) + static_cast<size_t>(l)] = (next() - 0.5) * pv;
        c[static_cast<size_t>(l)]                           = pv * pv;
    }
}

float timeKernel(void (*launch)(int, long long, const double*, const double*, double*),
                 int nxyz, long long vs, const double* ta, const double* tc, double* out) {
    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
    launch(nxyz, vs, ta, tc, out);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(a));
    for (int i = 0; i < 100; ++i) launch(nxyz, vs, ta, tc, out);
    CUDA_OK(cudaEventRecord(b));
    CUDA_OK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, a, b));
    CUDA_OK(cudaEventDestroy(a));
    CUDA_OK(cudaEventDestroy(b));
    return ms * 1000.0f / 100.0f;
}

void launchRef(int n, long long vs, const double* ta, const double* tc, double* out) {
    fold_reference<<<dim3(1, 1), 32>>>(n, vs, ta, tc, out);
}
void launchShip(int n, long long vs, const double* ta, const double* tc, double* out) {
    fold_shipped<<<dim3(1, 1), 32>>>(n, vs, ta, tc, out);
}

void runWidth(int nxyz, unsigned seed, bool timed) {
    const long long     vs = 2LL * nxyz;
    std::vector<double> ab(static_cast<size_t>(2 * nxyz)), c(static_cast<size_t>(2 * nxyz), 0.0);
    fillFixture(ab, c, nxyz, seed);

    double *d_ab = nullptr, *d_c = nullptr, *d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_ab, ab.size() * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_c, c.size() * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_out, 8 * sizeof(double)));
    CUDA_OK(cudaMemcpy(d_ab, ab.data(), ab.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_c, c.data(), c.size() * sizeof(double), cudaMemcpyHostToDevice));

    double ref[3] = {0, 0, 0}, ship[3] = {0, 0, 0};
    launchRef(nxyz, vs, d_ab, d_c, d_out);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(ref, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    launchShip(nxyz, vs, d_ab, d_c, d_out);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(ship, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    // The host's own serial fold, so the reference kernel is pinned too.
    double host[3] = {0, 0, 0};
    for (int l = 0; l < nxyz; ++l) host[0] = host[0] + ab[static_cast<size_t>(l)];
    for (int l = 0; l < nxyz; ++l)
        host[1] = host[1] + ab[static_cast<size_t>(nxyz) + static_cast<size_t>(l)];
    for (int l = 0; l < nxyz; ++l) host[2] = host[2] + c[static_cast<size_t>(l)];

    ++g_widths;
    for (int k = 0; k < 3; ++k) {
        if (asBits(ref[k]) != asBits(ship[k])) {
            std::fprintf(stderr,
                         "FAIL nxyz=%d sum[%d]: reference %016llx != shipped %016llx\n", nxyz,
                         k, static_cast<unsigned long long>(asBits(ref[k])),
                         static_cast<unsigned long long>(asBits(ship[k])));
            ++g_failures;
        }
        if (asBits(ref[k]) != asBits(host[k])) {
            std::fprintf(stderr,
                         "FAIL nxyz=%d sum[%d]: reference %016llx != host serial %016llx\n",
                         nxyz, k, static_cast<unsigned long long>(asBits(ref[k])),
                         static_cast<unsigned long long>(asBits(host[k])));
            ++g_failures;
        }
    }

    // ---- the OPT-IN chunked arm -------------------------------------------
    //
    // Two separate claims, and only the first is a gate.
    //
    //   ASSERTED: at a fixed chunk config the fold is a pure function of its
    //   inputs -- ten launches, byte-identical every time.  That is the whole
    //   N1 claim (deterministic run to run on a given arch) and the reason it
    //   holds is structural: fixed partition, fixed traversal, fixed tree,
    //   strict stage-2 walk, no atomic anywhere.  It is worth RUNNING rather
    //   than reasoning about, because an accidental atomicAdd or a
    //   grid-size-dependent partition would still pass every static check.
    //
    //   REPORTED ONLY: the ULP distance from serial.  Chunked is expected to
    //   differ -- reassociating a rounded sum moves it -- so a delta is data
    //   for the Gate A/B on 238, not a failure here.  It IS worth printing:
    //   a delta of 0 would mean the fixture is too flat to be evidence of
    //   anything, and a wild one would mean the partition is wrong rather
    //   than merely different.
    double *d_partial = nullptr, *d_cout = nullptr;
    CUDA_OK(cudaMalloc(&d_partial, 3 * kMaxReduceBlocks * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_cout, 8 * sizeof(double)));
    const int blocks = reduce_blocks_for(nxyz);

    double chunked[3] = {0, 0, 0}, repeat[3] = {0, 0, 0};
    auto   launchChunked = [&](double* dst) {
        CUDA_OK(cudaMemset(d_partial, 0, 3 * kMaxReduceBlocks * sizeof(double)));
        chunked_stage1<<<dim3(static_cast<unsigned>(blocks), 1), kReduceThreads>>>(
            nxyz, vs, d_ab, d_c, d_partial);
        chunked_stage2<<<dim3(1, 1), 32>>>(blocks, d_partial, d_cout);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemcpy(dst, d_cout, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    };
    launchChunked(chunked);
    ++g_chunked_widths;
    for (int again = 0; again < 9; ++again) {
        launchChunked(repeat);
        for (int k = 0; k < 3; ++k)
            if (asBits(chunked[k]) != asBits(repeat[k])) {
                std::fprintf(stderr,
                             "FAIL nxyz=%d sum[%d]: chunked is not run-to-run "
                             "deterministic (%016llx then %016llx at blocks=%d)\n",
                             nxyz, k, static_cast<unsigned long long>(asBits(chunked[k])),
                             static_cast<unsigned long long>(asBits(repeat[k])), blocks);
                ++g_failures;
            }
    }
    long worst = 0;
    for (int k = 0; k < 3; ++k) {
        // Both sums are finite and same-signed here, so the ULP distance is
        // the distance between the bit patterns read as integers.
        const long long a = static_cast<long long>(asBits(ref[k]));
        const long long b = static_cast<long long>(asBits(chunked[k]));
        const long      d = static_cast<long>(a > b ? a - b : b - a);
        if (d > worst) worst = d;
    }
    if (worst > g_ulp_max) g_ulp_max = worst;
    std::printf("  nxyz=%-6d blocks=%-3d chunked vs serial: max %ld ULP (informational)\n",
                nxyz, blocks, worst);

    if (timed) {
        const float tr = timeKernel(launchRef, nxyz, vs, d_ab, d_c, d_out);
        const float ts = timeKernel(launchShip, nxyz, vs, d_ab, d_c, d_out);
        cudaEvent_t a, b;
        CUDA_OK(cudaEventCreate(&a));
        CUDA_OK(cudaEventCreate(&b));
        CUDA_OK(cudaEventRecord(a));
        for (int i = 0; i < 100; ++i) {
            chunked_stage1<<<dim3(static_cast<unsigned>(blocks), 1), kReduceThreads>>>(
                nxyz, vs, d_ab, d_c, d_partial);
            chunked_stage2<<<dim3(1, 1), 32>>>(blocks, d_partial, d_cout);
        }
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        const float tc_us = ms * 1000.0f / 100.0f;
        CUDA_OK(cudaEventDestroy(a));
        CUDA_OK(cudaEventDestroy(b));
        std::printf("  nxyz=%-6d flat %8.1f us   batched(%d) %8.1f us   %.2fx   "
                    "chunked %8.1f us   %.1fx\n",
                    nxyz, static_cast<double>(tr), kWielFoldBatch,
                    static_cast<double>(ts),
                    tr > 0.0f ? static_cast<double>(tr / ts) : 0.0,
                    static_cast<double>(tc_us),
                    tc_us > 0.0f ? static_cast<double>(tr / tc_us) : 0.0);
    }

    CUDA_OK(cudaFree(d_partial));
    CUDA_OK(cudaFree(d_cout));
    CUDA_OK(cudaFree(d_ab));
    CUDA_OK(cudaFree(d_c));
    CUDA_OK(cudaFree(d_out));
}

} // namespace

int main(int argc, char** argv) {
    const int mesh = (argc > 1) ? std::atoi(argv[1]) : 8451; // KNGR nxyz

    // Every remainder class that matters, then the real mesh.
    const int widths[] = {1,
                          2,
                          3,
                          kWielFoldBatch - 1,
                          kWielFoldBatch,
                          kWielFoldBatch + 1,
                          2 * kWielFoldBatch,
                          2 * kWielFoldBatch + 7,
                          255,
                          256,
                          257,
                          1000,
                          4096,
                          4097};
    for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i)
        runWidth(widths[i], 1u + i, false);
    // Two seeds at the production width: one shaped fixture is not evidence.
    std::printf("cmfd wiel fold replay (timings at the production mesh):\n");
    runWidth(mesh, 101u, true);
    runWidth(mesh, 202u, false);

    if (g_failures != 0) {
        std::fprintf(stderr, "cmfd wiel fold replay: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("cmfd wiel fold replay: PASS (%ld widths bit-identical to the flat fold "
                "and to the host serial fold; %ld widths chunked-deterministic over 10 "
                "launches, max %ld ULP from serial -- informational)\n",
                g_widths, g_chunked_widths, g_ulp_max);
    return 0;
}
