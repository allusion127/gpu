#include "CudaBICGBackend.h"

#ifdef RASBERY_ENABLE_CUDA

#include "pch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace rasbery {
namespace {

constexpr int kMaxFixedPoint = 1000;
constexpr int kVecBlock      = 256;
constexpr int kScalarBlock   = 256;
constexpr int kDefaultRbSweeps = 2;

constexpr int kRhoOld  = 0;
constexpr int kRhoNew  = 1;
constexpr int kAlpha   = 2;
constexpr int kOmega   = 3;
constexpr int kBeta    = 4;
constexpr int kNorm0   = 5;
constexpr int kNorm    = 6;
constexpr int kEps     = 7;
constexpr int kErrAcc  = 8;
constexpr int kGammaD  = 9;
constexpr int kGammaN  = 10;
constexpr int kScalarCount = 11;
constexpr int kSweepCount  = 5;
constexpr int kSweepEigv   = 0;
constexpr int kSweepShift  = 1;
constexpr int kSweepErrL2  = 2;
constexpr int kSweepNorm0  = 3;
constexpr int kSweepNorm   = 4;

static_assert(kSweepCount == 5, "keep host/device sweep metadata layout stable");

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t rasbery_cuda_error = (call);                                         \
        if (rasbery_cuda_error != cudaSuccess) {                                               \
            std::ostringstream rasbery_cuda_message;                                           \
            rasbery_cuda_message << #call << " failed: "                                      \
                                 << cudaGetErrorString(rasbery_cuda_error);                     \
            throw std::runtime_error(rasbery_cuda_message.str());                              \
        }                                                                                      \
    } while (false)

#define CUBLAS_CHECK(call)                                                                     \
    do {                                                                                       \
        const cublasStatus_t rasbery_cublas_error = (call);                                    \
        if (rasbery_cublas_error != CUBLAS_STATUS_SUCCESS) {                                   \
            std::ostringstream rasbery_cublas_message;                                         \
            rasbery_cublas_message << #call << " failed with status "                         \
                                   << static_cast<int>(rasbery_cublas_error);                   \
            throw std::runtime_error(rasbery_cublas_message.str());                            \
        }                                                                                      \
    } while (false)

bool envEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

int envInt(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

int batchWaitMicros() {
    static const int value = [] {
        const char* raw = std::getenv("RASBERY_BATCH_WAIT_US");
        if (raw == nullptr || *raw == '\0') return 0;
        const int parsed = std::atoi(raw);
        return parsed > 0 ? parsed : 0;
    }();
    return value;
}

struct UploadMirrorTelemetry {
    std::uint64_t bulk_h2d_calls_during_iteration = 0;
    std::uint64_t bulk_h2d_skipped_during_iteration = 0;
    std::uint64_t bulk_h2d_bytes_during_iteration = 0;
    std::uint64_t bulk_d2h_calls_during_iteration = 0;
    std::uint64_t bulk_d2h_bytes_during_iteration = 0;
    std::uint64_t stream_sync_calls_during_iteration = 0;
    std::uint64_t graph_launches = 0;
    std::uint64_t graph_fallbacks = 0;
    std::uint64_t graph_failures = 0;
    std::uint64_t direct_launches = 0;
    std::uint64_t fixed_point_iterations = 0;
};

struct MirroredUpload {
    std::vector<double> shadow;
    const double*       host = nullptr;
    size_t              count = 0;
    bool                valid = false;
};

bool mirrorMatches(const MirroredUpload& mirror, const double* host, size_t count) {
    return mirror.valid && mirror.count == count && mirror.shadow.size() == count &&
           std::memcmp(mirror.shadow.data(), host, count * sizeof(double)) == 0;
}

void rememberMirror(MirroredUpload& mirror, const double* host, size_t count) {
    mirror.shadow.assign(host, host + count);
    mirror.host  = host;
    mirror.count = count;
    mirror.valid = true;
}

__device__ inline int redblack_row_offset(int color, int plane, int row) {
    return color * plane + row;
}

__global__ void cmfd_reset_iteration(const int n,
                                     const int vec_stride,
                                     const int scalar_stride,
                                     const std::uint32_t* active,
                                     double* r,
                                     double* r0,
                                     double* p,
                                     double* v,
                                     double* scalars,
                                     std::uint32_t* iter_halt,
                                     int* iter_stop) {
    const int m = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (active[m] == 0u) return;
    if (tid == 0) {
        double* sm = scalars + static_cast<long long>(m) * scalar_stride;
        sm[kRhoOld] = 1.0;
        sm[kRhoNew] = 1.0;
        sm[kAlpha]  = 1.0;
        sm[kOmega]  = 1.0;
        sm[kBeta]   = 0.0;
        sm[kNorm0]  = 0.0;
        sm[kNorm]   = 0.0;
        iter_halt[m] = 0u;
        iter_stop[m] = kMaxFixedPoint;
    }
    if (tid >= n) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    r[base + tid]  = 0.0;
    r0[base + tid] = 0.0;
    p[base + tid]  = 0.0;
    v[base + tid]  = 0.0;
}

__global__ void cmfd_apply_operator(const int nxyz,
                                    const int ng,
                                    const int vec_stride,
                                    const int mat_stride,
                                    const int cpl_stride,
                                    const std::uint32_t* active,
                                    const std::uint32_t* iter_halt,
                                    const int* neib,
                                    const double* diag,
                                    const double* cc,
                                    const double* x,
                                    double* y) {
    const int m = static_cast<int>(blockIdx.y);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int n = nxyz * ng;
    if (idx >= n || active[m] == 0u || iter_halt[m] != 0u) return;

    const int l  = idx / ng;
    const int ig = idx - l * ng;
    const long long vb = static_cast<long long>(m) * vec_stride;
    const long long mb = static_cast<long long>(m) * mat_stride;
    const long long cb = static_cast<long long>(m) * cpl_stride;

    double value = 0.0;
    for (int igs = 0; igs < ng; ++igs)
        value += diag[mb + static_cast<long long>(l) * ng * ng + ig * ng + igs] *
                 x[vb + l * ng + igs];

    for (int idir = 0; idir < NDIRMAX; ++idir) {
        for (int lr = 0; lr < LR; ++lr) {
            const int ln = neib[(l * NDIRMAX + idir) * LR + lr];
            if (ln == -1) continue;
            value += cc[cb + static_cast<long long>(l) * ng * NDIRMAX * LR +
                        ig * NDIRMAX * LR + idir * LR + lr] *
                     x[vb + ln * ng + ig];
        }
    }
    y[vb + idx] = value;
}

__global__ void cmfd_copy(const int n,
                          const int vec_stride,
                          const std::uint32_t* active,
                          const std::uint32_t* iter_halt,
                          const double* src,
                          double* dst) {
    const int m = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n || active[m] == 0u || iter_halt[m] != 0u) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    dst[base + i] = src[base + i];
}

__global__ void cmfd_subtract(const int n,
                              const int vec_stride,
                              const std::uint32_t* active,
                              const std::uint32_t* iter_halt,
                              const double* a,
                              const double* b,
                              double* out) {
    const int m = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n || active[m] == 0u || iter_halt[m] != 0u) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    out[base + i] = a[base + i] - b[base + i];
}

__global__ void cmfd_dot_terms(const int n,
                               const int vec_stride,
                               const std::uint32_t* active,
                               const std::uint32_t* iter_halt,
                               const double* a,
                               const double* b,
                               double* terms) {
    const int m = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n || active[m] == 0u || iter_halt[m] != 0u) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    terms[base + i] = a[base + i] * b[base + i];
}

__global__ void cmfd_sum_terms(const int n,
                               const int vec_stride,
                               const int scalar_stride,
                               const std::uint32_t* active,
                               const std::uint32_t* iter_halt,
                               const double* terms,
                               double* scalars,
                               int scalar_index) {
    const int m = static_cast<int>(blockIdx.y);
    if (threadIdx.x != 0 || active[m] == 0u || iter_halt[m] != 0u) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    double value = 0.0;
    for (int i = 0; i < n; ++i) value = value + terms[base + i];
    scalars[static_cast<long long>(m) * scalar_stride + scalar_index] = value;
}

__global__ void cmfd_init_norm(const int scalar_stride,
                               const std::uint32_t* active,
                               const std::uint32_t* iter_halt,
                               double* scalars) {
    const int m = static_cast<int>(blockIdx.y);
    if (threadIdx.x != 0 || active[m] == 0u || iter_halt[m] != 0u) return;
    double* sm = scalars + static_cast<long long>(m) * scalar_stride;
    sm[kNorm0] = sqrt(fmax(sm[kNorm], 0.0));
    sm[kNorm]  = sm[kNorm0];
}

__global__ void cmfd_check_convergence(const int scalar_stride,
                                       const std::uint32_t* active,
                                       std::uint32_t* iter_halt,
                                       int* iter_stop,
                                       double* scalars,
                                       int iteration) {
    const int m = static_cast<int>(blockIdx.y);
    if (threadIdx.x != 0 || active[m] == 0u || iter_halt[m] != 0u) return;
    double* sm = scalars + static_cast<long long>(m) * scalar_stride;
    sm[kNorm] = sqrt(fmax(sm[kNorm], 0.0));
    const double scale = sm[kNorm0] > 0.0 ? sm[kNorm0] : 1.0;
    if (sm[kNorm] / scale <= sm[kEps]) {
        iter_halt[m] = 1u;
        iter_stop[m] = iteration;
    }
}

__global__ void cmfd_prepare_p(const int n,
                               const int vec_stride,
                               const int scalar_stride,
                               const std::uint32_t* active,
                               const std::uint32_t* iter_halt,
                               const double* r,
                               const double* v,
                               double* p,
                               double* scalars,
                               int iteration) {
    const int m = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (active[m] == 0u || iter_halt[m] != 0u) return;
    double* sm = scalars + static_cast<long long>(m) * scalar_stride;
    if (i == 0) {
        sm[kRhoOld] = sm[kRhoNew];
        sm[kRhoNew] = 0.0;
    }
    if (i >= n) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    if (iteration == 0) {
        p[base + i] = r[base + i];
    } else {
        const double beta = sm[kBeta];
        p[base + i] = r[base + i] + beta * (p[base + i] - sm[kOmega] * v[base + i]);
    }
}

__global__ void cmfd_update_rho_beta(const int scalar_stride,
                                     const std::uint32_t* active,
                                     const std::uint32_t* iter_halt,
                                     double* scalars,
                                     int iteration) {
    const int m = static_cast<int>(blockIdx.y);
    if (threadIdx.x != 0 || active[m] == 0u || iter_halt[m] != 0u) return;
    double* sm = scalars + static_cast<long long>(m) * scalar_stride;
    if (iteration > 0) {
        const double denom = sm[kRhoOld] * sm[kOmega];
        sm[kBeta] = denom != 0.0 ? (sm[kRhoNew] / sm[kRhoOld]) *
                                      (sm[kAlpha] / sm[kOmega])
                                : 0.0;
    }
}

__global__ void cmfd_update_alpha(const int scalar_stride,
                                  const std::uint32_t* active,
                                  const std::uint32_t* iter_halt,
                                  double* scalars) {
    const int m = static_cast<int>(blockIdx.y);
    if (threadIdx.x != 0 || active[m] == 0u || iter_halt[m] != 0u) return;
    double* sm = scalars + static_cast<long long>(m) * scalar_stride;
    const double denom = sm[kGammaD];
    sm[kAlpha] = denom != 0.0 ? sm[kRhoNew] / denom : 0.0;
}

__global__ void cmfd_update_s(const int n,
                              const int vec_stride,
                              const int scalar_stride,
                              const std::uint32_t* active,
                              const std::uint32_t* iter_halt,
                              const double* r,
                              const double* v,
                              double* s,
                              const double* scalars) {
    const int m = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n || active[m] == 0u || iter_halt[m] != 0u) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    const double alpha = scalars[static_cast<long long>(m) * scalar_stride + kAlpha];
    s[base + i] = r[base + i] - alpha * v[base + i];
}

__global__ void cmfd_update_omega(const int scalar_stride,
                                  const std::uint32_t* active,
                                  const std::uint32_t* iter_halt,
                                  double* scalars) {
    const int m = static_cast<int>(blockIdx.y);
    if (threadIdx.x != 0 || active[m] == 0u || iter_halt[m] != 0u) return;
    double* sm = scalars + static_cast<long long>(m) * scalar_stride;
    sm[kOmega] = sm[kGammaD] != 0.0 ? sm[kGammaN] / sm[kGammaD] : 0.0;
}

__global__ void cmfd_update_solution(const int n,
                                     const int vec_stride,
                                     const int scalar_stride,
                                     const std::uint32_t* active,
                                     const std::uint32_t* iter_halt,
                                     double* phi,
                                     double* r,
                                     const double* p,
                                     const double* s,
                                     const double* v,
                                     const double* t,
                                     const double* scalars) {
    const int m = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n || active[m] == 0u || iter_halt[m] != 0u) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    const double* sm = scalars + static_cast<long long>(m) * scalar_stride;
    phi[base + i] += sm[kAlpha] * p[base + i] + sm[kOmega] * s[base + i];
    r[base + i] = s[base + i] - sm[kOmega] * t[base + i];
}

__global__ void cmfd_wiel_terms(const int nxyz,
                                const int ng,
                                const int vec_stride,
                                const std::uint32_t* active,
                                const std::uint32_t* sweep_halt,
                                const double* phi,
                                const double* psi,
                                const double* xsnf,
                                const double* vol,
                                double* terms_ab,
                                double* terms_c) {
    const int m = static_cast<int>(blockIdx.y);
    const int l = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (l >= nxyz || active[m] == 0u || sweep_halt[m] != 0u) return;
    const long long base = static_cast<long long>(m) * vec_stride;
    double next_psi = 0.0;
    for (int ig = 0; ig < ng; ++ig)
        next_psi += phi[base + l * ng + ig] * xsnf[base + l * ng + ig];
    next_psi *= vol[base + l];
    const double old_psi = psi[base + l];
    terms_ab[base + l] = (next_psi - old_psi) * (next_psi - old_psi);
    terms_ab[base + nxyz + l] = old_psi;
    terms_c[base + l] = next_psi;
}

__global__ void cmfd_wiel_finalize(const int nxyz,
                                   const int vec_stride,
                                   const std::uint32_t* sweep_halt,
                                   double* scalars,
                                   double* sweep_meta,
                                   const double* terms_ab,
                                   const double* terms_c) {
    const int m = static_cast<int>(blockIdx.y);
    if (sweep_halt[m] != 0u) return; // uniform for the whole slot block
    const int lane = static_cast<int>(threadIdx.x);
    const double* ta = terms_ab + m * vec_stride;
    const double* tc = terms_c + m * vec_stride;
    __shared__ double lane_sum[3];

    // Each independent sum retains the original l-ascending dependency chain.
    // Lanes 0, 1 and 2 therefore run concurrently without changing a sum's
    // operand pairing or deterministic double result.
    if (lane < 3) {
        const double* values = lane == 0 ? ta : (lane == 1 ? ta + nxyz : tc);
        double sum = 0.0;
        for (int l = 0; l < nxyz; ++l) sum = sum + values[l];
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;

    const double err    = lane_sum[0];
    const double gammad = lane_sum[1];
    const double gamman = lane_sum[2];
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    sm[kErrAcc] = err;
    sm[kGammaD] = gammad;
    sm[kGammaN] = gamman;

    double* meta = sweep_meta + static_cast<long long>(m) * kSweepCount;
    const double shift = meta[kSweepShift];
    const double eigv = meta[kSweepEigv];
    const double denom = gamman - gammad;
    const double gamma = denom != 0.0 ? gamman / denom : 0.0;
    meta[kSweepEigv] = shift != 0.0 ? shift + (eigv - shift) * gamma : eigv * gamma;
    meta[kSweepErrL2] = sqrt(fmax(err, 0.0));
    meta[kSweepNorm]  = gamman;
    if (!(meta[kSweepNorm0] > 0.0)) meta[kSweepNorm0] = gamman;
}

__global__ void cmfd_wiel_update_src(const int nxyz,
                                     const int ng,
                                     const int vec_stride,
                                     const std::uint32_t* active,
                                     const std::uint32_t* sweep_halt,
                                     const double* psi,
                                     double* src,
                                     double* sweep_meta) {
    const int m = static_cast<int>(blockIdx.y);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int n = nxyz * ng;
    if (idx >= n || active[m] == 0u || sweep_halt[m] != 0u) return;
    const int l = idx / ng;
    const long long base = static_cast<long long>(m) * vec_stride;
    const double eigv = sweep_meta[static_cast<long long>(m) * kSweepCount + kSweepEigv];
    src[base + idx] = psi[base + l] / eigv;
}

__global__ void cmfd_wiel_check(const int scalar_stride,
                                const std::uint32_t* active,
                                std::uint32_t* sweep_halt,
                                int* sweep_stop,
                                double* sweep_meta,
                                const double epsl2,
                                int iteration) {
    const int m = static_cast<int>(blockIdx.y);
    if (threadIdx.x != 0 || active[m] == 0u || sweep_halt[m] != 0u) return;
    double* meta = sweep_meta + static_cast<long long>(m) * kSweepCount;
    const double scale = meta[kSweepNorm0] > 0.0 ? meta[kSweepNorm0] : 1.0;
    if (meta[kSweepErrL2] / scale <= epsl2) {
        sweep_halt[m] = 1u;
        sweep_stop[m] = iteration;
    }
}

__global__ void cmfd_rb_sweep(const int nxyz,
                              const int ng,
                              const int vec_stride,
                              const int mat_stride,
                              const int cpl_stride,
                              const std::uint32_t* active,
                              const std::uint32_t* iter_halt,
                              const int* neib,
                              const int* rb_rows,
                              const int* rb_nodes,
                              const double* diag,
                              const double* cc,
                              const double* src,
                              double* phi,
                              int color,
                              int sweep) {
    const int m = static_cast<int>(blockIdx.y);
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (active[m] == 0u || iter_halt[m] != 0u) return;
    const int plane = nxyz + 1;
    const int begin = rb_rows[redblack_row_offset(color, plane, row)];
    const int end   = rb_rows[redblack_row_offset(color, plane, row + 1)];
    if (row >= nxyz || begin == end) return;

    const long long vb = static_cast<long long>(m) * vec_stride;
    const long long mb = static_cast<long long>(m) * mat_stride;
    const long long cb = static_cast<long long>(m) * cpl_stride;
    for (int at = begin; at < end; ++at) {
        const int l = rb_nodes[color * nxyz + at];
        for (int ig = 0; ig < ng; ++ig) {
            double rhs = src[vb + l * ng + ig];
            for (int igs = 0; igs < ng; ++igs) {
                if (igs == ig) continue;
                rhs -= diag[mb + static_cast<long long>(l) * ng * ng + ig * ng + igs] *
                       phi[vb + l * ng + igs];
            }
            for (int idir = 0; idir < NDIRMAX; ++idir) {
                for (int lr = 0; lr < LR; ++lr) {
                    const int ln = neib[(l * NDIRMAX + idir) * LR + lr];
                    if (ln == -1) continue;
                    rhs -= cc[cb + static_cast<long long>(l) * ng * NDIRMAX * LR +
                              ig * NDIRMAX * LR + idir * LR + lr] *
                           phi[vb + ln * ng + ig];
                }
            }
            const double diagonal = diag[mb + static_cast<long long>(l) * ng * ng +
                                         ig * ng + ig];
            if (diagonal != 0.0) phi[vb + l * ng + ig] = rhs / diagonal;
        }
    }
}

class CudaBatchArena {
public:
    struct Slot {
        bool          reserved = false;
        int           nxyz = 0;
        int           ng = 0;
        int           n = 0;
        size_t        matrix_count = 0;
        size_t        coupling_count = 0;
        const double* host_diag = nullptr;
        const double* host_cc = nullptr;
        const double* host_src = nullptr;
        double*       host_phi = nullptr;
        double        eps = 0.0;
        double        eps_on_device = std::numeric_limits<double>::quiet_NaN();
        MirroredUpload cc_mirror;
        MirroredUpload phi_mirror;
        bool          push_cc = true;
        bool          push_phi = true;
        double        sweep_in[kSweepCount]  = {};
        double        sweep_out[kSweepCount] = {};
        const double* host_xsnf = nullptr;
        const double* host_vol = nullptr;
        const double* host_psi = nullptr;
        double*       host_sweep_phi = nullptr;
        int           requested_outer = 0;
        bool          sweep_enabled = false;
        MirroredUpload xsnf_mirror;
        MirroredUpload vol_mirror;
        MirroredUpload psi_mirror;
        bool          push_xsnf = true;
        bool          push_vol = true;
        bool          push_psi = true;
        int           result = 0;
        std::string   error;
        bool          ready = false;
        bool          sweep_ready = false;
        bool          done = false;
        std::uint64_t generation = 0;
    };

    explicit CudaBatchArena(int width) : slots(width), slot(static_cast<size_t>(width)) {
        try {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUBLAS_CHECK(cublasCreate(&handle));
            CUBLAS_CHECK(cublasSetStream(handle, stream));
            CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
            available = true;
        } catch (const std::exception& e) {
            status = e.what();
            cleanup();
        }
    }

    ~CudaBatchArena() { cleanup(); }

    bool isAvailable() const { return available; }
    const std::string& statusMessage() const { return status; }

    int acquire(int nxyz, int ng) {
        std::unique_lock<std::mutex> lock(mutex);
        cv_slot.wait(lock, [&] {
            for (const Slot& s : slot)
                if (!s.reserved) return true;
            return false;
        });
        for (int i = 0; i < slots; ++i) {
            Slot& s = slot[static_cast<size_t>(i)];
            if (s.reserved) continue;
            s = Slot{};
            s.reserved = true;
            s.nxyz = nxyz;
            s.ng = ng;
            s.n = nxyz * ng;
            return i;
        }
        return -1;
    }

    void release(int index) {
        std::lock_guard<std::mutex> lock(mutex);
        slot[static_cast<size_t>(index)] = Slot{};
        cv_slot.notify_one();
    }

    void stageSlot(int index,
                   int nxyz,
                   int ng,
                   const double* host_diag,
                   const double* host_cc,
                   const double* host_src,
                   double* host_phi,
                   double eps) {
        std::lock_guard<std::mutex> lock(mutex);
        Slot& sl = slot[static_cast<size_t>(index)];
        sl.nxyz = nxyz;
        sl.ng = ng;
        sl.n = nxyz * ng;
        sl.matrix_count = static_cast<size_t>(nxyz) * static_cast<size_t>(ng) *
                          static_cast<size_t>(ng);
        sl.coupling_count = static_cast<size_t>(nxyz) * static_cast<size_t>(ng) *
                            static_cast<size_t>(NDIRMAX) * static_cast<size_t>(LR);
        sl.host_diag = host_diag;
        sl.host_cc   = host_cc;
        sl.host_src  = host_src;
        sl.host_phi  = host_phi;
        sl.eps       = eps;
        sl.push_cc   = !mirrorMatches(sl.cc_mirror, host_cc, sl.coupling_count);
        sl.push_phi  = !mirrorMatches(sl.phi_mirror, host_phi, static_cast<size_t>(sl.n));
        sl.ready = true;
        sl.done = false;
        sl.error.clear();
    }

    void stageSweep(int index,
                    int requested_outer,
                    double eigv,
                    double shift,
                    double errl2,
                    const double* host_diag,
                    const double* host_cc,
                    const double* host_src,
                    double* host_phi,
                    const double* host_xsnf,
                    const double* host_vol,
                    const double* host_psi) {
        std::lock_guard<std::mutex> lock(mutex);
        Slot& sl = slot[static_cast<size_t>(index)];
        sl.requested_outer = requested_outer;
        sl.sweep_enabled = true;
        sl.sweep_in[kSweepEigv]  = eigv;
        sl.sweep_in[kSweepShift] = shift;
        sl.sweep_in[kSweepErrL2] = errl2;
        sl.sweep_in[kSweepNorm0] = 0.0;
        sl.sweep_in[kSweepNorm]  = 0.0;
        sl.host_diag = host_diag;
        sl.host_cc = host_cc;
        sl.host_src = host_src;
        sl.host_sweep_phi = host_phi;
        sl.host_xsnf = host_xsnf;
        sl.host_vol = host_vol;
        sl.host_psi = host_psi;
        sl.push_cc = !mirrorMatches(sl.cc_mirror, host_cc, sl.coupling_count);
        sl.push_phi = !mirrorMatches(sl.phi_mirror, host_phi, static_cast<size_t>(sl.n));
        sl.push_xsnf = !mirrorMatches(sl.xsnf_mirror, host_xsnf, static_cast<size_t>(sl.n));
        sl.push_vol = !mirrorMatches(sl.vol_mirror, host_vol, static_cast<size_t>(sl.nxyz));
        sl.push_psi = !mirrorMatches(sl.psi_mirror, host_psi, static_cast<size_t>(sl.nxyz));
        sl.sweep_ready = true;
        sl.done = false;
        sl.error.clear();
    }

    int submit(int index) {
        std::unique_lock<std::mutex> lock(mutex);
        const std::uint64_t generation = slot[static_cast<size_t>(index)].generation;
        cv_ready.notify_all();
        cv_done.wait(lock, [&] {
            const Slot& s = slot[static_cast<size_t>(index)];
            return s.done && s.generation != generation;
        });
        return slot[static_cast<size_t>(index)].result;
    }

    int submitSweep(int index) {
        return submit(index);
    }

    void runLauncher() {
        for (;;) {
            std::vector<int> active;
            bool sweep = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_ready.wait(lock, [&] {
                    if (stop) return true;
                    for (const Slot& s : slot)
                        if ((s.ready || s.sweep_ready) && !s.done) return true;
                    return false;
                });
                if (stop) return;

                for (int i = 0; i < slots; ++i) {
                    const Slot& s = slot[static_cast<size_t>(i)];
                    if (s.sweep_ready && !s.done) {
                        sweep = true;
                        break;
                    }
                }

                const int wait_us = batchWaitMicros();
                if (wait_us > 0) {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
                    lock.lock();
                }

                for (int i = 0; i < slots; ++i) {
                    const Slot& s = slot[static_cast<size_t>(i)];
                    if (sweep ? (s.sweep_ready && !s.done) : (s.ready && !s.done))
                        active.push_back(i);
                }
                for (int i : active) {
                    Slot& s = slot[static_cast<size_t>(i)];
                    if (sweep) s.sweep_ready = false;
                    else s.ready = false;
                }
            }

            std::string error;
            try {
                ensureShape(active);
                if (sweep)
                    executeSweep(active);
                else
                    executePlain(active);
            } catch (const std::exception& e) {
                error = e.what();
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                for (int i : active) {
                    Slot& s = slot[static_cast<size_t>(i)];
                    s.error = error;
                    s.result = error.empty() ? 0 : 1;
                    s.done = true;
                    ++s.generation;
                }
            }
            cv_done.notify_all();
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv_ready.notify_all();
        cv_done.notify_all();
        cv_slot.notify_all();
    }

    void report() const {
        std::cout << "[RASBERY][CUDA][BACKEND_COUNTERS] {"
                  << "\"bulk_h2d_calls_during_iteration\":"
                  << telemetry.bulk_h2d_calls_during_iteration
                  << ",\"bulk_h2d_skipped_during_iteration\":"
                  << telemetry.bulk_h2d_skipped_during_iteration
                  << ",\"bulk_h2d_bytes_during_iteration\":"
                  << telemetry.bulk_h2d_bytes_during_iteration
                  << ",\"bulk_d2h_calls_during_iteration\":"
                  << telemetry.bulk_d2h_calls_during_iteration
                  << ",\"bulk_d2h_bytes_during_iteration\":"
                  << telemetry.bulk_d2h_bytes_during_iteration
                  << ",\"stream_sync_calls_during_iteration\":"
                  << telemetry.stream_sync_calls_during_iteration
                  << ",\"graph_launches\":" << telemetry.graph_launches
                  << ",\"graph_fallbacks\":" << telemetry.graph_fallbacks
                  << ",\"graph_failures\":" << telemetry.graph_failures
                  << ",\"direct_launches\":" << telemetry.direct_launches
                  << ",\"fixed_point_iterations\":"
                  << telemetry.fixed_point_iterations << "}" << std::endl;
    }

private:
    int slots = 0;
    std::vector<Slot> slot;
    bool available = false;
    bool stop = false;
    std::string status;
    mutable std::mutex mutex;
    std::condition_variable cv_ready;
    std::condition_variable cv_done;
    std::condition_variable cv_slot;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    std::thread launcher;

    int nxyz = 0;
    int ng = 0;
    int n = 0;
    size_t matrix_count = 0;
    size_t coupling_count = 0;
    int rb_sweeps = kDefaultRbSweeps;
    bool shape_ready = false;

    int* neib = nullptr;
    int* rb_rows = nullptr;
    int* rb_nodes = nullptr;
    double* diag = nullptr;
    double* cc = nullptr;
    double* src = nullptr;
    double* phi = nullptr;
    double* r = nullptr;
    double* r0 = nullptr;
    double* p = nullptr;
    double* v = nullptr;
    double* s = nullptr;
    double* t = nullptr;
    double* terms = nullptr;
    double* psi = nullptr;
    double* xsnf = nullptr;
    double* vol = nullptr;
    double* sweep_meta = nullptr;
    double* wiel_terms_ab = nullptr;
    double* wiel_terms_c = nullptr;
    double* scalars = nullptr;
    std::uint32_t* device_active = nullptr;
    std::uint32_t* iter_halt = nullptr;
    std::uint32_t* sweep_halt = nullptr;
    int* iter_stop = nullptr;
    int* sweep_stop = nullptr;
    std::uint32_t* host_active = nullptr;
    int* host_iter_stop = nullptr;
    int* host_sweep_stop = nullptr;
    double* host_sweep_meta = nullptr;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_ready = false;
    bool graph_failed = false;

    UploadMirrorTelemetry telemetry;

    void cleanup() {
        if (launcher.joinable()) {
            shutdown();
            launcher.join();
        }
        destroyGraph();
        freeShape();
        if (host_active != nullptr) cudaFreeHost(host_active);
        if (host_iter_stop != nullptr) cudaFreeHost(host_iter_stop);
        if (host_sweep_stop != nullptr) cudaFreeHost(host_sweep_stop);
        if (host_sweep_meta != nullptr) cudaFreeHost(host_sweep_meta);
        host_active = nullptr;
        host_iter_stop = nullptr;
        host_sweep_stop = nullptr;
        host_sweep_meta = nullptr;
        if (handle != nullptr) cublasDestroy(handle);
        handle = nullptr;
        if (stream != nullptr) cudaStreamDestroy(stream);
        stream = nullptr;
        available = false;
    }

    void destroyGraph() {
        if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
        graph_exec = nullptr;
        graph = nullptr;
        graph_ready = false;
    }

    void freeShape() {
        auto free_device = [](auto*& ptr) {
            if (ptr != nullptr) cudaFree(ptr);
            ptr = nullptr;
        };
        free_device(neib);
        free_device(rb_rows);
        free_device(rb_nodes);
        free_device(diag);
        free_device(cc);
        free_device(src);
        free_device(phi);
        free_device(r);
        free_device(r0);
        free_device(p);
        free_device(v);
        free_device(s);
        free_device(t);
        free_device(terms);
        free_device(psi);
        free_device(xsnf);
        free_device(vol);
        free_device(sweep_meta);
        free_device(wiel_terms_ab);
        free_device(wiel_terms_c);
        free_device(scalars);
        free_device(device_active);
        free_device(iter_halt);
        free_device(sweep_halt);
        free_device(iter_stop);
        free_device(sweep_stop);
        shape_ready = false;
        nxyz = ng = n = 0;
        matrix_count = coupling_count = 0;
    }

    long long vec_stride() const { return static_cast<long long>(std::max(2 * nxyz, n)); }
    long long mat_stride() const { return static_cast<long long>(matrix_count); }
    long long cpl_stride() const { return static_cast<long long>(coupling_count); }

    dim3 vec_grid() const {
        return dim3(static_cast<unsigned int>((n + kVecBlock - 1) / kVecBlock),
                    static_cast<unsigned int>(slots), 1u);
    }

    dim3 node_grid() const {
        return dim3(static_cast<unsigned int>((nxyz + kVecBlock - 1) / kVecBlock),
                    static_cast<unsigned int>(slots), 1u);
    }

    dim3 scalar_grid() const {
        return dim3(1u, static_cast<unsigned int>(slots), 1u);
    }

    void ensureShape(const std::vector<int>& active) {
        if (active.empty()) return;
        const Slot& first = slot[static_cast<size_t>(active.front())];
        for (int index : active) {
            const Slot& sl = slot[static_cast<size_t>(index)];
            if (sl.nxyz != first.nxyz || sl.ng != first.ng)
                throw std::runtime_error("batch slots must share one geometry");
        }
        if (shape_ready && nxyz == first.nxyz && ng == first.ng) return;

        destroyGraph();
        freeShape();
        nxyz = first.nxyz;
        ng = first.ng;
        n = nxyz * ng;
        matrix_count = static_cast<size_t>(nxyz) * static_cast<size_t>(ng) *
                       static_cast<size_t>(ng);
        coupling_count = static_cast<size_t>(nxyz) * static_cast<size_t>(ng) *
                         static_cast<size_t>(NDIRMAX) * static_cast<size_t>(LR);
        rb_sweeps = envInt("RASBERY_GPU_RB_SWEEPS", kDefaultRbSweeps);

        const long long vb = vec_stride();
        const long long mb = mat_stride();
        const long long cb = cpl_stride();
        CUDA_CHECK(cudaMalloc(&neib, static_cast<size_t>(nxyz * NDIRMAX * LR) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rb_rows, static_cast<size_t>(2 * (nxyz + 1)) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rb_nodes, static_cast<size_t>(2 * nxyz) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&diag, static_cast<size_t>(slots * mb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&cc, static_cast<size_t>(slots * cb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&src, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&phi, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&r, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&r0, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&v, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&t, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&terms, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&psi, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&xsnf, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&vol, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sweep_meta, static_cast<size_t>(slots * kSweepCount) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&wiel_terms_ab, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&wiel_terms_c, static_cast<size_t>(slots * vb) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&scalars, static_cast<size_t>(slots * kScalarCount) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&device_active, static_cast<size_t>(slots) * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMalloc(&iter_halt, static_cast<size_t>(slots) * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMalloc(&sweep_halt, static_cast<size_t>(slots) * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMalloc(&iter_stop, static_cast<size_t>(slots) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&sweep_stop, static_cast<size_t>(slots) * sizeof(int)));
        if (host_active == nullptr)
            CUDA_CHECK(cudaMallocHost(&host_active, static_cast<size_t>(slots) * sizeof(std::uint32_t)));
        if (host_iter_stop == nullptr)
            CUDA_CHECK(cudaMallocHost(&host_iter_stop, static_cast<size_t>(slots) * sizeof(int)));
        if (host_sweep_stop == nullptr)
            CUDA_CHECK(cudaMallocHost(&host_sweep_stop, static_cast<size_t>(slots) * sizeof(int)));
        if (host_sweep_meta == nullptr)
            CUDA_CHECK(cudaMallocHost(&host_sweep_meta,
                                      static_cast<size_t>(slots * kSweepCount) * sizeof(double)));

        std::vector<int> host_neib(static_cast<size_t>(nxyz * NDIRMAX * LR));
        std::vector<int> host_rb_rows(static_cast<size_t>(2 * (nxyz + 1)), 0);
        std::vector<int> host_rb_nodes(static_cast<size_t>(2 * nxyz), 0);
        const Slot& geometry_slot = slot[static_cast<size_t>(active.front())];
        const Geometry* geometry = nullptr;
        (void)geometry_slot;
        for (int l = 0; l < nxyz; ++l)
            for (int idir = 0; idir < NDIRMAX; ++idir)
                for (int lr = 0; lr < LR; ++lr)
                    host_neib[(l * NDIRMAX + idir) * LR + lr] = -1;
        // The topology is uploaded by setTopology() before the first solve.
        CUDA_CHECK(cudaMemsetAsync(neib,
                                   0xff,
                                   static_cast<size_t>(nxyz * NDIRMAX * LR) * sizeof(int),
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(rb_rows,
                                   0,
                                   static_cast<size_t>(2 * (nxyz + 1)) * sizeof(int),
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(rb_nodes,
                                   0,
                                   static_cast<size_t>(2 * nxyz) * sizeof(int),
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(device_active,
                                   0,
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        shape_ready = true;
        graph_failed = false;
        captureGraph();
    }

    void setTopology(const int* host_neib,
                     const int* host_rb_rows,
                     const int* host_rb_nodes) {
        CUDA_CHECK(cudaMemcpyAsync(neib,
                                   host_neib,
                                   static_cast<size_t>(nxyz * NDIRMAX * LR) * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(rb_rows,
                                   host_rb_rows,
                                   static_cast<size_t>(2 * (nxyz + 1)) * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(rb_nodes,
                                   host_rb_nodes,
                                   static_cast<size_t>(2 * nxyz) * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void pushOrSkip(double* device_buffer,
                    const double* host_buffer,
                    size_t count,
                    bool push) {
        if (!push) {
            ++telemetry.bulk_h2d_skipped_during_iteration;
            return;
        }
        const size_t bytes = count * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(device_buffer,
                                   host_buffer,
                                   bytes,
                                   cudaMemcpyHostToDevice,
                                   stream));
        ++telemetry.bulk_h2d_calls_during_iteration;
        telemetry.bulk_h2d_bytes_during_iteration += bytes;
    }

    void issueUploads(const int* active_slots, int count) {
        std::memset(host_active, 0, static_cast<size_t>(slots) * sizeof(std::uint32_t));
        for (int i = 0; i < count; ++i) host_active[active_slots[i]] = 1u;
        CUDA_CHECK(cudaMemcpyAsync(device_active,
                                   host_active,
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice,
                                   stream));

        for (int i = 0; i < count; ++i) {
            const int m = active_slots[i];
            Slot& sl = slot[static_cast<size_t>(m)];
            const long long vb = static_cast<long long>(m) * vec_stride();
            const long long mb = static_cast<long long>(m) * mat_stride();
            const long long cb = static_cast<long long>(m) * cpl_stride();
            CUDA_CHECK(cudaMemcpyAsync(diag + mb,
                                       sl.host_diag,
                                       matrix_count * sizeof(double),
                                       cudaMemcpyHostToDevice,
                                       stream));
            ++telemetry.bulk_h2d_calls_during_iteration;
            telemetry.bulk_h2d_bytes_during_iteration += matrix_count * sizeof(double);
            pushOrSkip(cc + cb, sl.host_cc, coupling_count, sl.push_cc);
            CUDA_CHECK(cudaMemcpyAsync(src + vb,
                                       sl.host_src,
                                       static_cast<size_t>(n) * sizeof(double),
                                       cudaMemcpyHostToDevice,
                                       stream));
            ++telemetry.bulk_h2d_calls_during_iteration;
            telemetry.bulk_h2d_bytes_during_iteration += static_cast<size_t>(n) * sizeof(double);
            pushOrSkip(phi + vb, sl.host_phi, static_cast<size_t>(n), sl.push_phi);
            if (!(sl.eps == sl.eps_on_device)) {
                CUDA_CHECK(cudaMemcpyAsync(scalars + static_cast<long long>(m) * kScalarCount + kEps,
                                           &sl.eps,
                                           sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           stream));
                // Graph/direct kernels are submitted to this same stream, so
                // stream order publishes eps without draining the pipeline.
                sl.eps_on_device = sl.eps;
            }
            sl.phi_mirror.valid = false;
        }
    }

    void issueSweepUploads(const int* active_slots, int count) {
        issueUploads(active_slots, count);
        for (int i = 0; i < count; ++i) {
            const int m = active_slots[i];
            Slot& sl = slot[static_cast<size_t>(m)];
            const long long vb = static_cast<long long>(m) * vec_stride();
            pushOrSkip(xsnf + vb, sl.host_xsnf, static_cast<size_t>(n), sl.push_xsnf);
            pushOrSkip(vol + vb, sl.host_vol, static_cast<size_t>(nxyz), sl.push_vol);
            pushOrSkip(psi + vb, sl.host_psi, static_cast<size_t>(nxyz), sl.push_psi);
            CUDA_CHECK(cudaMemcpyAsync(sweep_meta + static_cast<long long>(m) * kSweepCount,
                                       sl.sweep_in,
                                       kSweepCount * sizeof(double),
                                       cudaMemcpyHostToDevice,
                                       stream));
            ++telemetry.bulk_h2d_calls_during_iteration;
            telemetry.bulk_h2d_bytes_during_iteration += kSweepCount * sizeof(double);
        }
    }

    void captureGraph() {
        if (graph_failed) return;
        destroyGraph();
        try {
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launchFixedPointGraphBody();
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
            graph_ready = true;
        } catch (const std::exception&) {
            cudaGetLastError();
            if (graph != nullptr) cudaGraphDestroy(graph);
            graph = nullptr;
            graph_exec = nullptr;
            graph_ready = false;
            graph_failed = true;
            ++telemetry.graph_failures;
        }
    }

    void launchFixedPointGraphBody() {
        const dim3 vg = vec_grid();
        const dim3 sg = scalar_grid();
        cmfd_reset_iteration<<<vg, kVecBlock, 0, stream>>>(
            n, static_cast<int>(vec_stride()), kScalarCount, device_active,
            r, r0, p, v, scalars, iter_halt, iter_stop);
        cmfd_apply_operator<<<vg, kVecBlock, 0, stream>>>(
            nxyz, ng, static_cast<int>(vec_stride()), static_cast<int>(mat_stride()),
            static_cast<int>(cpl_stride()), device_active, iter_halt, neib, diag, cc, phi, v);
        cmfd_subtract<<<vg, kVecBlock, 0, stream>>>(
            n, static_cast<int>(vec_stride()), device_active, iter_halt, src, v, r);
        cmfd_copy<<<vg, kVecBlock, 0, stream>>>(
            n, static_cast<int>(vec_stride()), device_active, iter_halt, r, r0);
        cmfd_dot_terms<<<vg, kVecBlock, 0, stream>>>(
            n, static_cast<int>(vec_stride()), device_active, iter_halt, r, r, terms);
        cmfd_sum_terms<<<sg, 1, 0, stream>>>(
            n, static_cast<int>(vec_stride()), kScalarCount, device_active, iter_halt,
            terms, scalars, kNorm);
        cmfd_init_norm<<<sg, 1, 0, stream>>>(
            kScalarCount, device_active, iter_halt, scalars);

        for (int iteration = 0; iteration < kMaxFixedPoint; ++iteration) {
            cmfd_prepare_p<<<vg, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                iter_halt, r, v, p, scalars, iteration);
            cmfd_dot_terms<<<vg, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), device_active, iter_halt, r0, r, terms);
            cmfd_sum_terms<<<sg, 1, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                iter_halt, terms, scalars, kRhoNew);
            cmfd_update_rho_beta<<<sg, 1, 0, stream>>>(
                kScalarCount, device_active, iter_halt, scalars, iteration);
            cmfd_apply_operator<<<vg, kVecBlock, 0, stream>>>(
                nxyz, ng, static_cast<int>(vec_stride()), static_cast<int>(mat_stride()),
                static_cast<int>(cpl_stride()), device_active, iter_halt, neib,
                diag, cc, p, v);
            cmfd_dot_terms<<<vg, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), device_active, iter_halt, r0, v, terms);
            cmfd_sum_terms<<<sg, 1, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                iter_halt, terms, scalars, kGammaD);
            cmfd_update_alpha<<<sg, 1, 0, stream>>>(
                kScalarCount, device_active, iter_halt, scalars);
            cmfd_update_s<<<vg, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                iter_halt, r, v, s, scalars);
            cmfd_apply_operator<<<vg, kVecBlock, 0, stream>>>(
                nxyz, ng, static_cast<int>(vec_stride()), static_cast<int>(mat_stride()),
                static_cast<int>(cpl_stride()), device_active, iter_halt, neib,
                diag, cc, s, t);
            cmfd_dot_terms<<<vg, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), device_active, iter_halt, t, s, terms);
            cmfd_sum_terms<<<sg, 1, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                iter_halt, terms, scalars, kGammaN);
            cmfd_dot_terms<<<vg, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), device_active, iter_halt, t, t, terms);
            cmfd_sum_terms<<<sg, 1, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                iter_halt, terms, scalars, kGammaD);
            cmfd_update_omega<<<sg, 1, 0, stream>>>(
                kScalarCount, device_active, iter_halt, scalars);
            cmfd_update_solution<<<vg, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                iter_halt, phi, r, p, s, v, t, scalars);
            cmfd_dot_terms<<<vg, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), device_active, iter_halt, r, r, terms);
            cmfd_sum_terms<<<sg, 1, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                iter_halt, terms, scalars, kNorm);
            cmfd_check_convergence<<<sg, 1, 0, stream>>>(
                kScalarCount, device_active, iter_halt, iter_stop, scalars, iteration + 1);
        }
    }

    void launchFixedPoint() {
        if (graph_ready) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
            ++telemetry.graph_launches;
        } else {
            ++telemetry.graph_fallbacks;
            launchFixedPointGraphBody();
            ++telemetry.direct_launches;
        }
    }

    void executePlain(const std::vector<int>& active) {
        issueUploads(active.data(), static_cast<int>(active.size()));
        launchFixedPoint();
        CUDA_CHECK(cudaMemcpyAsync(host_iter_stop,
                                   iter_stop,
                                   static_cast<size_t>(slots) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        ++telemetry.bulk_d2h_calls_during_iteration;
        telemetry.bulk_d2h_bytes_during_iteration += static_cast<size_t>(slots) * sizeof(int);
        for (int m : active) {
            Slot& sl = slot[static_cast<size_t>(m)];
            CUDA_CHECK(cudaMemcpyAsync(sl.host_phi,
                                       phi + static_cast<long long>(m) * vec_stride(),
                                       static_cast<size_t>(n) * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            ++telemetry.bulk_d2h_calls_during_iteration;
            telemetry.bulk_d2h_bytes_during_iteration += static_cast<size_t>(n) * sizeof(double);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++telemetry.stream_sync_calls_during_iteration;
        for (int m : active) {
            Slot& sl = slot[static_cast<size_t>(m)];
            rememberMirror(sl.cc_mirror, sl.host_cc, coupling_count);
            rememberMirror(sl.phi_mirror, sl.host_phi, static_cast<size_t>(n));
            sl.result = host_iter_stop[m] < kMaxFixedPoint ? host_iter_stop[m] : kMaxFixedPoint;
            telemetry.fixed_point_iterations += static_cast<std::uint64_t>(sl.result);
        }
    }

    void executeSweep(const std::vector<int>& active) {
        issueSweepUploads(active.data(), static_cast<int>(active.size()));
        std::memset(host_active, 0, static_cast<size_t>(slots) * sizeof(std::uint32_t));
        for (int m : active) host_active[m] = 1u;
        CUDA_CHECK(cudaMemcpyAsync(sweep_halt,
                                   host_active,
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(sweep_stop,
                                   0xff,
                                   static_cast<size_t>(slots) * sizeof(int),
                                   stream));
        const dim3 ng_grid = vec_grid();
        const dim3 n_grid = node_grid();
        const dim3 scalar = scalar_grid();
        for (int outer = 0; outer < kMaxFixedPoint; ++outer) {
            cmfd_reset_iteration<<<ng_grid, kVecBlock, 0, stream>>>(
                n, static_cast<int>(vec_stride()), kScalarCount, device_active,
                r, r0, p, v, scalars, iter_halt, iter_stop);
            for (int sweep_i = 0; sweep_i < rb_sweeps; ++sweep_i) {
                cmfd_rb_sweep<<<n_grid, kVecBlock, 0, stream>>>(
                    nxyz, ng, static_cast<int>(vec_stride()), static_cast<int>(mat_stride()),
                    static_cast<int>(cpl_stride()), device_active, iter_halt, neib,
                    rb_rows, rb_nodes, diag, cc, src, phi, 0, sweep_i);
                cmfd_rb_sweep<<<n_grid, kVecBlock, 0, stream>>>(
                    nxyz, ng, static_cast<int>(vec_stride()), static_cast<int>(mat_stride()),
                    static_cast<int>(cpl_stride()), device_active, iter_halt, neib,
                    rb_rows, rb_nodes, diag, cc, src, phi, 1, sweep_i);
            }
            cmfd_wiel_terms<<<n_grid, kVecBlock, 0, stream>>>(
                nxyz, ng, static_cast<int>(vec_stride()), device_active, sweep_halt,
                phi, psi, xsnf, vol, wiel_terms_ab, wiel_terms_c);
            cmfd_wiel_finalize<<<scalar_grid(), 32, 0, stream>>>(
                nxyz, static_cast<int>(vec_stride()), sweep_halt, scalars, sweep_meta,
                wiel_terms_ab, wiel_terms_c);
            cmfd_wiel_update_src<<<ng_grid, kVecBlock, 0, stream>>>(
                nxyz, ng, static_cast<int>(vec_stride()), device_active, sweep_halt,
                psi, src, sweep_meta);
            cmfd_wiel_check<<<scalar, 1, 0, stream>>>(
                kScalarCount, device_active, sweep_halt, sweep_stop, sweep_meta,
                1.0e-5, outer + 1);
        }
        CUDA_CHECK(cudaMemcpyAsync(host_sweep_stop,
                                   sweep_stop,
                                   static_cast<size_t>(slots) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(host_sweep_meta,
                                   sweep_meta,
                                   static_cast<size_t>(slots * kSweepCount) * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        ++telemetry.bulk_d2h_calls_during_iteration;
        telemetry.bulk_d2h_bytes_during_iteration +=
            static_cast<size_t>(slots) * sizeof(int) +
            static_cast<size_t>(slots * kSweepCount) * sizeof(double);
        for (int m : active) {
            Slot& sl = slot[static_cast<size_t>(m)];
            CUDA_CHECK(cudaMemcpyAsync(sl.host_sweep_phi,
                                       phi + static_cast<long long>(m) * vec_stride(),
                                       static_cast<size_t>(n) * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            ++telemetry.bulk_d2h_calls_during_iteration;
            telemetry.bulk_d2h_bytes_during_iteration += static_cast<size_t>(n) * sizeof(double);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++telemetry.stream_sync_calls_during_iteration;
        for (int m : active) {
            Slot& sl = slot[static_cast<size_t>(m)];
            std::copy(host_sweep_meta + static_cast<long long>(m) * kSweepCount,
                      host_sweep_meta + static_cast<long long>(m + 1) * kSweepCount,
                      sl.sweep_out);
            rememberMirror(sl.cc_mirror, sl.host_cc, coupling_count);
            rememberMirror(sl.phi_mirror, sl.host_sweep_phi, static_cast<size_t>(n));
            rememberMirror(sl.xsnf_mirror, sl.host_xsnf, static_cast<size_t>(n));
            rememberMirror(sl.vol_mirror, sl.host_vol, static_cast<size_t>(nxyz));
            rememberMirror(sl.psi_mirror, sl.host_psi, static_cast<size_t>(nxyz));
            sl.result = host_sweep_stop[m] >= 0 ? host_sweep_stop[m] : kMaxFixedPoint;
            telemetry.fixed_point_iterations += static_cast<std::uint64_t>(sl.result);
        }
    }

public:
    void startLauncher() {
        if (launcher.joinable()) return;
        launcher = std::thread([this] { runLauncher(); });
    }

    void uploadTopology(const int* host_neib,
                        const int* host_rb_rows,
                        const int* host_rb_nodes) {
        std::lock_guard<std::mutex> lock(mutex);
        setTopology(host_neib, host_rb_rows, host_rb_nodes);
    }

    Slot snapshot(int index) const {
        std::lock_guard<std::mutex> lock(mutex);
        return slot[static_cast<size_t>(index)];
    }
};

std::mutex g_arena_mutex;
std::unique_ptr<CudaBatchArena> g_arena;
int g_batch_width = 0;

CudaBatchArena* arena() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    return g_arena.get();
}

void ensureArena() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (g_arena != nullptr || g_batch_width <= 0) return;
    g_arena = std::make_unique<CudaBatchArena>(g_batch_width);
    if (g_arena->isAvailable()) g_arena->startLauncher();
}

} // namespace

void rasberySetBatchWidth(int width) {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    g_batch_width = std::max(0, width);
}

void rasberyReleaseBatchArena() {
    std::unique_ptr<CudaBatchArena> local;
    {
        std::lock_guard<std::mutex> lock(g_arena_mutex);
        local = std::move(g_arena);
        g_batch_width = 0;
    }
    if (local != nullptr) {
        local->report();
        local->shutdown();
    }
}

bool rasberyCudaBatchEnabled() {
    return g_batch_width > 0 && envEnabled("RASBERY_GPU");
}

int rasberyCudaBatchAcquire(int nxyz, int ng) {
    if (!rasberyCudaBatchEnabled()) return -1;
    ensureArena();
    CudaBatchArena* a = arena();
    if (a == nullptr || !a->isAvailable()) return -1;
    return a->acquire(nxyz, ng);
}

void rasberyCudaBatchRelease(int slot) {
    CudaBatchArena* a = arena();
    if (a != nullptr && slot >= 0) a->release(slot);
}

int rasberyCudaBatchSolve(int slot,
                          int nxyz,
                          int ng,
                          const double* diag,
                          const double* cc,
                          const double* src,
                          double* phi,
                          double eps) {
    CudaBatchArena* a = arena();
    if (a == nullptr || slot < 0) return -1;
    a->stageSlot(slot, nxyz, ng, diag, cc, src, phi, eps);
    return a->submit(slot);
}

int rasberyCudaBatchSweep(int slot,
                          int requested_outer,
                          double& eigv,
                          double shift,
                          double& errl2,
                          const double* diag,
                          const double* cc,
                          const double* src,
                          double* phi,
                          const double* xsnf,
                          const double* vol,
                          const double* psi) {
    CudaBatchArena* a = arena();
    if (a == nullptr || slot < 0) return -1;
    a->stageSweep(slot, requested_outer, eigv, shift, errl2,
                  diag, cc, src, phi, xsnf, vol, psi);
    const int result = a->submitSweep(slot);
    const CudaBatchArena::Slot snapshot = a->snapshot(slot);
    eigv  = snapshot.sweep_out[kSweepEigv];
    errl2 = snapshot.sweep_out[kSweepErrL2];
    return result;
}

} // namespace rasbery

#else

namespace rasbery {
void rasberySetBatchWidth(int) {}
void rasberyReleaseBatchArena() {}
bool rasberyCudaBatchEnabled() { return false; }
int rasberyCudaBatchAcquire(int, int) { return -1; }
void rasberyCudaBatchRelease(int) {}
int rasberyCudaBatchSolve(int,
                          int,
                          int,
                          const double*,
                          const double*,
                          const double*,
                          double*,
                          double) {
    return -1;
}
int rasberyCudaBatchSweep(int,
                          int,
                          double&,
                          double,
                          double&,
                          const double*,
                          const double*,
                          const double*,
                          double*,
                          const double*,
                          const double*,
                          const double*) {
    return -1;
}
} // namespace rasbery

#endif
