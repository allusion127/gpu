#include "CudaXsReconBackend.h"

#include "XsReconKernel.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// This translation unit must be compiled with --fmad=false (set in
// CMakeLists.txt).  The kernel's contract is bit-identical reproduction of the
// host loop; letting nvcc contract a*b+c would make the comparison depend on
// which expressions each compiler chose to fuse.  See XsReconKernel.h.

namespace rasbery {

namespace xsr = rasbery::xsrecon;

namespace {

std::atomic<unsigned long long> g_nodes_solved{0};

bool envFlagEnabled(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string s(v);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE");
}

#define RASBERY_CUDA_TRY(expr, sink)                                         \
    do {                                                                     \
        const cudaError_t _e = (expr);                                       \
        if (_e != cudaSuccess) {                                             \
            (sink) = std::string(#expr) + " -> " + cudaGetErrorString(_e);   \
            return false;                                                    \
        }                                                                    \
    } while (0)

// max_change >= 0 always (|dXe| / max(|Xe|, 1e-30)), and for non-negative
// IEEE doubles the unsigned-integer order of the bit patterns is the value
// order, so a 64-bit atomicMax is an exact, order-insensitive max reduction.
__global__ void kernelXsRecon(xsr::BatchView v, unsigned long long* max_bits,
                              unsigned long long* solved) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.n_fuel) return;
    const int l = v.fuel[i];

    double mc = 0.0;
    if (xsreconSolveNode(v, l, &mc)) {
        atomicMax(max_bits, static_cast<unsigned long long>(__double_as_longlong(mc)));
        atomicAdd(solved, 1ULL);
    }
}

} // namespace

struct XsReconBackend::Impl {
    bool          available = false;
    std::string   status    = "not initialised";
    cudaStream_t  stream    = nullptr;

    int nxyz   = 0;
    int n_fuel = 0;

    double*             dev_block   = nullptr;
    std::size_t         block_doubles = 0;
    int*                dev_fuel    = nullptr;
    unsigned long long* dev_scalars = nullptr; // [0]=max bits, [1]=solved
    double*             dev_dep     = nullptr; // depTrans rows: [0..38]=I135, [39..77]=Xe135

    unsigned long long resident_micx_generation = 0; // 0 = nothing resident
    bool               fuel_uploaded            = false;

    // Offsets into dev_block, in doubles.
    std::size_t off_mic[xsr::NXS] = {};
    std::size_t off_mic_ssm       = 0;
    std::size_t off_lmp[xsr::NXS] = {};
    std::size_t off_lmp_ssm       = 0;
    std::size_t off_iden          = 0;
    std::size_t off_xs[xsr::NXS]  = {};
    std::size_t off_xs_ssm        = 0;
    std::size_t off_phif          = 0;

    ~Impl() {
        if (dev_block) cudaFree(dev_block);
        if (dev_fuel) cudaFree(dev_fuel);
        if (dev_scalars) cudaFree(dev_scalars);
        if (dev_dep) cudaFree(dev_dep);
        if (stream) cudaStreamDestroy(stream);
    }

    bool ensure(int want_nxyz, int want_fuel) {
        if (want_nxyz == nxyz && dev_block != nullptr && want_fuel <= n_fuel) {
            n_fuel = want_fuel;
            return true;
        }
        if (dev_block) { cudaFree(dev_block); dev_block = nullptr; }
        if (dev_fuel) { cudaFree(dev_fuel); dev_fuel = nullptr; }
        resident_micx_generation = 0;
        fuel_uploaded            = false;

        nxyz   = want_nxyz;
        n_fuel = want_fuel;

        const std::size_t nx  = static_cast<std::size_t>(nxyz);
        const std::size_t mic = static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx;
        const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
        const std::size_t msm = static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx;
        const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;

        std::size_t off = 0;
        for (int xt = 0; xt < xsr::NXS; ++xt) { off_mic[xt] = off; off += mic; }
        off_mic_ssm = off; off += msm;
        for (int xt = 0; xt < xsr::NXS; ++xt) { off_lmp[xt] = off; off += lmp; }
        off_lmp_ssm = off; off += ssm;
        off_iden = off; off += static_cast<std::size_t>(xsr::NISO) * nx;
        for (int xt = 0; xt < xsr::NXS; ++xt) { off_xs[xt] = off; off += lmp; }
        off_xs_ssm = off; off += ssm;
        off_phif = off; off += static_cast<std::size_t>(xsr::NG) * nx;
        block_doubles = off;

        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev_block),
                                    block_doubles * sizeof(double)), status);
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev_fuel),
                                    static_cast<std::size_t>(nxyz) * sizeof(int)), status);
        return true;
    }

    bool upload(const double* src, std::size_t off, std::size_t count) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(dev_block + off, src, count * sizeof(double),
                                         cudaMemcpyHostToDevice, stream), status);
        return true;
    }

    bool download(double* dst, std::size_t off, std::size_t count) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(dst, dev_block + off, count * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream), status);
        return true;
    }
};

XsReconBackend::XsReconBackend() : _impl(std::make_unique<Impl>()) {
    if (!rasberyGpuXsReconEnabled()) {
        _impl->status = "disabled (RASBERY_GPU_XSRECON unset)";
        return;
    }
    int count = 0;
    const cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        _impl->status = std::string("no CUDA device: ") +
                        (e == cudaSuccess ? "count 0" : cudaGetErrorString(e));
        return;
    }
    const cudaError_t se = cudaStreamCreateWithFlags(&_impl->stream, cudaStreamNonBlocking);
    if (se != cudaSuccess) {
        _impl->status = std::string("stream: ") + cudaGetErrorString(se);
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&_impl->dev_scalars),
                   2 * sizeof(unsigned long long)) != cudaSuccess) {
        _impl->status = "scalar buffer allocation failed";
        return;
    }
    _impl->available = true;
    _impl->status    = "ready";
}

XsReconBackend::~XsReconBackend() = default;

bool XsReconBackend::available() const { return _impl->available; }
const std::string& XsReconBackend::status() const { return _impl->status; }

bool XsReconBackend::solve(const xsr::BatchView& host, unsigned long long micx_generation,
                           double* max_change_out) {
    Impl& d = *_impl;
    if (!d.available || host.n_fuel <= 0 || host.nxyz <= 0) return false;
    if (!d.ensure(host.nxyz, host.n_fuel)) {
        d.available = false; // one allocation failure disables the instance
        return false;
    }

    const std::size_t nx  = static_cast<std::size_t>(d.nxyz);
    const std::size_t mic = static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx;
    const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
    const std::size_t msm = static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx;
    const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;

    if (!d.fuel_uploaded) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_fuel, host.fuel,
                                         static_cast<std::size_t>(host.n_fuel) * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        d.fuel_uploaded = true;
    }

    // _micx and _lmpx move together (both are outputs of the same host-side
    // rebuild paths), so one generation covers both.
    if (micx_generation != d.resident_micx_generation) {
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.mic[xt], d.off_mic[xt], mic)) return false;
        if (!d.upload(host.mic_ssm, d.off_mic_ssm, msm)) return false;
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.lmp[xt], d.off_lmp[xt], lmp)) return false;
        if (!d.upload(host.lmp_ssm, d.off_lmp_ssm, ssm)) return false;
        d.resident_micx_generation = micx_generation;
    }

    // Per-call state.  _iden and _xs are uploaded whole so the kernel's
    // fuel-only writes round-trip the non-fuel entries unchanged, keeping the
    // host arrays authoritative for every node after the download.
    if (!d.upload(host.iden, d.off_iden, static_cast<std::size_t>(xsr::NISO) * nx)) return false;
    for (int xt = 0; xt < xsr::NXS; ++xt)
        if (!d.upload(host.xs[xt], d.off_xs[xt], lmp)) return false;
    if (!d.upload(host.xs_ssm, d.off_xs_ssm, ssm)) return false;
    if (!d.upload(host.phif, d.off_phif, static_cast<std::size_t>(xsr::NG) * nx)) return false;

    RASBERY_CUDA_TRY(cudaMemsetAsync(d.dev_scalars, 0, 2 * sizeof(unsigned long long),
                                     d.stream), d.status);

    xsr::BatchView v = host;
    for (int xt = 0; xt < xsr::NXS; ++xt) {
        v.mic[xt] = d.dev_block + d.off_mic[xt];
        v.lmp[xt] = d.dev_block + d.off_lmp[xt];
        v.xs[xt]  = d.dev_block + d.off_xs[xt];
    }
    v.mic_ssm = d.dev_block + d.off_mic_ssm;
    v.lmp_ssm = d.dev_block + d.off_lmp_ssm;
    v.iden    = d.dev_block + d.off_iden;
    v.xs_ssm  = d.dev_block + d.off_xs_ssm;
    v.phif    = d.dev_block + d.off_phif;
    v.fuel    = d.dev_fuel;

    // depTrans rows: 39 doubles each, constant for the process, but owned per
    // instance for simplicity.  Upload every call -- 624 bytes on a stream
    // that is about to move megabytes.
    if (d.dev_dep == nullptr) {
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_dep),
                                    2 * xsr::NISO * sizeof(double)), d.status);
    }
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_dep, host.dep_i135,
                                     xsr::NISO * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_dep + xsr::NISO, host.dep_xe135,
                                     xsr::NISO * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    v.dep_i135  = d.dev_dep;
    v.dep_xe135 = d.dev_dep + xsr::NISO;

    const int block = 128;
    const int grid  = (host.n_fuel + block - 1) / block;
    kernelXsRecon<<<grid, block, 0, d.stream>>>(v, d.dev_scalars, d.dev_scalars + 1);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    // Results the host needs back: the reconstructed xs, the three Xe-chain
    // density rows (contiguous, iso-major), and the two scalars.
    for (int xt = 0; xt < xsr::NXS; ++xt)
        if (!d.download(host.xs[xt], d.off_xs[xt], lmp)) return false;
    if (!d.download(host.xs_ssm, d.off_xs_ssm, ssm)) return false;
    if (!d.download(host.iden + static_cast<std::size_t>(xsr::I135) * nx,
                    d.off_iden + static_cast<std::size_t>(xsr::I135) * nx, 3 * nx))
        return false;

    unsigned long long scalars[2] = {0, 0};
    RASBERY_CUDA_TRY(cudaMemcpyAsync(scalars, d.dev_scalars,
                                     2 * sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);

    double max_change;
    static_assert(sizeof(max_change) == sizeof(scalars[0]), "bit width");
    std::memcpy(&max_change, &scalars[0], sizeof(max_change));
    *max_change_out = max_change;

    g_nodes_solved.fetch_add(scalars[1], std::memory_order_relaxed);
    return true;
}

unsigned long long XsReconBackend::nodesSolved() {
    return g_nodes_solved.load(std::memory_order_relaxed);
}

void XsReconBackend::pinHost(const void* p, size_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const cudaError_t rc =
        cudaHostRegister(const_cast<void*>(p), bytes, cudaHostRegisterDefault);
    if (rc != cudaSuccess) cudaGetLastError(); // already registered / exotic host
}

bool rasberyGpuXsReconEnabled() {
    static const bool on = envFlagEnabled("RASBERY_GPU_XSRECON");
    return on;
}

unsigned long long rasberyGpuXsReconNodes() {
    return g_nodes_solved.load(std::memory_order_relaxed);
}

} // namespace rasbery
