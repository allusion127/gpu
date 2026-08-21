#include "CudaXsReconBackend.h"

#include "FlatXsKernel.h"
#include "NodalKernel.h"
#include "XsReconKernel.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

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

namespace fxs = rasbery::flatxs;

std::atomic<unsigned long long> g_flatxs_nodes_solved{0};

// One thread per target node; the shared body does the rest.  StaticForms
// folds the mined mask at compile time (this TU builds with --fmad=false, so
// only the explicit fma() arms fuse).
__global__ void kernelFlatXs(fxs::FlatXsView v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.n_nodes) return;
    fxs::flatxsSolveNode(v, i, fxs::StaticForms{});
}

namespace ndl = rasbery::nodal;

std::atomic<unsigned long long> g_nodal_drives{0};

__global__ void kNodalTrl0(ndl::NodalView v) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalTrlcff0(v, lk);
}
__global__ void kNodalTrl12(ndl::NodalView v) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalTrlcff12(v, lk, ndl::StaticForms{});
}
__global__ void kNodalMat(ndl::NodalView v) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalUpdateMatrix(v, lk, ndl::StaticForms{});
}
__global__ void kNodalEven(ndl::NodalView v) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalCalculateEven(v, lk, ndl::StaticForms{});
}
__global__ void kNodalJnet(ndl::NodalView v) {
    const int ls = blockIdx.x * blockDim.x + threadIdx.x;
    if (ls < v.nsurf) ndl::nodalCalculateJnet(v, ls, ndl::StaticForms{});
}

// The 9 ACTIVE_XT slots of XSSet.cpp, as Chiffon::XSTYPE values.  The enum
// order is pinned by the static_assert-style constants in XsReconKernel.h
// (T_XSTF..T_XS3N); XSDF and XSRF are derived and skipped.
constexpr int ACTIVE_XT9[fxs::N_ACTIVE] = {
    xsr::T_XSTF, xsr::T_XSAF, xsr::T_XSFF, xsr::T_XSNF, xsr::T_XSKF,
    xsr::T_XSSF, xsr::T_FYLD, xsr::T_XS2N, xsr::T_XS3N};

std::uint64_t fnvMix(const void* p, std::size_t bytes, std::uint64_t h) {
    const unsigned char* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < bytes; ++i)
        h = (h ^ c[i]) * 1099511628211ULL;
    return h;
}

/// Process-wide device copies of the flat coefficient tables, shared across
/// the batch instances (the tables are immutable after library load, and the
/// 64 instances of one benchmark load the same library).  Keyed by a full
/// FNV-1a over sizes and bytes; the uploader copies synchronously while
/// holding the mutex, so a concurrent instance can never launch against a
/// half-uploaded table.
struct FlatXsLibDevice {
    std::uint64_t       hash = 0;
    double*             block = nullptr; // [9 lmp | lsm | 9 mic | msm | knots]
    fxs::DeltaMeta*     deltas = nullptr;
    std::size_t         off_lmp[fxs::N_ACTIVE] = {};
    std::size_t         off_lsm = 0;
    std::size_t         off_mic[fxs::N_ACTIVE] = {};
    std::size_t         off_msm = 0;
    std::size_t         off_knots = 0;
};

std::mutex                    g_flatxs_lib_mutex;
std::vector<FlatXsLibDevice>* g_flatxs_libs = nullptr; // leaked on purpose (process lifetime)

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

    unsigned long long resident_micx_generation  = 0; // 0 = nothing resident
    unsigned long long resident_state_generation = 0; // _xs/_iden host==device
    bool               fuel_uploaded             = false;

    // --- flat-XS extension (RASBERY_GPU_FLATXS) ---------------------------
    double*            dev_ref = nullptr; // [9 mic | msm | 9 lmp | lsm]
    std::size_t        off_ref_mic[fxs::N_ACTIVE] = {};
    std::size_t        off_ref_msm = 0;
    std::size_t        off_ref_lmp[fxs::N_ACTIVE] = {};
    std::size_t        off_ref_lsm = 0;
    unsigned long long resident_ref_generation = 0;

    double*     dev_pernode = nullptr; // [wvfr | dmod | bppm], nx each
    int*        dev_nodes   = nullptr; // node list + off + cnt, grow-only
    int*        dev_off     = nullptr;
    int*        dev_cnt     = nullptr;
    std::size_t nodes_cap   = 0;
    int*        dev_sdid    = nullptr; // delta stream, grow-only
    double*     dev_sx      = nullptr;
    double*     dev_sscale  = nullptr;
    std::size_t stream_cap  = 0;

    // --- nodal extension (RASBERY_GPU_NODAL) ------------------------------
    double*            ndev_dbl   = nullptr; // hmesh|albedo|consts x9|chif|jnet|flux|phis|work
    int*               ndev_int   = nullptr; // lktosfc|neib|lklr|idirlr|sgnlr
    bool               nodal_geom_uploaded = false;
    unsigned long long resident_const_generation = 0;
    unsigned long long resident_chif_generation  = 0;
    int                nodal_nsurf = 0;
    std::size_t n_off_hmesh = 0, n_off_albedo = 0, n_off_consts = 0,
                n_off_chif = 0, n_off_jnet = 0, n_off_flux = 0, n_off_phis = 0,
                n_off_work = 0;
    std::size_t n_ioff_lktosfc = 0, n_ioff_neib = 0, n_ioff_lklr = 0,
                n_ioff_idirlr = 0, n_ioff_sgnlr = 0;

    const FlatXsLibDevice* lib = nullptr;      // shared, process lifetime
    std::uint64_t          lib_hash_cached = 0; // host tables are immutable;
    const void*            lib_hash_key    = nullptr; // hash once per source

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
        if (dev_ref) cudaFree(dev_ref);
        if (dev_pernode) cudaFree(dev_pernode);
        if (dev_nodes) cudaFree(dev_nodes);
        if (dev_off) cudaFree(dev_off);
        if (dev_cnt) cudaFree(dev_cnt);
        if (dev_sdid) cudaFree(dev_sdid);
        if (dev_sx) cudaFree(dev_sx);
        if (dev_sscale) cudaFree(dev_sscale);
        if (ndev_dbl) cudaFree(ndev_dbl);
        if (ndev_int) cudaFree(ndev_int);
        if (stream) cudaStreamDestroy(stream);
    }

    bool ensure(int want_nxyz, int want_fuel) {
        if (want_nxyz == nxyz && dev_block != nullptr && want_fuel <= n_fuel) {
            n_fuel = want_fuel;
            return true;
        }
        if (dev_block) { cudaFree(dev_block); dev_block = nullptr; }
        if (dev_fuel) { cudaFree(dev_fuel); dev_fuel = nullptr; }
        if (dev_ref) { cudaFree(dev_ref); dev_ref = nullptr; }
        if (dev_pernode) { cudaFree(dev_pernode); dev_pernode = nullptr; }
        resident_micx_generation  = 0;
        resident_state_generation = 0;
        resident_ref_generation   = 0;
        fuel_uploaded             = false;

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
    if (!rasberyGpuXsReconEnabled() && !rasberyGpuFlatXsEnabled() &&
        !rasberyGpuNodalEnabled()) {
        _impl->status = "disabled (RASBERY_GPU_XSRECON/RASBERY_GPU_FLATXS unset)";
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
                           unsigned long long state_generation, double* max_change_out) {
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
    // host arrays authoritative for every node after the download.  While the
    // host-state generation matches the resident copy, the host has not
    // written _xs/_iden since our last download, so both are already
    // bit-identical on the device and the ~4.4 MB re-upload is skipped.
    if (state_generation != d.resident_state_generation) {
        if (!d.upload(host.iden, d.off_iden, static_cast<std::size_t>(xsr::NISO) * nx)) return false;
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.xs[xt], d.off_xs[xt], lmp)) return false;
        if (!d.upload(host.xs_ssm, d.off_xs_ssm, ssm)) return false;
    }
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

    // depTrans rows: 39 doubles each, constant for the process, so they are
    // uploaded exactly once per instance.  With 100 Xe calls per case the two
    // per-call copies were pure API-call overhead (nsys: memcpy CALL COUNT,
    // not payload, dominates the timeline).
    if (d.dev_dep == nullptr) {
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_dep),
                                    2 * xsr::NISO * sizeof(double)), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_dep, host.dep_i135,
                                         xsr::NISO * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_dep + xsr::NISO, host.dep_xe135,
                                         xsr::NISO * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
    }
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

    // The downloads above just made host _xs and the Xe-chain _iden rows
    // equal to the device copy again; nothing else on the host wrote since
    // state_generation was read.
    d.resident_state_generation = state_generation;

    g_nodes_solved.fetch_add(scalars[1], std::memory_order_relaxed);
    return true;
}

bool XsReconBackend::solveFlatXs(const fxs::FlatXsView& host,
                                 const FlatXsLibShape& shape,
                                 unsigned long long micx_generation,
                                 unsigned long long micx_generation_next,
                                 unsigned long long ref_generation,
                                 unsigned long long state_generation,
                                 bool mark_micx_resident) {
    Impl& d = *_impl;
    if (!d.available || host.n_nodes <= 0 || host.nxyz <= 0) return false;
    // Reuse ensure() so the mic/lmp/xs/iden regions exist; keep the resident
    // fuel-count contract of the xsrecon solve intact by passing its current
    // value (or the node count on first contact).
    if (!d.ensure(host.nxyz, d.n_fuel > 0 ? d.n_fuel : host.n_nodes)) {
        d.available = false;
        return false;
    }

    const std::size_t nx  = static_cast<std::size_t>(d.nxyz);
    const std::size_t mic = static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx;
    const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
    const std::size_t msm = static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx;
    const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;

    // --- shared library tables (upload once per distinct content) ---------
    if (d.lib == nullptr || d.lib_hash_key != static_cast<const void*>(host.coeff_lsm)) {
        std::uint64_t h = 1469598103934665603ULL;
        h = fnvMix(&shape, sizeof shape, h);
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            h = fnvMix(host.coeff_lmp[t], shape.lmp_slot * sizeof(double), h);
        h = fnvMix(host.coeff_lsm, shape.lsm * sizeof(double), h);
        if (host.has_coeff_micx) {
            for (int t = 0; t < fxs::N_ACTIVE; ++t)
                h = fnvMix(host.coeff_mic[t], shape.mic_slot * sizeof(double), h);
            h = fnvMix(host.coeff_msm, shape.msm * sizeof(double), h);
        }
        h = fnvMix(host.knots, shape.n_knots * sizeof(double), h);
        h = fnvMix(host.deltas, shape.n_deltas * sizeof(fxs::DeltaMeta), h);
        d.lib_hash_cached = h;
        d.lib_hash_key    = host.coeff_lsm;

        std::lock_guard<std::mutex> lock(g_flatxs_lib_mutex);
        if (g_flatxs_libs == nullptr) g_flatxs_libs = new std::vector<FlatXsLibDevice>;
        const FlatXsLibDevice* found = nullptr;
        for (const auto& e : *g_flatxs_libs)
            if (e.hash == h) { found = &e; break; }
        if (found == nullptr) {
            FlatXsLibDevice e;
            e.hash = h;
            std::size_t off = 0;
            for (int t = 0; t < fxs::N_ACTIVE; ++t) { e.off_lmp[t] = off; off += shape.lmp_slot; }
            e.off_lsm = off; off += shape.lsm;
            for (int t = 0; t < fxs::N_ACTIVE; ++t) { e.off_mic[t] = off; off += shape.mic_slot; }
            e.off_msm = off; off += shape.msm;
            e.off_knots = off; off += shape.n_knots;
            RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&e.block),
                                        off * sizeof(double)), d.status);
            RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&e.deltas),
                                        shape.n_deltas * sizeof(fxs::DeltaMeta)), d.status);
            // Synchronous copies under the mutex: nothing can race a
            // half-uploaded table, and this happens once per process.
            for (int t = 0; t < fxs::N_ACTIVE; ++t)
                RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_lmp[t], host.coeff_lmp[t],
                                            shape.lmp_slot * sizeof(double),
                                            cudaMemcpyHostToDevice), d.status);
            RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_lsm, host.coeff_lsm,
                                        shape.lsm * sizeof(double),
                                        cudaMemcpyHostToDevice), d.status);
            if (host.has_coeff_micx) {
                for (int t = 0; t < fxs::N_ACTIVE; ++t)
                    RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_mic[t], host.coeff_mic[t],
                                                shape.mic_slot * sizeof(double),
                                                cudaMemcpyHostToDevice), d.status);
                RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_msm, host.coeff_msm,
                                            shape.msm * sizeof(double),
                                            cudaMemcpyHostToDevice), d.status);
            }
            if (shape.n_knots > 0)
                RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_knots, host.knots,
                                            shape.n_knots * sizeof(double),
                                            cudaMemcpyHostToDevice), d.status);
            RASBERY_CUDA_TRY(cudaMemcpy(e.deltas, host.deltas,
                                        shape.n_deltas * sizeof(fxs::DeltaMeta),
                                        cudaMemcpyHostToDevice), d.status);
            g_flatxs_libs->push_back(e);
            found = &g_flatxs_libs->back();
        }
        d.lib = found;
    }

    // --- per-instance reference block (re-upload on ref generation) -------
    if (d.dev_ref == nullptr) {
        std::size_t off = 0;
        for (int t = 0; t < fxs::N_ACTIVE; ++t) { d.off_ref_mic[t] = off; off += mic; }
        d.off_ref_msm = off; off += msm;
        for (int t = 0; t < fxs::N_ACTIVE; ++t) { d.off_ref_lmp[t] = off; off += lmp; }
        d.off_ref_lsm = off; off += ssm;
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_ref),
                                    off * sizeof(double)), d.status);
        d.resident_ref_generation = 0;
    }
    if (ref_generation != d.resident_ref_generation) {
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_ref + d.off_ref_mic[t], host.ref_mic[t],
                                             mic * sizeof(double),
                                             cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_ref + d.off_ref_msm, host.ref_msm,
                                         msm * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_ref + d.off_ref_lmp[t], host.ref_lmp[t],
                                             lmp * sizeof(double),
                                             cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_ref + d.off_ref_lsm, host.ref_lsm,
                                         ssm * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        d.resident_ref_generation = ref_generation;
    }

    // --- live micx/lmpx blocks: same residency contract as the xsrecon
    // solve.  On mismatch upload ALL 11 slots (the kernel writes only the
    // ACTIVE 9, but the xsrecon condense loop reads every slot later).
    if (micx_generation != d.resident_micx_generation) {
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.mic_all[xt], d.off_mic[xt], mic)) return false;
        if (!d.upload(host.msm, d.off_mic_ssm, msm)) return false;
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.lmp_all[xt], d.off_lmp[xt], lmp)) return false;
        if (!d.upload(host.lsm, d.off_lmp_ssm, ssm)) return false;
        d.resident_micx_generation = micx_generation;
    }

    // --- per-call state: xs and iden whole (target-only writes round-trip
    // every other column unchanged), skipped while the host-state generation
    // says the resident copies are still bit-identical.
    if (state_generation != d.resident_state_generation) {
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.xs[xt], d.off_xs[xt], lmp)) return false;
        if (!d.upload(host.xs_ssm, d.off_xs_ssm, ssm)) return false;
        if (!d.upload(host.iden, d.off_iden, static_cast<std::size_t>(xsr::NISO) * nx)) return false;
    }

    if (d.dev_pernode == nullptr)
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_pernode),
                                    3 * nx * sizeof(double)), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_pernode, host.wvfr, nx * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_pernode + nx, host.dmod, nx * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_pernode + 2 * nx, host.bppm, nx * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);

    const std::size_t n_nodes = static_cast<std::size_t>(host.n_nodes);
    if (n_nodes > d.nodes_cap) {
        if (d.dev_nodes) cudaFree(d.dev_nodes);
        if (d.dev_off) cudaFree(d.dev_off);
        if (d.dev_cnt) cudaFree(d.dev_cnt);
        d.dev_nodes = d.dev_off = d.dev_cnt = nullptr;
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_nodes),
                                    n_nodes * sizeof(int)), d.status);
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_off),
                                    n_nodes * sizeof(int)), d.status);
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_cnt),
                                    n_nodes * sizeof(int)), d.status);
        d.nodes_cap = n_nodes;
    }
    std::size_t stream_len = 0;
    if (host.n_nodes > 0)
        stream_len = static_cast<std::size_t>(host.node_off[host.n_nodes - 1]) +
                     static_cast<std::size_t>(host.node_cnt[host.n_nodes - 1]);
    if (stream_len > d.stream_cap) {
        if (d.dev_sdid) cudaFree(d.dev_sdid);
        if (d.dev_sx) cudaFree(d.dev_sx);
        if (d.dev_sscale) cudaFree(d.dev_sscale);
        d.dev_sdid = nullptr; d.dev_sx = d.dev_sscale = nullptr;
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_sdid),
                                    stream_len * sizeof(int)), d.status);
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_sx),
                                    stream_len * sizeof(double)), d.status);
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.dev_sscale),
                                    stream_len * sizeof(double)), d.status);
        d.stream_cap = stream_len;
    }
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_nodes, host.nodes, n_nodes * sizeof(int),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_off, host.node_off, n_nodes * sizeof(int),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_cnt, host.node_cnt, n_nodes * sizeof(int),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    if (stream_len > 0) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_sdid, host.stream_did,
                                         stream_len * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_sx, host.stream_x,
                                         stream_len * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_sscale, host.stream_scale,
                                         stream_len * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
    }

    // --- repoint the view at the device copies ----------------------------
    fxs::FlatXsView v = host;
    for (int t = 0; t < fxs::N_ACTIVE; ++t) {
        v.coeff_lmp[t] = d.lib->block + d.lib->off_lmp[t];
        v.coeff_mic[t] = d.lib->block + d.lib->off_mic[t];
        v.ref_lmp[t]   = d.dev_ref + d.off_ref_lmp[t];
        v.ref_mic[t]   = d.dev_ref + d.off_ref_mic[t];
        v.lmp[t]       = d.dev_block + d.off_lmp[ACTIVE_XT9[t]];
        v.mic[t]       = d.dev_block + d.off_mic[ACTIVE_XT9[t]];
    }
    v.coeff_lsm = d.lib->block + d.lib->off_lsm;
    v.coeff_msm = d.lib->block + d.lib->off_msm;
    v.knots     = d.lib->block + d.lib->off_knots;
    v.deltas    = d.lib->deltas;
    v.ref_lsm   = d.dev_ref + d.off_ref_lsm;
    v.ref_msm   = d.dev_ref + d.off_ref_msm;
    v.lsm       = d.dev_block + d.off_lmp_ssm;
    v.msm       = d.dev_block + d.off_mic_ssm;
    for (int xt = 0; xt < xsr::NXS; ++xt)
        v.xs[xt] = d.dev_block + d.off_xs[xt];
    v.xs_ssm       = d.dev_block + d.off_xs_ssm;
    v.iden         = d.dev_block + d.off_iden;
    v.wvfr         = d.dev_pernode;
    v.dmod         = d.dev_pernode + nx;
    v.bppm         = d.dev_pernode + 2 * nx;
    v.stream_did   = d.dev_sdid;
    v.stream_x     = d.dev_sx;
    v.stream_scale = d.dev_sscale;
    v.node_off     = d.dev_off;
    v.node_cnt     = d.dev_cnt;
    v.nodes        = d.dev_nodes;

    const int block = 128;
    const int grid  = (host.n_nodes + block - 1) / block;
    kernelFlatXs<<<grid, block, 0, d.stream>>>(v);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    // --- results the host needs back --------------------------------------
    // EXPERIMENT (RASBERY_FLATXS_SKIP_MICX_DL=1): leave micx/lmpx device-only
    // -- 57 of the 61 MB/call downloads.  UNSAFE whenever the host reads
    // _micx/_lmpx afterwards (depletion, rodded NodeSpectralIndex fallback,
    // full micx export, CPU fallback arms); the full-deck A/B is the gate for
    // any deck class this is enabled on.  Default off.
    static const bool skip_micx_dl = envFlagEnabled("RASBERY_FLATXS_SKIP_MICX_DL");
    if (!skip_micx_dl) {
        for (int t = 0; t < fxs::N_ACTIVE; ++t) {
            if (!d.download(host.lmp[t], d.off_lmp[ACTIVE_XT9[t]], lmp)) return false;
            if (!d.download(host.mic[t], d.off_mic[ACTIVE_XT9[t]], mic)) return false;
        }
        if (!d.download(host.lsm, d.off_lmp_ssm, ssm)) return false;
        if (!d.download(host.msm, d.off_mic_ssm, msm)) return false;
    }
    for (int xt = 0; xt < xsr::NXS; ++xt)
        if (!d.download(host.xs[xt], d.off_xs[xt], lmp)) return false;
    if (!d.download(host.xs_ssm, d.off_xs_ssm, ssm)) return false;
    // Light-isotope rows H-1/B-10/O-16 are 0..2 -- contiguous by registry design.
    if (!d.download(host.iden, d.off_iden, 3 * nx)) return false;

    RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);

    // After the download the host and device copies are bit-identical, so the
    // caller's post-call generation bump can be marked already-resident and
    // the next xsrecon call skips its ~70 MB re-upload.  A rodded CPU pass
    // after this call invalidates that (mark_micx_resident=false).
    d.resident_micx_generation  = mark_micx_resident ? micx_generation_next : 0;
    // A rodded CPU pass right after this call rewrites _xs/_iden columns on
    // the host, so the caller only lets us keep the state residency when the
    // whole call was device-side.
    d.resident_state_generation = mark_micx_resident ? state_generation : 0;

    g_flatxs_nodes_solved.fetch_add(static_cast<unsigned long long>(host.n_nodes),
                                    std::memory_order_relaxed);
    return true;
}

bool XsReconBackend::solveNodal(const ndl::NodalView& host,
                                unsigned long long const_generation,
                                unsigned long long ref_generation,
                                unsigned long long state_generation) {
    // calculateEven stays on the host until its 1-ULP residual class is
    // mined out (RASBERY_GPU_NODAL_FULL=1 forces the all-device path).
    static const bool hybrid_even = !envFlagEnabled("RASBERY_GPU_NODAL_FULL");
    Impl& d = *_impl;
    if (!d.available || host.nxyz <= 0 || host.nsurf <= 0) return false;
    if (!d.ensure(host.nxyz, d.n_fuel > 0 ? d.n_fuel : host.nxyz)) {
        d.available = false;
        return false;
    }
    const std::size_t nx = static_cast<std::size_t>(host.nxyz);
    const std::size_t ns = static_cast<std::size_t>(host.nsurf);

    if (d.ndev_dbl == nullptr || d.nodal_nsurf != host.nsurf) {
        if (d.ndev_dbl) { cudaFree(d.ndev_dbl); d.ndev_dbl = nullptr; }
        if (d.ndev_int) { cudaFree(d.ndev_int); d.ndev_int = nullptr; }
        d.nodal_geom_uploaded       = false;
        d.resident_const_generation = 0;
        d.resident_chif_generation  = 0;
        d.nodal_nsurf               = host.nsurf;
        const std::size_t ndg0 = nx * ndl::NDIR * ndl::NG;
        std::size_t off = 0;
        d.n_off_hmesh = off; off += nx * ndl::NDIR;
        d.n_off_albedo = off; off += ndl::NDIR * ndl::NLR;
        d.n_off_consts = off; off += 9 * ndg0;
        d.n_off_chif = off; off += ndl::NG * nx;
        d.n_off_jnet = off; off += ns * ndl::NG;
        d.n_off_flux = off; off += nx * ndl::NG;
        d.n_off_phis = off; off += ns * ndl::NG;
        d.n_off_work = off;
        off += 3 * ndg0 + 2 * nx * ndl::NDIR * ndl::NG2 + 4 * nx * ndl::NG2 +
               3 * ndg0;
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.ndev_dbl),
                                    off * sizeof(double)), d.status);
        std::size_t ioff = 0;
        d.n_ioff_lktosfc = ioff; ioff += nx * ndl::NDIR * ndl::NLR;
        d.n_ioff_neib = ioff; ioff += nx * ndl::NEWSB;
        d.n_ioff_lklr = ioff; ioff += ns * ndl::NLR;
        d.n_ioff_idirlr = ioff; ioff += ns * ndl::NLR;
        d.n_ioff_sgnlr = ioff; ioff += ns * ndl::NLR;
        RASBERY_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d.ndev_int),
                                    ioff * sizeof(int)), d.status);
    }
    const std::size_t ndg = nx * ndl::NDIR * ndl::NG;

    if (!d.nodal_geom_uploaded) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_hmesh, host.hmesh,
                                         nx * ndl::NDIR * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_albedo, host.albedo,
                                         ndl::NDIR * ndl::NLR * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_lktosfc, host.lktosfc,
                                         nx * ndl::NDIR * ndl::NLR * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_neib, host.neib,
                                         nx * ndl::NEWSB * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_lklr, host.lklr,
                                         ns * ndl::NLR * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_idirlr, host.idirlr,
                                         ns * ndl::NLR * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_sgnlr, host.sgnlr,
                                         ns * ndl::NLR * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        d.nodal_geom_uploaded = true;
    }

    if (const_generation != d.resident_const_generation) {
        const double* consts[9] = {host.eta1, host.eta2, host.m260,
                                   host.m251, host.m253, host.m262,
                                   host.m264, host.diagD, host.diagDI};
        for (int i = 0; i < 9; ++i)
            RASBERY_CUDA_TRY(
                cudaMemcpyAsync(d.ndev_dbl + d.n_off_consts + i * ndg, consts[i],
                                ndg * sizeof(double), cudaMemcpyHostToDevice,
                                d.stream), d.status);
        d.resident_const_generation = const_generation;
    }

    if (!host.chif_empty && ref_generation != d.resident_chif_generation) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_chif, host.chif,
                                         ndl::NG * nx * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        d.resident_chif_generation = ref_generation;
    }

    // xs inputs: resident when the host-state generation matches; upload the
    // three rows for this call otherwise (residency untouched -- iden may be
    // stale, and the xs arms own that contract).
    const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
    const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;
    if (state_generation != d.resident_state_generation) {
        if (!d.upload(host.xsrf, d.off_xs[xsr::T_XSRF], lmp)) return false;
        if (!d.upload(host.xsnf, d.off_xs[xsr::T_XSNF], lmp)) return false;
        if (!d.upload(host.xssm, d.off_xs_ssm, ssm)) return false;
    }

    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_jnet, host.jnet,
                                     ns * ndl::NG * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_flux, host.flux,
                                     nx * ndl::NG * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);

    ndl::NodalView v = host;
    v.hmesh   = d.ndev_dbl + d.n_off_hmesh;
    v.albedo  = d.ndev_dbl + d.n_off_albedo;
    v.lktosfc = d.ndev_int + d.n_ioff_lktosfc;
    v.neib    = d.ndev_int + d.n_ioff_neib;
    v.lklr    = d.ndev_int + d.n_ioff_lklr;
    v.idirlr  = d.ndev_int + d.n_ioff_idirlr;
    v.sgnlr   = d.ndev_int + d.n_ioff_sgnlr;
    v.eta1   = d.ndev_dbl + d.n_off_consts + 0 * ndg;
    v.eta2   = d.ndev_dbl + d.n_off_consts + 1 * ndg;
    v.m260   = d.ndev_dbl + d.n_off_consts + 2 * ndg;
    v.m251   = d.ndev_dbl + d.n_off_consts + 3 * ndg;
    v.m253   = d.ndev_dbl + d.n_off_consts + 4 * ndg;
    v.m262   = d.ndev_dbl + d.n_off_consts + 5 * ndg;
    v.m264   = d.ndev_dbl + d.n_off_consts + 6 * ndg;
    v.diagD  = d.ndev_dbl + d.n_off_consts + 7 * ndg;
    v.diagDI = d.ndev_dbl + d.n_off_consts + 8 * ndg;
    v.chif   = d.ndev_dbl + d.n_off_chif;
    v.xsrf = d.dev_block + d.off_xs[xsr::T_XSRF];
    v.xsnf = d.dev_block + d.off_xs[xsr::T_XSNF];
    v.xssm = d.dev_block + d.off_xs_ssm;
    v.jnet = d.ndev_dbl + d.n_off_jnet;
    v.flux = d.ndev_dbl + d.n_off_flux;
    v.phis = d.ndev_dbl + d.n_off_phis;
    double* wk = d.ndev_dbl + d.n_off_work;
    v.trlcff0 = wk; wk += ndg;
    v.trlcff1 = wk; wk += ndg;
    v.trlcff2 = wk; wk += ndg;
    v.mu = wk; wk += nx * ndl::NDIR * ndl::NG2;
    v.tau = wk; wk += nx * ndl::NDIR * ndl::NG2;
    v.matM = wk; wk += nx * ndl::NG2;
    v.matMI = wk; wk += nx * ndl::NG2;
    v.matMs = wk; wk += nx * ndl::NG2;
    v.matMf = wk; wk += nx * ndl::NG2;
    v.dsncff2 = wk; wk += ndg;
    v.dsncff4 = wk; wk += ndg;
    v.dsncff6 = wk; wk += ndg;

    const int B  = 128;
    const int gn = (host.nxyz + B - 1) / B;
    const int gs = (host.nsurf + B - 1) / B;
    kNodalTrl0<<<gn, B, 0, d.stream>>>(v);
    kNodalTrl12<<<gn, B, 0, d.stream>>>(v);
    kNodalMat<<<gn, B, 0, d.stream>>>(v);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    if (hybrid_even) {
        // calculateEven runs on the HOST (the production member function is
        // its own bit-exact reference; its mined mask still has a 1-ULP
        // residual class).  Ship the inputs it needs, let the caller run it,
        // and finish in solveNodalPost.
        const std::size_t ndg2 = nx * ndl::NDIR * ndl::NG;
        double* wk0 = d.ndev_dbl + d.n_off_work;
        RASBERY_CUDA_TRY(cudaMemcpyAsync(host.trlcff0, wk0,
                                         ndg2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(host.trlcff2, wk0 + 2 * ndg2,
                                         ndg2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, d.stream), d.status);
        double* wmat = wk0 + 3 * ndg2 + 2 * nx * ndl::NDIR * ndl::NG2;
        RASBERY_CUDA_TRY(cudaMemcpyAsync(host.matM, wmat,
                                         nx * ndl::NG2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);
        return true; // caller: run host even, then solveNodalPost
    }

    kNodalEven<<<gn, B, 0, d.stream>>>(v);
    kNodalJnet<<<gs, B, 0, d.stream>>>(v);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    RASBERY_CUDA_TRY(cudaMemcpyAsync(host.jnet, v.jnet,
                                     ns * ndl::NG * sizeof(double),
                                     cudaMemcpyDeviceToHost, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(host.phis, v.phis,
                                     ns * ndl::NG * sizeof(double),
                                     cudaMemcpyDeviceToHost, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);

    g_nodal_drives.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool XsReconBackend::solveNodalPost(const ndl::NodalView& host) {
    Impl& d = *_impl;
    if (!d.available || d.ndev_dbl == nullptr) return false;
    const std::size_t nx   = static_cast<std::size_t>(host.nxyz);
    const std::size_t ns   = static_cast<std::size_t>(host.nsurf);
    const std::size_t ndg2 = nx * ndl::NDIR * ndl::NG;

    // Upload the host-computed dsncff blocks into the device work region.
    double* wk0 = d.ndev_dbl + d.n_off_work;
    double* wds = wk0 + 3 * ndg2 + 2 * nx * ndl::NDIR * ndl::NG2 +
                  4 * nx * ndl::NG2;
    RASBERY_CUDA_TRY(cudaMemcpyAsync(wds, host.dsncff2, ndg2 * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(wds + ndg2, host.dsncff4,
                                     ndg2 * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(wds + 2 * ndg2, host.dsncff6,
                                     ndg2 * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);

    ndl::NodalView v = host;
    v.lklr    = d.ndev_int + d.n_ioff_lklr;
    v.idirlr  = d.ndev_int + d.n_ioff_idirlr;
    v.sgnlr   = d.ndev_int + d.n_ioff_sgnlr;
    v.lktosfc = d.ndev_int + d.n_ioff_lktosfc;
    v.neib    = d.ndev_int + d.n_ioff_neib;
    v.hmesh   = d.ndev_dbl + d.n_off_hmesh;
    v.albedo  = d.ndev_dbl + d.n_off_albedo;
    v.eta1   = d.ndev_dbl + d.n_off_consts + 0 * ndg2;
    v.eta2   = d.ndev_dbl + d.n_off_consts + 1 * ndg2;
    v.m260   = d.ndev_dbl + d.n_off_consts + 2 * ndg2;
    v.m251   = d.ndev_dbl + d.n_off_consts + 3 * ndg2;
    v.m253   = d.ndev_dbl + d.n_off_consts + 4 * ndg2;
    v.m262   = d.ndev_dbl + d.n_off_consts + 5 * ndg2;
    v.m264   = d.ndev_dbl + d.n_off_consts + 6 * ndg2;
    v.diagD  = d.ndev_dbl + d.n_off_consts + 7 * ndg2;
    v.diagDI = d.ndev_dbl + d.n_off_consts + 8 * ndg2;
    v.jnet = d.ndev_dbl + d.n_off_jnet;
    v.flux = d.ndev_dbl + d.n_off_flux;
    v.phis = d.ndev_dbl + d.n_off_phis;
    double* wk = wk0;
    v.trlcff0 = wk; wk += ndg2;
    v.trlcff1 = wk; wk += ndg2;
    v.trlcff2 = wk; wk += ndg2;
    v.mu = wk; wk += nx * ndl::NDIR * ndl::NG2;
    v.tau = wk; wk += nx * ndl::NDIR * ndl::NG2;
    v.matM = wk; wk += nx * ndl::NG2;
    v.matMI = wk; wk += nx * ndl::NG2;
    v.matMs = wk; wk += nx * ndl::NG2;
    v.matMf = wk; wk += nx * ndl::NG2;
    v.dsncff2 = wk; wk += ndg2;
    v.dsncff4 = wk; wk += ndg2;
    v.dsncff6 = wk; wk += ndg2;

    const int B  = 128;
    const int gs = (host.nsurf + B - 1) / B;
    kNodalJnet<<<gs, B, 0, d.stream>>>(v);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    RASBERY_CUDA_TRY(cudaMemcpyAsync(host.jnet, v.jnet,
                                     ns * ndl::NG * sizeof(double),
                                     cudaMemcpyDeviceToHost, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(host.phis, v.phis,
                                     ns * ndl::NG * sizeof(double),
                                     cudaMemcpyDeviceToHost, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);

    g_nodal_drives.fetch_add(1, std::memory_order_relaxed);
    return true;
}

unsigned long long XsReconBackend::nodalDrivesSolved() {
    return g_nodal_drives.load(std::memory_order_relaxed);
}

unsigned long long XsReconBackend::nodesSolved() {
    return g_nodes_solved.load(std::memory_order_relaxed);
}

unsigned long long XsReconBackend::flatXsNodesSolved() {
    return g_flatxs_nodes_solved.load(std::memory_order_relaxed);
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

bool rasberyGpuFlatXsEnabled() {
    static const bool on = envFlagEnabled("RASBERY_GPU_FLATXS");
    return on;
}

bool rasberyGpuNodalEnabled() {
    static const bool on = envFlagEnabled("RASBERY_GPU_NODAL");
    return on;
}

unsigned long long rasberyGpuNodalDrives() {
    return g_nodal_drives.load(std::memory_order_relaxed);
}

unsigned long long rasberyGpuXsReconNodes() {
    return g_nodes_solved.load(std::memory_order_relaxed);
}

unsigned long long rasberyGpuFlatXsNodes() {
    return g_flatxs_nodes_solved.load(std::memory_order_relaxed);
}

} // namespace rasbery
