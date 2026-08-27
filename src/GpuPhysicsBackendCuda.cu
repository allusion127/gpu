// CUDA arm of GpuPhysicsBackend -- Rev.7.1 Sec 1.5.
//
// THE HOLE THIS CLOSES.  GpuPhysicsBackendStub.cpp is REMOVED from the source
// list when RASBERY_ENABLE_CUDA=ON, and nothing replaced it: a CUDA build had a
// declared GpuPhysicsBackend, a declared rasberyGpuPhysicsEnabled(), and no
// definition of either.  It linked only because nothing called them yet.  The
// first call site would have been an undefined-symbol error at the end of a
// twenty-minute build.
//
// WHAT THE PROBE PROMISES.  Only what it measured.  The tier ladder in
// GpuPhysicsTypes.h has five rungs; this fills in the two Rev.7.1 Sec 1.5 says
// this campaign will actually produce, and reports Unsupported for everything
// else rather than guessing:
//
//   G3 ConditionalGraph  device-side phase selection (Sec 5.5) -- needs CUDA 12.4+
//                        for cudaGraphConditionalHandle and compute capability
//                        >= 7.5, both checked here, not assumed.
//   G2 EpochScheduler    graphs plus a host-driven epoch loop (Sec 5.6).
//   Unsupported          no device, or a device this build cannot drive.
//
// Memory-pool support is asked of the driver (cudaDevAttrMemoryPoolsSupported)
// rather than inferred from the CUDA version, because the arena's single
// allocation goes through cudaMallocFromPoolAsync and a wrong answer here is a
// failed reserve() much later.
//
// NO COOPERATIVE CAPABILITY FIELD.  W0 measured c_barrier = 0.78 us against the
// 0.384 us kill threshold, so the persistent track is closed (constraint 17).
// GpuCapabilityReceipt deliberately has no cooperative-launch flag; adding one
// here would invite code that reads it.

#include "GpuPhysicsTypes.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace rasbery::gpu {

struct GpuPhysicsBackend::Impl {
    GpuCapabilityReceipt receipt;
    std::string          status;
};

GpuPhysicsBackend::GpuPhysicsBackend()
    : _impl(new Impl()) {
    GpuCapabilityReceipt& r = _impl->receipt;
    r.kind         = GpuBackendKind::Cuda;
    r.tier         = GpuSupportTier::Unsupported;
    r.backend_name = backendName(GpuBackendKind::Cuda);

    int               count = 0;
    const cudaError_t ce    = cudaGetDeviceCount(&count);
    if (ce != cudaSuccess) {
        _impl->status = std::string("cudaGetDeviceCount -> ") + cudaGetErrorString(ce);
        return;
    }
    if (count == 0) {
        _impl->status = "no CUDA device";
        return;
    }

    int               device = 0;
    const cudaError_t de     = cudaGetDevice(&device);
    if (de != cudaSuccess) {
        _impl->status = std::string("cudaGetDevice -> ") + cudaGetErrorString(de);
        return;
    }

    cudaDeviceProp prop{};
    const cudaError_t pe = cudaGetDeviceProperties(&prop, device);
    if (pe != cudaSuccess) {
        _impl->status = std::string("cudaGetDeviceProperties -> ") + cudaGetErrorString(pe);
        return;
    }

    r.device_name           = prop.name;
    r.compute_major         = prop.major;
    r.compute_minor         = prop.minor;
    r.multiprocessors       = prop.multiProcessorCount;
    r.total_global_bytes    = static_cast<long long>(prop.totalGlobalMem);
    r.l2_cache_bytes        = static_cast<long long>(prop.l2CacheSize);
    r.shared_bytes_per_block = static_cast<int>(prop.sharedMemPerBlock);

    // Graphs have been available since CUDA 10 on every architecture this build
    // targets; the two that vary are asked for explicitly.
    r.has_graphs = true;

    int pools = 0;
    if (cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device) == cudaSuccess) {
        r.has_memory_pools         = pools != 0;
        r.has_stream_ordered_alloc = pools != 0;
    }

#if CUDART_VERSION >= 12040
    // Conditional graph nodes: the runtime has to know them AND the device has
    // to be able to run them.  Both, not either.
    r.has_conditional_graphs = (prop.major > 7) || (prop.major == 7 && prop.minor >= 5);
#else
    r.has_conditional_graphs = false;
#endif

    r.tier = r.has_conditional_graphs ? GpuSupportTier::G3 : GpuSupportTier::G2;
    _impl->status = "ok";
}

GpuPhysicsBackend::~GpuPhysicsBackend() {
    delete _impl;
    _impl = nullptr;
}

bool GpuPhysicsBackend::available() const {
    return _impl->receipt.tier != GpuSupportTier::Unsupported;
}

const std::string& GpuPhysicsBackend::status() const { return _impl->status; }

const GpuCapabilityReceipt& GpuPhysicsBackend::capability() const { return _impl->receipt; }

GpuBackendKind GpuPhysicsBackend::compiledKind() { return GpuBackendKind::Cuda; }

const char* GpuPhysicsBackend::backendName(GpuBackendKind kind) {
    switch (kind) {
        case GpuBackendKind::Cuda: return "cuda";
        case GpuBackendKind::Hip:  return "hip";
        case GpuBackendKind::Sycl: return "sycl";
        case GpuBackendKind::None: break;
    }
    return "none";
}

std::string GpuPhysicsBackend::receiptJson() const {
    const GpuCapabilityReceipt& r = _impl->receipt;
    std::string                 out;
    out += "{\"backend\":\"";
    out += r.backend_name;
    out += "\",\"tier\":\"";
    out += gpuSupportTierName(r.tier);
    out += "\",\"tier_ordinal\":";
    out += std::to_string(static_cast<unsigned>(r.tier));
    out += ",\"available\":";
    out += available() ? "true" : "false";
    out += ",\"device\":\"";
    out += r.device_name;
    out += "\",\"compute\":\"";
    out += std::to_string(r.compute_major);
    out += ".";
    out += std::to_string(r.compute_minor);
    out += "\",\"sms\":";
    out += std::to_string(r.multiprocessors);
    out += ",\"total_global_bytes\":";
    out += std::to_string(r.total_global_bytes);
    out += ",\"l2_cache_bytes\":";
    out += std::to_string(r.l2_cache_bytes);
    out += ",\"graphs\":";
    out += r.has_graphs ? "true" : "false";
    out += ",\"conditional_graphs\":";
    out += r.has_conditional_graphs ? "true" : "false";
    out += ",\"memory_pools\":";
    out += r.has_memory_pools ? "true" : "false";
    out += ",\"status\":\"";
    out += _impl->status;
    out += "\"}";
    return out;
}

/// Read once, like rasberyGpuNodalFullEnabled(): one process-wide value, no new
/// exported symbol.  Unlike the stub arm this gate CAN open, so it reads the
/// environment.
bool rasberyGpuPhysicsEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("RASBERY_GPU_PHYSICS");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}

} // namespace rasbery::gpu
