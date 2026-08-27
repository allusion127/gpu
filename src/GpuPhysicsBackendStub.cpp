// CPU-only builds: the same symbols the CUDA arm defines, with no CUDA
// anywhere, so call sites never need an #ifdef.  Mirrors
// CudaXsReconBackendStub.cpp and CudaBICGBackendStub.cpp.
//
// STUB PARITY is the property this file exists to hold: every symbol declared
// in GpuPhysicsTypes.h has a definition here, and every one of them answers
// "no device" rather than failing to link.  A build with RASBERY_ENABLE_CUDA=OFF
// therefore exercises the same call graph, the same lifetimes and the same
// receipts as a device build -- on a machine with no GPU.

#include "GpuPhysicsTypes.h"

#include <string>

namespace rasbery::gpu {

struct GpuPhysicsBackend::Impl {
    GpuCapabilityReceipt receipt;
    std::string          status;
};

GpuPhysicsBackend::GpuPhysicsBackend()
    : _impl(new Impl()) {
    _impl->receipt.kind         = GpuBackendKind::None;
    _impl->receipt.tier         = GpuSupportTier::Unsupported;
    _impl->receipt.backend_name = backendName(GpuBackendKind::None);
    _impl->receipt.device_name  = "";
    _impl->status               = "built without CUDA (stub)";
}

GpuPhysicsBackend::~GpuPhysicsBackend() {
    delete _impl;
    _impl = nullptr;
}

bool GpuPhysicsBackend::available() const { return false; }

const std::string& GpuPhysicsBackend::status() const { return _impl->status; }

const GpuCapabilityReceipt& GpuPhysicsBackend::capability() const { return _impl->receipt; }

GpuBackendKind GpuPhysicsBackend::compiledKind() { return GpuBackendKind::None; }

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
    std::string out;
    out += "{\"backend\":\"";
    out += r.backend_name;
    out += "\",\"tier\":";
    out += std::to_string(static_cast<unsigned>(r.tier));
    out += ",\"available\":false,\"device\":\"";
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
    out += ",\"graphs\":false,\"conditional_graphs\":false,\"memory_pools\":false";
    out += ",\"status\":\"";
    out += _impl->status;
    out += "\"}";
    return out;
}

/// Same spelling as rasberyGpuXsReconEnabled() in the xsrecon stub: with no
/// backend compiled in the gate cannot open, whatever RASBERY_GPU_PHYSICS says.
/// The env variable is read only by the CUDA arm, where opening it is possible.
bool rasberyGpuPhysicsEnabled() { return false; }

} // namespace rasbery::gpu
