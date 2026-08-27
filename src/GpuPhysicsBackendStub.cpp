// CPU-only builds: the same symbols the CUDA arm defines, with no CUDA
// anywhere, so call sites never need an #ifdef.  Mirrors
// CudaXsReconBackendStub.cpp and CudaBICGBackendStub.cpp.
//
// STUB PARITY is the property this file exists to hold: every symbol declared
// in GpuPhysicsTypes.h and GpuPhysicsArena.h has a definition here, and every
// one of them answers "no device" rather than failing to link.  A build with
// RASBERY_ENABLE_CUDA=OFF therefore exercises the same call graph, the same
// lifetimes and the same receipts as a device build -- on a machine with no GPU.
//
// The arena stub still COMPUTES the layout.  That is deliberate: the offsets,
// the admission arithmetic and the receipt are pure host code
// (GpuPhysicsArenaLayout.h), so a CPU-only build can answer "how many bytes
// would 64 slots need" exactly, which is what the layout contract test asks it.
// Only the allocation and the transfers are absent.

#include "GpuPhaseScheduler.h"
#include "GpuPhysicsArena.h"
#include "GpuPhysicsTypes.h"

#include <ostream>
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
    out += "\",\"tier\":\"";
    out += gpuSupportTierName(r.tier);
    out += "\",\"tier_ordinal\":";
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

// ---------------------------------------------------------------------------
// GpuPhysicsArena, no-CUDA arm.
//
// reserve() computes the layout and runs admission -- both are pure host
// arithmetic and both are worth having in a CPU-only build -- and then refuses,
// because there is nothing to allocate from.  So `offsets()` and `receiptJson()`
// answer truthfully on a machine with no GPU while `available()` stays false and
// every pointer is null.
// ---------------------------------------------------------------------------

struct GpuPhysicsArena::Impl {
    ArenaOffsets   offsets{};
    ArenaAdmission admission{};
    std::string    status = "built without CUDA (stub)";
};

GpuPhysicsArena::GpuPhysicsArena()
    : _impl(new Impl()) {}

GpuPhysicsArena::~GpuPhysicsArena() {
    delete _impl;
    _impl = nullptr;
}

bool GpuPhysicsArena::reserve(const ArenaDims& dims) {
    _impl->offsets = arenaComputeLayout(dims);
    // No device: there is no free/total to admit against, so the arithmetic runs
    // against the layout alone and the answer is recorded, not acted on.
    _impl->admission = arenaAdmit(_impl->offsets, 0, 0);
    _impl->status    = "built without CUDA (stub)";
    return false;
}

void GpuPhysicsArena::release() {}

bool GpuPhysicsArena::available() const { return false; }

const std::string& GpuPhysicsArena::status() const { return _impl->status; }

const ArenaOffsets& GpuPhysicsArena::offsets() const { return _impl->offsets; }

const ArenaDims& GpuPhysicsArena::dims() const { return _impl->offsets.dims; }

const ArenaAdmission& GpuPhysicsArena::admission() const { return _impl->admission; }

void* GpuPhysicsArena::base() const { return nullptr; }

void* GpuPhysicsArena::geometryRegion(GeometryRegion) const { return nullptr; }

void* GpuPhysicsArena::libraryRegion(LibraryRegion) const { return nullptr; }

void* GpuPhysicsArena::slotRegion(int, SlotRegion) const { return nullptr; }

void* GpuPhysicsArena::scratch(int, DevicePhase, ScratchId) const { return nullptr; }

DeviceGeometryView GpuPhysicsArena::geometryView() const { return DeviceGeometryView{}; }

DeviceXsLibraryView GpuPhysicsArena::libraryView() const { return DeviceXsLibraryView{}; }

DeviceSlotView GpuPhysicsArena::slotView(int) const { return DeviceSlotView{}; }

bool GpuPhysicsArena::importGeometryAsync(GeometryRegion, const void*, std::size_t,
                                          GpuStreamHandle) {
    return false;
}

bool GpuPhysicsArena::importLibraryAsync(LibraryRegion, const void*, std::size_t,
                                         GpuStreamHandle) {
    return false;
}

bool GpuPhysicsArena::importSlotAsync(int, SlotRegion, const void*, std::size_t,
                                      GpuStreamHandle) {
    return false;
}

bool GpuPhysicsArena::exportSnapshotAsync(int, SlotRegion, void*, std::size_t,
                                          GpuStreamHandle) {
    return false;
}

bool GpuPhysicsArena::clearSlotAsync(int, GpuStreamHandle) { return false; }

std::string GpuPhysicsArena::receiptJson() const {
    const ArenaOffsets& o = _impl->offsets;
    std::string         out;
    out += "{\"backend\":\"none\",\"available\":false,\"shared_geometry_bytes\":";
    out += std::to_string(o.shared_geometry_bytes);
    out += ",\"shared_library_bytes\":";
    out += std::to_string(o.shared_library_bytes);
    out += ",\"control_block_bytes\":";
    out += std::to_string(o.control_block_bytes);
    out += ",\"per_slot_bytes\":";
    out += std::to_string(o.per_slot_bytes);
    out += ",\"slot_count\":";
    out += std::to_string(o.slot_count);
    out += ",\"scratch_bytes\":";
    out += std::to_string(o.per_slot_scratch_bytes * o.slot_count);
    out += ",\"per_slot_scratch_bytes\":";
    out += std::to_string(o.per_slot_scratch_bytes);
    out += ",\"total_bytes\":";
    out += std::to_string(o.total_bytes);
    out += ",\"alignment\":";
    out += std::to_string(kArenaAlignment);
    out += ",\"slot_base\":";
    out += std::to_string(o.slot_base);
    out += ",\"vram_free_bytes\":0,\"vram_total_bytes\":0,\"driver_reserve_bytes\":";
    out += std::to_string(_impl->admission.driver_reserve_bytes);
    out += ",\"fragmentation_reserve_bytes\":";
    out += std::to_string(_impl->admission.fragmentation_reserve_bytes);
    out += ",\"per_slot_ceiling_bytes\":";
    out += std::to_string(kArenaPerSlotByteCeiling);
    out += ",\"admitted\":false,\"pool\":\"none\",\"status\":\"";
    out += _impl->status;
    out += "\"}";
    return out;
}

void GpuPhysicsArena::emitReceipt(std::ostream& os) const {
    os << "[RASBERY][GPU_ARENA] " << receiptJson() << "\n";
}

// ---------------------------------------------------------------------------
// GpuPhaseScheduler, no-CUDA arm.
//
// The launchers refuse; the CLASSIFICATION SEMANTICS do not live here at all.
// gpuClassifySerial() in GpuPhaseScheduler.h is the definition, it is pure, and
// it runs on the host in any build -- which is what lets the queue ordering,
// the padding, the bucket choice and both Sec 5.2 fatal faults be checked with
// no device (test/gpu_phase_compaction.cpp).
// ---------------------------------------------------------------------------

bool gpuLaunchClassify(DeviceSlotPhase*, int, DevicePhaseQueues*, GpuSchedulerStream) {
    return false;
}

bool gpuLaunchRefill(const GpuRefillArgs&, GpuSchedulerStream) { return false; }

bool gpuLaunchRefillThenClassify(const GpuRefillArgs&, DevicePhaseQueues*,
                                 GpuSchedulerStream) {
    return false;
}

} // namespace rasbery::gpu
