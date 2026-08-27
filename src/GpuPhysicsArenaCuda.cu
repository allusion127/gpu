// CUDA arm of GpuPhysicsArena -- Rev.7.1 plan Task 2, Sec 4.
//
// ONE ALLOCATION, TAKEN ONCE, NEVER MOVED.  reserve() is the only place in this
// translation unit that allocates device memory; everything else is an offset
// add or a byte copy.  That is not a style preference -- a captured CUDA graph
// bakes its kernel arguments, so a pointer that can move invalidates every
// graph that ever saw it, and re-capturing is exactly the cost the graph was
// bought to avoid.  The contract test greps this file to keep it true.
//
// WHY A MEMORY POOL FOR A SINGLE BLOCK.  cudaMallocFromPoolAsync is
// stream-ordered, so the allocation is sequenced against the stream the rest of
// the setup runs on instead of being a device-wide synchronisation point.  The
// pool's release threshold is pinned at the block size so the driver does not
// hand the pages back between statepoints and then have to fault them in again
// -- which is the behaviour the default (threshold 0) pool has.  There is no
// suballocation: the block is carved by GpuPhysicsArenaLayout.h, on the host,
// before any of this runs.
//
// SYNCHRONISE BEFORE PUBLISHING.  The base pointer is not readable by anything
// until the stream that allocated it has caught up, so reserve() synchronises
// once.  Every later transfer is async again.  "Pointers fixed before any graph
// work" means exactly this: by the time anyone can capture a graph, the address
// is final.
//
// ADMISSION FAILS LOUD (Sec 4.4).  A refusal writes a [RASBERY][GPU_ARENA][FAIL]
// line and leaves the arena unavailable.  It does NOT retry with fewer slots.

#include "GpuPhysicsArena.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <string>

namespace rasbery::gpu {

namespace {

std::string cudaWhy(const char* what, cudaError_t e) {
    return std::string(what) + " -> " + cudaGetErrorString(e);
}

} // namespace

struct GpuPhysicsArena::Impl {
    unsigned char* base   = nullptr;
    cudaMemPool_t  pool   = nullptr;
    bool           owns_pool = false;
    cudaStream_t   setup  = nullptr;
    int            device = -1;

    ArenaOffsets   offsets{};
    ArenaAdmission admission{};
    std::string    status = "not reserved";
    bool           ready  = false;

    long long free_at_reserve  = 0;
    long long total_at_reserve = 0;

    bool fail(std::string why) {
        status = std::move(why);
        std::fprintf(stderr, "[RASBERY][GPU_ARENA][FAIL] {\"reason\":\"%s\"}\n", status.c_str());
        return false;
    }
};

GpuPhysicsArena::GpuPhysicsArena()
    : _impl(new Impl()) {}

GpuPhysicsArena::~GpuPhysicsArena() {
    release();
    delete _impl;
    _impl = nullptr;
}

bool GpuPhysicsArena::reserve(const ArenaDims& dims) {
    Impl& d = *_impl;
    if (d.ready) return d.fail("reserve() called twice; the arena is allocated once");

    d.offsets = arenaComputeLayout(dims);
    if (d.offsets.slot_count == 0 || d.offsets.total_bytes == 0)
        return d.fail("layout is empty (slots == 0 or dimensions are zero)");

    cudaError_t rc = cudaGetDevice(&d.device);
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaGetDevice", rc));

    std::size_t free_bytes = 0, total_bytes = 0;
    rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaMemGetInfo", rc));
    d.free_at_reserve  = static_cast<long long>(free_bytes);
    d.total_at_reserve = static_cast<long long>(total_bytes);

    // Sec 4.4.  Both refusals are hard; neither reduces the slot count.
    d.admission = arenaAdmit(d.offsets, free_bytes, total_bytes);
    if (!d.admission.granted) {
        char why[512];
        if (d.admission.per_slot_over_ceiling) {
            std::snprintf(why, sizeof(why),
                          "per-slot footprint %llu B exceeds the Sec 3.6 ceiling %llu B",
                          static_cast<unsigned long long>(d.admission.per_slot_bytes),
                          static_cast<unsigned long long>(kArenaPerSlotByteCeiling));
        } else {
            std::snprintf(why, sizeof(why),
                          "VRAM admission refused: requested %llu B (layout %llu + 10%% "
                          "fragmentation) > usable %llu B (free %llu - 10%% driver reserve %llu); "
                          "%llu slots x %llu B",
                          static_cast<unsigned long long>(d.admission.requested_bytes),
                          static_cast<unsigned long long>(d.admission.required_bytes),
                          static_cast<unsigned long long>(d.admission.usable_bytes),
                          static_cast<unsigned long long>(free_bytes),
                          static_cast<unsigned long long>(d.admission.driver_reserve_bytes),
                          static_cast<unsigned long long>(d.offsets.slot_count),
                          static_cast<unsigned long long>(d.offsets.per_slot_bytes));
        }
        return d.fail(why);
    }

    rc = cudaStreamCreateWithFlags(&d.setup, cudaStreamNonBlocking);
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaStreamCreateWithFlags", rc));

    // A dedicated pool, so the release threshold below cannot disturb anything
    // else on the device.  If pool creation is unavailable, fall back to the
    // device's default pool rather than to a non-pooled allocation: the
    // stream-ordered semantics are what the setup path is written against.
    cudaMemPoolProps props{};
    props.allocType     = cudaMemAllocationTypePinned;
    props.handleTypes   = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id   = d.device;
    rc                  = cudaMemPoolCreate(&d.pool, &props);
    if (rc == cudaSuccess) {
        d.owns_pool = true;
    } else {
        d.pool      = nullptr;
        d.owns_pool = false;
        rc          = cudaDeviceGetDefaultMemPool(&d.pool, d.device);
        if (rc != cudaSuccess) return d.fail(cudaWhy("cudaDeviceGetDefaultMemPool", rc));
    }

    // Hold the pages for the life of the run.  With the default threshold of 0
    // the driver returns freed blocks to the OS at every synchronisation, which
    // for a block this size is a fault storm at the next statepoint.
    std::uint64_t threshold = static_cast<std::uint64_t>(d.offsets.total_bytes);
    rc = cudaMemPoolSetAttribute(d.pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaMemPoolSetAttribute(ReleaseThreshold)", rc));

    void* raw = nullptr;
    rc = cudaMallocFromPoolAsync(&raw, d.offsets.total_bytes, d.pool, d.setup);
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaMallocFromPoolAsync", rc));

    // The one synchronisation: after this the base address is final, which is
    // the precondition for capturing a graph over any pointer derived from it.
    rc = cudaStreamSynchronize(d.setup);
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaStreamSynchronize(reserve)", rc));

    d.base = static_cast<unsigned char*>(raw);
    if (d.base == nullptr) return d.fail("pool returned a null block");

    // Deterministic initial contents: a slot that is read before it is imported
    // reads zeros, not a previous run's bytes.
    rc = cudaMemsetAsync(d.base, 0, d.offsets.total_bytes, d.setup);
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaMemsetAsync(arena)", rc));
    rc = cudaStreamSynchronize(d.setup);
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaStreamSynchronize(memset)", rc));

    d.ready  = true;
    d.status = "ok";
    return true;
}

void GpuPhysicsArena::release() {
    Impl& d = *_impl;
    if (d.base != nullptr) {
        cudaFreeAsync(d.base, d.setup);
        cudaStreamSynchronize(d.setup);
        d.base = nullptr;
    }
    if (d.owns_pool && d.pool != nullptr) {
        cudaMemPoolDestroy(d.pool);
        d.owns_pool = false;
    }
    d.pool = nullptr;
    if (d.setup != nullptr) {
        cudaStreamDestroy(d.setup);
        d.setup = nullptr;
    }
    d.ready = false;
}

bool GpuPhysicsArena::available() const { return _impl->ready; }

const std::string& GpuPhysicsArena::status() const { return _impl->status; }

const ArenaOffsets& GpuPhysicsArena::offsets() const { return _impl->offsets; }

const ArenaDims& GpuPhysicsArena::dims() const { return _impl->offsets.dims; }

const ArenaAdmission& GpuPhysicsArena::admission() const { return _impl->admission; }

void* GpuPhysicsArena::base() const { return _impl->base; }

void* GpuPhysicsArena::geometryRegion(GeometryRegion region) const {
    const Impl& d = *_impl;
    if (!d.ready) return nullptr;
    return d.base + d.offsets.geometry[static_cast<int>(region)].offset;
}

void* GpuPhysicsArena::libraryRegion(LibraryRegion region) const {
    const Impl& d = *_impl;
    if (!d.ready) return nullptr;
    return d.base + d.offsets.library[static_cast<int>(region)].offset;
}

void* GpuPhysicsArena::slotRegion(int slot, SlotRegion region) const {
    const Impl& d = *_impl;
    if (!d.ready || slot < 0 || static_cast<std::size_t>(slot) >= d.offsets.slot_count)
        return nullptr;
    return d.base + d.offsets.slotRegionOffset(slot, region);
}

void* GpuPhysicsArena::scratch(int slot, DevicePhase phase, ScratchId id) const {
    const Impl& d = *_impl;
    if (!d.ready || slot < 0 || static_cast<std::size_t>(slot) >= d.offsets.slot_count)
        return nullptr;
#ifndef NDEBUG
    // Sec 4.2 debug trap.  Release builds do not read the tag; this is the
    // build where a global-lifetime alias -- slot A in Outer, slot B in
    // Depletion, both handed the same band -- gets caught.
    if (!arenaScratchPhaseAllowed(id, phase)) {
        std::fprintf(stderr,
                     "[RASBERY][GPU_ARENA][TRAP] {\"scratch\":\"%s\",\"phase\":%u,"
                     "\"slot\":%d,\"reason\":\"phase does not own this scratch\"}\n",
                     arenaScratchName(id), static_cast<unsigned>(phase), slot);
        return nullptr;
    }
#else
    (void)phase;
#endif
    return d.base + d.offsets.scratchOffset(slot, id);
}

// ---------------------------------------------------------------------------
// Views.  Pure index rebases over the fixed offsets -- no arithmetic on the
// values, only on which slot's array a read lands in (cf. nodalSlotView).
// ---------------------------------------------------------------------------

namespace {

template <typename T>
T* at(unsigned char* base, std::size_t offset) {
    return reinterpret_cast<T*>(base + offset);
}

} // namespace

DeviceGeometryView GpuPhysicsArena::geometryView() const {
    const Impl&        d = *_impl;
    DeviceGeometryView v{};
    if (!d.ready) return v;
    unsigned char*     b = d.base;
    const ArenaOffsets& o = d.offsets;
    const auto go = [&](GeometryRegion r) { return o.geometry[static_cast<int>(r)].offset; };

    v.hmesh   = at<double>(b, go(GeometryRegion::Hmesh));
    v.hz      = at<double>(b, go(GeometryRegion::Hz));
    v.vol     = at<double>(b, go(GeometryRegion::Vol));
    v.vola    = at<double>(b, go(GeometryRegion::Vola));
    v.albedo  = at<double>(b, go(GeometryRegion::Albedo));
    v.neib    = at<int>(b, go(GeometryRegion::Neib));
    v.neibr   = at<int>(b, go(GeometryRegion::Neibr));
    v.neibrb  = at<int>(b, go(GeometryRegion::Neibrb));
    v.lklr    = at<int>(b, go(GeometryRegion::Lklr));
    v.idirlr  = at<int>(b, go(GeometryRegion::Idirlr));
    v.sgnlr   = at<int>(b, go(GeometryRegion::Sgnlr));
    v.lktosfc = at<int>(b, go(GeometryRegion::Lktosfc));
    v.ltola   = at<int>(b, go(GeometryRegion::Ltola));
    v.ltolc   = at<int>(b, go(GeometryRegion::Ltolc));
    v.comps   = at<int>(b, go(GeometryRegion::Comps));
    v.fuel_nodes = at<int>(b, go(GeometryRegion::FuelNodes));
    v.is_fuel    = at<int>(b, go(GeometryRegion::IsFuel));

    v.nxyz   = o.dims.nxyz;
    v.nsurf  = o.dims.nsurf;
    v.nxy    = o.dims.nxy;
    v.nz     = o.dims.nz;
    v.nxya   = o.dims.nxya;
    v.n_fuel = o.dims.n_fuel;
    v.ng     = o.dims.ng;
    v.symang = 0;
    return v;
}

DeviceXsLibraryView GpuPhysicsArena::libraryView() const {
    const Impl&         d = *_impl;
    DeviceXsLibraryView v{};
    if (!d.ready) return v;
    unsigned char*      b = d.base;
    const ArenaOffsets& o = d.offsets;
    const auto lo = [&](LibraryRegion r) { return o.library[static_cast<int>(r)].offset; };

    v.lib_lmpx        = at<double>(b, lo(LibraryRegion::LibLmpxScalar));
    v.lib_lmpx_ssm    = at<double>(b, lo(LibraryRegion::LibLmpxScatter));
    v.lib_micx        = at<double>(b, lo(LibraryRegion::LibMicxScalar));
    v.lib_micx_ssm    = at<double>(b, lo(LibraryRegion::LibMicxScatter));
    v.lib_iden        = at<double>(b, lo(LibraryRegion::LibIden));
    v.lib_burn        = at<double>(b, lo(LibraryRegion::LibBurn));
    v.lib_wvfr        = at<double>(b, lo(LibraryRegion::LibWvfr));
    v.lib_flux        = nullptr;
    v.lib_chix        = nullptr;
    v.coeff_lmpx      = at<double>(b, lo(LibraryRegion::CoeffLmpxScalar));
    v.coeff_lmpx_ssm  = at<double>(b, lo(LibraryRegion::CoeffLmpxScatter));
    v.coeff_micx      = at<double>(b, lo(LibraryRegion::CoeffMicxScalar));
    v.coeff_micx_ssm  = at<double>(b, lo(LibraryRegion::CoeffMicxScatter));
    v.knots           = at<double>(b, lo(LibraryRegion::Knots));
    v.knot_offsets    = nullptr;
    v.dep_decay       = at<double>(b, lo(LibraryRegion::DepDecay));
    v.dep_trans       = at<double>(b, lo(LibraryRegion::DepTrans));
    v.cram_alpha      = at<double>(b, lo(LibraryRegion::CramAlpha));
    v.cram_theta      = at<double>(b, lo(LibraryRegion::CramTheta));
    v.cram_alpha0     = 0.0;

    v.n_ref_points   = o.dims.n_ref_points;
    v.n_coeff_points = o.dims.n_coeff_points;
    v.n_knots        = o.dims.n_knots;
    v.niso           = o.dims.niso;
    v.ng             = o.dims.ng;
    v.cram_order     = 8;              // XSSet.cpp:4008 CRAM_ORDER
    v.cram_poles     = o.dims.cram_poles;
    v.cram_first     = 3;              // XSSet.cpp:4008 first = iI135
    return v;
}

DeviceSlotView GpuPhysicsArena::slotView(int slot) const {
    const Impl&    d = *_impl;
    DeviceSlotView v{};
    if (!d.ready || slot < 0 || static_cast<std::size_t>(slot) >= d.offsets.slot_count) return v;

    unsigned char*      b = d.base;
    const ArenaOffsets& o = d.offsets;
    const auto so = [&](SlotRegion r) { return o.slotRegionOffset(slot, r); };

    v.phase  = at<DeviceSlotPhase>(b, so(SlotRegion::SlotPhase));
    v.state  = at<DeviceSlotState>(b, so(SlotRegion::SlotState));
    v.search = at<DeviceSearchState>(b, so(SlotRegion::SearchState));
    v.params = at<DeviceScheduleParams>(b, so(SlotRegion::ScheduleParams));

    v.phif = at<double>(b, so(SlotRegion::Phif));
    v.phis = at<double>(b, so(SlotRegion::Phis));
    v.jnet = at<double>(b, so(SlotRegion::Jnet));
    v.psi  = at<double>(b, so(SlotRegion::Psi));
    v.phic = at<double>(b, so(SlotRegion::Phic));

    v.dtil      = at<double>(b, so(SlotRegion::Dtil));
    v.dhat      = at<double>(b, so(SlotRegion::Dhat));
    v.diag      = at<double>(b, so(SlotRegion::Diag));
    v.cc        = at<double>(b, so(SlotRegion::Cc));
    v.src       = at<double>(b, so(SlotRegion::Src));
    v.cmfd_psi  = at<double>(b, so(SlotRegion::CmfdPsi));
    v.bicg_vec  = at<double>(b, so(SlotRegion::BicgVec));
    v.bicg_dinv = at<double>(b, so(SlotRegion::BicgDinv));

    v.trlcff      = at<double>(b, so(SlotRegion::Trlcff));
    v.nodal_const = at<double>(b, so(SlotRegion::NodalConst));
    v.constant_xs = at<double>(b, so(SlotRegion::ConstantXs));
    v.dsncff      = at<double>(b, so(SlotRegion::Dsncff));
    v.mutau       = at<double>(b, so(SlotRegion::MuTau));
    v.matm        = at<double>(b, so(SlotRegion::MatM));

    v.iden     = at<double>(b, so(SlotRegion::Iden));
    v.xs       = at<double>(b, so(SlotRegion::Xs));
    v.xs_ssm   = at<double>(b, so(SlotRegion::XsSsm));
    v.lmpx     = at<double>(b, so(SlotRegion::Lmpx));
    v.lmpx_ssm = at<double>(b, so(SlotRegion::LmpxSsm));
    v.micx     = at<double>(b, so(SlotRegion::Micx));
    v.micx_ssm = at<double>(b, so(SlotRegion::MicxSsm));

    v.ref_micx     = at<double>(b, so(SlotRegion::RefMicx));
    v.ref_micx_ssm = at<double>(b, so(SlotRegion::RefMicxSsm));
    v.ref_lmpx     = at<double>(b, so(SlotRegion::RefLmpx));
    v.ref_lmpx_ssm = at<double>(b, so(SlotRegion::RefLmpxSsm));
    v.ref_iden     = at<double>(b, so(SlotRegion::RefIden));

    v.bos_micx     = at<double>(b, so(SlotRegion::BosMicx));
    v.bos_xskf     = at<double>(b, so(SlotRegion::BosXskf));
    v.bos_iden     = at<double>(b, so(SlotRegion::BosIden));
    v.bos_flux     = at<double>(b, so(SlotRegion::BosFlux));
    v.bos_burn_key = at<int>(b, so(SlotRegion::BosBurnKey));

    v.xe_aa_history = at<double>(b, so(SlotRegion::XeAaHistory));

    v.bppm         = at<double>(b, so(SlotRegion::Bppm));
    v.tful         = at<double>(b, so(SlotRegion::Tful));
    v.tmod         = at<double>(b, so(SlotRegion::Tmod));
    v.dmod         = at<double>(b, so(SlotRegion::Dmod));
    v.rod_fraction = at<double>(b, so(SlotRegion::RodFraction));
    v.node_wvfr    = at<double>(b, so(SlotRegion::NodeWvfr));

    v.ppr_p  = at<double>(b, so(SlotRegion::PprP));
    v.ppr_a  = at<double>(b, so(SlotRegion::PprA));
    v.ppr_c  = at<double>(b, so(SlotRegion::PprC));
    v.ppr_q  = at<double>(b, so(SlotRegion::PprQ));
    v.ppr_l  = at<double>(b, so(SlotRegion::PprL));
    v.ppr_bt = at<double>(b, so(SlotRegion::PprBt));

    v.burn_key = at<int>(b, so(SlotRegion::BurnKey));
    v.ctyp_key = at<int>(b, so(SlotRegion::CtypKey));
    v.out_pack = at<double>(b, so(SlotRegion::OutPack));

    v.nxyz   = o.dims.nxyz;
    v.nsurf  = o.dims.nsurf;
    v.nxy    = o.dims.nxy;
    v.n_fuel = o.dims.n_fuel;
    v.ng     = o.dims.ng;
    v.slot   = slot;
    return v;
}

// ---------------------------------------------------------------------------
// Byte transport.  No interpretation: an id, a host pointer, a byte count.  A
// request wider than the region is refused rather than clipped -- a short copy
// that succeeds is a wrong answer waiting for a reader.
// ---------------------------------------------------------------------------

namespace {

bool copyAsync(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind,
               cudaStream_t stream, std::string& status) {
    const cudaError_t rc = cudaMemcpyAsync(dst, src, bytes, kind, stream);
    if (rc != cudaSuccess) {
        status = cudaWhy("cudaMemcpyAsync", rc);
        return false;
    }
    return true;
}

} // namespace

bool GpuPhysicsArena::importGeometryAsync(GeometryRegion region, const void* host,
                                          std::size_t bytes, GpuStreamHandle stream) {
    Impl& d = *_impl;
    if (!d.ready) return d.fail("importGeometryAsync on an unavailable arena");
    const ArenaRegion& r = d.offsets.geometry[static_cast<int>(region)];
    if (bytes > r.bytes) return d.fail("importGeometryAsync: payload wider than the region");
    return copyAsync(d.base + r.offset, host, bytes, cudaMemcpyHostToDevice,
                     static_cast<cudaStream_t>(stream), d.status);
}

bool GpuPhysicsArena::importLibraryAsync(LibraryRegion region, const void* host,
                                         std::size_t bytes, GpuStreamHandle stream) {
    Impl& d = *_impl;
    if (!d.ready) return d.fail("importLibraryAsync on an unavailable arena");
    const ArenaRegion& r = d.offsets.library[static_cast<int>(region)];
    if (bytes > r.bytes) return d.fail("importLibraryAsync: payload wider than the region");
    return copyAsync(d.base + r.offset, host, bytes, cudaMemcpyHostToDevice,
                     static_cast<cudaStream_t>(stream), d.status);
}

bool GpuPhysicsArena::importSlotAsync(int slot, SlotRegion region, const void* host,
                                      std::size_t bytes, GpuStreamHandle stream) {
    Impl& d = *_impl;
    if (!d.ready) return d.fail("importSlotAsync on an unavailable arena");
    if (slot < 0 || static_cast<std::size_t>(slot) >= d.offsets.slot_count)
        return d.fail("importSlotAsync: slot out of range");
    if (bytes > d.offsets.slotRegionBytes(region))
        return d.fail("importSlotAsync: payload wider than the region");
    return copyAsync(d.base + d.offsets.slotRegionOffset(slot, region), host, bytes,
                     cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream), d.status);
}

bool GpuPhysicsArena::exportSnapshotAsync(int slot, SlotRegion region, void* host,
                                          std::size_t bytes, GpuStreamHandle stream) {
    Impl& d = *_impl;
    if (!d.ready) return d.fail("exportSnapshotAsync on an unavailable arena");
    if (slot < 0 || static_cast<std::size_t>(slot) >= d.offsets.slot_count)
        return d.fail("exportSnapshotAsync: slot out of range");
    if (bytes > d.offsets.slotRegionBytes(region))
        return d.fail("exportSnapshotAsync: request wider than the region");
    return copyAsync(host, d.base + d.offsets.slotRegionOffset(slot, region), bytes,
                     cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream), d.status);
}

bool GpuPhysicsArena::clearSlotAsync(int slot, GpuStreamHandle stream) {
    Impl& d = *_impl;
    if (!d.ready) return d.fail("clearSlotAsync on an unavailable arena");
    if (slot < 0 || static_cast<std::size_t>(slot) >= d.offsets.slot_count)
        return d.fail("clearSlotAsync: slot out of range");
    const cudaError_t rc = cudaMemsetAsync(d.base + d.offsets.slotBase(slot), 0,
                                           d.offsets.per_slot_bytes,
                                           static_cast<cudaStream_t>(stream));
    if (rc != cudaSuccess) return d.fail(cudaWhy("cudaMemsetAsync(slot)", rc));
    return true;
}

// ---------------------------------------------------------------------------
// Receipt (Sec 9.3).
// ---------------------------------------------------------------------------

std::string GpuPhysicsArena::receiptJson() const {
    const Impl&         d = *_impl;
    const ArenaOffsets& o = d.offsets;
    char                buf[1400];
    std::snprintf(
        buf, sizeof(buf),
        "{\"backend\":\"cuda\",\"available\":%s,\"shared_geometry_bytes\":%llu,"
        "\"shared_library_bytes\":%llu,\"per_slot_bytes\":%llu,\"slot_count\":%llu,"
        "\"scratch_bytes\":%llu,\"per_slot_scratch_bytes\":%llu,\"total_bytes\":%llu,"
        "\"alignment\":%llu,\"slot_base\":%llu,\"vram_free_bytes\":%lld,"
        "\"vram_total_bytes\":%lld,\"driver_reserve_bytes\":%llu,"
        "\"fragmentation_reserve_bytes\":%llu,\"per_slot_ceiling_bytes\":%llu,"
        "\"admitted\":%s,\"pool\":\"%s\",\"status\":\"%s\"}",
        d.ready ? "true" : "false",
        static_cast<unsigned long long>(o.shared_geometry_bytes),
        static_cast<unsigned long long>(o.shared_library_bytes),
        static_cast<unsigned long long>(o.per_slot_bytes),
        static_cast<unsigned long long>(o.slot_count),
        static_cast<unsigned long long>(o.per_slot_scratch_bytes * o.slot_count),
        static_cast<unsigned long long>(o.per_slot_scratch_bytes),
        static_cast<unsigned long long>(o.total_bytes),
        static_cast<unsigned long long>(kArenaAlignment),
        static_cast<unsigned long long>(o.slot_base), d.free_at_reserve, d.total_at_reserve,
        static_cast<unsigned long long>(d.admission.driver_reserve_bytes),
        static_cast<unsigned long long>(d.admission.fragmentation_reserve_bytes),
        static_cast<unsigned long long>(kArenaPerSlotByteCeiling),
        d.admission.granted ? "true" : "false", d.owns_pool ? "dedicated" : "default",
        d.status.c_str());
    return std::string(buf);
}

void GpuPhysicsArena::emitReceipt(std::ostream& os) const {
    os << "[RASBERY][GPU_ARENA] " << receiptJson() << "\n";
}

} // namespace rasbery::gpu
