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

#include "GpuCaptureArbiter.h"
#include "GpuPhysicsArena.h"
#include "XferLedger.h"

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
    // Rev.7.1 Task 18d: cudaMemPoolCreate and the setup drain below are
    // synchronising, and this reserve runs per deck on the deck's own thread.
    rasbery::AllocWindow _alloc_window("physics.arena.reserve");
    Impl& d = *_impl;
    if (d.ready) return d.fail("reserve() called twice; the arena is allocated once");

    d.offsets = arenaComputeLayout(dims);
    if (d.offsets.slots_exceed_cap) {
        char why[192];
        std::snprintf(why, sizeof(why),
                      "%d slots requested but the scheduler classifies at most %d; "
                      "the extra slots would never be scheduled",
                      dims.slots, kMaxDeviceSlots);
        return d.fail(why);
    }
    if (!d.offsets.valid) return d.fail("layout was rejected");
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

    // From here on the arena owns a stream, and soon a pool and a block.  Every
    // failure below therefore tears them down before reporting: a refused
    // reserve() must leave the object exactly as constructed, not holding a
    // stream and a 14 GiB reservation nobody will ever free.
    const auto abort_reserve = [this, &d](std::string why) {
        release();
        return d.fail(std::move(why));
    };

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
        if (rc != cudaSuccess) return abort_reserve(cudaWhy("cudaDeviceGetDefaultMemPool", rc));
    }

    // Hold the pages for the life of the run.  With the default threshold of 0
    // the driver returns freed blocks to the OS at every synchronisation, which
    // for a block this size is a fault storm at the next statepoint.
    std::uint64_t threshold = static_cast<std::uint64_t>(d.offsets.total_bytes);
    rc = cudaMemPoolSetAttribute(d.pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    if (rc != cudaSuccess)
        return abort_reserve(cudaWhy("cudaMemPoolSetAttribute(ReleaseThreshold)", rc));

    void* raw = nullptr;
    rc = cudaMallocFromPoolAsync(&raw, d.offsets.total_bytes, d.pool, d.setup);
    if (rc != cudaSuccess) return abort_reserve(cudaWhy("cudaMallocFromPoolAsync", rc));

    // The one synchronisation: after this the base address is final, which is
    // the precondition for capturing a graph over any pointer derived from it.
    rc = rasbery::xfer::streamSync("GpuPhysicsArenaCuda.cu:reserve", "alloc", d.setup);
    if (rc != cudaSuccess) return abort_reserve(cudaWhy("cudaStreamSynchronize(reserve)", rc));

    d.base = static_cast<unsigned char*>(raw);
    if (d.base == nullptr) return abort_reserve("pool returned a null block");

    // Every region offset is a multiple of 256, so the whole layout is only
    // 256-aligned if the BASE is.  cudaMallocFromPoolAsync gives at least 256
    // in practice; refusing rather than assuming means a driver that ever gives
    // less produces a clear failure instead of misaligned vector loads.
    if ((reinterpret_cast<std::uintptr_t>(d.base) % kArenaAlignment) != 0) {
        char why[160];
        std::snprintf(why, sizeof(why),
                      "pool block is not %zu-byte aligned (base %% %zu = %zu)", kArenaAlignment,
                      kArenaAlignment,
                      static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(d.base) %
                                               kArenaAlignment));
        return abort_reserve(why);
    }

    // Deterministic initial contents: a slot that is read before it is imported
    // reads zeros, not a previous run's bytes.
    rc = cudaMemsetAsync(d.base, 0, d.offsets.total_bytes, d.setup);
    if (rc != cudaSuccess) return abort_reserve(cudaWhy("cudaMemsetAsync(arena)", rc));
    rc = rasbery::xfer::streamSync("GpuPhysicsArenaCuda.cu:reserve", "memset", d.setup);
    if (rc != cudaSuccess) return abort_reserve(cudaWhy("cudaStreamSynchronize(memset)", rc));

    d.ready  = true;
    d.status = "ok";
    return true;
}

void GpuPhysicsArena::release() {
    rasbery::AllocWindow _alloc_window("physics.arena.release");
    Impl& d = *_impl;
    if (d.base != nullptr) {
        cudaFreeAsync(d.base, d.setup);
        rasbery::xfer::streamSync("GpuPhysicsArenaCuda.cu:release", "setup", d.setup);
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

/// A region with no elements gets a NULL pointer, not a pointer to the next
/// region's first byte.  An empty library (no deck loaded yet) or an empty
/// fuel-node set would otherwise hand a kernel an address that is valid memory
/// belonging to something ELSE, so the guard against reading it would have to
/// be repeated at every call site instead of living here once.
template <typename T>
T* atOrNull(unsigned char* base, const ArenaRegion& r) {
    return r.bytes == 0 ? nullptr : reinterpret_cast<T*>(base + r.offset);
}

/// Same rule for a per-slot region, whose stored offset is slot-relative.
template <typename T>
T* slotPtr(unsigned char* base, std::size_t slot_base, const ArenaRegion& r) {
    return r.bytes == 0 ? nullptr : reinterpret_cast<T*>(base + slot_base + r.offset);
}

} // namespace

DeviceGeometryView GpuPhysicsArena::geometryView() const {
    const Impl&        d = *_impl;
    DeviceGeometryView v{};
    if (!d.ready) return v;
    unsigned char*     b = d.base;
    const ArenaOffsets& o = d.offsets;
    const auto go = [&](GeometryRegion r) -> const ArenaRegion& {
        return o.geometry[static_cast<int>(r)];
    };

    v.hmesh   = atOrNull<double>(b, go(GeometryRegion::Hmesh));
    v.hz      = atOrNull<double>(b, go(GeometryRegion::Hz));
    v.vol     = atOrNull<double>(b, go(GeometryRegion::Vol));
    v.vola    = atOrNull<double>(b, go(GeometryRegion::Vola));
    v.albedo  = atOrNull<double>(b, go(GeometryRegion::Albedo));
    v.neib    = atOrNull<int>(b, go(GeometryRegion::Neib));
    v.neibr   = atOrNull<int>(b, go(GeometryRegion::Neibr));
    v.neibrb  = atOrNull<int>(b, go(GeometryRegion::Neibrb));
    v.lklr    = atOrNull<int>(b, go(GeometryRegion::Lklr));
    v.idirlr  = atOrNull<int>(b, go(GeometryRegion::Idirlr));
    v.sgnlr   = atOrNull<int>(b, go(GeometryRegion::Sgnlr));
    v.lktosfc = atOrNull<int>(b, go(GeometryRegion::Lktosfc));
    v.ltola   = atOrNull<int>(b, go(GeometryRegion::Ltola));
    v.ltolc   = atOrNull<int>(b, go(GeometryRegion::Ltolc));
    v.fuel_nodes = atOrNull<int>(b, go(GeometryRegion::FuelNodes));
    v.is_fuel    = atOrNull<int>(b, go(GeometryRegion::IsFuel));

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
    const auto lo = [&](LibraryRegion r) -> const ArenaRegion& {
        return o.library[static_cast<int>(r)];
    };

    v.lib_lmpx        = atOrNull<double>(b, lo(LibraryRegion::LibLmpxScalar));
    v.lib_lmpx_ssm    = atOrNull<double>(b, lo(LibraryRegion::LibLmpxScatter));
    v.lib_micx        = atOrNull<double>(b, lo(LibraryRegion::LibMicxScalar));
    v.lib_micx_ssm    = atOrNull<double>(b, lo(LibraryRegion::LibMicxScatter));
    v.lib_iden        = atOrNull<double>(b, lo(LibraryRegion::LibIden));
    v.lib_burn        = atOrNull<double>(b, lo(LibraryRegion::LibBurn));
    v.lib_wvfr        = atOrNull<double>(b, lo(LibraryRegion::LibWvfr));
    v.lib_flux        = atOrNull<double>(b, lo(LibraryRegion::LibFlux));
    v.lib_chix        = atOrNull<double>(b, lo(LibraryRegion::LibChix));
    v.coeff_lmpx      = atOrNull<double>(b, lo(LibraryRegion::CoeffLmpxScalar));
    v.coeff_lmpx_ssm  = atOrNull<double>(b, lo(LibraryRegion::CoeffLmpxScatter));
    v.coeff_micx      = atOrNull<double>(b, lo(LibraryRegion::CoeffMicxScalar));
    v.coeff_micx_ssm  = atOrNull<double>(b, lo(LibraryRegion::CoeffMicxScatter));
    v.knots           = atOrNull<double>(b, lo(LibraryRegion::Knots));
    v.dep_decay       = atOrNull<double>(b, lo(LibraryRegion::DepDecay));
    v.dep_trans       = atOrNull<double>(b, lo(LibraryRegion::DepTrans));
    v.cram_alpha      = atOrNull<double>(b, lo(LibraryRegion::CramAlpha));
    v.cram_theta      = atOrNull<double>(b, lo(LibraryRegion::CramTheta));
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
    const std::size_t slot_base = o.slotBase(slot);
    const auto        so = [&](SlotRegion r) -> const ArenaRegion& {
        return o.slot[static_cast<int>(r)];
    };

    // DENSE, out of the contiguous control block -- element `slot` of each
    // array, NOT slot_base + offset.  This is exactly what the scheduler
    // kernels' `phases[tid]` / `states[s]` indexing means, and it is why the
    // four control structs are not in the slot stride.
    v.phase  = at<DeviceSlotPhase>(b, o.controlOffset(slot, ControlRegion::SlotPhase));
    v.state  = at<DeviceSlotState>(b, o.controlOffset(slot, ControlRegion::SlotState));
    v.search = at<DeviceSearchState>(b, o.controlOffset(slot, ControlRegion::SearchState));
    v.params = at<DeviceScheduleParams>(b, o.controlOffset(slot, ControlRegion::ScheduleParams));

    v.phif = slotPtr<double>(b, slot_base, so(SlotRegion::Phif));
    v.phis = slotPtr<double>(b, slot_base, so(SlotRegion::Phis));
    v.jnet = slotPtr<double>(b, slot_base, so(SlotRegion::Jnet));
    v.psi  = slotPtr<double>(b, slot_base, so(SlotRegion::Psi));
    v.phic = slotPtr<double>(b, slot_base, so(SlotRegion::Phic));

    v.dtil      = slotPtr<double>(b, slot_base, so(SlotRegion::Dtil));
    v.dhat      = slotPtr<double>(b, slot_base, so(SlotRegion::Dhat));
    v.diag      = slotPtr<double>(b, slot_base, so(SlotRegion::Diag));
    v.cc        = slotPtr<double>(b, slot_base, so(SlotRegion::Cc));
    v.src       = slotPtr<double>(b, slot_base, so(SlotRegion::Src));
    v.cmfd_psi  = slotPtr<double>(b, slot_base, so(SlotRegion::CmfdPsi));
    v.bicg_vec  = slotPtr<double>(b, slot_base, so(SlotRegion::BicgVec));
    v.bicg_dinv = slotPtr<double>(b, slot_base, so(SlotRegion::BicgDinv));

    v.trlcff      = slotPtr<double>(b, slot_base, so(SlotRegion::Trlcff));
    v.nodal_const = slotPtr<double>(b, slot_base, so(SlotRegion::NodalConst));
    v.constant_xs = slotPtr<double>(b, slot_base, so(SlotRegion::ConstantXs));
    v.dsncff      = slotPtr<double>(b, slot_base, so(SlotRegion::Dsncff));
    v.mutau       = slotPtr<double>(b, slot_base, so(SlotRegion::MuTau));
    v.matm        = slotPtr<double>(b, slot_base, so(SlotRegion::MatM));

    v.iden     = slotPtr<double>(b, slot_base, so(SlotRegion::Iden));
    v.xs       = slotPtr<double>(b, slot_base, so(SlotRegion::Xs));
    v.xs_ssm   = slotPtr<double>(b, slot_base, so(SlotRegion::XsSsm));
    v.lmpx     = slotPtr<double>(b, slot_base, so(SlotRegion::Lmpx));
    v.lmpx_ssm = slotPtr<double>(b, slot_base, so(SlotRegion::LmpxSsm));
    v.micx     = slotPtr<double>(b, slot_base, so(SlotRegion::Micx));
    v.micx_ssm = slotPtr<double>(b, slot_base, so(SlotRegion::MicxSsm));

    v.ref_micx     = slotPtr<double>(b, slot_base, so(SlotRegion::RefMicx));
    v.ref_micx_ssm = slotPtr<double>(b, slot_base, so(SlotRegion::RefMicxSsm));
    v.ref_lmpx     = slotPtr<double>(b, slot_base, so(SlotRegion::RefLmpx));
    v.ref_lmpx_ssm = slotPtr<double>(b, slot_base, so(SlotRegion::RefLmpxSsm));
    v.ref_iden     = slotPtr<double>(b, slot_base, so(SlotRegion::RefIden));

    v.bos_micx     = slotPtr<double>(b, slot_base, so(SlotRegion::BosMicx));
    v.bos_xskf     = slotPtr<double>(b, slot_base, so(SlotRegion::BosXskf));
    v.bos_iden     = slotPtr<double>(b, slot_base, so(SlotRegion::BosIden));
    v.bos_flux     = slotPtr<double>(b, slot_base, so(SlotRegion::BosFlux));
    v.bos_burn_key = slotPtr<int>(b, slot_base, so(SlotRegion::BosBurnKey));

    v.xe_aa_history = slotPtr<double>(b, slot_base, so(SlotRegion::XeAaHistory));

    v.bppm         = slotPtr<double>(b, slot_base, so(SlotRegion::Bppm));
    v.tful         = slotPtr<double>(b, slot_base, so(SlotRegion::Tful));
    v.tmod         = slotPtr<double>(b, slot_base, so(SlotRegion::Tmod));
    v.dmod         = slotPtr<double>(b, slot_base, so(SlotRegion::Dmod));
    v.rod_fraction = slotPtr<double>(b, slot_base, so(SlotRegion::RodFraction));
    v.node_wvfr    = slotPtr<double>(b, slot_base, so(SlotRegion::NodeWvfr));

    v.ppr_p  = slotPtr<double>(b, slot_base, so(SlotRegion::PprP));
    v.ppr_a  = slotPtr<double>(b, slot_base, so(SlotRegion::PprA));
    v.ppr_c  = slotPtr<double>(b, slot_base, so(SlotRegion::PprC));
    v.ppr_q  = slotPtr<double>(b, slot_base, so(SlotRegion::PprQ));
    v.ppr_l  = slotPtr<double>(b, slot_base, so(SlotRegion::PprL));
    v.ppr_bt = slotPtr<double>(b, slot_base, so(SlotRegion::PprBt));

    v.burn_key = slotPtr<int>(b, slot_base, so(SlotRegion::BurnKey));
    v.ctyp_key = slotPtr<int>(b, slot_base, so(SlotRegion::CtypKey));
    v.out_pack = slotPtr<double>(b, slot_base, so(SlotRegion::OutPack));

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

/// WP13.1: `leaf` is the CALLER, because this helper is the arena's only
/// copy and four different regions come through it.
bool copyAsync(const char* leaf, void* dst, const void* src, std::size_t bytes,
               cudaMemcpyKind kind, cudaStream_t stream, std::string& status) {
    const cudaError_t rc = rasbery::xfer::memcpyAsync(
        "GpuPhysicsArenaCuda.cu:copyAsync", leaf, dst, src, bytes, kind, stream);
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
    return copyAsync("importGeometryAsync", d.base + r.offset, host, bytes,
                     cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream), d.status);
}

bool GpuPhysicsArena::importLibraryAsync(LibraryRegion region, const void* host,
                                         std::size_t bytes, GpuStreamHandle stream) {
    Impl& d = *_impl;
    if (!d.ready) return d.fail("importLibraryAsync on an unavailable arena");
    const ArenaRegion& r = d.offsets.library[static_cast<int>(region)];
    if (bytes > r.bytes) return d.fail("importLibraryAsync: payload wider than the region");
    return copyAsync("importLibraryAsync", d.base + r.offset, host, bytes,
                     cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream), d.status);
}

bool GpuPhysicsArena::importSlotAsync(int slot, SlotRegion region, const void* host,
                                      std::size_t bytes, GpuStreamHandle stream) {
    Impl& d = *_impl;
    if (!d.ready) return d.fail("importSlotAsync on an unavailable arena");
    if (slot < 0 || static_cast<std::size_t>(slot) >= d.offsets.slot_count)
        return d.fail("importSlotAsync: slot out of range");
    if (bytes > d.offsets.slotRegionBytes(region))
        return d.fail("importSlotAsync: payload wider than the region");
    return copyAsync("importSlotAsync", d.base + d.offsets.slotRegionOffset(slot, region),
                     host, bytes, cudaMemcpyHostToDevice,
                     static_cast<cudaStream_t>(stream), d.status);
}

bool GpuPhysicsArena::exportSnapshotAsync(int slot, SlotRegion region, void* host,
                                          std::size_t bytes, GpuStreamHandle stream) {
    Impl& d = *_impl;
    if (!d.ready) return d.fail("exportSnapshotAsync on an unavailable arena");
    if (slot < 0 || static_cast<std::size_t>(slot) >= d.offsets.slot_count)
        return d.fail("exportSnapshotAsync: slot out of range");
    if (bytes > d.offsets.slotRegionBytes(region))
        return d.fail("exportSnapshotAsync: request wider than the region");
    return copyAsync("exportSnapshotAsync", host,
                     d.base + d.offsets.slotRegionOffset(slot, region), bytes,
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
    // Sized for the longest status string this class produces (the admission
    // refusal, ~380 chars) plus every numeric field at full width.
    char buf[2048];
    std::snprintf(
        buf, sizeof(buf),
        "{\"backend\":\"cuda\",\"available\":%s,\"shared_geometry_bytes\":%llu,"
        "\"shared_library_bytes\":%llu,\"control_block_bytes\":%llu,"
        "\"per_slot_bytes\":%llu,\"slot_count\":%llu,"
        "\"scratch_bytes\":%llu,\"per_slot_scratch_bytes\":%llu,\"total_bytes\":%llu,"
        "\"alignment\":%llu,\"slot_base\":%llu,\"vram_free_bytes\":%lld,"
        "\"vram_total_bytes\":%lld,\"driver_reserve_bytes\":%llu,"
        "\"fragmentation_reserve_bytes\":%llu,\"per_slot_ceiling_bytes\":%llu,"
        "\"admitted\":%s,\"pool\":\"%s\",\"status\":\"%s\"}",
        d.ready ? "true" : "false",
        static_cast<unsigned long long>(o.shared_geometry_bytes),
        static_cast<unsigned long long>(o.shared_library_bytes),
        static_cast<unsigned long long>(o.control_block_bytes),
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
