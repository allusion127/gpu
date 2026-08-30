#pragma once

// GpuDeviceBlockPool -- WP10.6.  A process-lifetime free list for the device
// blocks a CASE owns, and the counters that let a receipt say what VRAM this
// process is holding without asking the board.
//
// THE HOLE THIS CLOSES.  `XsReconBackend::Impl` is per XSSet, per Driver, per
// deck -- CudaXsReconBackend.cu says so in as many words beside `xe_hist`.  So
// every case in a generation runs the same twelve `cudaMalloc` calls on the way
// in and the same twelve `cudaFree` calls on the way out, for blocks whose
// SIZES are a function of the geometry and are therefore identical across the
// candidates of one GA generation.  At width 16 that is ~24 device API calls per
// case, ~430 per generation, ~4.3 million over the 10k generations the GA
// wants -- and `cudaFree` is a synchronising call, so each one is a device-wide
// barrier paid to hand back a block the next case will immediately ask for
// again in the same size.
//
// WHAT THIS IS NOT.  It is not a suballocator and it is not a cache with a
// policy.  It is a map from an EXACT byte count to a list of blocks of exactly
// that many bytes, and `take(bytes)` either returns one of those or returns
// nullptr.  There is no splitting, no coalescing, no best fit and no rounding:
// a block handed back to a caller is byte-for-byte the block `cudaMalloc` would
// have produced for that request, so nothing downstream can observe which of
// the two it got.  A pool that could hand out a 74.9 MB block for a 74.8 MB
// request would be a pool that changes what a kernel may read.
//
// WHY IT IS PURE HOST.  Same reason HostPinRegistry.h is: the bookkeeping has
// to be readable from EvaluatorServer.h, which is compiled by the host compiler
// and has no CUDA runtime.  So this header owns the map, the free list and the
// counters, and the two functions that actually call `cudaMalloc`/`cudaFree`
// live in the .cu that includes it (guarded on `__CUDACC__` below).
//
// EXACTNESS (B0).  Allocation LIFETIME is not observable in a result.  A pooled
// block and a fresh `cudaMalloc` block are both uninitialised device memory --
// `cudaMalloc` makes no zeroing promise either -- so any code that could tell
// them apart is code that reads memory it never wrote, which is a defect the
// pool exposes rather than causes.  It is nevertheless GATED (see enabled()):
// `RASBERY_ARENA_PERSIST=1` turns reuse on, and with it unset the wrappers
// count and register but always go to the driver, which is the pre-WP10.6
// behaviour instruction for instruction.  The counters are NOT gated -- a
// receipt that only reports when a flag is set is a receipt that cannot be
// compared across the two arms of the A/B that promotes the flag.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rasbery::gpu::blockpool {

/// One line of the WP10.6 half of `[RASBERY][EVALUATOR][MEM]`.
struct Stats {
    std::uint64_t device_allocs   = 0; ///< cudaMalloc calls that reached the driver
    std::uint64_t device_frees    = 0; ///< cudaFree calls that reached the driver
    std::uint64_t pool_hits       = 0; ///< allocations served from the free list
    std::uint64_t pool_parks      = 0; ///< frees absorbed by the free list
    std::uint64_t arena_rebuilds  = 0; ///< live regions freed and re-laid-out
    std::uint64_t bytes_live      = 0; ///< registered and handed to a caller
    std::uint64_t bytes_pooled    = 0; ///< parked in the free list, still ours
    std::uint64_t bytes_high_water = 0; ///< max(bytes_live + bytes_pooled)
    std::size_t   blocks_live     = 0;
    std::size_t   blocks_pooled   = 0;
};

namespace detail {

struct Registration {
    std::size_t bytes    = 0;
    bool        poolable = false;
};

struct State {
    std::mutex                                        mutex;
    std::unordered_map<const void*, Registration>     live;
    std::unordered_map<std::size_t, std::vector<void*>> parked;
    Stats                                             stats;
};

inline State& state() {
    // Leaked on purpose, exactly as g_flatxs_libs is: tearing device
    // bookkeeping down during static destruction races the CUDA runtime's own
    // teardown, and the process is exiting anyway.
    static State* s = new State();
    return *s;
}

inline void noteFootprintLocked(State& s) {
    const std::uint64_t total = s.stats.bytes_live + s.stats.bytes_pooled;
    if (total > s.stats.bytes_high_water) s.stats.bytes_high_water = total;
}

} // namespace detail

/// `RASBERY_ARENA_PERSIST` -- read once, because a flag that can change under a
/// running process is a flag that can park a block the free list will never be
/// allowed to hand back.
inline bool enabled() {
    static const bool on = [] {
        const char* raw = std::getenv("RASBERY_ARENA_PERSIST");
        if (raw == nullptr) return false;
        return !(raw[0] == '\0' || std::strcmp(raw, "0") == 0);
    }();
    return on;
}

/// A block of EXACTLY `bytes`, or nullptr.  Never a bigger one.
inline void* take(std::size_t bytes) {
    if (!enabled() || bytes == 0) return nullptr;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.parked.find(bytes);
    if (it == s.parked.end() || it->second.empty()) return nullptr;
    void* block = it->second.back();
    it->second.pop_back();
    s.stats.pool_hits += 1;
    s.stats.bytes_pooled -= bytes;
    s.stats.bytes_live   += bytes;
    s.stats.blocks_pooled -= 1;
    s.stats.blocks_live   += 1;
    s.live[block] = detail::Registration{bytes, true};
    return block;
}

/// Record a block the driver just produced.  `poolable` is false for the
/// process-lifetime singletons (the nodal arena, the flat-XS library, the one
/// GpuPhysicsArena block): they are counted so `bytes_live` is the whole device
/// footprint, and they are never parked because they are never freed.
inline void noteAllocated(const void* block, std::size_t bytes, bool poolable) {
    if (block == nullptr) return;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats.device_allocs += 1;
    s.stats.bytes_live    += bytes;
    s.stats.blocks_live   += 1;
    s.live[block] = detail::Registration{bytes, poolable};
    detail::noteFootprintLocked(s);
}

/// True when the free list took ownership and the caller must NOT call
/// cudaFree.  False means "still yours to free", which is also the answer for
/// a pointer this pool never saw -- an unregistered block is freed exactly as
/// it was before, so a site this header does not know about is not broken by
/// it.
inline bool give(const void* block) {
    if (block == nullptr) return true; // nothing to free, and cudaFree(nullptr) is a no-op
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.live.find(block);
    if (it == s.live.end()) return false;
    const detail::Registration reg = it->second;
    s.live.erase(it);
    s.stats.bytes_live  -= reg.bytes;
    s.stats.blocks_live -= 1;
    if (!enabled() || !reg.poolable || reg.bytes == 0) {
        s.stats.device_frees += 1;
        return false;
    }
    s.parked[reg.bytes].push_back(const_cast<void*>(block));
    s.stats.pool_parks   += 1;
    s.stats.bytes_pooled += reg.bytes;
    s.stats.blocks_pooled += 1;
    detail::noteFootprintLocked(s);
    return true;
}

/// A live device region was freed and re-laid-out because its SHAPE changed
/// under it -- the one event that would make a per-generation arena teardown
/// visible in a receipt instead of inferred from a VRAM trace.
inline void noteArenaRebuild() {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats.arena_rebuilds += 1;
}

inline Stats snapshot() {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.stats;
}

/// Bytes this PROCESS holds on the device: in use plus parked.  The number
/// `nvidia-smi` cannot give, because `nvidia-smi` answers for the board and a
/// board can have eight other tenants on it (WP10.6: it did).
inline std::uint64_t deviceBytes() {
    const Stats s = snapshot();
    return s.bytes_live + s.bytes_pooled;
}

/// Release every parked block through `freer`, for a shutdown path that wants
/// the driver to see the frees.  Returns the number released.  Not called on
/// the hot path and not required for correctness: process exit reclaims the
/// device just as thoroughly.
template <typename Freer>
inline std::size_t drain(Freer freer) {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::size_t released = 0;
    for (auto& entry : s.parked) {
        for (void* block : entry.second) {
            freer(block);
            s.stats.device_frees += 1;
            s.stats.bytes_pooled -= entry.first;
            s.stats.blocks_pooled -= 1;
            ++released;
        }
        entry.second.clear();
    }
    s.parked.clear();
    return released;
}

/// TEST ONLY.  Forgets every parked block without freeing it.  A soak must
/// never call this; the contract test does, so one process can exercise both
/// arms.
inline void resetForTest() {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.live.clear();
    s.parked.clear();
    s.stats = Stats{};
}

} // namespace rasbery::gpu::blockpool

// The CUDA half.  Guarded on `__CUDACC__` rather than on CUDART_VERSION so the
// include order of the translation unit cannot decide whether these exist: nvcc
// defines it for both passes of a .cu, and a host TU that only wants the
// counters (EvaluatorServer.h) gets the pure-host half above and no CUDA
// dependency at all.
#ifdef __CUDACC__

#include "GpuCaptureArbiter.h"

#include <cuda_runtime.h>

namespace rasbery::gpu {

/// `cudaMalloc`, through the WP10.6 free list.  Byte-for-byte the same contract
/// as `cudaMalloc(pp, bytes)`: on success `*pp` is a device pointer to `bytes`
/// uninitialised bytes, on failure the return value is the driver's.
inline cudaError_t deviceBlockAlloc(void** pp, std::size_t bytes) {
    if (pp == nullptr) return cudaErrorInvalidValue;
    // A POOL HIT TOUCHES NO CUDA API AT ALL, so it takes no AllocWindow: there
    // is nothing for a sibling lane's capture to be invalidated by, and a
    // window opened here would only inflate `alloc_windows` with events that
    // never reached the driver.  The window is taken on the path that does.
    if (void* reused = blockpool::take(bytes)) {
        *pp = reused;
        return cudaSuccess;
    }
    rasbery::AllocWindow _alloc_window("blockpool.cudaMalloc");
    const cudaError_t rc = cudaMalloc(pp, bytes);
    if (rc == cudaSuccess) blockpool::noteAllocated(*pp, bytes, /*poolable=*/true);
    return rc;
}

/// The same, for a block that lives as long as the process (the nodal arena,
/// the flat-XS library, the one physics-arena block).  Counted so the receipt's
/// `vram_mb` is the whole footprint; never parked, because it is never freed.
inline cudaError_t deviceBlockAllocOnce(void** pp, std::size_t bytes) {
    if (pp == nullptr) return cudaErrorInvalidValue;
    rasbery::AllocWindow _alloc_window("blockpool.cudaMalloc.once");
    const cudaError_t rc = cudaMalloc(pp, bytes);
    if (rc == cudaSuccess) blockpool::noteAllocated(*pp, bytes, /*poolable=*/false);
    return rc;
}

/// `cudaFree`, through the free list.  A parked block does not reach the
/// driver, which is the point: `cudaFree` is a synchronising call and the next
/// case is about to ask for the same size back.
inline cudaError_t deviceBlockFree(void* block) {
    // Parked, or nullptr: no driver call, so no window, for the same reason.
    if (blockpool::give(block)) return cudaSuccess;
    rasbery::AllocWindow _alloc_window("blockpool.cudaFree");
    return cudaFree(block);
}

} // namespace rasbery::gpu

#endif // __CUDACC__
