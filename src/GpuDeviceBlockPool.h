#pragma once

// GpuDeviceBlockPool -- WP10.6, bounded and renamed by WP10.8.  A free list for
// the device blocks a CASE owns, and the counters that let a receipt say what
// VRAM this process is holding without asking the board.
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
// WHAT THIS IS NOT.  It is not a suballocator.  It is a map from an EXACT byte
// count to a list of blocks of exactly that many bytes, and `take(bytes)`
// either returns one of those or returns nullptr.  There is no splitting, no
// coalescing, no best fit and no rounding: a block handed back to a caller is
// byte-for-byte the block `cudaMalloc` would have produced for that request, so
// nothing downstream can observe which of the two it got.  A pool that could
// hand out a 74.9 MB block for a 74.8 MB request would be a pool that changes
// what a kernel may read.
//
// WP10.8 -- IT IS NOW A CACHE WITH A POLICY, BECAUSE UNBOUNDED WAS A POLICY TOO.
// The WP10.6 free list only ever grew: a block parked at generation 0 for a
// geometry no later case asks for again was held to process exit, and nothing
// in any receipt said so.  The 238 arm-B soak (block 38 phase 2) settled at
// 6039 MB of device footprint and could not distinguish "this is the working
// set" from "this is the working set plus every size class the run has ever
// seen".  So the free list is CAPPED (bytes, blocks, and depth per size class)
// and evicts LEAST RECENTLY PARKED FIRST across size classes -- the class the
// steady state is not asking for is the class that goes -- and every eviction
// is counted and weighed in the MEM receipt.  Eviction needs a device free, and
// this header is pure host (below), so it calls a RECLAIMER the CUDA half
// installs; with no reclaimer installed the pool refuses to park rather than
// exceed its cap, which is bounded by construction and never leaks.
//
// WHY IT IS PURE HOST.  Same reason HostPinRegistry.h is: the bookkeeping has
// to be readable from EvaluatorServer.h, which is compiled by the host compiler
// and has no CUDA runtime.  So this header owns the map, the free list and the
// counters, and the functions that actually call `cudaMalloc`/`cudaFree` live
// in the .cu that includes it (guarded on `__CUDACC__` below).
//
// EXACTNESS (B0).  Allocation LIFETIME is not observable in a result.  A pooled
// block and a fresh `cudaMalloc` block are both uninitialised device memory --
// `cudaMalloc` makes no zeroing promise either -- so any code that could tell
// them apart is code that reads memory it never wrote, which is a defect the
// pool exposes rather than causes.  Eviction changes lifetime and nothing else,
// so it is exact for the same reason.  It is nevertheless GATED (see
// enabled()): `RASBERY_ARENA_PERSIST=1` turns reuse on, and with it unset the
// wrappers count and register but always go to the driver, which is the
// pre-WP10.6 behaviour instruction for instruction.  The counters are NOT
// gated -- a receipt that only reports when a flag is set is a receipt that
// cannot be compared across the two arms of the A/B that promotes the flag.
//
// WP10.8 -- THE COUNTER THAT WAS MISNAMED.  `arena_rebuilds` was incremented by
// three sites in CudaXsReconBackend.cu and one in GpuPhysicsArenaCuda.cu, and
// the 238 soak read +17 per generation in BOTH arms -- one per case, which is
// what a per-instance device block being re-laid-out under a shape change looks
// like, and not what an ARENA teardown looks like.  The quantity is real and
// worth counting; the name promised something else, and a reader who trusted it
// concluded that `RASBERY_ARENA_PERSIST=1` was failing to do what it never
// claimed.  So:
//
//   block_reshapes   a live per-instance device region freed and re-laid-out
//                    under a shape change.  This is what the old counter
//                    counted.  `noteBlockReshape()` is the ONLY spelling: the
//                    deprecated `noteArenaRebuild()` alias is gone, so no call
//                    site can put a per-case event behind an arena-shaped name
//                    again.  The three XsRecon regrow sites are the only
//                    callers; the physics arena needs none, because its
//                    stand-up and teardown are derived (below).
//   arena_standups   process-lifetime (`poolable=false`) regions REGISTERED --
//                    the nodal arena, the flat-XS library, the physics arena.
//   arena_teardowns  process-lifetime regions DEREGISTERED.  These are taken
//                    once for the process and handed back only at shutdown, so
//                    a MEM receipt printed BETWEEN generations must read 0, and
//                    a nonzero one is the arena teardown the old name promised
//                    to detect.  The receipt prints this as `arena_rebuilds`,
//                    which is now true to its name.
//
// Neither derived counter needs a call site to cooperate: `arena_standups` and
// `arena_teardowns` come from the `poolable` flag the registration already
// carries.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rasbery::gpu::blockpool {

/// One line of the WP10.6/WP10.8 half of `[RASBERY][EVALUATOR][MEM]`.
struct Stats {
    std::uint64_t device_allocs   = 0; ///< cudaMalloc calls that reached the driver
    std::uint64_t device_frees    = 0; ///< cudaFree calls that reached the driver
    std::uint64_t pool_hits       = 0; ///< allocations served from the free list
    std::uint64_t pool_parks      = 0; ///< frees absorbed by the free list
    /// WP10.8.  A live PER-INSTANCE region freed and re-laid-out under a shape
    /// change.  Formerly, and misleadingly, `arena_rebuilds`.
    std::uint64_t block_reshapes  = 0;
    std::uint64_t arena_standups  = 0; ///< process-lifetime regions registered
    std::uint64_t arena_teardowns = 0; ///< process-lifetime regions deregistered
    std::uint64_t pool_evictions  = 0; ///< parked blocks returned to the driver by policy
    std::uint64_t park_refusals   = 0; ///< frees the cap sent to the driver instead
    std::uint64_t bytes_live      = 0; ///< registered and handed to a caller
    std::uint64_t bytes_pooled    = 0; ///< parked in the free list, still ours
    std::uint64_t bytes_high_water = 0; ///< max(bytes_live + bytes_pooled)
    std::uint64_t bytes_evicted   = 0; ///< cumulative bytes evicted from the free list
    std::size_t   blocks_live     = 0;
    std::size_t   blocks_pooled   = 0;
    /// Distinct byte counts the free list currently holds.  A number that keeps
    /// climbing is a size key carrying something per case.
    std::size_t   size_classes    = 0;
    /// What the pool's OWN host bookkeeping weighs.  WP10.8: the 238 arm-B RSS
    /// finding could not be argued about because nothing weighed the suspect.
    std::uint64_t bookkeeping_bytes = 0;
};

/// The device free the pure-host half cannot call itself.  Installed by the
/// CUDA half (see `ensureBlockPoolReclaimer` below).
using Reclaimer = void (*)(void*);

namespace detail {

struct Registration {
    std::size_t bytes    = 0;
    bool        poolable = false;
};

/// A parked block and WHEN it was parked.  The stamp is what makes eviction
/// least-recently-parked-first ACROSS size classes: the class the steady state
/// stopped asking for is the class whose stamps stop moving.
struct Parked {
    void*         block = nullptr;
    std::uint64_t seq   = 0;
};

struct State {
    std::mutex                                           mutex;
    std::unordered_map<const void*, Registration>        live;
    std::unordered_map<std::size_t, std::vector<Parked>> parked;
    std::uint64_t                                        seq       = 0;
    Reclaimer                                            reclaimer = nullptr;
    Stats                                                stats;
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

/// An unsigned integer from the environment, or *fallback*.
inline std::uint64_t envUnsigned(const char* name, std::uint64_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return fallback;
    char*                    end   = nullptr;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw) return fallback;
    return static_cast<std::uint64_t>(value);
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

/// THE CAP, in bytes.  `RASBERY_ARENA_PERSIST_CAP_MB`, default 8 GiB -- above
/// any steady-state working set this campaign has measured (the 238 arm-B soak
/// settled at 6039 MB of TOTAL device footprint, of which the free list is a
/// fraction) and far below a board, so a healthy run never evicts and an
/// eviction in a receipt is a finding rather than noise.  0 disables the byte
/// cap; the block and depth caps still hold.
inline std::uint64_t capBytes() {
    static const std::uint64_t cap =
        detail::envUnsigned("RASBERY_ARENA_PERSIST_CAP_MB", 8192ull) * 1024ull * 1024ull;
    return cap;
}

/// THE CAP, in blocks.  A pool cannot be bounded by bytes alone: ten thousand
/// 4 KB blocks are 40 MB of device and ten thousand entries of bookkeeping.
inline std::uint64_t capBlocks() {
    static const std::uint64_t cap =
        detail::envUnsigned("RASBERY_ARENA_PERSIST_CAP_BLOCKS", 4096ull);
    return cap;
}

/// THE DEPTH OF ONE SIZE CLASS.  The arena width bounds how many blocks of one
/// size can be in flight at once, so a class deeper than that is a class
/// nothing will ask for.  Default 64 = four times the widest arena in use.
inline std::uint64_t capClassDepth() {
    static const std::uint64_t cap =
        detail::envUnsigned("RASBERY_ARENA_PERSIST_CLASS_DEPTH", 64ull);
    return cap;
}

/// Install the device free the free list uses to EVICT.  First call wins; the
/// CUDA half calls it on every allocation path, so a process that ever
/// allocated through the pool can always evict from it.
inline void setReclaimer(Reclaimer fn) {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.reclaimer == nullptr) s.reclaimer = fn;
}

inline bool hasReclaimer() {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.reclaimer != nullptr;
}

namespace detail {

/// Bookkeeping weight, from the containers themselves.  Approximate by
/// construction -- an unordered_map's node overhead is a libstdc++ detail --
/// and deliberately generous: the point is to be able to say "the pool is four
/// orders of magnitude too small to be the RSS finding", and a number that errs
/// high makes that claim harder to make, not easier.
inline std::uint64_t bookkeepingBytesLocked(const State& s) {
    constexpr std::uint64_t kNodeOverhead = 32; // pointer + hash + allocator slack
    std::uint64_t bytes = 0;
    bytes += static_cast<std::uint64_t>(s.live.size()) *
             (sizeof(const void*) + sizeof(Registration) + kNodeOverhead);
    bytes += static_cast<std::uint64_t>(s.live.bucket_count()) * sizeof(void*);
    bytes += static_cast<std::uint64_t>(s.parked.size()) *
             (sizeof(std::size_t) + sizeof(std::vector<Parked>) + kNodeOverhead);
    bytes += static_cast<std::uint64_t>(s.parked.bucket_count()) * sizeof(void*);
    for (const auto& entry : s.parked)
        bytes += static_cast<std::uint64_t>(entry.second.capacity()) * sizeof(Parked);
    return bytes;
}

/// Pull the least-recently-parked block out of the free list.  nullptr when the
/// list is empty.  Caller holds the lock and owns the returned block.
inline void* evictOneLocked(State& s, std::size_t& bytes_out) {
    auto          victim_class = s.parked.end();
    std::size_t   victim_index = 0;
    std::uint64_t oldest       = 0;
    bool          found        = false;
    for (auto it = s.parked.begin(); it != s.parked.end(); ++it) {
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            if (!found || it->second[i].seq < oldest) {
                oldest       = it->second[i].seq;
                victim_class = it;
                victim_index = i;
                found        = true;
            }
        }
    }
    if (!found) return nullptr;
    void* block = victim_class->second[victim_index].block;
    bytes_out   = victim_class->first;
    victim_class->second.erase(victim_class->second.begin() +
                               static_cast<std::ptrdiff_t>(victim_index));
    if (victim_class->second.empty()) s.parked.erase(victim_class);
    s.stats.bytes_pooled   -= bytes_out;
    s.stats.blocks_pooled  -= 1;
    s.stats.pool_evictions += 1;
    s.stats.bytes_evicted  += bytes_out;
    s.stats.device_frees   += 1; // it is about to reach the driver
    return block;
}

/// True when the free list is within every cap.
inline bool withinCapsLocked(const State& s) {
    if (capBytes() != 0 && s.stats.bytes_pooled > capBytes()) return false;
    if (capBlocks() != 0 &&
        static_cast<std::uint64_t>(s.stats.blocks_pooled) > capBlocks())
        return false;
    return true;
}

} // namespace detail

/// A block of EXACTLY `bytes`, or nullptr.  Never a bigger one.
inline void* take(std::size_t bytes) {
    if (!enabled() || bytes == 0) return nullptr;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.parked.find(bytes);
    if (it == s.parked.end() || it->second.empty()) return nullptr;
    void* block = it->second.back().block;
    it->second.pop_back();
    if (it->second.empty()) s.parked.erase(it);
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
    if (!poolable) s.stats.arena_standups += 1;
    s.live[block] = detail::Registration{bytes, poolable};
    detail::noteFootprintLocked(s);
}

/// True when the free list took ownership and the caller must NOT call
/// cudaFree.  False means "still yours to free", which is also the answer for
/// a pointer this pool never saw -- an unregistered block is freed exactly as
/// it was before, so a site this header does not know about is not broken by
/// it.
///
/// WP10.8.  Parking can now EVICT: a block that would push the free list past a
/// cap makes room by returning the least-recently-parked block to the driver
/// first.  The eviction runs with the lock RELEASED, because `cudaFree` is a
/// synchronising call and holding a mutex across it would serialise every lane.
inline bool give(const void* block) {
    if (block == nullptr) return true; // nothing to free, and cudaFree(nullptr) is a no-op
    detail::State&     s       = detail::state();
    std::vector<void*> victims;
    bool               parked  = false;
    Reclaimer          reclaim = nullptr;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.live.find(block);
        if (it == s.live.end()) return false;
        const detail::Registration reg = it->second;
        s.live.erase(it);
        s.stats.bytes_live  -= reg.bytes;
        s.stats.blocks_live -= 1;
        if (!reg.poolable) s.stats.arena_teardowns += 1;
        if (!enabled() || !reg.poolable || reg.bytes == 0) {
            s.stats.device_frees += 1;
            return false;
        }
        reclaim = s.reclaimer;
        // THE DEPTH RULE IS CHECKED BEFORE THE BLOCK GOES IN, not after: a
        // class already at depth is a class nothing is asking for, and adding
        // to it and then evicting the oldest member of some OTHER class would
        // let one runaway size push the working set out.
        const std::uint64_t depth = capClassDepth();
        auto&               slot  = s.parked[reg.bytes];
        if (depth != 0 && static_cast<std::uint64_t>(slot.size()) >= depth) {
            s.stats.park_refusals += 1;
            s.stats.device_frees  += 1;
            return false;
        }
        slot.push_back(detail::Parked{const_cast<void*>(block), ++s.seq});
        s.stats.pool_parks    += 1;
        s.stats.bytes_pooled  += reg.bytes;
        s.stats.blocks_pooled += 1;
        parked = true;
        detail::noteFootprintLocked(s);
        // OVER A CAP AND NO WAY TO GIVE ANYTHING BACK: undo the park and let
        // the caller free this one.  Refusing to grow is bounded; parking
        // anyway is the unbounded list WP10.8 exists to close.
        if (!detail::withinCapsLocked(s) && reclaim == nullptr) {
            auto& undo = s.parked[reg.bytes];
            undo.pop_back();
            if (undo.empty()) s.parked.erase(reg.bytes);
            s.stats.pool_parks    -= 1;
            s.stats.bytes_pooled  -= reg.bytes;
            s.stats.blocks_pooled -= 1;
            s.stats.park_refusals += 1;
            s.stats.device_frees  += 1;
            return false;
        }
        while (!detail::withinCapsLocked(s)) {
            std::size_t evicted_bytes = 0;
            void*       victim        = detail::evictOneLocked(s, evicted_bytes);
            if (victim == nullptr) break;
            victims.push_back(victim);
        }
    }
    for (void* victim : victims) reclaim(victim);
    return parked;
}

/// WP10.8.  A live PER-INSTANCE device region was freed and re-laid-out because
/// its SHAPE changed under it.  One per case is normal and expected -- it is
/// what a backend instance standing its blocks up at the first real geometry
/// looks like.  It is NOT an arena teardown; see the header note.
inline void noteBlockReshape() {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats.block_reshapes += 1;
}

inline Stats snapshot() {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Stats out = s.stats;
    out.size_classes      = s.parked.size();
    out.bookkeeping_bytes = detail::bookkeepingBytesLocked(s);
    return out;
}

/// THE NUMBER `arena_rebuilds` ALWAYS CLAIMED TO BE.  A process-lifetime region
/// is taken once and handed back only at shutdown, so between generations this
/// is 0; anything else is a teardown of the nodal arena, the flat-XS library or
/// the physics arena, which is exactly the event the VRAM sawtooth raised.
inline std::uint64_t arenaRebuilds(const Stats& s) { return s.arena_teardowns; }

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
        for (const detail::Parked& parked : entry.second) {
            freer(parked.block);
            s.stats.device_frees  += 1;
            s.stats.bytes_pooled  -= entry.first;
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
    s.seq   = 0;
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

/// WP10.8.  Hand the pure-host free list the device free it needs to EVICT.
/// Called from every allocation path so a process that ever pooled a block can
/// always give one back; `setReclaimer` ignores every call after the first.
inline void ensureBlockPoolReclaimer() {
    static const bool installed = [] {
        blockpool::setReclaimer([](void* block) { (void)cudaFree(block); });
        return true;
    }();
    (void)installed;
}

/// `cudaMalloc`, through the WP10.6 free list.  Byte-for-byte the same contract
/// as `cudaMalloc(pp, bytes)`: on success `*pp` is a device pointer to `bytes`
/// uninitialised bytes, on failure the return value is the driver's.
inline cudaError_t deviceBlockAlloc(void** pp, std::size_t bytes) {
    if (pp == nullptr) return cudaErrorInvalidValue;
    ensureBlockPoolReclaimer();
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
    ensureBlockPoolReclaimer();
    rasbery::AllocWindow _alloc_window("blockpool.cudaMalloc.once");
    const cudaError_t rc = cudaMalloc(pp, bytes);
    if (rc == cudaSuccess) blockpool::noteAllocated(*pp, bytes, /*poolable=*/false);
    return rc;
}

/// `cudaFree`, through the free list.  A parked block does not reach the
/// driver, which is the point: `cudaFree` is a synchronising call and the next
/// case is about to ask for the same size back.  An eviction inside `give()`
/// DOES reach the driver, through the reclaimer installed above, and is counted
/// in `device_frees` beside the frees this line makes.
inline cudaError_t deviceBlockFree(void* block) {
    ensureBlockPoolReclaimer();
    // Parked, or nullptr: no driver call, so no window, for the same reason.
    if (blockpool::give(block)) return cudaSuccess;
    rasbery::AllocWindow _alloc_window("blockpool.cudaFree");
    return cudaFree(block);
}

} // namespace rasbery::gpu

#endif // __CUDACC__
