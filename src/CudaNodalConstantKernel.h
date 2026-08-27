#pragma once

// GPU Nodal::updateConstant -- Rev.7.1 plan Task 4 / Sec 6.1.
//
// WHAT MOVES.  The nine SENM coefficient arrays (eta1, eta2, m260, m251, m253,
// m262, m264, diagDI, diagD) stop being host-computed and uploaded on a
// generation change and become a device phase.  Nothing about the ARITHMETIC
// moves: the body is rasbery::nodal::nodalConstantCoefficients() from
// NodalConstantKernel.h, the same one Nodal::updateConstant calls, and this
// header must never grow a second copy of those formulas.  The contract test
// greps for that (a `kp2` or a `sinhkp` appearing here is a duplicated formula).
//
// CLASSIFICATION: N1, NOT B0.  Task 4 Step 0's B0-rescue spike measured device
// sqrt/exp against glibc 2.39 on sm_61 (test/nodal_constant_exp_probe.cu):
// sqrt matched on all 4,000,000 swept arguments; exp differed on 3.34% of them
// (5.34% inside the physical kp2 band), always by exactly 1 ulp.  So the rescue
// FAILED, this phase is a trajectory-changing transition, and Gate A/B plus the
// v3 freeze stay on Task 22.  The full reasoning is in NodalConstantKernel.h.
// The obligation this leaves behind is run-to-run BIT determinism, which is why
// nothing below reduces, atomics or races: every output element is written by
// exactly one thread, from inputs no other thread writes.
//
// THE TU IS BUILT WITH --fmad=false.  Everything outside exp/sqrt is IEEE basic
// arithmetic, and --fmad=false is what stops nvcc from contracting the
// multiply-adds inside nodalConstantCoefficients differently than gcc did.
// Without it the N1 deviation would not be confined to exp and the replay gate
// could not tell a library difference from a contraction difference.
//
// ---------------------------------------------------------------------------
// Sec 6.1 thread mapping
// ---------------------------------------------------------------------------
//
//     thread   = (slot, node, group)      grid.y = the dispatch BUCKET
//     block    = 128
//     grid.x   = ceil(nxyz * ng / 128)
//     grid.y   = queue.bucket             one lane per queued slot
//     the direction loop X/Y/Z is INSIDE the thread
//
// WHY THE DIRECTION LOOP STAYS INSIDE.  The three directions of one (node,
// group) share xsrf, xsdf and the whole `unchanged` decision; they differ only
// in hmesh.  Making direction a fourth thread axis would re-load the two cross
// sections three times and re-evaluate the early-out three times, for three
// stores that are already contiguous in `lkd`.  It would also triple the number
// of threads that have to agree about a node-scoped predicate.
//
// ---------------------------------------------------------------------------
// The early-out is NODE-scoped, and that is not an optimisation
// ---------------------------------------------------------------------------
//
// Nodal::updateConstant (src/Nodal.cpp:122-167) tests BOTH groups of a node
// against the cached _constant_xsrf/_constant_xsdf and returns early only when
// BOTH are unchanged; if either group moved it rewrites every group and every
// direction.  A per-(node,group) early-out is therefore NOT the same function:
// on a node where group 0 moved and group 1 did not, the host rewrites group 1
// as well (to the same values, so the arrays agree) -- but it also updates
// _constant_xsdf for both, and it returns `true`, which is what advances
// Nodal's _const_generation.  So each thread evaluates the whole node's
// predicate, reading 2*ng cache entries instead of 2.  Four extra loads to keep
// the function the same function.
//
// ---------------------------------------------------------------------------
// TWO KERNELS, and the second one is not optional
// ---------------------------------------------------------------------------
//
// The compute kernel READS the xsrf/xsdf cache (that is the early-out) and the
// host function also WRITES it (Nodal.cpp:163-166).  Doing both in one kernel
// is a race between the two group threads of a node: thread (lk, 0) writes
// cache[lk*ng+0] while thread (lk, 1) is still evaluating the node-scoped
// predicate over BOTH groups.  On a node where only group 1 moved, thread 1 can
// then see a cache that already agrees on group 0 and agrees on group 1 (it did
// not move), conclude "unchanged", and skip -- leaving group 1's coefficients
// stale.  Caught by test/nodal_constant_gpu_replay.cpp's "one xsdf on one node"
// scenario, which is precisely a node where one group moved and the other did
// not; it reported one wrong decision out of 8192 before the split.
//
// The same split is needed for `nodal_constant_generation`, for the same class
// of reason: stamping it from inside the compute kernel publishes it from
// whichever block finished first, while other blocks are still writing, so the
// next phase could read a stale coefficient behind a fresh generation.  There is
// no grid-wide barrier to fix either (W0: c_barrier = 0.78 us against a 0.384 us
// kill threshold, constraint 17).
//
// So: kernel 1 computes, kernel 2 publishes -- the cache AND the generation --
// on the SAME stream.  Stream order is the barrier, the publish costs one
// c_dispatch = 0.783 us for the whole bucket, and enqueueNodalUpdateConstant
// issues both so a caller cannot get the order wrong.
//
// The publish writes the cache UNCONDITIONALLY, including for nodes that took
// the early-out.  That is not a shortcut: "unchanged" is defined as cache ==
// xs for every group of the node, so on those nodes the write is a copy of a
// value onto itself and the final state is identical to the host's, which
// writes only on changed nodes.

// ---------------------------------------------------------------------------
// This header is includable WITHOUT nvcc
// ---------------------------------------------------------------------------
//
// Everything above the `#if defined(__CUDACC__)` line is CUDA-free: the index
// helpers and nodalConstantUpdateThread(), which is the whole per-thread body.
// The __global__ below is a five-line wrapper around it.
//
// That split is what lets test/nodal_constant_gpu_replay.cpp -- a plain .cpp,
// no CUDA, no device -- drive THE SAME body the kernel runs and score it
// bit-for-bit against a verbatim quotation of Nodal::updateConstant.  A replay
// that re-typed the mapping would be testing a second transcription instead of
// the shipped one, and the indexing (which of the nine packed arrays is diagD,
// which is diagDI) is exactly where that goes wrong silently.

#include "GpuPhaseScheduler.h"
#include "GpuPhysicsTypes.h"
#include "NodalConstantKernel.h"

#if defined(__CUDACC__)
    #include <cuda_runtime.h>
#endif

namespace rasbery::gpu {

/// Sec 7: 128 threads per block for the (node, group) axis.
inline constexpr int kNodalConstantBlock = 128;

/// Packing of DeviceSlotView::nodal_const, which is nine arrays of
/// [nxyz*NDIRMAX*ng] laid end to end.  THE ORDER IS THE ARENA'S
/// (GpuPhysicsArenaLayout.h, SlotRegion::NodalConst): diagDI comes BEFORE
/// diagD.  Reading it the other way round is silent -- both are finite, both
/// are per-(node,dir,group), and the answer is simply wrong -- so the order
/// lives in one named enum that the layout gate and this kernel share.
enum NodalConstSlot : int {
    kNcEta1 = 0,
    kNcEta2,
    kNcM260,
    kNcM251,
    kNcM253,
    kNcM262,
    kNcM264,
    kNcDiagDI,
    kNcDiagD,
    kNcCount
};

/// Packing of DeviceSlotView::constant_xs: two arrays of [nxyz*ng], the
/// device-resident twins of Nodal's _constant_xsrf / _constant_xsdf caches.
enum ConstantXsSlot : int { kCxXsrf = 0, kCxXsdf = 1, kCxCount };

/// One (node, direction, group) index into a nodal_const sub-array, matching
/// Nodal.cpp's `#define eta1(ig, lkd) (_eta1[(lkd) * _ng + ig])` with
/// lkd = lk*NDIRMAX + idir.
RASBERY_GPU_HD inline long long nodalConstIndex(int lk, int idir, int ig, int ng) {
    return (static_cast<long long>(lk) * kDevNdirMax + idir) * ng + ig;
}

/// xsrf / xsdf out of the live macroscopic block, which is NXS packed scalar
/// slots each laid out [ig*nxyz + l] (DeviceSlotView::xs).
RASBERY_GPU_HD inline long long macroXsIndex(int xt, int ig, int l, int ng, int nxyz) {
    return (static_cast<long long>(xt) * ng + ig) * nxyz + l;
}

/// Stride between two of the nine packed nodal_const sub-arrays.
RASBERY_GPU_HD inline long long nodalConstStride(int nxyz, int ng) {
    return static_cast<long long>(nxyz) * kDevNdirMax * ng;
}

// ---------------------------------------------------------------------------
// The per-thread body -- the whole phase, CUDA-free
// ---------------------------------------------------------------------------

/// One thread's work: node `lk`, group `ig`, all three directions.
///
/// Returns true when the node was recomputed, false when the node-scoped
/// early-out fired -- which is exactly Nodal::updateConstant's return value,
/// and the reason the host replay can score this against the CPU function
/// including its "unchanged" behaviour.
///
/// Writes: `v.nodal_const` and NOTHING else.  It READS `v.constant_xs` and
/// must not write it -- see the two-kernel note in the file header; the cache
/// update belongs to the publish pass, because a sibling group thread of the
/// same node is still reading it here.  It does not touch DeviceSlotPhase, and
/// above all not queued_phase/queued_epoch, which belong to classify (Sec 5.2):
/// a phase kernel that stamps them re-validates the queue entry it is currently
/// consuming and the slot is queued twice on the next epoch.
RASBERY_GPU_HD inline bool nodalConstantUpdateThread(const DeviceSlotView& v,
                                                     const DeviceGeometryView& geom, int lk,
                                                     int ig) {
    const int ng   = v.ng;
    const int nxyz = v.nxyz;

    double* const constant_xsrf =
        v.constant_xs + static_cast<long long>(kCxXsrf) * nxyz * ng;
    double* const constant_xsdf =
        v.constant_xs + static_cast<long long>(kCxXsdf) * nxyz * ng;

    // The node-scoped early-out, evaluated identically by every group thread of
    // the node.  See the header comment: a per-group test is a different
    // function, because the host rewrites EVERY group of a node on which ANY
    // group moved.
    bool unchanged = true;
    for (int g = 0; g < ng; ++g) {
        const double r = v.xs[macroXsIndex(kXtXsrf, g, lk, ng, nxyz)];
        const double d = v.xs[macroXsIndex(kXtXsdf, g, lk, ng, nxyz)];
        unchanged = unchanged && constant_xsrf[lk * ng + g] == r &&
                    constant_xsdf[lk * ng + g] == d;
    }
    if (unchanged) return false;

    const double xsrf = v.xs[macroXsIndex(kXtXsrf, ig, lk, ng, nxyz)];
    const double xsdf = v.xs[macroXsIndex(kXtXsdf, ig, lk, ng, nxyz)];

    // Direction loop X/Y/Z, inside the thread.
    const long long stride = nodalConstStride(nxyz, ng);
    for (int idir = 0; idir < kDevNdirMax; ++idir) {
        const double hmesh = geom.hmesh[static_cast<long long>(lk) * kDevNdirMax + idir];

        // The one and only arithmetic site: the shared pure body.  No formula
        // is spelled out in this file, by contract.
        const nodal::NodalConstantCoefficients c =
            nodal::nodalConstantCoefficients(xsrf, xsdf, hmesh);

        const long long idx = nodalConstIndex(lk, idir, ig, ng);
        v.nodal_const[kNcEta1 * stride + idx]   = c.eta1;
        v.nodal_const[kNcEta2 * stride + idx]   = c.eta2;
        v.nodal_const[kNcM260 * stride + idx]   = c.m260;
        v.nodal_const[kNcM251 * stride + idx]   = c.m251;
        v.nodal_const[kNcM253 * stride + idx]   = c.m253;
        v.nodal_const[kNcM262 * stride + idx]   = c.m262;
        v.nodal_const[kNcM264 * stride + idx]   = c.m264;
        v.nodal_const[kNcDiagDI * stride + idx] = c.diagDI;
        v.nodal_const[kNcDiagD * stride + idx]  = c.diagD;
    }
    return true;
}

/// The publish half: refresh one (node, group)'s xsrf/xsdf cache entry, which
/// is what Nodal.cpp:163-166 does at the end of a recomputed node.
RASBERY_GPU_HD inline void nodalConstantPublishThread(const DeviceSlotView& v, int lk,
                                                      int ig) {
    const int ng   = v.ng;
    const int nxyz = v.nxyz;
    v.constant_xs[static_cast<long long>(kCxXsrf) * nxyz * ng + lk * ng + ig] =
        v.xs[macroXsIndex(kXtXsrf, ig, lk, ng, nxyz)];
    v.constant_xs[static_cast<long long>(kCxXsdf) * nxyz * ng + lk * ng + ig] =
        v.xs[macroXsIndex(kXtXsdf, ig, lk, ng, nxyz)];
}

/// The slot-level gate, in one place so the kernel and the replay agree: the
/// nine arrays are current when the constant generation has caught up with the
/// material generation.
RASBERY_GPU_HD inline bool nodalConstantSlotIsCurrent(const DeviceSlotState& st) {
    return st.nodal_constant_generation == st.material_generation;
}

#if defined(__CUDACC__)

// ---------------------------------------------------------------------------
// Phase kernels + host enqueue (nvcc only)
// ---------------------------------------------------------------------------

/// Sec 6.1.  One thread per (node, group) of one queued slot.
__global__ __launch_bounds__(kNodalConstantBlock) void k_nodal_update_constant(
    DeviceArenaView arena, DevicePhaseQueue queue, DeviceGeometryView geom) {
    // 1. The dispatch is at the bucket width; padding lanes must not read the
    //    queue at all (the value there is kQueueEmptySlot).
    const int logical = static_cast<int>(blockIdx.y);
    if (gpuDispatchIsPadding(logical, queue.count)) return;

    // 2. Queue entry -> physical slot -> the slot's view.  blockIdx is never a
    //    slot id: the queue is what maps a lane to a tenant.
    const int             slot = queue.slots[logical];
    const DeviceSlotView& v    = arena.slotView(slot);

    // 3. Slot-level generation gate: when no material has moved since the
    //    coefficients were last built, the whole (node, group) grid returns.
    if (nodalConstantSlotIsCurrent(arena.states[slot])) return;

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= v.nxyz * v.ng) return;
    const int lk = tid / v.ng;
    const int ig = tid - lk * v.ng;

    nodalConstantUpdateThread(v, geom, lk, ig);
}

/// Publish pass 1: the xsrf/xsdf cache, AFTER the compute kernel has finished
/// on the same stream.  Same (node, group) x bucket grid as the compute kernel.
__global__ __launch_bounds__(kNodalConstantBlock) void k_nodal_constant_publish_cache(
    DeviceArenaView arena, DevicePhaseQueue queue) {
    const int logical = static_cast<int>(blockIdx.y);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    const int             slot = queue.slots[logical];
    const DeviceSlotView& v    = arena.slotView(slot);

    if (nodalConstantSlotIsCurrent(arena.states[slot])) return;

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= v.nxyz * v.ng) return;
    const int lk = tid / v.ng;
    const int ig = tid - lk * v.ng;
    nodalConstantPublishThread(v, lk, ig);
}

/// Publish pass 2: the generation, one thread per lane.
///
/// It is a THIRD launch and not the tail of the cache kernel for a reason that
/// is easy to get wrong: the generation is exactly what the cache kernel's own
/// gate reads.  Stamping it from inside that kernel lets a block that has not
/// yet reached its gate observe the new value and skip its cache write, so the
/// cache is left half-updated -- the same shape of bug as the compute/cache
/// split above, one level up.  A separate launch makes the gate read and the
/// stamp unambiguously ordered by the stream.
__global__ void k_nodal_constant_publish_generation(DeviceArenaView arena,
                                                    DevicePhaseQueue queue) {
    const int logical = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    DeviceSlotState& st          = arena.states[queue.slots[logical]];
    st.nodal_constant_generation = st.material_generation;
}

/// Grid.x for the (node, group) axis at the campaign's block width.
inline int nodalConstantGridX(int nxyz, int ng) {
    const int total = nxyz * ng;
    return (total + kNodalConstantBlock - 1) / kNodalConstantBlock;
}

/// Issue the phase for one classified queue.
///
/// `queue` is passed BY VALUE into both kernels, so the caller may reuse its
/// storage; `arena` and `geom` are the fixed-address views the arena published
/// after reserve().  Both launches go on `stream` in order -- that ordering is
/// the barrier that lets the publish kernel see the compute kernel's writes,
/// and it is why this function issues both rather than exposing them
/// separately.
///
/// `nxyz` and `ng` are the cohort shape (every slot in a cohort shares the
/// geometry, Sec 3.3), so one grid covers every lane.
inline cudaError_t enqueueNodalUpdateConstant(const DeviceArenaView& arena,
                                              const DevicePhaseQueue& queue,
                                              const DeviceGeometryView& geom, int nxyz,
                                              int ng, cudaStream_t stream) {
    if (queue.count <= 0) return cudaSuccess;

    const dim3 block(kNodalConstantBlock, 1, 1);
    const dim3 grid(static_cast<unsigned>(nodalConstantGridX(nxyz, ng)),
                    static_cast<unsigned>(queue.bucket), 1);
    k_nodal_update_constant<<<grid, block, 0, stream>>>(arena, queue, geom);
    if (const cudaError_t rc = cudaGetLastError(); rc != cudaSuccess) return rc;

    k_nodal_constant_publish_cache<<<grid, block, 0, stream>>>(arena, queue);
    if (const cudaError_t rc = cudaGetLastError(); rc != cudaSuccess) return rc;

    const int gen_grid = (queue.bucket + kNodalConstantBlock - 1) / kNodalConstantBlock;
    k_nodal_constant_publish_generation<<<gen_grid, kNodalConstantBlock, 0, stream>>>(arena,
                                                                                      queue);
    return cudaGetLastError();
}

#endif // __CUDACC__

} // namespace rasbery::gpu
