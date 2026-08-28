#pragma once

// GPU CMFD pre/post kernels -- Rev.7.1 plan Task 5 / Sec 6.2, 6.3, 6.6, 6.12,
// 6.13.  Five phases of the outer, in the order Driver.h runs them:
//
//   enqueueUpdPsi              CMFD fission source            (before setls)
//   enqueueUpdDtil             surface diffusion coupling     (after cusping)
//   enqueueUpdJnet             net current from the operator  (after drive)
//   enqueueUpdDhat             CNCC correction + counters     (after nodal)
//   enqueueOuterConvergence    the flux convergence / stall state machine
//
// The arithmetic is in CmfdOuterKernel.h and is CLASS B0: bit-identical to the
// CPU loops, no transcendental anywhere, so a deviation is a bug and not a
// classification.  This file adds only the launch shape, the queue plumbing and
// the two things that CANNOT live in a pure body -- the counter reduction and
// the decision write-back.
//
// ---------------------------------------------------------------------------
// Sec 6.12  The dhat counters, and why they are not four atomicAdds
// ---------------------------------------------------------------------------
//
// CMFD::upddhat keeps four diagnostics: _dhat_total, _dhat_fsum_guard,
// _dhat_clamped (three integer counts) and _dhat_ratio_max (a maximum).  A
// naive port does one global atomic per (surface, group), which at APR1400 size
// is 4 x 26692 x 2 = 213k atomics on four addresses per outer -- serialised on
// the same cache lines, every outer, for the whole run.
//
// The three counts are order-insensitive SUMS, so they block-reduce in shared
// memory and pay ONE atomicAdd per block.  Integer, so the reduction order does
// not affect the value at all -- there is no determinism question to answer.
//
// `ratio_max` is a MAXIMUM over non-negative doubles, which is where the trick
// is: for x >= 0 the IEEE-754 bit pattern of a double, read as an unsigned
// 64-bit integer, is monotonically increasing in x.  So atomicMax on the bit
// pattern computes the maximum of the doubles exactly, with no float atomic and
// no CAS loop.  The precondition is |dhat|/|dtil| >= 0, which holds because both
// operands are magnitudes -- and the body returns -1 for "no contribution"
// rather than 0, so a surface with |dtil| == 0 cannot be mistaken for one whose
// ratio really was zero.
//
// ---------------------------------------------------------------------------
// Sec 6.13  The convergence kernel is ONE THREAD PER SLOT
// ---------------------------------------------------------------------------
//
// It is a scalar state machine over ~10 words of per-slot state.  Rev.7.1 Sec
// 6.17 makes the same call for the critical search and for the same reason: a
// subgroup mapping here would have 31 idle lanes and a reduction to write.
//
// It does NOT write the phase word itself.  It publishes a decision
// (CmfdOuterDecision) that Task 6/7's transition kernel consumes, because the
// phase transition is the scheduler's business (Sec 5.2): a phase kernel that
// writes `phase` races with classify, and one that writes queued_phase /
// queued_epoch re-validates the queue entry it is being run from.
//
// ---------------------------------------------------------------------------
// THE HALT GATE  (Rev.7.1 Task 9)
// ---------------------------------------------------------------------------
//
// Every kernel here takes an optional `const std::uint32_t* halt` indexed by
// SLOT and returns immediately when it is set.  It exists because a device outer
// SEGMENT (CudaOuterGraph.h) enqueues its whole budget of outers up front and
// never observes between them: without a gate, a segment that converged at outer
// 3 of 8 would still run outers 4..8 and move the trajectory, which is the one
// thing Sec 9.1 Class B0 on trajectory forbids.  This is the same shape
// CudaBICGBackend.cu's resident sweep already uses (`sweep_halt[m]`), and it is
// what Task 10's conditional WHILE predicate replaces.
//
// nullptr IS THE UNGATED PATH AND IT IS THE DEFAULT.  Every pre-Task-9 call site
// passes nothing, gets nullptr, and compiles to the branch it had before -- so
// the feature-off arm is byte-identical by construction rather than by
// comparison.
//
// THE CHECK GOES AFTER THE SLOT IS RESOLVED, never before.  `halt` is indexed by
// physical slot, and the slot only exists once gpuDispatchIsPadding has cleared
// the lane and queue.slots[logical] has been read.  A halt test on `logical`
// would index the wrong tenant at any width below the bucket.
//
// IT IS UNIFORM OVER A BLOCK, which k_cmfd_upd_dhat depends on: that kernel's
// counter reduction contains a __syncthreads(), so a non-uniform early return
// would leave part of the block waiting on a barrier nobody reaches.  `logical`
// is blockIdx.y there, so every thread of a block reads the same halt word and
// the whole block returns together.
//
// ---------------------------------------------------------------------------
// This header is includable WITHOUT nvcc: everything above the
// `#if defined(__CUDACC__)` line is CUDA-free, so test/cmfd_outer_replay.cpp
// can drive the same bodies and the same decision encoding on the host.

#include "CmfdOuterKernel.h"
#include "GpuPhaseScheduler.h"
#include "GpuPhysicsTypes.h"

#if defined(__CUDACC__)
    #include <cuda_runtime.h>
#endif

namespace rasbery::gpu {

/// Sec 7: surface and node kernels are 128-wide, the same as the nodal phases.
inline constexpr int kCmfdOuterBlock = 128;

/// Where each queued slot's outer arrays live.  ONE of these is built by the
/// host after reserve() and uploaded once, exactly like DeviceArenaView's slot
/// view table and for the same reason (Sec 4.1: fixed addresses are what makes
/// a captured graph's baked arguments stay valid).
///
/// It is separate from DeviceSlotView because the CMFD bodies use the HOST's
/// array layouts ([ls*ng + ig], [l*ng + ig]) while DeviceSlotView annotates
/// dtil as the transpose -- see the layout note in CmfdOuterKernel.h.  Keeping
/// the two apart makes that a visible decision for Task 6/7 instead of a silent
/// reinterpretation of the same pointer.
struct CmfdOuterSlotTable {
    const cmfd::CmfdOuterView* views; ///< [slot_count]
    int                        slot_count;
};

/// Accumulated dhat diagnostics for ONE slot, mirroring CMFD's four members.
/// `ratio_max_bits` is the IEEE bit pattern of the maximum ratio (see the
/// header note); read it back with cmfdRatioFromBits.
struct CmfdOuterCounters {
    unsigned long long total;
    unsigned long long fsum_guard;
    unsigned long long clamped;
    unsigned long long ratio_max_bits;
};

RASBERY_GPU_HD inline double cmfdRatioFromBits(unsigned long long bits) {
    // 0 is the bit pattern of +0.0, which is also the identity for a maximum
    // over non-negative doubles, so an untouched counter reads as 0.0.
    double d;
    const unsigned char* src = reinterpret_cast<const unsigned char*>(&bits);
    unsigned char*       dst = reinterpret_cast<unsigned char*>(&d);
    for (unsigned i = 0; i < sizeof(double); ++i) dst[i] = src[i];
    return d;
}

RASBERY_GPU_HD inline unsigned long long cmfdRatioToBits(double d) {
    unsigned long long       bits = 0;
    const unsigned char*     src  = reinterpret_cast<const unsigned char*>(&d);
    unsigned char*           dst  = reinterpret_cast<unsigned char*>(&bits);
    for (unsigned i = 0; i < sizeof(double); ++i) dst[i] = src[i];
    return bits;
}

/// One slot's outer decision, written by the convergence kernel and consumed by
/// Task 6/7's transition kernel.
///
/// `next_phase` is a DevicePhase value.  The kernel writes it; it does NOT write
/// DeviceSlotPhase::phase, and above all not queued_phase / queued_epoch, which
/// belong to classify (Sec 5.2).
struct CmfdOuterDecision {
    unsigned int next_phase;  ///< DevicePhase
    unsigned int escape;      ///< DeviceEscape
    int          flux_converged;
    int          xe_interim;
    int          stall_sample;
    int          warn_limit_cycle; ///< the host prints a WARN; the device counts it
};

/// The Sec 6.21 edge each CmfdOuterAction takes.  Written here, once, so the
/// kernel and test/cmfd_outer_replay.cpp cannot disagree about which
/// kPhaseTransitions edge a decision claims to be.
RASBERY_GPU_HD inline DevicePhase cmfdOuterActionPhase(cmfd::CmfdOuterAction a) {
    switch (a) {
        case cmfd::CmfdOuterAction::RequeueOuter:      return DevicePhase::Outer;
        case cmfd::CmfdOuterAction::Xenon:             return DevicePhase::Xenon;
        case cmfd::CmfdOuterAction::ThermalHydraulics: return DevicePhase::ThermalHydraulics;
        case cmfd::CmfdOuterAction::Search:            return DevicePhase::Search;
        case cmfd::CmfdOuterAction::Converged:         return DevicePhase::NormalizeFluxSign;
        case cmfd::CmfdOuterAction::Fatal:             return DevicePhase::Failed;
    }
    return DevicePhase::Failed;
}

/// CmfdOuterEscape -> DeviceEscape.  The two enumerations are deliberately
/// separate (a pure numerical body must not include the scheduler); this is the
/// single place they are joined, and the contract test pins the values.
RASBERY_GPU_HD inline DeviceEscape cmfdOuterEscapeCode(cmfd::CmfdOuterEscape e) {
    switch (e) {
        case cmfd::CmfdOuterEscape::None:                 return DeviceEscape::None;
        case cmfd::CmfdOuterEscape::FluxConverged:        return DeviceEscape::FluxConverged;
        case cmfd::CmfdOuterEscape::FluxLimitCycleSample: return DeviceEscape::FluxLimitCycleSample;
        case cmfd::CmfdOuterEscape::FluxStallFatal:       return DeviceEscape::FluxStallFatal;
    }
    return DeviceEscape::None;
}

/// Pack one outer result into the decision record.  Shared with the host replay
/// so the encoding is written once.
RASBERY_GPU_HD inline CmfdOuterDecision cmfdPackDecision(const cmfd::CmfdOuterResult& r) {
    CmfdOuterDecision d{};
    d.next_phase       = static_cast<unsigned int>(cmfdOuterActionPhase(r.action));
    d.escape           = static_cast<unsigned int>(cmfdOuterEscapeCode(r.escape));
    d.flux_converged   = r.flux_converged;
    d.xe_interim       = r.xe_interim;
    d.stall_sample     = r.stall_sample;
    d.warn_limit_cycle = r.warn_limit_cycle;
    return d;
}

/// Copy the six carried fields between DeviceSlotState and the body's small
/// state struct.  Two functions rather than one aliasing cast, because
/// DeviceSlotState is 128-byte aligned control memory whose layout is fixed by
/// Sec 3.2 and must not acquire a second definition here.
RASBERY_GPU_HD inline cmfd::CmfdOuterState cmfdLoadOuterState(const DeviceSlotState& s) {
    cmfd::CmfdOuterState o{};
    o.prev_inner         = s.previous_eigv;
    o.flux_stall         = s.flux_stall;
    o.stall_events       = s.stall_events;
    o.stall_sample_taken = s.stall_sample_taken;
    o.clean_iters        = s.clean_iters;
    o.xe_interim_count   = s.xe_interim_count;
    o.total_outer        = s.total_outer;
    return o;
}

RASBERY_GPU_HD inline void cmfdStoreOuterState(const cmfd::CmfdOuterState& o,
                                               DeviceSlotState& s) {
    s.previous_eigv      = o.prev_inner;
    s.flux_stall         = o.flux_stall;
    s.stall_events       = o.stall_events;
    s.stall_sample_taken = o.stall_sample_taken;
    s.clean_iters        = o.clean_iters;
    s.xe_interim_count   = o.xe_interim_count;
    s.total_outer        = o.total_outer;
}

#if defined(__CUDACC__)

// ---------------------------------------------------------------------------
// Phase kernels (nvcc only)
// ---------------------------------------------------------------------------

namespace detail {

/// Block-reduce three integer counts and one bit-pattern maximum, then pay ONE
/// atomic per block per quantity.  Sec 6.12.
///
/// SHARED MEMORY, NOT __shfl_down_sync.  A warp shuffle would be the obvious
/// spelling and is banned by constraint 35 (Rev.7.1): a warp intrinsic in a
/// shared header costs nothing today and costs a rewrite at Task 24/25, and the
/// difference between the two spellings is a habit rather than a design.  There
/// is no performance argument to weigh against that here -- this reduction runs
/// once per BLOCK of a pass that has already touched 128 surface/group elements,
/// so it is a rounding error on a kernel whose cost is the elementwise work.
/// When GpuSubgroup.h exists (Task 23), a wrapper may replace this; a raw
/// intrinsic may not.
///
/// 2.5 KiB of shared at block 128, which does not move occupancy on any target
/// in RASBERY_CUDA_ARCHITECTURES.
__device__ inline void cmfdReduceCounters(unsigned int total, unsigned int fsum_guard,
                                          unsigned int clamped, unsigned long long ratio_bits,
                                          CmfdOuterCounters* out) {
    static_assert((kCmfdOuterBlock & (kCmfdOuterBlock - 1)) == 0,
                  "the halving tree below needs a power-of-two block");

    __shared__ unsigned int       s_total[kCmfdOuterBlock];
    __shared__ unsigned int       s_guard[kCmfdOuterBlock];
    __shared__ unsigned int       s_clamp[kCmfdOuterBlock];
    __shared__ unsigned long long s_ratio[kCmfdOuterBlock];

    const unsigned tid = threadIdx.x;
    s_total[tid]       = total;
    s_guard[tid]       = fsum_guard;
    s_clamp[tid]       = clamped;
    s_ratio[tid]       = ratio_bits;
    __syncthreads();

    for (unsigned half = kCmfdOuterBlock / 2; half > 0; half >>= 1) {
        if (tid < half) {
            s_total[tid] += s_total[tid + half];
            s_guard[tid] += s_guard[tid + half];
            s_clamp[tid] += s_clamp[tid + half];
            if (s_ratio[tid + half] > s_ratio[tid]) s_ratio[tid] = s_ratio[tid + half];
        }
        __syncthreads();
    }

    if (tid != 0) return;
    if (s_total[0]) atomicAdd(&out->total, static_cast<unsigned long long>(s_total[0]));
    if (s_guard[0]) atomicAdd(&out->fsum_guard, static_cast<unsigned long long>(s_guard[0]));
    if (s_clamp[0]) atomicAdd(&out->clamped, static_cast<unsigned long long>(s_clamp[0]));
    if (s_ratio[0]) atomicMax(&out->ratio_max_bits, s_ratio[0]);
}

} // namespace detail

/// Sec 6.3.  One thread per (node) of one queued slot; the group loop is inside,
/// because the accumulation order is part of the contract.
__global__ __launch_bounds__(kCmfdOuterBlock) void k_cmfd_upd_psi(
    DeviceArenaView arena, DevicePhaseQueue queue, cmfd::CmfdGeometryView geom,
    CmfdOuterSlotTable table, unsigned long long forms, const std::uint32_t* halt) {
    const int logical = static_cast<int>(blockIdx.y);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    const int                slot = queue.slots[logical];
    if (halt != nullptr && halt[slot] != 0u) return;
    const cmfd::CmfdOuterView& v  = table.views[slot];
    (void)arena;

    const int l = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (l >= geom.nxyz) return;
    v.psi[l] = cmfd::cmfdUpdPsiNode(geom, v, l, forms);
}

/// Sec 6.2.  One thread per (surface, group).
__global__ __launch_bounds__(kCmfdOuterBlock) void k_cmfd_upd_dtil(
    DeviceArenaView arena, DevicePhaseQueue queue, cmfd::CmfdGeometryView geom,
    CmfdOuterSlotTable table, const std::uint32_t* halt) {
    const int logical = static_cast<int>(blockIdx.y);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    const int                  slot = queue.slots[logical];
    if (halt != nullptr && halt[slot] != 0u) return;
    const cmfd::CmfdOuterView& v    = table.views[slot];
    (void)arena;

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= geom.nsurf * geom.ng) return;
    const int ls = tid / geom.ng;
    const int ig = tid - ls * geom.ng;
    v.dtil[ls * geom.ng + ig] = cmfd::cmfdUpdDtilSurface(geom, v, ls, ig);
}

/// Sec 6.6.  One thread per (surface, group).
__global__ __launch_bounds__(kCmfdOuterBlock) void k_cmfd_upd_jnet(
    DeviceArenaView arena, DevicePhaseQueue queue, cmfd::CmfdGeometryView geom,
    CmfdOuterSlotTable table, unsigned long long forms, const std::uint32_t* halt) {
    const int logical = static_cast<int>(blockIdx.y);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    const int                  slot = queue.slots[logical];
    if (halt != nullptr && halt[slot] != 0u) return;
    const cmfd::CmfdOuterView& v    = table.views[slot];
    (void)arena;

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= geom.nsurf * geom.ng) return;
    const int ls = tid / geom.ng;
    const int ig = tid - ls * geom.ng;
    v.jnet[ls * geom.ng + ig] = cmfd::cmfdUpdJnetSurface(geom, v, ls, ig, forms);
}

/// Sec 6.6 + 6.12.  One thread per (surface, group), plus the counter reduction.
///
/// EVERY thread of the block reaches cmfdReduceCounters, including the ones past
/// the end of the surface list: the reduction contains a __syncthreads(), and a
/// thread that returned early would leave the rest of the block waiting on a
/// barrier it can never reach.  Out-of-range threads simply contribute zeros.
/// The two returns above it are UNIFORM over the block -- both test only
/// blockIdx.y -- so they take the whole block out together or not at all.
__global__ __launch_bounds__(kCmfdOuterBlock) void k_cmfd_upd_dhat(
    DeviceArenaView arena, DevicePhaseQueue queue, cmfd::CmfdGeometryView geom,
    CmfdOuterSlotTable table, bool clamp_enabled, CmfdOuterCounters* counters,
    unsigned long long forms, const std::uint32_t* halt) {
    const int logical = static_cast<int>(blockIdx.y);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    const int                  slot = queue.slots[logical];
    if (halt != nullptr && halt[slot] != 0u) return;
    const cmfd::CmfdOuterView& v    = table.views[slot];
    (void)arena;

    const int tid   = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = geom.nsurf * geom.ng;

    unsigned int       n_total = 0, n_guard = 0, n_clamp = 0;
    unsigned long long ratio_bits = 0;

    if (tid < total) {
        const int ls = tid / geom.ng;
        const int ig = tid - ls * geom.ng;
        const cmfd::CmfdDhatContribution c =
            cmfd::cmfdUpdDhatSurface(geom, v, ls, ig, clamp_enabled, forms);
        v.dhat[ls * geom.ng + ig] = c.dhat;
        n_total                   = static_cast<unsigned int>(c.counted);
        n_guard                   = static_cast<unsigned int>(c.fsum_guard);
        n_clamp                   = static_cast<unsigned int>(c.clamped);
        // ratio < 0 means "no contribution" (|dtil| == 0), and a negative double
        // must never reach the bit-pattern maximum: its sign bit would make it
        // the largest unsigned value there is.
        if (c.ratio >= 0.0) ratio_bits = cmfdRatioToBits(c.ratio);
    }

    if (counters != nullptr)
        detail::cmfdReduceCounters(n_total, n_guard, n_clamp, ratio_bits, &counters[slot]);
}

/// Sec 6.13.  ONE THREAD PER SLOT (Sec 6.17's mapping, same reasoning): the body
/// is a scalar state machine over ~10 words.
__global__ void k_cmfd_outer_convergence(DeviceArenaView arena, DevicePhaseQueue queue,
                                         const cmfd::CmfdOuterInputs* inputs,
                                         CmfdOuterDecision* decisions,
                                         const std::uint32_t* halt) {
    const int logical = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gpuDispatchIsPadding(logical, queue.count)) return;
    const int        slot = queue.slots[logical];
    if (halt != nullptr && halt[slot] != 0u) return;
    DeviceSlotState& s    = arena.states[slot];

    cmfd::CmfdOuterState       st = cmfdLoadOuterState(s);
    const cmfd::CmfdOuterResult r = cmfd::cmfdOuterConvergence(inputs[slot], st);
    cmfdStoreOuterState(st, s);
    // The eigenvalue the outer just produced is published here, not by the
    // solver: DeviceSlotState::eigv is what the next phase reads.
    s.eigv          = inputs[slot].eigv;
    s.flux_l2       = inputs[slot].residual;
    decisions[slot] = cmfdPackDecision(r);
}

// ---------------------------------------------------------------------------
// Host enqueue
// ---------------------------------------------------------------------------

inline dim3 cmfdOuterGrid(int elements, int bucket) {
    return dim3(static_cast<unsigned>((elements + kCmfdOuterBlock - 1) / kCmfdOuterBlock),
                static_cast<unsigned>(bucket), 1);
}

inline cudaError_t enqueueUpdPsi(const DeviceArenaView& arena, const DevicePhaseQueue& queue,
                                 const cmfd::CmfdGeometryView& geom,
                                 const CmfdOuterSlotTable& table, unsigned long long forms,
                                 cudaStream_t stream, const std::uint32_t* halt = nullptr) {
    if (queue.count <= 0) return cudaSuccess;
    k_cmfd_upd_psi<<<cmfdOuterGrid(geom.nxyz, queue.bucket), kCmfdOuterBlock, 0, stream>>>(
        arena, queue, geom, table, forms, halt);
    return cudaGetLastError();
}

inline cudaError_t enqueueUpdDtil(const DeviceArenaView& arena, const DevicePhaseQueue& queue,
                                  const cmfd::CmfdGeometryView& geom,
                                  const CmfdOuterSlotTable& table, unsigned long long forms,
                                  cudaStream_t stream, const std::uint32_t* halt = nullptr) {
    (void)forms; // upddtil has no contraction site; see CmfdOuterKernel.h
    if (queue.count <= 0) return cudaSuccess;
    k_cmfd_upd_dtil<<<cmfdOuterGrid(geom.nsurf * geom.ng, queue.bucket), kCmfdOuterBlock, 0,
                      stream>>>(arena, queue, geom, table, halt);
    return cudaGetLastError();
}

inline cudaError_t enqueueUpdJnet(const DeviceArenaView& arena, const DevicePhaseQueue& queue,
                                  const cmfd::CmfdGeometryView& geom,
                                  const CmfdOuterSlotTable& table, unsigned long long forms,
                                  cudaStream_t stream, const std::uint32_t* halt = nullptr) {
    if (queue.count <= 0) return cudaSuccess;
    k_cmfd_upd_jnet<<<cmfdOuterGrid(geom.nsurf * geom.ng, queue.bucket), kCmfdOuterBlock, 0,
                      stream>>>(arena, queue, geom, table, forms, halt);
    return cudaGetLastError();
}

inline cudaError_t enqueueUpdDhat(const DeviceArenaView& arena, const DevicePhaseQueue& queue,
                                  const cmfd::CmfdGeometryView& geom,
                                  const CmfdOuterSlotTable& table, unsigned long long forms,
                                  bool clamp_enabled, CmfdOuterCounters* counters,
                                  cudaStream_t stream, const std::uint32_t* halt = nullptr) {
    if (queue.count <= 0) return cudaSuccess;
    k_cmfd_upd_dhat<<<cmfdOuterGrid(geom.nsurf * geom.ng, queue.bucket), kCmfdOuterBlock, 0,
                      stream>>>(arena, queue, geom, table, clamp_enabled, counters, forms,
                                halt);
    return cudaGetLastError();
}

inline cudaError_t enqueueOuterConvergence(const DeviceArenaView& arena,
                                           const DevicePhaseQueue& queue,
                                           const cmfd::CmfdOuterInputs* inputs,
                                           CmfdOuterDecision* decisions, cudaStream_t stream,
                                           const std::uint32_t* halt = nullptr) {
    if (queue.count <= 0) return cudaSuccess;
    const int grid = (queue.bucket + kCmfdOuterBlock - 1) / kCmfdOuterBlock;
    k_cmfd_outer_convergence<<<grid, kCmfdOuterBlock, 0, stream>>>(arena, queue, inputs,
                                                                   decisions, halt);
    return cudaGetLastError();
}

#endif // __CUDACC__

} // namespace rasbery::gpu
