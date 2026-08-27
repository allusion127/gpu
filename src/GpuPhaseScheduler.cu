// CUDA arm of the case-phase scheduler core -- Rev.7.1 Task 3, W0-scoped.
//
// TWO KERNELS, BOTH ONE-PASS, NEITHER PERSISTENT.
//
//   k_classify_case_phases   Level-1.  ONE CTA of 128 threads, one thread per
//                            slot, reading DeviceSlotPhase and nothing else.
//   k_refill_free_slots      Sec 8.2.  One thread per slot; free slots reset all
//                            four control structs and claim the next input.
//
// WHY ONE CTA.  Level-1's whole job is to read 64 x 32 B = 2 KiB and write ~4 KiB
// of queues.  A grid would need a global barrier to compute the cross-block
// prefix, and W0 priced that: c_barrier = 0.78 us against a 0.384 us kill
// threshold, which is also why there is no cooperative launch anywhere in this
// file and must never be one.  A single CTA does the prefix in shared memory
// for free.
//
// WHY THE COMPACTION IS BALLOT-AND-PREFIX AND NOT AN ATOMIC BUMP.  An
// atomicAdd-per-slot would produce a queue whose ORDER depends on warp
// scheduling.  Queue order is the order phase kernels touch slot arrays, so a
// nondeterministic queue makes the memory access pattern -- and any reduction
// written in queue order -- vary run to run.  The ballot gives each eligible
// slot its rank among the lower-numbered eligible slots of the same phase, so
// the queue is ascending by construction and identical to what
// gpuClassifySerial() in the header produces -- which is the reference
// test/gpu_phase_compaction.cpp checks exhaustively, without a GPU.
//
// The counters (active_count, free_count, phase totals) ARE order-insensitive
// sums, so those use shared-memory atomics; nothing here uses a float atomic.

#include "GpuPhaseScheduler.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace rasbery::gpu {

namespace {

constexpr int kClassifyBlock = 128;
constexpr int kClassifyWarps = kClassifyBlock / 32;

static_assert(kMaxSchedulerSlots <= kClassifyBlock,
              "Level-1 is one thread per slot in one CTA");

} // namespace

// ---------------------------------------------------------------------------
// Level-1 classify / compact
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(kClassifyBlock) void k_classify_case_phases(
    DeviceSlotPhase* __restrict__ phases, int slot_count,
    DevicePhaseQueues* __restrict__ out) {
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Per-phase, per-warp population counts, for the deterministic cross-warp
    // prefix.  17 phases x 4 warps of int is 272 B of shared.
    __shared__ int  s_warp_count[kDevicePhaseCount][kClassifyWarps];
    __shared__ int  s_active;
    __shared__ int  s_free;
    __shared__ unsigned int s_fault_flags;
    __shared__ unsigned int s_fault_slot;

    // 1. Clear the output.  Every padding entry is written, so a stale slot id
    //    from a previous epoch can never be read as live work.
    for (int i = tid; i < kDevicePhaseCount * kMaxSchedulerSlots; i += kClassifyBlock) {
        const int p = i / kMaxSchedulerSlots;
        const int k = i - p * kMaxSchedulerSlots;
        out->queue[p].slots[k] = kQueueEmptySlot;
    }
    for (int p = tid; p < kDevicePhaseCount; p += kClassifyBlock) {
        out->queue[p].count  = 0;
        out->queue[p].bucket = 0;
        out->phase_count[p]  = 0;
    }
    for (int i = tid; i < kDevicePhaseCount * kClassifyWarps; i += kClassifyBlock)
        s_warp_count[i / kClassifyWarps][i % kClassifyWarps] = 0;
    if (tid == 0) {
        s_active      = 0;
        s_free        = 0;
        s_fault_flags = kSchedFaultNone;
        s_fault_slot  = kSchedNoFaultSlot;
    }
    __syncthreads();

    // 2. Read the ONE struct Level-1 is allowed to read.
    const bool   in_range = tid < slot_count && tid < kMaxSchedulerSlots;
    std::uint8_t my_phase = static_cast<std::uint8_t>(DevicePhase::Empty);
    bool         eligible = false;

    if (in_range) {
        const DeviceSlotPhase p = phases[tid];
        my_phase                = p.phase;

        if (p.phase >= kDevicePhaseCount) {
            atomicOr(&s_fault_flags, kSchedFaultBadPhase);
            atomicMin(&s_fault_slot, static_cast<unsigned int>(tid));
        } else {
            const DevicePhase phase = static_cast<DevicePhase>(p.phase);
            if (!gpuPhaseIsSchedulable(phase)) {
                atomicAdd(&s_free, 1);
            } else if (slotActive(p)) {
                atomicAdd(&s_active, 1);
                // Sec 5.2, both fatal.  in_flight is checked first: a slot that
                // is being driven right now is a harder error than one that is
                // merely already queued, and reporting the softer one would
                // hide it.
                if (slotInFlight(p)) {
                    atomicOr(&s_fault_flags, kSchedFaultInFlightRequeue);
                    atomicMin(&s_fault_slot, static_cast<unsigned int>(tid));
                } else if (slotAlreadyQueued(p)) {
                    atomicOr(&s_fault_flags, kSchedFaultDuplicateQueue);
                    atomicMin(&s_fault_slot, static_cast<unsigned int>(tid));
                } else {
                    eligible = true;
                }
            }
        }
    }
    __syncthreads();

    // 3. Ballot per phase.  Each thread keeps only its own phase's rank, but the
    //    ballot has to be executed by the whole warp for every phase, so the
    //    loop is uniform and __ballot_sync's mask is the full warp.
    int my_rank_in_warp = 0;
    for (int p = 0; p < kDevicePhaseCount; ++p) {
        const unsigned int mask =
            __ballot_sync(0xffffffffu, eligible && static_cast<int>(my_phase) == p);
        if (lane == 0) s_warp_count[p][warp] = __popc(mask);
        if (eligible && static_cast<int>(my_phase) == p) {
            const unsigned int lower = (lane == 0) ? 0u : (0xffffffffu >> (32 - lane));
            my_rank_in_warp          = __popc(mask & lower);
        }
    }
    __syncthreads();

    // 4. Cross-warp prefix, summed in warp order, so the queue index of a slot
    //    depends only on the slot's number -- not on which warp finished first.
    if (eligible) {
        const int p      = static_cast<int>(my_phase);
        int       offset = 0;
        for (int w = 0; w < warp; ++w) offset += s_warp_count[p][w];
        const int index = offset + my_rank_in_warp;
        if (index < kMaxSchedulerSlots) {
            out->queue[p].slots[index] = tid;
        } else {
            atomicOr(&s_fault_flags, kSchedFaultSlotOverflow);
            atomicMin(&s_fault_slot, static_cast<unsigned int>(tid));
        }

        // Capture the epoch (Sec 5.2).  From here the entry is valid only while
        // the slot has not transitioned; nothing has to walk the queue to
        // invalidate it.
        phases[tid].queued_phase = my_phase;
        phases[tid].queued_epoch = phases[tid].state_epoch;
    }
    __syncthreads();

    // 5. Totals, buckets and the fixed-order selection, written by thread 0.
    if (tid == 0) {
        for (int p = 0; p < kDevicePhaseCount; ++p) {
            int total = 0;
            for (int w = 0; w < kClassifyWarps; ++w) total += s_warp_count[p][w];
            out->queue[p].count  = total;
            out->queue[p].bucket = gpuSelectBucket(total);
            out->phase_count[p]  = total;
        }
        out->active_count = s_active;
        out->free_count   = s_free;
        out->fault_flags  = s_fault_flags;
        out->fault_slot   = s_fault_slot;

        if (slot_count > kMaxSchedulerSlots) {
            out->fault_flags |= kSchedFaultSlotOverflow;
            if (out->fault_slot == kSchedNoFaultSlot)
                out->fault_slot = static_cast<unsigned int>(kMaxSchedulerSlots);
        }

        out->selected_phase  = gpuSelectPhase(out->phase_count);
        out->selected_count  = 0;
        out->selected_bucket = 0;
        if (out->selected_phase >= 0) {
            out->selected_count  = out->queue[out->selected_phase].count;
            out->selected_bucket = out->queue[out->selected_phase].bucket;
        }

        // CONSUME the fault.  Without this the bits were written and nobody
        // read them: the offending slot would keep running with two bodies
        // driving it, which is the corruption the check exists to stop.  One
        // slot dies (Sec 9.2), the rest of the fleet continues.
        if (gpuSchedulerFaultIsFatal(out->fault_flags) &&
            out->fault_slot < static_cast<unsigned int>(slot_count) &&
            out->fault_slot < static_cast<unsigned int>(kMaxSchedulerSlots))
            gpuMarkSlotFailed(phases[out->fault_slot], out->fault_flags);
    }
}

// ---------------------------------------------------------------------------
// Sec 8.2  Immediate slot refill
// ---------------------------------------------------------------------------

__global__ void k_refill_free_slots(DeviceSlotPhase* __restrict__ phases,
                                    DeviceSlotState* __restrict__ states,
                                    DeviceSearchState* __restrict__ searches,
                                    DeviceScheduleParams* __restrict__ params, int slot_count,
                                    const DeviceInputDescriptor* __restrict__ inputs,
                                    int input_count, int* __restrict__ next_input,
                                    GpuRefillCounters* __restrict__ counters) {
    const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (s >= slot_count) return;

    DeviceSlotPhase& p = phases[s];
    if (p.phase >= kDevicePhaseCount) return;
    const DevicePhase phase = static_cast<DevicePhase>(p.phase);

    // A slot is free when its tenant is finished (Done/Failed) or when it never
    // had one.  An Empty slot that still carries the active flag is mid-refill
    // from another pass and is left alone.
    const bool finished = phase == DevicePhase::Done || phase == DevicePhase::Failed;
    const bool vacant   = phase == DevicePhase::Empty && !slotActive(p);
    if (!finished && !vacant) return;

    // Sec 8.2 order: bump the epoch, reset ALL FOUR structs, then claim.
    // Resetting before the claim is what makes an exhausted claim safe -- the
    // previous tenant is gone either way, so a slot can never be left holding
    // half of one deck and half of another.
    //
    // An ALREADY-Empty slot is skipped: it was reset by a previous pass and is
    // already clean, so re-running four resets on it every epoch is work that
    // buys nothing.
    if (finished) {
        const std::uint32_t next_epoch = p.state_epoch + 1u;
        deviceSlotPhaseReset(p, next_epoch);
        deviceSlotStateReset(states[s]);
        deviceSearchStateReset(searches[s]);
        deviceScheduleParamsReset(params[s]);
    }

    // Read the cursor BEFORE touching it.  Once the inputs are exhausted every
    // later pass would otherwise issue one atomicAdd per free slot per epoch --
    // an unbounded cursor and an `exhausted` count that grows with the number
    // of PASSES rather than the number of starved slots.  This early return
    // makes the counter mean "slots that lost a race for the last inputs".
    if (*next_input >= input_count) return; // draining; the slot stays Empty

    const int claim = atomicAdd(next_input, 1);
    if (claim < 0 || claim >= input_count) {
        // claim < 0 is the cursor having wrapped, which only a runaway loop can
        // do; treat it exactly like exhaustion rather than indexing on it.
        atomicAdd(&counters->exhausted, 1u);
        return; // stays Empty and inactive; the fleet is draining
    }

    const DeviceInputDescriptor d = inputs[claim];
    p.input_id                    = d.input_id;
    p.job_id                      = d.job_id;
    p.flags                       = kSlotFlagActive | kSlotFlagInputReady;
    p.phase                       = static_cast<std::uint8_t>(DevicePhase::Import);

    states[s].schedule_index = d.schedule_index;
    states[s].statepoint     = d.statepoint;

    atomicAdd(&counters->refilled, 1u);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

namespace {

bool launchFailed(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return false;
    std::fprintf(stderr, "[RASBERY][GPU_SCHED][FAIL] {\"kernel\":\"%s\",\"error\":\"%s\"}\n", what,
                 cudaGetErrorString(e));
    return true;
}

} // namespace

bool gpuLaunchClassify(DeviceSlotPhase* phases, int slot_count, DevicePhaseQueues* queues,
                       GpuSchedulerStream stream) {
    if (phases == nullptr || queues == nullptr || slot_count <= 0) return false;
    k_classify_case_phases<<<1, kClassifyBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        phases, slot_count, queues);
    return !launchFailed("k_classify_case_phases");
}

bool gpuLaunchRefill(const GpuRefillArgs& a, GpuSchedulerStream stream) {
    if (a.phases == nullptr || a.states == nullptr || a.searches == nullptr ||
        a.params == nullptr || a.next_input == nullptr || a.counters == nullptr ||
        a.slot_count <= 0)
        return false;
    // A null descriptor array with a positive count would have the kernel index
    // nullptr the moment a slot won a claim.
    if (a.inputs == nullptr && a.input_count > 0) return false;
    const int block = 128;
    const int grid  = (a.slot_count + block - 1) / block;
    k_refill_free_slots<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        a.phases, a.states, a.searches, a.params, a.slot_count, a.inputs, a.input_count,
        a.next_input, a.counters);
    return !launchFailed("k_refill_free_slots");
}

bool gpuLaunchRefillThenClassify(const GpuRefillArgs& a, DevicePhaseQueues* queues,
                                 GpuSchedulerStream stream) {
    // One stream, in order.  Classify must see the slots refill just stamped
    // Active/Import; taking a single stream parameter is what makes that
    // structural rather than a convention someone has to remember.
    if (!gpuLaunchRefill(a, stream)) return false;
    return gpuLaunchClassify(a.phases, a.slot_count, queues, stream);
}

} // namespace rasbery::gpu
