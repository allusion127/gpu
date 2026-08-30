// CUDA arm of the device outer segment -- Rev.7.1 plan Task 9.
//
// The state machine, the plan and the two kernels are in CudaOuterGraph.h,
// where the pure half is host-runnable.  What is here is the part that needs a
// device: the scratch, the one stream, the enqueue sequence, and the single
// synchronisation at the end.
//
// ---------------------------------------------------------------------------
// ONE STREAM, ONE SYNC.  THAT IS THE ENTIRE PERFORMANCE CLAIM.
// ---------------------------------------------------------------------------
//
// runSegment issues `budget` copies of the body back to back on ONE stream and
// calls cudaStreamSynchronize exactly ONCE, after the last one.  There is no
// per-outer D2H, no per-outer event, no per-outer query.  Stream order is the
// only synchronisation inside a segment, and it is sufficient: every step reads
// what the previous step wrote, on the same stream, so the ordering the CMFD
// outer needs is the ordering the stream already guarantees.
//
// The counter that proves it is `host_outer_observations`: it counts SEGMENT
// EXITS, not outers.  With a budget of 8 and a solve that takes 600 outers, v2
// paid 600 host rendezvous and this pays at most 75 -- and the M64 campaign said
// the rendezvous, not the kernels, is what the wall was made of.
//
// ---------------------------------------------------------------------------
// WHY EVERY SEGMENT REFUSES TODAY, AND WHY THAT IS THE CORRECT STATE
// ---------------------------------------------------------------------------
//
// Two of the eight body steps -- the resident CMFD sweep and the nodal drive --
// have no stream-ordered enqueue in this tree yet.  Both exist only as
// host-driven calls that rendezvous, launch, drain and copy back
// (CudaBatchArena::solveSweeps, XsReconBackend::solveNodal).  Calling either
// inside a segment would reinstate the per-outer round trip, so the runner
// declares them as hooks and REFUSES to run a segment until both are supplied.
//
// GpuPhysicsArena is likewise not reserved on the production path yet (nothing
// outside the Task 2 gate calls reserve()), so `NoArena` refuses first.
//
// A refusal is not a silent fallback: it is counted, named, and printed in the
// [RASBERY][OUTER_GPU] receipt -- as `idle_reason` when nothing ran at all, so a
// deck that never reaches the delegation still says why.  The failure mode of an
// opt-in fast path is that it never engages and the A/B measures the slow path
// twice; the receipt is what makes that impossible to miss.
//
// ---------------------------------------------------------------------------
// WHAT IS ACTUALLY IN THE WAY, MEASURED  (the standing work item)
// ---------------------------------------------------------------------------
//
// An audit of 8be6bee found that the Task 4/5/7 device bodies have NO
// production caller: the enqueue block below is their only one in src/, and it
// is unreachable for the reasons above.  The receipt now MEASURES that -- a
// two-statepoint i-SMR CY01 run with RASBERY_GPU_OUTER=1 prints
//
//     "device_outers":0, "idle_reason":"no_runner",
//     "host_body_calls":{"updpsi":207,"updjnet":207,"upddhat":207,
//                        "upddtil":3,"nodal_constants":207}
//
// i.e. the host ran every one of the 207 outer bodies.  Closing that is a chain
// of four dependencies, and each link is somebody's task rather than an
// oversight here:
//
//  (1) NO ARENA EXISTS AT RUNTIME.  GpuPhysicsArena::reserve()
//      (GpuPhysicsArenaCuda.cu:79) has zero callers in src/ and test/, so there
//      is no DeviceSlotView, no fixed per-slot addresses, and no
//      CmfdOuterSlotTable to build.  That is Task 2/18's lifecycle, not this
//      file's -- and it is what NoArena names.
//
//      NOTE, for whoever picks it up: the four BODY kernels
//      (k_cmfd_upd_psi/dtil/jnet/dhat) take the arena and `(void)arena` it --
//      only k_cmfd_outer_convergence reads arena.states.  So a bridged wiring
//      that only wants the arithmetic needs a slot table and a queue, not a
//      reserved arena.
//
//  (2) THE SWEEP READS THE HOST ARRAYS.  BICGCMFD::driveDeviceSweeps fills
//      CmfdSweepIO with the raw host pointers (BICGCMFD.cpp:382-386:
//      io.dtil = _dtil, io.dhat = _dhat, io.psi = _psi) and CudaBICGBackend's
//      issueSweepUploads pushes dhat unconditionally every outer
//      (CudaBICGBackend.cu:2918-2919, surface_group_count doubles) because
//      "dhat changes after every nodal correction".  So a device-resident dhat
//      does not remove that H2D -- pointing the sweep at the device buffer
//      does, and that is the sweep/arena refactor, deliberately NOT touched
//      here.  The same holds for psi, which is in/out across
//      issueSweepUploads:2944 and issueSweepDownloads:2967.
//
//  (3) TWO LAYOUTS.  CmfdOuterView::flux is node-major [l*ng+ig]
//      (Geometry::Phif), while BatchCore::phi and the xs mirrors are
//      group-major [ig*nxyz+l].  dtil/dhat/jnet agree at [ls*ng+ig] on both
//      sides.  Whoever binds the view has to choose a transpose site rather
//      than alias the sweep's phi.
//
//  (4) TWO PSIs.  DeviceSlotView carries both `psi` (SlotRegion::Psi) and
//      `cmfd_psi` (SlotRegion::CmfdPsi).  CmfdOuterView::psi is the CMFD
//      fission source, so it binds to cmfd_psi.  Getting this wrong is silent.
//
// The nodal constants have a fifth, separate blocker:
// nodalConstantSlotIsCurrent gates on
// `nodal_constant_generation == material_generation`, and material_generation is
// one of the eight SPECULATIVE counters nothing on the host ever bumps
// (GpuSlotControl.h:280-287) -- so the device cache would read "current"
// forever.  That is Task 13/18's, and it is why enqueueNodalUpdateConstant is
// issued from the prologue here but nothing calls the prologue yet.

#include "CudaOuterGraph.h"

// Rev.7.1 Task 10 part 3: the sweep accumulator the host-free arm allocates and
// hands to every enqueued drive of a segment lives in the CMFD backend's
// vocabulary (CmfdSweepProbeSink::Accum), so the runner has to see it.
#include "CudaBICGBackend.h"

#include "CudaTransferMirror.h"
#include "GpuCaptureArbiter.h"
#include "GpuGraphSplice.h"
#include "GpuOuterWhile.h"
#include "GpuPhysicsArena.h"
#include "OuterTrace.h"
#include "XferLedger.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

// Forward-declared rather than included: the receipt needs the batch width to name
// the idle reason, and CudaBICGBackend.h would drag the whole solver surface into a
// file that only wants one int.  The declaration matches CudaBICGBackend.h:336.
namespace rasbery {
int rasberyBatchWidth();
} // namespace rasbery

namespace rasbery::gpu {

namespace {

/// Same truthiness rule as every other gate in this tree (rasberyGpuNodalFull,
/// canonicalSharedStateEnabled): present and not one of the false spellings.
bool envFlagOn(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string s(v);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
             s == "FALSE");
}

// ---------------------------------------------------------------------------
// Process-wide counters
// ---------------------------------------------------------------------------
//
// RELAXED ATOMICS, NOT A MUTEX.  A refusal is counted once per outer of every
// worker that has the feature armed, so at APR1400 outer counts this is a
// ~68k/statepoint path; a lock there would be a measurable tax on the arm that
// is not even using the device.  Nothing reads a counter to make a decision --
// they exist only to be printed at the end -- so relaxed ordering is exactly the
// guarantee needed and no more.

struct AtomicCounters {
    std::atomic<std::uint64_t> segment_launches{0};
    std::atomic<std::uint64_t> device_outers{0};
    std::atomic<std::uint64_t> host_outer_observations{0};
    std::atomic<std::uint64_t> budget_exits{0};
    std::atomic<std::uint64_t> halted_outer_launches{0};
    std::atomic<std::uint64_t> jnet_bridge_bytes{0};
    std::atomic<std::uint64_t> updjnet_reissued{0};
    std::atomic<std::uint64_t> flux_sync_bytes{0};
    std::atomic<std::uint64_t> host_mirror_bytes{0};
    std::atomic<std::uint64_t> cusping_fired{0};
    std::atomic<std::uint64_t> cusping_dtil_bytes{0};
    std::atomic<std::uint64_t> device_flux_outers{0};
    std::atomic<std::uint64_t> flux_uploads_elided{0};
    std::atomic<std::uint64_t> xsnf_uploads_elided{0};
    std::atomic<std::uint64_t> dtil_uploads_elided{0};
    std::atomic<std::uint64_t> mirror_exits{0};
    std::atomic<std::uint64_t> canonical_nodal_outers{0};
    std::atomic<std::uint64_t> nodal_event_waits{0};
    std::atomic<std::uint64_t> reigv_device_outers{0};
    std::atomic<std::uint64_t> reigv_reissued{0};
    std::atomic<std::uint64_t> in_body_host_syncs{0};
    std::atomic<std::uint64_t> sync_exit_observation{0};
    std::atomic<std::uint64_t> sync_mirror_drain{0};
    std::atomic<std::uint64_t> sync_pre_nodal{0};
    std::atomic<std::uint64_t> sync_sweep_reissue{0};
    std::atomic<std::uint64_t> sync_segment_exit{0};
    std::atomic<std::uint64_t> hostfree_segments{0};
    std::atomic<std::uint64_t> hostfree_outers{0};
    std::atomic<std::uint64_t> hostfree_enqueued{0};
    std::atomic<std::uint64_t> sync_hostfree_exit{0};
    std::atomic<std::uint64_t> hostfree_repairs{0};
    std::atomic<std::uint64_t> hostfree_refusals[static_cast<int>(OuterHostFreeRefusal::Count)];
    // Rev.7.1 Task 10 part 4: the device-side WHILE.
    std::atomic<std::uint64_t> graph_segments{0};
    std::atomic<std::uint64_t> graph_launches{0};
    std::atomic<std::uint64_t> graph_iterations{0};
    std::atomic<std::uint64_t> graph_instantiations{0};
    std::atomic<std::uint64_t> graph_refusals[static_cast<int>(OuterGraphRefusal::Count)];
    std::atomic<std::uint64_t> phis_mirror_bytes{0};
    std::atomic<std::uint64_t> jnet_mirror_bytes{0};
    std::atomic<std::uint64_t> refusals[static_cast<int>(OuterSegmentRefusal::Count)];
    std::atomic<std::uint64_t> escapes[kDeviceEscapeCount];
    // WP14: the exit reason by PHASE, the loop's pass census, and the V2 arm.
    std::atomic<std::uint64_t> exit_reasons[kDevicePhaseCount];
    std::atomic<std::uint64_t> segment_passes{0};
    std::atomic<std::uint64_t> discovery_passes{0};
    std::atomic<std::uint64_t> v2_exit_syncs_elided{0};
    std::atomic<std::uint64_t> v2_state_d2h_elided{0};

    AtomicCounters() {
        for (auto& c : refusals) c.store(0, std::memory_order_relaxed);
        for (auto& c : escapes) c.store(0, std::memory_order_relaxed);
        for (auto& c : exit_reasons) c.store(0, std::memory_order_relaxed);
        for (auto& c : hostfree_refusals) c.store(0, std::memory_order_relaxed);
        for (auto& c : graph_refusals) c.store(0, std::memory_order_relaxed);
    }
};

/// ONE COUNTER SET PER SLOT, REACHED THROUGH THE THREAD.
///
/// Rev.7.1 Task 18-lite.  The bump sites -- about fifty of them, all inside the
/// body -- say `bump(counters().x)` and none of them holds a slot index.  A
/// Driver owns one host thread and one CMFD slot for its whole life, so the
/// thread is the index: outerSetThreadSlot() stamps it once per solve loop and
/// this returns that slot's set.  Two consequences, both wanted.  The atomics
/// stop being contended across M worker threads, and the receipt can print what
/// each deck did instead of one number that is the sum of four solves.
///
/// The run-wide totals are recovered by summing (outerSegmentCounters), so every
/// field the [OUTER_GPU] line printed before still means exactly what it meant.
/// A thread that never called the setter reads slot 0, which is what a single
/// run is.
thread_local int t_outer_slot = 0;

std::array<AtomicCounters, kMaxDeviceSlots>& slotCounters() {
    static std::array<AtomicCounters, kMaxDeviceSlots> c;
    return c;
}

AtomicCounters& counters() {
    const int s = (t_outer_slot >= 0 && t_outer_slot < kMaxDeviceSlots) ? t_outer_slot : 0;
    return slotCounters()[static_cast<std::size_t>(s)];
}

void bump(std::atomic<std::uint64_t>& c, std::uint64_t by = 1) {
    c.fetch_add(by, std::memory_order_relaxed);
}

} // namespace

void outerSetThreadSlot(int slot) {
    t_outer_slot = (slot >= 0 && slot < kMaxDeviceSlots) ? slot : 0;
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

bool outerGpuEnabled() {
    static const bool on = envFlagOn("RASBERY_GPU_OUTER");
    return on;
}

/// Rev.7.1 Task 10 part 3: may a segment run with no in-body synchronise?
///
/// DEFAULT ON, KILL-SWITCH OFF, which is the opposite of how a new arm usually
/// arrives -- and it is deliberate.  The arm is not a different physics path
/// with its own risk: it is the SAME body with the host's per-outer look
/// removed, and it is admitted only where that removal is proved inert (the
/// hostfree_refusals ladder).  A default-off flag would mean the exactness gate
/// and the production run exercise different code, which is exactly how the
/// batch-mode divergence of invariant 8 survived six gates.
///
/// RASBERY_GPU_OUTER_HOSTFREE=0 restores the per-outer arm byte for byte, which
/// is what an A/B against this task's commit compares.
bool outerHostFreeEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_OUTER_HOSTFREE");
        return v == nullptr || std::string(v) != "0";
    }();
    return on;
}

/// Rev.7.1 Task 10 part 3: ZERO in-body synchronises, at the price of the overrun.
///
/// THE TASK'S DELIVERABLE, AND OFF BY DEFAULT ON A MEASUREMENT.  A host-free
/// segment removes TWO drains, and they are not the same trade:
///
///   the PRE-NODAL drain (the sweep observation) is pure cost.  Removing it
///       removes nothing else -- the segment commits exactly the outers it
///       committed before -- so it is on by default (RASBERY_GPU_OUTER_HOSTFREE).
///
///   the TOP-OF-PASS drain (the exit word) is what lets the segment STOP.
///       Without it a segment cannot know its exit has latched, so it enqueues
///       its whole budget every time and the halt gate turns the remainder into
///       no-ops.  That is `hostfree_enqueued - hostfree_outers` in the receipt,
///       and on kngr3 at budget 8 it is 2322 no-op outers against 638 real ones
///       -- 4.29 s becomes 7.45 s locally.  Bit-exact, and slower.
///
/// SO THE ZERO-SYNC ARM SHIPS OFF, AND IT SHIPS ANYWAY, because it is the thing
/// the conditional WHILE needs: with the loop captured, the DEVICE decides the
/// trip count, the overrun stops existing, and the arm that is a loss at a fixed
/// budget becomes the only one that can run a segment without a host at all.
/// RASBERY_GPU_OUTER_HOSTFREE_FULL=1 turns it on and drives
/// `in_body_host_syncs` to zero for every non-exit outer; leaving it off keeps
/// the exit observation and one drain per outer instead of two.
bool outerHostFreeFull() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_OUTER_HOSTFREE_FULL");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}

/// WP14: RASBERY_GPU_OUTER_SEGMENT_V2 -- the segment exit, observed ONCE.
///
/// TWO ELISIONS, AND BOTH ARE ARGUED FROM THE SAME FACT: when the segment loop
/// breaks at the top of a pass it has just returned from
/// cudaStreamSynchronize(m.stream), so the stream is EMPTY and the pinned exit
/// word `m.h_seg` holds the 32 bytes the last committed outer's transition
/// wrote.  On that path
///
///   (1) the host-free exit's SECOND cudaStreamSynchronize has nothing to wait
///       for.  Its only job is to make the sweep accumulator visible, and a
///       blocking cudaMemcpy on an empty stream does that in one host call
///       instead of an async copy plus a rendezvous.
///   (2) the exit observation's D2H of DeviceOuterSegmentState re-reads bytes
///       that are already in `m.h_seg`, byte for byte, with no kernel in between
///       that could have written d_segments.
///
/// WHAT IS NOT CLAIMED.  This does not make segments longer and does not remove
/// the 0.88 ms the exit observation costs -- WP14 measured that segments stop
/// after 3.30 outers because the CMFD decision hands the slot to a HOST phase
/// (Xenon, overwhelmingly), and the observation is the host waiting for an outer
/// it had to wait for anyway.  The two elisions above are worth their own
/// counters and nothing more; the 2x is in the WHILE arm.
///
/// B0 BY CONSTRUCTION, so it is NOT in trajectory::kArmEnv.  Nothing device-side
/// is asked a different question and no host decision changes its inputs or its
/// order: the ON arm reads the same bytes through a shorter path.
bool outerSegmentV2Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_OUTER_SEGMENT_V2");
        if (v == nullptr) return false;
        const std::string s(v);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
                 s == "FALSE");
    }();
    return on;
}

unsigned int outerSegmentBudget() {
    static const unsigned int budget = [] {
        const char* v = std::getenv("RASBERY_GPU_OUTER_SEGMENT_MAX");
        if (v == nullptr) return kOuterSegmentBudgetDefault;
        const long parsed = std::atol(v);
        // A budget of 0 would be a segment that commits no outers and returns
        // true, which is an infinite loop at the call site; a budget past the
        // cap would outrun the queue width the buckets are built for.  Both fall
        // back to the default rather than to a silently clamped experiment,
        // because a run that answered a different question than the one asked is
        // the outcome Sec 4.4's admission policy already refuses elsewhere.
        if (parsed < 1 || parsed > static_cast<long>(kOuterSegmentBudgetMax)) {
            std::fprintf(stderr,
                         "[RASBERY][OUTER_GPU][WARN] RASBERY_GPU_OUTER_SEGMENT_MAX=%s is "
                         "outside [1, %u]; using the default %u\n",
                         v, kOuterSegmentBudgetMax, kOuterSegmentBudgetDefault);
            return kOuterSegmentBudgetDefault;
        }
        return static_cast<unsigned int>(parsed);
    }();
    return budget;
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

namespace {

/// One slot's counters, flattened.  The run-wide totals below are the sum of
/// these, so nothing that reads the [OUTER_GPU] line has to know slots exist.
OuterSegmentCounters snapshotSlotCounters(const AtomicCounters& a) {
    OuterSegmentCounters  out;
    out.segment_launches        = a.segment_launches.load(std::memory_order_relaxed);
    out.device_outers           = a.device_outers.load(std::memory_order_relaxed);
    out.host_outer_observations = a.host_outer_observations.load(std::memory_order_relaxed);
    out.budget_exits            = a.budget_exits.load(std::memory_order_relaxed);
    out.halted_outer_launches   = a.halted_outer_launches.load(std::memory_order_relaxed);
    out.jnet_bridge_bytes       = a.jnet_bridge_bytes.load(std::memory_order_relaxed);
    out.updjnet_reissued        = a.updjnet_reissued.load(std::memory_order_relaxed);
    out.flux_sync_bytes         = a.flux_sync_bytes.load(std::memory_order_relaxed);
    out.host_mirror_bytes       = a.host_mirror_bytes.load(std::memory_order_relaxed);
    out.cusping_fired           = a.cusping_fired.load(std::memory_order_relaxed);
    out.cusping_dtil_bytes      = a.cusping_dtil_bytes.load(std::memory_order_relaxed);
    out.device_flux_outers      = a.device_flux_outers.load(std::memory_order_relaxed);
    out.flux_uploads_elided     = a.flux_uploads_elided.load(std::memory_order_relaxed);
    out.xsnf_uploads_elided     = a.xsnf_uploads_elided.load(std::memory_order_relaxed);
    out.dtil_uploads_elided     = a.dtil_uploads_elided.load(std::memory_order_relaxed);
    out.mirror_exits            = a.mirror_exits.load(std::memory_order_relaxed);
    out.canonical_nodal_outers  = a.canonical_nodal_outers.load(std::memory_order_relaxed);
    out.nodal_event_waits       = a.nodal_event_waits.load(std::memory_order_relaxed);
    out.reigv_device_outers     = a.reigv_device_outers.load(std::memory_order_relaxed);
    out.reigv_reissued          = a.reigv_reissued.load(std::memory_order_relaxed);
    out.in_body_host_syncs      = a.in_body_host_syncs.load(std::memory_order_relaxed);
    out.sync_exit_observation  = a.sync_exit_observation.load(std::memory_order_relaxed);
    out.sync_mirror_drain      = a.sync_mirror_drain.load(std::memory_order_relaxed);
    out.sync_pre_nodal         = a.sync_pre_nodal.load(std::memory_order_relaxed);
    out.sync_sweep_reissue     = a.sync_sweep_reissue.load(std::memory_order_relaxed);
    out.sync_segment_exit      = a.sync_segment_exit.load(std::memory_order_relaxed);
    out.hostfree_segments      = a.hostfree_segments.load(std::memory_order_relaxed);
    out.hostfree_outers        = a.hostfree_outers.load(std::memory_order_relaxed);
    out.hostfree_enqueued      = a.hostfree_enqueued.load(std::memory_order_relaxed);
    out.sync_hostfree_exit     = a.sync_hostfree_exit.load(std::memory_order_relaxed);
    out.hostfree_repairs       = a.hostfree_repairs.load(std::memory_order_relaxed);
    for (int i = 0; i < static_cast<int>(OuterHostFreeRefusal::Count); ++i)
        out.hostfree_refusals[i] = a.hostfree_refusals[i].load(std::memory_order_relaxed);
    out.graph_segments         = a.graph_segments.load(std::memory_order_relaxed);
    out.graph_launches         = a.graph_launches.load(std::memory_order_relaxed);
    out.graph_iterations       = a.graph_iterations.load(std::memory_order_relaxed);
    out.graph_instantiations   = a.graph_instantiations.load(std::memory_order_relaxed);
    out.graph_warmup_misses =
        rasbery::g_graph_warmup_misses.load(std::memory_order_relaxed);
    for (int i = 0; i < static_cast<int>(OuterGraphRefusal::Count); ++i)
        out.graph_refusals[i] = a.graph_refusals[i].load(std::memory_order_relaxed);
    out.phis_mirror_bytes       = a.phis_mirror_bytes.load(std::memory_order_relaxed);
    out.jnet_mirror_bytes       = a.jnet_mirror_bytes.load(std::memory_order_relaxed);
    for (int i = 0; i < static_cast<int>(OuterSegmentRefusal::Count); ++i)
        out.refusals[i] = a.refusals[i].load(std::memory_order_relaxed);
    for (int i = 0; i < kDeviceEscapeCount; ++i)
        out.escapes[i] = a.escapes[i].load(std::memory_order_relaxed);
    for (int i = 0; i < kDevicePhaseCount; ++i)
        out.exit_reasons[i] = a.exit_reasons[i].load(std::memory_order_relaxed);
    out.segment_passes       = a.segment_passes.load(std::memory_order_relaxed);
    out.discovery_passes     = a.discovery_passes.load(std::memory_order_relaxed);
    out.v2_exit_syncs_elided = a.v2_exit_syncs_elided.load(std::memory_order_relaxed);
    out.v2_state_d2h_elided  = a.v2_state_d2h_elided.load(std::memory_order_relaxed);
    return out;
}

/// Add @p add into @p into, field by field.
void addCounters(OuterSegmentCounters& into, const OuterSegmentCounters& add) {
    into.segment_launches        += add.segment_launches;
    into.device_outers           += add.device_outers;
    into.host_outer_observations += add.host_outer_observations;
    into.budget_exits            += add.budget_exits;
    into.halted_outer_launches   += add.halted_outer_launches;
    into.jnet_bridge_bytes       += add.jnet_bridge_bytes;
    into.updjnet_reissued        += add.updjnet_reissued;
    into.flux_sync_bytes         += add.flux_sync_bytes;
    into.host_mirror_bytes       += add.host_mirror_bytes;
    into.cusping_fired           += add.cusping_fired;
    into.cusping_dtil_bytes      += add.cusping_dtil_bytes;
    into.device_flux_outers      += add.device_flux_outers;
    into.flux_uploads_elided     += add.flux_uploads_elided;
    into.xsnf_uploads_elided     += add.xsnf_uploads_elided;
    into.dtil_uploads_elided     += add.dtil_uploads_elided;
    into.mirror_exits            += add.mirror_exits;
    into.canonical_nodal_outers  += add.canonical_nodal_outers;
    into.nodal_event_waits       += add.nodal_event_waits;
    into.reigv_device_outers     += add.reigv_device_outers;
    into.reigv_reissued          += add.reigv_reissued;
    into.in_body_host_syncs      += add.in_body_host_syncs;
    into.sync_exit_observation   += add.sync_exit_observation;
    into.sync_mirror_drain       += add.sync_mirror_drain;
    into.sync_pre_nodal          += add.sync_pre_nodal;
    into.sync_sweep_reissue      += add.sync_sweep_reissue;
    into.sync_segment_exit       += add.sync_segment_exit;
    into.hostfree_segments       += add.hostfree_segments;
    into.hostfree_outers         += add.hostfree_outers;
    into.hostfree_enqueued       += add.hostfree_enqueued;
    into.sync_hostfree_exit      += add.sync_hostfree_exit;
    into.hostfree_repairs        += add.hostfree_repairs;
    for (int i = 0; i < static_cast<int>(OuterHostFreeRefusal::Count); ++i)
        into.hostfree_refusals[i] += add.hostfree_refusals[i];
    into.graph_segments          += add.graph_segments;
    into.graph_launches          += add.graph_launches;
    into.graph_iterations        += add.graph_iterations;
    into.graph_instantiations    += add.graph_instantiations;
    // NOT summed: the warm-up misses are a PROCESS-wide atomic that every slot's
    // snapshot already reports in full, so adding them per slot would multiply
    // the same number by the slot count.
    into.graph_warmup_misses      = add.graph_warmup_misses;
    for (int i = 0; i < static_cast<int>(OuterGraphRefusal::Count); ++i)
        into.graph_refusals[i] += add.graph_refusals[i];
    into.phis_mirror_bytes       += add.phis_mirror_bytes;
    into.jnet_mirror_bytes       += add.jnet_mirror_bytes;
    for (int i = 0; i < static_cast<int>(OuterSegmentRefusal::Count); ++i)
        into.refusals[i] += add.refusals[i];
    for (int i = 0; i < kDeviceEscapeCount; ++i)
        into.escapes[i] += add.escapes[i];
    for (int i = 0; i < kDevicePhaseCount; ++i)
        into.exit_reasons[i] += add.exit_reasons[i];
    into.segment_passes       += add.segment_passes;
    into.discovery_passes     += add.discovery_passes;
    into.v2_exit_syncs_elided += add.v2_exit_syncs_elided;
    into.v2_state_d2h_elided  += add.v2_state_d2h_elided;
}

} // namespace

/// The run-wide totals: every slot, summed.
///
/// Rev.7.1 Task 18-lite.  A single run has one live slot and this is a copy of
/// it, which is what keeps the ON-vs-OFF receipt comparison a byte comparison.
OuterSegmentCounters outerSegmentCounters() {
    OuterSegmentCounters out;
    for (int i = 0; i < kMaxDeviceSlots; ++i)
        addCounters(out, snapshotSlotCounters(slotCounters()[static_cast<std::size_t>(i)]));
    return out;
}

std::string outerSegmentReceiptJson() {
    const OuterSegmentCounters c = outerSegmentCounters();
    std::string                s = "{\"segment_launches\":" + std::to_string(c.segment_launches) +
                    ",\"device_outers\":" + std::to_string(c.device_outers) +
                    ",\"host_outer_observations\":" +
                    std::to_string(c.host_outer_observations) +
                    ",\"budget_exits\":" + std::to_string(c.budget_exits) +
                    ",\"halted_outer_launches\":" + std::to_string(c.halted_outer_launches) +
                    ",\"jnet_bridge_bytes\":" + std::to_string(c.jnet_bridge_bytes) +
                    ",\"updjnet_reissued\":" + std::to_string(c.updjnet_reissued) +
                    ",\"flux_sync_bytes\":" + std::to_string(c.flux_sync_bytes) +
                    ",\"host_mirror_bytes\":" + std::to_string(c.host_mirror_bytes) +
                    ",\"cusping_fired\":" + std::to_string(c.cusping_fired) +
                    ",\"cusping_dtil_bytes\":" + std::to_string(c.cusping_dtil_bytes) +
                    ",\"device_flux_outers\":" + std::to_string(c.device_flux_outers) +
                    ",\"flux_uploads_elided\":" + std::to_string(c.flux_uploads_elided) +
                    ",\"xsnf_uploads_elided\":" + std::to_string(c.xsnf_uploads_elided) +
                    ",\"dtil_uploads_elided\":" + std::to_string(c.dtil_uploads_elided) +
                    ",\"mirror_exits\":" + std::to_string(c.mirror_exits) +
                    ",\"canonical_nodal_outers\":" +
                    std::to_string(c.canonical_nodal_outers) +
                    ",\"nodal_event_waits\":" + std::to_string(c.nodal_event_waits) +
                    ",\"reigv_device_outers\":" +
                    std::to_string(c.reigv_device_outers) +
                    ",\"reigv_reissued\":" + std::to_string(c.reigv_reissued) +
                    ",\"in_body_host_syncs\":" +
                    std::to_string(c.in_body_host_syncs) +
                    ",\"sync_exit_observation\":" + std::to_string(c.sync_exit_observation) +
                    ",\"sync_mirror_drain\":" + std::to_string(c.sync_mirror_drain) +
                    ",\"sync_pre_nodal\":" + std::to_string(c.sync_pre_nodal) +
                    ",\"sync_sweep_reissue\":" + std::to_string(c.sync_sweep_reissue) +
                    ",\"sync_segment_exit\":" + std::to_string(c.sync_segment_exit) +
                    ",\"hostfree_segments\":" + std::to_string(c.hostfree_segments) +
                    ",\"hostfree_outers\":" + std::to_string(c.hostfree_outers) +
                    ",\"hostfree_enqueued\":" + std::to_string(c.hostfree_enqueued) +
                    ",\"sync_hostfree_exit\":" + std::to_string(c.sync_hostfree_exit) +
                    ",\"hostfree_repairs\":" + std::to_string(c.hostfree_repairs) +
                    ",\"phis_mirror_bytes\":" + std::to_string(c.phis_mirror_bytes) +
                    ",\"jnet_mirror_bytes\":" + std::to_string(c.jnet_mirror_bytes) +
                    ",\"hostfree_full\":" + std::to_string(outerHostFreeFull() ? 1 : 0) +
                    ",\"segment_budget\":" + std::to_string(outerSegmentBudget()) +
                    // Rev.7.1 Task 10 part 4.  iterations_per_launch is printed
                    // rather than left to the reader because it is THE number
                    // the WHILE exists to move: it is 1.0 on the stream arm by
                    // definition (one host launch per outer) and the segment's
                    // mean outer count on the graph arm.
                    ",\"graph_arm\":" + std::to_string(outerGraphEnabled() ? 1 : 0) +
                    ",\"graph_segments\":" + std::to_string(c.graph_segments) +
                    ",\"graph_launches\":" + std::to_string(c.graph_launches) +
                    ",\"graph_iterations\":" + std::to_string(c.graph_iterations) +
                    ",\"iterations_per_launch\":" +
                    (c.graph_launches == 0
                         ? std::string("0")
                         : std::to_string(static_cast<double>(c.graph_iterations) /
                                          static_cast<double>(c.graph_launches))) +
                    ",\"graph_instantiations\":" + std::to_string(c.graph_instantiations) +
                    ",\"graph_warmup_misses\":" + std::to_string(c.graph_warmup_misses) +
                    // WP14.  segment_passes - device_outers IS discovery_passes;
                    // both are printed so the identity is checkable rather than
                    // assumed, and outers_per_segment is printed because it is
                    // the number that decides whether the budget is reachable
                    // at all on this deck.
                    ",\"segment_v2_arm\":" + std::to_string(outerSegmentV2Enabled() ? 1 : 0) +
                    ",\"segment_passes\":" + std::to_string(c.segment_passes) +
                    ",\"discovery_passes\":" + std::to_string(c.discovery_passes) +
                    ",\"outers_per_segment\":" +
                    (c.segment_launches == 0
                         ? std::string("0")
                         : std::to_string(static_cast<double>(c.device_outers) /
                                          static_cast<double>(c.segment_launches))) +
                    ",\"v2_exit_syncs_elided\":" + std::to_string(c.v2_exit_syncs_elided) +
                    ",\"v2_state_d2h_elided\":" + std::to_string(c.v2_state_d2h_elided);

    // Only the non-zero buckets, so a healthy run's line stays readable and a
    // nonzero escape stands out instead of hiding among ten zeros.
    s += ",\"escapes\":{";
    bool first = true;
    for (int i = 0; i < kDeviceEscapeCount; ++i) {
        if (c.escapes[i] == 0) continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        s += outerEscapeName(static_cast<DeviceEscape>(i));
        s += "\":" + std::to_string(c.escapes[i]);
    }
    s += "}";

    // WP14: the same census by the PHASE the segment left for.  Same non-zero
    // rule.  This is the line that answers "why does a budget of 8 stop at 3":
    // an `xenon` bucket the size of segment_launches says the CMFD decision is
    // handing the slot to a host phase, and no device-side predicate can
    // continue across that.
    s += ",\"exit_reasons\":{";
    first = true;
    for (int i = 0; i < kDevicePhaseCount; ++i) {
        if (c.exit_reasons[i] == 0) continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        s += outerExitPhaseName(static_cast<DevicePhase>(i));
        s += "\":" + std::to_string(c.exit_reasons[i]);
    }
    s += "}";

    s += ",\"refusals\":{";
    first = true;
    for (int i = 0; i < static_cast<int>(OuterSegmentRefusal::Count); ++i) {
        if (c.refusals[i] == 0) continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        s += outerRefusalName(static_cast<OuterSegmentRefusal>(i));
        s += "\":" + std::to_string(c.refusals[i]);
    }
    s += "}";

    // Rev.7.1 Task 10 part 3: why the segments that DID run still paid a
    // synchronise per outer.  Non-zero buckets only, same rule as above.
    s += ",\"hostfree_refusals\":{";
    first = true;
    for (int i = 0; i < static_cast<int>(OuterHostFreeRefusal::Count); ++i) {
        if (c.hostfree_refusals[i] == 0) continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        s += outerHostFreeRefusalName(static_cast<OuterHostFreeRefusal>(i));
        s += "\":" + std::to_string(c.hostfree_refusals[i]);
    }
    s += "}";

    // Rev.7.1 Task 10 part 4: why the host-free segments still walked the loop.
    s += ",\"graph_refusals\":{";
    first = true;
    for (int i = 0; i < static_cast<int>(OuterGraphRefusal::Count); ++i) {
        if (c.graph_refusals[i] == 0) continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        s += outerGraphRefusalName(static_cast<OuterGraphRefusal>(i));
        s += "\":" + std::to_string(c.graph_refusals[i]);
    }
    s += "},";
    // Printed ONLY when nothing ran: on a healthy run the reason is "none" and
    // saying so would be noise, while on an idle run it is the whole message.
    if (c.segment_launches == 0) {
        // SLOT 0'S LADDER, and that is honest for the reason it always was: this
        // is a run-wide line printed from a singleton with no Driver to ask, and
        // the reasons it can still see -- no runner, no arena, unbound, a batch
        // wider than the arena -- are run-wide.  A per-slot reason that differs
        // from slot 0's shows up in the per-slot lines below, which is where a
        // reader who needs it will look.
        s += outerIdleReasonJson(
            rasberyOuterSegment().refusal(rasberyBatchWidth(), false, false, true));
        s += ",";
    }
    s += outerHostBodyJson();
    s += "}";
    return s;
}

/// One slot's line: what THIS deck's runner did.
///
/// Rev.7.1 Task 18-lite.  The run-wide line above is a sum, and a sum is exactly
/// the wrong shape for the question a batch raises -- "did the segment engage
/// for every deck, or for one of them four times".  Printed only for slots that
/// were touched, so a single run adds one line and an idle slot adds none.
static std::string outerSlotReceiptJson(int slot, const OuterSegmentCounters& c) {
    std::string s = "{\"slot\":" + std::to_string(slot) +
                    ",\"segment_launches\":" + std::to_string(c.segment_launches) +
                    ",\"device_outers\":" + std::to_string(c.device_outers) +
                    ",\"host_outer_observations\":" +
                    std::to_string(c.host_outer_observations) +
                    ",\"canonical_nodal_outers\":" +
                    std::to_string(c.canonical_nodal_outers) +
                    ",\"jnet_bridge_bytes\":" + std::to_string(c.jnet_bridge_bytes) +
                    ",\"refusals\":{";
    bool first = true;
    for (int i = 0; i < static_cast<int>(OuterSegmentRefusal::Count); ++i) {
        if (c.refusals[i] == 0) continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        s += outerRefusalName(static_cast<OuterSegmentRefusal>(i));
        s += "\":" + std::to_string(c.refusals[i]);
    }
    s += "}}";
    return s;
}

void reportOuterSegment(std::ostream& os) {
    if (!outerGpuEnabled()) return;
    os << "[RASBERY][OUTER_GPU] " << outerSegmentReceiptJson() << std::endl;
    // The per-slot breakdown, and the width it was measured against.  A batch
    // whose receipt says `device_outers > 0` on the run-wide line and names only
    // one slot here is a batch where three decks got nothing, which is the
    // failure this line exists to make impossible to miss.
    const int arena_slots = rasberyOuterArenaSlots();
    if (rasberyBatchWidth() <= 1 && arena_slots <= 1) return;
    os << "[RASBERY][OUTER_GPU][SLOTS] {\"batch_width\":" << rasberyBatchWidth()
       << ",\"arena_slots\":" << arena_slots << "}" << std::endl;
    for (int i = 0; i < kMaxDeviceSlots; ++i) {
        const OuterSegmentCounters c =
            snapshotSlotCounters(slotCounters()[static_cast<std::size_t>(i)]);
        bool touched = c.segment_launches != 0 || c.device_outers != 0;
        for (int r = 0; r < static_cast<int>(OuterSegmentRefusal::Count) && !touched; ++r)
            touched = c.refusals[r] != 0;
        if (!touched) continue;
        os << "[RASBERY][OUTER_GPU][SLOT] " << outerSlotReceiptJson(i, c) << std::endl;
    }
}

void noteOuterSegmentRefusal(OuterSegmentRefusal why) {
    if (why == OuterSegmentRefusal::None) return;
    bump(counters().refusals[static_cast<int>(why)]);
}

// ---------------------------------------------------------------------------
// The runner
// ---------------------------------------------------------------------------

namespace {

/// WP19.1.  HAS THE BUILD ALREADY MOVED HOST STATE BY THE TIME IT REFUSED?
///
/// `stage` is buildOuterWhile()'s cursor and its eight values are the contract
/// written out in GpuOuterWhile.h.  The first eight are pure graph plumbing:
/// nothing host-side has moved and nothing device-side has run, so the build is
/// free to be abandoned or repeated.  From `record(body)` on, the body's thirty
/// enqueue helpers HAVE run as host calls -- they committed the CMFD backend's
/// byte-exact upload shadows (CudaTransferMirror.h commits AT THE ISSUE) and
/// the segment's own residency generations, for copies that were only RECORDED
/// into a capture that is about to be thrown away.
///
/// TWO CALLERS, ONE RULE, AND THAT IS THE POINT.  The refusal path below has
/// asked this question since Task 10 (a repeat there would be "a plausible
/// wrong answer rather than a slow one").  WP19's capture-race retry did not
/// ask it, and re-recorded the body unconditionally: the second record found
/// every shadow already committed, elided the uploads, and instantiated a WHILE
/// whose body has no H2D node for data the device never received.  On a lane's
/// FIRST case that device memory is uninitialised and the replay lands in
/// BICGCMFD's non-finite guard with no CUDA error anywhere -- which is the
/// silent second face of the WP19 race.  One predicate, asked by both.
/// [[maybe_unused]]: both callers are inside `#if RASBERY_HAS_OUTER_WHILE`, and
/// a stub build that compiles this TU without the WHILE arm must not warn.
[[maybe_unused]] bool outerWhileStageMovedHostState(const char* stage) {
    if (stage == nullptr) return true; // unknown cursor: assume the worst
    static const char* const kBeforeBody[] = {
        "BeginCapture(root)",     "GetCaptureInfo(root)", "ConditionalHandleCreate",
        "arm",                    "GetCaptureInfo(arm)",  "AddNode(while)",
        "UpdateCaptureDependencies", "BeginCaptureToGraph(body)"};
    for (const char* known : kBeforeBody)
        if (std::strcmp(stage, known) == 0) return false;
    return true;
}

/// One physical slot's queue, built on the host.  A segment is single-slot for
/// now (Sec 3.2 batch integration is Task 10/18), so count and bucket are 1 and
/// every other entry is padding -- which the kernels must never dereference,
/// hence the full fill rather than just slots[0].
DevicePhaseQueue singleSlotQueue(int slot) {
    DevicePhaseQueue q{};
    for (int i = 0; i < kMaxSchedulerSlots; ++i) q.slots[i] = kQueueEmptySlot;
    q.slots[0] = slot;
    q.count    = 1;
    q.bucket   = gpuSelectBucket(1);
    return q;
}

} // namespace

struct CudaOuterSegment::Impl {
    bool            ready      = false;
    std::string     status     = "uninitialised";
    DeviceArenaView arena{};
    int             slot_count = 0;
    /// Rev.7.1 Task 18-lite: WHICH arena slot this runner serves.  Fixed at
    /// initialize() and never re-aimed, so a Driver cannot be handed a runner
    /// that is quietly pointing somewhere else.
    int             slot       = -1;

    /// The stream the segment issues on.  Rev.7.1 Task 10 part 2: normally the
    /// CMFD arena's, so the sweep's kernels and the segment's are ordered by the
    /// stream itself rather than by an event pair; `own_stream` is the private
    /// fallback and the thing release() destroys.
    cudaStream_t stream     = nullptr;
    cudaStream_t own_stream = nullptr;

    // Device scratch, all sized [slot_count] except the one counter.
    DeviceOuterProbe*        d_probes    = nullptr;
    DeviceOuterSegmentState* d_segments  = nullptr;
    std::uint32_t*           d_halt      = nullptr;
    cmfd::CmfdOuterInputs*   d_inputs    = nullptr;
    CmfdOuterDecision*       d_decisions = nullptr;
    unsigned long long*      d_halted    = nullptr;
    // ---- Rev.7.1 Task 10 part 3: the host-free arm's two device records -----
    //
    // The sweep summary the verdict kernel keeps as the segment runs, so the
    // host can reconstruct BICGCMFD's counters at the exit instead of observing
    // every outer.  One per slot; zeroed at each host-free segment entry, which
    // is what makes it a SEGMENT total.  See
    // CudaBatchArena::CmfdSweepProbeSink::Accum.
    CudaBatchArena::CmfdSweepProbeSink::Accum* d_sweep_accum = nullptr;
    CudaBatchArena::CmfdSweepProbeSink::Accum* h_sweep_accum = nullptr;
    /// Rev.7.1 Task 10 part 3: the segment -> nodal handover on the host-free
    /// arm.  Recorded on `stream` after updjnet and waited on by the XS-recon
    /// backend's own stream, which is the ordering the per-outer arm gets from
    /// its `sync_pre_nodal` drain.  Created lazily, once per runner.
    cudaEvent_t              nodal_handover_ev = nullptr;
    /// Is a HOST-FREE segment running on this runner right now?  Read by
    /// probeAddresses, which is how the sweep hook learns to hand the arena an
    /// accumulator instead of expecting to be observed.
    bool                     hostfree_active = false;

    OuterSegmentResidency residency{};
    /// The generation each device copy was last filled AT.  Zero is `never
    /// uploaded`, which the host generations cannot collide with because they
    /// start at 1.
    unsigned long long resident_flux_generation = 0;
    unsigned long long resident_dtil_generation = 0;
    /// The BYTES of xsnf the device copy was last filled FROM.
    ///
    /// A GENERATION IS NOT USABLE FOR THIS ONE, and the codebase already says
    /// so where the sweep does the same upload: XSSet::hoststateGeneration()
    /// means "the device mirror of _xs is stale", NOT "the host bytes changed",
    /// and XSSet.cpp deliberately does not bump it when the GPU XS arm writes a
    /// freshly reconstructed _xs into host memory (XSSet.cpp:2772,
    /// XSSet::UpdateEquilibriumXenon's device arm).  The segment's device xsnf
    /// is a DIFFERENT device buffer from that mirror -- it is the CMFD arena's
    /// -- so those are exactly the writes it must not miss, and it missed them:
    /// on a host with RASBERY_GPU_XSRECON / RASBERY_GPU_FLATXS set, kngr_238
    /// statepoint 1 outer 28 (the first boron trial commit) rebuilt xsnf on the
    /// host, the generation did not move, this upload was elided, and updpsi
    /// then computed psi = flux . xsnf . vol from the PREVIOUS xsnf -- the psi
    /// hash of outer 28 equal to outer 27's, with everything downstream in that
    /// outer diverging from the host arm.
    ///
    /// A byte-exact shadow cannot miss a writer by construction, which is the
    /// whole reason it and not another generation: memcmp over 135 KB is ~2 us
    /// at the kngr mesh against ~19.5 us for the H2D issue it elides, the same
    /// trade CudaBICGBackend.cu's flux mirror measured and kept.
    cuda_transfer::ByteExactMirror<double> resident_xsnf;
    bool                  residency_bound = false;
    /// Raised by republishAfterHostSweep: THIS outer's drive did not finish on
    /// the device and the host finished it at the observation.  Read by the body
    /// to re-issue the step the sweep verdict's halt swallowed; cleared at the
    /// top of every outer so it can never describe an older one.
    bool                  sweep_host_continued = false;
    /// RASBERY_OUTER_TRACE scratch.  The per-step tracer has to hash DEVICE
    /// memory on this arm -- the host mirrors are elided inside a segment -- so
    /// it needs somewhere to land the copies.  Grown on first use and only when
    /// the tracer is on; an untraced run never allocates it.
    std::vector<double>   trace_scratch;
    /// The segment exit word, read once per outer with no extra synchronise.
    ///
    /// A D2H of it rides the stream right behind the transition, so at the
    /// segment's NEXT per-outer synchronise the host can see whether the
    /// PREVIOUS outer ended without paying for a second rendezvous.  Pinned
    /// because it is copied every outer and a pageable 32-byte D2H stages
    /// through the driver on the launcher's critical path.
    DeviceOuterSegmentState* h_seg = nullptr;

    OuterSegmentHooks   hooks{};
    OuterSegmentBinding binding{};
    bool                is_bound = false;
    /// Rev.7.1 Task 18-lite: has this segment told the host side that the DEVICE
    /// owns the canonical nodal regions, and not yet taken it back?
    ///
    /// A LATCH AND NOT A RECOMPUTE, because it answers two questions at once and
    /// they must give the same answer: whether the release is owed, and whether
    /// Geometry::Jnet/Phis are behind the device at the exit.  It goes DOWN
    /// inside a segment too -- an outer whose drive falls back to the CPU body
    /// takes the bridge and writes the host arrays itself, and mirroring the
    /// device over that would overwrite the newer values with the older ones.
    bool                canonical_nodal_live = false;
    /// Rev.7.1 W3 item 3: has this segment told the nodal backend that the reigv
    /// slot is written on the device, and not yet taken it back?
    ///
    /// A LATCH FOR THE SAME REASON canonical_nodal_live IS ONE.  The declaration
    /// is sticky on the backend -- it suppresses an upload -- so a segment that
    /// set it and then returned without clearing it would hand every subsequent
    /// HOST nodal drive a slot nobody writes.  That is the sticky-flag failure
    /// that cost 169 host outers on kngr_238 in its dhat costume, and it is
    /// cheaper to keep the latch than to audit five return paths.
    bool                reigv_device_claimed = false;
    /// Rev.7.1 Task 10 part 4: the instantiated outer WHILEs, by shape.
    ///
    /// PER RUNNER AND NOT PROCESS-WIDE, because every address the body bakes
    /// is this runner's -- its segment state, its halt word, its arena slot --
    /// and a cache shared between slots would hand one deck another's outer.
    OuterWhileCache     while_cache{};
};

CudaOuterSegment::CudaOuterSegment() : _impl(new Impl) {}

CudaOuterSegment::~CudaOuterSegment() {
    release();
    delete _impl;
}

// ---------------------------------------------------------------------------
// Rev.7.1 Task 18-lite: one runner per arena slot
// ---------------------------------------------------------------------------

CudaOuterSegment& rasberyOuterSegment(int slot) {
    // kMaxDeviceSlots objects, constructed on first use and never destroyed
    // before the process ends -- the arena's own lifetime rule, for the arena's
    // own reason: the addresses a runner hands out must outlive every Driver.
    // An empty Impl is a pointer and a short string, so the slots a run never
    // touches cost nothing but their construction.
    static std::array<CudaOuterSegment, kMaxDeviceSlots> segments;
    // THE OUT-OF-RANGE ANSWER IS A RUNNER THAT REFUSES, NOT SLOT 0.  Clamping
    // would alias one Driver's residency onto another's, which is the exact
    // failure this table exists to stop; this object is never initialised, so
    // available() is false and the ladder answers `no_runner`.
    static CudaOuterSegment unserved;
    if (slot < 0 || slot >= kMaxDeviceSlots) return unserved;
    return segments[static_cast<std::size_t>(slot)];
}

bool CudaOuterSegment::initialize(const DeviceArenaView& arena, int slot_count, int slot) {
    release();
    if (slot_count <= 0 || slot_count > kMaxSchedulerSlots) {
        _impl->status = "slot_count outside [1, kMaxSchedulerSlots]";
        return false;
    }
    if (slot < 0 || slot >= slot_count) {
        _impl->status = "slot outside [0, slot_count)";
        return false;
    }
    _impl->arena      = arena;
    _impl->slot_count = slot_count;
    _impl->slot       = slot;

    // A PARTIAL initialise MUST NOT KEEP ITS ALLOCATIONS.  The runner is
    // unavailable either way, so nothing will ever free them through the normal
    // path, and a caller that retries -- a different slot count, a later device
    // -- would leak every buffer the first attempt got to.  release() clears the
    // pointers and the ready flag but does not touch `status`, so the reason
    // survives the cleanup and status() still says what went wrong.
    const auto fail = [&](const char* what, cudaError_t rc) {
        const std::string why = std::string(what) + ": " + cudaGetErrorString(rc);
        release();
        _impl->status = why;
        return false;
    };

    const std::size_t n = static_cast<std::size_t>(slot_count);
    cudaError_t       rc;
    // Rev.7.1 Task 18d: this stand-up runs per slot on the owning Driver's
    // thread, which in a batch is while earlier decks capture the shared CMFD
    // arena's graph.  See GpuCaptureArbiter.h.
    rasbery::AllocWindow _alloc_window("outer.segment.standup");
    if ((rc = cudaStreamCreateWithFlags(&_impl->own_stream, cudaStreamNonBlocking)) !=
        cudaSuccess)
        return fail("cudaStreamCreateWithFlags", rc);
    _impl->stream = _impl->own_stream;
    if ((rc = cudaMallocHost(&_impl->h_seg, sizeof(DeviceOuterSegmentState))) != cudaSuccess)
        return fail("cudaMallocHost(segment state)", rc);
    if ((rc = cudaMalloc(&_impl->d_probes, n * sizeof(DeviceOuterProbe))) != cudaSuccess)
        return fail("cudaMalloc(probes)", rc);
    if ((rc = cudaMalloc(&_impl->d_segments, n * sizeof(DeviceOuterSegmentState))) !=
        cudaSuccess)
        return fail("cudaMalloc(segments)", rc);
    if ((rc = cudaMalloc(&_impl->d_halt, n * sizeof(std::uint32_t))) != cudaSuccess)
        return fail("cudaMalloc(halt)", rc);
    if ((rc = cudaMalloc(&_impl->d_inputs, n * sizeof(cmfd::CmfdOuterInputs))) != cudaSuccess)
        return fail("cudaMalloc(inputs)", rc);
    if ((rc = cudaMalloc(&_impl->d_decisions, n * sizeof(CmfdOuterDecision))) != cudaSuccess)
        return fail("cudaMalloc(decisions)", rc);
    if ((rc = cudaMalloc(&_impl->d_halted, sizeof(unsigned long long))) != cudaSuccess)
        return fail("cudaMalloc(halted)", rc);

    if ((rc = cudaMalloc(&_impl->d_sweep_accum,
                         n * sizeof(CudaBatchArena::CmfdSweepProbeSink::Accum))) !=
        cudaSuccess)
        return fail("cudaMalloc(sweep accumulator)", rc);
    // PINNED, because it is D2H'd on the segment exit observation that is
    // already carrying four other small copies, and a pageable destination
    // there would stage through the driver on the critical path.
    if ((rc = cudaMallocHost(&_impl->h_sweep_accum,
                             sizeof(CudaBatchArena::CmfdSweepProbeSink::Accum))) !=
        cudaSuccess)
        return fail("cudaMallocHost(sweep accumulator)", rc);

    if ((rc = cudaMemset(_impl->d_halted, 0, sizeof(unsigned long long))) != cudaSuccess)
        return fail("cudaMemset(halted)", rc);
    if ((rc = cudaMemset(_impl->d_sweep_accum, 0,
                         n * sizeof(CudaBatchArena::CmfdSweepProbeSink::Accum))) !=
        cudaSuccess)
        return fail("cudaMemset(sweep accumulator)", rc);

    _impl->ready  = true;
    _impl->status = "ready";
    return true;
}

void CudaOuterSegment::release() {
    if (_impl == nullptr) return;
    rasbery::AllocWindow _alloc_window("outer.segment.release");
    // Rev.7.1 Task 10 part 4: FIRST, because every cached WHILE bakes the
    // addresses freed below.  A cache that outlived them would hand the next
    // arm an exec whose nodes point into freed device memory.
    _impl->while_cache.release();
    if (_impl->d_probes != nullptr) cudaFree(_impl->d_probes);
    if (_impl->d_segments != nullptr) cudaFree(_impl->d_segments);
    if (_impl->d_halt != nullptr) cudaFree(_impl->d_halt);
    if (_impl->d_inputs != nullptr) cudaFree(_impl->d_inputs);
    if (_impl->d_decisions != nullptr) cudaFree(_impl->d_decisions);
    if (_impl->d_halted != nullptr) cudaFree(_impl->d_halted);
    if (_impl->d_sweep_accum != nullptr) cudaFree(_impl->d_sweep_accum);
    if (_impl->nodal_handover_ev != nullptr) cudaEventDestroy(_impl->nodal_handover_ev);
    if (_impl->h_seg != nullptr) cudaFreeHost(_impl->h_seg);
    if (_impl->h_sweep_accum != nullptr) cudaFreeHost(_impl->h_sweep_accum);
    if (_impl->own_stream != nullptr) cudaStreamDestroy(_impl->own_stream);
    _impl->d_probes    = nullptr;
    _impl->d_segments  = nullptr;
    _impl->d_halt      = nullptr;
    _impl->d_inputs    = nullptr;
    _impl->d_decisions = nullptr;
    _impl->d_halted    = nullptr;
    _impl->d_sweep_accum = nullptr;
    _impl->nodal_handover_ev = nullptr;
    _impl->h_seg       = nullptr;
    _impl->h_sweep_accum = nullptr;
    _impl->hostfree_active = false;
    _impl->stream      = nullptr;
    _impl->own_stream  = nullptr;
    _impl->ready       = false;
    _impl->slot        = -1;
}

int CudaOuterSegment::slot() const { return _impl != nullptr ? _impl->slot : -1; }

bool CudaOuterSegment::useStream(void* stream) {
    if (_impl == nullptr || !_impl->ready) return false;
    _impl->stream = (stream != nullptr) ? static_cast<cudaStream_t>(stream)
                                        : _impl->own_stream;
    return true;
}

CudaOuterSegment::ProbeAddresses CudaOuterSegment::probeAddresses(int slot) const {
    ProbeAddresses a;
    if (_impl == nullptr || !_impl->ready) return a;
    if (slot < 0 || slot >= _impl->slot_count) return a;
    DeviceOuterProbe* probe = _impl->d_probes + slot;
    a.eigv     = &probe->eigv;
    a.residual = &probe->residual;
    a.negative = &probe->negative_flux;
    a.rayleigh  = &probe->rayleigh;
    a.nonfinite = &probe->nonfinite;
    a.halt      = _impl->d_halt;
    // Rev.7.1 Task 10 part 3: NON-NULL EXACTLY WHILE A HOST-FREE SEGMENT RUNS.
    //
    // It is the hook's whole signal.  A null accumulator means `you will be
    // observed after this launch, as you always were`; a non-null one means
    // `nobody is going to look until the segment ends, so keep the summary
    // yourself`.  Scoped to the segment rather than to the arm, because the
    // very same hook serves the per-outer arm two lines later on a deck that
    // cusps or a statepoint still inside its Wielandt warm-up.
    a.accum     = _impl->hostfree_active
                      ? static_cast<void*>(_impl->d_sweep_accum + slot)
                      : nullptr;
    a.valid    = true;
    return a;
}

bool CudaOuterSegment::available() const { return _impl != nullptr && _impl->ready; }

const std::string& CudaOuterSegment::status() const { return _impl->status; }

void CudaOuterSegment::setHooks(const OuterSegmentHooks& hooks) { _impl->hooks = hooks; }

OuterSegmentHooks CudaOuterSegment::hooks() const { return _impl->hooks; }

bool CudaOuterSegment::bindResidency(const OuterSegmentResidency& residency) {
    // Rev.7.1 Task 10 part 4: A CHANGED RESIDENCY INVALIDATES EVERY CAPTURED
    // BODY -- AND ONLY A CHANGED ONE.
    //
    // The WHILE's nodes carry the residency's addresses: the flux the H2D
    // targets, the psi updpsi writes, the jnet the nodal drive reads.  A re-bind
    // is the one event that can move them without moving the shape key, so it
    // has to throw the cache away.
    //
    // WHAT IT MUST NOT DO IS THROW IT AWAY AT EVERY ARM, and the first
    // measurement of this arm is why the test is here rather than an
    // unconditional clear: armOuterSegment runs at every SolveLoop and every
    // ReconvergeFlux, so kngr_238 re-binds about seventy times -- and with an
    // unconditional clear it re-instantiated seventy times too, for a residency
    // whose twelve pointers are the arena's and had not moved once.  The
    // comparison is a memcmp because the struct is trivially copyable and every
    // field is load-bearing; padding can only make it say "changed", which
    // costs an instantiation and cannot cost correctness.
    static_assert(std::is_trivially_copyable_v<OuterSegmentResidency>);
    if (std::memcmp(&_impl->residency, &residency, sizeof(residency)) != 0)
        _impl->while_cache.clear();
    _impl->residency       = residency;
    _impl->residency_bound = false;
    if (!_impl->ready || !_impl->is_bound) return false;
    if (!residency.valid || residency.flux == nullptr || residency.psi == nullptr ||
        residency.dtil == nullptr || residency.dhat == nullptr ||
        residency.xsnf == nullptr || residency.host_jnet == nullptr ||
        residency.host_flux == nullptr || residency.host_dhat == nullptr ||
        residency.host_psi == nullptr || residency.host_xsnf == nullptr ||
        residency.host_dtil == nullptr || residency.host_phis == nullptr)
        return false;
    if (residency.arena_slot < 0 || residency.arena_slot >= _impl->slot_count) return false;

    // The patch is a single-thread kernel rather than a host memcpy into the
    // table because the table lives in device memory and the five fields are
    // written INSIDE one CmfdOuterView; a host-side partial write would have to
    // download the struct, edit it and upload it, which is three transfers to
    // change five pointers.
    // CmfdOuterSlotTable::views is const because the BODIES only read it; the
    // allocation is this runner's own device buffer, so the cast is back to the
    // type the table was built from rather than away from an immutable one.
    k_cmfd_bind_resident<<<1, 1>>>(
        const_cast<cmfd::CmfdOuterView*>(_impl->binding.table.views),
        residency.arena_slot,
                                   residency.flux, residency.psi, residency.dtil,
                                   residency.dhat, residency.xsnf);
    cudaError_t rc = cudaGetLastError();
    if (rc == cudaSuccess) {
        // A DEVICE-WIDE synchronise, taken on this deck's Driver thread while
        // the siblings may be mid-capture.  It is the loudest of the unsafe
        // APIs in this tree and it is on the arming path of every deck, so it
        // gets the window like the allocations do (Task 18d).
        rasbery::AllocWindow _alloc_window("outer.bind.device_sync");
        rc = rasbery::xfer::deviceSync("CudaOuterGraph.cu:bindResidency", "device drain");
    }
    if (rc != cudaSuccess) {
        _impl->status = std::string("bind residency: ") + cudaGetErrorString(rc);
        return false;
    }
    _impl->binding.host_jnet = residency.host_jnet;
    // SEED THE DEVICE COPIES FROM THE HOST, ONCE PER ARM.
    //
    // THE BUG THIS FIXES, in the number that found it.  The segment writes dhat
    // at the END of an outer (step 8) and the sweep reads it in the MIDDLE
    // (step 2), so on the first outer after arming the sweep reads a dhat the
    // segment has not written yet -- and the H2D that used to fill it is
    // precisely what dhat_device_resident now elides.  With cudaMalloc memory
    // behind it, i-SMR CY01 came out at k_eff = -0.034501 with negative_flux on
    // 447 of its 516 outers.
    //
    // PER ARM, NOT ONCE PER PROCESS, because Driver.h:2656/2673 call
    // CMFD::resetDhat() between statepoints: the host zeroes _dhat, and a device
    // copy seeded only at the first arm would still be holding the previous
    // statepoint's.  armOuterSegment runs at every SolveLoop and ReconvergeFlux
    // entry, which is after every such reset.
    const std::size_t seed_dhat_bytes =
        static_cast<std::size_t>(_impl->binding.geom.nsurf) *
        static_cast<std::size_t>(_impl->binding.ng) * sizeof(double);
    const std::size_t seed_psi_bytes =
        static_cast<std::size_t>(_impl->binding.nxyz) * sizeof(double);
    if ((rc = rasbery::xfer::memcpy("CudaOuterGraph.cu:bindResidency", "seed dhat",
                            residency.dhat, residency.host_dhat, seed_dhat_bytes,
                            cudaMemcpyHostToDevice)) != cudaSuccess ||
        (rc = rasbery::xfer::memcpy("CudaOuterGraph.cu:bindResidency", "seed psi",
                            residency.psi, residency.host_psi, seed_psi_bytes,
                            cudaMemcpyHostToDevice)) != cudaSuccess) {
        _impl->status = std::string("seed residency: ") + cudaGetErrorString(rc);
        return false;
    }

    _impl->binding.host_flux = residency.host_flux;
    _impl->binding.host_xsnf   = residency.host_xsnf;
    // const_cast: the view hands xsnf out as read-only because the BODIES only
    // read it; the buffer is the sweep's own xs_xsnf and the segment writes it
    // for exactly the same reason issueSweepUploads does.
    _impl->binding.device_xsnf = const_cast<double*>(residency.xsnf);
    _impl->binding.host_dtil   = residency.host_dtil;
    _impl->binding.device_dtil = residency.dtil;
    _impl->binding.host_dhat   = residency.host_dhat;
    _impl->binding.host_psi    = residency.host_psi;
    _impl->binding.device_dhat = residency.dhat;
    _impl->binding.device_psi  = residency.psi;
    // Rev.7.1 Task 18-lite: the two ends of the canonical nodal set the stand-up
    // half could not know.  device_phis came from the arena at stand-up; the
    // FLUX half is the sweep's own phi, which only exists once the sweep arena
    // does, and Geometry::Phis is the host array a statepoint consumer reads.
    // The binding stays OFF until a backend adopts the set
    // (setCanonicalNodalBound); a rebind that handed over a different phi would
    // otherwise leave a backend holding an address this segment no longer uses.
    _impl->binding.device_flux     = residency.flux;
    _impl->binding.host_phis       = residency.host_phis;
    _impl->binding.canonical_nodal = false;
    // A rebind may hand over different device memory, so nothing that was
    // uploaded to the old buffers describes the new ones.
    _impl->resident_flux_generation = 0;
    _impl->resident_dtil_generation = 0;
    _impl->resident_xsnf.invalidate();
    _impl->residency_bound   = true;
    return true;
}

bool CudaOuterSegment::residencyBound() const { return _impl->residency_bound; }

// ---------------------------------------------------------------------------
// Rev.7.1 Task 18-lite: the canonical nodal set
// ---------------------------------------------------------------------------

CanonicalSlotBuffers CudaOuterSegment::canonicalNodalSet() const {
    const OuterSegmentBinding& b = _impl->binding;
    CanonicalSlotBuffers       out{};
    // ALL THREE OR NONE, checked here rather than by the caller.  A partial set
    // would pair the segment's jnet with the nodal arena's own flux -- two
    // different outer iterations, silently blended -- which is what
    // canonicalNodalSetIsCoherent refuses one layer down.  Answering with an
    // empty set is the honest version of `not yet`.
    if (!_impl->residency_bound || b.device_flux == nullptr || b.device_jnet == nullptr ||
        b.device_phis == nullptr)
        return out;
    out.flux = b.device_flux;
    out.jnet = b.device_jnet;
    out.phis = b.device_phis;
    return out;
}

void CudaOuterSegment::setCanonicalNodalBound(bool bound) {
    // A binding cannot be live without the set that would be bound; saying so
    // here keeps the runner's per-outer test to one field read plus one hook.
    _impl->binding.canonical_nodal = bound && canonicalNodalSet().shared();
}

bool CudaOuterSegment::canonicalNodalBound() const {
    return _impl->binding.canonical_nodal;
}

bool CudaOuterSegment::publishProbe(int slot, double eigv, double residual,
                                    bool negative_flux, bool rayleigh) {
    if (!_impl->ready || slot < 0 || slot >= _impl->slot_count) return false;
    DeviceOuterProbe probe{};
    probe.eigv          = eigv;
    probe.residual      = residual;
    probe.negative_flux = negative_flux ? 1u : 0u;
    probe.rayleigh      = rayleigh ? 1u : 0u;
    // nonfinite is raised by k_outer_refresh_inputs, which is the one place that
    // holds both values at once; material_changed cannot fire on an eligible
    // deck.  Writing either here would be guessing at another kernel's job.
    //
    // ==================================================================
    // ON THE SEGMENT'S STREAM, NOT THE DEFAULT ONE
    // ==================================================================
    //
    // THIS WAS A RACE AND IT COST A GATE.  It used to be a plain synchronous
    // cudaMemcpy, on the argument that "blocking" means the bytes are down
    // before the call returns.  For an H2D out of PAGEABLE host memory that is
    // not what cudaMemcpy promises: it returns once the source has been staged
    // into the driver's pinned buffer, and the DMA into device memory is
    // enqueued on the DEFAULT stream.  The segment runs on the sweep arena's
    // stream, which is created non-blocking, so nothing orders that DMA against
    // the next kernel this segment launches.
    //
    // It was invisible for as long as the probe's only device reader
    // (k_outer_refresh_inputs) sat at the far end of the body behind a sweep, a
    // updjnet and a synchronise.  W3 item 3 put a reader -- k_outer_publish_reigv
    // -- immediately after the sweep hook, and on kngr3 three outers out of ~600
    // read the PREVIOUS statepoint's converged eigenvalue: outer 0 of a segment
    // whose drive took the host loop, where this publish is the only writer of
    // the probe and the kernel two lines later is the only reader.  Those three
    // were the whole ON-vs-OFF divergence (11216 outers against 12017).
    //
    // Async on the segment's own stream and then drained makes the copy both
    // stream-ordered and still-blocking, which is what every caller here already
    // believed it was getting.
    cudaStream_t stream = _impl->stream;
    if (stream == nullptr) {
        const cudaError_t rc =
            rasbery::xfer::memcpy("CudaOuterGraph.cu:publishProbe", "probe (streamless)",
                           _impl->d_probes + slot, &probe, sizeof(probe),
                           cudaMemcpyHostToDevice);
        return rc == cudaSuccess;
    }
    cudaError_t rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:publishProbe", "probe",
                                         _impl->d_probes + slot, &probe, sizeof(probe),
                                         cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return false;
    rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:publishProbe", "probe drain", stream);
    return rc == cudaSuccess;
}

bool CudaOuterSegment::republishAfterHostSweep(int slot, double eigv, double residual,
                                               bool negative_flux, bool rayleigh) {
    if (!publishProbe(slot, eigv, residual, negative_flux, rayleigh)) return false;
    // SAY THAT THE HOST FINISHED IT.  The body has to know, because the verdict
    // kernel's halt no-opped every kernel enqueued between it and this call --
    // updjnet among them -- and taking the halt off does not un-skip them.
    _impl->sweep_host_continued = true;
    // THE HALT COMES OFF LAST, and that ordering is the point: until the probe
    // holds the finished drive's numbers, an outer that resumed would compute
    // its convergence from the half drive the verdict kernel saw.  publishProbe
    // is a blocking H2D, so by the time this runs the new numbers are down.
    //
    // SAME STREAM, SAME REASON as publishProbe just above: a default-stream H2D
    // is not ordered against the non-blocking stream the body's kernels run on,
    // and this one lifts the gate every one of them tests.
    const std::uint32_t clear = 0u;
    cudaStream_t        stream = _impl->stream;
    if (stream == nullptr) {
        const cudaError_t rc = rasbery::xfer::memcpy(
            "CudaOuterGraph.cu:republishAfterHostSweep", "halt (streamless)",
            _impl->d_halt + slot, &clear, sizeof(clear), cudaMemcpyHostToDevice);
        return rc == cudaSuccess;
    }
    cudaError_t rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:republishAfterHostSweep",
                                         "halt", _impl->d_halt + slot, &clear,
                                         sizeof(clear), cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return false;
    rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:republishAfterHostSweep", "halt drain",
                            stream);
    return rc == cudaSuccess;
}

bool rasberyBindOuterResidency(const OuterSegmentResidency& residency) {
    // THE RESIDENCY CARRIES ITS OWN SLOT, so this needs no second index space:
    // the Driver filled arena_slot from the CMFD slot it holds, and that is the
    // runner it must reach.
    return rasberyOuterSegment(residency.arena_slot).bindResidency(residency);
}

bool rasberyPublishOuterProbe(int slot, double eigv, double residual, bool negative_flux,
                              bool rayleigh) {
    return rasberyOuterSegment(slot).publishProbe(slot, eigv, residual, negative_flux,
                                                  rayleigh);
}

bool rasberySyncSegmentStream(OuterSegmentStream stream) {
    if (stream == nullptr) return true;
    const cudaError_t rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:rasberySyncSegmentStream",
                                              "segment", static_cast<cudaStream_t>(stream));
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

void CudaOuterSegment::bind(const OuterSegmentBinding& binding) {
    _impl->binding = binding;
    // A binding is usable only when it can ADDRESS the CMFD bodies' slot views
    // and knows the mesh they index; everything else in it has a meaningful zero.
    _impl->is_bound = binding.table.views != nullptr && binding.table.slot_count > 0 &&
                      binding.geom.nxyz > 0 && binding.geom.nsurf > 0;
}

bool CudaOuterSegment::bound() const { return _impl->is_bound; }

OuterSegmentRefusal CudaOuterSegment::refusal(int batch_width, bool fractional_rods,
                                              bool critical_search,
                                              bool slot_admitted) const {
    if (!outerGpuEnabled()) return OuterSegmentRefusal::FeatureOff;
    OuterSegmentEligibility e{};
    e.runner_available = available() ? 1 : 0;
    // The arena view is what makes a segment addressable; a runner that
    // initialised without one has nothing to drive.
    e.arena_reserved   = (_impl->arena.slot_views != nullptr && _impl->arena.phases != nullptr)
                             ? 1
                             : 0;
    e.bound            = _impl->is_bound ? 1 : 0;
    e.batch_width      = batch_width;
    // THE WIDTH THIS RUNNER WAS STOOD AT, not the one the run asked for.  An
    // uninitialised runner reports 0, which makes any batch wider than it --
    // but `runner_available` is ranked above and answers first, so the receipt
    // still says `no_runner` rather than blaming the batch.
    e.arena_slots      = _impl->slot_count;
    e.slot_admitted    = slot_admitted ? 1 : 0;
    e.fractional_rods  = fractional_rods ? 1 : 0;
    e.critical_search  = critical_search ? 1 : 0;
    e.residency_bound  = _impl->residency_bound ? 1 : 0;
    e.have_sweep_hook  = _impl->hooks.enqueue_cmfd_sweep != nullptr ? 1 : 0;
    e.have_nodal_hook  = _impl->hooks.enqueue_nodal_drive != nullptr ? 1 : 0;
    e.have_cusping_hook = _impl->hooks.apply_cusping != nullptr ? 1 : 0;
    return outerSegmentRefusal(e);
}

namespace {

/// Bytes of one psi + dhat mirror pair, for the receipt arithmetic.
std::size_t mirrorPairBytes(const OuterSegmentBinding& b) {
    return static_cast<std::size_t>(b.nxyz) * sizeof(double) +
           static_cast<std::size_t>(b.geom.nsurf) * static_cast<std::size_t>(b.ng) *
               sizeof(double);
}

} // namespace

bool CudaOuterSegment::runSegment(const OuterSegmentScalars& scalars, int batch_width,
                                  bool fractional_rods, bool critical_search,
                                  OuterSegmentResume& resume) {
    const OuterSegmentRefusal why =
        refusal(batch_width, fractional_rods, critical_search, scalars.slot_admitted != 0);
    if (why != OuterSegmentRefusal::None) {
        bump(counters().refusals[static_cast<int>(why)]);
        return false;
    }
    // AND IT MUST BE THIS RUNNER'S SLOT.  One runner per slot means an index
    // that does not match is a caller that reached for the wrong object, and
    // running it anyway would drive somebody else's residency.
    if (scalars.slot < 0 || scalars.slot >= _impl->slot_count ||
        scalars.slot != _impl->slot) {
        bump(counters().refusals[static_cast<int>(OuterSegmentRefusal::LaunchFailed)]);
        return false;
    }

    // Rev.7.1 W3 item 1: everything a host body does from here to the return --
    // through a hook, through the nodal drive, anywhere -- is charged to the
    // segment.  Placed AFTER the two refusals above, so a segment that never
    // started cannot make the in-segment claim look worse than it is.
    const hostouter::SegmentScope host_body_scope;

    Impl&                      m      = *_impl;
    const bool                 m_hooks_synchronize = _impl->hooks.sweep_synchronizes;
    const OuterSegmentBinding& bound_ = m.binding;
    // The stream-ordered sweep is only stream-ordered if BOTH halves are here:
    // a hook that enqueues and never publishes its observation would leave the
    // flux the nodal drive reads one outer behind.
    const bool                 stream_sweep =
        !m_hooks_synchronize && m.hooks.finish_cmfd_sweep != nullptr;
    // THE SYNCHRONISING SWEEP HOOK FORCES A SEGMENT OF ONE.
    //
    // BICGCMFD::drive drains its stream and copies the flux back, so the host
    // has already observed outer i by the time outer i+1 could be enqueued.
    // Enqueueing further outers under that hook would issue kernels whose
    // halt state the transition has not been able to publish yet -- a different
    // trajectory, not a faster one.  One outer per segment is the honest
    // description of what this can do until the sweep is stream-ordered, and it
    // is why `device_outers` equals the host outer count rather than exceeding
    // `segment_launches` by the budget.
    const unsigned int         budget =
        stream_sweep ? outerSegmentBudget() : 1u;
    const int                  slot   = scalars.slot;
    const DevicePhaseQueue     queue  = singleSlotQueue(slot);

    // The per-segment-CONSTANT half of CmfdOuterInputs.  eigv and residual are
    // seeded from the entry state and then refreshed on the device every outer;
    // nothing else in this struct may move inside a segment, which is exactly
    // what makes one upload enough (see DeviceOuterProbe).
    //
    // search_pending / search_is_boron stay ZERO because a search deck is
    // refused outright (OuterSegmentEligibility::critical_search).  Zero is not
    // a shortcut here -- it is the only value that cannot be mistaken for a real
    // search state if that refusal is ever lifted without a device Search phase
    // to feed it.
    cmfd::CmfdOuterInputs inputs{};
    inputs.eigv            = scalars.eigv;
    inputs.residual        = scalars.residual;
    inputs.keff_tol        = scalars.keff_tol;
    inputs.flux_tol        = scalars.flux_tol;
    inputs.max_outer_iter  = scalars.max_outer_iter;
    inputs.xe_pending      = scalars.xe_pending;
    inputs.xe_interim_l2   = scalars.xe_interim_l2;
    inputs.xe_once_mode    = scalars.xe_once_mode;
    inputs.xe_budget_probe = scalars.xe_budget_probe;
    inputs.th_pending      = scalars.th_pending;
    inputs.search_pending  = 0;
    inputs.search_is_boron = 0;

    // --- arm the segment -----------------------------------------------------
    //
    // These three uploads are the ONLY H2D of a segment and they happen before
    // the first outer, never between them.  Everything after this point is
    // stream-ordered device work until the single sync at the bottom.
    DeviceOuterSegmentState seg{};
    deviceOuterSegmentReset(seg, budget);

    DeviceOuterProbe probe{};
    probe.eigv     = scalars.eigv;
    probe.residual = scalars.residual;

    const std::uint32_t clear_halt = 0u;

    // --- Rev.7.1 Task 18-lite: giving the canonical nodal set back ----------
    //
    // WHAT THE HOST IS OWED WHEN THE BINDING STOPS.  While it is live the device
    // nodal reads and writes the arena's jnet and phis and nothing comes back;
    // the moment it stops -- at the segment exit, or mid-segment on an outer
    // whose drive falls back to the CPU body -- two things are owed, and they
    // are not interchangeable.  The OWNERSHIP, so the next drive uploads again
    // rather than eliding against a host array somebody has since rewritten.
    // And, at a segment exit only, the BYTES: Geometry::Jnet and Geometry::Phis
    // have four host consumers outside a segment and all of them read.
    //
    // `stream_ordered` is false on the failure paths, where the segment is
    // abandoning a stream whose state it can no longer reason about: there the
    // mirror is a blocking copy and its errors are swallowed, because a mirror
    // that failed must not turn a host fallback into a hard stop.
    /// Rev.7.1 W3 item 3: declare who writes the nodal drive's reigv slot.
    ///
    /// Idempotent and latched, so the common case -- the same answer on every
    /// outer of a segment -- is one branch and no call, and the backend never
    /// sees a flip it did not need.
    auto setReigvDevice = [&](bool on) {
        if (m.reigv_device_claimed == on) return;
        m.reigv_device_claimed = on;
        if (m.hooks.nodal_reigv_mode != nullptr)
            m.hooks.nodal_reigv_mode(m.hooks.ctx, on ? 1 : 0);
    };

    auto releaseCanonicalNodal = [&](bool stream_ordered) {
        // FIRST, AND OUTSIDE THE EARLY RETURN.  The reigv claim is not tied to
        // the canonical binding's liveness -- it is a second sticky declaration
        // on the same backend -- so clearing it has to happen on every path
        // through here, including the one where there was no binding to give
        // back.  Every failure path in this function calls this lambda, which is
        // why it and not five separate resets.
        setReigvDevice(false);
        if (!m.canonical_nodal_live) return;
        m.canonical_nodal_live = false;
        if (!stream_ordered) {
            const std::size_t surf_bytes =
                static_cast<std::size_t>(bound_.geom.nsurf) *
                static_cast<std::size_t>(bound_.ng) * sizeof(double);
            if (bound_.host_jnet != nullptr && bound_.device_jnet != nullptr)
                rasbery::xfer::memcpy("CudaOuterGraph.cu:releaseCanonicalNodal", "jnet",
                               bound_.host_jnet, bound_.device_jnet, surf_bytes,
                               cudaMemcpyDeviceToHost);
            if (bound_.host_phis != nullptr && bound_.device_phis != nullptr)
                rasbery::xfer::memcpy("CudaOuterGraph.cu:releaseCanonicalNodal", "phis",
                               bound_.host_phis, bound_.device_phis, surf_bytes,
                               cudaMemcpyDeviceToHost);
            cudaGetLastError();
        }
        if (m.hooks.canonical_nodal_mode != nullptr)
            m.hooks.canonical_nodal_mode(m.hooks.ctx, 0);
    };

    auto launchFailed = [&](const char* what, cudaError_t rc) {
        std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] %s: %s -- falling back to the host "
                             "outer for this iteration\n",
                     what, cudaGetErrorString(rc));
        bump(counters().refusals[static_cast<int>(OuterSegmentRefusal::LaunchFailed)]);
        releaseCanonicalNodal(false);
        return false;
    };
    // A hook returning false is not a CUDA error, so it does not go through
    // launchFailed -- cudaGetErrorString(cudaSuccess) is "no error", which is
    // the most misleading thing a diagnostic could say here.
    auto hookFailed = [&](const char* what) {
        std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] %s refused to enqueue -- falling "
                             "back to the host outer for this iteration\n",
                     what);
        bump(counters().refusals[static_cast<int>(OuterSegmentRefusal::LaunchFailed)]);
        releaseCanonicalNodal(false);
        return false;
    };

    // =======================================================================
    // Rev.7.1 W3 item 4: THE OPERATOR UPLOADS A SEGMENT STAGES ONCE
    // =======================================================================
    //
    // WHICH HOST INPUTS CAN MOVE INSIDE A SEGMENT, PROVED RATHER THAN ASSUMED.
    // The body makes exactly four host calls -- the sweep hook, the sweep
    // observation, the nodal drive and apply_cusping -- so the writers reachable
    // from inside a segment are the writers those four can reach:
    //
    //   Geometry::Phif (flux)  MOVES EVERY OUTER.  BICGCMFD::drive's host loop
    //       writes it directly, and absorbSweepLaunch adopts the device phi into
    //       it on the enqueued path.  Its elision is therefore a genuinely
    //       per-outer question and stays in the body, re-decided from the
    //       generation the drive that just ran left behind.
    //
    //   XSSet::_xs (xsnf)      MOVES ONLY IF CUSPING FIRES.  The device-arm
    //       writers of host _xs -- UpdateFlatXS and UpdateEquilibriumXenon --
    //       belong to the Xe and search steps of the HOST LADDER, and reaching
    //       either of them means the segment has already exited (they are
    //       DevicePhase::Xenon / ::Search, and every phase that is not
    //       Outer -> Outer sets exit_segment).  The boron trial commit that
    //       produced the sp1-outer-28 divergence is one of those: it happens
    //       BETWEEN segments, which is why staging at entry still sees it.
    //       Inside the body only XSSet::ApplyRodCusping writes _xs.
    //
    //   CMFD::_dtil            MOVES ONLY IF CUSPING FIRES.  upddtil() is its
    //       only writer, and it is called from SolveLoop's entry (before the
    //       delegation) and from the cusping hook.  Driver.h:2718/2735's
    //       resetDhat calls are in Drive(), outside SolveLoop entirely.
    //
    // So the two that cannot move are staged here, once, and re-staged by the
    // cusping branch -- which is the one place they can move -- and the one that
    // can move every outer stays where it was.  What this buys is not bytes (the
    // elisions already removed those) but a body whose H2D NODE SET is fixed:
    // Task 10's conditional WHILE cannot capture a loop whose set of memcpy
    // nodes is decided by a host memcmp each iteration.
    //
    // THE MEMCMP IS STILL THE GATE, and per SEGMENT rather than per outer.  A
    // generation cannot be substituted here for the reason invariant 5 gives:
    // XSSet::hoststateGeneration() means `the device mirror of _xs is stale`,
    // not `the host bytes of _xs changed`, and the arms that write _xs from the
    // device deliberately do not bump it.
    auto stageXsnf = [&]() -> bool {
        if (bound_.host_xsnf == nullptr || bound_.device_xsnf == nullptr) return true;
        const std::size_t xsnf_count = static_cast<std::size_t>(bound_.nxyz) *
                                       static_cast<std::size_t>(bound_.ng);
        if (m.resident_xsnf.matches(bound_.host_xsnf, xsnf_count)) {
            bump(counters().xsnf_uploads_elided);
            return true;
        }
        const std::size_t xsnf_bytes = xsnf_count * sizeof(double);
        const cudaError_t r = rasbery::xfer::memcpyAsync(
            "CudaOuterGraph.cu:stageXsnf", "xsnf", bound_.device_xsnf, bound_.host_xsnf,
            xsnf_bytes, cudaMemcpyHostToDevice, m.stream);
        if (r != cudaSuccess) return launchFailed("upload xsnf", r);
        bump(counters().flux_sync_bytes, xsnf_bytes);
        // COMMITTED AT THE ISSUE.  The general rule (CudaTransferMirror.h) is to
        // commit only after the copy has landed; the shadow records the bytes
        // this copy was HANDED, and the only host writer that can run before the
        // copy drains is ApplyRodCusping -- which re-stages through this same
        // lambda, so a writer between the two is caught rather than elided.
        m.resident_xsnf.commit(bound_.host_xsnf, xsnf_count);
        return true;
    };

    // upddtil() is the only writer of _dtil and it runs once per SolveLoop
    // entry, plus once for every cusping that fires.  On a still deck this copy
    // happens once per segment and then never again.
    auto stageDtil = [&](const OuterSegmentLiveState& live) -> bool {
        if (bound_.host_dtil == nullptr || bound_.device_dtil == nullptr) return true;
        const bool dtil_current = m.hooks.read_live_state != nullptr &&
                                  m.resident_dtil_generation != 0 &&
                                  m.resident_dtil_generation == live.dtil_generation;
        if (dtil_current) {
            bump(counters().dtil_uploads_elided);
            return true;
        }
        const std::size_t dtil_bytes = static_cast<std::size_t>(bound_.geom.nsurf) *
                                       static_cast<std::size_t>(bound_.ng) * sizeof(double);
        const cudaError_t r = rasbery::xfer::memcpyAsync(
            "CudaOuterGraph.cu:stageDtil", "dtil", bound_.device_dtil, bound_.host_dtil,
            dtil_bytes, cudaMemcpyHostToDevice, m.stream);
        if (r != cudaSuccess) return launchFailed("upload dtil", r);
        bump(counters().flux_sync_bytes, dtil_bytes);
        m.resident_dtil_generation = live.dtil_generation;
        return true;
    };

    cudaError_t rc;
    // The host mirror of the exit word is the loop's stopping condition, and it
    // survives the previous segment.  Clearing it here rather than trusting the
    // first D2H is what stops a converged segment's exit from breaking the NEXT
    // segment out at its second outer.
    if (m.h_seg != nullptr) *m.h_seg = seg;
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "segment state",
                                  m.d_segments + slot, &seg, sizeof(seg),
                                  cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
        return launchFailed("upload segment state", rc);
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "probe",
                                  m.d_probes + slot, &probe, sizeof(probe),
                                  cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
        return launchFailed("upload probe", rc);
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "halt clear",
                                  m.d_halt + slot, &clear_halt, sizeof(clear_halt),
                                  cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
        return launchFailed("clear halt", rc);
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "outer inputs",
                                  m.d_inputs + slot, &inputs, sizeof(inputs),
                                  cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
        return launchFailed("upload outer inputs", rc);

    // prev_inner lives in DeviceSlotState::previous_eigv and the convergence
    // body reads it through cmfdLoadOuterState.  The host owns it BETWEEN
    // segments (Driver.h's `prev_inner` local); inside one, the device does.
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "prev_inner",
                                  &m.arena.states[slot].previous_eigv,
                                  &scalars.prev_inner, sizeof(double),
                                  cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
        return launchFailed("upload prev_inner", rc);

    // flux_stall lives in DeviceSlotState::flux_stall and is read through the
    // same cmfdLoadOuterState.  Uploaded for exactly prev_inner's reason: it is
    // a SolveLoop local the device carries INSIDE a segment and the host owns
    // BETWEEN segments.  Left unseeded, the device counted from the previous
    // segment's leftovers and could not stop at the outer the host's
    // limit-cycle test would have stopped at -- so a budget-8 segment ran up to
    // eight times as far past a stalling trial point as the host ever would.
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "flux_stall",
                                  &m.arena.states[slot].flux_stall, &scalars.flux_stall,
                                  sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                  m.stream)) != cudaSuccess)
        return launchFailed("upload flux_stall", rc);

    // --- the staged operator uploads, once -----------------------------------
    //
    // Issued HERE, in the arm block, so the claim two comments up -- `the only
    // H2D of a segment happen before the first outer` -- is true of the operator
    // as well as of the control packet.  The flux is the one exception and it
    // says why for itself, at the top of the body.
    {
        OuterSegmentLiveState arm_live;
        if (m.hooks.read_live_state != nullptr) m.hooks.read_live_state(m.hooks.ctx, arm_live);
        if (!stageXsnf()) return false;
        if (!stageDtil(arm_live)) return false;
    }

    // --- prologue: the nodal constants, once ---------------------------------
    //
    // ONCE PER SEGMENT, NOT ONCE PER OUTER, and that is Driver.h's shape:
    // Nodal::updateConstant re-runs when the macro-XS move, and the only in-outer
    // XS write is rod cusping -- which an eligible slot cannot do.
    //
    // =====================================================================
    // THIS IS INERT ON THE PRODUCTION PATH, AND IT MUST STAY THAT WAY UNTIL
    // TWO THINGS ARE FIXED.  W3 item 1 audit, 2026-08-29.
    // =====================================================================
    //
    // OuterSegmentScalars::run_nodal_constants has no setter anywhere in src/,
    // so this branch never runs.  That is not an oversight to correct by adding
    // one; the phase cannot produce a correct answer from where it stands:
    //
    //  (1) ITS INPUTS ARE NOT THERE.  k_nodal_update_constant reads
    //      DeviceSlotView::xs (SlotRegion::Xs) and DeviceSlotView::constant_xs
    //      (SlotRegion::ConstantXs) out of the PHYSICS arena.  rasberyStandUpOuterSegment
    //      imports exactly five geometry regions -- Lklr, Idirlr, Hmesh, Vol,
    //      Albedo -- and nothing in src/ ever writes either slot region on the
    //      production path.  The kernel would build the nine coefficient arrays
    //      out of an uninitialised cudaMallocFromPoolAsync block.
    //
    //  (2) ITS OUTPUT IS AT THE WRONG OFFSET FOR THE ONLY READER.  The reader is
    //      the nodal backend, which addresses the nine packed arrays as
    //      `d.ndev_dbl + d.n_off_consts + i*ndg` with i=7 -> diagD and i=8 ->
    //      diagDI (CudaXsReconBackend.cu:2138-2140, :2182-2183, :2548-2549, and
    //      the h_const staging at :496-500).  The arena's packing is the
    //      OPPOSITE: NodalConstSlot has kNcDiagDI = 7 and kNcDiagD = 8
    //      (CudaNodalConstantKernel.h:135-137), which test/nodal_constant_gpu_replay.cpp:332
    //      and tools/test_nodal_constant_gpu_contract.py:165 both pin on
    //      purpose.  Binding one to the other swaps D and 1/D on every node and
    //      direction, and both arrays are finite and plausible, so nothing
    //      crashes -- the answer is simply wrong.
    //
    // AND EVEN WITH BOTH FIXED IT IS A CLASS N1 TRANSITION (CUDA exp differs
    // from glibc by 1 ulp on 3.34% of arguments, NodalConstantKernel.h), so it
    // may not be turned on inside a gate that requires ON == OFF.  W3 item 1
    // therefore removed the host call the other way round -- by not running the
    // sweep when its inputs have not moved, which is bit-exact by construction
    // (Nodal::updateConstantsIfMoved).  The phase and its replay tests stay for
    // Task 22, where Gate A/B admits N1.
    if (scalars.run_nodal_constants) {
        if ((rc = enqueueNodalUpdateConstant(m.arena, queue, bound_.geometry, bound_.nxyz,
                                             bound_.ng, m.stream)) != cudaSuccess)
            return launchFailed("enqueue nodal constants", rc);
    }

    // --- the per-step tracer (RASBERY_OUTER_TRACE) ---------------------------
    //
    // IT HASHES DEVICE MEMORY, and that is the whole reason it exists here
    // rather than at the Driver.h call site.  The per-outer tracer hashes the
    // HOST arrays on the argument that this arm mirrors psi and dhat back and
    // bridges jnet every outer -- which stopped being true when the mirrors
    // moved to the segment exit (01b599b) and the bridge was dropped for the
    // canonical nodal binding (e5f53bc).  A host hash of this arm is now a hash
    // of whatever the last mirror left behind, which reads as a difference at
    // every step and localises nothing.
    //
    // IT SYNCHRONISES AND COPIES, PER STEP.  That is a real cost and it is paid
    // only under the environment gate: `trace_steps` is a cached bool, so an
    // untraced run pays one predictable branch per step and no allocation.
    const bool trace_steps = outertrace::active();
    auto traceHash = [&](const double* dev, std::size_t n) -> std::uint64_t {
        if (dev == nullptr || n == 0) return 0ull;
        if (rasbery::xfer::streamSync("CudaOuterGraph.cu:traceHash", "drain", m.stream) !=
            cudaSuccess) { cudaGetLastError(); return 0ull; }
        if (m.trace_scratch.size() < n) m.trace_scratch.resize(n);
        if (rasbery::xfer::memcpy("CudaOuterGraph.cu:traceHash", "scratch",
                           m.trace_scratch.data(), dev, n * sizeof(double),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            return 0ull;
        }
        return outertrace::hashDoubles(m.trace_scratch.data(), n);
    };
    const std::size_t trace_nn  = static_cast<std::size_t>(bound_.nxyz) *
                                  static_cast<std::size_t>(bound_.ng);
    const std::size_t trace_nsg = static_cast<std::size_t>(bound_.geom.nsurf) *
                                  static_cast<std::size_t>(bound_.ng);
    /// The eigenvalue this outer's sweep published, read from the device probe
    /// rather than from the hook context: the runner does not hold Driver.h's
    /// `eigv` local, and the probe is where the sweep's verdict kernel wrote it.
    auto traceProbeEigv = [&]() -> double {
        DeviceOuterProbe p{};
        if (rasbery::xfer::streamSync("CudaOuterGraph.cu:traceProbeEigv", "drain", m.stream) !=
            cudaSuccess) { cudaGetLastError(); return 0.0; }
        if (rasbery::xfer::memcpy("CudaOuterGraph.cu:traceProbeEigv", "probe", &p,
                           m.d_probes + slot, sizeof(p), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            cudaGetLastError();
            return 0.0;
        }
        return p.eigv;
    };

    // =======================================================================
    // Rev.7.1 Task 10 part 3: IS THIS SEGMENT HOST-FREE?
    // =======================================================================
    //
    // ASKED ONCE, HERE, AND EVERY TERM IS FROZEN FOR THE SEGMENT.  That is the
    // whole safety argument, and it is the argument Task 18-lite got wrong the
    // other way round (an arm-scope canonical-nodal answer, which a rod search
    // moving the bank INSIDE SolveLoop falsified a few outers later).  So each
    // term below carries the fact that makes a segment-scope answer sound:
    //
    //   stream_sweep        a property of the arm, decided at armOuterSegment.
    //   sweep_will_enqueue  canEnqueueDrive(): `_wiel_sweep >= 5`, and
    //                       _wiel_sweep only GROWS except at resetIteration(),
    //                       which is a statepoint boundary and therefore outside
    //                       any segment.  True here means true for every outer.
    //   canonical_nodal     Nodal::TryDriveGpu's own predicate.  It turns on
    //                       Geometry::rod_fraction, which moves only when the
    //                       rod BANK moves -- the Search phase, which is not an
    //                       Outer -> Outer transition and so ends the segment.
    //   cusping_quiescent   the same fact, one level up: with no fractional node
    //                       and an empty carry-over set, ApplyRodCusping returns
    //                       false without writing anything, for every outer of
    //                       this segment.  i-SMR CY02 fails this and keeps the
    //                       per-outer arm, which is the intended outcome.
    //   !trace_steps        the per-step tracer synchronises and copies by
    //                       construction; a traced run is a diagnostic run.
    //
    // THE THREE HOOKS ARE ASKED FOR TOGETHER because they are one mechanism:
    // without the deferred observation there is nothing to defer, without the
    // halt gate the overrun outers would re-solve the nodal problem on their own
    // output, and without the quiescence question cusping could blend cross
    // sections off a leakage that was never produced.
    OuterHostFreeRefusal hostfree_why = OuterHostFreeRefusal::None;
    {
        OuterSegmentLiveState hf_live;
        if (m.hooks.read_live_state != nullptr) m.hooks.read_live_state(m.hooks.ctx, hf_live);
        if (!outerHostFreeEnabled())
            hostfree_why = OuterHostFreeRefusal::FeatureOff;
        else if (m.hooks.finish_cmfd_sweep_deferred == nullptr ||
                 m.hooks.cusping_quiescent == nullptr ||
                 m.hooks.nodal_halt_gate == nullptr)
            hostfree_why = OuterHostFreeRefusal::NoHooks;
        else if (!stream_sweep)
            hostfree_why = OuterHostFreeRefusal::NotStreamSweep;
        else if (!hf_live.sweep_will_enqueue)
            hostfree_why = OuterHostFreeRefusal::SweepWontEnqueue;
        else if (!bound_.canonical_nodal || m.hooks.canonical_nodal_eligible == nullptr ||
                 m.hooks.canonical_nodal_eligible(m.hooks.ctx) == 0)
            hostfree_why = OuterHostFreeRefusal::NoCanonicalNodal;
        else if (m.hooks.cusping_quiescent(m.hooks.ctx) == 0)
            hostfree_why = OuterHostFreeRefusal::CuspingLive;
        // Rev.7.1 Task 10 part 3: THE TRACER DOES NOT REFUSE THIS ARM, and that
        // is a debuggability decision taken on purpose.  outertrace only READS
        // -- it synchronises the segment stream and copies device memory to
        // hash it -- so a traced host-free segment computes exactly what an
        // untraced one computes, with extra observation around it.  Refusing
        // here would make the one arm whose whole point is that the host never
        // looks the one arm nobody can look at, and localising a trajectory
        // difference is precisely what the tracer exists for.  (It does mean a
        // traced run cannot prove the ABSENCE of a race, because its drains
        // would mask one; the ON == OFF gate is run untraced for that reason.)
    }
    const bool hostfree = hostfree_why == OuterHostFreeRefusal::None;
    const bool keep_exit_obs = hostfree && !outerHostFreeFull();
    // WP14.  Read ONCE per segment into a local, because the two elisions below
    // are a pair: a segment that took the shorter accumulator read must also be
    // the one that skips the second synchronise, and a static read twice would
    // let a mid-run reconfiguration split them.  (It cannot today -- the reader
    // caches -- and that is exactly why the local costs nothing.)
    const bool segment_v2 = outerSegmentV2Enabled();
    // Set by the loop below when it BREAKS on an exit observation.  Its meaning
    // is narrow and load-bearing: cudaStreamSynchronize(m.stream) returned
    // success and nothing has been enqueued on m.stream since, so the stream is
    // empty and `m.h_seg` holds the bytes the last committed outer's transition
    // wrote.  Every V2 elision is argued from this one fact.
    bool observed_exit = false;
    if (!hostfree) bump(counters().hostfree_refusals[static_cast<int>(hostfree_why)]);

    // THE FLAG THE SWEEP HOOK READS, AND THE ONE THING THAT MUST NOT LEAK.
    //
    // probeAddresses hands the arena an accumulator only while this is up, so a
    // failure path that left it up would tell the NEXT segment's per-outer arm
    // to keep a summary nobody reads -- and, worse, would leave the halt gate
    // installed with no segment to clear the halt word.  Both come down in a
    // destructor rather than at five returns.
    struct HostFreeScope {
        Impl* impl = nullptr;
        ~HostFreeScope() {
            if (impl != nullptr) impl->hostfree_active = false;
        }
    } hostfree_scope;
    if (hostfree) {
        bump(counters().hostfree_segments);
        m.hostfree_active   = true;
        hostfree_scope.impl = &m;
        // INSTALLED AND LEFT INSTALLED, and that is a performance decision with
        // a correctness consequence.  The pair is a nodal GRAPH KEY, so taking
        // it off at every segment exit would re-instantiate the captured drive
        // twice per segment -- 6422 instantiations on kngr_238.  Left on, the
        // gate is inert exactly while `d_halt[slot]` is zero, which is why the
        // exit below now clears that word: a halt allowed to outlive its segment
        // would mask the HOST outers that follow it.
        m.hooks.nodal_halt_gate(m.hooks.ctx, static_cast<const void*>(m.d_halt), slot);
        // The handover event, created on the first host-free segment of the run
        // and reused for every outer after it.  cudaEventDisableTiming because
        // nothing measures it: it exists to carry a dependency, and the timing
        // variant costs a device-side timestamp per record.
        if (m.nodal_handover_ev == nullptr) {
            // WP19.  Created once, on the first host-free segment -- stand-up,
            // which is the window a sibling lane's capture is open in.
            rasbery::AllocWindow _alloc_window("outer.handover.event");
            if ((rc = cudaEventCreateWithFlags(&m.nodal_handover_ev,
                                               cudaEventDisableTiming)) != cudaSuccess)
                return launchFailed("create the nodal handover event", rc);
        }
        if (m.d_sweep_accum != nullptr &&
            (rc = cudaMemsetAsync(m.d_sweep_accum + slot, 0,
                                  sizeof(*m.d_sweep_accum), m.stream)) != cudaSuccess)
            return launchFailed("clear the sweep accumulator", rc);
    }

    // =======================================================================
    // Rev.7.1 Task 10 part 3: THE TAIL OF AN OUTER, AS ONE NAMED THING
    // =======================================================================
    //
    // WHY IT IS A LAMBDA NOW AND WAS STRAIGHT-LINE BEFORE.  A host-free segment
    // learns that the device abandoned a sweep only at its EXIT, and the outer
    // that was abandoned still owes the host everything from the re-issued
    // updjnet onwards: the nodal drive, upddhat, the refresh, the decision and
    // the transition that commits it.  That is exactly this block.  Written
    // twice it would be two spellings of the most delicate sequence in the tree
    // -- and the three-in-twelve-thousand path is precisely the one nobody
    // would notice drifting -- so it is written once and called from both.
    //
    // NOTHING IN IT CHANGED.  The extraction is mechanical: the same statements
    // in the same order, with the four values the head decided passed in
    // instead of read from enclosing scope.
    auto runOuterTail = [&](const unsigned int i, double* const reigv_slot,
                            const bool canonical_now, const bool bridge_jnet,
                            const std::size_t jnet_bytes) -> bool {
            // ==================================================================
            // THE STEP THE HALT SWALLOWED
            // ==================================================================
            //
            // cmfd_sweep_verdict raises the segment's halt when the drive did not
            // finish on the device -- sweep state 0 (the launch's slot budget ran
            // out) or 2 (the Wielandt gamma degenerated) -- so that the rest of the
            // body does not correct the current from a half sweep.  That is right,
            // and it is why the halt is there.  What it did not account for is that
            // updjnet is enqueued BEHIND the verdict on the same stream: it is
            // therefore already in flight as a no-op by the time the host finishes
            // the drive, and republishAfterHostSweep taking the halt off cannot
            // un-skip it.  Everything after that point in the body -- upddhat, the
            // refresh, the decision, the transition -- then ran normally, against
            // the jnet of the PREVIOUS outer.
            //
            // The host loop has no such window: drive() returns only when the drive
            // is over, and updjnet runs after it (Driver.h:1569).  So the repair is
            // to run the step where the host runs it, on the outers where the device
            // could not.
            //
            // MEASURED, NOT HYPOTHETICAL.  kngr_238, budget 8: three outers out of
            // 11,993 take this path -- statepoint 23 outer 208, 25/113, 29/19 -- and
            // before this block those three were the entire remaining ON-vs-OFF
            // divergence.  Statepoints 1..22 were already bit-identical; 23 onward
            // were not, and the per-step trace named `updjnet` at exactly 23/208
            // with a hash equal to the previous outer's nodal jnet, which is what a
            // step that did not run looks like.
            //
            // IT COSTS A SYNCHRONISE, ON THREE OUTERS IN TWELVE THOUSAND.  The
            // common path is untouched: no extra transfer, no extra sync, and the
            // branch is one predicted test per outer.
            if (m.sweep_host_continued) {
                bump(counters().updjnet_reissued);
                if ((rc = enqueueUpdJnet(m.arena, queue, bound_.geom, bound_.table, bound_.forms,
                                         m.stream, m.d_halt)) != cudaSuccess)
                    return launchFailed("re-enqueue updjnet after the host finished the sweep",
                                        rc);
                // The bridge copy above carried the pre-updjnet bytes, so it has to
                // be taken again -- and the nodal drive below is a HOST call reading
                // a HOST array, so it has to have landed before that call, not
                // merely been enqueued.
                if (bridge_jnet &&
                    (rc = rasbery::xfer::memcpyAsync(
                         "CudaOuterGraph.cu:runOuterTail", "jnet re-download",
                         bound_.host_jnet, bound_.device_jnet, jnet_bytes,
                         cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                    return launchFailed("re-download jnet after the host finished the sweep", rc);
                // AND SO DOES THE RECIPROCAL, FOR THE SAME REASON AND ONE MORE.
                // The kernel above the sweep read the probe the VERDICT kernel
                // published, and on this path that probe described a half drive --
                // republishAfterHostSweep has since overwritten it with the numbers
                // the host loop finished on.  Re-issuing is what makes the slot the
                // finished drive's reciprocal rather than the half one's, and the
                // nodal drive four lines down is its only reader.
                if (reigv_slot != nullptr) {
                    bump(counters().reigv_reissued);
                    if ((rc = enqueueOuterPublishReigv(m.d_probes, reigv_slot, slot,
                                                       m.stream)) != cudaSuccess)
                        return launchFailed("re-enqueue the reigv publish after the host "
                                            "finished the sweep",
                                            rc);
                }
                // AND THE CANONICAL ARM NEEDS IT TOO.  There the nodal drive reads
                // the DEVICE jnet, on the backend's own stream, ordered against this
                // one by nothing but this synchronise.
                bump(counters().in_body_host_syncs);
                bump(counters().sync_sweep_reissue);
                if ((rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:runOuterTail",
                                            "updjnet reissue drain", m.stream)) !=
                    cudaSuccess)
                    return launchFailed("synchronize after the re-issued updjnet", rc);
            }

            // The sweep and updjnet step lines are emitted HERE, after the
            // observation, because that is the first point at which the sweep's
            // output has actually landed -- before it, the enqueue has only been
            // issued.  updjnet was enqueued behind the sweep on the same stream, so
            // one synchronise serves both.
            if (trace_steps) {
                outertrace::emitStepEigv("dev", "sweep",
                                         traceHash(bound_.device_flux, trace_nn),
                                         traceProbeEigv());
                outertrace::emitStep("dev", "updjnet", "jnet",
                                     traceHash(bound_.device_jnet, trace_nsg), nullptr, 0);
            }

            // WHAT THE DRIVE LEFT BEHIND.  This is the only point at which the
            // host flux and the device flux are known to agree -- the sweep's
            // observation has just adopted the device phi into Geometry::Phif -- so
            // it is the only point at which the next outer's elision can be earned.
            //
            // The generation is re-read here rather than reused from the top of the
            // outer because the drive itself bumped it: drive() is handed
            // PhifMutable(), which is what makes it a declared writer.
            if (m.hooks.read_live_state != nullptr) {
                OuterSegmentLiveState after;
                m.hooks.read_live_state(m.hooks.ctx, after);
                // Rev.7.1 Task 10 part 3: ON THE HOST-FREE ARM THE ANSWER IS
                // KNOWN, AND THE FLAG CANNOT GIVE IT.
                //
                // `device_owns_flux` is BICGCMFD::_last_drive_device_flux, which
                // absorbSweepLaunch sets -- and absorbSweepLaunch is exactly the
                // observation this arm defers, so inside a host-free segment the
                // flag still describes the drive BEFORE the segment.  Reading it
                // here would forget the residency after the first outer and
                // re-upload Geometry::Phif over the device phi on the second,
                // while the flux D2H of the previous launch is still filling it.
                //
                // The answer it cannot give is the arm's own precondition: every
                // outer of a host-free segment took the enqueued device drive
                // (canEnqueueDrive() is frozen true), and an enqueued drive ends
                // with issueFluxDownloads writing Geometry::Phif FROM the device
                // phi.  So the device owns the flux, by construction, at every
                // outer of this segment.
                if (hostfree || after.device_owns_flux) {
                    m.resident_flux_generation = after.flux_generation;
                    bump(counters().device_flux_outers);
                } else {
                    // The Wielandt warm-up, or a declined enqueue: the host loop
                    // moved Phif and the device never saw it.  Forget the copy.
                    m.resident_flux_generation = 0;
                }
            }

            // Rev.7.1 Task 18-lite: WHO OWNS THE THREE REGIONS FOR THIS DRIVE.
            //
            // Said once per outer rather than once per segment, because the FLUX
            // answer changes per outer: a drive that fell back to the host CMFD loop
            // (the Wielandt warm-up, a declined enqueue) left Geometry::Phif ahead of
            // the device phi, and the nodal call has to upload it exactly then.  The
            // hook, not the runner, knows how to say so, because the hook is the call
            // site that adopted the buffers.
            //
            // The claim is IDEMPOTENT and writes the same values on every outer of a
            // canonical run, so repeating it does not churn the nodal backend's
            // captured graph -- whose key carries both the ownerships and the mask.
            //
            // Rev.7.1 W3 item 3 rides the SAME predicate, and that is the whole
            // safety argument for it.  `canonical_now` is Nodal::TryDriveGpu's own
            // refusal test, so it is true exactly when the drive about to run is the
            // FULL device pipeline -- the one arm that reads reigv_dev at all.  A
            // drive that falls back to the CPU body reads Nodal::_reigv, which the
            // host hook still sets from the host eigenvalue on every outer; a hybrid
            // drive leaves reigv_dev null and reads the by-value scalar.  Both are
            // exactly what they were, because the claim is not made for them.
            // Rev.7.1 Task 10 part 3: THE ONE TERM OF THE ARM THAT IS RE-ASKED.
            //
            // It cannot change -- Nodal::TryDriveGpu's predicate turns on
            // rod_fraction and a bank move ends the segment -- and it is checked
            // anyway, because the failure it would cause is silent: the CPU body
            // reads Geometry::Jnet, which a host-free segment has stopped
            // filling, and it is not halt-gated, so an overrun outer would solve
            // on its own output.  That is i-SMR CY02's 639-outer divergence in a
            // rarer costume, and it is cheaper to refuse than to audit.
            if (hostfree && !canonical_now)
                return hookFailed("the canonical nodal binding inside a host-free segment");
            if (canonical_now && m.hooks.canonical_nodal_mode != nullptr) {
                m.canonical_nodal_live = true;
                m.hooks.canonical_nodal_mode(m.hooks.ctx, 1);
                bump(counters().canonical_nodal_outers);
                if (reigv_slot != nullptr) {
                    setReigvDevice(true);
                    bump(counters().reigv_device_outers);
                } else {
                    setReigvDevice(false);
                }
            } else {
                // Not eligible: this outer's drive is the CPU body (or a host-owned
                // device drive), it has its bridge, and it is about to write the host
                // arrays itself.  Hand ownership back BEFORE it runs, or its upload
                // is elided against the device copy it is trying to replace.
                releaseCanonicalNodal(true);
            }
            // ==============================================================
            // Rev.7.1 Task 10 part 3: THE FIRST HANDOVER, AS AN EVENT
            // ==============================================================
            //
            // WHAT THE DRAIN WAS ALSO DOING.  `sync_pre_nodal` was described as
            // the sweep observation's cost, and it was -- but it was also the
            // only thing ordering THIS stream's updjnet against the nodal
            // backend's stream, which is where the drive reads that jnet.
            // Removing it for the observation's sake removed the ordering with
            // it, and the drive then read whichever jnet happened to be resident
            // -- usually this outer's, because the segment stream is usually
            // ahead, and occasionally the previous outer's.  On the trimmed
            // kngr3 deck that was 51 of 644 datasets and one outer, at every
            // budget including 1, which is the signature of a race and not of an
            // arithmetic change.
            //
            // The exactness contract's invariant 4 always allowed the other
            // mechanism -- "a synchronise or an event" -- and W3 item 2 had
            // already taken the RETURN half of this pair that way
            // (nodal_completion_event, twenty lines below).  This is the
            // outbound half, and with it the outer's cross-stream pair is two
            // events and no host.
            //
            // RECORDED HERE, AFTER EVERYTHING THE DRIVE READS: the sweep, the
            // reigv publish and updjnet are all behind it on this stream.
            if (hostfree && m.nodal_handover_ev != nullptr &&
                m.hooks.nodal_wait_event != nullptr) {
                if ((rc = cudaEventRecord(m.nodal_handover_ev, m.stream)) != cudaSuccess)
                    return launchFailed("record the nodal handover event", rc);
                if (!m.hooks.nodal_wait_event(m.hooks.ctx,
                                              static_cast<void*>(m.nodal_handover_ev)))
                    return hookFailed("the nodal handover wait hook");
            }
            if (!m.hooks.enqueue_nodal_drive(m.hooks.ctx, m.stream, slot, i))
                return hookFailed("the nodal drive hook");

            // ==================================================================
            // Rev.7.1 W3 item 2: THE NODAL -> SEGMENT HANDOVER, WITHOUT THE BLOCK
            // ==================================================================
            //
            // The drive runs on the nodal backend's OWN stream and upddhat, twenty
            // lines below, runs on this one and reads the jnet the drive produced.
            // Nothing but a host synchronise or an event orders the two, and until
            // now it was the synchronise -- XsReconBackend::solveNodal ended in
            // cudaStreamSynchronize(d.stream) on every outer, so the host blocked
            // once per outer for work whose only consumer is a kernel.
            //
            // THE CONTRACT IS THE SAME, THE MECHANISM IS THE CHEAPER OF THE TWO
            // THE EXACTNESS GATE ALREADY ALLOWS (tools/test_device_outer_exactness_contract.py
            // invariant 4: "a synchronise or an event").  The backend defers its
            // drain only when the drive left NOTHING on the host -- both canonical
            // downloads elided, which is what setCanonicalNodalSegmentMode(true)
            // establishes and nothing outside a segment asks for -- and reports the
            // event here.  On every other outer (the Wielandt warm-up, the CPU
            // body, a materialised drive) it blocks for itself and hands back
            // nullptr, and this is a branch not taken.
            //
            // A NULL HOOK IS ALSO `ALREADY ORDERED`.  A caller that installs no
            // completion hook gets a backend that never defers, because the deferral
            // and the wait were added together and the backend's condition does not
            // consult the hook -- so the failure mode of a half-installed hook table
            // is the old synchronise, not an unordered read.  The state-machine test
            // pins that the two are installed together.
            if (m.hooks.nodal_completion_event != nullptr) {
                void* const nodal_done = m.hooks.nodal_completion_event(m.hooks.ctx);
                if (nodal_done != nullptr) {
                    if ((rc = cudaStreamWaitEvent(
                             m.stream, static_cast<cudaEvent_t>(nodal_done), 0)) != cudaSuccess)
                        return launchFailed("wait for the nodal drive on the segment stream",
                                            rc);
                    bump(counters().nodal_event_waits);
                }
            }
            if (trace_steps)
                outertrace::emitStep("dev", "nodal", "jnet",
                                     traceHash(bound_.device_jnet, trace_nsg), "phis",
                                     traceHash(bound_.device_phis, trace_nsg));

            // (7) cusping -- Driver.h, between the nodal drive and upddhat.
            //
            // RUN, NOT SKIPPED, AND NOT REFUSED.  Stage A used to refuse any deck
            // with a fractional rod and skip this step on the rest, on the argument
            // that eligibility made it a no-op.  It does not: ApplyRodCusping can
            // return true from its own prev_scratch even when no node is fractional
            // NOW, and eligibility was evaluated once per SolveLoop entry while the
            // host asks once per OUTER.  i-SMR CY02 is the deck that shows the
            // difference -- 298 outers and k_eff 1.000003 against the host's 707 and
            // 0.999975, with the same converged-looking answer.
            //
            // The fix is not a device port and not an escape: the nodal drive above
            // is already a host call in an already-synchronised window, so cusping
            // runs in that window, at the point of the outer the host runs it, with
            // the leakage the host nodal drive just produced.
            //
            // Rev.7.1 Task 10 part 3: NOT CALLED AT ALL ON THE HOST-FREE ARM,
            // and that is an identity rather than a skip.  The arm is admitted
            // only when OuterSegmentHooks::cusping_quiescent says no node is
            // fractional and XSSet's carry-over set is empty, and in that state
            // ApplyRodCusping walks its loop, writes nothing, bumps nothing and
            // returns false -- so calling it and not calling it leave the same
            // bytes.  What NOT calling it buys is that the question cannot be
            // asked on a HALTED outer, where the leakage it would read was never
            // produced and the blend it would write could not be undone.
            if (!hostfree && m.hooks.apply_cusping != nullptr &&
                m.hooks.apply_cusping(m.hooks.ctx, slot, i)) {
                bump(counters().cusping_fired);
                // IT REBUILT THE HOST d-TILDE, AND upddhat READS THE DEVICE ONE.
                //
                // The top-of-outer sync cannot cover this: it ran before the nodal
                // drive that produced the leakage cusping needed.  Without the
                // re-upload the outer would blend the cross sections and then
                // correct the current with the d-tilde from before the blend, which
                // is neither the host's outer nor a consistent one.
                if (bound_.host_dtil != nullptr && bound_.device_dtil != nullptr) {
                    const std::size_t dtil_bytes =
                        static_cast<std::size_t>(bound_.geom.nsurf) *
                        static_cast<std::size_t>(bound_.ng) * sizeof(double);
                    if ((rc = rasbery::xfer::memcpyAsync(
                             "CudaOuterGraph.cu:runOuterTail", "dtil after cusping",
                             bound_.device_dtil, bound_.host_dtil, dtil_bytes,
                             cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
                        return launchFailed("re-upload dtil after cusping", rc);
                    bump(counters().cusping_dtil_bytes, dtil_bytes);
                    // upddtil() bumped the generation; the re-upload above made the
                    // device copy current AT the new one.  Re-reading it here is
                    // what stops the top of the next outer copying the same array
                    // again -- on a deck that cusps every outer, that is the whole
                    // d-tilde traffic doubled.
                    if (m.hooks.read_live_state != nullptr) {
                        OuterSegmentLiveState after_cusp;
                        m.hooks.read_live_state(m.hooks.ctx, after_cusp);
                        m.resident_dtil_generation = after_cusp.dtil_generation;
                    }
                }
                // AND IT BLENDED THE CROSS SECTIONS, which is the other half of what
                // ApplyRodCusping does and the half the segment used to catch by
                // accident.  While the xsnf memcmp ran at the top of every outer,
                // the NEXT outer's gate saw the blended bytes and uploaded them.
                // W3 item 4 moved that gate to the segment entry -- on the proof
                // that this hook is the only writer of _xs inside a segment -- so
                // the proof has to be discharged HERE, where the writer is.
                //
                // updpsi of the next outer is the reader that makes it load-bearing:
                // psi = flux . xsnf . vol runs before the sweep's own byte-exact
                // upload can correct the buffer, so a missed re-stage is a fission
                // source built from the pre-blend cross sections while the rest of
                // the outer uses the post-blend ones.
                if (!stageXsnf()) return false;
            }

            if (bridge_jnet) {
                if ((rc = rasbery::xfer::memcpyAsync(
                         "CudaOuterGraph.cu:runOuterTail", "jnet bridge upload",
                         bound_.device_jnet, bound_.host_jnet, jnet_bytes,
                         cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
                    return launchFailed("upload jnet after the nodal drive", rc);
                bump(counters().jnet_bridge_bytes, jnet_bytes);
            }


            // (8) upddhat -- Driver.h:1584
            if ((rc = enqueueUpdDhat(m.arena, queue, bound_.geom, bound_.table, bound_.forms,
                                     bound_.dhat_clamp, bound_.dhat_counters, m.stream,
                                     m.d_halt)) != cudaSuccess)
                return launchFailed("enqueue upddhat", rc);
            if (trace_steps)
                outertrace::emitStep("dev", "upddhat", "dhat",
                                     traceHash(bound_.device_dhat, trace_nsg), nullptr, 0);

            // dhat back to the host, for the same reason psi went back: setls/axb on
            // the host drive path read _dhat, and the segment replaced
            // BICGCMFD::upddhat.  The sweep itself does NOT need this -- it reads the
            // device buffer and its H2D is elided (CmfdSweepIO::dhat_device_resident)
            // -- so this copy exists only to keep the fallback path honest.
            // (4) the convergence INPUTS -- Driver.h:1562.
            //
            // MOVED BEHIND THE OBSERVATION, deliberately.  The plan puts this right
            // after the drive because that is where Driver.h forms eigv and
            // residual; nothing between there and the decision reads
            // CmfdOuterInputs, so the only thing the position decides is WHICH
            // probe it reads.  On the exceptional launches the host finished, the
            // probe the verdict kernel published is a half drive's -- so refreshing
            // before the observation would carry that half drive into the decision.
            if ((rc = enqueueOuterRefreshInputs(queue, m.d_probes, m.d_inputs, m.d_halt,
                                                m.stream)) != cudaSuccess)
                return launchFailed("enqueue input refresh", rc);

            // the decision -- Driver.h:1601-1705, 1834-1860
            if ((rc = enqueueOuterConvergence(m.arena, queue, m.d_inputs, m.d_decisions, m.stream,
                                              m.d_halt)) != cudaSuccess)
                return launchFailed("enqueue outer convergence", rc);

            // the transition -- this is what latches the halt
            if ((rc = enqueueOuterTransition(m.arena, queue, m.d_decisions, m.d_probes,
                                             m.d_segments, m.d_halt, m.d_halted, m.stream)) !=
                cudaSuccess)
                return launchFailed("enqueue outer transition", rc);

            // The exit word, straight behind the transition that writes it, so the
            // NEXT pass's synchronise makes it visible without one of its own.
            if (m.h_seg != nullptr &&
                (rc = rasbery::xfer::memcpyAsync(
                     "CudaOuterGraph.cu:runOuterTail", "segment exit word", m.h_seg,
                     m.d_segments + slot, sizeof(*m.h_seg), cudaMemcpyDeviceToHost,
                     m.stream)) != cudaSuccess)
                return launchFailed("download segment exit", rc);
        return true;
    };

    // --- the body, budget times, on ONE stream -------------------------------
    //
    // Rev.7.1 Task 10 part 2 CHANGED WHERE THE SYNCHRONISE IS, NOT HOW MANY.
    // The v1 body synchronised TWICE per outer -- once so the sweep hook could
    // make its host call, once so the nodal hook could -- and the sweep hook
    // then rendezvoused and drained on its own account.  With the sweep
    // enqueued (BICGCMFD::enqueueDrive) the first of those disappears: the sweep
    // rides this stream and is observed at the sync the NODAL drive was going to
    // force anyway.  One sync per outer is what remains, and it is host
    // arithmetic over Geometry::Jnet that costs it -- Task 18, not this task.
    //
    // WHAT THE HALT GATE IS DOING HERE.  With a budget above one the outers are
    // enqueued back to back, so outer i+1's kernels are in flight before outer
    // i's transition has been observed.  Every one of them tests the segment's
    // halt word first (CudaCmfdOuterKernels.h, and cmfd_sweep_gate for the sweep
    // graph), so an outer whose kernels are already in flight when the exit
    // latches is a sequence of no-ops rather than a sequence of wrong answers.
    // The host never enqueues the NEXT one: it reads the exit word that rode the
    // stream behind the transition at the top of every pass.
    //
    // =======================================================================
    // WHAT USED TO STOP TASK 10 FROM CAPTURING THIS LOOP AS A CONDITIONAL WHILE
    // =======================================================================
    //
    // KEPT AS THE RECORD OF HOW THE HOLE WAS CLOSED, not as a description of
    // today.  The WHILE exists: `runGraphWhile` below captures runOneOuter into
    // a conditional body and launches it once per segment
    // (RASBERY_GPU_OUTER_GRAPH=1, src/GpuOuterWhile.h).  Hole (1), the sweep
    // observation, is exactly what part 3's host-free arm deferred to the
    // segment exit -- which is why the graph arm is a strict subset of that one
    // and refuses (`not_hostfree`) wherever it refuses.  What the list below
    // still earns its place for is the ARGUMENT: every term in it is a fact the
    // captured body depends on, and a change that breaks one of them breaks the
    // WHILE silently.
    //
    // Re-surveyed after W3 items 3 and 4, which closed two of the four entries
    // this list used to have.  A stream in capture mode RECORDS work; it may not
    // be synchronised, and a host call that reads device memory inside the body
    // is not a node, it is a hole.  What is left is ONE hole, and everything
    // else in the body is downstream of it.
    //
    // WHAT IS DONE.
    //
    //   THE EIGENVALUE ROUND TRIP (W3 item 3) is gone.  k_outer_publish_reigv
    //   writes 1/eigv from the device probe straight into NodalView::reigv_dev,
    //   so the reciprocal no longer travels device -> host -> pinned slot ->
    //   device.  The drive's ONE remaining dependence on this outer's
    //   observation is which side owns the flux -- setCanonicalNodalSegmentMode
    //   takes lastDriveLeftDeviceFlux(), which absorbSweepLaunch sets -- and on
    //   the enqueued path (11946 of kngr_238's 12017 outers) that answer is
    //   always "the device".  It is the 71 host-loop outers that make it a read
    //   rather than a constant, which puts it inside hole (1) below and not
    //   beside it.
    //
    //   THE HOST-DECIDED UPLOAD ELISIONS (W3 item 4) are two-thirds gone.  _xs
    //   and _dtil are staged in the arm block, on the proof that
    //   XSSet::ApplyRodCusping is their only in-body writer; the body's H2D node
    //   set is now {flux} rather than {flux, xsnf, dtil}.
    //
    // WHAT IS LEFT, AND IT IS ONE THING WITH THREE CONSEQUENCES.
    //
    //  (1) THE SWEEP OBSERVATION.  BICGCMFD::finishDrive reads the sweep's
    //      scalar block out of pinned host memory the stream filled, so the
    //      stream must be drained first.  That is `sync_pre_nodal` in the
    //      receipt, it is exactly `device_outers` on every arm and every budget,
    //      and it is the whole of what a WHILE capture is blocked on.
    //
    //      ON THE NORMAL PATH (sweep state 1 or 3) the host does nothing with
    //      what it reads that the device does not already hold: it copies eigv,
    //      residual and four counters into host locals and sets
    //      _last_drive_device_flux.  Deferring it to the segment exit needs
    //      three things, and the third is the hard one:
    //
    //        (a) the NEXT outer's sweep must get eigv/reigv/reigvs from device
    //            memory.  BICGCMFD::setls is a no-op on the device-assembly arm
    //            (BICGCMFD.cpp: it only records _device_assembly_pending), so
    //            the only host input is the three scalars enqueueDrive stages
    //            into CmfdSweepIO -- a patch kernel over the staged block, the
    //            same shape as k_outer_publish_reigv.
    //        (b) BICGCMFD's own counters (iter, _wiel_sweep, _bicg_iters) must
    //            be reconstructed at the exit from the device's sweeps_done.
    //            _wiel_sweep is the one with teeth: it gates canEnqueueDrive()
    //            through WIELANDT_WARMUP_SWEEPS = 5 and resetIteration() zeroes
    //            it per statepoint, which is why 71 outers of kngr_238's 12017
    //            still take the host CMFD loop.
    //        (c) A HALTED OUTER MUST BE A NO-OP FOR THE HOST CALLS TOO.  Today
    //            the host learns of sweep state 0 or 2 at this drain and repairs
    //            the outer in place -- re-issuing the updjnet the verdict's halt
    //            swallowed -- which is the only structure that reproduces the
    //            host's trajectory, because the host does not START A NEW OUTER
    //            on those states, it CONTINUES the same drive.  With the
    //            observation deferred, outers i+1..N are enqueued behind a halt
    //            that has already fired: their KERNELS are no-ops (every one
    //            tests it), but the nodal drive would still run and overwrite
    //            the canonical jnet/phis, and apply_cusping would still read a
    //            stale leakage and could blend the cross sections.  So (c) is
    //            really "gate the nodal graph on the segment halt as well" plus
    //            Task 11 for cusping -- two task-sized pieces, not a tidy-up.
    //
    //  (2) apply_cusping IS A HOST CALL ON EVERY OUTER, and it must stay one
    //      until Task 11: i-SMR CY02 proved the question has to be asked per
    //      outer (774 firings in 857 outers there), and ApplyRodCusping answers
    //      it from host scratch.
    //
    //  (3) THE EXIT OBSERVATION at the top of every pass is (1) in its other
    //      costume: it exists because the two host calls above cannot read a
    //      device word.  It is what makes a WIDE segment pay MORE rendezvous per
    //      outer than a narrow one -- kngr_238 measures 2.006 per outer at b1
    //      against 2.217 at b8 and 2.224 at b16 -- so the budget's advantage
    //      today is transfer amortisation, not fewer round trips.  At b8, 2535
    //      of 14552 passes are DISCOVERY-ONLY: they synchronise, see the exit,
    //      and break without committing an outer.
    //
    //      IT CAN BE REMOVED WITHOUT (1), AND HERE IS THE SHAPE.  Nothing between the sweep
    //      and upddhat touches the three inputs of the convergence decision
    //      (eigv, residual, prev_inner) -- that is the same argument the header
    //      note makes for evaluating the decision AFTER upddhat, run backwards --
    //      so refresh + convergence + a publish of `exit` could be issued right
    //      behind the sweep instead.  The host would then learn outer i's exit
    //      at outer i's OWN pre-nodal drain, which it already pays, and pass i+1
    //      would simply not be entered.  sync_exit_observation goes to zero and
    //      the discovery-only passes stop existing.
    //
    //      THE HAZARD IS cmfdOuterConvergence's SIDE EFFECTS, and it is why this
    //      is written down rather than done here.  That body MUTATES
    //      CmfdOuterState -- ++total_outer, prev_inner = eigv, xe_interim_count,
    //      flux_stall, clean_iters (CmfdOuterKernel.h) -- so it is not
    //      re-runnable, and the exceptional path (sweep state 0 or 2, 3 outers
    //      in 12000) has to re-run it: republishAfterHostSweep overwrites the
    //      probe the first evaluation read.  Running it twice would compare the
    //      second evaluation's prev_inner against the FIRST one's eigenvalue,
    //      which is a different convergence test on exactly the outers that are
    //      already the hardest to reproduce.  The fix is to make the early
    //      kernel write its state mutations to a SHADOW DeviceSlotState and have
    //      the end-of-body commit apply the shadow -- decide early, commit late
    //      -- so a re-issue recomputes from an unmutated state.  That is a real
    //      change to the tree's most delicate kernel and it belongs in its own
    //      gated commit, not as a rider on this one.
    //
    //      AND SIZE IT BEFORE BUILDING IT, because the rendezvous COUNT
    //      overstates it.  Removing this drain deletes 11341 of b8's 26643
    //      rendezvous -- 43% of the count -- but a rendezvous is not a wall.
    //      This one waits for outer i's TAIL (upddhat, the refresh, the
    //      decision, the transition and a 32-byte D2H), so what it costs is not
    //      the round trip, it is the failure to overlap that tail with the
    //      enqueue of outer i+1's front half: tens of microseconds times 11341,
    //      order half a second of a 44.5 s server run.  `sync_pre_nodal` is a
    //      different animal -- it waits for the SWEEP, which is real work, so
    //      deleting it does not save its duration either.  What (1) actually
    //      buys is the end of the serialisation: with the body captured, the
    //      device stops going idle while the host observes, launches and asks
    //      about cusping.  Whoever picks this up should measure that idle window
    //      on the 238 box FIRST and let the number choose the order of work.
    //
    // MEASURED ON THE LOCAL BOX (tools/probe_conditional_graph.cu, CUDA 12.6,
    // sm_61, GTX 1080 Ti): WHILE conditional nodes are legal and the handle
    // scope the runtime accepts is the BODY graph, not the root; SWITCH is not
    // available below CUDA 12.8 and the nested-IF fallback costs 8.44 us per
    // iteration against WHILE's own 4.55 us; a cooperative kernel node inside a
    // conditional body is accepted; instantiation of 1505 nodes takes 3.24 ms,
    // against the plan's 250 ms gate.  At 4.55 us per outer a WHILE over this
    // deck's 12017 outers costs 55 ms of a 60 s run, so the control flow has
    // never been what decides this -- hole (1) is.
    // =======================================================================
    // Rev.7.1 Task 10 part 4: ONE OUTER, AS ONE NAMED THING
    // =======================================================================
    //
    // WHY IT IS A LAMBDA NOW AND WAS THE BODY OF A `for` BEFORE.  The device-side
    // WHILE (src/GpuOuterWhile.h) has to CAPTURE exactly one outer, and a capture
    // is a call, not a loop iteration.  Written twice -- once for the stream arm
    // to run and once for the graph arm to record -- the two would be two
    // spellings of the same thirty enqueues, and the whole exactness claim of
    // this task is that they are ONE.  So the loop calls it and the capture calls
    // it, and neither has a body of its own.
    //
    // NOTHING IN IT CHANGED.  The extraction is mechanical: the same statements
    // in the same order, with the per-pass preamble (the exit observation and the
    // `hostfree_enqueued` bump, both of which belong to the LOOP and not to the
    // outer) left behind at the call site.
    auto runOneOuter = [&](const unsigned int i) -> bool {
        // THE LIVE STATE, RE-READ PER OUTER.
        //
        // Not once per segment: a segment with a budget above one runs outers
        // 2..N without returning to the host, but the host DID run between
        // them -- the sweep hook, the nodal drive and cusping are all host
        // calls -- and each of them can move a generation.  Deciding an
        // elision from a segment-entry value is what made i-SMR CY02 fail at
        // b8 and b16 while passing at b1.
        //
        // W3 item 4 NARROWED WHAT IS DECIDED FROM IT rather than contradicting
        // it.  Two of the three elisions moved to the segment entry, on the
        // proof that _xs and _dtil have exactly one in-body writer
        // (ApplyRodCusping) which re-stages them itself.  The flux has no such
        // proof and did not move: BICGCMFD::drive writes Geometry::Phif on every
        // host-loop outer, which is precisely the CY02 shape.
        m.sweep_host_continued = false;

        // The tracer's outer index inside the segment.  Driver.h set the context
        // to the host loop counter before delegating; `base + i` is then the
        // outer ordinal the OFF arm prints for the same outer, so the two logs
        // align line for line and a diff names the step.
        if (trace_steps) {
            static thread_local int trace_base = 0;
            if (i == 0) trace_base = outertrace::context().outer;
            outertrace::context().outer = trace_base + static_cast<int>(i);
        }

        OuterSegmentLiveState live;
        if (m.hooks.read_live_state != nullptr) m.hooks.read_live_state(m.hooks.ctx, live);

        // (0) the flux the whole outer is computed from.
        //
        // See OuterSegmentBinding::host_flux: drive() takes the HOST loop for
        // the Wielandt warm-up and whenever the device sweep declines, and after
        // such a drive the device phi is behind the host flux.  Uploading it
        // here makes the segment independent of which path the previous drive
        // took, which is the difference between a fast path and a correct one.
        // SKIPPED WHEN THE DEVICE COPY IS ALREADY THE HOST'S.  Two things have
        // to hold and neither is enough alone: no host writer has touched Phif
        // since the copy was made (the generation), and the copy was made from
        // a drive that actually downloaded it (recorded below).
        const bool flux_current = m.hooks.read_live_state != nullptr &&
                                  m.resident_flux_generation != 0 &&
                                  m.resident_flux_generation == live.flux_generation;
        if (flux_current) bump(counters().flux_uploads_elided);
        if (bound_.host_flux != nullptr && m.residency.flux != nullptr && !flux_current) {
            const std::size_t flux_bytes =
                static_cast<std::size_t>(bound_.nxyz) *
                static_cast<std::size_t>(bound_.ng) * sizeof(double);
            if ((rc = rasbery::xfer::memcpyAsync(
                     "CudaOuterGraph.cu:runOneOuter", "flux upload", m.residency.flux,
                     bound_.host_flux, flux_bytes, cudaMemcpyHostToDevice, m.stream)) !=
                cudaSuccess)
                return launchFailed("upload flux", rc);
            bump(counters().flux_sync_bytes, flux_bytes);
        }

        // (1) updpsi -- Driver.h:1547
        if ((rc = enqueueUpdPsi(m.arena, queue, bound_.geom, bound_.table, bound_.forms,
                                m.stream, m.d_halt)) != cudaSuccess)
            return launchFailed("enqueue updpsi", rc);
        if (trace_steps)
            outertrace::emitStep("dev", "updpsi", "psi",
                                 traceHash(bound_.device_psi,
                                           static_cast<std::size_t>(bound_.nxyz)),
                                 nullptr, 0);

        // psi AND dhat back to the host -- ONLY when a host reader is about to
        // run, which here means only when this outer's drive will take the HOST
        // loop.
        //
        // WHAT CHANGED.  These two used to go back every outer, 483 KiB of the
        // 495 KiB pair on kngr_238 and 327 MB over a run, to serve a reader that
        // almost never came: 656 of 661 outers took the device sweep, which
        // reads the DEVICE buffers, has its uploads elided, and downloads for
        // itself on the two exceptional states
        // (issueExceptionalOperatorDownloads).  The host loop reads _psi through
        // wiel and _dhat through the host assembly, and canEnqueueDrive() is
        // exactly the gate that decides which of the two runs.
        //
        // dhat is mirrored HERE and not after upddhat because the reader is the
        // NEXT drive: what the host assembly wants is the d-hat the previous
        // outer produced, which is what the device holds at this point.
        const bool host_reader_next = !live.sweep_will_enqueue;
        if (host_reader_next && bound_.host_dhat != nullptr &&
            bound_.device_dhat != nullptr) {
            const std::size_t dhat_bytes =
                static_cast<std::size_t>(bound_.geom.nsurf) *
                static_cast<std::size_t>(bound_.ng) * sizeof(double);
            if ((rc = rasbery::xfer::memcpyAsync(
                     "CudaOuterGraph.cu:runOneOuter", "dhat mirror", bound_.host_dhat,
                     bound_.device_dhat, dhat_bytes, cudaMemcpyDeviceToHost, m.stream)) !=
                cudaSuccess)
                return launchFailed("mirror dhat to the host", rc);
            bump(counters().host_mirror_bytes, dhat_bytes);
        }
        if (host_reader_next && bound_.host_psi != nullptr && bound_.device_psi != nullptr) {
            const std::size_t psi_bytes =
                static_cast<std::size_t>(bound_.nxyz) * sizeof(double);
            if ((rc = rasbery::xfer::memcpyAsync(
                     "CudaOuterGraph.cu:runOneOuter", "psi mirror", bound_.host_psi,
                     bound_.device_psi, psi_bytes, cudaMemcpyDeviceToHost, m.stream)) !=
                cudaSuccess)
                return launchFailed("mirror psi to the host", rc);
            bump(counters().host_mirror_bytes, psi_bytes);
        }
        // A SYNCHRONISING SWEEP HOOK IS STILL A HOST CALL.  Kept for the arm
        // that has no stream-ordered drive (no arena, or the Wielandt warm-up),
        // where the budget is forced to one and this is the only sync of the
        // segment.
        //
        // AND THE MIRROR ABOVE IS AN ASYNC D2H THE HOST READER IS ABOUT TO READ.
        //
        // `host_reader_next` says this outer's drive takes the HOST loop, and
        // the first thing that loop's call site does -- before it reaches
        // BICGCMFD::drive at all -- is setls(eigv), which on the warm-up takes
        // assembleHostLinearSystem and reads _dhat for every node
        // (CMFD::setls: `(-dtil(ige, ls) + dhat(ige, ls)) * area`).  The two
        // cudaMemcpyAsyncs just issued are filling _dhat and _psi FROM the
        // device, on this stream, and outerSweepEnqueueHook's own
        // syncSweepStream() comes AFTER setls -- so setls was reading a pinned
        // buffer with a DMA in flight into it.  _dhat is page-locked
        // (prepareDeviceSweeps leases it), so that copy is a real asynchronous
        // transfer whose timing the driver owns: the operator was assembled
        // from the new d-hat, the old one, or a mix, and which one decided how
        // many Wielandt sweeps the warm-up took.
        //
        // WHY IT WAS INVISIBLE AT A BUDGET OF ONE.  At budget 1 the previous
        // segment EXITED after the previous outer, and the exit mirror plus the
        // observation's synchronise had already put that same d-hat in _dhat.
        // The in-body copy then rewrites identical bytes and the race cannot be
        // observed.  From the second outer of a wider segment there has been no
        // exit since, _dhat still holds the d-hat of the last exit, and the
        // copy is the first arrival of the current one.  kngr_238 statepoint 35
        // is where it showed: its first outer's warm-up drive stops in four
        // sweeps, so outer 1 is still inside WIELANDT_WARMUP_SWEEPS and still
        // takes the host loop, and two b8 runs of the same binary took 245 and
        // 253 outers there while statepoints 1..34 were bit-identical.
        //
        // ONE SYNCHRONISE, ON THE OUTERS THAT ARE ABOUT TO BLOCK ANYWAY.  An
        // outer with host_reader_next runs a blocking host CMFD drive three
        // lines below; the enqueued outers -- 656 of 661 on this deck -- issue
        // no mirror and pay nothing.
        if (!stream_sweep || host_reader_next) {
            bump(counters().in_body_host_syncs);
            bump(counters().sync_mirror_drain);
            if ((rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:runOneOuter",
                                        "pre-sweep drain", m.stream)) != cudaSuccess)
                return launchFailed("synchronize before the CMFD sweep", rc);
        }

        // (2,3) setls + drive -- Driver.h:1551-1555.  On the stream-ordered arm
        // this only ENQUEUES; the sweep's own verdict kernel publishes this
        // outer's DeviceOuterProbe from device memory, so nothing between here
        // and the convergence kernel needs the host.
        if (!m.hooks.enqueue_cmfd_sweep(m.hooks.ctx, m.stream, slot, i))
            return hookFailed("the CMFD sweep hook");

        // ==================================================================
        // Rev.7.1 W3 item 3: 1/eigv, WHERE BOTH ENDPOINTS ALREADY ARE
        // ==================================================================
        //
        // The nodal drive reads its reciprocal from a device double
        // (NodalView::reigv_dev).  The sweep just wrote the eigenvalue into a
        // device probe.  Between two device facts sat three hops of host: the
        // observation copied eigv up, Driver.h's nodal hook divided, and a
        // staged H2D carried the quotient back down.  This kernel does the
        // divide in place and the middle two hops stop existing.
        //
        // ISSUED HERE, BEHIND THE SWEEP ON THE SAME STREAM, so it reads the
        // probe THIS outer's verdict kernel published rather than the previous
        // one's -- and ahead of the pre-nodal synchronise, so the value has
        // landed before the drive that reads it is launched on the other stream.
        //
        // THE SLOT IS ASKED FOR EVERY OUTER.  The backend allocates its device
        // block inside the first drive and re-lays it out when nsurf changes, so
        // a remembered address is a write into a freed allocation; and a null
        // answer (before that first drive, on the hybrid arm) is the ordinary
        // way of saying the host still owns the reciprocal.
        double* const reigv_slot =
            m.hooks.nodal_reigv_slot != nullptr
                ? static_cast<double*>(m.hooks.nodal_reigv_slot(m.hooks.ctx))
                : nullptr;
        if (reigv_slot != nullptr &&
            (rc = enqueueOuterPublishReigv(m.d_probes, reigv_slot, slot, m.stream)) !=
                cudaSuccess)
            return launchFailed("enqueue the reigv publish", rc);
        // (5) updjnet -- Driver.h:1569
        if ((rc = enqueueUpdJnet(m.arena, queue, bound_.geom, bound_.table, bound_.forms,
                                 m.stream, m.d_halt)) != cudaSuccess)
            return launchFailed("enqueue updjnet", rc);

        // (6) nodal reset + drive -- Driver.h:1573-1575
        //
        // THE JNET BRIDGE.  The nodal drive is still a HOST call and it reads
        // and writes Geometry::Jnet, while updjnet above wrote the device copy
        // and upddhat below reads it.  So the device jnet goes down, the host
        // runs, and the result comes back -- in the RUNNER rather than in the
        // hook, so the hook stays pure host physics with no device vocabulary.
        // Both halves are counted; see OuterSegmentCounters::jnet_bridge_bytes.
        //
        // Rev.7.1 Task 18-lite REMOVED THE BRIDGE WHERE THE BINDING IS LIVE.
        // A nodal backend that has adopted the segment's own jnet reads the
        // buffer updjnet wrote two lines up and writes the buffer upddhat reads
        // six lines down, so the two halves of the bridge would be copying an
        // array to the host and back so that a device kernel could read what a
        // device kernel had just written.
        //
        // THE QUESTION IS ASKED PER OUTER, AND THAT IS THE WHOLE CORRECTNESS
        // ARGUMENT.  Nodal::TryDriveGpu falls back to the CPU body on any deck
        // with a fractional rod, and the CPU body reads Geometry::Jnet -- so an
        // outer that will fall back must keep its bridge.  Asking once per ARM
        // is not enough and i-SMR CY02 is the deck that proves it: the ROD
        // SEARCH moves the bank INSIDE SolveLoop, so a loop that armed with
        // every rod integral hits fractional ones several outers later.  Armed
        // once, that deck ran 639 outers whose drive had quietly become a CPU
        // body reading a jnet the device had stopped sending home, and converged
        // to k_eff 1.000043 where the host gets 0.999975.
        //
        // The eligibility hook is Nodal::TryDriveGpu's own predicate, asked
        // without running anything -- the same relationship canEnqueueDrive()
        // has to drive().
        const bool canonical_now =
            bound_.canonical_nodal && m.hooks.canonical_nodal_eligible != nullptr &&
            m.hooks.canonical_nodal_eligible(m.hooks.ctx) != 0;
        const bool bridge_jnet =
            !canonical_now && bound_.host_jnet != nullptr && bound_.device_jnet != nullptr;
        std::size_t jnet_bytes = 0;
        if (bridge_jnet) {
            jnet_bytes = static_cast<std::size_t>(bound_.geom.nsurf) *
                         static_cast<std::size_t>(bound_.ng) * sizeof(double);
            if ((rc = rasbery::xfer::memcpyAsync(
                     "CudaOuterGraph.cu:runOneOuter", "jnet bridge download",
                     bound_.host_jnet, bound_.device_jnet, jnet_bytes,
                     cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                return launchFailed("download jnet for the nodal drive", rc);
            bump(counters().jnet_bridge_bytes, jnet_bytes);
        }

        // THE ONE SYNCHRONISE OF THE OUTER, and it belongs to the nodal drive.
        //
        // Rev.7.1 Task 10 part 3 REMOVED ITS ONLY REASON on the host-free arm.
        // Since W3 item 3 this drain existed for the sweep OBSERVATION alone --
        // the nodal drive's `1.0 / eigv` had already stopped needing the host --
        // and the observation is now a per-SEGMENT summary the verdict kernel
        // keeps in device memory (CmfdSweepProbeSink::Accum).  Nothing between
        // here and the exit reads a device word from the host, so nothing here
        // waits.
        if (!hostfree) {
            bump(counters().in_body_host_syncs);
            bump(counters().sync_pre_nodal);
            if ((rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:runOneOuter",
                                        "pre-nodal drain", m.stream)) != cudaSuccess)
                return launchFailed("synchronize before the nodal drive", rc);
        }

        // The sweep's observation, on the sync that just happened.  It adopts
        // the flux into Geometry::Phif and the eigenvalue into the host local
        // the nodal hook is about to divide by -- and, on the exceptional
        // launches the device could not finish (sweep state 0 or 2), it runs the
        // remaining blocking launches and republishes the probe the verdict
        // kernel had to guess at.
        if (stream_sweep && !hostfree &&
            !m.hooks.finish_cmfd_sweep(m.hooks.ctx, m.stream, slot, i))
            return hookFailed("the CMFD sweep observation hook");

        return runOuterTail(i, reigv_slot, canonical_now, bridge_jnet, jnet_bytes);
    };

    // =======================================================================
    // Rev.7.1 Task 10 part 4: IS THIS SEGMENT ONE DEVICE-SIDE WHILE?
    // =======================================================================
    //
    // A STRICT SUBSET OF THE HOST-FREE ARM, asked in the same place and with the
    // same rule: every term is a property that cannot move inside a segment.
    // What is added to the ladder above is the four things a CAPTURE needs and a
    // host loop does not.
    //
    //   hostfree            everything that arm proved -- the sweep enqueues,
    //                       the nodal drive is canonical, cusping cannot fire --
    //                       the body needs too, and needs frozen, because it is
    //                       recorded once and replayed.
    //   batch_width <= 1    A CAPTURE WINDOW IS EXCLUSIVE OF EVERY ALLOCATION IN
    //                       THE PROCESS (GpuCaptureArbiter.h, and the measurement
    //                       in its header: four runs in twenty died 1.8 s in).
    //                       The arbiter serialises the ones it can see, but a
    //                       batch's whole point is M host threads on one arena
    //                       and the window here is a whole OUTER wide, not a
    //                       bucket capture.  Refused by name; the receipt says
    //                       so, and graph_refusals{"batch":N} is the gate.
    //   !trace_steps        the per-step tracer synchronises the segment stream
    //                       and copies device memory to hash it.  Unlike the
    //                       host-free arm -- where that is merely extra
    //                       observation -- here it is a host call reading device
    //                       memory INSIDE the recorded body, which is not a node
    //                       but a hole.  Refused, and the ON == OFF gate is run
    //                       untraced anyway.
    //   budget >= 2         outer 0 runs eagerly (the warm-up rule), so a budget
    //                       of one leaves nothing to capture.  Not a failure --
    //                       b1 is a legitimate arm and it simply has no WHILE.
    //
    // CY02 needs no entry of its own: the host-free ladder already refuses it by
    // name (cusping_live), and this ladder refuses whatever that one did.
    OuterGraphRefusal graph_why = OuterGraphRefusal::None;
#if RASBERY_HAS_OUTER_WHILE
    if (!outerGraphEnabled())        graph_why = OuterGraphRefusal::FeatureOff;
    else if (!hostfree)              graph_why = OuterGraphRefusal::NotHostFree;
    else if (batch_width > 1)        graph_why = OuterGraphRefusal::Batch;
    else if (trace_steps)            graph_why = OuterGraphRefusal::Traced;
    else if (budget < 2u)            graph_why = OuterGraphRefusal::BudgetOne;
#else
    // Built against a runtime with no conditional nodes.  Asking for the arm and
    // silently not getting it is the one answer a receipt must never give.
    graph_why = outerGraphEnabled() ? OuterGraphRefusal::Unsupported
                                    : OuterGraphRefusal::FeatureOff;
#endif
    const bool graph_arm = graph_why == OuterGraphRefusal::None;
    bool       graph_ran = false;
    if (!graph_arm) bump(counters().graph_refusals[static_cast<int>(graph_why)]);

    // Capture outer 1 into a WHILE body, instantiate (or reuse), and launch.
    //
    // RETURNS THREE ANSWERS, not two:  1 = the WHILE ran and the caller must
    // stop walking the loop;  0 = the arm was not taken and the stream loop
    // continues from outer 1 exactly as it would have;  -1 = a failure the
    // caller must propagate.
    //
    // THE DIFFERENCE BETWEEN 0 AND -1 IS WHERE THE BUILD REFUSED.  Everything up
    // to record(body) is pure graph plumbing: nothing host-side has moved and
    // nothing device-side has run, so falling back is free.  From record(body)
    // on, the body's thirty enqueue helpers HAVE run as host calls -- they moved
    // the CMFD backend's byte-exact upload shadows and the nodal backend's
    // residency claims -- while the capture that was to carry their work has
    // been discarded.  Re-running them on the stream arm would elide uploads
    // whose bytes never left the host, which is a plausible wrong answer rather
    // than a slow one.  So that half is a hard stop.
    auto runGraphWhile = [&]() -> int {
#if !RASBERY_HAS_OUTER_WHILE
        return 0;
#else
        // THE DYNAMIC PRECONDITION, and it is the flux.  A body captured at
        // outer 1 has no flux H2D node in it; that is right only if outer 1
        // would not have issued one.  The host-free arm makes it so (no host
        // writer of Geometry::Phif inside the segment, so the generation the
        // eager outer 0 adopted is still live) -- and "makes it so" is checked
        // here rather than assumed, because a captured upload whose source
        // generation has moved is the one shape this mechanism must not have.
        OuterSegmentLiveState live1{};
        if (m.hooks.read_live_state != nullptr) m.hooks.read_live_state(m.hooks.ctx, live1);
        const bool flux_current = m.hooks.read_live_state != nullptr &&
                                  m.resident_flux_generation != 0 &&
                                  m.resident_flux_generation == live1.flux_generation;
        if (!flux_current && bound_.host_flux != nullptr && m.residency.flux != nullptr) {
            bump(counters()
                     .graph_refusals[static_cast<int>(OuterGraphRefusal::FluxUploadLive)]);
            return 0;
        }

        OuterWhileKey key{};
        key.budget        = budget;
        key.slot          = slot;
        key.nxyz          = bound_.nxyz;
        key.ng            = bound_.ng;
        key.nsurf         = bound_.geom.nsurf;
        key.canonical     = bound_.canonical_nodal ? 1u : 0u;
        key.hostfree_full = outerHostFreeFull() ? 1u : 0u;
        // THE ONE ADDRESS IN THE BODY THAT CAN MOVE.  Everything else the
        // capture bakes is an arena pointer, and the arena's contract is that
        // its addresses never move; the nodal backend allocates its own device
        // block inside its first drive and re-lays it out when nsurf changes, so
        // a cached body holding the previous layout's slot would publish 1/eigv
        // into freed memory.  In the key, it cannot.
        key.reigv_slot = m.hooks.nodal_reigv_slot != nullptr
                             ? m.hooks.nodal_reigv_slot(m.hooks.ctx)
                             : nullptr;

        cudaGraphExec_t exec = m.while_cache.find(key);
        if (exec == nullptr) {
            // The scratch stream the ROOT is captured on.  It carries two nodes
            // and never executes anything: the root graph is LAUNCHED on the
            // segment stream, and which stream a node was captured on decides
            // topology, not where it runs.
            if (m.while_cache.root_stream == nullptr) {
                // WP19.  The last unguarded stand-up call on the WHILE path,
                // and the one immediately in front of the capture below: it
                // runs once, on the first segment of the run, on whichever
                // lane got there first.
                rasbery::AllocWindow _alloc_window("outer.while.root_stream");
                if ((rc = cudaStreamCreateWithFlags(&m.while_cache.root_stream,
                                                    cudaStreamNonBlocking)) != cudaSuccess) {
                    launchFailed("create the WHILE root stream", rc);
                    return -1;
                }
            }
            // Before the capture, not after: the splice reads this flag on the
            // sweep and nodal drive launch sites, both of which are inside the
            // body about to be recorded.
            rasbery::graphCapturePossible();
            const char* stage = "";
            cudaGraph_t root  = nullptr;
            {
                // The arbiter's exclusive window.  Single mode has no sibling to
                // exclude, so this costs an uncontended lock once per shape --
                // and it is what makes the refusal above ("batch") a policy
                // rather than the only thing standing between this capture and a
                // sibling deck's cudaDeviceSynchronize.
                rasbery::CaptureWindow window(m.stream, "outer.while");
                rc = buildOuterWhile(m.while_cache.root_stream, m.stream, m.d_segments,
                                     m.d_halt, slot, &stage,
                                     [&](cudaStream_t) { return runOneOuter(1u); },
                                     &root, &exec);
            }
            // WP19.  A capture-concurrency code is a statement about the other
            // lanes, not about this graph, so it is worth exactly one rebuild
            // with the process quiet -- and the sticky error has to be cleared
            // first or the retry inherits it.  A second refusal is a real one
            // and falls through to the named refusal below, loudly.
            //
            // WP19.1, AND THIS GATE IS THE WHOLE FIX.  "Worth one rebuild" is
            // true only while the build has moved nothing.  Past record(body)
            // the body's enqueue helpers have committed their upload shadows
            // for copies the discarded capture never carried, so the SECOND
            // record elides them and bakes a body with no H2D node for data the
            // device does not hold -- a graph that replays into a non-finite
            // flux with no CUDA error anywhere.  See
            // outerWhileStageMovedHostState() above.  When the cursor says the
            // host state moved, the honest rung is the hard stop below, said
            // out loud, and NOT a retry.
            if (rasbery::captureIllegal(static_cast<int>(rc)) &&
                outerWhileStageMovedHostState(stage)) {
                rasbery::noteCaptureRaceAbandoned("outer.while", stage,
                                                  static_cast<int>(rc), slot,
                                                  cudaGetErrorString(rc));
            } else if (rasbery::captureIllegal(static_cast<int>(rc))) {
                cudaGetLastError();
                rasbery::noteCaptureRaceRetry("outer.while", stage,
                                              static_cast<int>(rc), slot);
                root = nullptr;
                exec = nullptr;
                {
                    rasbery::CaptureWindow window(m.stream, "outer.while.retry");
                    rc = buildOuterWhile(m.while_cache.root_stream, m.stream, m.d_segments,
                                         m.d_halt, slot, &stage,
                                         [&](cudaStream_t) { return runOneOuter(1u); },
                                         &root, &exec);
                }
                if (rasbery::captureIllegal(static_cast<int>(rc)))
                    rasbery::noteCaptureRaceUnrecovered("outer.while",
                                                        static_cast<int>(rc),
                                                        cudaGetErrorString(rc));
            }
            if (rc != cudaSuccess) {
                std::fprintf(stderr,
                             "[RASBERY][OUTER_GPU][WARN] the outer WHILE refused at %s: %s\n",
                             stage, cudaGetErrorString(rc));
                cudaGetLastError();
                bump(counters()
                         .graph_refusals[static_cast<int>(OuterGraphRefusal::CaptureFailed)]);
                // WP19.1: ONE predicate, shared with the retry gate above, so
                // the two can never disagree about what "the body ran" means.
                const bool host_state_moved = outerWhileStageMovedHostState(stage);
                if (!host_state_moved) return 0;
                bump(counters().refusals[static_cast<int>(OuterSegmentRefusal::LaunchFailed)]);
                releaseCanonicalNodal(false);
                return -1;
            }
            // The cap is a leak guard and not a policy, exactly as the nodal
            // cache's is: a key space wider than expected degrades to
            // re-instantiating, which is what a tree with no cache would do.
            if (m.while_cache.entries.size() >= OuterWhileCache::kMax) m.while_cache.clear();
            m.while_cache.entries.push_back(OuterWhileGraph{key, root, exec});
            bump(counters().graph_instantiations);
        }

        if ((rc = cudaGraphLaunch(exec, m.stream)) != cudaSuccess) {
            launchFailed("launch the outer WHILE", rc);
            return -1;
        }
        bump(counters().graph_launches);
        graph_ran = true;
        return 1;
#endif
    };

    for (unsigned int i = 0; i < budget; ++i) {
        // WP14: the pass census, counted at the TOP so a pass the WHILE arm
        // consumes is a pass.  On the stream arm `segment_passes -
        // device_outers` is exactly `discovery_passes`, and that identity is
        // what makes "the exit observation fires once per outer" a measurement
        // rather than an inference from sync_exit_observation alone.
        bump(counters().segment_passes);
        // THE ONE PLACE THE GRAPH ARM DIVERGES FROM THE STREAM ARM, and it is
        // one break.  Outer 0 has run eagerly above (the warm-up rule), its
        // transition has published outer_in_segment = 1, and everything the body
        // will bake is now frozen.  What follows is outers 1..N as ONE launch,
        // with the stop rule -- exit, halt, budget -- evaluated on the device
        // after each one.
        if (graph_arm && i == 1u) {
            const int taken = runGraphWhile();
            if (taken < 0) return false;
            if (taken > 0) break;
            // Not taken, and not a failure: the ladder above says why, and this
            // segment finishes on the stream arm from outer 1 exactly as it
            // would have with the feature off.
        }
        // THE PREVIOUS OUTER'S EXIT, AND WHY IT COSTS A SYNCHRONISE.
        //
        // The halt gate makes an outer past the exit a sequence of no-op
        // KERNELS, but two steps of the body are host CALLS -- the nodal drive
        // and, on the arm that has no stream-ordered drive, the sweep itself --
        // and a host call cannot read a device word.  Running either on a halted
        // outer would not be a no-op: the nodal drive would re-solve on the
        // previous outer's jnet, and a blocking drive would advance the
        // eigenvalue past the exit the segment already published.
        //
        // So the exit is observed here, before anything of this outer is
        // enqueued.  The sync is on a stream whose only outstanding work is the
        // tail of the previous outer -- upddhat, the decision, the transition
        // and three small copies -- all of which ran while the host was in the
        // nodal drive, so in practice it returns immediately.  It is what bounds
        // the overrun at ZERO outers rather than the budget - 1 the v1 note
        // priced at 3.9us.
        //
        // Rev.7.1 Task 10 part 3: AND WHY A HOST-FREE SEGMENT DOES NOT PAY IT.
        // Both of those host calls now refuse a halted outer for themselves --
        // the nodal drive through NodalView::halt, the sweep through
        // cmfd_sweep_gate as it always did -- and the third, cusping, is proved
        // unable to fire for the whole segment before the arm is taken.  With
        // nothing left in the body that a stale halt could mislead, the exit
        // does not have to be observed until the segment ends.
        //
        // WHAT IT COSTS INSTEAD is the overrun the observation used to prevent:
        // this segment will enqueue its whole budget whatever the exit says, and
        // `hostfree_enqueued - hostfree_outers` is how many of those outers were
        // no-ops.  At a fixed budget that is real launches doing nothing; it is
        // the conditional WHILE, not this task, that removes them.
        if (i > 0 && (!hostfree || keep_exit_obs)) {
            bump(counters().in_body_host_syncs);
            bump(counters().sync_exit_observation);
            // NOT TIMED HERE, AND THAT IS A RULE RATHER THAN AN OMISSION.
            // tools/test_device_outer_state_machine.py bans host clocks from
            // this file: a timer on a per-outer path is a tax on the thing it
            // measures.  The wall this synchronise costs is already a ledger
            // row -- `CudaOuterGraph.cu:runSegment:exit observation` under
            // RASBERY_XFER_LEDGER carries calls AND ns -- so the number exists
            // and is paid for once, in the wrapper.
            if ((rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:runSegment",
                                                "exit observation", m.stream)) != cudaSuccess)
                return launchFailed("synchronize on the segment exit", rc);
            if (m.h_seg != nullptr && m.h_seg->exit != 0u) {
                // WP14: THE DISCOVERY-ONLY PASS, NAMED.  It synchronised, saw the
                // exit and committed nothing, and it is the pass the receipt
                // could not previously distinguish from a committing one.  The
                // flag it sets is what licenses the two V2 elisions at the exit:
                // from here to the end of this function nothing is enqueued on
                // m.stream that a kernel could observe, so the stream is empty
                // and the exit word is current.
                bump(counters().discovery_passes);
                observed_exit = true;
                break;
            }
        }
        if (hostfree) bump(counters().hostfree_enqueued);

        if (!runOneOuter(i)) return false;
    }

    // =======================================================================
    // Rev.7.1 Task 10 part 3: THE OBSERVATION, ONCE, FOR THE WHOLE SEGMENT
    // =======================================================================
    //
    // ONE SYNCHRONISE WHERE THERE WERE `budget` OF THEM.  Everything the host
    // used to read per outer is either reconstructed from DeviceSlotState at the
    // exit observation twenty lines below (eigv, the residual, flux_stall,
    // total_outer) or summed on the device as the segment ran (BICGCMFD's
    // attempt counters, the last verdict, and -- if there was one -- the whole
    // scalar block of a drive the device abandoned).
    //
    // IT IS A SECOND SYNCHRONISE OF THE SEGMENT AND NOT A SPARE ONE.  The exit
    // observation below cannot serve: it must be issued AFTER the exit mirrors,
    // and the mirrors' correctness depends on what this observation decides --
    // whether an abandoned outer still owes its tail.
    //
    // WP14 V2: AND WHY THE SEGMENT THAT ALREADY OBSERVED DOES NOT PAY IT TWICE.
    // `observed_exit` says the loop broke at the top of a pass, on a
    // cudaStreamSynchronize of this same stream that returned success, and that
    // nothing has been enqueued on it since -- the break is the next statement.
    // The accumulator D2H below is then the ONLY outstanding work this
    // synchronise would have waited for, so a blocking cudaMemcpy on an empty
    // stream reads the same bytes at the same point in the same order, in one
    // host call instead of two.  It is a shorter path to identical bytes, which
    // is the whole B0 claim; if the loop did NOT break on an observation (the
    // budget was spent, or the WHILE ran) the stream is live and the arm falls
    // through to the synchronise it always did.
    //
    // REFUSED IN A BATCH, BY NAME.  A blocking cudaMemcpy is issued on the
    // LEGACY stream, which implicitly synchronises with every blocking stream in
    // the context -- harmless when this segment owns the only one, and a sibling
    // deck's whole sweep to wait on when it does not.  That is a scheduling
    // hazard rather than a trajectory one, and it is refused rather than
    // measured: the async pair below is what a batch keeps.
    const bool v2_exit_drained = segment_v2 && observed_exit && batch_width <= 1;
    if (hostfree) {
        if (v2_exit_drained) {
            if (m.h_sweep_accum != nullptr && m.d_sweep_accum != nullptr &&
                (rc = rasbery::xfer::memcpy(
                     "CudaOuterGraph.cu:runSegment", "sweep accumulator (v2 drained)",
                     m.h_sweep_accum, m.d_sweep_accum + slot, sizeof(*m.h_sweep_accum),
                     cudaMemcpyDeviceToHost)) != cudaSuccess)
                return launchFailed("download the sweep accumulator", rc);
            bump(counters().v2_exit_syncs_elided);
        } else {
            if (m.h_sweep_accum != nullptr && m.d_sweep_accum != nullptr) {
                if ((rc = rasbery::xfer::memcpyAsync(
                         "CudaOuterGraph.cu:runSegment", "sweep accumulator",
                         m.h_sweep_accum, m.d_sweep_accum + slot, sizeof(*m.h_sweep_accum),
                         cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                    return launchFailed("download the sweep accumulator", rc);
            }
            bump(counters().sync_hostfree_exit);
            if ((rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:runSegment",
                                                "host-free exit", m.stream)) != cudaSuccess)
                return launchFailed("synchronize on the host-free segment exit", rc);
        }
        m.sweep_host_continued = false;
        if (!m.hooks.finish_cmfd_sweep_deferred(m.hooks.ctx, m.h_sweep_accum, slot))
            return hookFailed("the deferred CMFD sweep observation hook");

        // THE OBSERVATION MOVED Geometry::Phif's GENERATION, and the elision has
        // to be told.  finishDeferredDrives is handed PhifMutable(), which is
        // what makes it a declared writer, so without this re-read the NEXT
        // segment's first outer would find a generation it does not recognise
        // and re-upload the flux it already owns.  Same rule as the body's own
        // post-drive read: adopt the generation only when the device still owns
        // the bytes, forget the copy when the host loop took the drive.
        if (m.hooks.read_live_state != nullptr) {
            OuterSegmentLiveState after_obs;
            m.hooks.read_live_state(m.hooks.ctx, after_obs);
            m.resident_flux_generation =
                after_obs.device_owns_flux ? after_obs.flux_generation : 0;
        }

        // THE OUTER THE DEVICE ABANDONED STILL OWES ITS TAIL.
        //
        // sweep state 0 or 2: the verdict kernel raised the segment's halt, so
        // that outer's updjnet, nodal drive, upddhat, refresh, decision and
        // transition were all no-ops, and every outer enqueued behind it was a
        // no-op too.  The host has just finished the drive verbatim and
        // republishAfterHostSweep has taken the halt off, so the device state is
        // exactly what it was when that sweep ended -- which is where the tail
        // starts.  Running it here commits that outer, which is what the
        // per-outer arm does in place.
        //
        // ONE PASS AND THEN THE SEGMENT ENDS.  The outers past it were never
        // committed and the budget is spent; the caller's next segment picks up
        // from the state this one leaves.
        if (m.sweep_host_continued) {
            bump(counters().hostfree_repairs);
            // WP14 V2: THE REPAIR PUTS A WHOLE OUTER TAIL BACK ON THE STREAM,
            // and its transition writes d_segments.  So the fact the elisions
            // below are argued from -- "the stream is empty and `m.h_seg` is
            // current" -- stops being true here, and the flag that carries it
            // is retired rather than reasoned around.  The exit observation
            // reverts to the D2H it always made.
            observed_exit = false;
            double* const repair_reigv_slot =
                m.hooks.nodal_reigv_slot != nullptr
                    ? static_cast<double*>(m.hooks.nodal_reigv_slot(m.hooks.ctx))
                    : nullptr;
            const bool repair_canonical =
                bound_.canonical_nodal && m.hooks.canonical_nodal_eligible != nullptr &&
                m.hooks.canonical_nodal_eligible(m.hooks.ctx) != 0;
            const bool repair_bridge = !repair_canonical && bound_.host_jnet != nullptr &&
                                       bound_.device_jnet != nullptr;
            const std::size_t repair_jnet_bytes =
                repair_bridge ? static_cast<std::size_t>(bound_.geom.nsurf) *
                                    static_cast<std::size_t>(bound_.ng) * sizeof(double)
                              : 0;
            if (!runOuterTail(budget, repair_reigv_slot, repair_canonical, repair_bridge,
                              repair_jnet_bytes))
                return false;
        }
    }

    // --- the segment exit: psi and dhat back, ONCE ---------------------------
    //
    // THE ONE PLACE A HOST READER IS CERTAIN.  When runSegment returns, the
    // host ladder runs -- Xe, T/H, the search commit -- and may run a whole host
    // outer of its own on the `!outer_on_device` path, which reads _psi through
    // wiel and _dhat through setls.  Neither can be left holding what the device
    // wrote several outers ago.
    //
    // ONCE PER SEGMENT AND NOT ONCE PER OUTER, which is the whole saving: at a
    // budget of 8 on kngr_238 that is 374 pairs instead of 661, and the outers
    // inside a segment have no host reader between them by construction -- the
    // sweep hook is the only host call that reads either array, and the copies
    // above cover the outers where it takes the host loop.
    if (bound_.host_psi != nullptr && bound_.device_psi != nullptr &&
        bound_.host_dhat != nullptr && bound_.device_dhat != nullptr) {
        const std::size_t psi_bytes = static_cast<std::size_t>(bound_.nxyz) * sizeof(double);
        const std::size_t dhat_bytes = static_cast<std::size_t>(bound_.geom.nsurf) *
                                       static_cast<std::size_t>(bound_.ng) * sizeof(double);
        if ((rc = rasbery::xfer::memcpyAsync(
                 "CudaOuterGraph.cu:runSegment", "psi exit mirror", bound_.host_psi,
                 bound_.device_psi, psi_bytes, cudaMemcpyDeviceToHost, m.stream)) !=
            cudaSuccess)
            return launchFailed("mirror psi to the host at the segment exit", rc);
        if ((rc = rasbery::xfer::memcpyAsync(
                 "CudaOuterGraph.cu:runSegment", "dhat exit mirror", bound_.host_dhat,
                 bound_.device_dhat, dhat_bytes, cudaMemcpyDeviceToHost, m.stream)) !=
            cudaSuccess)
            return launchFailed("mirror dhat to the host at the segment exit", rc);
        bump(counters().host_mirror_bytes, mirrorPairBytes(bound_));
        bump(counters().mirror_exits);
    }

    // --- the segment exit: jnet and phis back, ONCE  (Task 18-lite) ---------
    //
    // THE OTHER HALF OF DROPPING THE BRIDGE.  While the binding was live the
    // device nodal wrote jnet and phis into the arena on every outer and neither
    // came home; every host consumer of those two arrays lives OUTSIDE a
    // segment, so this is the point -- and the only point -- at which they are
    // owed the bytes.  The reader list, audited and pinned by the contract test:
    //
    //   PPR               Driver.h: pin_power_reconstruction.reset(1/eigv,
    //                     Jnet(), Phif(), Phis()) at every statepoint.
    //   NormalizeFluxSign XSSet.cpp: negates Jnet and Phis in place, statepoint
    //                     level, so it both reads and writes them.
    //   OuterTrace        Driver.h: hashDoubles(Jnet(), nsg) under
    //                     RASBERY_OUTER_TRACE.
    //   the host outer body  CMFD::updjnet WRITES the whole of Jnet before the
    //                     host nodal drive reads it, so that reader is covered
    //                     by the write -- but only because the release above put
    //                     the ownership back and the drive uploads again.
    //
    // GATED ON `canonical_nodal_live` AND NOT ON `canonical_nodal`.  If the last
    // outer of this segment fell back to the CPU body, the host arrays are the
    // NEWER copy -- the CPU body wrote them and the bridge pushed jnet back --
    // and mirroring the device over them would overwrite new values with old.
    //
    // ONE D2H EACH PER SEGMENT against the bridge's two per OUTER: at a budget
    // of 8 that is a quarter of the traffic, and at 16 an eighth.
    if (m.canonical_nodal_live) {
        const std::size_t surf_bytes = static_cast<std::size_t>(bound_.geom.nsurf) *
                                       static_cast<std::size_t>(bound_.ng) * sizeof(double);
        if (bound_.host_jnet != nullptr && bound_.device_jnet != nullptr) {
            if ((rc = rasbery::xfer::memcpyAsync(
                     "CudaOuterGraph.cu:runSegment", "jnet exit mirror", bound_.host_jnet,
                     bound_.device_jnet, surf_bytes, cudaMemcpyDeviceToHost, m.stream)) !=
                cudaSuccess)
                return launchFailed("mirror jnet to the host at the segment exit", rc);
            bump(counters().jnet_mirror_bytes, surf_bytes);
        }
        if (bound_.host_phis != nullptr && bound_.device_phis != nullptr) {
            if ((rc = rasbery::xfer::memcpyAsync(
                     "CudaOuterGraph.cu:runSegment", "phis exit mirror", bound_.host_phis,
                     bound_.device_phis, surf_bytes, cudaMemcpyDeviceToHost, m.stream)) !=
                cudaSuccess)
                return launchFailed("mirror phis to the host at the segment exit", rc);
            bump(counters().phis_mirror_bytes, surf_bytes);
        }
    }

    // --- the halt does not outlive the segment -------------------------------
    //
    // Rev.7.1 Task 10 part 3.  It never had to before: `d_halt` was read only by
    // the body's own kernels, and the next segment's entry cleared it before any
    // of them ran, so a halt left standing between segments was invisible.  The
    // nodal halt gate makes it visible -- once installed it stays installed, for
    // the graph-key reason at the arm -- and a HOST outer's nodal drive between
    // two segments reads the same word.  Left raised by the transition that
    // ended this segment, it would mask that drive completely: five kernels that
    // return on their first instruction, a jnet nobody updated, and a run that
    // is finite, plausible and wrong.
    //
    // ONE 4-BYTE H2D PER SEGMENT, on a stream that is about to be synchronised
    // anyway, and issued for EVERY segment rather than only the host-free ones:
    // the gate is process-lived, so the segment that raises the halt is not
    // necessarily the one that installed it.
    {
        const std::uint32_t clear_exit_halt = 0u;
        if ((rc = rasbery::xfer::memcpyAsync(
                 "CudaOuterGraph.cu:runSegment", "exit halt clear", m.d_halt + slot,
                 &clear_exit_halt, sizeof(clear_exit_halt), cudaMemcpyHostToDevice,
                 m.stream)) != cudaSuccess)
            return launchFailed("clear the halt at the segment exit", rc);
    }

    // --- the single observation ---------------------------------------------
    DeviceOuterSegmentState seg_out{};
    DeviceSlotState         state_out{};
    unsigned long long      halted_out = 0;
    CmfdOuterDecision       decision_out{};
    // WP14 V2: THE 32 BYTES THAT ARE ALREADY ON THE HOST.
    //
    // `runOuterTail` D2H's this exact struct into the pinned `m.h_seg` behind
    // every transition, and the observation the loop broke on made the last
    // one visible.  Between that synchronise and here the stream has carried
    // the exit mirrors and a 4-byte halt clear and NOTHING THAT WRITES
    // d_segments -- the repair pass, which does, retires `observed_exit` where
    // it runs.  So the copy below would read bytes the host already holds, and
    // taking them from the pinned word is the same value by a shorter path.
    //
    // THE FALLBACK IS THE COPY, not an assumption: without `observed_exit` (a
    // budget exit, the WHILE arm, a repair) the exit word is stale by
    // construction and the D2H is the only correct read.
    const bool v2_seg_from_pin = segment_v2 && observed_exit && m.h_seg != nullptr;
    if (v2_seg_from_pin) {
        seg_out = *m.h_seg;
        bump(counters().v2_state_d2h_elided);
    } else if ((rc = rasbery::xfer::memcpyAsync(
                    "CudaOuterGraph.cu:runSegment", "exit segment state", &seg_out,
                    m.d_segments + slot, sizeof(seg_out), cudaMemcpyDeviceToHost,
                    m.stream)) != cudaSuccess) {
        return launchFailed("download segment state", rc);
    }
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "exit slot state",
                                  &state_out, m.arena.states + slot, sizeof(state_out),
                                  cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download slot state", rc);
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "exit halted count",
                                  &halted_out, m.d_halted, sizeof(halted_out),
                                  cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download halted count", rc);
    // The decision, for SolveLoop's ladder.  It rides the SAME observation as
    // the other three -- one more 24-byte copy on a stream that is about to be
    // synchronised anyway -- rather than becoming a second rendezvous.
    if ((rc = rasbery::xfer::memcpyAsync("CudaOuterGraph.cu:runSegment", "exit decision",
                                  &decision_out, m.d_decisions + slot,
                                  sizeof(decision_out), cudaMemcpyDeviceToHost,
                                  m.stream)) != cudaSuccess)
        return launchFailed("download outer decision", rc);
    bump(counters().sync_segment_exit);
    if ((rc = rasbery::xfer::streamSync("CudaOuterGraph.cu:runSegment", "exit drain",
                                m.stream)) != cudaSuccess)
        return launchFailed("segment synchronize", rc);

    // THE RELEASE RIDES THE SYNC THAT JUST HAPPENED, which is the whole reason it
    // is here and not at the call site: the exit mirror above was an ENQUEUE, and
    // only now are Geometry::Jnet and Geometry::Phis actually current.  A release
    // published one line earlier would let a host consumer read the arrays in the
    // window before the copies landed.
    releaseCanonicalNodal(true);

    if (seg_out.outer_in_segment == 0u) {
        // Nothing was committed, so there is nothing for the host to adopt and
        // returning true would advance the caller's loop by zero outers.  This is
        // a launch defect, not a physics result.
        bump(counters().refusals[static_cast<int>(OuterSegmentRefusal::LaunchFailed)]);
        return false;
    }

    if (hostfree) bump(counters().hostfree_outers, seg_out.outer_in_segment);
    // Rev.7.1 Task 10 part 4: WHAT THE WHILE DID, COUNTED WHERE THE HOST CAN SEE
    // IT.  The host was not present for the body iterations -- that is the whole
    // point -- so the count comes from the device: outer 0 was the eager one and
    // every committed outer after it was one iteration.  hostfree_enqueued gains
    // the same number, which is what makes hostfree_enqueued - hostfree_outers
    // exactly zero on this arm instead of the 21 (or 13,639) the stream arm pays.
    if (graph_ran) {
        bump(counters().graph_segments);
        const std::uint64_t iters =
            seg_out.outer_in_segment > 0u
                ? static_cast<std::uint64_t>(seg_out.outer_in_segment) - 1u
                : 0u;
        bump(counters().graph_iterations, iters);
        bump(counters().hostfree_enqueued, iters);
    }
    resume.device_outers      = seg_out.outer_in_segment;
    resume.next_phase         = seg_out.next_phase;
    resume.escape             = seg_out.escape;
    resume.flux_converged     = decision_out.flux_converged;
    resume.eigv               = state_out.eigv;
    resume.residual           = state_out.flux_l2;
    resume.prev_inner         = state_out.previous_eigv;
    resume.flux_stall         = state_out.flux_stall;
    resume.stall_events       = state_out.stall_events;
    resume.stall_sample_taken = state_out.stall_sample_taken;
    resume.clean_iters        = state_out.clean_iters;
    resume.xe_interim_count   = state_out.xe_interim_count;
    resume.total_outer        = state_out.total_outer;

    bump(counters().segment_launches);
    bump(counters().device_outers, seg_out.outer_in_segment);
    bump(counters().host_outer_observations);
    if (seg_out.escape < static_cast<std::uint32_t>(kDeviceEscapeCount))
        bump(counters().escapes[seg_out.escape]);
    // WP14: the exit reason the escape could not name.  `next_phase` is written
    // by the same transition that wrote the escape, in the same 32 bytes, so
    // this costs one bounds test and one relaxed increment on a path that runs
    // once per SEGMENT.
    if (seg_out.next_phase < static_cast<std::uint32_t>(kDevicePhaseCount))
        bump(counters().exit_reasons[seg_out.next_phase]);
    if (seg_out.escape == static_cast<std::uint32_t>(DeviceEscape::SegmentBudget))
        bump(counters().budget_exits);
    // The device counter is cumulative across segments; publish the delta.
    static thread_local unsigned long long halted_seen = 0;
    if (halted_out > halted_seen) {
        bump(counters().halted_outer_launches, halted_out - halted_seen);
        halted_seen = halted_out;
    }
    return true;
}


// ---------------------------------------------------------------------------
// Standing the segment up  (Rev.7.1 Task 9, links 1/3/4/5)
// ---------------------------------------------------------------------------

namespace {

/// The one arena of the single-run path.
///
/// A function-local static rather than a member of anything: the arena outlives
/// every Driver in the process -- its whole contract is that its addresses never
/// move -- and giving it an owner would mean giving it that owner's lifetime.
GpuPhysicsArena& outerArena() {
    static GpuPhysicsArena arena;
    return arena;
}

/// The device tables the arena does not own.
///
/// ALLOCATED ONCE AND NEVER AGAIN, which is the arena's own rule and for the
/// arena's own reason: Task 10 captures a graph over these pointers, and a table
/// that moved would invalidate it.  They are NOT carved out of the arena block
/// because the layout has no region for them, and adding one would move every
/// offset in a layout that has its own gate.
struct StandUpTables {
    DeviceSlotView*      slot_views    = nullptr; ///< [slots] = DeviceArenaView::slot_views
    cmfd::CmfdOuterView* cmfd_views    = nullptr; ///< [slots], derived ON THE DEVICE
    CmfdOuterCounters*   dhat_counters = nullptr; ///< [slots]
    bool                 stood         = false;
    /// Rev.7.1 Task 18-lite: the width the arena was ACTUALLY stood at, and the
    /// deck shape it was stood on.
    ///
    /// The arena is one allocation whose slot block is replicated from one
    /// ArenaDims, so every slot carries the FIRST deck's strides.  Recording the
    /// shape is what lets a later Driver be told `geometry_mismatch` instead of
    /// being handed a slot indexed with somebody else's mesh.  Written once,
    /// under the stand-up mutex, before `stood` goes true; read afterwards
    /// without the lock, which is safe because `stood` is what publishes it.
    int                   slots = 0;
    OuterSegmentDeckShape shape{};
};

StandUpTables& standUpTables() {
    static StandUpTables t;
    return t;
}

/// One geometry import, sized from the layout calculator rather than recomputed.
///
/// Recomputing an element count at the call site is how an import silently
/// copies half an array: the layout is the only thing that knows how big a
/// region is, and it is already unit-tested with no CUDA.
template <typename T>
bool importGeometryRegion(GeometryRegion region, const T* host, const ArenaDims& dims,
                          const char* what) {
    const std::size_t bytes = arenaGeometryElements(region, dims) * sizeof(T);
    if (!outerArena().importGeometryAsync(region, host, bytes, nullptr)) {
        std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] geometry import failed (%s): %s\n",
                     what, outerArena().status().c_str());
        return false;
    }
    return true;
}

} // namespace

bool rasberyStandUpOuterSegment(const OuterSegmentDeck& deck, std::ostream& receipt) {
    // The gate is tested HERE and not at the call site, so Driver.h holds one
    // unconditional call and the feature-off path is this early return.
    if (!outerGpuEnabled()) return false;

    // ONE STAND-UP PER PROCESS MEANS ONE AT A TIME.
    //
    // Driver::Run calls this, and `--batch-mode M` runs M Drivers on M host
    // threads: `stood` is a plain bool read and written with no ordering, so two
    // workers that arrive together both read false, both reserve the arena, and
    // both walk the cudaMalloc/bind sequence over the SAME process-wide objects.
    // MEASURED on a 4-deck local batch with RASBERY_GPU_OUTER=1: two identical
    // [RASBERY][GPU_ARENA] receipts, one per racing worker, both reporting the
    // same pre-allocation vram_free -- i.e. two 237 MB reservations of a
    // single-allocation arena, one of them leaked, and a runner left bound to
    // whichever finished last.  The segment refuses in batch anyway, so nothing
    // NUMERIC came of it; that is not a reason to leave the allocation racing.
    //
    // The lock is held for the whole body rather than around `stood` alone,
    // because the later arrivals must see a FINISHED stand-up, not one in
    // progress.  It is taken once per Driver and the fast path is one
    // uncontended lock plus a bool.
    static std::mutex stand_up_mutex;
    std::lock_guard<std::mutex> stand_up_lock(stand_up_mutex);

    // A LATER ARRIVAL DOES NOT RE-STAND ANYTHING -- it checks that the arena it
    // is about to share was stood on ITS deck.  The caller ignores this return,
    // so the enforcing gate is the ladder's `geometry_mismatch`, which
    // rasberyOuterSlotAdmitted answers from the same recorded shape; saying it
    // once here, loudly, is what stops that refusal from reading like the
    // feature quietly doing nothing.
    if (standUpTables().stood) {
        if (standUpTables().shape != deck.shape()) {
            std::fprintf(stderr,
                         "[RASBERY][OUTER_GPU][WARN] the device outer arena was stood up on "
                         "nxyz=%d nsurf=%d nxy=%d n_fuel=%d ng=%d and this deck is nxyz=%d "
                         "nsurf=%d nxy=%d n_fuel=%d ng=%d; one arena layout cannot serve "
                         "both, so this deck refuses with geometry_mismatch\n",
                         standUpTables().shape.nxyz, standUpTables().shape.nsurf,
                         standUpTables().shape.nxy, standUpTables().shape.n_fuel,
                         standUpTables().shape.ng, deck.nxyz, deck.nsurf, deck.nxy,
                         deck.n_fuel, deck.ng);
            return false;
        }
        return rasberyOuterSegment().bound();
    }

    if (deck.nxyz <= 0 || deck.nsurf <= 0 || deck.nxy <= 0 || deck.ng <= 0 ||
        deck.surface_node == nullptr || deck.surface_dir == nullptr ||
        deck.node_hmesh == nullptr || deck.node_volume == nullptr ||
        deck.boundary_albedo == nullptr) {
        std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] incomplete deck description; the "
                             "device outer stays off\n");
        return false;
    }
    // Sec 3.3: the device bodies are built for a two-group deck (kDevNg), and a
    // three-group deck would index every packed block with the wrong stride.
    // Refusing is the only safe answer, and it has to be loud.
    if (deck.ng != kDevNg) {
        std::fprintf(stderr,
                     "[RASBERY][OUTER_GPU][WARN] ng=%d but the device bodies are built for "
                     "%d groups; the device outer stays off\n",
                     deck.ng, kDevNg);
        return false;
    }

    // --- 1. the ONE allocation, with Sec 4.4 admission -----------------------
    // --- 0. how wide ---------------------------------------------------------
    //
    // Rev.7.1 Task 18-lite: THE ARENA IS THE RUN'S WIDTH, NOT ONE.
    //
    // Link 1 stood this up at width 1 and the refusal ladder then refused every
    // batch, which is a consistent pair and a dead end: `--batch-mode M` got
    // nothing from the device outer at all.  The width the run asked for is the
    // width every other batched arena in this process already uses -- the CMFD
    // arena (rasberyBatchArena) and the batched nodal arena both size themselves
    // from rasberyBatchWidth() -- so taking the same number keeps ONE slot index
    // space across all three, and the Driver can hand its CMFD slot straight to
    // the physics arena instead of inventing a second mapping.
    //
    // A WIDER ARENA IS AN ADMISSION QUESTION, NOT A PROMISE.  reserve() applies
    // Sec 4.4 exactly as before and nothing is silently shrunk: if M slots do not
    // fit it refuses, `[RASBERY][GPU_ARENA]` says `admitted:false` and why, and
    // no runner is ever initialised -- so the ladder answers `no_runner`, which
    // is ranked above the batch and is the more useful of the two true things.
    // At the kngr_238 mesh a slot is ~235 MB, so M64 is ~15 GB of arena and this
    // is the path a wide manifest takes on a card that cannot hold it.  Serving
    // K < M slots and refusing the rest is deliberately NOT done: an
    // inhomogeneous batch is a scheduling decision, not an allocator's.
    const int arena_width = rasberyBatchWidth() > 0 ? rasberyBatchWidth() : 1;
    ArenaDims dims = arenaDims(deck.nxyz, deck.nsurf, deck.nxy, deck.n_fuel, arena_width);
    dims.ng             = deck.ng;
    const bool reserved = outerArena().reserve(dims);
    // The receipt goes out on BOTH paths.  A refusal that printed nothing is
    // exactly the silent-shrink failure Sec 4.4 was written against.
    outerArena().emitReceipt(receipt);
    if (!reserved) {
        std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] arena refused: %s\n",
                     outerArena().status().c_str());
        return false;
    }

    // --- 2. the immutable topology, uploaded once ----------------------------
    if (!importGeometryRegion(GeometryRegion::Lklr, deck.surface_node, dims, "surface_node") ||
        !importGeometryRegion(GeometryRegion::Idirlr, deck.surface_dir, dims, "surface_dir") ||
        !importGeometryRegion(GeometryRegion::Hmesh, deck.node_hmesh, dims, "node_hmesh") ||
        !importGeometryRegion(GeometryRegion::Vol, deck.node_volume, dims, "node_volume") ||
        !importGeometryRegion(GeometryRegion::Albedo, deck.boundary_albedo, dims, "albedo")) {
        outerArena().release();
        return false;
    }

    // --- 3. the slot-view table, which is what a DeviceArenaView IS ----------
    StandUpTables& t  = standUpTables();
    const int      n  = dims.slots;
    cudaError_t    rc = cudaSuccess;

    auto fail = [&](const char* what) {
        std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] %s: %s\n", what,
                     cudaGetErrorString(rc));
        rasberyTearDownOuterSegment();
        return false;
    };

    rasbery::AllocWindow _alloc_window("outer.tables.standup");
    if ((rc = cudaMalloc(&t.slot_views, sizeof(DeviceSlotView) * n)) != cudaSuccess)
        return fail("cudaMalloc(slot_views)");
    if ((rc = cudaMalloc(&t.cmfd_views, sizeof(cmfd::CmfdOuterView) * n)) != cudaSuccess)
        return fail("cudaMalloc(cmfd_views)");
    if ((rc = cudaMalloc(&t.dhat_counters, sizeof(CmfdOuterCounters) * n)) != cudaSuccess)
        return fail("cudaMalloc(dhat_counters)");
    if ((rc = cudaMemset(t.dhat_counters, 0, sizeof(CmfdOuterCounters) * n)) != cudaSuccess)
        return fail("cudaMemset(dhat_counters)");

    // slotView() is a pure index rebase on the host, so the table is built here
    // and uploaded once; after this the DEVICE never asks the host for an
    // address again, which is the precondition for capturing a graph over them.
    std::vector<DeviceSlotView> host_views(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        host_views[static_cast<std::size_t>(i)] = outerArena().slotView(i);
    if ((rc = rasbery::xfer::memcpy("CudaOuterGraph.cu:rasberyStandUpOuterSegment", "slot_views",
                            t.slot_views, host_views.data(), sizeof(DeviceSlotView) * n,
                            cudaMemcpyHostToDevice)) != cudaSuccess)
        return fail("upload slot_views");

    // The four control arrays are DENSE and below slot_base, so slot 0's
    // pointers ARE the array bases (GpuPhysicsArenaLayout.h ControlRegion note).
    // Taking them from slotView(0) rather than recomputing an offset is what
    // keeps this agreeing with the layout gate.
    DeviceArenaView arena{};
    arena.slot_views = t.slot_views;
    arena.phases     = outerArena().slotView(0).phase;
    arena.states     = outerArena().slotView(0).state;
    arena.searches   = outerArena().slotView(0).search;
    arena.params     = outerArena().slotView(0).params;
    arena.slot_count = n;

    // --- 4. derive the CMFD table FROM the arena, on the device --------------
    if ((rc = enqueueBuildCmfdSlotTable(arena, t.cmfd_views, n, dims.ng, dims.nxyz,
                                        nullptr)) != cudaSuccess)
        return fail("build cmfd slot table");

    // --- 5. seed the control packet (link 5) ---------------------------------
    // XSSet::_hoststate_generation starts at 1 and only ever increments, so a
    // zero here means the caller did not fill the field.  Clamping to 1 keeps
    // the seed kernel out of the one input that would wrap its `gen - 1` to
    // UINT64_MAX and leave the constants phase permanently stale.
    const unsigned long long seed_generation =
        deck.material_generation != 0ull ? deck.material_generation : 1ull;
    //
    // EVERY SLOT, NOT SLOT 0.  Rev.7.1 Task 18-lite: at width 1 the two were the
    // same statement and this read `..., 0, ...`.  At the run's width they are
    // not, and an unseeded slot is exactly the failure this kernel's own comment
    // describes -- cmfdOuterConvergence BRANCHES on flux_stall, stall_events,
    // clean_iters and xe_interim_count, so a slot whose control packet still
    // holds pool garbage publishes `prev_inner = eigv + 1.0` and SolveLoop adopts
    // it.  MEASURED on the 4-deck local batch: 56 datasets per deck differed
    // against OUTER unset, all four decks, with the receipt saying the segment
    // engaged cleanly on every slot.  With the loop: 0.
    for (int i = 0; i < n; ++i) {
        k_outer_seed_slot<<<1, 1>>>(arena, i, seed_generation);
        if ((rc = cudaGetLastError()) != cudaSuccess) return fail("seed slot state");
    }
    if ((rc = rasbery::xfer::deviceSync("CudaOuterGraph.cu:rasberyStandUpOuterSegment",
                                "stand-up")) != cudaSuccess)
        return fail("stand-up synchronize");

    // --- 6. hand it to the runners -------------------------------------------
    //
    // ONE PER SLOT.  Each holds its own residency, hook set, binding and the two
    // sticky latches, which is the whole reason a batch can be served at all;
    // the device scratch each allocates is [slot_count] copies of six small
    // structs, so even the widest arena this build allows costs a few hundred
    // kilobytes of it.
    for (int i = 0; i < n; ++i) {
        if (!rasberyOuterSegment(i).initialize(arena, n, i)) {
            std::fprintf(stderr,
                         "[RASBERY][OUTER_GPU][WARN] runner initialise failed (slot %d): %s\n",
                         i, rasberyOuterSegment(i).status().c_str());
            rasberyTearDownOuterSegment();
            return false;
        }
    }

    OuterSegmentBinding binding{};
    binding.geom.surface_node = static_cast<const int*>(
        outerArena().geometryRegion(GeometryRegion::Lklr));
    binding.geom.surface_dir = static_cast<const int*>(
        outerArena().geometryRegion(GeometryRegion::Idirlr));
    binding.geom.node_hmesh = static_cast<const double*>(
        outerArena().geometryRegion(GeometryRegion::Hmesh));
    binding.geom.node_volume = static_cast<const double*>(
        outerArena().geometryRegion(GeometryRegion::Vol));
    binding.geom.boundary_albedo = static_cast<const double*>(
        outerArena().geometryRegion(GeometryRegion::Albedo));
    binding.geom.nxyz  = dims.nxyz;
    binding.geom.nsurf = dims.nsurf;
    binding.geom.ng    = dims.ng;

    binding.geometry      = outerArena().geometryView();
    binding.table         = CmfdOuterSlotTable{t.cmfd_views, n};
    binding.forms         = cmfd::cmfdOuterFormsRuntime();
    binding.dhat_clamp    = deck.dhat_clamp;
    binding.dhat_counters = t.dhat_counters;
    // The device end of the jnet bridge.  jnet has no twin in the sweep arena,
    // so it stays in the physics arena and the runner moves it around the host
    // nodal drive.
    binding.device_jnet   = outerArena().slotView(0).jnet;  // re-aimed per slot below
    // Rev.7.1 Task 18-lite: the phis half of the canonical nodal set.
    //
    // THE ARENA HAS A phis REGION AND NOTHING ON THE DEVICE READS IT.  Every
    // other slot region is written by a CMFD body or read by one; SlotRegion::Phis
    // is the one Geometry array the device outer never touches, because phis is a
    // NODAL output and the CMFD side has no use for it.  That is what makes it
    // the right buffer to hand the nodal drive: the sharing costs no coupling,
    // and the region is already sized as Geometry sizes it (LR*ng*NDIRMAX*nxyz,
    // GpuPhysicsArenaLayout.h:485).
    binding.device_phis   = outerArena().slotView(0).phis;  // re-aimed per slot below
    binding.nxyz          = dims.nxyz;
    binding.ng            = dims.ng;

    // THE CANONICAL NODAL SET IS PER SLOT, AND THAT IS WHAT THIS TASK IS ABOUT.
    // jnet and phis used to come from slotView(0) for the whole process, so in a
    // batch every Driver adopted the SAME two device buffers as its backend's
    // canonical nodal set and the batched nodal drive stopped having a per-deck
    // jnet and phis.  Each runner now takes its own slot's regions; everything
    // above -- the geometry, the CMFD view table, the forms, the clamp -- is
    // genuinely shared and is copied across unchanged.
    for (int i = 0; i < n; ++i) {
        OuterSegmentBinding b = binding;
        b.device_jnet         = outerArena().slotView(i).jnet;
        b.device_phis         = outerArena().slotView(i).phis;
        rasberyOuterSegment(i).bind(b);
    }

    t.slots = n;
    t.shape = deck.shape();
    t.stood = true;
    return rasberyOuterSegment().bound();
}

int rasberyOuterArenaSlots() {
    const StandUpTables& t = standUpTables();
    return t.stood ? t.slots : 0;
}

bool rasberyOuterSlotAdmitted(int slot, const OuterSegmentDeckShape& shape) {
    const StandUpTables& t = standUpTables();
    // NOT STOOD IS NOT ADMITTED, and it does not have to say so: every reason
    // ranked above `slot_admitted` in the ladder -- no runner, no arena, unbound
    // -- is already false in that state and is reported first.
    if (!t.stood) return false;
    if (slot < 0 || slot >= t.slots) return false;
    return t.shape == shape;
}

void rasberyTearDownOuterSegment() {
    rasbery::AllocWindow _alloc_window("outer.tables.teardown");
    StandUpTables& t = standUpTables();
    for (int i = 0; i < kMaxDeviceSlots; ++i) rasberyOuterSegment(i).release();
    if (t.slot_views != nullptr) cudaFree(t.slot_views);
    if (t.cmfd_views != nullptr) cudaFree(t.cmfd_views);
    if (t.dhat_counters != nullptr) cudaFree(t.dhat_counters);
    t.slot_views    = nullptr;
    t.cmfd_views    = nullptr;
    t.dhat_counters = nullptr;
    t.stood         = false;
    t.slots         = 0;
    t.shape         = OuterSegmentDeckShape{};
    outerArena().release();
}

} // namespace rasbery::gpu
