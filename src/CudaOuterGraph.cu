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

#include "CudaTransferMirror.h"
#include "GpuPhysicsArena.h"
#include "OuterTrace.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    std::atomic<std::uint64_t> phis_mirror_bytes{0};
    std::atomic<std::uint64_t> jnet_mirror_bytes{0};
    std::atomic<std::uint64_t> refusals[static_cast<int>(OuterSegmentRefusal::Count)];
    std::atomic<std::uint64_t> escapes[kDeviceEscapeCount];

    AtomicCounters() {
        for (auto& c : refusals) c.store(0, std::memory_order_relaxed);
        for (auto& c : escapes) c.store(0, std::memory_order_relaxed);
    }
};

AtomicCounters& counters() {
    static AtomicCounters c;
    return c;
}

void bump(std::atomic<std::uint64_t>& c, std::uint64_t by = 1) {
    c.fetch_add(by, std::memory_order_relaxed);
}

} // namespace

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

bool outerGpuEnabled() {
    static const bool on = envFlagOn("RASBERY_GPU_OUTER");
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

OuterSegmentCounters outerSegmentCounters() {
    const AtomicCounters& a = counters();
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
    out.phis_mirror_bytes       = a.phis_mirror_bytes.load(std::memory_order_relaxed);
    out.jnet_mirror_bytes       = a.jnet_mirror_bytes.load(std::memory_order_relaxed);
    for (int i = 0; i < static_cast<int>(OuterSegmentRefusal::Count); ++i)
        out.refusals[i] = a.refusals[i].load(std::memory_order_relaxed);
    for (int i = 0; i < kDeviceEscapeCount; ++i)
        out.escapes[i] = a.escapes[i].load(std::memory_order_relaxed);
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
                    ",\"phis_mirror_bytes\":" + std::to_string(c.phis_mirror_bytes) +
                    ",\"jnet_mirror_bytes\":" + std::to_string(c.jnet_mirror_bytes) +
                    ",\"segment_budget\":" + std::to_string(outerSegmentBudget());

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
    s += "},";
    // Printed ONLY when nothing ran: on a healthy run the reason is "none" and
    // saying so would be noise, while on an idle run it is the whole message.
    if (c.segment_launches == 0) {
        s += outerIdleReasonJson(
            rasberyOuterSegment().refusal(rasberyBatchWidth(), false, false));
        s += ",";
    }
    s += outerHostBodyJson();
    s += "}";
    return s;
}

void reportOuterSegment(std::ostream& os) {
    if (!outerGpuEnabled()) return;
    os << "[RASBERY][OUTER_GPU] " << outerSegmentReceiptJson() << std::endl;
}

void noteOuterSegmentRefusal(OuterSegmentRefusal why) {
    if (why == OuterSegmentRefusal::None) return;
    bump(counters().refusals[static_cast<int>(why)]);
}

// ---------------------------------------------------------------------------
// The runner
// ---------------------------------------------------------------------------

namespace {

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
};

CudaOuterSegment::CudaOuterSegment() : _impl(new Impl) {}

CudaOuterSegment::~CudaOuterSegment() {
    release();
    delete _impl;
}

bool CudaOuterSegment::initialize(const DeviceArenaView& arena, int slot_count) {
    release();
    if (slot_count <= 0 || slot_count > kMaxSchedulerSlots) {
        _impl->status = "slot_count outside [1, kMaxSchedulerSlots]";
        return false;
    }
    _impl->arena      = arena;
    _impl->slot_count = slot_count;

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

    if ((rc = cudaMemset(_impl->d_halted, 0, sizeof(unsigned long long))) != cudaSuccess)
        return fail("cudaMemset(halted)", rc);

    _impl->ready  = true;
    _impl->status = "ready";
    return true;
}

void CudaOuterSegment::release() {
    if (_impl == nullptr) return;
    if (_impl->d_probes != nullptr) cudaFree(_impl->d_probes);
    if (_impl->d_segments != nullptr) cudaFree(_impl->d_segments);
    if (_impl->d_halt != nullptr) cudaFree(_impl->d_halt);
    if (_impl->d_inputs != nullptr) cudaFree(_impl->d_inputs);
    if (_impl->d_decisions != nullptr) cudaFree(_impl->d_decisions);
    if (_impl->d_halted != nullptr) cudaFree(_impl->d_halted);
    if (_impl->h_seg != nullptr) cudaFreeHost(_impl->h_seg);
    if (_impl->own_stream != nullptr) cudaStreamDestroy(_impl->own_stream);
    _impl->d_probes    = nullptr;
    _impl->d_segments  = nullptr;
    _impl->d_halt      = nullptr;
    _impl->d_inputs    = nullptr;
    _impl->d_decisions = nullptr;
    _impl->d_halted    = nullptr;
    _impl->h_seg       = nullptr;
    _impl->stream      = nullptr;
    _impl->own_stream  = nullptr;
    _impl->ready       = false;
}

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
    a.valid    = true;
    return a;
}

bool CudaOuterSegment::available() const { return _impl != nullptr && _impl->ready; }

const std::string& CudaOuterSegment::status() const { return _impl->status; }

void CudaOuterSegment::setHooks(const OuterSegmentHooks& hooks) { _impl->hooks = hooks; }

OuterSegmentHooks CudaOuterSegment::hooks() const { return _impl->hooks; }

bool CudaOuterSegment::bindResidency(const OuterSegmentResidency& residency) {
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
    if (rc == cudaSuccess) rc = cudaDeviceSynchronize();
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
    if ((rc = cudaMemcpy(residency.dhat, residency.host_dhat, seed_dhat_bytes,
                         cudaMemcpyHostToDevice)) != cudaSuccess ||
        (rc = cudaMemcpy(residency.psi, residency.host_psi, seed_psi_bytes,
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
    const cudaError_t rc = cudaMemcpy(_impl->d_probes + slot, &probe, sizeof(probe),
                                      cudaMemcpyHostToDevice);
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
    const std::uint32_t clear = 0u;
    const cudaError_t   rc =
        cudaMemcpy(_impl->d_halt + slot, &clear, sizeof(clear), cudaMemcpyHostToDevice);
    return rc == cudaSuccess;
}

bool rasberyBindOuterResidency(const OuterSegmentResidency& residency) {
    return rasberyOuterSegment().bindResidency(residency);
}

bool rasberyPublishOuterProbe(int slot, double eigv, double residual, bool negative_flux,
                              bool rayleigh) {
    return rasberyOuterSegment().publishProbe(slot, eigv, residual, negative_flux, rayleigh);
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
                                              bool critical_search) const {
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
    const OuterSegmentRefusal why = refusal(batch_width, fractional_rods, critical_search);
    if (why != OuterSegmentRefusal::None) {
        bump(counters().refusals[static_cast<int>(why)]);
        return false;
    }
    if (scalars.slot < 0 || scalars.slot >= _impl->slot_count) {
        bump(counters().refusals[static_cast<int>(OuterSegmentRefusal::LaunchFailed)]);
        return false;
    }

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
    auto releaseCanonicalNodal = [&](bool stream_ordered) {
        if (!m.canonical_nodal_live) return;
        m.canonical_nodal_live = false;
        if (!stream_ordered) {
            const std::size_t surf_bytes =
                static_cast<std::size_t>(bound_.geom.nsurf) *
                static_cast<std::size_t>(bound_.ng) * sizeof(double);
            if (bound_.host_jnet != nullptr && bound_.device_jnet != nullptr)
                cudaMemcpy(bound_.host_jnet, bound_.device_jnet, surf_bytes,
                           cudaMemcpyDeviceToHost);
            if (bound_.host_phis != nullptr && bound_.device_phis != nullptr)
                cudaMemcpy(bound_.host_phis, bound_.device_phis, surf_bytes,
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

    cudaError_t rc;
    // The host mirror of the exit word is the loop's stopping condition, and it
    // survives the previous segment.  Clearing it here rather than trusting the
    // first D2H is what stops a converged segment's exit from breaking the NEXT
    // segment out at its second outer.
    if (m.h_seg != nullptr) *m.h_seg = seg;
    if ((rc = cudaMemcpyAsync(m.d_segments + slot, &seg, sizeof(seg), cudaMemcpyHostToDevice,
                              m.stream)) != cudaSuccess)
        return launchFailed("upload segment state", rc);
    if ((rc = cudaMemcpyAsync(m.d_probes + slot, &probe, sizeof(probe), cudaMemcpyHostToDevice,
                              m.stream)) != cudaSuccess)
        return launchFailed("upload probe", rc);
    if ((rc = cudaMemcpyAsync(m.d_halt + slot, &clear_halt, sizeof(clear_halt),
                              cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
        return launchFailed("clear halt", rc);
    if ((rc = cudaMemcpyAsync(m.d_inputs + slot, &inputs, sizeof(inputs),
                              cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
        return launchFailed("upload outer inputs", rc);

    // prev_inner lives in DeviceSlotState::previous_eigv and the convergence
    // body reads it through cmfdLoadOuterState.  The host owns it BETWEEN
    // segments (Driver.h's `prev_inner` local); inside one, the device does.
    if ((rc = cudaMemcpyAsync(&m.arena.states[slot].previous_eigv, &scalars.prev_inner,
                              sizeof(double), cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
        return launchFailed("upload prev_inner", rc);

    // flux_stall lives in DeviceSlotState::flux_stall and is read through the
    // same cmfdLoadOuterState.  Uploaded for exactly prev_inner's reason: it is
    // a SolveLoop local the device carries INSIDE a segment and the host owns
    // BETWEEN segments.  Left unseeded, the device counted from the previous
    // segment's leftovers and could not stop at the outer the host's
    // limit-cycle test would have stopped at -- so a budget-8 segment ran up to
    // eight times as far past a stalling trial point as the host ever would.
    if ((rc = cudaMemcpyAsync(&m.arena.states[slot].flux_stall, &scalars.flux_stall,
                              sizeof(std::uint32_t), cudaMemcpyHostToDevice, m.stream)) !=
        cudaSuccess)
        return launchFailed("upload flux_stall", rc);

    // --- prologue: the nodal constants, once ---------------------------------
    //
    // ONCE PER SEGMENT, NOT ONCE PER OUTER, and that is Driver.h's shape:
    // Nodal::updateConstant re-runs when the macro-XS move, and the only in-outer
    // XS write is rod cusping -- which an eligible slot cannot do.
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
        if (cudaStreamSynchronize(m.stream) != cudaSuccess) { cudaGetLastError(); return 0ull; }
        if (m.trace_scratch.size() < n) m.trace_scratch.resize(n);
        if (cudaMemcpy(m.trace_scratch.data(), dev, n * sizeof(double),
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
        if (cudaStreamSynchronize(m.stream) != cudaSuccess) { cudaGetLastError(); return 0.0; }
        if (cudaMemcpy(&p, m.d_probes + slot, sizeof(p), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            cudaGetLastError();
            return 0.0;
        }
        return p.eigv;
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
    for (unsigned int i = 0; i < budget; ++i) {
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
        if (i > 0) {
            if ((rc = cudaStreamSynchronize(m.stream)) != cudaSuccess)
                return launchFailed("synchronize on the segment exit", rc);
            if (m.h_seg != nullptr && m.h_seg->exit != 0u) break;
        }

        // THE LIVE STATE, RE-READ PER OUTER.
        //
        // Not once per segment: a segment with a budget above one runs outers
        // 2..N without returning to the host, but the host DID run between
        // them -- the sweep hook, the nodal drive and cusping are all host
        // calls -- and each of them can move a generation.  Deciding an
        // elision from a segment-entry value is what made i-SMR CY02 fail at
        // b8 and b16 while passing at b1.
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
            if ((rc = cudaMemcpyAsync(m.residency.flux, bound_.host_flux, flux_bytes,
                                      cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
                return launchFailed("upload flux", rc);
            bump(counters().flux_sync_bytes, flux_bytes);
        }
        // xsnf, gated on the BYTES and not on a generation -- see
        // Impl::resident_xsnf for the failure a generation gate produced.
        //
        // updpsi is the step that makes this load-bearing: it is
        // psi = flux . xsnf . vol, it runs three lines below, and it is the ONE
        // reader of the device xsnf that runs before the sweep's own upload
        // (CudaBICGBackend.cu's push_pending, which has always been byte-exact
        // through Slot::xsnf_mirror) can correct it.  A missed upload here is
        // therefore not a stale-by-one-outer psi, it is a psi built from the
        // wrong cross sections while the rest of the outer uses the right ones.
        const std::size_t xsnf_count = static_cast<std::size_t>(bound_.nxyz) *
                                       static_cast<std::size_t>(bound_.ng);
        const bool xsnf_current =
            bound_.host_xsnf != nullptr && bound_.device_xsnf != nullptr &&
            m.resident_xsnf.matches(bound_.host_xsnf, xsnf_count);
        if (xsnf_current) bump(counters().xsnf_uploads_elided);
        if (bound_.host_xsnf != nullptr && bound_.device_xsnf != nullptr && !xsnf_current) {
            const std::size_t xsnf_bytes = xsnf_count * sizeof(double);
            if ((rc = cudaMemcpyAsync(bound_.device_xsnf, bound_.host_xsnf, xsnf_bytes,
                                      cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
                return launchFailed("upload xsnf", rc);
            bump(counters().flux_sync_bytes, xsnf_bytes);
            // COMMITTED AT THE ISSUE, and that is safe HERE for a reason the
            // general rule (CudaTransferMirror.h: commit only after the copy has
            // landed) does not cover: the shadow records the bytes this copy was
            // handed, and the only host writer of _xs that can run before the
            // outer's synchronise is ApplyRodCusping -- which runs after it, in
            // the nodal drive's window.  A writer that appears between the two
            // would be caught by the NEXT outer's memcmp, never elided.
            m.resident_xsnf.commit(bound_.host_xsnf, xsnf_count);
        }
        // upddtil() is the only writer of _dtil and it runs once per SolveLoop
        // entry, plus once for every cusping that fires.  On a still deck this
        // copy happens once and then never again.
        const bool dtil_current = m.hooks.read_live_state != nullptr &&
                                  m.resident_dtil_generation != 0 &&
                                  m.resident_dtil_generation == live.dtil_generation;
        if (dtil_current) bump(counters().dtil_uploads_elided);
        if (bound_.host_dtil != nullptr && bound_.device_dtil != nullptr && !dtil_current) {
            const std::size_t dtil_bytes =
                static_cast<std::size_t>(bound_.geom.nsurf) *
                static_cast<std::size_t>(bound_.ng) * sizeof(double);
            if ((rc = cudaMemcpyAsync(bound_.device_dtil, bound_.host_dtil, dtil_bytes,
                                      cudaMemcpyHostToDevice, m.stream)) != cudaSuccess)
                return launchFailed("upload dtil", rc);
            bump(counters().flux_sync_bytes, dtil_bytes);
            m.resident_dtil_generation = live.dtil_generation;
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
            if ((rc = cudaMemcpyAsync(bound_.host_dhat, bound_.device_dhat, dhat_bytes,
                                      cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                return launchFailed("mirror dhat to the host", rc);
            bump(counters().host_mirror_bytes, dhat_bytes);
        }
        if (host_reader_next && bound_.host_psi != nullptr && bound_.device_psi != nullptr) {
            const std::size_t psi_bytes =
                static_cast<std::size_t>(bound_.nxyz) * sizeof(double);
            if ((rc = cudaMemcpyAsync(bound_.host_psi, bound_.device_psi, psi_bytes,
                                      cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                return launchFailed("mirror psi to the host", rc);
            bump(counters().host_mirror_bytes, psi_bytes);
        }
        // A SYNCHRONISING SWEEP HOOK IS STILL A HOST CALL.  Kept for the arm
        // that has no stream-ordered drive (no arena, or the Wielandt warm-up),
        // where the budget is forced to one and this is the only sync of the
        // segment.
        if (!stream_sweep && (rc = cudaStreamSynchronize(m.stream)) != cudaSuccess)
            return launchFailed("synchronize before the CMFD sweep", rc);

        // (2,3) setls + drive -- Driver.h:1551-1555.  On the stream-ordered arm
        // this only ENQUEUES; the sweep's own verdict kernel publishes this
        // outer's DeviceOuterProbe from device memory, so nothing between here
        // and the convergence kernel needs the host.
        if (!m.hooks.enqueue_cmfd_sweep(m.hooks.ctx, m.stream, slot, i))
            return hookFailed("the CMFD sweep hook");

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
            if ((rc = cudaMemcpyAsync(bound_.host_jnet, bound_.device_jnet, jnet_bytes,
                                      cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                return launchFailed("download jnet for the nodal drive", rc);
            bump(counters().jnet_bridge_bytes, jnet_bytes);
        }

        // THE ONE SYNCHRONISE OF THE OUTER, and it belongs to the nodal drive.
        if ((rc = cudaStreamSynchronize(m.stream)) != cudaSuccess)
            return launchFailed("synchronize before the nodal drive", rc);

        // The sweep's observation, on the sync that just happened.  It adopts
        // the flux into Geometry::Phif and the eigenvalue into the host local
        // the nodal hook is about to divide by -- and, on the exceptional
        // launches the device could not finish (sweep state 0 or 2), it runs the
        // remaining blocking launches and republishes the probe the verdict
        // kernel had to guess at.
        if (stream_sweep && !m.hooks.finish_cmfd_sweep(m.hooks.ctx, m.stream, slot, i))
            return hookFailed("the CMFD sweep observation hook");

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
                (rc = cudaMemcpyAsync(bound_.host_jnet, bound_.device_jnet, jnet_bytes,
                                      cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                return launchFailed("re-download jnet after the host finished the sweep", rc);
            // AND THE CANONICAL ARM NEEDS IT TOO.  There the nodal drive reads
            // the DEVICE jnet, on the backend's own stream, ordered against this
            // one by nothing but this synchronise.
            if ((rc = cudaStreamSynchronize(m.stream)) != cudaSuccess)
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
            if (after.device_owns_flux) {
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
        if (canonical_now && m.hooks.canonical_nodal_mode != nullptr) {
            m.canonical_nodal_live = true;
            m.hooks.canonical_nodal_mode(m.hooks.ctx, 1);
            bump(counters().canonical_nodal_outers);
        } else {
            // Not eligible: this outer's drive is the CPU body (or a host-owned
            // device drive), it has its bridge, and it is about to write the host
            // arrays itself.  Hand ownership back BEFORE it runs, or its upload
            // is elided against the device copy it is trying to replace.
            releaseCanonicalNodal(true);
        }
        if (!m.hooks.enqueue_nodal_drive(m.hooks.ctx, m.stream, slot, i))
            return hookFailed("the nodal drive hook");
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
        if (m.hooks.apply_cusping != nullptr &&
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
                if ((rc = cudaMemcpyAsync(bound_.device_dtil, bound_.host_dtil, dtil_bytes,
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
        }

        if (bridge_jnet) {
            if ((rc = cudaMemcpyAsync(bound_.device_jnet, bound_.host_jnet, jnet_bytes,
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
            (rc = cudaMemcpyAsync(m.h_seg, m.d_segments + slot, sizeof(*m.h_seg),
                                  cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
            return launchFailed("download segment exit", rc);
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
        if ((rc = cudaMemcpyAsync(bound_.host_psi, bound_.device_psi, psi_bytes,
                                  cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
            return launchFailed("mirror psi to the host at the segment exit", rc);
        if ((rc = cudaMemcpyAsync(bound_.host_dhat, bound_.device_dhat, dhat_bytes,
                                  cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
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
            if ((rc = cudaMemcpyAsync(bound_.host_jnet, bound_.device_jnet, surf_bytes,
                                      cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                return launchFailed("mirror jnet to the host at the segment exit", rc);
            bump(counters().jnet_mirror_bytes, surf_bytes);
        }
        if (bound_.host_phis != nullptr && bound_.device_phis != nullptr) {
            if ((rc = cudaMemcpyAsync(bound_.host_phis, bound_.device_phis, surf_bytes,
                                      cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
                return launchFailed("mirror phis to the host at the segment exit", rc);
            bump(counters().phis_mirror_bytes, surf_bytes);
        }
    }

    // --- the single observation ---------------------------------------------
    DeviceOuterSegmentState seg_out{};
    DeviceSlotState         state_out{};
    unsigned long long      halted_out = 0;
    CmfdOuterDecision       decision_out{};
    if ((rc = cudaMemcpyAsync(&seg_out, m.d_segments + slot, sizeof(seg_out),
                              cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download segment state", rc);
    if ((rc = cudaMemcpyAsync(&state_out, m.arena.states + slot, sizeof(state_out),
                              cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download slot state", rc);
    if ((rc = cudaMemcpyAsync(&halted_out, m.d_halted, sizeof(halted_out),
                              cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download halted count", rc);
    // The decision, for SolveLoop's ladder.  It rides the SAME observation as
    // the other three -- one more 24-byte copy on a stream that is about to be
    // synchronised anyway -- rather than becoming a second rendezvous.
    if ((rc = cudaMemcpyAsync(&decision_out, m.d_decisions + slot, sizeof(decision_out),
                              cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download outer decision", rc);
    if ((rc = cudaStreamSynchronize(m.stream)) != cudaSuccess)
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
    if (standUpTables().stood) return rasberyOuterSegment().bound();

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
    ArenaDims dims = arenaDims(deck.nxyz, deck.nsurf, deck.nxy, deck.n_fuel, 1);
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
    if ((rc = cudaMemcpy(t.slot_views, host_views.data(), sizeof(DeviceSlotView) * n,
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
    k_outer_seed_slot<<<1, 1>>>(arena, 0, seed_generation);
    if ((rc = cudaGetLastError()) != cudaSuccess) return fail("seed slot state");
    if ((rc = cudaDeviceSynchronize()) != cudaSuccess) return fail("stand-up synchronize");

    // --- 6. hand it to the runner --------------------------------------------
    if (!rasberyOuterSegment().initialize(arena, n)) {
        std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] runner initialise failed: %s\n",
                     rasberyOuterSegment().status().c_str());
        rasberyTearDownOuterSegment();
        return false;
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
    binding.device_jnet   = outerArena().slotView(0).jnet;
    // Rev.7.1 Task 18-lite: the phis half of the canonical nodal set.
    //
    // THE ARENA HAS A phis REGION AND NOTHING ON THE DEVICE READS IT.  Every
    // other slot region is written by a CMFD body or read by one; SlotRegion::Phis
    // is the one Geometry array the device outer never touches, because phis is a
    // NODAL output and the CMFD side has no use for it.  That is what makes it
    // the right buffer to hand the nodal drive: the sharing costs no coupling,
    // and the region is already sized as Geometry sizes it (LR*ng*NDIRMAX*nxyz,
    // GpuPhysicsArenaLayout.h:485).
    binding.device_phis   = outerArena().slotView(0).phis;
    binding.nxyz          = dims.nxyz;
    binding.ng            = dims.ng;
    rasberyOuterSegment().bind(binding);

    t.stood = true;
    return rasberyOuterSegment().bound();
}

void rasberyTearDownOuterSegment() {
    StandUpTables& t = standUpTables();
    rasberyOuterSegment().release();
    if (t.slot_views != nullptr) cudaFree(t.slot_views);
    if (t.cmfd_views != nullptr) cudaFree(t.cmfd_views);
    if (t.dhat_counters != nullptr) cudaFree(t.dhat_counters);
    t.slot_views    = nullptr;
    t.cmfd_views    = nullptr;
    t.dhat_counters = nullptr;
    t.stood         = false;
    outerArena().release();
}

} // namespace rasbery::gpu
