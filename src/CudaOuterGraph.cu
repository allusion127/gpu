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

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <string>

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

    cudaStream_t stream = nullptr;

    // Device scratch, all sized [slot_count] except the one counter.
    DeviceOuterProbe*        d_probes    = nullptr;
    DeviceOuterSegmentState* d_segments  = nullptr;
    std::uint32_t*           d_halt      = nullptr;
    cmfd::CmfdOuterInputs*   d_inputs    = nullptr;
    CmfdOuterDecision*       d_decisions = nullptr;
    unsigned long long*      d_halted    = nullptr;

    OuterSegmentHooks   hooks{};
    OuterSegmentBinding binding{};
    bool                is_bound = false;
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
    if ((rc = cudaStreamCreateWithFlags(&_impl->stream, cudaStreamNonBlocking)) != cudaSuccess)
        return fail("cudaStreamCreateWithFlags", rc);
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
    if (_impl->stream != nullptr) cudaStreamDestroy(_impl->stream);
    _impl->d_probes    = nullptr;
    _impl->d_segments  = nullptr;
    _impl->d_halt      = nullptr;
    _impl->d_inputs    = nullptr;
    _impl->d_decisions = nullptr;
    _impl->d_halted    = nullptr;
    _impl->stream      = nullptr;
    _impl->ready       = false;
}

bool CudaOuterSegment::available() const { return _impl != nullptr && _impl->ready; }

const std::string& CudaOuterSegment::status() const { return _impl->status; }

void CudaOuterSegment::setHooks(const OuterSegmentHooks& hooks) { _impl->hooks = hooks; }

OuterSegmentHooks CudaOuterSegment::hooks() const { return _impl->hooks; }

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
    e.have_sweep_hook  = _impl->hooks.enqueue_cmfd_sweep != nullptr ? 1 : 0;
    e.have_nodal_hook  = _impl->hooks.enqueue_nodal_drive != nullptr ? 1 : 0;
    return outerSegmentRefusal(e);
}

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
    const OuterSegmentBinding& bound_ = m.binding;
    const unsigned int         budget = outerSegmentBudget();
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

    auto launchFailed = [&](const char* what, cudaError_t rc) {
        std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] %s: %s -- falling back to the host "
                             "outer for this iteration\n",
                     what, cudaGetErrorString(rc));
        bump(counters().refusals[static_cast<int>(OuterSegmentRefusal::LaunchFailed)]);
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
        return false;
    };

    cudaError_t rc;
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

    // --- the body, budget times, on ONE stream -------------------------------
    for (unsigned int i = 0; i < budget; ++i) {
        // (1) updpsi -- Driver.h:1547
        if ((rc = enqueueUpdPsi(m.arena, queue, bound_.geom, bound_.table, bound_.forms,
                                m.stream, m.d_halt)) != cudaSuccess)
            return launchFailed("enqueue updpsi", rc);

        // (2,3) setls + drive -- Driver.h:1551-1555.  The hook also publishes
        // this outer's DeviceOuterProbe; nothing after it can see eigv,
        // residual or the negative/Rayleigh signals otherwise.
        if (!m.hooks.enqueue_cmfd_sweep(m.hooks.ctx, m.stream, slot, i))
            return hookFailed("the CMFD sweep hook");

        // (4) the convergence INPUTS -- Driver.h:1562.  See the header note on
        // why the decision itself is published at the end of the body.
        if ((rc = enqueueOuterRefreshInputs(queue, m.d_probes, m.d_inputs, m.d_halt,
                                            m.stream)) != cudaSuccess)
            return launchFailed("enqueue input refresh", rc);

        // (5) updjnet -- Driver.h:1569
        if ((rc = enqueueUpdJnet(m.arena, queue, bound_.geom, bound_.table, bound_.forms,
                                 m.stream, m.d_halt)) != cudaSuccess)
            return launchFailed("enqueue updjnet", rc);

        // (6) nodal reset + drive -- Driver.h:1573-1575
        if (!m.hooks.enqueue_nodal_drive(m.hooks.ctx, m.stream, slot, i))
            return hookFailed("the nodal drive hook");

        // (7) cusping -- Driver.h:1579-1580.  Stage A: eligibility guarantees
        // ApplyRodCusping would return false for this deck, so the host's own
        // branch here does nothing and the device skips nothing.  A deck for
        // which it WOULD fire never gets here (OuterSegmentRefusal::
        // FractionalRods), and Task 11 is what changes that.

        // (8) upddhat -- Driver.h:1584
        if ((rc = enqueueUpdDhat(m.arena, queue, bound_.geom, bound_.table, bound_.forms,
                                 bound_.dhat_clamp, bound_.dhat_counters, m.stream,
                                 m.d_halt)) != cudaSuccess)
            return launchFailed("enqueue upddhat", rc);

        // the decision -- Driver.h:1601-1705, 1834-1860
        if ((rc = enqueueOuterConvergence(m.arena, queue, m.d_inputs, m.d_decisions, m.stream,
                                          m.d_halt)) != cudaSuccess)
            return launchFailed("enqueue outer convergence", rc);

        // the transition -- this is what latches the halt
        if ((rc = enqueueOuterTransition(m.arena, queue, m.d_decisions, m.d_probes,
                                         m.d_segments, m.d_halt, m.d_halted, m.stream)) !=
            cudaSuccess)
            return launchFailed("enqueue outer transition", rc);
    }

    // --- the single observation ---------------------------------------------
    DeviceOuterSegmentState seg_out{};
    DeviceSlotState         state_out{};
    unsigned long long      halted_out = 0;
    if ((rc = cudaMemcpyAsync(&seg_out, m.d_segments + slot, sizeof(seg_out),
                              cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download segment state", rc);
    if ((rc = cudaMemcpyAsync(&state_out, m.arena.states + slot, sizeof(state_out),
                              cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download slot state", rc);
    if ((rc = cudaMemcpyAsync(&halted_out, m.d_halted, sizeof(halted_out),
                              cudaMemcpyDeviceToHost, m.stream)) != cudaSuccess)
        return launchFailed("download halted count", rc);
    if ((rc = cudaStreamSynchronize(m.stream)) != cudaSuccess)
        return launchFailed("segment synchronize", rc);

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

} // namespace rasbery::gpu
