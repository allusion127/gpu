// Case-phase classify/compact gate -- Rev.7.1 Task 3.
//
// The classification is pure (gpuClassifySerial in GpuPhaseScheduler.h), so
// every property that matters can be checked exhaustively with no GPU.  This
// harness includes GpuPhaseScheduler.h and nothing else; no CUDA header, no
// runtime.  `cmake -DRASBERY_ENABLE_TESTS=ON` builds it everywhere.
//
// What it proves:
//
//   1. COMPACTION, EXHAUSTIVELY.  All 256 subsets of slots 0-7, each with a
//      rotating phase assignment, plus representative 64-slot subsets: every
//      active slot appears exactly once, in ascending order, in its own phase's
//      queue; every other entry is kQueueEmptySlot; the bucket is the smallest
//      configured width >= count.  Exhaustive because the interesting bugs here
//      are off-by-one at a warp boundary, and 8 slots is small enough to try
//      every arrangement while 64 slots is where the boundary actually is.
//   2. SEC 5.2 OWNERSHIP.  A slot whose captured entry is still current is a
//      FATAL duplicate, not a silent re-insert; a slot with in_flight set is a
//      fatal requeue; the epoch capture on insert is what makes an entry go
//      stale by itself at the next transition.
//   3. RECYCLED-SLOT RESET (Sec 8.2).  The strong postcondition, the same one
//      the Task 20 audit kernel will check: reset from TWO DIFFERENT previous
//      tenants must produce byte-identical structs.  A field the reset forgets
//      differs between the two fills, so this catches an omission that
//      per-field spot checks would miss.
//   4. TRANSITION TABLE (Sec 6.21).  The map is data (kPhaseTransitions); this
//      compares it against an independently written expected-edge list.  The
//      ORDER against Driver.h is checked by
//      tools/test_gpu_phase_scheduler_contract.py, which can mine the file for
//      the anchors these edges carry.
//
// Run: rasbery_gpu_phase_compaction          (silent, exit 0)
//      rasbery_gpu_phase_compaction --print  (dumps the selection policy table)

#include "GpuPhaseScheduler.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace rasbery::gpu;

namespace {

int g_failures = 0;

void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

/// A slot in `phase`, active, not in flight, with no live queue entry.
DeviceSlotPhase makeReady(DevicePhase phase, std::uint32_t epoch) {
    DeviceSlotPhase p;
    std::memset(&p, 0, sizeof(p));
    deviceSlotPhaseReset(p, epoch);
    p.phase = static_cast<std::uint8_t>(phase);
    p.flags = kSlotFlagActive;
    return p;
}

/// The schedulable phases, in enum order, for the rotating assignment.
const DevicePhase kReadyPhases[] = {
    DevicePhase::Outer,      DevicePhase::Material, DevicePhase::Xenon,
    DevicePhase::Search,     DevicePhase::Ppr,      DevicePhase::ThermalHydraulics,
    DevicePhase::OutputPack, DevicePhase::Import,
};
constexpr int kReadyPhaseCount = static_cast<int>(sizeof(kReadyPhases) / sizeof(kReadyPhases[0]));

/// The whole compaction contract for one arrangement.
void checkArrangement(std::vector<DeviceSlotPhase>& phases, const std::vector<int>& active_slots) {
    const int slot_count = static_cast<int>(phases.size());

    DevicePhaseQueues q;
    gpuClassifySerial(phases.data(), slot_count, q);

    CHECK(q.fault_flags == kSchedFaultNone);
    CHECK(q.fault_slot == kSchedNoFaultSlot);
    CHECK(q.active_count == static_cast<int>(active_slots.size()));

    // Rebuild the expectation independently: for each phase, the ascending list
    // of active slots carrying it.
    for (int p = 0; p < kDevicePhaseCount; ++p) {
        std::vector<int> want;
        for (const int s : active_slots)
            if (static_cast<int>(phases[s].phase) == p) want.push_back(s);

        const DevicePhaseQueue& queue = q.queue[p];
        CHECK(queue.count == static_cast<int>(want.size()));
        CHECK(q.phase_count[p] == queue.count);
        CHECK(queue.bucket == gpuSelectBucket(queue.count));
        if (queue.count > 0) CHECK(queue.bucket >= queue.count);

        for (int i = 0; i < queue.count && i < static_cast<int>(want.size()); ++i) {
            CHECK(queue.slots[i] == want[i]);
            // Ascending, stated directly rather than inferred from the rebuild.
            if (i > 0) CHECK(queue.slots[i] > queue.slots[i - 1]);
        }
        // Everything past the count is padding, all the way to the array end --
        // not just to the bucket, so a stale id can never be read.
        for (int i = queue.count; i < kMaxSchedulerSlots; ++i)
            CHECK(queue.slots[i] == kQueueEmptySlot);

        // The graph-path padding guard agrees with the queue contents.
        for (int i = 0; i < kMaxSchedulerSlots; ++i) {
            const bool padding = gpuDispatchIsPadding(i, queue.count);
            CHECK(padding == (i >= queue.count));
            if (padding && i < kMaxSchedulerSlots) CHECK(queue.slots[i] == kQueueEmptySlot);
        }
    }

    // Every active slot appears EXACTLY once across all queues.
    int seen[kMaxSchedulerSlots] = {};
    int total                    = 0;
    for (int p = 0; p < kDevicePhaseCount; ++p)
        for (int i = 0; i < q.queue[p].count; ++i) {
            const int s = q.queue[p].slots[i];
            CHECK(s >= 0 && s < slot_count);
            if (s >= 0 && s < kMaxSchedulerSlots) ++seen[s];
            ++total;
        }
    CHECK(total == static_cast<int>(active_slots.size()));
    for (const int s : active_slots) CHECK(seen[s] == 1);

    // Selection: the largest ready queue, ties by kPhaseScanOrder position.
    int want_phase = -1, want_count = 0, want_rank = kPhaseScanOrderCount;
    for (int i = 0; i < kPhaseScanOrderCount; ++i) {
        const int p = static_cast<int>(kPhaseScanOrder[i]);
        if (q.phase_count[p] > want_count || (q.phase_count[p] == want_count && want_count > 0 &&
                                              i < want_rank)) {
            if (q.phase_count[p] > want_count) {
                want_count = q.phase_count[p];
                want_phase = p;
                want_rank  = i;
            }
        }
    }
    CHECK(q.selected_phase == want_phase);
    if (want_phase >= 0) {
        CHECK(q.selected_count == want_count);
        CHECK(q.selected_bucket == gpuSelectBucket(want_count));
    } else {
        CHECK(q.selected_count == 0 && q.selected_bucket == 0);
    }

    // Insertion captured the epoch, so the entry is now "already queued" and a
    // second classify pass must refuse it (Sec 5.2).
    for (const int s : active_slots) {
        CHECK(phases[s].queued_epoch == phases[s].state_epoch);
        CHECK(phases[s].queued_phase == phases[s].phase);
        CHECK(slotAlreadyQueued(phases[s]));
    }
}

void testExhaustiveEightSlots() {
    for (unsigned mask = 0; mask < 256u; ++mask) {
        std::vector<DeviceSlotPhase> phases(8);
        std::vector<int>             active;
        for (int s = 0; s < 8; ++s) {
            if (mask & (1u << s)) {
                phases[s] = makeReady(kReadyPhases[s % kReadyPhaseCount], 100u + s);
                active.push_back(s);
            } else {
                // Inactive tenants must be invisible to the queues whatever
                // phase word they carry.
                phases[s] = makeReady(DevicePhase::Outer, 7u);
                phases[s].flags = 0;
            }
        }
        checkArrangement(phases, active);
    }

    // Same 256 subsets with every active slot in ONE phase: the queue is then
    // the subset itself, which is the tightest possible ordering check.
    for (unsigned mask = 0; mask < 256u; ++mask) {
        std::vector<DeviceSlotPhase> phases(8);
        std::vector<int>             active;
        for (int s = 0; s < 8; ++s) {
            phases[s] = makeReady(DevicePhase::Outer, 200u + s);
            if (mask & (1u << s))
                active.push_back(s);
            else
                phases[s].flags = 0;
        }
        checkArrangement(phases, active);
        DevicePhaseQueues q;
        std::vector<DeviceSlotPhase> fresh(8);
        for (int s = 0; s < 8; ++s) {
            fresh[s] = makeReady(DevicePhase::Outer, 200u + s);
            if (!(mask & (1u << s))) fresh[s].flags = 0;
        }
        gpuClassifySerial(fresh.data(), 8, q);
        CHECK(q.queue[static_cast<int>(DevicePhase::Outer)].count ==
              static_cast<int>(active.size()));
    }
}

void testSixtyFourSlotSubsets() {
    struct Named {
        const char* name;
        bool (*member)(int);
    };
    const Named cases[] = {
        {"all", [](int) { return true; }},
        {"none", [](int) { return false; }},
        {"even", [](int s) { return s % 2 == 0; }},
        {"odd", [](int s) { return s % 2 == 1; }},
        {"first_half", [](int s) { return s < 32; }},
        {"second_half", [](int s) { return s >= 32; }},
        // The warp boundary, from both sides -- this is where a ballot-and-
        // prefix compaction goes wrong if the cross-warp offset is off by one.
        {"boundary", [](int s) { return s == 31 || s == 32; }},
        {"first_only", [](int s) { return s == 0; }},
        {"last_only", [](int s) { return s == 63; }},
        {"warp3_only", [](int s) { return s >= 32 && s < 64 && (s % 8) == 3; }},
        {"prime_ish", [](int s) { return (s * 7 + 3) % 11 < 4; }},
        // Exactly the bucket boundaries, so gpuSelectBucket is exercised at the
        // point where it steps.
        {"exactly_24", [](int s) { return s < 24; }},
        {"exactly_25", [](int s) { return s < 25; }},
        {"exactly_48", [](int s) { return s < 48; }},
        {"exactly_49", [](int s) { return s < 49; }},
    };

    for (const Named& c : cases) {
        // one phase
        {
            std::vector<DeviceSlotPhase> phases(64);
            std::vector<int>             active;
            for (int s = 0; s < 64; ++s) {
                phases[s] = makeReady(DevicePhase::Outer, 1000u + s);
                if (c.member(s))
                    active.push_back(s);
                else
                    phases[s].flags = 0;
            }
            checkArrangement(phases, active);
        }
        // rotating phases
        {
            std::vector<DeviceSlotPhase> phases(64);
            std::vector<int>             active;
            for (int s = 0; s < 64; ++s) {
                phases[s] = makeReady(kReadyPhases[s % kReadyPhaseCount], 2000u + s);
                if (c.member(s))
                    active.push_back(s);
                else
                    phases[s].flags = 0;
            }
            checkArrangement(phases, active);
        }
    }

    // Bucket steps, stated directly.
    CHECK(gpuSelectBucket(0) == 0);
    CHECK(gpuSelectBucket(1) == 1);
    CHECK(gpuSelectBucket(2) == 2);
    CHECK(gpuSelectBucket(3) == 4);
    CHECK(gpuSelectBucket(24) == 24);
    CHECK(gpuSelectBucket(25) == 32);
    CHECK(gpuSelectBucket(33) == 48);
    CHECK(gpuSelectBucket(49) == 64);
    CHECK(gpuSelectBucket(64) == 64);
    CHECK(gpuSelectBucket(65) == -1);
    CHECK(kDispatchBuckets[kDispatchBucketCount - 1] == kMaxSchedulerSlots);
}

void testOwnershipFaults() {
    // (a) Duplicate: the slot's captured entry is still current, so it is
    //     already queued and a second insertion would let two graph bodies
    //     drive one slot.  Fatal, not deduplicated.
    {
        std::vector<DeviceSlotPhase> phases(4);
        for (int s = 0; s < 4; ++s) phases[s] = makeReady(DevicePhase::Outer, 5u);
        phases[2].queued_phase = phases[2].phase;
        phases[2].queued_epoch = phases[2].state_epoch;
        CHECK(slotAlreadyQueued(phases[2]));

        DevicePhaseQueues q;
        gpuClassifySerial(phases.data(), 4, q);
        CHECK((q.fault_flags & kSchedFaultDuplicateQueue) != 0u);
        CHECK(q.fault_slot == 2u);
        CHECK(q.queue[static_cast<int>(DevicePhase::Outer)].count == 3); // the other three
        for (int i = 0; i < 3; ++i) CHECK(q.queue[static_cast<int>(DevicePhase::Outer)].slots[i] != 2);
    }

    // (b) in_flight: a slot being driven right now.  Reported in preference to
    //     the duplicate flag when both would apply.
    {
        std::vector<DeviceSlotPhase> phases(4);
        for (int s = 0; s < 4; ++s) phases[s] = makeReady(DevicePhase::Outer, 5u);
        phases[1].flags |= kSlotFlagInFlight;
        phases[1].queued_phase = phases[1].phase;
        phases[1].queued_epoch = phases[1].state_epoch;

        DevicePhaseQueues q;
        gpuClassifySerial(phases.data(), 4, q);
        CHECK((q.fault_flags & kSchedFaultInFlightRequeue) != 0u);
        CHECK((q.fault_flags & kSchedFaultDuplicateQueue) == 0u);
        CHECK(q.fault_slot == 1u);
    }

    // (c) The epoch rule invalidates a stale entry with no queue walk: bump
    //     state_epoch and the same slot becomes insertable again.
    {
        std::vector<DeviceSlotPhase> phases(1);
        phases[0]              = makeReady(DevicePhase::Outer, 9u);
        phases[0].queued_phase = phases[0].phase;
        phases[0].queued_epoch = phases[0].state_epoch;

        DevicePhaseQueues q;
        gpuClassifySerial(phases.data(), 1, q);
        CHECK((q.fault_flags & kSchedFaultDuplicateQueue) != 0u);

        ++phases[0].state_epoch; // a phase transition happened
        CHECK(!slotAlreadyQueued(phases[0]));
        gpuClassifySerial(phases.data(), 1, q);
        CHECK(q.fault_flags == kSchedFaultNone);
        CHECK(q.queue[static_cast<int>(DevicePhase::Outer)].count == 1);
    }

    // (d) Overflow and bad phase.
    {
        std::vector<DeviceSlotPhase> phases(kMaxSchedulerSlots);
        for (int s = 0; s < kMaxSchedulerSlots; ++s) phases[s] = makeReady(DevicePhase::Outer, 3u);
        DevicePhaseQueues q;
        gpuClassifySerial(phases.data(), kMaxSchedulerSlots + 1, q);
        CHECK((q.fault_flags & kSchedFaultSlotOverflow) != 0u);
    }
    {
        std::vector<DeviceSlotPhase> phases(2);
        phases[0]       = makeReady(DevicePhase::Outer, 3u);
        phases[1]       = makeReady(DevicePhase::Outer, 3u);
        phases[1].phase = static_cast<std::uint8_t>(kDevicePhaseCount + 5);
        DevicePhaseQueues q;
        gpuClassifySerial(phases.data(), 2, q);
        CHECK((q.fault_flags & kSchedFaultBadPhase) != 0u);
        CHECK(q.fault_slot == 1u);
    }

    // (e) Done/Failed/Empty are not schedulable; they are the refill kernel's
    //     input and are counted as free, not queued.
    {
        std::vector<DeviceSlotPhase> phases(3);
        phases[0] = makeReady(DevicePhase::Done, 1u);
        phases[1] = makeReady(DevicePhase::Failed, 1u);
        phases[2] = makeReady(DevicePhase::Empty, 1u);
        DevicePhaseQueues q;
        gpuClassifySerial(phases.data(), 3, q);
        CHECK(q.fault_flags == kSchedFaultNone);
        CHECK(q.free_count == 3);
        CHECK(q.active_count == 0);
        CHECK(q.selected_phase == -1);
        CHECK(!gpuPhaseIsSchedulable(DevicePhase::Done));
        CHECK(!gpuPhaseIsSchedulable(DevicePhase::Failed));
        CHECK(!gpuPhaseIsSchedulable(DevicePhase::Empty));
        CHECK(gpuPhaseIsSchedulable(DevicePhase::Outer));
    }
}

// ---------------------------------------------------------------------------
// Sec 8.2 recycled-slot reset.
//
// The STRONG postcondition, and the one the Task 20 audit kernel will use:
// reset from two DIFFERENT previous tenants must produce byte-identical
// structs.  A field the reset forgets keeps whichever fill it had, so the two
// copies differ -- which a per-field spot check cannot detect for a field
// nobody thought to name.
// ---------------------------------------------------------------------------

template <typename T, typename Reset>
void checkResetErasesTenant(const char* what, Reset reset) {
    T a, b;
    std::memset(&a, 0x5a, sizeof(T));
    std::memset(&b, 0xa5, sizeof(T));
    reset(a);
    reset(b);
    if (std::memcmp(&a, &b, sizeof(T)) != 0) {
        std::fprintf(stderr, "FAIL: %s reset leaves previous-tenant bytes\n", what);
        ++g_failures;
        return;
    }
    // And it is idempotent: resetting an already-reset struct changes nothing.
    T c = a;
    reset(c);
    if (std::memcmp(&a, &c, sizeof(T)) != 0) {
        std::fprintf(stderr, "FAIL: %s reset is not idempotent\n", what);
        ++g_failures;
    }
}

void testRecycledSlotReset() {
    checkResetErasesTenant<DeviceSlotPhase>(
        "DeviceSlotPhase", [](DeviceSlotPhase& p) { deviceSlotPhaseReset(p, 42u); });
    checkResetErasesTenant<DeviceSlotState>("DeviceSlotState", deviceSlotStateReset);
    checkResetErasesTenant<DeviceSearchState>("DeviceSearchState", deviceSearchStateReset);
    checkResetErasesTenant<DeviceScheduleParams>("DeviceScheduleParams", deviceScheduleParamsReset);

    // The refilled slot is invisible to classify until it is stamped Active:
    // a slot that reset left Empty must not be queued, and must not look queued.
    DeviceSlotPhase p;
    std::memset(&p, 0x5a, sizeof(p));
    deviceSlotPhaseReset(p, 42u);
    CHECK(p.phase == static_cast<std::uint8_t>(DevicePhase::Empty));
    CHECK(!slotActive(p));
    CHECK(!slotInFlight(p));
    CHECK(!slotAlreadyQueued(p));
    CHECK(p.state_epoch == 42u);

    std::vector<DeviceSlotPhase> one(1, p);
    DevicePhaseQueues            q;
    gpuClassifySerial(one.data(), 1, q);
    CHECK(q.fault_flags == kSchedFaultNone);
    CHECK(q.active_count == 0);
    CHECK(q.free_count == 1);

    // Now stamp it the way k_refill_free_slots does and it becomes schedulable.
    one[0].input_id = 7u;
    one[0].job_id   = 3u;
    one[0].flags    = kSlotFlagActive | kSlotFlagInputReady;
    one[0].phase    = static_cast<std::uint8_t>(DevicePhase::Import);
    gpuClassifySerial(one.data(), 1, q);
    CHECK(q.active_count == 1);
    CHECK(q.free_count == 0);
    CHECK(q.queue[static_cast<int>(DevicePhase::Import)].count == 1);
    CHECK(q.queue[static_cast<int>(DevicePhase::Import)].slots[0] == 0);
}

// ---------------------------------------------------------------------------
// Sec 6.21 transition table
// ---------------------------------------------------------------------------

struct ExpectedEdge {
    DevicePhase    from;
    DevicePhase    to;
    PhaseEdgeGuard guard;
};

/// Written independently of kPhaseTransitions, from Sec 6.21 and from what
/// Driver.h's statepoint loop actually does.  If the two disagree, one of them
/// is wrong and the test says which edge.
const ExpectedEdge kExpectedEdges[] = {
    {DevicePhase::Empty, DevicePhase::Import, PhaseEdgeGuard::Always},
    {DevicePhase::Import, DevicePhase::Material, PhaseEdgeGuard::Always},
    {DevicePhase::Material, DevicePhase::Outer, PhaseEdgeGuard::Always},

    {DevicePhase::Outer, DevicePhase::Outer, PhaseEdgeGuard::FluxNotConverged},
    {DevicePhase::Outer, DevicePhase::Xenon, PhaseEdgeGuard::XePending},
    {DevicePhase::Xenon, DevicePhase::Outer, PhaseEdgeGuard::Always},
    {DevicePhase::Outer, DevicePhase::ThermalHydraulics, PhaseEdgeGuard::ThPending},
    {DevicePhase::ThermalHydraulics, DevicePhase::Material, PhaseEdgeGuard::Always},
    {DevicePhase::Outer, DevicePhase::Search, PhaseEdgeGuard::SearchPending},
    {DevicePhase::Search, DevicePhase::Material, PhaseEdgeGuard::Always},

    {DevicePhase::Outer, DevicePhase::NormalizeFluxSign, PhaseEdgeGuard::FluxConverged},
    {DevicePhase::NormalizeFluxSign, DevicePhase::DepletionPredictor,
     PhaseEdgeGuard::ScheduleIsDepletion},
    {DevicePhase::DepletionPredictor, DevicePhase::Outer, PhaseEdgeGuard::Always},
    {DevicePhase::NormalizeFluxSign, DevicePhase::DepletionCorrector,
     PhaseEdgeGuard::ScheduleIsDepletion},
    {DevicePhase::DepletionCorrector, DevicePhase::Outer, PhaseEdgeGuard::Always},
    {DevicePhase::NormalizeFluxSign, DevicePhase::Derivative, PhaseEdgeGuard::ScheduleIsDerivative},
    {DevicePhase::NormalizeFluxSign, DevicePhase::RodOp, PhaseEdgeGuard::ScheduleIsRod},
    {DevicePhase::RodOp, DevicePhase::Material, PhaseEdgeGuard::Always},
    {DevicePhase::Derivative, DevicePhase::RodOp, PhaseEdgeGuard::DerivativeCollapsed},
    {DevicePhase::Derivative, DevicePhase::Ppr, PhaseEdgeGuard::DerivativeOk},
    {DevicePhase::NormalizeFluxSign, DevicePhase::Ppr, PhaseEdgeGuard::Always},
    {DevicePhase::Ppr, DevicePhase::Ppr, PhaseEdgeGuard::IterationRemaining},
    {DevicePhase::Ppr, DevicePhase::ResultAggregate, PhaseEdgeGuard::IterationComplete},
    {DevicePhase::ResultAggregate, DevicePhase::OutputPack, PhaseEdgeGuard::Always},
    {DevicePhase::OutputPack, DevicePhase::Done, PhaseEdgeGuard::Always},

    {DevicePhase::Outer, DevicePhase::Failed, PhaseEdgeGuard::Fatal},
    {DevicePhase::DepletionPredictor, DevicePhase::Failed, PhaseEdgeGuard::Fatal},
    {DevicePhase::DepletionCorrector, DevicePhase::Failed, PhaseEdgeGuard::Fatal},

    {DevicePhase::Done, DevicePhase::Empty, PhaseEdgeGuard::Always},
    {DevicePhase::Failed, DevicePhase::Empty, PhaseEdgeGuard::Always},
};
constexpr int kExpectedEdgeCount =
    static_cast<int>(sizeof(kExpectedEdges) / sizeof(kExpectedEdges[0]));

void testTransitionTable() {
    CHECK(kPhaseTransitionCount == kExpectedEdgeCount);

    for (int i = 0; i < kExpectedEdgeCount; ++i) {
        bool found = false;
        for (int j = 0; j < kPhaseTransitionCount; ++j) {
            if (kPhaseTransitions[j].from == kExpectedEdges[i].from &&
                kPhaseTransitions[j].to == kExpectedEdges[i].to &&
                kPhaseTransitions[j].guard == kExpectedEdges[i].guard) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::fprintf(stderr, "FAIL: expected edge %d -> %d (guard %u) is not in the map\n",
                         static_cast<int>(kExpectedEdges[i].from),
                         static_cast<int>(kExpectedEdges[i].to),
                         static_cast<unsigned>(kExpectedEdges[i].guard));
            ++g_failures;
        }
    }
    // And nothing EXTRA: an edge in the map that Sec 6.21 does not sanction is
    // as wrong as a missing one -- it is a path the device would take and the
    // host would not.
    for (int j = 0; j < kPhaseTransitionCount; ++j) {
        bool found = false;
        for (int i = 0; i < kExpectedEdgeCount; ++i) {
            if (kPhaseTransitions[j].from == kExpectedEdges[i].from &&
                kPhaseTransitions[j].to == kExpectedEdges[i].to &&
                kPhaseTransitions[j].guard == kExpectedEdges[i].guard) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::fprintf(stderr, "FAIL: map edge %d -> %d (guard %u) has no Sec 6.21 sanction\n",
                         static_cast<int>(kPhaseTransitions[j].from),
                         static_cast<int>(kPhaseTransitions[j].to),
                         static_cast<unsigned>(kPhaseTransitions[j].guard));
            ++g_failures;
        }
    }

    // Structural properties the map must have whatever its contents.
    for (int j = 0; j < kPhaseTransitionCount; ++j) {
        CHECK(static_cast<int>(kPhaseTransitions[j].from) < kDevicePhaseCount);
        CHECK(static_cast<int>(kPhaseTransitions[j].to) < kDevicePhaseCount);
        CHECK(kPhaseTransitions[j].host_anchor != nullptr);
    }
    // Every terminal reachable, and Done/Failed both recycle (Sec 8.2).
    bool done_recycles = false, failed_recycles = false, reaches_done = false;
    for (int j = 0; j < kPhaseTransitionCount; ++j) {
        if (kPhaseTransitions[j].from == DevicePhase::Done &&
            kPhaseTransitions[j].to == DevicePhase::Empty)
            done_recycles = true;
        if (kPhaseTransitions[j].from == DevicePhase::Failed &&
            kPhaseTransitions[j].to == DevicePhase::Empty)
            failed_recycles = true;
        if (kPhaseTransitions[j].to == DevicePhase::Done) reaches_done = true;
    }
    CHECK(done_recycles);
    CHECK(failed_recycles);
    CHECK(reaches_done);

    // Sec 5.4: the phases the physics ORDER depends on are all present as
    // distinct edges out of Outer.  Xe, TH and Search each get their own.
    int out_of_outer = 0;
    for (int j = 0; j < kPhaseTransitionCount; ++j)
        if (kPhaseTransitions[j].from == DevicePhase::Outer) ++out_of_outer;
    CHECK(out_of_outer == 6); // requeue, Xe, TH, Search, NormalizeFluxSign, Failed
}

void testOuterQuantum() {
    // "One outer" is the seven SolveLoop steps plus the convergence test.
    CHECK(kOuterQuantumStepCount == 8);
    const char* want[] = {"updpsi",  "setls", "drive",   "conv_check",
                          "updjnet", "nodal", "cusping", "upddhat"};
    for (int i = 0; i < kOuterQuantumStepCount; ++i) {
        CHECK(std::strcmp(kOuterQuantumSteps[i].name, want[i]) == 0);
        CHECK(kOuterQuantumSteps[i].host_anchor != nullptr);
        CHECK(kOuterQuantumSteps[i].host_anchor[0] != '\0');
    }

    // Sec 5.1: the default CMFD quantum is one outer, and the escalated mode is
    // still reachable -- the flag is a rollback path, not a dead branch.
    CHECK(gpuPhaseQuantum(DevicePhase::Outer, CmfdQuantumMode::OneOuter) == PhaseQuantum::OneOuter);
    CHECK(gpuPhaseQuantum(DevicePhase::Outer, CmfdQuantumMode::RemainingBudget) ==
          PhaseQuantum::RemainingBudget);
    CHECK(gpuPhaseQuantum(DevicePhase::Xenon, CmfdQuantumMode::OneOuter) == PhaseQuantum::OneStep);
    CHECK(gpuPhaseQuantum(DevicePhase::ThermalHydraulics, CmfdQuantumMode::OneOuter) ==
          PhaseQuantum::OneStep);
    CHECK(gpuPhaseQuantum(DevicePhase::Search, CmfdQuantumMode::OneOuter) == PhaseQuantum::OneTrial);
    CHECK(gpuPhaseQuantum(DevicePhase::DepletionPredictor, CmfdQuantumMode::OneOuter) ==
          PhaseQuantum::OneStage);
    CHECK(gpuPhaseQuantum(DevicePhase::DepletionCorrector, CmfdQuantumMode::OneOuter) ==
          PhaseQuantum::OneStage);
    CHECK(gpuPhaseQuantum(DevicePhase::Ppr, CmfdQuantumMode::OneOuter) ==
          PhaseQuantum::OneCornerIteration);
    CHECK(gpuPhaseQuantum(DevicePhase::Done, CmfdQuantumMode::OneOuter) == PhaseQuantum::None);

    // The Sec 5.1(2) escalation arithmetic, with the W0 measurement.  At
    // 68,000 epochs x 5.13 us = 0.349 s against a 55.35 s v2 baseline wall the
    // default SURVIVES -- 0.63%, against the constraint-32 ceiling of 3%.
    const double epochs      = 68000.0;
    const double epoch_cost  = kW0SwitchEvalMicros * 1.0e-6;
    const double v2_wall     = 55.35;
    CHECK(!gpuCmfdQuantumShouldEscalate(epochs, epoch_cost, v2_wall));
    // ...and it does NOT survive a wall 5x shorter, which is the situation the
    // retained flag exists for.
    CHECK(gpuCmfdQuantumShouldEscalate(epochs, epoch_cost, v2_wall / 5.0));
    CHECK(kW0DispatchMicros > 0.0);
    CHECK(kQuantumCmfdEscalationNote != nullptr);
}

void testScanOrderPolicy() {
    // Every scan entry is schedulable, and every schedulable phase is in the
    // scan -- a phase missing from the order can never be selected, which is a
    // silent hang, not an error.
    for (int i = 0; i < kPhaseScanOrderCount; ++i) CHECK(gpuPhaseIsSchedulable(kPhaseScanOrder[i]));
    for (int p = 0; p < kDevicePhaseCount; ++p) {
        if (!gpuPhaseIsSchedulable(static_cast<DevicePhase>(p))) continue;
        bool present = false;
        for (int i = 0; i < kPhaseScanOrderCount; ++i)
            if (static_cast<int>(kPhaseScanOrder[i]) == p) present = true;
        if (!present) {
            std::fprintf(stderr, "FAIL: phase %d is schedulable but absent from the scan order\n", p);
            ++g_failures;
        }
    }
    // No duplicates.
    for (int i = 0; i < kPhaseScanOrderCount; ++i)
        for (int j = i + 1; j < kPhaseScanOrderCount; ++j)
            CHECK(kPhaseScanOrder[i] != kPhaseScanOrder[j]);

    // Outer is first and Import is last: a tie between admitting new work and
    // advancing work already in flight goes to the work in flight.
    CHECK(kPhaseScanOrder[0] == DevicePhase::Outer);
    CHECK(kPhaseScanOrder[kPhaseScanOrderCount - 1] == DevicePhase::Import);

    // Ties resolve by scan position, not by enum value.
    int counts[kDevicePhaseCount] = {};
    counts[static_cast<int>(DevicePhase::Import)] = 5;
    counts[static_cast<int>(DevicePhase::Outer)]  = 5;
    CHECK(gpuSelectPhase(counts) == static_cast<int>(DevicePhase::Outer));
    // A strictly larger queue wins regardless of position.
    counts[static_cast<int>(DevicePhase::Import)] = 6;
    CHECK(gpuSelectPhase(counts) == static_cast<int>(DevicePhase::Import));
    // Nothing ready -> idle, not phase 0.
    int empty_counts[kDevicePhaseCount] = {};
    CHECK(gpuSelectPhase(empty_counts) == -1);
}

} // namespace

int main(int argc, char** argv) {
    const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;

    testExhaustiveEightSlots();
    testSixtyFourSlotSubsets();
    testOwnershipFaults();
    testRecycledSlotReset();
    testTransitionTable();
    testOuterQuantum();
    testScanOrderPolicy();

    if (print) {
        std::printf("scan order (tie-break only, fairness deferred by the W0 verdict):\n");
        for (int i = 0; i < kPhaseScanOrderCount; ++i)
            std::printf("  %2d  phase %2d\n", i, static_cast<int>(kPhaseScanOrder[i]));
        std::printf("buckets:");
        for (int i = 0; i < kDispatchBucketCount; ++i) std::printf(" %d", kDispatchBuckets[i]);
        std::printf("\nqueue set bytes: %zu\n", sizeof(DevicePhaseQueues));
        std::printf("hot phase array at 64 slots: %zu bytes\n",
                    sizeof(DeviceSlotPhase) * kMaxSchedulerSlots);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "gpu phase compaction: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("gpu phase compaction: PASS\n");
    return 0;
}
