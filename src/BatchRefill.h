#pragma once

// BatchRefill -- the ledger and the tenancy audit behind Rev.7.1 Task 20
// ("immediate slot refill", plan Sec 8.2).
//
// WHAT REFILL ALREADY IS.  main.cpp's batch branch is an OpenMP dynamic queue
// over the job list with `host_threads` workers, and every Driver acquires its
// arena slot in its constructor and releases it in its destructor.  So a job
// list LONGER than the arena width already refills: the worker that finishes
// deck 3 destroys its Driver (slot released, arena woken, inUseCount() drops so
// the rendezvous stops waiting for it), takes the next job off the queue, and
// builds a new Driver that acquires a slot again.  The batch does not drain
// first.  That is measured, not assumed -- 8 jobs on width 4 are h5diff -c 0
// against the same decks run one at a time.
//
// WHAT WAS MISSING, AND IS HERE.  Three things, all of them the difference
// between "it happens to work" and "it is a contract":
//
//   1. A JOB QUEUE THAT IS NOT ARGV.  Task 20's acceptance case is 1,280 jobs.
//      1,280 --rasi paths plus 1,280 --raso paths do not fit an exec argument
//      list on any host worth measuring on.  `--jobs <manifest>` reads the same
//      pairs from a file and appends them to the same two vectors, so every
//      downstream rule (the distinct --raso namespace check, the counts match
//      check, the batch predicate) is unchanged and untouched.
//
//   2. A RECEIPT.  Without one, "the tail went away" is a claim about a wall
//      clock and nothing else.  [RASBERY][REFILL] states how many admissions
//      there were, how much of the run each lane was actually holding a deck,
//      and how long the lanes sat empty at the end waiting for the slowest one
//      -- which is the number the whole task exists to move.
//
//   3. A TENANCY AUDIT.  The plan's Sec 3.2/8.2 rule is that a refill is a FULL
//      reset and that a slot may never be queued twice or released twice.  Both
//      arenas already reset the whole slot struct on acquire; what did not
//      exist was anything that would NOTICE if a future change put per-slot
//      state somewhere the reset does not reach.  `duplicates` and
//      `stale_tenants` are those witnesses, and the gate is that both are 0.
//
// LAYERING.  Header-only and CUDA-free: the arenas live in .cu files and main()
// does not, so the counters have to be reachable from both, and the no-CUDA
// stub build has to compile this too.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <ostream>
#include <vector>

namespace rasbery::refill {

// ---------------------------------------------------------------------------
// Tenancy counters -- the Sec 3.2 ownership rule, counted
// ---------------------------------------------------------------------------

/// Process-wide witnesses that slot ownership stayed sane across refills.
///
/// EVERY ONE OF THESE IS A BUG COUNTER, not a statistic.  A run whose receipt
/// shows a nonzero value is not a valid measurement, because each of them names
/// a way one deck's numbers can end up computed from another deck's state:
///
///   queue_duplicates -- the same physical slot appeared twice in one
///       rendezvous batch.  The launcher would stage it twice and the second
///       stage would overwrite the first participant's operator.  Only possible
///       if a slot were shared by two Drivers, which is what `taken` prevents;
///       counting it is how we learn if that ever stops being true.
///
///   stale_tenants -- a slot was admitted while per-slot state that lives
///       OUTSIDE the slot struct (and is therefore not covered by the
///       `Slot{}` reset) was still carrying the previous tenant's value.  The
///       admission path clears it, so the physics stays right; the counter is
///       what says the reset path had a hole.
///
///   double_releases -- releaseSlot on a slot nobody held.  Harmless on its own,
///       but it means a Driver's lifetime is not what the arena thinks it is,
///       and the next acquire could hand the same slot to two tenants.
struct TenancyCounters {
    std::atomic<unsigned long long> admissions{0};
    std::atomic<unsigned long long> queue_duplicates{0};
    std::atomic<unsigned long long> stale_tenants{0};
    std::atomic<unsigned long long> double_releases{0};
};

inline TenancyCounters& tenancy() {
    static TenancyCounters counters;
    return counters;
}

// ---------------------------------------------------------------------------
// The refill ledger
// ---------------------------------------------------------------------------

/// One admission: which host lane ran it, and between which two instants.
struct Tenancy {
    int    lane  = -1;
    double start = 0.0; ///< seconds since the ledger opened
    double end   = 0.0;
};

/// Records the tenancies of one batch run so the receipt can be arithmetic
/// rather than narration.
///
/// LANE, NOT SLOT.  The ledger keys on the OpenMP worker, not on the arena slot
/// index, and that is deliberate.  The worker is what holds a tenancy for a
/// whole deck; the arena slot index is whatever `acquireSlot` had free at the
/// moment the Driver's BICGSolver was constructed, and it is re-scanned from 0
/// on every acquire, so it is not stable across a refill and is not the thing
/// whose idleness costs throughput.  With `host_threads` lanes over `M` slots
/// (main.cpp caps lanes at M) the two counts agree, which is what makes
/// `slot_busy_fraction` an honest name for a per-lane number.
class Ledger {
public:
    void begin(int jobs, int slots, int lanes) {
        std::lock_guard<std::mutex> lock(_mutex);
        _open  = true;
        _jobs  = jobs;
        _slots = slots;
        _lanes = lanes;
        _tenancies.assign(static_cast<std::size_t>(jobs > 0 ? jobs : 0), Tenancy{});
        _t0 = std::chrono::steady_clock::now();
        _t1 = _t0;
    }

    /// Called on the worker thread, immediately before the Driver is built.
    void jobStarted(int job, int lane) {
        if (!_open || job < 0 || job >= static_cast<int>(_tenancies.size())) return;
        const double now = elapsed();
        std::lock_guard<std::mutex> lock(_mutex);
        _tenancies[static_cast<std::size_t>(job)].lane  = lane;
        _tenancies[static_cast<std::size_t>(job)].start = now;
    }

    /// Called on the worker thread, immediately after the Driver is destroyed --
    /// which is the instant the slot is free for the next tenant.
    void jobFinished(int job) {
        if (!_open || job < 0 || job >= static_cast<int>(_tenancies.size())) return;
        const double now = elapsed();
        std::lock_guard<std::mutex> lock(_mutex);
        _tenancies[static_cast<std::size_t>(job)].end = now;
    }

    void end() {
        std::lock_guard<std::mutex> lock(_mutex);
        _t1 = std::chrono::steady_clock::now();
    }

    [[nodiscard]] bool open() const { return _open; }

    /// The Task 20 receipt.  One line, machine-readable, always the same keys.
    void report(std::ostream& out) const {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_open) return;

        const double wall = std::chrono::duration<double>(_t1 - _t0).count();

        // Per-lane: how much of the run it held a deck, and when it last let go.
        std::vector<double> lane_busy(static_cast<std::size_t>(std::max(_lanes, 1)), 0.0);
        std::vector<double> lane_last(static_cast<std::size_t>(std::max(_lanes, 1)), 0.0);
        std::vector<int>    lane_jobs(static_cast<std::size_t>(std::max(_lanes, 1)), 0);
        // Refill latency: the gap between one tenancy ending on a lane and the
        // next one starting there.  This is the cost of the refill itself --
        // Driver teardown, the next deck's import, the new Driver's acquire --
        // and it is what the receipt has to show is small, because it is paid
        // once per admission and is pure serial time on that lane.
        std::vector<double> gaps;

        for (const Tenancy& t : _tenancies) {
            if (t.lane < 0 || t.lane >= static_cast<int>(lane_busy.size())) continue;
            const auto l = static_cast<std::size_t>(t.lane);
            const double end = t.end > 0.0 ? t.end : wall;
            lane_busy[l] += std::max(0.0, end - t.start);
            ++lane_jobs[l];
            if (lane_last[l] > 0.0 && t.start > lane_last[l])
                gaps.push_back(t.start - lane_last[l]);
            lane_last[l] = std::max(lane_last[l], end);
        }

        int    lanes_used = 0;
        double busy_total = 0.0;
        double tail_idle  = 0.0;
        for (std::size_t l = 0; l < lane_busy.size(); ++l) {
            if (lane_jobs[l] == 0) continue;
            ++lanes_used;
            busy_total += lane_busy[l];
            tail_idle  += std::max(0.0, wall - lane_last[l]);
        }
        // A lane that never took a job at all is not "tail" -- it never
        // started.  It is still idle capacity, so it is reported separately
        // rather than folded in and made to look like a drain cost.
        const int lanes_idle = std::max(0, _lanes - lanes_used);

        // refills = admissions that reused a lane a previous deck had already
        // finished on.  This IS the thing Task 20 is named after: with 8 jobs
        // on 4 lanes it is 4, and with jobs <= lanes it is 0.
        const int refills = std::max(0, static_cast<int>(_tenancies.size()) - lanes_used);

        const double denom = static_cast<double>(std::max(lanes_used, 1)) * (wall > 0.0 ? wall : 1.0);
        const double busy_fraction = busy_total / denom;

        std::sort(gaps.begin(), gaps.end());
        const double gap_p50 = gaps.empty() ? 0.0 : gaps[gaps.size() / 2];
        const double gap_max = gaps.empty() ? 0.0 : gaps.back();

        const auto& c = tenancy();
        out << "[RASBERY][REFILL] {\"jobs\":" << _tenancies.size()
            << ",\"slots\":" << _slots
            << ",\"lanes\":" << _lanes
            << ",\"lanes_used\":" << lanes_used
            << ",\"lanes_never_admitted\":" << lanes_idle
            << ",\"refills\":" << refills
            << ",\"wall_s\":" << wall
            << ",\"tail_idle_s\":" << tail_idle
            << ",\"slot_busy_fraction\":" << busy_fraction
            << ",\"refill_latency_p50_ms\":" << (gap_p50 * 1.0e3)
            << ",\"refill_latency_max_ms\":" << (gap_max * 1.0e3)
            << ",\"admissions\":" << c.admissions.load()
            << ",\"duplicates\":" << c.queue_duplicates.load()
            << ",\"stale_tenants\":" << c.stale_tenants.load()
            << ",\"double_releases\":" << c.double_releases.load()
            << "}" << std::endl;
    }

private:
    [[nodiscard]] double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
    }

    mutable std::mutex                    _mutex;
    bool                                  _open  = false;
    int                                   _jobs  = 0;
    int                                   _slots = 0;
    int                                   _lanes = 0;
    std::vector<Tenancy>                  _tenancies;
    std::chrono::steady_clock::time_point _t0{};
    std::chrono::steady_clock::time_point _t1{};
};

inline Ledger& ledger() {
    static Ledger instance;
    return instance;
}

} // namespace rasbery::refill
