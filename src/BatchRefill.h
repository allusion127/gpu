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
#include <cstdint>
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

// ---------------------------------------------------------------------------
// WP18  The ROLLING ledger -- per-slot refill, counted
// ---------------------------------------------------------------------------

/// What the wave ledger above cannot say, because in wave mode it is not true.
///
/// THE DIFFERENCE, IN ONE SENTENCE.  `Ledger` measures a run whose job list was
/// known when the run began: `begin(jobs, ...)` sizes a vector by it.  A rolling
/// session has no such number -- cases arrive on the stream while the arena is
/// already full -- so the thing to measure is not "how did N jobs tile M lanes"
/// but "how often was a lane holding a deck at all, and how long did it sit
/// empty between two of them".
///
/// WHY IT IS A SECOND LINE AND NOT SIX MORE KEYS ON THE FIRST.  `[RASBERY]
/// [REFILL]` is parsed by tools/run_multi_gpu_batch.py (REFILL_RECEIPT) and by
/// the campaign tables built off it, and WP18's own gate is that the flag OFF
/// produces byte-identical output.  Adding keys to that line would change every
/// wave-mode run's stdout, which is exactly what the gate forbids.  So the
/// rolling receipt is `[RASBERY][REFILL][ROLLING]`, which the existing regex
/// cannot match (it requires `{` immediately after the tag), and the wave line
/// is emitted unchanged beside it.
///
/// EVERY COUNTER IS EITHER A LEVER OR A BUG WITNESS.
///
///   admits / immediate_admits   the lever.  An admit is IMMEDIATE when the
///       lane found a case already queued and never waited -- i.e. the refill
///       cost the lane nothing.  `immediate_admits / admits` is the fraction of
///       the run in which the prefetch queue was actually ahead of the arena;
///       below 1 the queue ran dry, and that is a HARNESS fact (the top-up did
///       not keep pace) rather than an arena one.
///
///   wave_barriers_avoided       what the mode is FOR.  In wave mode every
///       request batch ends in a barrier: the lanes drain, the slowest case
///       runs alone, and only then does the next chunk start.  In rolling mode
///       every request batch but the last is merged into the running session,
///       so this counts the drains that did not happen.
///
///   slot_idle_ms_total          the cost that is left.  Summed over lanes, the
///       wall between one case ending on a lane and the next beginning there.
///       In wave mode that number is dominated by the barrier; here what
///       remains is Driver teardown plus the next deck's import, which is the
///       honest floor of the mode.
///
///   width_* percentiles         the number the campaign compares on.  Sampled
///       at every admit AND every finish, so it is the occupancy of the arena
///       over EVENTS and not over a clock the sampler chose.
///
///   stale_tenant_refusals       a bug witness, and it must be 0.  A lane that
///       asked for a new case while the ledger still believed it held one means
///       a Driver outlived the finish stamp -- the tenancy rule of Sec 3.2
///       broken on the host side, where the arena's own `stale_tenants` audit
///       (CudaBICGBackend.cu batchSlotIsReset) cannot see it because the slot
///       really was reset; what was not reset is the LANE.
///
///   epoch_regressions           the other one.  Every admission takes a
///       monotonic epoch and every finish must retire exactly the epoch its
///       admission took.  A finish that retires an epoch nobody holds is a
///       completion attributed to the wrong tenancy, which is how a rolling
///       receipt would report a width it never had.
class RollingLedger {
public:
    void open(int lanes, int width, int capacity) {
        std::lock_guard<std::mutex> lock(_mutex);
        _open     = true;
        _lanes    = std::max(1, lanes);
        _width    = std::max(1, width);
        _capacity = capacity;
        _lane_busy.assign(static_cast<std::size_t>(_lanes), false);
        _lane_last_end.assign(static_cast<std::size_t>(_lanes), 0.0);
        _lane_jobs.assign(static_cast<std::size_t>(_lanes), 0);
        _widths.clear();
        _admit_wait_ms.clear();
        _live_epochs.clear();
        _admits            = 0;
        _immediate         = 0;
        _barriers_avoided  = 0;
        _stale_refusals    = 0;
        _epoch_regressions = 0;
        _epoch             = 0;
        _idle_ms_total     = 0.0;
        _t0                = std::chrono::steady_clock::now();
        _t1                = _t0;
    }

    [[nodiscard]] bool isOpen() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _open;
    }

    /// One admission.  Returns the epoch this tenancy owns; `finish` must be
    /// handed the same value back, which is what makes the pairing checkable.
    std::uint64_t admit(int lane, bool immediate, double waited_ms, int width_now) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_open) return 0;
        const double now = elapsed();
        const auto   l   = laneIndex(lane);
        // The lane still believes it holds a deck.  See stale_tenant_refusals
        // above: counted, never thrown, because the physics is not what is
        // wrong here -- the accounting is, and losing the run would lose the
        // evidence with it.
        if (_lane_busy[l]) ++_stale_refusals;
        if (_lane_jobs[l] > 0) _idle_ms_total += std::max(0.0, now - _lane_last_end[l]) * 1.0e3;
        _lane_busy[l] = true;
        ++_admits;
        if (immediate) ++_immediate;
        else _admit_wait_ms.push_back(waited_ms);
        _widths.push_back(width_now);
        const std::uint64_t epoch = ++_epoch;
        _live_epochs.push_back(epoch);
        return epoch;
    }

    void finish(int lane, std::uint64_t epoch, int width_now) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_open) return;
        const auto l      = laneIndex(lane);
        _lane_busy[l]     = false;
        _lane_last_end[l] = elapsed();
        ++_lane_jobs[l];
        _widths.push_back(width_now);
        const auto it = std::find(_live_epochs.begin(), _live_epochs.end(), epoch);
        if (it == _live_epochs.end()) ++_epoch_regressions;
        else _live_epochs.erase(it);
    }

    /// One request batch merged into a session that was already running.
    void noteBarrierAvoided() {
        std::lock_guard<std::mutex> lock(_mutex);
        ++_barriers_avoided;
    }

    void close() {
        std::lock_guard<std::mutex> lock(_mutex);
        _t1 = std::chrono::steady_clock::now();
    }

    /// The WP18 receipt.  One line, machine-readable, always the same keys.
    void report(std::ostream& out, long long session_id) const {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_open) return;
        const double wall = std::chrono::duration<double>(_t1 - _t0).count();

        // The drain that is left: with per-slot refill it is ONE case, not a
        // whole wave, and that claim is exactly this number over the per-case
        // wall.
        double tail_ms = 0.0;
        for (std::size_t l = 0; l < _lane_last_end.size(); ++l) {
            if (_lane_jobs[l] == 0) continue;
            tail_ms += std::max(0.0, wall - _lane_last_end[l]) * 1.0e3;
        }

        std::vector<int> widths = _widths;
        std::sort(widths.begin(), widths.end());
        double width_mean = 0.0;
        for (int w : widths) width_mean += static_cast<double>(w);
        if (!widths.empty()) width_mean /= static_cast<double>(widths.size());

        std::vector<double> waits = _admit_wait_ms;
        std::sort(waits.begin(), waits.end());

        int lanes_used = 0;
        for (int n : _lane_jobs)
            if (n > 0) ++lanes_used;

        out << "[RASBERY][REFILL][ROLLING] {\"session\":" << session_id
            << ",\"arena_width\":" << _width
            << ",\"lanes\":" << _lanes
            << ",\"lanes_used\":" << lanes_used
            << ",\"queue_capacity\":" << _capacity
            << ",\"admits\":" << _admits
            << ",\"immediate_admits\":" << _immediate
            << ",\"wave_barriers_avoided\":" << _barriers_avoided
            << ",\"slot_idle_ms_total\":" << _idle_ms_total
            << ",\"tail_idle_ms\":" << tail_ms
            << ",\"wall_s\":" << wall
            << ",\"width_history\":{\"samples\":" << widths.size()
            << ",\"p10\":" << pct(widths, 0.10)
            << ",\"p50\":" << pct(widths, 0.50)
            << ",\"p90\":" << pct(widths, 0.90)
            << ",\"mean\":" << width_mean
            << ",\"max\":" << (widths.empty() ? 0 : widths.back())
            << "}"
            << ",\"width_fill\":" << (_width > 0 ? width_mean / static_cast<double>(_width) : 0.0)
            << ",\"admit_wait_ms\":{\"p50\":" << pctd(waits, 0.50)
            << ",\"max\":" << (waits.empty() ? 0.0 : waits.back()) << "}"
            << ",\"epoch\":" << _epoch
            << ",\"epoch_regressions\":" << _epoch_regressions
            << ",\"stale_tenant_refusals\":" << _stale_refusals
            << ",\"live_tenancies_at_close\":" << _live_epochs.size()
            << "}" << std::endl;
    }

private:
    [[nodiscard]] std::size_t laneIndex(int lane) const {
        if (lane < 0 || lane >= static_cast<int>(_lane_busy.size())) return 0;
        return static_cast<std::size_t>(lane);
    }
    [[nodiscard]] double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
    }
    static int pct(const std::vector<int>& sorted, double q) {
        if (sorted.empty()) return 0;
        std::size_t i = static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1) + 0.5);
        if (i >= sorted.size()) i = sorted.size() - 1;
        return sorted[i];
    }
    static double pctd(const std::vector<double>& sorted, double q) {
        if (sorted.empty()) return 0.0;
        std::size_t i = static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1) + 0.5);
        if (i >= sorted.size()) i = sorted.size() - 1;
        return sorted[i];
    }

    mutable std::mutex         _mutex;
    bool                       _open     = false;
    int                        _lanes    = 0;
    int                        _width    = 0;
    int                        _capacity = 0;
    std::vector<bool>          _lane_busy;
    std::vector<double>        _lane_last_end;
    std::vector<int>           _lane_jobs;
    std::vector<int>           _widths;
    std::vector<double>        _admit_wait_ms;
    std::vector<std::uint64_t> _live_epochs;
    unsigned long long         _admits            = 0;
    unsigned long long         _immediate         = 0;
    unsigned long long         _barriers_avoided  = 0;
    unsigned long long         _stale_refusals    = 0;
    unsigned long long         _epoch_regressions = 0;
    std::uint64_t              _epoch             = 0;
    double                     _idle_ms_total     = 0.0;

    std::chrono::steady_clock::time_point _t0{};
    std::chrono::steady_clock::time_point _t1{};
};

inline RollingLedger& rollingLedger() {
    static RollingLedger instance;
    return instance;
}

} // namespace rasbery::refill
