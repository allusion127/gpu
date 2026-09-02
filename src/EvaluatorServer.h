#pragma once

// EvaluatorServer -- WP8 stage 1: the process lives, the case object does not.
//
// WHAT THIS IS.  BOTTLENECK plan Sec 5.1/WP8 stage 1 and GA evaluator plan
// Sec 6.2 Task 6: one RASBERY process that stands up its CUDA context, its
// immutable XSLIB parse, its T/H tables, its device library and its arenas
// ONCE, and then answers a stream of case requests until it is told to stop.
// Every case still builds and destroys its own Driver/CaseContext -- stage 1
// changes WHEN the process-lifetime half is torn down, and nothing about what
// a case computes.  That is what makes it a B0 change: the per-case `digest`
// must be the SAME sixteen hex digits the same deck produces from a one-shot
// `--jobs` run, and the gate is exactly that comparison.
//
// WHY IT IS WORTH ANYTHING.  Measured, not assumed (GA evaluator plan Sec 2.3):
// `outside_drive` -- the process image, the loader, CUDA context creation and
// CUDA teardown -- is 1.75-4.92 s, and it is paid ONCE PER PROCESS.  Today the
// chunked M64 launcher (tools/run_multi_gpu_batch.py) starts a new process per
// queue chunk, so a 64-case generation pays it once and three back-to-back
// generations pay it three times.  This mode pays it once for all three.  The
// honest size of that lever is therefore NOT 2-4% of a case; it is
// `outside_drive x (chunks - 1)` off a campaign, and Sec 2.3 is explicit that
// per-case physics is 88-93% of the wall and is untouched by any of this.
//
// THE CROSS-CASE SURFACE IS NOT NEW -- ALMOST.  `--batch-mode M` with more
// decks than lanes ALREADY recycles an OpenMP worker onto a second deck inside
// one process, so every process-lifetime and thread_local object in the solver
// already survives a case boundary today, and that path is bit-identity gated.
// What this mode adds is exactly the state that survives across a WAVE
// boundary -- i.e. the three teardown steps the single-shot batch branch runs
// at the end of main() and this loop defers to shutdown:
//
//   iowriter::shutdown()          per wave: NOT needed.  Every case fences its
//                                 own sessions in IO::~IO / IO::CloseResult
//                                 (IO.cpp:265-275, IO.cpp:2048-2060), so a
//                                 finished case has nothing in flight.  The
//                                 line sink is flushed per wave; the writer
//                                 thread is joined once, at shutdown.
//   rasberyReleaseBatchArena()    per wave: DELIBERATELY NOT.  This is the
//                                 lever.  Released once, at shutdown.
//   rasberyDrainPinnedRegistry()  per wave: NOT needed, and ASSERTED instead.
//                                 Every registration is leased and released by
//                                 its owner's destructor (HostPinRegistry.h),
//                                 so between waves rasberyHostPinLiveRanges()
//                                 must be 0.  The wave receipt carries it; a
//                                 nonzero value is a lease that outlived its
//                                 Driver, which is the exact defect the lease
//                                 was introduced to kill.
//
// WHAT IS REFUSED RATHER THAN APPROXIMATED.
//
//   `batch_width`  the arena is one allocation sized at the first admission and
//                  fixed for the process (CudaBICGBackend.cu).  The FIRST wave
//                  latches the width; a later wave asking for a different one
//                  is refused by name.  Stage 2's CohortContext is where a
//                  second shape becomes legal.
//
// WHAT WP10.3 TURNED FROM A REFUSAL INTO A HOOK: `fidelity`.
//
// Stage 1 refused a per-case fidelity and said why -- PhysicsFidelity resolved
// once per process from function-local statics, and Driver::SolveLoop read the
// staged-tolerance knobs into three more of its own, so a request could assert
// the process's fidelity and never change it.  That reasoning was right about
// the code and wrong about the physics: NOTHING ABOUT AN ARENA DEPENDS ON A
// CONVERGENCE TOLERANCE.  What actually bound fidelity to the process was six
// `static const` reads, and src/CaseFidelity.h is those six reads moved one
// level out, into a value the case owns:
//
//   staged tolerances    per Driver, carried into SolveLoop on SolverContext
//   loose-settle gate    same
//   burnup grid          per Driver, applied inside IO::ReadInput BEFORE the
//                        deck digest, so the coarse lane cannot share a case
//                        key with the full one (src/StatepointGrid.h)
//   feedback passes      STILL process-level, and a FLOOR: no request undoes it
//   RASBERY_PHYSICS_FIDELITY  still process-level, and coarser-only already
//
// So a wave may now carry a screening population and a promoted elite, and each
// case's receipt says which it was.  What has NOT become negotiable is the
// equality: resolveCaseFidelity() refuses unless what the case will solve is
// exactly the word it declared -- a case can no more run finer than declared
// (a strict number filed in the A2 column) than coarser (an approximation
// walking into an acceptance table).
//
// A quiet "close enough" here is how a screening result reaches an acceptance
// table, which is the failure mode WP1's exact-only contract exists for.
//
// THE `promote` OP is the other half.  It re-runs a case_key at strict/full,
// warm-started from its own screening result when one exists, and STAMPS THE
// LINK: the strict receipt carries `promoted_from` = the screening case_key.
// Without that link a promotion is two unrelated rows and "did the elite we
// shipped actually get re-run" is a question nobody can answer from the
// receipts alone.

#include "BatchRefill.h"
#include "CaseFidelity.h"
#include "CohortContext.h"
#include "CrashReport.h"
#include "CudaXsReconBackend.h"
#include "Driver.h"
#include "GpuDeviceBlockPool.h"
#include "GpuCaptureArbiter.h"
#include "HostPinRegistry.h"
#include "IoWriter.h"
#include "RunContract.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// WP10.6.  mallinfo2() lives here on glibc.  <cstdlib> above has already
// pulled <features.h> in, so __GLIBC__ is decided by the time this is read.
#if defined(__GLIBC__)
    #include <malloc.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace rasbery::evaluator {

// ---------------------------------------------------------------------------
// The request stream
// ---------------------------------------------------------------------------

/// One case, as the client names it.
struct CaseRequest {
    std::string deck;                 ///< --rasi
    std::string output;               ///< --raso; the job's output namespace
    std::string key;                  ///< opaque candidate key, echoed back
    ResultMode  result_mode = ResultMode::Full;
    /// WP10.3.  Declared AND APPLIED.  `resolved` is what
    /// resolveCaseFidelity() made of `request` against the process default, and
    /// it is what the Driver is configured with -- so the word in the receipt
    /// and the tolerances in the loop have exactly one source.
    FidelityRequest request_fidelity;
    CaseFidelity    resolved_fidelity;
    /// True when this case arrived as `{"op":"promote"}`.  Reported; nothing in
    /// the solve reads it.
    bool promoted = false;
    /// WP10.2.  `warm_start_from` seeds this case's BOC flux/boron/keff from a
    /// parent's saved state; `save_warm_state` writes this case's own for a
    /// child.  Both empty is the default and runs nothing.  This is the field
    /// a generation-to-generation warm start travels on: the controller names
    /// the parent it chose, and the evaluator refuses -- to a COLD start, in
    /// the receipt -- anything it cannot honour.
    std::string warm_start_from;
    std::string save_warm_state;
};

/// One generation.
struct WaveRequest {
    long long                wave_id     = 0;
    std::string              jobs_manifest;
    int                      batch_width = 0; ///< 0 = the process default
    std::vector<CaseRequest> cases;
    /// WP10.3.  What a MANIFEST job runs at.  A manifest line names a deck, an
    /// output and a result mode and nothing else, so its fidelity is the wave's
    /// declaration resolved once -- and with no declaration that is the process
    /// default, which is what a manifest wave has always been.
    CaseFidelity manifest_fidelity = processCaseFidelity();
};

/// What ended the stream.  Reported in the process receipt: a run that stopped
/// because nobody spoke to it for `--evaluator-idle-timeout` seconds and a run
/// that was told to shut down are not the same run.
enum class StopReason { Shutdown, EndOfStream, IdleTimeout, Error };

inline const char* stopReasonName(StopReason r) {
    switch (r) {
        case StopReason::Shutdown:    return "shutdown";
        case StopReason::EndOfStream: return "end_of_stream";
        case StopReason::IdleTimeout: return "idle_timeout";
        case StopReason::Error:       return "error";
    }
    return "unknown";
}

/// A JSONL line source that can FOLLOW a file being appended to.
///
/// Three shapes, one class, because the GA controller gets to pick and the
/// evaluator must not care: `-` is stdin (a pipe from the controller; EOF is
/// final and no timeout applies, because a portable timed read on a blocking
/// stdin does not exist and pretending otherwise would just hang differently);
/// a path with no idle timeout is a manifest read once; a path WITH an idle
/// timeout is a queue file the controller appends generations to, polled until
/// it has been quiet for that long.
class RequestStream {
public:
    RequestStream(const std::string& path, double idle_timeout_s)
        : _path(path), _idle_timeout_s(idle_timeout_s), _stdin(path == "-") {
        if (!_stdin) {
            _file.open(_path);
            _ok = static_cast<bool>(_file);
        }
    }

    [[nodiscard]] bool ok() const { return _ok; }
    [[nodiscard]] const std::string& path() const { return _path; }

    /// Next non-blank line, or false with `reason` set.
    bool next(std::string& line, StopReason& reason) {
        const auto deadline_start = std::chrono::steady_clock::now();
        for (;;) {
            std::istream& in = _stdin ? std::cin : static_cast<std::istream&>(_file);
            if (std::getline(in, line)) {
                // Windows-authored queue file read under WSL: a trailing \r
                // would become part of a deck path and every case would fail
                // to open a file whose name only differs by an invisible byte.
                // The same trap main.cpp's manifest reader was built around.
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                    line.pop_back();
                if (line.find_first_not_of(" \t") == std::string::npos) continue;
                if (line[line.find_first_not_of(" \t")] == '#') continue;
                return true;
            }
            if (_stdin || _idle_timeout_s <= 0.0) {
                reason = StopReason::EndOfStream;
                return false;
            }
            const double idle =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - deadline_start)
                    .count();
            if (idle >= _idle_timeout_s) {
                reason = StopReason::IdleTimeout;
                return false;
            }
            // Clear the EOF bit so an append becomes visible, and poll.  50 ms
            // is two orders of magnitude below the cheapest case and four below
            // a generation, so it costs nothing and bounds the latency a
            // controller sees between writing a line and the wave starting.
            _file.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

private:
    std::string   _path;
    double        _idle_timeout_s = -1.0;
    bool          _stdin          = false;
    std::ifstream _file;
    bool          _ok             = true;
};

// ---------------------------------------------------------------------------
// Options and receipts
// ---------------------------------------------------------------------------

/// Reads one `--jobs` manifest into the three parallel vectors.  Injected from
/// main.cpp rather than reimplemented here ON PURPOSE: the manifest grammar
/// (quoting, the third result-mode field, the CRLF strip, the line-numbered
/// errors) has exactly one implementation, and a second one would be a second
/// answer to "what is a job".
using ManifestReader = std::function<bool(const std::string& path,
                                          std::vector<std::string>& inputs,
                                          std::vector<std::string>& outputs,
                                          std::vector<ResultMode>&  modes,
                                          ResultMode                default_mode,
                                          std::string&              error)>;

struct Options {
    std::string    request_path = "-";
    int            batch_width  = 0;
    ResultMode     default_result_mode = ResultMode::Full;
    double         idle_timeout_s      = -1.0;
    bool           isolation_check     = false;
    int            host_threads_override = 0;
    int            visible_cpus          = 1;
    ManifestReader read_manifest;
};

namespace detail {

/// How many per-case samples the process keeps resident, per series.
///
/// 4096 is chosen against the two things it has to serve.  A generation is 16
/// to 64 cases, so the window spans 64 to 256 GENERATIONS -- far more than any
/// percentile needs to be stable, and far more than the drift a reader is
/// looking for.  And 4096 doubles is 32 kB per series: it can never again be
/// the largest number in the memory receipt, which is exactly what went wrong.
inline std::size_t sampleWindowCapacity() {
    static const std::size_t capacity = [] {
        const char*     v = std::getenv("RASBERY_EVALUATOR_SAMPLE_WINDOW");
        const long long requested = (v && *v) ? std::atoll(v) : 0;
        return requested > 0 ? static_cast<std::size_t>(requested) : std::size_t{4096};
    }();
    return capacity;
}

// ---------------------------------------------------------------------------
// WP18  The rolling gates
// ---------------------------------------------------------------------------

/// RASBERY_EVALUATOR_ROLLING -- read ONCE, like every other gate in this tree.
///
/// OFF IS BYTE-IDENTICAL, and that is a design constraint and not a hope: the
/// flag is read in exactly two places (here, and once in run()), every line the
/// mode prints is inside a branch this predicate guards, and the wave path is
/// not touched at all.  A campaign that measures the two arms is comparing one
/// binary against itself, so a difference in the receipts is a difference in
/// the SCHEDULING and can be nothing else.
inline bool rollingEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_EVALUATOR_ROLLING");
        return v != nullptr && *v != '\0' && std::string(v) != "0";
    }();
    return on;
}

/// How many cases the queue holds AHEAD of the arena.
///
/// WHY IT IS BOUNDED AT ALL.  An unbounded queue would let a client hand the
/// evaluator its whole campaign in one write and then stop reading, which is
/// the classic two-pipe deadlock: stdin fills, the reader blocks, stdout fills,
/// the lanes block in reportCase, and nothing moves.  A bound turns that into
/// backpressure the client feels immediately.
///
/// WHY THE DEFAULT IS GENEROUS (4 per lane).  The number that has to stay above
/// zero is `queue.size()` AT THE MOMENT A LANE FINISHES -- that is what makes an
/// admit `immediate`.  With one case in hand per lane, a burst of simultaneous
/// completions empties it and the next admits wait for the client's round trip.
/// Four per lane is ~2 s of arena at the measured 30-90 s case, and it is 800
/// bytes a lane.
inline int rollingQueueCapacity(int lanes) {
    static const int per_lane = [] {
        const char*     v = std::getenv("RASBERY_EVALUATOR_ROLLING_QUEUE");
        const long long r = (v != nullptr && *v != '\0') ? std::atoll(v) : 0;
        return r > 0 ? static_cast<int>(r) : 4;
    }();
    return std::max(1, lanes) * per_lane;
}

/// What the READY receipt tells a client to keep in flight: width + prefetch.
inline int rollingPrefetch() {
    static const int n = [] {
        const char*     v = std::getenv("RASBERY_EVALUATOR_ROLLING_PREFETCH");
        const long long r = (v != nullptr && *v != '\0') ? std::atoll(v) : -1;
        return r >= 0 ? static_cast<int>(r) : 2;
    }();
    return n;
}

} // namespace detail

/// A fixed-capacity ring of the most recent samples, plus the two facts a ring
/// would otherwise destroy: how many were ever observed, and the exact maximum.
///
/// WHY A RING AND NOT A RESERVOIR.  A reservoir sample keeps the whole run in
/// view but needs a random number generator, and a receipt whose p90 depends on
/// a seed is a receipt two runs of the same deck can disagree about.  This
/// repository's entire gate structure is bit-identity; a nondeterministic
/// percentile has no place in it.  A ring is deterministic, and what it gives
/// up -- the first cases of a very long run -- is the part a stability question
/// least wants: "is the process still behaving" is a question about NOW.
///
/// `observed` and `max` are exact over every case regardless, so nothing that
/// was true of the whole run before is unavailable after.
class SampleWindow {
public:
    void push(double value) {
        ++_observed;
        if (value > _max) _max = value;
        const std::size_t capacity = detail::sampleWindowCapacity();
        if (_ring.size() < capacity) {
            _ring.push_back(value);
            return;
        }
        // A capacity lowered by the environment between two pushes would leave
        // the ring longer than the cap; trim before writing so the bound holds
        // from the first push after the change rather than never.
        while (_ring.size() > capacity) _ring.erase(_ring.begin());
        if (_cursor >= _ring.size()) _cursor = 0;
        _ring[_cursor] = value;
        _cursor = (_cursor + 1) % _ring.size();
    }

    /// The resident samples.  Order is not preserved and does not matter:
    /// percentile() sorts.
    [[nodiscard]] const std::vector<double>& values() const { return _ring; }
    [[nodiscard]] std::size_t   resident() const { return _ring.size(); }
    [[nodiscard]] std::uint64_t observed() const { return _observed; }
    [[nodiscard]] std::size_t   bytes() const { return _ring.size() * sizeof(double); }
    /// Exact over every case, not over the window.
    [[nodiscard]] double max() const { return _observed == 0 ? 0.0 : _max; }

private:
    std::vector<double> _ring;
    std::size_t         _cursor   = 0;
    std::uint64_t       _observed = 0;
    double              _max      = 0.0;
};

/// WP19.1.  IS THIS DEATH THE CAPTURE RACE'S SILENT FACE?
///
/// The loud face of the stand-up race (WP19) named itself: a cudaError_t in the
/// 900 block, printed with the API that refused.  The silent face does not.  A
/// graph built while a sibling lane was capturing can come back INSTANTIATED
/// and wrong -- a body whose uploads were elided against shadows the discarded
/// first attempt committed -- and the first thing that notices is BICGCMFD's
/// non-finite guard, four frames and one graph replay away from the cause.
///
/// So the string is the signal, and it is deliberately the exact text
/// CudaBICGBackend.cu:solveCommon and BICGCMFD.cpp throw.  A substring match on
/// a message this tree owns is not a heuristic about CUDA; it is a match on our
/// own thrown text, and tools/test_capture_standup_isolation_contract.py holds
/// the two spellings against each other so a rename cannot silently unhook it.
inline bool captureRaceCorruptionSuspect(const std::string& failure) {
    return failure.find("non-finite") != std::string::npos;
}

/// Everything the process receipt reports, accumulated across waves.
struct Summary {
    long long cases        = 0;
    long long ok           = 0;
    long long failed       = 0;
    long long generations  = 0;
    long long refused      = 0; ///< requests rejected before a Driver was built
    /// WP10.3.  Cases whose resolved fidelity was NOT the process default, and
    /// cases that arrived as `{"op":"promote"}`.  A soak asserts on both: a run
    /// that meant to exercise mixed fidelity and reports zero overrides
    /// exercised nothing, which is the way a soak passes for the wrong reason.
    long long fidelity_overrides   = 0;
    long long promotions           = 0;
    long long isolation_checks     = 0;
    long long isolation_mismatches = 0;
    long long isolation_adjacent   = 0;
    double    uptime_s     = 0.0;
    double    drive_s      = 0.0;
    int       latched_width = 0;
    /// WP10.5.  BOUNDED, because the host 181 soak at 55c0dce watched these two
    /// climb 18 -> 36 -> 54 -> 72 -> 90 and the new attribution named them as the
    /// only container that moved -- on a run whose RSS moved by 3.3 GB.  Ninety
    /// doubles is 1.4 kB; they were never the leak, but "the only thing that
    /// grew" is what an attribution reports when the only thing that grows is
    /// the thing that cannot matter.  Two repairs, and the second is the one
    /// that mattered: the window is capped so it stops being a false lead, and
    /// the receipt now publishes its BYTES so a mover can be weighed instead of
    /// merely noticed.
    SampleWindow case_seconds;
    SampleWindow teardown_ms;
    StopReason stop = StopReason::EndOfStream;
    std::uint64_t pin_live_ranges_between_waves = 0;
    /// WP10.4.  The previous generation's RSS sample, so the per-generation
    /// DELTA is a fact the process states rather than a subtraction a reader
    /// has to do across two lines that may not both be present.
    std::uint64_t rss_bytes_last  = 0;
    std::uint64_t rss_bytes_first = 0;
    /// WP10.6.  The same, for the DEVICE footprint this process holds.
    std::uint64_t device_bytes_last = 0;
};

namespace detail {

inline double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const auto   lo  = static_cast<std::size_t>(pos);
    const auto   hi  = std::min(lo + 1, v.size() - 1);
    const double f   = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - f) + v[hi] * f;
}

/// JSON string escape for the receipt fields that carry client text (paths, the
/// candidate key, an exception's what()).  Small on purpose: the receipt is one
/// line and a line that could be split by a stray quote is not machine
/// readable, which is the only property it has.
inline std::string jsonEscape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) out += ' ';
                else out += c;
        }
    }
    return out;
}

inline std::string quoted(const std::string& text) {
    return "\"" + jsonEscape(text) + "\"";
}

// ---------------------------------------------------------------------------
// WP10.4 -- THE MEMORY RECEIPT
// ---------------------------------------------------------------------------
//
// WHY THIS EXISTS.  The host 181 soak at `91004f7` (5 generations x width 16,
// persistent evaluator) measured RSS growing +17.41 MB per generation over its
// second half, against an 8.0 MB/generation budget -- and the run before it
// measured +37.64 MB/generation, so it is a signal and not a sample.  Every
// mechanism receipt the soak asserts on came back ZERO (no slot duplicates, no
// stale tenants, no double releases, no pin ranges between waves), which is the
// problem: the process was growing and NOTHING IN ANY RECEIPT COULD NAME WHAT
// WAS GROWING.  An evaluator that has to survive 10k generations cannot be
// debugged one profiler session at a time on a host nobody can attach to.
//
// So the process now reports, once per generation, the number the soak measures
// from outside (RSS) BESIDE the sizes of every container that could explain it.
// A generation where rss_delta_mb is large and every cache_entries number is
// flat says the growth is not in these caches and the next place to look is the
// allocator or a library; a generation where one of them moves with the RSS has
// named itself.  That is the whole design goal: the NEXT soak should not have
// to guess.
//
// `live_cases` is the other half.  It is incremented before a Driver is
// constructed and decremented after it is destroyed, so at the moment this line
// prints -- after every case in the wave has been joined -- it MUST be 0.  A
// nonzero value is a Driver that outlived its case, which is a leak of
// everything a case owns and would otherwise be invisible.

/// Drivers currently constructed.  See the note above.
inline std::atomic<long long>& liveCases() {
    static std::atomic<long long> live{0};
    return live;
}

/// Resident set size of this process in bytes, or 0 where it cannot be read.
///
/// /proc/self/statm field 2 is the resident page count -- the same quantity
/// tools/soak_run.py samples from outside with `ps`, deliberately, so the two
/// numbers can be compared instead of merely coexisting.  Returning 0 rather
/// than guessing on a platform without /proc is the honest failure: the receipt
/// then says the process could not measure itself, which is a different fact
/// from "the process did not grow".
inline std::uint64_t residentBytes() {
    std::ifstream statm("/proc/self/statm");
    if (!statm) return 0;
    unsigned long long total = 0, resident = 0;
    if (!(statm >> total >> resident)) return 0;
    return static_cast<std::uint64_t>(resident) * static_cast<std::uint64_t>(4096);
}

/// Peak RSS in bytes (VmHWM), or 0.  A peak that is far above the current value
/// says the growth is transient and the allocator is holding it, which is a
/// different repair from a container that never shrinks.
inline std::uint64_t peakResidentBytes() {
    std::ifstream status("/proc/self/status");
    if (!status) return 0;
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) != 0) continue;
        std::istringstream fields(line.substr(6));
        unsigned long long kb = 0;
        if (!(fields >> kb)) return 0;
        return static_cast<std::uint64_t>(kb) * 1024ull;
    }
    return 0;
}

inline double toMiB(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

/// WP10.6.  Bytes the C library is holding that no case is using.
///
/// The 238 soak grew 3,563.8 MB and every container the receipt could see had
/// moved by 0.005 MB, so the report handed off with "look at the allocator
/// next".  This IS the allocator, asked directly.  `arena` + `hblkhd` is what
/// glibc has taken from the kernel; `uordblks` is what callers currently hold;
/// the difference is retention -- free lists a per-thread arena will hand back
/// to the next case and will not hand back to the OS.  A soak that sees RSS
/// climb while THIS number climbs with it has found a high-water mark, not a
/// leak, and the two need opposite repairs.
///
/// `{0, 0, false}` where the C library cannot answer, and the receipt says
/// `malloc_readable:false` rather than printing a zero that would read as
/// "the allocator is holding nothing".
struct MallocRetention {
    std::uint64_t taken_bytes  = 0;
    std::uint64_t in_use_bytes = 0;
    bool          readable     = false;
};

inline MallocRetention mallocRetention() {
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 33)
    // mallinfo2, NOT mallinfo: the fields of the older struct are `int`, and an
    // evaluator holding 4.5 GB overflows them silently.  A wrapped counter is
    // worse than no counter, because it is believable.
    const struct mallinfo2 info = mallinfo2();
    MallocRetention out;
    out.taken_bytes  = static_cast<std::uint64_t>(info.arena) +
                       static_cast<std::uint64_t>(info.hblkhd);
    out.in_use_bytes = static_cast<std::uint64_t>(info.uordblks);
    out.readable     = true;
    return out;
#else
    return MallocRetention{};
#endif
#else
    return MallocRetention{};
#endif
}

/// `<output>` with `.iso<n>` spliced before the extension.
///
/// The isolation re-run must not write where the real case wrote: the result
/// file would be reopened and the restart namespace is DERIVED from the output
/// path (Driver::RestartPath), so sharing it would have the check overwrite the
/// thing it is checking.  A different stem changes where bytes land and nothing
/// the trajectory reads, which is why comparing the DIGEST across the two is a
/// valid isolation test and comparing the files would not be.
inline std::string isolationOutput(const std::string& output, long long tag) {
    const std::size_t dot   = output.find_last_of('.');
    const std::size_t slash = output.find_last_of("/\\");
    const std::string suffix = ".iso" + std::to_string(tag);
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return output + suffix;
    return output.substr(0, dot) + suffix + output.substr(dot);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

/// Parse one JSONL request line.  `op` is returned separately because it is the
/// only field whose absence is not recoverable.
inline bool parseRequest(const std::string& line, nlohmann::json& object, std::string& op,
                         std::string& error) {
    try {
        object = nlohmann::json::parse(line);
    } catch (const std::exception& e) {
        error = std::string("not JSON: ") + e.what();
        return false;
    }
    if (!object.is_object()) {
        error = "a request must be a JSON object";
        return false;
    }
    if (!object.contains("op") || !object["op"].is_string()) {
        error = R"(a request must carry a string "op" (case | wave | run | shutdown))";
        return false;
    }
    op = object["op"].get<std::string>();
    return true;
}

/// Defined below, beside applyWaveFidelityDefault, because the two are one
/// rule: what a request may say about fidelity and what a wave may default.
inline bool parseFidelityFields(const nlohmann::json& object, FidelityRequest& out,
                                std::string& error);

/// Pull one case out of a `{"op":"case", ...}` object.
inline bool parseCase(const nlohmann::json& object, ResultMode default_mode, CaseRequest& out,
                      std::string& error) {
    if (!object.contains("deck") || !object["deck"].is_string() ||
        object["deck"].get<std::string>().empty()) {
        error = R"(a case needs a non-empty string "deck")";
        return false;
    }
    out.deck = object["deck"].get<std::string>();
    if (!object.contains("output") || !object["output"].is_string() ||
        object["output"].get<std::string>().empty()) {
        // The same rule main.cpp's manifest reader enforces, and for the same
        // reason: without an output path every case would write the deck
        // directory's default result.h5, i.e. all of them would collide.
        error = R"(a case needs a non-empty string "output" -- without one every case )"
                R"(writes the same default result.h5 and they overwrite each other)";
        return false;
    }
    out.output = object["output"].get<std::string>();
    out.result_mode = default_mode;
    if (object.contains("result_mode")) {
        if (!object["result_mode"].is_string() ||
            !ParseResultMode(object["result_mode"].get<std::string>(), out.result_mode)) {
            error = R"("result_mode" must be one of full | pin-off | light)";
            return false;
        }
    }
    // WP10.2.  Both are plain strings and both default to empty (= off); a
    // non-string is refused by name rather than coerced, because a warm start
    // the client thought it asked for and did not get is a silent A/B.
    if (object.contains("warm_start_from")) {
        if (!object["warm_start_from"].is_string()) {
            error = R"("warm_start_from" must be a string)";
            return false;
        }
        out.warm_start_from = object["warm_start_from"].get<std::string>();
    }
    if (object.contains("save_warm_state")) {
        if (!object["save_warm_state"].is_string()) {
            error = R"("save_warm_state" must be a string)";
            return false;
        }
        out.save_warm_state = object["save_warm_state"].get<std::string>();
    }
    if (object.contains("key")) {
        if (!object["key"].is_string()) {
            error = R"("key" must be a string)";
            return false;
        }
        out.key = object["key"].get<std::string>();
    }
    if (!parseFidelityFields(object, out.request_fidelity, error)) return false;
    return true;
}

/// WP10.3.  Pull the fidelity half of a request out of *object*.
///
/// Shared by `case`, `promote` and `wave`, because a wave-level declaration is
/// a DEFAULT for the cases that named nothing and has to mean the same thing
/// there as it does on a case.  Two parsers would eventually mean two things.
inline bool parseFidelityFields(const nlohmann::json& object, FidelityRequest& out,
                                std::string& error) {
    const auto string_field = [&object, &error](const char* name, std::string& value) {
        if (!object.contains(name)) return true;
        if (!object[name].is_string()) {
            error = std::string("\"") + name + "\" must be a string";
            return false;
        }
        value = object[name].get<std::string>();
        return true;
    };
    // `physics_fidelity` is the plan Sec 6.2 spelling and `fidelity` the
    // campaign shorthand; both name one field, and parsePhysicsFidelity accepts
    // either vocabulary, so a controller may use whichever its manifests use.
    if (!string_field("fidelity", out.fidelity)) return false;
    if (out.fidelity.empty() && !string_field("physics_fidelity", out.fidelity)) return false;
    if (object.contains("statepoint_grid")) {
        if (!object["statepoint_grid"].is_string()) {
            error = R"("statepoint_grid" must be a string: full | coarse | three | a )"
                    R"(cumulative GWd/t list such as "0.5,1,2,4,8,16")";
            return false;
        }
        out.statepoint_grid = object["statepoint_grid"].get<std::string>();
        out.has_grid        = true;
    }
    const auto number_field = [&object, &error](const char* name, bool& has, double& value) {
        if (!object.contains(name)) return true;
        if (!object[name].is_number()) {
            error = std::string("\"") + name + "\" must be a number >= 1.0";
            return false;
        }
        value = object[name].get<double>();
        has   = true;
        return true;
    };
    if (!number_field("staged_flux_tol", out.has_flux_mult, out.flux_mult)) return false;
    if (!number_field("staged_xe_tol", out.has_xe_mult, out.xe_mult)) return false;
    if (object.contains("staged_loose_settle")) {
        if (!object["staged_loose_settle"].is_boolean()) {
            error = R"("staged_loose_settle" must be a boolean)";
            return false;
        }
        out.loose_settle     = object["staged_loose_settle"].get<bool>();
        out.has_loose_settle = true;
    }
    if (!string_field("promoted_from", out.promoted_from)) return false;
    return true;
}

/// Fill in a case's unset fidelity fields from the wave's declaration.
///
/// A WAVE DECLARATION IS A DEFAULT AND NEVER AN OVERRIDE.  If it overrode, a
/// mixed wave -- the whole point of WP10.3 -- would be impossible to express:
/// the wave line arrives AFTER the case lines it collects, so an override would
/// retroactively restate what sixty-four cases had already declared, and the
/// case that asked for strict inside a screening generation would silently
/// become a screening case.  So: a case that named a field keeps it.
///
/// WP10.7.  A PROMOTION TAKES NO WAVE DEFAULT AT ALL, and `promoted` is the
/// whole of the rule.
///
/// THE ASYMMETRY THIS REMOVES.  `op":"promote"` is one op with two admission
/// doors today.  In ROLLING mode the request is resolved the instant it is read
/// (Server::run, `if (detail::rollingEnabled())`), against
/// processCaseFidelity() -- there is no wave to take a default from.  In WAVE
/// mode the same line waits for the `wave` line and is resolved against THAT,
/// so the identical request means two different things depending on which mode
/// the process is in.  The op exists precisely so that a promotion's defaults
/// are "the ones a promotion must have rather than the ones the process happens
/// to carry"; a wave-mode default is that same accident wearing a wave's name.
///
/// AND IT IS NOT A THEORETICAL ONE.  `promote` flips three fields at parse time
/// -- `strict`, `full` output, the full burnup grid -- and does NOT flip
/// `flux_mult`, `xe_mult` or `loose_settle`.  So a wave that declares staged
/// tolerances (RunContract.h: PhysicsFidelity::StagedA2, `acceptance_eligible`
/// FALSE) used to hand them to the promotion inside it, and the acceptance
/// lane's re-run -- the one row in a GA generation whose entire job is to be
/// acceptance-eligible -- silently converged at screening tolerances while
/// reporting `policy:"strict"`.  Refusing the wave instead would be worse: a
/// mixed wave is legal by construction (WP10.3) and a promotion inside a
/// staged screening generation is exactly the case the op was written for.
inline void applyWaveFidelityDefault(const FidelityRequest& wave, FidelityRequest& request,
                                     bool promoted) {
    if (promoted) return;
    if (request.fidelity.empty()) request.fidelity = wave.fidelity;
    if (!request.has_grid && wave.has_grid) {
        request.statepoint_grid = wave.statepoint_grid;
        request.has_grid        = true;
    }
    if (!request.has_flux_mult && wave.has_flux_mult) {
        request.flux_mult     = wave.flux_mult;
        request.has_flux_mult = true;
    }
    if (!request.has_xe_mult && wave.has_xe_mult) {
        request.xe_mult     = wave.xe_mult;
        request.has_xe_mult = true;
    }
    if (!request.has_loose_settle && wave.has_loose_settle) {
        request.loose_settle     = wave.loose_settle;
        request.has_loose_settle = true;
    }
}

// ---------------------------------------------------------------------------
// The server
// ---------------------------------------------------------------------------

/// One wave's jobs, resolved from either a manifest or inline case lines.
struct WaveJobs {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<ResultMode>  modes;
    std::vector<std::string> keys;
    std::vector<std::string> warm_from;
    std::vector<std::string> warm_save;
    /// WP10.3.  One resolved fidelity per job, parallel to the five above.  A
    /// manifest-sourced job gets the wave's, an inline case gets its own.
    std::vector<CaseFidelity> fidelity;
    std::vector<bool>         promoted;
};

// ---------------------------------------------------------------------------
// WP18  The rolling queue
// ---------------------------------------------------------------------------
//
// WHAT THE ARENA ALREADY IS, AND WHY THAT IS THE WHOLE ARGUMENT.
//
// The batch arena is NOT lockstep.  A slot is owned by one host thread for one
// deck's whole life; that thread runs its own statepoint loop, its own T/H
// loop and its own boron search, and GpuPhysicsArena.h:40 says so in as many
// words ("under an asynchronous scheduler slot A is in Outer while slot B is in
// Depletion").  The only coupling between slots is a per-LAUNCH opportunistic
// rendezvous: a thread arriving at a CMFD solve joins whatever batch is open,
// and the elected launcher lingers only while `pending.size() < inUseCount()`
// (CudaBICGBackend.cu:6907) -- i.e. it waits for the slots that are CURRENTLY
// HELD, not for a phase, not for a statepoint, and not for a wave.  Release a
// slot and inUseCount() drops and the rendezvous stops waiting for it
// (CudaBICGBackend.cu:6452-6466).  The one thing participants of a single
// launch must share is the inner BiCGSTAB budget `nmax`, which is a per-deck
// environment constant and not a position in a schedule.
//
// So there is no barrier in the arena to remove.  The barrier is HERE: runWave
// takes a fixed job vector and ends in an implicit OpenMP `for` barrier, and
// the dispatcher claims exactly `batch_width` jobs per chunk
// (auto_claim_size(): 128 jobs / 8 workers / width 8 -> claim 8), so every wave
// is ONE case per lane, no lane ever refills, and the wave ends when its
// slowest case does.  That is the measured 0.447 width_fill and the 155 s tail
// of a 390 s wall -- an artefact of how the job list is cut up, not of the GPU.
//
// WHAT THIS QUEUE CHANGES.  Cases stream in while the lanes run.  A lane that
// finishes pops the next one and admits it immediately: the Driver is destroyed
// (slot released, `Slot{}` reset, batchSlotIsReset audited, admissions++), and
// the next Driver acquires a slot through exactly the same door.  Nothing about
// what a case computes moves, which is why the gate is per-case digest equality
// against the wave arm.
//
// THE ONE RULE THAT IS NOT THE WAVE'S.  Wave mode refuses a wave whose --raso
// paths collide, because two Drivers on one output race inside one HDF5 file.
// A rolling session has no wave to scope that to, and reusing an output across
// generations is legitimate (it is how a GA re-evaluates a promoted elite).  So
// the rule becomes what it always meant: an output may not be held by two
// tenants AT THE SAME TIME.  A case whose output is in flight is not refused
// and not dropped -- it waits for the tenant that holds it, which is the same
// serialisation a wave boundary used to provide by accident.

/// One admitted case, and the epoch its tenancy owns.
struct RollingJob {
    std::string  deck;
    std::string  output;
    std::string  key;
    std::string  warm_from;
    std::string  warm_save;
    ResultMode   mode = ResultMode::Full;
    CaseFidelity fidelity;
    bool         promoted = false;
    int          index    = 0;
};

/// A bounded MPSC queue with one extra rule: an output may have one tenant.
class RollingQueue {
public:
    void configure(std::size_t capacity) {
        std::lock_guard<std::mutex> lock(_mutex);
        _capacity = capacity > 0 ? capacity : 1;
        _closed   = false;
    }

    /// Producer side.  Blocks while the queue is at capacity; returns false
    /// only if the queue closed while waiting, in which case the job was NOT
    /// taken and the caller still owns it.
    bool push(RollingJob job) {
        std::unique_lock<std::mutex> lock(_mutex);
        _space.wait(lock, [&] { return _queue.size() < _capacity || _closed; });
        if (_closed) return false;
        _queue.push_back(std::move(job));
        _work.notify_one();
        return true;
    }

    /// Consumer side.  Returns false ONLY when the queue is closed and there is
    /// nothing left to run -- an empty-but-open queue waits, because in this
    /// mode "empty" means "the client has not sent the next one yet" and not
    /// "the run is over".  That distinction is the whole mode.
    ///
    /// `immediate` is false iff the lane had to wait, which is what separates
    /// "the arena was kept full" from "the arena drained and the harness was
    /// late"; the two have the same throughput symptom and different fixes.
    bool pop(RollingJob& out, bool& immediate, double& waited_ms) {
        std::unique_lock<std::mutex> lock(_mutex);
        const auto t0 = std::chrono::steady_clock::now();
        immediate     = true;
        waited_ms     = 0.0;
        for (;;) {
            const auto it = firstAdmissible();
            if (it != _queue.end()) {
                out = std::move(*it);
                _queue.erase(it);
                _inflight.push_back(out.output);
                if (!immediate)
                    waited_ms = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count() *
                                1.0e3;
                _space.notify_one();
                return true;
            }
            // Closed AND nothing left: the session is over.  Note the order --
            // a closed queue that still holds a job whose output is in flight
            // must WAIT for that tenant, not drop the job.
            if (_closed && _queue.empty()) return false;
            immediate = false;
            _work.wait(lock);
        }
    }

    /// A tenancy ended: its output is free for the next case that names it.
    void finish(const std::string& output) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto it = std::find(_inflight.begin(), _inflight.end(), output);
            if (it != _inflight.end()) _inflight.erase(it);
        }
        _work.notify_all();
        _space.notify_all();
    }

    /// No more work is coming: the lanes drain what is queued and then exit.
    void close() {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _closed = true;
        }
        _work.notify_all();
        _space.notify_all();
    }

    /// A new session on the same queue object.  Only ever called after every
    /// lane has exited, so there is nothing to race with.
    void reopen() {
        std::lock_guard<std::mutex> lock(_mutex);
        _closed = false;
    }

    [[nodiscard]] std::size_t queued() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }
    [[nodiscard]] std::size_t inflight() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _inflight.size();
    }

private:
    /// The first queued job whose output nobody is holding.  FIFO otherwise:
    /// skipping is the exception the one-tenant-per-output rule forces, never a
    /// scheduling policy.
    std::deque<RollingJob>::iterator firstAdmissible() {
        for (auto it = _queue.begin(); it != _queue.end(); ++it)
            if (std::find(_inflight.begin(), _inflight.end(), it->output) == _inflight.end())
                return it;
        return _queue.end();
    }

    mutable std::mutex       _mutex;
    std::condition_variable  _work;
    std::condition_variable  _space;
    std::deque<RollingJob>   _queue;
    std::vector<std::string> _inflight;
    std::size_t              _capacity = 64;
    bool                     _closed   = false;
};

class Server {
public:
    Server(const Options& options, std::ostream& out) : _options(options), _out(out) {
        _t0 = std::chrono::steady_clock::now();
    }

    /// Read the stream to its stop condition.  Returns the process exit code:
    /// nonzero if any case failed or any request was refused, because a
    /// generation that silently lost four candidates is not a generation.
    int run() {
        // WP19.2.  BEFORE THE FIRST REQUEST, because a worker that dies with no
        // record is exactly what block 38's two SIGSEGVs were: `completed:0`,
        // no [CASE] receipt, no core (ulimit -c 0, coredumpctl unreadable), and
        // a forensic dead end.  The handler costs five sigaction calls and one
        // warm-up backtrace() at stand-up and nothing at all afterwards; it
        // re-raises, so the dispatcher still sees returncode -11.  This is the
        // worker's entry point, which is why it is installed here and not in
        // main.cpp: every batch process the dispatcher starts comes through
        // `--evaluator-jsonl`.
        rasbery::crash::install();
        RequestStream stream(_options.request_path, _options.idle_timeout_s);
        if (!stream.ok()) {
            _out << "[RASBERY][EVALUATOR][FAIL] {\"what\":\"cannot open request stream\","
                    "\"path\":"
                 << detail::quoted(_options.request_path) << "}" << std::endl;
            _summary.stop = StopReason::Error;
            return 2;
        }

        _out << "[RASBERY][EVALUATOR][READY] {\"request_stream\":"
             << detail::quoted(_options.request_path)
             << ",\"batch_width\":" << _options.batch_width
             << ",\"result_mode\":\"" << ResultModeName(_options.default_result_mode)
             << "\",\"physics_fidelity\":\"" << physicsFidelityName(effectivePhysicsFidelity())
             // WP10.3.  The DEFAULT a case inherits, and the floor no case can
             // climb above -- two different facts that used to be one.  A
             // client reads `fidelity_per_case` to know this binary will honour
             // a per-case declaration instead of refusing it; an older
             // evaluator has no such field, which is the honest answer.
             << "\",\"fidelity_per_case\":true"
             << ",\"fidelity_default\":\"" << processCaseFidelity().policy()
             << "\",\"fidelity_floor\":\"" << physicsPolicyName(processFidelityFloor())
             << "\",\"statepoint_grid_default\":\"" << processCaseFidelity().gridToken()
             << "\",\"idle_timeout_s\":" << _options.idle_timeout_s
             << ",\"isolation_check\":" << (_options.isolation_check ? "true" : "false");
        // WP18.  PRINTED ONLY WHEN THE MODE IS ON, so a wave-mode run's stdout
        // is byte-for-byte what it was before this work package existed -- the
        // feature-off identity gate is a property of the source, not of a diff
        // somebody remembered to run.  A client reads `rolling` to learn that
        // this binary will run cases as they arrive instead of collecting them,
        // and `rolling_target_inflight` to learn how many to keep sent.
        if (detail::rollingEnabled())
            _out << ",\"rolling\":true,\"rolling_prefetch\":" << detail::rollingPrefetch()
                 << ",\"rolling_target_inflight\":"
                 << (std::max(1, _options.batch_width) + detail::rollingPrefetch());
        _out << "}" << std::endl;

        std::vector<CaseRequest> pending;
        std::string              line;
        StopReason               reason = StopReason::EndOfStream;
        bool                     stop   = false;

        while (!stop && stream.next(line, reason)) {
            nlohmann::json object;
            std::string    op;
            std::string    error;
            if (!parseRequest(line, object, op, error)) {
                refuse(error, line);
                continue;
            }
            if (op == "shutdown") {
                reason = StopReason::Shutdown;
                stop   = true;
                break;
            }
            if (op == "case" || op == "promote") {
                CaseRequest request;
                // WP10.3.  A PROMOTION IS A CASE WITH THREE DEFAULTS FLIPPED,
                // and it is a separate op only so that those defaults are the
                // ones a promotion must have rather than the ones the process
                // happens to carry.  `strict`, `full` output and the full
                // burnup grid are the acceptance lane by definition (plan Sec
                // 6.2), so an operator who has to spell all three out every
                // time will eventually not, and the run that forgets is a
                // screening answer wearing an elite's name.
                const bool promote = (op == "promote");
                if (!parseCase(object,
                               promote ? ResultMode::Full : _options.default_result_mode,
                               request, error)) {
                    refuse(error, line);
                    continue;
                }
                if (promote) {
                    request.promoted = true;
                    if (request.request_fidelity.fidelity.empty())
                        request.request_fidelity.fidelity = "strict";
                    if (!request.request_fidelity.has_grid) {
                        request.request_fidelity.statepoint_grid = "full";
                        request.request_fidelity.has_grid        = true;
                    }
                    // The LINK, and the reason the op exists at all.  Without
                    // it the strict re-run is an unrelated row and nothing in
                    // the receipts answers "was this elite actually re-run".
                    if (request.request_fidelity.promoted_from.empty()) {
                        refuse(R"("op":"promote" needs "promoted_from": the case_key of the )"
                               R"(screening result this run replaces. A promotion with no )"
                               R"(link is two unrelated rows, and the audit cannot then tell )"
                               R"(a promoted elite from a case that merely ran strict.)",
                               line);
                        continue;
                    }
                }
                // WP18.  IN ROLLING MODE A CASE IS ADMITTED WHERE IT IS READ.
                // Wave mode collects `case` lines and runs them when the `wave`
                // line arrives, which is what makes a wave a barrier; here the
                // line goes straight into the queue and a free lane takes it.
                //
                // The fidelity is therefore resolved HERE and not at the wave
                // line, which is the one thing this mode cannot do the wave
                // way: applyWaveFidelityDefault fills a case's unset fields
                // from a declaration that arrives AFTERWARDS, and a case that
                // is already running cannot be told what it should have been.
                // A `wave` line that carries fidelity fields after a case has
                // been admitted is refused by name below.
                if (detail::rollingEnabled()) {
                    if (!resolveCaseFidelity(request.request_fidelity, processCaseFidelity(),
                                             request.resolved_fidelity, error)) {
                        refuse(error + "  (deck: " + request.deck + ")", line);
                        continue;
                    }
                    rollingAdmit(request);
                    continue;
                }
                pending.push_back(std::move(request));
                continue;
            }
            if (op == "wave" || op == "run") {
                WaveRequest wave;
                wave.wave_id = _summary.generations + 1;
                if (object.contains("wave_id") && object["wave_id"].is_number_integer())
                    wave.wave_id = object["wave_id"].get<long long>();
                if (object.contains("jobs_manifest")) {
                    if (!object["jobs_manifest"].is_string()) {
                        refuse(R"("jobs_manifest" must be a string path)", line);
                        continue;
                    }
                    wave.jobs_manifest = object["jobs_manifest"].get<std::string>();
                }
                if (object.contains("batch_width")) {
                    if (!object["batch_width"].is_number_integer() ||
                        object["batch_width"].get<int>() <= 0) {
                        refuse(R"("batch_width" must be a positive integer)", line);
                        continue;
                    }
                    wave.batch_width = object["batch_width"].get<int>();
                }
                // WP10.3.  The wave's fidelity is a DEFAULT for the cases that
                // declared none, resolved per case below.  It is no longer an
                // assertion about the process, because the process no longer
                // has one fidelity to assert.
                FidelityRequest wave_fidelity;
                if (!parseFidelityFields(object, wave_fidelity, error)) {
                    refuse(error, line);
                    pending.clear();
                    continue;
                }
                bool wave_refused = false;
                for (CaseRequest& c : pending) {
                    // WP10.7.  ONE DOOR FOR `promote`, whichever mode this
                    // process is in: `c.promoted` is what rolling mode's
                    // admission implies by resolving before any wave exists.
                    applyWaveFidelityDefault(wave_fidelity, c.request_fidelity, c.promoted);
                    if (!resolveCaseFidelity(c.request_fidelity, processCaseFidelity(),
                                             c.resolved_fidelity, error)) {
                        // ONE case's refusal fails the WAVE, deliberately.  A
                        // wave is a generation; running sixty-three of its
                        // sixty-four cases and dropping the one whose fidelity
                        // could not be honoured is how a GA silently loses a
                        // candidate and never learns it did.
                        refuse(error + "  (deck: " + c.deck + ")", line);
                        wave_refused = true;
                        break;
                    }
                }
                if (wave_refused) {
                    pending.clear();
                    continue;
                }
                // The manifest half of the same wave, resolved once: manifest
                // lines carry no fidelity of their own, so they run at the
                // wave's.
                if (!resolveCaseFidelity(wave_fidelity, processCaseFidelity(),
                                         wave.manifest_fidelity, error)) {
                    refuse(error, line);
                    pending.clear();
                    continue;
                }
                ResultMode wave_mode = _options.default_result_mode;
                if (object.contains("result_mode")) {
                    if (!object["result_mode"].is_string() ||
                        !ParseResultMode(object["result_mode"].get<std::string>(), wave_mode)) {
                        refuse(R"("result_mode" must be one of full | pin-off | light)", line);
                        pending.clear();
                        continue;
                    }
                }
                if (detail::rollingEnabled()) {
                    // A rolling `wave` line is a BARRIER REQUEST, not a run
                    // request: the cases it would have carried are already
                    // running.  What it still does is exactly what the wave
                    // receipt has always meant -- "tell me when this much is
                    // finished" -- so it drains the session, prints the same
                    // receipt tags in the same order, and lets the next case
                    // open a new session on the same process and the same
                    // arena.
                    // A refusal is reported BY NAME inside; either way this
                    // line is done and the stream moves on.
                    (void)rollingWaveLine(wave, wave_fidelity, wave_mode, line);
                    continue;
                }
                wave.cases.swap(pending);
                runWave(wave, wave_mode);
                continue;
            }
            refuse("unknown op \"" + op + "\" (case | wave | run | shutdown)", line);
        }

        // WP18.  A rolling session that the stream ended in the middle of gets
        // the same final barrier a `wave` line would have given it: the lanes
        // drain, the receipts print, and the process receipt below is final.
        // Dropping the queue here would lose whatever the client had sent and
        // not yet been told about, which is the one thing that must not happen.
        if (detail::rollingEnabled() && _roll.running) {
            WaveRequest closing;
            closing.wave_id = _roll.session;
            rollingBarrier(closing, _options.default_result_mode);
        }

        // A stream that ended with cases enqueued and no wave line meant them:
        // running them is what the client asked for and dropping them silently
        // is the one thing that must not happen.
        if (!pending.empty()) {
            WaveRequest tail;
            tail.wave_id = _summary.generations + 1;
            // The same resolution the `wave` branch does, because these cases
            // are about to run and an unresolved CaseFidelity is the process
            // default wearing whatever word the request used.
            bool tail_refused = false;
            for (CaseRequest& c : pending) {
                std::string error;
                if (!resolveCaseFidelity(c.request_fidelity, processCaseFidelity(),
                                         c.resolved_fidelity, error)) {
                    refuse(error + "  (deck: " + c.deck + ")", c.deck);
                    tail_refused = true;
                    break;
                }
            }
            tail.cases.swap(pending);
            if (!tail_refused) runWave(tail, _options.default_result_mode);
        }

        _summary.stop     = reason;
        _summary.uptime_s = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - _t0)
                                .count();
        return _exit_code;
    }

    [[nodiscard]] const Summary& summary() const { return _summary; }

    /// The WP8 process receipt.  Printed by main.cpp after the shared teardown
    /// receipts, so `xslib_loads`/`xslib_hits` are final.
    void reportProcess(std::ostream& out) const {
        const XsLibraryCacheStats xslib  = XsLibraryCacheSnapshot();
        const cohort::Stats       cohorts = cohort::snapshot();
        const auto&               ten    = refill::tenancy();
        out << "[RASBERY][EVALUATOR] {\"cases\":" << _summary.cases
            << ",\"ok\":" << _summary.ok
            << ",\"failed\":" << _summary.failed
            << ",\"refused\":" << _summary.refused
            << ",\"generations\":" << _summary.generations
            << ",\"batch_width\":" << _summary.latched_width
            << std::fixed << std::setprecision(3)
            << ",\"process_uptime_s\":" << _summary.uptime_s
            << ",\"drive_s\":" << _summary.drive_s
            // Cases that began after the process had already stood the CUDA
            // context, the arenas and the device library up -- i.e. everything
            // but the first.  It is derived, and its WITNESS is the pair below
            // it: nothing released the arena between waves (arena_releases is
            // stamped at shutdown, once) and every case admitted a slot.
            << ",\"cuda_context_reuse\":" << (_summary.cases > 0 ? _summary.cases - 1 : 0)
            << ",\"arena_releases\":" << _arena_releases
            << ",\"arena_standups\":" << (ten.admissions.load() > 0 ? 1 : 0)
            << ",\"slot_admissions\":" << ten.admissions.load()
            << ",\"slot_duplicates\":" << ten.queue_duplicates.load()
            << ",\"slot_stale_tenants\":" << ten.stale_tenants.load()
            << ",\"slot_double_releases\":" << ten.double_releases.load()
            << ",\"xslib_loads\":" << xslib.loads
            << ",\"xslib_hits\":" << xslib.hits
            << ",\"xslib_digest_computes\":" << xslib.digest_computes
            << ",\"library_loads\":" << xslib.loads
            // WP8 stage 2 -- THE MIDDLE LIFETIME, and the two numbers that say
            // whether it exists.  `cohort_builds` must equal the number of
            // distinct (geometry, library) pairs this process saw and must NOT
            // grow with the case count: a GA generation of M candidates over
            // one core is builds=1, hits=M-1.  If it is ever builds=M the
            // cohort key has started covering something a candidate changes,
            // and the middle lifetime is gone again without a word from any
            // other number in this receipt.
            << ",\"cohort_builds\":" << cohorts.builds
            << ",\"cohort_hits\":" << cohorts.hits
            << ",\"cohorts\":" << cohorts.cohorts
            << ",\"ppr_quadrature_builds\":" << cohorts.quadrature_builds
            // One CaseContext per case, by construction: the Geometry object
            // itself is still per case (CohortContext.h says exactly why), so
            // this still equals `cases` and is NOT the same claim as
            // `cohort_builds`.  Reporting both is what keeps the difference
            // legible instead of letting one number stand for two facts.
            << ",\"geometry_builds\":" << _summary.cases
            << ",\"pin_live_ranges_between_waves\":"
            << _summary.pin_live_ranges_between_waves
            // WP10.5.  The percentiles are over the RESIDENT WINDOW -- the last
            // `window` cases -- and the window says how many that was, so a
            // reader is never left guessing what the p50 is a p50 OF.  `max` is
            // NOT windowed: it is tracked exactly, over every case the process
            // ran, because the worst teardown a fleet ever paid is a fact about
            // the run and not about the last four thousand cases of it.
            << ",\"case_seconds\":{\"p50\":"
            << detail::percentile(_summary.case_seconds.values(), 0.50)
            << ",\"p90\":" << detail::percentile(_summary.case_seconds.values(), 0.90)
            << ",\"window\":" << _summary.case_seconds.resident()
            << ",\"observed\":" << _summary.case_seconds.observed() << "}"
            << ",\"case_teardown_ms\":{\"p50\":"
            << detail::percentile(_summary.teardown_ms.values(), 0.50)
            << ",\"p90\":" << detail::percentile(_summary.teardown_ms.values(), 0.90)
            << ",\"max\":" << _summary.teardown_ms.max()
            << ",\"window\":" << _summary.teardown_ms.resident()
            << ",\"observed\":" << _summary.teardown_ms.observed() << "}"
            << ",\"fidelity_overrides\":" << _summary.fidelity_overrides
            << ",\"promotions\":" << _summary.promotions
            << ",\"isolation_checks\":" << _summary.isolation_checks
            << ",\"isolation_mismatches\":" << _summary.isolation_mismatches
            << ",\"isolation_adjacent\":" << _summary.isolation_adjacent
            << ",\"capture_race_case_retries\":"
            << _capture_race_case_retries.load(std::memory_order_relaxed)
            << ",\"capture_race_case_recovered\":"
            << _capture_race_case_recovered.load(std::memory_order_relaxed)
            << ",\"stop_reason\":\"" << stopReasonName(_summary.stop) << "\"}"
            << std::endl;
        out.unsetf(std::ios::floatfield);
    }

    /// Called by main.cpp at shutdown, immediately after
    /// rasberyReleaseBatchArena(), so the receipt's witness is a fact.
    void stampArenaRelease() { ++_arena_releases; }

private:
    void refuse(const std::string& why, const std::string& line) {
        // WP18.  A refusal can now be raised by the READER thread while lanes
        // are mid-case, so `_summary.refused`, `_exit_code` and `_out` are all
        // shared.  The lock is uncontended in wave mode (nothing else holds it
        // there), so the line this prints is byte-for-byte what it printed
        // before -- the ordering is what is being bought, not the content.
        std::lock_guard<std::mutex> lock(_out_mutex);
        ++_summary.refused;
        if (_exit_code == 0) _exit_code = 2;
        _out << "[RASBERY][EVALUATOR][REFUSED] {\"what\":" << detail::quoted(why)
             << ",\"line\":" << detail::quoted(line.substr(0, 400)) << "}" << std::endl;
    }

    /// Resolve a wave's jobs, apply the namespace rule, then run them.
    void runWave(const WaveRequest& wave, ResultMode wave_mode) {
        WaveJobs jobs;
        if (!wave.jobs_manifest.empty()) {
            std::string error;
            if (!_options.read_manifest || !_options.read_manifest(wave.jobs_manifest, jobs.inputs,
                                                                  jobs.outputs, jobs.modes,
                                                                  wave_mode, error)) {
                refuse(error.empty() ? "no manifest reader is installed" : error,
                       wave.jobs_manifest);
                return;
            }
            jobs.keys.assign(jobs.inputs.size(), std::string());
            jobs.warm_from.assign(jobs.inputs.size(), std::string());
            jobs.warm_save.assign(jobs.inputs.size(), std::string());
            // A manifest names decks and output paths and says nothing about
            // fidelity, so its jobs run at the WAVE's -- which, with no wave
            // declaration, is the process default and therefore exactly what a
            // manifest wave did before WP10.3.
            jobs.fidelity.assign(jobs.inputs.size(), wave.manifest_fidelity);
            jobs.promoted.assign(jobs.inputs.size(), false);
        }
        for (const CaseRequest& c : wave.cases) {
            jobs.inputs.push_back(c.deck);
            jobs.outputs.push_back(c.output);
            jobs.modes.push_back(c.result_mode);
            jobs.keys.push_back(c.key);
            jobs.warm_from.push_back(c.warm_start_from);
            jobs.warm_save.push_back(c.save_warm_state);
            jobs.fidelity.push_back(c.resolved_fidelity);
            jobs.promoted.push_back(c.promoted);
        }
        jobs.keys.resize(jobs.inputs.size());
        jobs.warm_from.resize(jobs.inputs.size());
        jobs.warm_save.resize(jobs.inputs.size());
        jobs.fidelity.resize(jobs.inputs.size(), processCaseFidelity());
        jobs.promoted.resize(jobs.inputs.size(), false);
        if (jobs.inputs.empty()) {
            refuse("a wave with no jobs (neither \"jobs_manifest\" nor preceding "
                   "\"op\":\"case\" lines)",
                   wave.jobs_manifest);
            return;
        }

        // THE SAME NAMESPACE RULE main.cpp applies to argv and manifest decks,
        // applied here for the same reason: two Drivers on one --raso race
        // inside one HDF5 file and share a restart namespace.  Scoped to the
        // WAVE, not the process: reusing an output path in a LATER generation
        // is legitimate (it is how a GA re-evaluates a promoted elite) and
        // nothing is concurrent across waves.
        for (std::size_t i = 0; i < jobs.outputs.size(); ++i)
            for (std::size_t j = i + 1; j < jobs.outputs.size(); ++j)
                if (jobs.outputs[i] == jobs.outputs[j]) {
                    refuse("--raso paths must be distinct within a wave: \"" + jobs.outputs[i] +
                               "\" appears at entries " + std::to_string(i + 1) + " and " +
                               std::to_string(j + 1),
                           wave.jobs_manifest);
                    return;
                }

        const int requested = wave.batch_width > 0 ? wave.batch_width : _options.batch_width;
        if (_summary.latched_width == 0) {
            _summary.latched_width = std::max(1, requested);
            rasberySetBatchWidth(_summary.latched_width);
            rasberyNodalSetBatchWidth(_summary.latched_width);
            rasberySetHostPinningEnabled(rasberyHostPinningMode() != HostPinningMode::Off);
        } else if (requested != _summary.latched_width) {
            refuse("this process latched batch_width=" + std::to_string(_summary.latched_width) +
                       " on its first wave and the arena is one allocation fixed for the "
                       "process; wave " +
                       std::to_string(wave.wave_id) + " asked for " + std::to_string(requested) +
                       ". Start a second evaluator for the other width (WP8 stage 2 is where a "
                       "second arena shape becomes legal).",
                   wave.jobs_manifest);
            return;
        }

        const int width = _summary.latched_width;
        const int njobs = static_cast<int>(jobs.inputs.size());
        int       host_threads = std::min(width, njobs);
#ifdef _OPENMP
        if (_options.host_threads_override > 0)
            host_threads = std::min({_options.host_threads_override, width, njobs});
        omp_set_num_threads(host_threads);
#else
        host_threads = 1;
#endif

        _out << "[RASBERY][EVALUATOR][WAVE_START] {\"wave_id\":" << wave.wave_id
             << ",\"jobs\":" << njobs << ",\"arena_width\":" << width
             << ",\"host_threads\":" << host_threads
             << ",\"visible_cpus\":" << _options.visible_cpus
             << ",\"result_mode\":\"" << ResultModeName(wave_mode)
             << "\",\"process_reused\":" << (_summary.generations > 0 ? "true" : "false")
             << "}" << std::endl;

        // THE SAME [BATCH_HOST] RECEIPT THE BATCH BRANCH PRINTS (main.cpp:1118),
        // with the same fields and the same meaning -- printed PER WAVE, because
        // a wave is what the batch branch calls a process.
        //
        // WHY IT IS HERE AND NOT ONLY IN WAVE_START.  The launchers' post-run
        // audit (run_single_gpu_batch.check_run_receipts) reads exactly this tag
        // to answer "did the multi-instance batch branch actually run, and with
        // the host_threads that were asked for".  Without it, every wave a
        // dispatcher sends through an evaluator would be audited as "the batch
        // branch never ran", and the honest fix is not to weaken the audit for
        // one mode -- it is for this mode to print the receipt the audit is
        // about.  `legacy_pinning_criterion` follows main.cpp's rule verbatim:
        // workers are recycled onto later decks whenever there are fewer of them
        // than decks.
        {
            const bool host_pinning = rasberyHostPinningMode() != HostPinningMode::Off;
            _out << "[RASBERY][BATCH_HOST] {\"jobs\":" << njobs
                 << ",\"arena_width\":" << width
                 << ",\"host_threads\":" << host_threads
                 << ",\"visible_cpus\":" << _options.visible_cpus
                 << ",\"host_pinning\":" << (host_pinning ? "true" : "false")
                 << ",\"pin_lease\":true"
                 << ",\"legacy_pinning_criterion\":"
                 << (host_threads >= njobs ? "true" : "false")
                 << ",\"wave_id\":" << wave.wave_id << "}" << std::endl;
        }

        const auto wave_start = std::chrono::steady_clock::now();
        refill::ledger().begin(njobs, width, host_threads);

        std::vector<int>         status(static_cast<std::size_t>(njobs), 0);
        std::vector<std::string> failure(static_cast<std::size_t>(njobs));
        std::vector<Driver::CaseReceipt> receipts(static_cast<std::size_t>(njobs));
        std::vector<double>      seconds(static_cast<std::size_t>(njobs), 0.0);
        std::vector<double>      teardown(static_cast<std::size_t>(njobs), 0.0);
        std::vector<int>         lanes(static_cast<std::size_t>(njobs), 0);
        // WP19.1.  ONE FLAG PER LANE, WRITTEN ONLY BY THAT LANE.
        //
        // The stand-up race is first-case-only by construction: the WHILE graph
        // caches and the PPR graph are per-slot and process-lived, so the ONE
        // case that can be building a graph while a sibling stands up is the
        // first case a lane takes.  A `char` per host thread, touched by that
        // thread alone, is therefore the whole bookkeeping -- no atomics, and
        // nothing on the steady path.
        std::vector<char> lane_first(
            static_cast<std::size_t>(std::max(1, host_threads)), 1);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(host_threads)
#endif
        for (int i = 0; i < njobs; ++i) {
#ifdef _OPENMP
            const int lane = omp_get_thread_num();
#else
            const int lane = 0;
#endif
            lanes[static_cast<std::size_t>(i)] = lane;
            refill::ledger().jobStarted(i, lane);
            // WP19.2.  The four facts a SIGSEGV record needs, published before
            // the case runs and cleared by the Scope's destructor.
            const rasbery::crash::Scope _crumb(
                lane, i, jobs.inputs[static_cast<std::size_t>(i)].c_str(),
                rasbery::crash::kPhaseDrive);
            // WP19.2.  The capture-race counter across THIS case.  See
            // captureRaceEvents(): the belt below is no longer first-case-only,
            // because the PPR WHILE's graph cache dies with the Driver and its
            // build window therefore opens on every case, not on a lane's first.
            const unsigned long long races_before = rasbery::captureRaceEvents();
            runOneCase(jobs.inputs[static_cast<std::size_t>(i)],
                       jobs.outputs[static_cast<std::size_t>(i)],
                       jobs.modes[static_cast<std::size_t>(i)],
                       jobs.warm_from[static_cast<std::size_t>(i)],
                       jobs.warm_save[static_cast<std::size_t>(i)],
                       jobs.fidelity[static_cast<std::size_t>(i)],
                       status[static_cast<std::size_t>(i)],
                       failure[static_cast<std::size_t>(i)],
                       receipts[static_cast<std::size_t>(i)],
                       seconds[static_cast<std::size_t>(i)],
                       teardown[static_cast<std::size_t>(i)]);
            const std::size_t li = static_cast<std::size_t>(lane) < lane_first.size()
                                       ? static_cast<std::size_t>(lane)
                                       : 0u;
            const bool lane_first_case = lane_first[li] != 0;
            lane_first[li] = 0;
            const bool race_spanned =
                rasbery::captureRaceEvents() != races_before;
            _crumb.phase(rasbery::crash::kPhaseRetry);
            retryAfterCaptureRace(wave.wave_id, i, lane, lane_first_case, race_spanned,
                                  jobs.inputs[static_cast<std::size_t>(i)],
                                  jobs.outputs[static_cast<std::size_t>(i)],
                                  jobs.modes[static_cast<std::size_t>(i)],
                                  jobs.warm_from[static_cast<std::size_t>(i)],
                                  jobs.warm_save[static_cast<std::size_t>(i)],
                                  jobs.fidelity[static_cast<std::size_t>(i)],
                                  status[static_cast<std::size_t>(i)],
                                  failure[static_cast<std::size_t>(i)],
                                  receipts[static_cast<std::size_t>(i)],
                                  seconds[static_cast<std::size_t>(i)],
                                  teardown[static_cast<std::size_t>(i)]);
            refill::ledger().jobFinished(i);
        }
        refill::ledger().end();

        long long ok = 0;
        long long failed = 0;
        for (int i = 0; i < njobs; ++i) {
            const auto u = static_cast<std::size_t>(i);
            if (status[u] == 0) ++ok; else ++failed;
            reportCase(wave.wave_id, i, jobs.keys[u], jobs.inputs[u], jobs.outputs[u],
                       jobs.modes[u], status[u], failure[u], receipts[u], seconds[u],
                       teardown[u], lanes[u], false);
            _summary.case_seconds.push(seconds[u]);
            _summary.teardown_ms.push(teardown[u]);
        }
        _summary.cases  += njobs;
        _summary.ok     += ok;
        _summary.failed += failed;
        {
            const std::string default_policy = processCaseFidelity().policy();
            const std::string default_grid   = processCaseFidelity().gridToken();
            for (int i = 0; i < njobs; ++i) {
                const auto u = static_cast<std::size_t>(i);
                if (jobs.fidelity[u].policy() != default_policy ||
                    jobs.fidelity[u].gridToken() != default_grid)
                    ++_summary.fidelity_overrides;
                if (jobs.promoted[u]) ++_summary.promotions;
            }
        }
        if (failed > 0 && _exit_code == 0) _exit_code = 1;

        // ---------------------------------------------------------------
        // The runtime isolation check (plan WP8 "cross-case isolation", case 1)
        // ---------------------------------------------------------------
        //
        // A -> ... -> A, in ONE process, with the second A separated from the
        // first by every other case in the wave.  If a case wrote a value into
        // process-lifetime or thread_local state that a later case reads, the
        // two digests differ -- and the digest is a fold of (step, outers,
        // th_steps, efpd, keff, ppm) at every statepoint, so it moves on a
        // change far below what any printed field would show.
        //
        // NOT ADJACENT is the whole point.  Running the same deck twice back to
        // back would mostly re-exercise ONE worker's thread_local buffers; the
        // first-then-last ordering puts the second run on whichever lane the
        // queue gives it, after every other deck has been through.
        bool isolation_mismatch = false;
        if (_options.isolation_check && njobs > 0 && status[0] == 0) {
            const auto u0 = static_cast<std::size_t>(0);
            const std::string recheck_output =
                detail::isolationOutput(jobs.outputs[u0], wave.wave_id);
            int                  recheck_status = 0;
            std::string          recheck_error;
            Driver::CaseReceipt  recheck;
            double               recheck_seconds  = 0.0;
            double               recheck_teardown = 0.0;
            // The isolation recheck is a REPEAT of the first deck, and it must
            // repeat the same case: the same warm start in, and NO warm state
            // out -- writing one would overwrite the parent state the wave's own
            // first case just produced.
            // The SAME fidelity, necessarily: an isolation check that re-ran the
            // deck under a different convergence policy would compare two
            // digests that had no reason to agree and would report every wave
            // as leaking.
            runOneCase(jobs.inputs[u0], recheck_output, jobs.modes[u0], jobs.warm_from[u0],
                       std::string(), jobs.fidelity[u0], recheck_status,
                       recheck_error, recheck, recheck_seconds, recheck_teardown);
            ++_summary.isolation_checks;
            if (njobs < 2) ++_summary.isolation_adjacent;
            isolation_mismatch = recheck_status != 0 ||
                                 !recheck.complete || !receipts[u0].complete ||
                                 recheck.digest != receipts[u0].digest;
            if (isolation_mismatch) {
                ++_summary.isolation_mismatches;
                if (_exit_code == 0) _exit_code = 1;
            }
            reportCase(wave.wave_id, -1, jobs.keys[u0], jobs.inputs[u0], recheck_output,
                       jobs.modes[u0], recheck_status, recheck_error, recheck, recheck_seconds,
                       recheck_teardown, -1, true);
            _out << "[RASBERY][EVALUATOR][ISOLATION] {\"wave_id\":" << wave.wave_id
                 << ",\"deck\":" << detail::quoted(jobs.inputs[u0])
                 << ",\"cases_between\":" << (njobs - 1)
                 << ",\"adjacent\":" << (njobs < 2 ? "true" : "false")
                 << ",\"digest_first\":\"" << std::hex << std::setw(16) << std::setfill('0')
                 << receipts[u0].digest << std::dec << std::setfill(' ')
                 << "\",\"digest_recheck\":\"" << std::hex << std::setw(16)
                 << std::setfill('0') << recheck.digest << std::dec << std::setfill(' ')
                 << "\",\"match\":" << (isolation_mismatch ? "false" : "true") << "}"
                 << std::endl;
        }

        // Between-wave assertions.  See the header comment: these are the two
        // teardown steps this mode DEFERS, so the state they would have reset
        // has to be provably empty instead.
        iowriter::flushLines();
        const std::uint64_t live_ranges =
            static_cast<std::uint64_t>(rasberyHostPinLiveRanges());
        _summary.pin_live_ranges_between_waves =
            std::max(_summary.pin_live_ranges_between_waves, live_ranges);
        if (live_ranges > 0 && _exit_code == 0) _exit_code = 1;

        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wave_start).count();
        _summary.drive_s += wall;
        ++_summary.generations;

        refill::ledger().report(_out);
        const XsLibraryCacheStats xslib = XsLibraryCacheSnapshot();
        _out << "[RASBERY][EVALUATOR][WAVE] {\"wave_id\":" << wave.wave_id
             << ",\"jobs\":" << njobs << ",\"ok\":" << ok << ",\"failed\":" << failed
             << std::fixed << std::setprecision(3) << ",\"wall_s\":" << wall
             << ",\"cases_per_hour\":"
             << (wall > 0.0 ? 3600.0 * static_cast<double>(njobs) / wall : 0.0)
             << ",\"process_reused\":" << (_summary.generations > 1 ? "true" : "false")
             << ",\"xslib_loads\":" << xslib.loads << ",\"xslib_hits\":" << xslib.hits
             << ",\"pin_live_ranges\":" << live_ranges
             << ",\"isolation_match\":"
             << (_options.isolation_check ? (isolation_mismatch ? "false" : "true") : "null")
             << "}" << std::endl;
        _out.unsetf(std::ios::floatfield);
        reportMemory(wave.wave_id);
    }

    /// WP10.4.  One line per generation: what the process weighs, and the size
    /// of every container that could explain a change.  See the note beside
    /// detail::residentBytes().
    ///
    /// PRINTED BETWEEN GENERATIONS, NOT DURING ONE, for the same reason
    /// tools/soak_run.py samples RSS between waves: a mid-wave sample measures
    /// where the wave happened to be, and the question is what is LEFT BEHIND
    /// when it is over.
    void reportMemory(long long wave_id) {
        const std::uint64_t rss   = detail::residentBytes();
        const std::uint64_t peak  = detail::peakResidentBytes();
        const std::uint64_t prev  = _summary.rss_bytes_last;
        if (_summary.rss_bytes_first == 0) _summary.rss_bytes_first = rss;
        _summary.rss_bytes_last = rss;

        // WP10.6.  DEVICE bytes, measured from INSIDE the process.
        //
        // The 238 soak's VRAM trace sawtoothed between 298 MB and 47,000 MB and
        // neither number was this process: `nvidia-smi` answers for a BOARD,
        // and that board had eight other tenants on it.  A footprint the
        // process reports for itself cannot be contaminated by a neighbour, and
        // it is the only VRAM number a 10k-generation gate can be built on.
        const rasbery::gpu::blockpool::Stats dev  = rasbery::gpu::blockpool::snapshot();
        const std::uint64_t dev_bytes = dev.bytes_live + dev.bytes_pooled;
        const std::uint64_t dev_prev  = _summary.device_bytes_last;
        _summary.device_bytes_last = dev_bytes;
        const detail::MallocRetention mall = detail::mallocRetention();

        const XsLibraryCacheStats xslib   = XsLibraryCacheSnapshot();
        const cohort::Stats       cohorts = cohort::snapshot();
        const long long           live    = detail::liveCases().load(std::memory_order_relaxed);

        _out << std::fixed << std::setprecision(3)
             << "[RASBERY][EVALUATOR][MEM] {\"wave_id\":" << wave_id
             << ",\"rss_mb\":" << detail::toMiB(rss)
             // Signed on purpose: a generation that gives memory BACK is a fact
             // an unsigned delta would hide, and it is the fact that separates
             // allocator retention from a container that never shrinks.
             << ",\"rss_delta_mb\":"
             << (rss == 0 || prev == 0
                     ? 0.0
                     : detail::toMiB(rss) - detail::toMiB(prev))
             << ",\"rss_since_first_mb\":"
             << (rss == 0 || _summary.rss_bytes_first == 0
                     ? 0.0
                     : detail::toMiB(rss) - detail::toMiB(_summary.rss_bytes_first))
             << ",\"rss_peak_mb\":" << detail::toMiB(peak)
             << ",\"rss_readable\":" << (rss > 0 ? "true" : "false")
             // WP10.6.  THE PROCESS'S OWN DEVICE FOOTPRINT, and the four
             // counters that say what moved it.  `vram_mb` is in-use plus
             // parked -- what this process is holding on the device, not what
             // the board is holding for everybody -- and `vram_delta_mb` is
             // signed for the same reason `rss_delta_mb` is: a generation that
             // gives memory back is a fact an unsigned delta would hide.
             << ",\"vram_mb\":" << detail::toMiB(dev_bytes)
             << ",\"vram_delta_mb\":"
             << (dev_prev == 0 && dev_bytes == 0
                     ? 0.0
                     : detail::toMiB(dev_bytes) - detail::toMiB(dev_prev))
             << ",\"vram_live_mb\":" << detail::toMiB(dev.bytes_live)
             << ",\"vram_pooled_mb\":" << detail::toMiB(dev.bytes_pooled)
             << ",\"vram_high_water_mb\":" << detail::toMiB(dev.bytes_high_water)
             // cudaMalloc / cudaFree calls that reached the DRIVER, cumulative.
             // A steady state that still allocates is a steady state paying a
             // synchronising API call per case for a block it already owns.
             << ",\"device_allocs\":" << dev.device_allocs
             << ",\"device_frees\":" << dev.device_frees
             << ",\"device_pool_hits\":" << dev.pool_hits
             << ",\"device_pool_parks\":" << dev.pool_parks
             << ",\"device_blocks_live\":" << dev.blocks_live
             << ",\"device_blocks_pooled\":" << dev.blocks_pooled
             // MUST be 0 in steady state, AND NOW MEANS THAT.
             //
             // WP10.8.  Until this build the field carried
             // `blockpool::Stats::arena_rebuilds`, which three sites in
             // CudaXsReconBackend.cu and one in GpuPhysicsArenaCuda.cu
             // incremented -- and the 238 block-38 soak read +17 per generation
             // in BOTH arms, i.e. one per CASE, which is a per-instance device
             // block being re-laid-out at its first real geometry and not an
             // arena teardown at all.  The report that read it concluded
             // `RASBERY_ARENA_PERSIST=1` was failing to prevent rebuilds; the
             // flag never claimed to prevent THOSE.  The quantity is still
             // printed, under the name of the thing it counts
             // (`block_reshapes`), and this field now carries the number its
             // own name always promised: process-lifetime regions handed back.
             // Between generations that is 0, and a nonzero one IS the arena
             // teardown the VRAM sawtooth raised.
             << ",\"arena_rebuilds\":" << rasbery::gpu::blockpool::arenaRebuilds(dev)
             << ",\"arena_standups\":" << dev.arena_standups
             << ",\"block_reshapes\":" << dev.block_reshapes
             << ",\"arena_persist\":"
             << (rasbery::gpu::blockpool::enabled() ? "true" : "false")
             // WP10.8.  THE FREE LIST IS BOUNDED, AND THIS IS THE PROOF.
             // `pool_size_classes` climbing generation on generation is a size
             // key that carries something per case; `pool_evictions` /
             // `pool_park_refusals` are the cap doing its job, and both being 0
             // with `vram_pooled_mb` flat is a pool inside its budget with no
             // policy pressure at all.  `pool_bookkeeping_bytes` weighs the
             // pool's OWN host containers, so "the pool is the RSS growth" is
             // an arithmetic claim a reader can check rather than a suspicion:
             // the 238 arm-B finding was 115.97 MB/generation and nothing in
             // that report could weigh the accused.
             << ",\"pool_cap_mb\":"
             << detail::toMiB(rasbery::gpu::blockpool::capBytes())
             << ",\"pool_cap_blocks\":" << rasbery::gpu::blockpool::capBlocks()
             << ",\"pool_class_depth\":" << rasbery::gpu::blockpool::capClassDepth()
             << ",\"pool_size_classes\":" << dev.size_classes
             << ",\"pool_evictions\":" << dev.pool_evictions
             << ",\"pool_evicted_mb\":" << detail::toMiB(dev.bytes_evicted)
             << ",\"pool_park_refusals\":" << dev.park_refusals
             << ",\"pool_bookkeeping_bytes\":" << dev.bookkeeping_bytes
             << ",\"pool_reclaimer\":"
             << (rasbery::gpu::blockpool::hasReclaimer() ? "true" : "false")
             // WP10.6.  The allocator, asked directly.  See mallocRetention().
             << ",\"malloc_taken_mb\":" << detail::toMiB(mall.taken_bytes)
             << ",\"malloc_in_use_mb\":" << detail::toMiB(mall.in_use_bytes)
             << ",\"malloc_retained_mb\":"
             << detail::toMiB(mall.taken_bytes > mall.in_use_bytes
                                  ? mall.taken_bytes - mall.in_use_bytes
                                  : 0)
             << ",\"malloc_readable\":" << (mall.readable ? "true" : "false")
             // MUST be 0.  A Driver that outlived its case leaks everything a
             // case owns and nothing else in any receipt would say so.
             << ",\"live_cases\":" << live
             << ",\"cache_entries\":{"
             << "\"xslib\":" << xslib.entries
             << ",\"xslib_digest\":" << xslib.digest_entries
             << ",\"cohorts\":" << cohorts.cohorts
             << ",\"quadratures\":" << cohorts.quadrature_builds
             << ",\"pin_records\":" << rasberyHostPinLiveRanges()
             << ",\"digest_memo\":" << BatchLightResult::HashCacheEntries()
             // The two per-case sample series this receipt itself keeps.  They
             // WERE bounded by the case count and by nothing else; WP10.5 caps
             // them at `case_samples_cap` after the 55c0dce soak reported them
             // as the only growing container on a run that grew by 3.3 GB.
             << ",\"case_samples\":" << _summary.case_seconds.resident()
             << ",\"case_samples_cap\":" << detail::sampleWindowCapacity()
             << "}"
             // WP10.5.  BYTES BESIDE COUNTS, for every container that has a
             // cheap answer.  The 55c0dce soak's attribution named
             // `case_samples` because it was the only count that moved -- and
             // it moved by 72 doubles while RSS moved by 3.3 GB.  A mover a
             // reader cannot weigh is a mover a reader will believe.
             << ",\"cache_bytes\":{\"xslib\":" << xslib.bytes
             << ",\"case_samples\":"
             << (_summary.case_seconds.bytes() + _summary.teardown_ms.bytes()) << "}"
             << ",\"evictions\":{\"xslib\":" << xslib.evictions
             << ",\"xslib_digest\":" << xslib.digest_evictions
             << ",\"cohort\":" << cohorts.evictions
             << ",\"digest_memo_clears\":" << BatchLightResult::HashCacheClears()
             // WP10.8.  The device free list evicts too, and an eviction count
             // that lived only beside the VRAM fields would be the one cache in
             // this receipt whose evictions a reader had to look somewhere else
             // for.
             << ",\"blockpool\":" << dev.pool_evictions
             << ",\"blockpool_refusals\":" << dev.park_refusals
             << "}"
             // Bytes page-locked through HostPinRegistry right now.  The arena's
             // own cudaMallocHost blocks are stood up once for the process and
             // are NOT here: this is the number that moves with cases.
             << ",\"cuda_host_bytes\":" << rasberyHostPinLiveBytes()
             << "}" << std::endl;
        _out.unsetf(std::ios::floatfield);
    }

    // -----------------------------------------------------------------------
    // WP18  Rolling admission
    // -----------------------------------------------------------------------

    /// Everything one rolling session owns.  A session is opened by the first
    /// case admitted after a barrier and closed by the next barrier; the arena,
    /// the CUDA context, the library and the cohort cache all outlive it, which
    /// is what WP8 already established and this mode does not touch.
    struct RollingState {
        bool               running = false;
        long long          session = 0;
        int                width   = 0;
        int                lanes   = 0;
        int                capacity = 0;
        int                jobs    = 0;   ///< admitted this session
        long long          ok      = 0;
        long long          failed  = 0;
        std::thread        pool;
        std::chrono::steady_clock::time_point t0{};
        /// The session's FIRST case, kept for the isolation recheck -- the same
        /// A -> ... -> A the wave path runs, with the same "not adjacent" point.
        bool                first_seen   = false;
        int                 first_status = 0;
        Driver::CaseReceipt first_receipt;
        RollingJob          first_job;
    };

    /// Admit one case: open a session if none is running, then queue it.
    void rollingAdmit(CaseRequest& request) {
        if (!_roll.running) rollingOpen();
        else refill::rollingLedger().noteBarrierAvoided();

        RollingJob job;
        job.deck      = request.deck;
        job.output    = request.output;
        job.key       = request.key;
        job.warm_from = request.warm_start_from;
        job.warm_save = request.save_warm_state;
        job.mode      = request.result_mode;
        job.fidelity  = request.resolved_fidelity;
        job.promoted  = request.promoted;
        job.index     = _roll.jobs++;
        // Defensive: the queue is only ever closed by a barrier, and a barrier
        // runs on THIS thread, so a closed queue here would mean the reader and
        // the barrier had come apart.  Refuse loudly rather than drop a case.
        const std::string dropped_deck   = job.deck;
        const std::string dropped_output = job.output;
        if (!_queue.push(std::move(job))) {
            --_roll.jobs;
            refuse("the rolling queue was closed while admitting \"" + dropped_deck +
                       "\" -> \"" + dropped_output +
                       "\"; the case was NOT run. (rolling_queue_closed)",
                   dropped_deck);
        }
    }

    /// Stand a session up: latch the width exactly as runWave does, open the
    /// ledger, and start the lanes.
    ///
    /// THE LANES RUN ON A HELPER THREAD, and the reason is structural: the
    /// OpenMP team's master blocks until the region ends, and the thread that
    /// has to keep reading the request stream is this one.  So the pool thread
    /// opens `#pragma omp parallel` and this thread goes back to the stream.
    /// The lanes are OpenMP threads and not std::threads deliberately -- the
    /// campaign's whole environment (OMP_PROC_BIND, RASBERY_OMP_THREADS, the
    /// taskset cpu-list) places OpenMP threads, and a pool that opted out of
    /// that would be measuring a different machine.
    void rollingOpen() {
        if (_summary.latched_width == 0) {
            _summary.latched_width = std::max(1, _options.batch_width);
            rasberySetBatchWidth(_summary.latched_width);
            rasberyNodalSetBatchWidth(_summary.latched_width);
            rasberySetHostPinningEnabled(rasberyHostPinningMode() != HostPinningMode::Off);
        }
        _roll.width = _summary.latched_width;
        // No `njobs` term, and that is the difference: runWave caps the lanes
        // at the size of the chunk it was handed, which is precisely why a
        // claim of 1 collapsed the arena to one lane.  A rolling session does
        // not know how many cases it will see, so it stands the full width up
        // and lets the queue decide how much of it is busy.
        _roll.lanes = _roll.width;
#ifdef _OPENMP
        if (_options.host_threads_override > 0)
            _roll.lanes = std::min(_options.host_threads_override, _roll.width);
        omp_set_num_threads(_roll.lanes);
#else
        _roll.lanes = 1;
#endif
        _roll.capacity = detail::rollingQueueCapacity(_roll.lanes);
        _roll.session  = _summary.generations + 1;
        _roll.jobs     = 0;
        _roll.ok       = 0;
        _roll.failed   = 0;
        _roll.first_seen = false;
        _roll.t0       = std::chrono::steady_clock::now();
        _queue.configure(static_cast<std::size_t>(_roll.capacity));
        refill::rollingLedger().open(_roll.lanes, _roll.width, _roll.capacity);
        _out << "[RASBERY][EVALUATOR][ROLLING_START] {\"session\":" << _roll.session
             << ",\"arena_width\":" << _roll.width << ",\"host_threads\":" << _roll.lanes
             << ",\"queue_capacity\":" << _roll.capacity
             << ",\"prefetch\":" << detail::rollingPrefetch()
             << ",\"visible_cpus\":" << _options.visible_cpus
             << ",\"process_reused\":" << (_summary.generations > 0 ? "true" : "false")
             << "}" << std::endl;
        _roll.running = true;
        _roll.pool    = std::thread(&Server::rollingPool, this, _roll.lanes);
    }

    /// The team, on its own thread.  A plain member function and NOT a lambda:
    /// an OpenMP directive inside a lambda body is a compiler-support question
    /// this tree has no reason to ask, and the answer differs between the nvcc
    /// build and the MSVC stub build that also has to compile this header.
    void rollingPool(int lanes) {
#ifdef _OPENMP
    #pragma omp parallel num_threads(lanes)
        { rollingLane(); }
#else
        (void)lanes;
        rollingLane();
#endif
    }

    /// One lane.  Pop, admit, run, release, pop again -- and no barrier
    /// anywhere in it.  This IS the per-slot refill: `runOneCase` scopes the
    /// Driver, so the arena slot is released, whole-struct reset and audited
    /// (CudaBICGBackend.cu acquireSlot) between one tenancy and the next,
    /// through exactly the door the wave path uses.
    void rollingLane() {
#ifdef _OPENMP
        const int lane = omp_get_thread_num();
#else
        const int lane = 0;
#endif
        RollingJob job;
        bool       immediate = true;
        double     waited_ms = 0.0;
        // WP19.1: the same first-case rule as the wave path, and here it needs
        // no vector at all -- the lane IS this call frame.
        bool       lane_first = true;
        while (_queue.pop(job, immediate, waited_ms)) {
            const auto epoch = refill::rollingLedger().admit(
                lane, immediate, waited_ms, static_cast<int>(_queue.inflight()));
            int                 status   = 0;
            std::string         failure;
            Driver::CaseReceipt receipt;
            double              seconds  = 0.0;
            double              teardown = 0.0;
            // WP19.2: the same two lines as the wave path, for the same two
            // reasons -- a crash record that names the case, and a capture-race
            // delta that is not a guess about which case can be corrupted.
            const rasbery::crash::Scope _crumb(lane, job.index, job.deck.c_str(),
                                               rasbery::crash::kPhaseDrive);
            const unsigned long long    races_before = rasbery::captureRaceEvents();
            runOneCase(job.deck, job.output, job.mode, job.warm_from, job.warm_save,
                       job.fidelity, status, failure, receipt, seconds, teardown);
            const bool lane_first_case = lane_first;
            lane_first                 = false;
            const bool race_spanned    = rasbery::captureRaceEvents() != races_before;
            _crumb.phase(rasbery::crash::kPhaseRetry);
            retryAfterCaptureRace(_roll.session, job.index, lane, lane_first_case,
                                  race_spanned,
                                  job.deck, job.output, job.mode, job.warm_from,
                                  job.warm_save, job.fidelity, status, failure, receipt,
                                  seconds, teardown);
            // Order matters and is the tenancy rule: the ledger is told the
            // tenancy ended BEFORE the output is released, so no other lane can
            // take the output while this one still counts as holding it.
            refill::rollingLedger().finish(lane, epoch, static_cast<int>(_queue.inflight()));
            _queue.finish(job.output);
            {
                // `_out` is one stream and the lanes are many; a receipt is
                // built into a string and written under this lock so two cases
                // finishing together cannot interleave a line.
                std::lock_guard<std::mutex> lock(_out_mutex);
                reportCase(_roll.session, job.index, job.key, job.deck, job.output, job.mode,
                           status, failure, receipt, seconds, teardown, lane, false);
                if (status == 0) ++_roll.ok; else ++_roll.failed;
                ++_summary.cases;
                if (status == 0) ++_summary.ok; else ++_summary.failed;
                _summary.case_seconds.push(seconds);
                _summary.teardown_ms.push(teardown);
                if (job.fidelity.policy() != processCaseFidelity().policy() ||
                    job.fidelity.gridToken() != processCaseFidelity().gridToken())
                    ++_summary.fidelity_overrides;
                if (job.promoted) ++_summary.promotions;
                if (status != 0 && _exit_code == 0) _exit_code = 1;
                // THE FIRST ADMITTED case, not the first to finish: the
                // isolation recheck's whole claim is "after every other case in
                // the session has been through", and the earliest case is the
                // one that has the most of them behind it.  The lanes start
                // together, so whichever finishes first is an accident of deck
                // length and would make the check adjacent by luck.
                if (job.index == 0 && !_roll.first_seen) {
                    _roll.first_seen   = true;
                    _roll.first_status = status;
                    _roll.first_receipt = receipt;
                    _roll.first_job     = job;
                }
            }
        }
    }

    /// A `wave`/`run` line while rolling.  Returns false when it was refused.
    bool rollingWaveLine(const WaveRequest& wave, const FidelityRequest& wave_fidelity,
                         ResultMode wave_mode, const std::string& line) {
        // The width refusal is the wave path's, verbatim in meaning: the arena
        // is one allocation fixed at the first admission.
        const int requested = wave.batch_width > 0 ? wave.batch_width : _options.batch_width;
        if (_summary.latched_width != 0 && requested != _summary.latched_width) {
            refuse("this process latched batch_width=" + std::to_string(_summary.latched_width) +
                       " and the arena is one allocation fixed for the process; wave " +
                       std::to_string(wave.wave_id) + " asked for " + std::to_string(requested) +
                       ". (rolling_batch_width_latched)",
                   line);
            return false;
        }
        // A fidelity DEFAULT cannot be applied to a case that is already
        // running.  Wave mode fills a case's unset fields from the wave line
        // that arrives after it; rolling mode resolved them at the `case` line
        // because the case started there.  Silently ignoring the declaration
        // would be a screening wave whose cases ran at the process default --
        // exactly the class WP10.3's equality check exists for.
        if (_roll.running && _roll.jobs > 0 && !wave_fidelity.empty()) {
            refuse("a rolling session resolves each case's fidelity when the case line is "
                   "read, because the case starts there; wave " + std::to_string(wave.wave_id) +
                   " declares fidelity fields after " + std::to_string(_roll.jobs) +
                   " case(s) were already admitted, and a declaration cannot be applied "
                   "retroactively. Declare fidelity on the case lines, or send the wave line "
                   "first. (rolling_wave_fidelity_after_admit)",
                   line);
            return false;
        }
        // A manifest still expands here, and its --raso namespace rule is the
        // wave's, checked before a single job is admitted.
        if (!wave.jobs_manifest.empty()) {
            WaveJobs    jobs;
            std::string error;
            if (!_options.read_manifest ||
                !_options.read_manifest(wave.jobs_manifest, jobs.inputs, jobs.outputs, jobs.modes,
                                        wave_mode, error)) {
                refuse(error.empty() ? "no manifest reader is installed" : error,
                       wave.jobs_manifest);
                return false;
            }
            for (std::size_t i = 0; i < jobs.outputs.size(); ++i)
                for (std::size_t j = i + 1; j < jobs.outputs.size(); ++j)
                    if (jobs.outputs[i] == jobs.outputs[j]) {
                        refuse("--raso paths must be distinct within a manifest: \"" +
                                   jobs.outputs[i] + "\" appears at entries " +
                                   std::to_string(i + 1) + " and " + std::to_string(j + 1),
                               wave.jobs_manifest);
                        return false;
                    }
            CaseFidelity manifest_fidelity = processCaseFidelity();
            if (!resolveCaseFidelity(wave_fidelity, processCaseFidelity(), manifest_fidelity,
                                     error)) {
                refuse(error, line);
                return false;
            }
            for (std::size_t i = 0; i < jobs.inputs.size(); ++i) {
                CaseRequest request;
                request.deck              = jobs.inputs[i];
                request.output            = jobs.outputs[i];
                request.result_mode       = jobs.modes[i];
                request.resolved_fidelity = manifest_fidelity;
                rollingAdmit(request);
            }
        }
        if (!_roll.running) {
            refuse("a wave with no jobs (neither \"jobs_manifest\" nor preceding "
                   "\"op\":\"case\" lines)",
                   wave.jobs_manifest);
            return false;
        }
        rollingBarrier(wave, wave_mode);
        return true;
    }

    /// Drain the session and print the wave-shaped receipts.
    ///
    /// SAME TAGS, SAME ORDER as runWave's tail, and that is not cosmetic: the
    /// dispatcher's post-run audit (run_single_gpu_batch.check_run_receipts)
    /// reads [RASBERY][BATCH_HOST] to answer "did the multi-instance batch
    /// branch actually run", and it pumps the child until the
    /// [EVALUATOR][WAVE] receipt with the wave id it asked for.  A mode that
    /// printed different tags would be audited as a run that never happened.
    void rollingBarrier(const WaveRequest& wave, ResultMode wave_mode) {
        _queue.close();
        if (_roll.pool.joinable()) _roll.pool.join();
        _roll.running = false;
        refill::rollingLedger().close();
        _queue.reopen();

        const int njobs = _roll.jobs;
        {
            const bool host_pinning = rasberyHostPinningMode() != HostPinningMode::Off;
            _out << "[RASBERY][BATCH_HOST] {\"jobs\":" << njobs
                 << ",\"arena_width\":" << _roll.width
                 << ",\"host_threads\":" << _roll.lanes
                 << ",\"visible_cpus\":" << _options.visible_cpus
                 << ",\"host_pinning\":" << (host_pinning ? "true" : "false")
                 << ",\"pin_lease\":true"
                 << ",\"legacy_pinning_criterion\":"
                 << (_roll.lanes >= njobs ? "true" : "false")
                 << ",\"wave_id\":" << wave.wave_id << "}" << std::endl;
        }

        // The same A -> ... -> A the wave path runs, for the same reason and
        // with the same comparand: the session's FIRST case, re-run after every
        // other case in the session has been through the lanes.  Run here, on
        // the reader thread, with every lane already joined -- so it is the one
        // case in the session that is not concurrent with anything, which is
        // what makes a digest mismatch a statement about carried state.
        bool isolation_mismatch = false;
        if (_options.isolation_check && _roll.first_seen && _roll.first_status == 0) {
            const RollingJob& f = _roll.first_job;
            const std::string recheck_output = detail::isolationOutput(f.output, wave.wave_id);
            int                 recheck_status = 0;
            std::string         recheck_error;
            Driver::CaseReceipt recheck;
            double              recheck_seconds  = 0.0;
            double              recheck_teardown = 0.0;
            runOneCase(f.deck, recheck_output, f.mode, f.warm_from, std::string(), f.fidelity,
                       recheck_status, recheck_error, recheck, recheck_seconds,
                       recheck_teardown);
            ++_summary.isolation_checks;
            if (njobs < 2) ++_summary.isolation_adjacent;
            isolation_mismatch = recheck_status != 0 || !recheck.complete ||
                                 !_roll.first_receipt.complete ||
                                 recheck.digest != _roll.first_receipt.digest;
            if (isolation_mismatch) {
                ++_summary.isolation_mismatches;
                if (_exit_code == 0) _exit_code = 1;
            }
            reportCase(wave.wave_id, -1, f.key, f.deck, recheck_output, f.mode, recheck_status,
                       recheck_error, recheck, recheck_seconds, recheck_teardown, -1, true);
            _out << "[RASBERY][EVALUATOR][ISOLATION] {\"wave_id\":" << wave.wave_id
                 << ",\"deck\":" << detail::quoted(f.deck)
                 << ",\"cases_between\":" << (njobs - 1)
                 << ",\"adjacent\":" << (njobs < 2 ? "true" : "false")
                 << ",\"digest_first\":\"" << std::hex << std::setw(16) << std::setfill('0')
                 << _roll.first_receipt.digest << std::dec << std::setfill(' ')
                 << "\",\"digest_recheck\":\"" << std::hex << std::setw(16) << std::setfill('0')
                 << recheck.digest << std::dec << std::setfill(' ')
                 << "\",\"match\":" << (isolation_mismatch ? "false" : "true") << "}"
                 << std::endl;
        }

        // The two deferred teardown steps, asserted rather than run -- the
        // between-wave contract of WP8 stage 1, unchanged.
        iowriter::flushLines();
        const std::uint64_t live_ranges =
            static_cast<std::uint64_t>(rasberyHostPinLiveRanges());
        _summary.pin_live_ranges_between_waves =
            std::max(_summary.pin_live_ranges_between_waves, live_ranges);
        if (live_ranges > 0 && _exit_code == 0) _exit_code = 1;

        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - _roll.t0).count();
        _summary.drive_s += wall;
        ++_summary.generations;

        refill::rollingLedger().report(_out, _roll.session);
        const XsLibraryCacheStats xslib = XsLibraryCacheSnapshot();
        _out << "[RASBERY][EVALUATOR][WAVE] {\"wave_id\":" << wave.wave_id
             << ",\"jobs\":" << njobs << ",\"ok\":" << _roll.ok << ",\"failed\":" << _roll.failed
             << std::fixed << std::setprecision(3) << ",\"wall_s\":" << wall
             << ",\"cases_per_hour\":"
             << (wall > 0.0 ? 3600.0 * static_cast<double>(njobs) / wall : 0.0)
             << ",\"process_reused\":" << (_summary.generations > 1 ? "true" : "false")
             << ",\"xslib_loads\":" << xslib.loads << ",\"xslib_hits\":" << xslib.hits
             << ",\"pin_live_ranges\":" << live_ranges
             << ",\"isolation_match\":"
             << (_options.isolation_check ? (isolation_mismatch ? "false" : "true") : "null")
             << ",\"rolling\":true}" << std::endl;
        _out.unsetf(std::ios::floatfield);
        (void)wave_mode;
        reportMemory(wave.wave_id);
    }

    /// WP19.1.  THE LOUD PATH FOR THE STAND-UP RACE'S SILENT FACE.
    ///
    /// WHAT IT IS FOR.  WP19 closed the race that announced itself
    /// (cudaErrorStreamCapture*, "slot":-1, ~15.7 s).  The 238 evidence run of
    /// 2026-08-30 then produced the same shape with no CUDA error at all: one
    /// case of 128, ~18.4 s, "CUDA BiCGSTAB detected a non-finite value",
    /// always the FIRST case a lane took, a different deck every rerun, every
    /// deck bit-identical single-shot.  The cause is fixed where it lives
    /// (CudaOuterGraph.cu's capture-race retry no longer re-records a body
    /// whose enqueue helpers have already committed their upload shadows), and
    /// this is the belt beside that brace: if a lane's FIRST case still dies
    /// non-finite, say so with the arbiter's stand-up provenance on the line,
    /// and run the case once more from a clean slot.
    ///
    /// WHY EXACTLY ONE, AND ONLY THE FIRST CASE.  The corrupting window is the
    /// graph BUILD, and the caches that build feeds are per-slot and
    /// process-lived: a lane's second case replays a graph the first case
    /// instantiated, so a second case dying non-finite is physics, not a race,
    /// and retrying it would launder a real answer into a lucky one.  A retry
    /// that fails is reported exactly as the first failure would have been.
    ///
    /// THE SLOT IS CLEAN BY CONSTRUCTION: runOneCase scopes the Driver, so the
    /// arena slot is released, whole-struct reset and audited between the two
    /// runs -- the same door a refill goes through.
    void retryAfterCaptureRace(long long wave_id, int index, int lane,
                               bool lane_first_case, bool race_spanned,
                               const std::string& deck,
                               const std::string& output, ResultMode mode,
                               const std::string& warm_from, const std::string& warm_save,
                               const CaseFidelity& fidelity, int& status,
                               std::string& failure, Driver::CaseReceipt& receipt,
                               double& seconds, double& teardown_ms) {
        // WP19.2.  THE HOLE THIS OR CLOSES, AND WHY IT IS AN OR AND NOT A
        // REPLACEMENT.
        //
        // WP19.1 gated on `lane_first_case` alone, reasoning that the graph
        // caches a build feeds are per-slot and process-lived.  That is true of
        // the outer WHILE's cache; it is NOT true of the PPR WHILE's.
        // `s.graph_valid` lives on the PprBackend, the PprBackend lives on the
        // Driver, and the Driver lives for exactly one case -- which is why
        // every [RASBERY][PPR_GPU] receipt in the block-38 logs reads
        // `"graph_builds":1`, once per case rather than once per lane.  A
        // capture race can therefore corrupt a lane's EIGHTH case, and on 238
        // run3 proc5 it did: candidate_0060, case 8, lane 5, preceded by a
        // second `ppr.while` EndCapture(root)/901 retry on tid 1 -- and the
        // first-case-only belt never fired.
        //
        // `race_spanned` is the honest widening: the process's capture-race
        // counter moved WHILE THIS CASE RAN.  In a quiet process it is always
        // false, so this cannot launder a physics failure into a lucky rerun --
        // which is the property the first-case rule was protecting and which is
        // kept here by a measurement instead of by a proxy.
        if (status == 0 || (!lane_first_case && !race_spanned) ||
            !captureRaceCorruptionSuspect(failure))
            return;
        _capture_race_case_retries.fetch_add(1, std::memory_order_relaxed);
        {
            std::ostringstream line;
            line << "[RASBERY][EVALUATOR][CAPTURE_RACE] {\"wave_id\":" << wave_id
                 << ",\"case\":" << index << ",\"lane\":" << lane
                 << ",\"deck\":" << detail::quoted(deck)
                 // WP19.2: no longer a constant.  The two reasons the belt can
                 // fire are now different facts and the receipt says which one
                 // did -- a `race_spanned` retry on a lane's fifth case is the
                 // evidence WP19.1's log could not produce.
                 << ",\"lane_first_case\":" << (lane_first_case ? "true" : "false")
                 << ",\"race_spanned\":" << (race_spanned ? "true" : "false")
                 << ",\"error\":" << detail::quoted(failure)
                 << ",\"action\":\"retry_once_clean_slot\","
                 << rasbery::captureArbiterProvenance() << "}";
            // BOTH streams, for the same reason the [ERROR] line takes both:
            // `_out` may be a pipe a controller owns, and a race that fired must
            // reach a human reading the log even when nobody reads the protocol.
            {
                std::lock_guard<std::mutex> lock(_out_mutex);
                _out << line.str() << std::endl;
            }
            std::cerr << line.str() << std::endl;
        }
        int                 retry_status   = 0;
        std::string         retry_failure;
        Driver::CaseReceipt retry_receipt;
        double              retry_seconds  = 0.0;
        double              retry_teardown = 0.0;
        runOneCase(deck, output, mode, warm_from, warm_save, fidelity, retry_status,
                   retry_failure, retry_receipt, retry_seconds, retry_teardown);
        if (retry_status == 0)
            _capture_race_case_recovered.fetch_add(1, std::memory_order_relaxed);
        // The retry's answer REPLACES the first, whichever way it went: a case
        // that survived is an ok case with its own digest, and a case that died
        // twice is reported with the second death's message so the receipt
        // describes the run that actually produced the output file.
        status      = retry_status;
        failure     = retry_failure;
        receipt     = retry_receipt;
        seconds    += retry_seconds;
        teardown_ms = retry_teardown;
        std::ostringstream line;
        line << "[RASBERY][EVALUATOR][CAPTURE_RACE][RESULT] {\"wave_id\":" << wave_id
             << ",\"case\":" << index << ",\"lane\":" << lane
             << ",\"deck\":" << detail::quoted(deck)
             << ",\"lane_first_case\":" << (lane_first_case ? "true" : "false")
             << ",\"race_spanned\":" << (race_spanned ? "true" : "false")
             << ",\"recovered\":" << (retry_status == 0 ? "true" : "false")
             << ",\"error\":"
             << (retry_failure.empty() ? std::string("null") : detail::quoted(retry_failure))
             << ',' << rasbery::captureArbiterProvenance() << "}";
        {
            std::lock_guard<std::mutex> lock(_out_mutex);
            _out << line.str() << std::endl;
        }
        std::cerr << line.str() << std::endl;
    }

    /// Build a Driver, drive it, destroy it -- and time the destruction apart
    /// from the drive.
    ///
    /// THE DRIVER IS SCOPED so its destructor -- which releases the arena slot
    /// -- runs before the teardown stamp closes, for the same reason main.cpp's
    /// batch branch scopes it before jobFinished(): the number this mode has to
    /// keep small is the refill latency, and teardown IS that latency.
    static void runOneCase(const std::string& deck, const std::string& output, ResultMode mode,
                           const std::string& warm_from, const std::string& warm_save,
                           const CaseFidelity& fidelity,
                           int& status, std::string& failure, Driver::CaseReceipt& receipt,
                           double& seconds, double& teardown_ms) {
        const auto started = std::chrono::steady_clock::now();
        auto       drove   = started;
        // FAILURE ISOLATION (plan WP8 "실패 격리").  An escaping exception would
        // terminate the OpenMP region and take the other cases' partial results
        // with it; a RASBERY_GPU_FULL=1 fail-closed refusal arrives here as
        // exactly such an exception and MUST fail only this case.  The process
        // keeps answering.
        try {
            // WP10.4.  `live_cases` counts CONSTRUCTED Drivers.  The guard is
            // declared outside the Driver's own scope so it is destroyed after
            // the Driver is -- i.e. the count falls when the case's state is
            // actually gone, which is what the between-generation receipt
            // asserts is 0.  The Driver keeps its own brace: the slot release
            // in its destructor has to land inside the measured teardown.
            detail::liveCases().fetch_add(1, std::memory_order_relaxed);
            struct LiveGuard {
                ~LiveGuard() { detail::liveCases().fetch_sub(1, std::memory_order_relaxed); }
            } live_guard;
            {
                Driver driver(deck, output, mode);
                driver.setWarmStart(warm_from, warm_save);
                // WP10.3.  BEFORE Drive(), because the burnup grid is applied
                // inside ReadInput and the tolerances are copied into
                // SolverContext at the top of the solve.
                driver.setCaseFidelity(fidelity);
                status  = driver.Drive();
                drove   = std::chrono::steady_clock::now();
                receipt = driver.caseReceipt();
            }
        } catch (const std::exception& error) {
            status  = 1;
            failure = error.what();
            drove   = std::chrono::steady_clock::now();
        } catch (...) {
            status  = 1;
            failure = "unknown exception";
            drove   = std::chrono::steady_clock::now();
        }
        const auto done = std::chrono::steady_clock::now();
        seconds     = std::chrono::duration<double>(done - started).count();
        teardown_ms = std::chrono::duration<double>(done - drove).count() * 1.0e3;
    }

    void reportCase(long long wave_id, int index, const std::string& key,
                    const std::string& deck, const std::string& output, ResultMode mode,
                    int status, const std::string& failure,
                    const Driver::CaseReceipt& receipt, double seconds, double teardown_ms,
                    int lane, bool isolation) {
        std::ostringstream line;
        line << "[RASBERY][EVALUATOR][CASE] {\"wave_id\":" << wave_id << ",\"case\":" << index
             << ",\"key\":" << (key.empty() ? std::string("null") : detail::quoted(key))
             // WP10.1.  `key` above is the CLIENT's opaque label, echoed back so
             // a controller can match a reply to the request it sent.
             // `case_key` is the SOLVER's canonical duplicate key of what was
             // actually computed -- the loading pattern folded onto its symmetry
             // orbit, the fidelity, the arm environment and the library digest.
             // They are different questions and a cache must not use the first
             // for the second: two labels can name one case, and one label can
             // name two after a deck edit.
             << ",\"case_key\":"
             << (receipt.case_key.empty() ? std::string("null")
                                          : detail::quoted(receipt.case_key))
             << ",\"deck\":" << detail::quoted(deck)
             << ",\"output\":" << detail::quoted(output)
             << ",\"result_mode\":\"" << ResultModeName(mode)
             << "\",\"status\":\"" << (status == 0 ? "ok" : "failed")
             << "\",\"exit_code\":" << status
             << ",\"digest\":";
        if (receipt.complete) {
            std::ostringstream hex;
            hex << std::hex << std::setw(16) << std::setfill('0') << receipt.digest;
            line << "\"" << hex.str() << "\"";
        } else {
            line << "null";
        }
        // WP10.3.  WHAT THIS CASE SOLVED AT, on the case's own line.
        //
        // The process prints ONE [PHYSICS_MODE] receipt and a mixed wave has
        // sixty-four answers, so the process receipt can no longer stand for
        // any of them.  These five are what tools/exact_audit.py
        // audit_case_fidelity checks PER CASE against the per-request
        // declaration, and they come from Driver::CaseReceipt -- i.e. from the
        // configuration the Driver was actually built with, not from the
        // request, so a request that was mis-applied shows up as a mismatch
        // rather than as an echo of itself.
        //
        // A case that never got a receipt (it threw before the fold closed)
        // prints nulls, not defaults: "this case has no fidelity to report" and
        // "this case ran strict" must not look the same to an audit.
        const bool have_fidelity = !receipt.policy.empty();
        line << ",\"policy\":"
             << (have_fidelity ? detail::quoted(receipt.policy) : std::string("null"))
             << ",\"physics_fidelity\":"
             << (have_fidelity ? detail::quoted(receipt.physics_fidelity) : std::string("null"))
             << ",\"statepoint_grid\":"
             << (have_fidelity ? detail::quoted(receipt.statepoint_grid) : std::string("null"))
             << ",\"acceptance_eligible\":"
             << (have_fidelity ? (receipt.acceptance_eligible ? "true" : "false") : "null")
             << ",\"fidelity_declared\":"
             << (receipt.fidelity_declared.empty() ? std::string("null")
                                                   : detail::quoted(receipt.fidelity_declared))
             << ",\"promoted_from\":"
             << (receipt.promoted_from.empty() ? std::string("null")
                                               : detail::quoted(receipt.promoted_from))
             << ",\"statepoints\":" << receipt.statepoints
             << ",\"outers\":" << receipt.outers
             << ",\"th_updates\":" << receipt.th_updates
             << ",\"slot\":" << receipt.slot
             << ",\"lane\":" << lane
             << std::fixed << std::setprecision(3)
             << ",\"wall_s\":" << seconds
             << ",\"teardown_ms\":" << teardown_ms
             << ",\"isolation_check\":" << (isolation ? "true" : "false")
             << ",\"error\":"
             << (failure.empty() ? std::string("null") : detail::quoted(failure)) << "}";
        _out << line.str() << std::endl;

        // WP19.  A DEAD CASE IS NOT A FIELD OF A RECEIPT.
        //
        // The [CASE] line above has carried `"error"` since WP8, and that is
        // exactly why the capture race was invisible for a campaign: the
        // dispatcher parsed the receipt, took `deck` and `status` out of it and
        // threw the error text away, and the harness log for a run that lost
        // five cases contained no error text at all -- only five deck names on
        // a "did not produce a result" line.  So the failure gets its OWN line,
        // at its own severity, with the case id and the message on it, and the
        // dispatcher lifts THIS into the [MULTI_GPU][FAIL] text
        // (tools/run_multi_gpu_batch.py, EVALUATOR_CASE_ERROR).
        //
        // Printed for every non-zero status, including one whose exception left
        // no message: "failed, and nothing said why" is itself the report.
        if (status != 0) {
            std::ostringstream err;
            err << "[RASBERY][EVALUATOR][ERROR] {\"wave_id\":" << wave_id
                << ",\"case\":" << index
                << ",\"deck\":" << detail::quoted(deck)
                << ",\"output\":" << detail::quoted(output)
                << ",\"lane\":" << lane
                << ",\"slot\":" << receipt.slot
                << ",\"exit_code\":" << status
                << ",\"error\":"
                << (failure.empty() ? detail::quoted("no message (the case died "
                                                     "without an exception)")
                                    : detail::quoted(failure))
                << "}";
            _out << err.str() << std::endl;
            // stderr as well, unconditionally: _out can be a pipe a controller
            // owns, and a case death must reach a human reading the log even
            // when nobody is reading the protocol.
            std::cerr << err.str() << std::endl;
        }
    }

    /// WP19.1.  Cases that hit the loud path, and cases the one retry saved.
    /// Atomic because the wave path runs its lanes under `omp parallel for`;
    /// they move at most once per lane per wave, so the counter is never on a
    /// path that repeats.
    std::atomic<long long>                _capture_race_case_retries{0};
    std::atomic<long long>                _capture_race_case_recovered{0};

    Options                               _options;
    std::ostream&                         _out;
    Summary                               _summary;
    int                                   _exit_code      = 0;
    long long                             _arena_releases = 0;
    std::chrono::steady_clock::time_point _t0{};
    // WP18.  Untouched -- not merely unused -- when RASBERY_EVALUATOR_ROLLING
    // is unset: no thread is started, no queue is configured, and the wave path
    // never names any of it.
    RollingQueue                          _queue;
    RollingState                          _roll;
    std::mutex                            _out_mutex;
};

} // namespace rasbery::evaluator
