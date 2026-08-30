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
#include "CudaXsReconBackend.h"
#include "Driver.h"
#include "HostPinRegistry.h"
#include "IoWriter.h"
#include "RunContract.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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
inline void applyWaveFidelityDefault(const FidelityRequest& wave, FidelityRequest& request) {
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

class Server {
public:
    Server(const Options& options, std::ostream& out) : _options(options), _out(out) {
        _t0 = std::chrono::steady_clock::now();
    }

    /// Read the stream to its stop condition.  Returns the process exit code:
    /// nonzero if any case failed or any request was refused, because a
    /// generation that silently lost four candidates is not a generation.
    int run() {
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
             << ",\"isolation_check\":" << (_options.isolation_check ? "true" : "false")
             << "}" << std::endl;

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
                    applyWaveFidelityDefault(wave_fidelity, c.request_fidelity);
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
                wave.cases.swap(pending);
                runWave(wave, wave_mode);
                continue;
            }
            refuse("unknown op \"" + op + "\" (case | wave | run | shutdown)", line);
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
            << ",\"stop_reason\":\"" << stopReasonName(_summary.stop) << "\"}"
            << std::endl;
        out.unsetf(std::ios::floatfield);
    }

    /// Called by main.cpp at shutdown, immediately after
    /// rasberyReleaseBatchArena(), so the receipt's witness is a fact.
    void stampArenaRelease() { ++_arena_releases; }

private:
    void refuse(const std::string& why, const std::string& line) {
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
             << "}"
             // Bytes page-locked through HostPinRegistry right now.  The arena's
             // own cudaMallocHost blocks are stood up once for the process and
             // are NOT here: this is the number that moves with cases.
             << ",\"cuda_host_bytes\":" << rasberyHostPinLiveBytes()
             << "}" << std::endl;
        _out.unsetf(std::ios::floatfield);
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
    }

    Options                               _options;
    std::ostream&                         _out;
    Summary                               _summary;
    int                                   _exit_code      = 0;
    long long                             _arena_releases = 0;
    std::chrono::steady_clock::time_point _t0{};
};

} // namespace rasbery::evaluator
