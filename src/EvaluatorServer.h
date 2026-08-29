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
// WHAT IS REFUSED RATHER THAN APPROXIMATED.  Two request fields cannot be
// honoured per case in stage 1, and both fail the case instead of being
// silently ignored:
//
//   `batch_width`  the arena is one allocation sized at the first admission and
//                  fixed for the process (CudaBICGBackend.cu).  The FIRST wave
//                  latches the width; a later wave asking for a different one
//                  is refused by name.  Stage 2's CohortContext is where a
//                  second shape becomes legal.
//   `fidelity`     PhysicsFidelity resolves ONCE per process from the
//                  environment and caches (RunContract.h's function-local
//                  statics), and the [PHYSICS_MODE] receipt has already been
//                  printed by the time the first request is read.  A per-case
//                  fidelity is therefore a claim this process cannot keep, so a
//                  request that names one that disagrees with the process's
//                  effective fidelity is refused.  Agreeing is allowed, and is
//                  how a client asserts what it thinks it is talking to.
//
// A quiet "close enough" here is how a screening result reaches an acceptance
// table, which is the failure mode WP1's exact-only contract exists for.

#include "BatchRefill.h"
#include "CudaXsReconBackend.h"
#include "Driver.h"
#include "HostPinRegistry.h"
#include "IoWriter.h"
#include "RunContract.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
    std::string fidelity;             ///< declared, VALIDATED, never applied
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

/// Everything the process receipt reports, accumulated across waves.
struct Summary {
    long long cases        = 0;
    long long ok           = 0;
    long long failed       = 0;
    long long generations  = 0;
    long long refused      = 0; ///< requests rejected before a Driver was built
    long long isolation_checks     = 0;
    long long isolation_mismatches = 0;
    long long isolation_adjacent   = 0;
    double    uptime_s     = 0.0;
    double    drive_s      = 0.0;
    int       latched_width = 0;
    std::vector<double> case_seconds;
    std::vector<double> teardown_ms;
    StopReason stop = StopReason::EndOfStream;
    std::uint64_t pin_live_ranges_between_waves = 0;
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
    if (object.contains("fidelity")) {
        if (!object["fidelity"].is_string()) {
            error = R"("fidelity" must be a string)";
            return false;
        }
        out.fidelity = object["fidelity"].get<std::string>();
    }
    return true;
}

/// The fidelity gate.  See the header comment: a request may ASSERT the
/// process's fidelity and may not change it.
inline bool fidelityAgrees(const std::string& declared, std::string& error) {
    if (declared.empty()) return true;
    PhysicsFidelity parsed = PhysicsFidelity::FullExact;
    if (!parsePhysicsFidelity(declared, parsed)) {
        error = "\"fidelity\":\"" + declared + "\" is not a fidelity this build knows";
        return false;
    }
    const PhysicsFidelity effective = effectivePhysicsFidelity();
    if (parsed != effective) {
        error = std::string("\"fidelity\":\"") + declared + "\" but this process resolved " +
                physicsPolicyName(effective) +
                " from its environment and cannot change fidelity per case (WP8 stage 1). "
                "Start a process whose RASBERY_* fidelity environment matches, or drop the "
                "field.";
        return false;
    }
    return true;
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
            if (op == "case") {
                CaseRequest request;
                if (!parseCase(object, _options.default_result_mode, request, error)) {
                    refuse(error, line);
                    continue;
                }
                if (!fidelityAgrees(request.fidelity, error)) {
                    refuse(error, line);
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
                std::string declared;
                if (object.contains("physics_fidelity") && object["physics_fidelity"].is_string())
                    declared = object["physics_fidelity"].get<std::string>();
                else if (object.contains("fidelity") && object["fidelity"].is_string())
                    declared = object["fidelity"].get<std::string>();
                if (!fidelityAgrees(declared, error)) {
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
            tail.cases.swap(pending);
            runWave(tail, _options.default_result_mode);
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
        const XsLibraryCacheStats xslib = XsLibraryCacheSnapshot();
        const auto&               ten   = refill::tenancy();
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
            << ",\"library_loads\":" << xslib.loads
            // One CaseContext per case, by construction: stage 1 does not reuse
            // geometry.  Stage 2 is where this stops equalling `cases`.
            << ",\"geometry_builds\":" << _summary.cases
            << ",\"pin_live_ranges_between_waves\":"
            << _summary.pin_live_ranges_between_waves
            << ",\"case_seconds\":{\"p50\":"
            << detail::percentile(_summary.case_seconds, 0.50)
            << ",\"p90\":" << detail::percentile(_summary.case_seconds, 0.90) << "}"
            << ",\"case_teardown_ms\":{\"p50\":"
            << detail::percentile(_summary.teardown_ms, 0.50)
            << ",\"p90\":" << detail::percentile(_summary.teardown_ms, 0.90)
            << ",\"max\":" << (_summary.teardown_ms.empty()
                                   ? 0.0
                                   : *std::max_element(_summary.teardown_ms.begin(),
                                                       _summary.teardown_ms.end()))
            << "}"
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
        }
        for (const CaseRequest& c : wave.cases) {
            jobs.inputs.push_back(c.deck);
            jobs.outputs.push_back(c.output);
            jobs.modes.push_back(c.result_mode);
            jobs.keys.push_back(c.key);
            jobs.warm_from.push_back(c.warm_start_from);
            jobs.warm_save.push_back(c.save_warm_state);
        }
        jobs.keys.resize(jobs.inputs.size());
        jobs.warm_from.resize(jobs.inputs.size());
        jobs.warm_save.resize(jobs.inputs.size());
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
            _summary.case_seconds.push_back(seconds[u]);
            _summary.teardown_ms.push_back(teardown[u]);
        }
        _summary.cases  += njobs;
        _summary.ok     += ok;
        _summary.failed += failed;
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
            runOneCase(jobs.inputs[u0], recheck_output, jobs.modes[u0], jobs.warm_from[u0],
                       std::string(), recheck_status,
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
            {
                Driver driver(deck, output, mode);
                driver.setWarmStart(warm_from, warm_save);
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
        line << ",\"statepoints\":" << receipt.statepoints
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
