#include "Benchmark.h"
#include "Exporter.h"
#include "Importer.h"

#include "BatchRefill.h"
#include "CaseFidelity.h"
#include "CudaXsReconBackend.h"
#include "Driver.h"
#include "EvaluatorServer.h"
#include "GpuFullContract.h"
#include "RunContract.h"
#include "XferLedger.h"
#include "XSTiming.h"
#include "plog/Appenders/ConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Init.h"
#include "plog/Log.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <system_error>
#include <vector>
#ifdef _OPENMP
    #include <omp.h>
#endif
#if !defined(_WIN32)
    #include <unistd.h>
#endif
#if defined(__linux__)
    #include <sched.h>
#endif

namespace rasbery {
// Definition of the solver OpenMP gate declared in Geometry.h. Default of 256 nodes:
// below this the per-node solver loops stay serial (fork/join overhead wins). Overridable at
// runtime via RASBERY_OMP_GATE for tuning on a given machine/core size.
int rasbery_omp_gate = 256;
} // namespace rasbery

namespace {

constexpr int RASBERY_OMP_THREADS = 8;

/// WP10.2.  Where a job's warm state goes.
///
/// The DERIVED default keys off the job's OUTPUT path, deliberately, and it is
/// the same rule Driver::RestartPath already uses: `--raso` is the one string
/// main() and the launcher both guarantee is unique per deck, so deriving from
/// it is what makes "distinct outputs implies distinct warm-state namespace"
/// true.  An explicit path is honoured as given -- and refused for a multi-job
/// run below, because N decks writing one file is N-1 lost states and one
/// arbitrary winner.
std::string WarmStatePath(bool enabled, const std::string& explicit_path,
                          const std::string& output) {
    if (!enabled) return {};
    if (!explicit_path.empty()) return explicit_path;
    const std::filesystem::path out(output);
    const std::string           stem = out.stem().string();
    return (out.parent_path() / ((stem.empty() ? std::string("result") : stem) + ".warm"))
        .string();
}

/// Live process affinity capacity straight from the kernel, or 0 when the
/// platform/query cannot answer.  Only trustworthy while nothing has bound the
/// calling thread yet -- see rasberyVisibleCpuThreads() below.
int rasberyAffinityCpuCount() {
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        const int count = CPU_COUNT(&affinity);
        if (count > 0) return count;
    }
#endif
    return 0;
}

int rasberyVisibleCpuThreads() {
    // RASBERY_STARTUP_CPUS is stamped by rasberyPrepareOpenMPStartup() in the
    // FIRST process image, before OMP_PROC_BIND=TRUE / OMP_PLACES=cores are
    // exported and the process re-execs itself.  In the re-exec'd image libgomp
    // binds the initial thread to a single place before main() is entered, so a
    // sched_getaffinity here reports that one place (2 CPUs on an SMT host)
    // instead of the 24-CPU cpuset the job actually owns.  Prefer the stamped
    // value; the live query stays the fallback for images that never re-exec.
    if (const char* startup = std::getenv("RASBERY_STARTUP_CPUS")) {
        const int stamped = std::atoi(startup);
        if (stamped > 0) return stamped;
    }
    const int live = rasberyAffinityCpuCount();
    if (live > 0) return live;
#ifdef _OPENMP
    return std::max(1, omp_get_num_procs());
#else
    return 1;
#endif
}

void rasberySetEnv(const char* key, const char* value, bool overwrite = false) {
#if defined(_WIN32)
    if (!overwrite && std::getenv(key) != nullptr) return;
    _putenv_s(key, value);
#else
    setenv(key, value, overwrite ? 1 : 0);
#endif
}

bool rasberySetEnvIfNeeded(const char* key, const char* value, bool overwrite = false) {
    const char* current = std::getenv(key);
    if (current != nullptr && (!overwrite || std::strcmp(current, value) == 0))
        return false;

    rasberySetEnv(key, value, true);
    return true;
}

/// The process cost ledger (GA evaluator plan Sec 2.2, T_process ~= 2.0 s/case).
///
/// WHAT IT DECOMPOSES.  `wall - TOTAL DRIVER TIME` was the only number for
/// everything a case pays outside Drive(), and it is a subtraction of two
/// measurements taken by different observers.  This splits it into the parts a
/// persistent evaluator would each amortise differently:
///
///   exec_s        the OpenMP re-exec: a second process image, loader and all.
///                 Null when the environment was already stamped, which is what
///                 a launcher that exports OMP_* itself achieves TODAY.
///   pre_drive_s   argv/manifest parsing, the receipts, host-pinning setup.
///   drive_s       every Driver::Drive in this process, wall to wall.
///   post_drive_s  writer drain, arena teardown, CUDA teardown, the receipts.
///
/// What is left over against /usr/bin/time is the first image's loader and the
/// static destructors after main returns -- deliberately not guessed at here.
void rasberyPrintProcessLedger(std::ostream& out, double exec_seconds,
                               double pre_drive_seconds, double drive_seconds,
                               double post_drive_seconds, int jobs) {
    out << "[RASBERY][PROCESS] {\"jobs\":" << jobs << ",\"reexec\":"
        << (exec_seconds >= 0.0 ? "true" : "false") << ",\"exec_s\":";
    if (exec_seconds >= 0.0)
        out << std::fixed << std::setprecision(3) << exec_seconds;
    else
        out << "null";
    out << std::fixed << std::setprecision(3)
        << ",\"pre_drive_s\":" << pre_drive_seconds
        << ",\"drive_s\":" << drive_seconds
        << ",\"post_drive_s\":" << post_drive_seconds
        << ",\"in_main_s\":" << (pre_drive_seconds + drive_seconds + post_drive_seconds)
        << "}" << std::endl;
    out.unsetf(std::ios::floatfield);
}

/// One word for what a whole run was asked to write: the common mode, or
/// "mixed" when the jobs disagree.  Empty job lists report the default.
std::string rasberyResultModeSummary(const std::vector<rasbery::ResultMode>& modes) {
    if (modes.empty()) return rasbery::ResultModeName(rasbery::BatchLightResult::DefaultMode());
    for (const rasbery::ResultMode m : modes)
        if (m != modes.front()) return "mixed";
    return rasbery::ResultModeName(modes.front());
}

/// Comparison key for job-namespace collisions (plan Rev.4 Sec 7).
///
/// weakly_canonical resolves symlinks and `..` for the part of the path that
/// exists and normalises the rest, so it answers for files that have not been
/// written yet -- which every --raso is at argument-parsing time.  On Windows
/// the comparison is additionally case-folded, because the filesystem is:
/// `out/A.h5` and `OUT/a.h5` are one file there and two strings everywhere.
std::string rasberyPathKey(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path        resolved = fs::weakly_canonical(fs::path(path), ec);
    if (ec) {
        ec.clear();
        resolved = fs::absolute(fs::path(path), ec);
        if (ec) resolved = fs::path(path);
    }
    std::string key = resolved.lexically_normal().string();
#if defined(_WIN32)
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
    return key;
}

/// Read one `--jobs` manifest, appending its pairs to the deck vectors.
///
/// WHY A FILE AT ALL.  Task 20's acceptance case is 1,280 jobs, and 1,280
/// --rasi paths plus 1,280 --raso paths is a ~200 kB argument list -- past
/// MAX_ARG_STRLEN on Linux and past every practical limit on Windows.  The
/// manifest is the same information off the filesystem instead of off argv.
///
/// FORMAT.  One job per line, `<input.json> <output.h5> [result-mode]`,
/// separated by whitespace.  `#` starts a comment, blank lines are skipped, and
/// a line may quote either path with `"` so paths with spaces survive.
/// Deliberately not JSON: this is read before anything else is initialised, it
/// has to fail with a line number a human can act on, and the launcher
/// (tools/run_multi_gpu_batch.py) has to be able to split one by line count
/// without a parser.
///
/// THE THIRD FIELD is optional and is this job's result mode -- `full`,
/// `pin-off` or `light` -- overriding `--result` and the environment FOR THIS
/// JOB ONLY.  That is the whole reason it exists: a GA wave runs its population
/// light and its promoted elites full, and both belong in one --batch-mode
/// process, which one process-wide environment variable cannot express.
///
/// The pairs land in the SAME two vectors the --rasi/--raso flags fill, before
/// any of the validation below runs.  So the distinct-output rule, the counts
/// match rule and the batch predicate all apply to manifest jobs unchanged, and
/// a manifest may be mixed with explicit flags.
bool rasberyReadJobManifest(const std::string&              manifest_path,
                            std::vector<std::string>&       inputs,
                            std::vector<std::string>&       outputs,
                            std::vector<rasbery::ResultMode>& modes,
                            rasbery::ResultMode             default_mode,
                            std::string&                    error) {
    std::ifstream file(manifest_path);
    if (!file) {
        error = "cannot open job manifest: " + manifest_path;
        return false;
    }

    const auto next_field = [](const std::string& line, std::size_t& pos) -> std::string {
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
        if (pos >= line.size()) return {};
        if (line[pos] == '"') {
            const std::size_t open = ++pos;
            while (pos < line.size() && line[pos] != '"') ++pos;
            const std::string value = line.substr(open, pos - open);
            if (pos < line.size()) ++pos; // closing quote
            return value;
        }
        const std::size_t start = pos;
        while (pos < line.size() && !std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
        return line.substr(start, pos - start);
    };

    std::string line;
    int         lineno = 0;
    while (std::getline(file, line)) {
        ++lineno;
        // Manifests written on Windows and read in WSL are the campaign's
        // standing trap (see the APR1400 CRLF finding): a trailing \r would
        // become part of the output path and every deck would write to a file
        // nothing downstream can find.
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);

        std::size_t       pos    = 0;
        const std::string input  = next_field(line, pos);
        if (input.empty()) continue; // blank or comment-only
        const std::string output = next_field(line, pos);
        if (output.empty()) {
            error = manifest_path + ":" + std::to_string(lineno) +
                    ": every manifest line needs both an input deck and an output path "
                    "(`<input.json> <output.h5>`). A missing --raso would make every job "
                    "write the same default result.h5.";
            return false;
        }
        rasbery::ResultMode mode = default_mode;
        const std::string   third = next_field(line, pos);
        if (!third.empty() && !rasbery::ParseResultMode(third, mode)) {
            error = manifest_path + ":" + std::to_string(lineno) +
                    ": third field must be a result mode (full | pin-off | light), got \"" +
                    third + "\". Quote a path that contains spaces.";
            return false;
        }
        const std::string trailing = next_field(line, pos);
        if (!trailing.empty()) {
            error = manifest_path + ":" + std::to_string(lineno) +
                    ": expected at most three fields, found a fourth (" + trailing +
                    "). Quote a path that contains spaces.";
            return false;
        }
        inputs.push_back(input);
        outputs.push_back(output);
        modes.push_back(mode);
    }
    return true;
}

void rasberyPrepareOpenMPStartup(char* argv[]) {
#if !defined(_WIN32)
    if (std::getenv("RASBERY_OMP_ENV_READY") != nullptr)
        return;

    // Stamp the true cpuset capacity here, in the first image, while nothing has
    // bound this thread yet.  The OMP_PROC_BIND/OMP_PLACES pair set below makes
    // libgomp pin the initial thread of the re-exec'd image, and from that point
    // on sched_getaffinity can only see the one place it was pinned to.  An
    // inherited/explicit value is left alone.
    if (std::getenv("RASBERY_STARTUP_CPUS") == nullptr) {
        const int startup_cpus = rasberyAffinityCpuCount();
        if (startup_cpus > 0)
            rasberySetEnv("RASBERY_STARTUP_CPUS", std::to_string(startup_cpus).c_str(), true);
    }

    bool changed = false;
    changed |= rasberySetEnvIfNeeded("OMP_WAIT_POLICY", "PASSIVE");
    changed |= rasberySetEnvIfNeeded("GOMP_SPINCOUNT", "0");
    changed |= rasberySetEnvIfNeeded("OMP_NUM_THREADS", "8", true);
    changed |= rasberySetEnvIfNeeded("OMP_PROC_BIND", "TRUE", true);
    changed |= rasberySetEnvIfNeeded("OMP_PLACES", "cores", true);
    rasberySetEnv("RASBERY_OMP_ENV_READY", "1", true);

    if (changed) {
        // Stamp the monotonic clock across the exec.  The re-exec is a SECOND
        // process image -- loader, CUDA runtime init, libgomp -- and it is a
        // per-CASE cost today because every case is a process.  Nothing could
        // see it before: Driver::Drive starts after it, and `wall - drive` is a
        // subtraction that lumps it with teardown (GA evaluator plan Sec 2.2's
        // T_process = 2.0 s).  The environment survives execvp, which is what
        // makes this measurable at all.
        rasberySetEnv(
            "RASBERY_STARTUP_STAMP_NS",
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()).c_str(),
            true);
        execvp(argv[0], argv);
    }
#else
    rasberySetEnv("OMP_WAIT_POLICY", "PASSIVE");
    rasberySetEnv("GOMP_SPINCOUNT", "0");
    rasberySetEnv("OMP_NUM_THREADS", "8", true);
    rasberySetEnv("OMP_PROC_BIND", "TRUE", true);
    rasberySetEnv("OMP_PLACES", "cores", true);
#endif
}

} // namespace

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    rasberyPrepareOpenMPStartup(argv);
    // Process cost ledger (GA evaluator plan Sec 2.2, Task C).  A persistent
    // evaluator amortises everything measured here over the whole population;
    // today every case pays all of it.
    const auto process_start = std::chrono::steady_clock::now();
    double     exec_seconds  = -1.0;
    if (const char* stamp = std::getenv("RASBERY_STARTUP_STAMP_NS")) {
        try {
            const long long then = std::stoll(stamp);
            exec_seconds =
                static_cast<double>(process_start.time_since_epoch().count() - then) / 1.0e9;
        } catch (const std::exception&) {
            exec_seconds = -1.0; // an inherited stamp from another run; report nothing
        }
    }
    const auto since_start = [&process_start](
                                 const std::chrono::steady_clock::time_point& t) {
        return std::chrono::duration<double>(t - process_start).count();
    };
    // Set by whichever branch runs the decks; the receipt reads both.
    double pre_drive_seconds = 0.0;
    auto   drive_end         = process_start;

    // Capture the taskset/cpuset capacity before the first OpenMP API call.
    // With OMP_PROC_BIND=TRUE libgomp later narrows the main thread to one
    // place, so querying sched_getaffinity inside the batch branch is too late.
    const int startup_visible_cpus = rasberyVisibleCpuThreads();

    // OpenMP env is configured in rasberyPrepareOpenMPStartup() above, before libgomp starts.
#ifdef _OPENMP
    omp_set_dynamic(0);
    const char* omp_threads_env = std::getenv("RASBERY_OMP_THREADS");
    omp_set_num_threads(omp_threads_env ? std::max(1, std::atoi(omp_threads_env)) : RASBERY_OMP_THREADS);
#endif
    if (const char* gate_env = std::getenv("RASBERY_OMP_GATE"))
        rasbery::rasbery_omp_gate = std::max(0, std::atoi(gate_env));

    plog::ConsoleAppender<plog::TxtFormatter> console_appender;
    plog::init(plog::error, &console_appender);

    const fs::path repo_dir = DATA_DIR;

    std::vector<std::string> chiffon_inputs;
    std::vector<std::string> chiffon_outputs;
    std::vector<std::string> rasbery_inputs;
    std::vector<std::string> rasbery_outputs;
    std::vector<std::string> iso_hgcs;  // --isohgc: HGCs to export per-point isotope state from
    std::string              iso_csv;   // --isocsv: output CSV (offline correlation/VIF study)
    std::vector<std::string> validate_args; // --validate <input.json> <out.csv> [fineKey coarseKey]
    int                      batch_width = 0; // --batch-mode M: M instances in one process
    std::vector<std::string> job_manifests;   // --jobs: deck/output pairs read from a file
    // --result: what every job produces unless its manifest line says otherwise.
    // Unset means "whatever the environment says", so the default path is byte
    // for byte what it was before the flag existed.
    rasbery::ResultMode      result_mode = rasbery::BatchLightResult::DefaultMode();
    std::vector<rasbery::ResultMode> rasbery_result_modes;

    // WP8 stage 1 (BOTTLENECK plan Sec 5.1 / GA evaluator plan Sec 6.2 Task 6).
    // `--evaluator-jsonl [path]` turns this process into a long-lived case
    // server instead of a one-shot job runner.  Everything ABOVE this branch --
    // the exact-only gate, the [PHYSICS_MODE] receipt, the OpenMP startup -- is
    // unchanged and still runs first: an evaluator that skipped the fidelity
    // contract would be the fastest way yet to get a screening result into an
    // acceptance table.
    bool        evaluator_mode            = false;
    std::string evaluator_request_path    = "-"; ///< "-" is stdin
    double      evaluator_idle_timeout    = -1.0;
    bool        evaluator_isolation_check = false;
    // WP10.2 warm start.  Empty means off; `--save-warm-state` with no path
    // derives one per job from that job's output, so a batch cannot have two
    // decks writing one warm-state file.
    std::string warm_start_from;
    std::string warm_state_out;
    /// WP10.3.  `--statepoint-grid full|coarse|three|<GWd/t list>`: the burnup
    /// grid this run's decks are loaded on, applied inside IO::ReadInput before
    /// the deck digest (src/StatepointGrid.h).  Run-wide here, because a
    /// one-shot invocation IS one lane; the per-case form is the evaluator's
    /// `"statepoint_grid"` request field.  Unset is the deck as written.
    std::string statepoint_grid;
    bool        save_warm_state = false;

    int argi = 1;
    while (argi < argc) {
        const std::string option = argv[argi];

        // The three evaluator flags are handled HERE, before the generic
        // value-consuming loop below, because that loop requires every option
        // to be followed by at least one non-`--` value.  `--evaluator-jsonl`
        // takes an OPTIONAL path (default stdin) and
        // `--evaluator-isolation-check` takes none, so both would be rejected
        // by it with a misleading "Missing value after option".
        if (option == "--evaluator" || option == "--evaluator-jsonl") {
            evaluator_mode = true;
            ++argi;
            if (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
                evaluator_request_path = argv[argi];
                ++argi;
            }
            continue;
        }
        if (option == "--evaluator-idle-timeout") {
            ++argi;
            if (argi >= argc || std::string(argv[argi]).rfind("--", 0) == 0) {
                std::cerr << "Missing value after option: " << option << std::endl;
                return 1;
            }
            evaluator_idle_timeout = std::atof(argv[argi]);
            ++argi;
            continue;
        }
        if (option == "--evaluator-isolation-check") {
            evaluator_isolation_check = true;
            ++argi;
            continue;
        }

        // WP10.2.  Handled HERE, before the generic value-consuming loop,
        // because --save-warm-state takes an OPTIONAL path (default: derived
        // from the job's own output namespace) and that loop rejects any option
        // not followed by a value.
        if (option == "--warm-start-from") {
            ++argi;
            if (argi >= argc || std::string(argv[argi]).rfind("--", 0) == 0) {
                std::cerr << "Missing value after option: " << option << std::endl;
                return 1;
            }
            warm_start_from = argv[argi];
            ++argi;
            continue;
        }
        if (option == "--save-warm-state") {
            save_warm_state = true;
            ++argi;
            if (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
                warm_state_out = argv[argi];
                ++argi;
            }
            continue;
        }

        // WP10.3.  Before the generic loop for the same reason the two warm
        // start options are: the grid is ONE value, not a list, and a list
        // option would happily swallow the deck paths that follow it.
        if (option == "--statepoint-grid") {
            ++argi;
            if (argi >= argc || std::string(argv[argi]).rfind("--", 0) == 0) {
                std::cerr << "Missing value after option: " << option << std::endl;
                return 1;
            }
            statepoint_grid = argv[argi];
            std::vector<double> grid_check;
            std::string         grid_error;
            if (!rasbery::spgrid::parseGrid(statepoint_grid, grid_check, grid_error)) {
                std::cerr << grid_error << std::endl;
                return 1;
            }
            // The measured non-monotonicity, said where an operator can still
            // change the grid: cost is SUPERLINEAR in the burnup step, and the
            // 3-statepoint grid measured MORE outers than the 35-statepoint
            // deck.  A warning, not a refusal -- calibrating a grid is a
            // legitimate thing to do and this is the number to calibrate
            // against.
            if (rasbery::spgrid::largestStep(grid_check) > rasbery::spgrid::kWarnBurnupStepGwd)
                std::cerr << "[RASBERY][WARN][statepoint_grid] largest burnup step is "
                          << rasbery::spgrid::largestStep(grid_check) << " GWd/t (> "
                          << rasbery::spgrid::kWarnBurnupStepGwd
                          << "). Cost is superlinear in the step: the measured 3-statepoint "
                             "grid ran 5,104 outers against the 35-statepoint deck's 4,609. "
                             "Calibrate before believing the wall (tools/make_screening_deck.py)."
                          << std::endl;
            ++argi;
            continue;
        }

        if (option == "-h" || option == "--help") {
            std::cout << "Usage:\n"
                      << "  RASBERY [--chiffoni input1.json input2.json ...] [--chiffono output1.h5 output2.h5 ...]\n"
                      << "          [--rasi input1.json input2.json ...] [--raso output1.h5 output2.h5 ...]\n"
                      << "          [--batch-mode M] [--result full|pin-off|light]\n\n"
                      << "Notes:\n"
                      << "  - Values after each flag are consumed until the next --flag.\n"
                      << "  - The number of --chiffoni and --chiffono paths must match.\n"
                      << "  - The number of --rasi and --raso paths must match.\n"
                      << "  - --batch-mode M runs the --rasi decks M at a time in one process,\n"
                      << "    one host thread per instance, with their CMFD linear solves\n"
                      << "    batched into a single GPU grid.  Every deck must share one\n"
                      << "    geometry; only the state (temperatures, power, rods, schedule)\n"
                      << "    may differ.  Requires a CUDA build.  Each instance is\n"
                      << "    bit-identical to running that deck on its own.\n"
                      << "  - --result selects what every job WRITES; the physics and the\n"
                      << "    trajectory are identical in all three (same digest):\n"
                      << "      full     result HDF5 + restarts + pin-power CSV  (default)\n"
                      << "      pin-off  result HDF5 + restarts, no pin output\n"
                      << "      light    one JSONL line per statepoint, no HDF5\n"
                      << "    It mirrors RASBERY_BATCH_LIGHT_RESULT=1 (which is --result\n"
                      << "    light) and overrides it.  NONE of the three is gated: an\n"
                      << "    output mode is not a fidelity, so a light run is\n"
                      << "    acceptance-eligible and reports policy=strict (see\n"
                      << "    RASBERY_PHYSICS_FIDELITY below).  What IS gated is the\n"
                      << "    approximation: RASBERY_GA_FEEDBACK_PASSES>0 needs\n"
                      << "    RASBERY_ALLOW_SCREENING=1 and additionally requires light.\n"
                      << "  - RASBERY_PHYSICS_FIDELITY=strict|A2|L3coarse|feedback_limited\n"
                      << "    DECLARES the compute fidelity when the environment cannot\n"
                      << "    show it -- an L3 coarse deck is a reduced burnup grid and\n"
                      << "    sets no variable.  A declaration can only make the effective\n"
                      << "    policy COARSER; strict and A2 are detected from\n"
                      << "    RASBERY_GA_FEEDBACK_PASSES and RASBERY_STAGED_*_TOL.\n"
                      << "  - RASBERY_GPU_FULL=1 (alias RASBERY_GPU_STRICT=1) makes every\n"
                      << "    GPU arm fail CLOSED: a decline or CUDA failure that would\n"
                      << "    silently run the CPU body fails that CASE instead, naming the\n"
                      << "    site.  Default off; [RASBERY][GPU_FULL] reports the counts\n"
                      << "    either way.\n"
                      << "  - A --jobs manifest line may carry a THIRD field, that job's own\n"
                      << "    result mode, which overrides --result for that job alone --\n"
                      << "    how one wave runs a light population beside full elites.\n"
                      << "  - --evaluator-jsonl [path] (alias --evaluator) keeps ONE process\n"
                      << "    alive -- CUDA context, immutable XSLIB parse, T/H tables,\n"
                      << "    device library and arenas -- and reads JSONL case requests\n"
                      << "    from `path` (default stdin, `-`).  Requires --batch-mode M,\n"
                      << "    which is latched for the process: the arena is ONE allocation\n"
                      << "    sized at the first admission.  Requests:\n"
                      << "      {\"op\":\"case\",\"deck\":\"a.json\",\"output\":\"a.h5\",\n"
                      << "       \"result_mode\":\"light\",\"key\":\"cand-17\"}\n"
                      << "      {\"op\":\"wave\",\"wave_id\":17,\"jobs_manifest\":\"w17.jobs\"}\n"
                      << "      {\"op\":\"run\"}        run the cases enqueued since the last wave\n"
                      << "      {\"op\":\"shutdown\"}   stop, release the arena, print receipts\n"
                      << "    Each case still builds a fresh Driver/CaseContext, so its\n"
                      << "    trajectory digest is the one the same deck produces from a\n"
                      << "    one-shot --jobs run.  A case that throws (including a\n"
                      << "    RASBERY_GPU_FULL=1 fail-closed refusal) fails alone and the\n"
                      << "    process keeps answering.\n"
                      << "  - --evaluator-idle-timeout S stops the process after S seconds\n"
                      << "    with no new request.  It applies to a FILE request stream,\n"
                      << "    which the controller may append to; stdin stops at EOF.\n"
                      << "  - --evaluator-isolation-check re-runs each wave's FIRST deck once\n"
                      << "    more at the END of that wave and compares the two digests --\n"
                      << "    the A -> ... -> A cross-case leak test, run in production.\n"
                      << "  - --warm-start-from FILE seeds this case's BOC flux, critical\n"
                      << "    boron and k_eff from a parent case's saved warm state, which\n"
                      << "    shortens the `initial` outer bucket (7.5 % of a case).  It is\n"
                      << "    GATE N1: a different starting point may select a different\n"
                      << "    root, so the digest MAY move and the gate is keff/CBC/Fq/FdH\n"
                      << "    inside the acceptance thresholds -- not digest equality.  Any\n"
                      << "    refusal (missing file, wrong geometry, implausible seed)\n"
                      << "    degrades to a COLD start and says so in [RASBERY][WARMSTART].\n"
                      << "    Unset, nothing in that path runs.\n"
                      << "  - --save-warm-state [FILE] writes this case's BOC state for a\n"
                      << "    child to warm-start from.  With no FILE the path is derived\n"
                      << "    per job from its --raso (<dir>/<stem>.warm), so a batch cannot\n"
                      << "    have two decks writing one file.  It is independent of\n"
                      << "    --result: light writes no restart, and this is the channel\n"
                      << "    that replaces it for seeding.\n"
                      << "    Evaluator requests take the same two as \"warm_start_from\"\n"
                      << "    and \"save_warm_state\" per case.\n"
                      << "  - --statepoint-grid full|coarse|three|<GWd/t list> rewrites the\n"
                      << "    deck's burnup schedule at load (WP10.3).  `coarse` is the\n"
                      << "    measured 9-point grid (0.5,1,2,4,6,8,10,13,16); a list is\n"
                      << "    cumulative GWd/t past BOC.  Anything but `full` makes the run\n"
                      << "    L3coarse -- screening only, never acceptance -- and the deck\n"
                      << "    digest and case key are taken of the COARSE deck, so a\n"
                      << "    screening answer can never be served for a full request.\n"
                      << "    Evaluator requests take it per case as \"statepoint_grid\".\n";
            return 0;
        }

        if (option != "--chiffoni" && option != "--chiffono" &&
            option != "--rasi" && option != "--raso" &&
            option != "--isohgc" && option != "--isocsv" && option != "--validate" &&
            option != "--batch-mode" && option != "--jobs" && option != "--result") {
            std::cerr << "Unknown option: " << option << std::endl;
            return 1;
        }

        ++argi;
        if (argi >= argc || std::string(argv[argi]).rfind("--", 0) == 0) {
            std::cerr << "Missing value after option: " << option << std::endl;
            return 1;
        }

        while (argi < argc) {
            const std::string value = argv[argi];
            if (value.rfind("--", 0) == 0)
                break;

            if (option == "--chiffoni")
                chiffon_inputs.push_back(value);
            else if (option == "--chiffono")
                chiffon_outputs.push_back(value);
            else if (option == "--rasi")
                rasbery_inputs.push_back(value);
            else if (option == "--raso")
                rasbery_outputs.push_back(value);
            else if (option == "--isohgc")
                iso_hgcs.push_back(value);
            else if (option == "--validate")
                validate_args.push_back(value);
            else if (option == "--batch-mode")
                batch_width = std::max(0, std::atoi(value.c_str()));
            else if (option == "--jobs")
                job_manifests.push_back(value);
            else if (option == "--result") {
                if (!rasbery::ParseResultMode(value, result_mode)) {
                    std::cerr << "--result must be one of full | pin-off | light, got: "
                              << value << std::endl;
                    return 1;
                }
            }
            else
                iso_csv = value;

            ++argi;
        }
    }

    if (chiffon_inputs.size() != chiffon_outputs.size()) {
        std::cerr << "The number of --chiffoni and --chiffono paths must match." << std::endl;
        return 1;
    }

    // Manifest jobs are appended BEFORE every check below, so a manifest is
    // validated by exactly the rules an argv deck list is -- one namespace, one
    // counts-match test, one batch predicate.
    // argv decks take the flag's mode; a manifest line may override its own.
    rasbery_result_modes.assign(rasbery_inputs.size(), result_mode);
    for (const std::string& manifest : job_manifests) {
        std::string manifest_error;
        if (!rasberyReadJobManifest(manifest, rasbery_inputs, rasbery_outputs,
                                    rasbery_result_modes, result_mode, manifest_error)) {
            std::cerr << manifest_error << std::endl;
            return 1;
        }
    }

    if (rasbery_inputs.size() != rasbery_outputs.size()) {
        std::cerr << "The number of --rasi and --raso paths must match." << std::endl;
        return 1;
    }
    rasbery_result_modes.resize(rasbery_inputs.size(), result_mode);

    // Job namespace policy (plan Rev.4 Sec 7).  Repeating a --rasi deck is
    // ALLOWED and is how a batch sweeps states off one input file.  Repeating a
    // --raso path is not: the two Drivers would race inside one HDF5 file, and
    // since the restart namespace is now derived from the output path (see
    // Driver::RestartPath) they would collide on their restart files too.  The
    // launcher already refuses this; enforcing it here covers direct
    // invocations, which is where the GPU time actually gets spent.
    {
        std::map<std::string, std::size_t> seen_outputs;
        for (std::size_t i = 0; i < rasbery_outputs.size(); ++i) {
            const std::string key = rasberyPathKey(rasbery_outputs[i]);
            const auto [it, inserted] = seen_outputs.emplace(key, i);
            if (!inserted) {
                std::cerr << "--raso paths must be distinct, one per deck: entry "
                          << (it->second + 1) << " (" << rasbery_outputs[it->second]
                          << ") and entry " << (i + 1) << " (" << rasbery_outputs[i]
                          << ") resolve to the same file. Identical --rasi decks are fine; "
                             "identical outputs would overwrite each other and share a "
                             "restart namespace." << std::endl;
                return 1;
            }
        }
    }

    // WP10.2, and the SAME namespace rule one line down: an explicit
    // --save-warm-state path with more than one deck is N decks writing one
    // file -- N-1 lost states and an arbitrary winner.  Refused by name rather
    // than silently, because the run would otherwise look like it worked and
    // the child would warm-start from the wrong parent.
    if (save_warm_state && !warm_state_out.empty() && rasbery_outputs.size() > 1) {
        std::cerr << "--save-warm-state with an explicit path takes exactly one deck ("
                  << rasbery_outputs.size() << " given). Drop the path and each job "
                     "derives its own from its --raso." << std::endl;
        return 1;
    }

    // -----------------------------------------------------------------------
    // Execution mode, declared ONCE, here.
    //
    // This is the same predicate that selects the batch branch further down --
    // it is computed once and both sites use it, so "what the run does" and
    // "what the run says it does" cannot drift.  It has to be latched HERE,
    // before the [PHYSICS_MODE] receipt: the mode-dependent Anderson default
    // (Driver.h) resolves on first read and caches, and the receipt is the
    // first read.
    // -----------------------------------------------------------------------
    //
    // WP8 ADDS ONE PRODUCER, NOT A SECOND PREDICATE.  An evaluator process is a
    // batch process whose job list arrives later: it stands the same arenas up
    // at the same width and runs its cases through the same OpenMP queue, so it
    // must resolve the SAME mode-dependent defaults (the Anderson default is
    // single-ON / batch-OFF, Driver.h).  Declaring Single here and then running
    // waves would make every evaluator digest differ from the `--jobs
    // --batch-mode M` run it is supposed to reproduce bit for bit -- the B0
    // gate would fail for a reason that has nothing to do with the lifetime
    // change under test.  The evaluator branch below returns before
    // `if (batch_execution)`, so one predicate still selects exactly one path.
    const bool batch_execution = (batch_width > 0 && !rasbery_inputs.empty()) || evaluator_mode;
    rasbery::declareExecutionMode(batch_execution ? rasbery::ExecutionMode::Batch
                                                  : rasbery::ExecutionMode::Single);

    // -----------------------------------------------------------------------
    // Exact-only hard contract (plan Rev.4 Sec 2), as WP1 rewrote it
    // (BOTTLENECK plan Sec 6.2, priority P0).
    //
    // The campaign accepts full-exact results and nothing else, so an
    // approximation must be impossible to enable BY ACCIDENT.  A warning would
    // not do it: an inherited environment variable from a screening job is
    // exactly how a screening run ends up in an acceptance table, and nothing
    // downstream would know.  So this refuses to start, and
    // RASBERY_ALLOW_SCREENING=1 is the only way to get a screening run.
    //
    // WHAT WP1 CHANGED, AND WHY.  This used to read
    //
    //     const bool screening = (ga_feedback_passes > 0) || light_result;
    //
    // which made `--result light` -- a switch that changes only what LEAVES the
    // process -- declare the SOLVE approximate.  It is not: full, pin-off and
    // light run the same solve, the same PPR and the same feedback loops, and
    // the campaign measured all three to one trajectory digest
    // (814201df0583e1d2).  The cost was paid in both directions: a strict run
    // that wrote scalars was voided by the acceptance audit, and the harness
    // worked around that by EXPECTING the screening receipt for a light chunk,
    // so a receipt claiming screening was accepted for a run that was not.
    //
    // Screening is now a property of the FIDELITY alone (src/RunContract.h):
    // coarse statepoints or a GA feedback-pass limit.  What the case writes is
    // reported beside it and is never a reason to void a run.
    // -----------------------------------------------------------------------
    const int  ga_feedback_passes = rasbery::BatchLightResult::FeedbackPasses();
    // Still computed, because `full_hdf5` is still a fact about the run and the
    // feedback-pass limit is still coupled to the light writer below.  It is no
    // longer an input to `screening`.
    const bool any_light =
        std::any_of(rasbery_result_modes.begin(), rasbery_result_modes.end(),
                    [](rasbery::ResultMode m) { return m == rasbery::ResultMode::Light; });
    const bool light_result       = rasbery::BatchLightResult::Enabled() || any_light;

    // A declaration this build cannot parse is a declaration that silently did
    // not happen -- and the case it exists for, an L3 coarse DECK, is the one
    // fidelity nothing else can detect.  Refuse rather than report `strict`.
    if (rasbery::declaredPhysicsFidelityIsUnknown()) {
        std::cerr << "[RASBERY][EXACT_ONLY][FAIL] RASBERY_PHYSICS_FIDELITY=\""
                  << rasbery::declaredPhysicsFidelityText()
                  << "\" is not a fidelity this build knows. Use one of: ";
        for (int i = 0; i < 4; ++i)
            std::cerr << (i ? " | " : "") << rasbery::kFidelityTraits[i].policy;
        std::cerr << " (or the plan spellings: ";
        for (int i = 0; i < 4; ++i)
            std::cerr << (i ? " | " : "") << rasbery::kFidelityTraits[i].physics_fidelity;
        std::cerr << ")." << std::endl;
        return 2;
    }
    // WP10.3.  THE RUN'S FIDELITY, and it is no longer the environment's alone:
    // `--statepoint-grid` is a deck transform, so it is the ONE fidelity input
    // RunContract.h could never detect and had to be declared through
    // RASBERY_PHYSICS_FIDELITY.  Now the flag that causes it is the flag that
    // reports it, and an operator cannot run a coarse deck and forget to say so
    // -- which was the entire hole in the L3coarse lane.
    //
    // WP10.1 FOLLOW-UP -- ONE BUILDER, NOT TWO.  This used to be
    // `processCaseFidelity()` copied and then written into by hand:
    //
    //     run_fidelity.statepoint_grid =
    //         spgrid::isFullGrid(statepoint_grid) ? "" : statepoint_grid;
    //
    // which is a second spelling of resolveCaseFidelity's own grid clause
    // (CaseFidelity.h).  The two agreed, line for line -- and the whole point
    // of the case key is that the single-shot CLI and the evaluator describe
    // the same run with the same digest, so `they agree today` is not the
    // property to rely on.  Both paths reach Driver::setCaseFidelity through
    // resolveCaseFidelity now, with the request built here, so a future change
    // to how a grid resolves cannot reach one path and miss the other.
    //
    // THERE IS NO --fidelity FLAG, deliberately, so the request declares no
    // WORD: a fidelity the CLI could declare and not honour is the mixing plan
    // Sec 6.2 forbids, and with `fidelity` empty resolveCaseFidelity's
    // declaration clauses are all skipped and the result is the process default
    // plus the grid.  Byte for byte what the two lines above produced.
    rasbery::CaseFidelity run_fidelity;
    {
        rasbery::FidelityRequest cli_fidelity_request;
        cli_fidelity_request.has_grid        = true;
        cli_fidelity_request.statepoint_grid = statepoint_grid;
        std::string cli_fidelity_error;
        if (!rasbery::resolveCaseFidelity(cli_fidelity_request,
                                          rasbery::processCaseFidelity(), run_fidelity,
                                          cli_fidelity_error)) {
            std::cerr << cli_fidelity_error << std::endl;
            return 2;
        }
    }
    const rasbery::PhysicsFidelity fidelity = run_fidelity.solved();
    const bool screening          = rasbery::fidelityIsScreening(fidelity);
    const bool allow_screening    = [] {
        const char* value = std::getenv("RASBERY_ALLOW_SCREENING");
        if (value == nullptr || *value == '\0') return false;
        const std::string requested(value);
        return !(requested == "0" || requested == "off" || requested == "OFF" ||
                 requested == "false" || requested == "FALSE");
    }();

    if (screening && !allow_screening) {
        std::cerr << "[RASBERY][EXACT_ONLY][FAIL] this build runs full-exact physics only.\n"
                  << "  policy=" << rasbery::physicsPolicyName(fidelity)
                  << " (required: strict)\n"
                  << "  RASBERY_GA_FEEDBACK_PASSES=" << ga_feedback_passes
                  << " (required: 0 or unset)\n"
                  << "  RASBERY_PHYSICS_FIDELITY="
                  << (rasbery::declaredPhysicsFidelityText() != nullptr
                          ? rasbery::declaredPhysicsFidelityText()
                          : "<unset>")
                  << " (a declaration can only make the run COARSER)\n"
                  << "  NOTE: --result light / RASBERY_BATCH_LIGHT_RESULT is an OUTPUT "
                     "mode and is NOT why this refused; the same solve and the same "
                     "trajectory digest come out of full, pin-off and light.\n"
                  << "  Unset the approximations to run exact, or set "
                     "RASBERY_ALLOW_SCREENING=1 to run a screening job on purpose. "
                     "Screening results are never acceptance results."
                  << std::endl;
        return 2;
    }

    if (ga_feedback_passes > 0 && !light_result) {
        std::cerr << "RASBERY_GA_FEEDBACK_PASSES is a GA screening approximation and "
                     "requires RASBERY_BATCH_LIGHT_RESULT=1. Rerun selected candidates "
                     "with RASBERY_GA_FEEDBACK_PASSES unset for acceptance."
                  << std::endl;
        return 2;
    }

    // Machine-readable physics-mode receipt, emitted by EVERY run before any
    // deck starts (Sec 2.2).  The benchmark parser voids a run whose receipt is
    // missing or whose fields disagree with full-exact.
    //
    // `physics_mode` KEEPS ITS LEGACY VOCABULARY.  Every stored 238 arm --
    // strict and A2 alike -- was measured against the string
    // "full_exact_nodal", and the harness audits that string, so repointing
    // this field at the new policy words would void every manifest on disk.
    // The one new value is coarse10's, which no stored arm can carry because
    // the binary could not report a coarse deck before WP1.  The FIDELITY lives
    // in `policy`/`physics_fidelity` below, which is what a WP1-aware audit
    // (tools/exact_audit.py) reads.
    const char* const legacy_physics_mode =
        (fidelity == rasbery::PhysicsFidelity::FeedbackLimited) ? "ga_screen_feedback_limited"
        : (fidelity == rasbery::PhysicsFidelity::Coarse10State) ? "ga_screen_coarse10"
                                                                : "full_exact_nodal";
    std::cout << "[RASBERY][PHYSICS_MODE] {\"physics_mode\":\""
              << legacy_physics_mode
              << "\",\"screening\":" << (screening ? "true" : "false")
              << ",\"feedback_pass_limit\":" << ga_feedback_passes
              << ",\"full_hdf5\":" << (light_result ? "false" : "true")
              // Additive field: which in-core xenon treatment this run used
              // (RASBERY_XE_MODE).  "equilibrium" is the default and the only
              // value the exact-only acceptance path produces; "frozen" marks a
              // same-mode MASTER comparison run, not an acceptance measurement.
              << ",\"xe_mode\":\"" << rasbery::xeModeName() << "\""
              // Additive fields (adoption 2026-08-27).  The Anderson default is
              // MODE-DEPENDENT (single ON, batch OFF), so the state alone does
              // not identify a run: exec_mode says which default applied and
              // xe_anderson_source says whether the default is what actually
              // decided it.  A comparison whose two arms differ here is not an
              // A/B of the thing it claims to be.
              << ",\"exec_mode\":\"" << rasbery::executionModeName() << "\""
              << ",\"xe_anderson\":" << (rasbery::xeAnderson() ? "true" : "false")
              << ",\"xe_anderson_source\":\"" << rasbery::xeAndersonSourceName() << "\""
              // Additive field: what the jobs of this run were asked to WRITE.
              // "mixed" is a legitimate value -- one wave, light population,
              // full elites -- and is why this is not a boolean.
              << ",\"result_mode\":\"" << rasberyResultModeSummary(rasbery_result_modes)
              << "\""
              // ---- WP1 (BOTTLENECK plan Sec 6.2): THE FIDELITY AXIS --------
              // The three fields above this line describe what the run WRITES.
              // The four below describe how it SOLVES, and they are the ones an
              // acceptance audit keys on (tools/exact_audit.py).  They are
              // additive: `physics_mode`, `screening`, `feedback_pass_limit`
              // and `full_hdf5` keep the vocabulary every stored 238 arm was
              // measured with, so no existing manifest is invalidated -- what
              // changed is that `physics_mode`/`screening` no longer flip
              // because a job wrote scalars.
              << ",\"physics_fidelity\":\"" << rasbery::physicsFidelityName(fidelity)
              // WP10.3.  The L3coarse lane's input, named.  `full` on every run
              // that did not ask for a grid, so an audit can REQUIRE the field
              // rather than treat its absence as "probably full".
              << "\",\"statepoint_grid\":\"" << run_fidelity.gridToken()
              << "\",\"policy\":\"" << rasbery::physicsPolicyName(fidelity)
              << "\",\"acceptance_eligible\":"
              << (rasbery::fidelityIsAcceptanceEligible(fidelity) ? "true" : "false")
              << ",\"requires_exact_rerun\":"
              << (rasbery::fidelityRequiresExactRerun(fidelity) ? "true" : "false")
              // Raw, so a declaration that changed nothing is still visible: it
              // can only ever make the effective fidelity COARSER.
              << ",\"fidelity_declared\":"
              << (rasbery::declaredPhysicsFidelityText() != nullptr
                      ? "\"" + std::string(rasbery::declaredPhysicsFidelityText()) + "\""
                      : std::string("null"))
              // ---- WP1 (plan Sec 6.3): was the fail-closed gate on? --------
              << ",\"gpu_full\":"
              << (rasbery::gpufull::required() ? "true" : "false")
              << "}" << std::endl;

    if (screening) {
        std::cout << "[RASBERY][GA][SCREEN] {\"physics_mode\":\""
                  << legacy_physics_mode << "\",\"policy\":\""
                  << rasbery::physicsPolicyName(fidelity)
                  << "\",\"feedback_passes\":" << ga_feedback_passes
                  << ",\"requires_exact_rerun\":true}" << std::endl;
    }

    // An evaluator has no jobs on argv BY DESIGN -- they arrive on the request
    // stream -- so it is the one mode that may start with an empty deck list.
    if (chiffon_inputs.empty() && rasbery_inputs.empty() && iso_csv.empty() &&
        validate_args.empty() && !evaluator_mode) {
        std::cerr << "No jobs were provided. Use --help for usage." << std::endl;
        return 1;
    }

    Chiffon::Isotope::Initialize(repo_dir / "include" / "Database");

    // Offline isotope-state export for the correlation/VIF study (Benchmark owns the I/O).
    if (!iso_csv.empty()) {
        Chiffon::Benchmark::DumpIsotopeStates(iso_hgcs, iso_csv);
        return 0;
    }

    // Interpolation-validation CSV dump (fine vs coarse vs CHIFFON interpolation) for plotting.
    if (!validate_args.empty()) {
        if (validate_args.size() < 2) {
            std::cerr << "Usage: --validate <input.json> <out.csv> [fineKey coarseKey]" << std::endl;
            return 1;
        }
        const fs::path    in_path   = validate_args[0];
        const std::string out_csv   = validate_args[1];
        const std::string fineKey   = validate_args.size() > 2 ? validate_args[2] : "FN";
        const std::string coarseKey = validate_args.size() > 3 ? validate_args[3] : "CR";

        std::vector<Chiffon::Model> models;
        Chiffon::Importer           importer;
        importer.ReadInput(in_path, models);

        const Chiffon::Model* fine   = nullptr;
        const Chiffon::Model* coarse = nullptr;
        for (const auto& m : models) {
            if (m.name() == fineKey) fine = &m;
            if (m.name() == coarseKey) coarse = &m;
        }
        if (fine == nullptr || coarse == nullptr) {
            std::cerr << "[validate] could not find models '" << fineKey << "' / '" << coarseKey
                      << "' in " << in_path.string() << std::endl;
            return 1;
        }
        std::cout << "[validate] " << in_path.string() << "  fine=" << fineKey
                  << " coarse=" << coarseKey << " -> " << out_csv << std::endl;
        Chiffon::Benchmark::ValidateCSV(*fine, *coarse, out_csv);
        return 0;
    }

    for (std::size_t i = 0; i < chiffon_inputs.size(); ++i) {
        const fs::path chiffon_input_path  = chiffon_inputs[i];
        const fs::path chiffon_output_path = chiffon_outputs[i];

        std::cout << "\n[CHIFFON] " << chiffon_input_path.string()
                  << " -> " << chiffon_output_path.string() << std::endl;

        if (chiffon_output_path.has_parent_path())
            fs::create_directories(chiffon_output_path.parent_path());

        std::vector<Chiffon::Model> models;
        Chiffon::Importer           importer;

        importer.ReadInput(chiffon_input_path, models);
        Chiffon::Exporter::SaveHDF(chiffon_output_path.string(), models);
    }

    int exit_code = 0;

    // -----------------------------------------------------------------------
    // WP8 stage 1 -- the long-lived evaluator.
    //
    // Placed HERE, and not earlier, because everything above it is the part of
    // a run that must not change: the exact-only gate, the [PHYSICS_MODE]
    // receipt, the isotope registry and the depletion chain.  An evaluator is a
    // batch process whose job list arrives on a stream instead of on argv, so
    // it enters with the same execution mode declared, the same host-pinning
    // decision and the same I/O writer configuration, and it leaves through the
    // same teardown order (plan Sec 6.6): every Driver gone, the writer drained
    // and joined, the arena released, then the pin registry.
    //
    // The ONE difference is when those last three happen -- once, at shutdown,
    // instead of once per process image per chunk -- and that difference is the
    // whole work package.
    // -----------------------------------------------------------------------
    if (evaluator_mode) {
        if (batch_width <= 0) {
            std::cerr << "--evaluator requires --batch-mode M: the arena is one allocation "
                         "sized at the first admission and fixed for the process, so the "
                         "width has to be known before the first request. Its comparand is a "
                         "`--jobs ... --batch-mode M` run at the same M."
                      << std::endl;
            return 1;
        }
        if (!rasbery_inputs.empty()) {
            std::cerr << "--evaluator takes its jobs from the request stream, not from --rasi/"
                         "--jobs. " << rasbery_inputs.size()
                      << " deck(s) were also given on the command line; they would run under "
                         "a different job-namespace rule than the waves. Feed them as "
                         "{\"op\":\"case\"} lines or a {\"op\":\"wave\",\"jobs_manifest\":...} "
                         "instead."
                      << std::endl;
            return 1;
        }

        rasbery::evaluator::Options options;
        options.request_path        = evaluator_request_path;
        options.batch_width         = batch_width;
        options.default_result_mode = result_mode;
        options.idle_timeout_s      = evaluator_idle_timeout;
        options.isolation_check     = evaluator_isolation_check;
        options.visible_cpus        = startup_visible_cpus;
        if (const char* host_env = std::getenv("RASBERY_BATCH_HOST_THREADS"))
            options.host_threads_override = std::atoi(host_env);
        // ONE manifest grammar.  The evaluator does not reimplement the `--jobs`
        // reader; it is handed the same function, so the quoting, the optional
        // third result-mode field, the CRLF strip and the line-numbered errors
        // have exactly one definition.
        options.read_manifest = rasberyReadJobManifest;

#ifdef _OPENMP
    #ifndef _MSC_VER
        omp_set_max_active_levels(1);
    #else
        omp_set_nested(0);
    #endif
#endif
        rasbery::iowriter::reportConfig(std::cout);

        rasbery::evaluator::Server server(options, std::cout);
        pre_drive_seconds = since_start(std::chrono::steady_clock::now());
        exit_code         = server.run();
        drive_end         = std::chrono::steady_clock::now();

        // Same order, same tags, same reasons as the batch branch below: every
        // Driver has joined, so the writer queue can only shrink; then the
        // arena; then the leases that outlived it.  This is the ONLY place the
        // evaluator tears any of it down.
        const std::uint64_t io_writer_failures = rasbery::iowriter::shutdown();
        if (io_writer_failures > 0 && exit_code == 0) exit_code = 1;
        rasbery::rasberyReleaseBatchArena();
        server.stampArenaRelease();
        rasbery::rasberyDrainPinnedRegistry();

        std::cout << "[RASBERY][BATCH_HOST][PIN] {";
        rasbery::rasberyAppendHostPinReceiptFields(std::cout);
        std::cout << "}" << std::endl;
        std::cout << "[RASBERY][GPU_FULL] {";
        rasbery::gpufull::appendReceiptFields(std::cout);
        std::cout << "}" << std::endl;
        // WP1 follow-up: `contract_pass:false` IS the failure, not a note about
        // one.  Every Driver has joined, so the counters are final and the
        // first-violation text is safe to read.  See gpufull::enforceExitCode.
        exit_code = rasbery::gpufull::enforceExitCode(std::cout, exit_code);
        if (rasbery::rasberyGpuXsReconEnabled())
            std::cout << "[RASBERY][XSRECON][GPU] {\"nodes_solved\":"
                      << rasbery::rasberyGpuXsReconNodes() << "}" << std::endl;
        if (rasbery::rasberyGpuFlatXsEnabled())
            std::cout << "[RASBERY][FLATXS][GPU] {\"nodes_solved\":"
                      << rasbery::rasberyGpuFlatXsNodes() << "}" << std::endl;
        if (rasbery::xfer::receiptWanted()) {
            std::cout << "[RASBERY][XFER] {";
            rasbery::xfer::appendXferReceiptFields(std::cout);
            std::cout << "}" << std::endl;
        }
        if (rasbery::rasberyGpuXeEnabled()) {
            std::cout << "[RASBERY][XE_GPU] {";
            rasbery::xe::appendXeGpuReceiptFields(std::cout);
            std::cout << ",\"fuel_node_evaluations\":" << rasbery::rasberyGpuXeEvaluations()
                      << ",\"fuel_node_commits\":" << rasbery::rasberyGpuXeCommits()
                      << ",\"dot_partitions\":" << rasbery::rasberyGpuXeDotPartitions()
                      << "}" << std::endl;
        }
        if (rasbery::rasberyGpuNodalEnabled()) {
            std::cout << "[RASBERY][NODAL][GPU] {\"drives_solved\":"
                      << rasbery::rasberyGpuNodalDrives() << "}" << std::endl;
            std::cout << "[RASBERY][NODAL][CANON] {\"elided_upload_bytes\":"
                      << rasbery::rasberyGpuNodalCanonicalElidedUploadBytes()
                      << ",\"elided_download_bytes\":"
                      << rasbery::rasberyGpuNodalCanonicalElidedDownloadBytes()
                      << "}" << std::endl;
        }
        rasbery::gpu::reportOuterSegment(std::cout);
        rasbery::xsphase::report(std::cout);
        rasbery::outer_timing::report(std::cout);
        const auto evaluator_hdf5_stats = Chiffon::GetHdf5LockStats();
        std::cout << "[RASBERY][HDF5][LOCK] {\"acquires\":"
                  << evaluator_hdf5_stats.acquisitions << ",\"wait_ms\":"
                  << static_cast<double>(evaluator_hdf5_stats.wait_nanoseconds) / 1.0e6
                  << "}" << std::endl;
        // THE FIELD THE WHOLE WORK PACKAGE IS JUDGED ON: loads must be the
        // number of distinct library CONTENTS this process saw, and must not
        // grow with the case count or the wave count.
        rasbery::PrintXsLibraryCacheReceipt(std::cout);
        // WP8 stage 2's equivalent for geometry.  Printed in both branches, and
        // beside the XSLIB receipt on purpose: the two answer the same shape of
        // question about the two halves of what a cohort is.
        rasbery::cohort::printReceipt(std::cout);
        rasberyPrintProcessLedger(std::cout, exec_seconds, pre_drive_seconds,
                                  since_start(drive_end) - pre_drive_seconds,
                                  since_start(std::chrono::steady_clock::now()) -
                                      since_start(drive_end),
                                  static_cast<int>(server.summary().cases));
        // AFTER PrintXsLibraryCacheReceipt and after the arena release, so its
        // xslib and arena_releases fields are final.
        server.reportProcess(std::cout);
        rasbery::iowriter::reportSummary(std::cout);
        // WP12: the other half of io_wall -- the pin-power CSV serialisation,
        // and whether it ran on the solver thread or the writer thread.
        rasbery::iowriter::reportResultIo(std::cout);
        return exit_code;
    }

    // -----------------------------------------------------------------------
    // Multi-instance batch mode.
    //
    // The DB/RL workload is "same core, many states", and a single state point
    // cannot fill a modern GPU: the measured CMFD system (nxyz ~ 3e3) leaves a
    // 188-SM device idle, which is why per-process GPU offload lost to the CPU
    // on throughput.  Batch mode fixes the occupancy problem at its root by
    // running M instances in one process and folding their CMFD solves into a
    // single grid (see CudaBatchArena).
    //
    // Everything above the linear solve stays exactly as it is: each instance
    // is a plain Driver on its own host thread, running its own control flow at
    // its own pace.  That is deliberate.  Lock-stepping the instances would
    // have meant rewriting Driver/Scheduler and would have bought nothing --
    // the batched solver does not care who rides along, because per-instance
    // results do not depend on the batch composition.
    // -----------------------------------------------------------------------
    if (batch_execution) {
        rasbery::rasberySetBatchWidth(batch_width);
        // Same width, published to the nodal arm.  The two arenas are separate
        // (different phases, different rendezvous), but they are the same M.
        rasbery::rasberyNodalSetBatchWidth(batch_width);
        const int jobs = static_cast<int>(rasbery_inputs.size());
        int host_threads = std::min(batch_width, jobs);
#ifdef _OPENMP
        const int visible_cpus = startup_visible_cpus;
        // The CUDA arena width and the number of CPU Driver workers are separate
        // resources.  Use an explicit, benchmarked cap because OMP_PROC_BIND can
        // narrow the main-thread affinity before this branch and make automatic
        // CPU-count discovery platform/runtime dependent.  With no override the
        // historical one-worker-per-live-instance behavior is preserved.
        if (const char* host_env = std::getenv("RASBERY_BATCH_HOST_THREADS")) {
            const int requested = std::atoi(host_env);
            if (requested > 0) host_threads = std::min({requested, batch_width, jobs});
        }
        const int concurrent_workers = host_threads;
#else
        const int visible_cpus       = 1;
        // No OpenMP: the instance loop below is a plain serial for, so every deck
        // after the first runs on a thread that has already torn one down.
        const int concurrent_workers = 1;
#endif
        // Host page-locking used to be permanent and un-undoable, so it was
        // only sound while every Driver outlived the run: with fewer workers
        // than decks the OpenMP queue recycles a worker onto a second deck,
        // whose Geometry/XSSet/BICGCMFD land on the freed -- and still
        // registered -- addresses of the deck that just finished.  That
        // aliasing is what turned `--batch-mode 64` with 24 workers into 54
        // failed decks, and `workers >= jobs` was the criterion that avoided it
        // by giving up the pinning instead.
        //
        // HostPinRegistry.h closes the lifecycle hole itself: every
        // registration is leased, the owner releases it in its destructor
        // before the memory is freed, and cudaHostUnregister runs at the
        // registered address.  A recycled worker therefore hands the next deck
        // an EMPTY registry, which is the state in which registering afresh is
        // correct.  So under `auto` the gate is simply "not off" -- the
        // worker/deck ratio no longer decides it.  The legacy criterion is kept
        // below for the receipt (and is what to fall back to if the lease is
        // ever compiled out).
        const bool legacy_pinning_criterion = (concurrent_workers >= jobs);
        const bool host_pinning =
            (rasbery::rasberyHostPinningMode() != rasbery::HostPinningMode::Off);
        rasbery::rasberySetHostPinningEnabled(host_pinning);
        std::cout << "\n[RASBERY][BATCH] " << jobs << " deck(s), width " << batch_width
                  << ", " << host_threads << " host Driver worker(s)" << std::endl;
        std::cout << "[RASBERY][BATCH_HOST] {\"jobs\":" << jobs
                  << ",\"arena_width\":" << batch_width
                  << ",\"host_threads\":" << host_threads
                  << ",\"visible_cpus\":" << visible_cpus
                  << ",\"host_pinning\":" << (host_pinning ? "true" : "false")
                  << ",\"pin_lease\":true"
                  << ",\"legacy_pinning_criterion\":"
                  << (legacy_pinning_criterion ? "true" : "false") << "}" << std::endl;
        // Which I/O path this run is on, published BEFORE the first deck for the
        // same reason [BATCH_HOST] is: it is the declared configuration, and an
        // A/B whose two arms cannot be told apart from the log is void.
        rasbery::iowriter::reportConfig(std::cout);
        if (host_pinning && !legacy_pinning_criterion)
            std::cout << "[RASBERY][BATCH_HOST] concurrent Driver workers("
                      << concurrent_workers << ") < jobs(" << jobs
                      << "): workers are recycled onto later decks. Host page-locking stays ON "
                         "-- every registration is leased and released by its owner's "
                         "destructor (RASBERY_HOST_PINNING=off to force pageable copies)."
                      << std::endl;
#ifdef _OPENMP
        // One OpenMP level only: the instance loop is the parallelism. Nested
        // Driver regions reduce the measured GPU rendezvous width and lose
        // aggregate throughput even when total CPU threads are held constant.
#ifndef _MSC_VER
        omp_set_max_active_levels(1);
#else
        omp_set_nested(0); // MSVC omp.h lacks omp_set_max_active_levels; nested-off is the same contract
#endif
        omp_set_num_threads(host_threads);
#endif
        std::vector<int> job_status(static_cast<std::size_t>(jobs), 0);
        std::vector<std::string> job_error(static_cast<std::size_t>(jobs));

        // Rev.7.1 Task 20.  THE REFILL IS THIS LOOP, and it always was: the
        // OpenMP queue is dynamic with a chunk of one, so a worker that
        // finishes a deck takes the next job immediately rather than waiting
        // for its siblings.  The Driver's destructor releases the arena slot
        // (CudaBatchArena::releaseSlot, NodalArena::releaseSlot), which drops
        // inUseCount() and wakes the rendezvous, so the remaining decks stop
        // waiting on a slot that is between tenants; the next Driver's
        // constructor acquires a slot again and gets a full reset with it.
        // The batch is never drained in between.
        //
        // What the ledger adds is the arithmetic: how many admissions reused a
        // lane, how long each lane actually held a deck, and how long the lanes
        // sat empty at the end.  Without it "the tail went away" is a claim
        // about a stopwatch.
        rasbery::refill::ledger().begin(jobs, batch_width, host_threads);
        pre_drive_seconds = since_start(std::chrono::steady_clock::now());

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(host_threads)
#endif
        for (int i = 0; i < jobs; ++i) {
#ifdef _OPENMP
            const int lane = omp_get_thread_num();
#else
            const int lane = 0;
#endif
            rasbery::refill::ledger().jobStarted(i, lane);
            // An escaping exception would terminate the whole parallel region,
            // taking the other instances' partial results with it.  One bad
            // deck must only fail its own job.
            try {
                // The Driver is scoped so its destructor -- which is what
                // releases the slot -- runs BEFORE jobFinished stamps the
                // tenancy end.  Otherwise the refill latency measured below
                // would exclude the teardown, which is exactly the part of the
                // refill this task has to keep small.
                {
                    rasbery::Driver driver(rasbery_inputs[static_cast<std::size_t>(i)],
                                           rasbery_outputs[static_cast<std::size_t>(i)],
                                           rasbery_result_modes[static_cast<std::size_t>(i)]);
                    driver.setWarmStart(
                        warm_start_from,
                        WarmStatePath(save_warm_state, warm_state_out,
                                      rasbery_outputs[static_cast<std::size_t>(i)]));
                    driver.setCaseFidelity(run_fidelity);
                    job_status[static_cast<std::size_t>(i)] = driver.Drive();
                }
            } catch (const std::exception& error) {
                job_status[static_cast<std::size_t>(i)] = 1;
                job_error[static_cast<std::size_t>(i)]  = error.what();
            } catch (...) {
                job_status[static_cast<std::size_t>(i)] = 1;
                job_error[static_cast<std::size_t>(i)]  = "unknown exception";
            }
            rasbery::refill::ledger().jobFinished(i);
        }
        drive_end = std::chrono::steady_clock::now();
        rasbery::refill::ledger().end();

        // Same rule as the CUDA teardown below: every Driver has joined, so the
        // writer queue can only shrink from here.  Drain and join it BEFORE the
        // per-job verdicts are read, so a write that failed after its Driver
        // returned still reaches this exit code instead of a static destructor.
        const std::uint64_t io_writer_failures = rasbery::iowriter::shutdown();
        if (io_writer_failures > 0 && exit_code == 0) exit_code = 1;

        for (int i = 0; i < jobs; ++i) {
            if (job_status[static_cast<std::size_t>(i)] == 0) continue;
            std::cout << "[RASBERY][FAIL] exit_code=" << job_status[static_cast<std::size_t>(i)]
                      << " path=" << rasbery_inputs[static_cast<std::size_t>(i)]
                      << (job_error[static_cast<std::size_t>(i)].empty()
                              ? std::string()
                              : " what=" + job_error[static_cast<std::size_t>(i)])
                      << std::endl;
            if (exit_code == 0) exit_code = job_status[static_cast<std::size_t>(i)];
        }

        // Plan Sec 6.6, in order: every Driver is gone (the parallel region
        // joined above), the arena drains and tears down the backend streams,
        // and only then does the pin registry unregister what is left.  CUDA
        // teardown is never left to a function-local static destructor.
        rasbery::rasberyReleaseBatchArena();
        rasbery::rasberyDrainPinnedRegistry();
        // The Sec 6.8 pinning receipt.  It extends the BATCH_HOST family rather
        // than the BATCH_HOST line itself because the counters are only final
        // here, at the end of the run, while that line reports the requested
        // configuration before the first deck starts.  The serial branch emits
        // the same tag, so one parser rule covers both.
        std::cout << "[RASBERY][BATCH_HOST][PIN] {";
        rasbery::rasberyAppendHostPinReceiptFields(std::cout);
        std::cout << "}" << std::endl;
        // WP1 (plan Sec 6.3).  Printed WHETHER OR NOT the gate is on, and in
        // both branches with one tag, for the same reason [BATCH_HOST][PIN] is:
        // "the arm was on and never engaged" must not look like "the arm was
        // off".  contract_pass is the whole receipt in one boolean.
        std::cout << "[RASBERY][GPU_FULL] {";
        rasbery::gpufull::appendReceiptFields(std::cout);
        std::cout << "}" << std::endl;
        // WP1 follow-up: `contract_pass:false` IS the failure, not a note about
        // one.  Every Driver has joined, so the counters are final and the
        // first-violation text is safe to read.  See gpufull::enforceExitCode.
        exit_code = rasbery::gpufull::enforceExitCode(std::cout, exit_code);
        // Rev.7.1 Task 20.  Printed HERE, after the arena has been released, so
        // the tenancy counters it carries are final: releaseSlot runs in the
        // Driver destructors (all joined) and the arena teardown above is the
        // last thing that can touch a slot.
        rasbery::refill::ledger().report(std::cout);
        // Receipt for the xsrecon device path: a zero here means it never ran,
        // whatever the flag said, and an A/B built on it is void (G0).
        if (rasbery::rasberyGpuXsReconEnabled())
            std::cout << "[RASBERY][XSRECON][GPU] {\"nodes_solved\":"
                      << rasbery::rasberyGpuXsReconNodes() << "}" << std::endl;
        if (rasbery::rasberyGpuFlatXsEnabled())
            std::cout << "[RASBERY][FLATXS][GPU] {\"nodes_solved\":"
                      << rasbery::rasberyGpuFlatXsNodes() << "}" << std::endl;
        if (rasbery::xfer::receiptWanted()) {
            std::cout << "[RASBERY][XFER] {";
            rasbery::xfer::appendXferReceiptFields(std::cout);
            std::cout << "}" << std::endl;
        }
        // Rev.7.1 Task 13.  device_updates == 0 with the flag set means the arm
        // never ran and every A/B measured against this run is void (G0); the
        // node counts are the same claim at fuel-node granularity.
        if (rasbery::rasberyGpuXeEnabled()) {
            std::cout << "[RASBERY][XE_GPU] {";
            rasbery::xe::appendXeGpuReceiptFields(std::cout);
            std::cout << ",\"fuel_node_evaluations\":"
                      << rasbery::rasberyGpuXeEvaluations()
                      << ",\"fuel_node_commits\":" << rasbery::rasberyGpuXeCommits()
                      << ",\"dot_partitions\":" << rasbery::rasberyGpuXeDotPartitions()
                      << "}" << std::endl;
        }
        if (rasbery::rasberyGpuNodalEnabled()) {
            std::cout << "[RASBERY][NODAL][GPU] {\"drives_solved\":"
                      << rasbery::rasberyGpuNodalDrives() << "}" << std::endl;
            // Rev.7.1 Task 18-lite.  The transfers the canonical nodal binding
            // kept off the bus -- four per drive inside a device outer segment
            // (jnet and flux up, jnet and phis back), and invisible in the
            // segment's own receipt because the segment does not issue them.
            std::cout << "[RASBERY][NODAL][CANON] {\"elided_upload_bytes\":"
                      << rasbery::rasberyGpuNodalCanonicalElidedUploadBytes()
                      << ",\"elided_download_bytes\":"
                      << rasbery::rasberyGpuNodalCanonicalElidedDownloadBytes()
                      << "}" << std::endl;
        }
        // Rev.7.1 Task 9.  Printed whenever RASBERY_GPU_OUTER was set, even when
        // every segment was refused: "on and never engaged" must not look like
        // "off" (the same G0 rule the three receipts above exist for).
        rasbery::gpu::reportOuterSegment(std::cout);
        rasbery::xsphase::report(std::cout);
        rasbery::outer_timing::report(std::cout);
        const auto hdf5_stats = Chiffon::GetHdf5LockStats();
        std::cout << "[RASBERY][HDF5][LOCK] {\"acquires\":"
                  << hdf5_stats.acquisitions << ",\"wait_ms\":"
                  << static_cast<double>(hdf5_stats.wait_nanoseconds) / 1.0e6
                  << "}" << std::endl;
        // Read beside [HDF5][LOCK]: what that lock used to hold was this parse,
        // once per deck (XsLibrary.h).  loads == 1 with hits == jobs - 1 is the
        // shape the batch is supposed to have.
        rasbery::PrintXsLibraryCacheReceipt(std::cout);
        // WP8 stage 2, same shape of claim for geometry: a batch of M decks
        // over one core is builds=1, hits=M-1.
        rasbery::cohort::printReceipt(std::cout);
        rasberyPrintProcessLedger(std::cout, exec_seconds, pre_drive_seconds,
                                  since_start(drive_end) - pre_drive_seconds,
                                  since_start(std::chrono::steady_clock::now()) -
                                      since_start(drive_end),
                                  static_cast<int>(rasbery_inputs.size()));
        // Final writer counters.  Read together with [HDF5][LOCK] above, but
        // note the ACQUISITION count does not move: inline already took the
        // guard once per write function.  What the thread path changes is who
        // waits -- so the signals here are enqueue_block_ms (what the Drivers
        // still pay) and writer_busy_ms (whether the writer is the new floor).
        rasbery::iowriter::reportSummary(std::cout);
        // WP12: the other half of io_wall -- the pin-power CSV serialisation,
        // and whether it ran on the solver thread or the writer thread.
        rasbery::iowriter::reportResultIo(std::cout);
        return exit_code;
    }

    // Same rule as the batch branch above.  This loop destroys each Driver
    // before building the next one, which used to make page-locking unsafe for
    // more than one deck: the permanent registrations pinHost left behind were
    // inherited by whatever the allocator put at those addresses next.  With
    // the lease (HostPinRegistry.h) each Driver's destructors unregister what
    // they registered, so the deck count no longer decides this -- only
    // RASBERY_HOST_PINNING does.
    rasbery::rasberySetHostPinningEnabled(rasbery::rasberyHostPinningMode() !=
                                          rasbery::HostPinningMode::Off);
    // Same tag in both branches, deliberately: one parser rule.
    if (!rasbery_inputs.empty()) rasbery::iowriter::reportConfig(std::cout);

    pre_drive_seconds = since_start(std::chrono::steady_clock::now());
    for (std::size_t i = 0; i < rasbery_inputs.size(); ++i) {
        const fs::path rasbery_input_path  = rasbery_inputs[i];
        const fs::path rasbery_output_path = rasbery_outputs[i];

        std::cout << "\n[RASBERY] " << rasbery_input_path.string()
                  << " -> " << rasbery_output_path.string() << std::endl;

        // Same rule as the batch branch above: one bad deck must fail only its
        // own job.  This loop used to let an exception escape main() -- an
        // unwritable output directory aborted the process and took every deck
        // AFTER the bad one with it, which is exactly the collateral the batch
        // branch already refuses.  Reported for every failure, not just the
        // first, so a multi-deck run's log names all of them.
        int         driver_exit_code = 0;
        std::string driver_error;
        try {
            rasbery::Driver driver(rasbery_input_path.string(), rasbery_output_path.string(),
                                   rasbery_result_modes[i]);
            driver.setWarmStart(warm_start_from,
                                WarmStatePath(save_warm_state, warm_state_out,
                                              rasbery_output_path.string()));
            driver.setCaseFidelity(run_fidelity);
            driver_exit_code = driver.Drive();
        } catch (const std::exception& error) {
            driver_exit_code = 1;
            driver_error     = error.what();
        } catch (...) {
            driver_exit_code = 1;
            driver_error     = "unknown exception";
        }
        if (driver_exit_code != 0) {
            std::cout << "[RASBERY][FAIL] exit_code=" << driver_exit_code
                      << " path=" << rasbery_input_path.string()
                      << (driver_error.empty() ? std::string() : " what=" + driver_error)
                      << std::endl;
            if (exit_code == 0) exit_code = driver_exit_code;
        }
    }

    // Same explicit shutdown order as the batch branch (plan Sec 6.6): the
    // Drivers are gone, so release the leases that outlived them before the
    // CUDA context does.  Same receipt tag too, deliberately: one parser rule.
    if (rasbery::iowriter::shutdown() > 0 && exit_code == 0) exit_code = 1;
    rasbery::rasberyDrainPinnedRegistry();
    std::cout << "[RASBERY][BATCH_HOST][PIN] {";
    rasbery::rasberyAppendHostPinReceiptFields(std::cout);
    std::cout << "}" << std::endl;
    // Same tag in both branches, deliberately: one parser rule.  See the batch
    // arm above for why it is unconditional.
    std::cout << "[RASBERY][GPU_FULL] {";
    rasbery::gpufull::appendReceiptFields(std::cout);
    std::cout << "}" << std::endl;
    // See the batch arm above: the run-level half of the fail-closed gate.
    exit_code = rasbery::gpufull::enforceExitCode(std::cout, exit_code);

    if (rasbery::rasberyGpuXsReconEnabled())
        std::cout << "[RASBERY][XSRECON][GPU] {\"nodes_solved\":"
                  << rasbery::rasberyGpuXsReconNodes() << "}" << std::endl;
    if (rasbery::rasberyGpuFlatXsEnabled())
        std::cout << "[RASBERY][FLATXS][GPU] {\"nodes_solved\":"
                  << rasbery::rasberyGpuFlatXsNodes() << "}" << std::endl;
    if (rasbery::xfer::receiptWanted()) {
        std::cout << "[RASBERY][XFER] {";
        rasbery::xfer::appendXferReceiptFields(std::cout);
        std::cout << "}" << std::endl;
    }
    // See the batch arm above: the Task 13 device Xe receipt, same fields.
    if (rasbery::rasberyGpuXeEnabled()) {
        std::cout << "[RASBERY][XE_GPU] {";
        rasbery::xe::appendXeGpuReceiptFields(std::cout);
        std::cout << ",\"fuel_node_evaluations\":" << rasbery::rasberyGpuXeEvaluations()
                  << ",\"fuel_node_commits\":" << rasbery::rasberyGpuXeCommits()
                  << ",\"dot_partitions\":" << rasbery::rasberyGpuXeDotPartitions()
                  << "}" << std::endl;
    }
    drive_end = std::chrono::steady_clock::now();
    if (rasbery::rasberyGpuNodalEnabled()) {
        std::cout << "[RASBERY][NODAL][GPU] {\"drives_solved\":"
                  << rasbery::rasberyGpuNodalDrives() << "}" << std::endl;
        // See the batch arm above: Task 18-lite's elided per-drive transfers.
        std::cout << "[RASBERY][NODAL][CANON] {\"elided_upload_bytes\":"
                  << rasbery::rasberyGpuNodalCanonicalElidedUploadBytes()
                  << ",\"elided_download_bytes\":"
                  << rasbery::rasberyGpuNodalCanonicalElidedDownloadBytes()
                  << "}" << std::endl;
    }
    rasbery::gpu::reportOuterSegment(std::cout);
    rasbery::xsphase::report(std::cout);
    rasbery::outer_timing::report(std::cout);
    const auto hdf5_stats = Chiffon::GetHdf5LockStats();
    std::cout << "[RASBERY][HDF5][LOCK] {\"acquires\":"
              << hdf5_stats.acquisitions << ",\"wait_ms\":"
              << static_cast<double>(hdf5_stats.wait_nanoseconds) / 1.0e6
              << "}" << std::endl;
    rasbery::PrintXsLibraryCacheReceipt(std::cout);
    // A single case is builds=1, hits=0.  Printed here too so the three
    // branches' teardown receipts stay one set and not three.
    rasbery::cohort::printReceipt(std::cout);
    rasberyPrintProcessLedger(std::cout, exec_seconds, pre_drive_seconds,
                              since_start(drive_end) - pre_drive_seconds,
                              since_start(std::chrono::steady_clock::now()) -
                                  since_start(drive_end),
                              static_cast<int>(rasbery_inputs.size()));
    rasbery::iowriter::reportSummary(std::cout);
    rasbery::iowriter::reportResultIo(std::cout);
    return exit_code;
}
