#include "Benchmark.h"
#include "Exporter.h"
#include "Importer.h"

#include "CudaXsReconBackend.h"
#include "Driver.h"
#include "XSTiming.h"
#include "plog/Appenders/ConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Init.h"
#include "plog/Log.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
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

int rasberyVisibleCpuThreads() {
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        const int count = CPU_COUNT(&affinity);
        if (count > 0) return count;
    }
#endif
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

void rasberyPrepareOpenMPStartup(char* argv[]) {
#if !defined(_WIN32)
    if (std::getenv("RASBERY_OMP_ENV_READY") != nullptr)
        return;

    bool changed = false;
    changed |= rasberySetEnvIfNeeded("OMP_WAIT_POLICY", "PASSIVE");
    changed |= rasberySetEnvIfNeeded("GOMP_SPINCOUNT", "0");
    changed |= rasberySetEnvIfNeeded("OMP_NUM_THREADS", "8", true);
    changed |= rasberySetEnvIfNeeded("OMP_PROC_BIND", "TRUE", true);
    changed |= rasberySetEnvIfNeeded("OMP_PLACES", "cores", true);
    rasberySetEnv("RASBERY_OMP_ENV_READY", "1", true);

    if (changed)
        execvp(argv[0], argv);
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

    int argi = 1;
    while (argi < argc) {
        const std::string option = argv[argi];

        if (option == "-h" || option == "--help") {
            std::cout << "Usage:\n"
                      << "  RASBERY [--chiffoni input1.json input2.json ...] [--chiffono output1.h5 output2.h5 ...]\n"
                      << "          [--rasi input1.json input2.json ...] [--raso output1.h5 output2.h5 ...]\n"
                      << "          [--batch-mode M]\n\n"
                      << "Notes:\n"
                      << "  - Values after each flag are consumed until the next --flag.\n"
                      << "  - The number of --chiffoni and --chiffono paths must match.\n"
                      << "  - The number of --rasi and --raso paths must match.\n"
                      << "  - --batch-mode M runs the --rasi decks M at a time in one process,\n"
                      << "    one host thread per instance, with their CMFD linear solves\n"
                      << "    batched into a single GPU grid.  Every deck must share one\n"
                      << "    geometry; only the state (temperatures, power, rods, schedule)\n"
                      << "    may differ.  Requires a CUDA build.  Each instance is\n"
                      << "    bit-identical to running that deck on its own.\n";
            return 0;
        }

        if (option != "--chiffoni" && option != "--chiffono" &&
            option != "--rasi" && option != "--raso" &&
            option != "--isohgc" && option != "--isocsv" && option != "--validate" &&
            option != "--batch-mode") {
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
            else
                iso_csv = value;

            ++argi;
        }
    }

    if (chiffon_inputs.size() != chiffon_outputs.size()) {
        std::cerr << "The number of --chiffoni and --chiffono paths must match." << std::endl;
        return 1;
    }

    if (rasbery_inputs.size() != rasbery_outputs.size()) {
        std::cerr << "The number of --rasi and --raso paths must match." << std::endl;
        return 1;
    }

    const int ga_feedback_passes = rasbery::BatchLightResult::FeedbackPasses();
    if (ga_feedback_passes > 0) {
        if (!rasbery::BatchLightResult::Enabled()) {
            std::cerr << "RASBERY_GA_FEEDBACK_PASSES is a GA screening approximation and "
                         "requires RASBERY_BATCH_LIGHT_RESULT=1. Rerun selected candidates "
                         "with RASBERY_GA_FEEDBACK_PASSES unset for acceptance."
                      << std::endl;
            return 2;
        }
        std::cout << "[RASBERY][GA][SCREEN] {\"physics_mode\":"
                     "\"ga_screen_feedback_limited\",\"feedback_passes\":"
                  << ga_feedback_passes
                  << ",\"requires_exact_rerun\":true}" << std::endl;
    }

    if (chiffon_inputs.empty() && rasbery_inputs.empty() && iso_csv.empty() && validate_args.empty()) {
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
    if (batch_width > 0 && !rasbery_inputs.empty()) {
        rasbery::rasberySetBatchWidth(batch_width);
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
#else
        const int visible_cpus = 1;
#endif
        std::cout << "\n[RASBERY][BATCH] " << jobs << " deck(s), width " << batch_width
                  << ", " << host_threads << " host Driver worker(s)" << std::endl;
        std::cout << "[RASBERY][BATCH_HOST] {\"jobs\":" << jobs
                  << ",\"arena_width\":" << batch_width
                  << ",\"host_threads\":" << host_threads
                  << ",\"visible_cpus\":" << visible_cpus << "}" << std::endl;
#ifdef _OPENMP
        // One OpenMP level only: the instance loop is the parallelism. Nested
        // Driver regions reduce the measured GPU rendezvous width and lose
        // aggregate throughput even when total CPU threads are held constant.
        omp_set_max_active_levels(1);
        omp_set_num_threads(host_threads);
#endif
        std::vector<int> job_status(static_cast<std::size_t>(jobs), 0);
        std::vector<std::string> job_error(static_cast<std::size_t>(jobs));

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(host_threads)
#endif
        for (int i = 0; i < jobs; ++i) {
            // An escaping exception would terminate the whole parallel region,
            // taking the other instances' partial results with it.  One bad
            // deck must only fail its own job.
            try {
                rasbery::Driver driver(rasbery_inputs[static_cast<std::size_t>(i)],
                                       rasbery_outputs[static_cast<std::size_t>(i)]);
                job_status[static_cast<std::size_t>(i)] = driver.Drive();
            } catch (const std::exception& error) {
                job_status[static_cast<std::size_t>(i)] = 1;
                job_error[static_cast<std::size_t>(i)]  = error.what();
            } catch (...) {
                job_status[static_cast<std::size_t>(i)] = 1;
                job_error[static_cast<std::size_t>(i)]  = "unknown exception";
            }
        }

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

        rasbery::rasberyReleaseBatchArena();
        // Receipt for the xsrecon device path: a zero here means it never ran,
        // whatever the flag said, and an A/B built on it is void (G0).
        if (rasbery::rasberyGpuXsReconEnabled())
            std::cout << "[RASBERY][XSRECON][GPU] {\"nodes_solved\":"
                      << rasbery::rasberyGpuXsReconNodes() << "}" << std::endl;
        rasbery::xsphase::report(std::cout);
        const auto hdf5_stats = Chiffon::GetHdf5LockStats();
        std::cout << "[RASBERY][HDF5][LOCK] {\"acquires\":"
                  << hdf5_stats.acquisitions << ",\"wait_ms\":"
                  << static_cast<double>(hdf5_stats.wait_nanoseconds) / 1.0e6
                  << "}" << std::endl;
        return exit_code;
    }

    for (std::size_t i = 0; i < rasbery_inputs.size(); ++i) {
        const fs::path rasbery_input_path  = rasbery_inputs[i];
        const fs::path rasbery_output_path = rasbery_outputs[i];

        std::cout << "\n[RASBERY] " << rasbery_input_path.string()
                  << " -> " << rasbery_output_path.string() << std::endl;

        rasbery::Driver driver(rasbery_input_path.string(), rasbery_output_path.string());
        const int       driver_exit_code = driver.Drive();
        if (driver_exit_code != 0 && exit_code == 0) {
            std::cout << "[RASBERY][FAIL] exit_code=" << driver_exit_code
                      << " path=" << rasbery_input_path.string() << std::endl;
            exit_code = driver_exit_code;
        }
    }

    if (rasbery::rasberyGpuXsReconEnabled())
        std::cout << "[RASBERY][XSRECON][GPU] {\"nodes_solved\":"
                  << rasbery::rasberyGpuXsReconNodes() << "}" << std::endl;
    rasbery::xsphase::report(std::cout);
    const auto hdf5_stats = Chiffon::GetHdf5LockStats();
    std::cout << "[RASBERY][HDF5][LOCK] {\"acquires\":"
              << hdf5_stats.acquisitions << ",\"wait_ms\":"
              << static_cast<double>(hdf5_stats.wait_nanoseconds) / 1.0e6
              << "}" << std::endl;
    return exit_code;
}
