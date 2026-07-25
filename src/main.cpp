#include "Exporter.h"
#include "Importer.h"

#include "Driver.h"
#include "plog/Appenders/ConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Init.h"
#include "plog/Log.h"
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

namespace rasbery {
// Definition of the solver OpenMP gate declared in Geometry.h. Aggressive default of 1024 nodes:
// below this the per-node solver loops stay serial (fork/join overhead wins). Overridable at
// runtime via RASBERY_OMP_GATE for tuning on a given machine/core size.
int rasbery_omp_gate = 256;
} // namespace rasbery

namespace {

constexpr int RASBERY_OMP_THREADS = 8;

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

    int argi = 1;
    while (argi < argc) {
        const std::string option = argv[argi];

        if (option == "-h" || option == "--help") {
            std::cout << "Usage:\n"
                      << "  RASBERY [--chiffoni input1.json input2.json ...] [--chiffono output1.h5 output2.h5 ...]\n"
                      << "          [--rasi input1.json input2.json ...] [--raso output1.h5 output2.h5 ...]\n\n"
                      << "Notes:\n"
                      << "  - Values after each flag are consumed until the next --flag.\n"
                      << "  - The number of --chiffoni and --chiffono paths must match.\n"
                      << "  - The number of --rasi and --raso paths must match.\n";
            return 0;
        }

        if (option != "--chiffoni" && option != "--chiffono" &&
            option != "--rasi" && option != "--raso") {
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
            else
                rasbery_outputs.push_back(value);

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

    if (chiffon_inputs.empty() && rasbery_inputs.empty()) {
        std::cerr << "No jobs were provided. Use --help for usage." << std::endl;
        return 1;
    }

    Chiffon::Isotope::Initialize(repo_dir / "include" / "Database");

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

    return exit_code;
}
