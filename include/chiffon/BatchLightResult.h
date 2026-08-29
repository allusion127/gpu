#pragma once

// Lightweight, process-safe scalar receipts for in-process batch runs.
//
// Full result HDF5 and restart writes are intentionally skipped in this mode:
// HDF5 1.10.x is process-global and non-threadsafe, and pin/node payloads are
// too large to queue without copying.  The solver still computes the same
// Schedule fields; this writer persists only the scalar fitness/safety fields
// needed by the GA.  Selected finalists can rerun with ordinary full I/O.

#include "Sha256.h"

#include "nlohmann/json.hpp"

#include <cmath>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace rasbery {

/// What a case is asked to produce.  Orthogonal to the physics: all three
/// modes run the same solve and the same PPR, and the campaign has measured
/// all three to the same trajectory digest (814201df0583e1d2, GA evaluator
/// plan Sec 2.1).  What differs is only what leaves the process.
///
///   Full    result HDF5 + restarts + pin-power CSV   (~301.6 MB/case)
///   PinOff  result HDF5 + restarts, no pin output    (~12.0 MB/case)
///   Light   one JSONL line per statepoint, no HDF5   (~25.1 kB/case)
///
/// PER JOB, not per process.  A GA wave wants Light for the population and
/// Full for the handful of elites it promotes, in ONE batch; the environment
/// variable cannot express that, because every worker in a --batch-mode
/// process shares it.
enum class ResultMode { Full, PinOff, Light };

inline const char* ResultModeName(ResultMode mode) {
    switch (mode) {
        case ResultMode::Light:  return "light";
        case ResultMode::PinOff: return "pin-off";
        case ResultMode::Full:   break;
    }
    return "full";
}

/// Parse `full` | `pin-off` | `light`.  False (and `out` untouched) on anything
/// else, so the caller can name the offending word in its own error.
inline bool ParseResultMode(const std::string& text, ResultMode& out) {
    if (text == "full") { out = ResultMode::Full; return true; }
    if (text == "light") { out = ResultMode::Light; return true; }
    if (text == "pin-off" || text == "pinoff" || text == "pin_off") {
        out = ResultMode::PinOff;
        return true;
    }
    return false;
}

class BatchLightResult {
public:
    static bool Enabled() {
        const char* value = std::getenv("RASBERY_BATCH_LIGHT_RESULT");
        return value && *value && std::string(value) != "0";
    }

    /// The process default, from the environment.  `--result` overrides it and
    /// a manifest's third field overrides that; nothing here changes when
    /// neither is given, which is why the flag is additive.
    static ResultMode DefaultMode() {
        return Enabled() ? ResultMode::Light : ResultMode::Full;
    }

    static int FeedbackPasses() {
        const char* value = std::getenv("RASBERY_GA_FEEDBACK_PASSES");
        return value == nullptr ? 0 : std::max(0, std::atoi(value));
    }

    static std::filesystem::path Path() {
        const char* value = std::getenv("RASBERY_BATCH_RECEIPT_JSONL");
        if (value && *value) return std::filesystem::path(value);
        return std::filesystem::path("batch_light_receipts.jsonl");
    }

    static std::string CandidateId(const std::string& input) {
        const char* override_id = std::getenv("RASBERY_CANDIDATE_ID");
        if (override_id && *override_id) return std::string(override_id);
        const auto path = std::filesystem::path(input);
        const auto stem = path.stem().string();
        return stem.empty() ? input : stem;
    }

    // A scheduler has no cycle field: cycle identity belongs to the restart
    // chain/launcher rather than to an individual depletion step.  Preserve
    // it in the receipt when the launcher supplies RASBERY_CYCLE_ID, while
    // keeping an explicit value for standalone runs.
    static std::string CycleId() {
        const char* value = std::getenv("RASBERY_CYCLE_ID");
        return (value && *value) ? std::string(value) : std::string("unknown");
    }

    // Stable, dependency-free fingerprint for provenance correlation.  This is
    // deliberately labelled fingerprint64, not SHA-256; the launcher may add
    // authoritative SHA-256 values to the manifest beside the receipt.
    static std::string Fingerprint64(const std::string& path) {
        {
            std::lock_guard<std::mutex> lock(HashMutex());
            const auto it = FingerprintCache().find(path);
            if (it != FingerprintCache().end()) return it->second;
        }

        std::uint64_t hash = 1469598103934665603ULL;
        std::ifstream input(path, std::ios::binary);
        if (input) {
            char byte = 0;
            while (input.get(byte)) {
                hash ^= static_cast<std::uint8_t>(byte);
                hash *= 1099511628211ULL;
            }
        } else {
            for (const unsigned char byte : path) {
                hash ^= byte;
                hash *= 1099511628211ULL;
            }
        }
        std::ostringstream text;
        text << std::hex << std::setfill('0') << std::setw(16) << hash;
        const std::string result = text.str();
        std::lock_guard<std::mutex> lock(HashMutex());
        FingerprintCache().emplace(path, result);
        return result;
    }

    // Dependency-free SHA-256 for the two provenance inputs, streamed through
    // the shared transform (Sha256.h).  The digest is cached per process
    // because every schedule step in a case references the same input and XS
    // files.  A missing file yields an empty string and is represented as JSON
    // null rather than a misleading path hash.
    //
    // WP10.1 moved the transform out of this function -- unchanged, constant
    // for constant -- because the case key needs the same digest over an
    // in-memory payload, and two copies of a hash function is how a cache key
    // ends up disagreeing with the receipt that named it.
    static std::string Sha256File(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) return {};
        Sha256                        sha;
        std::array<char, 64 * 1024>   buffer{};
        while (input) {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto got = input.gcount();
            if (got <= 0) break;
            sha.update(buffer.data(), static_cast<std::size_t>(got));
        }
        return sha.hex();
    }

    static std::string Sha256FileCached(const std::string& path) {
        {
            std::lock_guard<std::mutex> lock(HashMutex());
            const auto it = Sha256Cache().find(path);
            if (it != Sha256Cache().end()) return it->second;
        }
        const std::string result = Sha256File(path);
        std::lock_guard<std::mutex> lock(HashMutex());
        Sha256Cache().emplace(path, result);
        return result;
    }

    template <typename Value>
    static void PutFinite(nlohmann::ordered_json& object,
                          const char* key, Value value) {
        if constexpr (std::is_floating_point_v<Value>) {
            if (std::isfinite(value)) object[key] = value;
            else object[key] = nullptr;
        } else {
            object[key] = value;
        }
    }

    static void Write(const std::string& input,
                      const std::string& xs_path,
                      const std::string& case_key,
                      const std::string& warm_state_path,
                      int step,
                      int substep,
                      double efpd,
                      double bu_avg,
                      double keff,
                      double ppm,
                      double ao,
                      double fqp,
                      double frp,
                      int search_status,
                      double search_dk,
                      double search_tol) {
        // NO SECOND OPINION ON WHETHER TO WRITE.  This used to re-read
        // RASBERY_BATCH_LIGHT_RESULT here, which was harmless while the
        // environment was the only way to ask for a light result and became a
        // silent no-op the moment `--result light` existed: the case ran, the
        // scalars were computed, and nothing was written anywhere.  The caller
        // (Driver::Drive, on its own per-case ResultMode) has already decided;
        // a writer that second-guesses its caller from a process global cannot
        // serve a per-job mode at all.

        nlohmann::ordered_json receipt;
        receipt["mode"] = "batch_light";
        const int feedback_passes = FeedbackPasses();
        receipt["physics_mode"] = feedback_passes > 0
                                      ? "ga_screen_feedback_limited"
                                      : "exact";
        receipt["feedback_passes"] = feedback_passes;
        receipt["requires_exact_rerun"] = feedback_passes > 0;
        receipt["candidate_id"] = CandidateId(input);
        // WP10.1.  The GA's duplicate cache keys on this; `candidate_id` above
        // is a NAME the launcher chose and two names can denote one case, which
        // is exactly the confusion the key exists to end.  Empty prints as null
        // rather than as an empty string that a cache might index on.
        receipt["case_key"] = case_key.empty() ? nlohmann::ordered_json(nullptr)
                                               : nlohmann::ordered_json(case_key);
        // WP10.2.  Where this case's BOC warm state was written, so a GA can
        // seed the next generation from the light result it already reads --
        // without grepping a log for [RASBERY][WARMSTART].  Null when the case
        // was not asked to save one, which is the default.
        receipt["warm_state"] = warm_state_path.empty()
                                    ? nlohmann::ordered_json(nullptr)
                                    : nlohmann::ordered_json(warm_state_path);
        receipt["cycle"] = CycleId();
        receipt["input_path"] = input;
        receipt["xs_path"] = xs_path;
        receipt["input_fingerprint64"] = Fingerprint64(input);
        receipt["xs_fingerprint64"] = Fingerprint64(xs_path);
        const auto input_sha256 = Sha256FileCached(input);
        const auto xs_sha256 = Sha256FileCached(xs_path);
        receipt["input_sha256"] = input_sha256.empty() ? nlohmann::ordered_json(nullptr)
                                                        : nlohmann::ordered_json(input_sha256);
        receipt["xs_sha256"] = xs_sha256.empty() ? nlohmann::ordered_json(nullptr)
                                                  : nlohmann::ordered_json(xs_sha256);
        receipt["step"] = step;
        receipt["substep"] = substep;
        PutFinite(receipt, "efpd", efpd);
        PutFinite(receipt, "bu_avg", bu_avg);
        PutFinite(receipt, "keff", keff);
        PutFinite(receipt, "ppm", ppm);
        PutFinite(receipt, "ao", ao);
        PutFinite(receipt, "fqp", fqp);
        PutFinite(receipt, "frp", frp);
        receipt["search_status"] = search_status;
        PutFinite(receipt, "search_dk", search_dk);
        PutFinite(receipt, "search_tol", search_tol);
        receipt["converged"] = search_status == 1;

        const auto path = Path();
        std::lock_guard<std::mutex> lock(Mutex());
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::app);
        if (!output)
            throw std::runtime_error("failed to open batch-light receipt: " + path.string());
        output << receipt.dump() << '\n';
        output.flush();
    }

private:
    static std::mutex& HashMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_map<std::string, std::string>& FingerprintCache() {
        static std::unordered_map<std::string, std::string> cache;
        return cache;
    }

    static std::unordered_map<std::string, std::string>& Sha256Cache() {
        static std::unordered_map<std::string, std::string> cache;
        return cache;
    }

    static std::mutex& Mutex() {
        static std::mutex mutex;
        return mutex;
    }
};

} // namespace rasbery
