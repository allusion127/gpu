#pragma once

// Lightweight, process-safe scalar receipts for in-process batch runs.
//
// Full result HDF5 and restart writes are intentionally skipped in this mode:
// HDF5 1.10.x is process-global and non-threadsafe, and pin/node payloads are
// too large to queue without copying.  The solver still computes the same
// Schedule fields; this writer persists only the scalar fitness/safety fields
// needed by the GA.  Selected finalists can rerun with ordinary full I/O.

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

class BatchLightResult {
public:
    static bool Enabled() {
        const char* value = std::getenv("RASBERY_BATCH_LIGHT_RESULT");
        return value && *value && std::string(value) != "0";
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

    // Dependency-free SHA-256 for the two provenance inputs.  The digest is
    // cached per process because every schedule step in a case references the
    // same input and XS files.  A missing file yields an empty string and is
    // represented as JSON null rather than a misleading path hash.
    static std::string Sha256File(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) return {};

        constexpr std::array<std::uint32_t, 64> K = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        const auto rotr = [](std::uint32_t x, unsigned n) {
            return (x >> n) | (x << (32u - n));
        };
        std::array<std::uint32_t, 8> h = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::array<std::uint8_t, 64> block{};
        std::uint64_t total = 0;

        const auto process = [&](const std::uint8_t* bytes) {
            std::array<std::uint32_t, 64> w{};
            for (unsigned i = 0; i < 16; ++i)
                w[i] = (static_cast<std::uint32_t>(bytes[4 * i]) << 24) |
                       (static_cast<std::uint32_t>(bytes[4 * i + 1]) << 16) |
                       (static_cast<std::uint32_t>(bytes[4 * i + 2]) << 8) |
                       static_cast<std::uint32_t>(bytes[4 * i + 3]);
            for (unsigned i = 16; i < 64; ++i) {
                const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
            std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
            for (unsigned i = 0; i < 64; ++i) {
                const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                const std::uint32_t ch = (e & f) ^ ((~e) & g);
                const std::uint32_t t1 = hh + s1 + ch + K[i] + w[i];
                const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                const std::uint32_t t2 = s0 + maj;
                hh = g;
                g  = f;
                f  = e;
                e  = d + t1;
                d  = c;
                c  = b;
                b  = a;
                a  = t1 + t2;
            }
            h[0] += a;
            h[1] += b;
            h[2] += c;
            h[3] += d;
            h[4] += e;
            h[5] += f;
            h[6] += g;
            h[7] += hh;
        };

        while (input) {
            input.read(reinterpret_cast<char*>(block.data()), static_cast<std::streamsize>(block.size()));
            const auto got = input.gcount();
            if (got <= 0) break;
            total += static_cast<std::uint64_t>(got);
            if (got == static_cast<std::streamsize>(block.size())) {
                process(block.data());
            } else {
                const auto n = static_cast<std::size_t>(got);
                block[n] = 0x80;
                std::fill(block.begin() + n + 1, block.end(), 0);
                if (n >= 56) {
                    process(block.data());
                    block.fill(0);
                }
                const std::uint64_t bits = total * 8;
                for (unsigned i = 0; i < 8; ++i)
                    block[63 - i] = static_cast<std::uint8_t>(bits >> (8 * i));
                process(block.data());
                std::ostringstream out;
                out << std::hex << std::setfill('0');
                for (const auto v : h) out << std::setw(8) << v;
                return out.str();
            }
        }

        block.fill(0);
        block[0] = 0x80;
        if (total % 64 >= 56) {
            process(block.data());
            block.fill(0);
        }
        const std::uint64_t bits = total * 8;
        for (unsigned i = 0; i < 8; ++i)
            block[63 - i] = static_cast<std::uint8_t>(bits >> (8 * i));
        process(block.data());
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto v : h) out << std::setw(8) << v;
        return out.str();
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
        if (!Enabled()) return;

        nlohmann::ordered_json receipt;
        receipt["mode"] = "batch_light";
        const int feedback_passes = FeedbackPasses();
        receipt["physics_mode"] = feedback_passes > 0
                                      ? "ga_screen_feedback_limited"
                                      : "exact";
        receipt["feedback_passes"] = feedback_passes;
        receipt["requires_exact_rerun"] = feedback_passes > 0;
        receipt["candidate_id"] = CandidateId(input);
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
