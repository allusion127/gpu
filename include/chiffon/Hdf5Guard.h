#pragma once

// HDF5 1.10.x builds used by RASBERY are not thread-safe.  Batch mode keeps
// physics state per worker, but the importer/restart/result paths still enter
// the process-global HDF5 runtime.  This recursive guard serialises those
// calls while allowing nested helpers (e.g. ReadInput -> LoadGeometryFromRestart
// or XSSet -> Importer::LoadHDF) to use the same protection.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace Chiffon {

struct Hdf5LockStats {
    std::uint64_t acquisitions = 0;
    std::uint64_t wait_nanoseconds = 0;
};

inline std::recursive_mutex hdf5Mutex;
inline std::atomic<std::uint64_t> hdf5LockAcquisitions{0};
inline std::atomic<std::uint64_t> hdf5LockWaitNanoseconds{0};

class Hdf5Guard {
public:
    Hdf5Guard()
        : lock(hdf5Mutex, std::defer_lock), wait_start(std::chrono::steady_clock::now()) {
        lock.lock();
        const auto waited = std::chrono::steady_clock::now() - wait_start;
        hdf5LockWaitNanoseconds.fetch_add(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count()),
            std::memory_order_relaxed);
        hdf5LockAcquisitions.fetch_add(1, std::memory_order_relaxed);
    }

    Hdf5Guard(const Hdf5Guard&) = delete;
    Hdf5Guard& operator=(const Hdf5Guard&) = delete;

private:
    std::unique_lock<std::recursive_mutex> lock;
    std::chrono::steady_clock::time_point   wait_start;
};

inline Hdf5LockStats GetHdf5LockStats() {
    return {hdf5LockAcquisitions.load(std::memory_order_relaxed),
            hdf5LockWaitNanoseconds.load(std::memory_order_relaxed)};
}

} // namespace Chiffon
