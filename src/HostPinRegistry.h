#pragma once

// HostPinLease -- process-wide ownership registry for cudaHostRegister ranges.
//
// THE HOLE THIS CLOSES.  Both device backends used to page-lock caller-owned
// host memory and never unregister it: XsReconBackend::pinHost and
// CudaBatchArena::pinHost were documented as permanent registrations.  That is
// only sound while every registered buffer outlives the process, which holds
// exactly when no Driver is ever destroyed before the run ends (a single deck
// per process, or --batch-mode with one host worker per deck).
//
// The moment a worker thread recycles (host_threads < jobs), Driver::Drive()
// returns and destroys its stack-local Geometry/XSSet/BICGCMFD, freeing ranges
// the CUDA driver still holds page-locked.  The allocator then hands those same
// addresses to the next deck -- same thread arena, same sizes, same order -- so
// the new deck's buffers alias a dead tenant's registration: cudaHostRegister
// answers "already mapped", the DMA lands on the previous tenant's physical
// pages, and any block that malloc served with mmap (and freed with munmap)
// turns every later cudaMemcpyAsync on it into "invalid argument".  That is
// what turned `--batch-mode 64` with 24 workers into 54 failed decks.
//
// The fix is a lease, not a bigger hammer: every registration is owned, the
// owner releases it in its destructor BEFORE the memory is freed, and the
// registration is torn down with cudaHostUnregister at the same address
// cudaHostRegister saw.  A recycled address therefore arrives at an EMPTY
// registry and is registered afresh, which is the only state in which the
// device copies are legal.
//
// WHO OWNS A LEASE.  The lease belongs to the memory, not to the caller that
// happened to want it pinned.  Geometry's Phif is handed to pinHost by the CMFD
// sweep path, by the XSSet reconstruct arm, by the per-instance nodal arm and
// by the nodal batch arena -- four callers, one buffer, one owner (~Geometry).
// So a repeat request for a base that already holds a lease is idempotent
// (deduplicated_requests++ only); a request for a DIFFERENT base inside an
// existing registration is a second owner (owners++), and both have to release
// before the range is unregistered.
//
// LAYERING.  Header-only and CUDA-free on purpose.  The owners that must
// release leases (Geometry, Nodal, CMFD, BICGCMFD, XSSet) are plain C++
// translation units that must keep compiling in the no-CUDA stub build, and
// several test targets link one backend TU without the other.  The actual
// cudaHostRegister/cudaHostUnregister calls therefore arrive as function
// POINTER HOOKS that the CUDA translation units install at first use; a stub
// build never installs them, so every operation below degrades to registry
// bookkeeping with no device call -- the same lifecycle, exercised by the same
// destructors, on a machine with no GPU.  Same ODR-safe inline
// function-local-static pattern as rasberyGpuNodalFullEnabled() and
// rasberyNodalBatchWidthRef() in CudaXsReconBackend.h: one process-wide value,
// no new exported symbol in either backend.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rasbery {

/// Page granularity for the CONFLICT interval only.  Registration itself always
/// uses the caller's address and byte count (see PinRecord).
inline constexpr std::uintptr_t kHostPinPageSize = 4096;

/// RASBERY_HOST_PINNING (plan Rev.4 Sec 6.7).
enum class HostPinningMode {
    Auto,  ///< default: pin under the lease contract
    Off,   ///< force pageable copies everywhere
    Force  ///< development experiment only; ignores the programmatic gate
};

/// One live cudaHostRegister range.
///
/// The conflict interval is page-normalised because the CUDA driver's own
/// bookkeeping is per page: two allocations sharing a page share a
/// registration, and asking to register the second one answers "already
/// mapped".  The address handed BACK to cudaHostUnregister must nevertheless be
/// the one cudaHostRegister saw (Sec 6.1) -- unregistering a normalised page
/// start the driver never registered is not the same call.
struct PinRecord {
    std::uintptr_t conflict_page_begin = 0;
    std::uintptr_t conflict_page_end   = 0;

    void*       registered_address = nullptr;
    std::size_t registered_bytes   = 0;

    unsigned owners    = 0;
    /// Reserved for the per-stream event drain of Sec 6.4.  This commit relies
    /// on the conservative variant that section permits: an owner releases its
    /// leases in its destructor, after its backend handles are torn down and
    /// therefore after the last Drive() on those buffers returned, so no DMA
    /// can still be in flight.  Kept in the record (and checked before every
    /// unregister) so the event-tracking version is a fill-in, not a rewrite.
    unsigned in_flight = 0;

    /// Distinct base addresses currently holding this lease, sorted.  Its size
    /// IS `owners`; the addresses are what makes release idempotent and makes a
    /// release from a caller that was refused a lease a harmless no-op.
    std::vector<std::uintptr_t> owner_bases;
};

/// Sec 6.8 receipt counters.
struct HostPinCounters {
    unsigned long long registered_ranges     = 0;
    unsigned long long registered_bytes      = 0;
    unsigned long long deduplicated_requests = 0;
    unsigned long long pageable_fallbacks    = 0;
    unsigned long long unregistered_ranges   = 0;
    unsigned long long overlap_rejections    = 0;
    unsigned long long stale_evicted         = 0;
};

/// Installed by the CUDA translation units; 0 means success, anything else is
/// the cudaError_t the call returned.
using HostPinRegisterHook   = int (*)(void* address, std::size_t bytes);
using HostPinUnregisterHook = int (*)(void* address);

inline std::mutex& rasberyHostPinMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::map<std::uintptr_t, PinRecord>& rasberyHostPinRecords() {
    static std::map<std::uintptr_t, PinRecord> records;
    return records;
}

inline HostPinCounters& rasberyHostPinCountersRef() {
    static HostPinCounters counters;
    return counters;
}

inline HostPinRegisterHook& rasberyHostPinRegisterHookRef() {
    static HostPinRegisterHook hook = nullptr;
    return hook;
}

inline HostPinUnregisterHook& rasberyHostPinUnregisterHookRef() {
    static HostPinUnregisterHook hook = nullptr;
    return hook;
}

/// Called by the CUDA backends at first use.  Idempotent, and taken under the
/// registry lock so the pointers are never written while another thread reads
/// them inside an operation.
inline void rasberyInstallHostPinHooks(HostPinRegisterHook reg, HostPinUnregisterHook unreg) {
    std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
    rasberyHostPinRegisterHookRef()   = reg;
    rasberyHostPinUnregisterHookRef() = unreg;
}

/// True once a CUDA translation unit has installed the register/unregister
/// hooks.  Stub builds keep the bookkeeping and skip the device calls.
inline bool rasberyHostPinHooksInstalled() {
    std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
    return rasberyHostPinRegisterHookRef() != nullptr;
}

/// RASBERY_HOST_PINNING=auto|off|force, read once per process (Sec 6.7).
inline HostPinningMode rasberyHostPinningMode() {
    static const HostPinningMode mode = [] {
        const char* value = std::getenv("RASBERY_HOST_PINNING");
        if (value == nullptr || *value == '\0') return HostPinningMode::Auto;
        std::string requested(value);
        std::transform(requested.begin(), requested.end(), requested.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (requested == "off" || requested == "0" || requested == "false" ||
            requested == "no")
            return HostPinningMode::Off;
        if (requested == "force") {
            std::cerr << "[RASBERY][WARN][PIN] RASBERY_HOST_PINNING=force overrides the "
                         "process gate and page-locks every buffer the backends ask for, "
                         "including configurations main() declined. Development experiment "
                         "only -- benchmark and acceptance runs use auto.\n";
            return HostPinningMode::Force;
        }
        if (requested != "auto" && requested != "1" && requested != "on" &&
            requested != "true")
            std::cerr << "[RASBERY][WARN][PIN] unknown RASBERY_HOST_PINNING=" << value
                      << " -- using auto.\n";
        return HostPinningMode::Auto;
    }();
    return mode;
}

inline const char* rasberyHostPinningModeName() {
    switch (rasberyHostPinningMode()) {
        case HostPinningMode::Off: return "off";
        case HostPinningMode::Force: return "force";
        default: return "auto";
    }
}

/// RASBERY_PIN_STRICT=1 turns an overlap refusal from a pageable fallback into
/// a thrown error (Sec 6.2), for debug/CI runs that want the aliasing to be
/// loud rather than merely slower.
inline bool rasberyHostPinStrict() {
    static const bool strict = [] {
        const char* value = std::getenv("RASBERY_PIN_STRICT");
        if (value == nullptr) return false;
        const std::string s(value);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
                 s == "FALSE");
    }();
    return strict;
}

/// Process-wide programmatic gate, published by main() for the configuration it
/// is about to run.
///
/// Deliberately an inline function-local static: several test targets link
/// CudaXsReconBackend.cu without CudaBICGBackend.cu (and vice versa), and a
/// header-only flag adds no exported symbol to either.  Defaults to enabled so
/// every target that never calls the setter (the replay/consistency
/// executables, the stub build) behaves exactly as before.
inline bool& rasberyHostPinningRef() {
    static bool enabled = true;
    return enabled;
}
inline void rasberySetHostPinningEnabled(bool enabled) { rasberyHostPinningRef() = enabled; }

/// The env mode is authoritative in both directions: `off` never pins whatever
/// main() decided, `force` always does, `auto` defers to the gate.
inline bool rasberyHostPinningEnabled() {
    const HostPinningMode mode = rasberyHostPinningMode();
    if (mode == HostPinningMode::Off) return false;
    if (mode == HostPinningMode::Force) return true;
    return rasberyHostPinningRef();
}

namespace host_pin_detail {

using RecordMap  = std::map<std::uintptr_t, PinRecord>;
using RecordIter = RecordMap::iterator;

/// Records never overlap each other, so the only candidate below `begin` is the
/// immediate predecessor.  Caller holds the registry lock.
inline std::vector<RecordIter> overlaps(RecordMap& records, std::uintptr_t begin,
                                        std::uintptr_t end) {
    std::vector<RecordIter> hits;
    RecordIter              it = records.lower_bound(begin);
    if (it != records.begin()) {
        RecordIter prev = std::prev(it);
        if (prev->second.conflict_page_end > begin) it = prev;
    }
    for (; it != records.end() && it->second.conflict_page_begin < end; ++it)
        if (it->second.conflict_page_end > begin) hits.push_back(it);
    return hits;
}

/// cudaHostUnregister at the ORIGINAL registered address (Sec 6.1).  Failure is
/// not actionable at teardown -- the range is going away either way -- so the
/// return code only suppresses the counter.  Caller holds the registry lock.
inline void unregisterLocked(PinRecord& record, HostPinCounters& counters) {
    HostPinUnregisterHook hook = rasberyHostPinUnregisterHookRef();
    if (hook != nullptr && record.registered_address != nullptr) hook(record.registered_address);
    ++counters.unregistered_ranges;
}

} // namespace host_pin_detail

/// Acquire a lease on [p, p+bytes).  Returns true when the range is page-locked
/// (freshly registered or covered by an existing registration this call now
/// co-owns), false when the caller must fall back to a pageable copy.
///
/// Overlap policy, verbatim from Sec 6.2:
///   requested == existing  or  requested SUBSET existing -> reuse, owners++
///   existing SUBSET requested, or partial overlap        -> pageable fallback
/// A wider or straddling request is never allowed to expand or re-register an
/// existing record: unregistering a range another owner is copying from is
/// exactly the aliasing failure this registry exists to prevent, and pageable
/// async copies are already a legal path (they are what RASBERY_HOST_PINNING=off
/// runs everywhere).
inline bool rasberyPinHost(const void* p, std::size_t bytes) {
    if (p == nullptr || bytes == 0) return false;
    if (!rasberyHostPinningEnabled()) return false;

    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t page_begin = base & ~(kHostPinPageSize - 1);
    const std::uintptr_t page_end =
        (base + bytes + kHostPinPageSize - 1) & ~(kHostPinPageSize - 1);

    std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
    auto&                       records  = rasberyHostPinRecords();
    HostPinCounters&            counters = rasberyHostPinCountersRef();

    std::vector<host_pin_detail::RecordIter> hits =
        host_pin_detail::overlaps(records, page_begin, page_end);

    // An overlapping record with owners == 0 is a stray: its owner freed the
    // buffer without releasing the lease (a bug once the destructors below are
    // wired, and the pre-lease state of the world for anything that predates
    // them).  owners == 0 means no live lease, so evicting it is safe -- and
    // necessary, because the new tenant now occupies those pages.  Records with
    // owners > 0 are NEVER evicted (Sec 6.3).
    bool evicted = false;
    for (host_pin_detail::RecordIter it : hits) {
        if (it->second.owners != 0 || it->second.in_flight != 0) continue;
        host_pin_detail::unregisterLocked(it->second, counters);
        records.erase(it);
        ++counters.stale_evicted;
        evicted = true;
    }
    if (evicted) hits = host_pin_detail::overlaps(records, page_begin, page_end);

    if (hits.empty()) {
        void* address = const_cast<void*>(p);
        if (HostPinRegisterHook hook = rasberyHostPinRegisterHookRef(); hook != nullptr) {
            if (hook(address, bytes) != 0) {
                ++counters.pageable_fallbacks;
                return false;
            }
        }
        PinRecord record;
        record.conflict_page_begin = page_begin;
        record.conflict_page_end   = page_end;
        record.registered_address  = address;
        record.registered_bytes    = bytes;
        record.owner_bases.push_back(base);
        record.owners = 1;
        records.emplace(page_begin, std::move(record));
        ++counters.registered_ranges;
        counters.registered_bytes += bytes;
        return true;
    }

    if (hits.size() == 1) {
        PinRecord& record = hits.front()->second;
        if (page_begin >= record.conflict_page_begin && page_end <= record.conflict_page_end) {
            ++counters.deduplicated_requests;
            const auto pos =
                std::lower_bound(record.owner_bases.begin(), record.owner_bases.end(), base);
            if (pos == record.owner_bases.end() || *pos != base) {
                record.owner_bases.insert(pos, base);
                record.owners = static_cast<unsigned>(record.owner_bases.size());
            }
            return true;
        }
    }

    ++counters.overlap_rejections;
    ++counters.pageable_fallbacks;
    if (rasberyHostPinStrict())
        throw std::runtime_error(
            "RASBERY_PIN_STRICT: host pin request overlaps a live registration without "
            "being contained by it; the request would have to unregister memory another "
            "owner is still copying from");
    return false;
}

/// Release the lease `base` holds.  Unregisters only when the last owner is
/// gone and nothing is in flight (Sec 6.3).  Releasing an address that never
/// held a lease -- a caller that took the pageable fallback, or an owner
/// enumerating buffers the backends never asked to pin -- is a no-op, which is
/// what lets the destructors below list their arrays unconditionally.
inline void rasberyUnpinHost(const void* base_pointer) {
    if (base_pointer == nullptr) return;
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(base_pointer);
    const std::uintptr_t page = base & ~(kHostPinPageSize - 1);

    std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
    auto&                       records  = rasberyHostPinRecords();
    HostPinCounters&            counters = rasberyHostPinCountersRef();

    // Records never overlap each other, so a page is covered by at most one.
    const auto hits = host_pin_detail::overlaps(records, page, page + kHostPinPageSize);
    if (hits.empty()) return;

    const host_pin_detail::RecordIter it     = hits.front();
    PinRecord&                        record = it->second;
    const auto pos =
        std::lower_bound(record.owner_bases.begin(), record.owner_bases.end(), base);
    if (pos == record.owner_bases.end() || *pos != base) return; // never leased here
    record.owner_bases.erase(pos);
    record.owners = static_cast<unsigned>(record.owner_bases.size());
    if (record.owners == 0 && record.in_flight == 0) {
        host_pin_detail::unregisterLocked(record, counters);
        records.erase(it);
    }
}

/// Live records, for tests and diagnostics.
inline std::size_t rasberyHostPinLiveRanges() {
    std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
    return rasberyHostPinRecords().size();
}

/// Owner count of the record covering `base`, or 0 when nothing covers it.
inline unsigned rasberyHostPinOwners(const void* base_pointer) {
    if (base_pointer == nullptr) return 0;
    const std::uintptr_t page = reinterpret_cast<std::uintptr_t>(base_pointer) &
                                ~(kHostPinPageSize - 1);
    std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
    const auto hits = host_pin_detail::overlaps(rasberyHostPinRecords(), page,
                                                page + kHostPinPageSize);
    return hits.empty() ? 0u : hits.front()->second.owners;
}

inline HostPinCounters rasberyHostPinCounters() {
    std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
    return rasberyHostPinCountersRef();
}

/// Explicit registry shutdown (Sec 6.6).  main() calls this after every Driver
/// is gone and rasberyReleaseBatchArena() has drained the backends, and BEFORE
/// the CUDA context goes away -- function-local static destructors are not
/// allowed to own CUDA teardown, because their order against the runtime's own
/// is not ours to choose.
///
/// Ranges with owners > 0 at this point are leaks: an owner destructor that
/// does not release.  They are reported, not force-unregistered -- the whole
/// point of the lease is that a registration in use is never torn down under
/// its user, and at process exit leaving them registered is exactly the
/// pre-lease behaviour, which is safe for the seconds that remain.
inline void rasberyDrainPinnedRegistry() {
    unsigned long long leaked       = 0;
    unsigned long long leaked_bytes = 0;
    unsigned long long drained      = 0;
    {
        std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
        auto&                       records  = rasberyHostPinRecords();
        HostPinCounters&            counters = rasberyHostPinCountersRef();
        for (auto it = records.begin(); it != records.end();) {
            if (it->second.owners == 0 && it->second.in_flight == 0) {
                host_pin_detail::unregisterLocked(it->second, counters);
                it = records.erase(it);
                ++drained;
            } else {
                ++leaked;
                leaked_bytes += it->second.registered_bytes;
                ++it;
            }
        }
    }
    if (leaked != 0)
        std::cerr << "[RASBERY][WARN][PIN] {\"leaked_ranges\":" << leaked
                  << ",\"leaked_bytes\":" << leaked_bytes << ",\"drained_ranges\":" << drained
                  << "} -- host pin leases still held at shutdown; an owner destructor did "
                     "not call rasberyUnpinHost\n";
}

/// The Sec 6.8 receipt fields, without the enclosing braces, so a caller can
/// splice them into a receipt object it is already building.
inline void rasberyAppendHostPinReceiptFields(std::ostream& os) {
    const HostPinCounters counters = rasberyHostPinCounters();
    os << "\"pinning_mode\":\"" << rasberyHostPinningModeName()
       << "\",\"registered_ranges\":" << counters.registered_ranges
       << ",\"registered_bytes\":" << counters.registered_bytes
       << ",\"deduplicated_requests\":" << counters.deduplicated_requests
       << ",\"pageable_fallbacks\":" << counters.pageable_fallbacks
       << ",\"unregistered_ranges\":" << counters.unregistered_ranges
       << ",\"overlap_rejections\":" << counters.overlap_rejections
       << ",\"stale_evicted\":" << counters.stale_evicted;
}

} // namespace rasbery
