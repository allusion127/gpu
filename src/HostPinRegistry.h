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
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <new>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rasbery {

/// Page granularity for the CONFLICT interval only.  Registration itself always
/// uses the caller's address and byte count (see PinRecord).
inline constexpr std::uintptr_t kHostPinPageSize = 4096;

// PAGE-EXCLUSIVE HOST STORAGE -- the SOURCE fix for Sec 6.2 overlap refusals.
//
// cudaHostRegister works on whole pages.  Two DIFFERENT allocations that happen
// to share one page therefore cannot both be registered: the second call is
// answered "already mapped" and its whole buffer -- however many megabytes --
// takes the pageable path.  The registry sees exactly the same thing one level
// up (a request whose page interval straddles a live record without being
// contained by it) and refuses it, which is what a receipt reports as
// overlap_rejections.
//
// Nothing about that is a policy bug: the request really is unregisterable.
// The bug is upstream, in how the buffers are allocated.  A general-purpose
// allocator packs its chunks 16 bytes apart, so a RUN of adjacent buffers --
// exactly what CMFD, Nodal, Geometry and the XS array sets allocate, one after
// another in their constructors -- has every member after the first starting
// inside its predecessor's last page.  Every one of those is a refusal, on
// every deck, forever.  A worker-recycled batch run re-pays it per deck.
//
// So the buffers the backends page-lock get storage no other allocation can
// live in: the block is page-aligned and its length is rounded up to whole
// pages, which makes the page interval of ANY sub-range of it disjoint from
// every other such block.  Two consequences worth stating: the registry's
// containment test then always succeeds for a repeat request, and a caller that
// asks for a sub-range (jnet's first nsurf*ng entries, say) can never collide
// with a caller that asks for the whole thing.
//
// The payload is placed SKEW bytes into the block, with skew rotating in
// 64-byte steps.  Page-aligning a dozen arrays that are indexed together would
// otherwise map the same element of each onto the same L1 cache set; the skew
// is the standard fix and costs nothing, because the block is page-exclusive
// either way.  skew < kHostPinPageSize is what lets the free path recover the
// block by masking the payload address -- no header, no bookkeeping, and a
// deallocation that cannot disagree with its allocation.
inline constexpr std::size_t kHostPinSkewStride = 64;

inline std::atomic<std::size_t>& rasberyHostPinSkewRef() {
    static std::atomic<std::size_t> next{0};
    return next;
}

/// Storage for `bytes` whose pages belong to this block alone.  Uninitialised,
/// like operator new[]; the array helpers below add the zero fill where the
/// call site had one.
inline void* rasberyPageExclusiveAlloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    const std::size_t slots = static_cast<std::size_t>(kHostPinPageSize) / kHostPinSkewStride;
    const std::size_t skew =
        (rasberyHostPinSkewRef().fetch_add(1, std::memory_order_relaxed) % slots) *
        kHostPinSkewStride;
    const std::size_t total = (skew + bytes + static_cast<std::size_t>(kHostPinPageSize) - 1) &
                              ~(static_cast<std::size_t>(kHostPinPageSize) - 1);
    char* const block = static_cast<char*>(
        ::operator new[](total, std::align_val_t{static_cast<std::size_t>(kHostPinPageSize)}));
    return block + skew;
}

/// Free a rasberyPageExclusiveAlloc payload.  The block base is the payload
/// address masked to its page, because the skew is always below one page.
inline void rasberyPageExclusiveFree(void* payload) noexcept {
    if (payload == nullptr) return;
    void* const block = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(payload) &
                                                ~(kHostPinPageSize - 1));
    ::operator delete[](block, std::align_val_t{static_cast<std::size_t>(kHostPinPageSize)});
}

/// `new T[count]` with page-exclusive backing.  Scalar arrays only -- the free
/// path runs no destructors, which is the same contract the raw `new double[]`
/// arrays these replace already had.
template <class T>
inline T* rasberyPageExclusiveArray(std::size_t count) {
    static_assert(std::is_trivially_default_constructible_v<T> &&
                      std::is_trivially_destructible_v<T>,
                  "page-exclusive arrays hold trivial scalars only");
    return static_cast<T*>(rasberyPageExclusiveAlloc(count * sizeof(T)));
}

/// `new T[count]{}` with page-exclusive backing.
template <class T>
inline T* rasberyPageExclusiveZeroedArray(std::size_t count) {
    T* const data = rasberyPageExclusiveArray<T>(count);
    if (data != nullptr) std::fill_n(data, count, T{});
    return data;
}

/// Matching `delete[]`.
template <class T>
inline void rasberyPageExclusiveDeleteArray(T* data) noexcept {
    rasberyPageExclusiveFree(data);
}

/// std::vector allocator over the same storage, for the page-locked buffers
/// that are vectors rather than raw arrays (BICGCMFD's sweep staging).
template <class T>
struct PageExclusiveAllocator {
    using value_type = T;

    PageExclusiveAllocator() noexcept = default;
    template <class U>
    PageExclusiveAllocator(const PageExclusiveAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* const p = rasberyPageExclusiveAlloc(n * sizeof(T));
        if (p == nullptr) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { rasberyPageExclusiveFree(p); }

    template <class U>
    bool operator==(const PageExclusiveAllocator<U>&) const noexcept {
        return true;
    }
    template <class U>
    bool operator!=(const PageExclusiveAllocator<U>&) const noexcept {
        return false;
    }
};

/// A std::vector whose storage is page-exclusive.  Same interface, same
/// value_type, same .data() contract -- only the pages underneath differ.
template <class T>
using PageExclusiveVector = std::vector<T, PageExclusiveAllocator<T>>;

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

    /// Static string naming the call site that REGISTERED this range (never a
    /// heap string -- the record keeps the pointer, not a copy).  Only a
    /// diagnostic: it is what lets a RASBERY_PIN_DEBUG=1 refusal name both
    /// halves of the colliding pair instead of two bare addresses.
    const char* tag = nullptr;
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
    /// Registrations retaken WIDER for their own sole owner (see the safe
    /// upgrade in rasberyPinHost).  A subset of registered_ranges, carried
    /// separately so a receipt shows the path was taken at all.
    unsigned long long upgraded_ranges       = 0;
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

/// RASBERY_PIN_DEBUG=1 prints one line per Sec 6.2 refusal and per safe
/// upgrade, naming the call-site tag and byte count on BOTH sides.  That is the
/// colliding-pair table, emitted by the run that has the addresses rather than
/// reconstructed from a receipt afterwards.  Off by default -- it is a
/// std::cerr write per event.
inline bool rasberyHostPinDebug() {
    static const bool debug = [] {
        const char* value = std::getenv("RASBERY_PIN_DEBUG");
        if (value == nullptr) return false;
        const std::string s(value);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
                 s == "FALSE");
    }();
    return debug;
}

/// RASBERY_PIN_UPGRADE=0 disables the safe-upgrade path in rasberyPinHost and
/// puts the wider-request case back on the plain Sec 6.2 refusal.  The kill
/// switch exists because the upgrade is the one place the registry unregisters
/// a live record outside a destructor; with page-exclusive storage (above) it
/// should never fire on a default run, and upgraded_ranges in the receipt says
/// whether it did.
inline bool rasberyHostPinUpgradeEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("RASBERY_PIN_UPGRADE");
        if (value == nullptr) return true;
        const std::string s(value);
        return !(s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE" ||
                 s == "no");
    }();
    return enabled;
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

/// One line per Sec 6.2 refusal under RASBERY_PIN_DEBUG=1.  Caller holds the
/// registry lock, so the line is atomic with respect to other pin requests.
inline void logRejectionLocked(std::uintptr_t base, std::size_t bytes,
                               std::uintptr_t page_begin, std::uintptr_t page_end,
                               const char* tag, const std::vector<RecordIter>& hits) {
    const auto    name = [](const char* t) { return t != nullptr ? t : "?"; };
    std::ostream& os   = std::cerr;
    os << "[RASBERY][PIN][reject] {\"tag\":\"" << name(tag) << "\",\"base\":\"0x"
       << std::hex << base << "\",\"page_begin\":\"0x" << page_begin
       << "\",\"page_end\":\"0x" << page_end << "\"" << std::dec
       << ",\"bytes\":" << bytes << ",\"holders\":[";
    bool first = true;
    for (const RecordIter& it : hits) {
        const PinRecord& record = it->second;
        if (!first) os << ',';
        first = false;
        os << "{\"tag\":\"" << name(record.tag) << "\",\"base\":\"0x" << std::hex
           << reinterpret_cast<std::uintptr_t>(record.registered_address)
           << "\",\"page_begin\":\"0x" << record.conflict_page_begin
           << "\",\"page_end\":\"0x" << record.conflict_page_end << "\"" << std::dec
           << ",\"bytes\":" << record.registered_bytes << ",\"owners\":" << record.owners
           << '}';
    }
    os << "],\"reason\":\""
       << (hits.size() == 1 ? "wider-or-straddling-single-record" : "spans-several-records")
       << "\"}\n";
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
/// runs everywhere).  The ONE exception is the safe upgrade below, which is
/// gated so tightly that neither hazard can be present.
///
/// `tag` names the CALL SITE and must be a string literal or other static
/// string: the record stores the pointer.  It is a diagnostic only -- it takes
/// no part in any decision.
inline bool rasberyPinHost(const void* p, std::size_t bytes, const char* tag = nullptr) {
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
        record.tag                 = tag;
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

        // SAFE UPGRADE.  A request that CONTAINS the single record it overlaps
        // is normally the "existing SUBSET requested" refusal, because widening
        // means unregistering a range somebody else may still be copying from.
        // Both halves of that objection are absent when the record's SOLE owner
        // is this very base and nothing is in flight on it: there is no foreign
        // owner to evict (Sec 6.3 is not touched), and the new range is a strict
        // superset, so no partial-overlap aliasing is created either -- every
        // address the old registration covered stays covered.  Re-registering
        // the owner's own range wider is then the same operation the owner would
        // have got by asking for the wide range first.
        //
        // in_flight is the conservative Sec 6.4 counter, so this holds on the
        // same terms the destructors do: the leases are taken at first use from
        // the thread that then issues the copies, and a wider late request comes
        // from that same thread between drives, not from under one.
        if (rasberyHostPinUpgradeEnabled() && record.in_flight == 0 && record.owners == 1 &&
            record.owner_bases.size() == 1 && record.owner_bases.front() == base &&
            page_begin <= record.conflict_page_begin &&
            page_end >= record.conflict_page_end) {
            const char* const previous_tag   = record.tag;
            const std::size_t previous_bytes = record.registered_bytes;
            host_pin_detail::unregisterLocked(record, counters);
            records.erase(hits.front());

            void* const address = const_cast<void*>(p);
            if (HostPinRegisterHook hook = rasberyHostPinRegisterHookRef(); hook != nullptr) {
                if (hook(address, bytes) != 0) {
                    // The old registration is gone and the new one did not
                    // take: the buffer is simply pageable now, and the owner's
                    // later release finds nothing, which is a no-op by design.
                    ++counters.pageable_fallbacks;
                    return false;
                }
            }
            PinRecord upgraded;
            upgraded.conflict_page_begin = page_begin;
            upgraded.conflict_page_end   = page_end;
            upgraded.registered_address  = address;
            upgraded.registered_bytes    = bytes;
            upgraded.tag                 = tag != nullptr ? tag : previous_tag;
            upgraded.owner_bases.push_back(base);
            upgraded.owners = 1;
            records.emplace(page_begin, std::move(upgraded));
            ++counters.registered_ranges;
            ++counters.upgraded_ranges;
            counters.registered_bytes += bytes;
            if (rasberyHostPinDebug())
                std::cerr << "[RASBERY][PIN][upgrade] {\"tag\":\""
                          << (tag != nullptr ? tag : "?") << "\",\"was\":\""
                          << (previous_tag != nullptr ? previous_tag : "?")
                          << "\",\"bytes\":" << bytes << ",\"was_bytes\":" << previous_bytes
                          << "}\n";
            return true;
        }
    }

    ++counters.overlap_rejections;
    ++counters.pageable_fallbacks;
    if (rasberyHostPinDebug())
        host_pin_detail::logRejectionLocked(base, bytes, page_begin, page_end, tag, hits);
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

/// WP10.4.  Bytes currently page-locked through this registry.
///
/// `rasberyHostPinLiveRanges()` above counts RECORDS, which is the right number
/// for the between-wave lease assertion (a leaked lease is a leaked lease
/// whatever its size) and the wrong one for a memory receipt: 40 records of a
/// megabyte each and 40 of a kilobyte each are the same count and are not the
/// same process.  The soak's RSS finding needs the second number, so it has a
/// name of its own rather than being inferred from the first.
inline std::uint64_t rasberyHostPinLiveBytes() {
    std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
    std::uint64_t bytes = 0;
    for (const auto& entry : rasberyHostPinRecords())
        bytes += static_cast<std::uint64_t>(entry.second.registered_bytes);
    return bytes;
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
       << ",\"stale_evicted\":" << counters.stale_evicted
       << ",\"upgraded_ranges\":" << counters.upgraded_ranges;
}

} // namespace rasbery
