#pragma once

// The [RASBERY][XFER] receipt -- WP13, the host<->device traffic census.
//
// WHY IT EXISTS.  nsys on 238 (v4, single deck, 4377 outers) measured
// 55,183 D2H copies / 28.6 GB, 54,918 H2D copies / 11.7 GB and 9,477
// cudaStreamSynchronize calls at 653 us each -- 6.19 s of host wait against
// 1.61 s of GPU kernel time.  A source census
// (docs/WP13_HOST_DEVICE_TRAFFIC_20260830_KO.md) reconciles the D2H side to
// within ~5 % and the H2D BYTES exactly, but it leaves roughly half the H2D
// CALL count unattributed, because several sites' firing rate is decided by a
// residency predicate whose hit rate cannot be read out of the source.  This
// ledger is how that residual gets settled: the same run that quotes a number
// counts it.
//
// PROCESS-WIDE, relaxed atomics, and read once at shutdown -- the same scope
// and the same reasoning as rasbery::xe::XeGpuTally (src/XeGpuReceipt.h).
// Nothing branches on these values; a counter that cost a fence on the copy
// path would be paying for a receipt with the thing the receipt measures.
//
// WHAT IS AND IS NOT COUNTED.  Only the sites this header was threaded through
// are counted, and `covered` names them, so the receipt can never be mistaken
// for a whole-process total that nsys would confirm.  The gap between this and
// nsys IS the deliverable: it says which backends still have uninstrumented
// copies.
//
// ---------------------------------------------------------------------------
// RASBERY_GPU_XFER_ELIDE -- and why it is NOT in trajectory::kArmEnv
// ---------------------------------------------------------------------------
//
// Every elision behind this flag is a PURE TRANSFER ELISION: a host->device
// copy is skipped only when the bytes it would have written are byte-identical
// to the bytes the device already holds, proved by a host-side shadow of what
// was last uploaded (rasbery::cuda_transfer::ByteExactMirror, or an
// equality-compared std::vector for the integer masks).  The device buffer's
// contents after the elided copy are the same bits as after the copy, so no
// kernel can observe the difference, so no trajectory can move.  That is the
// definition of B0, and a knob that cannot move a trajectory must NOT be in the
// case key -- listing it would say it could, and would fork the evaluator's
// cache for two runs that are the same run.
//
// THE ONE PRECONDITION, stated so it can be checked rather than assumed: the
// device buffer must have NO device-side writer.  A shadow of what the HOST
// last sent is not a shadow of what the DEVICE holds if a kernel wrote it in
// between.  Each elision site below names the buffer and why it is
// device-read-only; `sweep_halt` is the one that is NOT (initialize_solver_state
// raises it and issueSweepDownloads memsets it), and it is deliberately absent
// from the elided set.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string>
#include <vector>

// Named explicitly rather than relying on nvcc's implicit pre-include: this
// header is included AHEAD of <cuda_runtime.h> by several of the backends and
// the wrappers' signatures are made of its types.  Outside a CUDA translation
// unit the wrappers are not declared at all, which is what lets main.cpp and
// the CPU-only stubs include this header for the receipt alone.
#if defined(__CUDACC__) || defined(RASBERY_XFER_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace rasbery::xfer {

struct Ledger {
    /// Device->host copies ISSUED at an instrumented site, and their payload.
    std::atomic<unsigned long long> d2h_calls{0};
    std::atomic<unsigned long long> d2h_bytes{0};
    /// Host->device copies ISSUED at an instrumented site, and their payload.
    std::atomic<unsigned long long> h2d_calls{0};
    std::atomic<unsigned long long> h2d_bytes{0};
    /// cudaStreamSynchronize calls taken at an instrumented site.
    std::atomic<unsigned long long> syncs{0};
    /// Copies NOT issued because the shadow said the device already held the
    /// bytes, and the payload they would have carried.  These are the saving,
    /// and they are counted at the elision site itself.
    std::atomic<unsigned long long> elided_calls{0};
    std::atomic<unsigned long long> elided_bytes{0};
    /// Elision opportunities TESTED.  `elided_calls / tested` is the hit rate;
    /// without it a zero saving cannot be told apart from a flag that never
    /// reached the site.
    std::atomic<unsigned long long> tested_calls{0};
};

/// One ledger per process, and the only global in this feature.
inline Ledger& ledger() {
    static Ledger l;
    return l;
}

/// Opt-IN, unset means off -- the same spelling every other RASBERY_GPU_* flag
/// reader in the tree uses (CudaBICGBackend.cu / CudaXsReconBackend.cu
/// envFlagEnabled), duplicated here rather than shared because this header is
/// included by .cu and .cpp translation units that do not see either of them.
inline bool elideEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_XFER_ELIDE");
        if (v == nullptr) return false;
        const std::string s(v);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
                 s == "FALSE");
    }();
    return on;
}

inline void countH2D(std::size_t bytes) {
    Ledger& l = ledger();
    l.h2d_calls.fetch_add(1, std::memory_order_relaxed);
    l.h2d_bytes.fetch_add(static_cast<unsigned long long>(bytes), std::memory_order_relaxed);
}

inline void countD2H(std::size_t bytes) {
    Ledger& l = ledger();
    l.d2h_calls.fetch_add(1, std::memory_order_relaxed);
    l.d2h_bytes.fetch_add(static_cast<unsigned long long>(bytes), std::memory_order_relaxed);
}

inline void countSync() {
    ledger().syncs.fetch_add(1, std::memory_order_relaxed);
}

/// An elision site was reached and the shadow was consulted.  Called on BOTH
/// outcomes; `hit` says which.
inline void countElisionTest(bool hit, std::size_t bytes) {
    Ledger& l = ledger();
    l.tested_calls.fetch_add(1, std::memory_order_relaxed);
    if (!hit) return;
    l.elided_calls.fetch_add(1, std::memory_order_relaxed);
    l.elided_bytes.fetch_add(static_cast<unsigned long long>(bytes), std::memory_order_relaxed);
}

/// Print the receipt at all?  Yes when anything was counted -- which is every
/// GPU run, arm on or off -- and yes when the arm is on even if nothing was
/// counted, because "the flag was set and no site was reached" is the one
/// answer the receipt must be able to give.  A CPU-only run prints nothing.
inline bool receiptWanted() {
    const Ledger& l = ledger();
    return elideEnabled() || l.h2d_calls.load(std::memory_order_relaxed) != 0 ||
           l.d2h_calls.load(std::memory_order_relaxed) != 0;
}

/// Write the receipt's fields (no braces, no tag) so main.cpp can wrap them the
/// way it wraps every other receipt.
inline void appendXferReceiptFields(std::ostream& os) {
    const Ledger& l = ledger();
    const unsigned long long tested = l.tested_calls.load(std::memory_order_relaxed);
    const unsigned long long elided = l.elided_calls.load(std::memory_order_relaxed);
    // A hit rate with no tests is not zero, it is undefined -- reporting 0.0
    // would read as "the flag was on and nothing was elidable", which is the
    // opposite of "the flag never reached a site".  -1 is the not-measured
    // value, the same convention anderson_accept_rate uses.
    const double hit_rate =
        (tested > 0) ? static_cast<double>(elided) / static_cast<double>(tested) : -1.0;
    os << "\"elide_arm\":" << (elideEnabled() ? 1 : 0)
       << ",\"d2h_calls\":" << l.d2h_calls.load(std::memory_order_relaxed)
       << ",\"d2h_bytes\":" << l.d2h_bytes.load(std::memory_order_relaxed)
       << ",\"h2d_calls\":" << l.h2d_calls.load(std::memory_order_relaxed)
       << ",\"h2d_bytes\":" << l.h2d_bytes.load(std::memory_order_relaxed)
       << ",\"syncs\":" << l.syncs.load(std::memory_order_relaxed)
       << ",\"elided_calls\":" << elided
       << ",\"elided_bytes\":" << l.elided_bytes.load(std::memory_order_relaxed)
       << ",\"elision_tests\":" << tested << ",\"elision_hit_rate\":" << hit_rate
       // WP13.1: every cudaMemcpy*/cuda*Synchronize on the single-deck GPU path
       // now goes through the wrappers below, so the aggregate above is no
       // longer a hand-picked subset.  `covered` says so by naming the FILES
       // rather than a list of hand-instrumented points, and
       // tools/test_xfer_ledger_contract.py is what keeps that claim true.
       << ",\"covered\":\"CudaXsReconBackend.cu,CudaOuterGraph.cu,CudaBICGBackend.cu,"
          "CudaCramBackend.cu,CudaPprBackend.cu,GpuPhysicsArenaCuda.cu\"";
}


// ===========================================================================
// WP13.1 -- THE SITE LEDGER (RASBERY_XFER_LEDGER=1)
// ===========================================================================
//
// WHY A SECOND LAYER.  The aggregate above answers "how much", and WP13 used
// it to prove that the source census could not answer "where": 7.0 GB of H2D
// across ~25,000 copies and ~6,100 cudaStreamSynchronize (4.0 s of an 11.2 s
// wall) had no name, because the sites that issue them fire on a RUNTIME
// predicate.  nsys gives totals only.  This layer gives the same totals BROKEN
// DOWN BY CALL SITE, so the residual is settled by the run that quotes it.
//
// THE CONTRACT, and it is the whole reason this is safe to land:
//
//   * B0 BY CONSTRUCTION.  Every wrapper forwards to exactly the CUDA call the
//     site used to make, with the same arguments, in the same order, on the
//     same stream, and returns the same cudaError_t.  It adds counter
//     arithmetic AFTER the call and nothing else.  No data, no ordering and no
//     stream is touched, so no trajectory can move -- and, like the elision
//     arm, RASBERY_XFER_LEDGER is therefore NOT in trajectory::kArmEnv.
//
//   * ONE BRANCH WHEN OFF.  With RASBERY_XFER_LEDGER unset the per-site table
//     is never read or written; the cost is the aggregate counters that were
//     already there plus one predictable branch on a cached static bool.  The
//     clock reads that time the synchronises happen ONLY inside that branch --
//     a receipt must not be the reason the thing it measures got slower.
//
// SITES ARE STRING-LITERAL TAGS, `scope` + `leaf`, and the pair is the key.
// `scope` names file:function, `leaf` names the buffer.  Both must be string
// literals (static storage, stable address) because the table is keyed BY
// POINTER: a hash of two pointers is O(1) on the copy path, where a strcmp
// over 200 rows would not be.  Rows are rendered "scope:leaf".
//
// WHAT THE RECEIPT CANNOT SEE, stated here so the number is never misread: a
// copy RECORDED INTO A CUDA GRAPH is counted once, at capture, not once per
// replay.  Those sites carry "(captured)" in their scope, and their rows are
// therefore a lower bound whose replay multiplier is the graph's launch count.

/// Per-site direction.  A site has exactly one; the first writer sets it.
enum class XferDir : std::uint16_t { kH2D = 0, kD2H = 1, kD2D = 2, kSync = 3, kOther = 4 };

inline const char* dirName(XferDir d) {
    switch (d) {
        case XferDir::kH2D: return "h2d";
        case XferDir::kD2H: return "d2h";
        case XferDir::kD2D: return "d2d";
        case XferDir::kSync: return "sync";
        default: return "other";
    }
}

/// Open-addressed, fixed capacity, never resized.  512 rows against ~200 tags
/// keeps the load factor under 0.4, and a table that cannot grow is a table
/// that cannot allocate on the copy path.
inline constexpr std::size_t kSiteCapacity = 512;

struct SiteRow {
    std::atomic<const char*> scope{nullptr};
    std::atomic<const char*> leaf{nullptr};
    std::atomic<std::uint16_t> dir{static_cast<std::uint16_t>(XferDir::kOther)};
    std::atomic<unsigned long long> calls{0};
    std::atomic<unsigned long long> bytes{0};
    std::atomic<unsigned long long> ns{0};
};

inline SiteRow* siteTable() {
    static SiteRow t[kSiteCapacity];
    return t;
}

/// Set once, at the first tagged call, and read on every one after it.
inline bool ledgerEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_XFER_LEDGER");
        if (v == nullptr) return false;
        const std::string s(v);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
                 s == "FALSE");
    }();
    return on;
}

/// Overflow is REPORTED, not silently dropped: a table that quietly lost rows
/// would make the ledger's totals disagree with its own rows and there would be
/// nothing in the output to say why.
inline std::atomic<unsigned long long>& siteOverflow() {
    static std::atomic<unsigned long long> n{0};
    return n;
}

inline SiteRow* findOrCreateRow(const char* scope, const char* leaf, XferDir dir) {
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(scope);
    const std::uintptr_t b = reinterpret_cast<std::uintptr_t>(leaf);
    std::size_t h = static_cast<std::size_t>((a * 1315423911ull) ^ (b * 2654435761ull));
    SiteRow* const t = siteTable();
    for (std::size_t probe = 0; probe < kSiteCapacity; ++probe) {
        SiteRow& r = t[(h + probe) % kSiteCapacity];
        const char* s0 = r.scope.load(std::memory_order_acquire);
        if (s0 == nullptr) {
            const char* expect = nullptr;
            if (r.scope.compare_exchange_strong(expect, scope, std::memory_order_acq_rel)) {
                r.leaf.store(leaf, std::memory_order_release);
                r.dir.store(static_cast<std::uint16_t>(dir), std::memory_order_relaxed);
                return &r;
            }
            s0 = r.scope.load(std::memory_order_acquire);
        }
        if (s0 != scope) continue;
        // The winner of the CAS publishes `leaf` a beat after `scope`; a loser
        // that arrives in between must wait rather than compare against null,
        // or it would walk past its own row and claim a second one.
        const char* l0 = r.leaf.load(std::memory_order_acquire);
        while (l0 == nullptr) l0 = r.leaf.load(std::memory_order_acquire);
        if (l0 == leaf) return &r;
    }
    siteOverflow().fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

inline void recordSite(const char* scope, const char* leaf, XferDir dir, std::size_t bytes,
                       unsigned long long elapsed_ns) {
    SiteRow* const r = findOrCreateRow(scope, leaf, dir);
    if (r == nullptr) return;
    r->calls.fetch_add(1, std::memory_order_relaxed);
    if (bytes != 0)
        r->bytes.fetch_add(static_cast<unsigned long long>(bytes), std::memory_order_relaxed);
    if (elapsed_ns != 0) r->ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
}

/// A transfer that CANNOT go through a wrapper -- a copy the caller records into
/// a graph by hand, a cudaMemcpyToSymbol whose signature has no stream -- still
/// gets a row, declared at the site with a comment saying why.
inline void note(const char* scope, const char* leaf, XferDir dir, std::size_t bytes) {
    if (dir == XferDir::kH2D) countH2D(bytes);
    else if (dir == XferDir::kD2H) countD2H(bytes);
    else if (dir == XferDir::kSync) countSync();
    if (!ledgerEnabled()) return;
    recordSite(scope, leaf, dir, bytes, 0);
}

#if defined(__CUDACC__) || defined(RASBERY_XFER_HAS_CUDA)

inline XferDir dirOf(cudaMemcpyKind kind) {
    switch (kind) {
        case cudaMemcpyHostToDevice: return XferDir::kH2D;
        case cudaMemcpyDeviceToHost: return XferDir::kD2H;
        case cudaMemcpyDeviceToDevice: return XferDir::kD2D;
        default: return XferDir::kOther;
    }
}

/// THE WRAPPER.  Forwards first, counts second; the return value is the CUDA
/// call's own, unmodified, so `RASBERY_CUDA_TRY` / `CUDA_CHECK` at the site see
/// exactly what they saw before.
inline cudaError_t memcpyAsync(const char* scope, const char* leaf, void* dst,
                               const void* src, std::size_t bytes, cudaMemcpyKind kind,
                               cudaStream_t stream) {
    const cudaError_t rc = ::cudaMemcpyAsync(dst, src, bytes, kind, stream);
    const XferDir dir = dirOf(kind);
    if (dir == XferDir::kH2D) countH2D(bytes);
    else if (dir == XferDir::kD2H) countD2H(bytes);
    if (ledgerEnabled()) recordSite(scope, leaf, dir, bytes, 0);
    return rc;
}

inline cudaError_t memcpy(const char* scope, const char* leaf, void* dst, const void* src,
                          std::size_t bytes, cudaMemcpyKind kind) {
    // The BLOCKING form: it is its own synchronise, so the elapsed time is as
    // much a part of its cost as the bytes are, and it is measured for the same
    // reason streamSync's is.
    const XferDir dir = dirOf(kind);
    if (!ledgerEnabled()) {
        const cudaError_t rc = ::cudaMemcpy(dst, src, bytes, kind);
        if (dir == XferDir::kH2D) countH2D(bytes);
        else if (dir == XferDir::kD2H) countD2H(bytes);
        return rc;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t rc = ::cudaMemcpy(dst, src, bytes, kind);
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    if (dir == XferDir::kH2D) countH2D(bytes);
    else if (dir == XferDir::kD2H) countD2H(bytes);
    recordSite(scope, leaf, dir, bytes, static_cast<unsigned long long>(dt < 0 ? 0 : dt));
    return rc;
}

inline cudaError_t streamSync(const char* scope, const char* leaf, cudaStream_t stream) {
    countSync();
    if (!ledgerEnabled()) return ::cudaStreamSynchronize(stream);
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t rc = ::cudaStreamSynchronize(stream);
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    recordSite(scope, leaf, XferDir::kSync, 0,
               static_cast<unsigned long long>(dt < 0 ? 0 : dt));
    return rc;
}

inline cudaError_t deviceSync(const char* scope, const char* leaf) {
    countSync();
    if (!ledgerEnabled()) return ::cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t rc = ::cudaDeviceSynchronize();
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    recordSite(scope, leaf, XferDir::kSync, 0,
               static_cast<unsigned long long>(dt < 0 ? 0 : dt));
    return rc;
}

inline cudaError_t eventSync(const char* scope, const char* leaf, cudaEvent_t ev) {
    countSync();
    if (!ledgerEnabled()) return ::cudaEventSynchronize(ev);
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t rc = ::cudaEventSynchronize(ev);
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    recordSite(scope, leaf, XferDir::kSync, 0,
               static_cast<unsigned long long>(dt < 0 ? 0 : dt));
    return rc;
}

#endif // __CUDACC__ || RASBERY_XFER_HAS_CUDA

// ---------------------------------------------------------------------------
// The [RASBERY][XFER][LEDGER] receipt
// ---------------------------------------------------------------------------

struct SiteSnapshot {
    const char* scope;
    const char* leaf;
    XferDir dir;
    unsigned long long calls;
    unsigned long long bytes;
    unsigned long long ns;
};

inline std::vector<SiteSnapshot> snapshotSites() {
    std::vector<SiteSnapshot> out;
    SiteRow* const t = siteTable();
    for (std::size_t i = 0; i < kSiteCapacity; ++i) {
        const char* scope = t[i].scope.load(std::memory_order_acquire);
        if (scope == nullptr) continue;
        const char* leaf = t[i].leaf.load(std::memory_order_acquire);
        out.push_back(SiteSnapshot{
            scope, leaf == nullptr ? "" : leaf,
            static_cast<XferDir>(t[i].dir.load(std::memory_order_relaxed)),
            t[i].calls.load(std::memory_order_relaxed),
            t[i].bytes.load(std::memory_order_relaxed),
            t[i].ns.load(std::memory_order_relaxed)});
    }
    return out;
}

inline void writeSiteRows(std::ostream& os, const std::vector<SiteSnapshot>& rows) {
    os << '[';
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const SiteSnapshot& r = rows[i];
        if (i != 0) os << ',';
        os << "{\"site\":\"" << r.scope << ':' << r.leaf << "\",\"dir\":\"" << dirName(r.dir)
           << "\",\"calls\":" << r.calls << ",\"bytes\":" << r.bytes << ",\"ns\":" << r.ns
           << '}';
    }
    os << ']';
}

/// Four lines, not one: a 200-row table on a single line is a line no one can
/// read in a terminal, and the totals -- the row that gets compared with nsys --
/// would be buried in the middle of it.
inline void printLedgerReceipt(std::ostream& os) {
    if (!ledgerEnabled()) return;
    std::vector<SiteSnapshot> rows = snapshotSites();

    unsigned long long h2d_calls = 0, h2d_bytes = 0, d2h_calls = 0, d2h_bytes = 0;
    unsigned long long d2d_calls = 0, d2d_bytes = 0, sync_calls = 0, sync_ns = 0;
    for (const SiteSnapshot& r : rows) {
        switch (r.dir) {
            case XferDir::kH2D: h2d_calls += r.calls; h2d_bytes += r.bytes; break;
            case XferDir::kD2H: d2h_calls += r.calls; d2h_bytes += r.bytes; break;
            case XferDir::kD2D: d2d_calls += r.calls; d2d_bytes += r.bytes; break;
            case XferDir::kSync: sync_calls += r.calls; sync_ns += r.ns; break;
            default: break;
        }
    }

    os << "[RASBERY][XFER][LEDGER] {\"sites\":" << rows.size()
       << ",\"overflow\":" << siteOverflow().load(std::memory_order_relaxed)
       << ",\"h2d_calls\":" << h2d_calls << ",\"h2d_bytes\":" << h2d_bytes
       << ",\"d2h_calls\":" << d2h_calls << ",\"d2h_bytes\":" << d2h_bytes
       << ",\"d2d_calls\":" << d2d_calls << ",\"d2d_bytes\":" << d2d_bytes
       << ",\"sync_calls\":" << sync_calls << ",\"sync_ns\":" << sync_ns << "}"
       << std::endl;

    std::sort(rows.begin(), rows.end(), [](const SiteSnapshot& a, const SiteSnapshot& b) {
        return a.bytes != b.bytes ? a.bytes > b.bytes : a.calls > b.calls;
    });
    os << "[RASBERY][XFER][LEDGER][BY_BYTES] ";
    writeSiteRows(os, rows);
    os << std::endl;

    std::sort(rows.begin(), rows.end(), [](const SiteSnapshot& a, const SiteSnapshot& b) {
        return a.calls != b.calls ? a.calls > b.calls : a.bytes > b.bytes;
    });
    os << "[RASBERY][XFER][LEDGER][BY_CALLS] ";
    writeSiteRows(os, rows);
    os << std::endl;

    std::sort(rows.begin(), rows.end(), [](const SiteSnapshot& a, const SiteSnapshot& b) {
        return a.ns != b.ns ? a.ns > b.ns : a.calls > b.calls;
    });
    os << "[RASBERY][XFER][LEDGER][BY_SYNC_NS] ";
    writeSiteRows(os, rows);
    os << std::endl;
}

} // namespace rasbery::xfer
