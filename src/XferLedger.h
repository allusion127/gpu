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

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <string>

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
       << ",\"covered\":\"xsrecon.stage,xsrecon.drain,flatxs.inputs,flatxs.download,"
          "cmfd.slotmap,cmfd.masks\"";
}

} // namespace rasbery::xfer
