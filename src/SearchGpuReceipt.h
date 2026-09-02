#pragma once

// The [RASBERY][SEARCH_GPU] receipt -- WP22 commit 2.
//
// WHY A RECEIPT FOR SO SMALL AN ARM.  Precisely BECAUSE it is small.  The T/H
// arm moves 0.70 s and a run that lost it would be visibly slower; this one
// moves an nxyz-double copy per boron trial, which is inside the noise of a
// single-deck wall.  So there is no timing signal that would notice
// `RASBERY_GPU_SEARCH=1` silently doing nothing, and `device_applies == 0` with
// the flag set is the ONLY way that situation gets said out loud.
//
// THE FIELD THAT IS NOT A COUNT.  `bytes_elided` here is not an estimate: it is
// the exact per-apply payload (nxyz * 8) summed over the applies the device
// actually served.  Quoting a saving from a multiplication the reader has to do
// themselves is how a 160 kB/trial number becomes a paragraph nobody checks.
//
// PROCESS-WIDE, relaxed atomics, read once at shutdown -- the same scope and
// the same reasoning as rasbery::th::ThGpuTally and rasbery::xe::XeGpuTally.

#include <atomic>
#include <ostream>

namespace rasbery::search {

struct SearchGpuTally {
    /// Boron applies taken while the arm was armed -- device + host_fallbacks.
    std::atomic<unsigned long long> applies{0};
    /// Of those, the ones whose per-node write reached the resident device
    /// block instead of an nxyz-double upload.
    std::atomic<unsigned long long> device_applies{0};
    /// Of those, the ones a device refusal left to the plain host broadcast and
    /// the flat-XS backend's usual guarded upload.
    std::atomic<unsigned long long> host_fallbacks{0};
    /// Bytes the guarded bppm upload did not carry because the device block was
    /// already correct.
    std::atomic<unsigned long long> bytes_elided{0};
    /// Trials proposed by the host secant while the arm was armed.  Printed
    /// beside `applies` because they should be equal: a proposal that did not
    /// reach an apply is a trial the search committed and the cross sections
    /// never saw.
    std::atomic<unsigned long long> proposals{0};
    /// Device transfers the PROPOSE step made.  MUST STAY ZERO -- the secant
    /// reads a k_eff the solve already published to the host, and a download
    /// appearing here would mean the arm had grown a round trip whose whole
    /// purpose was to avoid one.  Counted rather than asserted in a comment so
    /// a run can be quoted for it.
    std::atomic<unsigned long long> propose_transfers{0};
};

/// The GRADE of the RASBERY_GPU_SEARCH arm, in the receipt that reports it.
///
/// B0, AND UNLIKE THE T/H ARM THIS ONE IS NOT WAITING ON A MEASUREMENT.  The
/// kernel evaluates no arithmetic: every node receives the same double the host
/// secant proposed, so the device block afterwards holds the bytes the upload
/// would have written.  That is the same argument RASBERY_GPU_XFER_ELIDE rests
/// on and it is a property of the code rather than of a host.
inline constexpr const char* kSearchGpuPolicyNote =
    "RASBERY_GPU_SEARCH=1 is B0 by construction: the boron apply is a broadcast "
    "store of the host secant's own double, so the resident block holds the same "
    "bits the elided upload would have written; the secant and the bracket stay "
    "on the host (docs/WP22_TH_SEARCH_GPU_20260902_KO.md section 7)";

inline SearchGpuTally& searchGpuTally() {
    static SearchGpuTally t;
    return t;
}

/// Print the receipt at all?  Yes when the arm was asked for or actually fired.
inline bool searchReceiptWanted() {
    const SearchGpuTally& t = searchGpuTally();
    return t.applies.load(std::memory_order_relaxed) != 0 ||
           t.host_fallbacks.load(std::memory_order_relaxed) != 0;
}

/// Write the receipt's fields (no braces, no tag).
inline void appendSearchGpuReceiptFields(std::ostream& os) {
    const SearchGpuTally&    t      = searchGpuTally();
    const unsigned long long total  = t.applies.load(std::memory_order_relaxed);
    const unsigned long long device = t.device_applies.load(std::memory_order_relaxed);
    // -1 is "not measured", the convention every other receipt in this tree
    // uses: a device share of 0.0 would read as "the arm refused everything",
    // which is the opposite of "the arm was never reached".
    const double share =
        (total > 0) ? static_cast<double>(device) / static_cast<double>(total) : -1.0;
    os << "\"arm\":" << (total > 0 ? 1 : 0) << ",\"applies\":" << total
       << ",\"device_applies\":" << device << ",\"host_fallbacks\":"
       << t.host_fallbacks.load(std::memory_order_relaxed)
       << ",\"device_share\":" << share
       << ",\"proposals\":" << t.proposals.load(std::memory_order_relaxed)
       << ",\"propose_transfers\":" << t.propose_transfers.load(std::memory_order_relaxed)
       << ",\"bytes_elided\":" << t.bytes_elided.load(std::memory_order_relaxed)
       << ",\"policy_note\":\"" << kSearchGpuPolicyNote << "\"";
}

} // namespace rasbery::search
