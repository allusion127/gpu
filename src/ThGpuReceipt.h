#pragma once

// The [RASBERY][TH_GPU] receipt -- WP22.
//
// WHY A RECEIPT AND NOT A LOG LINE.  The one thing an A/B on this arm cannot
// survive is not knowing whether the arm ran.  `RASBERY_GPU_TH=1` on a machine
// whose device refused, or on a deck outside the kernel's shape, falls back to
// XSSet::SolveTH silently and correctly -- and then the "device" run and the
// "host" run are the same run, and every number measured from them is void.
// `device_updates == 0` with the flag set is exactly that situation, said out
// loud.  Same G0 role XSRECON's nodes_solved and CRAM's gs_iters_mean play.
//
// PROCESS-WIDE, and that is the right scope for this one.  In --batch-mode each
// deck is its own Driver with its own XSSet and its own backend; these counters
// sum over all of them because the question they answer -- "did the device arm
// fire, how often did it fall back, how many bytes stopped crossing the bus" --
// is a question about the run.  Nothing here is per-deck STATE; a counter is not
// a buffer.
//
// Relaxed atomics: nothing branches on these values, they are read once at
// shutdown, and a counter that cost a fence on the T/H path would be paying for
// a receipt with the thing the receipt is measuring.

#include <atomic>
#include <ios>
#include <ostream>

namespace rasbery::th {

struct ThGpuTally {
    /// T/H updates taken while the arm was armed -- device_updates + host_fallbacks.
    std::atomic<unsigned long long> th_updates{0};
    /// Of those, the ones that completed on the device.
    std::atomic<unsigned long long> device_updates{0};
    /// Of those, the ones a device refusal handed back to XSSet::SolveTH.
    std::atomic<unsigned long long> host_fallbacks{0};
    /// Shape the arm last ran at.  Not a sum: `channels * planes == nodes` is the
    /// reader's check that the lane-per-channel mapping covered the whole core,
    /// and a sum over statepoints could not be checked that way.
    std::atomic<unsigned long long> channels{0};
    std::atomic<unsigned long long> nodes{0};
    /// Bytes that did NOT cross the bus because the arm produced its inputs or
    /// consumed its outputs where they already were.  Counted at the elision
    /// site, the same way rasbery::xfer::countElisionTest counts.
    std::atomic<unsigned long long> bytes_elided{0};
    /// Bytes that DID cross, so `bytes_elided` is never read as a total saving.
    /// The tful/tmod/dmod download is here and is deliberately not elided: the
    /// flat-XS branch stream (XSSet::BuildFlatXsStream) resolves on the host and
    /// reads all three, so the arm that removed the T/H numerics did not remove
    /// their consumer.  WP13 tracks that elision; this field is what makes the
    /// residual visible instead of implied.
    std::atomic<unsigned long long> bytes_d2h{0};
    std::atomic<unsigned long long> bytes_h2d{0};
    /// Wall time inside the device entry point, whole call.
    std::atomic<unsigned long long> wall_us{0};
    /// The mask the kernels were launched with, written by the arm on its first
    /// device update and by nothing else -- printing thFormMask() here would MINE
    /// the mask in a run that never armed the feature.
    std::atomic<unsigned long long> forms_mask{0};
    std::atomic<unsigned long long> forms_seen{0};
};

/// The GRADE of the RASBERY_GPU_TH arm, in the receipt that reports it.
///
/// B0 IS A TARGET UNTIL A HOST MEASURES IT, AND THIS STRING SAYS WHICH.  Every
/// expression in ThKernel.h is +, -, * or / -- no transcendental, no libm call
/// whose device implementation could differ -- so bit-identity is REACHABLE, and
/// the form mask plus --fmad=false is the mechanism that reaches it.  Reachable
/// is not measured.  The claim a gate may quote is the digest of a v6 run with
/// the flag on against 1f36e75dc00ed2b4/4377; until that exists the arm is
/// declared N1 here so a reader cannot mistake an intention for a measurement.
inline constexpr const char* kThGpuPolicyNote =
    "RASBERY_GPU_TH=1 targets B0 (all arithmetic is +-*/ under a mined "
    "contraction mask, device TU built --fmad=false); the class is N1 until a "
    "238 run reproduces the flag-off digest "
    "(docs/WP22_TH_SEARCH_GPU_20260902_KO.md section 6)";

/// One tally per process, and the only global in the arm.
inline ThGpuTally& thGpuTally() {
    static ThGpuTally t;
    return t;
}

/// Print the receipt at all?  Yes when the arm was asked for or actually fired.
/// A flag-off log is the log of a build without this feature, which is what
/// "feature-off identity" has to mean for a receipt as well as for a result.
inline bool thReceiptWanted() {
    const ThGpuTally& t = thGpuTally();
    return t.th_updates.load(std::memory_order_relaxed) != 0 ||
           t.host_fallbacks.load(std::memory_order_relaxed) != 0;
}

/// Write the receipt's fields (no braces, no tag) so the caller can wrap them
/// the way every other receipt in this tree is wrapped.
inline void appendThGpuReceiptFields(std::ostream& os) {
    const ThGpuTally&        t      = thGpuTally();
    const unsigned long long total  = t.th_updates.load(std::memory_order_relaxed);
    const unsigned long long device = t.device_updates.load(std::memory_order_relaxed);
    // A device share with no updates is not zero, it is undefined -- reporting
    // 0.0 would read as "the arm refused everything", which is the opposite of
    // "the arm was never reached".  -1 is the not-measured value, the same
    // convention anderson_accept_rate and elision_hit_rate use.
    const double share =
        (total > 0) ? static_cast<double>(device) / static_cast<double>(total) : -1.0;

    os << "\"arm\":" << (total > 0 ? 1 : 0) << ",\"th_updates\":" << total
       << ",\"device_updates\":" << device << ",\"host_fallbacks\":"
       << t.host_fallbacks.load(std::memory_order_relaxed)
       << ",\"device_share\":" << share
       << ",\"channels\":" << t.channels.load(std::memory_order_relaxed)
       << ",\"nodes\":" << t.nodes.load(std::memory_order_relaxed)
       << ",\"bytes_elided\":" << t.bytes_elided.load(std::memory_order_relaxed)
       << ",\"bytes_h2d\":" << t.bytes_h2d.load(std::memory_order_relaxed)
       << ",\"bytes_d2h\":" << t.bytes_d2h.load(std::memory_order_relaxed)
       << ",\"wall_ms\":"
       << static_cast<double>(t.wall_us.load(std::memory_order_relaxed)) / 1000.0
       << ",\"forms_mask\":\"";
    if (t.forms_seen.load(std::memory_order_relaxed) > 0) {
        const std::ios_base::fmtflags saved = os.flags();
        os << "0x" << std::hex << t.forms_mask.load(std::memory_order_relaxed);
        os.flags(saved);
    } else {
        os << '~';
    }
    os << "\",\"policy_note\":\"" << kThGpuPolicyNote << "\"";
}

} // namespace rasbery::th
