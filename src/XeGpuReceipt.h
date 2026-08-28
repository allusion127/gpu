#pragma once

// The [RASBERY][XE_GPU] receipt -- Rev.7.1 Task 13.
//
// WHY A RECEIPT AND NOT A LOG LINE.  The one thing an A/B on this arm cannot
// survive is not knowing whether the arm ran.  `RASBERY_GPU_XE=1` on a machine
// whose device refused, or on a deck outside the kernel's fixed NG/NISO, falls
// back to the host loop silently and correctly -- and then the "device" run and
// the "host" run are the same run, and every number measured from them is void.
// device_updates == 0 with the flag set is exactly that situation, said out
// loud.  Same G0 role XSRECON's nodes_solved plays.
//
// PROCESS-WIDE, AND THAT IS THE RIGHT SCOPE FOR THIS ONE.  In --batch-mode each
// deck is its own Driver with its own XSSet, its own backend and its own
// Anderson history; these counters sum over all of them because the question
// they answer -- "did the device arm fire, how often did it fall back, did the
// acceptance rate move" -- is a question about the run.  Nothing here is per-deck
// STATE; a counter is not a buffer, and the batch bug this tree just fixed was a
// process-wide slot-0 BUFFER that every Driver adopted.
//
// Relaxed atomics: nothing branches on these values, they are read once at
// shutdown, and a counter that cost a fence on the Xe path would be paying for
// a receipt with the thing the receipt is measuring.

#include <atomic>
#include <ostream>

namespace rasbery::xe {

struct XeGpuTally {
    /// Xe steps taken while the arm was armed -- device_updates + host_fallbacks.
    std::atomic<unsigned long long> xe_updates{0};
    /// Of those, the ones that completed on the device.
    std::atomic<unsigned long long> device_updates{0};
    /// Of those, the ones a device refusal handed back to the host loop.
    std::atomic<unsigned long long> host_fallbacks{0};
    /// Anderson candidates built and committed BY THE DEVICE ARM.  The host
    /// telemetry counts the same events per statepoint; these are the run-total
    /// the +-10 %p acceptance-rate gate is read off.
    std::atomic<unsigned long long> aa_proposed{0};
    std::atomic<unsigned long long> aa_accepted{0};
    /// History discards at the four map-moving edges (damper engagement, T/H
    /// commit, search commit, cascade re-arm).  The replay gate requires this to
    /// match the host's count exactly: a missed edge leaves the OLD map's
    /// difference columns alive and fits its curvature onto the new one.
    std::atomic<unsigned long long> reset_edges{0};
};

/// One tally per process, and the only global in the arm.
inline XeGpuTally& xeGpuTally() {
    static XeGpuTally t;
    return t;
}

/// Write the receipt's fields (no braces, no tag) so the caller can wrap them
/// the way every other receipt in main.cpp is wrapped.
inline void appendXeGpuReceiptFields(std::ostream& os) {
    const XeGpuTally&        t         = xeGpuTally();
    const unsigned long long proposed  = t.aa_proposed.load(std::memory_order_relaxed);
    const unsigned long long accepted  = t.aa_accepted.load(std::memory_order_relaxed);
    // An acceptance rate with no proposals is not zero, it is undefined -- and
    // reporting 0.0 there would read as "the arm rejected everything", which is
    // the opposite of what happened.  -1 is the "not measured" value the gate
    // scripts test for.
    const double rate =
        (proposed > 0) ? static_cast<double>(accepted) / static_cast<double>(proposed)
                       : -1.0;
    os << "\"xe_updates\":" << t.xe_updates.load(std::memory_order_relaxed)
       << ",\"device_updates\":" << t.device_updates.load(std::memory_order_relaxed)
       << ",\"host_fallbacks\":" << t.host_fallbacks.load(std::memory_order_relaxed)
       << ",\"anderson_proposed\":" << proposed << ",\"anderson_accepted\":" << accepted
       << ",\"anderson_accept_rate\":" << rate
       << ",\"reset_edges\":" << t.reset_edges.load(std::memory_order_relaxed);
}

} // namespace rasbery::xe
