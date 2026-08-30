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
#include <ios>
#include <ostream>

namespace rasbery::xe {

struct XeGpuTally {
    /// Xe steps taken while the arm was armed -- device_updates + host_fallbacks.
    std::atomic<unsigned long long> xe_updates{0};
    /// Of those, the ones that completed on the device.
    std::atomic<unsigned long long> device_updates{0};
    /// Of those, the ones a device refusal handed back to the host loop.
    std::atomic<unsigned long long> host_fallbacks{0};
    /// F10 (review doc Sec 3).  THE FUSED ARM HAD NO TALLY AT ALL.  The three
    /// counters above are the SPLIT arm (RASBERY_GPU_XE).  The fused arm
    /// (RASBERY_GPU_XSRECON) is a second, independent device path through the
    /// same Xe step, and it carried nothing: it could decline every step of
    /// every statepoint and `device_updates:0` would be indistinguishable from
    /// "the split arm was simply off" -- exactly the G0 failure this whole
    /// receipt exists to prevent, one arm to the left of where it was looking.
    ///
    /// A SEPARATE TRIPLE, NOT A SHARED ONE.  Folding the fused arm into
    /// host_fallbacks would charge ONE Xe step twice whenever the split arm
    /// declines and the fused arm is tried next (XSSet.cpp does exactly that),
    /// and `xe_updates == device_updates + host_fallbacks` -- the invariant
    /// that makes the split receipt readable -- would stop holding.  So the
    /// fused arm gets its own sum-to-whole triple, and each arm's receipt is
    /// still an accounting identity on its own.
    std::atomic<unsigned long long> fused_updates{0};
    std::atomic<unsigned long long> fused_device_updates{0};
    std::atomic<unsigned long long> fused_host_fallbacks{0};
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
    // --- WP7 stage C: the per-step sync census, MEASURED -------------------
    //
    // The doc's before/after table is not a model.  These three are counted at
    // the places they happen -- xeSync() at every cudaStreamSynchronize on the
    // Xe stream, countXeD2H() at every device->host transfer the Xe path makes,
    // and xe_device_steps once per committed step, which is the only event that
    // happens exactly once per step on both arms.  Their ratio is the census,
    // and it is read off the same run that claims it.
    std::atomic<unsigned long long> xe_device_steps{0};
    /// Of those, the ones taken as ONE device transaction (RASBERY_GPU_XE_TXN).
    std::atomic<unsigned long long> txn_steps{0};
    /// Transactions that ended in a committed Anderson candidate; the rest
    /// committed the damped Picard image the fallback would have committed.
    std::atomic<unsigned long long> txn_accepted{0};
    /// A transaction the device refused before touching anything, so the
    /// round-tripping arm ran the step instead.  Non-zero means the census
    /// below is a mixture and the A/B is not measuring what it says.
    std::atomic<unsigned long long> txn_declined{0};
    std::atomic<unsigned long long> host_syncs{0};
    std::atomic<unsigned long long> d2h_bytes{0};
    // --- WP7-C: is the mined mask the PRODUCTION contraction? --------------
    //
    // RASBERY_XE_FORMS_AUDIT (src/XeFormAudit.h).  Zero unless the audit was
    // asked for; `forms_audits > 0 && forms_audit_mismatch == 0` is the only
    // form in which the TXN arm's bit-identity claim can be believed on a given
    // build, and it is the form host 181 could not produce.
    std::atomic<unsigned long long> forms_audits{0};
    std::atomic<unsigned long long> forms_audit_mismatch{0};
    /// The mask the audit ran under, so the receipt names what it measured.
    /// Written by the audit and by nothing else: printing xeFormMask() here
    /// would MINE the mask in a run that never touched the device Xe arm, and
    /// add a [RASBERY][FORMS] line to a log whose feature was off.
    std::atomic<unsigned long long> forms_audit_mask{0};
    /// The mask the PRODUCTION BLOCK was spelled under -- Driver.h's own
    /// RASBERY_XE_HOST_FORMS, bits 5..12.  Written by the audit and by nothing
    /// else, for the same reason as the line above.
    ///
    /// WHY BOTH NUMBERS ARE IN THE RECEIPT.  Before the host sites were
    /// barriered there was only one mask to name, because the host's spelling
    /// was whatever gcc decided that build and was not a value anyone could
    /// write down.  It is one now, so `forms_audit_mismatch: 0` has a readable
    /// precondition: the algebra channel of `forms_audit_mask` must equal
    /// `forms_audit_host_mask`, and a receipt where those two differ says
    /// which knob to move rather than only that TXN=1 is N1.
    std::atomic<unsigned long long> forms_audit_host_mask{0};
};

/// The GRADE of the RASBERY_GPU_XE_TXN arm, in the receipt that reports it.
///
/// WP7-C shipped claiming B0 against the round-tripping device arm.  Host 181
/// (2026-08-30) measured otherwise -- different digest, different Xe step count
/// -- with a mask that mined clean, and src/XeFormAudit.h carries the reason:
/// the mask is calibrated against a QUOTATION of Driver.h's algebra, and a
/// quotation is not a call site.  The claim is downgraded here rather than in a
/// document alone, because the number a gate script reads is this one.
inline constexpr const char* kXeTxnPolicyNote =
    "RASBERY_GPU_XE_TXN=1 is N1 against TXN=0: the device normal equations run a "
    "mined contraction mask, not Driver.h's inlined gcc codegen -- set "
    "RASBERY_XE_FORMS_AUDIT=1 to measure the gap on this build "
    "(docs/WP7C_XE_TXN_20260831_KO.md section 9)";

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
       << ",\"fused_updates\":" << t.fused_updates.load(std::memory_order_relaxed)
       << ",\"fused_device_updates\":"
       << t.fused_device_updates.load(std::memory_order_relaxed)
       << ",\"fused_host_fallbacks\":"
       << t.fused_host_fallbacks.load(std::memory_order_relaxed)
       << ",\"anderson_proposed\":" << proposed << ",\"anderson_accepted\":" << accepted
       << ",\"anderson_accept_rate\":" << rate
       << ",\"reset_edges\":" << t.reset_edges.load(std::memory_order_relaxed);

    // WP7-C.  Per-step means per COMMITTED STEP; with none taken the ratios are
    // undefined and -1 says so, for the same reason anderson_accept_rate does.
    const unsigned long long steps = t.xe_device_steps.load(std::memory_order_relaxed);
    const unsigned long long syncs = t.host_syncs.load(std::memory_order_relaxed);
    const unsigned long long d2h   = t.d2h_bytes.load(std::memory_order_relaxed);
    const double             sps =
        (steps > 0) ? static_cast<double>(syncs) / static_cast<double>(steps) : -1.0;
    const double bps =
        (steps > 0) ? static_cast<double>(d2h) / static_cast<double>(steps) : -1.0;
    os << ",\"xe_device_steps\":" << steps
       << ",\"txn_steps\":" << t.txn_steps.load(std::memory_order_relaxed)
       << ",\"txn_accepted\":" << t.txn_accepted.load(std::memory_order_relaxed)
       << ",\"txn_declined\":" << t.txn_declined.load(std::memory_order_relaxed)
       << ",\"host_syncs\":" << syncs << ",\"host_syncs_per_step\":" << sps
       << ",\"d2h_bytes\":" << d2h << ",\"d2h_bytes_per_step\":" << bps;

    // WP7-C.  The audit's two counters and the mask it measured, then the
    // grade.  `forms_audit_mask` prints as a hex string because that is how
    // [RASBERY][FORMS] prints it and how RASBERY_XE_FORMS is written.
    const unsigned long long audits = t.forms_audits.load(std::memory_order_relaxed);
    os << ",\"forms_audits\":" << audits << ",\"forms_audit_mismatch\":"
       << t.forms_audit_mismatch.load(std::memory_order_relaxed)
       << ",\"forms_audit_mask\":\"";
    if (audits > 0) {
        const std::ios_base::fmtflags saved = os.flags();
        os << "0x" << std::hex << t.forms_audit_mask.load(std::memory_order_relaxed);
        os.flags(saved);
    } else {
        os << '~';
    }
    os << "\",\"forms_audit_host_mask\":\"";
    if (audits > 0) {
        const std::ios_base::fmtflags saved = os.flags();
        os << "0x" << std::hex
           << t.forms_audit_host_mask.load(std::memory_order_relaxed);
        os.flags(saved);
    } else {
        os << '~';
    }
    os << "\",\"policy_note\":\"" << kXeTxnPolicyNote << "\"";
}

} // namespace rasbery::xe
