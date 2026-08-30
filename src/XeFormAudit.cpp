// WP7-C.  The measurement the mining cannot make -- see src/XeFormAudit.h for
// why it is a translation unit of its own and not four lines in Driver.h.
//
// NOTHING IN THIS FILE MAY BE MOVED INTO A HEADER.  It includes XeKernel.h, and
// the one thing that must never share a translation unit with the shipped body
// is the production block this audits.

#include "XeFormAudit.h"

#include "XeFormMask.h"
#include "XeGpuReceipt.h"
#include "XeKernel.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace rasbery::xe {

namespace {

/// Bit pattern of a double.  Comparing the VALUES would call two contractions
/// that differ in the last bit "equal", which is the entire thing being looked
/// for.
unsigned long long bitsOf(double v) {
    unsigned long long u = 0;
    static_assert(sizeof(u) == sizeof(v), "a double is not eight bytes here");
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

/// One line, once.  A run whose mask is wrong is wrong on most steps, and
/// thousands of identical lines would bury the receipt that summarises them.
std::atomic<bool> g_reported{false};

} // namespace

bool xeFormAuditEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_XE_FORMS_AUDIT");
        return v != nullptr && *v != '\0' && std::string(v) != "0";
    }();
    return on;
}

void auditAndersonFit(const double* dots, int ncol, double min_gram, bool solved,
                      double gamma0, double gamma1, double proj) {
    // TWO MASKS, AND THE AUDIT IS NOW A COMPARISON OF THEM AT THE OPERANDS.
    //
    // `forms` is what the DEVICE transaction will spell the algebra with
    // (mined, RASBERY_XE_FORMS).  `host_forms` is what the production block in
    // Driver.h just spelled it with (RASBERY_XE_HOST_FORMS) -- which is a
    // WRITTEN-DOWN value only since the four host sites were routed through
    // xeSiteSub/xeSiteAdd.  Before that the host's spelling was whatever gcc
    // decided for that inlining context, so this audit could report a
    // mismatch but never name the knob that would close it.  It can now:
    // TXN=1 is B0 against TXN=0 exactly when the two algebra channels are the
    // same number AND this audit sees zero mismatches on a real deck.
    const unsigned long long forms      = xeFormMask();
    const unsigned long long host_forms = xeHostFormMask();
    XeGpuTally&              tally      = xeGpuTally();
    tally.forms_audit_mask.store(forms, std::memory_order_relaxed);
    tally.forms_audit_host_mask.store(host_forms, std::memory_order_relaxed);
    tally.forms_audits.fetch_add(1, std::memory_order_relaxed);

    double     shipped_g0 = 0.0, shipped_g1 = 0.0, shipped_proj = 0.0;
    const bool shipped_solved =
        xeAndersonFit(dots, ncol, min_gram, forms, &shipped_g0, &shipped_g1,
                      &shipped_proj);

    const char* where = nullptr;
    if (shipped_solved != solved)
        where = "solved";
    else if (!solved)
        return; // a refused fit writes no coefficients on either side
    else if (bitsOf(shipped_g0) != bitsOf(gamma0))
        where = "gamma0";
    else if (bitsOf(shipped_g1) != bitsOf(gamma1))
        where = "gamma1";
    else if (bitsOf(shipped_proj) != bitsOf(proj))
        where = "proj";
    if (where == nullptr)
        return;

    tally.forms_audit_mismatch.fetch_add(1, std::memory_order_relaxed);
    bool expected = false;
    if (!g_reported.compare_exchange_strong(expected, true,
                                            std::memory_order_relaxed))
        return;
    std::cerr << "[RASBERY][WARN][FORMS] the mined XE_FORMS mask 0x" << std::hex
              << forms << " (algebra channel 0x" << (forms & XE_ALGEBRA_FORMS)
              << ") does NOT reproduce this build's own Anderson algebra at the "
                 "production call site, which ran under XE_HOST_FORMS 0x"
              << host_forms << std::dec << " (first disagreement: " << where
              << ", ncol=" << ncol
              << ").  RASBERY_GPU_XE_TXN=1 evaluates those four expressions on the "
                 "device under the mined mask, so TXN=1 is N1 against TXN=0 on this "
                 "build and its output is NOT bit-identical.  ";
    // THE ACTIONABLE HALF.  Both spellings are now explicit, so the two ways
    // this can fail are distinguishable at the log line instead of by reading
    // two headers.
    if ((forms & XE_ALGEBRA_FORMS) != host_forms)
        std::cerr << "The two algebra channels are DIFFERENT MASKS: set "
                     "RASBERY_XE_FORMS so its bits 5..12 equal 0x"
                  << std::hex << host_forms << std::dec
                  << ", or re-pin XE_HOST_FORMS_DEFAULT, and re-run this audit.";
    else
        std::cerr << "The two algebra channels are the SAME mask, so the "
                     "disagreement is not a mask choice: one of the two bodies "
                     "is not the other's spelling.  Compare "
                     "Driver.h::TryAndersonXeStepGpu against xe::xeAndersonFit "
                     "site by site.";
    std::cerr << "  See docs/WP7C_XE_TXN_20260831_KO.md section 9 and "
                 "docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md section 8.\n";
}

} // namespace rasbery::xe
