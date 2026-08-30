// Production self-calibration of the Xe Anderson contraction mask -- Rev.7.1
// Task 13, following CmfdOuterFormMiner.cpp exactly.
//
// THE HOLE THIS CLOSES IS THE ONE THAT FILE DOCUMENTS.  A baked mask is a
// record of the machine it was measured on:
//
//     WSL2 / g++ 13.3 dev box   CMFD_OUTER_FORMS = 0x6
//     238 / Xeon Gold 5317      CMFD_OUTER_FORMS = 0x7
//
// and the CMFD device outer produced a flux that differed from the host loop's
// in the last bits, from the FIRST device outer, for exactly one rounding's
// worth of that mismatch.  The Xe Anderson algebra is new arithmetic with the
// same exposure, so it gets the same treatment from the start rather than after
// a campaign: this binary ANSWERS the question at startup, in microseconds,
// from the same fixture the gates use, and prints what it found.
//
// ITS OWN TRANSLATION UNIT, AND THAT IS NOT AN ACCIDENT.  XeAndersonReference
// .cpp holds the verbatim CPU quotations and must never see the shipped bodies:
// with both in one TU gcc common-subexpressions across them and changes the
// QUOTATION's contraction, so the mining scores a reference that is no longer
// the production one.  This file sees both, which is exactly what the test
// driver does -- it IS the test driver's mining half, promoted into the binary
// it was always describing.

#include "XeFormMine.h"
#include "XeFormMask.h"
#include "GpuFormMask.h"

#include <cstdlib>
#include <iostream>

namespace rasbery::xe {

unsigned long long mineXeFormsOnThisHost(bool& sound, bool& algebra_sound) {
    // 4096 ordinals: above the fuel-node count of the trimmed decks and the
    // same order as kngr_238's, so the inner product's accumulation runs long
    // enough for a contraction difference to survive into the sum rather than
    // being scored on a handful of terms.
    static const xeref::Fixture fixture = xemine::buildMiningFixture(4096);
    return xemine::mineStable(fixture, sound, algebra_sound);
}

unsigned long long xeFormMask() {
    // Resolved ONCE per process, on the host, and passed into kernels as an
    // argument: getenv has no device implementation, and the pure bodies take
    // the mask as a parameter precisely so this is possible.
    //
    // Lazily, and only from the device arm's own call sites, so a run with
    // RASBERY_GPU_XE unset never mines and never prints -- a feature that is
    // off must not add a line to the log of the run it is off in.
    static const unsigned long long mask = [] {
        bool                     sound         = false;
        bool                     algebra_sound = false;
        const unsigned long long mined = mineXeFormsOnThisHost(sound, algebra_sound);
        // THE VERDICT THAT DECIDES THE MASK IS THE SHIPPED CHANNEL'S, and only
        // it.  WP7-C (71092e2) folded four new sites into one `sound`, so a
        // host that could not mine the new ones would have fallen back to
        // XE_FORMS_DEFAULT for the dot and the candidate too -- changing the
        // contraction contract of the RASBERY_GPU_XE arm with no flag moved and
        // no gate asked, which is exactly what the campaign's B0 rule forbids.
        // The WP7-C channel gets its own line and its own consequence below.
        const unsigned long long resolved = gpu::resolveCalibratedFormMask(
            "RASBERY_XE_FORMS", XE_FORMS_DEFAULT, mined, sound, "XE_FORMS");
        if (!algebra_sound)
            std::cerr << "[RASBERY][WARN][FORMS] the WP7-C normal-equations sites (bits "
                         "5..12) could not be mined to zero mismatches on this host; the "
                         "shipped dot/candidate mask above is unaffected, but "
                         "RASBERY_GPU_XE_TXN=1 has no measured contraction contract on "
                         "this build and must not be trusted for a bit-identity claim\n";
        // THE SPLIT, NAMED IN THE RECEIPT.  resolveCalibratedFormMask reports
        // ONE number because CMFD_OUTER_FORMS is one channel; XE_FORMS is two,
        // and a log that printed only the union could not answer the question
        // 238 actually asked -- "did the bits the PRODUCTION arm runs under
        // move?".  On 238 the union moved (0xd -> 0xd2d) and the shipped
        // sub-mask did not (0xd -> 0xd); those are opposite verdicts about the
        // B0 rule and only this line tells them apart.
        std::cerr << "[RASBERY][FORMS] {\"mask\":\"XE_FORMS\",\"resolved\":\"0x"
                  << std::hex << resolved << "\",\"shipped\":\"0x"
                  << (resolved & XE_SHIPPED_FORMS) << "\",\"algebra\":\"0x"
                  << (resolved & XE_ALGEBRA_FORMS) << std::dec
                  << "\",\"live_arm\":\"shipped\",\"txn_arm\":\"resolved\","
                     "\"algebra_sound\":"
                  << (algebra_sound ? 1 : 0) << "}" << std::endl;
        return resolved;
    }();
    return mask;
}

unsigned long long xeShippedFormMask() {
    // No second resolution and no second receipt: the same cached value, with
    // the channel the caller is not allowed to see removed.  See XeFormMask.h
    // for why the removal is here and not left to each kernel body.
    return xeFormMask() & XE_SHIPPED_FORMS;
}

unsigned long long xeHostFormMask() {
    // NO MINING HERE, AND THAT IS THE POINT.  This mask describes how
    // Driver.h's own normal equations are spelled; a fixture in another
    // translation unit cannot measure that (src/XeFormAudit.h), so the value is
    // a build constant plus an override and the MEASUREMENT is the 238 sweep.
    //
    // Resolved once and cached, like every other knob on this path: the four
    // sites read it on every Anderson step and a getenv per step would be both
    // a cost and a second opinion (a setenv mid-run would give two halves of
    // one solve two different contraction contracts).
    static const unsigned long long mask = [] {
        unsigned long long value  = XE_HOST_FORMS_DEFAULT;
        const char*        source = "build_default";
        const char*        raw    = std::getenv("RASBERY_XE_HOST_FORMS");
        if (raw != nullptr) {
            unsigned long long parsed = 0;
            if (gpu::parseFormMask(raw, parsed)) {
                value  = parsed;
                source = "env";
            } else {
                std::cerr << "[RASBERY][WARN][FORMS] RASBERY_XE_HOST_FORMS=\"" << raw
                          << "\" is not a number; ignoring the override.  A malformed "
                             "override must not silently pass for a valid one.\n";
                source = "env_rejected";
            }
        }
        // THE SHIPPED CHANNEL IS UNREACHABLE FROM HERE.  A typo that set bit 3
        // would otherwise change how the CANDIDATE loop contracts on the
        // device, from a knob whose name says "host".
        const unsigned long long kept = value & XE_ALGEBRA_FORMS;
        if (kept != value) {
            std::cerr << "[RASBERY][WARN][FORMS] RASBERY_XE_HOST_FORMS=0x" << std::hex
                      << value << " carries bits outside the algebra channel 0x"
                      << XE_ALGEBRA_FORMS << "; using 0x" << kept << std::dec
                      << ".  The dot and the candidate are device sites and are "
                         "decided by RASBERY_XE_FORMS, not by this knob.\n";
            source = "env_trimmed";
        }
        // The four per-site digits, spelled out.  A sweep over 81 combinations
        // reads THESE; the hex is for the campaign log.
        std::cerr << "[RASBERY][FORMS] {\"mask\":\"XE_HOST_FORMS\",\"value\":\"0x"
                  << std::hex << kept << "\",\"source\":\"" << source
                  << "\",\"build_default\":\"0x" << XE_HOST_FORMS_DEFAULT << std::dec
                  << "\",\"det\":" << xeSiteState(kept, XE_TXN_DET_BIT)
                  << ",\"g0\":" << xeSiteState(kept, XE_TXN_G0_BIT)
                  << ",\"g1\":" << xeSiteState(kept, XE_TXN_G1_BIT)
                  << ",\"proj\":" << xeSiteState(kept, XE_TXN_PROJ_BIT)
                  << "}" << std::endl;
        return kept;
    }();
    return mask;
}

} // namespace rasbery::xe
