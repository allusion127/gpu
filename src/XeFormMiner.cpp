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
                         "5..12) could not be mined to zero mismatches on this host; "
                         "neither the shipped dot/candidate mask above nor the algebra "
                         "channel composed below takes its value from that mining, so "
                         "this is a diagnostic about the FIXTURE and not about the run\n";

        // ------------------------------------------------------------------
        // THE COMPOSITION -- WHY THE MINED ALGEBRA BITS ARE NOT THE ONES THAT
        // SHIP, AND WHY A PIN IS NO LONGER NEEDED TO GET B0.
        //
        // Host 181, 2026-08-31 (docs/WP7C_XE_TXN_20260831_KO.md section 9.6):
        // the mining answers 0xd3d, the production host block runs 0xac0, and
        // RASBERY_GPU_XE_TXN=1 walked a different trajectory from TXN=0 (1190
        // versus 1195 Xe steps) until a human typed RASBERY_XE_FORMS=0xadd.
        // With that pinned, TXN 0/1 were byte-identical: h5diff 0, digest
        // 88dc35e408c86ad4, 1199 Xe steps on both arms, forms_audit_mismatch 0.
        //
        // 0xadd is not a discovery.  It is (mined & bits 0..4) | (host mask).
        // The pin was a human recomputing, by hand, a number the binary already
        // holds both halves of -- and a contract that only holds while somebody
        // remembers to type it is not a contract.  So it is COMPOSED here:
        //
        //   bits 0..4  (XE_SHIPPED_FORMS): the fixed-partition dot and the
        //     candidate loop.  These are DEVICE sites with no host counterpart
        //     anywhere else in the tree, the fixture reaches them honestly, and
        //     the mining is the best evidence there is about them.  Mined.
        //
        //   bits 5..12 (XE_ALGEBRA_FORMS): the four normal-equations sites.
        //     The device evaluates them under RASBERY_GPU_XE_TXN=1; the HOST
        //     evaluates the very same four expressions on every Xe step of the
        //     RASBERY_GPU_XE arm, through xe::xeSiteSub/xeSiteAdd under
        //     xeHostFormMask().  TXN=1 must reproduce TXN=0, so the host's
        //     spelling IS the specification for those four sites and there is
        //     nothing left for a fixture to decide.  What the mining answers
        //     here is how gcc contracts xeref::refAlgebra in ANOTHER
        //     translation unit -- a quotation, not the call site, which is the
        //     entire argument of src/XeFormAudit.h.  A measurement of the wrong
        //     thing does not become the right thing by being a measurement.
        //
        // So the device mask now agrees with the host call site BY
        // CONSTRUCTION, and RASBERY_XE_FORMS_AUDIT=1 goes from "the only way to
        // discover the gap" to "the way to confirm there is none".
        //
        // WHAT THIS CANNOT MOVE.  Only bits 5..12 change, and the only consumer
        // of those is kXeAndersonSolve inside XsReconBackend::xeTransaction,
        // which Driver.h reaches only behind rasberyGpuXeTxnEnabled().  The
        // production split arm is launched with xeShippedFormMask() -- bits
        // 0..4 -- so a RASBERY_GPU_XE_TXN unset/0 run is bit-for-bit the run it
        // was before this commit; the receipt text below is the whole diff.
        //
        // AN EXPLICIT OVERRIDE STILL WINS VERBATIM.  A human who types
        // RASBERY_XE_FORMS means the number they typed, including its algebra
        // bits: the sweep in section 9.6 item 2 is exactly the procedure of
        // disagreeing with this composition, and it has to stay possible.
        // resolveCalibratedFormMask has already applied the override to
        // `resolved`, so all that is needed here is to know whether it did.
        // ------------------------------------------------------------------
        const char*        env_raw    = std::getenv("RASBERY_XE_FORMS");
        unsigned long long env_parsed = 0;
        const bool         env_pinned =
            (env_raw != nullptr) && gpu::parseFormMask(env_raw, env_parsed);

        // Resolved once here and cached in ITS OWN static, so the number this
        // composition uses is the same number Driver.h's block spells its
        // algebra with -- not a second reading of the same environment that a
        // mid-run setenv could split in two.
        const unsigned long long host_forms = xeHostFormMask();
        const unsigned long long composed =
            (resolved & XE_SHIPPED_FORMS) | (host_forms & XE_ALGEBRA_FORMS);
        const unsigned long long value  = env_pinned ? resolved : composed;
        const char*              source = env_pinned ? "env" : "build_default_composed";

        // THE SPLIT, NAMED IN THE RECEIPT.  resolveCalibratedFormMask reports
        // ONE number because CMFD_OUTER_FORMS is one channel; XE_FORMS is two,
        // and a log that printed only the union could not answer the question
        // 238 actually asked -- "did the bits the PRODUCTION arm runs under
        // move?".  On 238 the union moved (0xd -> 0xd2d) and the shipped
        // sub-mask did not (0xd -> 0xd); those are opposite verdicts about the
        // B0 rule and only this line tells them apart.
        //
        // BOTH COMPONENTS ARE PRINTED, not just the answer.  `mined` and `host`
        // are the two inputs of the composition and `resolved` is the value the
        // run uses; a reviewer reads 0xd3d and 0xac0 off the same line that says
        // 0xadd and checks the arithmetic without a second run or a second file.
        std::cerr << "[RASBERY][FORMS] {\"mask\":\"XE_FORMS\",\"resolved\":\"0x"
                  << std::hex << value << "\",\"source\":\"" << source
                  << "\",\"mined\":\"0x" << mined << "\",\"host\":\"0x" << host_forms
                  << "\",\"composed\":\"0x" << composed << "\",\"shipped\":\"0x"
                  << (value & XE_SHIPPED_FORMS) << "\",\"algebra\":\"0x"
                  << (value & XE_ALGEBRA_FORMS) << std::dec
                  << "\",\"live_arm\":\"shipped\",\"txn_arm\":\"resolved\","
                     "\"algebra_sound\":"
                  << (algebra_sound ? 1 : 0) << "}" << std::endl;
        return value;
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
