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

namespace rasbery::xe {

unsigned long long mineXeFormsOnThisHost(bool& sound) {
    // 4096 ordinals: above the fuel-node count of the trimmed decks and the
    // same order as kngr_238's, so the inner product's accumulation runs long
    // enough for a contraction difference to survive into the sum rather than
    // being scored on a handful of terms.
    static const xeref::Fixture fixture = xeref::buildFixture(4096);
    return xemine::mineStable(fixture, sound);
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
        bool                     sound = false;
        const unsigned long long mined = mineXeFormsOnThisHost(sound);
        return gpu::resolveCalibratedFormMask("RASBERY_XE_FORMS", XE_FORMS_DEFAULT,
                                              mined, sound, "XE_FORMS");
    }();
    return mask;
}

} // namespace rasbery::xe
