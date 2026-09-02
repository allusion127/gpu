// Production self-calibration of the T/H contraction mask -- WP22.
//
// ITS OWN TRANSLATION UNIT, AND THAT IS NOT AN ACCIDENT.  ThReference.cpp holds
// the verbatim CPU quotations and must never see the shipped bodies: with both
// in one TU gcc common-subexpressions across them and changes the QUOTATION's
// contraction, so the mining would score a reference that is no longer the
// production one.  This file sees both, which is exactly what the test driver
// does -- it IS the test driver's mining half, promoted into the binary it was
// always describing, for the same reason CmfdOuterFormMiner.cpp was.
//
// WHY MINING AND NOT A SECOND BAKED CONSTANT.  A per-host `#if` is the same bug
// with more branches: it is right until somebody upgrades gcc, changes -march,
// or builds on a third machine, and nothing tells them.  What the mask actually
// asserts is "the shipped bodies and the verbatim CPU quotation agree bit for
// bit ON THIS BINARY", and that is a question this binary can ANSWER at startup,
// in milliseconds, from the same fixture the gate uses.  So it does, once, and
// prints what it found.

#include "ThFormMask.h"

#include "GpuFormMask.h"
#include "ThFormMine.h"

#include <iostream>

namespace rasbery::th {

unsigned long long mineThFormsOnThisHost(bool& sound) {
    // 64 channels x 24 planes.  Wide enough that the fixture reaches every
    // branch of the quoted bodies -- the `P_node <= 0` skip, both reflector
    // stacks, the table-ceiling clamp on both the fuel and the kec node, and the
    // sub-50 W/cm linear continuation in GetTfuel -- and small enough that the
    // whole four-seed descent is milliseconds.  A mask mined on operands that
    // never reach a branch would be mined on a fraction of the function.
    static const thref::Fixture fixture = thref::buildFixture(64, 24, 2, 22);
    const unsigned long long    mined   = thmine::mineStable(fixture, sound);

    // AND THE CENSUS THAT MAKES `sound` MEAN SOMETHING.  A zero residual says the
    // mask reproduces the reference; it does not say the reference could tell the
    // alternatives apart.  WP22 shipped a fixture that reached neither x-lerp --
    // 0x54 and 0x57 both scored zero -- and the receipt reported `mined_sound:1`
    // for a mask the deck disagreed with by 866 lines.  A site nobody can pin is
    // now a SENTENCE rather than a silence.
    const unsigned long long dc = thmine::dontCareMask(fixture, mined);
    if (dc != TH_EXPECTED_DONT_CARE) {
        std::cerr << "[RASBERY][WARN][FORMS] TH_FORMS: the mining fixture cannot pin site"
                     " mask 0x"
                  << std::hex << (dc & ~TH_EXPECTED_DONT_CARE) << std::dec
                  << " (expected only TH_HAVG and TH_TFUEL_LINEAR, 0x180).  Those bits of the"
                     " mined mask 0x"
                  << std::hex << mined << std::dec
                  << " are whatever the descent's seed started from, NOT a measurement of"
                     " this host.  src/ThReference.cpp::buildFixture must reach them.\n";
    }
    std::cerr << "[RASBERY][FORMS] {\"mask\":\"TH_FORMS\",\"dontcare\":\"0x" << std::hex << dc
              << std::dec << "\",\"expected_dontcare\":\"0x" << std::hex
              << TH_EXPECTED_DONT_CARE << std::dec << "\"}" << std::endl;
    return mined;
}

unsigned long long thFormMask() {
    static const unsigned long long value = [] {
        bool                     sound = false;
        const unsigned long long mined = mineThFormsOnThisHost(sound);
        return rasbery::gpu::resolveCalibratedFormMask("RASBERY_TH_FORMS", TH_FORMS_DEFAULT,
                                                       mined, sound, "TH_FORMS");
    }();
    return value;
}

} // namespace rasbery::th
