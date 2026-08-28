// Production self-calibration of the CMFD outer contraction mask.
//
// ---------------------------------------------------------------------------
// THE HOLE THIS CLOSES
// ---------------------------------------------------------------------------
//
// CmfdOuterKernel.h carries CMFD_OUTER_FORMS, a record of which multiply-adds
// THE HOST COMPILER fused, so that the device build can reproduce them.  Its
// own header says the value is a property of the build machine and not of the
// physics:
//
//     WSL2 / g++ 13.3 dev box   0x6   (CO_PSI_ACC not fused)
//     238  / Xeon Gold 5317     0x7   (CO_PSI_ACC     fused)
//
// and the TESTS were made to mine the host's value first, so that a gate run on
// the wrong host fails for the right reason.  THE PRODUCTION BINARY WAS NOT.
// It used the baked default, on both machines, and on both machines that
// default is now 0x6 while the host mines 0x7.  So the device updpsi rounded
// `psi += flux*xsnf` in two steps where the host loop fused it, every node,
// every outer -- and RASBERY_GPU_OUTER=1 produced a flux that differed from the
// host loop's in the last bits from the FIRST device outer.
//
// Measured on kngr_238 (the full 35-statepoint deck), device outer ON vs OFF:
//
//     build default 0x6      51 of 644 datasets differ on the trimmed deck,
//                            421 on the full one, and no statepoint is exact
//     mined      0x7          0 of 644 on the trimmed deck; statepoints 1..22
//                            of the full deck bit-identical
//
// One rounding, and it is the difference between a bit-exact acceleration and a
// campaign that cannot tell a real regression from its own build host.
//
// ---------------------------------------------------------------------------
// WHY MINING AND NOT A SECOND BAKED CONSTANT
// ---------------------------------------------------------------------------
//
// A per-host `#if` is the same bug with more branches: it is right until
// somebody upgrades gcc, changes -march, or builds on a third machine, and
// nothing tells them.  What the mask actually asserts is "the shipped host
// bodies and the verbatim CPU quotation agree bit for bit ON THIS BINARY", and
// that is a question this binary can ANSWER at startup, in microseconds, from
// the same fixture the gates use.  So it does, once, and prints what it found.
//
// The env override survives and still wins: a binary built on one host and
// validated against a reference produced on another must be able to say so.
//
// ---------------------------------------------------------------------------
// ITS OWN TRANSLATION UNIT, AND THAT IS NOT AN ACCIDENT
// ---------------------------------------------------------------------------
//
// CmfdOuterReference.cpp holds the verbatim CPU quotations and must never see
// the shipped bodies: with both in one TU gcc common-subexpressions across them
// and changes the QUOTATION's contraction, so the mining scores a reference
// that is no longer the production one (CmfdOuterReference.h says so; Task 4
// paid for the lesson).  This file sees both, which is exactly what the test
// drivers do -- it IS the test driver's mining half, promoted into the binary
// it was always describing.

#include "CmfdOuterFormMine.h"
#include "CmfdOuterReference.h"

namespace rasbery::cmfd {

unsigned long long mineCmfdOuterFormsOnThisHost(bool& sound) {
    // 512 nodes: the size the gate runs at, and the size at which every one of
    // upddhat's six guard branches is reached by the fixture's poisoned
    // surfaces.  A mask mined on operands that never reach a branch would be
    // mined on a fraction of the function.
    static const cmfdref::Fixture fixture = cmfdref::buildFixture(512);
    return cmfdmine::mineStable(fixture, sound);
}

} // namespace rasbery::cmfd
