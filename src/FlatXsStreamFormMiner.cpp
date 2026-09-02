// Production self-calibration of the WP23 branch-stream contraction mask, and
// resolution of the libm policy -- WP23.1.
//
// ITS OWN TRANSLATION UNIT, AND THAT IS NOT AN ACCIDENT.
// FlatXsStreamReference.cpp holds the verbatim CPU quotation and must never see
// the shipped bodies: with both in one TU gcc common-subexpressions across them
// and changes the QUOTATION's contraction, so the mining would score a
// reference that is no longer the production one.  This file sees both, which
// is exactly what the probe driver does -- it IS the probe's mining half,
// promoted into the binary it was always describing, for the same reason
// ThFormMiner.cpp and CmfdOuterFormMiner.cpp were.
//
// WHY MINING AND NOT A SECOND BAKED CONSTANT.  WP23 shipped
// `kStreamFormsDefault = 0` with the comment "a GUESS about what gcc did and
// not a measurement", and that was honest but not free: it is reason (a) of the
// arm's N1 grade, and it is right until somebody upgrades gcc, changes -march
// or builds on a third machine, with nothing telling them.  What the mask
// actually asserts is "the shipped bodies and the verbatim CPU quotation agree
// bit for bit ON THIS BINARY", and that is a question this binary can ANSWER at
// startup, in milliseconds, from the same fixture the probe uses.  So it does,
// once, and prints what it found.

#include "FlatXsStreamFormMask.h"

#include "FlatXsStreamFormMine.h"
#include "GpuFormMask.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace rasbery::flatxs_stream {

namespace {

/// The fixture reaches every branch of the three quoted lambdas -- the empty
/// key list, the single-key list, the equal-burnup guard, the out-of-registry
/// isotope, both burnup clamps and the ordinary interior bracket (see
/// FlatXsStreamReference.cpp) -- and the whole four-seed descent is tens of
/// milliseconds.  A mask mined on operands that never reach a branch would be
/// mined on a fraction of the function.
///
/// The SHAPE lives in fsref::buildProductionFixture so the gate mines the same
/// operands this binary does.
const fsref::Fixture& fixture() {
    static const fsref::Fixture f = fsref::buildProductionFixture();
    return f;
}

struct Resolved {
    unsigned    mask   = kStreamFormsDefault;
    unsigned    mined  = kStreamFormsDefault;
    bool        sound  = false;
    const char* source = "build_default";
};

/// The resolution, done ONCE.  `rasbery::gpu::resolveCalibratedFormMask` owns
/// the precedence and the receipt line; this wrapper only has to recover WHICH
/// branch it took, because the receipt in FlatXsStreamReceipt.h reports the
/// source beside the value and a second guess at it would be a second answer.
const Resolved& resolved() {
    static const Resolved r = [] {
        Resolved out;
        out.mined = mineStreamFormsOnThisHost(out.sound);
        out.mask  = static_cast<unsigned>(rasbery::gpu::resolveCalibratedFormMask(
                        "RASBERY_FLATXS_STREAM_FORMS", kStreamFormsDefault, out.mined,
                        out.sound, "FLATXS_STREAM_FORMS")) &
                    FS_ALL;

        const char* raw = std::getenv("RASBERY_FLATXS_STREAM_FORMS");
        unsigned long long parsed = 0;
        if (raw != nullptr && rasbery::gpu::parseFormMask(raw, parsed))
            out.source = "env";
        else if (out.sound)
            out.source = (out.mined == kStreamFormsDefault) ? "mined_matches_default"
                                                            : "mined";
        else
            out.source = "build_default_mining_failed";
        return out;
    }();
    return r;
}

} // namespace

unsigned mineStreamFormsOnThisHost(bool& sound) {
    return fssmine::mineStable(fixture(), sound);
}

unsigned    streamFormMask()    { return resolved().mask; }
bool        streamFormsSound()  { return resolved().sound; }
const char* streamFormsSource() { return resolved().source; }

unsigned streamLibmMode() {
    // ABSENT MEANS `fast`, and an unrecognised value is REFUSED LOUDLY rather
    // than silently taken for one of the two: a typo here would otherwise look
    // like a passing run against the wrong rounding contract, which is the
    // failure GpuFormMask.h's parser exists to prevent for the mask.
    static const unsigned mode = [] {
        const char* raw = std::getenv("RASBERY_GPU_FLATXS_STREAM_LIBM");
        if (raw == nullptr) return kStreamLibmDefault;
        const std::string v(raw);
        if (v == "exact") return FS_LIBM_EXACT;
        if (v == "fast") return FS_LIBM_FAST;
        std::cerr << "[RASBERY][WARN][FORMS] RASBERY_GPU_FLATXS_STREAM_LIBM=\"" << v
                  << "\" is neither \"exact\" nor \"fast\"; using \"fast\". "
                     "A malformed override must not silently pass for a valid one.\n";
        return kStreamLibmDefault;
    }();
    return mode;
}

const char* streamLibmName() {
    return streamLibmMode() == FS_LIBM_EXACT ? "exact" : "fast";
}

} // namespace rasbery::flatxs_stream
