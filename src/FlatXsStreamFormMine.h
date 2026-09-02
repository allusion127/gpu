#pragma once

// Scoring and mining for the WP23 branch-stream contraction mask -- WP23.1.
//
// DELIBERATELY NOT IN FlatXsStreamReference.h.  This header pulls in the
// SHIPPED body (FlatXsStreamKernel.h); the reference translation unit must
// never see it, because with both in one TU gcc common-subexpressions across
// them and changes the REFERENCE's contraction -- which is the whole reason the
// reference gets a TU of its own.  Only FlatXsStreamFormMiner.cpp and the test
// driver include this file.  ThFormMine.h is the same design and
// CmfdOuterFormMiner.cpp records what it cost to learn the lesson.
//
// The mask records which multiply-adds THE HOST COMPILER FUSED.  That is a
// property of the build machine, not of the physics, so this is a MEASUREMENT
// the binary makes of itself rather than a constant somebody wrote down.

#include "FlatXsStreamKernel.h"

#include "FlatXsStreamReference.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace fssmine {

namespace fss = rasbery::flatxs_stream;

inline std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

/// Wrap the fixture's arrays in the shipped body's view.  THE SAME BYTES the
/// quotation reads -- the view carries pointers, it does not copy -- so the two
/// sides can differ only in how they round.
inline fss::StreamLibView viewOf(const fsref::Fixture& f) {
    fss::StreamLibView lib{};
    lib.refr0_key_off    = f.refr0_key_off.data();
    lib.refr0_key_cnt    = f.refr0_key_cnt.data();
    lib.refr0_present    = f.refr0_present.data();
    lib.refr0_keys       = f.refr0_keys.data();
    lib.refr0_base       = f.refr0_base.data();
    lib.refr_burn_stride = f.refr_burn_stride.data();
    lib.lib_iden         = f.lib_iden.data();
    lib.lib_burn         = f.lib_burn.data();
    lib.lib_ref_branch_x = f.lib_ref_branch_x.data();
    lib.nmodel           = f.nmodel;
    lib.niso             = f.niso;
    return lib;
}

/// Drive the SHIPPED bodies over the fixture under `mask`, in the same order and
/// with the same operands the quotation used.
///
/// EVERY SITE IS SCORED THROUGH THE BODY THAT SHIPS, not through a re-spelling
/// of it: fsReferenceDensity / fsReferenceDensity0 / fsReferenceCondition are
/// the very functions kFlatXsStreamBuild calls, so a site the author forgot to
/// give a field shows up as an unremovable residual rather than as a silent
/// agreement.
inline void shippedBuild(const fsref::Fixture& f, unsigned mask, double* dens,
                         double* dens0, double* cond) {
    const fss::StreamLibView lib = viewOf(f);
    fss::StreamForms         pol;
    pol.mask = mask;
    pol.libm = fss::kStreamLibmDefault; // no libm on any of the three sites

    for (int l = 0; l < f.nnode; ++l) {
        const int m    = f.node_model[static_cast<std::size_t>(l)];
        const int burn = f.node_burn[static_cast<std::size_t>(l)];
        const int* keys = f.refr0_keys.data() + f.refr0_key_off[static_cast<std::size_t>(m)];
        const int  nkey = f.refr0_key_cnt[static_cast<std::size_t>(m)];
        const int  keyIndex = fss::fsFindLoBurn(keys, nkey, burn);
        const int  keyBurnup = keyIndex >= 0 ? keys[keyIndex] : burn;

        for (int p = 0; p < f.nprobe; ++p) {
            const int iso =
                f.probe_iso[static_cast<std::size_t>(l) * f.nprobe + p];
            const std::size_t u =
                (static_cast<std::size_t>(l) * f.nprobe + p) * 2u;
            dens[u + 0]  = fss::fsReferenceDensity(lib, m, iso, burn, pol);
            dens[u + 1]  = fss::fsReferenceDensity(lib, m, iso, keyBurnup, pol);
            const int partner = static_cast<int>((static_cast<unsigned>(iso) + 7u) %
                                                 static_cast<unsigned>(f.niso + 4));
            dens0[u + 0] = fss::fsReferenceDensity0(lib, m, iso, burn, pol);
            dens0[u + 1] = fss::fsReferenceDensity0(lib, m, partner, burn, pol);
        }
        for (int axis = 0; axis < 3; ++axis)
            cond[static_cast<std::size_t>(l) * 3 + axis] =
                fss::fsReferenceCondition(lib, m, axis, burn, pol);
    }
}

/// The quotation's answer, evaluated ONCE.  Held in a value the caller owns
/// rather than in a function-local static: a cache keyed on nothing is right
/// until the day a second fixture is scored, and then it is silently wrong.
struct Reference {
    std::vector<double> dens, dens0, cond;
};

inline Reference referenceOf(const fsref::Fixture& f) {
    Reference r;
    r.dens.resize(f.densWords());
    r.dens0.resize(f.densWords());
    r.cond.resize(f.condWords());
    fsref::refBuildStream(f, r.dens.data(), r.dens0.data(), r.cond.data());
    return r;
}

/// Output words on which the shipped bodies under `mask` disagree with the
/// separately compiled quotation.
inline long scoreMask(const fsref::Fixture& f, const Reference& ref, unsigned mask) {
    std::vector<double> got_dens(f.densWords()), got_dens0(f.densWords()),
        got_cond(f.condWords());
    shippedBuild(f, mask, got_dens.data(), got_dens0.data(), got_cond.data());

    long bad = 0;
    for (std::size_t i = 0; i < got_dens.size(); ++i) {
        if (bits(got_dens[i]) != bits(ref.dens[i])) ++bad;
        if (bits(got_dens0[i]) != bits(ref.dens0[i])) ++bad;
    }
    for (std::size_t i = 0; i < got_cond.size(); ++i)
        if (bits(got_cond[i]) != bits(ref.cond[i])) ++bad;
    return bad;
}

/// Mismatching words attributable to ONE site, with the other two held at
/// `mask`.  Used by the probe to say whether a site is DECISIVE on this host --
/// without that, a mask of all zeros passes on a machine whose compiler fused
/// nothing and the gate would be asserting that the fixture is boring rather
/// than that the mask is right.
inline long scoreSiteState(const fsref::Fixture& f, const Reference& ref,
                           unsigned mask, unsigned bit, unsigned state) {
    const unsigned m = (mask & ~(3u << bit)) | (state << bit);
    return scoreMask(f, ref, m);
}

/// Coordinate descent from `seed`.  The seed is a parameter so the caller can
/// establish that the answer is a property of the HOST rather than of where the
/// search happened to start.
inline unsigned mineForms(const fsref::Fixture& f, const Reference& ref, unsigned seed,
                          bool verbose) {
    const unsigned sites[3] = {fss::FS_REFDENS, fss::FS_REFDENS0, fss::FS_REFCOND};

    unsigned best       = seed & fss::FS_ALL;
    long     best_score = scoreMask(f, ref, best);
    for (int pass = 0; pass < 6 && best_score > 0; ++pass) {
        const long before = best_score;
        for (unsigned bit : sites)
            for (unsigned state = 0; state < 3; ++state) {
                const unsigned mask = (best & ~(3u << bit)) | (state << bit);
                if (mask == best) continue;
                const long score = scoreMask(f, ref, mask);
                if (score < best_score) {
                    best_score = score;
                    best       = mask;
                }
            }
        if (verbose)
            std::printf("  flatxs-stream mine pass %d: %ld mismatching words, mask 0x%x\n",
                        pass, best_score, best);
        if (best_score == before) break;
    }
    return best;
}

/// Mine from four different seeds and report whether the DERIVATION is sound.
///
/// "Sound" is every descent reaching ZERO mismatches -- NOT every descent
/// producing the same bit pattern.  Those are different properties and the
/// second is too strong here for a reason this arm can prove rather than
/// suspect: FS_SITE_P1 and FS_SITE_P2 are the same fma with its first two
/// arguments swapped, IEEE-754 fma is exactly commutative in those, so a fused
/// site is a genuine DON'T-CARE between states 1 and 2 and different seeds
/// legitimately settle on different ones.  Demanding pattern equality would
/// fail on a mask that is provably right; ThFormMine.h's TH_HAVG is the same
/// situation and the same verdict.
inline unsigned mineStable(const fsref::Fixture& f, bool& sound) {
    // FOUR CORNERS OF THE SEARCH SPACE, not four arbitrary numbers: nothing
    // fused, all three FS_SITE_P1, all three FS_SITE_P2, and the shipped
    // per-build record.  A descent that reaches zero from every one of them is
    // measuring the host; a descent that only reaches zero from the answer is
    // measuring its own starting point.
    const unsigned seeds[4] = {0u, 0x15u, 0x2au, fss::kStreamFormsDefault};
    const Reference ref     = referenceOf(f);
    unsigned        mined   = 0u;
    sound                   = true;
    for (int i = 0; i < 4; ++i) {
        const unsigned m = mineForms(f, ref, seeds[i], false);
        if (scoreMask(f, ref, m) != 0) sound = false;
        if (i == 0) mined = m;
    }
    return mined;
}

} // namespace fssmine
