#pragma once

// Scoring and mining for the Xe Anderson contraction mask -- Rev.7.1 Task 13.
//
// DELIBERATELY NOT IN XeAndersonReference.h, for the reason that header states:
// this file pulls in the SHIPPED bodies (XeKernel.h), and the reference
// translation unit must never see them or gcc common-subexpressions across the
// two and changes what the reference measures.  Only XeFormMiner.cpp and the
// test drivers include this file.
//
// WHY MINING IS PART OF THE GATE AND NOT A ONE-OFF.  The mask records which
// multiply-adds THE HOST COMPILER fused in Driver.h's Anderson algebra.  That is
// a property of the build machine: CmfdOuterFormMiner.cpp records 0x6 on the
// WSL2 / g++ 13.3 box and 0x7 on the 238 Xeon for the CMFD outer, and there is
// no reason this mask behaves differently.  A gate that asserted one literal
// would fail on the other host for a reason that has nothing to do with the
// code under test.  So the gates derive the host's mask, check the DERIVATION
// is sound, and score against that.

#include "XeAndersonReference.h"
#include "XeKernel.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

namespace xemine {

namespace xe = rasbery::xe;

inline std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

/// How many contiguous slices the inner product is scored over.
///
/// The production dot returns ONE number, and one word is nearly no signal for
/// a coordinate descent -- several masks reach it by luck.  Sixteen slices of
/// the same fixture are sixteen independent words at no extra cost, and they
/// are also the shape the kernel actually runs in (one partition per thread),
/// so what is scored is what ships.
constexpr int SCORE_SLICES = 16;

inline xe::XeTripleConst leftOf(const xeref::Fixture& f) {
    xe::XeTripleConst t;
    t.i135   = f.a_i.data();
    t.xe135  = f.a_x.data();
    t.xe135m = f.a_m.data();
    return t;
}

inline xe::XeTripleConst rightOf(const xeref::Fixture& f) {
    xe::XeTripleConst t;
    t.i135   = f.b_i.data();
    t.xe135  = f.b_x.data();
    t.xe135m = f.b_m.data();
    return t;
}

/// Output words on which the shipped bodies under `mask` disagree with the
/// separately compiled reference.
inline long scoreMask(const xeref::Fixture& f, unsigned long long mask) {
    long bad = 0;

    const xe::XeTripleConst a = leftOf(f);
    const xe::XeTripleConst b = rightOf(f);
    for (int s = 0; s < SCORE_SLICES; ++s) {
        int i0 = 0, i1 = 0;
        xe::xeDotPartitionRange(f.n, SCORE_SLICES, s, &i0, &i1);
        if (bits(xe::xeDotChunk(a, b, i0, i1, mask)) != bits(xeref::refDot(f, i0, i1)))
            ++bad;
    }

    // The candidate loop, both window widths: a one-column step is the
    // secant fallback the host takes whenever the two-column Gram matrix is
    // ill-conditioned, and it is a different number of trips through the same
    // site, so both are scored.
    for (int ncol = 1; ncol <= xe::XE_DEPTH; ++ncol) {
        std::vector<double> ri, rx, rm;
        xeref::refCandidate(f, ncol, ri, rx, rm);

        std::vector<double> gi(static_cast<std::size_t>(f.n)),
            gx(static_cast<std::size_t>(f.n)), gm(static_cast<std::size_t>(f.n));
        xe::XeTriple cand;
        cand.i135   = gi.data();
        cand.xe135  = gx.data();
        cand.xe135m = gm.data();

        xe::XeTripleConst fv;
        fv.i135   = f.f_i.data();
        fv.xe135  = f.f_x.data();
        fv.xe135m = f.f_m.data();
        // The trust-region metric needs a base point; the fixture's F doubles as
        // one here.  Only the candidate VALUES are scored -- the metric itself
        // has no multiply-add and therefore no site.
        const xe::XeTripleConst xv = fv;

        for (int k = 0; k < f.n; ++k) {
            int    physics_bad = 0;
            double step        = 0.0;
            xe::xeCandidateOrdinal(fv, xv, f.d_i.data(), f.d_x.data(), f.d_m.data(),
                                   ncol, f.gamma, cand, k, f.n, mask, &physics_bad,
                                   &step);
            const auto u = static_cast<std::size_t>(k);
            if (bits(gi[u]) != bits(ri[u])) ++bad;
            if (bits(gx[u]) != bits(rx[u])) ++bad;
            if (bits(gm[u]) != bits(rm[u])) ++bad;
        }
    }
    return bad;
}

/// Coordinate descent from `seed`.  The seed is a parameter so the caller can
/// establish that the answer is a property of the HOST rather than of where the
/// search happened to start.
inline unsigned long long mineForms(const xeref::Fixture& f, unsigned long long seed,
                                    bool verbose) {
    struct Site {
        int bit;
        int states;
    };
    const Site sites[4] = {{xe::XE_DOT_FIRST_BIT, 3},
                           {xe::XE_DOT_THIRD_BIT, 2},
                           {xe::XE_CAND1_BIT, 2},
                           {xe::XE_CAND2_BIT, 2}};

    unsigned long long best       = seed;
    long               best_score = scoreMask(f, best);
    for (int pass = 0; pass < 6 && best_score > 0; ++pass) {
        const long before = best_score;
        for (const Site& s : sites)
            for (int state = 0; state < s.states; ++state) {
                const unsigned long long field = (s.states == 2) ? 1ull : 3ull;
                const unsigned long long mask =
                    (best & ~(field << s.bit)) |
                    (static_cast<unsigned long long>(state) << s.bit);
                if (mask == best) continue;
                const long score = scoreMask(f, mask);
                if (score < best_score) {
                    best_score = score;
                    best       = mask;
                }
            }
        if (verbose)
            std::printf("  mine pass %d: %ld mismatching words, mask 0x%llXull\n", pass,
                        best_score, best);
        if (best_score == before) break;
    }
    return best;
}

/// Mine from four different seeds and report whether the DERIVATION is sound.
///
/// "Sound" is every descent reaching ZERO mismatches -- NOT every descent
/// producing the same bit pattern.  Those are different properties and the
/// second is too strong: a site whose states are arithmetically identical on
/// every operand of this fixture is a genuine DON'T-CARE, so different seeds
/// legitimately settle on different bits there and both masks are equally
/// correct.  What under-determination of a site that DOES matter looks like is a
/// seed that cannot reach zero, and that is what this reports.  Same rule, same
/// words, as cmfdmine::mineStable.
inline unsigned long long mineStable(const xeref::Fixture& f, bool& sound) {
    const unsigned long long seeds[4] = {
        0ull, ((1ull << xe::XE_BIT_COUNT) - 1ull), 0x1ull, xe::XE_FORMS_DEFAULT};
    unsigned long long mined = 0ull;
    sound                    = true;
    for (int i = 0; i < 4; ++i) {
        const unsigned long long m = mineForms(f, seeds[i], false);
        if (scoreMask(f, m) != 0) sound = false;
        if (i == 0) mined = m;
    }
    return mined;
}

} // namespace xemine
