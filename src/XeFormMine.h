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

/// Driver.h's XE_ANDERSON_MIN_GRAM, and the ONE place this file spells it.
/// Not #included from Driver.h because that header pulls in the whole solver;
/// tools/test_xe_txn_contract.py compares the two literals and fails when they
/// part company, which is the same guard XE_EQUILIBRIUM_TOLERANCE gets.
constexpr double XE_MIN_GRAM = 1.0e-8;

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

/// Output words on which the SHIPPED (pre-WP7-C) bodies under `mask` disagree
/// with the separately compiled reference: the fixed-partition inner product
/// and the candidate loop, i.e. the sites at bits 0..4.
///
/// THIS IS THE WHOLE OF WHAT `scoreMask` WAS BEFORE WP7-C, and it is a separate
/// function now for one reason: bits 0..4 are the mask the PRODUCTION device Xe
/// arm runs under (XeFormMiner.cpp -> kXeDotStage1 / kXeCandidate), and nothing
/// added later is allowed to change either the value they descend to or the
/// verdict on whether that descent was sound.  WP7-C added four more sites into
/// the one `bad` total, so a host on which the new sites could not reach zero
/// would report the WHOLE mask unsound and fall back to XE_FORMS_DEFAULT --
/// silently swapping the contraction contract of an arm nobody had touched.
inline long scoreShippedMask(const xeref::Fixture& f, unsigned long long mask) {
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

/// Output words on which the WP7-C normal-equations body under `mask` disagrees
/// with its quotation: the sites at bits 5..12, and NOTHING ELSE.  Scored and
/// descended separately from the shipped sites above so the two cannot move
/// each other -- see scoreShippedMask.  Returns 0 when the fixture carries no
/// algebra cases, which is what a caller that never asked for them gets.
inline long scoreAlgebraMask(const xeref::Fixture& f, unsigned long long mask) {
    long bad = 0;
    // WP7-C.  The normal equations, both window widths.  The one-column branch
    // has no site of its own, but it is scored anyway for the reason the
    // candidate loop's two widths are: it is what the fallback runs, and a mask
    // that broke it would otherwise mine clean.
    //
    // THE FLOOR IS THE PRODUCTION ONE.  1e-8 is Driver.h's
    // XE_ANDERSON_MIN_GRAM, and mining against a different floor would score a
    // conditioning test the run never makes.
    for (int ncol = 1; ncol <= xe::XE_DEPTH; ++ncol) {
        for (int ci = 0; ci < f.alg_cases; ++ci) {
            double       rg[2] = {0.0, 0.0}, rproj = 0.0, rdet = 0.0;
            const bool   rsolved =
                xeref::refAlgebra(f, ci, ncol, XE_MIN_GRAM, rg, &rproj, &rdet);
            double       g0 = 0.0, g1 = 0.0, proj = 0.0;
            const double* dots = f.alg.data() + static_cast<std::size_t>(6 * ci);
            const bool   solved =
                xe::xeAndersonFit(dots, ncol, XE_MIN_GRAM, mask, &g0, &g1, &proj);
            if (solved != rsolved) {
                ++bad;
                continue;
            }
            if (!solved) continue;
            if (bits(g0) != bits(rg[0])) ++bad;
            if (bits(g1) != bits(rg[1])) ++bad;
            if (bits(proj) != bits(rproj)) ++bad;
        }
    }
    return bad;
}

/// The two together, for a reader (test/xe_form_probe.cpp) that wants one
/// number for the whole mask.  Not used by the descent or by the soundness
/// verdict, both of which are per-channel on purpose.
inline long scoreMask(const xeref::Fixture& f, unsigned long long mask) {
    return scoreShippedMask(f, mask) + scoreAlgebraMask(f, mask);
}


struct Site {
    int bit;
    int states;
};

/// The four sites that existed before WP7-C, in the order they have always been
/// visited.  THE PRODUCTION MASK IS THESE FOUR: the device Xe dot and candidate
/// kernels read bits 0..4 and nothing above them.
inline constexpr Site kShippedSites[4] = {{xe::XE_DOT_FIRST_BIT, 3},
                                          {xe::XE_DOT_THIRD_BIT, 2},
                                          {xe::XE_CAND1_BIT, 2},
                                          {xe::XE_CAND2_BIT, 2}};

/// WP7-C's four, which only the RASBERY_GPU_XE_TXN arm evaluates.
inline constexpr Site kAlgebraSites[4] = {{xe::XE_TXN_DET_BIT, 3},
                                          {xe::XE_TXN_G0_BIT, 3},
                                          {xe::XE_TXN_G1_BIT, 3},
                                          {xe::XE_TXN_PROJ_BIT, 3}};

/// The pass budget the shipped descent has always had.  A CHANNEL CONSTANT, not
/// a shared one: raising it for a new channel would be a change to the old
/// channel's answer on any host where the descent had not converged.
constexpr int SHIPPED_PASSES = 6;
constexpr int ALGEBRA_PASSES = 6;

/// One coordinate descent: `sites` scored by `score`, from `seed`, keeping every
/// bit outside `sites` exactly as the seed left it.
///
/// `score` MUST depend only on the bits `sites` names.  That is what makes the
/// two channels independent rather than merely separate, and it is the whole
/// point of the split: a channel added tomorrow cannot move the mask an arm
/// that shipped yesterday runs under.
template <int N, typename Score>
inline unsigned long long descend(const xeref::Fixture& f, unsigned long long seed,
                                  const Site (&sites)[N], Score score, int max_passes,
                                  const char* tag, bool verbose) {
    unsigned long long best       = seed;
    long               best_score = score(f, best);
    for (int pass = 0; pass < max_passes && best_score > 0; ++pass) {
        const long before = best_score;
        for (const Site& s : sites)
            for (int state = 0; state < s.states; ++state) {
                const unsigned long long field = (s.states == 2) ? 1ull : 3ull;
                const unsigned long long mask =
                    (best & ~(field << s.bit)) |
                    (static_cast<unsigned long long>(state) << s.bit);
                if (mask == best) continue;
                const long candidate = score(f, mask);
                if (candidate < best_score) {
                    best_score = candidate;
                    best       = mask;
                }
            }
        if (verbose)
            std::printf("  mine %s pass %d: %ld mismatching words, mask 0x%llXull\n",
                        tag, pass, best_score, best);
        if (best_score == before) break;
    }
    return best;
}

/// Coordinate descent from `seed`, ONE CHANNEL AT A TIME.  The seed is a
/// parameter so the caller can establish that the answer is a property of the
/// HOST rather than of where the search happened to start.
inline unsigned long long mineForms(const xeref::Fixture& f, unsigned long long seed,
                                    bool verbose) {
    const unsigned long long shipped =
        descend(f, seed, kShippedSites, scoreShippedMask, SHIPPED_PASSES, "shipped",
                verbose);
    return descend(f, shipped, kAlgebraSites, scoreAlgebraMask, ALGEBRA_PASSES,
                   "algebra", verbose);
}

/// Mine from four different seeds and report whether the DERIVATION is sound --
/// PER CHANNEL.
///
/// "Sound" is every descent reaching ZERO mismatches -- NOT every descent
/// producing the same bit pattern.  Those are different properties and the
/// second is too strong: a site whose states are arithmetically identical on
/// every operand of this fixture is a genuine DON'T-CARE, so different seeds
/// legitimately settle on different bits there and both masks are equally
/// correct.  What under-determination of a site that DOES matter looks like is a
/// seed that cannot reach zero, and that is what this reports.  Same rule, same
/// words, as cmfdmine::mineStable.
///
/// TWO VERDICTS, NOT ONE, and that is the fix for the WP7-C regression.
/// `sound` is the verdict on the sites the PRODUCTION device Xe arm runs under,
/// and it is computed from scoreShippedMask alone -- exactly the predicate this
/// function used before WP7-C existed.  `algebra_sound` is the verdict on the
/// RASBERY_GPU_XE_TXN sites.  A host that cannot mine the WP7-C sites now fails
/// the WP7-C gate instead of demoting a mask nobody asked it to touch.
inline unsigned long long mineStable(const xeref::Fixture& f, bool& sound,
                                     bool& algebra_sound) {
    const unsigned long long seeds[4] = {
        0ull, ((1ull << xe::XE_BIT_COUNT) - 1ull), 0x1ull, xe::XE_FORMS_DEFAULT};
    unsigned long long mined = 0ull;
    sound                    = true;
    algebra_sound            = true;
    for (int i = 0; i < 4; ++i) {
        const unsigned long long m = mineForms(f, seeds[i], false);
        if (scoreShippedMask(f, m) != 0) sound = false;
        if (scoreAlgebraMask(f, m) != 0) algebra_sound = false;
        if (i == 0) mined = m;
    }
    return mined;
}

/// The fixture the mining runs on: the WP7-A one, plus the WP7-C algebra cases
/// its own translation unit builds.  ONE ENTRY POINT so no caller can score the
/// algebra channel against an empty `alg` and mine four don't-cares.
inline xeref::Fixture buildMiningFixture(int n) {
    xeref::Fixture f = xeref::buildFixture(n);
    xeref::buildAlgebraFixture(f);
    return f;
}

} // namespace xemine
