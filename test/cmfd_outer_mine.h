#pragma once

// Scoring and mining for the CMFD outer contraction mask.
//
// DELIBERATELY NOT IN cmfd_outer_reference.h.  This header pulls in the SHIPPED
// bodies (CmfdOuterKernel.h); the reference translation unit must never see
// them, because with both in one TU gcc common-subexpressions across them and
// changes the REFERENCE's contraction -- which is the whole reason the reference
// gets a TU of its own.  Only the test drivers include this file.
//
// WHY MINING IS PART OF THE GATE AND NOT A ONE-OFF.  The mask records which
// multiply-adds the HOST COMPILER fused.  That is a property of the build
// machine: measured 0x6 on the authoring box (WSL2, g++ 13.3) and 0x7 on 238
// (Xeon Gold 5317, where CO_PSI_ACC is fused).  A gate that asserted one literal
// would fail on the other host for a reason that has nothing to do with the code
// under test.  So the gates derive the host's mask, check the DERIVATION is
// stable, and score against that.

#include "../src/CmfdOuterKernel.h"

#include "cmfd_outer_reference.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cmfdmine {

namespace co = rasbery::cmfd;

inline std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

inline co::CmfdGeometryView geomOf(const cmfdref::Mesh& m) {
    co::CmfdGeometryView g{};
    g.surface_node    = m.surface_node;
    g.surface_dir     = m.surface_dir;
    g.node_hmesh      = m.node_hmesh;
    g.node_volume     = m.node_volume;
    g.boundary_albedo = m.boundary_albedo;
    g.nxyz            = m.nxyz;
    g.nsurf           = m.nsurf;
    g.ng              = m.ng;
    return g;
}

inline co::CmfdOuterView viewOf(const cmfdref::Mesh& m, std::vector<double>& jnet,
                                std::vector<double>& dtil, std::vector<double>& dhat,
                                std::vector<double>& psi) {
    co::CmfdOuterView v{};
    v.xsdf = m.xsdf;
    v.xsnf = m.xsnf;
    v.flux = m.flux;
    v.jnet = jnet.data();
    v.dtil = dtil.data();
    v.dhat = dhat.data();
    v.psi  = psi.data();
    return v;
}

/// Output words on which the shipped bodies under `mask` disagree with the
/// separately compiled reference.
inline long scoreMask(const cmfdref::Fixture& f, unsigned long long mask,
                      bool clamp_enabled) {
    const cmfdref::Mesh m = f.mesh();

    std::vector<double> ref_dtil(f.dtil.size()), ref_jnet(f.jnet.size()),
        ref_dhat(f.dhat.size()), ref_psi(static_cast<size_t>(f.nxyz));
    cmfdref::DhatCounters ref_counters;
    cmfdref::refUpdDtil(m, ref_dtil.data());
    cmfdref::refUpdPsi(m, ref_psi.data());
    cmfdref::refUpdJnet(m, ref_jnet.data());
    cmfdref::refUpdDhat(m, clamp_enabled, ref_dhat.data(), &ref_counters);

    std::vector<double> got_dtil = f.dtil, got_jnet = f.jnet, got_dhat = f.dhat;
    std::vector<double> got_psi(static_cast<size_t>(f.nxyz), 0.0);
    const co::CmfdGeometryView g = geomOf(m);
    const co::CmfdOuterView    v = viewOf(m, got_jnet, got_dtil, got_dhat, got_psi);

    long bad = 0;
    for (int ls = 0; ls < m.nsurf; ++ls)
        for (int ig = 0; ig < m.ng; ++ig) {
            const size_t k = static_cast<size_t>(ls) * m.ng + ig;
            if (bits(co::cmfdUpdDtilSurface(g, v, ls, ig)) != bits(ref_dtil[k])) ++bad;
            if (bits(co::cmfdUpdJnetSurface(g, v, ls, ig, mask)) != bits(ref_jnet[k])) ++bad;
            const co::CmfdDhatContribution c =
                co::cmfdUpdDhatSurface(g, v, ls, ig, clamp_enabled, mask);
            if (bits(c.dhat) != bits(ref_dhat[k])) ++bad;
        }
    for (int l = 0; l < m.nxyz; ++l)
        if (bits(co::cmfdUpdPsiNode(g, v, l, mask)) != bits(ref_psi[static_cast<size_t>(l)]))
            ++bad;
    return bad;
}

/// Coordinate descent from `seed`.  The seed is a parameter so the caller can
/// establish that the answer is a property of the HOST rather than of where the
/// search happened to start.
inline unsigned long long mineForms(const cmfdref::Fixture& f, unsigned long long seed,
                                    bool verbose) {
    struct Site {
        int bit;
        int states;
    };
    std::vector<Site> sites;
    for (int b = 0; b < co::CO_ONE_BIT_COUNT; ++b) sites.push_back({b, 2});
    for (int b = co::CO_ONE_BIT_COUNT; b < co::CO_BIT_COUNT; b += 2) sites.push_back({b, 3});

    unsigned long long best       = seed;
    long               best_score = scoreMask(f, best, false);
    for (int pass = 0; pass < 6 && best_score > 0; ++pass) {
        const long before = best_score;
        for (const Site& s : sites)
            for (int state = 0; state < s.states; ++state) {
                const unsigned long long mask =
                    (best & ~(static_cast<unsigned long long>(s.states == 2 ? 1 : 3) << s.bit)) |
                    (static_cast<unsigned long long>(state) << s.bit);
                if (mask == best) continue;
                const long score = scoreMask(f, mask, false);
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
/// producing the same bit pattern.  Those are different properties, and the
/// second one is too strong: a site whose two states are arithmetically
/// identical on every operand (m253's `2*kp2 + 5`, where the product is exact)
/// is a genuine DON'T-CARE, so different seeds legitimately settle on different
/// bits there and both masks are equally correct.  Demanding pattern equality
/// would fail on a mask that is provably right.
///
/// What under-determination of a site that DOES matter looks like is a seed that
/// cannot reach zero, and that is what this reports.  The per-site decisiveness
/// check in the probe is the complementary half: it names the sites this fixture
/// actually pins.
inline unsigned long long mineStable(const cmfdref::Fixture& f, bool& sound) {
    const unsigned long long seeds[4] = {
        0ull, ((1ull << co::CO_BIT_COUNT) - 1ull), 0x1ull, 0x6ull};
    unsigned long long mined = 0ull;
    sound                    = true;
    for (int i = 0; i < 4; ++i) {
        const unsigned long long m = mineForms(f, seeds[i], false);
        if (scoreMask(f, m, false) != 0) sound = false;
        if (i == 0) mined = m;
    }
    return mined;
}

} // namespace cmfdmine
