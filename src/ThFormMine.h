#pragma once

// Scoring and mining for the T/H contraction mask -- WP22.
//
// DELIBERATELY NOT IN ThReference.h.  This header pulls in the SHIPPED bodies
// (ThKernel.h); the reference translation unit must never see them, because
// with both in one TU gcc common-subexpressions across them and changes the
// REFERENCE's contraction -- which is the whole reason the reference gets a TU
// of its own.  Only ThFormMiner.cpp and the test driver include this file.
//
// The mask records which multiply-adds THE HOST COMPILER fused.  That is a
// property of the build machine, not of the physics, so this is a MEASUREMENT
// the binary makes of itself rather than a constant somebody wrote down.
// CmfdOuterFormMine.h is the same design and CmfdOuterFormMiner.cpp records
// what it cost to learn that lesson.

#include "ThKernel.h"

#include "ThReference.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace thmine {

namespace th = rasbery::th;

inline std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

/// Wrap the reference fixture's arrays in the shipped bodies' view.  Output
/// pointers are the caller's, so scoring never writes the fixture.
inline th::ThView viewOf(const thref::Fixture& f, std::vector<double>& node_power,
                         std::vector<double>& tmod, std::vector<double>& dmod,
                         std::vector<double>& tful) {
    th::ThView v{};
    v.nxy  = f.nxy;
    v.nz   = f.nz;
    v.nxyz = f.nxyz;
    v.ng   = f.ng;
    v.kbc  = f.kbc;
    v.kec  = f.kec;

    v.pressure             = f.pressure;
    v.inlet_h              = f.inlet_h;
    v.h_table_max          = f.h_table_max;
    v.fuel_temp_rise_scale = f.fuel_temp_rise_scale;
    v.th_relaxation        = f.th_relaxation;

    v.xskf    = f.xskf.data();
    v.phif    = f.phif.data();
    v.vol     = f.vol.data();
    v.hmesh_x = f.hmesh_x.data();
    v.hmesh_y = f.hmesh_y.data();
    v.hz      = f.hz.data();
    v.burn    = f.burn.data();

    v.mod_t   = {f.mod_t.x.data(), f.mod_t.y.data(), f.mod_t.v.data(), f.mod_t.nx, f.mod_t.ny};
    v.mod_rho = {f.mod_rho.x.data(), f.mod_rho.y.data(), f.mod_rho.v.data(), f.mod_rho.nx,
                 f.mod_rho.ny};
    v.tf      = {f.tf.x.data(), f.tf.y.data(), f.tf.v.data(), f.tf.nx, f.tf.ny};

    v.node_power = node_power.data();
    v.tmod       = tmod.data();
    v.dmod       = dmod.data();
    v.tful       = tful.data();
    return v;
}

/// Output words on which the shipped bodies under `mask` disagree with the
/// separately compiled reference.
///
/// EVERY SITE IS SCORED THROUGH THE BODY THAT SHIPS, not through a re-spelling
/// of it: refSolveTH and thChannelSweep are driven over the same operands and
/// their tmod/dmod/tful outputs are compared bit for bit, so a site the author
/// forgot to give a bit shows up as an unremovable residual rather than as a
/// silent agreement.
///
/// THE TWO SIDES ARE DRIVEN AT DIFFERENT GRANULARITY ON PURPOSE.  The reference
/// is called ONCE for the whole core because XSSet::SolveTH is one function with
/// the channel loop inside it, and that loop is part of the context gcc
/// contracts in (ThReference.h says what scoring a per-channel entry point cost:
/// bits 0-1 mined the wrong way and 238 measured 866 differing lines).  The
/// shipped body is called per channel because the device launches it per lane,
/// and the ceiling triple is folded here exactly as CudaThBackend.cu's
/// kernelFolds folds it -- ascending channel order, first wins -- so the
/// reduction the run publishes is the reduction the mining scored.
inline long scoreMask(const thref::Fixture& f, unsigned long long mask) {
    const std::size_t n = static_cast<std::size_t>(f.nxyz);

    std::vector<double> ref_power(n), got_power(n);
    thref::refNodePower(f, ref_power.data());

    std::vector<double> ref_tmod = f.tmod_old, ref_dmod = f.dmod_old, ref_tful = f.tful_old;
    std::vector<double> got_tmod = f.tmod_old, got_dmod = f.dmod_old, got_tful = f.tful_old;

    const th::ThView v = viewOf(f, got_power, got_tmod, got_dmod, got_tful);

    long bad = 0;
    for (int lk = 0; lk < f.nxyz; ++lk) {
        got_power[static_cast<std::size_t>(lk)] = th::thNodePower(v, lk, mask);
        if (bits(got_power[static_cast<std::size_t>(lk)]) !=
            bits(ref_power[static_cast<std::size_t>(lk)]))
            ++bad;
    }

    const thref::Overflow ro = thref::refSolveTH(f, ref_power.data(), ref_tmod.data(),
                                                 ref_dmod.data(), ref_tful.data());

    int    go_count = 0;
    double go_worst = 0.0;
    int    go_node  = -1;
    for (int l = 0; l < f.nxy; ++l) {
        const th::ThChannelOverflow go =
            th::thChannelSweep(v, l, f.norm, f.flow_per_channel, mask);
        go_count += go.count;
        if (go.count > 0 && go.worst > go_worst) {
            go_worst = go.worst;
            go_node  = go.node;
        }
    }
    if (ro.count != go_count || bits(ro.worst) != bits(go_worst) || ro.node != go_node) ++bad;

    for (std::size_t i = 0; i < n; ++i) {
        if (bits(got_tmod[i]) != bits(ref_tmod[i])) ++bad;
        if (bits(got_dmod[i]) != bits(ref_dmod[i])) ++bad;
        if (bits(got_tful[i]) != bits(ref_tful[i])) ++bad;
    }

    const double w = f.th_relaxation;
    for (std::size_t i = 0; i < n; ++i) {
        const double r = thref::refRelax(f.tful_old[i], got_tful[i], w);
        const double g = th::thRelaxNode(f.tful_old[i], got_tful[i], w, mask);
        if (bits(r) != bits(g)) ++bad;
    }
    return bad;
}

/// Coordinate descent from `seed`.  The seed is a parameter so the caller can
/// establish that the answer is a property of the HOST rather than of where the
/// search happened to start.
inline unsigned long long mineForms(const thref::Fixture& f, unsigned long long seed,
                                    bool verbose) {
    struct Site {
        int bit;
        int states;
    };
    std::vector<Site> sites;
    for (int b = 0; b < th::TH_ONE_BIT_END; ++b) sites.push_back({b, 2});
    sites.push_back({th::TH_RELAX, 3});
    for (int b = th::TH_TFUEL_LINEAR; b < th::TH_BIT_COUNT; ++b) sites.push_back({b, 2});

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
            std::printf("  th mine pass %d: %ld mismatching words, mask 0x%llXull\n", pass,
                        best_score, static_cast<unsigned long long>(best));
        if (best_score == before) break;
    }
    return best;
}

/// Mine from four different seeds and report whether the DERIVATION is sound.
///
/// "Sound" is every descent reaching ZERO mismatches -- NOT every descent
/// producing the same bit pattern.  Those are different properties, and the
/// second is too strong: TH_HAVG is a genuine DON'T-CARE (the halving is exact
/// in binary floating point), so different seeds legitimately settle on
/// different bits there and both masks are equally correct.  Demanding pattern
/// equality would fail on a mask that is provably right.
/// WHICH SITES THIS FIXTURE CANNOT PIN ON THIS HOST -- the check that was
/// missing, and whose absence cost WP22 a wrong mask with a clean receipt.
///
/// A residual of zero says "the mask I found reproduces the reference".  It does
/// NOT say "the reference could tell the alternatives apart".  When a site's
/// other forms ALSO score zero, the fixture reaches no operand that
/// distinguishes them: the coordinate descent then settles wherever its seed
/// started, `mineStable` reports sound, and the bit that reaches the device is an
/// accident.  That is precisely what shipped -- see TH_EXPECTED_DONT_CARE in
/// ThKernel.h for the arithmetic of how it happened.
///
/// Returns the set of sites whose every alternative form reproduces the
/// reference.  TH_RELAX is scored as ONE site with THREE states, because that is
/// what it is; a bit-at-a-time census would call it pinned on the strength of one
/// surviving alternative.
inline unsigned long long dontCareMask(const thref::Fixture& f, unsigned long long mined) {
    unsigned long long dc = 0ull;
    auto oneBit = [&](int b) {
        if (scoreMask(f, mined ^ (1ull << b)) == 0) dc |= 1ull << b;
    };
    for (int b = 0; b < th::TH_ONE_BIT_END; ++b) oneBit(b);
    for (int b = th::TH_TFUEL_LINEAR; b < th::TH_BIT_COUNT; ++b) oneBit(b);

    const unsigned long long cur  = (mined >> th::TH_RELAX) & 3ull;
    const unsigned long long base = mined & ~(3ull << th::TH_RELAX);
    for (unsigned long long s = 0; s < 3ull; ++s) {
        if (s == cur) continue;
        if (scoreMask(f, base | (s << th::TH_RELAX)) == 0) {
            dc |= 3ull << th::TH_RELAX;
            break;
        }
    }
    return dc;
}

inline unsigned long long mineStable(const thref::Fixture& f, bool& sound) {
    const unsigned long long seeds[4] = {0ull, th::TH_ALL_FORMS, 0x1ull,
                                         th::TH_FORMS_DEFAULT};
    unsigned long long       mined    = 0ull;
    sound                             = true;
    for (int i = 0; i < 4; ++i) {
        const unsigned long long m = mineForms(f, seeds[i], false);
        if (scoreMask(f, m) != 0) sound = false;
        if (i == 0) mined = m;
    }
    return mined;
}

} // namespace thmine
