// The verbatim WP23 burnup-lerp quotation.  See FlatXsStreamReference.h for
// why this is its own translation unit and what may and may not appear in it.
//
// NOTHING IN THIS FILE MAY INCLUDE FlatXsStreamKernel.h.  That is the whole
// contract, and it is asserted by tools/test_flatxs_stream_forms_contract.py
// rather than left to a reviewer's memory.

#include "FlatXsStreamReference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fsref {

namespace {

/// XSSet.cpp's findLoBurn, quoted.  Integer work: no rounding to disagree
/// about, and it is here so the quotation reaches the same bracket the shipped
/// body reaches rather than a bracket this file chose.
int findLoBurn(const std::vector<int>& keys, int burn_key) {
    if (keys.empty()) return -1;
    auto it = std::lower_bound(keys.begin(), keys.end(), burn_key);
    if (it == keys.end()) return static_cast<int>(keys.size() - 1);
    if (it == keys.begin()) return 0;
    return static_cast<int>(std::distance(keys.begin(), std::prev(it)));
}

/// XSSet.cpp's findHiBurn, quoted.
int findHiBurn(const std::vector<int>& keys, int burn_key) {
    if (keys.empty()) return -1;
    auto it = std::lower_bound(keys.begin(), keys.end(), burn_key);
    if (it == keys.end()) return static_cast<int>(keys.size() - 1);
    return static_cast<int>(std::distance(keys.begin(), it));
}

/// A deterministic 64-bit LCG.  The same one ThReference.cpp uses, and for the
/// same reason: the fixture has to be the same field on every host and in every
/// process, or the mined mask would depend on which operands the run happened
/// to see.
struct Rng {
    std::uint64_t s;
    double        next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((s >> 11) & ((1ull << 53) - 1ull)) /
               static_cast<double>(1ull << 53);
    }
    int nextInt(int lo, int hi) {
        return lo + static_cast<int>(next() * static_cast<double>(hi - lo + 1)) %
                        (hi - lo + 1);
    }
};

/// Number densities as the library carries them: log-uniform over the decades a
/// real inventory occupies, which is what makes `hi - lo` in the lerp a
/// cancelling subtraction on some rows and not on others.
double density(Rng& rng) {
    const double e = -12.0 + 10.0 * rng.next();
    return std::pow(10.0, e);
}

} // namespace

Fixture buildFixture(int nmodel, int nnode, int nprobe) {
    Fixture f;
    f.nmodel = nmodel;
    f.nnode  = nnode;
    f.nprobe = nprobe;
    f.niso   = 26;

    Rng rng{0xF1A7C5ull};

    // The reference burnup ladder every model shares the SHAPE of: dense at low
    // burnup where the isotopics move, coarse past 40 GWd/THM.  Per model the
    // ladder is truncated differently so the fixture carries an empty list, a
    // one-key list and several interior lists.
    static const int kLadder[] = {0,     150,   500,   1000,  2000,  4000,  6000,
                                  9000,  12000, 16000, 20000, 25000, 30000, 36000,
                                  42000, 48000, 54000, 60000};
    const int        kLadderN  = static_cast<int>(sizeof(kLadder) / sizeof(kLadder[0]));

    f.refr0_key_off.assign(static_cast<std::size_t>(nmodel), 0);
    f.refr0_key_cnt.assign(static_cast<std::size_t>(nmodel), 0);
    f.refr0_present.assign(static_cast<std::size_t>(nmodel), 1);
    f.refr0_base.assign(static_cast<std::size_t>(nmodel), 0);
    f.refr_burn_stride.assign(static_cast<std::size_t>(nmodel), 0);

    long long row = 0;
    for (int m = 0; m < nmodel; ++m) {
        // model 0 gets the whole ladder; model 1 a single key (the
        // loIndex == hiIndex path); model 2 an EMPTY list with the ctype
        // present, which is the case that returns 0.0 from a present ctype and
        // is not the same fact as an absent one; the rest get interior slices.
        int n = kLadderN;
        if (m == 1)
            n = 1;
        else if (m == 2)
            n = 0;
        else if (m > 2)
            n = 4 + (m * 3) % (kLadderN - 4);

        f.refr0_key_off[static_cast<std::size_t>(m)] =
            static_cast<int>(f.refr0_keys.size());
        f.refr0_key_cnt[static_cast<std::size_t>(m)] = n;
        // A stride > 1 so the flat id arithmetic is exercised the way a real
        // library's ctype block exercises it.
        const long long stride = 1 + (m % 3);
        f.refr0_base[static_cast<std::size_t>(m)]       = row;
        f.refr_burn_stride[static_cast<std::size_t>(m)] = stride;
        for (int k = 0; k < n; ++k) f.refr0_keys.push_back(kLadder[k]);
        row += static_cast<long long>(n) * stride + 2; // +2 of slack, as a real block has
    }
    f.n_rows = row + 4;

    f.lib_burn.resize(static_cast<std::size_t>(f.n_rows));
    f.lib_iden.resize(static_cast<std::size_t>(f.n_rows) * static_cast<std::size_t>(f.niso));
    f.lib_ref_branch_x.resize(static_cast<std::size_t>(f.n_rows) * 3u);

    // lib_burn is the ladder in GWd/THM, which is the number the lerp's
    // numerator is differenced against.  Two adjacent rows are given the SAME
    // burnup on purpose: `lib_burn[hi] != lib_burn[lo]` is a guard in all three
    // quoted lambdas and a fixture that never fails it would not score it.
    for (long long r = 0; r < f.n_rows; ++r) f.lib_burn[static_cast<std::size_t>(r)] = 0.0;
    for (int m = 0; m < nmodel; ++m) {
        const long long base   = f.refr0_base[static_cast<std::size_t>(m)];
        const long long stride = f.refr_burn_stride[static_cast<std::size_t>(m)];
        const int       n      = f.refr0_key_cnt[static_cast<std::size_t>(m)];
        for (int k = 0; k < n; ++k) {
            const long long rr = base + static_cast<long long>(k) * stride;
            const int       kk = (m == 5 && k > 0) ? k - 1 : k; // the equal-burnup row
            f.lib_burn[static_cast<std::size_t>(rr)] =
                static_cast<double>(kLadder[kk]) / 1000.0;
        }
    }
    for (std::size_t i = 0; i < f.lib_iden.size(); ++i) f.lib_iden[i] = density(rng);
    for (long long r = 0; r < f.n_rows; ++r) {
        const std::size_t u = static_cast<std::size_t>(r) * 3u;
        f.lib_ref_branch_x[u + 0] = 2.0e-3 * rng.next();       // boron*wvfr*bppm*factor
        f.lib_ref_branch_x[u + 1] = 25.0 + 12.0 * rng.next();  // sqrt(tful)
        f.lib_ref_branch_x[u + 2] = 0.60 + 0.20 * rng.next();  // dmod
    }

    f.node_model.resize(static_cast<std::size_t>(nnode));
    f.node_burn.resize(static_cast<std::size_t>(nnode));
    f.probe_iso.resize(static_cast<std::size_t>(nnode) * static_cast<std::size_t>(nprobe));
    for (int i = 0; i < nnode; ++i) {
        f.node_model[static_cast<std::size_t>(i)] = i % nmodel;
        // Burnups ON a ladder key (the skipped-lerp path), between keys (the
        // ordinary path), below the first and above the last (both clamps).
        const int pick = i % 5;
        int       b    = 0;
        if (pick == 0)
            b = kLadder[i % kLadderN];
        else if (pick == 1)
            b = -500 + static_cast<int>(400.0 * rng.next());
        else if (pick == 2)
            b = 60000 + static_cast<int>(9000.0 * rng.next());
        else
            b = static_cast<int>(60000.0 * rng.next());
        f.node_burn[static_cast<std::size_t>(i)] = b;
        for (int p = 0; p < nprobe; ++p)
            // The last probe of every node is an OUT-OF-REGISTRY isotope, which
            // is the `isotope >= niso -> 0.0` early return.
            f.probe_iso[static_cast<std::size_t>(i) * nprobe + p] =
                (p == nprobe - 1) ? f.niso + 3 : rng.nextInt(0, f.niso - 1);
    }
    return f;
}

Fixture buildProductionFixture() {
    // 8 models x 2048 nodes x 8 isotope probes.  The node count is set by the
    // THINNEST site and not by the fattest: FS_REFCOND is pinned by the fewest
    // words of the three (a lerp whose increment is small against its
    // accumulator flips the sum's rounding on only a few percent of operands),
    // so the fixture is sized until that site is decisively wrong when it is
    // set wrong -- see the per-site counts the gate prints.  The whole
    // four-seed descent is still tens of milliseconds.
    return buildFixture(8, 2048, 8);
}

// ===========================================================================
// THE QUOTATION
// ===========================================================================

namespace {

/// XSSet::ResolveSpectralHistoryDeltas, reduced to the three lambdas and their
/// surrounding locals, with the ORIGINAL NAMES.  What is being measured is what
/// gcc does to the three `+=` statements below and to nothing else, so the
/// names, the order of the locals and the fact that all three lambdas are
/// captured `[&]` from one enclosing scope are all part of the quotation.
void refResolveSpectralHistoryDeltas(const Fixture& fx, int l, double* dens_out,
                                     double* dens0_out, double* cond_out) {
    const std::size_t modelIndex = static_cast<std::size_t>(
        fx.node_model[static_cast<std::size_t>(l)]);
    const int burn = fx.node_burn[static_cast<std::size_t>(l)];

    const std::vector<int>& allKeys = fx.refr0_keys;
    const int               keyOff  = fx.refr0_key_off[modelIndex];
    const int               keyCnt  = fx.refr0_key_cnt[modelIndex];
    std::vector<int>        referenceBurnups(
        allKeys.begin() + keyOff, allKeys.begin() + keyOff + keyCnt);

    const std::size_t referenceBase  = static_cast<std::size_t>(fx.refr0_base[modelIndex]);
    const std::size_t referenceBase0 = referenceBase; // the ctype-0 collapse
    const std::size_t burnStride =
        static_cast<std::size_t>(fx.refr_burn_stride[modelIndex]);
    const std::size_t niso = static_cast<std::size_t>(fx.niso);

    const std::vector<double>& lib_iden         = fx.lib_iden;
    const std::vector<double>& lib_burn         = fx.lib_burn;
    const std::vector<double>& lib_ref_branch_x = fx.lib_ref_branch_x;

    auto referenceDensity0 = [&](std::size_t isotope, int burnup) {
        if (isotope >= niso)
            return 0.0;
        const std::vector<int>& burns = referenceBurnups;
        const int               lo    = findLoBurn(burns, burnup);
        const int               hi    = findHiBurn(burns, burnup);
        if (lo < 0 || hi < 0)
            return 0.0;
        const std::size_t lb = referenceBase0 + static_cast<std::size_t>(lo) * burnStride;
        const std::size_t hb = referenceBase0 + static_cast<std::size_t>(hi) * burnStride;
        double            v  = lib_iden[lb * niso + isotope];
        if (lb != hb && lib_burn[hb] != lib_burn[lb]) {
            const double f = (static_cast<double>(burnup) / 1000.0 - lib_burn[lb]) /
                             (lib_burn[hb] - lib_burn[lb]);
            v += f * (lib_iden[hb * niso + isotope] - v);
        }
        return v;
    };

    auto referenceDensity = [&](std::size_t isotope, int burnup) {
        if (isotope >= niso)
            return 0.0;

        const int loIndex = findLoBurn(referenceBurnups, burnup);
        const int hiIndex = findHiBurn(referenceBurnups, burnup);
        if (loIndex < 0 || hiIndex < 0)
            return 0.0;

        const std::size_t lo =
            referenceBase + static_cast<std::size_t>(loIndex) * burnStride;
        const std::size_t hi =
            referenceBase + static_cast<std::size_t>(hiIndex) * burnStride;

        double value = lib_iden[lo * niso + isotope];
        if (lo != hi && lib_burn[hi] != lib_burn[lo]) {
            const double fraction =
                (static_cast<double>(burnup) / 1000.0 - lib_burn[lo]) /
                (lib_burn[hi] - lib_burn[lo]);
            value += fraction * (lib_iden[hi * niso + isotope] - lib_iden[lo * niso + isotope]);
        }
        return value;
    };

    auto referenceCondition = [&](int axis, int burnup) {
        const int loIndex = findLoBurn(referenceBurnups, burnup);
        const int hiIndex = findHiBurn(referenceBurnups, burnup);
        if (loIndex < 0 || hiIndex < 0)
            return 0.0;
        const std::size_t lo =
            referenceBase + static_cast<std::size_t>(loIndex) * burnStride;
        const std::size_t hi =
            referenceBase + static_cast<std::size_t>(hiIndex) * burnStride;
        const std::size_t a = static_cast<std::size_t>(axis);
        double            value = lib_ref_branch_x[lo * 3 + a];
        if (lo != hi && lib_burn[hi] != lib_burn[lo]) {
            const double fraction =
                (static_cast<double>(burnup) / 1000.0 - lib_burn[lo]) /
                (lib_burn[hi] - lib_burn[lo]);
            value += fraction * (lib_ref_branch_x[hi * 3 + a] - value);
        }
        return value;
    };

    // The call pattern is the production one: referenceDensity is reached from
    // the coordinate chain AND twice more from keyCoordinate (at `burn` and at
    // the bracket key), referenceDensity0 twice from burnRatioCoordinate, and
    // referenceCondition once per branch axis.  A fixture that called each of
    // them once would present gcc with a different inlining problem.
    for (int p = 0; p < fx.nprobe; ++p) {
        const std::size_t isotope = static_cast<std::size_t>(
            fx.probe_iso[static_cast<std::size_t>(l) * fx.nprobe + p]);
        const double referenceNow   = referenceDensity(isotope, burn);
        const int    keyIndex       = findLoBurn(referenceBurnups, burn);
        const int    keyBurnup      = keyIndex >= 0
                                          ? referenceBurnups[static_cast<std::size_t>(keyIndex)]
                                          : burn;
        const double referenceAtKey = referenceDensity(isotope, keyBurnup);

        // BOTH values are written out, and that is not redundancy: the miner
        // must score the SITE, not a composite of it, or a difference in one
        // call could be cancelled by the next -- and a value that is computed
        // and never stored is a call gcc is entitled to delete, which would
        // change the very inlining problem this driver exists to reproduce.
        const std::size_t u = (static_cast<std::size_t>(l) * fx.nprobe + p) * 2u;
        dens_out[u + 0] = referenceNow;
        dens_out[u + 1] = referenceAtKey;

        const std::size_t partner = (isotope + 7u) % (niso + 4u);
        dens0_out[u + 0]          = referenceDensity0(isotope, burn);
        dens0_out[u + 1]          = referenceDensity0(partner, burn);
    }
    for (int axis = 0; axis < 3; ++axis)
        cond_out[static_cast<std::size_t>(l) * 3 + axis] = referenceCondition(axis, burn);
}

/// XSSet::ResolveNodeApplications' function boundary, so the lambdas above are
/// one inline level below the OpenMP body exactly as they are in production.
void refResolveNode(const Fixture& f, int l, double* dens_out, double* dens0_out,
                    double* cond_out) {
    refResolveSpectralHistoryDeltas(f, l, dens_out, dens0_out, cond_out);
}

} // namespace

void refBuildStream(const Fixture& f, double* dens_out, double* dens0_out,
                    double* cond_out) {
    // XSSet::BuildFlatXsStream's loop, pragma included.  gcc outlines an OpenMP
    // body into its own function, which is a DIFFERENT inlining context from a
    // plain loop -- and per WP22 that is exactly the kind of difference that
    // pins a different mask.  So the pragma stays, with the production schedule
    // clause; the `if` is false because the mining wants ONE deterministic
    // traversal and the outlining -- which is the part that matters here -- is
    // decided at compile time and happens either way.
    //
    // THIS TU MUST BE COMPILED WITH OpenMP ENABLED.  Without -fopenmp the
    // pragma is ignored, the body is not outlined, and the quotation is being
    // compiled in a context the production loop is not.  CMakeLists.txt links
    // it into the OpenMP-carrying targets for that reason and
    // tools/test_flatxs_stream_forms_contract.py pins the pragma.
#pragma omp parallel for schedule(dynamic, 64) if (false)
    for (int i = 0; i < f.nnode; ++i)
        refResolveNode(f, i, dens_out, dens0_out, cond_out);
}

} // namespace fsref
