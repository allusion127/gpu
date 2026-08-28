// Task 4 mapping gate: does the SHIPPED per-thread body reproduce
// Nodal::updateConstant, element for element and decision for decision?
//
//   ./rasbery_nodal_constant_gpu_replay [nxyz]            run the gate
//   ./rasbery_nodal_constant_gpu_replay --dump <file>     also write the CPU
//                                                        reference the device
//                                                        arm replays
//
// WHAT THIS FILE IS NOT.  It is not the N1 gate.  Both arms here run on the
// same host libm, so the transcendentals cannot differ and the comparison below
// is exact by construction on that axis.  What CAN differ -- and what this file
// exists to catch -- is everything the port added around the arithmetic:
//
//   * the PACKING of the nine coefficient arrays.  The arena's order puts
//     diagDI BEFORE diagD (GpuPhysicsArenaLayout.h, SlotRegion::NodalConst)
//     while Nodal.cpp's own dump order is the other way round.  Swapping them
//     is silent: both are finite, both are per-(node, dir, group), and the
//     answer is simply wrong three phases later.
//   * the (node, dir, group) INDEX, which the kernel spells as
//     (lk*NDIRMAX + idir)*ng + ig and Nodal.cpp spells as `eta1(ig, lkd)`.
//   * the EARLY-OUT SCOPE.  Nodal::updateConstant tests both groups of a node
//     and returns early only when both are unchanged; a per-(node, group) test
//     is a different function on any node where one group moved and the other
//     did not.  Scenario "one group moved" below is the case that separates
//     them, and it is not hypothetical: a boron or rod change moves the thermal
//     removal cross section on nodes where the fast one is unchanged to the
//     bit.
//   * the "unchanged means UNTOUCHED" property.  The host leaves the nine
//     arrays alone on an early-out; a device thread that writes them anyway
//     would agree numerically and still be wrong, because the arrays are what
//     the generation counter is a promise about.
//
// The device arm (test/nodal_constant_device_replay.cu) is where the N1
// deviation is measured.  Splitting them this way means a device failure is
// unambiguous: mapping bugs are already excluded here, without a GPU.

#include "CudaNodalConstantKernel.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ng_ = rasbery::gpu;
namespace nk  = rasbery::nodal;

namespace {

constexpr int NG   = 2;
constexpr int NDIR = 3;

// ---------------------------------------------------------------------------
// The PRE-POLICY body, verbatim.  This is the CPU baseline.
//
// NodalConstantKernel.h now routes every multiply-add through ncMa1/ncMa2 under
// a mined mask, because nvcc with --fmad=false will not reproduce the twenty
// FMAs gcc emits when it inlines this into Nodal::updateConstant.  That is only
// safe if the mask reproduces THIS function bit for bit -- otherwise the port
// silently moved the CPU answer, which is the one thing the v2 freeze forbids.
//
// So this copy stays, compiled in this TU at the production flags (-O3
// -march=native, gcc's default -ffp-contract=fast), and `--mine` searches the
// mask that reproduces it exactly.  tools/test_nodal_constant_kernel.py keeps a
// second, independent copy for the same purpose; two copies that agree is the
// point, not duplication to be tidied away.
// ---------------------------------------------------------------------------
nk::NodalConstantCoefficients legacyNodalConstantCoefficients(double xsrf, double xsdf,
                                                              double hmesh) {
    double kp2    = xsrf * hmesh * hmesh / (4 * xsdf);
    double kp     = std::sqrt(kp2);
    double kp3    = kp2 * kp;
    double kp4    = kp2 * kp2;
    double rkp    = 1 / kp;
    double rkp2   = rkp * rkp;
    double rkp3   = rkp2 * rkp;
    double rkp4   = rkp2 * rkp2;
    double rkp5   = rkp2 * rkp3;
    double ekp    = std::exp(kp);
    double iekp   = 1.0 / ekp;
    double sinhkp = 0.5 * (ekp - iekp);
    double coshkp = 0.5 * (ekp + iekp);

    double bfcff0 = -sinhkp * rkp;
    double bfcff2 = -5 * (-3 * kp * coshkp + 3 * sinhkp + kp2 * sinhkp) * rkp3;
    double bfcff4 =
        -9. * (-105 * kp * coshkp - 10 * kp3 * coshkp + 105 * sinhkp +
               45 * kp2 * sinhkp + kp4 * sinhkp) * rkp5;
    double bfcff1 = -3 * (kp * coshkp - sinhkp) * rkp2;
    double bfcff3 =
        -7 * (15 * kp * coshkp + kp3 * coshkp - 15 * sinhkp -
              6 * kp2 * sinhkp) * rkp4;

    double oddtemp  = 1 / (sinhkp + bfcff1 + bfcff3);
    double eventemp = 1 / (coshkp + bfcff0 + bfcff2 + bfcff4);

    nk::NodalConstantCoefficients out{};
    out.eta1 = (kp * coshkp + bfcff1 + 6 * bfcff3) * oddtemp;
    out.eta2 = (kp * sinhkp + 3 * bfcff2 + 10 * bfcff4) * eventemp;

    out.m260 = 2 * out.eta2;
    out.m251 = 2 * (kp * coshkp - sinhkp + 5 * bfcff3) * oddtemp;
    out.m253 = 2 * (kp * (15 + kp2) * coshkp - 3 * (5 + 2 * kp2) * sinhkp) *
               oddtemp * rkp2;
    out.m262 = 2 * (-3 * kp * coshkp + (3 + kp2) * sinhkp + 7 * kp * bfcff4) *
               eventemp * rkp;
    out.m264 = 2 * (-5 * kp * (21 + 2 * kp2) * coshkp +
                    (105 + 45 * kp2 + kp4) * sinhkp) * eventemp * rkp3;

    out.diagD  = 4 * xsdf / (hmesh * hmesh);
    out.diagDI = 1.0 / out.diagD;
    return out;
}

std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

/// One synthetic slot: the arrays the phase reads and writes, laid out exactly
/// as DeviceSlotView says they are.
struct Slot {
    int                 nxyz = 0;
    std::vector<double> xs;           ///< NXS packed slots, each [ig*nxyz + l]
    std::vector<double> hmesh;        ///< [l*NDIRMAX + dir]
    std::vector<double> nodal_const;  ///< 9 packed, each [nxyz*NDIRMAX*ng]
    std::vector<double> constant_xs;  ///< 2 packed, each [nxyz*ng]

    void resize(int n) {
        nxyz = n;
        xs.assign(static_cast<size_t>(rasbery::gpu::kDevNxs) * NG * n, 0.0);
        hmesh.assign(static_cast<size_t>(n) * NDIR, 0.0);
        nodal_const.assign(static_cast<size_t>(9) * n * NDIR * NG, 0.0);
        // The host cache starts as quiet NaN (Nodal.cpp:69-70), so the FIRST
        // pass can never take the early-out.  Reproducing that here matters:
        // starting it at 0.0 would make a node whose xsrf really is 0 look
        // already-current on pass one.
        constant_xs.assign(static_cast<size_t>(2) * n * NG,
                           std::nan(""));
    }

    double& xsAt(int xt, int ig, int l) {
        return xs[static_cast<size_t>(ng_::macroXsIndex(xt, ig, l, NG, nxyz))];
    }

    ng_::DeviceSlotView view() {
        ng_::DeviceSlotView v{};
        v.xs          = xs.data();
        v.nodal_const = nodal_const.data();
        v.constant_xs = constant_xs.data();
        v.nxyz        = nxyz;
        v.ng          = NG;
        return v;
    }

    ng_::DeviceGeometryView geom() {
        ng_::DeviceGeometryView g{};
        g.hmesh = hmesh.data();
        g.nxyz  = nxyz;
        g.ng    = NG;
        return g;
    }
};

// ---------------------------------------------------------------------------
// The reference: Nodal::updateConstant (src/Nodal.cpp:122-167), quoted.
//
// Kept as close to the original as the array plumbing allows -- same statement
// order, same `unchanged` accumulation, same early return, same trailing cache
// write.  The nine outputs are written through named pointers rather than the
// Nodal.cpp macros so the PACKING is visible at the call site and can be given
// the arena's order deliberately.
// ---------------------------------------------------------------------------

struct Reference {
    std::vector<double> nodal_const;
    std::vector<double> constant_xs;
    std::vector<char>   recomputed; ///< per (node,group): did updateConstant return true?
};

Reference referenceUpdateConstant(Slot& s, const std::vector<double>& prev_constant_xs,
                                  unsigned long long forms) {
    const int    nxyz = s.nxyz;
    const int    ng   = NG;
    Reference    out;
    out.nodal_const = s.nodal_const; // untouched entries must stay untouched
    out.constant_xs = prev_constant_xs;
    out.recomputed.assign(static_cast<size_t>(nxyz) * ng, 0);

    const long long stride = static_cast<long long>(nxyz) * NDIR * ng;
    double* const   c_xsrf = out.constant_xs.data();
    double* const   c_xsdf = out.constant_xs.data() + static_cast<size_t>(nxyz) * ng;

    for (int lk = 0; lk < nxyz; ++lk) {
        const int lkg0 = lk * ng;

        double xsrf_node[NG];
        double xsdf_node[NG];
        bool   unchanged = true;
        for (int ig = 0; ig < ng; ++ig) {
            xsrf_node[ig] = s.xsAt(rasbery::gpu::kXtXsrf, ig, lk);
            xsdf_node[ig] = s.xsAt(rasbery::gpu::kXtXsdf, ig, lk);
            unchanged     = unchanged && c_xsrf[lkg0 + ig] == xsrf_node[ig] &&
                        c_xsdf[lkg0 + ig] == xsdf_node[ig];
        }
        if (unchanged) continue;

        double hmesh_node[NDIR];
        for (int idir = 0; idir < NDIR; ++idir)
            hmesh_node[idir] = s.hmesh[static_cast<size_t>(lk) * NDIR + idir];

        for (int idir = 0; idir < NDIR; ++idir) {
            for (int ig = 0; ig < ng; ++ig) {
                const nk::NodalConstantCoefficients c = nk::nodalConstantCoefficients(
                    xsrf_node[ig], xsdf_node[ig], hmesh_node[idir], forms);
                const long long idx = (static_cast<long long>(lk) * NDIR + idir) * ng + ig;
                out.nodal_const[static_cast<size_t>(ng_::kNcEta1 * stride + idx)]   = c.eta1;
                out.nodal_const[static_cast<size_t>(ng_::kNcEta2 * stride + idx)]   = c.eta2;
                out.nodal_const[static_cast<size_t>(ng_::kNcM260 * stride + idx)]   = c.m260;
                out.nodal_const[static_cast<size_t>(ng_::kNcM251 * stride + idx)]   = c.m251;
                out.nodal_const[static_cast<size_t>(ng_::kNcM253 * stride + idx)]   = c.m253;
                out.nodal_const[static_cast<size_t>(ng_::kNcM262 * stride + idx)]   = c.m262;
                out.nodal_const[static_cast<size_t>(ng_::kNcM264 * stride + idx)]   = c.m264;
                out.nodal_const[static_cast<size_t>(ng_::kNcDiagDI * stride + idx)] = c.diagDI;
                out.nodal_const[static_cast<size_t>(ng_::kNcDiagD * stride + idx)]  = c.diagD;
            }
        }

        for (int ig = 0; ig < ng; ++ig) {
            c_xsrf[lkg0 + ig]                             = xsrf_node[ig];
            c_xsdf[lkg0 + ig]                             = xsdf_node[ig];
            out.recomputed[static_cast<size_t>(lkg0 + ig)] = 1;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Deck-shaped inputs.  Deterministic, no RNG library dependence: the values
// span the 2-group PWR/SMR envelope the exp probe's anchors were drawn from,
// including reflector rows and the two extreme mesh sizes.
// ---------------------------------------------------------------------------

void fillDeck(Slot& s, int nxyz, int variant) {
    s.resize(nxyz);
    const double xsrf_f[] = {0.0203, 0.0247, 0.0289, 0.0071};
    const double xsrf_t[] = {0.0612, 0.0937, 0.1153, 0.1487, 0.0304};
    const double xsdf_f[] = {1.4326, 1.2887, 1.1204};
    const double xsdf_t[] = {0.3721, 0.4015, 0.2687};
    const double hm[]     = {1.26, 3.81, 5.355, 10.71, 15.24, 21.42, 30.48};

    for (int l = 0; l < nxyz; ++l) {
        const int k = l + variant * 7;
        s.xsAt(rasbery::gpu::kXtXsrf, 0, l) = xsrf_f[k % 4];
        s.xsAt(rasbery::gpu::kXtXsrf, 1, l) = xsrf_t[k % 5];
        s.xsAt(rasbery::gpu::kXtXsdf, 0, l) = xsdf_f[k % 3];
        s.xsAt(rasbery::gpu::kXtXsdf, 1, l) = xsdf_t[(k / 3) % 3];
        for (int d = 0; d < NDIR; ++d)
            s.hmesh[static_cast<size_t>(l) * NDIR + d] = hm[(k + d * 2) % 7];
    }
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL %s\n", what.c_str());
        ++failures;
    }
}

/// Run one scenario: reference vs the shipped thread body, over every (node,
/// group), and compare the nine arrays, the two caches AND the recompute
/// decisions, bit for bit.
void runScenario(const char* name, Slot& s, bool expect_any_recompute,
                 unsigned long long forms) {
    const std::vector<double> before_const_xs = s.constant_xs;
    const std::vector<double> before_nodal    = s.nodal_const;

    Reference ref = referenceUpdateConstant(s, before_const_xs, forms);

    // The candidate arm starts from the same state the reference did.
    s.nodal_const = before_nodal;
    s.constant_xs = before_const_xs;
    ng_::DeviceSlotView     v = s.view();
    ng_::DeviceGeometryView g = s.geom();

    // The phase is TWO passes on the device (compute, then publish), because
    // the compute pass reads the cache the publish pass writes.  Driving them
    // in one loop here would hide exactly the race the split exists to remove,
    // so the loops are separate and in the enqueue's order.
    std::vector<char> got_recomputed(static_cast<size_t>(s.nxyz) * NG, 0);
    for (int lk = 0; lk < s.nxyz; ++lk)
        for (int ig = 0; ig < NG; ++ig)
            got_recomputed[static_cast<size_t>(lk * NG + ig)] =
                ng_::nodalConstantUpdateThread(v, g, lk, ig, forms) ? 1 : 0;
    for (int lk = 0; lk < s.nxyz; ++lk)
        for (int ig = 0; ig < NG; ++ig) ng_::nodalConstantPublishThread(v, lk, ig);

    long bad_const = 0, bad_cache = 0, bad_decision = 0;
    for (size_t i = 0; i < ref.nodal_const.size(); ++i)
        if (bits(ref.nodal_const[i]) != bits(s.nodal_const[i])) ++bad_const;
    for (size_t i = 0; i < ref.constant_xs.size(); ++i)
        if (bits(ref.constant_xs[i]) != bits(s.constant_xs[i])) ++bad_cache;
    for (size_t i = 0; i < ref.recomputed.size(); ++i)
        if (ref.recomputed[i] != got_recomputed[i]) ++bad_decision;

    long recomputed = 0;
    for (char c : got_recomputed) recomputed += c;

    std::printf("  %-28s nodes=%d recomputed=%ld/%zu  const_bad=%ld cache_bad=%ld "
                "decision_bad=%ld\n",
                name, s.nxyz, recomputed, got_recomputed.size(), bad_const, bad_cache,
                bad_decision);

    check(bad_const == 0, std::string(name) + ": nine coefficient arrays differ");
    check(bad_cache == 0, std::string(name) + ": constant_xsrf/xsdf cache differs");
    check(bad_decision == 0, std::string(name) + ": early-out decision differs");
    check(expect_any_recompute == (recomputed > 0),
          std::string(name) + ": scenario did not exercise what it claims");
}

/// Nodal.cpp's own dump order is eta1 eta2 m260 m251 m253 m262 m264 diagD
/// diagDI; the arena's is ... diagDI diagD.  Nothing enforces that difference
/// except this assertion, and a swap is invisible to every numerical test
/// because both arrays are populated with plausible numbers.
void checkPackingOrderIsDeliberate() {
    check(ng_::kNcDiagDI == 7 && ng_::kNcDiagD == 8,
          "arena packing: diagDI must precede diagD (SlotRegion::NodalConst)");
    check(ng_::kNcCount == 9, "arena packing: nine coefficient arrays");
    check(ng_::kCxXsrf == 0 && ng_::kCxXsdf == 1, "constant_xs packing: xsrf then xsdf");
}

/// The index the kernel writes must be the index Nodal.cpp reads.
void checkIndexAgreement(int nxyz) {
    for (int lk = 0; lk < nxyz; ++lk)
        for (int idir = 0; idir < NDIR; ++idir)
            for (int ig = 0; ig < NG; ++ig) {
                const long long want = (static_cast<long long>(lk) * NDIR + idir) * NG + ig;
                if (ng_::nodalConstIndex(lk, idir, ig, NG) != want) {
                    check(false, "nodalConstIndex disagrees with Nodal.cpp's eta1(ig, lkd)");
                    return;
                }
            }
}

// ---------------------------------------------------------------------------
// Contraction mining
// ---------------------------------------------------------------------------

/// The operand set the mask is mined over: the full 2-group envelope, so a bit
/// that only matters on reflector-sized kp is still pinned.
std::vector<std::array<double, 3>> miningOperands() {
    const double xsrf[] = {0.0203, 0.0247, 0.0289, 0.0612, 0.0937,
                           0.1153, 0.1487, 0.0071, 0.0304, 0.0015, 0.31};
    const double xsdf[] = {1.4326, 1.2887, 0.3721, 0.4015, 1.1204, 0.2687, 2.1, 0.15};
    const double hm[]   = {1.26, 3.81, 5.355, 10.71, 15.24, 21.42, 30.48, 0.63, 45.0};
    std::vector<std::array<double, 3>> v;
    for (double r : xsrf)
        for (double d : xsdf)
            for (double h : hm) v.push_back({r, d, h});
    return v;
}

/// A wide randomised operand set, log-uniform over the same envelope.  Fixed
/// LCG rather than <random> so the set is identical on every platform and a
/// failure is reproducible from the count alone.
std::vector<std::array<double, 3>> randomOperands(int n) {
    std::vector<std::array<double, 3>> v;
    v.reserve(static_cast<size_t>(n));
    std::uint64_t s = 0x243F6A8885A308D3ull;
    auto next = [&]() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((s >> 11) & ((1ull << 53) - 1)) /
               static_cast<double>(1ull << 53);
    };
    auto logu = [&](double lo, double hi) { return lo * std::pow(hi / lo, next()); };
    for (int i = 0; i < n; ++i)
        v.push_back({logu(1.0e-4, 1.0), logu(0.05, 5.0), logu(0.3, 60.0)});
    return v;
}

/// Number of differing output words for one candidate mask.
long scoreMask(const std::vector<std::array<double, 3>>& ops, unsigned long long m) {
    long bad = 0;
    for (const auto& o : ops) {
        const nk::NodalConstantCoefficients want = legacyNodalConstantCoefficients(o[0], o[1], o[2]);
        const nk::NodalConstantCoefficients got  = nk::nodalConstantCoefficients(o[0], o[1], o[2], m);
        const double* w = &want.eta1;
        const double* g = &got.eta1;
        for (int k = 0; k < 9; ++k)
            if (bits(w[k]) != bits(g[k])) ++bad;
    }
    return bad;
}

/// Coordinate descent over the sites: each 1-bit site has two states, each
/// 2-bit site three.  Sweep until a full pass improves nothing.  This is the
/// same method test/nodal_replay.cpp --sweep uses; it is adequate here because
/// the sites are nearly independent (each governs one rounding of one term).
unsigned long long mineForms(const std::vector<std::array<double, 3>>& ops,
                             unsigned long long seed = 0ull) {
    struct Site { int bit; int states; };
    std::vector<Site> sites;
    for (int b = 0; b < nk::NC_ONE_BIT_COUNT; ++b) sites.push_back({b, 2});
    for (int b = nk::NC_ONE_BIT_COUNT; b < nk::NC_BIT_COUNT; b += 2) sites.push_back({b, 3});

    unsigned long long best = seed;
    long               best_score = scoreMask(ops, best);
    for (int pass = 0; pass < 8 && best_score > 0; ++pass) {
        const long before = best_score;
        for (const Site& s : sites) {
            for (int state = 0; state < s.states; ++state) {
                const unsigned long long mask =
                    (best & ~(static_cast<unsigned long long>(s.states == 2 ? 1 : 3) << s.bit)) |
                    (static_cast<unsigned long long>(state) << s.bit);
                if (mask == best) continue;
                const long score = scoreMask(ops, mask);
                if (score < best_score) {
                    best_score = score;
                    best       = mask;
                }
            }
        }
        std::printf("  mine pass %d: %ld mismatching words, mask 0x%llXull\n", pass, best_score,
                    best);
        if (best_score == before) break;
    }
    return best;
}

/// Write the CPU reference the device arm replays: shape, inputs, and the nine
/// arrays this host libm produced.
bool writeDump(const char* path, Slot& s, const Reference& ref) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const std::int64_t hdr[4] = {s.nxyz, NDIR, NG, static_cast<std::int64_t>(9)};
    std::fwrite(hdr, sizeof hdr[0], 4, f);
    std::fwrite(s.xs.data(), sizeof(double), s.xs.size(), f);
    std::fwrite(s.hmesh.data(), sizeof(double), s.hmesh.size(), f);
    std::fwrite(ref.nodal_const.data(), sizeof(double), ref.nodal_const.size(), f);
    std::fwrite(ref.constant_xs.data(), sizeof(double), ref.constant_xs.size(), f);
    std::fclose(f);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    int         nxyz = 4096;
    const char* dump = nullptr;
    bool        mine = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump = argv[++i];
        } else if (std::strcmp(argv[i], "--mine") == 0) {
            mine = true;
        } else {
            nxyz = std::atoi(argv[i]);
        }
    }
    if (nxyz < 8) nxyz = 8;

    const std::vector<std::array<double, 3>> ops = miningOperands();

    if (mine) {
        std::printf("mining contraction forms over %zu (xsrf, xsdf, hmesh) triples\n",
                    ops.size());
        const unsigned long long m = mineForms(ops);
        const long               s = scoreMask(ops, m);
        std::printf("MINED NODAL_CONST_FORMS = 0x%llXull   (%ld mismatching words)\n", m, s);
        return s == 0 ? 0 : 1;
    }

    std::printf("nodal constant gpu replay: nxyz=%d ng=%d ndir=%d\n", nxyz, NG, NDIR);

    // 0. The mask that ships must reproduce the pre-policy CPU body EXACTLY.
    //    Everything else in this file is downstream of that: if the coefficients
    //    moved, comparing mappings is beside the point.
    //
    //    Scored on the mining grid AND on a much larger randomised operand set,
    //    because a mask fitted to a grid is only evidence about that grid: a
    //    site whose two states happen to agree on every mined triple would be
    //    pinned arbitrarily, and the wide set is what catches it.
    //
    //    SELF-CALIBRATING.  The mask records which multiply-adds THIS HOST's
    //    compiler fused, which is a property of the build machine: the CMFD mask
    //    was measured at 0x6 on the authoring box and 0x7 on 238's Xeon Gold
    //    5317, and this one has not been re-mined on 238 yet.  Asserting a
    //    literal would therefore fail on one of the two hosts for a reason that
    //    has nothing to do with the code.  So the gate derives the host's mask,
    //    checks the DERIVATION is stable across four search seeds, and scores
    //    against that -- the shipped constant is reported, not asserted.
    unsigned long long host_forms = nk::nodalConstForms();
    {
        // SOUND means every descent reaches ZERO mismatches -- not that every
        // descent produces the same bits.  Several sites here are provable
        // DON'T-CARES: NC_M253_W is `2*kp2 + 5` and NC_M264_P is `2*kp2 + 21`,
        // where the product is exact, so fused and unfused give the identical
        // double and different seeds settle on different bits with equal right.
        // Demanding pattern equality would fail a mask that is provably correct.
        bool stable = true;
        const unsigned long long seeds[4] = {0ull, (1ull << nk::NC_BIT_COUNT) - 1ull, 0x1ull,
                                             nk::NODAL_CONST_FORMS};
        for (int i = 0; i < 4; ++i) {
            const unsigned long long m = mineForms(ops, seeds[i]);
            if (scoreMask(ops, m) != 0) stable = false;
            if (i == 0) host_forms = m;
        }
        std::printf("  MINED ON THIS HOST: NODAL_CONST_FORMS = 0x%llX   (build default 0x%llX)\n",
                    host_forms, nk::NODAL_CONST_FORMS);
        check(stable,
              "a search seed failed to reach zero mismatches -- this operand set does not "
              "determine the mask, which is a defect on any host");
        if (host_forms != nk::NODAL_CONST_FORMS)
            std::printf("  NOTE: this host's contraction differs from the build default. "
                        "Expected across hosts; set RASBERY_NODAL_CONST_FORMS=0x%llX for a "
                        "runtime build/reference mismatch.\n",
                        host_forms);

        const long bad = scoreMask(ops, host_forms);
        std::printf("  %-28s %zu triples, %ld mismatching words vs the pre-policy body\n",
                    "mined form mask (grid)", ops.size(), bad);
        check(bad == 0, "the mined mask does not reproduce the CPU baseline");
        check(nk::NODAL_CONST_FORMS == nk::nodalConstForms(),
              "NODAL_CONST_FORMS and nodalConstForms() disagree");
        check(nk::nodalConstFormsRuntime() == nk::NODAL_CONST_FORMS ||
                  std::getenv("RASBERY_NODAL_CONST_FORMS") != nullptr,
              "nodalConstFormsRuntime() differs from the build default with no override set");

        const std::vector<std::array<double, 3>> wide = randomOperands(200000);
        const long wide_bad = scoreMask(wide, host_forms);
        std::printf("  %-28s %zu triples, %ld mismatching words\n", "mined form mask (random)",
                    wide.size(), wide_bad);
        check(wide_bad == 0,
              "the mined mask reproduces the mining grid but not the wider operand space "
              "-- a site is pinned to the wrong state");
    }

    checkPackingOrderIsDeliberate();
    checkIndexAgreement(nxyz < 64 ? nxyz : 64);

    Slot s;

    // 1. First pass: the cache is NaN, so every node recomputes.
    fillDeck(s, nxyz, 0);
    runScenario("first pass (all recompute)", s, true, host_forms);

    // 2. Immediately again with no material change: every node takes the
    //    early-out and NOTHING may be written.
    {
        const std::vector<double> frozen = s.nodal_const;
        runScenario("no change (all early-out)", s, false, host_forms);
        long touched = 0;
        for (size_t i = 0; i < frozen.size(); ++i)
            if (bits(frozen[i]) != bits(s.nodal_const[i])) ++touched;
        check(touched == 0, "early-out wrote to the coefficient arrays");
    }

    // 3. THERMAL group only moves, on every second node.  This is the scenario
    //    that separates a node-scoped early-out from a group-scoped one: on a
    //    perturbed node, group 0 is unchanged to the bit and must be rewritten
    //    anyway, because the host rewrites it.
    for (int l = 0; l < nxyz; l += 2)
        s.xsAt(rasbery::gpu::kXtXsrf, 1, l) *= 1.0009765625; // exact binary scale
    runScenario("thermal group only, half", s, true, host_forms);

    // 4. Diffusion coefficient only, one node.  The narrowest possible change.
    s.xsAt(rasbery::gpu::kXtXsdf, 0, nxyz / 3) *= 1.00390625;
    runScenario("one xsdf on one node", s, true, host_forms);

    // 5. A whole-deck material swap.
    {
        Slot t;
        fillDeck(t, nxyz, 1);
        t.constant_xs = s.constant_xs; // carry the cache: this is a perturbation
        t.nodal_const = s.nodal_const;
        runScenario("whole-deck material swap", t, true, host_forms);

        if (dump) {
            Slot                      fresh;
            fillDeck(fresh, nxyz, 1);
            const std::vector<double> start = fresh.constant_xs;
            const Reference           ref   = referenceUpdateConstant(fresh, start, host_forms);
            check(writeDump(dump, fresh, ref), std::string("cannot write dump ") + dump);
            if (!failures) std::printf("  wrote CPU reference dump: %s\n", dump);
        }
    }

    if (failures) {
        std::printf("nodal constant gpu replay: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("nodal constant gpu replay: PASS\n");
    return 0;
}
