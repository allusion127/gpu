// Task 5 form probe: which multiply-add did gcc fuse in the CMFD outer bodies,
// and does every guard branch of upddhat actually get taken?
//
//   ./rasbery_cmfd_outer_form_probe [nxyz]
//   ./rasbery_cmfd_outer_form_probe --mine        re-derive CMFD_OUTER_FORMS
//   ./rasbery_cmfd_outer_form_probe --dump <file> write the operand/result dump
//
// WHY A PROBE AND NOT A GUESS.  Task 5's bodies are Class B0: bit-identical to
// the CPU, no transcendental to hide behind.  gcc at -O3 -march=native
// (-ffp-contract=fast) fuses SOME of the multiply-adds and not others, and the
// pattern is not guessable from the source -- `_psi[l] += flux*xsnf` looks like
// the most obvious fma in the file and is NOT one, because the accumulator goes
// through memory every iteration.  So the mask is mined here and cross-checked
// against the production assembly quotations recorded in CmfdOuterKernel.h:
//
//     upddtil   no site
//     updpsi    NOT fused        (vmulsd then vaddsd)
//     updjnet   fused, second product rounded first
//               (vmulpd dhat*(fr+fl), then vfnmsub231pd)
//     upddhat   fused            (vfnmsub213sd)
//
// WHY THE BRANCH COVERAGE IS PART OF THIS FILE.  upddhat's guards are where the
// CNCC correction is allowed to throw its own answer away -- a zero d-hat, or a
// clamp that CMFD.cpp:164-178 records as worth ~+100 pcm at i-SMR CY01 BOC when
// enabled.  A mask mined on operands that never reach those branches would be
// mined on a fraction of the function.  So the fixture poisons specific
// surfaces to force each branch, and this probe FAILS if any of them is not
// reached -- coverage is checked, not hoped for.

#include "CmfdOuterKernel.h"

#include "cmfd_outer_reference.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace co = rasbery::cmfd;

namespace {

std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

co::CmfdGeometryView geomOf(const cmfdref::Mesh& m) {
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

co::CmfdOuterView viewOf(const cmfdref::Mesh& m, std::vector<double>& jnet,
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

/// Number of output words on which the shared bodies, under `mask`, disagree
/// with the separately compiled reference.
long scoreMask(const cmfdref::Fixture& f, unsigned long long mask, bool clamp_enabled) {
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

unsigned long long mineForms(const cmfdref::Fixture& f) {
    struct Site {
        int bit;
        int states;
    };
    std::vector<Site> sites;
    for (int b = 0; b < co::CO_ONE_BIT_COUNT; ++b) sites.push_back({b, 2});
    for (int b = co::CO_ONE_BIT_COUNT; b < co::CO_BIT_COUNT; b += 2) sites.push_back({b, 3});

    unsigned long long best       = 0ull;
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
        std::printf("  mine pass %d: %ld mismatching words, mask 0x%llXull\n", pass, best_score,
                    best);
        if (best_score == before) break;
    }
    return best;
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL %s\n", what.c_str());
        ++failures;
    }
}

/// Every guard branch, and which sites reached it.  Returns the per-branch hit
/// count so the caller can require all of them.
std::vector<long> branchCoverage(const cmfdref::Fixture& f, bool clamp_enabled) {
    const cmfdref::Mesh        m = f.mesh();
    std::vector<double>        jnet = f.jnet, dtil = f.dtil, dhat = f.dhat;
    std::vector<double>        psi(static_cast<size_t>(f.nxyz), 0.0);
    const co::CmfdGeometryView g = geomOf(m);
    const co::CmfdOuterView    v = viewOf(m, jnet, dtil, dhat, psi);

    std::vector<long> hits(static_cast<size_t>(cmfdref::DhatBranch::Count), 0);
    for (int ls = 0; ls < m.nsurf; ++ls)
        for (int ig = 0; ig < m.ng; ++ig) {
            const co::CmfdDhatContribution c =
                co::cmfdUpdDhatSurface(g, v, ls, ig, clamp_enabled, co::cmfdOuterForms());
            const double dtl = v.dtil[ls * m.ng + ig];
            cmfdref::DhatBranch b = cmfdref::DhatBranch::Normal;
            if (c.fsum_guard) {
                // Distinguish the three ways the guards fire, using the same
                // operands the body saw.
                const int    ll = m.surface_node[ls * cmfdref::NLR + 0];
                const int    lr = m.surface_node[ls * cmfdref::NLR + 1];
                const double fsum = (ll < 0)   ? m.flux[lr * m.ng + ig]
                                    : (lr < 0) ? m.flux[ll * m.ng + ig]
                                               : m.flux[lr * m.ng + ig] + m.flux[ll * m.ng + ig];
                if (!std::isfinite(fsum))
                    b = cmfdref::DhatBranch::FsumNonFinite;
                else if (!(std::fabs(fsum) >
                           1.0e-12 * (1.0 > std::fabs(dtl) ? 1.0 : std::fabs(dtl))))
                    b = cmfdref::DhatBranch::FsumBelowFloor;
                else
                    b = cmfdref::DhatBranch::QuotientNonFinite;
            } else if (dtl == 0.0) {
                b = cmfdref::DhatBranch::ZeroDtil;
            } else if (c.clamped) {
                b = cmfdref::DhatBranch::OverEnvelope;
            }
            ++hits[static_cast<size_t>(b)];
        }
    return hits;
}

const char* kBranchName[] = {"normal", "fsum<floor", "fsum non-finite",
                             "quotient non-finite", "|dtil|==0", "over-envelope"};

/// The operand/result dump the task asks for: every surface's topology, its
/// operands and the reference results, plus the branch each site took.  Written
/// as text on purpose -- its consumer is a person reading a diff, not a tool.
bool writeDump(const char* path, const cmfdref::Fixture& f, bool clamp_enabled) {
    std::FILE* fp = std::fopen(path, "w");
    if (!fp) return false;
    const cmfdref::Mesh m = f.mesh();

    std::vector<double>   ref_dtil(f.dtil.size()), ref_jnet(f.jnet.size()),
        ref_dhat(f.dhat.size()), ref_psi(static_cast<size_t>(f.nxyz));
    cmfdref::DhatCounters ref_counters;
    cmfdref::refUpdDtil(m, ref_dtil.data());
    cmfdref::refUpdPsi(m, ref_psi.data());
    cmfdref::refUpdJnet(m, ref_jnet.data());
    cmfdref::refUpdDhat(m, clamp_enabled, ref_dhat.data(), &ref_counters);

    std::fprintf(fp, "# cmfd outer form probe dump: nxyz=%d nsurf=%d ng=%d clamp=%d\n",
                 m.nxyz, m.nsurf, m.ng, clamp_enabled ? 1 : 0);
    std::fprintf(fp, "# counters total=%lld fsum_guard=%lld clamped=%lld ratio_max=%a\n",
                 ref_counters.total, ref_counters.fsum_guard, ref_counters.clamped,
                 ref_counters.ratio_max);
    std::fprintf(fp, "# ls ig kind ll lr idirl idirr fl fr dtil_in dhat_in jnet_in "
                     "-> dtil dhat jnet\n");
    for (int ls = 0; ls < m.nsurf; ++ls) {
        const int ll = m.surface_node[ls * cmfdref::NLR + 0];
        const int lr = m.surface_node[ls * cmfdref::NLR + 1];
        const char* kind = (ll < 0) ? "left-boundary" : (lr < 0) ? "right-boundary" : "internal";
        for (int ig = 0; ig < m.ng; ++ig) {
            const size_t k = static_cast<size_t>(ls) * m.ng + ig;
            std::fprintf(fp,
                         "%d %d %s %d %d %d %d %a %a %a %a %a -> %a %a %a\n", ls, ig, kind, ll,
                         lr, m.surface_dir[ls * cmfdref::NLR + 0],
                         m.surface_dir[ls * cmfdref::NLR + 1],
                         ll < 0 ? 0.0 : m.flux[ll * m.ng + ig],
                         lr < 0 ? 0.0 : m.flux[lr * m.ng + ig], m.dtil[k], m.dhat[k], m.jnet[k],
                         ref_dtil[k], ref_dhat[k], ref_jnet[k]);
        }
    }
    for (int l = 0; l < m.nxyz; ++l)
        std::fprintf(fp, "psi %d vol=%a f0=%a f1=%a nf0=%a nf1=%a -> %a\n", l,
                     m.node_volume[l], m.flux[l * m.ng + 0], m.flux[l * m.ng + 1],
                     m.xsnf[0 * m.nxyz + l], m.xsnf[1 * m.nxyz + l],
                     ref_psi[static_cast<size_t>(l)]);
    std::fclose(fp);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    int         nxyz = 512;
    bool        mine = false;
    const char* dump = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mine") == 0) {
            mine = true;
        } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump = argv[++i];
        } else {
            nxyz = std::atoi(argv[i]);
        }
    }
    const cmfdref::Fixture f = cmfdref::buildFixture(nxyz);

    if (mine) {
        std::printf("mining CMFD outer contraction forms: nxyz=%d nsurf=%d\n", f.nxyz, f.nsurf);
        const unsigned long long m = mineForms(f);
        const long               s = scoreMask(f, m, false);
        std::printf("MINED CMFD_OUTER_FORMS = 0x%llXull   (%ld mismatching words)\n", m, s);
        return s == 0 ? 0 : 1;
    }

    std::printf("cmfd outer form probe: nxyz=%d nsurf=%d ng=%d\n", f.nxyz, f.nsurf,
                cmfdref::NG);

    // 1. Coverage FIRST: a mask scored on operands that never reach the guards
    //    is a mask for part of the function.
    for (const bool clamp : {false, true}) {
        const std::vector<long> hits = branchCoverage(f, clamp);
        std::printf("  branch coverage (clamp=%s):", clamp ? "on" : "off");
        for (size_t b = 0; b < hits.size(); ++b)
            std::printf(" %s=%ld", kBranchName[b], hits[b]);
        std::printf("\n");
        for (size_t b = 0; b < hits.size(); ++b)
            check(hits[b] > 0,
                  std::string("upddhat branch '") + kBranchName[b] +
                      "' is never reached, so the mask was mined without it");
    }

    // 2. The shipped mask must reproduce the reference, with the clamp both off
    //    (production) and on (RASBERY_DHAT_CLAMP=1).
    for (const bool clamp : {false, true}) {
        const long bad = scoreMask(f, co::cmfdOuterForms(), clamp);
        std::printf("  shipped mask 0x%llX, clamp=%-3s : %ld mismatching words\n",
                    co::cmfdOuterForms(), clamp ? "on" : "off", bad);
        check(bad == 0, std::string("CMFD_OUTER_FORMS does not reproduce the CPU bodies "
                                    "(clamp=") +
                            (clamp ? "on" : "off") + ") -- re-run with --mine");
    }
    check(co::CMFD_OUTER_FORMS == co::cmfdOuterForms(),
          "CMFD_OUTER_FORMS and cmfdOuterForms() disagree");

    // 3. Each mined site must be DECISIVE on this fixture: if flipping a bit
    //    changes nothing, the fixture does not constrain it and the recorded
    //    value is a guess wearing a measurement's clothes.
    const unsigned long long shipped = co::cmfdOuterForms();
    for (int b = 0; b < co::CO_ONE_BIT_COUNT; ++b) {
        const long flipped = scoreMask(f, shipped ^ (1ull << b), false);
        std::printf("  site %d flipped: %ld mismatching words\n", b, flipped);
        check(flipped > 0,
              "a 1-bit form site is not constrained by this fixture; either the site does "
              "not exist or the operands never make the two roundings differ");
    }
    for (int b = co::CO_ONE_BIT_COUNT; b < co::CO_BIT_COUNT; b += 2) {
        const unsigned long long cur = (shipped >> b) & 3ull;
        long                     worst = 0;
        for (unsigned st = 0; st < 3; ++st) {
            if (st == cur) continue;
            const unsigned long long mask =
                (shipped & ~(3ull << b)) | (static_cast<unsigned long long>(st) << b);
            const long s = scoreMask(f, mask, false);
            if (s > worst) worst = s;
        }
        std::printf("  2-bit site %d (state %u): worst alternative %ld mismatching words\n", b,
                    static_cast<unsigned>(cur), worst);
        check(worst > 0, "a 2-bit form site is not constrained by this fixture");
    }

    if (dump) {
        check(writeDump(dump, f, false), std::string("cannot write dump ") + dump);
        if (!failures) std::printf("  wrote operand/result dump: %s\n", dump);
    }

    if (failures) {
        std::printf("cmfd outer form probe: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("cmfd outer form probe: PASS\n");
    return 0;
}
