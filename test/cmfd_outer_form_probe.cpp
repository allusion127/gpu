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

#include "CmfdOuterFormMine.h"
#include "CmfdOuterReference.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace co = rasbery::cmfd;

namespace {

using cmfdmine::bits;
using cmfdmine::geomOf;
using cmfdmine::mineForms;
using cmfdmine::mineStable;
using cmfdmine::scoreMask;
using cmfdmine::viewOf;

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
        const unsigned long long m = mineForms(f, 0ull, true);
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

    // 2. SELF-CALIBRATING: mine THIS host's mask and assert against that.
    //
    //    The mask records which multiply-adds the HOST COMPILER fused, which is
    //    a property of the build machine -- 0x6 on the authoring box, 0x7 on
    //    238's Xeon Gold 5317, where CO_PSI_ACC is fused.  Asserting the shipped
    //    literal therefore fails on one of the two hosts for a reason that has
    //    nothing to do with the code under test.  So the gate is: derive the
    //    host's mask, check the DERIVATION is stable, and score against it.
    //
    //    Stability is checked by descending from four different seeds.  If the
    //    fixture pins the answer, every descent lands on the same mask; if it
    //    does not, the mask is under-determined and THAT is the failure worth
    //    reporting -- not a disagreement with another machine's value.
    bool                     mine_sound = true;
    const unsigned long long mined       = mineStable(f, mine_sound);
    std::printf("  MINED ON THIS HOST: CMFD_OUTER_FORMS = 0x%llX   (build default 0x%llX)\n",
                mined, co::CMFD_OUTER_FORMS);
    check(mine_sound,
          "a search seed failed to reach zero mismatches -- this fixture does not "
          "determine the mask, which is a defect on any host");
    if (mined != co::CMFD_OUTER_FORMS)
        std::printf("  NOTE: this host's contraction differs from the build default. That "
                    "is expected across hosts; set RASBERY_CMFD_OUTER_FORMS=0x%llX for a "
                    "runtime build/reference mismatch.\n",
                    mined);

    // The mined mask must reproduce the reference, with the clamp both off
    // (production) and on (RASBERY_DHAT_CLAMP=1).
    for (const bool clamp : {false, true}) {
        const long bad = scoreMask(f, mined, clamp);
        std::printf("  mined mask 0x%llX, clamp=%-3s : %ld mismatching words\n", mined,
                    clamp ? "on" : "off", bad);
        check(bad == 0, std::string("the mined mask does not reproduce the CPU bodies "
                                    "(clamp=") +
                            (clamp ? "on" : "off") + ")");
    }
    check(co::CMFD_OUTER_FORMS == co::cmfdOuterForms(),
          "CMFD_OUTER_FORMS and cmfdOuterForms() disagree");
    // THE PRODUCTION RESOLVER MUST RETURN THE MINED MASK, not the baked one.
    //
    // This check used to accept the build default, which is precisely how the
    // campaign shipped a binary whose device outer rounded updpsi differently
    // from its own host loop: the gate mined 0x7, printed it as a NOTE, and then
    // asserted that the runtime was allowed to keep using 0x6.  The mined value
    // IS the contract on this host, so that is what the resolver has to hand the
    // kernels.
    check(co::cmfdOuterFormsRuntime() == mined ||
              std::getenv("RASBERY_CMFD_OUTER_FORMS") != nullptr,
          "cmfdOuterFormsRuntime() does not return this host's mined mask with no "
          "override set -- the device bodies would not reproduce the host loop");

    // 3. Each site must be DECISIVE on this fixture: if flipping a bit changes
    //    nothing, the fixture does not constrain it and the recorded value is a
    //    guess wearing a measurement's clothes.
    const unsigned long long shipped = mined;
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
