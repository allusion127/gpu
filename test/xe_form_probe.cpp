// Rev.7.1 Task 13 gate: the Xe Anderson contraction mask and the fixed-partition
// inner product.
//
// SELF-CALIBRATING, LIKE cmfd_outer_form_probe.  It does not assert a literal
// mask -- a literal is a record of the machine it was measured on and would fail
// on the next host for a reason that has nothing to do with the code under test.
// It MINES this host's mask, checks the derivation is sound, and scores against
// that.  The shipped default is printed beside it so a host that contracts
// differently says so in one line.
//
// Four things are checked, and they are four different claims:
//
//   1. THE MINING IS SOUND.  Every one of four seeds reaches zero mismatching
//      words.  A seed that cannot is a site this fixture does not pin, and then
//      nothing here knows the contract.
//
//   2. EACH SITE IS DECISIVE.  Flipping a site away from the mined value must
//      BREAK something.  Without this, a mask of all zeros passes on a host
//      whose compiler fused nothing, and the gate would be asserting that the
//      fixture is boring rather than that the mask is right.
//
//   3. ONE PARTITION IS THE HOST'S FOLD, BIT FOR BIT.  This is the claim that
//      separates "the algebra is right" from "the partition moved it": with
//      RASBERY_GPU_XE_DOT_PARTITIONS=1 the device reduction is the host's serial
//      accumulation and must agree exactly.
//
//   4. THE PARTITION IS FIXED.  The same partition count gives the same answer
//      every time (that is what makes a run reproducible against itself), and a
//      DIFFERENT count is allowed to differ -- but by a rounding, not by a
//      physics-sized amount.  Reported, not asserted to be zero: it is exactly
//      the N1 difference Gate A/B is measuring, and a gate that demanded it be
//      zero would be demanding the feature not exist.

#include "XeFormMine.h"
#include "XeFormMask.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace xe = rasbery::xe;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

/// The whole-range reduction the kernel performs: partition, serial fold.
double partitionedDot(const xeref::Fixture& f, int parts, unsigned long long mask) {
    xe::XeTripleConst a = xemine::leftOf(f);
    xe::XeTripleConst b = xemine::rightOf(f);
    std::vector<double> partials(static_cast<std::size_t>(parts));
    for (int p = 0; p < parts; ++p) {
        int i0 = 0, i1 = 0;
        xe::xeDotPartitionRange(f.n, parts, p, &i0, &i1);
        partials[static_cast<std::size_t>(p)] = xe::xeDotChunk(a, b, i0, i1, mask);
    }
    return xe::xeDotFold(partials.data(), parts);
}

} // namespace

int main(int argc, char** argv) {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 4096;
    // buildMiningFixture, not buildFixture: the WP7-C sites are scored against
    // the `alg` cases its own translation unit builds, and a probe that skipped
    // them would mine four don't-cares and call them measured.
    const xeref::Fixture f = xemine::buildMiningFixture(n);

    // ---- 1. the mining is sound -------------------------------------------
    bool                     sound         = false;
    bool                     algebra_sound = false;
    const unsigned long long mined = xemine::mineStable(f, sound, algebra_sound);
    std::printf("mined XE_FORMS = 0x%llXull (shipped_sound=%d, algebra_sound=%d), "
                "build default 0x%llXull\n",
                mined, sound ? 1 : 0, algebra_sound ? 1 : 0,
                static_cast<unsigned long long>(xe::XE_FORMS_DEFAULT));
    // TWO CHECKS, because there are two channels and only one of them is in the
    // production arm.  A build on which the WP7-C sites cannot be mined still
    // has a well-defined RASBERY_GPU_XE mask; what it does not have is a
    // bit-identity claim for RASBERY_GPU_XE_TXN.
    check(sound, "the shipped-site mining reached zero mismatches from every seed");
    check(algebra_sound,
          "the WP7-C normal-equations mining reached zero mismatches from every seed");
    check(xemine::scoreShippedMask(f, mined) == 0,
          "the mined mask scores zero mismatching words on the shipped sites");
    check(xemine::scoreMask(f, mined) == 0, "the mined mask scores zero mismatching words");
    if (mined != xe::XE_FORMS_DEFAULT)
        std::printf("NOTE: this host contracts 0x%llXull where the build default says "
                    "0x%llXull; the default is a record of another machine, not of the "
                    "physics.\n",
                    mined, static_cast<unsigned long long>(xe::XE_FORMS_DEFAULT));

    // ---- 2. every site is decisive ----------------------------------------
    // A site whose states are arithmetically identical on THIS fixture is a
    // genuine don't-care and is reported rather than failed -- but all three
    // being don't-cares would mean the fixture pins nothing, and that is a
    // failure, because then the mask is unmeasured rather than measured to be
    // anything.
    struct Site {
        const char* name;
        int         bit;
        int         states;
    };
    // WP7-C added the last four: the normal equations moved onto the device, so
    // the four multiply-adds inside them became sites too.  They are listed here
    // for the same reason the first four are -- a site nobody probes is a site
    // that mines as whatever the seed happened to say.
    const Site sites[8] = {{"dot first product", xe::XE_DOT_FIRST_BIT, 3},
                           {"dot third product", xe::XE_DOT_THIRD_BIT, 2},
                           {"candidate, 1 column", xe::XE_CAND1_BIT, 2},
                           {"candidate, 2 columns", xe::XE_CAND2_BIT, 2},
                           {"gram determinant", xe::XE_TXN_DET_BIT, 3},
                           {"gamma[0] numerator", xe::XE_TXN_G0_BIT, 3},
                           {"gamma[1] numerator", xe::XE_TXN_G1_BIT, 3},
                           {"projection", xe::XE_TXN_PROJ_BIT, 3}};
    int decisive = 0;
    for (const Site& s : sites) {
        const unsigned long long field = (s.states == 2) ? 1ull : 3ull;
        const auto current = static_cast<unsigned>((mined >> s.bit) & field);
        bool       moved   = false;
        for (int state = 0; state < s.states; ++state) {
            if (static_cast<unsigned>(state) == current) continue;
            const unsigned long long alt =
                (mined & ~(field << s.bit)) | (static_cast<unsigned long long>(state) << s.bit);
            if (xemine::scoreMask(f, alt) > 0) moved = true;
        }
        std::printf("site %-22s state %u  %s\n", s.name, current,
                    moved ? "DECISIVE" : "don't-care on this fixture");
        if (moved) ++decisive;
    }
    check(decisive > 0, "the fixture pins at least one contraction site");

    // ---- 3. one partition is the host's fold, bit for bit -----------------
    const double host_dot = xeref::refDot(f, 0, n);
    const double one_part = partitionedDot(f, 1, mined);
    check(std::memcmp(&host_dot, &one_part, sizeof(double)) == 0,
          "RASBERY_GPU_XE_DOT_PARTITIONS=1 reproduces the host XeDot bit for bit");
    std::printf("host XeDot        = %.17e\n", host_dot);
    std::printf("1   partition     = %.17e  (bit-identical: %d)\n", one_part,
                std::memcmp(&host_dot, &one_part, sizeof(double)) == 0 ? 1 : 0);

    // ---- 4. the partition is fixed, and its cost is a rounding ------------
    for (int parts : {2, 64, 256, 1024}) {
        if (parts > n) continue;
        const double a = partitionedDot(f, parts, mined);
        const double b = partitionedDot(f, parts, mined);
        check(std::memcmp(&a, &b, sizeof(double)) == 0,
              "the fixed partition gives the same answer twice");
        const double rel =
            (host_dot != 0.0) ? std::fabs(a - host_dot) / std::fabs(host_dot) : 0.0;
        std::printf("%4d partitions   = %.17e  rel vs host %.3e\n", parts, a, rel);
        // A partition difference is a REASSOCIATION of the same terms; anything
        // beyond a handful of ULP over 3*n terms is not that, it is a wrong
        // partition map (an ordinal counted twice, or not at all).
        check(rel < 1.0e-12, "the partitioned dot differs from the host fold only by "
                             "reassociation");
    }

    // ---- the partition map itself: a cover, with no overlap ---------------
    for (int parts : {1, 3, 7, 64, 1024}) {
        if (parts > n) continue;
        int prev_end = 0;
        for (int p = 0; p < parts; ++p) {
            int i0 = 0, i1 = 0;
            xe::xeDotPartitionRange(n, parts, p, &i0, &i1);
            check(i0 == prev_end, "partitions are contiguous and in ascending order");
            check(i1 >= i0, "a partition is not inverted");
            prev_end = i1;
        }
        check(prev_end == n, "the partitions cover every ordinal exactly once");
    }

    // The production resolver, exercised rather than a test-only copy of it, so
    // a broken override or a broken receipt fails here and not in a campaign.
    const unsigned long long resolved = xe::xeFormMask();
    std::printf("resolved XE_FORMS = 0x%llXull\n", resolved);
    if (std::getenv("RASBERY_XE_FORMS") == nullptr)
        check(resolved == mined,
              "with no override the production resolver returns the mined mask");

    std::printf(failures == 0 ? "PASS\n" : "FAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
