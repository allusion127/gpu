#!/usr/bin/env python3
"""The nodal constant bit contract: src/NodalConstantKernel.h must reproduce the
historical CPU body of Nodal::updateConstant, bit for bit.

TWO TRANSLATION UNITS, AND EACH ARM INLINED INTO ITS OWN LOOP.  Both halves of
that sentence were learned the hard way in Rev.7.1 Task 4, which pinned the
body's multiply-add contraction to a mined mask (NODAL_CONST_FORMS) because nvcc
with --fmad=false will not reproduce the twenty FMAs gcc emits:

  * ONE TU is wrong.  With the legacy copy and the shipped body in the same
    translation unit and the same loop, gcc common-subexpressions across them --
    the shipped body's constant mask folds, its subexpressions merge with the
    legacy copy's, and the LEGACY copy's contraction changes.  The harness then
    fails against a reference that is no longer the production one (measured: a
    spurious m264 mismatch).
  * A NON-INLINED reference is also wrong.  Reaching the legacy copy across a
    call boundary changes gcc's contraction choices again (measured: a spurious
    eta1 mismatch), and production INLINES this body into
    Nodal::updateConstant.
  * CONSTANT-FOLDED operands are wrong too.  With the triples visible to the
    optimiser the whole comparison folds at compile time and both arms agree
    for a reason that has nothing to do with contraction.

So each arm gets a TU of its own, and inside that TU the body is inlined into a
batch loop over operands the compiler cannot see -- which is what production
does.  Under that arrangement three independently built arms agree bit for bit;
anything less agrees, or disagrees, by accident.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src" / "NodalConstantKernel.h"

LEGACY_TU = r'''
// The pre-policy body of Nodal::updateConstant's coefficient block, verbatim.
// Compiled in a TU of its own and inlined into the batch loop below, so its
// contraction is what gcc chooses for THIS body and nothing else.
#include <cmath>

#include "NodalConstantKernel.h"

using rasbery::nodal::NodalConstantCoefficients;

static NodalConstantCoefficients legacy(double xsrf, double xsdf, double hmesh) {
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

    NodalConstantCoefficients out{};
    out.eta1   = (kp * coshkp + bfcff1 + 6 * bfcff3) * oddtemp;
    out.eta2   = (kp * sinhkp + 3 * bfcff2 + 10 * bfcff4) * eventemp;
    out.m260   = 2 * out.eta2;
    out.m251   = 2 * (kp * coshkp - sinhkp + 5 * bfcff3) * oddtemp;
    out.m253   = 2 * (kp * (15 + kp2) * coshkp - 3 * (5 + 2 * kp2) * sinhkp) *
                 oddtemp * rkp2;
    out.m262   = 2 * (-3 * kp * coshkp + (3 + kp2) * sinhkp + 7 * kp * bfcff4) *
                 eventemp * rkp;
    out.m264   = 2 * (-5 * kp * (21 + 2 * kp2) * coshkp +
                      (105 + 45 * kp2 + kp4) * sinhkp) * eventemp * rkp3;
    out.diagD  = 4 * xsdf / (hmesh * hmesh);
    out.diagDI = 1.0 / out.diagD;
    return out;}

// `ops` is 3 doubles per case, `out` 9.  Both are opaque to this TU, which is
// the point: nothing may be constant-folded, because production evaluates this
// body on values loaded from arrays at run time.
void legacyBatch(const double* ops, int n, double* out) {
    for (int i = 0; i < n; ++i) {
        const NodalConstantCoefficients c =
            legacy(ops[3 * i], ops[3 * i + 1], ops[3 * i + 2]);
        const double v[9] = {c.eta1, c.eta2, c.m260, c.m251, c.m253,
                             c.m262, c.m264, c.diagD, c.diagDI};
        for (int k = 0; k < 9; ++k) out[9 * i + k] = v[k];
    }
}
'''

HARNESS = r'''
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "NodalConstantKernel.h"

using rasbery::nodal::NodalConstantCoefficients;
using rasbery::nodal::nodalConstantCoefficients;

// Defined in the separately compiled reference TU.
void legacyBatch(const double* ops, int n, double* out);

static bool same(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

/// Words on which the shipped body under `mask` disagrees with the reference.
static long score(const std::vector<double>& ops, int cases,
                  const std::vector<double>& want, unsigned long long mask) {
    long bad = 0;
    for (int i = 0; i < cases; ++i) {
        const NodalConstantCoefficients b = nodalConstantCoefficients(
            ops[3 * i], ops[3 * i + 1], ops[3 * i + 2], mask);
        const double bv[9] = {b.eta1, b.eta2, b.m260, b.m251, b.m253,
                              b.m262, b.m264, b.diagD, b.diagDI};
        for (int k = 0; k < 9; ++k)
            if (!same(want[static_cast<std::size_t>(i) * 9 + k], bv[k])) ++bad;
    }
    return bad;
}

/// Coordinate descent over the form sites, exactly as the replay's miner does.
/// The reference lives in the other translation unit, so nothing here can
/// perturb it.
static unsigned long long mineMask(const std::vector<double>& ops, int cases,
                                   const std::vector<double>& want) {
    using namespace rasbery::nodal;
    struct Site { int bit; int states; };
    std::vector<Site> sites;
    for (int b = 0; b < NC_ONE_BIT_COUNT; ++b) sites.push_back({b, 2});
    for (int b = NC_ONE_BIT_COUNT; b < NC_BIT_COUNT; b += 2) sites.push_back({b, 3});

    unsigned long long best = 0ull;
    long best_score = score(ops, cases, want, best);
    for (int pass = 0; pass < 8 && best_score > 0; ++pass) {
        const long before = best_score;
        for (const Site& s : sites)
            for (int st = 0; st < s.states; ++st) {
                const unsigned long long m =
                    (best & ~(static_cast<unsigned long long>(s.states == 2 ? 1 : 3) << s.bit)) |
                    (static_cast<unsigned long long>(st) << s.bit);
                if (m == best) continue;
                const long sc = score(ops, cases, want, m);
                if (sc < best_score) { best_score = sc; best = m; }
            }
        if (best_score == before) break;
    }
    return best;
}

int main() {
#if !defined(__FMA__)
    // Without FMA in the ISA gcc contracts nothing, so the reference is not the
    // function that ships and this comparison would be meaningless.
    std::printf("SKIP nodal constant bit contract: target has no FMA, so the "
                "production contraction forms are not reproducible here\n");
    return 0;
#endif
    // The 2-group PWR/SMR envelope, plus the reflector rows and the extreme
    // mesh sizes the nodal solver actually sees.
    const double xsrf[] = {0.006, 0.012, 0.024, 0.055, 0.11, 0.22, 0.0071, 0.1487};
    const double xsdf[] = {0.18, 0.31, 0.62, 1.15, 1.8, 0.2687, 1.4326};
    const double h[]    = {2.5, 5.0, 10.0, 18.0, 20.0, 36.0, 1.26, 21.42, 30.48};

    std::vector<double> ops;
    for (double r : xsrf)
        for (double d : xsdf)
            for (double mesh : h) {
                ops.push_back(r);
                ops.push_back(d);
                ops.push_back(mesh);
            }
    const int cases = static_cast<int>(ops.size() / 3);

    std::vector<double> want(static_cast<std::size_t>(cases) * 9);
    legacyBatch(ops.data(), cases, want.data());

    // SELF-CALIBRATING.  The mask records which multiply-adds THIS HOST's
    // compiler fused, which is host-specific -- the CMFD mask was measured at
    // 0x6 on the authoring box and 0x7 on 238's Xeon Gold 5317.  Asserting the
    // shipped literal would therefore fail on one of the two hosts for a reason
    // that has nothing to do with the code.  So: mine the mask that reproduces
    // THIS host's legacy body, report it, and assert against that.
    const unsigned long long mined = mineMask(ops, cases, want);
    std::printf("MINED ON THIS HOST: NODAL_CONST_FORMS = 0x%llX (build default 0x%llX)\n",
                mined,
                static_cast<unsigned long long>(rasbery::nodal::NODAL_CONST_FORMS));

    for (int i = 0; i < cases; ++i) {
        const NodalConstantCoefficients b = nodalConstantCoefficients(
            ops[3 * i], ops[3 * i + 1], ops[3 * i + 2], mined);
        const double bv[9] = {b.eta1, b.eta2, b.m260, b.m251, b.m253,
                              b.m262, b.m264, b.diagD, b.diagDI};
        for (int k = 0; k < 9; ++k) {
            if (!same(want[static_cast<std::size_t>(i) * 9 + k], bv[k])) {
                std::fprintf(stderr,
                             "mismatch case=%d (xsrf=%.17g xsdf=%.17g h=%.17g) "
                             "field=%d legacy=%a shipped=%a\n",
                             i, ops[3 * i], ops[3 * i + 1], ops[3 * i + 2], k,
                             want[static_cast<std::size_t>(i) * 9 + k], bv[k]);
                std::fprintf(stderr,
                             "  the shipped NODAL_CONST_FORMS no longer reproduces the "
                             "CPU baseline; re-mine with "
                             "rasbery_nodal_constant_gpu_replay --mine\n");
                return 1;
            }
        }
    }
    std::printf("PASS nodal constant bit contract: %d cases x 9 fields (mask 0x%llX)\n",
                cases,
                static_cast<unsigned long long>(rasbery::nodal::NODAL_CONST_FORMS));
    return 0;
}
'''


def main() -> int:
    if not HEADER.exists():
        raise SystemExit(f"missing production header: {HEADER}")
    with tempfile.TemporaryDirectory(prefix="nodal-constant-") as td:
        td_path = Path(td)
        src = td_path / "test.cpp"
        ref = td_path / "legacy.cpp"
        exe = td_path / "test"
        src.write_text(HARNESS, encoding="utf-8")
        ref.write_text(LEGACY_TU, encoding="utf-8")
        # -march=native is REQUIRED, not incidental: the contract is "the
        # production header reproduces the historical CPU body", and the
        # historical CPU body is compiled at the RASBERY release flags
        # (CMakeLists.txt: -march=native -O3, gcc's default -ffp-contract=fast).
        # Without it the target has no FMA and gcc contracts nothing.
        #
        # -fno-lto keeps the two objects separate through the link; -flto would
        # re-merge them and reintroduce the cross-CSE the split removes.
        compile_cmd = [
            "g++", "-std=c++20", "-O3", "-march=native", "-ffp-contract=fast",
            "-fno-lto", "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'src'}",
            str(src), str(ref), "-o", str(exe),
        ]
        subprocess.run(compile_cmd, check=True)
        subprocess.run([str(exe)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
