#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src" / "NodalConstantKernel.h"

HARNESS = r'''
#include "NodalConstantKernel.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>

using rasbery::nodal::NodalConstantCoefficients;
using rasbery::nodal::nodalConstantCoefficients;

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
    return out;
}

static bool same(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

int main() {
    const std::array<double, 6> xsrf = {0.006, 0.012, 0.024, 0.055, 0.11, 0.22};
    const std::array<double, 5> xsdf = {0.18, 0.31, 0.62, 1.15, 1.8};
    const std::array<double, 6> h    = {2.5, 5.0, 10.0, 18.0, 20.0, 36.0};

    int cases = 0;
    for (double r : xsrf) {
        for (double d : xsdf) {
            for (double mesh : h) {
                const auto a = legacy(r, d, mesh);
                const auto b = nodalConstantCoefficients(r, d, mesh);
                const double av[] = {a.eta1, a.eta2, a.m260, a.m251, a.m253,
                                     a.m262, a.m264, a.diagD, a.diagDI};
                const double bv[] = {b.eta1, b.eta2, b.m260, b.m251, b.m253,
                                     b.m262, b.m264, b.diagD, b.diagDI};
                for (int i = 0; i < 9; ++i) {
                    if (!same(av[i], bv[i])) {
                        std::fprintf(stderr,
                                     "mismatch case=%d field=%d a=%a b=%a\n",
                                     cases, i, av[i], bv[i]);
                        return 1;
                    }
                }
                ++cases;
            }
        }
    }
    std::printf("PASS nodal constant bit contract: %d cases x 9 fields\n", cases);
    return 0;
}
'''


def main() -> int:
    if not HEADER.exists():
        raise SystemExit(f"missing production header: {HEADER}")
    with tempfile.TemporaryDirectory(prefix="nodal-constant-") as td:
        td_path = Path(td)
        src = td_path / "test.cpp"
        exe = td_path / "test"
        src.write_text(HARNESS, encoding="utf-8")
        compile_cmd = [
            "g++", "-std=c++20", "-O3", "-ffp-contract=fast",
            "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'src'}",
            str(src), "-o", str(exe),
        ]
        subprocess.run(compile_cmd, check=True)
        subprocess.run([str(exe)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
