#!/usr/bin/env python3
"""Compile/run the host side of the shared two-group CMFD assembly body.

The test quotes the legacy CPU loop and compares every output byte against
cmfd_assembly::assembleNode2G for deterministic random inputs, both with and
without the Wielandt shift. It is intentionally a standalone C++20 harness so
it does not need HDF5 or CUDA.

WP21-A.  The legacy loop below is left in the NODE-MAJOR order it always had --
it is the historical text and the whole point of quoting it -- while
assembleNode2G now writes through rasbery::cmfd_layout.  So the comparison goes
through the layout helper, which makes this test the executable statement of
the permutation: every output value is identical, only its address moved.  With
kNodeInnermost = false the helper degenerates to l*per_node + k and this test is
byte-for-byte the one that ran before WP21-A.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("cmfd assembly kernel: FAIL: no C++ compiler available")

    source = r'''
#include "CmfdAssemblyKernel.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int NG = 2;
constexpr int NDIR = 3;
constexpr int LR = 2;
constexpr int NEWSBT = NDIR * LR;

bool same(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

void legacy(int nxyz,
            const std::vector<int>& node_surface,
            const std::vector<double>& face_area,
            const std::vector<double>& volume,
            const std::vector<double>& xsrf,
            const std::vector<double>& xssm,
            const std::vector<double>& chif,
            const std::vector<double>& xsnf,
            const std::vector<double>& dtil,
            const std::vector<double>& dhat,
            double reigvs,
            double eshift,
            std::vector<double>& diag,
            std::vector<double>& cc,
            std::vector<double>& udiag) {
    for (int l = 0; l < nxyz; ++l) {
        const double vol = volume[l];
        for (int ige = 0; ige < NG; ++ige) {
            for (int igs = 0; igs < NG; ++igs)
                diag[l * 4 + ige * 2 + igs] =
                    -xssm[(igs * 2 + ige) * nxyz + l] * vol;
            double& diagonal = diag[l * 4 + ige * 2 + ige];
            diagonal += xsrf[ige * nxyz + l] * vol;
            for (int idir = NDIR - 1; idir >= 0; --idir) {
                const int ls = node_surface[(l * NDIR + idir) * LR + 0];
                const double area = face_area[l * NDIR + idir];
                const double dt = dtil[ls * NG + ige];
                const double dh = dhat[ls * NG + ige];
                cc[l * 12 + ige * 6 + idir * 2 + 0] = (-dt + dh) * area;
                diagonal += (dt + dh) * area;
            }
            for (int idir = 0; idir < NDIR; ++idir) {
                const int ls = node_surface[(l * NDIR + idir) * LR + 1];
                const double area = face_area[l * NDIR + idir];
                const double dt = dtil[ls * NG + ige];
                const double dh = dhat[ls * NG + ige];
                cc[l * 12 + ige * 6 + idir * 2 + 1] = (-dt - dh) * area;
                diagonal += (dt - dh) * area;
            }
        }
        for (int ige = 0; ige < NG; ++ige) {
            for (int igs = 0; igs < NG; ++igs) {
                const int i = l * 4 + ige * 2 + igs;
                udiag[i] = diag[i];
                if (eshift != 0.0)
                    diag[i] = udiag[i] -
                        (chif[ige * nxyz + l] * xsnf[igs * nxyz + l] * reigvs * vol);
            }
        }
    }
}
} // namespace

int main() {
    constexpr int nxyz = 37;
    constexpr int nsurf = 113;
    std::mt19937_64 rng(20260825ULL);
    auto value = [&] (double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };

    std::vector<int> node_surface(nxyz * NDIR * LR);
    for (int& x : node_surface) x = static_cast<int>(rng() % nsurf);
    std::vector<double> face_area(nxyz * NDIR);
    std::vector<double> volume(nxyz);
    std::vector<double> xsrf(NG * nxyz);
    std::vector<double> xssm(NG * NG * nxyz);
    std::vector<double> chif(NG * nxyz);
    std::vector<double> xsnf(NG * nxyz);
    std::vector<double> dtil(nsurf * NG);
    std::vector<double> dhat(nsurf * NG);
    for (double& x : face_area) x = value(0.1, 500.0);
    for (double& x : volume) x = value(0.1, 5000.0);
    for (double& x : xsrf) x = value(1.0e-5, 3.0);
    for (double& x : xssm) x = value(-0.2, 0.7);
    for (double& x : chif) x = value(0.0, 1.0);
    for (double& x : xsnf) x = value(0.0, 0.2);
    for (double& x : dtil) x = value(1.0e-5, 2.0);
    for (double& x : dhat) x = value(-1.5, 1.5);

    for (double eshift : {0.0, 0.04}) {
        const double reigvs = eshift == 0.0 ? 0.0 : 0.9312456789012345;
        std::vector<double> expected_diag(nxyz * 4, 0.0);
        std::vector<double> expected_cc(nxyz * 12, 0.0);
        std::vector<double> expected_udiag(nxyz * 4, 0.0);
        legacy(nxyz, node_surface, face_area, volume, xsrf, xssm, chif, xsnf,
               dtil, dhat, reigvs, eshift,
               expected_diag, expected_cc, expected_udiag);

        std::vector<double> actual_diag(nxyz * 4, 0.0);
        std::vector<double> actual_cc(nxyz * 12, 0.0);
        std::vector<double> actual_udiag(nxyz * 4, 0.0);
        // WP21-A: the View reads node_surface and face_area through
        // cmfd_layout too, so they go in permuted the same way.
        std::vector<int> view_node_surface(node_surface.size());
        std::vector<double> view_face_area(face_area.size());
        for (int l = 0; l < nxyz; ++l) {
            for (int f = 0; f < NDIR * LR; ++f)
                view_node_surface[rasbery::cmfd_layout::face(nxyz, l, f)] =
                    node_surface[l * NDIR * LR + f];
            for (int d = 0; d < NDIR; ++d)
                view_face_area[rasbery::cmfd_layout::dir(nxyz, l, d)] =
                    face_area[l * NDIR + d];
        }
        rasbery::cmfd_assembly::View view{
            nxyz,
            view_node_surface.data(), view_face_area.data(), volume.data(),
            xsrf.data(), xssm.data(), chif.data(), xsnf.data(),
            dtil.data(), dhat.data(),
            reigvs, eshift,
            actual_diag.data(), actual_cc.data(), actual_udiag.data()
        };
        for (int l = 0; l < nxyz; ++l)
            rasbery::cmfd_assembly::assembleNode2G(view, l);

        // Permute the node-major reference into the layout the kernel wrote,
        // then compare BYTE FOR BYTE: WP21-A moved addresses, not values.
        std::vector<double> permuted_diag(nxyz * 4, 0.0);
        std::vector<double> permuted_cc(nxyz * 12, 0.0);
        std::vector<double> permuted_udiag(nxyz * 4, 0.0);
        for (int l = 0; l < nxyz; ++l) {
            for (int k = 0; k < 4; ++k) {
                const auto at = rasbery::cmfd_layout::mat(nxyz, l, k);
                permuted_diag[at] = expected_diag[l * 4 + k];
                permuted_udiag[at] = expected_udiag[l * 4 + k];
            }
            for (int j = 0; j < 12; ++j)
                permuted_cc[rasbery::cmfd_layout::cpl(nxyz, l, j)] =
                    expected_cc[l * 12 + j];
        }

        if (!same(permuted_diag, actual_diag)) return eshift == 0.0 ? 10 : 11;
        if (!same(permuted_cc, actual_cc)) return eshift == 0.0 ? 12 : 13;
        if (!same(permuted_udiag, actual_udiag)) return eshift == 0.0 ? 14 : 15;
    }
    return 0;
}
'''

    # The contract only holds at the production optimization level: gcc fuses
    # the setls accumulations into FMA at -O3 (Release) and not at -O2 or
    # below.  Prefer the flags the actual build used so a non-Release build
    # cannot pass this gate while breaking bit-exactness in the binary.
    opt_flags = ["-O3", "-march=native"]
    for flags_make in sorted(ROOT.glob("build*/CMakeFiles/RASBERY.dir/flags.make")):
        text = flags_make.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("CXX_FLAGS"):
                found = [tok for tok in line.split()
                         if tok.startswith("-O") or tok.startswith("-march")
                         or tok.startswith("-mtune") or tok.startswith("-ffp-contract")
                         or tok in ("-mno-fma", "-ffast-math")]
                if found:
                    opt_flags = found
                    print(f"cmfd assembly kernel: using build flags from {flags_make}: {' '.join(opt_flags)}")
                break
        break

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cpp = tmp_path / "cmfd_assembly_test.cpp"
        exe = tmp_path / "cmfd_assembly_test"
        cpp.write_text(source, encoding="utf-8")
        subprocess.run(
            [compiler, "-std=c++20", *opt_flags, "-Wall", "-Wextra",
             "-Werror", "-I", str(ROOT / "src"), str(cpp), "-o", str(exe)],
            check=True,
        )
        subprocess.run([str(exe)], check=True)
    print("cmfd assembly kernel: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
