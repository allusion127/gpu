#!/usr/bin/env python3
"""Apply the Nodal::updateConstant hot-path refactor exactly once."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

NEW_UPDATE = r'''bool Nodal::updateConstant(const int& lk) {
    const int lkg0 = lk * _ng;

    // Nodal is a two-group method throughout (the matrix/even phases use
    // fixed 2x2 storage). Snapshot the material inputs once per node instead
    // of re-running XSSet/Geometry indexing in every direction and output.
    double xsrf_node[nodal::NG];
    double xsdf_node[nodal::NG];
    bool   unchanged = true;
    for (int ig = 0; ig < _ng; ++ig) {
        xsrf_node[ig] = xs.xsrf(ig, lk);
        xsdf_node[ig] = xs.xsdf(ig, lk);
        unchanged = unchanged &&
                    _constant_xsrf[lkg0 + ig] == xsrf_node[ig] &&
                    _constant_xsdf[lkg0 + ig] == xsdf_node[ig];
    }
    if (unchanged) return false;

    double hmesh_node[NDIRMAX];
    for (int idir = 0; idir < NDIRMAX; ++idir)
        hmesh_node[idir] = _g.hmesh(idir, lk);

    const int lkd0 = lk * NDIRMAX;
    for (int idir = 0; idir < NDIRMAX; ++idir) {
        const int lkd = lkd0 + idir;
        for (int ig = 0; ig < _ng; ++ig) {
            const nodal::NodalConstantCoefficients c =
                nodal::nodalConstantCoefficients(xsrf_node[ig], xsdf_node[ig],
                                                  hmesh_node[idir]);
            eta1(ig, lkd)   = c.eta1;
            eta2(ig, lkd)   = c.eta2;
            m260(ig, lkd)   = c.m260;
            m251(ig, lkd)   = c.m251;
            m253(ig, lkd)   = c.m253;
            m262(ig, lkd)   = c.m262;
            m264(ig, lkd)   = c.m264;
            diagD(ig, lkd)  = c.diagD;
            diagDI(ig, lkd) = c.diagDI;
        }
    }

    for (int ig = 0; ig < _ng; ++ig) {
        _constant_xsrf[lkg0 + ig] = xsrf_node[ig];
        _constant_xsdf[lkg0 + ig] = xsdf_node[ig];
    }
    return true;
}'''

OLD_GPU_LOOP = r'''    // Phase 1 on the host: recompute only where xsrf/xsdf moved (shadowed),
    // bumping _const_generation when anything did.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (_nxyz > rasbery_omp_gate)
#endif
    for (int lk = 0; lk < _nxyz; ++lk)
        updateConstant(lk);
'''

NEW_GPU_LOOP = r'''    // Phase 1 on the host: recompute only where xsrf/xsdf moved. The
    // per-node function returns a dirty bit; OpenMP combines those bits and
    // advances the device-residency generation exactly once per drive. This
    // removes the previous data race on _const_generation and avoids thousands
    // of redundant increments when a whole core state changes.
    int constants_changed = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(| : constants_changed) schedule(static) if (_nxyz > rasbery_omp_gate)
#endif
    for (int lk = 0; lk < _nxyz; ++lk)
        constants_changed |= updateConstant(lk) ? 1 : 0;
    if (constants_changed != 0)
        ++_const_generation;
'''

OLD_DRIVE_PREFIX = r'''void Nodal::driveBody() {
    // Each per-node / per-surface routine is independent (writes its own node/surface data; reads
    // neighbours only across the implicit barrier between phases). One parallel region with a
    // barrier per phase amortizes fork/join; results are bit-identical (no cross-node reduction).
#pragma omp parallel if (_nxyz > rasbery_omp_gate)
    {
#pragma omp for schedule(static)
        for (int lk = 0; lk < _nxyz; ++lk)
            updateConstant(lk);
'''

NEW_DRIVE_PREFIX = r'''void Nodal::driveBody() {
    // Each per-node / per-surface routine is independent (writes its own node/surface data; reads
    // neighbours only across the implicit barrier between phases). One parallel region with a
    // barrier per phase amortizes fork/join; results are bit-identical (no cross-node reduction).
    int constants_changed = 0;
#pragma omp parallel if (_nxyz > rasbery_omp_gate)
    {
#pragma omp for reduction(| : constants_changed) schedule(static)
        for (int lk = 0; lk < _nxyz; ++lk)
            constants_changed |= updateConstant(lk) ? 1 : 0;
#pragma omp single
        {
            if (constants_changed != 0)
                ++_const_generation;
        }
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def apply(root: Path) -> None:
    cpp_path = root / "src" / "Nodal.cpp"
    h_path = root / "src" / "Nodal.h"
    cpp = cpp_path.read_text(encoding="utf-8-sig")
    hdr = h_path.read_text(encoding="utf-8-sig")

    if 'bool Nodal::updateConstant' in cpp and '#include "NodalConstantKernel.h"' in cpp:
        print("nodal constant refactor already applied")
        return

    cpp = replace_once(
        cpp,
        '#include "Nodal.h"\n',
        '#include "Nodal.h"\n#include "NodalConstantKernel.h"\n',
        "include",
    )
    pattern = re.compile(
        r"void Nodal::updateConstant\(const int& lk\) \{.*?\n\}\n\nvoid Nodal::updateMatrix",
        re.S,
    )
    cpp, count = pattern.subn(NEW_UPDATE + "\n\nvoid Nodal::updateMatrix", cpp, count=1)
    if count != 1:
        raise RuntimeError(f"updateConstant body: expected one match, found {count}")

    cpp = replace_once(cpp, OLD_GPU_LOOP, NEW_GPU_LOOP, "TryDriveGpu dirty reduction")
    cpp = replace_once(cpp, OLD_DRIVE_PREFIX, NEW_DRIVE_PREFIX, "driveBody dirty reduction")
    hdr = replace_once(
        hdr,
        "    void updateConstant(const int& lk);",
        "    bool updateConstant(const int& lk);",
        "Nodal.h signature",
    )

    cpp_path.write_text(cpp, encoding="utf-8")
    h_path.write_text(hdr, encoding="utf-8")
    print("applied nodal constant refactor")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    apply(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
