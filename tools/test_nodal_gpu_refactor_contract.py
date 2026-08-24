#!/usr/bin/env python3
"""Static safety contract for the nodal CUDA refactor."""
from __future__ import annotations

import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"nodal GPU refactor contract: FAIL: {message}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    source_path = Path(args[0]) if args else Path(__file__).resolve().parents[1] / "src" / "CudaXsReconBackend.cu"
    source = source_path.read_text(encoding="utf-8")

    required = (
        '#include "CudaTransferMirror.h"',
        '// RASBERY_NODAL_REFACTOR_V1',
        '// RASBERY_NODAL_XS_MIRROR_NO_BATCH_ALLOCATION',
        'bool nodalFuseMatEvenEnabled()',
        'bool nodalXsMirrorEnabled()',
        '__global__ void kNodalMatEven',
        'ndl::nodalUpdateMatrix(v, lk, forms);',
        'ndl::nodalCalculateEven(v, lk, forms);',
        'RASBERY_GPU_NODAL_FUSE_MAT_EVEN',
        'RASBERY_GPU_NODAL_XS_MIRROR',
        'cuda_transfer::ByteExactMirror<double> xsrf_mirror;',
        'cuda_transfer::ByteExactMirror<double> xsnf_mirror;',
        'cuda_transfer::ByteExactMirror<double> xssm_mirror;',
        'bool pushed_xsrf = false',
        'sl.pushed_xsrf = _mirror_xs && push_xsrf;',
        'xs_h2d_skipped_bytes',
        '\\"mat_even_fused\\"',
        '\\"xs_h2d_bytes\\"',
        '\\"xs_h2d_skipped_bytes\\"',
    )
    missing = [token for token in required if token not in source]
    if missing:
        fail(f"missing tokens: {missing}")

    trl0 = source.find('kNodalTrl0<true>')
    trl12 = source.find('kNodalTrl12<true>', trl0)
    mat = source.find('kNodalMatEven<true>', trl12)
    jnet = source.find('kNodalJnet<true>', mat)
    if not (0 <= trl0 < trl12 < mat < jnet):
        fail("batched phase order Trl0 -> Trl12 -> MatEven -> Jnet was not preserved")

    for token in ('kNodalMat<true>', 'kNodalEven<true>',
                  'kNodalMat<false>', 'kNodalEven<false>'):
        if token not in source:
            fail(f"rollback token removed: {token}")

    sync = source.find('const cudaError_t src = cudaStreamSynchronize(_stream);')
    commit = source.find('sl.xsrf_mirror.commit', sync)
    if sync < 0 or commit < sync:
        fail("mirror commit is not after a successful stream synchronize")

    for token in ('PendingXsMirror', 'pending_xs.reserve', 'pending_xs.push_back'):
        if token in source:
            fail(f"per-batch heap-allocation token remains: {token}")

    unconditional = source.find('xsrf/xsnf/xssm upload UNCONDITIONALLY')
    if unconditional >= 0:
        fail("stale unconditional XS upload path remains")

    print("nodal GPU refactor contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
