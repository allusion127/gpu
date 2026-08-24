#!/usr/bin/env python3
"""Regression gate: the nodal XS mirror launch path allocates no batch container."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "CudaXsReconBackend.cu"


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    required = (
        "// RASBERY_NODAL_XS_MIRROR_NO_BATCH_ALLOCATION",
        "bool pushed_xsrf = false, pushed_xsnf = false, pushed_xssm = false;",
        "sl.pushed_xsrf = _mirror_xs && push_xsrf;",
        "sl.pushed_xsnf = _mirror_xs && push_xsnf;",
        "sl.pushed_xssm = _mirror_xs && push_xssm;",
        "if (sl.pushed_xsrf) sl.xsrf_mirror.commit",
        "if (sl.pushed_xsnf) sl.xsnf_mirror.commit",
        "if (sl.pushed_xssm) sl.xssm_mirror.commit",
        "sl.pushed_xsrf = sl.pushed_xsnf = sl.pushed_xssm = false;",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise SystemExit(f"nodal XS mirror no-allocation: FAIL missing={missing}")

    forbidden = (
        "PendingXsMirror",
        "std::vector<PendingXsMirror>",
        "pending_xs.reserve",
        "pending_xs.push_back",
    )
    stale = [token for token in forbidden if token in source]
    if stale:
        raise SystemExit(f"nodal XS mirror no-allocation: FAIL stale={stale}")

    sync = source.find("const cudaError_t src = cudaStreamSynchronize(_stream);")
    commit = source.find("if (sl.pushed_xsrf) sl.xsrf_mirror.commit", sync)
    if sync < 0 or commit < sync:
        raise SystemExit("nodal XS mirror no-allocation: FAIL commit precedes drain")

    print("nodal XS mirror no-allocation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
