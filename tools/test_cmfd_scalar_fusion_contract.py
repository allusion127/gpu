#!/usr/bin/env python3
"""Static dependency/order contract for BiCGSTAB scalar-node fusion."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src" / "CudaBICGBackend.cu").read_text(encoding="utf-8-sig")


def fail(message: str) -> None:
    raise SystemExit(f"cmfd scalar fusion contract: FAIL: {message}")


required = (
    'bool cmfdScalarFusionEnabled()',
    'RASBERY_GPU_CMFD_SCALAR_FUSION',
    '__global__ void reduce_norm_store_reference_stage2',
    '__global__ void reduce_norm_accumulate_stage2',
    'for (int i = 0; i < blocks; ++i) sum += pm[i];',
    'sm[kInitialNorm] = norm;',
    'sm[kR20] = norm;',
    'accumulate_iteration_active(',
    'if (active[m] == 0u) return;',
    '++cm[kOverrunCount];',
    'if (scalar_fusion)',
    'store_reference_norm<<<',
    'accumulate_iteration<<<',
)
missing = [token for token in required if token not in SRC]
if missing:
    fail(f"missing {missing}")

# The fused residual finalizer must be after update_solution and before the
# next captured iteration; the separate rollback path must remain compiled.
update = SRC.find('update_solution<<<')
fused = SRC.find('reduce_norm_accumulate_stage2<<<', update)
rollback = SRC.find('accumulate_iteration<<<', update)
if not (0 <= update < fused and rollback >= 0):
    fail("iteration fusion/rollback ordering is incomplete")

# The initial-reference fusion must follow stage1 and replace only stage2.
outer = SRC.find('void enqueue_outer(int nmax)')
stage1 = SRC.find('reduce_dot_stage1<<<', outer)
reference = SRC.find('reduce_norm_store_reference_stage2<<<', stage1)
if not (0 <= outer < stage1 < reference):
    fail("initial norm strict stage1 -> fused stage2 order missing")

print("cmfd scalar fusion contract: PASS")
