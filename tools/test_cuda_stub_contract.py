#!/usr/bin/env python3
"""Ensure every batch bridge method has a non-CUDA stub definition."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = (ROOT / "src" / "CudaBICGBackendStub.cpp").read_text(encoding="utf-8-sig")
required = (
    "bool CudaBatchArena::nodalDeviceBridge(int, NodalDeviceBridge&) const",
    "void CudaBatchArena::markNodalDhatCurrent(int)",
    "void CudaBatchArena::invalidateNodalDhatCurrent(int)",
)
missing = [token for token in required if token not in STUB]
if missing:
    raise SystemExit(f"CUDA stub contract: FAIL: missing {missing}")
print("CUDA stub contract: PASS")
