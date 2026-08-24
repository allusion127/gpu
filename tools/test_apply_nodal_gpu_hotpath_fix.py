#!/usr/bin/env python3
"""Test the allocation-free normalization after the main nodal transformer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main_transform = load("nodal_main_transform", ROOT / "tools" / "apply_nodal_gpu_refactor.py")
hotfix = load("nodal_hotfix", ROOT / "tools" / "apply_nodal_gpu_hotpath_fix.py")
fixture_mod = load("nodal_fixture", ROOT / "tools" / "test_apply_nodal_gpu_refactor.py")


def main() -> int:
    v1 = main_transform.apply(fixture_mod.fixture())
    assert "PendingXsMirror" in v1
    patched = hotfix.apply(v1)
    assert hotfix.HOTPATH_MARKER in patched
    assert "PendingXsMirror" not in patched
    assert "std::vector<PendingXsMirror>" not in patched
    assert patched == hotfix.apply(patched), "hot-path normalization must be idempotent"

    try:
        hotfix.apply(v1.replace("pending_xs.reserve(part.size());", "pending_xs.reserve(1);"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("hot-path anchor drift did not fail closed")

    print("nodal XS mirror hot-path transformer: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
