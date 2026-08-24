#!/usr/bin/env python3
"""Static contract for the exact-base one-GPU source patch."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

PATCHER = Path(__file__).resolve().with_name("apply_single_gpu_safe_patches.py")
spec = importlib.util.spec_from_file_location("safe_patch", PATCHER)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

source = PATCHER.read_text(encoding="utf-8")
for token in (
    "parallel_wiel_finalize",
    "wiel_finalize_launch_width",
    "remove_eps_stream_drain",
    "auto-visible-cpus",
    "legacy-live-slots",
    "batch_host_policy_receipt",
):
    if token not in source:
        raise SystemExit(f"single-GPU safe source patch: FAIL missing {token}")

if module.blob_sha(b"abc") != "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f":
    raise SystemExit("single-GPU safe source patch: FAIL Git blob hash contract")

try:
    module.one("abc", "x", "y", "negative")
except RuntimeError:
    pass
else:
    raise SystemExit("single-GPU safe source patch: FAIL missing-block rejection")

if module.one("abc", "b", "B", "positive") != "aBc":
    raise SystemExit("single-GPU safe source patch: FAIL exact replacement")

print("single-GPU safe source patch: PASS")
