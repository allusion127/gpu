#!/usr/bin/env python3
"""Contract tests for tools/run_single_gpu_batch.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_single_gpu_batch.py"
spec = importlib.util.spec_from_file_location("run_single_gpu_batch", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert module.values_after(["x", "--rasi", "a", "b", "--raso", "oa", "ob"], "--rasi") == ["a", "b"]
assert module.batch_width_from_command(["x", "--batch-mode", "64"]) == 64
assert module.compute_host_workers("auto", batch_width=64, jobs=64, visible_cpus=24, worker_factor=1.0) == (24, "auto_cpu_x1")
assert module.compute_host_workers("auto", batch_width=64, jobs=64, visible_cpus=24, worker_factor=1.5) == (36, "auto_cpu_x1.5")
assert module.compute_host_workers("legacy", batch_width=64, jobs=64, visible_cpus=24, worker_factor=1.0)[0] == 64
assert module.compute_host_workers("16", batch_width=64, jobs=64, visible_cpus=24, worker_factor=1.0)[0] == 16
assert module.parse_overrides(["RASBERY_BATCH_WAIT_US=25", "FOO=bar"]) == {"RASBERY_BATCH_WAIT_US": "25", "FOO": "bar"}

args = SimpleNamespace(
    batch_width=4,
    gpu="0",
    host_workers="auto",
    worker_factor=1.0,
    set_values=[],
)
old_visible = module.visible_cpu_threads
module.visible_cpu_threads = lambda: 2
try:
    plan, command, env = module.build_plan(
        args,
        ["RASBERY", "--rasi", "a", "b", "c", "d", "--raso", "oa", "ob", "oc", "od"],
    )
finally:
    module.visible_cpu_threads = old_visible
assert command[-2:] == ["--batch-mode", "4"]
assert plan.host_workers == 2
assert env["CUDA_VISIBLE_DEVICES"] == "0"
assert env["RASBERY_BATCH_HOST_THREADS"] == "2"
assert env["RASBERY_GPU_CMFD_SWEEP"] == "1"
assert env["RASBERY_GPU_NODAL_FULL"] == "1"
print("single gpu batch profile: PASS")
