#!/usr/bin/env python3
"""Contract tests for tools/benchmark_single_gpu_matrix.py."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "benchmark_single_gpu_matrix.py"
spec = importlib.util.spec_from_file_location("benchmark_single_gpu_matrix", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

specs = module.parse_specs("24,auto:1.5,legacy")
assert [(s["label"], s["workers"], s["factor"]) for s in specs] == [
    ("w24", "24", None), ("auto_x1.5", "auto", 1.5), ("legacy", "legacy", None),
]
assert module.values_after(
    ["RASBERY", "--rasi", "a", "b", "--raso", "oa", "ob", "--batch-mode", "2"],
    "--rasi",
) == ["a", "b"]

scratch = ROOT / ".matrix_contract_tmp"
rewritten = module.private_outputs(
    ["RASBERY", "--rasi", "a", "b", "--raso", "oa.h5", "dir/ob.h5", "--batch-mode", "2"],
    scratch,
)
assert module.values_after(rewritten, "--raso") == [
    str(scratch / "000_oa.h5"), str(scratch / "001_ob.h5"),
]
line = "noise\n" + module.COUNTERS + json.dumps({"graph_fallbacks": 0, "graph_launches": 12}) + "\n"
assert module.parse_json_line(line, module.COUNTERS) == {"graph_fallbacks": 0, "graph_launches": 12}

args = argparse.Namespace(batch_width=64, gpu="0", set_values=["RASBERY_BATCH_WAIT_US=25"])
cmd = module.wrapper_command(args, specs[1], ["./RASBERY", "--rasi", "a", "--raso", "o"])
assert "--worker-factor" in cmd and "1.5" in cmd
assert cmd[-5:] == ["./RASBERY", "--rasi", "a", "--raso", "o"]

row = {
    "worker_label": "w24", "requested_host_workers": "24", "requested_worker_factor": None,
    "actual_host_workers": 24, "phase": "run", "repeat": 0, "jobs": 64,
    "returncode": 0, "wall_s": 1000.0, "cases_per_hour": 230.4, "valid": True,
    "telemetry_missing": False, "graph_fallbacks": 0, "graph_launches": 100,
    "stream_sync_calls": 10, "bulk_h2d_bytes": 123, "mean_batch_width": 22.0,
    "launches": 10, "log_path": "run.log", "output_dir": "out",
}
summary = module.summarize([row], [specs[0]], 218.0, 218.0)
assert summary["gate_passed"] is True
assert summary["best_valid_configuration"]["median_speedup_vs_master"] > 1.0

for child in scratch.glob("*"):
    child.unlink()
try:
    scratch.rmdir()
except OSError:
    pass
print("single gpu benchmark matrix: PASS")
