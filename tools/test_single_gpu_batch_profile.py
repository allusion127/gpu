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

# --- job namespace policy (plan Rev.4 Sec 7) --------------------------------
# Same input file for many decks: allowed.  Same output path: forbidden.
assert module.validate_deck_paths(
    ["RASBERY", "--rasi", "deck.json", "deck.json", "--raso", "o0.h5", "o1.h5"]
) == ["deck.json", "deck.json"]
for bad in (
    ["RASBERY", "--rasi", "a.json", "b.json", "--raso", "o0.h5", "o0.h5"],
    ["RASBERY", "--rasi", "a.json", "b.json", "--raso", "o0.h5", "./o0.h5"],
    ["RASBERY", "--rasi", "a.json", "b.json"],
):
    try:
        module.validate_deck_paths(bad)
    except ValueError:
        pass
    else:
        raise SystemExit(f"single gpu batch profile: FAIL accepted {bad}")
# The key canonicalises, so `out/../out/x.h5` and `out/x.h5` are one namespace.
assert module.path_key("out/x.h5") == module.path_key("out/../out/x.h5")

# --- the C++ side of the same policy ----------------------------------------
main_cpp = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8")
driver_h = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8")
io_h = (ROOT / "src" / "IO.h").read_text(encoding="utf-8")
for token, text, name in (
    ("weakly_canonical", main_cpp, "main.cpp"),
    ("rasberyPathKey", main_cpp, "main.cpp"),
    ("--raso paths must be distinct", main_cpp, "main.cpp"),
    ("std::string result_dir() const", io_h, "IO.h"),
    ("std::string result_stem() const", io_h, "IO.h"),
    ("RestartPath(input_output, step_number)", driver_h, "Driver.h"),
    ("RASBERY_RESTART_AT_INPUT", driver_h, "Driver.h"),
    ("{}_restart_{}.h5", driver_h, "Driver.h"),
):
    if token not in text:
        raise SystemExit(f"single gpu batch profile: FAIL {name} missing {token!r}")

print("single gpu batch profile: PASS")
