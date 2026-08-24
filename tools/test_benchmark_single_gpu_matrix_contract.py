#!/usr/bin/env python3
"""Import-safe contract tests for benchmark_single_gpu_matrix.py; no GPU required."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("benchmark_single_gpu_matrix.py")
SPEC = importlib.util.spec_from_file_location("benchmark_single_gpu_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def test_worker_specs() -> None:
    specs = matrix.parse_worker_specs("24,36,auto:1.5,legacy")
    assert [(s.label, s.host_workers, s.worker_factor) for s in specs] == [
        ("w24", "24", None),
        ("w36", "36", None),
        ("auto-x1p5", "auto", 1.5),
        ("legacy", "legacy", None),
    ]
    for invalid in ("", "0", "auto:0", "24,24", "not-a-worker"):
        try:
            matrix.parse_worker_specs(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid worker specification: {invalid!r}")


def test_private_outputs() -> None:
    command = [
        "./RASBERY",
        "--rasi",
        "deck0.json",
        "deck1.json",
        "--raso",
        "old/a.h5",
        "old/b.h5",
        "--batch-mode",
        "64",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "outputs"
        rewritten = matrix.private_output_command(command, out)
        assert matrix.values_after(rewritten, "--rasi") == ["deck0.json", "deck1.json"]
        assert matrix.values_after(rewritten, "--raso") == [str(out / "a.h5"), str(out / "b.h5")]
        assert matrix.values_after(rewritten, "--batch-mode") == ["64"]
        try:
            matrix.private_output_command(command, out)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing per-run output directory must be rejected")


def test_receipts_and_launch() -> None:
    lines = [
        matrix.PROFILE_PREFIX + json.dumps({"host_workers": 24}),
        matrix.OCCUPANCY_PREFIX + json.dumps({"mean_width": 18.5}),
        matrix.COUNTERS_PREFIX + json.dumps({"graph_fallbacks": 0}),
    ]
    assert matrix.parse_json_receipt(lines, matrix.PROFILE_PREFIX) == {"host_workers": 24}
    assert matrix.parse_json_receipt(lines, matrix.OCCUPANCY_PREFIX) == {"mean_width": 18.5}
    assert matrix.parse_json_receipt(lines, matrix.COUNTERS_PREFIX) == {"graph_fallbacks": 0}

    class Args:
        python = "python3"
        runner = Path("tools/run_single_gpu_batch.py")
        batch_width = 64
        gpu = "0"
        set_values = ["RASBERY_BATCH_WAIT_US=25"]

    spec = matrix.WorkerSpec("auto-x1p5", "auto", 1.5)
    command = matrix.command_for_run(Args(), spec, ["./RASBERY", "--batch-mode", "64"])
    assert command[:10] == [
        "python3",
        "tools/run_single_gpu_batch.py",
        "--batch-width",
        "64",
        "--gpu",
        "0",
        "--host-workers",
        "auto",
        "--worker-factor",
        "1.5",
    ]
    assert command[-3:] == ["./RASBERY", "--batch-mode", "64"]


def main() -> int:
    test_worker_specs()
    test_private_outputs()
    test_receipts_and_launch()
    print("test_benchmark_single_gpu_matrix_contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
