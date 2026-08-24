#!/usr/bin/env python3
"""Interleaved one-GPU RASBERY host-worker throughput matrix.

Only the host Driver pool changes.  GPU id, CUDA arena width, inputs and
physics options remain fixed.  Each trial receives private --raso paths and a
full log; runs with missing telemetry, graph fallbacks or no graph launches are
invalid.  Results are written to runs.csv and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROFILE = "[RASBERY][SINGLE_GPU_PROFILE] "
OCC = "[RASBERY][CUDA][BATCH_OCCUPANCY] "
COUNTERS = "[RASBERY][CUDA][BACKEND_COUNTERS] "


def values_after(cmd: Sequence[str], flag: str) -> list[str]:
    try:
        i = cmd.index(flag) + 1
    except ValueError:
        return []
    j = i
    while j < len(cmd) and not cmd[j].startswith("--"):
        j += 1
    return list(cmd[i:j])


def span(cmd: Sequence[str], flag: str) -> tuple[int, int] | None:
    try:
        i = cmd.index(flag) + 1
    except ValueError:
        return None
    j = i
    while j < len(cmd) and not cmd[j].startswith("--"):
        j += 1
    return i, j


def parse_specs(text: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    labels: set[str] = set()
    for token in text.split(","):
        raw = token.strip().lower()
        if raw == "legacy":
            spec = {"label": "legacy", "workers": "legacy", "factor": None}
        elif raw == "auto":
            spec = {"label": "auto_x1", "workers": "auto", "factor": 1.0}
        elif raw.startswith("auto:"):
            factor = float(raw.split(":", 1)[1])
            if factor <= 0:
                raise ValueError("auto worker factor must be positive")
            spec = {"label": f"auto_x{factor:g}", "workers": "auto", "factor": factor}
        else:
            count = int(raw)
            if count <= 0:
                raise ValueError("worker count must be positive")
            spec = {"label": f"w{count}", "workers": str(count), "factor": None}
        if spec["label"] in labels:
            raise ValueError("worker matrix contains duplicate configurations")
        labels.add(spec["label"])
        specs.append(spec)
    if not specs:
        raise ValueError("worker matrix is empty")
    return specs


def parse_json_line(text: str, prefix: str) -> dict[str, Any] | None:
    found = None
    for line in text.splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                found = value
    return found


def private_outputs(cmd: Sequence[str], run_dir: Path) -> list[str]:
    result = list(cmd)
    loc = span(result, "--raso")
    if loc is None:
        raise ValueError("RASBERY_command must include --raso")
    i, j = loc
    original = result[i:j]
    if not original:
        raise ValueError("--raso must contain outputs")
    run_dir.mkdir(parents=True, exist_ok=True)
    result[i:j] = [str(run_dir / f"{n:03d}_{Path(p).stem}{Path(p).suffix or '.h5'}")
                   for n, p in enumerate(original)]
    return result


def wrapper_command(args: argparse.Namespace, spec: dict[str, Any], command: Sequence[str]) -> list[str]:
    wrapper = Path(__file__).resolve().with_name("run_single_gpu_batch.py")
    result = [sys.executable, str(wrapper), "--batch-width", str(args.batch_width),
              "--gpu", args.gpu, "--host-workers", spec["workers"]]
    if spec["factor"] is not None:
        result += ["--worker-factor", f"{spec['factor']:g}"]
    for item in args.set_values:
        result += ["--set", item]
    return result + ["--"] + list(command)


def run_trial(args: argparse.Namespace, spec: dict[str, Any], phase: str, repeat: int,
              jobs: int, command: Sequence[str], root: Path) -> dict[str, Any]:
    name = f"{phase}_{repeat:02d}_{spec['label']}"
    out_dir = root / "outputs" / name
    log_path = root / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = wrapper_command(args, spec, private_outputs(command, out_dir))
    if args.dry_run:
        output = ""
        rc, wall = 0, 0.0
        log_path.write_text(" ".join(json.dumps(x) for x in cmd) + "\n", encoding="utf-8")
    else:
        started = time.perf_counter()
        try:
            done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=args.timeout_s or None, check=False)
            output, rc = done.stdout, done.returncode
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            output += f"\n[matrix] timeout after {args.timeout_s:g}s\n"
            rc = 124
        wall = time.perf_counter() - started
        log_path.write_text(output, encoding="utf-8")

    profile = parse_json_line(output, PROFILE)
    occ = parse_json_line(output, OCC)
    counters = parse_json_line(output, COUNTERS)
    def integer(data: dict[str, Any] | None, key: str) -> int | None:
        try:
            return int(data[key]) if data is not None else None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
    def number(data: dict[str, Any] | None, key: str) -> float | None:
        try:
            return float(data[key]) if data is not None else None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    graph_fallbacks = integer(counters, "graph_fallbacks")
    graph_launches = integer(counters, "graph_launches")
    launches = integer(occ, "launches")
    telemetry_missing = profile is None or occ is None or counters is None
    valid = (not args.dry_run and rc == 0 and not telemetry_missing and
             graph_fallbacks == 0 and graph_launches is not None and graph_launches > 0 and
             launches is not None and launches > 0)
    return {
        "worker_label": spec["label"], "requested_host_workers": spec["workers"],
        "requested_worker_factor": spec["factor"],
        "actual_host_workers": integer(profile, "host_workers"),
        "phase": phase, "repeat": repeat, "jobs": jobs, "returncode": rc,
        "wall_s": wall, "cases_per_hour": jobs * 3600.0 / wall if wall > 0 else 0.0,
        "valid": valid, "telemetry_missing": telemetry_missing,
        "graph_fallbacks": graph_fallbacks, "graph_launches": graph_launches,
        "stream_sync_calls": integer(counters, "stream_sync_calls_during_iteration"),
        "bulk_h2d_bytes": integer(counters, "bulk_h2d_bytes_during_iteration"),
        "mean_batch_width": number(occ, "mean_width"), "launches": launches,
        "log_path": str(log_path), "output_dir": str(out_dir),
    }


def summarize(rows: Sequence[dict[str, Any]], specs: Sequence[dict[str, Any]],
              master: float, gate: float) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for spec in specs:
        all_rows = [r for r in rows if r["phase"] == "run" and r["worker_label"] == spec["label"]]
        valid = [r for r in all_rows if r["valid"]]
        item: dict[str, Any] = {
            "worker_label": spec["label"], "requested_host_workers": spec["workers"],
            "requested_worker_factor": spec["factor"], "valid_runs": len(valid),
            "total_runs": len(all_rows), "all_valid": bool(all_rows) and len(valid) == len(all_rows),
        }
        if valid:
            rate = statistics.median(r["cases_per_hour"] for r in valid)
            item.update({
                "median_wall_s": statistics.median(r["wall_s"] for r in valid),
                "median_cases_per_hour": rate, "median_speedup_vs_master": rate / master,
                "meets_throughput_gate": rate >= gate,
                "median_mean_batch_width": statistics.median(
                    r["mean_batch_width"] for r in valid if r["mean_batch_width"] is not None)
                    if any(r["mean_batch_width"] is not None for r in valid) else None,
                "actual_host_workers": sorted({r["actual_host_workers"] for r in valid
                                                if r["actual_host_workers"] is not None}),
            })
        configs.append(item)
    candidates = [x for x in configs if x.get("all_valid") and x.get("median_cases_per_hour") is not None]
    best = max(candidates, key=lambda x: x["median_cases_per_hour"]) if candidates else None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_cases_per_hour": master, "throughput_gate_cases_per_hour": gate,
        "gate_passed": bool(best) and bool(best.get("meets_throughput_gate")),
        "configurations": configs, "best_valid_configuration": best,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-GPU RASBERY host-worker throughput matrix")
    p.add_argument("--batch-width", type=int, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--workers", default="24,36,48,64")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout-s", type=float, default=0.0)
    p.add_argument("--master-cases-per-hour", type=float, default=218.0)
    p.add_argument("--throughput-gate", type=float, default=218.0)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd.pop(0)
    if (not cmd or args.batch_width <= 0 or args.warmups < 0 or args.repeats <= 0 or
            args.timeout_s < 0 or args.master_cases_per_hour <= 0 or args.throughput_gate <= 0):
        print("error: invalid command or numeric option", file=sys.stderr)
        return 2
    inputs, outputs = values_after(cmd, "--rasi"), values_after(cmd, "--raso")
    if not inputs or len(inputs) != len(outputs):
        print("error: equal non-empty --rasi/--raso lists are required", file=sys.stderr)
        return 2
    try:
        specs = parse_specs(args.workers)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.output_root or Path(f"single_gpu_matrix_{stamp}")).resolve()
    if root.exists() and any(root.iterdir()):
        print(f"error: output root is not empty: {root}", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "batch_width": args.batch_width, "gpu": args.gpu, "jobs": len(inputs),
                "workers": specs, "warmups": args.warmups, "repeats": args.repeats,
                "command": cmd, "set_values": args.set_values}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for phase, rounds in (("warmup", args.warmups), ("run", args.repeats)):
        for repeat in range(rounds):
            for spec in specs:  # interleave configurations to reduce clock/temperature order bias
                print(f"[matrix] start {phase}_{repeat:02d}_{spec['label']}", flush=True)
                rows.append(run_trial(args, spec, phase, repeat, len(inputs), cmd, root))
                with (root / "runs.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
    result = summarize(rows, specs, args.master_cases_per_hour, args.throughput_gate)
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("[matrix] summary " + json.dumps(result, separators=(",", ":")), flush=True)
    if args.dry_run:
        return 0
    measured = [r for r in rows if r["phase"] == "run"]
    return 0 if measured and all(r["valid"] for r in measured) and result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
