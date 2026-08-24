#!/usr/bin/env python3
"""Launch one-GPU RASBERY multi-input runs with an explicit CPU/GPU work split.

Example:
  python tools/run_single_gpu_batch.py --batch-width 64 --gpu 0 -- \
      ./RASBERY --rasi deck0.json deck1.json ... \
                --raso out0.h5 out1.h5 ... --batch-mode 64

The CUDA arena width and the number of host Driver workers are deliberately
separate.  The arena can keep 64 slots while a 24-thread host runs 24 Drivers;
the OpenMP dynamic queue feeds the remaining decks as workers finish.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


DEFAULT_ENV = {
    "RASBERY_GPU": "1",
    "RASBERY_GPU_RB_SWEEPS": "4",
    "RASBERY_GPU_XSRECON": "1",
    "RASBERY_GPU_FLATXS": "1",
    "RASBERY_GPU_NODAL": "1",
    "RASBERY_GPU_NODAL_FULL": "1",
    "RASBERY_GPU_CMFD_SWEEP": "1",
    "RASBERY_PPR_MODE": "master",
    "RASBERY_PC_MODE": "decart",
    # Fixed waiting was neutral/slower on the measured M64 campaign.  Start
    # with opportunistic batching; benchmark non-zero values explicitly.
    "RASBERY_BATCH_WAIT_US": "0",
    "RASBERY_NODAL_BATCH_WAIT_US": "0",
    "OMP_WAIT_POLICY": "PASSIVE",
    "GOMP_SPINCOUNT": "0",
    "OMP_PROC_BIND": "TRUE",
    "OMP_PLACES": "cores",
}


@dataclass(frozen=True)
class LaunchPlan:
    batch_width: int
    jobs: int
    visible_cpus: int
    host_workers: int
    worker_policy: str
    gpu: str


def visible_cpu_threads() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def values_after(command: Sequence[str], flag: str) -> list[str]:
    """Return values following *flag* until the next long option."""
    try:
        start = command.index(flag) + 1
    except ValueError:
        return []
    values: list[str] = []
    for token in command[start:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def batch_width_from_command(command: Sequence[str]) -> int | None:
    values = values_after(command, "--batch-mode")
    if not values:
        return None
    try:
        value = int(values[0])
    except ValueError as exc:
        raise ValueError(f"invalid --batch-mode value: {values[0]!r}") from exc
    if value <= 0:
        raise ValueError("--batch-mode must be positive")
    return value


def compute_host_workers(
    request: str,
    *,
    batch_width: int,
    jobs: int,
    visible_cpus: int,
    worker_factor: float,
) -> tuple[int, str]:
    cap = max(1, min(batch_width, jobs))
    normalized = request.strip().lower()
    if normalized == "legacy":
        return cap, "legacy_one_worker_per_live_slot"
    if normalized == "auto":
        if worker_factor <= 0.0:
            raise ValueError("--worker-factor must be positive")
        cpu_budget = max(1, int(round(visible_cpus * worker_factor)))
        return min(cap, cpu_budget), f"auto_cpu_x{worker_factor:g}"
    try:
        explicit = int(request)
    except ValueError as exc:
        raise ValueError("--host-workers must be auto, legacy, or a positive integer") from exc
    if explicit <= 0:
        raise ValueError("--host-workers must be positive")
    return min(cap, explicit), "explicit"


def parse_overrides(items: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"invalid environment key: {key!r}")
        result[key] = value
    return result


def build_plan(args: argparse.Namespace, command: list[str]) -> tuple[LaunchPlan, list[str], dict[str, str]]:
    command_width = batch_width_from_command(command)
    batch_width = args.batch_width or command_width
    if batch_width is None:
        raise ValueError("provide --batch-width or include --batch-mode M in the RASBERY command")
    if command_width is not None and command_width != batch_width:
        raise ValueError(
            f"--batch-width {batch_width} disagrees with command --batch-mode {command_width}"
        )
    if command_width is None:
        command.extend(["--batch-mode", str(batch_width)])

    rasi = values_after(command, "--rasi")
    raso = values_after(command, "--raso")
    if rasi and raso and len(rasi) != len(raso):
        raise ValueError(f"--rasi count {len(rasi)} does not match --raso count {len(raso)}")
    jobs = len(rasi) if rasi else batch_width
    cpus = visible_cpu_threads()
    workers, policy = compute_host_workers(
        args.host_workers,
        batch_width=batch_width,
        jobs=jobs,
        visible_cpus=cpus,
        worker_factor=args.worker_factor,
    )

    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["RASBERY_BATCH_HOST_THREADS"] = str(workers)
    env.update(parse_overrides(args.set_values))

    plan = LaunchPlan(
        batch_width=batch_width,
        jobs=jobs,
        visible_cpus=cpus,
        host_workers=workers,
        worker_policy=policy,
        gpu=str(args.gpu),
    )
    return plan, command, env


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a same-geometry RASBERY deck set on one GPU with a bounded host worker pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-width", type=int, help="CUDA arena width; inferred from --batch-mode when omitted")
    p.add_argument("--gpu", default="0", help="physical GPU index exposed as the process's sole CUDA device")
    p.add_argument(
        "--host-workers",
        default="auto",
        help="auto, legacy, or an explicit count; auto uses visible CPUs times --worker-factor",
    )
    p.add_argument(
        "--worker-factor",
        type=float,
        default=1.0,
        help="auto-mode host oversubscription factor; benchmark 1.0, 1.5 and 2.0 on the target host",
    )
    p.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE", help="override one profile environment variable")
    p.add_argument("--dry-run", action="store_true", help="print the plan without executing RASBERY")
    p.add_argument("command", nargs=argparse.REMAINDER, help="RASBERY executable and arguments, preceded by --")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("error: missing RASBERY command after --", file=sys.stderr)
        return 2
    if args.batch_width is not None and args.batch_width <= 0:
        print("error: --batch-width must be positive", file=sys.stderr)
        return 2

    try:
        plan, command, env = build_plan(args, command)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    receipt = {
        "gpu": plan.gpu,
        "jobs": plan.jobs,
        "arena_width": plan.batch_width,
        "visible_cpus": plan.visible_cpus,
        "host_workers": plan.host_workers,
        "worker_policy": plan.worker_policy,
        "cmfd_wait_us": env.get("RASBERY_BATCH_WAIT_US"),
        "nodal_wait_us": env.get("RASBERY_NODAL_BATCH_WAIT_US"),
        "command": command,
    }
    print("[RASBERY][SINGLE_GPU_PROFILE] " + json.dumps(receipt, separators=(",", ":")))
    if args.dry_run:
        return 0
    try:
        return subprocess.run(command, env=env, check=False).returncode
    except OSError as exc:
        print(f"error: failed to execute {command[0]!r}: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
