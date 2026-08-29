#!/usr/bin/env python3
"""Run one refill batch per GPU off a shared job queue (plan Rev.7.1 Sec 13.3).

    python tools/run_multi_gpu_batch.py --gpus 0 --batch-width 4 \
        --jobs manifest.txt -- ./RASBERY

    python tools/run_multi_gpu_batch.py --gpus 0,1,2,3 --batch-width 64 \
        --jobs manifest.txt --pin numactl -- ./build/RASBERY

WHY PROCESS-PER-GPU AND NOT ONE PROCESS WITH FOUR CONTEXTS.  Every arena in
this codebase -- CMFD, nodal, physics -- is a process-lifetime singleton sized
once for width M against the current device, and the rendezvous elects one
launcher for one stream.  Making them device-aware is a rewrite of the batch
core; giving each GPU its own process is a launcher.  CUDA_VISIBLE_DEVICES makes
each process believe it owns device 0, so nothing in the C++ has to change and
the per-GPU numbers stay directly comparable to the single-GPU harness.

WHY A SHARED QUEUE AND NOT A STATIC SPLIT.  A static split finishes when its
SLOWEST partition finishes, and the partitions are not equal: decks differ in
statepoint count, and on a mixed host the GPUs differ too.  The queue is a
manifest plus a claim file, and a claim is one flock'd read-modify-write.  Each
GPU runs a SEQUENCE of RASBERY processes, claiming a chunk at a time -- so a GPU
that runs dry steals from what is left instead of idling, and a GPU that is slow
simply claims less.  `--claim all` degenerates to the static split.

WHY CHUNKS AND NOT ONE JOB AT A TIME.  A process pays arena stand-up and graph
capture once, and Task 20's in-process refill is what makes a chunk cheap: the
chunk is admitted into M slots and refilled from inside the process, so the
claim granularity only has to be coarse enough to amortise startup.  `auto`
halves the remaining queue across the GPUs each round, which ends at chunks of
`--batch-width`.

The rc/FAIL accounting, the exact-only physics-mode audit and the graph-fallback
audit are IMPORTED from run_single_gpu_batch.py rather than restated, so a
change to what counts as a valid run cannot apply to one harness and not the
other.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_single_gpu_batch import (  # noqa: E402
    DEFAULT_ENV,
    LaunchPlan,
    check_run_receipts,
    parse_overrides,
    path_key,
    visible_cpu_threads,
)

REFILL_RECEIPT = re.compile(r"\[RASBERY\]\[REFILL\]\s*(\{.*\})")
FAIL_LINE = re.compile(r"\[RASBERY\]\[FAIL\]")

# Plan Sec 13.3 host budget: "라이터 ~8스레드 + GPU당 로더 8~16".  The writer is
# ONE pool for the process (IoWriter.h), so its share is charged per process;
# the loader budget is the OpenMP Driver workers plus the per-Driver solver
# threads, which is what RASBERY_BATCH_HOST_THREADS and RASBERY_OMP_THREADS
# split between them.
WRITER_THREADS_PER_PROCESS = 8
LOADER_THREADS_PER_GPU_MIN = 8
LOADER_THREADS_PER_GPU_MAX = 16


# ---------------------------------------------------------------------------
# The shared queue
# ---------------------------------------------------------------------------


RESULT_MODES = ("full", "pin-off", "light")


def read_manifest(path: Path) -> list[tuple[str, str, str]]:
    """Parse the same `--jobs` manifest main.cpp reads.

    Kept deliberately in lockstep with rasberyReadJobManifest(): two paths per
    line plus an OPTIONAL third field (that job's result mode), `#` comments,
    `"` quoting, blank lines skipped.  The dispatcher has to agree with the
    executable about what a job is, because it is the dispatcher that decides
    which jobs go to which GPU -- and, since it rewrites the chunk manifests, a
    third field it did not understand would be silently dropped and every
    promoted elite would come back as a screening receipt.

    A job with no third field carries the empty string and inherits --result.
    """
    jobs: list[tuple[str, str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            fields = shlex.split(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if len(fields) not in (2, 3):
            raise ValueError(
                f"{path}:{lineno}: expected `<input.json> <output.h5> [result-mode]`, "
                f"got {len(fields)} field(s)"
            )
        mode = fields[2] if len(fields) == 3 else ""
        if mode and mode not in RESULT_MODES:
            raise ValueError(
                f"{path}:{lineno}: third field must be a result mode "
                f"{'|'.join(RESULT_MODES)}, got {mode!r}"
            )
        jobs.append((fields[0], fields[1], mode))
    if not jobs:
        raise ValueError(f"{path}: no jobs")
    seen: dict[str, int] = {}
    for index, (_, output, _mode) in enumerate(jobs):
        key = path_key(output)
        if key in seen:
            raise ValueError(
                "%s: output paths must be distinct, one per job; line %d and line %d "
                "resolve to the same file. Two Drivers would race inside one HDF5 file "
                "and share a restart namespace." % (path, seen[key] + 1, index + 1)
            )
        seen[key] = index
    return jobs


class Queue:
    """A claim cursor over the manifest, shared by the per-GPU workers.

    flock, not a lock-free trick: the workers are threads in THIS process today,
    but the file lock means a second dispatcher (or an operator's manual run)
    against the same queue file claims correctly too, which is what makes it
    safe to restart a half-finished campaign.
    """

    def __init__(self, state_path: Path, total: int) -> None:
        self._path = state_path
        self._total = total
        self._path.write_text(json.dumps({"claimed": 0, "total": total}), encoding="utf-8")

    def claim(self, size: int) -> tuple[int, int]:
        """Reserve up to *size* jobs; returns [start, end) into the manifest."""
        with open(self._path, "r+", encoding="utf-8") as handle:
            _lock(handle)
            try:
                state = json.load(handle)
                start = int(state["claimed"])
                end = min(self._total, start + max(1, size))
                state["claimed"] = end
                handle.seek(0)
                handle.truncate()
                json.dump(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                _unlock(handle)
        return start, end

    def remaining(self) -> int:
        with open(self._path, encoding="utf-8") as handle:
            _lock(handle)
            try:
                return self._total - int(json.load(handle)["claimed"])
            finally:
                _unlock(handle)


def _lock(handle) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except ImportError:  # Windows: the dispatcher is single-process there
        pass


def _unlock(handle) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Host budget
# ---------------------------------------------------------------------------


@dataclass
class HostBudget:
    visible_cpus: int
    gpus: int
    cpus_per_gpu: int
    driver_workers: int
    solver_threads: int
    writer_threads: int
    cpu_sets: list[list[int]]
    pin: str


def plan_host_budget(
    *,
    gpus: Sequence[str],
    batch_width: int,
    visible_cpus: int,
    pin: str,
    driver_workers: int | None,
    solver_threads: int | None,
) -> HostBudget:
    """Split the host between the GPU processes (plan Sec 13.3).

    The split is by CPU COUNT, not by preference: two processes that both think
    they own all 96 cores oversubscribe by 2x, and OMP_PROC_BIND=TRUE then pins
    their threads onto the same places.  That costs more than the GPU work it is
    feeding, and it is invisible in any per-GPU receipt.
    """
    n = len(gpus)
    cpus_per_gpu = max(1, visible_cpus // n)
    # The Driver workers are the refill lanes; capping them at the arena width
    # matches main.cpp, which will not run more Drivers than there are slots.
    workers = driver_workers if driver_workers else min(batch_width, cpus_per_gpu)
    workers = max(1, min(workers, batch_width))
    # What is left of this process's share, after the writer pool, is what the
    # per-Driver solver regions may spend.  At least 1: a zero would be a
    # silently serial solver.
    spare = max(0, cpus_per_gpu - workers - WRITER_THREADS_PER_PROCESS)
    threads = solver_threads if solver_threads else max(1, min(3, 1 + spare // max(1, workers)))

    sets: list[list[int]] = []
    if pin != "none":
        for index in range(n):
            lo = index * cpus_per_gpu
            sets.append(list(range(lo, lo + cpus_per_gpu)))
    return HostBudget(
        visible_cpus=visible_cpus,
        gpus=n,
        cpus_per_gpu=cpus_per_gpu,
        driver_workers=workers,
        solver_threads=threads,
        writer_threads=WRITER_THREADS_PER_PROCESS,
        cpu_sets=sets,
        pin=pin,
    )


def pin_prefix(budget: HostBudget, index: int) -> list[str]:
    if budget.pin == "none" or not budget.cpu_sets:
        return []
    cpus = budget.cpu_sets[index]
    if budget.pin == "numactl":
        return ["numactl", f"--physcpubind={cpus[0]}-{cpus[-1]}", "--localalloc"]
    return ["taskset", "-c", f"{cpus[0]}-{cpus[-1]}"]


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------


@dataclass
class GpuResult:
    gpu: str
    processes: int = 0
    jobs: int = 0
    returncode: int = 0
    fail_lines: int = 0
    problems: list[str] = field(default_factory=list)
    refill_receipts: list[dict] = field(default_factory=list)
    wall_s: float = 0.0


def run_gpu(
    *,
    gpu: str,
    index: int,
    queue: Queue,
    jobs: list[tuple[str, str, str]],
    budget: HostBudget,
    batch_width: int,
    claim: str,
    result_mode: str | None,
    executable: list[str],
    workdir: Path,
    cwd: Path | None,
    overrides: dict[str, str],
    dry_run: bool,
) -> GpuResult:
    result = GpuResult(gpu=gpu)
    started = time.monotonic()
    chunk_index = 0

    while True:
        if claim == "all":
            # Static split: each GPU takes its equal share once, and that is the
            # whole run.  Chunking it further would only add process restarts.
            size = -(-len(jobs) // budget.gpus)
        elif claim == "auto":
            remaining = queue.remaining()
            if remaining <= 0:
                break
            if budget.gpus == 1:
                # There is nobody to steal from.  Splitting the queue here would
                # DRAIN the batch between chunks, which is precisely the tail
                # Task 20 exists to remove -- so one GPU claims the lot and lets
                # the in-process refill do the work.
                size = remaining
            else:
                size = max(batch_width, -(-remaining // (2 * budget.gpus)))
        else:
            size = int(claim)

        start, end = queue.claim(size)
        if start >= end:
            break

        chunk_index += 1
        manifest = workdir / f"gpu{gpu}.chunk{chunk_index:04d}.txt"
        manifest.write_text(
            "".join(
                f'"{i}" "{o}"' + (f" {m}" if m else "") + "\n"
                for i, o, m in jobs[start:end]
            ),
            encoding="utf-8",
            newline="\n",
        )

        env = os.environ.copy()
        env.update(DEFAULT_ENV)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["RASBERY_BATCH_HOST_THREADS"] = str(budget.driver_workers)
        env["RASBERY_OMP_THREADS"] = str(budget.solver_threads)
        env.update(overrides)

        command = (
            pin_prefix(budget, index)
            + list(executable)
            + ["--jobs", str(manifest), "--batch-mode", str(batch_width)]
            + (["--result", result_mode] if result_mode else [])
        )
        log = workdir / f"gpu{gpu}.chunk{chunk_index:04d}.log"
        # The manifest carries absolute paths, but a deck's own references (its
        # cross-section HDF5, its restart namespace) are relative to the deck
        # directory, so the child has to run where the decks live.
        manifest_arg = str(manifest.resolve())
        command[command.index("--jobs") + 1] = manifest_arg

        if dry_run:
            print(f"[RASBERY][MULTI_GPU][DRY] gpu={gpu} jobs={end - start} " + " ".join(command))
            result.processes += 1
            result.jobs += end - start
            continue

        with open(log, "w", encoding="utf-8") as sink:
            child = subprocess.run(  # noqa: S603
                command, env=env, cwd=str(cwd) if cwd else None,
                stdout=sink, stderr=subprocess.STDOUT, check=False,
            )
        text = log.read_text(encoding="utf-8", errors="replace")

        result.processes += 1
        result.jobs += end - start
        if child.returncode != 0 and result.returncode == 0:
            result.returncode = child.returncode
        result.fail_lines += len(FAIL_LINE.findall(text))
        # Same audit the single-GPU harness applies: a run whose physics-mode
        # receipt is not full-exact, or that fell back off a captured graph, is
        # not an acceptance measurement however fast it was.
        # main.cpp caps host_threads at min(batch_width, jobs) before the
        # RASBERY_BATCH_HOST_THREADS override, so a chunk smaller than the
        # worker count legitimately reports fewer threads.  Expect what the
        # executable will actually do, or the audit fails the last chunk of
        # every run.
        plan = LaunchPlan(
            batch_width=batch_width,
            jobs=end - start,
            visible_cpus=budget.visible_cpus,
            host_workers=min(budget.driver_workers, batch_width, end - start),
            worker_policy="multi_gpu",
            gpu=gpu,
        )
        result.problems.extend(
            f"gpu{gpu} chunk{chunk_index}: {p}" for p in check_run_receipts(text, plan)
        )
        for match in REFILL_RECEIPT.finditer(text):
            try:
                result.refill_receipts.append(json.loads(match.group(1)))
            except ValueError:
                result.problems.append(f"gpu{gpu} chunk{chunk_index}: unparseable REFILL receipt")

        if claim == "all":
            break

    result.wall_s = time.monotonic() - started
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one width-M refill batch per GPU off a shared job queue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gpus",
        default="0",
        help="comma-separated physical GPU indices, one process each. `--gpus 0` is the "
        "single-GPU form; use it where a second device is off limits",
    )
    p.add_argument("--batch-width", type=int, required=True, help="CUDA arena width per process")
    p.add_argument("--jobs", required=True,
                   help="job manifest: `<input.json> <output.h5> [result-mode]` per line; the "
                        "optional third field overrides --result for that job alone")
    p.add_argument(
        "--result",
        choices=RESULT_MODES,
        help=(
            "what every job writes unless its manifest line says otherwise: full (result "
            "HDF5 + restarts + pin CSV), pin-off (no pin output), light (scalar JSONL, no "
            "HDF5).  All three run the same physics and produce the same trajectory digest; "
            "light additionally needs RASBERY_ALLOW_SCREENING=1"
        ),
    )
    p.add_argument(
        "--claim",
        default="auto",
        help="jobs per claim: auto (half the remaining queue per GPU, floored at "
        "--batch-width), all (static split, one process per GPU), or an integer",
    )
    p.add_argument("--pin", default="taskset", choices=("taskset", "numactl", "none"),
                   help="how each GPU process is bound to its CPU share")
    p.add_argument("--driver-workers", type=int, help="Driver refill lanes per process; default min(width, cpus/gpu)")
    p.add_argument("--solver-threads", type=int, help="RASBERY_OMP_THREADS per process")
    p.add_argument("--workdir", default="multi_gpu_run", help="where chunk manifests and per-chunk logs land")
    p.add_argument(
        "--cwd",
        help="run each RASBERY process here. A deck's cross-section file and its "
        "restart namespace are relative to the deck directory, so a manifest of "
        "relative inputs needs this; absolute manifests do not",
    )
    p.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER, help="RASBERY executable, preceded by --")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    executable = list(args.command)
    if executable and executable[0] == "--":
        executable.pop(0)
    if not executable:
        print("error: missing RASBERY executable after --", file=sys.stderr)
        return 2
    if args.batch_width <= 0:
        print("error: --batch-width must be positive", file=sys.stderr)
        return 2

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        print("error: --gpus needs at least one index", file=sys.stderr)
        return 2
    if len(set(gpus)) != len(gpus):
        print("error: --gpus must be distinct", file=sys.stderr)
        return 2
    if args.claim not in ("auto", "all"):
        try:
            if int(args.claim) <= 0:
                raise ValueError
        except ValueError:
            print("error: --claim must be auto, all, or a positive integer", file=sys.stderr)
            return 2

    try:
        jobs = read_manifest(Path(args.jobs))
        overrides = parse_overrides(args.set_values)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cwd = Path(args.cwd).resolve() if args.cwd else None
    if cwd is not None and not cwd.is_dir():
        print(f"error: --cwd is not a directory: {cwd}", file=sys.stderr)
        return 2

    budget = plan_host_budget(
        gpus=gpus,
        batch_width=args.batch_width,
        visible_cpus=visible_cpu_threads(),
        pin=args.pin,
        driver_workers=args.driver_workers,
        solver_threads=args.solver_threads,
    )
    queue = Queue(workdir / "queue.json", len(jobs))

    print(
        "[RASBERY][MULTI_GPU][PLAN] "
        + json.dumps(
            {
                "gpus": gpus,
                "jobs": len(jobs),
                "batch_width": args.batch_width,
                "claim": args.claim,
                "visible_cpus": budget.visible_cpus,
                "cpus_per_gpu": budget.cpus_per_gpu,
                "driver_workers": budget.driver_workers,
                "solver_threads": budget.solver_threads,
                "writer_threads": budget.writer_threads,
                "pin": budget.pin,
            },
            separators=(",", ":"),
        )
    )

    started = time.monotonic()
    results: list[GpuResult] = []
    if len(gpus) == 1:
        # One GPU: run it inline so the operator sees the failure directly and
        # the numbers are comparable to run_single_gpu_batch.py with no thread
        # scheduling in between.
        results.append(
            run_gpu(
                gpu=gpus[0], index=0, queue=queue, jobs=jobs, budget=budget,
                batch_width=args.batch_width, claim=args.claim, result_mode=args.result,
                executable=executable,
                workdir=workdir, cwd=cwd, overrides=overrides, dry_run=args.dry_run,
            )
        )
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [
                pool.submit(
                    run_gpu,
                    gpu=gpu, index=index, queue=queue, jobs=jobs, budget=budget,
                    batch_width=args.batch_width, claim=args.claim, result_mode=args.result,
                    executable=executable,
                    workdir=workdir, cwd=cwd, overrides=overrides, dry_run=args.dry_run,
                )
                for index, gpu in enumerate(gpus)
            ]
            results = [f.result() for f in futures]
    wall = time.monotonic() - started

    if args.dry_run:
        return 0

    total_jobs = sum(r.jobs for r in results)
    total_fail = sum(r.fail_lines for r in results)
    problems = [p for r in results for p in r.problems]
    exit_code = next((r.returncode for r in results if r.returncode != 0), 0)

    for r in results:
        tail = sum(x.get("tail_idle_s", 0.0) for x in r.refill_receipts)
        refills = sum(x.get("refills", 0) for x in r.refill_receipts)
        busy = [x.get("slot_busy_fraction", 0.0) for x in r.refill_receipts]
        print(
            "[RASBERY][MULTI_GPU][GPU] "
            + json.dumps(
                {
                    "gpu": r.gpu,
                    "processes": r.processes,
                    "jobs": r.jobs,
                    "wall_s": round(r.wall_s, 3),
                    "cases_per_hour": round(3600.0 * r.jobs / r.wall_s, 1) if r.wall_s > 0 else 0.0,
                    "refills": refills,
                    "tail_idle_s": round(tail, 3),
                    "slot_busy_fraction": round(sum(busy) / len(busy), 4) if busy else 0.0,
                    "rc": r.returncode,
                    "fail_lines": r.fail_lines,
                },
                separators=(",", ":"),
            )
        )

    # duplicates/stale_tenants are the Task 20 acceptance counters.  Aggregated
    # here rather than left per-chunk, because the gate is "zero across the
    # campaign", not "zero in the chunk somebody happened to read".
    duplicates = sum(x.get("duplicates", 0) for r in results for x in r.refill_receipts)
    stale = sum(x.get("stale_tenants", 0) for r in results for x in r.refill_receipts)
    if duplicates or stale:
        problems.append(
            f"tenancy audit: duplicates={duplicates} stale_tenants={stale} (both must be 0)"
        )

    print(
        "[RASBERY][MULTI_GPU][TOTAL] "
        + json.dumps(
            {
                "gpus": len(gpus),
                "jobs": total_jobs,
                "wall_s": round(wall, 3),
                "cases_per_hour": round(3600.0 * total_jobs / wall, 1) if wall > 0 else 0.0,
                "duplicates": duplicates,
                "stale_tenants": stale,
                "rc": exit_code,
                "fail_lines": total_fail,
            },
            separators=(",", ":"),
        )
    )

    for problem in problems:
        print("[RASBERY][MULTI_GPU][FAIL] " + problem, file=sys.stderr)
    if exit_code != 0:
        return exit_code
    if problems or total_fail:
        return 3
    print("[RASBERY][MULTI_GPU][OK] " + json.dumps({"jobs": total_jobs}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
