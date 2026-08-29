#!/usr/bin/env python3
"""Run one refill batch per GPU off a shared job queue (plan Rev.7.1 Sec 13.3).

    python tools/run_multi_gpu_batch.py --gpus 0 --batch-width 4 \
        --jobs manifest.txt -- ./RASBERY

    python tools/run_multi_gpu_batch.py --gpus 0,1,2,3 --batch-width 64 \
        --jobs manifest.txt --pin numactl -- ./build/RASBERY

    python tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 4 \
        --batch-width 16 --mps --jobs manifest.txt -- ./build/RASBERY

WHY PROCESS-PER-GPU AND NOT ONE PROCESS WITH FOUR CONTEXTS.  Every arena in
this codebase -- CMFD, nodal, physics -- is a process-lifetime singleton sized
once for width M against the current device, and the rendezvous elects one
launcher for one stream.  Making them device-aware is a rewrite of the batch
core; giving each GPU its own process is a launcher.  CUDA_VISIBLE_DEVICES makes
each process believe it owns device 0, so nothing in the C++ has to change and
the per-GPU numbers stay directly comparable to the single-GPU harness.

WHY K PROCESSES PER GPU (L5, GA evaluator plan Sec 5.5).  The same singleton
property is what makes `--procs-per-gpu K` a WIDTH lever.  nsys on the 238 M64
batch says the run is host-bound, not device-bound: `pthread_cond_wait` 72.6 %
and `pthread_mutex_lock` 17.6 % of osrt, mean rendezvous width 14.5 of 64
declared slots, SM 62 %, GPU memory throughput 7 %.  One arena of 64 slots
whose rendezvous only ever gathers 14.5 participants is paying the DECLARED
width in kernel grid and the ACHIEVED width in useful work.  K processes of
M/K slots each give K independent arenas, K independent rendezvous, and K
independent host mutex sets -- the same aggregate declared width, split so that
each rendezvous is narrow enough to fill.  Nothing in src/ changes: the
processes never see each other.

MPS IS AN ACCELERANT, NOT THE MECHANISM.  Without MPS the K contexts on one
device are TIME-SLICED: correctness is unaffected (contexts are isolated) but
the kernels of different processes cannot overlap on the SMs, so the width
recovered on the host is partly given back on the device.  `--mps` puts the
processes into one MPS server so their kernels share the device concurrently.
The dispatcher starts and stops the control daemon itself, in the workdir, so
a campaign leaves no daemon behind.

WHY A SHARED QUEUE AND NOT A STATIC SPLIT.  A static split finishes when its
SLOWEST partition finishes, and the partitions are not equal: decks differ in
statepoint count, and on a mixed host the GPUs differ too.  The queue is a
manifest plus a claim file, and a claim is one flock'd read-modify-write.  Each
worker runs a SEQUENCE of RASBERY processes, claiming a chunk at a time -- so a
worker that runs dry steals from what is left instead of idling, and a worker
that is slow simply claims less.  `--claim all` degenerates to the static split.

WHY CHUNKS AND NOT ONE JOB AT A TIME.  A process pays arena stand-up and graph
capture once, and Task 20's in-process refill is what makes a chunk cheap: the
chunk is admitted into M slots and refilled from inside the process, so the
claim granularity only has to be coarse enough to amortise startup.  `auto`
halves the remaining queue across the WORKERS each round, which ends at chunks
of `--batch-width`.

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
import shutil
import subprocess
import sys
import threading
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
OCCUPANCY_RECEIPT = re.compile(r"\[RASBERY\]\[CUDA\]\[BATCH_OCCUPANCY\]\s*(\{.*\})")
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
# The memory guard (L5)
# ---------------------------------------------------------------------------
#
# The arenas are sized ONCE, at process start, for the declared width -- so K
# processes of width W hold K*W slots of device memory on ONE device for the
# whole run, not K*W/2 on average.  The reference point is the measured 238
# M64 full-output peak: ~13 GB resident for 64 slots.  Per slot that is
# 13/64 = 0.203 GB, and it is the only number in this file that came off a
# device rather than out of the source.
#
# It is charged PER DEVICE, not per campaign: two GPUs each have their own
# memory, so what has to fit is (procs-per-gpu * batch-width * per-slot) on
# each listed device, and the aggregate is reported only so the operator can
# see the campaign's total footprint.
VRAM_GB_M64_FULL_OUTPUT = 13.0
VRAM_SLOTS_M64 = 64
VRAM_GB_PER_SLOT = VRAM_GB_M64_FULL_OUTPUT / VRAM_SLOTS_M64  # 0.203 GB
VRAM_MARGIN_GB = 1.0


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
    """A claim cursor over the manifest, shared by the workers.

    flock, not a lock-free trick: the workers are threads in THIS process today,
    but the file lock means a second dispatcher (or an operator's manual run)
    against the same queue file claims correctly too, which is what makes it
    safe to restart a half-finished campaign.  With `--procs-per-gpu K` the
    claimants on one device are K SEPARATE worker threads driving K separate
    RASBERY processes, and the lock is what keeps them from claiming the same
    span -- there is no per-GPU serialisation anywhere else.
    """

    def __init__(self, state_path: Path, total: int) -> None:
        self._path = state_path
        self._total = total
        # flock covers OTHER processes; it does not exist on Windows, and where
        # it does exist it is per-open-file-description, so two threads of THIS
        # process each opening the file would both take it and both win.  The
        # in-process mutex is what makes concurrent claims safe, and with
        # --procs-per-gpu the workers on one device are exactly that: threads of
        # one dispatcher.  Without it the read-modify-write interleaves and a
        # claimant reads a half-truncated file.
        self._mutex = threading.Lock()
        self._path.write_text(json.dumps({"claimed": 0, "total": total}), encoding="utf-8")

    def claim(self, size: int) -> tuple[int, int]:
        """Reserve up to *size* jobs; returns [start, end) into the manifest."""
        with self._mutex:
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
        with self._mutex:
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
    procs_per_gpu: int
    processes: int
    cpus_per_proc: int
    driver_workers: int
    solver_threads: int
    writer_threads: int
    cpu_sets: list[list[int]]
    pin: str

    @property
    def cpus_per_gpu(self) -> int:
        """The whole device's host share, however many processes carry it."""
        return self.cpus_per_proc * self.procs_per_gpu


def plan_host_budget(
    *,
    gpus: Sequence[str],
    batch_width: int,
    visible_cpus: int,
    pin: str,
    driver_workers: int | None,
    solver_threads: int | None,
    procs_per_gpu: int = 1,
) -> HostBudget:
    """Split the host between the RASBERY processes (plan Sec 13.3, Sec 5.5).

    The split is by CPU COUNT, not by preference: two processes that both think
    they own all 96 cores oversubscribe by 2x, and OMP_PROC_BIND=TRUE then pins
    their threads onto the same places.  That costs more than the GPU work it is
    feeding, and it is invisible in any per-GPU receipt.

    With `--procs-per-gpu K` the denominator is the PROCESS count, G*K, not the
    GPU count.  L5 exists because the host is the bottleneck; handing K
    processes the whole host each would recreate the contention it is meant to
    remove, one layer up.

    The two thread budgets are derived from that per-process share and not from
    it directly: RASBERY_BATCH_HOST_THREADS is the number of Driver refill
    lanes (capped at the arena width, because main.cpp will not run more
    Drivers than there are slots) and RASBERY_OMP_THREADS is what EACH of those
    lanes may spend inside its solver region.  Setting the latter to the whole
    per-process core count would oversubscribe by the lane count, which is the
    exact mistake the split is here to prevent.
    """
    n_gpus = len(gpus)
    procs_per_gpu = max(1, procs_per_gpu)
    processes = n_gpus * procs_per_gpu
    cpus_per_proc = max(1, visible_cpus // processes)
    # The Driver workers are the refill lanes; capping them at the arena width
    # matches main.cpp, which will not run more Drivers than there are slots.
    workers = driver_workers if driver_workers else min(batch_width, cpus_per_proc)
    workers = max(1, min(workers, batch_width))
    # What is left of this process's share, after the writer pool, is what the
    # per-Driver solver regions may spend.  At least 1: a zero would be a
    # silently serial solver.
    spare = max(0, cpus_per_proc - workers - WRITER_THREADS_PER_PROCESS)
    threads = solver_threads if solver_threads else max(1, min(3, 1 + spare // max(1, workers)))

    sets: list[list[int]] = []
    if pin != "none":
        for index in range(processes):
            lo = index * cpus_per_proc
            sets.append(list(range(lo, lo + cpus_per_proc)))
    return HostBudget(
        visible_cpus=visible_cpus,
        gpus=n_gpus,
        procs_per_gpu=procs_per_gpu,
        processes=processes,
        cpus_per_proc=cpus_per_proc,
        driver_workers=workers,
        solver_threads=threads,
        writer_threads=WRITER_THREADS_PER_PROCESS,
        cpu_sets=sets,
        pin=pin,
    )


def pin_prefix(budget: HostBudget, index: int) -> list[str]:
    """CPU binding for GLOBAL process *index* (0 .. gpus*procs_per_gpu - 1)."""
    if budget.pin == "none" or not budget.cpu_sets:
        return []
    cpus = budget.cpu_sets[index]
    if budget.pin == "numactl":
        return ["numactl", f"--physcpubind={cpus[0]}-{cpus[-1]}", "--localalloc"]
    return ["taskset", "-c", f"{cpus[0]}-{cpus[-1]}"]


# ---------------------------------------------------------------------------
# The memory guard
# ---------------------------------------------------------------------------


@dataclass
class DeviceVram:
    gpu: str
    total_gb: float | None
    demand_gb: float
    budget_gb: float | None
    verdict: str  # "fits" | "over" | "unverified"


@dataclass
class VramPlan:
    per_slot_gb: float
    margin_gb: float
    batch_width: int
    procs_per_gpu: int
    per_process_gb: float
    per_device_gb: float
    aggregate_gb: float
    devices: list[DeviceVram]

    @property
    def over(self) -> list[DeviceVram]:
        return [d for d in self.devices if d.verdict == "over"]

    @property
    def unverified(self) -> list[DeviceVram]:
        return [d for d in self.devices if d.verdict == "unverified"]

    def receipt(self) -> dict:
        return {
            "per_slot_gb": round(self.per_slot_gb, 4),
            "batch_width": self.batch_width,
            "procs_per_gpu": self.procs_per_gpu,
            "per_process_gb": round(self.per_process_gb, 3),
            "per_device_gb": round(self.per_device_gb, 3),
            "aggregate_gb": round(self.aggregate_gb, 3),
            "margin_gb": round(self.margin_gb, 3),
            "devices": [
                {
                    "gpu": d.gpu,
                    "total_gb": round(d.total_gb, 3) if d.total_gb is not None else None,
                    "budget_gb": round(d.budget_gb, 3) if d.budget_gb is not None else None,
                    "demand_gb": round(d.demand_gb, 3),
                    "verdict": d.verdict,
                }
                for d in self.devices
            ],
        }


def query_device_memory_gb(gpu: str) -> float | None:
    """Total memory of physical device *gpu*, in GB, or None if unknowable.

    `nvidia-smi -i <index>` is asked BEFORE any child is launched, so the index
    is still the physical one -- inside a child CUDA_VISIBLE_DEVICES has
    already renamed it to 0.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        done = subprocess.run(  # noqa: S603
            [exe, "-i", str(gpu), "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    for line in done.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return float(line) / 1024.0
        except ValueError:
            return None
    return None


def plan_vram(
    *,
    gpus: Sequence[str],
    procs_per_gpu: int,
    batch_width: int,
    per_slot_gb: float = VRAM_GB_PER_SLOT,
    margin_gb: float = VRAM_MARGIN_GB,
    device_memory_gb: float | None = None,
    probe=query_device_memory_gb,
) -> VramPlan:
    """What K processes of width W will hold on each listed device.

    A process's arenas are sized for its OWN declared width, so K of them cost
    K times one -- the point of L5 is that the aggregate declared width stays
    the same while the rendezvous narrows, and that means the memory stays the
    same too.  The guard exists for the arm that changes both (`4 x M64` rather
    than `4 x M16`), which is the easy thing to type and the one that dies at
    arena stand-up, after the queue has been claimed.
    """
    per_process = batch_width * per_slot_gb
    per_device = procs_per_gpu * per_process
    devices: list[DeviceVram] = []
    for gpu in gpus:
        total = device_memory_gb if device_memory_gb is not None else probe(gpu)
        if total is None:
            devices.append(DeviceVram(gpu=gpu, total_gb=None, demand_gb=per_device,
                                      budget_gb=None, verdict="unverified"))
            continue
        budget = total - margin_gb
        devices.append(
            DeviceVram(
                gpu=gpu, total_gb=total, demand_gb=per_device, budget_gb=budget,
                verdict="fits" if per_device <= budget else "over",
            )
        )
    return VramPlan(
        per_slot_gb=per_slot_gb,
        margin_gb=margin_gb,
        batch_width=batch_width,
        procs_per_gpu=procs_per_gpu,
        per_process_gb=per_process,
        per_device_gb=per_device,
        aggregate_gb=per_device * len(gpus),
        devices=devices,
    )


# ---------------------------------------------------------------------------
# MPS
# ---------------------------------------------------------------------------


class MpsSession:
    """The MPS control daemon for one run, started and stopped by us.

    Started in the WORKDIR (its own pipe and log directories) rather than in
    /tmp/nvidia-mps, so two campaigns on one host do not join each other's
    server by accident and so `stop()` cannot take down a daemon somebody else
    is using.  `stop()` is idempotent and is called from a finally, because a
    daemon left running holds the device's compute mode and the next arm's
    "without MPS" control would silently be an MPS run.
    """

    CONTROL = "nvidia-cuda-mps-control"

    def __init__(
        self,
        *,
        workdir: Path,
        gpus: Sequence[str],
        thread_percent: int | None,
        procs_per_gpu: int,
    ) -> None:
        self.workdir = workdir
        self.gpus = list(gpus)
        self.procs_per_gpu = max(1, procs_per_gpu)
        # Each client's share of the SMs.  100/K by default: K clients that
        # each ask for 100 % is how an MPS server ends up time-slicing anyway.
        self.thread_percent = (
            thread_percent if thread_percent is not None else max(1, 100 // self.procs_per_gpu)
        )
        self.pipe_dir = workdir / "mps" / "pipe"
        self.log_dir = workdir / "mps" / "log"
        self.control: str | None = None
        self.active = False
        self.reason = ""
        self._started = False

    # -- lifecycle ---------------------------------------------------------

    def _control_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_dir)
        env["CUDA_MPS_LOG_DIRECTORY"] = str(self.log_dir)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpus)
        return env

    def start(self) -> bool:
        """Try to bring the daemon up.  False (with .reason set) if we cannot."""
        self.control = shutil.which(self.CONTROL)
        if self.control is None:
            self.reason = (
                f"{self.CONTROL} is not on PATH: this host has no MPS control daemon "
                "(WSL and most container images do not ship one)"
            )
            return False
        self.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = self._control_env()
        try:
            done = subprocess.run(  # noqa: S603
                [self.control, "-d"], env=env, capture_output=True, text=True,
                timeout=120, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.reason = f"could not start {self.CONTROL}: {exc}"
            return False
        if done.returncode != 0:
            self.reason = (
                f"{self.CONTROL} -d exited {done.returncode}: "
                + (done.stderr or done.stdout or "").strip().replace("\n", " ")[:400]
            )
            return False
        self._started = True
        # -d returns as soon as the daemon forks; a run that assumed it was up
        # would put every client into a plain context and report an MPS arm.
        ok, detail = self._ping()
        if not ok:
            self.reason = f"{self.CONTROL} started but does not answer: {detail}"
            self.stop()
            return False
        self.active = True
        return True

    def _ping(self) -> tuple[bool, str]:
        assert self.control is not None
        try:
            done = subprocess.run(  # noqa: S603
                [self.control], env=self._control_env(), input="get_server_list\n",
                capture_output=True, text=True, timeout=60, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        if done.returncode != 0:
            return False, (done.stderr or done.stdout or "").strip().replace("\n", " ")[:400]
        return True, done.stdout.strip()

    def stop(self) -> None:
        if not self._started or self.control is None:
            self._started = False
            self.active = False
            return
        try:
            subprocess.run(  # noqa: S603
                [self.control], env=self._control_env(), input="quit\n",
                capture_output=True, text=True, timeout=120, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            print(f"[RASBERY][MULTI_GPU][MPS] shutdown warning: {exc}", file=sys.stderr)
        finally:
            self._started = False
            self.active = False

    # -- what the children inherit ----------------------------------------

    def client_env(self) -> dict[str, str]:
        """Env additions for a RASBERY child.  Empty unless the server is up."""
        if not self.active:
            return {}
        return {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(self.log_dir),
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.thread_percent),
        }

    def receipt(self, *, requested: bool) -> dict:
        return {
            "requested": requested,
            "active": self.active,
            "control": self.control,
            "thread_percent": self.thread_percent if self.active else None,
            "pipe_dir": str(self.pipe_dir) if self.active else None,
            "log_dir": str(self.log_dir) if self.active else None,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Per-process worker
# ---------------------------------------------------------------------------


def chunk_result_mode(chunk: Sequence[tuple[str, str, str]], default: str | None) -> str:
    """The result mode ONE process will report for *chunk*.

    main.cpp:541 takes `any_of(light)` over the whole job list -- one scalar-only
    job makes the whole process a screening run, and the PHYSICS_MODE receipt
    the audit reads is printed once, before any deck starts.  The dispatcher has
    to agree, because it is the dispatcher that decides which jobs share a
    process: a chunk that happens to mix a light job into full ones would
    otherwise be failed for printing exactly the receipt it was asked to print.
    """
    effective = [mode or (default or "full") for _i, _o, mode in chunk]
    if "light" in effective:
        return "light"
    if effective and all(mode == effective[0] for mode in effective):
        return effective[0]
    # A mix of full and pin-off: both write the result HDF5, so both are audited
    # as exact, and "full" is the name of that expectation.
    return "full"


@dataclass
class WorkerResult:
    gpu: str
    proc: int = 0
    index: int = 0
    cpus: str = ""
    processes: int = 0
    jobs: int = 0
    returncode: int = 0
    fail_lines: int = 0
    problems: list[str] = field(default_factory=list)
    refill_receipts: list[dict] = field(default_factory=list)
    occupancy_receipts: list[dict] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def mean_width(self) -> float:
        """Rendezvous width this PROCESS achieved, launch-weighted.

        Launch-weighted and not a mean of means: a chunk with three launches
        and a chunk with three hundred are not equal evidence, and the whole
        point of L5 is what this number does when the arena narrows.
        """
        solves = sum(float(x.get("instance_solves", 0.0)) for x in self.occupancy_receipts)
        launches = sum(float(x.get("launches", 0.0)) for x in self.occupancy_receipts)
        return solves / launches if launches > 0 else 0.0

    @property
    def width_fill(self) -> float:
        """mean_width / declared slots -- the number the L5 arms compare on.

        mean_width alone cannot compare arms: a 4-slot arena cannot reach the
        mean width of a 64-slot one, so a working L5 arm reports a SMALLER
        mean_width than the control while doing more work.  The claim of L5 is
        that the FRACTION of the declared width the rendezvous actually gathers
        goes up -- 14.5/64 = 22.7 % on the 238 M64 control is the number it has
        to beat.
        """
        slots = max((float(x.get("slots", 0.0)) for x in self.occupancy_receipts), default=0.0)
        return self.mean_width / slots if slots > 0 else 0.0

    @property
    def tail_idle_s(self) -> float:
        return sum(float(x.get("tail_idle_s", 0.0)) for x in self.refill_receipts)

    @property
    def refills(self) -> int:
        return sum(int(x.get("refills", 0)) for x in self.refill_receipts)

    @property
    def cases_per_hour(self) -> float:
        return 3600.0 * self.jobs / self.wall_s if self.wall_s > 0 else 0.0


def run_worker(
    *,
    gpu: str,
    proc: int,
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
    extra_env: dict[str, str],
    dry_run: bool,
) -> WorkerResult:
    """One process SEQUENCE: claim, run, repeat, on device *gpu* as slot *proc*."""
    cpus = budget.cpu_sets[index] if budget.cpu_sets else []
    result = WorkerResult(
        gpu=gpu, proc=proc, index=index,
        cpus=f"{cpus[0]}-{cpus[-1]}" if cpus else "",
    )
    started = time.monotonic()
    chunk_index = 0
    # Every artefact this worker writes is named by (gpu, proc): with K
    # processes on one device, a name keyed only by the GPU would have two
    # workers overwrite each other's chunk manifest between write and exec.
    stem = f"gpu{gpu}.p{proc}"

    while True:
        if claim == "all":
            # Static split: each WORKER takes its equal share once, and that is
            # the whole run.  Chunking it further would only add process
            # restarts.
            size = -(-len(jobs) // budget.processes)
        elif claim == "auto":
            remaining = queue.remaining()
            if remaining <= 0:
                break
            if budget.processes == 1:
                # There is nobody to steal from.  Splitting the queue here would
                # DRAIN the batch between chunks, which is precisely the tail
                # Task 20 exists to remove -- so one worker claims the lot and
                # lets the in-process refill do the work.
                size = remaining
            else:
                size = max(batch_width, -(-remaining // (2 * budget.processes)))
        else:
            size = int(claim)

        start, end = queue.claim(size)
        if start >= end:
            break

        chunk_index += 1
        manifest = workdir / f"{stem}.chunk{chunk_index:04d}.txt"
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
        env.update(extra_env)
        env.update(overrides)

        command = (
            pin_prefix(budget, index)
            + list(executable)
            + ["--jobs", str(manifest), "--batch-mode", str(batch_width)]
            + (["--result", result_mode] if result_mode else [])
        )
        log = workdir / f"{stem}.chunk{chunk_index:04d}.log"
        # The manifest carries absolute paths, but a deck's own references (its
        # cross-section HDF5, its restart namespace) are relative to the deck
        # directory, so the child has to run where the decks live.
        manifest_arg = str(manifest.resolve())
        command[command.index("--jobs") + 1] = manifest_arg

        if dry_run:
            print(
                f"[RASBERY][MULTI_GPU][DRY] gpu={gpu} proc={proc} jobs={end - start} "
                + " ".join(command)
            )
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
            result_mode=chunk_result_mode(jobs[start:end], result_mode),
        )
        result.problems.extend(
            f"gpu{gpu} p{proc} chunk{chunk_index}: {p}" for p in check_run_receipts(text, plan)
        )
        for match in REFILL_RECEIPT.finditer(text):
            try:
                result.refill_receipts.append(json.loads(match.group(1)))
            except ValueError:
                result.problems.append(
                    f"gpu{gpu} p{proc} chunk{chunk_index}: unparseable REFILL receipt"
                )
        for match in OCCUPANCY_RECEIPT.finditer(text):
            try:
                result.occupancy_receipts.append(json.loads(match.group(1)))
            except ValueError:
                pass  # occupancy is a judgement field, not a gate

        if claim == "all":
            break

    result.wall_s = time.monotonic() - started
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run width-M refill batches across GPUs (and across processes per GPU) "
                    "off a shared job queue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gpus",
        default="0",
        help="comma-separated physical GPU indices, one process SEQUENCE each (times "
        "--procs-per-gpu). `--gpus 0` is the single-GPU form; use it where a second "
        "device is off limits",
    )
    p.add_argument(
        "--procs-per-gpu", type=int, default=1, metavar="K",
        help="independent RASBERY processes per listed GPU (L5, GA plan Sec 5.5). Each "
        "gets its own arenas, its own rendezvous and its own slice of the host, so "
        "--batch-width is the width PER PROCESS: `--procs-per-gpu 4 --batch-width 16` "
        "declares the same 64 slots per device as `--batch-width 64` does alone",
    )
    p.add_argument("--batch-width", type=int, required=True,
                   help="CUDA arena width per PROCESS")
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
        help="jobs per claim: auto (half the remaining queue per worker, floored at "
        "--batch-width), all (static split, one process per worker), or an integer",
    )
    p.add_argument("--pin", default="taskset", choices=("taskset", "numactl", "none"),
                   help="how each process is bound to its CPU share")
    p.add_argument("--driver-workers", type=int,
                   help="Driver refill lanes per process; default min(width, cpus/process)")
    p.add_argument("--solver-threads", type=int, help="RASBERY_OMP_THREADS per process")
    p.add_argument("--workdir", default="multi_gpu_run",
                   help="where chunk manifests, per-chunk logs and the MPS pipe/log land")
    p.add_argument(
        "--cwd",
        help="run each RASBERY process here. A deck's cross-section file and its "
        "restart namespace are relative to the deck directory, so a manifest of "
        "relative inputs needs this; absolute manifests do not",
    )
    p.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE")

    mps = p.add_argument_group("CUDA MPS")
    mps.add_argument(
        "--mps", action="store_true",
        help="start an MPS control daemon for this run (pipe/log under --workdir) and put "
        "every child into it, so the K contexts on one device share the SMs instead of "
        "time-slicing. Stopped again on the way out, including on failure",
    )
    mps.add_argument(
        "--mps-optional", action="store_true",
        help="with --mps: fall back to time-sliced contexts when the control daemon is "
        "unavailable, instead of refusing the run. The receipt still says active:false",
    )
    mps.add_argument(
        "--mps-thread-percent", type=int, metavar="P",
        help="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE per child; default 100/K",
    )

    mem = p.add_argument_group("device memory guard")
    mem.add_argument(
        "--vram-per-slot-gb", type=float, default=VRAM_GB_PER_SLOT,
        help="arena cost of one batch slot; default is the measured 238 M64 full-output "
        "peak (13 GB / 64 slots)",
    )
    mem.add_argument("--vram-margin-gb", type=float, default=VRAM_MARGIN_GB,
                     help="device memory held back from the arenas")
    mem.add_argument("--device-memory-gb", type=float,
                     help="assume this much memory per device instead of asking nvidia-smi")
    mem.add_argument(
        "--allow-vram-overcommit", action="store_true",
        help="run even when the guard says the arenas will not fit. The failure it is "
        "predicting happens at arena stand-up, after the queue has been claimed",
    )

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
    if args.procs_per_gpu <= 0:
        print("error: --procs-per-gpu must be positive", file=sys.stderr)
        return 2
    if args.mps_thread_percent is not None and not 1 <= args.mps_thread_percent <= 100:
        print("error: --mps-thread-percent must be in [1, 100]", file=sys.stderr)
        return 2
    if args.mps_optional and not args.mps:
        print("error: --mps-optional means nothing without --mps", file=sys.stderr)
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

    # A light job needs RASBERY_ALLOW_SCREENING, and main.cpp refuses without
    # it -- once per chunk, AFTER the queue was claimed and the arenas stood up.
    # Asked here, the campaign loses nothing.  The parse mirrors main.cpp:545
    # exactly (set and not one of the falsey words), because a guard that is
    # stricter than the executable refuses runs that would have worked.
    if args.result == "light" or any(mode == "light" for _i, _o, mode in jobs):
        allow = overrides.get("RASBERY_ALLOW_SCREENING",
                              os.environ.get("RASBERY_ALLOW_SCREENING", ""))
        if allow == "" or allow in ("0", "off", "OFF", "false", "FALSE"):
            print(
                "error: this run writes scalar-only (light) results, which RASBERY "
                "classifies as a screening run and refuses unless "
                "RASBERY_ALLOW_SCREENING=1 is set. Export it, or pass "
                "--set RASBERY_ALLOW_SCREENING=1, if a screening campaign is what you "
                "meant. Screening results are never acceptance results.",
                file=sys.stderr,
            )
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
        procs_per_gpu=args.procs_per_gpu,
    )
    queue = Queue(workdir / "queue.json", len(jobs))

    print(
        "[RASBERY][MULTI_GPU][PLAN] "
        + json.dumps(
            {
                "gpus": gpus,
                "procs_per_gpu": budget.procs_per_gpu,
                "processes": budget.processes,
                "jobs": len(jobs),
                "batch_width": args.batch_width,
                "declared_width_per_gpu": args.batch_width * budget.procs_per_gpu,
                "claim": args.claim,
                "visible_cpus": budget.visible_cpus,
                "cpus_per_proc": budget.cpus_per_proc,
                "cpus_per_gpu": budget.cpus_per_gpu,
                "driver_workers": budget.driver_workers,
                "solver_threads": budget.solver_threads,
                "writer_threads": budget.writer_threads,
                "pin": budget.pin,
            },
            separators=(",", ":"),
        )
    )

    # --- the memory guard, before anything claims the queue -----------------
    vram = plan_vram(
        gpus=gpus,
        procs_per_gpu=budget.procs_per_gpu,
        batch_width=args.batch_width,
        per_slot_gb=args.vram_per_slot_gb,
        margin_gb=args.vram_margin_gb,
        device_memory_gb=args.device_memory_gb,
    )
    print("[RASBERY][MULTI_GPU][VRAM] " + json.dumps(vram.receipt(), separators=(",", ":")))
    if vram.over and not args.allow_vram_overcommit:
        for device in vram.over:
            print(
                "[RASBERY][MULTI_GPU][FAIL] device %s: %d process(es) x width %d x %.3f GB/slot "
                "= %.2f GB needed, but the device has %.2f GB and the guard holds back "
                "%.2f GB (%.2f GB usable). Lower --batch-width or --procs-per-gpu, or pass "
                "--allow-vram-overcommit to find out the hard way at arena stand-up."
                % (device.gpu, budget.procs_per_gpu, args.batch_width, vram.per_slot_gb,
                   device.demand_gb, device.total_gb or 0.0, vram.margin_gb,
                   device.budget_gb or 0.0),
                file=sys.stderr,
            )
        return 2
    for device in vram.unverified:
        print(
            "[RASBERY][MULTI_GPU][WARN] device %s: nvidia-smi could not be asked how much "
            "memory it has, so the %.2f GB the arenas will hold is UNVERIFIED. Pass "
            "--device-memory-gb to check it." % (device.gpu, device.demand_gb),
            file=sys.stderr,
        )

    # --- MPS, likewise before anything claims the queue ---------------------
    mps = MpsSession(
        workdir=workdir, gpus=gpus, thread_percent=args.mps_thread_percent,
        procs_per_gpu=budget.procs_per_gpu,
    )
    if args.mps and not args.dry_run:
        mps.start()
    elif args.mps and args.dry_run:
        mps.reason = "not started: --dry-run"
    print(
        "[RASBERY][MULTI_GPU][MPS] "
        + json.dumps(mps.receipt(requested=args.mps), separators=(",", ":"))
    )
    if args.mps and not mps.active and not args.mps_optional and not args.dry_run:
        print(
            "[RASBERY][MULTI_GPU][FAIL] --mps was requested and MPS is not available: "
            + mps.reason
            + ". Refusing rather than running time-sliced under an MPS label; re-run with "
            "--mps-optional to take the time-sliced arm deliberately, or without --mps.",
            file=sys.stderr,
        )
        return 2

    extra_env = mps.client_env()
    # stop() clears .active, and the TOTAL receipt is printed after it: what
    # the run was is not what the daemon is by then.
    mps_active = mps.active
    workers = [
        (gpu, proc, gpu_index * budget.procs_per_gpu + proc)
        for gpu_index, gpu in enumerate(gpus)
        for proc in range(budget.procs_per_gpu)
    ]

    started = time.monotonic()
    results: list[WorkerResult] = []
    try:
        common = dict(
            queue=queue, jobs=jobs, budget=budget, batch_width=args.batch_width,
            claim=args.claim, result_mode=args.result, executable=executable,
            workdir=workdir, cwd=cwd, overrides=overrides, extra_env=extra_env,
            dry_run=args.dry_run,
        )
        if len(workers) == 1:
            # One process: run it inline so the operator sees the failure
            # directly and the numbers are comparable to run_single_gpu_batch.py
            # with no thread scheduling in between.
            gpu, proc, index = workers[0]
            results.append(run_worker(gpu=gpu, proc=proc, index=index, **common))
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                futures = [
                    pool.submit(run_worker, gpu=gpu, proc=proc, index=index, **common)
                    for gpu, proc, index in workers
                ]
                results = [f.result() for f in futures]
    finally:
        # A daemon left running holds the device's compute mode, and the next
        # arm's "without MPS" control would silently be an MPS run.
        mps.stop()
    wall = time.monotonic() - started

    if args.dry_run:
        return 0

    total_jobs = sum(r.jobs for r in results)
    total_fail = sum(r.fail_lines for r in results)
    problems = [p for r in results for p in r.problems]
    exit_code = next((r.returncode for r in results if r.returncode != 0), 0)

    # Per PROCESS: with K > 1 the per-GPU line is a sum of things that ran
    # concurrently, and the number L5 is judged on -- mean rendezvous width --
    # is a property of one arena, so it has to be reported where the arena is.
    for r in sorted(results, key=lambda x: x.index):
        busy = [x.get("slot_busy_fraction", 0.0) for x in r.refill_receipts]
        print(
            "[RASBERY][MULTI_GPU][PROC] "
            + json.dumps(
                {
                    "gpu": r.gpu,
                    "proc": r.proc,
                    "cpus": r.cpus,
                    "processes": r.processes,
                    "jobs": r.jobs,
                    "wall_s": round(r.wall_s, 3),
                    "cases_per_hour": round(r.cases_per_hour, 1),
                    "mean_width": round(r.mean_width, 3),
                    "width_fill": round(r.width_fill, 4),
                    "refills": r.refills,
                    "tail_idle_s": round(r.tail_idle_s, 3),
                    "slot_busy_fraction": round(sum(busy) / len(busy), 4) if busy else 0.0,
                    "rc": r.returncode,
                    "fail_lines": r.fail_lines,
                },
                separators=(",", ":"),
            )
        )

    for gpu in gpus:
        share = [r for r in results if r.gpu == gpu]
        if not share:
            continue
        gpu_jobs = sum(r.jobs for r in share)
        # The GPU was busy from the first worker's start to the last worker's
        # finish, and they started together: the device's wall is the LONGEST
        # worker's, not the sum.  Summing would divide the throughput by K.
        gpu_wall = max(r.wall_s for r in share)
        widths = [r.mean_width for r in share if r.mean_width > 0]
        print(
            "[RASBERY][MULTI_GPU][GPU] "
            + json.dumps(
                {
                    "gpu": gpu,
                    "procs": len(share),
                    "processes": sum(r.processes for r in share),
                    "jobs": gpu_jobs,
                    "wall_s": round(gpu_wall, 3),
                    "cases_per_hour": round(3600.0 * gpu_jobs / gpu_wall, 1) if gpu_wall > 0 else 0.0,
                    "mean_width_per_proc": [round(w, 3) for w in
                                            (r.mean_width for r in share)],
                    "mean_width_mean": round(sum(widths) / len(widths), 3) if widths else 0.0,
                    "width_fill_per_proc": [round(r.width_fill, 4) for r in share],
                    "refills": sum(r.refills for r in share),
                    "tail_idle_s": round(sum(r.tail_idle_s for r in share), 3),
                    "tail_idle_max_s": round(max(r.tail_idle_s for r in share), 3),
                    "rc": next((r.returncode for r in share if r.returncode != 0), 0),
                    "fail_lines": sum(r.fail_lines for r in share),
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

    all_widths = [r.mean_width for r in results if r.mean_width > 0]
    all_fills = [r.width_fill for r in results if r.width_fill > 0]
    print(
        "[RASBERY][MULTI_GPU][TOTAL] "
        + json.dumps(
            {
                "gpus": len(gpus),
                "procs_per_gpu": budget.procs_per_gpu,
                "processes": len(workers),
                "batch_width": args.batch_width,
                "jobs": total_jobs,
                "wall_s": round(wall, 3),
                "cases_per_hour": round(3600.0 * total_jobs / wall, 1) if wall > 0 else 0.0,
                "mean_width_per_proc": [round(r.mean_width, 3)
                                        for r in sorted(results, key=lambda x: x.index)],
                "mean_width_mean": round(sum(all_widths) / len(all_widths), 3) if all_widths else 0.0,
                "width_fill_per_proc": [round(r.width_fill, 4)
                                        for r in sorted(results, key=lambda x: x.index)],
                "width_fill_mean": round(sum(all_fills) / len(all_fills), 4) if all_fills else 0.0,
                "tail_idle_s": round(sum(r.tail_idle_s for r in results), 3),
                "tail_idle_max_s": round(max((r.tail_idle_s for r in results), default=0.0), 3),
                "mps_requested": bool(args.mps),
                "mps": mps_active,
                "mps_thread_percent": mps.thread_percent if mps_active else None,
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
