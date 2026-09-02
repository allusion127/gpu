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
    launch_env,
    parse_overrides,
    path_key,
    resolve_declared_fidelity,
    resolve_profile_env,
    resolve_unset,
    visible_cpu_threads,
)
from exact_audit import DECLARABLE_FIDELITIES, receipt_policy  # noqa: E402

# WP16 host-spin: TWO KNOBS THAT ARE DELIBERATELY NOT IN DEFAULT_ENV.
#
#   RASBERY_CUDA_SYNC_MODE = auto | spin | yield | blocking
#       How the host thread waits inside cudaStreamSynchronize.  CUDA's default
#       (`auto`) SPINS while the number of contexts is below the core count,
#       which is exactly the 8-process-on-24-CPU shape the 238 batch runs, and
#       is why that batch shows ~79 % host CPU while ~92 % of process time is
#       inside a synchronise.  `blocking` parks the thread instead, giving the
#       core back at the cost of a wake-up latency per fence.
#       Applied by src/CudaHostSchedule.cu, before the CUDA context exists.
#
#   RASBERY_OMP_WAIT = active | passive
#       Pass-through to OMP_WAIT_POLICY (and GOMP_SPINCOUNT=0 for passive),
#       exported by src/main.cpp's OpenMP re-exec so libgomp sees it in the
#       second image.  DEFAULT_ENV already states PASSIVE/0 directly, so this
#       is how the ACTIVE arm is taken without editing the harness default.
#
# NEITHER BELONGS IN DEFAULT_ENV.  DEFAULT_ENV is the 238 reference line, key
# for key (test/reference/batch_reference_env_238.json), and the reference sets
# neither; adding a key the reference never had is exactly the silent deviation
# RASBERY_PPR_MODE=master was.  Both are ordinary `--set KEY=VALUE` overrides:
# `--set` is `action="append"` and parse_overrides() accepts any [A-Za-z0-9_]
# key, so no per-key plumbing exists or is needed, and every resolved value is
# already printed per process in the [RASBERY][MULTI_GPU][ENV] receipt.
#
#   ... --mps --procs-per-gpu 8 --batch-width 8 #       --set RASBERY_CUDA_SYNC_MODE=blocking --set RASBERY_OMP_WAIT=passive
#
# The child's own [RASBERY][CUDA][SCHED] receipt is what closes the loop: it
# reports what was requested AND what the driver actually applied, so an arm
# that asked too late (a context already created -> cudaErrorSetOnActiveProcess)
# is not silently tabulated as a `blocking` row.

REFILL_RECEIPT = re.compile(r"\[RASBERY\]\[REFILL\]\s*(\{.*\})")
# WP18.  A SEPARATE TAG, on purpose: REFILL_RECEIPT above requires `{`
# immediately after the tag, so it cannot match this line, and the wave-mode
# receipt it parses is unchanged byte for byte.  That is what makes
# `--claim rolling` measurable against `--claim auto` with one binary.
ROLLING_RECEIPT = re.compile(r"\[RASBERY\]\[REFILL\]\[ROLLING\]\s*(\{.*\})")
OCCUPANCY_RECEIPT = re.compile(r"\[RASBERY\]\[CUDA\]\[BATCH_OCCUPANCY\]\s*(\{.*\})")
FAIL_LINE = re.compile(r"\[RASBERY\]\[FAIL\]")

# Plan Sec 13.3 host budget: "라이터 ~8스레드 + GPU당 로더 8~16".  The writer is
# ONE pool for the process (IoWriter.h), so its share is charged per process;
# the loader budget is the OpenMP Driver workers plus the per-Driver solver
# threads, which is what RASBERY_BATCH_HOST_THREADS and RASBERY_OMP_THREADS
# split between them.
#
# WP4 ("개선해야 할 현재 runner 가정"): THE EXECUTABLE DOES NOT HAVE EIGHT WRITER
# THREADS.  src/IoWriter.h runs exactly ONE (`_worker`, a single std::thread --
# IoWriter.h:405, :492) in `thread` mode and NONE in `inline` mode, and with
# `--result light` there is no result HDF5 at all: the scalar JSONL leaves
# through flushLines(), not through the writer's queue.  Charging 8 was not a
# harmless over-estimate -- it is SUBTRACTED from the per-process core share
# before the solver threads are sized (the --no-oversubscribe arm), so on a K=8
# split of 24 cores the phantom writer was larger than the whole process's
# share and pinned the solver count to its floor.  The constant stays as the
# last-resort fallback; plan_writer_threads() derives the real number and
# writer_threads_from_receipt() replaces it with what the binary printed.
WRITER_THREADS_PER_PROCESS = 8
LOADER_THREADS_PER_GPU_MIN = 8
LOADER_THREADS_PER_GPU_MAX = 16

IO_WRITER_RECEIPT = re.compile(r"\[RASBERY\]\[IO_WRITER\]\s*(\{.*\})")


def writer_threads_from_receipt(text: str) -> int | None:
    """Writer threads the EXECUTABLE reported, or None if it did not say.

    `[RASBERY][IO_WRITER] {"mode":"thread"|"inline",...}` (IoWriter.h:841) is
    printed before the first deck, so a calibration wave's log already carries
    it.  `thread` is one writer thread; `inline` is none.  An unknown mode word
    returns None rather than a guess -- the point of reading the receipt is to
    stop guessing, and a guess dressed as a reading is worse than no reading.
    """
    modes = {"thread": 1, "inline": 0}
    seen: int | None = None
    for match in IO_WRITER_RECEIPT.finditer(text):
        try:
            mode = json.loads(match.group(1)).get("mode")
        except ValueError:
            continue
        if mode in modes:
            seen = modes[mode]
    return seen


def plan_writer_threads(
    *,
    result_mode: str | None,
    io_writer_mode: str | None = None,
    observed: int | None = None,
) -> tuple[int, str]:
    """(threads, policy) for ONE process's HDF5 writer pool.

    Precedence: what the executable printed, then RASBERY_IO_WRITER, then the
    result mode.  The policy word goes into the [PLAN] receipt so a change of
    default is visible in the log rather than only in the throughput.
    """
    if observed is not None and observed >= 0:
        return observed, "receipt"
    if (io_writer_mode or "").strip().lower() == "inline":
        return 0, "io_writer_inline"
    if (result_mode or "full") == "light":
        return 0, "light_no_hdf5"
    return 1, "io_writer_thread"

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

# ...AND K PROCESSES OF WIDTH W COST MORE THAN ONE OF WIDTH K*W.  This file used
# to say they cost the same -- "L5 keeps the aggregate declared width fixed, so
# the memory stays the same too" -- and the 238 matrix says otherwise:
#
#     1 x M64  12.3 GB      2 x M32  14.9 GB      4 x M16  20.0 GB
#                           8 x M8   30.2 GB
#
# The increments are 2.60, 2.57, 2.56 GB per ADDITIONAL process: a straight
# line, which is what a per-process fixed cost looks like -- the CUDA context,
# the module images, cuBLAS/cuDNN handles and the allocator's own pool, none of
# which is a function of the arena width.  The slot term did stay flat, exactly
# as the lever promised; what nobody charged was the process.
#
# It matters because the guard is the tuner's upper bound.  Under the old model
# 16 x M4 "needs 13.0 GB"; it needs about 50.7, and on anything smaller than a
# 96 GB device the difference is the whole verdict.  The failure it predicts
# still happens at arena stand-up, after the queue has been claimed.
VRAM_GB_PER_EXTRA_PROCESS = 2.56


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

    def __init__(self, state_path: Path, total: int, processes: int = 1) -> None:
        self._path = state_path
        self._total = total
        # EVERY WORKER GETS A TURN BEFORE ANY WORKER GETS SECONDS.
        #
        # The fair-share cap in auto_claim_size() sizes a claim; it cannot keep a
        # worker that finished its chunk from coming back and taking the share of
        # a worker whose thread had not reached its first claim yet.  Measured
        # here with a millisecond-long fake child at K=12: worker 4 claimed
        # twice (10 jobs) and worker 11 got NOTHING -- the same artefact the cap
        # was added to remove, one layer up.  Real chunks take minutes, so the
        # race is narrow, but "narrow" is how the 115.6 c/h control arm was
        # reported as data three times.
        #
        # So the queue holds back one job for each worker that has not claimed
        # yet: a REPEAT claimant may not take work that would leave a first-time
        # claimant with nothing, and gets an empty span (it stops) instead.  Once
        # every worker has had a turn the reservation is zero and stealing is
        # exactly as free as it was.
        self._processes = max(1, processes)
        self._claimants: set[int] = set()
        # flock covers OTHER processes; it does not exist on Windows, and where
        # it does exist it is per-open-file-description, so two threads of THIS
        # process each opening the file would both take it and both win.  The
        # in-process mutex is what makes concurrent claims safe, and with
        # --procs-per-gpu the workers on one device are exactly that: threads of
        # one dispatcher.  Without it the read-modify-write interleaves and a
        # claimant reads a half-truncated file.
        self._mutex = threading.Lock()
        self._path.write_text(json.dumps({"claimed": 0, "total": total}), encoding="utf-8")

    def claim(self, size: int, index: int | None = None) -> tuple[int, int]:
        """Reserve up to *size* jobs; returns [start, end) into the manifest.

        *index* is the global process index of the claimant.  Given one, the
        queue keeps a job in hand for every worker that has not claimed yet (see
        __init__); an empty span means "there is work, but it is not yours" and
        the caller stops so the worker it is held for can have it.
        """
        with self._mutex:
            if index is not None:
                self._claimants.add(index)
            pending = (self._processes - len(self._claimants)
                       if index is not None else 0)
            with open(self._path, "r+", encoding="utf-8") as handle:
                _lock(handle)
                try:
                    state = json.load(handle)
                    start = int(state["claimed"])
                    allowed = max(0, self._total - start - max(0, pending))
                    end = min(self._total, start + min(max(1, size), allowed))
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
    worker_policy: str = "binary_default_width"
    writer_policy: str = "io_writer_thread"
    result_mode: str = "full"

    @property
    def cpus_per_gpu(self) -> int:
        """The whole device's host share, however many processes carry it."""
        return self.cpus_per_proc * self.procs_per_gpu

    @property
    def host_threads_per_proc(self) -> int:
        """Threads one process can have IN EXISTENCE -- not cores it needs.

        The distinction is the whole WP4 host-budget correction.  The OpenMP
        region that carries the Driver lanes is sized by OMP_NUM_THREADS, and
        OMP_MAX_ACTIVE_LEVELS=1 makes the inner solver regions serial, so the
        process's thread count is that one region plus the single writer
        thread.  Nearly all of those threads are BLOCKED on the GPU rendezvous
        -- which is exactly why the lanes are not divided by K, and why this
        number must never be turned back into a core requirement.  It is
        reported so the operator can see the ratio, and used only as the
        tuner's third tie-break ("lower CPU usage" in the plan's ordering).
        """
        return max(1, self.solver_threads) + self.writer_threads

    @property
    def host_thread_demand(self) -> int:
        """The whole host's: every process's, summed."""
        return self.host_threads_per_proc * self.processes

    @property
    def host_thread_ratio(self) -> float:
        """Threads per visible CPU.  The reference arm is ~2.7 and is correct."""
        return self.host_thread_demand / self.visible_cpus if self.visible_cpus else 0.0


def plan_host_budget(
    *,
    gpus: Sequence[str],
    batch_width: int,
    visible_cpus: int,
    pin: str,
    driver_workers: int | None,
    solver_threads: int | None,
    procs_per_gpu: int = 1,
    oversubscribe: bool = True,
    result_mode: str | None = None,
    io_writer_mode: str | None = None,
    writer_threads_observed: int | None = None,
) -> HostBudget:
    """Split the host between the RASBERY processes (plan Sec 13.3, Sec 5.5).

    TWO RESOURCES, SPLIT DIFFERENTLY.  The CORES are partitioned: with K
    processes on a device, each gets visible_cpus // (G*K) of them and a taskset
    range that does not overlap anybody else's, because two processes that both
    believe they own all 96 cores really do oversubscribe by 2x on the
    CPU-bound parts and that contention is what L5 exists to remove.

    The LANES are not.  `RASBERY_BATCH_HOST_THREADS` is the number of Driver
    refill lanes, and a lane is not a CPU worker: it spends nearly all of its
    life blocked on the GPU rendezvous, and the arena can only gather the lanes
    that are inside it.  So the default is the EXECUTABLE's own default --
    min(batch_width, jobs), main.cpp:698 -- i.e. one lane per arena slot,
    regardless of how many cores the process was given.  That is what the raw
    238 production line runs (64 lanes on 24 cores, 582 c/h), and capping the
    lanes at the core count instead is what made this dispatcher's own control
    arm measure 115.6 c/h with the same binary and the same decks: 24 lanes
    cannot fill 64 slots, so `width_fill` collapsed to 0.03.

    What keeps 64 lanes on 24 cores from exploding is OMP_MAX_ACTIVE_LEVELS=1
    (DEFAULT_ENV): the lanes do not spawn nested solver teams, so the thread
    count is the lane count and the lanes are mostly blocked.

    `oversubscribe=False` (--no-oversubscribe) restores the old CPU-capped
    policy as a deliberate arm, and `driver_workers` states a count outright.
    `RASBERY_OMP_THREADS` defaults to the arena width per process -- the
    reference's 64 at width 64 -- rather than to a share of the cores.
    """
    n_gpus = len(gpus)
    procs_per_gpu = max(1, procs_per_gpu)
    processes = n_gpus * procs_per_gpu
    cpus_per_proc = max(1, visible_cpus // processes)
    writer, writer_policy = plan_writer_threads(
        result_mode=result_mode, io_writer_mode=io_writer_mode,
        observed=writer_threads_observed,
    )
    if driver_workers:
        workers, policy = driver_workers, "explicit"
    elif oversubscribe:
        # main.cpp caps at min(batch_width, jobs) itself; the chunk size is not
        # known here, so declare the width and let the executable do the rest.
        workers, policy = batch_width, "binary_default_width"
    else:
        workers, policy = min(batch_width, cpus_per_proc), "no_oversubscribe_cpu_capped"
    workers = max(1, min(workers, batch_width))
    if solver_threads:
        threads = solver_threads
    elif oversubscribe:
        # The reference sets OMP_NUM_THREADS = OMP_THREAD_LIMIT =
        # RASBERY_OMP_THREADS = the arena width (tools/ga_two_stage_40x_pipeline
        # .py:209-212 does the same).  One active level makes it a ceiling, not
        # a multiplier.
        threads = batch_width
    else:
        # What is left of this process's share, after the writer pool, is what
        # the per-Driver solver regions may spend.  At least 1: a zero would be
        # a silently serial solver.  The writer term is now the REAL one
        # (0 or 1, not the plan's 8), which is why this arm no longer collapses
        # to the floor on a narrow per-process core share.
        spare = max(0, cpus_per_proc - workers - writer)
        threads = max(1, min(3, 1 + spare // max(1, workers)))

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
        writer_threads=writer,
        cpu_sets=sets,
        pin=pin,
        worker_policy=policy,
        writer_policy=writer_policy,
        result_mode=result_mode or "full",
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
    extra_process_gb: float
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
            "extra_process_gb": round(self.extra_process_gb, 3),
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
    extra_process_gb: float = VRAM_GB_PER_EXTRA_PROCESS,
    device_memory_gb: float | None = None,
    probe=query_device_memory_gb,
) -> VramPlan:
    """What K processes of width W will hold on each listed device.

    TWO TERMS, and the second one is the 238 correction.  A process's arenas are
    sized for its OWN declared width, so the SLOT term is K x W x per-slot and
    holding K x W fixed does hold it fixed -- that part of the lever's promise
    measured true.  But each process also pays a fixed cost that has nothing to
    do with the width (CUDA context, module images, library handles, allocator
    pool), and the 238 matrix prices it at 2.56 GB per additional process:
    12.3 / 14.9 / 20.0 / 30.2 GB at K = 1 / 2 / 4 / 8 with K x W = 64 throughout.

    The guard still exists for the arm that changes both (`4 x M64` rather than
    `4 x M16`) -- the easy thing to type and the one that dies at arena
    stand-up, after the queue has been claimed -- but it now also catches the
    arm that keeps the declared width and adds processes until the contexts
    alone fill the device.
    """
    per_process = batch_width * per_slot_gb
    per_device = procs_per_gpu * per_process + max(0, procs_per_gpu - 1) * extra_process_gb
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
        extra_process_gb=extra_process_gb,
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


def fair_share(*, index: int, processes: int, jobs: int) -> int:
    """This worker's share of *jobs* under `--claim auto`: floor, then remainder.

    floor(jobs / processes) each, and the remainder handed out ROUND-ROBIN by
    global process index, so the shares tile the manifest exactly and every
    process gets at least one job whenever there are at least as many jobs as
    processes.
    """
    processes = max(1, processes)
    base, remainder = divmod(max(0, jobs), processes)
    return base + (1 if index < remainder else 0)


def auto_claim_size(*, index: int, processes: int, jobs: int, remaining: int,
                    batch_width: int) -> int:
    """How many jobs this worker claims next under `--claim auto`.

    THE DEFECT THIS FIXES (238, 12 x M6).  The size was
    `max(batch_width, ceil(remaining / (2 * processes)))` -- a width floor over a
    halved share.  With 64 jobs across 12 processes of width 6 that floor is 6
    while the fair share is 5.33, so the first ten workers took 60, the eleventh
    took 4 and the TWELFTH GOT NOTHING.  A process that never claims a job
    starts, prints its receipts, measures a wall of pure startup and reports 0
    cases/hour, which is not a datum about width -- and at K=12 it is a twelfth
    of the device's declared slots standing idle for the whole wave.

    So the fair share is a CEILING on the claim, not a replacement for the
    policy: a worker never takes more than floor(jobs/processes) (+1 for the
    first `jobs % processes` workers), which is exactly enough to make the
    shares tile and nobody starve.  The halving stays UNDER that ceiling and is
    what leaves work to steal on a long manifest -- with 128 jobs at K=8 the
    claim is 8 (sixteen chunks, refill measured, docs/W4_L5 Sec 4.6), where a
    plain fair share would hand each worker 16 in one go and there would be no
    refill left in the arm at all.
    """
    processes = max(1, processes)
    share = fair_share(index=index, processes=processes, jobs=jobs)
    halved = max(batch_width, -(-max(0, remaining) // (2 * processes)))
    if share <= 0:
        # More processes than jobs: nobody is owed a share, and whoever asks
        # first may have what is left.  Refusing here would leave jobs unrun.
        return max(1, halved)
    return max(1, min(halved, share))


def rolling_prefetch(overrides: dict[str, str], extra_env: dict[str, str]) -> int:
    """How many cases beyond the arena width stay outstanding, in the CHILD's terms.

    Read from the resolved environment rather than from an argument of its own,
    because the evaluator reads the same variable and the two must not be able
    to disagree: the dispatcher keeps `width + prefetch` sent, the evaluator
    sizes its queue from the same number, and a private flag here would let one
    of them be tuned without the other.
    """
    raw = overrides.get("RASBERY_EVALUATOR_ROLLING_PREFETCH") or extra_env.get(
        "RASBERY_EVALUATOR_ROLLING_PREFETCH") or os.environ.get(
        "RASBERY_EVALUATOR_ROLLING_PREFETCH")
    try:
        value = int(raw) if raw is not None else 2
    except ValueError:
        value = 2
    return max(0, value)


# ---------------------------------------------------------------------------
# The persistent evaluator (WP8 stage 1.5)
# ---------------------------------------------------------------------------
#
# WHAT CHANGES.  Nothing about what a case computes, and nothing about how the
# queue is claimed.  What changes is WHAT A WORKER IS: it was a SEQUENCE of
# RASBERY processes, one per claimed chunk; it becomes ONE long-lived
# `RASBERY --evaluator-jsonl -` that is fed each chunk as a `wave` request on
# stdin.  docs/WP8_EVALUATOR_STAGE1 Sec 0.1 is explicit about the size of that
# lever and about what it is NOT: `outside_drive` (the process image, the
# loader, CUDA context creation and CUDA teardown) is 1.75-4.92 s and is paid
# ONCE PER PROCESS, so a worker that ran C chunks paid it C times and now pays
# it once.  Per case it is 0 %.  The whole value is in how many process
# boundaries a campaign crosses, and that count is what this converts from
# "chunks" to "workers".
#
# WHY IT IS THE DEFAULT.  The chunked path pays a cost that buys nothing: the
# arena stand-up, the graph capture and the 34 MB library parse are identical
# across the chunks of ONE worker, and the process boundary between them exists
# only because the launcher had no way to say "another wave, same process".
# `--no-evaluator` keeps the old shape as a NAMED arm -- it is the wall control
# the WP8 gate is measured against, and it is the fallback for a build with no
# evaluator mode.
#
# WHAT IS STILL A PROCESS PROPERTY, AND THEREFORE NOW A WORKER PROPERTY.  The
# fidelity ([RASBERY][PHYSICS_MODE], resolved once from the environment before
# the first request), the arena WIDTH (one allocation, latched by the first
# wave), the I/O writer mode and the host-pinning decision.  A campaign that
# wanted two fidelities would need two evaluators, not two waves -- which is why
# `--strict` and the declared-fidelity audit are unchanged here: they were
# always per-process statements, and a worker is now a process.

EVALUATOR_READY = re.compile(r"\[RASBERY\]\[EVALUATOR\]\[READY\]\s*(\{.*\})")
EVALUATOR_WAVE_RECEIPT = re.compile(r"\[RASBERY\]\[EVALUATOR\]\[WAVE\]\s*(\{.*\})")
EVALUATOR_CASE_RECEIPT = re.compile(r"\[RASBERY\]\[EVALUATOR\]\[CASE\]\s*(\{.*\})")
EVALUATOR_REFUSED = re.compile(r"\[RASBERY\]\[EVALUATOR\]\[REFUSED\]\s*(\{.*\})")
# WP19.  The line EvaluatorServer prints for a case that DIED, with the message
# on it.  Parsed on its own rather than lifted out of the [CASE] receipt because
# a case that dies before the fold closes may print a receipt this dispatcher
# cannot parse -- and the whole point of the line is that it survives that.
EVALUATOR_CASE_ERROR = re.compile(r"\[RASBERY\]\[EVALUATOR\]\[ERROR\]\s*(\{.*\})")
# WP19.1.  The capture-race evidence, which WP19 counted but never surfaced.
#
# `capture_race_retry` lived ONLY in the arbiter's once-per-process teardown
# receipt, which this dispatcher did not read: the 238 run of 2026-08-30 printed
# `"capture_race_retry":1` in both the process that lost a case and the process
# that did not, in a per-worker evaluator log, and the harness log contained the
# term nowhere at all.  Three lines are lifted now -- the arbiter receipt, the
# per-event capture-race lines the backends print, and the evaluator's own
# first-case retry -- so a campaign that fired the race says so on its own log.
CUDA_CAPTURE_ARBITER = re.compile(r"\[RASBERY\]\[CUDA\]\[CAPTURE_ARBITER\]\s*(\{.*\})")
CAPTURE_RACE_EVENT = re.compile(
    r"\[RASBERY\]\[(?:CUDA|EVALUATOR)\]\[CAPTURE_RACE\](?:\[[A-Z_]+\])?\s*(\{.*\})")
# `[EVALUATOR] {` with WHITESPACE after the tag -- the once-per-process receipt.
# The wave/case/refused tags are each followed by `[`, so this cannot match one.
EVALUATOR_PROCESS = re.compile(r"\[RASBERY\]\[EVALUATOR\]\s+(\{.*\})")
# WP19.2.  The worker's own crash record, printed by src/CrashReport.h's signal
# handler straight to fd 2 -- which this dispatcher already merges into the
# child's pipe (`stderr=subprocess.STDOUT`), so no new plumbing carries it.
# Block 38's two SIGSEGVs produced a FATAL record whose entire content was
# `completed:0`; these three patterns are what turns the next one into a case
# id, a lane, a slot, a phase and a frame list.
CRASH_RECORD = re.compile(r"\[RASBERY\]\[CRASH\]\s*(\{.*\})")
CRASH_FRAME = re.compile(r"\[RASBERY\]\[CRASH\]\[FRAME\]")
CRASH_END = re.compile(r"\[RASBERY\]\[CRASH\]\[END\]\s*(\{.*\})")

#: How many trailing lines of a dead child's output the FATAL record keeps.
#: Big enough for a 64-frame backtrace plus the record that introduces it, and
#: small enough that a chunk full of [WARN][th] lines cannot bury it.
CRASH_TAIL_LINES = 120


def crash_evidence(text: str) -> dict:
    """What a dead child said on its way out, as a JSON-able dict.

    THREE FIELDS, AND WHY NOT ONE.  `crash` is the structured record the handler
    printed (case, lane, slot, phase, deck, capture_open) -- the thing an
    analyst greps for.  `backtrace` is the frame list, kept verbatim because a
    symbol line is evidence and reformatting it loses the offsets.
    `stderr_tail` is the raw tail REGARDLESS of whether a crash record exists,
    because the failure mode this exists to close is precisely the one where
    the child said something nobody kept -- an OOM killer note, a CUDA error, a
    glibc `free(): invalid pointer`, or nothing at all, which is itself a fact.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    tail = lines[-CRASH_TAIL_LINES:]
    records: list[dict] = []
    for match in CRASH_RECORD.finditer(text):
        record = _json_or_none(match.group(1))
        if record is not None:
            records.append(record)
    frames: list[str] = []
    capturing = False
    for line in lines:
        if CRASH_END.search(line):
            capturing = False
            continue
        if CRASH_FRAME.search(line):
            capturing = True
            continue
        if capturing:
            frames.append(line)
    evidence: dict = {"stderr_tail": tail}
    if records:
        evidence["crash"] = records
    if frames:
        evidence["backtrace"] = frames
    return evidence


def _json_or_none(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def collect_case_errors(result: "WorkerResult", text: str) -> None:
    """WP19.  Lift every [EVALUATOR][ERROR] line out of a child's output.

    CALLED WHEREVER `fail_lines` IS COUNTED, and for the same reason: this is
    the dispatcher's one pass over raw child text, so a message that exists at
    all is caught here whether or not the case's [CASE] receipt parsed, whether
    or not the child then died, and whether the text came from a wave, a
    rolling session or the shutdown epilogue.

    An error already recorded is not recorded twice -- a wave's text is scanned
    once by the streaming reader and once again at the end -- so the FAIL line
    reports five dead cases as five lines and not as ten.
    """
    for match in EVALUATOR_CASE_ERROR.finditer(text):
        record = _json_or_none(match.group(1))
        if record is None:
            entry = "unparseable [EVALUATOR][ERROR]: " + match.group(1)[:200]
        else:
            entry = "%s: %s" % (
                record.get("deck") or record.get("output") or "?",
                record.get("error") or "no message",
            )
        if entry not in result.failed_case_errors:
            result.failed_case_errors.append(entry)
    collect_capture_race(result, text)


def collect_capture_race(result: "WorkerResult", text: str) -> None:
    """WP19.1.  Lift the capture-race evidence a child prints.

    The arbiter receipt is a SNAPSHOT (last one wins -- it is cumulative and
    printed at teardown); the per-event lines are a SET, deduplicated the same
    way the case errors are, because the dispatcher scans a wave's text twice.
    """
    for match in CUDA_CAPTURE_ARBITER.finditer(text):
        record = _json_or_none(match.group(1))
        if record is not None:
            result.capture_arbiter = record
    for match in CAPTURE_RACE_EVENT.finditer(text):
        entry = match.group(1)[:400]
        if entry not in result.capture_race:
            result.capture_race.append(entry)


@dataclass
class WaveOutcome:
    """One `wave` request and everything the evaluator said about it."""

    text: str = ""
    cases: list[dict] = field(default_factory=list)
    receipt: dict | None = None
    refused: list[dict] = field(default_factory=list)
    #: False means the child did not survive the wave.  The cases it never
    #: reported are the ones nobody can account for, and they are exactly what
    #: gets re-queued once.
    alive: bool = True
    returncode: int | None = None


class EvaluatorSession:
    """One long-lived RASBERY evaluator, and its restarts.

    The session owns the pipe protocol and NOTHING about the queue: it is handed
    a manifest path and a wave id and returns what the evaluator said.  That
    split is deliberate -- the chunk accounting, the receipt audit and the
    re-queue decision stay in run_worker, where the chunked path already put
    them, so both modes are audited by the same code and their numbers are
    comparable.
    """

    def __init__(self, *, command: Sequence[str], env: dict[str, str], cwd: str | None,
                 log_path: Path, max_restarts: int = 3) -> None:
        self._command = list(command)
        self._env = dict(env)
        self._cwd = cwd
        self.log_path = log_path
        self.max_restarts = max(0, int(max_restarts))
        self._proc: subprocess.Popen | None = None
        self._sink = open(log_path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
        #: Everything the CURRENT child printed before it was ready for work --
        #: [PHYSICS_MODE], [IO_WRITER], [GPU_FULL] and friends.  The per-wave
        #: audit needs it: those receipts are printed ONCE PER PROCESS, so a
        #: wave slice on its own would read as a run that declared no fidelity.
        self.preamble = ""
        self.epilogue = ""
        self.starts = 0
        self.restarts = 0
        self.returncode: int | None = None
        self.process_receipts: list[dict] = []
        self.ready_receipts: list[dict] = []

    # -- lifecycle ---------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        """The live child's pid, for a caller that has to sample its RSS.

        WP11's soak reads /proc/<pid>/status between generations, and a leak is
        a slope over a session that RESTARTS: each restart is a new pid, so the
        caller has to re-read this rather than cache it, which is why it is a
        property and not a constructor argument.
        """
        return self._proc.pid if self._proc is not None else None

    def start(self) -> bool:
        """Stand a child up and read it to its [READY] line.  False if it died."""
        self._note("start", {"command": list(self._command), "attempt": self.starts + 1})
        try:
            self._proc = subprocess.Popen(  # noqa: S603
                self._command, env=self._env, cwd=self._cwd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
        except OSError as exc:
            self._proc = None
            self.returncode = 127
            self._note("start_failed", {"error": str(exc)})
            return False
        self.starts += 1
        text, died = self._pump(lambda line: EVALUATOR_READY.search(line) is not None)
        self.preamble = text
        for match in EVALUATOR_READY.finditer(text):
            receipt = _json_or_none(match.group(1))
            if receipt is not None:
                self.ready_receipts.append(receipt)
        if died:
            self._reap()
            return False
        return True

    def restart(self) -> bool:
        """Replace a dead child.  False when the restart budget is spent."""
        self._reap()
        if self.restarts >= self.max_restarts:
            self._note("restart_refused", {"restarts": self.restarts,
                                           "max_restarts": self.max_restarts})
            return False
        self.restarts += 1
        return self.start()

    def close(self) -> str:
        """Shut the child down and read its teardown receipts to EOF.

        The teardown receipts are the ONLY place [CUDA][BATCH_OCCUPANCY] and the
        process-lifetime counters appear in this mode -- the arena is released
        once, at shutdown, which is the whole point of the mode -- so a session
        that was not closed has no width_fill and no xslib_loads to report.
        """
        if self.alive:
            self._send('{"op":"shutdown"}')
            try:
                if self._proc is not None and self._proc.stdin is not None:
                    self._proc.stdin.close()
            except OSError:
                pass
            text, _died = self._pump(lambda _line: False)
            self.epilogue += text
            for match in EVALUATOR_PROCESS.finditer(text):
                receipt = _json_or_none(match.group(1))
                if receipt is not None:
                    self.process_receipts.append(receipt)
        self._reap()
        try:
            self._sink.close()
        except OSError:
            pass
        return self.epilogue

    # -- one wave ----------------------------------------------------------
    def wave(self, *, wave_id: int, manifest: str | None = None,
             result_mode: str | None = None,
             cases: Sequence[dict] | None = None,
             fidelity: dict | None = None) -> WaveOutcome:
        """One wave: a manifest, a list of inline case requests, or both.

        WHY INLINE CASES EXIST HERE.  A manifest line is a deck, an output and a
        result mode -- it cannot carry a warm-start parent, a per-case fidelity
        or a promotion link, and after WP10.3 those are exactly what a mixed
        wave is made of.  The dispatcher's own chunked path still sends
        manifests (it is auditing throughput, not fidelity) and WP11's soak
        sends inline cases, off ONE session class, so the two are audited by the
        same code and their numbers stay comparable.
        """
        out = WaveOutcome()
        if not self.alive:
            out.alive = False
            out.returncode = self.returncode
            return out
        # The evaluator collects `op:case` lines and runs them when the wave
        # line arrives, so the cases go FIRST and a failure to write one aborts
        # before the wave is asked for -- a half-sent generation must not run.
        for case in cases or ():
            payload = dict(case)
            payload.setdefault("op", "case")
            if not self._send(json.dumps(payload, separators=(",", ":"))):
                out.text, _ = self._pump(lambda _line: False)
                self._reap()
                out.alive = False
                out.returncode = self.returncode
                return out
        request: dict[str, object] = {"op": "wave", "wave_id": wave_id}
        if manifest:
            request["jobs_manifest"] = manifest
        if result_mode:
            request["result_mode"] = result_mode
        if fidelity:
            request.update(fidelity)
        if not self._send(json.dumps(request, separators=(",", ":"))):
            out.text, _ = self._pump(lambda _line: False)
            self._reap()
            out.alive = False
            out.returncode = self.returncode
            return out

        def done(line: str) -> bool:
            match = EVALUATOR_WAVE_RECEIPT.search(line)
            if match is not None:
                receipt = _json_or_none(match.group(1))
                # An unparseable receipt stops the read instead of hanging: a
                # dispatcher that waits forever for a line it cannot recognise
                # is worse than one that fails the chunk it could not read.
                return receipt is None or receipt.get("wave_id") == wave_id
            # A REFUSED wave never produces a WAVE receipt (EvaluatorServer.h
            # returns before running one), so it is a stop condition too.
            return EVALUATOR_REFUSED.search(line) is not None

        text, died = self._pump(done)
        out.text = text
        for match in EVALUATOR_CASE_RECEIPT.finditer(text):
            case = _json_or_none(match.group(1))
            if case is not None and not case.get("isolation_check"):
                out.cases.append(case)
        for match in EVALUATOR_WAVE_RECEIPT.finditer(text):
            receipt = _json_or_none(match.group(1))
            if receipt is not None:
                out.receipt = receipt
        for match in EVALUATOR_REFUSED.finditer(text):
            refused = _json_or_none(match.group(1))
            out.refused.append(refused if refused is not None else {"what": match.group(1)})
        if died:
            self._reap()
            out.alive = False
            out.returncode = self.returncode
        return out

    # -- the rolling protocol (WP18) ---------------------------------------
    #
    # WHAT IS DIFFERENT FROM `wave`.  Nothing about the pipe, and everything
    # about WHEN.  `wave()` sends a whole chunk and then reads until the chunk's
    # receipt: the child cannot be given more work while that read is running,
    # which is exactly the barrier the arena does not need.  These two let the
    # caller interleave -- send a case, read a completion, send the next case --
    # so the queue is topped up FROM INSIDE the read loop and the arena never
    # drains.  They are deliberately thin: every decision about what to send and
    # when to stop stays in run_worker, where the claim accounting lives.
    def send_case(self, payload: dict) -> bool:
        """One `{"op":"case"}` line.  False means the pipe is gone."""
        request = dict(payload)
        request.setdefault("op", "case")
        return self._send(json.dumps(request, separators=(",", ":")))

    def send_request(self, payload: dict) -> bool:
        return self._send(json.dumps(payload, separators=(",", ":")))

    def pump_until(self, done) -> tuple[str, bool]:
        """Read child stdout until *done(line)* or EOF.  (text, died).

        *done* MAY WRITE TO THE CHILD.  That is the point -- the rolling top-up
        happens inside it -- and it is safe in exactly one direction: the child
        drains its stdin on a dedicated thread while the lanes run, so a write
        here cannot block behind work.  The reverse (a client that writes
        without reading) is what the evaluator's bounded queue exists to turn
        into backpressure instead of a wedge.
        """
        return self._pump(done)

    # -- plumbing ----------------------------------------------------------
    def _send(self, line: str) -> bool:
        self._note("request", {"line": line})
        try:
            if self._proc is None or self._proc.stdin is None:
                return False
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):
            return False
        return True

    def _pump(self, done) -> tuple[str, bool]:
        """Read child stdout until *done(line)* or EOF.  Returns (text, died).

        EOF IS THE DEATH SIGNAL and it needs no timeout: the child's stdout is
        this dispatcher's pipe, so a child that exits -- cleanly, by signal, or
        by a CUDA abort -- closes it, and readline returns "".
        """
        chunks: list[str] = []
        if self._proc is None or self._proc.stdout is None:
            return "", True
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                return "".join(chunks), True
            chunks.append(line)
            try:
                self._sink.write(line)
                self._sink.flush()
            except OSError:
                pass
            if done(line):
                return "".join(chunks), False

    def _note(self, what: str, payload: dict) -> None:
        """A dispatcher-side annotation in the worker log, tagged so it can
        never be mistaken for one of the executable's own receipts."""
        try:
            self._sink.write("[RASBERY][MULTI_GPU][EVALUATOR][" + what.upper() + "] "
                             + json.dumps(payload, separators=(",", ":"), default=str) + "\n")
            self._sink.flush()
        except OSError:
            pass

    def _reap(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            self.returncode = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - a wedged child
            proc.kill()
            self.returncode = proc.wait()
        self._note("exit", {"returncode": self.returncode})


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
    #: WP18.  `[RASBERY][REFILL][ROLLING]` -- one per rolling session.  Empty in
    #: every other mode, which is what keeps `--claim auto` numbers comparable
    #: to every measurement taken before this work package.
    rolling_receipts: list[dict] = field(default_factory=list)
    occupancy_receipts: list[dict] = field(default_factory=list)
    wall_s: float = 0.0
    # The env this worker's children were actually launched with, on top of the
    # inherited one.  Carried into the [PROC] receipt: an environment mismatch
    # is the one defect that shows up as a THROUGHPUT number and nothing else,
    # so the run has to say what it ran with, not what it meant to.
    env: dict[str, str] = field(default_factory=dict)
    # The fidelity this worker's children were told to solve at, and what each
    # chunk's [PHYSICS_MODE] receipt actually said.  One entry per chunk (the
    # receipt is printed once per process, before any deck starts, so a chunk IS
    # the finest grain there is), carrying the job count so a campaign table can
    # be labelled with the fidelity its cases were measured at.
    declared_fidelity: str = "strict"
    case_fidelity: list[dict] = field(default_factory=list)
    # WP8 stage 1.5.  `processes` counts RASBERY IMAGES; `waves` counts claimed
    # chunks.  In the chunked mode they are equal BY CONSTRUCTION, and that
    # equality is exactly the cost the evaluator removes -- so they have to be
    # counted apart, or the receipt cannot show the removal.
    evaluator: bool = False
    waves: int = 0
    restarts: int = 0
    #: The [RASBERY][EVALUATOR] process receipt(s) this worker's children
    #: printed at shutdown: xslib_loads, cohort_builds/cohort_hits, the slot
    #: tenancy counters, stop_reason.
    evaluator_receipts: list[dict] = field(default_factory=list)
    #: One entry per wave whose child DIED.  Names the cases nobody accounted
    #: for, so a lost candidate is a line in a receipt instead of a silent gap
    #: between the manifest and the outputs.
    fatal_waves: list[dict] = field(default_factory=list)
    #: Decks whose [EVALUATOR][CASE] receipt said `failed`, plus the ones no
    #: receipt ever mentioned once the single re-queue was spent.
    failed_cases: list[str] = field(default_factory=list)
    #: WP10.8.  Cases that were re-queued onto a fresh child after their worker
    #: died and DID report the second time.  A recovered case is not a lost one
    #: and must not read as a failure -- but it ran on a cold process, so its
    #: wall time is not comparable and a campaign has to be able to see which
    #: ones they were.
    restart_recovered: list[str] = field(default_factory=list)
    #: WP19.  "<deck>: <message>" for every case that died with a message.
    #:
    #: THE DEFECT THIS CLOSES.  `failed_cases` is a list of NAMES, and it was
    #: the only thing the dispatcher kept out of a failed [CASE] receipt.  A
    #: run that lost five decks to the capture race therefore printed five deck
    #: names and no error text ANYWHERE in the harness log -- the message
    #: ("cudaGetLastError(): operation not permitted when stream is capturing")
    #: existed, in the child's receipt, in a per-worker log file nobody opens
    #: until they already know what to look for.  These strings go into the
    #: problems list, which is what reaches [MULTI_GPU][FAIL].
    failed_case_errors: list[str] = field(default_factory=list)
    #: WP19.1.  The arbiter's once-per-process teardown receipt (cumulative;
    #: the last one seen wins) and every per-event capture-race line the child
    #: printed.  Both go into the PROC receipt so `capture_race_retry` is a term
    #: a campaign log carries rather than one a per-worker file hides.
    capture_arbiter: dict | None = None
    capture_race: list[str] = field(default_factory=list)

    @property
    def fidelities(self) -> dict[str, int]:
        """jobs measured, keyed by the policy word the run printed."""
        counts: dict[str, int] = {}
        for entry in self.case_fidelity:
            key = entry.get("policy") or "unreported"
            counts[key] = counts.get(key, 0) + int(entry.get("jobs", 0))
        return counts

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
    def rolling_admits(self) -> int:
        return sum(int(x.get("admits", 0)) for x in self.rolling_receipts)

    @property
    def rolling_immediate_fraction(self) -> float:
        """immediate_admits / admits -- was the queue ahead of the arena?

        This is the number that separates the two ways `--claim rolling` can
        fail to beat `--claim auto`.  Near 1 with no throughput gain means the
        arena was kept full and the GPU is the wall.  Below 1 means the LANES
        waited for the dispatcher's round trip, which is a prefetch that is too
        small and not an arena fact at all.
        """
        admits = self.rolling_admits
        if admits <= 0:
            return 0.0
        return sum(int(x.get("immediate_admits", 0))
                   for x in self.rolling_receipts) / admits

    @property
    def rolling_width_fill(self) -> float:
        """The arena occupancy the LEDGER saw, sampled at every admit/finish.

        Reported beside `width_fill` (the rendezvous one) and never instead of
        it: they answer different questions.  `width_fill` is how many slots a
        CMFD launch actually gathered; this is how many slots held a deck at
        all.  The first cannot exceed the second, and the gap between them is
        host skew rather than scheduling.
        """
        fills = [float(x.get("width_fill", 0.0)) for x in self.rolling_receipts]
        return sum(fills) / len(fills) if fills else 0.0

    @property
    def rolling_tail_idle_s(self) -> float:
        return sum(float(x.get("tail_idle_ms", 0.0))
                   for x in self.rolling_receipts) / 1000.0

    @property
    def cases_per_hour(self) -> float:
        return 3600.0 * self.jobs / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def evaluator_totals(self) -> dict:
        """The process receipts folded into one dict; {} in chunked mode.

        SUMMED, not merged: a worker that was restarted has more than one
        process receipt, and two images that each loaded the library really did
        load it twice -- which is the whole point of watching `xslib_loads`.
        `stop_reason` keeps the LAST child's word, because that is the one that
        says how the worker ended.
        """
        if not self.evaluator_receipts:
            return {}
        summed = ("cases", "ok", "failed", "refused", "generations", "xslib_loads",
                  "xslib_hits", "geometry_builds", "cohort_builds", "cohort_hits",
                  "arena_releases", "arena_standups", "slot_admissions",
                  "slot_duplicates", "slot_stale_tenants", "slot_double_releases",
                  "isolation_checks", "isolation_mismatches")
        out: dict = {}
        for key in summed:
            values = [r.get(key) for r in self.evaluator_receipts
                      if isinstance(r.get(key), int)]
            if values:
                out[key] = sum(values)
        out["images"] = len(self.evaluator_receipts)
        out["stop_reason"] = self.evaluator_receipts[-1].get("stop_reason")
        return out


def _run_rolling_worker(
    *,
    result: "WorkerResult",
    queue: Queue,
    jobs: list[tuple[str, str, str]],
    index: int,
    budget: HostBudget,
    batch_width: int,
    prefetch: int,
    evaluator_command: Sequence[str],
    env: dict[str, str],
    cwd: str | None,
    workdir: Path,
    stem: str,
    gpu: str,
    proc: int,
    result_mode: str | None,
    declared_fidelity: str,
    evaluator_max_restarts: int,
) -> EvaluatorSession | None:
    """`--claim rolling`: one worker, one session, one barrier.

    WHAT THE CHUNKED PATH DOES, AND WHY IT COSTS.  `auto` claims `batch_width`
    jobs (auto_claim_size: 128 jobs / 8 workers / width 8 -> 8), sends them as
    ONE wave, and cannot claim again until the evaluator has answered -- which
    it does only after the slowest of the eight finishes.  So every chunk is one
    case per lane, no lane ever refills, and the arena drains sixteen times over
    a 128-job manifest.  Measured on 238: width_fill 0.447, tail_idle_max 155 s
    of a 390 s wall at M8; 0.367 and 356 s of 373 s at M16.

    WHAT THIS DOES INSTEAD.  Claim ONE job at a time, keep `batch_width +
    prefetch` of them outstanding, and re-claim from inside the read loop the
    moment a case reports.  The evaluator admits each one into whichever lane
    just went free (RASBERY_EVALUATOR_ROLLING=1), so a case that finishes early
    does not leave a slot empty until its wave ends.  One barrier remains, at
    the very end, and it is where the wave-shaped receipts are printed.

    WHY IT CLAIMS ONE AT A TIME.  A claim is a flock'd read-modify-write over a
    file, it costs microseconds, and the whole reason `auto` claims in chunks is
    to amortise a PROCESS START that this mode does not pay.  Claiming one keeps
    the queue steal-able to the last job, which is the property that made
    `--claim auto` beat `--claim all` in the first place.

    NO RESTARTS.  A chunked worker can lose a chunk and re-queue it; a rolling
    worker's outstanding set is spread across the arena, so a death loses up to
    `batch_width + prefetch` cases at once.  Rather than re-queue a set that
    large into a process that has just proved it can die, the worker stops
    claiming and the rest of the manifest is left to the workers that are still
    up -- the same decision `run_worker` already makes when restarts run out.
    """
    # The first claim comes BEFORE the process, for the reason the chunked path
    # stands its session up lazily: a worker with nothing to claim must start no
    # process at all, or it measures its own startup and reports 0 cases/hour.
    start, end = queue.claim(1, index)
    if start >= end:
        return None

    session = EvaluatorSession(
        command=list(evaluator_command), env=env, cwd=cwd,
        log_path=workdir / f"{stem}.evaluator.log",
        max_restarts=evaluator_max_restarts,
    )
    if not session.start():
        result.problems.append(
            "gpu%s p%d: the evaluator never reached [READY] (rc=%r); see %s"
            % (gpu, proc, session.returncode, session.log_path)
        )
        if result.returncode == 0:
            result.returncode = session.returncode or 1
        result.jobs += 1
        result.failed_cases.append(jobs[start][0])
        return session

    ready = session.ready_receipts[-1] if session.ready_receipts else {}
    if not ready.get("rolling"):
        # FAIL LOUD.  A binary without WP18 answers a `case` line by COLLECTING
        # it and running nothing until a `wave` line arrives, so this loop would
        # send one case, wait forever for a receipt, and be killed as a hang.
        # An operator who set --claim rolling and got wave scheduling would
        # publish the wave arm's number under the rolling arm's name.
        result.problems.append(
            "gpu%s p%d: --claim rolling needs an evaluator that reports "
            '"rolling":true in its [READY] receipt (RASBERY_EVALUATOR_ROLLING=1 '
            "and a binary that has WP18); this one reported %r"
            % (gpu, proc, ready.get("rolling"))
        )
        if result.returncode == 0:
            result.returncode = 2
        return session

    target = max(1, batch_width + max(0, prefetch))
    #: output path -> the manifest job that named it.  Keyed on the output
    #: because that is what the evaluator echoes back and what the manifest
    #: guarantees is unique.
    outstanding: dict[str, tuple[str, str, str]] = {}
    reported: set[str] = set()
    cases: list[dict] = []
    exhausted = False
    pipe_ok = True

    def send(job: tuple[str, str, str]) -> bool:
        deck, output, mode = job
        payload: dict[str, object] = {"op": "case", "deck": deck, "output": output,
                                      "key": output}
        effective = mode or result_mode
        if effective:
            payload["result_mode"] = effective
        if not session.send_case(payload):
            return False
        outstanding[output] = job
        result.jobs += 1
        return True

    def top_up() -> None:
        nonlocal exhausted, pipe_ok
        while pipe_ok and not exhausted and len(outstanding) < target:
            lo, hi = queue.claim(1, index)
            if lo >= hi:
                exhausted = True
                return
            if not send(jobs[lo]):
                pipe_ok = False
                # The claim is spent and the case was never sent: it is lost
                # unless it is named, and a lost candidate has to be a line in a
                # receipt rather than a gap between the manifest and the outputs.
                result.jobs += 1
                result.failed_cases.append(jobs[lo][0])
                return

    if not send(jobs[start]):
        result.problems.append("gpu%s p%d: could not write the first case" % (gpu, proc))
        result.jobs += 1
        result.failed_cases.append(jobs[start][0])
        return session
    top_up()

    refused: list[dict] = []

    def on_line(line: str) -> bool:
        # A REFUSAL ENDS THE READ.  A refused request never produces a CASE
        # receipt, so `outstanding` would never empty and this pump would block
        # until the campaign's own timeout killed a child that was perfectly
        # healthy and had told us exactly what was wrong.
        refusal = EVALUATOR_REFUSED.search(line)
        if refusal is not None:
            refused.append(_json_or_none(refusal.group(1)) or {"what": refusal.group(1)})
            return True
        match = EVALUATOR_CASE_RECEIPT.search(line)
        if match is not None:
            case = _json_or_none(match.group(1))
            if case is not None and not case.get("isolation_check"):
                output = case.get("output")
                if output in outstanding and output not in reported:
                    reported.add(output)
                    cases.append(case)
                    del outstanding[output]
                    # THE REFILL, from this side: a completion is a claim.
                    top_up()
        return exhausted and not outstanding

    text, died = session.pump_until(on_line)

    # The one barrier.  It is what makes the evaluator print the wave-shaped
    # receipts -- [BATCH_HOST], [REFILL], [REFILL][ROLLING], [EVALUATOR][WAVE] --
    # that the post-run audit below and every campaign table are built on.
    wave_id = 1
    if not died and session.alive and not refused:
        if session.send_request({"op": "wave", "wave_id": wave_id}):
            barrier, died = session.pump_until(
                lambda line: EVALUATOR_WAVE_RECEIPT.search(line) is not None
                or EVALUATOR_REFUSED.search(line) is not None
            )
            text += barrier

    for match in EVALUATOR_CASE_RECEIPT.finditer(text):
        case = _json_or_none(match.group(1))
        if case is not None and not case.get("isolation_check"):
            if case.get("status") == "failed":
                result.failed_cases.append(str(case.get("deck")))
    for match in EVALUATOR_REFUSED.finditer(text):
        refusal = _json_or_none(match.group(1)) or {"what": match.group(1)}
        result.problems.append(
            "gpu%s p%d: the evaluator REFUSED a rolling request: %s"
            % (gpu, proc, refusal.get("what"))
        )
    if outstanding:
        result.failed_cases.extend(job[0] for job in outstanding.values())
        result.problems.append(
            "gpu%s p%d: %d case(s) were never reported (child alive=%s); see %s"
            % (gpu, proc, len(outstanding), session.alive, session.log_path)
        )
        if result.returncode == 0:
            result.returncode = session.returncode or 1

    result.waves += 1
    result.processes = session.starts
    result.restarts = session.restarts
    result.fail_lines += len(FAIL_LINE.findall(text))
    collect_case_errors(result, text)

    audit_text = session.preamble + text
    plan = LaunchPlan(
        batch_width=batch_width,
        jobs=max(result.jobs, batch_width),
        visible_cpus=budget.visible_cpus,
        # NO `jobs` TERM.  runWave caps its lanes at the size of the chunk it was
        # handed; a rolling session does not know how many cases it will see and
        # stands the full width up, so the number the child prints -- and the
        # number this audit has to expect -- has no chunk in it.
        host_workers=min(budget.driver_workers, batch_width),
        worker_policy="multi_gpu_rolling",
        gpu=gpu,
        result_mode=chunk_result_mode(list(outstanding.values()) or jobs[start:end],
                                      result_mode),
        declared_fidelity=declared_fidelity,
    )
    result.problems.extend(
        f"gpu{gpu} p{proc} rolling: {p}" for p in check_run_receipts(audit_text, plan)
    )
    result.case_fidelity.append({
        "chunk": 1,
        "jobs": result.jobs,
        "policy": receipt_policy(audit_text),
        "result_mode": plan.result_mode,
        "scope": "process",
    })
    for match in REFILL_RECEIPT.finditer(text):
        try:
            result.refill_receipts.append(json.loads(match.group(1)))
        except ValueError:
            result.problems.append(f"gpu{gpu} p{proc}: unparseable REFILL receipt")
    for match in ROLLING_RECEIPT.finditer(text):
        try:
            result.rolling_receipts.append(json.loads(match.group(1)))
        except ValueError:
            result.problems.append(f"gpu{gpu} p{proc}: unparseable ROLLING receipt")
    for match in OCCUPANCY_RECEIPT.finditer(text):
        try:
            result.occupancy_receipts.append(json.loads(match.group(1)))
        except ValueError:
            pass
    return session


def _run_wave_chunk(
    *,
    session: EvaluatorSession,
    result: WorkerResult,
    chunk: Sequence[tuple[str, str, str]],
    chunk_index: int,
    manifest_arg: str,
    workdir: Path,
    stem: str,
    gpu: str,
    proc: int,
) -> tuple[str, list[str]]:
    """Send one claimed chunk to the persistent evaluator.  (text, problems).

    FAILURE ISOLATION HAS TWO LAYERS AND THEY ARE NOT THE SAME LAYER.

      * One case throwing is already isolated INSIDE the evaluator
        (EvaluatorServer::runOneCase catches and reports), and arrives here as a
        `[EVALUATOR][CASE]` line with `"status":"failed"`.  The process keeps
        answering and this function does nothing special.
      * The process DYING is the layer that only the dispatcher can handle: a
        CUDA abort, an OOM kill, a fail-closed refusal that escaped.  Then the
        cases that already printed a receipt are accounted for and the rest are
        NOT, and neither the evaluator nor the queue knows which is which.

    So the rule is: whatever printed a receipt is done; the remainder is
    re-queued ONCE onto a fresh child, and if that dies too every case still
    unaccounted for is REPORTED FAILED BY NAME.  Once, and not until it works:
    a chunk that kills two children in a row is a chunk with a poisoned case in
    it, and retrying it forever turns one bad candidate into a hung campaign.
    """
    problems: list[str] = []
    texts: list[str] = []
    pending = list(chunk)
    manifest = manifest_arg
    attempt = 0
    #: WP10.8.  What the previous attempt handed to a fresh child.  A receipt
    #: for one of these is a RECOVERED case, and it is marked as such: the
    #: 238 block-38 restarts were recovered and nothing downstream could tell a
    #: recovered case from one that never had a problem.
    requeued_keys: set[str] = set()
    while pending:
        attempt += 1
        # Unique per worker AND per attempt: the wave id is the sentinel this
        # dispatcher reads its own reply on, so a repeated id after a restart
        # would let a stale receipt end the wrong wait.
        wave_id = chunk_index * 100 + attempt
        outcome = session.wave(wave_id=wave_id, manifest=manifest)
        texts.append(outcome.text)

        done_keys: set[str] = set()
        for case in outcome.cases:
            key = path_key(str(case.get("output", "")))
            done_keys.add(key)
            if key in requeued_keys:
                result.restart_recovered.append(
                    str(case.get("deck") or case.get("output")))
                print("[RASBERY][MULTI_GPU][EVALUATOR][CASE][RESTART_RECOVERED] "
                      + json.dumps({"gpu": gpu, "proc": proc, "chunk": chunk_index,
                                    "attempt": attempt, "wave_id": wave_id,
                                    "output": case.get("output"),
                                    "status": case.get("status"),
                                    "restart_recovered": True},
                                   separators=(",", ":")))
            if case.get("status") != "ok":
                result.failed_cases.append(str(case.get("deck") or case.get("output")))
        for refusal in outcome.refused:
            problems.append(
                "gpu%s p%d chunk%d: the evaluator REFUSED a wave: %s"
                % (gpu, proc, chunk_index, refusal.get("what"))
            )
        unfinished = [job for job in pending if path_key(job[1]) not in done_keys]

        if outcome.alive:
            if unfinished:
                # The child lived and still said nothing about these: a refused
                # wave, or a receipt this dispatcher could not parse.  Either
                # way they are not results.
                problems.append(
                    "gpu%s p%d chunk%d: the wave finished but %d case(s) got no "
                    "[EVALUATOR][CASE] receipt: %s"
                    % (gpu, proc, chunk_index, len(unfinished),
                       ", ".join(job[0] for job in unfinished[:8]))
                )
                result.failed_cases.extend(job[0] for job in unfinished)
            return "".join(texts), problems

        record = {
            "gpu": gpu, "proc": proc, "chunk": chunk_index, "attempt": attempt,
            "wave_id": wave_id, "returncode": outcome.returncode,
            "completed": len(done_keys),
            "unfinished": [job[0] for job in unfinished],
            "requeued": bool(unfinished) and attempt == 1,
            "restarts": session.restarts,
        }
        # WP19.2.  THE RESTART RECORD KEEPS WHAT THE DEAD CHILD SAID.  Block
        # 38's two SIGSEGVs were unreconstructable not because the worker was
        # silent but because this record threw its last words away: `outcome.text`
        # was appended to `texts` for the throughput scrape and never looked at
        # again, and the FATAL line -- the one thing a reader greps -- carried
        # `completed:0` and nothing else.  A crashed worker's tail belongs ON
        # the record that says it crashed.
        record.update(crash_evidence(outcome.text))
        result.fatal_waves.append(record)
        print("[RASBERY][MULTI_GPU][EVALUATOR][FATAL] "
              + json.dumps(record, separators=(",", ":")))
        if result.returncode == 0:
            result.returncode = outcome.returncode or 1

        restarted = session.restart()
        if not unfinished:
            return "".join(texts), problems
        if attempt >= 2 or not restarted:
            problems.append(
                "gpu%s p%d chunk%d: %d case(s) were never evaluated -- the worker died "
                "%s and its %s: %s"
                % (gpu, proc, chunk_index, len(unfinished),
                   "twice" if attempt >= 2 else "and could not be restarted",
                   "one re-queue is spent" if attempt >= 2 else "restart budget is spent",
                   ", ".join(job[0] for job in unfinished[:8]))
            )
            result.failed_cases.extend(job[0] for job in unfinished)
            return "".join(texts), problems

        pending = unfinished
        requeued_keys = {path_key(job[1]) for job in pending}
        print("[RASBERY][MULTI_GPU][EVALUATOR][RESTART_RECOVERY] "
              + json.dumps({"gpu": gpu, "proc": proc, "chunk": chunk_index,
                            "attempt": attempt, "requeued": len(pending),
                            "cases": [job[0] for job in pending]},
                           separators=(",", ":")))
        retry = workdir / f"{stem}.chunk{chunk_index:04d}.retry.txt"
        retry.write_text(
            "".join(
                f'"{i}" "{o}"' + (f" {m}" if m else "") + "\n"
                for i, o, m in pending
            ),
            encoding="utf-8",
            newline="\n",
        )
        manifest = str(retry.resolve())
    return "".join(texts), problems


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
    pin_omp: bool,
    dry_run: bool,
    declared_fidelity: str = "strict",
    unset: Sequence[str] = (),
    deadline: float | None = None,
    evaluator: bool = True,
    evaluator_max_restarts: int = 3,
) -> WorkerResult:
    """One process SEQUENCE: claim, run, repeat, on device *gpu* as slot *proc*.

    *deadline* (time.monotonic) bounds a CALIBRATION wave: it is checked before
    a chunk is claimed, never inside one, so a bounded wave still measures whole
    chunks and the cases/hour it reports is over work that actually finished.
    Killing a child mid-chunk would leave a half-written output and a claim that
    nobody completed, which is a worse thing to own than a slightly long wave.

    *evaluator* (WP8 stage 1.5, the default) makes that sequence ONE process: a
    persistent `--evaluator-jsonl -` child fed each claimed chunk as a `wave`
    request.  EVERYTHING AFTER THE CHILD PRINTS ITS RECEIPTS IS UNCHANGED -- the
    same audit, the same [PHYSICS_MODE] check, the same refill parsing -- because
    the evaluator prints per wave the receipts the batch branch prints per
    process.  What changes is the count of process images, which is the only
    thing this lever touches (docs/WP8_EVALUATOR_STAGE1 Sec 0.1).
    """
    cpus = budget.cpu_sets[index] if budget.cpu_sets else []
    result = WorkerResult(
        gpu=gpu, proc=proc, index=index,
        cpus=f"{cpus[0]}-{cpus[-1]}" if cpus else "",
        declared_fidelity=declared_fidelity,
        env=resolve_profile_env(
            batch_width=batch_width,
            driver_workers=budget.driver_workers,
            solver_threads=budget.solver_threads,
            gpu=gpu,
            pin_omp=pin_omp,
            extra=extra_env,
            overrides=overrides,
            unset=unset,
        ),
    )
    started = time.monotonic()
    chunk_index = 0
    # Every artefact this worker writes is named by (gpu, proc): with K
    # processes on one device, a name keyed only by the GPU would have two
    # workers overwrite each other's chunk manifest between write and exec.
    stem = f"gpu{gpu}.p{proc}"
    env = launch_env(result.env, unset)
    result.evaluator = bool(evaluator) and not dry_run
    # Stood up LAZILY, at the first non-empty claim.  A worker with nothing to
    # claim must start no process at all: at K > jobs that is the difference
    # between "a worker that idled" and "a worker that measured its own startup
    # and reported 0 cases/hour", which is the defect auto_claim_size() exists
    # to prevent one layer up.
    session: EvaluatorSession | None = None
    evaluator_command = (
        pin_prefix(budget, index)
        + list(executable)
        + ["--evaluator-jsonl", "-", "--batch-mode", str(batch_width)]
        + (["--result", result_mode] if result_mode else [])
    )

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if claim == "rolling":
            # ONE session for this worker's whole share.  Everything the chunk
            # loop does per chunk -- claim, send, audit, parse receipts -- is
            # done there, interleaved instead of serialised, and the teardown
            # block below is reached with the same `session` object it would
            # have had.
            if dry_run:
                print(
                    f"[RASBERY][MULTI_GPU][DRY] gpu={gpu} proc={proc} claim=rolling "
                    + " ".join(list(evaluator_command) + ["<<", "case*, wave"])
                )
                result.processes += 1
                result.waves += 1
                break
            session = _run_rolling_worker(
                result=result, queue=queue, jobs=jobs, index=index, budget=budget,
                batch_width=batch_width, prefetch=rolling_prefetch(overrides, extra_env),
                evaluator_command=evaluator_command, env=env,
                cwd=str(cwd) if cwd else None, workdir=workdir, stem=stem,
                gpu=gpu, proc=proc, result_mode=result_mode,
                declared_fidelity=declared_fidelity,
                evaluator_max_restarts=0,
            )
            break
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
                size = auto_claim_size(
                    index=index, processes=budget.processes, jobs=len(jobs),
                    remaining=remaining, batch_width=batch_width,
                )
        else:
            size = int(claim)

        start, end = queue.claim(size, index)
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
            shown = (evaluator_command + ["<<", f"wave:{manifest_arg}"]
                     if evaluator else command)
            print(
                f"[RASBERY][MULTI_GPU][DRY] gpu={gpu} proc={proc} jobs={end - start} "
                + " ".join(shown)
            )
            result.processes += 1
            result.waves += 1
            result.jobs += end - start
            continue

        # `audit_text` is what check_run_receipts sees; `text` is what the
        # per-wave counters are read from.  They differ in ONE mode and for one
        # reason: [PHYSICS_MODE] and the other declaration receipts are printed
        # ONCE PER PROCESS, before the first request, so a wave slice on its own
        # would be audited as a run that declared no fidelity at all.  Fidelity
        # is a process property (docs/WP8_EVALUATOR_STAGE1 Sec 8.3); the audit
        # has to read it where the process states it.
        if evaluator:
            if session is None:
                session = EvaluatorSession(
                    command=evaluator_command, env=env,
                    cwd=str(cwd) if cwd else None,
                    log_path=workdir / f"{stem}.evaluator.log",
                    max_restarts=evaluator_max_restarts,
                )
                if not session.start():
                    result.problems.append(
                        "gpu%s p%d: the evaluator never reached [READY] (rc=%r); see %s"
                        % (gpu, proc, session.returncode, session.log_path)
                    )
                    if result.returncode == 0:
                        result.returncode = session.returncode or 1
                    result.jobs += end - start
                    result.failed_cases.extend(job[0] for job in jobs[start:end])
                    break
            text, wave_problems = _run_wave_chunk(
                session=session, result=result, chunk=jobs[start:end],
                chunk_index=chunk_index, manifest_arg=manifest_arg,
                workdir=workdir, stem=stem, gpu=gpu, proc=proc,
            )
            result.problems.extend(wave_problems)
            result.processes = session.starts
            result.restarts = session.restarts
            audit_text = session.preamble + text
        else:
            with open(log, "w", encoding="utf-8") as sink:
                child = subprocess.run(  # noqa: S603
                    command, env=env, cwd=str(cwd) if cwd else None,
                    stdout=sink, stderr=subprocess.STDOUT, check=False,
                )
            text = log.read_text(encoding="utf-8", errors="replace")
            result.processes += 1
            if child.returncode != 0 and result.returncode == 0:
                result.returncode = child.returncode
            audit_text = text

        result.waves += 1
        result.jobs += end - start
        result.fail_lines += len(FAIL_LINE.findall(text))
        collect_case_errors(result, text)
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
            declared_fidelity=declared_fidelity,
        )
        result.problems.extend(
            f"gpu{gpu} p{proc} chunk{chunk_index}: {p}"
            for p in check_run_receipts(audit_text, plan)
        )
        # Per case, in the sense the receipt allows: [PHYSICS_MODE] is printed
        # once per process before any deck starts, so a chunk is the finest
        # grain that exists, and the job count is what makes the aggregate a
        # count of CASES rather than of chunks.
        result.case_fidelity.append({
            "chunk": chunk_index,
            "jobs": end - start,
            # In evaluator mode this reads the PROCESS's declaration, which is
            # the finest grain that exists there too -- one evaluator cannot mix
            # strict and A2 (EvaluatorServer.h refuses a wave that asks for a
            # fidelity the process did not resolve), so a wave inherits its
            # worker's word and the label stays true per case.
            "policy": receipt_policy(audit_text),
            "result_mode": plan.result_mode,
            "scope": "process" if evaluator else "chunk",
        })
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
        if evaluator and session is not None and not session.alive:
            # Dead and out of restarts: stop claiming.  What is left in the
            # queue is better stolen by a worker that still has a process than
            # claimed by one that would fail it.
            break

    # THE TEARDOWN RECEIPTS ARE ONLY HERE.  The arena is released once, at
    # shutdown -- that IS the mode -- so [CUDA][BATCH_OCCUPANCY], the tenancy
    # counters, [XSLIB_CACHE] and the [EVALUATOR] process receipt arrive after
    # the last wave and nowhere else.  A session that was not closed reports no
    # width_fill, which would silently read as "the arena never gathered".
    if session is not None:
        epilogue = session.close()
        result.processes = session.starts
        result.restarts = session.restarts
        result.evaluator_receipts = list(session.process_receipts)
        if session.returncode not in (0, None) and result.returncode == 0:
            result.returncode = session.returncode
        result.fail_lines += len(FAIL_LINE.findall(epilogue))
        collect_case_errors(result, epilogue)
        for match in REFILL_RECEIPT.finditer(epilogue):
            try:
                result.refill_receipts.append(json.loads(match.group(1)))
            except ValueError:
                result.problems.append(
                    f"gpu{gpu} p{proc}: unparseable REFILL receipt at shutdown")
        for match in OCCUPANCY_RECEIPT.finditer(epilogue):
            try:
                result.occupancy_receipts.append(json.loads(match.group(1)))
            except ValueError:
                pass
        if result.failed_cases:
            result.problems.append(
                "gpu%s p%d: %d case(s) did not produce a result: %s"
                % (gpu, proc, len(result.failed_cases),
                   ", ".join(result.failed_cases[:8]))
            )
        # WP19.  THE MESSAGE, NOT JUST THE NAME.  One problem line per dead
        # case, carrying what the child actually said, so [MULTI_GPU][FAIL]
        # names the defect instead of naming eight decks.  A case that died
        # with no message still gets a line -- see EvaluatorServer::reportCase.
        for detail in result.failed_case_errors[:16]:
            result.problems.append("gpu%s p%d: case died: %s" % (gpu, proc, detail))
        if len(result.failed_case_errors) > 16:
            result.problems.append(
                "gpu%s p%d: and %d more case death(s); see %s"
                % (gpu, proc, len(result.failed_case_errors) - 16, workdir)
            )

    result.wall_s = time.monotonic() - started
    return result


# ---------------------------------------------------------------------------
# The K auto-tuner (WP4)
# ---------------------------------------------------------------------------
#
# WHY A TUNER AND NOT A TABLE.  The 238 matrix (docs/W4_L5 Sec 4.8) says the
# knee MOVES.  Without MPS the best K on an RTX PRO 6000 is 2 (648 c/h against
# the 582 raw control) and K=8 is a LOSS (457); with MPS that same K=8 is the
# best measured arm (878 c/h, width_fill 0.41).  The winning K is a function of
# the device, the driver, whether MPS is up, the deck mix and the host's core
# count -- none of which this file can read off a constant.  So it measures, on
# this host, and then runs the queue at what it measured.
#
# WHAT THE CALIBRATION MAY NOT DO.
#
#   1. It may not CONSUME the queue.  Every candidate has to see the same jobs
#      or the comparison is between decks, not between widths.  So calibration
#      re-runs one fixed subset K times, OUT OF BAND, and the production queue
#      still holds every job afterwards: `[TOTAL].jobs` counts each manifest job
#      exactly once and the re-runs are reported separately as
#      `calibration_jobs`.  (Splitting the queue across candidates instead would
#      make each candidate a measurement of different decks, and would make the
#      campaign's own job accounting depend on how long tuning took.)
#   2. It may not overwrite a production output.  Each candidate's outputs are
#      redirected under `<workdir>/tune/k<K>/`, so a `--result full` campaign
#      comes out of calibration with its real outputs untouched.
#   3. It may not leave an MPS daemon behind.  The thread percentage is 100/K,
#      so each candidate needs its own server; each is stopped in a finally
#      before the next starts.  A daemon left up would make the NEXT candidate's
#      "no MPS" arm silently an MPS run -- the trap docs/W4_L5 Sec 5 #3 names.
#   4. It may not spend the campaign.  Each candidate is bounded by
#      --tune-budget-s, checked before a worker claims its next chunk.
#
# THE OBJECTIVE, and where it departs from the plan.  The plan's function is
#
#     score = median_cases_per_hour - tail_penalty - failure_penalty
#
# with ties broken by (p90 case latency, CPU, VRAM, K).  Two deliberate changes:
#
#   * A FAILURE IS A DISQUALIFICATION, NOT A PENALTY.  A candidate whose wave
#     returned nonzero, printed a [FAIL] line, or failed the imported receipt
#     audit is removed from the choice entirely.  A penalty big enough to be
#     safe is indistinguishable from exclusion; one small enough to be a penalty
#     lets a fast-and-broken K win.
#   * THE TAIL IS A TIE-BREAK, NOT A SUBTRACTION.  `score` stays in cases/hour
#     so the number in the receipt is the number the operator compares against
#     the 582 c/h raw line.  p90 case latency is not measured by this harness
#     (the executable reports per-batch occupancy, not per-case wall), so the
#     tail proxy is `tail_idle_max_s` from the REFILL receipts: the time the
#     last worker held the device with an emptying arena.  The substitution is
#     stated here because it is the one place the tuner is not the plan.
#
# Ties are broken by (tail_idle_max_s, host threads, VRAM, K), in that order, and
# "tie" means within --tune-tie-rel of the best.  Every term is in the receipt,
# so the choice can be recomputed from the receipt alone.
TUNE_CANDIDATES = (1, 2, 4, 8, 12, 16)
TUNE_SCHEMA = 1
TUNE_TIE_REL = 0.02


def width_for_procs(procs: int, *, batch_width: int | None, total_width: int | None) -> int:
    """The per-process arena width for a K-candidate.

    `--total-width T` is the WP4 arm: the DECLARED width per GPU is held fixed
    and only the rendezvous narrows, so W = ceil(T/K) -- rounded UP so K
    processes still declare at least T slots (12 x M6 = 72, not 12 x M5 = 60,
    which is the arm the 238 runner is actually queuing next).  `--batch-width W`
    holds the width PER PROCESS instead, which makes the declared width grow
    with K; the VRAM guard is what keeps that from being typed by accident.
    """
    procs = max(1, procs)
    if total_width:
        return max(1, -(-int(total_width) // procs))
    return max(1, int(batch_width or 1))


@dataclass
class TuneCandidate:
    """One measured -- or refused -- K."""

    procs: int
    width: int
    jobs: int = 0
    wall_s: float = 0.0
    samples: list[float] = field(default_factory=list)  # cases/hour, one per repeat
    width_fill: float = 0.0
    tail_idle_max_s: float = 0.0
    vram_per_device_gb: float = 0.0
    cpus_per_proc: int = 0
    host_thread_demand: int = 0
    mps: bool = False
    mps_thread_percent: int | None = None
    rc: int = 0
    fail_lines: int = 0
    problems: list[str] = field(default_factory=list)
    refused: str = ""  # nonempty: never measured, and why
    # What the wave DECLARED and what its chunks reported.  A candidate that is
    # disqualified on fidelity has to say so in its own receipt: the 238 tuner
    # disqualified all six candidates with every wave at rc=0 / dup=0, and the
    # only line that could have explained it printed `problems: 6`.
    declared_fidelity: str = "strict"
    fidelities: dict[str, int] = field(default_factory=dict)

    @property
    def cases_per_hour(self) -> float:
        """The score: the MEDIAN over repeats, so one unlucky wave cannot win."""
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @property
    def eligible(self) -> bool:
        return (
            not self.refused
            and bool(self.samples)
            and self.rc == 0
            and self.fail_lines == 0
            and not self.problems
        )

    @property
    def disqualified(self) -> str:
        if self.refused:
            return self.refused
        if not self.samples:
            return "no measurement"
        if self.rc != 0:
            return f"rc={self.rc}"
        if self.fail_lines:
            return f"{self.fail_lines} [FAIL] line(s)"
        if self.problems:
            return f"{len(self.problems)} receipt problem(s): {self.problems[0]}"
        return ""

    def receipt(self) -> dict:
        # The first three problems verbatim, not a count.  `problems: 6` is what
        # the 238 tuner printed while disqualifying every candidate for an A2
        # receipt under a strict declaration, and it named neither word.
        detail = [p for p in self.problems[:3]]
        return {
            "procs_per_gpu": self.procs,
            "batch_width": self.width,
            "declared_width_per_gpu": self.procs * self.width,
            "jobs": self.jobs,
            "wall_s": round(self.wall_s, 3),
            "cases_per_hour": round(self.cases_per_hour, 1),
            "samples_cases_per_hour": [round(s, 1) for s in self.samples],
            "width_fill": round(self.width_fill, 4),
            "tail_idle_max_s": round(self.tail_idle_max_s, 3),
            "vram_per_device_gb": round(self.vram_per_device_gb, 3),
            "cpus_per_proc": self.cpus_per_proc,
            "host_thread_demand": self.host_thread_demand,
            "mps": self.mps,
            "mps_thread_percent": self.mps_thread_percent,
            "rc": self.rc,
            "fail_lines": self.fail_lines,
            "problems": len(self.problems),
            "problem_detail": detail,
            "declared_fidelity": self.declared_fidelity,
            "fidelity_measured": self.fidelities,
            "eligible": self.eligible,
            "disqualified": self.disqualified,
        }


def candidate_refusal(
    *,
    procs: int,
    width: int,
    gpus: Sequence[str],
    visible_cpus: int,
    vram: VramPlan,
    allow_overcommit: bool,
) -> str:
    """Why this K cannot be MEASURED at all, or "" if it can.

    Two bounds, both the plan's: the VRAM guard (arenas are sized at process
    start and held for the run, so K x W slots must fit on EVERY listed device)
    and the host cores (a process that cannot be given a core of its own is not
    an independent process, it is contention wearing a taskset).
    """
    processes = len(gpus) * max(1, procs)
    if processes > visible_cpus:
        return (
            f"host has {visible_cpus} visible CPU(s) for {processes} process(es): each "
            "process needs at least one core of its own or the split measures contention "
            "rather than width"
        )
    if vram.over and not allow_overcommit:
        worst = vram.over[0]
        return (
            f"device {worst.gpu}: {procs} x width {width} needs {worst.demand_gb:.2f} GB "
            f"of a {worst.budget_gb or 0.0:.2f} GB budget"
        )
    return ""


def choose_candidate(
    candidates: Sequence[TuneCandidate], *, tie_rel: float = TUNE_TIE_REL
) -> TuneCandidate | None:
    """The winner, deterministically.

    Best median cases/hour; everything within *tie_rel* of it is a tie, and a
    tie goes to lower tail idle, then fewer host threads, then lower VRAM, then
    the smaller K.  Pure and total -- the same list always gives the same
    answer, which is what lets tools/test_fleet_tuner.py drive it with no GPU.
    """
    live = [c for c in candidates if c.eligible]
    if not live:
        return None
    best = max(c.cases_per_hour for c in live)
    if best <= 0.0:
        return None
    floor = best * (1.0 - max(0.0, tie_rel))
    tied = [c for c in live if c.cases_per_hour >= floor]
    return min(
        tied,
        key=lambda c: (
            round(c.tail_idle_max_s, 3),
            c.host_thread_demand,
            round(c.vram_per_device_gb, 3),
            c.procs,
        ),
    )


def tune_key(gpus: Sequence[str]) -> dict:
    """What a saved tuning result is keyed on (plan: device UUID + driver).

    A tuning result is a measurement of ONE fleet: the same K on a different
    device, or after a driver change, is a different number.  Unknowable fields
    come back None with `verified:false` rather than a key that pretends.
    """
    key: dict = {"gpus": [str(g) for g in gpus], "uuids": [], "names": [],
                 "driver": None, "verified": False}
    exe = shutil.which("nvidia-smi")
    if not exe:
        return key
    try:
        done = subprocess.run(  # noqa: S603
            [exe, "-i", ",".join(str(g) for g in gpus),
             "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return key
    if done.returncode != 0:
        return key
    for line in done.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        key["uuids"].append(parts[0])
        key["names"].append(parts[1])
        key["driver"] = parts[2]
    key["verified"] = bool(key["uuids"])
    return key


def tune_key_matches(saved: dict, current: dict) -> bool:
    """Whether a saved result describes THIS fleet.

    Unverified on either side is NOT a match: a result whose device is unknown
    cannot be shown to describe this one, and the caller refuses rather than
    spending a campaign at a K that was measured somewhere else.
    """
    if not saved or not current:
        return False
    if not saved.get("verified") or not current.get("verified"):
        return False
    return (saved.get("uuids") == current.get("uuids")
            and saved.get("driver") == current.get("driver"))


def calibration_jobs(
    jobs: Sequence[tuple[str, str, str]], *, count: int, procs: int, tune_dir: Path
) -> list[tuple[str, str, str]]:
    """The SAME head-of-manifest subset for every candidate, redirected.

    Same jobs so the candidates are comparable; redirected outputs so a
    calibration can never overwrite a production result or share a restart
    namespace with one.  The index prefix keeps the outputs distinct inside one
    candidate even when two decks have the same basename.
    """
    subset = list(jobs)[: max(1, count)]
    out: list[tuple[str, str, str]] = []
    for index, (deck, output, mode) in enumerate(subset):
        name = f"k{procs:02d}_{index:04d}_{Path(output).name or 'case.h5'}"
        out.append((deck, str((tune_dir / name).resolve()), mode))
    return out


def _measure_candidate(
    *,
    procs: int,
    width: int,
    gpus: Sequence[str],
    jobs: Sequence[tuple[str, str, str]],
    visible_cpus: int,
    args,
    executable: Sequence[str],
    overrides: dict[str, str],
    cwd: Path | None,
    tune_root: Path,
    repeat: int,
    declared_fidelity: str,
    unset: Sequence[str],
) -> tuple[TuneCandidate, list[WorkerResult]]:
    """Run ONE candidate wave and score it.  Never raises for a measurement."""
    budget = plan_host_budget(
        gpus=gpus, batch_width=width, visible_cpus=visible_cpus, pin=args.pin,
        driver_workers=args.driver_workers, solver_threads=args.solver_threads,
        procs_per_gpu=procs, oversubscribe=not args.no_oversubscribe,
        result_mode=args.result, io_writer_mode=overrides.get("RASBERY_IO_WRITER"),
    )
    vram = plan_vram(
        gpus=gpus, procs_per_gpu=procs, batch_width=width,
        per_slot_gb=args.vram_per_slot_gb, margin_gb=args.vram_margin_gb,
        extra_process_gb=args.vram_per_extra_process_gb,
        device_memory_gb=args.device_memory_gb,
    )
    cand = TuneCandidate(
        procs=procs, width=width,
        vram_per_device_gb=vram.per_device_gb,
        cpus_per_proc=budget.cpus_per_proc,
        host_thread_demand=budget.host_thread_demand,
        declared_fidelity=declared_fidelity,
    )
    cand.refused = candidate_refusal(
        procs=procs, width=width, gpus=gpus, visible_cpus=visible_cpus, vram=vram,
        allow_overcommit=args.allow_vram_overcommit,
    )
    if cand.refused:
        return cand, []

    workdir = tune_root / f"k{procs:02d}" / f"r{repeat}"
    workdir.mkdir(parents=True, exist_ok=True)
    subset = calibration_jobs(jobs, count=args.tune_jobs, procs=procs,
                              tune_dir=tune_root / f"k{procs:02d}" / "out")
    (tune_root / f"k{procs:02d}" / "out").mkdir(parents=True, exist_ok=True)

    mps = MpsSession(workdir=workdir, gpus=gpus,
                     thread_percent=args.mps_thread_percent, procs_per_gpu=procs)
    results: list[WorkerResult] = []
    started = time.monotonic()
    try:
        if args.mps:
            mps.start()
            if not mps.active and not args.mps_optional:
                cand.refused = "MPS requested and unavailable: " + mps.reason
                return cand, []
        cand.mps = mps.active
        cand.mps_thread_percent = mps.thread_percent if mps.active else None
        queue = Queue(workdir / "queue.json", len(subset),
                      processes=len(gpus) * procs)
        workers = [
            (gpu, proc, gpu_index * procs + proc)
            for gpu_index, gpu in enumerate(gpus)
            for proc in range(procs)
        ]
        common = dict(
            queue=queue, jobs=list(subset), budget=budget, batch_width=width,
            claim=args.claim, result_mode=args.result, executable=list(executable),
            workdir=workdir, cwd=cwd, overrides=overrides, extra_env=mps.client_env(),
            pin_omp=args.pin_omp, dry_run=args.dry_run,
            declared_fidelity=declared_fidelity, unset=unset,
            deadline=time.monotonic() + max(1.0, float(args.tune_budget_s)),
            # THE CALIBRATION RUNS THROUGH EVALUATORS TOO when the campaign will.
            # A K measured against per-chunk process images is not a measurement
            # of the campaign that will not pay them: the per-process fixed cost
            # is what the split multiplies (VRAM_GB_PER_EXTRA_PROCESS above), and
            # a candidate that pays it once per worker sits at a different knee
            # from one that pays it once per chunk.
            evaluator=args.evaluator,
            evaluator_max_restarts=args.evaluator_max_restarts,
        )
        if len(workers) == 1:
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
        # Before the NEXT candidate, always: a live daemon would make the next
        # arm's "no MPS" measurement silently an MPS one.
        mps.stop()

    wall = time.monotonic() - started
    cand.jobs = sum(r.jobs for r in results)
    cand.wall_s = wall
    if wall > 0 and cand.jobs > 0:
        cand.samples.append(3600.0 * cand.jobs / wall)
    fills = [r.width_fill for r in results if r.width_fill > 0]
    cand.width_fill = sum(fills) / len(fills) if fills else 0.0
    cand.tail_idle_max_s = max((r.tail_idle_s for r in results), default=0.0)
    cand.rc = next((r.returncode for r in results if r.returncode != 0), 0)
    cand.fail_lines = sum(r.fail_lines for r in results)
    cand.problems = [p for r in results for p in r.problems]
    for result in results:
        for policy, count in result.fidelities.items():
            cand.fidelities[policy] = cand.fidelities.get(policy, 0) + count
    return cand, results


def calibrate(
    *,
    gpus: Sequence[str],
    jobs: Sequence[tuple[str, str, str]],
    visible_cpus: int,
    args,
    executable: Sequence[str],
    overrides: dict[str, str],
    cwd: Path | None,
    workdir: Path,
    declared_fidelity: str,
    unset: Sequence[str],
) -> tuple[TuneCandidate | None, list[TuneCandidate], float, int | None]:
    """Measure every candidate K and pick one.

    Returns (chosen, candidates, calibration_s, writer_threads_observed).  The
    last one is the WP4 host-budget fix in action: the calibration logs carry
    `[RASBERY][IO_WRITER]`, so the production wave's writer budget can be what
    the executable said rather than what the plan guessed.
    """
    tune_root = workdir / "tune"
    tune_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    measured: list[TuneCandidate] = []
    observed_writer: int | None = None
    for procs in sorted({max(1, int(k)) for k in args.tune_candidates}):
        width = width_for_procs(procs, batch_width=args.batch_width,
                                total_width=args.total_width)
        merged: TuneCandidate | None = None
        for repeat in range(max(1, int(args.tune_repeats))):
            cand, results = _measure_candidate(
                procs=procs, width=width, gpus=gpus, jobs=jobs,
                visible_cpus=visible_cpus, args=args, executable=executable,
                overrides=overrides, cwd=cwd, tune_root=tune_root, repeat=repeat,
                declared_fidelity=declared_fidelity, unset=unset,
            )
            if merged is None:
                merged = cand
            else:
                merged.samples.extend(cand.samples)
                merged.jobs += cand.jobs
                merged.wall_s += cand.wall_s
                merged.tail_idle_max_s = max(merged.tail_idle_max_s, cand.tail_idle_max_s)
                merged.rc = merged.rc or cand.rc
                merged.fail_lines += cand.fail_lines
                merged.problems.extend(cand.problems)
                merged.refused = merged.refused or cand.refused
                for policy, count in cand.fidelities.items():
                    merged.fidelities[policy] = merged.fidelities.get(policy, 0) + count
            if observed_writer is None:
                for log in sorted((tune_root / f"k{procs:02d}").rglob("*.log")):
                    observed_writer = writer_threads_from_receipt(
                        log.read_text(encoding="utf-8", errors="replace"))
                    if observed_writer is not None:
                        break
            if cand.refused:
                break
        assert merged is not None
        measured.append(merged)
        print("[RASBERY][MULTI_GPU][TUNE][CAND] "
              + json.dumps(merged.receipt(), separators=(",", ":")))
    chosen = choose_candidate(measured, tie_rel=args.tune_tie_rel)
    return chosen, measured, time.monotonic() - started, observed_writer


def tune_payload(
    *,
    chosen: TuneCandidate,
    candidates: Sequence[TuneCandidate],
    calibration_s: float,
    key: dict,
    tie_rel: float,
    source: str,
    calibration_jobs_run: int,
) -> dict:
    """The [MULTI_GPU][TUNE] receipt, which is also the --tune-from file."""
    return {
        "schema": TUNE_SCHEMA,
        "source": source,
        "key": key,
        # A K measured under A2 is not a K measured under strict: the staged
        # tolerances change how many outers a case costs, which is the whole
        # quantity being divided between processes.  The saved result says which
        # one it was so a reuse can be read for what it is.
        "declared_fidelity": chosen.declared_fidelity,
        "tie_rel": tie_rel,
        "calibration_s": round(calibration_s, 3),
        "calibration_jobs": calibration_jobs_run,
        "candidates": [c.receipt() for c in candidates],
        "chosen": {
            "procs_per_gpu": chosen.procs,
            "batch_width": chosen.width,
            "declared_width_per_gpu": chosen.procs * chosen.width,
            "cases_per_hour": round(chosen.cases_per_hour, 1),
            "width_fill": round(chosen.width_fill, 4),
            "mps": chosen.mps,
        },
    }


def load_tune_result(path: Path) -> dict:
    """Read a saved --tune-from file, refusing a shape it cannot trust."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: not a tuning result object")
    if int(payload.get("schema", -1)) != TUNE_SCHEMA:
        raise ValueError(
            f"{path}: tuning result schema {payload.get('schema')!r}, this dispatcher "
            f"writes {TUNE_SCHEMA}. Re-tune rather than reinterpreting an older shape")
    chosen = payload.get("chosen") or {}
    if int(chosen.get("procs_per_gpu", 0)) <= 0 or int(chosen.get("batch_width", 0)) <= 0:
        raise ValueError(f"{path}: tuning result names no usable (procs_per_gpu, batch_width)")
    return payload


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
        "--procs-per-gpu", default="1", metavar="K",
        help="independent RASBERY processes per listed GPU (L5, GA plan Sec 5.5). Each "
        "gets its own arenas, its own rendezvous and its own slice of the host, so "
        "--batch-width is the width PER PROCESS: `--procs-per-gpu 4 --batch-width 16` "
        "declares the same 64 slots per device as `--batch-width 64` does alone. "
        "`auto` calibrates K on this host first (WP4) -- see the tuner group",
    )
    p.add_argument("--batch-width", type=int,
                   help="CUDA arena width per PROCESS. Give this or --total-width")
    p.add_argument(
        "--total-width", type=int, metavar="T",
        help="DECLARED width per GPU, split across the processes: the per-process width "
        "is ceil(T/K). This is the WP4 arm -- the device sees the same declared width "
        "at every K and only the rendezvous narrows, so `--total-width 64` gives "
        "1xM64, 2xM32, 4xM16, 8xM8, 12xM6, 16xM4. Rounded UP so K processes never "
        "declare less than T between them",
    )
    p.add_argument("--jobs", required=True,
                   help="job manifest: `<input.json> <output.h5> [result-mode]` per line; the "
                        "optional third field overrides --result for that job alone")
    p.add_argument(
        "--result",
        choices=RESULT_MODES,
        help=(
            "what every job writes unless its manifest line says otherwise: full (result "
            "HDF5 + restarts + pin CSV), pin-off (no pin output), light (scalar JSONL, no "
            "HDF5).  All three run the same physics and produce the same trajectory digest, "
            "and since WP1 none of them is a screening run: light needs no "
            "RASBERY_ALLOW_SCREENING"
        ),
    )
    p.add_argument(
        "--claim",
        default="auto",
        help="jobs per claim: auto (half the remaining queue per worker, floored at "
        "--batch-width), all (static split, one process per worker), rolling (WP18: "
        "one job per claim, re-claimed as each case reports, so the arena refills "
        "per slot instead of per wave -- sets RASBERY_EVALUATOR_ROLLING=1 and "
        "requires --evaluator), or an integer",
    )
    p.add_argument("--pin", default="taskset", choices=("taskset", "numactl", "none"),
                   help="how each process is bound to its CPU share")
    p.add_argument(
        "--driver-workers", type=int,
        help="Driver refill lanes per process. Default is the ARENA WIDTH -- the "
        "executable's own default with RASBERY_BATCH_HOST_THREADS absent "
        "(main.cpp:698), which is what the 582 c/h reference line runs. It "
        "oversubscribes the process's cores on purpose: a lane is blocked on the GPU "
        "rendezvous nearly all of its life, so lanes are not CPU workers, and capping "
        "them at the core count caps the achievable rendezvous width before the run "
        "starts (24 lanes into 64 slots measured 115.6 c/h against the same binary's "
        "582, width_fill 0.03). OMP_MAX_ACTIVE_LEVELS=1 is what keeps the lanes from "
        "spawning nested solver teams",
    )
    p.add_argument(
        "--no-oversubscribe", action="store_true",
        help="lanes = min(width, cpus per process), the old policy. A deliberate arm: it "
        "measures the core count, not the rendezvous. Cores are ALWAYS partitioned by "
        "taskset regardless of this flag -- what it changes is the lane count",
    )
    p.add_argument("--solver-threads", type=int,
                   help="RASBERY_OMP_THREADS / OMP_NUM_THREADS / OMP_THREAD_LIMIT per "
                        "process; default is the arena width per process (the reference's "
                        "64 at --batch-width 64)")
    p.add_argument(
        "--pin-omp", action="store_true",
        help="also export OMP_PROC_BIND=TRUE and OMP_PLACES=cores. The reference line sets "
        "neither, and RASBERY sets both itself and re-execs (main.cpp:286), so this "
        "changes what the harness DECLARES rather than what libgomp does",
    )
    p.add_argument(
        "--print-env", action="store_true",
        help="print the resolved per-process environment as [MULTI_GPU][ENV] receipts and "
        "exit, so a mismatch against test/reference/batch_reference_env_238.json is "
        "visible before a seven-minute run",
    )
    # WP19.  DEFAULT IS A RUN, NOT A DIRECTORY.
    #
    # `multi_gpu_run` was a FIXED path, so run N+1 wrote its chunk manifests,
    # its per-chunk logs, its evaluator logs and its queue.json on top of run
    # N's.  Three reruns of a flaky 128-case campaign therefore left exactly one
    # run's evidence -- and it was the run that had already been explained.
    # Diagnosing an intermittent defect requires the runs you have already done,
    # so the default is now a timestamped subdirectory and reruns cannot
    # collide.  --workdir still takes an explicit path verbatim, which is what a
    # resumed queue needs (see --resume: it reads <workdir>/queue.json).
    p.add_argument("--workdir", default=None,
                   help="where chunk manifests, per-chunk logs and the MPS pipe/log land "
                        "(default: multi_gpu_run/run_<UTC timestamp>, printed at start; "
                        "pass an explicit path to resume an earlier run's queue)")
    p.add_argument(
        "--cwd",
        help="run each RASBERY process here. A deck's cross-section file and its "
        "restart namespace are relative to the deck directory, so a manifest of "
        "relative inputs needs this; absolute manifests do not",
    )
    p.add_argument(
        "--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE",
        help="export one key into every child's environment, on top of DEFAULT_ENV. "
             "Repeatable, and any key is accepted -- this is how the WP16 host-wait "
             "arms are taken (--set RASBERY_CUDA_SYNC_MODE=auto|spin|yield|blocking, "
             "--set RASBERY_OMP_WAIT=active|passive), neither of which is a "
             "DEFAULT_ENV key because the 238 reference line sets neither. Every "
             "resolved value is printed per process in [RASBERY][MULTI_GPU][ENV]",
    )
    p.add_argument(
        "--fidelity", "--expect-fidelity", dest="fidelity",
        choices=DECLARABLE_FIDELITIES,
        help=(
            "the convergence/statepoint policy this campaign DECLARES. Every chunk's "
            "[RASBERY][PHYSICS_MODE] receipt must report exactly this policy; a "
            "mismatch either way is a hard failure, because a throughput number "
            "measured at a fidelity other than the declared one belongs in neither "
            "column (plan Sec 6.2: never mix strict and A2 in one table). The DEFAULT "
            "is DERIVED from the resolved child environment by the binary's own rule "
            "(src/RunContract.h) -- and DEFAULT_ENV carries the A2 staged tolerances, "
            "so the default declaration of a production wave is A2"
        ),
    )
    p.add_argument(
        "--strict", action="store_true",
        help="run the STRICT-policy control arm: delete RASBERY_STAGED_FLUX_TOL / "
             "_XE_TOL / _LOOSE_SETTLE, RASBERY_GA_FEEDBACK_PASSES and "
             "RASBERY_PHYSICS_FIDELITY from every child's environment, inherited ones "
             "included. Without it the campaign measures the A2 arm; the two must "
             "never share a table, and this is how the second row is produced",
    )
    p.add_argument(
        "--set-unset", dest="set_unset", action="append", default=[], metavar="KEY",
        help="delete one key from every child's environment. Beats --set and the "
             "inherited environment both",
    )

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

    tune = p.add_argument_group(
        "K auto-tuner (WP4)",
        "Calibrate --procs-per-gpu on THIS host before the campaign: a few jobs per "
        "candidate K, the winner runs the queue. The calibration never claims a "
        "production job and never writes a production output -- it re-runs one fixed "
        "subset into <workdir>/tune/, so the queue is intact afterwards and "
        "[TOTAL].jobs still counts each manifest job exactly once.",
    )
    tune.add_argument(
        "--tune", action="store_true",
        help="calibrate K even if --procs-per-gpu names a number (same as "
        "`--procs-per-gpu auto`)",
    )
    tune.add_argument(
        "--tune-candidates", default=",".join(str(k) for k in TUNE_CANDIDATES),
        metavar="K,K,...",
        help="candidate process counts. K=1 is in the default set on purpose: it is the "
        "control, and a tuner that cannot choose it cannot decline the lever",
    )
    tune.add_argument(
        "--tune-jobs", type=int, default=16, metavar="N",
        help="jobs per candidate, the SAME head-of-manifest N for every K. Fewer than K "
        "would leave processes with nothing to claim and measure startup",
    )
    tune.add_argument(
        "--tune-budget-s", type=float, default=600.0, metavar="S",
        help="wall bound per candidate wave, checked before a worker claims its next "
        "chunk (never mid-chunk: a killed child leaves a half-written output)",
    )
    tune.add_argument("--tune-repeats", type=int, default=1, metavar="R",
                      help="waves per candidate; the score is the MEDIAN over them")
    tune.add_argument("--tune-tie-rel", type=float, default=TUNE_TIE_REL, metavar="F",
                      help="candidates within this fraction of the best count as tied, and "
                           "a tie goes to (lower tail idle, lower CPU, lower VRAM, smaller K)")
    tune.add_argument(
        "--tune-from", metavar="JSON",
        help="reuse a previous [MULTI_GPU][TUNE] result instead of calibrating. Refused "
        "unless its device UUIDs and driver match this host -- a K measured on another "
        "fleet is not a measurement of this one",
    )
    tune.add_argument("--tune-save", metavar="JSON",
                      help="where to write the tuning result (default <workdir>/tune/result.json)")
    tune.add_argument(
        "--allow-tune-mismatch", action="store_true",
        help="use a --tune-from result whose device key does not match (or cannot be "
        "verified). The receipt still records key_match:false",
    )

    mem = p.add_argument_group("device memory guard")
    mem.add_argument(
        "--vram-per-slot-gb", type=float, default=VRAM_GB_PER_SLOT,
        help="arena cost of one batch slot; default is the measured 238 M64 full-output "
        "peak (13 GB / 64 slots)",
    )
    mem.add_argument(
        "--vram-per-extra-process-gb", type=float, default=VRAM_GB_PER_EXTRA_PROCESS,
        help="fixed device cost of each process AFTER the first (CUDA context, modules, "
        "library handles, allocator pool). Default is the measured 238 slope: 12.3/14.9/"
        "20.0/30.2 GB at K=1/2/4/8 with K x W = 64 throughout",
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

    ev = p.add_argument_group(
        "persistent evaluator (WP8 stage 1.5)",
        "A worker is ONE long-lived `RASBERY --evaluator-jsonl -` fed a `wave` per "
        "claimed chunk, instead of one RASBERY process per chunk. Nothing about the "
        "physics, the queue or the audit changes -- only how many process images a "
        "campaign pays for. Measured cost of an image: 1.75-4.92 s of `outside_drive` "
        "(docs/WP8_EVALUATOR_STAGE1 Sec 0.1).",
    )
    ev.add_argument(
        "--no-evaluator", dest="evaluator", action="store_false",
        help="run the OLD shape: one `--jobs` process per claimed chunk. This is the "
        "wall control the evaluator is measured against, and the fallback for a "
        "build with no --evaluator-jsonl mode",
    )
    ev.add_argument(
        "--evaluator-max-restarts", type=int, default=3, metavar="N",
        help="how many times a worker's evaluator may be replaced after it DIES "
        "(a CUDA abort, an OOM kill). The cases of the dead wave that already "
        "printed a receipt are kept; the rest are re-queued ONCE onto the fresh "
        "child and, if that dies too, reported failed BY NAME. 0 disables "
        "replacement: the worker stops claiming and the queue is left to the others",
    )
    p.set_defaults(evaluator=True)

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
    if args.batch_width is None and args.total_width is None:
        print("error: one of --batch-width (width per process) or --total-width "
              "(declared width per GPU, split across the processes) is required",
              file=sys.stderr)
        return 2
    if args.batch_width is not None and args.batch_width <= 0:
        print("error: --batch-width must be positive", file=sys.stderr)
        return 2
    if args.total_width is not None and args.total_width <= 0:
        print("error: --total-width must be positive", file=sys.stderr)
        return 2

    # --procs-per-gpu is a COUNT or the word `auto`; --tune is the same request
    # spelled next to an explicit count, so a matrix script can add one flag.
    tune_requested = bool(args.tune)
    procs_per_gpu = 1
    if isinstance(args.procs_per_gpu, str) and args.procs_per_gpu.strip().lower() == "auto":
        tune_requested = True
    else:
        try:
            procs_per_gpu = int(args.procs_per_gpu)
        except (TypeError, ValueError):
            print("error: --procs-per-gpu must be a positive integer or `auto`",
                  file=sys.stderr)
            return 2
        if procs_per_gpu <= 0:
            print("error: --procs-per-gpu must be positive", file=sys.stderr)
            return 2
    try:
        tune_candidates = sorted({int(k) for k in str(args.tune_candidates).replace(",", " ").split()})
    except ValueError:
        print("error: --tune-candidates must be a comma-separated list of integers",
              file=sys.stderr)
        return 2
    if not tune_candidates or any(k <= 0 for k in tune_candidates):
        print("error: --tune-candidates must be positive", file=sys.stderr)
        return 2
    args.tune_candidates = tune_candidates
    if args.tune_repeats <= 0:
        print("error: --tune-repeats must be positive", file=sys.stderr)
        return 2
    if args.tune_jobs <= 0:
        print("error: --tune-jobs must be positive", file=sys.stderr)
        return 2
    if args.tune_from and tune_requested:
        print("error: --tune-from reuses a measurement and `auto`/--tune makes a new "
              "one; pick one", file=sys.stderr)
        return 2
    if args.mps_thread_percent is not None and not 1 <= args.mps_thread_percent <= 100:
        print("error: --mps-thread-percent must be in [1, 100]", file=sys.stderr)
        return 2
    if args.mps_optional and not args.mps:
        print("error: --mps-optional means nothing without --mps", file=sys.stderr)
        return 2
    if args.evaluator_max_restarts < 0:
        print("error: --evaluator-max-restarts must be >= 0", file=sys.stderr)
        return 2

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        print("error: --gpus needs at least one index", file=sys.stderr)
        return 2
    if len(set(gpus)) != len(gpus):
        print("error: --gpus must be distinct", file=sys.stderr)
        return 2
    if args.claim == "rolling" and not args.evaluator:
        print("error: --claim rolling needs the persistent evaluator; it streams cases "
              "into ONE running arena and has no chunked form", file=sys.stderr)
        return 2
    if args.claim not in ("auto", "all", "rolling"):
        try:
            if int(args.claim) <= 0:
                raise ValueError
        except ValueError:
            print("error: --claim must be auto, all, or a positive integer", file=sys.stderr)
            return 2

    try:
        jobs = read_manifest(Path(args.jobs))
        overrides = parse_overrides(args.set_values)
        unset = resolve_unset(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # THE DECLARED FIDELITY, decided once and before anything is measured.
    # It is a property of the child ENVIRONMENT, not of the width or the K, so
    # one resolution covers the calibration waves and the campaign alike: the
    # per-process env differs only in CUDA_VISIBLE_DEVICES and the width-derived
    # thread counts, none of which src/RunContract.h reads.
    declared_fidelity, fidelity_source = resolve_declared_fidelity(
        args,
        launch_env(
            resolve_profile_env(
                batch_width=args.batch_width or (args.total_width or 1),
                driver_workers=1, solver_threads=1, gpu=gpus[0],
                pin_omp=args.pin_omp, overrides=overrides, unset=unset,
            ),
            unset,
        ),
    )

    # WHAT USED TO BE HERE, AND WHY IT IS GONE (review doc R5).  This refused a
    # `--result light` wave unless RASBERY_ALLOW_SCREENING was set, because
    # main.cpp used to compute `screening = (ga_feedback_passes > 0) ||
    # light_result` and would refuse the chunk itself -- once per chunk, after
    # the queue was claimed.  Since WP1 `screening` is a property of the
    # FIDELITY alone (src/RunContract.h), a strict/light wave starts with no
    # permission at all, and a guard STRICTER than the executable refuses runs
    # that would have worked.  RASBERY_ALLOW_SCREENING still means exactly one
    # thing -- "a non-strict fidelity is allowed" -- and it stays the
    # OPERATOR's: tools/test_harness_env_parity.py keeps both harnesses from
    # granting it to themselves.

    # WP19.  The default is resolved HERE and not in add_argument, so the
    # timestamp is the run's and not the parser's, and so `--workdir X` is still
    # taken exactly as written.
    if args.workdir:
        workdir = Path(args.workdir)
    else:
        workdir = Path("multi_gpu_run") / time.strftime("run_%Y%m%dT%H%M%SZ",
                                                        time.gmtime())
    workdir.mkdir(parents=True, exist_ok=True)
    # PRINTED, ALWAYS.  A run whose logs are somewhere the operator has to guess
    # is a run whose logs do not exist.
    print("[RASBERY][MULTI_GPU][WORKDIR] "
          + json.dumps({"workdir": str(workdir.resolve()),
                        "explicit": bool(args.workdir)}, separators=(",", ":")))
    cwd = Path(args.cwd).resolve() if args.cwd else None
    if cwd is not None and not cwd.is_dir():
        print(f"error: --cwd is not a directory: {cwd}", file=sys.stderr)
        return 2

    # --- WP4: decide K before anything is planned on it ---------------------
    #
    # The tuner runs FIRST and out of band.  It has to: K decides the host
    # split, the VRAM demand and the MPS thread percentage, so a plan printed
    # before the calibration would be a plan for a K nobody chose.
    visible_cpus = visible_cpu_threads()
    writer_observed: int | None = None
    calibration_jobs_run = 0
    tune_receipt: dict | None = None
    if tune_requested or args.tune_from:
        key = tune_key(gpus)
        if args.tune_from:
            try:
                payload = load_tune_result(Path(args.tune_from))
            except (OSError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            matched = tune_key_matches(payload.get("key") or {}, key)
            payload["key_match"] = matched
            payload["source"] = "file"
            payload["file"] = str(args.tune_from)
            if not matched and not args.allow_tune_mismatch:
                print(
                    "[RASBERY][MULTI_GPU][TUNE] "
                    + json.dumps(payload, separators=(",", ":")))
                print(
                    "[RASBERY][MULTI_GPU][FAIL] --tune-from %s was measured on a "
                    "different fleet (saved key %r, this host %r) or on a host where the "
                    "device could not be identified. A K measured somewhere else is not a "
                    "measurement of this device: re-tune with --procs-per-gpu auto, or "
                    "pass --allow-tune-mismatch to use it anyway."
                    % (args.tune_from, (payload.get("key") or {}).get("uuids"),
                       key.get("uuids")),
                    file=sys.stderr)
                return 2
            procs_per_gpu = int(payload["chosen"]["procs_per_gpu"])
            args.batch_width = int(payload["chosen"]["batch_width"])
            tune_receipt = payload
        else:
            if args.dry_run:
                print("error: --procs-per-gpu auto measures; it cannot be dry-run",
                      file=sys.stderr)
                return 2
            if args.total_width is None:
                print(
                    "[RASBERY][MULTI_GPU][WARN] tuning with --batch-width holds the width "
                    "PER PROCESS, so the declared width per GPU grows with K and the VRAM "
                    "guard will refuse the larger candidates. --total-width is the arm "
                    "that holds the declared width fixed and narrows only the rendezvous.",
                    file=sys.stderr)
            chosen, candidates, calibration_s, writer_observed = calibrate(
                gpus=gpus, jobs=jobs, visible_cpus=visible_cpus, args=args,
                executable=executable, overrides=overrides, cwd=cwd, workdir=workdir,
                declared_fidelity=declared_fidelity, unset=unset,
            )
            calibration_jobs_run = sum(c.jobs for c in candidates)
            if chosen is None:
                for cand in candidates:
                    print("[RASBERY][MULTI_GPU][TUNE][CAND] "
                          + json.dumps(cand.receipt(), separators=(",", ":")),
                          file=sys.stderr)
                print(
                    "[RASBERY][MULTI_GPU][FAIL] the calibration produced no usable "
                    "candidate: every K was refused by the guards or failed its own "
                    "receipt audit. The campaign is not started, because a K chosen "
                    "from failed waves is not a measurement. Each [TUNE][CAND] line "
                    "above carries `disqualified` and `problem_detail`; this wave "
                    "declared fidelity %r (%s), so a candidate whose chunks reported "
                    "another policy was disqualified for THAT and not for its speed."
                    % (declared_fidelity, fidelity_source),
                    file=sys.stderr)
                return 2
            procs_per_gpu = chosen.procs
            args.batch_width = chosen.width
            tune_receipt = tune_payload(
                chosen=chosen, candidates=candidates, calibration_s=calibration_s,
                key=key, tie_rel=args.tune_tie_rel, source="calibration",
                calibration_jobs_run=calibration_jobs_run,
            )
            tune_receipt["key_match"] = True
            save_to = Path(args.tune_save) if args.tune_save else workdir / "tune" / "result.json"
            try:
                save_to.parent.mkdir(parents=True, exist_ok=True)
                save_to.write_text(json.dumps(tune_receipt, indent=2), encoding="utf-8")
                tune_receipt["saved"] = str(save_to)
            except OSError as exc:  # pragma: no cover - a full disk, not a policy
                print(f"[RASBERY][MULTI_GPU][WARN] could not save the tuning result: {exc}",
                      file=sys.stderr)
        print("[RASBERY][MULTI_GPU][TUNE] "
              + json.dumps(tune_receipt, separators=(",", ":")))
    elif args.total_width is not None:
        args.batch_width = width_for_procs(
            procs_per_gpu, batch_width=args.batch_width, total_width=args.total_width)

    budget = plan_host_budget(
        gpus=gpus,
        batch_width=args.batch_width,
        visible_cpus=visible_cpus,
        pin=args.pin,
        driver_workers=args.driver_workers,
        solver_threads=args.solver_threads,
        procs_per_gpu=procs_per_gpu,
        oversubscribe=not args.no_oversubscribe,
        result_mode=args.result,
        io_writer_mode=overrides.get("RASBERY_IO_WRITER"),
        writer_threads_observed=writer_observed,
    )
    queue = Queue(workdir / "queue.json", len(jobs),
                  processes=budget.processes)

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
                "driver_worker_policy": budget.worker_policy,
                "solver_threads": budget.solver_threads,
                "writer_threads": budget.writer_threads,
                "writer_policy": budget.writer_policy,
                "host_threads_per_proc": budget.host_threads_per_proc,
                "host_thread_demand": budget.host_thread_demand,
                "host_thread_ratio": round(budget.host_thread_ratio, 3),
                "width_policy": "total_width" if args.total_width else "batch_width",
                "procs_policy": ("tuned" if tune_receipt else "explicit"),
                "pin": budget.pin,
                "pin_omp": bool(args.pin_omp),
                # WP8 stage 1.5.  `worker_shape` is the one word that says how
                # many process images this campaign will pay for: `evaluator`
                # is one per worker, `chunked` is one per claimed chunk.
                "worker_shape": "evaluator" if args.evaluator else "chunked",
                "evaluator_max_restarts": args.evaluator_max_restarts,
                # The word every chunk's receipt will be audited against, and
                # the label this campaign's throughput row has to carry.
                "declared_fidelity": declared_fidelity,
                "declared_fidelity_source": fidelity_source,
                "env_unset": list(unset),
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
        extra_process_gb=args.vram_per_extra_process_gb,
        device_memory_gb=args.device_memory_gb,
    )
    print("[RASBERY][MULTI_GPU][VRAM] " + json.dumps(vram.receipt(), separators=(",", ":")))
    if vram.over and not args.allow_vram_overcommit:
        for device in vram.over:
            print(
                "[RASBERY][MULTI_GPU][FAIL] device %s: %d process(es) x width %d x %.3f GB/slot "
                "plus %d x %.2f GB of per-process fixed cost = %.2f GB needed, but the "
                "device has %.2f GB and the guard holds back %.2f GB (%.2f GB usable). "
                "Lower --batch-width or --procs-per-gpu, or pass --allow-vram-overcommit "
                "to find out the hard way at arena stand-up."
                % (device.gpu, budget.procs_per_gpu, args.batch_width, vram.per_slot_gb,
                   max(0, budget.procs_per_gpu - 1), vram.extra_process_gb,
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
    if args.claim == "rolling":
        # WP18.  Set HERE, so it lands in the [MULTI_GPU][ENV] receipt every
        # worker prints before anything runs: the arm a run took has to be
        # readable off the run, not inferred from the command line somebody
        # remembers typing.
        extra_env["RASBERY_EVALUATOR_ROLLING"] = "1"
    # stop() clears .active, and the TOTAL receipt is printed after it: what
    # the run was is not what the daemon is by then.
    mps_active = mps.active
    workers = [
        (gpu, proc, gpu_index * budget.procs_per_gpu + proc)
        for gpu_index, gpu in enumerate(gpus)
        for proc in range(budget.procs_per_gpu)
    ]

    # The resolved child environment, PER PROCESS, before anything runs.  An
    # env mismatch is the one defect class here that produces no error, no FAIL
    # line and no bad receipt -- only a throughput number that looks like
    # physics.  Three L5 sweeps (103-116 c/h) were reported as data before
    # anyone compared this against the reference line; printing it costs one
    # line per process and `--print-env` turns the whole run into that line.
    for gpu, proc, index in workers:
        env_receipt = resolve_profile_env(
            batch_width=args.batch_width,
            driver_workers=budget.driver_workers,
            solver_threads=budget.solver_threads,
            gpu=gpu,
            pin_omp=args.pin_omp,
            extra=extra_env,
            overrides=overrides,
            unset=unset,
        )
        print(
            "[RASBERY][MULTI_GPU][ENV] "
            + json.dumps(
                {
                    "gpu": gpu,
                    "proc": proc,
                    "cpus": (f"{budget.cpu_sets[index][0]}-{budget.cpu_sets[index][-1]}"
                             if budget.cpu_sets else ""),
                    "pin_prefix": pin_prefix(budget, index),
                    "declared_fidelity": declared_fidelity,
                    "declared_fidelity_source": fidelity_source,
                    "env_unset": list(unset),
                    "env": env_receipt,
                },
                separators=(",", ":"),
            )
        )
    if args.print_env:
        mps.stop()
        return 0

    started = time.monotonic()
    results: list[WorkerResult] = []
    try:
        common = dict(
            queue=queue, jobs=jobs, budget=budget, batch_width=args.batch_width,
            claim=args.claim, result_mode=args.result, executable=executable,
            workdir=workdir, cwd=cwd, overrides=overrides, extra_env=extra_env,
            pin_omp=args.pin_omp, dry_run=args.dry_run,
            declared_fidelity=declared_fidelity, unset=unset,
            evaluator=args.evaluator,
            evaluator_max_restarts=args.evaluator_max_restarts,
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
                    # WP8 stage 1.5: images vs chunks.  In `--no-evaluator` these
                    # two are equal; the gap between them IS what the mode bought.
                    "waves": r.waves,
                    "evaluator": r.evaluator,
                    "restarts": r.restarts,
                    "evaluator_totals": r.evaluator_totals,
                    "fatal_waves": r.fatal_waves,
                    "failed_cases": r.failed_cases,
                    "failed_case_errors": r.failed_case_errors,
                    "capture_arbiter": r.capture_arbiter,
                    "capture_race": r.capture_race,
                    "jobs": r.jobs,
                    "wall_s": round(r.wall_s, 3),
                    "cases_per_hour": round(r.cases_per_hour, 1),
                    "mean_width": round(r.mean_width, 3),
                    "width_fill": round(r.width_fill, 4),
                    "refills": r.refills,
                    "tail_idle_s": round(r.tail_idle_s, 3),
                    "slot_busy_fraction": round(sum(busy) / len(busy), 4) if busy else 0.0,
                    # WP18.  ABSENT unless the worker ran `--claim rolling`, so
                    # every receipt taken before this work package still parses
                    # and every arm's table has exactly the columns its arm
                    # produced.
                    **({"rolling": {
                        "admits": r.rolling_admits,
                        "immediate_fraction": round(r.rolling_immediate_fraction, 4),
                        "width_fill": round(r.rolling_width_fill, 4),
                        "tail_idle_s": round(r.rolling_tail_idle_s, 3),
                        "wave_barriers_avoided": sum(
                            int(x.get("wave_barriers_avoided", 0))
                            for x in r.rolling_receipts),
                        "slot_idle_ms_total": round(sum(
                            float(x.get("slot_idle_ms_total", 0.0))
                            for x in r.rolling_receipts), 1),
                    }} if r.rolling_receipts else {}),
                    "rc": r.returncode,
                    "fail_lines": r.fail_lines,
                    "declared_fidelity": r.declared_fidelity,
                    # Cases by the policy their chunk's receipt REPORTED, not by
                    # the one the plan declared.  The audit already fails a
                    # mismatch; this is what lets a table say which arm it is.
                    "fidelity_measured": r.fidelities,
                    "case_fidelity": r.case_fidelity,
                    # What the children were launched with, not what the plan
                    # intended: the [ENV] line above is a prediction, this is
                    # the record, and the two disagreeing is itself the bug.
                    "env": r.env,
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

    # WP18's two, and they gate the same way for the same reason: each names a
    # way a rolling session's numbers can come from the wrong tenancy.
    # `stale_tenant_refusals` is a lane that asked for a case while the ledger
    # still had it holding one; `epoch_regressions` is a completion retiring an
    # epoch nobody held.  Neither can happen while one Driver owns one lane,
    # which is exactly why a nonzero value voids the measurement rather than
    # merely annotating it.
    rolling_stale = sum(int(x.get("stale_tenant_refusals", 0))
                        for r in results for x in r.rolling_receipts)
    rolling_epochs = sum(int(x.get("epoch_regressions", 0))
                         for r in results for x in r.rolling_receipts)
    rolling_live = sum(int(x.get("live_tenancies_at_close", 0))
                       for r in results for x in r.rolling_receipts)
    if rolling_stale or rolling_epochs or rolling_live:
        problems.append(
            "rolling tenancy audit: stale_tenant_refusals=%d epoch_regressions=%d "
            "live_tenancies_at_close=%d (all three must be 0)"
            % (rolling_stale, rolling_epochs, rolling_live)
        )

    # The WP8 acceptance counters, aggregated for the same reason the tenancy
    # ones are: the gate is "across the campaign", not "in the receipt somebody
    # happened to read".  `xslib_loads` must track the number of distinct
    # library CONTENTS the fleet saw and NOT the case count; `cohort_builds` is
    # stage 2's equivalent for geometry.
    evaluator_totals: dict[str, int] = {}
    for r in results:
        for key, value in r.evaluator_totals.items():
            if isinstance(value, int):
                evaluator_totals[key] = evaluator_totals.get(key, 0) + value
    for key in ("slot_duplicates", "slot_stale_tenants", "slot_double_releases",
                "isolation_mismatches"):
        if evaluator_totals.get(key):
            problems.append(
                f"evaluator audit: {key}={evaluator_totals[key]} (must be 0)")

    all_widths = [r.mean_width for r in results if r.mean_width > 0]
    all_fills = [r.width_fill for r in results if r.width_fill > 0]
    # One count for the whole campaign, so the row this run becomes in a table
    # carries its own fidelity label.  Plan Sec 6.2: strict and A2 numbers do
    # not go in one table, and the only way to keep that rule is for every
    # number to say which one it is.
    fidelity_measured: dict[str, int] = {}
    for r in results:
        for policy, count in r.fidelities.items():
            fidelity_measured[policy] = fidelity_measured.get(policy, 0) + count
    print(
        "[RASBERY][MULTI_GPU][TOTAL] "
        + json.dumps(
            {
                "gpus": len(gpus),
                "procs_per_gpu": budget.procs_per_gpu,
                "processes": len(workers),
                # Images actually started, against waves actually sent.  With
                # `--no-evaluator` they are equal and the ratio is 1; with the
                # evaluator the ratio is the number of process stand-ups this
                # campaign did NOT pay for, which is the whole WP8 lever.
                "images": sum(r.processes for r in results),
                "waves": sum(r.waves for r in results),
                "worker_shape": "evaluator" if args.evaluator else "chunked",
                "evaluator_restarts": sum(r.restarts for r in results),
                "fatal_waves": sum(len(r.fatal_waves) for r in results),
                "failed_cases": sum(len(r.failed_cases) for r in results),
                "failed_case_errors": [e for r in results
                                      for e in r.failed_case_errors],
                "capture_race": [e for r in results for e in r.capture_race],
                "capture_race_retry": sum(
                    int((r.capture_arbiter or {}).get("capture_race_retry", 0))
                    for r in results),
                "capture_race_abandoned": sum(
                    int((r.capture_arbiter or {}).get("capture_race_abandoned", 0))
                    for r in results),
                "capture_race_unrecovered": sum(
                    int((r.capture_arbiter or {}).get("capture_race_unrecovered", 0))
                    for r in results),
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
                "evaluator_totals": evaluator_totals,
                # The tuner's footprint, kept OUT of `jobs`: the calibration
                # re-ran a subset of the manifest out of band, so counting it
                # here would make the campaign's throughput a function of how
                # long tuning took.
                "tuned": bool(tune_receipt),
                "tune_source": (tune_receipt or {}).get("source"),
                "calibration_jobs": calibration_jobs_run,
                "declared_fidelity": declared_fidelity,
                "declared_fidelity_source": fidelity_source,
                "fidelity_measured": fidelity_measured,
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
    print("[RASBERY][MULTI_GPU][OK] "
          + json.dumps({"jobs": total_jobs, "fidelity": declared_fidelity},
                       separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
