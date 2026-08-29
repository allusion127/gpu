#!/usr/bin/env python3
"""Unit contract for the multi-GPU dispatcher (plan Rev.7.1 Sec 13.3).

Every check here is about a way the dispatcher could send GPU time somewhere it
does not belong: two GPUs onto one output file, a claim that hands the same job
twice, a host split that oversubscribes the CPU, or a single-GPU run that gets
chunked into drained batches and loses the refill it was supposed to keep.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "tools"))

import run_multi_gpu_batch as mg  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)

    # --- the manifest is the same grammar main.cpp reads --------------------
    manifest = work / "jobs.txt"
    manifest.write_text(
        "# comment\n"
        "\n"
        "d0.json  out/d0.h5\n"
        '"a deck.json"  "out/a b.h5"   # quoted, with spaces\n'
        "d1.json  out/d1.h5  light\n"
        "d2.json  out/d2.h5  pin-off\n",
        encoding="utf-8",
    )
    jobs = mg.read_manifest(manifest)
    # THREE fields now, the third optional: a job's own result mode, which the
    # dispatcher has to carry into the chunk manifests it rewrites.  A job that
    # does not name one carries the empty string and inherits --result.
    check(jobs == [("d0.json", "out/d0.h5", ""),
                   ("a deck.json", "out/a b.h5", ""),
                   ("d1.json", "out/d1.h5", "light"),
                   ("d2.json", "out/d2.h5", "pin-off")],
          f"read_manifest: wrong parse {jobs!r}")

    # ...and an unknown word in that field is refused here rather than reaching
    # RASBERY, which would refuse it once per chunk after the GPU was already
    # claimed.
    junk = work / "junk.txt"
    junk.write_text("d0.json out/d0.h5 sloppy\n", encoding="utf-8")
    try:
        mg.read_manifest(junk)
        failures.append("read_manifest: accepted an unknown result mode")
    except ValueError:
        pass

    # Two jobs on one output would have two Drivers race inside one HDF5 file
    # and share a restart namespace -- the rule main.cpp enforces, enforced here
    # too so the GPU time is never spent.
    clash = work / "clash.txt"
    clash.write_text("d0.json out/x.h5\nd1.json out/x.h5\n", encoding="utf-8")
    try:
        mg.read_manifest(clash)
        failures.append("read_manifest: accepted two jobs writing one output")
    except ValueError:
        pass

    bad = work / "bad.txt"
    bad.write_text("only_one_field\n", encoding="utf-8")
    try:
        mg.read_manifest(bad)
        failures.append("read_manifest: accepted a line with no output path")
    except ValueError:
        pass

    # --- the queue hands every job out exactly once -------------------------
    queue = mg.Queue(work / "queue.json", 10)
    spans = [queue.claim(4), queue.claim(4), queue.claim(4), queue.claim(4)]
    covered: list[int] = []
    for start, end in spans:
        covered.extend(range(start, end))
    check(covered == list(range(10)), f"Queue: claims did not tile [0,10) exactly: {spans!r}")
    check(queue.remaining() == 0, "Queue: remaining() nonzero after the manifest drained")
    check(queue.claim(4) == (10, 10), "Queue: an exhausted queue must hand back an empty span")
    check(json.loads((work / "queue.json").read_text())["total"] == 10,
          "Queue: the state file must record the total")

# --- the host split ---------------------------------------------------------
#
# CORES are partitioned; LANES are not.  A Driver lane is blocked on the GPU
# rendezvous nearly all of its life, so capping the lanes at the per-process
# core count caps the ACHIEVABLE rendezvous width before the run starts.  That
# cap is what made this dispatcher's own single-GPU control measure 115.6 c/h
# where the raw production line measured 582 with the same binary, the same
# decks and the same physics env (width_fill 0.03).
budget = mg.plan_host_budget(
    gpus=["0", "1", "2", "3"], batch_width=64, visible_cpus=96,
    pin="taskset", driver_workers=None, solver_threads=None,
)
check(budget.cpus_per_gpu == 24, f"host budget: cpus_per_gpu={budget.cpus_per_gpu}, expected 24")
check(budget.driver_workers == 64,
      f"host budget: driver_workers={budget.driver_workers}, expected the ARENA WIDTH (64) "
      "-- the executable's own default with RASBERY_BATCH_HOST_THREADS absent "
      "(main.cpp:698), which is what the reference line runs. Capping at the 24-core "
      "share is the defect, not the policy")
check(budget.solver_threads == 64,
      f"host budget: solver_threads={budget.solver_threads}, expected the arena width "
      "(the reference's RASBERY_OMP_THREADS=64 at width 64)")
check(budget.worker_policy == "binary_default_width",
      f"host budget: worker_policy={budget.worker_policy!r} must name the policy in the "
      "receipt, or a silent change of default is invisible in the log")

# --no-oversubscribe restores the OLD policy, deliberately and by name.
capped = mg.plan_host_budget(
    gpus=["0", "1", "2", "3"], batch_width=64, visible_cpus=96,
    pin="taskset", driver_workers=None, solver_threads=None, oversubscribe=False,
)
check(capped.driver_workers == 24 and capped.worker_policy == "no_oversubscribe_cpu_capped",
      f"--no-oversubscribe must give back the CPU-capped policy: "
      f"{capped.driver_workers} / {capped.worker_policy!r}")
check(capped.cpu_sets == budget.cpu_sets,
      "--no-oversubscribe must change the LANE count only: the cores are partitioned by "
      "taskset either way, and two processes sharing a taskset range is a different bug")

explicit = mg.plan_host_budget(
    gpus=["0"], batch_width=64, visible_cpus=24, pin="taskset",
    driver_workers=32, solver_threads=7,
)
check(explicit.driver_workers == 32 and explicit.solver_threads == 7
      and explicit.worker_policy == "explicit",
      "--driver-workers/--solver-threads must win over both policies")
# Disjoint CPU sets: two processes pinned to the same places oversubscribe by
# exactly the factor OMP_PROC_BIND makes invisible.
flat = [cpu for cpus in budget.cpu_sets for cpu in cpus]
check(len(flat) == len(set(flat)), "host budget: CPU sets overlap between GPU processes")
check(len(budget.cpu_sets) == 4, "host budget: one CPU set per GPU")
check(mg.pin_prefix(budget, 1)[:2] == ["taskset", "-c"], "pin_prefix: expected a taskset binding")
check(mg.pin_prefix(budget, 1)[2] == "24-47", f"pin_prefix: wrong range {mg.pin_prefix(budget, 1)!r}")

numa = mg.plan_host_budget(
    gpus=["0", "1"], batch_width=64, visible_cpus=32,
    pin="numactl", driver_workers=None, solver_threads=None,
)
check(mg.pin_prefix(numa, 0)[0] == "numactl", "pin_prefix: numactl mode must use numactl")
check("--localalloc" in mg.pin_prefix(numa, 0),
      "pin_prefix: numactl binding must pin memory too, or the pinning is half done")

none = mg.plan_host_budget(
    gpus=["0"], batch_width=4, visible_cpus=24,
    pin="none", driver_workers=None, solver_threads=None,
)
check(mg.pin_prefix(none, 0) == [], "pin_prefix: --pin none must add no prefix")
# Workers never exceed the arena width: main.cpp will not run more Drivers than
# there are slots, and asking for more makes the receipt audit fail every run.
check(none.driver_workers == 4,
      f"host budget: driver_workers={none.driver_workers} must be capped at the arena width")

# --- the writer budget is the EXECUTABLE's, not the plan's ------------------
#
# WP4 replaced the fixed constant with a model of what src/IoWriter.h actually
# runs (one thread, or none in `inline` mode, or none when --result light writes
# no HDF5 at all).  The constant survives only as the last-resort fallback, and
# the full contract for the replacement is in tools/test_fleet_tuner.py Sec 8.
check(mg.WRITER_THREADS_PER_PROCESS == 8,
      "the plan Sec 13.3 host budget is writer ~8 threads per process")
check(mg.plan_writer_threads(result_mode="full")[0] == 1,
      "the writer budget charged to a process must be the ONE thread IoWriter.h starts, "
      "not the plan's eight: the difference is subtracted from the per-process core "
      "share before the solver threads are sized")

# --- the child environment is resolved in ONE place and PRINTED -------------
#
# An env mismatch between this dispatcher and the raw production line is the
# only defect class here that produces no error, no [FAIL] line and no bad
# receipt -- only a throughput number that reads like physics.  It cost three
# L5 sweeps (103-116 c/h) that were reported as data.  So: one resolver, shared
# with the single-GPU harness, and the resolved set printed before the run.
source = (root / "tools" / "run_multi_gpu_batch.py").read_text(encoding="utf-8")
check("resolve_profile_env" in source and "launch_env(result.env" in source,
      "the child env must come from run_single_gpu_batch.resolve_profile_env(), and the "
      "child's WHOLE environment from launch_env(), not be reassembled here: a "
      "difference between the two harnesses' defaults is invisible in every receipt "
      "either of them prints, and only launch_env() also applies --set-unset to the "
      "INHERITED environment, which is what makes --strict mean anything")
check("[RASBERY][MULTI_GPU][ENV]" in source and '"env": r.env' in source,
      "the resolved per-process env must be printed BEFORE the run ([MULTI_GPU][ENV]) and "
      "recorded in the [MULTI_GPU][PROC] receipt, so a mismatch is visible before the "
      "seven minutes rather than after them")

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    manifest = work / "one.txt"
    manifest.write_text("d0.json out/d0.h5\n", encoding="utf-8")
    rc = mg.main(["--gpus", "0", "--batch-width", "64", "--jobs", str(manifest),
                  "--workdir", str(work / "run"), "--pin", "none",
                  "--device-memory-gb", "64", "--print-env", "--", sys.executable])
    check(rc == 0, f"--print-env must resolve and exit 0 (rc={rc})")
    check(not (work / "run" / "gpu0.p0.chunk0001.txt").exists(),
          "--print-env must not claim the queue or write a chunk manifest: it is a "
          "pre-flight check, and a pre-flight that consumes jobs is not one")

# --- claim policy -----------------------------------------------------------
source = (root / "tools" / "run_multi_gpu_batch.py").read_text(encoding="utf-8")
check("if budget.processes == 1:" in source and "size = remaining" in source,
      "claim policy: a run with ONE worker under `auto` must claim the whole queue -- "
      "chunking it drains the batch between chunks, which is the tail Task 20 removes. "
      "The condition is on the PROCESS count, not the GPU count: with --procs-per-gpu K "
      "on one GPU there are K claimants and they do have each other to steal from")
check("check_run_receipts" in source and "from run_single_gpu_batch import" in source,
      "the exact-only and graph-fallback audits must be imported from the single-GPU "
      "harness, not restated, so the two harnesses cannot disagree about a valid run")
check("min(budget.driver_workers, batch_width, end - start)" in source,
      "the expected host_threads must match main.cpp's min(width, jobs) cap, or the "
      "last chunk of every run fails its own receipt audit")
check('"duplicates"' in source and '"stale_tenants"' in source,
      "the aggregate receipt must carry the Task 20 tenancy counters")

# --- `auto` must not starve a worker ----------------------------------------
#
# THE 238 ARTEFACT.  `--claim auto` was max(batch_width, ceil(remaining/(2K))).
# At 12 x M6 over 64 jobs that floor is 6 against a fair share of 5.33: the
# first ten workers took 60, the eleventh took 4, and the TWELFTH GOT NOTHING.
# A process with no jobs still starts, still prints its receipts, and reports a
# wall of pure startup -- and a twelfth of the device's declared slots stood
# idle for the whole wave while the arm was recorded as 12 x M6.
shares = [mg.fair_share(index=i, processes=12, jobs=64) for i in range(12)]
check(sum(shares) == 64 and min(shares) >= 1 and shares == [6, 6, 6, 6] + [5] * 8,
      f"fair_share must be floor(64/12)=5 each with the remainder ROUND-ROBIN over the "
      f"first four, tiling the manifest exactly: {shares}")
sizes = [mg.auto_claim_size(index=i, processes=12, jobs=64, remaining=64, batch_width=6)
         for i in range(12)]
check(min(sizes) >= 1 and sum(sizes) == 64,
      f"`--claim auto` at 12 x M6 over 64 jobs must give every worker at least one job "
      f"and hand out exactly 64: {sizes}")
check(max(sizes) <= 6,
      f"no worker may claim more than its fair share under auto: {sizes}")
# The arms that already worked must not move: at K=8/W=8 over 64 jobs each
# worker still takes exactly one width (docs/W4_L5 Sec 4.3, "리필도 없고 스틸도
# 없다"), and on the 128-job refill arm (Sec 4.6) the claim is still a width, so
# there is still something left to steal.
check([mg.auto_claim_size(index=i, processes=8, jobs=64, remaining=64, batch_width=8)
       for i in range(8)] == [8] * 8,
      "K=8 W=8 over 64 jobs must still be one width per worker: the L5 matrix arm is "
      "defined by having no refill and no steal in it")
check(mg.auto_claim_size(index=0, processes=8, jobs=128, remaining=128, batch_width=8) == 8,
      "on the 128-job refill arm the claim must stay a WIDTH (16 chunks), not jump to "
      "the 16-job fair share -- a fair share claimed in one go removes the refill the "
      "arm exists to measure")
check(mg.auto_claim_size(index=7, processes=8, jobs=4, remaining=4, batch_width=8) >= 1,
      "with fewer jobs than processes nobody is owed a share and whoever asks may take "
      "what is left; refusing there would leave jobs unrun")
check("auto_claim_size(" in source and "size = max(batch_width, -(-remaining" not in source,
      "the claim size must come from auto_claim_size(), not from the bare width floor "
      "that starved the twelfth worker")

# ...and the SIZE alone is not the guarantee.  A worker that finishes its chunk
# can come back before a worker whose thread has not reached its first claim,
# and take the share the cap computed for it: measured with a millisecond-long
# fake child at K=12, worker 4 claimed twice (10 jobs) and worker 11 got NOTHING.
# So the queue holds one job back for every worker that has not claimed yet.
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    queue = mg.Queue(work / "q.json", 64, processes=12)
    greedy: list[tuple[int, int]] = []
    while True:
        span = queue.claim(6, 0)   # one worker, looping as fast as it can
        if span[0] >= span[1]:
            break
        greedy.append(span)
    taken = sum(end - start for start, end in greedy)
    check(taken <= 64 - 11,
          f"a repeat claimant took {taken} of 64 with 11 workers still to claim: the "
          "queue must keep one job in hand for every worker that has not had a turn")
    late = [queue.claim(6, i) for i in range(1, 12)]
    check(all(end > start for start, end in late),
          f"every worker that arrives late must still get at least one job: {late}")
    check(taken + sum(end - start for start, end in late) == 64,
          "the reservation must not strand jobs: everything is still handed out")
    check(queue.remaining() == 0, "the queue must drain")

with tempfile.TemporaryDirectory() as tmp:
    import threading  # noqa: E402

    # The same thing under the real shape: 12 threads, staggered starts, each
    # looping until the queue refuses it.  Nobody may finish with zero.
    work = Path(tmp)
    queue = mg.Queue(work / "q.json", 64, processes=12)
    got = [0] * 12
    guard = threading.Lock()

    def worker(index: int) -> None:
        while True:
            start, end = queue.claim(
                mg.auto_claim_size(index=index, processes=12, jobs=64,
                                   remaining=max(0, queue.remaining()), batch_width=6),
                index)
            if start >= end:
                return
            with guard:
                got[index] += end - start

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(min(got) >= 1 and sum(got) == 64,
          f"12 concurrent workers over 64 jobs: every one must get at least one job and "
          f"all 64 must be handed out, got {got}")

# ===========================================================================
# L5 -- K processes per GPU (GA evaluator plan Sec 5.5)
# ===========================================================================
#
# The lever is a WIDTH lever that touches no C++: K processes of width M/K on
# one device give K independent arenas, K independent rendezvous and K
# independent host mutex sets.  Everything below is a way that could go wrong
# quietly -- a host split that hands K processes the whole machine each, a
# claim policy that thinks one GPU means one claimant, chunk files two workers
# both write, an MPS label on a run that was time-sliced, or arenas that do not
# fit and only say so after the queue has been claimed.

# --- the host split is by PROCESS, not by GPU -------------------------------
k4 = mg.plan_host_budget(
    gpus=["0"], batch_width=16, visible_cpus=24, pin="taskset",
    driver_workers=None, solver_threads=None, procs_per_gpu=4,
)
check(k4.processes == 4, f"K>1: processes={k4.processes}, expected 1 GPU x 4 = 4")
check(k4.cpus_per_proc == 6,
      f"K>1: cpus_per_proc={k4.cpus_per_proc}, expected 24//(1*4)=6. Four processes that "
      "each believe they own 24 cores oversubscribe by 4x, and OMP_PROC_BIND=TRUE pins "
      "them onto the same places -- which is the host contention L5 exists to remove, "
      "recreated one layer up")
check(k4.cpus_per_gpu == 24,
      f"K>1: cpus_per_gpu={k4.cpus_per_gpu}: the DEVICE's host share is unchanged; only "
      "the way it is divided changed")
check(len(k4.cpu_sets) == 4, "K>1: one CPU set per PROCESS, not per GPU")
flat4 = [cpu for cpus in k4.cpu_sets for cpu in cpus]
check(len(flat4) == len(set(flat4)),
      "K>1: CPU sets overlap between the processes on one GPU")
check(mg.pin_prefix(k4, 0)[2] == "0-5" and mg.pin_prefix(k4, 3)[2] == "18-23",
      f"K>1: taskset ranges must tile the host: {[mg.pin_prefix(k4, i)[2] for i in range(4)]!r}")
check(k4.driver_workers == 16,
      f"K>1: driver_workers={k4.driver_workers}, expected the WIDTH PER PROCESS (16). The "
      "cores are split K ways; the lanes are not, because a lane is GPU-wait-blocked and "
      "K arenas of 16 slots still need 16 lanes each to fill")
check(k4.solver_threads == 16,
      f"K>1: solver_threads={k4.solver_threads}, expected the width per process (16)")
check(k4.solver_threads >= 1, "K>1: a zero solver thread count is a silently serial solver")

# Negative control: the same GPU list with K=1 must get a STRICTLY larger share
# per process.  If these two are equal, the denominator forgot K.
k1 = mg.plan_host_budget(
    gpus=["0"], batch_width=16, visible_cpus=24, pin="taskset",
    driver_workers=None, solver_threads=None, procs_per_gpu=1,
)
check(k1.cpus_per_proc > k4.cpus_per_proc,
      "K>1: the per-process CPU share did not shrink when K grew -- the split is not "
      "counting processes")
check(k1.processes == 1 and len(k1.cpu_sets) == 1,
      "K=1 must degenerate to exactly the old one-process-per-GPU plan")

# Two GPUs x two processes: still one disjoint set per process, in GPU-major
# order, so worker (gpu i, proc j) is global index i*K+j.
g2k2 = mg.plan_host_budget(
    gpus=["0", "1"], batch_width=32, visible_cpus=24, pin="taskset",
    driver_workers=None, solver_threads=None, procs_per_gpu=2,
)
check(g2k2.processes == 4 and g2k2.cpus_per_proc == 6,
      f"G=2,K=2: processes={g2k2.processes} cpus_per_proc={g2k2.cpus_per_proc}, "
      "expected 4 and 6")
flat22 = [cpu for cpus in g2k2.cpu_sets for cpu in cpus]
check(len(flat22) == len(set(flat22)),
      "G=2,K=2: CPU sets overlap across the GPUxprocess grid")
check(f'gpu{{gpu}}.p{{proc}}' in source,
      "chunk manifests and logs must be named by (gpu, proc): with K workers on one "
      "device a name keyed only by the GPU has two of them overwrite each other's "
      "manifest between write and exec")
check("gpu_index * budget.procs_per_gpu + proc" in source,
      "the global process index -- which selects the CPU set -- must be GPU-major, or "
      "two workers share a taskset range")

# --- mean_width is per PROCESS and launch-weighted --------------------------
wr = mg.WorkerResult(gpu="0", proc=1)
wr.occupancy_receipts = [
    {"launches": 2, "instance_solves": 10},   # a short chunk, wide
    {"launches": 8, "instance_solves": 8},    # a long chunk, width 1
]
check(abs(wr.mean_width - 1.8) < 1e-9,
      f"mean_width={wr.mean_width}: must be launch-weighted (18/10), not a mean of "
      "means (3.0). L5 is judged on this number, and a two-launch chunk is not equal "
      "evidence to a two-hundred-launch one")
check('"mean_width_per_proc"' in source and "[MULTI_GPU][PROC]" in source,
      "the receipt must report mean_width PER PROCESS: it is a property of one arena's "
      "rendezvous, and the whole claim of L5 is what it does when the arena narrows")
check("gpu_wall = max(r.wall_s for r in share)" in source,
      "a device's wall is its slowest worker's, not the sum: the K workers run "
      "concurrently, and summing divides the device's throughput by K")

# --- the device-memory guard ------------------------------------------------
check(abs(mg.VRAM_GB_PER_SLOT - 13.0 / 64.0) < 1e-12,
      "the per-slot cost must stay the measured 238 M64 number (13 GB / 64 slots)")

# THE SLOT TERM IS FLAT IN K; THE PROCESS TERM IS NOT (238 matrix, docs/W4_L5
# Sec 4.8).  Holding K x W = 64 measured 12.3 / 14.9 / 20.0 / 30.2 GB at
# K = 1 / 2 / 4 / 8 -- a straight line of 2.56 GB per ADDITIONAL process, which
# is the CUDA context and its friends, not the arena.  This file used to assert
# the opposite ("4 x M16 must be charged the same 64 slots as 1 x M64"), and
# under that model 16 x M4 "needs 13 GB" when it needs about 51.
fits = mg.plan_vram(gpus=["0"], procs_per_gpu=4, batch_width=16, device_memory_gb=24.0)
check(abs(fits.per_device_gb - (mg.VRAM_GB_PER_SLOT * 64
                                + 3 * mg.VRAM_GB_PER_EXTRA_PROCESS)) < 1e-9,
      f"4 x M16 is the same 64 SLOTS as 1 x M64 plus THREE more processes: "
      f"{fits.per_device_gb} GB charged, {mg.VRAM_GB_PER_SLOT * 64 + 3 * mg.VRAM_GB_PER_EXTRA_PROCESS} expected")
check(fits.per_device_gb > mg.VRAM_GB_PER_SLOT * 64,
      "negative control: a guard that charges K x W slots and nothing else says 16 x M4 "
      "fits in 13 GB. It needs about 51, and the arm dies at arena stand-up with the "
      "queue already claimed")
check([d.verdict for d in fits.devices] == ["fits"],
      f"20.7 GB of arenas and contexts must fit 24 GB: {fits.receipt()!r}")

single = mg.plan_vram(gpus=["0"], procs_per_gpu=1, batch_width=64, device_memory_gb=24.0)
check(abs(single.per_device_gb - mg.VRAM_GB_PER_SLOT * 64) < 1e-9,
      "K=1 must be charged NO extra-process term: the first process's context is inside "
      "the measured 13 GB, and charging it twice would refuse the control arm")

over = mg.plan_vram(gpus=["0"], procs_per_gpu=4, batch_width=64, device_memory_gb=24.0)
check([d.verdict for d in over.devices] == ["over"],
      "4 x M64 is 4x the arenas, not the same ones split: the guard must catch it")
check(over.over and not over.unverified, "an over-budget device must be reported as over")

# Charged PER DEVICE, not per campaign: two GPUs each have their own memory.
two = mg.plan_vram(gpus=["0", "1"], procs_per_gpu=2, batch_width=32, device_memory_gb=24.0)
check([d.verdict for d in two.devices] == ["fits", "fits"],
      "the guard must charge each device its own K*W slots; summing across GPUs would "
      "refuse a run that fits on both")
check(abs(two.aggregate_gb - 2 * two.per_device_gb) < 1e-9,
      "the aggregate is reported for the operator, not used as the test")

blind = mg.plan_vram(gpus=["0"], procs_per_gpu=2, batch_width=64,
                     probe=lambda gpu: None)
check([d.verdict for d in blind.devices] == ["unverified"],
      "when nvidia-smi cannot be asked, the guard must say UNVERIFIED rather than "
      "inventing a verdict in either direction")

# --- MPS lifecycle ----------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    session = mg.MpsSession(workdir=work, gpus=["0"], thread_percent=None, procs_per_gpu=4)
    check(session.thread_percent == 25,
          f"MPS: default thread percent {session.thread_percent}, expected 100/K = 25. "
          "K clients that each ask for 100 % is how an MPS server ends up time-slicing "
          "anyway")
    check(str(session.pipe_dir).startswith(str(work))
          and str(session.log_dir).startswith(str(work)),
          "MPS: the pipe and log directories must live in the workdir, not the shared "
          "/tmp/nvidia-mps default -- otherwise two campaigns join each other's server "
          "and stop() takes down a daemon somebody else is using")
    check(session.client_env() == {},
          "MPS: an inactive session must hand the children NOTHING. Exporting "
          "CUDA_MPS_PIPE_DIRECTORY without a running server is how a time-sliced run "
          "gets reported as an MPS run")
    session.stop()  # never started: must be a silent no-op, not an exception
    check(not session.active, "MPS: stop() on a never-started session must be a no-op")

    explicit = mg.MpsSession(workdir=work, gpus=["0"], thread_percent=60, procs_per_gpu=4)
    check(explicit.thread_percent == 60, "MPS: --mps-thread-percent must win over 100/K")

    # Negative control: no control binary on this host.  start() must fail
    # SOFTLY -- a reason, no exception, nothing started, no client env.
    import shutil as _shutil  # noqa: E402

    saved_which = _shutil.which
    try:
        _shutil.which = lambda name: None
        missing = mg.MpsSession(workdir=work, gpus=["0"], thread_percent=None, procs_per_gpu=2)
        check(missing.start() is False, "MPS: start() must return False when the daemon is absent")
        check(not missing.active and missing.reason,
              "MPS: an unavailable daemon must leave active=False and SAY WHY in the receipt")
        check(missing.client_env() == {},
              "MPS: a failed start must not leak MPS env into the children")
        check(missing.receipt(requested=True)["active"] is False,
              "MPS: the receipt must report active:false when the start failed")
        missing.stop()  # must not try to quit a daemon it never started

        # ...and the dispatcher must REFUSE rather than run time-sliced under
        # an MPS label, unless the operator asked for that arm on purpose.
        manifest = work / "one.txt"
        manifest.write_text("d0.json out/d0.h5\n", encoding="utf-8")
        base = ["--gpus", "0", "--batch-width", "4", "--jobs", str(manifest),
                "--workdir", str(work / "run"), "--pin", "none",
                "--device-memory-gb", "24", "--dry-run"]
        rc = mg.main(base + ["--procs-per-gpu", "2", "--mps", "--", sys.executable])
        # --dry-run never starts a daemon, so the refusal cannot be exercised
        # there; what it fixes is that --dry-run must not CLAIM one either.
        check(rc == 0, f"--mps --dry-run must plan, not start a daemon (rc={rc})")
    finally:
        _shutil.which = saved_which

    check("mps.stop()" in source and "finally:" in source,
          "the MPS daemon must be stopped from a finally: one left running holds the "
          "device's compute mode, and the next arm's without-MPS control would silently "
          "be an MPS run")
    check("--mps-optional" in source,
          "there must be a deliberate way to take the time-sliced arm; the default is "
          "to refuse, because a receipt that says MPS when it was time-slicing corrupts "
          "the comparison the arm exists to make")

# --- the queue under K concurrent claimants ---------------------------------
#
# With one process per GPU the claims were effectively serialised; with
# --procs-per-gpu K the K workers on one device are THREADS OF THIS PROCESS and
# claim at the same instant.  flock does not cover them -- it does not exist on
# Windows at all, and the file was observed half-truncated mid-claim there --
# so the queue needs an in-process mutex as well.  A lost claim here is a job
# run twice into one output, or a job never run at all.
with tempfile.TemporaryDirectory() as tmp:
    import threading  # noqa: E402

    work = Path(tmp)
    queue = mg.Queue(work / "q.json", 400)
    spans: list[tuple[int, int]] = []
    guard = threading.Lock()
    barrier = threading.Barrier(8)

    def drain() -> None:
        barrier.wait()
        while True:
            start, end = queue.claim(3)
            if start >= end:
                return
            with guard:
                spans.append((start, end))

    threads = [threading.Thread(target=drain) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    covered = sorted(i for start, end in spans for i in range(start, end))
    check(covered == list(range(400)),
          "Queue: 8 concurrent claimants did not tile [0,400) exactly -- a job was "
          "handed out twice (two Drivers into one HDF5) or dropped")
    check("self._mutex" in source,
          "the queue needs an in-process mutex, not only flock: with --procs-per-gpu the "
          "claimants are threads of ONE dispatcher, and on Windows there is no flock at "
          "all -- the read-modify-write was observed interleaving into a truncated file")

# --- the audit is keyed on FIDELITY, not on what the chunk wrote ------------
#
# BEFORE WP1 this block pinned the opposite rule: `expected_physics_mode()`
# picked SCREENING_PHYSICS_MODE for a light chunk, so a light launch PASSED by
# printing `screening:true` and FAILED by printing the exact receipt.  That
# accepted a screening claim from a run that was not screening, and voided a
# strict run for writing scalars.  Both names are gone; the audit now lives in
# tools/exact_audit.py and reads `policy`/`acceptance_eligible` (review R5).
import run_single_gpu_batch as sg  # noqa: E402

for gone in ("EXACT_PHYSICS_MODE", "SCREENING_PHYSICS_MODE", "expected_physics_mode",
             "check_physics_mode"):
    check(not hasattr(sg, gone),
          f"run_single_gpu_batch.{gone} is back: the acceptance audit is keyed on the "
          "OUTPUT mode again, which voids a strict/light arm and accepts a receipt that "
          "claims screening from a run that is not screening (review R5)")
check(sg.audit_physics_mode is not None,
      "run_single_gpu_batch does not import the fidelity audit from tools/exact_audit.py")
check("audit_physics_mode(output, plan.declared_fidelity)" in
      (Path(__file__).resolve().parents[1] / "tools" / "run_single_gpu_batch.py")
      .read_text(encoding="utf-8"),
      "check_run_receipts must audit on (output, declared fidelity) and nothing else: "
      "with the result mode back in it, what a case WRITES is back in the acceptance "
      "decision; without the declared fidelity, the audit is against a hardcoded "
      "`strict` that the A2 production environment can never satisfy")


def wp1_receipt(**fields: object) -> str:
    base = {"physics_mode": "full_exact_nodal", "screening": False,
            "feedback_pass_limit": 0, "full_hdf5": True,
            "physics_fidelity": "full_exact", "policy": "strict",
            "acceptance_eligible": True, "requires_exact_rerun": False,
            "result_mode": "full", "fidelity_declared": None, "gpu_full": False}
    base.update(fields)
    return "[RASBERY][PHYSICS_MODE] " + json.dumps(base) + "\n"


LIGHT_RECEIPT = wp1_receipt(result_mode="light", full_hdf5=False)
FULL_RECEIPT = wp1_receipt()
check(sg.audit_physics_mode(LIGHT_RECEIPT) == [],
      "a strict run that wrote scalar-only output is an acceptance measurement and the "
      "dispatcher's audit must accept it")
check(sg.audit_physics_mode(FULL_RECEIPT) == [],
      "the audit rejects a plain strict/full run")
# NEGATIVE CONTROLS: fidelity, not output shape, is what voids a run.
check(sg.audit_physics_mode(wp1_receipt(
          policy="feedback_limited", physics_fidelity="feedback_limited",
          physics_mode="ga_screen_feedback_limited", screening=True,
          acceptance_eligible=False, feedback_pass_limit=2,
          result_mode="light", full_hdf5=False)) != [],
      "negative control: light output with a NONZERO GA feedback limit is the "
      "approximation and must still be voided")
check(sg.audit_physics_mode(wp1_receipt(
          policy="L3coarse", physics_fidelity="coarse10", screening=True,
          acceptance_eligible=False, requires_exact_rerun=True)) != [],
      "negative control: a coarse-statepoint run that wrote the FULL HDF5 is still "
      "screening -- the output shape must not rescue it")
check(sg.audit_physics_mode(
          '[RASBERY][PHYSICS_MODE] {"physics_mode":"full_exact_nodal","screening":false,'
          '"feedback_pass_limit":0,"full_hdf5":true}\n') != [],
      "negative control: a pre-WP1 receipt cannot be audited on fidelity at all, and a "
      "run that cannot be audited is void, not a pass")

# The dispatcher decides which jobs share a process, so it has to agree with
# main.cpp's any_of(): one light job makes the whole chunk a screening run.
check(mg.chunk_result_mode([("a.json", "a.h5", ""), ("b.json", "b.h5", "light")], "full")
      == "light",
      "chunk_result_mode: one light job in a chunk makes the PROCESS a screening run "
      "(main.cpp:541 is any_of), and the receipt is printed once for the process")
check(mg.chunk_result_mode([("a.json", "a.h5", ""), ("b.json", "b.h5", "")], "light")
      == "light",
      "chunk_result_mode: jobs with no mode of their own inherit --result")
check(mg.chunk_result_mode([("a.json", "a.h5", "full")], "light") == "full",
      "chunk_result_mode: a manifest line's own mode wins over --result, so an all-full "
      "chunk of a light campaign is audited as exact")
check(mg.chunk_result_mode([("a.json", "a.h5", "pin-off")], None) == "pin-off",
      "chunk_result_mode: pin-off is not light")
check("result_mode=chunk_result_mode(" in source,
      "the per-chunk result mode must reach the audit, or every light arm of the L5 "
      "matrix comes back rc=3 on a field about output shape")

# --- main() argument guards -------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    manifest = work / "one.txt"
    manifest.write_text("d0.json out/d0.h5\n", encoding="utf-8")
    base = ["--gpus", "0", "--batch-width", "64", "--jobs", str(manifest),
            "--workdir", str(work / "run"), "--pin", "none", "--dry-run"]

    check(mg.main(["--gpus", "0", "--batch-width", "4", "--procs-per-gpu", "0",
                   "--jobs", str(manifest), "--", sys.executable]) == 2,
          "--procs-per-gpu 0 must be refused, not silently promoted to 1")
    check(mg.main(base + ["--procs-per-gpu", "4", "--device-memory-gb", "11",
                          "--", sys.executable]) == 2,
          "4 x M64 on an 11 GB device must be refused BEFORE the queue is claimed: the "
          "failure it predicts happens at arena stand-up, when the jobs are already gone")
    check(mg.main(base + ["--procs-per-gpu", "4", "--device-memory-gb", "11",
                          "--allow-vram-overcommit", "--", sys.executable]) == 0,
          "--allow-vram-overcommit must let the operator find out the hard way")
    check(mg.main(base + ["--procs-per-gpu", "4", "--device-memory-gb", "64",
                          "--", sys.executable]) == 0,
          "4 x M64 on a 64 GB device must be allowed")
    check(mg.main(["--gpus", "0", "--batch-width", "4", "--jobs", str(manifest),
                   "--mps-optional", "--", sys.executable]) == 2,
          "--mps-optional without --mps must be refused rather than silently ignored")

    # A light wave is NOT a screening run since WP1 (src/RunContract.h), so the
    # dispatcher must not demand a screening permission for it.  A guard that is
    # stricter than the executable refuses runs that would have worked -- and
    # this one used to refuse every strict/light L5 wave before the queue was
    # even claimed.
    saved_allow = os.environ.pop("RASBERY_ALLOW_SCREENING", None)
    try:
        check(mg.main(base + ["--result", "light", "--device-memory-gb", "64",
                              "--", sys.executable]) == 0,
              "--result light without RASBERY_ALLOW_SCREENING must now be ACCEPTED: "
              "light is an output shape and the binary no longer refuses it")
        check(mg.main(base + ["--result", "light", "--device-memory-gb", "64",
                              "--set", "RASBERY_ALLOW_SCREENING=1",
                              "--", sys.executable]) == 0,
              "--set RASBERY_ALLOW_SCREENING=1 must remain harmless: it still means "
              "'a non-strict fidelity is allowed', which is a different statement")
        os.environ["RASBERY_ALLOW_SCREENING"] = "0"
        check(mg.main(base + ["--result", "light", "--device-memory-gb", "64",
                              "--", sys.executable]) == 0,
              "RASBERY_ALLOW_SCREENING=0 must not refuse a light wave either")
    finally:
        os.environ.pop("RASBERY_ALLOW_SCREENING", None)
        if saved_allow is not None:
            os.environ["RASBERY_ALLOW_SCREENING"] = saved_allow
    # NEGATIVE CONTROL for the removal: the refusal text must be gone from the
    # dispatcher, or it can come back by being re-guarded somewhere else.
    check("classifies as a screening run and refuses unless" not in source,
          "run_multi_gpu_batch still refuses a light wave for want of "
          "RASBERY_ALLOW_SCREENING (review R5)")

# ===========================================================================
# The DECLARED fidelity: printed before the run, audited after it
# ===========================================================================
#
# THE DEFECT (238, after 5ccf879).  The production batch environment IS the A2
# staged-tolerance arm -- RASBERY_STAGED_FLUX_TOL=50 / _XE_TOL=1000 /
# _LOOSE_SETTLE=1 have been in DEFAULT_ENV since 7099e54 -- so every child
# printed `policy:'A2'` and every audit compared it against a hardcoded
# `strict`.  Result: rc=3 on a 12 x M6 wave and a WP4 tuner that disqualified
# ALL SIX candidates (rc=2, no winner) with every wave at rc=0, dup=0.  Nothing
# in any receipt said the word "fidelity".
import contextlib  # noqa: E402
import io  # noqa: E402


def dispatch_receipts(argv: list[str]) -> dict[str, list[dict]]:
    """Run main() and index the JSON receipts it printed by tag."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        mg.main(argv)
    out: dict[str, list[dict]] = {}
    for line in buffer.getvalue().splitlines():
        if not line.startswith("[RASBERY][MULTI_GPU][") or "] {" not in line:
            continue
        tag, payload = line.split("] ", 1)
        out.setdefault(tag.rsplit("[", 1)[1], []).append(json.loads(payload))
    return out


with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    manifest = work / "one.txt"
    manifest.write_text("d0.json out/d0.h5\n", encoding="utf-8")
    base = ["--gpus", "0", "--batch-width", "8", "--jobs", str(manifest),
            "--workdir", str(work / "run"), "--pin", "none",
            "--device-memory-gb", "96", "--print-env", "--", sys.executable]

    seen = dispatch_receipts(base)
    plan = (seen.get("PLAN") or [{}])[0]
    env_lines = seen.get("ENV") or [{}]
    check(plan.get("declared_fidelity") == "A2",
          f"the default declaration must be DERIVED from the resolved environment, and "
          f"DEFAULT_ENV is the A2 arm: [PLAN].declared_fidelity="
          f"{plan.get('declared_fidelity')!r}. Declaring `strict` by default is the "
          "defect: it voids every production wave on a policy the harness itself set")
    check(plan.get("declared_fidelity_source") == "env",
          "a derived declaration must say it was derived, or a receipt cannot be read "
          "back as an explanation of what was audited")
    check(env_lines[0].get("declared_fidelity") == "A2",
          "--print-env must show the declared fidelity in [MULTI_GPU][ENV]: it is a "
          "pre-flight check for exactly the mismatches that cost seven minutes each")

    strict = dispatch_receipts(base[:-2] + ["--strict", "--", sys.executable])
    strict_plan = (strict.get("PLAN") or [{}])[0]
    strict_env = (strict.get("ENV") or [{}])[0].get("env", {})
    check(strict_plan.get("declared_fidelity") == "strict",
          f"--strict must DECLARE strict: {strict_plan.get('declared_fidelity')!r}")
    check(not [k for k in strict_env if k.startswith("RASBERY_STAGED_")],
          f"--strict must remove the staged tolerances from the child environment, not "
          f"merely relabel the run: {sorted(k for k in strict_env if 'STAGED' in k)}")
    check(strict_plan.get("env_unset") and
          "RASBERY_GA_FEEDBACK_PASSES" in strict_plan["env_unset"],
          "--strict must also delete RASBERY_GA_FEEDBACK_PASSES and "
          "RASBERY_PHYSICS_FIDELITY: either can move an inherited environment off "
          "strict, and a control arm that only unsets what the HARNESS set is not one")

    stated = dispatch_receipts(base[:-2] + ["--fidelity", "A2", "--", sys.executable])
    check((stated.get("PLAN") or [{}])[0].get("declared_fidelity_source") == "operator",
          "--fidelity must be recorded as the OPERATOR's word, not as a derivation")
    aliased = dispatch_receipts(
        base[:-2] + ["--expect-fidelity", "L3coarse", "--", sys.executable])
    check((aliased.get("PLAN") or [{}])[0].get("declared_fidelity") == "L3coarse",
          "--expect-fidelity must be an alias for --fidelity")
    refused = None
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            refused = mg.main(["--gpus", "0", "--batch-width", "8", "--jobs", str(manifest),
                               "--workdir", str(work / "run2"), "--pin", "none",
                               "--device-memory-gb", "96", "--print-env",
                               "--fidelity", "sloppy", "--", sys.executable])
    except SystemExit as exit_code:  # argparse refuses an unknown choice itself
        refused = exit_code.code
    check(refused == 2,
          f"a fidelity word that is not a policy must be refused (rc={refused!r}), not "
          "defaulted: a typo that silently became `strict` would void the wave it named")

    # A --set that turns the staged tolerance off is the same statement as
    # --strict for the FLUX knob alone, and the derivation has to follow the
    # environment rather than the flag.
    off = dispatch_receipts(
        base[:-2] + ["--set-unset", "RASBERY_STAGED_FLUX_TOL",
                     "--set-unset", "RASBERY_STAGED_XE_TOL", "--", sys.executable])
    check((off.get("PLAN") or [{}])[0].get("declared_fidelity") == "strict",
          "--set-unset of both staged tolerances leaves a strict child, and the "
          "derivation must read the ENVIRONMENT, not the presence of --strict")

# ===========================================================================
# WP8 stage 1.5 -- the persistent evaluator, and the ways a worker can die
# ===========================================================================
#
# The dispatcher no longer spawns a RASBERY per claimed chunk; it keeps ONE
# `--evaluator-jsonl -` per (GPU, K-slot) and feeds it a `wave` per chunk.  Four
# things can go wrong that could not go wrong before, and each has a check here
# with a negative control that drives it:
#
#   1. the receipts a wave prints are not the receipts the audit reads, so a
#      whole campaign is audited as "the batch branch never ran";
#   2. the chunk accounting silently counts waves as processes, so the very
#      number the mode exists to reduce is the one that cannot be seen;
#   3. a child DIES mid-wave and the cases it never reported vanish -- no
#      output, no receipt, no FAIL line, just a manifest longer than the
#      results;
#   4. the re-queue retries forever, and one poisoned candidate hangs the
#      campaign instead of failing it.
#
# The child is tools/fake_rasbery_child.py, the SAME fake the tuner contract
# uses, driven through its FAKE_RASBERY_* knobs.  Nothing here needs a GPU.

FAKE = str((root / "tools" / "fake_rasbery_child.py").resolve())


def dispatch(argv: list[str]) -> tuple[int, dict[str, list[dict]], str]:
    """Run main(); return (rc, receipts by tag, the whole log)."""
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = mg.main(argv)
    text = out.getvalue() + err.getvalue()
    seen: dict[str, list[dict]] = {}
    for line in out.getvalue().splitlines():
        if not line.startswith("[RASBERY][MULTI_GPU][") or "] {" not in line:
            continue
        tag, payload = line.split("] ", 1)
        try:
            seen.setdefault(tag.rsplit("[", 1)[1], []).append(json.loads(payload))
        except ValueError:
            pass
    return rc, seen, text


def campaign(work: Path, name: str, decks: list[str], *, extra: list[str],
             claim: str = "2") -> tuple[int, dict[str, list[dict]], str, Path]:
    outdir = work / f"out_{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = work / f"jobs_{name}.txt"
    manifest.write_text(
        "".join(f'"{d}" "{(outdir / (Path(d).stem + ".h5")).as_posix()}"\n'
                for d in decks),
        encoding="utf-8")
    argv = ["--gpus", "0", "--procs-per-gpu", "1", "--batch-width", "4",
            "--jobs", str(manifest), "--workdir", str(work / f"run_{name}"),
            "--pin", "none", "--device-memory-gb", "96", "--claim", claim]
    rc, seen, text = dispatch(argv + extra + ["--", sys.executable, FAKE])
    return rc, seen, text, outdir


DECKS = [f"d{i}.json" for i in range(6)]

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)

    # --- 1.  the happy path: ONE image, three waves ------------------------
    #
    # `--claim 2` over six jobs is three chunks.  In the old shape that was
    # three RASBERY images and three `outside_drive` payments (1.75-4.92 s
    # each, GA evaluator plan Sec 2.3); here it must be ONE.  That inequality
    # -- images < waves -- IS the work package, and it has to be readable off a
    # receipt or nobody can tell whether it happened.
    rc, seen, text, outdir = campaign(work, "happy", DECKS, extra=[])
    proc = (seen.get("PROC") or [{}])[0]
    total = (seen.get("TOTAL") or [{}])[0]
    check(rc == 0, f"the evaluator campaign must finish clean (rc={rc})\n{text[-1200:]}")
    check(proc.get("evaluator") is True,
          "[PROC].evaluator must say which shape ran; a throughput number whose "
          "worker shape is unknown belongs in neither column")
    check(proc.get("waves") == 3,
          f"three claimed chunks must be three WAVES: waves={proc.get('waves')}")
    check(proc.get("processes") == 1,
          f"three waves must cost ONE process image: processes={proc.get('processes')}. "
          "If this ever equals `waves` the dispatcher went back to spawning per chunk "
          "and the whole work package is undone while every other number stays right")
    check(total.get("images") == 1 and total.get("waves") == 3,
          f"[TOTAL] must report images and waves apart: {total.get('images')!r} / "
          f"{total.get('waves')!r}")
    check(total.get("worker_shape") == "evaluator",
          f"[TOTAL].worker_shape={total.get('worker_shape')!r}")
    check((total.get("evaluator_totals") or {}).get("xslib_loads") == 1,
          f"the library must be parsed ONCE for the whole worker, not once per wave: "
          f"xslib_loads={(total.get('evaluator_totals') or {}).get('xslib_loads')!r}. "
          "This is the field plan WP8 names as the witness that the process-lifetime "
          "half was not torn down between waves")
    check(sorted(q.name for q in outdir.glob("*.h5")) ==
          [f"d{i}.h5" for i in range(6)],
          "every job must have produced its output")
    check(not proc.get("fatal_waves") and not proc.get("failed_cases"),
          f"a clean campaign must report no fatal wave and no failed case: "
          f"{proc.get('fatal_waves')!r} {proc.get('failed_cases')!r}")

    # NEGATIVE CONTROL for the same check: --no-evaluator is the old shape, and
    # there images == waves.  If that stops being true the two arms have stopped
    # being different and check 1 above is passing vacuously.
    rc, seen, text, outdir = campaign(work, "chunked", DECKS, extra=["--no-evaluator"])
    proc = (seen.get("PROC") or [{}])[0]
    total = (seen.get("TOTAL") or [{}])[0]
    check(rc == 0, f"the chunked control must finish clean (rc={rc})\n{text[-1200:]}")
    check(proc.get("evaluator") is False and total.get("worker_shape") == "chunked",
          "--no-evaluator must be recorded as the shape that ran")
    check(proc.get("processes") == 3 and proc.get("waves") == 3,
          f"negative control: in the chunked shape every wave IS an image "
          f"({proc.get('processes')} vs {proc.get('waves')}) -- if these differ the "
          "control is not measuring the old shape")
    check(sorted(q.name for q in outdir.glob("*.h5")) == [f"d{i}.h5" for i in range(6)],
          "the chunked control must produce the same outputs")

    # --- 2.  one case fails; the PROCESS survives --------------------------
    #
    # This is the isolation EvaluatorServer::runOneCase already provides, seen
    # from the dispatcher: `d3` reports `failed`, nothing is re-queued, no
    # restart happens, and the other five still produce outputs.  The campaign
    # still fails -- a generation that quietly lost a candidate is not a
    # generation -- but it fails by NAME.
    rc, seen, text, outdir = campaign(
        work, "casefail", DECKS, extra=["--set", "FAKE_RASBERY_FAIL=d3"])
    proc = (seen.get("PROC") or [{}])[0]
    check(rc != 0, "a failed case must fail the campaign")
    check(proc.get("restarts") == 0 and proc.get("processes") == 1,
          f"a CASE failure must not replace the worker: restarts="
          f"{proc.get('restarts')!r} processes={proc.get('processes')!r}. Restarting on "
          "a case that merely threw would pay a process stand-up for every bad "
          "candidate a GA generates")
    check(proc.get("failed_cases") == ["d3.json"],
          f"the failed case must be named: {proc.get('failed_cases')!r}")
    check(not proc.get("fatal_waves"),
          "a case failure is not a fatal wave; conflating them makes the FATAL "
          "receipt useless for finding the deaths it is for")
    survived = sorted(q.name for q in outdir.glob("*.h5"))
    check(survived == [f"d{i}.h5" for i in range(6) if i != 3],
          f"every OTHER case must still have produced its output: {survived}")

    # --- 3.  the worker DIES mid-wave: restart, re-queue once, finish ------
    #
    # The poison kills the child at `d2` on the FIRST image only (the marker
    # file spends it), so this drives the whole path: the wave dies after d0/d1
    # have printed receipts, the dispatcher notices EOF, names the cases nobody
    # accounted for, starts a fresh child and re-runs exactly those.
    marker = work / "poison.marker"
    rc, seen, text, outdir = campaign(
        work, "fatal", DECKS, claim="all",
        extra=["--set", "FAKE_RASBERY_POISON=d2",
               "--set", f"FAKE_RASBERY_POISON_MARKER={marker}"])
    proc = (seen.get("PROC") or [{}])[0]
    total = (seen.get("TOTAL") or [{}])[0]
    fatal = proc.get("fatal_waves") or []
    check(rc != 0, "a worker that died must fail the campaign, however well it recovered")
    check(len(fatal) == 1, f"exactly one fatal wave must be recorded: {fatal!r}")
    if fatal:
        check(fatal[0].get("completed") == 2,
              f"the cases that printed a receipt before the death are ACCOUNTED FOR: "
              f"completed={fatal[0].get('completed')!r} (expected d0, d1)")
        check(fatal[0].get("unfinished") == ["d2.json", "d3.json", "d4.json", "d5.json"],
              f"the cases nobody accounted for must be named, in order: "
              f"{fatal[0].get('unfinished')!r}. A death that is reported as a count "
              "and not as a list leaves the operator diffing a manifest against a "
              "directory")
        check(fatal[0].get("requeued") is True,
              "the remainder of a dead wave must be re-queued once")
    check(proc.get("restarts") == 1 and proc.get("processes") == 2,
          f"the dead worker must be REPLACED, exactly once here: restarts="
          f"{proc.get('restarts')!r} processes={proc.get('processes')!r}")
    check(sorted(q.name for q in outdir.glob("*.h5")) == [f"d{i}.h5" for i in range(6)],
          "after the restart and the re-queue every case must have run: the point of "
          "re-queueing is that a dead process costs a wave, not a generation")
    check(total.get("evaluator_restarts") == 1 and total.get("fatal_waves") == 1,
          f"the campaign receipt must carry the deaths: {total.get('evaluator_restarts')!r} "
          f"{total.get('fatal_waves')!r}")

    # --- 4.  NEGATIVE CONTROL: it dies again on the retry ------------------
    #
    # No marker, so the poison is permanent.  The re-queue is spent, and the
    # rule is that the still-unaccounted cases are REPORTED FAILED BY NAME and
    # the dispatcher stops.  A retry loop with no bound would turn one poisoned
    # candidate into a hung campaign, which is worse than a failed one.
    rc, seen, text, outdir = campaign(
        work, "poison", DECKS, claim="all",
        extra=["--set", "FAKE_RASBERY_POISON=d2"])
    proc = (seen.get("PROC") or [{}])[0]
    check(rc != 0, "a chunk that kills two children must fail the campaign")
    check(len(proc.get("fatal_waves") or []) == 2,
          f"two deaths, two FATAL records: {proc.get('fatal_waves')!r}")
    check(proc.get("failed_cases") == ["d2.json", "d3.json", "d4.json", "d5.json"],
          f"once the re-queue is spent every case still unaccounted for must be "
          f"reported failed BY NAME: {proc.get('failed_cases')!r}")
    check("never evaluated" in text and "[MULTI_GPU][FAIL]" in text,
          "the failure must reach the operator as a [FAIL] line, not only as a "
          "receipt field: a campaign that lost four candidates has to say so where "
          "an operator is looking")
    check(proc.get("restarts") <= 2,
          f"the retry is ONCE, not until it works: restarts={proc.get('restarts')!r}")
    survived = sorted(q.name for q in outdir.glob("*.h5"))
    check(survived == ["d0.h5", "d1.h5"],
          f"only the cases that ran before the first death may have outputs: {survived}")

    # --- 5.  NEGATIVE CONTROL: no restart budget ---------------------------
    #
    # `--evaluator-max-restarts 0` is the arm that says "do not replace a dead
    # worker".  The dispatcher must then STOP CLAIMING rather than keep sending
    # waves into a closed pipe: what is left in the queue is better stolen by a
    # worker that still has a process.
    rc, seen, text, outdir = campaign(
        work, "norestart", DECKS, claim="2",
        extra=["--set", "FAKE_RASBERY_POISON=d2", "--evaluator-max-restarts", "0"])
    proc = (seen.get("PROC") or [{}])[0]
    check(rc != 0, "a death with no restart budget must fail the campaign")
    check(proc.get("restarts") == 0 and proc.get("processes") == 1,
          f"--evaluator-max-restarts 0 must start no replacement: restarts="
          f"{proc.get('restarts')!r} processes={proc.get('processes')!r}")
    check(proc.get("waves") == 2,
          f"the worker must stop claiming after the death it cannot recover from "
          f"(waves={proc.get('waves')!r} of the three chunks the queue holds)")

    # --- 6.  a child that never becomes usable -----------------------------
    #
    # Not the same failure as a death mid-wave: nothing ran, so nothing is
    # accounted for, and the message has to point at the log rather than at a
    # case.
    rc, seen, text, outdir = campaign(
        work, "noready", DECKS, extra=["--set", "FAKE_RASBERY_NO_READY=1"])
    proc = (seen.get("PROC") or [{}])[0]
    check(rc != 0, "an evaluator that never reached [READY] must fail the campaign")
    check("never reached [READY]" in text,
          "the refusal must say the child never became usable, and where its log is")
    check(not list(outdir.glob("*.h5")),
          "a child that never started must have produced nothing")


if failures:
    raise SystemExit("multi-gpu dispatch: FAIL\n  " + "\n  ".join(failures))
print("multi-gpu dispatch: PASS")
