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
budget = mg.plan_host_budget(
    gpus=["0", "1", "2", "3"], batch_width=64, visible_cpus=96,
    pin="taskset", driver_workers=None, solver_threads=None,
)
check(budget.cpus_per_gpu == 24, f"host budget: cpus_per_gpu={budget.cpus_per_gpu}, expected 24")
check(budget.driver_workers == 24,
      f"host budget: driver_workers={budget.driver_workers}, expected the CPU share (24)")
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

# --- the writer budget is the plan's, stated once ---------------------------
check(mg.WRITER_THREADS_PER_PROCESS == 8,
      "the plan Sec 13.3 host budget is writer ~8 threads per process")

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
check(k4.driver_workers == 6,
      f"K>1: driver_workers={k4.driver_workers}, expected min(width 16, cpus_per_proc 6)")
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

fits = mg.plan_vram(gpus=["0"], procs_per_gpu=4, batch_width=16, device_memory_gb=24.0)
check(fits.per_device_gb == mg.VRAM_GB_PER_SLOT * 64,
      "4 x M16 must be charged the same 64 slots as 1 x M64: the arenas are sized per "
      "process, and L5 keeps the aggregate declared width fixed")
check([d.verdict for d in fits.devices] == ["fits"],
      f"13 GB of arenas must fit 24 GB: {fits.receipt()!r}")

over = mg.plan_vram(gpus=["0"], procs_per_gpu=4, batch_width=64, device_memory_gb=24.0)
check([d.verdict for d in over.devices] == ["over"],
      "4 x M64 is 4x the arenas, not the same ones split: the guard must catch it")
check(over.over and not over.unverified, "an over-budget device must be reported as over")

# Charged PER DEVICE, not per campaign: two GPUs each have their own memory.
two = mg.plan_vram(gpus=["0", "1"], procs_per_gpu=2, batch_width=32, device_memory_gb=16.0)
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

# --- the audit has to expect the receipt the mode will actually print -------
#
# `--result light` is an output-shape switch, but main.cpp:541 classifies a
# light job as a screening run, so its PHYSICS_MODE receipt reads
# screening:true / full_hdf5:false.  Audited against the full-exact expectation
# EVERY light arm returns 3 -- on fields that say nothing about the physics.
import run_single_gpu_batch as sg  # noqa: E402

check(sg.expected_physics_mode("light") is sg.SCREENING_PHYSICS_MODE
      and sg.expected_physics_mode("full") is sg.EXACT_PHYSICS_MODE
      and sg.expected_physics_mode("pin-off") is sg.EXACT_PHYSICS_MODE,
      "the expected PHYSICS_MODE must follow the result mode: pin-off still writes the "
      "result HDF5 and is still an exact run; only light is a screening receipt")
check(sg.SCREENING_PHYSICS_MODE["feedback_pass_limit"] == 0,
      "the screening expectation must STILL demand feedback_pass_limit=0: the GA "
      "feedback approximation is the one field here that changes the answer, and "
      "relaxing it would let an approximate run pass as a measurement")

LIGHT_RECEIPT = ('[RASBERY][PHYSICS_MODE] {"physics_mode":"ga_screen_feedback_limited",'
                 '"screening":true,"feedback_pass_limit":0,"full_hdf5":false}\n')
FULL_RECEIPT = ('[RASBERY][PHYSICS_MODE] {"physics_mode":"full_exact_nodal",'
                '"screening":false,"feedback_pass_limit":0,"full_hdf5":true}\n')
check(sg.check_physics_mode(LIGHT_RECEIPT, sg.expected_physics_mode("light")) == [],
      "a light run printing the screening receipt must pass its own audit")
check(sg.check_physics_mode(LIGHT_RECEIPT) != [],
      "negative control: the DEFAULT expectation must still refuse a screening receipt, "
      "so an unannounced screening run cannot pass as an acceptance measurement")
check(sg.check_physics_mode(FULL_RECEIPT, sg.expected_physics_mode("light")) != [],
      "negative control: a light launch that printed a FULL receipt did not do what it "
      "was told and must not pass either")
approx = ('[RASBERY][PHYSICS_MODE] {"physics_mode":"ga_screen_feedback_limited",'
          '"screening":true,"feedback_pass_limit":2,"full_hdf5":false}\n')
check(sg.check_physics_mode(approx, sg.expected_physics_mode("light")) != [],
      "negative control: light output with a NONZERO GA feedback limit is the "
      "approximation, and it must fail even the screening audit")

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

    # A light run without RASBERY_ALLOW_SCREENING is refused by main.cpp once
    # per chunk, AFTER the queue was claimed.  Asked here it costs nothing.
    saved_allow = os.environ.pop("RASBERY_ALLOW_SCREENING", None)
    try:
        check(mg.main(base + ["--result", "light", "--device-memory-gb", "64",
                              "--", sys.executable]) == 2,
              "--result light without RASBERY_ALLOW_SCREENING must be refused before "
              "the queue is claimed: RASBERY refuses it once per chunk anyway")
        check(mg.main(base + ["--result", "light", "--device-memory-gb", "64",
                              "--set", "RASBERY_ALLOW_SCREENING=1",
                              "--", sys.executable]) == 0,
              "--set RASBERY_ALLOW_SCREENING=1 must be an accepted way to say it")
        os.environ["RASBERY_ALLOW_SCREENING"] = "0"
        check(mg.main(base + ["--result", "light", "--device-memory-gb", "64",
                              "--", sys.executable]) == 2,
              "RASBERY_ALLOW_SCREENING=0 is main.cpp's falsey spelling and must be "
              "refused here the same way, or the guard is stricter or laxer than the "
              "executable it is guarding")
    finally:
        os.environ.pop("RASBERY_ALLOW_SCREENING", None)
        if saved_allow is not None:
            os.environ["RASBERY_ALLOW_SCREENING"] = saved_allow

if failures:
    raise SystemExit("multi-gpu dispatch: FAIL\n  " + "\n  ".join(failures))
print("multi-gpu dispatch: PASS")
