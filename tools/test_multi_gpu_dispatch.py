#!/usr/bin/env python3
"""Unit contract for the multi-GPU dispatcher (plan Rev.7.1 Sec 13.3).

Every check here is about a way the dispatcher could send GPU time somewhere it
does not belong: two GPUs onto one output file, a claim that hands the same job
twice, a host split that oversubscribes the CPU, or a single-GPU run that gets
chunked into drained batches and loses the refill it was supposed to keep.
"""
import json
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
        '"a deck.json"  "out/a b.h5"   # quoted, with spaces\n',
        encoding="utf-8",
    )
    jobs = mg.read_manifest(manifest)
    check(jobs == [("d0.json", "out/d0.h5"), ("a deck.json", "out/a b.h5")],
          f"read_manifest: wrong parse {jobs!r}")

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
check("if budget.gpus == 1:" in source and "size = remaining" in source,
      "claim policy: one GPU under `auto` must claim the whole queue -- chunking it "
      "drains the batch between chunks, which is the tail Task 20 removes")
check("check_run_receipts" in source and "from run_single_gpu_batch import" in source,
      "the exact-only and graph-fallback audits must be imported from the single-GPU "
      "harness, not restated, so the two harnesses cannot disagree about a valid run")
check("min(budget.driver_workers, batch_width, end - start)" in source,
      "the expected host_threads must match main.cpp's min(width, jobs) cap, or the "
      "last chunk of every run fails its own receipt audit")
check('"duplicates"' in source and '"stale_tenants"' in source,
      "the aggregate receipt must carry the Task 20 tenancy counters")

if failures:
    raise SystemExit("multi-gpu dispatch: FAIL\n  " + "\n  ".join(failures))
print("multi-gpu dispatch: PASS")
