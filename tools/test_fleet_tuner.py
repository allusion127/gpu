#!/usr/bin/env python3
"""Contract for the WP4 K auto-tuner (`--procs-per-gpu auto`, `--tune-from`).

The tuner spends GPU time to decide how the rest of the campaign spends GPU
time, so the ways it can be wrong are expensive and quiet:

  * it picks a K from waves that FAILED, and the campaign runs at a K that was
    never measured working;
  * it picks a K by a rule that depends on dict order or float noise, so two
    identical calibrations disagree and neither is reproducible;
  * the calibration eats the queue, and the campaign silently runs fewer jobs
    than the manifest lists -- or re-runs some of them into the same output;
  * the calibration leaves an MPS daemon up, and the next arm's "no MPS"
    control is silently an MPS run (docs/W4_L5 Sec 5 trap #3);
  * `--tune-from` reuses a K measured on a different device or driver;
  * the host-budget model keeps charging the plan's eight writer threads that
    src/IoWriter.h does not have.

Every check below is one of those, and the selection rules are driven with
SYNTHETIC candidates so the logic is testable with no GPU and no RASBERY.  The
one end-to-end calibration uses a fake executable that prints the receipts the
audit reads; it asserts INVARIANTS (queue intact, outputs redirected, same jobs
per candidate, no daemon left, receipts printed), never which K won -- a timing
race decided on this machine would be a flake, not a contract.

Run:  python tools/test_fleet_tuner.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
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


def cand(procs: int, width: int, cph: float, **kw) -> mg.TuneCandidate:
    """A measured candidate, with the fields the chooser reads."""
    c = mg.TuneCandidate(procs=procs, width=width)
    c.samples = list(kw.pop("samples", [cph]))
    c.jobs = kw.pop("jobs", 16)
    c.wall_s = kw.pop("wall_s", 100.0)
    c.width_fill = kw.pop("width_fill", 0.0)
    c.tail_idle_max_s = kw.pop("tail_idle_max_s", 0.0)
    c.vram_per_device_gb = kw.pop("vram_per_device_gb", 13.0)
    c.host_thread_demand = kw.pop("host_thread_demand", procs * width)
    c.rc = kw.pop("rc", 0)
    c.fail_lines = kw.pop("fail_lines", 0)
    c.problems = list(kw.pop("problems", []))
    c.refused = kw.pop("refused", "")
    assert not kw, f"unknown field(s): {sorted(kw)}"
    return c


# ===========================================================================
# 1.  The width policy: --total-width narrows the rendezvous, --batch-width
#     multiplies the declared width
# ===========================================================================
#
# `--total-width 64` is the WP4 arm: the device sees 64 declared slots at every
# K and only the rendezvous narrows.  The division ROUNDS UP, which is why the
# runner's next two arms are 12 x M6 and 16 x M4 and not 12 x M5: K processes
# must never declare less than T between them, or the arm silently becomes a
# width reduction as well as a split.
ladder = {k: mg.width_for_procs(k, batch_width=None, total_width=64)
          for k in (1, 2, 4, 8, 12, 16)}
check(ladder == {1: 64, 2: 32, 4: 16, 8: 8, 12: 6, 16: 4},
      f"--total-width 64 must give the 238 arm ladder, got {ladder}")
check(mg.width_for_procs(12, batch_width=None, total_width=64) * 12 >= 64,
      "negative control: floor division would give 12 x M5 = 60 declared slots, which "
      "is a NARROWER device declaration than the control and not the arm being measured")
check(mg.width_for_procs(8, batch_width=16, total_width=None) == 16,
      "--batch-width is the width PER PROCESS and must not be divided by K")
check(mg.width_for_procs(0, batch_width=16, total_width=None) == 16,
      "a zero K must not divide by zero on the way to being refused")

# ===========================================================================
# 2.  The objective: median, not mean
# ===========================================================================
c = cand(4, 16, 0.0, samples=[100.0, 900.0, 110.0])
check(abs(c.cases_per_hour - 110.0) < 1e-9,
      f"the score must be the MEDIAN over repeats ({c.cases_per_hour}); the mean (370) "
      "lets one lucky wave carry a candidate")
check(abs(cand(4, 16, 0.0, samples=[100.0, 200.0]).cases_per_hour - 150.0) < 1e-9,
      "an even number of repeats must average the two middles, not pick one arbitrarily")
check(cand(4, 16, 0.0, samples=[]).cases_per_hour == 0.0,
      "a candidate with no sample must score zero, not raise")

# ===========================================================================
# 3.  The choice, on the SHAPE OF THE REAL 238 MATRIX
# ===========================================================================
#
# With MPS: 1xM64 578, 2xM32 810, 4xM16 864, 8xM8 878 c/h.  The knee is at 8 and
# the tuner has to find it even though 4 is within 2 % of it -- 878 vs 864 is
# 1.6 %, INSIDE the default tie band, so this case is decided by the tie-break
# and not by the score.  That is the point: the tie-break must be stated, not
# emergent.
mps_matrix = [
    cand(1, 64, 578.0, tail_idle_max_s=30.0, host_thread_demand=64),
    cand(2, 32, 810.0, tail_idle_max_s=20.0, host_thread_demand=66),
    cand(4, 16, 864.0, tail_idle_max_s=12.0, host_thread_demand=68),
    cand(8, 8, 878.0, tail_idle_max_s=18.0, host_thread_demand=72),
]
best = mg.choose_candidate(mps_matrix)
check(best is not None and best.procs == 4,
      f"with 864 and 878 inside the 2 % tie band the LOWER TAIL must win "
      f"(got K={best.procs if best else None}). If this ever picks 8, the tie-break "
      "stopped being applied, not the score")
check(mg.choose_candidate(mps_matrix, tie_rel=0.0) is not None
      and mg.choose_candidate(mps_matrix, tie_rel=0.0).procs == 8,
      "with no tie band the raw maximum must win: --tune-tie-rel is the only thing "
      "that can make a slower candidate the choice")

# Without MPS the same host measured 1xM64 582, 2xM32 648, 4xM16 579, 8xM8 457:
# the knee is at 2 and 8 is a LOSS.  A tuner that cannot report that is a tuner
# that always says "more processes".
nomps_matrix = [
    cand(1, 64, 582.0, tail_idle_max_s=30.0, host_thread_demand=64),
    cand(2, 32, 648.0, tail_idle_max_s=25.0, host_thread_demand=66),
    cand(4, 16, 579.0, tail_idle_max_s=22.0, host_thread_demand=68),
    cand(8, 8, 457.0, tail_idle_max_s=21.0, host_thread_demand=72),
]
best = mg.choose_candidate(nomps_matrix)
check(best is not None and best.procs == 2, f"the no-MPS knee is K=2, got {best}")

# ...and K=1 must be reachable, or the tuner cannot DECLINE the lever, which is
# exactly what the plan's kill criterion asks it to be able to do.
decline = [cand(1, 64, 582.0), cand(2, 32, 500.0), cand(4, 16, 480.0)]
check(mg.choose_candidate(decline).procs == 1,
      "a tuner that cannot choose K=1 cannot report that the lever does not work here")
check(1 in mg.TUNE_CANDIDATES,
      "K=1 must be in the default candidate set: it is the control the plan's 1.05x "
      "kill criterion is measured against")

# ===========================================================================
# 4.  NEGATIVE CONTROLS: a failed wave is not a fast wave
# ===========================================================================
poisoned = [
    cand(1, 64, 582.0),
    cand(8, 8, 5000.0, rc=3),
    cand(12, 6, 4000.0, fail_lines=2),
    cand(16, 4, 3000.0, problems=["gpu0 p0 chunk1: graph_fallbacks=4"]),
]
best = mg.choose_candidate(poisoned)
check(best is not None and best.procs == 1,
      f"a candidate that returned rc!=0, printed a [FAIL] line or failed the receipt "
      f"audit must be DISQUALIFIED, not penalised (chose K={best.procs if best else None}). "
      "A wave that ran 5000 c/h and failed its audit measured nothing")
for bad in poisoned[1:]:
    check(not bad.eligible and bad.disqualified,
          f"K={bad.procs} must report WHY it is disqualified, or the receipt cannot be "
          "read back as an explanation of the choice")

check(mg.choose_candidate([]) is None, "no candidates must give no choice, not a crash")
check(mg.choose_candidate([cand(2, 32, 600.0, rc=1)]) is None,
      "all-disqualified must give None so the caller can refuse to start the campaign")
check(mg.choose_candidate([cand(2, 32, 0.0)]) is None,
      "a zero-throughput wave is not a winner even when it is the only one")
refused = cand(16, 4, 0.0, refused="device 0: 16 x width 4 needs 100 GB")
check(not refused.eligible and refused.disqualified.startswith("device 0"),
      "a candidate refused by a guard must carry the guard's reason into the receipt")

# ===========================================================================
# 5.  Determinism
# ===========================================================================
#
# Same numbers, different order in the list: the answer must not move.  A
# chooser that used max() over an unordered comparison, or that broke ties by
# whichever came first, would pass every check above and still make two
# identical calibrations disagree.
twins = [
    cand(2, 32, 800.0, tail_idle_max_s=10.0, host_thread_demand=66, vram_per_device_gb=13.0),
    cand(4, 16, 800.0, tail_idle_max_s=10.0, host_thread_demand=66, vram_per_device_gb=13.0),
    cand(8, 8, 800.0, tail_idle_max_s=10.0, host_thread_demand=66, vram_per_device_gb=13.0),
]
picks = {mg.choose_candidate(order).procs
         for order in (twins, list(reversed(twins)), [twins[1], twins[2], twins[0]])}
check(picks == {2},
      f"a total tie must always give the SMALLEST K regardless of list order, got {picks}")

vram_tie = [
    cand(2, 32, 800.0, tail_idle_max_s=10.0, host_thread_demand=66, vram_per_device_gb=26.0),
    cand(4, 16, 800.0, tail_idle_max_s=10.0, host_thread_demand=66, vram_per_device_gb=13.0),
]
check(mg.choose_candidate(vram_tie).procs == 4,
      "with tail and host threads equal, the tie-break is lower VRAM before smaller K")
thread_tie = [
    cand(2, 32, 800.0, tail_idle_max_s=10.0, host_thread_demand=99, vram_per_device_gb=13.0),
    cand(4, 16, 800.0, tail_idle_max_s=10.0, host_thread_demand=66, vram_per_device_gb=13.0),
]
check(mg.choose_candidate(thread_tie).procs == 4,
      "host-thread demand outranks VRAM and K in the tie-break order")

# ===========================================================================
# 6.  The guards bound the candidate set (VRAM, host cores)
# ===========================================================================
over = mg.plan_vram(gpus=["0"], procs_per_gpu=16, batch_width=64, device_memory_gb=96.0)
why = mg.candidate_refusal(procs=16, width=64, gpus=["0"], visible_cpus=24,
                           vram=over, allow_overcommit=False)
check(why and "GB" in why,
      "16 x M64 = 208 GB on a 96 GB device must be refused BEFORE it is measured")
check(mg.candidate_refusal(procs=16, width=64, gpus=["0"], visible_cpus=24,
                           vram=over, allow_overcommit=True) == "",
      "--allow-vram-overcommit must lift the VRAM refusal, and only that one")

fits = mg.plan_vram(gpus=["0"], procs_per_gpu=32, batch_width=2, device_memory_gb=96.0)
why = mg.candidate_refusal(procs=32, width=2, gpus=["0"], visible_cpus=24,
                           vram=fits, allow_overcommit=True)
check(why and "CPU" in why,
      "32 processes on a 24-core host must be refused even when the arenas fit: a "
      "process that cannot be given a core of its own is contention wearing a taskset")
ok = mg.plan_vram(gpus=["0"], procs_per_gpu=8, batch_width=8, device_memory_gb=96.0)
check(mg.candidate_refusal(procs=8, width=8, gpus=["0"], visible_cpus=24,
                           vram=ok, allow_overcommit=False) == "",
      "the arm that actually measured 878 c/h on 238 (8 x M8 on 24 cores, 96 GB) must "
      "not be refused by its own guards")

# The bound the tuner uses has to be the MEASURED memory model, not the one
# that says K x W = 64 costs the same at every K.  238: 12.3 / 14.9 / 20.0 /
# 30.2 GB at K = 1 / 2 / 4 / 8 (docs/W4_L5 Sec 4.8).
measured = {1: 12.3, 2: 14.9, 4: 20.0, 8: 30.2}
for k, seen in measured.items():
    modelled = mg.plan_vram(gpus=["0"], procs_per_gpu=k, batch_width=64 // k,
                            device_memory_gb=96.0).per_device_gb
    check(seen <= modelled <= seen + 1.5,
          f"K={k}: the guard models {modelled:.2f} GB where 238 measured {seen} GB. It "
          "must be conservative but not by a lot -- a guard that is wrong in either "
          "direction refuses a working arm or admits one that dies at stand-up")
check(mg.candidate_refusal(
          procs=8, width=8, gpus=["0"], visible_cpus=24,
          vram=mg.plan_vram(gpus=["0"], procs_per_gpu=8, batch_width=8,
                            device_memory_gb=24.0),
          allow_overcommit=False) != "",
      "negative control: 8 x M8 does NOT fit a 24 GB device (about 31 GB of arenas and "
      "contexts). Under the old flat model it 'needed 13 GB' and the tuner would have "
      "measured it by dying at arena stand-up with the queue claimed")

# ===========================================================================
# 7.  The saved result is keyed to the fleet
# ===========================================================================
here = {"gpus": ["0"], "uuids": ["GPU-1111"], "names": ["RTX PRO 6000"],
        "driver": "580.00", "verified": True}
check(mg.tune_key_matches(dict(here), here), "the same fleet must match itself")
check(not mg.tune_key_matches({**here, "driver": "575.00"}, here),
      "a driver change must invalidate a saved tuning result: the scheduler that "
      "decides whether K contexts overlap lives in the driver")
check(not mg.tune_key_matches({**here, "uuids": ["GPU-2222"]}, here),
      "a different device must invalidate a saved tuning result")
blind = {**here, "verified": False}
check(not mg.tune_key_matches(blind, blind),
      "negative control: an UNVERIFIED key must not match even itself. `nvidia-smi is "
      "not here` is not evidence that the device is the same one")
check(not mg.tune_key_matches({}, here) and not mg.tune_key_matches(here, {}),
      "an empty key is not a match")

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    good = work / "tune.json"
    good.write_text(json.dumps({
        "schema": mg.TUNE_SCHEMA, "key": here,
        "chosen": {"procs_per_gpu": 8, "batch_width": 8},
        "candidates": [],
    }), encoding="utf-8")
    payload = mg.load_tune_result(good)
    check(payload["chosen"]["procs_per_gpu"] == 8, "a valid tuning result must load")

    for name, body in (
        ("old.json", {"schema": mg.TUNE_SCHEMA - 1, "chosen": {"procs_per_gpu": 8,
                                                               "batch_width": 8}}),
        ("empty.json", {"schema": mg.TUNE_SCHEMA, "chosen": {}}),
        ("zero.json", {"schema": mg.TUNE_SCHEMA,
                       "chosen": {"procs_per_gpu": 0, "batch_width": 8}}),
        ("list.json", [1, 2, 3]),
    ):
        path = work / name
        path.write_text(json.dumps(body), encoding="utf-8")
        try:
            mg.load_tune_result(path)
            failures.append(f"load_tune_result accepted {name}: an unreadable tuning "
                            "result must be refused, not reinterpreted")
        except ValueError:
            pass

# ===========================================================================
# 8.  The writer budget is the executable's, not the plan's
# ===========================================================================
#
# WP4 "개선해야 할 현재 runner 가정".  src/IoWriter.h runs ONE writer thread in
# `thread` mode and NONE in `inline`, and --result light writes no result HDF5
# at all.  The plan's eight was never in the executable.
check(mg.plan_writer_threads(result_mode="full") == (1, "io_writer_thread"),
      "full output is ONE writer thread (IoWriter.h `_worker`), not eight")
check(mg.plan_writer_threads(result_mode="light") == (0, "light_no_hdf5"),
      "--result light writes no result HDF5, so the writer costs nothing")
check(mg.plan_writer_threads(result_mode="full", io_writer_mode="inline")
      == (0, "io_writer_inline"),
      "RASBERY_IO_WRITER=inline replays on the Driver thread: no writer thread exists")
check(mg.plan_writer_threads(result_mode="full", observed=3) == (3, "receipt"),
      "what the executable PRINTED must win over any model of it")
for mode in ("full", "light", "pin-off"):
    threads, _policy = mg.plan_writer_threads(result_mode=mode)
    check(threads != mg.WRITER_THREADS_PER_PROCESS,
          f"negative control: result_mode={mode} still charges the plan's "
          f"{mg.WRITER_THREADS_PER_PROCESS} phantom writer threads")

check(mg.writer_threads_from_receipt(
    '[RASBERY][IO_WRITER] {"mode":"thread","mode_source":"default","queue_limit":8}\n') == 1,
    "the [IO_WRITER] receipt's `thread` mode is one writer thread")
check(mg.writer_threads_from_receipt(
    '[RASBERY][IO_WRITER] {"mode":"inline","mode_source":"env"}\n') == 0,
    "`inline` is no writer thread")
check(mg.writer_threads_from_receipt("nothing here") is None,
      "no receipt must read as None -- unknown, so the model decides -- and never as 0")
check(mg.writer_threads_from_receipt('[RASBERY][IO_WRITER] {"mode":"quantum"}') is None,
      "negative control: an unknown mode word must not be guessed at")

light = mg.plan_host_budget(gpus=["0"], batch_width=64, visible_cpus=24, pin="taskset",
                            driver_workers=None, solver_threads=None, result_mode="light")
check(light.writer_threads == 0 and light.writer_policy == "light_no_hdf5",
      f"the light budget still charges {light.writer_threads} writer thread(s)")
full = mg.plan_host_budget(gpus=["0"], batch_width=64, visible_cpus=24, pin="taskset",
                           driver_workers=None, solver_threads=None, result_mode="full")
check(full.writer_threads == 1, "the full budget charges the one writer thread")
check(full.driver_workers == 64 and full.solver_threads == 64,
      "the writer correction must not have touched the LANE policy: 64 lanes on 24 "
      "cores is what the 582 c/h reference line runs (docs/W4_L5 Sec 4.7)")
check(full.host_threads_per_proc == 65 and abs(full.host_thread_ratio - 65 / 24) < 1e-9,
      "the host-thread demand is the OpenMP region plus the writer, reported as a "
      "ratio -- the reference arm is ~2.7 and that is correct, not a warning")

k8 = mg.plan_host_budget(gpus=["0"], batch_width=8, visible_cpus=24, pin="taskset",
                         driver_workers=None, solver_threads=None, procs_per_gpu=8,
                         result_mode="light")
check(k8.cpus_per_proc == 3 and k8.driver_workers == 8 and k8.solver_threads == 8,
      f"8 x M8 on 24 cores: 3 cores per process, 8 lanes, 8 solver threads "
      f"(got {k8.cpus_per_proc}/{k8.driver_workers}/{k8.solver_threads})")

# ===========================================================================
# 9.  The calibration subset: same jobs, redirected outputs, queue untouched
# ===========================================================================
manifest_jobs = [(f"deck{i}.json", f"/prod/out/case{i}.h5", "") for i in range(64)]
with tempfile.TemporaryDirectory() as tmp:
    tune_dir = Path(tmp) / "tune" / "k04" / "out"
    a = mg.calibration_jobs(manifest_jobs, count=8, procs=4, tune_dir=tune_dir)
    b = mg.calibration_jobs(manifest_jobs, count=8, procs=8,
                            tune_dir=Path(tmp) / "tune" / "k08" / "out")
    check(len(a) == 8, f"the calibration subset must be --tune-jobs long, got {len(a)}")
    check([d for d, _o, _m in a] == [d for d, _o, _m in b],
          "every candidate must see the SAME decks, or the comparison is between decks "
          "and not between widths")
    prod = {mg.path_key(o) for _d, o, _m in manifest_jobs}
    for _d, out, _m in a + b:
        check(mg.path_key(out) not in prod,
              f"calibration output {out} collides with a production output: a tuning "
              "wave must never overwrite a campaign result or share its restart "
              "namespace")
    check(len({mg.path_key(o) for _d, o, _m in a}) == len(a),
          "two calibration jobs writing one output would race two Drivers into one file")
    check({mg.path_key(o) for _d, o, _m in a}.isdisjoint({mg.path_key(o) for _d, o, _m in b}),
          "two CANDIDATES must not share output paths either: they run one after the "
          "other, and a shared path makes the second one's write the first one's result")

# ===========================================================================
# 10.  main() argument guards
# ===========================================================================
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    manifest = work / "jobs.txt"
    manifest.write_text("".join(f"d{i}.json out/d{i}.h5\n" for i in range(4)),
                        encoding="utf-8")
    base = ["--gpus", "0", "--jobs", str(manifest), "--workdir", str(work / "run"),
            "--pin", "none", "--device-memory-gb", "96"]

    check(mg.main(base + ["--", sys.executable]) == 2,
          "neither --batch-width nor --total-width must be refused: there is no default "
          "width, and guessing one is guessing the arm")
    check(mg.main(base + ["--total-width", "0", "--", sys.executable]) == 2,
          "--total-width 0 must be refused")
    check(mg.main(["--gpus", "0", "--batch-width", "4", "--procs-per-gpu", "sixteen",
                   "--jobs", str(manifest), "--", sys.executable]) == 2,
          "--procs-per-gpu must be a positive integer or `auto`, and a word that is "
          "neither must be refused rather than defaulted to 1")
    check(mg.main(base + ["--batch-width", "4", "--procs-per-gpu", "0",
                          "--", sys.executable]) == 2,
          "--procs-per-gpu 0 must still be refused now that the flag takes a word too")
    check(mg.main(base + ["--batch-width", "4", "--tune-candidates", "2,zero",
                          "--", sys.executable]) == 2,
          "an unparseable --tune-candidates list must be refused")
    check(mg.main(base + ["--batch-width", "4", "--tune-candidates", "0,4",
                          "--", sys.executable]) == 2,
          "a non-positive candidate must be refused, not clamped")
    check(mg.main(base + ["--batch-width", "4", "--tune-repeats", "0",
                          "--", sys.executable]) == 2, "--tune-repeats 0 measures nothing")
    check(mg.main(base + ["--batch-width", "4", "--tune-jobs", "0",
                          "--", sys.executable]) == 2, "--tune-jobs 0 measures nothing")
    check(mg.main(base + ["--batch-width", "4", "--procs-per-gpu", "auto", "--tune-from",
                          str(work / "nope.json"), "--", sys.executable]) == 2,
          "--tune-from reuses a measurement and `auto` makes a new one: asking for both "
          "must be refused rather than silently preferring one")
    check(mg.main(base + ["--batch-width", "4", "--procs-per-gpu", "auto", "--dry-run",
                          "--", sys.executable]) == 2,
          "`auto` cannot be dry-run: it MEASURES, and a dry calibration would report a "
          "chosen K that nothing ran")

    # --total-width without tuning still divides by the stated K.
    rc = mg.main(base + ["--total-width", "64", "--procs-per-gpu", "8", "--dry-run",
                         "--", sys.executable])
    check(rc == 0, f"--total-width with an explicit K must plan (rc={rc})")

    # --- --tune-from: the key is checked before the campaign starts ---------
    saved = work / "tuned.json"
    saved.write_text(json.dumps({
        "schema": mg.TUNE_SCHEMA,
        "key": {"gpus": ["0"], "uuids": ["GPU-ELSEWHERE"], "names": ["other"],
                "driver": "1.0", "verified": True},
        "chosen": {"procs_per_gpu": 2, "batch_width": 4},
        "candidates": [],
    }), encoding="utf-8")
    check(mg.main(base + ["--batch-width", "4", "--tune-from", str(saved),
                          "--dry-run", "--", sys.executable]) == 2,
          "a tuning result from another fleet must be refused BEFORE the campaign runs")
    check(mg.main(base + ["--batch-width", "4", "--tune-from", str(saved),
                          "--allow-tune-mismatch", "--dry-run", "--", sys.executable]) == 0,
          "--allow-tune-mismatch must let the operator reuse it deliberately")

# ===========================================================================
# 11.  End-to-end calibration against a fake executable
# ===========================================================================
#
# No GPU and no RASBERY: the child prints the receipts the audit reads and
# touches its outputs.  What is asserted is what must be true whatever the
# timings did -- the queue is untouched, every candidate ran the same number of
# jobs, the outputs went to the tune directory, the [TUNE] receipts were
# printed, the result file was saved, and no MPS daemon was started.  Which K
# won is NOT asserted: on a shared host that would be a flake dressed as a
# contract.
# THE FAKE IS SHARED WITH tools/test_multi_gpu_dispatch.py (WP8 stage 1.5).
# It speaks BOTH child shapes -- `--jobs` chunks and `--evaluator-jsonl` waves --
# off one body of receipt-printing code, because the claim the dispatcher makes
# is that the two are audited identically.  Two fakes that drifted would let one
# mode pass against receipts the other never prints.
FAKE_CHILD_PATH = str((root / "tools" / "fake_rasbery_child.py").resolve())

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    child = Path(FAKE_CHILD_PATH)
    outdir = work / "prod"
    outdir.mkdir()
    manifest = work / "jobs.txt"
    manifest.write_text(
        "".join(f'"d{i}.json" "{(outdir / f"case{i}.h5").as_posix()}"\n' for i in range(8)),
        encoding="utf-8")
    rundir = work / "run"

    argv = ["--gpus", "0", "--procs-per-gpu", "auto", "--total-width", "4",
            "--tune-candidates", "1,2", "--tune-jobs", "4", "--tune-budget-s", "120",
            "--jobs", str(manifest), "--workdir", str(rundir), "--pin", "none",
            "--device-memory-gb", "96", "--claim", "auto",
            "--", sys.executable, str(child)]
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = mg.main(argv)
    out = buffer.getvalue()
    check(rc == 0, f"the end-to-end calibration must finish (rc={rc})\n{out[-2000:]}")

    tune_lines = [json.loads(l.split("] ", 1)[1]) for l in out.splitlines()
                  if l.startswith("[RASBERY][MULTI_GPU][TUNE] ")]
    cand_lines = [json.loads(l.split("] ", 1)[1]) for l in out.splitlines()
                  if l.startswith("[RASBERY][MULTI_GPU][TUNE][CAND] ")]
    check(len(tune_lines) == 1, f"exactly one [TUNE] receipt, got {len(tune_lines)}")
    check(len(cand_lines) == 2,
          f"one [TUNE][CAND] receipt per candidate, got {len(cand_lines)}")
    if tune_lines:
        rec = tune_lines[0]
        for field in ("candidates", "chosen", "calibration_s", "calibration_jobs",
                      "key", "tie_rel", "schema", "source"):
            check(field in rec, f"the [TUNE] receipt has no {field}")
        check(rec["source"] == "calibration", "a fresh measurement must say so")
        check(rec["calibration_s"] > 0.0, "the calibration must report its own cost")
        check(all("cases_per_hour" in c for c in rec["candidates"]),
              "every candidate must report cases/hour: it IS the score")
        check({c["procs_per_gpu"] for c in rec["candidates"]} == {1, 2},
              "the receipt must list every candidate that was tried, refused ones too")
        check(rec["chosen"]["procs_per_gpu"] in (1, 2),
              "the chosen K must be one of the candidates")
        check(rec["chosen"]["batch_width"] * rec["chosen"]["procs_per_gpu"] >= 4,
              "the chosen width must still declare --total-width slots per GPU")
        check(len({c["jobs"] for c in rec["candidates"]}) == 1,
              f"every candidate must run the SAME number of jobs, got "
              f"{[c['jobs'] for c in rec['candidates']]} -- a comparison over different "
              "job counts is not a comparison of widths")
        saved = Path(rec.get("saved", rundir / "tune" / "result.json"))
        check(saved.exists(), "the tuning result must be saved for --tune-from")
        if saved.exists():
            again = mg.load_tune_result(saved)
            check(again["chosen"] == rec["chosen"],
                  "the saved result must load back to the same choice")

    total = [json.loads(l.split("] ", 1)[1]) for l in out.splitlines()
             if l.startswith("[RASBERY][MULTI_GPU][TOTAL] ")]
    check(len(total) == 1, "one [TOTAL] receipt")
    if total:
        check(total[0]["jobs"] == 8,
              f"the campaign must still run EVERY manifest job ({total[0]['jobs']} of 8): "
              "a calibration that consumed the queue would silently shrink the campaign")
        check(total[0]["tuned"] is True and total[0]["tune_source"] == "calibration",
              "the [TOTAL] receipt must say the K was tuned and where it came from")
        check(total[0]["calibration_jobs"] > 0,
              "the calibration's own job count must be reported SEPARATELY, so the "
              "campaign's throughput is not a function of how long tuning took")
        check(total[0]["calibration_jobs"] not in (total[0]["jobs"], 0)
              or total[0]["calibration_jobs"] == 8,
              "calibration jobs are counted out of band")

    # The production outputs exist and were written by the CAMPAIGN, and the
    # calibration wrote only into the tune directory.
    produced = sorted(p.name for p in outdir.glob("*.h5"))
    check(produced == [f"case{i}.h5" for i in range(8)],
          f"the campaign must have written every production output, got {produced}")
    tune_out = list((rundir / "tune").rglob("*.h5"))
    check(tune_out, "the calibration must have written its outputs under <workdir>/tune")
    check(all("tune" in p.parts for p in tune_out),
          "no calibration output may land outside the tune directory")

    # No MPS was asked for, so none may have been started -- and none may be
    # left running either way.
    check("[RASBERY][MULTI_GPU][MPS] " in out, "the MPS receipt must be printed")
    if shutil.which("pgrep"):
        found = subprocess.run(["pgrep", "-a", "nvidia-cuda-mps"],
                               capture_output=True, text=True, check=False)
        check(found.returncode != 0 or not found.stdout.strip(),
              "a calibration must leave no MPS control daemon behind: the next arm's "
              "no-MPS control would silently be an MPS run")

# ===========================================================================
# 11b.  The tuner audits candidates at the DECLARED fidelity (WP4 defect)
# ===========================================================================
#
# THE DEFECT, measured on 238 after 5ccf879.  DEFAULT_ENV has carried the A2
# staged tolerances since 7099e54, so every calibration child printed
# `policy:'A2'`; the candidate audit compared it against a hardcoded `strict`;
# and the tuner disqualified ALL SIX candidates -- rc=2, no winner, campaign not
# started -- with every wave at rc=0, fail_lines=0, duplicates=0.  The only line
# that could have explained it printed `problems: 6` and named neither word.
#
# The same fake child as Sec 11, which now derives its receipt from its own
# environment the way src/RunContract.h does.  Three arms: declare nothing (the
# A2 production default), declare strict AND make it strict (--strict), declare
# strict over the A2 environment (must fail, loudly).
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    child = Path(FAKE_CHILD_PATH)
    import contextlib  # noqa: E402
    import io  # noqa: E402

    def calibration(name: str, extra: list[str]) -> tuple[int, str]:
        outdir = work / f"prod_{name}"
        outdir.mkdir()
        jobs_file = work / f"jobs_{name}.txt"
        jobs_file.write_text(
            "".join(f'"d{i}.json" "{(outdir / f"case{i}.h5").as_posix()}"\n'
                    for i in range(4)),
            encoding="utf-8")
        argv = ["--gpus", "0", "--procs-per-gpu", "auto", "--total-width", "4",
                "--tune-candidates", "1,2", "--tune-jobs", "2", "--tune-budget-s", "120",
                "--jobs", str(jobs_file), "--workdir", str(work / f"run_{name}"),
                "--pin", "none", "--device-memory-gb", "96", "--claim", "auto"]
        buffer = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
            rc = mg.main(argv + extra + ["--", sys.executable, str(child)])
        return rc, buffer.getvalue() + errors.getvalue()


    def cand_receipts(text: str) -> list[dict]:
        return [json.loads(line.split("] ", 1)[1]) for line in text.splitlines()
                if line.startswith("[RASBERY][MULTI_GPU][TUNE][CAND] ")]

    rc, out = calibration("a2", [])
    cands = cand_receipts(out)
    check(rc == 0,
          f"the DEFAULT calibration must finish (rc={rc}): DEFAULT_ENV is the A2 arm, "
          "and a tuner that audits it against a hardcoded `strict` disqualifies every "
          "candidate it measures -- which is exactly what 238 reported\\n" + out[-1500:])
    check(all(c["declared_fidelity"] == "A2" for c in cands),
          f"every candidate receipt must carry the declared fidelity: "
          f"{[c.get('declared_fidelity') for c in cands]}")
    check(all(c["fidelity_measured"] == {"A2": c["jobs"]} for c in cands if c["jobs"]),
          f"each candidate must report the policy its chunks actually printed, per "
          f"case: {[c.get('fidelity_measured') for c in cands]}")
    check(all(c["eligible"] and not c["disqualified"] for c in cands),
          f"no candidate may be disqualified when every wave ran clean at the declared "
          f"fidelity: {[(c['procs_per_gpu'], c['disqualified']) for c in cands]}")

    rc, out = calibration("strict", ["--strict"])
    check(rc == 0, f"--strict must calibrate too (rc={rc})\\n" + out[-1500:])
    check(all(c["fidelity_measured"] == {"strict": c["jobs"]}
              for c in cand_receipts(out) if c["jobs"]),
          "--strict must actually reach the child: the receipt has to say `strict`, not "
          "merely the declaration")

    # NEGATIVE CONTROL: declaring strict over the A2 environment.  Every wave is
    # clean, and the campaign must still refuse -- an A2 number filed in a strict
    # table is the mixing plan Sec 6.2 forbids.
    rc, out = calibration("mismatch", ["--fidelity", "strict"])
    bad = cand_receipts(out)
    check(rc == 2,
          f"declaring `strict` over the A2 environment must refuse the campaign "
          f"(rc={rc}): the runs are fine, but they are not the runs that were asked for")
    check(bad and all(not c["eligible"] for c in bad),
          "every candidate measured at the wrong fidelity must be disqualified")
    check(bad and all(any("A2" in p and "strict" in p for p in c["problem_detail"])
                      for c in bad if c["problems"]),
          f"the [TUNE][CAND] line must NAME BOTH the declared and the measured word -- "
          f"`problems: 6` is what the 238 tuner printed and it explained nothing: "
          f"{[c.get('problem_detail') for c in bad]}")
    check("declared fidelity 'strict'" in out,
          "the no-winner refusal must say which fidelity was declared, or the operator "
          "is left re-running a calibration that cannot succeed")

# ===========================================================================
# 12.  The source says what the receipts promise
# ===========================================================================
source = (root / "tools" / "run_multi_gpu_batch.py").read_text(encoding="utf-8")
check("deadline is not None and time.monotonic() >= deadline" in source,
      "the per-candidate wall budget must be checked BEFORE a chunk is claimed; killing "
      "a child mid-chunk leaves a half-written output and a claim nobody completed")
check("mps.stop()" in source and source.count("finally:") >= 2,
      "the calibration's MPS session must be stopped in a finally of its own, not only "
      "the production wave's")
check('"[RASBERY][MULTI_GPU][TUNE]' in source or "MULTI_GPU][TUNE]" in source,
      "the tuner must print a [MULTI_GPU][TUNE] receipt")

# ===========================================================================
# 13.  The calibration runs through EVALUATORS when the campaign will (WP8 1.5)
# ===========================================================================
#
# A K measured against per-chunk process images is not a measurement of a
# campaign that will not pay them.  The per-process fixed cost is exactly what
# the K split multiplies -- 2.56 GB of VRAM and 1.75-4.92 s of `outside_drive`
# EACH -- so a candidate that pays it once per worker sits at a different knee
# from one that pays it once per chunk.  If the tuner ever calibrates in one
# shape and runs in the other, its answer is about a configuration nobody ran.
check("evaluator=args.evaluator" in source,
      "the calibration waves must inherit the campaign's --evaluator/--no-evaluator "
      "shape, or the K is chosen from waves of a different shape than the campaign")
check(source.count("evaluator=args.evaluator") >= 2,
      "BOTH the calibration wave and the production wave must pass the shape through: "
      f"found {source.count('evaluator=args.evaluator')} site(s)")


# ===========================================================================
# 14.  End-to-end calibration in BOTH worker shapes
# ===========================================================================
#
# Same fake, same jobs, same invariants -- and the receipt has to say which
# shape it measured.  What is NOT asserted is that one shape is faster: with a
# millisecond-long fake child that would be a flake, and the real number is a
# 238 number (docs/WP8_EVALUATOR_STAGE2 Sec 4).
for shape, extra in (("evaluator", []), ("chunked", ["--no-evaluator"])):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        outdir = work / "prod"
        outdir.mkdir()
        jobs_file = work / "jobs.txt"
        jobs_file.write_text(
            "".join(f'"d{i}.json" "{(outdir / f"case{i}.h5").as_posix()}"\n'
                    for i in range(6)),
            encoding="utf-8")
        import contextlib  # noqa: E402
        import io  # noqa: E402

        buffer = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
            rc = mg.main(["--gpus", "0", "--procs-per-gpu", "auto", "--total-width", "4",
                          "--tune-candidates", "1,2", "--tune-jobs", "2",
                          "--tune-budget-s", "120", "--jobs", str(jobs_file),
                          "--workdir", str(work / "run"), "--pin", "none",
                          "--device-memory-gb", "96", "--claim", "auto"]
                         + extra + ["--", sys.executable, FAKE_CHILD_PATH])
        out = buffer.getvalue() + errors.getvalue()
        check(rc == 0, f"the {shape} calibration must finish (rc={rc})\n{out[-1500:]}")
        totals = [json.loads(line.split("] ", 1)[1]) for line in out.splitlines()
                  if line.startswith("[RASBERY][MULTI_GPU][TOTAL] ")]
        check(len(totals) == 1, f"{shape}: one [TOTAL] receipt, got {len(totals)}")
        if totals:
            check(totals[0].get("worker_shape") == shape,
                  f"[TOTAL].worker_shape must name the shape that was measured: "
                  f"{totals[0].get('worker_shape')!r} for {shape}")
            check(totals[0]["jobs"] == 6,
                  f"{shape}: the campaign must still run every manifest job "
                  f"({totals[0]['jobs']} of 6)")
            if shape == "evaluator":
                check(totals[0]["images"] <= totals[0]["waves"],
                      "the evaluator must not start more images than it sent waves: "
                      f"images={totals[0]['images']} waves={totals[0]['waves']}")
                check((totals[0].get("evaluator_totals") or {}).get("xslib_loads"),
                      "the campaign must report the fleet's xslib_loads: it is the "
                      "field WP8 is judged on and it must not grow with the case count")
            else:
                check(totals[0]["images"] == totals[0]["waves"],
                      "in --no-evaluator every wave IS a process image, and the "
                      "receipt has to show that: "
                      f"images={totals[0]['images']} waves={totals[0]['waves']}")
        produced = sorted(q.name for q in outdir.glob("*.h5"))
        check(produced == [f"case{i}.h5" for i in range(6)],
              f"{shape}: every production output must exist, got {produced}")

if failures:
    raise SystemExit("fleet tuner: FAIL\n  " + "\n  ".join(failures))
print("fleet tuner: PASS")
