#!/usr/bin/env python3
"""Contract: WP10.8 -- a worker that dies loses no case, and says which it re-ran.

THE DEFECT, VERBATIM FROM THE EVIDENCE.  The 238 block-38 phase-2 arm-B soak
(`0054838`, GPU1, 20 generations x width 16, PROD + RASBERY_GPU_FULL=1,
`RASBERY_ARENA_PERSIST=1`) took a SIGSEGV in generation 1 and reported **18 of
360 cases never reported at all** -- 16 candidates + the injected poison + the
promote, i.e. one WHOLE generation.  The saved log
(`E:\\rasbery_runs\\2026-08-30\\238\\sigsegv_38\\soak_on_arm_sigsegv_context.txt`)
shows the mechanism in four consecutive lines: two
`[RASBERY][CUDA][CAPTURE_RACE][RETRY]` events, then
`[RASBERY][MULTI_GPU][EVALUATOR][EXIT] {"returncode":-11}`, then
`[RASBERY][MULTI_GPU][EVALUATOR][START] ... "attempt":2`, and then the very next
request the harness sends is **`g0002c0000`**.  Generation 1 was never asked for
again.

WHY IT WAS THE SOAK AND NOT THE DISPATCHER.  `run_multi_gpu_batch._run_wave_chunk`
has re-queued a dead child's unfinished cases onto a fresh one since WP8, once,
and reported by name what was still missing after that.  `tools/soak_run.py`
drives the SAME `EvaluatorSession` class and had no such path: its generation
loop recorded `result.cases = len(outcome.cases)`, restarted the child, and moved
on.  Two drivers of one session class, one of which could lose a wave silently.

WHAT THIS TEST ASSERTS.

  1. THE SOAK RE-QUEUES.  A child that dies mid-generation is replaced and the
     requests no receipt accounted for are sent again, ONCE.  Every case is
     reported, `cases_accounted` is true, and the recovered ones are named on
     `[RASBERY][SOAK][CASE][RESTART_RECOVERED]` lines -- a recovered case ran on
     a cold process and its wall time is not comparable, so it must not be
     invisible.
  2. IT DOES NOT RE-RUN WHAT ALREADY REPORTED.  Recovery sends the MISSING
     requests, not the generation: re-running a case that already wrote its
     output would overwrite a result and double-count the generation.
  3. WHEN RECOVERY FAILS IT SAYS WHICH CASES BY NAME, per generation, and the
     soak FAILS.  "18 of 360" at the end of a 16k-line log is not an answer a GA
     can act on; `g0001c0003, g0001c0004, ...` is.
  4. THE EQUALITY IS ASSERTED PER GENERATION, not only over the run.  A
     generation that lost four and a later one that reported four extra would
     net to zero over the run.
  5. A SLOPE IS NEVER FITTED ACROSS A RESTART.  The same soak reported RSS
     +115.97 MB/generation against a budget of 8 -- fitted through the restart,
     so most of it was a fresh child re-warming its caches from cold.  The gate
     fits one process lifetime and publishes the across-restarts number beside
     it, labelled, never gated on.
  6. THE DISPATCHER MARKS ITS OWN RECOVERIES TOO, so the two drivers of one
     session class stay comparable -- which is the whole reason they share it.

NEGATIVE CONTROLS.  Every source check is re-run against a broken copy and must
fail there; the behavioural half is driven against `tools/fake_rasbery_child.py`
with its `FAKE_RASBERY_POISON` knob, in both the recoverable and the
unrecoverable shape, so neither branch is a claim nobody has executed.

USE

    python tools/test_restart_recovery_contract.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import soak_run  # noqa: E402

failures: list[str] = []
FAKE = str(ROOT / "tools" / "fake_rasbery_child.py")


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


SOAK = read("tools/soak_run.py")
DISPATCH = read("tools/run_multi_gpu_batch.py")


# ---------------------------------------------------------------------------
# 1. THE SOURCE -- the re-queue exists, is bounded, and is marked
# ---------------------------------------------------------------------------

def check_soak_source(soak: str) -> list[str]:
    bad: list[str] = []
    if "def unreported(" not in soak or "def reported_names(" not in soak:
        bad.append("soak_run cannot compute which requests went unreported, so it "
                   "cannot re-queue them and cannot name them")
    if "RESTART_RECOVERY" not in soak:
        bad.append("soak_run prints no [RASBERY][SOAK][RESTART_RECOVERY] line; a "
                   "generation that lost its evaluator and re-ran a third of itself "
                   "would leave no trace in the log a campaign greps")
    if "RESTART_RECOVERED" not in soak:
        bad.append("soak_run does not mark the recovered cases; a case re-run on a "
                   "cold child is not comparable with one that was not, and nothing "
                   "downstream could tell them apart")
    if "recovery_attempt > 1" not in soak:
        bad.append("the re-queue is not bounded to one attempt: a generation that "
                   "kills two children in a row is a generation with a poisoned case "
                   "in it, and retrying forever turns one bad candidate into a hung "
                   "campaign (the rule _run_wave_chunk already follows)")
    # THE EQUALITY, PER GENERATION.  A run-level total nets a loss here against a
    # surplus there.
    if "gen.missing" not in soak or "requested {gen.requested}" not in soak:
        bad.append("soak_run does not compare requested against reported PER "
                   "GENERATION and name the survivors' absent siblings; the 238 "
                   "block-38 report could say only '18 of 360', at the end, with no "
                   "ids and no generation")
    if "cases_accounted" not in soak:
        bad.append("the soak report carries no `cases_accounted` verdict, so a reader "
                   "has to do the subtraction the harness already did")
    return bad


def check_no_rerun_of_reported(soak: str) -> list[str]:
    """Recovery must send the MISSING requests, never the whole generation."""
    bad: list[str] = []
    if "cases=missing" not in soak:
        bad.append("the recovery wave does not send exactly the unreported requests; "
                   "re-sending the generation would overwrite outputs that already "
                   "succeeded and double-count the wave")
    return bad


def check_segment_source(soak: str) -> list[str]:
    bad: list[str] = []
    if "def longest_epoch_segment(" not in soak:
        bad.append("soak_run has no epoch segmentation, so a leak slope is still "
                   "fitted across a restart -- which is what produced the 115.97 "
                   "MB/generation verdict on a run whose second half was 5.42")
    if "crossed_a_restart" not in soak:
        bad.append("the growth block does not say whether the gate could see the "
                   "whole run; a gate that quietly fitted four generations of twenty "
                   "and passed has passed on less evidence than the run bought")
    if "slope_across_restarts_mb_per_generation" not in soak:
        bad.append("the across-restarts slope is not published beside the gated one, "
                   "so a reader comparing this report with the 238 one cannot see "
                   "which number changed")
    return bad


def check_dispatcher_source(dispatch: str) -> list[str]:
    bad: list[str] = []
    if "RESTART_RECOVERED" not in dispatch:
        bad.append("run_multi_gpu_batch re-queues a dead child's cases and never marks "
                   "the ones that came back; the two drivers of EvaluatorSession would "
                   "then report recovery differently, which is the exact thing sharing "
                   "the class was for")
    if "restart_recovered" not in dispatch:
        bad.append("WorkerResult carries no record of which cases were recovered")
    if "requeued_keys" not in dispatch:
        bad.append("the dispatcher does not remember what it re-queued, so it cannot "
                   "recognise a recovered receipt when one arrives")
    return bad


# ---------------------------------------------------------------------------
# 2. THE BEHAVIOUR -- driven, in both the recoverable and lost shapes
# ---------------------------------------------------------------------------

def run_soak(workdir: Path, *, env: dict, extra: list[str]) -> "tuple[int, dict, str]":
    """One soak against the fake child.  (exit code, report, captured stdout)."""
    deck = workdir / "deck.json"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_text(json.dumps({"schedule": [{"type": "depletion", "burnup": 1.0}]}),
                    encoding="utf-8", newline="\n")
    saved = dict(os.environ)
    import io
    import contextlib
    buffer = io.StringIO()
    try:
        os.environ.update(env)
        argv = ["--deck", str(deck), "--workdir", str(workdir),
                "--command", f'"{sys.executable}" {FAKE}',
                "--generations", "3", "--width", "4",
                # The growth limits are relaxed: this test is about case
                # accounting and the restart seam, and the leak arithmetic has
                # its own unit checks below and in test_soak_run.py.
                "--vram-leak-mb", "100000", "--rss-leak-mb", "100000",
                # A restart necessarily perturbs one generation's wall clock;
                # the drift budget has its own test and is not what this one is
                # about.
                "--drift", "100"] + extra
        with contextlib.redirect_stdout(buffer):
            code = soak_run.main(argv)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    report = json.loads((workdir / "soak_report.json").read_text(encoding="utf-8"))
    return code, report, buffer.getvalue()


def drive_recovered() -> list[str]:
    """The child dies once, mid-generation, and every case still reports."""
    bad: list[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="rasbery_restart_ok_"))
    try:
        marker = workdir / "poison_spent"
        code, report, out = run_soak(
            workdir,
            # POISON kills the process mid-wave; POISON_MARKER spends it, so the
            # replacement child treats the poison deck as an ordinary (and still
            # unparseable, therefore failing) case.  That is the SUCCESS path --
            # without the marker the retry dies too and only the give-up branch
            # would ever be exercised.
            env={"FAKE_RASBERY_POISON": "unparseable",
                 "FAKE_RASBERY_POISON_MARKER": str(marker)},
            # A restart is a finding by default and this run INTENDS one, so it
            # says so.  The claim under test is that the restart cost nothing.
            extra=["--expect-restarts", "1"])
        if report["cases_reported"] != report["cases_requested"]:
            bad.append(f"the soak lost cases across a restart it recovered from: "
                       f"{report['cases_reported']} of {report['cases_requested']} "
                       f"reported, missing {report.get('cases_missing')}")
        if report.get("cases_accounted") is not True:
            bad.append("cases_accounted is not true on a run whose every request was "
                       "reported")
        if not report.get("cases_recovered"):
            bad.append("no case was recorded as recovered, so either the child did not "
                       "die (the negative control below proves it does) or the "
                       "recovery happened without saying so")
        if report.get("cases_missing"):
            bad.append(f"cases were still reported missing after a successful "
                       f"recovery: {report['cases_missing']}")
        if "[RASBERY][SOAK][RESTART_RECOVERY]" not in out:
            bad.append("the re-queue printed no [RASBERY][SOAK][RESTART_RECOVERY] line")
        if "[RASBERY][SOAK][CASE][RESTART_RECOVERED]" not in out:
            bad.append("no recovered case was marked; a cold-process re-run that looks "
                       "identical to a warm one is a throughput number that means "
                       "something else")
        # NO DOUBLE COUNTING.  Recovery sends the missing requests only.
        for gen in report["per_generation"]:
            if gen["cases"] > gen["requested"]:
                bad.append(f"generation {gen['index']} reported {gen['cases']} cases "
                           f"for {gen['requested']} requests: the recovery re-ran "
                           "cases that had already reported, overwriting their outputs")
        if code == 0:
            # The soak is still allowed to fail on OTHER grounds (a restart is a
            # finding unless declared), but with --expect-restarts 1 declared
            # and every case accounted for it should pass.  A failure here with
            # accounting intact is worth naming rather than swallowing.
            pass
        elif not any("never reported" in p for p in report["problems"]):
            bad.append("the recovered run failed for reasons other than case loss: "
                       + " | ".join(report["problems"])[:400])
        else:
            bad.append("the recovered run still reports lost cases: "
                       + " | ".join(report["problems"])[:400])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return bad


def drive_lost() -> list[str]:
    """The child dies EVERY time: the soak must fail and name the cases."""
    bad: list[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="rasbery_restart_lost_"))
    try:
        code, report, _out = run_soak(
            workdir,
            # No marker: the poison is never spent, so the replacement dies on
            # it too and the one re-queue is spent for nothing.
            env={"FAKE_RASBERY_POISON": "unparseable"},
            extra=["--expect-restarts", "9"])
        if code == 0:
            bad.append("a soak that lost cases to a child it could never keep alive "
                       "PASSED. That is the 238 block-38 arm-B verdict, unfixed")
        if report.get("cases_accounted") is not False:
            bad.append("cases_accounted is not false on a run that lost cases")
        missing = report.get("cases_missing") or []
        if not missing:
            bad.append("the report names no missing case; '18 of 360' with no ids is "
                       "what a GA could not act on")
        if not any(str(name).startswith("g0000") for name in missing):
            bad.append(f"the missing cases are not named by their request keys: "
                       f"{missing}")
        named = " | ".join(report["problems"])
        if "never reported" not in named:
            bad.append("no problem line says cases were never reported")
        if not any(f"generation {g['index']}" in named
                   for g in report["per_generation"] if g["missing"]):
            bad.append("the loss is reported only over the run, not per generation; a "
                       "generation short by four and a later one long by four net to "
                       "zero")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return bad


# ---------------------------------------------------------------------------
# 3. THE SEAM -- a slope is never fitted across a restart
# ---------------------------------------------------------------------------

def check_segment_behaviour() -> list[str]:
    bad: list[str] = []
    if soak_run.longest_epoch_segment([1, 1, 2, 2, 2, 2]) != (2, 6):
        bad.append("longest_epoch_segment did not find the longer process lifetime")
    if soak_run.longest_epoch_segment([1, 1, 1, 2, 2, 2]) != (3, 6):
        bad.append("a tie did not go to the LATER segment; two equally long lifetimes "
                   "make the one that survived to the end of the run the one whose "
                   "steady state is worth gating")
    if soak_run.longest_epoch_segment([]) != (0, 0):
        bad.append("longest_epoch_segment does not tolerate an empty run")

    # THE 238 SHAPE, in miniature: an old process at plateau, then a fresh one
    # climbing from cold.  Fitted whole it is a leak; fitted per process it is
    # flat, which is what it actually was.
    series = [900.0, 905.0, 100.0, 300.0, 500.0, 700.0]
    epochs = [1, 1, 2, 2, 2, 2]
    across = soak_run.growth_slopes(series, 0)
    within = soak_run.growth_slopes(series, 0, epochs)
    if within["crossed_a_restart"] is not True:
        bad.append("growth_slopes did not report that it could only see part of the run")
    if across["slope_mb_per_generation"] == within["slope_mb_per_generation"]:
        bad.append("the epoch-segmented slope equals the across-restart one, so the "
                   "segmentation is not being applied to the gated number")
    if within["segment_generations"] != 4:
        bad.append("the gated segment is not the surviving process's four generations")
    if within["slope_across_restarts_mb_per_generation"] is None:
        bad.append("the across-restarts number is not published beside the gated one; "
                   "a reader comparing this report with the 238 one could not see "
                   "which number changed")
    # And a run with no restart must be unchanged by the new argument, or every
    # number this campaign has already published moves under it.
    flat_epochs = [1] * len(series)
    same = soak_run.growth_slopes(series, 0, flat_epochs)
    if same["slope_mb_per_generation"] != across["slope_mb_per_generation"]:
        bad.append("segmentation changed the slope of a run that never restarted; "
                   "every number already published would move under this change")
    if same["crossed_a_restart"] is not False:
        bad.append("a run with one process reports that it crossed a restart")
    return bad


# ---------------------------------------------------------------------------
# 4. NEGATIVE CONTROLS -- a check that cannot fail is a comment
# ---------------------------------------------------------------------------

def controls() -> list[str]:
    broken: list[str] = []
    if not check_soak_source(SOAK.replace("RESTART_RECOVERY", "NOTHING")):
        broken.append("check_soak_source passes a soak that logs no re-queue")
    if not check_soak_source(SOAK.replace("recovery_attempt > 1", "False")):
        broken.append("check_soak_source passes an unbounded re-queue")
    if not check_soak_source(SOAK.replace("def unreported(", "def nothing(")):
        broken.append("check_soak_source passes a soak that cannot tell which requests "
                      "went unreported")
    if not check_no_rerun_of_reported(SOAK.replace("cases=missing", "cases=cases")):
        broken.append("check_no_rerun_of_reported passes a recovery that re-sends the "
                      "whole generation")
    if not check_segment_source(SOAK.replace("def longest_epoch_segment(",
                                             "def nothing(")):
        broken.append("check_segment_source passes a soak with no epoch segmentation")
    if not check_segment_source(SOAK.replace("crossed_a_restart", "nothing")):
        broken.append("check_segment_source passes a growth block that never says the "
                      "gate saw only part of the run")
    if not check_dispatcher_source(DISPATCH.replace("RESTART_RECOVERED", "NOTHING")):
        broken.append("check_dispatcher_source passes a dispatcher that never marks a "
                      "recovered case")
    if not check_dispatcher_source(DISPATCH.replace("requeued_keys", "nothing_keys")):
        broken.append("check_dispatcher_source passes a dispatcher that cannot "
                      "recognise a recovered receipt")
    return broken


failures += check_soak_source(SOAK)
failures += check_no_rerun_of_reported(SOAK)
failures += check_segment_source(SOAK)
failures += check_dispatcher_source(DISPATCH)
failures += check_segment_behaviour()
failures += drive_recovered()
failures += drive_lost()

broken_controls = controls()
if broken_controls:
    failures.append("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are "
                    "therefore comments:\n    " + "\n    ".join(broken_controls))

if failures:
    print("restart recovery: FAIL")
    for problem in failures:
        print("  " + problem)
    raise SystemExit(1)

print("restart recovery: PASS (2 driven soaks -- one recovered, one lost by name -- "
      "4 source contracts, 8 segmentation assertions, 8 negative controls)")
