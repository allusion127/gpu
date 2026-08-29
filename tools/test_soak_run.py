#!/usr/bin/env python3
"""Contract: the WP11 soak harness can actually fail.

WHY THIS TEST IS THE POINT.  A soak is a harness whose job is to say `PASS`
after ninety minutes.  It will say that on a clean run and it will also say it
if the counters it reads are never printed, if the receipt tag it greps for was
renamed, if its leak fit is over a series of Nones, or if it forgot to compare
the numbers at all.  Every one of those failure modes produces a green report,
which is the only thing anybody reads.  So the soak is driven here against
`tools/fake_rasbery_child.py` -- no GPU, no CUDA, a millisecond a case -- once
clean, and then once per defect it exists to catch, with the requirement that
the report comes back FAIL and names the right thing.

WHAT IS DRIVEN, END TO END.

  * A clean multi-generation soak PASSES, with mixed light/full, a warm-start
    chain, a screening lane, a promotion and a poisoned case per generation.
  * Every zero-receipt in the table is made nonzero in turn and must fail.
  * The [CAPTURE_ARBITER] line is REMOVED and the soak must refuse rather than
    record zero -- "we did not measure it" is not "it was zero".
  * A GPU-full fallback fails only under RASBERY_GPU_FULL=1, and is reported
    but not asserted without it: failing an arm nobody said was wrong would
    make the soak useless in the arm it is usually run in.
  * A poisoned case that kills the PROCESS (rather than failing alone) drives
    restarts past the injection and must fail -- that is the failure-isolation
    defect the poison is injected to find.
  * The per-case fidelity audit runs per generation: a mixed wave passes with
    per-case declarations, and a case that ran at the wrong fidelity fails.
  * The throughput and leak arithmetic is unit-tested on synthetic series,
    including the two ways it could quietly do nothing (all-None input, and a
    warm plateau being mistaken for a leak).

The report itself is checked for the fields the 238 runbook reads, because a
soak whose evidence cannot be read is a soak that has to be run again.
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

FAILED: list[str] = []


def fail(msg: str) -> None:
    FAILED.append(msg)


FAKE = str(ROOT / "tools" / "fake_rasbery_child.py")


def run_soak(workdir: Path, *, env: dict[str, str] | None = None,
             extra: list[str] | None = None) -> tuple[int, dict]:
    # THE DECK IS WRITTEN HERE, not borrowed from the tree.  The fake child
    # fails any deck that exists and does not parse -- that is how the soak's
    # poisoned case works -- so a test that reached for whatever JSON-ish file
    # happened to be in the repo would fail every case on some machines and
    # none on others, and the difference would look like a harness bug.
    deck = workdir / "deck.json"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_text(json.dumps({"schedule": [{"type": "depletion", "burnup": 1.0}]}),
                    encoding="utf-8", newline="\n")
    saved = dict(os.environ)
    try:
        os.environ.update(env or {})
        argv = ["--deck", str(deck), "--workdir", str(workdir),
                "--command", f'"{sys.executable}" {FAKE}',
                "--generations", "4", "--width", "4",
                # nvidia-smi may or may not exist here and the child's VRAM is
                # nobody's; the growth limits are relaxed so this test is about
                # the harness's logic, which is unit-tested separately below.
                "--vram-leak-mb", "100000", "--rss-leak-mb", "100000"]
        argv += extra or []
        code = soak_run.main(argv)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    report = json.loads((workdir / "soak_report.json").read_text(encoding="utf-8"))
    return code, report


def case(name: str, *, env: dict[str, str] | None = None,
         extra: list[str] | None = None, expect_pass: bool,
         expect_in: str | None = None) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="rasbery_soak_"))
    try:
        code, report = run_soak(workdir, env=env, extra=extra)
    except Exception as exc:  # noqa: BLE001 - a harness that throws is a failure
        fail(f"{name}: the soak raised {exc!r}")
        shutil.rmtree(workdir, ignore_errors=True)
        return
    problems = " | ".join(report["problems"])
    if expect_pass and code != 0:
        fail(f"{name}: expected PASS, got FAIL -- {problems}")
    if not expect_pass and code == 0:
        fail(f"{name}: expected FAIL and the soak passed. A soak that cannot fail "
             f"on this is a soak that reports PASS for ninety minutes and means "
             f"nothing.")
    if expect_in and expect_in not in problems:
        fail(f"{name}: the report does not name the defect ({expect_in!r} not in "
             f"{problems!r})")
    if report.get("schema") != "rasbery-soak/v1":
        fail(f"{name}: the report has no schema tag")
    shutil.rmtree(workdir, ignore_errors=True)


# ===========================================================================
# 1. THE CLEAN RUN
# ===========================================================================
case("clean soak", expect_pass=True)


def clean_report() -> dict:
    workdir = Path(tempfile.mkdtemp(prefix="rasbery_soak_"))
    try:
        _code, report = run_soak(workdir)
        return report
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


REPORT = clean_report()

# The workload actually did the things it claims to have done.  A soak that
# silently ran four identical strict cases per generation would pass every
# assertion above and have exercised none of WP10.3.
if REPORT["cases_reported"] != REPORT["cases_requested"]:
    fail("the clean soak lost cases")
if not all(g["screens"] > 0 for g in REPORT["per_generation"]):
    fail("no generation ran a screening case; the soak is single-fidelity and does "
         "not exercise the mixed-fidelity path at all")
if not all(g["promotions"] > 0 for g in REPORT["per_generation"]):
    fail("no generation promoted anything")
if not all(g["failed"] == 1 for g in REPORT["per_generation"]):
    fail("the poisoned case did not fail exactly once per generation; either the "
         "poison is not poisonous or it took more than its own case down")
if REPORT["restarts"] != 0:
    fail("the clean soak restarted the child; the poisoned case is supposed to fail "
         "ALONE")
for field in ("zero_receipts", "not_asserted", "throughput", "growth",
              "per_generation", "problems", "poisoned", "restarts", "wall_s"):
    if field not in REPORT:
        fail(f"the report has no {field!r} field; the 238 runbook reads it")
for name, _tag, _f in soak_run.ZERO_RECEIPTS:
    if name not in REPORT["zero_receipts"]:
        fail(f"the report does not state {name!r}; a receipt that is not in the "
             "report was not checked, whatever the verdict says")

# The markdown exists and names the verdict -- the report a human reads.
_md_dir = Path(tempfile.mkdtemp(prefix="rasbery_soak_"))
try:
    run_soak(_md_dir)
    md = (_md_dir / "soak_report.md").read_text(encoding="utf-8")
    if "PASS" not in md or "Zero receipts" not in md:
        fail("the markdown report does not carry the verdict and the zero-receipt table")
finally:
    shutil.rmtree(_md_dir, ignore_errors=True)


# ===========================================================================
# 2. EVERY ZERO-RECEIPT, MADE NONZERO
# ===========================================================================
case("duplicates", env={"FAKE_RASBERY_DUPLICATES": "1"},
     expect_pass=False, expect_in="refill_duplicates = 1")
case("stale tenants", env={"FAKE_RASBERY_STALE_TENANTS": "2"},
     expect_pass=False, expect_in="refill_stale_tenants = 2")
case("alloc in capture", env={"FAKE_RASBERY_ALLOC_IN_CAPTURE": "1"},
     expect_pass=False, expect_in="alloc_in_capture = 1")
case("captures unwound", env={"FAKE_RASBERY_CAPTURES_UNWOUND": "3"},
     expect_pass=False, expect_in="captures_unwound = 3")

# A COUNTER THAT WAS NEVER PRINTED IS NOT A COUNTER THAT WAS ZERO.
case("missing capture arbiter receipt", env={"FAKE_RASBERY_NO_ARBITER": "1"},
     expect_pass=False, expect_in="no [RASBERY][CUDA][CAPTURE_ARBITER] receipt")

# GPU-full fallbacks: asserted only under the gate.
case("host fallback under GPU_FULL",
     env={"FAKE_RASBERY_HOST_FALLBACKS": "1", "RASBERY_GPU_FULL": "1"},
     expect_pass=False, expect_in="cmfd_fallbacks = 1 under RASBERY_GPU_FULL=1")
case("host fallback without GPU_FULL is reported, not asserted",
     env={"FAKE_RASBERY_HOST_FALLBACKS": "1"}, expect_pass=True)

# FAILURE ISOLATION: a case that takes the process down, not just itself.
case("a poisoned case that kills the process",
     env={"FAKE_RASBERY_POISON": "unparseable"},
     expect_pass=False, expect_in="restarted")


# ===========================================================================
# 3. THE PER-CASE FIDELITY AUDIT, INSIDE THE SOAK
# ===========================================================================
# The clean run's mixed wave passes with per-case declarations.  Here the child
# is made to REPORT `strict` for every case while the soak still asks a quarter
# of each generation to run L3coarse -- a binary that echoes the fidelity
# contract instead of applying it.  Nothing else in the output changes, which is
# exactly why the per-case audit has to be inside the soak rather than an
# optional step somebody remembers.
case("a binary that echoes the fidelity instead of applying it",
     env={"FAKE_RASBERY_LIE_FIDELITY": "1"},
     expect_pass=False, expect_in="DECLARED 'L3coarse'")

# The A2 environment is NOT a defect: `strict` is allowed to clear the process's
# staged multipliers, because that is the promotion lane inside a production
# campaign process.  This run must PASS, or the soak would refuse the arm the
# campaign actually runs in.
case("a strict case inside an A2 process is the promotion lane, not a defect",
     env={"RASBERY_STAGED_FLUX_TOL": "50", "RASBERY_STAGED_XE_TOL": "1000"},
     expect_pass=True)


# ===========================================================================
# 4. THE ARITHMETIC, ON SYNTHETIC SERIES
# ===========================================================================
slope = soak_run.leak_slope_mb_per_generation
if slope([None] * 8) is not None:
    fail("leak_slope_mb_per_generation invents a slope from no samples; a soak on a "
         "box with no nvidia-smi would then report a perfectly flat VRAM trace and "
         "pass for the one reason that proves nothing")
if slope([1.0, 2.0]) is not None:
    fail("leak_slope_mb_per_generation fits a slope to two points")
# A WARM PLATEAU IS NOT A LEAK.  The first half climbs steeply (library, arenas,
# graph cache) and the second half is flat; a whole-run fit would call this a
# 100 MB/generation leak on every healthy soak.
plateau = [100.0, 400.0, 700.0, 900.0, 950.0, 950.0, 950.0, 950.0]
measured = slope(plateau)
if measured is None or abs(measured) > 1.0:
    fail(f"leak_slope_mb_per_generation reports {measured} on a warm plateau; a "
         "threshold set high enough to tolerate that would catch nothing")
# A REAL LEAK is a line still climbing after the caches stopped.
leak = [100.0, 400.0, 700.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0]
measured = slope(leak)
if measured is None or measured < 45.0:
    fail(f"leak_slope_mb_per_generation reports {measured} on a 50 MB/generation leak")

# The receipt reader must not let `[RASBERY][EVALUATOR]` swallow the per-case
# lines: they are a different receipt with a `case` field and no counters.
text = ('[RASBERY][EVALUATOR][CASE] {"slot_duplicates":9}\n'
        '[RASBERY][EVALUATOR] {"slot_duplicates":0}\n')
found = soak_run.receipts_of(text, "[RASBERY][EVALUATOR]")
if len(found) != 1 or found[0].get("slot_duplicates") != 0:
    fail(f"receipts_of confuses [EVALUATOR] with [EVALUATOR][CASE]: {found}")


if FAILED:
    raise SystemExit("soak harness contract: FAIL\n  - " + "\n  - ".join(FAILED))
print(f"soak harness contract: PASS ({len(soak_run.ZERO_RECEIPTS)} zero-receipts "
      f"driven, {len(soak_run.GPU_FULL_FALLBACKS)} gated fallbacks)")
