#!/usr/bin/env python3
"""Contract: WP10.3 -- fidelity is a property of the CASE, and it cannot lie.

WHAT GOES WRONG WITHOUT LOOKING WRONG.  Every failure this work package can
introduce produces a wave that finishes, prints finite keff and exits zero.

  * A case inherits the previous case's staged tolerances.  It converges, it is
    slightly cheaper, and its receipt says `strict`.
  * A case asks for the coarse burnup grid and gets the full deck.  It is a
    full-cost case filed in the screening lane -- the throughput number for that
    lane is then wrong in the safe-looking direction.
  * A coarse case and a full case of one candidate share a case key.  The cache
    serves the screening scalar to the acceptance request, once, quietly.
  * A promotion runs at whatever the process defaults to.  The elite is then
    unverified while looking verified, which is worse than not promoting it.

None of those is visible in any output the program produces, so they are pinned
here, in the source, plus a behavioural test of the audit that has to catch them
in a log.

FIVE PARTS.

  1. GRID PARITY.  src/StatepointGrid.h and tools/make_screening_deck.py are two
     implementations of one transform.  The constants and the key set are pinned
     against each other -- a C++ `coarse` that is not the tool's `coarse` is a
     second definition of the screening lane.

  2. WHERE THE GRID IS APPLIED.  Between the JSON parse and the deck digest, in
     IO::ReadInput.  Later would give the two lanes one case key.

  3. THE RESOLUTION RULES.  strict may CLEAR staged tolerances (that is the
     promotion lane); A2 may not INVENT them; L3coarse needs a grid; and the
     final check is an EQUALITY, both directions.

  4. THE WIRING.  The per-case value reaches SolveLoop, the deck load, the case
     key, the [CASE] receipt and the light JSONL -- five consumers, because a
     value that reaches four of them is a value one receipt disagrees about.

  5. THE AUDIT, run against synthetic logs, including the negative controls: a
     mixed wave audited per case, a coarse case reporting strict, a promotion
     that did not land at strict, and a pre-WP10.3 receipt (refused, not passed).

NEGATIVE CONTROLS.  Section 5 is behavioural and its controls are the bad logs
themselves.  Sections 1-4 are source checks and each is re-run against a copy
broken in the way it exists to catch.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import exact_audit  # noqa: E402
import make_screening_deck  # noqa: E402

FAILED: list[str] = []


def fail(msg: str) -> None:
    FAILED.append(msg)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


GRID = read("src/StatepointGrid.h")
FIDELITY = read("src/CaseFidelity.h")
DRIVER = read("src/Driver.h")
SERVER = read("src/EvaluatorServer.h")
IO_CPP = read("src/IO.cpp")
LIGHT = read("include/chiffon/BatchLightResult.h")


# ===========================================================================
# 1. GRID PARITY -- one transform, two implementations
# ===========================================================================
def cxx_double_array(source: str, name: str) -> list[float] | None:
    match = re.search(rf"{name}\[\]\s*=\s*\{{([^}}]*)\}}", source)
    if match is None:
        return None
    return [float(x) for x in match.group(1).replace("\n", " ").split(",") if x.strip()]


def cxx_string_array(source: str, name: str) -> list[str] | None:
    match = re.search(rf"{name}\[\]\s*=\s*\{{(.*?)\}};", source, re.S)
    if match is None:
        return None
    return re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))


def check_grid_parity(grid: str) -> list[str]:
    bad: list[str] = []
    coarse = cxx_double_array(grid, "kCoarseBurnups")
    if coarse != list(make_screening_deck.COARSE_BURNUPS):
        bad.append(f"kCoarseBurnups {coarse} is not make_screening_deck.COARSE_BURNUPS "
                   f"{make_screening_deck.COARSE_BURNUPS}: the evaluator's `coarse` lane "
                   "and the tool's would be two different screening arms whose numbers "
                   "are not comparable")
    three = cxx_double_array(grid, "kThreeBurnups")
    if three != list(make_screening_deck.THREE_BURNUPS):
        bad.append(f"kThreeBurnups {three} is not make_screening_deck.THREE_BURNUPS")
    warn = re.search(r"kWarnBurnupStepGwd\s*=\s*([0-9.]+)", grid)
    if warn is None or float(warn.group(1)) != make_screening_deck.WARN_BURNUP_STEP_GWD:
        bad.append("kWarnBurnupStepGwd is not make_screening_deck.WARN_BURNUP_STEP_GWD; "
                   "the C++ side would stay silent on a grid the tool warns about, and "
                   "cost is SUPERLINEAR in the burnup step (measured: the 3-statepoint "
                   "grid ran 5,104 outers against the 35-statepoint deck's 4,609)")
    keys = cxx_string_array(grid, "kDepletionTimeKeys")
    if keys != list(make_screening_deck.DEPLETION_TIME_KEYS):
        bad.append(f"kDepletionTimeKeys {keys} is not "
                   f"make_screening_deck.DEPLETION_TIME_KEYS "
                   f"{list(make_screening_deck.DEPLETION_TIME_KEYS)}: a key one side "
                   "strips and the other keeps puts a stale `time` beside the new "
                   "`burnup` in every rewritten entry")
    # The rebuild itself: one entry per grid point, steps=1, burnup=DELTA.  A
    # cumulative burnup written where a delta belongs would deplete the core to
    # 16 GWd/t nine times over.
    if 'entry["steps"]  = 1;' not in grid or 'entry["burnup"] = detail::round10(cumulative - previous);' not in grid:
        bad.append("applyGrid does not emit one steps=1 entry per grid point with the "
                   "DELTA burnup; make_screening_deck.build_schedule does exactly that")
    # The until-boron tail is what makes the cost candidate-dependent, and a
    # screening arm with a variable cost has no throughput number at all.
    if "until boron ppm" not in grid:
        bad.append("StatepointGrid.h never mentions the until-boron tail, so nothing "
                   "says whether the rewritten schedule keeps it; keeping it makes the "
                   "statepoint count a function of the candidate")
    return bad


# ===========================================================================
# 2. WHERE THE GRID IS APPLIED
# ===========================================================================
def check_apply_site(io_cpp: str) -> list[str]:
    bad: list[str] = []
    parse = io_cpp.find("input_stream >> config;")
    apply_ = io_cpp.find("rasbery::spgrid::applyGridSpec(config")
    digest = io_cpp.find("_deck_key_digest =")
    if apply_ < 0:
        return ["IO::ReadInput never applies the statepoint grid: the request field would "
                "be accepted and ignored, and a screening lane would silently run full "
                "decks at full cost"]
    if not (parse < apply_ < digest):
        bad.append("the statepoint grid is not applied between the JSON parse and the "
                   "deck key digest. AFTER the digest, a coarse case and a full case of "
                   "one candidate share a case key -- and a cache keyed on that serves "
                   "the ten-statepoint screening scalar to the thirty-five-statepoint "
                   "acceptance request.")
    grid_region = io_cpp[apply_ - 400:apply_ + 400]
    if "throw" not in grid_region:
        bad.append("a grid that cannot be applied does not throw; a deck with no "
                   "depletion entry would run in full under a screening declaration, "
                   "i.e. full cost filed in the screening lane")
    return bad


# ===========================================================================
# 3. THE RESOLUTION RULES
# ===========================================================================
def check_rules(fidelity: str) -> list[str]:
    bad: list[str] = []
    # strict CLEARS -- that is the promotion lane inside an A2 campaign process.
    strict_region = fidelity[fidelity.find("if (want == PhysicsFidelity::FullExact) {"):]
    strict_region = strict_region[:strict_region.find("if (want == PhysicsFidelity::StagedA2")]
    if not strict_region:
        bad.append("resolveCaseFidelity has no `strict` branch: a promoted elite could "
                   "not be re-run at strict inside a process whose environment is the A2 "
                   "arm, which is every production campaign process")
    else:
        if "out.staged_flux_mult = 1.0;" not in strict_region or \
                "out.staged_xe_mult   = 1.0;" not in strict_region:
            bad.append("a `strict` declaration does not clear the staged multipliers, so "
                       "a promotion inside an A2 process would run A2 and say strict")
        if "contradicts itself" not in strict_region:
            bad.append("`strict` with an explicit staged multiplier is not refused; the "
                       "request would contradict itself and one half would win silently")
    # A2 may not INVENT.  A2 is a family, not a point.
    if "A2 is a FAMILY" not in fidelity:
        bad.append("an `A2` declaration against a process with no staged multipliers is "
                   "not refused; the evaluator would have to guess an arm, and a receipt "
                   "saying A2 without saying WHICH A2 is not reproducible")
    # L3coarse is a deck property and needs the deck property.
    if "L3coarse is a DECK property" not in fidelity:
        bad.append("an `L3coarse` declaration with no statepoint_grid is not refused; the "
                   "case would run the full deck and be filed as screening")
    # THE EQUALITY, both directions.
    equality = fidelity[fidelity.find("const PhysicsFidelity got = out.solved();"):]
    if not equality:
        bad.append("resolveCaseFidelity never compares what was declared against what "
                   "will be solved; the declaration would be an echo of itself")
    else:
        if "COARSER than it declared" not in equality:
            bad.append("the equality does not name the coarser-than-declared direction "
                       "(an approximation walking into an acceptance table)")
        if "FINER than it declared" not in equality:
            bad.append("the equality does not refuse a case that solves FINER than "
                       "declared; a strict number filed in the A2 column is the mixing "
                       "plan Sec 6.2 forbids, not a happy accident")
    # The process floor cannot be undone by a request.
    floor = fidelity[fidelity.find("inline PhysicsFidelity processFidelityFloor()"):]
    floor = floor[:floor.find("\n}")] if "\n}" in floor else floor
    if "RASBERY_GA_FEEDBACK_PASSES" not in floor:
        bad.append("the process floor does not include RASBERY_GA_FEEDBACK_PASSES; a "
                   "request could then declare `strict` inside a feedback-limited process "
                   "and the receipt would agree with it")
    if "coarserOf" not in fidelity:
        bad.append("the case fidelity is not combined with the process floor by taking "
                   "the coarser; a request could climb above the floor")
    return bad


# ===========================================================================
# 4. THE WIRING -- five consumers of one value
# ===========================================================================
def check_wiring(driver: str, server: str, light: str) -> list[str]:
    bad: list[str] = []
    if "ctx.fidelity = _fidelity;" not in driver:
        bad.append("the Driver's fidelity never reaches SolverContext, so SolveLoop "
                   "converges on whatever the context was default-constructed with")
    if "input_output.ReadInput(_input, _fidelity.statepoint_grid);" not in driver:
        bad.append("the deck is not loaded on the case's own burnup grid")
    if "caseKeyProvenance(input_output, warm_provenance, _fidelity)" not in driver:
        bad.append("the case key is not computed from the case's own fidelity; a strict "
                   "promotion and the A2 screening result it replaces would collide")
    if "armEnvValue" not in driver:
        bad.append("the case key still reads the staged knobs from the environment, so "
                   "two cases at two fidelities in one process fold the same env string")
    for field in ('_case_receipt.policy', '_case_receipt.statepoint_grid',
                  '_case_receipt.acceptance_eligible', '_case_receipt.promoted_from'):
        if field not in driver:
            bad.append(f"the Driver's case receipt never carries {field}; the evaluator "
                       "has nothing to report and the per-case audit nothing to read")
    if "BatchLightResult::Fidelity light_fidelity;" not in driver:
        bad.append("the light JSONL is written without the case's fidelity; a GA reads "
                   "that line directly and would rank a screening scalar against a "
                   "strict one with nothing on either row saying so")
    for field in ("policy", "physics_fidelity", "statepoint_grid",
                  "acceptance_eligible", "promoted_from"):
        if f'receipt["{field}"]' not in light:
            bad.append(f"the light JSONL receipt has no {field!r} field")
    if "driver.setCaseFidelity(fidelity);" not in server:
        bad.append("the evaluator resolves a fidelity and does not apply it")
    return bad


# ===========================================================================
# 5. THE AUDIT, behaviourally
# ===========================================================================
def case_line(**fields) -> str:
    receipt = {"wave_id": 1, "case": 0, "key": "c", "case_key": "k",
               "deck": "d.json", "output": "o.h5", "result_mode": "light",
               "status": "ok", "policy": "strict", "physics_fidelity": "full_exact",
               "statepoint_grid": "full", "acceptance_eligible": True,
               "fidelity_declared": None, "promoted_from": None}
    receipt.update(fields)
    return "[RASBERY][EVALUATOR][CASE] " + json.dumps(receipt)


def check_audit() -> list[str]:
    bad: list[str] = []
    audit = exact_audit.audit_case_fidelity

    # The happy path.
    if audit(case_line(), "strict"):
        bad.append("a strict case declared strict is not accepted")

    # THE ORIGINAL DEFECT: an approximation walking into an acceptance table.
    coarse = case_line(key="c1", policy="L3coarse", physics_fidelity="coarse10",
                       statepoint_grid="coarse", acceptance_eligible=False)
    if not audit(coarse, "strict"):
        bad.append("a coarse case under a strict declaration is not caught -- this is the "
                   "defect the whole contract exists for")

    # THE OTHER DIRECTION.  Not a happy accident: a strict number filed in the
    # A2 column makes that column wrong with no line anywhere saying so.
    if not audit(case_line(), "A2"):
        bad.append("a strict case under an A2 declaration is not caught")

    # MIXED WAVE, audited PER CASE.  This is what the process receipt cannot do.
    mixed = "\n".join([
        case_line(key="screen1", case_key="k1", policy="L3coarse",
                  physics_fidelity="coarse10", statepoint_grid="coarse",
                  acceptance_eligible=False),
        case_line(key="elite1", case_key="k2"),
    ])
    declared = {"screen1": "L3coarse", "elite1": "strict"}
    if audit(mixed, declared):
        bad.append("a legitimately mixed wave is rejected when each case is declared "
                   "correctly; mixed waves are the point of WP10.3")
    if not audit(mixed, "strict"):
        bad.append("a mixed wave audited against ONE word is not caught; the screening "
                   "case in it would pass as strict")
    if not audit(mixed, {"elite1": "strict"}):
        bad.append("a case with no declaration is not reported; an unaccounted case is a "
                   "result whose provenance is a guess")

    # A grid that disagrees with the policy word, both ways.
    if not audit(case_line(statepoint_grid="coarse"), "strict"):
        bad.append("a case that coarsened its deck and reported `strict` is not caught")
    if not audit(case_line(policy="L3coarse", physics_fidelity="coarse10",
                           acceptance_eligible=False), "L3coarse"):
        bad.append("a case that claimed L3coarse without coarsening anything is not "
                   "caught; it paid full cost and was filed as screening")

    # A promotion that did not land at strict.
    promo = case_line(policy="A2", physics_fidelity="staged_a2",
                      acceptance_eligible=False, promoted_from="parentkey")
    if not audit(promo, "A2"):
        bad.append("a promotion that ran at A2 is not caught; the elite is unverified "
                   "while looking verified")

    # A pre-WP10.3 receipt is a REFUSAL, not a pass.
    old = '[RASBERY][EVALUATOR][CASE] {"key":"c","deck":"d","status":"ok"}'
    if not audit(old, "strict"):
        bad.append("a receipt with no fidelity fields passes; the field that would have "
                   "voided the case is the missing one")

    # No receipts at all is a void wave, not a clean one.
    if not audit("nothing here", "strict"):
        bad.append("a log with no per-case receipt passes")

    # A failed case that folded nothing is a FAILURE, not a fidelity violation:
    # the status field already reports it and the dispatcher already counts it.
    failed = case_line(status="failed", policy=None, physics_fidelity=None,
                       statepoint_grid=None, acceptance_eligible=None)
    if audit(failed, "strict"):
        bad.append("a failed case with no receipt is reported as a fidelity violation, "
                   "which would drown the real ones")

    # The promotion link, read back out of a log.
    links = exact_audit.promotion_links(
        case_line(case_key="child", promoted_from="parent"))
    if links != {"child": "parent"}:
        bad.append(f"promotion_links did not recover the link: {links}")
    return bad


# ===========================================================================
# RUN
# ===========================================================================
FAILED += check_grid_parity(GRID)
FAILED += check_apply_site(IO_CPP)
FAILED += check_rules(FIDELITY)
FAILED += check_wiring(DRIVER, SERVER, LIGHT)
FAILED += check_audit()

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS for the source checks.  A check that still passes against a
# copy broken in the exact way it exists to catch is a comment.
# ---------------------------------------------------------------------------
negative: list[str] = []


def control(name: str, checker, *args) -> None:
    if not checker(*args):
        negative.append(name)


control("check_grid_parity misses a coarse grid that is not the tool's",
        check_grid_parity, GRID.replace("0.5, 1.0, 2.0", "0.5, 1.0, 3.0"))
control("check_grid_parity misses a dropped time key",
        check_grid_parity, GRID.replace('"burnup_increment",', ""))
control("check_grid_parity misses a cumulative burnup written where a delta belongs",
        check_grid_parity,
        GRID.replace("detail::round10(cumulative - previous)", "cumulative"))
control("check_apply_site misses the grid applied after the deck digest",
        check_apply_site,
        IO_CPP.replace("rasbery::spgrid::applyGridSpec(config", "later::apply(config"))
control("check_rules misses a strict declaration that does not clear the multipliers",
        check_rules, FIDELITY.replace("out.staged_flux_mult = 1.0;", "/* kept */"))
control("check_rules misses a one-directional equality",
        check_rules, FIDELITY.replace("FINER than it declared", "fine"))
control("check_rules misses a floor a request can climb above",
        check_rules, FIDELITY.replace("coarserOf", "pickRequested"))
control("check_wiring misses a fidelity that never reaches the solve",
        check_wiring, DRIVER.replace("ctx.fidelity = _fidelity;", "/* nope */"),
        SERVER, LIGHT)
control("check_wiring misses a case key that ignores the case's fidelity",
        check_wiring,
        DRIVER.replace("caseKeyProvenance(input_output, warm_provenance, _fidelity)",
                       "caseKeyProvenance(input_output, warm_provenance)"),
        SERVER, LIGHT)
control("check_wiring misses a light JSONL with no fidelity",
        check_wiring, DRIVER, SERVER, LIGHT.replace('receipt["policy"]', 'receipt["p"]'))

if negative:
    FAILED.append("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are therefore "
                  "comments:\n    " + "\n    ".join(negative))

if FAILED:
    raise SystemExit("case fidelity contract: FAIL\n  - " + "\n  - ".join(FAILED))
print("case fidelity contract: PASS "
      f"({len(make_screening_deck.COARSE_BURNUPS)}-point coarse grid, "
      f"{len(exact_audit.CASE_REQUIRED_FIELDS)} required per-case fields)")
