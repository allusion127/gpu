#!/usr/bin/env python3
"""Contract: WP24 -- a NAMED fidelity preset, and the four ways it can lie.

WHY THIS FILE EXISTS.  Every failure this work package can introduce produces a
run that finishes, prints a finite k_eff and exits zero.

  * A preset that is INERT on half the deck families.  Scheduler's rod-crit
    clamp is a min() against a literal, so a preset that relaxed only
    `tolerance_search` would be a complete no-op on every RODCRIT deck while its
    receipt said screen100 -- a case solving FINER than it declared, which is
    the direction CaseFidelity.h calls a defect in its own right, and it would
    present only as a speedup that mysteriously does not reproduce.
  * A preset that is HALF applied.  The tolerances live in six places (a class
    constant, a deck field, a Scheduler literal, the CMFD sweep exit, an
    Anderson arming gate, a device Xe request), and a preset that reached five
    of them is a run converging to numbers no receipt names.
  * A preset the ENVIRONMENT overrides.  238 measured RASBERY_SEARCH_CARRY_SLOPE
    at +8.11 % outers, so a screen100 run inside a shell that exported it is
    eight percent slower under a receipt that says screen100.
  * A preset the AUDIT voids.  tools/exact_audit.py derives the fidelity it
    expects the child to print; a derivation that does not follow the binary's
    rule fails every screen100 run on a policy mismatch -- a verbatim re-commit
    of the WP4 defect that module exists to undo.

None of those is visible in any output the program produces, so they are pinned
here, in the source.

SIX PARTS.

  1. THE TABLE IS THE SINGLE SOURCE.  The `strict` row restates the tree's
     built-in constants and the `A2` row restates the measured production arm --
     both read out of the source they claim to restate, so the table cannot
     drift away from it.  This is requirement (2): strict and A2 stay
     byte-identical.
  2. THE screen100 ROW.  Every knob the directive fixes, plus the two the
     inventory found were traps (the rod-crit cap, the pinned oscillation
     floor), plus the ones it must NOT move.
  3. FEATURE-OFF IS THE OLD PATH.  Each preset-aware read keeps its pre-WP24
     expression as the no-preset branch, verbatim, and the receipt line is
     conditional so stdout is unchanged too.
  4. THE WIRING.  Six consumers, because a value that reaches five of them is a
     value one receipt disagrees about.
  5. THE RECEIPTS AND THE CASE KEY.  [RASBERY][FIDELITY], the [CASE] line, the
     light JSONL, the evaluator's per-case line, kArmEnv and armEnvValue.
  6. THE TOOLS' ENVELOPES, behaviourally: both Gate B tools take --envelope,
     both default to `production`, screen100 is 100 pcm / 1 % / 1 %, and a
     breach is a nonzero exit.

NEGATIVE CONTROLS.  Sections 1-5 are source checks and each is re-run against a
copy broken in the way it exists to catch; section 6 is behavioural and its
controls are the bad measurements themselves.  A check that cannot fail is a
comment, and this file says so out loud when it finds one.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import exact_audit  # noqa: E402
import gate_b_envelope  # noqa: E402

FAILED: list[str] = []


def fail(msg: str) -> None:
    FAILED.append(msg)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


PRESET_H = read("src/FidelityPreset.h")
DRIVER = read("src/Driver.h")
SCHEDULER = read("src/Scheduler.h")
CONTRACT = read("src/RunContract.h")
FIDELITY = read("src/CaseFidelity.h")
SERVER = read("src/EvaluatorServer.h")
MAIN = read("src/main.cpp")
LIGHT = read("include/chiffon/BatchLightResult.h")
BATCH = read("tools/run_single_gpu_batch.py")

PRESETS = dict(exact_audit.fidelity_presets())
NUM = exact_audit.preset_number
FLAG = exact_audit.preset_flag


def approx(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-15 * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
# 1. THE TABLE IS THE SINGLE SOURCE
# ---------------------------------------------------------------------------
#
# The point of reading BOTH sides out of the tree rather than hard-coding the
# numbers here: a test that carried its own copy of 1e-6 would pass a tree in
# which the solver moved and the table did not.

def cxx_constant(source: str, pattern: str) -> float | None:
    found = re.search(pattern, source)
    return float(found.group(1)) if found else None


def check_table_single_source(preset_h: str, driver: str, scheduler: str,
                              batch: str) -> list[str]:
    problems: list[str] = []
    # The rows come from the TEXT this check was handed, so a negative control
    # that breaks the header is actually judged against the broken header.
    try:
        presets = exact_audit.parse_fidelity_presets(preset_h)
    except SystemExit as exc:
        return [f"the preset table did not parse: {exc}"]

    # -- the tree's built-in production tolerances, read where the solver reads
    #    them --------------------------------------------------------------
    tree = {
        "flux_l2_tol": cxx_constant(
            driver, r"static constexpr double CMFD_FLUX_L2_TOLERANCE\s*=\s*([0-9.eE+-]+);"),
        "xe_tol": cxx_constant(
            driver, r"static constexpr double XE_EQUILIBRIUM_TOLERANCE\s*=\s*([0-9.eE+-]+);"),
        "rodcrit_search_cap": cxx_constant(
            scheduler, r"inline constexpr double kRodCritSearchTol\s*=\s*([0-9.eE+-]+);"),
    }
    # THE CMFD SWEEP EXIT HAS NO SECOND SPELLING LEFT TO READ.  Driver::Run used
    # to call `setEpsl2(1.0e-6)` in the constructor block and then overwrite it
    # twenty-four lines later with the per-case value, so this check could read
    # the tree's own literal.  The dead call is gone -- a number the table owns
    # with a second copy nothing forces to move is exactly the drift the
    # single-source rule exists to stop -- so what is pinned is the STRICT row's
    # value against the number the WP24 comment and the docs record, the same
    # way the cusping floor below is pinned.  The counterpart property (that
    # exactly ONE setEpsl2 call survives, and it is the per-case one) is checked
    # in check_wiring.
    tree.setdefault("cmfd_sweep_epsl2", 1.0e-6)
    for key, value in tree.items():
        if value is None:
            problems.append(f"cannot read the tree's built-in {key} -- the table has "
                            f"nothing left to be checked against")
    # XE_OSCILLATION_FLOOR is written as a MULTIPLE in the tree; its VALUE is
    # what the table must restate, and the whole point of the column is that it
    # must NOT float when xe_tol moves.
    osc = re.search(r"XE_OSCILLATION_FLOOR\s*=\s*([0-9.]+)\s*\*\s*XE_EQUILIBRIUM_TOLERANCE",
                    driver)
    if osc is None:
        problems.append("XE_OSCILLATION_FLOOR is no longer written as a multiple of "
                        "XE_EQUILIBRIUM_TOLERANCE; the table's absolute column can no "
                        "longer be checked against it")
    elif tree["xe_tol"] is not None:
        tree["xe_oscillation_floor"] = float(osc.group(1)) * tree["xe_tol"]
    # The cusping floor is a literal in Driver::SolveLoop's pre-WP24 form; after
    # WP24 it comes from the preset, so what is pinned is the STRICT ROW's value
    # against the number the WP24 comment and the docs record.
    tree.setdefault("rodcrit_search_floor_cusping", 5.0e-5)

    strict = presets.get("strict")
    if strict is None:
        return problems + ["src/FidelityPreset.h has no `strict` row: the table cannot "
                           "claim to restate the production tolerances"]
    for key, expected in tree.items():
        if expected is None:
            continue
        got = NUM(strict, key)
        if not approx(got, expected):
            problems.append(
                f"the `strict` preset row's {key} is {got!r} but the tree's own value is "
                f"{expected!r}. The row exists to RESTATE the production tolerances; a "
                f"row that does not is a second definition of `strict`.")
    for key in ("staged_flux_mult", "staged_xe_mult", "keff_tol_mult", "search_tol_mult"):
        if NUM(strict, key) != 1.0:
            problems.append(f"`strict` row {key} = {strict[key]}, must be exactly 1.0 -- "
                            f"strict is the identity and a multiplier of anything else "
                            f"makes the promotion lane a different physics")
    if strict["loose_settle"] != "false":
        problems.append("`strict` row sets loose_settle; strict must clear it")
    if strict["statepoint_grid"] != '""':
        problems.append("`strict` row names a statepoint grid; strict is the full deck")

    # -- the A2 row IS run_single_gpu_batch.DEFAULT_ENV's arm -----------------
    a2 = presets.get("A2")
    if a2 is None:
        problems.append("src/FidelityPreset.h has no `A2` row")
    else:
        env = {}
        block = re.search(r"DEFAULT_ENV\s*=\s*\{(.*?)\n\}", batch, re.S)
        if block is None:
            problems.append("cannot find DEFAULT_ENV in tools/run_single_gpu_batch.py")
        else:
            env = dict(re.findall(r'"([A-Z_0-9]+)":\s*"([^"]*)"', block.group(1)))
        for key, name in (("staged_flux_mult", "RASBERY_STAGED_FLUX_TOL"),
                          ("staged_xe_mult", "RASBERY_STAGED_XE_TOL")):
            if name in env and not approx(NUM(a2, key), float(env[name])):
                problems.append(
                    f"the `A2` preset row's {key} is {a2[key]} but DEFAULT_ENV exports "
                    f"{name}={env[name]}. The row's whole job is to NAME the measured "
                    f"production arm; a row that names a different one is a fourth "
                    f"unnamed arm.")
        if env.get("RASBERY_STAGED_LOOSE_SETTLE") and a2["loose_settle"] != "true":
            problems.append("`A2` row clears loose_settle but DEFAULT_ENV sets it")
        # And A2 must move NOTHING else: staging is a path change, not an
        # answer change, and that identity is what makes it byte-comparable.
        for key in ("keff_tol_mult", "search_tol_mult"):
            if NUM(a2, key) != 1.0:
                problems.append(f"`A2` row {key} = {a2[key]}; A2 must not move a "
                                f"PUBLISHED tolerance -- only screen100 does that")
        for key in ("flux_l2_tol", "xe_tol", "xe_oscillation_floor", "cmfd_sweep_epsl2",
                    "rodcrit_search_cap", "rodcrit_search_floor_cusping"):
            if not approx(NUM(a2, key), NUM(strict, key)):
                problems.append(f"`A2` row {key} differs from `strict`; A2 changed only "
                                f"the loose stage and this row must say so")
        if a2["staged_search_margin"] != "0.0":
            problems.append("`A2` row sets staged_search_margin; the shipped arm leaves "
                            "the built-in 4.0")
        for key in ("boron_bracket", "carry_slope", "warm_boron"):
            if FLAG(a2, key):
                problems.append(f"`A2` row turns {key} on; the measured arm has all "
                                f"three off")

    # -- ONE SPELLING.  A number in the table AND a literal at a consumption
    #    site is two definitions, and the second one is the one that runs. ----
    solve_loop = driver[driver.index("static void SolveLoop("):] if \
        "static void SolveLoop(" in driver else ""
    for literal, what in ((r"XE_EQUILIBRIUM_TOLERANCE\s*[;,)]", "the Xe tolerance"),
                          (r"XE_OSCILLATION_FLOOR", "the Xe oscillation floor"),
                          (r"CMFD_FLUX_L2_TOLERANCE", "the CMFD flux L2 tolerance")):
        body = re.sub(r"//[^\n]*", "", solve_loop)
        if re.search(literal, body):
            problems.append(
                f"Driver::SolveLoop still reads {what} from the class constant. The "
                f"preset then applies to every OTHER site and this one silently keeps "
                f"the production number, which is a run converging to a mixture no "
                f"receipt names.")
    if "std::min(tolerance_search, kRodCritSearchTol)" in scheduler:
        problems.append(
            "Scheduler::criticalSearchTolerance() still clamps against the "
            "kRodCritSearchTol LITERAL. A preset that relaxes the search tolerance is "
            "then a complete no-op on every RODCRIT deck (the iSMR / CY families) while "
            "its receipt claims otherwise.")
    return problems


# ---------------------------------------------------------------------------
# 2. THE screen100 ROW
# ---------------------------------------------------------------------------

def check_screen100(presets: dict) -> list[str]:
    problems: list[str] = []
    row = presets.get("screen100")
    if row is None:
        return ["src/FidelityPreset.h has no `screen100` row"]
    strict = presets["strict"]

    # The relaxations the directive authorises, and the ratios the reasoning
    # rests on rather than the bare values -- so a future retune that keeps the
    # argument still passes and one that breaks it does not.
    if NUM(row, "keff_tol_mult") <= 1.0:
        problems.append("screen100 does not relax the eigenvalue tolerance; it is then "
                        "an A2 arm with a new name")
    if NUM(row, "search_tol_mult") != NUM(row, "keff_tol_mult"):
        problems.append(
            "screen100's search and eigenvalue multipliers differ. The shipped tree has "
            "a 10:1 search-to-keff ratio (kCritSearchTol 1e-5 over kEigvTol 1e-6) and "
            "the preset preserves it exactly; moving one without the other either lets "
            "the secant read noise or converges the flux for digits nobody reads.")
    keff_tol_pcm = 1.0e-6 * NUM(row, "keff_tol_mult") * 1.0e5
    # THE SEARCH TOLERANCE THIS ROW CAN ACTUALLY SPEND, which is NOT the
    # multiplier applied to the built-in.  `search_tol_mult` scales
    # `schedule.tolerance_search`, and IO::ReadInput fills that from the DECK
    # (`search_tol`, or `search_pcm_tolerance` x 1e-5): a deck at
    # `search_pcm_tolerance: 2` is at 2e-5 before the multiplier and one at 5 is
    # at 5e-5, so x10 is 20 and 50 pcm -- half the row's entire budget, from a
    # knob every check here used to evaluate at 10 pcm because it computed from
    # the BUILT-IN and never looked at a deck.  `search_tol_cap` is the row's
    # absolute ceiling (0 = none), so the worst case a deck can reach is the cap
    # when there is one and UNBOUNDED when there is not.
    search_tol_cap = NUM(row, "search_tol_cap")
    if NUM(row, "search_tol_mult") > 1.0 and not (search_tol_cap > 0.0):
        problems.append(
            "screen100 relaxes the search tolerance by a bare MULTIPLIER with no "
            "`search_tol_cap`. The multiplier scales a DECK-STATED number, so a deck at "
            "`search_pcm_tolerance: 5` spends 50 pcm of a 100 pcm budget from this one "
            "knob while the row's own ppm arithmetic assumes 10 -- and the tree's "
            "dominant-risk finding (docs/A2_OUTER_REDUCTION_DESIGN_20260902_KO.md) is "
            "that at 2 pcm the published k_eff is already a function of the search PATH "
            "rather than the root. The row carries an absolute RODCRIT cap for exactly "
            "this reason; the boron lane needs one too.")
    search_tol_pcm = (search_tol_cap * 1.0e5 if search_tol_cap > 0.0
                      else float("inf"))
    envelope = gate_b_envelope.ENVELOPES["screen100"]
    if search_tol_pcm >= envelope.keff_pcm:
        problems.append(
            f"screen100's WORST-CASE critical-search tolerance is {search_tol_pcm:.0f} "
            f"pcm against a {envelope.keff_pcm:.0f} pcm envelope. A search tolerance at "
            f"or above the acceptance envelope is a preset that cannot pass its own "
            f"gate. (This is the absolute CAP, not the multiplier applied to the "
            f"built-in: an uncapped multiplier is unbounded against a deck-stated "
            f"tolerance and reads as inf here.)")
    if keff_tol_pcm > 0.2 * envelope.keff_pcm:
        problems.append(
            f"screen100's per-outer eigenvalue tolerance is {keff_tol_pcm:.1f} pcm, more "
            f"than a fifth of the {envelope.keff_pcm:.0f} pcm budget. The error is "
            f"~rho/(1-rho) x the tolerance and it accumulates along the depletion chain; "
            f"one decade (1e-5) is the measured-affordable step.")

    # The Xe tolerance moves ONE decade, not two: Driver.h records four
    # statepoints stopping at ~1e-5 instead of 1e-6 costing up to 8.9 pcm.
    xe_decades = NUM(row, "xe_tol") / NUM(strict, "xe_tol")
    # 1e-5/1e-6 is 10.000000000000002 in binary floating point, so the bound is
    # written with the slack that makes it a statement about decades rather than
    # about the representation of a ratio.
    if not (1.0 < xe_decades <= 10.0 * (1.0 + 1e-9)):
        problems.append(
            f"screen100's Xe tolerance is {xe_decades:g}x the production one. The tree "
            f"has a MEASURED datum for exactly one decade (up to 8.9 pcm, Driver.h); two "
            f"decades is ~10x that and is not in the 100 pcm budget.")

    # THE COUPLING.  The floor must not have floated with the tolerance.
    if not approx(NUM(row, "xe_oscillation_floor"), NUM(strict, "xe_oscillation_floor")):
        problems.append(
            "screen100's Xe oscillation floor differs from the production one. Letting it "
            "float at 100 x the relaxed tolerance moves it to 1e-3 -- exactly where the "
            "documented pathology sits (APR1400 cy01, ~1e-3 for 80+ steps) -- and the "
            "damper stops engaging on the case it was tuned for.")

    # THE TRAP, AND THE DELIBERATE NO-OP.  criticalSearchTolerance() min()s the
    # scaled tolerance against `rodcrit_search_cap`, so a row that moved only
    # the search tolerance is inert on every RODCRIT deck.  The COLUMNS
    # therefore have to exist -- that half is checked in
    # check_table_single_source, and it is what makes the relaxation
    # EXPRESSIBLE.  screen100 must not USE them, and this is the check that
    # stops that being prose.
    #
    # WHY.  Every measurement quoted in the screen100 row comes from BORON decks
    # (KNGR's 4,377 outers, the APR1400 slope, the -1.76 % bracket result), and
    # over a rod-crit deck the screen100 Gate B is blind in every scored scalar
    # column at once: `delta_pcm` is the search RESIDUAL (k_eff is at target by
    # construction) and is bounded by this very cap; `delta_ppm` is the deck's
    # own fixed boron on both sides; `delta_ao` is ADVISORY under this envelope,
    # and axial offset is precisely where a rod misposition shows; Fq/Fr are in
    # neither envelope's METRICS.  Loosest exactly where the gate is blindest,
    # with no measurement -- so the row stays at the production values, and the
    # receipt's resolved_entry0.search_tol prints the 1e-5 it really used so the
    # no-op is visible rather than inferred.
    for column in ("rodcrit_search_cap", "rodcrit_search_floor_cusping"):
        if not approx(NUM(row, column), NUM(strict, column)):
            problems.append(
                f"screen100's {column} is {row[column]} against production's "
                f"{strict[column]}. The rod-crit families (iSMR / CY) have NO measurement "
                f"behind any relaxation, and screen100's own Gate B cannot catch a "
                f"rod-position error on them: delta_pcm is the search residual, "
                f"delta_ppm is the deck's fixed boron on both sides, delta_ao is advisory "
                f"under this envelope, and Fq/Fr are in neither envelope. Measure the "
                f"rod-crit families against their own Gate B first, and give that Gate B "
                f"a column that can fail on a rod deck.")

    # THE CMFD SWEEP EXIT.  The bound is `<=`, and the reason is worth stating,
    # because two drafts of this file had the mechanism backwards and pinned the
    # STRICT inequality on the strength of it.
    #
    # BICGCMFD::drive() breaks on `errl2 < _epsl2` and otherwise exhausts
    # _ncmfd = 5 sweeps.  Read the outer's L2 half (`residual < flux_tol_now`)
    # against that: at epsl2 == flux_tol the outer test is exactly "the sweep
    # loop converged" (break -> pass, cap-exhaust -> fail), which is the
    # STRICTEST reading available; at epsl2 < flux_tol the break passes
    # trivially AND a cap-exhausted sweep landing anywhere in [epsl2, flux_tol)
    # passes although it did not converge.  So a TIGHTER epsl2 is strictly MORE
    # permissive here, and it also costs inner sweeps -- the earlier claim that
    # it stopped the outer verdict being "true by construction" had the
    # arithmetic backwards, and the test that encoded it forbade the cheaper AND
    # stricter direction.
    #
    # What survives the correction is the one direction that is a defect: a
    # sweep allowed to stop LOOSER than the outer verdict requires makes the
    # sweep cap, not the preset, decide the published flux.  screen100's 1e-5
    # against 1e-4 is legal here and so is 1e-4; the row says why it starts at
    # 1e-5 and names it as the first knob to sweep.
    flux_tol = max(NUM(row, "flux_l2_tol"), 1.0e-6 * NUM(row, "keff_tol_mult"))
    if not (NUM(row, "cmfd_sweep_epsl2") <= flux_tol * (1.0 + 1e-9)):
        problems.append(
            f"screen100's CMFD sweep exit ({row['cmfd_sweep_epsl2']}) is LOOSER than its "
            f"outer flux tolerance ({flux_tol:g}). The sweep would then stop before the "
            f"outer verdict can be satisfied, so the sweep cap and not the preset decides "
            f"the published flux.")

    # WHAT screen100 MUST NOT TOUCH.
    if FLAG(row, "carry_slope"):
        problems.append("screen100 turns RASBERY_SEARCH_CARRY_SLOPE on; 238 measured it "
                        "at +8.11 % outers and its own D1 rule discards it")
    if FLAG(row, "warm_boron"):
        problems.append("screen100 turns RASBERY_SEARCH_WARM_BORON on; its effect depends "
                        "on an unrelated CLI flag, so the preset would mean two different "
                        "physics under one name")
    if row["max_trials"] != "0":
        problems.append("screen100 caps the search trials. A capped SolveLoop exits "
                        "SEARCH_EXHAUSTED and publishes a boron that never converged -- "
                        "a statepoint a 100 pcm envelope cannot certify.")
    # Requirement (4): the statepoint-grid screening knob exists, and is OFF.
    if row["statepoint_grid"] != '""':
        problems.append(
            "screen100 names a statepoint grid. The knob is a column so it is "
            "expressible, but coarse() outranks staged() (CaseFidelity.h) and the preset "
            "would report L3coarse -- and the user authorised TOLERANCE relaxation only.")
    if "const char* statepoint_grid;" not in PRESET_H:
        problems.append("the preset table has no statepoint_grid column; requirement (4) "
                        "is that the screening knob EXISTS and is off, not that it is "
                        "absent")

    # And the staged multipliers must keep the LOOSE stage above the polish
    # stage but not absurdly so: they scale numbers screen100 has already moved.
    for mult, tol_key, name in ((NUM(row, "staged_xe_mult"), "xe_tol", "Xe"),
                                (NUM(row, "staged_flux_mult"), "flux_l2_tol", "flux")):
        if mult < 1.0:
            problems.append(f"screen100's staged {name} multiplier is below 1.0, which "
                            f"would TIGHTEN the loose stage")
        loose = mult * NUM(row, tol_key)
        if loose > 1.0e-2:
            problems.append(
                f"screen100's loose {name} tolerance is {loose:g}. The multipliers scale "
                f"tolerances the preset ALREADY relaxed, so carrying the A2 arm's numbers "
                f"forward puts the loose stage where a cascade exits after one step and "
                f"every polish transition re-runs it -- thrash, counted as "
                f"staged_relapses.")
    if NUM(row, "staged_search_margin") < 1.0:
        problems.append(
            "screen100's staged search margin is below 1.0. The margin is what keeps the "
            "loose sample BELOW the tolerance the secant reads; at or under 1 the sample "
            "is the noise the cap exists to exclude.")

    # ---------------------------------------------------------------- the ppm
    # THE ROW AND THE BORON COLUMN HAVE TO AGREE, and until this check existed
    # they did not.  A boron-search statepoint has no |dkeff| to score -- k_eff
    # is driven to target by construction -- so its whole MASTER-relative error
    # appears as a boron difference, and the row's knobs SPEND that column: the
    # search tolerance is a direct bound on CBC at the measured slope, and the
    # Xe tolerance's measured 8.9 pcm is another.  check_screen100 used to test
    # the search tolerance against `envelope.keff_pcm` (10 < 100, passes) and
    # never against `envelope.ppm`, so the two halves of this work package could
    # disagree about the boron budget without anything noticing -- and they did:
    # a first draft granted keff ~98 pcm of NEW error and boron ~3.2 ppm (~17
    # pcm) under one name, less than the row's own knobs spend.
    slope = gate_b_envelope.BORON_SLOPE_PCM_PER_PPM
    ppm_headroom = gate_b_envelope.headroom(envelope, "ppm")
    if ppm_headroom is None:
        problems.append(
            "the screen100 envelope grants no readable ppm headroom over its baseline, "
            "so nothing can check the row's search tolerance against the boron column")
    else:
        # The CAP, not the multiplier: an uncapped multiplier is unbounded
        # against a deck-stated tolerance and the check above already refuses
        # it, so this arithmetic is only ever run on a bounded row.
        search_tol_ppm = search_tol_pcm / slope
        # The measured Xe datum, Driver.h: four statepoints stopping at ~1e-5
        # instead of 1e-6 moved k_eff by up to 8.9 pcm.  It is spent out of the
        # same column and is not optional to count.
        xe_ppm = 8.9 / slope if NUM(row, "xe_tol") > NUM(strict, "xe_tol") else 0.0
        spent = search_tol_ppm + xe_ppm
        if spent > 0.5 * ppm_headroom:
            problems.append(
                f"screen100's own knobs spend {spent:.2f} ppm of a {ppm_headroom:.2f} ppm "
                f"boron headroom before any depletion accumulation (search tolerance "
                f"{search_tol_pcm:.1f} pcm = {search_tol_ppm:.2f} ppm at {slope} pcm/ppm; "
                f"the relaxed Xe tolerance's measured 8.9 pcm = {xe_ppm:.2f} ppm). Either "
                f"the ppm column is stated as a baseline-plus-budget increment, or "
                f"search_tol_mult comes down -- but the row and the envelope cannot "
                f"disagree about which.")
    return problems


# ---------------------------------------------------------------------------
# 2b. THE TABLE-WIDE INVARIANT: A ROW THAT MOVES A POLISH TOLERANCE IS NOT
#     ACCEPTANCE-ELIGIBLE
# ---------------------------------------------------------------------------
#
# detectedPhysicsFidelity() keys on `processStagedFluxMult() > 1.0`, i.e. on
# STAGING -- not on the polish tolerances.  screen100 satisfies the invariant
# only INCIDENTALLY, because its multipliers happen to be 5 and 100.  A future
# row with multipliers of exactly 1.0 and relaxed polish tolerances would report
# FullExact and acceptance_eligible=true while converging at screening
# tolerances: an approximation walking into an acceptance table, which is the
# defect class this whole header argues against.  So the invariant is checked
# here, over EVERY row, rather than left to be true by luck.

POLISH_COLUMNS = ("keff_tol_mult", "search_tol_mult", "flux_l2_tol", "xe_tol",
                  "xe_oscillation_floor", "cmfd_sweep_epsl2", "rodcrit_search_cap",
                  "rodcrit_search_floor_cusping")

# ---------------------------------------------------------------------------
# 2c. THE CASE KEY FOLDS THE PRESET'S *NAME*, AND THE NUMBERS LIVE IN A HEADER
# ---------------------------------------------------------------------------
#
# armEnvValue() folds the string "screen100"; the seventeen values behind it are
# compiled in.  `code_sha` is an OPERATOR-SUPPLIED environment token
# (src/CaseKey.h codeShaToken), not a build fingerprint, so editing
# kFidelityPresets without renaming the row makes two binaries with different
# tolerances compute the SAME case key -- a wrong HIT, which is the one
# direction Driver.h's kArmEnv comment says the list must never allow.
#
# Folding a digest of the row into the key would have to be reproduced byte for
# byte by tools/case_key.py, which reads the child ENVIRONMENT and not this
# header, and getting two languages to agree on the spelling of a double is how
# WP10.1's first live gate failed.  So this pins the DETECTION half instead: an
# edit to any shipped row fails here, loudly, with the two ways out named.  It
# costs one line to update when a row is deliberately retuned, and it makes the
# silent-collision path impossible to take by accident.
ROW_DIGESTS = {
    "strict":    "1bb83a9f075f4ca4",
    "A2":        "bc939817f0910505",
    "screen100": "d8c8d41e50c0272e",
}


def row_digest(row: dict) -> str:
    import hashlib  # noqa: PLC0415
    payload = "\n".join(f"{k}\t{row[k]}" for k in sorted(row) if k != "_constants")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def check_row_digests(presets: dict) -> list[str]:
    problems: list[str] = []
    for name in sorted(set(ROW_DIGESTS) | set(presets)):
        want = ROW_DIGESTS.get(name)
        row = presets.get(name)
        if row is None:
            problems.append(f"preset row {name!r} was deleted. Every case key ever "
                            f"computed under that name refers to tolerances this build "
                            f"no longer has.")
            continue
        got = row_digest(row)
        if want is None:
            problems.append(
                f"preset row {name!r} is new and unpinned. Add "
                f'"{name}": "{got}" to ROW_DIGESTS -- an unpinned row can be retuned '
                f"later with nothing noticing.")
        elif got != want:
            problems.append(
                f"preset row {name!r} changed: digest {want} -> {got}. The case key "
                f"folds only the NAME, so this build and the last one now answer "
                f"different tolerances under one key -- a wrong cache HIT. Either RENAME "
                f"the row (a retune is a new arm and deserves a new name), or move "
                f"RASBERY_CODE_SHA so the two builds key apart -- then update "
                f'ROW_DIGESTS["{name}"] to "{got}".')
    return problems


def check_polish_invariant(presets: dict) -> list[str]:
    problems: list[str] = []
    strict = presets.get("strict")
    if strict is None:
        return ["src/FidelityPreset.h has no `strict` row to measure the others against"]
    for name, row in presets.items():
        if name == "strict":
            continue
        moved = [c for c in POLISH_COLUMNS if not approx(NUM(row, c), NUM(strict, c))]
        if not moved:
            continue
        staged = (NUM(row, "staged_flux_mult") > 1.0 or NUM(row, "staged_xe_mult") > 1.0)
        coarse = row.get("statepoint_grid", '""') not in ('""', '"full"')
        if not (staged or coarse):
            problems.append(
                f"preset row {name!r} moves the PUBLISHED tolerance(s) "
                f"{', '.join(moved)} but leaves both staged multipliers at 1.0 and names "
                f"no statepoint grid. detectedPhysicsFidelity() keys on STAGING, so that "
                f"row reports FullExact / acceptance_eligible=true while converging at "
                f"screening tolerances -- a screening number walking into an acceptance "
                f"table under the word `strict`.")
    return problems


# ---------------------------------------------------------------------------
# 3. FEATURE-OFF IS THE OLD PATH
# ---------------------------------------------------------------------------

def check_feature_off(contract: str, fidelity: str, scheduler: str, driver: str,
                      preset_h: str) -> list[str]:
    problems: list[str] = []
    # The no-preset branch has to be the PRE-WP24 EXPRESSION, present verbatim,
    # not a re-derivation that happens to agree today.
    for source, needle, where in (
        (contract, 'detail::stagedMultiplier("RASBERY_STAGED_FLUX_TOL")',
         "RunContract processStagedFluxMult"),
        (contract, 'detail::stagedMultiplier("RASBERY_STAGED_XE_TOL")',
         "RunContract processStagedXeMult"),
        (fidelity, 'parseStagedLooseSettle(std::getenv("RASBERY_STAGED_LOOSE_SETTLE"))',
         "processCaseFidelity"),
        (scheduler, 'searchFlagEnabled(std::getenv("RASBERY_SEARCH_CARRY_SLOPE"))',
         "processSearchPolicy"),
    ):
        if needle not in source:
            problems.append(
                f"{where} no longer carries its pre-WP24 read ({needle}). Feature-off has "
                f"to be the OLD PATH textually, or the identity claim is a claim about a "
                f"diff somebody remembered to run.")
    # A default-constructed SolveTolerances IS the built-in set, so the
    # no-preset path multiplies by exactly 1.0 and compares against the same
    # constants.
    for field, const in (("keff_tol_mult", "1.0"), ("search_tol_mult", "1.0"),
                         ("search_tol_cap", "kProdSearchTolCap"),
                         ("flux_l2_tol", "kProdFluxL2Tol"), ("xe_tol", "kProdXeTol"),
                         ("xe_oscillation_floor", "kProdXeOscillationFloor"),
                         ("cmfd_sweep_epsl2", "kProdCmfdSweepEpsl2"),
                         ("rodcrit_search_cap", "kProdRodCritSearchCap"),
                         ("rodcrit_search_floor_cusping", "kProdRodCritFloorCusp")):
        if not re.search(rf"double {field}\s*=\s*{re.escape(const)};", preset_h):
            problems.append(
                f"SolveTolerances::{field} does not default to {const}. A default that is "
                f"not the built-in makes every preset-free run a new arm.")
    if "presetTolerances(const FidelityPresetSpec* preset)" not in preset_h or \
            "if (preset == nullptr) return t;" not in preset_h:
        problems.append("presetTolerances() does not answer the built-ins for a null "
                        "preset; there is then no feature-off path at all")
    # The receipt is CONDITIONAL, so a preset-free run's stdout is unchanged.
    if not re.search(r"if \(const FidelityPresetSpec\* preset_spec = _fidelity\.presetSpec\(\)\)",
                     driver):
        problems.append(
            "the [RASBERY][FIDELITY] receipt is not guarded on a preset being named. An "
            "unconditional line changes the stdout of every run in the tree, and the "
            "feature-off identity stops being a property of the source.")
    # Scheduler's default keeps the constant, so the clamp is unchanged.
    if not re.search(r"double\s+rodcrit_search_cap\s*=\s*kRodCritSearchTol;", scheduler):
        problems.append("Schedule::rodcrit_search_cap does not default to "
                        "kRodCritSearchTol; the rod-crit clamp then moves for every run")
    if "criticalSearchTolerance(double scale = 1.0," not in scheduler or \
            "double cap   = 0.0)" not in scheduler:
        problems.append("criticalSearchTolerance() lost a defaulted parameter (scale 1.0, "
                        "cap 0.0); every existing caller then has to be edited to stay "
                        "identical, and cap = 0.0 is what makes the ceiling branch "
                        "untaken on every preset-free run")
    return problems


# ---------------------------------------------------------------------------
# 4. THE WIRING -- six consumers
# ---------------------------------------------------------------------------

def check_wiring(driver: str, contract: str, fidelity: str, scheduler: str) -> list[str]:
    problems: list[str] = []
    checks = [
        (driver, "ctx.tolerances = _fidelity.tolerances();",
         "the case's tolerances never reach the solve"),
        (driver, "cmfd_solver.setEpsl2(ctx.tolerances.cmfd_sweep_epsl2);",
         "the CMFD SWEEP exit is not the case's -- every outer then burns all five "
         "sweeps chasing a residual the outer verdict stopped needing"),
        (driver, "schedule.rodcrit_search_cap = tol.rodcrit_search_cap;",
         "the RODCRIT clamp is not the case's, so the preset is inert on rod-crit decks"),
        (driver, "schedule.criticalSearchTolerance(tol.search_tol_mult, tol.search_tol_cap)",
         "the search tolerance is not scaled AND capped by the preset. The multiplier "
         "alone scales a DECK-STATED number, so without the absolute cap a deck at "
         "search_pcm_tolerance 5 spends 50 pcm of a 100 pcm budget"),
        (driver, "schedule.tolerance_keff * tol.keff_tol_mult",
         "the eigenvalue tolerance is not scaled by the preset"),
        (driver, "std::max(keff_tol, tol.flux_l2_tol)",
         "the outer flux tolerance is not the case's"),
        (driver, "req.eq_tol   = ctx.tolerances.xe_tol;",
         "the DEVICE Xe request still asks for the production equilibrium, so the host "
         "and device arms of one solve chase two different fixed points"),
        (driver, "picard < ctx.tolerances.xe_tol",
         "the Anderson arming gate still uses the production Xe tolerance"),
        (contract, "processStagedFluxMult()",
         "detectedPhysicsFidelity() does not consult the preset, so a screen100 run "
         "reports strict"),
        (fidelity, "f.preset = fidelityPresetEnvName();",
         "processCaseFidelity() does not read RASBERY_FIDELITY"),
        (scheduler, "lookupFidelityPreset(fidelityPresetEnvName())",
         "processSearchPolicy() does not assert the preset's search knobs, so a campaign "
         "environment decides them"),
        # WP24 (review).  THE SEARCH HALF OF THE ROW, PER CASE.
        #
        # processSearchPolicy() resolves the preset from the ENVIRONMENT into a
        # function-local static, so the check above is only half the property.
        # SolverContext::search_policy was initialised from it and never
        # reassigned, which meant a case asking for
        # `"fidelity_preset":"screen100"` over the evaluator socket got the
        # row's eight tolerances and the environment's five search knobs:
        # boron_bracket Off where the row says On, staged_search_margin back at
        # the built-in 4.0 (clipping loose_keff_tol to 2.5e-5 where the row
        # intends 5e-5), and carry_slope ON inside any campaign shell exporting
        # it -- the +8.11 % lever the row exists to pin off.
        (fidelity, "presetSearchPolicy(spec) : environmentSearchPolicy();",
         "CaseFidelity::searchPolicy() does not resolve the five search knobs, or "
         "resolves the NO-ROW case through processSearchPolicy() -- which reads "
         "RASBERY_FIDELITY, i.e. the PROCESS row. See check_cleared_preset below: that "
         "is how the acceptance lane inside a screen100 campaign ran the row's "
         "boron_bracket"),
        (driver, "ctx.search_policy = _fidelity.searchPolicy();",
         "the case's SEARCH POLICY never reaches the solve: the row's tolerances apply "
         "and its bracket / margin / carry-slope knobs come from the environment"),
        (scheduler, "inline SearchPolicy presetSearchPolicy(const FidelityPresetSpec* spec)",
         "the row's search knobs are not resolved through one function, so the per-case "
         "and per-process paths can form two opinions about one row"),
    ]
    for source, needle, why in checks:
        if needle not in source:
            problems.append(f"{why} (missing: {needle})")
    # AND THERE IS EXACTLY ONE setEpsl2 CALL LEFT.  The constructor-block
    # `setEpsl2(1.0e-6)` was dead (overwritten twenty-four lines later) and was a
    # second spelling of a number src/FidelityPreset.h now owns; leaving a copy
    # nothing forces to move is the drift kProdCmfdSweepEpsl2 exists to prevent.
    if driver.count("cmfd_solver.setEpsl2(") != 1:
        problems.append(
            f"src/Driver.h has {driver.count('cmfd_solver.setEpsl2(')} setEpsl2 calls; "
            f"exactly one -- the per-case `ctx.tolerances.cmfd_sweep_epsl2` -- is the "
            f"contract, because a second spelling of the sweep exit is a number the "
            f"table can no longer claim to own.")
    # The deck must NOT be rewritten: that would move the case key's DECK half.
    if re.search(r"ReadInput\([^)]*keff_tol_mult", driver):
        problems.append(
            "the preset is applied at the deck load. IO::ReadInput folds the deck's "
            "canonical digest immediately after the parse, so a rewritten tolerance moves "
            "the case key's DECK half and two fidelities of one core look like two cores.")
    return problems


# ---------------------------------------------------------------------------
# 5. THE RECEIPTS AND THE CASE KEY
# ---------------------------------------------------------------------------

def check_receipts(driver: str, server: str, light: str, main: str) -> list[str]:
    problems: list[str] = []
    if "[RASBERY][FIDELITY]" not in driver:
        problems.append("no [RASBERY][FIDELITY] receipt: the preset's NUMBERS are then "
                        "only in the source and a log cannot answer what the run solved at")
    for knob in ("staged_flux_mult", "staged_xe_mult", "staged_loose_settle",
                 "staged_search_margin", "keff_tol_mult", "search_tol_mult",
                 "search_tol_cap",
                 "flux_l2_tol", "xe_tol", "xe_oscillation_floor", "cmfd_sweep_epsl2",
                 "rodcrit_search_cap", "rodcrit_search_floor_cusping", "boron_bracket",
                 "carry_slope", "warm_boron", "max_trials", "statepoint_grid"):
        if f'\\"{knob}\\"' not in driver:
            problems.append(
                f"the [RASBERY][FIDELITY] receipt omits {knob}. A knob missing from the "
                f"receipt is a knob a reader has to go and guess the environment for, "
                f"which is the unnamed-arm defect in miniature.")
    # WP24 (review).  THE FIVE SEARCH KNOBS COME OFF THE EFFECTIVE POLICY.
    #
    # They used to be spelled straight out of `preset_spec`, so on the whole
    # per-case lane -- where the search policy came from the environment -- five
    # of seventeen fields asserted knobs the run did not use.  A receipt that
    # names the row instead of the run is the defect this WP exists to end.
    for spelling, why in (
        ("preset_spec->boron_bracket", "boron_bracket"),
        ("preset_spec->carry_slope", "carry_slope"),
        ("preset_spec->warm_boron", "warm_boron"),
        ("preset_spec->max_trials", "max_trials"),
        ("preset_spec->staged_search_margin", "staged_search_margin"),
    ):
        if spelling in driver:
            problems.append(
                f"the [RASBERY][FIDELITY] receipt prints {why} from the TABLE ROW "
                f"({spelling}). It has to come from ctx.search_policy, which is what "
                f"SolveLoop reads -- otherwise the receipt asserts knobs the run did "
                f"not use.")
    for needle, why in (
        ("sp.boron_bracket ?", "boron_bracket"),
        ("sp.carry_slope ?", "carry_slope"),
        ("sp.warm_boron ?", "warm_boron"),
        ("sp.max_trials,", "max_trials"),
        ("sp.stagedMargin(kStagedSearchMarginBuiltIn)", "the effective staged margin"),
    ):
        if needle not in driver:
            problems.append(f"the [RASBERY][FIDELITY] receipt does not print {why} from "
                            f"the effective search policy (missing: {needle})")
    # AND THE RESOLVED ABSOLUTES, not only the multipliers.  keff_tol_mult and
    # search_tol_mult scale numbers the DECK states, so a reader holding the
    # receipt alone could not say what the case converged to.
    for knob in ("resolved_entry0", "loose_keff_tol", "loose_flux_tol", "loose_xe_tol"):
        if f'\\"{knob}\\"' not in driver:
            problems.append(
                f"the [RASBERY][FIDELITY] receipt omits {knob}. The two multipliers scale "
                f"DECK-STATED tolerances -- a deck at search_pcm_tolerance 5 is already "
                f"at 5e-5, which x10 is 50 pcm -- so the receipt has to carry the "
                f"resolved absolutes or it does not answer the question it exists for.")
    if '[RASBERY][FIDELITY] {{\\"schema_version\\":2' not in driver:
        problems.append("the [RASBERY][FIDELITY] receipt did not bump its schema_version "
                        "when it gained the resolved-tolerance block and changed where "
                        "five of its knobs are read from; a new field and a new meaning "
                        "on an unchanged version is a reader that cannot tell an old log "
                        "from a new one")
    if '"schema_version":7' not in driver.replace('\\"', '"'):
        problems.append("the [RASBERY][CASE] line did not bump its schema_version; a new "
                        "field on an unchanged version is a reader that cannot tell an "
                        "old log from a new one")
    for source, needle, where in (
        (driver, "fidelity_preset", "the [RASBERY][CASE] line"),
        (server, "fidelity_preset", "the evaluator's per-case line"),
        (light, "fidelity_preset", "the light JSONL receipt"),
        (main, "fidelity_preset", "the process [PHYSICS_MODE] receipt"),
    ):
        if needle not in source:
            problems.append(f"{where} does not carry the preset name. `policy` says A2 "
                            f"for every staged arm this binary can run, so on its own it "
                            f"does not identify the run.")
    if '"fidelity_preset"' not in server:
        problems.append("the evaluator does not PARSE \"fidelity_preset\"; the JSON path "
                        "the GA controller uses cannot then ask for the preset by name")
    if "applyWaveFidelityDefault" in server and "wave.has_preset" not in server:
        problems.append("a wave-level preset is not a default for the cases in it, so a "
                        "generation has to spell it on sixty-four lines")
    if "request.request_fidelity.has_preset" not in server:
        problems.append(
            "`op\":\"promote` does not clear the preset. An evaluator standing in a "
            "screen100 campaign would then re-run the elite at screening tolerances "
            "while reporting policy:strict -- the exact WP10.7 defect, one axis over.")
    if "fidelityPresetEnvIsUnknown" not in main:
        problems.append("main.cpp does not refuse an unknown RASBERY_FIDELITY. A typo'd "
                        "preset is a preset that silently did not happen, and the run "
                        "publishes production numbers under a screening name.")
    return problems


# ---------------------------------------------------------------------------
# 4b. CLEARING THE PRESET CLEARS BOTH HALVES OF THE ROW
# ---------------------------------------------------------------------------
#
# THE DEFECT.  `CaseFidelity::searchPolicy()` answered `processSearchPolicy()`
# for a case with no row, and processSearchPolicy() itself resolves the row from
# RASBERY_FIDELITY.  So "this case names no preset" quietly meant "this case
# gets the PROCESS's preset's search knobs".  In the deployment WP24's runbook
# prescribes -- one process started with `--set RASBERY_FIDELITY=screen100` --
# the two ways a case clears the preset are `op:"promote"` (EvaluatorServer.h)
# and any `"fidelity":"strict"` request (CaseFidelity.h), i.e. THE ACCEPTANCE
# LANE.  Those cases got the built-in TOLERANCES and screen100's SEARCH POLICY:
# boron_bracket ON -- the lever 238 measured at Gate A 2.27 pcm and REJECTED on
# the 1.905 pcm production envelope -- and staged_margin 2.0.
#
# AND THE KEY COLLIDED.  armEnvValue() folds "" for RASBERY_FIDELITY (the case's
# empty preset differs from the process's) and "" for each of the five search
# names (the ROW, not the shell, was supplying them, so the raw text is empty
# and the effective values differ from base).  That payload is byte-identical to
# a genuine strict run in a preset-free process whose bracket is OFF: one case
# key, two solves, on the one lane whose entire job is to be
# acceptance-eligible.
#
# THE FIX IS A SPLIT, not a special case: environmentSearchPolicy() is the RAW
# five-knob read with no preset in it, processSearchPolicy() is that plus the
# row, and a CASE with no row resolves the former.  In a preset-free process the
# two answer the same bits, so nothing existing moves.

def check_cleared_preset(scheduler: str, fidelity: str) -> list[str]:
    problems: list[str] = []
    if "inline const SearchPolicy& environmentSearchPolicy()" not in scheduler:
        problems.append(
            "Scheduler.h has no environmentSearchPolicy(). There is then only one "
            "spelling of the five search knobs and it resolves RASBERY_FIDELITY, so a "
            "case that CLEARED its preset cannot express \"the environment, with no "
            "row\" at all.")
        return problems
    # The raw read must be the one WITHOUT the row, and the process read the one
    # WITH it.  Slicing on the two definitions is what makes this a property of
    # the source rather than of the order somebody wrote them in.
    start = scheduler.index("inline const SearchPolicy& environmentSearchPolicy()")
    end = scheduler.index("inline const SearchPolicy& processSearchPolicy()")
    if not (start < end):
        problems.append("environmentSearchPolicy() is defined after "
                        "processSearchPolicy(); the latter calls it")
        return problems
    env_body = scheduler[start:end]
    proc_body = scheduler[end:]
    proc_body = proc_body[:proc_body.index("\n}\n")]
    if "lookupFidelityPreset" in env_body:
        problems.append(
            "environmentSearchPolicy() resolves a preset row. It exists precisely to be "
            "the read with NO row in it -- the answer a case that cleared its preset is "
            "asking for -- and a row here makes the split a rename.")
    if "lookupFidelityPreset(fidelityPresetEnvName())" not in proc_body:
        problems.append(
            "processSearchPolicy() no longer resolves RASBERY_FIDELITY, so a campaign "
            "environment decides the five search knobs for a preset run")
    if "environmentSearchPolicy()" not in proc_body:
        problems.append(
            "processSearchPolicy() does not fall through to environmentSearchPolicy(); "
            "the preset-free path is then a second copy of the raw reads and the "
            "feature-off identity stops being a property of the source")
    if "environmentSearchPolicy();" not in fidelity:
        problems.append(
            "CaseFidelity::searchPolicy() does not resolve the no-row case through "
            "environmentSearchPolicy(). It then answers processSearchPolicy(), which "
            "reads RASBERY_FIDELITY -- so inside a `--set RASBERY_FIDELITY=screen100` "
            "process a promoted elite (or any \"fidelity\":\"strict\" case) runs the "
            "row\'s boron_bracket and staged margin while reporting policy:strict, and "
            "folds a case key indistinguishable from a genuine strict run.")
    if "processSearchPolicy();" in fidelity:
        problems.append(
            "CaseFidelity still falls back to processSearchPolicy() somewhere. That "
            "function resolves the PROCESS row, so it is never the right answer for a "
            "case that named no row.")
    return problems


def check_device_mirror() -> list[str]:
    """The device twin of the RODCRIT cap is RESET-ONLY, and says so.

    tools/test_gpu_physics_interface_contract.py pins that the field EXISTS --
    a host search parameter with no device twin is a mirror that stopped being
    one -- but nothing pins where its VALUE comes from, and today it comes from
    the built-in: `resetDeviceScheduleParams` writes kDevRodCritSearchTol and no
    path feeds it ctx.tolerances.rodcrit_search_cap.  The day the device search
    goes live that is one solve with two search tolerances, which is the "half
    applied" failure this whole file exists to catch, on the one axis nobody is
    watching.  Until it is wired, the REMINDER is the artifact and this is what
    stops it being deleted quietly.
    """
    problems: list[str] = []
    slots = read("src/GpuSlotControl.h")
    wired = "ctx.tolerances.rodcrit_search_cap" in slots or \
            "tol.rodcrit_search_cap" in slots
    if not wired and "TODO(WP24-device-search)" not in slots:
        problems.append(
            "src/GpuSlotControl.h neither feeds DeviceScheduleParams::rodcrit_search_cap "
            "from the case tolerances nor carries the TODO(WP24-device-search) marker "
            "that says it does not. A reset-only mirror field with no reminder is a "
            "device search that will silently clamp at the built-in while the host "
            "clamps at the preset\'s.")
    return problems


def check_case_key(driver: str) -> list[str]:
    problems: list[str] = []
    anchor = "inline constexpr const char* kArmEnv[] = {"
    if anchor not in driver:
        return ["cannot find kArmEnv in src/Driver.h"]
    block = driver[driver.index(anchor):]
    block = block[:block.index("};")]
    if '"RASBERY_FIDELITY"' not in block:
        problems.append(
            "RASBERY_FIDELITY is not in trajectory::kArmEnv. The preset moves the "
            "PUBLISHED tolerances, so two runs that differ only in it converge to two "
            "different answers -- and a cache that did not fold it would serve a 100 pcm "
            "screening k_eff to a production request.")
    if 'std::strcmp(name, "RASBERY_FIDELITY") == 0' not in driver:
        problems.append(
            "armEnvValue() has no per-case branch for the preset. A screen100 case and a "
            "promoted strict case answered by ONE process would then fold the same "
            "environment string and collide -- the defect the three staged branches "
            "beside it were written to fix.")
    # WP24 (review).  AND THE FIVE SEARCH KNOBS, FOLDED PER CASE.
    #
    # THE COLLISION.  A preset row ASSERTS these five, so with the row applied
    # per case the RAW environment string is not what the case ran: an env-level
    # screen100 (RASBERY_SEARCH_BORON_BRACKET unset, bracket ON from the row)
    # and a per-case screen100 in a clean process folded the SAME payload for
    # what were, before the search policy became per case, two different solves
    # -- a wrong cache hit, which is the one direction the kArmEnv comment says
    # the list must never allow.  Folding the EFFECTIVE policy is the same rule
    # the three staged branches beside it follow, and it keeps the raw string
    # whenever the case asked for nothing different, so every key this tree has
    # computed for a preset-free run is unchanged.
    for knob in ("RASBERY_SEARCH_CARRY_SLOPE", "RASBERY_SEARCH_WARM_BORON",
                 "RASBERY_SEARCH_BORON_BRACKET", "RASBERY_SEARCH_MAX_TRIALS",
                 "RASBERY_SEARCH_STAGED_MARGIN"):
        if knob not in block:
            problems.append(f"{knob} is not in trajectory::kArmEnv")
        if f'std::strcmp(name, "{knob}") == 0' not in driver:
            problems.append(
                f"armEnvValue() has no per-case branch for {knob}. A preset row asserts "
                f"this knob, so the RAW environment string is not what the case ran, and "
                f"an env-level preset and a per-case one fold one key for two policies.")
    if "const SearchPolicy  case_search = fidelity.searchPolicy();" not in driver:
        problems.append(
            "armEnvValue() does not resolve the CASE's search policy; folding "
            "processSearchPolicy() there would be the environment's answer again")
    # And the Python side must agree without a second copy.
    import case_key  # noqa: PLC0415
    if "RASBERY_FIDELITY" not in case_key.ARM_ENV:
        problems.append("tools/case_key.py does not see RASBERY_FIDELITY; the two key "
                        "implementations have parted company")
    # WP24 (review).  SEEING THE NAME IS NOT THE SAME AS FOLDING THE RIGHT
    # PAYLOAD.  `case_key.ARM_ENV` is parsed out of kArmEnv, so it picked up
    # RASBERY_FIDELITY for free -- but `effective_fidelity()` mirrors
    # effectivePhysicsFidelity() "rank for rank" and read ONLY RASBERY_STAGED_*
    # and RASBERY_GA_FEEDBACK_PASSES.  With `RASBERY_FIDELITY=screen100` and the
    # staged names UNSET (the spelling docs/WP24 Sec 7.1 calls mandatory) it
    # answered rank 0 -- ("strict", "full_exact") -- while the binary folds
    # ("staged_a2", "A2") and exact_audit derives "A2".  Both strings are
    # payload lines in payload_of(), so the mirror computed a DIFFERENT key from
    # the binary for exactly the invocation the runbook prescribes.
    checks = [
        ({"RASBERY_FIDELITY": "screen100"}, 1,
         "a screen100 environment must fold the A2 rank: the row\'s staged multipliers "
         "are above 1.0 and the binary reports A2"),
        ({"RASBERY_FIDELITY": "strict", "RASBERY_STAGED_FLUX_TOL": "50"}, 0,
         "a `strict` preset must CLEAR an exported A2 environment; the row replaces the "
         "two multipliers rather than defaulting them"),
        ({"RASBERY_FIDELITY": "A2"}, 1, "the A2 row folds the A2 rank"),
        ({}, 0, "an empty environment must still fold strict"),
        ({"RASBERY_STAGED_FLUX_TOL": "50"}, 1,
         "the pre-WP24 staged read must survive with no preset named"),
    ]
    for env, want, why in checks:
        got = case_key.effective_fidelity(env)
        if got != want:
            problems.append(
                f"tools/case_key.py effective_fidelity({env!r}) = {got}, want {want}: "
                f"{why}. The mirror then computes a different case key from the binary "
                f"for that environment, which is a cache MISS at best and, between two "
                f"tools that both claim to implement one key, a contract nobody can "
                f"rely on.")
    # ...and it must agree with the audit\'s derivation, which is the OTHER
    # python answer to the same question.
    for env in ({"RASBERY_FIDELITY": "screen100"}, {"RASBERY_FIDELITY": "strict"},
                {"RASBERY_STAGED_FLUX_TOL": "50"}, {}):
        rank = case_key.effective_fidelity(env)
        policy = case_key.FIDELITY_TRAITS[rank][0]
        derived = exact_audit.derive_declared_fidelity(env)
        if policy != derived:
            problems.append(
                f"for {env!r} tools/case_key.py folds policy {policy!r} while "
                f"tools/exact_audit.py derives {derived!r}. Two python answers to one "
                f"question is how the harness declares one fidelity and keys another.")
    return problems


# ---------------------------------------------------------------------------
# 6. THE TOOLS' ENVELOPES -- behavioural
# ---------------------------------------------------------------------------

def quiet_report(envelope, measured, label, pinned=()):
    """gate_b_envelope.report() with its printing swallowed.

    report() is a PRINTING function -- that is the half of it worth testing, and
    it is also why a contract test that called it raw would bury its own verdict
    under three envelope notes.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return gate_b_envelope.report(envelope, measured, label, pinned)


def check_envelopes() -> list[str]:
    problems: list[str] = []
    if gate_b_envelope.DEFAULT_ENVELOPE != "production":
        problems.append(
            "the Gate B default envelope is not `production`. A screening envelope that "
            "is reachable by omission is a screening number filed in an acceptance "
            "column.")
    prod = gate_b_envelope.ENVELOPES["production"]
    screen = gate_b_envelope.ENVELOPES["screen100"]

    # THE PRODUCTION ROW IS THE ACCEPTANCE COLUMN, AND THE PROOF IS THE FROZEN
    # RESULT.  The first draft of gate_b_envelope took the MEASURED v2 figures
    # (1.905 pcm / 15.309 ppm / 0.238 % pin RMS) out of the two-column table in
    # docs/A2_OUTER_REDUCTION_20260829_KO.md Sec 5 instead of the acceptance
    # bars beside them.  `production` is the DEFAULT envelope, so that made
    # every existing compare_master_rasbery.py invocation exit 1 on a run the
    # campaign had already accepted: docs/PRICING_PROD_20260830_KO.md records
    # the accepted production Gate B at max|dppm| = 15.334 against a 15.309
    # limit.  A limit set to a measurement is a limit the next run of the same
    # code fails by rounding, and this is the check that says so in a way that
    # cannot be argued with.
    accepted_production = {"keff_pcm": 1.847, "ppm": 15.334, "ao": 0.012,
                           "pin_rms_pct": 0.238, "pin_max_pct": 0.80}
    frozen_ok, frozen_code = quiet_report(prod, accepted_production, "GATE B test")
    if not frozen_ok or frozen_code != 0:
        problems.append(
            f"the ACCEPTED production Gate B (docs/PRICING_PROD_20260830_KO.md: "
            f"{accepted_production}) does not pass the `production` envelope. That is the "
            f"campaign\'s own frozen result failing the DEFAULT gate on its first day, "
            f"which is how a gate gets switched off permanently. State the row from the "
            f"ACCEPTANCE column of the source table, not from the measurement column "
            f"beside it.")
    for field, value in (("keff_pcm", 100.0), ("pin_rms_pct", 1.0), ("pin_max_pct", 1.0)):
        if getattr(screen, field) != value:
            problems.append(f"screen100 envelope {field} is {getattr(screen, field)}, "
                            f"the directive says {value}")
    if screen.ao is not None:
        problems.append("the screen100 envelope invents an AO limit. There is no "
                        "defensible tie from a k_eff budget to an axial offset, so AO is "
                        "reported and not failed on.")
    # THE CBC COLUMN, AND WHY IT IS NOT 100/5.4.
    #
    # A boron-search statepoint has no |dkeff| to score -- k_eff is at target by
    # construction -- so 100 pcm has to be carried into ppm or every boron deck
    # is ungated.  But these limits are ABSOLUTE MASTER-relative bounds, and the
    # production baseline is already inside them: keff is measured from 1.905
    # and CBC from 15.309.  A ppm limit of 100/5.4 = 18.5 would grant keff ~98
    # pcm of NEW error and boron 3.2 ppm (~17 pcm-equivalent) under one name --
    # less than screen100's own knobs spend, which is what check_screen100's ppm
    # cross-check now catches from the other side.  So the column is the
    # BASELINE PLUS THE IMAGE OF THE SAME BUDGET, and this check pins the
    # relationship rather than the number.
    if screen.baseline != "production":
        problems.append(
            "the screen100 envelope does not name production as its baseline, so no "
            "verdict line can say how much NEW error a breach actually had to work with "
            "-- which is the whole reading a screening FAIL needs")
    expected_ppm = prod.ppm + (screen.keff_pcm - prod.keff_pcm) / \
        gate_b_envelope.BORON_SLOPE_PCM_PER_PPM
    if screen.ppm is None or abs(screen.ppm - expected_ppm) > 1e-9:
        problems.append(
            f"the screen100 CBC limit is {screen.ppm!r}; it has to be the production "
            f"baseline plus the ppm image of the SAME keff budget "
            f"({prod.ppm} + ({screen.keff_pcm} - {prod.keff_pcm})/"
            f"{gate_b_envelope.BORON_SLOPE_PCM_PER_PPM} = {expected_ppm:.3f}), or the "
            f"boron column and the keff column grant different amounts of new error "
            f"under one name.")
    for field in ("keff_pcm", "ppm", "pin_rms_pct", "pin_max_pct"):
        if getattr(screen, field) is not None and \
                getattr(screen, field) <= getattr(prod, field):
            problems.append(f"screen100's {field} is not looser than production's; it is "
                            f"then not a screening envelope")
        room = gate_b_envelope.headroom(screen, field)
        if room is None or room <= 0.0:
            problems.append(
                f"gate_b_envelope.headroom(screen100, {field}) is {room!r}. The effective "
                f"headroom is the number a reader of a FAIL needs -- production already "
                f"measures 0.80 % pin max against a 1 % screening limit, i.e. 0.20 pp -- "
                f"and it must be computable, not a figure in a handoff note.")
    # The note is where a reader of a FAIL looks, so the binding column's
    # arithmetic has to be IN it and not only in this test.
    for token in ("0.20", "baseline"):
        if token not in screen.note:
            problems.append(
                f"the screen100 envelope note does not mention {token!r}. A reader of a "
                f"pin FAIL has to be able to see that only 0.20 pp of the 1 % limit was "
                f"ever screening headroom, and the note is what the tools print.")

    # THE VERDICT ITSELF.  Pass, fail, advisory, and absent.
    ok, _ = gate_b_envelope.verdict(screen, {"keff_pcm": 42.0, "pin_rms_pct": 0.3,
                                             "pin_max_pct": 0.9})
    if not ok:
        problems.append("verdict() failed a measurement inside the screen100 envelope")
    bad, _ = gate_b_envelope.verdict(screen, {"keff_pcm": 42.0, "pin_max_pct": 1.4})
    if bad:
        problems.append("verdict() passed a pin max of 1.4 % against a 1 % limit")
    tight, _ = gate_b_envelope.verdict(prod, {"keff_pcm": 42.0})
    if tight:
        problems.append("verdict() passed 42 pcm against the production envelope's 1.905")
    advisory, lines = gate_b_envelope.verdict(screen, {"ao": 99.0})
    if not advisory or not any("advisory" in line for line in lines):
        problems.append("an advisory metric is failing the gate or is not labelled")
    empty, _ = gate_b_envelope.verdict(screen, {})
    if not empty:
        problems.append("verdict() fails when nothing was measured; a metric a deck does "
                        "not produce must be skipped, not assumed")

    # ...AND NOTHING MAY REPORT THAT AS A PASS.  verdict()'s skip-don't-assume
    # rule is right, and it means an empty measurement dict "passes" -- so a
    # compare run that produced none of delta_pcm / delta_ppm / delta_ao would
    # have printed "GATE B scalars: PASS" having judged nothing, which is the
    # unconditional `exit 0` this module replaced, wearing a verdict's clothes.
    if gate_b_envelope.scored(screen, {}):
        problems.append("scored() claims metrics were judged for an empty measurement")
    if gate_b_envelope.scored(screen, {"ao": 1.0}):
        problems.append("scored() counts an ADVISORY metric as judged; an AO-only run "
                        "would then report a pass it never earned")
    if gate_b_envelope.scored(screen, {"keff_pcm": 1.0}) != ["keff_pcm"]:
        problems.append("scored() does not report the metric it judged")
    # A MEASURED COLUMN THAT COULD NOT HAVE FAILED IS NOT EVIDENCE.  On a
    # rod-crit deck (the iSMR / CY families) `delta_ppm` is the deck\'s own
    # fixed boron on both sides -- exactly 0.000 on every joined statepoint,
    # whatever the physics did -- and `delta_ao` is advisory under screen100.  A
    # compare run over such a deck would otherwise print "GATE B scalars: PASS"
    # having judged one structurally inert column, which is the unconditional
    # `exit 0` this module replaced wearing a verdict\'s clothes.
    if gate_b_envelope.scored(screen, {"ppm": 0.0}, ["ppm"]):
        problems.append("scored() counts a STRUCTURALLY PINNED metric as judged")
    if gate_b_envelope.scored(screen, {"ppm": 0.0, "keff_pcm": 3.0}, ["ppm"]) != \
            ["keff_pcm"]:
        problems.append("scored() drops more than the pinned metric, or keeps it")
    pinned_passed, pinned_code = quiet_report(
        screen, {"ppm": 0.0}, "GATE B test", ["ppm"])
    if pinned_passed or pinned_code != 2:
        problems.append(
            f"report() returned {(pinned_passed, pinned_code)!r} for a run whose only "
            f"judged column was structurally pinned. A gate that passes on a column that "
            f"cannot fail is the failure mode `pinned` exists to end -- and it is exactly "
            f"the state a rod-crit deck puts the screen100 envelope in.")
    # ...and a pinned column that somehow BREACHES is still a real finding.
    breach_pinned, _ = gate_b_envelope.verdict(screen, {"ppm": 999.0}, ["ppm"])
    if breach_pinned:
        problems.append("verdict() stops judging a pinned metric; a pinned column that "
                        "breaches is a finding, not a skip")
    nothing_passed, nothing_code = quiet_report(screen, {}, "GATE B test")
    if nothing_passed or nothing_code == 0:
        problems.append(
            f"report() returned {(nothing_passed, nothing_code)!r} for a run that "
            f"measured nothing. A gate that passes on no data is the failure mode this "
            f"module exists to end.")
    breach_passed, breach_code = quiet_report(
        screen, {"pin_max_pct": 1.4}, "GATE B test")
    if breach_passed or breach_code != 1:
        problems.append("report() does not return exit code 1 on a breach")
    ok_passed, ok_code = quiet_report(
        screen, {"pin_max_pct": 0.9}, "GATE B test")
    if not ok_passed or ok_code != 0:
        problems.append("report() does not return exit code 0 on a clean measurement")

    # And both tools have to actually take the flag and be able to exit nonzero.
    for tool in ("tools/compare_master_rasbery.py", "tools/gate_b_pin_rms.py"):
        text = read(tool)
        if "add_envelope_argument" not in text:
            problems.append(f"{tool} does not take --envelope")
        if "gate_b_envelope.report(" not in text or "return 0 if passed else code" not in text:
            problems.append(f"{tool} cannot fail: it does not take its exit code from "
                            f"gate_b_envelope.report(), which is the one place that "
                            f"refuses to call an unscored run a pass")
        if "raise SystemExit(main())" not in text:
            problems.append(f"{tool} computes a verdict and then throws the exit code "
                            f"away")

    # WP24 (review).  THE PIN TOOL MAY NOT SCORE BURNUP AGAINST A BOC MAP.
    #
    # `--all-steps` first shipped comparing EVERY statepoint against the single
    # positional PPI -- which every documented invocation passes as
    # `kngr_mas_ppi_boc.txt` -- and taking the verdict from the worst.  BOC-to-EOC
    # pin redistribution is real physics of order several to tens of percent, so
    # that gate returns FAIL on any correct run, which is as uninformative as the
    # exit 0 it replaced; and the runbook made it the screen100 pin verdict.
    pin = read("tools/gate_b_pin_rms.py")
    compare = read("tools/compare_master_rasbery.py")
    if "pinned" not in compare or "gate_b_envelope.report(envelope, measured" not in compare:
        problems.append(
            "tools/compare_master_rasbery.py does not tell gate_b_envelope which columns "
            "are STRUCTURALLY PINNED. On a rod-crit deck delta_ppm is one constant minus "
            "the same constant on every row, and scoring it is a verdict about the "
            "deck\'s input file.")
    if "maxes[column] == 0.0" not in compare:
        problems.append(
            "tools/compare_master_rasbery.py does not detect a delta column that is "
            "identically zero on every joined statepoint. That is the only signal it has "
            "for a quantity the DECK fixes on both sides, and the test has to be exact "
            "equality so a near-zero real agreement keeps scoring.")
    if "--ppi-step" not in pin:
        problems.append(
            "tools/gate_b_pin_rms.py takes no per-statepoint MASTER reference, so a "
            "statepoint past BOC can only ever be compared against a BOC map")
    if "scored_rows = [r for r in rows if r[3]]" not in pin:
        problems.append(
            "tools/gate_b_pin_rms.py does not separate SCORED statepoints (compared "
            "against their own reference) from BOC-referenced DRIFT. A verdict taken "
            "over the drift rows is a guaranteed FAIL that says nothing about screening "
            "error.")
    for needle in ("worst_rms = max(scored_rows,", "worst_max = max(scored_rows,"):
        if needle not in pin:
            problems.append(
                f"the pin verdict is not taken over the scored population "
                f"(missing: {needle}) -- the worst of the drift rows is a BOC-referenced "
                f"burnup reading and cannot certify anything either way")
    return problems


# ---------------------------------------------------------------------------
# 7. THE AUDIT'S DERIVATION
# ---------------------------------------------------------------------------

def check_audit() -> list[str]:
    problems: list[str] = []
    if exact_audit.derive_declared_fidelity({}) != "strict":
        problems.append("an empty environment no longer derives `strict`")
    if exact_audit.derive_declared_fidelity(
            {"RASBERY_STAGED_FLUX_TOL": "50"}) != "A2":
        problems.append("the pre-WP24 A2 derivation changed")
    if exact_audit.derive_declared_fidelity({"RASBERY_FIDELITY": "screen100"}) != "A2":
        problems.append(
            "a screen100 environment does not derive `A2`. The binary reports A2 (the "
            "row's multipliers are above 1.0), so a harness deriving anything else fails "
            "every screen100 run on a policy mismatch -- the WP4 defect, re-committed.")
    # PRESET WINS OVER THE ENVIRONMENT, both directions.
    if exact_audit.derive_declared_fidelity(
            {"RASBERY_FIDELITY": "strict", "RASBERY_STAGED_FLUX_TOL": "50"}) != "strict":
        problems.append(
            "a `strict` preset does not clear an exported A2 environment. That is the "
            "promotion lane, and a promotion that inherits the campaign's multipliers is "
            "an unverified elite that looks verified.")
    try:
        exact_audit.derive_declared_fidelity({"RASBERY_FIDELITY": "screeen100"})
    except ValueError:
        pass
    else:
        problems.append("a typo'd preset name derives a fidelity instead of raising. The "
                        "binary refuses it, so the harness would declare a policy for a "
                        "run that exits 2 before printing one.")
    if "RASBERY_FIDELITY" not in exact_audit.NON_STRICT_ENV_KEYS:
        problems.append(
            "`--strict` does not clear RASBERY_FIDELITY from the child. Clearing "
            "RASBERY_STAGED_* while leaving the preset behind produces a child that is "
            "still A2, still at 100 pcm tolerances, and says so nowhere the caller looked.")
    return problems


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

def run(name: str, checker, *args) -> None:
    problems = checker(*args)
    for problem in problems:
        fail(f"[{name}] {problem}")


run("table", check_table_single_source, PRESET_H, DRIVER, SCHEDULER, BATCH)
run("screen100", check_screen100, PRESETS)
run("polish-invariant", check_polish_invariant, PRESETS)
run("row-digest", check_row_digests, PRESETS)
run("feature-off", check_feature_off, CONTRACT, FIDELITY, SCHEDULER, DRIVER, PRESET_H)
run("wiring", check_wiring, DRIVER, CONTRACT, FIDELITY, SCHEDULER)
run("cleared-preset", check_cleared_preset, SCHEDULER, FIDELITY)
run("device-mirror", check_device_mirror)
run("receipts", check_receipts, DRIVER, SERVER, LIGHT, MAIN)
run("case-key", check_case_key, DRIVER)
run("envelope", check_envelopes)
run("audit", check_audit)

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS -- every source check re-run against a copy broken in the
# way it exists to catch.  A check that survives its own control is a comment.
# ---------------------------------------------------------------------------
negative: list[str] = []


def control(name: str, checker, *args) -> None:
    if not checker(*args):
        negative.append(name)


def broken_preset(row: str, field: str, value: str) -> dict:
    import copy
    presets = copy.deepcopy(PRESETS)
    presets[row][field] = value
    return presets


control("check_table_single_source misses a strict row that does not restate the tree",
        check_table_single_source,
        PRESET_H.replace("inline constexpr double kProdXeTol              = 1.0e-6;",
                         "inline constexpr double kProdXeTol              = 1.0e-5;"),
        DRIVER, SCHEDULER, BATCH)
control("check_table_single_source misses an A2 row that is not DEFAULT_ENV's arm",
        check_table_single_source,
        PRESET_H.replace("/*staged_xe_mult*/               1000.0,",
                         "/*staged_xe_mult*/               999.0,"),
        DRIVER, SCHEDULER, BATCH)
control("check_table_single_source misses the rod-crit clamp left as a literal",
        check_table_single_source, PRESET_H, DRIVER,
        SCHEDULER.replace("std::min(base, rodcrit_search_cap)",
                          "std::min(tolerance_search, kRodCritSearchTol)"),
        BATCH)
control("check_screen100 misses a search tolerance at the acceptance envelope",
        check_screen100, broken_preset("screen100", "search_tol_mult", "100.0"))
control("check_screen100 misses an oscillation floor left to float with the Xe tolerance",
        check_screen100, broken_preset("screen100", "xe_oscillation_floor", "1.0e-3"))
control("check_screen100 misses an UNMEASURED rod-crit cap relaxation, which is the "
        "one place the screen100 envelope has no column that can fail",
        check_screen100, broken_preset("screen100", "rodcrit_search_cap", "1.0e-4"))
control("check_screen100 misses an unmeasured rod-crit cusping floor relaxation",
        check_screen100,
        broken_preset("screen100", "rodcrit_search_floor_cusping", "1.0e-4"))
control("check_screen100 misses a sweep exit LOOSER than the outer flux tolerance",
        check_screen100, broken_preset("screen100", "cmfd_sweep_epsl2", "1.0e-3"))
# ...AND THE CORRECTION HAS TO BE A CORRECTION, not a rename.  The whole point
# of rewriting the epsl2 predicate is that 1e-4 -- production\'s 1:1 ratio, the
# cheaper AND stricter direction -- is now LEGAL.  A check that still forbade it
# would be the old wrong premise wearing a new comment.
if [q for q in check_screen100(
        {"strict": PRESETS["strict"],
         "screen100": dict(PRESETS["screen100"], cmfd_sweep_epsl2="1.0e-4")})
        if "sweep exit" in q]:
    fail("the CMFD sweep-exit check still forbids epsl2 == flux_tol. That is the "
         "STRICTEST reading of the outer L2 half (break -> pass, cap-exhaust -> fail) "
         "and the cheaper one; forbidding it re-commits the backwards premise the "
         "predicate was rewritten to correct.")
control("check_screen100 misses the statepoint-grid knob switched on",
        check_screen100, broken_preset("screen100", "statepoint_grid", '"coarse"'))
control("check_screen100 misses carry_slope turned on",
        check_screen100, broken_preset("screen100", "carry_slope", "PresetFlag::On"))
# The ppm control keeps the 10:1 search-to-keff ratio and stays under both the
# keff-column checks, so the ONLY check that can catch it is the boron one --
# which is the point: a control that trips a neighbouring check proves nothing
# about the check it was written for.
# It moves the CAP and not the multiplier, because the cap is now what bounds
# the worst case a deck can reach -- which is the whole point of the column.
BORON_HUNGRY = {"strict": PRESETS["strict"],
                "screen100": dict(PRESETS["screen100"], search_tol_cap="5.0e-4")}
if [p for p in check_screen100(BORON_HUNGRY) if "boron headroom" not in p]:
    fail("the boron-headroom negative control trips a DIFFERENT check, so it does not "
         "prove the ppm cross-check can fail:\n    " +
         "\n    ".join(check_screen100(BORON_HUNGRY)))
control("check_screen100 misses a search-tolerance CAP that eats the boron headroom",
        check_screen100, BORON_HUNGRY)
control("check_screen100 misses a search multiplier left UNCAPPED against a deck-stated "
        "tolerance",
        check_screen100, broken_preset("screen100", "search_tol_cap", "0.0"))
control("check_row_digests misses a row retuned in place",
        check_row_digests, broken_preset("screen100", "xe_tol", "1.0e-4"))
control("check_polish_invariant misses an acceptance-eligible row with moved polish "
        "tolerances",
        check_polish_invariant,
        {"strict": PRESETS["strict"],
         "sneaky": dict(PRESETS["screen100"], staged_flux_mult="1.0",
                        staged_xe_mult="1.0")})
control("check_wiring misses a search policy that never reaches the solve",
        check_wiring,
        DRIVER.replace("ctx.search_policy = _fidelity.searchPolicy();", "/* no */"),
        CONTRACT, FIDELITY, SCHEDULER)
control("check_wiring misses a CaseFidelity with no searchPolicy()",
        check_wiring, DRIVER, CONTRACT,
        FIDELITY.replace("presetSearchPolicy(spec) : environmentSearchPolicy();",
                         "environmentSearchPolicy();"),
        SCHEDULER)
control("check_cleared_preset misses a cleared preset that inherits the PROCESS row\'s "
        "search knobs -- the acceptance lane running screen100\'s boron_bracket",
        check_cleared_preset, SCHEDULER,
        FIDELITY.replace("presetSearchPolicy(spec) : environmentSearchPolicy();",
                         "presetSearchPolicy(spec) : processSearchPolicy();"))
control("check_cleared_preset misses environmentSearchPolicy() resolving a row, which "
        "would make the split a rename",
        check_cleared_preset,
        SCHEDULER.replace(
            "        p.carry_slope   = searchFlagEnabled(std::getenv(\"RASBERY_SEARCH_CARRY_SLOPE\"));",
            "        if (const FidelityPresetSpec* s = lookupFidelityPreset(\"A2\")) return presetSearchPolicy(s);\n"
            "        p.carry_slope   = searchFlagEnabled(std::getenv(\"RASBERY_SEARCH_CARRY_SLOPE\"));",
            1),
        FIDELITY)
control("check_wiring misses a resurrected second setEpsl2 spelling",
        check_wiring,
        DRIVER.replace("cmfd_solver.setNcmfd(5);",
                       "cmfd_solver.setNcmfd(5);\n        cmfd_solver.setEpsl2(1.0e-6);"),
        CONTRACT, FIDELITY, SCHEDULER)
control("check_receipts misses search knobs printed from the table row",
        check_receipts,
        DRIVER.replace("sp.boron_bracket ?", "preset_spec->boron_bracket == PresetFlag::On ?"),
        SERVER, LIGHT, MAIN)
control("check_case_key misses armEnvValue with no per-case search branch",
        check_case_key,
        DRIVER.replace('std::strcmp(name, "RASBERY_SEARCH_BORON_BRACKET") == 0',
                       'std::strcmp(name, "NOPE") == 0'))
control("check_feature_off misses a lost pre-WP24 read",
        check_feature_off,
        CONTRACT.replace('detail::stagedMultiplier("RASBERY_STAGED_FLUX_TOL")',
                         "presetOnly()"),
        FIDELITY, SCHEDULER, DRIVER, PRESET_H)
control("check_feature_off misses an unconditional receipt line",
        check_feature_off, CONTRACT, FIDELITY, SCHEDULER,
        DRIVER.replace("if (const FidelityPresetSpec* preset_spec = _fidelity.presetSpec())",
                       "{ const FidelityPresetSpec* preset_spec = &kFidelityPresets[0];"),
        PRESET_H)
control("check_feature_off misses a SolveTolerances default that is not the built-in",
        check_feature_off, CONTRACT, FIDELITY, SCHEDULER, DRIVER,
        PRESET_H.replace("double xe_tol                       = kProdXeTol;",
                         "double xe_tol                       = 1.0e-5;"))
control("check_wiring misses tolerances that never reach the solve",
        check_wiring, DRIVER.replace("ctx.tolerances = _fidelity.tolerances();", "/* no */"),
        CONTRACT, FIDELITY, SCHEDULER)
control("check_wiring misses a device Xe request left at the production tolerance",
        check_wiring, DRIVER.replace("req.eq_tol   = ctx.tolerances.xe_tol;",
                                     "req.eq_tol   = XE_EQUILIBRIUM_TOLERANCE;"),
        CONTRACT, FIDELITY, SCHEDULER)
control("check_wiring misses a CMFD sweep exit that ignores the preset",
        check_wiring,
        DRIVER.replace("cmfd_solver.setEpsl2(ctx.tolerances.cmfd_sweep_epsl2);", "/* no */"),
        CONTRACT, FIDELITY, SCHEDULER)
control("check_receipts misses a [CASE] line with no preset",
        check_receipts, DRIVER.replace("fidelity_preset", "fp"), SERVER, LIGHT, MAIN)
control("check_receipts misses a promotion that does not clear the preset",
        check_receipts, DRIVER,
        SERVER.replace("request.request_fidelity.has_preset", "false_flag"), LIGHT, MAIN)
control("check_case_key misses a preset outside kArmEnv",
        check_case_key, DRIVER.replace('    "RASBERY_FIDELITY",\n', ""))
control("check_case_key misses armEnvValue with no per-case branch",
        check_case_key,
        DRIVER.replace('std::strcmp(name, "RASBERY_FIDELITY") == 0',
                       'std::strcmp(name, "NOPE") == 0'))

if negative:
    fail("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are therefore "
         "comments:\n    " + "\n    ".join(negative))

if FAILED:
    raise SystemExit("fidelity preset contract: FAIL\n  - " + "\n  - ".join(FAILED))
print(f"fidelity preset contract: PASS ({len(PRESETS)} preset rows, "
      f"{len(gate_b_envelope.ENVELOPES)} Gate B envelopes, "
      f"screen100 = {gate_b_envelope.ENVELOPES['screen100'].keff_pcm:.0f} pcm / "
      f"{gate_b_envelope.ENVELOPES['screen100'].pin_rms_pct:.1f} % pin)")
