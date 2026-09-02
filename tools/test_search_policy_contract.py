#!/usr/bin/env python3
"""WP9-D stage D: the critical search's trial-reduction levers.

WP9-D shipped the instrument and changed nothing, because "137 trials" is one
number standing for four different problems and the bottleneck plan says which
lever applies depends on which of the four the 238 distribution is made of.
Stage D is the five levers themselves, and everything that can go wrong with a
lever like this is one of four things -- never an arithmetic one.

  1. IT RUNS WHEN NOBODY ASKED.  Feature-off byte identity is the campaign's
     standing mandate (there is an unattributed flag-off drift being chased
     between 7cfe3a4 and d7b81af), so every knob must default to false / zero /
     the built-in constant, and the expression each one guards must be the
     expression this tree already had.  Section 2 walks every site.

  2. THE ARM CANNOT BE NAMED.  A knob that moves a trajectory and is not in
     trajectory::kArmEnv is a knob the WP10.1 case key does not fold, and a
     cached scalar answer computed under one search policy would be served to a
     request made under another.  Section 3.

  3. THE LEVERS ARE FUSED.  The doc states a revert condition PER CANDIDATE
     ("probe+carry 대비 trials 증가 시 즉시 폐기"), so a single knob turning on
     two levers is a knob that cannot be reverted and an A/B that cannot say
     which half paid.  Section 4 requires five independent flags and five
     independent reads.

  4. THE RECEIPT CANNOT PRICE IT.  The adoption bar is stated in outers and
     trials, and the wall-timing arm runs with telemetry OFF.  A ledger that
     only exists under RASBERY_STATEPOINT_TELEMETRY is a ledger that cannot be
     read on the arm the lever is supposed to shorten.  Section 5.

And one that is arithmetic, because it is the only place stage D does any:
section 6 runs Scheduler.h's slope extrapolation as a transcription and pins
each of its three guards with a negative control.  A guard that had been
removed would otherwise show up only as a worse first trial on 238.
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

DRIVER = (SRC / "Driver.h").read_text(encoding="utf-8-sig")
SCHEDULER = (SRC / "Scheduler.h").read_text(encoding="utf-8-sig")

problems: list[str] = []


def fail(message: str) -> None:
    problems.append(message)


def want(code: str, needle: str, where: str, why: str) -> None:
    if needle not in code:
        fail(f"{where}: {why} (looked for {needle!r})")


def region(code: str, start: str, end: str, name: str) -> str:
    i = code.find(start)
    if i < 0:
        fail(f"cannot find {name!r} ({start!r})")
        return ""
    j = code.find(end, i + len(start))
    return code[i:j if j > 0 else len(code)]


KNOBS = (
    "RASBERY_SEARCH_CARRY_SLOPE",
    "RASBERY_SEARCH_WARM_BORON",
    "RASBERY_SEARCH_BORON_BRACKET",
    "RASBERY_SEARCH_MAX_TRIALS",
    "RASBERY_SEARCH_STAGED_MARGIN",
)

# ---------------------------------------------------------------------------
# 1. The policy is a VALUE ON THE CONTEXT, not a static inside the solve.
#
#    WP10.3 spent a commit removing exactly this defect for the staged
#    tolerances: SolveLoop is static, so a policy that is not on SolverContext
#    has to be a function-local static inside it, and that latches the first
#    case of a process onto every case after it.  In an evaluator answering a
#    mixed wave that is sixty-three cases running a policy nobody asked them
#    for under receipts that say otherwise.
# ---------------------------------------------------------------------------
want(DRIVER, "SearchPolicy search_policy = processSearchPolicy();", "Driver.h SolverContext",
     "the search policy must be a member of SolverContext, defaulted from the process "
     "environment, for the reason CaseFidelity is one")
want(DRIVER, "SearchCarry  search_carry{};", "Driver.h SolverContext",
     "the cross-statepoint slope history must live for exactly the Driver's lifetime: a "
     "carried boron worth belongs to one core and must not survive a slot refill")

# The single reader, and the single truthiness spelling.  A second spelling is
# how `RASBERY_SEARCH_WARM_BORON=0` turns a feature ON.
#
# RASBERY_SEARCH_TRACE is deliberately NOT one of these knobs and SolveLoop does
# read it: it prints a line and changes nothing, which is why it is not in
# kArmEnv either.  The scan below is per knob NAME rather than per prefix, so
# the trace knob is not mistaken for a policy one.
# WP24 (review).  THE ONE PLACE IS environmentSearchPolicy(), AND
# processSearchPolicy() IS THAT PLUS THE PRESET ROW.
#
# The split exists because "this case names no preset" has to be answerable
# WITHOUT the process's row: processSearchPolicy() resolves RASBERY_FIDELITY,
# so a case that CLEARED its preset -- a promotion, or any
# "fidelity":"strict" request -- used to inherit the process row's five
# search knobs, which inside a `--set RASBERY_FIDELITY=screen100` campaign
# meant the ACCEPTANCE lane ran screen100's boron_bracket and folded a case
# key indistinguishable from a genuine strict run.  The raw reads therefore
# live in environmentSearchPolicy() and processSearchPolicy() delegates to
# it; the "exactly one std::getenv per knob" rule below is unchanged and is
# what keeps the split from becoming two answers.
ENV_FN = region(SCHEDULER, "inline const SearchPolicy& environmentSearchPolicy() {",
                "\n}\n", "environmentSearchPolicy")
POLICY_FN = region(SCHEDULER, "inline const SearchPolicy& processSearchPolicy() {",
                   "\n}\n", "processSearchPolicy")
want(POLICY_FN, "environmentSearchPolicy()", "Scheduler.h processSearchPolicy",
     "processSearchPolicy() must fall through to environmentSearchPolicy() when no "
     "preset row is named, or the preset-free path is a second copy of the raw reads")
want(POLICY_FN, "lookupFidelityPreset(fidelityPresetEnvName())",
     "Scheduler.h processSearchPolicy",
     "processSearchPolicy() is the PROCESS answer and must resolve RASBERY_FIDELITY; "
     "a named row REPLACES these five knobs rather than defaulting them")
if "lookupFidelityPreset" in ENV_FN:
    fail("environmentSearchPolicy() resolves a preset row. It exists to be the read "
         "with NO row in it -- the answer a case that cleared its preset asked for "
         "-- and a row here makes the split a rename.")
for knob in KNOBS:
    want(ENV_FN, knob, "Scheduler.h environmentSearchPolicy",
         f"{knob} must be resolved in the one place every reader takes its value from")
    if DRIVER.count('std::getenv("%s")' % knob) != 0:
        fail(f"Driver.h reads {knob} directly; there must be exactly one reader")
    if SCHEDULER.count('std::getenv("%s")' % knob) != 1:
        fail(f"{knob} is read {SCHEDULER.count(chr(34) + knob + chr(34))} times in "
             "Scheduler.h; exactly one read is the contract")
want(SCHEDULER, 's == "0" || s == "off" || s == "OFF" || s == "false" ||',
     "Scheduler.h searchFlagEnabled",
     "the boolean knobs must use the truthiness spelling CaseFidelity.h already uses for "
     "RASBERY_STAGED_LOOSE_SETTLE; two spellings is how `=0` turns a feature on")

# ---------------------------------------------------------------------------
# 2. FEATURE-OFF IS THE OLD PATH, site by site.
#
#    Each lever is a value that is false / zero / the built-in, and the guarded
#    expression is quoted here so a refactor that "simplified" a fallback away
#    fails rather than drifting.
# ---------------------------------------------------------------------------
want(SCHEDULER, "bool   carry_slope   = false;", "Scheduler.h SearchPolicy",
     "RASBERY_SEARCH_CARRY_SLOPE must default OFF")
want(SCHEDULER, "bool   warm_boron    = false;", "Scheduler.h SearchPolicy",
     "RASBERY_SEARCH_WARM_BORON must default OFF")
want(SCHEDULER, "bool   boron_bracket = false;", "Scheduler.h SearchPolicy",
     "RASBERY_SEARCH_BORON_BRACKET must default OFF")
want(SCHEDULER, "int    max_trials    = 0;", "Scheduler.h SearchPolicy",
     "RASBERY_SEARCH_MAX_TRIALS must default to `no cap`")
want(SCHEDULER, "double staged_margin = 0.0;", "Scheduler.h SearchPolicy",
     "RASBERY_SEARCH_STAGED_MARGIN must default to `the built-in`")

want(SCHEDULER, "return staged_margin > 0.0 ? staged_margin : built_in;",
     "Scheduler.h SearchPolicy::stagedMargin",
     "an unset margin must answer with the caller's built-in constant, or the loose "
     "tolerance moves on a run that set nothing")
want(SCHEDULER, "return (max_trials > 0 && max_trials < deck_limit) ? max_trials : deck_limit;",
     "Scheduler.h SearchPolicy::trialCap",
     "the cap may only take trials AWAY: a knob above the deck's own max_search_iter "
     "must not grant trials the deck refused")

# D1's carry override: with the flag off it is a copy of the same double the
# expression always read, so the arithmetic is bit for bit the old one.
CARRY = region(SCHEDULER, "if (!search_has_prev) {", "} else {", "carry branch")
want(CARRY, "params.carry_override ? params.carry_slope : *params.secant_dkdx", "Scheduler.h",
     "the carry step must fall back to SearchMemory's own slope when D1 is off")
want(CARRY, "if (params.carry_available && std::abs(carried) >= min_secant_denom) {",
     "Scheduler.h", "and the carry test must read the same value the step uses, or the "
     "guard and the step disagree about which slope is being taken")

# The corrected slope must never be written back: the next correction is fitted
# to the MEASURED history, and a corrected value folded into it compounds.
PROPOSE = region(SCHEDULER, "bool ProposeNextSearchPoint(", "\n    /// WP9-D telemetry",
                 "ProposeNextSearchPoint")
if re.search(r"memory\.boron_secant_dkdx\s*=", PROPOSE):
    fail("ProposeNextSearchPoint writes the corrected slope back into SearchMemory; the "
         "extrapolation would then be fitted to its own output and compound")
want(PROPOSE, "params.carry_slope    = carriedBoronSlope(carry, memory.boron_secant_dkdx,",
     "Scheduler.h", "the correction must be computed from the MEASURED slope")
want(PROPOSE, "if (policy.carry_slope) {", "Scheduler.h",
     "and only when D1 is armed")
want(PROPOSE, "if (policy.boron_bracket) {", "Scheduler.h",
     "the boron bracket must be armed by its own flag and nothing else")
want(PROPOSE, "params.bracket_span_min = std::max(bracket_min_span, tolerance_boron);",
     "Scheduler.h",
     "a boron bracket must be judged against the deck's ppm resolution: bracket_min_span "
     "is a ROD-STEP quantity (1e-6) and a boron bracket that narrow is resolution, not a "
     "root, so bisection there would re-propose the point it is standing on forever")

# The rod arm's behaviour is unchanged: the searchType test stays in
# UpdateRodBracket so every existing caller keeps its existing predicate.
want(SCHEDULER, "void UpdateRodBracket(double k_residual) {", "Scheduler.h",
     "UpdateRodBracket must survive by name")
ROD = region(SCHEDULER, "void UpdateRodBracket(double k_residual) {", "\n    }", "UpdateRodBracket")
want(ROD, "if (searchType != SearchType::RODCRIT)", "Scheduler.h UpdateRodBracket",
     "the rod predicate must stay with the rod entry point, not migrate into the shared "
     "body where the boron arm would meet it")
want(ROD, "UpdateSearchBracket(k_residual);", "Scheduler.h UpdateRodBracket",
     "and it must delegate rather than keep a second copy of the bracket arithmetic")

# The Driver's three guarded sites.
want(DRIVER, "} else if (ctx.search_policy.boron_bracket) {", "Driver.h SolveLoop",
     "the boron bracket update must be an ELSE of the RODCRIT branch, so a rod search "
     "cannot take it twice and a boron search cannot take it unasked")
want(DRIVER, "ctx.search_policy.trialCap(schedule.max_search_iter)", "Driver.h SolveLoop",
     "the trial cap must go through trialCap, which is what makes an unset knob the "
     "deck's own limit")
want(DRIVER, "ctx.search_policy.stagedMargin(STAGED_SEARCH_MARGIN)", "Driver.h SolveLoop",
     "the staged margin must fall back to the built-in constant")
want(DRIVER, "if (ctx.search_policy.warm_boron) {", "Driver.h",
     "the warm-boron handover must be gated on its flag: with the knob unset the warm "
     "start must not even store the seed")

# The cap must exit through the deck limit's OWN path, so acceptance is
# unchanged by construction: SEARCH_EXHAUSTED lands on the deterministic
# best-fallback, which re-converges at PRODUCTION tolerance and publishes a
# search_exit_status that is not CONVERGED.
CAP = region(DRIVER, "if (has_search && !search_converged) {", "outer_timing::Scope propose_scope",
             "the trial cap")
# Comments stripped: this file's own prose about search_exit_status would
# otherwise trip the check below, which is a scan for a WRITER.
CAP_CODE = "\n".join(line.split("//")[0] for line in CAP.splitlines())
want(CAP_CODE, "exit_reason = SolveExit::SEARCH_EXHAUSTED;", "Driver.h",
     "the cap must take the deck limit's own exit; a private exit would be a second "
     "acceptance rule for the same event")
if "search_exit_status" in CAP_CODE:
    fail("the trial cap writes search_exit_status itself; the deterministic-acceptance "
         "block below the loop owns that field and a second writer is a second opinion "
         "about whether the statepoint converged")

# ---------------------------------------------------------------------------
# 3. THE ARM CAN BE NAMED.  Every knob is in trajectory::kArmEnv, which is what
#    folds it into the WP10.1 case key and prints it raw on the trajectory line.
# ---------------------------------------------------------------------------
ARM = region(DRIVER, "inline constexpr const char* kArmEnv[] = {", "};", "kArmEnv")
for knob in KNOBS:
    if f'"{knob}"' not in ARM:
        fail(f"{knob} moves the search's proposals but is not in trajectory::kArmEnv, so "
             "the case key does not fold it and a cached answer computed under another "
             "search policy could be served for it")

# ---------------------------------------------------------------------------
# 4. FIVE LEVERS, FIVE FLAGS.  Fusing two would make the 238 A/B unable to say
#    which one paid and would leave the doc's per-candidate revert conditions
#    with nothing to act on.
# ---------------------------------------------------------------------------
FIELDS = ("carry_slope", "warm_boron", "boron_bracket", "max_trials", "staged_margin")
for a in FIELDS:
    for b in FIELDS:
        if a == b:
            continue
        if re.search(r"policy\.%s\s*(&&|\|\|)\s*policy\.%s" % (a, b), SCHEDULER + DRIVER):
            fail(f"the {a} and {b} levers are combined in one condition; each must be "
                 "independently armable or the A/B cannot attribute the result")
# `any()` is the one place they are allowed to meet, and it exists only so the
# receipt can decide whether to print at all.
want(SCHEDULER, "return carry_slope || warm_boron || boron_bracket || max_trials > 0 ||",
     "Scheduler.h SearchPolicy::any",
     "any() is the receipt's gate and must cover every lever, or an arm can run with no "
     "receipt saying it did")

# ---------------------------------------------------------------------------
# 5. THE RECEIPT.  Printed only when armed; folded unconditionally so it exists
#    on the wall-timing arm, which runs with telemetry OFF.
# ---------------------------------------------------------------------------
want(DRIVER, "if (ctx.search_policy.any()) {", "Driver.h",
     "the SEARCH_POLICY receipt must be gated on the policy: with every knob unset this "
     "build's log has to be the log of a build without the feature")
RECEIPT = region(DRIVER, "if (ctx.search_policy.any()) {", "\n        }", "SEARCH_POLICY receipt")
for field in ("gate", "carry_slope", "warm_boron", "boron_bracket", "max_trials",
              "staged_margin", "trials", "proposals", "probe", "carry_secant", "extrap",
              "secant", "bisect", "search_outers", "outers"):
    if f'\\"{field}\\":' not in RECEIPT:
        fail(f"the SEARCH_POLICY receipt does not carry {field!r}; the adoption bar is "
             "stated in outers and the revert conditions in the classification, so both "
             "have to be on the line")
want(RECEIPT, 'sp.staged_margin > 0.0 ? "A2" : "N1"', "Driver.h SEARCH_POLICY receipt",
     "the gate word must be A2 exactly when a knob that relaxes a CONVERGENCE CRITERION "
     "is set -- the staged margin is the only one that does; the other four move the "
     "starting point and the trial sequence and leave the final acceptance test at "
     "production tolerance, which is N1")

# The ledger is folded beside warm_initial_outers, i.e. OUTSIDE `if (sp_telem)`.
LEDGER = region(DRIVER, "warm_initial_outers += ctx.telemetry.outers_by_cause",
                "// The BOC state, for a child.", "the search ledger fold")
for field in ("trials", "proposals", "refused", "probe", "carry", "extrap", "secant",
              "bisect", "outers"):
    want(LEDGER, f"sp_search.{field}", "Driver.h",
         "the run ledger must fold every classification bucket unconditionally; a lever "
         "whose before/after can only be read on the telemetry arm cannot be priced on "
         "the wall arm it is supposed to shorten")
if "sp_telem" in LEDGER:
    fail("the search ledger fold is gated on RASBERY_STATEPOINT_TELEMETRY; the wall arm "
         "runs with telemetry off and would then have no ledger at all")

# The per-statepoint receipt gained exactly one field, and the counter behind it
# is instrument only -- tools/test_statepoint_telemetry.py owns that scan; what
# is pinned here is that the field exists and is fed from the Schedule.
want(DRIVER, "ctx.telemetry.search_extrap     = schedule.search_n_extrap;", "Driver.h",
     "the extrapolation count must be carried from the Schedule that owns the "
     "classification, not re-derived at the receipt")

# ---------------------------------------------------------------------------
# 6. THE ONLY ARITHMETIC IN STAGE D, transcribed and negatively controlled.
#
#    carriedBoronSlope is a linear fit in EFPD through the two most recently
#    measured slopes, with three guards.  Each guard is a failure mode the doc
#    names for D1 ("보정이 틀리면 첫 trial이 더 멀어진다"), and a guard that had
#    been dropped would show up on 238 only as a worse first trial -- i.e. as
#    the thing the lever was supposed to fix, attributed to the lever's premise
#    rather than to a missing `if`.  So the transcription is run here.
# ---------------------------------------------------------------------------
CARRIED = region(SCHEDULER, "inline double carriedBoronSlope(", "\n}\n", "carriedBoronSlope")
for guard, why in (
        ("if (!std::isfinite(predicted))",
         "a non-finite prediction is the arithmetic saying two points cannot speak here"),
        ("if (predicted * carry.last_slope <= 0.0)",
         "a sign flip would send the first trial the wrong way outright"),
        ("if (!(ratio >= 0.5 && ratio <= 2.0))",
         "a magnitude more than a factor of two from the measured slope is an "
         "extrapolation past what two points support"),
        ("if (!(defpd > 0.0))",
         "two slopes measured at the same efpd have no trend, and dividing by that "
         "span is how a lever becomes a divide by zero"),
        ("if (!carry.has_prev || !carry.has_last)",
         "with fewer than two measured slopes there is nothing to extrapolate and the "
         "plain carry must stand"),
):
    want(CARRIED, guard, "Scheduler.h carriedBoronSlope", why)
if "used_extrapolation = false;" not in CARRIED or "used_extrapolation = true;" not in CARRIED:
    fail("carriedBoronSlope does not report whether it actually extrapolated; `the flag "
         "was on` and `the flag did something` are different runs and the A/B has to be "
         "able to tell them apart")


def carried_slope(has_prev, has_last, prev_slope, prev_efpd, last_slope, last_efpd,
                  memory_slope, efpd_now):
    """Transcription of Scheduler.h carriedBoronSlope.  Returns (slope, used)."""
    if not has_prev or not has_last:
        return memory_slope, False
    defpd = last_efpd - prev_efpd
    if not defpd > 0.0:
        return memory_slope, False
    trend = (last_slope - prev_slope) / defpd
    predicted = last_slope + trend * (efpd_now - last_efpd)
    import math
    if not math.isfinite(predicted):
        return memory_slope, False
    if predicted * last_slope <= 0.0:
        return memory_slope, False
    if last_slope == 0.0:
        return memory_slope, False
    ratio = abs(predicted) / abs(last_slope)
    if not (0.5 <= ratio <= 2.0):
        return memory_slope, False
    return predicted, True


# The lever doing its job: boron worth weakening steadily with burnup, so the
# statepoint 20 EFPD on starts from a slope that has been walked forward rather
# than from the one measured two statepoints ago.
value, used = carried_slope(True, True, -1.0e-4, 0.0, -0.9e-4, 100.0, -0.9e-4, 200.0)
if not used or abs(value - (-0.8e-4)) > 1e-12:
    fail(f"the slope extrapolation does not walk a measured trend forward: got {value!r}")
# Negative control A: the trend would more than double the slope -> refused.
value, used = carried_slope(True, True, -1.0e-4, 0.0, -2.0e-4, 1.0, -2.0e-4, 4.0)
if used:
    fail("negative control A: a prediction 2.5x the measured slope was accepted; the "
         "magnitude guard is not doing anything")
# ... and the boundary is INCLUSIVE, so the same fixture one EFPD earlier -- a
# prediction of exactly twice the measured slope -- is still taken.  Without
# this the guard could have been tightened to uselessness and control A would
# still pass.
value, used = carried_slope(True, True, -1.0e-4, 0.0, -2.0e-4, 1.0, -2.0e-4, 3.0)
if not used:
    fail("negative control A': the magnitude guard rejects a prediction exactly at its "
         "own boundary, so the band is narrower than the code says it is")
# Negative control B: the trend crosses zero -> refused, sign guard.
value, used = carried_slope(True, True, -1.0e-4, 0.0, -0.5e-4, 1.0, -0.5e-4, 4.0)
if used:
    fail("negative control B: a prediction that changed sign was accepted; the search "
         "would step the wrong way on its first trial")
# Negative control C: one measured slope is not a trend.
value, used = carried_slope(False, True, 0.0, 0.0, -1.0e-4, 100.0, -1.0e-4, 200.0)
if used or value != -1.0e-4:
    fail("negative control C: a single measured slope was extrapolated from")
# Negative control D: two slopes at the same efpd.
value, used = carried_slope(True, True, -1.0e-4, 50.0, -0.9e-4, 50.0, -0.9e-4, 60.0)
if used:
    fail("negative control D: a zero EFPD span was divided by")
# Negative control E: the flat case degrades to the plain carry, exactly.  This
# is the statepoint that took no secant step -- it pushes the same slope again.
value, used = carried_slope(True, True, -1.0e-4, 0.0, -1.0e-4, 100.0, -1.0e-4, 500.0)
if not used or value != -1.0e-4:
    fail("negative control E: a flat trend did not degrade to the plain carry")


def main() -> int:
    if problems:
        for problem in problems:
            print("search policy contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("search policy contract: PASS (%d knobs, 5 negative controls on the "
          "extrapolation, feature-off pinned at %d sites)" % (len(KNOBS), 11))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
