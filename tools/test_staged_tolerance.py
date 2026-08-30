#!/usr/bin/env python3
"""Contract: A2 staged tolerance never publishes a loosely-converged answer.

WHAT THIS PROTECTS.  RASBERY_STAGED_FLUX_TOL / RASBERY_STAGED_XE_TOL let the
solve converge to a multiplied tolerance while the statepoint is still moving.
That is only safe because of an invariant that lives in one branch and could be
deleted by a plausible-looking edit: the loose stage's "everything converged"
verdict is not an exit.  It sets `polishing`, restores the production
tolerances, re-arms the Xe cascade, and asks again -- and only the second
answer can end the solve.  Lose that and the code silently starts publishing
k_eff converged to 50 pcm instead of 1.

The second thing it protects is the default path.  Every tolerance the loop
reads is now a variable rather than a constant, so "the feature is off" has to
be a property of the source, not of a run: with no environment set, `polishing`
is true from construction and every variable is its production constant.

Source-level, like the other contract tests here: it reads Driver.h and pins the
expressions, because the failure mode is an edit, not a runtime state.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "src" / "Driver.h").read_text(errors="replace")

FAILED = []


def fail(msg):
    FAILED.append(msg)


def region(text, start, end, what):
    i = text.find(start)
    if i < 0:
        fail(f"cannot find the {what} region (looking for {start!r})")
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        fail(f"the {what} region is not closed (looking for {end!r})")
        return ""
    return text[i : j + len(end)]


SOLVE = region(SRC, "static void SolveLoop(", "\n    /// Where this run's restart_", "SolveLoop")

# ---------------------------------------------------------------- 1. defaults
# Both knobs default to 1.0 (off) and clamp anything below 1.0 back to 1.0.  A
# multiplier below one would TIGHTEN the loose stage past production, which is
# not a mode this design has -- the polish pass would then be the looser of the
# two and the invariant below would be inverted.
# WP10.3 MOVED THE THREE READS OUT OF SolveLoop.  They were `static const`
# lambdas here, which is a PER-PROCESS read inside a per-CASE solve: the first
# case an evaluator ran latched the convergence policy for every case after it.
# They now live on the case (src/CaseFidelity.h) and arrive on SolverContext, so
# this section pins two things instead of one -- that the defaults and the clamp
# still say what they said, wherever they live, and that SolveLoop takes its
# values FROM THE CASE and has not grown an environment read of its own.
FIDELITY = (ROOT / "src" / "CaseFidelity.h").read_text(errors="replace")
CONTRACT = (ROOT / "src" / "RunContract.h").read_text(errors="replace")

# The clamp and the default are RunContract.h's detail::stagedMultiplier, which
# processCaseFidelity() calls for both knobs.  Same two properties, one copy.
if "(m >= 1.0) ? m : 1.0" not in CONTRACT:
    fail("src/RunContract.h's stagedMultiplier does not clamp a below-1 multiplier back "
         "to 1.0; a multiplier below one would TIGHTEN the loose stage past production "
         "and invert the whole invariant below")
if '(value != nullptr) ? std::atof(value) : 1.0' not in CONTRACT:
    fail("src/RunContract.h's stagedMultiplier does not default to 1.0 (off) when the "
         "knob is unset")
for var, env in (("staged_flux_mult", "RASBERY_STAGED_FLUX_TOL"),
                 ("staged_xe_mult", "RASBERY_STAGED_XE_TOL")):
    if f'detail::stagedMultiplier("{env}")' not in FIDELITY:
        fail(f"src/CaseFidelity.h's processCaseFidelity() does not read {env} into {var}")
    if not re.search(rf"{var}\s*=\s*ctx\.fidelity\.{var};", SOLVE):
        fail(f"SolveLoop does not take {var} from the case; a per-process read inside a "
             f"per-case solve hands the first case's policy to every case after it")
if f'std::getenv("RASBERY_STAGED' in SOLVE:
    fail("SolveLoop reads a staged knob straight from the environment again")

if "const bool   staged_tol       = ctx.fidelity.staged();" not in SOLVE:
    fail("the feature gate is not the case's own `either multiplier is above 1`")
if "return staged_flux_mult > 1.0 || staged_xe_mult > 1.0;" not in FIDELITY:
    fail("CaseFidelity::staged() is not `either multiplier is above 1`")

# ------------------------------------------------- 2. the default path is off
# `polishing` starts TRUE when the feature is off, which is what makes every
# tolerance below the production one on the default path.
if "bool polishing = !staged_tol;" not in SOLVE:
    fail("polishing does not start at the production stage when the feature is off")

# ------------------------------------- 3. one resolution site, read by everyone
# The three effective tolerances are resolved once, at the top of the outer, so
# the device segment and the host ladder cannot be asked at different stages
# within one outer.
for line in (
    "const double keff_tol_now = polishing ? keff_tol : loose_keff_tol;",
    "const double flux_tol_now = polishing ? flux_tol : loose_flux_tol;",
    "const double xe_tol_now   = polishing ? XE_EQUILIBRIUM_TOLERANCE : loose_xe_tol;",
):
    if SOLVE.count(line) != 1:
        fail(f"the per-outer tolerance must be resolved in exactly one place: {line!r}")

# The convergence test and the device segment must both read the resolved pair.
# A site left on the raw constant would decide a different stage than its
# neighbour in the same outer.
if ("flux_converged = std::abs(prev_inner - eigv) < keff_tol_now && residual < flux_tol_now;"
        not in SOLVE):
    fail("the flux convergence test does not read the staged tolerances")
if "s.keff_tol       = keff_tol_now;" not in SOLVE or "s.flux_tol       = flux_tol_now;" not in SOLVE:
    fail("the device outer segment is handed the production tolerances, so the ON arm would "
         "converge to a different stage than the host ladder tests")

# Nothing inside the loop may still test the raw equilibrium tolerance for
# cascade progress -- that is the decision the staged value exists to move.
loop = region(SOLVE, "for (int iout = 0; iout < max_iter; ++iout) {",
              "\n        // Deterministic acceptance.", "outer loop")
for token in ("prev_xe_change >= XE_EQUILIBRIUM_TOLERANCE",
              "xe_change >= XE_EQUILIBRIUM_TOLERANCE"):
    if token in loop:
        fail(f"a cascade-progress test still reads the raw tolerance: {token!r}")

# --------------------------------------------- 4. the loose stage cannot exit
# THE central invariant.  Inside the all-converged branch, the polish transition
# has to come BEFORE the break, and the break has to be unreachable while
# `polishing` is false.
conv = region(loop, "if (search_converged && th_converged) {",
              "exit_reason = SolveExit::CONVERGED;", "all-converged branch")
if "if (!polishing) {" not in conv:
    fail("the all-converged branch can exit from the loose stage: no `if (!polishing)` guard "
         "before the CONVERGED break")
if "polishing   = true;" not in conv:
    fail("the polish transition does not latch `polishing`")
if "continue;" not in conv:
    fail("the polish transition does not go back around the loop, so the production tolerance "
         "is never actually applied to anything")

# The three things the polish pass must re-open, or it would publish state that
# was only ever settled against the loose flux.
if "prev_inner  = eigv + 1.0;" not in conv:
    fail("the polish transition does not poison prev_inner, so the next outer re-reads the "
         "loose iterate instead of re-driving the flux")
if "prev_xe_change = std::numeric_limits<double>::infinity();" not in conv:
    fail("the polish transition does not re-arm the Xe cascade, so the published inventory is "
         "the one equilibrated against the loose flux")
if "clean_iters = 0;" not in conv:
    fail("the polish transition does not clear the settling gate, so the search keeps the "
         "loose sample it just accepted")

# ------------------------------------------------------- 5. the relapse path
# Production tolerance disagreeing is not an error; it drops back to loose so
# the trial about to be committed is taken cheaply.  It must be counted, or an
# over-loose multiplier shows up only as an unexplained outer count.
if "if (staged_tol && polishing) {" not in loop or "++ctx.telemetry.staged_relapses;" not in loop:
    fail("a production-tolerance disagreement does not drop back to the loose stage and "
         "charge a relapse")

# ------------------------------------------------ 6. the search keeps a floor
# A secant search reading k_eff off a flux converged no better than its own
# tolerance samples noise.  The cap is what stops the loosening reaching the
# digits the search reads.
cap = region(SOLVE, "const double loose_keff_tol =", ";\n", "search tolerance cap")
if "search_tol / staged_search_margin" not in cap or "std::min" not in cap:
    fail("the loose keff tolerance is not capped below search_tol when a search is running")
if not re.search(r"constexpr double STAGED_SEARCH_MARGIN = [1-9]", SOLVE):
    fail("STAGED_SEARCH_MARGIN is missing or not a positive literal")
# WP9-D stage D (candidate D3) made the margin the sweepable thing it always
# had to be, and the ONE property that matters is that the knob can only
# REPLACE the built-in: unset, `stagedMargin` answers with the literal above and
# the divisor is the number this tree has always used.  A margin resolved any
# other way -- a second getenv here, an atof at the use site -- would be a
# second reading of the knob that could disagree with the receipt's.
margin = region(SOLVE, "const double staged_search_margin =", ";\n",
                "staged search margin")
if "ctx.search_policy.stagedMargin(STAGED_SEARCH_MARGIN)" not in margin:
    fail("the staged search margin is not resolved through SearchPolicy::stagedMargin "
         "with the built-in constant as its fallback")
if "getenv" in margin or "atof" in margin:
    fail("SolveLoop reads RASBERY_SEARCH_STAGED_MARGIN itself; the knob has exactly one "
         "reader (Scheduler.h processSearchPolicy) and everything else takes the value")
SCHEDULER_SRC = (ROOT / "src" / "Scheduler.h").read_text(encoding="utf-8-sig")
if "return staged_margin > 0.0 ? staged_margin : built_in;" not in SCHEDULER_SRC:
    fail("SearchPolicy::stagedMargin no longer falls back to the built-in margin, so an "
         "unset RASBERY_SEARCH_STAGED_MARGIN would change the loose tolerance")
if "p.staged_margin = (m >= 1.0) ? m : 0.0;" not in SCHEDULER_SRC:
    fail("RASBERY_SEARCH_STAGED_MARGIN is not clamped to off below 1.0; a margin under 1 "
         "would let the loose stage sample k_eff looser than the search tolerance, which "
         "is the noise the cap exists to keep out")

# ----------------------------------------- 7. the settle knob is subordinate
# RASBERY_STAGED_LOOSE_SETTLE may only skip the gate in the LOOSE stage.  With
# staging off `polishing` is true throughout, so the term is identically false
# and the gate is the original expression -- but only if the flag is ANDed with
# `!polishing` rather than standing alone.
gate = region(loop, "const bool staged_loose_settle = ctx.fidelity.loose_settle;",
              "clean_iters < SEARCH_SETTLE_ITERS) {", "settling gate")
# The knob's own reading moved to CaseFidelity.h with the other two, verbatim.
# It is pinned there, and pinned as PER CASE here.
if 'std::getenv("RASBERY_STAGED_LOOSE_SETTLE")' not in FIDELITY:
    fail("src/CaseFidelity.h does not read RASBERY_STAGED_LOOSE_SETTLE")
if "if (value == nullptr) return false;" not in FIDELITY:
    fail("the loose-settle knob does not default to off")
if 's == "0" || s == "off" || s == "OFF" || s == "false" ||' not in FIDELITY:
    fail("parseStagedLooseSettle no longer spells the truthiness test the way SolveLoop's "
         "own lambda did; the receipt and the solver would then read one knob two ways")
if "!(staged_loose_settle && !polishing) &&" not in gate:
    fail("the loose-settle knob is not gated on the LOOSE stage; it would skip the settling "
         "gate for the published sample too")

# --------------------------------------------- 8. production paths untouched
# The deterministic-acceptance fallback re-converges the best trial point and is
# outside the loop, i.e. always the published answer: it must use the production
# tolerances, never the staged ones.
tail = SOLVE[SOLVE.find("// Deterministic acceptance."):]
if "ReconvergeFlux(ctx, eigv, FALLBACK_RECONVERGE_ITER, keff_tol, flux_tol," not in tail:
    fail("the best-trial-point fallback no longer re-converges at the production tolerance")
if "keff_tol_now" in tail or "flux_tol_now" in tail:
    fail("the deterministic-acceptance fallback reads a staged tolerance; it publishes the "
         "answer and must not")

if FAILED:
    print("staged tolerance contract: FAIL")
    for m in FAILED:
        print(f"  - {m}")
    sys.exit(1)
print("staged tolerance contract: PASS")
