#!/usr/bin/env python3
"""The exact-only audit, keyed on FIDELITY -- not on what the run wrote.

WHY THIS MODULE EXISTS.  Until WP1 the campaign's acceptance audit read
`full_hdf5` and `physics_mode` out of `[RASBERY][PHYSICS_MODE]` and voided any
run whose output was scalar-only.  `--result light` is an OUTPUT-shape switch:
the same solve, the same PPR, the same trajectory digest (814201df0583e1d2,
GA evaluator plan Sec 2.1).  Auditing it against `full_hdf5:true` failed every
light arm on a field that says nothing about the physics, and the workaround --
expecting the SCREENING receipt for a light chunk -- accepted
`physics_mode:ga_screen_feedback_limited` for a run that was not screening at
all.  Both halves of that are wrong, and the wrongness runs in both directions:
a strict/light acceptance run was rejected, and a receipt that claimed screening
was accepted.

WHAT THE AUDIT KEYS ON (plan Sec 6.2).  Two fields, both computed by the
binary from the environment it actually ran under:

    policy                strict | A2 | L3coarse | feedback_limited
    acceptance_eligible   true only for `strict`

`result_mode` is reported beside them and is NEVER a reason to void a run.

DECLARED vs SOLVED, AND WHY THE AUDIT IS AN EQUALITY (the WP4 defect).  The
audit used to take `require=` -- a SET of acceptable policies, defaulting to
`{"strict"}`.  But the production batch environment IS the A2 staged-tolerance
arm: `RASBERY_STAGED_FLUX_TOL=50 / _XE_TOL=1000 / _LOOSE_SETTLE=1` have been in
`run_single_gpu_batch.DEFAULT_ENV` since 7099e54, so every child prints
`policy:'A2'`, every audit compared it against `strict`, and the run was voided.
Measured on 238: rc=3 on a 12 x M6 wave, and the WP4 tuner disqualified ALL SIX
candidates (rc=2, no winner) with every wave at rc=0 / dup=0.  The harness was
not catching an approximation; it was refusing the arm it had itself declared.

So the audit now takes the fidelity the operator DECLARED -- one word, derived
from the resolved child environment unless `--fidelity` names it -- and passes
iff the receipt's policy is THAT word.  Both directions are failures on purpose:

  * an A2 receipt under a strict declaration is the original defect class -- an
    approximation walking into an acceptance table;
  * a STRICT receipt under an A2 declaration is also a mismatch.  The operator
    asked for A2, the run did something else, and a number filed in the A2
    column that was measured under a different convergence policy is exactly
    the mixing plan Sec 6.2 forbids ("never mix strict and A2 in one table").
    Silently accepting the "better" fidelity leaves that table wrong with no
    line anywhere saying so.

`fidelity_declared` in the receipt is the binary's own view of
RASBERY_PHYSICS_FIDELITY, reported raw.  It can only make the effective policy
COARSER (src/RunContract.h), so it can never dress an A2 run up as strict.

USE.  `run_single_gpu_batch.py` / `run_multi_gpu_batch.py` call
`audit_physics_mode(output, declared)`; both harnesses share one function for
the same reason they share `check_run_receipts()` -- a difference between them
is invisible in every receipt either of them prints.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Mapping

PHYSICS_MODE_RECEIPT = re.compile(r"\[RASBERY\]\[PHYSICS_MODE\]\s*(\{.*\})")
# WP10.3.  The PER-CASE receipts.  `[RASBERY][CASE]` is printed by the Driver
# itself (one per case, even in a one-shot run); `[EVALUATOR][CASE]` is the
# evaluator's line for the same case, carrying the same five fidelity fields
# plus the request's own `key`.  Either is enough to audit a case; the
# evaluator's is what a dispatcher has.
CASE_RECEIPT = re.compile(r"\[RASBERY\]\[CASE\]\s*(\{.*\})")
EVALUATOR_CASE_RECEIPT = re.compile(r"\[RASBERY\]\[EVALUATOR\]\[CASE\]\s*(\{.*\})")

# Every policy the binary can report, coarsest last.  The order IS the ranking
# src/RunContract.h uses to combine the detected fidelity with a declared one.
POLICIES = ("strict", "A2", "L3coarse", "feedback_limited")

# What `--fidelity` may say.  `feedback_limited` is deliberately absent: it is
# not a word an operator types, it is what RASBERY_GA_FEEDBACK_PASSES > 0 makes
# the run, and derive_declared_fidelity() returns it when that is set.
DECLARABLE_FIDELITIES = ("strict", "A2", "L3coarse")

# The two knobs src/RunContract.h reads as multipliers.  Above 1.0 -> A2.
STAGED_TOLERANCE_KEYS = ("RASBERY_STAGED_FLUX_TOL", "RASBERY_STAGED_XE_TOL")
# RASBERY_STAGED_LOOSE_SETTLE is a staged knob but NOT a fidelity one: Driver.h
# :3886 only consults it inside a LOOSE stage, and with a single stage
# `polishing` is true throughout ("It is inert unless staging is on",
# Driver.h:3882), so the binary reports `strict` when it is the only staged key
# set.  Deriving A2 from it alone would declare a fidelity the run cannot print
# and would fail such a run on a mismatch -- a guard stricter than the binary,
# which is the shape of the defect this whole module exists to undo.
STAGED_KEYS = STAGED_TOLERANCE_KEYS + ("RASBERY_STAGED_LOOSE_SETTLE",)
# Everything that can move a child off `strict`.  `--strict` clears all of them
# from the child environment, including any the operator's shell exported.
NON_STRICT_ENV_KEYS = STAGED_KEYS + ("RASBERY_GA_FEEDBACK_PASSES",
                                     "RASBERY_PHYSICS_FIDELITY")

# The fields the audit needs.  A receipt without them is from a binary older
# than WP1 and cannot be audited on fidelity at all -- which is a refusal, not a
# pass, because the field that WOULD have voided it is the missing one.
REQUIRED_FIELDS = ("policy", "physics_fidelity", "acceptance_eligible",
                   "screening", "result_mode", "feedback_pass_limit")


def _staged_multiplier(env: Mapping[str, str], name: str) -> float:
    """`detail::stagedMultiplier` (src/RunContract.h:120), in Python.

    The same clamp: a multiplier below 1 would TIGHTEN the tolerance and reads
    as 1.0, and an unparseable value reads as 1.0 the way std::atof() returns 0
    and the clamp lifts it.
    """
    value = env.get(name)
    if value is None or value == "":
        return 1.0
    try:
        multiplier = float(value)
    except ValueError:
        return 1.0
    return multiplier if multiplier >= 1.0 else 1.0


def derive_declared_fidelity(env: Mapping[str, str]) -> str:
    """The fidelity *env* configures, by the binary's own rule.

    This is `detectedPhysicsFidelity()` combined with `effectivePhysicsFidelity()`
    (src/RunContract.h:130-165), and it has to STAY that rule: the harness
    declares what it expects the child to print, so a derivation that differs
    from the child's is a guaranteed mismatch rather than a guard.

      RASBERY_GA_FEEDBACK_PASSES > 0        -> feedback_limited
      _STAGED_FLUX_TOL / _XE_TOL   > 1.0    -> A2
      otherwise                             -> strict

    and then RASBERY_PHYSICS_FIDELITY may make it COARSER, never finer.

    Pass the FINAL child environment (inherited + profile + --set - --set-unset),
    not DEFAULT_ENV: an exported RASBERY_GA_FEEDBACK_PASSES the harness never
    set still reaches the child and still decides what it prints.
    """
    passes = env.get("RASBERY_GA_FEEDBACK_PASSES") or ""
    try:
        feedback = int(passes)
    except ValueError:
        feedback = 0
    if feedback > 0:
        detected = "feedback_limited"
    elif any(_staged_multiplier(env, key) > 1.0 for key in STAGED_TOLERANCE_KEYS):
        detected = "A2"
    else:
        detected = "strict"

    declared = (env.get("RASBERY_PHYSICS_FIDELITY") or "").strip()
    if declared in POLICIES and POLICIES.index(declared) > POLICIES.index(detected):
        return declared
    return detected


def parse_physics_mode(output: str) -> tuple[dict | None, str | None]:
    """The LAST [RASBERY][PHYSICS_MODE] receipt in *output*, or (None, why)."""
    receipt = None
    for match in PHYSICS_MODE_RECEIPT.finditer(output):
        try:
            receipt = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None, "could not parse the [RASBERY][PHYSICS_MODE] receipt: %s" % (
                match.group(1))
    if receipt is None:
        return None, ("no [RASBERY][PHYSICS_MODE] receipt in the run output: the "
                      "exact-only audit cannot be applied and the run is void")
    return receipt, None


def receipt_policy(output: str) -> str | None:
    """The policy word the run PRINTED, or None when it printed no receipt.

    Reporting only -- the dispatcher records it per case so a throughput table
    can be LABELLED with the fidelity it was measured at.  Judging is
    audit_physics_mode()'s job.
    """
    receipt, _why = parse_physics_mode(output)
    if receipt is None:
        return None
    policy = receipt.get("policy")
    return policy if isinstance(policy, str) else None


def audit_physics_mode(output: str, declared: str = "strict") -> list[str]:
    """Problems with *output*'s physics-mode receipt.  Empty list means PASS.

    The audit is on FIDELITY, and it is an EQUALITY against what the operator
    DECLARED: it verifies the run solved at the fidelity that was asked for,
    never that it solved at some fidelity in an acceptable set.  `result_mode`
    is read and reported, never judged.
    """
    if declared not in POLICIES:
        return ["the declared fidelity %r is not one of %s"
                % (declared, ", ".join(POLICIES))]
    receipt, why = parse_physics_mode(output)
    if receipt is None:
        return [why or "no receipt"]

    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in receipt:
            problems.append(
                "[RASBERY][PHYSICS_MODE] is missing %r: this binary predates the "
                "WP1 result/fidelity split and cannot be audited on fidelity" % field)
    if problems:
        return problems

    policy = receipt["policy"]
    if policy not in POLICIES:
        problems.append("[RASBERY][PHYSICS_MODE] policy=%r is not one of %s"
                        % (policy, ", ".join(POLICIES)))
        return problems
    if policy != declared:
        problems.append(
            "[RASBERY][PHYSICS_MODE] policy=%r but this run DECLARED fidelity %r: the "
            "run did not solve at the fidelity it was asked for, so its throughput "
            "belongs in neither column (plan Sec 6.2: never mix strict and A2 in one "
            "table). Declare what is actually running with --fidelity, or change the "
            "environment that decides it (%s)."
            % (policy, declared, ", ".join(STAGED_KEYS)))

    # Internal consistency.  The binary derives all three from one table
    # (src/RunContract.h kFidelityTraits); a receipt where they disagree is a
    # receipt from a build whose table was edited on one side only.
    if receipt["acceptance_eligible"] is not (policy == "strict"):
        problems.append(
            "[RASBERY][PHYSICS_MODE] acceptance_eligible=%r contradicts policy=%r"
            % (receipt["acceptance_eligible"], policy))
    if receipt["screening"] is not (policy in ("L3coarse", "feedback_limited")):
        problems.append(
            "[RASBERY][PHYSICS_MODE] screening=%r contradicts policy=%r"
            % (receipt["screening"], policy))

    # The one approximation that is never excused by any policy word: a nonzero
    # GA feedback-pass limit changes the answer, and any policy but
    # `feedback_limited` carrying one is a receipt that has already contradicted
    # itself.
    if receipt["feedback_pass_limit"] != 0 and policy != "feedback_limited":
        problems.append(
            "[RASBERY][PHYSICS_MODE] feedback_pass_limit=%r with policy=%r"
            % (receipt["feedback_pass_limit"], policy))

    return problems


# ---------------------------------------------------------------------------
# WP10.3 -- THE PER-CASE AUDIT
# ---------------------------------------------------------------------------
#
# WHY THE PROCESS RECEIPT STOPPED BEING ENOUGH.  audit_physics_mode() above
# reads ONE [PHYSICS_MODE] line and judges the whole run by it, which was exact
# while fidelity was a process property.  It no longer is: src/CaseFidelity.h
# makes the staged tolerances and the burnup grid per case, so one evaluator
# process can answer a screening population and a promoted elite in one wave,
# and its single [PHYSICS_MODE] line describes NEITHER of them -- it describes
# the default they were resolved against.
#
# An audit that kept reading only that line would pass a wave in which every
# case ran coarse under a `strict` process default, which is the original defect
# with more decimal places.  So the per-case receipts carry their own five
# fields and this is what checks them, one case at a time, against what that
# case was DECLARED as.

#: What a per-case receipt must carry to be auditable at all.  A receipt missing
#: one is from a binary older than WP10.3, and that is a REFUSAL rather than a
#: pass: the field that would have voided the case is the missing one.
CASE_REQUIRED_FIELDS = ("policy", "physics_fidelity", "statepoint_grid",
                        "acceptance_eligible")


def parse_case_receipts(output: str) -> list[dict]:
    """Every per-case receipt in *output*, evaluator lines preferred.

    The two tags describe the same case and carry the same fidelity fields; a
    log from a dispatcher has the evaluator's, a log from a one-shot run has the
    Driver's.  Taking both and deduplicating on nothing is deliberate: a wave
    that printed twelve evaluator lines and twelve driver lines has twenty-four
    statements about fidelity that all have to hold.
    """
    receipts: list[dict] = []
    for pattern in (EVALUATOR_CASE_RECEIPT, CASE_RECEIPT):
        for match in pattern.finditer(output):
            try:
                value = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                receipts.append(value)
    return receipts


def _case_label(receipt: Mapping) -> str:
    for field in ("key", "case_key", "output", "deck"):
        value = receipt.get(field)
        if isinstance(value, str) and value:
            return value
    return "<unnamed case>"


def audit_case_fidelity(output: str, declared: "str | Mapping[str, str]" = "strict",
                        *, require_any: bool = True) -> list[str]:
    """Problems with *output*'s PER-CASE fidelity receipts.  Empty means PASS.

    *declared* is either one word applied to every case, or a mapping from a
    case's `key` / `case_key` / `output` / `deck` to the word that case was
    requested at -- which is what a mixed-fidelity wave needs, because in one
    such wave "what was declared" has as many answers as there are cases.

    A case whose receipt is absent from the mapping is audited against the
    mapping's `"*"` entry if it has one, and is otherwise REPORTED: an
    unaccounted case is a case nobody declared, and running one is exactly how a
    generation acquires a result whose provenance is a guess.
    """
    receipts = parse_case_receipts(output)
    if not receipts:
        return (["no [RASBERY][CASE] or [RASBERY][EVALUATOR][CASE] receipt in the run "
                 "output: per-case fidelity cannot be audited and the wave is void"]
                if require_any else [])

    problems: list[str] = []
    for receipt in receipts:
        label = _case_label(receipt)
        # A case that failed before it folded a receipt prints nulls, and that
        # is not a fidelity violation -- it is a failed case, which the status
        # field already reports and the dispatcher already counts.
        if receipt.get("status") == "failed" and receipt.get("policy") is None:
            continue
        missing = [f for f in CASE_REQUIRED_FIELDS if f not in receipt]
        if missing:
            problems.append(
                "%s: the per-case receipt is missing %s -- this binary predates WP10.3 "
                "and its cases cannot be audited on fidelity individually"
                % (label, ", ".join(repr(f) for f in missing)))
            continue

        policy = receipt["policy"]
        if policy not in POLICIES:
            problems.append("%s: policy=%r is not one of %s"
                            % (label, policy, ", ".join(POLICIES)))
            continue

        if isinstance(declared, str):
            want = declared
        else:
            want = None
            for field in ("key", "case_key", "output", "deck"):
                value = receipt.get(field)
                if isinstance(value, str) and value in declared:
                    want = declared[value]
                    break
            if want is None:
                want = declared.get("*")
            if want is None:
                problems.append(
                    "%s: ran at policy=%r and no fidelity was declared for it. In a "
                    "mixed-fidelity wave every case has to be declared, or its number "
                    "has no column to go in." % (label, policy))
                continue
        if want not in POLICIES:
            problems.append("%s: the declared fidelity %r is not one of %s"
                            % (label, want, ", ".join(POLICIES)))
            continue
        if policy != want:
            coarser = POLICIES.index(policy) > POLICIES.index(want)
            problems.append(
                "%s: policy=%r but this case DECLARED %r -- it solved %s than it was "
                "asked to. %s"
                % (label, policy, want, "COARSER" if coarser else "FINER",
                   "An approximation in an acceptance table is the defect the exact-only "
                   "contract exists for." if coarser else
                   "A strict number filed in the A2 column is the mixing plan Sec 6.2 "
                   "forbids; it is wrong in the other direction, not right."))

        # Internal consistency, per case, against the same one table the binary
        # derives all of it from (src/RunContract.h kFidelityTraits).
        if receipt["acceptance_eligible"] is not (policy == "strict"):
            problems.append("%s: acceptance_eligible=%r contradicts policy=%r"
                            % (label, receipt["acceptance_eligible"], policy))
        grid = receipt["statepoint_grid"]
        if grid not in (None, "", "full") and policy != "L3coarse":
            problems.append(
                "%s: statepoint_grid=%r with policy=%r. A rewritten burnup grid IS "
                "L3coarse (src/StatepointGrid.h); a case that coarsened its deck and "
                "reported anything else is a screening cost wearing an acceptance word."
                % (label, grid, policy))
        if grid in (None, "", "full") and policy == "L3coarse":
            problems.append(
                "%s: policy='L3coarse' with statepoint_grid=%r -- nothing was coarsened, "
                "so the run paid full cost and filed it in the screening lane."
                % (label, grid))
        # A promotion is a strict re-run BY DEFINITION (plan Sec 6.2 lane 3).
        # One that is not strict is a promotion that promoted nothing.
        promoted_from = receipt.get("promoted_from")
        if isinstance(promoted_from, str) and promoted_from and policy != "strict":
            problems.append(
                "%s: promoted_from=%r but policy=%r. A promotion exists to re-run a "
                "screened candidate at STRICT; one that lands at %s leaves the elite "
                "unverified while looking verified."
                % (label, promoted_from, policy, policy))
    return problems


def promotion_links(output: str) -> dict[str, str]:
    """`{promoted case_key: the screening case_key it replaces}` from *output*.

    The link a `promote` request stamps, read back.  What it is FOR: answering
    "was this elite actually re-run at strict" out of the receipts, without the
    controller having to remember which requests it sent.
    """
    links: dict[str, str] = {}
    for receipt in parse_case_receipts(output):
        parent = receipt.get("promoted_from")
        child = receipt.get("case_key")
        if isinstance(parent, str) and parent and isinstance(child, str) and child:
            links[child] = parent
    return links


def strip_non_strict(env: dict[str, str],
                     keys: Iterable[str] = NON_STRICT_ENV_KEYS) -> None:
    """Remove, in place, every key that could move a child off `strict`."""
    for key in keys:
        env.pop(key, None)


__all__ = ["POLICIES", "DECLARABLE_FIDELITIES", "REQUIRED_FIELDS",
           "CASE_REQUIRED_FIELDS",
           "STAGED_TOLERANCE_KEYS", "STAGED_KEYS", "NON_STRICT_ENV_KEYS",
           "PHYSICS_MODE_RECEIPT", "CASE_RECEIPT", "EVALUATOR_CASE_RECEIPT",
           "parse_physics_mode", "receipt_policy", "parse_case_receipts",
           "derive_declared_fidelity", "audit_physics_mode", "audit_case_fidelity",
           "promotion_links", "strip_non_strict"]
