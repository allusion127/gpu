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


def strip_non_strict(env: dict[str, str],
                     keys: Iterable[str] = NON_STRICT_ENV_KEYS) -> None:
    """Remove, in place, every key that could move a child off `strict`."""
    for key in keys:
        env.pop(key, None)


__all__ = ["POLICIES", "DECLARABLE_FIDELITIES", "REQUIRED_FIELDS",
           "STAGED_TOLERANCE_KEYS", "STAGED_KEYS", "NON_STRICT_ENV_KEYS",
           "PHYSICS_MODE_RECEIPT", "parse_physics_mode", "receipt_policy",
           "derive_declared_fidelity", "audit_physics_mode", "strip_non_strict"]
