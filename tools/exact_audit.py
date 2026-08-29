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

WHAT THE AUDIT KEYS ON NOW (plan Sec 6.2).  Two fields, both computed by the
binary from the environment it actually ran under:

    policy                strict | A2 | L3coarse | feedback_limited
    acceptance_eligible   true only for `strict`

`result_mode` is reported beside them and is NEVER a reason to void a run.

`fidelity_declared` is the operator's RASBERY_PHYSICS_FIDELITY, reported raw.
It can only make the effective policy COARSER (src/RunContract.h), so it can
never be used to dress an A2 run up as strict -- but an audit that wants to
refuse declared coarseness can see it here.

USE.  `run_single_gpu_batch.py` / `run_multi_gpu_batch.py` should call
`audit_physics_mode(output, require=...)`; both harnesses share one function for
the same reason they share `check_run_receipts()` -- a difference between them
is invisible in every receipt either of them prints.
"""
from __future__ import annotations

import json
import re
from typing import Iterable

PHYSICS_MODE_RECEIPT = re.compile(r"\[RASBERY\]\[PHYSICS_MODE\]\s*(\{.*\})")

# Every policy the binary can report, coarsest last.  The order IS the ranking
# src/RunContract.h uses to combine the detected fidelity with a declared one.
POLICIES = ("strict", "A2", "L3coarse", "feedback_limited")

# What an ACCEPTANCE measurement may be.  A2 is deliberately absent: the plan
# (Sec 6.1/6.2) gives it its own acceptance flag and forbids mixing it with
# strict in one table.  Pass require={"strict", "A2"} to audit an A2 campaign.
ACCEPTANCE_POLICIES = frozenset({"strict"})

# The fields the audit needs.  A receipt without them is from a binary older
# than WP1 and cannot be audited on fidelity at all -- which is a refusal, not a
# pass, because the field that WOULD have voided it is the missing one.
REQUIRED_FIELDS = ("policy", "physics_fidelity", "acceptance_eligible",
                   "screening", "result_mode", "feedback_pass_limit")


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


def audit_physics_mode(output: str,
                       require: Iterable[str] = ACCEPTANCE_POLICIES) -> list[str]:
    """Problems with *output*'s physics-mode receipt.  Empty list means PASS.

    The audit is on FIDELITY.  `result_mode` is read and reported, never judged.
    """
    allowed = frozenset(require)
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
    if policy not in allowed:
        problems.append(
            "[RASBERY][PHYSICS_MODE] policy=%r but this audit accepts %s: the run "
            "used a different convergence/statepoint policy and is not comparable "
            "with the arms that did not" % (policy, ", ".join(sorted(allowed))))

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
    # GA feedback-pass limit changes the answer, and `strict` with a nonzero
    # limit is a receipt that has already contradicted itself.
    if receipt["feedback_pass_limit"] != 0 and policy != "feedback_limited":
        problems.append(
            "[RASBERY][PHYSICS_MODE] feedback_pass_limit=%r with policy=%r"
            % (receipt["feedback_pass_limit"], policy))

    return problems


__all__ = ["POLICIES", "ACCEPTANCE_POLICIES", "REQUIRED_FIELDS",
           "PHYSICS_MODE_RECEIPT", "parse_physics_mode", "audit_physics_mode"]
