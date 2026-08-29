#!/usr/bin/env python3
"""WP1(a)/(c): OUTPUT MODE AND COMPUTE FIDELITY ARE TWO DIFFERENT THINGS.

THE DEFECT THIS CLOSES (plan Sec 2.2 P0, Sec 6.2, Sec 10 row 1).  main.cpp used
to compute

    const bool screening = (ga_feedback_passes > 0) || light_result;

so `--result light` -- a switch that changes only what LEAVES the process --
made the run declare `physics_mode:ga_screen_feedback_limited`,
`screening:true`, and refuse to start without RASBERY_ALLOW_SCREENING.  The
campaign has measured full, pin-off and light to ONE trajectory digest
(814201df0583e1d2, GA evaluator plan Sec 2.1): the solve, the PPR and the
feedback loops are identical.  So the old receipt said the physics was
approximated when nothing about the physics had changed, and the harness worked
around it by EXPECTING the screening receipt for a light chunk -- which meant a
receipt claiming screening was accepted for a run that was not screening.

WHAT REPLACES IT.  Two independent axes, both in the receipt:

    result_mode        full | pin-off | light        what the case WRITES
    policy             strict | A2 | L3coarse | feedback_limited   how it SOLVES

`screening` and `acceptance_eligible` are functions of the POLICY alone.
`light + strict` is legal and acceptance-eligible; `full + L3coarse` is
screening whatever it writes.

WHAT WOULD BE SILENT IF THIS BROKE:

  1. An A2 arm reported as strict.  The staged-tolerance multipliers are read in
     TWO places now -- Driver.h's SolveLoop and RunContract.h -- and a receipt
     that re-derived them differently would disagree with the solver about what
     the run was asked for.  Both readings are pinned below.
  2. `light` classified as screening again.  One `||` restores it.
  3. An operator declaring a FINER fidelity than the environment configures.
     RASBERY_PHYSICS_FIDELITY may only make the effective policy COARSER.

Run:  python tools/test_result_fidelity_contract.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def strip_line_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


# --------------------------------------------------------------- the table ---
# src/RunContract.h holds ONE table and everything else reads it.  The test
# parses the table rather than the four accessors, because the accessors are
# generated from it and a mismatch between them is caught by the C++ compiler,
# while a wrong ROW is caught by nothing else.
EXPECTED_TRAITS = [
    # policy,            physics_fidelity,   screening, acceptance_eligible
    ("strict",           "full_exact",       "false",   "true"),
    ("A2",               "staged_a2",        "false",   "false"),
    ("L3coarse",         "coarse10",         "true",    "false"),
    ("feedback_limited", "feedback_limited", "true",    "false"),
]

TRAITS_ROW = re.compile(
    r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*"([A-Za-z0-9_]+)"\s*,\s*(true|false)\s*,\s*(true|false)\s*\}')


def parse_traits(source: str) -> list[tuple[str, str, str, str]]:
    """Rows of kFidelityTraits[], in declaration order."""
    start = source.find("kFidelityTraits[")
    if start < 0:
        return []
    end = source.find("};", start)
    if end < 0:
        return []
    return TRAITS_ROW.findall(source[start:end])


contract_path = ROOT / "src" / "RunContract.h"
if not contract_path.exists():
    failures.append("src/RunContract.h does not exist; WP1's fidelity axis has no home")
    contract = ""
else:
    contract = read("src/RunContract.h")

check("enum class PhysicsFidelity" in contract,
      "src/RunContract.h does not declare `enum class PhysicsFidelity`")
for member in ("FullExact", "StagedA2", "Coarse10State", "FeedbackLimited"):
    check(member in contract,
          f"src/RunContract.h: PhysicsFidelity is missing {member} (plan Sec 6.2)")

traits = parse_traits(strip_line_comments(contract))
check(traits == EXPECTED_TRAITS,
      "src/RunContract.h kFidelityTraits does not match the plan Sec 6.2 table.\n"
      f"    parsed:   {traits}\n"
      f"    expected: {EXPECTED_TRAITS}")

# NEGATIVE CONTROL for the parser itself: a mutated table must not match.
_mutant = ('inline constexpr FidelityTraits kFidelityTraits[4] = {\n'
           '    {"strict", "full_exact", false, true},\n'
           '    {"A2", "staged_a2", false, true},\n'          # <- A2 made acceptance-eligible
           '    {"L3coarse", "coarse10", true, false},\n'
           '    {"feedback_limited", "feedback_limited", true, false},\n'
           '};\n')
check(parse_traits(_mutant) != EXPECTED_TRAITS,
      "the kFidelityTraits parser accepts a table with A2 marked acceptance-eligible; "
      "it cannot detect the mutation it exists to detect")

# ------------------------------------------------- A2 detection must not drift -
# Driver.h decides staging with `staged_flux_mult > 1.0 || staged_xe_mult > 1.0`
# on multipliers clamped at 1.0.  RunContract.h reports it.  Two readings of one
# knob is exactly how a receipt starts lying, so both are pinned here.
driver = read("src/Driver.h")
check("const bool staged_tol = staged_flux_mult > 1.0 || staged_xe_mult > 1.0;" in driver,
      "src/Driver.h: the staged-tolerance predicate moved; src/RunContract.h's copy of it "
      "(and therefore the `policy` field) is now unverified")
for knob in ("RASBERY_STAGED_FLUX_TOL", "RASBERY_STAGED_XE_TOL"):
    check(knob in contract,
          f"src/RunContract.h does not read {knob}; an A2 arm would report policy=strict")
check("RASBERY_GA_FEEDBACK_PASSES" in contract,
      "src/RunContract.h does not read RASBERY_GA_FEEDBACK_PASSES")
check(re.search(r">\s*1\.0", contract) is not None,
      "src/RunContract.h does not compare a staged multiplier against 1.0; it cannot be "
      "reading the knob the way Driver.h does")

# A declared fidelity may only COARSEN.  The rank comparison is the mechanism.
check("RASBERY_PHYSICS_FIDELITY" in contract,
      "src/RunContract.h has no RASBERY_PHYSICS_FIDELITY: L3coarse is a DECK property "
      "(tools/make_screening_deck.py rewrites the burnup grid and sets no environment "
      "variable), so the binary cannot detect it and needs a declaration channel")
declared_region = contract[contract.find("RASBERY_PHYSICS_FIDELITY"):]
check(re.search(r"static_cast<int>\(\s*declared_fidelity\s*\)\s*>\s*static_cast<int>",
                declared_region) is not None,
      "src/RunContract.h does not take the COARSER of the detected and declared "
      "fidelities; a declaration that could refine one would let an A2 run be published "
      "as strict")

# --------------------------------------------------------------- main.cpp ----
main = read("src/main.cpp")
check('#include "RunContract.h"' in main, "src/main.cpp does not include RunContract.h")

guard_start = main.find("Exact-only hard contract")
receipt_at = main.find("[RASBERY][PHYSICS_MODE]")
check(guard_start >= 0 and receipt_at > guard_start,
      "src/main.cpp: could not locate the exact-only guard region")
guard_region = main[guard_start:receipt_at] if guard_start >= 0 else ""
# The comment block in main.cpp QUOTES the old line so the defect cannot come
# back by accident; scan the code, not the prose.
guard_code = strip_line_comments(guard_region)

# THE DEFECT, spelled out so it cannot come back by accident.
check(re.search(r"screening\s*=\s*\(?\s*ga_feedback_passes\s*>\s*0\s*\)?\s*\|\|\s*light_result",
                guard_code) is None,
      "src/main.cpp still computes `screening = (ga_feedback_passes > 0) || light_result`: "
      "a light OUTPUT still classifies the run as a screening SOLVE (plan Sec 6.2)")
_screening_decl = re.search(r"const bool screening[^;]*;", guard_code)
check(_screening_decl is not None and "light_result" not in _screening_decl.group(0),
      "src/main.cpp: `screening` is still a function of the result mode")
check(re.search(r"const bool screening\s*=\s*rasbery::fidelityIsScreening\(", guard_code)
      is not None,
      "src/main.cpp does not derive `screening` from the fidelity "
      "(`rasbery::fidelityIsScreening(fidelity)`)")

# The guard shape the exact-only contract already depends on must survive.
check("if (screening && !allow_screening)" in main,
      "src/main.cpp: the screening refusal lost its shape")
check("RASBERY_ALLOW_SCREENING" in main, "src/main.cpp: the opt-out vanished")
check("[RASBERY][EXACT_ONLY][FAIL]" in main, "src/main.cpp: the refusal banner vanished")

# The GA feedback limit still requires a light result: it is the one
# approximation whose receipt has always been coupled to the writer.
check("ga_feedback_passes > 0 && !light_result" in main,
      "src/main.cpp no longer requires a light result for a feedback-limited run")

# ------------------------------------------------------------- the receipt ---
receipt_region = main[receipt_at:main.find("std::endl;", receipt_at) + 40] if receipt_at >= 0 else ""
for field in ('\\"result_mode\\"', '\\"physics_fidelity\\"', '\\"policy\\"',
              '\\"acceptance_eligible\\"', '\\"requires_exact_rerun\\"',
              '\\"gpu_full\\"', '\\"screening\\"', '\\"feedback_pass_limit\\"',
              '\\"full_hdf5\\"', '\\"fidelity_declared\\"'):
    check(field in receipt_region,
          f"the [RASBERY][PHYSICS_MODE] receipt omits {field} (plan Sec 6.2)")

# `physics_mode` keeps its legacy vocabulary on purpose: every A2 arm measured
# so far prints `full_exact_nodal` and the 238 harness audits that string.
# Repointing it at the new policy words would void every stored A2 arm.
check('"full_exact_nodal"' in strip_line_comments(main),
      "the receipt no longer prints the legacy `full_exact_nodal` physics_mode; every "
      "stored 238 arm is audited against that string")
# A2 must keep printing it too: staging is not screening, and repointing
# `physics_mode` at the policy words would void every stored A2 manifest.
check(re.search(r"StagedA2\S*\s*\)\s*\?\s*\"ga_screen", main) is None,
      "an A2 run would print a `ga_screen_*` physics_mode; A2 is not screening and every "
      "A2 arm on disk was measured against `full_exact_nodal`")

# ------------------------------------------------------ the fidelity audit ---
spec = importlib.util.spec_from_file_location("rasbery_exact_audit",
                                              ROOT / "tools" / "exact_audit.py")
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def receipt_line(**fields: object) -> str:
    import json
    base = {
        "physics_mode": "full_exact_nodal",
        "screening": False,
        "feedback_pass_limit": 0,
        "full_hdf5": True,
        "physics_fidelity": "full_exact",
        "policy": "strict",
        "acceptance_eligible": True,
        "requires_exact_rerun": False,
        "result_mode": "full",
        "fidelity_declared": None,
        "gpu_full": False,
    }
    base.update(fields)
    return "[RASBERY][PHYSICS_MODE] " + json.dumps(base) + "\n"


# THE ACCEPTANCE ROW THE DEFECT USED TO REJECT.
check(audit.audit_physics_mode(receipt_line(result_mode="light", full_hdf5=False)) == [],
      "the fidelity audit still voids a strict run because it wrote scalar output")
check(audit.audit_physics_mode(receipt_line(result_mode="pin-off")) == [],
      "the fidelity audit voids a pin-off strict run")
check(audit.audit_physics_mode(receipt_line(result_mode="mixed")) == [],
      "the fidelity audit voids a mixed-output strict wave")

# NEGATIVE CONTROLS: every one of these must be refused.
for label, bad in (
    ("no receipt", "nothing here\n"),
    ("unparsable", "[RASBERY][PHYSICS_MODE] {not json}\n"),
    ("A2 as acceptance", receipt_line(policy="A2", physics_fidelity="staged_a2",
                                      acceptance_eligible=False)),
    ("coarse", receipt_line(policy="L3coarse", physics_fidelity="coarse10",
                            screening=True, acceptance_eligible=False,
                            requires_exact_rerun=True)),
    ("feedback-limited", receipt_line(policy="feedback_limited", feedback_pass_limit=2,
                                      physics_fidelity="feedback_limited",
                                      screening=True, acceptance_eligible=False,
                                      physics_mode="ga_screen_feedback_limited")),
    ("strict with a feedback limit", receipt_line(feedback_pass_limit=3)),
    ("policy/acceptance disagree", receipt_line(acceptance_eligible=False)),
    ("policy/screening disagree", receipt_line(screening=True)),
    ("unknown policy", receipt_line(policy="fast")),
    ("pre-WP1 binary", '[RASBERY][PHYSICS_MODE] {"physics_mode":"full_exact_nodal",'
                       '"screening":false,"feedback_pass_limit":0,"full_hdf5":true}\n'),
):
    check(audit.audit_physics_mode(bad) != [],
          f"the fidelity audit accepted a run it must void: {label}")

# An A2 campaign audits itself by widening `require`, not by relaxing a field.
check(audit.audit_physics_mode(
          receipt_line(policy="A2", physics_fidelity="staged_a2",
                       acceptance_eligible=False),
          require={"strict", "A2"}) == [],
      "the fidelity audit cannot be pointed at an A2 arm")

if failures:
    for problem in failures:
        print("result/fidelity contract: FAIL " + problem, file=sys.stderr)
    raise SystemExit(1)
print("result/fidelity contract: PASS")
