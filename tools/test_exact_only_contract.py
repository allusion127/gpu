#!/usr/bin/env python3
"""Exact-only hard contract (plan Rev.4 Sec 2).

Static side: main() must REFUSE to start when a screening approximation is
enabled, and must emit the machine-readable physics-mode receipt.
Behavioural side: the benchmark launcher must void a run whose receipt is
missing or is not full-exact.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

problems: list[str] = []

main_cpp = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8")

# The refusal itself: both approximations checked, no warn-and-continue, one
# explicit opt-out, and a nonzero exit.
required_main = (
    "[RASBERY][EXACT_ONLY][FAIL]",
    "RASBERY_ALLOW_SCREENING",
    "rasbery::BatchLightResult::FeedbackPasses()",
    "rasbery::BatchLightResult::Enabled()",
    "const bool screening",
    "if (screening && !allow_screening)",
    "return 2;",
    # The receipt, field by field, exactly as Sec 2.2 specifies it.
    "[RASBERY][PHYSICS_MODE]",
    '\\"physics_mode\\":\\"',
    "full_exact_nodal",
    '\\"screening\\":',
    '\\"feedback_pass_limit\\":',
    '\\"full_hdf5\\":',
)
problems += [f"src/main.cpp: missing {token!r}" for token in required_main
             if token not in main_cpp]

# The refusal must come before any deck runs: the PHYSICS_MODE receipt is
# emitted before the batch branch and before the serial deck loop.
receipt_at = main_cpp.find("[RASBERY][PHYSICS_MODE]")
gate_at = main_cpp.find("if (screening && !allow_screening)")
# The batch branch is now selected by `batch_execution`, the predicate main()
# computes once at startup and latches as the execution mode before any receipt
# (adoption 2026-08-27, mode-dependent RASBERY_XE_ANDERSON default). Same
# branch, one anchor.
batch_at = main_cpp.find("if (batch_execution) {")
serial_at = main_cpp.find("rasbery::Driver driver(rasbery_input_path.string()")
if -1 in (receipt_at, gate_at, batch_at, serial_at):
    problems.append("src/main.cpp: could not locate the startup ordering anchors")
else:
    if not gate_at < receipt_at < batch_at:
        problems.append(
            "src/main.cpp: the exact-only gate and PHYSICS_MODE receipt must both precede "
            "the batch branch"
        )
    if receipt_at > serial_at:
        problems.append(
            "src/main.cpp: the PHYSICS_MODE receipt must precede the serial deck loop"
        )

# The launcher has to enforce the receipt, not merely print it.
SCRIPT = ROOT / "tools" / "run_single_gpu_batch.py"
spec = importlib.util.spec_from_file_location("run_single_gpu_batch_exact", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# The audit is keyed on FIDELITY since WP1 (review R5): `full_hdf5` and
# `result_mode` are reported and never judged, because `--result light` is an
# output-shape switch that leaves the solve, the PPR and the trajectory digest
# untouched.  What still voids a run is the POLICY.
import json  # noqa: E402


def receipt(**fields: object) -> str:
    base = {"physics_mode": "full_exact_nodal", "screening": False,
            "feedback_pass_limit": 0, "full_hdf5": True,
            "physics_fidelity": "full_exact", "policy": "strict",
            "acceptance_eligible": True, "requires_exact_rerun": False,
            "result_mode": "full", "fidelity_declared": None, "gpu_full": False}
    base.update(fields)
    return "[RASBERY][PHYSICS_MODE] " + json.dumps(base) + "\n"


EXACT = receipt()

if module.audit_physics_mode(EXACT):
    problems.append("run_single_gpu_batch: rejected a valid full-exact receipt")
if module.audit_physics_mode(receipt(result_mode="light", full_hdf5=False)):
    problems.append(
        "run_single_gpu_batch: voided a strict run because it wrote scalar-only output; "
        "the acceptance audit is keyed on fidelity, not on what the case writes")
if not module.audit_physics_mode("nothing here\n"):
    problems.append("run_single_gpu_batch: accepted a run with no PHYSICS_MODE receipt")
for bad in (
    # feedback-limited: the one approximation that changes the answer.
    receipt(policy="feedback_limited", physics_fidelity="feedback_limited",
            physics_mode="ga_screen_feedback_limited", screening=True,
            acceptance_eligible=False, feedback_pass_limit=2,
            result_mode="light", full_hdf5=False),
    # strict claimed WITH a nonzero feedback limit: a self-contradicting receipt.
    receipt(feedback_pass_limit=3),
    # coarse statepoints, writing the full HDF5: output shape must not rescue it.
    receipt(policy="L3coarse", physics_fidelity="coarse10", screening=True,
            acceptance_eligible=False, requires_exact_rerun=True),
    # A pre-WP1 binary: it cannot be audited on fidelity, so it is void.
    '[RASBERY][PHYSICS_MODE] {"physics_mode":"full_exact_nodal","screening":false,'
    '"feedback_pass_limit":0,"full_hdf5":true}\n',
    '[RASBERY][PHYSICS_MODE] {"physics_mode":"full_exact_nodal"}\n',
    '[RASBERY][PHYSICS_MODE] {not json}\n',
):
    if not module.audit_physics_mode(bad):
        problems.append("run_single_gpu_batch: accepted a non-exact receipt: " + bad.strip())

# A missing receipt must reach the exit-3 path, not just print.
plan = module.LaunchPlan(batch_width=4, jobs=4, visible_cpus=4, host_workers=4,
                         worker_policy="legacy", gpu="0")
audited = module.check_run_receipts(
    '[RASBERY][BATCH_HOST] {"jobs":4,"arena_width":4,"host_threads":4}\n', plan)
if not any("PHYSICS_MODE" in problem for problem in audited):
    problems.append(
        "run_single_gpu_batch: check_run_receipts does not audit the physics-mode receipt")

if problems:
    for problem in problems:
        print("exact-only contract: FAIL " + problem, file=sys.stderr)
    raise SystemExit(1)
print("exact-only contract: PASS")
