#!/usr/bin/env python3
"""The harnesses must resolve the 238 reference environment, key for key.

WHY THIS TEST EXISTS.  On the 238 host the raw production line

    taskset --cpu-list 0-23 RASBERY --rasi ... --raso ... --batch-mode 64

measured 582 cases/hour.  The dispatcher, on the same host, the same binary,
the same decks and the same exported physics environment, measured 115.6 --
`width_fill` 0.03.  Nothing failed.  No receipt was wrong.  rc was 0.  The
whole 5.0x was in what the harness put into the child's environment and in one
integer it computed for RASBERY_BATCH_HOST_THREADS, and three L5 sweeps
(103-116 c/h) were reported as data before anybody diffed the two envs.

That is the defect class this file closes: an environment difference between a
harness and the line it claims to reproduce is invisible in every receipt
either of them prints, and presents only as a throughput figure that looks like
physics.  So the reference env is RECORDED
(test/reference/batch_reference_env_238.json) and the harness's resolved env is
compared against it, key for key, with the negative controls that would have
caught each of the three original mistakes.

Run: python tools/test_harness_env_parity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "tools"))

import run_multi_gpu_batch as mg  # noqa: E402
import run_single_gpu_batch as sg  # noqa: E402

REFERENCE = root / "test" / "reference" / "batch_reference_env_238.json"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
expected: dict[str, str] = dict(reference["env"])
ignored = set(reference["ignored_keys"])
operator_keys = {k for k in reference["operator_env"] if not k.startswith("_")}
removed_keys = {k for k in reference["harness_removed"] if not k.startswith("_")}


def comparable(env: dict[str, str]) -> dict[str, str]:
    """Drop the keys that are paths, workdirs, or RASBERY's own re-exec stamps."""
    return {k: v for k, v in env.items() if k not in ignored}


def diff(actual: dict[str, str], want: dict[str, str]) -> list[str]:
    problems = []
    for key in sorted(set(want) | set(actual)):
        if key not in actual:
            problems.append(f"MISSING {key}={want[key]!r}")
        elif key not in want:
            problems.append(f"EXTRA   {key}={actual[key]!r}")
        elif actual[key] != want[key]:
            problems.append(f"DIFFERS {key}: harness {actual[key]!r} != reference {want[key]!r}")
    return problems


# ===========================================================================
# 1. The dispatcher's control arm IS the reference line
# ===========================================================================
#
# `--gpus 0 --procs-per-gpu 1 --batch-width 64` on a 24-core host is the arm
# the whole L5 matrix is measured against.  With no --set flags at all, what it
# hands the child must be what the reference line ran.

budget = mg.plan_host_budget(
    gpus=["0"], batch_width=64, visible_cpus=24, pin="taskset",
    driver_workers=None, solver_threads=None, procs_per_gpu=1,
)
control = mg.resolve_profile_env(
    batch_width=64,
    driver_workers=budget.driver_workers,
    solver_threads=budget.solver_threads,
    gpu="0",
)
problems = diff(comparable(control), expected)
check(not problems,
      "the dispatcher's --gpus 0 --procs-per-gpu 1 --batch-width 64 control arm does not "
      "resolve the 238 reference environment. Every L5 multiple is measured against this "
      "arm, so a difference here voids the whole matrix:\n    "
      + "\n    ".join(problems))

# The two harnesses must not be able to drift apart: they share one resolver,
# so the single-GPU profiler at the same width resolves the same thing.
single = sg.resolve_profile_env(
    batch_width=64, driver_workers=64, solver_threads=64, gpu="0",
)
check(comparable(single) == comparable(control),
      "the single-GPU harness and the dispatcher resolved DIFFERENT environments at the "
      "same width. They import one resolver precisely so that a change to what a valid "
      "run looks like cannot apply to one harness and not the other:\n    "
      + "\n    ".join(diff(comparable(single), comparable(control))))

# ...and the same is true through each harness's own argument path, not only
# through the resolver called directly.
plan, command, env = sg.build_plan(
    __import__("types").SimpleNamespace(
        batch_width=64, gpu="0", host_workers="legacy", worker_factor=1.0,
        set_values=[], result=None, no_oversubscribe=False, solver_threads=None,
        pin_omp=False,
    ),
    ["RASBERY", "--rasi"] + [f"d{i}.json" for i in range(64)]
    + ["--raso"] + [f"o{i}.h5" for i in range(64)],
)
check(plan.host_workers == 64,
      f"single-GPU harness: host_workers={plan.host_workers} at width 64 with 64 decks. "
      "The default must be the executable's own min(batch_width, jobs) = 64 "
      "(main.cpp:698), not a core count")
check(plan.solver_threads == 64,
      f"single-GPU harness: solver_threads={plan.solver_threads}, expected the arena width")
check(all(env.get(k) == v for k, v in expected.items()),
      "the single-GPU harness's build_plan() env does not carry the reference values:\n    "
      + "\n    ".join(f"{k}: {env.get(k)!r} != {v!r}"
                      for k, v in expected.items() if env.get(k) != v))


# ===========================================================================
# 2. Keys the harness must NOT set
# ===========================================================================

for key in operator_keys:
    check(key not in control,
          f"the harness set {key}, which is the OPERATOR's. RASBERY_ALLOW_SCREENING is a "
          "permission and a harness that grants itself permission to run a screening job "
          "is not a guard; RASBERY_BATCH_LIGHT_RESULT is what `--result light` already "
          "says (main.cpp:399); RASBERY_BATCH_RECEIPT_JSONL is a path, and K processes "
          "appending to one would interleave")

for key in removed_keys:
    if key.endswith("_zero"):
        continue
    check(key not in control,
          f"{key} is back in the default environment. The reference line does not set it, "
          "and the reason it was removed is recorded in "
          "test/reference/batch_reference_env_238.json under harness_removed -- read it "
          "before re-adding the key")

check(control.get("RASBERY_BATCH_WAIT_US") == "auto"
      and control.get("RASBERY_BATCH_WAIT_MAX_US") == "2000",
      "the batch rendezvous must run the bounded adaptive linger the reference runs, not "
      "the forced 0 the harnesses used to set: a rendezvous that never lingers is how a "
      "64-slot arena ends up gathering three participants")


# ===========================================================================
# 3. Negative controls -- each one is a mistake this actually made
# ===========================================================================

# (a) The host-thread cap.  24 lanes into 64 slots: the 115.6 c/h defect.
capped = mg.plan_host_budget(
    gpus=["0"], batch_width=64, visible_cpus=24, pin="taskset",
    driver_workers=None, solver_threads=None, procs_per_gpu=1, oversubscribe=False,
)
capped_env = mg.resolve_profile_env(
    batch_width=64, driver_workers=capped.driver_workers,
    solver_threads=capped.solver_threads, gpu="0",
)
check(capped.driver_workers == 24,
      f"negative control: --no-oversubscribe must still BE the old policy "
      f"({capped.driver_workers} lanes, expected 24) -- it is kept as a deliberate arm")
check(diff(comparable(capped_env), expected),
      "negative control: the CPU-capped budget resolved an environment IDENTICAL to the "
      "reference. Then this test cannot see the 5x defect it exists to catch")

# (b) The forced OMP_PROC_BIND/OMP_PLACES pair.
pinned = mg.resolve_profile_env(
    batch_width=64, driver_workers=64, solver_threads=64, gpu="0", pin_omp=True,
)
check(pinned.get("OMP_PROC_BIND") == "TRUE" and pinned.get("OMP_PLACES") == "cores",
      "--pin-omp must still be able to declare the pair; it is an arm, not a deletion")
check(diff(comparable(pinned), expected),
      "negative control: --pin-omp resolved the reference environment exactly, so the "
      "comparison is not looking at OMP_PROC_BIND/OMP_PLACES at all")

# (c) The forced RASBERY_BATCH_WAIT_US=0.
zeroed = mg.resolve_profile_env(
    batch_width=64, driver_workers=64, solver_threads=64, gpu="0",
    overrides={"RASBERY_BATCH_WAIT_US": "0"},
)
check(diff(comparable(zeroed), expected),
      "negative control: forcing RASBERY_BATCH_WAIT_US=0 still compared equal to the "
      "reference, so the wait policy is outside the contract")

# (d) A harness-only key that nobody declared.
sneaked = mg.resolve_profile_env(
    batch_width=64, driver_workers=64, solver_threads=64, gpu="0",
    overrides={"RASBERY_PPR_MODE": "master"},
)
check(any(p.startswith("EXTRA   RASBERY_PPR_MODE") for p in diff(comparable(sneaked), expected)),
      "negative control: an EXTRA key must be reported as extra. A harness default the "
      "reference never had is exactly what RASBERY_PPR_MODE=master was, and it moved the "
      "measurement: PPR runs on every statepoint (Driver.h:4166 ff), light or not")

# (e) An ignored key must be genuinely ignored in BOTH directions.
with_path = dict(control)
with_path["LD_LIBRARY_PATH"] = "/some/toolchain/lib"
check(not diff(comparable(with_path), expected),
      "a path key must not break parity: the reference and the run are on different hosts "
      "and their paths are not comparable")


# ===========================================================================
# 4. K > 1: the cores are split, the lanes are not
# ===========================================================================
#
# The whole L5 matrix keeps the DECLARED width per device at 64 and varies how
# it is divided.  Each process is a width-W arena and needs W lanes to fill it;
# what is divided K ways is the taskset range, not the lane count.

k4 = mg.plan_host_budget(
    gpus=["0"], batch_width=16, visible_cpus=24, pin="taskset",
    driver_workers=None, solver_threads=None, procs_per_gpu=4,
)
k4_env = mg.resolve_profile_env(
    batch_width=16, driver_workers=k4.driver_workers,
    solver_threads=k4.solver_threads, gpu="0",
)
check(k4_env["RASBERY_BATCH_HOST_THREADS"] == "16",
      f"K=4 W=16: lanes={k4_env['RASBERY_BATCH_HOST_THREADS']}, expected the width per "
      "process (16). Lanes are GPU-wait-blocked; dividing them by K starves each arena")
check(k4_env["RASBERY_OMP_THREADS"] == "16"
      and k4_env["OMP_NUM_THREADS"] == "16"
      and k4_env["OMP_THREAD_LIMIT"] == "16",
      "K=4 W=16: the three OMP thread keys must be the width PER PROCESS, the way the "
      "reference's three 64s are its width")
check(k4.cpus_per_proc == 6 and [k4.cpu_sets[i][0] for i in range(4)] == [0, 6, 12, 18],
      f"K=4: the CORES must still be partitioned (cpus_per_proc={k4.cpus_per_proc}, "
      f"sets start at {[s[0] for s in k4.cpu_sets]}). Oversubscribing the lanes is the "
      "policy; sharing a taskset range between processes is a different bug")
# Everything that is not width-derived is the same in every arm.
width_keys = set(sg.WIDTH_DERIVED_KEYS)
check({k: v for k, v in k4_env.items() if k not in width_keys}
      == {k: v for k, v in control.items() if k not in width_keys},
      "K=4 changed something other than the width-derived thread counts, so the arms of "
      "the L5 matrix are not comparable to each other")


if failures:
    raise SystemExit("harness env parity: FAIL\n  " + "\n  ".join(failures))
print("harness env parity: PASS")
