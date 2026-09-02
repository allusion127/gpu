#!/usr/bin/env python3
"""Launch one-GPU RASBERY multi-input runs with an explicit CPU/GPU work split.

Example:
  python tools/run_single_gpu_batch.py --batch-width 64 --gpu 0 -- \
      ./RASBERY --rasi deck0.json deck1.json ... \
                --raso out0.h5 out1.h5 ... --batch-mode 64

The CUDA arena width and the number of host Driver workers are deliberately
separate, but the DEFAULT is not a CPU budget: it is min(batch_width, jobs),
the executable's own default (main.cpp:698), which puts 64 Driver lanes on a
24-core host on purpose.  A lane is blocked on the GPU rendezvous nearly all of
its life, so lanes are not CPU workers; capping them at the core count caps the
achievable rendezvous width at 24 of 64 slots before the run begins, and that
is worth 5x on the 238 host (582 c/h -> 115.6).  `--no-oversubscribe` takes the
old CPU-capped policy deliberately.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_audit import (  # noqa: E402
    DECLARABLE_FIDELITIES,
    NON_STRICT_ENV_KEYS,
    audit_physics_mode,
    derive_declared_fidelity,
    receipt_preset,
)


# `[RASBERY][BATCH_HOST] {...}` -- the JSON receipt only; main() also emits a
# plain-prose BATCH_HOST line, which this deliberately does not match.
BATCH_HOST_RECEIPT = re.compile(r"\[RASBERY\]\[BATCH_HOST\]\s*(\{.*\})")
# graph_fallbacks is reported by three receipts under the same key name:
# [RASBERY][CUDA][BACKEND_COUNTERS], [RASBERY][NODAL][GPU] and
# [RASBERY][NODAL][BATCH].  Any nonzero one means a capture was refused and the
# run is not the configuration the benchmark claims to measure.
GRAPH_FALLBACK_COUNTER = re.compile(r'"([A-Za-z_]*graph_fallbacks)"\s*:\s*(\d+)')
# `[RASBERY][PHYSICS_MODE] {...}` -- main() emits this before any deck starts.
# The audit of that receipt lives in tools/exact_audit.py and is keyed on
# FIDELITY (`policy` / `acceptance_eligible`), never on what the run WROTE.
#
# WHAT WP1 REMOVED FROM HERE, AND WHY (review doc R5).  This module used to hold
# EXACT_PHYSICS_MODE / SCREENING_PHYSICS_MODE and pick between them with
# expected_physics_mode(plan.result_mode).  Both halves were wrong, in opposite
# directions: a strict run that wrote scalars (`--result light`) was voided on
# `full_hdf5:false`, a field that says nothing about the physics; and the
# workaround -- EXPECTING the screening receipt for a light chunk -- accepted
# `screening:true` for a run that was not screening at all.  Since WP1 the
# binary reports the two axes separately (src/RunContract.h), so the audit reads
# the fidelity and `result_mode` is reporting only.
# ---------------------------------------------------------------------------
# The 238 production reference environment
# ---------------------------------------------------------------------------
#
# This is, key for key, the environment of the RAW production line that
# measured 582 cases/hour (12.3 GB) on the 238 host (24 cores, RTX PRO 6000):
#
#     taskset --cpu-list 0-23 RASBERY --rasi ... --raso ... --batch-mode 64
#
# It is recorded in test/reference/batch_reference_env_238.json and pinned by
# tools/test_harness_env_parity.py, which fails if what a harness resolves and
# what the reference ran stop being the same set.  BOTH harnesses resolve this
# by default, because a harness whose default environment is not the
# reference's cannot reproduce the reference's number -- and the way that
# failure presents is a throughput figure, not an error.  Measured cost of
# getting it wrong: the dispatcher's own control arm returned 115.6 c/h against
# the same binary's 582, a 5.0x loss that three earlier sweeps reported as data
# (docs/W4_L5_MULTIPROC_PER_GPU_20260901_KO.md Sec 4.7).
#
# WHAT IS DELIBERATELY NOT HERE.
#
#   * OMP_PROC_BIND / OMP_PLACES.  The reference sets NEITHER; `--pin-omp` adds
#     them.  (main.cpp:286-287 sets OMP_PROC_BIND=TRUE and OMP_PLACES=cores
#     itself and re-execs, and does so with overwrite, so the child is bound
#     either way -- what --pin-omp controls is whether the HARNESS declares it,
#     and therefore whether the declared env and the run agree.)
#   * The four width-derived thread counts -- see WIDTH_DERIVED_KEYS below.
#     They are computed per launch, because "the reference's 64" is the arena
#     width, not the constant 64.
#   * RASBERY_ALLOW_SCREENING / RASBERY_BATCH_LIGHT_RESULT /
#     RASBERY_BATCH_RECEIPT_JSONL.  The reference exported these for its light
#     arm.  `--result light` is how a harness says the same thing, and the
#     receipt path is a path.  The screening opt-in stays the OPERATOR's: a
#     harness that grants itself permission to run a screening job is not a
#     guard.
#   * RASBERY_PPR_MODE=master.  This WAS a harness default and the reference
#     line does not set it -- so it was a silent, undeclared deviation, and not
#     a free one: pin-power reconstruction runs on every statepoint
#     (Driver.h:4166 ff), light result or not, so the mode changes the
#     measurement as well as the pin powers.  MASTER-comparison campaigns pass
#     `--set RASBERY_PPR_MODE=master` explicitly.
#   * RASBERY_NODAL_BATCH_WAIT_US.  The reference sets neither nodal wait key,
#     and the arena's own default is already 0 (CudaXsReconBackend.cu:1637), so
#     stating it bought nothing and hid the CMFD/nodal asymmetry below.
DEFAULT_ENV = {
    # --- the v3 production physics arm (docs/W4_L5 Sec 4.1) -----------------
    "RASBERY_GPU": "1",
    "RASBERY_GPU_CMFD_SWEEP": "1",
    "RASBERY_GPU_CMFD_RESIDENT_SINGLE": "1",
    "RASBERY_GPU_NODAL": "1",
    "RASBERY_GPU_NODAL_FULL": "1",
    "RASBERY_GPU_XSRECON": "1",
    "RASBERY_GPU_FLATXS": "1",
    "RASBERY_GPU_WIEL_FOLD": "chunked",
    "RASBERY_GPU_XE": "1",
    "RASBERY_XE_ANDERSON": "1",
    "RASBERY_STAGED_FLUX_TOL": "50",
    "RASBERY_STAGED_XE_TOL": "1000",
    "RASBERY_STAGED_LOOSE_SETTLE": "1",
    "RASBERY_GPU_RB_SWEEPS": "4",
    "RASBERY_PC_MODE": "decart",
    # --- the batch rendezvous ------------------------------------------------
    # `auto` with a 2 ms ceiling, NOT 0.  The harnesses used to force 0 on the
    # strength of an early M64 sweep; the production line that actually reached
    # 582 c/h runs the bounded adaptive linger (CudaBICGBackend.cu:4705-4722),
    # and a rendezvous that never lingers is how a 64-slot arena ends up
    # gathering three participants.
    "RASBERY_BATCH_WAIT_US": "auto",
    "RASBERY_BATCH_WAIT_MAX_US": "2000",
    # --- OpenMP / BLAS -------------------------------------------------------
    # One active level is what makes 64 oversubscribed Driver lanes safe: the
    # lanes are GPU-wait-blocked, and their inner solver regions are serialised
    # rather than multiplied (main.cpp:764-769).
    "OMP_DYNAMIC": "FALSE",
    "OMP_NESTED": "FALSE",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "OMP_WAIT_POLICY": "PASSIVE",
    "OMP_STACKSIZE": "128M",
    "GOMP_SPINCOUNT": "0",
    "MKL_NUM_THREADS": "1",
    "KMP_BLOCKTIME": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # --- CUDA ----------------------------------------------------------------
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100",
}

# Added ONLY with --pin-omp.  See the note above: the executable sets both
# itself, so this is about what the harness declares, not about what libgomp
# ends up doing.
PIN_OMP_ENV = {
    "OMP_PROC_BIND": "TRUE",
    "OMP_PLACES": "cores",
}

# Set per launch from the arena width and the worker plan, never from a
# constant: the reference's "64" is `--batch-mode 64`, and a K-process arm of
# width 16 wants 16.
WIDTH_DERIVED_KEYS = (
    "RASBERY_BATCH_HOST_THREADS",
    "RASBERY_OMP_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)


def resolve_profile_env(
    *,
    batch_width: int,
    driver_workers: int,
    solver_threads: int,
    gpu: str | None = None,
    pin_omp: bool = False,
    extra: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
    unset: Iterable[str] | None = None,
) -> dict[str, str]:
    """The env a RASBERY child is launched with, ON TOP of the inherited one.

    One function for both harnesses, for the same reason check_run_receipts()
    is one function: an environment difference between the single-GPU profiler
    and the dispatcher is invisible in every receipt either of them prints, and
    it is worth 5x.

    Precedence, lowest to highest: DEFAULT_ENV, the width-derived thread
    counts, --pin-omp, CUDA_VISIBLE_DEVICES, *extra* (what the run itself
    computed, e.g. the MPS client env), *overrides* (--set), *unset*
    (--set-unset / --strict).  The operator's --set is last but one so that an
    arm can always be taken deliberately, and a deletion beats every setter
    because it is the only way to say "this key must not reach the child" --
    which is what the strict-policy control arm needs, DEFAULT_ENV carrying the
    A2 staged tolerances since 7099e54.

    WHY driver_workers IS NOT CAPPED AT THE CORE COUNT.  A Driver lane is not a
    CPU-bound worker: it spends nearly all of its life blocked on the GPU
    rendezvous, and the arena only ever gathers the lanes that happen to be
    inside it.  Capping the lanes at 24 on a 24-core host caps the ACHIEVABLE
    rendezvous width at 24 of 64 declared slots before the run starts -- which
    is why the raw reference deliberately runs 64 lanes on 24 cores and why the
    dispatcher's 24-lane "budget" measured 115.6 c/h against its 582.  What
    keeps the oversubscription bounded is OMP_MAX_ACTIVE_LEVELS=1: the lanes do
    not spawn nested solver teams.
    """
    env = dict(DEFAULT_ENV)
    env["RASBERY_BATCH_HOST_THREADS"] = str(driver_workers)
    env["RASBERY_OMP_THREADS"] = str(solver_threads)
    env["OMP_NUM_THREADS"] = str(solver_threads)
    env["OMP_THREAD_LIMIT"] = str(solver_threads)
    if pin_omp:
        env.update(PIN_OMP_ENV)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.update(extra or {})
    env.update(overrides or {})
    for key in unset or ():
        env.pop(key, None)
    return env


def launch_env(profile: dict[str, str], unset: Iterable[str] | None = None) -> dict[str, str]:
    """The child's WHOLE environment: this process's, plus the profile.

    The unset list is applied again here, and it has to be: a key the harness
    never set can still be exported in the operator's shell, and popping it out
    of the profile alone would leave it in the inherited environment where
    src/RunContract.h reads it.  That is the difference between `--strict`
    meaning "the harness does not set the staged tolerances" and `--strict`
    meaning "this child runs strict".
    """
    env = os.environ.copy()
    env.update(profile)
    for key in unset or ():
        env.pop(key, None)
    return env


@dataclass(frozen=True)
class LaunchPlan:
    batch_width: int
    jobs: int
    visible_cpus: int
    host_workers: int
    worker_policy: str
    gpu: str
    # RASBERY_OMP_THREADS / OMP_NUM_THREADS per process.  Defaults to the arena
    # width (the reference's 64 at --batch-mode 64); 0 means "not stated by
    # this plan", which is what the older callers that predate the field mean.
    solver_threads: int = 0
    # The EFFECTIVE result mode of this launch: "light" if any job in it writes
    # scalar-only output, else "full"/"pin-off".  REPORTING ONLY since WP1: the
    # acceptance audit is keyed on the run's FIDELITY (tools/exact_audit.py) and
    # what a case writes is never a reason to void it.
    result_mode: str = "full"
    # The fidelity this launch DECLARES: strict | A2 | L3coarse | feedback_limited.
    # The audit passes iff the child's [PHYSICS_MODE] receipt says exactly this.
    # Default `strict` so a caller that predates the field still gets the old
    # rule; every real launch derives it from the resolved child environment.
    declared_fidelity: str = "strict"
    # "operator" (--fidelity) or "env" (derived).  Reported, so a receipt says
    # whether the word was typed or computed.
    fidelity_source: str = "env"
    # WP24.  The NAMED preset this launch handed the child -- the child env's
    # RASBERY_FIDELITY, or "none" -- which check_run_receipts() requires the
    # child to have PRINTED BACK.  Default "none" is the preset-free launch, so
    # a caller that predates the field still states the truth.
    declared_preset: str = "none"


def declared_preset_from_env(env: "Mapping[str, str]") -> str:
    """The preset name a child launched with *env* is being ASKED for.

    ONE spelling, shared by both harnesses, and it is the same vocabulary the
    binary prints: CaseFidelity::presetToken() answers `"none"` when no row was
    named, so the request and the receipt are comparable without either side
    special-casing an empty string.
    """
    return (env.get("RASBERY_FIDELITY") or "").strip() or "none"


def visible_cpu_threads() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def values_after(command: Sequence[str], flag: str) -> list[str]:
    """Return values following *flag* until the next long option."""
    try:
        start = command.index(flag) + 1
    except ValueError:
        return []
    values: list[str] = []
    for token in command[start:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def batch_width_from_command(command: Sequence[str]) -> int | None:
    values = values_after(command, "--batch-mode")
    if not values:
        return None
    try:
        value = int(values[0])
    except ValueError as exc:
        raise ValueError(f"invalid --batch-mode value: {values[0]!r}") from exc
    if value <= 0:
        raise ValueError("--batch-mode must be positive")
    return value


def compute_host_workers(
    request: str,
    *,
    batch_width: int,
    jobs: int,
    visible_cpus: int,
    worker_factor: float,
) -> tuple[int, str]:
    cap = max(1, min(batch_width, jobs))
    normalized = request.strip().lower()
    if normalized == "legacy":
        # min(batch_width, jobs) -- EXACTLY what main.cpp:698 does with
        # RASBERY_BATCH_HOST_THREADS absent, which is what the raw reference
        # line runs.  Restating it here rather than leaving the variable unset
        # keeps the [BATCH_HOST] receipt audit meaningful: the harness declares
        # a number and then checks the executable printed it.
        return cap, "binary_default_min_width_jobs"
    if normalized in ("cores", "no-oversubscribe"):
        # The OLD dispatcher policy, kept as a deliberate arm: one lane per
        # visible CPU.  It is a measurement of core count, not of rendezvous
        # width, and on the 238 host it costs 5x -- take it on purpose or not
        # at all.
        return min(cap, max(1, visible_cpus)), "no_oversubscribe_cpu_capped"
    if normalized == "auto":
        if worker_factor <= 0.0:
            raise ValueError("--worker-factor must be positive")
        cpu_budget = max(1, int(round(visible_cpus * worker_factor)))
        return min(cap, cpu_budget), f"auto_cpu_x{worker_factor:g}"
    try:
        explicit = int(request)
    except ValueError as exc:
        raise ValueError(
            "--host-workers must be legacy, cores, auto, or a positive integer"
        ) from exc
    if explicit <= 0:
        raise ValueError("--host-workers must be positive")
    return min(cap, explicit), "explicit"


def path_key(path: str) -> str:
    """Job-namespace comparison key, matching main.cpp's rasberyPathKey().

    realpath, not abspath: `std::filesystem::weakly_canonical` on the C++ side
    resolves symlinks, so two --raso paths that reach one file through a link
    have to compare equal here too (plan Rev.4 Sec 7).  normcase folds case on
    Windows, where the filesystem does.
    """
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def validate_deck_paths(command: Sequence[str]) -> list[str]:
    """Reject deck/output combinations that would silently clobber results.

    RASBERY defaults a missing --raso to `<input_dir>/result.h5`, so a multi-deck
    run without --raso hands every Driver the same output path: the decks race on
    one HDF5 file and the run produces a single, arbitrary survivor.  That has to
    be caught here, before the executable is launched and the GPU time is spent.

    Repeated --rasi decks stay ALLOWED on purpose: sweeping many states off one
    input file is the batch workload.  Only the output namespace has to be
    distinct, and since Driver::RestartPath now derives restart files from the
    output path, distinct --raso is also what keeps the restart namespaces apart.
    """
    rasi = values_after(command, "--rasi")
    raso = values_after(command, "--raso")
    if not rasi:
        return rasi
    if len(rasi) > 1 and not raso:
        raise ValueError(
            "%d --rasi decks were given with no --raso: every Driver would write the same "
            "default result.h5 and the decks would overwrite each other. Pass one --raso "
            "path per --rasi deck." % len(rasi)
        )
    if raso and len(rasi) != len(raso):
        raise ValueError(
            "--rasi count %d does not match --raso count %d" % (len(rasi), len(raso))
        )
    seen = set()
    repeated = []
    for path in raso:
        key = path_key(path)
        if key in seen:
            repeated.append(path)
        seen.add(key)
    if repeated:
        raise ValueError(
            "--raso paths must be distinct, one per deck; repeated: %s"
            % ", ".join(sorted(set(repeated)))
        )
    return rasi


def check_run_receipts(output: str, plan: "LaunchPlan") -> list[str]:
    """Post-run receipt audit: did the run have the shape that was asked for?"""
    # Keyed on FIDELITY, not on what the run wrote: `plan.result_mode` is no
    # longer an argument (review doc R5).  What IS an argument is the fidelity
    # the launch declared -- the audit verifies the run solved at the fidelity
    # that was asked for, in both directions (tools/exact_audit.py).
    problems = audit_physics_mode(output, plan.declared_fidelity)

    # WP24.  AND THE PRESET THE CHILD ACTUALLY RAN, not only the policy word.
    #
    # THE FAILURE THIS CATCHES, which is the most likely operational mistake on
    # a multi-host fleet: run the documented `--set RASBERY_FIDELITY=screen100`
    # against a binary built BEFORE WP24 -- one host not rebuilt, a stale
    # RASBERY on the PATH.  That binary does not know the variable, ignores it,
    # solves the plain DEFAULT_ENV 50/1000 A2 arm at PRODUCTION polish
    # tolerances, and prints `policy:"A2"`.  derive_declared_fidelity() also
    # answers "A2" (it reads the ROW's multipliers), so the policy comparison
    # above sees A2 against A2 and PASSES -- and the campaign files a
    # production-tolerance throughput number under a screening name, which is
    # verbatim the defect src/FidelityPreset.h and main.cpp's
    # fidelityPresetEnvIsUnknown() refusal exist to prevent.  main.cpp cannot
    # help here because the OLD binary is the one running; only the harness,
    # which knows both halves, can.
    #
    # An ABSENT field is a REFUSAL and not a pass, the same shape
    # CASE_REQUIRED_FIELDS uses for the WP10.3 version check: the field that
    # would have voided the run is the missing one.
    printed_preset = receipt_preset(output)
    if printed_preset is None:
        problems.append(
            "[RASBERY][PHYSICS_MODE] carries no `fidelity_preset` field: this binary "
            "predates WP24 and cannot honour the preset it was given "
            "(RASBERY_FIDELITY=%s). It ignored the variable and solved at whatever the "
            "RASBERY_STAGED_* environment said, at PRODUCTION polish tolerances, while "
            "reporting the same `policy` word a screening arm reports -- so this run's "
            "number cannot be filed in either column. Rebuild this host."
            % plan.declared_preset)
    elif printed_preset != plan.declared_preset:
        problems.append(
            "[RASBERY][PHYSICS_MODE] fidelity_preset=%r but this launch asked for %r. "
            "`policy` is `A2` for every staged arm this binary can run, so the policy "
            "check above cannot see this: the run solved at a NAMED convergence policy "
            "nobody requested."
            % (printed_preset, plan.declared_preset))

    receipt = None
    for match in BATCH_HOST_RECEIPT.finditer(output):
        try:
            receipt = json.loads(match.group(1))
        except ValueError:
            problems.append(
                "could not parse the [RASBERY][BATCH_HOST] receipt: %s" % match.group(1)
            )
    if receipt is None:
        problems.append(
            "no [RASBERY][BATCH_HOST] JSON receipt in the run output: the multi-instance "
            "batch branch never ran, so nothing about this run is a batch measurement"
        )
    else:
        actual = receipt.get("host_threads")
        if actual != plan.host_workers:
            problems.append(
                "[RASBERY][BATCH_HOST] host_threads=%r but %d was requested via "
                "RASBERY_BATCH_HOST_THREADS (policy %s)"
                % (actual, plan.host_workers, plan.worker_policy)
            )

    for name, value in GRAPH_FALLBACK_COUNTER.findall(output):
        if int(value) != 0:
            problems.append(
                "%s=%s: a CUDA graph capture was refused and the work ran as plain "
                "launches, so this run is not the captured-graph configuration" % (name, value)
            )
    return problems


def resolve_unset(args: argparse.Namespace) -> list[str]:
    """Keys deleted from the child environment (--set-unset, --strict).

    `--strict` is not a synonym for "do not set the staged tolerances": the
    campaign's DEFAULT_ENV sets them, an operator's shell may also export them,
    and RASBERY_GA_FEEDBACK_PASSES / RASBERY_PHYSICS_FIDELITY can move a run off
    strict from the inherited environment alone.  So the switch names every key
    that can, and deletes all of them (tools/exact_audit.NON_STRICT_ENV_KEYS).
    """
    keys = list(getattr(args, "set_unset", None) or [])
    if getattr(args, "strict", False):
        keys.extend(k for k in NON_STRICT_ENV_KEYS if k not in keys)
    for key in keys:
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"invalid environment key: {key!r}")
    return keys


def resolve_declared_fidelity(args: argparse.Namespace,
                              env: dict[str, str]) -> tuple[str, str]:
    """(fidelity, source) for this launch.

    The operator's `--fidelity` wins; otherwise it is DERIVED from the child
    environment by the binary's own rule.  A derivation is not a guess: the
    default environment IS the A2 arm, and a harness that defaulted the
    declaration to `strict` would fail every production wave on a mismatch --
    which is the defect this argument exists to end, arriving from the other
    side.
    """
    stated = getattr(args, "fidelity", None)
    if stated:
        return stated, "operator"
    try:
        return derive_declared_fidelity(env), "env"
    except ValueError as exc:
        # WP24.  derive_declared_fidelity() raises on a RASBERY_FIDELITY this
        # build has no row for, because the binary REFUSES it and there is then
        # no policy to declare.  Its message is carefully worded; surfacing it
        # as a Python traceback out of the harness would bury it.
        # sys.argv[0] and not __file__: resolve_declared_fidelity() is imported
        # and called by run_multi_gpu_batch.py too, so naming this file sends the
        # reader to a script they did not run.
        raise SystemExit(f"{os.path.basename(sys.argv[0]) or __file__}: {exc}") from None


def parse_overrides(items: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"invalid environment key: {key!r}")
        result[key] = value
    return result


def build_plan(args: argparse.Namespace, command: list[str]) -> tuple[LaunchPlan, list[str], dict[str, str]]:
    command_width = batch_width_from_command(command)
    batch_width = args.batch_width or command_width
    if batch_width is None:
        raise ValueError("provide --batch-width or include --batch-mode M in the RASBERY command")
    if command_width is not None and command_width != batch_width:
        raise ValueError(
            f"--batch-width {batch_width} disagrees with command --batch-mode {command_width}"
        )
    if command_width is None:
        command.extend(["--batch-mode", str(batch_width)])
    if getattr(args, "result", None) and "--result" not in command:
        command.extend(["--result", args.result])

    rasi = validate_deck_paths(command)
    jobs = len(rasi) if rasi else batch_width
    cpus = visible_cpu_threads()
    request = args.host_workers
    if getattr(args, "no_oversubscribe", False) and request == "legacy":
        request = "cores"
    workers, policy = compute_host_workers(
        request,
        batch_width=batch_width,
        jobs=jobs,
        visible_cpus=cpus,
        worker_factor=args.worker_factor,
    )
    # The reference's RASBERY_OMP_THREADS is the arena width, not a core count.
    solver_threads = getattr(args, "solver_threads", None) or batch_width

    unset = resolve_unset(args)
    profile = resolve_profile_env(
        batch_width=batch_width,
        driver_workers=workers,
        solver_threads=solver_threads,
        gpu=str(args.gpu),
        pin_omp=getattr(args, "pin_omp", False),
        overrides=parse_overrides(args.set_values),
        unset=unset,
    )
    env = launch_env(profile, unset)
    fidelity, fidelity_source = resolve_declared_fidelity(args, env)

    plan = LaunchPlan(
        batch_width=batch_width,
        jobs=jobs,
        visible_cpus=cpus,
        host_workers=workers,
        worker_policy=policy,
        gpu=str(args.gpu),
        solver_threads=solver_threads,
        # A --result on the command line wins over --result here only because
        # build_plan appended ours when there was none; either way the audit
        # has to expect the receipt the executable will actually print.
        result_mode=(values_after(command, "--result") or ["full"])[0],
        declared_fidelity=fidelity,
        fidelity_source=fidelity_source,
        declared_preset=declared_preset_from_env(env),
    )
    return plan, command, env


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a same-geometry RASBERY deck set on one GPU with a bounded host worker pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-width", type=int, help="CUDA arena width; inferred from --batch-mode when omitted")
    p.add_argument("--gpu", default="0", help="physical GPU index exposed as the process's sole CUDA device")
    p.add_argument(
        "--host-workers",
        default="legacy",
        help=(
            "Driver refill lanes. legacy (default) is min(batch_width, jobs) -- the "
            "executable's own default with RASBERY_BATCH_HOST_THREADS absent "
            "(main.cpp:698), which is what the 582 c/h reference line runs. It "
            "OVERSUBSCRIBES the host on purpose: a lane is blocked on the GPU rendezvous "
            "almost all of its life, so capping lanes at the core count caps the "
            "achievable rendezvous width before the run starts (measured: 24 lanes on the "
            "238 host gave 115.6 c/h against the same binary's 582). cores (= "
            "--no-oversubscribe) is the old CPU-capped policy, auto is visible CPUs times "
            "--worker-factor, or give an explicit count"
        ),
    )
    p.add_argument(
        "--no-oversubscribe", action="store_true",
        help="shorthand for --host-workers cores: one lane per visible CPU. A deliberate "
             "arm, not a safety net -- it measures the core count, not the rendezvous",
    )
    p.add_argument(
        "--solver-threads", type=int,
        help="RASBERY_OMP_THREADS / OMP_NUM_THREADS / OMP_THREAD_LIMIT for the process; "
             "default is the arena width (the reference's 64 at --batch-mode 64). "
             "OMP_MAX_ACTIVE_LEVELS=1 keeps this from multiplying by the lane count",
    )
    p.add_argument(
        "--pin-omp", action="store_true",
        help="also export OMP_PROC_BIND=TRUE and OMP_PLACES=cores. The reference line "
             "sets neither; note that RASBERY sets both itself and re-execs "
             "(main.cpp:286), so this changes what the harness DECLARES, not what libgomp "
             "does",
    )
    p.add_argument(
        "--print-env", action="store_true",
        help="print the resolved child environment as a receipt and exit, so a mismatch "
             "against test/reference/batch_reference_env_238.json is visible before the "
             "GPU time is spent",
    )
    p.add_argument(
        "--worker-factor",
        type=float,
        default=1.0,
        help="auto-mode host oversubscription factor; benchmark 1.0, 1.5 and 2.0 on the target host",
    )
    p.add_argument(
        "--result",
        choices=("full", "pin-off", "light"),
        help=(
            "what every job writes: full (result HDF5 + restarts + pin CSV), pin-off (no "
            "pin output), light (scalar JSONL, no HDF5).  All three run the same physics "
            "and produce the same trajectory digest -- this is an output-shape switch, not "
            "a fidelity one.  Appended to the RASBERY command as --result MODE; a --jobs "
            "manifest line's own third field still wins.  Since WP1 light is NOT a "
            "screening run and needs no RASBERY_ALLOW_SCREENING"
        ),
    )
    p.add_argument(
        "--fidelity", "--expect-fidelity", dest="fidelity",
        choices=DECLARABLE_FIDELITIES,
        help=(
            "the convergence/statepoint policy this run DECLARES. The audit passes iff "
            "the child's [RASBERY][PHYSICS_MODE] receipt reports exactly this policy -- "
            "a mismatch either way is a hard failure, because a number measured at a "
            "fidelity other than the declared one belongs in neither column (plan Sec "
            "6.2 forbids mixing strict and A2 in one table). The DEFAULT is derived "
            "from the resolved child environment by the binary's own rule "
            "(src/RunContract.h): the staged tolerances in DEFAULT_ENV make it A2"
        ),
    )
    p.add_argument(
        "--strict", action="store_true",
        help="run the STRICT-policy control arm: delete RASBERY_STAGED_FLUX_TOL / "
             "_XE_TOL / _LOOSE_SETTLE, RASBERY_GA_FEEDBACK_PASSES and "
             "RASBERY_PHYSICS_FIDELITY from the child environment, inherited ones "
             "included, so the run really does solve strict. Without it the harness "
             "measures the A2 arm, which is what DEFAULT_ENV configures",
    )
    p.add_argument(
        "--set-unset", dest="set_unset", action="append", default=[], metavar="KEY",
        help="delete one key from the child environment. Beats --set and the inherited "
             "environment both; the only way to say a key must not reach the child",
    )
    p.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE", help="override one profile environment variable")
    p.add_argument("--dry-run", action="store_true", help="print the plan without executing RASBERY")
    p.add_argument("command", nargs=argparse.REMAINDER, help="RASBERY executable and arguments, preceded by --")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("error: missing RASBERY command after --", file=sys.stderr)
        return 2
    if args.batch_width is not None and args.batch_width <= 0:
        print("error: --batch-width must be positive", file=sys.stderr)
        return 2

    try:
        plan, command, env = build_plan(args, command)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The resolved child env, recomputed from the plan: printed rather than
    # summarised, because every mismatch this harness has ever hidden was one
    # key deep and the run that revealed it took seven minutes.
    profile = resolve_profile_env(
        batch_width=plan.batch_width,
        driver_workers=plan.host_workers,
        solver_threads=plan.solver_threads,
        gpu=plan.gpu,
        pin_omp=args.pin_omp,
        overrides=parse_overrides(args.set_values),
        unset=resolve_unset(args),
    )
    receipt = {
        "gpu": plan.gpu,
        "jobs": plan.jobs,
        "arena_width": plan.batch_width,
        "visible_cpus": plan.visible_cpus,
        "host_workers": plan.host_workers,
        "worker_policy": plan.worker_policy,
        "solver_threads": plan.solver_threads,
        "result_mode": args.result or "default",
        # What this run says it is solving, and whether the word was typed or
        # derived.  --print-env exists so a mismatch is visible before the GPU
        # time is spent, and the fidelity is the field that voided a whole 238
        # wave while every receipt in it said rc=0.
        "declared_fidelity": plan.declared_fidelity,
        "declared_fidelity_source": plan.fidelity_source,
        # WP24.  The row this launch is asking the child for, so `--print-env`
        # answers "which arm is this" before the GPU time is spent.
        "declared_preset": plan.declared_preset,
        "cmfd_wait_us": env.get("RASBERY_BATCH_WAIT_US"),
        "cmfd_wait_max_us": env.get("RASBERY_BATCH_WAIT_MAX_US"),
        "command": command,
        "env": profile,
    }
    print("[RASBERY][SINGLE_GPU_PROFILE] " + json.dumps(receipt, separators=(",", ":")))
    if args.print_env or args.dry_run:
        return 0

    # stdout is teed rather than swallowed: the operator still watches the run
    # live, and the receipts stay available for the audit below.  stderr is left
    # attached to this process's stderr -- every receipt this checks is on
    # std::cout, and merging the streams would reorder the child's diagnostics.
    captured = []
    try:
        child = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )
    except OSError as exc:
        print(f"error: failed to execute {command[0]!r}: {exc}", file=sys.stderr)
        return 127
    try:
        for line in child.stdout:
            sys.stdout.write(line)
            captured.append(line)
    finally:
        child.stdout.close()
        returncode = child.wait()
    sys.stdout.flush()

    problems = check_run_receipts("".join(captured), plan)
    for problem in problems:
        print("[RASBERY][SINGLE_GPU_PROFILE][FAIL] " + problem, file=sys.stderr)
    if returncode != 0:
        return returncode
    if problems:
        return 3
    print("[RASBERY][SINGLE_GPU_PROFILE][OK] "
          + json.dumps({"host_threads": plan.host_workers, "graph_fallbacks": 0,
                        "fidelity": plan.declared_fidelity,
                        "fidelity_preset": plan.declared_preset},
                       separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
