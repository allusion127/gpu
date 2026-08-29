#!/usr/bin/env python3
"""A RASBERY stand-in that prints the receipts the launchers audit.

NOT A TEST.  It is the child the dispatcher contract tests
(`tools/test_multi_gpu_dispatch.py`, `tools/test_fleet_tuner.py`) launch instead
of the real executable, so those tests need no GPU, no CUDA and no 40 s case.

WHY ONE FILE AND NOT ONE PER TEST.  The whole point of WP8 stage 1.5 is that the
persistent-evaluator worker and the chunked worker are audited by the SAME code
and produce comparable numbers.  Two fakes that drifted apart would let one
mode's contract pass against receipts the other mode never prints, which is the
defect the shared fake exists to make impossible.  This child therefore speaks
BOTH shapes off one body of receipt-printing code:

    fake_rasbery_child.py --jobs M.txt --batch-mode W [--result R]
    fake_rasbery_child.py --evaluator-jsonl - --batch-mode W [--result R]

THE FIDELITY RECEIPT IS DERIVED FROM THIS CHILD'S OWN ENVIRONMENT, exactly the
way src/RunContract.h derives it.  A fake that always said `strict` would hide
the defect that cost the 238 tuner every candidate it measured: DEFAULT_ENV is
the A2 arm, and the harness audited it against a hardcoded word.

NEGATIVE-CONTROL KNOBS (environment, all default off).  These are what let a
contract test drive the failure paths that only ever happen on real hardware:

  FAKE_RASBERY_FAIL=<substr>      a deck whose path contains <substr> reports
                                  `"status":"failed"` and the PROCESS SURVIVES.
                                  This is one case throwing -- the isolation
                                  EvaluatorServer::runOneCase already provides.
  FAKE_RASBERY_POISON=<substr>    a deck whose path contains <substr> KILLS the
                                  process mid-wave: the decks before it get
                                  their receipts, that one and everything after
                                  it get nothing, and no [EVALUATOR][WAVE] line
                                  is printed.  This is the layer only the
                                  dispatcher can handle.
  FAKE_RASBERY_POISON_MARKER=<p>  poison ONCE: the first image to die writes <p>
                                  and every later image treats the poisoned deck
                                  as an ordinary case.  This is what makes the
                                  restart-and-re-queue SUCCESS path testable --
                                  without it the retry dies too and the test can
                                  only ever see the give-up path.
  FAKE_RASBERY_NO_READY=1         exit before printing [READY], i.e. a child
                                  that never becomes usable at all.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# The receipts a launcher audits
# ---------------------------------------------------------------------------


def _multiplier(name: str) -> float:
    try:
        value = float(os.environ.get(name, "1"))
    except ValueError:
        return 1.0
    return value if value >= 1.0 else 1.0


def declared_policy() -> tuple[str, str]:
    """(policy, physics_fidelity) by src/RunContract.h's rule."""
    if int(os.environ.get("RASBERY_GA_FEEDBACK_PASSES") or 0) > 0:
        return "feedback_limited", "feedback_limited"
    if _multiplier("RASBERY_STAGED_FLUX_TOL") > 1.0 or _multiplier("RASBERY_STAGED_XE_TOL") > 1.0:
        return "A2", "staged_a2"
    return "strict", "full_exact"


def print_declarations(result_mode: str) -> None:
    """Everything a real process prints ONCE, before any deck."""
    policy, fidelity = declared_policy()
    emit('[RASBERY][IO_WRITER] {"mode":"thread","mode_source":"default","queue_limit":8}')
    emit("[RASBERY][PHYSICS_MODE] " + json.dumps({
        "physics_mode": "full_exact_nodal", "screening": False, "feedback_pass_limit": 0,
        "full_hdf5": True, "physics_fidelity": fidelity, "policy": policy,
        "acceptance_eligible": policy == "strict", "requires_exact_rerun": False,
        "result_mode": result_mode, "fidelity_declared": None, "gpu_full": False}))


def print_batch_host(jobs: int, width: int, host: int) -> None:
    emit("[RASBERY][BATCH_HOST] " + json.dumps({
        "jobs": jobs, "arena_width": width, "host_threads": host,
        "visible_cpus": 24, "host_pinning": True, "pin_lease": True,
        "legacy_pinning_criterion": host >= jobs}))


def print_wave_counters(jobs: int, width: int) -> None:
    emit("[RASBERY][REFILL] " + json.dumps(
        {"refills": 0, "tail_idle_s": 0.25, "duplicates": 0, "stale_tenants": 0,
         "double_releases": 0, "slot_busy_fraction": 0.9}))


def print_occupancy(jobs: int, width: int) -> None:
    emit("[RASBERY][CUDA][BATCH_OCCUPANCY] " + json.dumps(
        {"launches": 10, "instance_solves": 10 * min(max(jobs, 1), width), "slots": width}))


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Running decks
# ---------------------------------------------------------------------------


def read_manifest(path: Path) -> list[list[str]]:
    return [shlex.split(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def poisoned(deck: str) -> bool:
    mark = os.environ.get("FAKE_RASBERY_POISON") or ""
    if not mark or mark not in deck:
        return False
    marker = os.environ.get("FAKE_RASBERY_POISON_MARKER")
    if marker and Path(marker).exists():
        return False  # this image is the replacement; the poison is spent
    return True


def die_on(deck: str) -> None:
    marker = os.environ.get("FAKE_RASBERY_POISON_MARKER")
    if marker:
        try:
            Path(marker).write_text(deck, encoding="utf-8")
        except OSError:
            pass
    # _exit, not sys.exit: a real CUDA abort does not unwind, does not flush a
    # buffered receipt and does not print a wave line.  Anything gentler would
    # let the dispatcher's death detection pass for the wrong reason.
    sys.stdout.flush()
    os._exit(9)


def run_case(deck: str, output: str, mode: str, *, wave_id: int, index: int,
             receipts: bool) -> bool:
    """Touch the output and report.  False when the case FAILED."""
    if poisoned(deck):
        die_on(deck)
    failed = bool(os.environ.get("FAKE_RASBERY_FAIL")) and \
        os.environ["FAKE_RASBERY_FAIL"] in deck
    if not failed:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("fake", encoding="utf-8")
    if receipts:
        emit("[RASBERY][EVALUATOR][CASE] " + json.dumps({
            "wave_id": wave_id, "case": index, "key": None,
            "case_key": f"fake-{Path(deck).stem}",
            "deck": deck, "output": output, "result_mode": mode or "full",
            "status": "failed" if failed else "ok",
            "exit_code": 1 if failed else 0,
            "digest": None if failed else f"{abs(hash(deck)) & 0xFFFFFFFFFFFFFFFF:016x}",
            "statepoints": 3, "outers": 30, "th_updates": 3, "slot": index, "lane": 0,
            "wall_s": 0.001, "teardown_ms": 0.1, "isolation_check": False,
            "error": "fake failure" if failed else None}))
    return not failed


# ---------------------------------------------------------------------------
# The two shapes
# ---------------------------------------------------------------------------


def run_chunked(argv: list[str]) -> int:
    manifest = Path(argv[argv.index("--jobs") + 1])
    width = int(argv[argv.index("--batch-mode") + 1])
    result_mode = argv[argv.index("--result") + 1] if "--result" in argv else "full"
    jobs = read_manifest(manifest)
    host = min(int(os.environ.get("RASBERY_BATCH_HOST_THREADS", width)), width, len(jobs))
    print_declarations(result_mode)
    print_batch_host(len(jobs), width, host)
    ok = True
    for index, fields in enumerate(jobs):
        mode = fields[2] if len(fields) > 2 else ""
        ok = run_case(fields[0], fields[1], mode, wave_id=1, index=index,
                      receipts=False) and ok
    print_wave_counters(len(jobs), width)
    print_occupancy(len(jobs), width)
    return 0 if ok else 1


def run_evaluator(argv: list[str]) -> int:
    width = int(argv[argv.index("--batch-mode") + 1])
    result_mode = argv[argv.index("--result") + 1] if "--result" in argv else "full"
    print_declarations(result_mode)
    if os.environ.get("FAKE_RASBERY_NO_READY"):
        return 1
    emit("[RASBERY][EVALUATOR][READY] " + json.dumps({
        "request_stream": "-", "batch_width": width, "result_mode": result_mode,
        "physics_fidelity": declared_policy()[1], "idle_timeout_s": -1.0,
        "isolation_check": False}))

    cases = failed = waves = 0
    exit_code = 0
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            request = json.loads(line)
        except ValueError:
            emit('[RASBERY][EVALUATOR][REFUSED] {"what":"not JSON","line":'
                 + json.dumps(line) + "}")
            exit_code = 2
            continue
        op = request.get("op")
        if op == "shutdown":
            break
        if op not in ("wave", "run"):
            emit('[RASBERY][EVALUATOR][REFUSED] {"what":"unknown op","line":'
                 + json.dumps(line) + "}")
            exit_code = 2
            continue
        wave_id = int(request.get("wave_id", waves + 1))
        jobs = read_manifest(Path(request["jobs_manifest"]))
        host = min(int(os.environ.get("RASBERY_BATCH_HOST_THREADS", width)), width, len(jobs))
        emit("[RASBERY][EVALUATOR][WAVE_START] " + json.dumps({
            "wave_id": wave_id, "jobs": len(jobs), "arena_width": width,
            "host_threads": host, "visible_cpus": 24, "result_mode": result_mode,
            "process_reused": waves > 0}))
        print_batch_host(len(jobs), width, host)
        ok = 0
        for index, fields in enumerate(jobs):
            mode = fields[2] if len(fields) > 2 else ""
            if run_case(fields[0], fields[1], mode, wave_id=wave_id, index=index,
                        receipts=True):
                ok += 1
            else:
                failed += 1
                exit_code = 1
        cases += len(jobs)
        waves += 1
        print_wave_counters(len(jobs), width)
        emit("[RASBERY][EVALUATOR][WAVE] " + json.dumps({
            "wave_id": wave_id, "jobs": len(jobs), "ok": ok,
            "failed": len(jobs) - ok, "wall_s": 0.01,
            "cases_per_hour": 3600.0 * len(jobs) / 0.01,
            "process_reused": waves > 1, "xslib_loads": 1, "xslib_hits": cases - 1,
            "pin_live_ranges": 0, "isolation_match": None}))

    # Teardown, in the order main.cpp's evaluator branch uses it: the arena is
    # released ONCE, here, which is why the occupancy receipt is a shutdown
    # receipt in this mode and a per-process one in the other.
    print_occupancy(cases, width)
    emit('[RASBERY][XSLIB_CACHE] {"loads":1,"hits":%d,"waits":0,"entries":1,'
         '"bytes":0,"lock_wait_ms":0,"enabled":true}' % max(0, cases - 1))
    emit("[RASBERY][EVALUATOR] " + json.dumps({
        "cases": cases, "ok": cases - failed, "failed": failed, "refused": 0,
        "generations": waves, "batch_width": width, "process_uptime_s": 0.1,
        "drive_s": 0.05, "cuda_context_reuse": max(0, cases - 1),
        "arena_releases": 1, "arena_standups": 1 if cases else 0,
        "slot_admissions": cases, "slot_duplicates": 0, "slot_stale_tenants": 0,
        "slot_double_releases": 0, "xslib_loads": 1, "xslib_hits": max(0, cases - 1),
        "library_loads": 1, "geometry_builds": cases,
        "cohort_builds": 1 if cases else 0, "cohort_hits": max(0, cases - 1),
        "pin_live_ranges_between_waves": 0,
        "case_seconds": {"p50": 0.001, "p90": 0.001},
        "case_teardown_ms": {"p50": 0.1, "p90": 0.1, "max": 0.1},
        "isolation_checks": 0, "isolation_mismatches": 0, "isolation_adjacent": 0,
        "stop_reason": "shutdown"}))
    return exit_code


def main(argv: list[str]) -> int:
    if "--evaluator-jsonl" in argv or "--evaluator" in argv:
        return run_evaluator(argv)
    return run_chunked(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
