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

WP11 SOAK KNOBS (also environment, also default off).  The soak harness asserts
a table of counters to be zero at exit; a harness that has never SEEN one of
them nonzero is a harness nobody has proved can fail.  These make each one
happen on demand:

  FAKE_RASBERY_DUPLICATES=<n>       [REFILL].duplicates
  FAKE_RASBERY_STALE_TENANTS=<n>    [REFILL].stale_tenants
  FAKE_RASBERY_ALLOC_IN_CAPTURE=<n> [CUDA][CAPTURE_ARBITER].alloc_in_capture
  FAKE_RASBERY_CAPTURES_UNWOUND=<n> [CUDA][CAPTURE_ARBITER].captures_unwound
  FAKE_RASBERY_HOST_FALLBACKS=<n>   [GPU_FULL].cmfd_fallbacks
  FAKE_RASBERY_NO_ARBITER=1         omit the [CAPTURE_ARBITER] line, i.e. a
                                    binary that cannot answer the question. The
                                    soak must refuse, not record zero.

A DECK THAT EXISTS AND DOES NOT PARSE fails its case and leaves the process
answering -- the real binary's IO::ReadInput throwing inside runOneCase's try.
That is the poisoned case a soak injects, and it needs no knob.
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


def _injected(name: str) -> int:
    """A WP11 negative-control knob, as an integer.  0 (the default) is clean."""
    try:
        return max(0, int(os.environ.get(name) or 0))
    except ValueError:
        return 0


def print_wave_counters(jobs: int, width: int) -> None:
    emit("[RASBERY][REFILL] " + json.dumps(
        {"refills": 0, "tail_idle_s": 0.25,
         # FAKE_RASBERY_DUPLICATES is what lets tools/test_soak_run.py drive the
         # soak's central assertion.  A soak harness that has never SEEN a
         # nonzero duplicate count is a harness nobody has proved can fail.
         "duplicates": _injected("FAKE_RASBERY_DUPLICATES"),
         "stale_tenants": _injected("FAKE_RASBERY_STALE_TENANTS"),
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


def resolve_case_fidelity(request: dict) -> tuple[dict, str | None]:
    """src/CaseFidelity.h resolveCaseFidelity(), in Python.

    WP10.3.  The fake has to resolve a per-case fidelity the same way the binary
    does, or a soak driven against it would exercise a protocol nothing
    implements.  The same shape and the same refusals: `strict` CLEARS the
    process's staged multipliers (the promotion lane), `A2` may not invent them,
    `L3coarse` needs a grid, and the last check is an EQUALITY both ways.
    """
    process_policy, _ = declared_policy()
    staged = process_policy == "A2"
    grid = request.get("statepoint_grid") or ""
    declared = request.get("fidelity") or request.get("physics_fidelity") or ""

    if declared == "strict":
        staged = False
        if not request.get("statepoint_grid"):
            grid = ""
    if grid in ("", "full"):
        grid = ""

    if declared == "A2" and not staged:
        return {}, ('"fidelity":"A2" but no staged multiplier is configured')
    if declared == "L3coarse" and not grid:
        return {}, ('"fidelity":"L3coarse" but no "statepoint_grid" was named')

    if grid:
        policy, fidelity = "L3coarse", "coarse10"
    elif staged:
        policy, fidelity = "A2", "staged_a2"
    else:
        policy, fidelity = "strict", "full_exact"
    # The floor a request cannot undo.
    if int(os.environ.get("RASBERY_GA_FEEDBACK_PASSES") or 0) > 0:
        policy, fidelity = "feedback_limited", "feedback_limited"

    if declared and declared not in (policy, fidelity):
        return {}, (f'"fidelity":"{declared}" but this case resolves to {policy}')
    # FAKE_RASBERY_LIE_FIDELITY: the receipt reports `strict` whatever the case
    # actually did -- a binary that ECHOES the fidelity contract instead of
    # applying it, which is the defect WP10.3's per-case audit exists to catch
    # and the one that leaves no other trace anywhere.
    if os.environ.get("FAKE_RASBERY_LIE_FIDELITY"):
        policy, fidelity = "strict", "full_exact"
    return {"policy": policy, "physics_fidelity": fidelity,
            "statepoint_grid": grid or "full",
            "acceptance_eligible": policy == "strict",
            "fidelity_declared": declared or None,
            "promoted_from": request.get("promoted_from") or None}, None


def run_case(deck: str, output: str, mode: str, *, wave_id: int, index: int,
             receipts: bool, fidelity: dict | None = None, key: str | None = None) -> bool:
    """Touch the output and report.  False when the case FAILED."""
    if poisoned(deck):
        die_on(deck)
    failed = bool(os.environ.get("FAKE_RASBERY_FAIL")) and \
        os.environ["FAKE_RASBERY_FAIL"] in deck
    # A DECK THAT EXISTS AND DOES NOT PARSE FAILS THE CASE.  The real binary
    # reaches this through IO::ReadInput throwing inside runOneCase's try: ONE
    # case fails, the process keeps answering, and that is the failure-isolation
    # path WP11's soak injects a poisoned deck to exercise.
    #
    # EXISTS AND does not parse, both halves deliberately.  Most of this fake's
    # callers name decks that were never created -- the dispatcher contract
    # tests care about queue accounting and never write a deck at all -- so a
    # missing file has always meant "not the thing under test" here and still
    # does.  A file that is on disk and is not JSON is unambiguous.
    deck_error = None
    if not failed and Path(deck).is_file():
        try:
            json.loads(Path(deck).read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            failed = True
            deck_error = f"cannot parse input file: {deck}: {exc}"
    if not failed:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("fake", encoding="utf-8")
    resolved = fidelity or {
        "policy": declared_policy()[0], "physics_fidelity": declared_policy()[1],
        "statepoint_grid": "full",
        "acceptance_eligible": declared_policy()[0] == "strict",
        "fidelity_declared": None, "promoted_from": None}
    # WP10.5.  THE DRIVER'S OWN TAG, which the fake did not print.
    #
    # This is the second time the same hole cost a session.  WP10.4: the fake
    # emitted only [EVALUATOR][CASE], so nothing exercised the tag whose
    # `physics_fidelity` spelling was wrong.  WP10.5: the same silence hid that
    # [RASBERY][CASE] carried no per-case identifier, so exact_audit could not
    # resolve it to a declaration and reported 82 cases as undeclared on 181.
    # A tag the harness never prints is a tag the harness cannot defend, so the
    # fake now prints BOTH -- and unconditionally, exactly as Driver.h does
    # (the real one is not behind a telemetry gate either).
    if not failed:
        emit("  [RASBERY][CASE] " + json.dumps({
            "schema_version": 6,
            "case_key": f"fake-{Path(deck).stem}",
            "key_schema": "casekey/v1",
            "core_op": "op",
            "deck_digest": "d", "env_digest": "e", "env_set": "~",
            "xslib_digest": "x", "xslib_policy": "cached",
            "warm_start_token": "~", "code_sha": "sha",
            "fidelity": resolved.get("physics_fidelity"),
            "physics_fidelity": resolved.get("physics_fidelity"),
            "policy": resolved.get("policy"),
            "result_mode": mode or "full",
            # The identifier.  Unique per case by the evaluator's own wave
            # namespace rule, and the only name both halves can agree on.
            "output": output,
            "warm_start": "cold",
            "statepoint_grid": resolved.get("statepoint_grid"),
            "acceptance_eligible": resolved.get("acceptance_eligible"),
            "fidelity_declared": resolved.get("fidelity_declared"),
            "promoted_from": resolved.get("promoted_from")}))
    if receipts:
        receipt = {
            "wave_id": wave_id, "case": index, "key": key,
            "case_key": f"fake-{Path(deck).stem}",
            "deck": deck, "output": output, "result_mode": mode or "full",
            "status": "failed" if failed else "ok",
            "exit_code": 1 if failed else 0,
            "digest": None if failed else f"{abs(hash(deck)) & 0xFFFFFFFFFFFFFFFF:016x}"}
        # WP10.3.  Nulls, not defaults, on a case that folded no receipt: "no
        # fidelity to report" and "ran strict" must not look the same.
        receipt.update(resolved if not failed else {
            "policy": None, "physics_fidelity": None, "statepoint_grid": None,
            "acceptance_eligible": None,
            "fidelity_declared": resolved.get("fidelity_declared"),
            "promoted_from": resolved.get("promoted_from")})
        receipt.update({
            "statepoints": 3, "outers": 30, "th_updates": 3, "slot": index, "lane": 0,
            "wall_s": 0.001, "teardown_ms": 0.1, "isolation_check": False,
            "error": (deck_error or "fake failure") if failed else None})
        emit("[RASBERY][EVALUATOR][CASE] " + json.dumps(receipt))
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


def emit_mem(wave_id: int, cases: int) -> None:
    """WP10.4 `[RASBERY][EVALUATOR][MEM]`, the shape EvaluatorServer prints.

    The fake carries it for the same reason it carries every other receipt the
    harness asserts on: `tools/soak_run.py` reads this line to ATTRIBUTE an RSS
    finding to a container, and a reader exercised only against a stream that
    never contains the line is a reader nobody has run.  That is exactly how the
    `physics_fidelity` mismatch survived -- the fake never printed the tag that
    was wrong.  Flat numbers on purpose: a healthy process is what the default
    fake models, and FAKE_RASBERY_MEM_GROWTH_MB drives the other case.
    """
    grow = float(os.environ.get("FAKE_RASBERY_MEM_GROWTH_MB") or 0.0)
    emit("[RASBERY][EVALUATOR][MEM] " + json.dumps({
        "wave_id": wave_id,
        "rss_mb": 100.0 + grow * wave_id,
        "rss_delta_mb": grow,
        "rss_since_first_mb": grow * wave_id,
        "rss_peak_mb": 100.0 + grow * wave_id,
        "rss_readable": True,
        "live_cases": 0,
        "cache_entries": {"xslib": 1, "xslib_digest": 1, "cohorts": 1,
                          "quadratures": 1, "pin_records": 0, "digest_memo": 2,
                          "case_samples": cases},
        "cache_bytes": {"xslib": 34_000_000},
        "evictions": {"xslib": 0, "xslib_digest": 0, "cohort": 0,
                      "digest_memo_clears": 0},
        "cuda_host_bytes": 0}))


def run_evaluator(argv: list[str]) -> int:
    width = int(argv[argv.index("--batch-mode") + 1])
    result_mode = argv[argv.index("--result") + 1] if "--result" in argv else "full"
    print_declarations(result_mode)
    if os.environ.get("FAKE_RASBERY_NO_READY"):
        return 1
    emit("[RASBERY][EVALUATOR][READY] " + json.dumps({
        "request_stream": "-", "batch_width": width, "result_mode": result_mode,
        "physics_fidelity": declared_policy()[1],
        # WP10.3.  The DEFAULT a case inherits and the FLOOR it cannot climb
        # above -- two facts that used to be one, and a client reads
        # `fidelity_per_case` to know a per-case declaration will be honoured
        # rather than refused.
        "fidelity_per_case": True,
        "fidelity_default": declared_policy()[0],
        "fidelity_floor": ("feedback_limited"
                           if int(os.environ.get("RASBERY_GA_FEEDBACK_PASSES") or 0) > 0
                           else "strict"),
        "statepoint_grid_default": "full",
        "idle_timeout_s": -1.0, "isolation_check": False}))

    cases = failed = waves = refused = overrides = promotions = 0
    exit_code = 0
    #: `op:case` / `op:promote` lines collected until a wave line runs them,
    #: exactly as EvaluatorServer::run does.
    pending: list[dict] = []
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
            refused += 1
            continue
        op = request.get("op")
        if op == "shutdown":
            break
        if op in ("case", "promote"):
            if op == "promote":
                request.setdefault("fidelity", "strict")
                request.setdefault("statepoint_grid", "full")
                request.setdefault("result_mode", "full")
                if not request.get("promoted_from"):
                    emit('[RASBERY][EVALUATOR][REFUSED] {"what":"promote needs '
                         'promoted_from","line":' + json.dumps(line) + "}")
                    exit_code = 2
                    refused += 1
                    continue
            pending.append(request)
            continue
        if op not in ("wave", "run"):
            emit('[RASBERY][EVALUATOR][REFUSED] {"what":"unknown op","line":'
                 + json.dumps(line) + "}")
            exit_code = 2
            refused += 1
            continue
        wave_id = int(request.get("wave_id", waves + 1))
        inline, pending = pending, []
        if inline:
            resolved: list[tuple[dict, dict]] = []
            wave_error = None
            for case in inline:
                # The wave's fidelity is a DEFAULT for cases that named none.
                merged = dict(case)
                for field in ("fidelity", "physics_fidelity", "statepoint_grid"):
                    if not merged.get(field) and request.get(field):
                        merged[field] = request[field]
                fidelity, error = resolve_case_fidelity(merged)
                if error is not None:
                    wave_error = error
                    break
                resolved.append((merged, fidelity))
            if wave_error is not None:
                emit('[RASBERY][EVALUATOR][REFUSED] {"what":' + json.dumps(wave_error)
                     + ',"line":' + json.dumps(line) + "}")
                exit_code = 2
                refused += 1
                continue
            host = min(int(os.environ.get("RASBERY_BATCH_HOST_THREADS", width)),
                       width, len(resolved))
            emit("[RASBERY][EVALUATOR][WAVE_START] " + json.dumps({
                "wave_id": wave_id, "jobs": len(resolved), "arena_width": width,
                "host_threads": host, "visible_cpus": 24,
                "result_mode": result_mode, "process_reused": waves > 0}))
            print_batch_host(len(resolved), width, host)
            ok = 0
            for index, (case, fidelity) in enumerate(resolved):
                if fidelity["policy"] != declared_policy()[0] or \
                        fidelity["statepoint_grid"] != "full":
                    overrides += 1
                if case.get("op") == "promote":
                    promotions += 1
                if run_case(case["deck"], case["output"],
                            case.get("result_mode", result_mode),
                            wave_id=wave_id, index=index, receipts=True,
                            fidelity=fidelity, key=case.get("key")):
                    ok += 1
                else:
                    failed += 1
                    exit_code = 1
            cases += len(resolved)
            waves += 1
            print_wave_counters(len(resolved), width)
            emit("[RASBERY][EVALUATOR][WAVE] " + json.dumps({
                "wave_id": wave_id, "jobs": len(resolved), "ok": ok,
                "failed": len(resolved) - ok, "wall_s": 0.01,
                "cases_per_hour": 3600.0 * len(resolved) / 0.01,
                "process_reused": waves > 1, "xslib_loads": 1,
                "xslib_hits": cases - 1, "pin_live_ranges": 0,
                "isolation_match": None}))
            emit_mem(wave_id, cases)
            continue
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
        emit_mem(wave_id, cases)

    # Teardown, in the order main.cpp's evaluator branch uses it: the arena is
    # released ONCE, here, which is why the occupancy receipt is a shutdown
    # receipt in this mode and a per-process one in the other.
    print_occupancy(cases, width)
    emit('[RASBERY][XSLIB_CACHE] {"loads":1,"hits":%d,"waits":0,"entries":1,'
         '"bytes":0,"lock_wait_ms":0,"enabled":true}' % max(0, cases - 1))
    # The three receipts a WP11 soak asserts to be zero and cannot get from the
    # evaluator line: the capture arbiter's two and the GPU-full fallback tally.
    # FAKE_RASBERY_NO_ARBITER drops this line entirely -- an OLDER binary, or a
    # build with the arbiter compiled out.  The soak must then REFUSE rather
    # than record zero: a counter that was never printed is not a counter that
    # was zero, and telling those two apart is the soak's whole claim.
    if not os.environ.get("FAKE_RASBERY_NO_ARBITER"):
        emit('[RASBERY][CUDA][CAPTURE_ARBITER] {"tag":"shutdown","enabled":1,'
             '"capture_windows":4,"capture_wait_us":0,"alloc_windows":2,'
             '"alloc_overlapped":0,"alloc_blocked":0,"alloc_wait_us":0,'
             '"alloc_in_capture":%d,"captures_unwound":%d}'
             % (_injected("FAKE_RASBERY_ALLOC_IN_CAPTURE"),
                _injected("FAKE_RASBERY_CAPTURES_UNWOUND")))
    fallbacks = _injected("FAKE_RASBERY_HOST_FALLBACKS")
    # WP10.7: `outer_fallbacks` is emitted because tools/soak_run.py now asserts
    # it like every other subsystem.  A field the real binary prints and this
    # stand-in does not is a field no test in this tree ever drives -- which is
    # exactly how the 238 arm-A run reached `outer_fallbacks:9` with the soak's
    # verdict never looking at the number.  It rides FAKE_RASBERY_HOST_FALLBACKS
    # with cmfd so one knob still exercises the whole gated list.
    emit('[RASBERY][GPU_FULL] {"gpu_full":%s,"cmfd_fallbacks":%d,"outer_fallbacks":%d,'
         '"nodal_fallbacks":0,'
         '"xsrecon_fallbacks":0,"flatxs_fallbacks":0,"xe_fallbacks":0,'
         '"ppr_fallbacks":0,"cram_fallbacks":0,"contract_pass":%s}'
         % ("true" if os.environ.get("RASBERY_GPU_FULL") not in (None, "", "0")
            else "false", fallbacks, fallbacks,
            "false" if fallbacks else "true"))
    emit("[RASBERY][EVALUATOR] " + json.dumps({
        "cases": cases, "ok": cases - failed, "failed": failed, "refused": refused,
        "generations": waves, "batch_width": width, "process_uptime_s": 0.1,
        "drive_s": 0.05, "cuda_context_reuse": max(0, cases - 1),
        "arena_releases": 1, "arena_standups": 1 if cases else 0,
        "slot_admissions": cases, "slot_duplicates": 0, "slot_stale_tenants": 0,
        "slot_double_releases": 0, "xslib_loads": 1, "xslib_hits": max(0, cases - 1),
        "library_loads": 1, "geometry_builds": cases,
        "cohort_builds": 1 if cases else 0, "cohort_hits": max(0, cases - 1),
        "pin_live_ranges_between_waves": 0,
        "fidelity_overrides": overrides, "promotions": promotions,
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
