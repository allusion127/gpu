#!/usr/bin/env python3
"""WP11 -- the long-stability soak, and the receipts that must be zero at exit.

WHAT A SOAK IS FOR, AND WHY A THROUGHPUT RUN IS NOT ONE.  Every number the
campaign has measured came off a wave or two.  The defects WP8-WP10 can
introduce are not wave-shaped: a slot the refill ledger hands out twice, a
pinned host range whose lease outlived its Driver, a graph capture that unwound
and left an allocation behind, a cohort key that started covering something a
candidate changes.  Each of those is invisible for one generation and fatal over
twenty, and each is already COUNTED by a receipt this tree prints.  So the soak
does not measure anything new.  It runs long enough for the counters to have
something to say, and then it reads them.

THE PLAN'S LIST, VERBATIM (Sec WP11 "종료 시 0이어야 하는 receipt"), and where
each one actually comes from:

    duplicates                  [REFILL].duplicates / [EVALUATOR].slot_duplicates
    stale_tenants               [REFILL].stale_tenants / [EVALUATOR].slot_stale_tenants
    double_releases             [REFILL].double_releases / [EVALUATOR].slot_double_releases
    alloc_in_capture            [CUDA][CAPTURE_ARBITER].alloc_in_capture
    captures_unwound            [CUDA][CAPTURE_ARBITER].captures_unwound
    graph/cmfd/nodal/flatxs/ppr fallbacks    [GPU_FULL].*_fallbacks, under RASBERY_GPU_FULL=1
    queue duplicate claims      [EVALUATOR].slot_duplicates (one queue, one counter)
    output collisions           this driver's own: two cases on one --raso
    cross_case_digest_mismatch  [EVALUATOR].isolation_mismatches
    pin_live_ranges_between_waves           [EVALUATOR], asserted per wave by the binary

TWO OF THEM ARE CONDITIONAL AND SAYING SO IS THE POINT.  The GPU-full fallbacks
are only a zero-assertion when RASBERY_GPU_FULL=1 -- without it a fallback is
legal and counting it as a failure would make the soak refuse a legitimate arm.
And `restarts` is bounded by `--expect-restarts`, which is ZERO by default -- not
because restarts are impossible but because the poison this soak plants is a
deck that does not PARSE, and that is supposed to fail one case and leave the
process answering.  A restart there is the poison taking sixty-three other
candidates down with it, which is the failure-isolation defect the poison exists
to find.  A run that means to kill the child raises the bound and thereby says
so.

WHAT IS MEASURED RATHER THAN ASSERTED.

  c/h per generation      A drift budget, not a target.  The first generation is
                          the reference; every later one must stay within
                          --drift (default 3 %) of the MEDIAN.  A soak whose
                          throughput decays 0.5 % per generation ends 10 % down
                          and every individual step looks like noise -- which is
                          exactly what a slow leak looks like from inside.
  VRAM                    nvidia-smi, sampled between generations.  A LEAK IS A
                          SLOPE, not a level: warm plateau is expected and high,
                          so the test is MB per generation over the second half
                          of the run, after the caches have stopped growing.
  host RSS                /proc/<pid>/status VmRSS, same rule.  Unavailable off
                          Linux, and the report says `null` rather than 0 --
                          "not measured" and "measured zero" are different
                          claims and only one of them is evidence.

THE WORKLOAD.  N generations x W cases, and every case in it is doing something
the plan asks a soak to exercise:

    mixed light/full      --light-fraction of each generation is `light`, the
                          rest `full`.  Both write; only one writes HDF5.
    warm-start chain      case i of generation g warm-starts from case i of
                          generation g-1 and saves its own state for g+1.  A
                          chain is what finds a warm-state file that is written
                          but never closed.
    mixed fidelity        --screen-fraction of each generation runs L3coarse on
                          the coarse burnup grid (WP10.3), and one promoted
                          strict re-run per generation links back to the screen
                          it replaces.  A soak on ONE fidelity would never
                          exercise the path where two cases in one wave converge
                          differently.
    one poisoned case     a deck that does not parse.  The real binary reaches
                          this through IO::ReadInput throwing inside
                          runOneCase's try: ONE case fails, the process keeps
                          answering, and the wave receipt says 63/64.  If it
                          instead takes the process down, the dispatcher
                          restarts it and `restarts` exceeds --expect-restarts,
                          which is the assertion.

USE

    python tools/soak_run.py --deck kngr_238.json --workdir /tmp/soak \\
        --binary ./RASBERY --generations 20 --width 64

    # the contract-test shape: no GPU, no CUDA, no 40 s case
    python tools/soak_run.py --deck any.json --workdir /tmp/soak \\
        --command "python tools/fake_rasbery_child.py" --generations 3 --width 4

Exit status is 0 only when every zero-receipt is zero, the drift budget holds
and no leak slope is above its threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import exact_audit  # noqa: E402
from run_multi_gpu_batch import EvaluatorSession  # noqa: E402

# ---------------------------------------------------------------------------
# The zero-receipt table.  ONE list, and the report prints it whole -- including
# the ones that were zero, because "we checked and it was zero" and "we did not
# check" are different statements and a report that only shows failures cannot
# distinguish them.
# ---------------------------------------------------------------------------

#: (name, receipt tag, field).  Read from the session's whole transcript.
ZERO_RECEIPTS: tuple[tuple[str, str, str], ...] = (
    ("duplicates", "[RASBERY][EVALUATOR]", "slot_duplicates"),
    ("stale_tenants", "[RASBERY][EVALUATOR]", "slot_stale_tenants"),
    ("double_releases", "[RASBERY][EVALUATOR]", "slot_double_releases"),
    ("refill_duplicates", "[RASBERY][REFILL]", "duplicates"),
    ("refill_stale_tenants", "[RASBERY][REFILL]", "stale_tenants"),
    ("refill_double_releases", "[RASBERY][REFILL]", "double_releases"),
    ("alloc_in_capture", "[RASBERY][CUDA][CAPTURE_ARBITER]", "alloc_in_capture"),
    ("captures_unwound", "[RASBERY][CUDA][CAPTURE_ARBITER]", "captures_unwound"),
    ("cross_case_digest_mismatch", "[RASBERY][EVALUATOR]", "isolation_mismatches"),
    ("pin_live_ranges_between_waves", "[RASBERY][EVALUATOR]",
     "pin_live_ranges_between_waves"),
)

#: Only a zero-assertion under RASBERY_GPU_FULL=1.  Without the gate a fallback
#: is a legal thing for the binary to do and failing the soak on one would make
#: it refuse an arm nobody said was wrong.
GPU_FULL_FALLBACKS: tuple[str, ...] = (
    "cmfd_fallbacks", "nodal_fallbacks", "xsrecon_fallbacks", "flatxs_fallbacks",
    "xe_fallbacks", "ppr_fallbacks", "cram_fallbacks", "graph_fallbacks",
)


def receipts_of(text: str, tag: str) -> list[dict]:
    """Every `<tag> {json}` object in *text*.

    Tag matching is on the literal tag followed by whitespace and `{`, so
    `[RASBERY][EVALUATOR]` cannot swallow `[RASBERY][EVALUATOR][CASE]` -- the
    latter has `[` where this needs a space, which is the same discrimination
    run_multi_gpu_batch's EVALUATOR_PROCESS regex makes.
    """
    out: list[dict] = []
    start = 0
    while True:
        index = text.find(tag, start)
        if index < 0:
            return out
        cursor = index + len(tag)
        start = cursor
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            continue
        end = text.find("\n", cursor)
        blob = text[cursor:end if end >= 0 else len(text)]
        try:
            value = json.loads(blob)
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append(value)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_vram_mb(gpu: str) -> float | None:
    """Used VRAM on *gpu*, MB, or None when nvidia-smi cannot say.

    None rather than 0.0 on every failure path.  A soak that reported 0 MB
    because nvidia-smi was missing would report a perfectly flat VRAM trace and
    pass its leak check for the one reason that proves nothing.
    """
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(gpu)],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    line = out.stdout.strip().splitlines()
    if not line:
        return None
    try:
        return float(line[0].strip())
    except ValueError:
        return None


def sample_rss_mb(pid: int | None) -> float | None:
    """The child's resident set, MB, or None where /proc does not exist."""
    if pid is None:
        return None
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024.0
                except ValueError:
                    return None
    return None


def leak_slope_mb_per_generation(series: Sequence[float | None]) -> float | None:
    """MB per generation over the SECOND HALF of *series*, or None.

    THE SECOND HALF, deliberately.  A warm plateau is expected and it is steep:
    the library loads, the arenas stand up, the graph cache fills, and the first
    few generations climb for reasons that are the design working.  A slope
    fitted over the whole run would report that climb as a leak on every healthy
    soak, and the threshold would then have to be set so high that it caught
    nothing.  What a leak looks like is a line that is still climbing after the
    caches have stopped, which is what the second half is.
    """
    values = [(i, v) for i, v in enumerate(series) if v is not None]
    if len(values) < 4:
        return None
    tail = values[len(values) // 2:]
    if len(tail) < 2:
        return None
    xs = [float(i) for i, _ in tail]
    ys = [v for _, v in tail]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


# ---------------------------------------------------------------------------
# The workload
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    index: int
    cases: int = 0
    ok: int = 0
    failed: int = 0
    wall_s: float = 0.0
    cases_per_hour: float = 0.0
    vram_mb: float | None = None
    rss_mb: float | None = None
    poisoned: int = 0
    promotions: int = 0
    screens: int = 0
    alive: bool = True
    refused: list[dict] = field(default_factory=list)
    fidelity_problems: list[str] = field(default_factory=list)


def build_generation(*, generation: int, width: int, deck: Path, workdir: Path,
                     light_fraction: float, screen_fraction: float,
                     poison: bool, promote: bool, bad_deck: Path
                     ) -> tuple[list[dict], dict[str, str]]:
    """One generation's case requests, and the fidelity each was DECLARED at.

    The declaration map is returned beside the requests because the per-case
    audit needs it: after WP10.3 "what was this wave declared as" has as many
    answers as the wave has cases, and reconstructing them from the receipts
    afterwards would be asking the run to grade its own homework.
    """
    cases: list[dict] = []
    declared: dict[str, str] = {}
    n_light = int(round(width * light_fraction))
    n_screen = int(round(width * screen_fraction))
    for i in range(width):
        key = f"g{generation:04d}c{i:04d}"
        request: dict[str, object] = {
            "op": "case",
            "key": key,
            "deck": str(deck),
            # One output path per (generation, case).  Reusing a path ACROSS
            # generations would be legal for the evaluator (it scopes the
            # namespace rule to the wave) and would hide an output collision
            # from this driver, which is one of the things it is here to find.
            "output": str(workdir / "out" / f"{key}.h5"),
            "result_mode": "light" if i < n_light else "full",
            # The chain.  Generation 0 has no parent and starts cold, which is
            # also the control: if the chain is what leaks, generation 0 is the
            # only clean one and the trace says so.
            "save_warm_state": str(workdir / "warm" / f"{key}.warm"),
        }
        if generation > 0:
            request["warm_start_from"] = str(
                workdir / "warm" / f"g{generation - 1:04d}c{i:04d}.warm")
        if i < n_screen:
            request["fidelity"] = "L3coarse"
            request["statepoint_grid"] = "coarse"
            declared[key] = "L3coarse"
        else:
            declared[key] = "strict"
            request["fidelity"] = "strict"
        cases.append(request)

    if poison:
        key = f"g{generation:04d}poison"
        cases.append({"op": "case", "key": key, "deck": str(bad_deck),
                      "output": str(workdir / "out" / f"{key}.h5"),
                      "result_mode": "light"})
        # No declaration: the poisoned case is EXPECTED to fold no receipt, and
        # audit_case_fidelity skips a failed case that reported none.  Declaring
        # one would ask the audit to grade a case that never ran.
    if promote and n_screen > 0:
        parent = f"g{generation:04d}c0000"
        key = f"g{generation:04d}promote"
        cases.append({
            "op": "promote",
            "key": key,
            "deck": str(deck),
            "output": str(workdir / "out" / f"{key}.h5"),
            # The link.  `promoted_from` is the SCREENING case's key here rather
            # than its case_key, because the soak drives its own requests and
            # knows what it sent; a GA reads the case_key back off the screen's
            # receipt and sends that.  Both are strings the receipts carry.
            "promoted_from": parent,
        })
        declared[key] = "strict"
    return cases, declared


def write_bad_deck(path: Path) -> None:
    """A deck that cannot be parsed.

    NOT a deck that is merely wrong -- a deck the JSON parser itself refuses, so
    the failure lands in IO::ReadInput inside runOneCase's try and exercises the
    ONE path that matters: a case that throws must fail alone.  A semantically
    bad deck could be caught later, in a place with different unwinding, and
    would be testing something else.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schedule": [ this is not json', encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    lines = ["# RASBERY WP11 soak report", ""]
    lines.append(f"- verdict: **{'PASS' if report['pass'] else 'FAIL'}**")
    lines.append(f"- generations: {report['generations']} x {report['width']} "
                 f"= {report['cases_requested']} cases requested, "
                 f"{report['cases_reported']} reported")
    lines.append(f"- wall: {report['wall_s']:.1f} s")
    lines.append(f"- restarts: {report['restarts']} "
                 f"(injected poison: {report['poisoned']})")
    lines.append(f"- command: `{report['command']}`")
    lines.append("")
    lines.append("## Zero receipts")
    lines.append("")
    lines.append("| receipt | value | required |")
    lines.append("|---|---|---|")
    for name, value in report["zero_receipts"].items():
        required = "0" if name not in report["not_asserted"] else "not asserted"
        shown = "n/a" if value is None else value
        lines.append(f"| {name} | {shown} | {required} |")
    if report["not_asserted"]:
        lines.append("")
        lines.append("> Not asserted, and why: "
                     + "; ".join(f"`{k}` -- {v}"
                                 for k, v in report["not_asserted"].items()))
    lines.append("")
    lines.append("## Throughput and growth")
    lines.append("")
    lines.append("| gen | cases | ok | failed | wall s | c/h | VRAM MB | RSS MB |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for gen in report["per_generation"]:
        vram = "-" if gen["vram_mb"] is None else format(gen["vram_mb"], ".0f")
        rss = "-" if gen["rss_mb"] is None else format(gen["rss_mb"], ".0f")
        lines.append(
            f"| {gen['index']} | {gen['cases']} | {gen['ok']} | {gen['failed']} | "
            f"{gen['wall_s']:.2f} | {gen['cases_per_hour']:.1f} | {vram} | {rss} |")
    lines.append("")
    drift = report["throughput"]
    lines.append(f"- c/h median {drift['median']:.1f}, "
                 f"min {drift['min']:.1f}, max {drift['max']:.1f}, "
                 f"worst drift {drift['worst_drift']:.3%} "
                 f"(budget {drift['budget']:.1%})")
    for what in ("vram", "rss"):
        slope = report["growth"][what]["slope_mb_per_generation"]
        limit = report["growth"][what]["limit_mb_per_generation"]
        shown = "not measured" if slope is None else f"{slope:+.2f} MB/gen"
        lines.append(f"- {what} growth (second half): {shown}, limit {limit} MB/gen")
    if report["problems"]:
        lines.append("")
        lines.append("## Problems")
        lines.append("")
        for problem in report["problems"]:
            lines.append(f"- {problem}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, required=True,
                    help="the deck every case runs (one cohort, which is what the "
                         "plan's soak 1 asks for: 64 decks, ONE cohort, 10,000 cases)")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=None,
                    help="the RASBERY executable")
    ap.add_argument("--command", type=str, default=None,
                    help="the full child command instead of --binary (the contract "
                         "test passes the fake child here)")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--width", type=int, default=64,
                    help="cases per generation AND --batch-mode; the arena is one "
                         "allocation latched on the first wave")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--light-fraction", type=float, default=0.5)
    ap.add_argument("--screen-fraction", type=float, default=0.25,
                    help="fraction of each generation run at L3coarse on the coarse "
                         "burnup grid; 0 makes the soak single-fidelity, which does "
                         "not exercise WP10.3 at all")
    ap.add_argument("--no-poison", action="store_true",
                    help="do not inject a failing case per generation. Off by "
                         "default: a soak with no failure has tested the easy half")
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--drift", type=float, default=0.03,
                    help="allowed c/h deviation from the median, per generation")
    ap.add_argument("--vram-leak-mb", type=float, default=8.0,
                    help="MB per generation of VRAM growth over the run's second "
                         "half that counts as a leak")
    ap.add_argument("--rss-leak-mb", type=float, default=8.0)
    ap.add_argument("--max-restarts", type=int, default=3,
                    help="how many times the dispatcher may replace a dead child "
                         "before giving up; this is the RECOVERY budget")
    ap.add_argument("--expect-restarts", type=int, default=0,
                    help="how many restarts this run INTENDS to cause. The default "
                         "poison fails one case and leaves the process alive, so 0 "
                         "is the honest expectation and any restart is a finding")
    ap.add_argument("--report", type=Path, default=None,
                    help="JSON report path (default <workdir>/soak_report.json); the "
                         "markdown goes beside it with a .md suffix")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="an environment variable for the child, repeatable")
    ap.add_argument("--result", type=str, default="full",
                    help="the child's default --result; per-case modes override it")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    workdir: Path = args.workdir
    (workdir / "out").mkdir(parents=True, exist_ok=True)
    (workdir / "warm").mkdir(parents=True, exist_ok=True)
    (workdir / "log").mkdir(parents=True, exist_ok=True)
    bad_deck = workdir / "poison" / "unparseable.json"
    write_bad_deck(bad_deck)

    if args.command:
        # posix=False on Windows, because posix mode treats a backslash as an
        # escape and turns C:\Users\MK\python.exe into C:UsersMKpython.exe --
        # a path that does not exist, reported as "the child never became
        # ready", which is the least informative possible spelling of "your
        # command was mangled by the argument parser".  The soak itself runs on
        # Linux; its contract test runs here.
        command = shlex.split(args.command, posix=(os.name != "nt"))
        if os.name == "nt":
            command = [t[1:-1] if len(t) > 1 and t[0] == t[-1] == '"' else t
                       for t in command]
    elif args.binary:
        command = [str(args.binary)]
    else:
        print("soak_run: one of --binary or --command is required", file=sys.stderr)
        return 2
    command += ["--evaluator-jsonl", "-", "--batch-mode", str(args.width),
                "--result", args.result]

    env = dict(os.environ)
    for item in args.set:
        key, _, value = item.partition("=")
        env[key] = value
    env.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    gpu_full = env.get("RASBERY_GPU_FULL") not in (None, "", "0")

    session = EvaluatorSession(command=command, env=env, cwd=str(ROOT),
                               log_path=workdir / "log" / "soak.log",
                               max_restarts=args.max_restarts)
    started = time.time()
    if not session.start():
        print("soak_run: the child never became ready", file=sys.stderr)
        return 2

    generations: list[GenerationResult] = []
    declared_all: dict[str, str] = {}
    cases_requested = 0
    poisoned = 0
    transcript = session.preamble

    for g in range(args.generations):
        cases, declared = build_generation(
            generation=g, width=args.width, deck=args.deck, workdir=workdir,
            light_fraction=args.light_fraction, screen_fraction=args.screen_fraction,
            poison=not args.no_poison, promote=not args.no_promote, bad_deck=bad_deck)
        declared_all.update(declared)
        cases_requested += len(cases)
        if not args.no_poison:
            poisoned += 1
        result = GenerationResult(index=g)
        result.poisoned = 0 if args.no_poison else 1
        result.promotions = 0 if args.no_promote else 1
        result.screens = sum(1 for c in cases if c.get("fidelity") == "L3coarse")

        if not session.alive and not session.restart():
            result.alive = False
            generations.append(result)
            break
        wave_start = time.time()
        outcome = session.wave(wave_id=g + 1, cases=cases)
        transcript += outcome.text
        result.wall_s = time.time() - wave_start
        result.alive = outcome.alive
        result.refused = list(outcome.refused)
        result.cases = len(outcome.cases)
        result.ok = sum(1 for c in outcome.cases if c.get("status") == "ok")
        result.failed = result.cases - result.ok
        if outcome.receipt:
            result.wall_s = float(outcome.receipt.get("wall_s", result.wall_s))
            result.cases_per_hour = float(outcome.receipt.get("cases_per_hour", 0.0))
        elif result.wall_s > 0:
            result.cases_per_hour = 3600.0 * result.cases / result.wall_s
        # The per-case fidelity audit, ON THIS GENERATION's declarations.  Done
        # here rather than at the end because a mixed wave's declarations are
        # per generation and folding them all into one map would let generation
        # 7's key answer for generation 3's case.
        result.fidelity_problems = exact_audit.audit_case_fidelity(
            outcome.text, declared, require_any=result.cases > 0)
        # VRAM/RSS BETWEEN generations, not during: a sample taken mid-wave
        # measures where the wave happened to be, and the question is what is
        # left behind when it is over.
        result.vram_mb = sample_vram_mb(args.gpu)
        result.rss_mb = sample_rss_mb(session.pid)
        generations.append(result)
        if not outcome.alive and not session.restart():
            break

    transcript += session.close()
    wall = time.time() - started

    # -- the assertions ----------------------------------------------------
    problems: list[str] = []
    zero_values: dict[str, int | None] = {}
    not_asserted: dict[str, str] = {}
    for name, tag, field_name in ZERO_RECEIPTS:
        found = [r.get(field_name) for r in receipts_of(transcript, tag)
                 if field_name in r]
        if not found:
            zero_values[name] = None
            not_asserted[name] = (f"no {tag} receipt carried {field_name!r} in this "
                                  "run's output")
            problems.append(
                f"{name}: no {tag} receipt carried {field_name!r}. A counter that was "
                "never printed is not a counter that was zero, and this soak's whole "
                "claim is that it read them.")
            continue
        worst = max(int(v) for v in found if isinstance(v, (int, float)))
        zero_values[name] = worst
        if worst != 0:
            problems.append(f"{name} = {worst}, must be 0 at exit")

    for field_name in GPU_FULL_FALLBACKS:
        found = [r.get(field_name) for r in receipts_of(transcript, "[RASBERY][GPU_FULL]")
                 if field_name in r]
        if not found:
            continue
        worst = max(int(v) for v in found if isinstance(v, (int, float)))
        zero_values[field_name] = worst
        if not gpu_full:
            not_asserted[field_name] = ("RASBERY_GPU_FULL is not set, so a host "
                                        "fallback is legal in this arm")
        elif worst != 0:
            problems.append(f"{field_name} = {worst} under RASBERY_GPU_FULL=1")

    # RESTARTS ARE BOUNDED BY WHAT WAS INJECTED, and this soak's injection is
    # zero.  The poison it plants is a deck that does not PARSE: the real binary
    # reaches that through IO::ReadInput throwing inside runOneCase's try, which
    # fails one case and leaves the process answering.  So a restart is not "the
    # poison working", it is the poison taking the process with it -- the exact
    # failure-isolation defect the poison is planted to find.  --expect-restarts
    # raises the bound for a run that deliberately kills the child (a CUDA abort
    # rehearsal), so the number is a stated intention rather than a slack budget.
    if session.restarts > args.expect_restarts:
        problems.append(
            f"the evaluator restarted {session.restarts} times, expecting at most "
            f"{args.expect_restarts}. The injected poison is a deck that fails its OWN "
            "case (EvaluatorServer::runOneCase catches the throw); a restart means it "
            "took the process down with it, and the sixty-three candidates that were "
            "in flight went with it.")

    # Output collisions: this driver's own bookkeeping, because the evaluator
    # scopes its namespace rule to a wave and would not see a cross-generation
    # reuse at all.
    outputs: dict[str, str] = {}
    for receipt in exact_audit.parse_case_receipts(transcript):
        out_path = receipt.get("output")
        key = receipt.get("key") or receipt.get("case_key") or ""
        if isinstance(out_path, str) and out_path:
            if out_path in outputs and outputs[out_path] != key:
                problems.append(
                    f"output collision: {out_path!r} was written by both "
                    f"{outputs[out_path]!r} and {key!r}")
            outputs[out_path] = key

    for gen in generations:
        for problem in gen.fidelity_problems:
            problems.append(f"generation {gen.index}: {problem}")
        for refusal in gen.refused:
            problems.append(f"generation {gen.index}: refused -- "
                            f"{refusal.get('what', refusal)}")

    # Throughput drift.
    rates = [g.cases_per_hour for g in generations if g.cases_per_hour > 0]
    median = statistics.median(rates) if rates else 0.0
    worst_drift = 0.0
    if median > 0:
        for gen in generations:
            if gen.cases_per_hour <= 0:
                continue
            drift = abs(gen.cases_per_hour - median) / median
            worst_drift = max(worst_drift, drift)
            if drift > args.drift:
                problems.append(
                    f"generation {gen.index}: {gen.cases_per_hour:.1f} c/h is "
                    f"{drift:.2%} off the median {median:.1f}, budget {args.drift:.1%}")

    growth = {}
    for what, series, limit in (
            ("vram", [g.vram_mb for g in generations], args.vram_leak_mb),
            ("rss", [g.rss_mb for g in generations], args.rss_leak_mb)):
        slope = leak_slope_mb_per_generation(series)
        growth[what] = {"slope_mb_per_generation": slope,
                        "limit_mb_per_generation": limit,
                        "samples": series}
        if slope is not None and slope > limit:
            problems.append(
                f"{what} grew {slope:.2f} MB/generation over the run's second half "
                f"(limit {limit}); after the warm plateau nothing should still be "
                "climbing")

    cases_reported = sum(g.cases for g in generations)
    if cases_reported < cases_requested:
        problems.append(
            f"{cases_requested - cases_reported} of {cases_requested} cases were never "
            "reported. A generation that silently lost candidates is not a generation.")

    report = {
        "schema": "rasbery-soak/v1",
        "pass": not problems,
        "command": " ".join(command),
        "generations": args.generations,
        "width": args.width,
        "cases_requested": cases_requested,
        "cases_reported": cases_reported,
        "poisoned": poisoned,
        "restarts": session.restarts,
        "expect_restarts": args.expect_restarts,
        "starts": session.starts,
        "returncode": session.returncode,
        "wall_s": wall,
        "gpu_full": gpu_full,
        "zero_receipts": zero_values,
        "not_asserted": not_asserted,
        "throughput": {
            "median": median,
            "min": min(rates) if rates else 0.0,
            "max": max(rates) if rates else 0.0,
            "worst_drift": worst_drift,
            "budget": args.drift,
        },
        "growth": growth,
        "per_generation": [
            {"index": g.index, "cases": g.cases, "ok": g.ok, "failed": g.failed,
             "wall_s": g.wall_s, "cases_per_hour": g.cases_per_hour,
             "vram_mb": g.vram_mb, "rss_mb": g.rss_mb, "screens": g.screens,
             "promotions": g.promotions, "poisoned": g.poisoned, "alive": g.alive}
            for g in generations],
        "problems": problems,
    }

    report_path = args.report or (workdir / "soak_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    report_path.with_suffix(".md").write_text(render_markdown(report),
                                              encoding="utf-8", newline="\n")
    print(f"soak: {'PASS' if report['pass'] else 'FAIL'} "
          f"({cases_reported}/{cases_requested} cases, {len(problems)} problems)")
    for problem in problems:
        print("  - " + problem)
    print(f"wrote {report_path} and {report_path.with_suffix('.md')}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
