#!/usr/bin/env python3
"""Drive the WP8 `--evaluator` mode, and price it against the launcher it replaces.

WHY THIS EXISTS AND NOT A ONE-LINER.  The claim WP8 has to support is not "the
evaluator ran"; it is "G generations in ONE process produce the SAME per-case
digest as G generations in G processes, and cost less wall".  Both halves of
that need the same tool: a client that can run either shape, read the per-case
receipts out of both, and put the digests side by side.  Grepping two logs by
hand gets the wall clock right and the digest comparison wrong, because the
baseline shape's `[RASBERY][TRAJECTORY]` lines carry a slot, not a deck.

THREE ARMS, and the middle one is the honest control.

  persistent      one process, G waves.  The thing under test.
  per-generation  G processes, one wave each.  The SAME code path -- still
                  --evaluator, still one wave of M -- differing only in where
                  the process boundary is.  This is the arm the digest
                  comparison is exact against, because both arms emit
                  [RASBERY][EVALUATOR][CASE] with a deck AND a digest.
  chunked-jobs    G processes, each `--jobs <manifest> --batch-mode M`.  Today's
                  launcher shape, and the arm the WALL comparison is against.
                  Its digests are only comparable as a multiset (the batch
                  branch's trajectory receipt names a slot, not a deck), which
                  is why it is not the digest control.

Usage:

    tools/evaluator_client.py --batch-mode 64 \\
        --generation g1.jobs --generation g2.jobs --generation g3.jobs \\
        --arm persistent --arm per-generation --arm chunked-jobs \\
        --workdir wp8_run -- ./build/RASBERY

Everything after `--` is the executable and any prefix (taskset, numactl).  The
child runs in `--cwd` if given, because a deck's cross-section path and restart
namespace are relative to the deck directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

CASE_LINE = re.compile(r"^\[RASBERY\]\[EVALUATOR\]\[CASE\]\s+(\{.*\})\s*$", re.M)
WAVE_LINE = re.compile(r"^\[RASBERY\]\[EVALUATOR\]\[WAVE\]\s+(\{.*\})\s*$", re.M)
PROC_LINE = re.compile(r"^\[RASBERY\]\[EVALUATOR\]\s+(\{.*\})\s*$", re.M)
TRAJ_LINE = re.compile(r"^\[RASBERY\]\[TRAJECTORY\]\s+(\{.*\})\s*$", re.M)
LEDGER_LINE = re.compile(r"^\[RASBERY\]\[PROCESS\]\s+(\{.*\})\s*$", re.M)
FAIL_LINE = re.compile(r"^\[RASBERY\]\[(?:FAIL|EVALUATOR\]\[REFUSED)\]", re.M)


def loads(matches: Sequence[str]) -> list[dict]:
    out = []
    for text in matches:
        try:
            out.append(json.loads(text))
        except json.JSONDecodeError:
            pass
    return out


class ArmResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.wall = 0.0
        self.processes = 0
        self.cases = 0
        self.failed = 0
        self.refused = 0
        self.digests: dict[str, str] = {}      # deck -> digest (persistent arms)
        self.digest_bag: Counter[str] = Counter()
        self.waves: list[dict] = []
        self.process_receipts: list[dict] = []
        self.outside_drive = 0.0               # process image + CUDA, summed

    def absorb(self, text: str) -> None:
        for case in loads(CASE_LINE.findall(text)):
            if case.get("isolation_check"):
                continue
            self.cases += 1
            if case.get("status") != "ok":
                self.failed += 1
            digest = case.get("digest")
            if digest:
                self.digest_bag[digest] += 1
                deck = os.path.basename(str(case.get("deck", "")))
                # A deck may legitimately appear in more than one generation;
                # keep the FIRST and let the caller see a conflict as a mismatch.
                self.digests.setdefault(deck, digest)
        self.waves += loads(WAVE_LINE.findall(text))
        receipts = loads(PROC_LINE.findall(text))
        self.process_receipts += receipts
        for r in receipts:
            self.refused += int(r.get("refused", 0))
        # The chunked arm has no per-case receipt; fall back to the multiset.
        if not CASE_LINE.search(text):
            for traj in loads(TRAJ_LINE.findall(text)):
                digest = traj.get("digest")
                if digest:
                    self.digest_bag[digest] += 1
                    self.cases += 1
            self.failed += len(FAIL_LINE.findall(text))
        for ledger in loads(LEDGER_LINE.findall(text)):
            # What a case pays OUTSIDE Drive(): the re-exec image plus everything
            # main() does before and after the decks.  Summed over processes,
            # this is the number the persistent arm is supposed to delete.
            exec_s = ledger.get("exec_s") or 0.0
            self.outside_drive += float(exec_s) + float(ledger.get("pre_drive_s", 0.0)) \
                + float(ledger.get("post_drive_s", 0.0))


def run(command: list[str], *, stdin_text: str | None, log: Path, cwd: str | None,
        env: dict) -> tuple[str, int, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    child = subprocess.run(  # noqa: S603
        command,
        input=stdin_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=env,
        check=False,
    )
    wall = time.monotonic() - started
    log.write_text(child.stdout or "", encoding="utf-8", newline="\n")
    return child.stdout or "", child.returncode, wall


def arm_persistent(args, executable, generations, env) -> ArmResult:
    """One process, G waves.  The thing under test."""
    result = ArmResult("persistent")
    stream = "".join(
        json.dumps({"op": "wave", "wave_id": i + 1, "jobs_manifest": str(m.resolve())}) + "\n"
        for i, m in enumerate(generations)
    ) + json.dumps({"op": "shutdown"}) + "\n"
    command = list(executable) + ["--evaluator-jsonl", "-", "--batch-mode", str(args.batch_mode)]
    if args.result:
        command += ["--result", args.result]
    if args.isolation_check:
        command += ["--evaluator-isolation-check"]
    text, code, wall = run(command, stdin_text=stream,
                           log=Path(args.workdir) / "persistent.log", cwd=args.cwd, env=env)
    result.absorb(text)
    result.wall = wall
    result.processes = 1
    if code != 0:
        result.failed = max(result.failed, 1)
    return result


def arm_per_generation(args, executable, generations, env) -> ArmResult:
    """G processes, one wave each.  Same code path; only the boundary moves."""
    result = ArmResult("per-generation")
    for i, manifest in enumerate(generations):
        stream = (json.dumps({"op": "wave", "wave_id": i + 1,
                              "jobs_manifest": str(manifest.resolve())}) + "\n"
                  + json.dumps({"op": "shutdown"}) + "\n")
        command = list(executable) + ["--evaluator-jsonl", "-",
                                      "--batch-mode", str(args.batch_mode)]
        if args.result:
            command += ["--result", args.result]
        text, code, wall = run(command, stdin_text=stream,
                               log=Path(args.workdir) / f"per_generation.{i + 1:02d}.log",
                               cwd=args.cwd, env=env)
        result.absorb(text)
        result.wall += wall
        result.processes += 1
        if code != 0:
            result.failed = max(result.failed, 1)
    return result


def arm_chunked_jobs(args, executable, generations, env) -> ArmResult:
    """G processes, `--jobs ... --batch-mode M`.  Today's launcher shape."""
    result = ArmResult("chunked-jobs")
    for i, manifest in enumerate(generations):
        command = list(executable) + ["--jobs", str(manifest.resolve()),
                                      "--batch-mode", str(args.batch_mode)]
        if args.result:
            command += ["--result", args.result]
        text, code, wall = run(command, stdin_text=None,
                               log=Path(args.workdir) / f"chunked_jobs.{i + 1:02d}.log",
                               cwd=args.cwd, env=env)
        result.absorb(text)
        result.wall += wall
        result.processes += 1
        if code != 0:
            result.failed = max(result.failed, 1)
    return result


ARMS = {
    "persistent": arm_persistent,
    "per-generation": arm_per_generation,
    "chunked-jobs": arm_chunked_jobs,
}


def report(results: list[ArmResult], reference: str) -> int:
    print()
    print(f"{'arm':<16}{'proc':>5}{'cases':>7}{'failed':>7}{'wall_s':>10}"
          f"{'c/h':>9}{'outside_drive_s':>17}")
    for r in results:
        rate = 3600.0 * r.cases / r.wall if r.wall > 0 else 0.0
        print(f"{r.name:<16}{r.processes:>5}{r.cases:>7}{r.failed:>7}{r.wall:>10.2f}"
              f"{rate:>9.1f}{r.outside_drive:>17.2f}")

    ref = next((r for r in results if r.name == reference), None)
    if ref is None:
        return 1
    verdict = 0
    print()
    for r in results:
        if r is ref:
            continue
        # Per-deck where both arms name their decks; multiset otherwise.  Say
        # which comparison was made -- they are not the same claim.
        if r.digests and ref.digests:
            shared = set(r.digests) & set(ref.digests)
            differing = sorted(d for d in shared if r.digests[d] != ref.digests[d])
            missing = sorted(set(ref.digests) - set(r.digests))
            kind = f"per-deck over {len(shared)} deck(s)"
            bad = differing + [f"{d} (absent)" for d in missing]
        else:
            differing = []
            bad = [] if r.digest_bag == ref.digest_bag else ["digest multiset differs"]
            kind = f"multiset over {sum(ref.digest_bag.values())} case(s)"
        status = "IDENTICAL" if not bad else "MISMATCH"
        print(f"digest {r.name} vs {ref.name}: {status}  [{kind}]")
        for item in bad[:12]:
            print(f"    {item}")
        if bad:
            verdict = 1
    for r in results:
        for receipt in r.process_receipts:
            if receipt.get("isolation_mismatches", 0):
                print(f"ISOLATION MISMATCH in {r.name}: "
                      f"{receipt['isolation_mismatches']} wave(s)")
                verdict = 1
            if receipt.get("xslib_loads", 1) != 1:
                print(f"NOTE {r.name}: xslib_loads={receipt['xslib_loads']} "
                      "(expected 1 per distinct library content)")
            for field in ("slot_duplicates", "slot_stale_tenants", "slot_double_releases",
                          "pin_live_ranges_between_waves"):
                if receipt.get(field, 0):
                    print(f"TENANCY BUG in {r.name}: {field}={receipt[field]}")
                    verdict = 1
        if r.refused:
            print(f"REFUSED requests in {r.name}: {r.refused}")
            verdict = 1
    return verdict


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-mode", type=int, required=True, metavar="M",
                   help="arena width; latched for the whole evaluator process")
    p.add_argument("--generation", action="append", required=True, metavar="JOBS",
                   help="a --jobs manifest; repeat once per generation, in order")
    p.add_argument("--arm", action="append", choices=sorted(ARMS), metavar="ARM",
                   help="which shapes to run (default: all three)")
    p.add_argument("--reference", default="per-generation", choices=sorted(ARMS),
                   help="the arm every other arm's digests are compared against")
    p.add_argument("--result", choices=("full", "pin-off", "light"),
                   help="passed through as --result")
    p.add_argument("--isolation-check", action="store_true",
                   help="add --evaluator-isolation-check to the persistent arm")
    p.add_argument("--workdir", default="evaluator_run")
    p.add_argument("--cwd", help="run the child here (decks reference their XS relatively)")
    p.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE",
                   help="environment for the child; repeatable")
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="the RASBERY executable, preceded by --")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    executable = [a for a in args.command if a != "--"]
    if not executable:
        print("no executable: put it after `--`", file=sys.stderr)
        return 2
    generations = [Path(g) for g in args.generation]
    for g in generations:
        if not g.is_file():
            print(f"no such generation manifest: {g}", file=sys.stderr)
            return 2
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for pair in args.set_values:
        key, _, value = pair.partition("=")
        env[key] = value
    # GPU0 only, and the child must inherit it (plan Sec 6.4).
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    arms = args.arm or sorted(ARMS)
    if args.reference not in arms:
        arms = [args.reference] + [a for a in arms if a != args.reference]
    results = [ARMS[name](args, executable, generations, env) for name in arms]
    return report(results, args.reference)


if __name__ == "__main__":
    raise SystemExit(main())
