#!/usr/bin/env python3
"""Run a fail-closed RASBERY screen -> exact campaign.

The screen stage is explicitly approximate and may only rank candidates.  The
selected survivors are re-run from their original inputs through the exact
path.  The receipt reports *effective candidate throughput* including both
stages and compares it with a measured MASTER W16 baseline.

This tool deliberately does not turn a light/screen result into a physics
result.  A campaign is valid only after every selected survivor has a
non-empty exact output and every optional validator succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


EXIT_INVALID = 2
EXIT_TARGET_NOT_MET = 3
MAX_VALIDATED_BATCH_WIDTH = 64
SCREEN_PHYSICS_MODE = "ga_screen_feedback_limited"


class CampaignError(RuntimeError):
    """A fail-closed campaign contract violation."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    input_path: Path
    screen_output: Path
    exact_output: Path


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    screen_record: dict[str, Any]


@dataclass
class CommandResult:
    stage: str
    candidate_ids: list[str]
    command: list[str]
    returncode: int
    wall_seconds: float
    log_path: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_manifest_payload(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise CampaignError(f"empty manifest: {path}")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignError(
                    f"manifest JSONL parse error at line {lineno}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise CampaignError(f"manifest line {lineno} is not an object")
            rows.append(item)
        return rows

    if isinstance(payload, dict):
        payload = payload.get("candidates")
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise CampaignError("manifest must be a JSON list or {'candidates': [...]} object")
    return list(payload)


def load_candidates(manifest: Path, workdir: Path) -> list[Candidate]:
    rows = _load_manifest_payload(manifest)
    manifest_base = manifest.parent.resolve()
    screen_dir = workdir / "screen"
    exact_dir = workdir / "exact"
    screen_dir.mkdir(parents=True, exist_ok=True)
    exact_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    seen_inputs: set[Path] = set()
    for index, row in enumerate(rows):
        candidate_id = str(row.get("id", row.get("candidate_id", f"candidate_{index:06d}")))
        if not candidate_id or candidate_id in seen_ids:
            raise CampaignError(f"duplicate or empty candidate id: {candidate_id!r}")
        raw_input = row.get("input", row.get("input_path"))
        if not isinstance(raw_input, str) or not raw_input:
            raise CampaignError(f"candidate {candidate_id}: missing input path")
        input_path = _resolve_path(raw_input, manifest_base)
        if not input_path.is_file():
            raise CampaignError(f"candidate {candidate_id}: input not found: {input_path}")
        if input_path in seen_inputs:
            raise CampaignError(
                f"candidate {candidate_id}: duplicate input path is unsafe for concurrent runs: "
                f"{input_path}"
            )

        raw_screen = row.get("screen_output")
        raw_exact = row.get("exact_output")
        screen_output = (
            _resolve_path(str(raw_screen), manifest_base)
            if raw_screen is not None
            else (screen_dir / f"{candidate_id}.jsonl").resolve()
        )
        exact_output = (
            _resolve_path(str(raw_exact), manifest_base)
            if raw_exact is not None
            else (exact_dir / f"{candidate_id}.h5").resolve()
        )
        if screen_output == exact_output:
            raise CampaignError(
                f"candidate {candidate_id}: screen and exact outputs must be different files"
            )

        seen_ids.add(candidate_id)
        seen_inputs.add(input_path)
        candidates.append(
            Candidate(candidate_id, input_path, screen_output, exact_output)
        )

    if not candidates:
        raise CampaignError("manifest contains no candidates")
    return candidates


def chunked(items: Sequence[Candidate], width: int) -> Iterable[list[Candidate]]:
    if width < 1 or width > MAX_VALIDATED_BATCH_WIDTH:
        raise CampaignError(
            f"batch width must be in [1, {MAX_VALIDATED_BATCH_WIDTH}], got {width}"
        )
    for start in range(0, len(items), width):
        yield list(items[start : start + width])


def get_dotted(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for component in dotted_key.split("."):
        if not isinstance(value, dict) or component not in value:
            raise CampaignError(f"missing score key {dotted_key!r}")
        value = value[component]
    return value


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise CampaignError(f"missing or empty screen receipt: {path}")
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignError(
                    f"screen receipt {path} line {lineno} is not JSON: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise CampaignError(f"screen receipt {path} line {lineno} is not an object")
            records.append(item)
        return records

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return list(payload)
    raise CampaignError(f"screen receipt {path} is not a JSON object/list or JSONL")


def score_candidate(candidate: Candidate, score_key: str) -> ScoredCandidate:
    records = read_json_records(candidate.screen_output)
    matching = [record for record in records if score_key.split(".")[0] in record]
    record = matching[-1] if matching else records[-1]

    physics_mode = record.get("physics_mode")
    if physics_mode != SCREEN_PHYSICS_MODE:
        raise CampaignError(
            f"candidate {candidate.candidate_id}: unexpected screen physics_mode "
            f"{physics_mode!r}; expected {SCREEN_PHYSICS_MODE!r}"
        )
    if record.get("requires_exact_rerun") is not True:
        raise CampaignError(
            f"candidate {candidate.candidate_id}: screen receipt must set "
            "requires_exact_rerun=true"
        )
    raw_score = get_dotted(record, score_key)
    if isinstance(raw_score, bool):
        raise CampaignError(f"candidate {candidate.candidate_id}: boolean score is invalid")
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise CampaignError(
            f"candidate {candidate.candidate_id}: score is not numeric: {raw_score!r}"
        ) from exc
    if not math.isfinite(score):
        raise CampaignError(f"candidate {candidate.candidate_id}: score is not finite")
    return ScoredCandidate(candidate, score, record)


def select_survivors(
    scored: Sequence[ScoredCandidate], survivor_count: int, maximize: bool
) -> list[ScoredCandidate]:
    if survivor_count < 1 or survivor_count > len(scored):
        raise CampaignError(
            f"survivor count must be in [1, {len(scored)}], got {survivor_count}"
        )
    return sorted(
        scored,
        key=lambda item: (item.score, item.candidate.candidate_id),
        reverse=maximize,
    )[:survivor_count]


def effective_throughput(total_candidates: int, total_wall_seconds: float) -> float:
    if total_candidates < 1 or not math.isfinite(total_wall_seconds) or total_wall_seconds <= 0:
        raise CampaignError("throughput requires positive candidate count and wall time")
    return total_candidates * 3600.0 / total_wall_seconds


def speedup_vs_master(effective_cases_per_hour: float, master_w16_cases_per_hour: float) -> float:
    if (
        not math.isfinite(effective_cases_per_hour)
        or effective_cases_per_hour < 0
        or not math.isfinite(master_w16_cases_per_hour)
        or master_w16_cases_per_hour <= 0
    ):
        raise CampaignError("invalid throughput or MASTER W16 baseline")
    return effective_cases_per_hour / master_w16_cases_per_hour


def _command_for_batch(
    executable: Path,
    common_args: Sequence[str],
    batch: Sequence[Candidate],
    output_kind: str,
) -> list[str]:
    outputs = [
        candidate.screen_output if output_kind == "screen" else candidate.exact_output
        for candidate in batch
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
    return [
        str(executable),
        *common_args,
        "--rasi",
        *(str(candidate.input_path) for candidate in batch),
        "--raso",
        *(str(output) for output in outputs),
        "--batch-mode",
        str(len(batch)),
    ]


def run_batch(
    *,
    stage: str,
    executable: Path,
    common_args: Sequence[str],
    batch: Sequence[Candidate],
    env: dict[str, str],
    log_dir: Path,
    ordinal: int,
    timeout_seconds: float,
) -> CommandResult:
    command = _command_for_batch(executable, common_args, batch, stage)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}_{ordinal:04d}.log"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n[TIMEOUT]\n"
        returncode = 124
    wall = time.monotonic() - started
    log_path.write_text(output, encoding="utf-8", errors="replace")
    result = CommandResult(
        stage=stage,
        candidate_ids=[candidate.candidate_id for candidate in batch],
        command=command,
        returncode=returncode,
        wall_seconds=wall,
        log_path=str(log_path),
    )
    if returncode != 0:
        raise CampaignError(
            f"{stage} batch {ordinal} failed with rc={returncode}; log={log_path}"
        )
    return result


def validate_exact_output(candidate: Candidate, validator: str | None) -> None:
    output = candidate.exact_output
    if not output.is_file() or output.stat().st_size == 0:
        raise CampaignError(
            f"candidate {candidate.candidate_id}: exact output missing or empty: {output}"
        )
    if validator is None:
        return
    command = [part.format(output=str(output), input=str(candidate.input_path)) for part in shlex.split(validator)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise CampaignError(
            f"candidate {candidate.candidate_id}: exact validator failed rc="
            f"{completed.returncode}: {completed.stdout}{completed.stderr}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--common-arg", action="append", default=[])
    parser.add_argument("--batch-width", type=int, default=64)
    parser.add_argument("--screen-feedback-passes", type=int, default=2)
    parser.add_argument("--survivors", type=int, required=True)
    parser.add_argument("--score-key", required=True)
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--maximize", action="store_true")
    direction.add_argument("--minimize", action="store_true")
    parser.add_argument("--master-w16-cases-per-hour", type=float, default=217.0)
    parser.add_argument("--target-speedup", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=14400.0)
    parser.add_argument(
        "--exact-validator",
        help="optional command template; {output} and {input} are substituted",
    )
    parser.add_argument(
        "--exact-xe-anderson",
        action="store_true",
        help="enable experimental exact-path Xe Anderson acceleration for survivor reruns",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    receipt_path = workdir / "campaign_receipt.json"
    ranking_path = workdir / "screen_ranking.json"

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "physics_mode": "screen_then_exact",
        "valid": False,
        "requires_exact_rerun": True,
        "target_met": False,
    }
    try:
        if args.batch_width < 1 or args.batch_width > MAX_VALIDATED_BATCH_WIDTH:
            raise CampaignError(
                f"--batch-width must be in [1, {MAX_VALIDATED_BATCH_WIDTH}]"
            )
        if args.screen_feedback_passes < 1:
            raise CampaignError("--screen-feedback-passes must be positive")
        if args.target_speedup <= 0 or args.master_w16_cases_per_hour <= 0:
            raise CampaignError("speedup target and MASTER W16 baseline must be positive")

        manifest = args.manifest.expanduser().resolve()
        executable = args.executable.expanduser().resolve()
        if not executable.is_file():
            raise CampaignError(f"executable not found: {executable}")
        candidates = load_candidates(manifest, workdir)

        executable_sha256 = sha256_file(executable)
        input_receipts = [
            {
                "id": candidate.candidate_id,
                "input": str(candidate.input_path),
                "input_sha256": sha256_file(candidate.input_path),
                "screen_output": str(candidate.screen_output),
                "exact_output": str(candidate.exact_output),
            }
            for candidate in candidates
        ]

        base_env = dict(os.environ)
        screen_env = dict(base_env)
        screen_env["RASBERY_BATCH_LIGHT_RESULT"] = "1"
        screen_env["RASBERY_GA_FEEDBACK_PASSES"] = str(args.screen_feedback_passes)
        screen_env["RASBERY_BATCH_WIDTH"] = str(args.batch_width)

        screen_results: list[CommandResult] = []
        for ordinal, batch in enumerate(chunked(candidates, args.batch_width), start=1):
            screen_results.append(
                run_batch(
                    stage="screen",
                    executable=executable,
                    common_args=args.common_arg,
                    batch=batch,
                    env=screen_env,
                    log_dir=workdir / "logs",
                    ordinal=ordinal,
                    timeout_seconds=args.timeout,
                )
            )
        screen_wall = sum(result.wall_seconds for result in screen_results)

        scored = [score_candidate(candidate, args.score_key) for candidate in candidates]
        survivors = select_survivors(scored, args.survivors, maximize=args.maximize)
        ranking = [
            {
                "rank": rank,
                "id": item.candidate.candidate_id,
                "score": item.score,
                "selected_for_exact": item in survivors,
                "requires_exact_rerun": True,
            }
            for rank, item in enumerate(
                sorted(
                    scored,
                    key=lambda value: (value.score, value.candidate.candidate_id),
                    reverse=args.maximize,
                ),
                start=1,
            )
        ]
        ranking_path.write_text(json.dumps(ranking, indent=2) + "\n", encoding="utf-8")

        exact_env = dict(base_env)
        exact_env.pop("RASBERY_BATCH_LIGHT_RESULT", None)
        exact_env.pop("RASBERY_GA_FEEDBACK_PASSES", None)
        if args.exact_xe_anderson:
            exact_env["RASBERY_XE_ANDERSON"] = "1"
        else:
            exact_env.pop("RASBERY_XE_ANDERSON", None)

        exact_candidates = [item.candidate for item in survivors]
        exact_results: list[CommandResult] = []
        for ordinal, batch in enumerate(chunked(exact_candidates, args.batch_width), start=1):
            exact_results.append(
                run_batch(
                    stage="exact",
                    executable=executable,
                    common_args=args.common_arg,
                    batch=batch,
                    env=exact_env,
                    log_dir=workdir / "logs",
                    ordinal=ordinal,
                    timeout_seconds=args.timeout,
                )
            )
        for candidate in exact_candidates:
            validate_exact_output(candidate, args.exact_validator)
        exact_wall = sum(result.wall_seconds for result in exact_results)

        total_wall = screen_wall + exact_wall
        effective_cph = effective_throughput(len(candidates), total_wall)
        exact_cph = effective_throughput(len(exact_candidates), exact_wall)
        speedup = speedup_vs_master(effective_cph, args.master_w16_cases_per_hour)
        target_met = speedup >= args.target_speedup

        receipt.update(
            {
                "valid": True,
                "requires_exact_rerun": False,
                "target_met": target_met,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "executable": str(executable),
                "executable_sha256": executable_sha256,
                "candidate_count": len(candidates),
                "survivor_count": len(exact_candidates),
                "score_key": args.score_key,
                "score_direction": "maximize" if args.maximize else "minimize",
                "batch_width": args.batch_width,
                "screen": {
                    "approximate": True,
                    "physics_mode": SCREEN_PHYSICS_MODE,
                    "feedback_passes": args.screen_feedback_passes,
                    "wall_seconds": screen_wall,
                    "commands": [asdict(result) for result in screen_results],
                },
                "exact": {
                    "approximate": False,
                    "xe_anderson_requested": bool(args.exact_xe_anderson),
                    "wall_seconds": exact_wall,
                    "commands": [asdict(result) for result in exact_results],
                    "outputs": [
                        {
                            "id": candidate.candidate_id,
                            "path": str(candidate.exact_output),
                            "sha256": sha256_file(candidate.exact_output),
                        }
                        for candidate in exact_candidates
                    ],
                },
                "inputs": input_receipts,
                "performance": {
                    "screen_wall_seconds": screen_wall,
                    "exact_wall_seconds": exact_wall,
                    "total_wall_seconds": total_wall,
                    "effective_candidate_cases_per_hour": effective_cph,
                    "exact_survivor_cases_per_hour": exact_cph,
                    "master_w16_cases_per_hour": args.master_w16_cases_per_hour,
                    "speedup_vs_master_w16": speedup,
                    "target_speedup": args.target_speedup,
                },
                "ranking_file": str(ranking_path),
            }
        )
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt["performance"], indent=2))
        return 0 if target_met else EXIT_TARGET_NOT_MET
    except (CampaignError, OSError, subprocess.SubprocessError) as exc:
        receipt["error"] = str(exc)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"screen-exact campaign invalid: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
