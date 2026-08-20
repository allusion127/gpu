#!/usr/bin/env python3
"""Measured GA screening pipeline: bounded-feedback screen, then exact rerun.

The screen ranks candidates.  Every selected candidate is rerun with the
default exact feedback convergence before it can enter the GA survivor set.
Screen scalars are never published as physics results.

This is the version-controlled successor to the runner staged in
Codex_consolidated_20260820/04_benchmark_validation_and_presentation_scripts/.
Four promotion gates from GPU_RASBERRY_40X_OPTIMIZATION_LEDGER_20260820.md §7
are enforced here and audited afterwards by tools/test_ga_promotion_gate.py:

  G1  every screen receipt carries requires_exact_rerun=true, and every
      selected survivor has an exact receipt with requires_exact_rerun=false.
  G2  the screen writes no result HDF5, and the published survivor scalars come
      from the exact stage only.
  G3  TH-clamp, critical-search and flux limit-cycle warnings are counted
      separately for the screen and the exact stage, and a warning in the exact
      stage fails the run.
  G4  GPU0 is pinned by UUID and GPU1 is verified idle before and after.

Differences from the staged runner, all deliberate:

  * RASBERY_FULL_GPU, RASBERY_GPU_FORCE_ORDERED_PRECONDITIONER and
    RASBERY_GPU_DISABLE_CUDA_GRAPHS are gone.  grep over the authoritative
    source returns zero hits for all three -- they are leftovers from the
    rasbery_gpu_codex lineage and were no-ops in every run that set them.
    Leaving them in an environment block invites the claim that a run had
    "full GPU mode" or a forced ordered preconditioner.  It did not.
    (ANALYSIS_STRUCTURE_20260820.md §2.4, §5.5 item 1.)
  * GPU0 identity is asserted by UUID, not by trusting CUDA_VISIBLE_DEVICES=0.
  * Warning aggregation and the exact-stage quality gate did not exist.

Usage:
    ga_two_stage_40x_pipeline.py ROOT BINARY DECK [COUNT] [SELECT] [SCREEN_W] [EXACT_W]

Environment:
    RASBERY_GA_GPU0_UUID       expected GPU0 UUID (default: the GPU238 card)
    RASBERY_GA_ALLOW_TH_CLAMP  set to 1 to downgrade an exact-stage TH clamp
                               from FAIL to a recorded waiver
"""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Optional

# GPU238 GPU0.  The ledger pins every measurement to this card; GPU1 must stay
# idle so a co-tenant cannot inflate or deflate a wave.
DEFAULT_GPU0_UUID = "GPU-e73893d2-bd77-cccc-bf09-c0b3016cbe13"

WARNING_PATTERNS = {
    # "[RASBERY][WARN][th] N of M nodes ran off the water-property table"
    "th_clamped_events": re.compile(r"\[RASBERY\]\[WARN\]\[th\] (\d+) of (\d+) nodes"),
    "search_not_converged": re.compile(
        r"\[RASBERY\]\[WARN\]\[search\] (\w+) search NOT converged"),
    "search_best_fallback": re.compile(
        r"\[RASBERY\]\[WARN\]\[search\] fell back to best trial point"),
    "flux_limit_cycle": re.compile(r"\[RASBERY\]\[WARN\]\[flux\] limit cycle"),
    "dhat_guard": re.compile(r"\[RASBERY\]\[dhat-guard\]"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def marker(text: str, tag: str):
    matches = re.findall(re.escape(tag) + r"\s*(\{.*?\})", text)
    return json.loads(matches[-1]) if matches else None


def collect_warnings(text: str) -> dict:
    """G3: count every quality warning the solver can emit, per wave.

    Absence of a warning is a result too, so all keys are always present -- a
    manifest that simply lacks the field is indistinguishable from a clean run
    otherwise, and that ambiguity is what let TH clamps go unreported.
    """
    counts = {
        "th_clamped_events": 0,
        "th_clamped_nodes_max": 0,
        "search_not_converged": 0,
        "search_not_converged_boron": 0,
        "search_not_converged_rod": 0,
        "search_best_fallback": 0,
        "flux_limit_cycle": 0,
        "dhat_guard": 0,
    }
    for match in WARNING_PATTERNS["th_clamped_events"].finditer(text):
        counts["th_clamped_events"] += 1
        counts["th_clamped_nodes_max"] = max(
            counts["th_clamped_nodes_max"], int(match.group(1)))
    for match in WARNING_PATTERNS["search_not_converged"].finditer(text):
        counts["search_not_converged"] += 1
        kind = match.group(1).lower()
        if kind == "boron":
            counts["search_not_converged_boron"] += 1
        elif kind == "rod":
            counts["search_not_converged_rod"] += 1
    counts["search_best_fallback"] = len(
        WARNING_PATTERNS["search_best_fallback"].findall(text))
    counts["flux_limit_cycle"] = len(WARNING_PATTERNS["flux_limit_cycle"].findall(text))
    counts["dhat_guard"] = len(WARNING_PATTERNS["dhat_guard"].findall(text))
    return counts


def sum_warnings(waves) -> dict:
    total = {}
    for wave in waves:
        for key, value in wave["warnings"].items():
            if key.endswith("_max"):
                total[key] = max(total.get(key, 0), value)
            else:
                total[key] = total.get(key, 0) + value
    return total


def gpu_inventory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], universal_newlines=True
    )
    rows = []
    for line in output.splitlines():
        index, uuid, util, memory = [item.strip() for item in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "util_percent": int(util),
                     "memory_mib": int(memory)})
    return rows


def require_gpu_gate(inventory, expected_uuid: str, phase: str):
    """G4: pin GPU0 by UUID and keep GPU1 out of the measurement."""
    if len(inventory) < 2:
        raise RuntimeError("two-GPU inventory required for GPU1 isolation gate")
    if inventory[0]["uuid"] != expected_uuid:
        raise RuntimeError(
            f"GPU0 identity gate failed {phase}: expected {expected_uuid}, "
            f"found {inventory[0]['uuid']}.  CUDA_VISIBLE_DEVICES=0 selects an "
            f"ordinal, not a card; the ledger's numbers belong to one card.")
    if inventory[0]["util_percent"] != 0:
        raise RuntimeError(f"GPU0 is not idle {phase}: {inventory[0]}")
    if inventory[1]["util_percent"] != 0 or inventory[1]["memory_mib"] > 1:
        raise RuntimeError(f"GPU1 isolation gate failed {phase}: {inventory[1]}")


def make_candidates(root: Path, base_deck: Path, count: int):
    original = json.loads(base_deck.read_text(encoding="utf-8"))
    t4 = [(row, col) for row, line in enumerate(original["core"])
          for col, value in enumerate(line) if value == "T4"]
    t6 = [(row, col) for row, line in enumerate(original["core"])
          for col, value in enumerate(line) if value == "T6"]
    candidates = []
    fingerprints = set()
    candidate_index = 0
    attempt = 0
    while candidate_index < count:
        trial = json.loads(json.dumps(original))
        if candidate_index:
            rng = random.Random(20260820 + attempt)
            swaps = 1 + (attempt % min(12, len(t4), len(t6)))
            for left, right in zip(rng.sample(t4, swaps), rng.sample(t6, swaps)):
                trial["core"][left[0]][left[1]] = "T6"
                trial["core"][right[0]][right[1]] = "T4"
        attempt += 1
        canonical = json.dumps(trial["core"], separators=(",", ":"))
        if canonical in fingerprints:
            continue
        fingerprints.add(canonical)
        path = root / "inputs" / f"candidate_{candidate_index:04d}.json"
        path.write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
        candidates.append(path)
        candidate_index += 1
    return candidates


def run_wave(root: Path, binary: Path, candidates, mode: str, wave: int,
             feedback_passes: Optional[int]):
    work = root / mode / f"wave_{wave:02d}"
    work.mkdir(parents=True)
    receipt = work / "receipts.jsonl"
    outputs = [work / f"ignored_{index:04d}.h5" for index in range(len(candidates))]
    command = ([str(binary), "--rasi"] + [str(path) for path in candidates]
               + ["--raso"] + [str(path) for path in outputs]
               + ["--batch-mode", str(len(candidates))])
    env = os.environ.copy()
    width = len(candidates)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NESTED": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "GOMP_SPINCOUNT": "0",
        "MKL_NUM_THREADS": "1",
        "KMP_BLOCKTIME": "0",
        "OMP_NUM_THREADS": str(width),
        "OMP_THREAD_LIMIT": str(width),
        "RASBERY_OMP_THREADS": str(width),
        "RASBERY_GPU": "1",
        # RASBERY_FULL_GPU / RASBERY_GPU_FORCE_ORDERED_PRECONDITIONER /
        # RASBERY_GPU_DISABLE_CUDA_GRAPHS intentionally omitted: the
        # authoritative source never reads them.  CUDA Graphs are controlled by
        # RASBERY_GPU_GRAPH (default on) and the preconditioner by
        # RASBERY_PC_MODE below.
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "RASBERY_GPU_RB_SWEEPS": "4",
        "RASBERY_PC_MODE": "decart",
        "RASBERY_BATCH_WAIT_US": "auto",
        "RASBERY_BATCH_WAIT_MAX_US": "2000",
        "RASBERY_BATCH_LIGHT_RESULT": "1",
        "RASBERY_BATCH_RECEIPT_JSONL": str(receipt),
        "RASBERY_CYCLE_ID": f"two-stage-{mode}-wave{wave}",
    })
    env.pop("RASBERY_GA_FEEDBACK_PASSES", None)
    if feedback_passes is not None:
        env["RASBERY_GA_FEEDBACK_PASSES"] = str(feedback_passes)
    log = work / "run.log"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=work, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    wall = time.perf_counter() - started
    text = log.read_text(encoding="utf-8", errors="replace")
    rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()
            if line.strip()] if receipt.exists() else []
    by_id = {row["candidate_id"]: row for row in rows}
    # G2: a screen that emits result HDF5 is a screen whose scalars can be
    # mistaken for physics.  The solver skips the write in light mode; verify it.
    unexpected = [str(path) for path in outputs if path.exists() and path.stat().st_size]
    errors = re.findall(r"HDF5-DIAG|bad object header|\[ERROR\]|CUDA error|Aborted", text,
                        re.IGNORECASE)
    if result.returncode or len(by_id) != len(candidates) or unexpected or errors:
        raise RuntimeError(f"{mode} wave {wave} failed: rc={result.returncode} "
                           f"rows={len(by_id)} outputs={len(unexpected)} errors={errors[:3]}")
    # G1: the receipt, not the runner's intent, is the record of what ran.
    expected_physics = "ga_screen_feedback_limited" if feedback_passes else "exact"
    for row in rows:
        if row.get("physics_mode") != expected_physics:
            raise RuntimeError(f"wrong physics mode in {mode}: {row.get('physics_mode')}")
        if bool(row.get("requires_exact_rerun")) != bool(feedback_passes):
            raise RuntimeError(f"wrong exact-rerun flag in {mode}")
    unconverged = sorted(key for key, row in by_id.items()
                         if row.get("converged") is False or row.get("search_status") != 1)
    return {
        "return_code": result.returncode,
        "wall_s": wall,
        "throughput_cases_per_h": len(candidates) * 3600.0 / wall,
        "receipt": str(receipt),
        "receipt_sha256": sha256(receipt),
        "occupancy": marker(text, "[RASBERY][CUDA][BATCH_OCCUPANCY]"),
        "hdf5_lock": marker(text, "[RASBERY][HDF5][LOCK]"),
        "batch_host": marker(text, "[RASBERY][BATCH_HOST]"),
        "warnings": collect_warnings(text),
        "unconverged_candidates": unconverged,
        "rows": by_id,
    }


def run_waves(root: Path, binary: Path, candidates, mode: str, width: int,
              feedback_passes: Optional[int]):
    waves = []
    rows = {}
    for wave, offset in enumerate(range(0, len(candidates), width), start=1):
        result = run_wave(root, binary, candidates[offset:offset + width], mode, wave,
                          feedback_passes)
        rows.update(result.pop("rows"))
        waves.append(result)
    wall = sum(wave["wall_s"] for wave in waves)
    return {"wall_s": wall, "throughput_cases_per_h": len(candidates) * 3600.0 / wall,
            "waves": waves,
            "warnings_total": sum_warnings(waves),
            "unconverged_candidates": sorted(
                {key for wave in waves for key in wave["unconverged_candidates"]}),
            "rows": rows}


def zscores(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance) or 1.0
    return [(value - mean) / scale for value in values]


def quality_gate(exact_stage, allow_th_clamp: bool) -> dict:
    """G3, applied to the exact stage only.

    The screen is an approximation by construction, so its warnings are recorded
    and not fatal.  The exact rerun is what gets published, and ledger §7.4
    requires warning/convergence/TH-clamp gating on exactly that stage.
    """
    warnings = exact_stage["warnings_total"]
    failures = []
    waivers = []
    if exact_stage["unconverged_candidates"]:
        failures.append(
            "exact survivors did not converge: "
            + ", ".join(exact_stage["unconverged_candidates"]))
    if warnings["search_not_converged"]:
        failures.append(
            f"critical search did not converge {warnings['search_not_converged']} time(s) "
            f"(boron {warnings['search_not_converged_boron']}, "
            f"rod {warnings['search_not_converged_rod']})")
    if warnings["flux_limit_cycle"]:
        failures.append(f"flux limit cycle {warnings['flux_limit_cycle']} event(s)")
    if warnings["th_clamped_events"]:
        message = (f"T/H clamped at the water-property table edge in "
                   f"{warnings['th_clamped_events']} statepoint(s), worst "
                   f"{warnings['th_clamped_nodes_max']} node(s); coolant "
                   f"temperature and density there no longer respond to power")
        (waivers if allow_th_clamp else failures).append(message)
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "waivers": waivers,
        "th_clamp_waiver_requested": allow_th_clamp,
    }


def main():
    root = Path(sys.argv[1])
    binary = Path(sys.argv[2])
    deck = Path(sys.argv[3])
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    select_count = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    screen_width = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    exact_width = int(sys.argv[7]) if len(sys.argv) > 7 else select_count
    if not 0 < select_count < count:
        raise SystemExit("select_count must be between zero and candidate count")
    expected_uuid = os.environ.get("RASBERY_GA_GPU0_UUID", DEFAULT_GPU0_UUID)
    allow_th_clamp = os.environ.get("RASBERY_GA_ALLOW_TH_CLAMP") == "1"

    root.mkdir()
    (root / "inputs").mkdir()
    before = gpu_inventory()
    require_gpu_gate(before, expected_uuid, "before the run")
    candidates = make_candidates(root, deck, count)
    started = time.perf_counter()
    screen = run_waves(root, binary, candidates, "screen_pass1", screen_width, 1)
    ids = sorted(screen["rows"])
    fqp_z = zscores([float(screen["rows"][key]["fqp"]) for key in ids])
    frp_z = zscores([float(screen["rows"][key]["frp"]) for key in ids])
    score = {key: fqp + frp for key, fqp, frp in zip(ids, fqp_z, frp_z)}
    selected_ids = sorted(ids, key=lambda key: score[key])[:select_count]
    candidate_by_id = {path.stem: path for path in candidates}
    selected = [candidate_by_id[key] for key in selected_ids]
    exact = run_waves(root, binary, selected, "exact_selected", exact_width, None)
    total_wall = time.perf_counter() - started
    after = gpu_inventory()
    require_gpu_gate(after, expected_uuid, "after the run")

    exact_rows = exact.pop("rows")
    screen_rows = screen.pop("rows")
    # G1: no survivor may enter the GA set on a screen scalar.
    missing = [key for key in selected_ids if key not in exact_rows]
    if missing:
        raise RuntimeError(
            "survivor set discarded -- selected candidates lack an exact receipt: "
            + ", ".join(missing))
    quality = quality_gate(exact, allow_th_clamp)

    baseline_cases_per_h = 233.7
    effective_cases_per_h = count * 3600.0 / total_wall
    manifest = {
        "schema": "rasbery-ga-two-stage-exact-survivor-v2",
        "status": "PASS" if quality["status"] == "PASS" else "FAIL",
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "deck": str(deck),
        "deck_sha256": sha256(deck),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "candidate_count": count,
        "selected_exact_count": select_count,
        "selected_fraction": select_count / count,
        "selection_metric": "z(FQP_screen)+z(FRP_screen), ascending",
        "selection_excludes_ao": True,
        "selection_excludes_ao_reason":
            "AO rank correlation between screen and exact is ~0.619 (ledger §5)",
        "selected_ids": selected_ids,
        "selected_screen_and_exact": {
            key: {"screen": {field: screen_rows[key].get(field)
                             for field in ("keff", "ppm", "ao", "fqp", "frp")},
                  "exact": {field: exact_rows[key].get(field)
                            for field in ("keff", "ppm", "ao", "fqp", "frp")}}
            for key in selected_ids
        },
        "screen": screen,
        "exact_selected": exact,
        "total_wall_s": total_wall,
        "effective_cases_per_h": effective_cases_per_h,
        "speedup_vs_original_m1_233_7_cases_per_h":
            effective_cases_per_h / baseline_cases_per_h,
        "survivor_results_are_exact": True,
        "screen_results_must_not_be_published_as_exact": True,
        "gates": {
            "G1_survivor_exact_receipts": "PASS",
            "G2_screen_output_separated": "PASS",
            "G3_exact_quality": quality,
            "G4_gpu_pinned": {
                "status": "PASS",
                "expected_gpu0_uuid": expected_uuid,
                "gpu1_idle_before_and_after": True,
            },
        },
        "gpu_inventory_before": before,
        "gpu_inventory_after": after,
    }
    path = root / "two_stage_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
