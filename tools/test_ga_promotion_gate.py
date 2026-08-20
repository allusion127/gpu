#!/usr/bin/env python3
"""Promotion gate for the GA two-stage multi-precision 40x claim.

The 40x number is a *candidate-generation throughput* accounting result, not a
solver speedup: 256 bounded-feedback screens plus 16 exact reruns finish in
90.5302 s, which is 10,180.03 effective cases/h and 43.5602x over the original
M1 233.7 cases/h.  That accounting is only legitimate while four conditions
hold, so this script checks them rather than trusting the number.

  G1  Screening is marked as screening.  Every screen receipt carries
      physics_mode=ga_screen_feedback_limited and requires_exact_rerun=true,
      and every selected survivor has an exact receipt of its own.  A survivor
      set with one missing exact receipt is discarded, not patched.
  G2  Screen scalars never leave the screening lane.  The screen writes no
      result HDF5, and the published survivor numbers come from the exact
      stage.
  G3  Quality is gated on the exact stage: TH-clamp, critical-search and flux
      limit-cycle warnings are counted per stage and a warning in the exact
      stage fails the run.
  G4  GPU0 is pinned by UUID -- not by ordinal -- and GPU1 is verified idle
      before and after.

Two modes:

    test_ga_promotion_gate.py
        Static contract check over src/, include/chiffon/ and the runner.
        This is the CI-shaped mode; it needs no GPU and no run.

    test_ga_promotion_gate.py MANIFEST.json [...]
        Audit a produced two_stage_manifest.json.  Runs the static check first,
        then the manifest.  A manifest written by a runner that predates a gate
        is reported UNPROVEN, which is not a pass -- the historical
        ga_two_stage_256x16_manifest_20260820.json is schema v1 and carries no
        warning aggregation or GPU0 UUID assertion.

Exit status is 0 only when every gate is PASS.

References: GPU_RASBERRY_40X_OPTIMIZATION_LEDGER_20260820.md §1, §5, §7;
ANALYSIS_STRUCTURE_20260820.md §2.4, §2.5, §5.4(B), §5.5.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

# Environment variables the v8 runner set that the authoritative source never
# reads.  Keeping them out is a gate, not a cleanup: an environment block that
# says RASBERY_FULL_GPU=1 is evidence for a "full GPU mode" that does not exist.
DEAD_ENVIRONMENT = (
    "RASBERY_FULL_GPU",
    "RASBERY_GPU_FORCE_ORDERED_PRECONDITIONER",
    "RASBERY_GPU_DISABLE_CUDA_GRAPHS",
)

RUNNER = ROOT / "tools" / "ga_two_stage_40x_pipeline.py"


class Report:
    def __init__(self):
        self.gates = {}

    def record(self, gate: str, status: str, detail: str = ""):
        previous = self.gates.get(gate)
        # Worst status wins, so a later PASS cannot paper over an earlier FAIL.
        order = {"PASS": 0, "UNPROVEN": 1, "FAIL": 2}
        if previous is None or order[status] > order[previous[0]]:
            self.gates[gate] = (status, detail)
        elif status == previous[0] and detail:
            self.gates[gate] = (status, (previous[1] + "; " + detail).strip("; "))

    def require(self, gate: str, tokens, text: str, source: str):
        missing = [token for token in tokens if token not in text]
        if missing:
            self.record(gate, "FAIL",
                        f"{source} is missing {', '.join(repr(m) for m in missing)}")
        else:
            self.record(gate, "PASS")

    def failed(self) -> bool:
        return any(status != "PASS" for status, _ in self.gates.values())

    def render(self, heading: str):
        print(heading)
        for gate in sorted(self.gates):
            status, detail = self.gates[gate]
            print(f"  {status:8s} {gate}" + (f" -- {detail}" if detail else ""))


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def static_check(report: Report):
    main = read("src/main.cpp")
    driver = read("src/Driver.h")
    receipt = read("include/chiffon/BatchLightResult.h")

    # G1 -- the approximation announces itself and cannot be run silently.
    report.require("G1_screen_marked_and_survivors_exact", (
        "RASBERY_GA_FEEDBACK_PASSES",
        "requires RASBERY_BATCH_LIGHT_RESULT=1",
        "[RASBERY][GA][SCREEN]",
        "requires_exact_rerun",
        "return 2;",
    ), main, "src/main.cpp")
    report.require("G1_screen_marked_and_survivors_exact", (
        'receipt["physics_mode"]',
        'receipt["feedback_passes"]',
        'receipt["requires_exact_rerun"]',
        '"ga_screen_feedback_limited"',
        '"exact"',
    ), receipt, "include/chiffon/BatchLightResult.h")
    report.require("G1_screen_marked_and_survivors_exact", (
        "const int xe_budget = ga_feedback_passes > 0",
        "th_count >= ga_feedback_passes",
        "candidates must be rerun through the unlimited exact path",
    ), driver, "src/Driver.h")

    # G2 -- the fail-closed branch is the output separation.  Full HDF5 output
    # plus a truncated feedback budget is the one combination that could publish
    # a screen scalar as physics, and it cannot be spelled.
    if "if (!rasbery::BatchLightResult::Enabled())" not in main:
        report.record("G2_screen_output_separated", "FAIL",
                      "src/main.cpp no longer refuses screening without light-result mode")
    else:
        report.record("G2_screen_output_separated", "PASS")

    # Dead environment surface, checked against the whole authoritative source.
    sources = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((ROOT / "src").rglob("*"))
        if path.suffix in {".cpp", ".h", ".cu"}
    ) + "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((ROOT / "include" / "chiffon").rglob("*.h"))
    )
    resurrected = [name for name in DEAD_ENVIRONMENT if name in sources]
    if resurrected:
        report.record("G4_gpu_contract", "FAIL",
                      "source now reads " + ", ".join(resurrected)
                      + " -- update the runner and this gate together")

    if not RUNNER.exists():
        for gate in ("G1_screen_marked_and_survivors_exact", "G2_screen_output_separated",
                     "G3_exact_quality_gated", "G4_gpu_contract"):
            report.record(gate, "FAIL", f"{RUNNER.name} is absent")
        return

    runner = RUNNER.read_text(encoding="utf-8")

    report.require("G1_screen_marked_and_survivors_exact", (
        "wrong physics mode in",
        "wrong exact-rerun flag in",
        "lack an exact receipt",
    ), runner, RUNNER.name)

    report.require("G2_screen_output_separated", (
        "if path.exists() and path.stat().st_size",
        "screen_results_must_not_be_published_as_exact",
        "survivor_results_are_exact",
    ), runner, RUNNER.name)

    # G3 -- every warning family the solver can emit has to be counted, and the
    # exact stage has to be able to fail on them.
    report.require("G3_exact_quality_gated", (
        "def collect_warnings",
        "def quality_gate",
        r"\[RASBERY\]\[WARN\]\[th\]",
        r"\[RASBERY\]\[WARN\]\[search\]",
        r"\[RASBERY\]\[WARN\]\[flux\]",
        r"\[RASBERY\]\[dhat-guard\]",
        "th_clamped_events",
        "search_not_converged",
        "flux_limit_cycle",
        "unconverged_candidates",
        '"warnings_total"',
    ), runner, RUNNER.name)

    # G4 -- ordinal pinning is not identity pinning.
    report.require("G4_gpu_contract", (
        "DEFAULT_GPU0_UUID",
        "GPU0 identity gate failed",
        "GPU1 isolation gate failed",
        '"before the run"',
        '"after the run"',
    ), runner, RUNNER.name)

    still_set = [name for name in DEAD_ENVIRONMENT if f'"{name}":' in runner]
    if still_set:
        report.record("G4_gpu_contract", "FAIL",
                      "runner still sets no-op environment: " + ", ".join(still_set))


def audit_manifest(report: Report, path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = manifest.get("schema", "?")
    label = f"{path.name} ({schema})"

    # G1 -- survivors must carry their own exact numbers.
    selected = manifest.get("selected_ids") or []
    pairs = manifest.get("selected_screen_and_exact") or {}
    if not selected:
        report.record("G1_screen_marked_and_survivors_exact", "FAIL",
                      f"{label} lists no selected candidates")
    else:
        fields = ("keff", "ppm", "fqp", "frp")
        missing = [key for key in selected
                   if key not in pairs
                   or any(pairs[key].get("exact", {}).get(field) is None for field in fields)]
        if missing:
            report.record("G1_screen_marked_and_survivors_exact", "FAIL",
                          f"{label}: {len(missing)} survivor(s) without exact scalars: "
                          + ", ".join(missing[:5]))
        else:
            report.record("G1_screen_marked_and_survivors_exact", "PASS")

    # If the receipt JSONL is reachable, the flags themselves are checkable.
    checked_rows = 0
    for stage, expect_rerun in (("screen", True), ("exact_selected", False)):
        for wave in (manifest.get(stage) or {}).get("waves", []):
            receipt = Path(wave.get("receipt", ""))
            if not receipt.is_file():
                continue
            for line in receipt.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                checked_rows += 1
                if bool(row.get("requires_exact_rerun")) != expect_rerun:
                    report.record("G1_screen_marked_and_survivors_exact", "FAIL",
                                  f"{label}: {stage} receipt row "
                                  f"{row.get('candidate_id')} has "
                                  f"requires_exact_rerun={row.get('requires_exact_rerun')}")
    if checked_rows:
        report.record("G1_screen_marked_and_survivors_exact", "PASS")

    # G2 -- if the exact scalars equal the screen scalars the "rerun" was a copy.
    if pairs:
        identical = [key for key, value in pairs.items()
                     if value.get("screen") == value.get("exact")]
        if identical:
            report.record("G2_screen_output_separated", "FAIL",
                          f"{label}: exact scalars identical to screen for "
                          + ", ".join(identical[:5])
                          + " -- the exact rerun did not happen")
        elif manifest.get("screen_results_must_not_be_published_as_exact") is not True:
            report.record("G2_screen_output_separated", "FAIL",
                          f"{label} does not declare "
                          "screen_results_must_not_be_published_as_exact")
        else:
            report.record("G2_screen_output_separated", "PASS")

    # G3 -- warning aggregation per stage.
    graded = False
    for stage in ("screen", "exact_selected"):
        totals = (manifest.get(stage) or {}).get("warnings_total")
        if totals is None:
            report.record("G3_exact_quality_gated", "UNPROVEN",
                          f"{label} records no warning aggregation for {stage}; "
                          "TH clamp and search warnings were never collected, so a "
                          "clean run and an unreported one look the same")
            continue
        graded = True
        if stage == "exact_selected":
            offenders = {key: value for key, value in totals.items()
                         if not key.endswith("_max") and key != "dhat_guard" and value}
            if offenders:
                report.record("G3_exact_quality_gated", "FAIL",
                              f"{label}: exact stage warnings {offenders}")
    quality = ((manifest.get("gates") or {}).get("G3_exact_quality") or {})
    if quality.get("status") == "FAIL":
        report.record("G3_exact_quality_gated", "FAIL",
                      f"{label}: " + "; ".join(quality.get("failures", [])))
    elif graded:
        report.record("G3_exact_quality_gated", "PASS")
        for waiver in quality.get("waivers", []):
            print(f"  WAIVED   G3_exact_quality_gated -- {waiver}")

    # G4 -- GPU identity and GPU1 isolation, before and after.
    before = manifest.get("gpu_inventory_before") or []
    after = manifest.get("gpu_inventory_after") or []
    expected = ((manifest.get("gates") or {}).get("G4_gpu_pinned") or {}
                ).get("expected_gpu0_uuid")
    if len(before) < 2 or len(after) < 2:
        report.record("G4_gpu_contract", "FAIL", f"{label} has no two-GPU inventory")
    elif expected is None:
        report.record("G4_gpu_contract", "UNPROVEN",
                      f"{label} records GPU0 as {before[0].get('uuid')} but never "
                      "asserted it; CUDA_VISIBLE_DEVICES=0 selects an ordinal, so "
                      "nothing in this run proves which card produced the numbers")
    elif before[0].get("uuid") != expected or after[0].get("uuid") != expected:
        report.record("G4_gpu_contract", "FAIL",
                      f"{label}: GPU0 UUID moved during the run")
    else:
        report.record("G4_gpu_contract", "PASS")
    for phase, inventory in (("before", before), ("after", after)):
        if len(inventory) >= 2 and (inventory[1].get("util_percent")
                                    or inventory[1].get("memory_mib", 0) > 1):
            report.record("G4_gpu_contract", "FAIL",
                          f"{label}: GPU1 not idle {phase} ({inventory[1]})")

    if manifest.get("status") != "PASS":
        report.record("G3_exact_quality_gated", "FAIL",
                      f"{label} status={manifest.get('status')}")


def main(argv):
    report = Report()
    static_check(report)
    report.render("GA promotion gate -- static contract:")
    overall_failed = report.failed()

    for argument in argv:
        path = Path(argument)
        if not path.is_file():
            print(f"\nGA promotion gate -- manifest audit: FAIL -- {path} not found")
            overall_failed = True
            continue
        audit = Report()
        audit_manifest(audit, path)
        audit.render(f"\nGA promotion gate -- manifest audit ({path.name}):")
        overall_failed = overall_failed or audit.failed()

    print("\nGA promotion gate: " + ("FAIL" if overall_failed else "PASS"))
    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
