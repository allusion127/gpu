#!/usr/bin/env python3
"""Static fail-safe contract for bounded GA feedback screening."""

from pathlib import Path


source = (Path(__file__).resolve().parents[1] / "src" / "Driver.h").read_text(encoding="utf-8")
required = (
    "RASBERY_GA_FEEDBACK_PASSES",
    "return value == nullptr ? 0",
    "const int xe_budget = ga_feedback_passes > 0",
    "? ga_feedback_passes",
    "th_count >= ga_feedback_passes",
    "candidates must be rerun through the unlimited exact path",
)
missing = [token for token in required if token not in source]
if missing:
    raise SystemExit("GA feedback screen contract missing: " + ", ".join(missing))

main = (Path(__file__).resolve().parents[1] / "src" / "main.cpp").read_text(encoding="utf-8")
receipt = (Path(__file__).resolve().parents[1] / "include" / "chiffon" / "BatchLightResult.h").read_text(encoding="utf-8")
main_required = (
    "requires RASBERY_BATCH_LIGHT_RESULT=1",
    "[RASBERY][GA][SCREEN]",
    "requires_exact_rerun",
)
receipt_required = (
    'receipt["physics_mode"]',
    'receipt["feedback_passes"]',
    'receipt["requires_exact_rerun"]',
)
missing = [token for token in main_required if token not in main]
missing += [token for token in receipt_required if token not in receipt]
if missing:
    raise SystemExit("GA feedback fail-safe missing: " + ", ".join(missing))
print("GA feedback screen: PASS")
