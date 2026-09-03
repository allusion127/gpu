#!/usr/bin/env python3
"""Build a RASBERY i-SMR deck with rods FIXED at MASTER's critical positions.

WHY THIS EXISTS (anchor B, docs/ISMR_MASTER_COMPARISON_RUNBOOK_20260903_KO.md
Sec 3/4).  CY02/03/04 are rod-critical on both sides: MASTER prints SEARCH=5
and K-EFF == 1.000000 at every statepoint, RASBERY drives its own rod search
to the same target, and boron is 0 ppm on both sides for the whole cycle.  So
`delta_pcm` and `delta_ppm` on those decks carry no model-vs-MASTER
information at all -- they are a search residual and a constant minus itself.
The quantity that DOES carry information is WHERE THE BANK ENDED UP
(`delta_rod_cm_<BANK>`, which tools/compare_master_rasbery.py already reports),
and this tool turns that rod-position discrepancy into a number a physicist
reads the same way as any other keff bar: pcm.

It does this by DISABLING rod search and PINNING every bank at MASTER's own
converged cm for that statepoint, so RASBERY solves a plain eigenvalue problem
at MASTER's rod configuration.  keff != 1.0 at that fixed geometry is then
"how much reactivity the rod-position gap between the two codes is worth",
independent of anything either code's search residual might otherwise hide.
tools/ismr_rod_reactivity.py turns that keff into pcm.

THE DECK SCHEMA (verified against test/7_i-SMR_Validation/i-SMR_CY02.json and
src/IO.cpp; not assumed):

  * Rod search mode is toggled per-cycle by ONE key on the schedule's leading
    `{"type": "standard", ...}` entry: `"search": "rod"` (src/IO.cpp:34
    `ParseSearchTypeString`: "boron" -> BORON, "rod" -> RODCRIT, anything else
    (including absent) -> KEFF).  `ParseSearchTypeString` is called once for
    that entry and, because the entry is type "standard", ITS result is
    promoted to `default_search` for every later entry that does not carry
    its own `"search"` key (src/IO.cpp:487).  So flipping that single key to
    `"keff"` (a PLAIN eigenvalue solve, no search at all) is sufficient to
    disable rod search for the whole cycle, PROVIDED no other entry sets its
    own `"search"` -- true of every i-SMR CY02/03/04 deck checked here, and
    asserted below rather than assumed.

  * A schedule entry `{"type": "rod insertion", "<BANK>": <insertion_cm>, ...}`
    (src/IO.cpp:502-528) sets bank insertion DEPTHS directly (not positions --
    same convention as `rods/insertions` in the .h5, `pos_cm = rod_top_cm -
    insertion`, the identical relationship tools/compare_master_rasbery.py
    already uses).  It creates its own ScheduleType::ROD entry that runs
    before whatever follows it, and non-numeric / reserved keys (search*,
    print, rate, ...) are filtered out of the insertion map
    (src/IO.cpp:507-517), so the entry can be built with the bank names as
    the ONLY payload keys plus an optional "print" passthrough.
    Confirmed usage: test/3-1_Colinear/Base_Rasbery.json:69.

  * Bank names in "rod configuration" / "rod map" (R1..R4, S1..S6) are the
    SAME strings MASTER's SUMMARY EDIT 1 header prints -- no name mapping is
    needed between the two sides.

STATEPOINT ALIGNMENT.  A `{"type": "depletion", "steps": N, ...}` entry is one
schedule ITEM but N separate statepoints (src/IO.cpp:435, one iteration of the
step loop per statepoint).  This tool therefore walks the schedule in the
SAME order src/IO.cpp expands it -- the leading "standard" entry is
statepoint 0, then each depletion entry contributes one statepoint per
`steps` -- and zips that ordered statepoint list against MASTER's EDIT 1 rows
sorted by ascending EFPD.  Verified exactly equal on all three shipped decks
(2026-09-03): CY02 21 == 21, CY03 22 == 22, CY04 22 == 22.  A mismatch is
refused rather than silently zipped short, because a silent misalignment
would report a real 120+ cm error as agreement (see ROD_NAME_RE's docstring
in compare_master_rasbery.py for the identical concern about bank order).

Usage
-----
    python tools/make_ismr_fixed_rod_deck.py \\
        test/7_i-SMR_Validation/i-SMR_CY02.json \\
        test/7_i-SMR_Validation/Reference_output/depf_02.sum \\
        -o test/7_i-SMR_Validation/i-SMR_CY02_fixedrod.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
# REUSE, do not duplicate: parse_master_rods() and ROD_TOP_CM are
# tools/compare_master_rasbery.py's, and that is the ONLY place the MASTER
# EDIT 1 rod block is parsed anywhere in this tree.
import compare_master_rasbery as cmp_mod  # noqa: E402

ROD_TOP_CM = cmp_mod.ROD_TOP_CM

#: Sequential overlap scheme (background fact, this WP).  R2 leads R3 by
#: 120 cm of withdrawal, R4 trails it by 120 cm, both clamped to the travel
#: band.  This is a FALLBACK ONLY -- every bank the .sum states directly wins
#: over this formula, because a real MASTER-printed cm is never worse
#: evidence than an assumed overlap relationship.  Kept as a small, documented,
#: independently testable function regardless of whether a shipped .sum ever
#: needs it (none of CY02/03/04 do: all ten banks print a row).
OVERLAP_SPAN_CM = 120.0


def apply_overlap(r3_cm: float, rod_top_cm: float = ROD_TOP_CM) -> dict[str, float]:
    """R2/R4 implied by R3 under the sequential overlap scheme, clamped.

    R2 = R3 + 120, R4 = R3 - 120, each clamped to [0, rod_top_cm].  Returns a
    dict with exactly the two derived banks; R3 itself is the caller's free
    degree of freedom and is not echoed back here.
    """
    return {
        "R2": min(max(r3_cm + OVERLAP_SPAN_CM, 0.0), rod_top_cm),
        "R4": min(max(r3_cm - OVERLAP_SPAN_CM, 0.0), rod_top_cm),
    }


def cm_to_insertion(cm: float, rod_top_cm: float = ROD_TOP_CM) -> float:
    """insertion = rod_top_cm - cm (src/IO.cpp / rods/insertions convention).

    Clamped to [0, rod_top_cm]: a MASTER print rounded a hair outside the
    travel band (fully-withdrawn banks print exactly `rod_top_cm`, never
    negative cm, but this is defensive against a hand-built or corrupted
    .sum) must not hand the solver a negative insertion depth.
    """
    return min(max(rod_top_cm - cm, 0.0), rod_top_cm)


def bank_cm_for_statepoint(
    bank_names: list[str], sum_row: list[float], deck_banks: set[str],
    rod_top_cm: float = ROD_TOP_CM,
) -> dict[str, float]:
    """{bank: cm} for one MASTER row, restricted to banks the deck declares.

    Every bank present in BOTH the .sum row and the deck's own
    "rod configuration" is taken directly from the .sum.  A deck bank the
    .sum does NOT print is filled in from `apply_overlap()` when it is R2 or
    R4 and R3 is available; anything else missing is a hard error, because
    inventing a shutdown bank's position is not a fallback this tool can make
    safely.
    """
    direct = {name: cm for name, cm in zip(bank_names, sum_row) if name in deck_banks}
    missing = deck_banks - set(direct)
    if missing and "R3" in direct:
        overlap = apply_overlap(direct["R3"], rod_top_cm)
        for bank in ("R2", "R4"):
            if bank in missing and bank in overlap:
                direct[bank] = overlap[bank]
                missing.discard(bank)
    if missing:
        raise SystemExit(
            f"statepoint has no MASTER cm for bank(s) {sorted(missing)} and no "
            f"R3 to derive them from via the overlap scheme (R2=R3+120, "
            f"R4=R3-120) -- add the bank to the .sum or extend the fallback")
    return direct


#: Reserved keys IO.cpp:507-517 filters out of a "rod insertion" entry's
#: payload before reading it as bank->insertion.  Carried here ONLY as a
#: cross-check that no deck bank name collides with one of them (it would be
#: silently dropped as a reserved key rather than applied).
_RESERVED_ROD_KEYS = {
    "type", "rate", "rated power percent", "rated_power_percent", "search",
    "search_min", "search_max", "search_target", "search_tol",
    "search_max_iter", "search_relaxation", "max_eigen_iterations",
    "eigv_tolerance", "max_th_iterations", "th_tolerance", "minimum_keff",
    "minimum_carry_slope", "search_minimum_secant_denominator",
    "search_minimum_span", "search_pcm_tolerance",
    "search_slope_freeze_dx_threshold", "print",
}


def rod_insertion_entry(bank_cm: dict[str, float], rod_top_cm: float) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": "rod insertion"}
    for bank, cm in sorted(bank_cm.items()):
        entry[bank] = round(cm_to_insertion(cm, rod_top_cm), 6)
    return entry


def _strip_repeat_print(item: dict[str, Any]) -> dict[str, Any] | None:
    """What a non-first step of a repeated depletion entry keeps of "print".

    src/IO.cpp:415-423 (`finalize_entry`): only the FIRST step of a repeated
    `steps > 1` group gets the entry's full `print` block; every later step
    keeps only `node monitor` from it.  Splitting a `steps: N` entry into N
    separate `steps: 1` entries (this tool's statepoint expansion) would make
    EVERY split step "first in its own group" and over-apply `print`
    (`save`, `summary`, ...) to statepoints that never had it -- so the same
    rule is reproduced here on the split.
    """
    if "print" not in item:
        return None
    src = item["print"]
    if not isinstance(src, dict):
        return None
    kept = {k: v for k, v in src.items() if k in ("node monitor", "node_monitor")}
    return kept or None


def expand_statepoints(schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordered list of one-statepoint schedule entries, src/IO.cpp order.

    Mirrors src/IO.cpp:406-450's expansion: the leading "standard" entry is
    one statepoint, and a "depletion" entry with `steps: N` is N statepoints
    (each `steps: 1` in the output, all other keys preserved except `print`
    on steps after the first -- see `_strip_repeat_print`).  Any other entry
    type (derivative, rod insertion, ...) in the SOURCE deck is passed
    through as its own single statepoint slot, in place, so a source deck
    that already carries one is not silently dropped.
    """
    out: list[dict[str, Any]] = []
    for item in schedule:
        etype = item.get("type")
        if etype == "depletion":
            steps = int(item.get("steps", 1))
            for step_index in range(steps):
                step = copy.deepcopy(item)
                step["steps"] = 1
                if step_index > 0:
                    kept_print = _strip_repeat_print(item)
                    if kept_print is None:
                        step.pop("print", None)
                    else:
                        step["print"] = kept_print
                out.append(step)
        else:
            out.append(copy.deepcopy(item))
    return out


def force_keff_search(schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flip every "standard" entry's search mode to keff (rod search OFF).

    src/IO.cpp:475-487 only promotes `default_search` from a "standard"
    entry, and only when that entry itself carries a "search" key, so this
    is the single place rod search gets disabled deck-wide.  Any entry of
    another type carrying its OWN "search" key would locally override this
    (none of CY02/03/04 do -- see the module docstring), and is refused
    loudly rather than silently followed, since a rod-fixed deck with a
    stray per-entry rod search would not be a fixed-rod deck at all.
    """
    out = []
    for item in schedule:
        item = copy.deepcopy(item)
        if item.get("type") == "standard":
            item["search"] = "keff"
        elif "search" in item and item.get("type") != "rod insertion":
            raise SystemExit(
                f"schedule entry {item.get('type')!r} carries its own "
                f"\"search\": {item['search']!r}; force_keff_search() only "
                f"disables the cycle-wide default set on the leading "
                f"\"standard\" entry, so this override would survive it and "
                f"the output would not actually be a fixed-rod deck")
        out.append(item)
    return out


def build_fixed_rod_schedule(
    schedule: list[dict[str, Any]], bank_names: list[str],
    sum_rows: dict[float, list[float]], deck_banks: set[str],
    rod_top_cm: float = ROD_TOP_CM,
) -> list[dict[str, Any]]:
    statepoints = expand_statepoints(schedule)
    sorted_efpd = sorted(sum_rows)
    if len(statepoints) != len(sorted_efpd):
        raise SystemExit(
            f"deck expands to {len(statepoints)} statepoint(s) but the .sum "
            f"carries {len(sorted_efpd)} EDIT 1 row(s) -- refusing to zip a "
            f"mismatched grid rather than silently pairing the wrong "
            f"statepoints (deck: {len(statepoints)}, .sum EFPD: "
            f"{sorted_efpd})")

    out: list[dict[str, Any]] = []
    for sp, efpd in zip(statepoints, sorted_efpd):
        bank_cm = bank_cm_for_statepoint(
            bank_names, sum_rows[efpd], deck_banks, rod_top_cm)
        out.append(rod_insertion_entry(bank_cm, rod_top_cm))
        out.append(sp)
    return force_keff_search(out)


def load_deck(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("deck", type=Path, help="RASBERY rod-search deck (i-SMR_CY0N.json)")
    ap.add_argument("mas_sum", type=Path, help="MASTER .sum (depf_0N.sum)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output deck path (default: <stem>_fixedrod.json "
                         "alongside the source deck)")
    ap.add_argument("--rod-top-cm", type=float, default=ROD_TOP_CM,
                    help="withdrawn-bank position in cm (default: %(default)s)")
    args = ap.parse_args()

    deck = load_deck(args.deck)
    if "schedule" not in deck:
        raise SystemExit(f"{args.deck}: no 'schedule' block")
    if "rod configuration" not in deck:
        raise SystemExit(f"{args.deck}: no 'rod configuration' block -- this is "
                         f"not a rodded deck")
    deck_banks = set(deck["rod configuration"])
    collide = deck_banks & _RESERVED_ROD_KEYS
    if collide:
        raise SystemExit(
            f"deck bank name(s) {sorted(collide)} collide with a reserved "
            f"\"rod insertion\" key (src/IO.cpp filters these out of the "
            f"insertion map); rename the bank or fix the schema before "
            f"trusting this tool's output")

    bank_names, sum_rows = cmp_mod.parse_master_rods(args.mas_sum)
    if not sum_rows:
        raise SystemExit(f"{args.mas_sum}: no SUMMARY EDIT 1 rod rows parsed")
    if not (deck_banks & set(bank_names)):
        raise SystemExit(
            f"no bank name in common between the deck ({sorted(deck_banks)}) "
            f"and {args.mas_sum} ({bank_names})")

    deck["schedule"] = build_fixed_rod_schedule(
        deck["schedule"], bank_names, sum_rows, deck_banks, args.rod_top_cm)

    out = args.out or args.deck.with_name(f"{args.deck.stem}_fixedrod.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(deck, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"{len(sum_rows)} statepoint(s), rods fixed at MASTER cm from "
          f"{args.mas_sum} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
