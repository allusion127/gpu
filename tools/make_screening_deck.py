#!/usr/bin/env python3
"""Build reduced-fidelity RASBERY screening decks from a full depletion deck.

A GA generation is 128 full-cycle depletions.  Wall is set by how many
statepoints the schedule emits, so the cheapest lever on generation time is the
burnup schedule -- not the solver.  This tool rewrites the ``schedule`` block of
a deck into a coarse burnup grid while leaving every physics setting
(search type/tolerance, xenon, TH, convergence, geometry, core, batch) exactly
as the source deck had it.

Why burnup increments and not days
----------------------------------
``src/IO.cpp:423`` accepts ``burnup`` / ``bu`` / ``burnup_increment`` on a
depletion entry and ``src/Scheduler.h:311`` converts it at runtime with
``time = burnup * core_hm_kg / actual_power``.  A grid written in GWd/t is
therefore invariant to the heavy-metal loading of the candidate, which is
exactly what a loading-pattern GA needs: every candidate is compared at the same
burnup, not at the same number of days.

What is dropped, deliberately
-----------------------------
The full deck ends its depletion with ``"until boron ppm": 10.0``
(``src/IO.cpp:441``), which re-queues the entry at runtime until the converged
critical boron reaches the target.  That makes the statepoint count -- and thus
the wall -- a function of the candidate.  A screening arm must have a fixed
cost, so the generated decks stop at a fixed terminal burnup and cycle length is
recovered by boron extrapolation in ``tools/ga_fitness.py``.  Pass
``--keep-until-boron`` to restore the natural-EOC tail (and give up the fixed
cost).

Usage
-----
    python tools/make_screening_deck.py kngr_238.json --mode coarse -o coarse.json
    python tools/make_screening_deck.py kngr_238.json --mode three  -o three.json
    python tools/make_screening_deck.py kngr_238.json \
        --burnups 0.5,1,2,4,8,12,16 --pin-output boc-eoc -o custom.json

Tolerance flags are NOT written into the deck.  The A2 staged-tolerance arm is
environment-driven and documented in ``docs/A2_OUTER_REDUCTION_20260829_KO.md``;
``--print-loose-env`` echoes the block so a runner can export it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Reference point for the wall model.  238 server, single deck, v3 arm:
# 35 statepoints in 16.9 s (docs/V3_FREEZE_20260829_KO.md Sec 4 records 16.3 s
# for the v3 arm and 55.35 s for the v2 freeze; 16.9 s is the campaign's
# working single-deck number).
REF_STATEPOINTS = 35
REF_WALL_S = 16.9
# Batch M64 candidates/hour at full fidelity.  524 c/h is the campaign premise
# (128 candidates in 14.7 min).  docs/V3_FREEZE_20260829_KO.md Sec 4 records
# 217 case/h MEASURED for the v2 batch golden and marks the v3 batch arm
# unmeasured -- see docs/GA_SCREENING_ARM_20260831_KO.md Sec 2.
REF_BATCH_CH = 524.0

# Default coarse grid, cumulative GWd/t past BOC.  Tight early (Gd/Xe/Sm
# transient and the boron peak both land inside 2 GWd/t), then linear.
COARSE_BURNUPS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0]
# BOC / mid / near-EOC.
THREE_BURNUPS = [8.0, 16.0]

# ---------------------------------------------------------------------------
# MEASURED, 2026-08-31 (local WSL, GTX 1080 Ti, kngr_238 v3 arm).  See
# docs/GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md Sec 2.2 and 2.5.
#
# The statepoint-proportional and solve-unit models below are BOTH WRONG for
# coarse grids, and the counter-example is not marginal:
#
#   deck     statepoints   outers   drive(s)
#   full            35      4,609     42.07
#   coarse          10      2,334     18.77      <- 2.24x cheaper
#   three            3      5,104     25.76      <- MORE outers than the full deck
#
# The `three` grid burns 8 GWd/t (207 EFPD) in one step; its terminal statepoint
# alone costs 4,798 outers / 18.87 s, because the boron secant search and the
# equilibrium-xenon cascade both start from the previous statepoint's solution
# and the cascade is re-armed at every committed trial.  Cost is SUPERLINEAR in
# the burnup step, so cutting statepoints past a point makes the case dearer.
#
# The measured per-statepoint law is two-parameter, and it fits both decks:
#
#   t_statepoint = c + d x outers        c = 0.474 s, d = 4.805 ms   (35-sp deck)
#                                        c = 0.538 s, d = 4.720 ms   (10-sp deck)
#
# `c` (PPR + CRAM depletion + FlatXS + T/H, all host) is INDEPENDENT of outers,
# so it is what a coarse grid actually removes.  `d x outers` is not predicted
# by the grid alone -- calibrate it with --calibrate, and until then treat the
# wall numbers printed below as an UPPER BOUND ON THE SAVING, never a promise.
WARN_BURNUP_STEP_GWD = 4.0
MEASURED_LOCAL = {
    "full":   {"statepoints": 35, "outers": 4609, "drive_s": 42.07},
    "coarse": {"statepoints": 10, "outers": 2334, "drive_s": 18.77},
    "three":  {"statepoints": 3,  "outers": 5104, "drive_s": 25.76},
}
# ---------------------------------------------------------------------------

DEPLETION_TIME_KEYS = ("time", "steps", "substeps", "burnup", "bu",
                       "burnup_increment", "burnup increment",
                       "until boron ppm", "until_boron_ppm")

LOOSE_ENV = (
    "RASBERY_STAGED_FLUX_TOL=50",
    "RASBERY_STAGED_XE_TOL=1000",
    "RASBERY_STAGED_LOOSE_SETTLE=1",
)


def load_deck(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_schedule(schedule: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (prologue, depletion entries, epilogue).

    Prologue is everything before the first depletion entry (the ``standard``
    configuration step, rod insertions, ...).  Epilogue is everything after the
    last depletion entry.  Only the depletion run is rewritten.
    """
    first = last = None
    for i, item in enumerate(schedule):
        if item.get("type") == "depletion":
            if first is None:
                first = i
            last = i
    if first is None:
        raise SystemExit("deck has no depletion entry -- nothing to coarsen")
    return schedule[:first], schedule[first:last + 1], schedule[last + 1:]


def depletion_template(entries: list[dict]) -> dict[str, Any]:
    """The most common depletion entry, minus its time/burnup keys.

    The full KNGR deck varies only ``time`` and ``fuel temperature rise scale``
    across its depletion entries (the first one carries the BOC ramp value).
    Taking the modal settings keeps the screening deck on the steady-state
    values instead of the ramp outlier.
    """
    buckets: dict[str, list[dict]] = {}
    for item in entries:
        key = json.dumps({k: v for k, v in item.items()
                          if k not in DEPLETION_TIME_KEYS}, sort_keys=True)
        buckets.setdefault(key, []).append(item)
    winner = max(buckets.values(), key=len)[0]
    return {k: v for k, v in winner.items() if k not in DEPLETION_TIME_KEYS}


def set_pin_output(entry: dict[str, Any], enabled: bool) -> None:
    pr = entry.setdefault("print", {})
    for key in ("pin-wise information", "pin-wise infomration",
                "pin-wise power information", "pin-wise power infomration",
                "pin information"):
        if key in pr:
            pr[key] = enabled
    if not any(k in pr for k in ("pin-wise information", "pin information")):
        pr["pin-wise information"] = enabled


def substeps_for(delta_bu: float, max_substep_bu: float) -> int:
    if max_substep_bu <= 0.0:
        return 1
    return max(1, int(math.ceil(delta_bu / max_substep_bu - 1.0e-9)))


def build_schedule(deck: dict[str, Any], burnups: list[float], *,
                   pin_output: str, max_substep_bu: float,
                   keep_until_boron: bool, eoc_restate: bool) -> list[dict]:
    prologue, depl, epilogue = split_schedule(deck["schedule"])
    template = depletion_template(depl)
    until = None
    for item in depl:
        for key in ("until boron ppm", "until_boron_ppm"):
            if key in item:
                until = (key, item[key])

    out: list[dict] = [json.loads(json.dumps(e)) for e in prologue]
    # The prologue's own statepoint is the BOC state; only it carries pin output
    # under the "boc-eoc" policy.
    for entry in out:
        if pin_output == "off":
            set_pin_output(entry, False)
        elif pin_output == "boc-eoc":
            set_pin_output(entry, True)

    prev = 0.0
    for idx, cum in enumerate(burnups):
        delta = cum - prev
        if delta <= 0.0:
            raise SystemExit(f"burnup grid must be strictly increasing: {burnups}")
        entry = json.loads(json.dumps(template))
        entry["type"] = "depletion"
        entry["steps"] = 1
        entry["burnup"] = round(delta, 10)
        sub = substeps_for(delta, max_substep_bu)
        if sub > 1:
            entry["substeps"] = sub
        last = idx == len(burnups) - 1
        if pin_output == "off":
            set_pin_output(entry, False)
        elif pin_output == "boc-eoc":
            set_pin_output(entry, last)
        if last and keep_until_boron and until is not None:
            entry[until[0]] = until[1]
        out.append(entry)
        prev = cum

    if eoc_restate:
        restate = json.loads(json.dumps(template))
        restate["type"] = "depletion"
        restate["steps"] = 1
        restate["time"] = 0.0
        if pin_output == "off":
            set_pin_output(restate, False)
        elif pin_output == "boc-eoc":
            set_pin_output(restate, True)
        out.append(restate)

    out.extend(json.loads(json.dumps(e)) for e in epilogue)
    return out


def count_cost(schedule: list[dict]) -> tuple[int, int, bool]:
    """(statepoints, solve units, variable) for a concrete schedule.

    ``src/Driver.h:4014`` runs, per depletion statepoint, ``nsub`` predictor
    solves plus ``nsub-1`` re-BOS solves plus one final solve == ``2*nsub``
    SolveLoop calls.  Every other entry type is a single final solve.
    ``variable`` is True when an ``until boron ppm`` tail makes the count
    candidate-dependent.
    """
    n_sp = 0
    solves = 0
    variable = False
    for item in schedule:
        steps = int(item.get("steps", 1)) if item.get("type") == "depletion" else 1
        n_sp += steps
        if item.get("type") == "depletion":
            sub = max(1, int(item.get("substeps", 1)))
            solves += steps * 2 * sub
            if "until boron ppm" in item or "until_boron_ppm" in item:
                variable = True
        else:
            solves += steps
    return n_sp, solves, variable


def report(name: str, schedule: list[dict], ref_sp: int, ref_solves: int,
           args: argparse.Namespace, *, expanded: tuple[int, int] | None = None
           ) -> dict[str, Any]:
    n_sp, solves, variable = count_cost(schedule)
    if expanded is not None:
        # The until-boron tail re-queues at runtime; the static count under-reports.
        n_sp, solves = expanded
    fixed = args.fixed_overhead_s
    variable_ref = max(args.ref_wall - fixed, 1.0e-9)

    wall_sp = fixed + variable_ref * n_sp / ref_sp
    wall_solve = fixed + variable_ref * solves / ref_solves
    # Batch throughput scales on the variable part only; the per-case fixed cost
    # is amortised differently in a batch, so report the solve-unit model.
    per_case_full = 3600.0 / args.ref_batch_ch
    per_case = fixed + max(per_case_full - fixed, 1.0e-9) * solves / ref_solves
    batch_ch = 3600.0 / per_case
    gen_s = args.population / batch_ch * 3600.0

    tag = f"{name}{' (VARIABLE: until-boron tail)' if variable else ''}"
    print(f"  {tag}")
    print(f"    statepoints        : {n_sp}"
          + ("+ (until-boron requeues)" if variable else "")
          + f"   [full deck: {ref_sp}]")
    print(f"    solve units (2*sub): {solves}   [full deck: {ref_solves}]"
          f"   ratio {solves / ref_solves:.4f}")
    print(f"    wall, N_sp model   : {wall_sp:6.2f} s  "
          f"(= {fixed:.2f} + {variable_ref:.2f} x {n_sp}/{ref_sp})")
    print(f"    wall, solve model  : {wall_solve:6.2f} s  "
          f"(= {fixed:.2f} + {variable_ref:.2f} x {solves}/{ref_solves})")
    print(f"    batch M64          : {batch_ch:8.1f} c/h  "
          f"({per_case:.2f} s/case)   [full: {args.ref_batch_ch:.1f} c/h]")
    print(f"    generation of {args.population:<4d} : {gen_s / 60.0:6.2f} min "
          f"on 1 GPU  ({gen_s / 60.0 / 4:.2f} min on 4)")
    return {
        "name": name,
        "statepoints": n_sp,
        "statepoints_variable": variable,
        "solve_units": solves,
        "solve_ratio": solves / ref_solves,
        "wall_s_sp_model": wall_sp,
        "wall_s_solve_model": wall_solve,
        "batch_c_per_h": batch_ch,
        "batch_s_per_case": per_case,
        "generation_s": gen_s,
        "population": args.population,
    }


def parse_burnups(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split()]


def burnup_step_warning(burnups: list[float]) -> str | None:
    """The measured non-monotonicity, stated where a grid can trip over it.

    Returns None for a grid whose every step is inside the measured-safe
    envelope, and a printable warning otherwise.  This is not a model: it
    reports the one grid that was measured to be dearer than the full deck and
    the step size at which it happened.
    """
    if not burnups:
        return None
    steps = [b - a for a, b in zip([0.0] + burnups[:-1], burnups)]
    worst = max(steps)
    if worst <= WARN_BURNUP_STEP_GWD:
        return None
    m = MEASURED_LOCAL
    return (
        f"\n  WARNING: largest burnup step is {worst:g} GWd/t "
        f"(> {WARN_BURNUP_STEP_GWD:g} GWd/t).\n"
        "    The wall numbers above assume cost falls with statepoint count.  "
        "It does not.\n"
        f"    MEASURED (local, kngr_238): the 3-statepoint grid (8 GWd/t steps) "
        f"ran {m['three']['outers']:,} outers\n"
        f"    against the {m['full']['statepoints']}-statepoint full deck's "
        f"{m['full']['outers']:,} -- MORE work from FEWER statepoints, because "
        "the boron\n"
        "    search and the xenon cascade restart from the previous "
        "statepoint's solution and\n"
        "    both grow superlinearly with the burnup step.  The "
        f"{m['coarse']['statepoints']}-statepoint 'coarse' grid\n"
        f"    (max step 3 GWd/t) is the measured optimum at "
        f"{m['coarse']['outers']:,} outers.\n"
        "    Calibrate this grid before believing its wall: run it once and "
        "read the outer\n"
        "    count from [RASBERY][TRAJECTORY], or fold a log with "
        "tools/case_cost_profile.py.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck", type=Path, help="full-fidelity source deck JSON")
    ap.add_argument("--mode", choices=("coarse", "three", "custom", "full"),
                    default="coarse")
    ap.add_argument("--burnups", type=str, default=None,
                    help="cumulative GWd/t grid past BOC, e.g. 0.5,1,2,4,8,12,16 "
                         "(implies --mode custom)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output deck path (default: alongside the source, "
                         "<stem>_<mode>.json)")
    ap.add_argument("--pin-output", choices=("keep", "off", "boc-eoc"),
                    default="keep",
                    help="pin-wise HDF5/CSV output policy.  fqp/frp stay in "
                         "summary either way -- PPR runs unconditionally "
                         "(src/Driver.h:4065) and only the dataset write is "
                         "gated (src/IO.cpp:1499).  'off' drops pin_power, so "
                         "ga_fitness.py falls back to the nodal Fxy proxy.")
    ap.add_argument("--max-substep-burnup", type=float, default=0.0,
                    help="cap the burnup consumed per depletion substep (GWd/t)."
                         "  0 = one substep per statepoint (cheapest).  Each "
                         "substep costs 2 SolveLoop calls.")
    ap.add_argument("--keep-until-boron", action="store_true",
                    help="re-attach the source deck's 'until boron ppm' target "
                         "to the terminal entry (natural EOC, variable cost)")
    ap.add_argument("--eoc-restate", action="store_true",
                    help="append the source deck's zero-time EOC re-statepoint")
    ap.add_argument("--absolutize-data", action="store_true",
                    help="rewrite data/* paths absolute against the source deck "
                         "directory so the output can live anywhere")
    ap.add_argument("--population", type=int, default=128,
                    help="GA population per generation for the timing table")
    ap.add_argument("--ref-wall", type=float, default=REF_WALL_S,
                    help=f"single-deck wall of the full deck (default {REF_WALL_S} s)")
    ap.add_argument("--ref-batch-ch", type=float, default=REF_BATCH_CH,
                    help=f"batch M64 c/h at full fidelity (default {REF_BATCH_CH})")
    ap.add_argument("--fixed-overhead-s", type=float, default=0.0,
                    help="per-run cost that does not scale with statepoints "
                         "(process start, XS load, geometry build).  Measure it "
                         "as (wall_3pt*N_full - wall_full*N_3pt)/(N_full-N_3pt).")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the cost report as JSON")
    ap.add_argument("--print-loose-env", action="store_true",
                    help="echo the A2 staged-tolerance env block and exit")
    args = ap.parse_args()

    if args.print_loose_env:
        print("export " + " ".join(LOOSE_ENV))
        print("# docs/A2_OUTER_REDUCTION_20260829_KO.md -- default OFF, "
              "never baked into a deck")
        return

    deck = load_deck(args.deck)
    if "schedule" not in deck:
        raise SystemExit(f"{args.deck}: no 'schedule' block")

    ref_sp, ref_solves, ref_variable = count_cost(deck["schedule"])
    print(f"source deck: {args.deck}")
    if ref_variable:
        # The until-boron tail already ran once in the reference, so the honest
        # reference count is the emitted statepoint count, not the deck's
        # nominal entry count.  Ask for it explicitly.
        print(f"  NOTE: source schedule has an until-boron tail; its {ref_sp} "
              f"nominal entries expanded to {REF_STATEPOINTS} statepoints in the "
              f"reference run.  Using {REF_STATEPOINTS} / "
              f"{ref_solves - ref_sp + REF_STATEPOINTS} as the reference.")
        ref_solves = ref_solves - ref_sp + REF_STATEPOINTS
        ref_sp = REF_STATEPOINTS
    print()

    if args.burnups:
        burnups = parse_burnups(args.burnups)
        mode = "custom"
    elif args.mode == "coarse":
        burnups, mode = list(COARSE_BURNUPS), "coarse"
    elif args.mode == "three":
        burnups, mode = list(THREE_BURNUPS), "three"
    elif args.mode == "full":
        burnups, mode = [], "full"
    else:
        raise SystemExit("--mode custom requires --burnups")

    print("cost model (fixed overhead "
          f"{args.fixed_overhead_s:.2f} s, ref {args.ref_wall:.2f} s / "
          f"{ref_sp} sp / {ref_solves} solves):")
    report("full (source, as run)", deck["schedule"], ref_sp, ref_solves, args,
           expanded=(ref_sp, ref_solves) if ref_variable else None)

    if mode == "full":
        out_schedule = deck["schedule"]
    else:
        out_schedule = build_schedule(
            deck, burnups,
            pin_output=args.pin_output,
            max_substep_bu=args.max_substep_burnup,
            keep_until_boron=args.keep_until_boron,
            eoc_restate=args.eoc_restate)

    label = f"{mode} (BOC + {', '.join(f'{b:g}' for b in burnups)} GWd/t)" \
        if burnups else "full (copy)"
    stats = report(label, out_schedule, ref_sp, ref_solves, args)
    warning = burnup_step_warning(burnups)
    if warning:
        print(warning)
    stats["burnup_step_warning"] = warning
    stats["measured_local_reference"] = MEASURED_LOCAL
    stats["mode"] = mode
    stats["burnups_gwd_t"] = burnups
    stats["pin_output"] = args.pin_output
    stats["source_deck"] = str(args.deck)
    stats["ref_statepoints"] = ref_sp
    stats["ref_solve_units"] = ref_solves

    deck["schedule"] = out_schedule
    if args.absolutize_data and "data" in deck:
        base = args.deck.resolve().parent
        for key, value in list(deck["data"].items()):
            if isinstance(value, str) and not value.startswith("/"):
                deck["data"][key] = str((base / value).as_posix())

    out = args.out or args.deck.with_name(f"{args.deck.stem}_{mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(deck, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"\nwrote {out}")
    stats["out"] = str(out)

    if args.json:
        args.json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
