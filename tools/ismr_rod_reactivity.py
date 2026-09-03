#!/usr/bin/env python3
"""Anchor B: turn a fixed-rod RASBERY run's keff into Δρ (pcm), per statepoint.

Companion to tools/make_ismr_fixed_rod_deck.py.  That tool pins RASBERY's rod
banks at MASTER's converged critical cm and disables rod search, so RASBERY's
keff at each statepoint of the resulting run is no longer "driven to 1.0 by a
search" -- it is a plain eigenvalue solve AT MASTER'S ROD CONFIGURATION.  With
boron pinned at 0 ppm on both codes for the whole CY02/03/04 cycle (background
fact, docs/ISMR_MASTER_COMPARISON_RUNBOOK_20260903_KO.md Sec "anchor B"), that
keff is a clean read of the reactivity the two codes' original rod-position
disagreement was worth:

    delta_rho_pcm = (k - 1) / k * 1.0e5

(the same reactivity-from-keff identity tools/compare_master_rasbery.py's
`delta_pcm` column already uses -- rho = 1 - 1/k, in pcm -- specialised to
k_master == 1 exactly, which is what "MASTER prints K-EFF == 1.000000 on
every rod-critical statepoint" means).

Reuses tools/compare_master_rasbery.py's parsers rather than re-reading the
.sum or the .h5: `parse_master_rods` (MASTER EDIT 1, cm per bank per EFPD),
`parse_rasbery_summary` (RASBERY `summary/efpd` + `summary/keff`), and
optionally `parse_rasbery_rods` (RASBERY `rods/*`) when `--orig-h5` is given
so each row can also show how far the ORIGINAL rod-search run's own bank
landed from MASTER's -- the rod-cm gap this whole tool exists to price.

Usage
-----
    python tools/ismr_rod_reactivity.py \\
        test/7_i-SMR_Validation/Reference_output/depf_02.sum \\
        out_CY02_fixedrod.h5

    # with the rod-cm gap for context (the ORIGINAL rod-search run's h5):
    python tools/ismr_rod_reactivity.py depf_02.sum out_CY02_fixedrod.h5 \\
        --orig-h5 out_CY02_gpu.h5 -o cmp_CY02_rho
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# REUSE: MASTER EDIT 1 parsing, RASBERY summary parsing and RASBERY rod
# parsing all live in compare_master_rasbery.py; nothing here re-parses a
# .sum or an .h5.
import compare_master_rasbery as cmp_mod  # noqa: E402


def delta_rho_pcm(keff: float) -> float:
    """(k - 1) / k * 1e5 -- the pcm reactivity a keff away from 1 represents."""
    return (keff - 1.0) / keff * 1.0e5


def resolve_efpd_offset(spec: str, ras_efpd: list[float], mas_efpd: list[float]) -> float:
    """Same `auto` convention as compare_master_rasbery.py's --efpd-offset."""
    if str(spec).strip().lower() == "auto":
        return min(ras_efpd) - min(mas_efpd)
    try:
        return float(spec)
    except ValueError:
        raise SystemExit(
            f"--efpd-offset {spec!r} is neither a number nor `auto`") from None


def join_reactivity(
    mas_names: list[str], mas_rows: dict[float, list[float]],
    ras_summary: dict[float, dict[str, float]], *,
    efpd_tol: float = 1e-3, offset: float = 0.0,
    orig_names: list[str] | None = None,
    orig_rows: dict[float, list[float]] | None = None,
) -> list[dict]:
    """One row per EFPD-matched statepoint: efpd, ras_keff, delta_rho_pcm,
    master_rod_cm_<BANK> for every bank MASTER prints, and (when `orig_rows`
    is given) delta_rod_cm_<BANK> = original-run cm - MASTER cm for every
    bank both sides name.
    """
    ras = {e - offset: row for e, row in ras_summary.items()} if offset else dict(ras_summary)
    orig = None
    if orig_rows is not None:
        orig = {e - offset: row for e, row in orig_rows.items()} if offset else dict(orig_rows)

    joined: list[dict] = []
    for e_r, rrow in sorted(ras.items()):
        if "ras_keff" not in rrow:
            continue
        match_efpd = next((e_m for e_m in mas_rows if abs(e_m - e_r) <= efpd_tol), None)
        if match_efpd is None:
            continue
        k = rrow["ras_keff"]
        row: dict = {
            "efpd": e_r,
            "ras_keff": k,
            "delta_rho_pcm": delta_rho_pcm(k),
        }
        cm_row = mas_rows[match_efpd]
        for name, cm in zip(mas_names, cm_row):
            row[f"master_rod_cm_{name}"] = cm
        if orig is not None and orig_names:
            o_efpd = next((e_o for e_o in orig if abs(e_o - e_r) <= efpd_tol), None)
            if o_efpd is not None:
                o_row = orig[o_efpd]
                for name, cm in zip(mas_names, cm_row):
                    if name in orig_names:
                        oi = orig_names.index(name)
                        if oi < len(o_row):
                            row[f"delta_rod_cm_{name}"] = o_row[oi] - cm
        joined.append(row)
    return joined


def summarize(joined: list[dict]) -> tuple[float, float]:
    """(rms, max|delta_rho_pcm|) over the joined statepoints."""
    vals = [r["delta_rho_pcm"] for r in joined]
    if not vals:
        return 0.0, 0.0
    rms = math.sqrt(sum(v * v for v in vals) / len(vals))
    worst = max(abs(v) for v in vals)
    return rms, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mas_sum", type=Path, help="MASTER .sum (depf_0N.sum)")
    ap.add_argument("h5", type=Path,
                    help="RASBERY output .h5 from the FIXED-ROD deck "
                         "(tools/make_ismr_fixed_rod_deck.py's output, run)")
    ap.add_argument("--orig-h5", type=Path, default=None,
                    help="RASBERY output .h5 from the ORIGINAL rod-search "
                         "deck, for delta_rod_cm_<BANK> context (optional)")
    ap.add_argument("--rod-top-cm", type=float, default=cmp_mod.ROD_TOP_CM)
    ap.add_argument("--efpd-tol", type=float, default=1e-3)
    ap.add_argument("--efpd-offset", default="0",
                    help="subtract this many EFPD from the RASBERY side "
                         "before joining (default: 0). `auto` aligns the "
                         "first statepoints, same as compare_master_rasbery.py")
    ap.add_argument("-o", "--out-prefix", default=None,
                    help="also write <prefix>.csv (default: print only)")
    args = ap.parse_args()

    mas_names, mas_rows = cmp_mod.parse_master_rods(args.mas_sum)
    if not mas_rows:
        raise SystemExit(f"{args.mas_sum}: no SUMMARY EDIT 1 rod rows parsed")
    ras_summary = cmp_mod.parse_rasbery_summary(args.h5)
    if not ras_summary:
        raise SystemExit(f"{args.h5}: no summary statepoints")

    orig_names: list[str] = []
    orig_rows: dict[float, list[float]] = {}
    if args.orig_h5:
        orig_names, orig_rows = cmp_mod.parse_rasbery_rods(args.orig_h5, args.rod_top_cm)

    if str(args.efpd_offset).strip().lower() == "auto":
        offset = resolve_efpd_offset("auto", list(ras_summary), list(mas_rows))
        print(f"efpd offset: auto -> {offset:.6f} d")
    else:
        offset = resolve_efpd_offset(args.efpd_offset, [], [])

    joined = join_reactivity(
        mas_names, mas_rows, ras_summary, efpd_tol=args.efpd_tol, offset=offset,
        orig_names=orig_names or None, orig_rows=orig_rows or None)
    if not joined:
        raise SystemExit("no EFPD-matched statepoints between MASTER .sum and "
                         "the RASBERY .h5")

    print(f"anchor B: fixed-rod keff -> pcm  ({len(joined)} statepoint(s))")
    header = f"{'efpd':>10}  {'ras_keff':>10}  {'delta_rho_pcm':>14}"
    print(header)
    for r in joined:
        print(f"{r['efpd']:10.3f}  {r['ras_keff']:10.6f}  {r['delta_rho_pcm']:14.3f}")

    rms, worst = summarize(joined)
    print(f"\nrms(delta_rho_pcm)  = {rms:.3f}")
    print(f"max|delta_rho_pcm|  = {worst:.3f}")

    if args.out_prefix:
        import csv
        fieldnames = ["efpd", "ras_keff", "delta_rho_pcm"]
        for name in mas_names:
            col = f"master_rod_cm_{name}"
            if any(col in r for r in joined):
                fieldnames.append(col)
            dcol = f"delta_rod_cm_{name}"
            if any(dcol in r for r in joined):
                fieldnames.append(dcol)
        csv_path = Path(f"{args.out_prefix}.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(joined)
        print(f"\nwrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
