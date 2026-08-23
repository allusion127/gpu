#!/usr/bin/env python3
"""Step-aligned MASTER MAS_SUM vs RASBERY HDF5 statepoint comparator.

Ported from the rev00 tree (gad_rasbery_case/compare_master_rasbery.py) and
modernised for the current output layout:

* h5py instead of an h5dump subprocess;
* EFPD-based join instead of a step-index join, so decks whose automatic
  natural-EOC stepping produces different state counts still align;
* MAS_SUM files that ACCUMULATE several runs (MASTER appends SUMMARY EDIT
  blocks) are handled by parsing only the LAST run's blocks;
* optional datasets (T-H averages, bu) are compared only when the .h5 has them.

Usage:
    python compare_master_rasbery.py MAS_SUM result.h5 [-o out_prefix]
Writes <prefix>.csv and <prefix>.md (default prefix: master_vs_rasbery).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import h5py


def parse_master_summary(path: Path) -> dict[float, dict[str, float]]:
    """Return {efpd: row} from the LAST run's SUMMARY EDIT 2/3/4 blocks."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    # MASTER appends whole summary sets per run; keep only the final run.
    first_marker = "SUMMARY EDIT 1 :"
    last_run = text.rfind(first_marker)
    if last_run >= 0:
        text = text[last_run:]

    rows: dict[float, dict[str, float]] = {}
    section = None
    for line in text.splitlines():
        if "SUMMARY EDIT 2 : REACTIVITY" in line:
            section = "reactivity"
            continue
        if "SUMMARY EDIT 3 : AO, XE AND PEAK VALUES" in line:
            section = "peaks"
            continue
        if "SUMMARY EDIT 4 : THERMAL-HYDRAULIC VALUES" in line:
            section = "th"
            continue
        if "SUMMARY EDIT" in line:
            section = None
            continue
        if section is None or not re.match(r"^\s*\d+\s+[\d.]", line):
            continue
        parts = line.split()
        try:
            efpd = float(parts[2])
        except (IndexError, ValueError):
            continue
        row = rows.setdefault(efpd, {})
        if section == "reactivity" and len(parts) >= 8:
            row.update(
                master_day=float(parts[1]),
                master_bu=float(parts[3]),
                master_ppm=float(parts[6]),
                master_keff=float(parts[7]),
            )
        elif section == "peaks" and len(parts) >= 8:
            row.update(
                master_ao=float(parts[3]),
                master_fqn=float(parts[4]),
                master_frn=float(parts[5]),
                master_fqp=float(parts[6]),
                master_frp=float(parts[7]),
            )
        elif section == "th" and len(parts) >= 9:
            row.update(
                master_tfavg_c=float(parts[3]),
                master_tmavg_c=float(parts[5]),
                master_dmavg=float(parts[7]),
            )
    return rows


#: summary dataset -> (column name, unit transform)
RAS_DATASETS = {
    "ppm": ("ras_ppm", None),
    "keff": ("ras_keff", None),
    "ao": ("ras_ao", None),
    "asi": ("ras_asi", None),
    "fqn": ("ras_fqn", None),
    "frn": ("ras_frn", None),
    "fqp": ("ras_fqp", None),
    "frp": ("ras_frp", None),
    "bu_avg": ("ras_bu", None),
    "tf_avg": ("ras_tfavg_c", lambda v: v - 273.15),
    "tm_avg": ("ras_tmavg_c", lambda v: v - 273.15),
    "dm_avg": ("ras_dmavg", None),
}


def parse_rasbery_summary(path: Path) -> dict[float, dict[str, float]]:
    rows: dict[float, dict[str, float]] = {}
    with h5py.File(path, "r") as f:
        su = f["summary"]
        efpd = su["efpd"][()]
        cols = {}
        for ds, (name, xf) in RAS_DATASETS.items():
            if ds in su:
                v = su[ds][()]
                cols[name] = [xf(float(x)) if xf else float(x) for x in v]
        for i, e in enumerate(efpd):
            rows[float(e)] = {name: vals[i] for name, vals in cols.items()}
    return rows


DELTA_PAIRS = [
    # (master col, ras col, delta col, scale)
    ("master_bu", "ras_bu", "delta_bu", 1.0),
    ("master_ppm", "ras_ppm", "delta_ppm", 1.0),
    ("master_keff", "ras_keff", "delta_pcm", 1.0e5),
    ("master_ao", "ras_ao", "delta_ao", 1.0),
    ("master_fqn", "ras_fqn", "delta_fqn", 1.0),
    ("master_frn", "ras_frn", "delta_frn", 1.0),
    ("master_fqp", "ras_fqp", "delta_fqp", 1.0),
    ("master_frp", "ras_frp", "delta_frp", 1.0),
    ("master_tfavg_c", "ras_tfavg_c", "delta_tfavg_c", 1.0),
    ("master_tmavg_c", "ras_tmavg_c", "delta_tmavg_c", 1.0),
    ("master_dmavg", "ras_dmavg", "delta_dmavg", 1.0),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mas_sum", type=Path)
    ap.add_argument("h5", type=Path)
    ap.add_argument("-o", "--out-prefix", default="master_vs_rasbery")
    ap.add_argument("--efpd-tol", type=float, default=1e-3,
                    help="EFPD match tolerance for joining states")
    args = ap.parse_args()

    master = parse_master_summary(args.mas_sum)
    ras = parse_rasbery_summary(args.h5)

    joined = []
    for e_r, rrow in sorted(ras.items()):
        match = None
        for e_m, mrow in master.items():
            if abs(e_m - e_r) <= args.efpd_tol:
                match = mrow
                break
        if match is None:
            continue
        row: dict[str, float] = {"efpd": e_r}
        row.update(match)
        row.update(rrow)
        for mc, rc, dc, scale in DELTA_PAIRS:
            if mc in row and rc in row:
                if dc == "delta_pcm":
                    row[dc] = (1.0 / row[mc] - 1.0 / row[rc]) * scale
                else:
                    row[dc] = (row[rc] - row[mc]) * scale
        joined.append(row)

    if not joined:
        raise SystemExit("no EFPD-matched states between the two inputs")

    fieldnames = ["efpd"]
    for mc, rc, dc, _ in DELTA_PAIRS:
        for c in (mc, rc, dc):
            if any(c in r for r in joined) and c not in fieldnames:
                fieldnames.append(c)

    csv_path = Path(f"{args.out_prefix}.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(joined)

    md_path = Path(f"{args.out_prefix}.md")
    key_cols = ["efpd", "delta_ppm", "delta_pcm", "delta_ao", "delta_fqp", "delta_frp"]
    key_cols = [c for c in key_cols if any(c in r for r in joined)]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(key_cols) + " |\n")
        f.write("|" + "---|" * len(key_cols) + "\n")
        for r in joined:
            f.write("| " + " | ".join(f"{r.get(c, float('nan')):.3f}" for c in key_cols) + " |\n")
        maxes = {c: max(abs(r[c]) for r in joined if c in r) for c in key_cols[1:]}
        f.write("\nmax|delta|: " + ", ".join(f"{c}={v:.3f}" for c, v in maxes.items()) + "\n")

    print(f"{len(joined)} states joined -> {csv_path}, {md_path}")
    for c, v in {c: max(abs(r[c]) for r in joined if c in r) for c in key_cols[1:]}.items():
        print(f"  max|{c}| = {v:.3f}")


if __name__ == "__main__":
    main()
