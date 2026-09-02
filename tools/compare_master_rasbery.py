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
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_b_envelope  # noqa: E402


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


def column_maxes(joined: list[dict], key_cols: list[str]) -> dict[str, float]:
    """max|delta| per column, skipping a column no joined row actually carries.

    `max(abs(r[c]) for r in joined if c in r)` raises ValueError on an empty
    generator, and that is reachable: a column can survive the `any(c in r)`
    filter that builds the table and still be absent from every row this
    reduction sees.  A gate whose first act is a traceback is a gate nobody can
    read a verdict off, and the new nonzero exit makes that reachable from a
    `set -e` runbook.
    """
    out: dict[str, float] = {}
    for c in key_cols[1:]:
        values = [abs(r[c]) for r in joined if c in r]
        if values:
            out[c] = max(values)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mas_sum", type=Path)
    ap.add_argument("h5", type=Path)
    ap.add_argument("-o", "--out-prefix", default="master_vs_rasbery")
    ap.add_argument("--efpd-tol", type=float, default=1e-3,
                    help="EFPD match tolerance for joining states")
    # WP24.  This tool used to print max|delta| and exit 0 whatever it found, so
    # the production envelope (1.905 pcm / 15.309 ppm / 0.013 AO) lived only in
    # docs/ and nothing enforced it.  The default is `production`, so no
    # existing invocation silently gets the looser screening envelope; what
    # changes for existing callers is that a breach now exits nonzero.
    gate_b_envelope.add_envelope_argument(ap)
    args = ap.parse_args()
    envelope = gate_b_envelope.resolve(args.envelope)

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
        maxes = column_maxes(joined, key_cols)
        f.write("\nmax|delta|: " + ", ".join(f"{c}={v:.3f}" for c, v in maxes.items()) + "\n")
        # The envelope NAME goes in the .md, not only on the terminal: a table
        # of deltas with no envelope beside it is a table that can be quoted
        # into the wrong column six months later.
        f.write(f"\nenvelope: {envelope.name} -- {envelope.note}\n")

    print(f"{len(joined)} states joined -> {csv_path}, {md_path}")
    maxes = column_maxes(joined, key_cols)
    for c, v in maxes.items():
        print(f"  max|{c}| = {v:.3f}")

    # THE JOIN FROM COLUMN NAMES TO ENVELOPE METRICS.  A column the deck does
    # not produce -- `delta_ppm` on a deck with no boron search -- is simply
    # absent, and gate_b_envelope.verdict() skips what it was not given rather
    # than assuming a pass.
    measured = {}
    pinned = []
    for column, metric in (("delta_pcm", "keff_pcm"), ("delta_ppm", "ppm"),
                           ("delta_ao", "ao")):
        if column in maxes:
            measured[metric] = maxes[column]
            # A COLUMN THAT IS IDENTICALLY ZERO ON EVERY JOINED STATEPOINT WAS
            # NOT MEASURED, IT WAS FIXED.  The case this exists for is the
            # rod-crit deck families (iSMR / CY): the deck STATES the boron and
            # searches the rod, so `delta_ppm` is one constant minus the same
            # constant on every row and is exactly 0.000 whatever the physics
            # did.  Scoring it and calling the result PASS is a verdict about
            # the deck's input file.  Reported to gate_b_envelope as `pinned`,
            # which keeps the row visible, keeps a nonzero breach a failure, and
            # refuses to call a run SCORED when every judged column is one of
            # these.
            #
            # THE TEST IS EXACT EQUALITY WITH ZERO, deliberately: a near-zero
            # column is a real agreement and must keep scoring.  So this can
            # only ever UNDER-report a pinned column (both sides printing the
            # same quantity at different precision), which leaves the previous
            # behaviour rather than inventing a NOT SCORED.
            if maxes[column] == 0.0:
                pinned.append(metric)
    # gate_b_envelope.report() REFUSES to print PASS when nothing was scored.
    # That is what stops a compare run producing none of the three columns from
    # printing "GATE B scalars: PASS" having judged nothing -- a gate that
    # passes on no data is the same failure mode this WP exists to end.  It also
    # names the envelope columns this tool does not measure at all (the two pin
    # ones) and the peaking columns NEITHER envelope judges, so "Gate B passed"
    # cannot be quoted from half the envelope.
    passed, code = gate_b_envelope.report(envelope, measured, "GATE B scalars",
                                          pinned)
    return 0 if passed else code


if __name__ == "__main__":
    raise SystemExit(main())
