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

ROD POSITION IN CENTIMETRES (2026-09-03, the i-SMR anchor B column)
-------------------------------------------------------------------
The i-SMR CY02/CY03/CY04 decks are ROD-CRITICAL: MASTER prints SEARCH=5 and
K-EFF == 1.000000 on every statepoint, and RASBERY drives its own rod search to
the same target.  So `delta_pcm` on those decks is a search residual, `delta_ppm`
is one stated constant minus the same stated constant (both sides run 0.00 ppm
for the whole cycle), and the two columns this tool used to judge carry NO
model-vs-MASTER information at all.  The quantity that does is WHERE THE BANK
ENDED UP, and until now nothing compared it.

So this tool now also parses SUMMARY EDIT 1's `(CONTROL ROD POSITIONS)` block --
already centimetres from the bottom of the active core, bank order taken from
the block's own header line, first numeric column at index 10 (the convention
tools/plot_ismr_validation.py has used all along, and the one asserted below
rather than assumed) -- joins it to `rods/insertions` in the .h5 through
`pos_cm = ROD_TOP_CM - insertion`, and emits one `delta_rod_cm_<BANK>` column per
bank both sides carry.

TWO THINGS ABOUT THE ROD VERDICT, BOTH OF THEM THE SAME ARGUMENT THE SCALAR
ENVELOPE ALREADY MAKES ABOUT `delta_ppm`:

  * A BANK THAT NEVER MOVES IS NOT EVIDENCE.  On these decks S1..S6 sit at the
    top of the core for all 24 statepoints on BOTH sides.  Its delta is
    identically zero because the deck parked it there, not because the physics
    agreed, so it is REPORTED and labelled `pinned` and it is excluded from the
    scored population.  If no bank moved, the rod gate says NOT SCORED rather
    than PASS.
  * THE ROD GATE IS SEPARATE FROM `--envelope`, deliberately.  The two
    gate_b_envelope presets are five numbers apiece on the SAME max|delta|
    columns two Gate B tools have always reported; a sixth metric that only one
    deck family produces does not belong in a table whose whole purpose is that
    both presets say the same thing about the same columns.  It gets its own
    two thresholds (`--rod-max-cm`, `--rod-rms-cm`), its own printed verdict, and
    its own contribution to the exit code.

Usage:
    python compare_master_rasbery.py MAS_SUM result.h5 [-o out_prefix]
Writes <prefix>.csv and <prefix>.md (default prefix: master_vs_rasbery).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_b_envelope  # noqa: E402

#: Distance from the bottom of the active core to a fully withdrawn bank, cm.
#: The i-SMR decks' `rod configuration` profiles top out at 240.0 and every
#: MASTER EDIT 1 row prints 240.000 for a parked bank, which is the same number
#: seen from the two ends of the same convention.  Overridable, because it is a
#: property of the deck and not of this tool.
ROD_TOP_CM = 240.0

#: SUMMARY EDIT 1's first rod column.  The nine header NAMES
#: (NO. DAY EFPD XE/SM SEARCH RHO RST PPI CMP) cover TEN numeric fields, because
#: XE/SM prints as two.  tools/plot_ismr_validation.py hard-codes this same 10 as
#: REF_ROD_START_COL; here it is CHECKED against the row width rather than
#: trusted, so a MASTER edit that adds a column fails loudly instead of shifting
#: every bank by one and reporting a 120 cm disagreement as physics.
REF_ROD_START_COL = 10


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


#: A bank name in the EDIT 1 header: one letter, then digits.  R1..R4 are the
#: regulating banks, S1..S6 the shutdown banks.
ROD_NAME_RE = re.compile(r"^[A-Z]\d+$")


def parse_master_rods(path: Path) -> tuple[list[str], dict[float, list[float]]]:
    """(bank names in print order, {efpd: [cm per bank]}) from SUMMARY EDIT 1.

    The bank ORDER is read from the block's own header line and never assumed.
    The i-SMR outputs print `S1 R3 S3 S2 R4 R1 S4 R2 S5 S6` -- not sorted, not
    grouped by type -- so a hard-coded order silently pairs R3's trajectory with
    R4's column and reports a real agreement as a 120 cm error.

    LAST ROW AT AN EFPD WINS, which is the rule parse_master_summary() already
    applies to EDIT 2/3/4.  It matters here: rows 1-4 all print EFPD 0.000, and
    row 1 is the SEARCH=1 all-rods-out state MASTER solves before it starts
    searching.  Keeping row 1 would compare an unrodded reference against a
    converged rod-critical statepoint.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    last_run = text.rfind("SUMMARY EDIT 1 :")
    if last_run >= 0:
        text = text[last_run:]

    names: list[str] = []
    rows: dict[float, list[float]] = {}
    in_rod = False
    for line in text.splitlines():
        if "(CONTROL ROD POSITIONS)" in line:
            in_rod = True
            continue
        if "SUMMARY EDIT 2" in line or (in_rod and "SUMMARY EDIT" in line):
            in_rod = False
            continue
        if not in_rod:
            continue
        tokens = line.split()
        if not tokens:
            continue
        if not names:
            if all(ROD_NAME_RE.match(t) for t in tokens):
                names = tokens
            continue
        if not re.match(r"^\s*\d+\s+[\d.]", line):
            continue
        width = len(tokens) - len(names)
        if width != REF_ROD_START_COL:
            raise SystemExit(
                f"{path}: SUMMARY EDIT 1 row has {len(tokens)} fields for "
                f"{len(names)} banks, so the first rod column is {width} and not "
                f"the documented {REF_ROD_START_COL}. The MASTER edit format "
                f"changed; fix REF_ROD_START_COL rather than reading shifted "
                f"columns as physics.\n  {line.strip()}")
        try:
            efpd = float(tokens[2])
            positions = [float(t) for t in tokens[REF_ROD_START_COL:]]
        except ValueError:
            continue
        rows[efpd] = positions
    return names, rows


def parse_rasbery_rods(path: Path, rod_top_cm: float
                       ) -> tuple[list[str], dict[float, list[float]]]:
    """(bank names, {efpd: [cm per bank]}) from `rods/*` in the RASBERY .h5.

    `rods/insertions` is INSERTION depth; MASTER prints WITHDRAWN position.  The
    conversion is the one tools/plot_ismr_validation.py established,
    `pos_cm = 240.0 - insertion`, and it is the ONLY place the two conventions
    are reconciled -- `summary/rod_step` is neither (observed 0.78-2.13 on decks
    whose banks travel 240 cm) and must not be substituted for it.
    """
    with h5py.File(path, "r") as f:
        if "rods" not in f or "insertions" not in f["rods"]:
            return [], {}
        groups = [g.decode() if isinstance(g, bytes) else str(g)
                  for g in f["rods"]["groups"][()]]
        insertions = f["rods"]["insertions"][()]
        efpd = f["summary"]["efpd"][()]
    rows: dict[float, list[float]] = {}
    for i, e in enumerate(efpd):
        if i >= len(insertions):
            break
        rows[float(e)] = [rod_top_cm - float(v) for v in insertions[i]]
    return groups, rows


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


def rod_report(joined: list[dict], banks: list[str], max_cm: float,
               rms_cm: float) -> tuple[bool, int] | None:
    """Print the rod-position verdict.  None when the run carries no rod data.

    Returns (passed, exit_code) with the same 0/1/2 meaning gate_b_envelope.report
    uses, INCLUDING 2 for "measured, but nothing that could have failed".
    """
    if not banks:
        return None
    populations: dict[str, list[float]] = {}
    for bank in banks:
        column = f"delta_rod_cm_{bank}"
        values = [r[column] for r in joined if column in r]
        if values:
            populations[bank] = values
    if not populations:
        return None

    # A BANK BOTH SIDES PARKED IS NOT A MEASUREMENT.  Same test as the pinned
    # `delta_ppm` column below: exact constancy, so a bank that really did move
    # by a hair still scores and this can only ever under-report.
    pinned = []
    scored = []
    for bank in populations:
        m = [r[f"master_rod_cm_{bank}"] for r in joined
             if f"master_rod_cm_{bank}" in r]
        r_ = [r[f"ras_rod_cm_{bank}"] for r in joined if f"ras_rod_cm_{bank}" in r]
        if len(set(m)) <= 1 and len(set(r_)) <= 1:
            pinned.append(bank)
        else:
            scored.append(bank)

    print(f"rod gate: |delta| <= {max_cm:.3f} cm per bank, RMS <= {rms_cm:.3f} cm "
          f"(pos_cm = {ROD_TOP_CM:g} - insertion)")
    passed = True
    for bank in banks:
        if bank not in populations:
            continue
        values = populations[bank]
        worst = max(abs(v) for v in values)
        rms = math.sqrt(sum(v * v for v in values) / len(values))
        ok = worst <= max_cm and rms <= rms_cm
        if bank in scored:
            passed = passed and ok
        tail = ("   (STRUCTURALLY PINNED -- parked on both sides, not evidence)"
                if bank in pinned else "")
        print(f"  {bank:<4} max|d| = {worst:8.3f} cm   rms = {rms:7.3f} cm   "
              f"{'PASS' if ok else 'FAIL'}{tail}")
    if not scored:
        print(f"GATE B rod: NOT SCORED -- every bank ({', '.join(pinned)}) is "
              f"parked on both sides for all {len(joined)} joined statepoint(s), "
              f"so no bank's delta could have been anything but zero")
        return False, 2
    print(f"GATE B rod: {'PASS' if passed else 'FAIL'} "
          f"(scored banks: {', '.join(scored)}"
          + (f"; reported but pinned: {', '.join(pinned)}" if pinned else "")
          + ")")
    return passed, 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mas_sum", type=Path)
    ap.add_argument("h5", type=Path)
    ap.add_argument("-o", "--out-prefix", default="master_vs_rasbery")
    ap.add_argument("--efpd-tol", type=float, default=1e-3,
                    help="EFPD match tolerance for joining states")
    # RESTART DECKS COUNT EFPD FROM A DIFFERENT ZERO ON THE TWO SIDES, and until
    # now that made this tool exit on "no EFPD-matched states" for exactly the
    # three decks anchor B is about.  RASBERY's i-SMR_CY02 run carries the
    # cycle-1 burnup forward (its first statepoint is at 781.215 d), while
    # MASTER's depf_02.sum restarts its EFPD column at 0.000.  The default is
    # 0.0, so nothing that joined before joins differently now.
    ap.add_argument("--efpd-offset", default="0",
                    help="subtract this many EFPD from the RASBERY side before "
                         "joining (default: 0). `auto` takes min(ras) - "
                         "min(master), i.e. it aligns the two first "
                         "statepoints, which is the cycle offset a restart deck "
                         "carries.")
    ap.add_argument("--rod-top-cm", type=float, default=ROD_TOP_CM,
                    help="withdrawn-bank position in cm, i.e. the constant in "
                         "pos_cm = C - insertion (default: %(default)s)")
    ap.add_argument("--rod-max-cm", type=float, default=5.0,
                    help="rod gate: per-bank max|delta| bar in cm "
                         "(default: %(default)s)")
    ap.add_argument("--rod-rms-cm", type=float, default=2.0,
                    help="rod gate: per-bank RMS bar in cm (default: %(default)s)")
    ap.add_argument("--no-rod-gate", action="store_true",
                    help="compute and write the delta_rod_cm_* columns but do "
                         "not let them move the exit code")
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

    # ROD HALF.  Both sides may legitimately have none (a boron-search deck), in
    # which case `rod_banks` is empty and nothing downstream changes.
    mas_bank_names, mas_rods = parse_master_rods(args.mas_sum)
    ras_bank_names, ras_rods = parse_rasbery_rods(args.h5, args.rod_top_cm)
    if not master:
        raise SystemExit(f"{args.mas_sum}: no SUMMARY EDIT 2/3/4 statepoint rows")
    if not ras:
        raise SystemExit(f"{args.h5}: no summary statepoints")
    if str(args.efpd_offset).strip().lower() == "auto":
        offset = min(ras) - min(master)
        print(f"efpd offset: auto -> {offset:.6f} d (RASBERY {min(ras):.6f} "
              f"aligned onto MASTER {min(master):.6f})")
    else:
        try:
            offset = float(args.efpd_offset)
        except ValueError:
            raise SystemExit(
                f"--efpd-offset {args.efpd_offset!r} is neither a number nor "
                f"`auto`") from None
    if offset:
        ras = {e - offset: row for e, row in ras.items()}
        ras_rods = {e - offset: pos for e, pos in ras_rods.items()}

    ras_bank_at = {name: i for i, name in enumerate(ras_bank_names)}
    # PRINT ORDER IS MASTER'S, membership is the intersection: a bank only one
    # side models has no delta, and inventing one would be a comparison against
    # a bank that is not there.
    rod_banks = [n for n in mas_bank_names if n in ras_bank_at]
    if mas_bank_names and ras_bank_names and not rod_banks:
        print(f"warning: no bank name is common to MASTER "
              f"({', '.join(mas_bank_names)}) and the .h5 "
              f"({', '.join(ras_bank_names)}); no rod column written")

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
        # The rod half joins on the SAME EFPD and the SAME tolerance, so a
        # statepoint can never appear in the scalar table and the rod table with
        # two different meanings of "this state".
        if rod_banks:
            m_pos = next((v for e_m, v in mas_rods.items()
                          if abs(e_m - e_r) <= args.efpd_tol), None)
            r_pos = ras_rods.get(e_r)
            if m_pos is not None and r_pos is not None:
                for bank in rod_banks:
                    mi = mas_bank_names.index(bank)
                    ri = ras_bank_at[bank]
                    if mi >= len(m_pos) or ri >= len(r_pos):
                        continue
                    row[f"master_rod_cm_{bank}"] = m_pos[mi]
                    row[f"ras_rod_cm_{bank}"] = r_pos[ri]
                    row[f"delta_rod_cm_{bank}"] = r_pos[ri] - m_pos[mi]
        joined.append(row)

    if not joined:
        raise SystemExit("no EFPD-matched states between the two inputs")

    fieldnames = ["efpd"]
    for mc, rc, dc, _ in DELTA_PAIRS:
        for c in (mc, rc, dc):
            if any(c in r for r in joined) and c not in fieldnames:
                fieldnames.append(c)
    for bank in rod_banks:
        for c in (f"master_rod_cm_{bank}", f"ras_rod_cm_{bank}",
                  f"delta_rod_cm_{bank}"):
            if any(c in r for r in joined) and c not in fieldnames:
                fieldnames.append(c)

    csv_path = Path(f"{args.out_prefix}.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(joined)

    md_path = Path(f"{args.out_prefix}.md")
    key_cols = ["efpd", "delta_ppm", "delta_pcm", "delta_ao", "delta_fqp", "delta_frp"]
    key_cols += [f"delta_rod_cm_{b}" for b in rod_banks]
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

    rod = None if args.no_rod_gate else rod_report(
        joined, rod_banks, args.rod_max_cm, args.rod_rms_cm)
    if rod is None:
        return 0 if passed else code
    rod_passed, rod_code = rod

    # WHY A `NOT SCORED` SCALAR VERDICT IS NOT THE LAST WORD ON A ROD DECK.
    # gate_b_envelope returns 2 for "this run judged nothing", and on a
    # rod-critical deck that is the RIGHT answer about the scalars: k_eff is
    # driven to target and the boron is a stated constant.  It is the WRONG
    # answer about the run, because the rod gate judged something -- which is
    # the entire reason this column exists.  So a scored rod gate replaces a
    # scalar 2; it never rescues a scalar FAIL (1), and it can still fail on its
    # own.
    if code == 2 and rod_code != 2:
        print("GATE B: the scalar envelope scored nothing on this deck (above); "
              "the verdict is the ROD gate's.")
        return rod_code
    if rod_code == 2 and code != 2:
        return 0 if passed else code
    return 0 if (passed and rod_passed) else max(code, rod_code)


if __name__ == "__main__":
    raise SystemExit(main())
