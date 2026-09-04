#!/usr/bin/env python3
"""Contract: include/Database/tf_ce.csv is a MEASURED table, and it LOADS.

WHAT IT IS.  The ABB-CE fuel-temperature grid (`isolth = 12`) that the APR1400 /
KNGR decks actually use, regressed by tools/fit_tf_table.py from the MASTER
KNGR cycle-1 run in E:/rasbery_runs/2026-09-04/kngr_tf_edit -- 9 640 samples,
one per (assembly, statepoint), from that run's $FB2D / $P2D / $B2D edits.  Its
sibling include/Database/tf.csv is MASTER's WH grid (`isolth = 11`), and the two
are NOT interchangeable: at 200 W/cm the CE rise falls about 30 % over the first
16 GWd/t where the WH rise falls about 6 %.

WHAT IS ASSERTED, AND WHY EACH ONE IS HERE.

  1. IT EXISTS, and it is not a copy of tf.csv.  A `ce` that silently resolved
     to the WH numbers would be the exact defect src/ThTfTable.h was written to
     prevent, and it would pass every other check here.
  2. THE SAME KNOTS as tf.csv.  XSSet::GetTfuel's low-LPD continuation is
     written against the 50 W/cm first knot (src/XSSet.h:516-526), and
     tools/fit_tf_table.py fits ON tf.csv's knots; a table on different axes
     would interpolate, and be wrong, silently.
  3. THE LOADER ACCEPTS IT -- checked against milk::Table::ParseFromCSV's rules
     transcribed from include/milk.h, not against a friendlier CSV reader:
     first line is the x axis with an IGNORED corner cell, every later row has
     exactly one cell per x knot, and both axes are STRICTLY ASCENDING because
     ThTfTable.cpp::checkAscending refuses otherwise.
  4. MONOTONE IN LPD at every burnup row.  dT must not fall as local power
     rises; that is conduction, not taste, and the fit has a penalty for it.
     It is asserted on the SHIPPED file because the penalty is iterative and
     could in principle be left unconverged.
  5. THE PROVENANCE LINE.  The table has to say which MASTER run produced it
     and carry the sha256 of that run's MAS_SUM, or a later reader cannot tell
     a measurement from a guess.  It lives in the header's CORNER CELL because
     that is the only place milk::Table::ParseFromCSV ignores -- a `#` comment
     line ahead of the header would be eaten AS the header.
  6. THE RISE SCALE IS DIVIDED OUT, asserted as "at zero burnup the CE and WH
     rises agree over 100-250 W/cm".  SolveTH writes
     `tful = tmod + fuel_temp_rise_scale * GetTfuel(bu, lpd)`, and the KNGR deck
     carries that scale at ~1.32 -- an empirical stand-in for this very table,
     fitted so tf.csv's WH rise reproduces MASTER's BOC fuel temperature.  The
     first tf_ce.csv was regressed from MASTER's dT WITHOUT dividing it out, so
     the correction was counted twice and block 59 arm (c) ran +79.3 K hot at
     BOC where arm (b) (same run, WH table) was +0.9 K.  The two tables must
     therefore MEET at BOC -- the CE table's job is the burnup SLOPE (assertion
     8) and the LPD shape, not a 32 % offset at zero burnup.
  7. THE SOLVER AND THE MIRROR AGREE.  tools/case_key.py must now KEY the CE
     table (it refused while the file was absent) and must digest the same
     bytes src/ThTfTable.cpp does -- the sha256 of the file, CRLF included,
     which is why the file must be LF.

USAGE
    tools/test_tf_ce_table_contract.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import case_key  # noqa: E402

WH = ROOT / "include" / "Database" / "tf.csv"
CE = ROOT / "include" / "Database" / "tf_ce.csv"

FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)


# ---------------------------------------------------------------------------
# milk::Table::ParseFromCSV, transcribed from include/milk.h.
# ---------------------------------------------------------------------------
def parse_like_milk(path: Path):
    """Parse exactly as the solver does, or raise the way the solver throws."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty table CSV")
    header = [c.strip() for c in lines[0].split(",")]
    if len(header) < 2:
        raise ValueError("invalid table CSV header")
    x_axis = [float(c) for c in header[1:]]  # header[0] is the ignored corner
    y_axis, values = [], []
    for line in lines[1:]:
        if not line:
            continue  # ParseFromCSV skips empty lines
        row = [c.strip() for c in line.split(",")]
        if len(row) != len(header):
            raise ValueError(f"invalid table CSV row size: {len(row)}")
        y_axis.append(float(row[0]))
        values.append([float(c) for c in row[1:]])
    return x_axis, y_axis, values


def loads_and_has_the_wh_knots() -> tuple[list, list, list] | None:
    if not CE.exists():
        fail("include/Database/tf_ce.csv does not exist.  Regress it with "
             "tools/fit_tf_table.py (see docs/TH_TF_TABLE_SELECTION_20260904_KO.md); "
             "nothing in this tree may ship a guessed CE grid.")
        return None

    try:
        ce_lpd, ce_bu, ce_dt = parse_like_milk(CE)
    except Exception as exc:  # noqa: BLE001 -- the point is that ANY throw fails
        fail(f"milk::Table::ParseFromCSV would refuse tf_ce.csv: {exc}")
        return None
    wh_lpd, wh_bu, wh_dt = parse_like_milk(WH)

    if ce_lpd != wh_lpd:
        fail(f"tf_ce.csv LPD knots {ce_lpd} != tf.csv's {wh_lpd}; the fit is "
             "supposed to be ON the shipped knots")
    if ce_bu != wh_bu:
        fail(f"tf_ce.csv burnup knots {ce_bu} != tf.csv's {wh_bu}")
    for axis, what in ((ce_lpd, "LPD"), (ce_bu, "burnup")):
        if any(b <= a for a, b in zip(axis, axis[1:])):
            fail(f"the {what} axis of tf_ce.csv is not strictly ascending; "
                 "ThTfTable.cpp::checkAscending refuses it and milk::Table::Get "
                 "would bisect nonsense")
    if any(v <= 0.0 for row in ce_dt for v in row):
        fail("tf_ce.csv holds a non-positive dT; a fuel node cannot be colder "
             "than its coolant")

    if ce_dt == wh_dt:
        fail("tf_ce.csv is byte-identical in VALUES to tf.csv.  `ce` resolving to "
             "the WH numbers is the defect src/ThTfTable.h exists to prevent.")
    return ce_lpd, ce_bu, ce_dt


# ---------------------------------------------------------------------------
# 4. Monotone in LPD.
# ---------------------------------------------------------------------------
def monotone_in_lpd(lpd, bu, dt) -> None:
    for j, row in enumerate(dt):
        for i in range(len(row) - 1):
            if row[i + 1] < row[i]:
                fail(f"tf_ce.csv falls with LPD at burnup {bu[j]}: "
                     f"dT({lpd[i]}) = {row[i]} > dT({lpd[i+1]}) = {row[i+1]}.  "
                     "A node cannot get colder as its linear power rises.")
                return


# ---------------------------------------------------------------------------
# 5. Provenance.
# ---------------------------------------------------------------------------
def provenance() -> None:
    header = CE.read_text(encoding="utf-8").splitlines()[0]
    corner = header.split(",")[0]
    if "regressed-from" not in corner:
        fail("tf_ce.csv's corner cell does not say what it was regressed from; a "
             "reader cannot tell a measurement from a guess")
    if not re.search(r"run=\S+", corner):
        fail("tf_ce.csv's provenance names no MASTER run directory (run=...)")
    if not re.search(r"mas_sum_sha256=[0-9a-f]{64}\b", corner):
        fail("tf_ce.csv's provenance carries no sha256 of the source MAS_SUM; "
             "without it the table cannot be tied back to a run")
    if "isolth=12" not in corner:
        fail("tf_ce.csv's provenance does not record isolth=12 -- the whole point "
             "is that this is the CE table and not the WH one")
    if "fit_tf_table" not in corner:
        fail("tf_ce.csv's provenance does not name the tool that produced it")
    if not re.search(r"rise_scale=[0-9.]+", corner):
        fail("tf_ce.csv's provenance does not record the deck `fuel temperature "
             "rise scale` divided out of the samples (rise_scale=...).  Without "
             "it a reader cannot tell this table from the v1 one that counted "
             "the deck's ~1.32 calibration twice.")

    # And the corner cell must stay a corner cell: one line, no comma.
    if CE.read_bytes().count(b"\r\n"):
        fail("tf_ce.csv has CRLF line endings; .gitattributes normalises this tree "
             "to LF and the table's identity is the sha256 of its BYTES, so a CRLF "
             "copy would key differently from every other checkout")


# ---------------------------------------------------------------------------
# 6. The case key mirror now KEYS the table instead of refusing it.
# ---------------------------------------------------------------------------
def _deck_json():
    return {
        "geometry": {
            "dimensions": {"ng": 2, "xydivision": 2, "npins": 16, "nfrod": 236},
            "size": {"hx": 20.0, "hy": 20.0, "hz": [10.0, 10.0]},
            "symmetry": {"angle": 90, "mirror": True,
                         "center assembly divided": False},
            "albedo": {"west": 0.0, "east": 0.5, "north": 0.0, "south": 0.5,
                       "bottom": 0.5, "top": 0.5},
            "core": [["A1", "A1"], ["A1", "A1"]],
        },
        "batch": {"A1": ["F1", "F1"]},
    }


def case_key_keys_it() -> None:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RASBERY_")}
    with tempfile.TemporaryDirectory() as raw:
        deck = Path(raw) / "d.json"
        deck.write_text(json.dumps(_deck_json()), encoding="utf-8")
        try:
            ce = case_key.case_key(deck, env=dict(clean, RASBERY_TH_TF_TABLE="ce"),
                                   xslib=False)
            wh = case_key.case_key(deck, env=dict(clean, RASBERY_TH_TF_TABLE="wh"),
                                   xslib=False)
        except SystemExit as exc:
            fail(f"tools/case_key.py still refuses the CE table ({exc}); the mirror "
                 "must key what the solver can now load")
            return

    digest = hashlib.sha256(CE.read_bytes()).hexdigest()
    identity = ce.get("th_tf_table", "")
    if not identity.startswith("ce:"):
        fail(f"case_key reported th_tf_table = {identity!r}, want a `ce:` identity")
    elif identity.split(":", 1)[1] != digest:
        fail("case_key's CE digest is not the sha256 of tf_ce.csv's bytes; the "
             "mirror and src/ThTfTable.cpp would disagree about the same file")
    if ce.get("th_tf_table") == wh.get("th_tf_table"):
        fail("the CE and WH tables key IDENTICALLY; the case key would not "
             "separate two runs with different fuel temperatures")


# ---------------------------------------------------------------------------
# The headline number, asserted rather than remembered.
# ---------------------------------------------------------------------------
def ce_burnup_slope_is_the_steep_one(lpd, bu, dt) -> None:
    """At 200 W/cm the CE rise must fall far faster over 16 GWd/t than WH's."""
    def at(table, x, y):
        i = lpd.index(x)
        j = next(k for k in range(len(bu) - 1) if bu[k] <= y <= bu[k + 1])
        f = (y - bu[j]) / (bu[j + 1] - bu[j])
        return table[j][i] * (1 - f) + table[j + 1][i] * f

    _wl, _wb, wh_dt = parse_like_milk(WH)
    ce_drop = 1.0 - at(dt, 200.0, 16.0) / at(dt, 200.0, 0.0)
    wh_drop = 1.0 - at(wh_dt, 200.0, 16.0) / at(wh_dt, 200.0, 0.0)
    print(f"  at 200 W/cm over 16 GWd/t: CE falls {100*ce_drop:.1f} %, "
          f"WH falls {100*wh_drop:.1f} %")
    if ce_drop < 4.0 * wh_drop:
        fail(f"the CE burnup slope ({100*ce_drop:.1f} %) is not the steep one this "
             f"table exists to capture (WH {100*wh_drop:.1f} %); a table that "
             "flat is the WH grid wearing a CE name")


# ---------------------------------------------------------------------------
# 6. The rise scale is divided out: CE and WH MEET at zero burnup.
# ---------------------------------------------------------------------------
BOC_AGREE_LPD = (100.0, 150.0, 200.0, 250.0)
BOC_AGREE_LIMIT = 0.06


def boc_agrees_with_wh(lpd, bu, dt) -> None:
    """A CE table that is 30 % hotter than WH at BOC has the deck's rise scale
    baked in twice -- see assertion 6 in the module docstring."""
    _wl, _wb, wh_dt = parse_like_milk(WH)
    if bu[0] != 0.0:
        fail("tf_ce.csv's first burnup knot is not 0; the BOC check cannot run")
        return
    worst = 0.0
    worst_x = None
    for x in BOC_AGREE_LPD:
        i = lpd.index(x)
        dev = dt[0][i] / wh_dt[0][i] - 1.0
        if abs(dev) > abs(worst):
            worst, worst_x = dev, x
    print(f"  at zero burnup the CE rise is within {100*abs(worst):.1f} % of WH "
          f"over {BOC_AGREE_LPD[0]:g}-{BOC_AGREE_LPD[-1]:g} W/cm "
          f"(worst at {worst_x:g} W/cm)")
    if abs(worst) > BOC_AGREE_LIMIT:
        fail(f"at zero burnup and {worst_x:g} W/cm the CE rise is "
             f"{100*worst:+.1f} % off WH's (limit {100*BOC_AGREE_LIMIT:.0f} %).  "
             "The two tables must MEET at BOC; a ~32 % offset is the deck's "
             "`fuel temperature rise scale` counted twice -- regress with "
             "tools/fit_tf_table.py --rise-scale/--rise-scale-first.")


def main() -> int:
    parsed = loads_and_has_the_wh_knots()
    if parsed is not None:
        lpd, bu, dt = parsed
        monotone_in_lpd(lpd, bu, dt)
        provenance()
        boc_agrees_with_wh(lpd, bu, dt)
        case_key_keys_it()
        ce_burnup_slope_is_the_steep_one(lpd, bu, dt)

    if FAILURES:
        for message in FAILURES:
            print("FAIL: " + message, file=sys.stderr)
        print(f"tf_ce table contract: FAIL ({len(FAILURES)})", file=sys.stderr)
        return 1
    print("tf_ce table contract: PASS (loads + knots + monotone + provenance "
          "+ BOC agreement + case key + CE slope)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
