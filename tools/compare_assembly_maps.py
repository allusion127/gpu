#!/usr/bin/env python3
"""Physical-assembly power/burnup map comparison: MASTER SUMMARY EDIT 5 vs
RASBERY steps/*/assembly/{power,burn}, folded onto PHYSICAL assemblies.

WHY THIS EXISTS.  docs/ISMR_MASTER_COMPARISON_RUNBOOK_20260903_KO.md Sec 7
item 1 ("중심선 쌍 평균이 비교 도구에 없다") names a gap: no tool folds RASBERY's
quarter-map slots onto the physical assemblies MASTER prints one number for,
so a correct rotational-fold run can score WORSE than a wrong mirror-fold run
on a raw slot-by-slot diff.  `tools/compare_master_rasbery.py` sidesteps this
by staying scalar (core-wide sums); `tools/plot_ismr_validation.py` plots raw
quarter-map slots and is explicitly documented as not doing this fold.  This
tool does the fold and turns it into a scored table.

THE FOLD, from docs/ROTATIONAL_SHUFFLE_FIX_20260904_KO.md and the `gather`
lambda at src/IO.cpp:2557-2621 (comment at :2598-2602).  Under the 90-degree
rotational quarter-core fold (`symang=90, mirror:false, symdiv:true`, the
i-SMR convention -- see tools/test_ismr_tools_contract.py's `fold_contract`),
quarter-map slot (0,c) and slot (c,0), c>0, are the two HALVES of ONE physical
assembly: (0,c)'s visible fine nodes occupy the source frame's high ROWS,
(c,0)'s occupy its high COLUMNS (VisibleAssemblyRows/Cols in src/Geometry.cpp,
mirrored in tools/test_rotational_shuffle_contract.py's `visible_rows`/
`visible_cols`).  MASTER prints ONE value per physical assembly -- literally
the same 4-decimal number at both slots, verified against
test/7_i-SMR_Validation/Reference_output/depf_01.sum NO.=1: row 5 col F and
row 6 col E both print power 1.1301 -- while RASBERY's per-slot assembly
average is only over that slot's visible half.  So the PHYSICAL value this
tool reports is the node-weighted mean of the two RASBERY half-slot values,
weight = visible node count = VisibleAssemblyRows(row,n) *
VisibleAssemblyCols(col,n) with n=ndivxy; for ndivxy of either parity that
count is IDENTICAL for both arms (floor(n/2)*n == n*floor(n/2)), so the
weighted mean is the ordinary arithmetic mean of the two halves -- the same
average docs/ROTATIONAL_SHUFFLE_FIX_20260904_KO.md's residual note (14.12 vs
16.385) already reasons about.  The centre slot (0,0) is a QUARTER of its own
physical assembly and gets no partner: src/IO.cpp:2603-2606 reconstructs the
centre's full node buffer by rotating its OWN quarter three more times, so by
symmetry the quarter's average already equals the physical average.  An
interior slot (row>0 and col>0) is not on either symmetry line and stands for
itself.

For a `mirror:true` deck (APR1400 KNGR) there is no split: every quarter-map
slot IS one whole physical assembly already, and this tool's `--mirror` flag
turns the fold off (each slot compared as itself, no pairing, no averaging).

USAGE
    tools/compare_assembly_maps.py MAS_SUM result.h5 [-o out_prefix]
    tools/compare_assembly_maps.py MAS_SUM result.h5 --mirror   # KNGR/APR1400
Writes <prefix>.csv (one row per physical assembly per statepoint) and
<prefix>.md (per-statepoint tables + RMS/max, default prefix:
assembly_map_vs_rasbery).
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
# Reused rather than re-implemented: the LAST-RUN-WINS slice of a MAS_SUM that
# may accumulate several runs' SUMMARY EDIT blocks, and the `auto` EFPD-offset
# convention for restart decks whose two sides count EFPD from different zeros.
import compare_master_rasbery as cmr  # noqa: E402

#: `NO.=   1   DAY =     0.00  EFPD =     0.00` -- one assembly-map block header.
ASSEMBLY_NO_RE = re.compile(
    r"NO\.\=\s*(\d+)\s+DAY\s*=\s*([0-9.Ee+-]+)\s+EFPD\s*=\s*([0-9.Ee+-]+)")


def numeric_row(line: str) -> list[float] | None:
    tokens = line.split()
    if not tokens:
        return None
    try:
        return [float(tok) for tok in tokens]
    except ValueError:
        return None


def parse_master_assembly(path: Path) -> dict[float, list[dict]]:
    """{efpd: [{"no", "cells"}, ...]}, LAST run.  See parse_master_assembly_from_text."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return parse_master_assembly_from_text(text)


def parse_master_assembly_from_text(text: str) -> dict[float, list[dict]]:
    """{efpd: [{"no": int, "cells": {(array_row,array_col): {"label","power",
    "burn"}}}, ...]}, LAST RUN.  Split from parse_master_assembly() so a
    fixture string (tools/test_assembly_map_fold_contract.py) can be parsed
    without a temp file.

    Row/column order is READING order, not the printed row NUMBER or column
    LETTER: the first block row parsed becomes array_row 0, the first token
    after `Y\\X` becomes array_col 0, matching the convention
    tools/plot_ismr_validation.py's `rasbery_matrix` and RASBERY's own
    `geometry/ijtola` both use (array row/col 0 = the symmetry-adjacent
    row/column).  `label` keeps MASTER's own printed text (column letter +
    row number, e.g. "F5") so the table stays traceable to the .sum by eye.

    Each MASTER block prints FOUR lines per row: row-number + batch ID,
    power, burnup, k-infinite (in that order, per the block's own "FIRST
    LINE.. FOURTH LINE" legend).  Only the first three are read here.

    ONE EFPD CAN CARRY SEVERAL BLOCKS, and "last wins" -- the rule
    parse_master_summary()/parse_master_rods() apply -- is WRONG here in
    general.  Ground truth (238, 2026-09-04): CY01's depf_01.sum prints FOUR
    blocks at EFPD 0.000 -- NO.=1 is the SEARCH=1 ARO k-eff solve (anchor A,
    k-eff 1.026494) and NO.=2-4 are a SEPARATE SEARCH=5 rod-critical
    diagnostic (k-eff pinned to 1.000000, rods driven in) that RASBERY's own
    CY01 deck never performs (its rods sit at 240 cm all cycle).  Taking the
    last block there compares MASTER's rod-critical diagnostic against
    RASBERY's ARO physics and reads as a ~43% power RMS / 131% max artefact
    that is not physical at all.  On CY02-04 the SAME raw pattern (a SEARCH=1
    block, then SEARCH=5 blocks at the same EFPD) needs the OPPOSITE pick:
    those decks restart rod-critical, so the SEARCH=1 block is the throwaway
    and a later SEARCH=5 block is the real state.  No rule over MASTER's own
    print order can get both right, because the two deck families print the
    identical shape for opposite reasons.  So candidates are returned here
    UNRESOLVED, keyed by their own "no", and build_rows() resolves each one
    against RASBERY's actual k-eff at the joined statepoint (see
    select_master_candidate()) -- the one fact that is never ambiguous about
    which physical state RASBERY actually computed.
    """
    last_run = text.rfind("SUMMARY EDIT 1 :")
    if last_run >= 0:
        text = text[last_run:]
    lines = text.splitlines()

    states: dict[float, list[dict]] = {}
    in_assembly = False
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "SUMMARY EDIT 5" in line:
            in_assembly = True
            idx += 1
            continue
        if "SUMMARY EDIT" in line and "SUMMARY EDIT 5" not in line:
            in_assembly = False
        if not in_assembly:
            idx += 1
            continue

        m = ASSEMBLY_NO_RE.search(line)
        if not m:
            idx += 1
            continue

        no = int(m.group(1))
        efpd = float(m.group(3))
        cells: dict[tuple[int, int], dict] = {}
        letters: list[str] = []
        row_idx = -1
        idx += 1
        while idx < len(lines):
            row_line = lines[idx]
            if "NO.=" in row_line or "SUMMARY EDIT" in row_line:
                idx -= 1
                break
            tokens = row_line.split()
            if len(tokens) >= 2 and tokens[0] == "Y\\X":
                letters = tokens[1:]
                idx += 1
                continue
            if tokens and tokens[0].isdigit() and letters:
                row_no = tokens[0]
                row_idx += 1
                count = len(tokens) - 1
                power = numeric_row(lines[idx + 1]) if idx + 1 < len(lines) else None
                burn = numeric_row(lines[idx + 2]) if idx + 2 < len(lines) else None
                if power and burn and len(power) == count and len(burn) == count:
                    for c in range(min(count, len(letters))):
                        cells[(row_idx, c)] = {
                            "label": f"{letters[c]}{row_no}",
                            "power": power[c],
                            "burn": burn[c],
                        }
                    idx += 3
                    continue
            idx += 1
        states.setdefault(efpd, []).append({"no": no, "cells": cells})
        idx += 1
    return states


#: `NO.  DAY  EFPD  CYC-BU  TOT-BU  P(%)  PPM  K-EFF  ERRFLX  REACT.` -- SUMMARY
#: EDIT 2's own numeric row, one per "no" (the SAME "no" ASSEMBLY_NO_RE reads
#: off EDIT 5's block headers -- both blocks are printed by the same pass over
#: the same statepoint list). parts[7] = K-EFF, the column that tells the two
#: same-EFPD block families apart when nothing else does.
EDIT2_ROW_RE = re.compile(r"^\s*(\d+)\s+[\d.]")


def parse_master_keff_by_no(text: str) -> dict[int, float]:
    """{no: k-eff} from SUMMARY EDIT 2, LAST run, every row (no EFPD
    collapsing -- this is exactly the per-"no" data select_master_candidate()
    needs to tell apart same-EFPD blocks parse_master_assembly_from_text()
    deliberately leaves unresolved)."""
    last_run = text.rfind("SUMMARY EDIT 1 :")
    if last_run >= 0:
        text = text[last_run:]
    out: dict[int, float] = {}
    section = False
    for line in text.splitlines():
        if "SUMMARY EDIT 2 : REACTIVITY" in line:
            section = True
            continue
        if "SUMMARY EDIT" in line and "SUMMARY EDIT 2" not in line:
            section = False
            continue
        if not section or not EDIT2_ROW_RE.match(line):
            continue
        parts = line.split()
        if len(parts) >= 8:
            try:
                out[int(parts[0])] = float(parts[7])
            except ValueError:
                continue
    return out


def select_master_candidate(candidates: list[dict], keff_by_no: dict[int, float],
                            target_keff: float | None) -> dict:
    """Resolve parse_master_assembly_from_text()'s unresolved same-EFPD
    candidate list to ONE block's cells, by k-eff proximity to
    `target_keff` (RASBERY's own k-eff at the joined statepoint).

    Falls back to the LAST candidate -- the previous, documented-wrong-on-
    CY01 but harmless-elsewhere default -- only when there is nothing to
    disambiguate with (a single candidate, no EDIT 2 k-eff for any of them,
    or no RASBERY k-eff to compare against), so a .h5 without `summary/keff`
    still gets an answer instead of a crash.
    """
    if len(candidates) == 1 or target_keff is None:
        return candidates[-1]["cells"]
    scored = [(abs(keff_by_no[c["no"]] - target_keff), c)
              for c in candidates if c["no"] in keff_by_no]
    if not scored:
        return candidates[-1]["cells"]
    scored.sort(key=lambda t: t[0])
    return scored[0][1]["cells"]


def parse_rasbery_assembly(
    path: Path,
) -> tuple[dict[float, dict[tuple[int, int], dict]], dict[float, float], int, int, int]:
    """({efpd: {(row,col): {"power","burn"}}}, {efpd: keff}, ndivxy, nya, nxa).

    `keff` is read from `summary/keff` when present; it feeds
    select_master_candidate()'s disambiguation and is omitted from the ras
    map's own EFPDs otherwise (an older .h5 without it just loses that
    disambiguation and falls back to LAST-candidate, not to a crash).
    """
    with h5py.File(path, "r") as f:
        efpd = [float(v) for v in f["summary"]["efpd"][()]]
        keff = ([float(v) for v in f["summary"]["keff"][()]]
                if "keff" in f["summary"] else [])
        geo = f["geometry"]
        nxa = int(geo["nxa"][()])
        nya = int(geo["nya"][()])
        ndivxy = int(geo["ndivxy"][()]) if "ndivxy" in geo else 2
        ijtola_flat = [int(v) for v in geo["ijtola"][()]]
        step_keys = sorted(f["steps"].keys())
        power_by_step = [f[f"steps/{k}/assembly/power"][()] for k in step_keys]
        burn_by_step = [f[f"steps/{k}/assembly/burn"][()] for k in step_keys]

    ijtola = [[ijtola_flat[r * nxa + c] for c in range(nxa)] for r in range(nya)]

    states: dict[float, dict[tuple[int, int], dict]] = {}
    ras_keff: dict[float, float] = {}
    n = min(len(efpd), len(power_by_step))
    for i in range(n):
        cells: dict[tuple[int, int], dict] = {}
        for r in range(nya):
            for c in range(nxa):
                la = ijtola[r][c]
                if la < 0 or la >= len(power_by_step[i]):
                    continue
                cells[(r, c)] = {
                    "power": float(power_by_step[i][la]),
                    "burn": float(burn_by_step[i][la]),
                }
        states[efpd[i]] = cells
        if i < len(keff):
            ras_keff[efpd[i]] = keff[i]
    return states, ras_keff, ndivxy, nya, nxa


def visible_node_count(row: int, col: int, n: int) -> int:
    """Visible fine-node count of quarter-map slot (row,col), ndivxy=n.

    Mirrors VisibleAssemblyRows/Cols (src/Geometry.cpp) and
    tools/test_rotational_shuffle_contract.py's `visible_rows`/`visible_cols`:
    row 0 (col 0) sees only the HIGH half of its ndivxy span, everything else
    sees the full span.
    """
    rows = n // 2 if row == 0 else n
    cols = n // 2 if col == 0 else n
    return rows * cols


def weighted_mean(a: float | None, wa: int, b: float | None, wb: int) -> float | None:
    """Node-weighted mean of two half-slot values; falls back to whichever
    side is present so a partial join still reports something rather than
    dropping the physical assembly outright."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    if wa + wb == 0:
        return (a + b) / 2.0
    return (a * wa + b * wb) / (wa + wb)


def fold_master(mcells: dict[tuple[int, int], dict], mirror: bool
               ) -> list[tuple[str, tuple[int, int], tuple[int, int] | None,
                               float, float]]:
    """[(label, slot_a, slot_b_or_None, master_power, master_burn), ...], one
    entry per PHYSICAL assembly MASTER's map carries at this statepoint.

    `slot_b` is None for --mirror, the centre, and interior slots -- there is
    no partner to fold.  MASTER already prints the SAME value at both arms of
    a symmetry-line pair (that equality IS the fold's correctness signal, per
    docs/ROTATIONAL_SHUFFLE_FIX_20260904_KO.md); when both are present here
    the reported master value is their mean, which is the identity when they
    agree and a graceful reading when a MAS_SUM prints one digit of rounding
    disagreement.
    """
    out = []
    handled: set[tuple[int, int]] = set()
    for (r, c), cell in sorted(mcells.items()):
        if (r, c) in handled:
            continue
        if mirror or (r == 0 and c == 0) or (r > 0 and c > 0):
            handled.add((r, c))
            out.append((cell["label"], (r, c), None, cell["power"], cell["burn"]))
            continue
        # (0,c) or (c,0), c>0: the two halves of one physical assembly.
        k = c if r == 0 else r
        slot_a, slot_b = (0, k), (k, 0)
        handled.add(slot_a)
        handled.add(slot_b)
        ca, cb = mcells.get(slot_a), mcells.get(slot_b)
        if ca is None or cb is None:
            # Both arms of a real physical assembly are always printed
            # together or not at all (the core boundary is symmetric under
            # the fold); a lone arm means this slot is a MASTER printing
            # oddity, not a physical assembly -- report it as itself rather
            # than silently drop it.
            present = ca or cb
            out.append((present["label"], slot_a, slot_b, present["power"],
                        present["burn"]))
            continue
        label = ca["label"] if ca["label"] == cb["label"] else f"{ca['label']}={cb['label']}"
        out.append((label, slot_a, slot_b, (ca["power"] + cb["power"]) / 2.0,
                    (ca["burn"] + cb["burn"]) / 2.0))
    return out


def ras_physical_value(rcells: dict[tuple[int, int], dict], slot_a, slot_b,
                       field: str, ndivxy: int
                      ) -> tuple[float | None, float | None, float | None]:
    """(physical_value, half_a, half_b) for one field ("power"/"burn").

    physical_value is the node-weighted mean when both halves are present
    (see module docstring: the weight is IDENTICAL on both arms for any
    ndivxy, so this is the ordinary mean); half_a/half_b are the raw slot
    values, reported for the informational symmetry check even though they
    were never meant to agree with each other (only their fold does).
    """
    ca = rcells.get(slot_a)
    va = ca[field] if ca else None
    if slot_b is None:
        return va, None, None
    cb = rcells.get(slot_b)
    vb = cb[field] if cb else None
    wa = visible_node_count(*slot_a, ndivxy)
    wb = visible_node_count(*slot_b, ndivxy)
    return weighted_mean(va, wa, vb, wb), va, vb


def rms_max(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return (math.sqrt(sum(v * v for v in values) / len(values)),
            max(abs(v) for v in values))


def build_rows(master_states: dict[float, list[dict]], ras_states: dict[float, dict],
               ras_keff: dict[float, float], keff_by_no: dict[int, float],
               ndivxy: int, mirror: bool, efpd_tol: float
              ) -> list[dict]:
    rows: list[dict] = []
    for e_m in sorted(master_states):
        e_r = None
        for candidate in ras_states:
            if abs(candidate - e_m) <= efpd_tol:
                e_r = candidate
                break
        if e_r is None:
            continue
        mcells = select_master_candidate(master_states[e_m], keff_by_no,
                                         ras_keff.get(e_r))
        rcells = ras_states[e_r]
        for label, slot_a, slot_b, mpow, mburn in fold_master(mcells, mirror):
            rpow, pa, pb = ras_physical_value(rcells, slot_a, slot_b, "power", ndivxy)
            rburn, ba, bb = ras_physical_value(rcells, slot_a, slot_b, "burn", ndivxy)
            dpow_pct = (rpow - mpow) / mpow * 100.0 if rpow is not None and mpow else None
            dburn = (rburn - mburn) if rburn is not None else None
            rows.append({
                "efpd": e_m,
                "label": label,
                "master_power": mpow,
                "ras_power": rpow,
                "delta_power_pct": dpow_pct,
                "sym_power_a": pa,
                "sym_power_b": pb,
                "sym_power_diff": (pa - pb) if pa is not None and pb is not None else None,
                "master_burn": mburn,
                "ras_burn": rburn,
                "delta_burn_gwdt": dburn,
                "sym_burn_a": ba,
                "sym_burn_b": bb,
                "sym_burn_diff": (ba - bb) if ba is not None and bb is not None else None,
            })
    return rows


FIELDNAMES = ["efpd", "label", "master_power", "ras_power", "delta_power_pct",
              "sym_power_a", "sym_power_b", "sym_power_diff",
              "master_burn", "ras_burn", "delta_burn_gwdt",
              "sym_burn_a", "sym_burn_b", "sym_burn_diff"]


def fmt(v) -> str:
    return "" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mas_sum", type=Path)
    ap.add_argument("h5", type=Path)
    ap.add_argument("-o", "--out-prefix", default="assembly_map_vs_rasbery")
    ap.add_argument("--efpd-tol", type=float, default=1e-3,
                    help="EFPD match tolerance for joining statepoints")
    ap.add_argument("--efpd-offset", default="0",
                    help="subtract this many EFPD from the RASBERY side before "
                         "joining (default: 0); `auto` aligns the two first "
                         "statepoints, the cycle offset a restart deck carries "
                         "(same convention as tools/compare_master_rasbery.py)")
    ap.add_argument("--mirror", action="store_true",
                    help="the deck folds the core by MIRROR symmetry (APR1400 "
                         "KNGR), not the 90-degree ROTATIONAL fold: every "
                         "quarter-map slot is already one whole physical "
                         "assembly, so no row/col-0 pairing is done")
    ap.add_argument("--ndivxy", type=int, default=None,
                    help="override the fold's node-weighting divisor "
                         "(default: read geometry/ndivxy from the .h5)")
    args = ap.parse_args()

    mas_sum_text = args.mas_sum.read_text(encoding="utf-8", errors="ignore")
    master_states = parse_master_assembly_from_text(mas_sum_text)
    if not master_states:
        raise SystemExit(f"{args.mas_sum}: no SUMMARY EDIT 5 assembly-map blocks")
    keff_by_no = parse_master_keff_by_no(mas_sum_text)
    ras_states, ras_keff, ndivxy, nya, nxa = parse_rasbery_assembly(args.h5)
    if not ras_states:
        raise SystemExit(f"{args.h5}: no steps/*/assembly data")
    if args.ndivxy is not None:
        ndivxy = args.ndivxy
    if not ras_keff:
        print("warning: .h5 has no summary/keff -- same-EFPD MASTER blocks "
              "(e.g. an ARO anchor state vs a rod-critical diagnostic printed "
              "at the same EFPD) cannot be disambiguated and fall back to the "
              "LAST block, which is wrong on decks like i-SMR CY01")

    if str(args.efpd_offset).strip().lower() == "auto":
        offset = min(ras_states) - min(master_states)
        print(f"efpd offset: auto -> {offset:.6f} d (RASBERY {min(ras_states):.6f} "
              f"aligned onto MASTER {min(master_states):.6f})")
    else:
        try:
            offset = float(args.efpd_offset)
        except ValueError:
            raise SystemExit(
                f"--efpd-offset {args.efpd_offset!r} is neither a number nor "
                f"`auto`") from None
    if offset:
        ras_states = {e - offset: cells for e, cells in ras_states.items()}
        ras_keff = {e - offset: k for e, k in ras_keff.items()}

    fold_kind = "mirror (no fold)" if args.mirror else f"rotational (ndivxy={ndivxy})"
    print(f"fold: {fold_kind}, geometry {nya}x{nxa} assembly grid")

    rows = build_rows(master_states, ras_states, ras_keff, keff_by_no, ndivxy,
                      args.mirror, args.efpd_tol)
    if not rows:
        raise SystemExit("no EFPD-matched statepoints between the two inputs")

    csv_path = Path(f"{args.out_prefix}.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})

    by_efpd: dict[float, list[dict]] = {}
    for r in rows:
        by_efpd.setdefault(r["efpd"], []).append(r)

    md_path = Path(f"{args.out_prefix}.md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"fold: {fold_kind}, geometry {nya}x{nxa} assembly grid\n\n")
        for e in sorted(by_efpd):
            state_rows = by_efpd[e]
            f.write(f"## efpd = {e:.3f}\n\n")
            cols = ["label", "master_power", "ras_power", "delta_power_pct",
                    "sym_power_a", "sym_power_b", "sym_power_diff",
                    "master_burn", "ras_burn", "delta_burn_gwdt"]
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "---|" * len(cols) + "\n")
            for r in state_rows:
                f.write("| " + " | ".join(fmt(r.get(c)) if r.get(c) is not None
                                          else "" for c in cols) + " |\n")
            pw = [r["delta_power_pct"] for r in state_rows if r["delta_power_pct"] is not None]
            bu = [r["delta_burn_gwdt"] for r in state_rows if r["delta_burn_gwdt"] is not None]
            rms_p, max_p = rms_max(pw)
            rms_b, max_b = rms_max(bu)
            f.write(f"\nmap power: rms={rms_p:.4f} % max|d|={max_p:.4f} % "
                    f"({len(pw)} assemblies)\n")
            f.write(f"map burnup: rms={rms_b:.4f} GWd/t max|d|={max_b:.4f} GWd/t "
                    f"({len(bu)} assemblies)\n\n")
            print(f"efpd={e:.3f}: power rms={rms_p:.4f}% max={max_p:.4f}%  "
                  f"burnup rms={rms_b:.4f} GWd/t max={max_b:.4f} GWd/t "
                  f"({len(state_rows)} physical assemblies)")

    print(f"{len(by_efpd)} statepoint(s), {len(rows)} physical-assembly row(s) "
          f"-> {csv_path}, {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
