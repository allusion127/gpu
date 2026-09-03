#!/usr/bin/env python3
"""Regress a fuel-temperature dT(LPD, burnup) grid from MASTER per-node edits.

WHY THIS EXISTS.  include/Database/tf.csv is MASTER's BUILT-IN WH-type table
(`isolth = 11`); its top-left corner is the %DEF_TFT example in the MASTER-3.0
manual.  The APR1400 / KNGR decks are `isolth = 12`, the ABB-CE table, whose
burnup slope of the rise is about five times steeper -- over one cycle MASTER's
rise falls 29.6 % while the shipped grid falls 5.8 %, worth about -14.6 K on
KNGR tfavg at BOC and +71 K at EOC.  The CE grid is NOT published in a form this
tree can copy, so it has to be MEASURED from MASTER's own edits, on the same
knots the shipped table uses, and that is all this tool does.  It does not
invent a table from a slope ratio; src/ThTfTable.h refuses to load a CE table
that has not been through here.

INPUT SCHEMA (CSV, one row per node per statepoint, header required).  Column
names are matched case-insensitively and by alias; unknown columns are ignored.

    efpd            (optional) statepoint EFPD -- carried into the report only
    node | plane | l | k    (optional) node identity -- carried into the report
    burnup | bu     burnup at the node [GWd/tHM]         REQUIRED
    tfuel | tf      node fuel temperature   [K]          REQUIRED
    tmod  | tm      node moderator temperature [K]       REQUIRED
    lpd             node linear power density [W/cm]     REQUIRED unless --lpd-from
    power | pnode   node power [MW] (with --lpd-from)
    hz              node axial height [cm] (with --lpd-from)
    rods            fuel rods per node (with --lpd-from)

The fitted quantity is dT = tfuel - tmod, which is exactly what
XSSet::GetTfuel's table holds.

    --lpd-from power,hz,rods
        Compute lpd = 1000 * P_node / (rods_per_node * hz) -- the SolveTH
        spelling (src/XSSet.cpp), with P_node in MW so `1000 *` lands in W/cm.
        The three names are the columns to read, in that order; `hz` and `rods`
        may instead be constants (e.g. `--lpd-from power,15.24,59`).

THE FIT.  The unknowns are the nbu x nlpd grid values themselves, and each
sample enters through the SAME bilinear weights milk::Table::Get would use
(including its clamping outside the knots), so the fitted grid reproduces the
data under the interpolator that will actually read it.  A second-difference
(curvature) penalty in both directions is the smoothing; --monotone adds
iteratively reweighted penalties against a dT that falls with LPD, which is
physics, not taste.  Cells with no data are REFUSED rather than filled, unless
--extrapolate says the smoothing may carry them.

USAGE
    tools/fit_tf_table.py edits.csv -o include/Database/tf_ce.csv
    tools/fit_tf_table.py edits.csv --lpd-from power,hz,59 -o out.csv
    tools/fit_tf_table.py --self-test        # synthetic recovery, no input
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOT_TABLE = ROOT / "include" / "Database" / "tf.csv"

BURNUP_ALIASES = ("burnup", "bu", "burnup_gwd", "burnup gwd/t")
TFUEL_ALIASES = ("tfuel", "tf", "fuel_temperature", "fuel temperature")
TMOD_ALIASES = ("tmod", "tm", "moderator_temperature", "moderator temperature")
LPD_ALIASES = ("lpd", "linear_power_density", "linear power density")
EFPD_ALIASES = ("efpd", "day", "days")
NODE_ALIASES = ("node", "l", "plane", "k", "node_id", "id")


# --------------------------------------------------------------------------
# The table file format -- tf.csv's, exactly: header "Bu/LPD" then the LPD
# knots, then one row per burnup knot.
# --------------------------------------------------------------------------
def read_table(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.reader(handle) if r and any(c.strip() for c in r)]
    lpd = [float(c) for c in rows[0][1:]]
    bu = [float(r[0]) for r in rows[1:]]
    dt = [[float(c) for c in r[1:]] for r in rows[1:]]
    for row in dt:
        if len(row) != len(lpd):
            sys.exit(f"{path}: a row does not have one entry per LPD knot")
    return lpd, bu, dt


def write_table(path: Path, lpd, bu, dt) -> None:
    def num(x: float) -> str:
        # tf.csv is written to two decimals; keep the shape a human can diff.
        text = f"{x:.2f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"

    lines = ["Bu/LPD," + ",".join(num(x) for x in lpd)]
    for j, b in enumerate(bu):
        lines.append(num(b) + "," + ",".join(num(v) for v in dt[j]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# milk::Table::Get's weights, transcribed.  The CLAMP matters: a sample outside
# the knots constrains the edge, it does not extrapolate.
# --------------------------------------------------------------------------
def bilinear_weights(lpd, bu, x: float, y: float):
    nx, ny = len(lpd), len(bu)
    x = min(max(x, lpd[0]), lpd[-1])
    y = min(max(y, bu[0]), bu[-1])
    ix = max(0, min(nx - 2, _lower(lpd, x)))
    iy = max(0, min(ny - 2, _lower(bu, y)))
    fx = (x - lpd[ix]) / (lpd[ix + 1] - lpd[ix])
    fy = (y - bu[iy]) / (bu[iy + 1] - bu[iy])
    return (
        ((iy, ix), (1 - fx) * (1 - fy)),
        ((iy, ix + 1), fx * (1 - fy)),
        ((iy + 1, ix), (1 - fx) * fy),
        ((iy + 1, ix + 1), fx * fy),
    )


def _lower(axis, value: float) -> int:
    lo, hi = 0, len(axis) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if axis[mid] <= value:
            lo = mid
        else:
            hi = mid
    return lo


def evaluate(lpd, bu, dt, x: float, y: float) -> float:
    return sum(dt[j][i] * w for (j, i), w in bilinear_weights(lpd, bu, x, y))


# --------------------------------------------------------------------------
# The solve.  90 unknowns for the shipped knots, so a dense symmetric system
# and a plain Cholesky-free Gaussian elimination is the right amount of tool.
# --------------------------------------------------------------------------
def solve(matrix, rhs):
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            sys.exit("fit_tf_table: the normal equations are singular; raise "
                     "--smooth or supply data that touches every knot.")
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for r in range(col + 1, n):
            f = a[r][col] * inv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = a[r][n] - sum(a[r][c] * out[c] for c in range(r + 1, n))
        out[r] = acc / a[r][r]
    return out


def fit(samples, lpd, bu, smooth: float, monotone: bool, passes: int = 4):
    """samples: (lpd, bu, dt).  Returns the grid, row-major over bu."""
    nx, ny = len(lpd), len(bu)
    n = nx * ny
    idx = lambda j, i: j * nx + i  # noqa: E731

    def curvature_rows():
        rows = []
        for j in range(ny):
            for i in range(1, nx - 1):
                rows.append({idx(j, i - 1): 1.0, idx(j, i): -2.0, idx(j, i + 1): 1.0})
        for i in range(nx):
            for j in range(1, ny - 1):
                rows.append({idx(j - 1, i): 1.0, idx(j, i): -2.0, idx(j + 1, i): 1.0})
        return rows

    curvature = curvature_rows()
    mono_pairs = [(idx(j, i), idx(j, i + 1)) for j in range(ny) for i in range(nx - 1)]
    mono_weight = [0.0] * len(mono_pairs)

    grid = None
    for _ in range(passes if monotone else 1):
        a = [[0.0] * n for _ in range(n)]
        b = [0.0] * n
        for x, y, value in samples:
            terms = bilinear_weights(lpd, bu, x, y)
            for (j0, i0), w0 in terms:
                r = idx(j0, i0)
                b[r] += w0 * value
                for (j1, i1), w1 in terms:
                    a[r][idx(j1, i1)] += w0 * w1
        for row in curvature:
            for r, wr in row.items():
                for c, wc in row.items():
                    a[r][c] += smooth * wr * wc
        for k, (lo, hi) in enumerate(mono_pairs):
            w = mono_weight[k]
            if w == 0.0:
                continue
            # penalise (z[lo] - z[hi]) > 0, i.e. dT falling with LPD
            for r, wr in ((lo, 1.0), (hi, -1.0)):
                for c, wc in ((lo, 1.0), (hi, -1.0)):
                    a[r][c] += w * wr * wc
        # A whisper of ridge so an untouched knot is pinned by the smoother
        # rather than by the pivot search.
        for r in range(n):
            a[r][r] += 1e-9
        grid_flat = solve(a, b)
        grid = [grid_flat[j * nx:(j + 1) * nx] for j in range(ny)]
        if not monotone:
            break
        violated = 0
        for k, (lo, hi) in enumerate(mono_pairs):
            drop = grid_flat[lo] - grid_flat[hi]
            if drop > 0.0:
                mono_weight[k] = max(mono_weight[k], 1.0) * 4.0
                violated += 1
        if violated == 0:
            break
    return grid


def coverage(samples, lpd, bu):
    """How many samples touch each grid node with a non-negligible weight."""
    counts = [[0] * len(lpd) for _ in range(len(bu))]
    for x, y, _ in samples:
        for (j, i), w in bilinear_weights(lpd, bu, x, y):
            if w > 1e-6:
                counts[j][i] += 1
    return counts


def residuals(samples, lpd, bu, grid):
    if not samples:
        return 0.0, 0.0
    errs = [evaluate(lpd, bu, grid, x, y) - v for x, y, v in samples]
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    return rms, max(abs(e) for e in errs)


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
def _pick(header, aliases):
    lower = {name.strip().lower(): name for name in header}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def read_samples(path: Path, lpd_from: str | None):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit(f"{path}: no data rows")
    header = list(rows[0].keys())

    col_bu = _pick(header, BURNUP_ALIASES)
    col_tf = _pick(header, TFUEL_ALIASES)
    col_tm = _pick(header, TMOD_ALIASES)
    if not (col_bu and col_tf and col_tm):
        sys.exit(f"{path}: need burnup, tfuel and tmod columns (see the docstring)")

    lpd_col = _pick(header, LPD_ALIASES)
    power_col = hz_spec = rods_spec = None
    if lpd_from:
        parts = [p.strip() for p in lpd_from.split(",")]
        if len(parts) != 3:
            sys.exit("--lpd-from takes exactly power,hz,rods")
        power_col, hz_spec, rods_spec = parts
        if power_col.lower() not in {h.strip().lower() for h in header}:
            sys.exit(f"{path}: --lpd-from names no such power column: {power_col}")
        power_col = _pick(header, (power_col.lower(),))
    elif lpd_col is None:
        sys.exit(f"{path}: no lpd column; pass --lpd-from power,hz,rods")

    def constant_or_column(spec):
        try:
            return float(spec), None
        except ValueError:
            col = _pick(header, (spec.lower(),))
            if col is None:
                sys.exit(f"{path}: --lpd-from names no such column: {spec}")
            return None, col

    hz_const = hz_col = rods_const = rods_col = None
    if lpd_from:
        hz_const, hz_col = constant_or_column(hz_spec)
        rods_const, rods_col = constant_or_column(rods_spec)

    samples = []
    for row in rows:
        try:
            bu = float(row[col_bu])
            dt = float(row[col_tf]) - float(row[col_tm])
            if lpd_from:
                hz = hz_const if hz_const is not None else float(row[hz_col])
                rods = rods_const if rods_const is not None else float(row[rods_col])
                if hz <= 0.0 or rods <= 0.0:
                    continue
                # src/XSSet.cpp SolveTH: lpd = 1000 * P_node / (rods * hz)
                lpd_value = 1000.0 * float(row[power_col]) / (rods * hz)
            else:
                lpd_value = float(row[lpd_col])
        except (TypeError, ValueError):
            continue
        if lpd_value <= 0.0:
            continue
        samples.append((lpd_value, bu, dt))
    if not samples:
        sys.exit(f"{path}: no usable rows")
    return samples


# --------------------------------------------------------------------------
# The self-test: a KNOWN table in, the same table out.
# --------------------------------------------------------------------------
def self_test(tolerance: float = 0.5) -> int:
    lpd, bu, truth = read_table(DEFAULT_KNOT_TABLE)
    # A CE-like table: the same shape with a five-times-steeper burnup decay,
    # which is the very difference this tool exists to measure.
    truth = [[truth[j][i] * (1.0 - 0.05 * bu[j] / max(bu)) for i in range(len(lpd))]
             for j in range(len(bu))]
    rng = random.Random(20260904)
    samples = []
    for _ in range(4000):
        x = rng.uniform(lpd[0], lpd[-1])
        y = rng.uniform(bu[0], bu[-1])
        samples.append((x, y, evaluate(lpd, bu, truth, x, y)))
    grid = fit(samples, lpd, bu, smooth=1e-3, monotone=True)
    worst = max(abs(grid[j][i] - truth[j][i])
                for j in range(len(bu)) for i in range(len(lpd)))
    rms, max_res = residuals(samples, lpd, bu, grid)
    print(f"  self-test: worst knot error {worst:.4f} K, residual rms {rms:.4f} K, "
          f"max {max_res:.4f} K")
    if worst > tolerance:
        print(f"fit_tf_table self-test: FAIL (worst {worst:.4f} K > {tolerance} K)",
              file=sys.stderr)
        return 1
    print("fit_tf_table self-test: PASS")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edits", nargs="?", type=Path, help="MASTER per-node edit CSV")
    ap.add_argument("-o", "--out", type=Path, help="output table (tf.csv format)")
    ap.add_argument("--knots", type=Path, default=DEFAULT_KNOT_TABLE,
                    help="table whose LPD/burnup knots to fit on (default tf.csv)")
    ap.add_argument("--lpd-from", metavar="power,hz,rods",
                    help="derive lpd = 1000*P/(rods*hz); hz and rods may be constants")
    ap.add_argument("--smooth", type=float, default=1e-3,
                    help="curvature penalty weight (default 1e-3)")
    ap.add_argument("--no-monotone", action="store_true",
                    help="do not penalise a dT that falls with LPD")
    ap.add_argument("--extrapolate", action="store_true",
                    help="allow knots no sample touches to be carried by the smoother")
    ap.add_argument("--self-test", action="store_true",
                    help="fit a synthetic table and check it is recovered within 0.5 K")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.edits is None:
        ap.error("an edits CSV is required unless --self-test")

    lpd, bu, _ = read_table(args.knots)
    samples = read_samples(args.edits, args.lpd_from)
    counts = coverage(samples, lpd, bu)

    print(f"  [FIT] {len(samples)} samples on a {len(lpd)} LPD x {len(bu)} burnup grid")
    empty = [(bu[j], lpd[i]) for j in range(len(bu)) for i in range(len(lpd))
             if counts[j][i] == 0]
    print("  [FIT] coverage per knot (rows = burnup, columns = LPD):")
    for j in range(len(bu)):
        print(f"    bu {bu[j]:>6}: " + " ".join(f"{counts[j][i]:>5}"
                                                for i in range(len(lpd))))
    if empty and not args.extrapolate:
        print(f"fit_tf_table: {len(empty)} knot(s) have NO data "
              f"(first: bu {empty[0][0]}, lpd {empty[0][1]}).  Refusing to invent "
              "them; supply data that reaches them or pass --extrapolate and say so "
              "in the receipt.", file=sys.stderr)
        return 2

    grid = fit(samples, lpd, bu, args.smooth, not args.no_monotone)
    rms, worst = residuals(samples, lpd, bu, grid)
    print(f"  [FIT] residual rms {rms:.4f} K, max {worst:.4f} K")

    if args.out:
        write_table(args.out, lpd, bu, grid)
        print(f"  [FIT] wrote {args.out}")
    else:
        print("Bu/LPD," + ",".join(f"{x:g}" for x in lpd))
        for j in range(len(bu)):
            print(f"{bu[j]:g}," + ",".join(f"{v:.2f}" for v in grid[j]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
