#!/usr/bin/env python3
"""Turn a MASTER run's MAS_OUT edits into the sample CSV tools/fit_tf_table.py eats.

WHAT THE SAMPLES ARE.  One row per (fuel assembly, statepoint), from the three
map edits MASTER prints at every edit point:

    $FB2D_n   axially averaged Tm(C), Tf(C), Dm(g/cc) per assembly  -> dT
    $P2D_n    axially averaged relative FA power                    -> LPD
    $B2D_n    axially averaged FA burnup [MWd/kgU]                  -> burnup

WHY NOT PER 3-D NODE.  Because this MASTER build's restart file does not carry
node temperatures at all -- see the "WHAT THE RESTART DOES NOT CONTAIN" section
of tools/parse_master_rst.py, which proves it rather than assuming it.  The
`$FB2D` map is the finest fuel temperature MASTER prints.  The consequences are
stated plainly because they bound what the fitted table can claim:

  * The sample dT is mean_k dT(bu_k, lpd_k) over the 25 fuel planes of the
    assembly, not dT at the assembly's mean (bu, lpd).  dT is very nearly
    linear in LPD (the shipped WH grid's LPD slope varies by 7 % over its whole
    0-450 W/cm span), so this Jensen gap is second order in the axial power
    spread -- but it is not zero, and it biases the fit slightly LOW at high
    LPD where the grid is convex.
  * The LPD it pairs with is likewise the assembly's axial mean, so the samples
    reach only about 270 W/cm even though 3-D nodes in this core run past 330.
    That is a COVERAGE limit, and fit_tf_table.py reports it as one.

GEOMETRY COMES FROM THE RESTART, not from a MAS_INP that may have drifted: the
restart echoes the deck it was actually run with, so --restart fixes the core
power, the fuel height, nfrod and the assembly count in one place, and the same
file's 3-D burnup is checked against $B2D on the way through.

THE LPD SPELLING is SolveTH's (src/XSSet.cpp:6394),
    lpd = 1000 * P_node / (fuel_rods_per_node * hz)
with P_node in kW and hz in cm, so this tool emits `power` in kW and the fit is
run with `--lpd-from power,<fuel height>,<nfrod>`.  For an AXIALLY AVERAGED
sample the "node" is the whole fuel column, so hz is the full fuel height and
P_node is the whole assembly's power:

    P_assembly[kW] = 1000 * core_power[MW] * relative_FA_power / n_assemblies

USAGE
    tools/build_tf_dataset.py MAS_OUT --restart MAS_RST.APRQ_01_0360.00 -o edits.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_master_rst import Restart, read_map_blocks  # noqa: E402


def build(mas_out: Path, rst: Restart, drop_zero_power: float = 1e-6):
    """The samples, plus the receipt numbers worth printing."""
    deck = rst.deck
    n_fa = len(deck.fuel)

    fb2d = read_map_blocks(mas_out, "FB2D", 3)
    p2d = read_map_blocks(mas_out, "P2D", 2)
    b2d = read_map_blocks(mas_out, "B2D", 2)

    by_efpd = {}
    for source, blocks in (("fb", fb2d), ("p", p2d), ("b", b2d)):
        for efpd, grid in blocks:
            # The EOC statepoint is printed twice (the depletion's last edit and
            # the following %EXE_STD); the second is the same state, so the
            # first wins and the duplicate is dropped rather than double counted.
            by_efpd.setdefault(efpd, {}).setdefault(source, grid)

    rows = []
    for efpd in sorted(by_efpd):
        got = by_efpd[efpd]
        if len(got) != 3:
            continue
        fb, pw, bu = got["fb"], got["p"], got["b"]
        for y, cols in pw.items():
            for x, power in cols.items():
                relative = power[0]
                if relative <= drop_zero_power:
                    continue
                if y not in bu or x not in bu[y] or y not in fb or x not in fb[y]:
                    continue
                tm, tf, _dm = fb[y][x]
                if tf <= 0.0:
                    continue  # a reflector node in the 19x19 $FB2D grid
                rows.append({
                    "efpd": f"{efpd:g}",
                    "node": f"{x}{y}",
                    "burnup": f"{bu[y][x][0]:.4f}",
                    "power": f"{1000.0 * deck.power_mw * relative / n_fa:.4f}",
                    "tf": f"{tf:.2f}",
                    "tm": f"{tm:.2f}",
                })
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mas_out", type=Path)
    ap.add_argument("--restart", type=Path, required=True,
                    help="any MAS_RST of the same run -- the geometry and the "
                         "burnup cross-check both come from it")
    ap.add_argument("--validate-restart", action="store_true",
                    help="also check the restart's 3-D burnup against $B2D/$B1D")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rst = Restart(args.restart)
    deck = rst.deck
    if args.validate_restart:
        from parse_master_rst import validate
        problems = validate(rst, args.mas_out)
        for p in problems:
            print("build_tf_dataset: " + p, file=sys.stderr)
        if problems:
            return 1

    rows = build(args.mas_out, rst)
    if not rows:
        print("build_tf_dataset: no samples", file=sys.stderr)
        return 1

    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["efpd", "node", "burnup",
                                                    "power", "tf", "tm"])
        writer.writeheader()
        writer.writerows(rows)

    height, rods = deck.fuel_height, deck.nfrod
    lpd = [1000.0 * float(r["power"]) / (rods * height) for r in rows]
    dt = [float(r["tf"]) - float(r["tm"]) for r in rows]
    burnup = [float(r["burnup"]) for r in rows]
    efpds = sorted({r["efpd"] for r in rows}, key=float)
    print(f"  [DATA] {len(rows)} samples from {len(efpds)} statepoints "
          f"({efpds[0]} .. {efpds[-1]} EFPD), {len(deck.fuel)} assemblies")
    print(f"  [DATA] core power {deck.power_mw:g} MW, fuel height {height:g} cm, "
          f"nfrod {rods}")
    print(f"  [DATA] lpd    {min(lpd):7.2f} .. {max(lpd):7.2f} W/cm "
          f"(mean {sum(lpd)/len(lpd):.2f})")
    print(f"  [DATA] burnup {min(burnup):7.3f} .. {max(burnup):7.3f} MWd/kgU")
    print(f"  [DATA] dT     {min(dt):7.2f} .. {max(dt):7.2f} K")
    print(f"  [DATA] wrote {args.out}")
    print(f"  [DATA] fit with: --lpd-from power,{height:g},{rods}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
