#!/usr/bin/env python3
"""Build the 24-case i-SMR batch screening set from one base deck.

WHY THIS EXISTS.  The batch throughput number the campaign quotes -- cases/hour
at M8x8 under MPS -- has never been measured on i-SMR, because no i-SMR loading
pattern set exists.  The 68 `benchmark/m64_manual/inputs/candidate_*.json` on the
GPU host are all APR1400 (apr1400_chiffon_rf.h5, nx=ny=10, npins=16): running
them and calling the result an i-SMR throughput would be quoting one reactor's
number under another reactor's name.

WHAT A SCREENING SET HAS TO BE, AND WHAT EACH RULE COSTS IF BROKEN:

  DISTINCT.  Every case must have its own case_key (tools/case_key.py).  Two
  identical decks under two names is a cache hit, and a cache hit inside a
  throughput measurement is free wall-clock that inflates cases/hour without
  any physics happening.  This script REFUSES to write a set with a collision,
  and prints which two cases collided.

  FIXED-COST.  Every case must consume the same burnup grid, which is what
  `tools/make_screening_deck.py --mode coarse` is for: it replaces the source
  deck's natural-EOC tail (`until boron ppm`) with a stated GWd/t grid.  With
  the tail left on, a candidate that happens to deplete faster simply runs fewer
  statepoints, and the batch's cases/hour then measures the population's average
  cycle length as much as the machine.

  DERIVED FROM A REAL DECK.  The base is i-SMR_CY02.json -- restart + shuffle +
  rod search, the heaviest of the four and the only one that exercises the
  shuffle path -- so a candidate is a perturbation of a deck that is known to
  run, not a synthetic core map.

THE 24, AND WHY THESE 24 (12 + 8 + 4):

  rod   12  Each regulating bank R1..R4 gets its profile ENDPOINT moved to 60,
            120 or 180 cm from the deck's 240 (fully withdrawn).  The 0 cm level
            named in the plan is deliberately NOT in the family: an endpoint at
            0 leaves a bank fully inserted with no travel, and on a rod-search
            deck that is a search with no freedom rather than a different
            loading.  The ARO and all-in extremes are what the base deck and the
            static endpoint set already cover.

  shuf   8  The rotation spec of the shuffled entries in `core` ("3,3/1/270")
            advanced by 90, 180 or 270 degrees, over three subsets (every
            entry, every other entry, the complementary half).  A rotation is
            the perturbation a real reload study makes and the one the fold
            convention is most sensitive to, which is exactly why it belongs in
            a set whose purpose is to be REPRESENTATIVE of the work.

  boron  4  `default parameters.boron_ppm` at 0, 300, 600 and 900.  This family
            is legal on the built-in i-SMR_Validation.h5 library, whose boron
            axis is intact.  IT IS NOT LEGAL ON THE HIGA LIBRARY, whose boron
            branch is degenerate -- a boron sweep there varies a number the
            cross sections do not respond to, which produces 4 cases that differ
            in their case_key and not in their physics.  `--allow-degenerate-
            boron` exists so that refusal can be overridden knowingly; nothing
            in this file guesses.

WHAT THIS SCRIPT DOES NOT DO: run anything.  It writes decks and a manifest.

Usage:
    python tools/make_ismr_screening_set.py test/7_i-SMR_Validation/i-SMR_CY02.json \\
        -o /path/to/ismr24 [--manifest ismr24.txt]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import case_key as case_key_mod  # noqa: E402

#: The regulating banks, in the order the family numbers its cases.
ROD_BANKS = ("R1", "R2", "R3", "R4")
#: Profile endpoints, cm from the bottom of the active core.
ROD_ENDPOINTS_CM = (60.0, 120.0, 180.0)
#: (subset, degrees).  Subsets index the rotation-carrying `core` entries in
#: row-major order: every one, the even ones, the odd ones.
SHUFFLE_VARIANTS = (
    ("all", 90), ("all", 180), ("all", 270),
    ("even", 90), ("even", 180), ("even", 270),
    ("odd", 90), ("odd", 180),
)
BORON_PPM = (0.0, 300.0, 600.0, 900.0)

#: A shuffled `core` entry: "<row>,<col>/<batch>/<rotation>".
CORE_ENTRY_RE = re.compile(r"^(\d+,\d+)/(\d+)/(\d+)$")

#: Libraries whose boron branch is degenerate, so a boron family measures
#: nothing.  Matched on the deck's `data.cross-section` basename.
DEGENERATE_BORON_LIBS = ("ismr_higa_xs.json", "ismr_higa_xs_bppmon.json")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rotatable_entries(core: list[list[str]]) -> list[tuple[int, int]]:
    """(row, col) of every `core` cell carrying a rotation, row-major."""
    out = []
    for r, row in enumerate(core):
        for c, cell in enumerate(row):
            if isinstance(cell, str) and CORE_ENTRY_RE.match(cell):
                out.append((r, c))
    return out


def rod_case(base: dict, bank: str, endpoint_cm: float) -> dict:
    deck = copy.deepcopy(base)
    cfg = deck.get("rod configuration", {})
    if bank not in cfg:
        raise SystemExit(f"base deck has no rod bank {bank}; "
                         f"it has {', '.join(sorted(cfg))}")
    profile = list(cfg[bank].get("profile", []))
    if not profile:
        raise SystemExit(f"rod bank {bank} has no `profile` to move")
    profile[-1] = endpoint_cm
    cfg[bank]["profile"] = profile
    return deck


def shuffle_case(base: dict, subset: str, degrees: int) -> dict:
    deck = copy.deepcopy(base)
    core = deck.get("core")
    if not core:
        raise SystemExit("base deck has no `core` map to rotate")
    seats = rotatable_entries(core)
    if not seats:
        raise SystemExit(
            "base deck's `core` carries no shuffled entry (`row,col/batch/rot`), "
            "so there is nothing for the shuffle family to perturb. Use a "
            "restart deck such as i-SMR_CY02.json.")
    if subset == "all":
        chosen = seats
    elif subset == "even":
        chosen = seats[0::2]
    else:
        chosen = seats[1::2]
    for r, c in chosen:
        m = CORE_ENTRY_RE.match(core[r][c])
        rot = (int(m.group(3)) + degrees) % 360
        core[r][c] = f"{m.group(1)}/{m.group(2)}/{rot}"
    return deck


def boron_case(base: dict, ppm: float) -> dict:
    deck = copy.deepcopy(base)
    deck.setdefault("default parameters", {})["boron_ppm"] = ppm
    return deck


def build_cases(base: dict) -> list[tuple[str, dict]]:
    """The 24, in a stable order: 12 rod, then 8 shuffle, then 4 boron."""
    cases: list[tuple[str, dict]] = []
    for bank in ROD_BANKS:
        for cm in ROD_ENDPOINTS_CM:
            cases.append((f"rod_{bank}_{int(cm):03d}cm", rod_case(base, bank, cm)))
    for subset, deg in SHUFFLE_VARIANTS:
        cases.append((f"shuf_{subset}_{deg:03d}", shuffle_case(base, subset, deg)))
    for ppm in BORON_PPM:
        cases.append((f"boron_{int(ppm):04d}ppm", boron_case(base, ppm)))
    return cases


def coarsen(src: Path, dst: Path, extra: list[str]) -> None:
    """Run make_screening_deck.py --mode coarse, in-process cost of one subprocess.

    Shelling out rather than importing is deliberate: make_screening_deck.py's
    entry point is main() with its own argparse and its own cost report, and one
    schedule-rewriting implementation is the point of using it at all.
    """
    cmd = [sys.executable, str(TOOLS / "make_screening_deck.py"), str(src),
           "--mode", "coarse", "-o", str(dst)] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"make_screening_deck.py failed on {src.name}:\n"
                         f"{proc.stdout}\n{proc.stderr}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path,
                    help="base deck (i-SMR_CY02.json: restart + shuffle + rod search)")
    ap.add_argument("-o", "--out-dir", type=Path, required=True,
                    help="directory the candidate decks are written to")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="job list for run_multi_gpu_batch.py --jobs "
                         "(default: <out-dir>/manifest.txt)")
    ap.add_argument("--prefix", default="candidate",
                    help="candidate file prefix (default: %(default)s)")
    ap.add_argument("--no-coarsen", action="store_true",
                    help="skip make_screening_deck.py --mode coarse. Only for "
                         "inspecting the perturbations: an uncoarsened set has "
                         "per-case burnup grids and its cases/hour is not a "
                         "machine measurement.")
    ap.add_argument("--allow-degenerate-boron", action="store_true",
                    help="write the boron family even against a library whose "
                         "boron branch is degenerate")
    ap.add_argument("--screening-arg", action="append", default=[],
                    metavar="ARG",
                    help="extra argument passed through to make_screening_deck.py")
    args = ap.parse_args(argv)

    base = load(args.base)
    lib = str(base.get("data", {}).get("cross-section", ""))
    if (Path(lib).name.lower() in DEGENERATE_BORON_LIBS
            and not args.allow_degenerate_boron):
        raise SystemExit(
            f"{args.base}: cross-section library {lib!r} has a degenerate boron "
            f"branch, so the 4 boron cases would differ in their case_key and "
            f"not in their physics -- 4 of 24 cases measuring nothing. Use the "
            f"built-in i-SMR_Validation.h5, or pass --allow-degenerate-boron if "
            f"you mean to.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(base)
    written: list[tuple[str, Path]] = []
    for i, (name, deck) in enumerate(cases, start=1):
        stem = f"{args.prefix}_{i:04d}_{name}"
        raw = args.out_dir / f"{stem}.raw.json"
        raw.write_text(json.dumps(deck, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
        final = args.out_dir / f"{stem}.json"
        if args.no_coarsen:
            raw.replace(final)
        else:
            coarsen(raw, final, args.screening_arg)
            raw.unlink()
        written.append((name, final))

    # DISTINCTNESS IS CHECKED ON THE DECK HALF OF THE KEY, not the whole key.
    # The library, the environment and the build are identical across the set by
    # construction, so folding them in would only add the same constant to all
    # 24 -- while making the check fail whenever the library is not on this
    # machine, which is exactly where the set is authored.
    keys: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for name, path in written:
        digest = case_key_mod.case_key(path, env={}, xslib=False)["deck_digest"]
        if digest in keys:
            collisions.append((keys[digest], name))
        keys[digest] = name
    if collisions:
        lines = "\n".join(f"  {a} == {b}" for a, b in collisions)
        raise SystemExit(
            f"{len(collisions)} case_key collision(s); a duplicate deck in a "
            f"throughput set is a cache hit reported as a solved case:\n{lines}")

    # THE MANIFEST IS run_multi_gpu_batch.py's, not a list of paths.  Its
    # read_manifest() -- deliberately in lockstep with the executable's own
    # rasberyReadJobManifest() -- takes `<input.json> <output.h5> [result-mode]`
    # and REFUSES a line with one field, so a bare path list would stop the
    # batch before the first case.
    manifest = args.manifest or (args.out_dir / "manifest.txt")
    manifest.write_text(
        "".join(f"{p} {p.with_suffix('.h5')}\n" for _, p in written),
        encoding="utf-8", newline="\n")

    print(f"{len(written)} candidates -> {args.out_dir}")
    print(f"  rod   {len(ROD_BANKS) * len(ROD_ENDPOINTS_CM):2d}  "
          f"{', '.join(ROD_BANKS)} x {', '.join(f'{c:g}' for c in ROD_ENDPOINTS_CM)} cm")
    print(f"  shuf  {len(SHUFFLE_VARIANTS):2d}  "
          f"{', '.join(f'{s}/{d}' for s, d in SHUFFLE_VARIANTS)}")
    print(f"  boron {len(BORON_PPM):2d}  "
          f"{', '.join(f'{p:g}' for p in BORON_PPM)} ppm")
    print(f"all {len(keys)} deck digests distinct; manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
