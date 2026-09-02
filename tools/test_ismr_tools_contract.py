#!/usr/bin/env python3
"""Contract: the three i-SMR comparison tools mean what the runbook says.

docs/ISMR_MASTER_COMPARISON_RUNBOOK_20260903_KO.md turns three claims into
commands a GPU host runs unattended:

  1. `delta_rod_cm_<BANK>` is the anchor-B metric, because on a rod-critical
     deck the two columns Gate B used to judge carry no model information.
  2. `gate_b_pin_rms.py --lattice 17` reads an i-SMR MAS_PPI, and the 16x16
     default no longer reads one WRONGLY.
  3. The 24-case screening set is 24 DISTINCT cases, because a duplicate inside
     a throughput measurement is free wall clock.

Each claim is checked here against fixtures built in a temp directory, so this
runs with no solver, no GPU and no cross-section library.

AND ONE CLAIM ABOUT THE DECKS, not the tools: the five test/7_i-SMR_Validation
decks declare `mirror: false`.  The MASTER references they are scored against
carry `%GEN_DIM ... nsym=1` and `%GEN_SYM -1 -1`, which src/Geometry.cpp:169
names as the 90-degree ROTATIONAL fold with the centre assembly divided.  A deck
that says `mirror: true` against those references is folding the core a
different way from the code it is being compared to, and on CY02/03/04 that is
worth up to 7.5 pcm and 5.6 % node power -- small enough to be mistaken for a
physics finding, which is why it is pinned here rather than left to review.

USAGE
    tools/test_ismr_tools_contract.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
CASE_DIR = ROOT / "test" / "7_i-SMR_Validation"
sys.path.insert(0, str(TOOLS))

FAILED: list[str] = []


def check(ok: bool, message: str) -> None:
    if not ok:
        FAILED.append(message)


# ---------------------------------------------------------------------------
# 1.  The rod column
# ---------------------------------------------------------------------------
#: A minimal SUMMARY EDIT 1 in MASTER's layout.  Row 1 is the SEARCH=1 unrodded
#: state MASTER solves before it searches, printed at the same EFPD 0.000 as
#: row 2 -- the row a naive parser keeps and a correct one drops.
EDIT1 = """   SUMMARY EDIT 1 : OPTIONS

   NO.     DAY     EFPD   XE/SM   SEARCH  RHO  RST  PPI  CMP  (CONTROL ROD POSITIONS)
                                                                S1      R3      S3      S2      R4      R1      S4      R2      S5      S6

    1    0.000    0.000    1  2      1     0    1    0    0   240.000 240.000 240.000 240.000 240.000 240.000 240.000 240.000 240.000 240.000
    2    0.000    0.000    1  2      5     0    1    0    0   240.000 117.362 240.000 240.000   0.000 240.000 240.000 237.362 240.000 240.000
    3   19.631   19.631    1  2      5     0    1    0    0   240.000 149.877 240.000 240.000  29.877 240.000 240.000 240.000 240.000 240.000

   SUMMARY EDIT 2 : REACTIVITY
"""


def rod_parser_contract() -> None:
    import compare_master_rasbery as cmp_mod

    with tempfile.TemporaryDirectory() as td:
        sum_path = Path(td) / "depf_02.sum"
        sum_path.write_text(EDIT1, encoding="utf-8", newline="\n")
        names, rows = cmp_mod.parse_master_rods(sum_path)

    check(names == ["S1", "R3", "S3", "S2", "R4", "R1", "S4", "R2", "S5", "S6"],
          f"EDIT 1 bank order read as {names}; it is NOT sorted and NOT grouped "
          f"by type, and pairing R3's trajectory with R4's column reports "
          f"agreement as a 120 cm error")
    check(sorted(rows) == [0.0, 19.631],
          f"EDIT 1 rows keyed {sorted(rows)}; EFPD 0.000 must appear once")
    # Last row at an EFPD wins, so the SEARCH=1 all-out row 1 loses to row 2.
    check(rows.get(0.0, [None] * 2)[1] == 117.362,
          f"EFPD 0.000 kept R3 at {rows.get(0.0, ['?'] * 2)[1]}; row 1 is the "
          f"unrodded SEARCH=1 state and must not survive to be compared against "
          f"a converged rod-critical statepoint")

    # The column offset is CHECKED, not trusted: a MASTER edit that adds a field
    # must fail loudly rather than shift every bank by one.
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "shifted.sum"
        bad.write_text(EDIT1.replace(
            "    1  2      5     0    1    0    0   240.000 117.362",
            "    1  2      5     0    1    0    0  9 240.000 117.362"),
            encoding="utf-8", newline="\n")
        try:
            cmp_mod.parse_master_rods(bad)
            check(False, "a shifted EDIT 1 row parsed silently; the first rod "
                         "column must be asserted, not assumed")
        except SystemExit as exc:
            check("REF_ROD_START_COL" in str(exc) or "rod column" in str(exc),
                  f"shifted EDIT 1 refused with an unhelpful message: {exc}")

    check(cmp_mod.ROD_TOP_CM == 240.0,
          f"ROD_TOP_CM is {cmp_mod.ROD_TOP_CM}; pos_cm = 240.0 - insertion is "
          f"the convention tools/plot_ismr_validation.py already publishes")

    # THE PINNED RULE IS A NEGATIVE CONTROL AS MUCH AS A CHECK.  A set of banks
    # that never moved must NOT come back PASS.
    joined = [{"delta_rod_cm_S1": 0.0, "master_rod_cm_S1": 240.0,
               "ras_rod_cm_S1": 240.0},
              {"delta_rod_cm_S1": 0.0, "master_rod_cm_S1": 240.0,
               "ras_rod_cm_S1": 240.0}]
    with contextlib.redirect_stdout(io.StringIO()):
        result = cmp_mod.rod_report(joined, ["S1"], 5.0, 2.0)
    check(result == (False, 2),
          f"a bank parked at 240 on both sides scored {result}; a column that "
          f"could not have failed is not evidence and must be NOT SCORED (2)")

    moved = [{"delta_rod_cm_R3": 0.5, "master_rod_cm_R3": 117.362,
              "ras_rod_cm_R3": 117.862},
             {"delta_rod_cm_R3": -0.5, "master_rod_cm_R3": 149.877,
              "ras_rod_cm_R3": 149.377}]
    with contextlib.redirect_stdout(io.StringIO()):
        ok_pass = cmp_mod.rod_report(moved, ["R3"], 5.0, 2.0)
        ok_max = cmp_mod.rod_report(moved, ["R3"], 0.1, 2.0)
        ok_rms = cmp_mod.rod_report(moved, ["R3"], 5.0, 0.1)
        ok_none = cmp_mod.rod_report(moved, [], 5.0, 2.0)
    check(ok_pass == (True, 0), "a moving bank inside both bars did not PASS")
    check(ok_max == (False, 1), "a moving bank outside the max bar did not FAIL")
    check(ok_rms == (False, 1), "a moving bank outside the RMS bar did not FAIL")
    check(ok_none is None,
          "a deck with no rod banks must leave the exit code to the scalars")


def rod_reference_contract() -> None:
    """The shipped MASTER references really do carry the block this depends on."""
    import compare_master_rasbery as cmp_mod

    ref = CASE_DIR / "Reference_output" / "depf_02.sum"
    if not ref.is_file():
        print("ismr tools contract: NOTE: no Reference_output/depf_02.sum; the "
              "live EDIT 1 half did not run")
        return
    names, rows = cmp_mod.parse_master_rods(ref)
    check(set(names) >= {"R1", "R2", "R3", "R4"},
          f"depf_02.sum EDIT 1 has banks {names}; the four regulating banks are "
          f"the anchor-B metric")
    check(len(rows) > 5,
          f"depf_02.sum EDIT 1 yielded {len(rows)} statepoints; anchor B is the "
          f"WHOLE cycle, not a point")
    moving = [n for i, n in enumerate(names)
              if len({r[i] for r in rows.values() if i < len(r)}) > 1]
    check(bool(moving),
          "no bank moves anywhere in depf_02.sum, so the rod gate would be NOT "
          "SCORED on the deck family it exists for")


# ---------------------------------------------------------------------------
# 2.  The lattice-generic pin parser
# ---------------------------------------------------------------------------
def _ppi(seats: list[tuple[str, int]], npin: int, nplanes: int = 2) -> str:
    out = []
    for label, row in seats:
        out.append(f"  FANAME  {label}   {row}   {nplanes}    10.00")
        out.append("PIN 3-D POWER DISTRIBUTION (BOTTOM TO TOP)  (I - NX, J - NY)")
        for _ in range(nplanes):
            for _ in range(npin):
                out.append(" ".join(["1.000000"] * npin))
        out.append("**********")
    return "\n".join(out) + "\n"


def pin_parser_contract() -> None:
    import gate_b_pin_rms as pin

    check(pin.DEFAULT_LATTICE == 16 and pin.DEFAULT_ORIGIN == "J9",
          f"defaults moved to {pin.DEFAULT_LATTICE}/{pin.DEFAULT_ORIGIN}; every "
          f"existing two-argument KNGR invocation must still score exactly what "
          f"it scored before")

    with tempfile.TemporaryDirectory() as td:
        # i-SMR: 17x17, full core A..I / 1..9, origin E5.
        ismr = Path(td) / "ismr.ppi"
        seats = [(c, r) for r in range(1, 10) for c in "ABCDEFGHI"]
        ismr.write_text(_ppi(seats, 17), encoding="utf-8", newline="\n")

        parsed = pin.parse_ppi(ismr, npin=17, origin="E5")
        check((0, 0) in parsed,
              "origin E5 did not land on quarter index (0,0)")
        check(parsed[(0, 0)].shape == (17, 17),
              f"i-SMR seat parsed as {parsed[(0, 0)].shape}, not 17x17")
        check((4, 4) in parsed and (-4, -4) in parsed,
              "the full-core PPI did not produce the seats either side of the "
              "origin; only the non-negative quadrant is scored, and it must be "
              "the one containing (4,4)")

        # `auto` finds the same centre on a FULL-core file.
        auto = pin.parse_ppi(ismr, npin=17, origin="auto")
        check(set(auto) == set(parsed),
              "--origin auto disagreed with the explicit E5 centre on a "
              "full-core PPI")

        # THE FAILURE THIS REPLACES.  17x17x2 = 578 is not a whole number of
        # 16x16 planes; the old code reshaped 512 of them and printed an rms.
        try:
            pin.parse_ppi(ismr, npin=16, origin="E5")
            check(False, "a 17x17 PPI parsed as 16x16 without complaint -- this "
                         "is the silent reshape that produced a confident wrong "
                         "number in the units of the right one")
        except SystemExit as exc:
            check("--lattice" in str(exc),
                  f"the wrong-lattice refusal must name --lattice: {exc}")

        # The alphabet fallback: i-SMR's ninth column really is `I`, which the
        # PWR alphabet has no place for.
        alphabet, why = pin.resolve_labels({"A", "E", "I"}, None)
        check(alphabet == pin.ASCII_LABELS,
              f"a PPI containing column I resolved to {alphabet!r}; the PWR "
              f"alphabet skips I, so this file does not use it")
        check("A-Z" in why or "plain" in why,
              f"the fallback did not explain itself: {why}")
        kngr, _ = pin.resolve_labels({"A", "J", "T"}, None)
        check(kngr == pin.XL,
              "a PPI whose labels are all in the PWR alphabet must keep it; "
              "changing it would move every KNGR column offset")

        # An origin the file's own alphabet has no place for is a refusal, not
        # a silent offset.  `I` is the letter the PWR alphabet skips, so a
        # KNGR-labelled file is exactly where asking for it is a mistake.
        kngr_file = Path(td) / "kngr.ppi"
        kngr_file.write_text(
            _ppi([(c, r) for r in range(1, 4) for c in "ABC"], 16),
            encoding="utf-8", newline="\n")
        try:
            pin.parse_ppi(kngr_file, npin=16, origin="I2")
            check(False, "origin I2 was accepted against a PWR-alphabet PPI, "
                         "whose alphabet has no I at all")
        except SystemExit as exc:
            check("--col-labels" in str(exc),
                  f"the bad-origin refusal must say how to fix it: {exc}")


def half_pin_contract() -> None:
    """The fold's cut-line pins are half-pins, and that is not a pin error."""
    import numpy as np
    import gate_b_pin_rms as pin

    npin, mid = 17, 8
    full = np.ones((npin, npin))
    lower = np.full((npin, npin), np.nan)
    lower[mid:, :] = 1.0
    lower[mid, :] = 0.5          # the cut row really does carry half the power
    right = np.full((npin, npin), np.nan)
    right[:, mid:] = 1.0
    right[:, mid] = 0.5
    corner = np.full((npin, npin), np.nan)
    corner[mid:, mid:] = 1.0
    corner[mid, mid:] = 0.5
    corner[mid:, mid] = 0.5
    corner[mid, mid] = 0.25
    ras_pin = np.stack([corner, lower, right, full])
    ras_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

    cuts = pin.half_pin_cuts(ras_map, ras_pin, npin)
    check(set(cuts) == {(0, 0), (0, 1), (1, 0)},
          f"cut-line seats detected as {sorted(cuts)}; a full assembly must not "
          f"be corrected and every half one must be")
    check(cuts.get((0, 0)) == (True, True),
          "the corner seat is cut BOTH ways; its centre pin is a quarter-pin")
    check(cuts.get((0, 1)) == (True, False) and cuts.get((1, 0)) == (False, True),
          f"the two half seats were mis-classified: {cuts}")

    fixed = pin.apply_half_pin_correction(ras_pin, ras_map, cuts, npin)
    check(fixed[1][mid, 0] == 1.0,
          f"the cut row came back as {fixed[1][mid, 0]}, not the whole-pin 1.0")
    check(fixed[0][mid, mid] == 1.0,
          f"the quarter-pin came back as {fixed[0][mid, mid]}, not 1.0")
    check(np.array_equal(fixed[3], full),
          "a full assembly was modified by the correction")
    check(np.array_equal(ras_pin[1], lower, equal_nan=True),
          "the correction mutated its input instead of copying")

    # KNGR IS THE NEGATIVE CONTROL.  An even lattice has no centre pin row, so
    # there are no half-pins and the default reading must not move.
    even = np.full((2, 16, 16), np.nan)
    even[0, 8:, :] = 1.0
    check(pin.half_pin_cuts({(0, 0): 0}, even, 16) == {},
          "an even lattice reported a half-pin cut; with no centre pin row "
          "there is nothing to halve, and correcting one would move KNGR's "
          "frozen production figures")


def pin_reference_contract() -> None:
    """Against a real i-SMR MAS_PPI, when one is staged.

    `ISMR_HIGA_MASTER_DIR` points at the HIGA MASTER run directory (the one
    holding `ISMR-CY01*/MAS_PPI.SMF_01_*`).  It is an environment variable and
    not a path in this file because that directory is out-of-band fixture
    corpus and lives somewhere different on every host that has it.
    """
    import gate_b_pin_rms as pin

    root = os.environ.get("ISMR_HIGA_MASTER_DIR", "")
    candidates = sorted(Path(root).glob("*/MAS_PPI.SMF_01_0000.00")) if root else []
    if not candidates:
        print("ismr tools contract: NOTE: no i-SMR MAS_PPI (set "
              "ISMR_HIGA_MASTER_DIR to the HIGA MASTER run directory); the live "
              "pin half did not run")
        return
    parsed = pin.parse_ppi(candidates[0], npin=17, origin="E5")
    check(len(parsed) == 69,
          f"the HIGA MAS_PPI yielded {len(parsed)} seats; the i-SMR core has 69")
    check(all(v.shape == (17, 17) for v in parsed.values()),
          "a HIGA seat did not parse as 17x17")


# ---------------------------------------------------------------------------
# 3.  The 24-case screening set
# ---------------------------------------------------------------------------
def screening_set_contract() -> None:
    import make_ismr_screening_set as mk

    total = (len(mk.ROD_BANKS) * len(mk.ROD_ENDPOINTS_CM)
             + len(mk.SHUFFLE_VARIANTS) + len(mk.BORON_PPM))
    check(total == 24,
          f"the families sum to {total}, not the 24 the plan sizes as the "
          f"minimum meaningful set (>= 16)")
    check(0.0 not in mk.ROD_ENDPOINTS_CM,
          "a 0 cm endpoint leaves a bank fully inserted with no travel, which "
          "on a rod-search deck is a search with no freedom rather than a "
          "different loading")

    base = CASE_DIR / "i-SMR_CY02.json"
    if not base.is_file():
        print("ismr tools contract: NOTE: no i-SMR_CY02.json; the live set half "
              "did not run")
        return
    cases = mk.build_cases(json.loads(base.read_text(encoding="utf-8-sig")))
    check(len(cases) == 24, f"build_cases() produced {len(cases)} cases")
    check(len({n for n, _ in cases}) == 24, "two cases share a name")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        # --no-coarsen: this half is about DISTINCTNESS, and coarsening is a
        # schedule rewrite that make_screening_deck.py owns and already tests.
        proc = subprocess.run(
            [sys.executable, str(TOOLS / "make_ismr_screening_set.py"), str(base),
             "-o", str(out), "--no-coarsen"],
            capture_output=True, text=True)
        check(proc.returncode == 0,
              f"make_ismr_screening_set.py exited {proc.returncode}:\n"
              f"{proc.stdout}\n{proc.stderr}")
        decks = sorted(out.glob("candidate_*.json"))
        check(len(decks) == 24, f"{len(decks)} candidate decks written, not 24")
        manifest = out / "manifest.txt"
        check(manifest.is_file(), "no manifest written")
        if manifest.is_file():
            # The manifest is read by run_multi_gpu_batch.read_manifest(), which
            # is held in lockstep with the executable's own parser and refuses a
            # one-field line.  Parsing it HERE is the only way this file finds
            # out before a GPU host does.
            sys.path.insert(0, str(TOOLS))
            try:
                from run_multi_gpu_batch import read_manifest
                jobs = read_manifest(manifest)
                check(len(jobs) == 24,
                      f"the batch runner read {len(jobs)} jobs from the manifest")
                check(len({o for _, o, _ in jobs}) == 24,
                      "two manifest jobs write the same output .h5")
            except ImportError as exc:
                print(f"ismr tools contract: NOTE: run_multi_gpu_batch not "
                      f"importable ({exc}); the manifest half did not run")

        import case_key as ck
        digests = {}
        for d in decks:
            digests.setdefault(
                ck.case_key(d, env={}, xslib=False)["deck_digest"], []).append(d.name)
        dupes = {k: v for k, v in digests.items() if len(v) > 1}
        check(not dupes,
              f"case_key collision(s) {list(dupes.values())}; a duplicate deck "
              f"inside a throughput set is a cache hit reported as a solved case")


# ---------------------------------------------------------------------------
# 4.  The decks' fold convention
# ---------------------------------------------------------------------------
DECKS = ("i-SMR_CY01.json", "i-SMR_CY02.json", "i-SMR_CY03.json",
         "i-SMR_CY04.json", "cy02_step1.json")


def fold_contract() -> None:
    # THE DECKS ARE NOT IN THE REPOSITORY.  `.gitignore` excludes `/test/**` --
    # it is ~3.9 GB of fixture corpus distributed out of band -- so a clone has
    # no i-SMR deck to check and this half must SAY it did not run rather than
    # fail.  Where the fixtures ARE staged (this workstation, 238, 181), the
    # check is real and the flag is pinned.
    present = [n for n in DECKS if (CASE_DIR / n).is_file()]
    if not present:
        print("ismr tools contract: NOTE: test/7_i-SMR_Validation is not staged "
              "here (it is an out-of-band fixture corpus, .gitignore /test/**); "
              "the fold-convention half did not run")
        return
    if len(present) != len(DECKS):
        print(f"ismr tools contract: NOTE: only {len(present)} of {len(DECKS)} "
              f"i-SMR decks are staged here; checking those")
    for name in present:
        path = CASE_DIR / name
        sym = json.loads(path.read_text(encoding="utf-8-sig"))["geometry"]["symmetry"]
        check(sym.get("mirror") is False,
              f"{name} declares mirror={sym.get('mirror')}. The MASTER "
              f"references it is scored against carry %GEN_DIM nsym=1 and "
              f"%GEN_SYM -1 -1, which src/Geometry.cpp:169 names as the "
              f"90-degree ROTATIONAL fold; folding the core the other way is "
              f"worth up to 7.5 pcm and 5.6 % node power on CY02/03/04")
        check(sym.get("angle") == 90,
              f"{name} declares angle={sym.get('angle')}; the rotational fold "
              f"is a quarter-core convention")
        check(sym.get("center assembly divided") is True,
              f"{name} does not divide the centre assembly, which is what "
              f"%GEN_SYM isymlx=isymly=-1 states")
        albedo = json.loads(path.read_text(encoding="utf-8-sig"))["geometry"]["albedo"]
        # Geometry::Initialize REFUSES a rotational fold whose two cut planes
        # are not the same boundary, so this is a precondition of the flag above
        # and not a style note.
        check(albedo.get("west") == albedo.get("north"),
              f"{name} has west albedo {albedo.get('west')} and north "
              f"{albedo.get('north')}; the rotational fold identifies those two "
              f"cut planes and Geometry::Initialize throws when they differ")


def main() -> int:
    rod_parser_contract()
    rod_reference_contract()
    pin_parser_contract()
    half_pin_contract()
    pin_reference_contract()
    screening_set_contract()
    fold_contract()
    for name in ("compare_master_rasbery.py", "gate_b_pin_rms.py",
                 "make_ismr_screening_set.py"):
        py_compile.compile(str(TOOLS / name), doraise=True)
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)

    if FAILED:
        for message in FAILED:
            print(f"ismr tools contract: FAIL: {message}")
        return 1
    print("ismr tools contract: PASS (rod column + lattice-generic pin parser + "
          "24-case screening set + rotational fold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
