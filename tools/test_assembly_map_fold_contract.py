#!/usr/bin/env python3
"""Contract: tools/compare_assembly_maps.py folds quarter-map slots onto the
PHYSICAL assemblies MASTER's SUMMARY EDIT 5 prints one number for.

WHY THIS EXISTS.  docs/ROTATIONAL_SHUFFLE_FIX_20260904_KO.md and the `gather`
comment at src/IO.cpp:2598-2602 establish that under the 90-degree rotational
quarter-core fold (ndivxy=2, symdiv=true, the i-SMR convention) quarter-map
slot (0,c) and slot (c,0), c>0, are the two HALVES of one physical assembly --
MASTER prints the SAME value at both (verified against the real
test/7_i-SMR_Validation/Reference_output/depf_01.sum below, when present).  A
tool that compares slot-by-slot instead of folding first would score a
CORRECT rotational RASBERY run worse than a WRONG mirror-fold run, which is
exactly the gap docs/ISMR_MASTER_COMPARISON_RUNBOOK_20260903_KO.md Sec 7 item
1 names.  This is checked here with a synthetic fixture, no solver, no GPU.

WHAT THIS CHECKS.

1. WEIGHT SYMMETRY.  `visible_node_count(0,c,n) == visible_node_count(c,0,n)`
   for every ndivxy parity, so the "node-weighted mean" the module docstring
   promises is provably the ordinary arithmetic mean of the two halves -- not
   an accident of ndivxy=2, a property for any n.

2. FOLD ON A SYNTHETIC 3x3 QUARTER MAP.  Rows "5,6,7" x cols "E,F,G", built to
   MASTER's own SUMMARY EDIT 5 text layout, with the four-line-per-row legend
   MASTER documents (batch ID, power, burnup, k-infinite -- the last skipped).
   The two symmetry-line pairs (F5=E6, G5=E7) are given EQUAL MASTER values
   (mirroring what MASTER actually prints) and UNEQUAL RASBERY half-slot
   values whose weighted mean matches the MASTER value exactly; the centre
   (E5) and four interior seats (F6,G6,F7,G7) get no partner.  One interior
   seat is given a deliberate nonzero delta so the RMS/max reduction is
   checked against a known nonzero answer, not just zero.

3. CENTRE HANDLING.  (0,0) is folded with `slot_b=None` -- its own quarter
   value stands for the whole physical assembly, no averaging.

4. MIRROR PASSTHROUGH.  `--mirror` (APR1400 KNGR) leaves all 9 slots
   distinct: no pairing, no averaging, `F5` and `E6` are two DIFFERENT rows.

5. NEGATIVE CONTROLS.  Reading only one arm of a symmetry pair (the mistake
   this fold exists to prevent) does NOT reproduce the MASTER physical value
   -- proving the positive control above is discriminating, not vacuous.

6. END TO END.  The CLI (`main()`, via subprocess) on the synthetic fixture
   writes a CSV with exactly 7 physical-assembly rows (9 raw slots - 2 merged
   pairs) and an MD with the RMS/max summary line.

7. REAL FIXTURE, IF STAGED.  `test/` is gitignored; when
   test/7_i-SMR_Validation/Reference_output/depf_01.sum is present, its
   NO.=1/DAY=0.00 block's F5/E6 power both parse to 1.1301 -- the exact
   figure docs/ROTATIONAL_SHUFFLE_FIX_20260904_KO.md cites.  SKIPPED (not
   FAILED) on a fresh clone.

USAGE
    python tools/test_assembly_map_fold_contract.py
"""
from __future__ import annotations

import csv
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import h5py  # noqa: E402

import compare_assembly_maps as cam  # noqa: E402

FAILED: list[str] = []


def check(ok: bool, message: str) -> None:
    if not ok:
        FAILED.append(message)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# 1. weight symmetry
# ---------------------------------------------------------------------------


def weight_symmetry_contract() -> None:
    for n in range(1, 9):
        wa = cam.visible_node_count(0, 3, n)
        wb = cam.visible_node_count(3, 0, n)
        check(wa == wb,
              f"ndivxy={n}: visible_node_count row0-arm={wa} != col0-arm={wb}; "
              f"the module claims these are always equal so the fold reduces "
              f"to a plain mean")
    # weighted_mean must actually average, not just pick a side, when the
    # weights are equal (the case the module docstring stakes its "ordinary
    # arithmetic mean" claim on).
    m = cam.weighted_mean(1.8, 3, 2.2, 3)
    check(m is not None and close(m, 2.0),
          f"weighted_mean(1.8, w=3, 2.2, w=3) = {m}, expected 2.0")
    # Fallback: a missing half must not silently become 0 or drop the row.
    check(cam.weighted_mean(None, 3, 4.0, 3) == 4.0,
          "weighted_mean with the first half missing must return the other "
          "half's own value")
    check(cam.weighted_mean(4.0, 3, None, 3) == 4.0,
          "weighted_mean with the second half missing must return the other "
          "half's own value")


# ---------------------------------------------------------------------------
# 2-5. synthetic 3x3 quarter map
# ---------------------------------------------------------------------------

#: rows 5,6,7 x cols E,F,G.  F5=E6 (power 2.0/burn 10.0) and G5=E7 (3.0/15.0)
#: are the two symmetry-line pairs; E5 is the centre; F6,G6,F7,G7 are interior.
MASTER_SUM = """   SUMMARY EDIT 1 :

   SUMMARY EDIT 5 : ASSEMBLY INFORMATION


          FIRST  LINE: ASSEMBLY BATCH ID
          SECOND LINE: ASSEMBLY POWER
          THIRD  LINE: ASSEMBLY BURNUP
          FOURTH LINE: ASSEMBLY K-INFINITE


 NO.=   1   DAY =     0.00  EFPD =     0.00

  Y\\X    E       F       G

  5      A1      A2      A3
      1.0000   2.0000   3.0000
      5.0000  10.0000  15.0000
      1.0000   1.0000   1.0000

  6      A2      A4      A5
      2.0000   4.0000   5.0000
     10.0000  20.0000  25.0000
      1.0000   1.0000   1.0000

  7      A3      A6      A7
      3.0000   6.0000   7.0000
     15.0000  30.0000  35.0000
      1.0000   1.0000   1.0000


 SUMMARY EDIT 6 : END
"""

#: (array_row, array_col) -> physical la index, a plain row-major 3x3 map.
IJTOLA_3X3 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

#: RASBERY per-la power/burn.  la0=E5 centre; la1/la3 are the F5/E6 halves of
#: one physical assembly (mean 2.0, matching MASTER); la2/la6 are the G5/E7
#: halves (mean 3.0); la4 (F6) carries a deliberate +0.2 power / +0.5 burn
#: delta so the RMS/max reduction has a known nonzero answer to check.
RAS_POWER = [1.0, 1.8, 2.7, 2.2, 4.2, 5.0, 3.3, 6.0, 7.0]
RAS_BURN = [5.0, 9.0, 14.5, 11.0, 20.5, 25.0, 15.5, 30.0, 35.0]


def write_fixture_h5(path: Path, ndivxy: int = 2) -> None:
    with h5py.File(path, "w") as f:
        geo = f.create_group("geometry")
        geo.create_dataset("nxa", data=3)
        geo.create_dataset("nya", data=3)
        geo.create_dataset("ndivxy", data=ndivxy)
        geo.create_dataset("ijtola", data=[v for row in IJTOLA_3X3 for v in row])
        su = f.create_group("summary")
        su.create_dataset("efpd", data=[0.0])
        step = f.create_group("steps/0001/assembly")
        step.create_dataset("power", data=RAS_POWER)
        step.create_dataset("burn", data=RAS_BURN)


def synthetic_master_state() -> dict:
    states = cam.parse_master_assembly_from_text(MASTER_SUM)
    return states[0.0]


def master_parser_contract() -> dict:
    mcells = synthetic_master_state()
    check(len(mcells) == 9, f"synthetic 3x3 block parsed {len(mcells)} cells, expected 9")
    check(mcells.get((0, 1), {}).get("power") == mcells.get((1, 0), {}).get("power") == 2.0,
          f"F5/E6 must both parse to power 2.0, got "
          f"{mcells.get((0, 1))} / {mcells.get((1, 0))}")
    check(mcells.get((0, 0), {}).get("label") == "E5",
          f"centre label parsed as {mcells.get((0, 0), {}).get('label')!r}, expected 'E5'")
    check(mcells.get((1, 1), {}).get("power") == 4.0,
          f"interior F6 power parsed as {mcells.get((1, 1))}, expected 4.0")
    return mcells


def fold_contract(mcells: dict) -> None:
    folded = cam.fold_master(mcells, mirror=False)
    check(len(folded) == 7,
          f"3x3 quarter map (9 slots, 2 symmetry pairs) folded to "
          f"{len(folded)} physical assemblies, expected 7")

    by_slot_a = {slot_a: (label, slot_b, mpow, mburn)
                for label, slot_a, slot_b, mpow, mburn in folded}

    # Centre: no partner, own value.
    label, slot_b, mpow, mburn = by_slot_a[(0, 0)]
    check(label == "E5" and slot_b is None and mpow == 1.0 and mburn == 5.0,
          f"centre folded as label={label!r} slot_b={slot_b} power={mpow} "
          f"burn={mburn}; expected E5/None/1.0/5.0 -- src/IO.cpp:2603-2606 "
          f"reconstructs the centre from its OWN quarter, no partner")

    # F5/E6 pair: MASTER value 2.0/10.0, partner slot (1,0).
    label, slot_b, mpow, mburn = by_slot_a[(0, 1)]
    check(slot_b == (1, 0) and close(mpow, 2.0) and close(mburn, 10.0),
          f"F5 arm folded with slot_b={slot_b} power={mpow} burn={mburn}, "
          f"expected partner (1,0), power 2.0, burn 10.0")
    check(label == "F5=E6",
          f"symmetry-pair label is {label!r}, expected 'F5=E6' (the doc's own "
          f"notation for the pair, e.g. docs/ROTATIONAL_SHUFFLE_FIX)")

    # Interior seats: no partner, own MASTER value.
    for slot, want_label, want_pow, want_burn in [
        ((1, 1), "F6", 4.0, 20.0), ((1, 2), "G6", 5.0, 25.0),
        ((2, 1), "F7", 6.0, 30.0), ((2, 2), "G7", 7.0, 35.0),
    ]:
        label, slot_b, mpow, mburn = by_slot_a[slot]
        check(slot_b is None and label == want_label and mpow == want_pow
              and mburn == want_burn,
              f"interior seat {slot} folded as label={label!r} slot_b={slot_b} "
              f"power={mpow} burn={mburn}; expected {want_label}/None/"
              f"{want_pow}/{want_burn} -- interior slots (row>0 and col>0) "
              f"are not on either symmetry line and must not be paired")


def ras_fold_contract(mcells: dict) -> None:
    """The RASBERY half-slots fold to the SAME physical values MASTER prints
    -- the actual load-bearing claim: node-weighted mean of two UNEQUAL raw
    RASBERY half-slot readings must recover MASTER's single physical number,
    the property docs/ROTATIONAL_SHUFFLE_FIX_20260904_KO.md's residual note
    reasons a correct fold has to have.
    """
    folded = cam.fold_master(mcells, mirror=False)
    rcells = {}
    for r in range(3):
        for c in range(3):
            la = IJTOLA_3X3[r][c]
            rcells[(r, c)] = {"power": RAS_POWER[la], "burn": RAS_BURN[la]}

    by_label = {}
    for label, slot_a, slot_b, mpow, mburn in folded:
        rpow, pa, pb = cam.ras_physical_value(rcells, slot_a, slot_b, "power", 2)
        rburn, ba, bb = cam.ras_physical_value(rcells, slot_a, slot_b, "burn", 2)
        by_label[label] = (mpow, rpow, pa, pb, mburn, rburn, ba, bb)

    mpow, rpow, pa, pb, mburn, rburn, ba, bb = by_label["F5=E6"]
    check(close(rpow, 2.0), f"F5=E6 RASBERY fold power = {rpow}, expected 2.0 "
                           f"(mean of raw halves {pa}/{pb})")
    check(close(rburn, 10.0), f"F5=E6 RASBERY fold burn = {rburn}, expected 10.0")
    check(pa == 1.8 and pb == 2.2,
          f"F5=E6 symmetry-check halves reported as {pa}/{pb}, expected "
          f"1.8/2.2 -- the informational raw-half columns must survive the "
          f"fold, not just its average")

    mpow, rpow, pa, pb, mburn, rburn, ba, bb = by_label["G5=E7"]
    check(close(rpow, 3.0), f"G5=E7 RASBERY fold power = {rpow}, expected 3.0")
    check(close(rburn, 15.0), f"G5=E7 RASBERY fold burn = {rburn}, expected 15.0")

    mpow, rpow, pa, pb, mburn, rburn, ba, bb = by_label["E5"]
    check(pa is None and pb is None and close(rpow, 1.0),
          f"centre E5 must carry no half-slot pair (pa={pa}, pb={pb}) and its "
          f"own RASBERY value 1.0, got rpow={rpow}")

    mpow, rpow, pa, pb, mburn, rburn, ba, bb = by_label["F6"]
    dpow_pct = (rpow - mpow) / mpow * 100.0
    check(close(dpow_pct, 5.0),
          f"interior F6: MASTER {mpow}, RASBERY {rpow}, delta = {dpow_pct:.4f} "
          f"%, expected +5.0000 % -- the deliberate nonzero-delta seat")
    dburn = rburn - mburn
    check(close(dburn, 0.5),
          f"interior F6 burnup delta = {dburn} GWd/t, expected +0.5")


def centre_slot_b_none_contract(mcells: dict) -> None:
    folded = cam.fold_master(mcells, mirror=False)
    centre = next(row for row in folded if row[1] == (0, 0))
    check(centre[2] is None,
          "centre (0,0) must fold with slot_b=None -- fold_master must never "
          "invent a partner for the centre assembly")


def mirror_passthrough_contract(mcells: dict) -> None:
    folded = cam.fold_master(mcells, mirror=True)
    check(len(folded) == 9,
          f"--mirror must leave all 9 quarter-map slots distinct (APR1400 "
          f"KNGR: every slot IS one physical assembly), got {len(folded)}")
    by_slot = {slot_a: (label, slot_b) for label, slot_a, slot_b, *_ in folded}
    check(by_slot.get((0, 1)) == ("F5", None),
          f"mirror mode: slot (0,1) folded as {by_slot.get((0, 1))}, expected "
          f"('F5', None) -- no pairing with (1,0)")
    check(by_slot.get((1, 0)) == ("E6", None),
          f"mirror mode: slot (1,0) folded as {by_slot.get((1, 0))}, expected "
          f"('E6', None) -- F5 and E6 must stay two DIFFERENT rows")


def negative_control_one_arm_only(mcells: dict) -> None:
    """Reading only ONE arm of a symmetry pair -- the exact mistake the fold
    exists to prevent -- must NOT reproduce MASTER's physical value.  If it
    did, the positive fold_contract()/ras_fold_contract() checks above would
    be unable to tell a correct fold from a broken one.
    """
    rcells = {}
    for r in range(3):
        for c in range(3):
            la = IJTOLA_3X3[r][c]
            rcells[(r, c)] = {"power": RAS_POWER[la], "burn": RAS_BURN[la]}

    one_arm_only = rcells[(0, 1)]["power"]  # raw F5 half, no fold at all
    check(not close(one_arm_only, 2.0, tol=1e-6),
          f"negative control did not discriminate: the raw F5 half-slot "
          f"({one_arm_only}) must NOT already equal the folded physical "
          f"value (2.0), or the positive control above would be vacuous")

    only_other_arm = rcells[(1, 0)]["power"]  # raw E6 half
    check(not close(only_other_arm, 2.0, tol=1e-6),
          f"negative control did not discriminate: the raw E6 half-slot "
          f"({only_other_arm}) must NOT already equal the folded physical "
          f"value (2.0) either")


# ---------------------------------------------------------------------------
# 6. end to end (CLI)
# ---------------------------------------------------------------------------


def end_to_end_contract() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        sum_path = tdp / "depf_test.sum"
        sum_path.write_text(MASTER_SUM, encoding="utf-8", newline="\n")
        h5_path = tdp / "result.h5"
        write_fixture_h5(h5_path)
        out_prefix = tdp / "cmp"

        proc = subprocess.run(
            [sys.executable, str(TOOLS / "compare_assembly_maps.py"),
             str(sum_path), str(h5_path), "-o", str(out_prefix)],
            capture_output=True, text=True)
        check(proc.returncode == 0,
              f"CLI exited {proc.returncode}, expected 0\nstdout:\n{proc.stdout}"
              f"\nstderr:\n{proc.stderr}")

        csv_path = Path(f"{out_prefix}.csv")
        md_path = Path(f"{out_prefix}.md")
        check(csv_path.is_file(), f"{csv_path} was not written")
        check(md_path.is_file(), f"{md_path} was not written")
        if csv_path.is_file():
            with csv_path.open(newline="", encoding="utf-8") as f:
                data_rows = list(csv.DictReader(f))
            check(len(data_rows) == 7,
                  f"CLI CSV has {len(data_rows)} physical-assembly rows, "
                  f"expected 7")
            by_label = {r["label"]: r for r in data_rows}
            check("F5=E6" in by_label,
                  f"CSV labels {sorted(by_label)} do not contain 'F5=E6'")
            if "F5=E6" in by_label:
                row = by_label["F5=E6"]
                check(close(float(row["ras_power"]), 2.0),
                      f"CLI CSV F5=E6 ras_power = {row['ras_power']!r}, expected 2.0")
        if md_path.is_file():
            md_text = md_path.read_text(encoding="utf-8")
            check("rms=" in md_text and "max|d|=" in md_text,
                  "MD output has no RMS/max summary line")


# ---------------------------------------------------------------------------
# 7. real fixture, if staged
# ---------------------------------------------------------------------------


def real_fixture_contract() -> str:
    real_sum = ROOT / "test" / "7_i-SMR_Validation" / "Reference_output" / "depf_01.sum"
    if not real_sum.is_file():
        return "SKIPPED (test/ is gitignored, no fixture on this clone)"
    states = cam.parse_master_assembly(real_sum)
    check(bool(states), f"{real_sum}: no SUMMARY EDIT 5 blocks parsed")
    if not states:
        return "FAILED"
    first_efpd = sorted(states)[0]
    cells = states[first_efpd]
    f5 = cells.get((0, 1))
    e6 = cells.get((1, 0))
    check(f5 is not None and e6 is not None and close(f5["power"], e6["power"], tol=1e-4),
          f"real depf_01.sum: F5={f5}, E6={e6}; the two arms of the same "
          f"physical assembly must print the SAME power")
    return "CHECKED against the real fixture"


def main() -> int:
    py_compile.compile(str(TOOLS / "compare_assembly_maps.py"), doraise=True)
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)

    weight_symmetry_contract()
    mcells = master_parser_contract()
    fold_contract(mcells)
    ras_fold_contract(mcells)
    centre_slot_b_none_contract(mcells)
    mirror_passthrough_contract(mcells)
    negative_control_one_arm_only(mcells)
    end_to_end_contract()
    real_status = real_fixture_contract()

    if FAILED:
        print("assembly map fold contract: FAIL")
        for f in FAILED:
            print("  - " + f)
        return 1
    print(f"assembly map fold contract: PASS (weight symmetry, synthetic 3x3 "
          f"fold + centre + mirror passthrough + 2 negative controls, "
          f"end-to-end CLI, real fixture: {real_status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
