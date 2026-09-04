#!/usr/bin/env python3
"""Freeze the A=149 depletion-chain topology: Nd-149 -> Pm-149 -> Sm-149.

WHY THIS EXISTS.  RASBERY's transient Sm-149 poisoning depends on two data
files staying self-consistent: `include/Database/dep_trans.csv` (which
isotope receives which actinide's fission yield) and
`include/Database/dep_decay.csv` (which isotope decays into which, at what
rate).  If the Pm-149 fission-yield row ever moved to Nd-149, or the Sm-149
production rate ever drifted from Pm-149's decay constant, the A=149 chain
would silently poison the wrong node in time -- the total inventory at
equilibrium is nearly unchanged (Nd+Pm+Sm is conserved by the same total
yield), so a normal integral check would not catch it, but the *timing* of
the Sm-149 buildup after a Pm-149-forming event would be wrong by roughly the
1.96-day Pm-149 half-life.  This is exactly the class of bug the 181 3-D
Xe/Sm print comparison against MASTER is built to expose (see
docs/SM149_CHAIN_CONTRACT_20260904_KO.md), so this test pins the numbers the
spike hypothesis rests on.

WHAT THIS CHECKS.
  (a) dep_trans.csv row 611490 (Pm-149) carries the U-235/U-238/Pu-239
      fission-yield columns with the values MASTER uses, and row 601490
      (Nd-149) carries NONE of them (all zero).
  (b) dep_decay.csv row 621490 (Sm-149) has a nonzero production term from
      column 611490 (Pm-149) equal to lambda(Pm-149), and row 601490 (Nd-149)
      has a decay-loss term of -lambda(Nd-149) on its own diagonal.
  (c) a pure-python two-step Bateman check reproduces the closed-form
      Pm-149 -> Sm-149 buildup fraction at t = 1.963 d (the first RASBERY
      depletion-step size named in the spike-hypothesis discussion,
      docs/CAMPAIGN_STATUS_20260904_KO.md line ~116), cross-checked against
      an independently-coded forward-Euler integration of the same two-state
      ODE.  NOTE ON THE NUMBERS: with the CSV's actual lambda(Pm-149) =
      3.627372e-06 /s, N_Sm(1.963 d)/(P*t) = 1 - (1-exp(-lam*t))/(lam*t)
      evaluates to ~0.2531, not the "~0.50" ballpark a half-life-based mental
      estimate would suggest (lambda(Pm-149)'s own half-life is 2.21 d, not
      1.963 d -- the depletion step and the isotope's half-life are simply
      two different numbers that happen to be close).  This test freezes the
      REAL, reproducible closed-form value, not the approximate one, because
      a contract test that hard-codes an unreachable target is worse than no
      test: it would either be permanently red or get "fixed" by loosening
      the tolerance until it is vacuous.  The actual first-step keff spike
      amplitude (+205~266 pcm per docs/CAMPAIGN_STATUS_20260904_KO.md) is a
      3-D coupled-physics result this two-isotope closed form cannot
      reproduce on its own -- see docs/SM149_CHAIN_CONTRACT_20260904_KO.md.
  (d) negative controls: moving the Pm-149 yield row to Nd-149, or doubling
      lambda(Pm-149), must fail the corresponding assertion.
  (e) source-level checks: ComputeXeEquilibrium (src/XSSet.cpp) writes only
      the I-135/Xe-135/Xe-135m rows, and src/IO.cpp has no "samarium" deck
      key -- RASBERY's Sm-149 is computed transiently through the depletion
      matrix, never given an equilibrium shortcut, matching MASTER-3.0's
      SM mode 2 ("transient"), manual p.3-46.

DATA-FILE SEMANTICS (read from src/XSSet.cpp and include/chiffon/Model.h).
  include/chiffon/Model.h:118-119 loads both CSVs as
  `milk::LabeledMatrixFromCSV(dir / "dep_decay.csv", labels, labels)` and
  the same for dep_trans.csv, where `labels = isotopeIds` (Model.h:37-48),
  so BOTH matrices are square niso x niso with rows and columns in the same
  fixed isotope order; matrix(row, col) always reads as
  "column csv label" (row 1 of the file is the header holding the COLUMN
  labels, and cell [0] of each data row is the ROW label -- see
  `milk::LabeledMatrixFromCSV`, include/milk.h:1207-1249).

  depDecay(d, p): src/XSSet.cpp:4488-4523 `BuildTransitionMatrix` seeds
  `mat = depDecay` directly into the depletion ODE matrix dN/dt = mat * N.
  Row d, col p, d != p, holds +lambda_p (isotope d is PRODUCED by the decay
  of isotope p); the diagonal (p, p) holds -lambda_p (isotope p is LOST to
  its own decay). Confirmed on this tree: depDecay[621490][611490] (Sm-149
  row, Pm-149 col) = +3.627372e-06 = lambda(Pm-149); depDecay[611490][611490]
  = -3.627372e-06; depDecay[611490][601490] (Pm-149 row, Nd-149 col) =
  +1.114241e-04 = lambda(Nd-149) (Nd-149 decays into Pm-149);
  depDecay[601490][601490] = -1.114241e-04.

  depTrans(d, p): src/XSSet.cpp:4488-4514, same loop.  For an actinide parent
  p (iAcFirst..iAcLast, Model.h:67-69) and a non-actinide daughter d,
  `mat(d, p) += topo * xsff_val * sumflux` where `topo = depTrans(d, p)` is
  literally the isotope's cumulative fission yield for that actinide -- so
  dep_trans.csv's actinide-block columns (header positions for 922340..
  962450) hold fission-yield fractions, keyed by (daughter row, actinide
  column).  Confirmed: dep_trans.csv row 611490 (Pm-149), column "922350"
  (U-235) = 1.0820E-02; column "922380" (U-238) = 1.6250E-02; column
  "942390" (Pu-239) = 1.2160E-02.  Row 601490 (Nd-149) has all-zero values in
  every actinide column -- Nd-149 is populated only through decay of shorter-
  lived A=149 precursors, never directly by fission.

USAGE
    tools/test_sm149_chain_contract.py
"""
from __future__ import annotations

import csv
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEP_TRANS = ROOT / "include" / "Database" / "dep_trans.csv"
DEP_DECAY = ROOT / "include" / "Database" / "dep_decay.csv"
XSSET_CPP = ROOT / "src" / "XSSet.cpp"
IO_CPP = ROOT / "src" / "IO.cpp"

ND149 = "601490"
PM149 = "611490"
SM149 = "621490"
U235 = "922350"
U238 = "922380"
PU239 = "942390"

EXPECTED_YIELDS = {U235: 1.0820e-02, U238: 1.6250e-02, PU239: 1.2160e-02}
LAMBDA_PM149 = 3.627372e-06  # s^-1
LAMBDA_ND149 = 1.114241e-04  # s^-1

REL_TOL = 1e-6


def read_labeled_matrix(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Parse a milk::LabeledMatrixFromCSV-shaped CSV into row-label -> {col-label: value}."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header[0] == "isotope", f"{path}: unexpected corner cell {header[0]!r}"
    col_labels = header[1:]
    table: dict[str, dict[str, float]] = {}
    for r in rows[1:]:
        if not r:
            continue
        row_label = r[0]
        values = r[1:]
        assert len(values) == len(col_labels), (
            f"{path}: row {row_label} has {len(values)} values, expected {len(col_labels)}"
        )
        table[row_label] = {col: float(v) for col, v in zip(col_labels, values)}
    return col_labels, table


def check_trans_topology(trans: dict[str, dict[str, float]]) -> list[str]:
    """(a) Pm-149 carries the actinide yields; Nd-149 carries none."""
    failures = []
    pm_row = trans[PM149]
    for col, expected in EXPECTED_YIELDS.items():
        got = pm_row[col]
        if not math.isclose(got, expected, rel_tol=REL_TOL):
            failures.append(
                f"dep_trans.csv row {PM149} col {col}: got {got!r}, expected {expected!r}"
            )
    nd_row = trans[ND149]
    for col in EXPECTED_YIELDS:
        got = nd_row[col]
        if got != 0.0:
            failures.append(
                f"dep_trans.csv row {ND149} col {col}: expected 0.0 (no direct fission "
                f"yield into Nd-149), got {got!r}"
            )
    return failures


def check_decay_chain(decay: dict[str, dict[str, float]]) -> list[str]:
    """(b) Sm-149 production from Pm-149, Nd-149 self-decay loss."""
    failures = []
    got_sm_from_pm = decay[SM149][PM149]
    if not math.isclose(got_sm_from_pm, LAMBDA_PM149, rel_tol=REL_TOL):
        failures.append(
            f"dep_decay.csv row {SM149} col {PM149}: got {got_sm_from_pm!r}, "
            f"expected lambda(Pm-149) = {LAMBDA_PM149!r}"
        )
    got_nd_diag = decay[ND149][ND149]
    if not math.isclose(got_nd_diag, -LAMBDA_ND149, rel_tol=REL_TOL):
        failures.append(
            f"dep_decay.csv row {ND149} col {ND149}: got {got_nd_diag!r}, "
            f"expected -lambda(Nd-149) = {-LAMBDA_ND149!r}"
        )
    return failures


def bateman_pm_to_sm_fraction(lam: float, t: float) -> float:
    """N_Sm(t) / (P * t) for a fixed production P feeding Pm-149 which decays
    to Sm-149 (Sm-149 itself effectively stable on this timescale: its own
    depletion is via absorption, not decay, and is excluded here by design --
    this checks the DECAY topology only).

    ODE: dN_Pm/dt = P - lam*N_Pm,           N_Pm(0) = 0
         dN_Sm/dt = lam*N_Pm,               N_Sm(0) = 0

    Closed form: N_Pm(t) = (P/lam) * (1 - exp(-lam*t))
                 N_Sm(t) = P*t - (P/lam) * (1 - exp(-lam*t))
    so           N_Sm(t) / (P*t) = 1 - (1 - exp(-lam*t)) / (lam*t)
    """
    return 1.0 - (1.0 - math.exp(-lam * t)) / (lam * t)


def bateman_pm_to_sm_absolute(lam: float, prod: float, t: float, dt: float = 1.0) -> float:
    """Explicit two-species forward-Euler integration (fine dt) of the same
    ODE pair, used only to cross-check the closed form is being applied to
    the right two-step topology (Pm-149 explicit precursor)."""
    n_pm = 0.0
    n_sm = 0.0
    steps = int(round(t / dt))
    for _ in range(steps):
        d_pm = prod - lam * n_pm
        d_sm = lam * n_pm
        n_pm += d_pm * dt
        n_sm += d_sm * dt
    return n_sm


# The real closed-form value at t = 1.963 d with the CSV's lambda(Pm-149).
# See the module docstring / bullet (c) NOTE ON THE NUMBERS for why this is
# ~0.253 rather than the ~0.50 a half-life-based guess would suggest.
EXPECTED_BUILDUP_FRACTION = 0.2531463680183128


def check_bateman_timing() -> list[str]:
    """(c) The closed-form buildup fraction, frozen against its own real
    value, cross-checked by an independently-coded forward-Euler integration
    of the same ODE pair (this is the "two different implementations of the
    same physics must agree" check the negative controls in (d) also rely
    on)."""
    failures = []
    t = 1.963 * 86400.0  # 1.963 d in seconds -- the first RASBERY depletion
    # step named in the spike-hypothesis discussion.
    lam = LAMBDA_PM149

    frac = bateman_pm_to_sm_fraction(lam, t)
    if not math.isclose(frac, EXPECTED_BUILDUP_FRACTION, rel_tol=1e-6):
        failures.append(
            f"Bateman buildup fraction at t=1.963 d: got {frac!r}, "
            f"expected {EXPECTED_BUILDUP_FRACTION!r} (rel_tol 1e-6)"
        )

    prod = 1.0
    n_sm_explicit = bateman_pm_to_sm_absolute(lam, prod, t, dt=1.0)
    frac_numeric = n_sm_explicit / (prod * t)
    if not math.isclose(frac, frac_numeric, rel_tol=1e-3):
        failures.append(
            f"closed-form fraction {frac!r} disagrees with the independent "
            f"forward-Euler integration {frac_numeric!r} -- the two Bateman "
            f"derivations do not describe the same chain"
        )

    # Sanity bounds intrinsic to the physics, independent of the exact
    # numbers above: the buildup fraction must be strictly between 0 and 1,
    # and must increase monotonically with time (more of the produced
    # inventory has had time to decay through Pm-149 into Sm-149).
    if not (0.0 < frac < 1.0):
        failures.append(f"buildup fraction {frac!r} outside the physical range (0, 1)")
    frac_later = bateman_pm_to_sm_fraction(lam, 2.0 * t)
    if not (frac_later > frac):
        failures.append(
            f"buildup fraction did not increase with time: frac(t)={frac!r}, "
            f"frac(2t)={frac_later!r}"
        )
    return failures


def check_negative_controls() -> list[str]:
    """(d) Corrupted copies of dep_trans.csv / dep_decay.csv must FAIL."""
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # (d.1) Move the Pm-149 yield row to Nd-149.
        moved_trans = tmp_path / "dep_trans_moved.csv"
        shutil.copy(DEP_TRANS, moved_trans)
        _swap_rows_in_place(moved_trans, PM149, ND149)
        _, trans_bad = read_labeled_matrix(moved_trans)
        bad_result = check_trans_topology(trans_bad)
        if not bad_result:
            failures.append(
                "negative control PASSED -- moving the Pm-149 yield row to Nd-149 "
                "did not trip check_trans_topology()"
            )

        # (d.2) Double lambda(Pm-149) in dep_decay.csv.
        doubled_decay = tmp_path / "dep_decay_doubled.csv"
        shutil.copy(DEP_DECAY, doubled_decay)
        _scale_cell_in_place(doubled_decay, SM149, PM149, 2.0)
        _scale_cell_in_place(doubled_decay, PM149, PM149, 2.0)
        _, decay_bad = read_labeled_matrix(doubled_decay)
        bad_result2 = check_decay_chain(decay_bad)
        if not bad_result2:
            failures.append(
                "negative control PASSED -- doubling lambda(Pm-149) did not trip "
                "check_decay_chain()"
            )
    return failures


def _swap_rows_in_place(path: Path, row_a: str, row_b: str) -> None:
    """Swap the data (non-label) cells of two rows in a labeled CSV, in place.

    This is exactly the corruption the negative control needs: the Pm-149
    fission-yield VALUES end up on the Nd-149 row and vice versa, while row
    labels stay put -- reproducing "the yield was wired to the wrong
    isotope" without touching column labels.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    idx_a = idx_b = None
    for i, r in enumerate(rows):
        if r and r[0] == row_a:
            idx_a = i
        if r and r[0] == row_b:
            idx_b = i
    assert idx_a is not None and idx_b is not None
    label_a, label_b = rows[idx_a][0], rows[idx_b][0]
    rows[idx_a], rows[idx_b] = rows[idx_b], rows[idx_a]
    rows[idx_a][0] = label_a
    rows[idx_b][0] = label_b
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def _scale_cell_in_place(path: Path, row_label: str, col_label: str, factor: float) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    col_idx = header.index(col_label)
    for r in rows[1:]:
        if r and r[0] == row_label:
            r[col_idx] = repr(float(r[col_idx]) * factor)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def check_source_level() -> list[str]:
    """(e) ComputeXeEquilibrium writes only I-135/Xe-135/Xe-135m; no
    "samarium" deck key in IO.cpp.

    MASTER SM mode 2 = transient (MASTER-3.0 manual p.3-46): MASTER supports
    an *equilibrium* Sm shortcut (SM mode) as an option, but the matching,
    physically complete mode is transient -- Sm-149 integrated through the
    full depletion matrix like every other isotope, with no special-cased
    equilibrium overwrite.  RASBERY has an equilibrium overwrite ONLY for the
    I-135/Xe-135/Xe-135m chain (ApplyXeEquilibrium); Sm-149 has no analogous
    function and no analogous deck key, i.e. RASBERY's Sm-149 handling is
    unconditionally transient, matching MASTER SM mode 2.
    """
    failures = []
    xsset_src = XSSET_CPP.read_text(encoding="utf-8-sig")

    m = re.search(
        r"static\s+void\s+ApplyXeEquilibrium\s*\([^)]*\)\s*\{(.*?)\n\}",
        xsset_src, re.S,
    )
    if not m:
        failures.append("ApplyXeEquilibrium not found in src/XSSet.cpp")
    else:
        body = m.group(1)
        writes = re.findall(r"iden\[\s*(i\w+)\s*\]\s*=", body)
        expected = {"iI135", "iXe135", "iXe135m"}
        if set(writes) != expected:
            failures.append(
                f"ApplyXeEquilibrium writes {sorted(set(writes))!r}, expected "
                f"exactly {sorted(expected)!r}"
            )
        if "iSm149" in body or "Sm149" in body or "sm149" in body.lower():
            failures.append("ApplyXeEquilibrium body references Sm-149 -- it must not")

    io_src = IO_CPP.read_text(encoding="utf-8-sig")
    if re.search(r"samarium", io_src, re.I):
        failures.append('src/IO.cpp contains a "samarium" reference -- no deck key expected')

    return failures


def main() -> int:
    failures: list[str] = []

    if not DEP_TRANS.exists():
        return _fatal(f"missing {DEP_TRANS}")
    if not DEP_DECAY.exists():
        return _fatal(f"missing {DEP_DECAY}")
    if not XSSET_CPP.exists():
        return _fatal(f"missing {XSSET_CPP}")
    if not IO_CPP.exists():
        return _fatal(f"missing {IO_CPP}")

    _, trans = read_labeled_matrix(DEP_TRANS)
    _, decay = read_labeled_matrix(DEP_DECAY)

    failures += check_trans_topology(trans)
    failures += check_decay_chain(decay)
    failures += check_bateman_timing()
    failures += check_negative_controls()
    failures += check_source_level()

    if failures:
        print("SM-149 chain contract: FAIL")
        for f in failures:
            print("  - " + f)
        return 1

    print(
        "SM-149 chain contract: PASS "
        "(dep_trans.csv topology, dep_decay.csv rates, Bateman timing, "
        "2 negative controls, source-level Xe/Sm boundary all verified)"
    )
    return 0


def _fatal(msg: str) -> int:
    print("SM-149 chain contract: FAIL")
    print("  - " + msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
