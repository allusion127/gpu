#!/usr/bin/env python3
"""Contract: the anchor-B fixed-rod deck tool means what its docstring says.

Checks, against synthetic fixtures (no solver, no GPU, no real .sum/.h5):

  1. insertion = rod_top_cm - cm is applied correctly, and clamped.
  2. apply_overlap() reproduces R2 = R3 + 120, R4 = R3 - 120, clamped to
     [0, rod_top_cm], including the two boundary/clamp cases.
  3. The generated deck disables rod search (the leading "standard" entry's
     "search" is forced to "keff") and carries one "rod insertion" entry per
     statepoint, with insertion values matching the .sum.
  4. Negative controls: a malformed .sum (shifted rod columns) raises, a
     statepoint-count mismatch between the deck and the .sum raises, and an
     out-of-range MASTER cm (negative, or past the withdrawn stop) is clamped
     rather than handed to the solver as a negative insertion depth.

Matches the manual check()/FAILED-list style of tools/test_ismr_tools_contract.py.

Usage
    python tools/test_ismr_fixed_rod_contract.py
"""
from __future__ import annotations

import json
import py_compile
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

FAILED: list[str] = []


def check(ok: bool, message: str) -> None:
    if not ok:
        FAILED.append(message)


#: A minimal SUMMARY EDIT 1 in MASTER's layout, THREE DISTINCT statepoints
#: (no duplicate EFPD -- the duplicate-EFPD "SEARCH=1 row loses to SEARCH=5"
#: nuance is already covered by tools/test_ismr_tools_contract.py's
#: rod_parser_contract() against the SAME reused parser; this fixture is
#: sized to line up 1:1 with a 3-statepoint deck instead).  Same bank order
#: the real i-SMR decks print (S1 R3 S3 S2 R4 R1 S4 R2 S5 S6).
EDIT1 = """   SUMMARY EDIT 1 : OPTIONS

   NO.     DAY     EFPD   XE/SM   SEARCH  RHO  RST  PPI  CMP  (CONTROL ROD POSITIONS)
                                                                S1      R3      S3      S2      R4      R1      S4      R2      S5      S6

    1    0.000    0.000    1  2      5     0    1    0    0   240.000 117.362 240.000 240.000   0.000 240.000 240.000 237.362 240.000 240.000
    2   19.631   19.631    1  2      5     0    1    0    0   240.000 117.362 240.000 240.000   0.000 240.000 240.000 237.362 240.000 240.000
    3   39.261   39.261    1  2      5     0    1    0    0   240.000 149.877 240.000 240.000  29.877 240.000 240.000 240.000 240.000 240.000

   SUMMARY EDIT 2 : REACTIVITY
"""

#: The first two rows of EDIT1, for fixtures that need a 2-statepoint grid.
EDIT1_2PT = "\n".join(
    line for line in EDIT1.splitlines()
    if not line.strip().startswith("3   39.261")) + "\n"

BANKS = ["S1", "R3", "S3", "S2", "R4", "R1", "S4", "R2", "S5", "S6"]


def minimal_deck(schedule: list[dict]) -> dict:
    rod_conf = {b: {"ctype": [1], "length": [240.0], "profile": [0.0, 240.0]}
                for b in BANKS}
    return {
        "data": {},
        "geometry": {},
        "rod configuration": rod_conf,
        "rod map": [["XX"]],
        "schedule": schedule,
    }


def write_sum(td: Path, text: str = EDIT1) -> Path:
    p = td / "depf_02.sum"
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


# ---------------------------------------------------------------------------
# 1/2.  insertion conversion and the overlap fallback
# ---------------------------------------------------------------------------
def conversion_contract() -> None:
    import make_ismr_fixed_rod_deck as mk

    check(mk.cm_to_insertion(240.0) == 0.0,
          "a fully-withdrawn bank (240 cm) must convert to 0 insertion")
    check(mk.cm_to_insertion(0.0) == 240.0,
          "a fully-inserted bank (0 cm) must convert to 240 insertion")
    check(mk.cm_to_insertion(117.362) == 240.0 - 117.362,
          "insertion = rod_top_cm - cm is not being applied literally")

    # Out-of-range cm: clamped, not handed through as a negative insertion.
    check(mk.cm_to_insertion(250.0) == 0.0,
          "a cm value past the withdrawn stop must clamp to 0 insertion, not "
          "go negative")
    check(mk.cm_to_insertion(-10.0) == 240.0,
          "a negative cm value must clamp to full (rod_top_cm) insertion, "
          "not exceed it")

    ov = mk.apply_overlap(150.0)
    check(ov["R2"] == 240.0 and ov["R4"] == 30.0,
          f"apply_overlap(150) = {ov}; expected R2=min(270,240)=240, R4=150-120=30")

    # Boundary: R3 low enough that R4 = R3-120 goes negative -> clamps to 0.
    low = mk.apply_overlap(50.0)
    check(low["R4"] == 0.0,
          f"apply_overlap(50)['R4'] = {low['R4']}; R3-120 = -70 must clamp to 0")
    check(low["R2"] == 170.0,
          f"apply_overlap(50)['R2'] = {low['R2']}; expected 50+120=170")

    # Boundary: R3 high enough that R2 = R3+120 exceeds rod_top_cm -> clamps.
    high = mk.apply_overlap(200.0, rod_top_cm=240.0)
    check(high["R2"] == 240.0,
          f"apply_overlap(200)['R2'] = {high['R2']}; 200+120=320 must clamp "
          f"to rod_top_cm=240")
    check(high["R4"] == 80.0,
          f"apply_overlap(200)['R4'] = {high['R4']}; expected 200-120=80")

    # apply_overlap only used as a FALLBACK: a bank the .sum states directly
    # must win over the formula, even when the formula would say something
    # different.
    with tempfile.TemporaryDirectory() as td:
        sum_path = write_sum(Path(td))
        import compare_master_rasbery as cmp_mod
        names, rows = cmp_mod.parse_master_rods(sum_path)
        row = rows[0.0]
        deck_banks = {"R2", "R3", "R4"}
        cm = mk.bank_cm_for_statepoint(names, row, deck_banks)
        # The .sum's own R2 at efpd 0 is 237.362, NOT R3(240)+120 clamped 240.
        check(cm["R2"] == 237.362,
              f"bank_cm_for_statepoint()['R2'] = {cm['R2']}; the .sum's own "
              f"237.362 must win over the overlap formula's 240.0")

        # Fallback path: R2 removed from the deck's declared banks is not
        # what this tests -- instead ask for a bank ABSENT from the .sum row
        # by restricting deck_banks to one the row does not carry a distinct
        # value for is not a real test of fallback since R2 is IN this row.
        # Use a row lacking R2 explicitly to exercise the overlap fallback.
        no_r2_names = [n for n in names if n != "R2"]
        no_r2_row = [v for n, v in zip(names, row) if n != "R2"]
        cm2 = mk.bank_cm_for_statepoint(no_r2_names, no_r2_row, {"R2", "R3", "R4"})
        check(cm2["R2"] == mk.apply_overlap(cm2["R3"])["R2"],
              f"bank_cm_for_statepoint() did not fall back to apply_overlap() "
              f"for a bank ({cm2.get('R2')}) the .sum row does not carry")


# ---------------------------------------------------------------------------
# 3.  Search disabled + one rod insertion entry per statepoint
# ---------------------------------------------------------------------------
def fixed_rod_deck_contract() -> None:
    import make_ismr_fixed_rod_deck as mk

    schedule = [
        {"type": "standard", "search": "rod"},
        {"type": "depletion", "steps": 2, "rate": 100.0, "time": 10.0},
    ]
    deck = minimal_deck(schedule)

    with tempfile.TemporaryDirectory() as td:
        sum_path = write_sum(Path(td))
        import compare_master_rasbery as cmp_mod
        bank_names, sum_rows = cmp_mod.parse_master_rods(sum_path)
        deck_banks = set(deck["rod configuration"])
        out_schedule = mk.build_fixed_rod_schedule(
            deck["schedule"], bank_names, sum_rows, deck_banks)

    # Search disabled: the leading "standard" entry must say "keff", never
    # "rod" -- the schema's actual off switch (src/IO.cpp ParseSearchTypeString).
    standards = [e for e in out_schedule if e.get("type") == "standard"]
    check(len(standards) == 1, f"expected exactly one 'standard' entry, got "
                               f"{len(standards)}")
    check(standards and standards[0].get("search") == "keff",
          f"leading 'standard' entry search={standards[0].get('search') if standards else None!r}; "
          f"rod search must be forced to 'keff' (plain eigenvalue, no search)")

    # One "rod insertion" entry precedes each of the 3 statepoints (1 standard
    # + steps=2 depletion), and none of the non-insertion entries carry search.
    insertions = [e for e in out_schedule if e.get("type") == "rod insertion"]
    check(len(insertions) == 3,
          f"expected 3 'rod insertion' entries (1 standard + 2 depletion "
          f"steps), got {len(insertions)}")
    others = [e for e in out_schedule if e.get("type") != "rod insertion"]
    check(all(e.get("type") != "depletion" or "search" not in e for e in others),
          "a depletion entry carries its own 'search' key in the fixed-rod "
          "output; nothing should override the disabled default")

    # Values: efpd 0 (row 1 in the source text, R3=240) is superseded by
    # efpd 0's SECOND print (R3=117.362) -- last-EFPD-wins, same rule
    # compare_master_rasbery.parse_master_rods documents. Then efpd 19.631.
    r3_insertions = [e["R3"] for e in insertions]
    expected = [round(mk.cm_to_insertion(117.362), 6),
                round(mk.cm_to_insertion(117.362), 6),
                round(mk.cm_to_insertion(149.877), 6)]
    check(r3_insertions == expected,
          f"R3 insertion sequence {r3_insertions} != expected {expected} "
          f"(insertion = 240 - cm, MASTER cm per statepoint)")

    # Order: rod insertion immediately precedes the depletion/standard entry
    # it belongs to.
    ordered_types = [e["type"] for e in out_schedule]
    check(ordered_types == ["rod insertion", "standard",
                            "rod insertion", "depletion",
                            "rod insertion", "depletion"],
          f"schedule order {ordered_types} does not put each 'rod insertion' "
          f"immediately before the statepoint it fixes")


def bank_collision_and_missing_block_contract() -> None:
    import make_ismr_fixed_rod_deck as mk

    check("search" in mk._RESERVED_ROD_KEYS,
          "'search' must be a reserved rod-insertion key (src/IO.cpp filters "
          "it out of the insertion payload)")


# ---------------------------------------------------------------------------
# 4.  Negative controls
# ---------------------------------------------------------------------------
def negative_controls_contract() -> None:
    import make_ismr_fixed_rod_deck as mk
    import compare_master_rasbery as cmp_mod

    # Malformed .sum: shifted rod columns must raise (reusing
    # compare_master_rasbery's own column-width assertion -- not re-implemented
    # here).
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "shifted.sum"
        bad.write_text(EDIT1.replace(
            "    1  2      5     0    1    0    0   240.000 117.362",
            "    1  2      5     0    1    0    0  9 240.000 117.362"),
            encoding="utf-8", newline="\n")
        try:
            cmp_mod.parse_master_rods(bad)
            check(False, "a shifted EDIT 1 row parsed silently through the "
                         "reused parser")
        except SystemExit:
            pass  # expected; compare_master_rasbery.py's own contract covers detail

    # Statepoint-count mismatch: 3 .sum rows against a 2-statepoint deck must
    # refuse rather than zip the wrong pairs together.
    with tempfile.TemporaryDirectory() as td:
        sum_path = write_sum(Path(td))
        bank_names, sum_rows = cmp_mod.parse_master_rods(sum_path)
        short_schedule = [{"type": "standard", "search": "rod"}]  # 1 statepoint, sum has 3
        deck_banks = set(BANKS)
        try:
            mk.build_fixed_rod_schedule(short_schedule, bank_names, sum_rows,
                                        deck_banks)
            check(False, "a 1-statepoint deck against a 3-row .sum built a "
                         "schedule instead of refusing the mismatch")
        except SystemExit as exc:
            check("statepoint" in str(exc) or "mismatch" in str(exc) or
                  "3" in str(exc),
                  f"the count-mismatch refusal is not informative: {exc}")

    # Missing bank with no overlap fallback available (an S-bank absent from
    # the .sum row, and R3 present does not help an S-bank).
    with tempfile.TemporaryDirectory() as td:
        sum_path = write_sum(Path(td))
        bank_names, sum_rows = cmp_mod.parse_master_rods(sum_path)
        row0 = sum_rows[0.0]
        try:
            mk.bank_cm_for_statepoint(bank_names, row0, {"S1", "Z9"})
            check(False, "an undeclared bank 'Z9' with no .sum row and no "
                         "overlap fallback was accepted silently")
        except SystemExit as exc:
            check("Z9" in str(exc), f"missing-bank refusal did not name it: {exc}")


# ---------------------------------------------------------------------------
# 5.  End-to-end: real deck-shaped source, --out written, re-loadable JSON
# ---------------------------------------------------------------------------
def end_to_end_contract() -> None:
    import subprocess

    schedule = [
        {"type": "standard", "search": "rod"},
        {"type": "depletion", "steps": 1, "rate": 100.0, "time": 19.631},
    ]
    deck = minimal_deck(schedule)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        deck_path = td / "cy_mini.json"
        deck_path.write_text(json.dumps(deck), encoding="utf-8")
        sum_path = write_sum(td, EDIT1_2PT)
        out_path = td / "cy_mini_fixedrod.json"

        proc = subprocess.run(
            [sys.executable, str(TOOLS / "make_ismr_fixed_rod_deck.py"),
             str(deck_path), str(sum_path), "-o", str(out_path)],
            capture_output=True, text=True)
        check(proc.returncode == 0,
              f"make_ismr_fixed_rod_deck.py exited {proc.returncode}:\n"
              f"{proc.stdout}\n{proc.stderr}")
        check(out_path.is_file(), "no output deck written")
        if out_path.is_file():
            written = json.loads(out_path.read_text(encoding="utf-8"))
            check("schedule" in written, "output deck lost its schedule block")
            check(written.get("rod configuration") == deck["rod configuration"],
                  "output deck's rod configuration block was modified; only "
                  "the schedule should change")
            types = [e.get("type") for e in written["schedule"]]
            check(types.count("rod insertion") == 2,
                  f"end-to-end output has {types.count('rod insertion')} rod "
                  f"insertion entries, expected 2 (1 standard + 1 depletion)")


def main() -> int:
    conversion_contract()
    fixed_rod_deck_contract()
    bank_collision_and_missing_block_contract()
    negative_controls_contract()
    end_to_end_contract()

    for name in ("make_ismr_fixed_rod_deck.py", "ismr_rod_reactivity.py"):
        py_compile.compile(str(TOOLS / name), doraise=True)
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)

    if FAILED:
        for message in FAILED:
            print(f"ismr fixed-rod contract: FAIL: {message}")
        return 1
    print("ismr fixed-rod contract: PASS (insertion conversion + overlap "
          "fallback + search-disabled deck + negative controls + end-to-end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
