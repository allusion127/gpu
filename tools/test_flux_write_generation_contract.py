#!/usr/bin/env python3
"""Geometry::Phif write-generation contract (Rev.7.1, sync-elision prerequisite).

WHAT THIS PROTECTS.  The device outer segment uploads the flux at the top of
every outer -- 681 KiB on kngr_238, 34% of the segment's whole transfer budget
together with xsnf and dtil -- because the sweep's own download is CONDITIONAL:
it happens only when the device sweep finished, which the Wielandt warm-up and
every declined enqueue do not.  Skipping that upload safely needs one fact:
has a HOST writer touched Phif since the device last downloaded it.

WHY A GENERATION AND NOT A GREP LIST.  A counter every writer must remember to
bump is wrong the first time somebody forgets, and it fails silently -- the
elision skips an upload that was needed and the answer moves.  So the type
system does the enumeration: Phif() is const, PhifMutable() is the only door
that yields a writable pointer, and it bumps on the way through.  A new writer
does not compile until it says what it is.

THAT IS NOT A THEORETICAL ARGUMENT.  A hand grep for the writers found four
(the restart load, the fission-source sign flip, drive(), and the initial
fill).  Making Phif() const found a FIFTH -- XSSet::ResetFluxAndCurrents, which
writes through std::fill_n and matched none of the patterns a human would grep
for.  A generation built on the hand list would have been wrong from the first
commit, on the deck that resets the flux.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
problems: list[str] = []


def read(name: str) -> str:
    p = SRC / name
    if not p.is_file():
        problems.append(f"missing {name}")
        return ""
    return p.read_text(encoding="utf-8-sig")


GEOM = read("Geometry.h")

# --- 1. the door is the only way to a writable flux --------------------------
if "inline const double* Phif() const" not in GEOM:
    problems.append(
        "Geometry::Phif() is not const.  A non-const accessor hands every caller a "
        "writable pointer, the compiler stops enumerating the writers, and the "
        "generation goes back to being a promise")
if "double* PhifMutable()" not in GEOM:
    problems.append("Geometry has no PhifMutable(): there is no door that bumps")
_door = GEOM[GEOM.find("double* PhifMutable()"):][:400]
if "++_phif_generation" not in _door:
    problems.append(
        "PhifMutable() does not bump the generation, so the one door that is supposed "
        "to record a write does not record it")
if "fluxGeneration()" not in GEOM:
    problems.append("Geometry has no fluxGeneration() reader")
_gen = GEOM[GEOM.find("_phif_generation"):][:200]
if "= 1" not in _gen:
    problems.append(
        "the flux generation must START AT 1.  A consumer's 'never uploaded' sentinel "
        "is 0, and a generation that also started at 0 would compare equal to it and "
        "skip the very first upload instead of forcing it")

# --- 2. nobody casts the const away ------------------------------------------
for path in sorted(SRC.glob("*.h")) + sorted(SRC.glob("*.cpp")) + sorted(SRC.glob("*.cu")):
    text = path.read_text(encoding="utf-8-sig")
    for m in re.finditer(r"const_cast<\s*double\s*\*\s*>\s*\(([^)]*)\)", text):
        if "Phif" in m.group(1):
            problems.append(
                f"{path.name}: const_cast on Phif().  Casting the const away is a write "
                "that does not bump the generation, which is the exact failure the const "
                "accessor exists to prevent")

# --- 3. every writer goes through the door, and they are the known set -------
#
# The list is documentation, not the enforcement -- the compiler is the
# enforcement.  It is checked so that a writer DISAPPEARING is noticed too: a
# call site that stops writing the flux should stop bumping the generation, or
# the elision it feeds becomes permanently pessimistic.
EXPECTED = {
    "XSSet.cpp": 2,   # ResetFluxAndCurrents (fill_n) + the fission-source sign flip
    "IO.cpp": 1,      # the restart flux load
    "Driver.h": 6,    # drive x2 in the loops, drive x2 + enqueueDrive + finishDrive in the hooks
}
for name, want in EXPECTED.items():
    # comments mention the door by name; only CALLS count
    body = re.sub(chr(47) * 2 + "[^" + chr(10) + "]*", "", read(name))
    got = body.count("PhifMutable()")
    if got != want:
        problems.append(
            f"{name}: {got} PhifMutable() call sites, expected {want}.  A new one means a "
            "new flux writer -- confirm it really writes, then update this count; a lost "
            "one means an elision downstream will stop being told about a write")

if problems:
    for p in problems:
        print("flux write generation: FAIL " + p, file=sys.stderr)
    raise SystemExit(1)
total = sum(EXPECTED.values())
print(f"flux write generation: PASS ({total} writer sites, all through PhifMutable)")
