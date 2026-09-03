#!/usr/bin/env python3
"""The rotational-fold shuffle must move each source node exactly once.

WHY THIS EXISTS.  On 2026-09-04 the 90/270 cases of the `switch (rot)` in
`IO::ApplyShuffle` were reported as swapped relative to the MASTER convention,
on the evidence that i-SMR CY02's two rotationally equivalent cut targets --
quarter-map (0,1) rot 270 and (1,0) rot 180, both sourcing CY01 "3,3" -- receive
DISJOINT COMPLEMENTARY halves of the source (11.983 vs 16.260 GWd/t) where
`Reference_output/depf_02.sum` prints 16.385 at both F5 and E6.

That reading is wrong, and this contract exists so nobody re-derives it.  Under
the 90-degree fold (`symang=90, mirror:false, symdiv:true`) quarter-map entries
(0,c) and (c,0) are the two halves of ONE physical assembly, and disjoint
complementary halves are the DESIGN, not the bug: (0,c)'s visible nodes occupy
the source frame's high rows and (c,0)'s its high columns, so between them they
must carry the whole assembly exactly once.  MASTER prints one number per
physical assembly, so equal values at F5 and E6 are what a correct partition
predicts -- and 16.385 is precisely the source's own CY01 EOC full-assembly
average (`depf_01.sum`, H8), which only a partition reproduces.  Reading "the
same half" into both arms would instead copy one half twice and DISCARD the
other, losing half the burnup and half the number densities in the core.

The proposed swap does exactly that, for every arm pair in every deck in the
tree.  The invariant below is what makes that visible in one line, so it is
checked here rather than argued again.

WHAT THIS CHECKS.

1. SWITCH CONTRACT.  `switch (rot)` maps 90 to RotateAssemblyIndex, 270 to
   RotateAssemblyIndexInverse, 180 to the involution and 0 to the identity.

2. FOLD COVERAGE (the load-bearing one).  On a synthetic 3x3 quarter core,
   ndivxy=2, with per-node burnup `100*x + y`, the two arms (0,c) and (c,0) of
   one physical assembly must between them read every one of the source's n*n
   node slots EXACTLY ONCE -- checked for interior, row-arm-cut and
   col-arm-cut sources, for every arm rotation pair the decks use.  Their two
   half-averages must also average to the source's full-assembly average, which
   is the property MASTER's per-assembly edit is comparable against.
   Re-swapping cases 90 and 270 is the negative control: it must break coverage
   for every pair.

3. DECK CONTRACT.  Every shuffle entry of i-SMR_CY02/03/04.json is cross-checked
   against the `%LPD_SHF` card of the matching `Reference_input/depf_0N.inp`
   (MASTER letter = col + 5 over E,F,G,H,J; number = row + 5; rotation =
   code * 90).  This pins two things the C++ cannot state for itself: that
   `TryParseShuffleEntry` is right to bind the FIRST field of "i,j" to
   source_row -- 26 of the 33 entries are off-diagonal, and reading them the
   other way round contradicts the card in 7 of CY03's 11 alone -- and that
   each arm pair carries rotations differing by exactly 90 degrees, the
   relation fold coverage depends on.

USAGE
    tools/test_rotational_shuffle_contract.py
"""
from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IO_CPP = ROOT / "src" / "IO.cpp"
ISMR = ROOT / "test" / "7_i-SMR_Validation"

# --------------------------------------------------------------- index algebra


def visible_rows(row, n):
    return n // 2 if row == 0 else n


def visible_cols(col, n):
    return n // 2 if col == 0 else n


def node_x(a_col, l_col, n):
    x = a_col * n + l_col
    return x - n // 2 if a_col > 0 else x


def node_y(a_row, l_row, n):
    y = a_row * n + l_row
    return y - n // 2 if a_row > 0 else y


def rot_index(i, j, n):
    """RotateAssemblyIndex -- one 90-degree step."""
    return j, n - 1 - i


def rot_index_inv(i, j, n):
    """RotateAssemblyIndexInverse."""
    return n - 1 - j, i


def rho_shipped(rot, i, j, n):
    """Target node (i,j) takes source node rho(i,j) -- the shipped handedness."""
    if rot == 90:
        return rot_index(i, j, n)
    if rot == 180:
        return n - 1 - i, n - 1 - j
    if rot == 270:
        return rot_index_inv(i, j, n)
    return i, j


def rho_swapped(rot, i, j, n):
    """Negative control: cases 90 and 270 exchanged."""
    if rot == 90:
        return rot_index_inv(i, j, n)
    if rot == 270:
        return rot_index(i, j, n)
    return rho_shipped(rot, i, j, n)


def offsets(a_row, a_col, n):
    rc, cc = visible_rows(a_row, n), visible_cols(a_col, n)
    return rc, cc, (n - rc if a_row == 0 else 0), (n - cc if a_col == 0 else 0)


# ------------------------------------------------------------ source buffer ---


def gather(buf, burn, a_row, a_col, rot_steps, n):
    """Mirror of ApplyShuffle's `gather` lambda; slots already written are kept."""
    rc, cc, r_off, c_off = offsets(a_row, a_col, n)
    for li in range(rc):
        for lj in range(cc):
            val = burn(node_x(a_col, lj, n), node_y(a_row, li, n))
            fi, fj = r_off + li, c_off + lj
            for _ in range(rot_steps):
                fi, fj = rot_index_inv(fi, fj, n)
            buf.setdefault((fi, fj), val)


def source_buffer(s_row, s_col, burn, n):
    """The source assembly's full n x n frame, completed by the fold."""
    buf = {}
    gather(buf, burn, s_row, s_col, 0, n)
    if s_row == 0 or s_col == 0:
        if s_row == 0 and s_col == 0:
            for k in (1, 2, 3):
                gather(buf, burn, 0, 0, k, n)
        elif s_row == 0:
            gather(buf, burn, s_col, 0, 1, n)
        else:
            gather(buf, burn, 0, s_row, 3, n)
    return buf


def arm_reads(t_row, t_col, rot, rho, n):
    """The source slots the visible nodes of target (t_row,t_col) read."""
    rc, cc, r_off, c_off = offsets(t_row, t_col, n)
    return [rho(rot, r_off + li, c_off + lj, n)
            for li in range(rc) for lj in range(cc)]


# ------------------------------------------------------- 1. switch contract ---

SWITCH = re.compile(
    r"switch\s*\(\s*rot\s*\)\s*\{\s*"
    r"case\s+90\s*:\s*src_full_i\s*=\s*(?P<i90>[^;]+);\s*src_full_j\s*=\s*(?P<j90>[^;]+);\s*break;\s*"
    r"case\s+180\s*:\s*src_full_i\s*=\s*(?P<i180>[^;]+);\s*src_full_j\s*=\s*(?P<j180>[^;]+);\s*break;\s*"
    r"case\s+270\s*:\s*src_full_i\s*=\s*(?P<i270>[^;]+);\s*src_full_j\s*=\s*(?P<j270>[^;]+);\s*break;\s*"
    r"default\s*:\s*src_full_i\s*=\s*(?P<idef>[^;]+);\s*src_full_j\s*=\s*(?P<jdef>[^;]+);\s*break;",
    re.S,
)

I, J = "target_i[ii]", "target_j[jj]"
NI, NJ = f"n - 1 - {I}", f"n - 1 - {J}"
# 90 must be RotateAssemblyIndex(i,j)        = (j, n-1-i);
# 270 must be RotateAssemblyIndexInverse(i,j) = (n-1-j, i).
EXPECTED = {"i90": J, "j90": NI,
            "i180": NI, "j180": NJ,
            "i270": NJ, "j270": I,
            "idef": I, "jdef": J}

# TryParseShuffleEntry must bind the FIRST field of "i,j" to source_row.
FIELD_ORDER = re.compile(
    r"const int si\s*=\s*std::stoi\(ij_str\.substr\(0, comma\)\);\s*"
    r"const int sj\s*=\s*std::stoi\(ij_str\.substr\(comma \+ 1\)\);", re.S)
SPEC_ORDER = re.compile(r"out\s*=\s*\{tgt_row, tgt_col, si, sj, cyc, rot\};")


def check_source(text):
    bad = []
    m = SWITCH.search(text)
    if not m:
        return ["the `switch (rot)` in IO::ApplyShuffle no longer has the four-case "
                "shape this contract reads -- it cannot be checked"]
    for key, want in EXPECTED.items():
        got = " ".join(m.group(key).split())
        if got != want:
            bad.append(f"switch case body {key}: expected `{want}`, found `{got}`")
    if not FIELD_ORDER.search(text) or not SPEC_ORDER.search(text):
        bad.append("TryParseShuffleEntry no longer binds \"i,j\" as (si=row, sj=col) "
                   "into ShuffleSpec -- the MASTER decks require row first "
                   "(see the deck contract below)")
    return bad


def swap_switch_cases(text):
    """`text` with the case 90 and case 270 bodies exchanged (negative control)."""
    m = SWITCH.search(text)
    if not m:
        return None
    body = m.group(0)
    for a, b in (("i90", "i270"), ("j90", "j270")):
        f = "src_full_i" if a.startswith("i") else "src_full_j"
        body = (body.replace(f"{f} = {m.group(a)};", "\x00", 1)
                    .replace(f"{f} = {m.group(b)};", f"{f} = {m.group(a)};", 1)
                    .replace("\x00", f"{f} = {m.group(b)};"))
    return text[:m.start()] + body + text[m.end():]


# ------------------------------------------------------- 2. fold coverage ---

# (Ra, Rb) for arms (0,c) and (c,0). Every pair the i-SMR decks use has
# Ra == Rb + 90 (mod 360); the deck contract below re-derives that from the cards.
ARM_PAIRS = [(90, 0), (180, 90), (270, 180), (0, 270)]
SOURCES = [(1, 1, "interior source"),
           (0, 1, "cut source on the row arm"),
           (1, 0, "cut source on the col arm"),
           (0, 0, "centre source")]


def coverage_failures(rho):
    bad, n, c = [], 2, 1
    def burn(x, y):
        return 100 * x + y

    for s_row, s_col, what in SOURCES:
        buf = source_buffer(s_row, s_col, burn, n)
        if len(buf) != n * n:
            bad.append(f"{what}: the fold filled {len(buf)} of {n * n} slots")
            continue
        for ra, rb in ARM_PAIRS:
            a = arm_reads(0, c, ra, rho, n)
            b = arm_reads(c, 0, rb, rho, n)
            union = a + b
            if sorted(union) != sorted(set(union)) or len(set(union)) != n * n:
                dup = sorted({s for s in union if union.count(s) > 1})
                lost = sorted(set(buf) - set(union))
                bad.append(f"{what}, arms rot {ra}/{rb}: the two halves of one "
                           f"assembly do not partition the source -- duplicated "
                           f"{dup}, discarded {lost}")
                continue
            # The two half-averages must average to the source's full average,
            # which is the number MASTER's per-assembly edit prints.
            half = [sum(buf[s] for s in arm) / len(arm) for arm in (a, b)]
            full = sum(buf.values()) / len(buf)
            if abs((half[0] + half[1]) / 2 - full) > 1e-9:
                bad.append(f"{what}, arms rot {ra}/{rb}: half-averages "
                           f"{half[0]:.9f}/{half[1]:.9f} do not average to the "
                           f"source's full-assembly average {full:.9f}")
    return bad


# --------------------------------------------------------- 3. deck contract ---

COL = {"E": 0, "F": 1, "G": 2, "H": 3, "J": 4}
CARD = re.compile(r"%LPD_SHF(.*?)(?=^\s*%)", re.S | re.M)
FUEL = re.compile(r"^(\d+)\s+([EFGHJ])\s*(\d)\s+(\d)$")


def parse_card(inp):
    """The %LPD_SHF quarter map as rows of (source_row, source_col, rot) or None."""
    m = CARD.search(inp.read_text(encoding="utf-8", errors="replace"))
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        line = line.split("#")[0].strip().rstrip(",")
        if not line:
            continue
        row = []
        for tok in line.split(","):
            f = FUEL.match(" ".join(tok.split()))
            # A fresh assembly ("F H1 0") carries no source position.
            row.append((int(f.group(3)) - 5, COL[f.group(2)], int(f.group(4)) * 90)
                       if f else None)
        rows.append(row)
    return rows


def deck_failures():
    bad, checked, discriminating, pairs = [], 0, 0, 0
    missing = []
    for cyc in ("02", "03", "04"):
        inp = ISMR / "Reference_input" / f"depf_{cyc}.inp"
        js = ISMR / f"i-SMR_CY{cyc}.json"
        # The whole test/ tree is gitignored, so a fresh clone has neither the
        # deck nor the MASTER card. Report that rather than assume it clean.
        if not inp.is_file() or not js.is_file():
            missing.append(cyc)
            continue
        card = parse_card(inp)
        if card is None:
            bad.append(f"depf_{cyc}.inp: no %LPD_SHF card found")
            continue
        core = json.loads(js.read_text(encoding="utf-8"))["core"]
        got = {}
        for r, crow in enumerate(card):
            for c, want in enumerate(crow):
                entry = core[r][c]
                if want is None:
                    if "/" in entry:
                        bad.append(f"CY{cyc} ({r},{c}): card has a FRESH assembly, "
                                   f"deck has shuffle entry {entry!r}")
                    continue
                ij, _cycle, rot = entry.split("/")
                s_row, s_col = (int(v) for v in ij.split(","))
                checked += 1
                if want[0] != want[1]:
                    discriminating += 1
                if (s_row, s_col, int(rot)) != want:
                    bad.append(f"CY{cyc} ({r},{c}): deck says source "
                               f"({s_row},{s_col}) rot {rot}, %LPD_SHF says "
                               f"({want[0]},{want[1]}) rot {want[2]}")
                got[(r, c)] = int(rot)
        # Arm pairs (0,c) and (c,0) must differ by exactly 90 degrees.
        for c in range(1, len(card)):
            if (0, c) in got and (c, 0) in got:
                pairs += 1
                if (got[(0, c)] - got[(c, 0)]) % 360 != 90:
                    bad.append(f"CY{cyc} arm pair (0,{c})/({c},0): rotations "
                               f"{got[(0, c)]}/{got[(c, 0)]} do not differ by 90 "
                               f"degrees, so they cannot partition one assembly")
    if checked and discriminating < 8:
        bad.append(f"only {discriminating} off-diagonal entries -- too few to pin "
                   f"the row/col field order")
    return bad, checked, discriminating, pairs, missing


def main():
    failures = []
    text = IO_CPP.read_text(encoding="utf-8", errors="replace")

    failures += check_source(text)

    swapped = swap_switch_cases(text)
    if swapped is None or swapped == text:
        failures.append("negative control could not be built from the switch")
    elif not check_source(swapped):
        failures.append("negative control ACCEPTED -- re-swapping cases 90 and 270 "
                        "in src/IO.cpp still passes the switch contract")

    failures += coverage_failures(rho_shipped)
    if not coverage_failures(rho_swapped):
        failures.append("negative control ACCEPTED -- exchanging cases 90 and 270 "
                        "still partitions the source, so this mirror has drifted "
                        "from src/IO.cpp")

    deck_bad, checked, discriminating, pairs, missing = deck_failures()
    failures += deck_bad

    if failures:
        print("rotational shuffle contract: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    deck = (f"{checked} deck entries vs %LPD_SHF ({discriminating} off-diagonal) "
            f"and {pairs} arm pairs")
    if missing:
        deck += f"; CY{'/CY'.join(missing)} SKIPPED (test/ is gitignored)"
    print("rotational shuffle contract: PASS (switch + field order, %d arm-rotation "
          "pairs x %d source kinds partition the source, %s, 2 negative controls)"
          % (len(ARM_PAIRS), len(SOURCES), deck))
    return 0


if __name__ == "__main__":
    sys.exit(main())
