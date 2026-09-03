#!/usr/bin/env python3
"""Read a MASTER MAS_RST restart file, and the MAS_OUT edit maps beside it.

WHY THIS EXISTS.  tools/fit_tf_table.py needs MASTER's OWN (LPD, burnup) ->
dT = Tfuel - Tmod samples to regress the ABB-CE fuel-temperature table
(`isolth = 12`) that include/Database/tf_ce.csv must hold.  The obvious source
is the restart file, whose manual section 4.2 advertises node-wise Moderator
Temperature / Moderator Density / Fuel Temperature.  IT IS NOT THERE -- see
"WHAT THE RESTART DOES NOT CONTAIN" below.  So this module reads BOTH: the
restart, for the true 3-D burnup it really does carry, and the MAS_OUT edit
maps, which are the only place this MASTER build prints a fuel temperature.


================================================================================
THE ACTUAL MAS_RST LAYOUT  (MASTER 4.0m4, master4.0m4_r1.exe, APR1400 KNGR deck)
================================================================================

Fortran UNFORMATTED SEQUENTIAL, little-endian, with 4-byte record-length
markers on BOTH sides of every record (head == tail; verified for all 175 325
records of every statepoint file).  Reals are REAL*8 (float64) throughout --
NOT real*4.  Integers and A4 character data are 4 bytes.

The manual's "Record 1 .. Record 25" numbering is LOGICAL, not physical: the
writer emits many small Fortran records per logical record.  For the KNGR deck
(nx = ny = 19, nz = 27, nzf = 25 fuel planes, nn = 4 nodes per FA per xy-plane
because ndivxy = 2, npin = 16 rods per FA side, 241 fuel assemblies) the
physical file is, in order:

    [  0]  115 records x 256 bytes   -- the MAS_INP deck echoed verbatim, one
                                        input LINE per record, blank padded.
                                        This is where nx/ny/nz, the axial mesh,
                                        %GEN_PIN nfrod and the %LPD_BCH batch
                                        map are recovered from.
    [115]  241 records x   4 bytes   -- (A4) batch id of every FUEL assembly,
                                        in ROW-MAJOR order over the %LPD_BCH
                                        map, reflector positions skipped.
                                        This fixes the FA <-> (ix, iy) mapping.
    [356]    1 record  x   8 bytes   -- 2 x (I)
    [357]    1 record  x  56 bytes   -- 7 x (R8) core scalars:
                                        [tin_C, mflow_kg_s, pload, boron_ppm,
                                         0, 0, 0]
    [358]    1 record  x  12 bytes   -- 3 x (I)
    [359]  241 FA BLOCKS x 726 records  -- see below; 359 + 241*726 = 175 325.

Each FA BLOCK (726 records), FA order identical to the batch-id list above:

    +0       1 record  x   4 bytes   -- (I) FA sequence marker
    then 25 PLANE GROUPS of 29 records, plane 1 = BOTTOM fuel plane
    (global axial node 2; nodes 1 and 27 are the axial reflectors and are
    NOT written):

        4 x [ 56 bytes = 7 x (R8) ]  -- one per xy sub-node (nn = 4), the
                                        2 x 2 quadrants of the FA:
                                        [0] node burnup            [MWd/kgU]
                                        [1..6] surface burnup, nd*2 = 6
          each followed by
            [ 264 bytes = 33 x (R8) ] -- nuclide number densities, nu = 33
            [  40 bytes =  5 x (R8) ] -- ALWAYS IDENTICALLY ZERO in every
                                         statepoint file of this run
        1 x [   4 bytes = (I) ]      -- npin = 16
        16 x [ 128 bytes = 16 x (R8) ] -- the 2-D PIN BURNUP map of the plane,
                                         one 16-wide row per record.

    so a plane group is 4*(1+1+1) + 1 + 16 = 29 records, and
    1 + 25*29 = 726.


WHAT THE RESTART DOES NOT CONTAIN
---------------------------------
Records 23 / 24 / 25 of manual section 4.2 -- Moderator Temperature (C),
Moderator Density (g/cm3) and Fuel Temperature (C), each nt*(R) -- ARE NOT
WRITTEN by this build.  This is not a parsing failure and it is not an artefact
of one statepoint:

  * There is no record of length nt*4 (38 988 B) or nt*8 (77 976 B) anywhere.
  * Sweeping EVERY float64 in the file, only THREE fall in the physical fuel /
    moderator temperature window [280, 1500] -- and all three are in the single
    core-scalar record at index 357 (tin = 290.6 C, mflow = 3480, boron =
    3773.48 at BOC).  Checked on the 0, 360 and 1080 EFPD files alike.
  * The only temperature-shaped per-node slot, the 5 x (R8) record in every
    plane group, is identically zero at every node of every statepoint,
    including deep-burnup ones where every other field is populated.

There is likewise NO node power in the file: the manual's Part-3 "Axial
Node-wise Power Distribution" and "2-D Pin Power Distribution" are absent too;
the single 16x16 per-plane array present is BURNUP (it is zero in the 0 EFPD
file, where pin POWER is emphatically not zero, and it tracks the node burnup
of the same plane elsewhere).

Consequence, recorded here so nobody re-derives it: the fuel-temperature
regression CANNOT be done per 3-D node from the restart.  The dT samples come
from the MAS_OUT `$FB2D` maps (axially averaged Tm / Tf per assembly), and the
restart's role is (a) to prove the above and (b) to supply the true 3-D burnup
that CHECKS the axially-averaged burnup the fit consumes.


================================================================================
THE MAS_OUT EDIT MAPS
================================================================================
`read_map_blocks` reads MASTER's `$TAG_n` map edits.  Each is a `Y\\X` header
naming the column labels, then one row per y with `nline` fixed-width,
RIGHT-ALIGNED numeric lines under it.  Columns are recovered from the numeric
RIGHT EDGES rather than by slicing at a guessed width, because the reflector
gutters are blank and every Fortran F-format in these blocks (F9.2, F9.5,
F8.4) is right-aligned on the same edge as its column label.

`nline` counts the lines that carry DECIMAL numbers, which is not always the
number of lines MASTER's legend promises: the trailing "PIN LOCATION (I,J)"
line of $P2D / $B2D is a pair of integers and is skipped.

    $FB2D_n  19x19 incl. radial reflector, nline = 3: Tm(C), Tf(C), Dm(g/cc)
    $P2D_n   17x17 fuel only,              nline = 2: FA power, max rod power
    $B2D_n   17x17 fuel only,              nline = 2: FA burnup, max rod burnup

USAGE
    tools/parse_master_rst.py <MAS_RST>
    tools/parse_master_rst.py <MAS_RST> --validate <MAS_OUT>
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Fortran unformatted sequential
# ---------------------------------------------------------------------------
def read_records(path: Path) -> list[bytes]:
    """Every record payload, in file order.  Refuses a file whose markers lie."""
    blob = path.read_bytes()
    out: list[bytes] = []
    off = 0
    while off + 8 <= len(blob):
        (n,) = struct.unpack_from("<i", blob, off)
        if n < 0 or off + 8 + n > len(blob):
            raise ValueError(f"{path}: record {len(out)+1} claims {n} bytes at "
                             f"offset {off}; this is not little-endian Fortran "
                             "unformatted sequential")
        (tail,) = struct.unpack_from("<i", blob, off + 4 + n)
        if tail != n:
            raise ValueError(f"{path}: record {len(out)+1} head {n} != tail {tail}")
        out.append(blob[off + 4:off + 4 + n])
        off += 8 + n
    if off != len(blob):
        raise ValueError(f"{path}: {len(blob)-off} trailing bytes after the last record")
    return out


def _r8(payload: bytes) -> list[float]:
    return list(struct.unpack("<%dd" % (len(payload) // 8), payload))


def _a(payload: bytes) -> str:
    return payload.decode("latin-1").rstrip()


# ---------------------------------------------------------------------------
# The deck echo -- the geometry this parse needs, taken from the file itself
# rather than from a MAS_INP that may have drifted away from the run.
# ---------------------------------------------------------------------------
class Deck:
    """The %-cards this parser needs, recovered from the restart's deck echo."""

    def __init__(self, lines: list[str]):
        self.card: dict[str, list[str]] = {}
        key = None
        for raw in lines:
            text = raw.split("#")[0].rstrip()
            if not text.strip():
                continue
            if text.lstrip().startswith("%"):
                key = text.strip().split()[0]
                self.card[key] = []
            elif key is not None:
                self.card[key].append(text)

        dim = self.card["%GEN_DIM"][0].split()
        self.nx, self.ny, self.nz = int(dim[0]), int(dim[1]), int(dim[2])
        sub = self.card["%GEN_DIM"][1].split()
        self.ndivxy, self.ndivz = int(sub[3]), int(sub[4])
        self.nn = self.ndivxy * self.ndivxy

        pin = self.card["%GEN_PIN"][0].split()
        self.npin, self.nfrod = int(pin[2]), int(pin[3])

        self.power_mw = float(self.card["%GEN_THD"][0].split()[0])

        geo = self.card["%GEN_GEO"]
        self.wide = float(geo[0].split()[0])
        self.height = float(geo[0].split()[1])
        mesh: list[float] = []
        for row in geo[1:]:
            for token in row.split():
                if "*" in token:
                    count, value = token.split("*")
                    mesh.extend([float(value)] * int(count))
                else:
                    mesh.append(float(token))
        self.mesh = mesh
        if len(mesh) != self.nz:
            raise ValueError(f"%GEN_GEO gave {len(mesh)} axial meshes, nz = {self.nz}")

        cdn = self.card["%GEN_CDN"]
        self.extx = cdn[0].split()
        self.exty = cdn[1].split()

        rows = [r.split() for r in self.card["%LPD_BCH"]]
        if len(rows) != self.ny or any(len(r) != self.nx for r in rows):
            raise ValueError("%LPD_BCH is not nx*ny")
        self.batch_map = rows
        # A FUEL assembly is one whose batch id is neither 'o' (outside) nor a
        # reflector ('R*' in this deck).  Row-major -- the order MASTER writes.
        self.fuel: list[tuple[int, int, str]] = [
            (ix, iy, rows[iy][ix])
            for iy in range(self.ny)
            for ix in range(self.nx)
            if rows[iy][ix] != "o" and not rows[iy][ix].startswith("R")
        ]

    @property
    def nzf(self) -> int:
        """Fuel planes: the axial nodes that are not the two axial reflectors."""
        return self.nz - 2

    @property
    def hz(self) -> float:
        """The fuel-plane height.  Refuses a non-uniform fuel mesh."""
        fuel = self.mesh[1:self.nz - 1]
        if max(fuel) - min(fuel) > 1e-9:
            raise ValueError("the fuel axial mesh is not uniform; hz is ambiguous")
        return fuel[0]

    @property
    def fuel_height(self) -> float:
        return self.hz * self.nzf

    def label(self, ix: int, iy: int) -> str:
        return f"{self.extx[ix]}{self.exty[iy]}"


# ---------------------------------------------------------------------------
# The restart
# ---------------------------------------------------------------------------
class Restart:
    """One MAS_RST statepoint, parsed to the layout documented at module top."""

    def __init__(self, path: Path):
        self.path = path
        recs = read_records(path)

        n_echo = 0
        while n_echo < len(recs) and len(recs[n_echo]) == 256:
            n_echo += 1
        if n_echo < 20:
            raise ValueError(f"{path}: only {n_echo} deck-echo records; this is not "
                             "a MASTER 4.0 restart")
        self.deck = Deck([_a(r) for r in recs[:n_echo]])
        d = self.deck

        n_fa = len(d.fuel)
        ids = [_a(r) for r in recs[n_echo:n_echo + n_fa]]
        expect = [b for _, _, b in d.fuel]
        if ids != expect:
            bad = next(i for i, (a, b) in enumerate(zip(ids, expect)) if a != b)
            raise ValueError(f"{path}: the restart's batch-id sequence does not match "
                             f"%LPD_BCH row-major (first mismatch at index {bad}: "
                             f"{ids[bad]!r} vs {expect[bad]!r})")
        self.batch_ids = ids

        base = n_echo + n_fa
        scalars = _r8(recs[base + 1])
        self.tin_c, self.mflow, self.pload, self.boron_ppm = scalars[:4]

        per_plane = d.nn * 3 + 1 + d.npin
        per_fa = 1 + d.nzf * per_plane
        start = base + 3
        if len(recs) != start + n_fa * per_fa:
            raise ValueError(
                f"{path}: {len(recs)} records, but the documented layout wants "
                f"{start + n_fa * per_fa} ({start} header + {n_fa} FA x {per_fa}).  "
                "The layout in this module's docstring no longer describes the file.")

        # burnup[fa][plane][quadrant]; pin_burnup[fa][plane][row][col]
        self.burnup: list[list[list[float]]] = []
        self.pin_burnup: list[list[list[list[float]]]] = []
        self.temperature_slot_nonzero = 0
        at = start
        for _fa in range(n_fa):
            at += 1                                          # FA sequence marker
            fa_bu: list[list[float]] = []
            fa_pin: list[list[list[float]]] = []
            for _plane in range(d.nzf):
                quad: list[float] = []
                for _node in range(d.nn):
                    quad.append(_r8(recs[at])[0])            # node burnup
                    at += 2                                  # + nuclide densities
                    if any(x != 0.0 for x in _r8(recs[at])):
                        self.temperature_slot_nonzero += 1
                    at += 1                                  # the all-zero 5*(R8)
                fa_bu.append(quad)
                at += 1                                      # npin marker
                fa_pin.append([_r8(recs[at + r]) for r in range(d.npin)])
                at += d.npin
            self.burnup.append(fa_bu)
            self.pin_burnup.append(fa_pin)

    def fa_burnup_axially_averaged(self) -> dict[str, float]:
        """Per assembly, the mean node burnup over fuel planes and quadrants.

        The fuel mesh is uniform (Deck.hz refuses otherwise), so the volume
        weighting MASTER applies degenerates to a plain mean.
        """
        out: dict[str, float] = {}
        for k, (ix, iy, _b) in enumerate(self.deck.fuel):
            flat = [v for plane in self.burnup[k] for v in plane]
            out[self.deck.label(ix, iy)] = sum(flat) / len(flat)
        return out

    def fa_quadrant_burnup(self) -> dict[str, list[float]]:
        """Per assembly, the mean over planes of each of the nn quadrants."""
        out: dict[str, list[float]] = {}
        for k, (ix, iy, _b) in enumerate(self.deck.fuel):
            planes = self.burnup[k]
            out[self.deck.label(ix, iy)] = [
                sum(p[q] for p in planes) / len(planes) for q in range(self.deck.nn)
            ]
        return out


# ---------------------------------------------------------------------------
# The MAS_OUT edit maps
# ---------------------------------------------------------------------------
_TAG = re.compile(r"^\$([A-Z0-9]+)_(\d+)\s+([-\d.]+)\s+DAY\s+([-\d.]+)\s+EFPD")
_NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")
_LABEL_WIDTH = 5  # the "  17 " gutter every one of these maps prints
_NOT_A_MAP_ROW = re.compile(r"[A-Za-z=|]")


def _is_map_row(line: str) -> bool:
    """A map row is decimals and blanks only, past the label gutter."""
    cells = line[_LABEL_WIDTH:]
    return bool(_NUM.search(cells)) and not _NOT_A_MAP_ROW.search(cells)


def _column_edges(body: list[str], labels: list[str]) -> list[int]:
    """The right edge of each column, learned from the numbers themselves."""
    seen: set[int] = set()
    for line in body:
        for m in _NUM.finditer(line[_LABEL_WIDTH:]):
            seen.add(m.end() + _LABEL_WIDTH)
    edges = sorted(seen)
    if len(edges) != len(labels):
        raise ValueError(f"found {len(edges)} numeric column edges for "
                         f"{len(labels)} column labels; the map is not "
                         "right-aligned as this parser assumes")
    return edges


def read_map_blocks(mas_out: Path, tag: str, nline: int):
    """Every `$<tag>_n` map in MAS_OUT.

    `nline` is the number of consecutive DECIMAL-number lines printed per y
    row -- see the module docstring; an integer-only legend line such as
    $B2D's "PIN LOCATION (I,J)" is not one of them.

    Returns [(efpd, {row_label: {column_label: (v1 .. v_nline)}})], one entry
    per printed statepoint, in file order.  A cell that is blank (the reflector
    gutter) is simply absent.
    """
    text = mas_out.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = []
    for i, line in enumerate(text):
        m = _TAG.match(line)
        if not m or m.group(1) != tag:
            continue
        efpd = float(m.group(4))
        j = i
        while "Y\\X" not in text[j]:
            j += 1
            if j > i + 40:
                raise ValueError(f"${tag}_{m.group(2)}: no Y\\X header within 40 lines")
        labels = text[j].split()[1:]

        # Collect STRICT row groups: a line whose label gutter is filled starts
        # one, and the next nline-1 lines complete it.  Anything else in the
        # block is ignored outright -- both the integer "PIN LOCATION (I,J)"
        # legend line and the quadrant summary ("1.0000   |    1.0000") that
        # MASTER prints under $P2D would otherwise invent column edges of their
        # own and desynchronise the whole map.
        groups: list[tuple[str, list[str]]] = []
        k = j + 1
        while k < len(text) and not text[k].startswith("$") and "----" not in text[k]:
            head = text[k][:_LABEL_WIDTH].strip()
            if head and _is_map_row(text[k]):
                lines = [text[k]]
                while len(lines) < nline and k + 1 < len(text):
                    k += 1
                    lines.append(text[k])
                if all(_is_map_row(s) for s in lines):
                    groups.append((head, lines))
            k += 1
        body = [s for _h, ls in groups for s in ls]
        edges = _column_edges(body, labels)
        col_of = {e: labels[c] for c, e in enumerate(edges)}

        grid: dict[str, dict[str, tuple]] = {}
        for row_label, lines in groups:
            cells: list[dict[str, float]] = []
            for line in lines:
                found: dict[str, float] = {}
                for mm in _NUM.finditer(line[_LABEL_WIDTH:]):
                    edge = mm.end() + _LABEL_WIDTH
                    if edge in col_of:
                        found[col_of[edge]] = float(mm.group())
                cells.append(found)
            row = grid.setdefault(row_label, {})
            for col in labels:
                # The FIRST line decides whether the cell exists at all -- a
                # blank there is the reflector gutter.
                if col in cells[0]:
                    row[col] = tuple(c.get(col) for c in cells)
        blocks.append((efpd, grid))
    if not blocks:
        raise ValueError(f"{mas_out}: no ${tag}_n blocks")
    return blocks


def read_layer_blocks(mas_out: Path, tag: str):
    """Every `$<tag>_n` 1-D (axial) edit in MAS_OUT.

    These are plain whitespace tables -- a LAYER / HEIGHT header rule, then one
    row per global axial node, TOP first, the two reflector nodes omitted.
    Returns [(efpd, {global_layer: (col1, col2, ...)})].

        $B1D_n   layer-averaged burnup, max burnup, max pin burnup
        $P1D_n   layer-averaged power,  max power,  max pin power, layer Fxy
        $FB1D_n  Tm(C), Tf(C), Dm(g/cc)
    """
    text = mas_out.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = []
    for i, line in enumerate(text):
        m = _TAG.match(line)
        if not m or m.group(1) != tag:
            continue
        rows: dict[int, tuple] = {}
        for k in range(i + 1, min(i + 60, len(text))):
            body = text[k]
            if body.startswith("$"):
                break
            fields = body.split()
            if len(fields) < 3 or not fields[0].isdigit():
                continue
            try:
                values = tuple(float(f) for f in fields[1:])
            except ValueError:
                continue
            rows[int(fields[0])] = values
        if rows:
            blocks.append((float(m.group(4)), rows))
    if not blocks:
        raise ValueError(f"{mas_out}: no ${tag}_n blocks")
    return blocks


# ---------------------------------------------------------------------------
# Validation -- the restart's 3-D burnup against MAS_OUT's own axial averages
# ---------------------------------------------------------------------------
def _day_of(path: Path) -> float:
    m = re.search(r"_(\d+\.\d+)$", path.name)
    if not m:
        raise ValueError(f"{path.name}: no _<day> suffix to say which statepoint "
                         "this restart is")
    return float(m.group(1))


def validate(rst: Restart, mas_out: Path, tol_bu: float = 0.05) -> list[str]:
    problems: list[str] = []
    day = _day_of(rst.path)
    b2d = read_map_blocks(mas_out, "B2D", 2)
    b1d = read_layer_blocks(mas_out, "B1D")
    efpd = min((e for e, _ in b2d), key=lambda e: abs(e - day))
    if abs(efpd - day) > 1e-6:
        return [f"no MAS_OUT statepoint at {day} EFPD (nearest is {efpd})"]
    fa_map = next(g for e, g in b2d if e == efpd)
    layers = next(g for e, g in b1d if e == efpd)

    # RADIAL: the restart's node burnup averaged over planes, per assembly.
    mine = rst.fa_burnup_axially_averaged()
    worst_fa = 0.0
    n_fa = 0
    for row, cols in fa_map.items():
        for col, values in cols.items():
            key = f"{col}{row}"
            if key in mine:
                worst_fa = max(worst_fa, abs(mine[key] - values[0]))
                n_fa += 1

    # AXIAL: the same nodes averaged the other way.  Together with the radial
    # check this pins BOTH indices of the (assembly, plane) ordering -- either
    # one alone would survive a transposed or reversed axial sweep.
    worst_pl = 0.0
    n_pl = 0
    for plane in range(rst.deck.nzf):
        glob = plane + 2  # plane 1 is global axial node 2 (node 1 is reflector)
        if glob not in layers:
            continue
        flat = [fa[plane][q] for fa in rst.burnup for q in range(rst.deck.nn)]
        worst_pl = max(worst_pl, abs(sum(flat) / len(flat) - layers[glob][1]))
        n_pl += 1

    print(f"  [RST] {n_fa} assemblies vs $B2D_({efpd:g} EFPD): "
          f"max |dBu| = {worst_fa:.4f} MWd/kgU")
    print(f"  [RST] {n_pl} planes vs $B1D_({efpd:g} EFPD): "
          f"max |dBu| = {worst_pl:.4f} MWd/kgU")
    if n_fa < 200:
        problems.append(f"only {n_fa} assemblies matched the $B2D map; the label "
                        "mapping is wrong")
    if n_pl < rst.deck.nzf:
        problems.append(f"only {n_pl} of {rst.deck.nzf} fuel planes matched $B1D")
    if worst_fa > tol_bu:
        problems.append(f"axially-averaged FA burnup differs from $B2D by "
                        f"{worst_fa:.4f} MWd/kgU (> {tol_bu})")
    if worst_pl > tol_bu:
        problems.append(f"radially-averaged plane burnup differs from $B1D by "
                        f"{worst_pl:.4f} MWd/kgU (> {tol_bu})")
    if rst.temperature_slot_nonzero:
        problems.append(f"the 5*(R8) per-node slot is NOT all zero "
                        f"({rst.temperature_slot_nonzero} nonzero values) -- this "
                        "module's claim that the build writes no node temperatures "
                        "needs revisiting")
    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("restart", type=Path, help="a MAS_RST.<case>_<day> file")
    ap.add_argument("--validate", type=Path, metavar="MAS_OUT",
                    help="check the parsed burnup against MAS_OUT's $B2D/$B2DN maps")
    args = ap.parse_args(argv)

    rst = Restart(args.restart)
    d = rst.deck
    print(f"  [RST] {args.restart.name}")
    print(f"  [RST] nx={d.nx} ny={d.ny} nz={d.nz} nzf={d.nzf} nn={d.nn} "
          f"npin={d.npin} nfrod={d.nfrod} fuel_assemblies={len(d.fuel)}")
    print(f"  [RST] hz={d.hz:g} cm, fuel height={d.fuel_height:g} cm, "
          f"power={d.power_mw:g} MW")
    print(f"  [RST] boron={rst.boron_ppm:.2f} ppm, tin={rst.tin_c:g} C, "
          f"pload={rst.pload:g}")
    flat = [v for fa in rst.burnup for pl in fa for v in pl]
    print(f"  [RST] node burnup over {len(flat)} nodes: min {min(flat):.3f} "
          f"mean {sum(flat)/len(flat):.3f} max {max(flat):.3f} MWd/kgU")
    print(f"  [RST] node temperature slot nonzero values: "
          f"{rst.temperature_slot_nonzero}  (records 23/24/25 are NOT written)")

    if args.validate:
        problems = validate(rst, args.validate)
        for p in problems:
            print("parse_master_rst: " + p, file=sys.stderr)
        if problems:
            return 1
        print("  [RST] validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
