#!/usr/bin/env python3
"""master2rasi.py -- translate a MASTER card deck into a RASBERY case JSON.

The MASTER core deck (``%GEN_DIM``/``%LPD_BCH``/``%EXE_DEP``/...) is parsed and
re-expressed as the RASBERY case description consumed by ``RASBERY --rasi``.

Design rules
------------
* **fail-closed.**  Every card in the deck must be either translated or listed in
  ``INERT_CARDS`` (carries no information RASBERY can use).  Anything else aborts
  the translation with a diagnostic.  The same holds for card *values*: an
  unrecognised keyword or an out-of-range flag is an error, never a silent default.
* **no C++ changes.**  This is a pure front-end: it only produces JSON that the
  existing RASBERY input reader already accepts.
* **explicit provenance.**  Every emitted value is traceable to a card, a CLI
  option or a ``!RASI`` directive.  Values that a MASTER deck simply cannot carry
  (the cross-section library, the fixed fuel temperature) must be supplied.

Usage
-----
    master2rasi.py deck.inp --xs lib.h5 -o case.json [--fold quarter]

See ``tools/MASTER_DECK_INPUT.md`` for the card coverage table and limitations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

VERSION = "1.0"

# --------------------------------------------------------------------------
# Card registry
# --------------------------------------------------------------------------

# Cards whose content is translated into the case JSON.
TRANSLATED_CARDS = {
    "%GEN_DIM", "%GEN_GEO", "%GEN_SYM", "%GEN_THD", "%GEN_FDB", "%GEN_PIN",
    "%LPD_BCH", "%LPD_B&C", "%LPD_C&X",
    "%EXE_STD", "%EXE_DEP", "%EXE_ROD",
    "%EDT_OPT",
    "%ROD_CFG", "%ROD_MAP",
}

# Cards that carry nothing RASBERY consumes but whose *values* still have to be
# read, because a particular setting would make the translation wrong.  They
# emit no JSON; they either pass or abort.  Each entry is the reason it is
# checked, reproduced in the coverage report.
CHECKED_CARDS = {
    "%JOB_HEX": "rectangular/hexagonal selector; hexagonal cores are refused",
    "%GEN_MTH": "MASTER nodal method selectors; ibndc (albedo boundary source) "
                "is refused because the translator reads %GEN_SYM only",
}

# Cards accepted but carrying nothing RASBERY consumes.  Each entry is the
# reason the card is inert, reproduced in the coverage report.
INERT_CARDS = {
    "%JOB_TYP": "job/file bookkeeping (restart flag, MASTER file names)",
    "%JOB_VER": "MASTER version + actinide-chain selector",
    "%JOB_TIT": "free-text title",
    "%JOB_IDE": "plant/cycle identifiers",
    "%JOB_MDL": "MASTER solver model switches (no RASBERY analogue)",
    "%GEN_LMT": "MASTER iteration limits; RASBERY convergence is set separately",
    "%GEN_CDN": "printed row/column labels only",
    "%EDT_OUT": "MASTER output-listing switches",
    "%EDT_PIN": "MASTER pin-edit window selection (MASTER 3.0)",
    "%LPD_HFF": "form-function assignment; RASBERY takes pin factors from the XS library",
    "%COB_INP": "COBRA sub-channel T/H model input (RASBERY has its own T/H)",
    "%MSC_MEM": "MASTER memory allocation hints (MASTER 3.0)",
    "%DEF_MSC": "miscellaneous MASTER parameters",
}

# Cards that are understood but deliberately refused, with the reason shown to
# the user.  Anything not listed anywhere at all is refused generically.
REFUSED_CARDS = {
    "%LPD_SHF": "multi-cycle shuffling is not expressible in a single RASBERY case",
    "%LPD_PUL": "spent-fuel-pool decay timing has no RASBERY analogue",
    "%LPD_ASF": "asymmetric assembly nuclide overrides cannot be expressed in a "
                "RASBERY case",
    "%LPD_DB2": "radial buckling block is reserved in MASTER and undefined here",
    "%EXE_RHO": "reactivity-coefficient execution mode has no RASBERY analogue",
    "%EXE_CRS": "MASTER's control-rod search walks a bank sequence that the "
                "RASBERY rod search does not reproduce step for step",
    "%EXE_SHP": "axial power shape matching has no RASBERY analogue",
    "%EXE_TRN": "transient execution is outside the RASBERY case format",
    "%EXE_XED": "xenon-dynamics execution is outside the RASBERY case format",
    "%DEF_ETS": "external fixed-source problems are not supported",
    "%DEF_TFT": "user fuel-temperature tables have no RASBERY analogue",
    "%DET_CFG": "in-core detector modelling is not supported",
    "%DET_MAP": "in-core detector modelling is not supported",
    "%GEN_DCH": "decay-heat modelling is not supported",
    "%ROD_COR": "corner control rods are hexagonal-only and unsupported",
    "%THC_FLO": "per-channel flow distribution is not supported",
    "%THC_TIN": "per-channel inlet temperature is not supported",
    "%TRN_DEF": "transient group cards are not supported",
    "%TRN_ENT": "transient group cards are not supported",
    "%TRN_OPT": "transient group cards are not supported",
    "%TRN_PRN": "transient group cards are not supported",
    "%TRN_SHD": "transient group cards are not supported",
    "%XED_DEF": "xenon-dynamics group cards are not supported",
    "%XED_OPT": "xenon-dynamics group cards are not supported",
    "%XED_PRN": "xenon-dynamics group cards are not supported",
    "%XSM_DEV": "cross-section modification must be done when building the library",
    "%XSM_H2O": "cross-section modification must be done when building the library",
    "%XSM_MIC": "cross-section modification must be done when building the library",
    "%XSM_NUC": "cross-section modification must be done when building the library",
    "%XSM_ROD": "cross-section modification must be done when building the library",
    "%XSM_YLD": "cross-section modification must be done when building the library",
}

# MASTER composition-name -> RASBERY XS model-name normalisation, tried in order
# after an exact match and after any user alias.  Each rule yields a candidate
# that must exist in the library, otherwise the translation fails.
NAME_RULES = [
    (re.compile(r"^REF_AXIAL_B$", re.I), lambda m: "RB"),
    (re.compile(r"^REF_AXIAL_T$", re.I), lambda m: "RT"),
    (re.compile(r"^FA_(.+)$", re.I), lambda m: m.group(1)),
    (re.compile(r"^REF_(.+)$", re.I), lambda m: m.group(1)),
]

# Rules that normalise a MASTER *reflector* name.  RASBERY splits fuel from
# reflector on the first letter of the batch-layer model name (see
# MASTER_DECK_INPUT.md §6): a reflector whose model name does not begin with 'R'
# is loaded as fuel, and the active-core extent -- depletion range, peaking
# search window, rod insertion depth -- is then computed over the wrong nodes.
# A REF_ rule that resolves to a non-R name is therefore an error, not a name.
REFLECTOR_RULES = {r"^REF_AXIAL_B$", r"^REF_AXIAL_T$", r"^REF_(.+)$"}

VOID = "o"          # empty lattice position in %LPD_BCH / %ROD_MAP
KELVIN = 273.15


class DeckError(Exception):
    """Raised for any condition that must stop the translation."""


# --------------------------------------------------------------------------
# Lexing
# --------------------------------------------------------------------------

# The card name is anchored on both sides.  Without the trailing boundary a
# mistyped suffix ("%GEN_DIM1") matched the six characters of a *different* card
# and the stray tail became its first body token: the deck then parsed silently
# with nx shifted by one.  A card name must be followed by whitespace or the end
# of the line; anything else falls through to the "unrecognised card syntax"
# branch in parse_deck().
CARD_RE = re.compile(r"^\s*(%[A-Za-z0-9]{3}_[A-Za-z0-9&]{3})(?=\s|$)\s*(.*)$")
DIRECTIVE_RE = re.compile(r"^\s*[#!]\s*RASI\s+(.*)$", re.I)
REPEAT_RE = re.compile(r"^(\d+)\*(.+)$")


class Card:
    """One MASTER card: its name and the token rows of its body."""

    def __init__(self, name, line_no):
        self.name = name
        self.line_no = line_no
        self.rows = []          # list of list-of-token, comments stripped

    @property
    def tokens(self):
        """All body tokens, flattened."""
        return [t for row in self.rows for t in row]


def strip_comment(line):
    """Remove a MASTER trailing comment.

    Both '#' and '!' start a comment anywhere on the line (MASTER 4.0 manual,
    input syntax rules).
    """
    return re.split(r"[#!]", line, maxsplit=1)[0]


def expand_repeats(tokens):
    """Expand the MASTER ``n*value`` repeat syntax."""
    out = []
    for tok in tokens:
        m = REPEAT_RE.match(tok)
        if m:
            count = int(m.group(1))
            if count < 0:
                raise DeckError(f"negative repeat count in token '{tok}'")
            out.extend([m.group(2)] * count)
        else:
            out.append(tok)
    return out


def parse_deck(text):
    """Split a MASTER deck into ordered cards, state separators and directives.

    Returns ``(items, directives)`` where *items* is a list of either a
    :class:`Card` or the string ``"/"`` (execution-state separator), in deck
    order, and *directives* is the dict parsed from ``!RASI`` comment lines.
    """
    items = []
    directives = {}
    aliases = {}
    current = None

    for line_no, raw in enumerate(text.splitlines(), 1):
        # !RASI directives live in comments and are read before comment stripping.
        d = DIRECTIVE_RE.match(raw)
        if d:
            body = d.group(1).strip()
            if body.lower().startswith("alias "):
                spec = body[6:].strip()
                if "=" not in spec:
                    raise DeckError(
                        f"line {line_no}: !RASI alias needs NAME=MODEL, got '{spec}'")
                k, v = spec.split("=", 1)
                aliases[k.strip()] = v.strip()
            elif "=" in body:
                k, v = body.split("=", 1)
                directives[k.strip().lower()] = v.strip()
            else:
                raise DeckError(
                    f"line {line_no}: malformed !RASI directive '{body}'")
            continue

        line = strip_comment(raw)
        if not line.strip():
            continue

        stripped = line.strip()

        # END terminates the deck.
        if stripped.upper() == "END":
            break

        # A bare '/' closes the current execution state.
        if stripped == "/":
            current = None
            items.append("/")
            continue

        m = CARD_RE.match(line)
        if m:
            current = Card(m.group(1).upper(), line_no)
            items.append(current)
            rest = m.group(2).strip()
            if rest:
                current.rows.append(rest.split())
            continue

        if stripped.startswith("%"):
            raise DeckError(
                f"line {line_no}: unrecognised card syntax '{stripped.split()[0]}'")

        if current is None:
            raise DeckError(
                f"line {line_no}: data outside any card: '{stripped}'")

        # A '/' may terminate a card body on the same line as data.
        if stripped.endswith("/"):
            body = stripped[:-1].strip()
            if body:
                current.rows.append(body.split())
            current = None
            items.append("/")
            continue

        current.rows.append(stripped.split())

    directives["_aliases"] = aliases
    return items, directives


# --------------------------------------------------------------------------
# Small typed accessors -- every one fails loudly
# --------------------------------------------------------------------------

def as_int(tok, what):
    try:
        return int(tok)
    except ValueError:
        raise DeckError(f"{what}: expected an integer, got '{tok}'")


def as_float(tok, what):
    try:
        value = float(tok)
    except ValueError:
        raise DeckError(f"{what}: expected a number, got '{tok}'")
    # float() happily accepts 'nan', 'inf' and '-Infinity'.  Those would travel
    # through the whole translation and land in the JSON as bare NaN/Infinity
    # literals, which are not JSON and which RASBERY's reader turns into a
    # silently poisoned state point.
    if not math.isfinite(value):
        raise DeckError(f"{what}: '{tok}' is not a finite number")
    return value


def need(tokens, n, what, exact=False):
    """Require at least *n* tokens, or exactly *n* when ``exact``.

    Fixed-width MASTER cards carry a known field count.  Accepting a longer row
    on those cards is how a typo (a duplicated field, a comment character eaten
    by a missing '#') passes as valid input: the surplus is simply never read.
    ``exact=True`` makes the surplus an error instead.
    """
    if len(tokens) < n:
        raise DeckError(f"{what}: expected {n} values, found {len(tokens)}")
    if exact and len(tokens) > n:
        raise DeckError(
            f"{what}: expected exactly {n} values, found {len(tokens)} "
            f"(unread trailing data: {' '.join(tokens[n:])})")
    return tokens


def need_any(tokens, counts, what):
    """Require the token count to be one of *counts* (a fixed-width card with
    an optional documented tail, e.g. %GEN_THD's 5 or 7 fields)."""
    if len(tokens) not in counts:
        raise DeckError(
            f"{what}: expected {' or '.join(str(c) for c in sorted(counts))} "
            f"values, found {len(tokens)}"
            + (f" (unread trailing data: {' '.join(tokens[max(counts):])})"
               if len(tokens) > max(counts) else ""))
    return tokens


def onoff(tok, what):
    low = tok.lower()
    if low in ("off", "no", "0", "f", "false"):
        return False
    if low in ("on", "yes", "1", "t", "true"):
        return True
    raise DeckError(f"{what}: expected on/off, got '{tok}'")


# --------------------------------------------------------------------------
# Deck -> intermediate model
# --------------------------------------------------------------------------

class MasterDeck:
    """Everything the translator understands about one MASTER deck."""

    def __init__(self):
        self.dim = {}
        self.geo = {}
        self.sym = {}
        self.thd = {}
        self.fdb = {}
        self.pin = {}
        self.lp = []            # %LPD_BCH, full-core rows of batch names
        self.batch_axial = dict()   # batch -> [comp index, bottom->top]
        self.comp_names = {}    # comp index -> MASTER XS set name
        self.comp_extra = {}    # comp index -> trailing fields (diagnostics)
        self.rod_cfg = dict()
        self.rod_map = []
        self.states = []        # execution states in deck order
        self.seen_cards = []    # (name, line_no) in deck order


def collect(items, strict=True):
    """Walk the parsed items and build a :class:`MasterDeck`.

    With ``strict=False`` (the compatibility checker) an unsupported card is
    recorded in ``deck.blockers`` and skipped instead of aborting, so the
    report can list every problem in the deck at once.
    """
    deck = MasterDeck()
    deck.blockers = []

    # An execution state is the run of cards terminated by '/'.  Cards that
    # appear before any %EXE_* card configure the model globally.
    pending = []
    for item in items:
        if item == "/":
            if pending:
                deck.states.append(pending)
                pending = []
            continue

        card = item
        deck.seen_cards.append((card.name, card.line_no))

        if card.name in REFUSED_CARDS:
            msg = (f"line {card.line_no}: card {card.name} is not supported -- "
                   f"{REFUSED_CARDS[card.name]}")
            if strict:
                raise DeckError(msg)
            deck.blockers.append(msg)
            continue

        if card.name in CHECKED_CARDS:
            try:
                _check_only_card(card)
            except DeckError as e:
                if strict:
                    raise
                deck.blockers.append(str(e))
            continue

        if card.name in INERT_CARDS:
            continue

        if card.name not in TRANSLATED_CARDS:
            msg = (f"line {card.line_no}: card {card.name} is not known to this "
                   "translator; refusing to guess.  Add it to the card table in "
                   "tools/master2rasi.py (and tools/MASTER_DECK_INPUT.md) if it "
                   "should be translated or ignored.")
            if strict:
                raise DeckError(msg)
            deck.blockers.append(msg)
            continue

        if card.name.startswith("%EXE_") or card.name == "%EDT_OPT":
            pending.append(card)
        else:
            try:
                _read_model_card(deck, card)
            except DeckError as e:
                if strict:
                    raise
                deck.blockers.append(str(e))

    if pending:
        deck.states.append(pending)

    return deck


def _check_only_card(card):
    """Validate a card that emits nothing but whose value can invalidate the run.

    These used to sit in INERT_CARDS, which meant their tokens were never even
    looked at.  The coverage report claimed %JOB_HEX was "checked, then inert"
    while nothing checked it, and MASTER_DECK_INPUT.md documented %GEN_MTH's
    ibndc as a known divergence source that was nevertheless accepted silently.
    """
    where = f"{card.name} (line {card.line_no})"
    t = expand_repeats(card.tokens)

    if card.name == "%JOB_HEX":
        # %JOB_HEX selects the hexagonal geometry package.  A bare card, or a
        # leading 0, keeps MASTER rectangular; anything else asks for a
        # hexagonal core, which this translator (and RASBERY's Cartesian
        # geometry) cannot express.
        if not t:
            return
        ihex = as_int(t[0], where)
        if ihex != 0:
            raise DeckError(
                f"{where}: ihex={ihex} selects a hexagonal core.  RASBERY's "
                "geometry is Cartesian and this translator has no hexagonal "
                "path, so the deck cannot be translated.")

    elif card.name == "%GEN_MTH":
        # Field 3 is ibndc: 0 uses the %GEN_SYM boundary flags, 1 and 2 replace
        # them with an albedo / boundary-source specification that lives outside
        # %GEN_SYM.  The translator reads %GEN_SYM only, so it would emit the
        # wrong boundary condition and the run would quietly converge to the
        # wrong core leakage.
        need(t, 3, where)
        ibndc = as_int(t[2], where)
        if ibndc not in (0, 1, 2):
            raise DeckError(
                f"{where}: ibndc={ibndc} is outside the documented range 0..2")
        if ibndc != 0:
            raise DeckError(
                f"{where}: ibndc={ibndc} replaces the %GEN_SYM boundary flags "
                "with MASTER's own albedo/boundary-source specification.  This "
                "translator derives geometry.albedo from %GEN_SYM alone, so the "
                "boundary condition it would emit is not the one the deck asks "
                "for.  Set ibndc=0, or translate the boundary by hand.")

    else:   # pragma: no cover - guarded by CHECKED_CARDS
        raise DeckError(f"{where}: card claimed as checked but not handled")


def _read_model_card(deck, card):
    name = card.name
    where = f"{name} (line {card.line_no})"

    if name == "%GEN_DIM":
        t = expand_repeats(card.tokens)
        need(t, 11, where, exact=True)
        deck.dim = dict(
            nx=as_int(t[0], where), ny=as_int(t[1], where), nz=as_int(t[2], where),
            nbatch=as_int(t[3], where), ncomp=as_int(t[4], where),
            ndim=as_int(t[5], where), ngeo=as_int(t[6], where),
            nsym=as_int(t[7], where), ndivxy=as_int(t[8], where),
            ndivz=as_int(t[9], where), ng=as_int(t[10], where))

    elif name == "%GEN_GEO":
        rows = card.rows
        if len(rows) < 2:
            raise DeckError(f"{where}: expected a 'wide height' row and an axial mesh row")
        head = need(rows[0], 2, where)
        deck.geo["wide"] = as_float(head[0], where)
        deck.geo["height"] = as_float(head[1], where)
        mesh = expand_repeats([t for row in rows[1:] for t in row])
        deck.geo["hz"] = [as_float(v, where) for v in mesh]

    elif name == "%GEN_SYM":
        t = expand_repeats(card.tokens)
        need(t, 6, where, exact=True)
        deck.sym = dict(
            lx=as_int(t[0], where), ly=as_int(t[1], where), lz=as_int(t[2], where),
            rx=as_int(t[3], where), ry=as_int(t[4], where), rz=as_int(t[5], where))

    elif name == "%GEN_THD":
        # MASTER writes either the five-field form or the five fields plus the
        # optional 'mflow hgfl' pair.  Six fields, or eight, means the deck says
        # something this translator is not reading.
        t = expand_repeats(card.tokens)
        need_any(t, (5, 7), where)
        deck.thd = dict(
            power=as_float(t[0], where), tin=as_float(t[1], where),
            trise=as_float(t[2], where), tavg=as_float(t[3], where),
            press=as_float(t[4], where))
        if len(t) == 7:
            deck.thd["mflow"] = as_float(t[5], where)
            deck.thd["hgfl"] = as_float(t[6], where)
        # Physical plausibility.  RASBERY divides by the pressure and converts
        # both temperatures to kelvin; a swapped or mis-parsed field lands as a
        # non-positive pressure or a sub-zero absolute temperature, which the
        # solver then consumes without comment.
        if deck.thd["press"] <= 0.0:
            raise DeckError(
                f"{where}: pressure {deck.thd['press']:g} bar is not positive")
        if deck.thd["trise"] < 0.0:
            raise DeckError(
                f"{where}: coolant temperature rise {deck.thd['trise']:g} C is "
                "negative")
        for field, label in (("tin", "inlet temperature"),
                             ("tavg", "average moderator temperature")):
            if deck.thd[field] + KELVIN <= 0.0:
                raise DeckError(
                    f"{where}: {label} {deck.thd[field]:g} C is below absolute "
                    f"zero ({deck.thd[field] + KELVIN:g} K)")
        if deck.thd["tin"] + deck.thd["trise"] + KELVIN <= 0.0:
            raise DeckError(
                f"{where}: outlet temperature "
                f"{deck.thd['tin'] + deck.thd['trise']:g} C is below absolute zero")

    elif name == "%GEN_FDB":
        t = card.tokens
        need(t, 3, where)
        deck.fdb = dict(tfuel=onoff(t[0], where),
                        tmod=onoff(t[1], where),
                        dmod=onoff(t[2], where))

    elif name == "%GEN_PIN":
        t = expand_repeats(card.tokens)
        need(t, 4, where, exact=True)
        deck.pin = dict(icornf=as_int(t[0], where), iweigh=as_int(t[1], where),
                        npin=abs(as_int(t[2], where)),
                        nfrod=abs(as_int(t[3], where)))

    elif name == "%LPD_BCH":
        deck.lp = [list(row) for row in card.rows]

    elif name == "%LPD_B&C":
        for row in card.rows:
            if len(row) < 2:
                raise DeckError(f"{where}: batch row '{' '.join(row)}' has no composition list")
            batch = row[0]
            comps = expand_repeats(row[1:])
            deck.batch_axial[batch] = [as_int(c, where) for c in comps]

    elif name == "%LPD_C&X":
        # Rows are either "<icomp> <xsname> <n> [extra...]" or a continuation
        # line belonging to the previous composition (e.g. "STRM 1.0").
        last = None
        for row in card.rows:
            try:
                icomp = int(row[0])
            except ValueError:
                if last is None:
                    raise DeckError(
                        f"{where}: continuation row '{' '.join(row)}' before any composition")
                deck.comp_extra.setdefault(last, []).append(list(row))
                continue
            if len(row) < 2:
                raise DeckError(f"{where}: composition {icomp} has no XS set name")
            deck.comp_names[icomp] = row[1]
            deck.comp_extra.setdefault(icomp, [])
            if len(row) > 2:
                deck.comp_extra[icomp].append(list(row[2:]))
            last = icomp

    elif name == "%ROD_CFG":
        rows = [r for r in card.rows if r]
        if not rows:
            raise DeckError(f"{where}: empty card")
        ncrgr = as_int(rows[0][0], where) if len(rows[0]) == 1 else None
        body = rows[1:] if ncrgr is not None else rows
        for row in body:
            need(row, 11, f"{where} group '{row[0]}'", exact=True)
            deck.rod_cfg[row[0]] = dict(
                mattip=as_int(row[1], where), matabs=as_int(row[2], where),
                matfol=as_int(row[3], where), lentip=as_float(row[4], where),
                lenabs=as_float(row[5], where), crups=as_float(row[6], where),
                crlos=as_float(row[7], where), crpos1=as_float(row[8], where),
                crpos2=as_float(row[9], where), ifgtp=as_int(row[10], where))
        if ncrgr is not None and len(deck.rod_cfg) != ncrgr:
            raise DeckError(
                f"{where}: ncrgr={ncrgr} but {len(deck.rod_cfg)} groups were read")

    elif name == "%ROD_MAP":
        deck.rod_map = [list(row) for row in card.rows]

    else:  # pragma: no cover - guarded by TRANSLATED_CARDS
        raise DeckError(f"{where}: card claimed as translated but not handled")


# --------------------------------------------------------------------------
# Symmetry handling
# --------------------------------------------------------------------------

def rectangular(rows, fill=VOID):
    """Pad ragged rows into a rectangular grid."""
    width = max(len(r) for r in rows)
    return [list(r) + [fill] * (width - len(r)) for r in rows]


def check_diagonal_symmetry(grid, what="loading pattern"):
    """Verify an already-folded quarter map is invariant under the transpose.

    A quarter core is only a valid 90-degree sector with ``mirror: true`` if the
    two halves either side of the main diagonal agree; RASBERY reflects across
    it.  A map that does not is a 90-degree sector *without* mirror symmetry,
    which this translator has no way to express.
    """
    n = len(grid)
    if any(len(r) != n for r in grid):
        raise DeckError(
            f"{what}: an already-folded quarter map must be square, found "
            f"{n} rows of differing width")
    problems = [f"({i+1},{j+1})='{grid[i][j]}' vs transpose "
                f"({j+1},{i+1})='{grid[j][i]}'"
                for i in range(n) for j in range(n) if grid[i][j] != grid[j][i]]
    if problems:
        raise DeckError(
            f"{what}: the quarter map is not symmetric about its main diagonal, "
            "so it is not the mirrored 90-degree sector RASBERY reflects.  First "
            f"mismatches: {'; '.join(problems[:5])}"
            + (f" (+{len(problems)-5} more)" if len(problems) > 5 else ""))


def already_folded_core(deck, grid, report, what="loading pattern"):
    """Translate a deck that *is already* a partial core (%GEN_DIM ngeo != 1).

    This is the third geometry mode, and its absence was a silent 4x error.  The
    translator had exactly two: "full core" and "full core, fold it".  A deck
    with ngeo=4 matched neither, fell through to the full-core branch, and its
    quarter map was emitted with ``{"angle": 360, "mirror": false}`` and the
    full-core rated power -- a quarter of the geometry carrying all of the power.
    Nothing in the run says so; it converges and reports a plausible k-eff.

    Every field below is derived from a card, never assumed:

      ngeo                  -> the sector: 360/ngeo degrees, power x 1/ngeo
      %GEN_SYM isymlx/isymly-> "center assembly divided".  MASTER writes -1 for a
                               symmetry plane through the middle of an assembly
                               and +1 for one on an assembly boundary; RASBERY's
                               Geometry.cpp halves the first row/column exactly
                               when (angle == 90 and divided).
      the map itself        -> "mirror", from its diagonal symmetry.

    Anything not derivable is refused rather than guessed.
    """
    ngeo = deck.dim["ngeo"]
    if ngeo != 4:
        raise DeckError(
            f"%GEN_DIM ngeo={ngeo}: the deck describes 1/{ngeo} of the core.  "
            "This translator can express a full core (ngeo=1) and a quarter "
            "core (ngeo=4); RASBERY's 'symmetry' block has no representation "
            f"for a 1/{ngeo} sector, so the case cannot be written.")

    if not deck.sym:
        raise DeckError(
            f"%GEN_DIM ngeo={ngeo} says the deck is already a quarter core, but "
            "there is no %GEN_SYM card.  The cut faces have to be declared "
            "symmetry planes (isymlx/isymly = -1 or +1) -- without them neither "
            "the boundary condition nor 'center assembly divided' is known, and "
            "both change the answer.")

    lx, ly = deck.sym["lx"], deck.sym["ly"]
    for flag, field in ((lx, "isymlx"), (ly, "isymly")):
        if flag not in (-1, 1):
            raise DeckError(
                f"%GEN_SYM {field}={flag}: with %GEN_DIM ngeo={ngeo} the west "
                "and north faces are the planes the quarter core was cut on and "
                "must be declared symmetry planes (-1 through the middle of an "
                f"assembly, +1 on an assembly boundary).  {field}={flag} says "
                "otherwise, so the deck and %GEN_DIM disagree about what this "
                "map is.")
    if lx != ly:
        raise DeckError(
            f"%GEN_SYM isymlx={lx} but isymly={ly}: one cut plane passes through "
            "the middle of an assembly and the other along a boundary.  "
            "RASBERY's 'center assembly divided' is a single flag for both axes, "
            "so this geometry cannot be expressed.")

    check_diagonal_symmetry(grid, what)
    divided = (lx == -1)

    core = full_core_map(grid, what)
    symmetry = {"angle": 360 // ngeo, "mirror": True,
                "center assembly divided": divided}
    report.note(
        f"%GEN_DIM ngeo={ngeo}: the deck is already a quarter core.  Emitted as "
        f"a {360 // ngeo}-degree sector with mirror symmetry (map verified "
        f"symmetric about its diagonal), 'center assembly divided'={divided} "
        f"from %GEN_SYM isymlx={lx}, and rated power x 1/{ngeo}.")
    return core, 1.0 / ngeo, symmetry


def check_octant_symmetry(grid, what="loading pattern"):
    """Verify a square full-core map is symmetric under the octant group.

    The quarter-core fold is only meaningful if the map is invariant under the
    left-right mirror, the up-down mirror and the main-diagonal transpose.
    Anything less and the fold would silently discard information.
    """
    n = len(grid)
    if any(len(r) != n for r in grid):
        raise DeckError(f"{what}: not square ({n} rows), cannot fold")
    if n % 2 == 0:
        raise DeckError(
            f"{what}: even dimension {n}; the quarter fold expects an odd "
            "full-core map with a central assembly")

    problems = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] != grid[i][n - 1 - j]:
                problems.append(f"({i+1},{j+1}) vs left-right mirror ({i+1},{n-j})")
            if grid[i][j] != grid[n - 1 - i][j]:
                problems.append(f"({i+1},{j+1}) vs up-down mirror ({n-i},{j+1})")
            if grid[i][j] != grid[j][i]:
                problems.append(f"({i+1},{j+1}) vs transpose ({j+1},{i+1})")
    if problems:
        raise DeckError(
            f"{what}: not octant-symmetric, refusing to fold.  First "
            f"mismatches: {'; '.join(problems[:5])}"
            + (f" (+{len(problems)-5} more)" if len(problems) > 5 else ""))


def fold_quarter(grid, what="loading pattern"):
    """Reduce an octant-symmetric full-core map to the RASBERY quarter map.

    Returns the south-east quadrant including the central row/column, with
    trailing void positions trimmed (RASBERY expresses those as short rows).
    """
    check_octant_symmetry(grid, what)
    n = len(grid)
    c = n // 2                      # 0-based centre index
    quarter = [row[c:] for row in grid[c:]]

    trimmed = []
    for r, row in enumerate(quarter):
        while row and row[-1] == VOID:
            row = row[:-1]
        if VOID in row:
            first = row.index(VOID)
            raise DeckError(
                f"{what}: quarter-core row {r+1} has an interior empty position "
                f"at column {first+1}; RASBERY can only express trailing gaps")
        if row:
            trimmed.append(row)
    return trimmed


def full_core_map(grid, what="loading pattern"):
    """Emit a full-core map using RASBERY's void conventions.

    RASBERY expresses an absent lattice position two ways, and only two:
    a leading run of ``"XX"`` cells (the parser sums their spans into the row's
    west-side offset) and a short row (trailing gap).  An ``"XX"`` anywhere in
    between silently mis-places the whole row, so an interior gap is refused.
    """
    out = []
    for r, row in enumerate(grid):
        row = list(row)
        lead = 0
        while lead < len(row) and row[lead] == VOID:
            lead += 1
        while row and row[-1] == VOID:
            row = row[:-1]
        body = row[lead:]
        if VOID in body:
            first = lead + body.index(VOID)
            raise DeckError(
                f"{what}: row {r+1} has an empty position at column {first+1} "
                "that is neither a leading nor a trailing gap.  RASBERY can only "
                "express those two, so this map cannot be translated as written; "
                "use --fold quarter.")
        if body:
            out.append(["XX"] * lead + body)
    return out


# --------------------------------------------------------------------------
# Cross-section name resolution
# --------------------------------------------------------------------------

def load_model_names(xs_path):
    """Return the set of model names in a CHIFFON HDF5 library, or None."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(xs_path, "r") as f:
            if "Models" not in f:
                return None
            names = set()
            for key in f["Models"]:
                val = f["Models"][key]["name"][()]
                names.add(val.decode() if isinstance(val, bytes) else str(val))
            return names
    except OSError:
        return None


def resolve_model(master_name, models, aliases, cache):
    """Map a MASTER %LPD_C&X XS-set name onto a RASBERY XS model name."""
    if master_name in cache:
        return cache[master_name]

    candidates = []
    if master_name in aliases:
        candidates.append(("!RASI alias", aliases[master_name], None))
    candidates.append(("exact", master_name, None))
    for rx, fn in NAME_RULES:
        m = rx.match(master_name)
        if m:
            candidates.append((f"rule {rx.pattern}", fn(m), rx.pattern))

    def accept(how, cand, pattern):
        # The reflector invariant, checked on the candidate that is actually
        # adopted rather than on every candidate that was merely tried.
        if pattern in REFLECTOR_RULES and not cand.upper().startswith("R"):
            raise DeckError(
                f"composition '{master_name}' normalises to model '{cand}' via "
                f"{how}, but RASBERY decides fuel vs reflector from the first "
                "letter of the model name and this name does not start with "
                "'R'.  It would be loaded as fuel and the active-core extent "
                "would be wrong.  Rename the library model, or map it "
                f"explicitly with '!RASI alias {master_name}=<Rxx>'.")
        cache[master_name] = cand
        return cand

    if models is None:
        # No library to check against: take the first normalisation and say so.
        if master_name in aliases:
            pick = candidates[0]
        else:
            pick = candidates[-1] if len(candidates) > 1 else ("exact", master_name, None)
        return accept(*pick)

    for how, cand, pattern in candidates:
        if cand in models:
            return accept(how, cand, pattern)

    raise DeckError(
        f"composition '{master_name}' does not match any model in the "
        f"cross-section library.  Tried: "
        + ", ".join(f"'{c}' ({how})" for how, c, _ in candidates)
        + ".  Library contains: " + ", ".join(sorted(models))
        + ".  Add '!RASI alias " + master_name + "=<model>' to the deck or pass "
        "--alias to map it explicitly.")


# --------------------------------------------------------------------------
# Execution states -> schedule
# --------------------------------------------------------------------------

# %EXE_STD isearch.  MASTER 4.0 documents exactly five keywords; the two that
# RASBERY can express map onto its own search types, the rest are refused.
# (Control-rod search is a separate MASTER card, %EXE_CRS.)
SEARCH_KEYWORDS = {
    "keff": "keff",
    "boron": "boron",
}
SEARCH_REFUSED = {
    "power": "MASTER's power search has no RASBERY analogue",
    "extso": "external fixed-source problems are not supported",
    "tin": "inlet-temperature search has no RASBERY analogue",
}

# %EXE_STD ixe/ism.  MASTER 4.0 documents exactly three keywords.  RASBERY's
# xenon parser accepts only transient/equilibrium and throws on anything else,
# so 'no' has to be refused rather than silently downgraded.
XENON_KEYWORDS = {
    "tr": "transient",
    "eq": "equilibrium",
}
XENON_REFUSED = {
    "no": "RASBERY always tracks Xe/Sm; it has no 'no xenon' mode",
}


def build_schedule(deck, opts, report):
    """Turn the MASTER execution states into the RASBERY schedule array."""
    schedule = []
    # %EDT_OPT is sticky: it stays in force until the next %EDT_OPT.
    edt = dict(iwrst=0, icob=0, ippi=0)
    std_seen = False
    xenon = None
    xenon_states = []       # (mode, keyword, line) one per %EXE_STD
    rate = None
    search = None
    rod_states = []
    boron_targets = []      # (ppm, line) set by %EXE_DEP itg>0 under a keff search
    keff_targets = []       # (keff, line) set by %EXE_DEP itg>0 under a boron search

    for state in deck.states:
        exe = [c for c in state if c.name.startswith("%EXE_")]
        opt = [c for c in state if c.name == "%EDT_OPT"]

        for card in opt:
            t = expand_repeats(card.tokens)
            where = f"%EDT_OPT (line {card.line_no})"
            # MASTER 3.0 wrote four fields (iwrst icob ippi icmp); 4.0 writes six
            # (iwrst icob ippi icoms inxs issim).  Only iwrst and ippi are used
            # here and they occupy the same slots in both.
            need(t, 4, where)
            edt = dict(iwrst=as_int(t[0], where), icob=as_int(t[1], where),
                       ippi=as_int(t[2], where))
            # iwrst/ippi == 2 attaches a user file-name token in MASTER 4.0.
            # Only warn when such a token is actually present.
            names = [v for v in t[3:] if not re.fullmatch(r"[+-]?\d+", v)]
            if names:
                report.note(f"{where}: user-named output files "
                            f"({', '.join(names)}) are not carried over")

        if not exe:
            # A state block with only %EDT_OPT re-prints the current state;
            # MASTER lists it as a duplicate edit and it advances nothing.
            if opt:
                report.note("%EDT_OPT block with no %EXE_* card produces no new "
                            f"state (line {opt[0].line_no})")
            continue

        for card in exe:
            where = f"{card.name} (line {card.line_no})"

            if card.name == "%EXE_STD":
                t = card.tokens
                need(t, 4, where, exact=True)
                isearch, ixe, ism, pload = t[0].lower(), t[1].lower(), t[2].lower(), t[3]
                if isearch in SEARCH_REFUSED:
                    raise DeckError(
                        f"{where}: search mode '{t[0]}' -- {SEARCH_REFUSED[isearch]}")
                if isearch not in SEARCH_KEYWORDS:
                    raise DeckError(
                        f"{where}: unknown search mode '{t[0]}'; MASTER 4.0 defines "
                        "keff, boron, power, extso, tin")
                search = SEARCH_KEYWORDS[isearch]

                for label, kw in (("xenon", ixe), ("samarium", ism)):
                    if kw in XENON_REFUSED:
                        raise DeckError(
                            f"{where}: {label} mode '{kw}' -- {XENON_REFUSED[kw]}")
                    if kw not in XENON_KEYWORDS:
                        raise DeckError(
                            f"{where}: unknown {label} mode '{kw}'; MASTER 4.0 "
                            "defines no, eq, tr")
                # RASBERY carries ONE xenon setting for the whole case, in
                # "default parameters".  MASTER sets it per execution state, and
                # this used to be a plain assignment: the value of the *last*
                # %EXE_STD in the deck became the setting for every state before
                # it.  A cycle whose depletion states run equilibrium xenon and
                # whose end-of-cycle re-edit runs transient therefore silently
                # depleted the entire cycle on transient xenon.
                xenon = XENON_KEYWORDS[ixe]
                xenon_states.append((xenon, ixe, card.line_no))
                if XENON_KEYWORDS[ism] != xenon:
                    report.assume(
                        f"{where}: samarium mode '{ism}' differs from xenon mode "
                        f"'{ixe}'; RASBERY drives Xe and Sm from one 'xenon' "
                        f"setting, so the samarium request is dropped and "
                        f"'{xenon}' is used for both")
                rate = as_float(pload, where) * 100.0
                entry = dict(type="standard")
                entry["search"] = search
                schedule.append([entry, dict(edt)])
                std_seen = True

            elif card.name == "%EXE_DEP":
                rows = [r for r in card.rows if r]
                if not rows:
                    raise DeckError(f"{where}: empty card")
                head = need(rows[0], 2, where)
                delt = as_float(head[0], where)
                itg = as_int(head[1], where)
                extra = rows[1:]

                if delt < 0:
                    tgt = extra[0] if extra else None
                    raise DeckError(
                        f"{where}: negative delt ({delt:g}) selects MASTER's "
                        "automatic deplete-until-target mode"
                        + (f" ({' '.join(tgt)})" if tgt else "")
                        + ".  RASBERY's schedule takes explicit step lengths only, "
                        "so the number of steps is not known ahead of the run.  "
                        "Expand the target into explicit %EXE_DEP steps.")

                # itg > 0 attaches a "tgkeff tgppm" line.  MASTER's own rule is
                # that the quantity being searched carries a negative target, so
                # the *other* number is the one that actually sets a condition.
                if itg > 0:
                    if not extra:
                        raise DeckError(
                            f"{where}: itg={itg} requires a following "
                            "'tgkeff tgppm' line")
                    tg = need(extra[0], 2, f"{where} target line")
                    tgkeff = as_float(tg[0], where)
                    tgppm = as_float(tg[1], where)
                    if search == "keff":
                        if tgkeff >= 0:
                            raise DeckError(
                                f"{where}: isearch=keff requires a negative "
                                f"tgkeff, found {tgkeff:g}")
                        boron_targets.append((tgppm, card.line_no))
                    else:   # boron search
                        if tgppm >= 0:
                            raise DeckError(
                                f"{where}: isearch=boron requires a negative "
                                f"tgppm, found {tgppm:g}")
                        keff_targets.append((tgkeff, card.line_no))
                    extra = extra[1:]

                if extra:
                    raise DeckError(
                        f"{where}: unexpected trailing data "
                        f"'{' '.join(extra[0])}'")

                if delt == 0.0:
                    # No depletion: MASTER re-edits the state it is already at.
                    # RASBERY has no "re-edit without advancing" entry, and the
                    # state itself is produced by %EXE_STD.
                    if not std_seen:
                        entry = dict(type="standard")
                        entry["search"] = search or "keff"
                        schedule.append([entry, dict(edt)])
                        std_seen = True
                        report.assume(
                            f"{where}: zero-length %EXE_DEP taken as the initial "
                            "state and given search='" + (search or "keff") +
                            "'; the deck has no %EXE_STD, so neither the state "
                            "nor its search mode is stated")
                    else:
                        report.note(
                            f"{where}: zero-length %EXE_DEP re-edits the state "
                            "already defined; no extra schedule entry emitted")
                    continue

                if rate is None:
                    raise DeckError(
                        f"{where}: %EXE_DEP before any %EXE_STD, so the power "
                        "fraction (pload) is unknown")
                entry = dict(type="depletion", steps=1, rate=rate, time=delt)
                schedule.append([entry, dict(edt)])

            elif card.name == "%EXE_ROD":
                pos = {}
                for row in card.rows:
                    need(row, 2, where, exact=True)
                    pos[row[0]] = as_float(row[1], where)
                rod_states.append((len(schedule), pos, card.line_no))

            else:
                raise DeckError(f"{where}: execution card not handled")

    if not schedule:
        raise DeckError("the deck defines no execution state (%EXE_STD/%EXE_DEP)")

    if rod_states:
        raise DeckError(
            "%EXE_ROD sets an explicit rod bank position per state (first at "
            f"line {rod_states[0][2]}).  MASTER measures that position from the "
            "bottom of the axial stack in %ROD_CFG units, resets every unlisted "
            "bank to its fully-withdrawn stop, and matches bank names by prefix; "
            "RASBERY's 'rod insertion' entry is a depth in cm from the top of the "
            "active fuel with none of those rules.  Translating it would change "
            "the rod positions, so it is refused.")

    # Same rule as the targets below, applied to the xenon mode: a RASBERY case
    # carries one, so a deck that changes it mid-cycle is not translatable.  The
    # old code just kept the last one, which meant the end-of-cycle re-edit's
    # setting was applied retroactively to every depletion step before it.
    modes = {mode for mode, _, _ in xenon_states}
    listing = ", ".join(f"'{kw}' -> {mode} (line {ln})"
                        for mode, kw, ln in xenon_states)
    if opts.xenon is not None:
        if modes and opts.xenon not in modes:
            raise DeckError(
                f"--xenon {opts.xenon} was given, but no %EXE_STD in the deck "
                f"selects it ({listing}).  Refusing to emit a mode the deck "
                "never asks for.")
        if len(modes) > 1:
            report.note(
                f"xenon '{opts.xenon}' chosen explicitly; the deck changes mode "
                f"mid-cycle ({listing}) and RASBERY carries one setting for the "
                "whole case, so the states asking for the other mode will run "
                "on this one")
        xenon = opts.xenon
    elif len(modes) > 1:
        raise DeckError(
            "%EXE_STD selects more than one xenon mode during the cycle: "
            + listing
            + ".  RASBERY carries a single 'xenon' entry in 'default "
            "parameters' that applies to the whole case, so there is no way to "
            "reproduce a per-state change; taking the last one, as this "
            "translator used to, applies the end-of-cycle setting retroactively "
            "to every step before it.  Make the %EXE_STD xenon modes agree, "
            "split the deck where it changes, or state the one to use with "
            "--xenon <transient|equilibrium> (or '!RASI xenon=...') -- which "
            "mode MASTER actually ran for most of the cycle is in its own "
            "SUMMARY EDIT 'OPTIONS' column.")

    # A target set part-way through the cycle would have to become a schedule
    # override; only a single, constant target is translatable.
    if boron_targets:
        levels = {round(v, 9) for v, _ in boron_targets}
        if len(levels) > 1:
            raise DeckError(
                "%EXE_DEP sets more than one boron level during the cycle "
                + ", ".join(f"{v:g} ppm (line {ln})" for v, ln in boron_targets)
                + ".  A RASBERY case carries one boron level in 'default "
                "parameters', so this cannot be translated.")
    if keff_targets:
        levels = {round(v, 9) for v, _ in keff_targets}
        if len(levels) > 1:
            raise DeckError(
                "%EXE_DEP sets more than one target k-eff during the cycle "
                + ", ".join(f"{v:g} (line {ln})" for v, ln in keff_targets)
                + ".  This translator emits a single 'search_target'.")

    # Attach the print blocks.  %EDT_OPT ippi>=1 requests the pin-power edit.
    out = []
    for idx, (entry, edt_state) in enumerate(schedule):
        printing = dict()
        printing["pin-wise information"] = edt_state["ippi"] >= 1
        printing["summary"] = True
        # MASTER's iwrst writes a restart file at every edit point.  RASBERY's
        # "save" dumps the full restart state; writing it at every step would
        # multiply the output size for no benefit, so the translator keeps the
        # end-of-cycle save only.  --save-every overrides.
        if edt_state["iwrst"] >= 1 and (opts.save_every or idx == len(schedule) - 1):
            printing["save"] = True
        entry["print"] = printing
        out.append(entry)

    # A boron search needs its target k-eff; RASBERY defaults it to 1.0.
    if keff_targets and abs(keff_targets[0][0] - 1.0) > 1e-12:
        out[0]["search_target"] = keff_targets[0][0]
        report.note(f"target k-eff {keff_targets[0][0]:g} from %EXE_DEP "
                    f"(line {keff_targets[0][1]})")

    boron = boron_targets[0][0] if boron_targets else None
    return out, xenon, rate, boron


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

class Report:
    """Translation diary, split by severity.

    ``note``   -- the translation is fully determined by the deck; this is
                  provenance, not a decision.
    ``assume`` -- the deck did *not* state something the case JSON has to carry,
                  and the translator supplied a physical default (or skipped a
                  verification).  The output is a guess, however reasonable, so
                  it must not pass unnoticed: main() exits 3 unless
                  --allow-assumptions was given.
    """

    def __init__(self):
        self.notes = []
        self.assumptions = []

    def note(self, text):
        self.notes.append(text)

    def assume(self, text):
        self.assumptions.append(text)

    # The stream defaults resolve at call time, not at def time.  A default of
    # `stream=sys.stderr` captures whatever sys.stderr was when this module was
    # imported, so a caller that redirects sys.stderr -- the unit tests, or
    # anything embedding the translator -- never sees the diary at all.
    def dump(self, stream=None):
        stream = stream or sys.stderr
        for n in self.notes:
            print(f"[master2rasi] note: {n}", file=stream)
        for a in self.assumptions:
            print(f"[master2rasi] ASSUME: {a}", file=stream)

    def dump_assumptions(self, stream=None):
        """Always-printed summary, even under --quiet."""
        stream = stream or sys.stderr
        if not self.assumptions:
            return
        print(f"[master2rasi] {len(self.assumptions)} assumption(s) were needed "
              "to translate this deck:", file=stream)
        for a in self.assumptions:
            print(f"[master2rasi]   - {a}", file=stream)


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------

def translate(deck, opts, report):
    dim, geo, thd = deck.dim, deck.geo, deck.thd

    for name, obj in (("%GEN_DIM", dim), ("%GEN_GEO", geo), ("%GEN_THD", thd)):
        if not obj:
            raise DeckError(f"the deck has no {name} card")
    if not deck.lp:
        raise DeckError("the deck has no %LPD_BCH loading pattern")
    if not deck.batch_axial:
        raise DeckError("the deck has no %LPD_B&C axial composition card")
    if not deck.comp_names:
        raise DeckError("the deck has no %LPD_C&X composition/XS map")

    # %GEN_DIM ngeo says how much of the core the deck already describes
    # (1 = full core).  Folding a deck that is already a partial core halves it
    # again: the map may still pass the octant-symmetry check and the rated
    # power is divided by four a second time, so the run completes and is simply
    # wrong.  --check only warned about this; refuse it here.
    if opts.fold == "quarter" and dim["ngeo"] != 1:
        raise DeckError(
            f"--fold quarter needs a full-core deck, but %GEN_DIM ngeo="
            f"{dim['ngeo']} says the deck already describes 1/{dim['ngeo']} of "
            "the core.  Folding it again would halve the geometry and the rated "
            "power a second time.  Translate it with --fold none, or supply the "
            "full-core deck.")

    # ---- axial mesh -------------------------------------------------------
    hz = geo["hz"]
    if len(hz) != dim["nz"]:
        raise DeckError(
            f"%GEN_GEO axial mesh has {len(hz)} nodes but %GEN_DIM nz={dim['nz']}")
    total = sum(hz)
    report.note(f"axial mesh: {len(hz)} nodes, total {total:g} cm "
                f"(%GEN_GEO height={geo['height']:g} cm active)")

    # %GEN_GEO lists zmesh bottom -> top.  RASBERY reverses "hz" as it reads it
    # (IO.cpp: std::reverse right after the expansion loop), so the JSON list is
    # written top -> bottom.  The batch stacks below follow the same order.
    hz_json = []
    for h in reversed(hz):
        if hz_json and abs(hz_json[-1]["height"] - h) < 1e-12:
            hz_json[-1]["node"] += 1
        else:
            hz_json.append(dict(height=h, node=1))

    # ---- loading pattern --------------------------------------------------
    grid = deck.lp
    if len(grid) != dim["ny"]:
        raise DeckError(
            f"%LPD_BCH has {len(grid)} rows but %GEN_DIM ny={dim['ny']}")
    grid = rectangular(grid)
    if len(grid[0]) != dim["nx"]:
        raise DeckError(
            f"%LPD_BCH is {len(grid[0])} columns wide but %GEN_DIM nx={dim['nx']}")

    if opts.fold == "quarter":
        core = fold_quarter(grid)
        power_factor = 0.25
        symmetry = {"angle": 90, "mirror": True,
                    "center assembly divided": True}
        report.note(f"folded {dim['nx']}x{dim['ny']} full core to a "
                    f"{len(core[0])}x{len(core)} quarter map "
                    "(octant symmetry verified)")
    elif dim["ngeo"] != 1:
        # The deck is already a partial core.  Falling through to the branch
        # below would declare a quarter map as a 360-degree full core and give
        # it the full-core rated power: a 4x power-density error that runs to
        # completion and reports a plausible k-eff.
        core, power_factor, symmetry = already_folded_core(deck, grid, report)
    else:
        core = full_core_map(grid)
        power_factor = 1.0
        symmetry = {"angle": 360, "mirror": False,
                    "center assembly divided": False}

    nx = max(len(r) for r in core)
    ny = len(core)

    used_batches = {b for row in core for b in row if b != "XX"}
    missing = sorted(used_batches - set(deck.batch_axial))
    if missing:
        raise DeckError(
            f"loading pattern uses batches with no %LPD_B&C entry: {', '.join(missing)}")

    # ---- batches ----------------------------------------------------------
    models = None
    if opts.xs and not opts.no_verify_xs:
        models = load_model_names(opts.xs)
        if models is None:
            report.assume(f"could not read model names from '{opts.xs}' "
                          "(missing h5py or not a CHIFFON library): the "
                          "composition -> model normalisation below is taken on "
                          "trust and a typo will only surface inside RASBERY")
        else:
            report.note(f"cross-section library declares {len(models)} models")

    cache = {}
    aliases = dict(opts.aliases)
    batch_json = dict()
    for batch in sorted(used_batches):
        comps = deck.batch_axial[batch]
        if len(comps) != dim["nz"]:
            raise DeckError(
                f"%LPD_B&C batch '{batch}' has {len(comps)} axial entries but nz={dim['nz']}")
        # MASTER lists the stack bottom -> top; the RASBERY batch list is
        # top -> bottom (IO::ReadInput consumes it against the reversed hz).
        stack = []
        for icomp in reversed(comps):
            if icomp not in deck.comp_names:
                raise DeckError(
                    f"%LPD_B&C batch '{batch}' uses composition {icomp} which has "
                    "no %LPD_C&X entry")
            model = resolve_model(deck.comp_names[icomp], models, aliases, cache)
            if stack and stack[-1]["id"] == model:
                stack[-1]["count"] += 1
            else:
                stack.append(dict(id=model, count=1))
        batch_json[batch] = stack

    # ---- boundary conditions ---------------------------------------------
    # %GEN_SYM: in x/y, 0 = vacuum while -1 and +1 both name a symmetry plane
    # (through the middle of an assembly, or on an assembly boundary); in z,
    # 0 = vacuum and anything else is reflective.  RASBERY takes an albedo:
    # 0.0 is reflective, and the validated i-SMR convention writes an open face
    # as the partial-current value 0.5 rather than a full Marshak 1.0.
    VACUUM_ALBEDO = 0.5

    def albedo_of(flag, face, axis):
        # Same whitelist on all three axes.  The z faces used to accept any
        # non-zero flag as "reflective", so a typo (2, 10, -3) became a
        # reflective axial boundary without a word -- the single most
        # reactivity-sensitive thing this card can say.
        if flag == 0:
            return VACUUM_ALBEDO
        if flag in (-1, 1):
            return 0.0
        raise DeckError(
            f"%GEN_SYM {face}={flag}: only 0 (vacuum), -1 and +1 (symmetry "
            f"plane) are defined on the {axis} axis")

    if deck.sym:
        albedo = {
            "west": albedo_of(deck.sym["lx"], "isymlx", "x"),
            "east": albedo_of(deck.sym["rx"], "isymrx", "x"),
            "north": albedo_of(deck.sym["ly"], "isymly", "y"),
            "south": albedo_of(deck.sym["ry"], "isymry", "y"),
            "up": albedo_of(deck.sym["rz"], "isymrz", "z"),
            "bottom": albedo_of(deck.sym["lz"], "isymlz", "z"),
        }
    else:
        albedo = {f: VACUUM_ALBEDO for f in
                  ("west", "east", "north", "south", "up", "bottom")}
        report.assume("no %GEN_SYM card: vacuum (albedo 0.5) applied to all six "
                      "faces.  The deck states no boundary condition and the "
                      "core leakage follows entirely from this choice.")

    if opts.fold == "quarter":
        # The two cut planes become the symmetry planes of the quarter model.
        albedo["west"] = 0.0
        albedo["north"] = 0.0

    # ---- feedback / state point ------------------------------------------
    if not deck.fdb:
        raise DeckError("the deck has no %GEN_FDB card, so the feedback state "
                        "is undefined")
    any_fb = any(deck.fdb.values())
    th_mode = "steady" if any_fb else "none"
    if any_fb and not all(deck.fdb.values()):
        raise DeckError(
            "%GEN_FDB enables only part of the feedback set "
            f"(tfuel={deck.fdb['tfuel']}, tmod={deck.fdb['tmod']}, "
            f"dmod={deck.fdb['dmod']}).  RASBERY's T/H feedback is all-or-nothing, "
            "so a partial selection cannot be reproduced.")

    # %GEN_THD's optional tail (mflow, hgfl) is a T/H boundary condition that
    # was parsed and then dropped without a word.  RASBERY's TH block has no
    # mass-flow entry: it derives the core flow from the rated power and the
    # inlet/outlet enthalpy difference instead.  When the T/H solve is off that
    # is genuinely irrelevant and a note is enough; when it is on, the deck
    # states a flow that the run will not use, and the coolant state RASBERY
    # computes is its own -- an assumption, and one the caller has to accept.
    if "mflow" in thd:
        detail = (f"%GEN_THD states mflow={thd['mflow']:g} and "
                  f"hgfl={thd['hgfl']:g}, which RASBERY has no input for; it "
                  "derives the core flow from the rated power and the "
                  "inlet/outlet temperatures")
        if th_mode == "steady":
            report.assume(
                detail + ".  With %GEN_FDB feedback on, the T/H solve therefore "
                "runs on a flow the deck did not ask for; check that "
                "power/(h_out - h_in) reproduces the deck's mass flow.")
        else:
            report.note(
                detail + " (TH_mode='none' here, so no T/H solve consumes it)")

    tmod = thd["tavg"] + KELVIN

    tfuel = opts.tfuel
    if tfuel is None:
        raise DeckError(
            "the fuel temperature is not carried by any MASTER card.  With "
            "%GEN_FDB fuel feedback off, MASTER holds the fuel at the "
            "cross-section library's reference point, which the deck does not "
            "state.  Supply it with --tfuel <K> or a '!RASI tfuel=<K>' line in "
            "the deck (the DeCART reference branch value is the right choice).")

    # ---- assembly ---------------------------------------------------------
    case = dict()
    case["data"] = {"cross-section": opts.xs_ref}

    dims = {"ng": dim["ng"], "nx": nx, "ny": ny, "nz": dim["nz"]}
    if deck.pin:
        dims["npins"] = deck.pin["npin"]
    dims["xydivision"] = dim["ndivxy"]

    case["geometry"] = {
        "dimensions": dims,
        "size": {"hx": geo["wide"], "hy": geo["wide"], "hz": hz_json},
        "symmetry": symmetry,
        "albedo": albedo,
    }

    schedule, xenon, rate, deck_boron = build_schedule(deck, opts, report)

    boron = opts.boron
    if deck_boron is not None:
        if opts.boron_explicit and abs(deck_boron - opts.boron) > 1e-9:
            raise DeckError(
                f"the deck sets boron to {deck_boron:g} ppm (%EXE_DEP target) "
                f"but --boron {opts.boron:g} was given; refusing to pick one")
        boron = deck_boron
        report.note(f"boron {boron:g} ppm taken from the %EXE_DEP target line")

    defaults = {
        "boron_ppm": boron,
        "fuel_temperature": tfuel,
        "moderator_temperature": tmod,
    }
    if xenon is not None:
        defaults["xenon"] = xenon
    case["default parameters"] = defaults

    case["TH"] = {
        "TH_mode": th_mode,
        "global pressure": thd["press"] / 10.0,          # bar -> MPa
        "inlet temperature": thd["tin"] + KELVIN,
        "outlet temperature": thd["tin"] + thd["trise"] + KELVIN,
        "rated power": thd["power"] * power_factor,
    }

    case["convergence"] = {
        "max_eigen_iterations": opts.max_eigen_iterations,
        "eigv_tolerance": opts.eigv_tolerance,
    }

    case["batch"] = batch_json
    case["core"] = core

    # ---- rods -------------------------------------------------------------
    if deck.rod_cfg or deck.rod_map:
        case["rod configuration"], case["rod map"] = translate_rods(
            deck, opts, report, core)

    case["schedule"] = schedule

    report.note(f"schedule: {len(schedule)} states "
                f"({sum(1 for s in schedule if s['type'] == 'depletion')} depletion)")
    report.note(f"rated power {case['TH']['rated power']:g} MW "
                f"(deck {thd['power']:g} MW x {power_factor})")

    return case


def translate_rods(deck, opts, report, core):
    """Translate %ROD_CFG/%ROD_MAP into the RASBERY rod blocks."""
    if not deck.rod_cfg:
        raise DeckError("%ROD_MAP is present but %ROD_CFG is missing")
    if not deck.rod_map:
        raise DeckError("%ROD_CFG is present but %ROD_MAP is missing")

    grid = rectangular(deck.rod_map)
    if opts.fold == "quarter":
        rod_map = fold_quarter(grid, what="rod map")
    else:
        rod_map = full_core_map(grid, what="rod map")

    # RASBERY's rod map must line up with the core map row for row.
    if len(rod_map) != len(core):
        raise DeckError(
            f"rod map has {len(rod_map)} rows but the core map has {len(core)}")
    for r, (rrow, crow) in enumerate(zip(rod_map, core)):
        if len(rrow) < len(crow):
            rrow.extend(["XX"] * (len(crow) - len(rrow)))
        elif len(rrow) > len(crow):
            raise DeckError(
                f"rod map row {r+1} is wider ({len(rrow)}) than the core row "
                f"({len(crow)})")
    rod_map = [[("XX" if v == VOID else v) for v in row] for row in rod_map]

    used = {v for row in rod_map for v in row if v != "XX"}
    missing = sorted(used - set(deck.rod_cfg))
    if missing:
        raise DeckError(
            f"%ROD_MAP uses rod groups with no %ROD_CFG entry: {', '.join(missing)}")

    # %ROD_CFG describes the physical rodlet (absorber material and lengths) and
    # a pair of positions; RASBERY wants an absorber ctype plus the bank overlap
    # profile that a rod search walks.  The overlap sequence is a reload-design
    # decision that the MASTER card does not carry, so it must be supplied.
    if not opts.rod_profile:
        raise DeckError(
            "%ROD_CFG gives each bank an absorber material and stroke, but "
            "RASBERY's 'rod configuration' needs the bank *overlap profile* "
            "(the position each bank takes at each step of the withdrawal "
            "sequence).  A MASTER deck does not carry that sequence.  Supply it "
            "with --rod-profile 'BANK=p1,p2,...' (repeatable) or a "
            "'!RASI rod_profile BANK=p1,p2,...' directive.")

    cfg = dict()
    for name in sorted(used):
        entry = deck.rod_cfg[name]
        if name not in opts.rod_profile:
            raise DeckError(f"no --rod-profile supplied for rod group '{name}'")
        cfg[name] = {
            "ctype": entry["matabs"],
            "profile": opts.rod_profile[name],
        }
        report.note(f"rod group '{name}': ctype={entry['matabs']} from %ROD_CFG "
                    f"matabs, profile from --rod-profile")

    return cfg, rod_map


# --------------------------------------------------------------------------
# Coverage report
# --------------------------------------------------------------------------

def run_check(deck, opts, directives, stream=None):
    """Compatibility report: what this deck needs before it will translate.

    Runs before any translation and never raises, so a deck with several
    problems reports all of them in one pass.  Returns 0 if the deck is
    translatable as invoked, 1 otherwise.

    *stream* resolves at call time; see Report.dump().
    """
    stream = stream or sys.stdout
    P = lambda *a: print(*a, file=stream)
    blocking = []
    warnings = []

    P(f"MASTER deck compatibility report -- {opts.deck}")
    P(f"(master2rasi {VERSION})")
    P("")

    # --- 1. cards ---------------------------------------------------------
    rows = coverage_table(deck)
    width = max(len(r[0]) for r in rows) if rows else 8
    P("[1] cards")
    for name, n, status, why in rows:
        P(f"    {name.ljust(width)}  x{n:<3d}  {status}"
          + (f"  -- {why}" if why else ""))
    refused = [r for r in rows if r[2] == "REFUSED"]
    if refused:
        blocking.append(f"{len(refused)} unsupported card(s): "
                        + ", ".join(r[0] for r in refused))
    for b in getattr(deck, "blockers", []):
        if not any(b.endswith(r[3]) for r in refused):
            blocking.append(b)
    P("")

    # --- 2. geometry ------------------------------------------------------
    P("[2] geometry")
    if deck.dim:
        P(f"    {deck.dim['nx']} x {deck.dim['ny']} assemblies, "
          f"{deck.dim['nz']} axial nodes, {deck.dim['ng']} groups, "
          f"ndivxy={deck.dim['ndivxy']}")
        P(f"    ngeo={deck.dim['ngeo']} "
          f"({'full core' if deck.dim['ngeo'] == 1 else str(deck.dim['ngeo']) + 'th core'})"
          f", nbatch={deck.dim['nbatch']}, ncomp={deck.dim['ncomp']}")
        if deck.dim["ndim"] != 3:
            warnings.append(f"ndim={deck.dim['ndim']} (this translator targets 3-D decks)")
        if deck.dim["ngeo"] != 1:
            if opts.fold == "quarter":
                # translate() refuses this combination outright, so --check must
                # report it as blocking rather than as advice.
                blocking.append(
                    f"ngeo={deck.dim['ngeo']}: the deck is already a partial "
                    "core, so --fold quarter would fold it a second time")
            else:
                try:
                    already_folded_core(deck, rectangular(deck.lp), Report())
                    warnings.append(
                        f"ngeo={deck.dim['ngeo']}: translated as an "
                        "already-folded quarter core (sector symmetry and "
                        "1/4 rated power derived from %GEN_DIM and %GEN_SYM)")
                except DeckError as e:
                    blocking.append(str(e))
    else:
        blocking.append("no %GEN_DIM card")
    if deck.geo:
        P(f"    pitch {deck.geo['wide']:g} cm, active height {deck.geo['height']:g} cm, "
          f"axial stack {sum(deck.geo['hz']):g} cm over {len(deck.geo['hz'])} nodes")
    P("")

    # --- 3. symmetry / fold ----------------------------------------------
    P("[3] loading pattern and fold")
    if not deck.lp:
        blocking.append("no %LPD_BCH loading pattern")
    else:
        grid = rectangular(deck.lp)
        P(f"    map {len(grid[0])} x {len(grid)}, "
          f"{sum(1 for r in grid for c in r if c != VOID)} occupied positions")
        try:
            check_octant_symmetry(grid)
            foldable, why = True, "octant-symmetric"
        except DeckError as e:
            foldable, why = False, str(e).split(".  First")[0]
        P(f"    --fold quarter : {'AVAILABLE' if foldable else 'NOT AVAILABLE'}  ({why})")
        if opts.fold == "quarter" and not foldable:
            blocking.append("--fold quarter requested but the map is not octant-symmetric")
        try:
            full_core_map(grid)
            P("    --fold none    : AVAILABLE  (gaps are leading or trailing only)")
        except DeckError as e:
            P(f"    --fold none    : NOT AVAILABLE  ({e})")
            if opts.fold != "quarter":
                blocking.append("the map cannot be expressed full-core; use --fold quarter")
    P("")

    # --- 4. external inputs ----------------------------------------------
    P("[4] inputs a MASTER deck cannot carry")
    xs = opts.xs or directives.get("xs")
    P(f"    cross-section library : {xs or 'MISSING -- pass --xs or !RASI xs='}")
    if not xs:
        blocking.append("no cross-section library (--xs)")
    elif not os.path.exists(xs):
        warnings.append(f"cross-section library '{xs}' does not exist at this path")
    else:
        models = load_model_names(xs)
        if models is None:
            warnings.append(f"cannot read model names from '{xs}' (h5py missing?)")
        else:
            P(f"    library models        : {len(models)} -- {', '.join(sorted(models))}")
            cache, unresolved = {}, []
            for icomp, mname in sorted(deck.comp_names.items()):
                try:
                    resolve_model(mname, models, opts.aliases, cache)
                except DeckError:
                    unresolved.append(mname)
            if unresolved:
                blocking.append("composition(s) with no matching library model: "
                                + ", ".join(unresolved))
            elif deck.comp_names:
                P(f"    composition mapping   : all {len(deck.comp_names)} resolved "
                  + "(" + ", ".join(f"{k}->{v}" for k, v in sorted(cache.items())) + ")")

    tf = opts.tfuel if opts.tfuel is not None else directives.get("tfuel")
    P(f"    fuel temperature [K]  : {tf or 'MISSING -- pass --tfuel or !RASI tfuel='}")
    if tf is None:
        blocking.append("no fuel temperature (--tfuel)")
    P("")

    # --- 5. verdict -------------------------------------------------------
    if warnings:
        P("[5] warnings")
        for w in warnings:
            P(f"    - {w}")
        P("")
    P("[verdict] " + ("TRANSLATABLE as invoked" if not blocking else "BLOCKED"))
    for b in blocking:
        P(f"    - {b}")
    return 1 if blocking else 0


def coverage_table(deck):
    seen = dict()
    for name, line in deck.seen_cards:
        seen.setdefault(name, []).append(line)

    rows = []
    for name, lines in seen.items():
        if name in TRANSLATED_CARDS:
            status, why = "translated", ""
        elif name in CHECKED_CARDS:
            status, why = "checked (no output)", CHECKED_CARDS[name]
        elif name in INERT_CARDS:
            status, why = "accepted (inert)", INERT_CARDS[name]
        else:
            status, why = "REFUSED", REFUSED_CARDS.get(name, "unknown card")
        rows.append((name, len(lines), status, why))
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_rod_profile(specs):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise DeckError(f"--rod-profile expects BANK=p1,p2,...  got '{spec}'")
        name, vals = spec.split("=", 1)
        try:
            out[name.strip()] = [float(v) for v in vals.split(",") if v.strip()]
        except ValueError:
            raise DeckError(f"--rod-profile '{spec}': positions must be numbers")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="master2rasi.py",
        description="Translate a MASTER card deck into a RASBERY case JSON.")
    ap.add_argument("deck", help="MASTER input deck (e.g. depf_01.inp)")
    ap.add_argument("--xs", help="CHIFFON cross-section library (.h5)")
    ap.add_argument("--xs-ref", help="value written to data/cross-section "
                    "(defaults to the --xs path as given)")
    ap.add_argument("-o", "--output", help="output JSON (default: stdout)")
    ap.add_argument("--fold", choices=["none", "quarter"], default="none",
                    help="reduce an octant-symmetric full-core deck to a "
                         "quarter core (default: translate the deck as written)")
    ap.add_argument("--tfuel", type=float,
                    help="fixed fuel temperature [K]; required because no "
                         "MASTER card carries it")
    ap.add_argument("--boron", type=float, default=None,
                    help="initial boron concentration [ppm] (default 0)")
    ap.add_argument("--xenon", choices=["transient", "equilibrium"],
                    help="the single xenon mode to emit when the deck's "
                         "%%EXE_STD cards disagree (RASBERY carries one per "
                         "case); must be a mode the deck actually asks for")
    ap.add_argument("--max-eigen-iterations", type=int, default=100)
    ap.add_argument("--eigv-tolerance", type=float, default=1.0e-6)
    ap.add_argument("--alias", action="append", default=[],
                    metavar="MASTER=MODEL",
                    help="map a %%LPD_C&X XS-set name onto a library model name")
    ap.add_argument("--rod-profile", action="append", default=[],
                    metavar="BANK=p1,p2,...",
                    help="bank overlap profile for a %%ROD_CFG group")
    ap.add_argument("--save-every", action="store_true",
                    help="request a restart save at every state rather than "
                         "only at the end of the cycle")
    ap.add_argument("--no-verify-xs", action="store_true",
                    help="do not open the library to check model names")
    ap.add_argument("--allow-assumptions", action="store_true",
                    help="accept a translation that needed a physical default "
                         "the deck does not state (otherwise exit 3)")
    ap.add_argument("--check", action="store_true",
                    help="report card coverage, fold feasibility and missing "
                         "inputs without translating; exit 1 if blocked")
    ap.add_argument("--coverage", action="store_true",
                    help="print just the card coverage table and exit")
    ap.add_argument("--quiet", action="store_true", help="suppress notes")
    opts = ap.parse_args(argv)

    try:
        with open(opts.deck, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        print(f"master2rasi: cannot read deck: {e}", file=sys.stderr)
        return 2

    report = Report()
    try:
        items, directives = parse_deck(text)

        # Deck directives are defaults; explicit CLI options win.
        aliases = dict(directives.pop("_aliases", {}))
        for spec in opts.alias:
            if "=" not in spec:
                raise DeckError(f"--alias expects MASTER=MODEL, got '{spec}'")
            k, v = spec.split("=", 1)
            aliases[k.strip()] = v.strip()
        opts.aliases = aliases

        rod_profile = {}
        for key in list(directives):
            if key == "rod_profile":
                rod_profile.update(parse_rod_profile([directives[key]]))
        rod_profile.update(parse_rod_profile(opts.rod_profile))
        opts.rod_profile = rod_profile

        if opts.xs is None and "xs" in directives:
            opts.xs = directives["xs"]
        if opts.tfuel is None and "tfuel" in directives:
            opts.tfuel = float(directives["tfuel"])
        # --boron defaults to None so "was it given?" is a property of the
        # parsed options, not something re-derived by scanning argv for the
        # option spelling (which missed '-b'-style abbreviations argparse
        # accepts).
        opts.boron_explicit = opts.boron is not None
        if "boron" in directives and not opts.boron_explicit:
            opts.boron = float(directives["boron"])
            opts.boron_explicit = True
        if opts.boron is None:
            opts.boron = 0.0
        if "fold" in directives and opts.fold == "none":
            opts.fold = directives["fold"]
            if opts.fold not in ("none", "quarter"):
                raise DeckError(f"!RASI fold={opts.fold}: expected none|quarter")
        if opts.xenon is None and "xenon" in directives:
            opts.xenon = directives["xenon"]
            if opts.xenon not in ("transient", "equilibrium"):
                raise DeckError(
                    f"!RASI xenon={opts.xenon}: expected transient|equilibrium")

        unknown = set(directives) - {"xs", "tfuel", "boron", "fold", "xenon"}
        if unknown:
            raise DeckError("unknown !RASI directive(s): " + ", ".join(sorted(unknown)))

        if opts.check:
            deck = collect(items, strict=False)
            return run_check(deck, opts, directives)

        deck = collect(items)

        if opts.coverage:
            rows = coverage_table(deck)
            width = max(len(r[0]) for r in rows)
            print(f"{'card'.ljust(width)}  count  status")
            for name, n, status, why in rows:
                print(f"{name.ljust(width)}  {n:>5}  {status}"
                      + (f"  -- {why}" if why else ""))
            return 0

        if not opts.xs:
            raise DeckError(
                "no cross-section library given.  A MASTER deck cannot name a "
                "CHIFFON library, so pass --xs <lib.h5> or add a "
                "'!RASI xs=<lib.h5>' line to the deck.")
        opts.xs_ref = opts.xs_ref or opts.xs

        case = translate(deck, opts, report)

    except DeckError as e:
        print(f"master2rasi: {opts.deck}: {e}", file=sys.stderr)
        return 1

    # allow_nan=False: NaN/Infinity are not JSON.  as_float() already refuses
    # them at the deck boundary; this is the belt on the arithmetic done since.
    try:
        text_out = json.dumps(case, indent=2, allow_nan=False) + "\n"
    except ValueError as e:
        print(f"master2rasi: {opts.deck}: refusing to write a non-finite value "
              f"into the case JSON ({e})", file=sys.stderr)
        return 1
    if opts.output:
        with open(opts.output, "w") as fh:
            fh.write(text_out)
        if not opts.quiet:
            print(f"[master2rasi] wrote {opts.output}", file=sys.stderr)
    else:
        sys.stdout.write(text_out)

    if not opts.quiet:
        report.dump()

    # Exit 3 -- distinct from 0 (clean) and 1 (refused) -- when the case JSON
    # only exists because the translator filled in something the deck does not
    # state.  The JSON is written either way: the caller may well accept the
    # defaults, but it has to say so.  rasberry-masi propagates the code.
    if report.assumptions and not opts.allow_assumptions:
        if opts.quiet:
            report.dump_assumptions()
        print("[master2rasi] refusing to report success: re-run with "
              "--allow-assumptions to accept the assumption(s) above "
              "(exit 3)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
