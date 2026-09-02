# -*- coding: utf-8 -*-
# Pin power RMS/max %% comparison: RASBERY h5 vs MASTER PPI.
# Standalone extraction of the BOC-only logic from make_v2_master_cmp.py
# (no plotting / matplotlib dependency), for the v2 re-freeze protocol.
#
# WP24.  THREE THINGS CHANGED AND ONE DID NOT.
#
#   * `--envelope {production,screen100}` (tools/gate_b_envelope.py) and a
#     NONZERO EXIT on a breach.  This tool used to print one line and exit 0
#     unconditionally, so "Gate B pin passed" was a person comparing a number
#     against a figure in a design document from memory.
#   * `--ppi-step STEP=PATH` and `--all-steps`.  The BOC-only reading is the
#     wrong instrument for a staged or relaxed-tolerance arm: because POLISH
#     restores the production tolerance before anything is published, a staged
#     run's pin error is trajectory divergence through burnup rather than
#     convergence error, and the measured scan found arms whose deviation
#     exists ONLY at statepoints 28-33.  A gate that reads step 0001 cannot see
#     that.
#   * The two positional arguments keep their ORDER and their meaning, so every
#     existing invocation still runs and still scores exactly the BOC step it
#     always scored.
#
# WHAT DID NOT CHANGE: the statistic.  Each side is normalised by its own mean
# over positive pins (so this compares SHAPE, not power level), a pin enters the
# population only where MASTER exceeds 0.05, `rms` is the root-mean-square of
# the relative error in % over that population and `max` is the single worst
# pin in it -- a max, not a percentile and not an assembly average.
#
# ------------------------------------------------------------------------
# WHY `--all-steps` DOES NOT DECIDE THE EXIT CODE (review fix)
# ------------------------------------------------------------------------
# The first draft of this flag parsed ONE MASTER PPI and compared every RASBERY
# statepoint against it, then took the verdict from the worst step.  The PPI
# every documented invocation passes is a BOC distribution
# (`kngr_mas_ppi_boc.txt`, docs/V5_FREEZE Sec 8.4, docs/WP20 Sec), so that
# scored end-of-cycle pin shapes against a beginning-of-cycle reference.  A PWR
# pin power distribution redistributes by several to tens of percent between BOC
# and EOC for ordinary depletion reasons, so `--all-steps --envelope screen100`
# would have returned FAIL on every healthy run, for reasons with nothing to do
# with the preset.  A gate that cannot pass gets ignored, which is operationally
# identical to the unconditional `exit 0` it replaced -- and the runbook made
# that invocation the screen100 verdict.
#
# So the rule here is like-for-like or nothing:
#
#   SCORED   a statepoint compared against ITS OWN MASTER reference.  The
#            positional `ppi` is the reference for `--step` (default 0001), and
#            `--ppi-step 0018=/path/eoc.txt` adds more.  Only these reach
#            gate_b_envelope, and only these move the exit code.
#   DRIFT    every other statepoint under `--all-steps`, measured against the
#            positional PPI and REPORTED.  This is the reading the late-burnup
#            failure mode shows up in and it is worth printing; it is not an
#            acceptance number and it never touches the verdict.
#
# The late-burnup failure mode `--all-steps` exists to catch is real.  Catching
# it properly needs a MASTER PPI per statepoint, which is what `--ppi-step`
# takes; until a deck has them, the drift column is what there is to read.
#
# ------------------------------------------------------------------------
# LATTICE AND SEAT LABELLING ARE NOW ARGUMENTS (2026-09-03)
# ------------------------------------------------------------------------
# This parser was written against KNGR and hard-coded all three of its
# conventions: a 16x16 pin lattice (`nums.size // 256`, `.reshape(npl, 16, 16)`),
# the PWR column alphabet that skips I/O/Q, and the seat J9 as the quarter-core
# origin.  None of the three holds for i-SMR, and the first one failed in the
# worst possible way: an i-SMR MAS_PPI carries 24 planes x 17 x 17 = 6936 numbers
# per seat, 6936 // 256 = 27, and the old line reshaped 6912 of them into 27
# imaginary 16x16 planes and reported an rms off that.  A wrong number, printed
# confidently, in the units of the right one.
#
# So: `--lattice` (the pin pitch count), `--origin` (the seat that is quarter
# index (0,0)) and `--col-labels`.  Every default reproduces the KNGR reading
# byte for byte, and the two failure modes that used to be silent -- a pin block
# that is not a whole number of planes, and a MASTER seat whose lattice differs
# from the RASBERY one -- are now refusals that name `--lattice`.
#
# The alphabet has a fallback rather than a required flag because MASTER does not
# use one convention.  The i-SMR full-core PPI labels its nine columns A..I --
# including the `I` that the PWR alphabet exists to skip -- so a label outside
# XL is the file telling us which convention it used, and plain A-Z is the only
# other one there is.  Both alphabets are alphabetically ordered, so a label
# common to both keeps its offset from a common origin either way.
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_b_envelope  # noqa: E402

#: The PWR column alphabet: I, O and Q are skipped so a seat name cannot be
#: misread as a digit.  KNGR's PPI uses it; i-SMR's does not.
XL = "ABCDEFGHJKLMNPRST"
#: The other convention MASTER prints.  Used automatically when a FANAME label
#: is not in XL, which can only be I, O or Q.
ASCII_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: KNGR's quarter-core origin, and this tool's default so every existing
#: two-argument invocation scores exactly what it scored before.
DEFAULT_ORIGIN = "J9"
DEFAULT_LATTICE = 16

ORIGIN_RE = re.compile(r"^([A-Z]+)(\d+)$")


def resolve_labels(seen: set[str], explicit: str | None) -> tuple[str, str]:
    """(alphabet, why).  Explicit wins; otherwise XL unless the file rules it out."""
    if explicit:
        missing = sorted(s for s in seen if s not in explicit)
        if missing:
            raise SystemExit(
                f"--col-labels {explicit!r} has no place for FANAME column(s) "
                f"{', '.join(missing)} in this PPI")
        return explicit, "explicit --col-labels"
    outside = sorted(s for s in seen if s not in XL)
    if not outside:
        return XL, "PWR alphabet (I/O/Q skipped)"
    missing = sorted(s for s in seen if s not in ASCII_LABELS)
    if missing:
        raise SystemExit(
            f"FANAME column(s) {', '.join(missing)} are in neither the PWR "
            f"alphabet nor A-Z; pass --col-labels")
    return ASCII_LABELS, (f"plain A-Z (column {', '.join(outside)} is not in the "
                          f"PWR alphabet, so this PPI does not use it)")


def resolve_origin(origin: str, alphabet: str, seen_cols: set[str],
                   seen_rows: set[int]) -> tuple[int, int, str]:
    """(column index, row number, why) for the seat that is quarter index (0,0).

    `auto` takes the MIDPOINT of the labels and rows the file actually contains,
    which is the core centre for a FULL-core PPI and is WRONG for a quarter-core
    one (whose origin is its first seat).  That is why it is not the default:
    both shapes exist in the archive, and only the file's author knows which.
    """
    if origin.lower() == "auto":
        idx = sorted(alphabet.index(c) for c in seen_cols)
        rows = sorted(seen_rows)
        ocol = (idx[0] + idx[-1]) // 2
        orow = (rows[0] + rows[-1]) // 2
        return ocol, orow, (f"auto -> {alphabet[ocol]}{orow}, the midpoint of "
                            f"{alphabet[idx[0]]}..{alphabet[idx[-1]]} / "
                            f"{rows[0]}..{rows[-1]} (a FULL-core PPI centre; on a "
                            f"quarter-core PPI pass the corner seat instead)")
    m = ORIGIN_RE.match(origin.strip().upper())
    if not m:
        raise SystemExit(f"--origin {origin!r} is not a seat like J9, E5 or `auto`")
    label, row = m.group(1), int(m.group(2))
    if label not in alphabet:
        raise SystemExit(
            f"--origin {origin!r}: column {label} is not in the alphabet this "
            f"PPI uses ({alphabet[:12]}...); pass --col-labels")
    return alphabet.index(label), row, f"explicit --origin {label}{row}"


def parse_ppi(path, npin=DEFAULT_LATTICE, origin=DEFAULT_ORIGIN, col_labels=None,
              verbose=False):
    ppi = open(path, errors="replace").read()
    heads = list(re.finditer(r"^\s*FANAME\s+([A-Z]+)\s+(\d+)\s+(\d+)\s", ppi, re.M))
    if not heads:
        raise SystemExit(f"{path}: no FANAME seat header found")
    seen_cols = {h.group(1) for h in heads}
    seen_rows = {int(h.group(2)) for h in heads}
    alphabet, why_labels = resolve_labels(seen_cols, col_labels)
    ocol, orow, why_origin = resolve_origin(origin, alphabet, seen_cols, seen_rows)
    block = npin * npin
    if verbose:
        print(f"ppi {path}: {len(heads)} seats, lattice {npin}x{npin}, "
              f"labels {why_labels}, origin {why_origin}")
    mas_pin = {}
    for hi, h in enumerate(heads):
        body = ppi[h.end(): heads[hi + 1].start() if hi + 1 < len(heads) else len(ppi)]
        col = alphabet.index(h.group(1)) - ocol
        row = int(h.group(2)) - orow
        m = re.search(r"PIN 3-D POWER DISTRIBUTION[^\n]*\n(.*)\Z", body, re.S)
        if not m:
            continue
        seg = m.group(1)
        stop = re.search(r"\n[A-Z][A-Z \-0-9]+ DISTRIBUTION|\n\*{10,}", seg)
        if stop:
            seg = seg[: stop.start()]
        nums = np.array([float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", seg)])
        npl = nums.size // block
        if npl == 0:
            continue
        if nums.size % block:
            # THE FAILURE THIS REPLACES was silent: 6936 numbers // 256 = 27, and
            # the tool went on to average 27 fabricated planes and print an rms.
            guess = ""
            for cand in range(2, 33):
                if nums.size % (cand * cand) == 0:
                    guess += f" {cand}"
            raise SystemExit(
                f"{path}: seat {h.group(1)}{h.group(2)} has {nums.size} pin values, "
                f"which is not a whole number of {npin}x{npin} planes. Lattice "
                f"size(s) that divide it:{guess or ' (none)'}. Pass --lattice.")
        mas_pin[(row, col)] = nums.reshape(npl, npin, npin).mean(axis=0)
    return mas_pin


def half_pin_cuts(ras_map, ras_pin, npin):
    """{seat: (cut_row, cut_col)} for every RASBERY map that is a HALF assembly.

    A deck with `center assembly divided` folds its quarter through the MIDDLE
    of the centre assemblies, so the seats on the two cut lines hold half (or,
    at the corner, a quarter) of an assembly.  In the .h5 the absent half is
    blank and THE PINS ON THE CUT LINE ITSELF ARE HALF-PINS carrying half the
    power.  MASTER prints the whole assembly and whole pins at those same seats.

    Measured on the i-SMR HIGA BOC anchor, comparing the two directly makes the
    17 cut-line pins of each centre seat wrong by very nearly exactly 50 %,
    which lifts those seats from ~0.6 % rms to 15-21 % and the whole-core
    reading from 4.66 % to 9.51 % rms / 24.8 % to 50.1 % max.  That is an
    arithmetic artefact of the fold, reported in the units of a pin power error.

    Detection is from the DATA -- a seat whose populated rows start or end at the
    centre row -- and not from the deck, because this tool never reads the deck.
    Even lattices are excluded: with no centre pin row there are no half-pins.
    """
    if npin % 2 == 0:
        return {}
    mid = (npin - 1) // 2
    cuts = {}
    for seat, la in ras_map.items():
        rp = ras_pin[la]
        live = np.isfinite(rp) & (rp > 0)
        if not live.any():
            continue
        rows = np.flatnonzero(live.any(axis=1))
        cols = np.flatnonzero(live.any(axis=0))
        cut_row = rows.size < npin and mid in rows and mid in (rows[0], rows[-1])
        cut_col = cols.size < npin and mid in cols and mid in (cols[0], cols[-1])
        if cut_row or cut_col:
            cuts[seat] = (bool(cut_row), bool(cut_col))
    return cuts


def apply_half_pin_correction(ras_pin, ras_map, cuts, npin):
    """Restore whole-pin power on the cut lines, so MASTER's pins are the same pins.

    Doubling, not masking: the half-pin's power really is half of the pin
    MASTER prints, so scaling recovers a like-for-like comparison instead of
    discarding 17 real comparison points per centre seat.  The corner seat is
    cut both ways and its centre pin is a quarter-pin, which falls out of
    applying both factors.
    """
    mid = (npin - 1) // 2
    out = ras_pin.copy()
    for seat, (cut_row, cut_col) in cuts.items():
        la = ras_map[seat]
        block = out[la].copy()
        if cut_row:
            block[mid, :] *= 2.0
        if cut_col:
            block[:, mid] *= 2.0
        out[la] = block
    return out


def pin_err_stats(ras_map, mas_pin, ras_pin, nya=None, nxa=None):
    mas_all, ras_all = [], []
    for (r, c), mp in mas_pin.items():
        la = ras_map.get((r, c))
        if la is None:
            continue
        rp = ras_pin[la]
        if not np.any(rp > 0):
            continue
        if mp.shape != rp.shape:
            raise SystemExit(
                f"seat at quarter index ({r},{c}): MASTER pin map is "
                f"{mp.shape[0]}x{mp.shape[1]} and the RASBERY one is "
                f"{rp.shape[0]}x{rp.shape[1]}. These are different lattices; "
                f"pass --lattice {rp.shape[0]}.")
        mas_all.append(mp)
        ras_all.append(rp)
    if not mas_all:
        raise SystemExit(
            "no MASTER seat maps onto a RASBERY assembly. The quarter-core "
            "origin is almost certainly wrong for this PPI -- pass --origin "
            "(the seat that is quarter index (0,0)).")
    mnorm = np.mean([a[a > 0].mean() for a in mas_all])
    rnorm = np.mean([a[a > 0].mean() for a in ras_all])
    # THE QUADRANT BOUND COMES FROM THE .h5, not from KNGR's 9x9 quarter.  The
    # old literal `r > 8 or c > 8` was a restatement of one deck's nya/nxa, and
    # on a core with a larger quarter map it would have silently dropped real
    # seats from the population while looking like a symmetry guard.
    rmax = (nya - 1) if nya else None
    cmax = (nxa - 1) if nxa else None
    errs = []
    for (r, c), mp in mas_pin.items():
        la = ras_map.get((r, c))
        if la is None:
            continue
        if rmax is not None and (r > rmax or c > cmax):
            continue
        rp = ras_pin[la]
        if not np.any(rp > 0):
            continue
        e = np.where((mp > 0.05) & (rp > 0), (rp / rnorm) / (mp / mnorm) - 1.0, np.nan) * 100
        errs.append(e[np.isfinite(e)])
    allerr = np.concatenate(errs)
    return float(np.sqrt(np.nanmean(allerr ** 2))), float(np.nanmax(np.abs(allerr)))


def parse_ppi_step(spec: str) -> tuple[str, str]:
    """`--ppi-step 0018=/path/x.txt` -> ("0018", "/path/x.txt")."""
    step, sep, path = spec.partition("=")
    if not sep or not step.strip() or not path.strip():
        raise SystemExit(
            f"--ppi-step {spec!r} is not STEP=PATH. A per-statepoint reference has to "
            f"name the statepoint it is the reference FOR, or this tool would be back "
            f"to scoring burnup against a BOC map.")
    return step.strip(), path.strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate B pin power RMS/max comparison, RASBERY h5 vs MASTER PPI")
    ap.add_argument("h5")
    ap.add_argument("ppi", help="MASTER PPI for --step (the BOC map, by convention)")
    ap.add_argument("--step", default="0001",
                    help="RASBERY statepoint group the positional PPI is the reference "
                         "for, and the one SCORED by default (default: %(default)s, BOC)")
    ap.add_argument("--ppi-step", action="append", default=[], metavar="STEP=PATH",
                    help="an additional MASTER PPI and the statepoint it is the "
                         "reference for. Repeatable. Each one adds a SCORED statepoint: "
                         "like-for-like is the only comparison that can carry a verdict.")
    ap.add_argument("--all-steps", action="store_true",
                    help="ALSO report every other statepoint in the h5 against the "
                         "positional PPI. That is a BOC-referenced DRIFT reading -- it is "
                         "where the measured late-burnup failure mode (statepoints 28-33) "
                         "shows up -- and it is REPORT ONLY: BOC-to-EOC pin "
                         "redistribution is tens of percent of real physics, so scoring "
                         "it against a 1 %% envelope would fail every healthy run. The "
                         "verdict comes from the referenced statepoints; give more with "
                         "--ppi-step.")
    ap.add_argument("--lattice", type=int, default=DEFAULT_LATTICE, metavar="N",
                    help="pins per assembly side in the MASTER PPI (default: "
                         "%(default)s, KNGR). i-SMR is 17. Getting this wrong "
                         "used to produce a confident number off a reshape of "
                         "the wrong plane count; it is now a refusal.")
    ap.add_argument("--origin", default=DEFAULT_ORIGIN, metavar="SEAT",
                    help="the MASTER seat that is RASBERY quarter index (0,0) "
                         "(default: %(default)s, KNGR; i-SMR full-core PPI is "
                         "E5). `auto` takes the midpoint of the seats present, "
                         "which is the centre of a FULL-core PPI only.")
    ap.add_argument("--half-pin-correct", action="store_true",
                    help="restore whole-pin power on the fold's cut lines "
                         "before comparing. REQUIRED on a `center assembly "
                         "divided` deck such as i-SMR: without it the 17 "
                         "cut-line pins of each centre seat are half-pins "
                         "compared against MASTER's whole pins, worth 9.51 %% "
                         "vs 4.66 %% rms on the HIGA BOC anchor. OFF by default "
                         "because turning it on would move KNGR's frozen "
                         "production figures; the tool WARNS when it detects "
                         "cut seats and the flag is absent.")
    ap.add_argument("--col-labels", default=None, metavar="ALPHABET",
                    help="column-label alphabet. Default is the PWR one "
                         "(I/O/Q skipped) with an automatic fall back to plain "
                         "A-Z when the PPI uses a label the PWR alphabet has "
                         "no place for -- i-SMR's ninth column really is `I`.")
    gate_b_envelope.add_envelope_argument(ap)
    args = ap.parse_args()
    if args.lattice < 1:
        raise SystemExit(f"--lattice {args.lattice} is not a lattice size")

    envelope = gate_b_envelope.resolve(args.envelope)

    # step -> MASTER PPI path.  The positional argument is the reference for
    # --step, which keeps every existing two-argument invocation scoring exactly
    # what it scored before.
    references: dict[str, str] = {args.step: args.ppi}
    for spec in args.ppi_step:
        step, path = parse_ppi_step(spec)
        # A DUPLICATE IS A REFUSAL, not a last-one-wins.  The positional PPI is
        # the reference for --step, so `--ppi-step 0001=other.txt` beside a
        # positional `kngr_mas_ppi_boc.txt` used to REPLACE the positional
        # reference in silence -- the run scored a statepoint against a map the
        # operator did not think they had passed, and the verdict looked
        # ordinary.  Two references for one statepoint is a typo every time.
        if step in references and references[step] != path:
            raise SystemExit(
                f"two MASTER references for statepoint {step}: {references[step]!r} "
                f"(positional, for --step {args.step}) and {path!r} (--ppi-step). One "
                f"statepoint has one reference; drop one of them.")
        references[step] = path

    with h5py.File(args.h5, "r") as f:
        kbc, kec = int(f["geometry"]["kbc"][()]), int(f["geometry"]["kec"][()])
        ij = f["geometry"]["ijtola"][()]
        nxa_g = int(f["geometry"]["nxa"][()])
        nya_g = int(f["geometry"]["nya"][()])
        ij = ij.reshape(nya_g, nxa_g)
        ras_map = {}
        for jj in range(ij.shape[0]):
            for ii in range(ij.shape[1]):
                la = int(ij[jj, ii])
                if la >= 0:
                    ras_map[(jj, ii)] = la

        present = sorted(f["steps"].keys())
        missing = [s for s in sorted(references) if s not in present]
        if missing:
            # A KeyError traceback here used to be the whole message.
            raise SystemExit(
                f"{args.h5}: no statepoint group(s) {', '.join(missing)}; the file has "
                f"{', '.join(present) if present else '(none)'}")
        steps = present if args.all_steps else sorted(references)

        parsed: dict[str, dict] = {}
        for i, (step, path) in enumerate(sorted(references.items())):
            parsed[step] = parse_ppi(path, npin=args.lattice, origin=args.origin,
                                     col_labels=args.col_labels, verbose=(i == 0))
        drift_reference = parsed[args.step]

        rows = []
        warned = False
        for step in steps:
            pp = f[f"steps/{step}/pin_power"][()]
            ras = pp[kbc:kec].mean(axis=0)
            cuts = half_pin_cuts(ras_map, ras, args.lattice)
            if cuts and args.half_pin_correct:
                ras = apply_half_pin_correction(ras, ras_map, cuts, args.lattice)
                if not warned:
                    print(f"half-pin correction applied to {len(cuts)} cut-line "
                          f"seat(s): {', '.join(str(s) for s in sorted(cuts))}")
                    warned = True
            elif cuts and not warned:
                # NOT a debug line.  This is the difference between a 4.7 % pin
                # reading and a 9.5 % one on the same run, and the larger number
                # looks exactly like a physics failure.
                print(f"WARNING: {len(cuts)} seat(s) hold a HALF assembly "
                      f"({', '.join(str(s) for s in sorted(cuts))}), so their "
                      f"cut-line pins carry half the power MASTER prints for the "
                      f"same pins. Every number below is inflated by that "
                      f"artefact. Pass --half-pin-correct.")
                warned = True
            is_scored = step in references
            mas = parsed[step] if is_scored else drift_reference
            rms, mx = pin_err_stats(ras_map, mas, ras, nya=nya_g, nxa=nxa_g)
            rows.append((step, rms, mx, is_scored))

    for step, rms, mx, is_scored in rows:
        kind = "scored" if is_scored else "drift vs %s" % args.step
        print("pin[%s]: rms %.3f%% max %.2f%% (%s)" % (step, rms, mx, kind))
    # The pre-WP24 line, kept verbatim for the default invocation because
    # docs/V5_FREEZE_20260830_KO.md and docs/WP20_GPU_FP32_20260831_KO.md quote
    # it and anything grepping for it would otherwise break in silence.
    boc = next((r for r in rows if r[0] == args.step), None)
    if boc is not None and args.step == "0001":
        print("BOC pin: rms %.3f%% max %.2f%%" % (boc[1], boc[2]))

    scored_rows = [r for r in rows if r[3]]
    drift_rows = [r for r in rows if not r[3]]
    if drift_rows:
        worst_drift = max(drift_rows, key=lambda r: r[2])
        # `drift_max`, NOT `max`.  This number is tens of percent of real
        # BOC-to-EOC redistribution by design, and it prints on the same stdout
        # as the scored verdict -- a runbook grepping for `max` would pick it up
        # and read a healthy run as a catastrophic pin error.  The distinct
        # token is the whole point of the spelling.
        print("drift (report only, BOC-referenced, NOT a pin error): worst "
              "drift_max %.2f%% at %s over %d unreferenced statepoint(s). Not "
              "scored -- give it a reference with --ppi-step %s=<MASTER PPI> to "
              "make it a verdict."
              % (worst_drift[2], worst_drift[0], len(drift_rows), worst_drift[0]))

    worst_rms = max(scored_rows, key=lambda r: r[1])
    worst_max = max(scored_rows, key=lambda r: r[2])
    measured = {"pin_rms_pct": worst_rms[1], "pin_max_pct": worst_max[2]}
    print("scored statepoints: %s (worst rms at %s, worst max at %s)"
          % (", ".join(r[0] for r in scored_rows), worst_rms[0], worst_max[0]))
    passed, code = gate_b_envelope.report(envelope, measured, "GATE B pin")
    return 0 if passed else code


if __name__ == "__main__":
    raise SystemExit(main())
