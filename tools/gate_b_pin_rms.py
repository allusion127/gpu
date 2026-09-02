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
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_b_envelope  # noqa: E402

XL = "ABCDEFGHJKLMNPRST"


def parse_ppi(path):
    ppi = open(path, errors="replace").read()
    heads = list(re.finditer(r"^\s*FANAME\s+([A-Z]+)\s+(\d+)\s+(\d+)\s", ppi, re.M))
    mas_pin = {}
    for hi, h in enumerate(heads):
        body = ppi[h.end(): heads[hi + 1].start() if hi + 1 < len(heads) else len(ppi)]
        col = XL.index(h.group(1)) - XL.index("J")
        row = int(h.group(2)) - 9
        m = re.search(r"PIN 3-D POWER DISTRIBUTION[^\n]*\n(.*)\Z", body, re.S)
        if not m:
            continue
        seg = m.group(1)
        stop = re.search(r"\n[A-Z][A-Z \-0-9]+ DISTRIBUTION|\n\*{10,}", seg)
        if stop:
            seg = seg[: stop.start()]
        nums = np.array([float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", seg)])
        npl = nums.size // 256
        if npl == 0:
            continue
        mas_pin[(row, col)] = nums[: npl * 256].reshape(npl, 16, 16).mean(axis=0)
    return mas_pin


def pin_err_stats(ras_map, mas_pin, ras_pin):
    mas_all, ras_all = [], []
    for (r, c), mp in mas_pin.items():
        la = ras_map.get((r, c))
        if la is None:
            continue
        rp = ras_pin[la]
        if not np.any(rp > 0):
            continue
        mas_all.append(mp)
        ras_all.append(rp)
    mnorm = np.mean([a[a > 0].mean() for a in mas_all])
    rnorm = np.mean([a[a > 0].mean() for a in ras_all])
    errs = []
    for (r, c), mp in mas_pin.items():
        la = ras_map.get((r, c))
        if la is None or r > 8 or c > 8:
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
    gate_b_envelope.add_envelope_argument(ap)
    args = ap.parse_args()

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
        for step, path in references.items():
            parsed[step] = parse_ppi(path)
        drift_reference = parsed[args.step]

        rows = []
        for step in steps:
            pp = f[f"steps/{step}/pin_power"][()]
            ras = pp[kbc:kec].mean(axis=0)
            is_scored = step in references
            mas = parsed[step] if is_scored else drift_reference
            rms, mx = pin_err_stats(ras_map, mas, ras)
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
