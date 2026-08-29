#!/usr/bin/env python3
"""Gate A: one candidate's .h5 against the trajectory it is allowed to move.

WHAT GATE A IS.  Gate B asks whether the code agrees with MASTER; Gate A asks
whether a change moved the code's OWN answer, and by how much.  The two are not
the same question and the second is the one a candidate can be screened on
locally, without a reference the workstation does not have.

WHY A TOOL AND NOT h5diff.  h5diff answers "identical or not", which is the
right question for a feature-off arm and the wrong one for a candidate that is
expected to move the trajectory: what is then wanted is the SIZE of the move in
the four quantities the campaign states thresholds for -- k_eff in pcm, critical
boron in ppm, axial offset, and pin power as a relative difference -- and WHERE
in the depletion it happens, because a drift that grows monotonically with
burnup reads differently from one confined to a single statepoint.

THE SCREEN, NOT THE GATE.  The defaults below are the A2 screening thresholds
(5 pcm / 5 ppm / 0.01 AO / 1 % pin): a candidate outside them is stopped and
reported rather than carried to the 238 runner.  Passing them is not adoption --
adoption is Gate B against MASTER, which runs elsewhere.  Say so in the exit
status: 0 means inside the screen, 1 means outside it, and neither means the
candidate is right.

USAGE
    tools/gate_a_compare.py BASE.h5 CAND.h5
    tools/gate_a_compare.py BASE.h5 CAND.h5 --per-step
    tools/gate_a_compare.py BASE.h5 CAND.h5 --keff-pcm 1 --pin-pct 0.24
    tools/gate_a_compare.py BASE.h5 CAND.h5 --json
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np


def _summary(f, name):
    key = "summary/" + name
    return np.asarray(f[key][...], dtype=float) if key in f else None


def compare(base_path, cand_path, per_step=False):
    fa = h5py.File(base_path, "r")
    fb = h5py.File(cand_path, "r")

    out = {"base": str(base_path), "cand": str(cand_path)}
    rows = []

    keff_a, keff_b = _summary(fa, "keff"), _summary(fb, "keff")
    if keff_a is None or keff_b is None:
        raise SystemExit("error: summary/keff missing -- is this a RASBERY result file?")
    n = min(len(keff_a), len(keff_b))
    out["statepoints"] = int(n)
    out["statepoints_base"] = int(len(keff_a))
    out["statepoints_cand"] = int(len(keff_b))

    dk = (keff_b[:n] - keff_a[:n]) * 1.0e5
    out["keff_pcm_max"] = float(np.abs(dk).max()) if n else 0.0
    out["keff_pcm_rms"] = float(np.sqrt((dk**2).mean())) if n else 0.0
    out["keff_pcm_argmax"] = int(np.abs(dk).argmax()) + 1 if n else 0

    ppm_a, ppm_b = _summary(fa, "ppm"), _summary(fb, "ppm")
    dp = (ppm_b[:n] - ppm_a[:n]) if ppm_a is not None and ppm_b is not None else np.zeros(n)
    out["ppm_max"] = float(np.abs(dp).max()) if n else 0.0
    out["ppm_rms"] = float(np.sqrt((dp**2).mean())) if n else 0.0
    out["ppm_argmax"] = int(np.abs(dp).argmax()) + 1 if n else 0

    ao_a, ao_b = _summary(fa, "ao"), _summary(fb, "ao")
    da = (ao_b[:n] - ao_a[:n]) if ao_a is not None and ao_b is not None else np.zeros(n)
    out["ao_max"] = float(np.abs(da).max()) if n else 0.0
    out["ao_argmax"] = int(np.abs(da).argmax()) + 1 if n else 0

    # Pin power, worst RELATIVE difference over every statepoint that carries a
    # reconstruction.  Relative because the campaign's pin thresholds are stated
    # that way and because an absolute difference on a low-power peripheral pin
    # is not the quantity anyone is protecting.  Cells at or below 1e-8 are
    # excluded: they are outside the core, and a ratio there is a division by
    # the discretisation, not a physics difference.
    pin_worst, pin_worst_step, pin_rms_worst = 0.0, None, 0.0
    steps_a = set(fa["steps"].keys()) if "steps" in fa else set()
    steps_b = set(fb["steps"].keys()) if "steps" in fb else set()
    per_step_pin = {}
    for s in sorted(steps_a & steps_b):
        if "pin_power" not in fa["steps"][s] or "pin_power" not in fb["steps"][s]:
            continue
        x = np.asarray(fa["steps"][s]["pin_power"][...], dtype=float)
        y = np.asarray(fb["steps"][s]["pin_power"][...], dtype=float)
        if x.shape != y.shape:
            print(f"  !! pin_power shape differs at step {s}: {x.shape} vs {y.shape}")
            continue
        m = np.abs(x) > 1.0e-8
        if not m.any():
            continue
        r = np.abs(y[m] - x[m]) / np.abs(x[m])
        per_step_pin[s] = (float(r.max()), float(np.sqrt((r**2).mean())))
        if r.max() > pin_worst:
            pin_worst, pin_worst_step = float(r.max()), s
            pin_rms_worst = float(np.sqrt((r**2).mean()))
    out["pin_rel_max"] = pin_worst
    out["pin_rel_rms_at_max"] = pin_rms_worst
    out["pin_max_step"] = pin_worst_step

    if per_step:
        for i in range(n):
            rows.append(
                {
                    "sp": i + 1,
                    "keff_pcm": float(dk[i]),
                    "ppm": float(dp[i]),
                    "ao": float(da[i]),
                }
            )
        out["per_step"] = rows
        out["per_step_pin"] = per_step_pin
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--per-step", action="store_true")
    ap.add_argument("--json", action="store_true")
    # A2 screening thresholds.  Not the v2 Gate B envelope, which is tighter and
    # is measured against MASTER somewhere else.
    ap.add_argument("--keff-pcm", type=float, default=5.0)
    ap.add_argument("--ppm", type=float, default=5.0)
    ap.add_argument("--ao", type=float, default=0.01)
    ap.add_argument("--pin-pct", type=float, default=1.0)
    args = ap.parse_args()

    r = compare(args.base, args.cand, per_step=args.per_step)

    checks = [
        ("keff  max |d| (pcm)", r["keff_pcm_max"], args.keff_pcm, f"sp {r['keff_pcm_argmax']}"),
        ("ppm   max |d|", r["ppm_max"], args.ppm, f"sp {r['ppm_argmax']}"),
        ("AO    max |d|", r["ao_max"], args.ao, f"sp {r['ao_argmax']}"),
        ("pin   max rel (%)", 100.0 * r["pin_rel_max"], args.pin_pct, f"step {r['pin_max_step']}"),
    ]
    verdict = all(v <= lim for _, v, lim, _ in checks)
    r["pass"] = bool(verdict)

    if args.json:
        import json

        print(json.dumps(r, indent=2))
        return 0 if verdict else 1

    print(f"=== GATE A: {args.cand}")
    print(f"       vs   {args.base}")
    if r["statepoints_base"] != r["statepoints_cand"]:
        print(
            f"  !! statepoint COUNT differs: base {r['statepoints_base']} "
            f"cand {r['statepoints_cand']} -- comparing the first {r['statepoints']}"
        )
    print(f"  statepoints           : {r['statepoints']}")
    for label, value, limit, where in checks:
        mark = "ok " if value <= limit else "OVER"
        print(f"  {mark} {label:20s}: {value:12.4f}   (limit {limit:g}, at {where})")
    print(f"      keff rms (pcm)      : {r['keff_pcm_rms']:12.4f}")
    print(f"      ppm  rms            : {r['ppm_rms']:12.4f}")
    print(f"      pin  rms at worst sp: {100.0 * r['pin_rel_rms_at_max']:12.4f} %")
    if args.per_step:
        print("\n  sp   dkeff(pcm)      dppm        dAO     pin max %")
        pin = r.get("per_step_pin", {})
        for row in r["per_step"]:
            key = f"{row['sp']:04d}"
            p = pin.get(key)
            ptxt = f"{100.0 * p[0]:9.4f}" if p else "        -"
            print(
                f"{row['sp']:4d} {row['keff_pcm']:+12.4f} {row['ppm']:+10.4f} "
                f"{row['ao']:+10.6f} {ptxt}"
            )
    print(f"\n  VERDICT: {'INSIDE the A2 screen' if verdict else 'OUTSIDE the A2 screen'}")
    print("  (Gate A is a screen, not an adoption test -- Gate B vs MASTER decides.)")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
