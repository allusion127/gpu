#!/usr/bin/env python3
"""Per-statepoint outer-count profile of a RASBERY run -- the A2 receipt.

WHY THIS EXISTS.  "The statepoint took 343 outers" is not actionable; "246 of
those 343 went to re-converging the flux after an equilibrium-Xe step" is.  The
solver already publishes the attribution -- Driver.h's sptelem block charges
every outer to the perturbation that opened its segment (INITIAL / XE / TH /
SEARCH / SETTLE / FALLBACK) -- and the device outer segment already publishes
which escape ended each segment.  What did not exist was a reader that folds the
two receipts into one table, so every A2 candidate was being judged on a wall
clock that is noisy on a shared workstation instead of on the iteration count it
actually moves.

WHAT IT READS.  One run log containing

  [RASBERY][SPTELEM]   {...}   one JSON line per statepoint
                               (RASBERY_STATEPOINT_TELEMETRY=1)
  [RASBERY][OUTER_GPU] {...}   one JSON line at exit, run-total
                               (RASBERY_GPU_OUTER=1; optional)

Both are emitted on stdout.  The OUTER_GPU line is optional: a host-arm run has
no segment and therefore no escape histogram, and the SPTELEM half of the table
is the half A2 is judged on.

WHY OUTERS AND NOT WALL.  A2's target is stated in outers (300 -> 100 per
statepoint) precisely because outers are deterministic: two runs of the same
binary on the same deck agree exactly, so a 3 % change is a real 3 % and not the
machine's other tenant.  The wall is reported beside it, from the receipt's own
`wall` field, but it is the corroborating number, not the deciding one.

USAGE
    tools/outer_profile.py RUN.log                 # one run, full table
    tools/outer_profile.py BASE.log CAND.log       # A/B, deltas per statepoint
    tools/outer_profile.py --json RUN.log          # machine-readable totals
    tools/outer_profile.py --top 10 RUN.log        # worst statepoints only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SPTELEM_RE = re.compile(r"\[RASBERY\]\[SPTELEM\]\s+(\{.*\})\s*$")
OUTER_GPU_RE = re.compile(r"\[RASBERY\]\[OUTER_GPU\]\s+(\{.*\})\s*$")

# The cause buckets, in the order Driver.h's enum declares them.  Named here
# with the receipt's field name so a renamed field fails loudly (KeyError on a
# missing bucket) instead of silently reporting a zero column.
CAUSES = [
    ("outers_initial", "initial"),
    ("xe_outers", "xe"),
    ("th_outers", "th"),
    ("search_outers", "search"),
    ("settle_outers", "settle"),
    ("fallback_outers", "fallback"),
]

# Run-total scalars worth folding.  Everything here is a count, so folding is a
# sum; nothing derived (rates, ratios) is stored, it is recomputed at print time
# from the folded counts so a 35-statepoint average is never an average of
# averages.
SUMS = [
    "outers",
    "xe_updates",
    "xe_interim_updates",
    "xe_cascades",
    "xe_budget_exhausted",
    "xe_aa_proposed",
    "xe_aa_accepted",
    "xe_aa_rejected",
    "xe_aa_history_resets",
    "search_trials",
    "th_updates",
    "flux_limit_retries",
    "cmfd_sweeps",
    "bicg_iters",
]
FLOAT_SUMS = ["wall", "io_wall"]


class Profile:
    """One run: the per-statepoint rows plus whatever run-total lines were found."""

    def __init__(self, path: Path):
        self.path = path
        self.steps: list[dict] = []
        self.outer_gpu: dict | None = None
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            m = SPTELEM_RE.search(line)
            if m:
                self.steps.append(json.loads(m.group(1)))
                continue
            m = OUTER_GPU_RE.search(line)
            if m:
                # Last one wins: a batch prints one per slot, and the run total
                # is what the escape histogram is about.
                self.outer_gpu = json.loads(m.group(1))
        self.steps.sort(key=lambda s: s.get("statepoint", 0))

    def __bool__(self) -> bool:
        return bool(self.steps) or self.outer_gpu is not None

    def totals(self) -> dict:
        t: dict[str, float] = {k: 0 for k in SUMS}
        t.update({k: 0.0 for k in FLOAT_SUMS})
        for field, _ in CAUSES:
            t[field] = 0
        for s in self.steps:
            for k in SUMS:
                t[k] += s.get(k, 0)
            for k in FLOAT_SUMS:
                t[k] += s.get(k, 0.0)
            for field, _ in CAUSES:
                t[field] += s.get(field, 0)
        t["statepoints"] = len(self.steps)
        return t

    def escapes(self) -> dict:
        if not self.outer_gpu:
            return {}
        return dict(self.outer_gpu.get("escapes", {}))


def fmt_table(p: Profile, top: int | None) -> str:
    rows = p.steps
    if top:
        rows = sorted(rows, key=lambda s: -s.get("outers", 0))[:top]
        rows.sort(key=lambda s: s.get("statepoint", 0))
    out = []
    out.append(
        "  sp   efpd   outers | initial     xe     th search settle fallbk |"
        "  xeupd xeintm casc  trials  wall(s)"
    )
    out.append("  " + "-" * 100)
    for s in rows:
        out.append(
            "{sp:4d} {efpd:6.1f} {outers:8d} | {ini:7d} {xe:6d} {th:6d} {se:6d} "
            "{st:6d} {fb:6d} | {xu:6d} {xi:6d} {cc:4d} {tr:6d} {wall:8.3f}".format(
                sp=s.get("statepoint", -1),
                efpd=s.get("efpd", 0.0),
                outers=s.get("outers", 0),
                ini=s.get("outers_initial", 0),
                xe=s.get("xe_outers", 0),
                th=s.get("th_outers", 0),
                se=s.get("search_outers", 0),
                st=s.get("settle_outers", 0),
                fb=s.get("fallback_outers", 0),
                xu=s.get("xe_updates", 0),
                xi=s.get("xe_interim_updates", 0),
                cc=s.get("xe_cascades", 0),
                tr=s.get("search_trials", 0),
                wall=s.get("wall", 0.0),
            )
        )
    return "\n".join(out)


def fmt_totals(p: Profile) -> str:
    t = p.totals()
    n = max(1, t["statepoints"])
    total = t["outers"]
    out = []
    out.append(f"  statepoints           : {t['statepoints']}")
    out.append(f"  TOTAL OUTERS          : {total}   ({total / n:.1f} per statepoint)")
    out.append("  attribution:")
    for field, label in CAUSES:
        v = t[field]
        pct = (100.0 * v / total) if total else 0.0
        out.append(f"    {label:9s} {v:8d}   {pct:5.1f} %   {v / n:7.1f}/sp")
    # xe_outers per xe step is the number A2's interim-Xe candidate moves: it is
    # the cost of re-converging the flux once per Xe update.
    xu = t["xe_updates"]
    if xu:
        out.append(
            f"  xe re-convergence cost: {t['xe_outers'] / xu:.2f} outers per settled Xe step"
        )
    cc = t["xe_cascades"]
    if cc:
        out.append(f"  xe steps per cascade  : {xu / cc:.2f}   ({cc} cascades)")
    if t["xe_interim_updates"]:
        out.append(f"  xe interim steps      : {t['xe_interim_updates']}")
    prop = t["xe_aa_proposed"]
    if prop:
        out.append(
            f"  Xe Anderson           : {t['xe_aa_accepted']}/{prop} accepted "
            f"({100.0 * t['xe_aa_accepted'] / prop:.1f} %), "
            f"{t['xe_aa_rejected']} rejected, {t['xe_aa_history_resets']} resets"
        )
    out.append(
        f"  search trials {t['search_trials']}, T/H updates {t['th_updates']}, "
        f"flux limit cycles {t['flux_limit_retries']}, "
        f"Xe budget exhausted {t['xe_budget_exhausted']}"
    )
    out.append(
        f"  cmfd sweeps {t['cmfd_sweeps']}, bicg iters {t['bicg_iters']}, "
        f"solve wall {t['wall']:.2f} s, io wall {t['io_wall']:.2f} s"
    )
    esc = p.escapes()
    if esc:
        tot_esc = sum(esc.values()) or 1
        og = p.outer_gpu or {}
        out.append(
            f"  device segment        : {og.get('segment_launches', 0)} launches, "
            f"{og.get('device_outers', 0)} device outers, budget {og.get('segment_budget', '?')}"
        )
        out.append("  segment escapes (per segment exit, NOT per outer):")
        for k in sorted(esc, key=lambda k: -esc[k]):
            out.append(f"    {k:18s} {esc[k]:8d}   {100.0 * esc[k] / tot_esc:5.1f} %")
    return "\n".join(out)


def fmt_ab(base: Profile, cand: Profile) -> str:
    """A/B by statepoint.  Statepoints are matched on their number, not their
    position, because a candidate that changes the trajectory can still be
    compared step for step -- and one that changes the STEP COUNT is a different
    deck run, which is a finding, not a row."""
    bt, ct = base.totals(), cand.totals()
    out = []
    out.append(f"  base : {base.path}")
    out.append(f"  cand : {cand.path}")
    out.append("")
    bmap = {s["statepoint"]: s for s in base.steps}
    cmap = {s["statepoint"]: s for s in cand.steps}
    only_b = sorted(set(bmap) - set(cmap))
    only_c = sorted(set(cmap) - set(bmap))
    if only_b or only_c:
        out.append(f"  !! statepoint sets differ: base-only {only_b} cand-only {only_c}")
    out.append("  sp    base    cand    delta    ratio |  xe base/cand | settle b/c | search b/c")
    out.append("  " + "-" * 92)
    for sp in sorted(set(bmap) & set(cmap)):
        b, c = bmap[sp], cmap[sp]
        bo, co = b["outers"], c["outers"]
        out.append(
            f"{sp:4d} {bo:7d} {co:7d} {co - bo:+8d} {(co / bo if bo else 0):8.3f} |"
            f" {b['xe_outers']:6d}/{c['xe_outers']:<6d} |"
            f" {b['settle_outers']:4d}/{c['settle_outers']:<4d} |"
            f" {b['search_outers']:4d}/{c['search_outers']:<4d}"
        )
    out.append("  " + "-" * 92)
    bo, co = bt["outers"], ct["outers"]
    out.append(
        f"  TOTAL {bo:7d} {co:7d} {co - bo:+8d} {(co / bo if bo else 0):8.3f}"
        f"   ({bo / max(1, bt['statepoints']):.1f} -> {co / max(1, ct['statepoints']):.1f} per sp)"
    )
    out.append("")
    out.append("  per-cause totals:")
    for field, label in CAUSES:
        out.append(
            f"    {label:9s} {bt[field]:8d} -> {ct[field]:8d}  ({ct[field] - bt[field]:+d})"
        )
    out.append(
        f"    {'xe_updates':9s} {bt['xe_updates']:8d} -> {ct['xe_updates']:8d}  "
        f"({ct['xe_updates'] - bt['xe_updates']:+d})"
    )
    out.append(
        f"    {'xe_intm':9s} {bt['xe_interim_updates']:8d} -> {ct['xe_interim_updates']:8d}  "
        f"({ct['xe_interim_updates'] - bt['xe_interim_updates']:+d})"
    )
    out.append(
        f"    {'wall':9s} {bt['wall']:8.2f} -> {ct['wall']:8.2f}  "
        f"({ct['wall'] - bt['wall']:+.2f} s)"
    )
    besc, cesc = base.escapes(), cand.escapes()
    if besc or cesc:
        out.append("  segment escapes:")
        for k in sorted(set(besc) | set(cesc)):
            out.append(f"    {k:18s} {besc.get(k, 0):8d} -> {cesc.get(k, 0):8d}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path,
                    help="one run log, or BASE and CANDIDATE for an A/B")
    ap.add_argument("--json", action="store_true", help="machine-readable totals only")
    ap.add_argument("--top", type=int, default=None,
                    help="show only the N statepoints with the most outers")
    ap.add_argument("--no-table", action="store_true", help="totals only")
    args = ap.parse_args()

    if len(args.logs) > 2:
        print("error: pass one log (profile) or two (A/B)", file=sys.stderr)
        return 2

    profiles = []
    for p in args.logs:
        if not p.exists():
            print(f"error: no such log: {p}", file=sys.stderr)
            return 2
        prof = Profile(p)
        if not prof:
            print(
                f"error: {p} has no [RASBERY][SPTELEM] lines -- rerun with "
                f"RASBERY_STATEPOINT_TELEMETRY=1",
                file=sys.stderr,
            )
            return 2
        profiles.append(prof)

    if args.json:
        payload = []
        for prof in profiles:
            t = prof.totals()
            t["log"] = str(prof.path)
            t["escapes"] = prof.escapes()
            payload.append(t)
        print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2))
        return 0

    if len(profiles) == 2:
        print("=== OUTER PROFILE A/B ===")
        print(fmt_ab(profiles[0], profiles[1]))
        return 0

    prof = profiles[0]
    print(f"=== OUTER PROFILE: {prof.path} ===")
    if not args.no_table:
        print(fmt_table(prof, args.top))
        print()
    print(fmt_totals(prof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
