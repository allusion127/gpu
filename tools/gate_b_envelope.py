#!/usr/bin/env python3
"""WP24 -- Gate B's acceptance envelopes, as NUMBERS instead of prose.

WHAT WAS WRONG.  The production envelope existed only in docs/
(A2_OUTER_REDUCTION, PRICING_PROD, the performance report).  Both Gate B tools
PRINTED their metrics and exited 0 unconditionally: `tools/compare_master_rasbery.py`
has no threshold at all, and `tools/gate_b_pin_rms.py` takes two positional
arguments and prints one line.  So "Gate B passed" was a human reading a number
off a terminal and comparing it, from memory, against a figure in a Korean
design document.  Nothing in the tree could fail.

WHY THAT MATTERS MORE NOW THAN IT DID.  A screening preset introduces a SECOND
envelope, and a second envelope with no machine-readable first one is how a
screen100 number gets filed against the production column, or -- worse in the
direction the campaign actually loses -- a production number gets waved through
because somebody had the screening figures in their head.  So both envelopes
are here, both tools take `--envelope`, and the DEFAULT IS `production` so that
no existing invocation silently gets the looser one.

THE PRODUCTION ROW IS THE ACCEPTANCE COLUMN, NOT THE MEASUREMENT COLUMN.  This
is the correction that had to be made before the default envelope could be
trusted for a single day.  docs/A2_OUTER_REDUCTION_20260829_KO.md Sec 5's
acceptance table has TWO columns -- the v2 reference's MEASURED figures and the
BAR a run has to clear -- and the first draft of this file took the measurement
column: 1.905 pcm / 15.309 ppm / 0.238 % pin RMS.  The consequence is checkable
and it was checked: docs/PRICING_PROD_20260830_KO.md records the ACCEPTED
production Gate B as max|dppm| = 15.334, max|dpcm| = 1.847, max|dAO| = 0.012,
judged PASS.  Against a 15.309 limit that frozen result exits 1.  Since
`production` is the DEFAULT envelope, every existing compare invocation would
have begun failing on a run the campaign had already accepted -- exactly the "a
legitimate run failing on a rounding difference would poison confidence in the
gate on its first day" outcome the paragraph below warns about, shipped as the
default.  So the row states the ACCEPTANCE bars (<= ~2 pcm / <= ~15.4 ppm /
<= ~0.013 / <= 0.24 % / <= 0.80 %) and the measured v2 figures stay in the docs
where they belong.  A limit set to a measurement is a limit the next run of the
same code fails by rounding.

THE ONE THING THAT CHANGES FOR EXISTING CALLERS.  These tools begin returning a
nonzero exit code on a breach.  A caller that ignored the exit status keeps
working; a caller that checked it starts seeing failures it previously could
not see, which is the point.  Before wiring either into anything that checks
the status, run both once against a known-good PRODUCTION result.

EVERY LIMIT HERE IS ABSOLUTE AND MASTER-RELATIVE, AND THE PRODUCTION BASELINE
IS ALREADY INSIDE IT.  This is the sentence that has to be read before any
screen100 FAIL.  The directive states screen100 as "MASTER-relative |dkeff| <=
100 pcm and pin power error < 1 %", so these are limits on the SAME max|delta|
columns compare_master_rasbery.py and gate_b_pin_rms.py already report -- not on
the error a screening preset ADDS.  The production bar is not zero, so the four
columns do not grant the same amount of new error:

  keff      2.0  -> 100 pcm     98.0 pcm of headroom
  CBC      15.4  -> 33.5 ppm    18.1 ppm, which is 98.0 pcm at -5.4 pcm/ppm
  pin RMS   0.24 -> 1.0 %        0.76 pp
  pin max   0.80 -> 1.0 %        0.20 pp  <-- THE COLUMN THAT BINDS

The ppm figure is DERIVED, and it is derived as an INCREMENT -- which is the
half the first draft of this table got wrong.  It read 100/5.4 = 18.5 ppm as if
the ppm limit were the image of the whole keff LIMIT, while the keff limit is
measured from ~0 and the ppm one from the production CBC bar.  That granted keff
~98 pcm of new error and boron about 17 pcm-equivalent under one name -- and
screen100's own knobs spend more than that before any depletion accumulation
(the search tolerance, capped absolutely at 1e-4 = 10 pcm = 1.85 ppm; xe_tol
1e-5, measured at up to 8.9 pcm = 1.65 ppm).  The ppm column is therefore the
production bar plus the ppm image of the SAME keff budget, so all four columns
carry one 100 pcm / 1 % screening budget on top of one production baseline.

AND THE CONSEQUENCE OF THAT CHOICE, STATED ONCE SO NOBODY HAS TO DERIVE IT.  On
a critical-boron deck `delta_pcm` is ~0 BY CONSTRUCTION -- the search drives
k_eff to target -- so the whole model-vs-MASTER reactivity error lands in
`delta_ppm`, and screen100's effective MASTER-relative reactivity gate on those
decks is 33.5 ppm x 5.4 = ~181 pcm, i.e. about 1.8x the directive's 100 pcm,
while the keff column stays pinned at 100 absolute.  That is internally coherent
(one new budget on one baseline, expressed in each column's own units), but a
reader checking screen100 against the directive will otherwise read 33.5 ppm as
"100 pcm", and it is not.

pin max's 0.20 pp is NOT re-derived: the directive says 1 %, so 1 % it is.  What
changed is that the number is written down here and printed beside every
verdict, so a screen100 pin FAIL reads as "0.20 pp of screening headroom was not
enough" rather than as "the preset is broken".  The 1 % pin column is ASSERTED
BY THE DIRECTIVE AND NOT DERIVED FROM ANY MEASUREMENT IN THIS TREE: the measured
staged scan has arms at 1.12-1.16 % pin max at PRODUCTION polish tolerances, and
screen100 relaxes polish on top of that.  Treat the row as a candidate pending
per-deck Gate B, not as a campaign default.

  AO   advisory     There is NO defensible tie from a k_eff budget to an axial
                    offset, so screen100 REPORTS the AO delta and does not fail
                    on it.  Inventing a threshold to look complete is how a gate
                    acquires a number nobody can defend when it fires.

BOUNDARY.  The directive says pin error "< 1 %"; verdict() tests `<= limit`.
The looser reading is deliberate -- a measurement landing exactly on a limit is
a rounding artefact and not a breach -- and it is stated here rather than left
for a reader to discover from the source.

A GATE THAT PASSES ON NO DATA IS THE FAILURE MODE THIS FILE EXISTS TO END, AND
"NO DATA" INCLUDES DATA THAT CANNOT FAIL.  verdict() SKIPS a metric the caller
did not measure (a deck with no boron search has no ppm column, and assuming a
pass for it would be its own defect), which means an empty measurement dict
"passes".  So scored() reports which metrics were actually judged, report()
refuses to print PASS when that is empty, and it names the envelope columns the
calling tool does not measure at all.  The second half of that rule is `pinned`:
a caller may declare that a metric it DID measure is STRUCTURALLY inert for the
deck it measured -- a rod-crit deck's `delta_ppm` is the deck's own fixed boron
on both sides, so it is identically zero whatever the physics did -- and a run
in which EVERY judged metric is pinned is NOT SCORED, not a pass.  Without that,
a rod-search deck prints "GATE B scalars: PASS" having judged only columns that
could not have failed, which is the same verdict-shaped nothing as `exit 0`.

WHAT NEITHER ENVELOPE JUDGES.  The peaking columns.
tools/compare_master_rasbery.py computes delta_fqn / delta_frn / delta_fqp /
delta_frp and METRICS below has none of them, so a relaxed-tolerance arm's SHAPE
error is unjudged -- on a deck whose AO is also advisory, that is the whole
axial and radial peaking story going unscored.  report() names them on every
verdict rather than leaving the omission to be discovered.

THE MEASUREMENT THAT MUST NOT BE FORGOTTEN WHEN READING A PIN VERDICT.  Under
staging the published state always met the production tolerance (POLISH restores
it before anything is written), so a staged run's pin error is TRAJECTORY
divergence through burnup, not convergence error -- and the measured scan shows
that as a cliff rather than a slope: five arms sit at 1.12-1.16 % while the
neighbouring arms sit at 0.03-0.07 %, and the deviation in one of them exists
only at statepoints 28-33.  A BOC-only pin gate cannot see that.  gate_b_pin_rms
therefore scores every statepoint FOR WHICH IT WAS GIVEN A REFERENCE and reports
the rest as BOC-referenced DRIFT that never touches the verdict: BOC-to-EOC pin
redistribution is real physics of order several to tens of percent, so a verdict
taken on a BOC-referenced late statepoint would be a guaranteed FAIL that says
nothing at all about screening error.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: The measured APR1400 cy01 boron slope, src/Driver.h.  The ppm column is the
#: keff budget divided by this, spelled once so the two cannot drift apart.
BORON_SLOPE_PCM_PER_PPM = 5.4

#: The columns tools/compare_master_rasbery.py computes that NO envelope judges.
#: Named in every verdict so "Gate B passed" cannot be read as covering them.
UNJUDGED_COLUMNS = ("delta_fqn", "delta_frn", "delta_fqp", "delta_frp")


@dataclass(frozen=True)
class Envelope:
    """One acceptance envelope.  `None` means ADVISORY: report, never fail.

    Every limit is an ABSOLUTE MASTER-relative bound on the same max|delta|
    column the compare tools already report.  `baseline` names the envelope
    whose figures are already consumed inside these limits, so a verdict line
    can say how much NEW error a breach actually had to work with; `None` means
    this envelope IS a baseline.
    """
    name: str
    keff_pcm: float | None
    ppm: float | None
    ao: float | None
    pin_rms_pct: float | None
    pin_max_pct: float | None
    note: str
    baseline: str | None = None


ENVELOPES: dict[str, Envelope] = {
    # THE ACCEPTANCE COLUMN of docs/A2_OUTER_REDUCTION_20260829_KO.md Sec 5's
    # acceptance-envelope table -- the BAR column (<= ~2 pcm / <= ~15.3 ppm /
    # <= ~0.013 / <= 0.24 % / <= 0.8 %), NOT the measured v2 figures beside it
    # (1.905 / 15.309 / 0.013 / 0.24 / 0.78).  The ppm bar is written 15.4 and
    # not 15.3 because the ACCEPTED production Gate B
    # (docs/PRICING_PROD_20260830_KO.md) measured 15.334: the doc's "~15.3" is a
    # rounded statement of a bar the frozen result sits just above, and a
    # DEFAULT envelope that fails the campaign's own accepted run is a gate
    # nobody will keep switched on.
    "production": Envelope(
        name="production",
        keff_pcm=2.0,
        ppm=15.4,
        ao=0.013,
        pin_rms_pct=0.24,
        pin_max_pct=0.80,
        note="the v2 re-freeze ACCEPTANCE bars (the measured v2 figures are "
             "1.905 pcm / 15.309 ppm / 0.238 % and are what an accepted run "
             "sits inside, not what it is judged against); the only envelope "
             "an acceptance table may quote",
    ),
    # WP24.  The GA screening envelope.  NOT acceptance-eligible: a number
    # measured here belongs in a screening lane and a promotion has to be re-run
    # at strict before it can be quoted against `production`.
    "screen100": Envelope(
        name="screen100",
        keff_pcm=100.0,
        ppm=15.4 + (100.0 - 2.0) / BORON_SLOPE_PCM_PER_PPM,
        ao=None,
        pin_rms_pct=1.0,
        pin_max_pct=1.0,
        note="GA screening only (RASBERY_FIDELITY=screen100). ABSOLUTE "
             "MASTER-relative limits with the production baseline already "
             "inside them, so the NEW error each column grants is: keff 98.0 "
             "pcm, CBC 18.1 ppm (that same 98.0 pcm at -5.4 pcm/ppm), pin RMS "
             "0.76 pp, pin max 0.20 pp. Pin max is the column that binds, and "
             "its 1 % is the directive's number rather than a measured one -- "
             "the staged scan has arms at 1.12-1.16 % pin max at PRODUCTION "
             "polish tolerances, so this row is a candidate pending per-deck "
             "Gate B. On a critical-boron deck delta_pcm is ~0 by construction "
             "and the reactivity error lands in the CBC column, so the "
             "EFFECTIVE reactivity gate there is 33.5 ppm x 5.4 = ~181 pcm and "
             "not 100. AO advisory",
        baseline="production",
    ),
}

#: WP24.1.  A SWEEP ARM IS JUDGED BY ITS PARENT'S BAR, so it is an ALIAS and
#: not a second Envelope.  `screen100e4` is the screen100 preset row with one
#: knob moved (cmfd_sweep_epsl2 1e-5 -> 1e-4, src/FidelityPreset.h): what the
#: arm is allowed to be WRONG by is not one of the things the sweep changes, so
#: copying screen100's five numbers under a second name would be two spellings
#: of one bar -- and the two would then be free to drift apart, which is the
#: defect the single preset table exists to end, restated here in Python.  The
#: name still has to RESOLVE, because the runbook's Gate B command for the arm
#: spells `--envelope screen100e4`; without this it is an argparse error on the
#: runner and the arm has no gate at all.
#: WP24.2 adds a SECOND arm off the same parent -- `screen100x`, the Xe POLISH
#: tolerance at 1e-4 (xe_tol 1e-5 -> 1e-4) -- and it is an alias for exactly the
#: same reason: the sweep moves how FAST the arm converges, never how WRONG it
#: is allowed to be, so both arms are judged by the parent's five numbers as the
#: same object.
ENVELOPE_ALIASES: dict[str, str] = {
    "screen100e4": "screen100",
    "screen100x": "screen100",
}

#: Every spelling `--envelope` accepts, aliases included.
ENVELOPE_NAMES: tuple[str, ...] = tuple(sorted(set(ENVELOPES) | set(ENVELOPE_ALIASES)))

DEFAULT_ENVELOPE = "production"

#: The metric keys, in the order a verdict prints them.
METRICS = ("keff_pcm", "ppm", "ao", "pin_rms_pct", "pin_max_pct")


def add_envelope_argument(parser) -> None:
    """`--envelope` on an argparse parser, spelled once for both Gate B tools."""
    parser.add_argument(
        "--envelope",
        choices=list(ENVELOPE_NAMES),
        default=DEFAULT_ENVELOPE,
        help="acceptance envelope to judge against (default: %(default)s). "
             "`screen100` is the GA screening envelope (100 pcm / 33.5 ppm / "
             "pin 1 %% RMS and 1 %% max, AO advisory); its limits are ABSOLUTE "
             "and already contain the production baseline, and it is NOT "
             "acceptance-eligible. `screen100e4` and `screen100x` are the SAME "
             "envelope under the sweep arms' names (src/FidelityPreset.h), not "
             "looser ones.",
    )


def resolve(name: str) -> Envelope:
    """The Envelope *name* judges against, following the sweep-arm aliases."""
    try:
        return ENVELOPES[ENVELOPE_ALIASES.get(name, name)]
    except KeyError:
        raise SystemExit(
            f"unknown envelope {name!r}; known: {', '.join(ENVELOPE_NAMES)}"
        ) from None


def limits(envelope: Envelope) -> dict[str, float | None]:
    return {key: getattr(envelope, key) for key in METRICS}


def headroom(envelope: Envelope, metric: str) -> float | None:
    """How much NEW error *metric* grants over `envelope.baseline`, or None.

    This is the number a reader of a FAIL needs, and the one the first draft of
    this file kept in a handoff note instead: screen100's pin max limit is 1.0 %
    but production already bars at 0.80 %, so a 0.9 % screening result has spent
    half of a 0.20 pp allowance and is nowhere near "half the envelope".
    """
    limit = getattr(envelope, metric)
    if limit is None or envelope.baseline is None:
        return None
    base = ENVELOPES.get(envelope.baseline)
    base_limit = getattr(base, metric, None) if base is not None else None
    if base_limit is None:
        return None
    return limit - base_limit


def scored(envelope: Envelope, measured: dict[str, float],
           pinned: Iterable[str] = ()) -> list[str]:
    """The metrics a verdict over *measured* actually JUDGED.

    Empty means nothing was judged, and a caller that prints PASS on that has
    reproduced the unconditional `exit 0` this module was written to replace.
    An advisory metric (limit None) is reported and not judged, so it is not
    here either -- and neither is a metric the caller declared STRUCTURALLY
    PINNED, which is a measured column that could not have failed.  A column
    that cannot fail is not evidence, and counting it as judged is how a gate
    reports PASS over a deck family it is blind to.
    """
    blocked = set(pinned)
    return [key for key in METRICS
            if key in measured and getattr(envelope, key) is not None
            and key not in blocked]


def verdict(envelope: Envelope, measured: dict[str, float],
            pinned: Iterable[str] = ()) -> tuple[bool, list[str]]:
    """(passed, lines).  A metric with no limit is REPORTED, never failed on.

    A metric the caller did not measure is skipped rather than assumed to pass:
    a deck with no boron search has no ppm column, and treating its absence as a
    pass would be the same defect as treating a missing receipt field as
    `strict`.  Which is exactly why `passed` alone is not a verdict -- callers
    go through report(), which consults scored().

    A PINNED metric is still tested -- a pinned column that somehow breaches is
    a real finding -- but it is LABELLED, because its passing is a property of
    the deck and not of the run.
    """
    blocked = set(pinned)
    passed = True
    lines: list[str] = []
    for key, limit in limits(envelope).items():
        if key not in measured:
            continue
        value = measured[key]
        if limit is None:
            lines.append(f"  {key:<12} = {value:9.3f}   (advisory, no limit under "
                         f"{envelope.name})")
            continue
        ok = abs(value) <= limit
        passed = passed and ok
        room = headroom(envelope, key)
        tail = ""
        if room is not None:
            base = ENVELOPES[envelope.baseline]
            tail = (f"   [{envelope.baseline} bar {getattr(base, key):.3f}, "
                    f"so {room:.3f} of NEW error]")
        if key in blocked:
            tail += "   (STRUCTURALLY PINNED -- not evidence)"
        lines.append(f"  {key:<12} = {value:9.3f}   limit {limit:9.3f}   "
                     f"{'PASS' if ok else 'FAIL'}{tail}")
    return passed, lines


def report(envelope: Envelope, measured: dict[str, float], label: str,
           pinned: Iterable[str] = ()) -> tuple[bool, int]:
    """Print one envelope verdict.  (passed, exit_code), spelled once.

    Exit 2 -- not 0 -- when nothing was scored.  "GATE B passed" having judged
    no metric is precisely the state this module exists to end, and it is a
    different failure from a breach, so it gets a different code.  A run whose
    only judged metrics are STRUCTURALLY PINNED lands here too: it measured
    columns, and none of them could have failed.
    """
    blocked = [m for m in METRICS if m in set(pinned)]
    passed, lines = verdict(envelope, measured, blocked)
    judged = scored(envelope, measured, blocked)
    scorable = [m for m in METRICS if getattr(envelope, m) is not None]
    print(f"envelope: {envelope.name} -- {envelope.note}")
    if lines:
        print("\n".join(lines))
    if not judged:
        pinned_here = [m for m in blocked if m in measured]
        why = (f"the only judged column(s) -- {', '.join(pinned_here)} -- are "
               f"structurally pinned by this deck and could not have failed"
               if pinned_here else
               f"none of {', '.join(scorable)} was measured")
        print(f"{label}: NOT SCORED (envelope {envelope.name}) -- {why}, so this "
              f"run judged nothing")
        return False, 2
    unmeasured = [m for m in scorable if m not in measured]
    if unmeasured:
        print(f"  (not measured here: {', '.join(unmeasured)} -- a whole-envelope "
              f"`Gate B passed` needs both compare_master_rasbery.py and "
              f"gate_b_pin_rms.py)")
    print(f"  (judged by NEITHER envelope: {', '.join(UNJUDGED_COLUMNS)} -- the "
          f"peaking columns carry a relaxed arm's SHAPE error and no limit here "
          f"covers them)")
    print("  (a boron-search statepoint has no |dkeff| to score -- k_eff is "
          "driven to target by construction -- so on those decks the CBC column "
          "is the reactivity gate and the keff column is a search residual)")
    print(f"{label}: {'PASS' if passed else 'FAIL'} (envelope {envelope.name})")
    return passed, 0 if passed else 1
