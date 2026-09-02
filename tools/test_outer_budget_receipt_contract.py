#!/usr/bin/env python3
"""Contract: the A2 S0 outer-budget receipt says where the outers went, once each.

WHAT THIS GUARDS.  docs/A2_OUTER_REDUCTION_DESIGN_20260902_KO.md prices four
designs against a budget of 125-165 outers per statepoint, and its own Sec 1.1
and Sec 5 item 1 say the attribution behind that price is an INFERENCE from a
KNGR profile rather than a measurement -- and that no m64 attribution exists at
all.  Spike S0 is the receipt that replaces the inference.  A receipt that
attributed one outer twice, or that lost a phase, would replace it with a worse
inference that looks like a measurement, so the properties that make it readable
are pinned here rather than assumed.

FOUR HALVES.

SOURCE (always).  The emitter: one site, unconditional, before the trajectory
line, outside the telemetry gate; every counter it publishes folded into the run
total exactly once; the four `prev_inner = eigv + 1.0` sentinel sites each
carrying exactly one counter and no site carrying two; the counterfactual armed,
read and dropped without any solver variable seeing it; and BICGCMFD charging a
drive exit at every path a drive can end on.

IDENTITY (always).  check_receipt() is the arithmetic a consumer runs on the
line, and it is exercised here on a fixture that satisfies every identity.  The
identities are the receipt's whole claim: outers sum to the phases, the cascade
count equals what its three start sites predict, and the sweeps split into the
ones that were charged and the ones the negative-flux retry rule ran for free.

NEGATIVE CONTROLS (always).  Each identity is broken in turn and check_receipt()
must reject; each source scan is fed a source that violates it and must fail.  A
check that cannot fail is a check that is not being run.

LIVE (`--check RUN.log`).  The same check_receipt() against a real run's line,
for a host that has one.  This is what 238 runs after an S0 measurement pass.

Run: python tools/test_outer_budget_receipt_contract.py
     python tools/test_outer_budget_receipt_contract.py --check RUN.log
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8-sig")
BICG_H = (ROOT / "src" / "BICGCMFD.h").read_text(encoding="utf-8-sig")
BICG = (ROOT / "src" / "BICGCMFD.cpp").read_text(encoding="utf-8-sig")

RECEIPT = "[RASBERY][OUTER_BUDGET]"
TRAJECTORY = "[RASBERY][TRAJECTORY]"

FAILED: list[str] = []


def fail(message: str) -> None:
    FAILED.append(message)


def region(text: str, start: str, end: str, what: str) -> str:
    i = text.find(start)
    if i < 0:
        fail(f"{what}: anchor not found: {start!r}")
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        fail(f"{what}: closing anchor not found: {end!r}")
        return ""
    return text[i:j]


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# ===========================================================================
# THE IDENTITIES -- what a consumer runs on the line
# ===========================================================================
#
# These are the receipt's entire claim, and each one is a claim about
# ATTRIBUTION rather than about physics: a phase counted twice, or a sweep
# counted in neither bucket, would show up as a non-zero residual here and
# nowhere else.  The solver publishes the residuals itself
# (`cascade.residual`, `cmfd.sweep_split_residual`) precisely so that a reader
# does not have to re-derive them to know whether to trust the split.
def check_receipt(r: dict) -> list[str]:
    """Every identity the [RASBERY][OUTER_BUDGET] line asserts.  [] means PASS."""
    bad: list[str] = []

    def need(path: str):
        node = r
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                bad.append(f"missing field {path!r}")
                return None
            node = node[part]
        return node

    phases = need("outers_by_phase")
    attributed = need("outers_attributed")
    if phases is not None and attributed is not None:
        # EXACTLY ONCE.  SolveLoop charges `outers_by_cause[sp_cause]` once per
        # outer and ReconvergeFlux charges its whole delta to FALLBACK, so the
        # six buckets partition the outers.  They are not required to equal the
        # Driver's own `outers` -- the two counters have different reset points
        # and the receipt publishes both so the gap is visible -- but the
        # buckets must sum to the attributed total exactly.
        if sum(phases.values()) != attributed:
            bad.append(f"outers_by_phase sums to {sum(phases.values())}, "
                       f"outers_attributed says {attributed}: an outer was "
                       "charged to two phases or to none")
        for name in ("initial", "xe", "th", "search", "settle", "fallback"):
            if name not in phases:
                bad.append(f"outers_by_phase is missing the {name!r} bucket")
            elif phases[name] < 0:
                bad.append(f"outers_by_phase.{name} is negative")

    # Design Sec 1.3: the cascade identity.  A cascade reloads at SolveLoop
    # entry, at a committed T/H step and at a committed search trial, and a T/H
    # and a search in ONE outer are one reload.  If this residual is not zero,
    # the whole re-attribution of the search's true cost is void.
    cas = need("cascade")
    if isinstance(cas, dict) and all(
            k in cas for k in ("cascades", "solve_loops", "th_updates",
                               "search_trials", "th_search_coincident",
                               "predicted", "residual")):
        predicted = (cas["solve_loops"] + cas["th_updates"] + cas["search_trials"]
                     - cas["th_search_coincident"])
        if cas["predicted"] != predicted:
            bad.append(f"cascade.predicted is {cas['predicted']}, the identity "
                       f"gives {predicted}")
        if cas["residual"] != cas["cascades"] - cas["predicted"]:
            bad.append("cascade.residual is not cascades - predicted")
        # A run with equilibrium Xe must close the identity.  A run without it
        # never opens a cascade at all, and that is the one legal zero.
        if cas["cascades"] != 0 and cas["residual"] != 0:
            bad.append(f"the cascade identity does not close (residual "
                       f"{cas['residual']}): the three reload sites and the four "
                       "counters have drifted apart, so every per-trial cost "
                       "derived from them is void")

    # Design Sec 1.6: the sweeps nobody charges for.  BICGCMFD splits every
    # sweep it counts, so this is an identity and not an estimate.
    cmfd = need("cmfd")
    if isinstance(cmfd, dict):
        for k in ("sweeps", "sweeps_charged", "negative_retry_sweeps",
                  "sweep_split_residual", "drives", "exit_converged",
                  "exit_budget", "exit_deferred", "sweep_budget"):
            if k not in cmfd:
                bad.append(f"cmfd is missing {k!r}")
        if not bad:
            split = cmfd["sweeps_charged"] + cmfd["negative_retry_sweeps"]
            if cmfd["sweep_split_residual"] != cmfd["sweeps"] - split:
                bad.append("cmfd.sweep_split_residual is not sweeps - "
                           "(charged + negative_retry)")
            if cmfd["sweeps"] != split:
                bad.append(f"cmfd sweeps {cmfd['sweeps']} != charged + retries "
                           f"{split}: a sweep was counted in neither bucket, so "
                           "the Sec 1.6 loss cannot be read")
            if cmfd["negative_retry_sweeps"] < 0:
                bad.append("cmfd.negative_retry_sweeps is negative")
            # S3's K0 gate reads exit_budget / drives, so the denominator has to
            # be the drives whose exit was actually observed -- never the
            # segment-summed ones, whose exit the device never published.
            if cmfd["drives"] != cmfd["exit_converged"] + cmfd["exit_budget"]:
                bad.append("cmfd.drives != exit_converged + exit_budget: a drive "
                           "ended without being classified, or twice")

    settle = need("settle")
    if isinstance(settle, dict) and all(
            k in settle for k in ("outers", "outers_loose", "outers_polish")):
        if settle["outers"] != settle["outers_loose"] + settle["outers_polish"]:
            bad.append("settle.outers != outers_loose + outers_polish")
        if phases is not None and "settle" in phases and settle["outers"] != phases["settle"]:
            bad.append("settle.outers disagrees with outers_by_phase.settle")

    sent = need("sentinel")
    if isinstance(sent, dict) and all(
            k in sent for k in ("xe_step", "xe_interim", "settle", "polish", "total")):
        parts = sent["xe_step"] + sent["xe_interim"] + sent["settle"] + sent["polish"]
        if sent["total"] != parts:
            bad.append("sentinel.total is not the sum of its four sites")
        # The settling gate's sentinel fires on exactly the outers charged to the
        # settle bucket, so a mismatch means one of the two is counted somewhere
        # the other is not.
        if isinstance(settle, dict) and "outers" in settle and sent["settle"] != settle["outers"]:
            bad.append("sentinel.settle != settle.outers: the settling gate's "
                       "poison and its outer are charged at different sites")

    cf = need("counterfactual")
    if isinstance(cf, dict) and all(
            k in cf for k in ("probed", "would_converge", "loose_probed",
                              "loose_would_converge", "hit_rate", "probe_coverage")):
        if cf["would_converge"] > cf["probed"]:
            bad.append("counterfactual.would_converge exceeds probed")
        if cf["loose_probed"] > cf["probed"]:
            bad.append("counterfactual.loose_probed exceeds probed")
        if cf["loose_would_converge"] > cf["loose_probed"]:
            bad.append("counterfactual.loose_would_converge exceeds loose_probed")
        if isinstance(sent, dict) and "xe_step" in sent and cf["probed"] > sent["xe_step"]:
            bad.append("counterfactual.probed exceeds the number of Xe handoffs "
                       "that armed it")
        if cf["probed"] and abs(cf["hit_rate"] - cf["would_converge"] / cf["probed"]) > 5e-4:
            bad.append("counterfactual.hit_rate is not would_converge / probed")
    return bad


# The one fixture every identity holds on.  Shaped like the design's KNGR
# profile so a reader can recognise the numbers, but the test is about the
# arithmetic and not about the values.
FIXTURE = {
    "schema_version": 1, "slot": -1, "statepoints": 35,
    "outers": 4377, "outers_attributed": 4377, "outers_per_statepoint": 125.057,
    "outers_by_phase": {"initial": 347, "xe": 3184, "th": 86, "search": 707,
                        "settle": 53, "fallback": 0},
    "cascade": {"cascades": 228, "solve_loops": 69, "th_updates": 126,
                "search_trials": 137, "th_search_coincident": 104,
                "predicted": 228, "residual": 0, "outers_per_cascade": 19.197,
                "xe_steps": 1192, "xe_steps_per_cascade": 5.228},
    "search": {"trials": 137, "trials_per_statepoint": 3.914,
               "outers_per_trial_bucket": 5.161, "outers_per_trial_all_in": 24.358,
               "proposals": 139, "refused": 2, "iterations": 137},
    "xe": {"steps": 1192, "interim_steps": 0, "cascades": 228, "budget_exhausted": 0},
    "settle": {"outers": 53, "outers_loose": 0, "outers_polish": 53},
    "sentinel": {"xe_step": 964, "xe_interim": 0, "settle": 53, "polish": 69,
                 "total": 1086},
    "counterfactual": {"probed": 964, "would_converge": 96, "hit_rate": 0.0996,
                       "probe_coverage": 1.0, "loose_probed": 800,
                       "loose_would_converge": 80, "loose_hit_rate": 0.1},
    "cmfd": {"sweeps": 18627, "sweeps_per_outer": 4.2556, "sweeps_charged": 18512,
             "negative_retry_sweeps": 115, "negative_retry_frac": 0.0062,
             "sweep_split_residual": 0, "drives": 4377, "exit_converged": 700,
             "exit_budget": 3677, "exit_aborted": 0, "exit_deferred": 0,
             "exit_attributed_frac": 1.0, "budget_exhausted_frac": 0.8401,
             "deferred_sweeps_per_drive": 0.0, "sweep_budget": 5,
             "inner_budget": 3, "bicg_iters": 74508,
             "nmax_work": 4377 * 18627 * 4},
    "flux_limit_retries": 0, "staged_relapses": 0,
}


def identity_half() -> None:
    problems = check_receipt(json.loads(json.dumps(FIXTURE)))
    if problems:
        fail("the reference fixture does not satisfy the receipt's own "
             "identities: " + "; ".join(problems))

    # NEGATIVE CONTROLS.  Each mutation must be caught, and by the check that
    # exists for it -- a checker that rejects everything is as useless as one
    # that rejects nothing.
    def mutated(**path_values):
        r = json.loads(json.dumps(FIXTURE))
        for path, value in path_values.items():
            node = r
            parts = path.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
        return r

    controls = {
        "an outer charged to two phases":
            mutated(**{"outers_by_phase.xe": 3185}),
        "a phase dropped from the split":
            mutated(**{"outers_by_phase.settle": 0}),
        "the cascade identity broken":
            mutated(**{"cascade.search_trials": 150}),
        "a cascade residual that is not zero":
            mutated(**{"cascade.cascades": 230, "cascade.residual": 2}),
        "a sweep in neither bucket":
            mutated(**{"cmfd.sweeps_charged": 18000}),
        "a drive that ended without a classification":
            mutated(**{"cmfd.exit_budget": 3000}),
        "the settle split not adding up":
            mutated(**{"settle.outers_loose": 10}),
        "the sentinel total not adding up":
            mutated(**{"sentinel.total": 1000}),
        "a counterfactual hit rate over its own denominator":
            mutated(**{"counterfactual.would_converge": 2000}),
        "a hit rate that does not match its two counters":
            mutated(**{"counterfactual.hit_rate": 0.5}),
        "a receipt missing a whole section":
            {k: v for k, v in FIXTURE.items() if k != "cascade"},
    }
    for what, broken in controls.items():
        if not check_receipt(broken):
            fail(f"negative control passed: check_receipt() accepted {what}")


# ===========================================================================
# SOURCE HALF
# ===========================================================================
NEW_COUNTERS = (
    "settle_outers_loose",
    "poison_xe_step", "poison_xe_interim", "poison_settle", "poison_polish",
    "poison_probed", "poison_would_converge",
    "poison_probed_loose", "poison_would_converge_loose",
    "cmfd_drives", "cmfd_drives_converged", "cmfd_drives_budget",
    "cmfd_drives_aborted", "cmfd_drives_deferred", "cmfd_sweeps_deferred",
    "cmfd_sweeps_charged", "cmfd_negative_retry_sweeps",
)


def source_half() -> None:
    # REGIONS COME FROM THE RAW SOURCE, because the anchors that name a
    # namespace's end ARE comments; the stripped copy is used only where a
    # comment could otherwise satisfy a scan.
    code = DRIVER
    bare = strip_comments(DRIVER)

    # ---------------------------------------------------------------- 1. site
    # One emitter, unconditional, at Drive()'s body level (eight spaces), and
    # BEFORE the trajectory line -- every consumer in tools/ tails for that line
    # and the neutrality comparison reads it positionally.
    if bare.count(RECEIPT) != 1:
        fail(f"the {RECEIPT} receipt must be emitted from exactly one site "
             f"(found {bare.count(RECEIPT)} outside comments)")
    if code.count("outerbudget::line(") != 1:
        fail("the outer-budget receipt is formatted from more than one site")
    emit = code.find("std::cout << outerbudget::line(")
    if emit < 0:
        fail("the outer-budget receipt is not written to stdout by outerbudget::line")
    else:
        line_start = code.rfind("\n", 0, emit) + 1
        if code[line_start:emit] != " " * 8:
            fail("the outer-budget receipt is indented into a block; like the "
                 "trajectory receipt it must be unconditional at Drive()'s body "
                 "level, or a run that did not print it cannot be told from a "
                 "run that had nothing to print")
        traj = code.find(f'"{TRAJECTORY} ')
        if traj < 0 or emit > traj:
            fail("the outer-budget receipt is emitted after the trajectory "
                 "receipt; the trajectory line must stay the last line of a run")

    # -------------------------------------------------- 2. outside the gate
    # The design forbids mixing RASBERY_STATEPOINT_TELEMETRY with a timing arm
    # (Sec 4.2), so a budget behind that gate could never be quoted beside a
    # cases/hour number -- which is the only place the programme spends its
    # conclusions.
    if "sptelem::enabled" in region(code, "namespace outerbudget {",
                                    "} // namespace outerbudget", "outerbudget"):
        fail("outerbudget reads the telemetry gate; the budget receipt is "
             "always on, and that is the property that lets it be read beside a "
             "wall-clock arm")
    if "ob_run.accumulate(ctx.telemetry);" not in code:
        fail("the run accumulator is not folded from the per-statepoint counters")
    fold = code.find("ob_run.accumulate(ctx.telemetry);")
    if fold >= 0:
        # Walk back to the nearest `if (sp_telem) {` and make sure its block
        # closed before the fold: the fold must not be inside the gate.
        gate = code.rfind("if (sp_telem) {", 0, fold)
        if gate >= 0:
            depth, j = 0, code.find("{", gate)
            while j < fold:
                if code[j] == "{":
                    depth += 1
                elif code[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= fold:
                fail("ob_run is folded INSIDE an `if (sp_telem)` block; a run "
                     "with the telemetry off would then publish an empty budget")

    # ------------------------------------------------ 3. folded exactly once
    # A per-statepoint counter that never reaches accumulate() reads as zero for
    # the whole run and looks like a feature that never fired.
    counters = region(code, "struct Counters {", "} // namespace sptelem",
                      "sptelem::Counters")
    accumulate = region(counters, "void accumulate(const Counters& step) {",
                        "\n    }", "accumulate")
    for name in NEW_COUNTERS:
        if not re.search(rf"long long {name}\s", counters):
            fail(f"sptelem::Counters has no {name} member")
        folds = len(re.findall(rf"\b{name}\s*\+=\s*step\.{name};", accumulate))
        if folds != 1:
            fail(f"{name} is folded into the run total {folds} times, expected 1")

    # ------------------------------------- 4. the four sentinel sites, once each
    # The design's Sec 1.4 turns on how large each site's population is, and the
    # four sites are the only places `prev_inner` is poisoned inside SolveLoop.
    # One counter each, no site with two, and no site with none.
    solve_loop = region(code, "    static void SolveLoop(",
                        "    /// Where this run's restart_", "SolveLoop")
    # A STATEMENT, NOT THE DECLARATION.  SolveLoop opens with
    # `double prev_inner = eigv + 1.0;`, which is the loop arming itself rather
    # than a handoff poisoning a converged carry, so the scan anchors on a line
    # that STARTS with the assignment.
    sentinels = [m.start() for m in
                 re.finditer(r"^\s*prev_inner\s*=\s*eigv \+ 1\.0;", solve_loop, re.M)]
    if len(sentinels) != 4:
        fail(f"SolveLoop has {len(sentinels)} `prev_inner = eigv + 1.0` sites, "
             "expected the four the design names (Xe handoff, interim probe, "
             "settling gate, polish transition). A new one needs a counter and "
             "a row in the receipt")
    poison_counters = ("poison_xe_step", "poison_xe_interim", "poison_settle",
                       "poison_polish")
    for name in poison_counters:
        n = solve_loop.count(f"++ctx.telemetry.{name};")
        if n != 1:
            fail(f"{name} is charged {n} times in SolveLoop, expected exactly 1: "
                 "a sentinel site attributed twice would inflate the population "
                 "the whole Sec 1.4 dispute is about")
    # ...and each counter sits at its own site, within a few lines of it.
    for start, name in zip(sentinels, poison_counters):
        window = solve_loop[start:start + 1400]
        charged = [n for n in poison_counters if f"++ctx.telemetry.{n};" in window]
        if not charged or charged[0] != name:
            fail(f"the sentinel site at offset {start} is followed by "
                 f"{charged[:1] or 'no counter'}, expected {name!r}: the four "
                 "sites are counted in source order and one has moved")

    # ------------------------------ 5. the counterfactual is evaluated, not used
    # This is what keeps S0 free.  The probe reads the same two terms the ladder
    # reads, against the eigenvalue the sentinel replaced, and writes only
    # counters -- so the digest cannot move and S4 can be PRICED before it is
    # written.
    if "cf_armed" not in solve_loop or "cf_eigv" not in solve_loop:
        fail("SolveLoop no longer carries the S0 counterfactual probe")
    for consumer in ("prev_inner", "eigv ", "residual", "clean_iters",
                     "flux_converged", "polishing", "exit_reason"):
        if re.search(rf"^\s*{re.escape(consumer)}\s*=[^=\n]*cf_(eigv|armed|hit|probe)",
                     solve_loop, re.M):
            fail(f"the counterfactual reaches {consumer!r}; it must be evaluated "
                 "and never used, or the run it measures is not the run it "
                 "reports")
    for name in ("cf_hit", "cf_loose_hit", "cf_probe", "cf_loose"):
        if f"const bool {name}" not in solve_loop:
            fail(f"the counterfactual's {name} is not a const bool; the probe "
                 "must be four names the compiler can fold away, not a branch")
    digest = region(code, "struct Digest {", "\n};", "trajectory::Digest")
    for name in NEW_COUNTERS + ("cf_eigv", "cf_armed"):
        if name in digest:
            fail(f"the trajectory digest folds {name!r}; instrumentation that "
                 "moved the digest could never prove it moved nothing")

    # ------------------------------------- 6. BICGCMFD charges every drive exit
    bicg = strip_comments(BICG)
    if "_drive_exits             = DriveExits{};" not in bicg:
        fail("resetIteration() does not clear the drive-exit census; it would "
             "then accumulate across SolveLoops while cmfd_sweeps does not")
    charge = region(bicg, "void BICGCMFD::chargeDriveExit(double errl2) {", "\n}",
                    "chargeDriveExit")
    if "errl2 < _epsl2" not in charge:
        fail("chargeDriveExit no longer classifies on `errl2 < _epsl2` -- the "
             "drive's OWN exit test, which is what makes one rule cover the host "
             "loop, the device states and the Rayleigh hand-back")
    if charge.count("++_drive_exits.") != 3:
        fail("chargeDriveExit does not charge exactly one total and one of two "
             "buckets")
    # Every function that can END a drive has to charge it.  A `return true`
    # from one of these IS the drive finishing.
    for fn, end in (("bool BICGCMFD::driveDeviceSweeps(", "\n}"),
                    ("bool BICGCMFD::finishDrive(", "\n}"),
                    ("bool BICGCMFD::finishDeferredDrives(", "\n}")):
        body = region(bicg, fn, end, fn)
        returns_true = body.count("return true;")
        charges = body.count("chargeDriveExit(errl2);")
        # finishDeferredDrives has one `return true` that hands a whole SEGMENT
        # back, whose per-drive exits the device never published; it is charged
        # to the deferred bucket instead, so it is one short by design.
        expected = returns_true - (1 if "finishDeferredDrives" in fn else 0)
        if charges != expected:
            fail(f"{fn.strip()} charges {charges} drive exits against "
                 f"{returns_true} `return true` paths (expected {expected}): a "
                 "drive ended without landing in the exit census")
    if "deferred_drives += static_cast<long long>(acc.launches) - (exceptional ? 1 : 0)" \
            not in bicg:
        fail("the segment-summed drives are not excused the one the host tail "
             "finishes; that drive would be counted in two buckets")
    if "if (attempts > 0) {" not in bicg:
        fail("the negative-flux retry census is not guarded on the window "
             "`attempts` measures; finishDeferredDrives re-enters "
             "absorbSweepLaunch with attempts deliberately zero and would "
             "subtract that launch's sweeps a second time")

    # ------------------------------------------------------------- 7. schema
    emitter = region(code, "inline std::string line(const sptelem::Counters& c",
                     "\n} // namespace outerbudget", "outerbudget::line")
    for field in ("outers_by_phase", "outers_attributed", "outers_per_statepoint",
                  "cascade", "predicted", "residual", "outers_per_cascade",
                  "trials_per_statepoint", "outers_per_trial_all_in",
                  "interim_steps", "outers_loose", "sentinel",
                  "counterfactual", "probe_coverage", "loose_hit_rate",
                  "sweeps_per_outer", "negative_retry_sweeps",
                  "sweep_split_residual", "exit_budget", "exit_deferred",
                  "budget_exhausted_frac", "deferred_sweeps_per_drive",
                  "nmax_work", "schema_version"):
        if f'\\"{field}\\"' not in emitter:
            fail(f"the receipt no longer publishes {field!r}")
    # Every name the checker above reads must exist in the emitter, or the model
    # and the line have drifted apart and this whole file checks nothing.
    published = set(re.findall(r'\\"([a-z_0-9]+)\\":', emitter))
    for section, keys in (("", ("outers", "outers_attributed", "statepoints",
                                "flux_limit_retries", "staged_relapses")),
                          ("cascade", tuple(FIXTURE["cascade"])),
                          ("cmfd", tuple(FIXTURE["cmfd"])),
                          ("settle", tuple(FIXTURE["settle"])),
                          ("sentinel", tuple(FIXTURE["sentinel"])),
                          ("counterfactual", tuple(FIXTURE["counterfactual"]))):
        for key in keys:
            if key not in published:
                fail(f"the fixture and the checker read {section or 'the top level'}"
                     f".{key}, which the C++ emitter does not publish")

    # ---------------------------------------------- 8. negative source controls
    def scan_fails(mutation: str, replacement: str, what: str) -> None:
        broken = code.replace(mutation, replacement, 1)
        if broken == code:
            fail(f"negative control {what!r}: the pattern to break was not found")
            return
        # Re-run the scan the mutation is aimed at, on the broken source.
        sl = region(broken, "    static void SolveLoop(",
                    "    /// Where this run's restart_", "SolveLoop")
        sites = len(re.findall(r"^\s*prev_inner\s*=\s*eigv \+ 1\.0;", sl, re.M))
        if sites == 4 and all(sl.count(f"++ctx.telemetry.{n};") == 1
                              for n in poison_counters):
            fail(f"negative control {what!r} did not disturb the sentinel scan")

    scan_fails("++ctx.telemetry.poison_settle;", "", "a sentinel site with no counter")
    scan_fails("++ctx.telemetry.poison_polish;",
               "++ctx.telemetry.poison_polish;\n++ctx.telemetry.poison_polish;",
               "a sentinel site counted twice")


# ===========================================================================
# LIVE HALF
# ===========================================================================
LINE_RE = re.compile(r"\[RASBERY\]\[OUTER_BUDGET\]\s+(\{.*\})\s*$")


def live(paths: list[str]) -> int:
    rc = 0
    for path in paths:
        found = [m.group(1) for line in Path(path).read_text(errors="replace").splitlines()
                 if (m := LINE_RE.search(line))]
        if not found:
            print(f"{path}: no {RECEIPT} line")
            rc = 1
            continue
        for raw in found:
            problems = check_receipt(json.loads(raw))
            if problems:
                rc = 1
                for p in problems:
                    print(f"{path}: FAIL {p}")
            else:
                r = json.loads(raw)
                print(f"{path}: PASS outers/sp={r['outers_per_statepoint']:.1f} "
                      f"sweeps/outer={r['cmfd']['sweeps_per_outer']:.3f} "
                      f"budget_exhausted={r['cmfd']['budget_exhausted_frac']:.3f} "
                      f"(attributed {r['cmfd']['exit_attributed_frac']:.2f}) "
                      f"cf_hit={r['counterfactual']['hit_rate']:.3f} "
                      f"(coverage {r['counterfactual']['probe_coverage']:.2f})")
    return rc


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs="+", metavar="RUN.log",
                    help="run the identities against a real run's receipt")
    args = ap.parse_args(argv)
    if args.check:
        return live(args.check)
    identity_half()
    source_half()
    if FAILED:
        print("outer budget receipt contract: FAIL")
        for message in FAILED:
            print(f"  {message}")
        return 1
    print("outer budget receipt contract: PASS "
          "(11 identity negative controls, 4 sentinel sites, "
          f"{len(NEW_COUNTERS)} counters folded once each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
