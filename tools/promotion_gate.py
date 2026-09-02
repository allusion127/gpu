#!/usr/bin/env python3
"""WP11 -- which flags qualify for default-ON, as code rather than as a habit.

WHAT THIS IS NOT.  It is not `tools/test_ga_promotion_gate.py`, which audits the
GA two-stage 40x accounting claim.  This one answers a different question: after
a 238 gate block and a soak, is a FLAG allowed to become a default?

WHY IT IS CODE.  The plan states the rule in four lines (WP11 "기본값 승격
규칙") and the rule is easy to apply generously at 2 a.m. on a Friday:

    B0 기능       238 gate와 soak 통과 후 default ON 후보.
    N1 기능       정확성 envelope와 deterministic digest, PROJECT ACCEPTANCE
                  승인 후 ON.
    A2/coarse     mode-specific이고 strict default를 대체하지 않음.
    단일 이득/배치 회귀 기능은 execution-mode dependent default 허용.
    모든 기능은 재빌드 없는 env rollback을 유지한다.

Every one of those has a failure mode where the flag gets promoted anyway and
nothing looks wrong for a month: an N1 feature promoted without the acceptance
record, because the digest was deterministic and that felt like enough; an A2
knob promoted because it was faster and nobody was thinking about which column
its numbers go in; a flag that only helps at M1 promoted globally, quietly
costing the batch arm; a flag with no env rollback, so backing it out means a
rebuild on the one night that is not available.

THE PERFORMANCE BAR IS PER FEATURE, and it is the plan's §9 table -- not one
number.  XSLIB cache needs M64 +3 %; CMFD compaction needs +5 % AND a 30 %
padding drop; FlatXS cooperative needs kernel -30 % AND M64 +10 %.  A single
global +5 % would let the cheap ones in and keep the expensive ones out, which
is backwards.  Unlisted features fall back to the plan's general adoption
criterion: the target phase -25 % or the whole run +5 %.

THIS SCRIPT NEVER FLIPS A DEFAULT.  It reads the gate blocks, prints a verdict
per flag with the reason, and exits nonzero only when a block is malformed --
because "no flag qualified" is a legitimate, common and useful answer, and an
exit code that punished it would train everybody to ignore the exit code.

INPUT.  One JSON file per gate block, written by the 238 runner:

    {
      "schema": "rasbery-gate-block/v1",
      "name": "xslib_cache",            # the lever, as the plan names it
      "flag": "RASBERY_XSLIB_CACHE",    # the env var that turns it on
      "tip": "4a477b4",                 # the commit measured
      "grade": "B0",                    # B0 | N1 | A2 | L3coarse
      "env_rollback": true,             # off WITHOUT a rebuild
      "arms": [
        {"name": "off", "cases_per_hour": 518.2, "wall_s": 444.6,
         "digest": "814201df0583e1d2", "policy": "A2", "cases": 64, "ok": 64},
        {"name": "on",  "cases_per_hour": 541.9, "wall_s": 425.2,
         "digest": "814201df0583e1d2", "policy": "A2", "cases": 64, "ok": 64}
      ],
      "baseline_arm": "off",
      "candidate_arm": "on",
      "verdicts": {
        "B0": {"feature_off_byte_identity": true, "on_twice_deterministic": true,
               "batch_equals_single": true, "digest_match": true}
      },
      "extra_gains": {"padding_reduction": 0.34},   # named §9 side conditions
      "soak": {"pass": true, "report": "soak_report.json"},
      "mode": "batch"                   # batch | single | both  (optional)
    }

Anything the block does not state is NOT assumed.  A missing `soak` is not a
clean soak, a missing verdict is not a passed verdict, and both are reported as
what they are: unproven.

USE

    python tools/promotion_gate.py results/*.json
    python tools/promotion_gate.py results/*.json --json promotion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

SCHEMA = "rasbery-gate-block/v1"

# ---------------------------------------------------------------------------
# The plan's §9 table, as data.
# ---------------------------------------------------------------------------
#
# `gain` is the fractional throughput improvement the candidate arm must show
# over the baseline arm.  `extra` names side conditions the block has to state
# in `extra_gains` -- a number, and the minimum it must reach.  `note` is quoted
# back in the verdict so the reason a flag was held reads the way the plan
# states it rather than as a threshold nobody can source.


@dataclass(frozen=True)
class Bar:
    gain: float
    grade: str
    extra: tuple[tuple[str, float], ...] = ()
    note: str = ""


#: Plan §9, "정량적 채택 게이트 요약".  Keys are the block `name`.
SECTION_9: dict[str, Bar] = {
    "contract_telemetry": Bar(0.0, "B0", (("overhead", -0.01),),
                              "overhead <1%, 필수 기반"),
    "conditional_while": Bar(0.05, "B0", (), "단일 +5%, 배치 별도 판정"),
    "xslib_cache": Bar(0.03, "B0", (), "M64 +3% 또는 cold staircase 제거"),
    "cmfd_compaction": Bar(0.05, "B0", (("padding_reduction", 0.30),),
                           "M64 +5%, padding >=30% 감소"),
    "k_process": Bar(0.05, "B0", (), "K2 +5%, 미달 시 트랙 종료"),
    "flatxs_cooperative": Bar(0.10, "B0", (("kernel_reduction", 0.30),),
                              "kernel -30%, M64 +10%"),
    "flatxs_residency": Bar(0.05, "B0", (("transfer_reduction", 0.80),),
                            "transfer -80%, 전체 +5%, consumer audit 필수"),
    "ppr_device_loop": Bar(0.0, "N1", (("ppr_reduction", 0.25),),
                           "PPR -25%, host sync 제거"),
    "ppr_reconstruction": Bar(0.05, "N1", (("ppr_reduction", 0.40),),
                              "PPR 전체 -40%, 전체 +5%, Fq/FdH gate"),
    "xe_transaction": Bar(0.05, "N1", (), "Xe phase -20% 또는 전체 +5%"),
    "persistent_evaluator": Bar(0.05, "B0", (("process_cost", -0.01),),
                                "chunked 대비 +5%, process cost <1%"),
    "warm_start": Bar(0.0, "N1", (("outer_reduction", 0.20),),
                      "대상 outer -20%, cold fallback"),
}

#: What a lever NOT in §9 has to clear.  The plan's general adoption criterion
#: (WP-level "채택 기준"): the target phase by 25 %, or the whole run by 5 %.
DEFAULT_BAR = Bar(0.05, "B0", (), "계획 채택 기준: 대상 phase -25% 또는 전체 +5%")

#: The verdict fields each grade's 238 gate must state.  Plan §6.1.
GRADE_VERDICTS: dict[str, tuple[str, ...]] = {
    "B0": ("feature_off_byte_identity", "on_twice_deterministic",
           "batch_equals_single", "digest_match"),
    "N1": ("gate_a", "gate_b", "digest_repeat", "pin_rms", "boron_ao"),
    "A2": ("policy_a2_receipt", "strict_reevaluation_policy"),
    "L3coarse": ("screening_only", "promotion_rerun_strict"),
}

#: The verdicts.  ONE set of words, so a report cannot invent a fifth.
DEFAULT_ON = "DEFAULT_ON"
MODE_DEPENDENT = "MODE_DEPENDENT"
HOLD = "HOLD"
REJECT = "REJECT"
NEVER = "NEVER_DEFAULT"


@dataclass
class Verdict:
    name: str
    flag: str
    grade: str
    tip: str
    gain: float | None
    bar: float
    verdict: str
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


class BlockError(Exception):
    """A gate block this script cannot read.  Not a verdict -- an input fault."""


def load_block(path: Path) -> dict:
    try:
        block = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BlockError(f"{path}: {exc}") from exc
    if not isinstance(block, dict):
        raise BlockError(f"{path}: a gate block must be a JSON object")
    if block.get("schema") != SCHEMA:
        raise BlockError(
            f"{path}: schema is {block.get('schema')!r}, expected {SCHEMA!r}. A block "
            "written to a different schema may have the same field names and different "
            "meanings, and promoting a default off a misread block is exactly the "
            "mistake this gate exists to make impossible.")
    for field_name in ("name", "flag", "tip", "grade", "arms"):
        if not block.get(field_name):
            raise BlockError(f"{path}: the block has no {field_name!r}")
    return block


def arm(block: dict, which: str) -> dict | None:
    wanted = block.get(which)
    if not wanted:
        return None
    for entry in block["arms"]:
        if isinstance(entry, dict) and entry.get("name") == wanted:
            return entry
    return None


def evaluate(block: dict) -> Verdict:
    name = block["name"]
    bar = SECTION_9.get(name, DEFAULT_BAR)
    grade = str(block["grade"])
    result = Verdict(name=name, flag=str(block["flag"]), grade=grade,
                     tip=str(block["tip"]), gain=None, bar=bar.gain,
                     verdict=HOLD)
    result.reasons.append(f"bar: {bar.note}" if bar.note
                          else f"bar: +{bar.gain:.0%}")

    if grade not in GRADE_VERDICTS:
        result.verdict = REJECT
        result.blockers.append(
            f"grade {grade!r} is not one of {', '.join(GRADE_VERDICTS)} (plan §6.1). "
            "A lever with no accuracy grade has no gate to have passed.")
        return result

    # -- the measured gain ------------------------------------------------
    base = arm(block, "baseline_arm")
    cand = arm(block, "candidate_arm")
    if base is None or cand is None:
        result.blockers.append(
            "the block does not name both a baseline_arm and a candidate_arm that "
            "appear in `arms`; there is no A/B to read a gain from")
    else:
        b = base.get("cases_per_hour")
        c = cand.get("cases_per_hour")
        if not isinstance(b, (int, float)) or not isinstance(c, (int, float)) or b <= 0:
            result.blockers.append(
                "an arm has no positive `cases_per_hour`; throughput is what the bar is "
                "expressed in and a wall-only block cannot be compared to it")
        else:
            result.gain = (c - b) / b
            if result.gain < bar.gain:
                result.blockers.append(
                    f"measured {result.gain:+.2%} against a bar of +{bar.gain:.0%}")
            else:
                result.reasons.append(f"throughput {result.gain:+.2%}")
        # A/B HYGIENE.  Two arms measured at two fidelities are not an A/B of
        # the flag; they are an A/B of the fidelity with the flag along for the
        # ride.  Plan Sec 6.2: never mix strict and A2 in one table.
        if base.get("policy") != cand.get("policy"):
            result.blockers.append(
                f"the arms ran at different fidelities ({base.get('policy')!r} vs "
                f"{cand.get('policy')!r}); that comparison measures the policy, not "
                "the flag")
        for entry, label in ((base, "baseline"), (cand, "candidate")):
            cases, ok = entry.get("cases"), entry.get("ok")
            if isinstance(cases, int) and isinstance(ok, int) and ok < cases:
                result.blockers.append(
                    f"the {label} arm reports {ok}/{cases} cases ok; a throughput "
                    "number measured over a wave that lost candidates is a number "
                    "for a different wave")

    # -- the accuracy gate its grade requires ------------------------------
    verdicts = block.get("verdicts")
    verdicts = verdicts.get(grade) if isinstance(verdicts, dict) else None
    if not isinstance(verdicts, dict):
        result.blockers.append(
            f"the block states no {grade} verdicts. A missing verdict is not a passed "
            "verdict.")
    else:
        for required in GRADE_VERDICTS[grade]:
            value = verdicts.get(required)
            if value is None:
                result.blockers.append(f"{grade} verdict {required!r} was not stated")
            elif value is not True:
                result.blockers.append(f"{grade} verdict {required!r} is {value!r}")

    # -- the §9 side conditions -------------------------------------------
    extras = block.get("extra_gains") or {}
    for key, minimum in bar.extra:
        value = extras.get(key)
        if not isinstance(value, (int, float)):
            result.blockers.append(
                f"§9 requires {key} for this lever and the block does not state it")
        elif minimum >= 0 and value < minimum:
            result.blockers.append(f"{key} = {value:.2%}, §9 requires >= {minimum:.0%}")
        elif minimum < 0 and value > -minimum:
            result.blockers.append(f"{key} = {value:.2%}, §9 requires <= {-minimum:.0%}")

    # -- the soak ----------------------------------------------------------
    soak = block.get("soak")
    if not isinstance(soak, dict):
        result.blockers.append(
            "no soak result. WP11 makes the soak a REQUIREMENT for a default, not a "
            "follow-up: every defect a long-lived evaluator introduces is invisible "
            "for one generation.")
    elif soak.get("pass") is not True:
        result.blockers.append(f"the soak did not pass ({soak.get('pass')!r})")
    else:
        # WP10.7.  A SOAK THAT PASSED IS NOT A SOAK WHOSE CONTRACT HELD.
        #
        # The 238 arm-A soak carried `contract_pass:false` with
        # `outer_fallbacks:9` and `flatxs_fallbacks:4` beside it, and until
        # WP10.7 the soak's own verdict did not read either the outer counter or
        # the receipt's verdict -- so a block could arrive here with
        # `soak.pass:true` over a run in which thirteen cases were killed by the
        # fail-closed gate.  The soak now reports `gpu_full_contract`; a block
        # that carries it and says false is refused HERE too, because a flag
        # promoted on a run whose GPU arms were not engaged is a flag promoted
        # on the CPU's numbers.  A block that predates the field says nothing
        # and is neither credited nor blamed for it -- the same rule as every
        # other unstated fact in this file.
        contract = soak.get("gpu_full_contract")
        contract_pass = contract.get("contract_pass") if isinstance(contract, dict) else None
        if contract_pass is False:
            result.blockers.append(
                "the soak passed but its [RASBERY][GPU_FULL] receipt reports "
                "contract_pass:false"
                + (f", first_violation={contract.get('first_violation')!r}"
                   if isinstance(contract, dict) and contract.get("first_violation")
                   else "")
                + ". A default promoted on a run whose GPU arms fell back to CPU "
                  "numerics is a default promoted on the wrong measurement.")
        else:
            result.reasons.append(f"soak clean ({soak.get('report', 'inline')})")

    # -- WP10.8: the two facts a soak's own `pass` is allowed to be wrong about
    #
    # A SOAK THAT PASSED IS NOT A SOAK THAT EVALUATED EVERY CANDIDATE.  The 238
    # block-38 arm-B soak lost 18 of its 360 cases to a SIGSEGV that took the
    # evaluator down mid-generation, and the only record was a run-level total
    # short by 18 with no ids.  A flag promoted on a generation that silently
    # dropped a third of a wave is a flag promoted on a sample nobody chose, so
    # this is refused HERE too rather than trusted to the soak's own verdict --
    # and a block from a soak that predates the field says nothing and is
    # neither credited nor blamed, the same rule as every other unstated fact.
    #
    # A SLOPE FITTED ACROSS A RESTART IS NOT A LEAK MEASUREMENT.  The same run
    # reported RSS +115.97 MB/generation against a budget of 8 -- fitted through
    # the restart, so most of it was a fresh child re-warming.  A block whose
    # soak crossed a restart has a memory verdict nobody can read either way,
    # and "unreadable" is not "passed".
    if isinstance(soak, dict):
        accounted = soak.get("cases_accounted")
        missing = soak.get("cases_missing")
        if accounted is False or (isinstance(missing, list) and missing):
            names = ", ".join(str(m) for m in (missing or [])[:8]) or "unnamed"
            result.blockers.append(
                f"the soak did not account for every case it requested "
                f"({soak.get('cases_reported')} of {soak.get('cases_requested')} "
                f"reported; unaccounted for: {names}). A default promoted on a "
                "campaign that lost candidates mid-generation is a default promoted "
                "on whichever candidates happened to survive.")
        growth = soak.get("growth")
        if isinstance(growth, dict):
            for what in ("rss", "vram"):
                block_growth = growth.get(what)
                if not isinstance(block_growth, dict):
                    continue
                if block_growth.get("crossed_a_restart") is True:
                    result.blockers.append(
                        f"the soak's {what} slope was fitted over generations "
                        f"{block_growth.get('segment_first_generation')}.."
                        f"{block_growth.get('segment_last_generation')} of "
                        f"{len(block_growth.get('samples') or [])} because the "
                        "evaluator restarted mid-run. A slope across two process "
                        "lifetimes measures the new child's re-warm; re-run the soak "
                        "on a process that survives it before promoting.")

    # -- rollback ----------------------------------------------------------
    if block.get("env_rollback") is not True:
        result.blockers.append(
            "the block does not assert `env_rollback`. WP11 requires every feature to "
            "stay switchable off WITHOUT a rebuild; a default that needs a rebuild to "
            "back out is a default nobody can back out at 3 a.m.")

    # -- the grade's own promotion rule ------------------------------------
    if grade in ("A2", "L3coarse"):
        result.verdict = NEVER
        result.reasons.append(
            "A2/coarse is MODE-SPECIFIC and does not replace the strict default "
            "(WP11). It can be a documented arm; it cannot be what a run does when "
            "nobody said anything.")
        return result

    if result.blockers:
        result.verdict = HOLD
        return result

    if grade == "N1":
        acceptance = block.get("project_acceptance")
        if not (isinstance(acceptance, dict) and acceptance.get("approved") is True
                and acceptance.get("ref")):
            result.verdict = HOLD
            result.blockers.append(
                "N1 changes the trajectory. WP11 requires the accuracy envelope, a "
                "deterministic digest AND a recorded project acceptance before ON; "
                "state it as project_acceptance: {approved: true, by: ..., ref: ...}. "
                "A gate that passed is not an approval.")
            return result
        result.reasons.append(
            f"project acceptance {acceptance.get('ref')} "
            f"by {acceptance.get('by', 'unrecorded')}")

    mode = block.get("mode")
    if mode in ("single", "batch"):
        result.verdict = MODE_DEPENDENT
        result.reasons.append(
            f"gain measured in the {mode} execution mode only. WP11 allows an "
            "execution-mode dependent default for a lever that helps one mode and "
            "regresses the other -- and requires it to SAY which, rather than being "
            "promoted globally off one mode's number.")
        return result

    result.verdict = DEFAULT_ON
    return result


def render(verdicts: Sequence[Verdict]) -> str:
    lines = ["", "flag promotion (WP11) -- NOTHING BELOW HAS BEEN CHANGED", ""]
    width = max([len(v.name) for v in verdicts] + [4])
    lines.append(f"  {'lever'.ljust(width)}  {'grade':<9} {'gain':>8}  verdict")
    lines.append(f"  {'-' * width}  {'-' * 9} {'-' * 8}  {'-' * 14}")
    for v in verdicts:
        gain = "n/a" if v.gain is None else f"{v.gain:+.2%}"
        lines.append(f"  {v.name.ljust(width)}  {v.grade:<9} {gain:>8}  {v.verdict}")
    lines.append("")
    for v in verdicts:
        lines.append(f"{v.name}  [{v.flag}]  tip={v.tip}  ->  {v.verdict}")
        for reason in v.reasons:
            lines.append(f"    + {reason}")
        for blocker in v.blockers:
            lines.append(f"    - {blocker}")
        lines.append("")
    promoted = [v.name for v in verdicts if v.verdict == DEFAULT_ON]
    mode_dep = [v.name for v in verdicts if v.verdict == MODE_DEPENDENT]
    lines.append(f"qualifies for default-ON: {', '.join(promoted) if promoted else 'none'}")
    lines.append(f"execution-mode dependent: {', '.join(mode_dep) if mode_dep else 'none'}")
    lines.append("")
    lines.append("This script does not flip defaults. Editing the default is a separate, "
                 "reviewed commit that cites the block it came from.")
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blocks", type=Path, nargs="+",
                    help="gate block JSON files written by the 238 runner")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the verdicts as JSON")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    verdicts: list[Verdict] = []
    errors: list[str] = []
    for path in args.blocks:
        try:
            verdicts.append(evaluate(load_block(path)))
        except BlockError as exc:
            errors.append(str(exc))
    if verdicts:
        print(render(verdicts))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"schema": "rasbery-promotion/v1",
             "verdicts": [v.__dict__ for v in verdicts],
             "unreadable_blocks": errors}, indent=2) + "\n",
            encoding="utf-8", newline="\n")
    for message in errors:
        print("unreadable gate block: " + message, file=sys.stderr)
    # NONZERO ONLY ON AN UNREADABLE BLOCK.  "No flag qualified" is the normal
    # answer on most nights and an exit code that punished it would teach
    # everybody to stop reading the exit code.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
