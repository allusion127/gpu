#!/usr/bin/env python3
"""Contract: the default-promotion gate refuses the things it exists to refuse.

A promotion gate that only ever says DEFAULT_ON is a rubber stamp, and a
promotion gate nobody has watched say HOLD is indistinguishable from one.  So
every rule in WP11's "기본값 승격 규칙" and every threshold in plan §9 is driven
here from a synthetic gate block, once passing and once failing, with the
failing case required to name the reason.

WHAT IS PINNED.

  * §9 is per feature, not one number.  xslib_cache passes at +3.5 % and
    cmfd_compaction fails at the same gain, because its bar is +5 % -- a single
    global threshold would let the cheap levers in and keep the expensive ones
    out, which is backwards.
  * A §9 side condition that the block does not STATE is unproven, not passed.
  * A missing soak is not a clean soak.  A missing verdict is not a passed one.
  * N1 needs a recorded project acceptance.  A passed gate is not an approval,
    and this is the rule most likely to be waived by someone in a hurry.
  * A2 and L3coarse can NEVER be a default, whatever they measured.
  * An A/B whose two arms ran at different fidelities measures the fidelity.
  * A lever measured in one execution mode is MODE_DEPENDENT, not DEFAULT_ON.
  * No env rollback means no default: backing it out would need a rebuild.
  * A malformed block is an INPUT FAULT (exit 1), and "nothing qualified" is a
    normal answer (exit 0) -- an exit code that punished the normal answer would
    train everybody to ignore it.
  * The script never writes to any source file.  Checked by reading its text:
    the whole point is that flipping a default is a separate reviewed commit.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import promotion_gate as gate  # noqa: E402

FAILED: list[str] = []


def fail(msg: str) -> None:
    FAILED.append(msg)


def block(**overrides) -> dict:
    """A clean B0 block that qualifies, before each test breaks one thing."""
    value = {
        "schema": gate.SCHEMA,
        "name": "xslib_cache",
        "flag": "RASBERY_XSLIB_CACHE",
        "tip": "4a477b4",
        "grade": "B0",
        "env_rollback": True,
        "arms": [
            {"name": "off", "cases_per_hour": 500.0, "wall_s": 460.8,
             "digest": "814201df0583e1d2", "policy": "A2", "cases": 64, "ok": 64},
            {"name": "on", "cases_per_hour": 517.5, "wall_s": 445.2,
             "digest": "814201df0583e1d2", "policy": "A2", "cases": 64, "ok": 64},
        ],
        "baseline_arm": "off",
        "candidate_arm": "on",
        "verdicts": {"B0": {"feature_off_byte_identity": True,
                            "on_twice_deterministic": True,
                            "batch_equals_single": True,
                            "digest_match": True}},
        "soak": {"pass": True, "report": "soak_report.json"},
    }
    value.update(overrides)
    return value


def verdict_of(**overrides) -> gate.Verdict:
    return gate.evaluate(block(**overrides))


def expect(label: str, want: str, /, *, blocker: str | None = None, **overrides) -> None:
    """*label* is positional-only so a test may override the block's own `name`."""
    got = verdict_of(**overrides)
    if got.verdict != want:
        fail(f"{label}: expected {want}, got {got.verdict} "
             f"(reasons={got.reasons}, blockers={got.blockers})")
        return
    if blocker is not None and not any(blocker in b for b in got.blockers):
        fail(f"{label}: the verdict does not name the reason ({blocker!r} not in "
             f"{got.blockers})")


# ---------------------------------------------------------------------------
# The happy path, and the bar being per feature
# ---------------------------------------------------------------------------
expect("a clean B0 block at +3.5% against a +3% bar", gate.DEFAULT_ON)

# The SAME gain, a different lever, a different bar.  §9 is a table.
expect("cmfd_compaction at +3.5% against its +5% bar", gate.HOLD,
       blocker="bar of +5%", name="cmfd_compaction",
       extra_gains={"padding_reduction": 0.34})

# ... and the same lever at the gain its own bar asks for, WITH the side
# condition §9 attaches to it.
expect("cmfd_compaction at +6% with 34% padding reduction", gate.DEFAULT_ON,
       name="cmfd_compaction",
       arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "A2",
              "cases": 64, "ok": 64},
             {"name": "on", "cases_per_hour": 530.0, "policy": "A2",
              "cases": 64, "ok": 64}],
       extra_gains={"padding_reduction": 0.34})

# A SIDE CONDITION THE BLOCK DOES NOT STATE IS UNPROVEN, not satisfied.
expect("cmfd_compaction with no padding number at all", gate.HOLD,
       blocker="does not state it", name="cmfd_compaction",
       arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "A2"},
             {"name": "on", "cases_per_hour": 530.0, "policy": "A2"}])
expect("cmfd_compaction with a padding number below the bar", gate.HOLD,
       blocker="padding_reduction", name="cmfd_compaction",
       arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "A2"},
             {"name": "on", "cases_per_hour": 530.0, "policy": "A2"}],
       extra_gains={"padding_reduction": 0.10})

# A lever §9 does not list falls back to the general criterion, not to zero.
expect("an unlisted lever at +3.5% against the general +5%", gate.HOLD,
       blocker="bar of +5%", name="some_new_lever", flag="RASBERY_NEW")


# ---------------------------------------------------------------------------
# The accuracy gate its grade requires
# ---------------------------------------------------------------------------
expect("a B0 block that states no verdicts", gate.HOLD,
       blocker="states no B0 verdicts", verdicts={})
expect("a B0 block missing one verdict", gate.HOLD,
       blocker="'digest_match' was not stated",
       verdicts={"B0": {"feature_off_byte_identity": True,
                        "on_twice_deterministic": True,
                        "batch_equals_single": True}})
expect("a B0 block whose byte identity failed", gate.HOLD,
       blocker="feature_off_byte_identity",
       verdicts={"B0": {"feature_off_byte_identity": False,
                        "on_twice_deterministic": True,
                        "batch_equals_single": True, "digest_match": True}})


# ---------------------------------------------------------------------------
# The soak is a REQUIREMENT, not a follow-up
# ---------------------------------------------------------------------------
expect("no soak result", gate.HOLD, blocker="no soak result", soak=None)
expect("a soak that did not pass", gate.HOLD, blocker="soak did not pass",
       soak={"pass": False, "report": "soak_report.json"})


# ---------------------------------------------------------------------------
# N1 needs a recorded acceptance -- a passed gate is not an approval
# ---------------------------------------------------------------------------
N1_VERDICTS = {"N1": {"gate_a": True, "gate_b": True, "digest_repeat": True,
                      "pin_rms": True, "boron_ao": True}}
expect("an N1 lever with every gate passed and no acceptance record", gate.HOLD,
       blocker="project acceptance", name="xe_transaction", grade="N1",
       flag="RASBERY_GPU_XE_TXN", verdicts=N1_VERDICTS,
       arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "A2"},
             {"name": "on", "cases_per_hour": 530.0, "policy": "A2"}])
expect("an N1 lever with an acceptance record", gate.DEFAULT_ON,
       name="xe_transaction", grade="N1", flag="RASBERY_GPU_XE_TXN",
       verdicts=N1_VERDICTS,
       arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "A2"},
             {"name": "on", "cases_per_hour": 530.0, "policy": "A2"}],
       project_acceptance={"approved": True, "by": "campaign", "ref": "WP11-2026-08"})
expect("an N1 acceptance with no reference to anything", gate.HOLD,
       blocker="project acceptance", name="xe_transaction", grade="N1",
       flag="RASBERY_GPU_XE_TXN", verdicts=N1_VERDICTS,
       arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "A2"},
             {"name": "on", "cases_per_hour": 530.0, "policy": "A2"}],
       project_acceptance={"approved": True})


# ---------------------------------------------------------------------------
# A2 and coarse are mode-specific and can never be a default
# ---------------------------------------------------------------------------
for grade in ("A2", "L3coarse"):
    got = verdict_of(grade=grade, name="staged_tolerance",
                     flag="RASBERY_STAGED_FLUX_TOL",
                     arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "strict"},
                           {"name": "on", "cases_per_hour": 900.0, "policy": "A2"}],
                     verdicts={grade: dict.fromkeys(gate.GRADE_VERDICTS[grade], True)})
    if got.verdict != gate.NEVER:
        fail(f"a {grade} lever measuring +80% got {got.verdict}; A2/coarse is "
             "mode-specific and does not replace the strict default whatever it "
             "measured")


# ---------------------------------------------------------------------------
# A/B hygiene
# ---------------------------------------------------------------------------
expect("two arms at two fidelities", gate.HOLD,
       blocker="different fidelities",
       arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "strict"},
             {"name": "on", "cases_per_hour": 900.0, "policy": "A2"}])
expect("a candidate arm that lost cases", gate.HOLD,
       blocker="cases ok",
       arms=[{"name": "off", "cases_per_hour": 500.0, "policy": "A2",
              "cases": 64, "ok": 64},
             {"name": "on", "cases_per_hour": 700.0, "policy": "A2",
              "cases": 64, "ok": 60}])
expect("a block with no candidate arm", gate.HOLD,
       blocker="baseline_arm and a candidate_arm", candidate_arm="nonexistent")


# ---------------------------------------------------------------------------
# Rollback and execution mode
# ---------------------------------------------------------------------------
expect("a lever with no env rollback", gate.HOLD,
       blocker="env_rollback", env_rollback=False)
expect("a lever measured in the single mode only", gate.MODE_DEPENDENT,
       mode="single")
expect("a lever measured in both modes", gate.DEFAULT_ON, mode="both")


# ---------------------------------------------------------------------------
# Malformed input is an input fault, and a clean 'nothing qualified' is not
# ---------------------------------------------------------------------------
workdir = Path(tempfile.mkdtemp(prefix="rasbery_promo_"))
good = workdir / "good.json"
good.write_text(json.dumps(block()), encoding="utf-8", newline="\n")
held = workdir / "held.json"
held.write_text(json.dumps(block(soak=None)), encoding="utf-8", newline="\n")
wrong_schema = workdir / "wrong.json"
wrong_schema.write_text(json.dumps(block(schema="something/v9")), encoding="utf-8",
                        newline="\n")
out = workdir / "promotion.json"

if gate.main([str(held), "--json", str(out)]) != 0:
    fail("a run in which nothing qualified exits nonzero; the normal answer must not "
         "be an error, or nobody will read the exit code when it matters")
if gate.main([str(wrong_schema)]) == 0:
    fail("a block written to a different schema is accepted; the field names would be "
         "the same and the meanings would not")
if gate.main([str(workdir / "missing.json")]) == 0:
    fail("a missing block file is accepted")

report = json.loads(out.read_text(encoding="utf-8"))
if report.get("schema") != "rasbery-promotion/v1":
    fail("the JSON report has no schema tag")
if not report["verdicts"] or report["verdicts"][0]["verdict"] != gate.HOLD:
    fail(f"the JSON report does not carry the verdict: {report}")

text = gate.render([verdict_of(), verdict_of(soak=None)])
if "DEFAULT_ON" not in text or "HOLD" not in text:
    fail("the rendered table does not show both verdicts")
if "does not flip defaults" not in text:
    fail("the report does not say that nothing was changed; a promotion tool that "
         "reads as if it acted is a promotion tool somebody will believe acted")

# THE TOOL MUST NOT WRITE TO SOURCE.  Its whole premise is that flipping a
# default is a separate, reviewed commit.
source = (ROOT / "tools" / "promotion_gate.py").read_text(encoding="utf-8")
for forbidden in ("write_text(", "os.environ["):
    occurrences = source.count(forbidden)
    if forbidden == "write_text(" and occurrences > 1:
        fail("promotion_gate.py writes more than the one --json report; it must not "
             "touch anything else")
    if forbidden == "os.environ[" and occurrences:
        fail("promotion_gate.py sets an environment variable; it reports, it does not "
             "flip")

import shutil  # noqa: E402
shutil.rmtree(workdir, ignore_errors=True)

if FAILED:
    raise SystemExit("promotion gate contract: FAIL\n  - " + "\n  - ".join(FAILED))
print(f"promotion gate contract: PASS ({len(gate.SECTION_9)} §9 levers, "
      f"{len(gate.GRADE_VERDICTS)} grades)")
