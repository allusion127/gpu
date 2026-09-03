#!/usr/bin/env python3
"""WP10.4 -- the per-case receipt and the audit that reads it must spell the
fidelity the same way.

WHAT WENT WRONG, AND WHY NO EXISTING TEST SAW IT.

The host 181 soak at `91004f7` (2026-08-30, `tools/soak_run.py --deck
kngr_238.json --generations 5 --width 16`) came back FAIL with 83 of its 86
findings reading

    the per-case receipt is missing 'physics_fidelity' -- this binary predates
    WP10.3 and its cases cannot be audited on fidelity individually

on a binary that is NOT upstream of WP10.3.  Both halves were right about
themselves and wrong about each other:

  * `src/EvaluatorServer.h` reportCase() printed `physics_fidelity`, the plan
    Sec 6.2 spelling;
  * `src/Driver.h`'s own `[RASBERY][CASE]` line printed the SAME value under the
    name `fidelity`, because `tools/case_key.py` COMPONENT_FIELDS keys the case
    on that name and every manifest on disk carries it;
  * `exact_audit.parse_case_receipts()` reads BOTH tags into one list and
    `CASE_REQUIRED_FIELDS` audits every one of them for `physics_fidelity`.

So every Driver-side receipt in a real run was refused as pre-WP10.3.  The
existing harness could not see it: `tools/fake_rasbery_child.py` emits only
`[RASBERY][EVALUATOR][CASE]`, so `tools/test_soak_run.py` audits a stream that
never contains the tag that was wrong.  THAT is the hole this file closes -- it
checks the two sources against each other and against the audit's own field
list, and it exercises the Driver tag by name.

The fix is on both sides, and each half is load-bearing:
  * the Driver line now prints `physics_fidelity` BESIDE `fidelity`
    (schema_version 5).  Dropping the old name would invalidate every stored
    manifest and the case key computed from it.
  * `exact_audit.CASE_FIELD_SYNONYMS` accepts `fidelity` for
    `physics_fidelity`, so logs from binaries built before this commit audit
    cleanly -- WITHOUT weakening the version refusal, because the two fields
    that actually arrived in WP10.3 (`statepoint_grid`, `acceptance_eligible`)
    have no synonym.

Run: python3 tools/test_soak_receipt_schema_contract.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import exact_audit  # noqa: E402

DRIVER_H = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8", errors="replace")
SERVER_H = (ROOT / "src" / "EvaluatorServer.h").read_text(encoding="utf-8", errors="replace")
SOAK_PY = (ROOT / "tools" / "soak_run.py").read_text(encoding="utf-8", errors="replace")

FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)


# ---------------------------------------------------------------------------
# 1. THE TWO EMITTERS
# ---------------------------------------------------------------------------
def driver_case_block() -> str:
    """The `[RASBERY][CASE]` std::format call, format string and arguments."""
    marker = '"  [RASBERY][CASE] {{'
    if DRIVER_H.count(marker) != 1:
        fail("src/Driver.h must emit [RASBERY][CASE] from exactly one site; a "
             "second site is a second answer to what a case receipt says")
        return ""
    start = DRIVER_H.index(marker)
    return DRIVER_H[start:DRIVER_H.index(");", start)]


def server_case_block() -> str:
    """The `[RASBERY][EVALUATOR][CASE]` ostream chain in reportCase()."""
    marker = '"[RASBERY][EVALUATOR][CASE] {'
    if SERVER_H.count(marker) != 1:
        fail("src/EvaluatorServer.h must emit [RASBERY][EVALUATOR][CASE] from "
             "exactly one site")
        return ""
    start = SERVER_H.index(marker)
    end = SERVER_H.find("_out << line.str()", start)
    return SERVER_H[start:end if end > start else len(SERVER_H)]


def emitted_fields(block: str) -> set[str]:
    r"""Every JSON key the block writes, both `\"name\"` and `"name"` spellings."""
    return set(re.findall(r'\\"([A-Za-z_][A-Za-z0-9_]*)\\"\s*:', block)) | set(
        re.findall(r'"\\?"([A-Za-z_][A-Za-z0-9_]*)\\?"\s*:', block))


def emitters_carry_every_audited_field() -> None:
    """Both tags publish every field CASE_REQUIRED_FIELDS audits.

    Audited by the AUDIT's own list rather than a copy of it, so a field added
    to exact_audit and to only one emitter fails here instead of on a host.
    """
    for name, block in (("src/Driver.h [RASBERY][CASE]", driver_case_block()),
                        ("src/EvaluatorServer.h [RASBERY][EVALUATOR][CASE]",
                         server_case_block())):
        if not block:
            continue
        fields = emitted_fields(block)
        for required in exact_audit.CASE_REQUIRED_FIELDS:
            if required not in fields:
                fail(f"{name} does not publish {required!r}. "
                     f"exact_audit.CASE_REQUIRED_FIELDS audits BOTH case tags for "
                     f"it, so every receipt from this emitter is refused as "
                     f"'predates WP10.3' -- which is what the 181 soak reported "
                     f"83 times on a binary that was not.")
        # The old spelling is not optional either: tools/case_key.py keys the
        # case on `fidelity` and dropping it invalidates stored manifests.
        if name.startswith("src/Driver.h") and "fidelity" not in fields:
            fail("src/Driver.h [RASBERY][CASE] dropped 'fidelity'; "
                 "tools/case_key.py COMPONENT_FIELDS reads that name")


def driver_receipt_bumped_schema() -> None:
    block = driver_case_block()
    # WP24 took this to 7 when the line gained `fidelity_preset`; case-key v2
    # (2026-09-04) took it to 8 with exec_mode / the effective Xe arms /
    # forms_digest.  AT LEAST 7 is the contract: the fields this file audits
    # arrived at 7 and every later bump keeps them, so pinning an exact number
    # here would turn any future receipt field into a failure of this file.
    version = re.search(r'schema_version\\":(\d+)', block or "")
    if block and (version is None or int(version.group(1)) < 7):
        fail("[RASBERY][CASE] gained fields without bumping schema_version to 7 or "
             "later; a reader cannot tell a receipt that carries them from one "
             "that cannot")


def emitters_carry_a_per_case_identifier() -> None:
    """WP10.5.  A receipt has to name a CASE, not a duplicate class.

    THE 55c0dce REGRESSION.  With the WP10.4 field fix in, the Driver's receipts
    stopped being refused as pre-WP10.3 and immediately started being refused
    for a second reason: 82 of them, one per case, "ran at policy='strict' and
    no fidelity was declared for it".  [RASBERY][CASE] carried exactly one
    identifier -- case_key -- and a case key is the CANONICAL DUPLICATE key:
    every case of one deck at one fidelity in a cold wave has the same one, by
    design.  So the tag could not be matched to a per-case declaration in
    precisely the mixed-fidelity wave the per-case audit exists for.

    `output` is the identifier that works: unique per case by the evaluator's
    own wave namespace rule, present in argv, manifest and evaluator modes
    alike, and already in the Driver's hand -- nothing is plumbed to print it.
    """
    block = driver_case_block()
    if not block:
        return
    fields = emitted_fields(block)
    identifiers = [f for f in exact_audit.CASE_IDENTIFIER_FIELDS if f in fields]
    if identifiers == ["case_key"]:
        fail("[RASBERY][CASE]'s only identifier is 'case_key', which names a "
             "DUPLICATE CLASS and not a case: one deck at one fidelity gives a "
             "whole cold wave the same one. A per-case declaration can never be "
             "resolved for this tag -- 82 cases on host 181 at 55c0dce")
    if "output" not in fields:
        fail("[RASBERY][CASE] does not publish 'output'; it is the only per-case "
             "identifier the Driver holds in every mode")
    if not identifiers:
        fail("[RASBERY][CASE] carries no field in "
             "exact_audit.CASE_IDENTIFIER_FIELDS at all")

    server = server_case_block()
    if server and "key" not in emitted_fields(server):
        fail("[RASBERY][EVALUATOR][CASE] stopped echoing the client's 'key'")


def soak_declares_every_identifier() -> None:
    """The declarer has to key on a name the receipts actually carry."""
    if "def declare(" not in SOAK_PY:
        fail("tools/soak_run.py declares cases without the helper that registers "
             "every name a receipt could be identified by")
    if 'str(request["output"])' not in SOAK_PY:
        fail("tools/soak_run.py does not declare a case under its output path, so "
             "the Driver's own receipt has no declaration to resolve to")
    fake = (ROOT / "tools" / "fake_rasbery_child.py").read_text(
        encoding="utf-8", errors="replace")
    if '"  [RASBERY][CASE] "' not in fake:
        fail("tools/fake_rasbery_child.py does not emit [RASBERY][CASE]; the soak "
             "harness would again be exercising only the tag that was already "
             "right, which is how both WP10.4 and WP10.5 reached a host")
    if '"output": output' not in fake:
        fail("the fake's [RASBERY][CASE] carries no 'output', so the declaration "
             "path it is meant to exercise is not exercised")


def soak_audits_through_exact_audit() -> None:
    """One auditor, so there is one spelling to keep in step."""
    if "exact_audit.audit_case_fidelity(" not in SOAK_PY:
        fail("tools/soak_run.py no longer audits per-case fidelity through "
             "exact_audit.audit_case_fidelity; a second implementation is a "
             "second field list to drift")
    if "import exact_audit" not in SOAK_PY:
        fail("tools/soak_run.py does not import exact_audit")


# ---------------------------------------------------------------------------
# 2. THE AUDIT, BEHAVIOURALLY, ON THE DRIVER TAG
# ---------------------------------------------------------------------------
#: What the Driver line looks like after this commit.  Written out here rather
#: than generated from Driver.h on purpose: this is the contract, and a test
#: that derived it from the source under test could only ever agree with it.
DRIVER_RECEIPT = {
    "schema_version": 5, "case_key": "abc", "key_schema": "casekey/v1",
    "core_op": "op", "deck_digest": "d", "env_digest": "e", "env_set": "~",
    "xslib_digest": "x", "xslib_policy": "cached", "warm_start_token": "~",
    "code_sha": "sha", "fidelity": "full_exact", "physics_fidelity": "full_exact",
    "policy": "strict", "result_mode": "full", "output": "/w/out/g0000c0000.h5",
    "warm_start": "cold",
    "statepoint_grid": "full", "acceptance_eligible": True,
    "fidelity_declared": None, "promoted_from": None,
}


def driver_line(**overrides) -> str:
    receipt = dict(DRIVER_RECEIPT)
    for key, value in overrides.items():
        if value is ...:
            receipt.pop(key, None)
        else:
            receipt[key] = value
    return "  [RASBERY][CASE] " + json.dumps(receipt)


def audit(text: str, declared="strict") -> list[str]:
    return exact_audit.audit_case_fidelity(text, declared)


def audit_reads_the_driver_tag() -> None:
    problems = audit(driver_line())
    if problems:
        fail("the current [RASBERY][CASE] receipt does not audit clean: "
             + "; ".join(problems))

    # THE 181 REGRESSION, as a case.  A binary built before this commit prints
    # `fidelity` and no `physics_fidelity`; it is WP10.3-complete and must pass.
    legacy = audit(driver_line(physics_fidelity=..., schema_version=4))
    if legacy:
        fail("a pre-WP10.4 [RASBERY][CASE] receipt (fidelity, no "
             "physics_fidelity) is still refused: " + "; ".join(legacy)
             + " -- this is exactly the 83-finding host 181 soak result")

    # And the evaluator tag, which was always right, stays right.
    evaluator = audit("[RASBERY][EVALUATOR][CASE] " + json.dumps(
        {"key": "c0", "policy": "strict", "physics_fidelity": "full_exact",
         "statepoint_grid": "full", "acceptance_eligible": True}))
    if evaluator:
        fail("the evaluator case tag stopped auditing clean: " + "; ".join(evaluator))


def version_refusal_survives() -> None:
    """The synonym must not turn the WP10.3 refusal into a pass.

    `fidelity` is a WP10.1 field.  The two fields WP10.3 actually added --
    `statepoint_grid` and `acceptance_eligible` -- have no synonym, so a receipt
    from a binary that genuinely predates WP10.3 is still refused, and refused
    BY THE FIELD THAT WOULD HAVE VOIDED THE CASE.
    """
    for gone in ("statepoint_grid", "acceptance_eligible"):
        problems = audit(driver_line(**{gone: ...}))
        if not any(gone in p and "predates WP10.3" in p for p in problems):
            fail(f"a receipt with no {gone!r} was not refused as pre-WP10.3; the "
                 f"synonym map has widened the version check it must not touch")

    both_gone = audit(driver_line(fidelity=..., physics_fidelity=...))
    if not any("physics_fidelity" in p for p in both_gone):
        fail("a receipt carrying NEITHER spelling of the fidelity passed; the "
             "synonym is supposed to accept an alias, not the absence of one")


def undeclared_refusal_is_unambiguous() -> None:
    """The refusal must say WHICH side failed to name the case.

    The old message said only "no fidelity was declared", which is true both of
    an operator who forgot a case and of an emitter whose receipt carries no
    name the declarer could have used.  On 181 it was the second, 82 times, and
    the finding sat a whole session as "not yet diagnosed".
    """
    blind = audit(driver_line(output=...), {"g0000c0000": "strict"})
    if not blind:
        fail("a receipt with no declarable identifier was accepted; the audit "
             "cannot then tell a declared case from an unmatched one")
        return
    message = " ".join(blind)
    for token in ("case_key", "g0000c0000", "EMITTER"):
        if token not in message:
            fail(f"the undeclared-case refusal does not name {token!r}: it has to "
                 "print what the receipt WAS identified by, what the declaration "
                 "was keyed on, and which of the two is at fault")

    resolved = audit(driver_line(), {"g0000c0000": "strict",
                                     "/w/out/g0000c0000.h5": "strict"})
    if resolved:
        fail("a receipt declared under its output path still did not resolve: "
             + "; ".join(resolved))


def negative_controls() -> None:
    """Each check above must be able to fail."""
    controls: list[str] = []

    # The source scanner: a block with the field cut must be reported.
    block = driver_case_block()
    if block:
        cut = block.replace('\\"physics_fidelity\\"', '\\"NOTHING\\"')
        if "physics_fidelity" in emitted_fields(cut):
            controls.append("emitted_fields() still finds physics_fidelity in a "
                            "block that no longer prints it")
        if "physics_fidelity" not in emitted_fields(block):
            controls.append("emitted_fields() cannot see physics_fidelity in the "
                            "real Driver.h block, so the scan proves nothing")

    # The behavioural half: a coarse case declared strict must still be caught.
    mismatched = audit(driver_line(policy="L3coarse", physics_fidelity="coarse10",
                                   fidelity="coarse10", statepoint_grid="coarse",
                                   acceptance_eligible=False))
    if not any("COARSER" in p for p in mismatched):
        controls.append("a L3coarse case declared strict was not reported, so the "
                        "clean-audit result above means nothing")

    # And the synonym must not accept an arbitrary third name.
    invented = audit(driver_line(fidelity=..., physics_fidelity=...,
                                 physics_fidelity_v2="full_exact"))
    if not any("physics_fidelity" in p for p in invented):
        controls.append("an invented field name satisfied the required-field check")

    for message in controls:
        fail("NEGATIVE CONTROL FAILED: " + message)


def main() -> int:
    emitters_carry_every_audited_field()
    emitters_carry_a_per_case_identifier()
    driver_receipt_bumped_schema()
    soak_declares_every_identifier()
    soak_audits_through_exact_audit()
    audit_reads_the_driver_tag()
    version_refusal_survives()
    undeclared_refusal_is_unambiguous()
    negative_controls()

    if FAILURES:
        for message in FAILURES:
            print("FAIL " + message)
        print(f"soak receipt schema contract: FAIL ({len(FAILURES)} problems)")
        return 1
    print("soak receipt schema contract: PASS (2 emitters x "
          f"{len(exact_audit.CASE_REQUIRED_FIELDS)} audited fields, 1 synonym, "
          "version refusal intact, per-case identifier declarable both ways)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
