#!/usr/bin/env python3
"""Contract: the device XE_FORMS mask agrees with the host call site BY
CONSTRUCTION, not by a human typing a pin (WP7-C, section 9.6).

WHAT WENT WRONG.

Host 181, at 73f8627, mined `0xd3d` for the device mask while the production
host block in `Driver.h::TryAndersonXeStepGpu` spelled its normal equations
under `XE_HOST_FORMS_DEFAULT = 0xac0`.  `RASBERY_GPU_XE_TXN=1` then walked a
different trajectory from TXN=0 -- 1190 versus 1195 Xe steps -- and the only way
to get the two arms back together was to type

    RASBERY_XE_FORMS=0xadd

by hand, after which they were byte-identical (h5diff 0, digest
88dc35e408c86ad4, 1199 Xe steps on both, forms_audit_mismatch 0).

`0xadd` was never a discovery.  It is `(0xd3d & XE_SHIPPED_FORMS) | 0xac0`: the
mined bits for the two DEVICE sites the fixture legitimately reaches (the
fixed-partition dot and the candidate loop, bits 0..4) and the HOST mask for the
four normal-equations sites (bits 5..12), which TXN=1 exists to reproduce.  Both
halves were already inside the binary.  A human was recomputing them.

THE FIX, AND WHAT THIS FILE PINS.  `xeFormMask()` composes the two channels
itself.  The failure mode that would bring 181 back is not a crash, it is a
QUIET one: somebody edits the composition so the mined algebra bits ship again,
or so the host half is read from a second getenv that a mid-run setenv could
split, or so `return` hands back the pre-composition value -- and every existing
Xe contract test still passes, because none of them looks at this seam.  Each
rule below is a function of the source text so a negative control can feed it a
mutated copy and demand that it notices.

WHAT THIS FILE CANNOT PIN.  Whether TXN=0 and TXN=1 actually agree on a given
build is a RUNTIME fact.  Section 9.6.6 of the doc carries the env-free h5diff
runbook; this file only guarantees that the number the runbook exercises is
composed rather than hoped for.

    tools/test_xe_forms_host_consistency_contract.py
"""
from __future__ import annotations

import os
import py_compile
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures: list[str] = []
checks = 0


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8-sig") as handle:
        return handle.read()


def check(ok: bool, what: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(what)


def strip_comments(text: str) -> str:
    """Every rule reads CODE, never prose.

    These files EXPLAIN the composition at length, naming the very masks and the
    very env variable the rules are about; a checker that read the explanation
    as the fact would forbid the tree from explaining itself.  Same carve-out
    tools/test_xe_forms_audit_contract.py makes.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def body_of(text: str, signature_fragment: str) -> str:
    """The brace-balanced body of the first function whose header contains
    `signature_fragment`."""
    i = text.index(signature_fragment)
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j:k + 1]
    raise AssertionError("unbalanced braces after " + signature_fragment)


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


MINER = read("src", "XeFormMiner.cpp")
MASK_H = read("src", "XeFormMask.h")
KERNEL = read("src", "XeKernel.h")
BACKEND = read("src", "CudaXsReconBackend.cu")
PROBE = read("test", "xe_form_probe.cpp")
DOC = read("docs", "WP7C_XE_TXN_20260831_KO.md")
TRACKER = read("docs", "WP_PLAN_REVIEW_AND_TRACKER_20260831_KO.md")

MINER_CODE = strip_comments(MINER)
KERNEL_CODE = strip_comments(KERNEL)
BACKEND_CODE = strip_comments(BACKEND)
PROBE_CODE = strip_comments(PROBE)


def resolver_body(miner_code: str) -> str:
    return body_of(miner_code, "unsigned long long xeFormMask()")


# ---------------------------------------------------------------------------
# 1. THE COMPOSITION ITSELF.
#
# One expression, spelled from the two channel constants rather than from
# literals, so a site added at bit 13 moves both halves at once instead of
# falling off the end of a hand-written 0x1fe0.
# ---------------------------------------------------------------------------
def rule_composes_the_two_channels(miner_code: str) -> bool:
    body = resolver_body(miner_code)
    return ("(resolved&XE_SHIPPED_FORMS)|(host_forms&XE_ALGEBRA_FORMS)"
            in squash(body))


def rule_algebra_half_comes_from_the_host_mask(miner_code: str) -> bool:
    """The bits 5..12 half must be `xeHostFormMask()` -- the number Driver.h's
    own block spells its four expressions with -- and nothing else."""
    body = resolver_body(miner_code)
    if "xeHostFormMask()" not in squash(body):
        return False
    return re.search(r"host_forms\s*=\s*xeHostFormMask\(\s*\)", body) is not None


def rule_shipped_half_is_still_mined(miner_code: str) -> bool:
    """Bits 0..4 stay the MEASURED value: they are device sites with no host
    counterpart, and the fixture reaches them honestly.  The composition must
    not quietly turn the dot and the candidate into constants too."""
    body = resolver_body(miner_code)
    call = re.search(r"resolveCalibratedFormMask\((.*?)\);", body, re.S)
    if call is None:
        return False
    args = [a.strip() for a in call.group(1).split(",")]
    return len(args) == 5 and args[2] == "mined" and args[3] == "sound"


def rule_mined_algebra_bits_never_ship(miner_code: str) -> bool:
    """THE WHOLE POINT.  `mined` may reach the receipt and the shipped channel;
    it may not reach the algebra channel of the returned value.  181's 0xd3d is
    a measurement of xeref::refAlgebra in another translation unit, and a
    measurement of the wrong thing does not become the right thing by being a
    measurement."""
    body = resolver_body(miner_code)
    return re.search(r"mined\s*&\s*XE_ALGEBRA_FORMS", body) is None


def rule_host_channel_is_narrowed(miner_code: str) -> bool:
    """`xeHostFormMask()` already trims to XE_ALGEBRA_FORMS, but the composition
    narrows again: two guards, because a future host mask that carried a bit 0..4
    would otherwise reach the production dot from a knob whose name says host."""
    body = resolver_body(miner_code)
    return "host_forms&XE_ALGEBRA_FORMS" in squash(body)


check(rule_composes_the_two_channels(MINER_CODE),
      "xeFormMask() composes (resolved & XE_SHIPPED_FORMS) | (host_forms & "
      "XE_ALGEBRA_FORMS): the pin 0xadd is arithmetic the binary does, not a "
      "number a human retypes")
check(rule_algebra_half_comes_from_the_host_mask(MINER_CODE),
      "the algebra half comes from xeHostFormMask() -- the same cached number "
      "Driver.h's production block spells its four expressions with")
check(rule_shipped_half_is_still_mined(MINER_CODE),
      "the shipped half is still the MINED value under the SHIPPED soundness: "
      "the dot and the candidate are device sites the fixture reaches honestly")
check(rule_mined_algebra_bits_never_ship(MINER_CODE),
      "the MINED algebra bits never reach the resolved mask: they score a "
      "quotation (xeref::refAlgebra), not the call site TXN=1 must reproduce")
check(rule_host_channel_is_narrowed(MINER_CODE),
      "the host half is narrowed to XE_ALGEBRA_FORMS at the composition too, so "
      "a host mask that grew a bit 0..4 could not reach the production dot")


# ---------------------------------------------------------------------------
# 2. AN EXPLICIT OVERRIDE STILL WINS VERBATIM.
#
# A human who types RASBERY_XE_FORMS means the number they typed, algebra bits
# included: the 81-point sweep of section 9.6.7 item 2 is exactly the procedure
# of disagreeing with this composition, and it has to stay possible.
# ---------------------------------------------------------------------------
def rule_env_wins_verbatim(miner_code: str) -> bool:
    body = squash(resolver_body(miner_code))
    return "env_pinned?resolved:composed" in body


def rule_env_is_detected_by_parsing(miner_code: str) -> bool:
    """A malformed override is NOT an override -- `parseFormMask` is the same
    predicate resolveCalibratedFormMask used to decide whether `resolved` came
    from the environment, so the two cannot disagree about it."""
    body = resolver_body(miner_code)
    return ('std::getenv("RASBERY_XE_FORMS")' in body
            and "gpu::parseFormMask(env_raw, env_parsed)" in squash(body).replace(
                "gpu::parseFormMask(env_raw,env_parsed)",
                "gpu::parseFormMask(env_raw, env_parsed)"))


def rule_host_knob_is_not_re_read_here(miner_code: str) -> bool:
    """RASBERY_XE_HOST_FORMS is read once, inside xeHostFormMask's own cached
    static.  A second getenv here could be answered differently by a mid-run
    setenv, and then the device mask and Driver.h's block would spell one solve
    two ways."""
    return "RASBERY_XE_HOST_FORMS" not in resolver_body(miner_code)


check(rule_env_wins_verbatim(MINER_CODE),
      "an explicit RASBERY_XE_FORMS wins VERBATIM -- composition is skipped, not "
      "layered on top; the 81-point sweep is the procedure of disagreeing with "
      "the composition and must stay possible")
check(rule_env_is_detected_by_parsing(MINER_CODE),
      "whether the override applied is decided by gpu::parseFormMask on "
      "RASBERY_XE_FORMS -- the same predicate resolveCalibratedFormMask used, so "
      "a malformed value is not an override on one side and an override on the "
      "other")
check(rule_host_knob_is_not_re_read_here(MINER_CODE),
      "RASBERY_XE_HOST_FORMS is NOT re-read inside xeFormMask(): one cached "
      "resolution, or a mid-run setenv gives one solve two contraction contracts")


# ---------------------------------------------------------------------------
# 3. THE RECEIPT SHOWS THE ARITHMETIC, NOT ONLY THE ANSWER.
#
# `0xd3d` and `0xac0` and `0xadd` on one line, so a reviewer checks the
# composition without a second run and without opening a second file.
# ---------------------------------------------------------------------------
RECEIPT_FIELDS = ("source", "mined", "host", "composed", "resolved", "shipped",
                  "algebra")


def rule_receipt_names_every_component(miner: str) -> bool:
    body = body_of(miner, "unsigned long long xeFormMask()")
    return all('\\"%s\\":' % field in body for field in RECEIPT_FIELDS)


def rule_receipt_prints_the_components_themselves(miner_code: str) -> bool:
    """Naming the fields is not enough: they must be fed the VARIABLES.  A
    receipt that printed `value` under the name `mined` would be a lie with a
    green light next to it."""
    body = resolver_body(miner_code)
    line = body[body.index("[RASBERY][FORMS]"):]
    squashed = squash(line)
    return ("<<mined<<" in squashed and "<<host_forms<<" in squashed
            and "<<composed<<" in squashed)


def rule_source_has_both_labels(miner_code: str) -> bool:
    body = resolver_body(miner_code)
    return '"build_default_composed"' in body and '"env"' in body


def rule_returns_the_composed_value(miner_code: str) -> bool:
    """The lambda must hand back the composed/overridden value.  `return
    resolved;` would leave the receipt telling the truth about a number the run
    does not use -- the worst of the available failures."""
    body = resolver_body(miner_code)
    return (re.search(r"return\s+value\s*;", body) is not None
            and re.search(r"return\s+resolved\s*;", body) is None)


check(rule_receipt_names_every_component(MINER),
      "the [RASBERY][FORMS] XE_FORMS line names source, mined, host, composed, "
      "resolved, shipped and algebra: 0xd3d + 0xac0 -> 0xadd is checkable on one "
      "line")
check(rule_receipt_prints_the_components_themselves(MINER_CODE),
      "the receipt's mined/host/composed fields are fed those variables, not a "
      "restatement of the answer")
check(rule_source_has_both_labels(MINER_CODE),
      "source is build_default_composed or env, so a log says which of the two "
      "regimes the run was in")
check(rule_returns_the_composed_value(MINER_CODE),
      "xeFormMask() returns the composed/overridden value -- not `resolved`, "
      "which would make the receipt describe a mask the run did not use")


# ---------------------------------------------------------------------------
# 4. FEATURE-OFF BYTE IDENTITY.
#
# The composition moves bits 5..12 and nothing else.  For a RASBERY_GPU_XE_TXN
# unset/0 run to be the run it was, those bits must be unreachable from the
# production split arm -- which is exactly the property the split constant and
# the shipped accessor already carry, restated here because THIS commit is the
# one that makes them load-bearing.
# ---------------------------------------------------------------------------
def rule_channels_are_disjoint(kernel_code: str) -> bool:
    return re.search(
        r"XE_ALGEBRA_FORMS\s*=\s*\(\(\(1ull\s*<<\s*XE_BIT_COUNT\)\s*-\s*1ull\)"
        r"\s*&\s*~XE_SHIPPED_FORMS\)", kernel_code) is not None


def rule_shipped_accessor_drops_the_algebra_channel(miner_code: str) -> bool:
    body = body_of(miner_code, "xeShippedFormMask()")
    return squash(body) == "{returnxeFormMask()&XE_SHIPPED_FORMS;}"


def rule_live_launches_take_the_shipped_mask(backend_code: str) -> bool:
    """The split arm's two entry points; the composition cannot reach a run that
    only ever calls these."""
    for entry in ("XsReconBackend::xeDots(", "XsReconBackend::xeCandidate("):
        body = body_of(backend_code, entry)
        if "xeShippedFormMask()" not in body:
            return False
        if re.search(r"(?<!Shipped)xe::xeFormMask\(\s*\)", body) is not None:
            return False
    return True


def rule_full_mask_is_read_only_by_the_transaction(backend_code: str) -> bool:
    """`xe::xeFormMask()` -- the union, algebra bits included -- may appear in
    exactly one function body in the backend, and it must be the transaction."""
    hits = [m.start() for m in re.finditer(r"xe::xeFormMask\(\s*\)", backend_code)]
    if len(hits) != 1:
        return False
    txn = body_of(backend_code, "XsReconBackend::xeTransaction(")
    start = backend_code.index(txn)
    return start <= hits[0] < start + len(txn)


check(rule_channels_are_disjoint(KERNEL_CODE),
      "XE_ALGEBRA_FORMS is the complement of XE_SHIPPED_FORMS inside the mined "
      "width, so the composition provably cannot touch bits 0..4")
check(rule_shipped_accessor_drops_the_algebra_channel(MINER_CODE),
      "xeShippedFormMask() is exactly `xeFormMask() & XE_SHIPPED_FORMS`: the "
      "composed algebra bits are unrepresentable in the production launches' "
      "argument")
check(rule_live_launches_take_the_shipped_mask(BACKEND_CODE),
      "XsReconBackend::xeDots and ::xeCandidate launch with xeShippedFormMask() "
      "only -- a TXN=0 run reads no bit this commit moved")
check(rule_full_mask_is_read_only_by_the_transaction(BACKEND_CODE),
      "the union is read in exactly one backend function and it is "
      "XsReconBackend::xeTransaction, which Driver.h reaches only behind "
      "rasberyGpuXeTxnEnabled()")


# ---------------------------------------------------------------------------
# 5. THE GATE THAT CANNOT BE A SOURCE PROPERTY IS DECLARED, NOT ASSUMED.
#
# The header tells a reader of the DECLARATION that the mask is composed; the
# probe no longer asserts the pre-composition identity (a stale assertion here
# would fail the ctest gate on the first host that mines a non-zero algebra
# channel, for a reason that has nothing to do with the code under test); and
# the doc carries the env-free h5diff runbook and the pending gate.
# ---------------------------------------------------------------------------
def rule_header_documents_the_composition(mask_h: str) -> bool:
    return ("XE_SHIPPED_FORMS" in mask_h and "XE_ALGEBRA_FORMS" in mask_h
            and "xeHostFormMask()" in mask_h and "0xadd" in mask_h
            and "VERBATIM" in mask_h)


def rule_probe_scores_the_composition(probe_code: str) -> bool:
    if re.search(r"resolved\s*==\s*mined\b", probe_code) is not None:
        return False
    return ("xe::xeHostFormMask()" in probe_code
            and re.search(r"resolved\s*==\s*composed", probe_code) is not None)


def rule_doc_records_the_fix(doc: str) -> bool:
    return ("build_default_composed" in doc
            and "9.6.6" in doc
            and "h5diff" in doc
            and "0xadd" in doc)


def rule_tracker_carries_the_pending_gate(tracker: str) -> bool:
    return ("device mask composed from host XE_HOST_FORMS bits 5..12; "
            "B0 gate pending on 181" in tracker)


check(rule_header_documents_the_composition(MASK_H),
      "src/XeFormMask.h documents the composition at the DECLARATION: a caller "
      "reading the header learns which channel comes from where and that the "
      "env override is verbatim")
check(rule_probe_scores_the_composition(PROBE_CODE),
      "test/xe_form_probe.cpp scores the resolver against the COMPOSED value; "
      "the pre-composition `resolved == mined` assertion is gone")
check(rule_doc_records_the_fix(DOC),
      "docs/WP7C_XE_TXN_20260831_KO.md section 9.6 records the composition, the "
      "receipt source label and the env-free h5diff runbook")
check(rule_tracker_carries_the_pending_gate(TRACKER),
      "the WP7-C tracker row says the device mask is composed from the host "
      "XE_HOST_FORMS bits 5..12 and that the B0 gate is pending on 181")


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Each rule is handed a copy of the source with its property
# removed and must say so.  A rule that passes on the mutant was passing on the
# fixture, not on the code.
# ---------------------------------------------------------------------------
def _mined_algebra_ships(miner_code: str) -> str:
    """The regression itself: 181's 0xd3d back in the algebra channel."""
    return miner_code.replace(
        "(resolved & XE_SHIPPED_FORMS) | (host_forms & XE_ALGEBRA_FORMS)",
        "(resolved & XE_SHIPPED_FORMS) | (mined & XE_ALGEBRA_FORMS)", 1)


NEGATIVE = [
    (rule_composes_the_two_channels,
     MINER_CODE.replace(
         "(resolved & XE_SHIPPED_FORMS) | (host_forms & XE_ALGEBRA_FORMS)",
         "resolved", 1),
     "a resolver that ships the mined union unchanged"),
    (rule_algebra_half_comes_from_the_host_mask,
     MINER_CODE.replace("host_forms = xeHostFormMask()",
                        "host_forms = XE_HOST_FORMS_DEFAULT", 1),
     "an algebra half taken from the build constant, so RASBERY_XE_HOST_FORMS "
     "would move the host block and not the device mask"),
    (rule_shipped_half_is_still_mined,
     MINER_CODE.replace("XE_FORMS_DEFAULT, mined, sound,",
                        "XE_FORMS_DEFAULT, XE_FORMS_DEFAULT, sound,", 1),
     "a shipped half that stopped being mined"),
    (rule_mined_algebra_bits_never_ship, _mined_algebra_ships(MINER_CODE),
     "the mined algebra bits shipping again -- the 181 divergence, restored"),
    (rule_host_channel_is_narrowed,
     MINER_CODE.replace("(host_forms & XE_ALGEBRA_FORMS)", "(host_forms)", 1),
     "a host half that is not narrowed to the algebra channel"),
    (rule_env_wins_verbatim,
     MINER_CODE.replace("env_pinned ? resolved : composed", "composed", 1),
     "a composition that overrides the human's override"),
    (rule_env_is_detected_by_parsing,
     MINER_CODE.replace('std::getenv("RASBERY_XE_FORMS")', "nullptr", 1),
     "an override detection that never reads the variable"),
    (rule_host_knob_is_not_re_read_here,
     MINER_CODE.replace("const unsigned long long host_forms = xeHostFormMask();",
                        "const unsigned long long host_forms = "
                        'std::getenv("RASBERY_XE_HOST_FORMS") ? 0ull : 0xac0ull;', 1),
     "a second read of RASBERY_XE_HOST_FORMS inside the device resolver"),
    (rule_receipt_names_every_component,
     MINER.replace('\\"host\\":\\"0x', '\\"unrelated\\":\\"0x', 1),
     "a receipt that hides one half of the composition"),
    (rule_receipt_prints_the_components_themselves,
     MINER_CODE.replace('<< "\\",\\"mined\\":\\"0x" << mined',
                        '<< "\\",\\"mined\\":\\"0x" << value', 1),
     "a receipt that prints the answer under the name of an input"),
    (rule_source_has_both_labels,
     MINER_CODE.replace('"build_default_composed"', '"build_default"', 1),
     "a source label that does not say the value was composed"),
    (rule_returns_the_composed_value,
     MINER_CODE.replace("return value;", "return resolved;", 1),
     "a resolver whose receipt and whose return value disagree"),
    (rule_channels_are_disjoint,
     KERNEL_CODE.replace("& ~XE_SHIPPED_FORMS)", ")", 1),
     "an algebra channel that overlaps the shipped one"),
    (rule_shipped_accessor_drops_the_algebra_channel,
     MINER_CODE.replace("return xeFormMask() & XE_SHIPPED_FORMS;",
                        "return xeFormMask();", 1),
     "a shipped accessor that hands the composed union to the live launches"),
    (rule_live_launches_take_the_shipped_mask,
     BACKEND_CODE.replace("xe::xeShippedFormMask());", "xe::xeFormMask());", 1),
     "a live launch handed the union"),
    (rule_full_mask_is_read_only_by_the_transaction,
     BACKEND_CODE.replace("if (ncol < 1 || ncol > xek::XE_DEPTH) return false;",
                          "if (ncol < 1 || ncol > xek::XE_DEPTH) return false;\n"
                          "    (void)xe::xeFormMask();", 1),
     "a second reader of the union outside the transaction"),
    (rule_header_documents_the_composition,
     MASK_H.replace("0xadd", "the pin"),
     "a header that documents the composition without the worked value"),
    (rule_probe_scores_the_composition,
     PROBE_CODE.replace("check(resolved == composed,", "check(resolved == mined,", 1),
     "a probe that still asserts the pre-composition identity"),
    (rule_doc_records_the_fix, DOC.replace("build_default_composed", "TBD"),
     "a doc that does not name the receipt's new source label"),
    (rule_tracker_carries_the_pending_gate,
     TRACKER.replace("B0 gate pending on 181", "B0 gate cleared", 1),
     "a tracker row that claims a gate nobody ran"),
]

for rule, mutant, what in NEGATIVE:
    check(not rule(mutant), f"NEGATIVE CONTROL: {rule.__name__} accepted {what}")

py_compile.compile(os.path.abspath(__file__), doraise=True)

if failures:
    for message in failures:
        print(f"FAIL  {message}")
    print(f"FAIL  tools/test_xe_forms_host_consistency_contract.py  "
          f"({len(failures)} of {checks} checks)")
    sys.exit(1)

print(f"PASS  tools/test_xe_forms_host_consistency_contract.py  "
      f"({checks} checks, {len(NEGATIVE)} negative controls)")
print("  whether TXN=0 and TXN=1 agree on a build is a RUNTIME fact, not a "
      "source one:")
print("  run the env-free h5diff pair in "
      "docs/WP7C_XE_TXN_20260831_KO.md section 9.6.6 and")
print("  check both logs say "
      '[RASBERY][FORMS] ... "source":"build_default_composed".')
