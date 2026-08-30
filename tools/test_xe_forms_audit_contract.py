#!/usr/bin/env python3
"""Contract: the XE_FORMS audit is an instrument, not a comment (WP7-C, section 9).

WHAT WENT WRONG, AND WHY A NEW CONTRACT.

WP7-C shipped claiming that `RASBERY_GPU_XE_TXN=1` is bit-identical to TXN=0.
Host 181 (2026-08-30) measured otherwise -- different digest, 854 h5diff lines,
1195 versus 1190 Xe steps -- with a contraction mask that MINED CLEAN: the same
`[RASBERY][FORMS] 0xd3d` line in both arms, no `[WARN][FORMS]`, and every one of
the four WP7-C sites moved off the all-zero seed, so the fixture discriminates
all of them.

The mask therefore reproduces what it is scored against (`xeref::refAlgebra`, in
src/XeAlgebraReference.cpp) and not what it has to reproduce
(`Driver.h::TryAndersonXeStepGpu`, inlined into SolveLoop).  No fixture can
close that: a call site is only reachable inside its own inline.  So the gap is
MEASURED instead, at the call site, by src/XeFormAudit.{h,cpp} under
`RASBERY_XE_FORMS_AUDIT=1`.

THE INSTRUMENT HAS ONE WAY TO FAIL SILENTLY and this file exists for it.  If the
audit were an inline function in a header, gcc could common-subexpression the
PRODUCTION `a * c - b * b` into the audit's recomputation -- and then the audit
compares a value with itself, reports zero mismatches on every host forever, and
the false claim it exists to detect survives with a green light next to it.
Every structural property that keeps that from happening is checked here, each
with a negative control that was made to fail before it was made to pass.

    tools/test_xe_forms_audit_contract.py
"""
from __future__ import annotations

import os
import py_compile
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), "r", encoding="utf-8-sig") as handle:
        return handle.read()


AUDIT_H = read("src", "XeFormAudit.h")
AUDIT_CPP = read("src", "XeFormAudit.cpp")
DRIVER = read("src", "Driver.h")
RECEIPT = read("src", "XeGpuReceipt.h")
DOC = read("docs", "WP7C_XE_TXN_20260831_KO.md")

FAILURES: list[str] = []
CHECKS = 0


def check(ok: bool, message: str) -> bool:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(message)
    return bool(ok)


def code_of(text: str) -> str:
    """The text with `//` comments removed.

    Every file here EXPLAINS the rule it obeys, in prose, naming the very
    include and the very call it must not make.  A checker that read the
    explanation as the offence would forbid the tree from explaining itself --
    the same carve-out tools/test_case_key_contract.py makes around
    caseKeyProvenance.
    """
    return re.sub(r"//[^\n]*", "", text)


# ---------------------------------------------------------------------------
# The rules, as functions of the source text, so a negative control can feed
# each one a MUTATED copy and demand that it notices.
# ---------------------------------------------------------------------------
def rule_header_is_declaration_only(audit_h: str) -> bool:
    """The header must not pull in the shipped bodies.

    THIS IS THE WHOLE INSTRUMENT.  The shipped body in this header would put
    `xeAndersonFit` in every translation unit that includes it -- Driver.h's
    among them -- and the recomputation would then be inlinable next to the
    production block it is supposed to be independent of.
    """
    if '#include "XeKernel.h"' in code_of(audit_h):
        return False
    # A definition would have the same effect as an include.
    return re.search(r"void\s+auditAndersonFit\([^)]*\)\s*\{", audit_h, re.S) is None


def rule_definition_is_its_own_tu(audit_cpp: str) -> bool:
    return ('#include "XeKernel.h"' in audit_cpp
            and re.search(r"void\s+auditAndersonFit\([^)]*\)\s*\{", audit_cpp, re.S)
            is not None)


def rule_compares_bit_patterns(audit_cpp: str) -> bool:
    """`==` on two doubles calls two contractions that differ in the last bit
    equal, which is exactly the thing being looked for."""
    if "std::memcpy" not in audit_cpp:
        return False
    for name in ("gamma0", "gamma1", "proj"):
        if f"bitsOf({name})" not in audit_cpp:
            return False
    return re.search(r"shipped_g0\s*==\s*gamma0", audit_cpp) is None


def rule_audits_solved_too(audit_cpp: str) -> bool:
    """The conditioning test divides by a determinant one of the mined sites
    decides, so a mask can move `solved` without moving any coefficient."""
    return "shipped_solved != solved" in audit_cpp


def rule_off_by_default(audit_cpp: str) -> bool:
    fn = audit_cpp[audit_cpp.find("bool xeFormAuditEnabled()"):]
    fn = fn[:fn.find("\n}")]
    return ('std::getenv("RASBERY_XE_FORMS_AUDIT")' in fn
            and "v != nullptr" in fn
            and 'std::string(v) != "0"' in fn)


def rule_uses_the_resolved_mask(audit_cpp: str) -> bool:
    """The mask the RUN uses, not the shipped default: a mask read from
    XE_FORMS_DEFAULT would audit a value no kernel was launched with."""
    code = code_of(audit_cpp)
    return "xeFormMask()" in code and "XE_FORMS_DEFAULT" not in code


def rule_reports_once(audit_cpp: str) -> bool:
    return ("compare_exchange_strong" in audit_cpp
            and "[RASBERY][WARN][FORMS]" in audit_cpp)


def _call_site(driver: str) -> str:
    """The production algebra block of TryAndersonXeStepGpu.

    Anchored on the DEVICE arm, because the host arm's TryAndersonXeStep carries
    the same four expressions and auditing that one would measure a mask nothing
    runs under.
    """
    start = driver.find("static bool TryAndersonXeStepGpu(SolverContext& ctx,")
    if start < 0:
        return ""
    end = driver.find("\n    /// One safeguarded Anderson step", start)
    return driver[start:end if end > start else len(driver)]


def rule_call_site_is_the_device_arm(driver: str) -> bool:
    block = _call_site(driver)
    return bool(block) and "xe::auditAndersonFit(" in block


def rule_call_site_is_before_the_rejection(driver: str) -> bool:
    """After both solve branches and before the `condition` rejection returns.

    A call placed after the return would never see a refused fit, and a refused
    fit is one of the two outcomes the mask can move.
    """
    block = _call_site(driver)
    call = block.find("xe::auditAndersonFit(")
    reject = block.find('RejectXeAnderson(ctx, "condition"')
    secant = block.find("gamma[j] = p / a;")
    return 0 < secant < call < reject


def rule_call_passes_production_values(driver: str) -> bool:
    block = _call_site(driver)
    call = block[block.find("xe::auditAndersonFit("):]
    call = call[:call.find(";") + 1]
    for token in ("dots", "aa.ncol", "XE_ANDERSON_MIN_GRAM", "solved",
                  "gamma[0]", "gamma[1]", "proj"):
        if token not in call:
            return False
    return True


def rule_call_is_gated_and_cached(driver: str) -> bool:
    return ("static const bool audit = xe::xeFormAuditEnabled();"
            in _call_site(driver))


def rule_driver_includes_the_header(driver: str) -> bool:
    return '#include "XeFormAudit.h"' in driver


def rule_knob_is_not_in_arm_env(driver: str) -> bool:
    """RASBERY_XE_FORMS_AUDIT is a MEASUREMENT of the arm, not a knob of it.

    In kArmEnv it would land in every case key's payload and move every key in
    the campaign for a flag that changes no number.  Same rule, same reason, as
    RASBERY_STATEPOINT_TELEMETRY's absence.
    """
    block = driver[driver.find("inline constexpr const char* kArmEnv[] = {"):]
    block = block[:block.find("};")]
    return "RASBERY_XE_FORMS_AUDIT" not in block


def rule_receipt_carries_the_counters(receipt: str) -> bool:
    for name in ("forms_audits", "forms_audit_mismatch", "forms_audit_mask"):
        if f"std::atomic<unsigned long long> {name}{{0}}" not in receipt:
            return False
        if f'\\"{name}\\":' not in receipt:
            return False
    return True


def rule_receipt_does_not_mine_when_idle(receipt: str) -> bool:
    """Printing xeFormMask() unconditionally would MINE the mask in a run whose
    device Xe arm never ran, and put a [RASBERY][FORMS] line in the log of a
    feature that was off.  The receipt prints what the AUDIT recorded instead."""
    return "xeFormMask()" not in code_of(receipt) and "audits > 0" in receipt


def rule_receipt_carries_the_grade(receipt: str) -> bool:
    note = receipt[receipt.find("kXeTxnPolicyNote"):]
    note = note[:note.find(";")]
    return ("N1" in note and "B0" not in note
            and "RASBERY_XE_FORMS_AUDIT" in note
            and '\\"policy_note\\":' in receipt)


def rule_doc_is_downgraded(doc: str) -> bool:
    row = next((line for line in doc.splitlines()
                if line.startswith("| 게이트 등급 |")), "")
    return ("N1" in row and "**B0" not in row
            and "## 9." in doc
            and "RASBERY_XE_FORMS_AUDIT" in doc)


RULES = [
    (rule_header_is_declaration_only, AUDIT_H,
     "src/XeFormAudit.h must be a DECLARATION ONLY -- no shipped body, no "
     "definition.  An inline audit can be common-subexpressioned with the "
     "production block it audits, and then it compares a value with itself and "
     "passes forever."),
    (rule_definition_is_its_own_tu, AUDIT_CPP,
     "src/XeFormAudit.cpp must define auditAndersonFit and be the only place "
     "the shipped body reaches it"),
    (rule_compares_bit_patterns, AUDIT_CPP,
     "the audit must compare BIT PATTERNS; `==` on doubles calls two "
     "contractions that differ in the last bit equal"),
    (rule_audits_solved_too, AUDIT_CPP,
     "the audit must compare `solved`: the conditioning test divides by a "
     "determinant one of the mined sites decides"),
    (rule_off_by_default, AUDIT_CPP,
     "RASBERY_XE_FORMS_AUDIT must be off unless set to something other than 0"),
    (rule_uses_the_resolved_mask, AUDIT_CPP,
     "the audit must run under xeFormMask() -- the mask the RUN resolved -- and "
     "never under XE_FORMS_DEFAULT"),
    (rule_reports_once, AUDIT_CPP,
     "the audit must print its first mismatch once and then only count"),
    (rule_call_site_is_the_device_arm, DRIVER,
     "the audit call must be in TryAndersonXeStepGpu -- the arm whose algebra "
     "RASBERY_GPU_XE_TXN moves onto the device"),
    (rule_call_site_is_before_the_rejection, DRIVER,
     "the audit call must sit after both solve branches and before the "
     "`condition` rejection returns, or a refused fit is never audited"),
    (rule_call_passes_production_values, DRIVER,
     "the audit must be handed this block's own dots, ncol, min_gram, solved, "
     "gammas and proj"),
    (rule_call_is_gated_and_cached, DRIVER,
     "the audit must be behind a cached `static const bool`, so a run that did "
     "not ask for it is the run it was"),
    (rule_driver_includes_the_header, DRIVER,
     "Driver.h must include XeFormAudit.h"),
    (rule_knob_is_not_in_arm_env, DRIVER,
     "RASBERY_XE_FORMS_AUDIT must NOT be in kArmEnv: it changes no number and "
     "would move every case key in the campaign"),
    (rule_receipt_carries_the_counters, RECEIPT,
     "XeGpuReceipt.h must declare AND print forms_audits, "
     "forms_audit_mismatch and forms_audit_mask"),
    (rule_receipt_does_not_mine_when_idle, RECEIPT,
     "the receipt must not call the mask resolver: that mines, and prints a "
     "[RASBERY][FORMS] line, in a run whose device Xe arm never ran"),
    (rule_receipt_carries_the_grade, RECEIPT,
     "kXeTxnPolicyNote must state the N1 grade and name the audit knob, and the "
     "receipt must print it as policy_note"),
    (rule_doc_is_downgraded, DOC,
     "docs/WP7C_XE_TXN_20260831_KO.md must carry the N1 grade and section 9"),
]

for rule, text, message in RULES:
    check(rule(text), message)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Every rule above is handed a copy of the source with the
# property removed, and must say so.  A rule that passes on the mutant is a rule
# that was passing on the fixture rather than on the code.
# ---------------------------------------------------------------------------
def _moved_after_the_rejection(driver: str) -> str:
    """The audit call lifted out of its place and dropped after the refusal.

    Built by DELETING the real call rather than commenting it out: a `//` in
    front of it leaves the text where a substring search would still find it,
    and a control a substring search defeats is not a control.
    """
    call = re.search(r"\n *xe::auditAndersonFit\(dots[^;]*;", driver, re.S)
    assert call is not None, "the audit call moved; this control needs updating"
    moved = driver[:call.start()] + driver[call.end():]
    return moved.replace(
        'RejectXeAnderson(ctx, "condition", aa.ncol, picard, gg);',
        'RejectXeAnderson(ctx, "condition", aa.ncol, picard, gg);\n'
        '            xe::auditAndersonFit(dots, aa.ncol, XE_ANDERSON_MIN_GRAM,'
        ' solved, gamma[0], gamma[1], proj);', 1)


NEGATIVE = [
    (rule_header_is_declaration_only,
     AUDIT_H.replace("namespace rasbery::xe {",
                     '#include "XeKernel.h"\nnamespace rasbery::xe {', 1),
     "a shipped-body include in the header"),
    (rule_definition_is_its_own_tu, AUDIT_CPP.replace('#include "XeKernel.h"', "", 1),
     "the definition TU without the shipped body"),
    (rule_compares_bit_patterns, AUDIT_CPP.replace("bitsOf(gamma0)", "gamma0", 1),
     "a value comparison instead of a bit comparison"),
    (rule_audits_solved_too,
     AUDIT_CPP.replace("shipped_solved != solved", "false", 1),
     "an audit that ignores `solved`"),
    (rule_off_by_default,
     AUDIT_CPP.replace('std::getenv("RASBERY_XE_FORMS_AUDIT")', "nullptr", 1),
     "an audit whose knob is not read"),
    (rule_uses_the_resolved_mask,
     AUDIT_CPP.replace("xeFormMask()", "XE_FORMS_DEFAULT", 1),
     "an audit against the shipped default mask"),
    (rule_reports_once, AUDIT_CPP.replace("compare_exchange_strong", "load", 1),
     "an audit that reports on every step"),
    (rule_call_site_is_the_device_arm,
     DRIVER.replace("xe::auditAndersonFit(", "xe::somethingElse(", 1),
     "no audit call in the device arm"),
    (rule_call_site_is_before_the_rejection, _moved_after_the_rejection(DRIVER),
     "an audit call placed at the rejection instead of before it"),
    (rule_call_passes_production_values,
     DRIVER.replace("gamma[0], gamma[1], proj);", "0.0, 0.0, 0.0);", 1),
     "an audit handed constants instead of the block's own values"),
    (rule_call_is_gated_and_cached,
     DRIVER.replace("static const bool audit = xe::xeFormAuditEnabled();",
                    "const bool audit = true;", 1),
     "an audit that is always on and re-reads its knob"),
    (rule_driver_includes_the_header,
     DRIVER.replace('#include "XeFormAudit.h"', "", 1),
     "Driver.h without the audit header"),
    (rule_knob_is_not_in_arm_env,
     DRIVER.replace('    "RASBERY_XE_MODE",',
                    '    "RASBERY_XE_FORMS_AUDIT",\n    "RASBERY_XE_MODE",', 1),
     "the audit knob folded into kArmEnv"),
    (rule_receipt_carries_the_counters,
     RECEIPT.replace('\\"forms_audit_mismatch\\":', '\\"unrelated\\":', 1),
     "a counter declared but not printed"),
    (rule_receipt_does_not_mine_when_idle,
     RECEIPT.replace("t.forms_audit_mask.load(std::memory_order_relaxed)",
                     "xeFormMask()", 1),
     "a receipt that mines the mask to print it"),
    (rule_receipt_carries_the_grade,
     RECEIPT.replace("is N1 against TXN=0", "is B0 against TXN=0", 1),
     "a policy note that still claims B0"),
    (rule_doc_is_downgraded,
     DOC.replace("| 게이트 등급 | **N1", "| 게이트 등급 | **B0", 1),
     "a doc that still grades WP7-C B0"),
]

for rule, mutant, what in NEGATIVE:
    check(not rule(mutant),
          f"NEGATIVE CONTROL: {rule.__name__} accepted {what}")

py_compile.compile(os.path.abspath(__file__), doraise=True)

if FAILURES:
    for message in FAILURES:
        print(f"FAIL  {message}")
    print(f"FAIL  tools/test_xe_forms_audit_contract.py  "
          f"({len(FAILURES)} of {CHECKS} checks)")
    sys.exit(1)

print(f"PASS  tools/test_xe_forms_audit_contract.py  "
      f"({CHECKS} checks, {len(NEGATIVE)} negative controls)")
print("  whether THIS BUILD's mined mask reproduces its own production algebra "
      "is a runtime")
print("  fact, not a source one: run RASBERY_XE_FORMS_AUDIT=1 and read "
      "forms_audit_mismatch")
print("  out of [RASBERY][XE_GPU] -- see "
      "docs/WP7C_XE_TXN_20260831_KO.md section 9.5.")
