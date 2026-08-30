#!/usr/bin/env python3
"""Source contract: THE ALGEBRA BITS NEVER REACH A PRODUCTION KERNEL LAUNCH.

WHY THIS FILE EXISTS -- the evidence it is written against.

238, same host, same arm-X environment, RASBERY_GPU_XE=1 and RASBERY_GPU_XE_TXN
unset:

    7cfe3a4   [FORMS] XE_FORMS = {"value":"0xd",  "source":"mined_matches_default"}
    d7b81af   [FORMS] XE_FORMS = {"value":"0xd2d","source":"mined"}
    8919331   [FORMS] XE_FORMS = {"value":"0xd2d","source":"mined"}

0xd2d is 0xd | 1<<5 | 1<<8 | 1<<11.  The SHIPPED bits (0..4) did not move; what
appeared is the WP7-C algebra channel (bits 5..12), which only
RASBERY_GPU_XE_TXN evaluates.  8919331 already stopped the algebra channel's
SOUNDNESS from demoting the shipped channel's VALUE.  It did not stop the
algebra channel's BITS from being handed to the shipped channel's kernels: the
split arm's kXeDotStage1 and kXeCandidate were launched with the whole 0xd2d.

They survived it -- xeDotChunk extracts `(forms >> 0) & 3` and `(forms >> 2) &
1`, xeCandidateOrdinal extracts `(forms >> 3or4) & 1`, so every bit above 4 is
discarded inside the body.  That is why the arm-X trajectory movement observed
across 7cfe3a4..d7b81af is NOT attributable to this mask, and this file is the
source-level proof of that negative.

But "every present and future reader of `forms` remembers to extract its own
field" is a hope, not a contract.  One `forms != XE_FORMS_DEFAULT`, one
`__popcll(forms)`, one enum whose new member overlaps a shipped bit position,
and a TXN-only mining result becomes a production trajectory change with no flag
moved and no gate asked -- the exact shape of the B0 violation WP7-C already
committed once through the soundness channel.

So the rules below are about the ARGUMENT, not the body:

  1. the split into XE_SHIPPED_FORMS (bits 0..4) and XE_ALGEBRA_FORMS (bits
     5..12) exists as a constant, is exhaustive and is disjoint;
  2. every launch reachable with RASBERY_GPU_XE_TXN unset is handed
     xeShippedFormMask(), never xeFormMask();
  3. xeShippedFormMask() is exactly `xeFormMask() & XE_SHIPPED_FORMS` -- it does
     not re-resolve, so it cannot disagree with the receipt;
  4. the TXN entry point is the ONLY place the full resolved mask crosses into
     device code, and it is gated by rasberyGpuXeTxnEnabled() on the host;
  5. the [RASBERY][FORMS] receipt prints BOTH sub-masks, because the union alone
     cannot answer "did the bits the production arm runs under move?" -- on 238
     the union moved and the shipped sub-mask did not, and those are opposite
     verdicts about the B0 rule;
  6. the shipped kernel bodies still narrow every field they read, so rule 2 is
     belt AND braces rather than a single point of failure.

WHAT THIS FILE CANNOT DECIDE.  Which mask a given host mines.  That is a runtime
fact and the receipt reports it; see
docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md for the 238 runbook.

NEGATIVE CONTROLS.  Every rule is also run against a synthetic snippet that
violates it, so a rule that has quietly stopped matching anything fails here
rather than passing forever.

Pure python, no build, no device.

Run:  python tools/test_xe_forms_shipped_split_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures: list[str] = []
checks = 0


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def check(ok: bool, what: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(what)


def strip_comments(text: str) -> str:
    """Comments cannot launch a kernel, so every rule here reads CODE only.
    Naive on purpose -- a rule that needed a lexer would be a second C++ front
    end nobody maintains."""
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


KERNEL = read(os.path.join("src", "XeKernel.h"))
MASK_H = read(os.path.join("src", "XeFormMask.h"))
MINER = read(os.path.join("src", "XeFormMiner.cpp"))
BACKEND = read(os.path.join("src", "CudaXsReconBackend.cu"))
AUDIT = read(os.path.join("src", "XeFormAudit.cpp"))

KERNEL_CODE = strip_comments(KERNEL)
MINER_CODE = strip_comments(MINER)
BACKEND_CODE = strip_comments(BACKEND)


# ---------------------------------------------------------------------------
# 1. THE SPLIT IS A CONSTANT, EXHAUSTIVE AND DISJOINT.
#
# Two named masks whose union is every mined bit and whose intersection is
# empty.  Spelled from the bit constants rather than as literals, so a future
# site added at bit 13 widens the algebra half automatically instead of falling
# off the end of a hand-written 0x1fe0.
# ---------------------------------------------------------------------------
def rule_shipped_mask_is_bits_0_to_4(text: str) -> bool:
    m = re.search(r"XE_SHIPPED_FORMS\s*=\s*([^;]+);", text)
    if m is None:
        return False
    expr = re.sub(r"\s+", "", m.group(1))
    return expr == "(1ull<<XE_TXN_DET_BIT)-1ull"


def rule_algebra_mask_is_the_complement(text: str) -> bool:
    m = re.search(r"XE_ALGEBRA_FORMS\s*=\s*([^;]+);", text)
    if m is None:
        return False
    expr = re.sub(r"\s+", "", m.group(1))
    return expr == "(((1ull<<XE_BIT_COUNT)-1ull)&~XE_SHIPPED_FORMS)"


def _bit_consts(text: str) -> dict:
    out = {}
    for name, value in re.findall(r"constexpr\s+int\s+(XE_\w+)\s*=\s*(\d+)\s*;", text):
        out[name] = int(value)
    return out


def rule_split_partitions_the_mined_bits(text: str) -> bool:
    """Evaluate the two expressions from the header's own constants, then assert
    the partition.  A test that re-typed 0x1f and 0x1fe0 would agree with a
    header that had drifted; this one cannot."""
    c = _bit_consts(text)
    if "XE_TXN_DET_BIT" not in c or "XE_BIT_COUNT" not in c:
        return False
    shipped = (1 << c["XE_TXN_DET_BIT"]) - 1
    algebra = ((1 << c["XE_BIT_COUNT"]) - 1) & ~shipped
    all_bits = (1 << c["XE_BIT_COUNT"]) - 1
    return (shipped & algebra) == 0 and (shipped | algebra) == all_bits and shipped != 0


def rule_shipped_sites_are_inside_the_shipped_mask(text: str) -> bool:
    """Every pre-WP7-C site's TWO-bit field lies inside XE_SHIPPED_FORMS, and
    every WP7-C site's lies inside XE_ALGEBRA_FORMS.  This is the check that
    catches an enum renumbering that makes DET/G0/G1/PROJ overlap the dot or the
    candidate -- the failure mode the shipped/algebra names would otherwise hide."""
    c = _bit_consts(text)
    shipped = (1 << c["XE_TXN_DET_BIT"]) - 1
    algebra = ((1 << c["XE_BIT_COUNT"]) - 1) & ~shipped
    widths = {"XE_DOT_FIRST_BIT": 2, "XE_DOT_THIRD_BIT": 1,
              "XE_CAND1_BIT": 1, "XE_CAND2_BIT": 1,
              "XE_TXN_DET_BIT": 2, "XE_TXN_G0_BIT": 2,
              "XE_TXN_G1_BIT": 2, "XE_TXN_PROJ_BIT": 2}
    for name, width in widths.items():
        if name not in c:
            return False
        field = ((1 << width) - 1) << c[name]
        home = shipped if name.startswith("XE_TXN_") is False else algebra
        if field & ~home:
            return False
    return True


check(rule_shipped_mask_is_bits_0_to_4(KERNEL_CODE),
      "XE_SHIPPED_FORMS is (1ull << XE_TXN_DET_BIT) - 1ull -- derived from where the "
      "algebra channel starts, not from a literal that can drift away from it")
check(rule_algebra_mask_is_the_complement(KERNEL_CODE),
      "XE_ALGEBRA_FORMS is every mined bit that is not shipped, spelled as the "
      "complement so a site added at bit 13 joins it without an edit here")
check(rule_split_partitions_the_mined_bits(KERNEL_CODE),
      "the two sub-masks partition the mined bits: disjoint, and their union is all "
      "XE_BIT_COUNT of them -- a bit in neither would be a bit no arm owns")
check(rule_shipped_sites_are_inside_the_shipped_mask(KERNEL_CODE),
      "every dot/candidate field lies inside XE_SHIPPED_FORMS and every WP7-C field "
      "inside XE_ALGEBRA_FORMS -- an overlapping renumbering is what would put an "
      "algebra measurement into a production contraction")


# ---------------------------------------------------------------------------
# 2. THE ACCESSOR MASKS, AND IT DOES NOT RE-RESOLVE.
#
# `xeFormMask() & XE_SHIPPED_FORMS` and nothing else.  A second static, a second
# getenv or a second mining here could disagree with the mask the receipt
# printed, and then the log would describe an arm the run did not take.
# ---------------------------------------------------------------------------
def rule_shipped_accessor_is_declared(text: str) -> bool:
    return re.search(r"unsigned\s+long\s+long\s+xeShippedFormMask\s*\(\s*\)\s*;",
                     text) is not None


def rule_shipped_accessor_masks_and_delegates(text: str) -> bool:
    body = body_of(text, "xeShippedFormMask()")
    squashed = re.sub(r"\s+", "", body)
    if squashed != "{returnxeFormMask()&XE_SHIPPED_FORMS;}":
        return False
    return "getenv" not in body and "static" not in body


check(rule_shipped_accessor_is_declared(strip_comments(MASK_H)),
      "src/XeFormMask.h declares xeShippedFormMask() -- the split is part of the "
      "header's contract, not a private trick inside the .cpp")
check(rule_shipped_accessor_masks_and_delegates(MINER_CODE),
      "xeShippedFormMask() is exactly `xeFormMask() & XE_SHIPPED_FORMS`: one "
      "resolution, one receipt, no second environment read that could disagree with "
      "the line the run printed")


# ---------------------------------------------------------------------------
# 3. EVERY LIVE-ARM LAUNCH TAKES THE SHIPPED MASK; ONLY THE TXN TAKES THE UNION.
#
# The split arm's two entry points are XsReconBackend::xeDots and
# ::xeCandidate; the transaction's is ::xeTransaction, which Driver.h calls only
# behind rasberyGpuXeTxnEnabled().  The rule is stated over the FUNCTION BODIES
# so a new launch added inside one of them is covered without an edit here.
# ---------------------------------------------------------------------------
LIVE_ENTRY_POINTS = ("XsReconBackend::xeDots(", "XsReconBackend::xeCandidate(")


def rule_live_entry_points_take_shipped(text: str) -> bool:
    for sig in LIVE_ENTRY_POINTS:
        body = body_of(text, sig)
        if "xeShippedFormMask()" not in body:
            return False
        # `xeFormMask()` is a substring of `xeShippedFormMask()`, so the
        # forbidden spelling has to be looked for with its qualifier.
        if re.search(r"(?<!Shipped)\bxe::xeFormMask\s*\(", body):
            return False
    return True


def rule_txn_is_the_only_full_mask_site(text: str) -> bool:
    """Count qualified xe::xeFormMask() uses in the backend TU; every one of
    them must be inside xeTransaction."""
    total = len(re.findall(r"\bxe::xeFormMask\s*\(", text))
    inside = len(re.findall(r"\bxe::xeFormMask\s*\(",
                            body_of(text, "XsReconBackend::xeTransaction(")))
    return total >= 1 and total == inside


def rule_txn_entry_is_host_gated(text: str) -> bool:
    """The full mask is allowed into xeTransaction only because nothing calls
    xeTransaction with the flag off.  That gate lives in Driver.h; assert it is
    still the flag and still an `&&` that short-circuits before the call."""
    body = body_of(text, "static bool TryAndersonXeStepGpu(")
    return ("rasberyGpuXeTxnEnabled()" in body
            and re.search(r"if\s*\(\s*txn\s*&&\s*TryAndersonXeStepGpuTxn\s*\(", body)
            is not None)


check(rule_live_entry_points_take_shipped(BACKEND_CODE),
      "XsReconBackend::xeDots and ::xeCandidate launch with xeShippedFormMask() and "
      "never with xe::xeFormMask() -- these two run on every Xe step of the "
      "RASBERY_GPU_XE arm, with RASBERY_GPU_XE_TXN unset")
check(rule_txn_is_the_only_full_mask_site(BACKEND_CODE),
      "every xe::xeFormMask() in src/CudaXsReconBackend.cu is inside "
      "XsReconBackend::xeTransaction -- the transaction is the only device path "
      "that reads bits 5..12, so it is the only one entitled to receive them")
check(rule_txn_entry_is_host_gated(strip_comments(read(os.path.join("src", "Driver.h")))),
      "TryAndersonXeStepGpu still reaches the transaction only through "
      "`txn && TryAndersonXeStepGpuTxn(...)` with txn from rasberyGpuXeTxnEnabled() -- "
      "that short circuit is why the union mask is unreachable at TXN=0")


# ---------------------------------------------------------------------------
# 4. THE RECEIPT PRINTS BOTH SUB-MASKS.
#
# The 238 evidence is the argument: the union moved (0xd -> 0xd2d) and the
# shipped half did not.  A receipt that printed only the union invited exactly
# the wrong diagnosis, and did.
# ---------------------------------------------------------------------------
def rule_receipt_prints_both_submasks(text: str) -> bool:
    body = body_of(text, "unsigned long long xeFormMask()")
    return ('\\"shipped\\":' in body
            and '\\"algebra\\":' in body
            and "XE_SHIPPED_FORMS" in body
            and "XE_ALGEBRA_FORMS" in body
            and "[RASBERY][FORMS]" in body)


def rule_receipt_still_prints_the_union(text: str) -> bool:
    """The full mined and resolved values stay in the log.  A split that hid the
    union would make the mining unauditable, which is the opposite failure."""
    body = body_of(text, "unsigned long long xeFormMask()")
    return "resolveCalibratedFormMask(" in body and '\\"resolved\\":' in body


check(rule_receipt_prints_both_submasks(MINER_CODE),
      "the [RASBERY][FORMS] receipt names `shipped` and `algebra` and derives both "
      "from XE_SHIPPED_FORMS / XE_ALGEBRA_FORMS -- one number cannot say which arm's "
      "contraction moved")
check(rule_receipt_still_prints_the_union(MINER_CODE),
      "the receipt still carries resolveCalibratedFormMask's full line plus the "
      "resolved union: the mined mask must be reportable whole")


# ---------------------------------------------------------------------------
# 5. BELT AND BRACES -- THE SHIPPED BODIES STILL NARROW WHAT THEY READ.
#
# Rule 3 makes the algebra bits absent from the argument.  These rules keep the
# bodies unable to see them even if a future caller passes the union anyway.
# They are also the source-level proof that 0xd2d could not have moved the
# split arm's arithmetic on 238.
# ---------------------------------------------------------------------------
def rule_dot_narrows_its_fields(text: str) -> bool:
    body = body_of(text, "double xeDotChunk(")
    squashed = re.sub(r"\s+", "", body)
    return ("(forms>>XE_DOT_FIRST_BIT)&3ull" in squashed
            and "(forms>>XE_DOT_THIRD_BIT)&1ull" in squashed
            and "forms" not in re.sub(r"forms>>XE_DOT_\w+\)&\dull", "", squashed)
            .replace("unsignedlonglongforms", ""))


def rule_candidate_narrows_its_field(text: str) -> bool:
    body = body_of(text, "void xeCandidateOrdinal(")
    squashed = re.sub(r"\s+", "", body)
    return ("cand_bit=(ncol<=1)?XE_CAND1_BIT:XE_CAND2_BIT" in squashed
            and "(forms>>cand_bit)&1ull" in squashed)


def rule_no_whole_mask_predicate_in_kernel_header(text: str) -> bool:
    """The failure this file was commissioned to look for: a body that reads the
    mask as a WHOLE -- compared against the default, popcounted, or switched on
    -- instead of extracting a field.  Such a body turns any algebra bit into a
    production branch."""
    forbidden = (r"forms\s*[!=]=\s*XE_FORMS_DEFAULT", r"popc\w*\s*\(\s*forms",
                 r"forms\s*[!=]=\s*0\b", r"switch\s*\(\s*forms\s*\)")
    return not any(re.search(p, text) for p in forbidden)


check(rule_dot_narrows_its_fields(KERNEL_CODE),
      "xeDotChunk reads only (forms >> XE_DOT_FIRST_BIT) & 3 and "
      "(forms >> XE_DOT_THIRD_BIT) & 1 -- nothing above bit 2 is observable in it")
check(rule_candidate_narrows_its_field(KERNEL_CODE),
      "xeCandidateOrdinal reads only one bit, chosen by window width -- nothing "
      "above bit 4 is observable in it")
check(rule_no_whole_mask_predicate_in_kernel_header(KERNEL_CODE),
      "no body in src/XeKernel.h tests the mask as a whole (== default, popcount, "
      "switch) -- a whole-mask predicate is how a TXN-only mining result would "
      "become a production branch")


# ---------------------------------------------------------------------------
# 6. THE AUDIT KEEPS THE UNION, AND STAYS OFF BY DEFAULT.
#
# xe::auditAndersonFit scores the WP7-C algebra, so the union is the right
# argument there.  What must not change is that it is unreachable unless
# RASBERY_XE_FORMS_AUDIT is set.
# ---------------------------------------------------------------------------
def rule_audit_takes_the_union_and_is_gated(text: str) -> bool:
    body = body_of(text, "void auditAndersonFit(")
    return ("xeFormMask()" in body and "xeShippedFormMask()" not in body
            and 'std::getenv("RASBERY_XE_FORMS_AUDIT")' in text)


check(rule_audit_takes_the_union_and_is_gated(strip_comments(AUDIT)),
      "auditAndersonFit still runs under the FULL mask (it scores bits 5..12) and is "
      "still reached only through RASBERY_XE_FORMS_AUDIT")


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Each rule, fed a snippet that breaks it, must fail.
# ---------------------------------------------------------------------------
NEGATIVES = [
    ("the shipped mask written as a literal",
     lambda: rule_shipped_mask_is_bits_0_to_4(
         "constexpr unsigned long long XE_SHIPPED_FORMS = 0x1full;")),
    ("the algebra mask written as a literal",
     lambda: rule_algebra_mask_is_the_complement(
         "constexpr unsigned long long XE_ALGEBRA_FORMS = 0x1fe0ull;")),
    ("a mined bit in neither half",
     lambda: rule_split_partitions_the_mined_bits(
         "constexpr int XE_TXN_DET_BIT = 5;\nconstexpr int XE_BIT_COUNT = 13;\n"
         .replace("XE_BIT_COUNT = 13", "XE_BIT_COUNT = 4"))),
    ("a WP7-C site renumbered onto the candidate's bit",
     lambda: rule_shipped_sites_are_inside_the_shipped_mask(
         "constexpr int XE_DOT_FIRST_BIT = 0;\nconstexpr int XE_DOT_THIRD_BIT = 2;\n"
         "constexpr int XE_CAND1_BIT = 3;\nconstexpr int XE_CAND2_BIT = 4;\n"
         "constexpr int XE_TXN_DET_BIT = 4;\nconstexpr int XE_TXN_G0_BIT = 7;\n"
         "constexpr int XE_TXN_G1_BIT = 9;\nconstexpr int XE_TXN_PROJ_BIT = 11;\n"
         "constexpr int XE_BIT_COUNT = 13;\n")),
    ("the accessor undeclared",
     lambda: rule_shipped_accessor_is_declared(
         "unsigned long long xeFormMask();\n")),
    ("the accessor resolving a second time",
     lambda: rule_shipped_accessor_masks_and_delegates(
         "unsigned long long xeShippedFormMask() {\n"
         "    static const unsigned long long m = mine() & XE_SHIPPED_FORMS;\n"
         "    return m;\n}\n")),
    ("the accessor handing back the union",
     lambda: rule_shipped_accessor_masks_and_delegates(
         "unsigned long long xeShippedFormMask() { return xeFormMask(); }\n")),
    ("a live entry point launched with the union",
     lambda: rule_live_entry_points_take_shipped(
         "bool XsReconBackend::xeDots(int ncol) {\n"
         "    kXeDotStage1<<<1, 1>>>(xe::xeFormMask());\n}\n"
         "bool XsReconBackend::xeCandidate(int ncol) {\n"
         "    kXeCandidate<<<1, 1>>>(xe::xeShippedFormMask());\n}\n")),
    ("the union leaking outside the transaction",
     lambda: rule_txn_is_the_only_full_mask_site(
         "bool XsReconBackend::xeCommit(int t) { auto f = xe::xeFormMask(); }\n"
         "bool XsReconBackend::xeTransaction(int t) { auto f = xe::xeFormMask(); }\n")),
    ("the transaction reached without its flag",
     lambda: rule_txn_entry_is_host_gated(
         "static bool TryAndersonXeStepGpu(int a) {\n"
         "    if (TryAndersonXeStepGpuTxn(a)) return true;\n}\n")),
    ("a receipt that prints only the union",
     lambda: rule_receipt_prints_both_submasks(
         "unsigned long long xeFormMask() {\n"
         "    std::cerr << \"[RASBERY][FORMS] 0x\" << resolved;\n"
         "    return resolved;\n}\n")),
    ("a receipt that hides the union",
     lambda: rule_receipt_still_prints_the_union(
         "unsigned long long xeFormMask() {\n"
         "    std::cerr << \\\"shipped\\\" << (m & XE_SHIPPED_FORMS);\n"
         "    return m;\n}\n")),
    ("the dot widened to three bits",
     lambda: rule_dot_narrows_its_fields(
         "double xeDotChunk(unsigned long long forms) {\n"
         "    const unsigned s = (forms >> XE_DOT_FIRST_BIT) & 7ull;\n"
         "    const bool t = (forms >> XE_DOT_THIRD_BIT) & 1ull;\n}\n")),
    ("the dot reading the mask a second time, unmasked",
     lambda: rule_dot_narrows_its_fields(
         "double xeDotChunk(unsigned long long forms) {\n"
         "    const unsigned s = (forms >> XE_DOT_FIRST_BIT) & 3ull;\n"
         "    const bool t = (forms >> XE_DOT_THIRD_BIT) & 1ull;\n"
         "    if (forms > 0x1full) return 0.0;\n}\n")),
    ("the candidate reading the raw mask",
     lambda: rule_candidate_narrows_its_field(
         "void xeCandidateOrdinal(unsigned long long forms) {\n"
         "    const bool fused = forms != 0;\n}\n")),
    ("a whole-mask predicate in a shipped body",
     lambda: rule_no_whole_mask_predicate_in_kernel_header(
         "if (forms != XE_FORMS_DEFAULT) { fused = true; }\n")),
    ("a popcount of the whole mask",
     lambda: rule_no_whole_mask_predicate_in_kernel_header(
         "const int n = __popcll(forms);\n")),
    ("the audit demoted to the shipped half",
     lambda: rule_audit_takes_the_union_and_is_gated(
         'bool on = std::getenv("RASBERY_XE_FORMS_AUDIT");\n'
         "void auditAndersonFit(int n) { auto f = xeShippedFormMask(); }\n")),
]
for label, probe in NEGATIVES:
    checks += 1
    try:
        fired = not probe()
    except Exception:
        fired = True
    if not fired:
        failures.append("negative control did not fire: " + label)


# ---------------------------------------------------------------------------
if failures:
    print("FAIL (%d of %d checks)" % (len(failures), checks))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("PASS  tools/test_xe_forms_shipped_split_contract.py  "
      "(%d checks, %d negative controls)" % (checks, len(NEGATIVES)))
print("  the split arm's kernels are launched with bits 0..4 only; bits 5..12 reach")
print("  XsReconBackend::xeTransaction and the RASBERY_XE_FORMS_AUDIT path and")
print("  nothing else.  Which bits a HOST mines is a runtime fact: read the")
print("  [RASBERY][FORMS] {\"shipped\":...,\"algebra\":...} line.")
