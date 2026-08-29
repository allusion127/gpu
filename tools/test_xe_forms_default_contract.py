#!/usr/bin/env python3
"""Source contract: THE PRODUCTION Xe CONTRACTION MASK IS NOT A SHARED CHANNEL.

WHY THIS FILE EXISTS -- the regression it would have caught.

RASBERY_XE_FORMS is not a baked constant.  XeFormMiner.cpp MINES it at startup,
on the machine the run is on, and hands the result to kXeDotStage1 and
kXeCandidate (src/CudaXsReconBackend.cu) -- the live device Xe Anderson path
whenever RASBERY_GPU_XE=1.  So the mask is a runtime-resolved NUMERIC CONTRACT
of a production arm, and anything that moves it moves the trajectory with no
flag touched and no gate asked.  That is precisely what the campaign's B0 rule
forbids, and it is invisible in a diff that reads like test scaffolding.

WP7-C (71092e2) moved it through two doors at once:

  1. THE SOUNDNESS VERDICT WAS SHARED.  Four new sites (bits 5..12, evaluated
     only by RASBERY_GPU_XE_TXN, which is default OFF) were scored into the one
     `bad` total that `mineStable` derives `sound` from.  A host that cannot
     mine the NEW sites then reports the WHOLE mask unsound, and
     resolveCalibratedFormMask falls back to XE_FORMS_DEFAULT -- swapping the
     dot/candidate contraction of an arm nobody had touched.  On a host whose
     mined bits 0..4 differ from the shipped default (the header itself records
     0x6 on the dev box vs 0x7 on 238 for the sibling CMFD mask, so this is not
     hypothetical) that is a silent physics change.

  2. THE REFERENCE TRANSLATION UNIT WAS EDITED.  src/XeAndersonReference.cpp
     exists as its own object file for one stated reason: with the quotation and
     the shipped body in one TU, gcc common-subexpressions across them and
     changes what the reference measures.  WP7-C added `#include <cmath>`, a
     `std::sqrt` fixture loop of 33 lines INSIDE buildFixture(), and a whole new
     exported function to that file -- directly ahead of `refDot` and
     `refCandidate`, which are the quotations bits 0..4 are mined against.  The
     separation argument applies one level down and was not applied.

So the rules below are about SHAPE, and shape is where this goes wrong quietly:
the shipped channel gets its own scorer, its own site list, its own pass budget
and its own soundness verdict, and the file that holds the shipped quotations
holds nothing else.

WHAT THIS FILE CANNOT DECIDE.  Whether a given host's mined mask is 0xd or
something else -- that needs a compiler and is what the `[RASBERY][FORMS]`
receipt line reports at runtime.  See
docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md for the 238 runbook that reads
it.

NEGATIVE CONTROLS.  Every structural rule is also run against a synthetic
snippet that violates it, so a rule that has quietly stopped matching anything
fails here rather than passing forever.

Pure python, no build, no device.

Run:  python tools/test_xe_forms_default_contract.py
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


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def strip_comments(text: str) -> str:
    """Comments cannot move codegen, so the TU-separation rules are about CODE.
    Naive on purpose: these two files carry no string literal with a `//` in it,
    and a rule that needed a lexer here would be a second C++ front end."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def body_of(text: str, signature_fragment: str) -> str:
    """The brace-balanced body of the first function whose header contains
    `signature_fragment`.  Crude on purpose: a real parser here would be a
    second C++ front end nobody maintains."""
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
MINE = read(os.path.join("src", "XeFormMine.h"))
MINER = read(os.path.join("src", "XeFormMiner.cpp"))
MASK_H = read(os.path.join("src", "XeFormMask.h"))
REF_SHIPPED = read(os.path.join("src", "XeAndersonReference.cpp"))
REF_ALGEBRA = read(os.path.join("src", "XeAlgebraReference.cpp"))
CMAKE = read("CMakeLists.txt")


# ---------------------------------------------------------------------------
# 1. THE SHIPPED DEFAULT AND THE SHIPPED SITES.
#
# XE_FORMS_DEFAULT is the value the run falls back to, so it is also the value a
# regression falls back TO.  It and the four bit positions under it are the
# pre-WP7-C contract and may not move without a gate of their own.
# ---------------------------------------------------------------------------
SHIPPED_BITS = {
    "XE_DOT_FIRST_BIT": 0,
    "XE_DOT_THIRD_BIT": 2,
    "XE_CAND1_BIT": 3,
    "XE_CAND2_BIT": 4,
}


def rule_shipped_bit_positions(text: str) -> bool:
    for name, want in SHIPPED_BITS.items():
        m = re.search(r"constexpr\s+int\s+%s\s*=\s*(\d+)\s*;" % name, text)
        if m is None or int(m.group(1)) != want:
            return False
    return True


def rule_shipped_default(text: str) -> bool:
    m = re.search(r"constexpr\s+unsigned\s+long\s+long\s+XE_FORMS_DEFAULT\s*=\s*"
                  r"(0x[0-9a-fA-F]+)ull\s*;", text)
    return m is not None and int(m.group(1), 16) == 0xd


check(rule_shipped_bit_positions(KERNEL),
      "the four pre-WP7-C sites still sit at bits 0, 2, 3 and 4 -- the device dot and "
      "candidate kernels read exactly those and a renumbering is a silent remap")
check(rule_shipped_default(KERNEL),
      "XE_FORMS_DEFAULT is still 0xdull; it is what an unsound mining falls back to, "
      "so moving it moves the trajectory of every host that falls back")
check(int(re.search(r"XE_FORMS_DEFAULT\s*=\s*(0x[0-9a-fA-F]+)ull", KERNEL).group(1), 16)
      >> 5 == 0,
      "no bit above the shipped four is set in XE_FORMS_DEFAULT -- a default for an "
      "unmeasured site is the absence of a measurement, and 0 is how the header says so")


# ---------------------------------------------------------------------------
# 2. THE SHIPPED QUOTATION'S TRANSLATION UNIT HOLDS ONLY THE SHIPPED QUOTATION.
#
# This is rule (2) of the header comment.  refDot and refCandidate are what bits
# 0..4 are measured against; anything else compiled beside them can move what
# the compiler does to them, and therefore can move the mask a production arm
# runs under.
# ---------------------------------------------------------------------------
FOREIGN_IN_SHIPPED_TU = ("refAlgebra", "buildAlgebraFixture", "alg_cases",
                         "f.alg", "std::sqrt", "<cmath>")


def rule_shipped_tu_is_clean(text: str) -> bool:
    return not any(token in text for token in FOREIGN_IN_SHIPPED_TU)


check(rule_shipped_tu_is_clean(strip_comments(REF_SHIPPED)),
      "src/XeAndersonReference.cpp still holds ONLY the WP7-A quotations (%s) -- "
      "anything else in that TU can move the codegen bits 0..4 are mined against"
      % ", ".join(FOREIGN_IN_SHIPPED_TU))
check("refDot" in REF_SHIPPED and "refCandidate" in REF_SHIPPED,
      "the shipped quotations are still where the mining expects them")
check("refAlgebra" in REF_ALGEBRA and "buildAlgebraFixture" in REF_ALGEBRA,
      "the WP7-C quotation and its fixture live in src/XeAlgebraReference.cpp")
check("refDot" not in strip_comments(REF_ALGEBRA)
      and "refCandidate" not in strip_comments(REF_ALGEBRA),
      "the WP7-C TU does not re-quote the shipped bodies -- the separation runs both "
      "ways or it is not a separation")
check("buildAlgebraFixture(Fixture& f)" in REF_ALGEBRA
      and "std::uint64_t s = 0x" in REF_ALGEBRA,
      "the WP7-C fixture is seeded in its own TU rather than drawn from the tail of "
      "buildFixture's stream, which is what forced it into the shipped TU")


# ---------------------------------------------------------------------------
# 3. TWO SCORERS, AND NEITHER READS THE OTHER'S SITES.
# ---------------------------------------------------------------------------
SHIPPED_ONLY = ("xeDotChunk", "refDot", "xeCandidateOrdinal", "refCandidate")
ALGEBRA_ONLY = ("xeAndersonFit", "refAlgebra", "f.alg", "alg_cases")


def rule_scorers_are_disjoint(text: str) -> bool:
    shipped = body_of(text, "long scoreShippedMask(")
    algebra = body_of(text, "long scoreAlgebraMask(")
    if any(token in shipped for token in ALGEBRA_ONLY):
        return False
    if any(token in algebra for token in SHIPPED_ONLY):
        return False
    return all(token in shipped for token in SHIPPED_ONLY) and \
        all(token in algebra for token in ("xeAndersonFit", "refAlgebra"))


check(rule_scorers_are_disjoint(MINE),
      "scoreShippedMask scores only the shipped sites and scoreAlgebraMask only the "
      "WP7-C ones -- one `bad` total for both is exactly how a site nobody ships "
      "demoted a mask everybody runs")


# ---------------------------------------------------------------------------
# 4. THE SOUNDNESS VERDICT IS PER CHANNEL, AND THE SHIPPED ONE DECIDES THE MASK.
# ---------------------------------------------------------------------------
def rule_soundness_is_split(text: str) -> bool:
    body = body_of(text, "mineStable(")
    ok_shipped = re.search(r"if\s*\(\s*scoreShippedMask\([^)]*\)\s*!=\s*0\s*\)\s*"
                           r"sound\s*=\s*false\s*;", body) is not None
    ok_algebra = re.search(r"if\s*\(\s*scoreAlgebraMask\([^)]*\)\s*!=\s*0\s*\)\s*"
                           r"algebra_sound\s*=\s*false\s*;", body) is not None
    # `sound` must never be written from the algebra channel.
    no_cross = re.search(r"scoreAlgebraMask\([^)]*\)\s*!=\s*0\s*\)\s*sound\s*=", body) is None
    return ok_shipped and ok_algebra and no_cross


def rule_resolver_uses_shipped_soundness(text: str) -> bool:
    call = re.search(r"resolveCalibratedFormMask\((.*?)\);", text, re.S)
    if call is None:
        return False
    args = [a.strip() for a in call.group(1).split(",")]
    # env_name, build_default, mined, sound, mask_name
    return len(args) == 5 and args[3] == "sound"


check(rule_soundness_is_split(MINE),
      "mineStable reports the shipped verdict and the WP7-C verdict separately, and "
      "the algebra channel never writes `sound`")
check(squash("bool& sound, bool& algebra_sound") in squash(MINE)
      and squash("bool& sound, bool& algebra_sound") in squash(MASK_H),
      "both the mining and its declaration hand the two verdicts back BY NAME, so a "
      "caller cannot merge them by accident")
check(rule_resolver_uses_shipped_soundness(MINER),
      "XeFormMiner.cpp passes the SHIPPED soundness -- not a merged flag -- to "
      "resolveCalibratedFormMask, which is the one place the mask can be demoted")
check("algebra_sound" in MINER and "[RASBERY][WARN][FORMS]" in MINER,
      "an unsound WP7-C channel is reported rather than swallowed; it is the "
      "RASBERY_GPU_XE_TXN gate's problem, not the shipped arm's")


# ---------------------------------------------------------------------------
# 5. THE DESCENT IS PER CHANNEL TOO, WITH THE PASS BUDGET IT HAS ALWAYS HAD.
#
# The two site lists must be disjoint and the shipped list must be exactly the
# four pre-WP7-C bits, in their original order: the descent is greedy, so its
# answer is a property of the order it visits sites in.
# ---------------------------------------------------------------------------
def rule_site_lists(text: str) -> bool:
    shipped = re.search(r"kShippedSites\[4\]\s*=\s*\{(.*?)\};", text, re.S)
    algebra = re.search(r"kAlgebraSites\[4\]\s*=\s*\{(.*?)\};", text, re.S)
    if shipped is None or algebra is None:
        return False
    got = re.findall(r"xe::(XE_[A-Z0-9_]+_BIT)", shipped.group(1))
    if got != ["XE_DOT_FIRST_BIT", "XE_DOT_THIRD_BIT", "XE_CAND1_BIT", "XE_CAND2_BIT"]:
        return False
    alg = re.findall(r"xe::(XE_[A-Z0-9_]+_BIT)", algebra.group(1))
    return alg == ["XE_TXN_DET_BIT", "XE_TXN_G0_BIT", "XE_TXN_G1_BIT",
                   "XE_TXN_PROJ_BIT"]


def rule_shipped_pass_budget(text: str) -> bool:
    m = re.search(r"constexpr\s+int\s+SHIPPED_PASSES\s*=\s*(\d+)\s*;", text)
    return m is not None and int(m.group(1)) == 6


def rule_descent_channels(text: str) -> bool:
    body = body_of(text, "mineForms(const xeref::Fixture&")
    shipped_call = re.search(
        r"descend\(\s*f\s*,\s*seed\s*,\s*kShippedSites\s*,\s*scoreShippedMask\s*,\s*"
        r"SHIPPED_PASSES", body) is not None
    algebra_call = re.search(
        r"descend\(\s*f\s*,\s*shipped\s*,\s*kAlgebraSites\s*,\s*scoreAlgebraMask\s*,\s*"
        r"ALGEBRA_PASSES", body) is not None
    return shipped_call and algebra_call


check(rule_site_lists(MINE),
      "the shipped site list is the four pre-WP7-C bits in their original visit order "
      "and shares no bit with the WP7-C list -- the descent is greedy, so the order "
      "IS part of the answer")
check(rule_shipped_pass_budget(MINE),
      "the shipped descent keeps its own pass budget of 6; raising a shared budget "
      "would change the shipped answer on any host where it had not converged")
check(rule_descent_channels(MINE),
      "mineForms descends the shipped sites against scoreShippedMask and the WP7-C "
      "sites against scoreAlgebraMask, in that order")


# ---------------------------------------------------------------------------
# 6. THE MINING FIXTURE HAS ONE ENTRY POINT.
#
# A caller that used xeref::buildFixture() directly would score the WP7-C
# channel against an empty `alg` and mine four don't-cares -- which reads as
# "measured" in every receipt.
# ---------------------------------------------------------------------------
def rule_one_fixture_entry(text: str) -> bool:
    body = body_of(text, "buildMiningFixture(int n)")
    return "xeref::buildFixture(n)" in body and "xeref::buildAlgebraFixture(f)" in body


check(rule_one_fixture_entry(MINE),
      "buildMiningFixture is the one entry point and it fills both halves")
check("xemine::buildMiningFixture(4096)" in MINER,
      "the production resolver mines on the full fixture")


# ---------------------------------------------------------------------------
# 7. THE TWO TRANSLATION UNITS ARE LINKED TOGETHER, ALWAYS.
#
# A target that lists the shipped quotation but not the WP7-C one would fail to
# link -- or, worse, would tempt the next author to merge them back.
# ---------------------------------------------------------------------------
def rule_cmake_pairs_the_tus(text: str) -> bool:
    shipped = text.count("src/XeAndersonReference.cpp")
    algebra = text.count("src/XeAlgebraReference.cpp")
    return shipped > 0 and shipped == algebra


check(rule_cmake_pairs_the_tus(CMAKE),
      "every CMake target that compiles XeAndersonReference.cpp also compiles "
      "XeAlgebraReference.cpp")


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Each rule, fed a snippet that breaks it, must fail.
# ---------------------------------------------------------------------------
NEGATIVES = [
    ("a shipped site renumbered",
     lambda: rule_shipped_bit_positions(
         "constexpr int XE_DOT_FIRST_BIT = 1;\nconstexpr int XE_DOT_THIRD_BIT = 2;\n"
         "constexpr int XE_CAND1_BIT = 3;\nconstexpr int XE_CAND2_BIT = 4;\n")),
    ("the shipped default moved",
     lambda: rule_shipped_default(
         "constexpr unsigned long long XE_FORMS_DEFAULT = 0x7ull;")),
    ("the WP7-C quotation back in the shipped TU",
     lambda: rule_shipped_tu_is_clean(
         "#include <cmath>\nbool refAlgebra(int) { return true; }\n")),
    ("one scorer for both channels",
     lambda: rule_scorers_are_disjoint(
         "long scoreShippedMask(int m) { return xeDotChunk(m) + xeAndersonFit(m); }\n"
         "long scoreAlgebraMask(int m) { return xeAndersonFit(m); }\n")),
    ("the algebra channel writes the shipped verdict",
     lambda: rule_soundness_is_split(
         "long mineStable(int f) {\n"
         "    if (scoreShippedMask(f, m) != 0) sound = false;\n"
         "    if (scoreAlgebraMask(f, m) != 0) sound = false;\n"
         "}\n")),
    ("the resolver fed a merged flag",
     lambda: rule_resolver_uses_shipped_soundness(
         'resolveCalibratedFormMask("RASBERY_XE_FORMS", XE_FORMS_DEFAULT, mined,\n'
         '                          sound && algebra_sound, "XE_FORMS");')),
    ("a WP7-C bit smuggled into the shipped site list",
     lambda: rule_site_lists(
         "kShippedSites[4] = {{xe::XE_DOT_FIRST_BIT, 3}, {xe::XE_DOT_THIRD_BIT, 2},\n"
         "  {xe::XE_CAND1_BIT, 2}, {xe::XE_TXN_DET_BIT, 3}};\n"
         "kAlgebraSites[4] = {{xe::XE_TXN_DET_BIT, 3}, {xe::XE_TXN_G0_BIT, 3},\n"
         "  {xe::XE_TXN_G1_BIT, 3}, {xe::XE_TXN_PROJ_BIT, 3}};\n")),
    ("the shipped pass budget raised",
     lambda: rule_shipped_pass_budget("constexpr int SHIPPED_PASSES = 10;")),
    ("the descent scored across channels",
     lambda: rule_descent_channels(
         "unsigned long long mineForms(const xeref::Fixture& f, int seed) {\n"
         "    return descend(f, seed, kShippedSites, scoreMask, SHIPPED_PASSES);\n"
         "}\n")),
    ("the fixture built without its algebra half",
     lambda: rule_one_fixture_entry(
         "xeref::Fixture buildMiningFixture(int n) { return xeref::buildFixture(n); }\n")),
    ("a CMake target that links only the shipped TU",
     lambda: rule_cmake_pairs_the_tus('"src/XeAndersonReference.cpp"\n')),
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
print("PASS  tools/test_xe_forms_default_contract.py  (%d checks, %d negative controls)"
      % (checks, len(NEGATIVES)))
print("  which mask a HOST mines is a runtime fact, not a source one: compare the")
print("  [RASBERY][FORMS] {\"mask\":\"XE_FORMS\"} line between two binaries -- see")
print("  docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md.")
