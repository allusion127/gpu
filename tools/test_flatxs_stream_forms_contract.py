#!/usr/bin/env python3
"""Branch-stream contraction mask and libm policy contract -- WP23.1.

Twelve properties.  Not one of them is visible to a numerical comparison of two
runs, which is exactly why they are asserted here: a passing kngr_238 A/B would
keep passing right up until the day one of them mattered.

  1. THE QUOTATION KEEPS ITS OWN TRANSLATION UNIT.  FlatXsStreamReference.cpp
     must never include FlatXsStreamKernel.h: with both in one TU gcc
     common-subexpressions across them and changes the QUOTATION's contraction,
     which is the reference the mining scores against.  Only
     FlatXsStreamFormMine.h -- and therefore only the miner and the gate -- sees
     both.  WP22 paid for this lesson twice and it is not re-learnable cheaply.

  2. THE QUOTATION IS THE PRODUCTION TEXT.  The three `+=` statements in the
     reference are XSSet.cpp's, character for character, modulo exactly three
     declared substitutions (the member arrow, the isotope registry constant,
     and the flat vs nested branch-x index).  A quotation that has been
     "tidied" is measuring a different function, and nothing in a passing run
     would say so.

  3. THE INLINING CONTEXT IS PART OF THE QUOTATION.  The production call graph
     is an OpenMP loop -> a per-node function -> one function holding all three
     lambdas, and the reference reproduces all three levels including the
     pragma and its schedule clause.  WP22's finding is that gcc re-makes the
     contraction decision per inlining context, so a fixture that calls the
     quoted expression from a plain loop pins a different mask.

  4. EVERY SITE IS WIRED, AND TO ITS OWN FIELD.  Each of the three shipped
     bodies routes its multiply-add through pol.ma() with ITS OWN site
     constant.  A site that shares another's field is one the mask cannot
     control, and it would mine as a don't-care rather than as a bug.

  5. TWO BITS PER SITE, XE_SITE_* ENCODING.  Offsets 0/2/4, FS_ALL == 0x3f, and
     the three state names.  A one-bit field would make the mask unable to
     express the encoding the sibling masks use and a reader would have to
     learn a second one.

  6. THE MASK IS MINED, NOT BAKED, AND THE RUNTIME READS THE RESOLVER.  The
     production accessor must delegate to streamFormMask(); a second getenv in
     the backend TU would be a second answer to the same question.

  7. FOUR SEEDS, AND SOUNDNESS IS "EVERY DESCENT REACHED ZERO".  Not "every
     descent agreed": P1 and P2 are the same bits at a single-product site, so
     pattern equality is a demand the arithmetic cannot meet.

  8. THE RECEIPT CARRIES THE SOURCE AND THE SOUNDNESS, not only the value.  A
     mask alone cannot distinguish "measured on this host", "typed by a human"
     and "fallen back to after a failed derivation".

  9. THE libm KNOB EXISTS, DEFAULTS TO fast, AND IS IN kArmEnv.  It selects a
     different coordinate on the arguments where glibc is not correctly
     rounded, so it moves the trajectory and must not be cacheable across
     values.

 10. EVERY libm CALL SITE TAKES THE POLICY.  A single `fsLog(x)` left without
     the mode would be a form that silently ignored the knob -- present in the
     receipt, absent from the arithmetic.

 11. THE EXACT MATH TOUCHES NO LIBM AND NO UNBARRIERED MULTIPLY.  Its whole
     claim is "the same bits under g++ and nvcc by construction", and that
     claim dies the moment a std::log, an ldexp or a bare `a * b` appears in
     it: twoProd's residual is only the exact residual if the product really
     was the rounded product, which is what xsrMul's barrier guarantees.

 12. THE GATE IS BUILT AND REGISTERED, with -march=native (no FMA in the ISA
     means nothing to mine) and with OpenMP (no -fopenmp means the quotation's
     body is not outlined and rule 3 is being asserted about text the compiler
     ignored).  And the FIXTURE SHAPE IS DEFINED ONCE, so the gate mines the
     operands the binary mines.

Every rule runs against a deliberately broken copy of the same text as a
negative control, so a rule that has stopped discriminating fails loudly instead
of passing vacuously.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "kernel": "src/FlatXsStreamKernel.h",
    "exact": "src/FlatXsStreamExactMath.h",
    "ref_h": "src/FlatXsStreamReference.h",
    "ref": "src/FlatXsStreamReference.cpp",
    "mine": "src/FlatXsStreamFormMine.h",
    "mask": "src/FlatXsStreamFormMask.h",
    "miner": "src/FlatXsStreamFormMiner.cpp",
    "receipt": "src/FlatXsStreamReceipt.h",
    "cu": "src/CudaXsReconBackend.cu",
    "probe": "test/flatxs_stream_form_probe.cpp",
    "xsset": "src/XSSet.cpp",
    "driver": "src/Driver.h",
    "cmake": "CMakeLists.txt",
}


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        raise AssertionError(f"missing file: {rel}")
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def region(text: str, start: str, end: str, what: str) -> str:
    i = text.find(start)
    assert i >= 0, f"{what}: opening marker {start!r} not found"
    j = text.find(end, i + len(start))
    assert j >= 0, f"{what}: closing marker {end!r} not found"
    return text[i : j + len(end)]


# ---------------------------------------------------------------------------
# The three quoted statements, and the ONLY substitutions the quotation may
# carry.  Each one is a naming difference forced by the reference having no
# XSSet to be a member of; none of them changes an operand, an operator or an
# order, which is the whole property being asserted.
# ---------------------------------------------------------------------------
SUBSTITUTIONS = [
    ("_lib->", ""),               # the reference holds the tables directly
    ("Isotope::niso", "niso"),    # the registry size travels as a fixture field
    ("[hi][a]", "[hi * 3 + a]"),  # branch-x is flat in the device layout
    ("[lo][a]", "[lo * 3 + a]"),
]

QUOTED_STATEMENTS = [
    # (name, the statement as XSSet.cpp writes it)
    ("referenceDensity0",
     "v += f * (lib_iden[hb * Isotope::niso + isotope] - v);"),
    ("referenceDensity",
     "value += fraction * (lib_iden[hi * Isotope::niso + isotope] - "
     "lib_iden[lo * Isotope::niso + isotope]);"),
    ("referenceCondition",
     "value += fraction * (lib_ref_branch_x[hi][a] - value);"),
]


def normalise(text: str) -> str:
    for needle, replacement in SUBSTITUTIONS:
        text = text.replace(needle, replacement)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def r_quotation_own_tu(src: dict[str, str]) -> None:
    for key in ("ref", "ref_h"):
        body = strip_comments(src[key])
        assert "FlatXsStreamKernel.h" not in body, (
            f"{FILES[key]} includes the shipped body; with both in one translation "
            "unit gcc common-subexpressions across them and the quotation stops "
            "being the production arithmetic")
        assert "FlatXsStreamExactMath.h" not in body, (
            f"{FILES[key]} reaches the shipped math header, which is the same "
            "contamination one level down")
    mine = strip_comments(src["mine"])
    assert "FlatXsStreamKernel.h" in mine and "FlatXsStreamReference.h" in mine, (
        "FlatXsStreamFormMine.h must be the ONE place that sees both sides; that "
        "is what makes the separation above enforceable")


def r_quotation_is_production_text(src: dict[str, str]) -> None:
    host = normalise(strip_comments(src["xsset"]))
    ref = normalise(strip_comments(src["ref"]))
    for name, statement in QUOTED_STATEMENTS:
        want = normalise(statement)
        assert want in host, (
            f"{name}: XSSet.cpp no longer contains {statement!r} -- the production "
            "site moved and the quotation is now measuring a function that is not "
            "the one that runs")
        assert want in ref, (
            f"{name}: the quotation in FlatXsStreamReference.cpp does not match "
            f"XSSet.cpp's statement {statement!r} under the declared "
            "substitutions; a 'tidied' quotation mines a different mask")


def r_inlining_context_quoted(src: dict[str, str]) -> None:
    ref = src["ref"]
    host = src["xsset"]
    pragma = "#pragma omp parallel for schedule(dynamic, 64)"
    assert pragma in host, (
        "BuildFlatXsStream's pragma changed; the quotation's context claim is now "
        "about a loop that no longer exists")
    assert pragma in ref, (
        "the quotation does not carry BuildFlatXsStream's OpenMP pragma -- gcc "
        "outlines an OpenMP body into its own function, which is a different "
        "inlining context and pins a different mask (WP22)")
    body = strip_comments(ref)
    assert "void refResolveNode(" in body, (
        "the per-node function level is missing; production reaches the lambdas "
        "through ResolveNodeApplications and the call depth is part of the "
        "context")
    holder = region(body, "void refResolveSpectralHistoryDeltas(",
                    "void refResolveNode(", "the lambda holder")
    for lam in ("auto referenceDensity0 = [&]", "auto referenceDensity = [&]",
                "auto referenceCondition = [&]"):
        assert lam in holder, (
            f"{lam!r} is not inside the single holder function; production has all "
            "three in one body and splitting them is a different inlining problem")


def r_every_site_wired(src: dict[str, str]) -> None:
    k = strip_comments(src["kernel"])
    expected = {
        "fsReferenceDensity(": "FS_REFDENS,",
        "fsReferenceDensity0(": "FS_REFDENS0,",
        "fsReferenceCondition(": "FS_REFCOND,",
    }
    for fn, site in expected.items():
        i = k.find("inline double " + fn)
        assert i >= 0, f"{fn} is gone from the shipped body"
        j = k.find("\n}", i)
        body = k[i:j]
        assert "pol.ma(" in body, f"{fn} no longer routes its lerp through the mask"
        assert site in body, (
            f"{fn} does not use {site.strip(',')}; a site sharing another's field "
            "is a site the mask cannot control and it would mine as a don't-care")
    # and no site is used by two bodies
    for site in ("FS_REFDENS,", "FS_REFDENS0,", "FS_REFCOND,"):
        assert k.count("pol.ma(" + site) == 1, (
            f"{site.strip(',')} is applied at more than one site; the three "
            "statements are contracted independently by gcc, which is the finding "
            "the xsrecon campaign paid for")


def r_two_bit_fields(src: dict[str, str]) -> None:
    k = strip_comments(src["kernel"])
    for decl in ("FS_REFDENS   = 0u", "FS_REFDENS0  = 2u", "FS_REFCOND   = 4u",
                 "FS_BIT_COUNT = 6u"):
        assert decl in k, f"the site field layout changed: {decl!r} is not declared"
    assert "constexpr unsigned FS_ALL = (1u << FS_BIT_COUNT) - 1u" in k, \
        "FS_ALL must be derived from the field count, not typed twice"
    for state in ("FS_SITE_NONE = 0u", "FS_SITE_P1   = 1u", "FS_SITE_P2   = 2u"):
        assert state in k, f"the XE_SITE_* state encoding is incomplete: {state!r}"
    ma = region(k, "double ma(unsigned bit,", "\n    }", "StreamForms::ma")
    assert "xsrFma(a, b, c)" in ma and "xsrFma(b, a, c)" in ma and \
           "xsrMul(a, b) + c" in ma, \
        "the three states must be the three spellings, all barriered"


def r_mask_is_mined(src: dict[str, str]) -> None:
    miner = strip_comments(src["miner"])
    assert "resolveCalibratedFormMask(" in miner, (
        "the resolution must go through GpuFormMask.h, which owns the precedence "
        "and prints the one receipt line a log is read for")
    assert '"RASBERY_FLATXS_STREAM_FORMS"' in miner, \
        "the env override name is not wired into the resolver"
    assert "mineStable(" in miner, "the miner does not mine"
    cu = strip_comments(src["cu"])
    accessor = region(cu, "unsigned rasberyGpuFlatXsStreamForms()", "\n}",
                      "rasberyGpuFlatXsStreamForms")
    assert "streamFormMask()" in accessor, (
        "the backend must read the production resolver; a private getenv here "
        "would be a second answer to the same question")
    assert "getenv" not in accessor, \
        "the backend re-reads the environment behind the resolver's back"


def r_four_seeds(src: dict[str, str]) -> None:
    mine = strip_comments(src["mine"])
    seeds = region(mine, "const unsigned seeds[", ";", "the seed list")
    values = [v.strip() for v in
              seeds[seeds.index("{") + 1 : seeds.index("}")].split(",")]
    assert len(values) >= 4, \
        f"only {len(values)} mining seeds; a single-seed descent measures its own "\
        "starting point"
    assert len(set(values)) == len(values), "the mining seeds are not distinct"
    stable = region(mine, "inline unsigned mineStable(", "\n}", "mineStable")
    assert "sound = false" in stable and "!= 0" in stable, \
        "soundness is not defined as every descent reaching zero mismatches"


def r_receipt_source_and_soundness(src: dict[str, str]) -> None:
    rc = src["receipt"]
    for field in (r'\"forms_mask\"', r'\"forms_source\"', r'\"forms_sound\"',
                  r'\"libm\"'):
        assert field in rc, f"the receipt is missing the {field} field"
    assert "forms_source{nullptr}" in rc and "libm_name{nullptr}" in rc, \
        "the receipt's source fields must start unset, so a run that never armed "\
        "the arm prints ~ rather than a value it never measured"
    cu = strip_comments(src["cu"])
    for store in ("tally.forms_source.store(", "tally.forms_sound.store(",
                  "tally.libm_name.store("):
        assert store in cu, f"the backend never writes {store}; the field would "\
            "always read as unmeasured"


def r_libm_knob(src: dict[str, str]) -> None:
    miner = strip_comments(src["miner"])
    knob = region(miner, "unsigned streamLibmMode()", "\n}", "streamLibmMode")
    assert '"RASBERY_GPU_FLATXS_STREAM_LIBM"' in knob, "the knob has no name"
    assert 'v == "exact"' in knob and 'v == "fast"' in knob, \
        "the knob must accept exactly the two documented values"
    assert "return kStreamLibmDefault;" in knob, \
        "an unrecognised value must fall back loudly, not be taken for one of them"
    k = strip_comments(src["kernel"])
    assert "kStreamLibmDefault = FS_LIBM_FAST" in k, (
        "the default must be `fast`: the exact path does not reproduce glibc, so "
        "making it the default would move the trajectory on the strength of a "
        "measurement that says it should not")
    arm = region(src["driver"], "inline constexpr const char* kArmEnv[] = {", "};",
                 "kArmEnv")
    assert '"RASBERY_GPU_FLATXS_STREAM_LIBM"' in arm, \
        "the libm knob selects a different coordinate; it must not be cacheable "\
        "across values"


def r_every_libm_site_takes_the_policy(src: dict[str, str]) -> None:
    k = strip_comments(src["kernel"])
    body = region(k, "flatxsStreamResolveNode(const flatxs::FlatXsView& v",
                  "out.node_cnt[i] = n;", "flatxsStreamResolveNode")
    calls = re.findall(r"\bfs(?:Log|Cbrt|RatioFormOf|NodeSpectralIndex)\b", body)
    assert calls, "the resolver evaluates no coordinate that could take the policy"
    # every one of those calls must carry spol.libm as its last argument
    for m in re.finditer(r"\bfs(Log|Cbrt)\(", body):
        tail = body[m.end():m.end() + 400]
        depth = 1
        arg = []
        for ch in tail:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            arg.append(ch)
        assert "spol.libm" in "".join(arg), (
            f"a fs{m.group(1)} call in the resolver does not take spol.libm; that "
            "form would silently ignore the knob")


def r_exact_math_is_self_contained(src: dict[str, str]) -> None:
    body = strip_comments(src["exact"])
    for banned in ("std::log", "std::cbrt", "std::pow", "std::fma", "ldexp",
                   "frexp", "__logf", "log(", "cbrt("):
        if banned in ("log(", "cbrt("):
            # allow the identifiers fsExactLog / fsExactCbrt themselves
            hits = [m for m in re.finditer(re.escape(banned), body)
                    if not re.search(r"[A-Za-z_]$", body[:m.start()])]
            assert not hits, \
                f"FlatXsStreamExactMath.h calls {banned!r}; its entire claim is that "\
                "it calls no libm at all"
            continue
        assert banned not in body, \
            f"FlatXsStreamExactMath.h uses {banned!r}; its entire claim is that it "\
            "is built from correctly-rounded primitives only"
    two_prod = region(body, "inline FsDD fsTwoProd(", "\n}", "fsTwoProd")
    assert "xsrMul(a, b)" in two_prod and "xsrFma(a, b, -p)" in two_prod, (
        "twoProd's residual is the exact residual only if the product really was "
        "the rounded product; an unbarriered `a * b` here makes every digit below "
        "the leading one noise")


def r_gate_is_built(src: dict[str, str]) -> None:
    cm = src["cmake"]
    target = region(cm, "add_executable(rasbery_flatxs_stream_form_probe",
                    "add_test(NAME flatxs_stream_form_probe", "the gate target")
    for src_file in ("test/flatxs_stream_form_probe.cpp",
                     "src/FlatXsStreamReference.cpp",
                     "src/FlatXsStreamFormMiner.cpp"):
        assert src_file in target, f"the gate does not build {src_file}"
    assert "-march=native" in target, (
        "the gate must be built with -march=native: with no FMA in the ISA gcc "
        "contracts nothing and the mask cannot be scored")
    assert "OpenMP::OpenMP_CXX" in target, (
        "the gate must link OpenMP, or the quotation's pragma is ignored and the "
        "body is not outlined -- which is the context difference the whole "
        "reference exists to reproduce")
    miner = strip_comments(src["miner"])
    probe = strip_comments(src["probe"])
    assert "buildProductionFixture()" in miner and "buildProductionFixture()" in probe, (
        "the gate and the binary must mine the same fixture; a gate that built its "
        "own would assert nothing about what the binary does")


RULES = [
    ("quotation-own-tu", r_quotation_own_tu, "ref",
     ('#include "FlatXsStreamReference.h"',
      '#include "FlatXsStreamReference.h"\n#include "FlatXsStreamKernel.h"')),
    ("quotation-is-production-text", r_quotation_is_production_text, "ref",
     ("v += f * (lib_iden[hb * niso + isotope] - v);",
      "v = v + f * (lib_iden[hb * niso + isotope] - v);")),
    ("inlining-context-quoted", r_inlining_context_quoted, "ref",
     ("#pragma omp parallel for schedule(dynamic, 64) if (false)", "")),
    ("every-site-wired", r_every_site_wired, "kernel",
     ("value = pol.ma(FS_REFCOND, fraction,", "value = pol.ma(FS_REFDENS, fraction,")),
    ("two-bit-fields", r_two_bit_fields, "kernel",
     ("FS_REFCOND   = 4u", "FS_REFCOND   = 1u")),
    ("mask-is-mined", r_mask_is_mined, "cu",
     ("return fss::streamFormMask();",
      'return static_cast<unsigned>(std::strtoul(std::getenv("X"), nullptr, 0));')),
    ("four-seeds", r_four_seeds, "mine",
     ("const unsigned seeds[4] = {0u, 0x15u, 0x2au, fss::kStreamFormsDefault};",
      "const unsigned seeds[4] = {0u, 0u, 0u, 0u};")),
    ("receipt-source-and-soundness", r_receipt_source_and_soundness, "receipt",
     (r'\"forms_source\"', r'\"x\"')),
    ("libm-knob", r_libm_knob, "kernel",
     ("kStreamLibmDefault = FS_LIBM_FAST", "kStreamLibmDefault = FS_LIBM_EXACT")),
    ("every-libm-site-takes-the-policy", r_every_libm_site_takes_the_policy, "kernel",
     ("coordinate = fsLog(fsMax(density, kSpectralLogDensityFloor), spol.libm);",
      "coordinate = fsLog(fsMax(density, kSpectralLogDensityFloor), 0u);")),
    ("exact-math-is-self-contained", r_exact_math_is_self_contained, "exact",
     ("const double p = xsrMul(a, b);", "const double p = a * b;")),
    ("gate-is-built", r_gate_is_built, "cmake",
     ("        PRIVATE OpenMP::OpenMP_CXX)\n    endif ()\n    if (NOT MSVC)\n"
      "        # -march=native: with no FMA in the ISA gcc contracts nothing and the",
      "        PRIVATE)\n    endif ()\n    if (NOT MSVC)\n"
      "        # -march=native: with no FMA in the ISA gcc contracts nothing and the")),
]


def main() -> int:
    failures: list[str] = []
    try:
        src = {k: read(v) for k, v in FILES.items()}
    except AssertionError as exc:
        print(f"FlatXS stream forms contract: FAIL {exc}")
        return 1

    for name, rule, _target, _control in RULES:
        try:
            rule(src)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")

    for name, rule, target, (needle, replacement) in RULES:
        broken = dict(src)
        if needle not in broken[target]:
            failures.append(
                f"{name}: negative control is stale, {needle!r} not in {FILES[target]}")
            continue
        broken[target] = broken[target].replace(needle, replacement, 1)
        try:
            rule(broken)
        except AssertionError:
            continue
        failures.append(f"{name}: negative control PASSED the rule -- the rule is vacuous")

    if failures:
        print("FlatXS stream forms contract: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"FlatXS stream forms contract: PASS ({len(RULES)} rules, each with a "
          "negative control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
