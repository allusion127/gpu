#!/usr/bin/env python3
"""GPU CRAM depletion contract (GA evaluator plan Sec 6.3 Task 16).

Fourteen properties.  Not one of them is visible to a numerical comparison of
two runs, which is exactly why they are asserted here: a passing kngr_238 A/B
would keep passing right up until the day one of them mattered.

  1. DEFAULT OFF, AND THE OFF PATH IS THE OLD PATH.  The backend is reached only
     through XSSet::DepleteGpu / XSSet::CorrectorStepGpu, whose false return must
     be followed by the untouched host node loop.  A refactor that dropped the
     fallback would turn "no CUDA device" into "no depletion", and the isotope
     inventory is every later statepoint's input.

  2. FAIL OPEN, NEVER THROW.  Every failure in the .cu resolves to a `return
     false`.  The host solve throws on two conditions (zero Gauss-Seidel
     diagonal, 64 sweeps without convergence); a throw out of a kernel is not a
     thing, and a throw out of the backend would take down a 64-deck batch for
     one slot.

  3. NO PROCESS-WIDE MUTABLE STATE.  The backend, its stream and every device
     buffer hang off an XSSet, which belongs to one Driver.  A mutable `static`
     in the .cu is the slot-0 bug class: deck 7 driving deck 0's buffers,
     correct-looking, wrong.  The CRAM pole tables are `__constant__` and
     immutable, which is the one allowed exception.

  4. --fmad=false ON THE TU.  The kernels restate DepleteNode/CorrectorStep and
     solveBatemanCRAM in the host's statement order.  The Gauss-Seidel sweep is
     `sum -= vals[i] * x[cols[i]]` up to 256 times per node, so one contracted
     multiply-add compounds before anything is compared.

  5. THE CLASS IS N1 AND SAYS SO.  Complex division goes through the device
     libm's logb/scalbn, so the arm is not guaranteed bit-identical and the
     header must not claim B0.

  6. THE POLE SET IS milk.h's.  alpha0, the four alpha and four theta pairs,
     max_iter, rel_tol, abs_tol and diag_tol must appear in the .cu exactly as
     the `order == 8` branch of milk.h spells them.  A device running a
     different quadrature would be a different method wearing the same receipt.

  7. THE ZERO COMPRESSION IS THE HOST'S.  The uploaded sparsity pattern is the
     node-independent union, so the kernel must still run the host's
     `if (value == 0.0) continue;` per node -- otherwise a node whose value
     happens to vanish carries a term the host never formed.

  8. THE DIVISION IS __divdc3, NOT cuCdiv.  libstdc++ lowers complex division to
     libgcc's rescaled algorithm.  CUDA's cuCdiv is Smith's method WITHOUT the
     logb rescale; using it would silently change the arithmetic the whole gate
     is measuring.

  9. FOUR CONDENSED SLOTS, AND THE HOST STILL ONLY READS FOUR.  The backend
     uploads XSAF/XSFF/XS2N/XS3N and no other micro-XS block, which is exact
     ONLY while BuildTransitionMatrix and ComputeXeEquilibrium read no other
     slot.  This rule is what makes a future fifth reader a test failure rather
     than a wrong answer.

 10. NOTHING REACHES THE HOST BEFORE THE STATUS REDUCTION.  The per-node status
     is downloaded and tested BEFORE the iden/burn D2H in both entry points, so
     a node that hit either throw condition -- or produced a non-finite density
     or burnup increment -- leaves the host inventory untouched and the host
     loop throws the same exception at the same node.

 11. THE STUB KEEPS CPU-ONLY BUILDS COMPILING, AND IT IS SYMBOL-COMPLETE.
     CudaCramBackendStub.cpp stands in for CudaCramBackend.cu whenever
     RASBERY_ENABLE_CUDA is OFF, so it must define EVERY member the header
     declares out of line -- and the member list is MINED FROM THE HEADER rather
     than kept by hand here, because a hand-kept list is exactly what was
     missing: WP21-D added kernelVariant/lanesPerNode/launches/launchUsMean to
     the header and the .cu, Driver.h called all four from the [RASBERY][CRAM_GPU]
     receipt with no `#ifdef RASBERY_HAS_CUDA`, the stub was never touched, and
     nothing in the tree noticed because the only configuration that links the
     stub is the one nobody builds.

 12. RASBERY_GPU_CRAM *IS* AN ARM KNOB, AND SAYS WHY.  The mirror image of
     test_ppr_gpu_contract.py's rule 10.  Depletion output feeds the next
     statepoint's XS, so the knob must be in trajectory::kArmEnv and the reason
     must be written where the list is.

 13. THE RECEIPT THE PLAN ASKS FOR.

 14. THE UNSUPPORTED MODES DECLINE.  RASBERY_PC_SUBSTEPS > 1 and a corrector
     whose BOS snapshot did not come from this statepoint's device predictor
     must both return false rather than run something plausible.

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
    "cu": "src/CudaCramBackend.cu",
    "hdr": "src/CudaCramBackend.h",
    "stub": "src/CudaCramBackendStub.cpp",
    "xsset_h": "src/XSSet.h",
    "xsset": "src/XSSet.cpp",
    "driver": "src/Driver.h",
    "cmake": "CMakeLists.txt",
    "milk": "include/milk.h",
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
# Rules.  Each takes the dict of raw sources and raises AssertionError.
# ---------------------------------------------------------------------------


def r_default_off(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert 'getenv("RASBERY_GPU_CRAM")' in cu, \
        "the arm must be gated on RASBERY_GPU_CRAM and default off"
    xs = strip_comments(src["xsset"])
    assert "if (DepleteGpu(dt, power, xe_transient)) {" in xs, \
        "XSSet::Deplete must try the device and fall through on false"
    assert "if (!corrector_on_device) {" in xs, \
        "XSSet::CorrectorStep must run its host node loop when the device declined"
    assert "++_cram_host_fallbacks;" in xs, \
        "a decline must be counted, or the receipt cannot say the arm never ran"


def r_fail_open(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "throw" not in cu, \
        "the backend must not throw: a statepoint-boundary throw kills a 64-deck batch"
    # Every cudaError_t test must end in a fail()/decline()/return false.
    assert cu.count("return s.fail(") + cu.count("return fail(") >= 10, \
        "CUDA errors must resolve to fail(), which returns false"
    assert "bool decline(const char* why)" in cu, \
        "an unsupported deck must decline, not assert"


def r_no_process_state(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    for m in re.finditer(r"\bstatic\b", cu):
        line = cu[cu.rfind("\n", 0, m.start()) + 1 : cu.find("\n", m.start())]
        assert "static_cast" in line or "static_assert" in line, \
            f"mutable process-wide state in the .cu: {line.strip()!r}"
    # __constant__ is allowed only for the immutable CRAM quadrature.
    for m in re.finditer(r"__constant__\s+\w+\s+(\w+)", cu):
        assert m.group(1) in ("kAlphaRe", "kAlphaIm", "kThetaRe", "kThetaIm"), \
            f"__constant__ {m.group(1)} is not one of the CRAM pole tables"


def r_fmad(src: dict[str, str]) -> None:
    # 2c04a6e moved the literal --fmad=false behind RASBERY_BITEXACT_CUDA_OPTS so
    # RASBERY_PTXAS_VERBOSE could APPEND to it.  What the contract needs is the
    # RESOLVED option list, not the spelling, so expand the variable -- and expand
    # a missing definition to "", so deleting the set() still fails this rule.
    cm = src["cmake"]
    m = re.search(
        r"set_source_files_properties\(\s*\"\$\{CMAKE_CURRENT_SOURCE_DIR\}/src/"
        r"CudaCramBackend\.cu\"(.{0,200}?)\)",
        cm, re.S)
    assert m, ("CudaCramBackend.cu has no set_source_files_properties entry -- nvcc would "
               "be free to contract the Gauss-Seidel sweep's multiply-adds")
    var = re.search(r"set\(RASBERY_BITEXACT_CUDA_OPTS\s+\"([^\"]*)\"\s*\)", cm)
    opts = m.group(1).replace("${RASBERY_BITEXACT_CUDA_OPTS}",
                              var.group(1) if var else "")
    assert "--fmad=false" in opts, \
        "CudaCramBackend.cu must be compiled with --fmad=false"


def r_class_is_n1(src: dict[str, str]) -> None:
    hdr = src["hdr"]
    assert "CLASS N1, NOT B0" in hdr, "the header must declare the gate class"
    assert "Gate A" in hdr, "the header must name the gate the class implies"
    assert "bit-identical" not in hdr.replace(
        "cannot simply copy the host", ""
    ) or "not guaranteed" in hdr or "equality is a measurement" in hdr, \
        "the header must not claim bit-identity it cannot promise"


def r_pole_set(src: dict[str, str]) -> None:
    milk = src["milk"]
    i = milk.find("if (order == 8) {")
    assert i >= 0, "milk.h: the order-8 branch moved"
    j = milk.find("} else if (order == 16)", i)
    assert j >= 0, "milk.h: the order-8 branch has no end"
    block = milk[i:j]
    cu = src["cu"]
    literals = set(re.findall(r"[-+]?\d+\.\d+e[-+]\d+", block, flags=re.I))
    # alpha0 plus the eight alpha and eight theta components, in whatever sign
    # form the file writes them.
    missing = [
        lit for lit in literals
        if lit.lstrip("+-") not in cu.replace("+", "").replace("-", "")
        and lit not in cu
    ]
    assert not missing, \
        f"CRAM order-8 literals in milk.h are absent from the .cu: {sorted(missing)[:4]}"
    for scalar in ("64", "1.0e-13", "1.0e-28", "1.0e-30"):
        assert scalar in cu, f"CRAM control constant {scalar} missing from the .cu"
    assert "kMatrixSgn = -1.0" in cu, "order 8 has matrix_sgn = -1"
    assert "kPoleCount = 4" in cu, "order 8 has four poles"


def r_zero_compression(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "if (value == 0.0) continue;" in cu, \
        "the kernel must re-run the host's per-node zero test on the mined pattern"
    assert "minePattern" in cu, \
        "the sparsity pattern must be mined once, not assumed"


def r_divdc3(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "cuCdiv" not in cu and "cuComplex" not in cu, \
        "CUDA's complex division is not libstdc++'s; the host algorithm must be transcribed"
    body = region(cu, "void cdiv(", "\n}\n", "cdiv")
    for fn in ("logb(", "scalbn(", "fmax(", "copysign("):
        assert fn in body, f"cdiv is not __divdc3: {fn} missing"


def r_four_slots(src: dict[str, str]) -> None:
    xs = strip_comments(src["xsset"])
    allowed = {"XSAF", "XSFF", "XS2N", "XS3N"}
    for fn, start, end in (
        ("BuildTransitionMatrix", "void XSSet::BuildTransitionMatrix(", "\n}\n"),
        ("ComputeXeEquilibrium", "static XeEquilibriumImage ComputeXeEquilibrium(", "\n}\n"),
    ):
        body = region(xs, start, end, fn)
        used = set(re.findall(r"N_XS_SCALAR\s*\+\s*(\w+)", body))
        extra = used - allowed
        assert not extra, (
            f"{fn} now reads condensed slot(s) {sorted(extra)}; the device uploads only "
            f"{sorted(allowed)}, so the arm would be reading zeros"
        )
    cu = strip_comments(src["cu"])
    assert "kSlot[4]  = {kXSAF, kXSFF, kXS2N, kXS3N}" in cu, \
        "the uploaded slot list must be exactly the four the host reads"


def r_status_before_d2h(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    for entry in ("bool CramBackend::predictor(", "bool CramBackend::corrector("):
        i = cu.find(entry)
        assert i >= 0, f"{entry} vanished"
        j = cu.find("\n}\n", i)
        body = cu[i:j]
        check = body.find("if (s.h_stats[0] != 0)")
        # WP21-D repaired this needle.  It used to spell the download as
        # `cudaMemcpyDeviceToHost, s.stream);` on one line, which stopped
        # being the text when WP13.1 routed the copy through
        # rasbery::xfer::memcpyAsync and the argument list wrapped -- so the
        # rule had been reporting "the iden download moved" against a file
        # where it had not moved at all.  The anchor is now the failure
        # label, which is what the ORDER is about and is invariant under
        # reformatting of the call it labels.
        d2h = body.find('return s.fail("D2H iden"')
        assert check >= 0, f"{entry}: the per-node status reduction is gone"
        assert d2h >= 0, f"{entry}: the iden download moved"
        assert check < d2h, (
            f"{entry}: the inventory is downloaded before the status is tested -- a node "
            "that hit a throw condition would publish a half-solved inventory"
        )


def r_stub(src: dict[str, str]) -> None:
    cm = src["cmake"]
    assert "src/CudaCramBackendStub.cpp" in cm, "the no-CUDA build needs the stub"
    assert "src/CudaCramBackend.cu" in cm, "the CUDA build needs the real TU"
    stub = strip_comments(src["stub"])
    assert re.search(r"bool CramBackend::predictor\([^)]*\)\s*\{\s*return false;\s*\}",
                     stub, re.S), "the stub predictor must return false"
    assert re.search(r"bool CramBackend::corrector\([^)]*\)\s*\{\s*return false;\s*\}",
                     stub, re.S), "the stub corrector must return false"


def _declared_members(hdr: str) -> list[str]:
    """Every member CramBackend declares OUT OF LINE, mined from the header.

    Mined and not hand-listed, because a hand-kept list is exactly what was
    missing.  A declaration with an inline body does not match (after its `)`
    comes a `{`, not a `;`), and neither does a `= delete` (the `= delete` sits
    between the `)` and the `;`), which is right in both cases: those need no
    definition in the stub.
    """
    body = region(hdr, "class CramBackend {", "\n};", "CudaCramBackend.h")
    body = strip_comments(body).split("private:")[0]
    names: list[str] = []
    for m in re.finditer(r"(\w+)\s*\([^;{()]*\)\s*(?:const\s*)?;", body, re.S):
        name = m.group(1)
        # The constructor and destructor share the class name and are defined in
        # the stub anyway; skipping them keeps the message about the accessors.
        if name == "CramBackend" or name in names:
            continue
        names.append(name)
    return names


def r_stub_symbol_complete(src: dict[str, str]) -> None:
    names = _declared_members(src["hdr"])
    assert len(names) >= 15, (
        f"the header member scan found only {len(names)} declarations -- the SCAN is "
        "broken, not the class, and a broken scan reports PASS")
    for required in ("micxD2dBytes", "kernelVariant", "lanesPerNode", "launches",
                     "launchUsMean"):
        assert required in names, (
            f"the header member scan missed {required}; the scan is broken")
    stub = strip_comments(src["stub"])
    missing = [n for n in names
               if not re.search(r"\bCramBackend::" + re.escape(n) + r"\s*\(", stub)]
    assert not missing, (
        "CudaCramBackendStub.cpp does not define " + ", ".join(missing) + ".  Its own "
        "first line promises 'same symbols as CudaCramBackend.cu', Driver.h calls the "
        "receipt accessors with no #ifdef RASBERY_HAS_CUDA, and CMakeLists.txt drops the "
        "stub only when RASBERY_ENABLE_CUDA is ON -- so a -DRASBERY_ENABLE_CUDA=OFF link "
        "fails with one undefined reference per name above")


def r_arm_knob(src: dict[str, str]) -> None:
    raw = src["driver"]
    code = strip_comments(raw)
    m = re.search(r"kArmEnv\[\]\s*=\s*\{(.*?)\};", code, re.S)
    assert m, "trajectory::kArmEnv vanished"
    assert "RASBERY_GPU_CRAM" in m.group(1), (
        "RASBERY_GPU_CRAM is NOT in trajectory::kArmEnv.  Depletion output is the next "
        "statepoint's isotope inventory, so the knob moves the trajectory; leaving it out "
        "lets two runs with different physics compare as the same arm."
    )
    # And the reason has to live where the list does, not only here.
    anchor = raw.find("kArmEnv[]")
    assert anchor > 0
    preamble = raw[max(0, anchor - 4000):anchor]
    assert "RASBERY_GPU_CRAM is deliberately PRESENT" in preamble, \
        "kArmEnv does not explain why RASBERY_GPU_CRAM is in the list"


def r_receipt(src: dict[str, str]) -> None:
    code = strip_comments(src["driver"])
    assert "[RASBERY][CRAM_GPU]" in code, "the arm must publish a receipt"
    for field in ('\\"statepoints\\":', '\\"nodes\\":', '\\"device\\":',
                  '\\"host_fallbacks\\":', '\\"gs_iters_mean\\":', '\\"wall_ms\\":'):
        assert field in code, f"the [RASBERY][CRAM_GPU] receipt is missing {field}"


def r_declines(src: dict[str, str]) -> None:
    xs = strip_comments(src["xsset"])
    assert "if (substeps != 1) return false;" in xs, \
        "RASBERY_PC_SUBSTEPS > 1 is a second device path nobody measured; it must decline"
    assert "if (_cram_bos_token == 0) return false;" in xs, \
        "a corrector must not run on a BOS snapshot no device predictor produced"
    cu = strip_comments(src["cu"])
    assert "v.bos_token != s.bos_token" in cu, \
        "the backend must check the corrector's BOS token, not trust the caller"


RULES = [
    ("default-off", r_default_off, "cu",
     ('getenv("RASBERY_GPU_CRAM")', 'getenv("RASBERY_ALWAYS_ON")')),
    ("fail-open", r_fail_open, "cu",
     ('return s.decline("ng != 2 (the condensation body is 2-group)");',
      'throw std::runtime_error("ng != 2");')),
    ("no-process-state", r_no_process_state, "cu",
     ("unsigned long long bos_token = 0;", "static unsigned long long bos_token = 0;")),
    # The control now breaks the option list at its DEFINITION rather than at the
    # use site: since 2c04a6e the use site names a variable, so emptying the
    # variable is exactly the way this contract can be lost silently.
    ("fmad-false", r_fmad, "cmake",
     ('set(RASBERY_BITEXACT_CUDA_OPTS "--fmad=false")',
      'set(RASBERY_BITEXACT_CUDA_OPTS "")')),
    ("class-n1", r_class_is_n1, "hdr",
     ("CLASS N1, NOT B0", "CLASS B0")),
    ("pole-set", r_pole_set, "cu",
     ("+1.83174069610856716e+00", "+1.83174069610856717e+00")),
    ("zero-compression", r_zero_compression, "cu",
     ("if (value == 0.0) continue;", "if (false) continue;")),
    ("divdc3", r_divdc3, "cu",
     ("const double logbw  = logb(fmax(fabs(c), fabs(d)));",
      "const double logbw  = 0.0;")),
    ("four-slots", r_four_slots, "xsset",
     ("double xsaf_val = cond[p * N_XS_SCALAR + XSAF];",
      "double xsaf_val = cond[p * N_XS_SCALAR + XSTF];")),
    ("status-before-d2h", r_status_before_d2h, "cu",
     ('    if (s.h_stats[0] != 0) {\n        s.status_text = "declined: node status " + std::to_string(s.h_stats[0]) +\n'
      '                        " (zero diagonal / no convergence / non-finite)";\n        return false;\n    }\n\n'
      '    // Rows [first, niso) are contiguous',
      "    // Rows [first, niso) are contiguous")),
    ("stub", r_stub, "stub",
     ("                            unsigned long long*) {\n    return false;\n}",
      "                            unsigned long long*) {\n    return true;\n}")),
    # The control renames ONE definition rather than deleting it, so the file it
    # produces is the shape the defect actually had: a stub that still compiles
    # on its own and only fails at link, against a header that still declares
    # the name.
    ("stub-symbol-complete", r_stub_symbol_complete, "stub",
     ("unsigned long long CramBackend::bosReuses() const { return 0; }",
      "unsigned long long CramBackend::bosReusesRenamed() const { return 0; }")),
    ("arm-knob", r_arm_knob, "driver",
     ('    "RASBERY_GPU_CRAM",\n', "")),
    ("receipt", r_receipt, "driver",
     ("[RASBERY][CRAM_GPU]", "[RASBERY][CRAM_TIMING]")),
    ("declines", r_declines, "xsset",
     ("if (substeps != 1) return false;", "if (substeps < 1) return false;")),
]


def main() -> int:
    failures: list[str] = []
    try:
        src = {k: read(v) for k, v in FILES.items()}
    except AssertionError as exc:
        print(f"CRAM GPU contract: FAIL {exc}")
        return 1

    for name, rule, _target, _control in RULES:
        try:
            rule(src)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")

    # Negative control: every rule must REJECT its own break.  A presence check
    # that no longer discriminates is worse than no check -- it reports PASS.
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
        print("CRAM GPU contract: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"CRAM GPU contract: PASS ({len(RULES)} rules, each with a negative control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
