#!/usr/bin/env python3
"""GPU thermal-hydraulics contract -- WP22, commit 1.

Fifteen properties.  Not one of them is visible to a numerical comparison of two
runs, which is exactly why they are asserted here: a passing kngr_238 A/B would
keep passing right up until the day one of them mattered.

  1. DEFAULT OFF, AND THE OFF PATH IS THE OLD PATH.  The backend is reached only
     through XSSet::TryUpdateTHGpu, whose false return must be followed by the
     untouched host SolveTH body.  A refactor that dropped the fallback would
     turn "no CUDA device" into "no T/H feedback", and the temperatures are
     every macroscopic cross section's input.

  2. FAIL OPEN, NEVER THROW.  Every failure in the .cu resolves to a `return
     false`.  A throw out of a kernel is not a thing, and a throw out of the
     backend would take down a 64-deck batch for one slot.

  3. NO PROCESS-WIDE MUTABLE STATE.  The backend, its stream and every device
     buffer hang off an XSSet, which belongs to one Driver.  A mutable `static`
     in the .cu is the slot-0 bug class: deck 7 driving deck 0's buffers,
     correct-looking, wrong.

  4. --fmad=false ON THE TU.  Every multiply-add in ThKernel.h reads its form
     from a MINED mask; letting nvcc fuse on its own would make the mask a lie
     and the B0 target unreachable for a reason no receipt could name.

  5. ONE BODY, TWO COMPILERS.  The .cu must not restate the arithmetic: it
     includes ThKernel.h and calls it.  A transcription is a second opinion that
     will drift, and there is nothing here that forces one (unlike CRAM, whose
     complex division cannot be handed to nvcc).

  6. THE ORDER-DEPENDENT FOLDS STAY SERIAL.  `total_power` over nodes and
     `total_area` over channels run in ONE lane in ascending index order.  A
     tree reduction is a different double, and the norm it produces multiplies
     EVERY node power -- so the difference would not stay in one scalar.

  7. THE CHANNEL SWEEP IS LANE-PER-CHANNEL.  The axial enthalpy carry is serial
     by construction; the parallelism is the radial index, which is what the
     host loop's outer index already was.

  8. THE FORM MASK IS MINED, NOT BAKED.  The production binary derives the
     host's mask against a verbatim quotation and only falls back to the build
     default when the derivation fails -- and says so.

  9. THE QUOTATION KEEPS ITS OWN TRANSLATION UNIT.  ThReference.cpp must never
     include ThKernel.h: with both in one TU gcc common-subexpressions across
     them and changes the QUOTATION's contraction, which is the reference the
     mining scores against.

 10. THE CLASS IS DECLARED, AND IT IS N1 UNTIL A HOST MEASURES B0.

 11. THE STUB KEEPS CPU-ONLY BUILDS COMPILING.

 12. RASBERY_GPU_TH *IS* AN ARM KNOB, AND SAYS WHY.  T/H output is this
     statepoint's cross-section input, so the knob moves the trajectory and must
     be in trajectory::kArmEnv with the reason written where the list is.
     RASBERY_TH_FORMS rides beside it because it selects the rounding.

 13. THE RECEIPT THE PLAN ASKS FOR, with the fields a G0 check reads.

 14. THE UNSUPPORTED SHAPES DECLINE rather than clamp: a kernel that quietly ran
     a fraction of the core would produce a plausible temperature field.

 15. THE GPU_FULL SEAM EXISTS.  The fallback is guarded for Subsystem::Th, so
     under RASBERY_GPU_FULL an arm that refuses every update fails the case
     instead of looking exactly like an arm that was never set.

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
    "cu": "src/CudaThBackend.cu",
    "hdr": "src/CudaThBackend.h",
    "stub": "src/CudaThBackendStub.cpp",
    "kernel": "src/ThKernel.h",
    "ref": "src/ThReference.cpp",
    "mine": "src/ThFormMine.h",
    "miner": "src/ThFormMiner.cpp",
    "mask": "src/ThFormMask.h",
    "receipt": "src/ThGpuReceipt.h",
    "xsset_h": "src/XSSet.h",
    "xsset": "src/XSSet.cpp",
    "driver": "src/Driver.h",
    "contract": "src/GpuFullContract.h",
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
# Rules
# ---------------------------------------------------------------------------


def r_default_off(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert 'getenv("RASBERY_GPU_TH")' in cu, \
        "the arm must be gated on RASBERY_GPU_TH and default off"
    xs = strip_comments(src["xsset"])
    assert "if (TryUpdateTHGpu(power_rate, gpu_delta_dop)) {" in xs, \
        "XSSet::UpdateTH must try the device and fall through on false"
    assert "++_th_host_fallbacks;" in xs, \
        "a decline must be counted, or the receipt cannot say the arm never ran"
    # The host body must still be there, below the seam.
    assert "SolveTH(node_power.data(), _burn.data(), power_rate);" in xs, \
        "the host SolveTH call vanished; there is nothing left to fall back to"


def r_fail_open(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "throw" not in cu, \
        "the backend must not throw: a statepoint-boundary throw kills a 64-deck batch"
    assert "bool decline(const char* why)" in cu, \
        "an unsupported deck must decline, not assert"
    assert cu.count("return s.fail(") + cu.count("return fail(") >= 8, \
        "CUDA errors must resolve to fail(), which returns false"


def r_no_process_state(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    for m in re.finditer(r"\bstatic\b", cu):
        line = cu[cu.rfind("\n", 0, m.start()) + 1 : cu.find("\n", m.start())]
        assert "static_cast" in line or "static_assert" in line, \
            f"mutable process-wide state in the .cu: {line.strip()!r}"


def r_fmad(src: dict[str, str]) -> None:
    cm = src["cmake"]
    m = re.search(
        r"set_source_files_properties\(\s*\"\$\{CMAKE_CURRENT_SOURCE_DIR\}/src/"
        r"CudaThBackend\.cu\"(.{0,200}?)\)",
        cm, re.S)
    assert m, ("CudaThBackend.cu has no set_source_files_properties entry -- nvcc would "
               "be free to contract multiply-adds the mined mask says the host did not")
    var = re.search(r"set\(RASBERY_BITEXACT_CUDA_OPTS\s+\"([^\"]*)\"\s*\)", cm)
    opts = m.group(1).replace("${RASBERY_BITEXACT_CUDA_OPTS}",
                              var.group(1) if var else "")
    assert "--fmad=false" in opts, \
        "CudaThBackend.cu must be compiled with --fmad=false"


def r_one_body(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert '#include "CudaThBackend.h"' in cu, "the .cu must include its own header"
    for call in ("th::thNodePower(", "th::thChannelSweep(", "th::thRelaxNode(",
                 "th::thTotalPowerSerial(", "th::thTotalAreaSerial("):
        assert call in cu, f"the .cu does not call the shared body {call!r}"
    # And it must NOT restate the interpolation: a second spelling of
    # milk::Table::Get in this TU is the drift this rule forbids.
    assert "FindLowerIndex" not in cu and "z01 - z00" not in cu, \
        "the .cu restates the table interpolation instead of calling ThKernel.h's"


def r_serial_folds(src: dict[str, str]) -> None:
    kernel = strip_comments(src["kernel"])
    body = region(kernel, "thTotalPowerSerial(const double* node_power, int nxyz)", "\n}",
                  "thTotalPowerSerial")
    assert "for (int lk = 0; lk < nxyz; ++lk)" in body and "total += node_power[lk];" in body, \
        "the total-power fold is no longer an ascending serial sum"
    cu = strip_comments(src["cu"])
    folds = region(cu, "void kernelFolds(", "\n}", "kernelFolds")
    assert "if (blockIdx.x != 0 || threadIdx.x != 0) return;" in folds, \
        "kernelFolds is not single-lane; a tree reduction of a non-associative fold is a " \
        "different norm, and the norm multiplies every node power"
    assert "<<<1, 1, 0, s.stream>>>" in cu, \
        "kernelFolds is not launched with one block of one thread"


def r_lane_per_channel(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    sweep = region(cu, "void kernelChannelSweep(", "\n}", "kernelChannelSweep")
    assert "const int l = blockIdx.x * blockDim.x + threadIdx.x;" in sweep and \
        "if (l >= v.nxy) return;" in sweep, \
        "the channel sweep is not one lane per radial position"
    kernel = strip_comments(src["kernel"])
    body = region(kernel, "thChannelSweep(const ThView& v, int l", "\n}", "thChannelSweep")
    assert "double            h_cur = v.inlet_h;" in body, \
        "the channel body no longer carries the enthalpy serially from the inlet"


def r_mask_is_mined(src: dict[str, str]) -> None:
    miner = strip_comments(src["miner"])
    assert "mineThFormsOnThisHost" in miner and "mineStable" in miner, \
        "the production binary does not mine its own mask"
    assert "resolveCalibratedFormMask" in miner, \
        "the mask resolution does not go through GpuFormMask.h's calibrated resolver, so " \
        "a mining failure could pass for a measurement"
    assert '"RASBERY_TH_FORMS"' in miner, "the mask has no environment override"
    cu = strip_comments(src["cu"])
    assert "th::thFormMask()" in cu, "the kernels are not launched under the resolved mask"
    assert "kernelNodePower<<<" in cu and "forms)" in cu, \
        "the mask does not reach the kernels as an argument"


def r_reference_tu(src: dict[str, str]) -> None:
    ref = src["ref"]
    assert '#include "ThKernel.h"' not in ref, \
        "ThReference.cpp includes the shipped bodies; with both in one TU gcc " \
        "common-subexpressions across them and the QUOTATION's contraction changes"
    assert "std::fma" not in ref and "thMul" not in ref, \
        "the quotation pins its own rounding; it must be plain +, * and / or it is not a " \
        "record of what the host compiler does"
    mine = src["mine"]
    assert '#include "ThKernel.h"' in mine and '#include "ThReference.h"' in mine, \
        "the mining harness must see both sides"
    # Soundness is residual-based, not pattern-based: a DON'T-CARE site would
    # otherwise fail a mask that is provably right.
    assert "sound = false" in mine and "scoreMask(f, m) != 0" in mine, \
        "mineStable does not report soundness by residual"


def r_class_declared(src: dict[str, str]) -> None:
    hdr = src["hdr"]
    assert "GATE CLASS" in hdr, "the header must declare the gate class"
    assert "Gate A" in hdr and "Gate B" in hdr, \
        "the header must name the gates the class implies"
    assert "REACHABLE IS NOT MEASURED" in hdr, \
        "the header claims B0 without saying it has not been measured"
    receipt = src["receipt"]
    assert "kThGpuPolicyNote" in receipt and "N1" in receipt, \
        "the receipt does not carry the grade a gate script reads"


def r_stub(src: dict[str, str]) -> None:
    cm = src["cmake"]
    assert "src/CudaThBackendStub.cpp" in cm, "the no-CUDA build needs the stub"
    assert "src/CudaThBackend.cu" in cm, "the CUDA build needs the real TU"
    stub = strip_comments(src["stub"])
    assert re.search(r"bool ThBackend::solveTh\([^)]*\)\s*\{\s*return false;\s*\}",
                     stub, re.S), "the stub solveTh must return false"
    assert re.search(r"bool ThBackend::available\(\)\s*const\s*\{\s*return false;\s*\}",
                     stub, re.S), "the stub must report itself unavailable"


def r_arm_knob(src: dict[str, str]) -> None:
    raw = src["driver"]
    code = strip_comments(raw)
    m = re.search(r"kArmEnv\[\]\s*=\s*\{(.*?)\};", code, re.S)
    assert m, "trajectory::kArmEnv vanished"
    for knob in ("RASBERY_GPU_TH", "RASBERY_TH_FORMS"):
        assert knob in m.group(1), (
            f"{knob} is NOT in trajectory::kArmEnv.  T/H output is this statepoint's "
            "cross-section input, so the knob moves the trajectory; leaving it out lets "
            "two runs with different physics compare as the same arm.")
    anchor = raw.find("kArmEnv[]")
    assert anchor > 0
    preamble = raw[max(0, anchor - 5000):anchor]
    assert "RASBERY_GPU_TH is deliberately PRESENT" in preamble, \
        "kArmEnv does not explain why RASBERY_GPU_TH is in the list"


def r_receipt(src: dict[str, str]) -> None:
    code = strip_comments(src["driver"])
    assert "[RASBERY][TH_GPU]" in code, "the arm must publish a receipt"
    fields = strip_comments(src["receipt"])
    for field in ('\\"arm\\":', '\\"channels\\":', '\\"nodes\\":',
                  '\\"host_fallbacks\\":', '\\"forms_mask\\":',
                  '\\"bytes_elided\\":', '\\"wall_ms\\":'):
        assert field in fields, f"the [RASBERY][TH_GPU] receipt is missing {field}"
    # The G0 identity has to be constructible from the receipt itself.
    assert '\\"th_updates\\":' in fields and '\\"device_updates\\":' in fields, \
        "the receipt cannot express th_updates == device_updates + host_fallbacks"
    # And the mask must not be MINED just to print it.
    assert "forms_seen" in fields, \
        "the receipt prints a mask it did not measure; a flag-off run would mine one"


def r_declines(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    for guard in ('return s.decline("nxy * nz != nxyz")',
                  'return s.decline("kbc/kec out of range")',
                  'return s.decline("degenerate shape")'):
        assert guard in cu, f"missing shape refusal: {guard}"
    assert 'decline("non-finite delta_dop")' in cu, \
        "a non-finite convergence metric must decline rather than be published"


def r_gpu_full_seam(src: dict[str, str]) -> None:
    contract = src["contract"]
    assert re.search(r"\bTh,", contract), \
        "gpufull::Subsystem has no Th member; the seam has nothing to name"
    assert 'case Subsystem::Th:     return "th";' in contract, \
        "Subsystem::Th has no name, so the receipt would print `unknown_fallbacks`"
    xs = strip_comments(src["xsset"])
    assert "RASBERY_GPU_FULL_GUARD_IF(th().available(), Th," in xs, \
        "the T/H fallback is not guarded for Subsystem::Th, so under RASBERY_GPU_FULL an " \
        "arm that refused every update looks exactly like an arm that was never set"


def r_scalars_before_arrays(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    body = region(cu, "bool ThBackend::solveTh(", "\n}\n", "solveTh")
    scalars = body.find('"CudaThBackend.cu:solveTh", "scalars"')
    arrays = body.find('s.fail("D2H state"')
    assert scalars >= 0, "solveTh: the scalar download is gone"
    assert arrays >= 0, "solveTh: the array download moved"
    assert scalars < arrays, (
        "solveTh downloads the temperature field before its own diagnostic scalars have "
        "been tested; a field its convergence metric had not cleared would be published")


RULES = [
    ("default-off", r_default_off, "cu",
     ('getenv("RASBERY_GPU_TH")', 'getenv("RASBERY_ALWAYS_ON")')),
    ("fail-open", r_fail_open, "cu",
     ('bool decline(const char* why)', 'bool refuse(const char* why)')),
    ("no-process-state", r_no_process_state, "cu",
     ("    cudaStream_t stream = nullptr;", "    static cudaStream_t stream = nullptr;")),
    ("fmad-false", r_fmad, "cmake",
     ('set(RASBERY_BITEXACT_CUDA_OPTS "--fmad=false")',
      'set(RASBERY_BITEXACT_CUDA_OPTS "")')),
    ("one-body", r_one_body, "cu",
     ("th::thNodePower(v, lk, forms)", "0.0")),
    ("serial-folds", r_serial_folds, "cu",
     ("if (blockIdx.x != 0 || threadIdx.x != 0) return;",
      "if (blockIdx.x != 0) return;")),
    ("lane-per-channel", r_lane_per_channel, "cu",
     ("if (l >= v.nxy) return;", "if (l >= v.nxyz) return;")),
    ("mask-is-mined", r_mask_is_mined, "miner",
     ("resolveCalibratedFormMask", "resolveFormMask")),
    ("reference-tu", r_reference_tu, "ref",
     ('#include "ThReference.h"', '#include "ThReference.h"\n#include "ThKernel.h"')),
    ("class-declared", r_class_declared, "hdr",
     ("REACHABLE IS NOT MEASURED", "MEASURED B0 ON EVERY HOST")),
    ("stub", r_stub, "stub",
     ("                        const thgpu::UpdateView&, double&) {\n    return false;\n}",
      "                        const thgpu::UpdateView&, double&) {\n    return true;\n}")),
    ("arm-knob", r_arm_knob, "driver",
     ('    "RASBERY_GPU_TH",\n', "")),
    ("receipt", r_receipt, "driver",
     ("[RASBERY][TH_GPU]", "[RASBERY][TH_TIMING]")),
    ("declines", r_declines, "cu",
     ('return s.decline("nxy * nz != nxyz")', 'return true')),
    ("gpu-full-seam", r_gpu_full_seam, "xsset",
     ("RASBERY_GPU_FULL_GUARD_IF(th().available(), Th,",
      "RASBERY_GPU_FULL_COUNT_IF(th().available(), Th,")),
    ("scalars-before-arrays", r_scalars_before_arrays, "cu",
     ('"CudaThBackend.cu:solveTh", "scalars"',
      '"CudaThBackend.cu:solveTh", "late_scalars"')),
]


def main() -> int:
    failures: list[str] = []
    try:
        src = {k: read(v) for k, v in FILES.items()}
    except AssertionError as exc:
        print(f"TH GPU contract: FAIL {exc}")
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
        print("TH GPU contract: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"TH GPU contract: PASS ({len(RULES)} rules, each with a negative control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
