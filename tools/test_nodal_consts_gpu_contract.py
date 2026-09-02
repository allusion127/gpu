#!/usr/bin/env python3
"""Device nodal-constants arm contract -- WP23 item 3.

Thirteen properties.  This arm is unusual in the tree in that its inexactness is
MEASURED rather than feared -- test/nodal_constant_exp_probe.cu found CUDA's exp
differing from glibc's by exactly 1 ulp on 3.34 % of the arguments this body
evaluates -- so most of what has to be held here is about not letting that known,
priced deviation be confused with an unknown one.

  1. DEFAULT OFF.  RASBERY_GPU_NODAL_CONSTS gates the arm and absent means off.

  2. FEATURE-OFF IS THE OLD PATH.  With the arm off, solveNodal still uploads the
     nine coefficient arrays through the branch that shipped.  A refactor that
     dropped it turns "no device" into "no SENM coefficients".

  3. THE HOST SWEEP STILL RUNS.  Nodal::drive calls updateConstantsIfMoved BEFORE
     solveNodal on both arms.  This arm removes UPLOADS, not host arithmetic --
     which is what keeps the CPU fallback correct and what makes the ULP
     self-check possible at all.  An arm that also skipped the sweep would be a
     different arm with a different obligation, and it must not be able to slip
     in under this flag.

  4. ONE BODY.  The kernel calls nodal::nodalConstantCoefficients and spells no
     formula of its own.  A `kp2` or a `sinhkp` in the .cu is a duplicated
     formula, which is the drift src/CudaNodalConstantKernel.h already forbids
     for the arena kernel.

  5. IT IS NOT THE ARENA LAUNCHER, AND THE REASON IS WRITTEN DOWN.
     CudaOuterGraph.cu records that enqueueNodalUpdateConstant is inert because
     its inputs are never written on the production path AND because its packing
     has kNcDiagDI = 7 / kNcDiagD = 8 where this backend's reader has diagD = 7 /
     diagDI = 8.  Binding one to the other swaps D and 1/D everywhere, finitely
     and plausibly.  So the arm writes THIS backend's packing.

  6. THE STORE ORDER IS THE READER'S ORDER: eta1, eta2, m260, m251, m253, m262,
     m264, diagD, diagDI.  See 5.

  7. xsdf TRAVELS AS A PARAMETER, not a NodalView field.  Nothing on the nodal
     critical path reads xsdf, so growing the view would have put a new field
     through nodalWideShell, the FP32 narrowing census and three replay tools for
     one optional arm.

  8. THE BUILD IS DEFERRED PAST THE xs UPLOAD GATE.  The body reads xsrf and xsdf
     out of the resident macroscopic block, whose own upload gate is below the
     constants gate; a build issued at the constants gate would read the previous
     statepoint's cross sections whenever the state generation had moved.

  9. THE CLASS IS DECLARED N1 BY MEASUREMENT, with the probe named.

 10. THE SELF-CHECK EXISTS AND IS BOUNDED.  It samples a contiguous run per
     array, not the whole block -- downloading the block would move exactly the
     bytes the arm exists to stop moving -- and it reports max_ulp, the array
     that carried it, and the count over 1 ulp.  An unsampled run reports -1, not
     0: "no sample" is not the same fact as "0 ulp".

 11. THE GPU_FULL SEAM EXISTS for Subsystem::NodalConsts, so a decline is not
     reported as "the nodal arm fell back" while the nine uploads quietly return.

 12. THE ARM KNOB IS IN trajectory::kArmEnv, and the STUB keeps CPU-only builds
     compiling.

Every rule runs against a deliberately broken copy of the same text as a
negative control.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "cu": "src/CudaXsReconBackend.cu",
    "hdr": "src/CudaXsReconBackend.h",
    "stub": "src/CudaXsReconBackendStub.cpp",
    "receipt": "src/NodalConstsReceipt.h",
    "nodal": "src/Nodal.cpp",
    "body": "src/NodalConstantKernel.h",
    "graph": "src/CudaOuterGraph.cu",
    "driver": "src/Driver.h",
    "full": "src/GpuFullContract.h",
    "xsset_h": "src/XSSet.h",
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


def r_default_off(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    body = region(cu, "bool rasberyGpuNodalConstsEnabled()", "\n}",
                  "rasberyGpuNodalConstsEnabled")
    assert 'envFlagEnabled("RASBERY_GPU_NODAL_CONSTS")' in body, \
        "the arm must be gated on RASBERY_GPU_NODAL_CONSTS and default off"


def r_feature_off_is_old_path(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "if (nc_arm) {" in cu and "} else {" in cu, \
        "the nine uploads must stay behind the arm's own else-branch"
    assert 'xfer::memcpyAsync("CudaXsReconBackend.cu:solveNodal", "consts"' in cu, \
        "the host upload of the nine coefficient arrays disappeared entirely"
    assert "g_nodal_const_uploads.fetch_add(9" in cu, \
        "the WP15 census of the nine uploads was removed with them"


def r_host_sweep_still_runs(src: dict[str, str]) -> None:
    nodal = strip_comments(src["nodal"])
    body = region(nodal, "backend->solveNodal(", "return false;", "TryDriveGpu")
    j = nodal.find("backend->solveNodal(")
    i = nodal.rfind("updateConstantsIfMoved();", 0, j)
    assert 0 <= i < j, \
        ("Nodal must still run updateConstantsIfMoved() BEFORE solveNodal: this arm "
         "removes the nine uploads, not the host arithmetic, and the CPU fallback "
         "and the ULP self-check both depend on the host arrays being correct")
    assert "xs.xsdfData()" in body, "solveNodal is not handed the xsdf column"


def r_one_body(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    kern = region(cu, "__global__ void kNodalConstsDevice(", "\n}",
                  "kNodalConstsDevice")
    assert "ndl::nodalConstantCoefficients(r, d, h, forms)" in kern, \
        "the kernel must call the shared body"
    for formula in ("kp2", "sinhkp", "coshkp", "std::sqrt", "exp("):
        assert formula not in kern, \
            f"the kernel restates the coefficient algebra ({formula!r})"


def r_not_the_arena_launcher(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "enqueueNodalUpdateConstant" not in cu, \
        ("this backend must NOT bind the arena launcher: CudaOuterGraph.cu records "
         "that its packing has kNcDiagDI = 7 / kNcDiagD = 8 where this backend's "
         "reader has diagD = 7 / diagDI = 8, so binding one to the other swaps D "
         "and 1/D on every node and direction")
    graph = src["graph"]
    assert "kNcDiagDI = 7" in graph and "kNcDiagD = 8" in graph, \
        ("the note that records WHY the arena launcher is not the vehicle has moved "
         "or been deleted; without it the next reader wires it up")


def r_store_order(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    kern = region(cu, "__global__ void kNodalConstsDevice(", "\n}",
                  "kNodalConstsDevice")
    order = re.findall(r"dst\[(\d) \* ndg \+ idx\] = static_cast<DstT>\(c\.(\w+)\);", kern)
    got = [name for _idx, name in order]
    want = ["eta1", "eta2", "m260", "m251", "m253", "m262", "m264", "diagD", "diagDI"]
    assert got == want, \
        (f"the kernel stores {got}; the reader's packing is {want} -- a swapped "
         "diagD/diagDI pair is finite, plausible and wrong everywhere")


def r_xsdf_is_a_parameter(src: dict[str, str]) -> None:
    hdr = strip_comments(src["hdr"])
    assert "const double* host_xsdf = nullptr);" in hdr, \
        "xsdf must be a defaulted parameter of solveNodal"
    assert "xsdfData()" in src["xsset_h"], \
        "XSSet must expose the xsdf column for it"
    nodal_kernel = read("src/NodalKernel.h")
    view = region(nodal_kernel, "struct NodalViewT {", "\n};", "NodalViewT")
    assert "xsdf" not in view, \
        ("xsdf must NOT be a NodalView field: nothing on the nodal critical path "
         "reads it, and adding one puts a new field through nodalWideShell, the "
         "FP32 narrowing census and three replay tools for one optional arm")


def r_deferred_past_xs_gate(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    solve = region(cu, "bool XsReconBackend::solveNodal(", "const bool nnarrow =",
                   "solveNodal")
    i_gate = solve.find("nc_pending = true;")
    i_xs   = solve.find('d.upload("nodal xsrf"')
    i_call = solve.find("d.nodalConstsOnDevice(")
    assert 0 <= i_gate < i_xs < i_call, \
        ("the device build must be issued AFTER the xs upload gate: the body reads "
         "xsrf/xsdf out of the resident macroscopic block, so a build at the "
         "constants gate reads the previous statepoint's cross sections whenever "
         "the state generation had moved")


def r_class_declared(src: dict[str, str]) -> None:
    note = region(src["receipt"], "kNodalConstsPolicyNote", "section 6)\";",
                  "kNodalConstsPolicyNote")
    assert "CLASS N1 BY MEASUREMENT" in note, \
        "the receipt must declare the class and say it was measured"
    assert "nodal_constant_exp_probe" in note, "the probe that measured it is not named"
    assert "3.34%" in note, "the measured deviation rate is not quoted"
    assert "Gate A/B" in note, "the gate the arm is priced with is not named"


def r_self_check(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    chk = region(cu, "const bool sample = (nc_builds == 1)", "return true;\n    }",
                 "self-check")
    assert "nodalconsts::ulpDistance(a, b)" in chk, "the self-check does not measure ULP"
    assert "run * elem" in chk, \
        "the self-check must copy a BOUNDED run, not the whole array"
    assert "cudaMemcpyDeviceToHost" in chk, "the self-check never reads the device back"
    rc = src["receipt"]
    for field in (r'\"max_ulp\"', r'\"max_ulp_array\"', r'\"over_1ulp\"',
                  r'\"self_checks\"', r'\"checked_elems\"'):
        assert field in rc, f"the receipt is missing the {field} field"
    assert 'os << "-1,\\"max_ulp_array\\":-1";' in rc, \
        ('an unsampled run must report -1, not 0: "no sample" is a different fact '
         'from "0 ulp"')


def r_gpu_full_seam(src: dict[str, str]) -> None:
    full = strip_comments(src["full"])
    assert "NodalConsts," in full, "Subsystem::NodalConsts is not declared"
    assert 'return "nodal_consts";' in full, "the subsystem has no receipt name"
    nodal = strip_comments(src["nodal"])
    assert "rasberyGpuNodalConstsEnabled(), NodalConsts," in nodal, \
        ("the decline seam must be guarded for this subsystem, or it is reported as "
         "'the nodal arm fell back' while the nine uploads quietly return")


def r_arm_env_and_stub(src: dict[str, str]) -> None:
    arm = region(src["driver"], "inline constexpr const char* kArmEnv[] = {", "};",
                 "kArmEnv")
    assert '"RASBERY_GPU_NODAL_CONSTS"' in arm, \
        "the knob is not in trajectory::kArmEnv; it is an N1 arm and moves the trajectory"
    stub = strip_comments(src["stub"])
    assert "bool     rasberyGpuNodalConstsEnabled() { return false; }" in stub, \
        "the CPU-only build is missing the flag reader"
    assert "unsigned long long, unsigned long long,\n                                const double*) {" in stub, \
        "the stub's solveNodal signature did not follow the header"


def r_receipt_printed(src: dict[str, str]) -> None:
    drv = strip_comments(src["driver"])
    assert "[RASBERY][NODAL_CONSTS]" in drv, "the receipt is never printed"
    assert "nodalConstsReceiptWanted()" in drv, \
        "the receipt must print only when the arm was asked for"


RULES = [
    ("default-off", r_default_off, "cu",
     ('envFlagEnabled("RASBERY_GPU_NODAL_CONSTS")', "true")),
    ("feature-off-is-old-path", r_feature_off_is_old_path, "cu",
     ("g_nodal_const_uploads.fetch_add(9", "g_nodal_const_uploads.fetch_add(0")),
    ("host-sweep-still-runs", r_host_sweep_still_runs, "nodal",
     ("xs.hoststateGeneration(), xs.xsdfData()", "xs.hoststateGeneration()")),
    ("one-body", r_one_body, "cu",
     ("ndl::nodalConstantCoefficients(r, d, h, forms)",
      "ndl::NodalConstantCoefficients{r * exp(d), 0, 0, 0, 0, 0, 0, 0, 0}")),
    ("not-the-arena-launcher", r_not_the_arena_launcher, "cu",
     ("kNodalConstsDevice<double><<<g, B, 0, stream>>>",
      "enqueueNodalUpdateConstant<<<g, B, 0, stream>>>")),
    ("store-order", r_store_order, "cu",
     ("dst[7 * ndg + idx] = static_cast<DstT>(c.diagD);",
      "dst[7 * ndg + idx] = static_cast<DstT>(c.diagDI);")),
    ("xsdf-is-a-parameter", r_xsdf_is_a_parameter, "hdr",
     ("const double* host_xsdf = nullptr);", "const double* host_xsdf);")),
    ("deferred-past-xs-gate", r_deferred_past_xs_gate, "cu",
     ("            nc_pending = true;", "            ;")),
    ("class-declared", r_class_declared, "receipt",
     ("CLASS N1 BY MEASUREMENT", "class B0 by construction")),
    ("self-check", r_self_check, "cu",
     ("nodalconsts::ulpDistance(a, b)", "0ULL * (unsigned long long)(a + b)")),
    ("gpu-full-seam", r_gpu_full_seam, "full",
     ("    NodalConsts,\n", "\n")),
    ("arm-env-and-stub", r_arm_env_and_stub, "driver",
     ('    "RASBERY_GPU_NODAL_CONSTS",\n', "")),
    ("receipt-printed", r_receipt_printed, "driver",
     ('rasbery::nodalconsts::nodalConstsReceiptWanted()', "false")),
]


def main() -> int:
    failures: list[str] = []
    try:
        src = {k: read(v) for k, v in FILES.items()}
    except AssertionError as exc:
        print(f"Nodal consts GPU contract: FAIL {exc}")
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
        print("Nodal consts GPU contract: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"Nodal consts GPU contract: PASS ({len(RULES)} rules, each with a negative control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
