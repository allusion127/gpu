#!/usr/bin/env python3
"""Device branch-stream resolver contract -- WP23.

Sixteen properties.  Not one of them is visible to a numerical comparison of two
runs, which is exactly why they are asserted here: a passing kngr_238 A/B would
keep passing right up until the day one of them mattered.

  1. DEFAULT OFF, AND IT IS A SUB-ARM.  RASBERY_GPU_FLATXS_STREAM resolves to
     false unless RASBERY_GPU_FLATXS is also set -- there is no device kernel for
     the stream to feed otherwise, and building a stream on the device to copy it
     back for a host loop is strictly worse than not building it there.

  2. FEATURE-OFF IS THE OLD PATH, at BOTH seams.  XSSet::UpdateFlatXS still calls
     BuildFlatXsStream when the arm is not eligible, and solveFlatXs still
     uploads node_off / node_cnt / the three stream arrays behind `!dev_stream`.
     A refactor that dropped either turns "no device" into "no branch stream",
     and the branch stream is every unrodded node's cross section.

  3. THE STREAM NEVER COMES BACK.  The phase's only DeviceToHost copy is
     `node_cnt`, which carries the refusal ladder.  A D2H of stream_did / _x /
     _scale would mean the arm had grown the round trip it exists to remove.

  4. THE REFUSAL LADDER REFUSES BY NAME.  An unimplemented coordinate form is
     declined with formName() in the message and a counted host fallback -- not
     approximated, not defaulted to plain density.  That is what makes the arm
     incremental AND honest rather than only incremental.

  5. ONE BODY, TWO COMPILERS.  The .cu must not restate the resolver: it
     includes FlatXsStreamKernel.h and calls flatxsStreamResolveNode.  A
     transcription is a second opinion that will drift.

  6. ONE HOST SPELLING TOO.  BuildFlatXsStream's per-node body is
     XSSet::ResolveNodeApplications, and the loop calls it.  The device arm needs
     the same resolution for anything it refuses, and two spellings would be two
     answers on exactly the nodes nobody would think to check.

  7. NODE INDEPENDENCE IS THE PARALLELISM, AND THE PACKING PROVES IT.  Node `i`
     writes `[i*stride, i*stride+cnt)` and `node_off[i] = i*stride`; the kernel
     holds no atomic, no shared memory and no reduction.  A dense pack would need
     a scan or an atomic bump and BOTH change the order a node's entries land in,
     which is the one thing the CTA kernel's determinism contract forbids.

  8. THE COORDINATE ENUMERATORS ARE HELD TO CHIFFON'S.  FlatXsStreamKernel.h
     restates them as plain ints so nvcc never parses Model.h (HighFive, HDF5);
     XSSet.cpp is the one TU that sees both and static_asserts every one.  A
     drift applies a fitted coefficient to the wrong coordinate.

  9. THE CLASS IS DECLARED, IT IS N1, AND IT CARRIES BOTH REASONS.  The
     contraction mask is UNMINED (reason a) and seven forms call libm (reason b).
     Collapsing them into one grade hides which fix is owed.

 10. THE libm FORM LIST IS THE SEVEN, and nothing else.  formUsesLibm() is what
     the receipt's `libm_form_hit` reads, so an eighth form quietly added to it
     would understate the arm and a missing one would overstate it.

 11. THE RECEIPT THE PLAN ASKS FOR, with the fields a G0 check reads.

 12. THE GPU_FULL SEAM EXISTS for Subsystem::FlatXsStream, so under
     RASBERY_GPU_FULL an arm that refuses every call fails the case instead of
     looking exactly like an arm that was never set.

 13. THE ARM KNOB IS IN trajectory::kArmEnv.  The stream is every unrodded node's
     branch and history coordinate; a bit that moves here moves the answer.

 14. --fmad=false ON THE TU, so nvcc cannot fuse what the (unmined) mask did not
     ask for and the residual stays attributable.

 15. THE STUB KEEPS CPU-ONLY BUILDS COMPILING.

 16. THE REQUEST IS A PARAMETER OF solveFlatXs, NOT A SEPARATE ENTRY POINT.  The
     phase reads the reference micx block, the coefficient tables, iden and the
     three per-node coordinate columns -- all of which solveFlatXs itself is what
     makes resident -- so a public buildFlatXsStream() called before it would run
     against null device pointers on the first call of a run.  This one cost a
     rewrite to notice, which is why it is pinned.

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
    "receipt": "src/FlatXsStreamReceipt.h",
    "cu": "src/CudaXsReconBackend.cu",
    "hdr": "src/CudaXsReconBackend.h",
    "stub": "src/CudaXsReconBackendStub.cpp",
    "xsset": "src/XSSet.cpp",
    "xsset_h": "src/XSSet.h",
    "driver": "src/Driver.h",
    "full": "src/GpuFullContract.h",
    "cmake": "CMakeLists.txt",
    "model": "include/chiffon/Model.h",
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


def r_default_off_subarm(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    body = region(cu, "bool rasberyGpuFlatXsStreamEnabled()", "\n}",
                  "rasberyGpuFlatXsStreamEnabled")
    assert 'envFlagEnabled("RASBERY_GPU_FLATXS_STREAM")' in body, \
        "the arm must be gated on RASBERY_GPU_FLATXS_STREAM and default off"
    assert "rasberyGpuFlatXsEnabled()" in body, \
        ("the stream arm must be a SUB-ARM of RASBERY_GPU_FLATXS: with the flat-XS "
         "device kernel off there is nothing for the stream to feed")


def r_feature_off_is_old_path(src: dict[str, str]) -> None:
    xs = strip_comments(src["xsset"])
    assert "if (!want_stream) {" in xs and xs.count("BuildFlatXsStream(unrodded);") == 2, \
        ("UpdateFlatXS must still call BuildFlatXsStream on BOTH host paths -- the "
         "deck the arm cannot serve, and the call whose device phase refused")
    assert "void XSSet::BuildFlatXsStream(" in xs, \
        "the host stream builder vanished; there is nothing left to fall back to"
    cu = strip_comments(src["cu"])
    assert "const bool dev_stream = stream != nullptr && stream->stride > 0;" in cu, \
        "solveFlatXs must resolve the arm into one bool"
    assert "if (!dev_stream) {" in cu, \
        ("the node_off / node_cnt / stream_* uploads must stay behind `!dev_stream`; "
         "without the guard the feature-off path is not the old path")
    for leaf in ('"node_off"', '"node_cnt"', '"stream_did"', '"stream_x"',
                 '"stream_scale"'):
        assert leaf in cu, f"the host upload of {leaf} disappeared entirely"


def r_no_stream_d2h(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    phase = region(cu, "bool buildStreamOnDevice(", "\n    }\n", "buildStreamOnDevice")
    d2h = re.findall(r"cudaMemcpyDeviceToHost", phase)
    assert len(d2h) == 1, \
        (f"the stream phase has {len(d2h)} DeviceToHost copies; exactly one is "
         "allowed and it is the node_cnt read-back that carries the refusal ladder")
    assert '"node_cnt"' in phase, "the one D2H must be node_cnt"
    for leaf in ("stream_did", "stream_x", "stream_scale"):
        assert f'"{leaf}", ' not in phase or "cudaMemcpyHostToDevice" in phase, \
            f"{leaf} must never be downloaded by the stream phase"


def r_refusal_by_name(src: dict[str, str]) -> None:
    k = strip_comments(src["kernel"])
    for token in ("formImplemented", "formName", "refusalName", "encodeRefusal",
                  "kRefusalCapacity", "kRefusalForm", "kRefusalModel"):
        assert token in k, f"the refusal ladder is missing {token!r}"
    xs = strip_comments(src["xsset"])
    assert "if (!fss::formImplemented(coord))" in xs, \
        "the host pre-check must reject an unimplemented coordinate form"
    assert "fss::formName(coord)" in xs, \
        ("the refusal must NAME the form; a refusal that only counts leaves the "
         "reader unable to say what would have to be implemented")


def r_one_device_body(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert '#include "FlatXsStreamKernel.h"' in cu, \
        "the .cu must include the shared body"
    assert "fss::flatxsStreamResolveNode(" in cu, \
        "the .cu does not call the shared resolver"
    launcher = region(cu, "__global__ void kFlatXsStreamBuild(", "\n}",
                      "kFlatXsStreamBuild")
    for formula in ("referenceDensity", "NodeSpectralIndex", "SPECTRAL_LOG",
                    "nodeBranchX", "keyCoordinate"):
        assert formula not in launcher, \
            f"the launcher restates the resolver ({formula!r}); it must be a wrapper"


def r_one_host_body(src: dict[str, str]) -> None:
    xs = strip_comments(src["xsset"])
    assert "void XSSet::ResolveNodeApplications(int l," in xs, \
        "the per-node host body must be its own function"
    loop = region(xs, "void XSSet::BuildFlatXsStream(", "\n}", "BuildFlatXsStream")
    assert "ResolveNodeApplications(" in loop, \
        ("BuildFlatXsStream must call the extracted body; a second inline copy is "
         "a second answer for exactly the nodes the device refused")


def r_node_independent_packing(src: dict[str, str]) -> None:
    k = strip_comments(src["kernel"])
    body = region(k, "flatxsStreamResolveNode(const flatxs::FlatXsView& v",
                  "out.node_cnt[i] = n;", "flatxsStreamResolveNode")
    assert "const int base   = i * out.stride;" in body, \
        "a node's slot must be the fixed i*stride, not a scanned offset"
    assert "out.node_off[i] = base;" in body, \
        "the kernel must write its own node_off"
    cu = strip_comments(src["cu"])
    launcher = region(cu, "__global__ void kFlatXsStreamBuild(", "\n}",
                      "kFlatXsStreamBuild")
    for banned in ("atomicAdd", "__shared__", "__syncthreads"):
        assert banned not in launcher, \
            (f"{banned} in the stream launcher: a dense pack or a reduction changes "
             "the ORDER a node's entries land in, which the CTA kernel forbids")


def r_enum_pinned(src: dict[str, str]) -> None:
    model = src["model"]
    names = re.findall(r"^\s*(\w+)\s*=\s*(\d+),", model, re.M)
    enum_body = region(model, "enum class SpectralCoordinate : int {", "\n};",
                       "SpectralCoordinate")
    declared = re.findall(r"^\s*(\w+)\s*=\s*(\d+),", enum_body, re.M)
    assert declared, "could not read Chiffon's SpectralCoordinate enumerators"
    xs = src["xsset"]
    for name, _value in declared:
        assert f"SpectralCoordinate::{name}) ==" in xs, \
            (f"SpectralCoordinate::{name} has no static_assert in XSSet.cpp; the "
             "restatement in FlatXsStreamKernel.h could drift silently")
    assert "Chiffon::SPECTRAL_LOG_DENSITY_FLOOR == fss::kSpectralLogDensityFloor" in xs, \
        "the log floor constant is restated and not pinned"
    assert "Chiffon::ROD_AGE_SCALE == fss::kRodAgeScale" in xs, \
        "the rod-age scale constant is restated and not pinned"


def r_class_declared(src: dict[str, str]) -> None:
    note = region(src["receipt"], "kStreamPolicyNote", "section 4)\";",
                  "kStreamPolicyNote")
    assert "CLASS N1" in note, "the receipt must declare the class"
    assert "NO miner" in note or "no miner" in note, \
        "reason (a) -- the unmined contraction mask -- must be stated"
    assert "log/cbrt" in note, "reason (b) -- the libm forms -- must be stated"
    k = strip_comments(src["kernel"])
    assert "kStreamFormsDefault = 0u" in k, \
        ("the default mask must be 0 (nothing fused) until a miner exists; a "
         "non-zero baked default would be a guess presented as a measurement")


def r_libm_forms(src: dict[str, str]) -> None:
    k = strip_comments(src["kernel"])
    body = region(k, "constexpr bool formUsesLibm(int coord)", "\n}", "formUsesLibm")
    expected = {"kLogDensity", "kFluxRatioInteraction", "kSpectralIndex",
                "kSpectralIndexInteraction", "kRelativeBurnRatio",
                "kLogDeviationSquared", "kCubeRootRatio"}
    found = set(re.findall(r"k[A-Za-z]+", body)) - {"kFormCount"}
    assert found == expected, \
        (f"formUsesLibm names {sorted(found)}; the libm-shaped forms are "
         f"{sorted(expected)} -- the receipt's libm_form_hit reads this list")


def r_receipt_fields(src: dict[str, str]) -> None:
    rc = src["receipt"]
    for field in (r'\"arm\"', r'\"nodes\"', r'\"forms_hit\"',
                  r'\"host_fallback_nodes\"', r'\"wall_ms\"', r'\"bytes_elided\"',
                  r'\"refusals\"', r'\"calls\"', r'\"device_calls\"',
                  r'\"libm_form_hit\"', r'\"policy_note\"'):
        assert field in rc, f"the receipt is missing the {field} field"
    drv = strip_comments(src["driver"])
    assert "[RASBERY][FLATXS][STREAM]" in drv, "the receipt is never printed"
    assert "streamReceiptWanted()" in drv, \
        "the receipt must print only when the arm was asked for or fired"


def r_gpu_full_seam(src: dict[str, str]) -> None:
    full = strip_comments(src["full"])
    assert "FlatXsStream," in full, "Subsystem::FlatXsStream is not declared"
    assert 'return "flatxs_stream";' in full, "the subsystem has no receipt name"
    xs = strip_comments(src["xsset"])
    assert len(re.findall(r"FlatXsStream,\s", xs)) >= 2, \
        ("both fallback seams -- the ineligible deck and the device refusal -- must "
         "be guarded, or one of them is a silent fallback under RASBERY_GPU_FULL")


def r_arm_env(src: dict[str, str]) -> None:
    drv = src["driver"]
    arm = region(drv, "inline constexpr const char* kArmEnv[] = {", "};", "kArmEnv")
    for knob in ('"RASBERY_GPU_FLATXS_STREAM"', '"RASBERY_FLATXS_STREAM_STRIDE"',
                 '"RASBERY_FLATXS_STREAM_FORMS"'):
        assert knob in arm, \
            f"{knob} is not in trajectory::kArmEnv; it selects the branch stream"


def r_fmad(src: dict[str, str]) -> None:
    cm = src["cmake"]
    m = re.search(
        r"set_source_files_properties\(\"\$\{CMAKE_CURRENT_SOURCE_DIR\}/src/"
        r"CudaXsReconBackend\.cu\"(.{0,200}?)\)", cm, re.S)
    assert m, "CudaXsReconBackend.cu has no set_source_files_properties entry"
    var = re.search(r"set\(RASBERY_BITEXACT_CUDA_OPTS\s+\"([^\"]*)\"\s*\)", cm)
    opts = m.group(1).replace("${RASBERY_BITEXACT_CUDA_OPTS}",
                              var.group(1) if var else "")
    assert "--fmad=false" in opts, \
        "the stream kernel's TU must be compiled with --fmad=false"


def r_stub(src: dict[str, str]) -> None:
    stub = strip_comments(src["stub"])
    for sym in ("bool     rasberyGpuFlatXsStreamEnabled() { return false; }",
                "int      rasberyGpuFlatXsStreamStride() { return 0; }",
                "unsigned rasberyGpuFlatXsStreamForms() { return 0u; }",
                "int XsReconBackend::flatXsStreamRefusal() const { return 0; }"):
        assert sym in stub, f"the CPU-only build is missing {sym!r}"
    assert "const flatxs_stream::StreamRequest*" in stub, \
        "the stub's solveFlatXs signature did not follow the header"


def r_request_is_a_parameter(src: dict[str, str]) -> None:
    hdr = strip_comments(src["hdr"])
    assert "const flatxs_stream::StreamRequest* stream = nullptr);" in hdr, \
        ("the request must be a defaulted PARAMETER of solveFlatXs: the phase reads "
         "device state solveFlatXs itself makes resident, so a separate entry point "
         "would run against null pointers on the first call of a run")
    k = strip_comments(src["kernel"])
    assert "struct StreamRequest {" in k, "StreamRequest is not defined in the body header"


RULES = [
    ("default-off-subarm", r_default_off_subarm, "cu",
     ('envFlagEnabled("RASBERY_GPU_FLATXS_STREAM") && rasberyGpuFlatXsEnabled()',
      'true')),
    ("feature-off-is-old-path", r_feature_off_is_old_path, "xsset",
     ("if (!want_stream) {", "if (false) {")),
    ("no-stream-d2h", r_no_stream_d2h, "cu",
     ('"node_cnt",\n                              sb_cnt_host.data(), dev_cnt,'
      ' n_nodes * sizeof(int),\n                              cudaMemcpyDeviceToHost,'
      ' stream) != cudaSuccess ||',
      '"node_cnt",\n                              sb_cnt_host.data(), dev_cnt,'
      ' n_nodes * sizeof(int),\n                              cudaMemcpyDeviceToHost,'
      ' stream) != cudaSuccess ||\n            xfer::memcpyAsync("x", "stream_x",'
      ' nullptr, dev_sx, 8, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||')),
    ("refusal-by-name", r_refusal_by_name, "xsset",
     ("if (!fss::formImplemented(coord))", "if (false)")),
    ("one-device-body", r_one_device_body, "cu",
     ("fss::flatxsStreamResolveNode(v, lib, nd, out, i, fxs::StaticForms{}, spol);",
      "(void)v;")),
    ("one-host-body", r_one_host_body, "xsset",
     ("ResolveNodeApplications(nodes[static_cast<size_t>(i)], hv,",
      "(void)hv; (void)(")),
    ("node-independent-packing", r_node_independent_packing, "kernel",
     ("const int base   = i * out.stride;", "const int base   = 0;")),
    ("enum-pinned", r_enum_pinned, "xsset",
     ("static_assert(static_cast<int>(SpectralCoordinate::CubeRootRatio) == "
      "fss::kCubeRootRatio, \"\");", "")),
    ("class-declared", r_class_declared, "receipt",
     ("CLASS N1 for two independent reasons", "class B0 by construction")),
    ("libm-forms", r_libm_forms, "kernel",
     ("coord == kCubeRootRatio;", "false;")),
    ("receipt-fields", r_receipt_fields, "receipt",
     (r'\"host_fallback_nodes\":', r'\"x\":')),
    ("gpu-full-seam", r_gpu_full_seam, "full",
     ("    FlatXsStream,\n", "\n")),
    ("arm-env", r_arm_env, "driver",
     ('    "RASBERY_GPU_FLATXS_STREAM",\n', "")),
    ("fmad", r_fmad, "cmake",
     ('set(RASBERY_BITEXACT_CUDA_OPTS "--fmad=false")',
      'set(RASBERY_BITEXACT_CUDA_OPTS "")')),
    ("stub", r_stub, "stub",
     ("int      rasberyGpuFlatXsStreamStride() { return 0; }", "")),
    ("request-is-a-parameter", r_request_is_a_parameter, "hdr",
     ("const flatxs_stream::StreamRequest* stream = nullptr);",
      "const void* stream = nullptr);")),
]


def main() -> int:
    failures: list[str] = []
    try:
        src = {k: read(v) for k, v in FILES.items()}
    except AssertionError as exc:
        print(f"FlatXS stream contract: FAIL {exc}")
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
        print("FlatXS stream contract: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"FlatXS stream contract: PASS ({len(RULES)} rules, each with a negative control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
