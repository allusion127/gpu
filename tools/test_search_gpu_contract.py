#!/usr/bin/env python3
"""GPU critical-boron search contract -- WP22, commit 2.

Twelve properties.  The arm this file guards is SMALL -- one broadcast kernel and
an elided upload -- and that is exactly why the properties have to be asserted
here rather than measured: the saving is inside the noise of a single-deck wall,
so no A/B would notice the arm quietly doing nothing, doing it twice, or doing it
to the wrong buffer.

  1. DEFAULT OFF, AND THE OFF PATH IS THE OLD PATH.  applyBoronDevice is reached
     only from XSSet::SetBoron and its false return leaves the host broadcast and
     the flat-XS backend's guarded upload exactly as they were.

  2. THE HOST MIRROR IS ALWAYS WRITTEN.  This is the rule most likely to be
     "optimised away" by a future reader who sees a device write and a host loop
     doing the same thing.  They are NOT the same thing:
     XSSet::BuildFlatXsStream reads `_g.bppm(l)` per node on the host to form the
     boron coordinate of the branch stream, so the host array is a live input of
     the very next UpdateFlatXS.  Skipping it would reconstruct the core at the
     PREVIOUS trial's boron.

  3. ONE TEXT, BOTH ARMS.  The kernel and the host loop write through the same
     body (rasbery::search::searchBoronBroadcastNode), so "the device wrote what
     the host would have" is a property of one function rather than of two
     spellings that agree today.

  4. NO FORM MASK, AND THE HEADER SAYS WHY.  There is no arithmetic to contract:
     every node receives the same double.  A form mask here would be cargo
     imported from the T/H arm.

  5. B0 IS CLAIMED BY CONSTRUCTION, and the claim is written where a gate script
     reads it.

  6. THE ARM DECLINES RATHER THAN CLAMPS.  No resident block, a shape mismatch,
     or RASBERY_GPU_XFER_ELIDE off must all return false.  The third is the
     subtle one: with the elision flag off the guarded upload would overwrite the
     kernel's block, so an arm that "succeeded" there would report device applies
     against a saving of zero.

  7. THE OWNER OWNS THE BUFFER.  The boron block is the flat-XS backend's
     `dev_pernode`, so the apply is a method on that backend and commits that
     backend's own shadow.  A second object writing it would be a second opinion
     about a buffer with one owner.

  8. THE PROPOSE STEP TRANSFERS NOTHING.  The secant reads a k_eff the solve
     already published to the host.  The whole point of the arm is to remove a
     round trip; growing one in the propose bucket would be the opposite.

  9. THE SECANT STAYS ON THE HOST.  Scheduler.h's proposal arithmetic must not
     acquire a device path -- there is nothing to gain and a second place for the
     bracket logic to live.

 10. RASBERY_GPU_SEARCH *IS* AN ARM KNOB, AND THE REASON IS WRITTEN DOWN.  The
     kernel cannot move a trajectory, so the knob is on the list for a different
     reason (which path carried each trial) and that reason has to be legible or
     someone will correctly remove it.

 11. THE RECEIPT THE PLAN ASKS FOR, including the two fields that make the arm
     falsifiable: `device_applies` and `propose_transfers`.

 12. THE GPU_FULL SEAM EXISTS, for Subsystem::Search.

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
    "kernel": "src/SearchKernel.h",
    "receipt": "src/SearchGpuReceipt.h",
    "cu": "src/CudaXsReconBackend.cu",
    "hdr": "src/CudaXsReconBackend.h",
    "stub": "src/CudaXsReconBackendStub.cpp",
    "xsset": "src/XSSet.cpp",
    "driver": "src/Driver.h",
    "scheduler": "src/Scheduler.h",
    "contract": "src/GpuFullContract.h",
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
    assert 'envFlagEnabled("RASBERY_GPU_SEARCH")' in cu, \
        "the arm must be gated on RASBERY_GPU_SEARCH and default off"
    xs = strip_comments(src["xsset"])
    assert "on_device = backend->applyBoronDevice(bppm, nxyz);" in xs, \
        "XSSet::SetBoron must try the device"
    assert "search_tally.host_fallbacks.fetch_add(1, std::memory_order_relaxed);" in xs, \
        "a decline must be counted, or the receipt cannot say the arm never ran"


def r_host_mirror_always(src: dict[str, str]) -> None:
    xs = strip_comments(src["xsset"])
    body = region(xs, "void XSSet::SetBoron(double bppm) {", "\n}\n", "SetBoron")
    call = "rasbery::search::searchBoronBroadcastHost(&_g.bppm(0), nxyz, bppm);"
    assert call in body, "SetBoron no longer writes the host bppm mirror"
    # ... and it must NOT be inside the else-branch of the device test.  The
    # cheapest way to say that in text: the mirror write must come AFTER the
    # whole device block closes, at the function's own indentation.
    assert "\n    " + call in body, (
        "the host bppm write moved inside a branch.  It is a live input of the "
        "very next UpdateFlatXS (BuildFlatXsStream reads _g.bppm(l) per node), "
        "not a copy the device made redundant")
    assert "UpdateFlatXS();" in body and body.index(call) < body.index("UpdateFlatXS();"), \
        "the mirror is written after the reconstruction that reads it"


def r_one_text(src: dict[str, str]) -> None:
    kernel = strip_comments(src["kernel"])
    assert "searchBoronBroadcastNode" in kernel, "the shared per-node body is gone"
    cu = strip_comments(src["cu"])
    assert "rasbery::search::searchBoronBroadcastNode(bppm, l, value);" in cu, \
        "the device kernel does not write through the shared body"
    xs = strip_comments(src["xsset"])
    assert "searchBoronBroadcastHost" in xs, \
        "the host loop does not write through the shared body"
    host = region(kernel, "inline void searchBoronBroadcastHost(", "\n}", "host broadcast")
    assert "searchBoronBroadcastNode(bppm, l, value)" in host, \
        "the host broadcast stopped going through the per-node body, so the two arms are " \
        "two texts again"


def r_no_form_mask(src: dict[str, str]) -> None:
    kernel = src["kernel"]
    assert "forms" not in strip_comments(kernel), \
        "SearchKernel.h has grown a form mask; there is no arithmetic here to contract"
    assert "no arithmetic" in kernel or "no rounding decision" in kernel, \
        "the header does not say why it has no form mask, so the next arm will copy one in"


def r_class_b0(src: dict[str, str]) -> None:
    receipt = src["receipt"]
    assert "kSearchGpuPolicyNote" in receipt, \
        "the receipt does not carry the grade a gate script reads"
    assert "B0 by construction" in receipt, "the grade is not stated"
    kernel = src["kernel"]
    assert "CLASS B0, BY CONSTRUCTION AND NOT BY MEASUREMENT" in kernel, \
        "SearchKernel.h does not declare the gate class"


def r_declines(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    body = region(cu, "bool XsReconBackend::applyBoronDevice(", "\n}\n", "applyBoronDevice")
    assert "if (d.dev_pernode == nullptr) {" in body, \
        "the arm does not decline before the resident block exists"
    assert "searchBoronApplyServable" in body, \
        "the arm does not decline on a shape mismatch; a kernel that wrote a prefix would " \
        "leave the tail holding the previous trial's boron"
    assert "if (!xfer::elideEnabled()) {" in body, \
        "the arm does not decline when RASBERY_GPU_XFER_ELIDE is off, so it would report " \
        "device applies against a saving the guarded upload immediately overwrote"


def r_owner_owns_buffer(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    body = region(cu, "bool XsReconBackend::applyBoronDevice(", "\n}\n", "applyBoronDevice")
    assert "d.dev_pernode + 2 * n" in body, \
        "the kernel does not write the flat-XS backend's own bppm sub-block"
    assert "d.mir_bppm.commit(" in body, \
        "the apply does not commit the owner's shadow, so the next guarded upload would " \
        "copy the bytes the kernel just wrote and the arm would save nothing"
    hdr = src["hdr"]
    assert "bool applyBoronDevice(double bppm, int nxyz);" in hdr, \
        "the apply is not declared on the buffer's owner"


def r_propose_transfers_nothing(src: dict[str, str]) -> None:
    driver = strip_comments(src["driver"])
    body = region(driver, "outer_timing::Scope propose_scope(sptelem::PH_SEARCH_PROPOSE);",
                  "schedule.CommitSearchPoint(", "search propose")
    for forbidden in ("xfer::memcpy", "cudaMemcpy", "streamSync", "cudaStreamSynchronize",
                      "Download", "download"):
        assert forbidden not in body, (
            f"the search propose bucket contains {forbidden!r}.  The secant reads a k_eff "
            "the solve already published to the host; a transfer here is the round trip "
            "this arm exists to avoid")


def r_secant_on_host(src: dict[str, str]) -> None:
    sched = strip_comments(src["scheduler"])
    assert "ProposeNextSearchPoint" in sched, "the proposal entry point moved"
    for forbidden in ("cuda", "Cuda", "Device", "device_"):
        assert forbidden not in sched, (
            f"Scheduler.h has acquired {forbidden!r}: the secant and the bracket must stay "
            "host-only, or the bracket logic lives in two places")


def r_arm_knob(src: dict[str, str]) -> None:
    raw = src["driver"]
    code = strip_comments(raw)
    m = re.search(r"kArmEnv\[\]\s*=\s*\{(.*?)\};", code, re.S)
    assert m, "trajectory::kArmEnv vanished"
    assert "RASBERY_GPU_SEARCH" in m.group(1), \
        "RASBERY_GPU_SEARCH is not in trajectory::kArmEnv"
    anchor = raw.find("kArmEnv[]")
    assert anchor > 0
    preamble = raw[max(0, anchor - 8000):anchor]
    assert "RASBERY_GPU_SEARCH is deliberately PRESENT" in preamble, (
        "kArmEnv does not explain why RASBERY_GPU_SEARCH is in the list.  The kernel is B0 "
        "by construction and cannot move a trajectory, so without the written reason "
        "(which path carried each trial) somebody will correctly remove it")


def r_receipt(src: dict[str, str]) -> None:
    code = strip_comments(src["driver"])
    assert "[RASBERY][SEARCH_GPU]" in code, "the arm must publish a receipt"
    fields = strip_comments(src["receipt"])
    for field in ('\\"arm\\":', '\\"applies\\":', '\\"device_applies\\":',
                  '\\"host_fallbacks\\":', '\\"proposals\\":',
                  '\\"propose_transfers\\":', '\\"bytes_elided\\":'):
        assert field in fields, f"the [RASBERY][SEARCH_GPU] receipt is missing {field}"
    driver = strip_comments(src["driver"])
    assert "searchGpuTally().proposals.fetch_add(" in driver, \
        "`proposals` is never incremented, so the applies/proposals comparison is vacuous"


def r_gpu_full_seam(src: dict[str, str]) -> None:
    contract = src["contract"]
    assert re.search(r"\bSearch,", contract), \
        "gpufull::Subsystem has no Search member; the seam has nothing to name"
    assert 'case Subsystem::Search: return "search";' in contract, \
        "Subsystem::Search has no name, so the receipt would print `unknown_fallbacks`"
    xs = strip_comments(src["xsset"])
    assert "Search,\n                                  \"XSSet::SetBoron\"," in xs or \
        "RASBERY_GPU_FULL_GUARD_IF(backend != nullptr && backend->boronArmed(), Search," in xs, \
        "the boron-apply decline is not guarded for Subsystem::Search"


def r_stub(src: dict[str, str]) -> None:
    stub = strip_comments(src["stub"])
    assert re.search(r"bool XsReconBackend::applyBoronDevice\([^)]*\)\s*\{\s*return false;\s*\}",
                     stub, re.S), "the stub applyBoronDevice must return false"
    assert re.search(r"bool XsReconBackend::boronArmed\(\)\s*const\s*\{\s*return false;\s*\}",
                     stub, re.S), "the stub must report the boron arm unavailable"


RULES = [
    ("default-off", r_default_off, "cu",
     ('envFlagEnabled("RASBERY_GPU_SEARCH")', 'envFlagEnabled("RASBERY_ALWAYS_ON")')),
    ("host-mirror-always", r_host_mirror_always, "xsset",
     ("\n    rasbery::search::searchBoronBroadcastHost(&_g.bppm(0), nxyz, bppm);",
      "\n        rasbery::search::searchBoronBroadcastHost(&_g.bppm(0), nxyz, bppm);")),
    ("one-text", r_one_text, "cu",
     ("rasbery::search::searchBoronBroadcastNode(bppm, l, value);",
      "bppm[l] = value;")),
    # The control has to be CODE, not a comment: r_no_form_mask reads the
    # comment-stripped text, so a `/// forms` line would be invisible to it and
    # the control would pass for the wrong reason.
    ("no-form-mask", r_no_form_mask, "kernel",
     ("struct BoronApplyView {",
      "struct BoronApplyView {\n    unsigned long long forms = 0;")),
    ("class-b0", r_class_b0, "kernel",
     ("CLASS B0, BY CONSTRUCTION AND NOT BY MEASUREMENT", "SOME NOTES")),
    ("declines", r_declines, "cu",
     ("if (!xfer::elideEnabled()) {", "if (false) {")),
    ("owner-owns-buffer", r_owner_owns_buffer, "cu",
     ("d.mir_bppm.commit(", "(void)(")),
    ("propose-transfers-nothing", r_propose_transfers_nothing, "driver",
     ("double      next_x = schedule.search_current_x;",
      "double      next_x = schedule.search_current_x; cudaMemcpy(0,0,0,0);")),
    ("secant-on-host", r_secant_on_host, "scheduler",
     ("bool ProposeNextSearchPoint(", "bool ProposeNextSearchPointCudaDevice(")),
    ("arm-knob", r_arm_knob, "driver",
     ('    "RASBERY_GPU_SEARCH",\n', "")),
    ("receipt", r_receipt, "driver",
     ("[RASBERY][SEARCH_GPU]", "[RASBERY][SEARCH_TIMING]")),
    ("gpu-full-seam", r_gpu_full_seam, "contract",
     ('case Subsystem::Search: return "search";', 'case Subsystem::Search: break;')),
    ("stub", r_stub, "stub",
     ("bool XsReconBackend::applyBoronDevice(double, int) { return false; }",
      "bool XsReconBackend::applyBoronDevice(double, int) { return true; }")),
]


def main() -> int:
    failures: list[str] = []
    try:
        src = {k: read(v) for k, v in FILES.items()}
    except AssertionError as exc:
        print(f"SEARCH GPU contract: FAIL {exc}")
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
        print("SEARCH GPU contract: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"SEARCH GPU contract: PASS ({len(RULES)} rules, each with a negative control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
