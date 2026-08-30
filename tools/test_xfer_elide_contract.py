#!/usr/bin/env python3
"""Contract gate for WP13's RASBERY_GPU_XFER_ELIDE arm and the [RASBERY][XFER] receipt.

WHAT THIS PROTECTS.  Every elision behind RASBERY_GPU_XFER_ELIDE is a claim of
B0 -- the device ends up holding the same BITS, so no kernel can tell an elided
run from a copied one.  That claim rests on three properties, none of which is
visible in a diff and all three of which are one careless edit away:

C1  THE FLAG IS NOT A TRAJECTORY KNOB.  A pure transfer elision cannot move a
    result, so RASBERY_GPU_XFER_ELIDE must NOT be in trajectory::kArmEnv.
    Putting it there would fork the evaluator's case key for two runs that are
    the same run.  (If a future change ever lets this flag change a value the
    host reads, kArmEnv is where that has to be declared -- and this test is
    where the declaration gets noticed.)

C2  ONLY DEVICE-READ-ONLY BUFFERS ARE SHADOWED.  A host shadow of "what I last
    uploaded" is a true statement about device content only while no kernel
    writes that buffer.  The three CMFD masks qualify; `sweep_halt` does NOT
    (initialize_solver_state raises it, issueSweepDownloads memsets it), so it
    must keep its unconditional copy.  Over-reach here is silent and produces a
    wrong answer, not a crash.

C3  EVERY SHADOW IS INVALIDATED WHEN ITS DEVICE BUFFER MOVES.  A mirror that
    survives a cudaFree/cudaMalloc describes bytes in a freed allocation, and
    the first elision against it writes nothing where the kernel reads garbage.

C4  THE RECEIPT EXISTS AND CARRIES THE SEVEN COUNTERS, and main.cpp emits it.
    An elision with no receipt is a claim with no measurement.

C5  THE OFF ARM IS THE CODE THAT SHIPPED.  Both guarded-upload helpers must
    consult and commit their shadow only under elideEnabled(), so an A/B
    measures the elision and not the bookkeeping.

NEGATIVE CONTROLS.  Each check is run a second time against a MUTATED copy of
the source that breaks exactly the property it tests, and the check must fail.
A gate that cannot fail is not a gate -- this file's own history is that three
of these checks were substring searches that would have passed on an empty file.

Run:  python tools/test_xfer_elide_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BICG = "src/CudaBICGBackend.cu"
XSR = "src/CudaXsReconBackend.cu"
LEDGER = "src/XferLedger.h"
MAIN = "src/main.cpp"
DRIVER = "src/Driver.h"

FLAG = "RASBERY_GPU_XFER_ELIDE"

RECEIPT_FIELDS = (
    "d2h_calls",
    "d2h_bytes",
    "h2d_calls",
    "h2d_bytes",
    "syncs",
    "elided_calls",
    "elided_bytes",
)


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    """Comments explain the contract; they must never SATISFY it."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def body_of(code: str, signature: str, end: str) -> str:
    start = code.find(signature)
    if start < 0:
        return ""
    stop = code.find(end, start)
    return code[start:stop] if stop > start else code[start:]


# ---------------------------------------------------------------------------
# the checks.  Each takes the source map and returns a list of problems.
# ---------------------------------------------------------------------------


def check_c1_not_an_arm_knob(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    driver = src[DRIVER]
    anchor = "inline constexpr const char* kArmEnv[] = {"
    at = driver.find(anchor)
    if at < 0:
        return [f"C1: cannot find kArmEnv in {DRIVER}; the absence test is void"]
    block = driver[at + len(anchor) : driver.find("};", at)]
    names = re.findall(r'"([A-Za-z0-9_]+)"', block)
    if not names:
        return ["C1: kArmEnv parsed empty; the absence test is void"]
    # The parse is only trusted if it finds a knob we KNOW is there.  Without
    # this, an empty or mis-anchored parse would report "not present" and pass.
    if "RASBERY_GPU_CRAM" not in names:
        problems.append(
            "C1: kArmEnv parse looks wrong -- RASBERY_GPU_CRAM missing from "
            f"{len(names)} parsed names; the absence test cannot be trusted"
        )
    if FLAG in names:
        problems.append(
            f"C1: {FLAG} is in trajectory::kArmEnv.  It is a pure transfer "
            "elision (identical device bytes, identical host-visible values), "
            "so it cannot move a trajectory and must not fork the case key.  "
            "If it CAN now move one, say so in the doc and delete this check."
        )
    return problems


# ===========================================================================
# THE LEDGER MOVED THE SPELLING, NOT THE CALL  (WP13.1, commit 914f6b3)
# ===========================================================================
#
# Two things happened to this file's scans at once, and both are spelling:
#
#   * the guarded-upload helpers gained a LEAF TAG as their FIRST argument --
#     `pushDeviceReadOnly("issueSweepUploads:device_active", device_active, ...)`
#     and `uploadGuarded("stream_x", d.dev_sx, ...)` -- so a regex that captured
#     the first argument as the destination captured a string literal, or
#     nothing, and every buffer read as "no longer guarded".
#   * the raw copies became `rasbery::xfer::memcpyAsync(scope, leaf, dst, ...)`,
#     which puts the DESTINATION third.  A scan for `cudaMemcpyAsync(dst` can
#     no longer find an unguarded copy, and C5's "does this helper still issue a
#     copy at all" read as no.
#
# So the destination is matched through an OPTIONAL tag prefix, and both copy
# spellings are accepted.  Nothing here is relaxed: the tag prefix is anchored
# to a single string literal, so a call with a computed destination still fails
# to match, and the raw-copy scans gained a spelling rather than losing one.
GUARD_TAG = r'(?:"[^"]*"\s*,\s*)?'
COPY_CALL = r'(?:cudaMemcpyAsync\(\s*|(?:rasbery::)?xfer::memcpyAsync\(\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*)'
COPY_SPELLINGS = ("cudaMemcpyAsync", "xfer::memcpyAsync")


def check_c2_only_readonly_buffers(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    code = strip_comments(src[BICG])

    if "void pushDeviceReadOnly(" not in code:
        return ["C2: BatchCore::pushDeviceReadOnly is gone; nothing is guarded"]

    guarded = re.findall(
        r"pushDeviceReadOnly\(\s*" + GUARD_TAG + r"([A-Za-z_][A-Za-z0-9_]*)", code)
    for want in ("d_slot_map", "device_active", "device_assembly_active"):
        if want not in guarded:
            problems.append(
                f"C2: {want} no longer goes through pushDeviceReadOnly -- the "
                "per-outer copy is back and the elision is silently off"
            )

    # THE OVER-REACH CONTROL.  sweep_halt has a device-side writer; a shadow of
    # the host's last value is not a statement about device content, and eliding
    # against it would skip a copy that is doing real work.
    if "sweep_halt" in guarded:
        problems.append(
            "C2: sweep_halt is shadow-guarded.  initialize_solver_state RAISES "
            "it and issueSweepDownloads memsets it, so the device value between "
            "two launches is NOT what the host last sent.  Revert it to an "
            "unconditional cudaMemcpyAsync."
        )

    # Every writer of device_active must share the one shadow, or a rendezvous
    # launch leaves it describing bytes the device no longer holds.
    raw_active = re.findall(
        COPY_CALL + r"(?:device_active|device_assembly_active|d_slot_map)\b",
        code,
    )
    if raw_active:
        problems.append(
            f"C2: {len(raw_active)} raw cudaMemcpyAsync into a shadowed buffer "
            "remain.  A second writer that bypasses the shadow makes every "
            "elision at the guarded site wrong."
        )
    return problems


def check_c2b_flatxs_inputs_guarded(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    code = strip_comments(src[XSR])
    if "bool uploadGuarded(" not in code:
        return ["C2b: Impl::uploadGuarded is gone; the flat-XS inputs are unguarded"]

    guarded = re.findall(
        r"uploadGuarded\(\s*" + GUARD_TAG + r"([A-Za-z_][A-Za-z0-9_.+ *]*?),", code)
    guarded_txt = " ".join(guarded)
    for want in (
        "d.dev_pernode",
        "d.dev_nodes",
        "d.dev_off",
        "d.dev_cnt",
        "d.dev_sdid",
        "d.dev_sx",
        "d.dev_sscale",
    ):
        if want not in guarded_txt:
            problems.append(f"C2b: {want} no longer goes through uploadGuarded")

    # The raw spelling must be gone, or the guarded call is dead code beside a
    # copy that still runs.
    for buf in ("d.dev_pernode", "d.dev_nodes", "d.dev_off", "d.dev_cnt",
                "d.dev_sdid", "d.dev_sx", "d.dev_sscale"):
        pattern = COPY_CALL + re.escape(buf) + r"\b"
        if re.search(pattern, code):
            problems.append(
                f"C2b: a raw cudaMemcpyAsync into {buf} is back beside the "
                "guarded one"
            )
    return problems


def check_c3_shadows_invalidated(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    code = strip_comments(src[XSR])

    # ensure(): the geometry regrow that frees dev_block / dev_pernode.
    ensure = body_of(code, "bool ensure(int want_nxyz", "bool xeEnsure")
    if not ensure:
        problems.append("C3: cannot find Impl::ensure; the invalidation test is void")
    else:
        for want in ("mir_wvfr.invalidate()", "invalidateNodeMirrors()",
                     "invalidateStreamMirrors()"):
            if want not in ensure:
                problems.append(
                    f"C3: Impl::ensure does not call {want}.  A geometry change "
                    "frees or re-lays-out the buffer the shadow describes."
                )

    flat = body_of(code, "bool XsReconBackend::solveFlatXs", "bool XsReconBackend::solveNodal")
    if not flat:
        problems.append("C3: cannot find solveFlatXs; the regrow test is void")
    else:
        # The two grow-only caps: each cudaFree/cudaMalloc pair must clear its
        # mirrors in the same block.
        if "d.nodes_cap = n_nodes;" in flat and "d.invalidateNodeMirrors();" not in flat:
            problems.append(
                "C3: the node-array regrow in solveFlatXs does not invalidate "
                "mir_nodes/mir_node_off/mir_node_cnt"
            )
        if "d.stream_cap = stream_len;" in flat and "d.invalidateStreamMirrors();" not in flat:
            problems.append(
                "C3: the stream-array regrow in solveFlatXs does not invalidate "
                "mir_stream_did/mir_stream_x/mir_stream_scale"
            )
        if "d.dev_pernode == nullptr" in flat and "d.mir_wvfr.invalidate();" not in flat:
            problems.append(
                "C3: the dev_pernode allocation in solveFlatXs does not "
                "invalidate mir_wvfr/mir_dmod/mir_bppm"
            )
    return problems


def check_c4_receipt(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    ledger = src[LEDGER]
    ledger_code = strip_comments(ledger)
    for field in RECEIPT_FIELDS:
        if f"std::atomic<unsigned long long> {field}" not in ledger_code:
            problems.append(f"C4: the ledger has no counter named {field}")
        if f'\\"{field}\\":' not in ledger_code and f'"{field}\\":' not in ledger_code:
            # the fields are written as ",\"name\":" inside a C++ string literal
            if f'{field}\\":' not in ledger_code:
                problems.append(f"C4: {field} is not written into the receipt")
    if "[RASBERY][XFER]" not in src[MAIN]:
        problems.append("C4: main.cpp does not emit the [RASBERY][XFER] receipt")
    if src[MAIN].count("appendXferReceiptFields") < 3:
        problems.append(
            "C4: the receipt is emitted from fewer than all three exit paths "
            "(single, batch, evaluator) -- a run on the missing arm reports "
            "nothing and its numbers cannot be read"
        )
    if '#include "XferLedger.h"' not in src[MAIN]:
        problems.append("C4: main.cpp does not include XferLedger.h")
    return problems


def check_c5_off_arm_is_untouched(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for path, sig, end in (
        (BICG, "void pushDeviceReadOnly(", "/// Decide this launch's dispatch width"),
        (XSR, "bool uploadGuarded(", "\n};"),
    ):
        code = strip_comments(src[path])
        body = body_of(code, sig, end)
        if not body:
            problems.append(f"C5: cannot find the guarded-upload helper in {path}")
            continue
        # Both the consult and the commit must sit inside an elideEnabled()
        # test, or the OFF arm pays for a shadow it never reads.
        if body.count("elideEnabled()") < 2:
            problems.append(
                f"C5: {path}'s guarded upload does not gate BOTH the shadow "
                "consult and the shadow commit on elideEnabled(); the OFF arm "
                "is then not the code that shipped"
            )
        if not any(spelling in body for spelling in COPY_SPELLINGS):
            problems.append(
                f"C5: {path}'s guarded upload no longer issues a copy at all"
            )
    return problems


CHECKS = (
    ("C1", check_c1_not_an_arm_knob),
    ("C2", check_c2_only_readonly_buffers),
    ("C2b", check_c2b_flatxs_inputs_guarded),
    ("C3", check_c3_shadows_invalidated),
    ("C4", check_c4_receipt),
    ("C5", check_c5_off_arm_is_untouched),
)

# (check name, description, file, mutation applied to that file's text)
NEGATIVE_CONTROLS = (
    (
        "C1",
        "the flag added to kArmEnv",
        DRIVER,
        lambda t: t.replace(
            'inline constexpr const char* kArmEnv[] = {',
            'inline constexpr const char* kArmEnv[] = {\n    "'
            + FLAG
            + '",',
            1,
        ),
    ),
    (
        "C2",
        "sweep_halt shadow-guarded (over-reach)",
        BICG,
        lambda t: t.replace(
            'CUDA_CHECK(rasbery::xfer::memcpyAsync(\n'
            '            "CudaBICGBackend.cu:issueSweepUploads", "sweep_halt", sweep_halt, halt,',
            'pushDeviceReadOnly("issueSweepUploads:sweep_halt", sweep_halt, halt,\n'
            '            static_cast<size_t>(slots), shadow_active); if (false) CUDA_CHECK(\n'
            '            rasbery::xfer::memcpyAsync("x", "y", sweep_halt, halt,',
            1,
        ),
    ),
    (
        "C2",
        "a raw copy into device_active restored beside the guarded one",
        BICG,
        lambda t: t.replace(
            '        pushDeviceReadOnly("issueSweepUploads:device_assembly_active",',
            '        CUDA_CHECK(rasbery::xfer::memcpyAsync(\n'
            '            "CudaBICGBackend.cu:issueSweepUploads", "raw", device_active,\n'
            '            active, 4, cudaMemcpyHostToDevice, stream));\n'
            '        pushDeviceReadOnly("issueSweepUploads:device_assembly_active",',
            1,
        ),
    ),
    (
        "C2b",
        "the flat-XS stream upload un-guarded",
        XSR,
        lambda t: t.replace(
            '        if (!d.uploadGuarded("stream_x", d.dev_sx, host.stream_x, stream_len,\n'
            '                             d.mir_stream_x))\n'
            '            return false;',
            '        RASBERY_CUDA_TRY(rasbery::xfer::memcpyAsync(\n'
            '            "CudaXsReconBackend.cu:solveFlatXs", "stream_x", d.dev_sx,\n'
            '            host.stream_x, stream_len * sizeof(double),\n'
            '            cudaMemcpyHostToDevice, d.stream), d.status);',
            1,
        ),
    ),
    (
        "C3",
        "the node-array regrow stops invalidating its mirrors",
        XSR,
        lambda t: t.replace("        d.invalidateNodeMirrors();\n", "", 1),
    ),
    (
        "C3",
        "ensure() stops invalidating the stream mirrors",
        XSR,
        lambda t: t.replace("        invalidateStreamMirrors();\n", "", 1),
    ),
    (
        "C4",
        "a receipt counter dropped",
        LEDGER,
        lambda t: t.replace("elided_bytes", "elided_octets"),
    ),
    (
        "C5",
        "the shadow commit escapes the elideEnabled() gate",
        XSR,
        lambda t: t.replace(
            "        if (xfer::elideEnabled()) mirror.commit(src, count);",
            "        mirror.commit(src, count);",
            1,
        ),
    ),
)


def main() -> int:
    src = {p: read(p) for p in (BICG, XSR, LEDGER, MAIN, DRIVER)}

    problems: list[str] = []
    for _, check in CHECKS:
        problems.extend(check(src))

    # --- the negative controls -------------------------------------------
    control_failures: list[str] = []
    for name, description, path, mutate in NEGATIVE_CONTROLS:
        mutated = dict(src)
        mutated[path] = mutate(src[path])
        if mutated[path] == src[path]:
            control_failures.append(
                f"negative control [{name}] {description}: the mutation did not "
                f"apply to {path} -- the anchor text has moved, so this control "
                "is measuring nothing.  Update the anchor."
            )
            continue
        fired = False
        for check_name, check in CHECKS:
            if check_name != name:
                continue
            if check(mutated):
                fired = True
        if not fired:
            control_failures.append(
                f"negative control [{name}] {description}: {name} PASSED on a "
                "source that breaks it -- the check does not test what it says"
            )

    for line in problems:
        print(f"FAIL {line}")
    for line in control_failures:
        print(f"FAIL {line}")

    if problems or control_failures:
        print(
            f"\ntest_xfer_elide_contract: {len(problems)} contract violation(s), "
            f"{len(control_failures)} dead negative control(s)"
        )
        return 1

    print(
        f"test_xfer_elide_contract: OK "
        f"({len(CHECKS)} checks, {len(NEGATIVE_CONTROLS)} negative controls)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
