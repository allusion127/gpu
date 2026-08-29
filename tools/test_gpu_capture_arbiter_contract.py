#!/usr/bin/env python3
"""Contract gate: no allocation API inside a capture window, and no capture window left open.

THE BUG THIS PINS (Rev.7.1 Task 18d).  `--batch-mode M` runs M host Drivers
against ONE arena stream.  The first decks to reach a CMFD solve capture their
outer/sweep graph on that stream while the LATER decks are still standing up --
page-locking their host buffers (BICGCMFD.cpp's first-touch `pinHost` block ->
cudaHostRegister), cudaMalloc'ing their device-outer segment, and
cudaDeviceSynchronize'ing in CudaOuterSegment::bindResidency.

Every one of those is a "potentially unsafe" API in CUDA's stream-capture
vocabulary.  The captures ask for cudaStreamCaptureModeThreadLocal, which
restrains only the CAPTURING thread; the sibling is not stopped and its call
INVALIDATES the capture instead.  Widening the window with
RASBERY_GPU_CAPTURE_STALL_US separates the culprits: 73 sibling
cudaHostRegister calls inside a capture window do not kill it, and a single
cudaDeviceSynchronize kills it every time -- so the device-wide drain is the
trigger and the first-touch pins are what put a sibling thread on the device
API at that moment.  The rule below covers both, because the next unsafe call
added to a stand-up path will not announce which kind it is.

Measured on the 8-deck local batch at 4acff55: 4 runs in 20 died, all of them
1.7-1.9 s in, every one of them reported as

  [RASBERY][FAIL] path=d0.json what=cudaMemcpyAsync(host_status, device_status,
  ...): operation failed due to a previous error during capture

followed by the other seven decks on cudaMemcpyAsync(d_slot_map, ...) with the
same error.  The amplifier was a second defect in the same lines: that
cudaMemcpyAsync is CUDA_CHECK'd, so it THROWS, and both capture sites were

    rc = cudaStreamBeginCapture(stream, ...);
    if (rc == cudaSuccess) { enqueue_...(); rc = cudaStreamEndCapture(...); }

-- the throw skipped the EndCapture, the arena stream stayed in capture mode for
the rest of the process, and one deck's transient became the batch's death.

THE TWO RULES
  1. A capture window is exclusive of every allocation, registration and
     device-wide synchronisation in the process.  In source terms: nothing
     between a cudaStreamBeginCapture and its cudaStreamEndCapture -- directly
     or through the enqueue helper it calls -- may name one of those APIs, and
     every capture site must be inside a rasbery::CaptureWindow (or the
     ScopedStreamCapture that owns one).
  2. A capture window cannot be left open.  Any capture site whose body can
     throw must use ScopedStreamCapture, whose destructor ends the capture.

Run:  python tools/test_gpu_capture_arbiter_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The CUDA translation units that both capture graphs and allocate.
CUDA_SOURCES = (
    "src/CudaBICGBackend.cu",
    "src/CudaXsReconBackend.cu",
    "src/CudaOuterGraph.cu",
    "src/GpuPhysicsArenaCuda.cu",
)

# "Potentially unsafe during capture", in CUDA's own words: allocation,
# page-locking and any synchronisation wider than the captured stream.
FORBIDDEN_IN_CAPTURE = (
    "cudaMalloc",
    "cudaMallocHost",
    "cudaHostAlloc",
    "cudaHostRegister",
    "cudaHostUnregister",
    "cudaFree",
    "cudaFreeHost",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaMemPoolCreate",
    "cudaMemPoolDestroy",
    "pinHost",
    "rasberyPinHost",
    "rasberyUnpinHost",
)

# The helpers a capture window calls to record its work.  Their bodies are
# checked too, because "inside the window" is a RUNTIME extent and the source
# rule has to follow the call.
CAPTURED_HELPERS = ("enqueue_outer", "enqueue_sweeps", "enqueueKernels", "enqueue_full")

# Every cudaHostRegister / cudaHostUnregister in the tree goes through these two
# hook functions (HostPinRegistry.h installs them), so guarding the hooks guards
# every first-touch pin in every deck.
PIN_HOOKS = ("cudaHostPinRegister", "cudaHostPinUnregister")


def read(path: str) -> str:
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as handle:
        return handle.read()


def body_of(code: str, name: str) -> str:
    """The braced body of the first function whose signature names `name`."""
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;{]*\)\s*(?:const\s*)?\{", code)
    if match is None:
        return ""
    start = code.index("{", match.start())
    depth = 0
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[start : i + 1]
    return code[start:]


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def capture_windows(code: str):
    """(begin_index, end_index, text) for every Begin..End capture extent.

    The ScopedStreamCapture guard's own begin()/~ScopedStreamCapture bodies are
    not capture windows -- they are the mechanism the windows are made of -- so
    the scan skips the class that defines them.
    """
    guard = code.find("class ScopedStreamCapture")
    guard_end = -1
    if guard >= 0:
        depth = 0
        for i in range(code.index("{", guard), len(code)):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    guard_end = i
                    break
    for begin in re.finditer(r"cudaStreamBeginCapture\s*\(", code):
        if guard >= 0 and guard < begin.start() < guard_end:
            continue
        end = code.find("cudaStreamEndCapture", begin.end())
        if end < 0:
            end = len(code)
        yield begin.start(), end, code[begin.start() : end], "raw"
    # A ScopedStreamCapture declaration IS a capture window: the literal
    # Begin/End pair lives inside the guard, and the extent the rule cares about
    # runs from the declaration to the matching `.end(`.
    for decl in re.finditer(r"ScopedStreamCapture\s+(\w+)\s*\(", code):
        if guard >= 0 and guard < decl.start() < guard_end:
            continue
        end = code.find(decl.group(1) + ".end(", decl.end())
        if end < 0:
            end = min(len(code), decl.end() + 2000)
        yield decl.start(), end, code[decl.start() : end], "scoped"


def main() -> int:
    problems: list[str] = []
    windows_checked = 0
    helpers_checked = 0

    for path in CUDA_SOURCES:
        raw = read(path)
        code = strip_comments(raw)

        for begin, _end, window, kind in capture_windows(code):
            windows_checked += 1
            line = code[:begin].count("\n") + 1

            # Rule 1a -- the window itself names nothing unsafe.
            for api in FORBIDDEN_IN_CAPTURE:
                if re.search(r"\b" + re.escape(api) + r"\s*\(", window):
                    problems.append(
                        f"{path}: capture window at line ~{line} calls {api}(); an "
                        "allocation, page-lock or wide synchronisation inside a "
                        "capture invalidates it"
                    )

            # Rule 1b -- the window is arbitrated.  Look back over the enclosing
            # statements for the guard that makes it exclusive of allocations.
            preamble = code[max(0, begin - 1200) : begin]
            if kind != "scoped" and not re.search(
                r"(rasbery::)?CaptureWindow\s+\w+\s*\(", preamble
            ):
                problems.append(
                    f"{path}: cudaStreamBeginCapture at line ~{line} is not inside a "
                    "rasbery::CaptureWindow / ScopedStreamCapture, so a sibling "
                    "deck's allocation can invalidate it"
                )

            # Rule 1c -- follow the enqueue helper the window records.
            for helper in CAPTURED_HELPERS:
                if not re.search(r"\b" + re.escape(helper) + r"\s*\(", window):
                    continue
                body = body_of(strip_comments(raw), helper)
                if not body:
                    continue
                helpers_checked += 1
                for api in FORBIDDEN_IN_CAPTURE:
                    if re.search(r"\b" + re.escape(api) + r"\s*\(", body):
                        problems.append(
                            f"{path}: {helper}() is recorded inside a capture window "
                            f"and calls {api}()"
                        )

        # Rule 2 -- the capture sites whose bodies can throw are the CUDA_CHECK
        # ones, and those must be exception-safe.
        if path == "src/CudaBICGBackend.cu":
            for begin, _end, window, kind in capture_windows(code):
                if kind != "scoped":
                    problems.append(
                        "src/CudaBICGBackend.cu: a capture window whose body CUDA_CHECKs "
                        "must use ScopedStreamCapture; a throw would otherwise leave the "
                        "arena stream in capture mode for the rest of the process"
                    )
                if "catch" not in code[begin : begin + 1200]:
                    problems.append(
                        "src/CudaBICGBackend.cu: the enqueue inside a capture window is "
                        "not wrapped in a try/catch, so a throw escapes the window "
                        "instead of being demoted to the direct-enqueue fallback"
                    )

    # Rule 3 -- the pin hooks, which is where every first-touch registration in
    # every deck actually lands, open an allocation window.
    for path in ("src/CudaBICGBackend.cu", "src/CudaXsReconBackend.cu"):
        code = strip_comments(read(path))
        for hook in PIN_HOOKS:
            body = body_of(code, hook)
            if not body:
                problems.append(f"{path}: expected the pin hook {hook}() to exist")
                continue
            if "AllocWindow" not in body:
                problems.append(
                    f"{path}: {hook}() must open a rasbery::AllocWindow -- it is the "
                    "one call every deck's first-touch pinHost reaches, and it runs "
                    "on that deck's Driver thread while the siblings capture"
                )

    # Rule 4 -- the arbiter itself declines rather than deadlocks, and can be
    # switched off for an A/B without losing its receipts.
    arbiter = read("src/GpuCaptureArbiter.h")
    for needle, why in (
        ("threadIsCapturing()", "the capturing thread must not take the shared lock"),
        ("allocDepthRef() > 0", "a capture opened inside an alloc window must not upgrade"),
        ("RASBERY_GPU_CAPTURE_ARBITER", "the locking must be switchable for the A/B"),
        ("captureArbiterReceipt", "the arbiter must publish a receipt"),
    ):
        if needle not in arbiter:
            problems.append(f"src/GpuCaptureArbiter.h: missing {needle} -- {why}")

    if windows_checked == 0:
        problems.append("no capture windows found; the scan is looking at the wrong files")

    if problems:
        print("FAIL: gpu capture arbiter contract")
        for problem in sorted(set(problems)):
            print("  - " + problem)
        return 1
    print("PASS: gpu capture arbiter contract")
    print(f"  capture windows checked: {windows_checked}")
    print(f"  recorded helpers checked: {helpers_checked}")
    print(f"  sources: {', '.join(CUDA_SOURCES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
