#!/usr/bin/env python3
"""Contract gate for the cmfd_wiel_finalize Wielandt fold.

The fold reproduces BICGCMFD::wiel's l-ascending `err`/`gammad`/`gamman`
accumulation bit for bit, and that exact rounding sequence is baked into the
frozen reference output.  A rounded floating-point add is not associative, so
the ONE thing this fold may never acquire is a partition: no stage-1/stage-2
split, no per-thread strided traversal, no tree.  The only latitude it has is
where the LOADS happen, because moving a load changes nothing about the adds.

What this asserts, statically:

  1. The shipped fold body and test/cmfd_wiel_fold_replay.cu's copy of it are
     the same text.  The replay is the only thing that proves bit-identity on
     a device; if the two drift the replay is testing a kernel that no longer
     ships.
  2. The fold still folds into ONE accumulator in ascending order: exactly one
     `sum = sum +` form, no `+=` on a partial array, no shared-memory tree, no
     `threadIdx.x` in the traversal, no `blockDim.x` stride.
  3. The reference kernel in the replay is still the flat loop -- the thing the
     shipped form is measured against must not be "optimised" too.
  4. reduce_dot_stage1/2's chunking is untouched (the fold must not be
     "unified" with it, and R2/R3 must not disturb it either).
  5. kWielFoldBatch agrees between the kernel and the replay.

Optionally (if nvcc and a device are present) it builds and runs the replay.
Run:  python tools/test_cmfd_wiel_fold_contract.py [--run]
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CU = os.path.join(ROOT, "src", "CudaBICGBackend.cu")
REPLAY = os.path.join(ROOT, "test", "cmfd_wiel_fold_replay.cu")

BEGIN = "RASBERY_CMFD_WIEL_FOLD_BEGIN"
END = "RASBERY_CMFD_WIEL_FOLD_END"


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def fold_body(text: str, where: str, problems: list[str]) -> str:
    start = text.find(BEGIN)
    stop = text.find(END)
    if start < 0 or stop < 0 or stop < start:
        problems.append(f"{where}: no {BEGIN}/{END} marked fold body")
        return ""
    # Take from the end of the BEGIN comment block to the END marker, and
    # normalise whitespace so an indent change is not a failure.
    body = text[start + len(BEGIN):stop]
    body = body.split("\n", 1)[1] if "\n" in body else body
    # Drop the marker comment tail lines (they name the other file, so they
    # legitimately differ in wording).
    lines = [ln.strip() for ln in body.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("//")]
    return "\n".join(lines)


def batch_constant(text: str, where: str, problems: list[str]) -> str:
    hit = re.search(r"constexpr\s+int\s+kWielFoldBatch\s*=\s*(\d+)\s*;", text)
    if hit is None:
        problems.append(f"{where}: kWielFoldBatch is not declared as a constexpr int")
        return ""
    return hit.group(1)


def main() -> int:
    problems: list[str] = []
    cu = read(CU)
    replay = read(REPLAY)

    # --- 1. the two fold bodies are the same text ---------------------------
    shipped = fold_body(cu, "src/CudaBICGBackend.cu", problems)
    mirrored = fold_body(replay, "test/cmfd_wiel_fold_replay.cu", problems)
    if shipped and mirrored and shipped != mirrored:
        problems.append(
            "the fold body in src/CudaBICGBackend.cu and the copy in "
            "test/cmfd_wiel_fold_replay.cu have drifted apart; the replay would be "
            "proving bit-identity for a kernel that no longer ships.\n"
            f"  shipped:\n{shipped}\n  replay:\n{mirrored}"
        )

    # --- 2. still one accumulator, still ascending --------------------------
    if shipped:
        adds = re.findall(r"sum\s*=\s*sum\s*\+", shipped)
        if len(adds) != 2:
            problems.append(
                "the fold must contain exactly two `sum = sum + ...` sites (the batched "
                f"body and the tail); found {len(adds)}. Any other accumulation shape is a "
                "reassociation and changes the result."
            )
        for banned, why in (
            (r"\bthreadIdx\b", "a per-thread traversal partitions the range"),
            (r"\bblockDim\b", "a strided traversal partitions the range"),
            (r"\bblockIdx\b", "a per-block chunk partitions the range"),
            (r"__shared__", "a shared-memory tree reassociates the sum"),
            (r"__shfl", "a warp shuffle reduction reassociates the sum"),
            (r"atomicAdd", "an atomic reduction has no defined order at all"),
            (r"\bfma\b", "the addends are already rounded; an fma would change them"),
            (r"sum\s*\+=", "use the explicit `sum = sum + x` form the host mines"),
        ):
            if re.search(banned, shipped):
                problems.append(f"the fold body contains `{banned}` -- {why}")
        if "for (; l < nxyz; ++l) sum = sum + values[l];" not in shipped:
            problems.append(
                "the fold has no ascending scalar tail loop; a width that is not a "
                "multiple of kWielFoldBatch would drop or reorder its last addends"
            )

    # --- 3. the replay's reference is still the flat loop -------------------
    ref = replay[replay.find("__global__ void fold_reference"):]
    ref = ref[: ref.find("__global__ void fold_shipped")]
    if "for (int l = 0; l < nxyz; ++l) sum = sum + values[l];" not in ref:
        problems.append(
            "test/cmfd_wiel_fold_replay.cu: fold_reference is no longer the flat "
            "l-ascending loop. It is the frozen form; it must never be optimised."
        )
    if BEGIN in ref:
        problems.append(
            "test/cmfd_wiel_fold_replay.cu: the marked fold body leaked into "
            "fold_reference, so the comparison would be a kernel against itself"
        )

    # --- 3b. the OPT-IN chunked arm (N1) ------------------------------------
    #
    # The chunked fold is a legitimate campaign arm and a catastrophic default:
    # it is 1.6e-14 away from the frozen reference output, which is far too
    # small to notice in a plot and far too large to accept.  So the gate is
    # not "is it correct" -- it is "can it ever be reached without someone
    # having asked for it", plus the determinism its N1 classification claims.
    code = strip_comments(cu)

    if "enum class WielFoldMode" not in code:
        problems.append("no WielFoldMode: the chunked arm must be a named mode, not a bool")
    gate = code[code.find("inline WielFoldMode wielFoldMode()"):][:1200]
    if not gate:
        problems.append("wielFoldMode() is missing")
    else:
        if "RASBERY_GPU_WIEL_FOLD" not in gate:
            problems.append("wielFoldMode does not read RASBERY_GPU_WIEL_FOLD")
        # DEFAULT SERIAL, and unreachable-chunked-when-unset.  Both spellings
        # of "nothing was asked for" -- unset, and set to empty/whitespace --
        # must land on serial.
        if "value == nullptr" not in gate or "return WielFoldMode::SERIAL" not in gate:
            problems.append("wielFoldMode does not default to serial when the env is unset")
        if 's.empty() || s == "serial"' not in gate:
            problems.append(
                "an empty or whitespace-only RASBERY_GPU_WIEL_FOLD must mean serial")
        if 's == "chunked"' not in gate:
            problems.append("wielFoldMode never returns CHUNKED for \"chunked\"")
        # chunked must be reachable ONLY through that one comparison: if any
        # other branch can return it, "unset means serial" stops being a
        # property of the parser.
        chunked_returns = gate.count("WielFoldMode::CHUNKED")
        if chunked_returns != 1:
            problems.append(
                f"wielFoldMode has {chunked_returns} paths to CHUNKED; exactly one "
                "(the explicit \"chunked\" spelling) may exist, or an unset variable "
                "could select an N1 arm")
        # trim + case-fold, like Driver.h's xeMode.
        if "tolower" not in gate:
            problems.append("wielFoldMode does not case-fold (xeMode does)")
        if "isspace" not in gate:
            problems.append("wielFoldMode does not trim surrounding whitespace")
        if "not a mode" not in cu:
            problems.append(
                "a typo in RASBERY_GPU_WIEL_FOLD must warn rather than silently buy a "
                "mode; the two arms produce different bits")

    # BOTH paths present, and the scalar tail shared between them so the two
    # modes cannot drift into different Rayleigh/latch semantics.
    for kernel in ("cmfd_wiel_finalize", "cmfd_wiel_stage1", "cmfd_wiel_finalize_chunked"):
        if f"__global__ void {kernel}(" not in code:
            problems.append(f"{kernel} is missing; both fold paths must be present")
    if "__device__ __forceinline__ void cmfd_wiel_apply(" not in code:
        problems.append(
            "the scalar tail is not factored into cmfd_wiel_apply; the two modes would "
            "each carry their own copy of the eigenvalue update and the sweep_state==2 "
            "Rayleigh latch, free to drift apart")
    if code.count("sm[kSweepState] = 2.0;") != 1:
        problems.append(
            "the Rayleigh latch appears more than once; there must be exactly one copy, "
            "shared by both fold modes")
    for kernel in ("cmfd_wiel_finalize", "cmfd_wiel_finalize_chunked"):
        body = code[code.find(f"__global__ void {kernel}("):]
        body = body[: body.find("\n__global__", 1) if body.find("\n__global__", 1) > 0
                    else len(body)]
        if "cmfd_wiel_apply(" not in body:
            problems.append(f"{kernel} does not go through the shared scalar tail")

    # The N1 determinism claim, statically: fixed partition, fixed tree, strict
    # stage-2 walk, and NO atomics anywhere in the chunked path.
    stage1 = code[code.find("__global__ void cmfd_wiel_stage1("):]
    stage1 = stage1[: stage1.find("__global__ void cmfd_wiel_finalize_chunked")]
    if ("const int chunk = (nxyz + static_cast<int>(gridDim.x) - 1) / "
            "static_cast<int>(gridDim.x);") not in stage1:
        problems.append(
            "cmfd_wiel_stage1's partition is not the (nxyz, gridDim.x) chunk; an N1 arm "
            "must be reproducible, which means the partition may depend on nothing else")
    if "for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1)" not in stage1:
        problems.append("cmfd_wiel_stage1 does not use the fixed binary reduction tree")
    stage2 = code[code.find("__global__ void cmfd_wiel_finalize_chunked("):][:1200]
    if "for (int i = 0; i < blocks; ++i) sum = sum + values[i];" not in stage2:
        problems.append("the chunked stage-2 fold is not a strict ascending index walk")
    for banned in ("atomicAdd", "atomicCAS", "__shfl", "cooperative"):
        if banned in stage1 or banned in stage2:
            problems.append(
                f"the chunked fold uses `{banned}`; an N1 arm has to be bit-identical "
                "run to run, and that rules out any order the hardware picks")

    # The receipt, and its three fields.
    if "[RASBERY][CMFD][WIEL_FOLD]" not in cu:
        problems.append("no [RASBERY][CMFD][WIEL_FOLD] receipt")
    receipt = cu[cu.find("inline void reportWielFold("):][:1200]
    # The fields are emitted as C++ string literals, so they carry their
    # backslash-escaped quotes in the source text.
    for field in ('\\"mode\\"', '\\"source\\"', '\\"chunk\\"'):
        if field not in receipt:
            problems.append(f"the WIEL_FOLD receipt has no {field} field")
    if "wielFoldModeFromEnv" not in receipt:
        problems.append(
            "the receipt does not record provenance; an N1 result that reached a report "
            "without one is indistinguishable from a B0 one")

    # And the launch actually forks on the mode.
    if "if (wielFoldMode() == WielFoldMode::CHUNKED) {" not in code:
        problems.append("enqueue_sweeps does not fork on the fold mode")

    # --- 4. the neighbouring reduction chunking is untouched ----------------
    for needle, why in (
        ("const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);",
         "reduce_dot_stage1's partition is a pure function of (n, gridDim.x)"),
        ("for (int i = 0; i < blocks; ++i) sum += pm[i];   // strict index order",
         "reduce_dot_stage2's strict fold order"),
    ):
        if needle not in cu:
            problems.append(f"src/CudaBICGBackend.cu: lost `{needle[:60]}...` -- {why}")

    # --- 5. the batch constant agrees ---------------------------------------
    a = batch_constant(cu, "src/CudaBICGBackend.cu", problems)
    b = batch_constant(replay, "test/cmfd_wiel_fold_replay.cu", problems)
    if a and b and a != b:
        problems.append(f"kWielFoldBatch differs: kernel {a}, replay {b}")
    if a and (int(a) < 1 or int(a) > 64):
        problems.append(f"kWielFoldBatch = {a} is outside the sane 1..64 register budget")

    if problems:
        for problem in problems:
            print(f"cmfd wiel fold contract: FAIL {problem}", file=sys.stderr)
        return 1

    if "--run" in sys.argv:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            print("cmfd wiel fold contract: replay SKIPPED (nvcc not on PATH)")
        else:
            arch = os.environ.get("RASBERY_TEST_ARCH", "sm_61")
            with tempfile.TemporaryDirectory() as tmp:
                exe = os.path.join(tmp, "replay")
                build = subprocess.run(
                    [nvcc, "-O3", "-std=c++17", f"-arch={arch}", "--fmad=false", REPLAY,
                     "-o", exe],
                    capture_output=True, text=True)
                if build.returncode != 0:
                    print(f"cmfd wiel fold contract: FAIL replay build\n{build.stderr}",
                          file=sys.stderr)
                    return 1
                run = subprocess.run([exe], capture_output=True, text=True)
                sys.stdout.write(run.stdout)
                sys.stderr.write(run.stderr)
                if run.returncode != 0:
                    return 1

    print("cmfd wiel fold contract: PASS (fold body mirrored, single ascending "
          "accumulator, reference flat form intact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
