#!/usr/bin/env python3
"""GPU Nodal updateConstant contract (Rev.7.1 plan Task 4, Sec 6.1).

Six properties, none of which any numerical test can see:

  1. THE FORMULAS ARE NOT DUPLICATED.  src/CudaNodalConstantKernel.h must call
     nodalConstantCoefficients() and must not contain a second spelling of the
     SENM coefficients.  A duplicated formula agrees on the day it is written and
     diverges on the first edit, silently, because both copies keep producing
     plausible numbers.
  2. THE W1 RESOLUTION ORDER.  Every phase kernel starts with
     gpuDispatchIsPadding, takes its slot from queue.slots[logical], and reaches
     the arrays through arena.slotView(slot).  A kernel that used blockIdx as a
     slot id would work at bucket 64 with a full fleet and quietly drive the
     wrong tenant at any other width.
  3. NO PHASE KERNEL WRITES queued_phase / queued_epoch (Sec 5.2).  Those are
     captured by classify; a phase kernel that stamps them re-validates the
     entry it is consuming and the slot is queued twice next epoch.
  4. THE PHASE IS THREE LAUNCHES, in order, on ONE stream: compute, publish
     cache, publish generation.  The splits are not tidiness -- each one removes
     a read/write race that has no barrier available to fix it (W0 closed the
     persistent/cooperative track), and collapsing them back is invisible until
     it is a wrong answer.
  5. --fmad=false ON THE CUDA TU.  With gcc fusing twenty multiply-adds in this
     body and nvcc fusing on its own schedule, the flag plus the mined
     NODAL_CONST_FORMS is the whole reason the GPU deviation is confined to exp.
  6. THE CLASSIFICATION IS RECORDED.  Task 4 Step 0's B0-rescue spike FAILED
     (device exp differs from glibc on ~5% of the arguments this body
     evaluates), so both headers must say N1 and must not claim B0.
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST = ROOT / "test"

KERNEL = SRC / "CudaNodalConstantKernel.h"
BODY = SRC / "NodalConstantKernel.h"
CMAKE = ROOT / "CMakeLists.txt"
REPLAY = TEST / "nodal_constant_gpu_replay.cpp"
DEVICE_REPLAY = TEST / "nodal_constant_device_replay.cu"
PROBE = TEST / "nodal_constant_exp_probe.cu"
GOLDEN = TEST / "nodal_constant_exp_golden.h"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


KERNEL_TEXT = read(KERNEL)
KERNEL_CODE = strip_comments(KERNEL_TEXT)
BODY_TEXT = read(BODY)
BODY_CODE = strip_comments(BODY_TEXT)
CMAKE_TEXT = read(CMAKE)
REPLAY_TEXT = read(REPLAY)
DEVICE_TEXT = read(DEVICE_REPLAY)
PROBE_TEXT = read(PROBE)
GOLDEN_TEXT = read(GOLDEN)


def want(text: str, needle: str, where: str, why: str) -> None:
    if needle not in text:
        problems.append(f"{where}: missing {needle!r} -- {why}")


def forbid(text: str, needle: str, where: str, why: str) -> None:
    if needle in text:
        problems.append(f"{where}: contains {needle!r} -- {why}")


# --- 1. the formulas live in ONE place ---------------------------------------
want(KERNEL_CODE, "nodal::nodalConstantCoefficients(", "CudaNodalConstantKernel.h",
     "the kernel must call the shared pure body, not restate it")
# Distinctive intermediates of the SENM coefficient body.  Any of them appearing
# in the CUDA header means the formulas were copied.
for token in ("sinhkp", "coshkp", "bfcff", "oddtemp", "eventemp", "rkp5", "iekp"):
    forbid(KERNEL_CODE, token, "CudaNodalConstantKernel.h",
           "a coefficient intermediate belongs to NodalConstantKernel.h; a second copy "
           "of the formulas will diverge on the first edit")
# `kp2` is the one name that could plausibly appear in an index expression, so
# check it as a whole word rather than a substring.
if re.search(r"\bkp2\b", KERNEL_CODE):
    problems.append("CudaNodalConstantKernel.h: mentions 'kp2' -- the coefficient body "
                    "must not be restated here")

# --- 2. the W1 resolution order ----------------------------------------------
GLOBALS = re.findall(r"__global__[^{]*?void\s+(\w+)\s*\(", KERNEL_CODE)
if not GLOBALS:
    problems.append("CudaNodalConstantKernel.h: no __global__ kernel found")
for name in GLOBALS:
    # Body of this kernel: from its opening brace to the next __global__ or EOF.
    start = KERNEL_CODE.index(name)
    rest = KERNEL_CODE[start:]
    nxt = rest.find("__global__", 1)
    body = rest[:nxt] if nxt > 0 else rest
    if "gpuDispatchIsPadding(" not in body:
        problems.append(f"{name}: does not start with gpuDispatchIsPadding -- a padding "
                        "lane would read kQueueEmptySlot as a slot id")
    if "queue.slots[logical]" not in body:
        problems.append(f"{name}: does not resolve its slot as queue.slots[logical] -- "
                        "blockIdx is an index into the queue, never a slot id")

want(KERNEL_CODE, "arena.slotView(", "CudaNodalConstantKernel.h",
     "the slot's arrays are reached through the arena view table (Sec 4.1 fixed "
     "addresses), not recomputed from a stride")

# --- 3. no phase kernel touches the queue capture ----------------------------
for field in ("queued_phase", "queued_epoch"):
    if re.search(r"\." + field + r"\s*=", KERNEL_CODE):
        problems.append(f"CudaNodalConstantKernel.h: writes {field} -- that field is "
                        "captured by classify (Sec 5.2); stamping it from a phase kernel "
                        "re-validates the entry being consumed and double-queues the slot")

# --- 4. three launches, in order, on one stream ------------------------------
ENQUEUE = KERNEL_CODE[KERNEL_CODE.find("enqueueNodalUpdateConstant"):] if \
    "enqueueNodalUpdateConstant" in KERNEL_CODE else ""
if not ENQUEUE:
    problems.append("CudaNodalConstantKernel.h: no enqueueNodalUpdateConstant")
else:
    order = [k for k in ("k_nodal_update_constant", "k_nodal_constant_publish_cache",
                         "k_nodal_constant_publish_generation")
             if f"{k}<<<" in ENQUEUE]
    expected = ["k_nodal_update_constant", "k_nodal_constant_publish_cache",
                "k_nodal_constant_publish_generation"]
    if order != expected:
        problems.append(
            "enqueueNodalUpdateConstant: launches %s, expected %s in that order -- the "
            "compute kernel READS the xsrf/xsdf cache the publish kernel writes, and the "
            "generation is what the cache kernel's own gate reads, so both splits are "
            "races if collapsed" % (order, expected))
    # Every launch must name the SAME stream: the stream order IS the barrier.
    launches = re.findall(r"<<<[^>]*?,\s*([A-Za-z_]\w*)\s*>>>", ENQUEUE)
    if launches and len(set(launches)) != 1:
        problems.append("enqueueNodalUpdateConstant: launches use different streams (%s); "
                        "the ordering between them is the only barrier there is"
                        % sorted(set(launches)))

want(KERNEL_CODE, "kNodalConstantBlock = 128", "CudaNodalConstantKernel.h",
     "Sec 7 fixes the (node, group) block at 128")
want(KERNEL_CODE, "queue.bucket", "CudaNodalConstantKernel.h",
     "grid.y is the dispatch bucket (Sec 5.5), not the ready count")
want(KERNEL_CODE, "material_generation", "CudaNodalConstantKernel.h",
     "Sec 6.1 gates the phase on the material generation")

# --- packing order, which no numerical test can see --------------------------
for token in ("kNcDiagDI,", "kNcDiagD,"):
    want(KERNEL_CODE, token, "CudaNodalConstantKernel.h",
         "the nine packed arrays are named, so the arena's order (diagDI before diagD) "
         "is stated once")
if KERNEL_CODE.find("kNcDiagDI") > KERNEL_CODE.find("kNcDiagD,"):
    problems.append("CudaNodalConstantKernel.h: diagD is declared before diagDI; the "
                    "arena packs diagDI first (SlotRegion::NodalConst)")

# --- 5. --fmad=false on the CUDA TUs -----------------------------------------
# 2c04a6e moved the literal --fmad=false behind RASBERY_BITEXACT_CUDA_OPTS so that
# RASBERY_PTXAS_VERBOSE could APPEND to it.  What this contract needs is the
# RESOLVED option list, not the spelling, so expand the variable instead of
# grepping for the literal -- and expand a missing definition to "", so deleting
# the set() still fails here.
_bitexact = re.search(r'set\(RASBERY_BITEXACT_CUDA_OPTS\s+"([^"]*)"\s*\)', CMAKE_TEXT)
_BITEXACT_OPTS = _bitexact.group(1) if _bitexact else ""
for cu in ("test/nodal_constant_device_replay.cu", "test/nodal_constant_exp_probe.cu"):
    opts = re.search(r'set_source_files_properties\(\s*"\$\{CMAKE_CURRENT_SOURCE_DIR\}/'
                     + re.escape(cu) + r'"\s*\n\s*PROPERTIES COMPILE_OPTIONS "([^"]*)"',
                     CMAKE_TEXT)
    if cu not in CMAKE_TEXT:
        problems.append(f"CMakeLists.txt: does not build {cu}")
    elif opts is None or "--fmad=false" not in opts.group(1).replace(
            "${RASBERY_BITEXACT_CUDA_OPTS}", _BITEXACT_OPTS):
        problems.append(f"CMakeLists.txt: {cu} is not compiled with --fmad=false; nvcc "
                        "would then contract on its own schedule and the deviation would "
                        "no longer be confined to exp")

# --- 6. the classification is recorded, and it is N1 -------------------------
for path, text in (("NodalConstantKernel.h", BODY_TEXT),
                   ("CudaNodalConstantKernel.h", KERNEL_TEXT)):
    if "N1" not in text:
        problems.append(f"{path}: does not record the Task 4 classification (N1)")
    if re.search(r"CLASSIFICATION:\s*B0", text):
        problems.append(f"{path}: claims B0; the Task 4 Step 0 rescue spike FAILED "
                        "(device exp differs from glibc on ~5% of the swept arguments)")

# --- the contraction mask is present, mined, and consistent ------------------
mask_const = re.search(r"NODAL_CONST_FORMS\s*=\s*(0x[0-9A-Fa-f]+)ull", BODY_CODE)
mask_fn = re.search(r"nodalConstForms\(\)\s*\{\s*return\s+(0x[0-9A-Fa-f]+)ull", BODY_CODE)
if not mask_const or not mask_fn:
    problems.append("NodalConstantKernel.h: NODAL_CONST_FORMS / nodalConstForms() not found")
elif mask_const.group(1).lower() != mask_fn.group(1).lower():
    problems.append("NodalConstantKernel.h: NODAL_CONST_FORMS (%s) and nodalConstForms() "
                    "(%s) disagree; device code reads the function, host code reads the "
                    "constant" % (mask_const.group(1), mask_fn.group(1)))
elif int(mask_const.group(1), 16) == 0:
    problems.append("NodalConstantKernel.h: NODAL_CONST_FORMS is still 0 -- the mask was "
                    "never mined, so every multiply-add is unfused and the host body has "
                    "MOVED off the CPU baseline")
want(BODY_CODE, "ncMa1(", "NodalConstantKernel.h",
     "the multiply-add sites must go through the form policy")
want(BODY_CODE, "ncMa2(", "NodalConstantKernel.h",
     "the two-product sites must go through the form policy")

# --- the spike and the replays exist and say what they measured --------------
want(REPLAY_TEXT, "--mine", "test/nodal_constant_gpu_replay.cpp",
     "the mask must be re-derivable, not just asserted")
want(REPLAY_TEXT, "legacyNodalConstantCoefficients", "test/nodal_constant_gpu_replay.cpp",
     "the pre-policy body is the CPU baseline and has to be kept to score against")
want(DEVICE_TEXT, "unattributed", "test/nodal_constant_device_replay.cu",
     "the N1 gate is attribution: every differing element must sit where exp differs")
want(DEVICE_TEXT, "run-to-run", "test/nodal_constant_device_replay.cu",
     "an N1 phase owes run-to-run bit determinism")
want(PROBE_TEXT, "VERDICT", "test/nodal_constant_exp_probe.cu",
     "the spike must state its verdict")
if GOLDEN_TEXT and GOLDEN_TEXT.count("ull,") < 3 * 32:
    problems.append("test/nodal_constant_exp_golden.h: fewer than 32 baked anchors; the "
                    "glibc reference is the thing the N1 receipt is written against")


def main() -> int:
    if problems:
        for problem in problems:
            print("nodal constant gpu contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("nodal constant gpu contract: PASS (%d kernels, mask %s)"
          % (len(GLOBALS), mask_const.group(1) if mask_const else "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
