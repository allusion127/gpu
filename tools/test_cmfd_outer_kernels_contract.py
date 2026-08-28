#!/usr/bin/env python3
"""GPU CMFD pre/post kernel contract (Rev.7.1 Task 5, Sec 6.2/6.3/6.6/6.12/6.13).

Nine properties, none of which a numerical test can see:

  1. THE ARITHMETIC IS NOT DUPLICATED.  src/CudaCmfdOuterKernels.h calls the
     shared bodies in src/CmfdOuterKernel.h and restates none of them.
  2. THE CONTRACTION MASK IS MINED AND CONSISTENT.  CMFD_OUTER_FORMS and
     cmfdOuterForms() must agree, and the sites must go through coMa1/coMa2.
     Class B0 means bit-identical to the CPU, and gcc fuses two of the three
     sites and not the third -- `_psi[l] += flux*xsnf` is NOT fused, which is
     the one a reasonable person guesses wrong.
  3. THE W1 RESOLUTION ORDER in every phase kernel: gpuDispatchIsPadding first,
     then queue.slots[logical].  A kernel that used blockIdx as a slot id works
     at full fleet width and drives the wrong tenant at any other.
  4. NO PHASE KERNEL WRITES queued_phase / queued_epoch (Sec 5.2), and none
     writes DeviceSlotPhase::phase either -- the transition belongs to the
     scheduler, and the convergence kernel publishes a DECISION instead.
  5. Sec 6.12's REDUCTION SHAPE: a block reduction and one atomic per block for
     the three integer counts, atomicMax over the bit pattern for the ratio.
     213k atomics per outer on four addresses is what this replaces.
  6. THE ratio MAXIMUM IS GUARDED.  atomicMax over bit patterns is only the
     maximum for NON-NEGATIVE doubles, so the negative sentinel must be excluded
     before the conversion.
  7. THE DHAT COUNTER SEMANTICS.  Both early exits count into fsum_guard, the
     clamp is a runtime flag (CMFD.cpp:164-178: enforcing it is worth ~+100 pcm
     at i-SMR CY01 BOC), and every evaluation bumps the total.
  8. Sec 6.13's FIVE NAMED FIELDS are all carried: flux_stall, stall_events,
     stall_sample_taken, clean_iters, xe_interim_count.  Rev.7 omitted the last
     three; a slot that drops any of them diverges on the stall path.
  9. THE EDGES THE CONVERGENCE KERNEL CAN EMIT ALL EXIST in kPhaseTransitions.
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST = ROOT / "test"

BODY = SRC / "CmfdOuterKernel.h"
KERNEL = SRC / "CudaCmfdOuterKernels.h"
SCHED = SRC / "GpuPhaseScheduler.h"
CMAKE = ROOT / "CMakeLists.txt"
PROBE = TEST / "cmfd_outer_form_probe.cpp"
REPLAY = TEST / "cmfd_outer_replay.cpp"
REFERENCE = TEST / "cmfd_outer_reference.cpp"
DEVICE = TEST / "cmfd_outer_device_replay.cu"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


BODY_TEXT = read(BODY)
BODY_CODE = strip_comments(BODY_TEXT)
KERNEL_TEXT = read(KERNEL)
KERNEL_CODE = strip_comments(KERNEL_TEXT)
SCHED_CODE = strip_comments(read(SCHED))
CMAKE_TEXT = read(CMAKE)
PROBE_TEXT = read(PROBE)
REPLAY_TEXT = read(REPLAY)
REFERENCE_TEXT = read(REFERENCE)
DEVICE_TEXT = read(DEVICE)


def want(text: str, needle: str, where: str, why: str) -> None:
    if needle not in text:
        problems.append(f"{where}: missing {needle!r} -- {why}")


# --- 1. the arithmetic lives in ONE place ------------------------------------
for body in ("cmfdUpdDtilSurface", "cmfdUpdPsiNode", "cmfdUpdJnetSurface",
             "cmfdUpdDhatSurface", "cmfdOuterConvergence"):
    want(BODY_CODE, body + "(", "CmfdOuterKernel.h", "the shared body must exist here")
    want(KERNEL_CODE, body + "(", "CudaCmfdOuterKernels.h",
         "the kernel must call the shared body")
# Distinctive operands of the CMFD bodies.  Their appearance in the CUDA header
# would mean the arithmetic was copied rather than called.
for token in ("betal", "betar", "jnet_fdm", "fdiff"):
    if token in KERNEL_CODE:
        problems.append(f"CudaCmfdOuterKernels.h: contains {token!r} -- the CMFD arithmetic "
                        "belongs to CmfdOuterKernel.h; a second copy will diverge")

# --- 2. the mined mask -------------------------------------------------------
mask_const = re.search(r"CMFD_OUTER_FORMS\s*=\s*(0x[0-9A-Fa-f]+)ull", BODY_CODE)
mask_fn = re.search(r"cmfdOuterForms\(\)\s*\{\s*return\s+(0x[0-9A-Fa-f]+)ull", BODY_CODE)
if not mask_const or not mask_fn:
    problems.append("CmfdOuterKernel.h: CMFD_OUTER_FORMS / cmfdOuterForms() not found")
elif mask_const.group(1).lower() != mask_fn.group(1).lower():
    problems.append("CmfdOuterKernel.h: CMFD_OUTER_FORMS (%s) and cmfdOuterForms() (%s) "
                    "disagree; device code reads the function, host code the constant"
                    % (mask_const.group(1), mask_fn.group(1)))
else:
    # NO ASSERTION ON THE BITS.  The mask records which multiply-adds THE BUILD
    # HOST's compiler fused, and that is host-specific: measured 0x6 on the
    # authoring box (WSL2, g++ 13.3) and 0x7 on 238 (Xeon Gold 5317, where
    # CO_PSI_ACC IS fused).  A contract test that pinned the bits would fail on
    # one of the two machines for a reason that has nothing to do with the code.
    #
    # What IS pinned instead: the mask has a runtime override, the gates mine it
    # rather than assert it, and the per-build-host nature is documented where
    # the next person will look.
    mask = int(mask_const.group(1), 16)
    if mask == 0:
        problems.append("CmfdOuterKernel.h: CMFD_OUTER_FORMS is 0 -- every site unfused, "
                        "which is the never-mined seed, not a measurement")
    for needle, why in (
            ("PER BUILD HOST", "the host-specific nature has to be stated where it is read"),
            ("RASBERY_CMFD_OUTER_FORMS", "the runtime override"),
            ("cmfdOuterFormsRuntime", "the resolved accessor")):
        if needle not in BODY_TEXT:
            problems.append(f"CmfdOuterKernel.h: missing {needle!r} -- {why}")
    if "resolveFormMask" not in BODY_CODE:
        problems.append("CmfdOuterKernel.h: the runtime mask is not resolved through "
                        "GpuFormMask.h, so an override would not be receipt-logged")
want(BODY_CODE, "coMa1(", "CmfdOuterKernel.h", "the multiply-add sites need the policy")
want(BODY_CODE, "coMa2(", "CmfdOuterKernel.h", "the two-product site needs the policy")

# --- 3. the W1 resolution order ----------------------------------------------
GLOBALS = re.findall(r"__global__[^{]*?void\s+(\w+)\s*\(", KERNEL_CODE)
if len(GLOBALS) < 5:
    problems.append("CudaCmfdOuterKernels.h: expected five phase kernels, found %d: %s"
                    % (len(GLOBALS), GLOBALS))
for name in GLOBALS:
    start = KERNEL_CODE.index("void " + name)
    rest = KERNEL_CODE[start:]
    nxt = rest.find("__global__", 1)
    kbody = rest[:nxt] if nxt > 0 else rest
    if "gpuDispatchIsPadding(" not in kbody:
        problems.append(f"{name}: does not start with gpuDispatchIsPadding -- a padding lane "
                        "would read kQueueEmptySlot as a slot id")
    if "queue.slots[logical]" not in kbody:
        problems.append(f"{name}: does not resolve its slot as queue.slots[logical]")

# --- 4. nothing writes the scheduler's fields --------------------------------
for field in ("queued_phase", "queued_epoch"):
    if re.search(r"\." + field + r"\s*=", KERNEL_CODE):
        problems.append(f"CudaCmfdOuterKernels.h: writes {field} -- classify captures it "
                        "(Sec 5.2); a phase kernel that stamps it double-queues the slot")
if re.search(r"\bphases\[[^\]]+\]\.phase\s*=", KERNEL_CODE) or \
        re.search(r"\.phase\s*=\s*static_cast<std::uint8_t>", KERNEL_CODE):
    problems.append("CudaCmfdOuterKernels.h: writes DeviceSlotPhase::phase -- the transition "
                    "is the scheduler's; the convergence kernel publishes a decision")
want(KERNEL_CODE, "CmfdOuterDecision", "CudaCmfdOuterKernels.h",
     "the convergence kernel publishes a decision rather than writing the phase word")

# --- 5. the Sec 6.12 reduction shape -----------------------------------------
want(KERNEL_CODE, "cmfdReduceCounters", "CudaCmfdOuterKernels.h",
     "Sec 6.12 wants a block reduction, not one global atomic per element")
if "__shfl" in KERNEL_CODE:
    problems.append("CudaCmfdOuterKernels.h: uses a warp shuffle intrinsic; constraint 35 "
                    "requires the wrapper, and the reduction runs once per block of an "
                    "already-elementwise pass so there is nothing to trade for it")
want(KERNEL_CODE, "atomicMax", "CudaCmfdOuterKernels.h",
     "the ratio maximum goes through atomicMax over the bit pattern")
dhat_start = KERNEL_CODE.find("k_cmfd_upd_dhat")
dhat_body = KERNEL_CODE[dhat_start:KERNEL_CODE.find("k_cmfd_outer_convergence")] \
    if dhat_start >= 0 else ""
if dhat_body.count("atomicAdd") != 0:
    problems.append("k_cmfd_upd_dhat: calls atomicAdd directly; the per-block atomics belong "
                    "inside the reduction helper so there is exactly one per block")
reduce_start = KERNEL_CODE.find("cmfdReduceCounters")
reduce_body = KERNEL_CODE[reduce_start:reduce_start + 3000] if reduce_start >= 0 else ""
if reduce_body.count("atomicAdd") != 3:
    problems.append("cmfdReduceCounters: expected exactly three atomicAdd sites (one per "
                    "integer count), found %d" % reduce_body.count("atomicAdd"))
if "__syncthreads" not in reduce_body:
    problems.append("cmfdReduceCounters: no __syncthreads between the warp results and the "
                    "cross-warp fold")
# Every thread must reach the barrier: an early `return` for out-of-range threads
# before a __syncthreads deadlocks the block.
if dhat_body and re.search(r"if\s*\(\s*tid\s*>=\s*total\s*\)\s*return", dhat_body):
    problems.append("k_cmfd_upd_dhat: returns early for out-of-range threads, but the "
                    "counter reduction contains a __syncthreads -- the rest of the block "
                    "would wait on a barrier nobody reaches")

# --- 6. the ratio sentinel is excluded before the bit conversion -------------
if "c.ratio >= 0.0" not in KERNEL_CODE:
    problems.append("k_cmfd_upd_dhat: does not exclude the negative 'no contribution' "
                    "sentinel before cmfdRatioToBits; a negative double's sign bit makes it "
                    "the largest unsigned value there is, so atomicMax would pick it")
want(BODY_CODE, "out.ratio   = -1.0", "CmfdOuterKernel.h",
     "the 'no ratio' sentinel must be negative, not 0.0, or a |dtil|==0 surface is "
     "indistinguishable from one whose ratio really was zero")

# --- 7. dhat counter semantics ----------------------------------------------
want(BODY_CODE, "fsum_guard", "CmfdOuterKernel.h", "the fsum/finiteness guard is counted")
want(BODY_CODE, "clamp_enabled", "CmfdOuterKernel.h",
     "RASBERY_DHAT_CLAMP must stay a runtime flag: CMFD.cpp records that enforcing the "
     "envelope biases i-SMR CY01 BOC by ~+100 pcm")
if BODY_CODE.count("out.fsum_guard = 1") != 2:
    problems.append("CmfdOuterKernel.h: expected BOTH upddhat early exits to count into "
                    "fsum_guard (CMFD.cpp:152 and :159), found %d"
                    % BODY_CODE.count("out.fsum_guard = 1"))

# --- 8. Sec 6.13's five named fields ----------------------------------------
for field in ("flux_stall", "stall_events", "stall_sample_taken", "clean_iters",
              "xe_interim_count"):
    want(BODY_CODE, field, "CmfdOuterKernel.h",
         "Rev.7.1 Sec 6.13 names this field explicitly; a slot that drops it diverges on "
         "the stall path")
    want(KERNEL_CODE, field, "CudaCmfdOuterKernels.h",
         "the field must be carried through DeviceSlotState")
want(BODY_CODE, "MAX_FLUX_STALL_EVENTS", "CmfdOuterKernel.h", "the fatal stall budget")
want(BODY_CODE, "SEARCH_SETTLE_ITERS", "CmfdOuterKernel.h", "the settling gate")
want(BODY_CODE, "FluxStallFatal", "CmfdOuterKernel.h", "the fatal exit must be reachable")
want(BODY_CODE, "FluxLimitCycleSample", "CmfdOuterKernel.h",
     "the limit-cycle sample is an escape the search consumes")

# --- 9. every emitted edge exists in the W1 table ----------------------------
EDGE = re.compile(r"\{DevicePhase::(\w+),\s*DevicePhase::(\w+),")
OUTER_EDGES = {m.group(2) for m in EDGE.finditer(SCHED_CODE) if m.group(1) == "Outer"}
ACTION = re.compile(r"case cmfd::CmfdOuterAction::\w+:\s*return DevicePhase::(\w+);")
EMITTED = {m.group(1) for m in ACTION.finditer(KERNEL_CODE)}
if not EMITTED:
    problems.append("CudaCmfdOuterKernels.h: cmfdOuterActionPhase has no mapping")
for phase in sorted(EMITTED):
    if phase not in OUTER_EDGES:
        problems.append("cmfdOuterActionPhase can emit Outer -> %s, which kPhaseTransitions "
                        "does not contain; the scheduler cannot execute that trajectory "
                        "(Sec 5.4)" % phase)

# --- the gates exist and are built -------------------------------------------
want(PROBE_TEXT, "--mine", "test/cmfd_outer_form_probe.cpp", "the mask must be re-derivable")
for path, text in (("test/cmfd_outer_form_probe.cpp", PROBE_TEXT),
                   ("test/cmfd_outer_replay.cpp", REPLAY_TEXT)):
    if "mineStable" not in text:
        problems.append(
            f"{path}: does not mine the host's mask before asserting.  The mask is "
            "per-build-host (0x6 on the authoring box, 0x7 on 238), so a gate scoring "
            "against the shipped literal fails on one of them for a reason unrelated to "
            "the code under test.")
    if "MINED ON THIS HOST" not in text and "mined=" not in text:
        problems.append(
            f"{path}: does not PRINT the mask it mined -- the operator needs that value "
            "to set RASBERY_CMFD_OUTER_FORMS, and Task 22's freeze needs it recorded.")
want(PROBE_TEXT, "branch coverage", "test/cmfd_outer_form_probe.cpp",
     "a mask mined on operands that never reach the guards covers part of the function")
want(REPLAY_TEXT, "kPhaseTransitions", "test/cmfd_outer_replay.cpp",
     "the emitted edges must be cross-checked against the W1 table")
want(REFERENCE_TEXT, "compiled ALONE", "test/cmfd_outer_reference.cpp",
     "the verbatim CPU quotation needs its own translation unit, or gcc "
     "common-subexpressions it with the shipped body and changes its contraction")
want(DEVICE_TEXT, "atomicMax", "test/cmfd_outer_device_replay.cu",
     "the Sec 6.12 reduction has no host equivalent, so the device arm is where it is "
     "checked at all")

for name, src in (("rasbery_cmfd_outer_form_probe", "test/cmfd_outer_form_probe.cpp"),
                  ("rasbery_cmfd_outer_replay", "test/cmfd_outer_replay.cpp"),
                  ("rasbery_cmfd_outer_device_replay", "test/cmfd_outer_device_replay.cu")):
    if name not in CMAKE_TEXT:
        problems.append(f"CMakeLists.txt: does not build {name} ({src})")
if "test/cmfd_outer_reference.cpp" not in CMAKE_TEXT:
    problems.append("CMakeLists.txt: the reference TU is not linked into the gates")
block = CMAKE_TEXT[CMAKE_TEXT.find("test/cmfd_outer_device_replay.cu"):] \
    if "test/cmfd_outer_device_replay.cu" in CMAKE_TEXT else ""
if block and "--fmad=false" not in block[:2500]:
    problems.append("CMakeLists.txt: cmfd_outer_device_replay.cu is not compiled with "
                    "--fmad=false; Class B0 is unreachable without it")

# --- repo hygiene: the 238 gate flagged an add_executable with no source -----
if "test/nodal_mine_device.cu" in CMAKE_TEXT and \
        'EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/nodal_mine_device.cu"' not in CMAKE_TEXT:
    problems.append("CMakeLists.txt: test/nodal_mine_device.cu is referenced by an "
                    "add_executable but was never committed, so RASBERY_ENABLE_TESTS=ON "
                    "fails to configure.  Guard it with if(EXISTS ...).")


def main() -> int:
    if problems:
        for problem in problems:
            print("cmfd outer kernels contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("cmfd outer kernels contract: PASS (%d kernels, mask %s, %d Outer edges)"
          % (len(GLOBALS), mask_const.group(1) if mask_const else "?", len(EMITTED)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
