#!/usr/bin/env python3
"""Constraint 35: no CUDA-only intrinsic in a shared pure body (Rev.7.1 Task 1 Step 2b).

The point is NOT portability.  Rev.7.1 Sec 1.5 is explicit that HIP and SYCL are
a trailing track with zero contribution to the speed target, and the only
hardware is one NVIDIA GPU.  The point is that the UPPER BOUND on a later port
is fixed now, for free: a shared host/device body that reaches for `__shfl_down`
or `__ldg` costs nothing today and costs a rewrite later, and the difference
between the two is a habit, not a design.  So the discipline is applied from
Task 1 even though Task 24/25 are not started.

WHAT COUNTS AS A PURE BODY.  The `*Kernel.h` headers, the scheduler core and the
arena layout calculator: files that are compiled by BOTH the host compiler and
nvcc from the same text.  A `.cu` file is not a pure body -- it is the CUDA arm,
and it is allowed to use whatever CUDA offers.

WHAT IS ALLOWED.  Warp/subgroup operations go through the GpuSubgroup.h /
GpuBackend.h wrappers (Sec 1.5 naming), not through the intrinsic.  Guards on
`__CUDA_ARCH__` / `__CUDACC__` are fine and necessary -- that is how a pure body
picks `fma` over `std::fma` (XsReconKernel.h) -- so the check is for CUDA API
CALLS and for the intrinsics themselves, not for the feature-test macros.
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Every shared pure body, present or planned.  Missing files are skipped, so the
# list can name Task 3's scheduler core before Task 3 lands; a file that appears
# is checked from that moment on.
PURE_BODIES = sorted(SRC.glob("*Kernel.h")) + [
    SRC / "GpuPhaseSchedulerCore.h",
    SRC / "GpuPhaseScheduler.h",
    SRC / "GpuPhysicsArenaLayout.h",
    SRC / "GpuSlotControl.h",
    SRC / "GpuPhysicsTypes.h",
]

# Intrinsics that exist only under nvcc.  A wrapper call site
# (rasberySubgroup*/GpuSubgroup) is allowed; the raw spelling is not.
FORBIDDEN_INTRINSICS = (
    "__shfl",
    "__ballot",
    "__activemask",
    "__syncwarp",
    "atomicAdd_block",
    "__fmaf_rn",
    "__ldg",
    "__double2hiint",
    "__double2loint",
    "cooperative_groups",
)

# Feature-test macros a pure body legitimately branches on.
ALLOWED_MACROS = ("__CUDACC__", "__CUDA_ARCH__")

CUDA_API_CALL = re.compile(r"\bcuda[A-Z]\w*\s*\(")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def main() -> int:
    problems: list[str] = []
    checked: list[str] = []
    for path in PURE_BODIES:
        if not path.is_file():
            continue
        checked.append(path.name)
        code = strip_comments(path.read_text(encoding="utf-8-sig"))
        for token in FORBIDDEN_INTRINSICS:
            if token in code:
                problems.append(
                    f"{path.name}: uses the CUDA-only intrinsic {token!r}; "
                    "route it through GpuSubgroup.h / GpuBackend.h (constraint 35)")
        for match in CUDA_API_CALL.finditer(code):
            call = match.group(0).rstrip("(").strip()
            problems.append(
                f"{path.name}: calls the CUDA API {call}(); a shared pure body must not "
                "(constraint 35)")
        # ALLOWED_MACROS are deliberately NOT scanned: `#if defined(__CUDACC__)`
        # is how a pure body picks its device spelling (XsReconKernel.h), so
        # banning them would ban the pattern the discipline depends on.

    if not checked:
        problems.append("no pure body was found to check -- the file list is stale")

    if problems:
        for problem in problems:
            print("pure body portability: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("pure body portability: PASS (%d files: %s)" % (len(checked), ", ".join(checked)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
