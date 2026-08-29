#!/usr/bin/env python3
"""GPU PPR contract (GA evaluator plan Sec 6.3 Task 10, promoted from Rev.7.1 19b).

Ten properties.  Not one of them is visible to a numerical comparison of two
runs, which is exactly why they are asserted here: every one of them is a rule
that a passing kngr_238 A/B would keep passing right up until the day it
mattered.

  1. DEFAULT OFF, AND THE OFF PATH IS THE OLD PATH.  The backend is reached
     only through PPR::resetAndDriveGpu, whose false return must be followed by
     the untouched host reset() + drive() pair.  A refactor that dropped the
     fallback would turn "no CUDA device" into "no pin power", and Fq/FdH are
     GA fitness inputs.

  2. FAIL OPEN, NEVER THROW.  Every CUDA failure in the .cu resolves to
     `return s.fail(...)`, which returns false.  A throw out of a statepoint
     boundary would take down a 64-deck batch for one slot's allocation.

  3. NO PROCESS-WIDE STATE.  The backend, its stream and every device buffer
     hang off a PPR object, which is a local of Driver::Drive().  A `static`
     anything in the .cu is the slot-0 bug class: deck 7 driving deck 0's
     buffers, correct-looking, wrong.

  4. --fmad=false ON THE TU.  The kernels restate PPR.cpp's node bodies in the
     host's statement order.  With nvcc free to contract a*b+c the deviation
     stops being attributable and the Gate A number stops meaning anything.

  5. THE CLASS IS N1 AND SAYS SO.  Device `exp` is not glibc `exp` and the
     corner sums are chunked, so the arm is not bit-identical and the header
     must not claim B0.

  6. THE BREAK TEST IS THE HOST'S.  Same sweep count, same tolerance, same
     RelativeChange shape, same "all four corners" conjunction.  A device loop
     that ran a different number of rounds would be a different reconstruction
     wearing the same receipt.

  7. THE DETERMINISTIC PARTITION IS THE HOST'S.  The corner-sum chunking must
     be Geometry.h's rasbery_det_chunks/chunk_begin, spelled the same way.  A
     partition that depended on the launch would make the arm non-deterministic
     run to run, which is the one thing N1 does not allow.

  8. THE STUB KEEPS CPU-ONLY BUILDS COMPILING.  CudaPprBackendStub.cpp is in
     the non-CUDA source list and the .cu in the CUDA one, and the stub's entry
     point returns false.

  9. PPR STILL DOES NOT FEED THE TRAJECTORY, AND THE LIBRARY STAYS CONST.  The
     digest folds (step, outer, th, efpd, keff, ppm) and nothing from the
     reconstruction; and nothing added here takes a mutable handle to the
     shared Chiffon parse (`models()` is const, `_refr_dpts[` inserts).

 10. RASBERY_GPU_PPR IS NOT AN ARM KNOB, AND SAYS WHY.  `trajectory::kArmEnv`
     is "the knobs that can move an iteration"; PPR is downstream of every one
     of them, so the knob is absent from that list -- which is also what keeps
     the arm-off binary's stdout identical to the one before this change.  The
     day PPR is allowed to feed back, kArmEnv is where that has to be declared,
     and this rule is what makes forgetting it a test failure rather than a
     silent trajectory change wearing an unchanged receipt.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

CU = SRC / "CudaPprBackend.cu"
HDR = SRC / "CudaPprBackend.h"
STUB = SRC / "CudaPprBackendStub.cpp"
PPR_H = SRC / "PPR.h"
PPR_CPP = SRC / "PPR.cpp"
DRIVER = SRC / "Driver.h"
GEOM = SRC / "Geometry.h"
CMAKE = ROOT / "CMakeLists.txt"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


CU_TEXT = read(CU)
CU_CODE = strip_comments(CU_TEXT)
HDR_TEXT = read(HDR)
STUB_CODE = strip_comments(read(STUB))
PPR_H_CODE = strip_comments(read(PPR_H))
PPR_CPP_CODE = strip_comments(read(PPR_CPP))
DRIVER_TEXT = read(DRIVER)
DRIVER_CODE = strip_comments(DRIVER_TEXT)
GEOM_CODE = strip_comments(read(GEOM))
CMAKE_TEXT = read(CMAKE)


def want(text: str, needle: str, where: str, why: str) -> None:
    if needle not in text:
        problems.append(f"{where}: missing {needle!r} -- {why}")


def forbid(text: str, needle: str, where: str, why: str) -> None:
    if needle in text:
        problems.append(f"{where}: contains {needle!r} -- {why}")


# --- 1. default off, and the off path is the old path ------------------------

want(CU_CODE, 'getenv("RASBERY_GPU_PPR")', "CudaPprBackend.cu",
     "the arm must be gated on RASBERY_GPU_PPR and default off")

if "resetAndDriveGpu(" not in DRIVER_CODE:
    problems.append("Driver.h: does not call PPR::resetAndDriveGpu -- the arm is unreachable")

# The fallback: the host pair must be inside a block guarded by the device call
# having failed.  Anchored on the guard rather than on proximity, so a longer
# call or a reordered receipt does not break the rule (617bac7's lesson).
m = re.search(r"ppr_on_device\s*=\s*pin_power_reconstruction\.resetAndDriveGpu\(", DRIVER_CODE)
if not m:
    problems.append("Driver.h: the device call must publish its verdict into ppr_on_device")
else:
    tail = DRIVER_CODE[m.end():]
    guard = re.search(r"if\s*\(\s*!\s*ppr_on_device\s*\)\s*\{(.*?)\n            \}", tail, re.S)
    if not guard:
        problems.append("Driver.h: no `if (!ppr_on_device)` host-fallback block after the "
                        "device call -- a device that declines would ship no reconstruction")
    else:
        body = guard.group(1)
        for call in ("pin_power_reconstruction.reset(",
                     "pin_power_reconstruction.drive("):
            if call not in body:
                problems.append(f"Driver.h: the fallback block does not call {call!r} -- "
                                "Fq/FdH are GA fitness inputs and must always be produced")

# reconstructPinPower is NOT part of the device arm and must stay unconditional.
if DRIVER_CODE.count("pin_power_reconstruction.reconstructPinPower(") != 1:
    problems.append("Driver.h: reconstructPinPower must be called exactly once, outside "
                    "the on/off branch -- it runs on the host in both arms")

# --- 2. fail open, never throw ------------------------------------------------

forbid(CU_CODE, "throw ", "CudaPprBackend.cu",
       "a throw at a statepoint boundary takes down every slot in the process")
if "return false;" not in CU_CODE:
    problems.append("CudaPprBackend.cu: no `return false` -- the fallback contract needs one")
# Every cudaMalloc / cudaMemcpyAsync / cudaStream* result must be checked.  The
# cheap structural proxy: the failure helper exists, returns false, and is
# referenced at least once per allocation-or-transfer helper.
want(CU_CODE, "bool fail(const char* what, cudaError_t rc)", "CudaPprBackend.cu",
     "one failure path, which releases the buffers and returns false")
if "release();" not in CU_CODE:
    problems.append("CudaPprBackend.cu: fail() must release the device buffers -- a half-built "
                    "instance that stays alive will be retried next statepoint")

# --- 3. no process-wide state -------------------------------------------------

# `static const` / `constexpr` are fine (they are read-only).  A mutable static
# is not.
for mline in re.finditer(r"^\s*static\s+(?!const|constexpr)([^\n;]*);", CU_CODE, re.M):
    problems.append(f"CudaPprBackend.cu: mutable static state {mline.group(1).strip()!r} -- "
                    "one batch slot would drive another's buffers")
want(PPR_H_CODE, "std::unique_ptr<PprBackend> _gpu;", "PPR.h",
     "the backend must be a per-PPR member; PPR is a local of Driver::Drive(), "
     "which is what makes it per-slot")
forbid(PPR_H_CODE, "static std::unique_ptr<PprBackend>", "PPR.h",
       "a static backend is one device buffer set shared by every deck in the process")

# --- 4. --fmad=false on the TU ------------------------------------------------

if not re.search(r"CudaPprBackend\.cu\"\s*\n\s*PROPERTIES COMPILE_OPTIONS \"--fmad=false\"",
                 CMAKE_TEXT):
    problems.append("CMakeLists.txt: src/CudaPprBackend.cu must carry --fmad=false -- "
                    "without it the deviation is nvcc's scheduling, not exp")

# --- 5. the class is N1 and says so ------------------------------------------

if "N1" not in HDR_TEXT:
    problems.append("CudaPprBackend.h: must record the gate class (N1)")
if re.search(r"\bB0\b", HDR_TEXT) and "NOT B0" not in HDR_TEXT and "not B0" not in HDR_TEXT:
    problems.append("CudaPprBackend.h: mentions B0 without disclaiming it -- device exp and "
                    "the chunked corner sums make bit-identity false")

# --- 6. the break test is the host's -----------------------------------------

host_sweeps = re.search(r"kSourceSweepsPerIteration\s*=\s*(\d+)", PPR_CPP_CODE)
dev_sweeps = re.search(r"kSourceSweepsPerIteration\s*=\s*(\d+)", CU_CODE)
if not host_sweeps or not dev_sweeps:
    problems.append("kSourceSweepsPerIteration missing from PPR.cpp or CudaPprBackend.cu")
elif host_sweeps.group(1) != dev_sweeps.group(1):
    problems.append(f"sweep count differs: PPR.cpp {host_sweeps.group(1)} vs "
                    f"CudaPprBackend.cu {dev_sweeps.group(1)}")

host_tol = re.search(r"kCornerFluxTolerance\s*=\s*([0-9.eE+-]+)", PPR_CPP_CODE)
dev_tol = re.search(r"kCornerFluxTolerance\s*=\s*([0-9.eE+-]+)", CU_CODE)
if not host_tol or not dev_tol:
    problems.append("kCornerFluxTolerance missing from PPR.cpp or CudaPprBackend.cu")
elif float(host_tol.group(1)) != float(dev_tol.group(1)):
    problems.append(f"corner-flux tolerance differs: PPR.cpp {host_tol.group(1)} vs "
                    f"CudaPprBackend.cu {dev_tol.group(1)}")

want(CU_CODE, "(previous != 0.0) ? fabs((current - previous) / previous) : 1.0",
     "CudaPprBackend.cu",
     "RelativeChange must be PPR.cpp's, including the previous==0 -> 1.0 first round")
if not re.search(r"err_nw < kCornerFluxTolerance && err_sw < kCornerFluxTolerance &&\s*"
                 r"err_ne < kCornerFluxTolerance && err_se < kCornerFluxTolerance",
                 CU_CODE):
    problems.append("CudaPprBackend.cu: the break test must require ALL FOUR corners under "
                    "tolerance, exactly as PPR::drive does")

# --- 7. the deterministic partition is the host's -----------------------------

host_chunk = re.search(r"RASBERY_DET_CHUNK_TARGET\s*=\s*(\d+)", GEOM_CODE)
dev_chunk = re.search(r"kDetChunkTarget\s*=\s*(\d+)", CU_CODE)
if not host_chunk or not dev_chunk:
    problems.append("chunk target missing from Geometry.h or CudaPprBackend.cu")
elif host_chunk.group(1) != dev_chunk.group(1):
    problems.append(f"deterministic chunk target differs: Geometry.h {host_chunk.group(1)} "
                    f"vs CudaPprBackend.cu {dev_chunk.group(1)}")
want(CU_CODE, "static_cast<int>((nn * c) / nchunk)", "CudaPprBackend.cu",
     "chunk_begin must be Geometry.h's, or the partition is not the host's")
forbid(CU_CODE, "atomicAdd", "CudaPprBackend.cu",
       "an atomic reduction is order-dependent, i.e. not run-to-run deterministic")

# --- 8. the stub keeps CPU-only builds compiling ------------------------------

want(CMAKE_TEXT, "src/CudaPprBackendStub.cpp", "CMakeLists.txt",
     "the no-CUDA build needs the stub in its source list")
want(CMAKE_TEXT, "src/CudaPprBackend.cu", "CMakeLists.txt",
     "the CUDA build needs the real TU in its source list")
if not re.search(r"bool PprBackend::resetAndDrive\([^)]*\)\s*\{\s*return false;\s*\}",
                 STUB_CODE, re.S):
    problems.append("CudaPprBackendStub.cpp: resetAndDrive must return false so the call "
                    "site needs no #ifdef")

# --- 9. PPR still does not feed the trajectory; the library stays const --------

traj = re.search(r"sp_traj\.step\(([^)]*)\)", DRIVER_CODE)
if not traj:
    problems.append("Driver.h: the trajectory digest step() call vanished")
else:
    args = traj.group(1)
    for forbidden in ("fqp", "frp", "PinPower", "pin_power"):
        if forbidden in args:
            problems.append(f"Driver.h: the trajectory digest folds {forbidden!r} -- PPR "
                            "outputs must stay outside the digest, or every PPR arm becomes "
                            "a trajectory change")

for token in ("_refr_dpts[", "models()[", "auto& model = _xs.models()"):
    if token in PPR_CPP_CODE and "const auto& model = _xs.models()" not in PPR_CPP_CODE:
        problems.append(f"PPR.cpp: {token!r} without a const binding -- the Chiffon parse is "
                        "shared by every Driver in the process (XsLibrary.h)")

# The receipt the plan asks for.
for field in ('\\"statepoints\\":', '\\"device\\":', '\\"host_fallbacks\\":', '\\"wall_ms\\":'):
    if field not in DRIVER_CODE:
        problems.append(f"Driver.h: the [RASBERY][PPR_GPU] receipt is missing {field}")
want(DRIVER_CODE, "[RASBERY][PPR_GPU]", "Driver.h", "the arm must publish a receipt")

# --- 10. RASBERY_GPU_PPR is not an arm knob ----------------------------------

arm = re.search(r"kArmEnv\[\]\s*=\s*\{(.*?)\};", DRIVER_CODE, re.S)
if not arm:
    problems.append("Driver.h: trajectory::kArmEnv vanished -- the trajectory receipt's "
                    "arm half is what makes an A/B comparable")
elif "RASBERY_GPU_PPR" in arm.group(1):
    problems.append("Driver.h: RASBERY_GPU_PPR is listed in trajectory::kArmEnv -- that list "
                    "is 'the knobs that can move an iteration', and PPR runs after the "
                    "statepoint's last SolveLoop and feeds nothing back.  If it now DOES "
                    "feed back, say so here and in the header, and expect the arm-off "
                    "binary to stop being stdout-identical to its predecessor.")

# And the reason must be written down where the list is, not only here -- in the
# RAW text, since the reason is a comment.
raw_arm = DRIVER_TEXT.find("kArmEnv[]")
if raw_arm >= 0 and "RASBERY_GPU_PPR" not in DRIVER_TEXT[max(0, raw_arm - 2000):raw_arm]:
    problems.append("Driver.h: kArmEnv does not explain why RASBERY_GPU_PPR is absent -- an "
                    "unexplained absence reads as an oversight and gets 'fixed'")


def main() -> int:
    if problems:
        print("PPR GPU contract: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PPR GPU contract: OK (10 properties)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
