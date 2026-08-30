#!/usr/bin/env python3
"""GPU PPR contract (GA evaluator plan Sec 6.3 Task 10; bottleneck plan WP6).

Twenty properties.  Not one of them is visible to a numerical comparison of two
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
     the non-CUDA source list and the .cu in the CUDA one, the stub's entry
     point returns false, and it defines EVERY receipt accessor the header
     declares -- a missing one is a link error nobody sees until the CPU build.

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

--- WP6 stage B: the stopping test on the device ----------------------------

 11. THE DEVICE FOLD IS THE HOST'S FOLD, ASSOCIATION AND ALL.  One thread per
     corner walking chunk 0..nchunk-1 ascending into a double started at 0.0 --
     byte for byte the loop the host ran over the D2H'd partials.  A tree
     reduction over 256 chunks would be faster and would move the sum, so the
     shape is asserted rather than left to whoever optimises next.

 12. THE SURPLUS ROUNDS WRITE NOTHING.  `device_stream` enqueues `niter` bodies
     and lets the device flag turn the ones after convergence into no-ops.
     That is only exact while EVERY kernel of the body reads the flag first, so
     every one of the four is a `template <bool kGuarded>` whose first
     statement is the guard, and the reset half instantiates the SAME kernels
     with `false` because there is no loop there to stop.

 13. THE PER-ROUND SYNCHRONISE IS GONE, AND COUNTED.  There is exactly one
     `cudaStreamSynchronize` call site in the file, it is inside the helper
     that increments `n_host_syncs`, and the receipt publishes
     host_syncs_per_statepoint -- so "did the sync go away" is a number in the
     receipt and not a claim in a header.

 14. NOTHING PER-STATEPOINT IS BAKED BY VALUE.  A captured graph freezes its
     kernel node parameters, so `reigv` may not travel in DevCtx: it lives in
     the device loop-state block and the body reads it through a pointer.  A
     by-value reigv would replay the FIRST statepoint's eigenvalue forever,
     finite and plausible and wrong.

 15. THE GRAPH KEY IS THE WHOLE BAKED OPERAND SET, and a refusal falls back to
     the stream arm rather than to the host.  The CUDA-13 cudaGraphAddNode
     signature split must be spelled exactly as GpuOuterWhile.h spells it.

--- WP6 stage C: canonical input --------------------------------------------

 16. THE BORROW IS ALL-OR-NOTHING, DEFAULT OFF, AND OFFERED PER STATEPOINT.
     A borrowed jnet paired with an uploaded phif is two outer iterations
     blended into one reconstruction; the rule is enforced at all three layers
     (Driver's offer, PPR's re-check, the backend's own test) because the cost
     of one layer forgetting is silent.

 17. VERIFY ACTUALLY VERIFIES.  `RASBERY_GPU_PPR_CANONICAL=verify` uploads the
     host arrays AND compares them bitwise against the borrowed buffers on the
     device, reporting the count.  A compare that used `!=` would call two NaNs
     different; a compare with a tolerance would answer an easier question.

 18. A GENERATION GATE MUST NOT BE ABLE TO SUPPRESS FOREVER.  chif/crdf upload
     when the caller cannot vouch for a generation (0), and the crdf counter is
     bumped AT THE WRITE in PPR.cpp -- both of them -- not at a caller.

--- WP6 stage E: the batch arena stays conditional --------------------------

 19. ALLOCATION IS ONCE PER SLOT, AND THE RECEIPT PROVES IT.  Every cudaMalloc
     is inside ensureShape, every one is counted, and a re-shape is counted
     separately -- `reallocations > 0` is how a run says it allocated inside
     the statepoint loop, which is the condition under which the plan's
     CudaPprArena becomes necessary rather than a guess.

 20. THE RECEIPT IS SCHEMA 2 AND CARRIES EVERY STAGE'S NUMBER.  A stage whose
     effect is not in the receipt cannot be priced on 238, and an unpriced
     stage does not get promoted.

Run with no arguments; the negative controls run every time (a contract test
that has never been seen to fail is a comment).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

FILES = {
    "cu": SRC / "CudaPprBackend.cu",
    "hdr": SRC / "CudaPprBackend.h",
    "stub": SRC / "CudaPprBackendStub.cpp",
    "ppr_h": SRC / "PPR.h",
    "ppr_cpp": SRC / "PPR.cpp",
    "driver": SRC / "Driver.h",
    "geom": SRC / "Geometry.h",
    "while_h": SRC / "GpuOuterWhile.h",
    "cmake": ROOT / "CMakeLists.txt",
}


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def load() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, path in FILES.items():
        out[key] = path.read_text(encoding="utf-8-sig") if path.is_file() else ""
    return out


def check(raw: dict[str, str]) -> list[str]:
    problems: list[str] = []

    for key, path in FILES.items():
        if not raw.get(key):
            problems.append(f"missing or empty file: {path.relative_to(ROOT)}")

    CU_TEXT = raw["cu"]
    CU_CODE = strip_comments(CU_TEXT)
    HDR_TEXT = raw["hdr"]
    HDR_CODE = strip_comments(HDR_TEXT)
    STUB_CODE = strip_comments(raw["stub"])
    PPR_H_CODE = strip_comments(raw["ppr_h"])
    PPR_CPP_CODE = strip_comments(raw["ppr_cpp"])
    DRIVER_TEXT = raw["driver"]
    DRIVER_CODE = strip_comments(DRIVER_TEXT)
    GEOM_CODE = strip_comments(raw["geom"])
    WHILE_CODE = strip_comments(raw["while_h"])
    CMAKE_TEXT = raw["cmake"]

    def want(text: str, needle: str, where: str, why: str) -> None:
        if needle not in text:
            problems.append(f"{where}: missing {needle!r} -- {why}")

    def forbid(text: str, needle: str, where: str, why: str) -> None:
        if needle in text:
            problems.append(f"{where}: contains {needle!r} -- {why}")

    # --- 1. default off, and the off path is the old path --------------------

    want(CU_CODE, 'getenv("RASBERY_GPU_PPR")', "CudaPprBackend.cu",
         "the arm must be gated on RASBERY_GPU_PPR and default off")

    if "resetAndDriveGpu(" not in DRIVER_CODE:
        problems.append("Driver.h: does not call PPR::resetAndDriveGpu -- the arm is unreachable")

    # The fallback: the host pair must be inside a block guarded by the device
    # call having failed.  Anchored on the guard rather than on proximity, so a
    # longer call or a reordered receipt does not break the rule (617bac7).
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

    # reconstructPinPower is NOT part of the reset+drive arm and must stay
    # unconditional at the call site.
    if DRIVER_CODE.count("pin_power_reconstruction.reconstructPinPower(") != 1:
        problems.append("Driver.h: reconstructPinPower must be called exactly once, outside "
                        "the on/off branch -- it runs in both arms")

    # --- 2. fail open, never throw -------------------------------------------

    forbid(CU_CODE, "throw ", "CudaPprBackend.cu",
           "a throw at a statepoint boundary takes down every slot in the process")
    if "return false;" not in CU_CODE:
        problems.append("CudaPprBackend.cu: no `return false` -- the fallback contract needs one")
    want(CU_CODE, "bool fail(const char* what, cudaError_t rc)", "CudaPprBackend.cu",
         "one failure path, which releases the buffers and returns false")
    if "release();" not in CU_CODE:
        problems.append("CudaPprBackend.cu: fail() must release the device buffers -- a half-built "
                        "instance that stays alive will be retried next statepoint")

    # --- 3. no process-wide state --------------------------------------------

    for mline in re.finditer(r"^\s*static\s+(?!const|constexpr)([^\n;]*);", CU_CODE, re.M):
        problems.append(f"CudaPprBackend.cu: mutable static state {mline.group(1).strip()!r} -- "
                        "one batch slot would drive another's buffers")
    want(PPR_H_CODE, "std::unique_ptr<PprBackend> _gpu;", "PPR.h",
         "the backend must be a per-PPR member; PPR is a local of Driver::Drive(), "
         "which is what makes it per-slot")
    forbid(PPR_H_CODE, "static std::unique_ptr<PprBackend>", "PPR.h",
           "a static backend is one device buffer set shared by every deck in the process")

    # --- 4. --fmad=false on the TU -------------------------------------------

    _bitexact = re.search(r'set\(RASBERY_BITEXACT_CUDA_OPTS\s+"([^"]*)"\s*\)', CMAKE_TEXT)
    _ppr_opts = re.search(r'CudaPprBackend\.cu"\s*\n\s*PROPERTIES COMPILE_OPTIONS "([^"]*)"',
                          CMAKE_TEXT)
    if _ppr_opts is None or "--fmad=false" not in _ppr_opts.group(1).replace(
            "${RASBERY_BITEXACT_CUDA_OPTS}", _bitexact.group(1) if _bitexact else ""):
        problems.append("CMakeLists.txt: src/CudaPprBackend.cu must carry --fmad=false -- "
                        "without it the deviation is nvcc's scheduling, not exp")

    # --- 5. the class is N1 and says so --------------------------------------

    if "N1" not in HDR_TEXT:
        problems.append("CudaPprBackend.h: must record the gate class (N1)")
    if re.search(r"\bB0\b", HDR_TEXT) and "NOT B0" not in HDR_TEXT and "not B0" not in HDR_TEXT:
        problems.append("CudaPprBackend.h: mentions B0 without disclaiming it -- device exp and "
                        "the chunked corner sums make bit-identity false")

    # --- 6. the break test is the host's -------------------------------------

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
    conj = re.compile(r"err_nw < kCornerFluxTolerance && err_sw < kCornerFluxTolerance &&\s*"
                      r"err_ne < kCornerFluxTolerance && err_se < kCornerFluxTolerance")
    n_conj = len(conj.findall(CU_CODE))
    if n_conj < 2:
        problems.append("CudaPprBackend.cu: the break test must require ALL FOUR corners under "
                        "tolerance, exactly as PPR::drive does -- once in the device fold kernel "
                        f"and once in the host_sync arm, found {n_conj}")

    # --- 7. the deterministic partition is the host's -------------------------

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

    # --- 8. the stub keeps CPU-only builds compiling --------------------------

    want(CMAKE_TEXT, "src/CudaPprBackendStub.cpp", "CMakeLists.txt",
         "the no-CUDA build needs the stub in its source list")
    want(CMAKE_TEXT, "src/CudaPprBackend.cu", "CMakeLists.txt",
         "the CUDA build needs the real TU in its source list")
    if not re.search(r"bool PprBackend::resetAndDrive\([^)]*\)\s*\{\s*return false;\s*\}",
                     STUB_CODE, re.S):
        problems.append("CudaPprBackendStub.cpp: resetAndDrive must return false so the call "
                        "site needs no #ifdef")
    # Every accessor the header declares must exist in the stub, or the CPU-only
    # build fails to LINK -- which no contract test and no GPU build would see.
    declared = set(re.findall(r"\]\]\s*(?:const char\*|unsigned long long|double|"
                              r"const std::string&)\s+(\w+)\(\) const;", HDR_CODE))
    for name in sorted(declared):
        if f"PprBackend::{name}()" not in STUB_CODE:
            problems.append(f"CudaPprBackendStub.cpp: does not define {name}() -- the CPU-only "
                            "build would fail to link, and nothing else in this tree builds it")

    # --- 9. PPR still does not feed the trajectory; the library stays const ----

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

    for field in ('\\"statepoints\\":', '\\"device\\":', '\\"host_fallbacks\\":', '\\"wall_ms\\":'):
        if field not in DRIVER_CODE:
            problems.append(f"Driver.h: the [RASBERY][PPR_GPU] receipt is missing {field}")
    want(DRIVER_CODE, "[RASBERY][PPR_GPU]", "Driver.h", "the arm must publish a receipt")

    # --- 10. RASBERY_GPU_PPR is not an arm knob -------------------------------

    arm = re.search(r"kArmEnv\[\]\s*=\s*\{(.*?)\};", DRIVER_CODE, re.S)
    if not arm:
        problems.append("Driver.h: trajectory::kArmEnv vanished -- the trajectory receipt's "
                        "arm half is what makes an A/B comparable")
    else:
        for knob in ("RASBERY_GPU_PPR", "RASBERY_GPU_PPR_DEVICE_LOOP",
                     "RASBERY_GPU_PPR_GRAPH", "RASBERY_GPU_PPR_CANONICAL"):
            if knob in arm.group(1):
                problems.append(
                    f"Driver.h: {knob} is listed in trajectory::kArmEnv -- that list is 'the "
                    "knobs that can move an iteration', and PPR runs after the statepoint's "
                    "last SolveLoop and feeds nothing back.  If it now DOES feed back, say so "
                    "here and in the header, and expect the arm-off binary to stop being "
                    "stdout-identical to its predecessor.")

    raw_arm = DRIVER_TEXT.find("kArmEnv[]")
    if raw_arm >= 0 and "RASBERY_GPU_PPR" not in DRIVER_TEXT[max(0, raw_arm - 2000):raw_arm]:
        problems.append("Driver.h: kArmEnv does not explain why RASBERY_GPU_PPR is absent -- an "
                        "unexplained absence reads as an oversight and gets 'fixed'")

    # --- 11. the device fold is the host's fold -------------------------------

    want(CU_CODE, "__global__ void kCornerFoldAndCheck(DevCtx x)", "CudaPprBackend.cu",
         "WP6 stage B moves the fold and the break test onto the device")
    want(CU_CODE, "for (int c = 0; c < nchunk; ++c) s += partials[t * nchunk + c];",
         "CudaPprBackend.cu",
         "the device fold must walk chunk 0..nchunk-1 ASCENDING into a double started at 0.0 -- "
         "that is the host's association, and a tree reduction over 256 chunks is not")
    want(CU_CODE, "kCornerFoldAndCheck<<<1, 4, 0, st>>>(x);", "CudaPprBackend.cu",
         "one thread per corner: four independent sequential folds, no cross-corner blending")
    want(CU_CODE, "__host__ __device__ inline double relativeChange", "CudaPprBackend.cu",
         "ONE RelativeChange, shared by the device fold and the host_sync arm -- two spellings "
         "of the break test are two chances to spell it differently")

    # --- 12. the surplus rounds write nothing ---------------------------------

    want(CU_CODE, "__device__ inline bool pprHalted(const DevCtx& x)", "CudaPprBackend.cu",
         "the one read that makes an already-enqueued body a no-op")
    for kernel in ("kUpdateSource", "kUpdateFused", "kUpdateCorner", "kCornerPartials"):
        pat = re.compile(r"template <bool kGuarded>\s*\n__global__ void " + kernel +
                         r"\(DevCtx x\) \{\s*\n\s*if \(kGuarded && pprHalted\(x\)\) return;")
        if not pat.search(CU_CODE):
            problems.append(
                f"CudaPprBackend.cu: {kernel} is not a `template <bool kGuarded>` whose FIRST "
                "statement is the halt guard -- the device_stream arm's exactness is exactly "
                "the claim that every surplus body writes nothing")
    want(CU_CODE, "kUpdateSource<false><<<blocks_n, threads, 0, s.stream>>>(x);",
         "CudaPprBackend.cu",
         "reset() instantiates the SAME kernel with the guard off: there is no loop there, and "
         "a guarded reset would read a flag the statepoint has not yet defined")
    want(CU_CODE, "kUpdateFused<true><<<blocks_ng, threads, 0, st>>>(x);", "CudaPprBackend.cu",
         "the Picard body instantiates the guarded form")
    if "if (pprHalted(x)) return;" not in CU_CODE:
        problems.append("CudaPprBackend.cu: kCornerFoldAndCheck must itself be guarded, or a "
                        "surplus round would advance `iters` past the host's count")

    # --- 13. the per-round synchronise is gone, and counted -------------------

    n_sync = CU_CODE.count("cudaStreamSynchronize(")
    if n_sync != 1:
        problems.append(f"CudaPprBackend.cu: {n_sync} cudaStreamSynchronize call sites -- there "
                        "must be exactly one, inside the helper that counts it, or "
                        "host_syncs_per_statepoint stops being the truth")
    want(CU_CODE, "++s.n_host_syncs;", "CudaPprBackend.cu",
         "every synchronise must be counted where it is issued")
    want(CU_CODE, "enum class LoopArm : int { HostSync = 0, DeviceStream, DeviceGraph };",
         "CudaPprBackend.cu", "the three arms, named")
    want(CU_CODE, 'getenv("RASBERY_GPU_PPR_DEVICE_LOOP")', "CudaPprBackend.cu",
         "the A/B control must exist: RASBERY_GPU_PPR_DEVICE_LOOP=0 restores the host-sync loop")

    # --- 14. nothing per-statepoint is baked by value -------------------------

    forbid(CU_CODE, "x.reigv", "CudaPprBackend.cu",
           "reigv may not travel in DevCtx: a captured graph bakes kernel parameters, so a "
           "by-value reigv replays the first statepoint's eigenvalue forever")
    want(CU_CODE, "x.loop->reigv", "CudaPprBackend.cu",
         "the body must read reigv from the device loop-state block")
    want(CU_CODE, "struct PprLoopState {", "CudaPprBackend.cu",
         "one block holds every per-statepoint scalar the captured body reads")

    # --- 15. the graph key, and the refusal -----------------------------------

    want(CU_CODE, "std::memcmp(&s.graph_ctx, &x, sizeof(DevCtx))", "CudaPprBackend.cu",
         "the instantiation key is the WHOLE baked operand set; a borrow that engaged between "
         "statepoints moves phif/phis/jnet and the old graph would read the wrong buffer")
    for text, name in ((CU_CODE, "CudaPprBackend.cu"), (WHILE_CODE, "GpuOuterWhile.h")):
        want(text, "cudaGraphAddNode(out_node, parent, deps, nullptr, ndeps, &np)", name,
             "the CUDA-13 signature split must be spelled identically in both files")
        want(text, "CUDART_VERSION >= 13000", name, "and guarded identically")
    want(CU_CODE, "s.arm           = LoopArm::DeviceStream;", "CudaPprBackend.cu",
         "a graph refusal falls back to the stream arm -- never to the host, which would make "
         "a capture failure look like a physics decision")
    want(CU_CODE, "cudaGraphSetConditional(\n        handle, (st->converged == 0 && st->iters < st->niter) ? 1u : 0u);",
         "CudaPprBackend.cu",
         "the WHILE predicate is PPR::drive's loop header and nothing else")

    # --- 16. the borrow is all-or-nothing, default off, offered per statepoint -

    want(HDR_CODE, 'getenv("RASBERY_GPU_PPR_CANONICAL")', "CudaPprBackend.h",
         "the borrow is its own knob and defaults off")
    if not re.search(r"if \(v == nullptr\) return CanonicalMode::Off;", HDR_CODE):
        problems.append("CudaPprBackend.h: canonicalModeFromEnv must return Off when the "
                        "variable is unset -- the borrow's premise is a 238 measurement, not a "
                        "default")
    want(CU_CODE, "const bool have_set = step.dev_phif != nullptr && step.dev_phis != nullptr &&",
         "CudaPprBackend.cu",
         "all three or none: a borrowed jnet beside an uploaded phif is two outer iterations")
    want(PPR_CPP_CODE, "_canonical.complete()", "PPR.cpp",
         "PPR re-checks completeness rather than trusting the caller")
    want(DRIVER_CODE, "seg.canonicalNodalBound()", "Driver.h",
         "the offer is conditioned on the SEGMENT saying the binding is live, not on the mere "
         "existence of arena pointers")
    want(DRIVER_CODE, "pin_power_reconstruction.adoptCanonicalDeviceInputs(ppr_canon);",
         "Driver.h",
         "the offer is made -- and withdrawn -- every statepoint; silence would leave the "
         "previous statepoint's offer standing")

    # --- 17. verify actually verifies -----------------------------------------

    want(CU_CODE, "__global__ void kCanonicalCompare(", "CudaPprBackend.cu",
         "verify mode must compare on the device, not assume")
    want(CU_CODE, "__double_as_longlong(borrowed[i]) != __double_as_longlong(uploaded[i])",
         "CudaPprBackend.cu",
         "the comparison is BITWISE: `!=` calls two NaNs different and a tolerance answers an "
         "easier question")
    want(CU_CODE, "if (!borrow || verify) {", "CudaPprBackend.cu",
         "verify uploads the host arrays as well, or there is nothing to compare against")

    # --- 18. a generation gate must not suppress forever ----------------------

    want(CU_CODE, "(step.chif_generation == 0)", "CudaPprBackend.cu",
         "a caller that cannot vouch for a generation must get an upload, not silence")
    want(CU_CODE, "(step.crdf_generation == 0)", "CudaPprBackend.cu",
         "same rule for crdf")
    n_bump = PPR_CPP_CODE.count("++_crdf_generation;")
    if n_bump != 2:
        problems.append(f"PPR.cpp: {n_bump} `++_crdf_generation` sites -- the counter is bumped "
                        "AT THE WRITE, and _crdf is written in both reset() and "
                        "resetAndDriveGpu(); a bump at the caller is a policy a second writer "
                        "can silently break")

    # --- 19. allocation is once per slot, and the receipt proves it -----------

    want(CU_CODE, "++n_allocations;", "CudaPprBackend.cu", "every cudaMalloc is counted")
    want(CU_CODE, "if (n_shapes > 0) ++n_reallocations;", "CudaPprBackend.cu",
         "a re-shape is counted separately: `reallocations > 0` is how a run says it allocated "
         "inside the statepoint loop, which is the plan's trigger for CudaPprArena")
    body = re.search(r"bool ensureShape\(const ppr::GeomView& g\) \{(.*?)\n    \}\n", CU_CODE, re.S)
    if not body:
        problems.append("CudaPprBackend.cu: ensureShape vanished")
    else:
        outside = CU_CODE.replace(body.group(1), "")
        if "cudaMalloc" in outside:
            problems.append("CudaPprBackend.cu: a cudaMalloc outside ensureShape -- per-slot "
                            "allocation must happen once per SHAPE, not once per statepoint")

    # --- 20. the receipt is schema 2 and carries every stage's number ---------

    receipt = re.search(r'"  \[RASBERY\]\[PPR_GPU\] \{\{(.*?)\\n",', DRIVER_CODE, re.S)
    if not receipt:
        problems.append("Driver.h: the [RASBERY][PPR_GPU] receipt format string vanished")
    elif '\\"schema_version\\":2' not in receipt.group(1):
        problems.append("Driver.h: the [RASBERY][PPR_GPU] receipt is not schema_version 2 -- "
                        "WP6 added fields, and a reader that keyed on schema 1 must be told")
    for field in ('\\"loop_arm\\":', '\\"host_syncs\\":', '\\"host_syncs_per_statepoint\\":',
                  '\\"graph_launches\\":', '\\"graph_builds\\":', '\\"graph_refusal\\":',
                  '\\"canonical_mode\\":', '\\"canonical_statepoints\\":',
                  '\\"canonical_mismatch\\":', '\\"h2d_bytes\\":', '\\"h2d_bytes_elided\\":',
                  '\\"d2h_bytes\\":', '\\"allocations\\":', '\\"reallocations\\":'):
        if field not in DRIVER_CODE:
            problems.append(f"Driver.h: the [RASBERY][PPR_GPU] receipt is missing {field} -- a "
                            "stage whose effect is not in the receipt cannot be priced on 238")

    return problems


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------
#
# Each entry breaks ONE rule and must produce at least one problem.  A contract
# test whose failure has never been observed is a comment: these are what make
# the twenty properties above assertions.

CONTROLS: list[tuple[str, str, str, str]] = [
    # (name, file key, find, replace)
    ("fallback removed", "driver",
     "pin_power_reconstruction.reset(1.0 / eigv", "pin_power_reconstruction.noreset(1.0 / eigv"),
    ("device fold reversed", "cu",
     "for (int c = 0; c < nchunk; ++c) s += partials[t * nchunk + c];",
     "for (int c = nchunk - 1; c >= 0; --c) s += partials[t * nchunk + c];"),
    ("fold kernel deleted", "cu",
     "__global__ void kCornerFoldAndCheck(DevCtx x)",
     "__global__ void kCornerFoldAndCheckX(DevCtx x)"),
    ("body kernel unguarded", "cu",
     "template <bool kGuarded>\n__global__ void kUpdateCorner(DevCtx x) {\n    if (kGuarded && pprHalted(x)) return;",
     "template <bool kGuarded>\n__global__ void kUpdateCorner(DevCtx x) {"),
    ("reset guarded by accident", "cu",
     "kUpdateSource<false><<<blocks_n, threads, 0, s.stream>>>(x);",
     "kUpdateSource<true><<<blocks_n, threads, 0, s.stream>>>(x);"),
    ("a second uncounted sync", "cu",
     "    cudaEventRecord(s.ev_stop, s.stream);",
     "    cudaStreamSynchronize(s.stream);\n    cudaEventRecord(s.ev_stop, s.stream);"),
    ("reigv baked by value", "cu",
     "const double reigv = x.loop->reigv;", "const double reigv = x.reigv;"),
    ("graph key dropped", "cu",
     "std::memcmp(&s.graph_ctx, &x, sizeof(DevCtx))", "false"),
    ("graph refusal falls to host", "cu",
     "s.arm           = LoopArm::DeviceStream;", "s.arm = LoopArm::HostSync;"),
    ("borrow accepts a partial set", "cu",
     "const bool have_set = step.dev_phif != nullptr && step.dev_phis != nullptr &&",
     "const bool have_set = step.dev_phif != nullptr ||"),
    ("borrow defaults on", "hdr",
     "if (v == nullptr) return CanonicalMode::Off;", "if (v == nullptr) return CanonicalMode::Borrow;"),
    ("offer no longer conditioned on the segment", "driver",
     "seg.canonicalNodalBound()", "true"),
    ("verify compares with !=", "cu",
     "__double_as_longlong(borrowed[i]) != __double_as_longlong(uploaded[i])",
     "borrowed[i] != uploaded[i]"),
    ("generation gate can suppress forever", "cu",
     "(step.chif_generation == 0)", "(false)"),
    ("crdf bump moved off the write", "ppr_cpp",
     "        ++_crdf_generation;\n    }\n\n    const size_t nng", "    }\n\n    const size_t nng"),
    ("allocation inside the statepoint path", "cu",
     "    cudaEventRecord(s.ev_start, s.stream);",
     "    { void* leak = nullptr; cudaMalloc(&leak, 8); }\n    cudaEventRecord(s.ev_start, s.stream);"),
    ("realloc counter dropped", "cu",
     "if (n_shapes > 0) ++n_reallocations;", ""),
    ("receipt loses a stage number", "driver",
     '\\"h2d_bytes_elided\\":{}', '\\"h2d_bytes_elidedX\\":{}'),
    ("schema not bumped", "driver",
     '[RASBERY][PPR_GPU] {{\\"schema_version\\":2',
     '[RASBERY][PPR_GPU] {{\\"schema_version\\":1'),
    ("stub misses an accessor", "stub",
     "unsigned long long PprBackend::h2dBytes() const { return 0; }", ""),
    ("PPR knob smuggled into kArmEnv", "driver",
     '    "RASBERY_GPU_CMFD_SWEEP",',
     '    "RASBERY_GPU_CMFD_SWEEP",\n    "RASBERY_GPU_PPR_CANONICAL",'),
    ("an atomic reduction returns", "cu",
     "counts[static_cast<long long>(slot) * nchunk + c] = bad;",
     "atomicAdd(counts, bad);"),
]


def selftest(raw: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for name, key, find, repl in CONTROLS:
        if find not in raw[key]:
            failures.append(f"negative control {name!r}: its anchor {find!r} is not in "
                            f"{FILES[key].name} -- the control no longer controls anything")
            continue
        mutated = dict(raw)
        mutated[key] = raw[key].replace(find, repl, 1)
        if not check(mutated):
            failures.append(f"negative control {name!r}: mutating {FILES[key].name} produced "
                            "NO problem -- the corresponding property is not actually asserted")
    return failures


def main() -> int:
    raw = load()
    problems = check(raw)
    controls = selftest(raw)
    if problems or controls:
        print("PPR GPU contract: FAIL")
        for p in problems:
            print(f"  - {p}")
        for c in controls:
            print(f"  ! {c}")
        return 1
    print(f"PPR GPU contract: OK (20 properties, {len(CONTROLS)} negative controls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
