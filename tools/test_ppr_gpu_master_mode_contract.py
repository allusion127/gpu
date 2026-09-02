#!/usr/bin/env python3
"""WP6 stage F: RASBERY_PPR_MODE=master on the device, and the refusal ladder.

WHAT WENT WRONG, IN ONE LINE.  `PPR::resetAndDriveGpu` opened with
`if (_mode_master) return false;`, and RASBERY_PPR_MODE=master is the
PRODUCTION configuration -- it is what gives Gate B pin RMS 0.238 % against the
default mode's 0.522 %.  So the GPU PPR arm declined 35 statepoints out of 35 on
every production run, and the only thing the receipt said was
`host_fallbacks:35`, a number that reads identically for "no CUDA device", "this
deck is not two-group" and "the mode the campaign actually runs was never
ported".  Measured on host 238 at 73f8627: master alone flips host_fallbacks
0 -> 35; RASBERY_PC_MODE=decart alone leaves it at 0 with iterations 816, which
is how we know decart is a depletion knob (XSSet.cpp's predictor-corrector) and
not a PPR one.

So this file holds TWO properties that a numerical A/B cannot see:

  A. THE MASTER BODY IS ON THE DEVICE, AND IT IS THE HOST'S ARITHMETIC.  Not
     "master no longer refuses" -- that would pass if the arm silently ran the
     SENM scheme under the master flag and produced a plausible, wrong, 0.5 %
     pin map.  The kernels' coefficient expressions are compared TERM BY TERM
     against PPR::driveMaster's.

  B. EVERY FALLBACK NAMES ITSELF.  A seam that only counts is how (A) survived
     a campaign.  ppr::Refusal is the ladder (BICGCMFD::EnqueueRefusal's shape),
     every `return false` on the way into the device sets a rung, the receipt
     publishes `refusal` and `refusals`, the RASBERY_GPU_FULL seam is handed the
     ladder's own name instead of a fixed sentence, and GpuFullContract names
     the first fallback whether or not the gate is on -- because
     `contract_pass:false` is printed whether or not the gate is on, and a null
     beside it is the receipt that started this investigation.

THE ONE PLACE THE DEVICE IS NOT THE HOST, AND IT IS DECLARED.  driveMaster's
corner-point-balance sweep is Gauss-Seidel: it writes `_phic` in place while
later nodes read it, a serial dependence over the whole mesh.  The device runs
the same balance as Jacobi (read `phic`, write `phic_next`, commit).  Both
contract on one diagonally dominant system to the same fixed point; they stop at
different distances from it, ~1e-5 relative.  That is class N1, it is written
down in CudaPprBackend.h and PPR.h, and this file refuses a build that quietly
claims otherwise -- or that quietly writes `phic` from the sweep kernel, which
would be a race wearing the same receipt.
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
    "recon": SRC / "PprReconstructionKernel.cuh",
    "driver": SRC / "Driver.h",
    "gpufull": SRC / "GpuFullContract.h",
}


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def load() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, path in FILES.items():
        out[key] = path.read_text(encoding="utf-8-sig") if path.is_file() else ""
    return out


def squeeze(text: str) -> str:
    """Whitespace-free form, so indentation and line breaks are not the contract."""
    return re.sub(r"\s+", "", text)


def slice_between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i:j] if j >= 0 else text[i:]


def function_body(text: str, signature: str) -> str:
    """From `signature` to the first line that is exactly `}` at column 0."""
    i = text.find(signature)
    if i < 0:
        return ""
    j = text.find("\n}", i)
    return text[i:j] if j >= 0 else text[i:]


def check(raw: dict[str, str]) -> list[str]:
    problems: list[str] = []

    for key, path in FILES.items():
        if not raw.get(key):
            problems.append(f"missing or empty file: {path.relative_to(ROOT)}")
    if problems:
        return problems

    CU = strip_comments(raw["cu"])
    HDR = strip_comments(raw["hdr"])
    HDR_TEXT = raw["hdr"]
    STUB = strip_comments(raw["stub"])
    PPR_H = strip_comments(raw["ppr_h"])
    PPR_H_TEXT = raw["ppr_h"]
    PPR_CPP = strip_comments(raw["ppr_cpp"])
    RECON = strip_comments(raw["recon"])
    DRIVER = strip_comments(raw["driver"])
    GPUFULL = strip_comments(raw["gpufull"])

    def want(text: str, needle: str, where: str, why: str) -> None:
        if needle not in text:
            problems.append(f"{where}: missing {needle!r} -- {why}")

    def forbid(text: str, needle: str, where: str, why: str) -> None:
        if needle in text:
            problems.append(f"{where}: contains {needle!r} -- {why}")

    # =====================================================================
    # 1. MASTER MODE NO LONGER REFUSES, AND THE MODE TRAVELS
    # =====================================================================
    #
    # The bug, spelled exactly as it was, is forbidden by name.  Not "the file
    # mentions master" -- the STATEMENT, because the statement is what turned a
    # whole campaign's GPU PPR arm off.

    reset_gpu = function_body(PPR_CPP, "bool PPR::resetAndDriveGpu(")
    if not reset_gpu:
        problems.append("PPR.cpp: PPR::resetAndDriveGpu not found")
    else:
        forbid(reset_gpu, "if (_mode_master) return false;", "PPR::resetAndDriveGpu",
               "MASTER mode is the PRODUCTION reconstruction mode (Gate B 0.238 % vs the "
               "default mode's 0.522 %); refusing it here is refusing every statepoint of "
               "every production run")
        want(reset_gpu, "step.mode_master     = _mode_master;", "PPR::resetAndDriveGpu",
             "the mode must travel to the backend in the StepView, or the device runs the "
             "SENM scheme under the master flag -- a plausible, wrong pin map")

    recon_planned = function_body(PPR_CPP, "bool PPR::reconPlanned()")
    if not recon_planned:
        problems.append("PPR.cpp: PPR::reconPlanned not found")
    else:
        forbid(recon_planned, "_mode_master", "PPR::reconPlanned",
               "the reconstruction plan decides whether the coefficient D2H is elided; "
               "declining it for master mode would make every master statepoint pay the "
               "9 MB download the device consumer exists to remove")

    recon_gpu = function_body(PPR_CPP, "bool PPR::reconstructPinPowerGpu(")
    if not recon_gpu:
        problems.append("PPR.cpp: PPR::reconstructPinPowerGpu not found")
    else:
        want(recon_gpu, "if (!use_quadrature || _ng != 2) return false;",
             "PPR::reconstructPinPowerGpu",
             "the POINTWISE sampling is still a different scheme and still declines; "
             "MASTER mode is not, and must not be in this list")
        want(recon_gpu, "step.mode_master      = _mode_master;",
             "PPR::reconstructPinPowerGpu",
             "the reconstruction has to know which expansion to evaluate")

    want(HDR, "bool mode_master = false;", "CudaPprBackend.h",
         "ppr::StepView and ppr::ReconStepView carry the mode")
    if HDR.count("bool mode_master = false;") < 2:
        problems.append("CudaPprBackend.h: only one view carries mode_master -- the drive "
                        "and the reconstruction each need it, and a reconstruction that "
                        "evaluated the SENM expansion over master coefficients would read "
                        "p/a/bt that master mode never wrote")

    # =====================================================================
    # 2. THE MASTER CAP FLOOR IS THE HOST'S
    # =====================================================================
    #
    # PPR::drive raises the cap for master mode (`std::max(niter, 200)`); the
    # device seam has to apply the same floor or the two arms iterate against
    # different bounds and "the same tolerance" means nothing.

    want(PPR_CPP, "driveMaster(std::max(niter, 200));", "PPR.cpp",
         "the host's master cap floor")
    want(PPR_CPP, "_mode_master ? std::max(niter, 200) : niter;", "PPR.cpp",
         "and the device seam must apply the SAME floor, or the device stops 100 rounds "
         "before the host it is compared against")

    # =====================================================================
    # 3. THE DEVICE BODY EXISTS AND IS DISPATCHED
    # =====================================================================

    for kernel in ("kMasterEven", "kMasterCpb", "kMasterCommit",
                   "kMasterFoldAndCheck", "kMasterCross"):
        want(CU, f"void {kernel}(", "CudaPprBackend.cu",
             f"{kernel} is one of the five kernels PPR::driveMaster decomposes into")

    want(CU, "const bool master = step.mode_master;", "CudaPprBackend.cu",
         "the round body is selected ONCE, outside the kernels, so neither scheme pays "
         "for the other's branch and a captured graph replays only what it captured")
    want(CU, "if (master) kMasterEven<<<", "CudaPprBackend.cu",
         "step 1 (the even-parity interpolant) runs after the whole of reset(), because "
         "PPR::reset has no mode branch and driveMaster consumes the corner flux it seeds")
    want(CU, "kMasterCross<<<", "CudaPprBackend.cu",
         "step 3 (the cross terms) runs after the loop and UNGUARDED -- the host evaluates "
         "them after its break, on whatever corners converged")
    want(CU, "int mode_master;", "CudaPprBackend.cu",
         "the mode is part of DevCtx and therefore part of the graph key; a capture taken "
         "under one scheme must not be replayed under the other")

    # The two loop kernels are guarded templates: the device_stream arm enqueues
    # `niter` bodies and lets the flag turn the surplus into no-ops, which is
    # only exact while every kernel of the body reads the flag first.
    for kernel in ("kMasterCpb", "kMasterCommit"):
        body = function_body(CU, f"__global__ void {kernel}(DevCtx x)")
        if not body:
            problems.append(f"CudaPprBackend.cu: {kernel} is not a `__global__ ... (DevCtx x)`")
            continue
        if "if (kGuarded && pprHalted(x)) return;" not in body:
            problems.append(
                f"CudaPprBackend.cu: {kernel} does not open with the kGuarded halt read -- "
                "the device_stream arm enqueues niter bodies and relies on every kernel of "
                "a post-convergence round writing NOTHING; one that writes turns the "
                "surplus launches into extra real iterations")
        if f"template <bool kGuarded>\n__global__ void {kernel}" not in raw["cu"]:
            problems.append(f"CudaPprBackend.cu: {kernel} is not a `template <bool kGuarded>`")

    # =====================================================================
    # 4. JACOBI, NOT AN IN-PLACE RACE
    # =====================================================================
    #
    # The whole reason this is N1 rather than B0.  The sweep kernel must read
    # the CURRENT iterate and write the NEXT one; a kernel that wrote x.phic
    # would be thread A reading a corner while thread B rewrites it -- which
    # produces a finite, plausible, launch-order-dependent pin map, i.e. the one
    # failure mode this arm may not have.

    cpb = function_body(CU, "__global__ void kMasterCpb(DevCtx x)")
    if not cpb:
        problems.append("CudaPprBackend.cu: kMasterCpb not found")
    else:
        # WP20.2 replaced the two `phic_next` POINTERS with accessors, because
        # the array's width is now the RASBERY_GPU_FP32_PPR arm's to choose and
        # a pointer cannot carry a width.  What the rule is about is unchanged
        # and is checked on the accessor instead: the sweep writes the NEXT
        # iterate and never the current one.
        want(cpb, "pprPhicNextStore(x, lk, g, dir,", "kMasterCpb",
             "the sweep writes the NEXT iterate")
        forbid(cpb, "phicNextPtr(", "kMasterCpb",
               "a raw pointer into phic_next is null on the narrow arm; the width has "
               "to travel with the address, which is what the accessors are for")
        want(cpb, "x.phic[(m * 4 * x.ng) + (g * 4) + adj1]", "kMasterCpb",
             "and reads the current one, from the neighbouring node")
        if re.search(r"phicPtr\(x,\s*lk,\s*g\)\s*;?\s*$", cpb, re.M) and \
                "const double* phic_in = phicPtr(x, lk, g);" not in cpb:
            problems.append("kMasterCpb: takes a MUTABLE phicPtr; the sweep may only read "
                            "the current iterate")
        forbid(cpb, "phic_in[dir] =", "kMasterCpb",
               "writing the current iterate from the sweep is the in-place Gauss-Seidel "
               "the host does and the device cannot: neighbouring threads read it")
        # The BALANCE stays FP64 on both arms.  Only the stored corner narrows;
        # narrowing the arithmetic would move the fixed point rather than the
        # resolution it is held at, and the break test would then be measuring
        # the arm instead of the sweep.
        want(cpb, "const double fnew = num / den;", "kMasterCpb",
             "the corner balance and its quotient are FP64 whatever the arm")
        want(cpb, "node_max = fmax(node_max, fabs((fnew - fold) / fold));", "kMasterCpb",
             "and so is the relative change the break test folds")

    commit = function_body(CU, "__global__ void kMasterCommit(DevCtx x)")
    if not commit:
        problems.append("CudaPprBackend.cu: kMasterCommit not found")
    else:
        want(commit, "dst[0] = pprPhicNext(x, lk, g, 0);", "kMasterCommit",
             "the commit is the only writer of phic during the master drive")
        want(commit, "m = fmax(m, pprMrel(x, lk, g));", "kMasterCommit",
             "and it folds the relative changes through the same accessor pair")
        forbid(commit, "phicNextPtr(", "kMasterCommit",
               "the commit reaches phic_next through the accessor too, so the arm's "
               "width is decided in ONE place for the whole master body")
        want(commit, "detChunkBegin(x.nxyz, nchunk, c)", "kMasterCommit",
             "and it folds the relative changes over the SAME deterministic partition the "
             "corner sums use, so the arm stays run-to-run deterministic")

    # WP20.2.  `phic_next` and `mrel` are the two arrays MASTER MODE ALLOCATES
    # FOR ITSELF, which is what makes them narrowable at all: the SENM arm --
    # B0 against the host's own corner update -- never touches either, so the
    # width is a master-mode property and belongs in this file.  What has to
    # hold is that the width travels WITH the address: a body that named the
    # wide pointer on the narrow arm would read a null.
    for accessor in ("pprPhicNext", "pprMrel"):
        body = function_body(CU, "inline double %s(const DevCtx& x" % accessor)
        if not body:
            problems.append("CudaPprBackend.cu: %s not found" % accessor)
        else:
            want(body, "x.narrow_corner ?", accessor,
                 "the accessor branches on the ctx's OWN width flag rather than on a "
                 "global, so the width cannot disagree with the pointers it describes")

    fold = function_body(CU, "__global__ void kMasterFoldAndCheck(DevCtx x)")
    if not fold:
        problems.append("CudaPprBackend.cu: kMasterFoldAndCheck not found")
    else:
        want(fold, "maxrel < kCornerFluxTolerance", "kMasterFoldAndCheck",
             "the break test is driveMaster's own test against driveMaster's own tolerance")
        want(fold, "st->iters        = st->iters + 1;", "kMasterFoldAndCheck",
             "and the round count moves only while the flag is down, so `iterations` in "
             "the receipt is the rounds that actually ran")
        want(fold, "__threadfence();", "kMasterFoldAndCheck",
             "the guarded kernels of the NEXT round read `converged`; without the fence "
             "they could see it raised before the state it describes is visible")

    # =====================================================================
    # 5. THE ARITHMETIC IS PPR::driveMaster's, TERM BY TERM
    # =====================================================================
    #
    # THE PROPERTY THAT MATTERS MOST, and the only one here a passing kngr_238
    # A/B could not fake: every coefficient the device writes is the host's
    # expression, in the host's order, on the host's operands.  A transcription
    # that dropped a term, reassociated a sum or swapped an axis would still
    # converge, still look like a pin map, and still be wrong by a few tenths of
    # a percent -- which is the size of the entire Gate B budget.

    host_master = function_body(PPR_CPP, "void PPR::driveMaster(int niter)")
    dev_master = slice_between(CU, "__global__ void kMasterEven(DevCtx x)",
                               "__global__ void kCanonicalCompare")
    if not host_master:
        problems.append("PPR.cpp: PPR::driveMaster not found")
    elif not dev_master:
        problems.append("CudaPprBackend.cu: the master kernel block not found")
    else:
        host_terms = [(i, j, squeeze(rhs)) for i, j, rhs in
                      re.findall(r"c\((\d),\s*(\d)\)\s*=\s*([^;]*);", host_master)]
        dev_terms = [(i, j, squeeze(rhs)) for i, j, rhs in
                     re.findall(r"c\[triIdx\((\d),\s*(\d)\)\]\s*=\s*([^;]*);", dev_master)]
        if not host_terms:
            problems.append("PPR.cpp: no c(i, j) assignments in driveMaster")
        if host_terms != dev_terms:
            host_map = {(i, j): r for i, j, r in host_terms}
            dev_map = {(i, j): r for i, j, r in dev_terms}
            for key in sorted(set(host_map) | set(dev_map)):
                if host_map.get(key) != dev_map.get(key):
                    problems.append(
                        f"the MASTER coefficient c({key[0]},{key[1]}) differs between "
                        f"PPR::driveMaster and the kernels: host {host_map.get(key)!r} "
                        f"vs device {dev_map.get(key)!r}")
            if len(host_terms) != len(dev_terms):
                problems.append(
                    f"PPR::driveMaster writes {len(host_terms)} coefficient terms and the "
                    f"kernels write {len(dev_terms)}; the two-pass shape (even parity "
                    "zeroed, then the cross terms) must be the same on both sides")

        # The corner-point-balance numerator and denominator, MM Eq. 6.7/6.8.
        # Held on BOTH sides: a term that moves on the host without moving on the
        # device is a stale transcription, and it looks exactly like a correct one.
        for dev_expr, host_expr in (
                ("num+=w*(5.0*(pox+poy)+(pix+piy)+jox+joy-6.0*pbm-fadj1-fadj2);",
                 "num+=w*(5.0*(pox+poy)+(pix+piy)+jox+joy-6.0*pbm-fadj1-fadj2);"),
                ("den+=4.0*w;", "den+=4.0*w;"),
                ("constdoublew=dXsdf(x,g,m)/dHmesh(x,kXDIR,m);",
                 "constdoublew=_xs.xsdf(g,m)/_g.hmesh(XDIR,m);")):
            if dev_expr not in squeeze(dev_master):
                problems.append(f"kMasterCpb: the CPB balance term {dev_expr!r} is not the "
                                "host's")
            if host_expr not in squeeze(host_master):
                problems.append(f"PPR::driveMaster: the CPB balance term {host_expr!r} "
                                "moved; the device transcription is now stale")

        # The host's two literals for the even-parity scaling, written as
        # quotients because 1/14 is not representable and a decimal literal is a
        # different double.
        for lit in ("r10 = 1.0 / 10.0", "r14 = 1.0 / 14.0"):
            want(host_master, lit, "PPR::driveMaster", "the host's scaling literal")
            want(dev_master, lit, "kMasterEven",
                 "the device must divide, not paste a rounded decimal -- 1/14 is not "
                 "representable and 0.07142857142857142 is a different double")

    # =====================================================================
    # 6. THE MASTER RECONSTRUCTION IS THE HOST'S DOT PRODUCT, AND B0
    # =====================================================================

    want(RECON, "int mode_master;", "PprReconstructionKernel.cuh",
         "ReconCtx must carry the mode")
    want(RECON, "const double* c;", "PprReconstructionKernel.cuh",
         "and the coefficients master mode evaluates")
    want(CU, "r.mode_master      = step.mode_master ? 1 : 0;", "CudaPprBackend.cu",
         "which the backend has to fill, or the device reconstruction silently evaluates "
         "the SENM expansion over p/a/bt that master mode never wrote")
    want(CU, "r.c                = s.d_c;", "CudaPprBackend.cu", "with the drive's own _c")

    master_branch = slice_between(RECON, "if (x.mode_master) {", "const double  bt")
    if not master_branch:
        problems.append("PprReconstructionKernel.cuh: no `if (x.mode_master)` branch in the "
                        "pin expansion -- the master interpolant is not a special case of "
                        "the SENM one, it replaces it")
    else:
        want(master_branch, "for (int tt = 0; tt < 15; ++tt) cflux += c_base[tt] * leg[tt];",
             "PprReconstructionKernel.cuh",
             "the 15-term dot product, in the host's order")
        want(master_branch, "integ += q_wt[qq] * cflux;", "PprReconstructionKernel.cuh",
             "weighted in the host's order")
        forbid(master_branch, "exp(", "PprReconstructionKernel.cuh",
               "MASTER mode's expansion has NO transcendental in it; an exp here would "
               "make a B0 branch N1 for nothing")
        forbid(master_branch, "x.p[", "PprReconstructionKernel.cuh",
               "master mode never writes p, so reading it is reading whatever the "
               "allocation left behind")
        forbid(master_branch, "x.a[", "PprReconstructionKernel.cuh",
               "and never writes a either")

    # The host branch this transcribes must still be there.
    want(PPR_CPP, "if (_mode_master) {", "PPR.cpp",
         "the host reconstruction keeps its master branch as the reference")

    # p and a do not come back in master mode: the device buffers were never
    # written, and the host arrays are equally unread.
    want(CU, "if (!master) {", "CudaPprBackend.cu",
         "the master D2H is five arrays, not seven -- p and a are never written by either "
         "arm in master mode, so copying the device's uninitialised ones over the host's "
         "would be a difference invented by the port")

    # =====================================================================
    # 7. THE REFUSAL LADDER
    # =====================================================================

    want(HDR, "enum class Refusal : int {", "CudaPprBackend.h",
         "the ladder, mirroring BICGCMFD::EnqueueRefusal")
    names = re.search(r"inline const char\* refusalName\(Refusal r\) \{(.*?)\n\}", HDR, re.S)
    enum = re.search(r"enum class Refusal : int \{(.*?)\n\};", HDR, re.S)
    if not names or not enum:
        problems.append("CudaPprBackend.h: Refusal or refusalName is missing")
    else:
        rungs = re.findall(r"^\s*(\w+)", enum.group(1), re.M)
        rungs = [r for r in rungs if r not in ("", "None")]
        for rung in rungs:
            if f"case Refusal::{rung}:" not in names.group(1):
                problems.append(f"CudaPprBackend.h: refusalName has no case for "
                                f"Refusal::{rung} -- a rung with no name reports '?' and "
                                "is the same defect as no ladder at all")
        if "default:" in names.group(1):
            problems.append("CudaPprBackend.h: refusalName has a `default:` -- it would "
                            "swallow a new rung instead of failing the switch's "
                            "exhaustiveness warning")
        if "case Refusal::None:" not in names.group(1):
            problems.append("CudaPprBackend.h: refusalName has no case for Refusal::None")

    # Every way into the device names its rung.
    if reset_gpu:
        for rung in ("ppr::Refusal::ArmOff", "ppr::Refusal::NotTwoGroup"):
            want(reset_gpu, f"_gpu->noteHostFallback({rung});", "PPR::resetAndDriveGpu",
                 "these statepoints never reach the backend, so a ladder kept only inside "
                 "it would report `none` for the two commonest production reasons")
        bare = [ln.strip() for ln in reset_gpu.split("\n") if ln.strip() == "return false;"]
        noted = reset_gpu.count("noteHostFallback(")
        if len(bare) > noted + 1:
            problems.append(
                f"PPR::resetAndDriveGpu: {len(bare)} bare `return false;` against "
                f"{noted} noteHostFallback calls -- the ladder is meant to be exhaustive, "
                "and an unnamed refusal is exactly the defect this file exists for "
                "(one unnamed return is allowed: the null-backend guard, which has no "
                "object to record on)")

    backend_entry = slice_between(CU, "bool PprBackend::resetAndDrive(", "if (!s.geometry_uploaded)")
    if not backend_entry:
        problems.append("CudaPprBackend.cu: PprBackend::resetAndDrive entry not found")
    else:
        for rung in ("ArmOff", "BackendDisabled", "NotTwoGroup", "NonPositiveIter",
                     "ShapeAllocFail"):
            want(backend_entry, f"ppr::Refusal::{rung}", "PprBackend::resetAndDrive",
                 "every early refusal names its rung")
    fail_body = function_body(CU, "    bool fail(const char* what, cudaError_t rc) {")
    if "noteRefusal(ppr::Refusal::CudaFailure);" not in CU:
        problems.append("CudaPprBackend.cu: fail() does not record the CudaFailure rung -- "
                        "it is the ONE writer for every `if (!h2d(...)) return false;` in "
                        "the file, and a seam added later must inherit it for free")

    # The receipt publishes it.
    for field in (r'\"refusal\":\"{}\"', r'\"refusals\":{}'):
        want(raw["driver"], field, "Driver.h",
             "the [RASBERY][PPR_GPU] receipt must carry the ladder; `host_fallbacks:35` "
             "with no reason beside it is the line that cost this campaign a release")
    want(DRIVER, "g.lastRefusalName(), g.refusalJson(),", "Driver.h",
         "and it must be fed from the ladder, not from a second string")

    # The RASBERY_GPU_FULL seam is handed the ladder's own name.
    want(DRIVER, "pin_power_reconstruction.gpu().lastRefusalName());", "Driver.h",
         "the GPU_FULL guard's `why` is what lands in first_violation; a fixed sentence "
         "there reports 'the device PPR arm declined' for all seven reasons")
    forbid(DRIVER, '"the device PPR arm declined; the host reset+drive runs"', "Driver.h",
           "that is the fixed sentence the ladder replaces")

    # And GpuFullContract names the first fallback with the gate off too.
    want(GPUFULL, "inline void nameFirstFallback(", "GpuFullContract.h",
         "`contract_pass:false` is printed whether or not the gate is on, so "
         "`first_violation` has to be filled whether or not the gate is on -- a null "
         "beside a false is a receipt that says a seam fired and refuses to say which")
    note_body = function_body(GPUFULL, "inline void note(Subsystem which,")
    if not note_body:
        problems.append("GpuFullContract.h: gpufull::note not found")
    elif "nameFirstFallback(which, where, why);" not in note_body:
        problems.append("GpuFullContract.h: note() does not name the first fallback")
    elif note_body.index("nameFirstFallback") > note_body.index("if (!required()) return;"):
        problems.append("GpuFullContract.h: note() names the first fallback only under the "
                        "gate; with the gate off the receipt is back to null")

    # The accessors exist on both backends.
    for decl in ("void noteHostFallback(ppr::Refusal reason);",
                 "[[nodiscard]] ppr::Refusal       lastRefusal() const;",
                 "[[nodiscard]] const char*        lastRefusalName() const;",
                 "[[nodiscard]] std::string        refusalJson() const;"):
        want(HDR, decl, "CudaPprBackend.h", "the ladder's receipt surface")
    for defn in ("void PprBackend::noteHostFallback(",
                 "ppr::Refusal PprBackend::lastRefusal() const",
                 "const char* PprBackend::lastRefusalName() const",
                 "std::string PprBackend::refusalJson() const"):
        want(STUB, defn, "CudaPprBackendStub.cpp",
             "a CPU-only build links the stub; a missing definition is a link error "
             "nobody sees until the non-CUDA build")

    # =====================================================================
    # 8. THE CLASS IS DECLARED, AND IT IS N1
    # =====================================================================

    want(HDR_TEXT, "CLASS N1, AND EXACTLY WHERE.", "CudaPprBackend.h",
         "the master arm's Jacobi CPB is not bit-identical to the host's Gauss-Seidel and "
         "the header may not claim it is -- and it has to say where, or the next reader "
         "cannot tell an attributable deviation from an unexplained one")
    want(HDR_TEXT, "Jacobi", "CudaPprBackend.h",
         "and it has to say WHICH iteration it runs, or the N1 claim is unattributable")
    want(PPR_H_TEXT, "JACOBI where the host's is Gauss-Seidel", "PPR.h",
         "the seam the caller reads says the same thing the backend does")
    forbid(HDR_TEXT, "master arm is bit-identical", "CudaPprBackend.h",
           "it is not")

    # PPR is still post-processing: nothing here may feed the trajectory.
    if host_master and "++_host_iters;" not in host_master:
        problems.append("PPR::driveMaster: does not count its own rounds -- the receipt's "
                        "host_iterations would read 0 for every master run and the "
                        "\"did the two schemes stop in the same place?\" comparison would "
                        "have only the device's side")
    forbid(DRIVER, '"RASBERY_PPR_MODE"', "Driver.h",
           "PPR is downstream of every arm knob; putting the mode in kArmEnv would declare "
           "it able to move a trajectory, which it is not")

    return problems


# ---------------------------------------------------------------------------
# Negative controls: every property above must be BREAKABLE.
# ---------------------------------------------------------------------------

CONTROLS = [
    ("master refuses again", "ppr_cpp",
     "    if (_gpu == nullptr) return false;\n",
     "    if (_gpu == nullptr) return false;\n    if (_mode_master) return false;\n"),
    ("the mode stops travelling to the drive", "ppr_cpp",
     "    step.mode_master     = _mode_master;\n", ""),
    ("the mode stops travelling to the reconstruction", "ppr_cpp",
     "    step.mode_master      = _mode_master;\n", ""),
    ("the master cap floor is dropped", "ppr_cpp",
     "_mode_master ? std::max(niter, 200) : niter;", "niter;"),
    ("the sweep kernel loses its halt guard", "cu",
     """template <bool kGuarded>
__global__ void kMasterCpb(DevCtx x) {
    if (kGuarded && pprHalted(x)) return;""",
     """template <bool kGuarded>
__global__ void kMasterCpb(DevCtx x) {"""),
    ("the sweep writes the current iterate (the in-place race)", "cu",
     "            pprPhicNextStore(x, lk, g, dir, fnew);",
     "            phic_in[dir] = fnew;"),
    ("the narrow arm loses its accessor and reads the wide pointer", "cu",
     "    return x.narrow_corner ? static_cast<double>(x.phic_next_f[i]) : x.phic_next[i];",
     "    return x.phic_next[i];"),
    ("a coefficient term is reassociated", "cu",
     "c[triIdx(2, 0)] = r14 * (10.0 * (pxr + pxl) + (jxr + jxl) - 20.0 * pb);",
     "c[triIdx(2, 0)] = r14 * ((jxr + jxl) + 10.0 * (pxr + pxl) - 20.0 * pb);"),
    ("a cross term loses its node average", "cu",
     "0.25 * (f1 + f2 + f3 + f4 - 2.0 * (pxr + pxl + pyr + pyl) + 4.0 * aflux);",
     "0.25 * (f1 + f2 + f3 + f4 - 2.0 * (pxr + pxl + pyr + pyl));"),
    ("the CPB balance drops the adjacent corners", "cu",
     "6.0 * pbm - fadj1 - fadj2);", "6.0 * pbm);"),
    ("the even-parity literal becomes a rounded decimal", "cu",
     "const double r14 = 1.0 / 14.0;", "const double r14 = 0.07142857142857142;"),
    ("the master reconstruction branch vanishes", "recon",
     "            if (x.mode_master) {", "            if (false) {"),
    ("the master reconstruction reads p", "recon",
     "                for (int tt = 0; tt < 15; ++tt) cflux += c_base[tt] * leg[tt];",
     "                for (int tt = 0; tt < 15; ++tt) cflux += x.p[tt] * leg[tt];"),
    ("the backend stops filling the recon mode", "cu",
     "    r.mode_master      = step.mode_master ? 1 : 0;\n", ""),
    ("a ladder rung loses its name", "hdr",
     "        case Refusal::NotTwoGroup:     return \"not_two_group\";\n", ""),
    ("the host-side refusals stop naming themselves", "ppr_cpp",
     "        _gpu->noteHostFallback(ppr::Refusal::NotTwoGroup);\n", ""),
    ("fail() stops recording the CUDA rung", "cu",
     "        noteRefusal(ppr::Refusal::CudaFailure);\n", ""),
    ("the receipt loses the ladder", "driver",
     '"\\"refusal\\":\\"{}\\",\\"refusals\\":{},"\n                    ', ""),
    ("the GPU_FULL seam goes back to a fixed sentence", "driver",
     "                    pin_power_reconstruction.gpu().lastRefusalName());",
     '                    "the device PPR arm declined; the host reset+drive runs");'),
    ("first_violation goes back behind the gate", "gpufull",
     "    nameFirstFallback(which, where, why);\n    if (!required()) return;\n    throw Violation",
     "    if (!required()) return;\n    nameFirstFallback(which, where, why);\n    throw Violation"),
    ("the stub loses an accessor", "stub",
     "std::string PprBackend::refusalJson() const {", "std::string PprBackend::refusalJsonX() const {"),
    ("the N1 declaration is dropped", "hdr",
     "CLASS N1, AND EXACTLY WHERE.", "CLASS B0, EVERYWHERE."),
    ("driveMaster stops counting its rounds", "ppr_cpp",
     "        ++_host_iters;\n        if (maxrel < kCornerFluxTolerance) break;",
     "        if (maxrel < kCornerFluxTolerance) break;"),
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
        print("PPR GPU master-mode contract: FAIL")
        for p in problems:
            print(f"  - {p}")
        for c in controls:
            print(f"  ! {c}")
        return 1
    print(f"PPR GPU master-mode contract: OK (8 property groups, "
          f"{len(CONTROLS)} negative controls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
