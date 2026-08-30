#!/usr/bin/env python3
"""Semantic check for a .cu translation unit on a host without nvcc.

The authoring machine for the depletion probe has a CUDA *driver* but no CUDA
*toolkit*, so `src/CudaDepletionBackend.cu` cannot be compiled the real way
there.  Leaving it unchecked is not an option: it is the only file in the change
that no other gate touches, and a typo in it would surface for the first time on
the benchmark host.

So the file is rewritten into valid C++ -- kernel launches lose their
`<<<grid, block, shmem, stream>>>` chevrons, the execution-space qualifiers and
the built-in variables are defined away, and a small shim supplies the subset of
the CUDA runtime the file calls -- and handed to the host compiler with
`-fsyntax-only`.

What this proves: every name resolves, every call matches a declaration, every
type lines up, the template instantiations are valid, and the kernel body (which
is `DepletionKernel.h`, shared verbatim with the host) type-checks.

What it does NOT prove: that nvcc accepts it, that the launch configuration is
legal, that the per-thread stack frame fits, or anything at all about runtime
behaviour.  Those need the real toolkit; this gate exists so that when the real
toolkit finally sees the file, the failures it reports are CUDA failures rather
than typos.

Usage:
    python3 tools/check_cuda_syntax.py [--cxx g++] [--keep] <file.cu> [...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SHIM = r'''
// Auto-generated CUDA shim for host-side syntax checking.  Not a build artefact
// and never compiled into RASBERY.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstring>

// The host <cmath> supplies logb/scalbn/hypot/copysign/isfinite/isnan/isinf,
// which the device libm also provides; the rewrite therefore type-checks the
// same call shapes nvcc will see.
using std::copysign;
using std::hypot;
using std::logb;
using std::scalbn;

using cudaError_t = int;
enum : int { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice, cudaMemcpyHostToHost };
enum : unsigned { cudaStreamNonBlocking = 1u, cudaStreamDefault = 0u };

struct CUstream_st;
using cudaStream_t = CUstream_st*;

inline const char* cudaGetErrorString(cudaError_t) { return ""; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 0; return cudaSuccess; }
inline cudaError_t cudaMalloc(void** p, std::size_t) { *p = nullptr; return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMallocHost(void** p, std::size_t) { *p = nullptr; return cudaSuccess; }
inline cudaError_t cudaFreeHost(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind) { return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned) { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceFlags(unsigned* f) { *f = 0; return cudaSuccess; }
inline const char* cudaGetErrorName(cudaError_t) { return ""; }
// WP16 host-spin (src/CudaHostSchedule.cu).  The four schedule bits and their
// mask, with the toolkit's values, so the switch over them type-checks AND
// keeps its four distinct cases here the way nvcc will see them.
enum : unsigned {
    cudaDeviceScheduleAuto         = 0x00u,
    cudaDeviceScheduleSpin         = 0x01u,
    cudaDeviceScheduleYield        = 0x02u,
    cudaDeviceScheduleBlockingSync = 0x04u,
    cudaDeviceScheduleMask         = 0x07u,
};

struct CUevent_st;
using cudaEvent_t = CUevent_st*;
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0.0f; return cudaSuccess; }

// Built-in variables.  Values are irrelevant to a syntax check; the shape is not.
struct RasberyDim3 { unsigned x = 0, y = 0, z = 0; };
static RasberyDim3 blockIdx, threadIdx, blockDim, gridDim;

inline int atomicMax(int* address, int val) {
    if (*address < val) *address = val;
    return *address;
}
inline unsigned long long atomicMax(unsigned long long* address, unsigned long long val) {
    if (*address < val) *address = val;
    return *address;
}
inline unsigned long long atomicAdd(unsigned long long* address, unsigned long long val) {
    unsigned long long old = *address;
    *address += val;
    return old;
}
inline unsigned long long atomicOr(unsigned long long* address, unsigned long long val) {
    unsigned long long old = *address;
    *address |= val;
    return old;
}
inline unsigned int atomicOr(unsigned int* address, unsigned int val) {
    unsigned int old = *address;
    *address |= val;
    return old;
}
inline long long __double_as_longlong(double x) {
    long long r;
    std::memcpy(&r, &x, sizeof r);
    return r;
}
inline double __longlong_as_double(long long x) {
    double r;
    std::memcpy(&r, &x, sizeof r);
    return r;
}
inline cudaError_t cudaMemsetAsync(void*, int, std::size_t, cudaStream_t) { return cudaSuccess; }
'''

# `name<<<a, b, c, d>>>(args)` -> `name(args)`.  Chevron arguments can nest
# parentheses (e.g. `dim3(x, y)`), so the body is matched non-greedily up to the
# closing `>>>`, which cannot appear inside a launch configuration.
LAUNCH = re.compile(r"<<<.*?>>>", re.DOTALL)

QUALIFIERS = re.compile(
    r"\b(__global__|__device__|__host__|__forceinline__|__restrict__"
    r"|__constant__|__shared__)\b")


def rewrite(text: str) -> str:
    text = LAUNCH.sub("", text)
    text = QUALIFIERS.sub("", text)
    # The shared kernel header keys off __CUDACC__ for its own qualifiers; it is
    # already qualifier-free on the host path, so nothing to do there.
    return '#include "rasbery_cuda_shim.h"\n' + text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--std", default="c++20")
    ap.add_argument("-I", "--include", action="append", default=[])
    ap.add_argument("-D", "--define", action="append", default=[])
    ap.add_argument("--keep", action="store_true", help="keep the rewritten sources")
    args = ap.parse_args()

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        (work / "rasbery_cuda_shim.h").write_text(SHIM, encoding="utf-8")
        # The file under test includes <cuda_runtime.h> for real; satisfy it
        # from the shim rather than stripping the include, so a missing shim
        # declaration still surfaces as an error.
        (work / "cuda_runtime.h").write_text(
            '#pragma once\n#include "rasbery_cuda_shim.h"\n', encoding="utf-8")

        for src in args.files:
            if not src.is_file():
                print(f"FAIL {src}: not a file")
                failures += 1
                continue
            out = work / (src.stem + "_syntax.cpp")
            out.write_text(rewrite(src.read_text(encoding="utf-8")), encoding="utf-8")

            cmd = [args.cxx, "-fsyntax-only", f"-std={args.std}", f"-I{work}"]
            cmd += [f"-I{p}" for p in args.include]
            cmd += [f"-D{d}" for d in args.define]
            cmd.append(str(out))

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                print(f"OK   {src}")
            else:
                print(f"FAIL {src}")
                sys.stdout.write(proc.stdout)
                sys.stderr.write(proc.stderr)
                failures += 1

            if args.keep:
                keep = src.with_suffix(".syntax.cpp")
                keep.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"     kept {keep}")

    print(f"{'FAILED' if failures else 'PASSED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
