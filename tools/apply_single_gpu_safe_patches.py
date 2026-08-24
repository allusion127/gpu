#!/usr/bin/env python3
"""Apply reviewed low-risk one-GPU source patches to exact base blobs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

CUDA_SHA = "4525d5136fa26b3092348af8dd1015bae17a8acd"
MAIN_SHA = "a3feddf5a3b34bde8e3949ad474ca98c16cd2861"


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def one(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected one exact block, found {count}")
    return text.replace(old, new, 1)


def patch_cuda(text: str) -> str:
    old_finalize = r'''    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);
    if (sweep_halt[m] != 0u) return;
    const double* ta = terms_ab + m * vec_stride;
    const double* tc = terms_c + m * vec_stride;
    double err = 0.0, gammad = 0.0, gamman = 0.0;
    for (int l = 0; l < nxyz; ++l) {
        err    = err + ta[l];
        gammad = gammad + ta[nxyz + l];
        gamman = gamman + tc[l];
    }
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    sm[kErrAcc] = err;
    sm[kGammaD] = gammad;
    sm[kGammaN] = gamman;
'''
    new_finalize = r'''    const int m = static_cast<int>(blockIdx.y);
    if (sweep_halt[m] != 0u) return; // uniform for the whole slot block
    const int lane = static_cast<int>(threadIdx.x);
    const double* ta = terms_ab + m * vec_stride;
    const double* tc = terms_c + m * vec_stride;
    __shared__ double lane_sum[3];

    // Each independent sum retains the original l-ascending dependency chain.
    // Lanes 0, 1 and 2 therefore run concurrently without changing a sum's
    // operand pairing or deterministic double result.
    if (lane < 3) {
        const double* values = lane == 0 ? ta : (lane == 1 ? ta + nxyz : tc);
        double sum = 0.0;
        for (int l = 0; l < nxyz; ++l) sum = sum + values[l];
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;

    const double err    = lane_sum[0];
    const double gammad = lane_sum[1];
    const double gamman = lane_sum[2];
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    sm[kErrAcc] = err;
    sm[kGammaD] = gammad;
    sm[kGammaN] = gamman;
'''
    text = one(text, old_finalize, new_finalize, "parallel_wiel_finalize")
    text = one(
        text,
        "cmfd_wiel_finalize<<<scalar_grid(), 1, 0, stream>>>",
        "cmfd_wiel_finalize<<<scalar_grid(), 32, 0, stream>>>",
        "wiel_finalize_launch_width",
    )
    old_eps = r'''                CUDA_CHECK(cudaMemcpyAsync(scalars + static_cast<long long>(m) * kScalarCount + kEps,
                                           &sl.eps,
                                           sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                ++telemetry.stream_sync_calls_during_iteration;
                sl.eps_on_device = sl.eps;
'''
    new_eps = r'''                CUDA_CHECK(cudaMemcpyAsync(scalars + static_cast<long long>(m) * kScalarCount + kEps,
                                           &sl.eps,
                                           sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           stream));
                // Graph/direct kernels are submitted to this same stream, so
                // stream order publishes eps without draining the pipeline.
                sl.eps_on_device = sl.eps;
'''
    return one(text, old_eps, new_eps, "remove_eps_stream_drain")


def patch_main(text: str) -> str:
    old = r'''        const int jobs = static_cast<int>(rasbery_inputs.size());
        int host_threads = std::min(batch_width, jobs);
#ifdef _OPENMP
        const int visible_cpus = startup_visible_cpus;
        // The CUDA arena width and the number of CPU Driver workers are separate
        // resources.  Use an explicit, benchmarked cap because OMP_PROC_BIND can
        // narrow the main-thread affinity before this branch and make automatic
        // CPU-count discovery platform/runtime dependent.  With no override the
        // historical one-worker-per-live-instance behavior is preserved.
        if (const char* host_env = std::getenv("RASBERY_BATCH_HOST_THREADS")) {
            const int requested = std::atoi(host_env);
            if (requested > 0) host_threads = std::min({requested, batch_width, jobs});
        }
#else
        const int visible_cpus = 1;
#endif
'''
    new = r'''        const int jobs = static_cast<int>(rasbery_inputs.size());
        int         visible_cpus = 1;
        int         host_threads = 1;
        const char* host_policy  = "serial";
#ifdef _OPENMP
        visible_cpus = startup_visible_cpus;
        // CUDA arena width and CPU Driver concurrency are separate resources.
        // Default to process affinity capacity instead of oversubscribing every
        // live GPU slot; explicit counts remain the production tuning knob.
        host_threads = std::min({visible_cpus, batch_width, jobs});
        host_policy  = "auto-visible-cpus";
        if (const char* host_env = std::getenv("RASBERY_BATCH_HOST_THREADS")) {
            if (std::strcmp(host_env, "legacy") == 0) {
                host_threads = std::min(batch_width, jobs);
                host_policy  = "legacy-live-slots";
            } else {
                const int requested = std::atoi(host_env);
                if (requested > 0) {
                    host_threads = std::min({requested, batch_width, jobs});
                    host_policy  = "explicit";
                }
            }
        }
#endif
'''
    text = one(text, old, new, "batch_host_auto_cap")
    old_receipt = r'''                  << ",\"host_threads\":" << host_threads
                  << ",\"visible_cpus\":" << visible_cpus << "}" << std::endl;
'''
    new_receipt = r'''                  << ",\"host_threads\":" << host_threads
                  << ",\"visible_cpus\":" << visible_cpus
                  << ",\"policy\":\"" << host_policy << "\"}" << std::endl;
'''
    return one(text, old_receipt, new_receipt, "batch_host_policy_receipt")


def patch(path: Path, expected: str, fn, check: bool) -> None:
    raw = path.read_bytes()
    actual = blob_sha(raw)
    if actual != expected:
        raise RuntimeError(f"{path}: git blob {actual} != reviewed base {expected}")
    patched = fn(raw.decode("utf-8-sig"))
    if not check:
        path.write_text(patched, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        patch(args.root / "src/CudaBICGBackend.cu", CUDA_SHA, patch_cuda, args.check)
        patch(args.root / "src/main.cpp", MAIN_SHA, patch_main, args.check)
    except (OSError, UnicodeDecodeError, RuntimeError) as exc:
        print(f"single-GPU safe patch: FAIL: {exc}", file=sys.stderr)
        return 1
    print("single-GPU safe patch: applicable" if args.check else "single-GPU safe patch: applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
