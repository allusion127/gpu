#!/usr/bin/env python3
"""Apply the reviewed nodal CUDA refactor to CudaXsReconBackend.cu.

The source transformer is intentionally fail-closed. Every replacement is
anchored to the exact v2 source text and must match once. A future source edit
therefore causes CI to stop instead of silently patching the wrong location.
The transformation is idempotent and leaves the historical kernels as a
runtime rollback path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "// RASBERY_NODAL_REFACTOR_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_between_once(text: str, start: str, end: str, new: str, label: str) -> str:
    first = text.find(start)
    if first < 0:
        raise RuntimeError(f"{label}: start anchor not found")
    if text.find(start, first + 1) >= 0:
        raise RuntimeError(f"{label}: start anchor is not unique")
    last = text.find(end, first)
    if last < 0:
        raise RuntimeError(f"{label}: end anchor not found")
    if text.find(end, last + 1) >= 0:
        raise RuntimeError(f"{label}: end anchor is not unique")
    return text[:first] + new + text[last:]


def validate_patched(text: str) -> None:
    required = (
        MARKER,
        '#include "CudaTransferMirror.h"',
        'RASBERY_GPU_NODAL_FUSE_MAT_EVEN',
        'RASBERY_GPU_NODAL_XS_MIRROR',
        '__global__ void kNodalMatEven',
        'cuda_transfer::ByteExactMirror<double> xsrf_mirror',
        '\\"mat_even_fused\\"',
        '\\"xs_h2d_bytes\\"',
        'sl.xsrf_mirror.commit',
        'if (_fuse_mat_even)',
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"patched-source validation failed; missing={missing}")

    sync = text.find("const cudaError_t src = cudaStreamSynchronize(_stream);")
    commit = text.find("sl.xsrf_mirror.commit", sync)
    if sync < 0 or commit < 0 or commit < sync:
        raise RuntimeError("XS mirrors must be committed only after a successful stream drain")

    if text.count("kNodalMat<true>") < 1 or text.count("kNodalEven<true>") < 1:
        raise RuntimeError("batched unfused rollback kernels were removed")
    if text.count("kNodalMat<false>") < 2 or text.count("kNodalEven<false>") < 1:
        raise RuntimeError("per-instance/hybrid rollback kernels were removed")


def apply(text: str) -> str:
    if MARKER in text:
        validate_patched(text)
        return text

    text = replace_once(
        text,
        '#include "CudaXsReconBackend.h"\n\n#include "FlatXsKernel.h"',
        '#include "CudaXsReconBackend.h"\n\n#include "CudaTransferMirror.h"\n#include "FlatXsKernel.h"',
        "include transfer mirror",
    )

    text = replace_once(
        text,
        '''bool envFlagDisabled(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string s(v);
    return s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE";
}
''',
        '''bool envFlagDisabled(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string s(v);
    return s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE";
}

''' + MARKER + '''
/// Mat and Even are node-local and have an exact producer/consumer relation:
/// calculateEven(lk) reads only the matrices and coefficients produced by
/// updateMatrix(lk). Running both bodies in one thread removes one graph node
/// without changing any floating-point expression or cross-node ordering.
bool nodalFuseMatEvenEnabled() {
    static const bool on = !envFlagDisabled("RASBERY_GPU_NODAL_FUSE_MAT_EVEN");
    return on;
}

/// The arena has a separate XS allocation, so hoststateGeneration is not an
/// honest residency key. A byte shadow is: an upload is skipped only when the
/// complete incoming xsrf/xsnf/xssm bytes equal the last successfully drained
/// upload for that same slot and destination.
bool nodalXsMirrorEnabled() {
    static const bool on = !envFlagDisabled("RASBERY_GPU_NODAL_XS_MIRROR");
    return on;
}
''',
        "runtime refactor gates",
    )

    text = replace_once(
        text,
        'std::atomic<unsigned long long> g_nodal_batch_linger_us{0};\n',
        '''std::atomic<unsigned long long> g_nodal_batch_linger_us{0};
std::atomic<unsigned long long> g_nodal_batch_xs_h2d_bytes{0};
std::atomic<unsigned long long> g_nodal_batch_xs_h2d_skipped_bytes{0};
''',
        "nodal XS telemetry counters",
    )

    text = replace_once(
        text,
        '''template <bool BATCHED>
__global__ void kNodalEven(ndl::NodalView base, const std::uint32_t* __restrict__ active) {
    RASBERY_NODAL_SLOT_GUARD(base, active, v);
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalCalculateEven(v, lk, ndl::StaticForms{});
}
''',
        '''template <bool BATCHED>
__global__ void kNodalEven(ndl::NodalView base, const std::uint32_t* __restrict__ active) {
    RASBERY_NODAL_SLOT_GUARD(base, active, v);
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalCalculateEven(v, lk, ndl::StaticForms{});
}

template <bool BATCHED>
__global__ void kNodalMatEven(ndl::NodalView base,
                              const std::uint32_t* __restrict__ active) {
    RASBERY_NODAL_SLOT_GUARD(base, active, v);
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk >= v.nxyz) return;
    const ndl::StaticForms forms{};
    ndl::nodalUpdateMatrix(v, lk, forms);
    ndl::nodalCalculateEven(v, lk, forms);
}
''',
        "fused Mat+Even kernel",
    )

    text = replace_once(
        text,
        '''        _use_graph = !envFlagDisabled("RASBERY_GPU_NODAL_GRAPH");
        init(proto);
''',
        '''        _use_graph     = !envFlagDisabled("RASBERY_GPU_NODAL_GRAPH");
        _fuse_mat_even = nodalFuseMatEvenEnabled();
        _mirror_xs     = nodalXsMirrorEnabled();
        init(proto);
''',
        "arena feature flags",
    )

    text = replace_once(
        text,
        '''        bool               have_const = false, have_chif = false;
        unsigned long long res_const_gen = 0, res_ref_gen = 0;
''',
        '''        bool               have_const = false, have_chif = false;
        unsigned long long res_const_gen = 0, res_ref_gen = 0;
        cuda_transfer::ByteExactMirror<double> xsrf_mirror;
        cuda_transfer::ByteExactMirror<double> xsnf_mirror;
        cuda_transfer::ByteExactMirror<double> xssm_mirror;
''',
        "slot byte mirrors",
    )

    text = replace_once(
        text,
        '''        kNodalTrl0<true><<<dim3(static_cast<unsigned>(gg), S), B, 0, _stream>>>(_base, _d_active);
        kNodalTrl12<true><<<dim3(static_cast<unsigned>(gg), S), B, 0, _stream>>>(_base, _d_active);
        kNodalMat<true><<<dim3(static_cast<unsigned>(gn), S), B, 0, _stream>>>(_base, _d_active);
        kNodalEven<true><<<dim3(static_cast<unsigned>(gn), S), B, 0, _stream>>>(_base, _d_active);
        kNodalJnet<true><<<dim3(static_cast<unsigned>(gs), S), B, 0, _stream>>>(_base, _d_active);
''',
        '''        kNodalTrl0<true><<<dim3(static_cast<unsigned>(gg), S), B, 0, _stream>>>(_base, _d_active);
        kNodalTrl12<true><<<dim3(static_cast<unsigned>(gg), S), B, 0, _stream>>>(_base, _d_active);
        if (_fuse_mat_even) {
            kNodalMatEven<true><<<dim3(static_cast<unsigned>(gn), S), B, 0, _stream>>>(
                _base, _d_active);
        } else {
            kNodalMat<true><<<dim3(static_cast<unsigned>(gn), S), B, 0, _stream>>>(
                _base, _d_active);
            kNodalEven<true><<<dim3(static_cast<unsigned>(gn), S), B, 0, _stream>>>(
                _base, _d_active);
        }
        kNodalJnet<true><<<dim3(static_cast<unsigned>(gs), S), B, 0, _stream>>>(_base, _d_active);
''',
        "arena Mat+Even launch",
    )

    text = replace_once(
        text,
        '''        const std::size_t ndgb = _cnt_ndg * sizeof(double);

        for (int m : part) pinSlot(_slot[static_cast<std::size_t>(m)]);
''',
        '''        const std::size_t ndgb = _cnt_ndg * sizeof(double);

        struct PendingXsMirror {
            int  slot = -1;
            bool xsrf = false;
            bool xsnf = false;
            bool xssm = false;
        };
        std::vector<PendingXsMirror> pending_xs;
        pending_xs.reserve(part.size());
        unsigned long long xs_h2d_bytes = 0;
        unsigned long long xs_h2d_skipped_bytes = 0;

        for (int m : part) pinSlot(_slot[static_cast<std::size_t>(m)]);
''',
        "pending XS mirror state",
    )

    text = replace_between_once(
        text,
        '            // xsrf/xsnf/xssm upload UNCONDITIONALLY',
        '            if (!memcpyAsyncOrFail(_base.jnet',
        '''            // The arena cannot use hoststateGeneration because its XS
            // allocation is separate from the per-instance backend. Instead,
            // compare every byte with the last upload that completed
            // successfully for this slot. This is conservative for all NaN
            // payloads and signed zeroes, and never assumes a physics
            // generation implies residency.
            const bool push_xsrf =
                !_mirror_xs || !sl.xsrf_mirror.matches(sl.h_xsrf, _cnt_ng1);
            const bool push_xsnf =
                !_mirror_xs || !sl.xsnf_mirror.matches(sl.h_xsnf, _cnt_ng1);
            const bool push_xssm =
                !_mirror_xs || !sl.xssm_mirror.matches(sl.h_xssm, _cnt_ng2);

            if (push_xsrf) {
                if (!memcpyAsyncOrFail(const_cast<double*>(_base.xsrf) + s * _cnt_ng1,
                                       sl.h_xsrf, ng1b, cudaMemcpyHostToDevice,
                                       "nodal arena xsrf H2D"))
                    return drained();
                xs_h2d_bytes += ng1b;
            } else {
                xs_h2d_skipped_bytes += ng1b;
            }
            if (push_xsnf) {
                if (!memcpyAsyncOrFail(const_cast<double*>(_base.xsnf) + s * _cnt_ng1,
                                       sl.h_xsnf, ng1b, cudaMemcpyHostToDevice,
                                       "nodal arena xsnf H2D"))
                    return drained();
                xs_h2d_bytes += ng1b;
            } else {
                xs_h2d_skipped_bytes += ng1b;
            }
            if (push_xssm) {
                if (!memcpyAsyncOrFail(const_cast<double*>(_base.xssm) + s * _cnt_ng2,
                                       sl.h_xssm, ng2b, cudaMemcpyHostToDevice,
                                       "nodal arena xssm H2D"))
                    return drained();
                xs_h2d_bytes += ng2b;
            } else {
                xs_h2d_skipped_bytes += ng2b;
            }
            if (_mirror_xs)
                pending_xs.push_back({m, push_xsrf, push_xsnf, push_xssm});
''',
        "replace unconditional arena XS uploads",
    )

    text = replace_once(
        text,
        '''        const cudaError_t src = cudaStreamSynchronize(_stream);
        if (src != cudaSuccess) { fail("nodal arena drain", src); return false; }
        g_nodal_d2h_bytes.store(2 * sgb, std::memory_order_relaxed);
''',
        '''        const cudaError_t src = cudaStreamSynchronize(_stream);
        if (src != cudaSuccess) { fail("nodal arena drain", src); return false; }

        // Commit only after the drain proved the queued H2D reached the device.
        // A failed batch leaves the previous mirrors untouched, so they never
        // describe bytes that may not be resident.
        for (const PendingXsMirror& pending : pending_xs) {
            Slot& sl = _slot[static_cast<std::size_t>(pending.slot)];
            if (pending.xsrf) sl.xsrf_mirror.commit(sl.h_xsrf, _cnt_ng1);
            if (pending.xsnf) sl.xsnf_mirror.commit(sl.h_xsnf, _cnt_ng1);
            if (pending.xssm) sl.xssm_mirror.commit(sl.h_xssm, _cnt_ng2);
        }
        g_nodal_batch_xs_h2d_bytes.fetch_add(xs_h2d_bytes, std::memory_order_relaxed);
        g_nodal_batch_xs_h2d_skipped_bytes.fetch_add(
            xs_h2d_skipped_bytes, std::memory_order_relaxed);
        g_nodal_d2h_bytes.store(2 * sgb, std::memory_order_relaxed);
''',
        "commit mirrors after drain",
    )

    text = replace_once(
        text,
        '''    cudaGraphExec_t _graph     = nullptr;
    bool            _use_graph = true;
''',
        '''    cudaGraphExec_t _graph     = nullptr;
    bool            _use_graph = true;
    bool            _fuse_mat_even = true;
    bool            _mirror_xs = true;
''',
        "arena refactor members",
    )

    text = replace_once(
        text,
        '''                  << ",\\\"full_mode\\\":"
                  << (rasbery::rasberyGpuNodalFullEnabled() ? 1 : 0)
                  << ",\\\"graph_launches\\\":"
''',
        '''                  << ",\\\"full_mode\\\":"
                  << (rasbery::rasberyGpuNodalFullEnabled() ? 1 : 0)
                  << ",\\\"mat_even_fused\\\":"
                  << (nodalFuseMatEvenEnabled() ? 1 : 0)
                  << ",\\\"graph_launches\\\":"
''',
        "GPU receipt fusion flag",
    )

    text = replace_once(
        text,
        '''             << ",\\\"last_linger_us\\\":"
             << g_nodal_batch_linger_us.load(std::memory_order_relaxed)
             << ",\\\"width_histogram\\\":[";
''',
        '''             << ",\\\"last_linger_us\\\":"
             << g_nodal_batch_linger_us.load(std::memory_order_relaxed)
             << ",\\\"xs_mirror\\\":" << (nodalXsMirrorEnabled() ? 1 : 0)
             << ",\\\"xs_h2d_bytes\\\":"
             << g_nodal_batch_xs_h2d_bytes.load(std::memory_order_relaxed)
             << ",\\\"xs_h2d_skipped_bytes\\\":"
             << g_nodal_batch_xs_h2d_skipped_bytes.load(std::memory_order_relaxed)
             << ",\\\"width_histogram\\\":[";
''',
        "batch receipt XS telemetry",
    )

    text = replace_once(
        text,
        '''        kNodalTrl0<false><<<gng, B, 0, d.stream>>>(v, nullptr);
        kNodalTrl12<false><<<gng, B, 0, d.stream>>>(v, nullptr);
        kNodalMat<false><<<gn, B, 0, d.stream>>>(v, nullptr);
        kNodalEven<false><<<gn, B, 0, d.stream>>>(v, nullptr);
        kNodalJnet<false><<<gs, B, 0, d.stream>>>(v, nullptr);
''',
        '''        kNodalTrl0<false><<<gng, B, 0, d.stream>>>(v, nullptr);
        kNodalTrl12<false><<<gng, B, 0, d.stream>>>(v, nullptr);
        if (nodalFuseMatEvenEnabled()) {
            kNodalMatEven<false><<<gn, B, 0, d.stream>>>(v, nullptr);
        } else {
            kNodalMat<false><<<gn, B, 0, d.stream>>>(v, nullptr);
            kNodalEven<false><<<gn, B, 0, d.stream>>>(v, nullptr);
        }
        kNodalJnet<false><<<gs, B, 0, d.stream>>>(v, nullptr);
''',
        "per-instance FULL Mat+Even launch",
    )

    validate_patched(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.in_place and args.output is not None:
        parser.error("choose --in-place or --output, not both")
    original = args.source.read_text(encoding="utf-8")
    patched = apply(original)
    if args.check:
        print("nodal CUDA refactor anchors: PASS")
        return 0
    target = args.source if args.in_place else args.output
    if target is None:
        parser.error("provide --in-place or --output")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(patched, encoding="utf-8", newline="\n")
    print(f"nodal CUDA refactor: {'UNCHANGED' if patched == original else 'APPLIED'} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
