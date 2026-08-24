#!/usr/bin/env python3
"""Unit-test the fail-closed source transformer on a minimal v2-shaped fixture."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "apply_nodal_gpu_refactor.py"

spec = importlib.util.spec_from_file_location("nodal_refactor", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture() -> str:
    return r'''#include "CudaXsReconBackend.h"

#include "FlatXsKernel.h"

bool envFlagDisabled(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string s(v);
    return s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE";
}

std::atomic<unsigned long long> g_nodal_batch_linger_us{0};

template <bool BATCHED>
__global__ void kNodalEven(ndl::NodalView base, const std::uint32_t* __restrict__ active) {
    RASBERY_NODAL_SLOT_GUARD(base, active, v);
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalCalculateEven(v, lk, ndl::StaticForms{});
}

class NodalArena {
public:
    NodalArena(const ndl::NodalView& proto, int slots) {
        _use_graph = !envFlagDisabled("RASBERY_GPU_NODAL_GRAPH");
        init(proto);
    }
private:
    struct Slot {
        bool               have_const = false, have_chif = false;
        unsigned long long res_const_gen = 0, res_ref_gen = 0;
        const double* h_xsrf = nullptr;
        const double* h_xsnf = nullptr;
        const double* h_xssm = nullptr;
        const double* h_jnet = nullptr;
        const double* h_flux = nullptr;
    };

    void enqueueKernels() {
        kNodalTrl0<true><<<dim3(static_cast<unsigned>(gg), S), B, 0, _stream>>>(_base, _d_active);
        kNodalTrl12<true><<<dim3(static_cast<unsigned>(gg), S), B, 0, _stream>>>(_base, _d_active);
        kNodalMat<true><<<dim3(static_cast<unsigned>(gn), S), B, 0, _stream>>>(_base, _d_active);
        kNodalEven<true><<<dim3(static_cast<unsigned>(gn), S), B, 0, _stream>>>(_base, _d_active);
        kNodalJnet<true><<<dim3(static_cast<unsigned>(gs), S), B, 0, _stream>>>(_base, _d_active);
    }

    bool launchBatch(const std::vector<int>& part) {
        const std::size_t ndgb = _cnt_ndg * sizeof(double);

        for (int m : part) pinSlot(_slot[static_cast<std::size_t>(m)]);
        for (int m : part) {
            Slot& sl = _slot[static_cast<std::size_t>(m)];
            const std::size_t s = static_cast<std::size_t>(m);
            // xsrf/xsnf/xssm upload UNCONDITIONALLY, and `state_gen` is
            // deliberately not used as a residency key here.
            if (!memcpyAsyncOrFail(const_cast<double*>(_base.xsrf) + s * _cnt_ng1, sl.h_xsrf,
                                   ng1b, cudaMemcpyHostToDevice, "nodal arena xsrf H2D") ||
                !memcpyAsyncOrFail(const_cast<double*>(_base.xsnf) + s * _cnt_ng1, sl.h_xsnf,
                                   ng1b, cudaMemcpyHostToDevice, "nodal arena xsnf H2D") ||
                !memcpyAsyncOrFail(const_cast<double*>(_base.xssm) + s * _cnt_ng2, sl.h_xssm,
                                   ng2b, cudaMemcpyHostToDevice, "nodal arena xssm H2D"))
                return drained();
            if (!memcpyAsyncOrFail(_base.jnet + s * _cnt_sg, sl.h_jnet, sgb,
                                   cudaMemcpyHostToDevice, "nodal arena jnet H2D")) return false;
        }
        const cudaError_t src = cudaStreamSynchronize(_stream);
        if (src != cudaSuccess) { fail("nodal arena drain", src); return false; }
        g_nodal_d2h_bytes.store(2 * sgb, std::memory_order_relaxed);
        return true;
    }

    cudaGraphExec_t _graph     = nullptr;
    bool            _use_graph = true;
    std::vector<Slot> _slot;
};

struct NodalReceipt {
    ~NodalReceipt() {
        std::cout << "[RASBERY][NODAL][GPU] {\"drives_solved\":"
                  << g_nodal_drives.load(std::memory_order_relaxed)
                  << ",\"full_mode\":"
                  << (rasbery::rasberyGpuNodalFullEnabled() ? 1 : 0)
                  << ",\"graph_launches\":"
                  << g_nodal_graph_launches.load(std::memory_order_relaxed);
        line << "prefix"
             << ",\"last_linger_us\":"
             << g_nodal_batch_linger_us.load(std::memory_order_relaxed)
             << ",\"width_histogram\":[";
    }
};

void hybrid() {
    kNodalMat<false><<<gn, B>>>(v, nullptr);
}
void full() {
        kNodalTrl0<false><<<gng, B, 0, d.stream>>>(v, nullptr);
        kNodalTrl12<false><<<gng, B, 0, d.stream>>>(v, nullptr);
        kNodalMat<false><<<gn, B, 0, d.stream>>>(v, nullptr);
        kNodalEven<false><<<gn, B, 0, d.stream>>>(v, nullptr);
        kNodalJnet<false><<<gs, B, 0, d.stream>>>(v, nullptr);
}
'''


def main() -> int:
    original = fixture()
    patched = mod.apply(original)
    assert mod.MARKER in patched
    assert patched == mod.apply(patched), "transform must be idempotent"

    try:
        mod.apply(original.replace("kNodalEven<true>", "kNodalEvenChanged<true>"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("anchor drift did not fail closed")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "patched.cu"
        out.write_text(patched, encoding="utf-8")
        assert "sl.xsrf_mirror.commit" in out.read_text(encoding="utf-8")
    print("nodal CUDA source transformer: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
