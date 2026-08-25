#!/usr/bin/env python3
"""Static integration contract for resident CMFD operator assembly."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA = (ROOT / "src" / "CudaBICGBackend.cu").read_text(encoding="utf-8-sig")
CUDA_H = (ROOT / "src" / "CudaBICGBackend.h").read_text(encoding="utf-8-sig")
BICG = (ROOT / "src" / "BICGCMFD.cpp").read_text(encoding="utf-8-sig")
BICG_H = (ROOT / "src" / "BICGCMFD.h").read_text(encoding="utf-8-sig")
XS = (ROOT / "src" / "XSSet.h").read_text(encoding="utf-8-sig")


def fail(message: str) -> None:
    raise SystemExit(f"cmfd GPU residency contract: FAIL: {message}")


required_cuda = (
    '#include "CmfdAssemblyKernel.h"',
    '#include "CudaTransferMirror.h"',
    'bool cmfdAssemblyEnabled()',
    'RASBERY_GPU_CMFD_ASSEMBLY',
    '__global__ void cmfd_assemble_operator_2g',
    'cmfd_assembly::assembleNode2G(view, l);',
    'device_assembly_active',
    'host_xsrf', 'host_xssm', 'host_dtil', 'host_dhat',
    'xsrf_mirror', 'xssm_mirror', 'dtil_mirror',
    'cmfd_assembly_gpu_calls',
    'cmfd_diag_h2d_elided_bytes',
    'cmfd_cc_h2d_elided_bytes',
    'issueExceptionalOperatorDownloads',
)
missing = [token for token in required_cuda if token not in CUDA]
if missing:
    fail(f"CudaBICGBackend.cu missing {missing}")

required_header = (
    'const double* xsrf', 'const double* xssm',
    'const double* dtil', 'const double* dhat',
    'device_assembly = false',
    'cmfd_assembly_gpu_calls', 'cmfd_assembly_cpu_fallbacks',
    'cmfd_diag_h2d_elided_bytes', 'cmfd_cc_h2d_elided_bytes',
)
missing = [token for token in required_header if token not in CUDA_H]
if missing:
    fail(f"CudaBICGBackend.h missing {missing}")

required_bicg = (
    'bool BICGCMFD::canUseDeviceAssembly() const',
    'void BICGCMFD::assembleHostLinearSystem(const double& eigv)',
    '_device_assembly_pending = canUseDeviceAssembly();',
    'io.device_assembly',
    '_device_assembly_pending;',
    'io.xsrf', '_x.xsrfData();',
    'io.xssm', '_x.xssmData();',
    'io.dtil', '_dtil;',
    'io.dhat', '_dhat;',
)
missing = [token for token in required_bicg if token not in BICG]
if missing:
    fail(f"BICGCMFD.cpp missing {missing}")

if '_device_assembly_pending' not in BICG_H:
    fail("BICGCMFD.h lacks pending ownership flag")
if '_device_assembly_pending = false;' not in BICG:
    fail("reset/exit paths do not clear device operator ownership")
if 'xsrfData()' not in XS or 'xssmData()' not in XS:
    fail("XSSet raw SoA accessors unavailable")

# The assembly launch must precede the sweep loop, and updls remains inside it.
assembly = CUDA.find('cmfd_assemble_operator_2g<<<')
sweep_loop = CUDA.find('for (int sweep = 0; sweep < unroll; ++sweep)', assembly)
updls = CUDA.find('cmfd_updls<<<', sweep_loop)
if not (0 <= assembly < sweep_loop < updls):
    fail("assembly -> sweep loop -> updls order is not preserved")

# Host diag/cc/udiag uploads must be conditional on the slot not owning assembly.
for token in (
    'if (!sl.device_assembly)',
    'telemetry.cmfd_diag_h2d_elided_bytes',
    'telemetry.cmfd_cc_h2d_elided_bytes',
):
    if token not in CUDA:
        fail(f"resident H2D elision token missing: {token}")

# A pending device-owned operator may not silently enter the pristine host loop.
if 'assembleHostLinearSystem(eigv);' not in BICG:
    fail("device assembly fallback does not rebuild the host operator")

print("cmfd GPU residency contract: PASS")
