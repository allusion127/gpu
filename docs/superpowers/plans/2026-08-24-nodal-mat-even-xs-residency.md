# Nodal Mat/Even Fusion and Arena XS Residency Implementation Plan

> **Execution basis:** `docs/GPU_NODAL_SUITABILITY_AND_SPEED_ROADMAP_20260824_KO.md`, latest validated branch `codex/single-gpu-batch-dispatch-v2`.

**Goal:** Remove one node-local CUDA graph kernel from every FULL nodal drive and eliminate byte-identical `xsrf/xsnf/xssm` uploads in the multi-input nodal arena while preserving RASBERY's byte-exact result contract and all CPU/per-instance fallbacks.

**Architecture:** Keep Trl0, Trl12 and Jnet as separate kernels because they cross node/surface boundaries. Fuse only `nodalUpdateMatrix(lk)` and `nodalCalculateEven(lk)` in one thread for the same node. Replace the arena's unsafe generation-based XS residency idea with a host byte shadow committed only after a successful CUDA stream drain. Both changes are default-on in the experimental branch and have environment-variable rollback switches.

**Technology:** CUDA C++17, C++20 host helper, CUDA Graph, Python fail-closed source transformer, GitHub Actions, static/compiled contract tests.

---

## Task 1: Byte-exact transfer mirror

**Files:**
- Create `src/CudaTransferMirror.h`
- Create `tools/test_cuda_transfer_mirror.py`

1. Write a compile-and-run test covering initial invalid state, exact match, changed data, signed zero and distinct NaN payloads.
2. Implement `ByteExactMirror<T>` with `memcmp`, `commit`, and `invalidate`.
3. Run the test with `-std=c++20 -Wall -Wextra -Werror`.

## Task 2: Fail-closed CUDA source transformation

**Files:**
- Create `tools/apply_nodal_gpu_refactor.py`
- Create `tools/test_apply_nodal_gpu_refactor.py`

1. Define exact anchors for the v2 source and require each to match once.
2. Add the transfer mirror include, feature gates, counters and fused kernel.
3. Replace the batch and per-instance FULL launch pairs with a fused/default and unfused/rollback branch.
4. Replace unconditional arena XS uploads with byte-match decisions and post-drain commits.
5. Add telemetry fields and validate idempotence.
6. Make any anchor drift fail the build.

## Task 3: Static safety contract

**Files:**
- Create `tools/test_nodal_gpu_refactor_contract.py`

1. Verify phase order `Trl0 -> Trl12 -> MatEven -> Jnet`.
2. Verify the unfused kernels remain compiled as rollback paths.
3. Verify mirror commits appear only after `cudaStreamSynchronize`.
4. Verify telemetry and both rollback environment variables are present.

## Task 4: Apply on the actual GitHub checkout

**Files:**
- Modify `src/CudaXsReconBackend.cu`
- Create `.github/workflows/apply-nodal-gpu-refactor.yml`

1. Create a branch from `codex/single-gpu-batch-dispatch-v2`.
2. Push helper/tests/plan first.
3. Trigger the path-scoped workflow by adding it last.
4. In GitHub Actions, check anchors, apply the source transformation, run all tests and `git diff --check`.
5. Commit and push the generated direct source change only when every gate passes.

## Task 5: Server verification before merge

**Required but not available on the generic GitHub runner:**

1. CUDA 13 / sm_120 production build on server 238.
2. Nodal graph on/off and MatEven fusion on/off A/B.
3. XS mirror on/off A/B with receipt counters.
4. Full 500/500 HDF5 dataset byte comparison.
5. M1 and M64 timing, at least three repetitions per arm.
6. Reject if graph/drive fallback is non-zero or throughput regresses.

**Runtime rollback:**

```bash
RASBERY_GPU_NODAL_FUSE_MAT_EVEN=0
RASBERY_GPU_NODAL_XS_MIRROR=0
```
