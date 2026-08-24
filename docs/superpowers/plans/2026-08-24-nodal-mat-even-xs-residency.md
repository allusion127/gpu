# Nodal Mat/Even Fusion and Arena XS Residency Plan

**Execution basis:** `docs/GPU_NODAL_SUITABILITY_AND_SPEED_ROADMAP_20260824_KO.md`
**Base:** `codex/single-gpu-batch-dispatch-v2`

## Goal

Reduce the fixed cost of the existing FULL CUDA nodal pipeline without moving branch-heavy Driver/Scheduler, depletion, search, T/H, or `updateConstant` transcendental work to the GPU.

## Implemented architecture

### Node-local Mat/Even fusion

The former FULL sequence was:

```text
Trl0 -> Trl12 -> Mat -> Even -> Jnet
```

`Mat(lk)` and `Even(lk)` are fused because the second body consumes only intermediate arrays produced for the same node. `Trl0 -> Trl12` remains a kernel boundary because Trl12 reads neighboring nodes. `MatEven -> Jnet` remains a boundary because Jnet reads both sides of a surface.

The existing separate kernels remain compiled as the rollback path:

```bash
RASBERY_GPU_NODAL_FUSE_MAT_EVEN=0
```

### Byte-exact arena XS residency

The nodal arena has separate device storage, so host generation counters do not prove its `xsrf/xsnf/xssm` bytes are current. Each slot therefore keeps a byte-exact shadow of the last H2D transfer that completed successfully. A copy is skipped only when `memcmp` confirms every byte still matches. Mirrors are committed after the stream drain, never when a copy is merely queued.

Upload decisions are stored in three booleans already owned by the slot; no per-batch container or allocator call is introduced.

Rollback:

```bash
RASBERY_GPU_NODAL_XS_MIRROR=0
```

## Retained verification

- `tools/test_cuda_transfer_mirror.py`: compiled C++20 checks for exact match, signed zero, and distinct NaN payloads.
- `tools/test_nodal_gpu_refactor_contract.py`: phase order, rollback kernels, telemetry, and post-drain mirror commit.
- `tools/test_apply_nodal_gpu_hotpath_fix.py`: verifies the final launch path contains no temporary batch container.
- `.github/workflows/apply-nodal-gpu-refactor.yml`: read-only execution of the reusable contracts.

## Server acceptance gate

Before merging:

1. CUDA 13 / sm_120 production build on server 238.
2. Mat/Even fusion on/off and XS mirror on/off A/B.
3. 500/500 HDF5 dataset byte comparison.
4. M1 and M64 timing with at least three repetitions per arm.
5. Graph, arena, and drive fallbacks must all be zero.
6. Accept only measured wall-time or throughput improvement; the code change alone is not a speedup claim.
