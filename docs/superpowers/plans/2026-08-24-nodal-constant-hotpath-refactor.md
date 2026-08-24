# Nodal Constant Hot-Path Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CPU-resident `Nodal::updateConstant` phase cheaper and race-free while preserving every generated double bit-for-bit, and isolate its transcendental arithmetic behind a testable boundary for later CUDA/table experiments.

**Architecture:** Keep `exp`/`sqrt` on the host because the current bit-exact contract is tied to the host libm. Move the coefficient formula into `NodalConstantKernel.h`, snapshot node XS/mesh inputs once, and aggregate a per-node dirty bit so `_const_generation` advances once per drive instead of racing inside an OpenMP loop.

**Tech Stack:** C++17/C++20 test harness, OpenMP reduction, Python contract tests, GitHub Actions.

**Spec:** `docs/GPU_NODAL_SUITABILITY_AND_SPEED_ROADMAP_20260824_KO.md`

## Global Constraints

- Base branch: `codex/single-gpu-batch-dispatch-v2`.
- Do not move transcendental evaluation to CUDA without a replay/mining gate.
- Preserve the exact expression order of the historical `updateConstant` implementation.
- Do not change the five CUDA nodal phases or the CPU/GPU fallback policy in this patch.
- Performance numbers require RTX PRO 6000 server measurement; this patch makes no measured speed claim.

---

### Task 1: Extract the bit-exact coefficient body

**Files:**
- Create: `src/NodalConstantKernel.h`
- Create: `tools/test_nodal_constant_kernel.py`

- [x] Write a C++ harness containing the historical formula.
- [x] Verify the test fails while the production header is missing.
- [x] Implement `nodalConstantCoefficients(xsrf, xsdf, hmesh)`.
- [x] Verify 180 material/mesh cases × 9 fields are bit-identical under `-O3 -ffp-contract=fast`.

### Task 2: Integrate cached inputs and a dirty reduction

**Files:**
- Modify: `src/Nodal.cpp`
- Modify: `src/Nodal.h`
- Create: `tools/test_nodal_update_constant_integration.py`

- [x] Make `updateConstant` return whether a node was recomputed.
- [x] Snapshot `xsrf`, `xsdf`, and `hmesh` once per node.
- [x] Use the extracted coefficient body for all nine outputs.
- [x] Combine dirty bits with OpenMP `reduction(|:...)` in both CPU and GPU-dispatch paths.
- [x] Advance `_const_generation` once per drive.
- [x] Verify the integration contract and run `git diff --check`.

### Task 3: Server validation gate

**Files:**
- No production changes.

- [ ] Build with gcc13 + CUDA 13.0, `sm_120`.
- [ ] Run the KNGR 500-dataset CPU golden comparison.
- [ ] Run GPU FULL graph on/off bit comparison.
- [ ] Run M1 and M64 at least three times and report median/dispersion.
- [ ] Accept only if fallbacks remain zero and performance does not regress.
