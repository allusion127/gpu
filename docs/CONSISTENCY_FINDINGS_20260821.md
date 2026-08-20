# Consistency findings and their disposition (2026-08-21)

Working through `ANALYSIS_STRUCTURE_20260820.md` §5.5, plus two items the
analysis did not list. Every "fixed" row below is in the git history of this
tree; every "owed" row needs the GPU238 host or the presentation deck.

| # | Issue | Disposition |
|---|---|---|
| 1 | v8 runner sets three environment variables the source never reads | **Fixed** — `tools/ga_two_stage_40x_pipeline.py`, gated by `tools/test_ga_promotion_gate.py` |
| 2 | V12 deck's speed ratio predates the Xe-fusion build | **Draft below** — deck edit is the user's |
| 3 | `hostcap_width_manifest_20260820.json` carries `status=RUNNING` | **Diagnosed below** — restamp owed on the host |
| 4 | Truncated SHA-256 constant in `patch_harness` | **Fixed** — `docs/PATCH_HARNESS_REAPPLY_WARNING.md` |
| 5 | Authoritative tree dirty and uncommitted (53 items) | **Fixed** — five commits from `e76d40d` |
| 6 | No source-to-binary equivalence for the GPU238 binaries | **Owed** — `docs/MODEL_CACHE_AB_HANDOFF.md` §6 |
| 7 | `model_cache_rebased` has no verdict | **Rebased and testable** — branch `codex/model-cache-rebase`, A/B spec in the handoff |
| 8 | Ledger §5 names a `physics_mode` string the code does not emit | **New — documented below** |
| 9 | Two static gates exist only outside `tools/` | **New — owed**, see below |

---

## 1. Dead environment variables

`RASBERY_FULL_GPU`, `RASBERY_GPU_FORCE_ORDERED_PRECONDITIONER` and
`RASBERY_GPU_DISABLE_CUDA_GRAPHS` return zero hits across `src/**` and
`include/chiffon/**`. They are `rasbery_gpu_codex` leftovers and were no-ops in
every run that set them.

This matters beyond tidiness: an environment block containing
`RASBERY_FULL_GPU=1` is the evidence someone would cite for a "full GPU mode",
and `RASBERY_GPU_FORCE_ORDERED_PRECONDITIONER=1` for a preconditioner choice
that was never made. Neither happened. What actually controls those knobs is
`RASBERY_GPU_GRAPH` (CUDA Graphs, default on) and `RASBERY_PC_MODE`.

`tools/ga_two_stage_40x_pipeline.py` drops all three.
`tools/test_ga_promotion_gate.py` fails if the runner sets them again **or** if
the source starts reading them, so the two cannot drift apart silently.

Still carrying them, not fixed here because it lives in the read-only snapshot:
`02_variants/model_cache_rebased/run_compare_v8.sh` exports all three. Strip
them before reusing it.

---

## 2. V12 presentation numbers — draft correction

**Not applied.** `PPT_..._v12_20260820.pptx` is the user's to edit. This is the
arithmetic, ready to drop in.

The V12 deck used the hostcap build `e17aa942…`, which finishes 64 cases in
`117.77 s`. The Xe-fusion build `d764e8d3…` finishes the same 64 cases in
`89.42 s` (64 × 3600 ÷ 2,576.60 cases/h). Every ratio on slides 15, 27 and 51
moves.

### Raw throughput ratio at matched concurrency (MASTER = 1)

| N | MASTER cases/h | GPU, V12 build | ratio (published) | GPU, Xe-fusion | ratio (corrected) |
|---:|---:|---:|---:|---:|---:|
| 1 | 256.15 | 233.7 | **0.91×** | 362.17 | **1.41×** |
| 64 | 1,518.58 | 1,956.4 | **1.29×** | 2,576.60 | **1.70×** |
| 128 | 1,542.12 | ~1,962 | **1.27×** | 2,594.30 | **1.68×** |

The N=1 row is the one worth saying out loud: the published deck shows GPU
RASBERRY *losing* at single concurrency, and with Xe fusion it no longer does.

### The other framing, if the deck keeps it

| Comparison | Value |
|---|---:|
| GPU M64 (2,576.60) ÷ MASTER W16 recommended (1,452.35) | 1.77× |
| GPU M128 (2,594.30) ÷ MASTER W16 | 1.79× |
| GPU M128 ÷ MASTER W192 maximum (1,558.83) | 1.66× |

### Interpretation limit — carry it forward verbatim

The V12 receipt's own caveat still holds and gets *more* important as the
number rises: MASTER runs APRQ CY02 at ~24 statepoints with full `MAS_SUM` and
restart output; GPU RASBERRY runs APR quarter standard at 1 statepoint with a
scalar light receipt. These ratios are raw benchmark diagnostics at matched
concurrency, not solver speedups. A publishable speedup needs the same deck,
state grid, convergence criteria and output policy on both sides — the
"identical-workload MASTER gate" that ANALYSIS §5.4(A) item 5 still lists as
not started.

Two figures that should stay on the slide next to any of these numbers:

- GPU0 utilization is **11–16%** at every batch width, peak 18–23%.
- CUDA kernels are **4.7%** of the M1 wall, so the Amdahl ceiling on the GPU
  contribution alone is **~1.05×**.

Nearly all of the ratio is 64 CPU cores and a lighter workload definition. The
Xe fusion is a genuine host-side algorithmic gain (+31.7% at M64) and is worth
claiming as such; it is not GPU acceleration.

---

## 3. `hostcap_width_manifest_20260820.json` — why it says RUNNING

Diagnosed, not merely noted.

`gpu0_hostcap_width_sweep_20260820.py` writes the manifest incrementally after
every run and only stamps a terminal status at the end:

- `FAIL_CLOSED` the moment any run fails `accepted()`
- `PASS` after the width-96 **and** width-128 extension runs both complete

The captured manifest holds 6 runs — width 64 at host caps 12/16/24/32/64, then
width 96 — and every one has `accepted=true`, `return_code=0`, no unexpected
full outputs, no error marker, and a 64- or 96-group scalar comparison. The
status is not a failure signal: **the sweep was truncated before the width-128
run finished.** `RUNNING` is the initial value, never overwritten.

The missing point is not missing evidence. Width 128 / host_threads 64 is
measured twice in `extended_width_manifest_20260820.json` (`status=PASS`) and
`width128_repeats_manifest_20260820.json` (`status=PASS`, 2 repeats).

Also worth recording from this manifest, because it is the direct evidence for
ledger §3's "production default NO-GO": the sweep's own baseline field is
width 64 / host 64 at `117.77 s`, and its measured `m64_repeat64` run came in
at `118.9 s`. The host-thread cap did not beat the baseline at M64.

**Owed on the host:** restamp the manifest `PASS` with a `truncated_at` note,
or mark it `SUPERSEDED` by the two width-128 manifests. As it stands it is
neither evidence nor the absence of it, which is the worst of the three.

---

## 8. Ledger §5 names a `physics_mode` string the code does not emit

`GPU_RASBERRY_40X_OPTIMIZATION_LEDGER_20260820.md` §5:

> receipt는 `physics_mode=ga_screen_bounded_feedback` … 를 기록한다.

The code emits `ga_screen_feedback_limited`:

- `include/chiffon/BatchLightResult.h:256`
- `src/main.cpp:212`

and both runners match the code, so nothing is broken at runtime. But an
auditor grepping receipts for the string the ledger names finds nothing and
would reasonably conclude the contract is not implemented. `ga_screen_bounded_feedback`
appears nowhere in the source.

Fix the ledger, not the code — the string is embedded in every receipt already
written, and changing it would invalidate the 256→16 evidence.

---

## 9. Two static gates live outside `tools/`

`ANALYSIS_STRUCTURE_20260820.md` §3.4 records `test_interpolator_batch_race.py`
and `test_batch_hdf5_guard.py` as PASS, but notes they exist only in the
isolated source, not in the consolidated `tools/`. They are not in this tree
either.

So the Interpolator race guard v2 and the recursive HDF5 guard — both landed as
correctness fixes in `combined_patch_receipt_20260820.md`, both specifically
about 64 Drivers in one process — currently have **no standing gate**. They are
the surface where a regression would be a data race: intermittent, load
dependent, and invisible in a scalar receipt.

Re-land them from the isolated source, or re-derive them against
`include/chiffon/Hdf5Guard.h` and `include/chiffon/Interpolator.h`.
