# Screen-Exact MASTER W16 Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a fail-closed batch campaign runner that may use approximate feedback-limited screening to rank many candidates, then exact-reruns every selected survivor and reports effective candidate throughput against a measured MASTER W16 baseline.

**Architecture:** The runner invokes the existing RASBERY executable in batches no wider than 64. Screen and exact outputs use disjoint paths and environments; the tool parses the screen receipt contract, selects survivors, removes every approximation variable for exact runs, validates exact outputs, and writes one auditable receipt containing both wall times and SHA-256 identities.

**Tech Stack:** Python 3.12 standard library, existing RASBERY CLI, JSON/JSONL light receipts, optional HDF5 validator command.

**Spec:** `docs/EXACT_AND_SCREENING_THROUGHPUT_STRATEGY_20260825_KO.md`

## Global Constraints

- A screen output is never accepted as an exact physics output.
- Every screen record must contain `physics_mode=ga_screen_feedback_limited` and `requires_exact_rerun=true`.
- `--batch-width` is restricted to 1–64.
- Duplicate input paths are rejected.
- Exact runs remove `RASBERY_BATCH_LIGHT_RESULT` and `RASBERY_GA_FEEDBACK_PASSES` from the environment.
- The reported speedup denominator is a user-supplied measured MASTER W16 cases/h value.
- A campaign returns success only when every survivor exact output is non-empty, every optional validator passes, and the target speedup is met.

---

### Task 1: Fail-closed campaign runner

**Files:**
- Create: `tools/run_screen_exact_campaign.py`
- Test: `tools/test_screen_exact_campaign.py`

**Interfaces:**
- Consumes: candidate manifest, RASBERY executable, score key, survivor count, MASTER W16 cases/h.
- Produces: `campaign_receipt.json`, `screen_ranking.json`, independent screen/exact outputs and logs.

- [x] **Step 1: Write failing contracts for batch-width, duplicate paths, approximate receipt labels, survivor exact outputs, and target exit code.**
- [x] **Step 2: Run the contracts and confirm the runner is absent.**
- [x] **Step 3: Implement manifest loading, batched execution, receipt parsing, exact reruns, validation, throughput calculation, and fail-closed exit codes.**
- [x] **Step 4: Run `python tools/test_screen_exact_campaign.py`.**
- [x] **Step 5: Run `python -m py_compile tools/run_screen_exact_campaign.py tools/test_screen_exact_campaign.py`.**

### Task 2: Reproducibility and operator documentation

**Files:**
- Create: `docs/EXACT_AND_SCREENING_THROUGHPUT_STRATEGY_20260825_KO.md`
- Create: `docs/SCREEN_EXACT_CAMPAIGN_USAGE_20260825_KO.md`

**Interfaces:**
- Consumes: current 213.9 cases/h exact GPU measurement, MASTER W16 216–218 cases/h measurement, existing GA screen contract.
- Produces: explicit distinction between exact throughput and effective design-candidate throughput, reproducible command and acceptance rules.

- [x] **Step 1: Document why kernel-only optimization cannot substantiate a 20× exact-throughput claim from the current baseline.**
- [x] **Step 2: Define the screen→exact speedup equation and evidence requirements.**
- [x] **Step 3: Document manifest, command, outputs, exit codes, validator integration, and interpretation limits.**

### Task 3: Continuous contract gate

**Files:**
- Create: `.github/workflows/screen-exact-throughput-contracts.yml`

**Interfaces:**
- Consumes: Task 1 scripts.
- Produces: read-only GitHub Actions verification for syntax, fake-runner integration, and patch integrity.

- [x] **Step 1: Add Python 3.12 syntax compilation.**
- [x] **Step 2: Execute the self-contained fake-RASBERY campaign contracts.**
- [x] **Step 3: Run PR-wide or commit-local `git diff --check`.**

## Server Acceptance

The workflow validates control contracts, not APR1400 physics. Production adoption additionally requires:

1. Use the same candidate set and objective as the MASTER W16 reference.
2. Run at least three interleaved campaigns.
3. Validate every exact survivor with the normal HDF5 and MASTER comparison gates.
4. Include screen and exact wall in the reported effective cases/h.
5. Report objective correlation and top-k recall.
6. Reject any campaign with exact failure, non-finite physics, graph fallback, or missing `requires_exact_rerun` evidence.
