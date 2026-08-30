# WP7 단계 C — Xe device transaction

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `RASBERY_GPU_XE` split Xe arm의 safeguarded Anderson step |
| 상위 계획 | `docs/GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` WP7 단계 C, §3.3, §6 |
| 게이트 등급 | **N1 (현 device arm 대비 — 181에서 측정, §9)** / N1 (host arm 대비, 이미 건너간 선) |
| 등급 이력 | 출하 시 **B0 (현 device arm 대비)**를 주장했다. 2026-08-30 호스트 181의 실측이 그것을 반증했고 §9가 이유와 함께 강등을 기록한다. 2026-08-31: 원인(device mask가 host call site와 다른 algebra 비트를 실었다)을 §9.6의 **합성**으로 제거했다 — 등급은 **181에서 env 없는 TXN 0 vs 1 h5diff가 나올 때까지 N1로 둔다**(§9.6.6). |
| 플래그 | `RASBERY_GPU_XE_TXN=1`, 기본 `0` |
| receipt | `[RASBERY][XE_GPU]` — `txn_steps`, `txn_accepted`, `txn_declined`, `host_syncs`, `host_syncs_per_step`, `d2h_bytes`, `d2h_bytes_per_step`, `xe_device_steps`, `forms_audits`, `forms_audit_mismatch`, `forms_audit_mask`, `policy_note` |
| 계약 테스트 | `tools/test_xe_txn_contract.py` (순수 python, negative control 8종), `tools/test_xe_forms_audit_contract.py` (§9.5의 계측, negative control 17종), `tools/test_xe_forms_host_consistency_contract.py` (§9.6의 합성, negative control 20종) |
| 소스 | `src/XeKernel.h`, `src/CudaXsReconBackend.{h,cu}`, `src/Driver.h`, `src/XeGpuReceipt.h`, `src/XeAndersonReference.{h,cpp}`, `src/XeAlgebraReference.cpp`, `src/XeFormMine.h`, `src/XeFormMask.h`, `src/XeFormMiner.cpp`, `src/XeFormAudit.{h,cpp}`, `src/XSSet.{h,cpp}` |
| 기준 덱 | KNGR, `nxyz = 8,451`, `NG = 2`, `NISO = 39`, `NXS = 11`, `XE_DEPTH = 2`, `XE_DOT_PARTITIONS = 1024` |

> 이 문서의 sync 수는 **모델이 아니라 계약**이다. `tools/test_xe_txn_contract.py`가
> `xeEvaluate` / `xeDots` / `xeCandidate` / `xeCommit` / `xeTransaction`의 본문에서
> `cudaStreamSynchronize`(및 이를 감싼 `xeSync()`) 호출을 직접 세어 아래 표와 대조한다.
> 커널이나 sync가 하나 추가되면 테스트가 실패하고 이 문서를 함께 고쳐야 한다.
> 실행 중의 실측값은 `[XE_GPU]`의 `host_syncs_per_step` / `d2h_bytes_per_step`이 말한다 —
> **census를 주장하는 런이 그 census를 스스로 인쇄한다.**

---

## 1. 왜 Xe인가, 그리고 왜 sync인가

§3.3의 배치 GPU 시간 Amdahl 표는 Xe Anderson을 **26.9 %**로 지목한다(kXeEvaluate 14.4 %,
kXeCommit 11.1 %). 50 % 단축이면 전체 약 1.16×, 완전 제거 상한이 약 1.37×이다. FlatXS 다음의
두 번째 레버이고, **배치가 이 레버가 실제로 돈이 되는 곳**이다.

그런데 단계 C가 겨냥하는 것은 그 26.9 %의 산술이 아니다. 한 Anderson step은 지금
**네 번의 host stream synchronization**을 거치는데, 그중 세 번은 커널 시간을 줄이려는 것이
아니라 **여덟 개 double짜리 결정 하나를 host가 보기 위해** 존재한다.

- 이 step이 armed인가 (`picard`)
- 2×2 정규방정식이 조건을 만족했는가 (`dots[6]`)
- candidate가 safeguard를 통과했는가 (`physics_bad`, `step`)

이 산술이 host에 있는 이유는 비싸서가 아니라 **거기에 쓰였기 때문**이다. 그래서 그대로 옮긴다.

---

## 2. Census — 한 Anderson step의 host 관측, before/after

### 2.1 채택된(accepted) step

| # | 단계 | 커널 | memset | D2D | H2D | D2H | **SYNC** |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `xeEvaluate` (`CudaXsReconBackend.cu:2656`) | 1 | 1 (16 B) | 0 | 1 (phif, 135,216 B) | 1 (8 B) | **1** |
| 2 | `xeRotateHistory` (`:2707`) | 0 | 0 | 6 (n_fuel×8 B 각) | 0 | 0 | 0 |
| 3 | `xeRecordColumn` (`:2727`) | 2 | 0 | 0 | 0 | 0 | 0 |
| 4 | `xeSaveEvaluation` (`:2743`) | 0 | 0 | 6 | 0 | 0 | 0 |
| 5 | `xeDots` (`:2760`) | 2 | 1 (48 B) | 0 | 2 (72 B) | 1 (48 B) | **1** |
| 6 | `xeCandidate` (`:2821`) | 1 | 2 (12 B) | 0 | 0 | 2 (12 B) | **1** |
| 7 | `xeCommit` (`:2856`) | 1 | 1 (8 B) | 0 | 0 | 14 (1,960,640 B) | **1** |
| | **TXN=0 합계** | **7** | **5** | **12** | **3** | **18** | **4** |
| | **TXN=1 합계** (`xeTransaction`, `:2939`) | **8** | **2** | **0** | **1** | **15** | **1** |

TXN=1의 8 커널: `kXeEvaluate` → `kXeHistory` → `kXeDotStage1` → `kXeDotStage2` →
`kXeAndersonSolve`(1 thread) → `kXeCandidateTxn` → `kXeAndersonGate`(1 thread) → `kXeCommitTxn`.

```
host sync per accepted step        4 -> 1        (-75 %)
device API call per accepted step  45 -> 26      (-42 %)
D2D copy per step                  12 -> 0       (n_fuel≈5,000 기준 480 KB)
D2H byte per step                  1,960,708 -> 1,960,720   (+12 B, 사실상 불변)
```

**D2H 바이트가 줄지 않는 것은 실패가 아니라 범위다.** 1.96 MB 중 1.96 MB가 `drainXeCommit`의
`_xs`(11 slot + scatter) + `_iden` 3행이고, 그것은 이 arm이 지울 수 있는 것이 아니다. 계획
WP7 단계 C가 **“state arrays가 device resident이면 `_xs/_iden`의 중간 D2H를 생략한다”**고 조건을
단 이유가 이것이고, residency는 이 work package가 아니다. 단계 C가 지운 것은
**materialization을 사지 않던 sync 전부**다.

### 2.2 거부된(rejected) step — 여기가 더 큰 몫이다

TXN=0에서 거부는 `false`를 반환하고 caller가 `UpdateEquilibriumXenon`을 돌린다. 그 경로는
split arm에서 **evaluate를 한 번 더** 수행한다(`XSSet.cpp:4086 TryUpdateEquilibriumXenonGpuSplit`).

| 거부 사유 | TXN=0 sync | TXN=0 `kXeEvaluate` launch | TXN=1 sync | TXN=1 evaluate |
|---|---:|---:|---:|---:|
| not armed (`ncol==0` 또는 `picard < 1e-6`) | 1 + 2 = **3** | 2 | **1** | 1 |
| condition (SAFEGUARD 1/4) | 2 + 2 = **4** | 2 | **1** | 1 |
| residual (SAFEGUARD 2/4) | 2 + 2 = **4** | 2 | **1** | 1 |
| physics (SAFEGUARD 3/4) | 3 + 2 = **5** | 2 | **1** | 1 |
| step / trust region (SAFEGUARD 4/4) | 3 + 2 = **5** | 2 | **1** | 1 |

거부된 step마다 **전 fuel node에 대한 39-isotope condensation 한 번이 통째로 사라진다.**
`[XE_GPU]`의 `fuel_node_evaluations`가 TXN=1에서 줄어드는 것은 그 때문이고, 이것은
**receipt가 달라져도 되는 유일한 항목**이다(§5).

### 2.3 두 상수의 출처

- `nxyz = 8,451` — `docs/GPU_RASBERY_METHODOLOGY_BENCHMARK_20260824_KO.md:24`
- `n_fuel ≈ 5,000` — `docs/GPU_RASBERY_100_PERCENT_GPU_ASYNC_SCHEDULER_PLAN_REV7_1_KO.md:684`의
  “약 15,000항(= 3 성분 × n_fuel)”에서 역산한 **추정치**. 238 런의
  `fuel_node_evaluations / xe_device_steps`가 정확한 값을 준다. D2D 480 KB와 phif 135 KB만 이
  추정치에 의존하고, sync 수와 D2H 바이트는 의존하지 않는다.

---

## 3. 무엇이 옮겨갔고, 무엇이 옮겨가지 않았는가

### 3.1 (a) 단일 block 커널로 옮긴 Anderson 대수

`kXeAndersonSolve<<<1,1>>>` 한 thread가 `XeKernel.h`의 `xeAndersonSolveControl`을 수행한다.
소비 순서는 바뀌지 않는다:

- `dots`를 `XeDotSlot` 순서(GG, A, B, C, P, Q)로 읽는다 — `kXeDotStage2`가 쓴 순서, host가 읽던 순서.
- arming → 2열 정규방정식 → 1열 secant fallback → SAFEGUARD 2/4(residual) 순서 그대로.
- 상수(`XE_EQUILIBRIUM_TOLERANCE`, `XE_ANDERSON_MIN_GRAM`, trust region)는 **인자로 전달**된다.
  `XeKernel.h`는 그중 어느 것도 선언하지 않으며, 계약 테스트가 그 부재를 검사한다. 두 번째
  철자는 두 번째 의견이고, 그 의견은 조용히 어긋난다.

grid 전역 reduction이 필요한 SAFEGUARD 3/4(physics)와 4/4(trust region)는
`kXeAndersonGate<<<1,1>>>`로 분리된다. candidate grid가 retire하기 전에는 그 OR/MAX를 읽을 수
없고, **kernel boundary가 그것을 말해 주는 유일한 barrier**이기 때문이다.

`kXeCommitTxn`은 `triple`과 `picard_skip`을 인자가 아니라 **device memory의 control block**에서
읽는다. accept이면 `XE_T_CAND`를 전 fuel node에 커밋하고(=`CommitXenon`의 계약), 아니면
`XE_T_F`를 zero-flux skip과 함께 커밋한다(=`UpdateEquilibriumXenon`의 `continue`).

### 3.2 거부된 step을 device가 커밋해도 되는 이유

TXN=0의 거부 경로는 “아무것도 쓰지 않은 상태에서 map을 다시 평가하고 `x + relax*(F-x)`를
커밋”한다. 그 재평가는 **아무도 쓰지 않은 상태**에서 **같은 결정적 커널**을 돌리므로 첫 번째
평가의 `F`를 그대로 재생산할 수밖에 없다. 그래서 transaction은 이미 들고 있는 `F`를 쓴다.

그리고 이 경로의 `relax`는 **항상 1.0**이다. `Driver.h`의 SolveLoop이
`xe_anderson && xe_relax == 1.0 && flux_converged`로 Anderson 시도를 무장시키고, 그 fallback이
같은 `xe_relax`를 쓴다. backend는 그 외의 relax를 **가정하지 않고 거부**한다
(`if (!(req.relax == 1.0)) return false;`). 계약 테스트가 양쪽을 모두 검사한다.

### 3.3 (b) fusion — 한 것과 하지 않은 것, 그리고 그 이유

**한 것: history maintenance 14 node → 1 커널.**
`kXeHistory`가 rotate(6 D2D) + record(2 kernel) + save(6 D2D)를 대신한다.

> **ORDER-PRESERVATION NOTE.** 세 연산 모두 하나의 fuel ordinal, 하나의 row에서
> **elementwise**다. reduction 없음, 이웃 참조 없음, 누산 없음 — ordinal *k*의 출력은 ordinal
> *k*의 입력만의 함수이므로 어떤 thread도 다른 thread의 write를 관측할 수 없고 grid는 임의
> 순서로 retire해도 된다. 보존해야 하는 것은 **하나의 ordinal 안에서의 순서**뿐이다: record는
> save가 덮기 전에 `F_prev`를 읽어야 하고, rotate는 record가 덮기 전에 `df[1]`을 읽어야 한다.
> `xeHistoryOrdinal`은 네 operand를 먼저 register로 읽고 나서 store한다. 산술은 추가되지도
> 제거되지도 재결합되지도 않는다: fuse되지 않은 뺄셈 둘과 복사 넷 — 12개의
> `cudaMemcpyAsync`와 2개의 `kXeSub`가 하던 그것이다.

**하지 않은 것: `kXeEvaluate` + dot stage-1.** 계획이 지시한 대로 검토했고, **채택하지 않는다.**
두 개의 독립적인 이유가 있다.

1. **의존성.** stage-1이 필요로 하는 여섯 쌍 중 다섯 쌍은 `dg0`/`dg1`을 읽는데, 그것들은
   evaluate **다음에** 도는 `kXeHistory`가 쓴다. evaluate 직후에 계산 가능한 것은 `<g,g>`
   하나, 즉 **6분의 1**뿐이다.
2. **점유율.** `<g,g>` 하나만이라도 fuse하려면 partition *p*의 ordinal 구간을 한 block이
   소유해야 순서가 보존된다(고정 partition = 1024). `n_fuel ≈ 5,000`이면 partition당 약 **5
   ordinal**이고, 그러면 evaluate가 1024 block × 유효 5 thread로 돈다. **14.4 %를 차지하는 가장
   무거운 커널을 망가뜨려 launch 하나를 아끼는** 거래다.

   partition 경계를 block 경계에 맞추는 것은 답이 아니다 — 경계는 `(n_fuel, 1024)`만의 함수여야
   하고, 그것이 재현성의 근거 전부다. 남는 길은 grid-wide `grid.sync()`뿐이며, 그것은 순서가
   아니라 **launch 형태**를 바꾼다. WP7-B가 colour sweep에 대해 내린 판정과 같은 판정이다.

따라서 (b)의 실적은 **14 node → 1**이고, evaluate/dot fusion은 **거부 사유와 함께 기록**한다.

### 3.4 새로 생긴 네 개의 contraction site — 이것이 가장 큰 함정이었다

device TU는 `--fmad=false`로 빌드되고 host `Driver.h`는 g++ `-O3`(`-ffp-contract=fast`)로
빌드된다. `Driver.h`에서 device로 옮긴 네 식은 각각 **add 하나에 multiply 둘**이 물린 site다:

```
det      = a * c - b * b;
gamma[0] = (c * p - b * q) / det;
gamma[1] = (a * q - b * p) / det;
proj     = gamma[0] * p + gamma[1] * q;
```

site당 3상태(둘 다 rounded / 첫 곱 fuse / 둘째 곱 fuse), 2비트. `XE_BIT_COUNT`는 5 → **13**.
`XeFormMine.h`의 descent site table에 넷 모두 들어갔고, `src/XeAlgebraReference.cpp`에
`refAlgebra`가 **위 네 식을 문자 그대로** 인용한다(8919331에서 `XeAndersonReference.cpp`로부터
분리되었다 — 그 TU는 비트 0..4가 채점되는 기준이라 WP7-C가 건드릴 수 없다). 채점 fixture는 별도다: 64개의 **잘 조건화된**
Gram case(`|b| < 0.86·sqrt(ac)`)여야 2열 분기가 실제로 실행되고, 그렇지 않으면 세 site가
don't-care로 채굴된다 — fixture에 대한 진술이지 compiler에 대한 진술이 아닌 값이 나온다.

site가 아닌 것도 명시한다: `XE_ANDERSON_MIN_GRAM * a * c`와 `max_step * picard`는 add 없는
multiply chain, `p / a`는 나눗셈, `gg - proj`와 1열의 `gamma[j] * p`는 multiply가 하나 이하다.

`XE_FORMS_DEFAULT`의 비트 5..12는 **0으로 출하되며 그것은 측정이 아니라 측정의 부재다** —
저작 호스트에 nvcc가 없다. 238 첫 빌드의 `[RASBERY][FORMS]`가 채굴값과 이 기록의 불일치를
선언할 것으로 **예상된다**. 그 줄이 뜨면 그것은 고장이 아니라 이 메커니즘이 작동한 것이다.

---

## 4. (c) receipt

```
[RASBERY][XE_GPU] {"xe_updates":…,"device_updates":…,"host_fallbacks":…,
  "fused_updates":…,"fused_device_updates":…,"fused_host_fallbacks":…,
  "anderson_proposed":…,"anderson_accepted":…,"anderson_accept_rate":…,"reset_edges":…,
  "xe_device_steps":…,"txn_steps":…,"txn_accepted":…,"txn_declined":…,
  "host_syncs":…,"host_syncs_per_step":…,"d2h_bytes":…,"d2h_bytes_per_step":…,
  "fuel_node_evaluations":…,"fuel_node_commits":…,"dot_partitions":…}
```

읽는 법:

| 필드 | 0이면 / 0이 아니면 |
|---|---|
| `txn_steps` | `RASBERY_GPU_XE_TXN=1`인데 **0이면 arm이 돌지 않았고 그 런으로 잰 모든 수는 무효다**(G0). `device_updates:0`과 같은 역할, 한 칸 왼쪽 |
| `txn_declined` | **0이 아니면 그 런은 두 arm의 혼합**이고 census가 census가 아니다 |
| `host_syncs_per_step` | §2 표의 4(또는 3~5) 대 1을 **그 런이 스스로 인쇄한 값** |
| `d2h_bytes_per_step` | 두 arm에서 ±12 B 안에서 같아야 한다. 크게 다르면 drain이 달라진 것이고 그것은 버그다 |
| `txn_accepted / txn_steps` | `anderson_accept_rate`와 같은 값이어야 한다 (다른 두 경로로 센 같은 사건) |

`host_syncs`와 `d2h_bytes`는 **양쪽 arm에서** 센다(`Impl::xeSync()`, `Impl::countXeD2H()`).
`xe_device_steps`는 커밋된 step마다 정확히 한 번, 양쪽 arm에서 오르는 유일한 사건이므로
per-step 비율의 분모가 된다.

---

## 5. 무엇이 달라져도 되고 무엇이 달라지면 안 되는가

| 항목 | TXN=0 vs TXN=1 | 근거 |
|---|---|---|
| HDF5 출력, digest | **완전히 동일** | 게이트. §6 |
| `xe_updates` / `device_updates` / `host_fallbacks` | 동일 | 거부 step에서 TXN=0은 `UpdateEquilibriumXenon`이 charge, TXN=1은 다운로드된 reason으로 charge |
| `anderson_proposed` / `anderson_accepted` / `reset_edges` | 동일 | 같은 사건, 같은 위치 |
| `fuel_node_evaluations` | **TXN=1에서 감소** | 거부 step의 두 번째 evaluate가 사라졌다. 실제로 없어진 작업이지 장부 조작이 아니다 |
| `xsphase` `eqxe` scope 수 | **TXN=1에서 감소**(거부 step당 1회) | 같은 이유 |
| `host_syncs`, `d2h_bytes` | **TXN=1에서 감소** | 이 work package의 목적 |
| `xsreconDebugHash` trace | TXN=1에서 미출력 | `UpdateEquilibriumXenon`을 거치지 않는다. debug 전용 |

---

## 6. 238 runbook

로컬에 nvcc가 없다. **첫 관문은 238 컴파일이다.**

### 6.0 빌드와 mining

```
cmake --build build -j
```

첫 `RASBERY_GPU_XE=1` 런에서 `[RASBERY][FORMS]` 한 줄이 `XE_FORMS`를 인쇄한다.
`sound=false`(어느 descent도 0에 닿지 못함)이면 **여기서 멈춘다** — mask를 모르는 채로
digest를 비교하는 것은 무의미하다.

### 6.1 단일, production arm, XE_TXN 0 vs 1 (B0 게이트)

```
export CUDA_VISIBLE_DEVICES=0
# A: 기준
RASBERY_GPU_XE=1 RASBERY_XE_ANDERSON=1 RASBERY_GPU_XE_TXN=0  <production arm> ...
# B: 전환
RASBERY_GPU_XE=1 RASBERY_XE_ANDERSON=1 RASBERY_GPU_XE_TXN=1  <production arm> ...
```

합격 조건 — **전부 만족해야 한다**:

1. `h5diff -c` 로 A vs B: **0 / 644 차이**
2. digest **`0d15abf29d222a02` / `4382`** 두 값 모두 A와 B가 동일
3. **ON×2 결정론**: B를 두 번 돌려 서로 bit 동일
4. `[RASBERY][XE_GPU]`: B에서 `txn_steps > 0`, `txn_declined == 0`,
   `host_syncs_per_step` A ≈ 4 → B == 1, `d2h_bytes_per_step` 두 값이 ±16 B 이내,
   `anderson_accept_rate` A == B, `reset_edges` A == B
5. `[RASBERY][GPU_FULL]`의 `mid_iteration_materializations`가 늘지 않을 것

1~4 중 하나라도 어긋나면 **성능은 재지 않는다.** B0 주장이 거짓인 arm의 wall은 의미가 없다.

### 6.2 단일 wall

warm-up 1회 버리고 **hot median of 3**, A/B 교대. 기준선은 **16.9 s**.

- 3 % 이하 개선은 노이즈로 간주하고 기본값을 바꾸지 않는다(§6.4).
- 단일은 Xe step 수가 적어 **여기서 이득이 안 나오는 것이 정상**이다. 단계 C의 근거는 §3.3의
  배치 26.9 %이고, 단일에서 회귀하면 계획이 허용한 **mode-dependent default**(batch만 ON)로 간다.

### 6.3 배치 — 여기가 값이 나오는 곳

```
8 × M8 + MPS,  CUDA_VISIBLE_DEVICES=0
```

기준선 **878 c/h**. 판정:

- 계획의 단계 C 성능 게이트: **배치 Xe phase 20 % 이상 단축 또는 전체 M64 5 % 이상**
- 함께 읽을 것: `width_fill`, `padding_fraction`, `tail_idle_s`, CPU wait, GPU SM,
  `[RASBERY][CUDA][CAPTURE_ARBITER]`의 `alloc_in_capture == 0`,
  `captures_unwound == 0` (transaction은 새 capture를 열지 않으므로 두 값 모두 변하면 안 된다)
- per-deck 동등성: 배치 각 deck의 결과가 같은 deck의 단일 실행과 동일할 것

### 6.4 nsys

```
nsys profile --trace=cuda -o wp7c_txn1 <B 명령>
```

확인할 것:

1. top kernel 목록에서 `kXeSub`가 사라지고 `kXeHistory`가 그 자리보다 **작게** 나타날 것
2. `kXeAndersonSolve` / `kXeAndersonGate`의 duration 합이 launch overhead 수준(각 ~2 µs)일 것 —
   그보다 크면 1-thread 커널이 아니라 뭔가 다른 것을 돌리고 있는 것이다
3. Memcpy DtoD 건수가 step당 12 → 0
4. `cudaStreamSynchronize` 건수가 Xe step 수의 4배 → 1배

---

## 7. 채택 조건

| 조건 | 판정 |
|---|---|
| §6.1 전부 통과 | **필수.** 하나라도 실패하면 채택하지 않고 원인을 찾는다 |
| 배치 Xe phase −20 % 또는 M64 −5 % | 채택 |
| 배치 이득 3 % 이하 | 기본값 유지, 코드는 남긴다(회귀 아님) |
| 단일 회귀 + 배치 이득 | mode-dependent default (batch ON, single OFF) — 계획 단계 C가 명시적으로 허용 |

---

## 8. 남은 것

- **`_xs`/`_iden` residency.** step당 1.96 MB D2H와 마지막 남은 sync 하나는 이 arm이 아니라
  residency가 지운다. 그것이 끝나면 receipt 다운로드는 statepoint 끝 또는 K step마다 한 번으로
  내려갈 수 있고, `xeTransaction`의 반환 경로는 그 형태를 이미 받아들일 수 있게 되어 있다
  (control block은 한 덩어리 POD다).
- **graph capture.** 계획 단계 C는 “full transaction을 graph로 캡처한다”고 적었다. 지금은
  캡처하지 않는다 — 8 launch를 graph로 묶는 이득은 launch 8회분이고, capture window는
  `GpuCaptureArbiter`의 배타 자원이다. 배치에서 M개 deck이 각자 Xe graph를 캡처하려 들면
  arbiter 대기가 이득보다 클 수 있다. **§6.3의 수치를 먼저 보고 결정한다.**
- **`XE_FORMS` 비트 5..12의 출하 기본값.** 238의 채굴값이 나오면 `XE_FORMS_DEFAULT`를 그
  호스트의 기록으로 갱신할지 결정한다(CmfdOuter의 0x6/0x7 선례대로, 기록은 갱신하되 런은
  여전히 채굴값을 쓴다).

---

## 9. 호스트 181의 반증 — B0 주장의 강등 (2026-08-30)

`E:\rasbery_runs\20260830\181\gates_8919331.md` 블록 (4). 덱 `kngr_238.json`, arm-X,
`RASBERY_GPU_XE_TXN` 0 대 1, 같은 바이너리(`8919331`), 같은 호스트.

| 항목 | TXN=0 | TXN=1 | 판정 |
|---|---|---|---|
| digest | `1d897e3f77204799` | `7f32414a742623b9` | **다름** |
| `h5diff` | — | 854줄 | **B0 실패** |
| `xe_updates` = `device_updates` = `xe_device_steps` | 1195 | 1190 | 5 step 적음 |
| `anderson_proposed` | 752 | 747 | 5 적음 |
| `anderson_accepted` | 737 | 733 | 4 적음 |
| `host_syncs_per_step` | 3.639 | **1.0** | 메커니즘은 §2대로 동작 |
| `d2h_bytes_per_step` | 1.96069e6 | 1.96072e6 | 허용 범위 |
| `txn_steps` / `txn_declined` | — | 1190 / **0** | arm 혼합 없음 |
| TXN=1 ×2 | — | 동일 digest | 결정성 있음 |

**sync 감축 메커니즘 자체는 문서대로 작동했다.** 4→1, drain 불변, arm 혼합 0.
무너진 것은 B0 주장 하나다.

### 9.1 왜 이것이 "mask가 unsound했다"가 아닌가

양쪽 arm 모두 **같은** `[RASBERY][FORMS]` 줄을 인쇄했다 — `0xd3d` 채굴값, `[WARN][FORMS]`
없음. 그 두 사실이 각각 다음을 뜻한다:

- `[WARN][FORMS]`가 없다 = `XeFormMine.h::mineStable`의 **algebra 채널이 네 seed 전부에서
  mismatch 0에 도달했다**(`algebra_sound == true`). 즉 채굴은 실패하지 않았다.
- `0xd3d`의 비트 5..12는 `0b01101001` — DET=`P1`, G0=`P2`, G1=`P2`, PROJ=`P1`. 네 site
  **모두 seed 0(전부 0)에서 움직였다**. `descend`는 점수가 *엄격히* 줄어야만 비트를 바꾸므로,
  네 site 모두 이 fixture가 판별한다. don't-care가 아니다.

그러므로 mask는 자기가 채점된 대상 — `xeref::refAlgebra` — 을 **정확히** 재현한다.
재현하지 못하는 것은 `Driver.h::TryAndersonXeStepGpu`에 **인라인된** 그 네 식이다.

### 9.2 채굴이 닫을 수 없는 틈

`refAlgebra`는 `src/XeAlgebraReference.cpp`의 **별도 함수, 별도 translation unit**이다.
피연산자는 `f.alg[]` 배열에서 오고, 결과는 출력 포인터로 나간다. production은 `SolveLoop`에
인라인된 블록이고, 피연산자는 `xs.XeGpuDots()`가 채운 로컬 `double dots[6]`에서 오며,
`gamma`/`proj`는 그 뒤 safeguard와 device 커널 인자로 흘러간다. **어느 곱을 add에 접어넣을지는
gcc의 per-call-site 결정**이고, register pressure와 사용처가 다르면 달라질 수 있다.

이 틈은 fixture를 더 좋게 만들어서는 닫히지 않는다. production call site는 **production 인라인
안에서만** 도달 가능하기 때문이다. 비트 0..4가 같은 위험을 지고도 지금까지 버틴 이유는 별개다:
그 비트들이 지배하는 커널(`kXeDotStage1`, `kXeCandidate`)은 **TXN=0과 TXN=1 양쪽에서 똑같이**
돈다. 오채굴이 있어도 A/B에서 상쇄된다. **비트 5..12만이 이 A/B가 실제로 시험하는 비트**이고,
그래서 이 A/B가 캠페인 최초로 채굴 mask의 production 충실도를 시험한 자리다 — 그리고 실패했다.

### 9.3 무엇이 원인이 아닌지 (배제한 것들)

소스 대조로 배제했다. `xeAndersonSolveControl`/`xeAndersonGateControl`은 arming(§`ncol==0`,
`picard < eq_tol`), SAFEGUARD 1~4의 순서·상수·`!(<=)` NaN 규약까지 `TryAndersonXeStepGpu`와
같다. 거부 step의 commit도 같다 — TXN=0은 `UpdateEquilibriumXenon` → `XeGpuEvaluate`(같은
상태, 같은 커널, 같은 F) → `XeGpuCommitPicard`(`XE_T_F`, `relax`, `picard_skip=true`)이고,
TXN=1은 `kXeCommitTxn`의 `accept==0` 가지(`xeBlendOrdinal` + `processed[k]==0` skip)로 같은
식·같은 skip이다(`relax != 1.0`은 `xeTransaction`이 거절한다). window 부기(`ncol_after`,
`hist_col`, `hist_rotate`)도 같고, dot layout은 `uploadXeTxnLayouts()`가 `xeDots()`와 같은
`add` 순서로 만든다. `txn_declined = 0`이므로 혼합도 아니다.

남는 유일한 산술적 차이가 det / γ₀ / γ₁ / proj — 즉 비트 5..12다.

### 9.4 판정

**WP7-C는 현 device arm 대비 N1이다.** B0는 취소한다.

이유를 한 줄로: **TXN=1은 CPU FP64(gcc가 고른 contraction)에서 device(`--fmad=false`,
명시적 form)로 대수를 옮기고, 그 이동이 비트 보존이려면 13비트 mask가 production call site의
gcc를 재현해야 하는데, mask는 그 call site가 아니라 그것의 인용을 상대로 교정된다.**

강등은 문서만이 아니라 receipt에도 적혀 있다 — `[RASBERY][XE_GPU]`의 `policy_note`
(`src/XeGpuReceipt.h::kXeTxnPolicyNote`). 게이트 스크립트가 읽는 값은 그쪽이다.

### 9.5 이것을 잡았어야 할 계측 — `RASBERY_XE_FORMS_AUDIT`

`src/XeFormAudit.{h,cpp}`. **채굴이 만들 수 없는 측정을, 문제의 call site에서 만든다.**
`RASBERY_GPU_XE=1`, `RASBERY_GPU_XE_TXN=0`으로 돌리면서
`RASBERY_XE_FORMS_AUDIT=1`을 주면, `TryAndersonXeStepGpu`가 자기 γ/proj를 계산한 직후
같은 `dots`·같은 `XE_ANDERSON_MIN_GRAM`·**해결된 그 mask**로 shipped body
(`xe::xeAndersonFit`)를 다시 돌려 비트 단위로 비교한다.

- `forms_audit_mismatch == 0` (그리고 `forms_audits > 0`) → 그 빌드에서 mask는 production을
  재현한다. B0 주장이 성립할 수 있는 **유일한** 형태다.
- `forms_audit_mismatch > 0` → N1이고, 숫자가 붙는다. 첫 불일치는 slot 이름과 함께 한 줄
  `[RASBERY][WARN][FORMS]`로 나온다.

**한 번의 런, h5diff 없음, bisect 없음.** 181은 이 값을 낼 방법이 없었기 때문에 854줄의
h5diff에서 "trajectory divergence"까지만 갈 수 있었다.

계측은 **자기만의 translation unit**에 있고 그것이 계측의 전부다. 헤더 인라인이었다면 gcc가
production의 `a * c - b * b`를 audit의 재계산으로 CSE해서 **값을 자기 자신과 비교**하게
만들 수 있고, 그러면 모든 호스트에서 영원히 mismatch 0이 나온다. 거짓 주장을 잡는 것이 유일한
일인 계측의 거짓 음성은 계측이 없는 것보다 나쁘다. `src/XeAlgebraReference.cpp`가 존재하는
이유와 같은 논증을, 기준이 아니라 측정에 적용한 것이다.

계약: `tools/test_xe_forms_audit_contract.py` (순수 python, negative control 포함).

### 9.6 채택된 수정 — device mask를 host call site로부터 구성한다

**결론부터: 위 3안 중 2번을 "손으로 pin"이 아니라 "구성(composition)"으로 상설화했다.**

#### 9.6.1 181에서 측정된 것 (73f8627)

| 항목 | 값 |
| --- | --- |
| 채굴된 device mask (`xeShippedFormMask` 소스의 union) | `0xd3d` |
| production host call site (`XE_HOST_FORMS_DEFAULT`) | `0xac0` — det=2, g0=1, g1=1, proj=1 |
| `RASBERY_GPU_XE_TXN` 0 vs 1, pin 없음 | **불일치** — Xe step 1190 vs 1195 |
| `RASBERY_XE_FORMS=0xadd` pin 후 0 vs 1 | **byte-identical** — h5diff 0, digest `88dc35e408c86ad4`, 양쪽 Xe step 1199 |
| 같은 조건의 `RASBERY_XE_FORMS_AUDIT=1` | `forms_audit_mismatch 0`, `forms_audit_mask`=채굴값, `forms_audit_host_mask`=`0xac0` |

`0xadd`는 발견된 수가 아니다. **`(0xd3d & 0x1f) | 0xac0`**, 즉 채굴된 비트 0..4와 host
mask의 비트 5..12를 이어붙인 것이다. 사람이 손으로 한 계산을 바이너리는 두 조각 다 이미
들고 있었다.

#### 9.6.2 무엇을 바꿨나

`src/XeFormMiner.cpp::xeFormMask()`가 이제 **채널별로 출처가 다른 두 조각을 합성**한다.

```
resolved = resolveCalibratedFormMask(...)        // 종전과 동일: 채굴 + fallback + env
host     = xeHostFormMask()                      // XE_HOST_FORMS_DEFAULT + env, 비트 5..12
composed = (resolved & XE_SHIPPED_FORMS) | (host & XE_ALGEBRA_FORMS)
value    = env_pinned ? resolved : composed      // source = "env" / "build_default_composed"
```

- **비트 0..4 (`XE_SHIPPED_FORMS`)** — 고정분할 dot과 candidate loop. 트리 어디에도 host
  대응물이 없는 **순수 device site**이고 fixture가 정직하게 도달한다. 채굴값을 그대로 쓴다.
- **비트 5..12 (`XE_ALGEBRA_FORMS`)** — 정규방정식 네 식. TXN=1은 TXN=0을 재현해야 하고,
  TXN=0에서 그 네 식을 계산하는 것은 `Driver.h::TryAndersonXeStepGpu`의 host 블록
  (`xe::xeSiteSub`/`xeSiteAdd`, `xeHostFormMask()`)이다. **그러므로 host의 철자가 곧 사양이고,
  fixture가 결정할 것이 남아 있지 않다.** 채굴이 그 자리에서 답하는 것은 §9.2가 이미 적은
  대로 *다른 translation unit의 인용문*(`xeref::refAlgebra`)을 gcc가 어떻게 접느냐이다.
  **틀린 대상을 잰 측정은 측정이라는 이유로 옳아지지 않는다.**

즉 device mask는 이제 **구성상(by construction)** host call site와 일치한다. `0xadd`를
손으로 넣을 필요가 없어졌고, §9.6.5의 pin 절차는 "B0를 되찾는 길"에서 "이 구성에 반대하는
절차"로 지위가 바뀌었다.

#### 9.6.3 env override는 여전히 축자적으로 이긴다

`RASBERY_XE_FORMS`를 친 사람은 **자기가 친 수를**, algebra 비트까지 포함해서, 의미한 것이다.
81 지점 스윕(§9.6.5의 2번)은 정확히 이 구성에 반대하는 절차이고 그것은 계속 가능해야 한다.
`resolveCalibratedFormMask`가 이미 override를 적용해 두므로 여기서 필요한 것은 *적용됐는지*
뿐이다 — `source`가 `env`면 합성을 건너뛴다.

#### 9.6.4 receipt

한 줄에 답과 두 재료가 같이 있어서, 리뷰어가 두 번째 런이나 두 번째 파일 없이
`0xd3d` ⊕ `0xac0` → `0xadd`를 검산할 수 있다.

```
[RASBERY][FORMS] {"mask":"XE_FORMS","resolved":"0xadd","source":"build_default_composed",
                  "mined":"0xd3d","host":"0xac0","composed":"0xadd","shipped":"0x1d",
                  "algebra":"0xac0","live_arm":"shipped","txn_arm":"resolved",
                  "algebra_sound":1}
```

`source`는 `build_default_composed`(합성) 또는 `env`(축자 override) 둘 중 하나다.

#### 9.6.5 TXN=0 궤적은 움직이지 않는다

합성이 건드리는 것은 비트 5..12뿐이고, 그 비트의 유일한 소비자는
`XsReconBackend::xeTransaction` 안의 `kXeAndersonSolve`이며 그것은 `Driver.h`가
`rasberyGpuXeTxnEnabled()` 뒤에서만 부른다. production split arm의 두 launch
(`xeDots`/`xeCandidate`)는 `xeShippedFormMask()` = `xeFormMask() & XE_SHIPPED_FORMS`,
즉 비트 0..4만 받고 `XE_SHIPPED_FORMS ∩ XE_ALGEBRA_FORMS = ∅`이다. **`RASBERY_GPU_XE_TXN`
unset/0 런은 이 커밋 이전의 그 런과 비트 단위로 같고, 차이는 위 receipt 텍스트가 전부다.**

계약: `tools/test_xe_forms_host_consistency_contract.py` (40 검사 / 음성대조 20).

```bash
python tools/test_xe_forms_host_consistency_contract.py
python tools/test_xe_forms_audit_contract.py
python tools/test_xe_forms_default_contract.py
python tools/test_xe_forms_shipped_split_contract.py
python tools/test_xe_host_forms_contract.py
```

**게이트: 181에서 env 없이 TXN 0 vs 1 B0.** 아직 미실행 — §9.6.6.

#### 9.6.6 181 runbook — env 없는 B0 증명

```bash
# 이 커밋에서 빌드한 뒤, arm 하나당 한 번씩. RASBERY_XE_FORMS는 어디에도 없다.
RASBERY_GPU_XE=1 RASBERY_GPU_XE_TXN=0 ./rasbery <deck>   # -> out_txn0.h5
RASBERY_GPU_XE=1 RASBERY_GPU_XE_TXN=1 ./rasbery <deck>   # -> out_txn1.h5
h5diff -c out_txn0.h5 out_txn1.h5                        # 기대: 0 차이
```

두 런의 `[RASBERY][FORMS] {"mask":"XE_FORMS"...}` 줄이 `"source":"build_default_composed"`,
`"resolved":"0xadd"`로 같아야 하고, `RASBERY_XE_FORMS_AUDIT=1`을 얹은 확인 런에서
`forms_audit_mismatch == 0`이어야 한다.

#### 9.6.7 남은 선택지 (채택되지 않음)

1. **`RASBERY_XE_FORMS_AUDIT=1`을 181/238에서 한 번 돌린다.** §9.4는 소거법의 결론이다.
   audit은 그것을 직접 측정으로 바꾼다. `forms_audit_mismatch`가 0이면 §9.4는 틀렸고 남은
   차이는 다른 곳에 있다 — 그때는 `RASBERY_GPU_XE_DOT_PARTITIONS=1`로 좁힌다.
2. **`RASBERY_XE_FORMS`로 site를 직접 쓸어본다.** 네 site × 3상태 = 81 조합. audit이 0을
   내는 조합이 있으면 그것이 이 호스트의 production mask이고, 그 값을 arm-X에
   `RASBERY_XE_FORMS`로 고정하면 TXN A/B가 B0로 돌아온다 — mask는 env override가 최우선이다
   (`resolveCalibratedFormMask`). **181에서 실제로 그 조합은 `0xadd`였고, 그것이 손 pin이
   아니라 §9.6.2의 합성으로 상설화됐다.** 스윕 자체는 합성에 반대하기 위한 절차로 남는다.
3. **또는 양쪽 arm이 같은 body를 쓰게 한다.** `TryAndersonXeStepGpu`가 자기 손으로 쓴 네 식
   대신 `xe::xeAndersonFit`을 호출하면, host g++ 빌드(`xsrMul`의 asm barrier)와 nvcc
   `--fmad=false` 빌드가 **어떤 mask 값에서도** 같은 비트를 낸다 — TXN A/B는 구성상 B0가 된다.
   대신 그것은 **`RASBERY_GPU_XE` arm 자신의 수치를 오늘로부터 움직이는 N1 변경**이고,
   8919331이 명문화한 규칙("production arm이 도는 mask는 새 site의 통로가 아니다")에 따라
   그 자체로 Gate A/B를 받아야 한다. 2번이 실패했을 때의 길이다.
