# WP7 단계 C — Xe device transaction

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `RASBERY_GPU_XE` split Xe arm의 safeguarded Anderson step |
| 상위 계획 | `docs/GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` WP7 단계 C, §3.3, §6 |
| 게이트 등급 | **B0 (현 device arm 대비)** / N1 (host arm 대비, 이미 건너간 선) |
| 플래그 | `RASBERY_GPU_XE_TXN=1`, 기본 `0` |
| receipt | `[RASBERY][XE_GPU]` — `txn_steps`, `txn_accepted`, `txn_declined`, `host_syncs`, `host_syncs_per_step`, `d2h_bytes`, `d2h_bytes_per_step`, `xe_device_steps` |
| 계약 테스트 | `tools/test_xe_txn_contract.py` (순수 python, negative control 8종) |
| 소스 | `src/XeKernel.h`, `src/CudaXsReconBackend.{h,cu}`, `src/Driver.h`, `src/XeGpuReceipt.h`, `src/XeAndersonReference.{h,cpp}`, `src/XeFormMine.h`, `src/XSSet.{h,cpp}` |
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
`XeFormMine.h`의 descent site table에 넷 모두 들어갔고, `XeAndersonReference.cpp`에
`refAlgebra`가 **위 네 식을 문자 그대로** 인용한다. 채점 fixture는 별도다: 64개의 **잘 조건화된**
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
