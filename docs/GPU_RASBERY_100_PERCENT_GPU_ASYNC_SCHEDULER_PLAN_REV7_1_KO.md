
# GPU RASBERY 100% Numerical GPU + Asynchronous Case-Phase Scheduler Implementation Plan (Rev.7.1)

> **이 문서의 성격**
> Rev.7(`GPU_RASBERY_100_PERCENT_GPU_ASYNC_SCHEDULER_PORTABILITY_PLAN_REV7_20260828_KO.md`, 3,993줄, 29태스크)의 **수정본**이다.
> Rev.7의 절 번호·태스크 번호(0–28)·체크박스(`- [ ]`) 형식을 그대로 승계한다.
>
> - **[Rev.7.1 수정]** 태그가 붙은 절만 본문을 다시 쓴다. 그 절의 Rev.7 본문은 **폐기**된다.
> - 태그가 없는 절은 `Rev.7 §x.y 유지`로만 표기한다. 원문을 그대로 읽는다(중복 전재하지 않는다).
> - 신설 항목은 `12a/12b`, `13a/13b`, `19b`처럼 **접미 번호**로 붙여 Rev.7 번호 체계를 깨지 않는다.
>
> **근거**
> - 승인된 최종 계획(Rev.7 델타 판정 + Rev.6 심층검토 31건 승계 + 정직 목표표 + wave 프로그램).
> - 실코드 대조: 브랜치 `codex/exact-throughput-campaign`, tip `fdef162`. 본문에 인용한 모든 `file:line`은 이 tip에서 재확인했다.
> - 실측 기준선: `docs/CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md`, `docs/PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md`, `docs/IO_WRITER_THREAD_DESIGN_20260827_KO.md`.

---

## 개정 요약 (Rev.7 → Rev.7.1)

| # | 대상 절/태스크 | Rev.7 | Rev.7.1 | 근거 |
|---|---|---|---|---|
| 1 | §1.4, §5.x | MC영감 2-level 스케줄러 제안 | **채택 확정** (배치 병목의 구조적 해답) | 캠페인 실측: 도착 폭 기아·tail·rendezvous |
| 2 | §5.1 CMFD quantum | sweep 1회 고정 | sweep 1회 + **잔여 예산 전체 폴백 조항** | 단일덱 ~68k 에폭 × 제어 5~10µs |
| 3 | §3.2 | `DeviceSlotControl` 단일 128 B | **4분할** Phase/State/Search/ScheduleParams | 실 상태 ~60필드 누락 (`Scheduler.h:55-61,127-177`) |
| 4 | §3.6 | 90–160 MiB → 22–35 MiB(압축) | **~200–225 MiB/slot, 압축 임계경로 제외** | `_ref_micx`+AA history 누락, 64슬롯=14.4 GiB=VRAM 15% |
| 5 | §3.6/Task 0 | 메모리가 위험 | **진짜 위험은 L2**(핫셋 60 MB ⊂ 128 MB) | 폭별 L2/DRAM 실측 게이트 신설 |
| 6 | §6.14 / Task 12 | 2-커널 분할 GPU resolve | **12a(host resolve+device apply) / 12b(device resolve, 후행·킬기준)** | `XSSet.cpp:2308` 부분적용 workspace 순서 의존 |
| 7 | §6.15 / Task 13 | Xe·Anderson을 warp-per-slot | **N1 재분류 + `k_xe_dot_reduce` + 기각=감쇠 Picard + history reset 엣지** | XeDot 15,000항 직렬 누적×6 |
| 8 | §6.17 / Task 15 | warp 32레인 trial-history scan | **슬롯당 1스레드** (history 배열 부재) | `Scheduler.h:155-177` = 유한 스칼라 26필드 |
| 9 | §6.18 / Task 16 | dense LU 39×39 CTA | **노드당 1스레드 희소 Gauss-Seidel** (order 8/4pole/64iter) | `milk.h:1649-1819`, `XSSet.cpp:4008` |
| 10 | §6.18 대상 | fuel node | **전 nxyz 8,451노드** | `XSSet.cpp:4036` |
| 11 | — | PPR 0회 등장 | **Task 19b 신설** (GPU PPR, 단일 wall 6~10%) | `PPR.cpp` 1,099줄, `Driver.h:2278-2295` |
| 12 | §9.1 | TH보간/search산법/output pack 분류 오류 | **B0/N1 재분류** + nodal exp/sqrt B0 구제 스파이크 | 실측 AO 재배열 1.2e-16 |
| 13 | §9.2/§14 | `gpu_numeric_fraction=1.0` 단일 목표 | **M1(세그먼트)/M2(상태점) 2단 마일스톤** + phase 5종 추가 | `Driver.h` 호출부 10곳 명시 |
| 14 | §9.3 | receipt 기본형 | **7종 추가**(xe_aa_*, ppr_host_calls, flatxs_host_resolve_calls, tail_efficiency, l2_hit_rate, dram_throughput_pct, host_numeric_allowlist[]) | 회귀 감시 공백 |
| 15 | Task 6 | unroll 키 제거 "가능" | **제거 완결 필수** | `CudaBICGBackend.cu:2744` 잔존 |
| 16 | §11 | W0→W10 순수성 우선 | **W0→W1→W2→W3→W3.5→W3.6(A2)→W3.7→W4-lite→W5+** 속도 우선 | 잔여 75인일 후행 재판정 |
| 17 | 제약 17 | persistent = 비필수 | **W3.7로 승격**(W0 게이트 통과 조건부) | 속도 목표의 2대 레버 중 하나 |
| 18 | 제약 35 | 없음 | **공용 pure body에 CUDA 전용 intrinsic 금지** 즉시 적용 | 후일 이식 비용 최소화 |
| 19 | Task 24/25/27/28 | W7–W9 정규 트랙 | **후행 트랙 분리**(하드웨어 확보/별도 캠페인) | 속도 목표 기여 0 |

---

## Global Constraints

**[Rev.7.1 수정]** — 제약 17 완화(persistent를 W3.7 정규 트랙으로 승격), 제약 35 신설(이식성 코딩 규율을 지금부터 적용). 나머지 1–16, 18–34는 **Rev.7 유지**.

**제약 17 (개정).** ~~Persistent kernel은 필수 경로가 아니다.~~
→ **Persistent CMFD는 W3.7의 정규 트랙이다.** 단 착수 조건은 **W0 스파이크 ②(`c_barrier(sm_120, 34~67 blocks)` 실측)** 통과이며, 채택 게이트는 **end-to-end median >1% 개선 + bit identity + blocksPerSM ≥ 2 + 타 스트림 대기 ≤5% 회귀**다.
**킬 기준: `c_barrier > 0.384 µs`이면 persistent 트랙을 영구 종결하고 W3.7을 건너뛴다.** (측정 절차는 `docs/PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md` §2.6 Stage 0-E를 재사용한다.)

**제약 35 (신설). 공용 pure body에 CUDA 전용 intrinsic 금지.**
`*Kernel.h`류 host/device 공용 순수 body와 `GpuPhaseSchedulerCore.h`에는 `__shfl_*`, `__ballot_sync`, `__activemask`, `atomicAdd_block`, `__fmaf_rn`, `__ldg`, `cuda*` API를 직접 쓰지 않는다. 반드시 `GpuSubgroup.h` / `GpuBackend.h` 래퍼를 경유한다.
- 이 규율은 **Task 24/25(HIP/SYCL)를 착수하지 않아도 지금부터 적용**한다. 규율 위반은 `tools/test_pure_body_portability.py`(Task 1 Step 2에 추가)가 grep 계약으로 fail한다.
- 목적은 이식이 아니라 **후일 이식 비용의 상한을 지금 고정**하는 것이다. 이식 자체는 후행 트랙이다.

**변경 없이 유지되는 핵심 게이트(재확인):**
- 제약 3 **fail-closed**: `RASBERY_GPU_FULL=1`에서 CPU numerical fallback 1회 = job 실패.
- 제약 32 **스케줄러 오버헤드 3%**: classify/compact/dispatch/refill 총합이 solver wall의 3% 초과 시 production 채택 보류.
- 제약 6/7 **B0 bit-golden / N1 Gate A·B + v3 동결**.

---

# 1. 설계 결론

## 1.1 "100% GPU" 실행 경계

**[Rev.7.1 수정]** — CPU 잔류 목록에 **PPR(핀출력 재구성)**이 누락되어 있었다. 100% 정의는 유지하되 경계 도달을 M1/M2 2단으로 나눈다(§9.2).

### CPU에 남는 작업 (최종 M2 상태)

```text
JSON/HDF5 입력 파싱
immutable library 파일 로드
job manifest 및 namespace 검증
GPU cohort에 초기 byte 업로드
GPU error receipt 수집
GPU가 만든 output snapshot을 HDF5에 기록
로그 문자열 formatting
```

### GPU로 이동하는 수치 작업 (Rev.7 목록 + 신규 3종)

Rev.7 §1.1 목록에 다음을 **추가**한다.

```text
pin power reconstruction (PPR)   ← Task 19b 신설, Driver.h:2278-2295
flux sign normalization           ← DevicePhase::NormalizeFluxSign
derivative 재수렴 재시도 판정      ← DevicePhase::Derivative
rod op (SetRod/ResetFlux/resetDhat) ← DevicePhase::RodOp
result aggregate (AddResult)      ← DevicePhase::ResultAggregate
```

### 실행 형태

`Rev.7 §1.1 실행 형태 유지` — 단 phase 목록에 위 5종이 추가된다(§3.1 참조).

## 1.2 최신 연구에서 채택할 원칙

`Rev.7 §1.2 유지.`

## 1.3 현재 코드에서 재사용하는 자산

**[Rev.7.1 수정]** — 목록 2건 추가.

Rev.7 §1.3 목록에 다음을 추가한다.

- `include/milk.h:1649-1819` — `solveBatemanCRAM`의 **희소 Gauss-Seidel pole solve**. GPU CRAM(§6.18)은 이 산법의 이식이지 dense LU의 신규 구현이 아니다.
- `src/PPR.cpp` — corner-balance 반복(`drive`), `updateFused/updateAxialLeakage/updateSource/updateCorner`, 3×3 Gauss-Legendre 핀면적 적분(`reconstructPinPower`). Task 19b의 pure body 원본.

## 1.4 Monte Carlo GPU scheduler에서 채택하는 개념

**[Rev.7.1 수정]** — 판정 문구 확정. 설계는 채택하되 채택 근거를 캠페인 실측에 고정한다.

**판정: 채택.** Rev.7의 2-level case-phase 스케줄러는 캠페인이 반복 측정해 온 배치 병목 — **도착 폭 기아(폭 12~22), cohort tail, rendezvous 대기** — 의 올바른 구조적 해답이다. 특히 다음 4요소를 그대로 채택한다.

1. **Level-1 = `DeviceSlotPhase`만 읽는 1-CTA classify/compact**, Level-2 = phase 전체를 quantum으로 실행.
2. **SWITCH(선택된 단일 phase) 구조**(§5.5) — Rev.6의 7-IF 사다리(에폭당 ~32 µs 제어비용)를 해소.
3. **`queued_epoch`/`state_epoch`/`in_flight` 소유권 규칙**(§5.2)과 refill 시 full slot-control reset(§8.2) — 재활용 슬롯의 stale tenant를 fail-closed로 차단.
4. **즉시 refill**(Done→free→새 input claim) — rolling queue의 tail을 device 측에서 해소.

**단 스케줄러의 기여 한계를 명시한다.** 스케줄러는 "폭 충전 1.35~1.7×"를 실현하는 **실행 기반**이며, 그 자체로 단일덱을 빠르게 만들지 않는다. 단일 2× 목표는 여전히 **persistent CMFD(W3.7) + outer 감축(A2, W3.6)** 두 레버가 결정한다(§13).

**Rev.7 §1.4의 변환 규칙 표·2-level 다이어그램·금지/허용 비동기성 정의는 유지.**

## 1.5 GPU vendor 지원 판정

**[Rev.7.1 수정]** — 후행 트랙으로 분리.

Rev.7 §1.5의 tier 표(G0~G4/U)와 `GpuCapabilityReceipt` 정의는 **유지**한다. 다만:

- **Task 24(HIP)·25(SYCL)는 AMD/Intel 하드웨어 확보 전 착수 금지.** 보유 하드웨어는 NVIDIA 단일(238 GPU0)이며, 이식은 속도 목표 기여가 0이다.
- **지금 적용하는 것은 제약 35(공용 pure body 규율)와 백엔드 중립 명명(`GpuPhysicsArena`/`GpuBackend`/`GpuSubgroup`)뿐이다.** 신규 파일은 이 명명 규약을 따르되 구현은 CUDA 우선이다.
- Task 23(capability probe·백엔드 계약)만 W4-lite에서 **CUDA 두 tier(conditional/epoch)** 범위로 축소 수행한다.

## 1.6 Reactor 범위 판정

**[Rev.7.1 수정]** — 후행 트랙으로 분리.

Rev.7 §1.6의 family matrix·`ReactorModelDescriptor`·"모든 원자로 대응" 완료 조건은 **유지**한다. 다만:

- **Task 26은 축소판만 W4-lite에 포함**한다: descriptor 정의 + 출력 provenance 수신증(descriptor hash / plugin name / support label). 문서·경계 정책은 저비용이며 잘못된 "지원" 표기를 막는 효과가 크다.
- **Task 27(generic topology/multigroup)·Task 28(family 검증 매트릭스)은 별도 캠페인**으로 분리한다. 본 계획의 인일 산정(§11)에 포함하지 않는다.

---

# 2. Target File Structure

**[Rev.7.1 수정]** — 신규 파일 5종 추가(구조체 4분할, FlatXS 12a/12b 분리, PPR, 이식 규율 테스트). 나머지 Rev.7 §2.1/§2.2 유지.

## 2.1 신규 핵심 파일 (Rev.7 목록 + 추가)

```text
src/GpuSlotControl.h
    DeviceSlotPhase(hot 32B) / DeviceSlotState(cold) /
    DeviceSearchState / DeviceScheduleParams 4분할 정의 + 재활용 감사 계약

src/CudaFlatXsApply.h
    Task 12a: host resolve된 delta stream을 device가 적용 (기존 FlatXsKernel.h 승계)

src/CudaFlatXsResolver.h
    Task 12b: device resolve (후행·킬기준부). 12a와 파일 분리

src/CudaPprKernels.h
    Task 19b: corner-balance 반복 + 3x3 Gauss-Legendre 핀적분 device body

tools/test_pure_body_portability.py
    제약 35 grep 계약: 공용 pure body의 CUDA 전용 intrinsic 유출 차단
```

`src/GpuPhysicsTypes.h`는 `GpuSlotControl.h`를 include하며, `DeviceSlotControl` 단일 구조체 정의는 **삭제**한다.

## 2.2 기존 파일 수정

`Rev.7 §2.2 유지` — 다음 1건 추가.

```text
src/PPR.{h,cpp}
    pure body 추출(updateFused/updateSource/updateCorner/reconstructPinPower),
    device view 지원, CPU reference 유지
```

---

# 3. Device Data Model

## 3.1 Phase와 exit 상태

**[Rev.7.1 수정]** — host 수치 경계 잔여 지점을 phase로 승격(5종), CRAM 탈출코드 2종 추가.

```cpp
enum class DevicePhase : std::uint32_t {
    Empty = 0,
    Import,
    Material,
    Outer,
    Xenon,
    ThermalHydraulics,
    Search,
    // --- Rev.7.1 신설: Driver.h 상태점 루프의 host 수치 잔여 지점 ---
    NormalizeFluxSign,   // Driver.h:2235, 2239, 2261, 2274
    Derivative,          // Driver.h:2245 (UpdateDerivative) + 2268-2274 (재수렴 재시도)
    RodOp,               // Driver.h:2252-2255 (SetRod/ResetFluxAndCurrents/resetDhat/eigv=1)
    Ppr,                 // Driver.h:2278-2295 (reset/drive/reconstructPinPower)
    ResultAggregate,     // Driver.h:2308 (AddResult)
    // --------------------------------------------------------------
    DepletionPredictor,
    DepletionCorrector,
    OutputPack,
    Done,
    Failed
};

enum class DeviceEscape : std::uint32_t {
    None = 0,
    FluxConverged,
    FluxLimitCycleSample,
    FluxStallFatal,
    NegativeFlux,
    RayleighFallback,
    SegmentBudget,
    MaterialChanged,
    NonFinite,
    MaxIteration,
    // --- Rev.7.1 신설: CPU CRAM의 두 throw 지점을 device escape로 (§6.18) ---
    CramNotConverged,    // milk.h:1807  "CRAM Gauss-Seidel did not converge"
    CramZeroDiagonal     // milk.h:1765  "CRAM Gauss-Seidel zero diagonal"
};
```

**계약:** device는 예외를 던질 수 없으므로 CPU가 `throw`하는 두 지점은 반드시 escape 코드 + 슬롯 실패로 사상되어야 한다. Task 16 Step 4의 guard는 이 두 코드를 포함해야 통과한다.

## 3.2 Per-slot control packet

**[Rev.7.1 수정 — 대폭]** — Rev.7의 단일 `DeviceSlotControl`(128 B)은 실제 케이스 상태의 **약 60필드를 누락**한다(search 전 상태, `SearchMemory`, 슬롯별 스케줄 파라미터, Xe damper/substep 상태). 또한 128 B 패킷을 Level-1이 통째로 읽으면 classify 트래픽이 4배가 된다. **4분할한다.**

### (A) `DeviceSlotPhase` — hot 32 B, Level-1이 읽는 유일한 구조체

```cpp
struct alignas(32) DeviceSlotPhase {
    std::uint8_t  phase;         // DevicePhase
    std::uint8_t  queued_phase;  // DevicePhase
    std::uint8_t  escape;        // DeviceEscape
    std::uint8_t  flags;         // bit0 active | bit1 in_flight | bit2 fatal | bit3 input_ready

    std::uint32_t state_epoch;   // phase 전이 및 tenant refill 시 증가
    std::uint32_t queued_epoch;  // 큐 삽입 시 캡처한 state_epoch
    std::uint32_t phase_age;     // fairness
    std::uint32_t input_id;
    std::uint32_t job_id;
    std::uint32_t error_code;
    std::uint32_t reserved;
};
static_assert(sizeof(DeviceSlotPhase) == 32);
static_assert(std::is_trivially_copyable_v<DeviceSlotPhase>);
```

**소유권 규칙(Rev.7 §5.2 유지):** 큐 엔트리는 캡처한 epoch이 여전히 `state_epoch`와 같을 때만 유효하다. `queued_epoch == state_epoch && queued_phase == phase` → 이미 큐에 있음(재삽입 금지). `in_flight != 0`인 슬롯을 다른 큐에 넣으면 fatal scheduler error.

**근거:** 64슬롯 × 32 B = 2 KiB → classify 1-CTA가 L1에 상주. Rev.7의 128 B는 8 KiB로 매 에폭 L1/L2를 오간다. 단일덱 ~68k 에폭(§5.1)에서 이 차이는 무시할 수 없다.

### (B) `DeviceSlotState` — cold ~1 KiB, phase 커널만 접근

```cpp
struct alignas(128) DeviceSlotState {
    // --- 스케줄/상태점 위치 ---
    std::uint32_t schedule_index;
    std::uint32_t statepoint;
    std::uint32_t substep;            // Driver.h:2227  nsub = max(1, schedule.substep)
    std::uint32_t substep_index;      // Driver.h:2229  isub  (predictor/corrector 루프 위치)
    std::uint32_t solve_call_kind;    // isub>0 pre-solve / predictor / final / derivative-retry

    // --- outer/수렴 카운터 ---
    std::uint32_t outer_in_segment;
    std::uint32_t total_outer;
    std::uint32_t flux_stall;
    std::uint32_t stall_events;
    std::uint32_t stall_sample_taken;   // limit-cycle 샘플 소비 여부
    std::uint32_t cmfd_sweeps;
    std::uint32_t bicg_iterations;
    std::uint32_t clean_iters;          // stall 없이 진행한 연속 outer 수

    // --- Xe damper / ONCE / Anderson 스칼라 상태 (Driver.h:1434-1500) ---
    std::uint32_t xe_iterations;
    std::uint32_t xe_total;             // SolveLoop 전체 정착 Xe step
    std::uint32_t xe_interim_count;     // RASBERY_XE_INTERIM_L2 loose-flux step
    std::uint32_t xe_no_progress;       // 수축 실패 연속 횟수
    std::uint32_t xe_streak;            // 진동 streak (xe_streak_limit 대비)
    std::uint32_t xe_cap_charged;       // cascade당 1회 starvation charge
    std::uint32_t xe_cascade_index;
    std::uint32_t xe_aa_ncol;           // Anderson 사용 가능 차분열 (Driver.h:1008)
    std::uint32_t xe_aa_have_prev;      //                          (Driver.h:1009)
    double        xe_relax;             // 1.0 또는 XE_DAMPED_RELAX
    double        prev_xe_change;
    double        xe_residual;

    // --- TH / search 잔차 ---
    std::uint32_t th_iterations;
    std::uint32_t search_iterations;
    double        th_residual;
    double        search_residual;

    // --- 고유값 ---
    double eigv;
    double previous_eigv;
    double eigv_before_segment;   // Driver.h:2259  eigv_before (재수렴 판정 기준)
    double flux_l2;
    double keff_tolerance;
    double flux_tolerance;

    // --- generation ---
    std::uint64_t geometry_generation;
    std::uint64_t material_generation;
    std::uint64_t operator_generation;
    std::uint64_t flux_generation;
    std::uint64_t current_generation;
    std::uint64_t dhat_generation;
    std::uint64_t nodal_constant_generation;
    std::uint64_t isotope_generation;
    std::uint64_t th_generation;
    std::uint64_t hoststate_generation;  // XSSet.cpp:4043 _hoststate_generation 대응
};
```

### (C) `DeviceSearchState` — `Scheduler.h:55-61` + `155-177` 전량 이식

Rev.7 §3.5의 `double* search_history; // bounded scalar packet` **한 줄이 실제로는 26필드**다.

```cpp
struct DeviceSearchState {
    // Scheduler.h:55-61  SearchMemory — 상태점 간 이월되는 secant 기억
    std::uint32_t has_boron_secant;
    double        boron_secant_dkdx;
    std::uint32_t has_rod_secant;
    double        rod_secant_x;
    double        rod_secant_dkdx;

    // Scheduler.h:155-171  SolveLoop 간 유지되는 runtime 검색 상태
    std::uint32_t initialized;
    std::uint32_t seeded_from_previous_step;
    std::uint32_t has_prev;
    std::uint32_t has_bracket;
    std::uint32_t has_best;
    std::uint32_t slope_frozen;
    std::uint32_t iteration;
    double        current_x;
    double        prev_x;
    double        prev_eigv;
    double        frozen_slope;
    double        best_x;
    double        best_residual;
    double        bracket_lo_x;
    double        bracket_lo_residual;
    double        bracket_hi_x;
    double        bracket_hi_residual;

    // Scheduler.h:174-177  종료 기록 (결과 파일로 발행)
    std::uint32_t exit_status;   // SearchExit{NONE,CONVERGED,BEST_FALLBACK,UNCONVERGED}
    double        exit_dk;
    double        exit_tol;
    std::uint32_t stall_count;
};
```

### (D) `DeviceScheduleParams` — 슬롯별 스케줄 파라미터 (read-mostly)

Rev.7은 슬롯마다 다른 tolerance/limit을 **공유 상수로 가정**했다. 실제로는 `Schedule` 인스턴스마다 다르며, 대량 입력에서는 슬롯마다 다른 deck이 들어온다.

```cpp
struct DeviceScheduleParams {
    // Scheduler.h:127-148  검색·수렴 파라미터
    std::uint32_t search_type;          // SearchType (KEFF/BORON/RODCRIT/...)
    std::uint32_t schedule_type;        // ScheduleType (STANDARD/DEPLETION/DERIVATIVE/ROD)
    std::uint32_t th_mode;              // THMode
    std::uint32_t max_outer_iter;       // 기본 kMaxEigenIter=200
    std::uint32_t max_search_iter;      // 기본 kMaxSearchIter=300
    std::uint32_t max_th_iter;          // 기본 kMaxThIter=10
    double tolerance_keff;              // kEigvTol=1e-6
    double tolerance_search;            // kCritSearchTol=1e-5
    double rodcrit_search_floor;
    double tolerance_th;                // kThTol=1e-6
    double target_keff;
    double tolerance_tmod, tolerance_tfuel;   // kTempSearchTol=0.01
    double search_boron_ppm, tolerance_boron; // kBoronSearchTol=0.01
    double tolerance_rodsearch;               // kRodSearchTol=0.01
    double search_relaxation;                 // kSearchRelax=1.0
    double search_low, search_hi;             // 0.0 / 1.0
    double slope_freeze_thres;                // kSlopeFreezeThres=0.01
    double min_secant_denom;                  // kMinSecantDenom=1e-12
    double bracket_min_span;                  // kBracketMinSpan=1e-6
    double search_boron_probe, search_rod_probe; // 50.0 / 0.25

    // Scheduler.h:91-124  물리 조건 / 섭동
    double time, burnup, rate, rated_power, actual_power, step_dt;
    std::uint32_t substeps, xenon_transient, use_burnup_time;
    double bppm0, tful0, tmod0, dmod0;
    double pressure, inlet_temp, outlet_temp, mass_flow_rate, fuel_temp_rise_scale;
    double delta_tful, delta_tmod, delta_dmod, delta_bppm, delta_xe, delta_sm;
};
```

### 재활용 감사와 epoch 규칙의 결합

**[Rev.7.1 필수 계약]** — Rev.7 §8.2의 "full slot-control reset"은 이제 **4개 구조체 전량**을 대상으로 한다.

```text
refill 시 반드시 0/기본값으로 재설정:
    DeviceSlotPhase       전 32 B
    DeviceSlotState       전 필드 (특히 xe_*, clean_iters, stall_*, 모든 generation)
    DeviceSearchState     전 26 필드 (SearchMemory 포함 — 이월 금지)
    DeviceScheduleParams  새 deck 값으로 완전 덮어쓰기
```

- 감사 커널 `k_audit_tenant_reset`이 refill 직후 4구조체의 non-reset 바이트를 검출하면 `stale_tenant_errors++` 및 fatal.
- **바이트 확장이 감사에 선행한다.** Rev.7 순서(감사 먼저)로는 존재하지 않는 필드를 감사할 수 없다. Task 1 → Task 20 순서를 고정한다.

## 3.3 Immutable shared geometry

`Rev.7 §3.3 유지.`

## 3.4 Immutable shared XS library

`Rev.7 §3.4 유지.`

## 3.5 Slot별 mutable bulk state

**[Rev.7.1 수정]** — 스칼라 패킷 3종을 §3.2로 이관, 누락 배열 2종 추가.

Rev.7 §3.5 `DeviceSlotView`에서 다음을 **교체**한다.

```cpp
    // 교체 전: DeviceSlotControl* control;
    DeviceSlotPhase*      phase;        // §3.2(A)
    DeviceSlotState*      state;        // §3.2(B)
    DeviceSearchState*    search;       // §3.2(C) — 구 double* search_history 대체
    DeviceScheduleParams* params;       // §3.2(D)
```

다음 배열을 **추가**한다(§3.6 메모리표에 반영).

```cpp
    // 참조 micro XS 블록 — XSSet.h:211 _ref_micx [(iso*ng+ig)*nxyz + l]
    double* ref_micx;          // device-resident, 재업로드는 라이브러리 교체 시에만
    double* ref_lmpx;

    // Xe Anderson history (Driver.h:999-1020 XeAndersonState)
    //   XeTriple = (I, Xe, Xem) × n_fuel
    //   x, f, g, f_prev, g_prev, cand = 6 triple
    //   df[DEPTH], dg[DEPTH]         = 2*DEPTH triple
    double* xe_aa_history;     // (6 + 2*XE_ANDERSON_DEPTH) * 3 * n_fuel
```

## 3.6 Per-slot memory budget

**[Rev.7.1 수정 — 대폭]** — Rev.7 표는 `_ref_micx`와 AA history를 누락했고, "압축 tier(22–35 MiB)"를 임계경로에 넣었다. **실측 기준 재산정 결과 압축은 불필요하다.**

| 범주 | Rev.7.1 실측 기반 산정 | Rev.7 표기 | 비고 |
|---|---:|---:|---|
| CMFD + Krylov | 약 4.5 MiB/slot | 4.5 | 유지 |
| Nodal state/work | 약 10 MiB/slot | 10 | 유지 |
| isotope density (39 × nxyz) | 약 2.5 MiB/slot | 2.5 | 유지 |
| live macro XS | 약 2–4 MiB/slot | 2–4 | 유지 |
| **참조 micro XS `_ref_micx`/`_ref_lmpx`** | **약 65 MiB/slot** | "제거" | **제거 불가**(§6.14 12a 경로가 참조) |
| predictor/corrector BOS 스냅샷 | **약 24 MiB/slot** | "제거" | **11→4 슬롯 축소**(bit-보존), 제거 아님 |
| **Xe Anderson history** | **약 3–5 MiB/slot** | 누락 | (6+2·DEPTH)×3×n_fuel×8 B |
| TH/search/output/control | 약 1–4 MiB | 1–4 | §3.2 4구조체 포함 |
| **합계** | **약 200–225 MiB/slot** | 90–160 → 22–35 | — |

**64슬롯 총계 ≈ 12.8–14.4 GiB.** RTX PRO 6000(96 GiB) 대비 **VRAM 약 15%**.

> **결론: 메모리는 위험이 아니다. 압축(Task 12 Step 4)을 임계경로에서 제거한다.**

**진짜 위험은 L2다.** 슬롯 폭 W에서 CMFD 스윕의 핫셋(flux/psi/diag/udiag/cc/dtil/dhat)은 대략 `W × ~1 MiB`다. **W=64에서 약 60 MB가 sm_120의 128 MB L2에 아직 들어가지만, `_ref_micx` 스트리밍이 얹히면 파괴된다.** 따라서:

- **Task 0에 폭별 L2 hit rate / DRAM throughput 측정을 신설**(§12 Task 0 Step 9).
- **W2 게이트: DRAM throughput ≤ peak의 60%.** 초과 시 폭을 낮추거나 `_ref_micx` 접근을 phase 단위로 격리한다.
- `RASBERY_PC_MODE=decart` v1 필수(predictor/corrector 스냅샷 경로 고정).

---

# 4. Memory Management

## 4.1 Allocation policy

`Rev.7 §4.1 유지.`

## 4.2 Scratch aliasing

**[Rev.7.1 수정]** — Rev.7의 aliasing 표는 **전역 phase lifetime**을 가정한다. 비동기 case-phase 스케줄러에서는 슬롯마다 phase가 다르므로 이 가정이 깨진다.

```text
Rev.7 가정:  "Outer phase 종료" 후 Nodal scratch를 depletion이 재사용
Rev.7.1 사실: 슬롯 A는 Outer, 슬롯 B는 Depletion에 동시에 있을 수 있다
              → 전역 lifetime 기준 aliasing은 슬롯 A의 Nodal scratch를 파괴한다
```

**규칙(개정): aliasing은 슬롯별 phase 키로만 허용한다.**

```cpp
// scratch 접근은 반드시 소유 phase를 명시한다
double* scratch = arena.scratch(physical_slot, DevicePhase::Outer, ScratchId::NodalTrl);
// debug 빌드: 슬롯의 현재 phase != 요청 phase 이면 trap
```

- 각 슬롯의 scratch 영역 헤더에 `owner_phase`(4 B)를 두고, phase 진입 커널이 CAS로 소유권을 잡는다.
- release 빌드에서는 태그를 쓰기만 하고 검사하지 않는다(hot path 비용 0).
- **동시에 살아 있지 않은 배열의 정의는 "같은 슬롯 안에서"로 한정**한다. 슬롯 간 aliasing은 금지다(§5.9 physical slot stride 유지).

허용 alias 쌍(슬롯 내):

```text
BiCG ax/s scratch  ↔ Wielandt terms ↔ outer convergence partials     (phase: Outer)
Nodal trlcff/matrix scratch ↔ depletion temporary                     (phase: Outer / Depletion*)
TH channel scratch ↔ output pack scratch ↔ PPR corner scratch         (phase: TH / OutputPack / Ppr)
predictor CRAM workspace ↔ corrector CRAM workspace                   (phase: Depletion*)
```

## 4.3 Host staging

`Rev.7 §4.3 유지.`

## 4.4 VRAM admission

**[Rev.7.1 수정]** — `per_slot_bytes` 기본값을 §3.6 값으로 교체.

Rev.7 §4.4 산식은 유지하되 `per_slot_bytes = 225 MiB`(상한)로 산정한다. 64슬롯 admission 실패는 여전히 명확한 실패이며 자동 축소하지 않는다. 압축 tier(22–35 MiB)를 전제한 admission 계산은 폐기한다.

---

# 5. Streams, Graphs, Case-Phase Queues, and Device Resource Allocation

## 5.1 Scheduler architecture

**[Rev.7.1 수정]** — Level-1이 읽는 구조체를 `DeviceSlotPhase`(32 B)로 고정, **CMFD quantum 폴백 조항** 신설.

Rev.7 §5.1의 Level-1/Level-2 구조, epoch 순서 7단계, phase quantum 표는 **유지**한다. 단 2건을 개정한다.

**(1) Level-1이 읽는 것은 `DeviceSlotPhase` 32 B뿐이다.** Rev.7 본문의 `DeviceCaseControl`은 §3.2(A)로 대체된다. bulk physics array는 물론 `DeviceSlotState`/`DeviceSearchState`/`DeviceScheduleParams`도 읽지 않는다.

**(2) CMFD quantum 폴백 조항 (신설).**

```text
기본:   CMFD quantum = Wielandt/CMFD sweep 1회
위험:   단일덱 총 sweep ≈ 68,000 → 스케줄러 에폭 ≈ 68,000
        에폭당 제어비용 5~10 µs 가정 시 0.34~0.68 s (단일 wall의 1~2%)
        → 제약 32(3%) 이내이나 여유가 크지 않다
```

**폴백 규칙:** W0 스파이크 ③이 conditional 노드 평가비용 `c_cond`를 실측한 뒤,

```text
if (68000 * c_epoch) > 0.03 * solver_wall:
    CMFD quantum := "잔여 sweep 예산 전체" (case별 device 스칼라 nmax)
    단, tail_efficiency 회귀 ≤ 3%p 를 동시 만족해야 승격 확정
else:
    CMFD quantum := sweep 1회 (기본 유지)
```

이 판정은 **Task 3 Step 6(phase quantum table contract)**에 계약으로 박아 넣고, `RASBERY_GPU_PHASE_QUANTUM_CMFD` 플래그로 두 모드를 모두 유지한다(제약 15 롤백 보존).

## 5.2 Queue ownership and duplicate prevention

**[Rev.7.1 수정]** — 구조체 이름만 `DeviceSlotPhase`로 교체. 규칙 4종은 **Rev.7 §5.2 유지**.

## 5.3 Stable classification kernel

`Rev.7 §5.3 유지.` (thread mapping·stable ordering·padding −1)

## 5.4 Phase selection and fairness

**[Rev.7.1 수정]** — Rev.7의 마지막 문장에 단서를 붙인다.

Rev.7 §5.4의 정책(max-age 우선 → ready count → fixed phase order)은 **유지**한다. 다만:

> Rev.7: "case 간 queue 실행 순서는 수치 결과에 영향을 주지 않는다."
> **Rev.7.1 단서: 이는 *케이스 내* 이야기다.** 상태점 간 물리 순서(Xe→TH→SEARCH 순서 규약, predictor→corrector, DERIVATIVE 재수렴 재시도)는 **phase 전이표(§6.21)가 host `Driver.h` 상태기계와 일치해야만** 보장된다. 전이표가 틀리면 fairness와 무관하게 결과가 달라진다.

→ **전이표↔host trace 계약 테스트를 Task 3에 신설한다**(Task 3 Step 6b).

## 5.5 NVIDIA CUDA conditional scheduler backend

`Rev.7 §5.5 유지.` (SWITCH 구조, bucket set 1/2/4/8/16/24/32/48/64, conditional body 제약)

## 5.6 Portable epoch scheduler backend

**[Rev.7.1 수정]** — 후행 트랙. `DispatchDescriptor` 정의와 "control transport ≠ numerical calculation" 원칙만 지금 확정하고, HIP/SYCL 구현은 Task 24/25(후행)로 이관한다. **CUDA epoch backend는 W4-lite Task 23에서 conditional 미지원 fallback으로만 구현**한다.

## 5.7 Optional persistent case-phase scheduler

**[Rev.7.1 수정]** — 제약 17 완화에 따라 "optional"에서 **W3.7 정규 트랙**으로 승격. 조건은 §Global Constraints 제약 17(개정) 참조. Rev.7 §5.7의 cooperative grid 요구사항(모든 블록 동시 상주, inactive 블록도 collective 참여)은 유지.

## 5.8 Stream topology

`Rev.7 §5.8 유지.` (heavy phase 순차 실행 원칙 포함)

## 5.9 Active bucket and physical slot mapping

`Rev.7 §5.9 유지.`

## 5.10 Resource and occupancy policy

**[Rev.7.1 수정]** — receipt 목록에 §9.3 추가 항목 반영. 나머지 유지.

Rev.7 §5.10 receipt 목록에 `l2_hit_rate`, `dram_throughput_pct`를 추가한다(§9.3).

## 5.11 L2, constant, and vendor-neutral math resources

**[Rev.7.1 수정]** — L2를 "실험 항목"에서 **게이트 항목**으로 승격.

Rev.7 §5.11 항목은 유지하되 첫 항목을 다음으로 교체한다.

- immutable geometry/XS는 read-mostly cache hint를 실험한다. **추가로 `_ref_micx` 스트리밍이 CMFD 핫셋의 L2 상주를 파괴하지 않는지 폭별로 측정하고(Task 0), W2 게이트(DRAM ≤ peak 60%)를 통과해야 폭을 확정한다**(§3.6).

---

# 6. Function-by-Function CUDA Mapping

§6.1–§6.13, §6.19는 **Rev.7 유지**. 단 §6.1에 B0 구제 스파이크 조항을 추가한다.

## 6.1 `Nodal::updateConstant`

**[Rev.7.1 수정]** — 분류를 "trajectory-changing 확정"에서 **"B0 구제 스파이크 선행"**으로 완화.

Rev.7 §6.1의 커널·thread mapping·`--fmad=false` 정책은 **유지**한다. 분류만 개정한다.

```text
Rev.7:    trajectory-changing (N1 확정)
Rev.7.1:  N1 후보. 단 Task 4 착수 전 1인일 B0 구제 스파이크를 선행한다.
```

**구제 스파이크(1인일):** `updateConstant`의 `sqrt`/`exp` 호출 인자 범위를 실덱에서 수집하고, 해당 범위에서 CUDA `sqrt`(IEEE 정확반올림 보장) 및 `exp`의 glibc 대비 ULP를 측정한다.
- `sqrt`는 IEEE-754 정확반올림이 규격상 보장되므로 **B0 가능성이 높다**.
- `exp`가 실덱 인자 범위에서 0 ULP면 전체를 B0로 구제하고 Gate A/B·v3 동결을 **생략**한다(검증 비용 대폭 절감).
- 실패 시 N1로 진행(Rev.7 경로 그대로).

## 6.14 Flat XS update

**[Rev.7.1 수정 — 대폭]** — **Rev.7의 2-커널 분할은 실코드에서 불가능하다.** Task 12를 12a/12b로 분리한다.

### 왜 분할이 불가능한가

`XSSet.cpp` 노드 해석 경로는 **부분 적용된 workspace 위에서 다음 좌표를 해석**한다.

```text
XSSet.cpp:2295-2308
    applyDelta(hi, x_vals[branch], wu * f);       // ← 브랜치 delta 적용
    ...
    ResolveSpectralHistoryDeltas(l, history, bm + nmic, ..., wu);
                                      ↑
    이 시점의 (bm + nmic)는 위 applyDelta가 이미 수정한 workspace다.
    즉 "좌표 해석"이 "적용"의 결과에 의존한다 → 순서 의존, 2-커널 분리 불가
```

추가로 `_lib_history_partner`를 경유한 partner 경로(2314-2322)가 같은 workspace에 2차 적용을 수행한다. resolver는 16개 좌표형 ~290줄이며, 실규모 device 이식은 **15~20인일**로 재산정한다.

### Task 12a (W5+ 우선순위) — host resolve + device apply

```text
현행 FlatXsKernel.h 경로를 유지한다.
  host: 좌표 해석 → delta stream 생성 (순서 보존)
  device: stream을 노드별로 적용 (bit-gated, 기존 body)
one CTA = one node, block 64, shared node-local lmp/mic work
```

**분류: B0** (기존 경로와 bit-identical). `flatxs_host_resolve_calls` 수신증으로 host 잔여량을 계측한다(§9.3). **M1은 12a로 달성 가능하다**(세그먼트 내부에서 FlatXS resolve가 호출되지 않는 경로를 확인할 것). M2에는 12b가 필요하다.

### Task 12b (후행·킬 기준부) — device resolve

```text
one thread = one node, 좌표 해석을 device에서 수행
정렬 스캔과 delta 적용 순서를 CPU와 동일하게 보존
log() 사용 → N1
```

**킬 기준: 착수 후 5인일 내에 16개 좌표형 전부가 ≤4 ULP를 달성하지 못하면 12b를 폐기하고 12a를 영구 유지한다.**

## 6.15 Xe equilibrium + Anderson

**[Rev.7.1 수정 — 대폭]** — **분류를 N1으로 재조정하고 4건을 개정한다.**

### (1) `k_xe_dot_reduce` 신설 — Anderson은 warp-per-slot으로 불가능하다

Rev.7 §6.15 `k_xe_anderson_coeff`는 "one native subgroup = one slot"으로 Gram 행렬을 계산한다. **불가능하다.** Anderson 내적 `XeDot`은 **약 15,000항(= 3 성분 × n_fuel)을 직렬 누적**하며, depth 2/3에서 이 내적이 **6회** 필요하다. 32레인이 90,000항을 나눠 가지면 CPU의 직렬 누적 순서가 깨진다.

```cpp
// 신설: 고정분할 2-stage 내적 (기존 CMFD reduce_dot과 동일 규약)
GPU_GLOBAL void k_xe_dot_reduce_stage1(   // 고정 chunk, gridDim.x 불변
    int n_fuel, const double* a, const double* b, double* partials, int slot);
GPU_GLOBAL void k_xe_dot_reduce_stage2(   // one thread/slot, strict fold
    int n_partials, const double* partials, double* out, int slot);
```

- **stage-1 partition은 CPU 직렬 누적과 다르다** → `XeDot` 자체가 궤적 변경이다. **Anderson 경로는 N1**이다.
- `k_xe_anderson_coeff`는 stage-2 결과(6개 스칼라)만 받아 슬롯당 1스레드로 depth 2/3 소규모 dense solve를 수행한다. 이 부분만 subgroup-free다.

### (2) 분류 분리 — Picard는 B0, Anderson은 N1

```text
Picard 경로 (xe_relax 적용, Anderson OFF):  B0 — bit-gated 유지
Anderson 경로 (k_xe_dot_reduce 경유):       N1 — Gate A/B + 수용률 ±10% + v3 동결
```

**N1 추가 게이트: Anderson 후보 수용률이 v2 대비 ±10%p 밖이면 fail.** 수용률이 흔들리면 궤적이 아니라 알고리즘이 바뀐 것이다.

### (3) 기각 커밋은 감쇠 Picard다 — Rev.7의 "기존 Picard image commit"은 틀렸다

Rev.7 §6.15 `k_xe_commit_reconstruct`: "Reject는 기존 Picard image를 commit한다."
**실제 CPU 경로는 기각 시 `xe_relax`(= `XE_DAMPED_RELAX`)를 적용한 감쇠 Picard를 커밋**한다(`Driver.h:1442-1444`). 완전 Picard를 커밋하면 진동이 재발한다.

```text
개정: reject → commit( x_k + xe_relax * (F(x_k) - x_k) )
      accept → commit( cand )
두 이미지 모두 device 버퍼에 보존하고 device-side swap. host rollback 없음(Rev.7 유지).
```

### (4) history reset 엣지 — TH/SEARCH 커밋에서 반드시 초기화

`Driver.h:1022-1024`: Anderson history는 **map이 고정된 동안에만 의미**가 있다. macro-XS를 움직이는 모든 사건이 history를 무효화한다.

```text
필수 reset 엣지 (DeviceSlotState.xe_aa_ncol=0, xe_aa_have_prev=0):
    ThermalHydraulics → Xenon    (TH step 커밋)
    Search            → Xenon    (search trial 커밋)
    Material          → Xenon    (rod/cusping으로 macro-XS 변경 시)
    DepletionPredictor/Corrector 진입 시 (SolveLoop 경계 = history 소멸)
```

**Task 13 Step 2 replay는 이 4개 엣지에서 reset 횟수가 CPU와 정확히 일치해야 통과한다.** 누락 시 잘못된 map의 차분열이 살아남아 궤적이 갈린다.

### 커널 매핑 (개정 후)

| 커널 | 매핑 | 분류 |
|---|---|---|
| `k_xe_evaluate` | one CTA = one fuel node, block 64 | B0 |
| `k_xe_dot_reduce_stage1/2` | 고정 chunk 256 / one thread per slot | **N1 (신설)** |
| `k_xe_anderson_coeff` | **one thread = one slot** (subgroup 아님) | N1 |
| `k_xe_candidate` | one thread = one fuel node, block 256 | B0 |
| `k_xe_commit_reconstruct` | one CTA = one fuel node, block 64 | B0(Picard) / N1(AA) |

## 6.16 Thermal-hydraulics

**[Rev.7.1 수정]** — 3건 개정.

Rev.7 §6.16 Kernel A/C/D/E의 thread mapping은 **유지**한다. 다음을 개정한다.

### (1) Kernel B: 슬롯 합계는 "기존 fixed partition 2-stage"가 아니라 **strict serial fold**다

CPU의 노드 파워 합계는 **노드 인덱스 오름차순 직렬 누적**이다. 2-stage partition reduction은 결과가 다르다.

```text
Rev.7:    "기존 fixed partition two-stage reduction을 사용한다" → 궤적 변경
Rev.7.1:  strict serial fold (one thread = one slot, l=0..nxyz-1 순차 누적)
          → B0 유지. 8,451회 직렬 누적의 비용은 슬롯당 ~µs 수준으로 무시 가능.
```

동일 규약을 TH의 모든 슬롯 총계(tf_avg/tm_avg/dm_avg 및 volume-weighted 평균)에 적용한다.

### (2) TH 테이블 보간은 **B0**다 (Rev.7의 N1 분류는 과보수)

Rev.7 §9.1은 "GPU TH table interpolation"을 N1에 넣었다. 실제 CPU 경로는 **bilinear 보간**이며, 곱셈·덧셈만 사용한다.

```text
분류: B0 (form mask 보존 조건부)
조건: 보간식의 곱셈·덧셈 순서와 FMA 마스크를 CPU와 동일하게 고정
      → --fmad=false + 명시적 fma() 사이트만 허용
```

→ **Gate A/B와 v3 동결이 불필요**해진다. 검증 비용 절감이 크다.

### (3) burnup 브래킷 키는 **정수 키**여야 한다

burnup 구간 탐색이 부동소수 비교면 GPU/CPU에서 경계 노드가 갈릴 수 있다. CPU가 산출한 브래킷 인덱스와 동일한 **정수 키**를 device가 재현하도록 계약을 건다(Task 14 Step 1 replay에 브래킷 인덱스 비교 포함).

## 6.17 Critical boron/rod search

**[Rev.7.1 수정]** — **thread mapping을 슬롯당 1스레드로 교체.** Rev.7의 32레인 매핑은 존재하지 않는 배열을 스캔한다.

```text
Rev.7:   "lanes 0–31: trial history scan, best residual, validity flags"
사실:    검색 상태는 배열이 아니라 유한 스칼라 26필드다 (Scheduler.h:155-177, §3.2(C)).
         스캔할 history 배열이 없다. 레인 1–31은 전부 유휴다.
```

```text
Rev.7.1 매핑:
    one thread = one slot   (block 128, grid = ceil(slots/128))
    스레드가 DeviceSearchState 26필드를 읽어 bracket/secant/relaxation을 결정하고
    선택된 boron/rod 값을 DeviceSlotState에 커밋한다.
```

- boron 갱신은 node-parallel 커널, rod 갱신/cusping은 rod 커널(Rev.7 유지).
- settle counter·noisy limit-cycle sample semantics는 host와 동일(Rev.7 유지).
- **분류: B0.** 검색 산법은 비교·사칙연산·secant 나눗셈뿐이며 초월함수가 없다. Rev.7 §9.1의 "device search algebra = N1"은 과보수이며 **B0로 재분류**한다(단 `min_secant_denom` 가드와 나눗셈 순서를 보존할 것).

## 6.18 Depletion predictor/corrector

**[Rev.7.1 수정 — 전면 재작성]** — **Rev.7의 dense LU CRAM은 실코드 산법이 아니다.**

### 실제 CPU 산법 (`include/milk.h:1649-1819`)

```text
solveBatemanCRAM(A, N, dt, out, workspace, order=8, first=3)
  ← XSSet.cpp:4008  CRAM_ORDER=8, first=iI135=3

산법: dense LU/pivot 아님. 극(pole)마다 희소 Gauss-Seidel 반복.

order 8 파라미터 (milk.h:1676-1701):
    alpha0     = 1.1722341374385704e-08
    pole_count = 4                (16차의 8극이 아니다)
    max_iter   = 64
    rel_tol    = 1.0e-13
    matrix_sgn = -1.0
    alpha[4], theta[4]            (본문 상수 그대로 device constant에 복사)

수렴/가드 (milk.h:1732-1733, 1800):
    abs_tol  = 1.0e-28
    diag_tol = 1.0e-30
    수렴 판정: max_residual <= abs_tol + rel_tol * rhs_norm   (max-norm)

희소 구조 (milk.h:1742-1758):
    행마다 대각을 분리 저장(base_diag), 비대각만 (cols, vals)로 압축
    대각은 pole-invariant → pole마다 (base_diag - theta[pole]) 한 번만 계산

대상 (XSSet.cpp:4036):
    for (int l = 0; l < nxyz; ++l) DepleteNode(...)
    → fuel node가 아니라 전 8,451 노드
```

### GPU 매핑 (개정)

```text
one thread = one node        (Rev.7의 one CTA = one node 폐기)
block = 128
grid  = ceil(nxyz/128) = 67,  grid.y = bucket
```

**왜 CTA가 아니라 스레드인가:** 39×39 dense가 아니라 희소 GS다. 한 노드의 작업은 `pole_count(4) × max_iter(≤64) × nnz(행당 ~3–6)`로, CTA 64스레드를 채울 병렬성이 없다. CTA를 쓰면 스레드 대부분이 유휴이고 `__syncthreads()` 비용만 남는다.

**메모리 배치:**

```text
공유(전 노드 동일): 희소 패턴 base_cols / row_offsets  → constant 또는 shared
                     alpha[4], theta[4]                  → constant
per-node(전역 SoA): base_vals(비대각 값, 노드마다 다름)
                     base_diag[39], pole_diag[39], rhs[39], x[39], accum[39]
                     complex double 39 × 5 배열 = 약 3.1 KiB/node
                     → shared 불가(128스레드 × 3.1 KiB = 397 KiB ≫ 128 KB/SM)
                     → 전역 workspace SoA, 노드-stride coalesced 배치
```

> sm_120의 shared memory는 **128 KB/SM**이다. Rev.7이 상정한 "complex 39×39 shifted matrix ≈ 24.3 KiB shared"는 dense 전제이며, 희소 GS에서는 그런 행렬을 만들지 않는다.

**순서 고정 계약 (B0 보존 조건):**

```text
pole 루프:  0 → 3 순서 고정
row  루프:  first(=3) → n-1 오름차순, in-place GS 갱신(x[row] 즉시 반영)
비대각 합:  cols 배열 저장 순서대로 직렬 누적 (재정렬 금지)
잔차:       max-norm, 같은 순서로 스캔
누적:       accum[row] += x[row]  (pole 순서 고정)
최종:       out = alpha0*N + 2*Re(accum),  |value|<1e-12 && value<0 → 0
```

**분류: B0 후보.** 순서를 위와 같이 완전 보존하면 초월함수가 없으므로(복소 곱/나눗셈뿐) bit-identical이 가능하다. 단 복소 나눗셈의 구현(Smith 알고리즘 vs naive)이 glibc와 다를 수 있으므로 **Task 16 Step 1 replay가 0 ULP를 못 내면 N1로 강등**한다.

**탈출 코드:** device는 throw할 수 없다.

```text
milk.h:1765  "CRAM Gauss-Seidel zero diagonal"  → DeviceEscape::CramZeroDiagonal
milk.h:1807  "CRAM Gauss-Seidel did not converge" → DeviceEscape::CramNotConverged
두 경우 모두 해당 슬롯을 Failed로 전이. 다른 슬롯은 계속 진행(Task 16 Step 4).
```

**킬 기준: GPU CRAM이 CPU 대비 1.5× 미만이면 CPU에 잔류**시킨다. depletion은 단일 wall에서 지배적이지 않으며, 8,451노드 × 4극 × 64반복의 직렬 GS는 GPU에 유리한 형상이 아니다.

### Predictor/Corrector

**[Rev.7.1 수정]** — BOS 스냅샷 정책 정정.

```text
Rev.7:   BOS microscopic XS 전체 복사를 "제거"
Rev.7.1: 제거가 아니라 11 → 4 슬롯 축소 (bit-보존 확인된 범위)
         보존 대상: BOS isotope density / BOS flux / BOS condensed reaction-rate / BOS burnup
         RASBERY_PC_MODE=decart 를 v1 필수로 고정
```

메모리 영향은 §3.6 표에 반영되어 있다(약 24 MiB/slot).

## 6.19 Burnup

**[Rev.7.1 수정]** — 슬롯 총계는 §6.16(1)과 동일하게 **strict serial fold**로 고정한다. 나머지 Rev.7 §6.19 유지.

## 6.20 Output numerical packing

**[Rev.7.1 수정]** — **분류 N1로 승격.** Rev.7 §9.1은 output pack을 분류 목록에서 누락했다(암묵 B0).

실측: AO 재배열만으로 **1.2e-16 상대 차이**가 발생한다. 축방향 재배열/정규화의 누적 순서가 CPU와 다르면 AO/Fq/Fr가 마지막 자리에서 갈린다.

```text
분류: N1
대상: keff/CBC/AO/ASI/Fq/Fr/Fqp/Frp, xe_avg/xe_ao/sm_avg/sm_ao/gd_avg,
      tf/tm/dm avg·max, asm_power/asm_burn/asm_kinf, ax_power/ax_tful/ax_tmod/ax_dmod
게이트: run-to-run bit identity + Gate A/B + v3 동결
완화 경로: 축방향/조립체 누적을 strict serial fold로 고정하면 B0 구제 가능.
          Task 19 Step 1에서 fold 순서를 CPU와 대조해 B0 여부를 먼저 판정한다.
```

CPU writer가 byte-only여야 한다는 원칙(Rev.7)은 유지한다.

## 6.22 Pin power reconstruction (PPR) — **[Rev.7.1 신설]**

**Rev.7에 PPR이 0회 등장한다.** `src/PPR.cpp`는 1,099줄이고 `Driver.h:2278-2295`에서 매 인쇄 상태점마다 실행되며, `RASBERY_PPR_ITERS` 기본값 **100회** corner-balance 반복 + 3×3 Gauss-Legendre 핀면적 적분을 수행한다. **단일 wall의 6~10%**를 차지한다.

```text
호출 경로 (Driver.h:2278-2295):
    pin_power_reconstruction.reset(1.0/eigv, geometry.Jnet(), Phif(), Phis());
    pin_power_reconstruction.drive(ppr_iters);          // 기본 100
    pin_power_reconstruction.reconstructPinPower(true, print_opt.pin_flux);
```

### 커널 분해 (Task 19b)

| 커널 | 원본 | 매핑 |
|---|---|---|
| `k_ppr_reset` | `PPR::reset` (PPR.cpp:179) | one thread = (node, group) |
| `k_ppr_axial_leakage` | `PPR::updateAxialLeakage` (:579) | one thread = node |
| `k_ppr_source` | `PPR::updateSource` (:630) | one thread = (node, group) |
| `k_ppr_corner` | `PPR::updateCorner` (:675) | one thread = corner |
| `k_ppr_update_fused` | `PPR::updateFused` (:285) | one CTA = (node, group), 3×3 스텐실 |
| `k_ppr_reconstruct` | `PPR::reconstructPinPower` (:767) | one thread = pin, 9-point GL 적분 |

**quantum:** `READY_PPR` = corner-balance 반복 **1회**(reset→leakage→source→corner→fused). 100회 전체를 한 quantum으로 묶으면 tail이 재발한다.

**분류:** 산술이 곱/합/스텐실이므로 **B0 후보**. `getJoutRed`(PPR.h:118)의 나눗셈 순서와 `_crdf` 인덱싱을 보존할 것. 3×3 GL 가중치 누적 순서는 `qpts[9]` 배열 순서로 고정한다.

**N1 자체 게이트 (승격 시):** 핀출력 rms가 MASTER 대비 0.84%(수렴값) 수준을 유지해야 한다. `docs/`의 기록에 따르면 cap 5에서 4.76% rms, 수렴 시 0.84%이며 50~200회는 동일하다. **GPU 경로가 100회에서 0.84%를 벗어나면 fail.**

**계약:** `RASBERY_PPR_ITERS` 환경변수 의미를 device 경로에서도 그대로 유지한다(Driver.h:2286-2290의 기본 100, 비양수 → 100 fallback 포함). 이 값이 A/B arm 기록에 포함되어야 한다(제약 5).

## 6.21 Function-to-queue transition map

**[Rev.7.1 수정]** — 신설 phase 5종 + PPR 엣지 추가, **계약 테스트 신설**.

Rev.7 §6.21 표에 다음 엣지를 **추가**한다.

| Completed kernel/graph | Same-case next phase | Scheduler action |
|---|---|---|
| flux 세그먼트 종료(SolveLoop 1회 완료) | `NormalizeFluxSign` | Driver.h:2235/2239/2261/2274 대응 |
| `NormalizeFluxSign` (predictor 전) | `DepletionPredictor` | Driver.h:2237 |
| `NormalizeFluxSign` (corrector 후) | `Ppr` 또는 `Derivative` | schedule.type 분기 |
| `Derivative` 판정: k 붕괴 | `RodOp`→`Outer` 재수렴 1회 | Driver.h:2268-2274, 재시도 1회 한정 |
| `Derivative` 판정: 정상 | `Ppr` | — |
| `RodOp` | `Material`→`Outer` | Driver.h:2252-2255 |
| `Ppr` (반복 미완료) | `Ppr` | requeue, iteration++ (최대 `RASBERY_PPR_ITERS`) |
| `Ppr` (반복 완료) | `ResultAggregate` | reconstructPinPower 후 |
| `ResultAggregate` | `OutputPack` | Driver.h:2308 |

**[Rev.7.1 신설 계약] 전이표 ↔ host trace 대조 테스트.**
`tools/test_phase_transition_vs_host_trace.py`가 Task 0에서 캡처한 host 상태기계 trace(`case_phase_trace.jsonl`)를 읽고, 위 전이표를 상태기계로 실행해 **동일한 phase 시퀀스**를 생성하는지 검증한다.

```text
검증 대상 deck 유형 (전부 통과 필수):
    STANDARD / DEPLETION(substep>1) / DERIVATIVE(재수렴 발생 케이스 포함)
    ROD / BORON search / RODCRIT search / restart / shuffle
실패 조건:
    phase 시퀀스 불일치 1건 → fail
    전이표에 없는 host 엣지 발견 → fail (표 누락 탐지)
```

이 테스트는 **Task 3 Step 6b**로 편입되며, W1 종료 게이트다. 전이표가 틀리면 스케줄러 fairness와 무관하게 물리 결과가 갈린다(§5.4 단서).

### CMFD quantum 선택 / Nodal quantum 선택

`Rev.7 §6.21 하위 절 유지` — 단 CMFD quantum에는 §5.1(2) 폴백 조항이 적용된다.

---

# 7. Thread and Block Assignment Summary

**[Rev.7.1 수정]** — 개정된 매핑 4행 교체 + 신설 3행 추가. 나머지 Rev.7 §7 표 유지.

| Task | Smallest independent unit | Default block | 변경 |
|---|---|---:|---|
| phase classify/compact | slot (`DeviceSlotPhase` 32 B만 읽음) | 128 | 읽는 구조체 축소 |
| **search** | **slot (1 thread)** | 128 | ~~slot subgroup~~ → §6.17 |
| **Xe Anderson coeff** | **slot (1 thread)** | 128 | ~~subgroup~~ → §6.15(1) |
| **Xe dot reduce s1/s2** | **fixed chunk / slot** | 256 / 128 | **신설** §6.15(1) |
| **depletion CRAM** | **node (1 thread)** | 128 | ~~fuel node CTA 64/128~~ → §6.18 |
| **TH/burnup slot total** | **slot (1 thread, strict fold)** | 128 | ~~2-stage reduction~~ → §6.16(1) |
| **PPR fused** | (node, group) CTA | 64/128 | **신설** §6.22 |
| **PPR reconstruct** | pin | 256 | **신설** §6.22 |

---

# 8. Large-Input Asynchronous Case-Phase Execution

§8.1, §8.3–§8.7, §8.9, §8.10은 **Rev.7 유지**.

## 8.2 Immediate slot refill

**[Rev.7.1 수정]** — reset 대상을 §3.2 4구조체 전량으로 확장. 절차는 Rev.7 유지.

```text
Done/Failed output snapshot 완료
→ state_epoch++
→ DeviceSlotPhase / DeviceSlotState / DeviceSearchState / DeviceScheduleParams 전량 reset
→ mask / mirror / graph status / Krylov flag reset
→ next input descriptor claim
→ import state
→ phase = Import/Material
```

`k_audit_tenant_reset`이 refill 직후 4구조체를 감사한다(§3.2 재활용 감사).

## 8.8 Scheduler trace replay before production

**[Rev.7.1 수정]** — **W1에서 W0으로 승격.** 저비용·고가치이며, 스케줄러 형태를 구현 전에 결정한다.

Rev.7 §8.8의 5개 정책 재생과 채택 예측 기준(GPU idle 20% 이상 감소, cohort tail 감소, tail efficiency 개선, scheduler operation count acceptable)은 **유지**한다. 실행 시점만 변경한다.

```text
Rev.7:   W1 (Task 3 Step 7)
Rev.7.1: W0 — Task 0과 병행. 기존 M64 trace로 오프라인 재생.
         산출: 5개 정책의 idle/tail 예측치 → W1 착수 전 스케줄러 형태 확정
         W3 종료 시 예측치 vs 실측을 대조한다(예측 모델 검증).
```

---

# 8A. GPU Portability Architecture and Support Matrix

**[Rev.7.1 수정]** — 후행 트랙. §8A.1(필수 capability 목록), §8A.2(subgroup 추상화), §8A.5(명시적 미지원)는 **설계 문서로 유지**하되 구현은 Task 24/25(하드웨어 확보 후)로 이관한다. §8A.3 `selectBackend`는 **CUDA conditional / CUDA epoch 두 tier만** W4-lite Task 23에서 구현한다. §8A.4 검증 매트릭스의 HIP/SYCL 행은 후행이다.

**지금 적용되는 것:** 제약 35(공용 pure body 규율)와 §8A.2의 "`one warp per slot` 문구는 portable code에서 `one subgroup/workgroup per slot`로 해석" 규약. 단 §6.15/§6.17의 개정으로 실제 slot 스칼라 커널은 **subgroup을 아예 쓰지 않고 1스레드**가 된다.

# 8B. Reactor Model Portability

**[Rev.7.1 수정]** — 별도 캠페인. §8B 전체는 설계 문서로 유지하되, 본 계획에서 실행하는 것은 **Task 26 축소판(descriptor + provenance 수신증)뿐**이다. §8B.2(Cartesian 2G fast path를 느리게 만들지 않는다)는 **지금부터 지켜야 할 제약**으로 유지한다.

---

# 9. Numerical Correctness and Failure Policy

## 9.1 Three validation classes

**[Rev.7.1 수정]** — 분류 6건 정정.

### Class B0 — bit-preserving

```text
state residency
slot compaction
shared pointers
updpsi / upddtil / updjnet / upddhat
CMFD / Nodal graph relocation
conditional control with identical trajectory
--- Rev.7.1 재분류(N1 → B0) ---
TH 테이블 bilinear 보간 (form mask 보존 조건)        ← §6.16(2)
device search 산법 (min_secant_denom 가드 보존)      ← §6.17
FlatXS 12a (host resolve + device apply)             ← §6.14
--- Rev.7.1 B0 후보(replay 0 ULP 시 확정) ---
GPU CRAM 희소 GS (순서 완전 보존)                     ← §6.18
GPU PPR (fold/나눗셈 순서 보존)                       ← §6.22
Nodal updateConstant (구제 스파이크 통과 시)          ← §6.1
```

Gate: `single 500/500 bit identical` + `batch golden bit identical` + `outer/sweep/BiCG counts exact` + **`fallback 0` + `graph miss 0` + `총 인스턴스화 wall ≤ 250 ms`**(신설, §12 Task 10).

### Class N1 — deterministic GPU baseline transition

```text
GPU Nodal constants (구제 스파이크 실패 시)
GPU FlatXS 12b coordinate log
GPU CRAM (replay 0 ULP 실패 시)
--- Rev.7.1 신규 N1 ---
Xe Anderson (k_xe_dot_reduce 고정분할)                ← §6.15(1)(2)
output numerical pack (AO 재배열 1.2e-16 실측)        ← §6.20
--- Rev.7.1 삭제 (B0로 이동) ---
~~GPU TH table interpolation~~
~~device search algebra~~
```

Gate: `GPU run-to-run bit identical on sm_120` + `Gate A` + `Gate B` + `v3 baseline freeze` + **Anderson의 경우 후보 수용률 ±10%p**.

### Class A2 — algorithmic acceleration

```text
interim Xe (device 발화 productionize)     ← Task 13a, W3.6
flux-space Anderson                        ← Task 13b, W3.6
vectorized-GS CRAM
adaptive inner tolerance
```

Gate: `Gate A/B` + `별도 브랜치` + `production residual polish` + `explicit rollback` (Rev.7 유지).

## 9.2 GPU-full fail closed

**[Rev.7.1 수정 — 중요]** — **`gpu_numeric_fraction=1.0`을 M1/M2 2단 마일스톤으로 분리한다.**

Rev.7은 host 수치 경계가 어디서 달성되는지 명시하지 않아, `host_numeric_calls=0`이 언제 참이 되는지 검증할 수 없었다. `Driver.h` 상태점 루프의 host 수치 호출부를 전부 열거해 두 마일스톤으로 나눈다.

### M1 — flux 세그먼트 내 host 수치 0

**정의:** 한 번의 `SolveLoop` 호출이 시작되어 반환할 때까지 host가 수치를 계산하지 않는다.
**달성 태스크:** Task 9(device outer 상태기계) + Task 10(conditional case-phase graph).
**검증:** `host_numeric_calls` 계측을 `SolveLoop` 진입/이탈로 구간화해 세그먼트 내 카운트가 0.
**이것이 W3의 종료 게이트다.**

### M2 — 상태점 내 host 수치 0

**정의:** `PrepareForStep`부터 `AddResult`까지 host가 수치를 계산하지 않는다.
**달성 조건: 다음 `Driver.h` 호출부가 전부 device phase로 이관되어야 한다.**

| `Driver.h` 위치 | 호출 | 담당 phase / Task |
|---|---|---|
| `2194` | `schedule.PrepareForStep(CoreHeavyMetalMassKg())` | `Import` 확장 / Task 20 |
| `2195` | `schedule.ApplyToGeometry(geometry)` | `Import` 확장 / Task 20 |
| `2200` | `cross_sections.SetPowerRate(power_fraction)` | `Material` / Task 12a |
| `2235`, `2239`, `2261`, `2274` | `cross_sections.NormalizeFluxSign()` | **`NormalizeFluxSign`** (신설) / Task 19 |
| `2245` | `cross_sections.UpdateDerivative(Δbppm,Δtful,Δtmod,Δdmod)` | **`Derivative`** (신설) / Task 14+15 |
| `2252-2255` | `SetRod` / `ResetFluxAndCurrents` / `resetDhat` / `eigv=1` | **`RodOp`** (신설) / Task 11 |
| `2268-2274` | DERIVATIVE k 붕괴 판정 + 재수렴 1회 | **`Derivative`** (신설) / Task 9 |
| `2278-2295` | PPR `reset`/`drive`/`reconstructPinPower` | **`Ppr`** (신설) / **Task 19b** |
| `2308` | `input_output.AddResult(...)` | **`ResultAggregate`** (신설) / Task 19 |

**`gpu_numeric_fraction=1.0`은 M2에서만 주장할 수 있다.** M1 단계의 수신증은 `gpu_numeric_fraction`을 기록하지 않고 `segment_gpu_numeric_fraction`만 기록한다.

### fail-closed 규칙

```text
RASBERY_GPU_FULL=1:
  any CPU numerical call not in host_numeric_allowlist[] -> error
  any CUDA fallback -> error
  any stale generation -> error
  any nonfinite -> slot failed
```

**[Rev.7.1 신설] `host_numeric_allowlist[]`.** M1↔M2 사이의 과도기에는 아직 이관되지 않은 호출부를 명시적 allowlist로 선언하고 수신증에 그대로 기록한다. **빈 allowlist = M2 달성.** 암묵적 허용은 금지하며, allowlist에 없는 호출은 즉시 fail한다.

개발 모드 `RASBERY_GPU_MODE=hybrid_verified`만 CPU reference fallback을 허용한다(Rev.7 유지).

## 9.3 Final receipt

**[Rev.7.1 수정]** — 7종 추가.

Rev.7 §9.3의 두 JSON 블록은 유지하되 다음 필드를 **추가**한다.

```json
{
  "milestone": "M1|M2",
  "segment_gpu_numeric_fraction": 1.0,
  "host_numeric_allowlist": [],

  "xe_aa_proposed": 0,
  "xe_aa_accepted": 0,
  "xe_aa_rejected": 0,
  "xe_aa_reset_events": 0,
  "xe_aa_accept_rate": 0.0,

  "ppr_host_calls": 0,
  "ppr_iters": 100,
  "flatxs_host_resolve_calls": 0,

  "tail_efficiency": 0.0,
  "l2_hit_rate": 0.0,
  "dram_throughput_pct": 0.0,

  "cram_escape_not_converged": 0,
  "cram_escape_zero_diagonal": 0,
  "graph_instantiation_wall_ms": 0.0,
  "graph_warmup_miss": 0
}
```

**게이트 연결:**

| 필드 | 게이트 |
|---|---|
| `host_numeric_allowlist` | 비어야 M2 주장 가능 |
| `xe_aa_accept_rate` | v2 대비 ±10%p 이내 (N1) |
| `xe_aa_reset_events` | CPU trace와 정확히 일치 (§6.15(4)) |
| `ppr_host_calls`, `flatxs_host_resolve_calls` | M2에서 0 |
| `tail_efficiency` | 스케줄러 채택 판정의 주지표 |
| `dram_throughput_pct` | **W2 게이트: ≤ 60** (§3.6) |
| `graph_instantiation_wall_ms` | **≤ 250** (§12 Task 10) |
| `graph_warmup_miss` | **warm-up 후 0** (§12 Task 10) |

---

# 10. Runtime Flags

**[Rev.7.1 수정]** — 신설 플래그 6종 추가. Rev.7 §10 목록은 유지.

```text
RASBERY_GPU_PHASE_QUANTUM_CMFD=1|budget   # 1=sweep 1회, budget=잔여 예산 전체 (§5.1(2))
RASBERY_GPU_PPR=1                          # Task 19b device PPR
RASBERY_PPR_ITERS=100                      # 기존 플래그. device 경로에서도 동일 의미 (§6.22)
RASBERY_GPU_FLATXS_RESOLVE=host|device     # 12a / 12b (§6.14)
RASBERY_PC_MODE=decart                     # v1 필수 (§6.18)
RASBERY_GPU_XE_AA_DEVICE=0|1               # k_xe_dot_reduce 경로 (§6.15)
RASBERY_GPU_L2_PROBE=0                     # Task 0 폭별 L2/DRAM 측정 모드
```

---

# 11. Implementation Waves

**[Rev.7.1 수정 — 전면 재편]** — Rev.7의 W0→W10은 **순수성(100% GPU) 우선** 순서다. 승인된 프로그램은 **속도 우선**이며, 순수성·이식성 잔여는 실측 후 재판정한다.

## 11.1 Wave 프로그램

| Wave | 인일 | 범위 | 종료 게이트 |
|---|---:|---|---|
| **W0 판별** | 5 | 4대 스파이크 + trace replay + Task 0 | persistent 관문·스케줄러 형태·CMFD quantum·residency 범위 확정 |
| **W1** | 10 | Task 1, 2, 3 | 스케줄러 계약 PASS + **전이표↔host trace 계약 테스트 PASS** |
| **W2** | 22 | Task 4, 5, 6, 7, 8 | bit-golden + **DRAM ≤ peak 60%** |
| **W3** | 16 | Task 9, 10 | **M1 달성** + trace replay 예측치 실측 대조 |
| **W3.5** | 10 | Task 13 | Xe Picard B0 + Anderson N1 게이트 |
| **W3.6 (A2)** | 12 | Task 13a, 13b | Gate A/B + v3 동결. **300→100 outer/상태** |
| **W3.7** | 5 | Task 21 (승격) | PDL >1% / persistent 7% 게이트 (c_barrier 통과 시) |
| **W4-lite** | 12 | Task 20, 23, 26-lite, 멀티GPU dispatcher, 22 | queue duplicate 0 / stale tenant 0 / v3 동결 |
| **W5+ 재판정** | ~75 | Task 11, 12b, 14, 15, 16, 17, 18, 19, 19b | 별도 승인 |
| **후행** | — | Task 24, 25 (하드웨어 게이트), 27, 28 (별도 캠페인) | — |

**core 합계 ≈ 92인일** + W5+ 재판정 대상의 조기 승격분을 포함해 **~115인일 ≈ 5~6개월**.

### W0 — 판별 (5인일)

속도 목표의 모든 분기점을 **구현 전에** 숫자로 닫는다.

```text
스파이크 ① dispatch 바닥      — 커널 launch/graph launch 하한 실측
스파이크 ② grid.sync 비용     — c_barrier(sm_120, 34~67 blocks)
                                 → persistent GO/NO-GO (킬: >0.384 µs)
스파이크 ③ WHILE{SWITCH{child}} — 인스턴스화 wall + conditional 노드 평가비용 c_cond
                                 → CMFD quantum 확정 (§5.1(2))
스파이크 ④ 폭별 L2/DRAM        — W ∈ {1,8,16,32,64}에서 l2_hit_rate/dram_throughput_pct
                                 → residency 범위·슬롯 폭 확정 (§3.6)
§8.8 trace replay              — 기존 M64 trace로 5개 스케줄러 정책 오프라인 재생
                                 → idle −20% 예측 확인, 스케줄러 형태 확정
Task 0                          — 기준선 동결, provenance/AB arm 분리, escape 비율 4종
```

### W1 (10인일) — Task 1, 2, 3

- **Task 1**: 타입 정의. **§3.2의 4분할 구조**를 그대로 구현한다(수정 4).
- **Task 2**: `GpuPhysicsArena`. **압축 없음**(§3.6). 슬롯당 225 MiB 상한으로 admission.
- **Task 3**: case-phase queue + classify/compact. epoch 소유권 규칙, **전이표↔host trace 계약 테스트**, **CMFD quantum 폴백 조항**.

### W2 (22인일) — 단일 wall 34% 삭제 + 기아 원천 제거

```text
Task 4 (nodal 상수)   ← B0 구제 스파이크 선행 (§6.1)
  → Task 5 (updpsi/dtil/jnet/dhat/수렴·stall — clean_iters/xe_interim/stall_sample 포함)
  → Task 6 (단일 resident CMFD — sweep_unroll 그래프 키 제거 완결)
  → Task 7 (canonical 공유 상태 — L2 게이트)
  → Task 8 (Nodal compaction)
```

### W3 (16인일) — Task 9, 10 → **M1**

- **Task 9**: device outer 상태기계. **확장된 `DeviceSlotState`** 사용, **interim-Xe 봉쇄 삭제**(Rev.7 Task 9 Step 4의 `RASBERY_XE_INTERIM_L2>0` 거부 계약을 제거 — W3.6 A2가 이를 필요로 한다).
- **Task 10**: conditional case-phase graph. 단일 스트림 + SWITCH, **인스턴스화 wall ≤250 ms / warm-up 후 miss 0 게이트**, staging lease 이벤트.
- **종료: M1 달성 + W0 trace replay 예측치와 실측 대조.**

### W3.5 (10인일) — Task 13 (GPU Xe)

3커널 분해, Picard bit-게이트 / Anderson N1 게이트, `k_xe_dot_reduce`, history reset 4엣지.

### W3.6 — A2 트랙 (12인일) — **최대 단일 레버**

**Task 13a: Interim-Xe device 발화 productionize (A2)**
**Task 13b: Flux-space Anderson (A2)** — 기존 safeguard 인프라 재사용, 스파이크 → 구현.

**목표: 상태점당 outer 300 → 100.** 단일·배치 목표 양쪽에서 가장 큰 단일 레버다. Gate A/B + 별도 브랜치 + v3 동결 + 명시적 롤백.

### W3.7 (5인일) — Task 21 승격

cooperative persistent CMFD. **W0 스파이크 ② 통과 시에만 착수**(제약 17 개정). PDL은 end-to-end median >1% 게이트.

### W4-lite (12인일)

```text
Task 20    즉시 refill + massive-input (1,280잡, queue duplicate 0 / stale tenant 0)
Task 23    capability probe · 백엔드 계약 (CUDA conditional / CUDA epoch 두 tier만)
Task 26-lite  descriptor + provenance 수신증
멀티GPU dispatcher  프로세스-per-GPU, NUMA 고정
           호스트 예산표: 4,340 c/h = 라이터 ~8스레드 + GPU당 로더 8~16
Task 22 1차  v3 동결 · 검증 · Draft PR
```

### W5+ 재판정 (~75인일) — 별도 승인 필요

```text
Task 11    rod cusping
Task 12b   device FlatXS resolve      킬: 5인일 내 16좌표형 ≤4 ULP 실패 시 12a 유지
Task 14    TH
Task 15    search
Task 16-17 CRAM (GS 재작성판)          킬: CPU 대비 1.5× 미만 시 CPU 잔류
Task 18    전 phase 통합 스케줄러
Task 19    output pack
Task 19b   GPU PPR                     ← 신설. 단일 wall 6~10%이므로 조기 승격 후보 1순위
```

**W0·W3.7 실측 후 재판정한다.** 특히 Task 19b는 M2 달성의 필수 항목이면서 단일 wall 기여가 커서, W3 종료 시점에 W4-lite로 승격할지 재검토한다.

### 후행 트랙

```text
Task 24 (HIP) / Task 25 (SYCL)   AMD/Intel 하드웨어 확보 후 착수. 지금은 제약 35만 적용.
Task 27 / Task 28 (원자로 일반화)  별도 캠페인. 본 계획 인일에 미포함.
```

## 11.2 킬 기준 (요약)

| 트랙 | 킬 기준 |
|---|---|
| persistent (Task 21) | `c_barrier > 0.384 µs` → 영구 종결 |
| FlatXS device resolve (12b) | 착수 5인일 내 16좌표형 ≤4 ULP 미달 → 12a 영구 유지 |
| GPU CRAM (16/17) | CPU 대비 1.5× 미만 → CPU 잔류 |
| 스케줄러 전체 | 오버헤드 > solver wall 3% → granularity 재조정, 채택 보류 |
| W2 residency 폭 | `dram_throughput_pct > 60` → 폭 축소 |

## 11.3 게이트·팀 (불변)

**검증 클래스 게이트**

```text
B0 = bit-golden
     단일 500/500 · batch 708/708 · outer/sweep/BiCG 카운트 일치
     + fallback 0 + graph miss 0 + 총 인스턴스화 wall ≤ 250 ms   ← Rev.7.1 신설
N1 = run-to-run bit identity (sm_120) + Gate A + Gate B + v3 baseline 1회 동결
     + (Anderson) 후보 수용률 ±10%p                              ← Rev.7.1 신설
A2 = Gate A/B + 별도 브랜치 + 명시적 롤백 + production residual polish
```

**스케줄러 채택 지표**

```text
주지표: cases/h
동반 필수: tail_efficiency · phase queue p50/p90 · refill latency · GPU idle gap
제약 32: classify/compact/dispatch/refill 총합 ≤ solver wall 3%
```

**측정 프로토콜**

```text
서버 238 GPU0 단독. 다른 GPU/서버 결과는 참고값.
warm-up 1회 제외, hot run arm당 ≥3회, balanced order, median 보고.
GPU clock · temperature · power 동반 기록.
A/B arm은 RASBERY_XE_ANDERSON / RASBERY_IO_WRITER / precision / graph /
compaction / RASBERY_PPR_ITERS 를 전부 명시 (제약 5).
```

**역할 분담**

| 역할 | 담당 |
|---|---|
| 오케스트레이션 | Fable |
| 코드 · 리뷰(별도 인스턴스) · probe · 분석 · 문서 | Opus medium |
| 배포 · 실행 · 수신증 · 감시 · 문헌 | Sonnet extra high |

---

# 12. Detailed Implementation Tasks

Rev.7 §12의 태스크 본문(Files / Interfaces / Step 체크박스 / commit 블록)은 아래에 명시한 수정을 제외하고 **전부 Rev.7 유지**한다.

## Task 0: Freeze v2 Baseline, Scheduler Trace, and Cross-Backend Capability Inputs

**[Rev.7.1 수정]** — Step 3종 추가, Step 5(HIP/SYCL probe) 후행 이관.

- [ ] **Step 1–4: Rev.7 유지** (수신증 파서 / 기준선 러너 / phase trace 캡처 / CUDA capability probe)
- [ ] **Step 5: HIP/SYCL probe → 후행 이관.** 하드웨어 미보유 상태에서 `not_built`만 기록하는 코드는 지금 쓰지 않는다. Task 24/25와 함께 작성한다.

- [ ] **Step 2b (신설): provenance / A/B arm 분리**

기준선 러너의 arm을 명시적으로 분리해 기록한다. 모드 기본값 차이를 성능 차이로 오인하지 않기 위함이다(제약 5).

```text
provenance arm : 재현 검증용. 고정 시드/고정 순서. 성능 비교에 쓰지 않는다.
A/B arm        : 성능 비교용. RASBERY_XE_ANDERSON / RASBERY_IO_WRITER /
                 precision / graph / compaction / RASBERY_PPR_ITERS 를 전부 명시.
두 arm의 수신증을 별도 파일로 산출하고, 교차 인용을 계약 테스트가 차단한다.
```

- [ ] **Step 7b (신설): escape 비율 4종 실측**

```text
FluxLimitCycleSample / FluxStallFatal / NegativeFlux / RayleighFallback
각각의 발생률을 단일·M64에서 측정한다.
→ device 상태기계(Task 9)가 어느 경로를 hot path로 최적화해야 하는지 결정한다.
→ 발생률 0인 경로는 conditional graph에서 cold branch로 배치한다.
```

- [ ] **Step 9 (신설): 폭별 L2 / DRAM 측정 (W0 스파이크 ④)**

```bash
for W in 1 8 16 32 64; do
  RASBERY_GPU_L2_PROBE=1 ncu --metrics \
    lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./build/RASBERY --batch-mode $W ...
done
```

산출: `l2_hit_rate(W)`, `dram_throughput_pct(W)`.
**판정: `_ref_micx` 스트리밍 포함 시 CMFD 핫셋의 L2 상주가 깨지는 폭 W\*를 찾는다.** W2의 슬롯 폭은 `W < W*` 및 `dram_throughput_pct ≤ 60`을 만족해야 한다(§3.6).

- [ ] **Step 10 (신설): trace replay 도구 (§8.8 W0 승격)**

`tools/replay_phase_scheduler.py`를 Task 3이 아니라 **여기서** 작성하고 기존 M64 trace로 5개 정책을 재생한다. 산출된 idle/tail 예측치가 W1 착수의 전제다.

## Task 1: Define Backend-Neutral GPU Runtime, Physics Types, and Stub Parity

**[Rev.7.1 수정]** — 구조체 4분할 반영, 제약 35 계약 테스트 추가.

- [ ] **Step 2b (신설): 제약 35 grep 계약 테스트**

`tools/test_pure_body_portability.py`가 공용 pure body 헤더에서 CUDA 전용 intrinsic을 검출하면 fail한다.

```text
검사 대상: src/*Kernel.h, src/GpuPhaseSchedulerCore.h, src/GpuPhysicsArenaLayout.h
금지 토큰: __shfl, __ballot, __activemask, __syncwarp, atomicAdd_block,
          __fmaf_rn, __ldg, __double2hiint, cuda*, cooperative_groups
허용: GpuSubgroup.h / GpuBackend.h 래퍼 경유 호출
```

- [ ] **Step 4 (교체): 물리 상태 타입 — 4분할**

Rev.7의 단일 `DeviceCaseControl` static_assert를 다음으로 **교체**한다.

```cpp
static_assert(sizeof(DeviceSlotPhase) == 32);
static_assert(alignof(DeviceSlotPhase) == 32);
static_assert(std::is_trivially_copyable_v<DeviceSlotPhase>);

static_assert(alignof(DeviceSlotState) == 128);
static_assert(sizeof(DeviceSlotState) % 128 == 0);
static_assert(std::is_trivially_copyable_v<DeviceSlotState>);

static_assert(std::is_trivially_copyable_v<DeviceSearchState>);
static_assert(std::is_trivially_copyable_v<DeviceScheduleParams>);
```

- [ ] **Step 4b (신설): 필드 완전성 계약**

`DeviceSearchState`가 `Scheduler.h:55-61` + `155-177`의 **모든** 필드를 갖는지, `DeviceScheduleParams`가 `Scheduler.h:127-148`의 모든 검색 파라미터를 갖는지 소스 대조로 검증한다. 필드 누락은 fail이다(수정 4의 재발 방지).

- **Files 추가:** `src/GpuSlotControl.h`, `tools/test_pure_body_portability.py`.

## Task 2: Implement `GpuPhysicsArena` Facade and CUDA Fixed-Address Storage

**[Rev.7.1 수정]** — 압축 tier 제거, phase 키 aliasing.

- [ ] **Step 1 (개정): layout 테스트** — scratch lifetime aliasing 검증을 **슬롯별 phase 키** 기준으로 바꾼다(§4.2). 전역 phase 기준 검증은 폐기한다.
- [ ] **Step 5 (개정): memory receipt** — `per_slot_bytes` 기대값은 **200–225 MiB**다. 22–35 MiB 압축 tier를 기대하는 assertion은 삭제한다.
- [ ] **Step 4 (개정): HIP/SYCL allocation hook** — 인터페이스만 확정하고 `unavailable`로 컴파일. Rev.7 유지이나 Task 24/25가 후행임을 명시한다.

## Task 3: Monte-Carlo-Inspired Stable Case-Phase Queues and Scheduler Core

**[Rev.7.1 수정]** — 계약 2종 추가, replay 도구는 Task 0으로 이관.

- [ ] **Step 6 (개정): phase quantum table contract + 폴백 조항**

```text
CMFD    = one sweep  (기본)  |  "잔여 sweep 예산 전체" (폴백, §5.1(2))
Nodal   = full phase chain (trl0→trl12→MatEven→Jnet)
Xe      = one step
TH      = one step
Search  = one trial
Depletion = predictor / corrector stage
PPR     = one corner-balance iteration   ← 신설 (§6.22)
```

W0 스파이크 ③의 `c_cond` 실측값으로 CMFD 모드를 확정하고, 두 모드 모두 계약 테스트를 통과해야 한다.

- [ ] **Step 6b (신설): 전이표 ↔ host trace 계약 테스트**

`tools/test_phase_transition_vs_host_trace.py`. §6.21의 검증 대상 deck 8종 전부에서 phase 시퀀스가 host trace와 일치해야 한다. **W1 종료 게이트.**

- [ ] **Step 7 (이관): trace replay 시뮬레이터 → Task 0 Step 10.** 여기서는 Task 0이 산출한 예측치를 계약 입력으로 소비만 한다.

## Task 4: GPU Nodal `updateConstant`

**[Rev.7.1 수정]** — B0 구제 스파이크 선행.

- [ ] **Step 0 (신설, 1인일): B0 구제 스파이크**

실덱에서 `sqrt`/`exp` 인자 범위를 수집하고 CUDA 대비 glibc ULP를 측정한다. 0 ULP면 §6.1에 따라 **B0로 구제**하고 Step 5의 Gate A/B·v3 동결을 생략한다. 실패 시 Rev.7 경로(N1) 그대로 진행한다.

- [ ] **Step 5 (조건부): trajectory-changing 게이트** — Step 0이 B0 구제에 성공하면 이 Step은 bit-golden으로 대체된다.

## Task 5: GPU CMFD Pre/Post Kernels

**[Rev.7.1 수정]** — 수렴/stall 커널의 상태 필드 명시.

- [ ] **Step 5 (개정): 수렴/stall 커널** — 복제 대상에 다음을 **명시**한다.

```text
flux_stall, stall_events, stall_sample_taken, clean_iters, xe_interim_count
+ search limit-cycle sample 소비 조건
+ fatal exit (FluxStallFatal)
```

`clean_iters`와 `stall_sample_taken`은 Rev.7이 언급하지 않았으나 host 상태기계가 실제로 유지하는 값이며, 누락 시 stall 경로에서 궤적이 갈린다.

## Task 6: Single-Instance Resident CMFD Without Batch Mode

**[Rev.7.1 수정]** — unroll 그래프 키 제거를 **완결 필수**로 승격.

- [ ] **Step 3 (개정): `nmax`, unroll, eps를 device 스칼라로 — 잔존 확인 필수**

`src/CudaBICGBackend.cu:2744`에 `sweep_graph_unroll != unroll` 조건이 **아직 그래프 캐시 키에 남아 있다**.

```cpp
// CudaBICGBackend.cu:2741-2744 (현재)
if (sweep_graph_exec == nullptr || sweep_graph_nmax != nmax ||
    sweep_graph_unroll != unroll ||              // ← 제거 대상
    sweep_graph_precision != precisionTag()) {
```

unroll이 키에 남아 있으면 unroll 값이 바뀔 때마다 재인스턴스화가 발생해 `graph_reinstantiations`가 증가하고 §12 Task 10의 인스턴스화 게이트를 통과할 수 없다.

```text
완결 조건:
    1. unroll을 device 스칼라로 이동
    2. sweep_graph_unroll 필드 및 위 비교 조건 삭제
    3. 계약 테스트: unroll 값을 바꿔가며 실행해도 graph_reinstantiations 증가 0
    4. nmax도 동일하게 처리 (Step 4의 nested device sweep loop 또는 fixed max sweep graph)
```

## Task 7: Canonical CMFD–Nodal Device State

**[Rev.7.1 수정]** — L2 게이트 추가.

- [ ] **Step 6b (신설): L2 게이트**

canonical 상태 통합 후 폭별 `l2_hit_rate` / `dram_throughput_pct`를 재측정해 Task 0 Step 9의 기준선과 비교한다.

```text
게이트: dram_throughput_pct ≤ 60 (peak 대비)
       l2_hit_rate 가 통합 전 대비 회귀하지 않을 것
실패 시: _ref_micx 접근을 phase 단위로 격리하거나 슬롯 폭을 축소한다.
```

## Task 8: Nodal Phase Graph and Active-Slot Compaction

`Rev.7 Task 8 유지.`

## Task 9: Full Device Outer Control State Machine

**[Rev.7.1 수정]** — 확장 상태 사용, interim-Xe 봉쇄 삭제, M1 게이트 명시.

- [ ] **Step 2/3 (개정)**: device 스칼라 제어 커널과 카운터는 **§3.2(B) `DeviceSlotState` 전 필드**를 사용한다. Rev.7의 축소된 `DeviceSlotControl` 기준 구현은 상태 누락으로 fail한다.
- [ ] **Step 4 (삭제): ~~`InterimXeReady` disabled contract~~**

```text
Rev.7:   GPU-full v1은 RASBERY_XE_INTERIM_L2>0 을 거부한다.
Rev.7.1: 이 봉쇄를 삭제한다. W3.6(Task 13a)이 interim-Xe device 발화를
         productionize하며, 봉쇄가 남아 있으면 A2 트랙이 착수 불가다.
         대신 interim-Xe는 별도 A2 arm으로 게이트한다(§9.1 Class A2).
```

- [ ] **Step 1 (개정): 상태기계 참조 테스트** — 재현 대상에 **DERIVATIVE k 붕괴 재수렴 1회**(Driver.h:2268-2274)를 추가한다. 재시도 횟수가 1회로 제한되는 것까지 재현해야 한다.
- [ ] **Step 5b (신설): M1 판정** — `SolveLoop` 진입/이탈 구간에서 `host_numeric_calls == 0`. **W3 종료 게이트의 절반.**

## Task 10: CUDA Conditional Case-Phase Scheduler Graph

**[Rev.7.1 수정]** — 인스턴스화 게이트 2종 신설.

- [ ] **Step 3b (신설): 인스턴스화 규모 게이트**

14개 phase queue × 9버킷(1/2/4/8/16/24/32/48/64) child graph는 최대 126개 child다. 인스턴스화 비용이 발산하면 hot-path allocation 금지(제약 9)를 형식적으로만 지키게 된다.

```text
게이트 1: 총 인스턴스화 wall ≤ 250 ms   (graph_instantiation_wall_ms)
게이트 2: warm-up 이후 graph miss = 0    (graph_warmup_miss)
측정: 프로세스 시작~첫 상태점 완료 구간에서 cudaGraphInstantiate 누적 시간
실패 시: 버킷 집합을 {1,8,32,64} 4종으로 축소하거나 phase별 child를 지연 생성한다.
```

이 두 지표는 §9.3 수신증에 기록되며 B0 게이트의 일부다(§9.1).

- [ ] **Step 8 (개정): host observation 감소 측정** — 목표를 `host 수치 0`(M1) + `host observation 감소 ≥95%`로 명시한다.
- [ ] **Step 8b (신설): trace replay 예측 대조** — W0 Task 0 Step 10이 산출한 idle/tail 예측치와 실측을 대조한다. 편차가 크면 예측 모델을 폐기하고 실측 기반으로 정책을 재선택한다. **W3 종료 게이트의 나머지 절반.**

## Task 11: GPU Rod Cusping and Fractional-Rod Nodal Path

**[Rev.7.1 수정]** — **W5+ 재판정 대상.** 본문 Rev.7 유지. 단 Step 3의 커널 매핑에 `RodOp` phase(Driver.h:2252-2255) 연결을 추가한다(M2 요건, §9.2).

## Task 12a: Flat XS — Host Resolve + Device Apply — **[Rev.7.1 신설/분리]**

**Files:**
- Create: `src/CudaFlatXsApply.h`
- Modify: `src/FlatXsKernel.h`, `src/XSSet.cpp`, `src/GpuPhysicsArenaCuda.cu`
- Create: `tools/test_flatxs_apply_gpu_contract.py`

- [ ] **Step 1: host resolve 경로를 명시적 인터페이스로 고정**

`XSSet.cpp:2295-2325`의 좌표 해석 + `ResolveSpectralHistoryDeltas` + partner 경로를 그대로 유지하되, 산출물을 `DeltaStream` 구조체로 표준화한다.

- [ ] **Step 2: device apply 커널**

```text
one CTA = one node, block 64
shared: node-local lmp/mic workspace
thread 0이 stream header를 순서대로 진행, element update는 CTA 병렬
→ 기존 FlatXsKernel.h body 승계 (bit-gated)
```

- [ ] **Step 3: `flatxs_host_resolve_calls` 계측 추가** (§9.3)
- [ ] **Step 4: B0 bit-golden** — 기존 경로와 bit-identical.
- [ ] **Step 5: Commit**

```bash
git add src/CudaFlatXsApply.h src/FlatXsKernel.h src/XSSet.cpp \
        src/GpuPhysicsArenaCuda.cu tools/test_flatxs_apply_gpu_contract.py
git commit -m "perf: apply resolved flat XS deltas on GPU (12a)"
```

## Task 12b: Flat XS — Device Resolve — **[Rev.7.1 신설/분리, 후행·킬기준부]**

**Files:** Rev.7 Task 12의 `src/CudaFlatXsResolver.h`, `test/flatxs_resolver_gpu_replay.cpp`, `tools/test_flatxs_full_gpu_contract.py`.

- [ ] **Step 1: 16개 좌표형 열거와 replay 캡처**

resolver는 16개 좌표형 ~290줄이다. 각 좌표형에 대해 입력/중간 workspace/출력 delta를 캡처한다.

- [ ] **Step 2: 순서 의존 재현**

`XSSet.cpp:2308` 경로 — **부분 적용된 workspace 위에서 다음 좌표를 해석**하는 순서를 device에서 그대로 재현한다. one thread = one node, 노드 내부는 완전 직렬.

- [ ] **Step 3: 킬 기준 판정 (착수 5인일 시점)**

```text
통과: 16개 좌표형 전부 ≤4 ULP
실패: 12b 폐기, 12a 영구 유지, RASBERY_GPU_FLATXS_RESOLVE=device 경로 삭제
```

- [ ] **Step 4: N1 게이트** (`log()` 사용) — Gate A/B + v3 동결.
- [ ] **Step 5 (삭제): ~~per-slot microscopic XS 제거~~**

`_ref_micx`는 제거하지 않는다(§3.6). Rev.7 Task 12 Step 4의 "Replace with shared library and on-the-fly node condensation"과 Step 5의 "per-slot bytes decreases" 게이트는 **폐기**한다.

## Task 13: GPU Xe Evaluate, Anderson, and Commit

**[Rev.7.1 수정]** — §6.15의 4건을 Step으로 반영.

- [ ] **Step 4 (교체): ~~slot-warp Anderson algebra~~ → `k_xe_dot_reduce` + 슬롯당 1스레드 계수 계산**

```text
k_xe_dot_reduce_stage1  고정 chunk 256, gridDim.x 불변
k_xe_dot_reduce_stage2  one thread/slot, strict fold
k_xe_anderson_coeff     one thread/slot, depth 2/3 소규모 dense solve
depth 2 우선, depth 3은 별도 플래그 (Rev.7 유지)
```

- [ ] **Step 5 (개정): device candidate/Picard 이중 버퍼 — 기각은 감쇠 Picard**

```text
accept → commit(cand)
reject → commit(x_k + xe_relax * (F(x_k) - x_k))    ← Driver.h:1442-1444
두 이미지 모두 device 버퍼 보존, device-side swap. host rollback 없음.
```

- [ ] **Step 5b (신설): history reset 엣지 4종**

TH→Xenon / Search→Xenon / Material→Xenon / Depletion 진입에서 `xe_aa_ncol=0, xe_aa_have_prev=0`. **replay가 CPU와 reset 횟수 일치를 요구한다**(`xe_aa_reset_events`, §9.3).

- [ ] **Step 5c (신설): N1 게이트 — Anderson 수용률**

`xe_aa_accept_rate`가 v2 대비 **±10%p** 이내여야 통과. 밖이면 알고리즘이 바뀐 것으로 간주하고 fail.

- [ ] **Step 2 (개정): replay 요구사항** — 제안/수용/기각 횟수 + **reset 횟수** + 최종 v2 출력 일치.
- [ ] **Step 6 (이관): interim-Xe escape/phase → Task 13a (W3.6).**

## Task 13a: Interim-Xe Device Productionize (A2) — **[Rev.7.1 신설]**

**Files:**
- Modify: `src/CudaXeKernels.h`, `src/Driver.h`, `src/CudaPhaseScheduler.cu`
- Create: `tools/test_interim_xe_a2_contract.py`
- Create: `docs/A2_INTERIM_XE_20260828_KO.md`

**목적:** `RASBERY_XE_INTERIM_L2` loose-flux Xe step을 device에서 발화시켜 상태점당 outer 수를 줄인다.

- [ ] **Step 1: Task 9 Step 4의 봉쇄가 삭제되었는지 확인** (선행 의존)
- [ ] **Step 2: interim 발화 조건을 device 스칼라로**

`DeviceSlotState.xe_interim_count`와 flux L2 임계를 device에서 판정한다. host 관측 없음.

- [ ] **Step 3: A2 arm 분리** — 별도 브랜치, `RASBERY_XE_INTERIM_L2` 값이 A/B arm 기록에 포함(제약 5).
- [ ] **Step 4: Gate A/B + outer 감축 측정**

```text
측정: 상태점당 평균 outer (v2 기준 ~300)
목표: Task 13b와 합쳐 100
게이트: Gate A(v2 baseline) + Gate B(MASTER) 동시 통과, v3 동결, 명시적 롤백 경로
```

- [ ] **Step 5: Commit** (별도 브랜치)

## Task 13b: Flux-Space Anderson (A2) — **[Rev.7.1 신설]**

**Files:**
- Create: `src/CudaFluxAndersonKernels.h`, `tools/spike_flux_anderson.cu`
- Modify: `src/CudaOuterGraph.cu`
- Create: `tools/test_flux_anderson_a2_contract.py`
- Create: `docs/A2_FLUX_ANDERSON_20260828_KO.md`

**목적:** Xe 공간이 아니라 **flux 공간**에 Anderson 가속을 적용해 outer 반복 자체를 줄인다(nTRACER incomplete Anderson, §15 참고문헌 5).

- [ ] **Step 1: 스파이크 (3인일)**

기존 Xe Anderson safeguard 인프라(trust cap, positivity, condition number, 수용/기각 통계)를 **그대로 재사용**해 flux 벡터에 적용한다. depth 2로 시작.

```text
스파이크 판정: outer 수가 v2 대비 25% 이상 감소하고 발산 케이스 0이면 GO
```

- [ ] **Step 2: `k_flux_dot_reduce` 재사용** — §6.15(1)의 고정분할 내적을 flux 벡터(2×nxyz)로 재사용한다. 새 reduce를 만들지 않는다.
- [ ] **Step 3: safeguard 이식** — positivity / finite / trust cap / condition number / 기각 시 감쇠 복귀.
- [ ] **Step 4: Gate A/B + v3 동결 + 롤백**
- [ ] **Step 5: 결합 측정** — Task 13a와 함께 **상태점당 outer 300 → 100** 달성 여부를 판정한다. 이것이 단일 2× 경로의 필수 조건이다(§13).
- [ ] **Step 6: Commit** (별도 브랜치)

## Task 14: GPU Thermal-Hydraulics

**[Rev.7.1 수정]** — **W5+ 재판정 대상.** 본문 Rev.7 유지, 3건 개정(§6.16).

- [ ] **Step 2 (교체): ~~fixed partition 2-stage reduction~~ → strict serial fold** (one thread = one slot, 노드 오름차순 직렬 누적). B0 유지의 필수 조건.
- [ ] **Step 1b (신설): burnup 브래킷 정수 키 검증** — replay에 브래킷 인덱스 비교를 포함한다.
- [ ] **Step 5 (개정): 게이트 완화** — 테이블 bilinear 보간이 B0(form mask 보존)이면 **Gate A/B와 v3 동결이 불필요**하다. Step 1 replay가 0 ULP를 확인하면 bit-golden으로 대체한다.
- [ ] **Step 6b (신설): `Derivative` phase 연결** — `UpdateDerivative`(Driver.h:2245)의 Δtful/Δtmod/Δdmod 적용을 이 태스크가 담당한다(M2 요건).

## Task 15: GPU Critical Boron and Rod Search

**[Rev.7.1 수정]** — **W5+ 재판정 대상.** 매핑과 분류를 교체(§6.17).

- [ ] **Step 2 (교체): ~~one-warp-per-slot search~~ → one-thread-per-slot search**

```text
one thread = one slot, block 128
스레드가 DeviceSearchState 26필드(§3.2(C))를 읽어 bracket/secant/relaxation 결정
trial history 배열은 존재하지 않는다. 32레인 스캔 매핑을 삭제한다.
```

- [ ] **Step 5 (개정): 분류 B0** — 검색 산법은 비교·사칙연산·secant 나눗셈뿐이다. `min_secant_denom`(1e-12) 가드와 나눗셈 순서를 보존하면 bit-identical이다. Gate A/B·v3 동결 불필요.
- [ ] **Step 1 (개정): capture 대상** — `SearchMemory`(Scheduler.h:55-61)의 상태점 간 이월과 `search_exit_status/dk/tol`(174-177)을 포함한다.
- [ ] **Step 6b (신설): `Derivative` phase 연결** — Δbppm 적용(Driver.h:2245).

## Task 16: Batched GPU CRAM Depletion

**[Rev.7.1 수정 — 전면]** — **W5+ 재판정 대상.** §6.18의 재작성판으로 교체한다.

- [ ] **Step 1 (개정): 노드-국소 CRAM 시스템 덤프**

캡처 대상: 희소 패턴(`base_cols`/`row_offsets`), 노드별 `base_vals`/`base_diag`, 각 pole의 `rhs`/`x`/반복 횟수/최종 잔차, `accum`, 최종 밀도. **전 nxyz 8,451노드 중 대표 노드 + 수렴이 느린 노드**를 반드시 포함한다.

- [ ] **Step 3 (교체): ~~compatibility CRAM order 8 (dense LU)~~ → 노드당 1스레드 희소 Gauss-Seidel**

```text
매핑: one thread = one node, block 128, grid = ceil(nxyz/128) = 67
상수: alpha0=1.1722341374385704e-08, pole_count=4, max_iter=64,
      rel_tol=1e-13, abs_tol=1e-28, diag_tol=1e-30, matrix_sgn=-1.0,
      alpha[4]/theta[4]  (milk.h:1676-1701 상수 그대로)
first: 3  (= iI135, XSSet.cpp:4008)
대상: 전 nxyz 노드 (XSSet.cpp:4036) — fuel node 한정 아님
메모리: 희소 패턴은 constant/shared 공유, 노드별 복소 배열은 전역 SoA
       (sm_120 shared = 128 KB/SM; dense 39×39 shared 전제는 폐기)
순서: pole 0→3, row first→n-1 in-place GS, cols 저장 순서 직렬 누적,
      max-norm 잔차, accum pole 순서 고정
```

- [ ] **Step 3b (신설): 분류 판정** — replay가 0 ULP면 **B0**, 아니면 N1로 강등하고 Gate A/B·v3 동결을 요구한다. 복소 나눗셈 구현 차이(Smith vs naive)가 주된 위험이다.
- [ ] **Step 4 (개정): 가드 → escape 코드**

```text
diag 크기 ≤ diag_tol      → DeviceEscape::CramZeroDiagonal + 슬롯 Failed
max_iter 소진 미수렴       → DeviceEscape::CramNotConverged + 슬롯 Failed
nonfinite / 음수 밀도      → 기존 가드 (|v|<1e-12 && v<0 → 0)
실패 노드는 자기 슬롯만 실패시킨다. 다른 슬롯은 계속 진행.
```

- [ ] **Step 5b (신설): 킬 기준 판정** — GPU CRAM이 **CPU 대비 1.5× 미만**이면 채택하지 않고 CPU에 잔류시킨다. 8,451노드 × 4극 × ≤64반복의 직렬 GS는 GPU에 유리한 형상이 아니다.

## Task 17: GPU Predictor, Corrector, Burnup, and Inventory State

**[Rev.7.1 수정]** — **W5+ 재판정 대상.** BOS 스냅샷 정책 정정.

- [ ] **Step 2 (교체): ~~BOS microscopic 복사 제거~~ → 11 → 4 슬롯 축소**

```text
보존: BOS isotope density / BOS flux / BOS condensed reaction-rate / BOS burnup
정책: RASBERY_PC_MODE=decart 를 v1 필수로 고정
메모리: 약 24 MiB/slot (§3.6). "제거"가 아니다.
```

- [ ] **Step 4 (개정): burnup 슬롯 총계 = strict serial fold** (§6.19).

## Task 18: Full Device Case-Phase Scheduler

**[Rev.7.1 수정]** — **W5+ 재판정 대상.** 본문 Rev.7 유지. 단 Step 1의 참조 trace 목록에 **PPR / NormalizeFluxSign / Derivative 재수렴 / RodOp / ResultAggregate**를 포함하고, Step 8 수신증에 §9.3 신규 필드를 추가한다.

## Task 19: GPU Output Packing and CPU Writer-Only I/O

**[Rev.7.1 수정]** — **W5+ 재판정 대상.** 분류 N1, phase 2종 담당 추가.

- [ ] **Step 1b (신설): fold 순서 B0 판정** — 축방향/조립체 누적을 strict serial fold로 고정할 수 있는지 먼저 판정한다. 가능하면 B0로 구제, 불가능하면 N1(§6.20).
- [ ] **Step 5 (개정): 게이트** — N1일 경우 run-to-run bit identity + Gate A/B + v3 동결. AO 재배열 1.2e-16 실측이 근거다.
- [ ] **Step 6b (신설): `NormalizeFluxSign` / `ResultAggregate` phase 구현** — Driver.h:2235/2239/2261/2274 및 2308을 device phase로 이관한다(M2 요건, §9.2).

## Task 19b: GPU Pin Power Reconstruction (PPR) — **[Rev.7.1 신설]**

**Files:**
- Create: `src/CudaPprKernels.h`
- Create: `test/ppr_gpu_replay.cpp`
- Modify: `src/PPR.h`, `src/PPR.cpp`, `src/Driver.h`, `src/GpuPhysicsArenaCuda.cu`
- Create: `tools/test_gpu_ppr_contract.py`

**Interfaces:**

```cpp
void enqueuePprIteration(
    DeviceGeometryView,
    DevicePhaseQueue ppr_slots,
    DeviceSlotView,
    GpuStream);

void enqueuePprReconstruct(
    DevicePhaseQueue, DeviceSlotView, bool use_quadrature, bool reconstruct_flux, GpuStream);
```

**근거:** Rev.7에 PPR이 0회 등장한다. `PPR.cpp` 1,099줄, `Driver.h:2278-2295`, **단일 wall의 6~10%**. M2 달성의 필수 항목이며 W5+ 중 **조기 승격 1순위**다.

- [ ] **Step 1: pure body 추출**

`updateFused`(PPR.cpp:285), `updateAxialLeakage`(:579), `updateSource`(:630), `updateCorner`(:675), `phig`(:705), `jnetDir/jnetX/jnetY`(:749-763), `reconstructPinPower`(:767)을 host/device 공용 body로 분리한다. `buildQuadratureTable`(:61)과 `buildStencil`(:139)은 초기화 1회이므로 host 유지.

- [ ] **Step 2: CPU replay 캡처**

BOC/중기/EOC, 봉 삽입, 재시작 케이스에서 반복별 corner flux, `_crdf`, 최종 핀출력을 캡처한다. `getJoutRed`(PPR.h:118)의 나눗셈 순서를 검증 대상에 포함한다.

- [ ] **Step 3: 커널 구현 (§6.22 표)**

```text
k_ppr_reset            one thread = (node, group)
k_ppr_axial_leakage    one thread = node
k_ppr_source           one thread = (node, group)
k_ppr_corner           one thread = corner
k_ppr_update_fused     one CTA = (node, group), 3x3 스텐실
k_ppr_reconstruct      one thread = pin, 9-point Gauss-Legendre (qpts[9] 순서 고정)
```

- [ ] **Step 4: `READY_PPR` quantum = corner-balance 1회**

100회 전체를 한 quantum으로 묶지 않는다(tail 재발). 미완료 시 requeue + iteration++.

- [ ] **Step 5: `RASBERY_PPR_ITERS` 계약**

device 경로에서도 동일 의미(기본 100, 비양수 → 100 fallback, Driver.h:2286-2290). 값이 A/B arm 기록에 포함되어야 한다(제약 5).

- [ ] **Step 6: 자체 게이트**

```text
B0 경로: fold/나눗셈 순서 보존 시 bit-identical
N1 경로: 핀출력 rms vs MASTER ≤ 0.84% (수렴값). 벗어나면 fail.
         (참고: cap 5 → 4.76% rms, 수렴 → 0.84%, 50~200회 동일)
계측: ppr_host_calls = 0 (§9.3)
```

- [ ] **Step 7: Commit**

```bash
git add src/CudaPprKernels.h test/ppr_gpu_replay.cpp \
        src/PPR.h src/PPR.cpp src/Driver.h src/GpuPhysicsArenaCuda.cu \
        tools/test_gpu_ppr_contract.py
git commit -m "feat: reconstruct pin power on GPU"
```

## Task 20: Immediate Device Slot Refill and Massive-Input Tail Control

**[Rev.7.1 수정]** — 테넌트 reset 대상 확장, `Import` phase 확장.

- [ ] **Step 4 (개정): full tenant reset** — §3.2 4구조체 전량 + mask/mirror/generation/graph status/Krylov flag. `k_audit_tenant_reset`이 non-reset 바이트를 검출하면 fatal.
- [ ] **Step 2b (신설): `Import` phase 확장** — `PrepareForStep`(Driver.h:2194)과 `ApplyToGeometry`(2195)를 device import 경로로 이관한다(M2 요건). `DeviceScheduleParams` 채우기를 포함한다.
- [ ] **Step 8 (개정): 1,280잡 안정성** — 수용 기준에 `stale_tenant_errors=0`, `queue_duplicate_errors=0`, `same_case_concurrency_errors=0`를 §9.3 수신증에서 직접 확인한다.

## Task 21: Optional PDL and Cooperative Persistent CMFD

**[Rev.7.1 수정]** — **"Optional" → W3.7 정규 트랙.** 제목의 Optional은 유지하되 wave 배치가 바뀐다.

- [ ] **Step 0 (신설, 선행 필수): W0 스파이크 ② 결과 확인**

```text
c_barrier(sm_120, 34~67 blocks) ≤ 0.384 µs  → 착수
c_barrier > 0.384 µs                        → 트랙 영구 종결, W3.7 생략
측정 절차: docs/PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md §2.6 Stage 0-E
```

- [ ] **Step 1 (개정): 전 residency 완료 후 재프로파일** — W2/W3 완료 시점의 dispatch 비율로 재계산한다. W0의 수치는 참고값이다.
- [ ] **Step 2 (유지): PDL 스파이크** — end-to-end median >1% 개선 시에만 채택.
- [ ] **Step 3 (유지): cooperative 반복 스파이크** — ≥7% 반복 개선 + bit identity + blocksPerSM ≥2 + 타 스트림 대기 ≤5% 회귀.
- [ ] **Step 4–7: Rev.7 유지.**

## Task 22: Validation, Baseline Freeze, Defaults, and PR

**[Rev.7.1 수정]** — 수신증 검증 목록 교체, M1/M2 구분.

- [ ] **Step 5 (교체): 최종 수신증 검증**

```text
M1 시점 (W3 종료):
    segment_gpu_numeric_fraction = 1
    host_numeric_allowlist = [<아직 이관 안 된 호출부 명시>]
    graph_instantiation_wall_ms <= 250
    graph_warmup_miss = 0
    dram_throughput_pct <= 60

M2 시점 (W5+ 완료 후):
    gpu_numeric_fraction = 1
    host_numeric_calls = 0
    host_numeric_allowlist = []          ← 비어 있어야 M2 주장 가능
    cpu_fallbacks = 0
    mid_iteration_h2d_bytes = 0
    mid_iteration_d2h_bytes = 0
    ppr_host_calls = 0
    flatxs_host_resolve_calls = 0
```

- [ ] **Step 4 (개정): 성능 측정** — M64 + 1,280잡에 더해 **`tail_efficiency`, phase queue p50/p90, refill latency, GPU idle gap**을 스케줄러 채택 판정 지표로 동반 기록한다(제약 32 검증).
- [ ] **Step 2 (개정)**: 4회 반복 캠페인은 warm-up 1회 제외 + hot ≥3회 median(제약 12).

## Task 23: Finalize Runtime Capability Selection and Backend Contract Tests

**[Rev.7.1 수정]** — 범위를 CUDA 두 tier로 축소.

- [ ] **Step 1 (개정): capability-tier 테스트 범위**

```text
W4-lite에서 구현/검증: CudaConditional, CudaEpoch, Unsupported,
                       insufficient FP64, insufficient VRAM, missing graph
후행 (Task 24/25 시): HipEpoch, SyclEpoch
```

HIP/SYCL 분기는 셀렉션 테이블에 남기되 `not_built`로 반환하며, 이는 PASS가 아니다.

- [ ] **Step 2–7: Rev.7 유지.**

## Task 24: HIP/ROCm Epoch Scheduler Backend

**[Rev.7.1 수정]** — **후행 트랙. AMD 하드웨어 확보 전 착수 금지.** 본문 Rev.7 유지. 지금 적용되는 것은 제약 35(공용 pure body 규율)뿐이다.

## Task 25: SYCL/Level Zero Epoch Scheduler Backend

**[Rev.7.1 수정]** — **후행 트랙. Intel 하드웨어 확보 전 착수 금지.** 본문 Rev.7 유지. 단 **Step 2("No CUDA/HIP intrinsic may leak into shared headers")는 제약 35로 승격되어 지금부터 강제**된다.

## Task 26: Reactor Model Descriptor and Numerical Plugin Registry

**[Rev.7.1 수정]** — **축소판(26-lite)만 W4-lite에서 수행.**

```text
W4-lite 범위 (26-lite):
    Step 1  descriptor 호환성 테스트 (Cartesian2G만 validated, 나머지 명시적 미지원)
    Step 3  현재 SENM2GCartesian plugin 등록
    Step 5  출력 provenance: reactor descriptor hash / plugin name·version / support label
            을 모든 HDF5 및 result 수신증에 기록
후행:
    Step 2  전체 호환성 술어 (CartesianNG/HexNG/moving fuel 등)
    Step 4  missing-plugin 진단 전체
```

**근거:** 문서·경계 정책은 저비용이며, 잘못된 "지원" 표기를 막는 효과가 크다. 반면 전체 plugin 레지스트리는 속도 목표 기여가 0이다.

## Task 27: Generic Topology and Multigroup CMFD/Nodal Foundations

**[Rev.7.1 수정]** — **별도 캠페인.** 본 계획의 인일 산정에 포함하지 않는다. 본문 Rev.7 유지. 단 §8B.2의 "Cartesian 2G fast path를 느리게 만들지 않는다"는 **지금부터 지켜야 할 제약**이다.

## Task 28: Reactor-Family Validation Matrix and Support Labels

**[Rev.7.1 수정]** — **별도 캠페인.** 본문 Rev.7 유지.

---

# 13. Performance Model and Honest Targets

**[Rev.7.1 수정 — 전면 교체]** — Rev.7 §13은 Rev.6와 동일한 순수성-우선 한계를 갖는다(residency만으로 단일 33~37 s = MASTER 27.2 s에 **미달**). 승인된 정직 수치표로 교체한다.

모든 미래 값은 **시나리오 추정**이며 실측이 아니다.

## 13.1 단일덱 (MASTER 27.2 s 대비)

| 단계 | wall | vs MASTER | 필요 레버 |
|---|---:|---:|---|
| v2 (현재) | 55.4 s | 0.49× | — |
| 상주화 + 스케줄러 전체 | 33–37 s | 0.8× | W1–W3 |
| **+ persistent CMFD** | 21–25 s | 1.1–1.3× | W3.7 (c_barrier 게이트) |
| **+ A2 (300→100 outer/상태)** | **11–14 s** | **1.9–2.5×** | **W3.6 (Task 13a+13b)** |
| + MASTER 동급(outer 59) + PPR GPU + IO 중첩 | 6–12 s | 2.3–4.5× | W5+ 조기 승격 |

## 13.2 단일 GPU 배치 (M64 현재 216.10 c/h)

| 단계 | 1-GPU c/h | 20× 달성 GPU 수 |
|---|---:|---:|
| v2 (현재) | 216 | — |
| 상주화 + 스케줄러 | 420–650 | 8–11 |
| **+ A2 outer 감축** | **900–1,400** | **4–5** |
| + MASTER 동급(59 outer) | ~2,000 | **3** |

Monte Carlo 스케줄러 연구의 배수를 직접 곱하지 않는다. **스케줄러 단독 상한은 실측된 `GPU idle + empty bucket + rendezvous/tail + host observation` 비율의 Amdahl 천장**으로 계산한다. 스케줄러의 기여 = 배치 tail/기아 제거로 "폭 충전 1.35~1.7×"를 실현하는 실행 기반 + **Anderson 배치 재채택의 전제조건**.

## 13.3 MASTER W16 20× aggregate

```text
목표 = 20 × 217 = 4,340 cases/h
N = ceil(4340 / (one_gpu_throughput × scaling_efficiency))
```

```text
 650 c/h × 0.85 →  8 GPU
 900 c/h × 0.85 →  6 GPU
1400 c/h × 0.85 →  4 GPU
2000 c/h × 0.85 →  3 GPU
```

호스트 예산(4,340 c/h 기준): 라이터 ~8스레드 + GPU당 로더 8~16, NUMA 고정, 프로세스-per-GPU.

## 13.4 판정 요약

> **단일 2× 경로 확정 (persistent + A2 필수). 3~4× 도전권. 5×는 목표로 유지하되 약속하지 않는다** — sm_120의 FP64 1:64 구조 한계 때문이다.
> **배치 20×는 A2 완주 시 GPU 3~5장.**

**residency만으로는 MASTER에 미달한다**는 사실이 이 계획의 핵심 제약이다. W1–W3은 필요조건이지 충분조건이 아니며, 속도 목표는 **W3.6(A2)와 W3.7(persistent)** 두 레버가 결정한다.

---

# 14. Completion Definition

**[Rev.7.1 수정]** — M1/M2 분리, PPR·전이표 계약 추가.

## 14.1 M1 완료 (W3 종료)

1. 한 번의 flux 세그먼트(`SolveLoop` 1회) 내부에서 CPU numerical function이 실행되지 않는다.
2. Geometry, XS, isotope, CMFD, Nodal 상태의 canonical owner가 GPU다.
3. CMFD와 Nodal이 동일 `flux/jnet/phis/dhat` device address를 사용한다.
4. Device phase scheduler가 세그먼트 내 모든 phase를 전환한다.
5. **전이표 ↔ host trace 계약 테스트가 deck 8종 전부에서 PASS한다.**
6. `graph_instantiation_wall_ms ≤ 250`, `graph_warmup_miss = 0`.
7. `dram_throughput_pct ≤ 60`.
8. `segment_gpu_numeric_fraction = 1.0`, `host_numeric_allowlist`가 명시적으로 열거되어 있다.
9. W0 trace replay 예측치와 실측이 대조되어 문서화되었다.

## 14.2 M2 완료 (W5+ 완료 후)

M1의 9개 조건에 더해:

10. Input import 이후 output snapshot 이전에 CPU numerical function이 실행되지 않는다.
11. `updateConstant`, rod cusping, Xe, T/H, search, predictor/corrector, CRAM, burnup, **PPR**, output pack이 GPU에서 실행된다.
12. **`host_numeric_allowlist = []`** (빈 배열). `gpu_numeric_fraction = 1.0`, `host_numeric_calls = 0`.
13. `ppr_host_calls = 0`, `flatxs_host_resolve_calls = 0`.
14. Mid-iteration H2D/D2H = 0.
15. CPU fallback = 0이며 오류는 fail-closed다.
16. Full HDF5 output과 validation quantity가 GPU-packed snapshot에서 생성된다.
17. Gate A/B 및 v3 baseline freeze가 완료된다.
18. Server 238 single/M64/massive-batch receipt가 문서와 PR에 첨부된다.
19. Runtime rollback path와 no-CUDA build가 유지된다.

## 14.3 스케줄러 완료 (W4-lite 종료)

20. 완료된 case가 느린 case를 기다리지 않고 다음 phase 또는 새 input으로 이동한다.
21. `queue_duplicate_errors = 0`, `same_case_concurrency_errors = 0`, `stale_tenant_errors = 0`.
22. 스케줄러 오버헤드가 solver wall의 3% 이하다(제약 32).
23. `tail_efficiency`, phase queue p50/p90, refill latency, GPU idle gap이 수신증에 기록된다.

## 14.4 후행 트랙 완료 (본 계획 범위 밖)

24. CUDA optimized와 portable epoch backend가 동일 phase semantics를 통과한다 — **Task 24/25 완료 시**.
25. "모든 GPU 지원"은 capability manifest 통과 backend만, "모든 원자로 지원"은 family manifest가 `validated`인 조합만 의미한다 — **Task 27/28 완료 시**.
26. 현재 CUDA Cartesian 2G 완료와 cross-vendor/cross-reactor 완료를 **별도 milestone으로 보고**한다.

---

# 15. Research References

`Rev.7 §15 유지` (1–13번 전체). 다음 1건을 추가한다.

14. 본 저장소 내부 실측 문서: `docs/CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md`(도착 폭 기아 실측), `docs/PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md` §2.6(`c_barrier` 마이크로벤치 절차), `docs/IO_WRITER_THREAD_DESIGN_20260827_KO.md`(라이터 스레드 예산).

---

## 부록 A. Rev.7 대비 file:line 검증 목록

본 개정의 모든 코드 인용은 tip `fdef162`에서 재확인했다.

| 인용 | 위치 | 확인 내용 |
|---|---|---|
| CRAM 희소 GS | `include/milk.h:1649-1819` | `solveBatemanCRAM`, order 8 → pole_count=4, max_iter=64, rel_tol=1e-13 |
| CRAM 가드 | `include/milk.h:1732-1733, 1765, 1807` | abs_tol=1e-28, diag_tol=1e-30, 두 throw |
| CRAM 호출 | `src/XSSet.cpp:4008` | `CRAM_ORDER=8`, `first=iI135=3` (`XsReconKernel.h:46`) |
| depletion 대상 | `src/XSSet.cpp:4036` | `for (int l = 0; l < nxyz; ++l) DepleteNode(...)` — 전 노드 |
| FlatXS 순서 의존 | `src/XSSet.cpp:2295-2325` | 부분 적용 workspace 위 `ResolveSpectralHistoryDeltas` + partner 경로 |
| sweep_unroll 그래프 키 | `src/CudaBICGBackend.cu:2744` | `sweep_graph_unroll != unroll` 잔존 |
| SearchMemory | `src/Scheduler.h:55-61` | 5필드, 상태점 간 이월 |
| 검색 파라미터 | `src/Scheduler.h:127-148` | searchType + 21개 tolerance/limit |
| 검색 runtime 상태 | `src/Scheduler.h:155-177` | 17 + 4필드 |
| Xe damper/AA 상태 | `src/Driver.h:1434-1500` | `xe_relax`, `xe_interim_count`, `xe_no_progress`, `xe_total`, `xe_cap_charged` |
| XeAndersonState | `src/Driver.h:999-1020` | x/f/g/f_prev/g_prev/df[]/dg[]/cand, ncol, have_prev |
| AA history 무효화 | `src/Driver.h:1022-1024` | "map이 고정된 동안에만 의미" |
| host 경계 (M2) | `src/Driver.h:2194,2195,2200,2235,2239,2245,2252-2255,2261,2268-2274,2274,2278-2295,2308` | 전 호출부 확인 |
| PPR 반복 계약 | `src/Driver.h:2286-2290` | `RASBERY_PPR_ITERS` 기본 100, 비양수 → 100 |
| PPR 구현 | `src/PPR.cpp` (1,099줄) | reset:179, updateFused:285, drive:532, updateAxialLeakage:579, updateSource:630, updateCorner:675, reconstructPinPower:767 |
| `_ref_micx` | `src/XSSet.h:211` | `[(iso*ng+ig)*nxyz + l]`, device-resident |
| nxyz | `docs/GPU_RASBERY_METHODOLOGY_BENCHMARK_20260824_KO.md:24` | 313 × 27 = 8,451 |
