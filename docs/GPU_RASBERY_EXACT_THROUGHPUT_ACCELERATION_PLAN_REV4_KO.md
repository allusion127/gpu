# GPU RASBERY: Exact 처리량 가속 캠페인 계획

**개정:** Rev.4 — 3차 검토 반영 및 구현 전 계약 동결  
**기준일:** 2026-08-26  
**목표:** screening이나 feedback-pass 제한 없이 모든 입력을 full-exact CMFD–노달–CNCC–Xe/T·H–임계검색 경로로 계산하면서 단일 GPU 처리량을 단계적으로 높인다.

---

## 1. 배경과 현재 기준선

2026-08-25에 push된 관련 브랜치를 분석한 결과, persistent CMFD와 safeguarded Anderson을 표방한 브랜치에는 핵심 C++/CUDA 구현이 온전히 남아 있지 않았다. 따라서 커밋 메시지를 구현의 증거로 취급하지 않고, 전달물의 SHA-256, 기대 base SHA, 실제 diff와 테스트 결과를 통해 소스 provenance를 다시 확립한다.

현재 검증된 full-exact 처리량 기준은 다음과 같다.

| 항목 | 값 |
|---|---:|
| GPU RASBERY M64 exact 처리량 | 213.9 cases/h |
| MASTER W16 처리량 기준 | 217 cases/h |
| 현재 상대 처리량 | 약 99% |
| 2배 milestone | 430 cases/h |
| 5배 milestone | 1,100 cases/h |
| 10배 milestone | 2,200 cases/h |

단일 GPU 현실적 기대 범위는 **보수적 1.5–2.5배, 목표 3–5배, 공격적 5–10배**로 둔다. MASTER W16 대비 20배 이상은 단일 GPU만으로는 현재 근거가 부족하며, 단일 GPU 상한을 확인한 뒤 멀티GPU로 확장한다.

각 Phase 종료 후 다음 Amdahl 모델을 실측값으로 갱신한다.

```text
T_total =
    T_fixed
  + N_outer  * T_outer
  + N_xe     * T_xe
  + N_search * T_search
  + T_depletion
  + T_IO
  + T_startup
```

단계별 기대 효과를 단순 곱하지 않는다. 새 profile에서 다음 milestone을 달성할 이론적 상한이 부족하면 후속 구현의 우선순위를 재설정한다.

---

## 2. 전체 캠페인의 Exact-only 하드 계약

이번 캠페인의 모든 채택 벤치마크와 물리 검증은 full-exact 계산만 인정한다.

### 2.1 실행 시작 시 필수 검사

다음 조건을 process 시작과 benchmark harness 양쪽에서 검사한다.

```text
RASBERY_GA_FEEDBACK_PASSES == 0 또는 unset
RASBERY_BATCH_LIGHT_RESULT == 0 또는 unset
full HDF5 output 활성
screening/surrogate 경로 비활성
```

위 조건을 위반하면 계산을 시작하지 않고 실패한다. 단순 경고 후 계속 실행하는 동작은 허용하지 않는다.

### 2.2 모든 실행의 physics-mode receipt

각 실행 로그와 machine-readable receipt에 다음을 기록한다.

```json
{
  "physics_mode": "full_exact_nodal",
  "screening": false,
  "feedback_pass_limit": 0,
  "full_hdf5": true
}
```

Benchmark parser는 필드 누락이나 값 불일치가 있으면 해당 run을 무효 처리한다.

---

## 3. 공통 검증 규약

### 3.1 변경 유형별 게이트

#### 궤적 보존 변경

수치식과 반복 순서를 변경하지 않는 메모리 수명, 전송, 캐시, dispatch 변경은 기존 bit-golden을 통과해야 한다.

```text
단일: 500/500 datasets byte-identical
배치: 708/708 datasets byte-identical
feature-off 경로: 항상 byte-identical
```

#### 궤적 변경 기능

Adaptive inner solve, Anderson, Krylov recycling처럼 중간 반복 경로를 바꾸는 기능은 Gate A와 Gate B를 동시에 통과해야 한다.

### 3.2 Gate A — 기존 RASBERY exact baseline 비퇴행

최소 필수 기준은 다음과 같다.

```text
|keff_candidate - keff_baseline| ≤ 1 pcm
|CBC_candidate  - CBC_baseline | ≤ 1 ppm
최종 CMFD residual            ≤ 기존 production tolerance
최종 비선형 residual          ≤ 기존 production tolerance
모든 상태점 finite 및 converged
search/TH/Xe convergence status 비퇴행
```

다음 출력도 기준 binary와 비교한다.

```text
AO, Fq, Fr
node power
pin power
Xe/Sm inventory
동위원소 inventory
burnup 및 depletion 결과
```

MASTER 대비 핀 분포의 비퇴행도 Gate A에 명시적으로 포함한다.

```text
MASTER_pin_RMS(candidate)
    ≤ MASTER_pin_RMS(baseline) + delta_RMS

MASTER_pin_max(candidate)
    ≤ MASTER_pin_max(baseline) + delta_max
```

`delta_RMS`와 `delta_max`는 구현 전에 동결한 기준 파일에서 읽는다.

```text
test/reference/validation_thresholds_v1.json
```

### 3.3 Gate B — MASTER 비교 봉투

```text
|delta reactivity| ≤ 2 pcm
|delta CBC|        ≤ 15 ppm
pin RMS            ≤ 0.30%
BOC + 중간 연소도 + EOC 핀 분포 포함
```

Gate B 안에 들어오는 것만으로 Gate A를 대체할 수 없다. Candidate가 MASTER 봉투 안에 있더라도 기존 RASBERY보다 악화되면 채택하지 않는다.

### 3.4 기준선 동결과 결정론 검사

구현 전에 기준 binary를 동일 입력·동일 GPU에서 3–5회 실행한다.

```text
1. 물리 출력 반복 재현성 확인
2. 시간 변동 범위 측정
3. 수치 gate 동결
4. 이후 threshold 수정 금지
```

물리 출력이 반복 실행마다 달라지면 허용치를 넓히지 않는다. 별도의 결정론 결함으로 분류하여 먼저 해결한다.

Baseline provenance는 다음 파일에 보존한다.

```text
test/reference/validation_baseline_manifest_v1.json
```

필수 필드:

```text
commit SHA
compiler 및 version
Release flags
GPU UUID
GPU driver / CUDA version
입력 SHA-256
골든 HDF5 SHA-256
MASTER 비교 데이터 version
threshold file SHA-256
```

### 3.5 처리량 지표

모든 run은 두 지표를 함께 기록한다.

```text
end_to_end_cases_per_hour
    process 시작부터 library load, 모든 solve, full HDF5 출력,
    process 종료까지 포함하는 최종 채택 지표

solver_only_cases_per_hour
    solver 최적화 효과 귀속을 위한 분석 지표
```

### 3.6 벤치마크 규약

```text
warm-up 1회: 결과에서 제외
유효 hot run: arm당 최소 3회
실행 순서: balanced 교차 또는 randomised balanced order
보고: median + min/max, 필요 시 bootstrap CI
기록: GPU clock/power/temp, CPU load, 다른 GPU process, VRAM, pinned RAM
채택: 사전 정의된 median 개선 + 물리 gate 통과 + fallback 0
```

성능 및 배치 정확도 판정은 **238 GPU0 단독**에서만 수행한다. 181 서버는 MSVC 컴파일, stub link, 단일 비배치 회귀 전용이며 batch 성능·batch 정확도·N>64·MASTER 처리량 비교에는 사용하지 않는다.

---

## 4. Phase 0 — 유실 소스 복구와 Provenance `[GO]`

`git am`을 바로 실행하지 않는다.

```text
전달물 SHA-256 기록
→ 기대 base SHA 확인
→ 격리 worktree에 적용
→ git range-diff
→ "커밋 메시지 기능 ↔ 실제 수정 파일" 대응표 작성
→ CUDA/Stub 인터페이스 parity
→ 정밀 코드 리뷰
→ feature-off build 및 bit regression
```

Persistent CMFD와 Anderson은 하나의 전달물에 묶여 있어도 별도 커밋과 별도 A/B arm으로 분리한다.

전달물이 없거나 불완전한 경우:

```text
Anderson: Phase 4 명세대로 새로 구현
Persistent CMFD: Phase 5 착수 기준을 만족한 뒤 새로 설계
```

---

## 5. Phase 1 — jobs > 64 안정화: 64-slot Rolling Queue

### 5.1 목표와 범위

```text
jobs ≠ arena width
```

1차 목표는 다음과 같다.

```text
jobs = 96 / 128 / 256
arena width = 64
host workers ≤ 64
완료 Driver가 slot과 pin lease를 반납한 뒤 다음 입력 시작
```

이 Phase는 폭 확대가 아니라 메모리 수명과 queue 안정화 단계다. 기대 효과는 0–20%로 보고, 핵심 성공 기준은 정확성과 무퇴행이다.

우선순위가 매겨진 근본 원인 가설:

```text
H1: 영구 host registration과 Driver 조기 소멸의 aliasing
H2: output/restart/scratch namespace 충돌
H3: CPU oversubscription 및 library 직렬 load
```

### 5.2 Phase 1A — 패치 전 원인 판별

| Arm | jobs / width / workers | Pinning | 목적 |
|---|---|---|---|
| A | 64 / 64 / 64 | 기존 on | 검증 기준 |
| B | 96 / 64 / 64 | 기존 off(auto) | rolling queue 기준 |
| E0 | 96 / 64 / 64 | off | 동일 input + 고유 namespace 영향 |
| F | 96 / 64 / 32,48,64 | off | host worker 스큐 영향 |

추가 수집:

```text
FAIL 발생 위치와 batch 폭의 관계
HDF5 LOCK wait_ms
BATCH_HOST host_pinning receipt
slot acquire/refusal
Driver 생성·소멸 sequence
```

### 5.3 Phase 1B — HostPinLease 적용 후

| Arm | jobs / width / workers | Pinning | 목적 |
|---|---|---|---|
| C | 96 / 64 / 64 | lease auto/on | pin 수명 수정 효과 |
| E1 | 96 / 64 / 64 | lease auto/on | namespace + pin 결합 검증 |
| G | 128 / 64 / 64 | lease auto/on | rolling queue 확장 |

기존 `96 / 96 / 96` 실험은 width와 CPU oversubscription이 동시에 바뀌므로 Phase 6으로 이동한다.

---

## 6. HostPinLease 구현 계약 `[Phase 1B blocker]`

### 6.1 Registry key와 실제 CUDA registration의 분리

Conflict 검색에는 페이지 정규화 interval을 사용하지만, 실제 CUDA API 호출 주소와 크기는 별도로 보존한다.

```cpp
struct PinRecord {
    uintptr_t conflict_page_begin;
    uintptr_t conflict_page_end;

    void*     registered_address;
    size_t    registered_bytes;

    unsigned  owners;
    unsigned  in_flight;
};
```

`cudaHostUnregister`에는 최초 `cudaHostRegister`에 사용한 동일 주소를 전달한다. 정규화된 페이지 시작 주소를 임의로 등록하지 않는다.

### 6.2 Overlap 정책

#### 기존 record 재사용 허용

```text
requested interval == existing interval
requested interval ⊂ existing interval
```

위 경우 기존 record의 `owners`를 증가시킨다.

#### 재사용 금지

```text
existing interval ⊂ requested interval
부분 overlap
```

요청 범위가 기존 registration보다 넓거나 일부만 겹치면 기존 record를 자동 확장하거나 unregister/re-register하지 않는다.

```text
해당 요청만 pageable fallback
overlap_rejections++
strict/debug 모드에서는 오류
```

### 6.3 Eviction과 해제 조건

```text
owners == 0
in_flight == 0
모든 관련 stream event 완료
```

세 조건이 모두 충족될 때만 unregister 또는 stale eviction을 허용한다. 사용 중인 registration을 강제로 eviction하지 않는다.

### 6.4 Stream별 마지막 event

동일 stream의 여러 async copy는 stream 순서가 보장되므로 stream별 마지막 event만 관리한다.

```cpp
std::unordered_map<cudaStream_t, cudaEvent_t> last_event_by_stream;
```

해제 순서:

```text
마지막 owner release
→ 각 stream의 마지막 event query/synchronize
→ 모든 event 완료 확인
→ in_flight = 0
→ cudaHostUnregister
```

`cudaDeviceSynchronize()`는 금지한다. 초기 보수 구현에서는 Driver 소멸 직전 그 Driver가 소유한 backend stream만 명시적으로 drain한 뒤 lease를 해제할 수 있다.

### 6.5 Vector 재할당 계약

Lease는 다음을 보존한다.

```cpp
address
bytes
allocation_generation
```

주소 또는 capacity가 바뀌는 경우:

```text
기존 stream event 완료
→ 기존 lease release
→ vector resize/reallocation
→ 새 주소에 새 lease acquire
```

기본 전략은 핫패스 진입 전에 최대 크기로 버퍼를 확정하고 lease 생존 중 재할당을 금지하는 것이다.

감사 대상:

```text
_sweep_chif
_sweep_xsnf
_sweep_vol
Nodal/XS scratch vectors
milk::Vector 기반 배열
```

### 6.6 명시적 Registry 종료 순서

Function-local static destructor에 CUDA 정리를 맡기지 않는다.

```text
모든 Driver 종료
→ 모든 backend stream drain
→ rasberyReleaseBatchArena()
→ rasberyDrainPinnedRegistry()
→ CUDA context 종료
```

### 6.7 환경 설정

```text
RASBERY_HOST_PINNING=auto   # 기본, 안전 lease 계약에 따라 사용
RASBERY_HOST_PINNING=off    # pageable copy 강제
RASBERY_HOST_PINNING=force  # 개발 실험 전용, 강한 경고
```

Legacy permanent-registration 구현에서는 `force`를 거부한다.

### 6.8 Pinning receipt

```json
{
  "pinning_mode": "auto",
  "registered_ranges": 0,
  "registered_bytes": 0,
  "deduplicated_requests": 0,
  "pageable_fallbacks": 0,
  "unregistered_ranges": 0,
  "overlap_rejections": 0,
  "stale_evicted": 0
}
```

---

## 7. Job Namespace 분리 `[GO]`

정책:

```text
동일 input 파일: 허용
동일 output 파일 경로: 금지
동일 restart namespace: 금지
동일 scratch namespace: 금지
동일 output parent directory: 허용
```

Job별 구조:

```text
job_root/<job_id>/
    output.h5
    restart.h5
    logs/
    scratch/
```

Path 비교:

```text
weakly_canonical
symlink 해소
Windows case-insensitive 비교
```

Output parent는 충돌 금지 대상으로 사용하지 않는다. Restart와 scratch 기본 위치를 파생하고 canonicalize하는 용도로만 사용한다.

Batch mode의 restart 기본값은 output 경로 기준으로 한다. 입력 경로 기반 restart는 legacy opt-out에서만 허용한다.

### Phase 1 성공 기준

```text
FAIL 0
stale_evicted 0
overlap_rejections는 사전 허용된 pageable fallback만
slot refusal 0
모든 case가 대응 N≤64 결과와 bit-identical
N≤64 end-to-end 처리량 무퇴행
```

---

## 8. Phase 2 — Outer 반복 해부 Telemetry `[GO]`

수렴 파라미터를 바꾸기 전에 상태점별 반복 구성과 wall time을 분해한다.

```text
총 outer
Xe update 수
Xe update 1회당 flux reconvergence outer
boron/rod trial 수와 trial당 outer
T/H update 수와 update당 outer
settling gate 추가 outer
flux-limit-cycle retry
CMFD sweep 수
BiCG iteration 수
phase wall
CUDA graph dispatch
H2D / D2H
I/O 및 library parsing
```

### 8.1 Telemetry 구현 계약

```text
thread-local counter 또는 relaxed atomic
핫패스 문자열 formatting 금지
상태점 종료 시 한 번만 집계
상세 trace는 표본 slot만 기록
schema_version 포함
```

Receipt key:

```text
job_id
slot
statepoint
phase
schema_version
```

Telemetry on/off end-to-end wall 차이는 1% 이하여야 한다. 초과 시 계측을 최적화한 뒤에만 생산 profile에 사용한다.

산출물:

```text
실측 Amdahl 계수
Phase별 이론적 상한표
다음 Phase의 go/no-go 근거
```

---

## 9. Phase 3 — 저위험 Exact 가속

### 9.1 Statepoint 초기화 감사와 Warm-start formalization `[GO]`

새 기능을 추가하기 전에 현재 코드가 이미 유지하는 state를 감사한다.

```text
Phif / Phis / Jnet 유지 여부
dhat reset 시점
Xe/I 유지 및 재초기화
restart/shuffle 후 초기화 배열
boron/rod/T·H 변경 시 폐기 state
```

이미 warm-start가 수행 중이면 신규 기능을 중복 구현하지 않고, generation 기반 invalidation을 명문화한다.

```text
geometry generation
material/XS generation
depletion step generation
restart/shuffle generation
rod/cusping generation
power/T·H generation
xenon mode generation
```

### 9.2 Adaptive inexact inner solve `[조건부 GO]`

Adaptive inner solve는 최종 tolerance가 같아도 중간 궤적을 바꾸므로 Gate A와 Gate B 대상이다.

```text
feature-off: byte-golden
feature-on: Gate A + Gate B
```

Forcing metric은 다음처럼 고정한다.

```text
outer_metric = max(
    normalized fission-source L2,
    abs(delta_keff) / keff_scale
)

eps_inner(k) = clamp(c * outer_metric, eps_final, eps_max)
```

다음 조건에서는 즉시 adaptive 모드를 해제하고 canonical production solve로 복귀한다.

```text
residual 증가
negative-flux retry
flux limit cycle
search trial 변경
Xe damping 상태 또는 relaxation 변경
T/H 또는 material generation 변경
```

수렴 후보에 도달하면 출력 확정 전에 canonical polish를 수행한다.

```text
기존 production inner tolerance로 전환
→ 최소 1회 full outer 재구동
→ 기존 CMFD/비선형 residual 재검사
→ 통과 시에만 결과 확정
```

### 9.3 Krylov/Subspace recycling `[HOLD]`

Operator generation invalidation과 memory budget 설계를 완료한 뒤 착수한다.

### 9.4 불필요한 full reconvergence 감사 `[GO]`

외부 상태가 실제로 바뀌지 않았는데도 full reconvergence를 수행하는 경로가 있는지 계측으로 판별한다. 최종 residual을 생략하거나 완화하는 변경은 허용하지 않는다.

검증 집합:

```text
KNGR CY1
평형 Xe
Xe transient
boron search
rod search + cusping
Gd 연소 구간
restart/shuffle
T/H feedback
강한 reflector 문제
negative-flux retry 사례
```

---

## 10. Phase 4 — Safeguarded Anderson for Xe Fixed Point `[조건부 GO]`

### 10.1 Raw fixed-point API

기존 update 함수의 사후 hook으로 구현하지 않는다.

```cpp
XeFixedPointImage EvaluateEquilibriumXenon(
    const XeState& input,
    const FluxState& converged_flux);

void CommitEquilibriumXenon(const XeState& accepted);
```

반환값:

```text
iodine image
xenon image
raw true residual F(x) - x
```

### 10.2 Side-effect-free candidate trial

Anderson acceptance를 위해 production `Driver::SolveLoop` 전체를 trial로 호출하지 않는다. 외부 상태를 고정한 side-effect-free 평가 함수를 둔다.

```cpp
TrialResult EvaluateCoupledXeCandidate(
    const CoupledStateSnapshot& base,
    const XeState& candidate);
```

Trial 동안 고정:

```text
boron
rod
T/H
power
depletion
search state
restart/HDF5 write
production iteration counters
```

Trial이 수행하는 계산:

```text
CMFD
nodal
current
d-hat
Xe coupled residual 평가
```

Trial 비용은 성능 wall에 포함하지만 production outer/search counter와 별도 계측한다.

### 10.3 Transactional rollback

Candidate 적용 후 변경될 수 있는 결합 상태 전체를 보존한다.

```text
I/Xe state
reconstructed XS
XS generation
flux
fission source
eigenvalue
current
d-hat
nodal/cusping state
```

Trial이 외부 상태를 변경할 가능성이 남아 있다면 다음도 snapshot에 포함한다.

```text
search history
T/H counter
iter / wiel_sweep / stall counter
telemetry delta
scheduler state
```

권장 구조는 scratch state에서 candidate를 평가하고 accept 시 swap하는 방식이다.

### 10.4 수용 조건

모든 조건을 통과해야 한다.

```text
I/Xe density ≥ 0
모든 값 finite
step norm ≤ trust region
history condition number ≤ 상한
예측 residual 감소
후속 full-exact true residual 감소
axial branch guard 통과
```

Axial branch guard:

```text
abs(AO_candidate - AO_picard) ≤ axial_trust
```

필요하면 정규화 axial power vector의 L2 거리도 함께 사용한다. `axial_trust`는 Gate A 기준 동결 단계에서 정한다.

### 10.5 Damping과 history 규칙

```text
Anderson history: raw undamped F(x)-x만 저장
Anderson 실패: 기존 production damping 적용
새 damping 발동 또는 relaxation 변경: history reset
Anderson candidate에 damping 중복 적용 금지
```

History reset 조건:

```text
boron 변화
rod 변화
T/H 변화
depletion 변화
restart/shuffle
power fraction 변화
xenon mode 변화
material/XS generation 변화
damping relaxation 변화
```

Depth는 2에서 시작하고 Gate 통과 후 3을 시험한다.

Telemetry:

```text
proposed
accepted
rejected_step
rejected_residual
rejected_physics
rejected_axial_branch
rollback_after_exact_check
history_reset
fallback_picard
trial_wall
```

---

## 11. Phase 5 — Persistent/Cooperative CMFD `[HOLD]`

Phase 2 Nsight 결과가 다음 조건을 모두 만족할 때만 착수한다.

```text
graph/kernel dispatch + device synchronization
    ≥ solver wall의 10%

이론적으로 제거 가능한 시간
    ≥ end-to-end wall의 5%
```

실행 전 eligibility:

```text
cudaOccupancyMaxActiveBlocksPerMultiprocessor
required_grid_blocks ≤ resident_blocks
```

조건을 만족하지 않으면 기존 CUDA Graph 경로를 유지한다.

Bit 보존 고정 항목:

```text
block count
block별 element range
thread traversal order
stage-1 partial 생성 순서
stage-2 fold 순서
FMA 선택
color 순서
slot mapping
```

검증 지표:

```text
kernel time
GPU active
SM occupancy
other-stream wait
CMFD batch width
Nodal batch width
arrival gap
end-to-end throughput
```

Persistent kernel이 다른 stream의 overlap을 제거하여 전체 wall을 악화시키면 채택하지 않는다. 기대 범위는 1.05–1.3배이며 2배는 공격적 상한으로만 취급한다.

---

## 12. Phase 6 — Arena Width 확대 `[HOLD]`

Phase 1 rolling queue 안정화와 Phase 2 profile 이후에만 검토한다.

### 12.1 폭 확대 전 전수 감사

```text
[65]
[64]
uint64_t mask
1ULL <<
bitset<64>
min(width, 64)
graph capture에 bake된 slot 수
active mask 길이
pinned scalar 배열 크기
histogram backing array
JSON 출력 loop 상한
```

Histogram clamp 해제와 slot-mask 확장은 Phase 1이 아니라 이 Phase에서 별도 커밋으로 수행한다.

### 12.2 Go/No-go 지표

Phase 2 보고서에서 다음 threshold를 동결한다.

```text
CMFD mean/p50/p90 batch width
Nodal mean/p50/p90 batch width
width ceiling hit fraction
GPU active
SM utilization
memory bandwidth
arrival gap
peak VRAM
peak pinned bytes
```

초기 Go 조건의 구조:

```text
현재 width ceiling에 반복적으로 도달
AND GPU 계산 자원이 유휴
AND peak VRAM/pinned RAM이 안전 상한 미만
```

평균 batch width가 낮다는 이유만으로 width를 늘리지 않는다. Go 조건이 충족되면 `64 → 80 → 96` 순서로만 시험한다. 한 번에 128 이상으로 확대하지 않는다.

기존 `96 / 96 / 96` 실험은 다음처럼 분해한다.

```text
width 효과: workers 고정
worker 효과: width 고정
```

---

## 13. Phase 7 — Multi-GPU `[설계만 GO]`

단일 process가 여러 GPU를 복잡하게 공유하는 구조 대신 다음을 기본으로 한다.

```text
GPU 0 process: 독립 64-slot arena
GPU 1 process: 독립 64-slot arena
...
상위 dispatcher: exact deck queue 분배
```

각 process는 동일한 full-exact 물리 경로를 실행한다. 단일 GPU 상한을 확인하고 장비를 확보한 뒤 구현한다.

---

## 14. 별도 트랙 — XSLIB Cache `[GO, 독립 A/B]`

HostPinLease 변경과 같은 arm에 넣지 않는다.

Cache key:

```text
canonical path
file-id/inode
size
nanosecond mtime
선택적 content hash
library schema/version
```

구조:

```text
std::shared_ptr<const ImmutableLibrary>
    parsing result
    coefficient tables
    metadata

PerXSSetMutableState
    fmap/gmap
    node mapping
    depletion state
    rod/burnup interpolation state
```

공유 전 mutable-field audit:

```text
const method 내부 lazy cache
mutable lookup table
thread-unsafe reference counter
model별 temporary scratch
문자열/HDF5 handle 수명
burnup/rod interpolation이 변경하는 필드
```

Immutable object는 API 수준에서 `const`로 강제한다.

---

## 15. 실행 순서와 산출물

### 15.1 실행 순서

```text
1. Phase 0 provenance 또는 전달물 검증
2. Gate A baseline/threshold 동결
3. Phase 1A 원인 판별
4. Phase 1B HostPinLease 구현
5. Namespace 분리 및 rolling queue 검증
6. Phase 2 telemetry와 Amdahl 모델 확정
7. Phase 3 warm-start 감사 및 adaptive solve
8. Phase 4 safeguarded Anderson
9. 조건 충족 시 Phase 5 persistent CMFD
10. 조건 충족 시 Phase 6 width 확대
11. 단일 GPU 상한 확인 후 Phase 7 multi-GPU
```

Phase 1B의 커밋 순서는 다음과 같이 분리한다.

```text
HostPinLease
→ namespace/restart isolation
→ rolling queue validation
```

Histogram과 slot-width 변경은 Phase 6까지 수행하지 않는다.

### 15.2 Phase 종료 산출물

각 Phase마다 다음을 남긴다.

```text
구현 및 설계 보고서: docs/*.md
benchmark raw logs
machine-readable receipt
physics comparison report
Amdahl ceiling update
GitHub commit / branch / PR
rollback 환경변수
```

태그는 모든 해당 gate를 통과한 시점에만 생성하고 `VERSIONS.md`의 stage timing을 새로 측정해 갱신한다. Gate A/B 대상 기능을 채택하면 MASTER 비교와 관련 그림을 전체 재생성한다.

---

## 16. 핵심 파일과 Symbol Anchor

라인 번호는 참고용이며 실제 작업 anchor는 symbol 기준으로 둔다.

### Phase 1

```text
src/CudaXsReconBackend.{h,cu}
  XsReconBackend::pinHost
  NodalArena::pinSlot
  host pinning registry / lease

src/CudaBICGBackend.cu
  CudaBatchArena::pinHost
  batch slot lifecycle

src/main.cpp
  batch execution
  exact-only gate
  host pinning receipt

src/Driver.h
  SaveRestart 및 job namespace

src/IO.{h,cpp}
  result/restart/scratch path

tools/run_single_gpu_batch.py
  path validation
  exact-mode receipt validation
```

### Phase 2

```text
src/Driver.h
src/BICGCMFD.cpp
src/XSSet.cpp
```

### Phase 4

```text
XSSet::UpdateEquilibriumXenon
  → EvaluateEquilibriumXenon / CommitEquilibriumXenon 분리

Driver의 Xe feedback hook
EvaluateCoupledXeCandidate
CoupledStateSnapshot
```

### XSLIB cache

```text
XSSet.cpp의 Importer::LoadHDF 호출부
include/chiffon/Importer.h
ImmutableLibrary / PerXSSetMutableState 경계
```

---

## 17. 최종 채택 조건

최종 성능 결과는 다음 조건을 모두 만족해야 한다.

```text
screening 0
feedback-pass 제한 0
모든 입력 full-exact
full HDF5 출력
물리 및 수렴 Gate A 통과
MASTER Gate B 통과
fallback 0 또는 사전 허용된 fail-open만 명시
재현 가능한 benchmark distribution
end-to-end 처리량 milestone 달성
```

성능만 높고 다른 Xe/axial branch로 이동하거나, 최종 residual을 완화하거나, 일부 입력만 exact로 재계산한 결과는 채택하지 않는다.
