# GPU RASBERY 병목 개선 및 단일 GPU 병렬 처리량 향상 구현 계획

> **For agentic workers:** 이 계획을 구현할 때는 각 Work Package별로 `superpowers:test-driven-development`를 먼저 적용하고, 독립 작업은 `superpowers:subagent-driven-development`, 순차 실행은 `superpowers:executing-plans`, 완료 판정 전에는 `superpowers:verification-before-completion`을 사용한다.

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 저장소 | `allusion127/gpu` |
| 대상 브랜치 | `codex/exact-throughput-campaign` |
| 분석 고정 SHA | `8b8f18e1c837121bd7873b43c207dafeda632edd` |
| 실제 코드 부모 SHA | `c502856d1aa0d57ca12681bc73bc3d093af3a240` |
| 기준 시각 | 2026-08-30 KST |
| 주 분석 문서 | `docs/GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md` |
| 교차 검증 문서 | `docs/GPU_RASBERY_PERFORMANCE_AND_ARCHITECTURE_REPORT_20260830_KO.md` |
| 주요 대상 장비 | 서버 238, RTX PRO 6000 Blackwell, `sm_120`, CUDA 13.0, GPU0 단독 |
| 주 목표 | 단일 물리 GPU에서 GA 평가 처리량 극대화와 단일 케이스 지연시간 단축 |
| 비목표 | 검증 없이 물리 충실도를 낮추는 것, 다중 GPU 수치로 단일 GPU 성능을 포장하는 것, 전체 계산을 무조건 GPU에 상주시키는 것 |

> **검토 한계:** 이 문서는 저장소 코드와 커밋된 성능 수신증·프로파일 보고서를 정적 검토하여 작성했다. 이 검토 과정에서 서버 238의 새로운 실행이나 Nsight 재측정은 수행하지 않았다. 따라서 아래의 절대 처리량은 기존 실측이며, 새 개선안의 효과는 모두 명시된 A/B 게이트를 통과한 뒤에만 채택한다.

---

## 1. 목표, 아키텍처 원칙, 성공 정의

### Goal

1. **단일 케이스 경로**에서는 반복적인 호스트 왕복, CUDA API 호출, 작은 커널 발사, 상태점별 CPU 바닥을 줄인다.
2. **병렬 M64 경로**에서는 선언 폭 64에 비해 실제 도착 폭이 14.5에 그치는 문제를 해결하고, 배치 GPU 시간의 39.9%를 차지하는 `kernelFlatXs`를 우선 개선한다.
3. **GA 평가기 경로**에서는 프로세스·CUDA 컨텍스트·정적 라이브러리·그래프 구축 비용을 세대마다 반복하지 않도록 장수명 evaluator를 도입한다.
4. 모든 성능 개선은 물리 모드, 출력 모드, 수치 궤적 등급을 분리해 기록하고, `RASBERY_GPU_FULL=1`에서는 CPU 수치 fallback을 허용하지 않는다.

### Architecture

선택한 구조는 **호스트 장수명 evaluator + GPU 위상별 비동기 백엔드 + 실제 활성 슬롯 기반 배치**이다.

- 호스트는 케이스의 불규칙한 스케줄·붕소 탐색·실패 격리를 관리한다.
- GPU는 CMFD/Nodal/FlatXS/Xe/PPR의 계산 본문과 반복 종료 판단을 가능한 범위에서 수행한다.
- 하나의 거대한 persistent GPU scheduler는 사용하지 않는다. 현재 W0 측정에서 device barrier 비용이 kill threshold를 초과했고, 케이스 간 불규칙성 때문에 이 구조는 재도입 우선순위가 아니다.
- 병렬화는 `K processes × W slots/process`, active-slot compaction, 동적 refill을 조합하되 **총 선언 슬롯과 CPU/VRAM 예산을 고정한 상태에서** 선택한다.

### Tech Stack

- C++17/20, CUDA C++, CUDA Graph conditional node
- OpenMP 동적 작업 큐
- HDF5 비동기 writer 경로
- Python 벤치마크·dispatcher·계약 테스트
- CMake/CTest
- `nsys`, `ncu`, `ptxas -v`, `h5diff`

### 성공 정의

아래 네 조건을 모두 만족해야 한다.

1. **정확성:** 선택한 게이트 등급(B0/N1/A2)을 위반하지 않고, 결정론·MASTER 비교·핀 출력·붕소·AO 게이트를 통과한다.
2. **실행 계약:** GPU full 모드에서 모든 요구 백엔드가 실제로 실행되며, 수치 fallback·capture 누수·중복 슬롯·stale tenant가 0이다.
3. **성능:** 동일 장비·동일 SHA·동일 덱·동일 출력/충실도 모드에서 사전 정의한 최소 개선율을 넘는다.
4. **운영성:** 수천~수만 케이스 연속 실행에서 메모리 증가, cross-case 상태 누출, 출력 충돌, queue 중복이 없다.

---

## 2. 핵심 결론

### 2.1 한 줄 결론

현재 다음 성능 향상의 핵심은 **GPU offload의 개수를 늘리는 것 자체가 아니라**, 이미 구현된 기능을 서버 238에서 가격 평가하고, M64의 빈 슬롯·FlatXS 작업구조·PPR 호스트 반복·프로세스 수명주기를 차례로 제거하는 것이다.

### 2.2 우선순위

| 우선순위 | 작업 | 이유 | 예상 위험 |
|---:|---|---|---|
| P0 | 출력 모드와 물리/수치 충실도 의미 분리 | 현재 `light` 출력이 실제 solve와 무관하게 screening으로 분류되는 계약 혼선이 있음 | 낮음 |
| P0 | GPU full fail-closed 계약 | 현재 PPR/Nodal/XS 경로는 CUDA 실패 시 CPU로 fail-open 가능 | 중간 |
| P0 | 이미 착지한 WHILE/XSLIB/K-process/CMFD compaction/GPU PPR의 238 A/B | 새 코드를 쓰기 전에 가장 싼 성능 레버의 실제 효과를 확정해야 함 | 낮음 |
| P1 | GPU당 K-process 자동 튜닝 + active-slot compaction | M64 평균 도착 폭 14.5/64, `width_fill=0.227`이 처리량의 직접 병목 | 중간 |
| P1 | FlatXS CTA-per-node 협업 커널 | 배치 GPU 시간 39.9%; 현재 스레드당 약 7 KiB 지역 작업공간이 spill/점유율 병목일 가능성 | 중간~높음 |
| P1 | PPR device convergence + device reconstruction | 상태점 CPU 바닥의 큰 비중이며 현재 매 반복 D2H+sync와 최종 대량 D2H 수행 | 높음 |
| P2 | 장수명 evaluator | 프로세스·CUDA 컨텍스트·그래프·정적 캐시 비용을 세대/청크마다 반복하지 않음 | 높음 |
| P2 | XS/PPR canonical residency | FlatXS와 PPR 사이의 대량 H2D/D2H 제거 | 높음 |
| P2 | GA 중복 캐시·warm-start·다단계 평가 | 코드 계산량 자체를 줄이는 최상위 처리량 레버 | 중간 |

### 2.3 단일과 병렬은 같은 최적화 순서를 쓰지 않는다

- **단일 케이스:** `colored_block_sweep`가 701,056회, 평균 2.23 µs로 실행되고 `cudaMemcpyAsync` 호출이 117,829회다. 핵심은 **발사·동기화·호스트 관측 횟수**이다.
- **M64 병렬:** `kernelFlatXs`가 GPU 시간의 39.9%, Xe가 26.9%, 실제 rendezvous 폭은 14.5/64다. 핵심은 **활성 슬롯 밀도, 호스트 도착률, 큰 커널 내부 효율**이다.
- 따라서 device outer segment처럼 단일에서 큰 효과를 낸 기능이 배치에서는 중립일 수 있으며, 배치에서 budget을 늘리는 방식은 오히려 5.6배 회귀한 전례가 있다.

---

## 3. 현재 성능 기준선과 해석

### 3.1 단일 덱 — 서버 238

| 단계 | wall | MASTER 27.2 s 대비 | 비고 |
|---|---:|---:|---|
| MASTER CPU | 27.2 s | 1.00× | 기준기 |
| v2 GPU arm | 55.4 s | 0.49× | 출발점 |
| + device outer segment | 44.5 s | 0.61× | 단일 경로에서 유효 |
| + chunked Wielandt | 32.1 s | 0.85× | N1 |
| + device Xe Anderson | 26.1 s | 1.04× | N1 |
| + A2 staged tolerance | 16.9 s | 1.61× | 별도 N1(A2) 계약으로 관리 |

- outer 수는 10,483에서 4,382로 감소했다.
- A2는 물리 모델 자체를 coarse하게 만드는 L3와 같지 않지만, 수렴 정책과 궤적을 바꾸므로 **strict baseline과 별도 arm으로 기록**해야 한다.
- strict/full-exact, A2, coarse-10-state를 같은 처리량 숫자로 혼합해서는 안 된다.

### 3.2 M64 — 서버 238 GPU0

| 단계 | 처리량 | MASTER W16 217 c/h 대비 | 비고 |
|---|---:|---:|---|
| v2 | 216 c/h | 1.00× | 기준선 |
| + chunked fold | 245 c/h | 1.13× | N1 |
| + device Xe | 322 c/h | 1.48× | N1 |
| + A2 | 518–534 c/h | 2.39–2.46× | A2 arm |
| + scalar-only 출력 | 577.6 c/h | 2.66× | solve는 동일, 출력 형태만 축소 |

> **중요:** `518–577.6 c/h`를 “B0 strict full-output 처리량”으로 부르면 안 된다. A2 사용 여부와 `full/light` 출력 여부를 receipt에서 분리하고, 개선 전후를 반드시 mode-matched baseline으로 비교한다.

### 3.3 배치 GPU 시간의 Amdahl 우선순위

| 서브시스템 | GPU 시간 비중 | 해당 부분 50% 단축 시 전체 이론 개선 | 완전 제거 시 상한 | 판정 |
|---|---:|---:|---:|---|
| FlatXS | 39.9% | 약 1.25× | 약 1.66× | 최우선 커널 |
| Xe Anderson | 26.9% | 약 1.16× | 약 1.37× | 두 번째 |
| CMFD/BiCGSTAB | 약 19% | 약 1.10× | 약 1.23× | compaction과 결합 |
| Nodal | 8.6% | 약 1.05× | 약 1.09× | 선행 레버 이후 |

이 표는 커널 시간만을 대상으로 한 상한이다. 호스트 대기·도착 폭이 동시에 바뀌면 단순 곱셈이 성립하지 않는다.

### 3.4 단일 GPU 시간의 Amdahl 우선순위

| 항목 | GPU 시간 비중 | 50% 단축 시 전체 이론 개선 | 주 레버 |
|---|---:|---:|---|
| `colored_block_sweep` | 23.7% | 약 1.13× | graph node/launch 축소, 안전한 fusion |
| reduce-dot 계열 | 15.5% | 약 1.08× | 고정 순서 fusion, device control |
| FlatXS | 14.4% | 약 1.08× | 협업 커널 |
| PPR/상태점 CPU 바닥 | GPU 프로파일 밖 | 별도 | device convergence/reconstruction |

---

## 4. 코드 구조와 실제 상태

### 4.1 주요 호출 경로

```text
src/main.cpp
 ├─ argv / --jobs / result-mode / physics receipt
 ├─ single Driver::Drive()
 └─ batch OpenMP schedule(dynamic,1)
      └─ Driver per job
           ├─ IO::ReadInput / XSSet::Initialize
           ├─ Scheduler statepoints
           ├─ SolveLoop / ReconvergeFlux
           │    ├─ CudaBICGBackend       : CMFD/BiCGSTAB/outer
           │    ├─ CudaXsReconBackend    : FlatXS/Xe/Nodal
           │    └─ host control          : search, convergence, schedule
           ├─ PPR
           │    ├─ CudaPprBackend        : reset + drive
           │    └─ host reconstructPinPower
           └─ IoWriter / result mode
```

### 4.2 모듈별 역할

| 모듈 | 현재 역할 | 개선의 핵심 |
|---|---|---|
| `src/main.cpp` | 실행 모드, manifest, OpenMP refill, receipt | 출력/충실도 분리, evaluator loop, fail-closed 집계 |
| `src/Driver.h` | 상태점·outer·Xe·PPR 전체 orchestration | GPU full 계약, 상태점 phase receipt, case reset 경계 |
| `src/EvaluatorContext.h` | process/case 수명 경계를 문서화한 seam | 실제 Process/Cohort/Case context 구현 |
| `src/XSSet.cpp`, `src/XsLibrary.h` | immutable XSLIB single-flight cache, mutable per-case XS | 장수명 캐시 키 강화, device residency |
| `src/CudaBICGBackend.cu` | CMFD batch arena, graph, outer, compaction 코드 | compaction 검증·기본화, launch/node 축소 |
| `src/CudaXsReconBackend.cu` | FlatXS/Xe/Nodal backend | FlatXS 협업 커널, XS ownership, Xe transaction graph |
| `src/FlatXsKernel.h` | 한 노드의 CPU/GPU 공유 산술 본문 | CTA 협업형 B0 본문 추가 |
| `src/CudaPprBackend.cu` | PPR reset+drive, 반복마다 host check | device WHILE, canonical input, device reconstruction |
| `tools/run_multi_gpu_batch.py` | multi-GPU/K-process/MPS/queue | GPU0 자동 튜닝, 실제 thread/VRAM budget, receipt 기반 선택 |
| `src/GpuPhaseScheduler.*` | 분류/compact/refill scaffold | persistent device scheduler로 확대하지 않고 보조 primitive로만 사용 |

### 4.3 “완료 / 착지했지만 미가격 / 미구현” 재분류

| 기능 | 코드 상태 | 서버 238 가격 | 본 계획의 처리 |
|---|---|---|---|
| XSLIB host cache | 기본 ON, single-flight 구현 | 미가격 | 새 구현보다 A/B 우선 |
| writer thread / light output | 구현·측정 | 측정됨 | 유지, 의미 계약만 수정 |
| host-free outer | 구현·채택 | 일부 측정 | 유지 |
| conditional WHILE | 구현, 기본 OFF | 미가격 | 단일 우선 A/B |
| GPU PPR | 구현, N1, 기본 OFF, fail-open | 미가격 | 정확성/성능 gate 후 fail-closed 확장 |
| K-process/GPU + MPS | launcher 구현 | 로컬만 측정 | GPU0 행렬 자동 튜닝 |
| CMFD active-slot compaction | **소스에 구현되어 있으며 기본 OFF** | 미가격 | “신규 개발”이 아니라 검증·하드닝·채택 작업 |
| immediate refill | 구현 | 측정됨 | 유지 |
| persistent evaluator | 수명 경계 문서만 존재 | 없음 | 2단계 구현 |
| FlatXS cooperative kernel | 없음 | 없음 | 신규 최우선 GPU 커널 작업 |
| XS canonical residency | 일부 nodal/canonical만 존재 | 없음 | consumer audit 후 확장 |
| PPR device convergence/reconstruct | 없음 | 없음 | 신규 |
| result-mode / fidelity 분리 | 불완전 | 해당 없음 | P0 계약 수정 |
| GPU full fail-closed | 없음 | 해당 없음 | P0 계약 수정 |

### 4.4 보고서와 소스의 중요한 차이

최신 종합 보고서는 slot compaction을 “아직 손대지 않음”으로 분류하지만, 고정 SHA의 `src/CudaBICGBackend.cu`에는 이미 `RASBERY_GPU_CMFD_COMPACT`, logical-to-physical slot map, bucket graph, `[RASBERY][CMFD][COMPACT]` receipt가 존재한다. 따라서 이 항목은 다시 구현하지 않고 다음만 수행한다.

1. 모든 CMFD 커널이 `blockIdx.y`를 직접 physical slot으로 사용하지 않는지 계약 테스트를 강화한다.
2. identity-off 경로가 이전 바이너리와 B0인지 확인한다.
3. 238 M64에서 padding 감소와 wall 개선을 측정한다.
4. 이득이 있을 때만 기본 ON 후보로 승격한다.

---

## 5. 목표 실행 구조

```text
GA controller
  │ generation manifest / candidate key / fidelity / output mode
  ▼
GPU0 fleet tuner and dispatcher
  ├─ Worker process 0 : evaluator, CMFD/Nodal/XS/PPR arenas W slots
  ├─ Worker process 1 : evaluator, CMFD/Nodal/XS/PPR arenas W slots
  └─ ... K processes on the same physical GPU
       │
       ├─ ProcessContext  : immutable library, CUDA context, graph caches
       ├─ CohortContext   : geometry/topology/library-hash scoped immutable state
       └─ CaseContext     : scheduler, mutable XS/isotopes, output, slot tenancy
             │
             ├─ CMFD active-slot compacted graph
             ├─ Nodal canonical state
             ├─ FlatXS CTA-per-node cooperative kernel
             ├─ Xe device transaction
             └─ PPR device convergence + reconstruction
                    │
                    ├─ light: GA scalar JSONL only
                    └─ full: selected HDF5/pin data materialized at output boundary
```

### 5.1 세 가지 수명

| 수명 | 포함 | 금지 사항 |
|---|---|---|
| Process | CUDA context, immutable XSLIB, T/H tables, device library, graph cache | solve 중 변경되는 배열 저장 금지 |
| Cohort | topology/geometry hash, immutable neighbor map, PPR quadrature, arena shape | 다른 geometry candidate와 공유 금지 |
| Case | schedule, mutable node XS, isotope state, convergence history, output | 다음 케이스로 값 누출 금지 |

### 5.2 persistent evaluator와 persistent GPU kernel을 구분한다

- 채택: **프로세스가 오래 살아 있는 evaluator**. CUDA context와 cache를 유지하고 케이스를 반복 처리한다.
- 비채택: **모든 케이스 위상을 하나의 cooperative persistent GPU kernel이 스케줄**하는 구조. 현재 barrier 비용과 불규칙 제어 흐름을 고려하면 우선순위가 아니다.

---

## 6. 전역 정확성·실행 계약

### 6.1 게이트 등급

| 등급 | 허용 변화 | 필수 게이트 |
|---|---|---|
| B0 | 비트 동일, 실행 위치·수명·발사만 변경 | feature-off byte identity, ON×2 결정론, per-deck batch=single, digest 동일 |
| N1 | 고정된 연산 분할/수학함수 차이로 궤적이 변할 수 있으나 결정론적 | Gate A, MASTER Gate B, digest 반복, 핀 RMS/max, 붕소/AO |
| A2 | staged tolerance 등 별도 수렴 정책 | N1 전체 + `policy=A2` receipt + strict 재평가 정책 |
| L3 coarse | 상태점 자체 축소 | screening 전용 + 승격 후보 strict/full rerun |

### 6.2 출력 모드와 계산 충실도 분리

현재 `main.cpp`는 `light` 출력 요청을 `screening=true`로 묶는다. 그러나 저장소 보고서와 코드 설명상 `full/pin-off/light`는 같은 solve를 수행하고 쓰는 결과만 다르다. 다음 두 enum을 분리한다.

```cpp
enum class PhysicsFidelity {
    FullExact,
    StagedA2,
    Coarse10State,
    FeedbackLimited
};

enum class ResultMode {
    Full,
    PinOff,
    Light
};
```

필수 receipt:

```json
{
  "physics_fidelity": "full_exact",
  "result_mode": "light",
  "acceptance_eligible": true,
  "screening": false,
  "requires_exact_rerun": false
}
```

`light`이면서 `FullExact`인 경우는 합법적이고 acceptance-eligible이다. `Coarse10State` 또는 `FeedbackLimited`인 경우에만 출력 형태와 무관하게 screening이다. A2는 프로젝트 승인 정책에 따라 별도 acceptance flag를 부여하되 strict와 섞지 않는다.

### 6.3 GPU full fail-closed

`RASBERY_GPU_FULL=1`일 때 다음을 강제한다.

1. CMFD, Nodal FULL, FlatXS/XSRecon, PPR의 요구 arm이 모두 **enabled이면서 engaged**여야 한다.
2. CUDA 오류·shape 거부·batch slot 거부·graph fallback이 발생하면 그 케이스를 실패 처리한다.
3. CPU 수치 fallback을 실행하지 않는다.
4. 실패한 케이스는 임시 출력만 삭제하고, 같은 batch의 다른 케이스는 계속 진행한다.
5. 허용된 host boundary materialization은 목록으로 관리한다. 임의의 중간 D2H/H2D는 계약 위반이다.

최종 receipt 예:

```json
{
  "gpu_full": true,
  "cmfd_fallbacks": 0,
  "nodal_fallbacks": 0,
  "flatxs_fallbacks": 0,
  "ppr_fallbacks": 0,
  "graph_fallbacks": 0,
  "mid_iteration_materializations": 0,
  "contract_pass": true
}
```

### 6.4 성능 실험 규칙

- GPU0만 사용한다. `CUDA_VISIBLE_DEVICES=0`을 자식 프로세스까지 고정한다.
- GPU clock/power/driver/CUDA/toolchain/SHA/input hash를 매니페스트에 기록한다.
- 첫 실행은 warm-up으로 버리고 최소 5회 median, p10/p90을 보고한다.
- telemetry 실행과 wall timing 실행을 분리한다.
- feature 하나의 A/B를 교대로 실행해 온도·clock drift를 상쇄한다.
- 출력 모드와 fidelity가 다른 실행을 비교하지 않는다.
- 개선율이 3% 이하이면 노이즈로 간주하고 기본값을 바꾸지 않는다.
- 절대 wall뿐 아니라 `width_fill`, `padding_fraction`, `tail_idle_s`, CPU wait, GPU SM, H2D/D2H, graph count를 함께 판정한다.

---

## 7. 의존성 순서

```text
WP0 측정 동결
 ├─ WP1 실행/충실도 계약
 └─ WP2 착지 기능 재가격
       ├─ WP3 CMFD compaction 채택
       ├─ WP4 K-process 자동 튜닝
       ├─ WP7 single WHILE/launch 축소
       └─ WP6 GPU PPR 1차 gate

WP3 + WP4 측정 완료
 └─ WP5 FlatXS cooperative + residency
       └─ WP6 PPR device loop/reconstruct/canonical
             └─ WP9 잔여 host floor

WP1 + WP2 안정화
 └─ WP8 persistent evaluator
       └─ WP10 GA cache/warm-start/multi-fidelity
             └─ WP11 soak/freeze/default adoption
```

WP3과 WP4는 같은 “빈 슬롯 비용”을 다른 방식으로 줄인다. 각각 단독 측정 후 조합해야 하며, 단독 개선율을 그대로 곱하지 않는다.

---

# 상세 구현 Work Packages

## WP0 — 재현 가능한 기준선과 자동 판정 하네스

### 목적

현재 보고서의 수치를 재현 가능한 machine-readable baseline으로 고정하고, 이후 모든 최적화가 같은 규칙으로 평가되게 한다.

### 변경 파일

- 생성: `tools/run_exact_throughput_matrix.py`
- 생성: `tools/parse_perf_receipts.py`
- 생성: `tools/compare_perf_runs.py`
- 생성: `test/reference/perf_manifest_sm120_8b8f18e1.json`
- 수정: `tools/run_multi_gpu_batch.py`
- 수정: `src/main.cpp`
- 선택 생성: `.github/workflows/contracts.yml` — CPU/소스 계약만, 하드웨어 성능 판정은 서버 238 별도

### 테스트 우선 절차

1. **실패 테스트 작성**
   - SHA, input SHA256, GPU UUID, CUDA driver/runtime, CMake flags, env 원문, result mode, fidelity 중 하나라도 누락된 run을 parser가 거부하게 한다.
   - 두 arm의 result mode/fidelity가 다르면 비교를 거부하게 한다.
   - telemetry가 켜진 timing run을 거부하게 한다.

2. **실패 확인**

```bash
python3 tools/test_perf_manifest_contract.py
# 예상: 새 필드가 구현되기 전 FAIL
```

3. **최소 구현**
   - `[RASBERY][BUILD]`, `[RASBERY][RUN_CONTRACT]`, `[RASBERY][PROCESS]` receipt를 하나의 JSON result로 합친다.
   - `run_exact_throughput_matrix.py`가 warm-up 1회 + 측정 5회를 자동 수행하고 median/p10/p90을 출력한다.
   - 모든 raw log, `nvidia-smi -q`, compile command, env를 arm별 디렉터리에 보존한다.

4. **통과 확인**

```bash
python3 tools/test_perf_manifest_contract.py
python3 tools/test_benchmark_parser.py
# 예상: PASS, mode mismatch fixture는 의도한 오류 메시지로 거부
```

5. **커밋 예시**

```bash
git add tools src/main.cpp test/reference .github/workflows
git commit -m "test(perf): freeze mode-aware sm120 throughput manifests"
```

### 서버 238 기준 빌드

```bash
export REPO=$HOME/gpu
export BLD=$HOME/build/rasbery-8b8f18e1-sm120
export DATA=$HOME/kngr_238
export OUT=$HOME/bench/rasbery-8b8f18e1

git -C "$REPO" checkout 8b8f18e1c837121bd7873b43c207dafeda632edd
cmake -S "$REPO" -B "$BLD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DRASBERY_ENABLE_CUDA=ON \
  -DRASBERY_CUDA_ARCHITECTURES=120 \
  -DRASBERY_ENABLE_TESTS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$BLD" -j
ctest --test-dir "$BLD" --output-on-failure
```

### 기준 arm

반드시 세 개를 별도 보관한다.

- `strict_full`: strict/full fidelity + full output
- `a2_full`: A2 + full output
- `a2_light`: A2 + light output

추가로 `strict_light`를 만들어 출력 비용만 분리한다.

### 완료 기준

- 모든 run이 SHA·모드·환경·입력 hash를 갖는다.
- 동일 arm 5회에서 digest가 모두 같다.
- baseline median 변동이 기존 보고서 대비 합리적인 범위인지 확인하고, 차이가 크면 새 baseline으로 명시적으로 재동결한다.
- GitHub combined status가 없는 현재 상태를 보완해 CPU 계약 테스트가 자동 실행된다.

---

## WP1 — 출력/충실도 계약 분리 및 GPU full fail-closed

### 목적

성능 숫자의 의미를 고정하고 “GPU arm이 켜졌다고 표시됐지만 실제로는 CPU fallback한 실행”이 acceptance에 들어오지 못하게 한다.

### 변경 파일

- 생성: `src/RunContract.h`
- 생성: `src/RunContract.cpp`
- 생성: `src/GpuFullContract.h`
- 생성: `src/GpuFullContract.cpp`
- 수정: `src/main.cpp`
- 수정: `src/Driver.h`
- 수정: `src/BatchLightResult.h`
- 수정: `src/CudaBICGBackend.{h,cu}`
- 수정: `src/CudaXsReconBackend.{h,cu}`
- 수정: `src/CudaPprBackend.{h,cu}`
- 수정: `CMakeLists.txt`
- 생성: `tools/test_result_fidelity_contract.py`
- 생성: `tools/test_gpu_full_fail_closed.py`

### 설계

`RunContract`는 다음을 immutable하게 보유한다.

```cpp
struct RunContract {
    ExecutionMode execution_mode;
    PhysicsFidelity fidelity;
    ResultMode result_mode;
    bool acceptance_eligible;
    bool gpu_full;
};
```

`GpuFullContract`는 각 subsystem의 요구/engagement/fallback을 원자 카운터로 기록한다. subsystem이 `false`를 반환해 caller가 CPU 경로로 이동하기 전에 다음 정책을 질의한다.

```cpp
if (!gpu_ok) {
    if (GpuFullContract::required())
        throw GpuFullViolation("PPR", reason);
    run_host_fallback();
}
```

### 테스트 우선 절차

1. `light + full_exact` fixture가 현재 코드에서 screening으로 잘못 분류되는 실패 테스트를 작성한다.
2. CUDA backend를 강제 거부하는 test seam에서 `RASBERY_GPU_FULL=1`이 CPU fallback 대신 job failure를 내는 실패 테스트를 작성한다.
3. batch 4덱 중 하나만 강제 실패시켜 나머지 3덱은 완료되고 실패 덱의 최종 출력은 없는지 검증한다.
4. 구현 후 다음을 실행한다.

```bash
python3 tools/test_result_fidelity_contract.py
python3 tools/test_gpu_full_fail_closed.py
ctest --test-dir "$BLD" --output-on-failure
```

### 필수 acceptance

- `light + full_exact` → `screening=false`, `acceptance_eligible=true`.
- `coarse + full` → `screening=true`, `acceptance_eligible=false`.
- GPU full에서 모든 fallback count가 0.
- feature-off 실행은 기존 출력과 B0.
- GPU full 위반은 케이스 단위 실패이며 batch 전체를 비정상 종료시키지 않는다.

### 성능 판정

계약 카운터의 기본 OFF/비계측 경로 오버헤드는 단일 및 M64에서 1% 미만이어야 한다. 초과하면 hot path atomic 대신 thread-local/per-case 집계로 변경한다.

---

## WP2 — 이미 착지한 기능의 서버 238 재가격

### 목적

새 구현 전에 현재 tree에 존재하지만 238에서 미측정인 기능의 실제 가치를 확정한다.

### 대상 기능

1. `RASBERY_GPU_OUTER_GRAPH=1`
2. `RASBERY_XSLIB_CACHE=1/0`
3. `RASBERY_GPU_CMFD_COMPACT=1/0`
4. `RASBERY_GPU_PPR=1/0`
5. `--procs-per-gpu K`, MPS on/off
6. `device_launch_in_body` probe — 지원 여부 확인용, 기본 경로 채택은 별도

### 변경 파일

- 생성: `tools/run_238_adoption_matrix.py`
- 생성: `tools/adoption_gate.py`
- 수정: `tools/run_multi_gpu_batch.py`
- 수정: `docs/`에 생성될 결과 문서 — 구현 시 `docs/ADOPTION_MATRIX_SM120_<date>_KO.md`

### 실험 매트릭스

#### 단일

| Arm | 변경 변수 | 비교 기준 |
|---|---|---|
| S0 | stream outer | 기준 |
| S1 | conditional WHILE | S0 |
| S2 | GPU PPR | S0 또는 S1 mode-matched |
| S3 | WHILE + PPR | 개별 arm 통과 후만 |

#### M64

| Arm | K×W | compact | XSLIB cache | PPR | MPS |
|---|---|---|---|---|---|
| B0 | 1×64 | off | on | off | off |
| B1 | 1×64 | on | on | off | off |
| B2 | 1×64 | off | off | off | off |
| B3 | 2×32 | off | on | off | off |
| B4 | 2×32 | off | on | off | on |
| B5 | 4×16 | off | on | off | on/off |
| B6 | 8×8 | off | on | off | on/off |
| B7 | 선택 K×W | on | on | off | 선택 |
| B8 | 선택 K×W | on | on | on | 선택 |

총 선언 슬롯 `K×W=64`를 우선 고정한다. 이 제한을 풀기 전에 VRAM과 CPU budget을 별도 승인한다.

### 측정값

- `cases_per_hour`, median/p10/p90
- `width_fill`, mean/p50/p95 rendezvous width
- `padding_fraction`, bucket histogram
- `tail_idle_s`, refill latency
- `pthread_cond_wait`, `pthread_mutex_lock`
- SM/DRAM/utilization
- process startup, Init+IO, XSLIB loads/hits/waits
- PPR host/device calls와 iteration
- capture arbiter counters와 graph fallback

### 채택/중단 기준

| 기능 | 채택 기준 | 중단 기준 |
|---|---|---|
| WHILE | 단일 median ≥5% 개선, 정확성 동일 | <3% 또는 회귀 |
| XSLIB cache | cold M64 Init+IO staircase 제거, 전체 ≥3% 또는 운영상 명확한 이득 | correctness/cache stale 문제 |
| CMFD compact | M64 ≥5%, padding blocks ≥30% 감소 | graph/bucket overhead로 <3% |
| K=2 | 1×64 대비 ≥1.05× | <1.05×이면 K 확장 트랙 종료 |
| GPU PPR 1차 | Gate A/B 통과, total ≥5% 또는 PPR phase ≥30% | N1 오차 envelope 초과 |

### 완료 기준

- 각 기능은 `adopt`, `keep-opt-in`, `reject` 중 하나로 판정된다.
- “기능 flag가 켜짐”이 아니라 engagement receipt가 0보다 큰 실행만 유효하다.
- 조합 arm은 단독 채택 기능만 포함한다.

---

## WP3 — CMFD active-slot compaction 검증·하드닝·채택

### 목적

현재 선언 폭 64에 대해 실제 활성 슬롯만큼만 CMFD grid를 발사하도록 하여 padding block 비용을 제거한다.

### 현재 코드 판단

`src/CudaBICGBackend.cu`에 다음이 이미 있다.

- `RASBERY_GPU_CMFD_COMPACT`
- logical lane → physical slot map
- power-of-two 계열 bucket graph
- identity map off path
- `[RASBERY][CMFD][COMPACT]` receipt
- `tools/test_cmfd_compaction_contract.py`

따라서 신규 알고리즘보다 **전 커널 적용 완전성, graph-key 안정성, runtime B0, 238 성능**을 먼저 다룬다.

### 변경 파일

- 수정: `src/CudaBICGBackend.cu`
- 수정: `src/CudaBICGBackend.h`
- 수정: `tools/test_cmfd_compaction_contract.py`
- 생성: `test/cmfd_compaction_runtime.cpp` 또는 CUDA integration test
- 수정: `tools/parse_perf_receipts.py`

### 테스트 우선 절차

1. 모든 `__global__` CMFD kernel이 slot map macro 또는 명시적 physical slot 인자를 사용하도록 소스 계약 테스트를 강화한다.
2. compaction ON에서 non-contiguous physical slots 예: `{1,4,7,31}`를 강제하고 각 결과가 올바른 slot에 기록되는 runtime 실패 테스트를 만든다.
3. bucket 경계 1/2/3/4/7/8/15/16/31/32/63/64를 전부 테스트한다.
4. tenancy 교체 직후 old slot이 map에 남지 않는지 확인한다.
5. OFF identity path를 이전 바이너리와 `h5diff -r`한다.

### 구현 세부

- slot map은 pinned host buffer와 stable device pointer를 유지한다. graph 인자 주소는 바뀌지 않고 내용만 갱신한다.
- bucket은 `1,2,4,8,16,32,64`로 제한해 graph cache 폭증을 막는다.
- map H2D가 rendezvous critical path에 있으면 다음 중 작은 쪽을 선택한다.
  - 현재 launcher thread가 async copy 후 graph launch
  - device-side occupancy bitmap → compact map kernel
- bitmap/device compaction은 map H2D가 실제 3% 이상일 때만 구현한다.
- physical-slot 기반 status D2H와 logical grid 인덱스를 혼합하지 않는다.
- `padding_fraction`, `map_upload_us`, `bucket_cache_hits/misses`를 receipt에 추가한다.

### 정확성 게이트

```bash
python3 tools/test_cmfd_compaction_contract.py
ctest --test-dir "$BLD" -R cmfd_compaction --output-on-failure
h5diff -r "$OUT/compact_off.h5" "$OUT/compact_on.h5"
```

- B0가 목표다.
- batch 4덱 ↔ single per-deck 동일성.
- ON×3 digest 동일.
- `duplicates/stale_tenants/double_releases=0`.

### 성능 게이트

- M64 mode-matched 처리량 ≥5%.
- `physical_slot_blocks + padding_blocks` 중 padding 비율이 유의미하게 감소.
- graph instantiation 수가 bucket 수를 초과해 지속 증가하지 않음.
- map upload+rebuild 비용이 절감된 kernel dispatch의 25%를 넘지 않음.

### 롤백

`unset RASBERY_GPU_CMFD_COMPACT`. 재빌드 없이 이전 identity path로 복귀해야 한다.

---

## WP4 — 단일 GPU K-process fleet 자동 튜닝과 호스트 예산 개선

### 목적

하나의 M64 arena가 host lock과 도착 편차로 14.5개만 모으는 문제를, 여러 독립 process/arena로 분산해 실제 폭을 높인다.

### 변경 파일

- 수정: `tools/run_multi_gpu_batch.py`
- 생성: `tools/tune_single_gpu_fleet.py`
- 생성: `tools/test_fleet_tuner_contract.py`
- 수정: `src/main.cpp` — writer/worker/thread receipt 강화
- 수정 가능: `src/IoWriter.h` — 실제 worker 수·queue 상태 receipt

### 핵심 설계

```text
GPU0
 ├─ process 0, W slots, CPU set A
 ├─ process 1, W slots, CPU set B
 └─ ...
```

- 총 슬롯은 우선 64로 고정한다.
- 각 process는 독립 CMFD/Nodal arena, mutex, CUDA context를 가진다.
- MPS는 correctness 메커니즘이 아니라 서로 다른 context kernel의 동시 실행을 돕는 선택 옵션이다.
- CPU pinning은 process별 disjoint cpuset을 사용한다.

### 자동 튜너 후보

```text
K=1, W=64
K=2, W=32
K=4, W=16
K=8, W=8
```

각 후보에서 다음을 탐색한다.

- MPS off/on
- `RASBERY_BATCH_HOST_THREADS`
- `RASBERY_OMP_THREADS`
- writer thread budget
- `RASBERY_BATCH_WAIT_US`

전 탐색을 brute-force하지 않고 두 단계로 한다.

1. K/W와 MPS를 고정된 보수적 thread budget으로 평가.
2. 상위 2개 후보만 host-thread/wait 파라미터를 세부 탐색.

### 개선해야 할 현재 runner 가정

현재 runner의 `WRITER_THREADS_PER_PROCESS=8` 고정 상수는 실제 executable의 writer 구조와 mode를 반영하지 못할 수 있다. 다음으로 교체한다.

- executable receipt에서 실제 writer thread 수를 읽는다.
- `light`에서는 HDF5 writer가 불필요하므로 writer CPU budget을 0 또는 실제 JSONL writer 수로 계산한다.
- full mode에서도 queue depth와 writer busy가 낮으면 process당 thread 수를 줄인다.
- CPU budget은 `Driver workers + nested solver threads + writer + MPS daemon overhead`를 초과하지 않게 한다.

### 테스트 우선 절차

1. K process가 같은 job을 중복 claim하지 않는 concurrency test.
2. crash 후 queue resume 시 이미 완료된 output을 재실행하지 않는 test.
3. CPU set이 겹치지 않는 test.
4. `K×W×per-slot VRAM + margin`이 GPU VRAM을 넘으면 실행 전 거부하는 test.
5. mixed result mode manifest의 third field가 chunking 후 보존되는 test.

```bash
python3 tools/test_fleet_tuner_contract.py
python3 tools/test_multi_gpu_batch.py
```

### 성능 판정

튜너의 목적함수:

```text
score = median_cases_per_hour
        - tail_penalty
        - failure_penalty
```

동률 판정:

1. 더 낮은 p90 case latency
2. 더 낮은 CPU 사용량
3. 더 낮은 VRAM
4. 더 작은 K

### 채택 기준

- K=2가 1×64 대비 1.05× 미만이면 K-process 트랙을 종료한다.
- 선택 후보는 최소 3개 연속 wave에서 처리량 회귀 없이 동일 정확성 receipt를 낸다.
- MPS 사용 시 daemon lifecycle이 정상 종료되고 다른 사용자의 GPU context를 침범하지 않는다.
- 생산 설정은 장비 UUID와 driver/CUDA 버전에 keyed된 tuning result로 저장한다.

### 롤백

`--procs-per-gpu 1 --batch-width 64 --no-mps`.

---

## WP5 — FlatXS CTA-per-node 협업 커널과 XS residency

### 목적

M64 GPU 시간의 39.9%를 차지하는 `kernelFlatXs`를 구조적으로 개선한다.

### 코드에서 확인된 병목 가설

현재 `kernelFlatXs`는 **한 스레드가 한 노드 전체**를 처리한다. `flatxsSolveNode()`의 지역 작업공간은 다음과 같다.

```text
bl   : 9 × 2   = 18 doubles
bls  : 4       = 4 doubles
bm   : 9 × 78  = 702 doubles
bms  : 156     = 156 doubles
합계             880 doubles ≈ 7,040 bytes/thread
```

이 구조는 register에 들어갈 수 없으므로 local memory spill 또는 낮은 occupancy가 발생할 가능성이 높다. 다만 이것은 **정적 가설**이며, ptxas와 Nsight Compute로 먼저 확인한다.

### 변경 파일

- 생성: `src/FlatXsCooperativeKernel.cuh`
- 수정: `src/FlatXsKernel.h`
- 수정: `src/CudaXsReconBackend.cu`
- 수정: `src/CudaXsReconBackend.h`
- 수정: `src/XSSet.cpp`
- 수정: `CMakeLists.txt`
- 생성: `test/flatxs_cooperative_replay.cu`
- 생성: `tools/test_flatxs_cooperative_contract.py`
- 생성: `tools/test_flatxs_residency_contract.py`

### 단계 A — 자원 사용 증명

빌드에 해당 TU만 ptxas verbose를 추가한다.

```cmake
set_source_files_properties(src/CudaXsReconBackend.cu PROPERTIES
  COMPILE_OPTIONS "--fmad=false;--ptxas-options=-v")
```

측정:

```bash
nsys profile -o "$OUT/flatxs_nsys" --trace=cuda,nvtx,osrt \
  "$BLD/RASBERY" --jobs "$OUT/m64jobs.txt" --batch-mode 64 --result light

ncu --target-processes all --set full \
  --kernel-name regex:kernelFlatXs \
  -o "$OUT/flatxs_ncu" \
  "$BLD/RASBERY" --jobs "$OUT/profile_jobs.txt" --batch-mode 4 --result light
```

기록할 항목:

- registers/thread
- local memory bytes/thread
- local load/store throughput
- achieved occupancy
- eligible warps/SM
- branch divergence
- instruction mix와 stall reason
- L1/L2 hit rate

#### 분기 기준

- local memory traffic가 높으면 CTA 협업 커널을 진행한다.
- spill이 거의 없다면 우선 stream upload/download, branch divergence, serial isotope accumulation을 대상으로 다시 설계하고 CTA 전환을 자동 채택하지 않는다.

### 단계 B — CTA-per-node B0 커널

선택 구조:

- 한 CTA가 한 노드를 담당한다.
- 128 또는 256 threads/CTA를 후보로 측정한다.
- 880 double 작업공간을 shared memory에 둔다. 약 7 KiB/CTA이므로 다중 CTA residency가 가능하다.
- 각 lane은 고정된 element ordinal을 담당한다.
- 각 element의 delta 적용 순서와 Horner 순서는 기존 CPU/GPU 본문과 동일하다.
- delta stream의 `DeltaMeta`, `x`, `scale`을 CTA가 한 번 읽어 shared memory에 배포한다.
- macro XS 재구축에서 각 output을 한 lane이 맡고 isotope index를 0→NISO-1 순서로 직렬 누적한다.
- atomics와 tree reduction을 사용하지 않는다.
- TU는 계속 `--fmad=false`; 기존 `FLATXS_FORMS`를 그대로 사용한다.

의사 코드:

```cpp
__global__ void kernelFlatXsCooperative(FlatXsView v) {
    const int node_ordinal = blockIdx.x;
    extern __shared__ double work[];

    cooperativeGatherReference(v, node_ordinal, work);
    __syncthreads();

    for (int s = node_stream_begin; s < node_stream_end; ++s) {
        cooperativeApplyOneDeltaInOriginalOrder(v, s, work);
        __syncthreads();
    }

    cooperativeRefreshLightIsotopes(v, node_ordinal, work);
    __syncthreads();

    cooperativeScatterAndRebuildMacroSerialIso(v, node_ordinal, work);
}
```

`__syncthreads()` 횟수가 delta 수만큼 늘어날 수 있으므로, delta count가 작은/큰 노드를 분리한 specialization도 후보로 둔다. 단, 먼저 단순한 B0 구현으로 정확성을 확보한다.

### 단계 C — replay 및 full-deck gate

1. 기존 `RASBERY_FLATXS_DUMP` capture를 old kernel과 cooperative kernel에 재생한다.
2. 모든 출력 element를 0 ULP 비교한다.
3. unrodded/rodded, branch delta 있음/없음, spectral history, Gd 모델 fixture를 포함한다.
4. 4덱 batch와 single per-deck를 비교한다.

```bash
ctest --test-dir "$BLD" -R flatxs --output-on-failure
python3 tools/test_flatxs_cooperative_contract.py
h5diff -r "$OUT/flatxs_old.h5" "$OUT/flatxs_coop.h5"
```

### 단계 D — XS canonical residency

현재 `solveFlatXs()`는 기본적으로 한 호출당 mic/lmp 대량 배열을 포함해 약 61 MB를 host로 돌려보낸다. `RASBERY_FLATXS_SKIP_MICX_DL=1`은 consumer 검증 없이 위험하다고 코드가 명시한다. 이를 production 기능으로 바로 승격하지 않는다.

다음 consumer audit를 먼저 수행한다.

| 데이터 | 가능한 host consumer | 처리 |
|---|---|---|
| `_micx/_lmpx` | depletion, rodded fallback, export, PPR 일부 | GPU consumer 이식 또는 명시적 materialize |
| `_xs` | CMFD/Nodal/PPR, 출력 | canonical device pointer 공유 |
| `_iden` | Xe/depletion/output | generation 기반 ownership |
| reference arrays | FlatXS | process/cohort device cache |

새 ownership 구조:

```cpp
enum class XsRegion { Mic, Lumped, Macro, Isotope, Reference };
enum class XsOwner  { Host, FlatXs, Xe, Depletion, Materialized };
```

- producer가 region owner와 generation을 갱신한다.
- host consumer 직전에만 해당 region을 materialize한다.
- light/full-exact GA 경로는 최종 scalar에 필요한 region만 가져온다.
- CPU fallback이 가능한 일반 모드에서는 기존 full download를 유지한다.
- GPU full에서는 audit되지 않은 host consumer가 나타나면 fail-closed한다.

### 성능 게이트

| 항목 | 최소 기준 |
|---|---:|
| FlatXS local memory traffic | 기존 대비 90% 이상 감소 또는 병목 원인에 맞는 동등한 지표 개선 |
| `kernelFlatXs` 시간 | 30% 이상 감소 |
| M64 전체 처리량 | 10% 이상 증가 |
| 결과 | B0 |
| VRAM | 기존 M64 guard 이내 또는 증가량을 명시하고 K/W 튜너에 반영 |

30% kernel 단축은 전체 처리량 약 13%의 Amdahl 여지를 준다. 전체 개선이 5% 미만이면 복잡도 대비 이득이 부족하므로 old kernel을 기본으로 유지한다.

### 롤백

- `RASBERY_GPU_FLATXS_COOP=0`
- `RASBERY_GPU_XS_CANONICAL=0`

두 기능을 독립적으로 되돌릴 수 있어야 한다.

---

## WP6 — GPU PPR의 반복 종료·재구성·배치 구조 개선

### 목적

상태점당 CPU 바닥을 줄이고, 현재 GPU PPR에서 남아 있는 반복별 host sync와 대량 최종 D2H를 제거한다.

### 현재 경로

한 상태점에서 현재 GPU PPR은 다음을 수행한다.

1. `phif/phis/jnet/xsdf/xsrf/xsnf/xssm/chif/crdf`를 H2D.
2. reset kernel 5개.
3. Picard iteration마다 약 8개 kernel.
4. 4×256 corner partial을 D2H하고 `cudaStreamSynchronize`.
5. host가 네 합을 순서대로 접고 convergence를 판정.
6. 종료 후 `p/a/c/bt/phic/q/l` 일곱 배열을 D2H하고 sync.
7. host `reconstructPinPower`가 Fq/FΔH 및 pin map을 생성.

현재 오류 시 host PPR로 fail-open한다. GPU full 계약에서는 허용하지 않는다.

### 변경 파일

- 수정: `src/CudaPprBackend.cu`
- 수정: `src/CudaPprBackend.h`
- 수정: `src/PPR.cpp`
- 수정: `src/PPR.h`
- 생성: `src/PprReconstructionKernel.cuh`
- 생성 가능: `src/CudaPprArena.{h,cu}`
- 수정: `src/GpuCanonicalState.h`
- 생성: `tools/test_ppr_device_loop_contract.py`
- 생성: `tools/test_ppr_reconstruction_gate.py`
- 생성: `tools/test_ppr_batch_isolation.py`

### 단계 A — 현재 GPU PPR 정확성·비용 gate

- 서버 238에서 GPU PPR OFF/ON을 동일 arm으로 측정한다.
- `ppr_host_calls`, `ppr_device_calls`, iteration, wall, upload/download bytes를 receipt에 추가한다.
- Fq/FΔH, pin map RMS/max, PPR coefficient arrays를 비교한다.

채택 전 조건:

- N1 Gate A/B 통과.
- 같은 GPU/동일 run에서 digest 반복.
- `ppr_device_calls == statepoints`, `ppr_host_calls == 0` in GPU full.

### 단계 B — device convergence와 conditional graph

host fold와 같은 association을 보존하기 위해 다음을 사용한다.

1. `kCornerPartials`는 현행 고정 256 chunk를 유지한다.
2. `kCornerFoldAndCheck<<<1,4>>>`에서 corner별 한 thread가 chunk 0→255 순서로 직렬 합산한다.
3. 네 값의 relative change와 tolerance 판정을 device에서 수행한다.
4. `halt`, `iterations_done`, previous sums를 device scalar block에 둔다.
5. Picard iteration body를 CUDA Graph conditional WHILE 또는 fixed-capacity graph+halt guard로 실행한다.
6. host는 statepoint 종료 후 한 번만 status를 읽는다.

이 경로는 CPU와 GPU의 division/absolute-value 차이 가능성 때문에 N1로 분류한다. 현재 GPU PPR 자체도 device `exp`와 reduction partition 때문에 N1이다.

### 단계 C — PPR canonical input

PPR backend가 독자 `d_phif/d_phis/d_jnet/d_xs*`를 소유하고 매 상태점 업로드하는 구조를 다음으로 바꾼다.

- CMFD/Nodal/XS canonical buffer를 borrowed pointer로 채택한다.
- canonical pointer가 없는 일반 모드만 기존 H2D path를 사용한다.
- cross-stream event로 producer→PPR 순서를 보장한다.
- PPR이 input을 수정하지 않는 region은 owner를 바꾸지 않는다.
- geometry와 `crdf`는 cohort/device cache로 유지한다.

### 단계 D — `reconstructPinPower` GPU 이식

- `buildQuadratureTable()` 결과는 geometry cohort별 immutable device table로 한 번 업로드한다.
- one node/group 또는 one pin tile per CTA 후보를 ncu로 비교한다.
- 기존 host loop의 pin ordinal, quadrature order, Legendre order를 유지한다.
- `exp` 차이 때문에 N1 Gate로 관리한다.
- light output에서는 device에서 Fq/FΔH/max 위치만 reduction하고 pin map을 D2H하지 않는다.
- full output에서만 최종 pin map을 output boundary에서 materialize한다.
- 중간 coefficient 일곱 배열은 다른 host consumer가 없다면 D2H하지 않는다.

### 단계 E — PPR batch arena는 조건부

앞 단계 이후 M64 프로파일에서 PPR이 전체 시간의 10% 이상이면 `CudaPprArena`를 구현한다.

- geometry cohort당 한 arena.
- slot별 pointer table과 active-slot compaction.
- graph bucket은 CMFD와 동일한 1/2/4/8/16/32/64 정책.
- one backend per Driver의 64개 stream/allocation을 대체한다.

10% 미만이면 batch arena는 구현하지 않고 per-slot backend를 유지한다.

### 성능 게이트

| 단계 | 기준 |
|---|---|
| device convergence | PPR phase 25% 이상 단축, iteration 동일 또는 N1 envelope 내 |
| canonical input | PPR H2D bytes 80% 이상 감소 |
| device reconstruction | PPR 전체 40% 이상 단축 |
| 전체 M64 | 각 큰 단계별 5% 이상 또는 결합 10% 이상 |
| 정확성 | Fq/FΔH·pin Gate 통과, 결정론 |

### 롤백

- `RASBERY_GPU_PPR=0`
- `RASBERY_GPU_PPR_DEVICE_LOOP=0`
- `RASBERY_GPU_PPR_RECON=0`
- `RASBERY_GPU_PPR_CANONICAL=0`

GPU full에서는 롤백 arm을 켠 상태로 실행하지 않고 명시적 configuration error를 낸다.

---

## WP7 — 단일 케이스 launch/sync 축소와 Xe transaction

### 목적

단일 경로의 작은 커널 발사와 host observation을 줄이고, 배치에서 26.9%를 차지하는 Xe 단계의 분절된 device/host 왕복을 줄인다.

### 변경 파일

- 수정: `src/CudaBICGBackend.cu`
- 수정: `src/CudaXsReconBackend.cu`
- 수정: `src/Driver.h`
- 수정: `src/XeKernel.h`
- 생성: `tools/test_xe_transaction_contract.py`
- 수정: `tools/test_telemetry_neutrality.py`

### 단계 A — conditional WHILE 채택 판정

이미 구현된 `RASBERY_GPU_OUTER_GRAPH=1`을 서버 238에서 stream arm과 교대 5쌍 비교한다.

필수 receipt:

- graph instantiation
- graph launches / iterations per launch
- in-body sync
- overrun
- warmup miss
- wall

채택 기준은 단일 5% 이상 개선이다. 배치에서는 device outer가 기존에 중립이므로 별도 3% 기준을 적용하고, 단일 ON/배치 OFF의 mode-dependent default를 허용한다.

### 단계 B — CMFD graph node census와 안전한 fusion

- nsys graph trace에서 node별 duration과 launch count를 집계한다.
- 동일 index domain, 동일 slot, producer-consumer가 인접한 elementwise kernel만 fusion한다.
- 한 thread가 이전 kernel A의 표현을 순서대로 수행한 뒤 kernel B의 표현을 수행한다.
- reduction 순서, red-black sweep 순서, barrier 위치를 바꾸지 않는다.
- 각각 B0 replay gate를 만든다.

3% 미만의 wall 이득인 fusion은 유지보수 비용 때문에 채택하지 않는다.

### 단계 C — Xe device transaction

현재 Xe 경로는 evaluate, dot, host 2×2 solve/safety, candidate, commit 사이에 여러 host sync가 있다. 다음 transaction을 opt-in으로 추가한다.

```text
stage once
 → evaluate
 → fixed-partition dots
 → one-thread deterministic 1×1/2×2 solve and safety gate
 → candidate or Picard select
 → commit
 → one status/materialization
```

- fixed partition과 serial fold는 유지한다.
- `kXeAndersonControl` 한 block이 host와 같은 pivot/fallback/safety 순서를 수행한다.
- full transaction을 graph로 캡처한다.
- state arrays가 device resident이면 `_xs/_iden`의 중간 D2H를 생략한다.
- host reference path는 validation과 feature-off용으로 유지한다.

### 정확성/성능 게이트

- N1: MASTER와 기존 device Xe envelope 이내.
- AA/Picard 선택 횟수, rejected candidate, fallback 사유가 mode-matched reference와 일치하거나 차이를 설명한다.
- 배치 Xe phase 20% 이상 단축 또는 전체 M64 5% 이상.
- 단일에서 회귀하면 mode-dependent default로 batch만 ON 가능.

---

## WP8 — 장수명 GA evaluator

### 목적

GA 세대/청크마다 반복되는 process image, CUDA context, arena stand-up, graph capture, immutable library 준비를 제거한다.

### 중요한 전제

현재 `--batch-mode` 한 invocation 안에서는 여러 job이 같은 process를 공유하지만, dispatcher는 queue chunk마다 새로운 RASBERY process를 실행한다. evaluator의 비교 기준은 “케이스당 process 1개”가 아니라 **현재 chunked M64 launcher**다.

### 변경 파일

- 생성: `src/EvaluatorServer.h`
- 생성: `src/EvaluatorServer.cpp`
- 확장: `src/EvaluatorContext.h`
- 수정: `src/main.cpp`
- 수정: `src/Driver.h`
- 수정: `src/Geometry.{h,cpp}`
- 수정: `src/XSSet.{h,cpp}`
- 수정: `src/IoWriter.h`
- 수정: `src/CudaBICGBackend.{h,cu}`
- 수정: `src/CudaXsReconBackend.{h,cu}`
- 생성: `tools/evaluator_client.py`
- 생성: `tools/test_evaluator_protocol.py`
- 생성: `tools/test_evaluator_case_isolation.py`
- 생성: `tools/test_evaluator_soak.py`

### 단계 1 — process만 지속, case object는 매번 재생성

가장 먼저 수명 변경을 최소화한다.

- `--evaluator-jsonl` 모드 추가.
- stdin으로 wave 요청을 받고 stdout에 wave receipt를 반환한다.
- ProcessContext/XSLIB/device library/CUDA context/arena graph cache는 process 종료까지 유지한다.
- 각 job은 기존처럼 새 `Driver`/`CaseContext`를 생성·파괴한다.
- wave 종료 시 writer queue는 drain하되 arena/context는 destroy하지 않는다.
- shutdown 명령에서만 arena와 CUDA resource를 명시적으로 해제한다.

프로토콜 예:

```json
{"op":"wave","wave_id":17,"jobs_manifest":"/path/w17.jobs","batch_width":32,"result_mode":"light","physics_fidelity":"full_exact","contract_hash":"..."}
{"op":"shutdown"}
```

응답 예:

```json
{"wave_id":17,"jobs":256,"ok":256,"failed":0,"cases_per_hour":912.4,"process_reused":true,"xslib_loads":1,"graph_reuses":18422}
```

### 단계 2 — CohortContext 도입

`Geometry::Initialize`가 bare allocation을 재초기화할 수 없는 현재 구조를 먼저 정리한다.

- Geometry ownership을 RAII container로 바꾼다.
- immutable topology와 candidate-dependent loading map을 분리한다.
- `CohortKey`에 다음을 포함한다.
  - topology/mesh hash
  - geometry dimensions
  - symmetry/boundary mode
  - XS library **content digest**와 schema version
  - energy group 수
  - PPR geometry parameters
- cohort당 immutable geometry maps, PPR quadrature, device library/graph를 유지한다.
- mutable `_core/_comp/_batch/is_fuel`과 case state는 reset API로 채운다.

### 단계 3 — mutable buffer reuse

- `CaseContext::reset(candidate)`를 추가한다.
- nxyz-sized XS/isotope/work arrays는 capacity를 유지하고 값만 명시적으로 초기화한다.
- reset coverage를 자동 검사하기 위해 debug build에서 generation poison을 사용한다.
- arena slot은 case 시작에 acquire, 종료에 release한다.

### XSLIB cache key 강화

장수명 process에서는 `(path,size,mtime,ng)`만으로 오래된 파일을 오인할 위험이 있다. 다음을 추가한다.

- file content SHA256 또는 빠른 content digest
- library schema/version
- isotope registry digest
- optional inode/file-id

파일이 바뀌면 새 entry를 만들고, 기존 실행 중인 cohort는 immutable old entry를 계속 사용한다.

### cross-case isolation 테스트

1. `A → B → A` 실행에서 두 A의 모든 결과와 digest가 동일.
2. 같은 64개 케이스를 20개 random order로 실행해 per-case 결과 동일.
3. 중간에 의도적 CUDA/입력 실패를 넣은 뒤 다음 케이스 정상.
4. 서로 다른 geometry cohort를 번갈아 실행해 cache cross-talk 없음.
5. 10,000 case soak에서 RSS/VRAM이 warm plateau 이후 단조 증가하지 않음.
6. process 종료 시 cuda-memcheck/compute-sanitizer leak 또는 invalid access 없음.

### 성능 게이트

- chunked launcher 대비 process/stand-up 비용이 전체 wall의 1% 이하.
- 전체 처리량 5% 이상 개선 또는 GA 세대 간 startup jitter가 유의하게 감소.
- XSLIB loads는 content/cohort 수와 같고 case 수와 무관.
- graph instantiation은 shape/bucket 수에 bounded.

### 실패 격리

한 job의 exception은 해당 job response로 변환한다. process-global CUDA 상태가 오염된 경우에만 worker가 `fatal=true`를 반환하고 dispatcher가 새 worker를 시작한다. 조용한 fallback은 금지한다.

---

## WP9 — 잔여 상태점 CPU 바닥: CRAM/Depletion/TH/Search

### 목적

FlatXS와 PPR 이후 다시 프로파일하여 남은 per-statepoint floor를 실제 비중 순으로 줄인다.

### 원칙

보고서는 상태점 바닥을 PPR/CRAM/FlatXS/TH로 분해하지만, 앞선 작업 후 비중이 크게 바뀐다. 따라서 이 WP는 **재프로파일에서 10% 이상인 항목만** 구현한다.

### 변경 파일 후보

- `src/XSSet.cpp`
- `src/Driver.h`
- `src/Scheduler.h`
- `src/CudaXsReconBackend.cu`
- 신규 가능: `src/CudaDepletionBackend.{h,cu}`
- 신규: `tools/case_cost_profile.py` 확장

### 단계 A — phase receipt 재분해

상태점마다 다음을 분리한다.

```text
th
precompute/reference
flatxs
xe
cmfd/nodal
ppr
cram/depletion
search/settle/control
output
```

wall timing 실행에는 receipt를 끄고, 별도 telemetry 실행으로 귀속한다.

### 단계 B — CRAM/depletion 조건부 GPU 이식

CRAM/depletion이 statepoint wall의 10% 이상이면:

- `DepletionWorkspace`를 case 수명 동안 재사용한다.
- 동일 크기의 독립 node/isotope system을 batched kernel로 처리한다.
- 작은 8차 CRAM의 행렬 연산 순서를 host reference와 맞춘 B0/N1 분류를 먼저 결정한다.
- isotope별 독립성이 없는 coupling을 임의로 병렬화하지 않는다.
- `_iden` canonical owner와 generation을 통합한다.

5% 미만이면 GPU 이식하지 않고 CPU vectorization/NUMA/allocator 제거만 검토한다.

### 단계 C — TH

TH가 10% 이상이면:

- 이미 process-static인 table parse는 유지한다.
- per-node lookup을 SoA/vectorized loop로 바꾸고 false sharing을 제거한다.
- GPU 이식은 XS/PPR consumer와 device state가 이미 이어질 때만 검토한다.

### 단계 D — search trial 감소

현재 search bucket이 outer의 약 15.3%, trial이 137회다.

- 이전 상태점의 boron slope와 동일 family의 accepted case를 이용해 초기 bracket을 제안한다.
- bracket은 물리적으로 유효한 범위로 clamp한다.
- secant 실패/비단조/다중근 징후에서 즉시 기존 cold bracket으로 복귀한다.
- 최종 strict convergence 조건은 바꾸지 않는다.

이 작업은 initial/search outer 수를 줄이는 N1/A2 알고리즘 레버로 별도 flag와 receipt를 사용한다.

### 채택 기준

- 대상 phase 자체 25% 이상 단축 또는 전체 5% 이상.
- strict result와 Gate 통과.
- fallback/cold retry 횟수와 실패 이유를 기록.

---

## WP10 — GA 수준 계산량 감소: 중복 캐시, warm-start, 승격 파이프라인

### 목적

커널 최적화보다 큰 총 캠페인 이득을 얻되, acceptance 결과는 strict evaluator로 보장한다.

### 변경 위치

저장소 내 GA controller 위치에 맞춰 구현하되, evaluator 쪽에는 다음 API를 추가한다.

- 생성: `src/CandidateKey.h`
- 생성: `src/CandidateKey.cpp`
- 생성: `tools/candidate_key.py`
- 생성: `tools/test_candidate_key_contract.py`
- evaluator protocol에 `candidate_key`, `parent_key`, `warm_start_key` 추가

### 10.1 canonical duplicate key

키에는 **실제 계산 결과에 영향을 주는 모든 입력**을 포함한다.

```text
core loading map after symmetry canonicalization
assembly IDs and burnup/isotope state
cycle/schedule and statepoint policy
rod/bank/thermal-hydraulic conditions
XS library content digest and schema
code SHA and build/FP flags
physics fidelity and all trajectory-affecting env strings
geometry/topology hash
initial condition and warm-start provenance
```

- 같은 물리 케이스가 대칭 표현만 다른 경우 canonicalization으로 하나의 key가 되게 한다.
- hash collision 방지를 위해 key payload도 저장하고 hit 시 byte compare한다.
- cache hit의 scalar 결과에는 원본 run contract와 digest를 함께 반환한다.
- 1% random hit를 실제 재계산해 silent corruption을 감시한다.

### 10.2 warm-start

대상은 initial bucket과 search 초기 구간이다.

- 동일 cohort에서 가장 가까운 accepted parent의 flux/boron/Xe state를 제안한다.
- 거리 metric은 loading difference, burnup, operating condition을 분리한다.
- warm start 실패, iteration 증가, 비물리 값에서 cold start로 즉시 재실행한다.
- acceptance-eligible elite는 cold 또는 검증된 strict rerun으로 확인한다.
- 결과 cache key에 warm-start가 결과 궤적에 영향을 줄 수 있으면 provenance를 포함한다.

### 10.3 multi-fidelity 승격

세 lane을 명확히 분리한다.

1. `coarse10` — screening only.
2. `A2` — 별도 수렴 정책, 프로젝트 승인 범위 내 ranking.
3. `strict_full` — 최종 fitness와 acceptance.

승격 규칙:

- coarse 상위 후보뿐 아니라 uncertainty/boundary candidate를 포함한다.
- generation elite와 archive 진입 후보는 strict rerun.
- screening 결과를 strict cache에 쓰지 않는다.
- ranking inversion rate와 promotion recall을 세대별 보고한다.

### 성능/정확성 판정

- duplicate hit rate가 20%라면 이론상 1.25×지만, 실제 hit rate를 먼저 측정한다.
- warm-start는 initial/search outer 20% 이상 감소가 목표다.
- strict rerun과 최종 선택 결과가 동일해야 한다.
- multi-fidelity는 최종 Pareto/elite recall 기준을 사전 정의한다.

---

## WP11 — 장기 안정성, 기본값 승격, 동결

### 목적

개별 benchmark 이득을 production default로 안전하게 승격한다.

### 변경 파일

- 수정: `test/reference/validation_baseline_manifest_v3.json`
- 생성: `test/reference/validation_baseline_manifest_v4.json`
- 생성: `tools/run_production_soak.py`
- 생성: `docs/GPU_RASBERY_V4_FREEZE_<date>_KO.md`
- 수정: `docs/README_CAMPAIGN.md`

### 필수 soak

1. 64덱 동일 cohort 10,000 case.
2. 서로 다른 geometry/library cohort 교차.
3. full/light mixed manifest.
4. 의도적 bad deck, CUDA OOM 경계, output permission failure.
5. evaluator 재시작·queue resume.
6. K-process/MPS daemon cleanup.
7. graph capture stress와 allocation overlap.

### 종료 시 0이어야 하는 receipt

```text
duplicates
stale_tenants
double_releases
alloc_in_capture
captures_unwound
graph_fallbacks          # GPU full
cmfd/nodal/flatxs/ppr fallbacks
queue duplicate claims
output collisions
cross_case_digest_mismatch
```

### 기본값 승격 규칙

- B0 기능: 238 gate와 soak 통과 후 default ON 후보.
- N1 기능: 정확성 envelope와 deterministic digest, project acceptance 승인 후 ON.
- A2/coarse: mode-specific이고 strict default를 대체하지 않음.
- 단일 이득/배치 회귀 기능은 execution-mode dependent default 허용.
- 모든 기능은 재빌드 없는 env rollback을 유지한다.

### Definition of Done

- `ctest` 및 모든 `tools/test_*.py`에서 신규 실패 0.
- 기존 사전 실패는 원인과 baseline을 명시하고 새 failure와 구분.
- 서버 238 GPU0에서 frozen benchmark manifest와 raw evidence 보존.
- strict/full, strict/light, A2/full, A2/light 처리량을 분리 보고.
- 최종 default arm이 MASTER Gate B와 pin gate를 통과.
- 장수명 evaluator 10,000 case에서 메모리 안정.
- 생산 설정과 롤백 명령이 한 문서에 고정.

---

## 8. 종합 벤치마크 설계

### 8.1 기본 환경

```bash
export CUDA_VISIBLE_DEVICES=0
export RASBERY_GPU=1
export RASBERY_GPU_CMFD_SWEEP=1
export RASBERY_GPU_CMFD_RESIDENT_SINGLE=1
export RASBERY_GPU_NODAL=1
export RASBERY_GPU_NODAL_FULL=1
export RASBERY_GPU_XSRECON=1
export RASBERY_GPU_FLATXS=1
export RASBERY_GPU_WIEL_FOLD=chunked
export RASBERY_GPU_XE=1
export RASBERY_IO_WRITER=thread
```

strict와 A2 변수는 별도 파일로 둔다.

```bash
# strict.env
unset RASBERY_STAGED_FLUX_TOL
unset RASBERY_STAGED_XE_TOL
unset RASBERY_STAGED_LOOSE_SETTLE

# a2.env
export RASBERY_STAGED_FLUX_TOL=50
export RASBERY_STAGED_XE_TOL=1000
export RASBERY_STAGED_LOOSE_SETTLE=1
```

### 8.2 반복 규칙

```text
warm-up 1
A1 B1 A2 B2 A3 B3 A4 B4 A5 B5
median + p10 + p90
```

GPU clock/temperature가 한 쌍 내에서 크게 다르면 쌍을 무효 처리한다.

### 8.3 프로파일 계층

1. **항상:** wall, receipts, dmon.
2. **후보가 3% 이상:** nsys 180초 또는 대표 wave.
3. **커널 작업:** ncu를 M4 또는 축소 representative workload에 적용.
4. **최종 후보:** M64 full wave와 장기 soak.

### 8.4 처리량 외 필수 지표

| 범주 | 지표 |
|---|---|
| 정확성 | digest, h5diff datasets, pcm, ppm, AO, pin RMS/max, Fq/FΔH |
| GPU | SM, DRAM, occupancy, local memory, kernel count, graph nodes |
| CUDA API | memcpy count/bytes, sync count/time, graph launch/instantiate |
| host | cond_wait, mutex, CPU utilization, NUMA remote, context switches |
| batch | width_fill, padding, refill, tenancy, tail |
| process | exec, pre-drive, drive, post-drive, cache loads/hits |
| operation | output bytes/case, queue depth, writer busy/block |

---

## 9. 정량적 채택 게이트 요약

| 개선 | 최소 성능 기준 | 정확성 등급 | 비고 |
|---|---:|---|---|
| 계약 계측 | 오버헤드 <1% | B0 | 필수 기반 |
| conditional WHILE | 단일 +5% | B0 목표 | 배치 별도 판정 |
| XSLIB cache | M64 +3% 또는 cold staircase 제거 | B0 | 운영 이득 포함 |
| CMFD compaction | M64 +5% | B0 | padding ≥30% 감소 |
| K-process | K2 +5% | B0 | 미달 시 트랙 종료 |
| FlatXS cooperative | kernel −30%, M64 +10% | B0 | spill 증명 선행 |
| FlatXS residency | transfer −80%, 전체 +5% | B0/N1 | consumer audit 필수 |
| PPR device loop | PPR −25% | N1 | host sync 제거 |
| PPR reconstruction | PPR 전체 −40%, 전체 +5% | N1 | Fq/FΔH gate |
| Xe transaction | Xe phase −20% 또는 전체 +5% | N1 | fixed partition 유지 |
| persistent evaluator | chunked 대비 +5%, process cost <1% | B0 | cross-case soak |
| search/warm-start | 대상 outer −20% | N1 | cold fallback |

최종 1,600–2,100 c/h/GPU는 기존 보고서의 **전 충실도 공학 목표 범위**이며 보장값이 아니다. 각 레버는 겹치므로 측정 없이 개선율을 곱하지 않는다.

---

## 10. 위험 원장과 완화책

| 위험 | 발생 위치 | 완화 |
|---|---|---|
| light 출력이 screening으로 오분류 | `main.cpp` | fidelity/result enum 분리, parser mismatch 거부 |
| GPU flag가 켜졌지만 fallback | 모든 backend | GPU full fail-closed와 engagement receipt |
| active map이 stale tenant를 가리킴 | CMFD/PPR arena | tenancy generation 포함, release 시 poison, runtime map test |
| graph가 잘못된 pointer/mask로 replay | CUDA graph cache | key에 topology/materialize/bucket/generation 포함, bounded cache |
| capture 중 allocation/device sync | batch startup/refill | 기존 capture arbiter 유지, stress gate |
| FlatXS cooperative가 산술 순서를 변경 | cooperative kernel | element owner 고정, isotope serial order, 0-ULP replay |
| XS download 제거 후 host consumer 누락 | residency | consumer audit, owner/generation, GPU full에서 fail-closed |
| PPR device convergence가 iteration을 변경 | PPR | fixed chunk/serial fold, iteration receipt, N1 gate |
| evaluator cross-case 상태 누출 | Context reuse | A-B-A, random order, poison reset, 10k soak |
| 장수명 XSLIB stale cache | XSLIB | content digest/schema key |
| K-process가 CPU/writer를 과구독 | runner | disjoint cpuset, 실제 writer receipt, 자동 budget |
| MPS daemon 잔류 | runner | scoped start/stop, finally cleanup, PID/pipe isolation |
| A2/coarse 숫자가 strict로 발표됨 | docs/parser | mode-aware manifest와 acceptance flag |
| 출력 파일 충돌 | manifest/refill | canonical path uniqueness, temp+atomic rename |

---

## 11. 수행하지 말아야 할 작업

1. **M96/M128로 선언 폭만 확대하지 않는다.** 이미 실제 폭이 감소하고 처리량이 flat/회귀했다.
2. **배치에서 outer segment budget 8을 재도입하지 않는다.** 폭 M launch가 폭 1 launch M개로 붕괴해 큰 회귀가 측정됐다.
3. **persistent GPU scheduler를 다시 주 경로로 만들지 않는다.** W0 barrier kill criterion을 통과하지 못했다.
4. **interim Xe를 되살리지 않는다.** Anderson이 외삽하는 사상과 맞지 않아 구조적으로 악화됐다.
5. **FP32를 핵심 레버로 가정하지 않는다.** 기존 M64 이득은 2.6% 수준이었다.
6. **모든 배열을 무조건 GPU 상주시킨 뒤 host access를 막지 않는다.** consumer audit와 ownership 없이 `skip download`를 production화하면 silent wrong answer 위험이 있다.
7. **새 compaction을 처음부터 다시 쓰지 않는다.** 현재 소스 구현을 검증하고 부족한 부분만 고친다.
8. **다중 GPU aggregate 수치를 단일 GPU 향상으로 보고하지 않는다.** 본 계획의 acceptance baseline은 물리 GPU GPU0 한 장이다.
9. **처리량 배수를 단순 곱하지 않는다.** K-process, compaction, PPR/FlatXS offload는 서로의 병목 비중을 바꾼다.
10. **정확성 계측과 timing을 같은 run에서 섞지 않는다.** telemetry 자체의 중립성 gate가 있어도 wall 측정은 별도다.

---

## 12. 권장 실행 순서와 의사결정 지점

### Stage 1 — 계약과 사실 확정

1. WP0 baseline harness.
2. WP1 result/fidelity + GPU full.
3. WP2 238 adoption matrix.

**결정:** WHILE, XSLIB, compact, PPR, K-process 중 실제 채택 후보를 확정한다.

### Stage 2 — 빈 슬롯과 큰 커널

1. WP3 CMFD compaction 하드닝.
2. WP4 K-process tuner.
3. 두 기능의 조합 A/B.
4. WP5 FlatXS 자원 증명과 cooperative kernel.

**결정:** 선택 K×W, compact default, FlatXS kernel default를 장비 profile로 고정한다.

### Stage 3 — 상태점 바닥

1. WP6 PPR device loop.
2. PPR canonical input.
3. PPR reconstruction.
4. WP7 Xe transaction/단일 launch fusion.
5. 재프로파일 후 WP9의 실제 상위 phase만 구현.

**결정:** full-fidelity single 및 M64의 새로운 `(c,d)` 비용 모델을 다시 적합한다.

### Stage 4 — GA 서비스화

1. WP8 evaluator 단계 1.
2. cohort/context refactor와 buffer reuse.
3. WP10 duplicate/warm-start/multi-fidelity.
4. WP11 soak와 v4 freeze.

**결정:** production default와 rollback matrix를 최종 승인한다.

---

## 13. 구현 완료 후 최종 보고서에 반드시 포함할 표

### 13.1 단일

```text
arm | fidelity | result | wall | statepoints | outers | PPR | FlatXS | CUDA syncs | digest | gate
```

### 13.2 병렬

```text
K×W | MPS | compact | fidelity | result | c/h | width_fill | padding | tail | CPU wait | SM | VRAM | failures
```

### 13.3 커널

```text
kernel | old time | new time | calls | local bytes | regs | occupancy | speedup | exactness
```

### 13.4 운영

```text
cases | process restarts | xslib loads | graph instantiations | peak RSS | peak VRAM | queue duplicates | output bytes
```

### 13.5 최종 주장 형식

- “GPU0 한 장, strict/full, SHA X, median N회에서 Y c/h.”
- “A2/light는 별도 arm에서 Z c/h.”
- “다중 GPU aggregate는 GPU 수와 효율을 명시.”
- “각 기능의 단독/결합 개선율과 정확성 등급을 명시.”

---

## 14. 소스 근거 색인

| 근거 | 파일 |
|---|---|
| GA evaluator 비용 모델·레버 L1–L8 | `docs/GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md` |
| 최신 성능·커널·코드 구조 종합 | `docs/GPU_RASBERY_PERFORMANCE_AND_ARCHITECTURE_REPORT_20260830_KO.md` |
| batch/refill/result/physics receipt | `src/main.cpp` |
| process/case 수명 seam | `src/EvaluatorContext.h` |
| immutable host library cache | `src/XsLibrary.h`, `src/XSSet.cpp` |
| CMFD graph/arena/compaction | `src/CudaBICGBackend.cu` |
| FlatXS/Xe/Nodal backend | `src/CudaXsReconBackend.cu` |
| FlatXS node 작업공간·산술 순서 | `src/FlatXsKernel.h` |
| GPU PPR 반복·D2H·fail-open | `src/CudaPprBackend.cu` |
| host PPR 및 reconstruction | `src/PPR.cpp`, `src/PPR.h` |
| K-process/MPS/queue/host budget | `tools/run_multi_gpu_batch.py` |
| device phase scheduler 실험 경계 | `src/GpuPhaseScheduler.h`, `src/GpuPhaseScheduler.cu` |
| 기존 정확성/계기 계약 | `tools/test_*.py`, `test/reference/validation_baseline_manifest_v3.json` |

---

## 15. 최종 권고

가장 먼저 수행할 실제 코딩은 대규모 evaluator 재작성이나 새로운 persistent kernel이 아니다. 다음 순서가 비용 대비 성공 확률이 가장 높다.

1. **결과/충실도 계약과 GPU full fail-closed를 고친다.** 이후의 모든 성능 숫자를 신뢰할 수 있게 만든다.
2. **현재 구현된 다섯 기능을 GPU0에서 재가격한다.** 특히 CMFD compaction은 이미 소스에 있으므로 재개발하지 않는다.
3. **K-process와 compaction으로 실제 활성 폭을 높인다.** 이는 M64의 가장 명백한 시스템 병목이다.
4. **FlatXS를 CTA-per-node 협업형으로 바꾼다.** 현재 배치 GPU 시간의 39.9%이며 코드 구조상 가장 큰 커널 최적화 여지가 있다.
5. **PPR의 host convergence와 reconstruction을 device로 이동하고 canonical input을 공유한다.** 이는 GPU가 빨라질수록 더 커지는 상태점 CPU 바닥을 제거한다.
6. **그 뒤에 장수명 evaluator를 도입한다.** 이미 안정화된 process/cohort/case 경계를 재사용하여 세대 간 startup과 cache 재구축을 없앤다.
7. **마지막으로 GA 중복 캐시와 warm-start를 결합한다.** 계산기 한 번의 속도와 필요한 계산 횟수를 동시에 줄인다.

이 순서를 따르면 각 단계가 다음 병목을 드러내고, 효과가 없는 경로는 kill criterion에서 조기에 중단할 수 있다. 최종 목표인 1,600–2,100 c/h/GPU는 단일 “초대형 커널”이 아니라 **활성 폭, FlatXS, PPR/상태점 바닥, process lifetime**의 측정된 조합으로 접근해야 한다.
