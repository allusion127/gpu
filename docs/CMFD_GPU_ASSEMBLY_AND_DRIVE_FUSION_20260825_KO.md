# GPU RASBERY CMFD operator assembly 및 BiCG drive 리팩토링 상세 보고서

**작성일:** 2026-08-25
**대상 브랜치:** `codex/cmfd-gpu-assembly-drive-fusion`
**기준 브랜치:** `codex/nodal-union`
**대상 PR:** `#3 perf: assemble CMFD operator on GPU and fuse BiCG scalar graph nodes`

## 1. 목적과 실측 근거

이번 변경은 노달 최적화 이후 새로 확인된 병목을 대상으로 한다. 기준 보고서
`docs/NODAL_REFACTOR_VALIDATION_BOTTLENECK_20260825_KO.md`의 서버 238 실측은 다음과 같다.

| 스테이지 | 단일 KNGR CY1 시간 | 전체 비중 |
|---|---:|---:|
| `drive` — CMFD BiCGSTAB, CUDA graph 및 arena wait | 55.76 s | 59.8 % |
| `setls` — CMFD operator 조립, CPU | 11.80 s | 12.7 % |
| `upddhat` — CNCC 계수 갱신, CPU | 5.74 s | 6.2 % |
| `updjnet` — 면전류 재구성, CPU | 2.53 s | 2.7 % |
| nodal FULL GPU | 5.84 s | 6.3 % |
| 전체 Driver | 93.21 s | 100 % |

같은 기준의 M64 결과는 1165.0 s, 197.8 cases/h이며 CMFD 평균 집계 폭은
21.42/64이다. 노달은 더 이상 주 병목이 아니며, 다음 두 비용이 남았다.

1. `drive` 안에서 반복되는 작은 CUDA graph node의 dispatch 비용.
2. 각 Driver가 CPU에서 `setls`를 완료한 뒤에야 CMFD arena에 도착하는 구조.

이번 패치는 **외부 Driver 상태기계 전체를 GPU로 옮기지 않는다.** 이미 존재하는
`CudaBatchArena`와 device-resident CMFD sweep 경계 안에서 다음 두 항목만 변경한다.

- post-Wielandt 구간의 `setls + 최초 updls`를 GPU에서 조립한다.
- BiCGSTAB residual reduction 직후의 scalar kernel을 안전한 범위에서 융합한다.

실제 RTX PRO 6000 성능은 서버 검증 전까지 **예상값**이며, 이 문서는 코드 구조와
검증 계약을 정의한다.

---

## 2. 변경 범위와 제외 범위

### 2.1 이번 패치에 포함되는 항목

- 2군 CMFD operator의 `diag`, `cc`, `udiag` 장치 조립
- 불변 Geometry mapping의 arena 공용 GPU 상주
- 슬롯별 XS, `dtil`, `dhat` 입력 staging
- host `diag/cc/udiag` H2D 제거
- initial residual norm stage-2와 `store_reference_norm` 융합
- iteration residual norm stage-2와 `accumulate_iteration` 융합
- 장치 조립 및 전송 제거량 telemetry
- CPU warm-up, CPU fallback 및 런타임 롤백 유지

### 2.2 이번 패치에 포함되지 않는 항목

- `upddhat` CUDA화
- `updjnet` CUDA화
- 노달 이후 CNCC 전체의 장치 상주
- Driver/Scheduler, Xe, T/H, 감손, critical search의 GPU 상태기계화
- 컬러 Gauss-Seidel sweep의 cooperative/persistent-kernel 변환
- reduction operand 순서 변경
- 반복 수, 수렴 허용오차 또는 물리 근사 변경

`upddhat`과 `updjnet`은 노달 전후의 별도 관측 경계에 있으므로 다음 단계로 남긴다.
이번 단계의 목적은 먼저 operator 소유권과 BiCG graph dispatch를 작은 범위에서
검증하는 것이다.

---

## 3. 변경 전후 데이터 흐름

### 3.1 변경 전

```text
CPU Driver
  ├─ updpsi
  ├─ setls: xssm/xsrf/dtil/dhat → host diag/cc/udiag
  └─ driveDeviceSweeps
       ├─ H2D: diag + cc + udiag + xsnf + chif + psi + phi ...
       ├─ CUDA graph: source → BiCGSTAB → Wielandt → updls → 종료판정
       └─ D2H: phi + psi + scalar status
```

이 경로에서는 CPU가 operator를 계산한 뒤 GPU가 동일 operator를 다시 받아야 한다.
M64에서는 각 입력의 CPU `setls` 도착 시차가 arena 폭도 제한한다.

### 3.2 변경 후 — resident assembly 활성 슬롯

```text
CPU Driver
  ├─ updpsi
  ├─ setls: 장치 조립 가능 조건만 판정, host operator 조립 생략
  └─ driveDeviceSweeps
       ├─ H2D: xsrf/xssm/xsnf/dtil/dhat 및 sweep state
       ├─ cmfd_assemble_operator_2g
       │    └─ device diag/cc/udiag에 직접 기록
       ├─ CUDA graph: source → BiCGSTAB → Wielandt → updls → 종료판정
       └─ D2H: phi + psi + scalar status
```

`diag`, `cc`, `udiag`는 조립 커널의 출력이면서 바로 다음 BiCGSTAB graph의 입력이다.
정상 resident 경로에서는 CPU로 돌아오지 않는다.

---

## 4. 파일별 구현 상세

### 4.1 `src/CmfdAssemblyKernel.h`

CPU 계약 테스트와 CUDA kernel이 공유하는 2군 operator 본체를 추가했다.

```cpp
namespace rasbery::cmfd_assembly {
struct View { ... };
__host__ __device__ void assembleNode2G(const View&, int node);
}
```

한 thread가 한 노드 `l`의 다음 출력을 전담한다.

- `diag[l][2][2]`
- `cc[l][2][3][2]`
- `udiag[l][2][2]`

서로 다른 thread가 같은 출력 위치를 쓰지 않으므로 atomic과 block barrier가 필요 없다.

### 4.2 `src/BICGCMFD.{h,cpp}`

`setls(eigv)`가 operator 소유권을 판정한다.

```text
canUseDeviceAssembly() == true
    → _device_assembly_pending = true
    → host setls/updls 생략

false
    → 기존 assembleHostLinearSystem(eigv)
```

장치 조립은 다음 조건을 모두 만족할 때만 허용한다.

- `RASBERY_GPU_CMFD_ASSEMBLY`가 비활성화되지 않음
- `RASBERY_GPU_CMFD_SWEEP=1`
- `RASBERY_CMFD_DUMP` 비활성
- 에너지군 수가 2
- CUDA batch arena 슬롯 보유
- Wielandt warm-up 5 sweep 완료

초기 Rayleigh warm-up은 기존 CPU operator 경로를 그대로 사용한다. `driveDeviceSweeps`
진입이 실패하면 host operator를 즉시 다시 조립한 뒤 원래 host loop로 복귀한다.

장치 sweep staging에는 다음 포인터를 추가했다.

```text
xsrfData(), xssmData(), _dtil, _dhat
```

### 4.3 `src/CudaBICGBackend.h`

`CmfdSweepIO`에 장치 조립 입력과 소유권 플래그를 추가했다.

```cpp
const double* xsrf;
const double* xssm;
const double* dtil;
const double* dhat;
double*       udiag;
bool          device_assembly;
```

또한 다음 counter를 추가했다.

```text
cmfd_assembly_gpu_calls
cmfd_assembly_cpu_fallbacks
cmfd_diag_h2d_elided_bytes
cmfd_cc_h2d_elided_bytes
```

### 4.4 `src/CudaBICGBackend.cu` — Geometry residency

arena 생성 시 모든 슬롯이 공유하는 다음 불변 배열을 한 번만 업로드한다.

| 배열 | 레이아웃 | 용도 |
|---|---|---|
| `assembly_node_surface` | `[node][direction][side]` | 노드 면의 surface index |
| `assembly_face_area` | `[node][direction]` | X/Y/Z 면적 |
| `assembly_volume` | `[node]` | 노드 체적 |

`compatibleGeometry()`는 기존 neighbor뿐 아니라 surface map, 면적, 체적까지 비교한다.
같은 `nxyz`이지만 topology가 다른 덱이 공용 arena를 잘못 사용하는 것을 막는다.

### 4.5 `src/CudaBICGBackend.cu` — 슬롯별 입력

각 슬롯에는 다음 mutable 입력을 둔다.

| 입력 | 원본 레이아웃 | 장치 레이아웃 |
|---|---|---|
| `xsrf` | group-major `[ig*nxyz+l]` | 동일 |
| `xssm` | `[(igs*2+ige)*nxyz+l]` | 동일 |
| `chif`, `xsnf` | group-major | 동일 |
| `dtil`, `dhat` | `[surface*2+ig]` | 동일 |
| `reigvs`, `eshift` | scalar block | 슬롯별 scalar |

`xsrf`, `xssm`, `xsnf`, `dtil`은 마지막 성공 H2D와 byte-exact 비교해 같으면
업로드하지 않는다. `dhat`은 노달 보정 뒤 매 outer 바뀌므로 현재 단계에서는 비교 없이
업로드한다. mirror commit은 stream drain 성공 뒤에만 수행한다.

장치 조립 슬롯은 다음 host output 전송을 생략한다.

```text
diag   : nxyz * 4 doubles
cc     : nxyz * 12 doubles
udiag  : nxyz * 4 doubles
```

telemetry의 `cmfd_diag_h2d_elided_bytes`와 `cmfd_cc_h2d_elided_bytes`는 각각
`diag`, `cc` 절감량을 기록한다. `udiag` 절감은 bulk skipped-call 수에 포함되며,
필요하면 후속 패치에서 별도 byte counter로 분리할 수 있다.

### 4.6 `cmfd_assemble_operator_2g`

커널 launch geometry는 기존 CMFD node kernel과 동일하다.

```text
grid.x = ceil(nxyz / block_size)
grid.y = arena slots
thread = 한 슬롯의 한 노드
```

`device_assembly_active[slot] == 0`이거나 `sweep_halt[slot] != 0`이면 즉시 반환한다.
동일 graph가 assembly 슬롯과 host-operator 슬롯을 동시에 처리할 수 있다.

### 4.7 예외 Rayleigh fallback

Wielandt의 `gammad` 또는 `gamman`이 퇴화하면 sweep state 2로 host Rayleigh branch에
제어를 돌려준다. 장치 조립 슬롯의 host `diag/cc/udiag`는 이 시점에 최신값이 아니므로
launcher가 stream 소유권을 해제하기 전에 다음을 한 번 D2H한다.

```text
device diag  → host _diag
device cc    → host _cc
device udiag → host _udiag
```

그 뒤에만 host Rayleigh/`updls` 경로를 진행한다. 정상 state 1/3에서는 operator D2H가
발생하지 않는다.

---

## 5. 부동소수점 및 물리 보존 계약

### 5.1 `setls` 누산 순서

`assembleNode2G()`는 기존 CPU 순서를 유지한다.

1. `diag[ige][igs] = -xssm[igs][ige] * volume`
2. diagonal에 `xsrf[ige] * volume` 추가
3. LEFT 면을 `Z → Y → X` 순서로 추가
4. RIGHT 면을 `X → Y → Z` 순서로 추가
5. 완성된 unshifted diagonal을 `udiag`에 저장
6. Wielandt shift 적용

면 방향 순서나 그룹 순서를 바꾸지 않는다.

### 5.2 명시적 곱셈/FMA 형식

기존 `cmfd_updls`에서 채굴한 형식과 맞추기 위해 shift는 다음 순서를 고정한다.

```text
c2 = round(round(chif * xsnf) * reigvs)
diag = fma(-c2, volume, udiag)
```

CUDA에서는 `__dmul_rn()`과 `fma()`를 사용한다. host 계약 테스트는 production과 같은
`-O3 -march=native`에서 원래 loop와 공유 본체의 모든 output byte를 비교한다.

### 5.3 reduction 보존

scalar fusion은 stage-1 partition을 변경하지 않는다.

```text
partial[0], partial[1], ... partial[blocks-1]
```

stage-2는 기존과 같은 index 증가 순서로 fold한다. 두 reduction을 합치거나 tree를
재구성하지 않는다.

---

## 6. BiCGSTAB CUDA Graph scalar-node fusion

기존 outer graph에는 다음 연속 scalar node가 있었다.

```text
initial norm stage2 → store_reference_norm
iteration residual norm stage2 → accumulate_iteration
```

다음 커널로 각각 융합했다.

```text
reduce_norm_store_reference_stage2
reduce_norm_accumulate_stage2
```

기본 `_nmaxbicg=3`은 1회 무조건 반복과 3회 추가 반복, 총 4회 BiCGSTAB iteration을
캡처한다. 따라서 한 outer당 제거되는 graph node는 다음과 같다.

```text
reference scalar node 1개
iteration scalar node 4개
합계 5개
```

기존 기록의 95-node outer를 기준으로 하면 구조상 약 90 node가 된다. 실제 node 수는
서버에서 `RASBERY_GPU_GRAPH_NODES=1`로 확인해야 한다.

### 6.1 halt/active 의미 보존

- inactive 슬롯은 partial을 읽지 않고 반환한다.
- 이미 halt된 iteration은 stale partial을 fold하지 않고 `overrun`만 증가시킨다.
- 정상 슬롯은 norm을 저장한 뒤 기존 `accumulate_iteration` 상태 전이를 그대로 수행한다.
- `iter_flags` re-arm, breakdown/early-exit counter, nonfinite 처리와 상대 residual test는
  기존 위치와 의미를 유지한다.

### 6.2 융합하지 않은 구간

- 컬러 Gauss-Seidel kernel 경계: 이웃 노드 의존성 때문에 유지
- dot stage-1 reduction: partition 및 tree 보존을 위해 유지
- `dot2`의 두 독립 reduction: 기존 구조 유지
- source/Wielandt/negative census의 phase boundary: 유지

---

## 7. 런타임 설정과 롤백

### 7.1 관련 환경변수

| 변수 | 기본 | 의미 |
|---|---:|---|
| `RASBERY_GPU_CMFD_SWEEP` | 기존 설정 따름 | resident multi-sweep 경로 활성화 |
| `RASBERY_GPU_CMFD_ASSEMBLY` | ON | post-warm-up operator 장치 조립 |
| `RASBERY_GPU_CMFD_SCALAR_FUSION` | ON | BiCG scalar stage-2 융합 |
| `RASBERY_GPU_GRAPH_NODES` | OFF | graph node receipt 출력 |

assembly는 변수 자체가 기본 ON이어도 `CMFD_SWEEP=1`, batch arena, 2군 및 post-warm-up
조건이 없으면 관여하지 않는다.

### 7.2 즉시 롤백

```bash
export RASBERY_GPU_CMFD_ASSEMBLY=0
export RASBERY_GPU_CMFD_SCALAR_FUSION=0
```

두 기능은 독립적으로 끌 수 있다. CPU warm-up과 기존 host fallback은 항상 컴파일되어
있다.

---

## 8. 자동 검증

### 8.1 공유 assembly 산술 계약

```bash
python tools/test_cmfd_assembly_kernel.py
```

독립 C++20 harness가 deterministic random 2군 문제를 만들고 다음 두 구현을 비교한다.

- 원래 `CMFD::setls + BICGCMFD::updls` 수식 인용본
- `cmfd_assembly::assembleNode2G`

`eshift=0`과 `eshift=0.04`에서 `diag`, `cc`, `udiag` 전체를 `memcmp`한다.

### 8.2 residency 및 fallback 정적 계약

```bash
python tools/test_cmfd_gpu_residency_contract.py
```

다음을 검사한다.

- assembly kernel과 raw input wiring
- Geometry 공유 배열
- warm-up 이후에만 host assembly 생략
- assembly launch가 sweep loop보다 앞에 위치
- resident 슬롯에서 host operator H2D 제거
- state 2 exceptional operator D2H
- device 경로 실패 시 host operator 재조립
- 관련 telemetry 존재

### 8.3 scalar fusion 계약

```bash
python tools/test_cmfd_scalar_fusion_contract.py
```

다음을 검사한다.

- strict-index stage-2 fold 유지
- reference와 iteration fusion kernel 존재
- active/halt/overrun 처리 유지
- 융합 OFF rollback kernel 유지
- `update_solution → residual stage1 → fused stage2` 순서

### 8.4 비-CUDA 전체 빌드

```bash
cmake -S . -B build-host \
  -DCMAKE_BUILD_TYPE=Release \
  -DRASBERY_ENABLE_CUDA=OFF
cmake --build build-host -j2
```

이 검증은 public header, non-CUDA stub, `BICGCMFD`, `BICGSolver` 및 전체 host link를
검사한다. CUDA kernel의 production 검증을 대체하지는 않는다.

### 8.5 GitHub Actions

`.github/workflows/cmfd-gpu-refactor-contracts.yml`은 read-only 권한으로 다음을 실행한다.

1. Python contract script compile
2. assembly C++20 byte contract
3. residency/fallback contract
4. scalar fusion contract
5. 관련 기존 CMFD/transfer mirror test
6. non-CUDA 전체 CMake build
7. PR 전체 `git diff --check`

workflow는 파일을 수정하거나 commit/push하지 않는다.

---

## 9. 서버 238 CUDA 검증 절차

### 9.1 생산 빌드

서버의 기존 gcc13/CUDA 13/no-pie wrapper가 있으면 그 경로를 우선 사용한다. 표준
CMake 명령은 다음과 같다.

```bash
git checkout codex/cmfd-gpu-assembly-drive-fusion

cmake -S . -B build238 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRASBERY_ENABLE_CUDA=ON \
  -DRASBERY_CUDA_ARCHITECTURES=120
cmake --build build238 -j
```

성공 조건:

- CUDA 13 `sm_120` compile/link 성공
- graph instantiate/fallback 오류 0
- 시작 receipt에 `assembly=on`, `scalar fusion=on` 표시

### 9.2 4-arm 기능 분리 A/B

같은 binary를 사용하고 환경변수만 변경한다.

| Arm | Assembly | Scalar fusion | 목적 |
|---|---:|---:|---|
| A | 0 | 0 | 완전 rollback 기준 |
| B | 1 | 0 | operator assembly 단독 효과 |
| C | 0 | 1 | graph scalar fusion 단독 효과 |
| D | 1 | 1 | 결합 효과 |

```bash
# A
RASBERY_GPU_CMFD_ASSEMBLY=0 \
RASBERY_GPU_CMFD_SCALAR_FUSION=0 \
<공통 실행 명령>

# B
RASBERY_GPU_CMFD_ASSEMBLY=1 \
RASBERY_GPU_CMFD_SCALAR_FUSION=0 \
<공통 실행 명령>

# C
RASBERY_GPU_CMFD_ASSEMBLY=0 \
RASBERY_GPU_CMFD_SCALAR_FUSION=1 \
<공통 실행 명령>

# D
RASBERY_GPU_CMFD_ASSEMBLY=1 \
RASBERY_GPU_CMFD_SCALAR_FUSION=1 \
<공통 실행 명령>
```

모든 arm에 동일하게 다음을 적용한다.

```bash
CUDA_VISIBLE_DEVICES=0
RASBERY_GPU=1
RASBERY_GPU_CMFD_SWEEP=1
RASBERY_GPU_RB_SWEEPS=4
RASBERY_GPU_XSRECON=1
RASBERY_GPU_FLATXS=1
RASBERY_GPU_NODAL=1
RASBERY_GPU_NODAL_FULL=1
RASBERY_PPR_MODE=master
RASBERY_PC_MODE=decart
RASBERY_OUTER_TIMING=1
```

M64는 같은 64개 덱, 같은 output 범위, 같은 `--batch-mode 64`, 같은 host-worker 정책을
사용해야 한다. 각 arm은 warm-up 1회 후 최소 3회 측정하고 중앙값으로 비교한다.

### 9.3 HDF5 정확도 gate

#### 단일 문제

- 기준: 검증된 `nodal-union` 또는 Arm A output
- 후보: Arm B/C/D
- 기대: **500/500 dataset byte-identical**

기존 `.regress.sh` Tier 3 또는 프로젝트의 HDF5 전체 비교 도구를 사용한다. timestamp와
명시적으로 제외된 비물리 metadata 외에는 차이를 허용하지 않는다.

#### M64 배치

- 기준 candidate와 동일한 입력 한 건 이상을 각 arm에서 추출
- 기대: **708/708 dataset byte-identical**
- 모든 case의 return code 0
- graph/arena/drive fallback 0

결과가 다르면 성능 수치는 폐기하고 다음 순서로 격리한다.

1. D와 B 비교: scalar fusion 영향 분리
2. D와 C 비교: assembly 영향 분리
3. `RASBERY_GPU_GRAPH=0`: graph capture 영향 분리
4. `RASBERY_GPU_CMFD_SWEEP=0`: resident sweep 영향 분리
5. `RASBERY_CMFD_DUMP`: 기존 form probe로 첫 divergent sweep 재생

### 9.4 telemetry 확인

로그에서 다음 JSON을 보존한다.

```text
[RASBERY][CUDA][BACKEND_COUNTERS]
[RASBERY][CUDA][BATCH_OCCUPANCY]
[RASBERY][OUTER][PHASE]
[RASBERY][CUDA][GRAPH_NODES]
```

Arm D의 필수 조건:

```text
cmfd_assembly_gpu_calls > 0
cmfd_diag_h2d_elided_bytes > 0
cmfd_cc_h2d_elided_bytes > 0
graph_fallbacks == 0
```

`cmfd_assembly_cpu_fallbacks`는 warm-up/비대상 슬롯을 포함할 수 있으므로 단순 0을
요구하지 않는다. 대신 post-warm-up assembly call과 elided bytes가 실제로 누적되는지
확인한다.

### 9.5 처리량 판정

현재 비교 기준:

```text
nodal-union M64 : 197.8 cases/h
MASTER W16      : 216–218 cases/h
```

채택 기준은 다음과 같다.

1. 모든 정확도 gate 통과
2. fallback 0
3. Arm D 중앙값이 Arm A보다 빠름
4. assembly 단독 B와 scalar 단독 C의 효과가 counter 및 graph-node 변화와 일치
5. 동일 물리·동일 출력 기준으로 MASTER와 비교

실측 전 예상 범위는 다음과 같이만 해석한다.

- `setls` 12.7 % 전체를 제거해도 Amdahl 상한은 단일 실행 약 1.15배이다.
- 실제로는 raw assembly input H2D와 커널 1개가 추가되므로 그보다 작다.
- scalar fusion은 outer당 graph node 5개를 줄이지만 `drive` 전체가 node dispatch만으로
  구성된 것은 아니므로 5/95를 전체 wall 가속도로 직접 환산할 수 없다.
- M64에서는 CPU `setls` 제거가 arena 도착 스큐를 줄여 단일시간 절감보다 큰 처리량
  효과를 낼 가능성이 있으나, 이는 반드시 실측으로 확인한다.

---

## 10. 실패 시 롤백 및 진단

### 10.1 기능별 롤백

```bash
# assembly만 끄기
export RASBERY_GPU_CMFD_ASSEMBLY=0

# scalar fusion만 끄기
export RASBERY_GPU_CMFD_SCALAR_FUSION=0

# resident sweep까지 끄기
export RASBERY_GPU_CMFD_SWEEP=0
```

### 10.2 예상되는 실패 분류

| 증상 | 우선 확인 |
|---|---|
| CUDA compile 오류 | `CmfdAssemblyKernel.h` host/device annotation, CUDA 13 API |
| 첫 post-warm-up 상태부터 HDF5 차이 | assembly 산술/FMA 형식 |
| gamma-degenerate case만 차이 | exceptional operator D2H와 host Rayleigh fallback |
| overrun/linear_iter counter 차이 | scalar fusion halt 경로 |
| graph node 수 변화 없음 | `RASBERY_GPU_CMFD_SCALAR_FUSION`, graph 재생성 여부 |
| elided bytes 0 | `CMFD_SWEEP`, warm-up 조건, batch arena 사용 여부 |
| M64 폭 하락 | raw input H2D, launcher critical path, host-worker 정책 |

---

## 11. 알려진 한계와 후속 단계

1. 이번 단계에서는 `upddhat`과 `updjnet`이 CPU에 남는다. 기준 단일 wall의 8.9 %이다.
2. assembly는 현재 `CudaBatchArena`의 resident sweep 경로에 한정된다. private one-slot
   backend는 scalar fusion만 적용된다.
3. `dhat`은 매 outer 업로드한다. 이후 CNCC GPU화가 완료되면 장치 상주가 가능하다.
4. `cmfd_wiel_finalize`의 엄격한 직렬 fold는 bit 재현성을 위해 유지한다.
5. CUDA 13/sm_120 build, 500/500 및 708/708 bit gate와 M1/M64 성능은 서버 238에서
   수행해야 하며, 완료 전에는 가속도를 확정하지 않는다.

후속 우선순위는 다음과 같다.

```text
1. 이번 operator assembly + scalar fusion 서버 검증
2. updjnet → nodal → upddhat 경계의 batched CUDA화
3. diag/cc/dtil/dhat 완전 장치 소유권 검토
4. HDF5 writer 경합 분리
5. N>64 resource/registration 오류 별도 격리
```

---

## 12. 현재 검증 상태

이 문서를 push하기 전 로컬 환경에서 수행한 결과:

| 항목 | 결과 |
|---|---|
| assembly host byte contract | PASS |
| residency/fallback static contract | PASS |
| scalar fusion static contract | PASS |
| 비-CUDA 전체 CMake build/link | PASS |
| 관련 기존 host contract tests | PASS |
| `git diff --check` | PASS |
| CUDA 13/sm_120 build | 미실시 — 서버 238 필요 |
| APR1400 HDF5 bit gate | 미실시 — 서버 데이터 필요 |
| M1/M64 성능 | 미실시 — 서버 238 필요 |

따라서 현재 단계에서 확인된 것은 **구조·산술 계약과 host 빌드**이다. GPU 물리 정합과
처리량은 9절의 서버 gate를 통과한 뒤에만 완료로 판정한다.
