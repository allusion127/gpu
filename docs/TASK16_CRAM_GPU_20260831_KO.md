# Task 16 — GPU CRAM 감쇠 (`RASBERY_GPU_CRAM`, 기본 OFF) (2026-08-31)

계획 Rev.7.1 개정 1 / GA evaluator 계획 §6.3의 **상태점 바닥(`c`)** 공격 두 번째 항목.
Task 10이 PPR `reset`+`drive`를 옮긴 뒤 바닥에 남은 가장 큰 덩어리 —
`XSSet::PredictorStep`/`CorrectorStep` 안의 **CRAM Bateman 감쇠 노드 루프 8,451개** —
를 디바이스로 옮긴다.

> **번호 충돌 주의.** GA evaluator 계획 **§6.4 W4의 "Task 16"은 island arm**이다.
> 이 문서의 Task 16은 Rev.7.1 개정 1의 **GPU CRAM 감쇠**이며 다른 트랙이다.

**이 문서는 측정이 아니다.** 로컬 PC/WSL에서는 아무것도 실행하지 않았다(사용자 규칙).
아래 §7의 런북이 238 GPU0에서 실행되어야 비로소 등급이 확정된다. §8이 "빌드 없이는
확인할 수 없었던 것"을 빠짐없이 적는다.

---

## 1. 왜 여기인가 — 비용 원장

`docs/GPU_RASBERY_PERFORMANCE_AND_ARCHITECTURE_REPORT_20260830_KO.md` §3.6:

| 항 | 238 단일 | 의미 |
|---|---:|---|
| `c` 상태점당 호스트 바닥 | **0.343 s/sp** | 16.9 s 케이스의 **12 s**. outer 본문보다 크다 |
| 그중 PPR (`ppr_reset`+`ppr_drive`+`ppr_recon`) | 0.158 s/sp | **바닥의 ~50 %**, `c502856`이 94.6 % 이식 |
| 나머지 ~50 % | ~0.185 s/sp | **CRAM 감쇠(pred+corr)** + FlatXS + T/H + result packing |

`floor_wall` 수신증의 `depl_predictor` / `depl_corrector`가 그 나머지의 어느 만큼인지를
말하는 유일한 계기이고, **§7의 G3가 그 숫자를 처음 기록한다.** 지금까지 캠페인은 이
두 버킷의 절대값을 문서에 적은 적이 없다 — PPR 표만 있다.

### 1.1 호스트 감쇠가 실제로 무엇인가 (밀도 LU가 아니다)

`include/milk.h` `solveBatemanCRAM`, `order == 8` 분기 (`~:1649-1819`),
호출부 `src/XSSet.cpp:4487` (predictor) / `:4708` (corrector):

```
alpha0     = 1.1722341374385704e-08     poles     = 4      matrix_sgn = -1
max_iter   = 64                         rel_tol   = 1.0e-13
abs_tol    = 1.0e-28                    diag_tol  = 1.0e-30
```

극점마다 `x = rhs/diag`로 씨를 뿌리고, 최대 64회의 **희소 Gauss-Seidel sweep**
(`x[row] = (rhs[row] - Σ vals·x[col]) / pole_diag[row]`) + **명시적 잔차 sweep** +
최대노름 파단 검사. 밀도 LU가 아니다.

계는 39×39이고 `first = iI135 = 3` (H/B/O는 그대로 통과)이므로 **미지수 36개**.
희소성은 **노드가 아니라 동위원소 사슬이 정한다** — `include/Database/dep_decay.csv`,
`dep_trans.csv`를 행·열 ≥ 3으로 잘라 합집합을 내면:

| 항목 | 값 |
|---|---:|
| `nonzeros(depDecay)` | 63 |
| `nonzeros(depTrans)` | 131 |
| 합집합 (대각 포함, 행·열 ≥ 3) | **160** |
| 그중 비대각 | **124** |
| 한 행 최대 비대각 | **19** |
| 한 행 평균 비대각 | 3.54 |

**이것이 GPU 문제인 이유다**: 8,451개의 독립적인 36-미지수 희소 해.

---

## 2. 설계

### 2.1 노드당 스레드 하나 — 정확성이 강제한다

Gauss-Seidel은 `row`에 대해 **순차적**이고(행 i는 이번 sweep이 이미 갱신한 x[j], j<i를
읽는다), 내부 `sum -= vals[i] * x[cols[i]]`도 i에 대해 순차적이다. warp-per-node
커널은 둘 중 하나를 재결합해야 하며, 그러면 **재현이 아니라 다른 방법**이 된다.

따라서 병렬성은 **노드 축에서만** 취한다. 노드당 스레드 하나, 노드당 상태(~4.9 KB:
`cond4` 156, 패턴값 124+36, `x`/`accum` 각 39 복소, `iden` 39)는 **스레드 로컬 배열**에
둔다 — CUDA가 로컬 메모리를 warp 간 인터리브하므로 별도의 노드-stride scratch 블록을
손으로 쓰지 않아도 coalesce된다.

- 점유율은 설계상 낮다. 이득은 **8,451개의 순차 해가 24개가 아니라 64–128개씩 동시에**
  도는 것이다.
- 블록 크기는 `RASBERY_GPU_CRAM_BLOCK` (32/64/128/256, 기본 **64**). §7 G3가 스윕한다.

### 2.2 채굴된 패턴은 상위집합이고, 0 검사는 그대로 돈다

호스트는 `A(row,col)`을 훑으면서 `value == 0.0`이면 건너뛴다. 업로드하는 패턴은
**노드 독립 합집합**이므로, 어떤 노드에서 값이 정확히 0이 되면 호스트가 만들지 않았을
항을 커널이 들고 있게 된다. 그래서 커널은 패턴을 믿지 않고 **노드마다 호스트의
`value == 0.0` 검사를 다시 돌려** 압축한다(`src/CudaCramBackend.cu`, `cramSolveNode`).
압축된 행은 호스트의 행과 **항 대 항, 오름차순 열 순서까지 같다**.

전이행렬 값 자체도 호스트의 갱신 순서대로 조립한다(`transEntry`):
`depDecay` → p-루프의 단 한 번의 접촉(p == col) → (n,2n) 특례 둘. 각 엔트리는 p-루프가
최대 한 번 건드리므로 **누적 순서 문제가 아예 없다**.

### 2.3 등급은 N1이고, 이유는 복소 나눗셈이다

실수 연산은 전부 호스트의 문장 순서로 옮겼고 TU는 `--fmad=false`다. 복소 곱은
gcc가 유한 피연산자에 대해 인라인으로 내는 naive `(ac−bd, ad+bc)`를 그대로 적었다.
**복소 나눗셈**만은 그렇게 되지 않는다: libstdc++는 `complex<double> / complex<double>`을
libgcc의 `__divdc3`로 내리는데, 이것은 나누기 전에 `logb`/`scalbn`으로 **재스케일**한다.
커널은 `__divdc3`의 알고리즘을 그대로 옮겼지만 `logb`/`scalbn`/나눗셈은 **디바이스
libm의 것**이다. 그래서:

- **등급 N1**, 게이트는 **Gate A**(핵종 수밀도 최대 상대차 + 다음 상태점의 keff/ppm/AO),
  `h5diff`가 아니다.
- CUDA의 `cuCdiv`는 **재스케일 없는 Smith 방법**이라 호스트와 다른 산술이다. 계약
  시험이 `cuCdiv`/`cuComplex` 사용을 금지한다.
- 실행 간 결정성은 보장된다: 출력당 스레드 하나, atomic 없음, 리덕션 없음, 고정 패턴.
  (`stats`의 `atomicOr`/`atomicAdd`는 **정수 카운터**이고 물리에 들어가지 않는다.)

`milk::detail::magnitude`의 두 분기(`sqrt(re²+im²)` / `std::abs` = `hypot`)와
`std::max`(= `(a<b)?b:a`, `fmax`가 **아니다**)도 분기 그대로 옮겼다.

### 2.4 `RASBERY_GPU_CRAM`은 **arm knob이다** — Task 10과의 결정적 차이

PPR은 상태점의 **하류**다: 수렴된 flux를 읽고 핀출력만 쓴다. 그래서
`RASBERY_GPU_PPR`은 `trajectory::kArmEnv`에서 **일부러 빠져 있고**,
`tools/test_ppr_gpu_contract.py`가 그 부재를 강제한다.

감쇠는 정반대다. 출력이 **다음 상태점의 동위원소 재고**이고, 다음 XS 재구성·붕소
탐색·그 뒤 모든 고유값이 이 밀도의 하류다. 그래서:

- `RASBERY_GPU_CRAM`은 `src/Driver.h:423`의 `kArmEnv`에 **들어간다**.
- 이유는 목록 바로 위 주석(`Driver.h:395`)에 적혀 있다.
- `tools/test_cram_gpu_contract.py`의 규칙 12가 **존재**를 강제한다 — PPR 계약이
  **부재**를 강제하는 것의 거울상.
- 결과로 **arm-off 실행의 stdout은 `8b8f18e`와 바이트 동일이 아니다**: `kArmEnv`가
  한 칸 늘었으므로 `[RASBERY][TRAJECTORY]`의 `env` 필드에 `"RASBERY_GPU_CRAM":null`이
  추가된다. `digest`는 바뀌지 않는다. §7 G1이 이 구분을 그대로 검사한다.

### 2.5 무엇을 옮기지 않았는가

| 호스트 경로 | 왜 아닌가 |
|---|---|
| `DepleteRodMaterials` | CRAM 해가 아니라 fine rod mesh 위의 fluence 누적. `solveBatemanCRAM`에 들어가지 않는다 |
| `DecayIsotopeDensityFlat` | restart 냉각 사슬. **런당 1회**, 상태점당이 아니다 |
| `RASBERY_PC_SUBSTEPS > 1` (Isotalo 하위단계) | k=1 본문의 직진 확장이지만 기본 OFF이고 캠페인 arm에 한 번도 들어간 적이 없다. **측정되지 않은 두 번째 디바이스 경로**보다 호스트 폴백이 싸고 정직하다. `XSSet::CorrectorStepGpu`가 `substeps != 1`이면 declines |
| corrector 뒤꼬리 (`DepleteRodMaterials` / `PrecomputeBranchCoefficients` / `UpdateFlatXS`) | 두 경로 공통. 디바이스 arm은 `_iden`/`_burn`에서 멈춘다 |

`RASBERY_PC_MODE=decart`(Eq. 6.20 밀도 평균)와 `RASBERY_PC_XE_EQUILIBRIUM_FIX`는
**둘 다 커널이 처리한다**(플래그로 전달).

---

## 3. 전송 원장 — 4슬롯 절단이 이 arm의 핵심 최적화

호스트는 `condensed`에 **11개 스칼라 XS 슬롯**을 응축한다. 그런데 하류에서 실제로
읽는 것은 넷뿐이다:

- `BuildTransitionMatrix` (`XSSet.cpp:3708`): `XSAF`, `XSFF`, `XS2N`, `XS3N`
- `ComputeXeEquilibrium` (`XSSet.cpp:3758`): `XSFF`, `XSAF`

나머지 일곱은 계산되고 버려진다. 그래서 백엔드는 **네 블록만 업로드**한다.

| 항목 | 11슬롯이면 | 4슬롯 | 크기 (nxyz=8,451, niso=39, ng=2) |
|---|---:|---:|---|
| micro XS 1회 업로드 | 58.0 MB | **21.1 MB** | 슬롯당 5.27 MB |
| predictor | 21.1 MB | | `_micx_generation`이 바뀐 경우에만 |
| corrector | 21.1 MB | | SolveLoop가 사이에 돌므로 사실상 매번 |
| corrector용 BOS 스냅샷 | — | **0 MB (D2D)** | predictor가 `d_mic → d_mic_bos`를 디바이스 안에서 복사 |
| `_iden` / `_iden_bos` | | 2.64 MB × 2 | |
| flux / flux_bos / xskf ×2 / burn_bos | | ~0.5 MB | |
| **상태점당 H2D 합** | ~120 MB | **≈ 47.9 MB** | |
| **상태점당 D2H** | | **≈ 4.9 MB** | `_iden` 행 [3,39) 2.43 MB + `_burn` 34 KB + stats 24 B |

**정확성 근거**: 아무도 읽지 않는 값은 결과를 바꿀 수 없다. 다만 이 근거는 *지금*
호스트가 넷만 읽기 때문에 성립하므로, 계약 시험 규칙 9가
`BuildTransitionMatrix`/`ComputeXeEquilibrium`을 파싱해 **다섯 번째 슬롯을 읽기 시작하면
테스트를 깨뜨린다.**

### 3.1 상주성 — 아직 빌려 쓰지 못한다 (후속 과제)

`CudaXsReconBackend`는 이미 `_micx`/`_lmpx`/`_xs`를 `micx_generation`/`state_generation`
키로 **디바이스 상주**시킨다(`src/CudaXsReconBackend.h:91,186`). 그러나 그 버퍼는
`XsReconBackend::Impl` 내부에 있고 **디바이스 포인터를 밖으로 내주는 접근자가 없다.**
Task 16은 그래서 자기 사본(4슬롯)을 따로 stage하고, 같은 `_micx_generation` 카운터로
스킵을 판정한다.

**후속(Task 16b 후보)**: `XsReconBackend`에 `residentMicx(slot) -> const double*`를
노출하거나 두 백엔드를 `GpuPhysicsArena` 아래로 합치면 상태점당 21.1 MB × 2가 사라진다.
§7 G3의 `micx_h2d_mb` 수신증 필드가 이 후속의 상한을 그대로 준다.

---

## 4. 실패 개방 — CUDA 오류보다 넓다

`predictor()`/`corrector()`가 **호스트 배열에 아무것도 쓰지 않은 채** false를 돌려주는
경우:

1. arm off / CUDA 디바이스 없음 / CPU 전용 빌드(stub)
2. `ng != 2`, `niso != 39`, 감쇠 데이터 없음, 합집합 패턴 > `kMaxNnz`(192)
3. `RASBERY_PC_SUBSTEPS > 1`
4. corrector의 BOS 토큰 불일치 — **이 상태점의 디바이스 predictor가 만든 스냅샷이 아님**
5. 임의의 CUDA 오류 (`Impl::fail`, 이후 이 인스턴스는 영구 비활성 + stderr 1줄)
6. **노드 상태 플래그가 하나라도 0이 아닌 경우**

6이 물리적으로 중요한 항목이다.

### 4.1 NaN 마스크 노트

호스트 `solveBatemanCRAM`은 두 조건에서 **던진다**:
`milk: CRAM Gauss-Seidel zero diagonal`, `milk: CRAM Gauss-Seidel did not converge`.
커널은 던질 수 없다. 대신 노드마다 상태 비트를 세우고
(`kZeroDiag=1`, `kNoConverge=2`, `kNonFinite=4`) `atomicOr`로 접는다:

- 커널은 결과를 **디바이스 staging 블록**(`d_iden_out`, `d_burn_out`)에 쓴다.
- 백엔드는 **먼저 3워드 stats를 D2H**하고, `stats[0] != 0`이면 그 자리에서 false를
  돌려준다 — `_iden`/`_burn`은 **손대지 않은 상태**다.
- 호스트 노드 루프가 손대지 않은 재고에서 다시 돌고, **같은 노드에서 같은
  `std::runtime_error`를 던진다.** 관측 가능한 동작이 동일하다.
- `kNonFinite`는 호스트에 없는 추가 방어다: (a) 발행 밀도가 유한하지 않은 경우,
  (b) corrector의 `static_cast<int>(burn_key_increment + 0.5)` — 비유한 값에서 **양쪽
  모두 UB인 캐스트** — 을 만나는 경우. 호스트는 조용히 쓰레기 burnup key를 쓰지만,
  디바이스 arm은 그것을 발행하는 대신 통째로 폴백한다. **이것은 의도적인 비대칭이고,
  arm이 켜졌을 때만 발동한다.**

`tools/test_cram_gpu_contract.py` 규칙 10이 두 진입점 모두에서
`stats` 검사가 `iden` D2H **앞**에 오는지를 강제한다(음성 대조 포함).

---

## 5. 수신증

```
[RASBERY][CRAM_GPU] {"schema_version":1,"slot":0,"statepoints":35,
  "predictor_calls":35,"corrector_calls":35,"nodes":591570,"device":0,
  "host_fallbacks":0,"gs_iters_mean":<x.xxx>,"gs_solves":2364280,
  "micx_h2d_mb":1477.3,"bos_reuse":35,"wall_ms":<...>,"status":"on"}
```

| 필드 | 읽는 법 |
|---|---|
| `statepoints` / `predictor_calls` / `corrector_calls` | 셋이 같아야 정상. corrector만 적으면 **상태점 중간 폴백** |
| `host_fallbacks` | arm ON에서 **0이 아니면 arm이 부분적으로만 돈 것** — 성능 수치를 읽기 전에 이것부터 |
| `nodes` | `nxyz × (predictor+corrector)`. arm ON인데 0이면 **arm은 없었다**(G0 유효성) |
| `gs_iters_mean` | (노드, 극점)당 평균 GS sweep 수. **디바이스가 호스트와 같은 반복을 풀었는지**를 말하는 유일한 관측량 |
| `micx_h2d_mb` | 4슬롯 업로드가 실제로 지불한 바이트. §3.1 후속의 상한 |
| `bos_reuse` | D2D BOS 스냅샷을 소비한 corrector 수 (= `corrector_calls`여야 한다) |
| `wall_ms` | 커널+전송의 CUDA event 누적. `floor_wall`의 `depl_*` 감소분과 대조 |

arm OFF에서도 `RASBERY_STATEPOINT_TELEMETRY=1`이면 인쇄된다(비교의 반대쪽이 있어야
하므로). 둘 다 없으면 인쇄되지 않는다.

---

## 6. 변경된 파일

| 파일 | 내용 |
|---|---|
| `src/CudaCramBackend.h` (신규) | 계약·비용 원장·등급·`kArmEnv` 근거·`LibView`/`PredictorView`/`CorrectorView` |
| `src/CudaCramBackend.cu` (신규) | `cdiv`(=`__divdc3`)/`cmul`/`cmagnitude`, `transEntry`, `cramSolveNode`, `applyXeEquilibrium`, `kPredictor`, `kCorrector`, `Impl`, `minePattern` |
| `src/CudaCramBackendStub.cpp` (신규) | CPU 전용 빌드용 동일 심볼, 전부 false |
| `CMakeLists.txt` | 두 소스 목록 + `--fmad=false` |
| `src/XSSet.h` | `_cram_backend`, `_cram_host_fallbacks`, `_cram_bos_token`, `_cram_dfac/_cram_vol`, `cram()`, `cramHostFallbacks()`, 세 private 진입점 |
| `src/XSSet.cpp` | `static_assert(CRAM_ORDER == cram::kOrder)`, `Deplete`/`CorrectorStep` 훅, `PrepareCramLib`/`DepleteGpu`/`CorrectorStepGpu` |
| `src/Driver.h` | `kArmEnv += RASBERY_GPU_CRAM` (+근거 주석), `[RASBERY][CRAM_GPU]` 수신증 |
| `tools/test_cram_gpu_contract.py` (신규) | 14규칙 × 음성 대조 |
| `tools/check_cuda_syntax.py` | shim 확장: `cudaEvent_*`, `cudaStreamCreate`, `cudaGetDevice/SetDevice`, `atomicOr`, `__longlong_as_double`, `logb/scalbn/hypot/copysign`, `__constant__`/`__shared__` 제거 |

---

## 7. 238 GPU0 런북

> **전제**: 238, GPU0 단독, `CUDA_VISIBLE_DEVICES=0`. `<bld>`는 빌드 디렉터리,
> `$O`는 출력 디렉터리. arm 기반은 `docs/V3_FREEZE_20260829_KO.md` §2의 v3 arm.

### G0 — 빌드 전 정적 게이트 (로컬에서도 가능, 실행 아님)

```bash
python3 tools/test_cram_gpu_contract.py     # PASS (14 rules, each with a negative control)
python3 tools/test_ppr_gpu_contract.py      # 여전히 OK (10 properties) — 거울상 규칙이 살아 있어야 한다
python3 tools/check_cuda_syntax.py --cxx g++ \
    -I include -I include/chiffon -I src src/CudaCramBackend.cu
for t in tools/test_*.py; do python3 $t >/dev/null 2>&1 || echo "FAIL $t"; done
```

**사전 실패 목록(이 변경 이전에도 실패, 신규 아님)**: `test_cmfd_assembly_kernel.py`,
`test_cmfd_fp32_contract.py`, `test_cuda_transfer_mirror.py`, `test_ga_feedback_screen.py`,
`test_ga_promotion_gate.py`, `test_gpu_phase_scheduler_contract.py`,
`test_nodal_constant_cache.py`, `test_nodal_constant_kernel.py`,
`test_segment_canonical_nodal_contract.py`. (앞의 세 개는 로컬에 C++ 컴파일러가 없어서;
나머지는 앵커 표류 — `git stash` 대조로 확인함.)

### 빌드

```bash
cmake -S . -B <bld> -DRASBERY_ENABLE_CUDA=ON -DRASBERY_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build <bld> -j
# 로컬 스택 프레임 확인 (설계 전제: 스레드당 ~4.9 KB)
cuobjdump -res-usage <bld>/RASBERY 2>/dev/null | grep -A3 -i "kPredictor\|kCorrector"
```

`kPredictor`/`kCorrector`의 `STACK`/`LMEM`이 **스레드당 6 KB를 크게 넘으면** 블록 크기를
32로 내리고 G3에서 재측정한다.

### G1 — arm OFF 동일성 (필수, 첫 관문)

```bash
# v3 arm 그대로, RASBERY_GPU_CRAM 은 설정하지 않는다
<bld>/RASBERY --rasi kngr_238.json --raso $O/off_new.h5 > $O/off_new.log 2>&1
h5diff -r $O/off_new.h5 $O/off_8b8f18e.h5          # 전 데이터셋 Δ=0 이어야 한다
grep TRAJECTORY $O/off_new.log $O/off_8b8f18e.log  # digest 가 같아야 한다
diff <(grep -v RASBERY_GPU_CRAM $O/off_new.log) $O/off_8b8f18e.log
```

- **h5 바이트 동일 + `digest` 동일**이 G1의 합격 조건이다.
- **stdout은 한 곳만 다르다**: `[RASBERY][TRAJECTORY]`의 `env`에
  `"RASBERY_GPU_CRAM":null`이 추가된다(§2.4). 위 `diff`가 그 한 줄만 걸러낸다.
  다른 차이가 나오면 **G1 불합격**이고 그 자리에서 멈춘다.
- `[RASBERY][CRAM_GPU]`는 arm OFF + telemetry OFF에서는 **인쇄되지 않는다**.

### G2 — arm ON: 결정성 ×2 + 상태점별 Gate A

```bash
export RASBERY_GPU_CRAM=1
for r in 1 2; do
  /usr/bin/time -f "%e" -o $O/on_r$r.wall <bld>/RASBERY --rasi kngr_238.json \
      --raso $O/on_r$r.h5 > $O/on_r$r.log 2>&1
done
h5diff -r $O/on_r1.h5 $O/on_r2.h5        # 결정성: 전 데이터셋 Δ=0 (N1이라도 run-to-run 은 0)
grep TRAJECTORY $O/on_r*.log             # digest 2개 동일
grep CRAM_GPU  $O/on_r1.log              # host_fallbacks=0, nodes>0, predictor=corrector
python3 tools/gate_a_compare.py $O/off_new.h5 $O/on_r1.h5 --per-step
```

**핵종 수밀도 Gate A** — `gate_a_compare.py`는 keff/ppm/AO/pin만 본다. 밀도는 별도로
잰다. 덱 사본에 `"node monitor"`(감시 노드 목록)를 넣거나 각 스케줄 항목에
`"save": true`를 주어 상태점마다 재고를 발행시킨 뒤:

```bash
python3 - "$O/off_new.h5" "$O/on_r1.h5" <<'PY'
import sys, h5py, numpy as np
a, b = (h5py.File(p, "r") for p in sys.argv[1:3])
worst = (0.0, None)
def walk(g, path=""):
    for k, v in g.items():
        p = f"{path}/{k}"
        if isinstance(v, h5py.Group): walk(v, p)
        elif k in ("isotope_density", "ref_isotope_density"): yield p
for p in walk(a):
    if p not in b: continue
    x, y = np.asarray(a[p]), np.asarray(b[p])
    d = np.abs(y - x) / np.maximum(np.abs(x), 1e-30)
    m = float(d.max())
    if m > worst[0]: worst = (m, p)
    print(f"{p:60s} max_rel={m:.3e}")
print(f"\nWORST max relative nuclide density difference: {worst[0]:.3e} at {worst[1]}")
PY
```

**판정 기준 (제안, 238 실측이 확정한다)**

| 지표 | 스크린 | 근거 |
|---|---:|---|
| 핵종 수밀도 max rel diff | **≤ 1e-10** | 산술 차이의 유일한 출처가 `__divdc3`의 `logb/scalbn`이므로 ULP급을 기대한다. 1e-8을 넘으면 **전사 오류를 의심하고 멈춘다** |
| 다음 상태점 keff | ≤ 1 pcm | Gate A 화면 기본(5 pcm)보다 타이트하게 시작 |
| 임계붕소 | ≤ 1 ppm | |
| AO | ≤ 0.002 | |
| BOC 핀 max rel | ≤ 0.05 % | |

> **밀도가 정확히 0.0으로 일치하면** 그것은 `__divdc3` 전사까지 비트 일치했다는 뜻이고,
> 그때는 등급을 **B0로 승격 제안**할 수 있다. 단 승격은 §7의 G5(3덱)까지 0을 유지할
> 때만이며, 헤더/계약 시험의 N1 문구를 같은 커밋에서 바꿔야 한다.

### G3 — 텔레메트리: 바닥 분해 `depl_predictor`/`depl_corrector` 호스트 ms → 디바이스 ms

```bash
# 타이밍 실행과 섞지 말 것
for arm in off on; do
  env $( [ $arm = on ] && echo RASBERY_GPU_CRAM=1 ) RASBERY_STATEPOINT_TELEMETRY=1 \
    <bld>/RASBERY --rasi kngr_238.json --raso $O/tel_$arm.h5 > $O/tel_$arm.log 2>&1
done
python3 tools/outer_profile.py $O/tel_off.log
python3 tools/outer_profile.py $O/tel_on.log
python3 tools/case_cost_profile.py $O/tel_*.log --wall-dir $O    # (c, d) 이동량
python3 tools/test_telemetry_neutrality.py --compare $O/on_r1.log $O/tel_on.log
grep -h "floor_wall" $O/tel_off.log | tail -1
grep -h "floor_wall" $O/tel_on.log  | tail -1
grep -h CRAM_GPU $O/tel_on.log
```

**기록할 표** (이 문서에 되돌려 채운다):

| 항목 | arm OFF | arm ON | 비고 |
|---|---:|---:|---|
| `depl_predictor` (s, ms/sp) | | | |
| `depl_corrector` (s, ms/sp) | | | |
| 두 항 합이 `c`에서 차지하는 비중 | | | **캠페인 최초 측정** |
| `[CRAM_GPU] wall_ms` | — | | 커널+전송 |
| `[CRAM_GPU] micx_h2d_mb` | — | | §3.1 후속의 상한 |
| `[CRAM_GPU] gs_iters_mean` | — | | |
| `c` (s/sp) | 0.343 | | |

**블록 크기 스윕** (같은 실행 형식, `RASBERY_GPU_CRAM_BLOCK ∈ {32,64,128}`):
`wall_ms`가 최소인 값을 채택하고, 채택값을 이 문서와 v3 arm 문서에 적는다.

**킬 기준**: arm ON의 `depl_predictor + depl_corrector`가 arm OFF 대비 **1.3× 미만으로만
줄면** 이 트랙을 §3.1의 상주성 후속(Task 16b)까지 보류한다 — 그때는 21.1 MB × 2의
버스 시간이 커널 이득을 먹고 있다는 뜻이고, 커널을 더 손대는 것은 순서가 틀렸다.

### G4 — 4덱 배치 on/off (per-slot 격리 확인)

```bash
for arm in off on; do
  env $( [ $arm = on ] && echo RASBERY_GPU_CRAM=1 ) \
    python3 tools/run_single_gpu_batch.py --batch-width 4 \
      --jobs ~/t18decks/kngr/jobs4.txt --workdir $O/batch_$arm -- <bld>/RASBERY
done
grep -h CRAM_GPU $O/batch_on/*.log      # slot 0..3 이 각자 device/statepoints/host_fallbacks 를 낸다
for i in 0 1 2 3; do h5diff -r $O/batch_off/case$i.h5 $O/batch_on/case$i.h5 || true; done
```

- **필수**: `[CRAM_GPU]`가 **슬롯마다 한 줄**, `host_fallbacks=0`, `slot` 값이 서로 다름.
  한 슬롯의 줄이 다른 슬롯의 `statepoints`를 들고 있으면 **slot-0 버그 재발**이다.
- 배치 4덱 출력이 단일덱 arm-ON 출력과 케이스별로 일치해야 한다(같은 arm이므로).

### G5 — 트림 덱 3종

```bash
for deck in kngr3 CY01 CY02; do
  for arm in off on; do
    env $( [ $arm = on ] && echo RASBERY_GPU_CRAM=1 ) <bld>/RASBERY \
      --rasi ~/t18decks/$deck.json --raso $O/${deck}_$arm.h5 > $O/${deck}_$arm.log 2>&1
  done
  python3 tools/gate_a_compare.py $O/${deck}_off.h5 $O/${deck}_on.h5 --per-step
  grep CRAM_GPU $O/${deck}_on.log
done
```

`host_fallbacks`가 0이 아닌 덱이 나오면 `status` 문자열이 이유를 말한다
(`ng != 2` / `niso != 39` / `substeps` / BOS 토큰 / 노드 상태). **폴백 자체는 결함이
아니다** — 결함은 폴백이 났는데 수신증이 그것을 말하지 않는 경우다.

### Gate B — MASTER 대조

```bash
python3 tools/compare_master_rasbery.py <MAS_SUM> $O/on_r1.h5 -o $O/master_cram_on
python3 tools/compare_master_rasbery.py <MAS_SUM> $O/off_new.h5 -o $O/master_cram_off
# BOC 핀은 기존 핀 스크립트로 별도 산출
```

**포락선 (v2/v3 기준선)**: 반응도 1.847–1.905 pcm, CBC 15.309–15.334 ppm, AO 0.012–0.013,
BOC 핀 RMS/max 0.238 % / 0.80 %. arm ON이 이 범위 **밖으로 나가면** N1 채택을 보류하고
G2의 밀도 차이부터 다시 읽는다.

### M64 — arm ON 처리량

```bash
RASBERY_GPU_CRAM=1 python3 tools/run_single_gpu_batch.py --batch-width 64 \
  --jobs ~/m256jobs.txt --cwd ~/t18decks/kngr --workdir $O/m64_on -- <bld>/RASBERY
grep -h "BATCH_OCCUPANCY\|cases_per_hour\|CRAM_GPU" $O/m64_on/*.log | tail -20
```

**기준선**: M64 518–534 c/h (light 577.6). arm ON이 **518 미만**이면 배치에서는 arm을
켜지 않는다 — 슬롯당 4슬롯 micro XS 사본(21.1 MB × 64 = 1.35 GB)과 로컬 메모리 풀이
폭을 먹고 있을 수 있고, 그때의 답은 §3.1의 상주성 공유이지 폭 축소가 아니다.

### 롤백

`unset RASBERY_GPU_CRAM` 하나. 코드 기본값은 OFF이고, OFF 경로는 G1이 h5 바이트
동일임을 증명한 그 경로다.

---

## 8. 빌드 없이 확인할 수 없었던 것 (명시)

로컬 PC/WSL에서 **컴파일도 실행도 하지 않았다**(사용자 규칙 + 이 머신에 C++ 컴파일러
없음). 아래는 전부 238에서 처음 확인된다.

1. **nvcc가 이 TU를 받아들이는지.** `tools/check_cuda_syntax.py`용 shim은 확장했지만
   (`cudaEvent_*`, `cudaStreamCreate`, `cudaGetDevice/SetDevice`, `atomicOr`,
   `__longlong_as_double`, `__constant__` 제거) **로컬에 `g++`가 없어 실제로 돌려보지
   못했다.** G0의 세 번째 명령이 238에서 이것을 처음 실행한다.
2. **`logb`/`scalbn`/`hypot`/`isfinite`/`isnan`/`isinf`/`copysign`이 디바이스 코드에서
   그 이름 그대로 해석되는지.** CUDA는 제공하지만 `<cmath>` 매크로와의 충돌은 툴킷
   버전마다 다르게 나타난다. 실패하면 `::logb` 등으로 한정하거나 `__nv_*` 내장을 쓴다.
3. **스레드당 로컬 메모리 실측치.** 설계 전제는 ~4.9 KB이고 `cuobjdump -res-usage`가
   답한다. 6 KB를 크게 넘으면 점유율이 무너지고 블록 32가 필요하다.
4. **성능이 실제로 나는지.** 8,451 스레드는 sm_120에 대해 **작다**. 커널은 지연 지배적일
   가능성이 높고, 이득의 크기는 G3만이 말한다. 이 문서는 배수를 하나도 주장하지 않는다.
5. **`__divdc3` 전사가 비트 일치를 내는지.** §7 G2의 밀도 max rel diff가 답한다.
   0이면 B0 승격 후보, ULP급이면 문서대로 N1, 1e-8을 넘으면 **전사 오류**다.
6. **`gs_iters_mean`이 호스트와 같은지.** 호스트에는 이 계기가 없다. 비교하려면
   `milk.h`에 임시 카운터를 넣은 디버그 빌드가 필요하며, **그것은 이 커밋에 넣지
   않았다**(생산 헤더에 계기를 남기지 않는다). G2의 밀도 일치가 간접 증거다.
7. **`RASBERY_PC_MODE=decart` / `RASBERY_PC_XE_EQUILIBRIUM_FIX` 경로.** 커널은 처리하지만
   캠페인 덱이 이 모드를 쓰지 않으므로 G1–G5 어디서도 **실행되지 않는다.** 쓰려면
   덱 하나를 그 모드로 돌려 별도 A/B를 해야 한다. 지금은 **미검증 경로**로 표시한다.
8. **`kNonFinite` 폴백이 실제로 발동하는 덱.** 정상 덱에서는 절대 발동하지 않는 것이
   정상이며, 발동을 보려면 인위적 입력이 필요하다. 계약 시험은 **순서**(stats 검사가
   D2H 앞)만 보증한다.
9. **배치 M64에서의 디바이스 메모리 총량.** 슬롯당 4슬롯 micro XS ×2(EOS+BOS) =
   42.2 MB + `iden` 계열 ~8 MB ≈ **50 MB/슬롯**, 64슬롯이면 **~3.2 GB**. 여기에 CUDA가
   `kPredictor`/`kCorrector`용으로 잡는 로컬 메모리 풀이 더해진다(컨텍스트당 1회,
   슬롯 간 공유). G4와 M64가 실측한다.

---

## 9. 커밋

| 커밋 | 내용 |
|---|---|
| (이 변경) | Task 16 GPU CRAM 감쇠 backend + XSSet 훅 + `kArmEnv` 등재 + 수신증 + 14규칙 계약 시험 + 이 문서 |

기준 tip: `8b8f18e` (`docs: what the campaign costs, where the time goes, and what the
tree looks like now`).
