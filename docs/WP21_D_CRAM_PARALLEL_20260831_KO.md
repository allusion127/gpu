# WP21-D — CRAM 노드 내부 재병렬화: 4극점 = 4레인

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `kPredictor` / `kCorrector` (CRAM Bateman 감손, `src/CudaCramBackend.cu`) |
| 플래그 | `RASBERY_GPU_CRAM_PARALLEL=pole4`(**기본**) \| `serial` · `RASBERY_GPU_CRAM_TIMING=1`(디버그 타이머) |
| 게이트 등급 | **B0** — 기존 커널(`serial`) 대비 **비트 동일**. arm 노브가 아니다 |
| 판정 | `h5diff -c` 0 차이 · digest **`1f36e75dc00ed2b4` / `4377`** 불변 · `gs_iters_mean` 불변 |
| 계약 테스트 | `tools/test_cram_parallel_contract.py`(신규, 11 규칙 · negative control 11) |
| 동반 테스트 | `tools/test_cram_gpu_contract.py`(stale needle 1곳 수리) · `tools/check_cuda_syntax.py`(warp intrinsic shim) |
| 소스 | `src/CudaCramBackend.cu` · `src/CudaCramBackend.h` · `src/Driver.h`(수신증 4필드) |
| 호스트 기준 | `include/milk.h:1649-1819` `solveBatemanCRAM` · `src/XSSet.cpp` `DepleteNode`/`CorrectorStep` (**읽기 전용**) |
| 기준 덱 | KNGR `kngr_238.json`, `nxyz = 8,451`, `NG = 2`, `niso = 39` |

> **이 문서가 주장하지 않는 것부터.** ncu가 보고한 `st 8.7 sectors/request`는 **비결합 전역
> 저장이 아니다**. 인벤토리 저장은 이미 `iden_out[iso * nxyz + l]` — 노드 인덱스가 최내곽인
> SoA — 이고, warp 32레인이 한 동위원소를 쓰면 256 B 연속, 즉 fp64의 **8 sectors/request
> 하한 그 자체**다. 8.7은 그 하한에 0.7이 붙은 값이며 붙은 쪽은 **로컬 메모리**다:
> `cval`/`ccol`의 `fill` 커서가 warp 안에서 갈라지기 때문이다(호스트의 `value == 0.0`
> 압축은 노드마다 다르다). 그러므로 **전치(transpose)는 하지 않았고 할 것도 없다.**
> WP15.1의 `FillCramMicDevice`/D2D 소비자, XSSet 호스트 리더, 행 `[first, niso)`을 한 번에
> 내리는 D2H — 어느 것도 레이아웃이 바뀌지 않았고, `docs/patches/wp21d_xsrecon.patch`는
> **필요하지 않으므로 생성하지 않았다**.

---

## 1. 노드 한 개의 일 — 무엇이 직렬이고 무엇이 아닌가

### 1.1 형상

| 항목 | 값 | 근거 |
|---|---|---|
| 동위원소 `niso` | **39** (`Chiffon::Isotope::niso`) | `kNiso`, 런타임에 덱과 대조 후 불일치 시 decline |
| 미지수 | **36** — 행 `[first, niso)`, `first = iI135 = 3` | H/B/O 3행은 그대로 복사 |
| 희소 패턴 | 노드 **불변**. `nonzeros(depDecay) ∪ 대각 ∪ nonzeros(depTrans) ∪ (n,2n) 특례 2개` | `minePattern()` |
| 합집합 크기 | **160** = 비대각 **124** + 대각 36, 한 행 최대 19 | 실측 |
| CRAM 차수 | 8 → **극점 4개**, `matrix_sgn = -1`, `alpha0 = 1.1722341374385704e-08` | `milk.h` `order == 8` |
| GS 상한 | `max_iter = 64`, `rel_tol = 1e-13`, `abs_tol = 1e-28`, `diag_tol = 1e-30` | 동상 |

패턴은 노드와 무관하지만 **값은 노드마다 다르고**, 호스트는 행을 훑으며 `value == 0.0`을
건너뛴다. 업로드된 패턴은 합집합(상위집합)이므로 커널은 **노드마다 그 0 판정을 다시 실행**한다
(`cramBuildSplit`). 그래야 압축된 행이 호스트의 행과 항목 단위로 같고, 산술이 "동등한 것"이
아니라 "같은 것"이 된다.

### 1.2 노드 본체의 순서 — 이것이 정확성 계약이다

```
DepleteNode(l):
  1. abs_flux[ig] = phif[l*ng+ig] * norm_factor            (ig 오름차순)
  2. invflux      = 1/Σ abs_flux                            (>0 일 때만)
  3. cond4[iso][s] = mic[s][iso,0,l]*f0 + mic[s][iso,1,l]*f1   (iso 오름차순, s = XSAF,XSFF,XS2N,XS3N)
  4. sumflux      = (Σ abs_flux) * 1e-24
  5. iden[i]      = iden_in[i*nxyz+l]                       (i 오름차순)
  6. solveBatemanCRAM(...)                                  ← 아래
  7. ApplyXeEquilibrium (xe_transient 아닐 때)
  8. iden_out[i*nxyz+l] = iden[i]                           (i = first..niso)

solveBatemanCRAM(A, N, dt, order 8, first):
  S. 분할: row = first..n-1 오름차순
       col ∈ 패턴(row) 오름차순:  value = A(row,col);  value == 0.0 이면 skip
                                  cval[fill++] = (-1) * value * dt
       mdiag[row] = (A(row,row) == 0.0) ? 0 : (-1)*A(row,row)*dt
  A. accum[row] = 0             (row = 0..n-1)
  P. pole = 0,1,2,3  **오름차순, 직렬**
       diag = mdiag[row] - theta[pole];  |diag| <= 1e-30 이면 throw(zero diagonal)
       rhs[row] = complex(N[row],0) * alpha[pole];  x[row] = rhs[row] / diag
       rhs_norm = max(rhs_norm, |rhs[row]|)  →  max(rhs_norm, 1e-30)
       iter = 0..63:
         (a) GS sweep:  row 오름차순
                sum = rhs[row]
                i = 0..cols(row)-1 **오름차순**:  sum -= vals[i] * x[cols[i]]   ← x는 이번 sweep의 최신값
                x[row] = sum / pole_diag[row]
         (b) 잔차 sweep: row 오름차순
                ax = pole_diag[row]*x[row] + Σ_i vals[i]*x[cols[i]]
                max_residual = max(max_residual, |rhs[row] - ax|)
         (c) max_residual <= abs_tol + rel_tol*rhs_norm 이면 converged, break
       미수렴이면 throw(did not converge)
       accum[row] += x[row]      (row = first..n-1 오름차순)  ← **극점 오름차순 누적**
  F. out[row] = alpha0*N[row] + 2.0*accum[row].real()
       value < 0 && |value| < 1e-12 이면 0
```

**직렬인 것 두 가지**: (a)의 `row`(행 i는 이번 sweep이 이미 갱신한 `x[j]`, j<i를 읽는다)와
그 안의 `i`(내적의 좌→우 누적). 둘 중 어느 하나라도 레인에 흩으면 **재결합(reassociation)** 이
일어나고, 재현이 아니라 다른 방법이 된다. **WP21-D는 이 둘을 건드리지 않는다.**

**직렬이 아닌 것 하나**: `P`의 극점 루프. 극점 4개는 같은 실수 행렬에 대한 **독립된 복소 선형
해**이고, 서로 공유하는 것은 `mdiag`와 압축된 비대각뿐이며, 유일한 결합은 `accum[row] +=
x_pole[row]`라는 **고정된 좌→우 4항 덧셈**이다.

또한 `accum`의 **허수부는 F에서 `.real()`만 취해지므로 어디서도 읽히지 않는다.** 죽은 합이며,
이번 커밋에서 제거했다(스레드당 로컬 312 B 절약, 결과 불변).

---

## 2. 매핑 — 노드 1 : 레인 4, 레인 == 극점

```
gtid = blockIdx.x*blockDim.x + threadIdx.x
l    = gtid / kLanesPerNode        (노드)
pole = gtid % kLanesPerNode        (극점, kLanesPerNode = kPoleCount = 4)
base = (threadIdx.x & 31) & ~3     (그룹의 첫 레인, warp 안)
mask = 0xF << base                 (그룹의 4레인)
```

`blockDim.x ∈ {32,64,128,256}`이고 4는 32를 나누므로 **그룹은 warp 경계를 넘지 않는다**. 꼬리
그룹은 노드 단위로 통째로 범위 밖이므로 `mask`가 이미 반환한 레인을 지목하는 일도 없다. 이것이
아래 모든 `__shfl_sync`/`__ballot_sync`의 전제이고, 계약 테스트 규칙 2가 지키는 것이다.

각 레인은

1. **프롤로그를 그대로 실행**한다(플럭스 정규화·`cond4` 응축·인벤토리 적재).
2. `cramBuildSplit`으로 **극점 불변 분할을 각자 재구성**한다.
3. 자기 극점 하나만 `cramPole`로 푼다 — GS는 여전히 `row` 직렬, `i` 직렬.
4. 그룹이 두 가지를 다시 모은다.

### 2.1 상태 — 첫 실패 극점

직렬 루프는 실패한 첫 극점에서 반환했다. 병렬 루프는 넷 다 돈다. 그래서 **ballot 위의
`__ffs`로 가장 낮은 실패 레인**을 고른다. 이것이 없으면 극점 0의 zero-diagonal 뒤에 있는
극점 3의 non-convergence가 decline 메시지를 뒤바꾼다. 스윕 카운트도 같은 지점에서 멈춘다 —
zero diagonal은 **첫 sweep 이전에** 거부하므로 스윕을 0 기여하고, non-convergence는 돈 64를
기여한다. `gs_iters_mean`은 "디바이스가 호스트와 같은 반복을 풀었다"는 수신증 관측량이므로,
둘 중 어느 쪽인지가 살아남아야 한다.

### 2.2 누적 — **비트 동일의 근거**

```cpp
// serial                                  // pole4
for (pole = 0; pole < 4; ++pole)           for (p = 0; p < 4; ++p)
  for (row = first; row < n; ++row)          for (row = first; row < n; ++row)
    accr[row] += xr[row];                      accr[row] += __shfl_sync(mask, xr[row], base + p);
```

같은 중첩, 같은 방향, 같은 0 초기화 누산기, 같은 네 피가수를 **같은 순서로**. `__shfl_sync`는
값을 옮길 뿐 산술을 하지 않는다. 그러므로 이것은 "동등한" 덧셈열이 아니라 **동일한** 덧셈열이고,
두 arm은 마무리(`cramFinish`)도 **같은 코드 한 벌**을 통과한다. 이것이 `pole4`가 `serial`에
대해 **B0**인 이유다.

> **B0가 아닌 것은 무엇인가.** 디바이스는 여전히 **호스트에 대해 N1**이다(복소 나눗셈이
> 디바이스 libm의 `logb`/`scalbn`을 통과한다 — `src/CudaCramBackend.h`). 두 주장은 동시에
> 참이다: 디바이스 vs 호스트 = N1, `pole4` vs `serial` = B0. 이번 작업은 후자만 만들었다.

### 2.3 8레인(실수/허수 분할)을 하지 않은 이유

실수/허수를 쪼개려면 GS **내적 안쪽**에서 shuffle이 필요하다. 그 자리의 호스트 산술은 복소
`sum -= vals[i] * x[cols[i]]`이고, 쪼개는 순간 재결합이 발생해 등급이 N1로 떨어진다. 점유율이
요구하지 않는 배수를 위해 등급을 파는 거래다 — 하지 않았다.

### 2.4 `jacobi` arm을 제공하지 않은 이유

계획서의 대안(레인을 동위원소에 걸치는 warp-per-node)은 **레인당 극점 매핑이 극점 합 순서를
보존하지 못할 때만** 필요하다. 보존한다(§2.2). Jacobi 스윕은 GS 의미론을 바꾸는 N1이므로,
필요 없는 N1을 기본 뒤에 숨겨 두지 않는다. `Variant` enum은 `kSerial`/`kPole4` **정확히 둘**
이고, 계약 테스트 규칙 9가 세 번째 arm이 조용히 끼어드는 것을 막는다.

### 2.5 극점 불변 분할을 공유 메모리에 두지 않은 이유 — 산수

노드당 공유해야 할 상태는 `cval`(192×8) + `ccol`(192) + `cend`(39×4) + `mdiag`(39×8) +
`cond4`(39×4×8) + `iden`(39×8) ≈ **3,756 B**. 64스레드 블록 = 16노드 → **60 KB**로 정적
48 KB 한도를 넘고, 32스레드 블록 = 8노드 → 30 KB에서는 100 KB 예산으로 SM당 3블록 = 96
스레드다. **지금보다 나쁜 점유율**을 사서 중복을 없애는 거래다.

반대로 중복의 비용은 작다: 분할은 노드당 `transEntry` 약 200회, 스윕은 **약 25,000회**의 복소
연산이다. 4배로 재구성해도 전체의 한 자리 %다. 그래서 **레인당 완전 복제**를 택했고, 공유
메모리는 쓰지 않으며 `__syncthreads()`도 없다(그룹은 warp 내부다).

---

## 3. 왜 이것이 옳은 수술인가 — ncu 블록 39의 진단

| 지표 | 값 | 읽는 법 |
|---|---|---|
| 형상 | 133 blocks × 64 threads (노드 8,451개, 스레드 1개씩) | GPU의 **약 2 %** |
| 시간 | `kPredictor` **4.38 ms** / `kCorrector` **4.49 ms** per launch | 런당 각 34 launch → 단일 약 **300 ms** |
| 배치 비중 | GPU 커널 시간의 **14.8 %** | 케이스당 4.4 ms × 2 × 51 statepoint ≈ **449 ms** |
| DRAM | peak의 **0.45 %** | 대역폭 바운드가 **아니다** |
| warps active | **4.15 %** | 지연 은닉이 전혀 안 된다 |
| ld / st sectors/req | **4.1 / 8.7** | 전역 저장은 8.0이 하한 — 초과분은 **로컬** 발산 |
| 점유율 한계 | **14** (레지스터 70) | 레지스터/로컬 압박 |

대역폭도 연산도 아니면 남는 것은 **직렬화**다. 노드 하나가 극점 4 × 최대 64 스윕 × 36행을
한 스레드로 걸어 내려간다. 처방은 "같은 산술을 더 넓게"이고, 그 넓이가 존재하는 유일한 축이
극점이다.

**기대치.**

- 레인 수 **8,451 → 33,804** (블록 64에서 **529 blocks**).
- 노드당 임계경로: 극점 4개 직렬 → 1개. 이론상 **÷4**.
- 되돌려 내는 비용: 프롤로그와 분할이 노드당 4회 실행(전체의 한 자리 %) + 그룹 리덕션
  (행당 shuffle 4회 × 36행 = 144회).
- **목표: launch당 4.4 ms → < 1.5 ms.** 넘으면 진단(직렬화 바운드)이 틀린 것이고, 그때
  다음 후보는 극점 불변 분할의 공유 메모리 이전이 아니라 **로컬 메모리 자체의 축소**다
  (`kMaxNnz = 192`는 실측 124에 대한 여유이고, `cval`은 스레드당 1,536 B다).

**떨어지는 것과 떨어지지 않는 것.**

- `warps active`: 4.15 % → 4배 근처까지. 이것이 측정의 1차 관측량이다.
- `st sectors/request`: 8.7 → 8.0 방향. 그룹 4레인의 `fill` 커서가 같아지므로 warp당 서로 다른
  커서가 **32개에서 8개로** 준다. **전역 저장 레이아웃 때문이 아니다**(§0의 경고).
- `dram`: 거의 그대로. 이 커널은 원래 DRAM을 안 쓴다.
- `gs_iters_mean`: **정확히 그대로여야 한다.** 움직이면 B0 주장이 무효다.

---

## 4. 수신증

`[RASBERY][CRAM_GPU]`에 네 필드가 붙었다.

```
"kernel_variant":"pole4","lanes_per_node":4,"launches":102,"launch_us_mean":1180.4
```

- `kernel_variant` / `lanes_per_node` — 어떤 매핑이 돌았는가. **arm이 아니라 매핑**이므로
  `RASBERY_GPU_CRAM_PARALLEL`은 `trajectory::kArmEnv`에 **의도적으로 없다**
  (`RASBERY_GPU_CRAM_BLOCK`과 같은 취급, `RASBERY_GPU_CRAM`과 반대). 넣으면 궤도가 같은 두
  실행이 다른 arm으로 갈라지고 케이스 키가 쪼개져 캐시가 통째로 무효가 된다.
- `launches` — decline한 호출도 센다. 나쁜 노드를 낸 launch도 시간은 썼다.
- `launch_us_mean` — **커널만**의 cudaEvent 시간. `RASBERY_GPU_CRAM_TIMING`이 없으면 **-1**
  ("측정하지 않음"은 "0 ms"와 전혀 다르게 읽혀야 한다). `wall_ms`는 예나 지금이나 전송·상태
  드레인을 포함한 호출 전체이며 ncu의 블록 39와 비교할 수 있는 숫자가 아니다.
- `gs_iters_mean`(기존) — 이 커밋의 **동치성 관측량**. `pole4`와 `serial`에서 같아야 한다.

---

## 5. 238 런북

```bash
export CUDA_VISIBLE_DEVICES=0
OUT=/path/to/run   # 로컬 디스크 금지: 결과는 E: 계열로
V6_ENV='RASBERY_PPR_MODE=master RASBERY_PC_MODE=decart RASBERY_GPU=1
RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 RASBERY_GPU_NODAL=1
RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1
RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8 RASBERY_GPU_WIEL_FOLD=chunked
RASBERY_GPU_XE=1 RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000
RASBERY_STAGED_LOOSE_SETTLE=1 RASBERY_OMP_THREADS=12 RASBERY_GPU_CRAM=1
RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_CMFD_FUSE=15
RASBERY_GPU_XE_TXN=1 RASBERY_RESULT_ASYNC=1 RASBERY_GPU_FLATXS_CTA=1
RASBERY_GPU_OUTER_GRAPH=1 RASBERY_GPU_MICX_RESIDENT=1 RASBERY_GPU_XFER_ELIDE=1
RASBERY_GPU_OUTER_SEGMENT_V2=1 RASBERY_GPU_CMFD_BLOCK=64'
```

### 5.0 소스 게이트 (하드웨어 없이)

```bash
python3 tools/test_cram_parallel_contract.py
python3 tools/test_cram_gpu_contract.py
python3 tools/test_enum_alias_contract.py
python3 tools/test_dependent_template_contract.py
python3 tools/test_xfer_ledger_contract.py
```

### 5.1 G0 — 매핑이 실제로 돌았는가

```bash
env $V6_ENV RASBERY_GPU_CRAM_TIMING=1 "$BLD/RASBERY" kngr_238.json -o "$OUT/a_pole4"
```

`[RASBERY][CRAM_GPU]`가 `"kernel_variant":"pole4","lanes_per_node":4`이고 `launches`가
`predictor_calls + corrector_calls`와 같아야 한다. `host_fallbacks`가 0이 아니면 **성능은 재지
않는다** — 호스트가 절반을 푼 런이다.

### 5.2 B0 판정 — 여기서 통과하지 못하면 나머지는 무의미하다

```bash
env $V6_ENV RASBERY_GPU_CRAM_PARALLEL=serial "$BLD/RASBERY" kngr_238.json -o "$OUT/b_serial"
h5diff -c "$OUT/a_pole4/kngr.h5" "$OUT/b_serial/kngr.h5"
```

합격 조건 — **전부**:

1. `h5diff -c` **0 차이**.
2. 양쪽 digest **`1f36e75dc00ed2b4` / `4377`**(feature-off 불변량과 동일해야 한다 — 이 커밋은
   궤도를 움직이지 않는다).
3. 양쪽 `[RASBERY][CRAM_GPU]`의 `gs_iters_mean` **동일**, `gs_solves` 동일, `nodes` 동일.
4. 결정론: `a_pole4`를 두 번 돌려 `h5diff -c` 0 차이(레인 그룹이 결과에 시간 의존성을 넣지
   않았다는 확인).

하나라도 어긋나면 **되돌린다.** B0 주장이 이 커밋의 전부다.

### 5.3 ncu — 블록 39 재측정

```bash
ncu --target-processes all --set full \
  --kernel-name "regex:kPredictorP4|kCorrectorP4" \
  -o "$OUT/cram_pole4_ncu" \
  env $V6_ENV "$BLD/RASBERY" kngr_238.json --result light

ncu --target-processes all --set full \
  --kernel-name "regex:kPredictor$|kCorrector$" \
  -o "$OUT/cram_serial_ncu" \
  env $V6_ENV RASBERY_GPU_CRAM_PARALLEL=serial "$BLD/RASBERY" kngr_238.json --result light
```

기록: **duration(launch당)**, **warps active**, **st sectors/request**, ld sectors/request,
achieved occupancy, registers/thread, local memory bytes/thread, block/grid.

| 지표 | serial(기준) | pole4(목표) |
|---|---|---|
| duration | 4.38 / 4.49 ms | **< 1.5 ms** |
| warps active | 4.15 % | > 12 % |
| st sectors/req | 8.7 | ≤ 8.3 |
| grid | 133 × 64 | 529 × 64 |

`launch_us_mean`(수신증)과 ncu duration이 서로 어긋나면 **수신증을 믿지 말고 둘 다 다시
잰다** — 그 불일치 자체가 버그다.

### 5.4 배치

```bash
python3 tools/run_multi_gpu_batch.py --batch-mode 16 --gpus 8 ... -- "$BLD/RASBERY"
```

기준: **8×M16, 1,321 c/h**. CRAM은 GPU 커널 시간의 14.8 %이고 케이스당 약 449 ms이므로,
launch당 4.4 → 1.2 ms면 케이스당 약 **326 ms**가 돌아온다. 그것이 처리량에서 몇 %로 보이는지는
배치가 커널 바운드인 정도에 달렸고 — **그 곱셈은 이 문서가 하지 않는다.** 측정이 한다.

---

## 6. 남은 것 (미착수, 이번 커밋 밖)

1. **로컬 메모리 축소.** `kMaxNnz = 192`는 실측 124에 대한 여유다. 스레드당 `cval` 1,536 B가
   그대로 남아 있고, 레지스터 70 / 점유율 한계 14의 실제 원인은 여기일 가능성이 높다.
   좁히려면 패턴 크기를 런타임 상한이 아니라 컴파일타임 상한으로 되돌려야 하는데, 그러면
   덱 의존성이 생긴다 — 별도 WP.
2. **극점 불변 분할의 공유 메모리 이전.** §2.5의 산수가 지금은 반대를 가리키지만, (1)이 로컬
   압박을 줄이고 나면 저울이 바뀔 수 있다.
3. **`RASBERY_PC_SUBSTEPS > 1`.** 여전히 decline이다. Isotalo 부분스텝 체인은 노드당 k회
   CRAM이고, 이 매핑은 그대로 확장되지만 캠페인 arm에 들어간 적이 없다.
