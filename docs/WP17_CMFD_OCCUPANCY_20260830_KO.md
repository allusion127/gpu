# WP17 — CMFD 점유율: 96 % "바쁜" GPU의 82 %는 비어 있다 (2026-08-30, KO)

브랜치 `codex/exact-throughput-campaign`, 기준 `8b3c62b`.
입력은 238 nsys/CUPTI 커널 레코드 한 벌이고, 이 문서는 **노브 2개의 구현**과
**그 노브로 측정해야 할 런북**이다. §5의 표는 채워질 표이지 채워진 표가 아니다.

---

## 0. 요약

1. **관측(238, RTX PRO 6000 Blackwell, 188 SM, v6 단일 덱 KNGR
   8,451 노드 × 2군, 35 statepoint, 4,377 outer, 18,627 CMFD sweep,
   74,508 BiCG iteration).** CMFD 커널은 전부 **그리드가 34·67·17·1 블록**이다.
   188 SM 중 **최대 18 %**만 쓰고, 커널 하나가 **2.5–4.3 µs**다.
   즉 이 커널들은 **연산 한계가 아니라 디스패치 한계**다. GPU 사용률 96 %는
   "SM이 차 있다"가 아니라 "커널이 끊이지 않는다"는 뜻이다.
2. **현행 분할은 노드당 스레드 1개이고, 블록은 색(colour)이 아니다.** §1.
   `colored_block_sweep`은 `target_color`를 **인자로** 받고 노드별 배열
   `colors[l]`로 거른다. 그래서 블록을 쪼개도 Gauss-Seidel 순서는 그대로다 — **B0**.
3. **리덕션은 다르다.** 분할이 `chunk = ceil(n / gridDim.x)`이고 트리가
   256 레인 고정이라 **블록 수를 바꾸면 합이 바뀐다 — N1**. 그래서 안 건드렸다. §2.3
4. **노브 2개. 기본값에서는 아무것도 바뀌지 않는다.**
   * `RASBERY_GPU_CMFD_BLOCK = 32|64|128|192|256` — 반복당 elementwise
     5개 클래스의 블록 폭. **미설정 = 0 = 종전 그대로**(`block_size`).
   * `RASBERY_GPU_CMFD_PERSISTENT = 1` — BiCGSTAB 반복 **1회 전체**를
     협조적(cooperative) 런치 **1개**와 `grid.sync()` 17개로. **기본 OFF**,
     그리고 캡처 그래프와 **상호 배타**다(§3.3).
5. **아직 측정치는 없다.** 블록 노브는 커널의 *꼬리*를 공격하고
   persistent 는 *바닥*을 공격한다. 어느 쪽도 예측으로 채택하지 않는다.

---

## 1. 현행 분할 — 8,451 노드 × 2군이 34 × 256에 어떻게 얹히는가

### 1.1 노드 도메인 (34 블록)

`nxyz = 8451`, `n = ngxyz = 2 * nxyz = 16902` (노드-메이저 2군 배치:
원소 `i = 2*l + ig`가 노드 `l`의 것). 생성자가 `n == 2 * nxyz`를 **선행조건으로
검사**한다.

per-node 커널은 전부 이 한 줄이다.

```cuda
const int l = blockIdx.x * blockDim.x + threadIdx.x;
if (l >= nxyz) return;
```

**스레드 1개 = 노드 1개 = 2군 모두.** 색으로 반씩 나누지도, 평면(plane)으로
자르지도 않는다. `ceil(8451 / 256) = 34`가 나온 곳이 여기다.
배치 축은 `gridDim.y`(= `lanes`)이고 x와 **절대 섞지 않는다**
(`node_grid()`의 주석이 그 규약이다).

| 클래스 | 도메인 | 256 | 192 | 128 | 64 | 32 |
|---|---|---:|---:|---:|---:|---:|
| `colored_block_sweep`, `matvec_two_group`, `prepare_p_jacobi`, `update_s_jacobi` | `nxyz = 8451` | **34** | 45 | 67 | **133** | **265** |
| `update_solution` | `n = 16902` | **67** | 89 | 133 | **265** | 529 |

188 SM 대비: 34 블록 = **18.1 %**, 67 = 35.6 %.
배치(8 MPS 클라이언트)에서도 8 × 34 = 272 블록 — 시분할이므로 한 시점에
차 있는 것은 여전히 그 근처다.

### 1.2 색(colour)은 데이터이지 런치 기하가 아니다

이것이 이 WP 전체가 걸린 사실이다.

```cuda
__global__ void colored_block_sweep(..., const int target_color,
                                    const int* __restrict__ colors, ...) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz || colors[l] != target_color) return;
```

* `colors[]`는 `init()`의 **greedy BFS 그래프 채색**이 노드마다 한 번 정한
  값이다. 이분 격자에서는 역사적 red/black 패리티와 **비트 단위로 같은** 채색이
  나오고, 90° 회전 1/4 노심 접합처럼 동일 패리티 간선이 생기는 곳에서만
  3색 이상으로 열린다.
* `init()`은 **인접 두 노드가 같은 색이면 throw** 한다. 따라서 한 색 안에서는
  어떤 스레드도 같은 런치의 다른 스레드가 쓰는 노드를 읽지 않는다.
* `precondition_sweeps()`가 `sweep % ncolors`로 색을 순회하고,
  **커널 경계가 곧 색 사이의 배리어**다.

⇒ **한 색을 더 많은 블록에 쪼개는 것은 순서를 바꾸지 않는다.** 같은 색,
같은 1회 갱신, 이웃 값은 이전 색 런치가 남긴 그 값.

### 1.3 리덕션의 고정 분할 (17 블록)과 stage2가 1×1인 이유

```cuda
constexpr int kReduceThreads = 256;
int reduce_blocks_for(n) { blocks = ceil(n / (256*4)); clamp [1, 256]; }
```

`n = 16902` → `ceil(16902 / 1024) = 17`. 커널 안에서:

```cuda
const int chunk = (n + gridDim.x - 1) / gridDim.x;   // 995
const int begin = blockIdx.x * chunk;                 // 0, 995, 1990, ...
for (int i = begin + threadIdx.x; i < end; i += blockDim.x) sum += am[i]*bm[i];
// 고정 이진 트리 (stride 128, 64, ..., 1)
```

세 가지가 **동시에** 결과를 고정한다: (a) `chunk` — `(n, gridDim.x)`만의 함수,
(b) 스레드별 stride 순회 — `blockDim.x`의 함수, (c) 256 레인 고정 이진 트리.
`gridDim.y`(배치 축)는 `chunk`에 **들어가지 않는다**. 배치가 단일과
비트 단위로 같은 이유가 이 한 줄이다.

**stage2가 `<<<1, 1>>>`인 이유는 결정론이다.**

```cuda
for (int i = 0; i < blocks; ++i) sum += pm[i];   // strict index order
```

17개 부분합을 **엄격한 오름차순 직렬 합**으로 접는다.
`cublasDdot`/`Dnrm2`가 실행마다 그리드와 누적 전략을 heuristic으로 고르기 때문에
마지막 몇 ulp가 흔들리고, 임계 탐색 outer가 그 ulp를 keff 산포로 증폭시킨다 —
그 문제를 없애려고 직렬로 고정한 것이다. `WIEL_FOLD=chunked`가 **정확히 반대
선택**(pairwise, O(log n) 오차 한계로 더 정확하지만 비트가 다르므로 N1 arm)을
한 자리이고, 이 stage2는 **serial 쪽에 남아 있는 자리**다.

### 1.4 반복 1회의 디스패치 예산

`FUSE=15`, `scalar fusion on`, `rb_sweeps = R`:

| 항목 | 개수 |
|---|---:|
| `dot(r0,r)`, `dot(r0,v)` (fused) | 2 |
| `dot2(s·t, t·t)` (fused) | 1 |
| `colored_block_sweep` | 2R |
| `prepare_p_jacobi`, `matvec` ×2, `update_s_jacobi`, `update_solution` | 5 |
| 잔차 `reduce_dot_stage1` | 1 |
| `reduce_norm_accumulate_stage2` | 1 |
| **합** | **10 + 2R** (R=4 → **18**) |

`launchesPerIteration()`가 이 모델을 소스에서 그대로 계산하고
`[RASBERY][CMFD][OCCUPANCY]`가 그 수를 찍는다.

**관측된 R.** 238 레코드는 `colored_block_sweep` 379,027 / 74,508 = **5.09**
회/반복이다. `2R = 5.09`이므로 그 런의 실효 R은 4가 아니라 **≈2.5** — 즉
런타임 R이 4가 아니었거나 halt로 조기 종료된 반복이 섞였다는 뜻이고,
런북에서 `[CMFD][OCCUPANCY]`의 `launches_per_iteration`으로 확정한다.

### 1.5 238 실측 (근거 표)

| 커널 | 런치 | 평균 µs | 합 ms | 그리드 | 블록 |
|---|---:|---:|---:|---:|---:|
| `colored_block_sweep` | 379,027 | 2.55 | **967.6** | **34** | 256 |
| `matvec_two_group` | 95,405 | 2.75 | 262 | 34 | 256 |
| `reduce_dot_fused` | 94,991 | 3.48 | 331 | 17 | 256 |
| `reduce_dot2_fused` | 47,925 | 4.32 | 207 | 17 | 256 |
| `reduce_dot_stage1` | 59,326 | 2.68 | 159 | 17 | 256 |
| `reduce_norm_accumulate_stage2` | 47,461 | 2.84 | 135 | **1** | **1** |
| `prepare_p_jacobi` | 47,483 | ~2.5 | ~119 | 34 | 256 |
| `update_s_jacobi` | 47,507 | ~2.5 | ~119 | 34 | 256 |
| `update_solution` | 47,514 | ~2.5 | ~119 | 67 | 256 |
| **합** | **866,639** | | **≈ 2.42 s** | | |

`kernelFlatXsCta<128>` 384회 × 280 µs, 그리드 8,451 — 이쪽은 정상이고
WP17의 대상이 아니다.

---

## 2. B0인 것과 N1이 될 것

### 2.1 B0 — 반복당 elementwise 5개 클래스의 재-블록화

`colored_block_sweep`, `matvec_two_group`, `prepare_p_jacobi`,
`update_s_jacobi`, `update_solution`. 근거 셋:

1. **인덱스만 바뀌고 집합은 그대로.** 인덱스는
   `blockIdx.x*blockDim.x + threadIdx.x`이고 `>= nxyz`(또는 `>= n`)로 가드된다.
   어느 블록이 어느 노드를 갖느냐만 바뀌고, **닿는 노드의 집합과 각 노드가
   계산하는 식은 같다**.
2. **블록 내 리덕션이 없다.** 다섯 중 어느 것에도 `__shared__`,
   `__syncthreads()`, warp shuffle이 없다. 재결합될 피연산자 짝이 **존재하지
   않는다**.
3. **스레드 간 쓰기는 두 종류뿐이고 둘 다 분할에 무관하다.**
   `atomicOr(flags + m, ...)`는 교환·멱등이라 블록 은퇴 순서와 무관하고,
   `l == 0` / `i == 0` 스칼라 저장은 **인덱스**에 걸려 있어 그 스레드가 어느
   블록에 있든 정확히 한 번 실행된다.

계약 시험 `rule_colour_sweep` / `rule_five_classes`가 1과 2를 소스에서 고정한다.

### 2.2 N1이 될 것 — 하지 않은 일

* **블록 = 색 그룹으로 만드는 것.** 지금은 아니다(§1.2). 만약 색 선택이
  `colors[l]` 대신 `blockIdx.x % ncolors`로 바뀌면, 블록 수를 바꾸는 순간
  Gauss-Seidel 순서가 바뀌고 **재-블록화는 그대로 N1이 된다.**
  계약 시험이 `colors[l] != target_color` 필터를 명시적으로 고정하는 이유다.
* **두 색을 한 런치로 합치는 것.** sweep k+1은 이웃 노드의 `x`를 읽으므로
  sweep k에 격자 전체로 의존한다. 커널 경계가 그 배리어다 — WP7의
  `NOT FUSABLE` 판정 그대로.
* **`rb_sweeps`나 `ncolors`를 바꾸는 것.** 물리 스케줄이다.

### 2.3 N1이라 손대지 않은 것 — 리덕션

과제가 제안한 "17개 부분합 경계를 유지한 채 각 부분합을 k개 블록이 계산"은
**이 구현에서는 불가능하다**. 이유:

부분합 `P_b`는 `sum over t = 0..255 of s_t`를 **256 레인 고정 이진 트리**로
접은 값이고, `s_t`는 `i = begin+t, begin+t+256, ...`의 직렬합이다.
트리의 첫 stride가 128, 즉 레인 t와 t+128을 짝짓는다 — **한 블록의 shared
메모리 안에서만 성립하는 짝짓기**다. 부분합 하나를 2개 블록에 쪼개면 그 짝이
블록을 가로지르므로, 같은 순서를 유지하려면 256개 레인합을 전역 메모리에
쓰고 같은 트리를 다시 재현하는 2단계가 필요하다 — **런치 1개가 2개가 되고**,
지금 이 커널들은 런치 한계이므로 정확히 반대 방향이다.

**정량.** 리덕션 계열은 202,242 런치 / **697 ms**다. 병렬도를 17→34로 올려도
커널당 실행시간이 절반이 되는 것이 아니라(이미 2.7–4.3 µs = 디스패치 근처),
**런치 수는 그대로**다. 기대 절감 ≈ 0. 따라서 **그대로 둔다.**

**`reduce_norm_accumulate_stage2`의 융합은?** 이미 되어 있다.
`RASBERY_GPU_CMFD_FUSE` bit 0/1(`reduce_dot_fused`, `reduce_dot2_fused`)이
정확히 그 기법(퇴장 카운터 + 마지막 티켓을 뽑은 블록이 동일한 직렬 fold 수행)
이고, WP7이 6항 논증과 함께 이미 채택했다. 잔차 노름 경로만 남아 있는데
그것도 `scalar_fusion`이 `reduce_dot_stage2 + accumulate_iteration`을 이미
한 노드로 합쳐 두었다. **stage1 꼬리로 한 번 더 접는 것**은 bit 0과 같은
논증으로 B0지만, 잔차 fold는 `active`/`halt` 판정과 `kOverrunCount` 증가를
동반하므로 halt된 슬롯에서 "부분합을 쓴 블록이 하나도 없는데 마지막 티켓을
뽑는" 경우를 따로 처리해야 한다. **한 노드/반복(74,508 × 2.84 µs ≈ 135 ms)**
이 걸린 자리이고, persistent arm이 성공하면 이 노드는 통째로 사라진다.
그래서 이번에는 **하지 않았다** — 두 길이 같은 목적지를 겨누고 있고
persistent 쪽이 18배 크다.

### 2.4 persistent arm의 B0

`bicg_iteration_persistent` 위의 **ORDER-PRESERVATION NOTE 6항**이 논증이다.
요약: (1) 모든 stage 본문은 참조 커널에서 **글자 그대로 복사**했고 같은 TU,
같은 `--fmad=false`다. (2) per-node stage의 grid-stride는 §2.1과 같은 논증.
(3) 리덕션 분할은 그리드가 아니라 `reduce_blocks`(= `reduce_blocks_for(n)`)에
**고정**되고 `blockDim.x = kReduceThreads`가 강제된다. (4) fold는 같은 직렬
fold를 **모든 스레드가 중복 수행**한다 — 엄격한 순서 + 동일 피연산자는
결정론적 함수이므로 모두 같은 비트를 얻고, 배리어 하나를 아낀다.
(5) `grid.sync()`가 커널 경계 자리에 1:1로 서고 device-wide fence를 포함한다.
(6) 배리어 앞의 모든 early return은 **grid-uniform**이다(`lanes != 1`을
거부하는 이유).

---

## 3. 구현

### 3.1 어디에 걸었나

| 항목 | 위치 |
|---|---|
| 블록 폭 게이트 | `src/CudaBICGBackend.cu` — `cmfdBlockThreads()` |
| 그리드 헬퍼 | `cmfd_block_threads()`, `cmfd_node_blocks()`, `cmfd_vector_blocks()`, `cmfd_node_grid()`, `cmfd_vector_grid()` |
| persistent 게이트 | `cmfdPersistentRequested()` |
| 거부 사다리 | `enum class PersistentRefusal` + `persistentRefusalName()` |
| 협조적 커널 | `__global__ void bicg_iteration_persistent(...)` + `persistentDotStage1/Dot2Stage1/Fold/ColourSweepNode/MatvecNode` |
| 무장 | `BatchCore::armPersistent(const cudaDeviceProp&)` (init 안, 1회) |
| 런치 | `BatchCore::enqueuePersistentIteration()` — `cudaLaunchCooperativeKernel` |
| 영수증 | `[RASBERY][CMFD][OCCUPANCY]` 1줄 + `[RASBERY][CMFD][GRAPH]`에 4필드 추가 |
| 계약 시험 | `tools/test_cmfd_occupancy_contract.py` |

`-rdc=true`는 이미 켜져 있다(`CMakeLists.txt`의 `CUDA_SEPARABLE_COMPILATION ON`),
그래서 `cg::grid_group::sync()`는 **새 제약이 아니다**.

### 3.2 왜 `RASBERY_GPU_BLOCK_SIZE`가 아니라 새 이름인가

기존 `RASBERY_GPU_BLOCK_SIZE`(64/128/192/256)는 **모든** per-node 커널의 폭이다 —
sweep 조립 계열(`cmfd_assemble_operator_2g`, `cmfd_src_build`, `cmfd_wiel_terms`,
`cmfd_updls`)과 outer 프롤로그(`begin_outer_fused`)까지 함께 움직인다.
그것들은 outer/sweep당 1회이므로 런치 바닥 위에 있지 않다. `CMFD_BLOCK`은
**반복당 5개 클래스만** 좁히고, **32**를 허용한다(265 블록 arm에 필요).
둘은 독립이고, 미설정 시 `CMFD_BLOCK`은 `BLOCK_SIZE`가 정한 폭을 그대로 쓴다.

### 3.3 persistent 와 캡처 그래프는 상호 배타다

협조적 런치는 **stream capture에 기록될 수 없다.** 그래서:

* `armPersistent()`가 `use_graph`(= `RASBERY_GPU_GRAPH != 0`)이면
  `OuterGraphActive`로 **무장 자체를 거부**한다.
* `enqueuePersistentIteration()`이 런치 직전 `graphCaptureActive(stream)`을
  한 번 더 보고 `CaptureActive`로 거부한다 — `RESIDENT_SINGLE` 세그먼트가
  나중에 캡처를 여는 경우 때문이다.
* 거부는 **아무것도 enqueue 하지 않고** `false`를 돌려주므로, 호출자의
  다음 문장이 그대로 참조 런치 체인이다. `launch_outer`의 캡처 폴백이 기대는
  것과 같은 CUDA 의미(캡처 중인 스트림에 제출된 일은 실행되지 않고 기록만
  된다)에 기댄다.

⇒ **런북에서 `PERSISTENT=1`은 반드시 `RASBERY_GPU_GRAPH=0` / `OUTER_GRAPH=0`과
함께 간다.**

### 3.4 거부 사다리 (이름)

`none` · `arm_off` · `outer_graph_active` · `capture_active` · `batch_width` ·
`no_cooperative_launch` · `occupancy_too_small` · `block_width_mismatch` ·
`fp32_inner` · `launch_failed`.

`persistentRefusalName()`에 `default:`가 없다 — 새 사유를 추가하면
**이름을 붙일 때까지 컴파일이 안 된다**. 계약 시험이 그것과, "선언은 했는데
어디서도 대입하지 않는 사유"(영수증이 영원히 찍을 수 없는 이름)를 함께 막는다.

### 3.5 영수증

```
[RASBERY][CMFD][OCCUPANCY] {"block_threads":256,"sweep_block_threads":256,
 "node_blocks":34,"vector_blocks":67,"reduce_blocks":17,"scalar_blocks":1,
 "launches_per_iteration":18,"persistent_arm":0,"persistent_blocks":0,
 "cooperative_supported":1,"persistent_refusal":"outer_graph_active"}
```

1회만 찍힌다(`std::atomic<bool>::exchange`). `[RASBERY][CMFD][GRAPH]`에는
`block_threads` / `node_blocks` / `vector_blocks` / `persistent_arm`이
추가되어, 그래프가 만들어진 런은 한 줄로 자기 기하를 말한다.
두 영수증 모두 **궤적 중립**이다 — 스트림을 건드리는 토큰이 하나도 없고
계약 시험이 그것을 검사한다.

---

## 4. 기대 효과 (산술, 예측 아님)

### 4.1 블록 노브 — *꼬리*를 공격한다

런치 **수**는 바뀌지 않는다. 바뀌는 것은 커널 하나의 실행시간이다.
34 블록 × 256 스레드 = 8,704 스레드가 **34개 SM**에 얹혀 있다.
64로 낮추면 133 블록 × 64 = 8,512 스레드가 **133개 SM**에 흩어진다 —
총 스레드 수는 같고 SM당 일이 4분의 1이 된다. 따라서:

* 2.55 µs 중 **실행 부분**(메모리 지연 꼬리)은 최대 4배까지 짧아질 수 있다.
* 2.55 µs 중 **디스패치 바닥**(W0 probe 1이 측정한 `c_dispatch ≈ 0.914 µs`)은
  **전혀 바뀌지 않는다**.
* 상한: `colored_block_sweep` 967.6 ms 중 바닥을 뺀 부분
  = 379,027 × (2.55 − 0.914) µs ≈ **620 ms**. 그 중 얼마가 회수되는지가
  §5.2가 측정할 값이다. 전체 CMFD 2.42 s 기준 상한은 대략 **1.1 s**.
* **위험 두 가지.** (a) 32스레드/블록은 블록당 1 warp라 ILP가 죽는다 —
  64가 유력하다. (b) 배치에서는 8 × 133 = 1,064 블록으로 과잉구독되어
  스케줄링 비용이 늘 수 있다. **배치는 예측하지 않고 측정한다.**

### 4.2 persistent arm — *바닥*을 공격한다

반복당 디스패치 **18 → 1**. 관측된 866,639 런치 / 74,508 반복 = 11.6 →
반복당 1이면 **약 792,000 디스패치가 사라진다**.

```
removable = N_node * c_dispatch  -  N_barrier * c_barrier
```

| 항 | 값 |
|---|---|
| `N_node` (제거되는 디스패치) | ≈ 792,000 (단일) |
| `c_dispatch` (W0 probe 1) | 0.914 µs |
| `N_barrier` (반복당 17) | 74,508 × 17 ≈ 1.27 M |
| `c_barrier` | **미측정** — `tools/probe_gridsync_cost.cu`가 그 수다 |
| 게이트 | `c_barrier ≤ 0.384 µs` (Rev.7.1) |

* `c_dispatch` 항: 792,000 × 0.914 µs = **0.72 s**.
* 게이트 값에서의 배리어 항: 1.27 M × 0.384 µs = **0.49 s**.
* 순 절감 ≈ **0.23 s** — **얇다.** 게이트를 아슬아슬하게 통과하는 값에서는
  채택할 이유가 없고, `c_barrier`가 0.2 µs 근처로 나와야 의미가 생긴다.
* 커널 *지속시간* 기준(2.5 µs)으로 세면 상한은 훨씬 크지만(≈2 s), 그 2.5 µs
  중 실제 일이 얼마인지를 배리어가 대신 물려받으므로 **그 산술은 상한일 뿐
  판정 근거가 아니다.** 판정은 `c_dispatch`/`c_barrier` 쪽이다.
* 과제가 제시한 "2.5 µs × 8 런치 × 74,508 ≈ 1.5 s"는 같은 수를 커널
  지속시간과 8-런치 모델로 센 것이고, 위 표의 상한 계열에 속한다.

**배치(×64).** 8 프로세스 × 8 슬롯 폭이 각각 자기 런치를 내므로 디스패치 항은
클라이언트 수만큼 곱해진다. 다만 persistent arm은 `lanes != 1`을
`batch_width`로 **거부**한다 — 배리어가 배치 축을 가로지르는데 lane마다 halt
시점이 다르기 때문이다. 따라서 **배치에서 persistent 는 프로세스당 단일 덱
(8×M8, 각 프로세스 batch-width 1)에서만 의미가 있고**, 현재의 M8 배치에는
그대로 적용되지 않는다. 이것이 이 spike의 가장 큰 실무적 한계다.

---

## 5. 238 런북

### 5.0 공통

* 계산은 **로컬에서 하지 않는다**. 238에서 돌리고 출력은 `E:` 쪽 run 디렉터리.
* 두 노브 모두 **`DEFAULT_ENV`에 넣지 않았다**. 평범한 `--set` 오버라이드로 전달.
* 매 arm마다 `[RASBERY][CMFD][OCCUPANCY]`를 **로그에 남기고 표에 함께 적는다**.
  `block_threads`가 요청과 다르거나 `persistent_refusal`이 `none`이 아닌데
  `persistent_arm:1`이면 그 행은 데이터가 아니다.
* **B0 확인은 매 arm 필수.** 기준 arm 대비
  digest `1f36e75dc00ed2b4` / outers **4377**이 같아야 하고,
  `h5diff` 차이 **0줄**, CSV `cmp` 동일이어야 한다. 다르면 노브의 결함이지
  물리가 아니다.

```sh
# 매 arm 끝에
python tools/exact_audit.py "$ARM/…" | tee "$ARM/digest.txt"     # 1f36e75dc00ed2b4 / 4377
h5diff "$BASE/out.h5" "$ARM/out.h5" | tee "$ARM/h5diff.txt"      # 0 lines
cmp "$BASE/summary.csv" "$ARM/summary.csv"                        # exit 0
```

### 5.1 단일 덱: v6 ± `CMFD_BLOCK`

```sh
for B in "" 192 128 64 32; do
  python tools/run_single_gpu_batch.py --gpu 0 --batch-width 64 \
      --jobs one_deck.txt --workdir "$RUN/wp17_block_${B:-off}" \
      ${B:+--set RASBERY_GPU_CMFD_BLOCK=$B} -- ./build/RASBERY
done
```

| `CMFD_BLOCK` | node blocks | vector blocks | hot1/2/3 (s) | median (s) | digest | outers | h5diff | 판정 |
|---|---:|---:|---|---:|---|---:|---:|---|
| 미설정 (256) | 34 | 67 | | | `1f36e75dc00ed2b4` | 4377 | — | 기준 |
| 192 | 45 | 89 | | | | | | |
| 128 | 67 | 133 | | | | | | |
| **64** | **133** | **265** | | | | | | |
| 32 | 265 | 529 | | | | | | |

**읽는 법.** 가설이 맞다면 64에서 median이 내려가고 32에서 다시 올라간다
(1 warp/block의 ILP 손실). 어느 arm에서도 digest/h5diff가 흔들리면
**즉시 중단** — §2.1의 논증이 틀린 것이다.

### 5.2 단일 덱: v6 ± `PERSISTENT` (그래프 OFF에서만)

**전제.** `c_barrier`가 먼저 있어야 한다.

```sh
nvcc -O3 -std=c++17 -arch=sm_120 -rdc=true \
     -o /tmp/probe_gridsync_cost tools/probe_gridsync_cost.cu -lcudadevrt
CUDA_VISIBLE_DEVICES=0 /tmp/probe_gridsync_cost | tee "$RUN/wp17_cbarrier.jsonl"
```

`c_barrier > 0.384 µs`면 §4.2의 순 절감이 음수다 — **그 자리에서 NO-GO**이고
아래 arm은 돌리지 않는다.

```sh
# (a) 그래프 OFF 기준선  — persistent 없이, 그래프만 끈 값
python tools/run_single_gpu_batch.py --gpu 0 --batch-width 1 \
    --jobs one_deck.txt --workdir "$RUN/wp17_nograph" \
    --set RASBERY_GPU_GRAPH=0 --set RASBERY_GPU_OUTER_GRAPH=0 \
    -- ./build/RASBERY

# (b) persistent arm
python tools/run_single_gpu_batch.py --gpu 0 --batch-width 1 \
    --jobs one_deck.txt --workdir "$RUN/wp17_persistent" \
    --set RASBERY_GPU_GRAPH=0 --set RASBERY_GPU_OUTER_GRAPH=0 \
    --set RASBERY_GPU_CMFD_PERSISTENT=1 -- ./build/RASBERY
```

`--batch-width 1`이 필수다 — 그렇지 않으면 arm이 `batch_width`로 거부하고
(b)는 (a)의 재실행이 된다. 영수증의 `persistent_arm:1`로 확인한다.

| arm | `persistent_arm` | `persistent_refusal` | `persistent_blocks` | median (s) | digest | outers | h5diff |
|---|---:|---|---:|---:|---|---:|---:|
| (a) graph off | 0 | `arm_off` | 0 | | | | 기준 |
| (b) + PERSISTENT | **1** | `none` | | | | | **0** |
| (c) PERSISTENT + GRAPH=1 (음성 대조) | **0** | `outer_graph_active` | 0 | ≈(v6) | | | |

(c)는 상호 배타가 **실제로 거부로 나타나는지** 확인하는 음성 대조다.
여기서 `persistent_arm:1`이 찍히면 §3.3이 깨진 것이다.

### 5.3 배치: 8×M8 + MPS, v6 ± `CMFD_BLOCK`

persistent 는 배치 arm에 없다(§4.2 마지막 문단).

```sh
python tools/run_multi_gpu_batch.py \
    --gpus 0 --procs-per-gpu 8 --batch-width 8 --mps \
    --jobs manifest.txt --pin taskset \
    --workdir "$RUN/wp17_batch_block-${B:-off}" \
    ${B:+--set RASBERY_GPU_CMFD_BLOCK=$B} \
    -- ./build/RASBERY
```

| `CMFD_BLOCK` | 동시 블록 (8×) | c/h | digest | outers | h5diff |
|---|---:|---:|---|---:|---:|
| 미설정 (256) | 272 | | `1f36e75dc00ed2b4` | 4377 | 기준 |
| 128 | 536 | | | | |
| 64 | 1,064 | | | | |

**읽는 법.** 단일에서 이긴 폭이 배치에서 지면, 이긴 것은 SM 확산이 아니라
단일 덱의 지연 꼬리였다는 뜻이다. 그 경우 노브는 **단일 전용**으로 문서화하고
`DEFAULT_ENV`에 넣지 않는다.

### 5.4 nsys 그리드 표 (사후)

채택 후보 arm 하나에 대해:

```sh
nsys profile -t cuda,nvtx -o "$RUN/wp17_after" --force-overwrite true \
    ./build/RASBERY <deck>
nsys stats --report cuda_gpu_kern_sum "$RUN/wp17_after.nsys-rep" \
    | tee "$RUN/wp17_after_kern_sum.txt"
```

§1.5의 표를 **같은 열로** 다시 채우고 나란히 둔다. 확인할 것은 두 가지:
(a) `colored_block_sweep`의 그리드가 34에서 133으로 바뀌었는가,
(b) 런치 **수**가 그대로인가(블록 노브) 또는 1/11.6로 줄었는가(persistent).

---

## 6. 계약 시험

```sh
python tools/test_cmfd_occupancy_contract.py
python tools/test_enum_alias_contract.py
python tools/test_dependent_template_contract.py
python tools/test_cmfd_fuse_contract.py
python tools/test_xfer_ledger_contract.py
```

`test_cmfd_occupancy_contract.py`가 고정하는 것: 기본값 불변, 색 순서 보존,
리덕션 순서 보존, 거부 이름(도달 가능성 포함), 협조적 런치 가드,
영수증 필드, 그리고 이 문서의 존재와 34/133/265 표. 규칙마다 **음성 대조**가
붙어 있어, 아무것도 맞히지 않게 된 규칙은 통과가 아니라 실패한다.

**새 memcpy 없음** — WP17은 전송을 하나도 추가하지 않으므로
`test_xfer_ledger_contract.py`의 site 수는 변하지 않는다(26, `CudaBICGBackend.cu`).

---

## 7. 미해결

1. `c_barrier`가 없다. §5.2의 전제이고, 없으면 persistent arm은 코드일 뿐이다.
2. persistent arm이 `lanes == 1`만 서비스한다. 배치로 확장하려면 배리어를
   lane별로 쪼개거나(cooperative group의 부분집합 동기화) halt를 grid-uniform
   하게 만들어야 한다 — 둘 다 별도 WP다.
3. `reduce_norm_accumulate_stage2`의 stage1 꼬리 융합(§2.3)은 남겨 두었다.
   persistent 가 NO-GO로 끝나면 그것이 다음 후보다 (135 ms).
4. 블록 노브가 FP32 inner 경로에도 걸려 있다(`*_f32` 6개). FP32 arm은
   이번 런북에 없다 — 채택 시 별도 확인이 필요하다.
