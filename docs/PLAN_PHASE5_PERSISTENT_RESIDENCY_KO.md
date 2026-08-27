# Phase 5 상세 계획 — Persistent/Cooperative CMFD 커널과 Outer 루프 전체 상주화

**작성일**: 2026-08-27 | **브랜치**: `codex/exact-throughput-campaign` | **상위 계약**: `docs/GPU_RASBERY_EXACT_THROUGHPUT_ACCELERATION_PLAN_REV4_KO.md` §11 (Phase 5 `[HOLD]`)
**대상 하드웨어**: RTX PRO 6000 Blackwell Server Edition (sm_120), CUDA 13, 238 서버 GPU0 단독
**설계**: Opus (본 문서) | **배치·계측**: Sonnet (238 GPU0 전용, GPU1 사용 금지)

---

## 0. 이 문서의 위치와 — 먼저 말하는 — 정직한 상한

Rev.4 §11은 Phase 5를 `[HOLD]`로 두고 착수 조건을 Nsight 실측에 걸어 두었다. 본 문서는 그 실측을 **어떻게** 하는지(Stage 0), 조건이 충족되었을 때 **무엇을** 구현하는지(Stage 1~3), 그리고 **각 단계가 실제로 값어치가 있는지**를 코드를 읽어 확인한 사실 위에서 정한다.

문서를 읽기 전에 알아야 할 네 가지 결론을 먼저 적는다. 이 네 가지가 이후 모든 설계 판단을 지배한다.

### 결론 1 — 단일덱과 배치는 서로 다른 코드 경로를 탄다. Phase 5의 이득도 서로 다르다

`BICGSolver::driveSweepsCuda`는 `_arena == nullptr`이면 **무조건 `false`를 반환한다**(`src/BICGSolver.cpp:336`). 그리고 `BICGCMFD::canUseDeviceAssembly()`는 `_ls->arena() != nullptr`을 요구한다(`src/BICGCMFD.cpp:215`). 단일덱 실행은 `--batch-mode`를 주지 않으면 `g_batch_width == 0`이므로 아레나가 만들어지지 않는다(`src/CudaBICGBackend.cu:3687`, `src/BICGSolver.cpp:53`).

> **따라서 지금 측정된 단일덱 94.6 s는 device-resident sweep 경로(`RASBERY_GPU_CMFD_SWEEP`)도, device operator assembly도 전혀 쓰지 않은 수치다.** setls 12.1 s는 그래서 호스트에서 전부 계산되고 있고, sweep마다 `synchronizeCudaFlux` + 호스트 wiel/updls/facilu로 스트림을 드레인한다.

이것은 Phase 5를 시작하기 전에 **코드 한 줄 없이** 확인할 수 있는 실험이 존재한다는 뜻이다 (Stage 0-A, §2.2). 우선순위상 이것이 먼저다.

### 결론 2 — CMFD inner는 이미 outer당 API 호출 1회다. 남은 dispatch 비용은 그래프 **내부**에 있다

측정치 "약 301,334 kernel calls / run"과 outer 21,271회를 나누면 **outer당 14.2회**다. CMFD inner 전체가 그래프 1회 launch이고(`launch_outer`, `src/CudaBICGBackend.cu:2691`), 나머지가 nodal 6개 커널 + flux memcpy + 산발적 xsrecon이면 정확히 이 자릿수가 나온다.

> **그러므로 Rev.4 §11의 "graph/kernel dispatch + device synchronization ≥ solver wall의 10%" 게이트를 `cudaGraphLaunch`/`cudaStreamSynchronize`의 CUDA **API** 지속시간으로 계산하면 반드시 오답이 나온다.** 그 게이트는 **GPU 타임라인의 커널 간 간극(gap)** 으로 계산해야 한다. Stage 0의 측량 규약은 이 점을 강제한다(§2.6).

### 결론 3 — 그 게이트는 **이미 소스 주석 안의 실험으로 통과되어 있다**

`src/CudaBICGBackend.cu:168–174`의 `RASBERY_GPU_ITER_BATCH` K=8 실험:

```text
K = 8 → 432,704 no-op iterations, h5 bit-identical, +8.7 s of 84.9 s
```

no-op iteration은 halt가 올라간 뒤 모든 커널이 첫 명령에서 return하는 것이므로 **순수 dispatch 비용**이다. 융합 후 iteration당 노드 수는 22개(`src/CudaBICGBackend.cu:196–201`)이므로

```text
c_dispatch = 8.7 s / (432,704 × 22) = 0.914 us / graph node
```

교차검증: no-op iteration 432,704개 ÷ 4(= 1+nmax) = **108,176 sweeps**. outer 21,271회로 나누면 **outer당 5.09 sweeps** — 이 값은 `_ncmfd` 예산과 정합한다. 총 노드 실행 수는

```text
108,176 sweeps × 95 nodes/sweep = 10.28 M node executions
10.28 M × 0.914 us = 9.4 s
```

즉 **drive 55.5 s 중 약 9.4 s(17.0%)가 그래프 내부 노드 dispatch**이고, solve 91.3 s 대비 **10.3%** 다. Rev.4의 첫 번째 조건(≥10%)을 아슬아슬하게 통과한다.

### 결론 4 — 그런데 두 번째 조건(제거 가능 ≥ end-to-end 5%)은 **grid.sync() 비용 하나에 전부 걸려 있다**

persistent kernel은 노드 경계를 `grid.sync()`로 대체한다. 9.4 s가 제거 가능하려면 배리어가 dispatch보다 **충분히 싸야** 한다.

**그리고 배리어를 아낄 여지는 거의 없다.** §3.4.3에서 노드 경계 21개를 전수 조사한 결과, **21개 중 21개가 grid 전역 배리어를 요구한다**. 이유는 "같은 인덱스 도메인이면 블록-지역이다"라는 직관이 이 코드에서 성립하지 않기 때문이다 — 예컨대 `update_solution`은 `vector_blocks = 67` 블록이 256개씩 나눠 갖는 파티션으로 `r`을 쓰고, 바로 다음 `reduce_dot_stage1`은 **17 블록이 995개씩 나눠 갖는 다른 파티션**으로 같은 `r`을 읽는다(`reduce_blocks_for`, `CudaBICGBackend.cu:227`). 원소 `i`의 생산자 블록과 소비자 블록이 다르므로 전역 배리어가 필요하다. 파티션을 맞추면 배리어를 없앨 수 있지만 그것은 Rev.4 frozen list의 `per-block element ranges`·`stage-2 fold 순서` 위반이라 **금지**다.

```text
Stage1_removable = N_node × c_dispatch  −  N_barrier × c_barrier
                 = 10.28M × 0.914us  −  (10.28M × 0.955) × c_barrier
```

| c_barrier 가정 | 제거량 | end-to-end 94.6 s 대비 | Rev.4 5% 게이트 |
|---|---:|---:|---|
| 0.2 us | 7.4 s | 7.8 % | **통과** |
| 0.3 us | 6.4 s | 6.8 % | **통과** |
| 0.45 us | 4.9 s | 5.2 % | **통과 (여유 없음)** |
| 0.6 us | 3.5 s | 3.7 % | **탈락** |
| 0.914 us (= dispatch와 동급) | 0.0 s | 0 % | **탈락** |
| 1.5 us | **−5.3 s (손해)** | — | **탈락** |

**Stage 1b 착수 임계: `c_barrier(sm_120, 34~67 blocks) < 0.45 us`.**

> **Phase 5 Stage 1b(persistent)의 GO/NO-GO는 `c_barrier(sm_120, G blocks)` 단 하나의 숫자로 결정된다.** 그리고 이 숫자는 30줄짜리 마이크로벤치로 반나절이면 측정된다. **구현을 한 줄이라도 쓰기 전에 이 측정이 선행되어야 한다**(Stage 0-E, §2.6).

**다만 배리어 산술이 포착하지 못하는 상방 요인이 하나 있다.** 그래프 노드 경계에서는 레지스터가 전부 소실되어 다음 커널이 `dinv`/`neighbors`/`colors`/`cc` 를 L2·DRAM에서 다시 읽는다. persistent 커널은 이 값들을 sweep 내내 레지스터·shared에 붙들어 둘 수 있다. FP32 실험이 밝힌 대로 이 커널들은 **0.13 FLOP/B의 대역폭 바운드**이므로, 재로드 제거는 dispatch 절감보다 클 수도 있다. 반대로 레지스터 압력이 올라가면 `blocksPerSM`이 떨어져 §2.4의 eligibility가 나빠진다. **양방향이고 마이크로벤치로는 측정할 수 없다.** 따라서 §3.4.13에 **2 인일 프로토타입 스파이크**를 두어, 14 인일을 투입하기 전에 iteration 1개 분량의 persistent 판을 실제로 만들어 A/B 한다.

이 때문에 본 계획은 Stage 1을 세 갈래로 분해하고, **persistent(1b)를 가장 마지막·가장 조건부**로 배치한다.

| 갈래 | 내용 | 위험 | 기대 | 우선순위 |
|---|---|---|---|---|
| **1a** | active-slot 압축 (persistent 없이) | 낮음 | 배치 모드에서 큼, 단일덱 0 | **1** |
| **1c** | PDL(Programmatic Dependent Launch, sm_90+) | 낮음 | dispatch 간극의 일부 | **2** |
| **1b** | persistent/cooperative 커널 | 높음 | c_barrier 의존 | **3, 조건부** |

---

## 1. 코드를 읽어 확정한 사실 (설계의 기반)

모든 설계 판단은 아래 사실 위에 선다. 각 항목에 앵커를 붙였으므로 배치 담당은 검증 없이 인용하지 말 것.

### 1.1 커널·그리드 기하

| 항목 | 값 | 앵커 |
|---|---|---|
| `nxyz` | 8,451 | 입력 (kngr_238) |
| `ng`, `n = ngxyz` | 2, 16,902 | — |
| `block_size` 기본 | **256** | `CudaBICGBackend.cu:31` `kDefaultBlockSize` |
| 허용 block_size | 64/128/192/256 | `CudaBICGBackend.cu:1898` |
| `node_blocks()` | `ceil(8451/256)` = **34** | `CudaBICGBackend.cu:2205` |
| `vector_blocks()` | `ceil(16902/256)` = **67** | `CudaBICGBackend.cu:2206` |
| `reduce_blocks_for(n)` | `ceil(16902/1024)` = **17** (cap 256) | `CudaBICGBackend.cu:227–233` |
| `kReduceThreads` | **256** (shared 배열 크기 고정) | `CudaBICGBackend.cu:224` |
| `scalar_grid()` | `dim3(1, slots)`, **1 thread** | `CudaBICGBackend.cu:2253` |
| 배치축 | **`gridDim.y = slots`**, 절대 x와 접지 않음 | `CudaBICGBackend.cu:2246–2255` |
| `slots` | 단일 실행 1, 배치 M | `CudaBICGBackend.cu:1774` |

**물리 grid 블록 수**

| 커널류 | slots=1 (단일덱) | slots=64 (M64) |
|---|---:|---:|
| node (matvec, sweep, jacobi) | 34 | 2,176 |
| vector (update_solution, negative scan) | 67 | 4,288 |
| reduce stage1 | 17 | 1,088 |
| scalar (1 thread/block) | 1 | 64 |

> **단일덱에서 CMFD 커널의 최대 grid는 67 블록이다. 188 SM GPU에서 SM의 2/3가 아예 놀고 있고, 일하는 SM도 블록 하나씩만 잡는다.** 단일덱 drive 55.5 s는 연산 시간이 아니라 dispatch + 메모리 지연 시간이라는 뜻이다. (§0 결론 3의 9.4 s dispatch가 이와 정합한다.)

### 1.2 현재 dispatch 구조

```text
outer (Driver.h:1367 SolveLoop)
 ├ updpsi          host      (CMFD.cpp:246)
 ├ setls           host      (CMFD.cpp:205 + BICGCMFD.cpp:259 updls)   ← 12.1 s
 ├ drive           BICGCMFD.cpp:463                                     ← 55.5 s
 │   └ ×5.09 sweep
 │       ├ src build          host  (BICGCMFD.cpp:519)
 │       ├ launch_outer       GPU graph, 95 nodes (CudaBICGBackend.cu:2629)
 │       │    ├ prologue 5 nodes
 │       │    ├ iteration ×4, 22 nodes each  (CudaBICGBackend.cu:2485)
 │       │    └ finalize + status D2H
 │       ├ synchronizeCudaFlux  → cudaStreamSynchronize (드레인)
 │       ├ wiel              host  (BICGCMFD.cpp:129)
 │       ├ updls + facilu    host  (BICGCMFD.cpp:252)
 │       └ negative census   host
 ├ updjnet         host      (CMFD.h:240)                               ← 2.5 s
 ├ nodal reset+drive  GPU (6 kernels)                                   ← 6.1 s
 ├ ApplyRodCusping host
 ├ upddhat         host      (CMFD.cpp:126)                             ← 5.8 s
 └ 수렴/Xe/보론/T-H 제어  host
```

배치 모드에서만 `enqueue_sweeps`(`CudaBICGBackend.cu:2708`)가 위 sweep 루프 전체를 GPU에 올린다: operator assembly 1회 + `unroll`회 sweep(각 sweep = 7 노드 + `enqueue_outer` 95 노드). 호스트 개입은 `unroll` 소진 또는 negative-flux 재시도 때만.

### 1.3 이미 존재하는 자산 (재발명 금지 목록)

| 자산 | 위치 | Phase 5에서의 역할 |
|---|---|---|
| device sweep 루프 | `CudaBICGBackend.cu:2708` `enqueue_sweeps` | **Stage 1의 persistent 범위는 정확히 이 본문** |
| device operator assembly | `CmfdAssemblyKernel.h` + `cmfd_assemble_operator_2g` | Stage 2의 setls 항목이 이미 절반 완성 |
| per-slot halt / active mask | `device_halt`, `device_active`, `sweep_halt` | Stage 1의 per-slot exit 기반 |
| FP32 혼합정밀 쌍둥이 커널 | `CudaBICGBackend.cu:1048–1520` | Stage 1은 템플릿 2본으로 계승 |
| 결정론적 reduction 계약 | `CudaBICGBackend.cu:235–250` | bit 보존의 근거 문서 |
| form probe 방법론 | `test/cmfd_form_probe.cpp`, `RASBERY_CMFD_DUMP` | **Stage 2 upddhat/updjnet 이식의 필수 선행** |
| device Xe 융합 커널 | `XsReconKernel.h`, `kernelXsRecon` | Stage 3-B의 출발점 (단, Evaluate-only 진입점 없음) |
| atomicMax-on-bits 패턴 | `kernelXsRecon(..., max_bits, ...)` | Stage 2 `_dhat_ratio_max` 이식 선례 |
| arrival gap EWMA | `CudaBICGBackend.cu:3488` | Stage 0/1의 폭 관련 지표 |

---

## 2. Stage 0 — Nsight 측량 프로토콜

**목적**: Rev.4 §11 진입 게이트의 두 숫자를 **재현 가능하게** 산출하고, Stage 1b의 GO/NO-GO를 결정하는 `c_barrier`를 실측한다.
**담당**: Sonnet, 238 GPU0 단독. **공수 2 인일.**
**전제**: Rev.4 §2.1 exact-only 검사 통과, §3.6 벤치마크 규약(warm-up 1 + hot 3, balanced order) 준수.

### 2.0 도구 가용성 프로브 (가장 먼저)

```bash
# 238 서버에서
which nsys ncu 2>/dev/null
nsys --version 2>/dev/null
ncu --version 2>/dev/null
ls /usr/local/cuda/bin/ | grep -E 'nsys|ncu'
ls /usr/local/cuda*/extras/CUPTI/lib64/libcupti.so* 2>/dev/null
# 비특권 프로파일링 허용 여부 (ncu는 이게 없으면 ERR_NVGPUCTRPERM으로 실패)
cat /proc/driver/nvidia/params 2>/dev/null | grep -i RmProfilingAdminOnly
```

**분기**:
- `nsys` 있음 → §2.3 (주 경로)
- `nsys` 없음, `libcupti.so` 있음 → §2.5-A (CUPTI 주입)
- 둘 다 없음 → §2.5-B (cudaEvent + 환경변수 A/B). **§2.5-B만으로도 게이트 판정은 가능하다** — §0 결론 3의 K=8 실험이 이미 그 방법이었기 때문이다.
- `ncu`가 `ERR_NVGPUCTRPERM`으로 실패 → 점유율은 §2.4-B(런타임 API 프로브)로 대체. **`RmProfilingAdminOnly` 변경은 서버 정책 변경이므로 임의로 하지 말 것.**

### 2.1 측정 대상 arm 정의

| arm | 구성 | 목적 |
|---|---|---|
| **S0-1** | 단일덱, 현재 기본 (아레나 없음) | 현행 기준선 재확인 (94.6 s 재현) |
| **S0-2** | 단일덱, `--batch-mode 1` + `RASBERY_GPU_CMFD_SWEEP=1` | **결론 1의 무료 실험** |
| **S0-3** | M64 배치, 현재 기본 | 배치 기준선 (214.8 c/h 재현) |
| **S0-4** | M64 배치, `RASBERY_GPU_GRAPH=0` | 그래프 절약분 역산 (상한 검증) |
| **S0-5** | 단일덱, `RASBERY_GPU_ITER_BATCH=8` | `c_dispatch` 재측정 (결론 3 재현) |

### 2.2 Stage 0-A — 코드 없는 무료 실험 (**최우선**)

```bash
# 기준
RASBERY_GPU_CMFD_SWEEP=0 ./RASBERY --rasi kngr_238.json ...

# device-resident sweep + device assembly 활성 (아레나 폭 1)
RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_ASSEMBLY=1 \
  ./RASBERY --batch-mode 1 --rasi kngr_238.json ...
```

**확인 항목**
1. `[RASBERY][CUDA][BATCH] arena: ... slots=1` 로그가 뜨는가.
2. `[RASBERY][SPTELEM]`의 `phase_wall.setls`가 12.1 s → ~0 s로 떨어지는가.
3. `cmfd_assembly_gpu_calls > 0`, `cmfd_diag_h2d_elided_bytes > 0`인가.
4. **HDF5 500/500 dataset byte-identical** (궤적 보존 변경이므로 bit-golden 필수).
5. `stream_sync_calls_during_iteration` 감소량.

**판정**: 4번이 통과하고 wall이 유의하게 줄면 **Phase 5보다 먼저 채택**한다. 통과하지 못하면 그 자체가 별건 결함이며 Phase 5보다 먼저 규명한다(단일덱과 배치가 다른 답을 낸다는 뜻이므로).

**예상**: setls 12.1 s 중 상당분 + sweep당 호스트 왕복 108,176회 제거. 낙관적으로 단일덱 94.6 → 78~84 s. 단, 폭 1 아레나는 rendezvous/linger 경로를 타므로 `RASBERY_BATCH_WAIT_US=0`로 linger를 꺼서 단일 참가자가 대기하지 않게 할 것 (`CudaBICGBackend.cu:3228`).

### 2.3 Stage 0-B — nsys 측량 (주 경로)

```bash
# 캡처. --sample=none 으로 CPU 샘플링 오버헤드를 빼고, GPU 타임라인만 정밀하게.
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node \
  --gpu-metrics-devices=cuda-visible \
  --gpu-metrics-frequency=10000 \
  --force-overwrite=true \
  -o ph5_s0b_single \
  ./RASBERY --rasi kngr_238.json ...
```

> **`--cuda-graph-trace=node`가 이 측정의 핵심이다.** 기본값(`graph`)은 그래프 실행 전체를 한 구간으로 접어버려 §0 결론 2가 말한 "그래프 **내부** 간극"이 보이지 않는다. 이 플래그 없이 얻은 데이터는 무효 처리한다.

```bash
# 통계 추출
nsys stats --report cuda_api_sum      -f csv -o s0b ph5_s0b_single.nsys-rep
nsys stats --report cuda_gpu_kern_sum -f csv -o s0b ph5_s0b_single.nsys-rep
nsys stats --report cuda_gpu_trace    -f csv -o s0b ph5_s0b_single.nsys-rep
# 원시 타임라인이 필요하면 SQLite로
nsys export --type sqlite -o ph5_s0b_single.sqlite ph5_s0b_single.nsys-rep
```

**추출할 지표**

| 기호 | 정의 | 출처 |
|---|---|---|
| `N_node` | CMFD 스트림의 커널 실행 총수 | `cuda_gpu_kern_sum` Instances 합 |
| `T_kern` | 커널 지속시간 **구간 합집합** (중첩 제거) | `cuda_gpu_trace` start/end |
| `T_solve` | solver wall (SPTELEM `wall - io_wall`) | RASBERY 자체 receipt |
| `G_intra` | **같은 sweep(graph launch) 안**의 연속 커널 간 간극 합 | `cuda_gpu_trace` 정렬 후 계산 |
| `G_inter` | sweep 경계를 넘는 간극 합 | 동상 |
| `A_launch` | `cudaGraphLaunch`+`cudaLaunchKernel`+`cudaMemcpyAsync` API 시간 합 | `cuda_api_sum` |
| `A_sync` | `cudaStreamSynchronize` API 시간 합 | `cuda_api_sum` |
| `A_sync_idle` | `A_sync` 중 GPU가 놀고 있던 부분 | sync 구간 − 그 구간의 `T_kern` |
| `H_dur` | 커널별 지속시간 히스토그램 (p50/p90/p99) | `cuda_gpu_trace` |

`G_intra` 계산 SQL 스케치 (sqlite export 후):

```sql
-- 같은 graphNodeId 계보(=같은 graphExec 실행)에 속한 인접 커널 간 간극
WITH k AS (
  SELECT start, end, graphNodeId,
         LAG(end) OVER (ORDER BY start) AS prev_end,
         LAG(graphNodeId) OVER (ORDER BY start) AS prev_gid
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  WHERE streamId = (SELECT streamId FROM ... /* CMFD 스트림 */)
)
SELECT SUM(MAX(start - prev_end, 0)) AS g_intra
FROM k WHERE prev_gid IS NOT NULL AND graphNodeId IS NOT NULL;
```

### 2.4 Stage 0-C — 점유율과 eligibility

**2.4-A (ncu 사용 가능 시)**

```bash
ncu --target-processes all \
    --kernel-name regex:"matvec_two_group|colored_block_sweep|reduce_dot_stage1|update_solution|prepare_p_jacobi" \
    --launch-skip 20000 --launch-count 60 \
    --metrics \
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,\
launch__occupancy_limit_blocks,\
sm__maximum_warps_per_active_cycle_pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
    --csv --log-file ph5_s0c.csv \
    ./RASBERY --rasi kngr_238.json ...
```

> ncu는 커널을 **직렬화**한다. 이 실행의 wall은 성능 지표로 절대 쓰지 말 것. 여기서 얻는 것은 **레지스터/공유메모리/점유율 한계**뿐이다.

**2.4-B (ncu 불가 시 — 런타임 API 프로브, 30줄)**

Stage 1 구현 전에는 persistent 커널이 없으므로, **현존 최대 레지스터 커널**(`matvec_two_group` 또는 `begin_outer_fused`)로 상한을 대신 잰다.

```cpp
// tools/probe_occupancy.cu  (독립 빌드, 프로덕션 코드 무변경)
#include <cstdio>
#include <cuda_runtime.h>
__global__ void dummy256() { }        // 실제로는 대상 커널 심볼을 링크해 질의
int main() {
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dummy256, 256, 0);
    std::printf("{\"sm\":%d,\"maxThreadsPerSM\":%d,\"regsPerSM\":%d,"
                "\"blocksPerSM@256\":%d,\"residentBlocks\":%d,"
                "\"cooperativeLaunch\":%d}\n",
                p.multiProcessorCount, p.maxThreadsPerMultiProcessor,
                p.regsPerMultiprocessor, blocks,
                blocks * p.multiProcessorCount, p.cooperativeLaunch);
}
```

**Rev.4 §11 eligibility 식의 적용**

```text
required_grid_blocks  ≤  resident_blocks = blocksPerSM × multiProcessorCount
```

| 모드 | required (물리) | resident 추정 (188 SM × 4) | 판정 |
|---|---:|---:|---|
| 단일덱 slots=1 | 67 | ~752 | **통과** |
| 배치 slots=64 | 4,288 | ~752 | **탈락** |
| 배치 slots=64, 활성 22로 압축 | 1,474 | ~752 | **탈락** |

> **배치 모드는 압축을 해도 물리 grid를 resident 아래로 내릴 수 없다.** 따라서 persistent 설계는 **논리 블록 가상화**(§3.4.2)를 반드시 포함해야 하며, Rev.4 §11의 eligibility 식은 "물리 grid ≤ resident, **논리 블록 수는 불변**"으로 읽어야 한다. 이는 계약 문언의 해석 확장이므로 **§7의 게이트 표에 별도 항목으로 명시하고 Rev.5에 반영을 건의한다** (임의 재해석으로 통과 처리하지 않는다).

### 2.5 Stage 0-D — nsys 부재 시 대체 계측

**2.5-A CUPTI 주입 라이브러리** (공수 1 인일)
`CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` + `CUPTI_ACTIVITY_KIND_RUNTIME` 구독, `LD_PRELOAD`로 주입, 종료 시 JSON 덤프. RASBERY 소스 무변경. 얻는 것은 §2.3과 동일한 `(start,end,streamId,graphNodeId)` 튜플.

**2.5-B cudaEvent + 환경변수 A/B** (공수 0.5 인일, **소스 무변경**)
이미 존재하는 노브만으로 §0 결론 3을 재현·정밀화한다.

```bash
for K in 4 6 8 12; do
  RASBERY_GPU_ITER_BATCH=$K ./RASBERY --rasi kngr_238.json ... 
done
```

`wall(K)` 대 `no_op_nodes(K) = (K-4) × 22 × sweeps` 를 선형회귀하면 기울기가 `c_dispatch`다. 절편은 실제 연산 시간. **이 방법은 dispatch 비용을 직접, 편향 없이 준다** — no-op 노드는 정의상 연산이 0이기 때문이다. `overrun_iterations` 카운터가 no-op 수를 이미 세고 있으므로(`CudaBICGBackend.h:96`) 회귀의 x축은 추정이 아니라 실측값이다.

동일 방식으로 `RASBERY_GPU_GRAPH=0/1` A/B는 **API launch 비용**을, `RASBERY_GPU_CMFD_SCALAR_FUSION=0/1` A/B는 노드 2개 제거의 한계효과를 준다.

### 2.6 Stage 0-E — `c_barrier` 마이크로벤치 (**Stage 1b의 유일한 결정 인자**)

```cpp
// tools/probe_gridsync.cu — 독립 빌드. 프로덕션 코드 무변경.
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

__global__ void barrier_only(int iters, double* sink) {
    cg::grid_group grid = cg::this_grid();
    double acc = 0.0;
    for (int i = 0; i < iters; ++i) { acc += 1.0; grid.sync(); }
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = acc;
}
int main() {
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    int perSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, barrier_only, 256, 0);
    const int resident = perSM * p.multiProcessorCount;
    double* sink; cudaMalloc(&sink, sizeof(double));
    // 실제 코드가 쓰는 grid 형상 전부를 훑는다.
    const int shapes[] = {17, 34, 67, 128, 256, 512, resident/2, resident};
    for (int g : shapes) {
        if (g < 1 || g > resident) continue;
        const int iters = 100000;
        void* args[] = {(void*)&iters, (void*)&sink};
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaLaunchCooperativeKernel((void*)barrier_only, g, 256, args, 0, 0); // warm
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        cudaLaunchCooperativeKernel((void*)barrier_only, g, 256, args, 0, 0);
        cudaEventRecord(b); cudaDeviceSynchronize();
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        std::printf("{\"blocks\":%d,\"c_barrier_us\":%.4f}\n", g, ms*1000.0/iters);
    }
}
```

빌드: `nvcc -arch=sm_120 -rdc=true tools/probe_gridsync.cu -o probe_gridsync`
(`-rdc=true`는 협조적 그룹 grid.sync()의 요구사항이며, 본 프로젝트는 이미 `CUDA_SEPARABLE_COMPILATION ON`이다 — `CMakeLists.txt:89`.)

**동시에 측정할 대조군** (같은 실행에서): 빈 커널 `<<<g,256>>>`를 그래프에 100,000노드로 넣고 replay하여 `c_dispatch`를 같은 장비·같은 형상에서 재측정. 두 숫자를 **동일 조건에서** 얻어야 §0 결론 4의 표가 의미를 갖는다.

### 2.7 게이트 계산식 (수치 GO/NO-GO)

```text
[게이트 1] Rev.4 §11 첫 조건
    D = (G_intra + A_launch + A_sync_idle) / T_solve   ≥  0.10

[게이트 2] Rev.4 §11 둘째 조건 — Stage 1b에만 적용
    R_1b = N_node × c_dispatch − N_barrier × c_barrier
    N_barrier = 0.955 × N_node         (§3.4.3 전수 조사로 확정. 낮출 수 없음)
    R_1b / T_end_to_end  ≥  0.05

[게이트 3] eligibility (§2.4)
    물리 grid ≤ resident_blocks,  논리 블록 수 불변   (Rev.4 문언 확장 — §7 명시)

[게이트 4] Stage 1a 단독 (persistent 무관)
    R_1a = (1 − mean_active_width / arena_width) × s × T_solve
    s = slot-비례 비용 분율, Stage 0-F(폭 스윕 48/64/80, 워커 고정)로 측정

[게이트 5] 독점 위험 (Rev.4 §11 마지막 문단)
    other_stream_wait(after) ≤ other_stream_wait(before) × 1.05
    AND end_to_end_throughput(after) ≥ before
```

**Stage 0-F — 폭 스윕** (`s` 측정 전용, 코드 무변경): 잡 수와 워커 수를 고정하고 `--batch-mode`만 48/64/80으로 바꿔 3회씩. Rev.4 §12.2의 "width 효과와 worker 효과 분해" 규약을 그대로 따른다.

### 2.8 Stage 0 산출물

`docs/PHASE5_STAGE0_MEASUREMENT_<date>_KO.md` — 위 5개 arm의 원시 CSV, 게이트 1~5의 계산 결과, `c_barrier`/`c_dispatch` 표, 그리고 **Stage 1b GO/NO-GO 단일 판정**.

---

## 3. Stage 1 — Persistent / Cooperative CMFD 커널

### 3.1 범위 결정

persistent 커널의 범위는 **`enqueue_sweeps`의 본문 전체**(`CudaBICGBackend.cu:2708–2736`)로 한다.

```text
cmfd_assemble_operator_2g            ×1
for sweep in 0..unroll:
    cmfd_sweep_begin
    cmfd_src_build
    enqueue_outer   (= 5 프롤로그 + 4×22 iteration + finalize + status)
    cmfd_wiel_terms
    cmfd_wiel_finalize
    cmfd_updls
    cmfd_negative_scan
    cmfd_sweep_end
```

**이 범위를 고른 이유**
1. 이미 device-only 커널만으로 구성되어 있다 — 새 물리를 device에 올리지 않는다.
2. `sweep_halt`가 호스트 루프의 break/retry 결정을 이미 device 워드로 표현하고 있다.
3. sweep당 노드 수가 최대(7 + 95 = 102)이므로 dispatch 제거 효과가 최대다.
4. 범위를 `enqueue_outer`로 좁히면 sweep 경계마다 커널을 재진입해야 하므로 배리어만 늘고 dispatch는 덜 줄어든다.

**범위 밖**: nodal, xsrecon, upddhat/updjnet(Stage 2), outer 제어(Stage 3).

### 3.2 Stage 1a — active-slot 압축 (persistent 없이, **먼저**)

#### 문제

모든 커널이 `gridDim.y = slots`로 발사된다(`node_grid()` 등). 활성 시스템이 평균 22/64여도 **42개 슬롯 몫의 블록이 발사되어 첫 명령의 `HALT_GUARD`에서 리타이어한다**. 폭 96 실험의 BASE −62%가 이 구조의 직접 증거다 (`CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md` §4).

#### 설계

1. **슬롯 간접 배열 도입.** `device_slot_map[j] = m` (j ∈ [0, n_active), m = 실제 아레나 슬롯). 모든 커널의 첫 줄
   ```cuda
   const int m = static_cast<int>(blockIdx.y);          // 현재
   const int m = slot_map[blockIdx.y];                  // 변경 후
   ```
   `slot_map`이 항등(`slot_map[j]==j`, n_active==slots)일 때 현재와 **완전히 동일한 실행**이 되도록 구성한다. 이것이 feature-off 바이트 불변의 근거다.

2. **grid 버킷화.** `gridDim.y`는 캡처된 그래프에 고정되므로 매 발사마다 바꿀 수 없다. 재인스턴스화는 비싸다(`graph_reinstantiations` 카운터가 정상 실행에서 0인 이유). 따라서 **버킷 집합** `{1, 2, 4, 8, 16, 24, 32, 48, 64}` 각각에 대해 `graph_exec`를 캐시하고, `bucket = 최소 버킷 ≥ n_active`를 고른다. 버킷은 9개이므로 재인스턴스화는 실행당 최대 9회(×정밀도 2 ×nmax 변화)로 유계다.
   - VRAM: `graph_exec` 9개는 노드 메타데이터뿐이며 페이로드 버퍼는 공유한다. 실측으로 확인할 것.
   - 대안(측정 후 선택): `cudaGraphExecKernelNodeSetParams`로 인스턴스화된 그래프의 `gridDim.y`만 갱신. 노드 95개 × API 호출 = outer당 95회 → §0 결론 2에 따르면 API 비용은 이미 작으므로 실현 가능하나, 버킷 방식이 더 단순하고 예측 가능하다. **기본은 버킷, 대안은 A/B로만.**

3. **압축 시점.** `issueSweepUploads`/`issueUploads`가 이미 `host_active`를 만들고 있다(`CudaBICGBackend.cu:2285`, `2787`). 여기서 동시에 `host_slot_map`을 채워 같은 H2D에 실어 보낸다. **추가 전송 0회.**

#### bit 보존

- `slot_map`은 **어느 슬롯의 데이터가 어느 논리 시스템인지**를 바꾸지 않는다. 슬롯 m의 벡터 기점은 여전히 `m * vec_stride()`다.
- reduction의 `chunk`는 `gridDim.x`에만 의존한다(`CudaBICGBackend.cu:267` 및 그 위의 계약 주석). y축 변경은 chunk에 영향이 없다.
- 따라서 **궤적 보존 변경**이며 bit-golden(단일 500/500, 배치 708/708)이 적용된다.
- Rev.4 §11 frozen list의 `slot mapping` 항목: "논리 시스템 ↔ 데이터 기점" 매핑은 불변, "논리 시스템 ↔ blockIdx.y" 매핑만 변한다. **이 구분을 계약 테스트로 못 박는다**(§3.12).

#### 기대 이득

- **단일덱: 0** (slots=1, `slot_map`이 항등).
- **배치 M64**: 게이트 4 식. `mean_active_width = 22`, `s`(slot-비례 분율)를 Stage 0-F로 측정. 폭 96 실험의 −62%는 `s`가 매우 크다는 것을 시사하나(폭 1.5배에 wall 2.63배는 초선형이므로 단순 선형 모형은 쓰지 않는다), **보수적으로 `s = 0.4`를 가정하면 배치 wall −27%**, `s = 0.6`이면 −39%.
- **부수 효과**: 폭 확대(Rev.4 Phase 6)가 다시 살아난다. 압축이 있으면 빈 슬롯이 비용을 물지 않으므로 폭 96/128의 기각 근거가 사라진다. **Phase 6 재개는 Stage 1a에 종속된다.**

#### 공수: 5 인일 (구현 2, 계약 테스트 1, 검증 2)

### 3.3 Stage 1c — Programmatic Dependent Launch (저위험 대안)

sm_90 이상에서 CUDA는 **PDL**을 제공한다: `cudaLaunchAttributeProgrammaticStreamSerialization`을 붙인 후속 커널이 선행 커널의 완전 종료를 기다리지 않고 **프리앰블**(주소 계산, 상수 로드)을 먼저 수행하고, 선행 커널의 쓰기를 읽기 직전에 `cudaGridDependencySynchronize()`로 동기한다. 선행 커널은 `cudaTriggerProgrammaticLaunchCompletion()`으로 조기 트리거한다.

- **적용 대상**: 노드 간 간극 `G_intra`가 큰 인접 쌍. 특히 `prepare_p_jacobi → colored_block_sweep ×4 → matvec` 사슬처럼 같은 인덱스 도메인을 도는 구간.
- **적용 금지**: reduction stage1 → stage2 (stage2가 stage1의 전 블록 결과를 읽으므로 프리앰블에서 할 일이 없다), color sweep 간 (이웃 읽기).
- **산술 무변경.** 각 커널의 연산·순서·피연산자가 그대로이므로 **궤적 보존**이고 bit-golden이 적용된다.
- **그래프 호환.** 커널 노드 속성이므로 캡처된다.
- **위험**: 낮음. 실패해도 속성을 떼면 원상복구.
- **238 확인 필요**: sm_120에서 PDL 속성이 그래프 노드에 적용될 때의 동작. Stage 0에서 `probe_pdl.cu` 10줄로 확인.
- **공수 3 인일.** **기대**: `G_intra`의 20~40% (Stage 0-B의 히스토그램이 정한다).

> Stage 1c는 **Stage 1b가 NO-GO여도 살아남는다**. 이것이 1b보다 먼저 배치된 이유다.

### 3.4 Stage 1b — Persistent / Cooperative 커널 설계

**전제**: Stage 0-E의 `c_barrier` 측정이 게이트 2를 통과했을 때만 착수한다.

#### 3.4.1 배리어 기법 — cooperative groups `grid.sync()` **채택**

| | `cg::this_grid().sync()` | multi-block lock-free 배리어 |
|---|---|---|
| 동시 상주 보장 | **드라이버가 보장** (`cudaLaunchCooperativeKernel`이 grid > resident면 실패 반환) | **없음** — grid > resident면 **교착**, 그것도 비결정적으로 |
| grid 크기 상한 | `resident_blocks` (질의 가능) | 실질적으로 동일한 상한을 **스스로** 지켜야 함 |
| 메모리 일관성 | 배리어가 release/acquire 의미를 포함 | `__threadfence()` + `volatile`/`cuda::atomic`을 직접 옳게 배치해야 함 |
| 그래프 캡처 | CUDA 12+ `cudaLaunchAttributeCooperative`로 노드화 가능 (**238에서 확인 필요**) | 일반 커널이므로 자명 |
| 디버깅 | `compute-sanitizer --tool synccheck` 지원 | 도구 지원 사실상 없음 |
| CUDA 13 지원 | 정식 API | 미정의 동작에 의존 |

**결정: `cg::this_grid().sync()`.**

**근거 (sm_120 + 배치 아레나 맥락)**
1. lock-free 배리어는 grid 크기 상한을 **없애주지 않는다**. 동시 상주가 아니면 교착하므로 결국 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`로 같은 상한을 지켜야 한다. **즉 유일한 장점이 존재하지 않는다.**
2. 64-슬롯 아레나는 grid가 커지는 방향이므로(§2.4의 표) 교착 위험이 가장 큰 형태다. 드라이버가 발사 자체를 거부해 주는 쪽이 안전하다 — 실패가 **런타임 행(hang)이 아니라 에러 코드**로 나온다는 것이 공유 서버에서는 결정적 차이다(§6 위험 R-2).
3. 본 프로젝트는 이미 `-rdc=true`(`CUDA_SEPARABLE_COMPILATION ON`)로 빌드하므로 추가 빌드 부담이 없다.
4. `synccheck`로 배리어 발산(divergent barrier)을 정적·동적으로 잡을 수 있다. §3.4.5의 최대 위험이 바로 배리어 발산이므로 도구 지원이 필수다.

#### 3.4.2 논리 블록 가상화 (**bit 보존의 핵심 장치**)

물리 grid는 `resident_blocks` 이하여야 하지만(§2.4), 논리 블록 수는 bit 보존 대상이다. 두 요구를 동시에 만족시키는 유일한 방법:

```cuda
// 논리 블록 L개를 물리 블록 G개가 grid-stride로 나눠 갖는다.
// L은 커널 인자로 명시 전달한다 — gridDim.x에서 유도하면 안 된다.
for (int lb = blockIdx.x; lb < L; lb += gridDim.x) {
    // 여기 본문은 기존 커널의 본문과 문자 그대로 동일하되,
    // blockIdx.x 대신 lb 를 쓴다.
}
```

reduction에 적용할 때의 **필수 조건**:

```cuda
// 현재 (CudaBICGBackend.cu:267)
const int chunk = (n + gridDim.x - 1) / gridDim.x;
const int begin = blockIdx.x * chunk;

// 가상화 후 — chunk 는 L 에서 유도한다
const int chunk = (n + L - 1) / L;      // L = reduce_blocks_for(n) = 17, 고정
const int begin = lb * chunk;
```

**bit 보존 논증**
- `chunk`가 `L`의 순함수이고 `L`은 `n`의 순함수(`reduce_blocks_for`)이므로, 물리 grid를 어떻게 잡든 **모든 논리 블록의 `[begin, end)` 구간이 불변**이다.
- 스레드 traversal `for (i = begin + threadIdx.x; i < end; i += blockDim.x)`는 `blockDim.x`가 불변(=256)이면 불변이다. → **persistent 커널의 blockDim은 반드시 256으로 고정**한다. 이것은 선택이 아니라 계약이다(`kReduceThreads == 256`이고 shared 배열 크기가 그에 고정되어 있다).
- shared 축약 트리 `for (stride = 128; stride > 0; stride >>= 1)`도 blockDim 불변이면 불변.
- stage-2 fold `for (i = 0; i < L; ++i) sum += pm[i]`는 `L`이 불변이므로 불변. 물리 블록이 논리 블록을 어떤 순서로 처리하든 `pm[]`의 **내용**은 논리 블록 인덱스에 결정론적으로 대응하고, fold는 인덱스 순서로 돌므로 **부동소수 결합 순서가 보존된다**.
- 하나의 물리 블록이 여러 논리 블록을 순차 처리할 때 shared 배열을 재사용하므로, 논리 블록 반복 사이에 `__syncthreads()`가 필요하다. 이는 블록 내 배리어이므로 grid 배리어와 무관하고 산술에 영향이 없다.

> **가상화 후의 논리 블록 수 L**: node = 34, vector = 67, reduce = 17. 슬롯 축까지 접으면 논리 워크아이템 = `L × n_active`. 물리 grid = `min(resident, L × n_active)`. 압축(§3.2)이 여기서 두 번째로 값을 한다 — 가상화 루프의 상한을 `L × 64`가 아니라 `L × 22`로 줄인다.

#### 3.4.3 배리어가 **실제로** 필요한 지점 — 전수 조사 (게이트 2의 `N_barrier` 확정)

iteration 22 노드의 경계 21개를 전수 조사한다. 판정 기준은 **원소 `i`의 생산자 블록과 소비자 블록이 같은가** 하나뿐이다.

각 커널의 파티션을 먼저 정리한다(§1.1):

```text
node   커널 : 34 블록, 블록당 256 노드      (matvec, sweep, jacobi, assembly)
vector 커널 : 67 블록, 블록당 256 원소      (update_solution, negative_scan)
reduce 커널 : 17 블록, 블록당 995 원소      (chunk = ceil(n/17))
scalar 커널 : 1 블록,  1 스레드
```

**세 파티션이 서로 다르다. 이것이 배리어 절감을 사실상 불가능하게 만든다.**

| # | 경계 | 파티션 전이 | 배리어 | 이유 |
|---|---|---|---|---|
| 1 | dot(r0,r) stage1 → stage2 | reduce → scalar | **필요** | stage2가 17개 partial 전부를 읽음 |
| 2 | stage2 → prepare_p_jacobi | scalar → node | **필요** | rho 스칼라 브로드캐스트 |
| 3 | prepare_p_jacobi → sweep(c0) | node → node | **필요** | 이웃 노드 읽기 |
| 4–6 | sweep c0→c1→c2→c3 | node → node | **필요 ×3** | Gauss-Seidel 색 순서 = 전역 의존 |
| 7 | sweep(c3) → matvec | node → node | **필요** | 이웃 노드 읽기 |
| 8 | matvec(v 쓰기) → dot(r0,v) stage1 | **node(34) → reduce(17)** | **필요** | 파티션 불일치 |
| 9 | dot stage1 → stage2 | reduce → scalar | **필요** | |
| 10 | stage2 → update_s_jacobi | scalar → node | **필요** | 스칼라 브로드캐스트 |
| 11 | update_s_jacobi → sweep(c0) | node → node | **필요** | 이웃 |
| 12–14 | sweep c0→c1→c2→c3 | node → node | **필요 ×3** | |
| 15 | sweep(c3) → matvec | node → node | **필요** | |
| 16 | matvec(t 쓰기) → dot2 stage1 | **node(34) → reduce(17)** | **필요** | 파티션 불일치 |
| 17 | dot2 stage1 → stage2 | reduce → scalar | **필요** | |
| 18 | stage2 → update_solution | scalar → vector | **필요** | 스칼라 브로드캐스트 |
| 19 | update_solution(r 쓰기) → norm stage1 | **vector(67) → reduce(17)** | **필요** | 파티션 불일치 |
| 20 | norm stage1 → stage2/accumulate | reduce → scalar | **필요** | |
| 21 | accumulate → 다음 iteration | scalar → reduce | **필요** | halt 브로드캐스트 |

**iteration당 grid 배리어 = 21개 / 노드 22개 → 축약비 0.955.**
sweep 전체(102 노드)로도 동일 구조이므로 축약비는 실질적으로 **0.95**로 본다.

> **이것이 본 분석에서 가장 중요한 정정이다.** "같은 인덱스 도메인이면 배리어가 필요 없다"는 직관은 이 코드에서 성립하지 않는다 — 경계 8·16·19가 그 반례이고, 셋 다 융합 캠페인이 **일부러 융합하지 않은** 지점(reduction 경계)이다. persistent 커널은 dispatch를 배리어로 **거의 1:1 치환**하는 구조이며, 이득은 오직 `c_dispatch − c_barrier` 한 항에서 나온다.

**배리어를 줄일 수 있는 이론적 경로와 그 판정**

| 경로 | 절감 | 판정 |
|---|---|---|
| reduce 파티션을 vector/node에 맞춤 | 경계 8·16·19의 3개 | **금지** — frozen list `per-block element ranges` 위반 |
| 색 sweep 4개를 하나로 | 경계 4–6, 12–14의 6개 | **금지** — frozen list `color 순서` 위반 |
| `rb_sweeps`(기본 4) 축소 | 대량 | **Phase 5 밖** — 알고리즘 변경. Rev.4 §9.2(adaptive inexact inner) 소관. 혼동 금지 |
| stage2 스칼라를 배리어 도착 직후 블록0이 처리 | 0 | 노드는 줄지만 배리어는 그대로 |

**결론: `N_barrier / N_node = 0.955`는 설계 상수이며 낮출 수 없다.** §0 결론 4의 표가 확정 수치다.

#### 3.4.4 디바이스 수렴 판정과 per-slot exit + 압축

**현재 구조**는 이미 device-side다: `accumulate_iteration`/`reduce_norm_accumulate_stage2`가 `r2/r20 < eps`를 device에서 평가하고 `halt[m]`을 올린다(`CudaBICGBackend.cu:880`, `904`). persistent 커널은 이 로직을 **그대로 인라인**한다 — 산술과 순서가 변하지 않는다.

**핵심 설계 — 두 단계 exit**

```text
1단계 (즉시, 무비용):  halt[m] != 0 인 슬롯을 담당하는 블록은 본문을 건너뛴다.
                       그러나 grid.sync() 는 반드시 통과한다.
2단계 (반복 경계, 압축): 각 BiCG iteration 종료 배리어 직후,
                       지정 블록 0이 halt[] 를 스캔해 live_map/n_live 를 재작성.
                       grid.sync().  이후 모든 블록이 새 매핑을 읽는다.
```

- 1단계만으로는 halt된 슬롯의 블록이 여전히 grid에 존재하며 배리어 비용을 문다. 2단계 압축이 **가상화 루프의 상한을 줄여** 실제로 일을 없앤다.
- 압축 스캔은 슬롯 ≤ 64개의 직렬 스캔(단일 스레드, ~64 사이클) + 배리어 1회. iteration당 이미 17개 배리어가 있으므로 **상대 비용 6%**.
- **압축은 산술에 개입하지 않는다.** 어느 논리 블록이 어느 물리 블록에서 도는지만 바꾼다. §3.4.2의 논증이 그대로 적용된다.
- 배치 M64에서 sweep 후반부에는 대부분 슬롯이 halt되므로, 압축은 **iteration이 진행될수록 grid를 축소**시킨다. 이것이 현재 그래프 경로가 절대 할 수 없는 일이다(그래프는 gridDim이 고정).

#### 3.4.5 최대 위험 두 가지 — **설계 단계에서 못 박아야 하는 것**

**위험 A — `return`이 배리어를 발산시킨다.**
현재 모든 커널의 첫 줄은 `HALT_GUARD(halt + m)` = `if (*halt != 0) return;`이다(`CudaBICGBackend.cu:221`). persistent 커널 안에서 일부 블록이 `return`하면 나머지 블록의 `grid.sync()`가 **영구 대기**한다.

> **계약: persistent 커널 본문 안에는 `return`, `break`(배리어를 포함하는 루프에서), 그리고 배리어를 감싸는 조건 분기가 존재해서는 안 된다.** 모든 halt 게이트는 `if (!halted) { ...본문... } grid.sync();` 형태의 **술어화(predication)** 로만 표현한다. 이는 정적 계약 테스트로 검사한다(§3.12).

**위험 B — `__restrict__`가 융합 후에는 거짓말이 된다.**
현재 커널들은 `const double* __restrict__` 로 배열을 받는다. 서로 다른 커널이 같은 배열을 각각 읽고 쓰기 때문에 커널 경계가 별칭 없음을 보장해 왔다. **하나의 커널로 융합하면 컴파일러가 같은 함수 안에서 쓰기와 읽기를 모두 보고, `__restrict__`를 근거로 배리어 너머로 로드를 캐시하거나 저장을 재배치할 수 있다.** 결과는 조용한 오답이다.

> **계약: persistent 커널의 모든 전역 포인터에서 `__restrict__`를 제거한다.** 성능 손실 우려가 있으면 배리어 사이의 **국소 스코프**에서만 지역 `__restrict__` 별칭을 만든다. 추가로 배리어 직후의 첫 로드는 `__ldg` 대신 일반 로드로 두어 read-only 캐시 경로를 타지 않게 한다. `cg::grid_group::sync()`가 메모리 순서를 보장하지만, **컴파일러 최적화까지 막아주지는 않는다** — 이 구분이 이 위험의 본질이다.

검증: `compute-sanitizer --tool synccheck` (위험 A), `--tool racecheck` 및 bit-golden (위험 B).

#### 3.4.6 FP32 변형과의 상호작용

`cmfdFp32InnerEnabled()`는 프로세스당 1회 읽히고 캐시되며(`CudaBICGBackend.cu:131`), 실패 시 `latchFp32Off()`가 아레나 전체를 FP64로 되돌린다.

**설계**: persistent 커널을 `template <typename T> __global__ void cmfd_sweeps_persistent(...)`로 두 본(double/float) 인스턴스화한다.
- 정밀도는 launch 시점 선택이므로 **그래프 재캡처가 불필요**하다. 현재 `latchFp32Off()`가 `graph_exec`를 파괴하고 재인스턴스화 카운터를 올리는 로직(`CudaBICGBackend.cu:3049`)은 persistent 경로에서 **자연스럽게 단순해진다**.
- 단, FP32 본은 레지스터를 덜 쓰므로 `blocksPerSM`이 다르다 → **eligibility 질의를 두 인스턴스 각각에 대해 수행**하고, 둘 중 **작은 쪽**으로 물리 grid를 정한다(latch 전환 시 grid를 바꾸지 않기 위해).
- `refresh_operator_mirror_f32` 노드는 persistent 본문의 프롤로그로 흡수된다.
- FP32 비-유한 감지 시의 폴백은 현행 그대로: 커널이 flux를 쓰지 않고 상태 플래그만 세우고, 호스트가 `drain`에서 latch한다.

#### 3.4.7 ITER_BATCH / 그래프의 대체와 텔레메트리 계약

persistent 경로에서는 **캡처 깊이라는 개념이 사라진다**. 루프가 `1 + nmax`까지 돌고 halt로 자연 종료하므로:

| 카운터 | 그래프 경로 | persistent 경로 | 조치 |
|---|---|---|---|
| `graph_launches` | sweep 배치당 1 | 0 | `persistent_launches` 신설, 기존은 0 유지 |
| `iter_batch` | `max(1+nmax, K)` | `1+nmax` | 알고리즘 예산을 그대로 기록 |
| `overrun_iterations` | >0 | **0 (구조적)** | 0이 정상임을 receipt에 명시 |
| `batched_graph_launches` | ≤ `graph_launches` | 0 | 불변식 `batched ≤ launches` 유지됨 |
| `stream_sync_calls_during_iteration` | drain당 1 | 동일 | 불변 |

> `RASBERY_GPU_ITER_BATCH`는 persistent 경로에서 **무시된다**. 두 노브가 동시에 설정되면 **경고가 아니라 실패**시킨다(Rev.4 §2.1의 "경고 후 계속 실행 금지" 정신).

**신설 카운터** (`BackendCounters`):
```text
persistent_launches            persistent 커널 발사 수
persistent_ineligible          eligibility 실패로 그래프 폴백한 횟수
persistent_grid_blocks         실제 물리 grid (x)
persistent_barriers            발사당 grid.sync() 횟수 (설계값, 검증용)
compaction_events              live_map 재작성 횟수
active_slots_sum / _count      평균 활성 슬롯 (압축 효과 측정)
```

#### 3.4.8 eligibility 폴백 경로

```text
초기화 시 1회:
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kernel<double>, 256, 0)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM_f, kernel<float>, 256, 0)
    resident = min(perSM, perSM_f) × multiProcessorCount
    if (resident < 1)                  → 폴백
    if (!prop.cooperativeLaunch)       → 폴백
발사 시마다:
    G = min(resident, L_max × n_live)
    rc = cudaLaunchCooperativeKernel(..., G, 256, ...)
    if (rc != cudaSuccess) {
        cudaGetLastError();
        ++persistent_ineligible;
        persistent_enabled = false;    // 프로세스 수명 동안 sticky
        enqueue_sweeps(nmax, unroll);  // 기존 경로로 이번 sweep 배치 실행
    }
```

`launch_outer`의 그래프 캡처 실패 폴백(`CudaBICGBackend.cu:2668–2686`)이 확립한 계약을 그대로 따른다: **실패한 발사는 아무것도 실행하지 않았으므로 재실행이 중복 적용이 아니다.** 단, cooperative launch는 캡처와 달리 **부분 실행 가능성이 있다** — 발사가 성공했으나 커널이 도중에 죽는 경우. 이때는 폴백이 아니라 **실패**여야 한다. 따라서 위 폴백은 **발사 API가 에러를 반환한 경우에만** 적용하고, 커널 실행 중 에러(`cudaGetLastError` after sync)는 기존과 동일하게 예외로 올린다.

#### 3.4.9 환경 게이팅과 receipt

```text
RASBERY_GPU_CMFD_COMPACT=0|1     기본 0   Stage 1a
RASBERY_GPU_CMFD_PDL=0|1         기본 0   Stage 1c
RASBERY_GPU_CMFD_PERSIST=0|1     기본 0   Stage 1b (RASBERY_GPU_CMFD_SWEEP=1 필수)
```

세 노브 모두 **opt-in**(`envFlagEnabled` 계열, `CudaBICGBackend.cu:120`)으로 하여 unset이 기존 경로임을 보장한다. `RASBERY_GPU_CMFD_PERSIST=1`인데 `RASBERY_GPU_CMFD_SWEEP`이 없으면 실패시킨다.

Receipt (stderr 1행, 초기화 시 1회):
```json
[RASBERY][CUDA][PERSIST] {"enabled":true,"engine":"persistent","precision":"fp64",
 "sm":188,"blocks_per_sm":4,"resident_blocks":752,"logical_blocks":{"node":34,"vector":67,"reduce":17},
 "grid_blocks":752,"block_dim":256,"barriers_per_sweep":78,"compaction":true,
 "cooperative_launch":true,"fallback":"graph"}
```

SPTELEM 스키마는 **additive-only**로 확장한다(`schema_version` 유지, Rev.4 §8 정신). `tools/test_statepoint_telemetry.py`를 함께 갱신한다.

#### 3.4.10 계약 테스트 (구현과 같은 커밋)

기존 `tools/test_*_contract.py`의 **정적 토큰 검사** 양식을 그대로 따른다.

**`tools/test_cmfd_persistent_contract.py`**
1. `cooperative_groups.h` include 존재, `cg::this_grid()` 사용.
2. persistent 커널 본문 영역(마커 주석 사이) 안에 `return;` / `HALT_GUARD` 매크로 사용이 **0회**.
3. 같은 영역 안의 전역 포인터 인자에 `__restrict__`가 **0회**.
4. `blockDim` 하드코딩 상수 256이 커널 진입부에 `assert`/`if(blockDim.x!=256) trap` 형태로 존재.
5. reduction의 `chunk`가 `gridDim.x`가 아니라 **명시 인자 `L`** 에서 유도됨 (`gridDim.x` 문자열이 chunk 계산식에 없음).
6. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 호출 존재, `persistent_ineligible` 카운터 존재.
7. `cudaLaunchCooperativeKernel` 실패 시 `enqueue_sweeps` 폴백 호출 존재.
8. `RASBERY_GPU_ITER_BATCH`와 `RASBERY_GPU_CMFD_PERSIST` 동시 설정 시 throw 하는 코드 존재.

**`tools/test_cmfd_slot_compaction_contract.py`**
1. `slot_map` 간접이 모든 `blockIdx.y` 사용처에 적용됨 (`blockIdx.y`를 직접 m으로 쓰는 곳이 0).
2. 버킷 테이블과 `graph_exec` 캐시 존재, 버킷 수 유계.
3. 항등 매핑 경로 존재 (`n_active == slots`일 때 `slot_map[j]==j`).
4. `host_slot_map`이 `host_active`와 **같은 H2D**에 실림 (전송 횟수 불변).

**`test/cmfd_partition_contract.cpp`** (신규, 호스트 전용 산술 계약 — `test/cmfd_form_probe.cpp` 양식)
- `L ∈ {17, 34, 67}`, `G ∈ {1..L}` 전수에 대해, 가상화 루프가 생성하는 `(lb, begin, end)` 집합이 물리 파티션의 그것과 **집합으로 동일**함을 단언.
- 임의 double 배열에 대해 stage1+stage2 fold 결과가 `G`에 무관하게 **bit-identical**임을 단언.
- CI에서 CUDA 없이 돌아야 하므로 호스트 코드로 공유 본문을 추출한다(`CmfdAssemblyKernel.h`가 `RASBERY_CMFD_ASSEMBLY_HD`로 한 것과 동일한 수법).

#### 3.4.11 검증 arm과 게이트

| arm | 구성 | 성격 | 게이트 |
|---|---|---|---|
| A0 | 전부 off | 기준선 | **byte-identical** 500/500 · 708/708 |
| A1 | `COMPACT=1` | 궤적 보존 | **bit-golden** |
| A2 | `COMPACT=1 PDL=1` | 궤적 보존 | **bit-golden** |
| A3 | `COMPACT=1 PERSIST=1` (FP64) | 궤적 보존 | **bit-golden** |
| A4 | A3 + `CMFD_FP32=1` | 궤적 변경(기존 FP32와 동일 성격) | 기존 FP32 arm과 **bit-identical** |
| A5 | A3, `--batch-mode 64`, 1,280 케이스 | 안정성 | FAIL 0, PIN 수신증 무결, `persistent_ineligible=0` |

> **Stage 1 전체가 궤적 보존 설계다.** bit-golden 불일치는 "허용오차 안"이 아니라 **버그**로 취급한다. Gate A/B는 Stage 1에서 **사용하지 않는다** — 사용해야 할 상황이 오면 설계가 틀린 것이다.

성능 게이트는 §2.7 게이트 4/5 + Rev.4 §3.6(warm-up 1, hot 3, balanced order, median 보고).

**독점 위험 측정** (Rev.4 §11 마지막 문단): cooperative launch는 GPU 전체를 점유할 수 있다. 다음을 A3에서 반드시 기록한다.
```text
GPU active (%)             nsys --gpu-metrics
SM occupancy               ncu 또는 gpu-metrics
other-stream wait          nodal/xsrecon 스트림의 대기 시간 (nsys)
CMFD batch width           SPTELEM
Nodal batch width          nodal arena receipt
arrival gap EWMA           [RASBERY][CUDA][BATCH] receipt
end-to-end throughput      cases/h
```
**other-stream wait가 5% 초과 악화하면 채택하지 않는다** — 이는 Rev.4의 명시 조건이다.

#### 3.4.12 기대 이득 (Amdahl, 가정 명시)

**가정**
- (a) 단일덱 solve 91.3 s, drive 55.5 s, end-to-end 94.6 s.
- (b) drive 안의 그래프 노드 dispatch = 9.4 s (§0 결론 3, Stage 0-B/D로 재확인 필수).
- (c) 배리어 축약비 **0.955** (§3.4.3, 전수 조사로 확정. 낮출 수 없음).
- (d) `c_barrier = 0.30 us` (Stage 0-E가 이 값을 주었다고 **가정**; 미측정. 임계는 0.45 us).
- (e) 압축은 단일덱에서 0, 배치에서 `s=0.4`.
- (f) Stage 1a/1c/1b의 이득은 같은 `G_intra` 풀에서 나오므로 **단순 합산하지 않는다**. 1c가 먼저 먹은 부분을 1b가 다시 먹지 못한다.
- (g) 레지스터 상주로 인한 재로드 절감(§0 결론 4 말미)은 **계산에 넣지 않는다** — 프로토타입(§3.4.13)만이 답할 수 있다. 상방 여지로만 취급.

**단일덱**
```text
9.4 s × (1 − 0.955 × 0.30/0.914) = 9.4 × 0.6866 = 6.5 s 제거
solve 91.3 → 84.8,  end-to-end 94.6 → 88.1   ⇒ 1.074×
Stage 0-A(무료 실험)가 먼저 성공하면 기준선이 이미 ~80 s이므로
    80 → 73.5 s  ⇒  1.09×

c_barrier 가 임계(0.45us)에 걸리면 제거량은 4.9 s 로 떨어지고
c_barrier 가 0.6us 면 3.5 s (게이트 2 탈락) — 즉 이 항의 불확실성이
Stage 1b 이득 전체의 폭과 같다.
```

**배치 M64** (`s=0.4`, 평균 폭 22/64)
```text
Stage 1a 단독:  slot-비례 비용의 (1 − 22/64) = 65.6% 제거
                 → solve 항의 0.4 × 0.656 = 26.2% 제거
Stage 1b 추가:  dispatch 잔여분 (압축 후에는 절대량이 이미 1/3로 줄어 있음)
```
**단, 배치 wall은 현재 io_wall 40.7%(SPTELEM 실측)가 지배한다.** HDF5 writer 분리(캠페인 우선순위 1번)가 선행되지 않으면 **Stage 1의 배치 이득은 대부분 I/O 뒤에 숨는다.** 이는 Anderson −38% outer 감축이 배치 처리량에 나타나지 않은 것과 정확히 같은 구조다.

> **판정: Stage 1의 배치 채택 벤치마크는 HDF5 writer 분리 이후에만 유효하다.** 그 전에 측정한 배치 c/h는 Stage 1의 효과를 측정하지 못한다. Stage 0-B에서 이를 확인(io_wall 비중 재측정)하고, 40% 이상이면 **Stage 1의 배치 arm을 HDF5 뒤로 미룬다**.

**공수**: Stage 1b = 14 인일 (설계 확정 2, 커널 구현 5, 압축/가상화 3, 폴백·텔레메트리 1, 계약 테스트 1, 검증 2). **단, §3.4.13의 스파이크 2 인일이 선행 조건이다.**

#### 3.4.13 프로토타입 스파이크 (14 인일 투입 전 필수 관문, **2 인일**)

마이크로벤치(§2.6)는 `c_barrier`만 준다. 레지스터 상주 효과(상방)와 레지스터 압력에 의한 점유율 하락(하방)은 **실제 커널로만** 측정된다. 따라서 본 구현 전에 다음을 만든다.

**스파이크 범위**: BiCGSTAB **iteration 1개**(22 노드 → 21 배리어)만 persistent로 작성. 단일 슬롯, 압축 없음, FP64만, 폴백 없음, 계약 테스트 없음. `tools/spike_persistent_iter.cu`로 독립 빌드하며 **프로덕션 소스는 건드리지 않는다.**

**측정 항목**
```text
1. registers_per_thread, blocksPerSM, resident_blocks   (융합 후 실제값)
2. 21 배리어 persistent 판  vs  22 노드 그래프 판  의 iteration 시간
3. 두 판의 DRAM read bytes  (재로드 절감이 실재하는지)
4. bit 동일성: 동일 입력에 대해 두 판의 출력 벡터가 byte-identical 인가
   (4번이 실패하면 §3.4.5-A/B 중 하나가 이미 발생한 것 — 본 구현 전에 잡는다)
```

**통과 조건**
```text
persistent iteration 시간  ≤  graph iteration 시간 × 0.93
AND  blocksPerSM ≥ 2   (resident_blocks 가 논리 블록 수를 감당)
AND  bit-identical
```
**하나라도 실패하면 Stage 1b는 NO-GO**이고, 남은 14 인일을 Stage 2/3-A로 돌린다.

---

## 4. Stage 2 — 잔여 CPU 조립의 GPU 이관

**대상**: `upddhat` 5.8 s, `updjnet` 2.5 s, `setls` 12.1 s (합계 20.4 s = solve 91.3 s의 **22.3%**).

> Stage 1(9.4 s 풀)보다 **더 큰 풀**이다. 그리고 `c_barrier` 같은 미지수에 걸려 있지 않다. **Stage 2가 Stage 1b보다 기대값이 높다.**

### 4.1 setls — 12.1 s

**진단**: §0 결론 1. 단일덱은 아레나가 없어 device assembly가 아예 불가능하다.
**조치**: 두 갈래.
- **2-A (무료)**: Stage 0-A가 성공하면 `--batch-mode 1`을 단일덱 표준 구성으로 채택. **코드 변경 0.**
- **2-B (구조)**: `canUseDeviceAssembly()`의 `arena() != nullptr` 요구를 없애고 단일 인스턴스 백엔드에도 `cmfd_assemble_operator_2g` + `enqueue_sweeps` 경로를 열어준다. 아레나 폭 1과 기능적으로 동치이나 rendezvous 오버헤드가 없다.
- **권고**: 2-A로 **먼저 측정**하고, rendezvous 오버헤드가 유의하면 2-B. 2-A만으로 충분하면 2-B는 하지 않는다.
- **공수**: 2-A는 0 (Stage 0에 포함), 2-B는 4 인일.
- **게이트**: 궤적 보존 → bit-golden. (device assembly는 이미 `CMFD_FUSION_VALIDATION_20260825_KO.md`에서 검증된 경로다.)

### 4.2 upddhat — 5.8 s

**의존성** (`CMFD.cpp:126`): 표면 `ls`마다 독립. 읽기 = `flux[node]`, `jnet[surface]`, `dtil[surface]`. 쓰기 = `dhat[surface]`. **표면 간 완전 독립 → 1 스레드 1 (표면, 그룹).** `nsurf × 2` 워크아이템.

**부작용 상태 4개**: `_dhat_total`, `_dhat_fsum_guard`, `_dhat_clamped` (카운터), `_dhat_ratio_max` (최대값).
- 카운터 3개 → 블록 내 `__syncthreads` 축약 후 블록당 `atomicAdd` 1회. **값은 순서 무관 정수합이므로 bit 문제 없음.**
- `_dhat_ratio_max` → `kernelXsRecon`의 `max_bits` 패턴(`atomicMax` on `unsigned long long` bit pattern) 재사용. **선례 존재.**

**bit 보존의 진짜 난점**: `dh = (jnet_fdm - jnet) / fsum` 및 `jnet_fdm = -dtil * fdiff`의 **contraction 형태**. gcc 13 `-O3 -march=native`가 FMA로 접었는지 여부를 코드를 읽어 알 수 없다 — `CmfdAssemblyKernel.h:46–52`가 정확히 이 문제를 겪었고, 거기서 확립한 해법이 **form probe**다.

> **필수 선행: `RASBERY_CMFD_DUMP` 계열의 dhat/jnet 덤프를 추가하고 `test/cmfd_form_probe.cpp`를 확장하여 gcc가 실제로 방출한 contraction을 채굴한다.** 이것 없이 커널을 쓰면 bit-golden이 반드시 깨지고, 원인을 사후에 이분탐색하게 된다. **공수의 절반이 여기다.**

**분기 두 개** (`fsum` floor 가드, `isfinite` 가드): 워프 발산은 성능 문제일 뿐 정확도 문제가 아니다. 두 가드가 각각 2.72%/소수 비율로 발화한다는 실측이 있으므로 발산 비용은 무시 가능.

**공수**: 6 인일 (form probe 3, 커널 2, 검증 1). **게이트**: form이 고정되면 궤적 보존 → bit-golden. 고정에 실패하면 Gate A/B.

### 4.3 updjnet — 2.5 s

`CMFD.h:240`. upddhat과 같은 표면 루프 구조이며 `dhat`을 읽어 `jnet`을 쓴다. **upddhat 커널과 같은 인덱스 도메인이므로 융합 후보이나, 두 함수는 outer 안에서 서로 다른 지점에 있다**(updjnet은 nodal 앞, upddhat은 nodal 뒤). 융합하면 순서가 바뀌므로 **금지**. 별도 커널로 이식한다.

**공수**: 3 인일 (form probe는 4.2와 공유). **게이트**: 동일.

### 4.4 Stage 2 종합

```text
제거 대상   20.4 s
GPU 실행비  −20% 가정 (표면 도메인은 nsurf×2 ≈ 5만 워크아이템, grid 200블록 — 여전히 GPU 기아)
전송비      dtil/dhat/jnet은 이미 mirror 대상(dtil_mirror 존재) → 추가 전송 최소
순 제거     ~15 s

solve 91.3 → 76.3,  end-to-end 94.6 → 79.6   ⇒  1.19×
Stage 1(6.5 s)과 합치면  94.6 → 73.1        ⇒  1.29×
```

**Stage 2 총 공수: 13 인일** (2-A 0 + 2-B 4 + upddhat 6 + updjnet 3, form probe 공유로 −0).

**위험**: Stage 2는 CPU 코드를 GPU로 옮기는 것이므로 **호스트가 이 배열들을 읽는 다른 소비자**를 전수 감사해야 한다. `_dhat`/`_dtil`은 `setls`/`assembleHostLinearSystem`/PPR/HDF5 출력이 읽을 수 있다. `CudaTransferMirror`의 mirror-back 규약(`issueExceptionalOperatorDownloads`)을 확장해 **호스트 관측 시점에 반드시 일치**하도록 한다. 이것이 Stage 2의 진짜 작업량이다.

---

## 5. Stage 3 — Outer 루프 전체 상주화

### 5.1 outer 한 바퀴의 해부와 device 이식 가능성

| # | 단계 | 현재 | device 이식 | 난이도 | 비고 |
|---|---|---|---|---|---|
| 1 | `updpsi` | host | 자명 | 1 인일 | 노드 독립, `cmfd_wiel_terms`와 동형 |
| 2 | `setls` | host/GPU | **완료 예정** | Stage 2 | |
| 3 | `drive` (sweeps) | GPU | 완료 | Stage 1 | |
| 4 | `updjnet` | host | Stage 2 | | |
| 5 | `nodal reset+drive` | GPU (6커널) | 완료 | | `kNodal*` |
| 6 | `ApplyRodCusping` | **host** | **어려움** | — | XS 수준 blend 로직, `_hoststate_generation` 갱신 |
| 7 | `upddhat` | host | Stage 2 | | |
| 8 | 수렴 판정 | host 스칼라 | 자명 | 0.5 인일 | `|dk|<tol && residual<tol` |
| 9 | Xe 평형 스텝 | host (또는 융합 GPU 커널) | **부분** | §5.3 | |
| 10 | settle gate | host 카운터 | 자명 | | |
| 11 | 보론 탐색 | host (secant + 이력) | 가능하나 **무가치** | — | statepoint당 수십 회 |
| 12 | T/H 피드백 | host | 가능하나 **무가치** | — | 동상 |
| 13 | depletion 경계 | host + HDF5 | **불가** | — | 파일 I/O |

> **6번(ApplyRodCusping)이 outer를 device에 완전히 가두는 것을 막는 유일한 물리 항목이다.** 11~13번은 device에 올릴 수는 있으나 호출 빈도가 outer의 1/100~1/1000이라 **처리량 이득이 0**이다.

### 5.2 세 가지 아키텍처 후보 평가

#### 후보 1 — device-callback outer (`cudaLaunchHostFunc`)

호스트 콜백을 스트림에 삽입해 제어를 device 스트림 순서 안에 넣는 방식.
- **기각.** `cudaLaunchHostFunc`의 콜백은 CUDA API를 호출할 수 없다. 즉 콜백 안에서 다음 커널을 발사할 수 없으므로 outer 루프를 구성할 수 없다. 또한 host 노드는 **conditional graph node 본문에 넣을 수 없다**. 구조적으로 막다른 길.

#### 후보 2 — 완전 device state machine (device graph launch / persistent mega-kernel)

- device-side graph launch(`cudaGraphLaunch` from device)는 **host 노드 불가, cooperative launch 불가, cuBLAS 불가**. Stage 1b의 persistent 커널과 **상호 배타**다.
- persistent mega-kernel로 outer 전체를 도는 것은 6/11/12/13번 때문에 불가능하며, 가능하다 해도 **디버깅 불가능한 단일 커널**이 된다.
- **기각.** 단, "6번을 만나면 device가 플래그를 세우고 호스트로 나온다"는 escape hatch를 두면 후보 3과 합류한다.

#### 후보 3 — CPU-orchestrated mega-graph + CUDA 13 conditional nodes **[채택]**

CUDA 12.3에서 conditional `IF`, 12.4에서 `WHILE`, 12.8에서 `IF/ELSE`·`SWITCH`가 도입되었고 CUDA 13은 전부 지원한다. 조건은 `cudaGraphConditionalHandle`을 device 커널이 `cudaGraphSetConditionalNodeValue()`로 갱신한다.

```text
graphExec(outer_loop):
  WHILE(handle_flux_not_converged) {
      updpsi_k
      assemble_operator_k          (Stage 2)
      sweeps  (graph 또는 persistent — §5.4의 분기)
      updjnet_k                    (Stage 2)
      nodal_k ×6
      IF(handle_cusping_needed) { set_escape_flag_k }   ← host로 탈출
      upddhat_k                    (Stage 2)
      converge_test_k              → handle_flux_not_converged 갱신
                                   → handle_escape 갱신
  }
```

- **호스트 개입이 outer당 1회에서 flux-수렴 세그먼트당 1회로 줄어든다.** flux 재수렴 세그먼트는 평균 수십 outer이므로 **호스트 왕복이 1~2 자릿수 감소**한다.
- 각 outer의 `cudaStreamSynchronize`(현재 sweep마다 1회 = 108,176회)가 세그먼트당 1회로.
- `escape` 경로: cusping이 발화하거나 Xe/search/T-H가 개입해야 하면 device가 플래그를 세우고 WHILE이 종료. 호스트가 처리 후 재진입. **cusping 발화율이 낮다는 실측(수렴 근처에서 드묾)이 이 설계의 전제이며, Stage 0에서 계수를 세어 확인한다.**

**확인 필요 사항 (238에서 프로브 필수, 본 문서가 단정하지 않음)**
1. conditional node **본문에 cooperative launch가 들어갈 수 있는가.** 들어갈 수 없다면 §5.4의 분기가 발동한다.
2. `cudaGraphSetConditionalNodeValue`를 호출하는 커널이 conditional 본문 **안**에서 다음 반복의 핸들을 갱신할 수 있는가(WHILE의 표준 사용법이므로 가능할 것이나 확인).
3. conditional 본문의 재인스턴스화 비용 — 본문이 고정 토폴로지이므로 1회면 족한가.

### 5.3 Xe 평형의 device 이식 — 별도 평가

- `EvaluateEquilibriumXenon`(`XSSet.cpp:3825`)은 **의도적으로 host 전용**이다. 소스 주석(`XSSet.cpp:3786–3794`)이 이유를 명시한다: device arm(`XsReconKernel.h`)은 evaluate+damp+apply+reconstruct를 **하나로 융합**해 두었으므로 evaluate-only 진입점이 없고, 커널을 쪼개면 이미 게이트를 통과한 bit-exact A/B가 갈라진다.
- Anderson은 `Evaluate`(부작용 없음) → 호스트 대수(m=2 Gram, 조건수, 4중 안전조건) → `Commit`을 요구한다.
- **이식 비용**: 융합 커널을 evaluate/commit로 분할(기존 bit-exact 계약 재검증 필요), 3n 벡터의 device 상주, m=2 대수의 device 구현(스칼라 수십 개, 자명), 이력 관리.
- **이식 가치**: Xe 스텝은 실행당 2,097회(AA1). outer 10,483회 대비 **1/5**. 각 스텝의 호스트 비용은 ~8.5k 노드 응축이며 주석이 "flux 재수렴보다 1~2 자릿수 작다"고 명시.
- **판정: 기각(또는 무기한 보류).** 비용 25~35 인일, 기대 이득 solve의 1% 미만. Stage 3-A의 escape 경로에 Xe를 포함시켜 호스트가 처리하게 두는 것이 옳다.

### 5.4 Stage 1b(persistent)와 Stage 3-A(conditional graph)의 상호 배타 가능성

**이것이 본 계획에서 유일한 진짜 아키텍처 분기다.**

| 케이스 | conditional 본문에 cooperative 가능? | 결론 |
|---|---|---|
| 가능 | — | Stage 1b + 3-A 병존. 최선. |
| 불가 | — | **택일.** 아래 규칙 적용. |

**택일 규칙** (Stage 0/§5.2 프로브 결과로 자동 결정):
```text
if (c_barrier ≥ 0.45 us  OR  스파이크(§3.4.13) 미통과)
                                        → Stage 1b 자체가 NO-GO.  3-A 단독 진행.
else if (conditional 본문 cooperative 불가):
     R_1b (§3.4.12 6.5 s)  vs  R_3A (§5.5)  를 비교해 큰 쪽.
     현재 추정으로는 R_3A > R_1b 이므로 3-A 우선.
     이 경우 sweeps 는 conditional 본문 안에서 기존 graph 경로로 실행하고,
     persistent 는 채택하지 않는다.
```

> **선제 권고: 3-A가 1b보다 기대값이 크다.** 1b는 노드 dispatch 9.4 s의 69%(6.5 s)를 노리고 미지수 하나에 걸려 있는 반면, 3-A는 호스트 왕복 108,176회 + outer당 호스트 제어 시간(잔여 9.3 s "기타" 항의 상당분)을 노리며 미지수가 API 지원 여부뿐이다.

### 5.5 Stage 3 하위 단계와 공수

| 하위 단계 | 내용 | 공수 | 판정 |
|---|---|---:|---|
| **3-A1** | `updpsi` + `converge_test` device 커널화, conditional 핸들 배선 | 4 인일 | **GO** |
| **3-A2** | WHILE conditional로 flux 재수렴 루프 구성, escape 경로 | 10 인일 | **GO** (Stage 2 완료 후) |
| **3-A3** | `ApplyRodCusping` escape의 IF 노드화 + 발화율 계측 | 4 인일 | **GO** |
| **3-B** | Xe evaluate/commit device 분할 + Anderson device 대수 | 25~35 인일 | **기각** (§5.3) |
| **3-C** | 보론 탐색 / T-H / settle을 device state machine으로 | 60+ 인일 | **영구 기각** — 호출 빈도가 outer의 1/100 이하, 이득 0 |
| **3-D** | depletion 경계 device 상주 | — | **불가** (HDF5 I/O) |

**Stage 3 채택 범위 = 3-A1 + 3-A2 + 3-A3 = 18 인일.**

**Stage 1+2 이후 잔여 천장 재계산 (Amdahl)**

```text
출발:            end-to-end 94.6 s  (solve 91.3, IO 2.8, fixed 0.48)
Stage 0-A 후:    ~80 s              (setls 12.1 + sweep 왕복 일부)   ← 코드 0줄
Stage 2 후:      ~72 s              (upddhat 5.8 + updjnet 2.5, GPU비 차감)
Stage 1 후:      ~66 s              (dispatch 6.5 s, c_barrier=0.30 가정)
Stage 3-A 후:    ~60 s              (호스트 제어·왕복 잔여의 60%)

이론 바닥:  IO 2.8 + fixed 0.5 + GPU 순연산
            GPU 순연산 = drive 55.5 − dispatch 9.4 = 46.1 s
            (단일덱 grid 34~67 블록 = 188 SM의 18~36% — 이 46.1 s 자체가
             연산이 아니라 메모리/지연 바운드이며, 폭을 못 채우는 한 줄지 않는다)
            바닥 ≈ 50 s

단일 GPU · 단일덱 상한 ≈ 94.6 / 50 = 1.9×
```

> **이 1.9×가 Phase 5 전체의 정직한 상한이다.** 수십 배는 여기서 나오지 않는다 — 캠페인 보고서 §7의 결론(멀티GPU dispatcher)이 그대로 유효하다. Phase 5의 가치는 배수가 아니라 **배치 모드에서 slot 압축이 폭 확대(Phase 6)를 되살린다**는 데 있다.

---

## 6. 리스크 레지스터

| ID | 위험 | 영향 | 확률 | 완화 | 검출 |
|---|---|---|---|---|---|
| **R-1** | `c_barrier ≥ 0.45 us` → Stage 1b 무가치 (배리어 축약비가 0.955로 확정되어 dispatch↔barrier가 거의 1:1 치환이므로) | Stage 1b 폐기 (14 인일 절약) | **높음** | 두 관문을 구현 **앞에** 배치: Stage 0-E 마이크로벤치 → §3.4.13 스파이크(2 인일). 1a/1c/2/3-A가 1b 없이도 독립적으로 성립하도록 설계 | Stage 0-E, 스파이크 |
| **R-2** | cooperative launch가 64-슬롯 아레나에서 점유율 천장에 걸림 | 발사 실패 또는 grid 축소로 이득 소멸 | **높음** | 논리 블록 가상화(§3.4.2) + 압축(§3.2)로 물리 grid를 resident 이하로 강제. 실패는 **에러 반환**이지 hang이 아님 | eligibility 질의, `persistent_ineligible` |
| **R-3** | persistent 커널이 다른 스트림(nodal/xsrecon)의 overlap을 제거 | 전체 wall 악화 | 중 | Rev.4 §11 명시 조건. `other-stream wait` 5% 악화 시 **채택 거부** | nsys `--gpu-metrics`, A5 arm |
| **R-4** | 공유 서버 GPU의 watchdog/preemption | 장시간 persistent 커널 강제 종료 | 중 | 238은 headless(디스플레이 없음)이므로 watchdog 부재로 **추정**하나 Stage 0에서 `cudaDevAttrKernelExecTimeout` 질의로 확인. MPS/MIG 사용 시 cooperative launch 제약도 함께 확인. sweep `unroll` 상한으로 커널 수명을 유계로 | `probe_occupancy` 확장 |
| **R-5** | 배리어 발산(`return`/조건부 배리어) → 영구 hang | 배치 전체 정지 | **높음** | §3.4.5-A 계약 + 정적 테스트 + `compute-sanitizer --tool synccheck` 필수 통과 | 계약 테스트, synccheck |
| **R-6** | `__restrict__` 융합 후 오최적화 → 조용한 오답 | **bit-golden 실패, 최악의 경우 통과하고 물리만 틀림** | 중 | §3.4.5-B 계약 (모든 전역 포인터에서 제거) + racecheck + bit-golden | 계약 테스트, bit-golden |
| **R-7** | MSVC/Windows 경로 | 181 서버 빌드 깨짐 | 중 | cooperative groups는 MSVC에서도 지원되나 `-rdc`/링크 차이 존재. **181은 단일·비배치 회귀 전용**(Rev.4 §3.6)이므로 `RASBERY_GPU_CMFD_PERSIST` 기본 off로 두면 경로가 아예 실행되지 않음. **빌드는 반드시 통과해야 하므로 stub 경로(`CudaBICGBackendStub.cpp`) 대칭 유지** | 181 빌드 CI |
| **R-8** | rolling queue / HostPinLease와의 상호작용 | pin 수명 앨리어싱 재발 | 중 | Stage 1a의 `slot_map`은 **새 호스트 버퍼를 도입하지 않는다**(`host_active`와 같은 H2D). 새 pinned 범위 0 → Rev.4 §6 계약 무영향. 계약 테스트로 못 박음 | `test_host_pin_registry.py`, PIN 수신증 |
| **R-9** | 버킷별 `graph_exec` 9개 × 정밀도 2 → VRAM/재인스턴스화 폭증 | 메모리·초기화 지연 | 낮 | 버킷 수를 9로 유계. `graph_reinstantiations` 카운터로 감시, 실행당 20 초과 시 실패 | 카운터 |
| **R-10** | SPTELEM 계측을 persistent 루프 **안**에 넣을 수 없음 | outer 해부 불가 → Phase 2 텔레메트리 회귀 | 중 | device 카운터를 커널 안에서 누적(`device_counters` 확장, 이미 `kCounterSlots` 존재)하고 drain에서 수확. **호스트 타이머는 세그먼트 단위로만.** 계측 자체가 배리어를 추가하지 않도록 `clock64()` 사용은 블록 0에 한정 | SPTELEM 스키마 테스트 |
| **R-11** | 디버깅 난이도 | 회귀 시 원인 규명 지연 | **높음** | `RASBERY_GPU_CMFD_PERSIST=0`이 항상 동등한 참조 경로. 이분탐색이 항상 가능하도록 **두 경로를 영구 병존**시키고 삭제하지 않는다. 배리어 카운트 receipt로 설계-구현 일치 확인 | receipt, A/B |
| **R-12** | 배치 이득이 HDF5 I/O에 가려짐 | Stage 1/2 채택 판단 오류 | **높음** | §3.4.12 판정. **HDF5 writer 분리 전에는 배치 c/h로 Stage 1/2를 판정하지 않는다.** 단일덱 + `solver_only_cases_per_hour`로 판정 | SPTELEM `io_wall` 비중 |
| **R-13** | conditional node 본문에 cooperative 불가 | Stage 1b와 3-A 택일 | 중 | §5.4 택일 규칙을 **사전에** 고정 (사후 합리화 방지) | Stage 0 프로브 |
| **R-14** | Rev.4 §11 eligibility 문언을 임의 확장 | 계약 위반 | 중 | §2.4의 확장 해석을 §7 게이트 표에 명시하고 **Rev.5 반영을 별도 건의**. 침묵하고 통과시키지 않음 | 문서 리뷰 |

---

## 7. 검증·게이트 총괄표

| 단계 | 변경 성격 | 정확도 게이트 | 성능 게이트 | 계약 테스트 | 진입 조건 |
|---|---|---|---|---|---|
| **S0-A** `--batch-mode 1` | 궤적 보존 | bit-golden 500/500 | median wall 개선 | 없음(구성만) | Rev.4 §2.1 |
| **S0-B~F** 측량 | 무변경 | — | — | — | 도구 프로브 |
| **1a** 압축 | 궤적 보존 | bit-golden 500/500·708/708, feature-off byte-불변 | §2.7 게이트 4 | `test_cmfd_slot_compaction_contract.py` | 없음 |
| **1c** PDL | 궤적 보존 | bit-golden | `G_intra` 감소 실측 | 토큰 계약 | PDL 프로브 통과 |
| **1b** persistent | 궤적 보존 | bit-golden (A3), FP32 arm과 bit-identical (A4) | §2.7 게이트 1·2·3·5 | `test_cmfd_persistent_contract.py`, `cmfd_partition_contract` | **`c_barrier < 0.45 us`** + **스파이크 §3.4.13 통과** + eligibility + Rev.4 §11 |
| **2-B** setls 단일덱 | 궤적 보존 | bit-golden | setls 항 소거 | 기존 assembly 계약 확장 | S0-A 결과 |
| **2** upddhat/updjnet | form 고정 시 궤적 보존 / 실패 시 궤적 변경 | bit-golden **또는** Gate A+B | 8.3 s 항 소거 | form probe 확장 + 토큰 계약 | form probe 성공 |
| **3-A** conditional graph | 궤적 보존 | bit-golden | 호스트 왕복 수 감소, wall | 신규 토큰 계약 | Stage 2 완료 + §5.4 택일 |
| **3-B** Xe device | — | — | — | — | **기각** |
| **3-C/D** | — | — | — | — | **기각** |

**공통 규약** (모든 단계에 적용, Rev.4 인용)
- §2.1 exact-only 시작 검사, §2.2 physics-mode receipt 필수. 누락 시 run 무효.
- §3.4 기준선 동결: 구현 전 동일 입력·동일 GPU 3~5회 실행, threshold 동결 후 수정 금지. `test/reference/validation_baseline_manifest_v1.json` 갱신.
- §3.5 두 지표 병기: `end_to_end_cases_per_hour`(채택 지표) + `solver_only_cases_per_hour`(귀속 지표).
- §3.6 warm-up 1 제외, arm당 hot 3회, balanced/randomised 순서, median + min/max, GPU clock/power/temp·CPU load·타 GPU 프로세스·VRAM·pinned RAM 기록.
- **성능·배치 정확도 판정은 238 GPU0 단독. GPU1 사용 금지. 181은 MSVC 빌드·stub 링크·단일 비배치 회귀 전용.**
- 모든 신규 노브는 **opt-in**, unset이 기존 경로. feature-off 경로는 **항상 byte-identical**.

**Gate A / Gate B 적용 규칙 (명확화)**
- Stage 1 전체와 Stage 2-B, Stage 3-A는 **궤적 보존 설계**다. bit-golden 불일치는 허용오차 사안이 아니라 **버그**이며, Gate A/B로 넘기지 않는다.
- Stage 2의 upddhat/updjnet만이 **form probe 실패 시** Gate A(Δkeff ≤ 1 pcm, ΔCBC ≤ 1 ppm, 핀 RMS 비퇴행) + Gate B(|Δρ| ≤ 2 pcm, |ΔCBC| ≤ 15 ppm, 핀 RMS ≤ 0.30%) 대상이 된다.

---

## 8. 실행 순서 · 담당 · 공수

### 8.1 순서 (의존성 순)

```text
[0]  S0-A  --batch-mode 1 무료 실험                     ← 최우선, 코드 0줄
      │
      ├─ 성공 → 단일덱 기준선 재동결 (setls 소거된 새 기준선)
      └─ 실패 → 별건 결함으로 분류, 규명 후 재개
[1]  S0-E  c_barrier + c_dispatch 마이크로벤치          ← Stage 1b 운명 결정
[2]  S0-B/C/D  Nsight 측량 (또는 대체 계측)
[3]  S0-F  폭 스윕 → s 측정
[4]  Stage 1a  압축                                     ← 배치 이득의 본체
[5]  Stage 1c  PDL                                      ← 저위험
[6]  Stage 2   upddhat/updjnet (+ 2-B 필요 시)          ← 최대 풀 (20.4 s)
[7]  Stage 1b  스파이크(2 인일) → 통과 시에만 본 구현 12 인일
                              [c_barrier < 0.45us AND 스파이크 통과]
[8]  Stage 3-A conditional mega-graph
      (병행 트랙, 본 계획 밖) HDF5 writer 분리 — 배치 판정의 전제
```

> **순서에 대한 명시적 권고**: Rev.4 §15.1은 Phase 5를 9번째에 두었다. 본 계획은 그 안에서 **Stage 2를 Stage 1b보다 앞에 둔다**. 근거는 §4 서두 — 풀이 2.2배 크고, 미지수가 없다.

### 8.2 담당

| 역할 | 담당 | 범위 |
|---|---|---|
| 설계·구현 | **Opus** | 커널 설계, 계약 정의, 소스 변경, 계약 테스트 작성 |
| 배치·계측·벤치 | **Sonnet** | 238 GPU0 단독. 빌드, arm 실행, nsys/ncu, CSV 수집, receipt 파싱, MASTER 비교(`tools/compare_master_rasbery.py`) |
| 금지 | — | **GPU1 사용 금지.** 181 서버에서 성능·배치 판정 금지. |

각 Stage 종료 시 산출물: 커밋 + `docs/PHASE5_STAGE<n>_*_KO.md` 보고서 (Rev.4 §15.2 양식) + 갱신된 baseline manifest.

### 8.3 공수 총계 (인일)

| 항목 | 공수 | 조건 |
|---|---:|---|
| Stage 0 (A~F, 프로브 포함) | 2 | — |
| Stage 1a 압축 | 5 | — |
| Stage 1c PDL | 3 | PDL 프로브 통과 |
| Stage 2 (2-B + upddhat + updjnet) | 13 | form probe 성공 |
| Stage 1b 스파이크 | 2 | `c_barrier < 0.45 us` |
| Stage 1b 본 구현 | 12 | 스파이크 통과 |
| Stage 3-A (A1+A2+A3) | 18 | Stage 2 완료, §5.4 택일 |
| **채택 경로 합계** | **55** | 전부 GO인 경우 |
| **최소 경로 합계** | **23** | S0 + 1a + 1c + 2 (1b/3-A 없이) |

**기대 결과**: 최소 경로만으로 단일덱 94.6 → 약 72 s (**1.31×**), 전 경로 완주 시 약 60 s (**1.58×**), 이론 바닥 50 s (**1.9×**).

---

## 부록 A — 본 문서가 **단정하지 않고 프로브를 요구하는** 사실

임의 재해석이나 기억에 의한 API 단정을 막기 위해, 확인이 필요한 항목을 명시한다. **Stage 0에서 전부 확인한 뒤 이 부록을 실측값으로 갱신한다.**

1. RTX PRO 6000 Blackwell Server Edition의 `multiProcessorCount`, `maxThreadsPerMultiProcessor`, `regsPerMultiprocessor`, `cooperativeLaunch`, `kernelExecTimeoutEnabled` — `probe_occupancy` (§2.4-B)
2. 대상 커널들의 `registers_per_thread` → `blocksPerSM` → `resident_blocks` (§2.4)
3. `c_barrier(sm_120, G)` 및 동일 조건 `c_dispatch` (§2.6) — **Stage 1b의 유일한 결정 인자**
4. CUDA 13에서 **conditional graph node 본문에 cooperative launch가 허용되는가** (§5.2 후보 3, §5.4)
5. sm_120에서 **PDL 속성이 graph kernel node에 적용될 때의 동작** (§3.3)
6. `cudaLaunchAttributeCooperative`로 cooperative 커널을 **스트림 캡처하여 그래프 노드화**할 수 있는가 (§3.4.1)
7. 238 GPU0에 MPS/MIG가 걸려 있는가 (cooperative launch 제약과 직결)
8. `ApplyRodCusping`의 outer당 실제 발화율 (§5.2 후보 3의 escape 빈도 전제)
9. 배치 M64의 현재 `io_wall` 비중 재확인 (§3.4.12의 R-12 판정)
10. `--batch-mode 1`에서 rendezvous/linger 오버헤드의 크기 (§4.1의 2-A vs 2-B 선택)

## 부록 B — 코드 앵커 색인

| 주제 | 파일:행 |
|---|---|
| 기본 block size 256 | `src/CudaBICGBackend.cu:31` |
| device 내부 루프 제어 설계 주석 (**필독**) | `src/CudaBICGBackend.cu:136–220` |
| `HALT_GUARD` | `src/CudaBICGBackend.cu:221` |
| 결정론적 reduction 계약 (**필독**) | `src/CudaBICGBackend.cu:235–250` |
| `reduce_dot_stage1` chunk 파티션 | `src/CudaBICGBackend.cu:266–286` |
| stage2 strict fold | `src/CudaBICGBackend.cu:300` |
| 색 sweep (융합 금지 사유) | `src/CudaBICGBackend.cu:2464–2479` |
| `enqueue_iteration` (22 노드) | `src/CudaBICGBackend.cu:2485` |
| `enqueue_outer` (95 노드) | `src/CudaBICGBackend.cu:2555` |
| `launch_outer` 그래프 캡처·폴백 | `src/CudaBICGBackend.cu:2629–2697` |
| `enqueue_sweeps` (**Stage 1 범위**) | `src/CudaBICGBackend.cu:2708–2736` |
| `drain` (스트림 드레인 1회) | `src/CudaBICGBackend.cu:2989` |
| grid 헬퍼 | `src/CudaBICGBackend.cu:2205, 2246–2255` |
| 아레나 생성 / 폭 | `src/CudaBICGBackend.cu:3687–3706` |
| rendezvous / arrival gap | `src/CudaBICGBackend.cu:3481–3560` |
| `driveSweepsCuda` 아레나 요구 (**결론 1**) | `src/BICGSolver.cpp:336` |
| `canUseDeviceAssembly` 아레나 요구 | `src/BICGCMFD.cpp:208–217` |
| `driveDeviceSweeps` | `src/BICGCMFD.cpp:289` |
| 호스트 drive 루프 | `src/BICGCMFD.cpp:463–617` |
| `wiel` / `updls` | `src/BICGCMFD.cpp:129, 252` |
| `CMFD::upddhat` (**Stage 2**) | `src/CMFD.cpp:126–190` |
| `CMFD::setls` | `src/CMFD.cpp:205` |
| `CMFD::updjnet` | `src/CMFD.h:240` |
| assembly 커널 contraction 주석 (**Stage 2 필독**) | `src/CmfdAssemblyKernel.h:46–52` |
| `SolveLoop` outer 본문 | `src/Driver.h:1367–1700` |
| SPTELEM receipt | `src/Driver.h:2186–2222` |
| `EvaluateEquilibriumXenon` (host 전용 사유) | `src/XSSet.cpp:3770–3920` |
| `CommitXenon` | `src/XSSet.cpp:3922` |
| CUDA separable compilation (`-rdc`) | `CMakeLists.txt:89` |
