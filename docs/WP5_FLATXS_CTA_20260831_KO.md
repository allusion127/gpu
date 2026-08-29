# WP5 단계 A–C — FlatXS CTA-per-node 커널

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `RASBERY_GPU_FLATXS` arm의 `kernelFlatXs` (unrodded flat-XS node update) |
| 상위 계획 | `docs/GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` WP5 단계 A/B/C, §6 |
| 게이트 등급 | **B0** — 실행 위치만 바뀐다. 부동소수 연산은 하나도 바뀌지 않는다 |
| 플래그 | `RASBERY_GPU_FLATXS_CTA=1`, 기본 `0` / `RASBERY_GPU_FLATXS_CTA_THREADS`(64·128·256, 기본 128) |
| 계약 테스트 | `tools/test_flatxs_cta_contract.py` (순수 python, negative control 13종) |
| 단계 A 도구 | `tools/flatxs_resource_report.py` (`--selftest` 포함, negative control 4종) |
| replay 게이트 | `test/flatxs_device_replay.cu --cta [T]` |
| 소스 | `src/FlatXsCtaKernel.cuh`(신규), `src/CudaXsReconBackend.{h,cu}`, `src/CudaXsReconBackendStub.cpp`, `src/Driver.h`, `test/flatxs_device_replay.cu`, `CMakeLists.txt` |
| 기준 덱 | KNGR, `nxyz = 8,451`, `NG = 2`, `NISO = 39`, `N_ACTIVE = 9` |
| 미변경 | `src/FlatXsKernel.h`(공유 본문), `src/XsReconKernel.h`, `src/XSSet.cpp` — **한 줄도 건드리지 않았다** |

> **이 문서가 주장하지 않는 것부터.** WP5의 출발점인 “`kernelFlatXs` = 배치 GPU 시간의
> **39.9 %**, 평균 3.08 ms/launch”는 추적표 **R1**이 결함 하네스에서 나온 숫자라고
> 판정한 지분이다(`docs/WP_PLAN_REVIEW_AND_TRACKER_20260831_KO.md` §R1, 도착 폭 14.5로
> 굶던 실행). 그 지분은 **폭이 찬 실행(8×M8+MPS, `width_fill` 0.41)에서 다시 재기 전까지
> 인용 금지**다. 아래 §6.0이 그 재측정을 이 문서의 **첫 단계**로 못박는다. 커널이 빨라졌다는
> 말은 그 재측정 뒤에만 의미가 있다.

---

## 1. 단계 A — 자원 사용 증명

### 1.1 정적 가설: 작업공간이 정확히 무엇인가

`flatxsSolveNode()`(`src/FlatXsKernel.h:205`)는 한 노드의 전체 상태를 **스레드 지역 배열**에
펼쳐 놓고 작업한다. `kernelFlatXs`(`src/CudaXsReconBackend.cu:432`)는 그 본문을 **노드당 한
스레드**로 돌린다. 따라서 아래가 통째로 per-thread 저장소다.

| 배열 | 선언 | doubles | bytes | 무엇인가 |
|---|---|---:|---:|---|
| `bl` | `[N_ACTIVE * NG]` | 18 | 144 | lumped 스칼라 XS, 9개 XT 슬롯 × 2군 |
| `bls` | `[NLSM]` | 4 | 32 | lumped 산란 행렬 2×2 |
| `bm` | `[N_ACTIVE * NMIC]` | **702** | **5,616** | micro 스칼라 XS, 9 슬롯 × 39 동위원소 × 2군 |
| `bms` | `[NMSM]` | 156 | 1,248 | micro 산란, 39 × 2 × 2 |
| `iden` | `[NISO]` | 39 | 312 | 노드 동위원소 밀도(RefreshLightIsotopes 후) |
| `active_xt` | `int[N_ACTIVE]` | — | 36 | XT 슬롯 인덱스 표 |
| **합계** | | **919** | **7,388** | |

`tools/flatxs_resource_report.py`는 이 표를 **헤더에서 다시 계산한다.** 7,388을 문서에
적어 두고 동위원소 레지스트리가 39에서 움직이면 문서만 틀리게 되는 사고를 막기 위해서다
(`--selftest`가 그 유도를 검사한다).

`bm` 하나가 5.6 KiB로 전체의 76 %다. **여기가 병목 가설의 전부다**: 39 × 2 = 78개 micro
원소 × 9 슬롯을, 한 스레드가 delta stream 전체 동안 들고 있어야 한다.

### 1.2 산술 구조 — 무엇이 병렬화 가능하고 무엇이 아닌가

한 노드의 본문은 다섯 단계다.

1. **gather** — reference 상태를 작업공간으로 복사. 순수 복사, 산술 없음.
2. **delta stream 적용** — `node_off[i] .. node_off[i]+node_cnt[i]` 범위의 각 항목 `s`마다:
   - 호스트가 이미 풀어 놓은 `(did, x, scale)`을 읽고, `mode == 1`이면 knot 구간을 찾아
     `xloc`과 `base`를 정한다(정수 + 정확한 뺄셈 하나).
   - 네 개의 계수 표(`coeff_lmp[t]`, `coeff_lsm`, `coeff_mic[t]`, `coeff_msm`)에 대해
     **원소별 Horner**를 `p = nord-2 .. 0` 내림차순으로 돌리고, 결과를
     `dst[e] = ma(ACC, scale, val, dst[e])`로 누적한다.
   - **원소 e끼리는 완전히 독립이다.** 원소 하나의 사슬은 `s`에 대해 순차적이지만, 서로 다른
     원소의 사슬은 서로를 전혀 보지 않는다. → **여기가 병렬 축이다(878개 원소).**
3. **RefreshLightIsotopes** — H-1/B-10/O-16 세 행만 다시 쓴다. 순수 곱셈, 누적 없음.
4. **scatter** — 작업공간을 SoA 배열로 되돌린다. 순수 복사.
5. **macro XS 재구성** — **반드시 보존해야 하는 순서가 여기 있다.**

   ```cpp
   double val = bl[t * NG + ig];
   for (int iso = 0; iso < NISO; ++iso)                       // 0 → 38, 오름차순
       val = pol.ma(F_MACRO_SCAL, mt[iso * NG + ig], iden[iso], val);
   ```

   길이 39의 **의존 FMA 사슬**이다. `fma(a,b,c)`는 단일 반올림이므로 이 사슬은 어떤
   의미로도 결합법칙을 만족하지 않는다 — 트리 리덕션은 반올림 위치만 옮기는 것이 아니라,
   **참조 구현이 애초에 만들지 않는 부분곱을 합산한다.** 산란 쪽도 같은 형태다:

   ```cpp
   double val = bls[igs * NG + ige];
   for (int iso = 0; iso < NISO; ++iso)
       val = pol.ma(F_MACRO_SSM, bms[iso * NLSM + igs * NG + ige], iden[iso], val);
   ```

   **보존해야 하는 리덕션 순서 = 출력 원소마다 iso 0→38 좌→우 순차 fold.**
   독립적인 사슬은 스칼라 18개 + 산란 4개 = **22개**이고, 병렬화는 **사슬들 사이에서만**
   허용된다. 사슬 하나를 둘로 쪼개는 순간 B0는 끝난다.

   마지막 XSDF/XSRF 패스도 같다: `rf += xs_ssm[...]`를 `ige`에 대해 순차로 접는다(군당 사슬,
   `NG = 2`).

### 1.3 238에서 실제로 잴 것

정적 계산은 **가설이지 증거가 아니다.** ptxas와 Nsight Compute가 답한다.

```bash
# (a) 컴파일 타임 절반 — 장치 불필요
cmake -S . -B build-ptxas -DRASBERY_ENABLE_CUDA=ON -DRASBERY_PTXAS_VERBOSE=ON
cmake --build build-ptxas -j 2>&1 | tee "$OUT/build_ptxas.log"

# (b) 이미 빌드된 바이너리에서 되읽기
cuobjdump -res-usage "$BLD/RASBERY" > "$OUT/res_usage.txt"

# (c) 런타임 절반 — occupancy와 실제 local traffic
ncu --target-processes all --set full --csv \
    --kernel-name regex:kernelFlatXs \
    --metrics launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warps_eligible.avg.per_cycle_active,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_average_branch_targets_threads_uniform.pct,\
gpu__time_duration.sum \
    "$BLD/RASBERY" --jobs "$OUT/profile_jobs.txt" --batch-mode 4 --result light \
    > "$OUT/flatxs_ncu.csv"

# (d) 판정
python3 tools/flatxs_resource_report.py \
    --ptxas-log "$OUT/build_ptxas.log" \
    --res-usage "$OUT/res_usage.txt" \
    --ncu-csv   "$OUT/flatxs_ncu.csv" \
    --json-out  "$OUT/flatxs_stageA.json"
```

기록표(238이 채운다):

| 항목 | `kernelFlatXs` | `kernelFlatXsCta<128>` |
|---|---|---|
| registers/thread | | |
| stack frame (B/thread) | | |
| spill stores / loads (B) | | |
| static smem (B/CTA) | | (예상 7,352) |
| achieved occupancy (%) | | |
| eligible warps/scheduler | | |
| local ld/st bytes | | (예상 0) |
| branch uniform (%) | | |
| L1 / L2 hit (%) | | |
| duration (µs/launch) | | |

### 1.4 단계 A의 분기 — 이것은 형식이 아니다

`flatxs_resource_report.py`가 인쇄하는 판정이 그대로 분기다. 종료 코드로도 나온다.

| 판정 | 조건 | 다음 |
|---|---|---|
| `PROCEED_CTA` (exit 0) | local/stack ≥ 1 KiB/thread **또는** 실측 local traffic > 0 | 단계 B/C를 게이트에 올린다 |
| `REDESIGN` (exit 2) | spill 없음, local traffic 없음 | **CTA 커널을 채택하지 않는다.** delta stream H2D, 호출당 ~61 MB 다운로드, divergence, 직렬 isotope fold를 대상으로 다시 설계한다 |
| `INSUFFICIENT_DATA` (exit 3) | 아티팩트 부족 | 다시 측정 |

`REDESIGN`은 실패가 아니라 **결과다.** 코드가 이미 있다는 이유로 우회하지 않는다.

---

## 2. 단계 B — 무엇을 만들었는가

`src/FlatXsCtaKernel.cuh`. **CTA 하나가 노드 하나**를 맡고, 919 double 작업공간이
스레드 지역이 아니라 **블록 공유(static `__shared__`, 7,352 B/CTA)**에 산다. 128
스레드 기준으로 스레드당 작업공간이 7,388 B → **57 B**가 된다.

```cpp
template <int T>
__global__ __launch_bounds__(T) void kernelFlatXsCta(FlatXsView v) {
    const int i = static_cast<int>(blockIdx.x);   // 노드 = 블록
    if (i >= v.n_nodes) return;
    __shared__ CtaWorkspace w;                    // bl / bls / bm / bms / iden
    flatxsSolveNodeCta<T>(v, i, StaticForms{}, w);
}
```

`StaticForms`는 **참조 arm이 쓰는 바로 그 타입**이고 마스크도 같은 `FLATXS_FORMS = 0x3FF`다.
트리 어디에도 form의 두 번째 정의는 없다.

### 2.1 B0 논증 — 코드 옆에 적힌 세 성질

전문은 `src/FlatXsCtaKernel.cuh` 상단에 있다. 요지:

- **P1 고정 lane 소유권.** 모든 작업공간 원소에 flat ordinal `q`가 있고, gather·apply·
  scatter·macro 모든 단계가 **같은** `for (int q = tid; q < N; q += T)` 형태로 자기 ordinal
  공간을 걷는다. 따라서 원소 `q`를 gather한 lane이 `q`에 모든 delta를 적용하고 `q`를
  scatter한다. **원소 하나의 누적 사슬은 한 lane 안에서 끝난다.** 블록 크기 `T`를 바꾸면
  “어느 lane이 `q`를 맡는가”만 바뀌고 `q`의 사슬은 바뀌지 않는다 — 그래서 블록 크기가
  수치 노브가 아니라 성능 노브다.
- **P2 isotope fold는 병렬화하지 않는다.** §1.2의 22개 사슬을 **사슬 단위로** 나눠 갖고,
  각 lane이 자기 사슬을 iso 0→38 순차로 접는다. `__shfl` 없음, `atomicAdd` 없음, `cub::`
  없음, 트리 없음. “단일 스레드가 접는다”보다 강한 형태다 — 22개 사슬이 서로 독립이므로
  **직렬화 없이** 순서를 보존한다.
- **P3 delta 스칼라는 broadcast하지 않고 lane마다 재계산한다.** `did/x/scale`, `DeltaMeta`,
  mode-1 구간 탐색, `xloc`, `base` 전부 정수 연산 + 정확한 뺄셈 하나라 lane마다 bit
  동일하다. 그래서 **delta stream 루프 안에 barrier가 하나도 없다.** 계획의 의사코드는
  delta마다 `__syncthreads()`를 두었는데, P1이 성립하면 그 루프에는 lane 간 의존이 아예
  없으므로 stream 길이와 무관하게 barrier 0이다.

barrier는 딱 두 개이고, 둘 다 **다른 lane이 쓴 것을 읽기 직전**에 있다:

1. 작업공간 + `sh_iden` 공개 후 (macro 재구성이 남의 원소를 읽는다),
2. macro 저장 후 (XSDF/XSRF 패스가 참조 구현과 똑같이 **global에서** `xs[XSTF]`/`xs[XSAF]`/
   `xs_ssm`를 다시 읽는다 — `__syncthreads()`가 그 global store를 블록 내에 보이게 한다).

### 2.2 FMA 주의

이 파일에는 `a * b + c`가 한 군데도 없다. 모든 곱셈-덧셈은 참조와 **같은 FormBit**으로
`pol.ma(<bit>, ...)`를 통과하고, TU는 계속 `--fmad=false`로 빌드된다(`CMakeLists.txt`의
`RASBERY_BITEXACT_CUDA_OPTS`). `RASBERY_PTXAS_VERBOSE`는 그 목록에 **덧붙일 뿐** 대체하지
않는다 — 측정이 계약을 끄는 일은 없다. 계약 테스트가 참조 본문과 CTA 본문의 FormBit
**출현 횟수를 10개 비트 전부 대조**한다.

### 2.3 dispatch

`XsReconBackend::solveFlatXs()` 안 **한 곳**, 캐시된 bool 하나:

```cpp
static const bool cta = rasberyGpuFlatXsCtaEnabled();
if (cta) fxs::flatxsCtaLaunch(v, rasberyGpuFlatXsCtaThreads(), d.stream);
else     kernelFlatXs<<<grid, block, 0, d.stream>>>(v);
```

launch 아래의 어떤 것도 — 다운로드, residency generation, 카운터 — 어느 arm이 돌았는지
알지 못한다. 두 arm이 같은 바이트를 쓰기 때문이고, 계약 테스트가 그 무지를 검사한다.
`src/XSSet.cpp`와 `src/FlatXsKernel.h`는 **변경 없음**이다.

### 2.4 engagement receipt를 새로 만들지 않은 이유

`[RASBERY][FLATXS][GPU]`의 `nodes_solved`는 두 arm에서 같은 값이 나온다 —
그것이 B0의 정의다. arm 구분용 카운터를 새로 넣으면 추적표가 F9/R2에서 이미 한 번
정리한 “하나의 사실에 숫자가 둘” 문제를 다시 만든다. **CTA arm이 실제로 발사되었다는
증거는 §6.4의 nsys kernel-name census**다: `kernelFlatXsCta`가 나타나고 `kernelFlatXs`가
사라진다. 이름은 위조할 수 없고 카운터와 달리 배선이 필요 없다.

---

## 3. 단계 C — replay 게이트

`test/flatxs_device_replay.cu`에 `--cta [T]` 모드를 붙였다. 새 하네스를 만들지 않은 이유는
기존 것이 이미 capture 포맷·ULP 채점·노드 목록을 알고 있기 때문이다.

```bash
RASBERY_FLATXS_DUMP="$OUT/cap" RASBERY_GPU_FLATXS=1 "$BLD/RASBERY" <deck>   # capture
"$BLD/rasbery_flatxs_replay"        "$OUT/cap"          # 호스트 arm (기존)
"$BLD/rasbery_flatxs_device_replay" "$OUT/cap"          # 장치 참조 arm (기존)
"$BLD/rasbery_flatxs_device_replay" "$OUT/cap" --cta 64
"$BLD/rasbery_flatxs_device_replay" "$OUT/cap" --cta 128
"$BLD/rasbery_flatxs_device_replay" "$OUT/cap" --cta 256
```

`--cta`는 **참조 커널과 CTA 커널을 한 런에서** 돌린다. 두 arm은 mutable 배열
(`lmp/mic/lsm/msm/xs/xs_ssm/iden`)의 **각자의 복사본** 위에서 돌고 immutable 표만 공유한다.
그래서 세 판정이 한 번에 나온다:

| 판정 | 의미 | 합격 |
|---|---|---|
| `ref-vs-capture` | 장치 참조 arm이 gcc 출력과 같은가 (기존 게이트) | 0 mismatch |
| **`cta-vs-ref`** | **CTA arm이 참조 커널과 원소 단위로 같은가 — B0 주장 그 자체** | **0 mismatch** |
| `cta-vs-capture` | CTA arm이 gcc 출력과 같은가 | 0 mismatch |

`cta-vs-ref`가 더 날카롭다: capture가 어떤 원소에 둔감한 덱이어도 lane 소유권/barrier 버그를
잡는다. `--cta`는 세 판정이 전부 0일 때만 exit 0이다.

**세 블록 크기 전부 돌린다.** 64/128/256이 다른 바이트를 낸다면 P1이 깨진 것이고, 그것은
성능 문제가 아니라 정확성 문제다.

fixture 범위(계획 단계 C가 요구): unrodded/rodded 혼합, branch delta 있음/없음, spectral
history, Gd 모델. 각각에서 capture를 하나씩 뜬다.

---

## 4. 무엇이 달라져도 되고 무엇이 달라지면 안 되는가

| 달라져도 되는 것 | 이유 |
|---|---|
| 블록 크기 64/128/256 | P1에 의해 사슬이 바뀌지 않는다 |
| launch 수(노드당 스레드 → 노드당 CTA로 grid 형태가 바뀜) | 발사 형태는 B0가 허용한다 |
| registers/occupancy/smem | 단계 A가 재는 대상 그 자체 |

| 달라지면 안 되는 것 | 어디서 잡히나 |
|---|---|
| 출력 원소 단 하나의 bit | `--cta`의 `cta-vs-ref` |
| iso fold 방향/분할 | 계약 테스트 규칙 5 + negative control 2종 |
| lane 소유권 형태 | 계약 테스트 규칙 4 + negative control 2종 |
| FormBit 배정 | 계약 테스트 규칙 6 (참조와 횟수 대조) |
| `--fmad=false` | 계약 테스트 규칙 8 (`RASBERY_PTXAS_VERBOSE`는 append 전용) |
| 기본값 OFF | 계약 테스트 규칙 1 |

---

## 5. 남은 것 (단계 D는 착수하지 않았다)

계획의 **단계 D(XS canonical residency)**는 이 작업에 포함되지 않았다. `solveFlatXs()`는
여전히 호출당 약 61 MB를 host로 되돌리고 `RASBERY_FLATXS_SKIP_MICX_DL=1`은 consumer
audit 없이 위험하다고 코드가 직접 적어 두었다. 단계 D는 그 audit(`_micx/_lmpx`의 depletion /
rodded fallback / export / PPR consumer)이 먼저다. **단계 B의 이득과 단계 D의 이득은
독립이고, 단계 B가 실패해도 단계 D는 살아 있다.**

---

## 6. 238 runbook

로컬에 nvcc가 없다. **첫 관문은 238 컴파일이다.**

### 6.0 선행: R1 재측정 (이것이 먼저다)

```bash
export CUDA_VISIBLE_DEVICES=0
# 폭이 찬 실행에서 §3.3 커널 지분을 다시 잰다
nsys profile --trace=cuda -o "$OUT/wp5_share" \
    python3 tools/run_single_gpu_batch.py --procs-per-gpu 8 --batch-mode 8 --mps ...
```

기록: `kernelFlatXs`의 배치 GPU 시간 지분(%)과 launch당 평균 시간(µs).
계획의 **39.9 % / 3.08 ms**는 도착 폭 14.5로 굶던 실행의 숫자다(R1). `width_fill`이 0.41인
8×M8+MPS에서 지분이 얼마인지가 **WP5 전체의 전제**다.

- 재측정 지분이 여전히 최대 항목이면 §6.1로 간다.
- 지분이 크게 내려갔다면 **WP5는 우선순위에서 내려간다.** 커널은 남기고 기본값 OFF로 둔다.
  그것도 결과다.

### 6.1 단계 A 자원 증명

§1.3의 (a)~(d)를 그대로 실행하고 §1.3 표를 채운다. 판정이 `REDESIGN`이면 §6.2 이하를
실행하지 않는다.

### 6.2 replay B0 (성능보다 먼저)

§3의 다섯 줄. `--cta 64/128/256` 세 줄이 전부 `cta_vs_ref_mismatches=0
cta_vs_capture_mismatches=0 -> PASS`여야 한다. 하나라도 어긋나면 **성능은 재지 않는다.**

### 6.3 단일 production arm, CTA 0 vs 1 (B0 게이트)

```bash
export CUDA_VISIBLE_DEVICES=0
# A: 기준
RASBERY_GPU_FLATXS=1 RASBERY_GPU_FLATXS_CTA=0  <production arm> ...
# B: 전환
RASBERY_GPU_FLATXS=1 RASBERY_GPU_FLATXS_CTA=1  <production arm> ...
```

합격 조건 — **전부 만족해야 한다**:

1. `h5diff -c` 로 A vs B: **0 / 644 차이**
2. digest **`0d15abf29d222a02` / `4382`** 두 값 모두 A와 B가 동일
3. **ON×2 결정론**: B를 두 번 돌려 서로 bit 동일
4. `[RASBERY][FLATXS][GPU]`의 `nodes_solved`가 A와 B에서 동일
5. `[RASBERY][GPU_FULL]`의 `flatxs_fallbacks == 0`, `[BACKEND_COUNTERS]`의
   `xs_cpu_fallbacks`가 A와 B에서 동일
6. `RASBERY_GPU_FLATXS_CTA_THREADS=64` 와 `=256`에서도 1·2가 그대로 성립

1~6 중 하나라도 어긋나면 채택하지 않는다. B0 주장이 거짓인 arm의 wall은 의미가 없다.

### 6.4 nsys — 커널 시간과 engagement

```bash
nsys profile --trace=cuda -o "$OUT/wp5_cta0" <A 명령>
nsys profile --trace=cuda -o "$OUT/wp5_cta1" <B 명령>
```

기록표:

| 항목 | CTA=0 | CTA=1 (T=64) | CTA=1 (T=128) | CTA=1 (T=256) |
|---|---|---|---|---|
| kernel 이름 | `kernelFlatXs` | `kernelFlatXsCta` | `kernelFlatXsCta` | `kernelFlatXsCta` |
| launch 수 | | | | |
| 총 시간 (ms) | | | | |
| **launch당 (µs)** | | | | |
| 배치 GPU 시간 지분 (%) | | | | |

**engagement 증거는 첫 줄이다**: CTA=1에서 `kernelFlatXsCta`가 나타나고 `kernelFlatXs`가
사라져야 한다. 둘 다 보이면 dispatch가 한 곳이 아니라는 뜻이다(계약 테스트 규칙 2가
소스에서 같은 것을 본다).

### 6.5 배치 — 값이 나오는 곳

```
8×M8 + MPS,  CUDA_VISIBLE_DEVICES=0
```

기준선 **878 c/h**(WP4 실측, A2 정책). 함께 읽을 것: `width_fill`, `padding_fraction`,
`tail_idle_s`, CPU wait, GPU SM, H2D/D2H, `[CAPTURE_ARBITER]`의 `alloc_in_capture == 0`.
단일 wall은 **16.9 s** 기준으로 hot median of 3, A/B 교대로 확인만 하고 판정에 쓰지 않는다
(FlatXS의 값은 배치에서 난다).

---

## 7. 채택 조건

| 조건 | 판정 |
|---|---|
| §6.0 재측정에서 FlatXS 지분이 여전히 유의미 | **필수 전제.** 아니면 기본값 OFF 유지 |
| §6.1이 `PROCEED_CTA` | **필수.** `REDESIGN`이면 채택하지 않는다 |
| §6.2 replay 3개 블록 크기 전부 0 mismatch | **필수** |
| §6.3 1~6 전부 통과 | **필수** |
| 배치 c/h가 878 대비 **+3 % 이상** | 채택 (계획 §6.4의 노이즈 문턱) |
| 배치 이득 3 % 이하 | 기본값 유지, 코드는 남긴다(회귀 아님) |
| `kernelFlatXs` 시간 −30 % 이나 배치 이득 <3 % | **채택하지 않는다.** 커널이 병목이 아니었다는 증거로 기록하고 단계 D로 넘어간다 |

계획의 WP5 성능 게이트(커널 −30 %, M64 +10 %)는 그대로 두되, **채택의 실제 문턱은 배치
+3 %**다 — §6.4의 전역 규칙이 3 % 이하를 노이즈로 규정하므로, 그 아래에서 기본값을 바꾸는
것은 이 캠페인의 규칙 위반이다.

---

## 8. 계획과 달라진 점

| 계획 | 여기 | 이유 |
|---|---|---|
| `src/FlatXsCooperativeKernel.cuh`, `RASBERY_GPU_FLATXS_COOP` | `src/FlatXsCtaKernel.cuh`, `RASBERY_GPU_FLATXS_CTA` | 구조가 cooperative groups가 아니라 단순 CTA다. 이름이 쓰지 않는 API를 암시하면 안 된다 |
| delta마다 `__syncthreads()` | barrier 0 | P1이 성립하면 stream 루프에 lane 간 의존이 없다(§2.1). 계획이 우려한 “delta 수만큼 늘어나는 barrier”가 아예 발생하지 않으므로 delta count별 specialization도 불필요하다 |
| `test/flatxs_cooperative_replay.cu` 신규 | 기존 `test/flatxs_device_replay.cu`에 `--cta` | 기존 하네스가 capture 포맷·ULP 채점·노드 목록을 이미 안다. 두 번째 채점기는 두 번째 진실이다 |
| `tools/test_flatxs_cooperative_contract.py` | `tools/test_flatxs_cta_contract.py` | 플래그 이름과 일치 |
| `src/FlatXsKernel.h`, `src/XSSet.cpp` 수정 | **수정 없음** | 참조 arm은 손대지 않는 것이 B0의 전제다 |
| 단계 D | 미착수 | consumer audit이 먼저(§5) |
