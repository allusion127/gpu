# WP7 단계 B — CMFD graph node census와 안전한 fusion

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `RASBERY_GPU_CMFD_SWEEP` sweep graph, BiCGSTAB outer graph |
| 상위 계획 | `docs/GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` WP7 단계 B, §3.4, §6 |
| 게이트 등급 | **B0** (비트 동일, 실행 위치·발사만 변경) |
| 플래그 | `RASBERY_GPU_CMFD_FUSE=<bitmask>`, 기본 `0` |
| receipt | `[RASBERY][CMFD][GRAPH]` |
| 계약 테스트 | `tools/test_cmfd_fuse_contract.py` (순수 python) |
| 소스 | `src/CudaBICGBackend.cu` |
| 기준 덱 | KNGR, `nmax=3`, `rb_sweeps=4`, `ncolors=2`, scalar fusion ON(기본), FP64 inner, chunked Wielandt fold |

> 이 문서의 모든 node 수는 **모델이 아니라 계약**이다. `tools/test_cmfd_fuse_contract.py`가
> `enqueue_iteration` / `enqueue_outer` / `enqueue_sweeps`의 launch 개수를 소스에서 직접 세고,
> 아래 표의 숫자를 그 구조로부터 재계산해 대조한다. 커널이 하나 추가되면 테스트가 실패하고
> 이 문서를 함께 고쳐야 한다. 즉 census는 늙지 않는다.

---

## 1. 왜 node 수인가

§3.4 단일 GPU Amdahl 표는 `colored_block_sweep` 23.7 %(2.23 µs × 701k launch), reduce-dot 계열
15.5 %를 지목한다. 두 항목 모두 **한 커널의 산술이 아니라 커널의 개수**가 지배한다. 실측 근거는
소스에 이미 있다: 17,000 미지수 계에서 captured outer 117 node의 산술은 약 10 µs인데 launch는
약 449 µs였다. 따라서 레버는 node 수이고, node 수를 낮추는 유일한 안전한 방법은 **같은 index
domain 위에서 인접한 두 커널의 본문을 이어 붙이는 것**이다.

단계 B는 그것을 한 비트씩 켤 수 있는 형태로 구현한다. 기준 커널은 지워지지 않고 mask 0이 그대로
발사하므로, B0 replay는 기억이 아니라 **살아 있는 reference**와 비교된다.

---

## 2. Census — 두 graph의 node 목록

### 2.1 BiCGSTAB outer graph (`launch_outer` / `enqueue_outer`)

`ScopedStreamCapture(stream, "cmfd.outer")`로 잡히는 graph. 한 번의 replay = **CMFD outer 1회**.

| # | node | 종류 | 개수 | 비고 |
|---|---|---|---:|---|
| P1 | `initialize_solver_state` | kernel | 1 | `full_scalar_grid()`. per-iteration `cudaMemsetAsync(iter_flags)` node를 흡수한 자리 |
| P2 | `refresh_operator_mirror_f32` | kernel | 0 또는 1 | FP32 arm에서만. 기본 arm에는 없음 |
| P3 | `begin_outer_fused` | kernel | 1 | block inversion + A·phi + 초기 잔차 (과거 3 node) |
| P4 | `reduce_dot_stage1` | kernel | 1 | 초기 잔차 norm, stage 1 |
| P5 | `reduce_norm_store_reference_stage2` | kernel | 1 | scalar fusion ON. OFF이면 `reduce_dot_stage2` + `store_reference_norm` = 2 |
| **prologue 소계** | | | **4** | scalar fusion OFF: 5 |
| I1 | `reduce_dot_stage1` (ρ_new) | kernel | 1 | `dot(r0, r)` |
| I2 | `reduce_dot_stage2` (ρ_new) | kernel | 1 | |
| I3 | `prepare_p_jacobi` | kernel | 1 | β 계산 + block Jacobi 대각 해를 흡수 |
| I4 | `colored_block_sweep` × `rb_sweeps` | kernel | 4 | **fusion 불가**, §4.1 |
| I5 | `matvec_two_group` | kernel | 1 | |
| I6 | `reduce_dot_stage1` (r0·v) | kernel | 1 | `dot(r0, v)` |
| I7 | `reduce_dot_stage2` (r0·v) | kernel | 1 | |
| I8 | `update_s_jacobi` | kernel | 1 | α 계산 + block Jacobi 대각 해를 흡수 |
| I9 | `colored_block_sweep` × `rb_sweeps` | kernel | 4 | **fusion 불가** |
| I10 | `matvec_two_group` | kernel | 1 | |
| I11 | `reduce_dot2_stage1` | kernel | 1 | (s·t, t·t) 한 쌍 |
| I12 | `reduce_dot2_stage2` | kernel | 1 | |
| I13 | `update_solution` | kernel | 1 | ω 계산을 흡수 |
| I14 | `reduce_dot_stage1` (잔차 norm) | kernel | 1 | |
| I15 | `reduce_norm_accumulate_stage2` | kernel | 1 | scalar fusion ON. OFF이면 2 |
| **iteration 소계** | | | **21** | scalar fusion OFF: 22 |
| E1 | `finalize_status` | kernel | 1 | `full_scalar_grid()` |
| E2 | `host_status` D2H | **memcpy** | 1 | §3.2 |
| **epilogue 소계** | | | **2** | |

`captured = 1 + nmax = 4`이므로

```
outer nodes = 4 + 4 x 21 + 2 = 90        (kernel 89, memcpy 1, memset 0)
```

scalar fusion을 끄면 `5 + 4 x 22 + 2 = 95`이고, 이는 소스의 fusion 원장이 기록한 값(“outer 117 → 95”)과
정확히 일치한다. 즉 이 census의 계수 방식은 이미 한 번 실측으로 교차 검증되어 있다.

### 2.2 Sweep graph (`launch_sweeps` / `enqueue_sweeps`)

`ScopedStreamCapture(stream, "cmfd.sweep")`. 한 번의 replay = **최대 `depth`개의 Wielandt sweep**.

| # | node | 종류 | 개수 | 비고 |
|---|---|---|---:|---|
| A1 | `cmfd_assemble_operator_2g` | kernel | 1 | **launch당 1회**, sweep당이 아님 |
| S1 | `cmfd_sweep_begin` | kernel | 1 | sweep당 |
| S2 | `cmfd_src_build` | kernel | 1 | sweep당 |
| S3 | *outer graph 본문* | — | 90 | §2.1 전체가 여기에 인라인된다 |
| S4 | `cmfd_wiel_terms` | kernel | 1 | |
| S5 | `cmfd_wiel_stage1` | kernel | 1 | chunked fold일 때만 |
| S6 | `cmfd_wiel_finalize_chunked` | kernel | 1 | serial fold이면 `cmfd_wiel_finalize` 1개 |
| S7 | `cmfd_updls` | kernel | 1 | |
| S8 | `cmfd_negative_scan` | kernel | 1 | |
| S9 | `cmfd_sweep_end` | kernel | 1 | |
| **sweep 1회 소계** | | | **98** | serial fold: 97 |

```
sweep graph nodes = 1 + depth x 98
nodes_per_sweep   = (nodes - 1) / depth = 98
```

한 sweep은 정확히 하나의 outer를 품으므로, 두 graph 모두에서
`launches_per_outer == nodes_per_sweep`이다. 이것이 receipt의 두 필드가 같은 값을 갖는 이유다.

### 2.3 receipt `[RASBERY][CMFD][GRAPH]`

graph instantiation 지점(`launch_outer`, `launch_sweeps`의 `cudaGraphInstantiate` 성공 직후)에서
graph당 1회 출력된다. instantiation은 구조적으로 드물다(bucket ladder 9개 × precision 2개가 상한이고,
정상 실행에서 `graph_reinstantiations`는 0). 따라서 실행당 몇 줄이다.

```json
[RASBERY][CMFD][GRAPH] {"graph":"sweep","nodes":393,"sweeps":4,
 "iterations_per_outer":4,"nodes_per_sweep":98,"kernel_nodes":389,
 "memcpy_nodes":4,"memset_nodes":0,"other_nodes":0,
 "launches_per_outer":98,"fuse_mask":0}
```

| 필드 | 의미 |
|---|---|
| `graph` | `"outer"` 또는 `"sweep"` |
| `nodes` | instantiate된 graph의 전체 node 수 (`cudaGraphGetNodes` 실측) |
| `sweeps` | 이 capture가 나르는 sweep 수. outer graph는 1 |
| `iterations_per_outer` | capture가 나르는 BiCGSTAB iteration 수 (`iter_batch_used`) |
| `nodes_per_sweep` | `"sweep"`: `(nodes - 1) / sweeps` (A1은 launch당 1회). `"outer"`: `nodes` |
| `kernel_nodes` / `memcpy_nodes` / `memset_nodes` / `other_nodes` | `cudaGraphNodeGetType` 실측 분류 |
| `launches_per_outer` | CMFD outer 1회가 치르는 dispatch 수 = `nodes_per_sweep` |
| `fuse_mask` | 이 실행의 `RASBERY_GPU_CMFD_FUSE` |

**궤적 중립성.** `cudaGraphGetNodes` / `cudaGraphNodeGetType`은 이미 만들어진 `cudaGraph_t`에 대한
host-side 질의다. enqueue도, sync도, device memory 읽기도, graph 수정도 없다. 질의 실패는 오류를
지우고 조용히 넘어간다(solver를 죽이지 않는다). 계약 테스트가 이 emitter 안에 `cudaMemcpy` /
`Synchronize` / `<<<`가 들어오는 것을 막는다.

---

## 3. Transfer census — 어떤 memcpy가 무엇인가

### 3.1 두 graph 안에 H2D node는 **하나도 없다**

이는 census의 가장 중요한 발견이다. 계획서가 지목한 `cudaMemcpyAsync` 117,829회는 **graph 밖**,
즉 `issueUploads` / `issueSweepUploads`(launch 앞)와 `issueFluxDownloads` / `issueSweepDownloads`
(launch 뒤)에 있다. graph 안의 유일한 전송 node는 outer당 1개의 D2H(§3.2)다.

launch 1회(`slot_budget` 1개 slot)당 stream에 올라가는 H2D 목록:

| 대상 | 크기 | 성격 | 이미 elide되는가 |
|---|---|---|---|
| `device_active` | `slots` × u32 | 참여 mask | 아니오 (매 launch) |
| `device_assembly_active` | `slots` × u32 | assembly mask | 아니오 |
| `sweep_halt` | `slots` × u32 | sweep mask | 아니오 |
| `xs_chif`, `node_vol` | n, nxyz double | 물리 배열 | `ByteExactMirror`로 대개 skip |
| `xs_xsnf`/`xs_xsrf`/`xs_xssm`/`dtil` | 물리 배열 | assembly arm | mirror로 skip |
| `dhat` | surface×group | | `dhat_resident`면 skip |
| `psi` | nxyz double | | `psi_resident` / `psi_dirty`면 skip |
| `phi` | n double | | mirror로 skip |
| **`scalars[m][kSweepFirst..]`** | `kSweepCount` double | **scalar 블록** | 아니오 (매 launch) |
| `scalars[m][kEps]` | 1 double | scalar | 값이 그대로면 skip (`eps_on_device`) |

**“scalar H2D를 device-side write로 바꿀 수 있는가”에 대한 답은 이미 코드 안에 있다.**
`cmfd_sweep_patch`가 정확히 그 일을 한다: staged 블록을 그대로 올린 뒤, 그 중 이전 outer의
observation이 만들어냈을 네 워드(`kEigv`, `kReigv`, `kReigvs`, `kErrl2`)를 device probe에서
덮어쓴다. 나머지 워드(`epsl2`, `eshift`, 각종 budget, 배열 포인터)는 **segment 전체에서 상수**이므로
H2D를 없애려면 “상수를 매 launch 올리지 않는다”는 별개의 mirror 작업이 필요하고, 그것은
node 수가 아니라 PCIe 트래픽 문제다(단계 B 범위 밖, §4.4 후보 D).

### 3.2 D2H — 지금 호스트 관측 전용인 것

| D2H | 위치 | 크기 | host-free outer/WHILE 이후의 실제 독자 |
|---|---|---|---|
| `host_status` ← `device_status` | **graph 안**, outer당 1 | `slots` × `DeviceSolveStatus` | `absorb()`뿐. host-free segment arm에서 `absorb`는 **segment당 1회**만 호출되므로(`finishSweepsDeferred`), 그 사이 outer들이 쓴 바이트는 **아무도 읽지 않는다**. 마지막 launch의 `NONFINITE_DETECTED`만 의미가 있고, 누적 tally 4개는 under-report된다(코드가 그렇게 명시하고 `hostfree_segments` receipt가 그 사실을 실는다) |
| `sl.sweep_out` ← 스윕 scalar 블록 | graph 밖, launch당 1 | `kSweepCount` double | `readSweepObservation`. host-free segment에서는 **읽지 않는다**(마지막 launch가 masked일 수 있어 읽으면 실행되지 않은 drive를 넘겨주게 된다) |
| `sl.out_phi` ← `phi` | graph 밖, launch당 1 | n double | 실제 결과. 필수 |
| psi | **삭제됨** | — | 과거 매 launch D2H였으나 유일한 독자가 degenerate-gamma 경로뿐이어서 예외 경로로 옮겨짐 |

즉 **outer당 D2H node 1개는 host-free arm에서 순수 관측용**이다. 이것을 제거하면 outer당 1 node와
`slots × sizeof(DeviceSolveStatus)` 바이트가 사라지지만, receipt의 네 tally 값이 바뀐다.
telemetry 값의 변화는 h5 궤적과 무관하지만 **receipt 비교 게이트를 깨뜨리므로 단계 B에서는
후보로만 기록하고 구현하지 않는다**(§4.4 후보 C).

---

## 4. Fusion 후보 판정

### 4.1 채택 — `RASBERY_GPU_CMFD_FUSE` 비트

| bit | 값 | fused kernel | 대체하는 reference | 절감 | 소스 |
|---:|---:|---|---|---|---|
| 0 | 1 | `reduce_dot_fused` | `reduce_dot_stage1` + `reduce_dot_stage2` | iteration당 **2 node** | `src/CudaBICGBackend.cu` |
| 1 | 2 | `reduce_dot2_fused` | `reduce_dot2_stage1` + `reduce_dot2_stage2` | iteration당 **1 node** | 〃 |
| 2 | 4 | `cmfd_wiel_fused` | `cmfd_wiel_stage1` + `cmfd_wiel_finalize_chunked` | sweep당 **1 node** | 〃 |
| 3 | 8 | `cmfd_sweep_gate_patch` | `cmfd_sweep_gate` + `cmfd_sweep_patch` | drive당 **1 launch** (graph 밖) | 〃 |

각 fused kernel 바로 위에 **ORDER-PRESERVATION NOTE**가 붙어 있다(계약 테스트가 강제한다).
핵심 논거는 네 비트에 공통이다.

1. **stage 1은 재작성이 아니라 복사다.** partition(`chunk`, `(n, gridDim.x)`의 순수 함수),
   thread별 stride walk, shared 배열, 고정 이진 트리, 그리고 `sum += am[i] * bm[i]`라는
   **표현식 자체**가 문자 단위로 같다. nvcc는 기본 `--fmad=true`에서 `x + a*b`를 FMA로 축약하므로,
   표현식을 “다시 타이핑”하면 반올림이 바뀔 수 있다. 그래서 복사한다.
2. **stage 2도 복사다.** `for (i = 0; i < blocks; ++i) sum += pm[i]` — 같은 partial 행에 대한
   **엄격한 오름차순 index 순서**, 한 thread. 어느 block이 그 fold를 실행하는지는 결과와 무관하다.
3. **둘 사이의 barrier는 여전히 barrier다.** kernel 경계가 주던 “모든 partial이 쓰였다”는 보증을
   retire counter가 대신한다: block의 thread 0이 partial을 쓰고 `__threadfence()` 후에야
   `atomicInc`에 참여하고, fold는 **마지막 티켓을 뽑은 block**에서만 돈다. 읽기는 volatile
   포인터를 통하므로 컴파일러가 자기 block의 partial을 레지스터에서 되돌려주지 못한다.
4. **guard가 서로 어긋날 수 없다.** 두 reference 모두 `RASBERY_CMFD_SLOT` + halt guard로 시작하고
   **그 사이에 도는 커널이 없으므로** `halt[m]`은 쌍 전체에서 불변이다. halt면 어떤 block도
   counter를 건드리지 않고 fold가 돌지 않으며 scalar도 쓰이지 않는다 — stage 2의 halt guard가
   하던 일 그대로다.
5. **counter는 스스로 재장전된다.** `atomicInc(retire+m, gridDim.x-1)`가 마지막 티켓을 0으로 감싼다.
   할당 시 한 번 0으로 두면 memset node가 필요 없고, halt된 launch는 0을 그대로 남긴다.
   fused kernel들이 **stream-ordered**(captured graph는 선형 체인)이므로 배열 하나면 충분하다.

비트별 추가 논거:

- **bit 1**: 두 reduction은 여전히 두 개다. 각자 accumulator, shared 배열, partial 행, 오름차순
  fold를 유지한다. retire counter를 공유하는 이유는 두 합이 섞이기 때문이 아니라 **같은 launch의
  같은 block들이 함께 retire하기 때문**이다.
- **bit 2**: tail이 한 thread가 아니다(3개 lane fold 후 lane 0이 `cmfd_wiel_apply`). 그래서
  `sh_last`를 thread 0이 쓰고 모든 thread가 지나는 `__syncthreads()` 뒤에 읽는다 — 분기가
  **block-uniform**이고, 이는 compaction guard가 의존하는 바로 그 성질이다. tail이 `sweep_halt`를
  쓰지만, 모든 block은 `atomicInc` 이전에 이미 `sweep_halt`를 읽었으므로 읽기가 쓰기보다 엄격히
  앞선다. reference는 32-thread block, fused는 `kReduceThreads` block에서 tail을 돌리지만
  산술을 하는 lane은 양쪽 모두 0..2뿐이다.
- **bit 3**: 두 reference 모두 `<<<1,1>>>` 한 thread 커널이고 서로 **disjoint한 메모리**
  (`sweep_halt[m]` vs 이 slot의 scalar 블록)를 쓴다. 한 thread가 gate 본문 다음 patch 본문을
  실행하는 것이 곧 두 node의 stream 순서다. `__ddiv_rn`을 포함해 산술은 그대로 복사했다.
  발사 여부 결정도 복사했다: `outer_halt`가 null이고 `patch_from_probe`가 false이면 — reference
  쌍이 아무 node도 올리지 않던 유일한 경우 — fused 경로도 아무것도 올리지 않는다.

### 4.2 거부 — `colored_block_sweep`의 colour pass 병합 (**NOT FUSABLE**)

단일 GPU 시간의 23.7 %가 여기 있고 outer graph node 90개 중 **32개**(`2 × rb_sweeps × captured`)가
여기 있으므로 가장 먼저 검토했다. 결론은 **불가**이며 이유는 물리적이다.

- sweep `k+1`은 **이웃 node의 `x`**를 읽는다. 따라서 sweep `k`가 **격자 전체에 대해** 쓴 값에
  의존한다. kernel 경계가 곧 그 barrier이고, colour 순서가 곧 Gauss–Seidel 의미론이다.
- 두 colour를 한 kernel에 넣으려면 그 사이에 **device-wide barrier**가 필요하다. 즉
  cooperative launch + `grid.sync()`. 이는 “같은 loop, 같은 `i`”라는 국소적 논거가 아니라
  **점유율과 launch 형태를 바꾸는** 전혀 다른 논거이며, 블록 수가 SM에 상주 가능한 수로 제한되어
  현재 grid 형상 자체가 달라진다. B0 등급으로 주장할 수 없다.
- reduce-fusion 계열과 달리 여기서는 `x`의 **읽기 시점**이 바뀐다. 즉 부동소수점 연산의 순서가
  아니라 **연산의 피연산자**가 바뀐다. 이는 정의상 B0가 아니다.

계약 테스트가 어떤 fused kernel 본문에도 `colored_block_sweep`이 들어오지 못하게 막고, 소스에
`NOT FUSABLE` 근거 문구가 남아 있는지 확인한다.

### 4.3 거부 — reduction을 통과하는 나머지 scalar 커널

- **iteration 말미의 `reduce_dot_stage1` + `reduce_norm_accumulate_stage2`**, 그리고
  **prologue의 `reduce_dot_stage1` + `reduce_norm_store_reference_stage2`**.
  두 stage-2는 이미 다른 커널의 본문을 흡수한 상태이고, **`active`를 보는 guard**를 쓴다.
  stage 1은 `halt`를 본다. 두 guard의 차이가 바로 over-run telemetry가 사는 자리이므로,
  fusion은 두 guard를 하나로 접어 counter의 의미를 바꾼다. **의미 손실이 있는 fusion은 거부한다.**
- **alpha/beta/omega scalar 갱신 커널**은 애초에 존재하지 않는다. 세 값 모두 이미
  이를 소비하는 커널(`prepare_p_jacobi`, `update_s_jacobi`, `update_solution`) 안에서
  계산된다. 계획서가 지목한 fusion은 **이미 landed**이며, census는 그 사실을 확인했다.
- **FP32 twin**(`reduce_dot_stage1_f32`, `reduce_dot2_stage1_f32`)은 건드리지 않는다.
  mixed-precision은 별도 opt-in arm이고, 두 게이트를 독립으로 두어야 어느 A/B도 상대의 분산을
  떠안지 않는다.

### 4.4 기록만 하고 구현하지 않은 후보

| 후보 | 절감 | 왜 지금은 아닌가 |
|---|---|---|
| C. outer당 `host_status` D2H node 제거 (host-free arm) | outer당 1 node + D2H 1회 | 결과 h5는 불변이지만 `[BACKEND_COUNTERS]`의 4개 tally가 바뀐다. receipt before/after 비교 게이트를 스스로 깨는 변경이므로 별도 작업으로 분리 |
| D. sweep scalar 블록 H2D의 상수부 mirror | launch당 H2D 1회의 대부분 | node 수가 아니라 PCIe 문제. `cmfd_sweep_patch`가 이미 “가변 4워드만 device-side write” 선례를 만들어 두었다 |
| E. `cmfd_sweep_begin` + `cmfd_src_build` 병합 | sweep당 1 node | `cmfd_sweep_begin`은 scalar grid(1 thread/slot), `cmfd_src_build`는 node grid. **index domain이 다르다.** 병합하려면 후자의 grid에서 전자를 slot당 1 thread로 조건 실행해야 하고, 그러면 begin의 쓰기가 src_build의 읽기보다 먼저라는 보증이 사라진다 |
| F. `cmfd_negative_scan` + `cmfd_sweep_end` 병합 | sweep당 1 node | scan은 `atomicAdd`로 `kNegative`를 누적하고 end는 그 총합을 읽는다. grid 전체 누적이 끝나야 하므로 §4.1의 retire 논거가 다시 필요하다. 이득 1 node로는 유지보수 비용을 넘지 못한다(계획서의 3 % 규칙) |

---

## 5. 기대 절감 (기준 덱, `nmax=3`, `rb_sweeps=4`, scalar fusion ON)

| `RASBERY_GPU_CMFD_FUSE` | outer graph nodes | sweep 1회 nodes | outer 대비 | sweep 대비 |
|---:|---:|---:|---:|---:|
| `0` (기본, reference) | **90** | **98** | — | — |
| `1` (dot) | 82 | 90 | −8.9 % | −8.2 % |
| `2` (dot2) | 86 | 94 | −4.4 % | −4.1 % |
| `4` (wiel) | 90 | 97 | 0 % | −1.0 % |
| `8` (sweep preamble) | 90 | 98 | 0 % | 0 % (graph 밖 launch −1/drive) |
| `3` (dot\|dot2) | **78** | 86 | −13.3 % | −12.2 % |
| `7` (dot\|dot2\|wiel) | 78 | **85** | −13.3 % | **−13.3 %** |
| `15` (전부) | 78 | 85 | −13.3 % | −13.3 % + launch −1/drive |

참고로 scalar fusion을 끄면 reference outer는 **95** node이며, 이는 소스의 fusion 원장이 기록한
“outer 117 → 95”와 정확히 일치한다 — census 계수 방식의 교차 검증이다.

`nodes_per_sweep` 98 → 85는 sweep당 dispatch 13개 감소다. W0가 측정한 node당 0.78 µs를 곱하면
sweep당 약 10 µs, 4,382 outer이면 약 44 ms — **이것만으로는 3 % 규칙을 통과하지 못한다.**
단계 B의 채택 판단은 반드시 238의 실측 wall로 해야 하며, 이 표는 “무엇이 얼마나 줄어드는가”만
말한다. 이득이 3 % 미만이면 계획서 §6.4에 따라 기본값을 바꾸지 않는다.

---

## 6. 서버 238 runbook

전제: GPU0 고정(`CUDA_VISIBLE_DEVICES=0`, 자식 프로세스까지), 첫 실행은 warm-up으로 폐기,
telemetry 실행과 wall 실행을 분리, arm을 교대로 실행해 clock drift를 상쇄(§6.4).

### 6.0 컴파일 (여기서는 할 수 없는 일)

`src/CudaBICGBackend.cu`는 nvcc로만 검증된다. 이 작업의 로컬 환경에는 CUDA toolkit이 없으므로
**컴파일은 238에서 처음 일어난다.** 새로 추가된 것은 `__global__` 4개, host 함수 2개
(`cmfdFuseMask`, `reportCmfdGraphCensus`), 멤버 6개(`fuse_retire` + 5개 플래그)다. 실패한다면
CUDA API 시그니처(`cudaGraphGetNodes` / `cudaGraphNodeGetType` / `atomicInc`) 쪽일 가능성이 높다.

### 6.1 단일 production arm

한 arm = `RASBERY_GPU_CMFD_FUSE=<mask>`만 다르고 나머지 env는 완전히 동일.
arm 목록: `0`(reference), `1`, `2`, `4`, `8`, `15`.

```sh
# 각 arm마다
export CUDA_VISIBLE_DEVICES=0
export RASBERY_GPU_CMFD_FUSE=<mask>
# 나머지 env는 production 단일 arm과 동일 (A2 포함 여부를 arm 간에 섞지 말 것)
```

### 6.2 B0 판정 (mask마다 전부 통과해야 한다)

| # | 검사 | 통과 기준 |
|---|---|---|
| B0-1 | `h5diff -c` mask arm vs `FUSE=0` arm | **0/644 데이터셋 차이** |
| B0-2 | `[TRAJECTORY]` digest | `0d15abf29d222a02`, outer `4382`, mask와 무관하게 동일 |
| B0-3 | ON×2 결정론 | 같은 mask 2회 실행이 서로 byte 동일 |
| B0-4 | `ctest` | 전부 통과 (`FUSE` 미설정 = 기본 경로) |
| B0-5 | `python tools/test_cmfd_fuse_contract.py` | PASS |
| B0-6 | `[RASBERY][CMFD][GRAPH]` before/after | `nodes_per_sweep`, `launches_per_outer`가 §5 표와 일치. `fuse_mask`가 요청한 값과 일치 |
| B0-7 | `[RASBERY][CUDA][BACKEND_COUNTERS]`, `[CMFD][COMPACT]` | mask 간 동일 (fusion은 telemetry를 바꾸지 않는다) |

B0-1 또는 B0-2가 어긋나면 **그 비트는 즉시 폐기**한다. 근사는 없다.

### 6.3 성능 측정

| # | 측정 | 방법 | 기준 |
|---|---|---|---|
| P-1 | 단일 wall | hot median of 3 (warm-up 1회 폐기), arm 교대 | reference **16.9 s** 대비 |
| P-2 | 배치 처리량 | `8 x M8 + MPS`, c/h | reference **878 c/h** 대비. sweep 커널은 단일/배치가 공유하므로 단일이 좋아지면 여기도 움직여야 한다 |
| P-3 | `nsys` `cuda_api_sum` | `cudaGraphLaunch` 횟수 불변, `cudaLaunchKernel` 감소 | graph 밖 launch는 bit 3에서만 −1/drive |
| P-4 | `nsys` top kernels | `reduce_dot_stage2` / `reduce_dot2_stage2` / `cmfd_wiel_finalize_chunked` 항목이 사라지고 fused 항목이 등장 | 총 kernel time이 늘지 않을 것 |
| P-5 | GPU SM / H2D·D2H / graph count | §6.4의 동반 지표 | 회귀 없음 |

채택 규칙: **단일 wall 3 % 이상 개선**이 아니면 기본값을 바꾸지 않는다(계획서 §6.4, WP7 단계 B의
“3 % 미만 fusion은 유지보수 비용 때문에 채택하지 않는다”). bit 8은 graph node를 바꾸지 않으므로
단독으로는 거의 확실히 3 % 미만이며, `15`에 묶어서만 의미가 있다.

### 6.4 실패 시 회수 절차

1. `FUSE=0`으로 되돌린다 — 기준 커널이 그대로 있으므로 코드 변경 없이 즉시 reference로 복귀한다.
2. 실패한 비트의 ORDER-PRESERVATION NOTE를 다시 읽고, 어느 항목(1~5)이 깨졌는지 특정한다.
   `h5diff`가 어긋났다면 거의 확실히 항목 3(barrier) 또는 항목 4(guard)다.
3. 비트를 되살리지 못하면 표에서 지우고 fused kernel을 제거한다. **비트를 남긴 채 “보통은 맞다”로
   두지 않는다.**

---

## 7. 변경 요약

| 파일 | 변경 |
|---|---|
| `src/CudaBICGBackend.cu` | `CmfdFuseBit` / `cmfdFuseMask()`; `reduce_dot_fused`, `reduce_dot2_fused`, `cmfd_wiel_fused`, `cmfd_sweep_gate_patch`; `reportCmfdGraphCensus()`와 두 instantiation 지점의 emission; `fuse_retire` 배열; `dot()` / `dot2()` / `enqueue_sweeps()` / `enqueueSweepPreamble()` 분기 |
| `tools/test_cmfd_fuse_contract.py` | 신규. 플래그 계약, order note, reference 생존, colour sweep 거부, receipt 궤적 중립성, 구조-모델 대조, runbook 게이트, negative control 9종 |
| `docs/WP7_CMFD_GRAPH_CENSUS_20260831_KO.md` | 본 문서 |
