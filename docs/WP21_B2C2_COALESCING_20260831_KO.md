# WP21-B2 / C2 — flatxs 스토어 전치와 nodal stride 뷰: **주소를 옮기지 않고 레인을 옮긴다**

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `kernelFlatXsCta`의 스토어 st 25.2 (B2) · `kNodalJnet<0>`의 로드 ld 16.7 (C2) |
| 선행 | `docs/WP21_BC_FLATXS_NODAL_COALESCING_20260831_KO.md` (인벤토리·미변환 귀속), `docs/WP21_A_CMFD_COALESCING_20260831_KO.md` |
| 근거 | 238 ncu, `E:\rasbery_runs\2026-08-30\238\pricing_388e8f2.md` **블록 39** |
| 기준 덱 | KNGR `kngr_238.json`, `nxyz = 8,451`, `nsurf = 26,692`, `NG = 2` |
| 게이트 등급 | **B0** — 값도 산술 순서도 안 바뀐다. 바뀌는 것은 **어느 스레드가 그 바이트를 쓰는가**뿐 |
| 판정 | digest **`1f36e75dc00ed2b4` / `4377`** 불변 + `h5diff -c` 0 + 핀 CSV Δ = 0 |
| 계약 테스트 | `tools/test_flatxs_cta_contract.py` (B2) · `tools/test_nodal_soa_contract.py` (C2) |
| 소스 | `src/FlatXsCtaKernel.cuh` · `src/CudaXsReconBackend.cu/.h` · `src/CudaXsReconBackendStub.cpp` · `src/main.cpp` · `test/flatxs_device_replay.cu` |
| 한 줄 | **WP21-B/C가 "제로섬"이라고 적은 것은 *배열 순열*이고, 이 WP가 바꾸는 것은 *병렬화 축*이다.** 그 둘은 같은 문제가 아니다 |

> **선행 문서의 무엇을 뒤집고 무엇을 뒤집지 않는가.**
> WP21-B는 `c*nxyz + l`을 `l*880 + c`로 **순열하는** 안을 계산해서 정확히 제로섬이라고
> 결론지었다. **그 결론은 그대로 유효하고, 이 WP는 주소를 한 바이트도 옮기지 않는다.**
> 뒤집는 것은 그 다음 문단, "CTA를 노드 V개로 타일링하는 안"의 기각 사유 (c)다 —
> *"공유 워크스페이스가 V배가 되어 occupancy가 무너진다"*. 그 계산은 **스레드 수를 고정한
> 채** 워크스페이스만 V배로 놓았다. 스레드 수도 같이 V배로 하면 **스레드당 공유 바이트가
> 불변**이고, 공유 메모리가 제약하는 것은 블록 수가 아니라 상주 **스레드** 수다. §2.3이
> 그 산수다.

---

## 1. 관측 — 다시 한 번, 블록 39

| 커널 | ld sectors/req | st sectors/req | dram %peak | occupancy | 그리드 |
|---|---:|---:|---:|---:|---|
| `kernelFlatXsCta<128>` | 7.8 | **25.2** | **23.1** | 62.4 % | nxyz CTA × 128 스레드, 280 µs × 384 |
| `kNodalJnet<0>` | **16.7** | 7.7 | — | — | nsurf 스레드, 58 µs × 2,352 |

8바이트 접근의 이상값은 2다(8-스레드 wavefront × 8 B = 64 B = 2섹터).

---

## 2. WP21-B2 — flatxs CTA 스토어 전치

### 2.1 25.2의 원인은 레이아웃이 아니라 축이다

블록은 이미 컴포넌트-메이저·노드-최내측(`block_layout::elem`, `c*nxyz + l`)이고, 그 순서는
**다른 모든 소비자에게 옳다**(`kernelFlatXs`, CRAM D2D fill, Xe commit, 호스트 접근자 —
전부 `l`에 대해 병렬이다). 문제는 CTA arm만 `l`이 **블록-유니폼**이라는 것이다: 워프의
32 레인이 32개의 서로 다른 컴포넌트 서수 `q`를 들고, `c*nxyz + l` 아래에서 인접 레인의
주소는 `nxyz × elem_bytes`만큼 떨어진다. 노드당 880회의 스캐터가 **전부** 그 형태다.

### 2.2 고치는 방법 — 공유 메모리 전치, 그리고 **두 개의 레인 매핑**

CTA 하나가 `v.nodes`의 연속 `TILE`개를 맡고 슬롯마다 워크스페이스를 하나씩 든다. 한 커널
안에서 매핑이 둘이다:

| 단계 | 매핑 | 이유 |
|---|---|---|
| 델타 스트림 적용 | `q = lane; q < Q_x; q += T` (**오늘과 같음**, 그룹 `j`가 레인 `[j*T,(j+1)*T)`) | 계수 읽기 `cdata[(base+p)*NMIC + e]`는 `e == q`이므로 **레인 간 stride 1**이다. 이것이 CTA arm이 존재하는 이유이고 커널 로드의 대부분이다. 여기서 노드-최내측으로 가면 레인마다 `base`가 달라져 그 스트림이 흩어진다 |
| 게더 · 스캐터 · 밀도 · 매크로 | `p = tid; p < Q_x * TILE; p += NT`, **`q = p / TILE`, `j = p % TILE`** | 인접 레인이 같은 서수에서 연속 `l`을 들어 주소가 `c*nxyz + l0 + j`로 **연속** |

두 매핑이 만나는 경계 두 곳에만 `__syncthreads()`가 붙는다. **배리어는 값을 옮기지 않는다.**

### 2.3 이 커널을 묶는 자원은 공유 메모리가 아니라 **레지스터**다

처음 쓴 논증은 이랬다: `blockDim.x = T * TILE`이므로 **스레드당** 공유 바이트가 불변이고,
공유 메모리는 상주 블록이 아니라 상주 스레드를 제약하니 occupancy는 그대로다.

| arm | 슬롯 | TILE | 워크스페이스 | `sh_l` | CTA 합계 | blockDim | B/thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP64 (오늘) | 7,352 | 1 | 7,352 | — | 7,352 | 128 | 57.4 |
| FP64 (기본) | 7,352 | **2** | 14,704 | 8 | **14,712** | 256 | 57.4 |
| FP32 (오늘) | 3,676 | 1 | 3,676 | — | 3,676 | 128 | 28.7 |
| FP32 (기본) | 3,676 | **2** | 7,352 | 8 | **7,360** | 256 | 28.7 |

**그 논증은 맞는 말이지만 틀린 질문에 답하고 있었다.** 238 블록 40(`ddd0ccc`)이 옳은 질문을
정했다:

| arm | duration | warps active | dram %peak | ld | st |
|---|---:|---:|---:|---:|---:|
| `kernelFlatXsCta<128>` FP64 | 280 µs | **62.4 %** | 23.1 | 7.8 | 25.2 |
| 같은 커널, `RASBERY_GPU_FP32=1` | **379 µs (+35 %)** | **39.8 %** | 14.9 | 7.8 | 25.2 |

**공유 메모리를 절반으로 줄였는데 occupancy가 떨어졌다.** 공유 메모리 모델에서는 불가능한
일이고, 레지스터 모델에서는 산수다(SM당 레지스터 **65,536**, 스레드 2,048):

- FP64는 48 regs/thread(`tools/flatxs_resource_report.py`) × 128 스레드 = 6,144 regs/block →
  `floor(65,536 / 6,144)` = **10블록** → 1,280 스레드 = **62.5 %**. ncu가 잰 62.4 %와 같다.
  공유 메모리는 13블록을 허용했지만 **한 번도 구속조건이 아니었다.**
- FP32의 39.8 %는 약 6블록, 즉 **약 80 regs/thread**다. WP20이 `CtaWorkspaceF32` 주석에 적은
  "13 → 16블록"은 공유 예산에서 나온 수이고, 측정은 **블록을 하나도 사지 못했고 대신
  float↔double 왕복에 스레드당 32 레지스터쯤을 지불했다**고 말한다.

그래서 **타일 사다리는 레지스터로 채점한다.** 기본값은 48 KiB에 들어가는 가장 큰 값이 아니라
**상주 스레드 수를 그대로 두는 값**이다:

| arm | T | TILE | threads/blk | regs/blk | blocks/SM | threads/SM | 오늘 대비 |
|---|---:|---:|---:|---:|---:|---:|---|
| FP64 | 128 | 1 | 128 | 6,144 | 10 | 1,280 | (오늘) |
| FP64 | 128 | **2** | 256 | 12,288 | 5 | **1,280** | **중립** |
| FP64 | 128 | 4 | 512 | 24,576 | 2 | 1,024 | −20 % |
| FP32 | 128 | 1 | 128 | 10,240 | 6 | 768 | (오늘) |
| FP32 | 128 | **2** | 256 | 20,480 | 3 | **768** | **중립** |
| FP32 | 128 | 4 | 512 | 40,960 | 1 | 512 | −33 % |

**두 arm 모두 기본 TILE = 2.** 4와 8은 사다리에 남는다 — 스토어 25.2 → ~4(TILE 2) 對 ~2~3(TILE 4)와
occupancy −20 %의 교환은 실재하고, **그 값은 238만 매길 수 있다.** 이 WP가 하지 않을 일은,
WP20이 바로 이 커널의 occupancy를 엉뚱한 자원에서 추측하는 것을 방금 본 뒤에 또 추측하는 것이다.

**`__launch_bounds__`에 대해서도 하나 배웠다.** `__launch_bounds__(T)`는 `maxThreadsPerBlock`만
정하고 **그 외에는 아무것도 하지 않는다** — ptxas는 스레드당 최대 `65,536/T`(T=128이면 512)
레지스터를 써도 된다. 그래서 공유 메모리 논증이 예측한 블록 수를 **전달할 수 없고, 전달하지
않았다.** 레지스터를 묶으려면 두 번째 인자(`minBlocksPerMultiprocessor`)가 필요하고, 그것은
local memory 스필로 occupancy를 사는 것 — 이 arm이 애초에 없애려던 바로 그 트래픽이다.
그것은 편집이 아니라 측정이고, 여기서 하지 않는다.

### 2.4 기대 섹터 수 — 정직하게

| arm | wavefront | 커버 | 기대 st |
|---|---|---|---:|
| FP64 TILE=2 | 8 스레드 × 8 B | 2노드 × 4서수 = 16 B 덩어리 4개 | **~4** (이상값 2) |
| FP64 TILE=4 (사다리) | 8 스레드 × 8 B | 4노드 × 2서수 = 32 B 덩어리 2개 | ~2~3 |
| FP32 TILE=2 | 32 스레드 × 4 B | 2노드 × 16서수 = 8 B 덩어리 16개 | ~16 (이상값 4) |
| FP32 TILE=8 (사다리, occupancy −33 %) | 32 스레드 × 4 B | 8노드 × 4서수 = 32 B 덩어리 4개 | ~4~8 |

`nxyz = 8,451`은 4의 배수가 아니므로 `c*nxyz + l0`의 정렬이 `c`마다 흔들린다. 그래서 이상값이
아니라 그 근처를 적는다. 게더도 같은 형태이므로 ld 7.8도 같이 내려가지만, 계수 읽기가 이미
최적이라 하락 폭은 작다.

**FP32의 정직한 결론.** 좁은 arm의 스토어를 이상값으로 데려가려면 TILE ≥ 8이 필요하고,
80 regs/thread에서는 그 값을 살 수 없다. **타일은 FP32 회귀를 구제하지 못한다** — 고쳐야 할
것은 레지스터 압력 자체이고(워크스페이스가 float인데 산술이 double이라 원소마다 F2D/D2F가
붙는다), WP20.1이 블록까지 float으로 좁힌 뒤(`v.narrow_blocks`) 스토어 경로의 원소별 변환은
이미 사라졌으므로, 남은 것은 **Horner 누산이 워크스페이스를 읽고 쓸 때마다 하는 왕복**이다.
그것을 없애는 것은 산술을 float으로 바꾸는 일이고 A2 arm의 정의를 바꾸는 일이다 — 이 WP가
아니라 다음 WP의 질문이다. 꼬리 커널은 타일과 **같은 워크스페이스 타입**을 받으므로(계약
테스트가 강제한다) 변환을 되살리지 않는다.

### 2.5 왜 B0인가

1. **(P1) 레인 소유권**이 살아 있다. 모든 원소의 누산 사슬은 여전히 한 레인 안에 통째로 있다.
   레인의 **정체**가 단계 사이에서 바뀌고, 그 경계 두 곳에 배리어가 붙는다.
2. **(P2) 동위원소 폴드**는 여전히 `iso = 0..NISO-1` 오름차순, 출력 사슬당 한 레인, 트리 없음.
3. **(P3)** did/x/scale/xloc/base는 여전히 레인마다 재계산되고, 스트림 루프에 배리어가 **없다**.
   타일 슬롯마다 스트림 길이가 다르므로 여기에 배리어가 있었다면 매 델타마다 가장 긴 슬롯을
   기다렸을 것이다. `T ≥ 32`이므로 한 워프가 두 슬롯에 걸치는 일도 없다.
4. **전송이 안 바뀐다** — `[RASBERY][XFER]` 총계 바이트/호출 불변.
5. **꼬리**는 `n_nodes % TILE`개(≤ 3 또는 ≤ 7)이고 **인증된 미타일 본문**
   (`kernelFlatXsCtaAt` → `flatxsSolveNodeCta`)이 계산한다. 세 번째 본문은 없다.

### 2.6 두 개의 본문을 남긴 이유

`flatxsSolveNodeCta`는 **한 글자도 건드리지 않았다.** 그것이
`RASBERY_GPU_FLATXS_CTA_TILE=1`이 돌리는 arm이고 238이 인증한 arm이며, 이 변경의 A/B
기준이다. 하나의 템플릿으로 합쳤다면 기준과 후보가 같은 텍스트가 되어 이분이 불가능해진다.
대가는 **산술의 중복**이고, 그 대가는 계약 테스트가 지불한다: FormBit 센서스, 동위원소 폴드
규칙, bare multiply-add 규칙, 배리어 규칙이 **두 본문 모두에 대해** 같은 참조
(`flatxsSolveNode`)를 상대로 돌아간다. 한쪽만 드리프트하면 GPU에 닿기 전에 센서스가 깨진다.

### 2.7 노브와 수신증

```
RASBERY_GPU_FLATXS_CTA_TILE=<n>     # 미설정 = arm 기본(FP64 4 / FP32 8), 1 = 미타일 arm
[RASBERY][FLATXS][GPU] {"nodes_solved":N,"tile":4,"tiles_launched":T,"tail_nodes":R}
```

`tiles_launched * tile + tail_nodes == nodes_solved`가 덱을 몰라도 확인 가능한 항등식이다.
`tile`은 **요청값이 아니라 해석된 값**이다(사다리가 공유 메모리와 1,024 스레드 상한으로
클램프한다).

**`Driver.h`의 `trajectory::kArmEnv`에 넣지 않았다.** 이 노브는 *어느 레인이* 바이트를
쓰는지를 바꾸고 *어느 바이트를* 쓰는지는 바꾸지 않는다 — `RASBERY_GPU_MICX_RESIDENT`가
같은 이유로 목록 밖에 있다. 넣으면 하나의 arm의 캐시가 타일 값마다 갈라진다. 계약 테스트가
**부재를 강제**하므로 이 판정은 잊히지 않는다. (238 게이트가 타일이 궤적을 움직인다고
말하면 그때 가장 먼저 바뀌어야 하는 줄이 바로 그 부재 규칙이다.)

---

## 3. WP21-C2 — nodal stride 뷰 (이 커밋에서는 미착수)

§4 이후는 두 번째 커밋에서 채운다.

---

## 6. 238 runbook

로컬에 nvcc가 없다. **첫 관문은 238 컴파일이다.** GPU0만 사용한다.

```bash
export CUDA_VISIBLE_DEVICES=0
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

### 6.0 선행 — 계약 게이트 (로컬에서 통과)

```bash
python3 tools/test_flatxs_cta_contract.py
python3 tools/test_nodal_soa_contract.py
python3 tools/test_micx_layout_contract.py
python3 tools/test_gpu_fp32_contract.py
python3 tools/test_xfer_ledger_contract.py
python3 tools/test_enum_alias_contract.py
python3 tools/test_dependent_template_contract.py
python3 tools/test_nodal_gpu_refactor_contract.py
ctest
```

### 6.1 리플레이 게이트 — **이것이 B0의 실증이다**

238에서 nvcc가 붙는 즉시, digest보다 먼저 돌린다.

```bash
for T in 64 128 256; do
  for TILE in 1 2 4; do
    RASBERY_GPU_FLATXS_CTA_TILE=$TILE ./build/rasbery_flatxs_device_replay \
        <capture-base> --cta $T
  done
done
```

합격 조건: **모든 (T, TILE) 조합에서 `cta_vs_ref_mismatches = 0`**. 하나라도 0이 아니면
(P1)이 깨진 것이고, 그 조합이 곧 이분점이다.

### 6.2 digest 불변

```bash
env $V6_ENV <production arm> ... -o "$OUT/a_b2"      # 기본 타일 (FP64 4)
env $V6_ENV RASBERY_GPU_FLATXS_CTA_TILE=1 <arm> ... -o "$OUT/a_tile1"   # 미타일 기준
```

합격 조건 — **전부**:

- digest **`1f36e75dc00ed2b4` / `4377`** (두 arm 모두)
- 두 arm 사이 `h5diff -c` **0 차이**, 직전 v6 산출물 대비도 0
- `[RASBERY][FLATXS][GPU]`의 `tiles_launched * tile + tail_nodes == nodes_solved`
- `[RASBERY][MICX][LAYOUT]`가 `"layout":"soa","layout_version":2` (불변)
- `[RASBERY][XFER]` 총계 바이트/호출 **불변**
- CSV(핀 파워/AO/keff) 전 항목 Δ = 0

### 6.3 ncu 재측정 — 블록 39와 같은 디렉티브

```bash
for k in kernelFlatXsCtaTile kernelFlatXsCta kNodalJnet; do
  ncu --kernel-name "regex:$k" --launch-skip 10 --launch-count 5 \
      --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
      --csv env $V6_ENV <production arm> ... > "$OUT/ncu_$k.csv"
done
```

| 커널 | ld 전 | **ld 후 기대** | st 전 | **st 후 기대** | occupancy |
|---|---:|---:|---:|---:|---:|
| `kernelFlatXsCtaTile<128,2,CtaWorkspace>` | 7.8 | **~6** | 25.2 | **~4** | **62.4 % 유지** |
| 같은 커널, `RASBERY_GPU_FP32=1` | 7.8 | ~7 | 25.2 | ~16 | **39.8 % 유지** |
| `kNodalJnet<0>` | 16.7 | **~3** (§3) | 7.7 | 7.7 |  |

**FP32 arm에 대한 별도 기대치(블록 40의 회귀에 대한 답):** duration이 **FP64 arm 이하**여야
한다. 379 µs가 그대로면 §2.3의 레지스터 진단이 맞고 타일은 그것을 고치지 못한다는 뜻이며(예측한
대로다), 379 µs보다 **올라가면** 타일이 FP32 레지스터 압력을 더 키운 것이므로
`RASBERY_GPU_FLATXS_CTA_TILE=1`로 즉시 되돌린다. **occupancy가 39.8 %보다 내려가면 그것이
신호다.**

`RASBERY_GPU_FLATXS_CTA_TILE=1`로 다시 재면 `kernelFlatXsCta`가 **25.2 그대로** 나와야 한다 —
그것이 이 측정이 타일을 재고 있다는 증거다. occupancy는 62.4 % → **~58 %**(§2.3의 1,536/1,664)를
기대한다. **occupancy가 오르면 계산이 틀린 것이고, 크게 내리면 register spill을 의심한다**
(`tools/flatxs_resource_report.py`).

### 6.4 단일덱 wall — 워밍업 1 + hot 3

```bash
for r in w 1 2 3; do env $V6_ENV <production arm> ... -o "$OUT/b_$r"; done
```

`kernelFlatXsCta`는 280 µs × 384 launch = **107 ms**이고 단일 wall 11.2 s의 약 1 %다. 스토어
6배가 커널을 절반으로 줄여도 **wall은 ~50 ms(0.5 %)**다. 이 줄이 크게 움직이면 측정 노이즈이거나
§6.2가 깨진 것이다. **이 WP의 값은 단일 wall이 아니라 배치의 메모리 압력에 있다.**

### 6.5 배치 — 여기가 판정선

```bash
python3.11 tools/run_multi_gpu_batch.py --set "$V6_ENV" ...   # 8 procs x M16 + MPS
```

기준: **1,321 c/h**(블록 37/38의 클린 V2 baseline 1,320.8~1,326.7). MPS 아래 8 프로세스가
같은 L2와 같은 DRAM을 나눠 쓰므로, dram 23 %를 먹던 유일한 커널의 스토어 트래픽이 6배 줄면
**여기서 보인다**. 기본 타일은 occupancy 중립(§2.3)이므로 **회귀할 이유가 없다** — 회귀하면
레지스터 수가 48을 넘었다는 뜻이고, `--ptxas-options=-v`(RASBERY_PTXAS_VERBOSE)가 그것을
바로 말해 준다. 반대로 개선이 작으면 다음 A/B는 `RASBERY_GPU_FLATXS_CTA_TILE=4`
(29,408 B/CTA, 2블록/SM = 1,024 스레드, 스토어 ~2~3섹터)이고, 그것이 §2.3 표의 교환을 처음으로
가격표로 만든다.
