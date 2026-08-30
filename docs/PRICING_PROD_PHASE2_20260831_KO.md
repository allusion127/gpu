# PROD env 가격 평가 2단계 — 238 실측, v5 이후 (2026-08-31)

| 항목 | 값 |
|---|---|
| 대상 | **OUTER_GRAPH**(Task 10 conditional WHILE) · **OUTER_SEGMENT_V2**(WP14) · **MICX_RESIDENT**(WP15) · **WP15.1**(CRAM mic H2D→D2D) · **CUDA_SYNC_MODE**(WP16) · **CMFD_BLOCK**(WP17) · **rolling refill**(WP18) · **WP9-D 탐색 레버 5종** |
| 호스트 | **238 GPU0 전용** (블록마다 사전 점검: GPU0 유휴, 경쟁 RASBERY 프로세스 0). GPU: RTX PRO 6000 Blackwell, **188 SM**, cc 12.0 |
| 브랜치 | `codex/exact-throughput-campaign` |
| 측정 커밋 | `27fa483`(GRAPH/SEGMENT_V2) · `7af4099`(v5 freeze 확인) · `6bdfdc2`(MICX_RESIDENT·v6) · `cd3630c`(WP15.1) · `8b3c62b`(WP16) · `d68de15`(WP17) · **`bd7a0d3`**(WP18, 현재 팁) |
| 덱 | 단일: `kngr_238.json`(35상태, 1/4 노심 8,451 노드, NG=2) · 배치: `benchmark/m64_manual` 64잡/128잡 매니페스트 · 동일덱 배치: **KNGR×64** 매니페스트 |
| 프로토콜 | 단일 arm당 **warm-up 1 + hot 3, 교차 순서, median**. 배치는 매니페스트 1회 완주 |
| 원천 | 238 러너 보고 `pricing_388e8f2.md` **블록 19–33**. 블록 0–9b는 [`PRICING_PROD_20260830_KO.md`](PRICING_PROD_20260830_KO.md), 블록 10–18은 [`V5_FREEZE_20260830_KO.md`](V5_FREEZE_20260830_KO.md)가 이미 기록했다 |
| 기준기 | MASTER (CPU) — 단일 **27.2 s** / W16 배치 **217 c/h**(m64 세트 기준) |

---

## 0. 한 줄 요약

**단일덱 `9.808 s` = MASTER 27.2 s 대비 2.77×**, **배치 m64 `1,246 c/h` = MASTER W16 217 c/h 대비 5.7×**,
그리고 **같은 덱으로 잰 배치(KNGR×64) `2,016 c/h` = 1.786 s/case**다.
v5(11.189 s)에서 **−1.381 s**가 더 내려왔고 그 전부가 **B0**다 —
`OUTER_GRAPH`(그래프로 캡처한 WHILE 루프가 h5diff 0으로 **비트가 같았다**),
`MICX_RESIDENT`(micx D2H 18.69 → 3.31 GB), `WP15.1`(CRAM mic H2D **0 바이트**),
`CMFD_BLOCK=64`(같은 일을 **더 많은 더 작은 블록**으로 — `vector_blocks` 67 → 265, 처음으로 2×SM 돌파).

**병목의 이름이 이번 패스에서 두 번 바뀌었다.** PCIe는 Gen4 x16 만속인데 배치에서 **25–35 %만 쓴다**(§3.4).
호스트 CPU 74 %는 실작업이 아니라 **`cudaStreamSynchronize` 스핀**이었고(§5.5), `blocking`이 그것을 40 %로 떨어뜨린다.
그리고 배치는 **폭 기아가 아니라 GPU 시간 바운드**다 — WP18의 슬롯 단위 리필이 `slot_idle`을 1 ms로,
`tail_idle`을 0으로 만들고도 c/h는 **오히려 떨어졌다**(§6).

**이번 패스가 남긴 가장 비싼 교훈**: 계기가 보여주는 "바쁨"은 "가득 참"이 아니다.
`nvidia-smi` sm % 96 %는 **커널이 상주했다**는 뜻이지 **188개 SM이 찼다**는 뜻이 아니었고(블록 27),
실제로는 상위 커널 15개 중 **14개가 376블록(2×SM) 미만**, 최대 소비자
`colored_block_sweep`는 **34블록으로 379,027번** 떴다.

---

## 1. 채택 레버 사다리 — 16.658 → 9.808 s

| # | 단계 | 커밋 | median (s) | Δ (s) | 등급 | MASTER 27.2 s 대비 |
|---:|---|---|---:|---:|---|---:|
| 0 | PROD 기준선 (블록 9 arm a) | `91004f7` | 16.658 | — | — | 1.63× |
| 1 | + CRAM · FUSE=15 · PPR · PPR_GRAPH · XE_TXN = **v4** | `91004f7` | 14.378 | −2.280 | **N1** (CRAM 지배) | 1.89× |
| 2 | + `RESULT_ASYNC` (WP12) | `b903225` | 11.952 | −2.480 | **B0** | 2.28× |
| 3 | + `FLATXS_CTA` (WP5-B) = **v5 동결** | `b903225` | **11.189** | −0.913 | **B0** | 2.43× |
| 4 | + `OUTER_GRAPH` · `MICX_RESIDENT` · `XFER_ELIDE` · `OUTER_SEGMENT_V2` = **v6** | `6bdfdc2` | **10.206** | −0.875 | **B0 넷 전부** | 2.67× |
| 4b | (v6 그대로, WP15.1 코드 변경만) | `cd3630c` | 10.217 | −0.098 | **B0** | 2.66× |
| 5 | + `RASBERY_GPU_CMFD_BLOCK=64` (WP17) | `d68de15` | **9.808** | −0.420 | **B0** | **2.77×** |

> **Δ를 세로로 더하지 말 것.** 각 Δ는 **자기 교차의 base**에 대한 것이고 base가 단계마다 다르다
> (4단계의 v5 base는 11.081 s, 5단계의 256-기본값 base는 10.228 s). V5_FREEZE가 세운 규칙 그대로,
> **정직한 누적은 양 끝뿐이다: 16.658 → 9.808 = −6.850 s (−41.1 %)**.

**등급의 의미.** 사다리에서 궤적을 움직이는 것은 **1단계의 CRAM 하나뿐**이다(N1).
2단계 이후의 다섯 레버는 전부 **B0** — digest `1f36e75dc00ed2b4` / outers **4377**이
`b903225`부터 `d68de15`까지 **한 번도 움직이지 않았고**, 각 A/B 쌍에서 `h5diff` 0 차이다.
즉 **v5 → 9.808 s의 −1.381 s는 정확도를 한 비트도 팔지 않고 얻은 것**이다.

---

## 2. v6 스택 — 구성요소별 가격표 (블록 19 · 21 · 22)

| 기능 | env | 등급 | 기준 arm (s) | 채택 arm (s) | Δ (s) | 게이트 | 판정 |
|---|---|---|---:|---:|---:|---|---|
| **OUTER_GRAPH** | `RASBERY_GPU_OUTER_GRAPH=1` | **B0 (실측)** | 11.161 | **10.876** | **−0.285 (−2.6 %)** | `h5diff` **0 차이(전 데이터셋)** · Gate A 전 상태점 **literal 0** · Gate B 1.847 pcm / 15.334 ppm / 0.012 AO / **BOC 핀 0.238 %·0.80 %** · 결정론 ×2 h5diff 0 | **채택** |
| **OUTER_SEGMENT_V2** | `RASBERY_GPU_OUTER_SEGMENT_V2=1` | **B0** | 11.161 | 11.135 | −0.026 (−0.2 %) | `h5diff` 0 · `cmp` 핀파워 CSV 동일 | **채택(단일)** — GRAPH 위에서는 −0.003 s |
| **MICX_RESIDENT** | `RASBERY_GPU_MICX_RESIDENT=1` | **B0** | 11.410 | **10.716** | **−0.694** | `h5diff` 0 · `cmp` CSV 동일 · `[ERROR][micx]` **0/0** | **채택(단일+배치)** |
| **XFER_ELIDE** | `RASBERY_GPU_XFER_ELIDE=1` | B0 | (블록 15) | — | H2D 소폭 | 원장으로 확인 | **채택** |
| **v6 전체** | 위 넷 동시 | **B0** | **11.081** | **10.206** | **−0.875 (−7.9 %)** | `h5diff` v5 vs v6 **0** · `cmp` CSV 동일 · micx ERROR 0 | **채택** |

### 2.1 OUTER_GRAPH — N1을 예상했는데 B0가 나왔다

런북은 그래프 캡처된 WHILE 루프를 **N1(부동소수 결합 순서 이동)** 등급으로 들어갔다.
238이 실제로 잰 결과는 그보다 강하다 — `gate_a_compare.py --per-step`이 **모든 상태점에서 literal 0**
(`keff/ppm/AO/pin` 전부 `0.0000`)을 찍었고, 그것만으로는 시사에 불과하므로
**`h5diff -c` 전 데이터셋을 직접 돌려 0 차이**를 확인했다. **이 덱에서 이 기능은 B0다.**

`[RASBERY][OUTER_GPU]` 수신증:
`graph_segments 1255` · `graph_launches 1255` · `graph_iterations 2781` ·
`iterations_per_launch 2.216` · **`graph_instantiations 2`**(35상태 전체에서 재인스턴스화 2회) ·
`graph_warmup_misses 0`.

**호스트 sync가 어디로 갔는가**: `sync_exit_observation` **4221 → 325 (−92 %)** —
블록 16의 원장이 "전체 sync 시간의 65 %(3.727 s)"로 지목했던 바로 그 사이트다.
동시에 `sync_hostfree_exit`는 147 → 1255으로 **일이 사라진 것이 아니라 옮겨갔다**.

**그런데 −2.6 %밖에 안 되는 이유**(§3의 원장이 답한다): sync **호출 수**는 9,546 → 5,650(−41 %)인데
sync **시간**은 5.715 → 5.687 s(−0.5 %)다. **GRAPH가 없앤 것은 싼 sync들이었다.**
비싼 꼬리는 이 플래그 혼자로는 거의 건드리지 못한다. SEGMENT_V2를 그 위에 얹어도
10.876 → 10.873으로 사실상 0인 이유도 같다 — 두 기능이 **같은 절약을 겹쳐서 노린다**.

세그먼트 쪽 수신증도 함께 남긴다(SEGMENT_V2 arm): `segment_passes 5547` · `discovery_passes 1163` ·
`outers_per_segment 3.301` · `v2_exit_syncs_elided 1108` · `v2_state_d2h_elided 1163` ·
`escapes {none:1, flux_converged:1086, negative_flux:115, segment_budget:124}`.
**escape 분포는 모든 arm에서 같다** — 그것은 세그먼트 구현의 성질이 아니라 **덱 자신의 수렴 성질**이다.

### 2.2 MICX_RESIDENT — 캠페인 전체 배치 최대 승리

`[RASBERY][MICX]` 수신증(schema 1): `resident_hits 384` · **`lazy_downloads 0`**(구 경로로 한 번도 되돌아가지 않았다) ·
`slice_downloads 68` · `materialised_per_hit 0.177` · **`bytes_saved 19,536,007,680 (18.63 GB)`**.

| 원장 행 (D2H) | 이전 (A) | 이후 (B) |
|---|---:|---:|
| `flatxs micx mic` | 3,456 call / **18.22 GB** | 612 call / **3.23 GB** |
| `flatxs lmpx lmp` | 3,456 call / 0.467 GB | 612 call / 0.083 GB |
| **micx D2H 합(두 행)** | **18.69 GB** | **3.31 GB** |
| `solveFlatXs:drain` sync | 905.85 ms | **124.64 ms** |

**배치**(8×M8+MPS12 %, 64잡): **1,163.7 c/h** — 직전 기준선 975.5 c/h 대비 **+19.3 %**.
블록 17의 형상 스윕 전체(무엇도 +5 %를 넘지 못했다)보다 이 플래그 하나가 더 컸다.
전 8프로세스 로그에서 `[ERROR][micx]` **0줄**.

### 2.3 WP15.1 (`cd3630c`) — CRAM mic이 PCIe를 떠났다

| arm | median (s) | digest | outers |
|---|---:|---|---:|
| OLD (`6bdfdc2`, v6) | 10.315 | `1f36e75dc00ed2b4` | 4377 |
| NEW (`cd3630c`, v6) | **10.217** | `1f36e75dc00ed2b4` | 4377 |

`h5diff` 0 · `cmp` CSV 동일 · micx ERROR 0. **Δ = −0.098 s (−1.0 %)**, 배치 **+0.6 %**(1,172.4 → 1,179.1 c/h).

MICX 수신증 schema 2가 그것을 직접 말한다: **`cram_micx_h2d_mb: 0.0`** ·
`cram_micx_d2d_mb: 1367.9` · `bytes_saved 21,191,051,520 (20.21 GB)`.
원장에서 `CudaCramBackend.cu:h2d:H2D mic (predictor/corrector)` 행이 **아예 사라졌고**
(`fillMic:D2D mic` 272 call / 1.43 GB + `predictor:mic_bos (D2D)` 136 call / 0.72 GB로 대체),
`d2d_bytes`가 717 MB → 2.15 GB로 올랐다.

**미engage 하나 — 정직하게 적는다**: 같은 커밋의 nodal jnet H2D 그림자/소거 기능은
`nodal_jnet_elided_mb: 0.0` · `nodal_jnet_elision_tests: 0` · `nodal_jnet_hit_rate: −1.000`으로
**이 단일덱 실행에서 한 번도 동작하지 않았다**. 커밋 메시지를 근거로 "동작한다"고 적지 않는다.

---

## 3. 전송 원장 — 무엇이 실제로 줄었는가

`RASBERY_XFER_LEDGER=1`(169 사이트 전부 계수 래퍼 경유), 단일덱 1회 기준.

| 지표 | 블록 16 기준선 | **v6** (`6bdfdc2`) | **WP15.1** (`cd3630c`) | 기준선 대비 |
|---|---:|---:|---:|---:|
| `sync_calls` | 9,546 | 5,718 | 5,684 | **−40.5 %** |
| `sync_ns` | 5.715 s | 4.937 s | 4.839 s | −15.3 % |
| **D2H bytes** | **28.62 GB** | **8.72 GB** | **7.06 GB** | **−75.3 %** |
| H2D bytes | 11.45 GB | 11.08 GB | 9.65 GB | −15.7 % |
| D2D bytes | 0.717 GB | 0.717 GB | 2.152 GB | +200 % (의도) |

**읽는 법 — 세 레버가 원장의 서로 다른 부분을 움직인다.**
D2H 붕괴(v6에서 −69.5 %)는 **MICX_RESIDENT**의 몫이고, sync 호출 수 −40 %는 **OUTER_GRAPH**,
H2D 감소와 D2D 상승은 **WP15.1**이다. 어느 하나도 다른 하나의 자리를 대신하지 않는다.

> **귀속 주의**: v6(8.72 GB) → WP15.1(7.06 GB)의 D2H 추가 −1.65 GB는 **CRAM H2D 제거로 설명되지 않는다**
> (CRAM의 D2H 쪽은 이 변경이 건드리지 않는다). 러너가 다른 사이트의 실행 간 변동일 가능성을 명시했고,
> 여기서도 **귀속하지 않은 채로 남긴다**.

### 3.4 PCIe — 더 이상 병목이 아니다

`nvidia-smi -q -i 0`: `PCIe Generation Max=4 / Current=4`(디바이스는 Gen5지만 **Host Max=4**로 보드가 묶는다),
`Link Width Max=16x / Current=16x`. **만속 Gen4 x16, 열화 없음.**

| 배치 arm | rx mean (MB/s) | rx p95 | tx mean | tx p95 | rx % of ~25 GB/s | tx % |
|---|---:|---:|---:|---:|---:|---:|
| OLD (`6bdfdc2`) | 8,934.9 | 14,131 | 7,740.6 | 10,196 | 34.9 % | 30.2 % |
| **NEW (`cd3630c`)** | 8,284.1 | 13,722 | **6,326.7** | 9,027 | 32.4 % | 24.7 % |
| NEW + SEGMENT_V2 | 8,343.5 | 13,829 | 6,337.6 | 8,819 | 32.6 % | 24.8 % |

**평균 25–35 %, p95도 천장의 55 % 미만.** 그리고 `cd3630c`는 **PCIe를 18.3 % 덜 쓰면서 오히려 더 빠르다** —
전송 대역은 처음부터 배치의 제약이 아니었다는 것을 같은 표가 두 번 말한다.

---

## 4. WP17 — 96 % "바쁜" GPU의 82 %는 비어 있었다

### 4.1 커널 격자 인구조사 (`nsys --cuda-graph-trace=node` → CUPTI 커널 레코드, 단일덱 v6)

| 커널 | 런치 | 총 ms | 격자(블록) | 188 SM 대비 |
|---|---:|---:|---:|---|
| `colored_block_sweep` | **379,027** | **967.57** | **34** | 0.18× |
| `reduce_dot_fused` | 94,991 | 330.68 | 17 | 0.09× |
| `matvec_two_group` | 95,405 | 262.43 | 34 | 0.18× |
| `reduce_dot2_fused` | 47,925 | 207.22 | 17 | 0.09× |
| `kXeCommitTxn` | 1,117 | 145.26 | 48 | 0.26× |
| `reduce_norm_accumulate_stage2` | 47,461 | 134.78 | **1** | **0.005×** |
| `kCorrector` / `kPredictor` (CRAM) | 34 / 34 | 120.19 / 117.91 | 133 | 0.71× |
| **`kernelFlatXsCta<128>`** | 384 | 107.34 | **8,451** | **44.9×** |

**상위 15개 중 14개가 2×SM(376블록) 미만이고, 유일한 예외가 이미 최적화된 flatXs CTA 커널이다.**
`nvidia-smi` sm % 96 %(블록 24)는 "커널이 상주했다"는 뜻이지 "SM 배열이 찼다"는 뜻이 아니었다.
**GPU busy ≠ GPU full.** 그리고 이 커널들은 느리지 않다(CMFD/BiCG 계열 평균 2.4–4.3 µs) —
**아주 많을 뿐**이다. 산술을 빠르게 해도 이 그림은 바뀌지 않고, **런치당 독립 작업을 더 담는 것**만이 바꾼다.

### 4.2 `CMFD_BLOCK` 스윕 (단일덱 v6, 교차, warm-up 1 + hot 3)

| block_threads | median (s) | node_blocks | vector_blocks | digest |
|---:|---:|---:|---:|---|
| 256 (기본) | 10.228 | 34 (0.18×) | 67 (0.36×) | `1f36e75dc00ed2b4` |
| 192 | 10.067 | 45 (0.24×) | 89 (0.47×) | 동일 |
| 128 | 9.836 | 67 (0.36×) | 133 (0.71×) | 동일 |
| **64** | **9.808** | 133 (0.71×) | **265 (1.41×)** | 동일 |
| 32 | 9.825 | 265 (1.41×) | **529 (2.81×)** | 동일 |

**5 arm 전부 B0** — digest/outers 동일, 모든 arm vs blk256 `h5diff` **0**, 양 극단(256 vs 32) `cmp` CSV 동일.
**`CMFD_BLOCK`은 순수한 런치 형상 노브이고 물리를 건드리지 않는다.**

**채택: 64, 9.808 s (−4.1 %).** 32는 점유율이 더 높은데도 더 빠르지 않다 —
더 작은 블록이 더 많아질수록 결국 grid-sync/런치 오버헤드로 되갚기 때문이고, 이 덱에서 64가 그 임계다.

**배치는 평평하다**: 8×M8+MPS12 %에서 **1,176.1 c/h (−0.3 %)**, micx ERROR 0.
→ **단일덱 전용 채택.** 배치 기본값 후보가 되지 않는다.

### 4.3 persistent cooperative BiCG — **영구 폐쇄**

`tools/probe_gridsync_cost.cu`(`nvcc -O3 -std=c++17 -arch=sm_120 -rdc=true -lcudadevrt`),
가정된 형상이 아니라 **덱의 실제 CMFD 형상**(34 · 67 블록)으로:

```
{"blocks":34,  "c_barrier_us":0.7818, "passes_gate":false}
{"blocks":67,  "c_barrier_us":0.7884, "passes_gate":false}
{"blocks":209, "c_barrier_us":0.9504, "passes_gate":false}
{"blocks":1188,"supported":false,"reason":"grid exceeds co-resident capacity"}
{"blocks":4224,"supported":false,"reason":"grid exceeds co-resident capacity"}
summary: {"c_barrier_us_gate_value":0.7884,"gate_c_barrier_us":0.3840,
          "removable_seconds_at_measured":1.656,"verdict":"NO-GO"}
```

**`c_barrier` = 0.788 µs, 게이트 0.384 µs의 2.05배.** 프로브 자신의 판정이 **NO-GO**다.
게다가 이 디바이스의 co-resident 최대는 **1,128블록(6/SM × 188)**이라
배치 규모의 형상(1,188 / 4,224블록)은 **협조 실행 자체가 불가능**하다.
블록의 자체 조건부 규칙에 따라 persistent 스파이크는 **시도조차 하지 않았다**. 이 트랙은 닫힌다.

---

## 5. 배치 진실표

### 5.1 형상 — m64 128잡 매니페스트

| 형상 | claim | sync mode | 커밋 | c/h | width_fill | tail_idle_max (s) |
|---|---|---|---|---:|---:|---:|
| **8×M16** | auto | auto | `bd7a0d3` | **1,246.2** | 0.3676 | 358.6 |
| **8×M8** | auto | auto | `bd7a0d3` | **1,186.1** | 0.4477 | 168.3 |
| 8×M16 | auto | auto | `cd3630c` | 1,235.1 | 0.3671 | 355.8 |
| 8×M8 (기준) | auto | auto | `cd3630c` | 1,182.0 | 0.4473 | 155.2 |
| 8×M16 | rolling | auto | `bd7a0d3` | 1,181.2 | 0.3564 | **0** |
| 16×M8 | auto | **blocking** | `8b3c62b` | 1,159.5 | 0.4218 | 114.1 |
| 8×M8 | rolling | auto | `bd7a0d3` | 1,158.3 | 0.4320 | **0** |
| 8×M16 | rolling | **blocking** | `bd7a0d3` | 1,109.8 | 0.3091 | **0** |
| 16×M8 | auto | auto | `cd3630c` | 1,035.3 | 0.5220 | 122.1 |
| 16×M16 | auto | auto | `cd3630c` | 1,023.9 | 0.2605 | — |
| 8×M8 | claim=4 | auto | `cd3630c` | 978.0 | 0.2580 | 190.0 |
| 12×M8 | auto | **blocking** | `8b3c62b` | 976.3 | 0.4120 | 154.2 |
| 12×M8 | auto | auto | `cd3630c` | 944.5 | 0.4714 | 101.7 |
| 8×M8 | claim=2 | auto | `cd3630c` | 840.3 | 0.1837 | 7.6 |
| 8×M8 | claim=1 | auto | `cd3630c` | 592.8 | 0.1250 | **0** |

**규칙 넷이 이 표에서 읽힌다.**
(i) **8프로세스를 넘으면 진다** — 12/16 프로세스는 MPS 스레드 지분이 12 % → 8 % → 6 %로 희석되고,
그 손실이 arena 채움 개선보다 크다. (ii) **폭을 넓히면 조금 이기고 꼬리가 나빠진다** — 8×M16이 최고(+5.1 %)지만
`tail_idle_max`가 자기 wall(373–390 s)의 **95 % 이상**이다. (iii) **`--claim` 청크를 줄이면 꼬리는 사라지고 처리량이 무너진다**.
(iv) **`blocking`은 좁히되 뒤집지 못한다** — 16×M8을 **+12.0 %** 끌어올리지만(1,035.3 → 1,159.5)
여전히 8×M8 기준선(1,182.0)에 −1.9 %다.

> **블록 25의 confound와 그 정정을 기록한다.** 원래 스윕은 12×M8(96슬롯)·16×M8(128슬롯)을
> **64잡** 풀로 먹였다 — `width_fill`이 산술적으로 0.667 / 0.5를 넘을 수 없었다. 128잡으로 재실행하니
> `width_fill`은 프로세스가 늘수록 **오히려 상승**했고(0.4473 → 0.4714 → 0.5220), 그런데도 c/h는 내려갔다.
> **채움이 문제가 아니었다는 첫 번째 증거**가 여기서 나온다.

### 5.2 덱 회계 — 배치 c/h가 낮아 보이는 이유의 절반

블록 30의 케이스별 수신증 255표본(8워커 전부)에서 직접 잰 값:

| 덱 | 상태점/케이스 | outers/케이스 | KNGR 대비 |
|---|---:|---:|---:|
| `kngr_238.json` (단일덱 벤치마크) | 35 | 4,377 | 1.00× |
| `benchmark/m64_manual` 평균 | **51.0** | **8,415.5** | **1.92×** (outer 기준, 상태점 기준 1.46×) |

**m64 덱은 실제로 더 무겁다.** 따라서 m64 배치 c/h를 KNGR 단일덱 wall과 직접 나누면 안 된다.
**MASTER W16 217 c/h 기준도 같은 m64 세트**이므로 배수는 그대로 성립한다:

> **1,246.2 / 217 = 5.7× (MASTER W16)**

### 5.3 같은 덱으로 잰 배치 — KNGR×64 (블록 31)

| 항목 | 값 |
|---|---|
| 형상 | 8×M8 + MPS auto, `bd7a0d3`, v6 env |
| wall / c/h | **114.274 s / 2,016.2 c/h** |
| s/case | **1.786 s** |
| 케이스별 outers | **128행 전부 `4377`** (KNGR PROD 궤적 그대로) |
| sm % | mean 94.8, p95 100, 표본의 94.7 %가 ≥85 % |

단일덱의 커널시간 이상치 **~1.2 s/case**에 대면 **1.786 / 1.2 = 1.49×**,
즉 **배치 모드 자체가 케이스당 +49 %를 더 쓴다**(뒤집으면 이상치의 **67 % 효율**).
**이것은 덱 무게로 설명될 수 없다 — 같은 덱이다.**

그래서 m64의 ~2.55배 격차는 둘 다 실재하는 두 요인의 곱에 가깝다:
**덱 무게 1.92× × 배치 오버헤드 1.49× = 2.86×**. 러너가 코디네이터의 두 갈래
(≈3,000 c/h "덱이 무거울 뿐" / ≈1,200 c/h "배치가 2.5배 비싸다") 중
**어느 쪽도 맞지 않았다**고 적은 그대로다 — 실측 2,016.2는 그 사이에 있다.

### 5.4 +49 %는 MPS가 아니라 **폭**이다 (블록 32, KNGR×64)

| arm | 형상 | MPS | wall (s) | c/h | s/case | mean_width |
|---|---|---|---:|---:|---:|---:|
| (a) 1×M64 raw | 1 proc × 폭 64 | 없음 | 149.844 | 1,537.6 | 2.341 | 19.693 |
| (b) 1×M8 raw | 1 proc × 폭 8 | 없음 | 288.128 | 799.6 | 4.502 | 4.103 |
| (c) 8×M8 no-MPS | 8 proc × 폭 8 | **없음** | 226.076 | 1,019.1 | 3.532 | 3.520 |
| (d) 2×M32 | 2 proc × 폭 32 | 50 % | 128.219 | 1,796.9 | 2.003 | 10.743 |
| (e) 4×M16 | 4 proc × 폭 16 | 25 % | 115.130 | 2,001.2 | 1.799 | 6.062 |
| **(31) 8×M8** | 8 proc × 폭 8 | **auto** | 114.274 | **2,016.2** | **1.786** | — |

- **MPS는 오버헤드의 원인이 아니라 동시성의 원천이다**: 같은 8×8 형상에서 MPS를 끄면
  2,016.2 → **1,019.1 c/h**로 **절반**이 된다(**1.98×**). MPS 없는 8프로세스 시분할이 느린 길이다.
- **폭 단독으로 1.92×**: 1×M64(1,537.6) vs 1×M8(799.6), 둘 다 단일 프로세스·MPS 없음.
- **KNGR 최적 형상은 8×M8+MPS auto와 4×M16+MPS25가 사실상 동률**(2,016.2 / 2,001.2, ~1 % 차),
  2×M32+MPS50은 −11 %로 프로세스가 너무 적다.
- 5개 arm 전부 64/64 ok, 실패 0, 중복 0, stale tenant 0, isolation mismatch 0 — **320케이스 전부 정확**.

### 5.5 호스트 CPU 74 % — 실작업이 아니라 스핀이었다 (WP16, `8b3c62b`)

**배치 8×M8+MPS12 %:**

| sync mode | wall (s) | c/h | vs auto | CPU (us+sy) | width_fill |
|---|---:|---:|---:|---:|---:|
| auto (기본) | 195.835 | **1,176.5** | — | **72.8 %** | 0.4459 |
| **blocking** | 203.128 | 1,134.3 | **−3.6 %** | **40.1 %** | 0.4126 |
| yield | 196.274 | 1,173.9 | −0.2 % | 67.0 % | 0.4199 |

세 arm 모두 `[RASBERY][CUDA][SCHED]`가 `applied == requested`, `rc:"cudaSuccess"` — 노옵이 아니다.

**단일덱은 정반대다**: auto 11.316 s vs blocking **12.296 s (+8.7 %)** (`h5diff` 0, B0 확인).
혼자 도는 프로세스에는 **해방된 코어를 줄 다른 일이 없다** — `cudaDeviceScheduleBlockingSync`는
스핀의 즉시 깨어남을 커널 수준 wait/wake 왕복과 바꾸고, 경합이 없으면 그것은 순손실이다.

> **블록 26의 정정을 그대로 남긴다.** `OMP_WAIT_POLICY=PASSIVE`/`GOMP_SPINCOUNT=0` 실험은
> 두 arm의 CPU가 74.6 % vs 74.2 %로 같다고 보고했고, 그것을 "74 %는 실작업"으로 읽었다.
> **틀렸다** — PASSIVE는 OpenMP 자신의 배리어 스핀만 없애고 CUDA의 device-schedule 스핀
> (`cudaDeviceScheduleSpin`, 기본값)은 건드리지 않는다.
> `blocking`이 CPU를 40.1 %로 떨어뜨린 위 표가 그 해석을 뒤집는다: **~33 %p가 `cudaStreamSynchronize` 스핀이었다.**
> (`perf`·`ncu`는 238에 설치되어 있지 않아 중간 측정으로 확인할 수 없었다.)

**판정: 기본값 아님, 배치 옵션.** 코디네이터의 바는 "c/h 손실 없음"이었고 −3.6 %는 그것을 넘지 못한다.
값은 **해방된 CPU가 무엇을 가능하게 하는가**에 있다 — 16×M8이 그 CPU로 +12.0 % 오른다(§5.1).
반대로 8×M16은 −1.3 %로 **개선되지 않는다**: 그 형상의 병목은 호스트 CPU가 아니라 꼬리다.

---

## 6. WP18 rolling refill — 기계는 옳고, 가설이 틀렸다 (`bd7a0d3`)

| arm | c/h | width_fill | mean_width | `tail_idle_max` | vs auto |
|---|---:|---:|---:|---:|---:|
| (a) 8×M8 auto | 1,186.1 | 0.4477 | 3.581 | 168.3 s | — |
| (b) 8×M8 **rolling** | 1,158.3 | 0.4320 | 3.456 | **0** | **−2.3 %** |
| (c) 8×M16 auto | 1,246.2 | 0.3676 | 5.882 | 358.6 s | — |
| (d) 8×M16 **rolling** | 1,181.2 | 0.3564 | 5.703 | **0** | **−5.2 %** |
| (e) 8×M16 rolling + blocking | 1,109.8 | 0.3091 | — | **0** | −6.1 % vs (d) |

`[RASBERY][REFILL][ROLLING]` 수신증(8×M8 워커 1): `arena_width 8` · `lanes_used 8` ·
`admits 15` · `immediate_admits 8`(53 %) · `wave_barriers_avoided 14` ·
**`slot_idle_ms_total 0.99726`** · `width_history {p10:2, p50:8, p90:8, mean:6.133, max:8}`(**p50 = 8, 즉 arena가 대체로 가득 찼다**).

**기계는 설계대로 동작한다** — 슬롯은 사실상 리필을 기다리지 않고(≈1 ms), 웨이브 배리어는 사라지고,
`tail_idle`은 **진짜 0**이 된다. **그런데 처리량이 오르지 않는다.**
기대 밴드는 1,500–1,900 c/h였고 실측 최고는 1,246.2(그것도 auto arm)다.
정확성은 완벽하다: **`cross_case_digest_mismatch = 0`** — 64개 후보 케이스 전부가
auto와 rolling 사이에서 **비트 동일 digest**를 냈다(요구된 ≥8케이스 검사를 훨씬 초과).

**결론: 배치는 arena 채움 바운드가 아니라 GPU 시간 바운드다.**
채움을 완벽하게 만들어도 c/h가 내려간다면, 남은 비용은 잡 리스트를 자르는 방식이 아니라
**케이스당 GPU 시간** 자체에 있다 — §5.3의 KNGR×64가 그것을 직접 쟀다.

**판정: 미채택. 코드는 스케줄러 기반(substrate)으로 남긴다.**
웨이브 배리어 없이 슬롯 단위로 잡을 admit하는 기계는 멀티GPU dispatcher와
statepoint-phase residency가 요구할 바로 그 골격이고, 그것을 다시 쓰는 것은 낭비다.

---

## 7. 기각·폐쇄 목록 — 각각 자기 수와 함께

| 항목 | 측정 | 바 | 판정 |
|---|---|---|---|
| **persistent cooperative CMFD/BiCG** | `c_barrier` **0.788 µs** (34블록 0.782 / 67블록 0.788 / 209블록 0.950) | 게이트 **0.384 µs** | **영구 폐쇄** — 2.05배. co-resident 최대 1,128블록이라 배치 형상은 협조 실행 자체가 불가 |
| **WP18 rolling refill** | 1,158.3 (M8, −2.3 %) / 1,181.2 (M16, −5.2 %) / 1,109.8 (M16+blocking, −6.1 %) | 기대 밴드 1,500–1,900 c/h | **미채택, 코드 유지** — 정확(digest mismatch **0/64**), 기계는 동작, 가설이 틀렸다 |
| **WP9-D 탐색 레버 5종** | `boron_bracket` outers **4,377 → 4,300 (−1.76 %)** · `carry_slope` **+8.11 %** · `all_together` **+15.8 %** · `warm_boron`·`max_trials=12` **no-op** · `staged_margin=2` +0.11 % | outers **−10 % 이상** + Gate B 봉투 안 | **기각** — 최선이 −1.76 %. Gate A는 keff **2.27 pcm**(한도 1.905)로 스크린 밖, Gate B도 1.945 pcm / 15.334 ppm으로 근소 초과 |
| ┗ 그 배치 arm | 8×M8 **1,316.7 c/h 클린(128/128)** / 8×M16 1,333.5 c/h **5/128 무성 실패** | — | **열린 결함** — 오류 텍스트 없이 proc6 한 워커에서만. 폭 8은 클린이라 폭/패킹 특이로 보이나 대조 미실행. 포렌식은 **블록 34** |
| **`--claim` 청크(1/2/4)** | 592.8 / 840.3 / 978.0 c/h (−49.8 / −28.9 / −17.2 %) | auto 대비 개선 | **기각** — 꼬리는 지우지만(0–8 s) 재청구 왕복 비용이 훨씬 크다. claim=4는 꼬리도 못 지운다(190 s) |
| **12×M8 / 16×M8 / 16×M16** | 944.5 / 1,035.3 / 1,023.9 c/h (−20.1 / −12.4 / −13.4 %) | 8×M8 대비 +5 % | **기각** — MPS 지분 희석(12 → 8 → 6 %) |
| **8×M16 형상 승격** | 1,235.1 c/h (+4.5 %), 팁에서 1,246.2 | 채택 바 **+5 %** | **미승격** — 바 미달이고 `tail_idle_max`가 자기 wall의 95 % |
| **`CUDA_SYNC_MODE=blocking` 기본화** | 배치 **−3.6 %** c/h, 단일 **+8.7 % 느림** | "c/h 손실 없음" | **기본값 아님 · 배치 옵션으로만** |
| **`OUTER_SEGMENT_V2` 배치** | 1,167.8 vs 1,179.1 c/h (**−1.0 %**) | — | **배치에서 제외** (단일에서는 B0 채택) |
| **`CMFD_BLOCK=64` 배치** | 1,176.1 c/h (**−0.3 %**) | — | **단일덱 전용 채택** |

> **덤으로 무너진 전제 하나**: "배치는 설계상 `SEGMENT_V2`를 거부한다"는 전제는 **확인되지 않았다**.
> 1프로세스·폭 1 프로브에서 `rc=0` · `contract_pass:true` · **`refused:0`** · `stop_reason:"shutdown"`으로
> **정상 완주**했다. 8프로세스/MPS 규모에서의 거부 가능성까지 배제하지는 못하므로,
> 본 문서는 이것을 **열린 질문**으로 남기고 배치 arm에서는 보수적으로 SEGMENT_V2를 뺐다.

---

## 8. 다음 레버 — 넷, 그리고 셋은 커널 바깥이다

| # | 레버 | 근거(이 패스가 잰 것) | 등급 | 크기 |
|---:|---|---|---|---|
| **1** | **A2 outer 감축 — W3.6 (flux-space Anderson + interim-Xe)** | 배치가 **GPU 시간 바운드**임이 §6에서 확정됐고, m64 케이스당 **8,415 outer**가 그 시간의 분모다. 단일덱도 9.808 s에서 커널이 아니라 outer 수에 막혀 있다 | **A2** — 별도 브랜치 · 불변식 계약 · 명시 롤백 | **캠페인 최대** (A2 문서의 승수 스캔은 outer −61.6 %를 관측했다) |
| **2** | **host statepoint-phase residency — W5+ Task 13–15** | §4.1(상위 커널 14/15가 2×SM 미만)과 §5.1(`width_fill` 0.45)이 **같은 방향**을 가리킨다: 런치당 독립 작업을 더 담을 것. 상태점 단위로 호스트 왕복을 없애는 것이 그 전제다 | N1 예상 | 중 |
| **3** | **배치의 Xe/CRAM 커널 시간** | 배치 워커 프로파일: `kernelFlatXsCta` 8,174.8 ms · `kXeEvaluate` 5,313.5 ms(격자 **48**, 15,441 런치) · `kCorrector` 4,418.3 ms · `kPredictor` 4,330.2 ms(격자 **133**) · `kXeCommitTxn` 3,890.3 ms(격자 **48**). Xe 계열은 14,892 런치를 24–48블록으로 띄운다 | B0/N1 (구현 의존) | 중 |
| **4** | **멀티GPU dispatcher** | 238에 GPU1이 놀고 있고, §5.4가 **동시성이 곧 처리량**임을 보였다(MPS 1.98×, 폭 1.92×). WP18의 슬롯 단위 admit 기계(§6)가 그 골격으로 이미 서 있다 | 하네스(B0) | 선형에 가까움 |

**#1이 다른 셋을 합친 것보다 크고, 유일하게 A2다.** #2–#4는 **전부 배치 처리량 레버**다.

---

## 9. 보고서가 열어 둔 구멍 (추정으로 메우지 않은 자리)

| # | 구멍 | 왜 아직 답이 없는가 |
|---:|---|---|
| 1 | **배치에서 `colored_block_sweep`/`matvec_two_group`의 실제 격자** | 배치 워커 nsys가 `--cuda-graph-trace=node`를 빠뜨렸다(단일덱 27a만 붙였다). CMFD sweep은 캡처된 그래프(`fuse_mask=15`) 안에서 돌아 리플레이 노드로 접힌 것으로 보인다 — 그 트레이스에 커널명이 **19개뿐**이고 두 커널은 **0행**이다. `--cuda-graph-trace=node`로 재실행해야 답이 나온다 |
| 2 | **nodal jnet H2D 소거 미engage** | `elided_mb 0.0` · `tests 0` · `hit_rate −1.000`. 단일덱에서 한 번도 동작하지 않았다 |
| 3 | **micx `msm` D2H 행 미포착** | A쪽 합이 18.69 GB로 커밋의 22.85 GB보다 낮다. `msm` ~4.05 GB를 더하면 22.74 GB로 반올림 안에서 맞지만 **같은 실행에서 확인되지 않았다** |
| 4 | **배치의 케이스당 바이트** | 배치는 `--result light`로 돌았고 원장을 붙이지 않았다(블록 18/21b의 선례). "~38 → ~2 GB/case" 기대치는 **검증되지 않았다** |
| 5 | **8×M16 + `boron_bracket` 5/128 무성 실패** | 한 워커(proc6)에서만, 오류 텍스트 없이(`[MULTI_GPU][FAIL] gpu0 p6`), 재시도도 실패. 같은 env의 8×M8은 128/128 클린. `boron_bracket`을 끈 8×M16 대조를 돌리지 않았다 → **블록 34 포렌식** |
| 6 | **WP9-D의 `max_ppm 15.334` 반복** | 두 arm에서 기준 15.309를 같은 0.025만큼 넘는다. 기저 노이즈로 보이나 **무플래그 대 MASTER 재실행으로 확인하지 않았다**. BOC 핀 0.238 %/0.80 %가 모든 arm에서 동일한 것도 같은 성질로 읽힌다 |
| 7 | **배치 `graph_arm` 필드** | grep이 `graph_launches`만 안정적으로 뽑았다(값 `0, 19888, 49456, 51` — **OUTER_GRAPH는 배치에서 확실히 engage한다**). `graph_arm` 자체는 값을 적지 않는다 |
| 8 | **181 cross-gate** | `OUTER_GRAPH` · `MICX_RESIDENT` · `WP15.1` · `SEGMENT_V2` · `CMFD_BLOCK` 어느 것도 181에서 재지 않았다. **사용자 지시로 181 계산이 현재 중지 중**이다 |

**계기 관련 부수 발견 하나** — 블록 24/26이 nsys 래퍼와 `pgrep`으로 워커를 못 잡은 근본 원인은
argv 철자였다. 하네스의 실제 자식 argv는 `--evaluator-jsonl - --batch-mode 8 --result light`이고
**`gpu0.pN.chunk` 문자열은 argv에 존재하지 않는다**(잡 큐 파일에만 있다).
그리고 `[MULTI_GPU][PLAN]`이 `procs_per_gpu:8, batch_width:8, declared_width_per_gpu:64`를 찍는다 —
**이 캠페인의 모든 "M8"은 진짜 폭 8 arena**이고, `width_fill ~0.45`는 그 8슬롯 중 평균 3.6개만 찼다는 실제 미충전이다.

---

## 10. 재현

### 10.1 v6 env (238, 단일덱)

`test/reference/validation_baseline_manifest_v3.json` 24행의 PROD env에서 출발해(`env -i`) 다음을 더한다:

```
RASBERY_GPU_CRAM=1 RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1
RASBERY_GPU_OUTER_GRAPH=1 RASBERY_GPU_MICX_RESIDENT=1
RASBERY_GPU_XFER_ELIDE=1 RASBERY_GPU_OUTER_SEGMENT_V2=1
```

**단일덱 최적 arm은 여기에 `RASBERY_GPU_CMFD_BLOCK=64`를 더한 것이다 → 9.808 s.**
`CMFD_BLOCK`은 **배치에 넣지 않는다**(평평하다). `OUTER_SEGMENT_V2`도 **배치에서는 뺀다**(−1.0 %).
`RASBERY_GPU_CMFD_FUSE=15` · `XE_TXN` · `RESULT_ASYNC` · `FLATXS_CTA`는 v5부터 **코드 기본값**이지만
매니페스트와 캠페인 스크립트는 그래도 명시한다(`src/CaseKey.h`가 **원문 문자열**을 다이제스트하므로
unset과 `1`은 서로 다른 `case_key`다 — 캐시 미스이지 잘못된 히트는 아니다).

블록 20이 확인한 것: `7af4099`에서 `RASBERY_GPU_CRAM=1 RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1`
**셋만** 명시하고 나머지를 기본값에 맡겨도 digest `1f36e75dc00ed2b4` / outers 4377로
**완전 명시 v5와 정확히 같은 궤적**이 나온다.

### 10.2 판정 절차

| 무엇 | 명령 |
|---|---|
| 궤적 | `[RASBERY][TRAJECTORY]` digest **`1f36e75dc00ed2b4`** / outers **4377** |
| B0 동일성 | `h5diff -r out_off.h5 out_on.h5`가 아무것도 인쇄하지 않고 exit 0. **핀파워 CSV는 `cmp`로 따로** (h5diff는 CSV를 보지 않는다) |
| Gate A | `tools/gate_a_compare.py --per-step` |
| Gate B | `tools/compare_master_rasbery.py` + `tools/gate_b_pin_rms.py` → **BOC 핀 0.238 % RMS / 0.80 % max** |
| 전송 원장 | `RASBERY_XFER_LEDGER=1`, **단일덱 동반 실행에만** 붙인다(배치 처리량을 왜곡한다) |
| 배치 | `tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 8 --batch-width 8 --mps --result light`, v6 델타는 `--set`으로 |
| grid-sync 게이트 | `tools/probe_gridsync_cost.cu` — `verdict:"NO-GO"`면 persistent 트랙은 **열지 않는다** |

### 10.3 인용하지 말 것

- **nsys를 씌운 배치 875.3 c/h** — 8워커 중 1개만 프로파일해도 나머지 7이 기다린다. 계기 오버헤드다.
- **블록 24의 첫 시도 GPU 사용률 47.5 %(0/100 % 이봉)** — nsys에 교란된 표본이다.
  판정은 깨끗한 dmon 표본(**sm % mean 96.2, p95 100**)으로만 한다.
- **블록 33의 8×M16 1,333.5 c/h** — `cases_per_hour`는 완주 수와 무관하게 `jobs/wall×3600`이므로,
  5케이스가 실패한 이 실행의 c/h는 **실제 완료 작업에 비해 부풀려져 있다**. 같은 블록의 8×M8 1,316.7이 믿을 수다.
- **블록 26의 "74 %는 실작업"** — §5.5가 뒤집었다.
- **블록 27c의 `--batch-mode 1`** — 그것은 argv 철자를 알아내려고 돌린 폭-1 스모크 테스트의 것이지
  8×M8 생산 형상의 것이 아니다(실제 argv는 `--batch-mode 8`).

---

## 11. 매니페스트

**v5는 동결된 채로 둔다**(`test/reference/validation_baseline_manifest_v5.json`, `frozen:true`).
이번 패스의 arm은 **후보**로 별도 파일에 적는다:

**`test/reference/validation_baseline_manifest_v6_candidate.json`** (`frozen:false`)
— v6 + `CMFD_BLOCK=64`, 238 digest `1f36e75dc00ed2b4` / outers 4377 / wall **9.808 s**,
그리고 **181 cross-gate가 비어 있다는 사실**을 `open_items`에 명시한다.
후보가 후보인 이유는 성능이 부족해서가 아니라 **두 번째 호스트가 아직 말하지 않았기 때문**이고,
181 계산은 현재 사용자 지시로 중지 상태다.
