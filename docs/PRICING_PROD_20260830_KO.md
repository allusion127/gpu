# PROD env 가격 평가 — 238 실측 (2026-08-30)

| 항목 | 값 |
|---|---|
| 대상 | GPU PPR(WP6-F) · GPU CRAM(Task 16 / WP9-B) · CMFD FUSE(WP7-B) · Xe TXN(WP7-C) |
| 호스트 | **238 GPU0 전용** (측정 블록마다 사전 점검: `nvidia-smi` GPU0 ~238 MiB / 0 % util, `ps -ef | grep RASBERY` 공백) |
| 브랜치 | `codex/exact-throughput-campaign` |
| 측정 커밋 | `73f8627`(기준선·CRAM·FUSE·pinned TXN) · `926497d`(v3 freeze) · `388e8f2` · `7cfe3a4` · **`91004f7`**(PPR master-mode device port, no-pin TXN, 스택) |
| 덱 | `kngr_238.json` (APR1400/KNGR CY1 PSAR, 35상태 자연 EOC, 1/4 노심 8,451 노드, NG=2), 단일덱, `--batch-mode` 없음 |
| 프로토콜 | arm당 **warm-up 1 + hot 3, median 보고**, 교차 순서. 예외는 본문에 그 자리에서 명시한다 |
| 원천 | 238 러너 보고 `pricing_388e8f2.md` (블록 0–8 + 2b/3b/4b/5b/5c). 블록 9(v4 전체 스택)는 **미완** |
| 기준기 | MASTER (CPU) 단일 27.2 s |

---

## 0. 한 줄 요약

**PROD env 아래에서 세 기능이 값을 냈고 하나는 공짜였다.** 스택(`CRAM=1` + `CMFD_FUSE=15` + `PPR=1`)이
같은 커밋 기준선 16.585 s를 **14.631 s**로 내렸다(−1.954 s, −11.8 %, MASTER 27.2 s 대비 **1.86×**).
PPR은 91004f7 이전까지 production에서 **한 번도 돌지 않았고**(`host_fallbacks:35/35`), CRAM은 궤적을
움직이지만 Gate A/B를 통과했으며, FUSE는 전 mask B0이고, XE_TXN의 B0는 238에서 **직접 감사로** 세웠다
(`forms_audit_mismatch 0/693`). 그리고 이번 패스가 남긴 가장 비싼 교훈은 숫자가 아니라 절차다 —
**매니페스트 env 밖에서 잰 수는 측정이 아니고, 한 기계에서 채굴한 pin은 다른 기계로 옮겨지지 않는다.**

---

## 1. PROD env 정의와 기준선

### 1.1 PROD env

`test/reference/validation_baseline_manifest_v3.json` 24행(`benchmark_input.env`)이 유일한 정의다.
`env -i`로 시작해 이것만 세운다.

```
RASBERY_PPR_MODE=master RASBERY_PC_MODE=decart RASBERY_GPU=1
RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1
RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1
RASBERY_GPU_FLATXS=1 RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8
RASBERY_GPU_WIEL_FOLD=chunked RASBERY_GPU_XE=1
RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1
RASBERY_OMP_THREADS=12
```

`RASBERY_XE_ANDERSON`은 **의도적으로 부재**(단일덱 기본 ON), `RASBERY_STATEPOINT_TELEMETRY`도 타이밍
반복에서 **부재**.

### 1.2 기준선 — 드리프트 없음 (블록 1, 6)

| 커밋 | hot1 | hot2 | hot3 | **median (s)** | digest | outers |
|---|---:|---:|---:|---:|---|---:|
| `73f8627` (현재 팁 계열) | 16.769 | 16.772 | 16.789 | **16.772** | `0d15abf29d222a02` | 4382 |
| `926497d` (v3 freeze) | 16.763 | 16.756 | 16.773 | **16.763** | `0d15abf29d222a02` | 4382 |
| `388e8f2` | 16.785 | 16.765 | 16.811 | **16.785** | `0d15abf29d222a02` | 4382 |
| `7cfe3a4` | 16.373 | 16.381 | 16.364 | **16.373** | `0d15abf29d222a02` | 4382 |

`h5diff` 73f8627 vs 926497d = **0 차이**(전 데이터셋), 73f8627 vs 7cfe3a4 = **0 차이**.

**판정: 기준선은 온전하다.** `926497d → 388e8f2 → 73f8627` 사슬 어디에도 wall 드리프트가 없다
(세 median이 0.022 s 안에 있다). 회귀 이등분 트리거(≥2 s)는 이 캠페인에서 **한 번도 발화하지 않았다**.

### 1.3 "arm X" 숫자는 env 아티팩트였다 (폐기)

같은 파일 앞부분에 기록된 arm-X 계열 수 — PPR off **23.98 s** / on **16.75 s**, outers **4566**,
digest `22b9a3187bfb4beb` — 는 드리프트도 회귀도 아니다. 그 arm이
`RASBERY_PC_MODE=decart` · `RASBERY_PPR_MODE=master` · `RASBERY_OMP_THREADS=12` **세 개를 빠뜨린**
env였을 뿐이다. 블록 2b가 이를 재현으로 확정했다(§3.2 arm C: 둘 다 unset → `22b9a3187bfb4beb`/4566,
Gate B 0.522 %/2.13 %). **아래 어떤 판정에도 arm-X 수를 쓰지 않는다.**

---

## 2. 기능별 가격표 (요약)

| 기능 | env | 등급 | 기준 arm (s) | 채택 arm (s) | Δ (s) | digest / outers | 게이트 | 판정 |
|---|---|---|---:|---:|---:|---|---|---|
| **GPU PPR** (WP6-F, `91004f7`) | `RASBERY_GPU_PPR=1` (+`_PPR_GRAPH=1`) | **B0** (pin 계열만 N1) | 16.885 | **15.024** (GRAPH) | **−1.861** | `0d15abf29d222a02` / 4382 | runbook §10.5 **7/7 PASS** | **채택 후보** |
| **GPU CRAM** (Task 16) | `RASBERY_GPU_CRAM=1` | **N1** (궤적 이동) | 16.729 | **16.043** | **−0.686** | `1f36e75dc00ed2b4` / **4377** | Gate A PASS · Gate B PASS · 결정론 PASS | **채택 후보** |
| **CMFD FUSE** (WP7-B) | `RASBERY_GPU_CMFD_FUSE=15` | **B0** (전 mask) | 16.724 | **16.328** | **−0.396** | `0d15abf29d222a02` / 4382 | B0 6/6 mask · 채택 문턱 0.2 s 통과 | **채택 후보 (mask 15)** |
| **Xe TXN** (WP7-C, `91004f7`) | `RASBERY_GPU_XE_TXN=1` | **B0** | 16.791 | 16.652 | −0.139 (단발, 노이즈 안) | `0d15abf29d222a02` / 4382 | `forms_audit_mismatch` **0/693** | **B0 확립 · wall 중립** |
| **스택** | 위 셋 동시 | N1 (CRAM이 지배) | 16.585 | **14.631** | **−1.954 (−11.8 %)** | `1f36e75dc00ed2b4` / 4377 | Gate B pin 0.238 %/0.80 % | **v4 후보** |

> **PPR OFF(16.885 s)는 단발 실행이다** — 블록 7에서 OFF arm만 hot 1회였다. −1.861 s는 median 대
> 단발의 차이이므로, 반복된 기준선(블록 1의 16.772 median)에 대면 −1.75 s다. 어느 쪽이든 노이즈
> (관측 폭 ±0.3 s)의 여섯 배 이상이다.

---

## 3. GPU PPR — 거절이 이름을 갖기까지 (WP6-F)

### 3.1 블록 2: production에서 값이 0이었다 (`73f8627`)

| arm | median (s) | digest | outers | `host_fallbacks` |
|---|---:|---|---:|---:|
| off | 16.781 | `0d15abf29d222a02` | 4382 | — |
| on (기본, device_stream) | 17.054 | 동일 | 4382 | **35/35** |
| on + `_PPR_GRAPH=1` | 16.413 | 동일 | 4382 | **35/35** |
| on + `_PPR_CANONICAL=verify` | 16.428 | 동일 | 4382 | **35/35** |
| on + `_PPR_RECON=1` | 16.427 | 동일 | 4382 | **35/35** |

네 ON arm 전부 `statepoints:0, iterations:0, host_syncs:0, graph_launches:0, wall_ms:0.000` —
**device PPR 루프가 한 번도 돌지 않았다.** 그래서 `canonical_mismatch:0`·`recon_repairs:0`은
통과가 아니라 **단계가 engage하지 않아 자명하게 0**인 값이다. 이 블록의 wall은
"PPR device 경로의 비용"이 아니라 "매 statepoint 들어갔다 거절하는 비용"으로 읽어야 한다.

### 3.2 블록 2b: 방아쇠 격리 — `RASBERY_PPR_MODE=master`

| arm | PROD 대비 유지한 변수 | digest | outers | `host_fallbacks` | `contract_pass` | Gate B pin rms/max |
|---|---|---|---:|---:|---|---|
| A | `RASBERY_PC_MODE=decart`만 | `0d15abf29d222a02` | 4382 | **0/35** | true | 0.522 % / 2.13 % |
| B | `RASBERY_PPR_MODE=master`만 | `22b9a3187bfb4beb` | 4566 | **35/35** | **false** | **0.238 % / 0.80 %** |
| C | 둘 다 없음 | `22b9a3187bfb4beb` | 4566 | 0/35 | true | 0.522 % / 2.13 % |

**`RASBERY_PPR_MODE=master`가 방아쇠다. `RASBERY_PC_MODE`가 아니다.** 그리고 master 모드가 바로
production configuration이다(Gate B pin RMS 0.238 %/0.80 %가 그것을 증명한다 — 기본 모드는
0.522 %/2.13 %).

원인은 `src/PPR.cpp:79`가 지금 주석으로 인용하고 있는 그 한 줄, `if (_mode_master) return false;`.

**수신증이 자기 거절을 부르지 않았다는 것이 이 결함의 수명을 결정했다.** 로그 전수 grep 결과,
실패 arm의 `[RASBERY][GPU_FULL]`은 `ppr_fallbacks:35, "contract_pass":false, "first_violation":null`
을 인쇄했다 — **`contract_pass`가 false인데 그 옆이 `null`이었다.** 사람이 읽을 거절 사유 문자열은
로그 어디에도 없었고, 신호는 `host_fallbacks` 카운터 하나뿐이었다.

### 3.3 블록 7: `91004f7` 이후 — master 모드가 device로 갔다

`91004f7`이 MASTER MM §6.1을 커널 다섯 개(`kMasterEven`/`kMasterCpb`/`kMasterCommit`/
`kMasterFoldAndCheck`/`kMasterCross`)로 이식하고, `ppr::Refusal` 거절 사다리와 `GPU_FULL`의
`first_violation` 수정을 함께 넣었다. 새 트리 빌드: cmake rc=0, build rc=0, **ctest 12/12**.

| arm | hot1/hot2/hot3 (s) | median (s) | digest | outers | `host_fallbacks` | `iterations` | `refusal` |
|---|---|---:|---|---:|---:|---:|---|
| OFF (단발) | 16.885 | 16.885 | `0d15abf29d222a02` | 4382 | — | — | — |
| ON (device_stream) | 15.372 / 15.205 / 15.206 | **15.206** | 동일 | 4382 | **0/35** | 766 | `none` |
| ON + GRAPH | 15.154 / 15.024 / 15.010 | **15.024** | 동일 | 4382 | **0/35** | 766 | `none` |

**OFF 16.885 → ON 15.206 (−1.679 s, −9.9 %) → ON+GRAPH 15.024 s (−1.861 s, −11.0 %).**
receipt 내부 PPR 루프 `wall_ms`: device_stream ~144–149 ms, device_graph **~106–107 ms**
(graph capture가 statepoint 34개분의 launch overhead를 걷어내며 ~40 ms를 깎는다).

**runbook `docs/WP6_PPR_DEVICE_LOOP_20260831_KO.md` §10.5의 7개 조건 전부 PASS:**

| # | 조건 | 실측 |
|---|---|---|
| 1 | `host_fallbacks:0` · `refusal:"none"` · `refusals:{}` | 전 ON receipt에서 충족 |
| 2 | `iterations != 0`, 비퇴화·비포화 | **766** (cap 200×35=7000). master CPB는 Jacobi이므로 SENM host의 816과 같을 이유가 없다 |
| 3 | `h5diff` OFF vs ON = pin 계열만 | **37 차이**: `steps/0028..0035/pin_power`(각 355,475, `print_opt.pin_flux`가 켜진 8개 statepoint) + `summary/fqp`(35) + `summary/frp`(35). **궤적 digest 동일** |
| 4 | ON×2 결정론 | `h5diff` **0 차이**, 전 데이터셋 |
| 5 | Gate B가 인쇄 자릿수까지 재현 | OFF 0.238 %/0.80 %, ON **0.238 %/0.80 %** (0.522 %/2.13 %가 아니다 = device가 master 플래그 아래 SENM을 돌지 않았다) |
| 6 | `reallocations:0` | 충족. `allocations:28` = 기준 26 + `phic_next`/`mrel` 2 (§10.4 예측 그대로) |
| 7 | `RASBERY_GPU_PPR_RECON=1` | `recon_statepoints:35`, `recon_repairs:0`, `recon_refusal:""`, `pin_materializations:35`. OFF vs RECON=1 h5diff가 OFF vs ON과 **같은 37 차이** — 재구성이 pin 발산을 추가하지 않는다 |

독립 확인 `RASBERY_GPU_FULL=1`:
`{"cmfd_fallbacks":0,"outer_fallbacks":0,"nodal_fallbacks":0,"flatxs_fallbacks":0,"xe_fallbacks":0,`
`"ppr_fallbacks":0,"cram_fallbacks":0,"allowed_refusals":{"wielandt_warmup":73},`
`"contract_pass":true,"first_violation":null}` — 전 서브시스템 fallback 0, `contract_pass:true`.

**판정: 채택 후보. GRAPH arm이 둘 중 낫다.** 블록 2의 결론은 **전면 반전**된다 — 그 블록의 빌드는
`91004f7` 이전이고 device 경로를 한 번도 실행한 적이 없다.

---

## 4. GPU CRAM — 궤적을 움직이고, 게이트를 통과한다 (Task 16 / WP9-B)

### 4.1 블록 3: wall과 궤적 (`73f8627`)

| arm | hot1 | hot2 | hot3 | median (s) | digest | outers |
|---|---:|---:|---:|---:|---|---:|
| off | 16.792 | 16.729 | 16.569 | **16.729** | `0d15abf29d222a02` | 4382 |
| `RASBERY_GPU_CRAM=1` | 16.019 | 16.043 | 16.051 | **16.043** | **`1f36e75dc00ed2b4`** | **4377** |

**Δ = −0.686 s.** receipt(`[RASBERY][CRAM_GPU]`, hot1):
```
{"schema_version":1,"slot":0,"statepoints":34,"predictor_calls":34,"corrector_calls":34,
 "nodes":574668,"device":0,"host_fallbacks":0,"gs_iters_mean":2.384,"gs_solves":2298672,
 "micx_h2d_mb":1367.9,"bos_reuse":34,"wall_ms":234.404,"status":"on"}
```
`host_fallbacks:0` — **device 경로가 실제로 돌았다**(블록 2의 PPR과 정반대).

`h5diff` on vs off = **419 차이** (`summary/tm_avg` 34, `tm_max` 34, `xe_ao` 34, `xe_avg` 34 등).
**이것은 결함 신호가 아니다.** CRAM은 depletion predictor/corrector 대수이고, device가 쓴 동위원소
재고를 다음 statepoint가 읽으므로 궤적이 움직이는 것이 **설계**다(outer 5개 감소). 등급은 처음부터
**N1**이며, 따라서 B0가 아니라 Gate A/B로 판정한다.

### 4.2 블록 3b: Gate A / Gate B / 결정론

**Gate A** (on vs off 궤적, 35 statepoint, `gate_a_compare.py --per-step`):

| 지표 | 실측 | 한도 |
|---|---:|---:|
| max \|dkeff\| | **0.0653 pcm** (sp29) | 5 |
| max \|dppm\| | **0.0038 ppm** | 5 |
| max \|dAO\| | **0.0001** | 0.01 |
| max pin rel | **0.0211 %** | 1 % |

전부 A2 스크린 **안**이고, 코디네이터가 인용한 봉투(1.905 pcm / 15.309 ppm / 0.013 AO)보다도 훨씬
안쪽이다. → **Gate A: PASS**.

**Gate B** (CRAM on vs MASTER, `compare_master_rasbery.py`, 33 상태 조인):

| 지표 | 실측 | v2 기준선-vs-MASTER 봉투 |
|---|---:|---:|
| max \|Δppm\| | **15.334** | 15.309 |
| max \|Δpcm\| | **1.847** | 1.905 |
| max \|ΔAO\| | **0.012** | 0.013 |
| max \|Δfqp\| | 0.071 | — |
| max \|Δfrp\| | 0.017 | — |
| BOC pin RMS / max | **0.238 % / 0.80 %** | 0.238 % / 0.80 % |

즉 **CRAM은 코드가 이미 갖고 있던 MASTER와의 불일치에 측정 가능한 양을 더하지 않는다.**
→ **Gate B: PASS**.

**결정론**: on hot1 vs hot2 = 0 차이, hot2 vs hot3 = 0 차이 → **PASS**.

### 4.3 receipt가 남긴 잔여 기회 — BOS micx 재업로드

`micx_h2d_mb:1367.9`은 34 statepoint의 **런 총계**이고(`predictor_calls` =
`corrector_calls` = `bos_reuse` = 34, statepoint당 하나씩), statepoint당 **~40.2 MB**다.
`bos_reuse:34`가 statepoint 수와 같다는 것은 beginning-of-step 버퍼가 **resident로 남지 않고
매 statepoint 다시 전송된다**는 뜻이다. 40 MB × 34 = **실재하는 residency 기회**이며 §9에 후보로
올린다(이번 패스는 측정 전용이므로 손대지 않았다).

---

## 5. CMFD FUSE — 전 mask가 공짜다 (WP7-B)

### 5.1 블록 4: mask 전수 (단발, warm-up 없음)

| arm | wall (s) | digest | outers | `fuse_mask` |
|---|---:|---|---:|---:|
| base (FUSE 없음) | 21.004 | `0d15abf29d222a02` | 4382 | — |
| `FUSE=1` | 17.386 | 동일 | 4382 | 1 |
| `FUSE=2` | 19.232 | 동일 | 4382 | 2 |
| `FUSE=4` | 16.993 | 동일 | 4382 | 4 |
| `FUSE=8` | 16.992 | 동일 | 4382 | 8 |
| `FUSE=15` | 16.546 | 동일 | 4382 | 15 |

**B0 확정: 여섯 digest가 전부 같고 outers도 전부 4382.** mask와 무관하게 비트가 같다.

이 블록의 wall은 arm당 전용 warm-up이 없어 신뢰하지 않는다(base 21.004 s와 fuse_2 19.232 s는
cold-start 이상치다). 수신증 태그도 정정해 둔다 — 실제 태그는 별도의 `[CMFD][FUSE]`가 아니라
**`[RASBERY][CMFD][GRAPH]`**이고 그 안에 `"fuse_mask":N`이 들어 있다.

### 5.2 블록 4b: 교차 재측정 (warm-up 1 + hot 3, base→fuse4→fuse15 ×3)

| arm | warm | hot1 | hot2 | hot3 | **median (s)** | base − median |
|---|---:|---:|---:|---:|---:|---:|
| base | 16.836 | 16.754 | 16.724 | 16.515 | **16.724** | — |
| `FUSE=4` | 16.728 | 16.724 | 16.541 | 16.507 | **16.541** | 0.183 |
| `FUSE=15` | 16.498 | 16.531 | 16.264 | 16.328 | **16.328** | **0.396** |

digest 셋 다 `0d15abf29d222a02` (B0 재확인).

**채택 규칙(median이 base보다 ≥0.2 s 낮을 것): `FUSE=4`는 미달(0.183), `FUSE=15`는 통과(0.396).**
→ **`RASBERY_GPU_CMFD_FUSE=15`를 기본값 후보로 채택.**

---

## 6. Xe TXN — pin을 폐기하고 감사로 세운 B0 (WP7-C)

### 6.1 블록 5: pin이 있는 arm과 없는 arm (단발)

| arm | 트리 | wall (s) | digest | outers | XE_FORMS |
|---|---|---:|---|---:|---|
| `TXN=0`, `RASBERY_XE_FORMS=0xadd` | `73f8627` | 16.850 | `3ceb8713b5d73f8f` | **4374** | 0xadd / env |
| `TXN=1`, `RASBERY_XE_FORMS=0xadd` | `73f8627` | 16.660 | `3ceb8713b5d73f8f` | **4374** | 0xadd / env |
| `TXN=0`, **pin 없음** | `91004f7` | 16.791 | **`0d15abf29d222a02`** | **4382** | 합성 |
| `TXN=1`, **pin 없음** | `91004f7` | 16.652 | **`0d15abf29d222a02`** | **4382** | 합성 |

`h5diff` TXN=1 vs TXN=0: 두 트리 모두 **0 차이**. 그리고 `91004f7`의 pin-없는 arm은
**PROD 기준선을 바이트 그대로 재현**한다(블록 1의 `73f8627` PROD-off와 교차 트리 h5diff 0 차이).

### 6.2 블록 5b: 4374 vs 4382의 원인은 **pin이지 TXN이 아니다**

- **env 확인**: `[RASBERY][TRAJECTORY]` 전체 덤프 대조 결과, 의도적으로 세운 변수 외 **전 필드 동일**.
- **`RASBERY_GPU_XE_TXN`을 아예 unset**한 `91004f7` 실행: `0d15abf29d222a02`/4382, 명시적 `TXN=0`과
  h5diff **0 차이** — 0이 문서화된 기본값이라는 사실 그대로.
- **pin된 `TXN=0`(73f8627) vs PROD off(73f8627)**: **410 차이**. `TXN=0`은 `kXeAndersonSolve`에
  닿지도 않는데 궤적이 움직였다.

**근본 원인.** `0xadd`의 하위 5비트(`XE_SHIPPED_FORMS`, `0xadd & 0x1F = 0x1D`)는 **호스트 181에서**
181 자신이 채굴한 `0xd3d`(`& 0x1F = 0x1D`)로부터 합성된 값이다. 238은 같은 shipped 비트를
**다르게** 채굴한다: `0xd2d & 0x1F = 0xD`. shipped 비트는 `RASBERY_GPU_XE=1`이면 **TXN과 무관하게
항상** 도는 device dot-product/candidate-loop 형태를 고르므로, 181의 `0xadd`를 238에 못박으면
238이 옳게 채굴한 0xD를 181의 0x1D로 덮어쓴다. 이것이 4382 → 4374다.

→ **`RASBERY_XE_FORMS=0xadd`는 교차 기계 아티팩트로 폐기한다.** 181 보고서
(`gates_8919331.md` §(A))가 "최소 수정"으로 권고한 그 pin은 **181에서만** 참이다.
238에서는 pin을 **걸지 않는 것**이 옳다.

### 6.3 블록 5c: 합성값 정정과 `forms_audit`

블록 5 표가 "no-pin arm은 `0xd2d`/mined로 해석된다"고 적은 것은 **불완전한 grep**이었다
(바이너리가 인쇄하는 세 FORMS 줄 중 첫 줄만 잡았다). 전문:

```
[RASBERY][FORMS] {"mask":"XE_FORMS","value":"0xd2d","source":"mined","build_default":"0xd",
                  "mined":"0xd2d","mined_sound":1}
[RASBERY][FORMS] {"mask":"XE_HOST_FORMS","value":"0xac0","source":"build_default",
                  "build_default":"0xac0","det":2,"g0":1,"g1":1,"proj":1}
[RASBERY][FORMS] {"mask":"XE_FORMS","resolved":"0xacd","source":"build_default_composed",
                  "mined":"0xd2d","host":"0xac0","composed":"0xacd","shipped":"0xd",
                  "algebra":"0xac0","live_arm":"shipped","txn_arm":"resolved","algebra_sound":1}
```

첫 줄은 **합성 전** shipped-only 채굴 중간값이고, **마지막 줄이 실제 해석**이다:
`resolved:"0xacd"`, `source:"build_default_composed"` — 정확히
`(mined 0xd2d & 0x1f) | (host 0xac0) = 0xd | 0xac0 = 0xacd`. **합성은 적용되고 있다.**

**`forms_audit` (PROD + `RASBERY_GPU_XE_TXN=0` + `RASBERY_XE_FORMS_AUDIT=1`, pin 없음, `91004f7`):**

```
"forms_audits":693, "forms_audit_mismatch":0,
"forms_audit_mask":"0xacd", "forms_audit_host_mask":"0xac0"
```

감사 호출 지점은 `src/XeFormAudit.h`의 주석이 말하는 대로 **`TXN=0` host 경로**에만 있다
(첫 시도를 `TXN=1`로 돌려 `forms_audits:0`을 받은 것은 no-op이었다). digest/outers는 불변
(`0d15abf29d222a02`/4382 — 감사는 측정 전용).

`forms_audit_mask(0xacd) & XE_ALGEBRA_FORMS(0xfe0) = 0xac0 = forms_audit_host_mask` — 합성된 device
mask의 대수 채널이 host의 것과 정확히 일치하고, **693개 Anderson-fit 스텝 전부에서 mismatch 0**이다
(693은 같은 receipt의 `anderson_proposed`와 일치 = 제안된 모든 스텝이 감사되었다).

→ **238에서 `RASBERY_GPU_XE_TXN`의 B0는 추론이 아니라 단계별 비트 감사로 확립되었다.**
238의 TXN 0/1 동일성은 우연이 아니다.

---

## 7. 스택 — v4 후보의 실측 (블록 8, `91004f7`)

교차 순서 a→b→c ×3, warm-up 1 + hot 3.

| arm | warm | hot1 | hot2 | hot3 | **median (s)** | digest | outers |
|---|---:|---:|---:|---:|---:|---|---:|
| (a) base | 16.889 | 16.851 | 16.585 | 16.500 | **16.585** | `0d15abf29d222a02` | 4382 |
| (b) `CRAM=1` + `FUSE=15` | 16.219 | 16.191 | 16.008 | 15.820 | **16.008** | `1f36e75dc00ed2b4` | **4377** |
| (c) (b) + `PPR=1` | 14.807 | 14.732 | **14.631** | 14.481 | **14.631** | `1f36e75dc00ed2b4` | **4377** |

- (b)와 (c)가 **둘 다 CRAM 자신의 서명**(`1f36e75dc00ed2b4`/4377)에 앉는다 — FUSE와 PPR은 궤적을
  더 움직이지 않고 그 위에 올라탄다.
- `h5diff` (b) vs (a) = **419 차이** — 블록 3의 CRAM 단독 서명과 같은 데이터셋·같은 개수.
- `h5diff` (c) vs (b) = **37 차이** — `steps/0028..0035/pin_power`(각 355,475) + `fqp`(35) +
  `frp`(35), 블록 7의 PPR 서명과 **바이트 단위로 같은 집합**.
- (c)의 Gate B pin RMS = **0.238 % / 0.80 %** — 스택 상태에서도 production 기준을 정확히 재현.
- (c)의 `[PPR_GPU]` `host_fallbacks:0/35`, `refusal:"none"`; `[CRAM_GPU]` `host_fallbacks:0`.

**스택 wall: 16.585 → 16.008 (−0.577) → 14.631 s (−1.954 s, −11.8 %).
MASTER 27.2 s 대비 1.86×.**

> **주의 — 이 14.631 s arm에는 `RASBERY_GPU_PPR_GRAPH=1`도 `RASBERY_GPU_XE_TXN=1`도 없다.**
> 둘은 §3.3·§6에서 **따로** 가격이 매겨졌다(GRAPH는 PPR 위에 −0.18 s, TXN은 B0·wall 중립).
> §8의 v4 후보 env는 그 둘을 포함하므로 **아직 통째로 측정된 적이 없다** — 블록 9가 그것을
> 재기로 되어 있었고 **미완**이다. 만들어진 후보 매니페스트는 이 사실을 명시한다.

---

## 8. v4 후보 env

**PROD(§1.1) + 다음 다섯.**

```
RASBERY_GPU_CRAM=1
RASBERY_GPU_CMFD_FUSE=15
RASBERY_GPU_PPR=1
RASBERY_GPU_PPR_GRAPH=1
RASBERY_GPU_XE_TXN=1
```

파일: `test/reference/validation_baseline_manifest_v4_candidate.json`
(`frozen:false`, `status:CANDIDATE`, v3와 **동일 스키마**).

| 항목 | 값 | 비고 |
|---|---|---|
| 측정 커밋 | `91004f7` | |
| digest | `1f36e75dc00ed2b4` | CRAM 서명 |
| outers | **4377** | v3 4382에서 −5 |
| 단일덱 wall | **14.631 s** | MASTER 27.2 s 대비 **1.86×** |
| 등급 | **N1** (CRAM이 지배; PPR/FUSE/TXN은 각각 B0) | |

**`181 cross-gate pending` 플래그가 붙는 항목 셋:**

| 기능 | 238 상태 | 181 상태 |
|---|---|---|
| **CRAM** | Gate A/B PASS, 결정론 PASS | G0/G1/G2/G4 PASS. **G2 밀도 비교기 미실행**(덱에 `isotope_density` 데이터셋이 없다), **G5 스킵**(kngr3/CY01/CY02 덱이 181에 없다) |
| **PPR** | runbook 7/7 PASS (`91004f7`) | **미실시** — 181 게이트 패스는 `8919331`이고 `91004f7`을 담지 않는다 |
| **XE_TXN** | B0 확립 (`forms_audit` 0/693) | `8919331`에서 **B0 FAIL**(digest 갈림, Xe step 1190 vs 1195). 원인은 채굴 mask `0xd3d` vs host `0xac0` 발산이고, 합성 수정(`d25efe6`)이 담긴 트리로는 **아직 재측정되지 않았다** |

**FUSE에는 플래그를 붙이지 않는다** — 181이 `8919331`에서 6개 mask 전부 digest 동일·h5diff 0을
확인했고, graph census node 수(90/98, 82/90, 86/94, 90/97, 90/98, 78/85)까지 238 문서 예측표와
정확히 일치했다(census는 config 유래이지 호스트 타이밍 유래가 아니다).

---

## 9. 다음 병목 후보 (측정으로 열린 것)

| # | 후보 | 근거 | 크기 추정 |
|---|---|---|---|
| **1** | **CRAM BOS micx residency** | `bos_reuse:34` = statepoint 수. BOS 버퍼가 resident로 남지 않고 statepoint마다 40.2 MB를 재전송한다(총 1,367.9 MB) | CRAM `wall_ms` 234 ms 안에서의 전송분 |
| **2** | **outer/CMFD 루프 자체** | 14.631 s / **4377 outer = ~3.3 ms/outer**. PPR 루프가 106 ms까지 내려온 지금, 남은 14.6 s를 지배하는 것은 outer 루프다 | 지배적 |
| **3** | **A2 — outer 수 감소** | 위의 직접적 귀결. 커널을 더 빠르게 하는 것보다 **outer를 덜 도는 것**이 이제 더 큰 레버다 (`docs/A2_OUTER_REDUCTION_20260829_KO.md`) | outer −61.6 % 스캔 기록 있음 |
| **4** | 배치 tail / `width_fill` | 8×M8+MPS의 `width_fill` 0.41 — 단일 wall이 내려가면 배치 tail의 상대 지분이 오른다 | 배치 878 c/h 축 |
| **5** | `7cfe3a4`가 이웃보다 0.4 s 빠른 건 | 블록 1: `7cfe3a4` 16.373 vs `926497d`/`388e8f2`/`73f8627` 16.76–16.79. `388e8f2`가 `73f8627` 쪽에 앉으므로, 실재한다면 `926497d`↔`7cfe3a4` 사이다 | **낮은 우선순위** — 0.4 s(~2.4 %)는 전 블록에서 관측된 통상 jitter 폭과 같은 크기 |

**PPR은 더 이상 후보가 아니다** — device 루프가 106 ms이면 14.6 s의 0.7 %다.

---

## 10. 교훈 (이 패스가 비싸게 산 것)

**1. 매니페스트 env 밖에서 잰 수는 측정이 아니다.**
arm-X의 "PPR off 23.98 s / 4566 outers"는 회귀도 드리프트도 아니었고, 단지
`RASBERY_PC_MODE=decart`·`RASBERY_PPR_MODE=master`·`RASBERY_OMP_THREADS=12`가 빠진 셸이었다.
그 하나의 누락이 **이등분 조사를 부를 뻔했고**, 실제로 블록 2 전체를 "PPR이 비싸다"는 잘못된 결론으로
끌고 갔다. **arm은 `test/reference/validation_baseline_manifest_v3.json`의 `env` 한 줄에서만
나온다. `env -i`로 시작하고, `[RASBERY][TRAJECTORY]` 덤프로 사후 확인한다.**

**2. 한 기계에서 채굴한 pin은 다른 기계로 옮겨지지 않는다.**
`RASBERY_XE_FORMS=0xadd`는 181에서 정당하게 유도되었고 181에서 정확히 동작했다. 같은 값을 238에
못박자 238이 옳게 채굴한 shipped 비트를 덮어써 궤적을 4382 → 4374로 옮겼고, 그것도 **`TXN=0`에서**
움직였다 — 즉 문제의 비트는 검사하려던 기능의 게이트 **바깥**에 있었다. **기계별로 채굴되는 값은
합성으로 해결하지 손 pin으로 해결하지 않는다**(`d25efe6`의 `build_default_composed`가 그것이다).
그리고 교차 기계 재현을 주장할 때는 **어느 기계가 무엇을 채굴했는지**를 함께 적어야 한다.

**3. 수신증은 자기 거절의 이름을 대야 한다.**
`if (_mode_master) return false;`는 **캠페인 하나가 통째로 지나가도록** 살아남았다. 살아남은 이유는
버그가 교묘해서가 아니라, 그것이 인쇄한 것이 `host_fallbacks:35`라는 **숫자 하나**뿐이었기 때문이다.
`contract_pass:false` 옆의 `first_violation`은 `null`이었고, 로그 전수 grep에도 사람이 읽을 사유
문자열이 없었다. **거절은 이름을 남겨야 하고(`ppr::Refusal` 사다리), `contract_pass:false`는 반드시
site와 rung을 동반해야 한다**(`91004f7`이 둘 다 고쳤다).

**4. 자명한 0과 통과한 0을 구분해서 적는다.**
블록 2의 `canonical_mismatch:0`·`recon_repairs:0`은 통과가 아니라 **단계가 engage조차 하지 않아서
0**이었다. 러너가 이것을 "검증되지 않음"으로 표시하고 device 경로 수치를 지어내지 않은 것이
이 패스에서 가장 값진 절차적 판단이다.

**5. warm-up 없는 단발 wall로는 arm 사이 순위를 매기지 않는다.**
블록 4의 base 21.004 s / `FUSE=2` 19.232 s는 전부 cold-start 이상치였고, 교차 재측정(블록 4b)에서
같은 arm들이 16.3–16.7 s에 앉았다. **B0 판정은 단발로도 되지만(digest는 jitter를 타지 않는다),
채택 판정은 warm-up 1 + hot 3 + 교차 순서 없이는 하지 않는다.**

---

## 부록 A — 이 보고서에서 인용하지 않는 수

| 수 | 왜 쓰지 않는가 |
|---|---|
| PPR off 23.98 s / on 16.75 s, outers 4566, digest `22b9a3187bfb4beb` | arm-X env 아티팩트 (§1.3) |
| `RASBERY_XE_FORMS=0xadd`, digest `3ceb8713b5d73f8f`/4374 | 181에서 채굴된 pin을 238에 강제한 결과 (§6.2) |
| 블록 4의 base 21.004 s, `FUSE=2` 19.232 s | warm-up 없는 cold-start 이상치 (§5.1) |
| 블록 2의 `canonical_mismatch`/`recon_repairs`/`graph_launches` | device 경로가 돌지 않아 자명하게 0 (§3.1) |
| 181의 wall·c/h 전부 | 181은 정확도 전용 호스트다. 성능 주장을 내지 않는다 |
