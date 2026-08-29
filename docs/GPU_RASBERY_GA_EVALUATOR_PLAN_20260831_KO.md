# GA evaluator 계획 — 노달 코드를 2.56M 평가에 맞추는 실측 실행안 (2026-08-31)

브랜치 `codex/exact-throughput-campaign`, 기준 `feff7e7`
측정 바이너리: `feff7e7` 트리 + 당시 미커밋이던 `src/` 작업본(이후 `f791f75..e85d0fb`로 착지).
로컬 실측: WSL, GTX 1080 Ti(sm_61) 1장, nvcc 12.6 / gcc 13.3, 24 코어, `kngr_238.json` 35상태
238 인용치: `docs/V3_FREEZE_20260829_KO.md` §4, `docs/CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md`, 캠페인 기록
신규 도구: `tools/case_cost_profile.py`(본 커밋), `tools/make_screening_deck.py`(기존 미커밋본, §5.3에서 수정)

> **이 문서가 하지 않는 것**: GA를 설계하지 않는다. 선택·교차·변이·캐시 정책·다충실도 정책은
> 전부 GA의 것이다. 이 문서는 **evaluator가 GA에게 무엇을 제공해야 하는가**와
> **그 제공을 위해 노달 코드에서 무엇을 재야 하고 무엇을 고쳐야 하는가**만 다룬다.
> `GA_deep_research_report.md`가 요구한 architecture 항목(상주 evaluator, batch LP API,
> warm-start flux, 비동기 결과 인터페이스, scalar-only I/O, duplicate cache **키**,
> island/multi-GPU, FP ranking stability)을 evaluator 쪽 계약으로 번역한 것이 §5다.

---

## 0. 결론 먼저

### 0.1 레버 요약표

| # | 레버 | 배수(1-GPU 처리량) | 근거 | 게이트 | Wave |
|---|---|---|---|---|---|
| **L1** | **scalar-only 결과**(`RASBERY_BATCH_LIGHT_RESULT=1`, **이미 존재**) | 처리량 **1.06×**(단일) / **1.00×**(배치, 실측) — **처리량 레버가 아니다** | **필수 조건**: 2.56 M 케이스 × 301.6 MB = **772 PB** vs × 25.1 kB = **64 GB**. 궤적 digest 동일(§2.4) | **B0-output**(digest 동일 — 이미 실측) | **W1** |
| **L2** | **persistent evaluator**(라이브러리·지오메트리 프로세스 상주, 케이스 = LP + 파라미터) | 단일 1.03–1.06×, 배치 **1.2–1.6×**(실측 근거 §3.1(a)) | 8케이스·폭 4 배치에서 `[HDF5][LOCK].wait_ms` = **15,179 ms**(전 출력) / **36,890 ms**(scalar-only, **쓰기 0**). 케이스당 `Init+IO` 0.86–1.63 s(단일) → **1.31–15.43 s**(배치) | **B0**(bit-golden) | **W2** |
| **L3** | **statepoint 스케줄을 evaluator 파라미터로**(coarse 10상태) | **2.24×**(실측, 로컬) | outer 4,609 → **2,334**, drive 42.07 → **18.77 s** | **A2**(답이 바뀐다 — 정책은 GA의 것) | **W1**(API), 정책은 GA |
| **L4** | **단일 경로 물리**(WHILE capture, nodal-constants N1, A2 2단, GPU PPR) | **1.5–2.5×** | outer/상태 238 **89** vs MASTER **59**(1.51× 여유) + 상태점당 호스트 바닥 **0.474 s**(§2.2) | B0 / N1 / A2 혼합 | **W3** |
| **L5** | **배치 폭·도착**(slot compaction 또는 GPU당 다중 프로세스+MPS) | **1.4–2.2×**(compaction) / **1.2–1.8×**(다중 프로세스) | `mean_width` **14.5/64 = 22.7 %**, 커널 grid는 **선언 폭**에 비례 | N1(compaction) / B0(다중 프로세스) | **W2**(프로세스) / **W4**(compaction) |
| **L6** | **multi-GPU island**(`tools/run_multi_gpu_batch.py` 존재) | **×N×0.85**(η 미실측) | GPU당 프로세스 + flock 공유 큐, 단일 parity 확인됨 | B0(출력 동일) | **W4** |
| **L7** | duplicate/대칭 canonical **키 노출** | GA의 것(문헌 1.1–1.4×) | evaluator는 키만 발행한다 | B0(순수 함수) | **W1** |
| **L8** | 부모 LP warm-start flux | 1.05–1.15× | `initial` outer 347/4,609 = **7.5 %** | N1 | **W3** |

### 0.2 정직한 상한

```text
현재 (238, M64, 1 GPU) :   524 c/h  =  6.87 s/case  =  2.42× MASTER W16(217 c/h)
공학 레버만(L2·L4·L5, 전 충실도) : 1,600 – 2,100 c/h = 7.4 – 9.7× MASTER
                                   → t_case 1.71 – 2.25 s   (중앙 1.9 s)
+ L3(coarse 10상태)              : 3,600 – 4,700 c/h = 17 – 22× MASTER
                                   → t_case 0.77 – 1.00 s   (중앙 0.85 s)
```

`N_eval = 2.56 M`(P=256 × G=10k) 완주 시간:

| 구성 | 1 GPU | 4 GPU(η .85) | 8 GPU(η .85) |
|---|---:|---:|---:|
| 현재 (6.87 s) | **203.6 일** | 59.9 일 | 29.9 일 |
| 공학 레버만 (1.9 s) | 56.3 일 | 16.6 일 | **8.3 일** |
| + coarse 충실도 (0.85 s) | 25.2 일 | 7.4 일 | **3.7 일** |
| + duplicate 20 % 회피 (0.68 s) | 20.2 일 | 5.9 일 | **3.0 일** |

> **≤3일은 "8 GPU + 공학 레버 완주 + coarse 충실도 정책 + duplicate 회피" 전부가 성립할 때
> 겨우 닿는다.** 35상태 자연-EOC 전 충실도로는 8 GPU에서 약 8일, 4 GPU에서 약 17일이
> 정직한 값이다. 1 GPU 3일(0.101 s/case = MASTER 대비 164×)은 **어떤 레버 조합으로도
> 도달하지 않는다.**
>
> **그리고 L1은 이 표의 어느 칸도 바꾸지 않는다 — 그것이 없으면 표 자체가 성립하지 않을 뿐이다.**
> 2.56 M 케이스를 지금의 산출물로 쓰면 **772 PB**다(케이스당 301.6 MB). scalar-only는
> **64 GB**다. 처리량 레버가 아니라 **캠페인이 존재할 수 있는 조건**이다(§3.1(d)).

### 0.3 이번 조사에서 나온, 계획을 바꾸는 사실 셋

1. **상태점을 줄이는 것은 선형이 아니고 단조도 아니다.** 3상태 덱(BOC+8+16 GWd/t)은 35상태
   전 덱보다 **outer가 더 많다**(5,104 vs 4,609). 한 번에 207 EFPD를 태우면 붕소 탐색과 Xe
   캐스케이드가 폭발한다 — 그 한 상태점 하나가 **4,798 outer / 18.87 s**다(§2.5).
   `tools/make_screening_deck.py`의 선형 비용 모델은 **틀렸고**, §5.3이 그것을 고친다.
2. **케이스 비용의 40 % 이상이 outer와 무관한 상태점당 호스트 바닥이다.** 로컬 회귀:
   `t_statepoint = 0.474 s + 4.805 ms × outer` (rms 0.100 s, n=35). 35상태면 바닥만
   **16.6 s**다. PPR·depletion(CRAM)·FlatXS·T/H가 여기 있고 전부 CPU다(§2.2).
3. **scalar-only 결과 경로는 이미 존재하고, 이미 궤적-동일이다.** `RASBERY_BATCH_LIGHT_RESULT=1`
   3회 실행 전부 `digest=814201df0583e1d2`로 전 충실도 실행과 **같다**. GA가 필요로 하는
   Fq/FΔH/CBC/keff/EFPD/평균연소도가 상태점당 JSONL 한 줄로 나온다(§2.4).

---

## 1. 측정 환경, 계기, 그리고 이 문서의 수치를 읽는 법

### 1.1 arm

`docs/V3_FREEZE_20260829_KO.md` §2의 v3 생산 arm 그대로다.

```bash
export RASBERY_GPU=1 RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1
export RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1
export RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1
export RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8
export RASBERY_GPU_WIEL_FOLD=chunked RASBERY_GPU_XE=1
export RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1
export RASBERY_OMP_THREADS=12
```

바이너리는 `feff7e7` 트리(+ 같은 시점의 `src/` 작업본)로 로컬 빌드한 것이다. 이 문서의 A/B는
**전부 같은 바이너리 안에서 env로만** 갈렸으므로 빌드 차이가 개입할 수 없다.

### 1.2 계기 — 새 수신증은 없다, 도구가 하나 생겼다

`tools/case_cost_profile.py`(신규)는 바이너리가 **이미 찍는** 수신증만 접는다.

| 수신증 | 출처 | 주는 것 |
|---|---|---|
| `[TIMING] Init+IO=` | `Driver.h:3950` | 덱+XSLIB 파싱, 솔버 객체 생성, 아레나 admission, `OpenResult` |
| `[TIMING] IO write=` | `Driver.h:4232` | **드라이버 스레드에 청구된** I/O |
| `TOTAL DRIVER TIME=` | `Driver.h:4233` | `Drive()` 진입~종료 |
| `[RASBERY][SPTELEM][SUMMARY]` | `Driver.h:4249` | `library_seconds`·`solve_wall`·`io_wall`·outer 귀속 전량 |
| `[RASBERY][IO_WRITER][SUMMARY]` | `IoWriter.h` | `bytes`·`writer_busy_ms`·`enqueue_block_ms` |
| `[RASBERY][TRAJECTORY]` | `Driver.h:4305` | `outers`·`statepoints`·**`digest`**·arm env |
| `[RASBERY][CUDA][BATCH_OCCUPANCY]` | `CudaBICGBackend.cu:5525` | `mean_width`·`claim_*.wait_ms` |
| `[RASBERY][REFILL]` | `BatchRefill.h` | `tail_idle_s`·`slot_busy_fraction`·테넌시 감사 |

프로세스 wall은 **수신증이 아니다** — Driver는 자기 기동·teardown을 볼 수 없다. 도구는
`/usr/bin/time -f "%e" -o NAME.wall`이 남긴 사이드카를 `--wall-dir`로 받고, 없으면
`outside_drive` 칸을 **비워 둔다**. 추정하지 않는다.

### 1.3 이 문서의 로컬 wall을 읽는 법

로컬 wall의 run-to-run 산포는 **±3 s(약 7 %)** 다(같은 arm 반복: 38.71 / 40.35 / 42.16 s).
반면 **드라이버 내부 회계**(`Init+IO` / `solve_wall` / `io_wall` / outer 수)는 훨씬 조용하고,
`digest`는 arm이 같으면 **비트로 같다**. 따라서:

- **비율과 내부 회계로 판정한다.** wall 배수 단독으로는 판정하지 않는다.
- 238의 절대값은 다르다(sm_120, 다른 CPU). **A2 arm에서 outer 수 자체가 호스트마다 다르다**
  (`V3_FREEZE` §8.4). 로컬 4,609 ↔ 238 3,114. 따라서 **같은 호스트 안의 A/B만이 판정**이다.

---

## 2. 케이스 하나의 비용 원장 (실측)

### 2.1 원장 — 산출물 3종 × 스케줄 3종

`tools/case_cost_profile.py ~/gaplan/*.log --wall-dir ~/gaplan` 출력(초, 텔레메트리 arm):

| run | 덱 | wall | drive | outside | init+io | library | solve | io(drv) | wr_busy ms | out bytes | sp | outers | ms/outer | digest |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `full_tel` | 35상태, 핀출력 ON | 44.67 | 42.07 | 2.60 | 1.25 | 0.45 | 38.20 | **2.60** | 607 | **182,266,343** | 35 | 4,609 | 8.29 | `814201df…` |
| `pinoff_tel` | 35상태, 핀출력 OFF | 41.91 | 39.93 | 1.98 | 1.13 | 0.30 | 38.74 | **0.05** | 191 | **11,954,663** | 35 | 4,609 | 8.40 | `814201df…` |
| `light_tel` | 35상태, **scalar-only** | 44.84 | 41.30 | 3.54 | 1.52 | 0.32 | 39.36 | **0.43** | 0 | **0**(+25.1 kB JSONL) | 35 | 4,609 | 8.54 | `814201df…` |
| `coarse_tel` | **10상태**(0.5–16 GWd/t) | 20.52 | **18.77** | 1.75 | 0.97 | 0.28 | 16.91 | 0.89 | 235 | 52,078,418 | 10 | **2,334** | 7.25 | `84065939…` |
| `three_tel` | **3상태**(BOC+8+16) | 27.58 | 25.76 | 1.82 | 2.68 | 0.35 | 22.86 | 0.21 | 32 | 15,625,799 | 3 | **5,104** | 4.48 | `e32a0a4c…` |

전 충실도 3종(`full`/`pinoff`/`light`)의 **digest가 모두 같다**(`814201df0583e1d2`, outer 4,609,
th 126). 즉 **산출물 형식은 물리를 건드리지 않는다** — 이것이 L1의 게이트다.

핀출력 ON일 때 케이스당 실제 산출물은 h5 182.6 MB **+ `*_pinpower.csv` 119.0 MB = 301.6 MB**다
(238의 ~254 MB/케이스와 같은 계열; 덱의 핀 인쇄 설정과 상태점 수가 다르면 값이 다르다).

### 2.2 비용 모델 — 상태점당 바닥이 outer보다 크다

인쇄되는 상태점 줄(`NO.= … outer=… t=…s`)에 대한 최소제곱:

```text
pinoff_tel (35 상태) : t_sp = 0.474 s + 4.805 ms x outer      rms 0.100 s
coarse_r1  (10 상태) : t_sp = 0.538 s + 4.720 ms x outer      rms 0.044 s
```

두 덱이 **독립적으로 같은 두 계수**를 준다. 이것이 이 캠페인이 지금까지 갖지 못했던 케이스
비용 모델이다.

```text
t_case  ≈  T_process  +  T_percase  +  N_sp x c  +  N_outer x d
              2.0 s        1.1 s        0.474 s      4.805 ms      (로컬 1080 Ti)
```

35상태 전 덱에 대입: `2.0 + 1.1 + 35×0.474 + 4609×0.004805 = 2.0 + 1.1 + 16.6 + 22.1 = 41.8 s`
(실측 wall 41.9–44.7). **상태점 바닥 16.6 s는 outer를 하나도 줄이지 않아도 남는다.**

바닥 `c = 0.474 s`의 내용물은 outer 루프 **밖**의 상태점 작업이다:

| 항목 | 위치 | GPU/CPU |
|---|---|---|
| PPR(핀출력 재구성) — `reset` + `drive(100)` + `reconstructPinPower` | `Driver.h:4065-4082`, `PPR.cpp`(1,099줄) | **CPU** |
| depletion predictor/corrector + CRAM Bateman(전 8,451 노드) | `XSSet.cpp:4008,4036`, `milk.h:1649-1819` | **CPU** |
| `UpdateFlatXS` / `UpdateTH` / `SetBoron` / `UpdateBurnup` | `XSSet.cpp` | 혼합 |
| 결과 packing (`AddResult`) | `IO.cpp` | CPU |

**PPR은 인쇄 여부와 무관하게 매 상태점 실행된다.** `Driver.h:4065-4082`에 게이트가 없고,
`schedule.print_opt`는 `reconstructPinPower`의 **두 번째 인자(pin_flux)** 에만 관여한다.
`pinoff` arm의 `solve_wall`이 `full` arm과 같은 것(38.74 vs 38.20)이 그 직접 증거다 —
핀출력을 꺼도 PPR 계산은 그대로 돈다.

그리고 **GA는 PPR을 필요로 한다**: fitness의 `fqp`(Fq)와 `frp`(FΔH)가 PPR 산출물이다
(`BatchLightResult::Write`가 그 둘을 싣는다). 즉 **PPR은 GA evaluator의 임계 경로이며,
호스트에 남아 있는 가장 큰 상태점당 비용이다.** 계획 Rev.7.1의 Task 19b(GPU PPR)를
W5+에서 앞으로 당겨야 하는 이유가 여기서 두 번째로 나온다.

#### `N_outer`가 어디서 오는가 — 피드백 귀속 (`full_tel` SPTELEM)

```json
{"outers":4609,"outers_initial":347,"xe_outers":3179,"search_outers":707,
 "th_outers":86,"settle_outers":290,"fallback_outers":0,
 "search_trials":137,"th_updates":126,"xe_cascades":228,"xe_updates":1192,
 "xe_steps_per_cascade":5.228,"xe_aa_proposed":750,"xe_aa_accepted":733,
 "staged_relapses":0,"solve_loops":69,"cmfd_sweeps":19603,"bicg_iters":78412,
 "graph_launches_delta":4908,"h2d_bytes_delta":877204680,
 "d2h_bytes_delta":664332336,"d2h_calls_delta":9462}
```

| 원인 | outer | 비중 | 케이스당 초(× 4.805 ms) | GA에게 의미하는 것 |
|---|---:|---:|---:|---|
| `xe` (평형-Xe 캐스케이드 재수렴) | **3,179** | **69.0 %** | 15.3 s | 붕소 trial·T/H 갱신마다 재장전. **케이스 비용의 최대 단일 항** |
| `search` (임계붕소 secant) | 707 | 15.3 % | 3.4 s | trial 137회. LP가 반응도상 나쁠수록 늘어난다 → **케이스 비용이 후보 의존** |
| `initial` | 347 | 7.5 % | 1.7 s | **warm-start(L8)가 겨냥하는 정확히 이 버킷** |
| `settle` | 290 | 6.3 % | 1.4 s | 탐색이 신뢰하는 표본을 만드는 게이트 |
| `th` | 86 | 1.9 % | 0.4 s | |
| `fallback` | 0 | 0 % | 0 | 탐색이 한 번도 최선점 복귀를 하지 않았다 |

`staged_relapses = 0`은 A2 단계화가 **느슨한 단계의 합의를 생산 허용오차가 한 번도 뒤집지
않았다**는 뜻이다(`A2_OUTER_REDUCTION` §4.1) — GA arm에서도 이 값이 0이 아니면 그 실행은
느슨한 단계가 다른 근을 찾고 있다는 신호다.

> **238에서 반드시 다시 재야 하는 것.** 위 두 계수는 1080 Ti + 이 CPU의 값이다. 238의
> `(c, d)`는 다를 뿐 아니라 **비율이 다를 것**이다(GPU가 1.8–2.6× 빠르고 CPU는 그만큼
> 빠르지 않다). 238에서 `c`가 0.2 s라면 35상태 바닥만 7 s이고, 그것은 16.9 s 케이스의
> **41 %** 다. 이 한 줄의 회귀가 W1의 첫 태스크인 이유다(§6.1 Task 1).

### 2.3 상각 가능한 오버헤드 vs 진짜 물리

| 버킷 | 로컬 실측 | 케이스당 성격 | persistent evaluator가 상각하는가 |
|---|---:|---|---|
| `outside_drive` (프로세스 기동 + CUDA 컨텍스트 + teardown) | 1.75–4.92 s | 배치에서는 **프로세스당 1회** | 이미 상각됨(배치) / 1케이스-1프로세스에서는 전액 낭비 |
| `Init+IO` (`ReadInput` + 솔버 생성 + 아레나 admission + `OpenResult`) | 0.86–1.63 s | **케이스당** | **전액 상각 대상**(§3.1(a)) |
| ↳ 그중 `library_seconds` (덱 JSON + **34 MB 단면 라이브러리 파싱·flatten**) | 0.28–0.45 s | **케이스당**, 프로세스 전역 락 아래 | **전액**, 그리고 배치에서는 그 이상 |
| `N_sp × 0.474 s` (PPR/depletion/FlatXS/TH) | 16.6 s (35상태) | **케이스당, 진짜 계산** | 아니오 — GPU 이식(L4)이 유일한 경로 |
| `N_outer × 4.805 ms` | 22.1 s | **케이스당, 진짜 계산** | 아니오 — outer 감축(L4)과 커널(L4/L5) |
| `io_wall`(드라이버 스레드) | 2.60 / 0.05 / 0.43 s | **케이스당** | L1이 0.43 s로, 핀출력 OFF가 0.05 s로 |

즉 **전 충실도 케이스의 88–93 %는 진짜 물리이고, persistent evaluator가 지울 수 있는 것은
케이스당 2–4 %(단일 실행 기준)** 다. persistent evaluator의 값어치는 단일 실행의 백분율이
아니라 **배치에서 그 2–4 %가 어디에 있는가**에 있다(§3.1).

### 2.4 L1 — scalar-only 결과는 이미 있고, 이미 궤적-동일이다

```bash
RASBERY_ALLOW_SCREENING=1 RASBERY_BATCH_LIGHT_RESULT=1 \
RASBERY_BATCH_RECEIPT_JSONL=<out>.jsonl  RASBERY <arm> --rasi deck.json --raso <unused>
```

| | 전 충실도 | scalar-only |
|---|---:|---:|
| 산출물 | h5 182.6 MB + csv 119.0 MB = **301.6 MB** | JSONL **25,105 B** |
| 드라이버 스레드 `io_wall` | 2.602 s | **0.371 s** |
| writer 스레드 `writer_busy_ms` | 607 | 0 |
| `digest` | `814201df0583e1d2` | **`814201df0583e1d2`** |

JSONL 한 줄이 상태점 하나이고, GA가 요구한 스칼라가 전부 들어 있다:
`efpd`, `bu_avg`, `keff`, `ppm`(CBC), `ao`, `fqp`(Fq), `frp`(FΔH), `search_status`,
`search_dk`, `search_tol`, `converged`, 그리고 provenance(`input_sha256`, `xs_sha256`,
`candidate_id`, `cycle`). **cycle length는 직접 들어 있지 않다** — `efpd` 궤적과 `ppm` 궤적에서
GA가 유도해야 하고, 자연-EOC 덱이면 마지막 `efpd`가 곧 cycle length다. §5.2가 이 계약을 적는다.

**fail-closed 가드가 있다.** `RASBERY_ALLOW_SCREENING=1` 없이 light를 켜면 프로세스가
**exit 2로 거부**한다(`main.cpp:409-433`). 스크리닝 결과가 수용 표에 우연히 들어가는 사고를
막는 장치이며, **GA 캠페인에서도 이 가드를 끄지 말 것** — elite 재계산 arm과 스크리닝 arm이
로그에서 구분되지 않으면 그 캠페인은 무효다.

### 2.5 L3 — 상태점 스케줄은 선형이 아니고 단조도 아니다 (**반례 실측**)

| 덱 | 상태점 | outer | drive (s) | outer/상태 | 최대 상태점 |
|---|---:|---:|---:|---:|---|
| full (자연 EOC) | 35 | 4,609 | 42.07 | 131.7 | — |
| **coarse** (0.5,1,2,4,6,8,10,13,16 GWd/t) | **10** | **2,334** | **18.77** | 233.4 | 759 outer / 4.12 s |
| **three** (BOC, 8, 16 GWd/t) | 3 | **5,104** | 25.76 | 1,701.3 | **4,798 outer / 18.87 s** |

`three`의 세 번째 상태점은 **207 EFPD를 한 번에** 태운다. 그 결과:

```text
  NO.=   1  EFPD=     0.000  PPM=  770.15  outer= 115  t= 1.08s
  NO.=   2  EFPD=   207.276  PPM=  558.31  outer= 191  t= 1.44s
  NO.=   3  EFPD=   414.552  PPM=  164.56  outer=4798  t=18.87s      <-- 여기
```

원인은 구조적이다. 붕소 secant 탐색과 평형-Xe 캐스케이드는 **직전 상태점의 해에서 출발**한다.
연소 간격이 커질수록 출발점이 멀어지고, 캐스케이드가 커밋된 trial마다 재장전되므로
(`A2_OUTER_REDUCTION` §2) 비용이 간격에 대해 **초선형**으로 자란다. `coarse`가 `three`보다
싼 것은 우연이 아니라 이 구조다.

**따라서 evaluator가 GA에게 제공해야 하는 것은 "상태점 수" 노브가 아니라 "연소 격자"이고,
그 격자의 비용은 상태점 수가 아니라 outer 수로 예측해야 한다.** §5.3이
`tools/make_screening_deck.py`의 비용 모델을 이 형태로 고친다.

부수 효과 하나가 더 중요하다: `coarse`는 `until boron ppm` 꼬리를 버리므로 **케이스 비용이
고정**된다. 전 덱은 자연 EOC 때문에 **상태점 수가 후보마다 다르다**(35는 이 덱의 값일 뿐이다).
그 가변성이 §3.1(c)의 랑데부 왜곡을 만드는 원천 중 하나다.

---

## 3. 병목 원장

### 3.1 배치 (M64, 238: 524 c/h, `mean_width` 14.5/64)

#### (0) 로컬 배치 A/B — 무엇을 측정할 수 있고 무엇을 측정할 수 없는가

로컬 8잡 / 폭 4(`--jobs`, `RASBERY_OMP_THREADS=1`, 24코어, 1080 Ti):

| | 전 출력 | scalar-only |
|---|---:|---:|
| wall (8 케이스) | **279.15 s** | **286.72 s** |
| 유효 c/h | 103.2 | 100.4 |
| `mean_width` (슬롯 4) | **1.104** | 1.116 |
| `width_histogram` | [32224, 2774, 460, **0**] | [31592, 2925, 566, 3] |
| **`[HDF5][LOCK]` acquires / wait_ms** | 320 / **15,178.8** | **24 / 36,889.9** |
| `[IO_WRITER][SUMMARY].bytes` | **1,458,130,744** | **0** |
| `writer_busy_ms` | 17,666 | 0 |
| 케이스당 `Init+IO` | 3.02, 3.06, 3.73, 5.12 / 1.31–3.96 | **10.95, 11.17, 12.66, 15.43** / 1.94–5.38 |
| `[REFILL]` 감사 3종 | 0 / 0 / 0 | 0 / 0 / 0 |
| `digest` 8/8 | `814201df…`(단일과 동일) | `814201df…`(동일) |

**측정할 수 있는 것 — 그리고 이 표가 정한 것 셋:**

1. **L1은 처리량 레버가 아니다.** 전 출력 1.458 GB를 **0으로** 만들었는데 wall이 오히려
   2.7 % 늘었다(잡음 범위). writer 스레드가 이미 쓰기를 겹치고 있으므로 남는 것이 없다 —
   238의 M64에서 writer 스레드 채택이 **+0.6 %** 였던 것과 같은 결론이다(캠페인 §8.1).
   §0.1의 L1 칸을 "1.1–1.3× 추정"에서 **"처리량 1.0×"** 로 고친 근거가 이것이다.
2. **배치의 진짜 직렬 자원은 HDF5 전역 락이고, 그 락의 내용은 쓰기가 아니라 읽기다.**
   scalar-only arm은 **한 바이트도 쓰지 않는데** 락 대기가 **36.9 s**다. 획득이 24회뿐인 것이
   그 증거다 — 그 24회가 `ReadInput`(8) + 그 안의 재진입이고, **한 번 잡으면 34 MB 라이브러리
   파싱이 끝날 때까지 놓지 않는다**(`IO.cpp:542`가 함수 전체를 감싼다). 단일덱 실행의
   같은 수치는 **0.0075 ms**다. 1 → 4 동시로 가면서 **500만 배**가 된 것이다.
3. **배치와 단일은 궤적이 같다.** 8/8 케이스가 단일덱과 같은 `digest`를 낸다. 배치화는 B0다.

**폭을 4에서 8로 넓히면** (같은 8잡, 전 출력):

| | 폭 4 | 폭 8 | |
|---|---:|---:|---|
| wall | 279.15 s | **206.94 s** | **1.35×** |
| `mean_width` | 1.104 | 1.492 | |
| `width_histogram` | [32224, 2774, 460, 0] | [18772, 4224, 1841, 841, 393, 138, 36, 0] | |
| **`[HDF5][LOCK].wait_ms`** | 15,179 | **63,841** | **4.2× 악화** |
| `writer_busy_ms` | 17,666 | 22,709 | |
| `max_queue_depth` | 4 | 14 | |

**그리고 폭 8의 `Init+IO` 여덟 개를 순서대로 읽으면 이 문서의 가장 선명한 그림이 나온다:**

```text
4.108  4.682  6.398  8.879  10.041  11.358  13.194  14.506   (s)
       +0.57  +1.72  +2.48   +1.16   +1.32   +1.84   +1.31
```

동시에 출발한 8개 워커가 **한 줄로 서서** 차례로 라이브러리를 읽는다. 기울기는
**케이스당 약 1.49 s**이고, 그것이 `IO.cpp:542`의 전역 `Hdf5Guard` 아래 34 MB 파싱 시간이다.
선형 외삽하면 **폭 64의 마지막 케이스는 첫 outer를 돌기 전에 약 98 s를 기다린다**
(238의 CPU·디스크가 3배 빨라도 ~33 s이고, 그 호스트의 케이스 전체가 16.9 s다).

**이것이 `mean_width` 14.5/64의 절반짜리 설명이다** — 슬롯이 랑데부에 없는 이유의 하나가
"아직 지오메트리를 못 세웠다"이고, 그것은 GPU와 아무 상관이 없다.

**측정할 수 없는 것:** 폭의 상한. 1080 Ti는 슬롯 4에서 `mean_width` 1.10, 슬롯 8에서 1.49다.
238의 `mean_width` 14.5/64 절대값은 **이 하드웨어에서 재현되지 않으며**, L5(compaction)의
판정은 238에서만 가능하다(§6.4 Task 13). 로컬이 확정한 것은 **방향**이다: 폭을 넓히면
GPU 쪽은 좋아지고 HDF5 락은 나빠진다 — **두 곡선의 교점이 최적 폭이고, L2가 그 교점을 옮긴다.**

#### (a) 케이스마다 34 MB 라이브러리를 프로세스 전역 락 아래에서 다시 읽는다

`IO::ReadInput`(`IO.cpp:538-965`)은 **함수 전체를 `Chiffon::Hdf5Guard`로 감싼다**(`IO.cpp:542`).
그 안에서 `XSSet::Initialize`(`XSSet.cpp:555-878`)가 `Importer::LoadHDF(xs_path)`
(`XSSet.cpp:586`)로 34 MB CHIFFON 라이브러리를 읽고, 이어서 `XSSet.cpp:678-844`가 그것을
전량 flatten한다(`_lib_lmpx/_lib_micx/_lib_coeff_*/_lib_deltas/_lib_knots/_refr_*/_brch_*`).

- `LoadHDF`는 **파일 이름만 받는 순수 함수**다(`Importer.h:821-901`). 덱의 `core`/`batch`를
  전혀 보지 않는다. flatten도 `_models` + `ng` + `niso`의 순수 함수다.
- 즉 **LP만 다른 64개 덱이 같은 결과를 64번 만든다.** 그것도 직렬로.
- **디바이스 쪽은 이미 캐시가 있다**: `CudaXsReconBackend.cu:561`의 `g_flatxs_libs`가 내용
  해시로 프로세스 수명 캐시를 유지한다(`:2720-2780`). **호스트 파싱·flatten만 캐시가 없다.**
- 계획 Rev.4 §14("XSLIB Cache")가 이 트랙을 이미 명세했고(`…REV4_KO.md:900-935`),
  `Driver.h:3842-3847`의 `library_seconds`는 **그 A/B를 위해 심어진 계기**다. 미구현이다.

로컬 실측 `library_seconds` = 0.28–0.45 s(**단일덱**). 배치에서 같은 일의 비용은
`Init+IO`가 직접 말한다:

| | 단일 | 배치 폭 4, 첫 파도 4케이스 | 배치 폭 4, 리필 4케이스 |
|---|---:|---:|---:|
| `Init+IO` (전 출력) | 0.86–1.63 s | 3.02 / 3.06 / 3.73 / **5.12 s** | 1.31–3.96 s |
| `Init+IO` (scalar-only) | 1.27–1.52 s | 10.95 / 11.17 / 12.66 / **15.43 s** | 1.94–5.38 s |

**동시에 시작한 4케이스의 마지막 하나가 지오메트리를 세우기까지 5–15 s를 기다린다.**
그 시간 동안 그 슬롯은 랑데부에 없다. 폭 64면 마지막 케이스는 63명 뒤에 선다.
`IO_WRITER` 설계 문서 §5.1이 같은 것을 이미 관측했다(**8덱 동시 진입, 24회 획득에 8.4–9.6 s**)
— 이번 측정은 그것을 케이스당 숫자로 옮겼을 뿐이다.

**그리고 이 비용은 전액 낭비다.** 64개 덱의 `LoadHDF` 결과는 **비트로 같다** — 파일명만의
순수 함수이기 때문이다.

#### (b) 호스트 CPU가 폭을 결정한다 — GPU가 아니라

배치 분기는 `omp_set_max_active_levels(1)`로 **중첩 OMP를 끈다**(`main.cpp:626-630`).
따라서 각 Driver의 호스트 작업(PPR, CRAM depletion, FlatXS, TH, HDF5 payload 복사)은
**단일 스레드**이고, M64는 24코어 호스트에서 **2.67× 오버섭스크립션**이다.

`mean_width` 14.5/64 = **22.7 %**를 그대로 읽으면: 케이스가 자기 시간의 약 23 %만 GPU CMFD
랑데부 안에 있고, **77 %는 호스트 작업 또는 CPU 대기**다. §2.2의 비용 모델이 같은 말을 한다 —
상태점 바닥 16.6 s / 41.8 s = 40 %가 호스트 계산이고, 여기에 `library_seconds`와 I/O가 더해진다.

**폭을 96/128로 넓히면 왜 손해인가**가 이것으로 닫힌다. 커널은 grid를 **선언된 슬롯 수**에
비례해 잡는데(캠페인 §4: 폭 96에서 BASE −62 %), 실점유는 호스트 공급이 정하므로 늘지 않는다.
`mean_width` 14.5 → 11.6(M96/M128)은 **호스트 몫이 워커당 줄어든 결과**이지 GPU의 성질이 아니다.

#### (c) 랑데부 왜곡의 두 원천

1. **케이스마다 상태점 수가 다르다.** `until boron ppm`이 재장전하는 자연 EOC(`IO.cpp:441`,
   `Driver.h`의 재큐잉)는 후보의 반응도에 따라 상태점 수를 바꾼다. 슬롯들은 같은 상태점
   경계에서 만나지 않는다.
2. **같은 상태점 안에서도 outer 수가 다르다.** `coarse`의 상태점별 outer는 61–759(12배)다.
   Anderson 채택 후 이 분산이 더 커졌고(`CAMPAIGN…` §8.2의 "도착 폭 기아"), 그것이 배치에서
   Anderson을 기본 OFF로 둔 이유다.

**이 왜곡은 제거할 수 없다 — 물리의 성질이다.** 제거할 수 있는 것은 **왜곡의 비용**이고,
그 방법은 둘뿐이다: (i) grid 비용을 선언 폭이 아니라 실점유에 묶는다(slot compaction,
Phase 5), 또는 (ii) 선언 폭 자체를 작게 여러 개 둔다(GPU당 다중 프로세스 + MPS).
**(ii)는 C++를 건드리지 않는다** — §5.5.

#### (d) 산출물 19.3 GB/파도 — 처리량이 아니라 캠페인의 존재 조건

64케이스 × 301.6 MB = **19.3 GB/파도**. **처리량으로는 이것이 공짜에 가깝다**(§3.1(0)의
A/B: 1.458 GB를 0으로 만들어도 wall 불변). writer 스레드가 이미 겹치고 있고, 238의 M64에서도
writer 채택 효과는 **+0.6 %** 였다.

문제는 처리량이 아니라 **총량**이다.

| | 케이스당 | 2.56 M 케이스 |
|---|---:|---:|
| 전 출력 (h5 + pin CSV) | 301.6 MB | **772 PB** |
| 핀출력 OFF (h5만) | 12.0 MB | 30.7 PB |
| **scalar-only (JSONL)** | **25.1 kB** | **64 GB** |

**772 PB는 저장할 수 없다.** 파일 개수도 문제다 — 케이스당 h5 1 + restart N + CSV 1이면
2.56 M 케이스는 수천만 개의 파일이고, 그것을 담을 네임스페이스 규약이 이 코드베이스에 없다
(현재 규약은 `--raso` 경로 유일성이다). 따라서 **L1은 선택이 아니라 전제**이고,
`RASBERY_ALLOW_SCREENING` 가드가 그것을 사고로 만들지 않게 지킨다.

#### (e) 원장 요약

| 항목 | 실측/유도 | 레버 | 등급 |
|---|---|---|---|
| **케이스 시작이 전역 HDF5 락 뒤에 줄을 선다** | `Init+IO` 4.1 → 14.5 s(폭 8 계단), 기울기 **1.49 s/케이스**, 락 대기 **63.8 s**(폭 8, 8잡) | **L2** | **1순위** |
| 호스트 단일 스레드 × M, 24코어 | `omp_set_max_active_levels(1)`(`main.cpp:626`), M64/24 = 2.67× 오버섭스크립션, 238 `mean_width` 22.7 % | **L4**(호스트 작업 GPU 이식) + **L5** | 2순위 |
| 커널 grid ∝ **선언** 폭 | 238 폭 96에서 BASE −62 %(캠페인 §4). 로컬은 폭 4 → 8에서 **+35 %**(락이 아직 이기지 않는 구간) | **L5** | 2순위 |
| 상태점/outer 분산 | outer 61–759(12×), 상태점 수는 자연 EOC 때문에 **후보 의존** | **L3**(고정 격자) + **L5** | 3순위 |
| 산출물 총량 | 19.3 GB/파도, 2.56 M 케이스 = **772 PB** | **L1** | **전제** |
| 결과 동일성 | 배치 8/8 케이스가 단일덱과 `digest` 동일 | — | 확인됨 |

### 3.2 단일 (238: 16.9 s, MASTER 27.2 s → 1.61×)

| 항목 | 현재 | MASTER | 여유 | 레버 |
|---|---:|---:|---:|---|
| outer/상태 | **89**(238, 3,114/35) | **59** | **1.51×** | L4(A2 2단, WHILE) |
| 상태점당 호스트 바닥 | **0.474 s**(로컬) | — | Task 19b·16·17 | L4 |
| in-body 호스트 동기화 | 세그먼트당 1회(`hostfree` 기본 ON) | — | WHILE이 no-op outer 제거 | L4 |
| `nodal_constants` 호스트 호출 | 2,433/run(캠페인 기록) | — | N1 이식 | L4 |
| launch 지연 | 137 µs/콜, 쿼터코어가 스레드슬롯 ~2 % 충전 | — | graph/WHILE | L4 |
| `graph_launches` | 4,908/run(로컬 SPTELEM) | — | WHILE로 세그먼트당 1회 | L4 |
| H2D / D2H | 877 MB / 664 MB, D2H 호출 9,462회 | — | canonical residency | L4 |

#### outer 본문의 위상 귀속 (실측)

`RASBERY_GPU_OUTER=1`이 켜져 있으면 호스트 outer 본문이 아예 돌지 않으므로
`[RASBERY][OUTER][PHASE]`의 7버킷이 **전부 0**이다(로컬 확인 — `V3_FREEZE` §8.2와 같은 관측).
`RASBERY_GPU_OUTER`를 끄고 `RASBERY_OUTER_TIMING=1`로 재실행하면:

```
[RASBERY][OUTER][PHASE] {"outers":4609,"updpsi":0.249,"setls":0.110,"drive":12.767,
                         "updjnet":1.115,"nodal":4.538,"cusping":0.090,"upddhat":1.543}
TOTAL DRIVER TIME=45.874 s      (device outer ON 인 같은 덱은 39.93 s)
```

**outer 수가 4,609로 동일하다** — device outer 세그먼트는 B0이고, 그 arm 전환이 궤적을
바꾸지 않는다는 것을 이 실행이 다시 확인한다.

| 위상 | 초 | ms/outer | 본문 내 비중 |
|---|---:|---:|---:|
| `drive` (BiCG outer, 아레나 대기 + sync 포함) | 12.767 | **2.770** | **62.6 %** |
| `nodal` (reset + drive) | 4.538 | 0.985 | 22.2 % |
| `upddhat` | 1.543 | 0.335 | 7.6 % |
| `updjnet` | 1.115 | 0.242 | 5.5 % |
| `updpsi` + `setls` + `cusping` | 0.449 | 0.097 | 2.2 % |
| **합** | **20.41** | **4.428** | 100 % |

**§2.2의 회귀 기울기 `d = 4.805 ms/outer`(device outer **ON** arm)와 위 합
`4.428 ms/outer`(**OFF** arm)가 일치한다.** 차이 0.38 ms/outer는 Scope 밖에 있는 것들
(Xe 갱신, 수렴 판정, 탐색 산법)이다. 즉 **회귀 계수 `d`는 outer 본문 비용 그 자체이고,
`c`는 그 밖의 전부**라는 해석이 두 개의 독립적인 계기로 확인된다.

전 케이스에서 보면: outer 본문 20.4 s / 상태점 바닥 16.6 s / 고정 오버헤드 3.1 s /
I/O 0.05–2.6 s. **`drive`(12.8 s)가 케이스의 단일 최대 항이고, 상태점 바닥(16.6 s)이
그것보다 크다.**

---

## 4. 목표 산술

### 4.1 필요 `t_case`

`N_eval = 2,560,000`. `t_case = B × N × η / N_eval` (η = 1 for N=1, 0.85 for N>1).

| wall 예산 | 1 GPU | 4 GPU | 8 GPU |
|---|---:|---:|---:|
| **3 일** | **0.101 s** (35,556 c/h) | **0.344 s** (10,459 c/h/GPU) | **0.689 s** (5,229 c/h/GPU) |
| 7 일 | 0.236 s (15,238) | 0.803 s (4,482) | 1.607 s (2,241) |
| 14 일 | 0.473 s (7,619) | 1.607 s (2,241) | 3.213 s (1,120) |
| 30 일 | 1.013 s (3,556) | 3.443 s (1,046) | 6.885 s (523) |

**현재의 524 c/h는 정확히 마지막 칸이다** — 8 GPU에서 30일. 이것이 출발점이다.

### 4.2 "수십 배 MASTER"의 번역

MASTER W16 = 217 c/h = **16.59 s/case**.

| 배수 | 필요 c/h(전체) | `t_case` | 2.56M 소요 (GPU-일) | 1 GPU | 4 GPU | 8 GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 2.42× (현재) | 524 | 6.87 s | 203.6 | 203.6 일 | 59.9 일 | 29.9 일 |
| 10× | 2,170 | 1.659 s | 49.2 | 49.2 일 | 14.5 일 | 7.2 일 |
| **20×** | 4,340 | **0.829 s** | 24.6 | 24.6 일 | **7.2 일** | **3.6 일** |
| **50×** | 10,850 | **0.332 s** | 9.8 | 9.8 일 | 2.9 일 | **1.4 일** |

**"20× MASTER"는 두 가지 다른 문장이다.**
- *aggregate* 20×(4,340 c/h 전체)는 현재 기술로 **GPU 8–9장**이면 오늘 달성된다(524 × 8 × 0.85 = 3,563; 10장이면 4,454). 새 물리가 필요 없다.
- *per-GPU* 20×(한 장이 4,340 c/h)는 **8.3× 개선**이 필요하고, §0.2의 공학 상한(1,800–2,200 c/h)을
  넘는다. **충실도 레버(L3) 없이는 도달하지 않는다.**

이 구분을 문서·발표에서 흐리지 말 것. 캠페인의 4,340 c/h 목표(계획 §13.3)는 aggregate 목표다.

### 4.3 배수 합성 (정직한 곱)

```text
현재                       524 c/h
 x L1 scalar-only          1.00      (실측 — 처리량 레버가 아니다, 그러나 필수. 3.1(d))
 x L2 persistent library   1.35      (배치 직렬 구간 제거 — W2가 잰다, 밴드 1.2-1.6)
 x L5 폭/도착              1.60      (compaction 또는 다중 프로세스 — W4가 잰다, 밴드 1.2-2.2)
 x L4 단일 물리            1.80      (outer 1.51x + 상태점 바닥 GPU 이식, 밴드 1.5-2.5)
 ------------------------------
 = 1,812 c/h  =   8.4x MASTER  =  1.99 s/case      <-- 전 충실도 공학 중앙값
                                                       밴드 1,600 - 2,100 c/h
 x L3 coarse               2.24      (실측)
 = 4,059 c/h  =  18.7x MASTER  =  0.89 s/case
 x L7 duplicate 20%        1.25      (GA의 것)
 = 5,074 c/h  =  23.4x MASTER  =  0.71 s/case
```

곱셈이 정직한가에 대한 주석: L1·L2·L5는 **서로 다른 자원**(파일시스템/호스트 락, 호스트 파싱,
GPU grid)을 푼다. L4는 그 셋 뒤에 남는 시간을 줄인다. **겹침은 있다** — L4가 호스트 상태점
바닥을 GPU로 옮기면 L5의 이득이 줄어든다(폭이 이미 넓어졌으므로). 그래서 위 곱을 상한이 아니라
**중앙 추정**으로 두고, §0.2는 1,800–2,200 c/h의 밴드로 적었다.

---

## 5. Evaluator 계약 — GA가 무엇을 받는가

### 5.1 케이스의 정체 — LP는 덱의 `core` 블록이다

```json
"core": [["A0","B2","A0","C1","A0","C1","A0","C1","B0","R1"],
         ["B2","A0","B1", …], …]        // 10x10 쿼터, "R*" = 반사체, "XX" = 없음
"batch": {"A0":[{"id":"RT","count":1},{"id":"A0","count":25},{"id":"RB","count":1}], …}
```

`core`/`batch`의 **유일한 소비자**는 `XSSet::_comp`(노드당 모델 인덱스)와 `_asmb`
(집합체당 모델 인덱스)다(`XSSet.cpp:636-676`). `Geometry`는 `_core`/`_batch`를 그대로 들고
있고(`Geometry.cpp:155-156`), `core`에서 **파생**되는 것은 `_is_fuel`/`_kbc`/`_kec`/`_hzcore`
넷뿐이다(`Geometry.cpp:585-628`) — 전부 nxyz 위의 싼 스캔이다.

> **결론: `"XX"` 패턴이 고정된 채 non-`XX` 자리의 ID만 치환하는 LP 순열에 대해,
> `Geometry`의 나머지 전부(`_neibr`, `_lklr`, `_nsurf`, `_vol`, `_ijtola`, `hmesh`, …)는
> 비트로 동일하다.** 그것이 persistent evaluator를 가능하게 하는 사실이다.

따라서 evaluator API의 케이스는:

```text
case = { core[10][10]  (또는 대칭섹터만),
         params { boron_search_tol, statepoint_grid, tolerance_stage, ppr_iters, ... },
         warm_start { parent_case_id | none },
         result_policy { scalar | scalar+pin | full_h5 } }
```

### 5.2 결과 계약 (`--evaluator` 서버 모드의 반환값)

**이미 존재하는 것**(`BatchLightResult::Write`, 상태점당 JSONL 한 줄):
`step`, `substep`, `efpd`, `bu_avg`, `keff`, `ppm`, `ao`, `fqp`(Fq), `frp`(FΔH),
`search_status`, `search_dk`, `search_tol`, `converged`, `candidate_id`, `cycle`,
`input_sha256`, `xs_sha256`, `physics_mode`, `requires_exact_rerun`.

**추가해야 하는 것**(W1 Task 3):

| 필드 | 왜 |
|---|---|
| `lp_canonical_key` | GA의 duplicate/대칭 캐시가 쓸 **키**. 캐시는 GA의 것이지만 키는 evaluator만 만들 수 있다 — 대칭 접기가 `Geometry::_symopt/_symang/_part`(`Geometry.cpp:159-179`)의 함수이기 때문 |
| `cycle_length_efpd` | 자연-EOC 덱이면 마지막 `efpd`, 고정격자 덱이면 붕소 궤적의 10 ppm 외삽. **evaluator가 계산해야 한다** — 외삽 규약이 두 곳에 있으면 두 개의 답이 생긴다 |
| `peak_burnup_gwd` | 방출 연소도 제약 |
| `outers`, `statepoints`, `digest` | 케이스 비용과 궤적 신원. 이미 `[TRAJECTORY]`에 있으나 케이스 결과에도 실려야 한다 |
| `fp_class` | `fp64` / `fp32_mixed`. FP ranking stability 감사가 이것 없이는 불가능하다 |
| `warnings{th_clamped, search_not_converged, best_fallback, flux_limit_cycle}` | `tools/ga_two_stage_40x_pipeline.py`의 G3가 이미 로그 grep으로 세는 것. 결과에 실리면 grep이 필요 없다 |

**금지**: 비-elite 케이스에서 pin/flux 필드를 반환하지 않는다. elite는
`result_policy = full_h5`로 **같은 evaluator에서 재실행**하며, 그 실행은
`RASBERY_ALLOW_SCREENING`을 **설정하지 않는다**(수용 경로).

### 5.3 충실도는 파라미터다 — 그리고 그 비용 모델을 고친다

`tools/make_screening_deck.py`(미커밋)는 이미 연소 격자를 다시 쓴다. 두 가지를 고쳐야 한다.

1. **비용 모델.** 현재 `wall = fixed + ref_wall × N_sp/35` (또는 solve-unit 비)다.
   §2.5가 그 모델의 반례를 준다(3상태가 35상태보다 비싸다). 고친 형태:
   ```text
   t_case = T_percase + N_sp x c + N_outer x d
   N_outer = sum_sp  n0 + k x (dBU_sp)^alpha        <- 격자에서 예측, alpha > 1
   ```
   `(c, d, n0, k, alpha)`는 **한 번 재고 덱에 박아 넣는 상수가 아니라**, 도구가
   `--calibrate <log...>`로 실측 로그에서 회귀해 저장하는 값이어야 한다.
   `tools/case_cost_profile.py --json`이 그 입력을 낸다.
2. **`tools/ga_fitness.py`가 존재하지 않는다.** `make_screening_deck.py`의 docstring이
   "cycle length는 `tools/ga_fitness.py`의 붕소 외삽으로 회복한다"고 적지만 그 파일은
   트리에 없다. §5.2가 그 외삽을 **evaluator 쪽**으로 옮기라고 적는 이유다.

권고 격자(실측된 것):

| 이름 | 격자 (GWd/t 누적) | 상태점 | outer | drive(로컬) | 배수 | 용도 |
|---|---|---:|---:|---:|---:|---|
| `full` | 자연 EOC | 35 | 4,609 | 42.07 s | 1.00× | elite 수용, 최종 검증 |
| **`coarse`** | 0.5,1,2,4,6,8,10,13,16 | 10 | 2,334 | 18.77 s | **2.24×** | **GA 주 arm 후보** |
| `three` | 8,16 | 3 | 5,104 | 25.76 s | **1.63×** | **쓰지 말 것**(§2.5) |

`three`가 `coarse`보다 **비싸다**는 것이 이 표의 요점이다. 더 거친 격자를 원하면 상태점 수가
아니라 **최대 연소 간격**을 제약해야 한다(권고: `dBU ≤ 4 GWd/t`).

기존 도구의 두 모델이 어디까지 맞는지도 실측으로 갈렸다:

| 격자 | N_sp 모델 예측 | solve-unit 모델 예측 | **실측** |
|---|---:|---:|---:|
| `coarse` | 3.50× | **2.26×** | **2.24×** — solve-unit 모델이 맞다 |
| `three` | 11.67× | 8.60× | **1.63×** — **두 모델 다 5–7배 과대** |

즉 문제는 모델의 형태가 아니라 **적용 범위**다. 연소 간격이 작은 동안에는 solve-unit 비가
비용 비를 잘 예측하고, 간격이 커지면 outer 폭발이 그것을 삼킨다. 본 커밋의
`make_screening_deck.py`는 최대 간격이 4 GWd/t를 넘으면 **위 반례를 인쇄하고 예측을 믿지 말라고
말한다**(`burnup_step_warning`). 모델을 고치는 것(§6.1 Task 4)은 실측 캘리브레이션이 있어야
할 수 있는 일이다.

### 5.4 warm-start flux — 이미 덱 스키마에 있다

`restart` 블록(`IO.cpp:557-606`, `LoadGeometryFromRestart` `IO.cpp:2524-2547`)이
flux/burnup/isotope/TH를 복원한다. 부모 LP의 restart 스냅샷을 자식의 시작점으로 주면
`initial` 버킷(로컬 347 outer = 4,609의 **7.5 %**)과 첫 붕소 탐색이 짧아진다.

**주의 — 이것은 N1이고, 어쩌면 A2다.** 다른 LP의 flux에서 출발한 해가 같은 해로 수렴한다는
보장은 허용오차가 준다(`eigv_tolerance=1e-6`, `search_tol=2e-5`). 그러나 **다근 구간**
(i-SMR CY02 sp1의 `primeXeDamping` 전례, `A2_OUTER_REDUCTION` §5)에서는 출발점이 근을 고른다.
게이트: 같은 LP를 (a) cold start, (b) 임의의 부모에서 warm start로 각각 풀고
**`digest`가 아니라 `keff/CBC/Fq/FΔH`가 수용 문턱 안에 드는지**로 판정한다.

### 5.5 GPU당 다중 프로세스 — C++를 건드리지 않는 폭 레버

`tools/run_multi_gpu_batch.py`는 GPU당 프로세스 하나를 띄우고 `--gpus`가 **서로 다를 것**을
요구한다(`main()`의 distinct 검사). 아레나는 프로세스 수명 싱글턴이고 현재 디바이스에 대해
폭 M으로 한 번 사이징되므로(W4 문서 §3.1), **같은 GPU에 프로세스를 여러 개 붙이면
"선언 폭이 작은 배치 여러 개"가 된다** — §3.1(b)의 곱 구조(폭 비용 × 반복 수)에서
폭 비용만 낮추는 형태다.

필요한 변경은 dispatcher **한 곳**이다:

```text
--procs-per-gpu N       하나의 물리 GPU에 N개의 프로세스열을 붙인다
                        CPU 몫은 (visible_cpus // (len(gpus) * N))로 다시 쪼갠다
                        taskset 구간은 여전히 서로 겹치지 않는다
                        각 프로세스는 --batch-mode (M / N)
MPS                     nvidia-cuda-mps-control 이 떠 있으면 컨텍스트 전환 비용이 사라진다
                        (없어도 동작한다 — 그때는 time-slicing)
```

**게이트는 B0다**: 출력이 단일 프로세스 M64와 `h5diff` 전 데이터셋 Δ=0이어야 한다.
아레나는 프로세스마다 독립이므로 이것은 구조적으로 만족되어야 하며, 만족되지 않으면
그 자체가 결함이다.

---

## 6. Wave 프로그램

**정렬 기준: (얻는 배수) / (인일).** 각 wave는 게이트를 통과하지 못하면 다음으로 가지 않는다.

### 6.1 W1 — 잰다, 그리고 이미 있는 것을 켠다 (3 인일)

가장 싼 배수가 여기 있다. **새 커널 없음. C++ 변경은 결과 필드 추가 하나뿐.**

#### Task 1 — 238에서 케이스 비용 모델을 잰다 (**최우선**)

로컬 회귀 `(c, d) = (0.474 s, 4.805 ms)`는 1080 Ti의 값이다. 238의 `(c, d)`가 이 계획의
모든 배수를 결정한다 — `c`가 크면 L4(호스트 작업 GPU 이식)가 1순위이고, `d`가 크면
L4(outer)와 L5가 1순위다.

```bash
# 238 GPU0, v3 arm (V3_FREEZE §2), 텔레메트리와 타이밍을 섞지 말 것
D=~/t18decks/kngr; O=~/gaout; mkdir -p $O; cd $D
for r in 1 2 3; do
  /usr/bin/time -f "%e" -o $O/full_r$r.wall \
    <bld>/RASBERY --rasi kngr_238.json --raso $O/full_r$r.h5 > $O/full_r$r.log 2>&1
done
RASBERY_STATEPOINT_TELEMETRY=1 <bld>/RASBERY --rasi kngr_238.json \
    --raso $O/full_tel.h5 > $O/full_tel.log 2>&1
python3 tools/case_cost_profile.py $O/full_r1.log $O/full_tel.log --wall-dir $O
python3 tools/case_cost_profile.py $O/*.log --json > $O/ledger.json
```

**수신증**: `[TIMING]` 3줄, `[SPTELEM][SUMMARY]`의 `library_seconds`/`solve_wall`/`io_wall`,
상태점 줄 35개(회귀 입력), `[TRAJECTORY].digest`.
**게이트**: 3회 wall 산포 ≤ 5 %, `digest` 3회 동일, 회귀 rms ≤ 0.15 s.
**산출**: `(c, d)` 그리고 상태점 바닥이 16.9 s 중 몇 %인지 — **이 한 숫자가 W3의 순서를 정한다.**

#### Task 2 — L1(scalar-only)을 배치에서 잰다

```bash
# 같은 64덱 매니페스트를 두 arm으로. 두 arm 모두 텔레메트리 OFF.
python3 tools/run_multi_gpu_batch.py --gpus 0 --batch-width 64 --claim auto \
    --jobs ~/m64jobs.txt --cwd ~/t18decks/kngr --workdir ~/gaout/m64_full \
    --pin taskset --set RASBERY_OMP_THREADS=12 -- ~/<bld>/RASBERY

RASBERY_ALLOW_SCREENING=1 RASBERY_BATCH_LIGHT_RESULT=1 \
RASBERY_BATCH_RECEIPT_JSONL=~/gaout/m64_light/receipts.jsonl \
python3 tools/run_multi_gpu_batch.py --gpus 0 --batch-width 64 --claim auto \
    --jobs ~/m64jobs.txt --cwd ~/t18decks/kngr --workdir ~/gaout/m64_light \
    --pin taskset --set RASBERY_OMP_THREADS=12 -- ~/<bld>/RASBERY
```

**게이트**:
- 두 arm의 **`[TRAJECTORY].digest`가 케이스별로 일치**(64/64). 불일치 1건 = 그 측정 무효.
- light arm의 `[PHYSICS_MODE].screening=true`, full arm은 `false`.
- `[IO_WRITER][SUMMARY].failures=0`, `skipped=0`.
- 판정: `[MULTI_GPU][TOTAL].cases_per_hour`, `mean_width`, `tail_idle_s`,
  그리고 **`[HDF5][LOCK].wait_ms`** — 여기가 진짜 판정 필드다(§3.1(0)).

**기대: 1.0×.** 로컬 8잡/폭 4에서 1.458 GB → 0인데 wall이 279.15 → 286.72 s로 **늘었다**.
238의 M64에서도 writer 스레드 채택이 +0.6 %였다. **따라서 이 태스크의 목적은 L1을
"채택할지 정하는 것"이 아니라 — 채택은 §3.1(d)의 772 PB가 이미 정했다 — 238에서도
처리량이 중립인지 확인하고, 그 실행에서 `[HDF5][LOCK].wait_ms`를 읽어 W2의 목표치를
얻는 것이다.** light arm은 쓰기가 0이므로 그 대기는 **순수하게 읽기 락**이고, 그것이
Task 5가 지워야 할 수다.

**퇴행 게이트**: light arm이 full arm보다 **5 % 이상 느리면** 조사한다 —
`BatchLightResult::Fingerprint64`가 34 MB XS 파일을 `istream::get(char&)`로 **한 바이트씩**
읽는다(`BatchLightResult.h`). 프로세스당 1회로 캐시되지만 워커들이 동시에 처음 진입하면
M개가 각자 한 번씩 한다.

#### Task 3 — 결과 계약 확장 (§5.2)

`include/chiffon/BatchLightResult.h`에 필드 6종 추가 + `tools/test_ga_promotion_gate.py`
갱신. **`lp_canonical_key`와 `cycle_length_efpd`는 evaluator가 계산한다.**
**게이트**: 전 충실도 arm의 `digest` 불변(필드 추가는 물리를 건드리지 않는다),
`tools/test_exact_only_contract.py` PASS.

#### Task 4 — `make_screening_deck.py` 비용 모델 수정 (§5.3) + `--calibrate`

**게이트**: `--calibrate`가 로컬 3덱 로그에서 `(c, d, n0, k, alpha)`를 회귀하고,
`coarse`/`three`의 실측 drive(18.77 / 25.76 s)를 ±15 % 안에서 재현한다.
**이 게이트가 실패하면 `three`류 격자를 GA에 노출하지 않는다.**

### 6.2 W2 — persistent evaluator (8 인일)

#### Task 5 — 호스트 XSLIB 캐시 (계획 Rev.4 §14의 구현)

`Importer::LoadHDF` 결과(`_models`)와 `XSSet.cpp:678-844`의 flatten 산출물을
`std::shared_ptr<const ImmutableLibrary>`로 분리하고, 키 = (정규화 경로, file-id/inode, size,
ns mtime, schema version)로 프로세스 수명 캐시한다.

**차단 요인(측정된 것, 전부 이미 열거되어 있다)**:
- `XSSet`은 `Geometry&`를 들고 복사·이동 불가(`XSSet.h:420-426`) → **캐시 대상은 `XSSet`이
  아니라 `ImmutableLibrary`**다.
- `XSSet::fmap()`/`gmap()`(`XSSet.cpp:5221-5232`)이 `_models` 안으로 **비-const 참조**를
  내준다. 호출자가 없고 인덱싱이 모델 크기를 넘으므로 사실상 죽은 코드이나,
  **캐시 도입 커밋에서 삭제하거나 const로 바꿔야 한다** — 공유 객체로 가는 mutable 별칭은
  캐시가 가질 수 없는 실패 모드다.
- `Isotope` 레지스트리는 이미 프로세스 전역·idempotent다(`Model.h:147-153`).

**게이트 (B0)**: 단일덱 `h5diff` 전 데이터셋 Δ=0 + `digest` 동일, M64 전 케이스 Δ=0.
성능 수신증 셋:

1. `library_seconds`가 **두 번째 케이스부터 ≥ 90 % 감소**.
2. `[HDF5][LOCK].wait_ms`가 Task 2의 light arm 값 대비 **≥ 80 % 감소**.
3. **`Init+IO` 계단이 평평해진다.** 로컬 폭 8의 4.108 → 14.506 s(기울기 1.49 s/케이스)가
   판정 기준이다. 계단이 남아 있으면 캐시가 락 밖으로 나오지 못한 것이다 —
   `LoadHDF` 결과를 캐시해도 **캐시 조회 자체가 `IO.cpp:542`의 guard 안에 있으면**
   줄은 그대로 선다. 캐시 조회는 그 guard **밖**이어야 한다.

**`digest`가 하나라도 움직이면 즉시 되돌린다.**

#### Task 6 — `--evaluator` 서버 모드

한 프로세스가 (라이브러리, 지오메트리 위상, 아레나)를 세우고, stdin/유닉스 소켓/파일 큐로
케이스를 받아 §5.2의 결과를 낸다. `Geometry::Initialize`는 **재진입 불가**
(`Geometry.cpp` 전역이 무조건 `new`, 대응 `delete` 없음)이므로 **캐시된 Geometry는 한 번만
짓고 재-Initialize하지 않는다**. 케이스마다 다시 만드는 것은 `_core`/`_batch`/`_is_fuel`/
`_kbc`/`_kec`/`_hzcore`(싼 스캔)와 `XSSet`의 nxyz-크기 배열·`_comp`/`_asmb`뿐이다.

**게이트 (B0)**: 서버 모드로 낸 64케이스가 `--jobs` 배치와 케이스별 `digest` 동일.
`[RASBERY][EVALUATOR]` 수신증에 `cases`, `library_loads`(= **1**이어야 한다),
`geometry_builds`, `arena_standups`, `p50/p90 case_seconds`.

#### Task 7 — GPU당 다중 프로세스 (§5.5), MPS 유무 양쪽

`tools/run_multi_gpu_batch.py --procs-per-gpu {1,2,4}` × `--batch-width {64,32,16}`.
**게이트 (B0)**: 출력 `h5diff` Δ=0 vs 단일 프로세스 M64. 판정: c/h와 `mean_width`.
**킬 기준**: `--procs-per-gpu 2`가 1.05× 미만이면 이 트랙을 닫고 W4의 slot compaction만 간다.

### 6.3 W3 — 단일 경로 물리 (16 인일, W1 Task 1의 결과가 순서를 정한다)

| Task | 내용 | 게이트 | 겨냥하는 항 (§2.2) |
|---|---|---|---|
| **8** | conditional WHILE 완주(`TASK10_CONDITIONAL_WHILE` §3의 전제 (a) pinned sweep 스테이징, (b) 완료, (c) nodal halt) | **B0** bit-동일 + `graph_instantiation_wall_ms ≤ 250` + `graph_warmup_miss=0` | `d` (drive 2.770 ms/outer) |
| **9** | `nodal_constants` 호스트 호출 2,433/run 제거 | **N1** Gate A/B + `host_body_calls.nodal_constants=0` | `d` (nodal 0.985 ms/outer) |
| **10** | **GPU PPR (계획 Task 19b) 조기 승격** | **N1** Gate A/B + `ppr_host_calls=0` | **`c` (상태점 바닥 0.474 s)** |
| **11** | A2 2단 — outer/상태 238 89 → 59 | **A2** Gate A/B + `staged_relapses` 보고 | `N_outer` (xe 69 %) |
| **12** | warm-start flux (§5.4) | **N1** + 다근 구간 별도 판정 | `initial` 347 outer |

**순서는 W1 Task 1이 정한다.** 238의 `c`(상태점 바닥)가 케이스의 30 % 이상이면 **Task 10을
Task 8보다 먼저** 한다 — 그때는 GPU 커널을 더 빠르게 만드는 것이 호스트 바닥에 가려진다.

각 태스크의 238 판정 명령은 공통이다(arm은 §1.1, 텔레메트리와 타이밍 분리):

```bash
# A/B 한 쌍 — BASE(현행 v3) 와 CAND(태스크 켠 것), 각 3회, 교차 순서
for r in 1 2 3; do
  /usr/bin/time -f "%e" -o $O/base_r$r.wall <bld>/RASBERY --rasi kngr_238.json \
      --raso $O/base_r$r.h5 > $O/base_r$r.log 2>&1
  /usr/bin/time -f "%e" -o $O/cand_r$r.wall env <TASK_ENV> <bld>/RASBERY \
      --rasi kngr_238.json --raso $O/cand_r$r.h5 > $O/cand_r$r.log 2>&1
done
python3 tools/case_cost_profile.py $O/*.log --wall-dir $O          # (c, d) 이동량
python3 tools/test_telemetry_neutrality.py --compare $O/base_r1.log $O/cand_r1.log
h5diff -r $O/base_r1.h5 $O/cand_r1.h5                              # B0 태스크만
python3 tools/gate_a_compare.py $O/base_r1.h5 $O/cand_r1.h5 --per-step   # N1/A2
python3 tools/compare_master_rasbery.py <MAS_SUM> $O/cand_r1.h5 -o $O/master_cand
```

**판정은 wall이 아니라 `(c, d)`의 이동으로 한다.** wall은 238에서도 산포가 있고, 어느 항이
움직였는지 말해 주지 않는다.

### 6.4 W4 — 폭과 대수 (12 인일)

- **Task 13**: slot compaction(Phase 5). grid 비용을 선언 폭이 아니라 실점유에 묶는다.
  게이트: **N1** + `mean_width`가 아니라 **c/h**로 판정. 동반 판정 하나가 더 있다 —
  compaction이 들어가면 **배치 Anderson(`RASBERY_XE_ANDERSON=1`)을 재평가**해야 한다.
  캠페인 §8.2가 배치 기본 OFF로 둔 이유가 정확히 폭 비용이었고, compaction이 그 이유를 없앤다.
- **Task 14**: **multi-GPU 실측 — η를 재고 0.85 가정을 교체한다.**
  ```bash
  for G in "0" "0,1" "0,1,2,3"; do
    python3 tools/run_multi_gpu_batch.py --gpus $G --batch-width 64 --claim auto \
        --jobs ~/m256jobs.txt --cwd ~/t18decks/kngr --workdir ~/gaout/mg_$G \
        --pin taskset --set RASBERY_OMP_THREADS=<budget> -- ~/<bld>/RASBERY
  done
  grep -h "MULTI_GPU\]\[TOTAL\]\|REFILL\]\|BATCH_OCCUPANCY" ~/gaout/mg_*/*.log
  ```
  게이트: 출력이 1-GPU arm과 케이스별 `h5diff` Δ=0. 보고: `cases_per_hour`, `tail_idle_s`,
  그리고 **η = (N-GPU c/h) / (N × 1-GPU c/h)**. **η가 0.85 미만이면 §4의 완주일 표를 다시
  쓴다** — 그 표는 0.85 가정 위에 서 있다.
  호스트 예산은 W4 문서 §3.3의 `plan_host_budget`이 계산한다: 24코어 로컬 호스트는
  GPU 2장·폭 64를 **먹일 수 없다**(GPU당 12 CPU < 워커 64). 4 GPU 캠페인의 최소 호스트는
  96 CPU다.
- **Task 15**: 1,280잡 대량 안정성(W4 문서 §5의 미실행 항목).
  게이트: `duplicates=0`, `stale_tenants=0`, `double_releases=0`, `alloc_in_capture=0`,
  `graph_fallbacks=0`, `rc=0`, `fail_lines=0`. **하나라도 0이 아니면 그 런은 유효한 측정이
  아니다** — 성능 수치를 읽기 전에 이것부터 읽는다.
- **Task 16**: **island arm 실측.** GA 연구설계가 요구한 비교(`GA_deep_research_report.md`
  §"Multi-GPU를 사용할 경우 island GA를 별도 실험군으로") 는 evaluator 쪽에서는 이미 가능하다 —
  dispatcher가 GPU당 프로세스이므로 GPU당 island가 자연스럽다. evaluator가 해야 할 일은
  **island 간 migration을 위한 케이스 주입 경로**(공유 큐에 케이스를 밀어 넣는 것)뿐이고,
  `queue.json` flock 커서가 이미 그 형태다. 게이트: 두 dispatcher를 같은 큐에 붙였을 때
  중복 claim 0(W4 문서 §3.1의 재개 시나리오와 같은 시험).

### 6.5 게이트 등급 요약

| 등급 | 의미 | 이 계획에서 |
|---|---|---|
| **B0** | 이전 경로와 **비트가 같다** | L1(digest 동일 — 이미 통과), L2, L5(다중 프로세스), L6 |
| **N1** | 결정론적 GPU 기준선 전이 — Gate A(크기) + Gate B(MASTER 방향) | L4의 대부분, L5(compaction), L8 |
| **A2** | 답이 바뀐다 — 별도 브랜치 + 명시적 롤백 + Gate A/B | L3(충실도), L4의 A2 2단 |

---

## 7. 위험

| 위험 | 완화 |
|---|---|
| **스크리닝 결과가 수용 표에 들어간다** | `RASBERY_ALLOW_SCREENING` fail-closed 가드를 **끄지 않는다**. `[PHYSICS_MODE].screening`과 `requires_exact_rerun`이 모든 결과에 실린다. elite는 가드 없이 재실행한다 |
| **FP ranking 불안정** | `fp_class` 필드(§5.2) + elite FP64 재계산 + fitness 차가 수치 불확도보다 작으면 동률 처리. `RASBERY_GPU_CMFD_FP32`는 GA arm에서 **기본 OFF**로 둔다 |
| **호스트마다 outer 수가 다르다** | A2 arm에서 outer 수는 호스트 상수가 아니다(`V3_FREEZE` §8.4). 캠페인의 모든 판정은 **같은 호스트 안의 A/B**로만 한다 |
| **coarse 격자가 다른 LP 순위를 준다** | 순위 안정성은 GA의 검증 항목이지만 evaluator가 재료를 줘야 한다: 같은 상위 100 LP를 `coarse`와 `full`로 풀고 Spearman ρ를 보고한다(W1 Task 4의 부수 산출) |
| **warm-start가 다른 근을 고른다** | §5.4의 별도 게이트. 다근 구간에서는 cold start로 강등한다 |
| **1,280잡에서 슬롯 누수** | `[REFILL]`의 버그 카운터 3종이 0이 아니면 그 런은 유효한 측정이 아니다(W4 문서 §1.4) |

---

## 부록 A. 로컬 측정 스크립트

`~/gaplan_measure.sh` / `~/gaplan3.sh` / `~/gaplan4.sh` / `~/gaplan5.sh`(로컬 WSL).
공통 형태:

```bash
run () { local name=$1 deck=$2; shift 2
  /usr/bin/time -f "%e" -o $O/$name.wall env "$@" \
     $BIN --rasi $deck --raso $O/$name.h5 > $O/$name.log 2>&1
  echo "$name rc=$? wall=$(cat $O/$name.wall)"
}
run full_tel   kngr_238.json          RASBERY_STATEPOINT_TELEMETRY=1
run pinoff_tel kngr_238_pinoff.json   RASBERY_STATEPOINT_TELEMETRY=1
run light_tel  kngr_238.json          RASBERY_ALLOW_SCREENING=1 \
                                      RASBERY_BATCH_LIGHT_RESULT=1 \
                                      RASBERY_BATCH_RECEIPT_JSONL=$O/light.jsonl \
                                      RASBERY_STATEPOINT_TELEMETRY=1
```

덱 변형:

```bash
# 핀출력 OFF (스케줄 동일)
python3 -c 'import json;d=json.load(open("kngr_238.json"));
[e["print"].__setitem__("pin-wise information",False) for e in d["schedule"] if "print" in e];
json.dump(d,open("kngr_238_pinoff.json","w"),indent=1)'
# 연소격자 변형
python3 tools/make_screening_deck.py kngr_238.json --mode coarse -o kngr_coarse.json
```

배치 A/B(§3.1(0))는 `--jobs` 매니페스트로 돌린다:

```bash
for i in $(seq 0 7); do cp kngr_238.json k$i.json; done
for i in $(seq 0 7); do echo "k$i.json $O/b/k$i.h5"; done > $O/m8.txt
RASBERY_OMP_THREADS=1 /usr/bin/time -f "%e" -o $O/b8w4_full.wall \
    $BIN --jobs $O/m8.txt --batch-mode 4 > $O/b8w4_full.log 2>&1
RASBERY_OMP_THREADS=1 RASBERY_ALLOW_SCREENING=1 RASBERY_BATCH_LIGHT_RESULT=1 \
RASBERY_BATCH_RECEIPT_JSONL=$O/b8w4_light.jsonl /usr/bin/time -f "%e" \
    -o $O/b8w4_light.wall $BIN --jobs $O/m8.txt --batch-mode 4 > $O/b8w4_light.log 2>&1
```

## 부록 B. `tools/case_cost_profile.py`

수신증만 접는다. 새 계기를 넣지 않았고, 프로세스 wall은 `--wall-dir`의 사이드카가 없으면
**비워 둔다**. 케이스 하나짜리 로그와 배치 로그를 구분한다 — 배치 로그에는 케이스당
`[TIMING]` 한 벌씩이 들어 있으므로 **케이스 원장을 만들지 않고** 배치 요약만 낸다
(첫 케이스의 `Drive()`를 배치 전체 wall에 대고 빼면 `outside_drive`가 배치 전체가 된다).

```bash
python3 tools/case_cost_profile.py ~/gaplan/*.log --wall-dir ~/gaplan
python3 tools/case_cost_profile.py ~/gaplan --json > ledger.json
```

본 문서의 모든 수치를 낸 실제 출력:

```text
        run    n     wall   drive  outside  init+io  library   solve  io(drv)  wr_busy    out_bytes   sp  outers  out/sp  ms/outer  width  digest
   full_tel    1    44.67   42.07     2.60     1.25     0.45   38.20     2.60      607    182266343   35    4609   131.7      8.29         814201df0583e1d2
 pinoff_tel    1    41.91   39.93     1.98     1.13     0.30   38.74     0.05      191     11954663   35    4609   131.7      8.40         814201df0583e1d2
  light_tel    1    44.84   41.30     3.54     1.52     0.32   39.36     0.43        0            0   35    4609   131.7      8.54         814201df0583e1d2
 coarse_tel    1    20.52   18.77     1.75     0.97     0.28   16.91     0.89      235     52078418   10    2334   233.4      7.25         8406593c9e36cc4d
  three_tel    1    27.58   25.76     1.82     2.68     0.35   22.86     0.21       32     15625799    3    5104  1701.3      4.48         e32a0a4cc3bd5e13
  b8w4_full    8   279.15                                                       17666   1458130744   35    4609                     1.10  814201df0583e1d2
 b8w4_light    8   286.72                                                           0            0   35    4609                     1.12  814201df0583e1d2
  b8w8_full    8   206.94                                                       22709   1458130744   35    4609                     1.49  814201df0583e1d2

   full_tel  amortisable   3.85 s (  8.6 %)  physics  85.5 %  output   5.8 %
 pinoff_tel  amortisable   3.12 s (  7.4 %)  physics  92.4 %  output   0.1 %
  light_tel  amortisable   5.06 s ( 11.3 %)  physics  87.8 %  output   0.9 %
 coarse_tel  amortisable   2.72 s ( 13.2 %)  physics  82.4 %  output   4.3 %
  three_tel  amortisable   4.50 s ( 16.3 %)  physics  82.9 %  output   0.8 %
  b8w4_full  BATCH cases 8  c/h 103.2  eff 34.89 s/case  drive/case  95.9-151.8 s  init/case  1.31- 5.12 s  hdf5_lock_wait  15.2 s
 b8w4_light  BATCH cases 8  c/h 100.4  eff 35.84 s/case  drive/case  97.9-147.8 s  init/case  1.94-15.43 s  hdf5_lock_wait  36.9 s
  b8w8_full  BATCH cases 8  c/h 139.2  eff 25.87 s/case  drive/case 161.3-191.2 s  init/case  4.11-14.51 s  hdf5_lock_wait  63.8 s
```

## 부록 C. 코드 인용 (tip `feff7e7`)

| 사실 | 위치 |
|---|---|
| `core`/`batch` 파싱, 유일 소비자 | `IO.cpp:654-666`, `XSSet.cpp:636-676` |
| `core` 파생 Geometry 멤버 넷 | `Geometry.cpp:155-156, 585-628` |
| `ReadInput` 전체가 `Hdf5Guard` 아래 | `IO.cpp:542` |
| 34 MB 라이브러리 로드(파일명만의 순수 함수) | `XSSet.cpp:586`, `Importer.h:821-901` |
| 라이브러리 flatten(라이브러리+ng+niso의 순수 함수) | `XSSet.cpp:678-844` |
| 디바이스 flat XS 라이브러리 캐시(존재) | `CudaXsReconBackend.cu:561, 2720-2780` |
| 호스트 XSLIB 캐시(계획, 미구현) | `…ACCELERATION_PLAN_REV4_KO.md:900-935` |
| `Geometry`/`XSSet`/`IO`가 `Drive()` 지역 변수 | `Driver.h:3837-3845` |
| `library_seconds` 계기 | `Driver.h:3842-3847` |
| 배치 중첩 OMP 금지 | `main.cpp:626-630` |
| 스크리닝 fail-closed 가드 | `main.cpp:409-433` |
| scalar-only 결과 작성 | `Driver.h:4108-4118`, `BatchLightResult.h` |
| PPR 무조건 실행(인쇄와 무관) | `Driver.h:4065-4082` |
| `[TRAJECTORY]` digest | `Driver.h:4305` |
| `mean_width` 수신증 | `CudaBICGBackend.cu:5525` |
