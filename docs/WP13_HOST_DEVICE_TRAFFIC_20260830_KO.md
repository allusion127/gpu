# WP13 — host↔device 전송 귀속과 첫 B0 소거 (2026-08-30, KO)

브랜치 `codex/exact-throughput-campaign`. 이 문서는 **소스 인구조사(census)** 가 본체이고,
구현은 그 인구조사가 지목한 것 중 **컴파일러 없이 확실하게 할 수 있는 만큼**이다.
`docs/WP_PLAN_REVIEW_AND_TRACKER_20260831_KO.md`는 건드리지 않았다.

---

## 0. 요약 (읽는 순서)

1. **D2H 28.6 GB의 85 %는 단 한 곳이다** — `solveFlatXs`의 micx/lmpx 다운로드
   (호출당 59.5 MB × 20 copy, 런당 ~24.4 GB). 나머지 D2H 전체를 합쳐도 그 1/6이다.
2. **H2D 11.7 GB / 54,918 copy 중 절반은 소스만으로 귀속되지 않는다.** 평균 크기 280 KB인
   ~25 K copy / ~7.0 GB가 남고, 그 크기는 jnet(427 KB)·flux(135 KB)·xssm(270 KB)와
   일치한다 — 즉 nodal/outer-segment의 per-outer 배열이며, 발화 빈도가
   `canonicalElidesUpload`라는 **런타임 술어**로 결정되어 소스에서 읽을 수 없다.
   이 잔차를 확정하는 것이 이번 커밋의 `[RASBERY][XFER]` 영수증이다.
3. **wall을 지배하는 것은 바이트가 아니라 sync다** — `cudaStreamSynchronize` 6.19 s
   (9,477 회 × 653 µs) > GPU 커널 시간 전체(1.61 s). 이번 커밋은 sync를 **하나도**
   제거하지 않는다. 그건 persistent-outer의 일이고, 이 문서의 로드맵 6번이다.
4. 이번에 넣은 것: `RASBERY_GPU_XFER_ELIDE=1` — **순수 전송 소거(B0)**. copy **횟수**를
   ~14 K개(전체의 13 %) 줄인다. 바이트는 거의 줄이지 않는다. 그건 로드맵 1번이다.

---

## 1. 기준 사실 (nsys, 238, v4 단일 실행)

env = PROD(§1.1, `docs/PRICING_PROD_20260830_KO.md`) + `RASBERY_GPU_CRAM=1
RASBERY_GPU_CMFD_FUSE=15 RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1
RASBERY_GPU_XE_TXN=1`. KNGR cycle-1, 35 statepoint, wall 14.4 s.

| 항목 | 값 |
|---|---:|
| outer | 4,377 |
| CMFD sweep / BiCG iteration | 18,627 / 74,508 (= 정확히 4.00 iter/sweep) |
| Xe device step (`txn_steps`) | 1,117 |
| depletion step (predictor=corrector 호출) | 34 |
| statepoint | 35 |
| GPU 커널 시간 합 | ~1.61 s (wall의 11 %) |
| `cudaStreamSynchronize` | **6.19 s / 9,477 회 / 653 µs 평균** |
| `cudaMemcpyAsync` | 2.27 s / **110,078 회** |
| D2H | **28.6 GB / 55,183 copy** |
| H2D | **11.7 GB / 54,918 copy** |
| outer당 | 12.6 copy · 2.2 sync · 6.5 MB D2H |

커널 분포: `kernelFlatXs` 58.9 %, Xe/TXN 20.8 %, CRAM 14.8 %, CMFD/outer 5.1 %,
nodal+PPR < 0.2 %.

## 2. 덱 상수 (모든 수식의 기호)

`nxyz = 8,451`(313 × 27), `NG = 2`, `nsurf = 26,692`, `NISO = 39`, `NXS = 11`,
`N_ACTIVE = 9`, `stream_len = 18,075`, CRAM `first = iI135 = 3`.
출처: `docs/GPU_RASBERY_METHODOLOGY_BENCHMARK_20260824_KO.md:24-25`,
`docs/WP5_FLATXS_CTA_20260831_KO.md:15`, `docs/WP6_PPR_DEVICE_LOOP_20260831_KO.md:15`,
`src/XsReconKernel.h:41-43`, `src/CudaCramBackend.cu:36-37`.

> **"574,668 CRAM 노드"는 메시 크기가 아니다.** CRAM 영수증의 `nodes:574668`은
> 누적 노드-방문 카운터로 `8,451 × (34 predictor + 34 corrector)`다. 메시는 8,451이다.

파생 블록 (모두 `sizeof(double) = 8`):

| 기호 | 정의 | doubles | bytes |
|---|---|---:|---:|
| `nng` | `nxyz·NG` — flux/phif/xs 한 종류 | 16,902 | 135,216 |
| `nsg` | `nsurf·NG` — jnet/dhat/dtil/phis | 53,384 | **427,072** |
| `mat` | `NG²·nxyz` — CMFD diag/udiag | 33,804 | 270,432 |
| `cpl` | `NG·6·nxyz` — CMFD cc | 101,412 | 811,296 |
| `ssm` | `NG²·nxyz` — 산란행렬 1종 | 33,804 | 270,432 |
| `mic` | `NISO·NG·nxyz` — micx 1 슬롯 | 659,178 | **5,273,424** |
| `msm` | `NISO·NG²·nxyz` — micx 산란 | 1,318,356 | **10,546,848** |
| `iden` | `NISO·nxyz` | 329,589 | 2,636,712 |
| `rows` | `(NISO−3)·nxyz` — CRAM iden 출력 | 304,236 | 2,433,888 |

복합 블록:

| 블록 | 구성 | copy | bytes |
|---|---|---:|---:|
| **flatxs 다운로드 (전체)** | 9·(lmp+mic) + ssm + msm + 11·lmp + ssm + 3·nxyz | **33** | **61,455,672** |
| ┗ 그중 micx/lmpx | 9·(lmp+mic) + ssm + msm | 20 | **59,495,040** |
| ┗ 그중 macro xs + Xe 3행 | 11·lmp + ssm + 3·nxyz | 13 | 1,960,632 |
| **micx H2D 블록** | 11·mic + msm + 11·lmp + ssm | 24 | 70,312,320 |
| **state H2D 블록** | 11·lmp + ssm + iden | 13 | 4,394,520 |
| **ref H2D 블록** | 9·mic + msm + 9·lmp + ssm | 20 | 59,495,040 |
| **flatxs per-call 입력** | 3·nxyz(d) + 3·n_nodes(i) + stream(i+2d) | 9 | 665,736 |
| **drainXeCommit** | 11·lmp + ssm + 3·nxyz | 13 | 1,960,632 |

---

## 3. 귀속표 — PROD v4 경로의 모든 전송/동기화 지점

**cadence 약어**: `1×` 프로세스 1회, `sp` statepoint(35), `dep` depletion step(34),
`out` outer(4,377), `swp` CMFD sweep(18,627), `xe` Xe txn step(1,117),
`fx` flatxs 호출(런당 **F ≈ 410**, §4에서 유도), `seg` outer segment(≈547).

### 3.1 `src/CudaXsReconBackend.cu` — 지배적

| 지점 | 함수 | dir | bytes/call | copy/call | cadence | 런 합 (copy / bytes) |
|---|---|---|---:|---:|---|---:|
| **3467-3471** | `solveFlatXs` micx/lmpx 다운로드 | **D2H** | 9·(135,216+5,273,424)+270,432+10,546,848 | **20** | fx | **8,200 / 24.39 GB** |
| 3474-3477 | `solveFlatXs` xs/xs_ssm/iden 다운로드 | D2H | 11·135,216+270,432+202,824 | 13 | fx | 5,330 / 0.80 GB |
| 3480 | `solveFlatXs` drain | sync | — | — | fx | 410 sync |
| 2545 `drainXeCommit` (2984·3160에서) | `xeTransaction`/`xeCommit` | D2H | 1,960,632 | 13 | xe | 14,521 / 2.19 GB |
| 3071-3073 | `xeTransaction` ctl+solved | D2H | 88 | 2 | xe | 2,234 / 0.1 MB |
| 3075 `xeSync` | `xeTransaction` | sync | — | — | xe | 1,117 sync |
| 3355-3357 | flatxs wvfr/dmod/bppm | H2D | 67,608 | 3 | fx | 1,230 / 0.083 GB |
| 3397-3399 | flatxs nodes/off/cnt | H2D | 33,804 | 3 | fx | 1,230 / 0.042 GB |
| 3401-3406 | flatxs stream did/x/scale | H2D | 72,300 / 144,600 ×2 | 3 | fx | 1,230 / 0.148 GB |
| 3296-3306 | flatxs **ref 블록** | H2D | 59,495,040 | 20 | sp | 700 / **2.08 GB** |
| 3324-3328 / 2432-2447 | **micx 블록** (`upload()`) | H2D | 70,312,320 | 24 | 잔차 (§4) | — |
| 3337-3339 / 2449-2455 | **state 블록** (`upload()`) | H2D | 4,394,520 | 13 | 잔차 (§4) | — |
| 2456 | `stage()` phif | H2D | 135,216 | 1 | xe | 1,117 / 0.151 GB |
| 3134-3199 | flatxs 라이브러리 테이블 | H2D(sync) | 라이브러리 형상 | 21 | 1× | 21 / ~수십 MB |
| 3516-3536 | nodal 지오메트리 | H2D | 1,249,128 합 | 7 | 1× | 7 / 1.2 MB |
| 3553 | nodal 상수 9종 | H2D | 405,648 | 9 | 드묾 | — |
| 3752/3762 | nodal `enqueue_full` jnet/flux 업로드 | H2D | 427,072 / 135,216 | 2 | out**†** | 잔차 (§4) |
| 3785/3794 | nodal `enqueue_full` jnet/phis 다운로드 | D2H | 427,072 | 2 | out**†** | 잔차 (§4) |
| 4042 | nodal 최종 drain | sync | — | — | out**†** | 잔차 |
| 1183-1699 | `NodalArena` 전체 | — | — | — | **DEAD** | batch width > 1 전용 |
| 3630-3689 (`hybrid_even`) · 4063-4155 (`solveNodalPost`) | — | — | — | **DEAD** | `NODAL_FULL=1` |
| 2697/2760/2802/2829/2861/2909 (Xe 분할 arm) | — | — | — | **DEAD** | `XE_TXN=1` |

**†** canonical 바인딩이 잡히면 이 넷 + reigv + sync가 **전부 0**이 된다
(`gpu::canonicalElidesUpload/Download`, `src/Driver.h:4216-4226`). 잡히지 않은 outer에서만
발화하며, 그 비율이 §4의 잔차다.

### 3.2 `src/CudaBICGBackend.cu` — copy **횟수**가 크고 바이트는 작다

`RASBERY_GPU_CMFD_SWEEP=1`(PROD §1.1)이므로 device-sweep arm이 산다. `issueUploads`
(3803-3846)와 예외 operator 다운로드(4853-4877, `state == 2`)는 **DEAD**.

| 지점 | 함수 | dir | bytes/call | cadence | 런 합 |
|---|---|---|---:|---|---:|
| **4812** | `issueFluxDownloads` out_phi | **D2H** | `n·8` = 135,216 | out | 4,377 / **0.592 GB** |
| 4793 | `issueSweepDownloads` sweep_out | D2H | `kSweepCount·8` = 152 | out | 4,377 / 0.67 MB |
| **4159** | `enqueue_outer` host_status | **D2H** | `slots·64` = 64 | swp | **≈11.4 K / 0.7 MB** |
| 4765 | `issueSweepUploads` sweep_in | H2D | 152 | out | 4,377 / 0.67 MB |
| **3786** | `buildSlotMap` d_slot_map | H2D | `slots·4` = 4 | out | 4,377 / 17 KB |
| **4605** | `issueSweepUploads` device_active | H2D | 4 | out | 4,377 / 17 KB |
| **4607** | `issueSweepUploads` device_assembly_active | H2D | 4 | out | 4,377 / 17 KB |
| 4609 | `issueSweepUploads` sweep_halt | H2D | 4 | out | 4,377 / 17 KB |
| 4636-4696 `push_pending` | xsnf/xsrf/xssm/dtil | H2D | 967,936 합 | XS 변화 | ≈1,640 / 0.40 GB |
| 4616-4628 `pushOrSkip` | chif/vol | H2D | 135,216 / 67,608 | mirror | ≈0 (거의 skip) |
| 4655/4699 | dhat / psi | H2D | 427,072 / 67,608 | — | 0 (`*_resident`) |
| 3234-3257, 3406 | 지오메트리·slot map 초기 | H2D | 710 KB 합 | 1× | 6 |
| 5842(`syncSweepStream`) | — | sync | — | seg | ≈547 |
| 4884(`drain`), 5771 | — | sync | — | — | **DEAD** |

**operator는 절대 업로드되지 않는다** — `cmfd_assemble_operator_2g`가 device에서 짓는다.
`diag`/`cc`/`udiag`(270+811+270 KB)의 D2H는 `state == 2`에서만 나고, `BICGCMFD.cpp:722`가
"1,690 outer 중 0회"로 기록한다.

### 3.3 `src/CudaOuterGraph.cu` — 이미 대부분 소거되어 있다

| 지점 | 내용 | dir | bytes | cadence | 런 합 |
|---|---|---|---:|---|---:|
| 2383 | flux 업로드 (generation gate) | H2D | 135,216 | 잔차 | — |
| 2421/2429 | dhat/psi 미러 | D2H | 427,072 / 67,608 | `!sweep_will_enqueue`인 outer만 | 드묾 |
| 2559 | jnet bridge 다운로드 | D2H | 427,072 | `!canonical_now` | 잔차 |
| 2100 | jnet bridge 업로드 | H2D | 427,072 | `!canonical_now` | 잔차 |
| 2068 | cusping 후 dtil 재업로드 | H2D | 427,072 | cusping 발화 시 | ≈0 |
| 2148 | segment exit word | D2H | `sizeof(DeviceOuterSegment)` | out | 4,377 / 작음 |
| 1520-1548 | segment 진입 스칼라 6종 | H2D | ≤ 수십 B | seg | ≈3,282 / 작음 |
| 2844 / 2850 | sweep accumulator + hostfree exit | D2H+sync | 작음 | seg | 547 + 547 sync |
| 3005-3021 | exit 관측 4종 + sync | D2H+sync | 작음 | seg | 2,188 + 547 sync |

### 3.4 `src/CudaCramBackend.cu` — H2D 바이트 2위

| 지점 | 내용 | dir | bytes/call | copy | cadence | 런 합 |
|---|---|---|---:|---:|---|---:|
| 896 / 999 | **micro-XS 4 슬롯** | H2D | 5,273,424 | 4 | dep ×2 | **272 / 1.37 GB** |
| 892 / 990 / 994 | iden / iden_bos / iden_pred | H2D | 2,636,712 | 1 | dep | 102 / 0.27 GB |
| 891, 986-999 | phif / flux·xskf eos·bos | H2D | 135,216 | 1~4 | dep | 170 / 0.02 GB |
| 874-886 | 라이브러리 + dfac/vol | H2D | 159,824 합 | 6 | 1× | 6 |
| 906 | BOS 스냅샷 | **D2D** | 21,093,696 | 4 | dep | 136 / 0.68 GB (PCIe 아님) |
| 945 / 1037 | iden 출력 | D2H | 2,433,888 | 1 | dep ×2 | 68 / 0.166 GB |
| 1040 | burn 출력 | D2H | 33,804 | 1 | dep | 34 / 1.1 MB |
| 926 / 1024 | 통계 | D2H | 24 | 1 | dep ×2 | 68 / 1.6 KB |
| 929/950/1027/1045 | — | sync | — | 4 | dep | **136 sync** |

**per-outer 전송 0.** 1.37 GB는 `_micx_generation`이 모든 FlatXS solve에서 증가하기
때문에(`src/XSSet.cpp:3117`) 68회 전부 재전송되는 것이다.

### 3.5 `src/CudaPprBackend.cu` — 작다, 그리고 statepoint당이다

`RASBERY_PPR_MODE=master`이므로 계수 다운로드는 5종(`c,bt,phic,q,l`), `p`/`a`는 생략.
`RASBERY_GPU_PPR_RECON` 부재 → `reconstructPinPower` 전체(2249·2382·memset) **DEAD**,
핀출력 재구성은 host. `PPR_GRAPH=1` → 2043·2063의 per-Picard D2H도 **DEAD**.

| 지점 | dir | bytes | copy | cadence | 런 합 |
|---|---|---:|---:|---|---:|
| 2174-2178 | D2H 계수 5종 | 5,949,504 | 5 | sp | 175 / 0.208 GB |
| 1870-1901 | H2D phif/phis/jnet/xs·chif/crdf | 1,665,440 합 | 8~9 | sp | ≈310 / 0.058 GB |
| 1846-1849 | H2D 지오메트리 | 419,107 합 | 4 | 1× | 4 |
| 1495 `syncStream` | sync | — | — | sp | **35 sync** |

**per-outer 전송 0.** nsys의 "nodal+PPR < 0.2 %"와 일치한다.

---

## 4. nsys 대조 — 맞는 것과 **맞출 수 없는 것**

### 4.1 D2H: 바이트 0.8 % 이내로 닫힌다

flatxs 호출 수 `F`는 계측되지 않으므로 **바이트 항등식에서 역산**한다:

```
28.6 GB = F·61,455,672            (flatxs)
        + 1,117·1,960,720         (Xe txn)        = 2.190 GB
        + 4,377·135,216           (CMFD flux)     = 0.592 GB
        + 0.166 + 0.208           (CRAM + PPR)    = 0.374 GB
        + ε
⇒ F ≈ 410
```

`F ≈ 410`(statepoint당 ≈11.7회 — boron 탐색 시행 + T/H 반복 + 고갈, `UpdateFlatXS`의
9개 호출처와 정합)를 넣으면 D2H 합 **28.36 GB**, 측정 28.6 GB 대비 **−0.8 %**.
남는 0.24 GB는 canonical이 잡히지 않은 outer의 nodal jnet/phis 다운로드
(854,144 B/outer)로 **약 280 outer**에 해당한다. 이는 §4.2의 가설과 같은 대상을 가리킨다.

copy **횟수**는: 8,200+5,330(flatxs) + 16,755(Xe) + 4,377(flux) + 4,377(sweep_out)
+ 4,377(segment exit) + 170+210(CRAM+PPR) = 43,796. 측정 55,183과의 차 **11,387**을
`host_status`(sweep당 64 B)로 귀속한다 — outer당 2.6회이며, `unroll = _ncmfd − iout`가
5→1로 줄어드는 것과 정합적이다. **이 값은 유도이지 측정이 아니다.**

### 4.2 H2D: 바이트·횟수 모두 **절반이 남는다 — 귀속하지 못했다**

cadence가 확실한 지점만 합하면:

| | copy | bytes |
|---|---:|---:|
| CMFD per-outer 5종 | 21,885 | 0.001 GB |
| CMFD `push_pending` | ≈1,640 | 0.40 GB |
| flatxs per-call 입력 9종 | 3,690 | 0.27 GB |
| flatxs ref 블록 | 700 | 2.08 GB |
| Xe phif | 1,117 | 0.15 GB |
| CRAM | 584 | 1.73 GB |
| PPR | 320 | 0.06 GB |
| 1회성 (nodal geom, lib, dep) | ≈30 | ≈0.01 GB |
| **합** | **29,966** | **4.70 GB** |
| **측정** | **54,918** | **11.7 GB** |
| **잔차** | **24,952** | **7.00 GB** |

**잔차의 평균 크기는 280 KB다.** micx 블록(70.3 MB/24 copy)이나 state 블록
(4.39 MB/13 copy)으로는 이 조합이 **수학적으로 불가능하다** — 두 미지수로 연립하면
음수 해가 나온다. 280 KB는 `jnet`(427 KB) · `flux`(135 KB) · `xssm`(270 KB) ·
`dtil`(427 KB)의 대역이고, 이들은 전부 **nodal `enqueue_full` + outer-segment의
per-outer 배열**이다.

> **가설(측정 대상)**: `canonicalElidesUpload/Download`가 소스를 읽고 기대한 것만큼
> 자주 참이 되지 않는다. 참이라면 이는 새 기능이 아니라 **residency 버그**이며,
> 로드맵 1번보다 먼저 볼 값어치가 있다. 소스만으로는 결론 낼 수 없으므로
> `[RASBERY][XFER]`가 이 커밋에 들어간다.

### 4.3 sync: 9,477 중 ~2,300만 이름이 붙는다

flatxs 410 + Xe 1,117 + CRAM 136 + PPR 35 + segment(hostfree exit·exit 관측) ≈1,094
+ `syncSweepStream` ≈547 = **≈3,339**. 나머지 **≈6,100**은 outer당 1.4회이며,
가장 유력한 후보는 §4.2와 같은 대상(canonical이 안 잡힌 outer의 nodal 최종 drain,
`CudaXsReconBackend.cu:4042`)이다. **653 µs × 6,100 = 4.0 s** — wall의 28 %가
이름 없는 sync에 있다.

### 4.4 상위 3 (바이트 / 횟수)

**바이트 기준**

| # | 지점 | outer당 copy | 런 GB |
|---|---|---:|---:|
| 1 | `CudaXsReconBackend.cu:3467-3471` flatxs micx/lmpx D2H | 0 (fx당 20) | **24.39** D2H |
| 2 | (미귀속) nodal/outer-segment per-outer 배열 H2D+D2H | ~5.7 | **~7.0** H2D |
| 3 | `CudaXsReconBackend.cu:3296-3306` flatxs ref 블록 H2D | 0 (sp당 20) | **2.08** H2D |
| 3′ | `CudaXsReconBackend.cu:2545` Xe `drainXeCommit` D2H | 0 (xe당 13) | 2.19 D2H |
| 3″ | `CudaCramBackend.cu:896/999` CRAM micro-XS H2D | 0 (dep당 8) | 1.37 H2D |

**횟수 기준**

| # | 지점 | outer당 | 런 copy |
|---|---|---:|---:|
| 1 | CMFD per-outer 5종 (3786·4605·4607·4609·4765) | **5** | **21,885** H2D |
| 2 | Xe `drainXeCommit` 13종 (`:2545`) | — | 14,521 D2H |
| 3 | flatxs 다운로드 33종 (`:3467-3477`) | — | 13,530 D2H |
| 3′ | CMFD `host_status` (`:4159`) | ~2.6 | ≈11,387 D2H |

---

## 5. 이번 커밋이 구현한 것 — `RASBERY_GPU_XFER_ELIDE=1`

**등급 B0, 그리고 `trajectory::kArmEnv`에 넣지 않았다.** 모든 소거는
"업로드하려는 바이트가 device가 이미 들고 있는 바이트와 **memcmp 동일**할 때만 건너뛴다"
이다. 소거 후 device 내용은 소거 전과 **같은 비트**이므로 어떤 커널도 두 실행을 구별할 수
없고, 따라서 trajectory를 움직일 수 없다. 움직일 수 없는 knob을 case key에 넣으면
같은 실행 두 개를 서로 다른 캐시 항목으로 만든다.

**성립 조건은 하나뿐이고, 지점마다 명시했다: 그 device 버퍼에 device 측 writer가 없을 것.**
host가 마지막으로 보낸 바이트의 그림자는, 그 사이에 커널이 썼다면 device 내용에 대한
참인 진술이 아니다.

### E1 — CMFD per-outer 마스크 3종 (`src/CudaBICGBackend.cu`)

`BatchCore::pushDeviceReadOnly`(:3735) 신설, 세 지점을 통과시켰다.

| 지점 | 버퍼 | 왜 안전한가 |
|---|---|---|
| :3786 | `d_slot_map` | 모든 커널 시그니처에서 `const int* __restrict__ slot_map` (`RASBERY_CMFD_SLOT_ARGS`, :471-478) |
| :4605 | `device_active` | `const std::uint32_t* __restrict__ active` (:1173, 1505, 1533, 1560) |
| :4607 | `device_assembly_active` | `const std::uint32_t* __restrict__` (:2177) |
| :4609 | `sweep_halt` | **제외.** `initialize_solver_state`가 올리고 `issueSweepDownloads`가 memset한다 — device writer가 있다 |

단일 덱에서 이 셋은 각각 4 B이고 런 내내 **불변**이다(`compact == false` → slot map은
항등, 참여 마스크는 `{1}`). `issueUploads`(:3810, PROD에서 dead)도 **같은 그림자**를
통과시켰다 — 두 writer가 그림자 하나를 공유하지 않으면 다른 쪽의 소거가 틀린다.

그림자는 staging 버퍼가 아니라 core 소유의 `std::vector`다. `stageActive()`/`stageSlotMap()`은
in-flight copy의 source가 host write에 밟히지 않도록 lane별로 회전하는 pinned 버퍼이므로,
그것과 비교하는 것은 다른 lane이 곧 덮어쓸 버퍼와 비교하는 것이다.

### E2 — flatxs per-call 입력 9종 (`src/CudaXsReconBackend.cu`)

`Impl::uploadGuarded`(:2626) 신설 + `ByteExactMirror` 9개.
`:3355-3357`(wvfr/dmod/bppm), `:3397-3399`(nodes/off/cnt), `:3401-3406`(stream did/x/scale).
전부 `fxs::FlatXsView`에서 `const double*` / `const int*`(`src/FlatXsKernel.h:170-187`)
— 커널 입력 전용이다. **재할당 경로 3곳에서 무효화**한다: `ensure()`의 지오메트리 regrow,
`solveFlatXs`의 node·stream cap regrow, `dev_pernode` 최초 할당.

플래그가 꺼져 있으면 그림자를 **읽지도 쓰지도 않는다** — OFF arm은 출하된 코드 그대로이고,
그래서 A/B가 bookkeeping이 아니라 소거를 잰다.

### E3 — `[RASBERY][XFER]` 영수증 (`src/XferLedger.h`)

`d2h_calls, d2h_bytes, h2d_calls, h2d_bytes, syncs, elided_calls, elided_bytes`
+ `elision_tests` + `elision_hit_rate` + `elide_arm` + `covered`.
`covered`가 계측된 지점을 이름으로 밝힌다 — 이 영수증은 프로세스 총계가 아니며 nsys와
같지 않다. **그 차이가 §4.2의 잔차를 좁히는 도구다.**
`main.cpp`의 세 종료 경로(단일·batch·evaluator) 모두에서 출력한다.

### 5.1 기대 절감 — 정직하게

| | 제거되는 것 | 가격 |
|---|---|---|
| E1 | H2D copy **13,131**회 (전체 H2D의 24 %, 전체 copy의 12 %), 52,524 B | 4 B pinned async issue는 2~4 µs이므로 **0.03~0.05 s**. nsys 평균 20.6 µs/call을 쓰면 0.27 s가 나오지만 그 평균은 대형 copy가 만든 것이므로 **쓰지 않는다** |
| E2 | H2D copy 최대 3,690회 / 0.27 GB. 현실적으로 nodes/off/cnt는 거의 항상 hit(1,230회), wvfr는 자주, dmod/bppm/stream은 boron·T/H 시행마다 miss → **1,200~2,500회 / 0.04~0.15 GB** | PCIe 12 GB/s로 **0.003~0.013 s** |
| **합** | **≈14,300~15,600 copy (전체 110,078의 13~14 %), ~0.1 GB** | **≈0.04~0.07 s (wall의 0.3~0.5 %)** |
| **sync** | **0** | — |

**이것은 바이트 소거가 아니라 호출-횟수 소거다.** 바이트는 전부 로드맵 1·2번에 있고,
sync는 전부 6번에 있다. 이 커밋의 실제 산출물은 §3의 인구조사와 §4.2의 잔차 진술,
그리고 그 잔차를 확정할 영수증이다.

---

## 6. 로드맵 — 기대 절감 순 (persistent-outer 계획의 입력)

| # | 항목 | 절감 | 난이도·전제 |
|---|---|---|---|
| **1** | **flatxs micx/lmpx D2H를 lazy/dirty-tracked로** (`:3467-3471`) | **D2H 24.4 GB(전체의 85 %), 8,200 copy.** 측정 D2H 28 GB/s로 **~0.87 s** | XSSet에 `_micx/_lmpx device-dirty` 플래그 + 첫 host read에서 materialize. 어려운 부분은 플래그가 아니라 **reader 205곳(`_micx` 127 + `_lmpx` 78)의 열거**다. 코드 내 `RASBERY_FLATXS_SKIP_MICX_DL` 실험이 같은 바이트를 **안전 없이** 지운다 — 그 실험이 아니라 dirty-tracking이 B0 판본이다 |
| **2** | **§4.2의 H2D 잔차 확정** (~7.0 GB / 25 K copy / 평균 280 KB) + §4.3의 sync 잔차(~6,100 sync ≈ **4.0 s**) | 가설대로 canonical 바인딩 미적중이면 **기능이 아니라 residency 버그** | 먼저 `[RASBERY][XFER]`로 잰다. 다음 238 실행 1회면 끝난다 |
| **3** | CRAM micro-XS H2D를 flatxs device 블록에서 **D2D**로 (`CudaCramBackend.cu:896/999`) | H2D **1.37 GB**, 544 copy → D2D(~free). **~0.11 s** | `_micx_generation == flatxs resident_micx_generation`이 그대로 gate. 두 backend가 스트림이 다르므로 event 필요 |
| **4** | Xe `drainXeCommit` 13 copy를 하나의 연속 블록으로 (`:2545`) | D2H copy 14,521 → 1,117 (**13,400 copy**), 바이트 불변 | device 블록에서 `xs[0..10]`·`xs_ssm`·`iden[I135..]`는 이미 연속이 아니다. **레이아웃 재배치**가 선행 |
| **5** | CMFD `host_status`(64 B)를 `sweep_out`(152 B)과 한 copy로 (`:4159`+`:4793`) | D2H copy ~11 K → ~4.4 K | 두 값이 캡처된 그래프의 memcpy node다. 토폴로지 변경 |
| **6** | **`cudaStreamSynchronize` 6.19 s** — persistent-outer | wall의 43 %. 위 1~5를 전부 합쳐도 이것보다 작다 | 별도 WP |
| 7 | flatxs ref 블록 (2.08 GB / 700 copy) | 0.17 s | statepoint마다 진짜로 바뀐다 — device 측 burnup 보간이 있어야 사라진다 |

---

## 7. 238 런북

로컬 계산 금지. 출력은 `E:`. IP는 어디에도 쓰지 않는다.

### 7.1 arm

```
# BASE (v4)
env -i RASBERY_PPR_MODE=master RASBERY_PC_MODE=decart RASBERY_GPU=1 \
  RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 \
  RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 \
  RASBERY_GPU_FLATXS=1 RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8 \
  RASBERY_GPU_WIEL_FOLD=chunked RASBERY_GPU_XE=1 \
  RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1 \
  RASBERY_OMP_THREADS=12 \
  RASBERY_GPU_CRAM=1 RASBERY_GPU_CMFD_FUSE=15 RASBERY_GPU_PPR=1 \
  RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_XE_TXN=1 \
  <bin> <kngr cycle-1 deck>

# ELIDE = BASE + RASBERY_GPU_XFER_ELIDE=1
```

교차 순서: warm-up 1 + hot 3, BASE/ELIDE 번갈아. median 채택.

### 7.2 게이트 (전부 통과해야 B0)

| 게이트 | 기준 |
|---|---|
| digest | `1f36e75dc00ed2b4` |
| outers | `4377` |
| h5diff | 0 |
| pin CSV | `cmp` 완전 동일 |
| 18지표 | Δ = 0 |
| `host_fallbacks` | 전 서브시스템 0 |
| `loop_arm` | `"device_graph"` |
| `txn_steps` | 1117 = `xe_device_steps` |

**하나라도 어긋나면 B0 주장은 철회다.** 전송 소거가 결과를 움직였다면 그 지점의
device 버퍼에 device writer가 있다는 뜻이고, §5의 "왜 안전한가" 열이 틀린 것이다.

### 7.3 새 계측 읽기

BASE 실행:

```
[RASBERY][XFER] {"elide_arm":0,"d2h_calls":...,"h2d_calls":...,"syncs":...,
                 "elided_calls":0,"elision_tests":0,"elision_hit_rate":-1,...}
```

- `h2d_calls`(BASE) 를 §4.2의 29,966과 비교한다. **차이가 잔차의 계측된 하한이다.**
- `d2h_calls`(BASE) / 33 이 §4.1의 `F ≈ 410`을 확인하거나 반증한다.
- `elide_arm:1 && elision_tests:0` 은 플래그가 어느 지점에도 닿지 않았다는 뜻 —
  이 값이 나오면 절감 수치는 무효다(G0).

ELIDE 실행: `elided_calls ≈ 13,131 + (1,200~2,500)`, `elided_bytes ≈ 0.04~0.2 GB`,
`syncs`는 BASE와 **같아야 한다**(이번 커밋은 sync를 제거하지 않는다).

### 7.4 nsys 전/후

```
nsys profile --stats=true -o E:/<run>/xfer_<arm> <위 커맨드>
```

`cuda_api_sum`에서 `cudaMemcpyAsync`의 **호출 수**를 본다(시간이 아니라). 기대:
110,078 → **≈95,000** (−13 %). `cudaStreamSynchronize`의 호출 수와 시간은
**변하지 않아야 한다** — 변했다면 그건 이 커밋이 하지 않은 일이므로 원인을 찾아야 한다.
프로파일링 오버헤드(+60 %)가 있으므로 절대 초는 인용하지 않고 **호출 수만** 인용한다.

### 7.5 계약 테스트 (로컬에서 실행 가능, 컴파일 불필요)

```
python tools/test_xfer_elide_contract.py
python tools/test_enum_alias_contract.py
python tools/test_dependent_template_contract.py
```

세 개 모두 이 커밋에서 통과했다. `test_xfer_elide_contract.py`는 6개 검사 각각을
**그 검사가 지키는 성질을 깨뜨린 소스 사본**에 다시 걸어 실패하는지 확인한다
(음성 대조 8개). 특히 `sweep_halt`를 그림자에 넣는 over-reach와,
regrow에서 mirror 무효화를 빠뜨리는 변이가 대조로 들어 있다.
