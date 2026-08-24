# GPU RASBERY 전체 방법론 및 벤치마크 보고서

작성일: 2026-08-24 · 기준 커밋: `527915e` (nodal batch arena) · 대상 트리: `GPU_Rasberry/rasbery_gpu`

---

## 1. 코드 개요 및 세부 스펙

### 1.1 물리 모델

| 항목 | 내용 |
|---|---|
| 중성자해석 | 2군 SENM(반해석 노달 전개법) + CMFD 가속 (Wielandt 이동 고유치) |
| 선형해법 | 블록(2×2) 전처리 BiCGSTAB, 컬러 블록 Gauss-Seidel 스윕 |
| 핀출력 재구성 | 2모드: SENM 반해석(소스반복, 코너균형 캡 100) / **MASTER MM §6.1 모드**(13항 르장드르 + CPB 코너 선형계, `RASBERY_PPR_MODE=master`) — MASTER 정합용 표준 |
| 감손 | CRAM(차수 8) 39핵종 사슬, 예측자-보정자 — 기본 midpoint / `RASBERY_PC_MODE=decart`(EOS율 보정자 + 밀도평균, MASTER 식 7.23 동일 스킴) — 표준 |
| Gd 처리 | 집중 의사핵종 640000(잔여포획 가중 5/4/3/2/1, MASTER BP01 동일 규약), 선택적 N_eff 축 σ보간(`RASBERY_GD_AXIS=neff`) |
| T/H | 폐채널 엔탈피 상승 모델, 연료온도 상관식(tfscale 보정 노브) |
| 임계 서치 | 붕소/제어봉 secant + bracket 하이브리드, 자연 EOC(`until boron ppm`), SearchExit 상태 기록 |
| XS 라이브러리 | CHIFFON HDF5(DeCART2D HGC 체인): 기준궤적 + bppm/tful/dmod 분기 델타(구간선형/절단 SVD), lmpx+Σ(iden·micx) 재구성, 반사체 붕소 의존성은 유효 B-10 %MICX 주입(`build_master_reflector_hgc.py --boron-micx-dmod`) |

### 1.2 문제 규모 (KNGR/APR1400 PSAR CY1 쿼터코어 기준)

- nxyz = 313 × 27 = **8,451 노드**, NG = 2 → 미지수 16,902 (CMFD)
- 표면 수 nsurf = 26,692, 집합체 88(쿼터, 회전 접기), 핀 16×16
- 전주기: 35 상태점(자연 EOC), 외부반복 약 2.1만(단일)/203만(M64 합계)

### 1.3 하드웨어/툴체인

| 서버 | GPU | 호스트 | 빌드 |
|---|---|---|---|
| 181 | RTX PRO 6000 Blackwell **Max-Q** (sm_120, 188 SM) | Windows 11, MSVC 14.44 | build181.bat: Ninja, CUDA 13, sm_120, /openmp:llvm |
| 238 | RTX PRO 6000 Blackwell **Server** ×2 (GPU0 사용, 97.9 GB) | RHEL, Xeon Gold 5317 24스레드, 251 GB | gcc13(conda) + CUDA 13.0, `-DRASBERY_CUDA_ARCHITECTURES=120`, no-pie 링크 |

GPU 스레드 슬롯: 188 SM × 1,536 = **288,768** (블록 256 기준 1,128 블록/웨이브), L2 **128 MB**.

### 1.4 재현성(비트-일치) 체계

- 공유 호스트/디바이스 노드 함수(`NodalKernel.h`, `FlatXsKernel.h`, `XsReconKernel.h`) — 한 소스가 CPU·GPU 양쪽으로 컴파일
- fma 축약 제어: 채굴된 마스크(예: 노달 phase-2 `0x2D555D57F55`, gcc asm 판독으로 확정)와 명시적 `fma()`/`__dmul_rn()` 형식 — TU별 `--fmad=false`
- 리덕션: 고정 청크 분할 + 순서 고정 2단 폴드(스레드 수 무관 bit-동일)
- 한계: MSVC(181) 호스트는 gcc-채굴 마스크와 축약 선택이 달라 노달 FULL이 0.07 pcm 잔차 → 181은 FULL 옵트인

---

## 2. 코드 구조

### 2.1 소스 배치

```
src/
  Driver.h            외부반복 지휘: 수렴판정, Xe 진동 감쇠, 서치 게이트, 자연EOC 재큐, PPR 호출
  Scheduler.h         스케줄/서치 상태기계 (SearchType/SearchExit, until_boron_ppm)
  XSSet.{h,cpp}       XS 재구성(lmpx+micx·iden), 분기델타, CRAM 감손, PC, T/H, 평형 Xe
  Nodal.{h,cpp}       SENM 노달 (updateConstant는 CPU 전용 — 유일한 초월함수 구간)
  NodalKernel.h       호스트/디바이스 공유 노달 본체 + fma 마스크 + 슬롯 뷰(nodalSlotView)
  CMFD.cpp, BICGCMFD.cpp  CMFD 계수 조립(setls/upddhat/updjnet — CPU 잔류), dhat 가드
  BICGSolver.cpp      솔버 진입: 아레나/단일 백엔드/CPU 선택, 카운터 JSON 방출
  CudaBICGBackend.cu  CMFD 디바이스 백엔드 + CudaBatchArena (아래 2.2)
  CudaXsReconBackend.cu  XSRECON/FLATXS/노달 디바이스 암 + CudaNodalArena
  PPR.cpp             핀출력 재구성 (SENM/master 2모드)
  main.cpp            CLI(--rasi/--raso/--batch-mode/--chiffoni 등), OMP 배치 루프
include/chiffon/      라이브러리 빌드 체인 (Importer/Exporter/Interpolator/ReflectorSolver)
tools/                master2rasi.py, compare_master_rasbery.py(MAS_SUM↔h5), GA 파이프라인
test/                 replay/mine 하니스 (gitignore — 디스크/서버에만 존재)
```

### 2.2 GPU 백엔드 구조

**CMFD — `CudaBICGBackend.cu`**
- 단일 인스턴스: BiCGSTAB 내부루프(1+nmax 반복) 전체를 **CUDA 그래프 1개로 캡처**, 스윕당 1 launch + 1 status D2H
- 커널 융합(커밋 `47704ac`): `begin_outer_fused`(역행렬+matvec+초기잔차), `prepare_p_jacobi`, `update_s_jacobi`, 쌍둥이 dot(`reduce_dot2`, 독립 누산기·동일 청크 순서), memset 흡수 → **반복당 커널 노드 27→22**
- halt 게이팅: 21커널 중 20개 첫 명령 가드(수렴 후 no-op; 43만 회 no-op에도 h5 byte-일치로 입증), `RASBERY_GPU_ITER_BATCH` 노브 + overrun 계측
- **CudaBatchArena**(64슬롯): 전 커널 `gridDim.y = slots`, `blockIdx.y = m` 슬롯 축 — 스레드 = (인스턴스×노드). 기하 테이블 공유, 상태는 슬롯 스트라이드. 기회주의 랑데부(도착자 집계) + 단일 런처 선출 + 슬롯별 halt/active 마스크로 수렴 시차(ragged) 흡수. 디바이스 Wielandt 스윕(`RASBERY_GPU_CMFD_SWEEP`, gcc-채굴 fma 형식): 소스 구축→내부루프→Wielandt 갱신→updls→음수 검사→종료 판정까지 1 그래프
- 슬롯당 메모리 3.52 MB → 64슬롯 225 MB (VRAM 0.2%), 활성 핫셋 ~60 MB ⊂ L2 128 MB

**XSRECON/FLATXS — `CudaXsReconBackend.cu`**
- XSRECON(평형 Xe 재구성): 스레드 = 연료노드, 분기 보간·동위원소 재조립 전체를 노드 단위 수행
- FLATXS: 스레드 = 노드, 마이닝된 정적 형식(StaticForms)으로 bit-일치

**노달 — `CudaXsReconBackend.cu` + `NodalKernel.h`** (커밋 `50fafbb`, `527915e`)
- 하이브리드(기본): 디바이스 Trl/Mat → D2H → CPU Even → H2D → 디바이스 Jnet (검증된 폴백)
- **FULL 상주**(`RASBERY_GPU_NODAL_FULL=1`): 5커널(Trl0→Trl12→Mat→Even→Jnet)을 1그래프로 디바이스 완주, D2H는 소비자 감사 기반 최소집합 **jnet+phis 854 KB/드라이브**만(하이브리드 대비 왕복 2.19 MB/드라이브 제거). Trl 스테이지는 (노드×그룹) 스레드 분할(그룹 분리 가능 확인), Mat/Even은 2×2 그룹 결합이라 노드 단위 유지, Jnet은 표면 단위
- **CudaNodalArena**(64슬롯): FULL 5커널을 `grid.y = slots`로 집계, CMFD 아레나 레시피 복제(공유 기하/슬롯 스트라이드/랑데부/고정 그래프/fail-open). 노브: `RASBERY_GPU_NODAL_BATCH=0`, `RASBERY_NODAL_BATCH_WAIT_US=auto|µs`

### 2.3 CPU-GPU 역할 분담 (설계 원칙: 분기는 CPU, 대량 단순병렬은 GPU)

| CPU (인스턴스별 1스레드) | GPU (공유 아레나) |
|---|---|
| 수렴판정, 붕소/봉 secant+bracket 서치, Xe 진동 감지·감쇠, 자연 EOC | CMFD BiCGSTAB 전체(융합 그래프), 컬러 GS 스윕, 디바이스 Wielandt |
| T/H 물성표 조회(분기 다수), 로드커스핑(부분삽입 분기) | 노달 5스테이지(FULL), 평형 Xe 재구성, FLATXS |
| CRAM 감손 지휘, PC 제어, HDF5 출력 | 슬롯 마스크에 의한 수렴 시차 흡수(디바이스 상주 판정값) |
| CMFD 계수 조립 setls/upddhat/updjnet (잔여 — 개선 후보 1순위) | |

### 2.4 실행 표준 규약(권장 env)

```
공통:  RASBERY_GPU=1 RASBERY_GPU_RB_SWEEPS=4 RASBERY_PPR_MODE=master RASBERY_PC_MODE=decart
효율:  RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1
238:   + RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1   (gcc: bit-일치; 181 MSVC는 0.07 pcm라 옵트인)
배치:  --batch-mode 64 (+CMFD_SWEEP=1; 노달 아레나는 FULL 조건에서 자동 관여)
```

---

## 3. 정확도 벤치마크 (MASTER 대비, KNGR/APR1400 PSAR CY1)

### 3.1 핀 오차 개선 체인

| 단계 | 조치 | 핀 1:1 rms (BOC) | max |
|---|---|---|---|
| 시작 | PPR 미수렴 상태 | 4.76 % | 32.4 % |
| 1차 | PPR 코너균형 캡 5→100 (`10c302a`) | 0.84 % | 5.2 % |
| 2차 | 반사체 XS 붕소 의존성 주입(무코드, 빌더 `--boron-micx-dmod`) | 0.52 % | 2.1 % |
| 3차 | MASTER MM §6.1 재구성 이식 (`f29549c`) | **0.24 %** | **0.78 %** |

집합체 rms 0.17 %. 코너/대각 핀 반대칭 오차(A0 −1.3/C1 +1.9 %) 및 Gd핀(0.92 %) 해소.

### 3.2 전주기 지표

| EFPD | 0 | 60 | 120 | 180 | 240 | 300 | 360 | 420 |
|---|---|---|---|---|---|---|---|---|
| 핀 rms % | 0.24 | 0.60 | 0.97 | 1.11 | 0.72 | 0.51 | 0.42 | 0.40 |
| 집합체 rms % | 0.18 | 0.53 | 0.83 | 0.87 | 0.35 | 0.24 | 0.22 | 0.22 |

- CBC 차이: BOC +2.2 ppm, 최대 −15 ppm(165 EFPD, Gd 창), EOC ~−9 ppm
- k-eff 차이: |max| 1.8 pcm (붕소 서치 잔차 수준, tol 2e-5)
- **Gd 창 험프 귀속**: 단일 B1 무한격자 격리(Gd 창 최대 −126 pcm, 소진 후 회복) + RASBERY 추적 N_Gd(t)가 라이브러리(DeCART) 궤적과 0.02 % 일치 + 진행률/κ/격자/N_BP 표 전부 배제 → **MASTER BP01 런타임 기구의 라이브러리 이탈**로 판정, 수용·문서화 결정(2026-08-23)

### 3.3 GPU-CPU 내부 정합 (KNGR 35상태, 스위치 효과 격리)

| 플랫폼 | 구성 | vs 순수 CPU |
|---|---|---|
| 238 (gcc) | base ≡ std ≡ full | 상호 **bit-동일 (500/500 데이터셋)** |
| 238 | GPU vs CPU | 2.0 pcm / 0.18 ppm (플랫폼 고유, GPU 암 무관) |
| 181 (MSVC) | GPU vs CPU | 0.04–0.07 pcm / 0.03 ppm |
| 스위치 단독 효과 | XSRECON/FLATXS/NODAL 각각 | ≤ 0.041 pcm / 0.004 ppm, 폴백 0 |

---

## 4. 속도 벤치마크

### 4.1 단일 실행 (KNGR CY1, 35상태 자연 EOC)

| 구성 | 181 (Max-Q) | 238 (Server GPU0) |
|---|---|---|
| CPU만 | 98.62 s | 175.83 s |
| GPU CMFD(+융합) | 82.11 s | 111.60 s |
| +XSRECON+FLATXS | 70.82 s | 98.52 s |
| **+노달 FULL (전체 스택)** | **69.08 s** | **88.88 s** |
| (리팩토링 전 최적) | 79.6 s | 114.0 s |
| MASTER (CPU 단일) | 27.2 s | — |

리팩토링 효과: 181 **−13 %**, 238 **−22 %**. 소형 단일 코어에서는 MASTER CPU가 여전히 3.1배 빠름(지연 지배 영역) — GPU의 이점은 동시 실행 처리량(4.2절).

### 4.2 64 동시 실행 처리량 (M64, CY02 64덱, 238 GPU0)

| 구성 | wall | 처리량 | CMFD 집계 폭 | 노달 아레나 폭 |
|---|---|---|---|---|
| 기존 캠페인 (sm_75 JIT, 하이브리드 노달) | 1565.9 s | 147.1 케이스/h | 16.0/64 | — |
| **현행 (native sm_120 + FULL + 아레나)** | **1361.9 s** | **169.2 케이스/h (+15 %)** | **22.7/64** | 6.3 (무대기) |
| 〃 + 2 ms linger | 1364.9 s | 168.8 | 21.2 | 15.2 |
| 〃 FULL 인스턴스별(아레나 off) | 1370.2 s | 168.1 | 24.0 | — |
| 참고: MASTER W16 (181 CPU 16병렬) | — | 216–218 케이스/h | — | — |

- 개선의 주경로: 노달 GPU 오프로드가 도착 스큐를 절반(7,955→4,388 µs)으로 줄여 **CMFD 집계 폭 자체가 상승** — "대기로 폭을 살 수 없고 호스트 경로 단축으로만 산다"는 캠페인 원칙 재확인
- 폴백/그래프 실패 0, 아레나 폭-4 검증에서 h5 500 데이터셋 bit-일치

### 4.3 세부 코드(스테이지)별 시간

**단일 실행, 181, gpu_all 구성 (wall 83.6 s):**

| 스테이지 | 초 | 비중 | 비고 |
|---|---|---|---|
| drive (CMFD BiCG) | 41.36 | 49.4 % | 302,763 디바이스 콜, **137 µs/콜 → 지연 지배**; 실비용 = 스윕당 ~110 커널노드 디스패치(449 µs) |
| nodal | 13.69 | 16.4 % | FULL 이식 전 수치 |
| setls | 7.48 | 8.9 % | CPU 잔류 (개선 1순위) |
| IO write | 4.99 | 6.0 % | |
| upddhat | 4.63 | 5.5 % | dhat 가드 분기 포함 |
| updjnet | 1.50 | 1.8 % | |
| flatxs/eqxe/기타 XS | ~4.4 | 5.3 % | eqxe 3,692콜 · 3,120만 노드 |
| updpsi/cusping | 0.70 | 0.8 % | |

**M64, 238 (스레드-초, 예산 100,218 = 64×1565.9; 기존 캠페인 실측):**

| 스테이지 | 스레드-초 | 비중 |
|---|---|---|
| drive (랑데부 대기 포함) | 47,085 | 47.0 % |
| **nodal (호스트, 하이브리드)** | **36,550** | **36.5 %** ← FULL/아레나로 이관된 대상 |
| HDF5 락 대기 | 2,677 | 2.7 % (734 ms/획득) |
| setls | 2,091 | 2.1 % |
| updjnet/upddhat | 1,868 | 1.9 % |
| XS 계열(flatxs/burnup/eqxe/boron/TH) | 1,612 | 1.6 % |

GPU0 가동률: 기존 14.3 %(평균)/23 %(피크) — 오프로드 후 폭 상승으로 개선(주경로는 스큐 축소).

### 4.4 커널 수준 사실

- CMFD 아레나 grid = (노드/256, 64슬롯) ≈ 218만 스레드(포화); 단일 인스턴스는 8.7천 스레드 = 슬롯의 3 %
- 과캡처는 순손실: `ITER_BATCH` K=8 → +8.4 s (43.3만 no-op 반복, byte-일치)
- 호스트 피닝(cudaHostRegister)은 중립 측정 후 미채택
- 융합 −3.5 s/단일런: 노드 19 % 감소 대비 작음 → 잔여는 컬러스윕 8노드(grid 배리어 필요)와 per-노드 고정비

---

## 5. 검증 체계

- 회귀: `.regress.sh` Tier 0(IISC 라이브러리 무결성)/1(IISC 배터리)/2(코어 keff/rod)/3(**CPU 골든 h5diff 전체 출력**, rev00에서 채택, /summary/reactivity 제외)
- bit-게이트: 융합·(lk,ig) 분할·아레나 — 각 단계 500/500 데이터셋 byte 비교로 통과; halt 게이팅은 강제 no-op 43만 회로 입증
- 채굴 하니스: `nodal_replay --sweep`/`nodal_mine_device`(디바이스 replay total_bad 1367→0), `cmfd_form_probe`
- MAS_SUM↔h5 비교기: `tools/compare_master_rasbery.py`(EFPD 조인, append 대응)

## 6. 잔여 개선 순위 (구조 분석 갭 리스트)

1. setls/upddhat/updjnet 디바이스화 → diag/cc 상주로 H2D 3.17 TB(M64) 제거
2. `cmfd_wiel_finalize` 단일스레드 8,451 직렬 폴드 병렬화(재기준선 필요 — 리덕션 재결합)
3. HDF5 락(734 ms/획득) — 전용 라이터 스레드 또는 1.14 thread-safe 빌드
4. CUDA 12.4+ 조건부 그래프 노드로 overrun 17.4 % 회수
5. 64스레드/24CPU 과가입 정책(`RASBERY_BATCH_HOST_THREADS` 재실험)
6. XSRECON/FLATXS 집계(이벤트성 도착이라 후순위), 융합 파이프라인 BiCGSTAB(Rupp, 정확도 케이브앳)
7. 배치 아레나 회귀 테스트 부재(추가 필요)

## 7. 관련 커밋 및 산출물

| 커밋 | 내용 |
|---|---|
| `10c302a` | PPR 코너균형 캡 5→100 |
| `f29549c` | PPR MASTER MM §6.1 모드 + 코너 DF 옵트인 |
| `3728c9a` | rev00 도구 채택(MAS_SUM 비교기, 골든 h5diff Tier3, 문서) + Gd 계측 |
| `048dc99` | summary gd_avg |
| `47704ac` | CMFD 커널 융합 + ITER_BATCH 계측 |
| `50fafbb` | 노달 FULL 상주 + (lk,ig) 분할 + phase-2 fma 마스크 채굴 |
| `527915e` | CudaNodalArena(64 인스턴스 집계) + g_flatxs_libs deque 수정 |

측정 산출물: 238 `~/m64_rebase_20260824/`(4개 런), `~/kngr_238/`, 181 `C:\Users\kmk\kngr_rasbery\`(fv_* 매트릭스), 로컬 `BENCH_APR1400/95_kngr_psar/report/`(fig15–18, fig_gpu_*), 발표자료 v7 덱(54장).
