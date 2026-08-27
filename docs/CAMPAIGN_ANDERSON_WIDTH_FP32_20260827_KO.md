# Exact 처리량 캠페인 — Anderson·FP32·폭 확장 결과 보고 (Phase 4~6)

**작성일**: 2026-08-27 | **브랜치**: `codex/exact-throughput-campaign` (tip `6a14d77`) | **서버**: 238 GPU0 | **기준선**: ref `aedae80` 계열, M64 BASE 214.95 cases/h

## 1. 요약 (TL;DR)

| 항목 | 결과 | 판정 |
|---|---|---|
| **Xe Anderson (단일덱)** | 93.6→**55.5 s = 1.69×**, outers −51 %, 수용률 95 % | **채택 권고** |
| **Anderson 정확도 (Gate B)** | MASTER 반응도 1.970→**1.905 pcm 개선**, AO 0.022→0.013 | 물리 개선 (QW2 패턴 재현) |
| Anderson 정확도 (Gate A) | Δkeff 2.27 pcm > 1 pcm 문턱 | 기준선의 미수렴 Xe 인공오차 방향 — baseline 재동결 대상 |
| Anderson M64 배치 | 199.1 c/h (BASE 215.0 대비 −7 %) | outers −38 % 실재하나 **HDF5 I/O 직렬화가 은폐** |
| **FP32 혼합정밀 (M64)** | +2.6 % | 채택 (대역폭 바운드 0.13 FLOP/B — ALU 여유 무의미) |
| **아레나 폭 96/128** | **무붕괴**(pin 수정 유효 입증), 그러나 BASE 77.4 c/h (−62 %) | **기각 — 폭 64 rolling 유지** |
| 배치 생산 구성 | **폭 64 rolling, 214.8 c/h** (1,280+ 케이스 무실패) | 유지 |

## 2. Phase 4 — safeguarded Anderson (RASBERY_XE_ANDERSON, commit `6a14d77`)

- Type-II AA(m=2), 대상 x=(I, Xe, Xe135m). Evaluate/Commit 분리(컴파일러 검증 const Evaluate), raw undamped 잔차만 history에 저장.
- 4중 안전조건(Gram 조건수, 예측 잔차 감소, finite/비음수, Picard 1.25× trust cap) + damper 연동 + 캐스케이드 재장전 시 history reset. 수용 후보는 반드시 flux 재수렴 후 확정.
- OFF 경로 byte-불변(골든 500/500), mutation 계약 테스트 47/47.

### 단일덱 A/B (kngr_238.json 35상태)

| | outers | xe_updates | wall (s) | AA proposed/accepted/rejected | resets |
|---|---:|---:|---:|---|---:|
| BASE | 21,271 | 3,512 | 93.62 | — | — |
| **AA1** | **10,483 (−51 %)** | **2,097 (−40 %)** | **55.47 (1.69×)** | 1,223 / 1,162 (95 %) / 61 | 139 |

### 정확도 게이트

- **Gate A** (vs 동결 기준선): Δkeff 최대 **2.27 pcm** (문턱 1 초과), ΔCBC 0.64 ppm (통과), 수렴 status 퇴행 0. EOC 핀 최대 상대차 ~1 % (연소 궤적 누적 드리프트, 버그 시그니처 아님).
- **Gate B** (vs MASTER, compare_master_rasbery.py — baseline 1.970 pcm/15.309 ppm 재현 확인): AA1 반응도 **1.905 pcm으로 개선**, AO 0.013.
- **해석**: QW2 감사와 동일 구조 — 동결 기준선 자체가 잘린(미수렴) Xe로 ~3.6 pcm 인공오차를 내장. Anderson은 Xe 고정점을 더 잘 수렴시켜 MASTER 쪽으로 이동. **Gate A 이탈은 기준선 결함의 폭로이며, 채택 시 baseline 재동결 필요.**

## 3. Anderson 배치 역설 — M64 −7 %의 원인 (SPTELEM 귀속)

M64(폭 64, 재시작 함대 64덱): AA1 1,157.4 s = 199.07 c/h (BASE 214.95 대비 −7 %), FAIL 0/64, PIN 수신증 무결.

SPTELEM=1 재실행 귀속(주의: 배치에서 telemetry 자체가 I/O 경합을 증폭 — 72.2 c/h로 왜곡, 패턴 판독용):

| M64 per-case 평균 | BASE | AA1 | Δ |
|---|---:|---:|---|
| outers | 31,728.8 | **19,573.8** | **−38.3 %** |
| xe_updates | 4,954.5 | 3,749.5 | −24.3 % |
| AA 수용률 | — | 95.6 % | solo 95.0 %와 동일 — 알고리즘 열화 없음 |

**wall 분해(SPTELEM run)**: io_wall **40.7 %**(단일덱 3 %에서 폭증), solve 50.3 %. 단일덱에서 3 %던 HDF5 쓰기가 64-way 동시성에서 지배 — 기왕에 문서화된 HDF5 락(734 ms/획득) 병목과 일치. **Anderson의 −38 % outer 감축은 실재하나, 배치 wall은 compute가 아니라 I/O 직렬화가 통제하므로 처리량에 반영되지 않음.** Anderson 도입/철회와 무관한 기존 병목.

## 4. Phase 6 — 아레나 폭 확장 실측 (96/128)

pin 수명 앨리어싱(폭>64 붕괴의 원인)이 HostPinLease+page-exclusive로 해소된 후 첫 직접 재시험:

| 구성 (96잡, 워커 64) | wall (s) | cases/h | mean_width | FAIL |
|---|---:|---:|---:|---:|
| 폭 64 rolling + BASE (참조) | — | **206.2** | ~22/64 | 0 |
| 폭 96 + BASE | 4,465.5 | **77.4** | 14.2/96 | 0 |
| 폭 96 + AA1 | 1,843.8 | **187.4** | 11.9/96 | 0 |

- **무붕괴 확인**: slots=96 정상 가동, PIN 수신증 무결(등록 7,680, 거부 0) — 계약 수정의 완전한 검증. VRAM 최대 관측 12.7/96 GB(제약 아님).
- **처리량 기각**: 커널 발사가 grid를 슬롯 수 비례로 잡는 구조에서 실점유(≤22)가 그대로인 채 폭만 키우면 빈 슬롯 비용이 전 커널에 부과. BASE −62 %.
- **역전 관찰**: 폭 96에서는 AA1이 BASE보다 **2.42×** 빠름 — outer 수가 적을수록 폭 비용을 덜 냄. 폭 64에서는 BASE 우위(215 vs 199). 폭 비용×반복 수의 곱 구조를 실증.
- W128은 무의미하여 취소. **생산 구성 = 폭 64 rolling(214.8 c/h) 유지.**

## 5. FP32 혼합정밀 (RASBERY_GPU_CMFD_FP32, commit `759d158`)

- FP32 inner(BiCG/color sweep/연산자 미러) + FP64 outer(잔차·보정·Wielandt·수렴판정), 실패 시 아레나 전체 FP64 latch.
- M64 **+2.6 %**. 커널이 대역폭 바운드(0.13 FLOP/B)라 정밀도 절반≠시간 절반. GPU FP64 스로틀(1/64)은 병목이 아니었음을 역확인.

## 6. 사용자 질의 응답 기록

1. **"여러 행렬을 하나의 거대 행렬로 합치면?"** — 블록대각 결합은 수렴 궤도를 오염(전역 내적·수렴판정이 최저속 시스템에 종속). 현 batched 커널이 이미 "공유 커널·독립 수렴"의 올바른 형태.
2. **"반대로 작은 행렬 다수로 쪼개서 FP32로 코어를 채우면?"** — 스레드=노드(시스템당 8,451스레드×64슬롯)로 이미 최소 단위. 추가 분할(Schwarz)은 반복 수 증가로 순손해. 실측: GPU는 포화가 아니라 **기아**(폭 12~22) — 제약은 행렬 크기/정밀도가 아니라 CPU→GPU 공급 속도.

## 7. 채택 권고·다음 단계 (우선순위 갱신)

| 순위 | 항목 | 근거 · 기대 |
|---|---|---|
| 1 | **HDF5 writer 분리** (전용 라이터 스레드/이중버퍼) | 배치 wall의 지배 항 실증(io_wall 40.7 %@SPTELEM, 락 734 ms/획득). 해소 시 base 상승 + Anderson −38 %가 처리량으로 발현 → M64 250~300 c/h 영역 기대 |
| 2 | **Anderson 채택** (단일덱 기본 ON 권고) | 1.69× + MASTER 정확도 개선. 채택 시 baseline 재동결(Gate A 2.27 pcm은 구기준선 결함). 배치는 1번 해소 후 재평가 |
| 3 | Phase 5 persistent kernel (Nsight 게이트 선행) | 단일덱 drive 55.5 s의 발사+동기 비중 실측 후 착수 판단 |
| 4 | 멀티GPU dispatcher | 수십배 목표의 구조적 경로(프로세스당 GPU 1 + 상위 큐) — 장비 확보 시 |

**수십배 목표 현황**: 단일 GPU 정직한 상한 재확인 — 현재 CPU(9950X W16) 대비 단일덱 1.69× 달성, 1~3번 완주 시 단일 GPU 3~5× 영역. 10~20×+는 멀티GPU 필요(4번).
