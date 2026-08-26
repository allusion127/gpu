# Exact 처리량 캠페인 — Phase 0~2 결과 보고 (Rev.4 계획 이행)

**작성일**: 2026-08-26 | **브랜치**: `codex/exact-throughput-campaign` (tip `7ddb93c`+) | **서버**: 238 GPU0

## 1. 게이트·기준선 (§3.4)

- 기준 binary 4회 반복 → **500/500 bit-exact, 수치 노이즈 0** → Gate A/B threshold 동결(`test/reference/validation_thresholds_v1.json`) + provenance manifest(전체 SHA-256).
- 벤치마크·게이트 규약, exact-only 하드 계약(`[RASBERY][PHYSICS_MODE]` receipt) 가동.

## 2. Phase 1 — jobs>64 rolling queue 안정화 (§5~§7) ✅

### 1A 원인 판별 (기존 binary, 5 arm 전부 실패 0)

| Arm | jobs/워커 | pin | cases/h |
|---|---|---|---:|
| A | 64/64 | on | 214.2 |
| B | 96/64 | off(auto) | 130.8 |
| E0 | 96/64 +job namespace | off | 130.9 |
| F32 | 96/32 | off | 125.6 |
| F48 | 96/48 | off | 133.0 |

판정: rolling(폭 64 유지)은 **기존 코드로 이미 무결** — 과거 N=96/128 붕괴는 "폭>64+pin on" 조합. namespace 격리는 이 영역에서 무영향(B≈E0). pin off 비용 ~38 %가 1B의 회복 몫.

### 1B HostPinLease (§6 계약 구현: page-interval·owner refcount·overlap 정책·재할당 가드·3상태 env·7카운터 receipt)

| Arm | jobs | cases/h | lease 수신증 |
|---|---:|---:|---|
| C | 96 | 183.8 | reg==unreg, stale 0 |
| E1 | 96 | 179.1 | 5,164==5,164 |
| G | 128 | 193.3 | ✓ |
| S192 | 192 | 192.1 | 9,736==9,736 |
| **S256** | **256** | **194.0** | **12,884==12,884** |
| S512 | 512 | (진행 중) | — |

- **총 768+ 케이스 연속 실패 0**, 전 arm stale_evicted 0, bit-게이트 3종(단일 500/500 ×2 + wraparound 708/708) 전부 PASS.
- **스케일링 결론: rolling 처리량은 128잡부터 ~193±1 cases/h로 포화** (64잡 단일 배치 214가 잡당 최속; rolling은 tail ~10 % 비용으로 잡 수 무제한).
- 부수 관찰: PIN receipt의 overlap_rejections≈pageable_fallbacks(S192에서 9,076/9,736) — §6.2 보수 정책의 의도된 동작이나 회수 가능한 성능 여지(후속 검토 항목).
- restart namespace(출력 경로 기준)·`--rasi` 중복 허용/`--raso` 정규화 검증·exact-only 계약 동시 가동.

## 3. Phase 2 — outer 반복 해부 telemetry (§8) ✅

- 구현: `[RASBERY][SPTELEM]` 상태점당 1줄(segment-boundary 귀속, 합 불변 필드 동봉), mutation 검증 계약 테스트(7종 파괴 전부 검출).
- 게이트: 오버헤드 **−0.12 %**(≤1 % 통과, ON=OFF 노이즈 수준), ON/OFF 모두 골든 **500/500 bit-exact**.

### 핵심 발견 — outer의 82.6 %가 Xe 재수렴

단일덱 35상태 합계 21,271 outers:

| 원인 | outers | 비중 |
|---|---:|---:|
| **Xe 갱신 재수렴** (xe_updates 3,512 = **~100회/상태점**) | **17,564** | **82.6 %** |
| 보론 탐색 (143 trial) | 1,707 | 8.0 % |
| settle 게이트 | 1,048 | 4.9 % |
| 초기 수렴 | 731 | 3.4 % |
| TH (142회) | 221 | 1.0 % |

Amdahl 실측 계수: total 94.6 s = T_fixed 0.48(라이브러리 0.13) + solve 91.3 + IO 2.8. solve 내부: drive 55.5(60.8 %) / setls 12.1 / nodal 6.1 / upddhat 5.8 / updjnet 2.5.

### Phase 3~5 go/no-go 함의

1. **Phase 4 (Xe Anderson) = 최대 레버 확정.** Xe 고정점 반복(100회/상태점)이 지배 — Anderson으로 3× 감축 시 outer −55 %, 이론상 단일덱 ~2× / M64 ~380 cases/h 영역. §10 계약(raw map·transactional rollback·7 안전조건)대로 구현 착수 권고.
2. **Phase 3 감사 필수 선행 항목**: 왜 상태점당 Xe 100회인가(반복 캡? xe_relax 과소? 수렴판정 과엄?) — Anderson 전에 현행 Picard의 낭비 요인 확인.
3. **Phase 5 (persistent kernel) 착수기준 계산**: drive가 solve의 61 %이나 이는 GPU 계산+대기 포함 — dispatch+sync 분리 비중은 Nsight로 확인 필요. Xe 감축이 선행되면 절대 이득도 함께 줄어듦 → 순서는 4→5.
4. settle 게이트 4.9 %·탐색 8 %는 2차 레버.

## 3.5 갱신 (8/26 밤 ~ 8/27): S512·pin 회귀 해소·QW 판정

### 스케일링 최종 곡선 (전부 실패 0)

| 잡 수 | pin 수정 전 | pin 수정(af3b827) 후 |
|---:|---:|---:|
| 64 | 214.2 | **214.8** |
| 96 | 183.8 | **206.2 (+12 %)** |
| 512 | 197.8 (수정 전 빌드) | — |

- **S512: 512/512 무실패, 197.8 cases/h** — 512잡에서 플래토(192~194)를 상회. 캠페인 누적 **1,280+ 케이스 연속 무실패**.
- **lease 회귀(−16 %) 근본 해소**: 원인 = 인접 할당의 4 KiB 페이지 공유 → 페이지 단위 cudaHostRegister가 요청 36 % 거부(수신증 9,736등록/9,076거부가 인벤토리 124요청/80베이스와 자릿수 일치). 수정 = 핀 대상 전 버퍼 page-exclusive 할당(`af3b827`). 검증: 앵커 1,072.9 s=214.8, PIN 수신증 거부 0·등록 5,120(=80×64 예측 정확).

### QW 판정 (Gate A/B 실측)

| Arm | 효과 | Gate A (기준선 비퇴행) | Gate B (MASTER) | 판정 |
|---|---|---|---|---|
| QW1 (corrector Xe 재평형) | outers ±0.3 % — **무효과** | FAIL (Δkeff 2.0 pcm) | — | **불채택** (기본 OFF 유지) |
| QW2 (캐스케이드별 예산) | xe_updates 3,512→7,200: **기준선이 다수 상태점에서 잘린 Xe 출력 중이었음을 폭로**, wall +12 % | FAIL (Δkeff 3.6 pcm) | **전 지표 개선** (반응도 1.97→1.91 pcm, AO 0.022→0.013, Fq 0.071→0.069) | **정확성 수정으로 채택 권고** — 기준선의 ~3.6 pcm은 미수렴 Xe 인공오차. 채택 시 baseline 재동결 필요, wall 비용은 Phase 4 Anderson이 회수 대상 |

캐스케이드 실측(QW1 arm): **239 캐스케이드 × ~14.8 스텝** — Phase 4 Anderson 설계 입력을 감사 모델(354×9.9)에서 갱신.

## 4. 미결·후속

- S512(512잡 최대압박) 진행 중(ETA 금일 밤) — 완료 시 본 문서 갱신.
- M64+telemetry 1,209.5 s(190.5 c/h)의 앵커(1,076 s) 대비 12 % 차: 빌드/telemetry/콜드스타트 교란 — OFF-앵커 재실행으로 귀속 예정.
- master_matrix3.sh 게이트 셸 버그(grep -c `|| echo 999` 이중 출력) 발견·수정 — S512 오탈락의 원인이었음.
- 데이터 파일: 로컬 scratchpad `dispatch_test\sptelem_single.jsonl`(36줄)·`sptelem_m64.jsonl`(3,328줄) — Phase 3/4 설계 입력.

## 5. 다음 단계 (계획 §15.1)

Phase 3 ①(statepoint 초기화·Xe 반복 정책 감사, opus) → Phase 4(safeguarded Anderson, §10 전 계약) → 검증(arm A/B/C/D, Gate A+B) → Phase 5 조건부.
