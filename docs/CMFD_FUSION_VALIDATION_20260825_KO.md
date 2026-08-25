# CMFD GPU Assembly + Scalar Fusion 검증 보고서 (238 + 181)

**작성일**: 2026-08-25 | **브랜치**: `codex/cmfd-gpu-assembly-drive-fusion-v2` (= CI 스텁 `8163460` + 저자 패치 `d8db387` + 리뷰 보완 `d84bae3`, push됨)
**참고**: push된 원 브랜치에는 소스가 없었고(CI 워크플로만), 실제 구현은 로컬 전달물 `0001-cmfd-gpu-assembly-drive-fusion.patch`에서 적용함.

## 1. 종합 판정

| 게이트 | 결과 |
|---|---|
| 코드 리뷰 (15항 정밀) | **NEEDS-CHANGES→보완 적용 완료** (블로커 없음) |
| 238(gcc) 빌드 + 계약 테스트 3종 | PASS |
| 238 단일덱 4-arm 골든 (A/B/C/D vs `ref_aedae80.h5`) | **전부 500/500 bit-exact (2000/2000)** |
| 181(MSVC) 빌드 | PASS |
| 181 arena 활성 arm 게이트 (B1/D1 vs A1, `--batch-mode 1`) | **96/96 bit-identical** — 패치는 추가 발산 0 |
| assembly 실가동 | 확인 — 단일 154콜, M64 202,260콜, **diag+cc H2D 219 GB 절감** |
| scalar fusion | 확인 — 그래프 노드 95→90, bit-무해 |
| M64 처리량 (D vs A) | **미완** — 238 타 사용자 점유로 취소, 181은 아래 별건 버그로 무효 |
| **[별건] 181/MSVC batch-mode 자체 결함** | **발견** — 패치 무관 (control arm 동일 발산·crash) |

## 2. 리뷰 핵심 (opus 15항 + gcc 13.3 실증 프로브)

- **산술 계약 실증**: setls의 `diagonal +=` 4개 사이트는 form-probe로 마이닝된 적 없는 **추정 축약**. 프로덕션 루프 형태를 재현한 프로브로 실측 — gcc 13.3은 **-O3 -march=native에서만** fma 융합(-O2 이하/-ffp-contract=off/-mno-fma 전부 불일치). 패치의 fma 선택은 Release 빌드에 한해 옳음 → 238 골든 PASS가 이를 확인.
- **인덱스/레이아웃 전수 감사**: xssm/xsrf/dtil/dhat/diag/cc/udiag/면적/체적 매핑 전부 정확. LEFT 역순(Z→Y→X)·RIGHT 정순, 대각 시드, udiag 4원소, Wielandt 시프트 철자(`__dmul_rn`×2+`fma`)가 기존 `cmfd_updls`와 문자 단위 일치.
- **reigvs 소스**: 조립 커널이 그래프 내부에 있으나 sweep 업로드가 스트림 순서상 선행해 **entry reigvs를 정확히 읽음**(wiel_finalize의 덮어쓰기가 발사 내에서 도달 불가).
- **state-0 잠복 버그 발견**: 음속 재시도로 state 0 반환 시 기존 코드는 **낡은 host diag를 재푸시**(잠복 버그), assembly는 현재 reigvs로 재조립(사실상 수리). 이번 KNGR 런에서는 미발현(골든 일치).
- **ASSEMBLY=0+FUSION=0 경로 byte-불변**: 훅 전수 검사로 확인.
- 적용한 보완(`d84bae3`): ①xsnf 미러 무효화 1건 ②축약 추정 문서화+`RASBERY_CMFD_ASSEMBLY_NO_CONTRACT` 이스케이프 해치 ③계약 테스트가 실제 빌드 플래그(flags.make)를 읽도록 강화 ④워크플로 -v2 트리거. 미적용 권고(저자 몫): 비활성 시 VRAM 사전할당 게이트(+40 % arena VRAM), state-2 throw→슬롯 플래그, accumulate 인자 전달.

## 3. 실계산 검증

### 238 (gcc, Release) — 4-arm 골든

| Arm | ASSEMBLY/FUSION | 그래프 노드 | vs ref | 비고 |
|---|---|---|---|---|
| A | 0/0 | 95 | 500/500 | 비활성 경로 불변 확인 |
| B | 1/0 | 95 | 500/500 | ¹ |
| C | 0/1 | **90** | 500/500 | fusion bit-무해 |
| D | 1/1 | **90** | 500/500 | ¹ |

¹ **주의**: 비배치 단일런에서는 arena가 생성되지 않아(`--batch-mode` 전용) assembly가 구조적으로 비활성(카운터 전 arm 0) — B/D 골든은 assembly를 검증하지 않음. wall(495~660 s)은 타 사용자 GPU 점유로 무효.

### 181 (MSVC) — arena 활성 검증 (`--batch-mode 1`)

- B1/D1 vs A1(control): **96/96 bit-identical** — MSVC 호스트에서도 assembly·fusion이 control 대비 추가 발산 0.
- assembly 카운터: 154콜, diag 41.6 MB + cc 124.9 MB 절감 (단일덱). 그래프 노드 A1/B1=95, D1=90.
- M64(`--batch-mode 64`): Arm D에서 assembly 202,260콜, **diag 54.7 GB + cc 164.1 GB H2D 절감** — 설계 의도대로 대규모 작동. graph/drive fallback 0.

### [별건·중대] 181/MSVC batch-mode 결함 (패치 무관)

`--batch-mode`가 **control arm 포함** 1스텝부터 per-instance 경로와 발산(kinf 14 %, flux 64 %)하고 비유한값으로 crash — 단일덱(A1 vs A0 게이트 FAIL)과 M64(55/64·51/64 실패) 모두. 238/gcc에서는 동일 경로가 수차례 bit-exact 검증됐으므로 **MSVC 빌드 한정의 arena/staging 결함**(역대 M64는 전부 238에서만 수행 — MSVC arena는 미검증 영역이었음). 이로 인해 181에서 M64 처리량·708/708 게이트 측정 불가. 후속 격리 필요.

## 4. MASTER 비교계산 (APR1400/KNGR PSAR)

238 4-arm 산출물이 골든과 bit-identical이므로 MASTER 대비 정확도는 기존 비교 결과와 **정확히 동일**: 반응도 max 2.0 pcm, CBC max 15.3 ppm(Gd 창·MASTER BP01 귀속), 핀 BOC RMS 0.24 %/max 0.79 %, EOC RMS 0.41 %. 상세는 `MASTER_vs_RASBERY_COMPARISON_20260824_KO.md`.

## 5. M64 처리량 게이트 결과 (2026-08-25 저녁, 238 유휴 상태에서 완료)

교차 5런 (D→A→D→A→D, 동일 조건 연속):

| 순서 | Arm | wall [s] | cases/h | 폭 | 도착 간격 [µs] |
|---|---|---:|---:|---:|---:|
| 1 | D r1 | 1245.9 | 184.9¹ | 19.4 | 5270 |
| 2 | A r1 | 1175.7 | 196.0 | 21.3 | 3678 |
| 3 | D r2 | 1076.2 | **214.1** | 21.97 | 3122 |
| 4 | A r2 | 1170.5 | 196.8 | 21.4 | 3734 |
| 5 | D r3 | 1078.2 | **213.7** | 21.94 | **3042** |

¹ 콜드스타트 이상치(새 빌드 첫 런: CUDA 컨텍스트·페이지캐시·클럭 램프).

**판정**:
- **Assembly+fusion 실효과 = +8.9 %** (정상상태 D 213.9 vs A 196.4; 양 arm 각 2회 재현, 산포 각각 0.2~0.4 %, 클러스터 무중첩). 도착 간격 3,678→3,042 µs 축소 — "CPU setls 제거→스큐 감소→폭 상승" 설계 메커니즘 그대로 실증.
- **MASTER W16 게이트(≥216–218): 미달** — 213.9는 하한 216에 0.97 %, 218에 1.9 % 부족.
- 처리량 이력: 169.2 → 196.2(v2) → 197.8(union) → **213.9(assembly+fusion)** — MASTER W16의 98~99 %.
- 부가 실증: 실제 CMFD.cpp 오브젝트 디스어셈블 **vfmadd=7** (setls 심볼 내) — fma 추정이 프로덕션 코드젠과 일치함을 최종 확인. 골든(d7dadb3 빌드) 500/500, M64 candidate_0031 708/708, 전 런 FAIL 0·폴백 0.
- 참고: 단일덱 비배치 골든런에서는 Wielandt warm-up 게이트로 assembly 미발화(카운터 0) — M64에서는 2,023,679콜/런, diag+cc **2.19 TB/런** 절감.

## 6. 잔여 작업

1. **잔여 격차 ~2 %** (213.9→218): 후속 후보 = HDF5 writer 분리(락 대기), upddhat/updjnet 이관, assembly 입력(dtil/dhat) 상주화.
2. 181/MSVC batch-mode 결함 격리 (전 슬롯 단일덱 재현→staging 이분).
3. 저자 몫 보완: 비활성 VRAM 게이트, state-2 throw 완화, state-0 잠복 버그 (assembly와 무관하게 존재).

## 부록 — 증거 경로

- 238: `~/gpu_dispatch_test`(cmfd-fusion-v2 @034ef9d), `~/kngr_238/cand_arm{A..D}.h5`+로그
- 181: `C:\Users\kmk\rasbery_cmfd_v2\`(빌드), `C:\Users\kmk\kngr_rasbery\cmfd_arm*.h5`, `C:\Users\kmk\m64_cmfd\`
- 로컬: scratchpad `dispatch_test\cand_arm*.h5, cmfd_arm*.h5, m64_{A,D}_candidate_0031.h5, run_{A,D}_181.log`
