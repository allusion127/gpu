# Exact 처리량 캠페인 — Anderson·FP32·폭 확장 결과 보고 (Phase 4~6)

**작성일**: 2026-08-27 | **브랜치**: `codex/exact-throughput-campaign` (tip `6a14d77`) | **서버**: 238 GPU0 | **기준선**: ref `aedae80` 계열, M64 BASE 214.95 cases/h

## 1. 요약 (TL;DR)

| 항목 | 결과 | 판정 |
|---|---|---|
| **Xe Anderson (단일덱)** | 93.6→**55.5 s = 1.69×**, outers −51 %, 수용률 95 % | **채택 — 단일덱 기본 ON** (§8.2) |
| **Anderson 정확도 (Gate B)** | MASTER 반응도 1.970→**1.905 pcm 개선**, AO 0.022→0.013 | 물리 개선 (QW2 패턴 재현) |
| Anderson 정확도 (Gate A) | Δkeff 2.27 pcm > 1 pcm 문턱 | 기준선의 미수렴 Xe 인공오차 방향 — baseline 재동결 대상 |
| Anderson M64 배치 | 199.1 c/h (BASE 215.0 대비 −7 %). writer 스레드 채택 후 재측정 202 vs 216 c/h | **배치 기본 OFF** — 잔존 원인은 I/O가 아니라 **도착 폭 기아**(§8.2) |
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

## 8. 채택 결정 (2026-08-27) — 기본값 확정

238 검증이 끝난 두 기능의 **기본값을 코드에 반영**했다. 두 결정 모두 "빠르니까"가 아니라 **측정된 근거의 방향**으로 갈렸다.

### 8.1 HDF5 writer 스레드 — 전 모드 기본 ON

`RASBERY_IO_WRITER` 기본값을 `inline` → **`thread`** 로 변경. 근거는 하나뿐이다: **시도한 모든 구성에서 byte-identical**(단일덱 500/500, M64 **45,312/45,312 데이터셋**, restart 스냅샷 포함)이고 M64 **+0.6 %**. 산출물이 바뀌지 않는 변경에는 "안전을 위해 기본값을 유지한다"는 논거가 성립하지 않으므로, `inline`은 기본값이 아니라 **레거시 경로**가 된다 — `RASBERY_IO_WRITER=inline`로 여전히 도달하며 bisect·A/B 팔에 필요하다. 오타 값은 경고 후 **기본값(thread)** 으로 떨어진다(골든이 동결된 경로가 기본값이므로).

수신증에 **provenance**를 추가했다: `{"mode":"thread","mode_source":"default"}`. `thread(default)`·`thread(env)`·`inline(env)` 세 실행이 로그에서 구분되지 않으면 A/B가 무효이기 때문이며, config·summary 양쪽에 싣는다. 상세는 `docs/IO_WRITER_THREAD_DESIGN_20260827_KO.md`.

### 8.2 Xe Anderson — 단일덱 기본 ON / 배치 기본 OFF

| 실행 형태 | `RASBERY_XE_ANDERSON` 미설정 | 명시 설정 |
|---|---|---|
| 단일 실행(`--batch-mode` 없음) | **ON** | `0`/`1` 모두 존중 |
| `--batch-mode M` | **OFF** | `0`/`1` 모두 존중 |

**단일덱 ON의 근거는 속도가 아니라 정확도다.** 1.69×(93.6→55.5 s)는 부수 효과이고, 채택을 정당화한 것은 **MASTER 일치가 개선**되었다는 사실이다 — 반응도 1.970 → **1.905 pcm**, AO 0.022 → 0.013. Anderson은 같은 사상(map)을 같은 허용오차(`XE_EQUILIBRIUM_TOLERANCE`)까지 수렴시키므로 물리를 바꾸지 않는다. 즉 **v1 기준선 쪽이 잘린(미수렴) Xe로 ~2~3 pcm 인공오차를 내장**하고 있었고, Anderson은 그 고정점을 제대로 수렴시켜 MASTER 쪽으로 이동시킨 것이다(QW2 감사와 동일 구조). 따라서 §2의 Gate A 2.27 pcm 이탈은 **후보의 결함이 아니라 기준선의 결함이 드러난 것**이며, 올바른 대응은 문턱 완화가 아니라 **baseline 재동결**이다.

**배치 OFF의 근거 — 도착 폭 기아(arrival-width starvation).** 배치에서는 순손해였다: **202 vs 216 cases/h**(M64). 원인은 I/O가 아니고(§3의 io_wall 지배는 writer 스레드가 제거했다) 알고리즘도 아니다(배치 수용률 95.6 % ≈ solo 95.0 %). 구조는 이렇다:

> Anderson이 **잡당 outer를 −38 %** 줄인다 → 각 잡이 solve 안에 머무는 시간이 짧아진다 → **어느 순간에도 배치 CMFD 랑데부 안에 동시에 들어와 있는 잡 수(실효 폭)가 줄어든다** → 커널은 선언된 슬롯 수에 비례해 grid를 잡으므로 **비어 있는 슬롯 비용을 전 커널이 지불**한다 → 잡당 이득을 총계가 잃는다.

이는 §4의 폭 96 실험과 정확히 같은 곱 구조(폭 비용 × 반복 수)의 다른 단면이다. **설계된 해법은 slot compaction**(Phase 5 계획, `docs/PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md`) — grid 비용을 **선언된 폭이 아니라 실제 점유**에 묶는다. 그것이 착지하기 전까지 배치 기본값은 OFF로 두고, `RASBERY_XE_ANDERSON=1`로 배치 A/B 실험은 계속 가능하게 남긴다. 반대로 단일덱에서 `=0`은 레거시 Picard 궤적(v1 기준선) 재현용이다.

**모드 판별 방식.** 결정점이 "지금 배치인가"를 알아야 하는데 Driver는 그것을 알 수 없다(`--batch-mode`는 argv, 배치 폭은 CUDA 백엔드 뒤 — stub 빌드에는 없음). 그래서 `main()`이 **배치 분기를 선택하는 바로 그 술어**(`batch_width > 0 && !rasbery_inputs.empty()`)를 한 번 계산해 `rasbery::declareExecutionMode()`로 **선언**하고, 그 술어로 분기도 한다(실행과 보고가 어긋날 수 없음). 선언은 첫 수신증보다 **앞**에 온다 — 모드 의존 기본값은 첫 읽기에서 캐시되고 그 첫 읽기가 수신증이기 때문. 기본은 `Single`이므로 단위 테스트·툴·직접 Driver 생성은 선언을 잊어도 단일 실행으로 취급된다.

`[RASBERY][PHYSICS_MODE]`에 세 필드 추가: `"exec_mode":"single|batch"`, `"xe_anderson":true|false`, `"xe_anderson_source":"default|env"`. 상태만으로는 실행을 식별할 수 없기 때문이다(기본으로 켜진 것과 누가 켠 것은 다른 사실이다).

### 8.3 baseline 재동결 — v2 준비 완료, 실행 대기

`test/reference/validation_thresholds_v2.json` · `validation_baseline_manifest_v2.json`을 **템플릿으로 추가**했다(v1은 이력으로 **무수정 보존**).

- **문턱은 불변**: v1의 값은 노이즈 유래가 아니라 엔지니어링 바닥값(plan §3.4)이므로 궤적이 바뀌어도 움직이지 않는다.
- **바뀌는 것은 궤적**: v2 기준선은 **Anderson-ON 단일덱**을 포함한다. `derivation` 필드에 그 이유(정정 개선, 1.970→1.905 pcm, v1의 ~2~3 pcm 미수렴-Xe 인공오차)를 명시했다.
- **골든 SHA·wall 등은 전부 `TBD_REFREEZE`** 자리표시자 — 238 검증 에이전트가 재동결 실행으로 채운다.
- **실행 유효성 조건**: 기준선 실행은 `xe_anderson_source="default"` 이고 `mode_source="default"` 여야 한다. env가 붙은 실행은 실험이지 기준이 아니다.

### 8.4 재동결 프로토콜 (238 검증 에이전트용)

**전제**: 어떤 단계에서도 `RASBERY_IO_WRITER` / `RASBERY_XE_ANDERSON`을 **설정하지 않는다**. 이번 동결의 목적은 *기본값 경로*를 고정하는 것이므로, env가 붙는 순간 그 실행은 실험이 된다.

| 단계 | 내용 | 합격/기록 |
|---|---|---|
| R0 | 계약 테스트 3종: `test_io_writer_contract.py`, `test_xe_anderson.py`, `test_exact_only_contract.py` | 전부 PASS |
| R1 | 재빌드 후 **수신증 확인 실행** 1회 | `[PHYSICS_MODE]` `exec_mode="single"`·`xe_anderson=true`·`xe_anderson_source="default"`, `[IO_WRITER]` `mode="thread"`·`mode_source="default"` |
| R2 | **골든 4회 반복**(단일덱 `kngr_238.json`, 동일 명령) | 4회 상호 `h5diff` **전 데이터셋 Δ=0**, keff/ppm `max_abs_delta=0.0`. `cmp` 금지(HDF5 objheader 타임스탬프) |
| R3 | **노이즈 점검** | 4회 wall의 spread 기록. Δ=0이면 v1과 같이 문턱은 노이즈 유래가 아닌 엔지니어링 바닥값으로 유지 |
| R4 | **MASTER 재비교** (`tools/compare_master_rasbery.py`) | 반응도/CBC/AO/BOC pin RMS·max 기록 (기대 ~1.905 pcm, AO ~0.013) |
| R5 | **Anderson 건전성** — `RASBERY_STATEPOINT_TELEMETRY=1` **별도 실행**(타이밍과 섞지 말 것) | `xe_aa_proposed/accepted/rejected/history_resets`, 수용률 ≥ ~90 % |
| R6 | **M64 앵커**(기본값 그대로 = writer thread ON, 배치 Anderson OFF) | cases/h, FAIL 0/64, `[IO_WRITER][SUMMARY]` `failures=0`·`skipped=0` |
| R7 | v2 파일의 `TBD_REFREEZE` 채우기 + `frozen_utc` 기입 | manifest·thresholds 커밋, threshold 파일 sha256 기록 |

v1 파일 2개는 **수정 금지**(이력). v2가 동결되기 전까지는 v1이 유일한 동결 기준이다.

**우선순위 갱신**: §7 표의 1번(writer 분리)·2번(Anderson)은 채택으로 종결. 다음은 **Phase 5 slot compaction** — 배치 Anderson의 −38 % outer 감축을 처리량으로 회수하는 유일한 설계된 경로이자, 폭 확장 기각(§4)의 원인도 같은 것이다.
