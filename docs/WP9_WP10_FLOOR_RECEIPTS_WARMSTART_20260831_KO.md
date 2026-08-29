# WP9-A / WP9-D / WP10.1 / WP10.2 — 상태점 바닥의 이름, 케이스의 이름, 그리고 자식을 부모에서 시작시키기 (2026-08-31)

문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 계획 | `GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` WP9 단계 A·C·D, WP10.1, WP10.2 |
| 참조 실측 | `GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md` §2.2(비용 모델), §3.2(위상 귀속), §5.1(LP = `core` 블록), §5.4(warm-start), §6.5(게이트 등급) |
| 커밋 | `5883023`(WP9-A) · `95481bc`(WP10.1) · `031ccbe`(WP10.2) · `3df4ea7`(WP9-D) |
| 로컬에서 한 것 | 소스 계약, 파이썬 계약, 그리고 **헤더 전용 TU의 MSVC 컴파일·실행**(`tools/test_case_key_contract.py`의 compiled half). 전체 빌드는 CUDA/HDF5 때문에 238에서만 된다 — **성능 수치는 하나도 없고, 있어서도 안 된다** |
| 238에서만 가능한 것 | §6 전부 |

---

## 0. 한 줄 결론

**상태점 바닥은 이제 잔여값이 아니라 이름이 붙은 항목들이고(`loop_wall`·`floor_wall`·
`nested_wall`·`floor_transfer`), 케이스에는 대칭 접힘까지 포함한 정식 이름이 생겼으며
(`case_key`), 자식은 부모의 BOC 상태에서 출발할 수 있다(N1). WP9-D는 탐색을 *재기만*
했고 아무것도 바꾸지 않았다 — 무엇을 고쳐야 하는지가 아직 측정되지 않았기 때문이다.**

---

## 1. WP9-A — 바닥의 재분해

### 1.1 무엇이 문제였는가

GA evaluator 계획 §2.2는 상태점을 `c + d × outer`로 회귀하고 `c = 0.474 s`,
`d = 4.805 ms/outer`를 얻는다. §3.2는 outer 본문 7위상의 합이 `4.428 ms/outer`이고
차이 `0.377 ms/outer`가 "Scope 밖에 있는 것들(Xe 갱신, 수렴 판정, 탐색 산법)"이라고
적는다 — **코드를 읽어서 붙인 이름이고, 그 뒤에 계기가 없었다.** `c` 쪽도 마찬가지로
6개 버킷과 이름 없는 나머지였다.

### 1.2 네 개의 객체, 그리고 왜 하나가 아닌가

| 객체 | 내용 | 합산 대상 |
|---|---|---|
| `phase_wall` | outer 본문 7위상 (`updpsi`…`upddhat`) | **변경 없음.** `tools/scheduler_trace_replay.py`가 `wall - sum(phase_wall)`로 경계 작업을 유도하고 그 값에 캘리브레이션되어 있다 |
| `loop_wall` | **신규.** SolveLoop 안, outer 본문 밖: `th_update`, `xe_step`, `search_propose`, `search_apply` | `d`의 나머지 — 위의 `0.377 ms/outer` |
| `floor_wall` | 상태점 경계: PPR 3종, depletion 2종, `result_add`, **신규 `result_write`** | `c` |
| `nested_wall` | **신규, 비가산.** `UpdateFlatXS` 전체 호출 + 호출 수 | 위 두 객체 **안에서** 돈다. 더하면 이중 계상 |
| `floor_transfer` | **신규.** 마지막 SolveLoop 이후의 H2D/D2H 바이트·호출 | `*_delta - floor_*` = 반복 중 절반 |

`phase_wall`을 건드리지 않은 것이 이 커밋의 유일한 하위호환 조건이다. 새 버킷을 거기에
접었다면 이미 모델이 의존하는 숫자를 조용히 옮겼을 것이다.

### 1.3 계기의 비용과 중립성

- 모든 신규 버킷은 기존 `outer_timing::Scope`다. 그 Scope는 이미 "시계를 읽고 두 누산기에
  더할 뿐"으로 계약에 고정되어 있다(`tools/test_telemetry_neutrality.py` §3).
- `nested_wall`의 `UpdateFlatXS`는 `XSTiming.h`에 **자기 게이트가 없다**.
  `RASBERY_STATEPOINT_TELEMETRY`는 트리에 **읽는 곳이 하나뿐**이고(계약이 강제),
  `Drive()`가 그 하나에서 미러를 *arm* 한다. 꺼져 있으면 Scope 하나가 relaxed atomic
  load 하나를 더 낼 뿐 시계를 읽지 않는다.
- `floor_transfer`의 기준선은 **마지막 SolveLoop 직후, PPR 직전**에 한 번만 arm 된다
  (`armFloor`, 재호출 무효). 다른 자리에서 arm 하면 다른 것을 재게 된다.

### 1.4 계약 (`tools/test_statepoint_telemetry.py` §6b)

음성 대조군 4개를 **먼저 실패시키고** 통과시켰다:

| 대조군 | 잡히는 결함 |
|---|---|
| 버킷을 두 가산 객체에 동시에 넣기 | 이중 계상 |
| `phase_wall`을 7개 이상으로 키우기 | replay 모델의 캘리브레이션 이동 |
| 명명된 호출부의 Scope 제거 | 아무것도 안 재는 버킷 |
| `std::format` placeholder/인자 수 불일치 | **전체 빌드가 로컬에서 안 되므로 리뷰로도 컴파일로도 못 잡는 결함** |

마지막 것은 실제로 값어치를 했다: 두 수신증 모두 placeholder 63/63, 66/66(WP9-D 이후
77/77, 74/74)로 재계수된다.

---

## 2. WP9-D — 탐색의 가격표 (계측만)

### 2.1 왜 아직 고치지 않는가

계획 §2.2: search = outer의 15.3 %, trial 137회, 그리고 **후보 의존**이다. 그러나
"137 trial"은 하나의 숫자이면서 네 개의 서로 다른 문제다.

| 분류 | 뜻 | 그 경우의 레버 |
|---|---|---|
| `probe` | 기울기가 **없어서** 부트스트랩했다 | 이전 상태점/동일 family의 slope 전달, bracket 시딩 |
| `carry` | 상태점 경계를 넘어 전달된 기울기로 시작했다 | 이미 동작 중 — 몇 번인지가 그 증거 |
| `secant` | 표본 두 개, 한 걸음 | 알고리즘이 아니라 **허용오차**가 비용이다 |
| `bisect` | secant가 bracket을 벗어나 중점을 잡았다 | bracket이 좁혀지지 않는다 — 노이즈 바닥 문제 |

세 레버는 서로 다르고, 지금까지 로그는 **어느 실행을 보고 있는지 말할 수 없었다.**

### 2.2 수신증

상태점마다(그리고 실행 합계로):
`trials`, `proposals`, `refused`, `secant`, `carry_secant`, `probe`, `bisect`,
`iterations`. 상태점에만(합산되지 않으므로): `tol`(계획이 인용하는 2e-5), `dk`,
`exit`, `x_first`(상태점이 **출발한** 점 — WP10.2가 정확히 이 값을 움직인다),
`x_final`, `dx_last`(마지막으로 커밋된 걸음의 크기 = 최종 |Δppm|).

`tools/case_cost_profile.py`가 `outers/trial`까지 인쇄한다. **trial 하나는 outer
하나가 아니라 Xe 캐스케이드 한 벌의 재수렴**이고(계획 §2.2: trial 1회 ≈ 38 outer),
그 비율이 모든 trial 감축안의 가격 기준이다.

### 2.3 궤적 중립성 — 기계적으로

계약이 `Scheduler.h`에서 WP9-D 카운터를 언급하는 **모든 줄**을 스캔해 선언·순수 증가·
순수 대입 중 하나임을 요구하고, **조건문 안에 등장하는 것을 명시적으로 금지**한다.
계기가 자기가 재는 탐색을 조종하기 시작하는 유일한 경로가 그것이기 때문이다. 가드
분해는 정규식이 아니라 괄호 계수로 한다 — 조건 안에 괄호가 있을 수 있고
(`method.rfind("bisection", 0) == 0`), 첫 `)`에서 멈추는 스캔은 조용히 통과하기 시작한다.

### 2.4 trial 감축 후보 (제안, **코드 아님**)

238에서 §2.2의 분류 분포를 본 **뒤에** 하나만 고른다. 각 후보의 게이트 등급은 미리
정해 둔다 — 나중에 정하면 결과를 보고 정하게 된다.

| # | 후보 | 무엇을 줄이는가 | 게이트 | 위험 / 즉시 복귀 조건 |
|---|---|---|---|---|
| D1 | **상태점 간 boron slope 전달 강화**: 현재 `SearchMemory.boron_secant_dkdx`는 이미 전달되지만 `carry_available`은 `has_boron_secant`만 본다. 연소도 변화량에 따라 slope를 1차 보정해서 전달 | `probe` → `carry`. 상태점당 trial 1~2회 | **N1** — 첫 제안점이 바뀌므로 궤적이 바뀐다. Gate A + digest 반복 + 핀 RMS/max | 보정이 틀리면 첫 trial이 더 멀어진다. `probe+carry` 대비 `trials` 증가 시 즉시 폐기 |
| D2 | **동일 family accepted case에서 초기 bracket 제안** (계획 §WP9-D 본문) | `probe`와 초기 `secant` | **N1** (경계 clamp 포함). 제안 bracket이 물리 범위를 벗어나면 cold bracket 복귀 | 다근 구간에서 bracket이 잘못된 근을 감싸면 수렴은 하되 다른 답이다. **`search_exit_status`와 `dk`를 acceptance에 넣어야 한다** |
| D3 | **탐색 표본용 flux 허용오차 완화**(A2 staged tolerance의 탐색 전용 확장): trial 점의 flux를 `search_tol / STAGED_SEARCH_MARGIN`까지만 수렴 | trial당 **outer**(38 → ?). trial 수는 그대로 | **A2** — 별도 수렴 정책. 이미 `RASBERY_STAGED_FLUX_TOL`이 이 형태이고 `staged_relapses`가 그 감시 계기다 | `staged_relapses`가 `search_trials`와 같은 자릿수면 느슨한 단계가 다른 근을 찾고 있다는 신호(계획 §2.2) |
| D4 | **settle 게이트 축소**: `SEARCH_SETTLE_ITERS`가 만든 290 outer(6.3 %) | `settle_outers` | **N1** | 게이트는 탐색이 신뢰하는 표본을 만든다. 줄이면 `bisect`와 `refused`가 오른다 — 그 둘이 오르면 되돌린다 |
| D5 | **`max_search_iter` 하향 + best-fallback 조기화** | 병리 케이스의 꼬리 | **N1**, 단 **acceptance는 그대로**: `search_exit_status != CONVERGED`인 상태점은 elite가 될 수 없다 | GA 적합도에 수렴 실패가 들어가면 나쁜 LP가 싸 보인다. **fitness에 `converged`를 반드시 반영** |

**D3만이 outer를 줄이고 나머지는 trial을 줄인다.** §2.2의 분포가 `secant` 위주로 나오면
D3가 유일한 후보이고, `probe` 위주로 나오면 D1/D2가 먼저다.

---

## 3. WP10.1 — 케이스의 정식 이름

### 3.1 왜 evaluator만 만들 수 있는가

두 LP가 같은 노심인지는 `geometry.symmetry.angle` / `mirror`의 함수다. GA 컨트롤러가
그것을 해석하기 시작하면 **하나의 질문에 두 개의 답**이 생기고, 그 실패는 한 노심의
적합도가 다른 노심에 서빙되는 것이다. GA evaluator 계획 §5.2가 정확히 이렇게 적는다.

### 3.2 어떤 접힘이 합법인가

| symmetry.angle | 궤도 | 논거 |
|---|---|---|
| 90 (쿼터) | `{identity, transpose}` | 쿼터를 전치하면 생성되는 전노심이 대각선에 대해 반사된다 — 정사각 노심의 등거리 변환이고 2군 확산은 그 아래 불변. **다른 7개 정사각 연산은 접힘 코너를 옮기므로 다른 노심이다** |
| 360 (전노심) | 이면체군 (정사각 8, 직사각 4) | 전노심의 회전/반사는 그대로 등거리 변환 |
| 그 외 (180, 특수 섹터) | `{identity}` — 그리고 `core_op: identity`로 **그렇게 말한다** | **논증되지 않은 접힘은 서로 다른 두 노심을 한 캐시 항목에 합치는 방법이다.** 섹터를 추가하려면 그 등거리 변환과 함께 추가한다 |

`core_op`는 **payload에 들어가지 않는다.** 그것은 정식 대표원에 *어떻게* 도달했는지이지
키가 *무엇을* 식별하는지가 아니다. 넣었다면 어떤 패턴과 그 전치가 서로 다른 키를 받아,
접힘이 없애려던 miss가 그대로 남았을 것이다(개발 중 실제로 이 버그를 냈고 계약이 잡았다 —
자기 전치인 fixture를 쓰면 그 체크가 아무 일도 안 하고 통과하므로 fixture에 대해
`QUARTER != transpose(QUARTER)`를 assert 한다).

### 3.3 키에 들어가는 것 / 들어가지 않는 것

**들어간다**: 대칭 정규화된 `core`, `batch` 인벤토리, **나머지 덱 전부**(schedule/연소격자/
T-H/rod/허용오차), 실효 `PhysicsFidelity`와 policy 이름, `trajectory::kArmEnv`의 모든 knob
(raw, 순서 포함), **단면 라이브러리의 내용 digest**(경로 아님), warm-start provenance,
`RASBERY_CODE_SHA`(하네스 선언; 미설정이면 `~`로 인쇄되고 **키가 두 빌드를 구분하지
못한다는 사실을 payload가 말한다**).

**들어가지 않는다**: result mode와 출력 경로. 세 출력 모드가 하나의 궤적 digest임이 이미
실측되었으므로(계획 §2.1), 캐시된 scalar 답은 나중에 다른 형태로 쓰겠다는 요청에도 유효하다.
핀맵처럼 캐시가 **줄 수 없는** 것은 키의 miss가 아니라 산출물의 miss이고, 요청자가 그 모드로
재실행을 요청한다.

### 3.4 두 구현이 한 정의를 갖게 하는 법 — 그리고 그것이 이미 증명되었다는 것

**로컬에서 증명되었다.** `tools/test_case_key_contract.py`의 compiled half가 MSVC로
`src/CaseKey.h`를 헤더 전용 TU로 컴파일해 `deckPayload`를 돌리고, 같은 덱에 대한
`tools/case_key.py`의 출력과 **바이트 단위로** 비교한다. fixture 덱에는 두 언어가 어긋날
수 있는 모든 철자를 심어 두었다(정수, `1.0`, `0.1`, `1e-6`, `2.5e-17`, `1e21`, 음수,
`1/3`, bool, null, 혼합 배열). 실제 `kngr_238.json`에 대해서도 확인했다:
3,604바이트 동일, `deck_digest = fd8714d6…`.

음성 대조군: 파이썬 쪽 float 정밀도를 `%.17g` → `%.16g`로 한 자리만 바꾸면 624번째
바이트에서 실패한다. 즉 §6.6이 최대 위험으로 적었던 항목은 **이미 닫혔고**, 238에 남은
것은 payload의 나머지 절반(환경·라이브러리 digest·fidelity) 뿐이다.

JSON 텍스트가 아니라 **토큰 스트림**이다. 키 순서·공백·부동소수 표기가 직렬화기마다
다르기 때문이다. 값마다 철자가 정확히 하나이고, float는 `%.17g` — C++ `std::format`과
Python이 동일하게 내는 유일한 표기.

arm knob 목록은 **복사하지 않고 `src/Driver.h`에서 파싱한다.** 복사본이 있으면 누군가
C++ 목록에만 knob를 추가하는 날 서로 다른 물리의 두 실행이 조용히 한 캐시 항목을 공유한다.
`kArmEnv`가 없는 `Driver.h`에 대해서는 **fail closed**(계약이 그 경로도 검증한다).

### 3.5 해시가 하나뿐인 이유

`BatchLightResult`가 자기 SHA-256을 들고 있었다. WP10.1이 같은 digest를 메모리 payload에
대해 필요로 하므로 변환을 `include/chiffon/Sha256.h`로 **상수 하나 안 바꾸고** 옮기고 두
호출자가 공유한다. 해시 함수 복사본 두 개는 캐시 키가 자기를 이름 붙인 수신증과 어긋나는
방법이다. 컴파일러가 없으므로 알고리즘은 **Python 전사(轉寫)로 hashlib에 대해 검증**했다
(0/55/56/63/64/119바이트 및 768바이트 메시지 전부 일치).

---

## 4. WP10.2 — warm start

### 4.1 덱 스키마에 이미 있던 채널, 그리고 왜 그것이 아닌가

**먼저 확인했다**(과제 요구): `restart` 블록이 실재하고
(`src/IO.cpp:588-644` 파싱, `src/IO.cpp:858-1006` 복원, `LoadGeometryFromRestart`
`src/IO.cpp:2558`) flux·burnup·isotope·T/H를 되살린다. **사이클 이어달리기에는 맞고
형제 시딩에는 틀리다:**

1. HDF5 1.10.x는 프로세스 전역·비스레드안전이라 배치 워커마다 줄을 선다(계획 §3.1(a)).
2. `--result light`는 restart를 **아예 쓰지 않는다** — 그리고 light가 GA arm이다.
3. burnup과 isotope 인벤토리를 복원한다. **다른 LP에 그것은 warm start가 아니라 틀린 연료다.**

### 4.2 최소 집합, 그리고 그 논거

**BOC flux + 임계붕소 + k_eff. 그 외 없음.** 셋 다 solve가 덮어쓰는 **추측**이고,
연료·기하·인벤토리가 아니다. 그것이 실패 모드를 "더 나쁜 출발점에서 수렴한다"로 묶고
"다른 문제로 수렴한다"가 되지 않게 하는 유일한 근거다.

**BOC만**: `initial` 버킷은 상태점마다 있지만 **덱에서 출발하는 상태점은 첫 번째뿐**이고
나머지는 이미 직전 상태점의 수렴 flux에서 출발한다 — 그보다 나은 warm start는 없다.
파일 35개는 이미 시딩된 상태점 34개를 다시 시딩하는 것이다. 케이스당 ~135 kB(ng×nxyz double).

### 4.3 게이트 등급 N1, 그리고 수신증이 그렇게 말한다

출발점이 다르면 Xe↔flux 사상이 다근인 구간에서 **다른 근을 고를 수 있다**
(`A2_OUTER_REDUCTION` §5의 i-SMR CY02 `primeXeDamping` 전례). 따라서 digest는 **움직여도
되고**, 게이트는 keff/CBC/Fq/FΔH가 수용 문턱 안인지다 — GA evaluator 계획 §5.4가 요구하는
그대로. warm-start provenance는 부모 상태 파일의 **내용 digest**이고 WP10.1 키에 접힌다.
따라서 warm 답이 cold 캐시 항목에서 나올 수 없다. **거절된 warm start는 provenance를
비운다** — cold로 돌아간 실행은 cold처럼 키를 받아야 하기 때문이다.

### 4.4 모든 거절은 cold start다

파일 없음 / 낯선 magic / 버전 상승 / 기하 불일치 / 비물리 seed(`keff ∉ (0.1, 3.0)` 또는
`boron < 0`): 각각 **이유를 반환**하고 cold flux가 그대로 서며 수신증이 어느 것인지 말한다.
`WarmState.h`는 아무것도 throw 하지 않는다.

### 4.5 feature-off 동일성

두 플래그 모두 미설정이면 파일을 열지 않고, flux를 쓰지 않고,
**`[RASBERY][WARMSTART]` 줄도 인쇄하지 않는다** — 이 빌드의 로그가 이 기능이 없는 빌드의
로그다.

### 4.6 API

```
--warm-start-from FILE           부모의 BOC 상태에서 시딩
--save-warm-state [FILE]         자식을 위해 이 케이스의 BOC 상태를 저장
                                 FILE 없으면 job의 --raso에서 <dir>/<stem>.warm 유도
                                 (Driver::RestartPath와 같은 네임스페이스 규칙)
```

evaluator 요청의 케이스마다 `"warm_start_from"` / `"save_warm_state"`. 명시 경로 +
덱 여러 개는 **이름을 대고 거절한다**(N개 덱이 한 파일에 쓰면 N-1개가 사라지고 승자가
임의다). isolation recheck는 같은 warm start를 넣고 **저장은 하지 않는다** — 자기 wave의
첫 케이스가 만든 부모 상태를 덮어쓸 것이기 때문이다.

light JSONL이 `warm_state`를 싣는다. GA가 로그를 grep 하지 않고 이미 읽는 결과에서
다음 세대를 시딩한다.

---

## 5. 로컬에서 실행한 검증 (하드웨어 없음)

```bash
python3 tools/test_statepoint_telemetry.py      # WP9-A + WP9-D, 음성 대조군 7
python3 tools/test_telemetry_neutrality.py      # 소스 절반
python3 tools/test_case_key_contract.py         # WP10.1, 소스 + 거동 + **컴파일 바이트 비교**
python3 tools/test_warm_start_contract.py       # WP10.2, 음성 대조군 5
python3 tools/case_cost_profile.py <log>        # 새 분해 인쇄
```

**로컬에서 확인한 것**: `src/CaseKey.h` + `include/chiffon/Sha256.h`가 컴파일되고,
SHA-256이 FIPS 벡터와 맞고, `deckPayload`가 Python과 바이트 동일하다(§3.4).

**로컬에서 확인할 수 없는 것**: 전체 빌드(CUDA/HDF5), 키의 환경·라이브러리 절반,
어떤 성능 수치든.

---

## 6. 238 runbook

### 6.0 공통 전제

계획 §6.4: GPU0만, 첫 실행은 warm-up으로 버리고, telemetry 실행과 wall timing 실행을
**분리**하며, A/B는 교대로 돌린다. 아래에서 `$BIN`은 이 트리에서 빌드한 바이너리,
`$O`는 출력 디렉터리다.

```bash
# 0) 빌드 (WP0 기준 플래그) — 이 커밋들의 첫 관문이 컴파일이다
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
export CUDA_VISIBLE_DEVICES=0
# 0b) 소스 계약을 같은 트리에서 한 번 더
for t in statepoint_telemetry telemetry_neutrality case_key_contract warm_start_contract; do
  python3 tools/test_$t.py || echo "FAIL $t"
done
```

### 6.1 단일 production arm — 수신증 (WP9-A + WP9-D)

**목적: `c`와 `d`의 항목별 내역, 그리고 탐색 분류 분포. 성능 판정이 아니다.**

```bash
# telemetry ON, v3 arm (V3_FREEZE §2). 이 실행의 wall은 인용하지 않는다.
RASBERY_STATEPOINT_TELEMETRY=1 RASBERY_OMP_THREADS=1 \
  /usr/bin/time -f "%e" -o $O/tel.wall \
  $BIN --rasi kngr_238.json --raso $O/tel.h5 > $O/tel.log 2>&1

python3 tools/case_cost_profile.py $O/tel.log --wall-dir $O
python3 tools/case_cost_profile.py $O/tel.log --json > $O/tel_ledger.json
```

읽어야 하는 것:

| 줄 | 무엇을 결정하는가 |
|---|---|
| `floor  ppr / depl / result = Σ` | WP9 단계 B(CRAM GPU 이식) 재가격. 계획 §WP9: **10 % 미만이면 이식하지 않는다** |
| `loop   th / xe / search = Σ` | 단계 C(TH)의 존폐. 그리고 `xe`가 §2.2의 69 %와 일치하는지 |
| `nested flatxs (ms/call)` | WP5의 실제 지분. FlatXS가 loop/floor 안에서 얼마인지 |
| `accounted … residual` | **아직 이름 없는 위상.** 여기가 크면 다음 계기가 필요하다 |
| `boundary H2D/D2H` | canonical residency(WP7-C 잔여)가 겨냥할 바이트 |
| `search trials / outers / outers-per-trial` | §2.4에서 어느 후보를 고를지 |
| `search method probe/carry/secant/bisect` | 같은 것. **분포가 후보를 고른다** |

동시에 `[RASBERY][SPTELEM]` 상태점 줄에서 `search.x_first` / `search.dx_last` /
`search.dk`를 뽑아 두면 6.4의 warm-start A/B에서 직접 비교된다.

**계기 중립성(필수, 계획 §6.4)**: telemetry OFF 실행을 한 번 더 하고

```bash
RASBERY_OMP_THREADS=1 $BIN --rasi kngr_238.json --raso $O/off.h5 > $O/off.log 2>&1
python3 tools/test_telemetry_neutrality.py --compare $O/off.log $O/tel.log
```

→ `env`와 `digest`가 같고 `telemetry`만 다르면 PASS. **여기서 FAIL이면 6.1의 모든 숫자를
버린다.**

### 6.2 wall timing arm (별도 실행)

```bash
for i in 1 2 3 4 5; do
  RASBERY_OMP_THREADS=1 /usr/bin/time -f "%e" -o $O/w$i.wall \
    $BIN --rasi kngr_238.json --raso $O/w$i.h5 > $O/w$i.log 2>&1
done
```
median/p10/p90을 기록한다. WP9-A/WP9-D는 **계기 추가**이므로 이 arm의 판정 기준은
"현행 16.9 s 대비 오버헤드 <1 %"이며, 초과하면 계기 자체가 결함이다.

### 6.3 WP10.1 — case key의 라이브 게이트

**남은 절반만이 게이트다.** 덱 payload의 바이트 동일성은 §3.4에서 이미 컴파일해서
증명했으므로, 여기서 새로 검증되는 것은 payload의 나머지 — 환경 knob 문자열, 단면
라이브러리 내용 digest, 실효 fidelity — 와 그것들이 실제 실행에서 같은 값을 갖는지다.

```bash
python3 tools/test_case_key_contract.py --compare $O/off.log kngr_238.json
```

`case_key` / `key_schema` / `core_op` / `deck_digest` 네 필드가 모두 일치해야 한다.
불일치가 나오면 거의 확실히 부동소수 표기다 — 그때는

```bash
python3 tools/case_key.py kngr_238.json --deck-payload > $O/py_deck_payload.txt
```
와 C++ 쪽 payload를 비교한다(필요하면 `deckPayload` 결과를 임시로 인쇄하는 1줄 패치).
**주의**: `case_key.py`는 XS 경로를 덱 디렉터리 기준으로 풀고 내용을 digest 한다. WSL UNC
경로 등으로 solver와 tool이 서로 다른 파일을 열면 digest가 비고 키가 어긋난다 —
**같은 호스트, 같은 경로에서 돌릴 것.**

대칭 접힘의 라이브 확인(선택이지만 값싸다):

```bash
# core 블록을 전치한 덱을 만들어 같은 키가 나오는지
python3 - <<'PY'
import json
d = json.load(open("kngr_238.json"))
d["core"] = [list(r) for r in zip(*[list(r) for r in d["core"]])]
json.dump(d, open("kngr_238_T.json", "w"), indent=1)
PY
diff <(python3 tools/case_key.py kngr_238.json) <(python3 tools/case_key.py kngr_238_T.json)
# 기대: 차이 없음.  그리고 전치 덱을 실제로 풀면 keff/CBC가 수용 문턱 안이어야 한다
```

### 6.4 WP10.2 — warm-start A/B (부모 = `candidate_0000`, 자식 = `candidate_0001`)

```bash
# 1) 부모: BOC 상태를 남긴다
RASBERY_OMP_THREADS=1 $BIN --rasi candidate_0000.json --raso $O/p.h5 \
    --save-warm-state $O/p.warm > $O/parent.log 2>&1
python3 tools/test_warm_start_contract.py --inspect $O/p.warm   # ng/nxyz/keff/boron/complete

# 2) 자식 COLD
RASBERY_OMP_THREADS=1 /usr/bin/time -f "%e" -o $O/child_cold.wall \
    $BIN --rasi candidate_0001.json --raso $O/c_cold.h5 > $O/child_cold.log 2>&1

# 3) 자식 WARM (같은 덱, 같은 arm, 부모 상태만 추가)
RASBERY_OMP_THREADS=1 /usr/bin/time -f "%e" -o $O/child_warm.wall \
    $BIN --rasi candidate_0001.json --raso $O/c_warm.h5 \
    --warm-start-from $O/p.warm > $O/child_warm.log 2>&1

# 4) 판정
python3 tools/test_warm_start_contract.py --compare $O/child_cold.log $O/child_warm.log
```

`--compare`가 인쇄/판정하는 것:

| 지표 | 출처 | 채택 기준 |
|---|---|---|
| `initial_outers` cold vs warm | `[RASBERY][WARMSTART].initial_outers` | 계획 §10.2 목표는 **initial/search outer 20 % 이상 감소**. 미달이면 "레버가 이 family에서는 작다"로 기록 |
| 전체 `outers` | `[RASBERY][TRAJECTORY].outers` | 증가하면 즉시 폐기(계획 §10.2: "iteration 증가에서 cold로 재실행") |
| wall | `.wall` 사이드카, 교대 5쌍 median | ≤3 %면 노이즈(계획 §6.4) |
| `digest` | `[RASBERY][TRAJECTORY]` | **동일할 필요 없다 — N1이다.** 다르면 그것만으로 실패가 아니다 |
| max \|dk\|, max \|dppm\| | `NO.=` 줄 | **이것이 acceptance 게이트다.** \|dk\| ≤ 2e-5(search_tol), \|dppm\| ≤ 1 ppm |
| `load` 상태 | `[RASBERY][WARMSTART].load` | `applied`가 아니면 A/B 자체가 성립하지 않는다 |

**Gate A 델타**(Fq/FΔH/AO/핀 RMS·max)는 기존 `tools/gate_a_compare.py`로 `c_cold.h5` vs
`c_warm.h5`를 비교한다 — 이것이 계획 §5.4가 요구하는 "digest가 아니라 keff/CBC/Fq/FΔH"의
나머지 절반이다.

**feature-off 동일성(필수, B0)**:

```bash
# 같은 바이너리, 플래그 없이 두 번 — 그리고 이전 커밋 바이너리와 h5diff
h5diff -c $O/c_cold.h5 $O/c_cold_prev.h5 && echo "feature-off B0 OK"
grep -c "RASBERY\]\[WARMSTART\]" $O/child_cold.log   # 기대: 0
```

### 6.5 M64 — 이전 세대에서 warm start (evaluator 경로)

```bash
# 세대 N: 각 케이스가 자기 warm 상태를 남긴다 (경로는 --raso에서 유도)
cat > $O/gen_N.jsonl <<'EOF'
{"op":"case","deck":"g0/c00.json","output":"g0/c00.h5","result_mode":"light","key":"g0-00","save_warm_state":"g0/c00.warm"}
...  (64줄)
{"op":"wave","wave_id":0}
{"op":"run"}
EOF

# 세대 N+1: 각 자식이 부모를 지목한다
cat > $O/gen_N1.jsonl <<'EOF'
{"op":"case","deck":"g1/c00.json","output":"g1/c00.h5","result_mode":"light","key":"g1-00","warm_start_from":"g0/c00.warm"}
...
{"op":"wave","wave_id":1}
{"op":"run"}
{"op":"shutdown"}
EOF

RASBERY_OMP_THREADS=1 RASBERY_BATCH_RECEIPT_JSONL=$O/g1.jsonl \
  $BIN --evaluator-jsonl $O/gen_N1.jsonl --batch-mode 64 > $O/g1.log 2>&1
```

판정:

1. `[RASBERY][EVALUATOR][CASE].case_key`가 64개 모두 서로 다른가 (같으면 중복 후보이고,
   그것은 **캐시가 잡아야 했던 것**이다 — 실제 hit rate의 첫 실측).
2. `[RASBERY][WARMSTART].load`가 64개 모두 `applied`인가. `cold_fallback`이 있으면
   `reason`이 기하 불일치인지(= 부모/자식 geometry가 실제로 다름) 파일 문제인지 본다.
3. c/h를 cold 세대(같은 64덱, `warm_start_from` 없이)와 교대로 비교. 계획 §6.4의 3 % 규칙.
4. **acceptance**: elite로 승격될 후보는 warm 결과를 그대로 쓰지 않는다. 계획 §10.2 —
   "acceptance-eligible elite는 cold 또는 검증된 strict rerun으로 확인한다".
   `case_key`의 `warm_start` 필드가 그 재실행을 강제하는 근거다(warm 키 ≠ cold 키).

### 6.6 이 커밋들이 238에서 실패할 수 있는 지점 (정직하게)

| 지점 | 증상 | 대응 |
|---|---|---|
| 컴파일 | 로컬에 컴파일러가 없다 | 첫 관문. `std::format("{:.17g}")`은 libstdc++/fmt 양쪽에서 지원되지만 확인 필요 |
| ~~`{:.17g}` vs `%.17g`~~ | ~~`deck_digest` 불일치~~ | **종결.** §3.4의 compiled half가 MSVC에서 바이트 동일을 증명했다(`1e-06` 형태 포함). 238에서 다른 컴파일러(GCC/libstdc++)를 쓰므로 §6.3 `--compare`가 그 툴체인에서의 확인을 계속한다 |
| warm state 바이트 순서 | 다른 머신의 `.warm`을 읽으면 무의미한 값 | 설계상 host-local. 기하 검사와 seed 타당성 검사가 대부분 잡지만, **캠페인 안에서만 쓸 것** |
| `nested_wall.flatxs` 오버헤드 | telemetry ON 실행에서 `UpdateFlatXS`가 매우 자주 호출되면 clock read가 보인다 | 6.1의 중립성 비교와 6.2의 <1 % 규칙이 이것을 잡는다. 문제가 되면 `LB_FLATXS`만 별도 게이트 |

---

## 7. 다음에 해야 할 것

1. **6.1을 먼저 돌린다.** WP9 단계 B/C의 존폐, WP5의 지분, WP9-D 후보 선택이 전부 그 한
   출력에 달려 있다. 그 전에 코드를 더 쓰지 않는다(계획 §WP9 원칙: "재프로파일에서 10 %
   이상인 항목만 구현한다").
2. 6.3이 통과해야 WP10.1이 GA에 쓸모가 있다. 통과 전까지 컨트롤러는 캐시를 켜지 않는다.
3. 6.4의 `initial_outers` 감소가 20 % 미달이면 **WP10.2는 여기서 멈춘다** — 계획이 정한
   목표이고, 미달을 "그래도 켜 두자"로 바꾸면 N1 위험만 남는다.
4. WP10.3(multi-fidelity 승격)은 **미착수**. 본 작업 범위가 solver-side hook까지였고,
   승격 정책은 GA 담당이다.
