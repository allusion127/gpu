# WP9-D 단계 D — 탐색 trial 감축 레버 다섯 개, 그리고 그것을 고르는 방법 (2026-08-31)

문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 계획 | `GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` WP9 **단계 D — search trial 감소** |
| 후보/게이트 등급의 출처 | `WP9_WP10_FLOOR_RECEIPTS_WARMSTART_20260831_KO.md` §2.4 (D1~D5, 게이트 등급 **사전 확정**) |
| 계측의 출처 | 같은 문서 §2.2 — `3df4ea7`이 상태점마다 `trials/proposals/refused/secant/carry_secant/probe/bisect/iterations`를 인쇄한다 |
| 구현 위치 | `src/Scheduler.h` (정책·산법), `src/Driver.h` (배선·수신증) |
| 로컬에서 한 것 | 소스 계약 + **MSVC `/Zs`로 `Scheduler.h`를 격리 TU로 구문 검증**(모든 신규 진입점 호출). 전체 빌드는 CUDA/HDF5 때문에 238에서만 된다 — **성능 수치는 하나도 없고, 있어서도 안 된다** |
| 238에서만 가능한 것 | §5 전부 |

---

## 0. 한 줄 결론

**다섯 개의 레버가 각자의 플래그 뒤에 들어갔고 전부 기본 OFF이며, 어느 것도 서로 묶여
있지 않다. 어느 것을 채택할지는 이 커밋이 정하지 않는다 — §2.2의 분류 분포가 정한다.**

계획 §WP9의 원칙("재프로파일에서 10 % 이상인 항목만 구현한다")과 충돌하지 않는다:
여기서 구현된 것은 **꺼져 있는 레버**이고, 켜는 판단은 §5.1의 출력을 본 뒤에 한다.
레버가 없으면 분포를 보고도 아무것도 할 수 없고, 레버를 하나로 묶으면 분포를 봐도
어느 것이 값을 냈는지 말할 수 없다.

---

## 1. 다섯 개의 레버

| 플래그 | 후보 | 게이트 | 무엇을 바꾸는가 | feature-off가 옛 경로인 이유 |
|---|---|---|---|---|
| `RASBERY_SEARCH_CARRY_SLOPE=1` | D1 | **N1** | 상태점 경계를 넘어 전달되는 boron secant 기울기를 **EFPD에 대한 1차 외삽**으로 보정 | `carry_override=false`이면 전달되는 값이 `*params.secant_dkdx`의 복사본. 나눗셈 피연산자가 같은 double |
| `RASBERY_SEARCH_WARM_BORON=1` | D2의 solver 절반 | **N1** | 첫 탐색의 첫 trial을 **부모(WarmState) 붕소**에서 출발. 덱의 `search_boron_ppm`을 이긴다 | `policy.warm_boron`가 false면 `has_warm_boron`을 **저장조차 하지 않고**(Driver.h), 출발점 식은 원래 삼항식 그대로 |
| `RASBERY_SEARCH_BORON_BRACKET=1` | — (D5의 착지점) | **N1** | boron 탐색에 RODCRIT의 부호변화 bracket + 이분 대체를 부여 | `use_bracket`이 false면 `AdvanceSecantSearch`의 bracket 블록 전체가 진입 불가 |
| `RASBERY_SEARCH_MAX_TRIALS=<n>` | D5 | **N1** | SolveLoop당 커밋 trial 상한을 덱의 `max_search_iter` **아래로** | `trialCap(deck)`은 미설정 시 `deck`을 그대로 반환 — 같은 비교식 |
| `RASBERY_SEARCH_STAGED_MARGIN=<x>` | D3 | **A2** | staged 단계에서 탐색 표본이 지키는 `search_tol / margin`의 그 margin (기존 리터럴 4.0) | `stagedMargin(STAGED_SEARCH_MARGIN)`이 미설정 시 4.0을 반환. 게다가 staging이 꺼져 있으면 loose 단계 자체가 없어 **읽히지도 않는다** |

### 1.1 D2가 "solver 절반"인 이유

원문 D2는 "동일 family accepted case에서 초기 bracket 제안"이다. **accepted case의
데이터베이스는 GA 컨트롤러 쪽에 있고 solver에는 없다.** solver가 순수하게 가진
동종의 채널은 WP10.2가 이미 만들어 둔 것 하나뿐이다 — 부모의 BOC 붕소. 그래서
여기서 구현된 것은 그 절반이고, bracket 시딩의 나머지 절반은 evaluator/GA 과제로
남는다. 이 문서는 그것을 D2라고 부르지 않고 "D2의 solver 절반"이라고 부른다.

### 1.2 D1의 보정이 물리 모델이 **아닌** 이유

붕소가 반응도에 갖는 가치의 연소도 의존성을 상관식으로 넣으면 그 상관식이 틀린 날
첫 trial이 지금보다 멀어진다 — 그리고 그것이 §2.4가 D1에 붙인 위험 그대로다.
그래서 모델은 **그 실행 자신이 방금 측정한 두 기울기**다:

```
trend     = (last_slope - prev_slope) / (last_efpd - prev_efpd)
predicted =  last_slope + trend * (efpd_now - last_efpd)
```

세 개의 가드 중 하나라도 걸리면 **보정 없이** 기존 carry가 그대로 선다.

| 가드 | 막는 것 |
|---|---|
| `isfinite(predicted)` | 두 점으로 말할 수 없다는 산술의 자백 |
| `predicted * last_slope > 0` | 부호가 뒤집히면 첫 trial이 **반대 방향**으로 간다 |
| `0.5 ≤ |predicted|/|last_slope| ≤ 2.0` | 두 점이 지지하지 않는 거리까지 나간 외삽 |
| `last_efpd > prev_efpd` | 같은 EFPD의 두 기울기 = 0으로 나누기 |
| `has_prev && has_last` | 기울기 하나는 추세가 아니다 |

측정 기울기가 움직이지 않은 상태점(= secant 걸음이 없었던 상태점)은 같은 값을 다시
push 하므로 trend가 0이 되고 보정은 **정확히 기존 carry로 퇴화한다.** 즉 최악의 경우가
현행 동작이다.

### 1.3 D5가 acceptance를 건드리지 않는 이유 — 구조적으로

상한에 걸린 SolveLoop는 덱의 `max_search_iter`가 걸렸을 때와 **같은 출구**
(`SolveExit::SEARCH_EXHAUSTED`)로 나간다. 그 뒤는 기존 결정론적 acceptance 블록이다:
best 관측점을 재적용하고 **production 허용오차로** 재수렴시킨 뒤
`search_exit_status`를 `BEST_FALLBACK`/`UNCONVERGED`로 publish 한다. `CONVERGED`가 아닌
상태점은 **이미** elite가 될 수 없었으므로 acceptance 규칙은 한 줄도 바뀌지 않는다
(§2.4가 D5에 건 조건 그대로). 계약이 이것을 두 갈래로 고정한다 — 상한이 자기 출구를
만들지 못하게, 그리고 상한이 `search_exit_status`를 스스로 쓰지 못하게.

### 1.4 D3가 새로운 fidelity 단어를 만들지 않는 이유

`RASBERY_SEARCH_STAGED_MARGIN`은 **loose 단계에서만** 읽힌다. staging이 꺼져 있으면
`polishing`이 처음부터 참이고 loose 허용오차는 어디서도 읽히지 않으므로, 이 knob 하나로
strict 실행이 strict가 아니게 되는 경로가 없다. 따라서 A2 탐지
(`RASBERY_STAGED_FLUX_TOL/_XE_TOL`, `RunContract.h`)는 그대로 두고 이 knob은 그 위에
얹힌다. 수신증의 `gate` 필드가 **A2인데 arm env에 staged 승수가 없다면 그 knob은 아무
일도 하지 않은 것**이고, 그 사실이 한 줄에서 읽힌다.

---

## 2. 수신증

### 2.1 상태점마다 — 기존 `search` 객체에 `extrap` 한 필드

`[RASBERY][SPTELEM]`의 `"search"`가 `"extrap"`을 얻었다. `carry_secant`는 레버가
**움직일 수 있었던** 걸음 수이고 `extrap`은 **실제로 보정된 기울기로 걸은** 수다.
차이가 곧 §1.2의 가드가 거절한 횟수다. `tools/case_cost_profile.py`의 method 줄이
`carry N (extrap M)`으로 인쇄한다.

### 2.2 실행마다 — `[RASBERY][SEARCH_POLICY]`

**하나라도 켜져 있을 때만 인쇄된다.** 전부 꺼져 있으면 이 빌드의 로그는 이 기능이 없는
빌드의 로그다.

```
[RASBERY][SEARCH_POLICY] {"schema_version":1,"slot":-1,"gate":"N1",
  "carry_slope":true,"warm_boron":false,"boron_bracket":false,"max_trials":0,
  "staged_margin":0.0000,"trials":137,"proposals":151,"refused":2,"probe":1,
  "carry_secant":34,"extrap":31,"secant":98,"bisect":0,
  "search_outers":707,"outers":4609,"statepoints":35}
```

집계는 **`RASBERY_STATEPOINT_TELEMETRY`와 무관하게** 누적된다(상태점당 정수 9회 덧셈).
이유는 WP10.2의 `initial_outers`와 같다 — wall timing arm은 telemetry를 끄고 돌리는데
(계획 §6.4), 짧게 만들려는 바로 그 arm에서 읽을 수 없는 수신증은 값을 매길 수 없는
수신증이다.

### 2.3 arm 이름과 캐시 키

다섯 knob 전부 `trajectory::kArmEnv`에 있다. 따라서

* `[RASBERY][TRAJECTORY]`의 `env`가 **원문 그대로** 다섯 값을 싣는다 — A/B가 "다른 knob은
  전부 같았다"를 기계적으로 증명할 수 있다.
* WP10.1 `case_key`가 이들을 접는다 — **한 탐색 정책으로 계산된 캐시 답이 다른 정책의
  요청에 서빙되지 않는다.**

부작용을 정직하게 적는다: kArmEnv가 5개 늘었으므로 **모든 실행의 `env_digest`와
`case_key`가 이 커밋에서 바뀐다**(knob이 미설정이어도 이름과 `null`이 payload에 들어간다).
궤적·결과·digest는 움직이지 않는다. 이전 캠페인의 캐시 항목은 이 빌드에서 miss가 되며,
그것이 옳다 — 다른 정책 공간을 가진 바이너리다.

---

## 3. 로컬에서 실행한 검증 (하드웨어 없음)

```bash
python3 tools/test_search_policy_contract.py            # 신규. 음성 대조군 5 + 변이 5
python3 tools/test_staged_tolerance.py                  # D3가 리터럴을 대체만 하는지
python3 tools/test_statepoint_telemetry.py              # extrap 필드 + WP9-D 중립성 스캔
python3 tools/test_telemetry_neutrality.py
python3 tools/test_gpu_physics_interface_contract.py    # SearchMemory 5필드 불변 (컴파일 포함)
python3 tools/test_case_key_contract.py                 # kArmEnv 파싱 + 컴파일 바이트 비교
python3 tools/test_warm_start_contract.py
python3 tools/test_case_fidelity_contract.py
python3 tools/test_segment_canonical_nodal_contract.py
python3 tools/test_enum_alias_contract.py
python3 tools/test_dependent_template_contract.py
python3 tools/test_xe_forms_default_contract.py
python3 tools/test_xe_forms_shipped_split_contract.py
python3 tools/test_xe_anderson.py
```

전부 PASS.

### 3.1 새 계약이 실제로 무엇을 잡는지 — 변이로 확인했다

계약이 통과하는 것만으로는 계약이 아니다. 다섯 개의 변이를 소스에 넣고 각각 FAIL을
확인한 뒤 되돌렸다.

| 변이 | 잡힌 항목 |
|---|---|
| `if (ctx.search_policy.warm_boron)` 게이트 제거 | knob이 꺼져 있어도 seed를 저장 → feature-off 위반 |
| `trialCap`이 덱 상한보다 큰 값을 반환하도록 | 상한 knob이 덱이 거절한 trial을 **부여** |
| `RASBERY_SEARCH_BORON_BRACKET`을 kArmEnv에서 삭제 | case key가 접지 않는 궤적 knob |
| 부호 가드 삭제 | 첫 trial이 반대 방향으로 갈 수 있는 외삽 |
| 크기 가드 삭제 | 두 점이 지지하지 않는 외삽 |

그리고 외삽 산법 자체는 계약 안에서 **전사(轉寫)로 실행**된다: 정상 케이스 1개 +
음성 대조군 5개(2.5배 거절 / 경계값 2.0배 **수용** / 부호전환 거절 / 기울기 1개 거절 /
EFPD 폭 0 거절 / 평탄 추세는 기존 carry로 퇴화). 경계값 대조군이 있어야 가드를
"전부 거절"로 조여 놓고도 통과하는 일이 없다.

### 3.2 컴파일

`src/Scheduler.h`를 `Geometry.h` 스텁과 함께 격리 TU로 MSVC `/Zs /std:c++20 /W3`에
통과시켰고, TU가 신규 진입점을 **전부 호출한다**(`processSearchPolicy`,
`carriedBoronSlope`, `SearchPolicy::{any,stagedMargin,trialCap}`, `searchFlagEnabled`,
`StartCriticalSearch`, `ProposeNextSearchPoint`, `UpdateRodBracket`,
`UpdateSearchBracket`). 서명 오류가 미사용 inline 멤버 뒤에 숨을 수 없다.

`src/Driver.h`는 로컬에서 컴파일되지 않는다(CUDA/HDF5). 그쪽에서 리뷰로도 컴파일로도
잡히지 않는 유일한 결함 — `std::format` placeholder/인자 수 불일치 — 는
`tools/test_statepoint_telemetry.py` §6c가 두 SPTELEM 수신증에 대해 **재계수**한다.
신규 `[RASBERY][SEARCH_POLICY]` 줄은 그 스캔 밖이므로 손으로 셌다: **placeholder 18 /
인자 18**. 238의 첫 관문은 여전히 컴파일이다.

---

## 4. 238에서 실패할 수 있는 지점 (정직하게)

| 지점 | 증상 | 대응 |
|---|---|---|
| 컴파일 | 로컬에 CUDA/HDF5가 없다 | 첫 관문. `Driver.h`의 신규 코드는 `std::format` 한 줄과 배선 5곳뿐 |
| feature-off drift | 모든 knob 미설정인데 digest가 움직인다 | **즉시 중단.** 캠페인이 쫓고 있는 `7cfe3a4`↔`d7b81af` 미귀속 drift와 섞이면 둘 다 못 푼다. §5.0의 B0 실행이 이것을 먼저 배제한다 |
| D1이 아무것도 안 한다 | `extrap` = 0 | 가드가 거절 중이거나 상태점이 3개 미만이다. `carry_secant` 대비 비율이 답 |
| D3가 근을 옮긴다 | `staged_relapses`가 `trials`와 같은 자릿수 | §2.4의 D3 즉시 폐기 조건. margin을 올린다(느슨함이 줄어든다) |
| bracket이 안 잡힌다 | `bisect` = 0 | boron 잔차가 부호를 바꾸지 않았다는 뜻 — 레버가 무해하게 놀고 있는 것이지 결함이 아니다 |
| 상한이 수렴을 깬다 | `exit != 1`인 상태점 증가 | 설계된 동작. **그 상태점은 acceptance 대상이 아니다**. 채택 판단은 §5.4의 Gate B가 한다 |

---

## 5. 238 runbook

### 5.0 공통 전제 + feature-off B0 (**먼저**)

계획 §6.4: GPU0만, 첫 실행은 warm-up으로 버리고, telemetry 실행과 wall timing 실행을
**분리**하며, A/B는 교대로 돌린다. arm은 v3 production arm 하나
(`tools/run_single_gpu_batch.py:DEFAULT_ENV`, 아래 `ARM`)를 고정한다.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
export CUDA_VISIBLE_DEVICES=0
BIN=build/RASBERY ; O=~/wp9d ; mkdir -p $O

ARM="RASBERY_GPU=1 RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 \
RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1 \
RASBERY_GPU_WIEL_FOLD=chunked RASBERY_GPU_XE=1 RASBERY_XE_ANDERSON=1 \
RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1 \
RASBERY_GPU_RB_SWEEPS=4 RASBERY_PC_MODE=decart RASBERY_PPR_MODE=master \
RASBERY_OMP_THREADS=1"

# B0: 이 커밋의 바이너리가 knob 없이 이전 커밋의 바이너리와 같은 답을 내는가
env $ARM $BIN --rasi kngr_238.json --raso $O/base.h5 > $O/base.log 2>&1
h5diff -q $O/base.h5 $O/base_prev.h5 && echo "feature-off B0 OK"
grep -c "RASBERY\]\[SEARCH_POLICY\]" $O/base.log    # 기대: 0
diff <(grep -o '"digest":"[^"]*"' $O/base.log) <(grep -o '"digest":"[^"]*"' $O/base_prev.log)
```

**여기서 FAIL이면 아래를 하나도 돌리지 않는다.** `base_prev`는 `c645124` 빌드다.
`env`는 knob 5개만큼 달라지는 것이 정상이고, `digest`는 **한 비트도** 달라지면 안 된다.

### 5.1 분포를 먼저 본다 — 무엇을 켤지는 이 출력이 정한다

```bash
env $ARM RASBERY_STATEPOINT_TELEMETRY=1 \
  $BIN --rasi kngr_238.json --raso $O/tel.h5 > $O/tel.log 2>&1
python3 tools/case_cost_profile.py $O/tel.log
python3 tools/test_telemetry_neutrality.py --compare $O/base.log $O/tel.log
```

| `search method` 줄이 이렇게 나오면 | 먼저 켤 것 |
|---|---|
| `probe`가 지배 | `RASBERY_SEARCH_WARM_BORON` (그리고 GA 쪽 bracket 시딩이 진짜 후보) |
| `carry`가 많고 `secant`가 뒤따름 | `RASBERY_SEARCH_CARRY_SLOPE` |
| `secant`가 지배 | `RASBERY_SEARCH_STAGED_MARGIN` (**trial 수가 아니라 trial당 outer를 줄이는 유일한 후보**) |
| `bisect`가 많거나 `refused` > 0 | `RASBERY_SEARCH_BORON_BRACKET` + `RASBERY_SEARCH_MAX_TRIALS` |

`outers/trial` 비율이 모든 감축안의 가격 기준이다(§2.2: trial 1회 ≈ 38 outer).

### 5.2 단독 arm — 플래그 하나씩

```bash
for k in "RASBERY_SEARCH_CARRY_SLOPE=1" \
         "RASBERY_SEARCH_WARM_BORON=1" \
         "RASBERY_SEARCH_BORON_BRACKET=1" \
         "RASBERY_SEARCH_MAX_TRIALS=12" \
         "RASBERY_SEARCH_STAGED_MARGIN=2"; do
  n=$(echo $k | cut -d= -f1 | sed 's/RASBERY_SEARCH_//' | tr A-Z a-z)
  env $ARM $k RASBERY_STATEPOINT_TELEMETRY=1 \
    $BIN --rasi kngr_238.json --raso $O/$n.h5 > $O/$n.log 2>&1
  python3 tools/case_cost_profile.py $O/$n.log
  grep "SEARCH_POLICY" $O/$n.log
done
```

`RASBERY_SEARCH_WARM_BORON`은 부모 상태 파일이 있어야 의미가 있다 — WP10.2 §6.4의
부모/자식 쌍을 그대로 쓰고, `--warm-start-from $O/p.warm`을 함께 준다. **부모 없이
켜면 아무 일도 일어나지 않고 그것이 정상이다**(`extrap`/`x_first`가 그렇게 말한다).

`MAX_TRIALS=12`와 `STAGED_MARGIN=2`는 출발값일 뿐이다. 5.1이 `secant` 지배로 나오면
margin은 2 → 1.5 → 1.2로, `bisect` 지배로 나오면 상한은 12 → 8로 좁힌다.

### 5.3 조합 arm

단독에서 **outer를 줄이면서 `exit != 1`을 늘리지 않은** 것들만 합친다. 조합은
최대 3개까지 — 그 이상은 어느 것이 값을 냈는지 말할 수 없다.

```bash
env $ARM RASBERY_SEARCH_CARRY_SLOPE=1 RASBERY_SEARCH_BORON_BRACKET=1 \
  RASBERY_STATEPOINT_TELEMETRY=1 \
  $BIN --rasi kngr_238.json --raso $O/combo.h5 > $O/combo.log 2>&1
```

읽어야 하는 표(각 arm 1행):

| 열 | 출처 |
|---|---|
| `outers` | `[RASBERY][TRAJECTORY].outers` — **채택 기준의 분모** |
| `trials` / `search_outers` | `[RASBERY][SEARCH_POLICY]` |
| `probe/carry/extrap/secant/bisect` | 같은 줄 — §2.4의 폐기 조건이 여기서 판정된다 |
| `staged_relapses` | `[RASBERY][SPTELEM][SUMMARY]` — D3 전용 감시 |
| `exit != 1` 상태점 수 | `[RASBERY][SPTELEM]`의 `search.exit` — D5 전용 감시 |
| `x_first` / `dx_last` | 같은 곳 — WARM_BORON이 실제로 출발점을 옮겼는지 |

### 5.4 Gate A / Gate B

```bash
# Gate A: 기준선 대비 지표 델타 (N1이므로 digest는 움직여도 된다)
python3 tools/gate_a_compare.py $O/base.h5 $O/<arm>.h5 --per-step
python3 tools/gate_a_compare.py $O/base.h5 $O/base_r2.h5      # 결정론: 전부 0

# Gate B: MASTER 대비 (판정의 본체)
python3 tools/compare_master_rasbery.py <MAS_SUM> $O/base.h5   -o $O/master_base
python3 tools/compare_master_rasbery.py <MAS_SUM> $O/<arm>.h5  -o $O/master_arm
# BOC 핀은 기존 핀 스크립트 + RASBERY_PPR_MODE=master 실행으로 별도 산출
```

**v2 envelope (A2_OUTER_REDUCTION §6.5와 동일한 표를 그대로 쓴다)**

| 지표 | v2 기준 | 합격 |
|---|---|---|
| 반응도 max | 1.905 pcm | ≤ ~2 pcm |
| CBC max | 15.309 ppm | ≤ ~15.3 ppm |
| AO | 0.013 | ≤ ~0.013 |
| BOC 핀 RMS / max | 0.24 % / 0.78 % | ≤ 0.24 % / 0.8 % |

### 5.5 wall (telemetry 끈 별도 실행, hot median of 3)

```bash
env $ARM $BIN --rasi kngr_238.json --raso $O/warm.h5 > /dev/null 2>&1   # 버린다
for i in 1 2 3; do
  env $ARM <ARM_KNOBS> /usr/bin/time -f "%e" -o $O/w$i.wall \
    $BIN --rasi kngr_238.json --raso $O/w$i.h5 > $O/w$i.log 2>&1
done
sort -n $O/w?.wall | sed -n 2p     # median
```

기준선과 후보를 **교대로** 돌린다. 3 % 이내 차이는 노이즈다(계획 §6.4).

### 5.6 채택 바

**전체 `outers`가 10 % 이상 줄고, Gate B가 v2 envelope 안일 것.** 둘 중 하나라도
미달이면 채택하지 않는다. 그리고 후보별 즉시 폐기 조건(§2.4)은 채택 바와 **별개로**
먼저 적용된다:

* D1: `trials`가 base 대비 증가하면 폐기 (외삽이 첫 trial을 멀게 만든 것)
* D3: `staged_relapses`가 `trials`와 같은 자릿수면 폐기 (loose 단계가 다른 근을 보고 있다)
* D5: `exit != 1` 상태점이 늘어난 만큼 GA 적합도에서 그 후보는 elite 불가 —
  **채택하더라도 acceptance는 strict 재실행으로만 한다**

채택되면 `docs/V3_FREEZE_*`에 arm을 추가하고 `tools/run_single_gpu_batch.py`의
`DEFAULT_ENV`에 넣는다. 그 전까지는 **기본 OFF가 유일한 정답**이다.

---

## 6. 다음에 해야 할 것

1. §5.0의 B0. feature-off drift가 살아 있으면 이 문서의 나머지는 의미가 없다.
2. §5.1의 분포. **그 출력을 보기 전에 §5.2의 다섯 arm을 전부 돌리지 말 것** — 분포가
   셋을 배제해 준다.
3. D2의 나머지 절반(동일 family accepted case의 bracket 시딩)은 GA 컨트롤러 과제로
   남아 있다. solver 쪽 채널은 `warm_start_from` + `SearchCarry::warm_boron`으로 이미
   열려 있으므로, 컨트롤러가 부모를 고르는 규칙만 정하면 된다.
4. WP9 단계 B/C(CRAM·TH)의 존폐는 여전히 §5.1의 `floor` 줄이 정한다 — 이 문서는 단계 D만
   다룬다.
