# WP10.3 / WP11 — 케이스가 선언한 충실도, 그리고 0이어야만 하는 영수증들 (2026-08-31)

문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 계획 | `GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` WP10.3, WP11, §6.2, §9 |
| 선행 | `WP8_EVALUATOR_STAGE1/2`(프로세스 수명), `WP9_WP10_FLOOR_RECEIPTS_WARMSTART`(§`case_key`, warm-start) |
| 커밋 | `63f4cfb`(WP10.3 hooks) · `bb5f11c`(WP11 soak harness) · 본 문서 커밋(promotion gate + 문서) |
| 로컬에서 한 것 | 순수 파이썬 계약 테스트, 그리고 **신규 헤더 2개만으로 구성한 TU의 MSVC `/Zs`**. `src/IO.cpp`·`src/main.cpp`는 로컬에 HDF5 헤더가 없어 문법 검사조차 불가 — **238 빌드가 이 둘의 첫 컴파일이다** |
| 238에서만 가능한 것 | §5 전부(빌드·B0 확인·soak 20×64·게이트 블록 생산) |
| 성능 수치 | **없다.** 이 문서에 처리량 숫자는 하나도 없고, 있어서는 안 된다 |

---

## 0. 한 줄 결론

**충실도를 프로세스의 성질로 만들고 있던 것은 아레나가 아니라 여섯 개의
`static const` 읽기였다.** 그것들을 케이스가 소유하는 값으로 한 단계 밖으로
옮기자(`src/CaseFidelity.h`), 하나의 evaluator 파동이 screening 개체군과 승격된
elite를 동시에 답할 수 있게 되었고, **각 케이스의 영수증이 자기 자신의 충실도를
말한다**. 그리고 WP11의 soak는 새로운 것을 측정하지 않는다 — 이미 이 트리가 세고
있는 계수기들이 할 말이 생길 만큼 오래 돌린 다음, **그것들을 읽는다**.

---

## 1. WP10.3 — 무엇이 충실도를 프로세스에 묶고 있었는가

### 1.1 여섯 개의 정적 읽기

WP8 1단계는 케이스별 충실도를 **거절**했고, 그 이유를 정직하게 적어 두었다:
"`PhysicsFidelity`는 프로세스당 한 번 환경에서 해결되고 캐시된다"
(`src/EvaluatorServer.h`). 그 서술은 코드에 대해서는 옳았고 물리에 대해서는 틀렸다.
실제로 묶고 있던 것은 다음 여섯 개뿐이다.

| 위치 | 무엇 | WP10.3 이후 |
|---|---|---|
| `RunContract.h` `detectedPhysicsFidelity()` | 함수 지역 static | **그대로.** 프로세스 영수증용 |
| `RunContract.h` `effectivePhysicsFidelity()` | 함수 지역 static | **그대로.** `[PHYSICS_MODE]`는 프로세스 기본값을 말한다 |
| `Driver.h` SolveLoop `staged_flux_mult` | `static const double` 람다 | **`ctx.fidelity.staged_flux_mult`** |
| `Driver.h` SolveLoop `staged_xe_mult` | `static const double` 람다 | **`ctx.fidelity.staged_xe_mult`** |
| `Driver.h` 정착 게이트 `staged_loose_settle` | `static const bool` 람다 | **`ctx.fidelity.loose_settle`** |
| (없음) — L3coarse는 **덱 성질**이라 탐지 불가였다 | — | **`--statepoint-grid` / 요청 필드**, `IO::ReadInput`에서 적용 |

**한 번 실행에서는 이것이 캐시다. evaluator에서는 첫 케이스가 자기 수렴 정책을 그
뒤 모든 케이스에게 넘기는 일이고, 그 63장의 영수증은 각각 다른 말을 한다.** 아레나는
이 넷 중 어느 것에도 의존하지 않는다 — 허용오차는 루프가 *언제* 멈추는지를 바꾸고,
격자는 덱이 *무엇을 말하는지*를 바꾸며, 둘 다 무엇도 재할당하지 않는다. 그래서 이것은
1단계와 호환되는 hook이고 `batch_width`는 여전히 아니다.

### 1.2 프로세스에 남는 두 가지 — 그리고 그것은 **바닥(floor)**이다

`RASBERY_GA_FEEDBACK_PASSES`와 `RASBERY_PHYSICS_FIDELITY`는 프로세스 수준으로
남는다. 전자는 `main.cpp`가 light 결과 없이는 거절하는 GA screening lane 전체가
키로 쓰는 값이고, 후자는 구조상 **거칠게만** 만들 수 있다. 두 값은
`processFidelityFloor()`로 합쳐지고, 케이스의 자체 설정과 **더 거친 쪽**으로 결합된다.
즉 **어떤 요청도 바닥 위로 올라갈 수 없다.**

### 1.3 무엇을 거절하고 무엇을 허용하는가 — 그리고 왜 비대칭인가

```
fidelity: strict     → staged 승수를 1.0으로 CLEAR 한다
fidelity: A2         → 승수를 발명하지 않는다. 거절한다
fidelity: L3coarse   → statepoint_grid 가 없으면 거절한다
(선언 없음)          → 프로세스 기본값. WP10.3 이전과 바이트 동일
```

`strict`가 지우는 것을 허용하는 이유는 **그것이 승격 lane 자체이기 때문**이다.
`run_single_gpu_batch.DEFAULT_ENV`가 A2 arm이므로 실제 캠페인 프로세스는 staged
허용오차를 환경에 갖고 있고, 승격된 elite를 strict로 재실행하려고 두 번째 프로세스를
세우면 CUDA 컨텍스트·라이브러리·따뜻한 cohort를 전부 버린다 — 즉 WP8이 존재하는 이유
전체를 버린다. 한 케이스를 위해 승수 두 개를 지우는 비용은 0이고 영수증이 정확히
무슨 일이 있었는지 말한다.

반대 방향은 대칭이 아니다. `strict`는 **점**(덱이 명시한 production 허용오차)이지만
A2는 **족(族)**이다. 50/1000은 측정된 arm이고 5/10은 다른 arm이며, 어느 쪽인지
말하지 않는 `A2` 영수증은 재현 불가능한 숫자다. 그래서 A2는 프로세스 환경에서
승수를 찾거나 요청이 `staged_flux_tol` / `staged_xe_tol`로 이름을 대야 한다.

### 1.4 마지막 검사는 **양방향 등식**이다

`resolveCaseFidelity()`는 설정을 만든 뒤 그 설정이 실제로 만들어내는 충실도를 계산하고,
요청이 쓴 단어와 **같지 않으면 거절한다**.

* 선언보다 **거칠게** 실행 → 승인 표에 걸어 들어간 근사. WP1 계약이 존재하는 이유.
* 선언보다 **곱게** 실행 → 다른 정책 아래 측정된 숫자를 한 칸에 넣는 것.
  계획 §6.2가 금지하는 혼합이며, "더 좋은 쪽이니 괜찮다"가 **아니다**.

### 1.5 L3coarse — 덱 성질을, 덱 성질로

`tools/make_screening_deck.py`는 두 번째 파일을 쓴다. 그것은 디렉터리를 소유하는
launcher에게는 되고, 요청 스트림에 답하는 evaluator에게는 안 된다. 그래서 같은 변환이
`src/StatepointGrid.h`에 있고, **JSON 파싱과 덱 키 다이제스트 사이**에서 돈다.

```
input_stream >> config;
applyGridSpec(config, statepoint_grid);   // ← 여기
_deck_key_digest = Sha256::hexOf(deckPayload(config));
```

**뒤였다면 한 후보의 10-상태점 screening 답과 35-상태점 승인 답이 같은 `case_key`를
갖는다.** 그 키로 캐시하면 근사가 승인 요청에 조용히 서빙된다 — 이 트리의 exact-only
계약이 막으려는 바로 그 실패를, 새 문으로 통과시키는 것이다.

`tools/test_case_fidelity_contract.py`가 C++ 상수(`kCoarseBurnups`,
`kThreeBurnups`, `kWarnBurnupStepGwd`, `kDepletionTimeKeys`)를 파이썬 도구의
것과 대조한다. **C++의 `coarse`가 도구의 `coarse`가 아니면 screening lane 정의가
두 개가 되고, 두 lane의 숫자는 비교할 수 없다.**

### 1.6 케이스 키가 환경이 아니라 케이스를 접는다

`trajectory::kArmEnv`는 `getenv()`로 순회된다. 한 프로세스가 strict 케이스와 A2
케이스를 연달아 돌릴 수 있게 된 순간, 둘 다 **같은** `RASBERY_STAGED_FLUX_TOL`
문자열을 키에 접는다 — 즉 승격된 strict 재실행이 그것이 대체하려는 screening 결과와
**충돌한다**. 그래서 `Driver::armEnvValue()`가 세 knob만 `CaseFidelity`에서 철자한다.

**아무것도 덮어쓰지 않았을 때는 원시 환경 문자열을 그대로 쓴다.** 키에는 두 번째
구현(`tools/case_key.py`)이 있고 그쪽은 이 셋을 원시 텍스트로 읽으며
`tools/test_case_key_contract.py`가 둘을 한 고정치에 묶는다. 이쪽만 `50.0`을 `50`으로
정규화했다면 아무도 바꾸지 않은 덱에서 둘이 갈라섰을 것이다.

### 1.7 `promote` op

```json
{"op":"promote","key":"e0007","deck":"...","output":"...","promoted_from":"<screen case_key>"}
```

`strict` / `full` 출력 / `full` 격자가 **기억해야 하는 것이 아니라 기본값**이 되도록
별도 op이다. 셋을 매번 적어야 하는 운영자는 언젠가 적지 않고, 잊은 그 실행이 elite의
이름을 단 screening 답이 된다. `promoted_from`은 **필수**다 — 링크가 없으면 승격은
무관한 두 행이고 "이 elite가 실제로 재실행됐는가"에 영수증만으로 답할 수 없다.

### 1.8 감사(audit)가 충실도를 따라 내려간다

프로세스는 `[PHYSICS_MODE]` 한 줄을 찍고 혼합 파동에는 64개의 답이 있다. 그 한 줄은
이제 **케이스들이 무엇에 대해 해결되었는지의 기본값**을 서술할 뿐 그중 어느 것도
서술하지 않는다. `exact_audit.audit_case_fidelity()`가 케이스별 영수증을 읽고 각각을
**그 케이스 자신의 선언**에 대해 판정한다. 선언이 없는 케이스는 통과가 아니라
보고된다 — 아무도 선언하지 않은 케이스는 출처가 추측인 결과다.

---

## 2. 영수증 스키마(추가분)

| 영수증 | 신규 필드 |
|---|---|
| `[RASBERY][CASE]` | `schema_version:2`, `statepoint_grid`, `acceptance_eligible`, `fidelity_declared`, `promoted_from`. `fidelity`/`policy`는 이제 **케이스의 것** |
| `[RASBERY][EVALUATOR][CASE]` | `policy`, `physics_fidelity`, `statepoint_grid`, `acceptance_eligible`, `fidelity_declared`, `promoted_from`. **영수증을 접지 못한 케이스는 기본값이 아니라 `null`** |
| `[RASBERY][EVALUATOR][READY]` | `fidelity_per_case:true`, `fidelity_default`, `fidelity_floor`, `statepoint_grid_default` |
| `[RASBERY][EVALUATOR]` (프로세스) | `fidelity_overrides`, `promotions` |
| `[RASBERY][PHYSICS_MODE]` | `statepoint_grid` |
| light JSONL | `policy`, `physics_fidelity`, `statepoint_grid`, `acceptance_eligible`, `fidelity_declared`, `promoted_from` |

**모든 실행이 이 필드들을 찍는다.** 일부 케이스만 갖는 필드는 어떤 감사도 *요구*할 수
없는 필드이고, 요구할 수 없는 감사는 "strict로 돌았다"와 "말하기를 잊은 빌드"를
구분하지 못한다.

---

## 3. WP11 — soak

### 3.1 무엇을 측정하지 않는가

soak는 새 숫자를 만들지 않는다. WP8–WP10이 넣을 수 있는 결함은 파동 모양이 아니다:
두 번 나눠준 슬롯, Driver보다 오래 산 pinned 임대, 풀리면서 할당을 남긴 graph
capture, 후보가 바꾸는 것을 덮기 시작한 cohort 키. 각각은 **한 세대 동안 보이지 않고
스무 세대에 치명적**이며, 각각은 이미 이 트리가 찍는 영수증이 세고 있다.

### 3.2 0이어야 하는 표 — 그리고 0이 아닌 두 가지

`tools/soak_run.py`의 `ZERO_RECEIPTS`가 계획의 목록을 **하나의 표**로 들고 있고,
보고서는 **0인 것까지 전부** 인쇄한다. "확인했고 0이었다"와 "확인하지 않았다"는 다른
진술이고, 실패만 보여주는 보고서는 둘을 구분할 수 없다. **영수증이 아예 찍히지 않은
계수기는 0이 아니라 문제로 보고된다.**

조건부 두 가지를 그렇게 말하는 것이 요점이다.

* `*_fallbacks`는 `RASBERY_GPU_FULL=1`에서만 0 주장이다. 게이트 없이 host fallback은
  바이너리가 해도 되는 일이고, 그것으로 실패시키면 **평소 돌리는 arm에서 soak가
  쓸모없어진다**.
* `restarts`는 `--expect-restarts`(기본 **0**)로 묶인다. 이 soak가 심는 독은
  **파싱되지 않는 덱**이고, 그것은 `runOneCase`의 try 안에서 한 케이스만 실패시키고
  프로세스를 살려 두어야 한다. 거기서의 재시작은 독이 작동한 것이 아니라 **독이
  다른 63개 후보를 함께 데려간 것**이다.

### 3.3 누수는 수준이 아니라 **기울기**이고, **후반부**에서 잰다

앞쪽 세대는 오른다 — 라이브러리, 아레나, graph 캐시. **그 상승은 설계가 작동하는
것이다.** 전 구간 회귀는 건강한 soak마다 그것을 누수라고 부르고, 그러면 임계값을 아무것도
못 잡을 만큼 높여야 한다. `leak_slope_mb_per_generation()`은 후반부만 적합시킨다.
`nvidia-smi`나 `/proc`이 없으면 계열은 `0.0`이 아니라 `None`이다 — 없는 측정으로
만든 평평한 궤적은 **아무것도 증명하지 않는 이유로** 누수 검사를 통과한다.

### 3.4 작업부하

| 요소 | 왜 |
|---|---|
| light/full 혼합 | 둘 다 쓰고, 하나만 HDF5를 쓴다 |
| 세대 간 warm-start 사슬 | 사슬이라야 "쓰였지만 닫히지 않은" warm state 파일이 나온다 |
| screening lane(L3coarse + coarse 격자) | 한 충실도짜리 soak는 한 파동 안에서 두 케이스가 다르게 수렴하는 경로를 **한 번도** 지나지 않는다 |
| 세대당 승격 1건 | `promoted_from` 링크가 실제로 영수증에 남는지 |
| 세대당 독 케이스 1건 | 실패 격리. 파동 영수증이 63/64여야 한다 |

`tools/test_soak_run.py`가 이 전부를 fake child로 몰아본다: **각 0-영수증을 차례로
0이 아니게 만들고**, arbiter 줄을 지우고, 게이트 안팎에서 fallback을 내고, 프로세스를
죽이는 독을 심고, **충실도 계약을 적용하는 대신 흉내만 내는 바이너리**를 시늉해서,
매번 FAIL이 나오고 이유를 이름으로 부르는지 확인한다.

---

## 4. 기본값 승격 규칙 — 코드로

`tools/promotion_gate.py`. `tools/test_ga_promotion_gate.py`(GA 2단계 40× 회계 주장
감사)와 **다른 것**이다.

성능 바는 **feature마다 다르고 그것이 계획 §9의 표다.** 단일 전역 +5%는 싼 레버를
들여보내고 비싼 레버를 막는다 — 거꾸로다.

| 판정 | 언제 |
|---|---|
| `DEFAULT_ON` | B0 + §9 바(부수 조건 포함) + soak 통과 + env rollback 명시 |
| `MODE_DEPENDENT` | 위 전부 + `mode`가 `single`/`batch` 한쪽 — 한쪽 모드 숫자로 전역 승격하는 대신 **어느 쪽인지 말하게** 한다 |
| `HOLD` | 무엇 하나라도 미달·미진술. **N1은 accuracy envelope·결정적 digest에 더해 기록된 `project_acceptance`가 없으면 여기 머문다 — 통과한 게이트는 승인이 아니다** |
| `NEVER_DEFAULT` | A2 / L3coarse. 무엇을 측정했든 mode-specific이고 strict 기본값을 대체하지 않는다 |
| `REJECT` | 등급이 §6.1의 넷 중 하나가 아님 |

**이 스크립트는 기본값을 뒤집지 않는다.** 읽고, 판정하고, 이유를 인쇄한다. 종료 코드는
**읽을 수 없는 블록에서만** 0이 아니다 — "아무것도 자격을 얻지 못했다"는 대부분의 밤에
정상적인 답이고, 그것을 처벌하는 종료 코드는 모두에게 종료 코드를 무시하도록 가르친다.

블록의 A/B 위생도 본다: 두 arm이 다른 충실도로 돌았다면 그 비교는 flag가 아니라
**정책을 잰 것**이고, 후보가 케이스를 잃었다면 그 처리량 숫자는 **다른 파동의 숫자**다.

---

## 5. 238 runbook

### 5.0 전제

```bash
cd ~/rasbery_gpu && git fetch && git checkout codex/exact-throughput-campaign && git pull
# 첫 컴파일: src/IO.cpp, src/main.cpp, src/EvaluatorServer.h 는 로컬 문법 검사조차 못 했다
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
export CUDA_VISIBLE_DEVICES=0
export RASBERY_CODE_SHA=$(git rev-parse --short HEAD)
```

### 5.1 B0 확인 — **soak 전에**

WP10.3은 feature-off가 이전과 바이트 동일해야 한다. 아무것도 선언하지 않은 케이스는
프로세스 기본값을 물려받고, 그것이 예전 경로다.

```bash
# 1) 단일 실행 digest 불변  (기준: 이전 tip 4a477b4 의 같은 덱)
./build/RASBERY --rasi kngr_238.json --raso /tmp/b0_new.h5 2>&1 | tee /tmp/b0_new.log
grep -E "\[RASBERY\]\[TRAJECTORY\]|\[RASBERY\]\[CASE\]" /tmp/b0_new.log
#    → digest 가 4a477b4 의 것과 같아야 한다
#    → [CASE].case_key 도 같아야 한다  (armEnvValue 가 원시 문자열을 유지하므로)

# 2) case_key 이중 구현 일치
python tools/case_key.py --deck kngr_238.json --compare /tmp/b0_new.log

# 3) A2 arm 에서도 같은 두 가지
RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1 \
  ./build/RASBERY --rasi kngr_238.json --raso /tmp/b0_a2.h5 2>&1 | tee /tmp/b0_a2.log
```

**여기서 digest 또는 case_key 가 움직이면 멈춘다.** WP10.3은 B0이고, 움직였다면
`Driver::armEnvValue`의 "덮어쓰지 않았으면 원시 문자열" 규칙이 깨진 것이다.

### 5.2 케이스별 충실도가 실제로 다르게 수렴하는지

```bash
# 같은 프로세스, 한 파동, 두 lane
cat > /tmp/mixed.jsonl <<'EOF'
{"op":"case","key":"screen0","deck":"kngr_238.json","output":"/tmp/s0.h5","result_mode":"light","fidelity":"L3coarse","statepoint_grid":"coarse"}
{"op":"case","key":"elite0","deck":"kngr_238.json","output":"/tmp/e0.h5","result_mode":"full","fidelity":"strict"}
{"op":"wave","wave_id":1}
{"op":"shutdown"}
EOF
RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 \
  ./build/RASBERY --evaluator-jsonl /tmp/mixed.jsonl --batch-mode 2 2>&1 | tee /tmp/mixed.log
```

확인할 것:

1. `[EVALUATOR][CASE]` 두 줄의 `policy`가 각각 `L3coarse`, `strict`.
2. `screen0`의 `statepoints`가 `elite0`의 것보다 **작다**(격자가 실제로 적용됐다).
3. 두 `case_key`가 **다르다**(격자가 다이제스트 앞에서 돌았다).
4. `elite0`가 A2 환경 안에서 `strict`로 돈다(`staged_relapses == 0`).
5. `python tools/exact_audit.py` 대신:
   `python -c "import sys;sys.path.insert(0,'tools');import exact_audit as a;print(a.audit_case_fidelity(open('/tmp/mixed.log').read(),{'screen0':'L3coarse','elite0':'strict'}))"`
   → `[]`

**부정 통제(반드시 거절되어야 한다):**

```bash
# A2 를 선언하는데 승수가 없다  → REFUSED
echo '{"op":"case","key":"x","deck":"kngr_238.json","output":"/tmp/x.h5","fidelity":"A2"}
{"op":"wave","wave_id":1}
{"op":"shutdown"}' | ./build/RASBERY --evaluator-jsonl - --batch-mode 1
# L3coarse 를 선언하는데 격자가 없다  → REFUSED
# promote 인데 promoted_from 이 없다  → REFUSED
```

### 5.3 승격 왕복

```bash
cat > /tmp/promote.jsonl <<'EOF'
{"op":"case","key":"s","deck":"kngr_238.json","output":"/tmp/ps.h5","result_mode":"light","fidelity":"L3coarse","statepoint_grid":"coarse","save_warm_state":"/tmp/ps.warm"}
{"op":"wave","wave_id":1}
{"op":"promote","key":"p","deck":"kngr_238.json","output":"/tmp/pp.h5","promoted_from":"<위 s 의 case_key>","warm_start_from":"/tmp/ps.warm"}
{"op":"wave","wave_id":2}
{"op":"shutdown"}
EOF
```

`p`의 영수증이 `policy:"strict"`, `statepoint_grid:"full"`,
`promoted_from:"<s의 case_key>"`, `acceptance_eligible:true`여야 한다.
`[EVALUATOR].promotions == 1`.

### 5.4 soak — 20 × 64, 8×M8 + MPS

```bash
export RASBERY_GPU=1 RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 \
       RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 \
       RASBERY_GPU_FLATXS=1 RASBERY_GPU_WIEL_FOLD=chunked RASBERY_GPU_XE=1 \
       RASBERY_IO_WRITER=thread
export RASBERY_GPU_FULL=1          # fallback 0 주장을 켠다

python tools/soak_run.py \
    --deck kngr_238.json \
    --workdir /scratch/soak_$(date +%Y%m%d) \
    --binary ./build/RASBERY \
    --generations 20 --width 64 \
    --light-fraction 0.5 --screen-fraction 0.25 \
    --drift 0.03 --vram-leak-mb 8 --rss-leak-mb 8 \
    --report /scratch/soak_$(date +%Y%m%d)/soak_report.json
```

**예상 벽시계 ~1.5 h.** 근거: `d7b81af` 시점 배치 M64가 518 c/h(GPU Xe + A2,
`gpu-dispatch-branch-verdict`). 1,280 케이스 + 세대당 독 1 + 승격 1 = 1,320 케이스이고
`1320 / 518 ≈ 2.55 h`가 **단일 M64 환산**이다. 8×M8 + MPS는 그 수를 위로 옮기지만
**얼마나인지는 측정되지 않았다** — 그래서 위 명령은 8×M8 배치가 아니라 **하나의
evaluator 세션**을 몬다(`--width 64`). 8×M8을 원하면 `tools/run_multi_gpu_batch.py`의
튜너로 폭을 정하고 각 worker에 soak를 하나씩 붙이되, **그때 VRAM 계열은 GPU 전체를
재므로 프로세스 8개의 합**이라는 것을 보고서에 적어야 한다.

`--screen-fraction 0.25`의 coarse lane은 케이스를 싸게 만든다. 따라서 **c/h drift 3 %
예산은 세대 간 구성이 동일하다는 전제**에 의존하고, soak는 그렇게 만든다(매 세대 같은
비율). 구성이 다른 두 soak의 c/h를 비교하지 말 것.

#### 종료 시 0이어야 하는 목록(보고서 표와 동일)

```
duplicates                      slot_duplicates          [EVALUATOR]
stale_tenants                   slot_stale_tenants       [EVALUATOR]
double_releases                 slot_double_releases     [EVALUATOR]
refill_duplicates               duplicates               [REFILL]
refill_stale_tenants            stale_tenants            [REFILL]
refill_double_releases          double_releases          [REFILL]
alloc_in_capture                alloc_in_capture         [CUDA][CAPTURE_ARBITER]
captures_unwound                captures_unwound         [CUDA][CAPTURE_ARBITER]
cross_case_digest_mismatch      isolation_mismatches     [EVALUATOR]
pin_live_ranges_between_waves   (동명)                    [EVALUATOR]
cmfd/nodal/xsrecon/flatxs/xe/ppr/cram/graph _fallbacks   [GPU_FULL]  ← GPU_FULL=1 에서만
restarts <= --expect-restarts (기본 0)
output collisions               soak_run 자체 집계
```

### 5.5 게이트 블록과 승격 판정

238 runner가 레버당 블록 하나를 쓴다(스키마는 §4와 `tools/promotion_gate.py`
docstring). 그 다음:

```bash
python tools/promotion_gate.py /scratch/gate/*.json --json /scratch/promotion.json
```

**출력은 제안이다. 기본값을 바꾸는 것은 그 블록을 인용하는 별도의 리뷰된 커밋이다.**

#### 승격 표 템플릿 (결과를 여기에 채운다)

| 레버 | flag | 등급 | tip | 기준(§9) | 측정 gain | 부수 조건 | soak | env rollback | 판정 | 근거 블록 |
|---|---|---|---|---|---:|---|---|---|---|---|
| xslib_cache | `RASBERY_XSLIB_CACHE` | B0 | | M64 +3% | | — | | | | |
| cmfd_compaction | `RASBERY_GPU_CMFD_COMPACT` | B0 | | M64 +5% | | padding ≥30% | | | | |
| persistent_evaluator | `--evaluator-jsonl` | B0 | | chunked +5% | | process cost <1% | | | | |
| conditional_while | | B0 | | 단일 +5% | | — | | | | |
| k_process | | B0 | | K2 +5% | | — | | | | |
| flatxs_cooperative | `RASBERY_GPU_FLATXS_CTA` | B0 | | M64 +10% | | kernel −30% | | | | |
| xe_transaction | `RASBERY_GPU_XE_TXN` | N1 | | 전체 +5% | | — | | | | project acceptance 필요 |
| warm_start | `--warm-start-from` | N1 | | 대상 outer −20% | | — | | | | project acceptance 필요 |
| staged_tolerance | `RASBERY_STAGED_*` | A2 | | — | | — | | | `NEVER_DEFAULT` | mode-specific |
| statepoint_grid | `--statepoint-grid` | L3coarse | | — | | — | | | `NEVER_DEFAULT` | screening 전용 |

---

## 6. 아직 하지 않은 것 / 하지 말아야 할 것

1. **기본값은 하나도 바뀌지 않았다.** `promotion_gate.py`는 판정만 인쇄한다.
2. **`RASBERY_GA_FEEDBACK_PASSES`는 여전히 프로세스 단위다.** 케이스별로 만들려면
   `BatchLightResult::FeedbackPasses()`의 소비자 전부를 감사해야 하고, 그것은
   screening lane의 정의를 건드리는 별도 작업이다. 지금은 **바닥**으로 남는다.
3. **coarse 격자의 비용 모델을 믿지 말 것.** 비용은 burnup step에 **초선형**이고,
   측정된 3-상태점 격자는 35-상태점 덱보다 **더 많은** outer(5,104 대 4,609)를 돌았다.
   `--statepoint-grid`는 4 GWd/t를 넘는 step에 경고를 찍지만 막지는 않는다.
4. **8×M8 + MPS에서의 soak 벽시계는 추정이다.** §5.4가 어떤 수를 어떻게 얻었는지
   적어 두었으니, 실측 후 그 문단을 교체할 것.
5. **`[PHYSICS_MODE]`만 읽는 감사는 이제 불충분하다.** 혼합 파동에서 그 한 줄은
   기본값을 말한다. `audit_case_fidelity`를 함께 쓰지 않는 파이프라인이 남아 있으면
   그것은 WP10.3 이전의 감사다.
