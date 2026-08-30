# WP10.5 — 귀속기가 지목한 것은 576바이트였고, 82건은 영수증에 케이스 이름이 없어서였다

호스트 181, `55c0dce`, 2026-08-30. 보고서 `E:\rasbery_runs\20260830\181\gates_8919331.md` §(14).

WP10.4의 두 주장은 **확인됐다**: `physics_fidelity` 누락 노트 **0건**,
`[RASBERY][CASE]`가 `schema_version:5`에 두 이름 모두 인쇄, `live_cases:0` 전 웨이브,
zero-receipt 16종 전부 0. 그리고 새로 만든 MEM 영수증/귀속 기구가 **작동했다**.

그런데 그것이 지목한 것이 틀렸고, 감사는 새로운 이유로 82건을 거부했다. 둘 다 이번 커밋에서 고친다.

---

## 1. `case_samples`는 범인이 아니었다 — 귀속기가 **무게를 재지 않았다**

### 181이 본 것

```
wave 1: rss 1263.301  case_samples 18
wave 2: rss 4251.379  case_samples 36   (rss_delta +2988.078)
wave 3: rss 4501.695  case_samples 54   (+250.316)
wave 4: rss 4581.957  case_samples 72   (+80.262)
wave 5: rss 4556.508  case_samples 90   (-25.449)
```

귀속기의 출력: `cache_entries.case_samples 18 -> 72`.
**그게 유일하게 움직인 카운터였기 때문에** 그렇게 말한 것이다.

`case_samples`는 `Summary::case_seconds`/`teardown_ms`의 원소 수다.
90개 × 2계열 × 8바이트 = **1,440바이트**. 구간 증가분은 **576바이트**.
같은 구간 RSS 증가분은 **3,293 MB**. 비율 1 : 6,000,000.

**즉 이 기구의 결함은 "무엇이 움직였나"만 말하고 "그게 그 크기를 설명할 수 있나"를
말하지 않은 것이다.** 귀속을 믿은 독자는 하루를 잃는다. 귀속이 없는 것보다 나쁘다.

### RSS 곡선 자체의 해석 — 그리고 기울기 규약의 결함

`+2988 → +250 → +80 → −25`. 첫 세대가 아레나·디바이스 라이브러리·그래프 캐시를 세우는
**1회성 계단**이고, 그 뒤는 거의 평평하며 마지막은 **내려간다**.

`leak_slope_mb_per_generation`은 "second half"를 쓰지만, 5세대에서 second half는
인덱스 2·3·4이고 기울기 **+27.4 MB/gen**이 나온다 — 계단은 이미 빠져 있는데도
예산 8.0을 넘는다. 그리고 이 규칙은 소크가 길어질수록 의미가 흐려진다(50세대면
second half는 25세대부터 시작한다).

**규약을 이름으로 바꿨다**: 워밍업 세대 수를 명시적으로 버린다(`--warmup-generations`,
기본 1). 181 데이터에서:

| 기울기 | 값 | 용도 |
|---|---|---|
| raw (전 세대) | **+691.70** MB/gen | 계단 포함, 참고용 |
| post-warm-up (1세대 제외, 전체 적합) | **+99.56** MB/gen | **게이트** |
| post-warm-up의 second half | −25.45 MB/gen | 2점 적합, 참고용 |
| 전체의 second half (구 게이트) | +27.41 MB/gen | 구 규약, 참고용 |

게이트를 **post-warm-up 전체 적합**으로 잡은 이유: 계단을 이름으로 뺀 뒤에는 반으로
자르는 것이 표본만 버린다. 5세대에서 post-warm-up second half는 **2점**이고, 2점으로
켜지는 게이트는 동전이다 — 실제로 그 값은 −25.45(PASS)인데 그 3세대 동안 프로세스는
305 MB를 얻었다. +99.56 MB/gen, 정직한 **FAIL**이다. 네 숫자 모두 리포트에 남는다.

### 수정

| 무엇 | 어디 | 조치 |
|---|---|---|
| `case_samples` 무한 증가 | `src/EvaluatorServer.h` `Summary` | `SampleWindow`(고정 용량 링, `RASBERY_EVALUATOR_SAMPLE_WINDOW`, 기본 **4096**). 링이 파괴하는 두 사실은 따로 보존: `observed()`는 **전 케이스** 개수, `max()`는 **전 케이스** 정확 최대. 4096 doubles = 32 kB/계열 — 다시는 MEM 영수증에서 가장 큰 숫자가 될 수 없다 |
| 저수지 표본이 아닌 이유 | 같은 곳 | 난수를 쓰면 같은 덱 두 실행의 p90이 달라진다. 이 트리의 게이트 구조 전체가 bit-identity다. 링은 결정적이고, 포기하는 것(아주 긴 실행의 앞부분)은 안정성 질문이 가장 원하지 않는 부분이다 |
| 백분위의 의미가 바뀐 것을 숨기지 않음 | `[RASBERY][EVALUATOR]` | `case_seconds`/`case_teardown_ms`에 `window`(상주 표본 수)와 `observed`(전 케이스 수) 추가. `max`는 창이 아니라 전 실행 기준 |
| 귀속기가 무게를 재지 않음 | `tools/soak_run.py` | MEM 영수증이 값을 매길 수 있는 컨테이너는 **바이트 증가분을 RSS 증가분과 비교**한다(`ATTRIBUTION_SHARE = 5 %`). 미달이면 `"these moved but CANNOT be the cause -- they are too small by orders of magnitude, so do not spend a day on them"` |
| 값을 매길 수 없던 컨테이너 | `src/EvaluatorServer.h` | `cache_bytes`에 `case_samples` 추가(`cache_entries`에는 `case_samples_cap`도) |
| 아무것도 설명하지 못할 때 | `tools/soak_run.py` | 침묵하지 않는다: "nothing [EVALUATOR][MEM] can see accounts for the growth: look at the allocator (**compare `rss_peak_mb` with `rss_mb`** — a large gap is retention, not a leak), HDF5's own free lists, and the device runtime's host-side allocations next" |

### 그래서 3.3 GB는 무엇인가 — 이 커밋은 **모른다고 말한다**

`case_samples`가 아니라는 것은 이제 증명됐다(576바이트). `xslib`/`cohorts`/`digest_memo`/
`pin_records`/`cuda_host_bytes`는 5세대 내내 **완전히 평평했다**. 즉 **evaluator가 볼 수
있는 어떤 컨테이너도 아니다.** 남은 후보와 각각을 가르는 관측:

1. **할당자 보유**(가장 유력) — 곡선 모양이 그렇다: 1회성 계단 뒤 평탄, 마지막 세대는
   **감소**. 진짜 누수는 내려가지 않는다. `rss_peak_mb`(4662) vs `rss_mb`(4556)의 격차가
   이를 가른다.
2. **HDF5 자체 free list** — 케이스마다 182 MB HDF5를 쓴다.
3. **CUDA 런타임 호스트 측 할당** — 아레나는 프로세스당 1회지만 그래프 캐시는 아니다.

`--generations 12` 이상의 소크가 셋을 가른다: 할당자 보유라면 곡선은 평탄해지고 게이트는
통과한다. 그래서 런북이 12세대를 요구한다.

---

## 2. 82건 "no fidelity was declared" — 영수증에 **케이스 이름이 없었다**

### 원인

WP10.4가 Driver 태그를 감사 가능하게 만들자, 그 태그들은 **다음 검사에서** 걸렸다.
`audit_case_fidelity`는 케이스별 선언(mixed-fidelity 파동에서 각 케이스가 어떤 단어로
요청됐는지)을 영수증의 식별자로 찾는다: `("key", "case_key", "output", "deck")`.

- `[RASBERY][EVALUATOR][CASE]`는 `key`(클라이언트 라벨), `deck`, `output`을 싣는다 → 해소된다.
- `[RASBERY][CASE]`는 **`case_key` 하나만** 실었다.

그리고 `case_key`는 **정준 중복 키**다. 폭 16의 콜드 파동에서 한 덱·한 충실도의
16개 케이스는 **설계상 같은 `case_key`를 갖는다**. 즉 케이스별 감사가 존재하는 바로 그
파동에서, 그 태그는 원리적으로 케이스를 지목할 수 없었다.

`soak_run.py`가 만드는 선언 맵은 클라이언트 `key`로만 키가 잡혀 있었으므로 매칭 실패.
5세대 × (16 + promote) = 85개 Driver 줄 중 82건이 그렇게 보고됐다.

재현(도구만으로):

```
abc: ran at policy='strict' and no fidelity was declared for it.
```

### 수정 — 플래그는 필요 없다

**운영자 플래그를 요구하지 않는다.** `RASBERY_ALLOW_SCREENING`도 `--fidelity`도 아니다.
이건 선언이 없어서가 아니라 **영수증에 이름이 없어서** 생긴 일이었고, 고칠 곳은 발신자다.

1. **`src/Driver.h`** — `[RASBERY][CASE]`에 **`output`** 추가(= `--raso` 경로).
   `schema_version` 5 → **6**.
   `output`을 고른 이유: (a) evaluator 자신의 파동 네임스페이스 규칙이 **케이스당 유일**을
   보장한다, (b) argv·`--jobs` 매니페스트·evaluator **모든 모드**에 존재한다(클라이언트
   `key`는 evaluator 모드에만 있다), (c) Driver가 **이미 갖고 있다** — 아무것도 배선하지
   않는다.
2. **`tools/soak_run.py`** — `declare(word, key, output)`: 한 케이스를 **영수증이 자신을
   부를 수 있는 모든 이름**으로 선언한다. 안전한 이유: 그 이름들은 한 케이스를 가리키고,
   evaluator가 파동 안에서 중복 `--raso`를 거부한다.
3. **`tools/exact_audit.py`** — 거부문이 **어느 쪽 잘못인지 말한다**. 옛 메시지는
   "선언되지 않았다"뿐이어서 *운영자가 빠뜨린 경우*와 *발신자가 이름을 안 실은 경우*를
   구분하지 못했고, 181에서는 후자였는데 한 세션 동안 "미진단"으로 남았다. 이제:

   > `abc: ran at policy='strict' and no fidelity was declared for it. The receipt is
   > identified by [case_key='abc']; the declaration is keyed on ['g0000c0000']. ...
   > If the two lists share no field, the fault is in the EMITTER, not in the operator:
   > a receipt whose only name is 'case_key' names a duplicate class, not a case, and
   > every case of one deck at one fidelity shares it.`

4. **`tools/fake_rasbery_child.py`** — 이제 `[RASBERY][CASE]`도 인쇄한다.
   **같은 구멍이 두 번 세션을 잡아먹었다**: WP10.4에서는 fake가 그 태그를 안 찍어서
   철자 불일치가 호스트까지 갔고, WP10.5에서는 같은 침묵이 식별자 부재를 가렸다.
   하네스가 한 번도 찍지 않는 태그는 하네스가 지킬 수 없는 태그다.

### 덤으로 드러난 진짜 결함 — 출력 충돌 검사가 자기 자신을 신고했다

fake가 두 태그를 다 찍기 시작하자 `soak_run.py`의 출력 충돌 검사가 **모든 케이스를
자기 자신과 충돌**로 신고했다. 그 검사는 `key or case_key`로 신원을 접고 있었는데,
이는 **두 태그 중 하나만 `output`을 실을 때에만** 무해했다. 두 태그는 같은 케이스를
서로 다른 신원으로 부른다(클라이언트 라벨 / 정준 중복 키). 이제 **같은 종류끼리만**
비교한다: `key`는 `key`와, `case_key`는 `case_key`와. `case_key` 불일치는 더 약한
언어로 보고한다 — 중복 클래스는 그 자체로 충돌을 증명하지 못한다.

---

## 3. 테스트

| 테스트 | 결과 |
|---|---|
| `tools/test_soak_receipt_schema_contract.py` | PASS (2 emitters × 4 audited fields, 1 synonym, version refusal intact, **per-case identifier declarable both ways**) |
| `tools/test_evaluator_mem_receipt_contract.py` | PASS (8 fields, 8 sized containers, 4 bounded caches, 7 compiled, **7 compiled window**) |
| `tools/test_enum_alias_contract.py` / `test_dependent_template_contract.py` | PASS |
| `tools/test_evaluator_contract.py` / `test_case_key_contract.py` (schema 6) / `test_case_fidelity_contract.py` | PASS |
| `tools/test_soak_run.py` / `test_xslib_cache_contract.py` / `test_cohort_context_contract.py` / `test_host_pin_registry.py` / `test_promotion_gate.py` | PASS |

**컴파일 절반이 두 개다.** cohort LRU 상한(cap 2에서 축출 victim이 LRU이고 생존자가
hit임을 확인)에 더해, 이번엔 `SampleWindow`의 **출하되는 소스 텍스트를 그대로 잘라내**
독립 컴파일해서 7가지를 확인한다: 상한 준수, `observed()`가 100(창이 8인데도),
`max()`가 창 밖으로 떨어진 999.0을 유지, `bytes()`, 링이 **최근** 표본을 담는 것,
빈 창의 두 경계. 링 버퍼는 읽어서 통과시킬 코드가 아니라 돌려봐야 하는 코드다.

전체 스위트: 이 커밋으로 **새로 깨지는 테스트 0개**. 남은 18건은 동시 진행 중인
WP13/WP15 작업 트리(`CudaBICGBackend.cu`, `CudaOuterGraph.cu`, `CudaXsReconBackend.*`,
`XSSet.*`, `IoWriter.h` 등)에 대한 것이며 이 커밋 이전 트리에서도 동일하게 실패한다.

---

## 4. 181 런북 (WP10.4 대비 델타)

```bash
git fetch && git checkout <this commit> && git rev-parse HEAD
cmake --build build -j32

env -i <PROD env> RASBERY_GPU_FULL=1 \
  python3 tools/soak_run.py \
    --deck kngr_238.json \
    --workdir ~/gates/wp10_5/soak \
    --binary ./build/RASBERY \
    --generations 12 --width 16 \
    --report ~/gates/wp10_5/soak_report.json
```

**WP10.4 런북과 다른 점은 세 가지뿐이다.**

1. `--generations` **5 → 12**. 워밍업 1세대를 버리고도 11세대가 남아야 기울기가
   할당자 보유와 진짜 누수를 가른다. 5세대는 post-warm-up 4점이고, 그중 절반은 2점이다.
2. `--warmup-generations`는 **기본 1이므로 적지 않는다.** 계단이 2세대에 걸친 것으로
   보이면(웨이브 3의 +250 MB가 반복되면) `--warmup-generations 2`로 **한 번 더** 돌리고
   두 리포트를 **둘 다** 첨부할 것. 값을 바꿔 통과시키는 것이 아니라, 계단의 길이를
   측정하는 것이다.
3. 기대치가 늘었다(아래).

**기대**

| 항목 | 기대 |
|---|---|
| fidelity 관련 problems | **0건**. `physics_fidelity` 노트도, `no fidelity was declared`도 없어야 한다. 남으면 그건 실제 fidelity 위반이므로 문장 그대로 보고 |
| `[RASBERY][CASE]` | `"schema_version":6`, `fidelity`·`physics_fidelity`·**`output`** 모두 존재. `grep -m1 'RASBERY\]\[CASE\]' ~/gates/wp10_5/soak/log/soak.log` |
| `cache_entries.case_samples` | **4096에서 멈춘다**(12×18=216이므로 이번 소크에서는 216에서 평평하지 않고 계속 오른다 — 상한에 닿지 않는다. 이것이 정상이다). `case_samples_cap:4096`이 함께 찍혀야 한다 |
| `cache_bytes.case_samples` | 존재하고 **3456 바이트 수준**. 이 숫자가 리포트에 있으면 다시는 이것이 GB를 설명한다고 말할 수 없다 |
| `live_cases` | **매 줄 0.** 0이 아닌 줄이 하나라도 있으면 다른 모든 것보다 먼저 보고 |
| `growth.rss` | 네 기울기가 전부 있어야 한다: `slope_mb_per_generation`(게이트, post-warm-up), `slope_raw_mb_per_generation`, `slope_post_warmup_second_half_mb_per_generation`, `slope_second_half_all_mb_per_generation` + `warmup_generations:1` |
| 게이트 | `slope_mb_per_generation ≤ 8.0`이면 **PASS이고 3.3 GB는 워밍업 계단이었다는 뜻**이다 — 그것이 이번 소크의 실제 질문이다 |
| 초과 시 | problems에 `"CANNOT be the cause"` 또는 `"growing by enough to matter"` 중 하나가 반드시 붙는다. 전자면 evaluator 캐시는 무죄이고 `rss_peak_mb` vs `rss_mb` 격차를 보고할 것 |

**보고할 것 (붙여넣기)**

- `soak_report.json`의 `pass`, `problems` 전문, `growth.rss` **블록 전체**(네 기울기 포함)
- `[RASBERY][EVALUATOR][MEM]` 12줄 원문
- `[RASBERY][EVALUATOR]` 프로세스 영수증의 `case_seconds`/`case_teardown_ms`
  (`window`/`observed`/`max`가 새로 붙었다)
- `[RASBERY][CASE]` 첫 줄 1개

**해석 규칙 (변경 없음)**: 181의 c/h는 전부 정보용. drift-budget findings는 이 범위 밖.
`evictions`가 0이 아니면 상한을 실제로 때린 것이고, 덱 하나·라이브러리 하나 소크에서
그것은 그 자체로 새 findings다.
