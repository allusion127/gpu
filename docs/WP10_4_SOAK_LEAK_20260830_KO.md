# WP10.4 — 소크가 FAIL한 두 이유: 이름이 어긋난 영수증 필드와, 아무도 지목할 수 없던 RSS 증가

호스트 181, `91004f7`, 2026-08-30.
`tools/soak_run.py --deck kngr_238.json --generations 5 --width 16`,
PROD env + `RASBERY_GPU_FULL=1`, 상주 evaluator 프로세스 1개.
원 보고서: `E:\rasbery_runs\20260830\181\gates_8919331.md` §(8b).

소크의 **메커니즘 영수증 16개는 전부 0**이었다 — slot duplicates, stale tenants,
double releases, refill 3종, `alloc_in_capture`, `captures_unwound`,
`cross_case_digest_mismatch`, `pin_live_ranges_between_waves`, fallback 6종.
`not_asserted: {}` (조용히 건너뛴 항목 없음), restart 0, poison 5건 모두 격리.
그런데 판정은 **FAIL**이었고, 이유는 두 가지였다. 둘 다 물리도 성능도 아니었다.

---

## 1. `physics_fidelity` — 어느 쪽이 틀렸나

### 증상

86개 findings 중 **83개**가 같은 문장이었다.

> the per-case receipt is missing 'physics_fidelity' -- this binary predates
> WP10.3 and its cases cannot be audited on fidelity individually

그런데 `91004f7`은 WP10.3 **상류가 아니다**. `src/CaseFidelity.h`가 존재하고,
`src/EvaluatorServer.h`의 `reportCase()`는 `physics_fidelity`를 인쇄한다.
즉 감사가 없는 필드를 찾은 것이 아니라, **있는 값을 다른 이름으로 찾고 있었다**.

### 원인 — 바이너리 쪽(정확히는 Driver 쪽 태그)이 틀렸다

케이스 영수증은 태그가 **둘**이고, `exact_audit.parse_case_receipts()`는 둘을
**하나의 리스트로 합쳐** 같은 `CASE_REQUIRED_FIELDS`로 감사한다.

| 태그 | 인쇄 위치 | 충실도 필드 이름 |
|---|---|---|
| `[RASBERY][EVALUATOR][CASE]` | `src/EvaluatorServer.h:reportCase()` | `physics_fidelity` (계획 §6.2 철자) |
| `[RASBERY][CASE]` | `src/Driver.h` (케이스당 1줄, one-shot 실행에도 인쇄) | **`fidelity`** (캠페인 약칭) |

`fidelity`는 WP10.1이 심은 이름이고, `tools/case_key.py`의 `COMPONENT_FIELDS`가
**그 이름으로 케이스 키를 계산한다** — 디스크의 모든 매니페스트가 그 철자를 쓴다.
그래서 Driver 줄은 계속 `fidelity`였고, 감사는 `physics_fidelity`만 봤다.
`91004f7`의 Driver 블록을 그대로 뜯어 확인:

```
fields: acceptance_eligible, case_key, code_sha, core_op, deck_digest, env_digest,
        env_set, fidelity, fidelity_declared, key_schema, policy, promoted_from,
        result_mode, schema_version, statepoint_grid, warm_start, warm_start_token,
        xslib_digest, xslib_policy
physics_fidelity present? False
```

WP10.3이 실제로 추가한 두 필드(`statepoint_grid`, `acceptance_eligible`)는
**둘 다 있었다**. 빠진 것은 이름 하나뿐이었고, 그 하나가 케이스마다 한 줄씩
"이 바이너리는 WP10.3 이전"이라는 오진을 찍었다.

### 왜 기존 테스트가 잡지 못했나 — 그리고 그 구멍이 이번 수정의 핵심

`tools/fake_rasbery_child.py`는 `[RASBERY][EVALUATOR][CASE]`만 인쇄한다.
`tools/test_soak_run.py`는 그 fake로 소크를 구동한다.
**틀린 태그가 한 번도 스트림에 들어간 적이 없었다.**
감사기는 자기가 읽을 두 태그 중 하나만 평생 본 것이다.

### 수정 — 양쪽 모두, 각각 이유가 있다

1. **`src/Driver.h`** — `[RASBERY][CASE]`가 `fidelity` **옆에**
   `physics_fidelity`를 같이 인쇄한다(값은 동일). `schema_version` 4 → **5**.
   옛 이름은 남긴다: 케이스 키가 그 이름으로 계산되고, 지우면 디스크의 매니페스트가
   전부 무효가 된다.
2. **`tools/exact_audit.py`** — `CASE_FIELD_SYNONYMS = {"physics_fidelity": ("fidelity",)}`.
   이 커밋 이전에 빌드된 바이너리의 로그도 깨끗하게 감사된다.
   **버전 거부는 약화되지 않는다**: `fidelity`는 WP10.1 필드이고, WP10.3이 실제로
   추가한 `statepoint_grid`/`acceptance_eligible`에는 동의어가 없다. 진짜로 WP10.3
   이전인 바이너리는 **케이스를 무효화했을 바로 그 필드로** 여전히 거부된다.
3. **`tools/test_soak_receipt_schema_contract.py`** (신규) — 두 emitter를
   `exact_audit.CASE_REQUIRED_FIELDS` **자체**로 대조하고(리스트 복사본이 아니라),
   Driver 태그를 이름으로 구동하며, 동의어가 버전 거부를 삼키지 않는지를 음성
   대조군으로 확인한다.

---

## 2. RSS +17.41 MB/generation — 컨테이너를 지목할 수 없던 이유

### 증상

second half 기울기 **+17.41 MB/generation** (예산 8.0). 직전 `8919331` 실행은
**+37.64 MB/generation**. 크기는 다르지만 **재현된다**. 10k 세대면 ~170 GB다.

그리고 소크가 assert하는 메커니즘 영수증은 전부 0이었다 — 이게 진짜 문제였다.
**프로세스는 자라는데, 어떤 영수증도 무엇이 자라는지 말할 수 없었다.**
181은 프로파일러를 붙일 수 있는 호스트가 아니다.

### 조사 결과 — 증명된 것과 증명되지 않은 것

**해소된(무죄 입증) 후보**

| 후보 | 판정 |
|---|---|
| `Geometry`의 raw `new[]` 배열 | `src/Geometry.cpp:59` 소멸자가 `new`된 멤버를 **전부** `delete[]` 한다 (기계적 대조: 누락 0). `XSSet/CMFD/Nodal/PPR/BICG*/IO`도 동일하게 대조, 누락 0 |
| `iowriter` 큐 | `queueDepthLimit()`/`queueByteLimit()`으로 개수·바이트 모두 유계 (`src/IoWriter.h:152,165`) |
| `iowriter::lineSink()` 버퍼 | `kLineSinkFlushBytes`(64 kB)에서 flush, `clear()`로 용량 재사용 (`src/IoWriter.h:905`) |
| `HostPinRegistry`의 `records` 맵 | 웨이브 사이 `rasberyHostPinLiveRanges() == 0`이 이미 assert되고, 소크에서 0이었다 |
| `cohort::acquirePinQuadrature`의 `quadEntries` | 키가 `(ndivxy, npins)`뿐 — 형상 수로 유계 |

**유계화한(수정) 컨테이너 — 셋 다 케이스가 바꿀 수 있는 것으로 키가 잡혀 있었다**

| 위치 | 무엇이 무한이었나 | 조치 |
|---|---|---|
| `src/XSSet.cpp:856` `XsLibraryCacheEntries()` | 엔트리 하나가 **~34 MB 파스 전체**. 키가 파일 **내용**(size·mtime·SHA-256)이라, 실행 중 라이브러리가 교체되면 **덮어쓰지 않고 하나 더 쌓인다**. one-shot에서는 불가능했던 일(프로세스가 교체보다 짧다), 상주 evaluator에서는 가능한 일 | LRU, `RASBERY_XSLIB_CACHE_ENTRIES`(기본 **2**). **placeholder(`value == nullptr`)는 절대 축출하지 않는다** — 그 키를 기다리는 워커가 영원히 기다리게 된다. `evictions` 계수 |
| `src/CohortContext.h:137` `detail::entries()` | 키가 geometry payload digest이고, **GA 후보가 곧 다른 payload**다. 캠페인 길이만큼 자란다. 게다가 `cohorts`는 "중간 수명이 존재한다"를 증명하기로 한 숫자였는데, 무한히 자라면 아무것도 증명하지 못한다 | LRU, `RASBERY_COHORT_CACHE_ENTRIES`(기본 **64**), `evictions` 계수. 축출은 재빌드 한 번의 비용뿐 — Context는 키 입력의 순함수다 |
| `include/chiffon/BatchLightResult.h:319,325` 두 digest memo | 키가 **경로 하나뿐**이고 만료가 없다. GA 후보는 각자 자기 덱 파일을 갖는다 | 상한(`RASBERY_DIGEST_MEMO_ENTRIES`, 기본 4096)에서 통째로 비운다. LRU가 아닌 이유: 파일의 순함수 memo이고 miss 비용은 재읽기 1회 — 관리 기계가 관리 대상보다 커진다. `HashCacheClears()` 계수 |

**계측만 한(수정하지 않은) 것**

- `Summary::case_seconds` / `teardown_ms` (`src/EvaluatorServer.h`) — 케이스당
  8바이트씩 영구 누적. 10k 세대 × width 64면 ~10 MB. 작지만 0은 아니고,
  p50/p90의 의미를 바꾸지 않기 위해 **자르지 않고 `case_samples`로 보고만** 한다.
- **17 MB/gen의 주범은 위 셋 중 어느 것도 아니다.** 소크는 덱 하나, 라이브러리
  하나였으므로 세 캐시 모두 엔트리 1개에서 움직이지 않았을 것이다.
  남은 후보는 계측으로 넘긴다: 할당자 보유(RSS는 free된 페이지를 곧바로 돌려주지
  않는다 — `rss_peak_mb`가 이를 분리한다), HDF5 자체 free list, CUDA 런타임의
  호스트 측 할당. **다음 소크가 이 셋 중 어느 쪽인지 말할 수 있게 만드는 것**이
  이번 작업의 실제 산출물이다.

### 수정 — 세대마다 프로세스가 자기 크기를 말한다

`src/EvaluatorServer.h::reportMemory()`, 웨이브 영수증 **직후**(웨이브 중이 아니라
— 중간 샘플은 웨이브가 마침 어디였는지를 재고, 질문은 *끝났을 때 무엇이 남았나*다):

```
[RASBERY][EVALUATOR][MEM] {"wave_id":N,"rss_mb":..,"rss_delta_mb":..,
 "rss_since_first_mb":..,"rss_peak_mb":..,"rss_readable":true,"live_cases":0,
 "cache_entries":{"xslib":..,"xslib_digest":..,"cohorts":..,"quadratures":..,
                  "pin_records":..,"digest_memo":..,"case_samples":..},
 "cache_bytes":{"xslib":..},
 "evictions":{"xslib":..,"xslib_digest":..,"cohort":..,"digest_memo_clears":..},
 "cuda_host_bytes":..}
```

- `rss_mb` — `/proc/self/statm`. 소크가 밖에서 `/proc/<pid>/status`로 재는 것과
  **같은 양**이다(일부러). 읽을 수 없으면 0과 `rss_readable:false` — "못 쟀다"와
  "안 자랐다"는 다른 사실이다.
- `rss_delta_mb` — **부호 있는** 값. 메모리를 **돌려준** 세대는 할당자 보유와
  줄지 않는 컨테이너를 가르는 사실이고, 부호 없는 델타는 그걸 숨긴다.
- `live_cases` — 생성된 Driver 수. 이 줄이 인쇄되는 시점(웨이브의 모든 케이스가
  join된 뒤)에는 **반드시 0**이다. 0이 아니면 Driver가 케이스보다 오래 산 것이고,
  그건 케이스가 소유한 모든 것의 누수다. 감소는 **소멸자**에서 한다 —
  `RASBERY_GPU_FULL=1`의 fail-closed는 설계상 throw로 도착한다.
- `cuda_host_bytes` — `rasberyHostPinLiveBytes()`(신규). 기존
  `rasberyHostPinLiveRanges()`는 **레코드 수**다: 1 MB짜리 40개와 1 kB짜리 40개는
  같은 개수이고 같은 프로세스가 아니다. 아레나의 `cudaMallocHost` 블록은
  프로세스당 한 번 서고 여기 포함되지 않는다 — 이 숫자는 **케이스와 함께 움직이는**
  쪽이다.

### 그리고 소크가 그것을 읽는다

`tools/soak_run.py`:
- 세대마다 `[RASBERY][EVALUATOR][MEM]`을 걷어 `per_generation[].mem`에 넣는다.
- RSS 기울기가 예산을 넘으면 `attribute_rss_growth()`가 **첫 영수증과 마지막
  영수증을 대조해 움직인 컨테이너를 지목**한다. 전부 평평하면 그것도 답이다 —
  "evaluator 캐시가 아니다, 다음은 할당자/HDF5 free list/디바이스 런타임" 이라고
  인쇄한다. 아무 말도 안 하는 것보다 낫다.
- 바이너리가 이 영수증을 인쇄하지 않으면 "귀속 불가, WP10.4 커밋으로 재빌드하라"고
  말한다.
- 자식의 `/proc`이 없는 호스트에서만, 프로세스 자기 보고를 **fallback**으로 쓴다
  (바깥 측정이 1순위인 이유: 재는 대상과 같은 방향으로 틀릴 수 없다).
- `tools/fake_rasbery_child.py`도 이 줄을 인쇄한다 — **§1의 구멍을 반복하지 않기
  위해서다.** 읽는 쪽만 있고 스트림에 그 태그가 한 번도 없으면, 그 리더는 아무도
  돌려본 적 없는 리더다. `FAKE_RASBERY_MEM_GROWTH_MB`로 증가 케이스를 구동한다.

---

## 3. 테스트

| 테스트 | 결과 |
|---|---|
| `tools/test_soak_receipt_schema_contract.py` (신규) | PASS (2 emitters × 4 audited fields, 1 synonym, version refusal intact) |
| `tools/test_evaluator_mem_receipt_contract.py` (신규) | PASS (8 fields, 7 sized containers, 3 bounded caches, **7 compiled**) |
| `tools/test_enum_alias_contract.py` | PASS |
| `tools/test_dependent_template_contract.py` | PASS |
| `tools/test_evaluator_contract.py` | PASS |
| `tools/test_case_key_contract.py` | PASS (schema_version 5로 갱신) |
| `tools/test_case_fidelity_contract.py` | PASS |
| `tools/test_soak_run.py` | PASS |
| `tools/test_xslib_cache_contract.py` / `test_cohort_context_contract.py` / `test_host_pin_registry.py` | PASS |

`test_evaluator_mem_receipt_contract.py`의 **컴파일 절반**은 cohort 레지스트리를
`RASBERY_COHORT_CACHE_ENTRIES=2`로 실제 구동해 (a) 상한이 지켜지고 (b) 축출된 것이
**가장 오래 안 쓴 것**이며 (c) 살아남은 것이 재빌드가 아니라 **hit**임을 확인한다.
이게 없으면 이 파일은 "어떤 파일에 어떤 단어가 있다"만 확인하는 파일이다.

전체 스위트: 이 커밋으로 **새로 깨지는 테스트 0개**. 남은 실패는 전부 동시에
진행 중인 WP13(CUDA 백엔드/전송 사이트) 작업 트리의 in-flight 파일들
(`CudaBICGBackend.cu`, `CudaOuterGraph.cu`, `CudaCramBackend.cu`,
`CudaPprBackend.cu`, `GpuPhysicsArenaCuda.cu`, `XferLedger.h`)에 대한 것으로,
이 커밋 이전 트리에서도 동일하게 실패한다.

---

## 4. 181 런북

이 커밋에서 재빌드한 뒤, §(8b)와 **같은 env·같은 인자**로:

```bash
git fetch && git checkout <this commit> && git rev-parse HEAD
# build/ 재빌드 (RASBERY_ENABLE_TESTS=OFF), 이전 181 설정과 동일한 cmake knob
cmake --build build -j32

env -i <PROD env> RASBERY_GPU_FULL=1 \
  python3 tools/soak_run.py \
    --deck kngr_238.json \
    --workdir ~/gates/wp10_4/soak \
    --binary ./build/RASBERY \
    --generations 5 --width 16 \
    --report ~/gates/wp10_4/soak_report.json
```

**기대**

1. **fidelity note 0개.** 83개 findings는 사라져야 한다. 남아 있으면 그건 이번
   수정이 아니라 실제 fidelity 위반이므로, 그 문장을 그대로 보고할 것.
2. `[RASBERY][CASE]` 줄이 `"schema_version":5`이고 `fidelity`와
   `physics_fidelity`를 **둘 다** 싣는다 (`grep -m1 'RASBERY\]\[CASE\]' ~/gates/wp10_4/soak/log/soak.log`).
3. 세대마다 `[RASBERY][EVALUATOR][MEM]` 한 줄. **`live_cases`는 매 줄 0이어야 한다.**
   0이 아닌 줄이 하나라도 있으면 다른 모든 항목보다 먼저 그것을 보고할 것 —
   Driver가 케이스보다 오래 산 것이고, 그게 곧 원인이다.
4. RSS 기울기 **≤ 8.0 MB/generation**, 또는 — 넘는다면 — soak 리포트의 problems에
   `attribute_rss_growth`가 붙인 문장이 **컨테이너를 지목**하고 있을 것.
   "every container ... was FLAT"이 나오면 evaluator 캐시는 무죄이고 다음 대상은
   할당자/HDF5/디바이스 런타임이다. 이때 `rss_peak_mb` vs `rss_mb` 격차가
   할당자 보유인지 아닌지를 가른다.

**보고할 것 (붙여넣기)**

- `soak_report.json`의 `pass`, `problems`, `growth.rss`,
  `per_generation[].mem` 전체
- `[RASBERY][EVALUATOR][MEM]` 5줄 원문
- `[RASBERY][XSLIB_CACHE]` 한 줄 (`entries`/`evictions`/`digest_entries`)
- `[RASBERY][COHORT]` 한 줄 (`cohorts`/`limit`/`evictions`)

**해석 규칙**: `evictions`가 0이 아니면 이 소크는 상한을 실제로 때린 것이다.
덱 하나·라이브러리 하나 소크에서는 0이 정상이고, 0이 아니면 키가 케이스마다
달라지고 있다는 뜻이므로 그 자체가 새 findings다.

**181 호스트 규칙 유지**: 이 호스트의 c/h 숫자는 전부 정보용이다. drift-budget
findings(§(8b)의 2건)는 이 작업 범위가 아니며 별도로 남는다.
