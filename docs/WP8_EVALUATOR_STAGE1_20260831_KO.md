# WP8 단계 1: 프로세스는 살고, 케이스 객체는 매번 죽는다

브랜치 `codex/exact-throughput-campaign`, 기준 `4273e39`
커밋 `1cdd724`(구현), `a51835f`(계약 테스트 + 클라이언트)
새 파일: `src/EvaluatorServer.h`, `tools/test_evaluator_contract.py`, `tools/evaluator_client.py`
로컬에서 수행한 것: 문법 검증(g++ 13.3 `-fsyntax-only -Wall -Wextra`, 경고 0)과
순수 파이썬 계약 테스트. **수치는 하나도 없다. 전부 238이 재야 한다(§7).**

---

## 0. 결론 먼저

BOTTLENECK 계획 §5.1 / WP8 단계 1과 GA evaluator 계획 §6.2 Task 6이 요구한 것:
**프로세스만 지속하고 케이스 객체는 매번 재생성한다.** 그것이 들어갔다.

| 항목 | 상태 |
|---|---|
| `--evaluator-jsonl [path]`(별칭 `--evaluator`) wave 프로토콜 | 완료 |
| 케이스마다 새 `Driver`/`CaseContext` (단계 3 미착수) | 완료(설계상 강제) |
| slot acquire/release 감사 | 완료 — 기존 `refill::tenancy()` 재사용, wave/process 수신증에 노출 |
| cross-case isolation: 정적 스캔 | 완료 — `tools/test_evaluator_contract.py`, 22건 분류 |
| cross-case isolation: 런타임 A→…→A digest 비교 | 완료 — `--evaluator-isolation-check` |
| 실패 격리(`RASBERY_GPU_FULL=1` fail-closed 포함) | 완료 |
| 정지 조건: `{"op":"shutdown"}` / EOF / `--evaluator-idle-timeout` | 완료 |
| 단일 CLI byte-identity | 완료 — batch/serial 분기 코드 무변경 |
| B0 게이트(케이스별 digest 동일), c/h, outside-Drive 제거량 | **미측정 — 238 필요** |

### 0.1 이 레버의 정직한 크기

GA evaluator 계획 §2.3의 실측을 그대로 읽으면:

| 버킷 | 로컬 실측 | 성격 | persistent evaluator가 지우는가 |
|---|---:|---|---|
| `outside_drive`(프로세스 이미지 + 로더 + CUDA 컨텍스트 + teardown) | 1.75–4.92 s | **프로세스당 1회** | **전액** |
| `Init+IO`(케이스당) | 0.86–1.63 s | 케이스당 | 단계 1은 **아니오** (XSLIB 캐시가 이미 그 절반을 먹었다) |
| `N_sp × 0.474 s` + `N_outer × 4.805 ms` | 38.7 s | 케이스당, 진짜 물리 | 아니오 |

즉 **케이스 하나에 대해서는 이 레버가 0 %다.** 값어치는 전적으로
"프로세스 경계가 몇 개냐"에 있다. 현재 `tools/run_multi_gpu_batch.py`는
큐 청크마다 새 RASBERY 프로세스를 띄우므로, G세대 캠페인은 `outside_drive`를
**G번** 낸다. 이 모드는 **1번** 낸다. 그래서 §7의 런북은 64케이스 1세대가 아니라
**64케이스 × 3세대를 연속으로** 돌린다 — 1세대짜리 비교는 이 레버가 지우는 항을
아예 만들지 않는다.

### 0.2 이 문서가 주장하지 않는 것

- "evaluator가 케이스를 빠르게 만든다" — 아니다. 물리는 한 글자도 안 바뀐다.
- "cross-case leak이 없다" — **증명하지 않았다.** 증명한 것은
  (a) 새로 생긴 노출면이 정확히 무엇인지(§3), (b) 그 면에 대한 런타임 witness가
  존재한다는 것(§3.3, §4), (c) 호스트 케이스 경로의 프로세스 수명 가변 상태가
  22건이고 전부 분류되었다는 것(§5)뿐이다.

---

## 1. 요청/응답 스키마

요청 스트림은 **JSONL**이다. `--evaluator-jsonl <path>`, `path`가 `-`(기본값)이면 stdin.
`#`로 시작하는 줄과 빈 줄은 무시한다. CRLF는 벗겨낸다(Windows에서 쓴 큐 파일을
WSL에서 읽는 것이 이 캠페인의 상비 함정이다).

### 1.1 op 네 가지

```json
{"op":"case","deck":"a.json","output":"out/a.h5","result_mode":"light","fidelity":"strict","key":"cand-17"}
{"op":"wave","wave_id":17,"jobs_manifest":"/path/w17.jobs","batch_width":64,"result_mode":"light"}
{"op":"run"}
{"op":"shutdown"}
```

| op | 뜻 |
|---|---|
| `case` | 다음 세대에 케이스 하나를 **큐잉**한다. 즉시 실행하지 않는다. |
| `wave` | 한 세대를 **실행**한다. 케이스는 `jobs_manifest`(기존 `--jobs` 매니페스트 문법 그대로) + 직전까지 큐잉된 `case` 줄의 합집합. |
| `run` | `wave`의 별칭. 매니페스트 없이 큐잉된 케이스만 돌린다. |
| `shutdown` | 정지. arena/CUDA 자원을 이때 **처음이자 마지막으로** 해제한다. |

스트림이 케이스를 큐에 남긴 채 끝나면 그 케이스들은 마지막 세대로 실행된다
(`src/EvaluatorServer.h:515-520`). 조용히 버리는 것만은 하지 않는다.

### 1.2 `case` 필드

| 필드 | 필수 | 의미 |
|---|---|---|
| `deck` | ✔ | `--rasi` |
| `output` | ✔ | `--raso`. 비면 **거부** — 없으면 모든 케이스가 덱 디렉터리의 기본 `result.h5`에 쓰고 서로 덮는다. |
| `result_mode` | | `full` / `pin-off` / `light`. 없으면 `--result`. |
| `key` | | 후보 키. 그대로 수신증에 되돌려준다. evaluator는 해석하지 않는다. |
| `fidelity` | | **검증만 한다. 적용하지 않는다.** §2 참조. |

### 1.3 응답 — 세 종류의 수신증

**케이스마다 한 줄** (`src/EvaluatorServer.h:845-878`):

```
[RASBERY][EVALUATOR][CASE] {"wave_id":17,"case":3,"key":"cand-17","deck":"...","output":"...",
 "result_mode":"light","status":"ok","exit_code":0,"digest":"814201df0583e1d2",
 "statepoints":35,"outers":4609,"th_updates":112,"slot":5,"lane":2,
 "wall_s":42.070,"teardown_ms":11.400,"isolation_check":false,"error":null}
```

`digest`는 `Driver::CaseReceipt`(`src/Driver.h:879-886`)에서 온다. 이것이 이 커밋이
`Driver`에 더한 유일한 것이다: trajectory 수신증과 **같은 변수에서, 그 줄 바로 앞에서**
찍는다(`src/Driver.h:4583-4588`). 케이스가 digest를 접기 전에 던지면 `complete=false`가
되고 수신증은 `"digest":null`을 낸다 — "digest가 없다"와 "digest가 0이다"는 다른 사실이다.

**세대마다 한 줄** (`[REFILL]` 바로 뒤):

```
[RASBERY][EVALUATOR][WAVE] {"wave_id":17,"jobs":64,"ok":64,"failed":0,"wall_s":...,
 "cases_per_hour":...,"process_reused":true,"xslib_loads":1,"xslib_hits":63,
 "pin_live_ranges":0,"isolation_match":true}
```

**프로세스에 한 줄** (teardown 수신증 전부 뒤, `src/main.cpp:969`):

```
[RASBERY][EVALUATOR] {"cases":192,"ok":192,"failed":0,"refused":0,"generations":3,
 "batch_width":64,"process_uptime_s":...,"drive_s":...,"cuda_context_reuse":191,
 "arena_releases":1,"arena_standups":1,"slot_admissions":192,"slot_duplicates":0,
 "slot_stale_tenants":0,"slot_double_releases":0,"xslib_loads":1,"xslib_hits":191,
 "library_loads":1,"geometry_builds":192,"pin_live_ranges_between_waves":0,
 "case_seconds":{"p50":...,"p90":...},
 "case_teardown_ms":{"p50":...,"p90":...,"max":...},
 "isolation_checks":3,"isolation_mismatches":0,"isolation_adjacent":0,
 "stop_reason":"shutdown"}
```

파생 필드 두 개는 **파생이라고 적어 둔다**:

- `cuda_context_reuse = cases - 1`. 프로세스가 이미 컨텍스트·아레나·디바이스
  라이브러리를 세운 뒤에 시작한 케이스 수다. 그 **증인**은 옆의 두 필드다:
  `arena_releases`는 shutdown에서 딱 1이 되고(세대 사이에 아무것도 아레나를 놓지
  않았다는 뜻), `slot_admissions == cases`는 모든 케이스가 슬롯을 획득했다는 뜻이다.
- `arena_standups`는 `slot_admissions > 0 ? 1 : 0`이다. 아레나는 첫 admission에서
  lazily 서고 shutdown에서만 풀리므로 프로세스당 1이 맞다 —
  `tools/test_evaluator_contract.py`가 "`rasberyReleaseBatchArena`는 wave 루프
  **밖에서만**, `EvaluatorServer.h` 안에서는 **절대** 호출되지 않는다"를 고정하는 이유가
  이 파생을 사실로 유지하기 위해서다.

---

## 2. 거부하는 것 두 가지 (그리고 왜 무시하지 않는가)

단계 1이 지킬 수 없는 요청 필드가 둘이다. 둘 다 **이름을 대며 그 케이스를 실패**시킨다.

| 필드 | 왜 지킬 수 없나 | 코드 |
|---|---|---|
| `batch_width` | 아레나는 첫 admission에서 크기가 정해지는 **단일 할당**이고 프로세스 내내 고정이다. 첫 wave가 폭을 latch하고, 다른 폭을 요구하는 뒤의 wave는 거부된다. | `src/EvaluatorServer.h:640-655` |
| `fidelity` | `PhysicsFidelity`는 프로세스당 한 번 환경에서 resolve되어 캐시되고(`src/RunContract.h`), `[PHYSICS_MODE]` 수신증은 첫 요청을 읽기 **전에** 이미 인쇄되었다. 요청은 프로세스의 fidelity를 **주장(assert)**할 수 있고 **변경**할 수 없다. | `src/EvaluatorServer.h:376-394` |

조용한 "close enough"가 screening 결과를 acceptance 표에 넣는 경로다. WP1의
exact-only 계약이 존재하는 이유가 그것이고, 이 모드는 그 계약을 **우회하지 않는다**:
evaluator 분기는 exact-only 게이트와 `[PHYSICS_MODE]` 수신증 **뒤**에 있다
(`src/main.cpp:862`, 계약 테스트가 순서를 고정한다).

이 외에 wave 단위로 거부하는 것:

- 알 수 없는 `op`
- `deck`/`output`이 없는 `case`
- **한 wave 안에서 중복된 `--raso`** — 두 Driver가 하나의 HDF5 안에서 경쟁하고
  restart 네임스페이스를 공유한다. wave **사이의** 중복은 합법이다(승격된 elite를
  다시 평가하는 방식이고, 동시성이 없다).

거부는 exit code를 움직인다. 후보 넷을 조용히 잃은 세대는 세대가 아니다.

---

## 3. 세 가지 수명, 그리고 실제로 새로 생긴 노출면

### 3.1 수명 표 (BOTTLENECK 계획 §5.1을 이 구현에 매핑)

| 수명 | 이 구현에서 무엇 | 어디 |
|---|---|---|
| **Process** | CUDA 컨텍스트, CMFD/nodal 아레나 + 그래프 캐시, immutable XSLIB 파스(`AcquireXsLibrary`), T/H 테이블(`XSSet::LoadTHTables`의 함수 지역 static), 동위원소 레지스트리·붕괴 사슬, 디바이스 flat-XS 라이브러리(`g_flatxs_libs`), host pin 레지스트리 | 프로세스 시작~`shutdown` |
| **Cohort** | **없음. 단계 2다.** 단계 1은 cohort 개념을 도입하지 않는다 — geometry는 케이스마다 다시 짓는다. | — |
| **Case** | `CaseContext`(Geometry / Scheduler / XSSet / IO), `BICGCMFD`/`Nodal`/`PPR`, 스케줄, 가변 노드 XS·동위원소 상태, 출력, **아레나 슬롯 임차** | `Driver` 생성~소멸 |

`geometry_builds == cases`가 수신증에 있는 이유가 이것이다. 단계 2에서 이 등식이
깨지는 것이 단계 2의 정의다.

### 3.2 새 노출면은 "wave 경계를 넘는 상태"뿐이다

이 문서에서 가장 중요한 한 문단이다.

**`--batch-mode M`은 이미 한 프로세스 안에서 케이스 경계를 넘긴다.** 덱이 lane보다
많으면 OpenMP 큐가 worker를 두 번째 덱에 재사용하므로, 솔버의 모든 프로세스 수명 /
`thread_local` 객체는 **오늘 이미** 케이스 경계를 살아남고 있고, 그 경로는
per-deck-vs-single 비트 동일성으로 게이트되어 있다.

따라서 evaluator가 **추가**하는 것은 정확히 `main()`의 batch 분기가 마지막에 하는
teardown 세 단계를 건너뛰어 생기는 상태다. 각각에 대해 이 구현이 한 일:

| batch 분기의 teardown | evaluator | 근거 |
|---|---|---|
| `iowriter::shutdown()` | **wave당 불필요.** 케이스마다 자기 세션을 `IO::~IO` / `IO::CloseResult`에서 fence한다(`src/IO.cpp:265-275`, `:2048-2060`). 라인 싱크만 wave마다 flush하고, writer 스레드는 shutdown에서 한 번 join한다. | `src/EvaluatorServer.h:779` |
| `rasberyReleaseBatchArena()` | **의도적으로 연기.** 이것이 레버 자체다. shutdown에서 정확히 한 번. | `src/main.cpp:917` |
| `rasberyDrainPinnedRegistry()` | **불필요, 그리고 assert로 대체.** 모든 등록은 임차(lease)이고 소유자의 소멸자가 반납하므로(`src/HostPinRegistry.h`), wave 사이에는 `rasberyHostPinLiveRanges() == 0`이어야 한다. | `src/EvaluatorServer.h:777-784` |

### 3.3 그 assert가 진짜 witness인 이유

`pin_live_ranges`가 0이 아니면 그것은 **Driver보다 오래 산 임차**다. 그 결함은
이 캠페인이 이미 한 번 값을 치른 결함이고(영구 `cudaHostRegister`가 자기 Driver보다
오래 살아 M64에서 64덱 중 54덱을 죽였다), lease가 도입된 이유가 그것이다. 그러니
"연기했지만 비어 있어야 한다"는 주장은 카운터 하나로 반증 가능하고, 0이 아니면
exit code가 움직인다.

---

## 4. cross-case isolation

계획 WP8이 요구한 여섯 개 중 이 커밋이 닫은 것과 남긴 것:

| # | 계획의 요구 | 상태 |
|---|---|---|
| 1 | `A → B → A`에서 두 A의 digest 동일 | **런타임 옵션으로 구현** — `--evaluator-isolation-check` |
| 2 | 같은 64케이스를 20가지 random order로 | 미구현 — `tools/evaluator_client.py`가 매니페스트를 받으므로 셔플은 클라이언트 쪽 한 줄이다. 238에서 수행(§7.4) |
| 3 | 의도적 실패 뒤 다음 케이스 정상 | **구현(실패 격리, §6)**, 238에서 확인 |
| 4 | 서로 다른 geometry cohort 번갈아 | **단계 2다.** 단계 1은 cohort 캐시가 없으므로 cross-talk할 대상이 없다(geometry는 매번 새로 짓는다). |
| 5 | 10,000 case soak에서 RSS/VRAM 단조 증가 없음 | 미수행 — 238 필요 |
| 6 | compute-sanitizer leak/invalid access 0 | 미수행 — 238 필요 |

### 4.1 런타임 검사가 하는 일

`--evaluator-isolation-check`는 각 wave의 **첫 덱**을 그 wave의 **끝에서** 한 번 더
돌리고 두 digest를 비교한다(`src/EvaluatorServer.h:741-772`).

- **비인접이 요점이다.** 연속으로 두 번 돌리면 대개 *한* worker의 `thread_local`
  버퍼만 다시 건드린다. 첫-그리고-마지막 순서는 재실행을 큐가 주는 아무 lane에
  올리고, 그 사이에 다른 모든 덱이 지나간 뒤다. 수신증은 `cases_between`과
  `adjacent`를 실어서, 덱이 하나뿐이라 인접해진 약한 검사를 강한 검사인 척하지 않는다.
- **재실행은 자기 출력 경로를 받는다**(`<output>.iso<wave>.h5`). restart 네임스페이스가
  출력 경로에서 파생되므로(`Driver::RestartPath`), 같은 경로를 쓰면 검사가 검사 대상을
  덮어쓴다. 바이트가 어디 떨어지는지는 바뀌고 궤적이 읽는 것은 안 바뀌므로,
  **digest 비교는 유효하고 파일 비교는 유효하지 않다**.
- 비교 대상은 **digest**다. 수신증이 인쇄하는 어떤 필드보다 아래에서 움직이는
  궤적 변화를 잡기 위해서다(digest는 statepoint마다 `(step, outers, th, efpd, keff, ppm)`의
  비트 패턴을 접는다).

### 4.2 정적 스캔이 실제로 찾은 것

`tools/test_evaluator_contract.py`는 호스트 케이스 경로
(`Driver.h`, `XSSet.cpp`, `Geometry.cpp`, `PPR.cpp`, `Nodal.cpp`, `IO.cpp`, `CMFD.cpp`,
`BICGCMFD.cpp`, `BICGSolver.cpp`)에서 프로세스 수명 **가변** 상태를 스캔한다.
`const`/`constexpr`, `std::atomic`, mutex류, 그리고 `static T x; return x;` 싱글턴
접근자는 자동 통과. 남는 것은 손으로 분류해야 한다.

**결과: 22건.**

| 분류 | 수 | 무엇 |
|---|---:|---|
| `SCRATCH` | 20 | `XSSet.cpp`/`PPR.cpp`의 `static thread_local` 작업 버퍼(`tls_xs`, `ws_tls`, `micprobe`, `history`, `gmap_interp`, …) |
| `DIAG` | 2 | `dump_done`(×2 사이트), `call` — 전부 기본 OFF인 `RASBERY_*_DUMP` 뒤의 일회성 latch/카운터 |

`SCRATCH` 20건은 **새 노출면이 아니다**(§3.2: batch refill이 이미 넘긴다). 그러나
stale 값을 읽으면 보이지 않을 것이고, 그 긍정적 보증은 **단계 3의 generation poison**이
줄 일이다. 지금은 "논증"이고 그때 "보증"이 된다. 새 항목이 하나 생기면 테스트가
실패하고 저자가 같은 논증을 명시적으로 해야 한다.

`DIAG` 2건은 **evaluator에서 케이스 간 동작을 실제로 바꾼다**: 프로세스의 **첫**
케이스만 덤프한다. 물리는 건드리지 않는다. 문서화하는 이유는, 나중에 누군가
한 케이스짜리 캡처를 케이스별 캡처로 읽지 않게 하기 위해서다.

`.cu` 아레나는 **일부러 스캔하지 않는다**: 그쪽 per-slot 상태는 텍스트 스캔보다 강한
**런타임 감사**(`duplicates` / `stale_tenants` / `double_releases`,
`src/BatchRefill.h`)가 이미 0으로 게이트하고 있다.

---

## 5. slot acquire/release 감사

새로 만든 것은 없다. `refill::ledger()`/`refill::tenancy()`를 그대로 쓴다.

- wave마다 `ledger().begin(jobs, width, host_threads)` → `jobStarted`/`jobFinished` →
  `end()` → `report()`. 즉 `[RASBERY][REFILL]` 줄이 **세대마다** 하나 나온다.
- `Driver`는 try 안의 자기 스코프에 있어서, 소멸자(= 슬롯 반납)가 teardown 스탬프가
  닫히기 **전에** 돈다(`src/EvaluatorServer.h:824-831`). batch 분기와 같은 규칙,
  같은 이유: 작게 유지해야 하는 수치가 refill latency이고 teardown이 곧 그것이다.
- 프로세스 수신증은 누적 카운터를 낸다: `slot_admissions`, `slot_duplicates`,
  `slot_stale_tenants`, `slot_double_releases`. **셋 다 0이 아니면 그 측정은 무효다.**

---

## 6. 실패 격리

`runOneCase`(`src/EvaluatorServer.h:814-842`)는 `catch (const std::exception&)`와
`catch (...)` 둘 다 잡고 **다시 던지지 않는다**.

- 탈출하는 예외는 OpenMP 병렬 영역을 종료시키고 형제 케이스들의 부분 결과를 함께
  가져간다. batch 분기가 이미 거부하는 부수 피해다.
- **`RASBERY_GPU_FULL=1` fail-closed 거부가 정확히 그런 예외로 도착한다.** 그러니
  그 케이스만 `status:"failed"` + `error:"<what()>"`로 보고되고, 프로세스는 계속
  요청을 받는다.
- 계획 WP8의 "`fatal=true`로 worker 교체" 경로는 **미구현**이다. 단계 1은
  프로세스 전역 CUDA 상태 오염을 판정할 수단이 없다 — `[GPU_FULL]` 수신증의
  `contract_pass`가 프로세스 끝에서만 최종값이 된다. 이것은 남은 구멍이고, §8에 적었다.

---

## 7. 238 런북

**전부 GPU0만, `CUDA_VISIBLE_DEVICES=0`을 자식까지 고정**(계획 §6.4).
텔레메트리 실행과 wall 측정 실행을 섞지 말 것. 첫 실행은 warm-up으로 버릴 것.

### 7.1 빌드

WP1 이후의 트리와 동일. 새 소스 파일은 **없다**(`src/EvaluatorServer.h`는 헤더이고
`src/main.cpp`가 include한다). CMake 변경 없음.

```bash
cmake -S . -B build -DRASBERY_ENABLE_CUDA=ON -DRASBERY_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 7.2 계약 테스트 (로컬에서 이미 PASS, 238에서도 돌릴 것)

```bash
python tools/test_evaluator_contract.py       # 22 classified statics
python tools/test_batch_refill_contract.py
python tools/test_io_writer_contract.py
python tools/test_xe_gpu_contract.py
python tools/test_xe_anderson.py
python tools/test_exact_only_contract.py
```

### 7.3 feature-off identity (먼저 이것부터)

evaluator 플래그를 **주지 않은** 단일/배치 실행이 `4273e39`와 바이트 동일해야 한다.
batch/serial 분기 코드는 손대지 않았으므로 이것이 깨지면 include 부작용이다.

```bash
# 이전 tip과 새 tip에서 각각
RASBERY --jobs g1.jobs --batch-mode 64 --result light > off.$SHA.log
diff <(grep -E '^\[RASBERY\]\[(TRAJECTORY|PHYSICS_MODE)\]' off.4273e39.log) \
     <(grep -E '^\[RASBERY\]\[(TRAJECTORY|PHYSICS_MODE)\]' off.a51835f.log)   # 기대: 빈 diff
```

### 7.4 본 실험 — 64케이스 × 3세대

`g1.jobs`/`g2.jobs`/`g3.jobs`는 각각 64줄짜리 기존 `--jobs` 매니페스트다.
**세 세대가 같은 64덱이어도 좋고, 그러면 §7.5의 세대 간 digest 비교가 가장 날카롭다.**

```bash
python tools/evaluator_client.py \
    --batch-mode 64 --result light \
    --generation g1.jobs --generation g2.jobs --generation g3.jobs \
    --arm persistent --arm per-generation --arm chunked-jobs \
    --reference per-generation \
    --isolation-check \
    --workdir wp8_238 --cwd /path/to/decks \
    -- ./build/RASBERY
```

세 arm:

| arm | 프로세스 수 | 무엇을 재나 |
|---|---:|---|
| `persistent` | **1** | 시험 대상 |
| `per-generation` | 3 | **같은 코드 경로**, 프로세스 경계만 다르다. digest 비교의 기준(reference) — 양쪽 다 케이스별 `[EVALUATOR][CASE]`를 내므로 덱 단위 정확 비교가 된다. |
| `chunked-jobs` | 3 | 오늘의 런처 형태. **wall 비교의 기준.** 이쪽의 trajectory 수신증은 덱이 아니라 slot을 실으므로 digest는 multiset 비교만 가능하다. |

반복 규칙: 각 arm 5회, median과 p10/p90. arm 순서를 교대해 clock drift를 상쇄.

### 7.5 판정표 — 이 표의 칸이 채워지기 전에는 어떤 주장도 없다

| 지표 | 출처 | 채택 기준 |
|---|---|---|
| 케이스별 `digest` (persistent vs per-generation) | `[EVALUATOR][CASE].digest` | **192/192 동일. B0.** 하나라도 다르면 채택 중단. |
| 세대 간 같은 덱의 `digest` (g1 vs g2 vs g3) | 동상 | **동일.** 이것이 "세대가 세대를 오염시키지 않는다"의 직접 증거다. |
| digest multiset (persistent vs chunked-jobs) | `[TRAJECTORY].digest` | 동일 |
| `isolation_mismatches` | `[EVALUATOR]` | **0** |
| `xslib_loads` | `[EVALUATOR]`, `[XSLIB_CACHE]` | **1** (계획 Task 6이 명시). 케이스 수·wave 수와 무관해야 한다. |
| `slot_duplicates` / `slot_stale_tenants` / `slot_double_releases` | `[EVALUATOR]`, `[REFILL]` | **전부 0** |
| `pin_live_ranges_between_waves` | `[EVALUATOR]` | **0** |
| `arena_releases` | `[EVALUATOR]` | **1** |
| 세대별 wall, c/h | `[EVALUATOR][WAVE].wall_s` / `.cases_per_hour` | 보고. 세대 1과 세대 2/3의 wall 차이가 곧 stand-up 비용이다. |
| 제거된 outside-Drive | 클라이언트의 `outside_drive_s` 열 (`[PROCESS]`의 `exec_s + pre_drive_s + post_drive_s` 합) | persistent가 chunked-jobs의 **1/3 근처**여야 한다 |
| stand-up 비용 비중 | 위 값 / 전체 wall | 계획 WP8 성능 게이트: **≤ 1 %** |
| 전체 처리량 | persistent c/h vs chunked-jobs c/h | 계획 WP8 성능 게이트: **≥ +5 %**, 또는 세대 간 startup jitter의 유의한 감소 |
| `case_teardown_ms` p50/p90/max | `[EVALUATOR]` | 보고. 세대가 진행되며 커지면 그것이 누수의 첫 신호다. |
| `case_seconds` p50/p90 | `[EVALUATOR]` | persistent와 per-generation이 같아야 한다(물리가 안 바뀌었으므로) |

3 % 이하의 개선은 노이즈다(계획 §6.4). 그 경우 결론은 "이 모드는 **정확성 계약과
운영 편의**를 얻었고 처리량은 얻지 못했다"이고, 그렇게 적어야 한다.

### 7.6 실패 격리 확인 (별도 실행, 1분)

```bash
# 3번째 줄의 deck 경로를 존재하지 않는 파일로 바꾼 매니페스트로 한 세대
python tools/evaluator_client.py --batch-mode 8 --generation g_bad.jobs \
    --arm persistent --reference persistent --workdir wp8_fail -- ./build/RASBERY
```

기대: `[EVALUATOR][CASE]` 한 줄이 `"status":"failed"`, 나머지 전부 `"ok"`,
프로세스는 `stop_reason:"shutdown"`으로 정상 종료, exit code 1.
`RASBERY_GPU_FULL=1`을 켠 채 arm 하나를 일부러 못 서게 해도 같은 모양이어야 한다.

### 7.7 append-follow / idle timeout 확인

```bash
: > q.jsonl
./build/RASBERY --evaluator-jsonl q.jsonl --evaluator-idle-timeout 30 --batch-mode 64 &
echo '{"op":"wave","wave_id":1,"jobs_manifest":"'$PWD'/g1.jobs"}' >> q.jsonl
# 세대 1이 끝나는 것을 보고
echo '{"op":"wave","wave_id":2,"jobs_manifest":"'$PWD'/g2.jobs"}' >> q.jsonl
# 아무것도 더 쓰지 않고 30 s 대기
wait   # 기대: stop_reason:"idle_timeout", generations:2
```

---

## 8. 남은 구멍 (그리고 이것들은 단계 1이 아니다)

1. **`tools/run_multi_gpu_batch.py`는 손대지 않았다.** 이 런처는 큐 청크마다
   `subprocess.run(... --jobs ...)`을 돌린다. evaluator로 바꾸려면 그 루프가
   **stdin으로 말하는 지속 `Popen`**이 되어야 하고, 청크 회계·MPS 관리·튜너와
   전부 얽힌다. "`--evaluator` 통과 인자 한 줄"이 아니다.
   **TODO(WP8 단계 1.5):** `run_multi_gpu_batch.py`에 `--evaluator` 모드를 넣어
   GPU/프로세스마다 지속 worker를 띄우고 청크를 wave로 보낸다. 그 전까지 §7의 A/B는
   `tools/evaluator_client.py`로 한다.
2. **worker 교체(`fatal=true`)가 없다.** §6 참조. 프로세스 전역 CUDA 상태 오염을
   판정할 신호가 아직 없다.
3. **fidelity는 프로세스 속성이다.** 한 프로세스가 light 모집단과 full elite를
   섞어 돌릴 수는 있지만(`result_mode`는 케이스별이다), strict와 A2를 섞을 수는 없다.
   그것을 원하면 evaluator 두 개다.
4. **XSLIB 캐시 키가 아직 `(path, size, mtime, ng)`이다.** 계획 WP8이 지적한
   그대로다. 장수명 프로세스에서 mtime 해상도 안에서 교체된 라이브러리 파일을
   오인할 수 있다. 이 커밋은 **손대지 않았다** — 노출면이 커진 것은 사실이지만
   내용 digest는 34 MB 재읽기이고, 단계 2의 `CohortKey`가 어차피 content digest를
   요구하므로 거기서 한 번에 한다. `xslib_loads`는 그 사이 witness다.
5. **단계 2/3은 미착수.** CohortContext(geometry RAII + `CohortKey`)와
   mutable buffer reuse(`CaseContext::reset` + generation poison).
   `geometry_builds == cases`가 그 둘이 아직 없다는 것의 기계 판독 가능한 증거다.

---

## 9. 파일 색인

| 파일 | 무엇 |
|---|---|
| `src/EvaluatorServer.h` (신규, 888줄) | 요청 스트림, wave 루프, 세 수신증, 거부, 격리 검사 |
| `src/main.cpp:385-415` | 플래그 파싱(일반 루프 **앞**에서) |
| `src/main.cpp:610` | `batch_execution` — 술어 하나, 생산자 둘 |
| `src/main.cpp:862-971` | evaluator 분기와 shutdown 수신증 블록 |
| `src/Driver.h:866-886, 3967, 4583-4588` | `CaseReceipt` — 순수 출력, trajectory 줄과 같은 변수에서 |
| `tools/test_evaluator_contract.py` (신규) | 5부 계약 + 음성 대조군 + 22건 인벤토리 |
| `tools/evaluator_client.py` (신규) | 3-arm A/B 하네스 |
