# WP19.1 — 캡처 경합의 조용한 얼굴: 재시도가 업로드를 생략해 만든 non-finite

238, `c4656c6`(= WP19), batch 8 프로세스 × M16, v6 env
(`RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_CRAM=1 RASBERY_GPU_OUTER=1
RASBERY_GPU_OUTER_GRAPH=1`) 기준.
증거: `E:\rasbery_runs\2026-08-30\238\nonfinite_0048\` (7개 파일).

---

## 1. 증상

WP19가 닫은 것과 **모양이 같고 메시지가 다른** 죽음이 남았다.
5회 중 약 2회, 128 케이스 중 1개가 ~18 s에서 죽는다.

```
[RASBERY][EVALUATOR][CASE] {"wave_id":101,"case":0,"deck":".../candidate_0048.json",
  "status":"failed","exit_code":1,"slot":-1,"lane":1,"wall_s":18.414,
  "error":"CUDA BiCGSTAB detected a non-finite value"}
```

| 성질 | 관측 | 의미 |
|---|---|---|
| 위치 | 항상 레인의 **첫 케이스** | 그래프 **빌드**가 일어나는 유일한 케이스 |
| 덱 | 재실행마다 이동 | 덱 고유 결함 아님 |
| 단독 | `solo_0048_run{1,2,3}.log` 전부 PASS, digest 동일 | 물리/입력 결함 아님 |
| CUDA 오류 | **없음** | 실패가 아니라 **오염** |
| 중재자 리시트 | p6·p7 **둘 다** `"capture_race_retry":1` | 경합은 두 프로세스에서 모두 발화 |

두 가지를 먼저 정정한다.

* `"slot":-1`은 **기동을 뜻하지 않는다.** `Driver.h:5885`가
  `_case_receipt.slot`을 **정상 완료 시에만** 찍기 때문에, 죽은 케이스는
  무조건 `-1`이다. WP19에서 이 값이 기동을 가리킨 것은 우연히 맞았을 뿐이다.
* `capture_race_retry` 필드는 **로그에 있었다** —
  `gpu0.p6.evaluator.log:1247`, `gpu0.p7.evaluator.log`. 다만 프로세스당
  누적 정수 하나뿐이었고 하네스 로그에는 그 단어가 없었다(§5).

---

## 2. 메커니즘 — 어느 채널인가

부모 과제가 제시한 후보 (a) 캡처 스트림 흡수, (b) 공유 워크스페이스,
(c) 이벤트, (d) memset 중 **어느 것도 단독 원인이 아니다.** 실제 채널은
WP19가 추가한 **재시도 그 자체**이며, 성질은 (b)의 결과("절반만 초기화된
메모리에 대고 그래프를 replay")와 정확히 같다.

### 2.1 세 아레나는 이미 자기 스트림을 직렬화한다

| 아레나 | 스트림 | 직렬화 |
|---|---|---|
| CMFD (`CudaBICGBackend.cu:4050` `BatchCore::stream`, 16슬롯 공유) | 1개 | `a.stream_mutex`(`TimedStreamClaim`) — 랑데부 런처·`enqueueSweeps`·`finishSweeps` 전부 |
| Nodal (`CudaXsReconBackend.cu:1264` `NodalArena::_stream`) | 1개 | `_mutex` + `_launching` 랑데부 |
| Outer segment (`CudaOuterGraph.cu:934` `own_stream`) | 슬롯당 1개 | 배치에서는 **사설**. `Driver.h:1970` `solo && …`가 CMFD 아레나 스트림 채택을 단독 실행으로 제한한다 |

즉 배치에서 `outer.while` 캡처의 body 스트림(`m.stream`)과 root
스트림(`m.while_cache.root_stream`)은 둘 다 그 레인만의 것이다. 형제 레인의
`cudaMemcpyAsync`가 캡처에 빨려 들어갈 경로는 **없다**. (a)는 성립하지 않는다.

### 2.2 실제 채널 — `record(body)` 이후의 재시도

`src/CudaOuterGraph.cu:2934-2949` (`c4656c6` 기준 `:2890-2906`):

```cpp
rc = buildOuterWhile(..., [&](cudaStream_t){ return runOneOuter(1u); }, &root, &exec);
if (rasbery::captureIllegal(static_cast<int>(rc))) {   // ← stage 를 보지 않는다
    rasbery::noteCaptureRaceRetry();
    rc = buildOuterWhile(..., [&](cudaStream_t){ return runOneOuter(1u); }, ...);
}
```

`buildOuterWhile`(`src/GpuOuterWhile.h:259`)의 커서 `stage`는 8단계의 순수
그래프 배관 뒤에 `record(body)`가 온다. **`record(body)` 이후에 refuse된
빌드는 이미 body의 호스트 측 enqueue 헬퍼를 전부 실행한 상태다.** 그리고 그
헬퍼들은 복사가 **랜딩된 뒤가 아니라 발행 시점에** 그림자를 커밋한다.

* `src/CudaOuterGraph.cu:1622` `m.resident_xsnf.commit(...)`
  — 주석 자체가 "COMMITTED AT THE ISSUE"
* `src/CudaOuterGraph.cu:1645` `m.resident_dtil_generation = live.dtil_generation;`
* `src/CudaOuterGraph.cu:2547` flux 업로드의 `flux_current` 판정
* 그 밖에 CMFD 백엔드의 byte-exact 그림자(`src/CudaTransferMirror.h`,
  "commit() only after the copy has completed")와 nodal residency 주장

캡처 중 발행된 복사는 **기록만 되고 실행되지 않는다.** 빌드가 실패해 그래프가
버려지면 바이트는 디바이스에 **도달한 적이 없는데** 호스트는 도달했다고 믿는다.
두 번째 `record`는 그림자가 일치하므로 그 업로드들을 **생략(elide)**하고,
결과적으로 **디바이스가 갖고 있지 않은 데이터에 대한 H2D 노드가 없는** WHILE
body가 instantiate된다. 레인의 **첫 케이스**에서 그 디바이스 메모리는
초기화 전이므로, replay는 non-finite flux를 낳고 네 프레임 떨어진
`CudaBICGBackend.cu:6891` / `:7032`(`a.core.slot[m].nonfinite`)에서 발견된다.

같은 함수는 **Task 10 이래 정확히 이 이유로** stream arm 폴백을 거부해 왔다
(`:2837` "Re-running them on the stream arm would elide uploads whose bytes
never left the host, which is a plausible wrong answer rather than a slow one",
`host_state_moved` 판정 `:2966`). WP19의 재시도는 **그 옆에 추가되면서 같은
질문을 하지 않았다.**

### 2.3 왜 PPR 재시도는 안전한가

`src/CudaPprBackend.cu:2159-2168`(재시도는 `:2191`)의 recorder는 `enqueue_round`(`:2044`) — **커널 런치뿐**이다.
이 statepoint의 H2D(phif/phis/jnet, XS 4종, chif/crdf, loop 스칼라)는
`:1909-1943`에서 **빌드 이전, 캡처 밖**에 이미 발행·기록된다. 두 번째 record가
생략할 업로드가 없다. p6/p7이 **둘 다** `capture_race_retry:1`인 것과, 그중
하나만 죽은 것이 이것으로 설명된다 — 같은 누적 카운터에 **안전한 재시도와
오염시키는 재시도가 구분 없이** 들어갔다.

---

## 3. 수정

### (a) `stage`를 묻는 술어 하나, 두 게이트가 공유 — `src/CudaOuterGraph.cu:759`

```cpp
[[maybe_unused]] bool outerWhileStageMovedHostState(const char* stage);
```

`GpuOuterWhile.h`가 실제로 대입하는 8개의 배관 단계만 "안 움직였다"이고,
`nullptr`과 그 밖의 모든 값은 "움직였다"(보수적 방향). 재시도 게이트와 기존
`host_state_moved` 폴백 게이트가 **같은 함수**를 부른다.

### (b) 움직인 빌드는 재시도하지 않고 **포기**한다 — `:2934`

```cpp
if (captureIllegal(rc) && outerWhileStageMovedHostState(stage)) {
    rasbery::noteCaptureRaceAbandoned("outer.while", stage, rc, slot, ...);   // 시끄럽게
} else if (captureIllegal(rc)) {
    rasbery::noteCaptureRaceRetry("outer.while", stage, rc, slot);            // 종전대로 1회
    ...
}
```

포기는 아래의 기존 하드스톱(`LaunchFailed` + `releaseCanonicalNodal(false)` +
`return -1`)으로 떨어진다. **조용한 오답이 이름 있는 실패로 바뀐다.**

### (c) 재시도 리시트가 실제로 찍힌다 — `src/GpuCaptureArbiter.h`

* `noteCaptureRaceRetry(tag, stage, cuda_error, slot)` — 이벤트마다
  `[RASBERY][CUDA][CAPTURE_RACE][RETRY]` 한 줄을 stderr에.
* `noteCaptureRaceAbandoned(...)` — `[RASBERY][CUDA][CAPTURE_RACE][ERROR]`,
  새 카운터 `capture_race_abandoned`.
* `captureArbiterProvenance()` — 첫 케이스 사망 라인에 붙일 기동 정황.
* 중재자 리시트에 `capture_race_abandoned` 추가.

### (d) 첫 케이스 non-finite는 시끄럽고, 깨끗한 슬롯에서 **1회** 재실행 — `src/EvaluatorServer.h`

`retryAfterCaptureRace()`가 wave 경로와 rolling 경로 **둘 다**에 걸린다.
게이트는 세 개: `status != 0`, **레인의 첫 케이스**,
`captureRaceCorruptionSuspect(failure)`(= 트리가 실제로 throw하는 문자열).
레인의 **나중** 케이스는 첫 케이스가 만든 그래프를 replay할 뿐이므로 재시도
대상이 아니다 — 그것을 재시도하면 물리를 운으로 세탁하게 된다. 슬롯은
`runOneCase`가 Driver를 스코프하므로 구조적으로 깨끗하다.
프로세스 리시트에 `capture_race_case_retries` / `capture_race_case_recovered`.

### (e) 하네스가 증거를 들어올린다 — `tools/run_multi_gpu_batch.py`

`collect_capture_race()`가 `[CUDA][CAPTURE_ARBITER]` 리시트(스냅샷)와
`[CUDA|EVALUATOR][CAPTURE_RACE]` 이벤트 라인(집합)을 PROC 리시트에 싣고,
TOTAL에 `capture_race_retry` / `_abandoned` / `_unrecovered` 합계를 찍는다.

---

## 4. WP19 재시도 리시트가 보이지 않았던 이유

세 겹이었다.

1. `noteCaptureRaceRetry()`는 **인자도 출력도 없었다** — 카운터 하나만 증가.
   어느 사이트(`ppr.while` vs `outer.while`), 어느 stage, 어느 슬롯인지 아무
   기록이 없으므로 **안전한 재시도와 오염시키는 재시도를 구분할 수 없었다.**
2. 그 카운터는 **teardown 시 중재자 리시트 한 줄**에만 나타난다 — 다른 10개
   숫자 사이에.
3. `tools/run_multi_gpu_batch.py`가 그 라인을 **읽지 않았다.** 그래서
   `batch_m16_run2.log`에는 `capture_race`라는 단어가 아예 없고, 값은
   `gpu0.p6.evaluator.log:1247`처럼 이미 무엇을 찾을지 아는 사람만 여는
   워커별 파일에만 남았다.

(1)은 (c)가, (3)은 (e)가 닫는다.

---

## 5. 계약 테스트

`tools/test_capture_standup_isolation_contract.py` — 7 검사 + **7 음성 대조**.

| 검사 | 지키는 성질 |
|---|---|
| 술어 하나, 빌더와 일치 | `outerWhileStageMovedHostState`의 배관 목록 == `GpuOuterWhile.h`가 `record(body)` **이전에** 대입하는 stage 집합, 두 게이트가 모두 호출 |
| 게이트 없는 재시도 금지 | 게이트가 재시도보다 **앞**에 있고 `else if`로 묶임, 모든 `noteCaptureRaceRetry(`가 site·stage·slot을 넘김, 예외인 recorder는 소스에 이유를 씀 |
| 리시트가 찍힌다 | `[RETRY]`/`[ERROR]` 라인 + `std::cerr`, 3개 카운터가 리시트에, 하네스가 들어올림 |
| 하네스가 실제 라인을 읽는다 | 방출되는 문자열 3종을 `collect_capture_race`에 통과(중복 스캔 포함) |
| 첫 케이스만, 1회만 | `lane_first_case` **조기 반환에서** 읽음, `runOneCase` 정확히 1회, provenance·stderr, 두 레인 루프 모두 |
| 공유 스트림이 캡처 body에 안 들어감 | `Driver.h`의 `shared_stream`이 `solo` 게이트를 유지, WHILE root는 전용 스크래치 스트림 |
| 문자열 일치 | `captureRaceCorruptionSuspect`가 찾는 부분문자열이 `CudaBICGBackend.cu`가 throw하는 문장에 실제로 들어 있음 |

음성 대조에는 **`c4656c6`의 모양 자체**(재시도가 stage를 묻지 않음)가 들어
있다. 그 변이가 잡히지 않으면 이 파일은 실패한다.

동반 통과: `tools/test_capture_arbiter_contract.py`,
`tools/test_gpu_capture_arbiter_contract.py`, `tools/test_enum_alias_contract.py`,
`tools/test_dependent_template_contract.py`, `tools/test_xfer_ledger_contract.py`.

---

## 6. 238 런북

로컬 계산 금지. 238에서, 출력은 `E:` 아래.

```bash
# 0) 빌드 (238, CUDA 13)
cd <tree> && cmake --build build -j

# 1) v6 배치 8 프로세스 × M16 × 5회.  --workdir 은 주지 않는다.
for i in 1 2 3 4 5; do
  python tools/run_multi_gpu_batch.py \
      --manifest <128-case manifest> \
      --gpus 0 --procs 8 --batch-width 16 \
      --evaluator --result light \
      --env-file <v6 env>
done
```

**합격 조건 (5회 모두):**

1. `failed_cases` **0**, `failed_case_errors` **비어 있음**,
   `[RASBERY][MULTI_GPU][FAIL]` 라인 없음. 특히 **non-finite 0건**.
2. TOTAL 리시트의 `"capture_race_unrecovered":0`.
3. `"capture_race_abandoned"`는 **0이 이상적이나 0이 아니어도 합격** — 값이
   있으면 그것은 "경합이 발화했고, 오염 대신 이름 있는 폴백을 택했다"는 뜻.
   실행별로 표에 적을 것. 이 값이 오르면서 케이스가 죽으면 그때는 별개 결함.
4. 경합이 발화했다면 **증거가 보여야 한다**:
   `[RASBERY][CUDA][CAPTURE_RACE][RETRY]` 또는 `[ERROR]` 라인이 하네스 로그
   (`capture_race` 배열)에 site·stage·slot과 함께 들어 있을 것. WP19에서는
   이것이 **한 줄도 없었다** — 그 부재 자체가 회귀 신호다.
5. 첫 케이스 재시도가 걸렸다면 `[RASBERY][EVALUATOR][CAPTURE_RACE]`와
   `…[RESULT] {"recovered":true}`가 짝으로 있고,
   `capture_race_case_recovered == capture_race_case_retries`.
6. `"alloc_blocked" > 0` — 중재자가 실제로 직렬화했다는 증거.
7. digest 불변: v5/v6 고정 digest와 대조(`tools/exact_audit.py`).
   재시도가 낳은 케이스도 **동일 digest**여야 한다.

**음성 대조는 이미 확보되어 있다.**
`E:\rasbery_runs\2026-08-30\238\nonfinite_0048\`가 `c4656c6`의 대조군이다
(1/128 non-finite, 18.4 s, `lane:1`, `capture_race_retry:1`). 다시 만들 필요는 없다.

**추가 스트레스(선택, 확률을 1로 끌어올림):**
`RASBERY_GPU_CAPTURE_STALL_US=200000`으로 캡처 창을 넓혀 1회 실행.
기대: non-finite 0건이면서 `capture_race_abandoned` 또는
`capture_race_retry`가 크게 오르고, **그 이벤트 라인이 로그에 보인다.**
`RASBERY_GPU_CAPTURE_TRACE=1`은 `[RASBERY][CAPTURE]` 라인으로 창 겹침을
스레드 단위로 보여준다.

---

## 7. 남은 것 (이번 범위 밖)

* **`m.while_cache`가 오염된 그래프를 캐시할 가능성은 제거됐다** — 포기 경로가
  instantiate 이전에 끊기므로. 다만 캐시가 슬롯당 프로세스 수명이라는 사실은
  "첫 케이스만"이라는 성질의 근거이므로, 캐시 수명을 바꾸는 변경은 이
  문서와 계약 테스트를 함께 갱신해야 한다.
* **`NodalArena::adoptCanonical()`**(`CudaXsReconBackend.cu:1502`)은 아레나
  `_mutex` 밖에서 공유 `_canon[]`과 `_views_dirty`를 쓴다. 런처가
  `refreshViews()` 안에서 `_views_dirty=false`를 놓는 사이에 낀 채택은
  **유실될 수 있다**. 이번 증거와는 무관하고(캡처와도 무관) 별건이지만,
  기동-대-구동 경합으로는 실재한다. 별도 WP.
* `src/GpuDeviceBlockPool.h`는 다른 작업(evaluator VRAM 할당)의 신규 파일이며
  이번 커밋에 포함되지 않는다.
