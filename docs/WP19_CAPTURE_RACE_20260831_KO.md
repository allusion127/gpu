# WP19 — 캡처 경합(capture race): 레인 기동 중 죽는 2~5/128 케이스

238, `bd7a0d3`, batch 8 프로세스 × M16, v6 env(`RASBERY_GPU_PPR=1
RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_CRAM=1`) 기준.

---

## 1. 증상

128 케이스 중 **2~5개**가 약 15.7 s 지점에서 죽는다. 리시트는 이렇게 남는다.

```
"slot":-1, "status":"failed",
"error":"cudaGetLastError(): operation not permitted when stream is capturing"
```

측정된 성질 네 가지가 원인을 거의 다 지목한다.

| 성질 | 관측 | 의미 |
|---|---|---|
| 위치 | 항상 레인에 **첫 번째로 배정된 케이스** | solve가 아니라 **기동(stand-up)** |
| `slot` | 항상 `-1` | 아레나 슬롯을 받기 전 |
| 재현성 | 재실행마다 **다른 덱** | 덱 고유 결함 아님 |
| 단독 실행 | 죽은 덱 전부 single-shot PASS | 물리/입력 결함 아님 |

`operation not permitted when stream is capturing` =
`cudaErrorStreamCaptureUnsupported`(900). "이 스레드가 지금 해서는 안 되는
CUDA 호출을 했다"는 뜻이고, 하네스 로그에는 **이 문장이 한 줄도 없었다**
(§5).

---

## 2. 메커니즘

`--batch-mode M`은 **하나의 디바이스**를 향해 **M개 호스트 Driver 스레드**를
띄운다. CUDA의 stream capture 규칙은 이 배치에서 다음과 같이 작동한다.

* `cudaStreamCaptureModeGlobal` — 캡처가 열려 있는 동안 **무관한 스레드의**
  "potentially unsafe" API가 전부 즉시 실패한다.
* `ThreadLocal` / `Relaxed` — 실패시키지는 않는다. 대신 그 호출이 **남의
  캡처를 무효화**한다. 캡처 중이던 레인이 엉뚱한 곳에서 죽는다.

둘 다 사고로는 살아남을 수 없으므로, 이 트리는 Rev.7.1 Task 18d에서
`src/GpuCaptureArbiter.h`(프로세스 전역 shared/exclusive 락)를 도입했다.
규칙은 하나다.

> **캡처 창(CaptureWindow)은 프로세스 안의 모든 할당·페이지락·스트림/이벤트
> 생성소멸·디바이스 전역 드레인(AllocWindow)과 배타적이다.**

캡처가 writer, 할당이 reader — 폴라리티가 뒤집힌 이유는 "조용한 프로세스를
봐야 하는 쪽"이 캡처이기 때문이다.

---

## 3. 발견된 구멍 (file:line)

중재자는 **다섯 개 캡처 지점 중 넷**에만 적용되어 있었다.

### 3.1 주 구멍 — PPR WHILE 캡처가 중재자 밖 (`src/CudaPprBackend.cu:2112`, `bd7a0d3` 기준)

```cpp
// bd7a0d3 src/CudaPprBackend.cu:2112 — 창(window)이 없다
const cudaError_t brc = buildPprWhile(
    s.graph_root_stream, s.stream, s.d_loop, &stage, ..., &root, &exec);
```

`buildPprWhile`(`src/CudaPprBackend.cu:1101`)은 **두 개의 BeginCapture**를 연다
(`:1106` root, `:1173` `BeginCaptureToGraph(body)`). 그 사이 프로세스를 조용히
붙잡아 두는 것이 아무것도 없었다. 이 TU는 `GpuCaptureArbiter.h`를 **include조차
하지 않았다.**

### 3.2 동반 구멍 — PPR 기동 경로 전체가 중재자 밖

| 위치(`bd7a0d3`) | 호출 |
|---|---|
| `src/CudaPprBackend.cu:1518` `ensureShape` | `cudaStreamCreate` ×1, `cudaEventCreate` ×2, `cudaMalloc` ×25, `cudaMallocHost` ×3 |
| `src/CudaPprBackend.cu:1608` `ensureReconShape` | `cudaMalloc` ×23, `cudaMallocHost` ×1 |
| `src/CudaPprBackend.cu:1666` `ensureReconFlux` | `cudaMalloc` ×2 |
| `src/CudaPprBackend.cu:1421` `releaseRecon` / `:1457` `release` | `cudaFree`/`cudaFreeHost`/`cudaStreamDestroy` 전량 |
| `src/CudaPprBackend.cu:2105` | `cudaStreamCreate(graph root)` |

`ensureShape`는 **모든 덱의 첫 statepoint**에서 돈다. M16이면 먼저 출발한 레인이
캡처를 여는 순간과, 몇 ms 뒤 출발한 레인이 기동 할당을 시작하는 순간이 겹친다 —
그래서 **간헐적이고, 위치 무관하고, 항상 레인의 첫 케이스**다.

### 3.3 같은 모양의 두 번째 백엔드 (`src/CudaCramBackend.cu:627`, `:686`)

`RASBERY_GPU_CRAM=1`도 v6 arm이다. `release()`/`ensureShape()`가 PPR과 똑같이
맨몸이었다. WP19의 소스 스캔이 찾아냈다.

### 3.4 남아 있던 개별 호출

| 위치 | 호출 |
|---|---|
| `src/CudaOuterGraph.cu:1877` | `cudaEventCreateWithFlags` (nodal handover) |
| `src/CudaOuterGraph.cu:2851` | `cudaStreamCreateWithFlags` (WHILE root stream) |
| `src/CudaBICGBackend.cu:6238/6240` | `cudaEventDestroy` (아레나 teardown) |
| `src/CudaBICGBackend.cu:6644/6650` | `cudaEventCreateWithFlags` (per-slot segment events) |
| `src/CudaXsReconBackend.cu:2194` | `cudaEventCreateWithFlags` (nodal done event) |
| `src/CudaXsReconBackend.cu:2829` | `cudaStreamCreateWithFlags` (backend stream) |
| `src/CudaXsReconBackend.cu:3792` | `cudaEventCreateWithFlags` (micx ready) |
| `src/GpuOuterWhile.h:179` | `cudaStreamDestroy` (WHILE 캐시 해제) |

### 3.5 왜 기존 계약 테스트가 못 잡았나

`tools/test_gpu_capture_arbiter_contract.py`는 **네 개 소스 이름을 하드코딩**해
검사한다(`CudaBICGBackend.cu`, `CudaXsReconBackend.cu`, `CudaOuterGraph.cu`,
`GpuPhysicsArenaCuda.cu`). 목록에 없는 백엔드는 검사 대상이 아니다 —
`CudaPprBackend.cu`와 `CudaCramBackend.cu`는 **한 번도 검사된 적이 없다.**

### 3.6 `cudaStreamCaptureModeGlobal`이 진단이 아니라 결함인 이유

`src/CudaBICGBackend.cu`의 `ScopedStreamCapture::captureMode()`에는
`RASBERY_GPU_CAPTURE_MODE=global` 분기가 남아 있었다. Global 모드에서
**무관한 스레드**가 받는 오류 문자열이 정확히
`operation not permitted when stream is capturing`이다. 즉 그 모드는 결함을
보고하는 도구가 아니라 **결함 그 자체**다.

---

## 4. 수정

### (a) 캡처 모드 — 사이트별로 고정, Global 제거

| 사이트 | 모드 | 이유 |
|---|---|---|
| `src/CudaBICGBackend.cu:204` (`ScopedStreamCapture`) | `ThreadLocal` (기본), env로 `Relaxed`만 허용 | 잎(leaf) 캡처 — 캡처 스레드만 제약 |
| `src/CudaPprBackend.cu:1117 / :1184` | `Relaxed` | 조건부 노드 빌드가 창 안에서 **두 번째 스트림**(body)을 만짐 |
| `src/GpuOuterWhile.h:265 / :332` | `Relaxed` | 같은 이유 |
| `src/CudaXsReconBackend.cu:1416 / :4487` | `ThreadLocal` | 잎 캡처 |

`cudaStreamCaptureModeGlobal`은 소스에서 **사라졌고**, 계약 테스트가 코드 줄에
다시 나타나는 것을 막는다(주석에서 설명하는 것은 허용).

### (b) 중재자 구멍 닫기

* `src/CudaPprBackend.cu` — `GpuCaptureArbiter.h` include, `buildPprWhile` 호출을
  `rasbery::CaptureWindow(s.stream, "ppr.while")` 안으로, 기동/해제 다섯 경로에
  `rasbery::AllocWindow`.
* `src/CudaCramBackend.cu` — 같은 처리(`cram.shape.standup`, `cram.release`).
  이 파일은 CRLF이며 CRLF로 유지된다.
* `src/CudaOuterGraph.cu`, `src/CudaBICGBackend.cu`, `src/CudaXsReconBackend.cu`,
  `src/GpuOuterWhile.h` — §3.4의 개별 호출에 `AllocWindow`.

결과: **`src/` 안의 unsafe 자원 호출 중 창 밖에 있는 것은 0개.** 계약 테스트의
허용 목록(`ALLOC_ALLOW`)은 **비어 있고**, 그것이 정상 상태다. "teardown이라
안전하다"는 M16 배치에서 논거가 아니다 — 열여섯 덱이 무너지는 동안 열여섯 덱이
선다. `CudaBICGBackend.cu`의 아레나 소멸자도 면제를 주장하지 않고 창을 잡는다.

### (c) 시끄럽고 회복 가능하게

* **재시도** — 그래도 capture-illegal 코드(900–908)가 돌아오면, sticky error를
  지우고 **중재자를 잡은 채 정확히 한 번** 다시 빌드한다
  (`src/CudaPprBackend.cu`, `src/CudaOuterGraph.cu`).
  루프가 아닌 이유: 두 번째 실패는 더 이상 경합이 아니다.
* **카운터** — `rasbery::captureArbiterStats().capture_race_retry` /
  `capture_race_unrecovered`, 리시트 라인
  `[RASBERY][CUDA][CAPTURE_ARBITER]`에 두 항목 추가.
* **사다리** — `ppr::Refusal::CaptureRaceRetry` → `"capture_race_retry"`.
  재시도가 **성공하면** `refusals[CaptureRaceRetry]`만 오르고 `refusal`은
  `none`으로 되돌아간다(폴백하지 않았으므로). 재시도까지 지면
  `[RASBERY][CUDA][CAPTURE_RACE][ERROR]`가 stderr로 나간다.
* **케이스 사망의 전파** — `src/EvaluatorServer.h::reportCase`가 status≠0인
  케이스마다 전용 라인을 찍는다:

  ```
  [RASBERY][EVALUATOR][ERROR] {"wave_id":..,"case":..,"deck":"..","lane":..,
                               "slot":..,"exit_code":..,"error":".."}
  ```

  프로토콜 스트림과 **stderr 양쪽**으로. `tools/run_multi_gpu_batch.py`는
  `EVALUATOR_CASE_ERROR` / `collect_case_errors()`로 이를 세 경로(wave·rolling·
  epilogue) 모두에서 걷어 `problems`에 넣고, 그것이
  `[RASBERY][MULTI_GPU][FAIL]`에 `gpu.. p..: case died: <deck>: <message>`로
  나온다. 리시트 JSON에도 `failed_case_errors`로 실린다.

### (d) 실행마다 새 workdir

`--workdir` 기본값이 고정 `multi_gpu_run/`이어서 연속 실행이 서로의 로그를
덮어썼다 — 간헐 결함을 세 번 재현해도 증거는 한 번치만 남는다. 기본값은 이제
`multi_gpu_run/run_<UTC timestamp>/`이고, 시작 시
`[RASBERY][MULTI_GPU][WORKDIR] {"workdir":"..","explicit":false}`로 찍는다.
`--workdir <path>`는 그대로 문자 그대로 쓰이며(재개용 큐가 필요로 한다) 그때는
`"explicit":true`다.

---

## 5. 하네스가 오류를 삼킨 자리

원 실행 로그에 오류 문장이 **없었던** 이유는 `tools/run_multi_gpu_batch.py`가
`[EVALUATOR][CASE]` 리시트에서 `deck`과 `status`만 꺼내고 `error`를 버렸기
때문이다(세 곳: rolling 파서, chunk 파서, 최종 problems). 다섯 덱을 잃은 실행이
남긴 것은 **덱 이름 다섯 개**뿐이었다. §4(c)가 이것을 닫는다.

---

## 6. 계약 테스트

`tools/test_capture_arbiter_contract.py` (신설). 사이트를 **열거하지 않고
스캔**하므로 새 백엔드는 추가된 날 자동으로 검사 대상이 된다.

| 검사 | 내용 |
|---|---|
| Global 금지 | 코드 줄에 `cudaStreamCaptureModeGlobal` 0건(주석은 허용) |
| 모드 명시 | 모든 BeginCapture가 4줄 안에서 ThreadLocal/Relaxed/`captureMode()`를 부름 |
| 캡처가 중재자 안 | 블록 깊이 추적으로 `CaptureWindow`가 살아 있는지; 헬퍼(`buildOuterWhile`/`buildPprWhile`)는 **모든 호출자**가 창을 잡았는지; 멤버 창(`ScopedStreamCapture`)은 클래스로 인정 |
| 할당이 중재자 안 | 15종 unsafe API 전량, 허용 목록 **비어 있음** |
| 재시도 경로 | `captureIllegal` / `noteCaptureRaceRetry` / `noteCaptureRaceUnrecovered` / 리시트 항목 / 사다리 rung |
| 오류 전파 | `[EVALUATOR][ERROR]` 발행 + stderr + 하네스 3경로 수집, 그리고 **실제 라인을 파서에 통과시키는 라이브 검사** |
| workdir | 고정 기본값 금지, 타임스탬프, 출력 |

부정 대조 7종(Global 복구, 모드 삭제, 캡처 창 삭제, PPR 창 삭제, PPR 기동 창
삭제, CRAM 기동 창 삭제, 카운터 삭제) 전부 잡힌다.

**동반 실행 결과 (로컬, 소스 전용):**

```
tools/test_capture_arbiter_contract.py      PASSED (8 checks, 7 controls)
tools/test_gpu_capture_arbiter_contract.py  PASS   (기존, 계속 통과)
tools/test_enum_alias_contract.py           PASS
tools/test_dependent_template_contract.py   PASS
tools/test_outer_segment_v2_contract.py     PASSED
tools/test_evaluator_rolling_contract.py    PASS
tools/test_xfer_ledger_contract.py          PASSED
```

전체 `tools/test_*.py` 스윕에서 실패하는 14종은 **이 변경 이전과 동일한
집합**이다(컴파일러 부재 또는 선행 드리프트). 회귀 없음.

---

## 7. 238 런북

로컬 계산 금지. 238에서, 출력은 `E:` 아래.

```bash
# 0) 빌드 (238, CUDA 13)
cd <tree> && cmake --build build -j

# 1) v6 배치 × 3회 재실행.  --workdir 을 주지 않는다: 매 실행이 자기
#    타임스탬프 디렉터리를 갖고, [MULTI_GPU][WORKDIR] 로 어디인지 찍는다.
for i in 1 2 3; do
  python tools/run_multi_gpu_batch.py \
      --manifest <128-case manifest> \
      --gpus 0 --procs 8 --batch-width 16 \
      --evaluator --result full \
      --env-file <v6 env>          # RASBERY_GPU_PPR=1 _PPR_GRAPH=1 _CRAM=1
done
```

**합격 조건 (3회 모두):**

1. `failed_cases` **0**, `failed_case_errors` **비어 있음**,
   `[RASBERY][MULTI_GPU][FAIL]` 라인 없음.
2. `[RASBERY][CUDA][CAPTURE_ARBITER]` 리시트에서
   `"capture_race_unrecovered":0`.
   `"capture_race_retry"`는 **0이 이상적이고 0이 아니어도 합격**이다 — 그 값은
   "경합이 실제로 일어났고 케이스가 살아남았다"는 뜻이므로 **기록한다**
   (실행별로 표에 적을 것).
3. `"alloc_blocked" > 0` — 중재자가 실제로 직렬화했다는 증거. 0이면 창이
   비어 있다는 뜻이므로 의심할 것.
4. PPR 리시트: `refusal`이 `capture_race_retry`가 **아님**
   (`refusals` 배열의 해당 항목은 0이 아니어도 됨).
5. 세 실행의 workdir이 서로 다르고 세 벌 로그가 모두 남아 있을 것.
6. digest 불변: v5/v6 고정 digest와 대조(`tools/exact_audit.py`).

**음성 대조는 이미 확보되어 있다.** `bd7a0d3` 구 바이너리의 실행이 기록으로
남아 있고 — 128 케이스 중 2~5개가 ~15.7 s에
`"error":"cudaGetLastError(): operation not permitted when stream is capturing"`,
`"slot":-1`로 사망 — 그것이 이 수정의 대조군이다. 다시 만들 필요는 없다.

**추가 스트레스(선택, 결함을 확률 1로 끌어올림):**
`RASBERY_GPU_CAPTURE_STALL_US=200000`으로 캡처 창을 넓히고 1회 실행.
`RASBERY_GPU_CAPTURE_ARBITER=0`과 짝지으면 구 동작(사망), 켜면 생존이어야 한다.
`RASBERY_GPU_CAPTURE_TRACE=1`은 `[RASBERY][CAPTURE]` 라인으로 창의 겹침을
스레드 단위로 보여준다.
