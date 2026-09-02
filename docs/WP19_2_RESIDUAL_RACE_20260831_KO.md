# WP19.2 — 남은 채널은 두 번째 빌더가 아니라 **두 개의 blocking 스트림**이었다

238, `0054838`(= WP19 + WP19.1), batch 8 프로세스 × M16 + MPS, 128-job manifest,
v6 env, 여섯 번의 런(block 38).
증거: `E:\rasbery_runs\2026-08-30\238\sigsegv_38\` (9개 파일),
`E:\rasbery_runs\2026-08-30\238\pricing_388e8f2.md` block (38).
선행: `docs/WP19_CAPTURE_RACE_20260831_KO.md`,
`docs/WP19_1_STANDUP_ISOLATION_20260831_KO.md`.

---

## 1. 증거 — 여섯 런의 캡처-경합 리시트 전수

`[MULTI_GPU][PROC]` 리시트의 `capture_arbiter` / `capture_race` 항목을 여섯 런 전부에서
파싱한 결과다. **한 줄도 빠뜨리지 않았다.**

| 런 | proc | cases/ok/failed | `capture_race_retry` | `abandoned` | 경합 이벤트 태그 | 결과 |
|---|---|---|---|---|---|---|
| v2_run1 | — | 128/128/0 | 1 | 0 | `ppr.while` | PASS |
| v2_run2 | p5 | 16/16/0 | 1 | 0 | `ppr.while` EndCapture(root) 901 tid0 | ok |
| v2_run2 | **p7** | 16/16/0 (재시작 후) | 1 | 0 | `ppr.while` … + evaluator 첫-케이스 retry(candidate_0056, lane 1, 복구) | **SIGSEGV** |
| v2_run3 | **p1** | 16/16/0 (재시작 후) | 0 | 0 | `ppr.while` EndCapture(root) 901 tid0 | **SIGSEGV** |
| v2_run3 | **p5** | 16/**15**/1 | **2** | 0 | `ppr.while` 901 **tid0**, evaluator retry(candidate_0057, lane 4, 복구), `ppr.while` 901 **tid1** | **non-finite: candidate_0060** |
| v2_run4/5, ctl_nov2 | — | 128/128/0 | 1 / 1 / 2 | 0 | `ppr.while` | PASS |

여기서 즉시 읽히는 네 가지.

1. **모든 경합 이벤트의 태그가 `ppr.while`이다.** 여섯 런, 다섯 건, 전부.
   `outer.while`은 **0건**. WP19.1이 게이트를 단 그 빌더는 이 캠페인에서 한 번도
   경합하지 않았다.
2. **`capture_race_abandoned`가 전 런 0이다.** 즉 WP19.1의 스테이지 게이트는
   **한 번도 발화하지 않았다**. WP19.1은 깨지지 않았다 — 관여한 적이 없다.
3. **오류 코드는 전부 901 `cudaErrorStreamCaptureInvalidated`, 스테이지는 전부
   `EndCapture(root)`.** "내 캡처를 남이 무효화했다"는 코드다.
4. **run3 p5의 두 번째 `ppr.while` 경합은 tid 1에서, 스탠드업이 아니라 런 도중에
   났다** (p5 evaluator 로그 line 1132; 첫 경합은 line 114). 그리고
   candidate_0060은 **case 8, lane 5** — 레인의 첫 케이스가 아니다.

### 1.1 세 사건의 귀속

| 사건 | 레인 첫 케이스? | 스탠드업 이후 경과 | 위상/커널 | 다른 레인 캡처 창 겹침 | 그 런에서 retry/abandoned 발화? |
|---|---|---|---|---|---|
| non-finite **candidate_0060** (run3 p5, case 8, lane 5, slot −1) | **아니오** | 런 중반 (로그 1132행 경합 → 1744행 실패) | `ppr.while` `EndCapture(root)` 901, tid 1 | `alloc_overlapped:0`, `alloc_blocked:2`, `alloc_in_capture:0` → **AllocWindow 겹침 아님** | retry **2**, abandoned 0 |
| **SIGSEGV** run2 p7 | 첫 청크, `completed:0` | proc wall 475.5 s − 재시작 후 uptime 287.4 s ≈ **~188 s**(첫 리필 경계) | 알 수 없음 (수신 없음) | 알 수 없음 | 그 런 retry 1(p5), 1(p7) |
| **SIGSEGV** run3 p1 | 첫 청크, `completed:0` | proc wall 350.6 s − 재시작 후 uptime 337.7 s ≈ **~10 s**(스탠드업) | 알 수 없음 (수신 없음) | 알 수 없음 | 그 런 retry 2(p5), 0(p1) |

두 SIGSEGV는 **애플리케이션 로그를 한 줄도 남기지 않았다**. 디스크에 있는
`gpu0.p7`/`gpu0.p1` evaluator 로그는 **재시작 후(attempt 2)** 것뿐이다.
`ulimit -c`=0, `core_pattern`은 systemd-coredump로 가는데 `coredumpctl`은 실행 계정이
읽을 수 없다(`core_pattern_238.txt`). **코어도, 스택도, 케이스 이름도 없다.**
harness의 재시작 레코드가 남긴 전부는 이것이다:

```
[RASBERY][MULTI_GPU][EVALUATOR][FATAL] {"gpu":"0","proc":7,"chunk":1,"attempt":1,
 "wave_id":101,"returncode":-11,"completed":0,"unfinished":[...16...],
 "requeued":true,"restarts":0}
```

`completed:0` 하나가 증거의 전부다. 복구는 2/2 완전했다(128/128, 127/128).

---

## 2. 채널 — `file:line`

### 2.1 1차 채널 (근본 원인): **legacy-blocking 스트림 위에서 캡처**

- `src/CudaPprBackend.cu:1570` — `cudaStreamCreate(&stream)` → PPR 백엔드의 작업
  스트림. 이것이 WHILE의 **body_stream**이다
  (`buildPprWhile`의 `cudaStreamBeginCaptureToGraph(body_stream, …)`,
  `src/CudaPprBackend.cu:1185`).
- `src/CudaPprBackend.cu:2147` — `cudaStreamCreate(&s.graph_root_stream)` → WHILE의
  **root_stream** (`cudaStreamBeginCapture(root_stream, …)`,
  `src/CudaPprBackend.cu:1114`).
- (동반) `src/CudaCramBackend.cu:700` — CRAM 백엔드 스트림. 캡처 대상은 아니지만
  트리에 남은 세 번째 blocking 스트림.

`cudaStreamCreate`는 플래그 `cudaStreamDefault`, 즉 **legacy NULL 스트림과 프로세스
전역으로 암묵 동기화하는** 스트림을 만든다. 그 스트림이 캡처 중일 때 프로세스 안
**아무 스레드든** 기본 스트림에 작업을 넣으면, CUDA는 그 호출자에게
`cudaErrorStreamCaptureImplicit`을 주고 **캡처를 무효화**한다. 무효화는 캡처한
쪽에서 `cudaStreamEndCapture`가 **901 `cudaErrorStreamCaptureInvalidated`** 로
드러난다 — 관측된 다섯 건의 **코드와 스테이지가 정확히 그것이다**.

이 트리의 나머지 스트림은 **전부 이미 `cudaStreamNonBlocking`이다**:
`CudaOuterGraph.cu:967`, `CudaOuterGraph.cu:2894`(outer.while root),
`CudaXsReconBackend.cu:1266`, `CudaXsReconBackend.cu:2845`,
`GpuPhysicsArenaCuda.cu:136`, `CudaBICGBackend.cu:4081`.
**경합 태그가 `ppr.while` 하나뿐이고 `outer.while`이 0건인 이유가 이 플래그 차이다.**
그리고 중재자(`GpuCaptureArbiter.h`)는 이것을 막을 수 없다 — **NULL 스트림 커널
런치는 `AllocWindow` 사이트가 아니다**. 리시트가 `alloc_in_capture:0`,
`captures_unwound:0`이면서도 경합이 계속 나는 것이 그 증거다.

### 2.2 2차 채널 (벨트 구멍): 첫-케이스 전용 게이트

- `src/EvaluatorServer.h:2180`(WP19.1) — `if (status == 0 || !lane_first_case || …) return;`

WP19.1의 근거는 "그래프 캐시는 슬롯당·프로세스 수명이므로 빌드가 일어나는 케이스는
레인의 첫 케이스뿐"이었다. **PPR WHILE에는 성립하지 않는다.** `s.graph_valid`는
`PprBackend`에, `PprBackend`는 `Driver`에, `Driver`는 **한 케이스**에 산다.
그래서 block 38의 모든 `[RASBERY][PPR_GPU]` 리시트가 케이스마다
`"graph_builds":1`을 찍는다 — 레인당이 아니라 **케이스당**이다.
따라서 빌드 창은 매 케이스 열리고, candidate_0060(case 8, lane 5)은 벨트가
**구조적으로 볼 수 없는** 자리에서 죽었다. 실제로 재시도되지 않았다.

### 2.3 판정: (a)이며, (a)보다 한 겹 아래

문제의 선택지 중 **(a) "두 번째 빌드 경로(PPR WHILE)가 게이트 없이 재시도한다"** 가 맞다.
다만 진짜 원인은 재시도의 부재가 아니라 **그 빌더가 유일하게 blocking 스트림 위에서
캡처한다는 것**이다. (`ppr.while`의 재시도 자체는 WP19.1의 논거대로 무해하다:
`enqueue_round`(`src/CudaPprBackend.cu:2044-2071`)는 커널 런치만 하고 업로드가 없다 —
소스로 재확인했다.)

**(b) `NodalArena::adoptCanonical`은 배제하지 않는다.** 이 캠페인의 로그로는
증명도 반증도 되지 않는다(해당 경로는 리시트를 내지 않는다). 실재하는 데이터 레이스는
맞으므로 **패치 파일로 남기고 계수기를 붙였다** — §4.

**두 SIGSEGV는 귀속하지 않는다.** 로그가 0줄이고 코어가 없으므로 어떤 귀속도
추측이다. 대신 **다음 SIGSEGV가 같은 침묵을 반복하지 못하게** 만들었다 — §3.3.

---

## 3. 고친 것

### 3.1 blocking 스트림 제거 (근본 원인)

| 파일 | 변경 |
|---|---|
| `src/CudaPprBackend.cu:1570` | `cudaStreamCreate(&stream)` → `cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)` |
| `src/CudaPprBackend.cu:2147` | `cudaStreamCreate(&s.graph_root_stream)` → `cudaStreamCreateWithFlags(…, cudaStreamNonBlocking)` |
| `src/CudaCramBackend.cu:700` | 동일 |

트리에 legacy-blocking 스트림은 **0개**가 되었다. 계약 시험이 `src/` 전체를 훑어
`cudaStreamCreate(`가 한 건이라도 남으면 실패한다.

### 3.2 벨트를 "첫 케이스"에서 "경합을 가로지른 케이스"로

| 파일 | 변경 |
|---|---|
| `src/GpuCaptureArbiter.h` | `captureRaceEvents()` — retry + abandoned + unrecovered의 프로세스 전역 합 |
| `src/EvaluatorServer.h` (wave 경로, rolling 경로) | 케이스 전후로 그 값을 스냅샷 → `race_spanned` |
| `src/EvaluatorServer.h` `retryAfterCaptureRace` | 게이트가 `(!lane_first_case && !race_spanned)` |
| 리시트 | `lane_first_case`가 상수 `true`가 아니라 사실이 되고, `race_spanned`가 추가됨 |

**대체가 아니라 OR이다.** 조용한 프로세스에서 델타는 항상 0이므로, 이 확장이
물리적 실패를 "운 좋은 재실행"으로 세탁할 수 없다 — WP19.1이 첫-케이스 규칙으로
지키려던 성질을 **대리 지표 대신 측정**으로 지켰다.

### 3.3 SIGSEGV를 시끄럽게 (코어 없이)

새 파일 `src/CrashReport.h`(헤더 온리, CUDA-free, POSIX 선택):

- `SIGSEGV/SIGBUS/SIGFPE/SIGILL/SIGABRT`에 `sigaction` 설치.
  설치 지점은 **worker의 진입점** `EvaluatorServer.h Server::run()` 맨 앞 —
  `main.cpp`가 잠겨 있어서가 아니라, dispatcher가 띄우는 모든 프로세스가
  `--evaluator-jsonl`로 여기 들어오기 때문이다.
- 핸들러가 찍는 것:
  ```
  [RASBERY][CRASH] {"signal":11,"name":"SIGSEGV","pid":…,"tid":…,
   "capture_open":1,"thread_capturing":0,"capture_race_events":2,
   "lanes":[{"lane":5,"case":8,"slot":3,"phase":"drive","deck":"…candidate_0060.json"}]}
  [RASBERY][CRASH][FRAME] begin
  …backtrace_symbols_fd…
  [RASBERY][CRASH][END] {"signal":11,"frames":23}
  ```
  block 38이 원했고 얻지 못한 네 가지(**case / lane / slot / phase**)가 앞 네 필드다.
  `capture_open`·`capture_race_events`는 "죽는 순간 어딘가에서 캡처가 열려 있었나"라는
  WP19의 미결 질문에 답한다.
- **async-signal-safe**: 핸들러 안에서 `write(2)`, `backtrace`,
  `backtrace_symbols_fd`, `sigaction`, `getpid`, `raise`만 호출한다. `std::ostream`·
  `std::string`·`printf` 계열·`new`/`malloc` 없음. 정수는 스택 버퍼에 역순 자리수
  루프로 렌더한다. `backtrace()`는 **설치 시점에 한 번** 예열해서 로더 해석과 최초
  할당이 핸들러 밖에서 끝나게 했다.
- **re-raise, not exit**: 기본 처리로 되돌리고 `raise(sig)` — dispatcher가 보는
  `returncode:-11`도, 코어를 쓸 수 있는 호스트의 코어도 그대로다. 증거만 늘고
  결과는 바뀌지 않는다.
- 빵부스러기는 **레인별 락-프리 고정 테이블**. 레인은 자기 행만 쓰고, `active`가
  마지막에 켜지고 처음에 꺼져서 반쯤 쓰인 행은 "비활성"으로 읽힌다.
  슬롯은 evaluator가 모르므로(그래서 `[ERROR]` 리시트가 `slot:-1`이다)
  `src/CudaOuterGraph.cu:947`의 `initialize()`가 찍는다.
- `RASBERY_CRASH_REPORT=0`으로 끌 수 있다 — 코어가 되는 호스트를 위한 A/B.

### 3.4 harness가 죽은 자식의 마지막 말을 보관

`tools/run_multi_gpu_batch.py`: `crash_evidence(text)`가 FATAL 레코드에
`stderr_tail`(마지막 120줄), 파싱된 `crash` 레코드, `backtrace`를 붙인다.
`stderr=subprocess.STDOUT`이라 핸들러의 fd 2 출력이 이미 같은 파이프로 오므로
**새 배선은 없다**. `[CRASH]` 줄이 아예 없는 죽음(OOM 킬 등)에도 `stderr_tail`은
남는다 — block 38을 재현 불가로 만든 것이 정확히 그 경우다.

---

## 4. 패치 파일로 남긴 것 — `NodalArena::adoptCanonical`

`docs/patches/wp19_2_adopt_canonical.patch` (unified diff).
`src/CudaXsReconBackend.cu`는 동시 진행 중인 WP20 FP32 arm이 잡고 있어 **커밋하지 않는다.**

- `adoptCanonical()`(0054838 기준 `:1503`)이 `_canon[slot]`과 `_views_dirty`를
  `_mutex` **밖에서** 쓴다 — 클래스에서 유일하게 잠금 밖인 접근이다.
- `refreshViews()`(`:1479`)가 같은 두 상태를 읽고 플래그를 내린다.
  호출자는 `launchBatch()`이고, `drive()`가 `lock.unlock()` **뒤에** 부르므로
  (`:1042-1044`) 잠금을 새로 잡아도 재귀 데드락이 아니다 — 패치가 그렇게 한다.
- 두 얼굴: **분실된 adoption**(refreshViews가 옛 `_canon`을 밀고 플래그를 내려버려
  슬롯이 아레나 자체 flux/jnet/phis를 그대로 쓰는데 호스트는 canonical을 쓴다고
  믿는다 → **CUDA 오류 없는 non-finite**)과 **찢어진 읽기**(새 `flux` + 옛 `jnet`
  = 서로 다른 outer의 포인터 쌍; 뷰 포인터가 읽는 중에 바뀌면 **SIGSEGV**도 그럴듯하다).
- 계수기 `[RASBERY][NODAL_ARENA] {"adopt_races_detected":N,"canon_locked":1}` 를
  프로세스 종료 시 **무조건** 찍는다 — `0`도 진술이 되도록.

적용:

```bash
git apply --3way docs/patches/wp19_2_adopt_canonical.patch
```

(`git apply --check --3way`로 WP20 커밋 `ddd0ccc` 시점 트리에서 clean 적용 확인함.)

---

## 5. 계약 시험

| 시험 | 결과 |
|---|---|
| `tools/test_capture_standup_isolation_contract.py` (규칙 7·8 추가, 음성 대조 3개 추가) | **PASSED** (8 checks, 9 controls) |
| `tools/test_crash_report_contract.py` (신규) | **PASSED** (4 checks, 6 source controls, 2 harness controls) |
| `tools/test_capture_arbiter_contract.py` | PASSED |
| `tools/test_enum_alias_contract.py` | PASSED |
| `tools/test_dependent_template_contract.py` | PASSED |
| `tools/test_evaluator_rolling_contract.py` | PASSED |

새 규칙과 그 음성 대조:

- **규칙 7 — `src/` 전체에 blocking 스트림 0개.** 대조: PPR WHILE root를 다시
  `cudaStreamCreate`로 되돌리면 잡힌다.
- **규칙 8 — 벨트가 첫-케이스 전용이 아니다.** 대조 2개: 게이트를 통째로 없애면
  잡히고, **첫-케이스 전용으로 되돌려도 잡힌다**(WP19.2 자신의 회귀).
- 크래시 계약의 대조 6개: SIGSEGV를 무장 집합에서 빼기 / worker가 설치를 멈추기 /
  핸들러가 `ostringstream`을 쓰기 / 레코드가 case를 안 찍기 / 재발생 대신 `_exit` /
  한쪽 레인 루프만 빵부스러기를 남기기.

**선재(先在) 실패, 이 작업과 무관:** `test_cmfd_slot_compaction_contract.py`,
`test_cram_gpu_contract.py`, `test_ppr_gpu_contract.py` 세 개는 `0054838`과
`ddd0ccc`(WP20) 양쪽에서 **동일한 메시지로 이미 실패한다** — 임시 worktree로
양쪽 커밋에서 직접 확인했다. 이 커밋이 만든 실패가 아니다.

---

## 6. 런북 — 다음 런이 무엇을 증명해야 하는가

빌드: 이 커밋에서 fresh worktree + `ctest`(12/12 기대).

```bash
# 238, GPU0, v6 env + RASBERY_GPU_OUTER_SEGMENT_V2=1, MPS auto, 128-job manifest
for i in 1 2 3 4 5; do
  python tools/run_multi_gpu_batch.py \
      --gpus 0 --procs-per-gpu 8 --batch-width 16 \
      --jobs-manifest <128-job manifest> \
      --worker-shape evaluator --evaluator-max-restarts 3 \
      --mps auto --claim auto \
      > ~/block39/batch_wp192_run$i.log 2>&1
done
```

**합격 기준 (5런 전부):**

1. `grep -c "non-finite" ` 한 건도 `[EVALUATOR][ERROR]`로 이어지지 않을 것.
   즉 `evaluator_totals`의 `failed`가 **전 프로세스 0**.
2. `returncode:-11` **0건** — SIGSEGV 0.
3. `[RASBERY][CUDA][CAPTURE_RACE][RETRY]` **0건**을 기대한다(§2.1이 맞다면
   `ppr.while` 901은 사라진다). 남는다면 §2.1이 틀렸다는 뜻이므로 **태그와 tid를
   그대로 보고할 것** — 이 항목은 합격/불합격이 아니라 가설 검정이다.
4. c/h가 1,320±10 대역을 유지할 것(block 37/38의 8×M16 기준 1,320.8 / 1,322–1,327).
   `cudaStreamNonBlocking`은 순서 보장을 **줄이지 않고** NULL 스트림과의 암묵 결합만
   없애므로 처리량 변화는 없어야 한다.

**만약 SIGSEGV가 그래도 난다면**, 합격 조건은 하나 더 있다:

5. 그 FATAL 레코드에 `crash` / `backtrace` / `stderr_tail`이 **들어 있을 것.**
   비어 있다면 §3.3이 실패한 것이고, 그때는 crash 레코드가 아니라 crash 레코드의
   부재가 다음 작업 항목이다. 들어 있다면 **처음으로** 어느 케이스·레인·슬롯·위상에서
   죽었는지 알 수 있다.

한 줄 확인:

```bash
grep -h "EVALUATOR\]\[FATAL\]" ~/block39/batch_wp192_run*.log | python -c "
import json,sys
for line in sys.stdin:
    r=json.loads(line.split('] ',1)[1])
    print(r['proc'], r['returncode'], 'crash' in r, len(r.get('backtrace',[])))
"
```

---

## 7. 남은 것

- `adoptCanonical` 패치는 WP20 이후 적용 — 적용한 첫 8×M16 런의
  `[NODAL_ARENA] adopt_races_detected`가 0이 아니면, 잠금 전의 트리는 그 횟수만큼
  adoption을 잃을 기회가 있었다는 뜻이다.
- 두 SIGSEGV는 **여전히 미귀속**이다. 이 커밋은 원인을 주장하지 않고 다음 사례의
  침묵만 없앤다.
- `ppr.while`의 재시도는 게이트 없이 남겨 두었다. `enqueue_round`가 업로드를 갖지
  않는 한 안전하고(소스 확인함), 게이트를 달면 지금 회복되는 경로가 stream arm으로
  영구 폴백한다. `CudaPprBackend.cu:2189`의 주석이 그 조건을 명시하고 있고,
  계약 시험이 그 주석의 존재를 지킨다.
