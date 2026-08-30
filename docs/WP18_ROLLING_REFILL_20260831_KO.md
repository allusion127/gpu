# WP18 — 슬롯 단위 즉시 refill (`RASBERY_EVALUATOR_ROLLING`)

238에서 측정된 것: 8×M8 = **1,182 c/h**, `width_fill 0.447`, 390 s wall 중
`tail_idle_max 155 s`. 8×M16 = 1,235 c/h(+4.5 %), `width_fill 0.367`,
373 s 중 356 s가 tail. `--claim 1/2/4`는 `width_fill`을 0.125/0.18/0.26으로,
c/h를 593/840/978로 무너뜨린다. sm%는 96 %이고 CMFD 커널은 케이스당 34 블록만
띄운다 — 즉 **살아 있는 케이스를 더 많이 담는 것**이 처리량을 올리는 유일한
레버이며, 위 숫자들은 GPU가 아니라 **잡 리스트를 자르는 방식**의 결과다.

이 문서는 (a) lockstep 판정, (b) 오늘 슬롯이 일찍 끝나면 무슨 일이 일어나는가,
(c) 스트리밍에 필요한 프로토콜을 file:line으로 확정하고, 그 위에서 구현한 것과
238 런북을 적는다.

---

## 1. lockstep 분석 (a)

### 1.1 결론: 아레나는 **슬롯별 비동기**다. lockstep이 아니다.

근거는 네 개이고, 전부 코드에 있다.

| 사실 | 위치 |
|---|---|
| 슬롯 A가 Outer일 때 슬롯 B가 Depletion인 것이 **설계 전제**다 | `src/GpuPhysicsArena.h:36-39` |
| 한 슬롯은 한 호스트 스레드(=한 Driver=한 덱)가 케이스 전체 동안 소유한다. statepoint 루프·T/H 루프·붕소 탐색이 전부 그 스레드 안에 있다 | `src/EvaluatorServer.h:1409-1432` (wave), `src/main.cpp:1393` (batch) |
| 슬롯 간 유일한 결합은 **런치 단위 랑데부**다. CMFD solve에 도착한 스레드가 열려 있는 배치에 합류하고, 선출된 launcher는 `pending.size() < inUseCount()`인 동안만 linger한다 — 즉 "지금 **점유 중인** 슬롯"을 기다릴 뿐 phase도 statepoint도 wave도 기다리지 않는다 | `src/CudaBICGBackend.cu:6907, 6911` |
| 슬롯을 놓으면 `inUseCount()`가 즉시 떨어지고 `cv.notify_all()`이 linger 중인 launcher를 깨운다 — 빠진 슬롯을 영원히 기다리지 않는다 | `src/CudaBICGBackend.cu:6452-6466` |

**한 런치의 참가자들이 공유해야 하는 유일한 값**은 inner BiCGSTAB 예산 `nmax`
(`RASBERY_BICG_NMAX`)이고, 이는 덱별 환경 상수이지 스케줄상의 위치가 아니다
(`src/CudaBICGBackend.cu:6944`). `nmax`가 다르면 배치가 통째로 실패하도록
보고된다.

공유 statepoint 그리드도, 공유 T/H 루프도, 공유 탐색도, cohort 전역
xslib digest 잠금도 **런치 경로에는 없다**. `CohortContext`/`CaseKey`는
(geometry, library) 쌍의 **캐시 키**이지 wave의 동기화 단위가 아니다 —
`cohort_builds`가 케이스 수와 함께 자라지 않는다는 것이 그 증거이고
(`src/EvaluatorServer.h` 프로세스 수신증), 그래서 서로 다른 statepoint에 있는
두 슬롯이 같은 cohort를 공유하는 것이 정상 동작이다.

### 1.2 그렇다면 배리어는 어디에 있는가

두 곳, 둘 다 호스트 측이다.

1. **`runWave`의 암묵 OpenMP 배리어.** `src/EvaluatorServer.h:1409`의
   `#pragma omp parallel for schedule(dynamic, 1)`은 **고정된 잡 벡터** 위를 돈다.
   `njobs > host_threads`이면 dynamic 스케줄이 이미 레인 단위 refill을 한다 —
   즉 refill 자체는 이미 존재한다. 문제는 `njobs`가 고정이라는 것이고, wave가
   끝날 때까지 새 케이스가 들어올 수 없다는 것이다.
   레인 수도 `host_threads = std::min(width, njobs)`로 **청크 크기에 캡**된다
   (`src/EvaluatorServer.h:1354`).

2. **디스패처의 청크 claim.** `auto_claim_size()`
   (`tools/run_multi_gpu_batch.py:879-905`)는
   `min(fair_share, max(batch_width, ceil(remaining / (2·processes))))`이다.
   128잡 / 8워커 / width 8 → `fair_share=16`, `halved=max(8, 8)=8` → **claim = 8 = width**.
   따라서 **wave마다 레인당 정확히 1케이스**이고, refill은 한 번도 일어나지 않으며,
   wave는 가장 느린 케이스가 끝날 때 끝난다. M16에서는 claim=16=width로
   같은 구조가 더 나쁘게 나타난다(tail 356/373 s).
   `--claim 1`에서 `width_fill 0.125 = 1/8`인 것이 이 해석의 직접 증거다:
   njobs=1이면 `host_threads=min(8,1)=1`, 즉 아레나의 8슬롯 중 1개만 선다.

**결론(a): lockstep 커널·런치는 없다. 0.447의 width_fill과 155 s의 tail은
전적으로 호스트 측 wave 경계의 산물이다.** 따라서 half-width 이중 버퍼링이나
statepoint 경계 refill 같은 우회는 불필요하며, 과제가 지정한
"아레나가 이미 phase-asynchronous이면 `RASBERY_EVALUATOR_ROLLING=1`" 경로를 그대로 간다.

### 1.3 오늘 슬롯이 일찍 끝나면 (b)

- Driver가 파괴되고 `releaseSlot`이 `taken[m]=0`, `slot[m].in_use=false`,
  `cv.notify_all()`을 한다 (`src/CudaBICGBackend.cu:6452-6466`).
- `inUseCount()`가 줄어들어 **다음 런치의 랑데부 목표가 즉시 좁아진다.**
  즉 빈 슬롯이 남은 케이스들을 붙잡지는 **않는다**.
- 그러나 그 레인은 wave의 `omp for`에 남은 반복이 없으면 **암묵 배리어에서 논다.**
  claim==width인 오늘의 설정에서는 항상 그렇다. 이것이 `tail_idle_max`다.
- `RASBERY_GPU_CMFD_COMPACT`(기본 OFF, `src/CudaBICGBackend.cu:625-628`)는
  **다른 문제**를 푼다: 이미 점유된 슬롯들만 blockIdx.y로 압축해 런치의 padding
  블록을 없앤다. 슬롯을 새 케이스로 채우지 않는다. 그래서 이 tail에 대해
  compaction은 아무것도 하지 않는다.
- `[RASBERY][REFILL]` 수신증(`src/BatchRefill.h`)의 `refills`는
  `jobs - lanes_used`이며, claim==width인 런에서는 정확히 **0**이다.

### 1.4 스트리밍에 필요한 프로토콜 (c)

wave 모드에서 `op:"case"` 줄은 **모아두기만** 하고 `op:"wave"` 줄이 와야 실행된다
(`src/EvaluatorServer.h` run 루프의 `pending`). 그래서 디스패처는 청크를 보내고
**그 청크의 수신증을 기다리는 동안 아무것도 더 보낼 수 없다.** 스트리밍에 필요한 것은
정확히 두 가지다.

1. **evaluator 측**: `case` 줄이 도착한 자리에서 큐에 넣고 자유 레인이 집어간다.
   그러려면 요청 스트림을 읽는 주체와 케이스를 도는 주체가 **분리**돼야 한다.
2. **디스패처 측**: 완료 수신증을 읽는 **루프 안에서** 다음 claim을 보낼 수 있어야 한다.
   `EvaluatorSession.wave()`는 "보내고 → 배리어까지 읽기"로 굳어 있어 불가능했다.

---

## 2. 구현한 것

### 2.1 evaluator (`RASBERY_EVALUATOR_ROLLING=1`)

- **게이트**: `detail::rollingEnabled()` — 환경변수를 함수 지역 static으로 **한 번만** 읽는다.
  `detail::rollingQueueCapacity()`(기본 레인당 4), `detail::rollingPrefetch()`(기본 2).
- **`RollingQueue`** (`src/EvaluatorServer.h`): bounded MPSC 큐 + **출력 경로당 한 명의 tenant**.
  - `pop()`은 큐가 비어 있어도 **닫히지 않았으면 기다린다.** "클라이언트가 아직 안 보냈다"와
    "런이 끝났다"를 구분하는 것이 이 모드의 전부다.
  - `firstAdmissible()`은 `--raso`가 in-flight인 잡을 건너뛴다. **거부도 폐기도 아니고 대기** —
    wave 경계가 우연히 제공하던 직렬화를 명시적으로 만든 것이다.
- **레인**: `rollingOpen()`이 폭을 latch하고(`min(RASBERY_BATCH_HOST_THREADS, width)`,
  **청크 캡 없음**) 헬퍼 스레드에서 `#pragma omp parallel num_threads(lanes)`를 연다.
  요청 스트림을 읽는 주 스레드는 계속 읽는다. 레인은 OpenMP 스레드다 — `OMP_PROC_BIND`,
  `RASBERY_OMP_THREADS`, taskset cpu-list가 배치하는 것이 OpenMP 스레드이기 때문.
- **admission = 기존 문 그대로**: `rollingLane()`은 `runOneCase()`를 부르고,
  그 안의 Driver 스코프가 슬롯을 놓는다 → `acquireSlot()`이 `Slot{}` 전체 리셋,
  `batchSlotIsReset()` 감사, `admissions++`를 한다
  (`src/CudaBICGBackend.cu:6409-6450`). **rolling 전용 admission 경로는 없다.**
- **epoch**: `refill::RollingLedger::admit()`이 락 아래에서 `++_epoch`를 발급하고
  live 목록에 넣는다. `finish()`는 자기 epoch를 회수하며, 회수 실패는
  `epoch_regressions++`. 레인이 이미 점유 중인데 또 admit하면
  `stale_tenant_refusals++`.
- **배리어**: `op:"wave"`는 rolling에서 **배리어 요청**이다. 큐를 닫고 레인을 join한 뒤
  `[BATCH_HOST]` → (isolation 재검) → `[REFILL]` → `[REFILL][ROLLING]` →
  `[EVALUATOR][WAVE]` → `[MEM]`을 wave 경로와 **같은 태그·같은 순서**로 찍는다.
  그 뒤 큐를 다시 열어 다음 세션이 같은 프로세스·같은 아레나 위에서 시작한다.
- **수신증**: `[RASBERY][REFILL][ROLLING] {session, arena_width, lanes, lanes_used,
  queue_capacity, admits, immediate_admits, wave_barriers_avoided, slot_idle_ms_total,
  tail_idle_ms, wall_s, width_history{samples,p10,p50,p90,mean,max}, width_fill,
  admit_wait_ms{p50,max}, epoch, epoch_regressions, stale_tenant_refusals,
  live_tenancies_at_close}`.

### 2.2 왜 `[RASBERY][REFILL]`에 키를 더하지 않았나

`REFILL_RECEIPT = re.compile(r"\[RASBERY\]\[REFILL\]\s*(\{.*\})")`
(`tools/run_multi_gpu_batch.py:124`)와 그 위에 세워진 캠페인 표들이 **그 줄**을 읽는다.
WP18의 게이트 중 하나가 "플래그 OFF면 byte-identical"이므로 그 줄에 키를 더하는 것은
곧 모든 wave-mode 런의 stdout을 바꾸는 것이다. 그래서 **두 번째 줄**을 쓴다.
기존 정규식은 태그 직후 `{`를 요구하므로 새 줄을 매치하지 못한다(계약 테스트가 양방향으로 확인).

### 2.3 feature-off 동일성 — 무엇이 그것을 보장하는가

- `runWave` 본문에 `_roll`/`_queue`/`rolling`/`Rolling` 토큰이 **하나도 없다**
  (계약 테스트가 brace-매칭으로 본문을 잘라 확인).
- rolling이 찍는 모든 줄은 rolling 전용 함수 안에 있다. 유일한 예외인 READY 수신증의
  `rolling`/`rolling_prefetch`/`rolling_target_inflight` 필드는
  `if (detail::rollingEnabled())` 안에 있다.
- `refuse()`에 `_out_mutex`가 추가됐다(rolling에서 리더 스레드가 레인과 동시에 쓰므로).
  wave 모드에서는 경합이 없어 **출력 바이트는 동일**하고 순서만 보장된다.
- wave `[REFILL]` 줄의 키 목록을 계약 테스트가 **순서까지** 고정한다.

### 2.4 per-case exactness

케이스가 계산하는 것은 **아무것도 바뀌지 않았다**. rolling 레인은 wave 레인과 같은
`runOneCase()`를 같은 인자로 부르고, 같은 `acquireSlot()` 문으로 슬롯을 얻는다.
바뀐 것은 잡 리스트가 벡터냐 큐냐 하나뿐이다. 게이트는 §4의 digest 비교와
`cross_case_digest_mismatch` 0, `slot_stale_tenants` 0, `rolling`의
`stale_tenant_refusals`/`epoch_regressions`/`live_tenancies_at_close` 0이다.

### 2.5 디스패처 `--claim rolling`

- 자식 환경에 `RASBERY_EVALUATOR_ROLLING=1`을 넣고(`[MULTI_GPU][ENV]` 수신증에 남는다),
  `--evaluator` 없이는 거부한다.
- READY 수신증이 `"rolling":true`가 아니면 **즉시 실패**한다. WP18 이전 바이너리에
  스트리밍을 하면 케이스 하나를 보내고 영원히 기다리다 hang으로 죽는다 — 그 죽음은
  "wave arm의 숫자가 rolling arm 이름을 달고 발표되는" 사고의 앞단계다.
- `queue.claim(1, index)` — **한 번에 하나**. 청크 claim의 이유는 프로세스 기동 비용
  상각인데 이 모드는 그것을 내지 않는다. 하나씩 claim해야 마지막 잡까지 다른 워커가
  훔칠 수 있고, 그것이 `--claim auto`가 `--claim all`을 이긴 이유다.
- `len(outstanding) < target`일 때만 claim한다(`target = width + prefetch`).
  더 claim하면 큐의 tail이 한 워커의 손 안으로 옮겨갈 뿐이다.
- **완료 콜백 안에서 재claim한다.** `_pump`의 `done(line)` 안에서 `top_up()`을 부른다.
  배리어 사이에서만 채우는 top-up은 그냥 wave다.
- 재시작 없음(`evaluator_max_restarts=0`): rolling 세션의 outstanding은 아레나 전체에
  퍼져 있어 한 번의 죽음이 최대 `width+prefetch` 케이스를 잃는다. 그만한 집합을
  방금 죽은 프로세스에 다시 넣느니 claim을 멈추고 나머지를 살아 있는 워커에게 넘긴다.
- `[MULTI_GPU][PROC]` 수신증에 `rolling{admits, immediate_fraction, width_fill,
  tail_idle_s, wave_barriers_avoided, slot_idle_ms_total}`가 붙는다(rolling일 때만).
- 캠페인 게이트: `stale_tenant_refusals` / `epoch_regressions` /
  `live_tenancies_at_close` 합이 0이 아니면 문제로 보고한다.

### 2.6 rolling에서만 다른 두 가지 (의도적)

1. **`--raso` 규칙**: wave는 "한 wave 안에서 중복이면 wave 전체 거부"였다. rolling은
   manifest 안의 중복은 같은 규칙으로 거부하고, 스트리밍된 `case` 줄끼리는
   **동시 점유만 금지**한다(직렬화). 세대 간 output 재사용은 정당하고(승격 elite 재평가),
   실제 위험은 "동시에 두 Driver"뿐이기 때문.
2. **fidelity 기본값**: wave 줄의 fidelity는 **먼저 온 case 줄들의 미지정 필드를 채우는
   기본값**이다. rolling에서 case는 읽힌 자리에서 시작하므로 소급 적용이 불가능하다.
   이미 admit된 케이스가 있는데 fidelity 필드를 단 wave 줄은
   `rolling_wave_fidelity_after_admit`으로 **거부**한다.

### 2.7 명명된 거부

| 이름 | 지키는 성질 |
|---|---|
| `rolling_batch_width_latched` | 아레나는 첫 admission에서 고정된 단일 할당이다 |
| `rolling_wave_fidelity_after_admit` | 이미 도는 케이스에 fidelity 선언을 소급 적용하지 않는다 |
| `rolling_queue_closed` | 케이스를 조용히 떨어뜨리지 않는다 |

---

## 3. 기대 효과 (모델)

wave 모드에서 워커 하나의 wall은 `Σ_chunks max(케이스 시간)`이다. 128잡 / 8워커 /
claim 8이면 워커당 청크는 2개(8+8), 전체 워커 8개 → 캠페인은 16개 wave 배리어.
`width_fill 0.447`은 "선언된 8슬롯 중 평균 3.6개만 랑데부에 모였다"는 뜻이다.

슬롯 단위 refill이면 레인은 큐가 마르기 전까지 쉬지 않는다. 남는 idle은
(i) Driver teardown + 다음 덱 import(= `refill_latency`, 기존 수신증에서 p50 수 백 ms),
(ii) 세션 마지막의 **케이스 하나짜리** drain이다.

- `width_fill` → **~0.9** (레인이 마지막 케이스 하나를 제외하고 항상 점유).
- `tail_idle_max` → **한 케이스의 duration** (390 s wall / 16잡·워커 ≈ 25~50 s 수준),
  155 s에서 내려온다.
- c/h 예측: 1,182 × (0.9 / 0.447) 는 **상한**이고 정직하지 않다. sm%가 이미 96 %이므로
  스케줄링이 회수할 수 있는 것은 **tail과 빈 슬롯이 만든 유휴분**뿐이다. 정직한 모델은
  wall 기준이다: 390 s 중 tail_idle_max 155 s → 실효 wall이 대략
  390 − (155 × 유휴 레인 비율) 로 줄어든다. 관측된 `width_fill` 비 0.9/0.447 = 2.01을
  그대로 믿으면 2,380 c/h이지만, GPU가 96 % 바쁜 상태에서 살아 있는 케이스를 2배로
  담는다고 커널 처리량이 2배가 되지는 않는다(WP17: CMFD 34블록/케이스 → 더 담을수록
  좋지만 선형은 아니다).
  **예측 밴드: 1,500~1,900 c/h (M8), 하한 1,300.** 밴드 아래면 GPU가 벽이라는
  뜻이고(그때 `immediate_fraction`은 ~1.0이어야 한다), `immediate_fraction`이 1보다
  뚜렷이 작으면 벽은 GPU가 아니라 **prefetch**다 — `RASBERY_EVALUATOR_ROLLING_PREFETCH`를 올린다.
- M16은 wave 모드에서 tail이 95 %(356/373 s)였으므로 **회수량이 더 크다.** rolling에서
  M16이 M8을 넘어서면(현재는 +4.5 %뿐) 그것이 "M16의 이득이 tail에 먹히고 있었다"는 증거다.

---

## 4. 238 런북

전제: v6 env, MPS 기동, 8 프로세스 × M8(그리고 M16), 128잡 manifest,
`tools/run_multi_gpu_batch.py`. 로컬 계산 금지 — 238에서만. 출력은 `E:` 아래.

### 4.1 대조군 (wave, 기존)

```
python tools/run_multi_gpu_batch.py \
  --jobs <128job.manifest> --batch-width 8 --procs-per-gpu 8 --gpus 0 \
  --mps --claim auto --result light --workdir <E:/.../wp18/wave_m8>
```
M16은 `--batch-width 16`. 기록: `[MULTI_GPU][TOTAL]`의 `cases_per_hour`,
`[PROC]`의 `width_fill`, `[GPU]`의 `tail_idle_max_s`,
`[RASBERY][REFILL]`의 `refills`/`slot_busy_fraction`,
`[EVALUATOR]`의 `slot_stale_tenants`/`slot_duplicates`.

### 4.2 처리군 (rolling)

```
python tools/run_multi_gpu_batch.py \
  --jobs <128job.manifest> --batch-width 8 --procs-per-gpu 8 --gpus 0 \
  --mps --claim rolling --result light --workdir <E:/.../wp18/roll_m8>
```
`--claim rolling`이 자식 환경에 `RASBERY_EVALUATOR_ROLLING=1`을 넣는다
(`[MULTI_GPU][ENV]`에서 확인할 것 — 없으면 그 런은 대조군이다).
prefetch를 바꾸려면 `--set RASBERY_EVALUATOR_ROLLING_PREFETCH=<n>` 을 쓰고, 디스패처와 evaluator가 **같은 값**을 읽는지
`[EVALUATOR][READY]`의 `rolling_target_inflight`로 확인한다.

M16도 같은 방식으로. 총 4런(M8/M16 × auto/rolling).

### 4.3 각 런에서 기록할 것

| 항목 | 어디서 |
|---|---|
| `cases_per_hour` | `[MULTI_GPU][TOTAL]` |
| `width_fill` (랑데부) | `[MULTI_GPU][PROC].width_fill` |
| `width_fill` (점유) | `[REFILL][ROLLING].width_fill` — rolling만 |
| `tail_idle_max_s` | `[MULTI_GPU][GPU].tail_idle_max_s`, rolling은 `[PROC].rolling.tail_idle_s` |
| REFILL 수신증 | `[RASBERY][REFILL]` 전문 |
| ROLLING 수신증 | `[RASBERY][REFILL][ROLLING]` 전문 |
| `immediate_admits / admits` | `[PROC].rolling.immediate_fraction` — **1.0에 가깝지 않으면 prefetch가 벽** |
| `wave_barriers_avoided` | 같은 곳 |
| tenancy 0 게이트 | `slot_duplicates`, `slot_stale_tenants`, `slot_double_releases`, `stale_tenant_refusals`, `epoch_regressions`, `live_tenancies_at_close` |

### 4.4 exactness 게이트 — rolling vs wave, ≥8 케이스

두 런의 `--raso`를 서로 다른 workdir에 쓰고, `[RASBERY][EVALUATOR][CASE]` 줄의
`digest`를 덱별로 맞춘다. 최소 8개 덱에 대해:

```
python - <<'PY'
import json, re, sys, pathlib
CASE = re.compile(r"\[RASBERY\]\[EVALUATOR\]\[CASE\]\s*(\{.*\})")
def digests(root):
    out = {}
    for log in pathlib.Path(root).rglob("*.evaluator.log"):
        for m in CASE.finditer(log.read_text(encoding="utf-8", errors="replace")):
            c = json.loads(m.group(1))
            if c.get("isolation_check"):
                continue
            out[pathlib.Path(c["deck"]).name] = c["digest"]
    return out
a, b = digests(sys.argv[1]), digests(sys.argv[2])
common = sorted(set(a) & set(b))
bad = [d for d in common if a[d] != b[d]]
print("compared", len(common), "decks; mismatches:", bad)
raise SystemExit(1 if bad or len(common) < 8 else 0)
PY
```

`E:/.../wp18/wave_m8 E:/.../wp18/roll_m8`로 실행. **mismatch 0이 아니면 그 런은 무효**이고
throughput 숫자는 폐기한다. `--result light` 런은 digest가 light-result digest이므로
그대로 비교 가능하고, `--result full` 런은 h5diff -c 0을 추가로 돌린다.

### 4.5 feature-off 동일성 확인 (한 번)

같은 8잡 manifest를 `--claim auto`로 두 번 돌리되 한 번은 자식 환경에
`RASBERY_EVALUATOR_ROLLING`을 넣지 않고, 한 번은 `RASBERY_EVALUATOR_ROLLING=0`으로 넣는다.
두 stdout에서 타임스탬프성 필드(`wall_s`, `*_ms`, `rss_*`, `cases_per_hour`)를 지운 뒤
diff가 비어야 한다.

---

## 5. 계약 테스트

`tools/test_evaluator_rolling_contract.py` — 8부, 23개 네거티브 컨트롤,
그리고 **실제 프로토콜 왕복 1회**(스레드로 레인을 흉내 내는 stand-in evaluator에
20잡을 흘려보내고, 디스패처가 `width+prefetch`를 초과해서도 미달해서도 안 된다는 것을
stand-in이 측정한 high-water로 확인한다).

같이 돌린 것: `test_enum_alias_contract`, `test_dependent_template_contract`,
`test_evaluator_contract`, `test_evaluator_mem_receipt_contract`,
`test_harness_env_parity`, `test_xfer_ledger_contract` — 전부 PASS.
`test_evaluator_mem_receipt_contract`의 "reportMemory는 wave당 정확히 1회" 검사는
"**세대를 닫는 함수마다 정확히 1회, 다른 곳에서는 0회**"로 바뀌었다(닫는 함수가
`runWave`와 `rollingBarrier` 둘이 되었으므로). 단순 카운트는 원래부터 대리 지표였고,
새 규칙은 `runOneCase`에서 샘플링하는 것 같은 실제 결함을 여전히 잡는다.

---

## 6. 미해결 / N1

1. **컴파일 미검증.** 이 트리에는 컴파일러가 없다. 새 코드는 헤더 하나
   (`src/EvaluatorServer.h`)와 `src/BatchRefill.h`에만 있고 문법·중괄호 균형은
   확인했지만, 238에서 첫 빌드가 필요하다.
2. **OpenMP 팀의 생성 스레드가 다르다.** wave 모드의 팀은 main 스레드가 만들고,
   rolling 팀은 헬퍼 스레드가 만든다. 둘 다 같은 taskset cpu-list 안이지만
   `OMP_PROC_BIND` 배치가 미세하게 다를 수 있다. 세션마다 팀이 새로 생기므로
   레인의 `thread_local` 캐시도 세션 경계에서 초기화된다 — 이는 wave 모드보다
   **덜** 이월하는 방향이므로 digest에 영향을 줄 수 없지만, 첫 케이스의 워밍업
   비용은 세션당 한 번 다시 든다.
3. **`wave_barriers_avoided`의 정의**는 "이미 돌고 있는 세션에 합류한 요청 줄의 수"다.
   wave 모드가 실제로 냈을 배리어 수(청크 수 − 1)와 같은 양이 아니다. 이름이 세는 것과
   재는 것을 문서에 적어 두는 것으로 갈음했다.
4. **`--claim rolling`의 fleet tuner 연동 미확인.** 튜너(`--tune-*`)는 wave 경로로
   후보를 잰다. rolling arm의 K는 wave arm의 K와 다를 수 있으므로, 튜닝은 wave로 하고
   그 K로 rolling을 재는 현재 조합은 rolling에 불리한 쪽으로 편향될 수 있다.
5. **prefetch 기본값 2 / 큐 용량 레인당 4는 측정되지 않았다.** 238에서
   `immediate_fraction`을 보고 조정할 것.
