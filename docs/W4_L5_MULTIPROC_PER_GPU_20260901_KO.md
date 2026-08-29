# L5 — GPU당 다중 프로세스 (`--procs-per-gpu`), C++를 건드리지 않는 폭 레버 (2026-09-01)

브랜치 `codex/exact-throughput-campaign`, 기준 `617bac7`
대상: GA evaluator 계획 §5.5 / §6.2 Task 7
변경 파일: `tools/run_multi_gpu_batch.py`, `tools/run_single_gpu_batch.py`,
`tools/test_multi_gpu_dispatch.py`, `tools/test_harness_env_parity.py`,
`tools/test_fleet_tuner.py`(WP4 튜너 계약),
`test/reference/batch_reference_env_238.json` — **`src/`는 한 줄도 건드리지 않는다.**
갱신 2026-09-01: §4.8에 238 실측 행렬 8 arm, §4.9에 WP4 튜너(`--procs-per-gpu auto`).
§2.5의 VRAM 모델은 §4.8-5가 반증했고 두 항으로 고쳤다.
로컬 실측: WSL, GTX 1080 Ti(sm_61) 1장, nvcc 12.6, 24 코어, `~/bfdecks` d0..d3 + e0..e3
238 인용치: nsys osrt(`pthread_cond_wait` 72.6 %, `pthread_mutex_lock` 17.6 %),
`mean_width` **14.5/64 = 22.7 %**, SM 62 %, GPU 메모리 처리량 7 %,
M64 원시 생산 라인 = **582 c/h (12.3 GB)** — 이것이 대조군이다(예전 인용치 524는 다른
빌드의 값이다). dispatcher 대조군이 이 값을 ±5 %로 재현하지 못하면 배수는 무효다(§4.7).

---

## 0. 결론 먼저

**레버의 정체.** 238의 M64 배치는 **디바이스가 아니라 호스트에 묶여 있다.** SM 62 %,
GPU 메모리 7 %인데 osrt의 90 %가 뮤텍스와 조건변수다. 그리고 64슬롯 아레나의 랑데부가
평균 14.5명밖에 모으지 못한다 — **커널 grid는 선언 폭에 비례하고 유용한 일은 달성 폭에
비례하므로**, 이 배치는 선언 폭 64를 지불하고 22.7 %를 쓴다.

아레나·랑데부·호스트 뮤텍스 집합이 전부 **프로세스 수명 싱글턴**이라는 사실이
그대로 레버가 된다. 같은 GPU에 K개의 프로세스를 붙이면 K개의 독립 아레나, K개의 독립
랑데부, K개의 독립 뮤텍스 집합이 생긴다. **선언 폭 총합은 그대로 두고 랑데부만 좁힌다.**
C++는 자기가 혼자라고 믿으므로 아무것도 바뀌지 않는다.

**이 문서가 주는 것.**

1. `tools/run_multi_gpu_batch.py --procs-per-gpu K` — 호스트 분할, CPU 핀, 스레드 예산,
   공유 큐, 집계 수신증, 메모리 가드, MPS 수명주기 (§2)
2. 로컬 게이트 결과 — **K=2가 K=1과 출력 36/36 `h5diff` 동일**, dup/stale 0 (§3)
3. 238 실행 명령 행렬 — 64잡 × {1×M64, 2×M32, 4×M16, 8×M8} × {MPS 무/유} (§4)

**게이트는 B0다**(§5.5): 출력이 단일 프로세스와 전 데이터셋 Δ=0이어야 한다. 아레나가
프로세스마다 독립이므로 이것은 **구조적으로 만족되어야 하며**, 만족되지 않으면 그 자체가
결함이다. 로컬에서 만족했다(§3).

**킬 기준**(계획 §6.2 Task 7): `--procs-per-gpu 2`가 **1.05× 미만**이면 이 트랙을 닫고
W4의 slot compaction만 간다.

### 0.1 판정 지표는 `mean_width`가 아니라 `width_fill`이다

**정직하게 읽는 법 하나만 먼저.** 4슬롯 아레나는 64슬롯 아레나의 `mean_width`에
도달할 수 없다. **성공한 L5 arm은 control보다 `mean_width`가 작게 나온다.** 로컬에서
실제로 그랬다(2.44 → 1.69). 비교할 수 있는 것은 **선언 폭 대비 채움률**이다:

```text
width_fill = mean_width / slots        (수신증 [MULTI_GPU][PROC].width_fill)

238 control  14.5 / 64 = 0.227   ← 이 숫자를 이겨야 한다
로컬 K=1 W8   2.439 / 8 = 0.305
로컬 K=2 W4   1.692 / 4 = 0.423   (+39 %)
```

`mean_width`만 보고 arm을 버리는 것이 이 캠페인에서 가장 하기 쉬운 실수다.

---

## 1. 왜 dispatcher 한 곳인가

`tools/run_multi_gpu_batch.py`는 원래 GPU당 프로세스 하나를 띄우고 `--gpus`가 서로
다를 것을 요구했다. 아레나가 프로세스 수명 싱글턴이고 현재 디바이스에 대해 폭 M으로
한 번 사이징되므로, **같은 GPU에 프로세스를 여러 개 붙이면 "선언 폭이 작은 배치
여러 개"가 된다**. §3.1(b)의 곱 구조(폭 비용 × 반복 수)에서 폭 비용만 낮추는 형태다.

필요한 변경은 전부 dispatcher 안에 있다. **`src/`에는 아무 근거도 없다** — 프로세스들은
서로를 보지 못하고, `CUDA_VISIBLE_DEVICES`는 각자에게 device 0을 보여 준다.

---

## 2. dispatcher 변경

### 2.1 `--procs-per-gpu K`

```text
--procs-per-gpu K       물리 GPU 하나에 K개의 프로세스열을 붙인다 (기본 1)
--batch-width W         프로세스 **하나당** 아레나 폭
                        → GPU당 선언 폭 = K × W
```

`--procs-per-gpu 4 --batch-width 16`은 `--batch-width 64` 단독과 **같은 64슬롯**을
디바이스에 선언한다. 이것이 arm 설계의 전제다: VRAM도, 선언 폭도 고정하고 **랑데부 폭만**
바꾼다.

| 값 | 소스 |
|---|---|
| 인자 정의 | `tools/run_multi_gpu_batch.py:903` (`--procs-per-gpu`) |
| 워커 격자 (GPU-major 전역 인덱스) | `tools/run_multi_gpu_batch.py:1179` |
| 프로세스열 본체 | `tools/run_multi_gpu_batch.py:730` (`run_worker`) |

### 2.2 호스트 분할 — 분모는 GPU가 아니라 **프로세스**

`plan_host_budget`(`tools/run_multi_gpu_batch.py:276`)의 분모가
`len(gpus) * procs_per_gpu`로 바뀌었다.

```text
processes      = G × K
cpus_per_proc  = visible_cpus // processes
cpu_sets       = [0..c), [c..2c), ...   (프로세스마다 하나, 겹치지 않음)
driver_workers = W                              → RASBERY_BATCH_HOST_THREADS
solver_threads = W                              → RASBERY_OMP_THREADS
                                                  (= OMP_NUM_THREADS = OMP_THREAD_LIMIT)
```

**나뉘는 것은 코어이고, 레인은 나뉘지 않는다.** 두 자원은 성격이 다르다.

- **코어**는 쪼갠다. K개의 프로세스가 각자 호스트 전체를 가졌다고 믿으면 CPU 바운드
  구간에서 정확히 K배 오버섭스크립션이 난다. `--pin taskset`이 겹치지 않는 범위를
  준다. 단위 테스트가 집합 disjoint와 `cpus_per_proc` 축소를 고정한다
  (`test_multi_gpu_dispatch.py`).
- **레인은 쪼개지 않는다.** `RASBERY_BATCH_HOST_THREADS`는 Driver 리필 레인 수이고,
  **레인은 CPU 워커가 아니다** — 수명의 대부분을 GPU 랑데부에서 블록된 채 보내고,
  아레나는 *그 안에 들어와 있는* 레인만 모을 수 있다. 그래서 기본값은 **바이너리 자신의
  기본값**인 `min(batch_width, jobs)`(`src/main.cpp:698`), 즉 **슬롯당 레인 하나**이며,
  프로세스가 받은 코어 수와 무관하다. 238의 원시 생산 라인이 정확히 이렇게 돈다
  (24코어에 64레인, 582 c/h).

> **레인을 코어 수로 캡했을 때의 실측 비용: 582 → 115.6 c/h (5.0배), `width_fill`
> 0.03.** 24개 레인은 64슬롯을 채울 수 없고, 이 실패는 오류도 FAIL 줄도 남기지 않는다.
> 처리량 숫자 하나로만 나타난다. §4.7을 볼 것.

64레인이 24코어 위에서 터지지 않는 이유는 `OMP_MAX_ACTIVE_LEVELS=1`(DEFAULT_ENV)이다.
레인은 중첩 솔버 팀을 만들지 않으므로 스레드 수는 레인 수이고, 그 레인들은 대부분
블록되어 있다(`src/main.cpp:764-769`).

`--no-oversubscribe`는 예전 정책(`min(W, cpus_per_proc)`)을 **의도적인 arm으로** 되살린다.
`--driver-workers N`은 숫자를 직접 지정한다. `RASBERY_OMP_THREADS`/`OMP_NUM_THREADS`/
`OMP_THREAD_LIMIT`는 **프로세스당 폭 W**가 기본값이다 — 기준선의 64는 코어 수가 아니라
`--batch-mode 64`의 폭이다.

### 2.2.1 자식 환경은 한 곳에서 만들고, 실행 전에 인쇄한다

`resolve_profile_env()`(`tools/run_single_gpu_batch.py`)가 두 하네스의 유일한 환경
조립 지점이다. 단일 GPU 프로파일러와 dispatcher가 서로 다른 기본 환경을 갖는 것은
**어느 수신증에도 나타나지 않는다.** 그래서:

- `DEFAULT_ENV`는 238 기준선의 환경을 **키 단위로 그대로** 담는다
  (`test/reference/batch_reference_env_238.json`).
- `[RASBERY][MULTI_GPU][ENV]`가 프로세스마다 **실행 전에** 해석된 환경을 인쇄하고,
  `[MULTI_GPU][PROC]`의 `env` 필드가 **실제로 실행된** 환경을 기록한다.
- `--print-env`는 해석만 하고 종료한다(큐를 claim하지 않는다). 7분을 쓰기 전에 볼 것.
- `tools/test_harness_env_parity.py`가 이 동등성을 계약으로 고정한다(음성 대조군 포함).

### 2.3 공유 큐 — claim 정책의 분모도 프로세스다

- `--claim auto`에서 "혼자면 큐 전체를 claim한다"의 조건이 `budget.gpus == 1`에서
  **`budget.processes == 1`**로 바뀌었다(`:781`). GPU 1장이라도 K개면 서로 훔칠 상대가
  있다.
- `--claim all`의 정적 분할도 `budget.processes`로 나눈다(`:776`).
- **큐에 in-process 뮤텍스가 추가되었다**(`:201`). flock은 *다른 프로세스*를 막는다.
  `--procs-per-gpu`에서 한 디바이스의 K개 워커는 **이 dispatcher의 스레드**이고,
  Windows에는 flock이 아예 없다. 실제로 K=4 계약 테스트에서 read-modify-write가
  끼어들어 **queue.json이 잘린 채로 읽히는** 것을 잡았다. 회귀는
  `test_multi_gpu_dispatch.py:338`(8스레드 × 400잡 tiling)이 고정한다.

### 2.4 수신증

프로세스 단위 줄이 새로 생겼다. `mean_width`는 **아레나 하나의 성질**이므로 아레나가
있는 곳에서 보고해야 한다.

```text
[RASBERY][MULTI_GPU][PLAN]  procs_per_gpu, processes, declared_width_per_gpu,
                            cpus_per_proc, cpus_per_gpu, driver_workers,
                            driver_worker_policy, solver_threads, pin, pin_omp
[RASBERY][MULTI_GPU][ENV]   프로세스마다 하나. 실행 전에 해석된 자식 환경 전체와
                            taskset 접두사. `--print-env`면 여기서 멈춘다
[RASBERY][MULTI_GPU][VRAM]  per_slot_gb, per_process_gb, per_device_gb, aggregate_gb,
                            devices[].{total_gb, budget_gb, demand_gb, verdict}
[RASBERY][MULTI_GPU][MPS]   requested, active, control, thread_percent, pipe_dir, reason
[RASBERY][MULTI_GPU][PROC]  gpu, proc, cpus, jobs, wall_s, cases_per_hour,
                            mean_width, width_fill, refills, tail_idle_s, rc,
                            env   ← 프로세스마다. env는 **실제로 실행된** 환경이다
[RASBERY][MULTI_GPU][GPU]   디바이스 집계. wall_s는 **가장 느린 워커의 것**(합이 아니다)
[RASBERY][MULTI_GPU][TOTAL] cases_per_hour, mean_width_per_proc[], width_fill_per_proc[],
                            tail_idle_max_s, mps, duplicates, stale_tenants, rc
```

**`[GPU].wall_s`가 합이 아니라 최대인 이유**: K개 워커는 동시에 돈다. 합으로 나누면
디바이스 처리량이 정확히 K로 나뉜다. `test_multi_gpu_dispatch.py:229`가 고정한다.

### 2.5 메모리 가드 (`[MULTI_GPU][VRAM]`)

> **§4.8-5가 이 절의 모델을 반증했다.** 슬롯 항은 예고대로 K에 대해 평평했지만,
> **프로세스마다 2.56 GB의 고정비**(CUDA 컨텍스트·모듈·라이브러리 핸들·할당자 풀)가
> 있고 아무도 청구하지 않고 있었다. 가드는 이제 두 항이다:
> `K×W×per_slot + (K−1)×extra_process_gb`. 아래 문단은 첫 항의 설명으로 읽을 것.

아레나는 프로세스 시작 시 **선언 폭으로 한 번** 사이징되고 런 내내 잡고 있다. 따라서
K개 프로세스 × 폭 W는 한 디바이스에서 K×W 슬롯이다.

```text
per_slot_gb = 13.0 GB / 64 slots = 0.203 GB      ← 238 M64 전출력 peak 실측
demand(device) = K × W × per_slot_gb
budget(device) = nvidia-smi total − margin(기본 1.0 GB)
demand > budget  → 큐를 claim하기 전에 rc=2로 거절
```

- **디바이스마다** 청구한다(캠페인 총합이 아니다). GPU 두 장은 각자 메모리를 가진다.
  `aggregate_gb`는 운영자에게 보여 주기 위한 값이지 판정값이 아니다.
- `nvidia-smi`를 물을 수 없으면 `verdict:"unverified"` + stderr 경고로 **넘어간다**.
  모르는 것을 아는 척하지 않는다. `--device-memory-gb`로 명시할 수 있다.
- `--allow-vram-overcommit`으로 강행할 수 있다. 그때 실패는 **아레나 stand-up 시점**,
  즉 큐를 이미 claim한 뒤에 난다 — 그것이 이 가드가 존재하는 이유다.
- 이 레버의 정상 arm에서는 항상 통과한다(K×W = 64 고정). 가드가 잡는 것은
  `--procs-per-gpu 4 --batch-width 64`처럼 **타이핑하기 쉬운 4배 arm**이다.
  RTX PRO 6000(96 GB)에서 4×M64 = 52 GB는 통과하고 **8×M64 = 104 GB는 거절**된다.

### 2.6 MPS 수명주기

```text
--mps                    이 런을 위한 MPS control daemon을 띄운다
                         CUDA_MPS_PIPE_DIRECTORY = <workdir>/mps/pipe
                         CUDA_MPS_LOG_DIRECTORY  = <workdir>/mps/log
                         자식마다 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = 100/K
--mps-thread-percent P   위 기본값 대신 P
--mps-optional           MPS가 없으면 거절하는 대신 time-sliced로 진행한다
```

원칙 넷:

1. **workdir 안에서 띄운다.** 기본 `/tmp/nvidia-mps`를 쓰면 두 캠페인이 서로의 서버에
   붙고, `stop()`이 남의 daemon을 내린다.
2. **`-d` 반환은 기동이 아니다.** `get_server_list`로 ping해서 응답을 확인한 뒤에만
   `active:true`다. 확인하지 않으면 전부 plain context로 돌면서 "MPS arm"으로 보고된다.
3. **`finally`에서 반드시 내린다**(`:1141`). 남은 daemon은 디바이스의 compute mode를
   쥐고 있고, **다음 arm의 "MPS 없음" control이 조용히 MPS 런이 된다.**
4. **없으면 거절한다.** `--mps`인데 daemon이 없으면 `[MPS]` 수신증에 `active:false`와
   `reason`을 찍고 **rc=2로 종료한다** — GPU 시간을 쓰기 전에. time-sliced 런을 MPS
   라벨로 보고하면 이 arm이 존재하는 이유인 비교 자체가 무너진다. 일부러 time-sliced를
   재고 싶으면 `--mps-optional`을 명시한다.

로컬(WSL)에는 `nvidia-cuda-mps-control`이 없다. 실측 거절:

```text
[RASBERY][MULTI_GPU][FAIL] --mps was requested and MPS is not available:
  nvidia-cuda-mps-control is not on PATH: this host has no MPS control daemon
  (WSL and most container images do not ship one). Refusing rather than running
  time-sliced under an MPS label; ...
[RASBERY][MULTI_GPU][MPS] {"requested":true,"active":false,"control":null,
  "thread_percent":null,"pipe_dir":null,"log_dir":null,"reason":"..."}
rc=2
```

**따라서 MPS의 성능은 로컬에서 재지 않았다. 238 runner의 몫이다.**

### 2.7 부수 수정 — `--result light`가 모든 arm을 rc=3으로 떨어뜨리던 문제

L5와 직접 관계는 없지만 §4의 행렬이 `--result light`를 쓰므로 여기서 고쳤다.

`--result light`는 **출력 형태 스위치**이지 충실도 스위치가 아니다(궤적 digest 동일,
계획 §2.4). 그러나 `main.cpp:541`은 결과 HDF5를 쓰지 않는 잡을 **screening run으로
분류**하므로 `[PHYSICS_MODE]`가 `screening:true` / `full_hdf5:false` /
`physics_mode:ga_screen_feedback_limited`로 나온다. 두 하네스의 감사는 이것을
`EXACT_PHYSICS_MODE`와 대조하고 있었다 — **light arm은 전부 rc=3으로 떨어진다.**
물리와 무관한 필드에서.

- `run_single_gpu_batch.py:54`에 `SCREENING_PHYSICS_MODE`와
  `expected_physics_mode(result_mode)`를 추가했다. **`feedback_pass_limit=0`은 완화하지
  않는다** — 그것이 답을 바꾸는 유일한 필드(GA feedback 근사)다.
- `LaunchPlan.result_mode`가 감사에 기대치를 고른다.
- dispatcher는 청크마다 `chunk_result_mode()`(`:625`)로 유효 모드를 계산한다.
  `main.cpp`의 `any_of(light)`와 같다 — 청크 하나에 light 잡이 하나라도 있으면
  그 **프로세스**가 screening run이고, 수신증은 프로세스당 한 번 찍힌다.
- `--result light`인데 `RASBERY_ALLOW_SCREENING`이 없으면 **큐를 claim하기 전에**
  rc=2로 거절한다(`main.cpp:545`의 falsey 철자까지 그대로 흉내낸다).

---

## 3. 로컬 게이트 (1080 Ti, `~/bfdecks`)

`~/l5_src` = `617bac7` `git archive` 그대로, `~/l5_b`로 빌드
(sm_61, Release, `RASBERY_ENABLE_CUDA=ON`). **작업 중인 `src/`가 섞이지 않도록
tip 커밋을 아카이브해서 빌드했다.**

### 3.1 기준 arm을 고르다가 발견한 함정 (**runner는 반드시 읽을 것**)

첫 시도에서 단일 모드 기준과 **모든** 배치 arm이 `h5diff` 불일치였다 — K=1 포함.
원인은 L5가 아니었다:

```text
단일 모드 : [PHYSICS_MODE] ... "exec_mode":"single","xe_anderson":true, "xe_anderson_source":"default"
배치 모드 : [PHYSICS_MODE] ... "exec_mode":"batch" ,"xe_anderson":false,"xe_anderson_source":"default"
```

**`RASBERY_XE_ANDERSON`의 기본값이 단일 모드에서 ON, 배치 모드에서 OFF다**
(`V3_FREEZE` §2가 단일 기본값을 명시적으로 지목한다). 명시하지 않은 단일 모드 기준은
**기준이 아니라 다른 arm**이다. `RASBERY_XE_ANDERSON=0`으로 고정하니 전부 일치했다.

> 238에서는 반대 방향으로 같은 함정이 있었다: 생산 arm은 `RASBERY_XE_ANDERSON=1`인데
> 예전 `DEFAULT_ENV`는 이 키를 설정하지 않아 환경 상속에 의존했다. **지금은 아니다** —
> `DEFAULT_ENV`가 238 기준선 환경을 키 단위로 담으므로 `RASBERY_XE_ANDERSON=1`은
> 하네스가 직접 준다(§2.2.1). 로컬에서 `=0` arm을 재려면 `--set RASBERY_XE_ANDERSON=0`
> 으로 **명시적으로** 뒤집어야 한다. 무엇이 실제로 갔는지는 `[MULTI_GPU][ENV]`가 말한다.

### 3.2 게이트 표

기준선 `single0`: 단일 모드 4덱(`RASBERY_XE_ANDERSON=0`), 각 덱 1프로세스.

| # | arm | 잡 | rc | wall | c/h | `mean_width` (per proc) | `width_fill` | refills | tail_idle | dup / stale |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| **A** | `K=1 W=8` `claim auto` | 8 | **0** | 48.24 s | 597.0 | 2.439 | **0.305** | 0 | 29.77 s | 0 / 0 |
| **B** | `K=2 W=4` `claim auto` | 8 | **0** | 38.55 s | **747.0** | 1.692 / 1.692 | **0.423** | 0 | 12.78 s (max) | 0 / 0 |
| **C** | `K=2 W=4` `claim auto` | 12 | **0** | 64.51 s | 669.7 | 1.707 / 1.621 | 0.427 / 0.405 | 0 | 22.05 s (max) | 0 / 0 |
| **D** | `K=2 W=4` `claim 6` | 12 | **0** | 60.11 s | 718.7 | 1.586 / 1.520 | 0.396 / 0.380 | **2 + 2** | 39.43 s (max) | 0 / 0 |
| **A′** | A 재실행 (최종 코드, `width_fill` 수신증 포함) | 8 | **0** | 76.15 s | 378.2 | 2.410 | **0.3013** | 0 | 42.53 s | 0 / 0 |
| **B′** | B 재실행 (최종 코드) | 8 | **0** | 45.09 s | **638.7** | 1.626 / 1.635 | **0.4075** | 0 | 12.84 s (max) | 0 / 0 |

A′/B′는 §2.4의 `width_fill` 필드를 넣은 뒤 **커밋되는 코드 그대로** 다시 잰 것이다.
A·B와 A′·B′ 사이의 절대 wall 차이는 전부 호스트 경합이다(§3.3의 3번).

| # | 게이트 | 결과 |
|---|---|---|
| **G1** | A·B·C·D 출력 **36개** 전부 `h5diff -c` vs `single0` | **PASS** (0 mismatch) |
| **G2** | B 출력 8개 `h5diff -c` vs **A 출력**(K=1 W=8) | **PASS** (0 mismatch) — §5.5의 B0. B′ vs A′도 **PASS** |
| **G3** | 모든 프로세스 `duplicates=0`, `stale_tenants=0`, `double_releases=0` | **PASS** |
| **G4** | 모든 arm rc=0, `fail_lines=0`, physics-mode/graph-fallback 감사 무결 | **PASS** |
| **G5** | D에서 프로세스 **안**의 리필이 실제로 돈다 (`refills=2`, `admissions=12`) | **PASS** |
| **G6** | `--mps` on WSL → rc=2, `[MPS].active=false` + `reason`, daemon 잔존 없음 | **PASS** |
| **G7** | `tools/test_multi_gpu_dispatch.py` (K>1 분할·MPS 수명주기·VRAM 가드·큐 동시성, 음성 대조군 포함) | **PASS** (Windows·WSL 양쪽) |
| **G8** | `test_single_gpu_batch_profile.py`, `test_result_mode_contract.py`, `test_batch_refill_contract.py` | **PASS** (회귀 없음) |

**배수와 채움률.**

| 쌍 | c/h 배수 | `width_fill` 배수 |
|---|---:|---:|
| B / A | 747.0 / 597.0 = **1.251×** | 0.423 / 0.305 = **1.387×** |
| B′ / A′ | 638.7 / 378.2 = **1.689×** | 0.4075 / 0.3013 = **1.352×** |

**c/h 배수는 두 번의 측정에서 1.25×와 1.69×로 갈렸고, `width_fill` 배수는 1.39×와
1.35×로 붙었다.** 호스트가 경합 중이었으므로 이것이 예상되는 모습이다 — 그리고 이것이
§0.1이 `width_fill`을 판정 지표로 쓰라고 한 이유다. 어느 쌍으로 읽어도
**킬 기준 1.05×는 넘는다.**

### 3.3 로컬 배수를 정직하게 읽는 법

**이 배수를 238에 옮겨 적으면 안 된다.**

1. 로컬 덱은 짧고(케이스 ~5 s) `width_fill`이 이미 0.305다. 238의 M64는 0.227에서
   출발한다 — **개선 여지가 더 크다.**
2. 로컬 폭은 8/4이고 238은 64/32/16/8이다. 뮤텍스 경합은 폭에 대해 선형이 아니다.
3. 측정 중 같은 호스트에서 다른 CUDA 빌드/실행이 돌고 있었다. **wall 산포가 평소보다
   크다** — 같은 A arm이 48.24 s와 76.15 s로 나왔다. 그래서 c/h 배수는 1.25×–1.69×로
   갈렸고 `width_fill` 배수는 1.35×–1.39×로 붙었다.
4. 1080 Ti는 MPS 없이 time-sliced다. 238에서 MPS를 켜면 **이 배수의 상한이 달라진다.**
   → **§4.8이 그것을 실측했다: 1.92배 달라진다**(8×M8이 457 → 878 c/h). 그리고 방향도
   달라진다 — MPS 없는 K=8은 이득이 아니라 **손실**이다.

**보고해야 할 것은 배수가 아니라 `width_fill`과 `cases_per_hour`의 쌍이다.**

---

## 4. 238 실행 명령 행렬 (sonnet runner)

**GPU0 전용.** dispatcher가 자식에게 `CUDA_VISIBLE_DEVICES=0`을 강제하므로 GPU1은
자식 프로세스에서 보이지 않는다.

### 4.1 환경 (한 번)

```bash
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$($HOME/opt/bin/micromamba shell hook -s bash)"; micromamba activate gpu

# --- v3 생산 arm(arm X + chunked + GPU_XE + Anderson + A2 CAND)은 이제
#     DEFAULT_ENV가 준다. 여기서 export할 필요가 없다. ---
unset  RASBERY_GPU_OUTER RASBERY_GPU_OUTER_SEGMENT_MAX RASBERY_GPU_OUTER_GRAPH   # segment OFF
# RASBERY_ALLOW_SCREENING 은 더 이상 필요 없다 (WP1): --result light 는 출력 모양이지
# screening 이 아니다. export 해도 무해하지만, 그것으로 고쳐지는 문제는 하나도 없다.
unset  RASBERY_STATEPOINT_TELEMETRY     # 타이밍과 텔레메트리를 섞지 말 것
export B=$HOME/<BUILD>                  # RASBERY 바이너리 디렉터리
export O=$HOME/l5out; mkdir -p $O
```

**환경은 이제 하네스가 준다.** `DEFAULT_ENV`(`tools/run_single_gpu_batch.py`)가 238
원시 생산 라인의 환경을 키 단위로 그대로 담는다 —
`RASBERY_GPU_XE=1`, `RASBERY_XE_ANDERSON=1`, `RASBERY_GPU_WIEL_FOLD=chunked`,
`RASBERY_STAGED_*`, `RASBERY_BATCH_WAIT_US=auto` + `_MAX_US=2000`,
`OMP_DYNAMIC/NESTED/MAX_ACTIVE_LEVELS/STACKSIZE/WAIT_POLICY`, `GOMP_SPINCOUNT=0`,
`MKL_NUM_THREADS=1`, `KMP_BLOCKTIME=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100`.
기준 사본은 `test/reference/batch_reference_env_238.json`이고
`tools/test_harness_env_parity.py`가 동등성을 고정한다.

**하네스가 주지 않는 것 셋.** `RASBERY_ALLOW_SCREENING`(권한 — 운영자가 export),
`RASBERY_BATCH_LIGHT_RESULT`(= `--result light`),
`RASBERY_BATCH_RECEIPT_JSONL`(경로 — K개 프로세스가 한 파일에 append하면 섞인다).

> **이 환경은 strict가 아니라 A2다 — 그리고 그것이 기본 arm이다.**
> `DEFAULT_ENV`의 `RASBERY_STAGED_FLUX_TOL=50` / `_XE_TOL=1000`을
> `src/RunContract.h`가 **A2 staged tolerance** 정책으로 읽는다. 따라서 모든 자식은
> `[PHYSICS_MODE].policy="A2"`를 인쇄하고, **§4.8 표와 원시 582 c/h를 포함해 이
> 문서의 모든 처리량 숫자는 A2 숫자다**. 계획 §6.2가 strict와 A2를 한 표에 섞는 것을
> 금지하므로, 인용할 때마다 라벨을 붙일 것. 하네스는 이제 실행마다
> `declared_fidelity`를 인쇄하고(`[PLAN]`, `[MULTI_GPU][ENV]`, `[TOTAL]`), 자식이
> 다른 policy를 인쇄하면 **양방향 모두 실패**한다(§4.10).

> **`RASBERY_PPR_MODE=master`는 더 이상 기본값이 아니다.** 기준선이 이 키를 설정하지
> 않는데 예전 `DEFAULT_ENV`는 강제하고 있었다 — 선언되지 않은 편차이고, 공짜도 아니다:
> PPR은 light 여부와 무관하게 **모든 statepoint에서 돈다**(`src/Driver.h:4166` 이하).
> MASTER 대조 캠페인은 `--set RASBERY_PPR_MODE=master`를 **명시**할 것.

**실행 전에 한 번 확인한다** (7분을 쓰기 전에):

```bash
python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 1 --batch-width 64 \
    --jobs $O/m64.txt --workdir /tmp/envcheck --print-env -- $B/RASBERY \
  | grep "MULTI_GPU\]\[ENV\]"
python3 tools/test_harness_env_parity.py     # "harness env parity: PASS"
```

### 4.2 매니페스트 (64잡, 한 번)

```bash
cd ~/t18decks/kngr
mkdir -p $O/out
: > $O/m64.txt
for i in $(seq 0 63); do
  printf '%s %s\n' "kngr_$i.json" "$O/out/case$i.h5" >> $O/m64.txt
done
wc -l $O/m64.txt      # 64
```

> **arm마다 출력 경로를 바꿀 필요는 없다.** `--result light`는 결과 HDF5를 쓰지 않으므로
> 세 번째 필드/`--result`가 곧 출력 형태다. 다만 arm 간 **로그를 섞지 않도록**
> `--workdir`는 반드시 arm마다 다르게 준다.

### 4.3 arm 행렬 — 64잡, 선언 폭 64 고정

| arm | `--procs-per-gpu` | `--batch-width` | GPU당 선언 폭 | `cpus_per_proc` (taskset) | `RASBERY_BATCH_HOST_THREADS` | `RASBERY_OMP_THREADS` | `--mps-thread-percent` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **control** | 1 | 64 | 64 | 24 (`0-23`) | **64** | **64** | — |
| **2×M32** | 2 | 32 | 64 | 12 (`0-11`,`12-23`) | **32** | **32** | 50 |
| **4×M16** | 4 | 16 | 64 | 6 | **16** | **16** | 25 |
| **8×M8** | 8 | 8 | 64 | 3 | **8** | **8** | 12 |

**두 스레드 값은 모두 "프로세스당 폭 W"이고, `--set`으로 줄 필요가 없다** — dispatcher가
계산해서 넣는다. 코어만 K등분되고(`taskset`), **레인은 등분되지 않는다**(§2.2). control의
`64 / 24코어`는 오버섭스크립션이 맞고, **그것이 기준선이 도는 방식이다.**
`--mps-thread-percent`는 생략하면 `100 // K`가 자동으로 쓰인다(표의 값과 같다;
`K=8`은 `100//8 = 12`).

각 arm은 **MPS 없음 / MPS 있음** 두 번 돈다. 총 **8회**.

```bash
# ---------- 공통 러너 ----------
run_arm () {                       # $1=이름  $2=K  $3=W  $4=extra
  W=$O/work/$1; mkdir -p $W
  nvidia-smi dmon -i 0 -s pucm -d 1 -o T > $O/dmon.$1.csv 2>&1 &
  DMON=$!
  /usr/bin/time -f "%e" -o $O/$1.wall \
  python3 tools/run_multi_gpu_batch.py \
      --gpus 0 --procs-per-gpu $2 --batch-width $3 \
      --claim auto --result light \
      --jobs $O/m64.txt --cwd ~/t18decks/kngr --workdir $W \
      --pin taskset $4 \
      -- $B/RASBERY > $O/$1.out 2>&1
  echo "$1 rc=$?"
  kill $DMON 2>/dev/null
  grep -h "MULTI_GPU\]\[" $O/$1.out
}

cd <REPO>

# ---------- MPS 없음 ----------
run_arm m64_nomps  1 64 ""
run_arm m32_nomps  2 32 ""
run_arm m16_nomps  4 16 ""
run_arm m8_nomps   8  8 ""

# ---------- MPS 있음 ----------
run_arm m64_mps    1 64 "--mps"
run_arm m32_mps    2 32 "--mps"
run_arm m16_mps    4 16 "--mps"
run_arm m8_mps     8  8 "--mps"
```

**control이 먼저다.** `m64_nomps`가 원시 생산 라인의 **582 c/h를 5 % 이내로**
재현하지 못하면 나머지 7개 arm은 돌리지 말 것 — §4.7이 그 이유다.

```bash
# 원시 기준선 (비교 대상). 하네스 없이, 같은 덱, 같은 폭.
taskset --cpu-list 0-23 $B/RASBERY --rasi <64덱> --raso <64출력> --batch-mode 64
```

- **`--claim auto`는 이 행렬에서 각 워커에게 정확히 폭 하나씩 준다**
  (`min(max(W, ceil(64/(2·G·K))), 공정몫)` = `W`, K = 1/2/4/8에서). 리필도 없고
  스틸도 없다 — **폭 비교만 남는 깨끗한 arm**이다. 리필까지 함께 재려면 매니페스트를
  128잡으로 늘린다(§4.6).
  공정몫 = `floor(잡수/(G·K))` + 나머지 라운드로빈이며, **상한**으로만 쓴다.
  이것이 없을 때 12×M6/64잡에서 12번째 워커가 0잡을 받았다(함정 20): 폭 바닥 6이
  공정몫 5.33보다 컸다. 128잡 리필 arm에서는 공정몫(16)이 폭(8)보다 크므로 청크는
  여전히 폭 하나이고 스틸도 그대로다.
- `--cwd`는 필수다. 덱의 단면 HDF5와 restart 네임스페이스가 덱 디렉터리 기준 상대
  경로다.
- `--pin taskset`은 K≥2에서 실제로 쪼갠다. **`--pin none`으로 재지 말 것** — K개
  프로세스가 24코어를 각자 다 가졌다고 믿는 순간 이 측정은 L5가 아니라
  오버섭스크립션을 재게 된다.
- `--mps` arm은 daemon을 workdir에 띄우고 끝나면 내린다. **arm 사이에
  `nvidia-cuda-mps-control` 프로세스가 남아 있지 않은지 확인할 것**:
  `pgrep -a nvidia-cuda-mps` 가 비어 있어야 다음 "MPS 없음" arm이 진짜 control이다.

### 4.4 `--result full` 확인 arm (B0 게이트)

행렬은 처리량용이라 `--result light`다. **B0는 별도로 한 번만 잰다** — control과
가장 공격적인 arm의 출력이 같은지:

```bash
: > $O/m64full.txt
for i in $(seq 0 63); do
  printf '%s %s\n' "kngr_$i.json" "$O/full_k1/case$i.h5" >> $O/m64full.txt
done
sed "s#full_k1#full_k8#" $O/m64full.txt > $O/m64full8.txt
mkdir -p $O/full_k1 $O/full_k8

python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 1 --batch-width 64 \
    --claim auto --result full --jobs $O/m64full.txt --cwd ~/t18decks/kngr \
    --workdir $O/work/full_k1 --pin taskset -- $B/RASBERY
python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 8 --batch-width 8 \
    --claim auto --result full --jobs $O/m64full8.txt --cwd ~/t18decks/kngr \
    --workdir $O/work/full_k8 --pin taskset -- $B/RASBERY

for i in $(seq 0 63); do
  h5diff -c $O/full_k1/case$i.h5 $O/full_k8/case$i.h5 > /dev/null \
    && echo "OK  case$i" || echo "FAIL case$i"
done | grep -c OK        # 64 여야 한다
```

**64/64가 아니면 그 자체가 결함이다** — 아레나는 프로세스마다 독립이고, 어떤 폭에서도
답은 같아야 한다. 이때는 성능 숫자를 보지 말고 이 불일치를 먼저 보고할 것.

VRAM 가드도 여기서 한 번 확인한다. 96 GB 디바이스에서:

```bash
# 통과해야 하는 것 (K×W = 64 → 13 GB)
# 거절되어야 하는 것:
python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 8 --batch-width 64 \
    --claim auto --jobs $O/m64.txt --cwd ~/t18decks/kngr --workdir /tmp/vg \
    --pin taskset --dry-run -- $B/RASBERY ; echo "rc=$? (2 여야 한다: 104 GB > 95 GB)"
```

### 4.5 읽어야 할 줄과 기대 수신증

```bash
grep -h "MULTI_GPU\]\[TOTAL\]\|MULTI_GPU\]\[PROC\]\|MULTI_GPU\]\[MPS\]" $O/*.out
grep -h "BATCH_OCCUPANCY\|REFILL\]\|HDF5\]\[LOCK\]" $O/work/*/*.log
```

**기대 수신증 — control (1×M64, MPS 없음).** 이 arm이 원시 라인의 **582 c/h를 5 % 이내로**
재현하지 않으면 나머지 arm의 배수는 의미가 없다(§4.7).

```text
[RASBERY][MULTI_GPU][PLAN]  {"gpus":["0"],"procs_per_gpu":1,"processes":1,"jobs":64,
                             "batch_width":64,"declared_width_per_gpu":64,
                             "cpus_per_proc":24,"driver_workers":64,
                             "driver_worker_policy":"binary_default_width",
                             "solver_threads":64,"pin":"taskset","pin_omp":false,...}
[RASBERY][MULTI_GPU][ENV]   {"gpu":"0","proc":0,"cpus":"0-23",
                             "pin_prefix":["taskset","-c","0-23"],
                             "env":{...RASBERY_BATCH_HOST_THREADS:"64",
                                    RASBERY_OMP_THREADS:"64",
                                    RASBERY_BATCH_WAIT_US:"auto",...}}
[RASBERY][MULTI_GPU][VRAM]  {"per_device_gb":13.0,...,"verdict":"fits"}
[RASBERY][MULTI_GPU][MPS]   {"requested":false,"active":false,...}
[RASBERY][MULTI_GPU][PROC]  {"gpu":"0","proc":0,"jobs":64,"cases_per_hour":~582,
                             "mean_width":~14.5,"width_fill":~0.227,"refills":0,...}
[RASBERY][MULTI_GPU][TOTAL] {"cases_per_hour":~582,"width_fill_mean":~0.227,
                             "duplicates":0,"stale_tenants":0,"rc":0,"fail_lines":0}
```

**`driver_workers`가 64가 아니면 그 자리에서 멈출 것.** 24라면 예전 결함이 돌아온 것이고
(§4.7), 그 arm의 숫자는 데이터가 아니다.

**기대 수신증 — 8×M8 + MPS.** `mean_width`가 **작아지는 것이 정상**이다(§0.1).

```text
[RASBERY][MULTI_GPU][MPS]   {"requested":true,"active":true,
                             "control":"/usr/bin/nvidia-cuda-mps-control",
                             "thread_percent":12,"pipe_dir":"<workdir>/mps/pipe",...}
[RASBERY][MULTI_GPU][PROC]  × 8줄. cpus 가 "0-2","3-5",..., "21-23" 로 서로 겹치지 않을 것
                             mean_width ≤ 8, width_fill 가 판정값
[RASBERY][MULTI_GPU][GPU]   {"procs":8,"jobs":64,"wall_s":<가장 느린 워커>,
                             "mean_width_per_proc":[8개], "width_fill_per_proc":[8개]}
[RASBERY][MULTI_GPU][TOTAL] {"processes":8,"mps":true,"mps_thread_percent":12,
                             "duplicates":0,"stale_tenants":0,"rc":0}
```

**수용 조건 (모든 arm 공통).**

| 조건 | 어디에 |
|---|---|
| `rc: 0`, `fail_lines: 0` | `[MULTI_GPU][TOTAL]` |
| `duplicates: 0`, `stale_tenants: 0`, `double_releases: 0` | `[TOTAL]`, `[RASBERY][REFILL]` |
| `graph_fallbacks: 0` (세 수신증 전부) | 자동 감사 — 걸리면 `[MULTI_GPU][FAIL]` |
| `[PHYSICS_MODE].feedback_pass_limit: 0` | 자동 감사 |
| MPS arm에서 `[MPS].active: true` | `[MULTI_GPU][MPS]` — false면 그 arm은 무효 |
| `cpus`가 프로세스 간 겹치지 않음 | `[MULTI_GPU][PROC].cpus` |
| arm 사이 daemon 잔존 없음 | `pgrep -a nvidia-cuda-mps` |

**판정.**

| 지표 | 어디 | 읽는 법 |
|---|---|---|
| `cases_per_hour` (control) | `[TOTAL]` | **원시 라인 582 c/h의 ±5 % 안이어야 한다.** 아니면 여기서 멈춘다 |
| `cases_per_hour` | `[TOTAL]` | control 대비 배수. **킬 기준 1.05×** (2×M32 기준) |
| `width_fill_per_proc` | `[TOTAL]` | control 0.227을 얼마나 올렸는가 — **레버가 작동하는지의 직접 증거** |
| `tail_idle_max_s` | `[TOTAL]` | 합이 아니라 최대. 워커 하나가 남아 끄는 시간 |
| `sm`, `mem` | `dmon.<arm>.csv` | control은 SM 62 %. **SM이 올라가지 않았는데 c/h가 올랐다면 호스트 쪽 이득**이고, 그것이 L5가 주장하는 이득이다 |
| `[HDF5][LOCK].wait_ms` | 자식 로그 | light arm은 쓰기 0이므로 순수 읽기 락 |

### 4.6 리필까지 함께 재고 싶을 때 (선택)

`§4.2`의 루프를 `seq 0 127`로 바꾸고(같은 64덱을 두 번, 출력 이름만 다르게) 매니페스트를
128잡으로 만든다. 폭은 아레나 크기이지 잡 수가 아니므로 `--batch-width`는 그대로 둔다.
그러면 각 워커가 자기 폭의 두 배를 claim하고 **프로세스 안에서 리필**한다
(`[PROC].refills > 0`). 폭 비교에 리필 비용이 섞이므로 **§4.3의 64잡 행렬을 먼저 끝낸
뒤에** 할 것.

### 4.7 앞선 103–116 c/h 스윕이 왜 무효였는가

**증상.** 238에서 같은 바이너리, 같은 64덱, 같은 물리 환경으로

```text
원시 생산 라인   taskset --cpu-list 0-23 RASBERY --batch-mode 64   → 582 c/h (12.3 GB)
dispatcher 대조  --gpus 0 --procs-per-gpu 1 --batch-width 64       → 115.6 c/h, width_fill 0.03
```

**5.0배.** rc는 0이었고, `[FAIL]`은 없었고, physics-mode·graph-fallback 감사도 전부
통과했다. 틀린 수신증은 하나도 없었다. **차이는 전부 하네스가 자식에게 넘긴 환경과,
하네스가 계산한 정수 하나에 있었다.**

**세 가지 원인.**

1. **`plan_host_budget()`이 레인을 코어 수로 캡했다.** `RASBERY_BATCH_HOST_THREADS =
   min(width, cpus_per_proc) = 24`. 바이너리의 기본값은 `min(width, jobs) = 64`
   (`src/main.cpp:698`)이고, 기준선은 **24코어에 64레인을 일부러 태운다.** 레인은
   GPU 랑데부에서 블록되어 있으므로 CPU 워커가 아니다 — 24레인으로는 64슬롯 랑데부가
   채워질 수 없고, `width_fill`이 0.03으로 무너졌다.
2. **`DEFAULT_ENV`가 `RASBERY_BATCH_WAIT_US="0"`을 조용히 강제했다.** 기준선은
   `auto` + `RASBERY_BATCH_WAIT_MAX_US=2000`(유계 적응 linger)로 돈다. 절대 기다리지
   않는 랑데부는 64슬롯 아레나가 서너 명만 모으는 또 하나의 방법이다.
3. **`DEFAULT_ENV`가 기준선 키 절반을 갖고 있지 않았다.** `RASBERY_GPU_XE`,
   `RASBERY_XE_ANDERSON`, `RASBERY_GPU_WIEL_FOLD=chunked`, `RASBERY_STAGED_*`,
   `OMP_*`/`MKL_*`/`KMP_*`/`CUBLAS_*` 일습. 운영자의 export에 의존했고, 운영자가
   빠뜨리면 **다른 arm이 조용히 실행된다.** 반대로 기준선이 설정하지 **않는**
   `OMP_PROC_BIND=TRUE`/`OMP_PLACES=cores`와 `RASBERY_PPR_MODE=master`는 강제하고
   있었다.

**따라서 그 스윕들은 폐기한다.** L5 arm들의 103–116 c/h는 K의 함수가 아니라 **결함 상수의
함수**였다. 하나의 arm이 아니라 **대조군 자체가 5배 느렸으므로, 그 위에서 계산한 어떤
배수도 의미가 없다** — 배수가 1.0 근처였다는 사실조차 정보가 아니다.

**같은 실패가 다시 조용히 지나가지 못하게 하는 것.**

| 장치 | 어디 |
|---|---|
| 기준 환경이 파일로 기록됨 | `test/reference/batch_reference_env_238.json` |
| 계약 테스트(음성 대조군 포함) | `tools/test_harness_env_parity.py` |
| 두 하네스가 환경 조립기를 공유 | `resolve_profile_env()` (`tools/run_single_gpu_batch.py`) |
| 실행 **전** 해석된 환경 인쇄 | `[RASBERY][MULTI_GPU][ENV]`, `--print-env` |
| 실행 **된** 환경 기록 | `[RASBERY][MULTI_GPU][PROC].env` |
| 레인 정책이 수신증에 이름으로 남음 | `[PLAN].driver_worker_policy` |

**교훈은 일반적이다.** 환경 불일치는 이 캠페인에서 **오류도, FAIL 줄도, 잘못된 수신증도
남기지 않는 유일한 결함 종류**다. 물리처럼 보이는 처리량 숫자 하나로만 나타난다. 그래서
새 arm의 첫 번째 게이트는 언제나 **"대조군이 원시 라인을 재현하는가"**여야 한다.

---

### 4.8 238 실측 행렬 — 결과 (2026-09-01), **정책 = A2**

RTX PRO 6000 96 GB, 24코어 호스트, 64잡, `DEFAULT_ENV` 생산 arm, `--result light`.
`--claim auto`이므로 워커마다 정확히 폭 하나, 리필도 스틸도 없다(§4.3).

> **표 전체의 fidelity 라벨: `policy = A2`** (staged tolerance 50/1000, §4.1).
> 원시 582 c/h 기준선도 같은 환경에서 잰 것이므로 같은 라벨이다. strict 숫자와 같은
> 표에 놓지 말 것(계획 §6.2). strict 대조 행은 §4.10에서 따로 잰다.

| arm | MPS | `cases_per_hour` | control 대비 | `width_fill` | VRAM | rc |
|---|---|---:|---:|---:|---:|---:|
| 원시 `--batch-mode 64` (하네스 없음) | — | **582** | 1.007× | — | 12.3 GB | 0 |
| **1×M64 (dispatcher control)** | 없음 | **578** | 1.000× | 0.28 | 12.3 GB | 0 |
| 2×M32 | 없음 | 648 | 1.121× | | 14.9 GB | 0 |
| 4×M16 | 없음 | 579 | 1.002× | | 20.0 GB | 0 |
| 8×M8 | 없음 | 457 | **0.791×** | | 30.2 GB | 0 |
| 2×M32 | 있음 | 810 | 1.401× | | 14.9 GB | 0 |
| 4×M16 | 있음 | 864 | 1.495× | | 20.0 GB | 0 |
| **8×M8** | **있음** | **878** | **1.519×** | **0.41** | 30.2 GB | 0 |

MPS arm은 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = 100/K`(50 / 25 / 12). 모든 arm에서
`rc=0`, `fail_lines=0`, `duplicates=0`, `stale_tenants=0`, arm 사이 daemon 잔존 없음.

**1) 게이트 8을 통과했다 — 그러므로 아래 배수는 데이터다.** dispatcher control
578 c/h는 원시 라인 582의 −0.7 %로, §4.5가 요구한 ±5 % 안이다. §4.7의 5.0배 결함은
재발하지 않았다. **이 줄이 맞지 않았다면 나머지 일곱 줄은 전부 버려야 했다.**

**2) 킬 기준 통과.** 2×M32(MPS 없음) = **1.121×** ≥ 1.05×. L5 트랙은 산다.

**3) MPS는 “가속제”가 아니라, K가 커지면 전제다.** §0은 MPS를 “correctness
메커니즘이 아니라 서로 다른 context kernel의 동시 실행을 돕는 선택 옵션”이라고 적었다.
정확성에 대해서는 옳고, **처리량에 대해서는 절반만 옳다**:

```text
K=2 :  648 (no MPS)  →  810 (MPS)      둘 다 이득
K=4 :  579 (no MPS)  →  864 (MPS)      MPS 없이는 본전
K=8 :  457 (no MPS)  →  878 (MPS)      MPS 없이는 손실 0.79×
```

시분할에서는 K가 커질수록 **디바이스에서 돌려주는 것이 호스트에서 얻은 것을 넘어선다.**
같은 arm이 MPS 유무로 1.92배 갈린다. → **MPS 없는 K≥4는 재지 말 것.** 그 숫자는
L5에 대한 정보가 아니라 시분할에 대한 정보다.

**4) `width_fill` 0.28 → 0.41 (+46 %).** §0.1이 판정 지표로 지목한 값이 실제로
그렇게 움직였고, `mean_width`는 예고대로 **작아졌다**. control의 0.28은 이 문서가
인용해 온 0.227보다 높다(다른 빌드·덱). **판정은 언제나 같은 실행 안의 control
대비로 한다** — 문서에 적힌 절대값과 비교하지 말 것.

**5) VRAM은 K에 대해 평평하지 않다. §2.5의 모델이 틀렸다.**

```text
K x W = 64 고정:  12.3   14.9   20.0   30.2 GB   (K = 1, 2, 4, 8)
추가 프로세스당 증분:   2.60   2.57   2.56 GB     ← 직선
```

§2.5는 “K×W를 고정하면 메모리도 그대로”라고 적었다. **슬롯 항은 실제로 평평했다** —
레버가 약속한 그대로다. 아무도 청구하지 않은 것은 **프로세스 자신**이다: CUDA 컨텍스트,
모듈 이미지, 라이브러리 핸들, 할당자 풀 — 폭의 함수가 아닌 고정비 2.56 GB.

가드를 두 항으로 고쳤다(`--vram-per-extra-process-gb`, 기본 2.56):

```text
demand(device) = K × W × per_slot_gb  +  (K − 1) × extra_process_gb
```

옛 모델에서 **16×M4는 “13 GB”였고 실제는 약 51 GB다.** 96 GB 장비에서는 어느 쪽이든
통과하지만, 그보다 작은 장비에서는 그 차이가 판정 전체이며, 예측되는 실패는
**큐를 이미 claim한 뒤 아레나 stand-up에서** 일어난다.

**6) 다음 두 arm.** 12×M6 / 16×M4로 무릎을 찾는다(`--total-width 64`이므로 폭은
올림: 12→6, 16→4). 새 모델의 예측 VRAM은 **42.8 GB / 51.4 GB**로 96 GB 안이다.
24코어에서 16 프로세스는 프로세스당 1코어이고, 그보다 큰 K는 튜너가 거절한다(§4.9).

---

### 4.9 튜너 — `--procs-per-gpu auto` (WP4)

§4.8의 표는 여덟 번의 수동 실행이다. 무릎이 장비·드라이버·MPS·덱 조합마다 움직이므로
(같은 K=8이 0.79×이기도 하고 1.52×이기도 하다), 이제 하네스가 **그 자리에서 재고**
남은 큐를 이긴 K로 돌린다.

```bash
python3 tools/run_multi_gpu_batch.py \
    --gpus 0 --procs-per-gpu auto --total-width 64 \
    --tune-candidates 1,2,4,8,12,16 --tune-jobs 16 --tune-budget-s 600 \
    --mps --claim auto --result light \
    --jobs $O/m64.txt --cwd ~/t18decks/kngr --workdir $O/work/auto \
    --pin taskset -- $B/RASBERY
```

**폭은 `--total-width`가 정한다.** `W = ceil(T/K)` — `--total-width 64`는
1×M64, 2×M32, 4×M16, 8×M8, 12×M6, 16×M4다. **올림**이므로 K개 프로세스가 T보다 좁게
선언하는 일은 없다(12×M5 = 60은 control보다 좁은 선언이고, 그것은 다른 arm이다).
`--batch-width`는 여전히 **프로세스당** 폭이며, 그 경우 선언 폭이 K에 비례해 커지므로
큰 후보는 VRAM 가드가 거절한다(하네스가 경고한다).

**캘리브레이션이 하지 않는 네 가지.**

1. **큐를 소비하지 않는다.** 후보마다 **같은** 잡 부분집합을 다시 돌린다 — 다른 잡을
   주면 폭이 아니라 덱을 비교하는 것이 된다. 그래서 재실행은 `<workdir>/tune/`으로
   **경로를 바꿔** 나가고, 캠페인은 그 뒤 매니페스트의 모든 잡을 정확히 한 번 돈다.
   `[TOTAL].jobs`는 매니페스트 잡만 세고, 재실행은 `calibration_jobs`로 따로 적힌다.
2. **생산 출력을 덮어쓰지 않는다.** 후보별 출력은 `<workdir>/tune/k<K>/out/`이다.
3. **MPS daemon을 남기지 않는다.** thread percent가 100/K이므로 후보마다 자기 서버가
   필요하고, 각각 다음 후보가 시작하기 **전에** `finally`에서 내려간다. 남았다면
   다음 후보의 “MPS 없음”이 조용히 MPS 런이 된다(§5 함정 3).
4. **캠페인을 다 쓰지 않는다.** `--tune-budget-s`는 워커가 **다음 청크를 claim하기
   전에** 검사된다. 청크 도중에 자식을 죽이면 반쯤 쓰인 출력과 아무도 완료하지 않은
   claim이 남는다.

**목적함수와 동률.** 점수는 후보의 `cases_per_hour` **중앙값**(`--tune-repeats`).
계획 §WP4의 `tail_penalty` / `failure_penalty`에서 두 가지를 의도적으로 바꿨다.

- **실패는 감점이 아니라 실격이다.** `rc≠0`, `[FAIL]` 줄, receipt 감사 실패 중
  하나라도 있으면 후보는 선택에서 **빠진다**. 안전할 만큼 큰 감점은 실격과 구별되지
  않고, 감점이라 부를 만큼 작은 감점은 “빠르고 깨진 K”를 이기게 한다.
- **tail은 감산이 아니라 동률 판정이다.** 점수를 c/h로 두어야 receipt의 숫자가
  운영자가 582와 비교하는 그 숫자로 남는다. p90 케이스 지연은 이 하네스가 재지 않으므로
  (실행 파일은 배치 점유율을 내지 케이스 wall을 내지 않는다) 대리값은 REFILL 수신증의
  `tail_idle_max_s`다.

동률(`--tune-tie-rel`, 기본 2 %)은 **(tail_idle_max_s, host thread 수, VRAM, K)**
순으로 깬다. §4.8의 MPS 표에서 864와 878은 1.6 % 차이 — **동률 밴드 안**이므로 튜너는
tail이 낮은 4×M16을 고른다. 그것이 의도다: 무릎 근처의 1.6 %는 다음 실행에서 뒤집히고,
tail이 낮은 쪽이 wave 경계에서 덜 잃는다. 원시 최대값을 원하면 `--tune-tie-rel 0`.

**후보 K=1은 기본 후보 집합에 들어 있다.** control을 고를 수 없는 튜너는 “여기서는
레버가 듣지 않는다”를 보고할 수 없고, 계획의 1.05× 킬 기준은 바로 그 보고를 요구한다.

**저장과 재사용.**

```bash
# 튜닝 결과는 <workdir>/tune/result.json 에 저장된다 (--tune-save 로 지정 가능)
python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 1 --total-width 64 \
    --tune-from $O/work/auto/tune/result.json ... -- $B/RASBERY
```

결과는 **장비 UUID와 드라이버 버전에 keyed**되어 있고, 맞지 않으면 캠페인 시작 전에
`rc=2`로 거절한다 — 다른 fleet에서 잰 K는 이 fleet의 측정이 아니다. `nvidia-smi`를
물을 수 없어 키를 확인하지 못한 경우도 거절이다(“모르는 것”은 “같다”가 아니다).
의도적으로 재사용하려면 `--allow-tune-mismatch`이며, 그때 receipt에 `key_match:false`가
남는다.

**읽어야 할 줄.**

```text
[RASBERY][MULTI_GPU][TUNE][CAND]  후보마다 한 줄, 재는 즉시 (진행 상황)
    {"procs_per_gpu":8,"batch_width":8,"declared_width_per_gpu":64,"jobs":16,
     "cases_per_hour":878.0,"width_fill":0.41,"tail_idle_max_s":...,
     "vram_per_device_gb":30.9,"mps":true,"mps_thread_percent":12,
     "declared_fidelity":"A2","fidelity_measured":{"A2":16},
     "eligible":true,"disqualified":"","problem_detail":[]}
[RASBERY][MULTI_GPU][TUNE]        하나. candidates / chosen / calibration_s /
                                  calibration_jobs / key / tie_rel
[RASBERY][MULTI_GPU][PLAN]        procs_policy:"tuned", width_policy:"total_width",
                                  writer_policy, host_thread_demand
[RASBERY][MULTI_GPU][TOTAL]       tuned / tune_source / calibration_jobs
```

**호스트 예산 receipt가 바뀌었다(WP4).** `writer_threads`는 더 이상 상수 8이 아니다 —
`src/IoWriter.h`는 writer 스레드를 **하나** 돌리고(`inline` 모드는 0), `--result light`는
결과 HDF5를 아예 쓰지 않는다. `writer_policy`가 어느 쪽인지 이름으로 말하고,
캘리브레이션 로그의 `[RASBERY][IO_WRITER]`를 읽었다면 `receipt`라고 적힌다.
`host_thread_demand`는 **코어 요구량이 아니라 스레드 수**다(레인은 대부분 랑데부에서
블록되어 있다). 기준 arm의 `host_thread_ratio` ≈ 2.7은 정상이며 경고가 아니다.

---

### 4.10 선언 fidelity와 strict 대조 arm (WP4 하네스 수정)

**무엇이 고장나 있었나 (238, `5ccf879` 이후).** 감사가 receipt의 `policy`를
하드코딩된 `strict`와 비교했다. 그런데 `DEFAULT_ENV`는 `7099e54` 이후 A2 staged
tolerance를 담고 있으므로 **모든 실행이** `policy='A2' vs strict`로 걸렸다:

```text
12×M6 wave              → dispatcher rc=3   (모든 wave 는 rc=0, dup=0)
--procs-per-gpu auto    → rc=2, 승자 없음   (후보 6개 전부 실격)
```

`RASBERY_ALLOW_SCREENING=1`은 아무 관계가 없다 — **A2는 screening이 아니다**
(`src/RunContract.h` `kFidelityTraits`). 하네스가 근사를 잡은 것이 아니라, 자기가
설정한 arm을 거절하고 있었다.

**규칙(계획 §6.2).** strict / A2 / L3coarse는 **선언되는 fidelity**이고, 감사는
“운영자가 선언한 fidelity로 실제로 풀었는가”를 검증한다. 더 낮은 fidelity를 조용히
받아들이지도, 더 높은 것을 조용히 받아들이지도 않는다 — **양방향 모두 실패**다.
strict receipt를 A2로 선언한 것도 실패인 이유: A2 칸에 다른 정책의 숫자를 넣는 것이
§6.2가 금지하는 그 혼합이다.

```bash
# 지금 무엇을 선언하고 도는지 먼저 본다 (GPU 시간 0)
python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 1 --batch-width 64 \
    --jobs $O/m64.txt --workdir /tmp/envcheck --print-env -- $B/RASBERY \
  | grep -o '"declared_fidelity":"[^"]*"'      # → "A2"  (환경에서 유도)
```

- `--fidelity strict|A2|L3coarse` (별칭 `--expect-fidelity`)로 **명시**할 수 있고,
  생략하면 자식 환경에서 유도한다(`src/RunContract.h`와 **같은 규칙**:
  `_STAGED_FLUX_TOL`/`_XE_TOL` > 1.0 → A2, `RASBERY_GA_FEEDBACK_PASSES` > 0 →
  feedback_limited, 그 외 strict, 그리고 `RASBERY_PHYSICS_FIDELITY`는 **더 거칠게만**).
- 인쇄되는 자리: `[MULTI_GPU][PLAN].declared_fidelity`(+`_source`),
  `[MULTI_GPU][ENV]`, `[MULTI_GPU][PROC].fidelity_measured`(케이스 수 단위),
  `[MULTI_GPU][TOTAL].fidelity_measured`, `[TUNE][CAND].declared_fidelity` /
  `problem_detail`, 단일 GPU 쪽은 `[SINGLE_GPU_PROFILE].declared_fidelity`.
- 튜너의 후보 감사도 **같은 선언**을 쓴다 — 그래서 A2를 선언한 캠페인의 후보가 A2라는
  이유로 실격되지 않는다. 실격되면 `[TUNE][CAND]`의 `disqualified`와
  `problem_detail`이 **두 단어를 모두** 적는다(`problems: 6`만 찍히던 것이 결함이었다).

**strict 대조 arm (표의 두 번째 행).** 같은 행렬을 strict로 한 번 더 돌린다. 세
STAGED 키뿐 아니라 `RASBERY_GA_FEEDBACK_PASSES`·`RASBERY_PHYSICS_FIDELITY`까지
**상속된 것도 포함해** 지운다 — 그것이 `--strict`가 하는 일이다:

```bash
# A2 (기본, §4.8 의 그 arm)
run_arm m64_a2   1 64 ""
run_arm m8_a2    8  8 "--mps"

# strict 대조 — 같은 덱, 같은 폭, 정책만 다르다
run_arm m64_strict 1 64 "--strict"
run_arm m8_strict  8  8 "--mps --strict"

# 확인: 선언과 실측이 모두 strict 여야 한다
grep -o '"declared_fidelity":"[^"]*"' $O/m64_strict.out | head -1
grep -o '"fidelity_measured":{[^}]*}' $O/m64_strict.out | tail -1   # {"strict":64}
```

`--set-unset KEY`는 같은 일을 키 하나 단위로 한다(`--set`과 상속 환경을 모두 이긴다).
**두 arm의 c/h는 한 표에 넣지 말고 두 표에 넣거나, 행마다 `policy` 열을 붙인다.**
A2는 수렴 정책이 다르므로 케이스당 outer 수가 다르고, 그것이 바로 나누어지고 있는 그
양이다 — 라벨 없는 비교는 폭 레버가 아니라 tolerance를 재는 것이 된다.

---

## 5. 함정 목록 (runner용)

| # | 함정 | 증상 | 대응 |
|---|---|---|---|
| 1 | `RASBERY_XE_ANDERSON`을 export하지 않음 | 단일/배치 궤적이 다르고 `h5diff`가 전부 깨진다 | §4.1대로 `=1` export. `[PHYSICS_MODE].xe_anderson`으로 확인 |
| 2 | `--pin none`으로 K 측정 | K배 오버섭스크립션. L5가 아니라 경합을 잰다 | `--pin taskset`, `[PROC].cpus`가 disjoint인지 확인 |
| 3 | MPS daemon 잔존 | 다음 "MPS 없음" arm이 조용히 MPS 런 | `pgrep -a nvidia-cuda-mps` 확인 |
| 4 | `mean_width` 하락을 실패로 읽음 | 작동하는 arm을 버린다 | `width_fill`로 본다 (§0.1) |
| 5 | `--workdir`를 arm 간 재사용 | 로그·MPS 파이프가 섞인다 | arm마다 다른 workdir |
| 6 | `[GPU].wall_s`를 합으로 오해 | 처리량이 K로 나뉜다 | 이미 최대값이다. `[TOTAL].cases_per_hour`를 쓸 것 |
| 7 | ~~`RASBERY_ALLOW_SCREENING` 미설정~~ **(WP1에서 소멸)** | 없음 — light는 screening이 아니다 | export해도 무해하지만 아무것도 고치지 않는다. rc=3의 원인은 fidelity 선언 불일치이지 이 권한이 아니다(함정 18) |
| 8 | control이 **582 c/h**를 ±5 %로 재현하지 못함 | 모든 배수가 무의미 | 배수를 보고하기 전에 control부터 맞출 것 (§4.7) |
| 9 | `--driver-workers`/`--no-oversubscribe`로 레인을 코어 수까지 낮춤 | `width_fill`이 무너지고 c/h가 5배 떨어진다. 오류는 없다 | `[PLAN].driver_workers`가 폭 W와 같은지, `driver_worker_policy`가 `binary_default_width`인지 확인 |
| 10 | 물리 키를 export로만 주고 `--set`을 안 씀 | `DEFAULT_ENV`가 이겨서 조용히 다른 arm이 돈다 | A/B는 반드시 `--set KEY=VALUE`로. `[MULTI_GPU][ENV]`로 확인 |
| 11 | MASTER 대조인데 `RASBERY_PPR_MODE`를 안 줌 | 핀 파워가 SENM 경로로 나온다 | `--set RASBERY_PPR_MODE=master` (더 이상 기본값이 아니다, §4.1) |
| 12 | **MPS 없이 K≥4를 재고 “무릎”이라고 보고** | 8×M8이 0.79×로 나온다. 오류는 없다 | K가 커지면 MPS는 선택이 아니라 전제다(§4.8-3). MPS 없는 K≥4는 시분할을 재는 것이다 |
| 13 | VRAM을 옛 한 항 모델로 예산 | 16×M4가 “13 GB”, 실제 약 51 GB. 아레나 stand-up에서 죽는다 — 큐를 claim한 뒤에 | `[VRAM].extra_process_gb`가 두 번째 항이다(§4.8-5) |
| 14 | `--procs-per-gpu auto --dry-run` | rc=2 | 튜너는 **잰다**. 마른 캘리브레이션은 아무것도 돌지 않은 K를 고른 것이 된다 |
| 15 | 튜닝 결과를 다른 장비·드라이버에서 재사용 | rc=2 (거절) | 결과는 UUID+드라이버에 keyed다. 의도적이면 `--allow-tune-mismatch`, 그러면 `key_match:false`가 남는다 |
| 16 | 캘리브레이션 잡을 캠페인 처리량에 섞어 읽음 | 배수가 “튜닝이 얼마나 걸렸나”의 함수가 된다 | `[TOTAL].jobs`는 매니페스트 잡만, 재실행은 `calibration_jobs` |
| 17 | `--total-width` 없이 튜닝 | 선언 폭이 K에 비례해 커져 큰 후보가 전부 VRAM 거절 | `--total-width 64`. 하네스가 경고는 하지만 막지는 않는다 |
| 18 | **fidelity 선언과 실행 불일치** | `[PHYSICS_MODE] policy='A2' ... DECLARED 'strict'` → dispatcher rc=3, 튜너는 rc=2·승자 없음. **wave 자체는 rc=0, dup=0이라 무엇이 틀렸는지 안 보인다** | `--print-env`로 `declared_fidelity`를 먼저 본다(§4.10). A2 arm이면 아무것도 하지 않아도 되고, strict를 재려면 `--strict`. `RASBERY_ALLOW_SCREENING`은 이것과 무관하다 |
| 19 | A2 숫자와 strict 숫자를 한 표에 | 폭 레버가 아니라 tolerance를 재게 된다 | 행마다 `policy` 열. `[TOTAL].fidelity_measured`가 케이스 단위로 적어 준다(계획 §6.2, §4.10) |
| 20 | **K가 잡 수를 나누지 못할 때 워커 하나가 0잡** | 12×M6/64잡에서 12번째 프로세스가 아무것도 claim하지 못했다. c/h는 0, 그 슬롯은 wave 내내 놀았고 arm은 “12×M6”으로 기록된다 | 수정됨: `auto`의 claim은 이제 `floor(jobs/(G·K))` + 나머지 라운드로빈을 **상한**으로 쓴다. `[PROC].jobs`가 전부 ≥1인지 확인할 것 |

---

## 6. 남은 것

| 항목 | 상태 | 왜 지금 아닌가 |
|---|---|---|
| **MPS 성능** | **실측 (§4.8)** | 8×M8+MPS = 878 c/h, 같은 arm이 MPS 없이는 457. 1.92배 |
| **238의 배수** | **실측 (§4.8)** | control 578 대비 **1.519×** (8×M8+MPS). 킬 기준 1.05×는 2×M32가 MPS 없이도 통과(1.121×) |
| **무릎의 위치** | 부분 실측 | 8까지는 단조 증가(MPS). 12×M6 / 16×M4가 다음이고, 그 뒤로는 24코어에서 프로세스당 1코어 미만이 되어 튜너가 거절한다 |
| **다중 GPU × K** | 미실측 | 코드 경로는 같다(`processes = G × K`). GPU 1장 전용 정책 때문에 로컬에서 잴 수 없다 |
| **per-slot VRAM 재측정** | **두 항 모델로 교체 (§4.8-5)** | 슬롯 항 0.203 GB는 그대로(K에 대해 평평한 것이 실측으로 확인되었다). 새로 생긴 것은 프로세스 고정비 2.56 GB — `--vram-per-extra-process-gb`. `--result light`의 실제 peak은 여전히 더 작고, 가드는 **보수적인 쪽**이 옳다 |
| **`--claim auto`의 K 인식** | 구현됨, 미튜닝 | `min(max(W, ceil(remaining/(2·G·K))), floor(잡수/(G·K))+나머지)`. 공정몫 상한이 워커 기아를 막고(함정 20), 그 아래에서는 예전 정책 그대로다. K가 크면 청크가 폭으로 고정되어 스틸이 사실상 사라진다 — 128잡 이상에서 다시 볼 것 |
