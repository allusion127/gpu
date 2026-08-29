# W4-lite: 즉시 슬롯 리필(Task 20)과 멀티GPU dispatcher

브랜치 `codex/exact-throughput-campaign`, 기준 `a6768f7`
커밋 `c4ff738`(리필), `a9c3e58`(dispatcher), `<이 문서>`
측정 하드웨어: 로컬 WSL, GTX 1080 Ti(sm_61) 1장, nvcc 12.6 / gcc 13.3

---

## 0. 결론 먼저

**리필은 이미 동작하고 있었다.** `--batch-mode M`의 배치 분기는 `schedule(dynamic, 1)`
OpenMP 큐이고, Driver는 생성자에서 아레나 슬롯을 얻고 소멸자에서 반납한다. 따라서
잡 수가 M보다 많으면 덱 하나가 끝난 워커가 **배치를 비우지 않고** 즉시 다음 잡을
집어 새 Driver를 만들며, 그 Driver가 슬롯을 다시 얻는다. `releaseSlot`이
`inUseCount()`를 낮추고 rendezvous를 깨우므로, 테넌트 사이에 있는 슬롯이 남은 덱을
붙잡지 않는다.

W4-lite가 추가한 것은 **동작이 아니라 계약**이다. 세 가지다.

| 추가 | 없으면 무엇이 무너지는가 |
|---|---|
| `--jobs <manifest>` | Task 20 수용 케이스는 1,280잡. argv로는 ~200 kB — `MAX_ARG_STRLEN` 초과 |
| `[RASBERY][REFILL]` 수신증 | "tail이 사라졌다"가 스톱워치에 대한 주장으로만 남는다 |
| 테넌시 감사(`duplicates` / `stale_tenants` / `double_releases`) | 리셋 경로에 구멍이 생겨도 아무것도 말해주지 않는다. 다음 테넌트가 앞 덱의 residency 플래그로 물리를 계산하고, 모든 값이 유한하고 그럴듯하다 |

멀티GPU dispatcher(`tools/run_multi_gpu_batch.py`)는 **GPU당 프로세스 1개 + flock 공유
큐**다. `--gpus 0` 단일 형태에서 기존 단일GPU 하네스와 **wall 51.04 s vs 51.12 s,
출력 바이트 동일**로 parity를 확인했다.

---

## 1. Task 20 — 즉시 리필

### 1.1 리필의 실체

```
워커 L: Driver(deck k)  ──소멸자──▶ releaseSlot(m)  ──▶ inUseCount()-- , cv.notify_all()
                                        │
                                        └── OpenMP dynamic 큐에서 deck k' 획득
                                            Driver(deck k') ──생성자──▶ acquireSlot() → m'
                                                                         slot = Slot{}  (전량 리셋)
```

CMFD 아레나(`CudaBICGBackend.cu:4602` `CudaBatchArena::acquireSlot`)와 nodal 아레나
(`CudaXsReconBackend.cu:634` `NodalArena::acquireSlot`)가 각각 슬롯을 스캔해 `Slot{}` 통짜 대입으로 리셋한다. nodal은 구조체 **밖**에
사는 `_canon`(`CanonicalSlotBuffers`)도 함께 비운다 — Task 18에서 실제로 새어나갔던
필드다.

### 1.2 추가된 코드

| 파일 | 내용 |
|---|---|
| `src/BatchRefill.h` (신규) | `refill::Ledger`(테넌시 원장, 수신증 산술) + `refill::TenancyCounters`(버그 카운터 4종). CUDA-free 헤더 — `.cu` 두 곳과 `main.cpp`가 모두 본다 |
| `src/main.cpp:142` | `rasberyReadJobManifest()` — `--jobs` 매니페스트 파서 |
| `src/main.cpp` 배치 분기 | `ledger().begin/jobStarted/jobFinished/end/report`. **Driver를 try 안에서 별도 스코프**로 감싼다 |
| `src/CudaBICGBackend.cu` | `batchSlotIsReset()` 사후조건 감사, rendezvous 큐 중복 검출, double-release 검출 |
| `src/CudaXsReconBackend.cu` | `nodalSlotIsReset()`(Slot + `_canon`), 동일한 큐/릴리스 검출 |
| `tools/test_batch_refill_contract.py` (신규) | 위 전부의 정적 계약 |

**Driver 스코프가 왜 중요한가.** 소멸자가 슬롯을 반납한다. 소멸자가 `jobFinished()`
**뒤에** 돌면, 측정된 리필 지연에서 teardown이 빠진다 — 이 태스크가 작게 유지해야 할
바로 그 부분이. 계약 테스트가 이 스코프를 고정한다.

### 1.3 `--jobs` 매니페스트 형식

```text
# 한 줄에 잡 하나. `#`는 주석, 빈 줄은 무시.
# 공백이 든 경로는 " 로 감싼다.
d0.json              /out/d0.h5
"deck with space.json"  "/out/a b.h5"
```

- `--rasi`/`--raso`와 **같은 두 벡터**에 append된다. 따라서 출력 네임스페이스 중복
  금지 규칙과 개수 일치 규칙이 매니페스트 잡에도 그대로 적용된다. 플래그와 혼용 가능.
- 확장은 **모든 검증보다 먼저** 일어난다. 계약 테스트가 이 순서를 고정한다 — 뒤로
  가면 매니페스트가 두 잡을 한 `--raso`에 몰래 넣을 수 있다.
- CRLF를 잘라낸다. Windows에서 쓴 매니페스트를 WSL에서 읽는 것이 이 캠페인의 상비
  함정이다(APR1400 CRLF 건).

### 1.4 수신증

```json
[RASBERY][REFILL] {"jobs":8,"slots":4,"lanes":4,"lanes_used":4,
 "lanes_never_admitted":0,"refills":4,"wall_s":23.2067,"tail_idle_s":3.97695,
 "slot_busy_fraction":0.957146,"refill_latency_p50_ms":0.000684,
 "refill_latency_max_ms":0.000821,"admissions":16,"duplicates":0,
 "stale_tenants":0,"double_releases":0}
```

| 필드 | 정의 |
|---|---|
| `slots` | 아레나 폭 M(`--batch-mode`) |
| `lanes` | 호스트 Driver 워커 수. `main.cpp`가 M으로 상한을 건다 |
| `refills` | 잡 수 − 실제 사용된 레인 수. 즉 "이미 덱을 끝낸 레인에 다시 들어간 admission" 수 |
| `tail_idle_s` | Σ레인 (배치 종료 − 그 레인의 마지막 반납). **이 태스크가 움직여야 하는 수** |
| `slot_busy_fraction` | Σ테넌시 길이 / (사용 레인 수 × wall) |
| `refill_latency_*` | 같은 레인에서 한 테넌시 종료 → 다음 테넌시 시작까지의 간격. **덱 import 비용은 여기가 아니라 테넌시 안에 계상된다** |
| `admissions` | 두 아레나의 `acquireSlot` 성공 총합. 8잡이면 CMFD 8 + nodal 8 = **16이 정상** |
| `duplicates` / `stale_tenants` / `double_releases` | 통계가 아니라 **버그 카운터**. 하나라도 0이 아니면 그 런은 유효한 측정이 아니다 |

**감사 정책.** 큐 중복은 `throw`(fail-closed) — 슬롯을 두 번 스테이징하면 두 번째가
첫 참가자의 연산자를 덮어쓴다. stale reset은 **세고 넘어간다** — 감사가 도는 시점에
상태는 이미 올바르다(리셋 직후다). 여기서 배치를 죽이면 수신증 한 줄과 런 전체를
맞바꾸는 셈이다. 게이트는 `stale_tenants: 0`이다.

---

## 2. 게이트 표 (로컬 1080 Ti, 덱 d0..d3 + 중복 e0..e3)

기준선: 단일 모드 4덱 순차 = **20.80 s**.

| # | 게이트 | 결과 |
|---|---|---|
| G1 | 8잡 / 폭 4 (`--jobs`) | rc=0, wall **23.49 s**, refills 4, tail_idle **3.977 s**, slot_busy **0.957**, dup 0, stale 0 |
| G1b | G1 출력 8개 전부 `h5diff -c 0` vs 단일 모드 | **PASS** (d0..d3, e0..e3 모두) |
| G2 | 12잡 / 폭 4 | rc=0, wall **34.71 s**, refills 8, tail_idle **7.80 s**, slot_busy **0.944** |
| G2b | G2 출력 12개 전부 `h5diff -c 0` vs 단일 모드 | **PASS** |
| G3 | ×2 결정론(8잡 재실행) | 8/8 run-to-run 동일 |
| G4 | feature-off(신규 바이너리, `--jobs` 없이 `--rasi` 8덱) vs `a6768f7` | 8/8 **바이트 동일** |
| G5 | capture arbiter | `alloc_in_capture 0`, `alloc_blocked 0`, `captures_unwound 0` |
| G6 | `ctest` | **12/12 passed** |
| G7 | 계약 스윕 `for t in tools/test_*.py` | 신규 `test_batch_refill_contract.py` PASS, `test_multi_gpu_dispatch.py` PASS. 실패는 브랜치 이전부터의 4건(`cmfd_fp32`, `ga_feedback_screen`, `ga_promotion_gate`, `nodal_constant_cache`)뿐 — **신규 실패 0** |
| W3 | 8잡 리필 vs 폭-4 배치 2회 순차 | **23.49 s vs 24.79 s = 1.055×** (parity 이상) |

**W3을 정직하게 읽는 법.** 덱이 5.2 s로 짧아서 tail이 절대량으로 작다. 리필이 지운
것은 "중간 드레인 1회 + 프로세스 재기동 1회"이고, 그게 1.3 s다. 238의 M64처럼 덱이
길고 폭이 넓을수록 `tail_idle_s`가 차지하는 비율이 커지므로 이 배수는 하한으로 읽어야
한다. **보고해야 할 지표는 배수가 아니라 `tail_idle_s`다.**

---

## 3. 멀티GPU dispatcher — `tools/run_multi_gpu_batch.py`

### 3.1 설계

```
manifest ──▶ queue.json (flock 커서)
               │  claim(size) = 원자적 read-modify-write
               ├──▶ GPU0 프로세스열: chunk0001 → chunk0002 → ...
               ├──▶ GPU1 프로세스열: ...
               └──▶ GPU2 ...
                    각 프로세스: CUDA_VISIBLE_DEVICES=<i>
                                 taskset/numactl로 CPU 몫에 고정
                                 RASBERY --jobs <chunk> --batch-mode M
                                   └─ 프로세스 안에서는 Task 20 리필이 tail을 먹는다
```

**GPU당 프로세스인 이유.** 이 코드베이스의 아레나(CMFD/nodal/physics)는 전부
프로세스 수명 싱글턴이고 현재 디바이스에 대해 폭 M으로 한 번 사이징된다. rendezvous는
한 스트림에 대해 launcher 하나를 뽑는다. 디바이스 인식을 넣는 것은 배치 코어 재작성이고,
`CUDA_VISIBLE_DEVICES`는 런처 한 줄이다. 각 프로세스가 자기를 device 0으로 믿으므로
C++는 손대지 않아도 되고, per-GPU 숫자가 단일GPU 하네스와 직접 비교된다.

**정적 분할이 아니라 공유 큐인 이유.** 정적 분할은 가장 느린 파티션이 끝날 때 끝난다.
덱마다 statepoint 수가 다르고, 혼합 호스트면 GPU도 다르다. claim은 flock 하나짜리
read-modify-write이므로, 마른 GPU가 남은 것을 훔치고 느린 GPU는 그냥 덜 가져간다.
같은 큐 파일에 두 번째 dispatcher를 붙여도 안전하다 — 반쯤 끝난 캠페인의 재개가 그래서
가능하다. `--claim all`은 정적 분할로 축퇴한다.

**GPU 1장 + `auto`는 큐 전체를 한 번에 claim한다.** 훔쳐올 상대가 없고, 쪼개면 청크
사이에서 배치가 **드레인**된다 — Task 20이 없앤 바로 그 tail이다. 실측:

| 12잡 / 폭 4 / GPU 1장 | wall | c/h | tail_idle | 프로세스 |
|---|---:|---:|---:|---:|
| `--claim auto` (1회 claim) | 88.33 s | **489.1** | 16.58 s | 1 |
| `--claim 4` (3회 claim) | 102.07 s | 423.3 | 26.99 s | 3 |

출력은 양쪽 동일. 청크 분할은 **13.5% 손해**다.

**감사 재사용.** exact-only physics-mode 감사, graph-fallback 감사, rc/FAIL 집계는
`run_single_gpu_batch.py`에서 **import**한다. 유효한 런의 정의가 한쪽 하네스에만
적용되는 사태를 막는다.

### 3.2 parity (요구사항: 멀티프로세스 경로가 단일GPU 하네스 수치로 축퇴할 것)

| 8잡 / 폭 4 / GPU0 | wall | c/h | 출력 |
|---|---:|---:|---|
| `run_multi_gpu_batch.py --gpus 0` | **51.04 s** | 564.3 | — |
| `run_single_gpu_batch.py` (동일 덱·동일 프로필) | **51.12 s** | 563.4 | 위와 **바이트 동일** |

> 51 s는 §2의 23.5 s와 다르다. 두 하네스가 공유하는 `DEFAULT_ENV` 프로필
> (`RASBERY_PPR_MODE=master`, `RASBERY_PC_MODE=decart`, `RASBERY_GPU_RB_SWEEPS=4`)이
> §2의 수동 실행과 다른 물리 설정이기 때문이다. **회귀가 아니다** — parity 비교는
> 같은 프로필끼리 한 것이다.

### 3.3 호스트 예산표 — 4,340 c/h (계획 §13.3)

계획의 예산: **라이터 ~8스레드/프로세스 + GPU당 로더 8~16, NUMA 고정, 프로세스-per-GPU.**

```text
목표 = 20 × 217 = 4,340 c/h
N = ceil(4340 / (1-GPU c/h × 0.85))
```

| 1-GPU c/h | 필요 GPU N | 프로세스 | 라이터 스레드 (8×N) | 로더/GPU | 총 호스트 스레드 | 권장 CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 650 | 8 | 8 | 64 | 8~16 | 128~192 | 192 |
| 900 | 6 | 6 | 48 | 8~16 | 96~144 | 144 |
| **1,400** | **4** | 4 | 32 | 8~16 | 64~96 | **96** |
| 2,000 | 3 | 3 | 24 | 8~16 | 48~72 | 72 |

dispatcher가 실제로 계산하는 분배(`plan_host_budget`):

```text
cpus_per_gpu   = visible_cpus // N                       ← 겹치지 않는 taskset 구간
driver_workers = min(batch_width, cpus_per_gpu)          ← 리필 레인 = RASBERY_BATCH_HOST_THREADS
spare          = cpus_per_gpu − driver_workers − 8       ← 8 = 라이터 몫
solver_threads = max(1, min(3, 1 + spare/driver_workers))← RASBERY_OMP_THREADS
```

예: 96 CPU / 4 GPU / 폭 64 → GPU당 24 CPU, 워커 24, 라이터 8, 솔버 1.
`--driver-workers` / `--solver-threads`로 덮어쓴다. 계획의 "로더 8~16"은 이
`driver_workers`(덱 import + Driver 오케스트레이션)에 해당하며, 폭 64에서 CPU 몫이
16 미만이면 그 GPU는 폭을 다 못 채운다 — 그때는 GPU를 줄이거나 CPU를 늘려야 한다.

**CPU 집합은 반드시 서로 겹치지 않아야 한다.** 두 프로세스가 96코어를 각자 다
가졌다고 믿으면 정확히 2배 오버섭스크립션이 나고, `OMP_PROC_BIND=TRUE`가 그 스레드들을
같은 place에 고정한다. 단위 테스트가 집합 disjoint를 고정한다.

---

## 4. 238 서버 실행 명령 (GPU0 전용 — GPU1은 사용 금지)

`--gpus 0`만 쓴다. dispatcher는 `CUDA_VISIBLE_DEVICES=0`을 자식에게 강제하므로
GPU1은 자식 프로세스에서 보이지 않는다.

### 4.1 매니페스트 만들기

```bash
cd ~/t18decks/kngr
mkdir -p ~/w4out/m64
: > ~/m64jobs.txt
for i in $(seq 0 63); do
  printf '%s %s\n' "kngr_$i.json" "$HOME/w4out/m64/case$i.h5" >> ~/m64jobs.txt
done
wc -l ~/m64jobs.txt      # 64
```

64잡보다 **많은** 잡(리필이 실제로 일하는 형태)을 쓰려면 `--batch-mode 64`는 그대로
두고 매니페스트만 늘린다. 폭은 아레나 크기이지 잡 수가 아니다.

### 4.2 M64 리필 실행 (권장)

```bash
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$($HOME/opt/bin/micromamba shell hook -s bash)"; micromamba activate gpu

# DEFAULT_ENV에 없는 키는 환경에서 그대로 상속된다.
export RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 RASBERY_GPU_WIEL_FOLD=chunked
export RASBERY_GPU_XE=1 RASBERY_XE_ANDERSON=1
export RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8

python3 tools/run_multi_gpu_batch.py \
    --gpus 0 \
    --batch-width 64 \
    --claim auto \
    --jobs ~/m64jobs.txt \
    --cwd ~/t18decks/kngr \
    --workdir ~/w4out/m64work \
    --pin taskset \
    --set RASBERY_OMP_THREADS=12 \
    -- ~/<BUILD>/RASBERY
```

- `--cwd`는 필수다. 덱의 단면 HDF5와 restart 네임스페이스가 덱 디렉터리 기준 상대
  경로다.
- `--set`이 필요한 키는 `DEFAULT_ENV`가 덮어쓰는 키뿐이다
  (`RASBERY_OMP_THREADS`, `RASBERY_BATCH_WAIT_US`, `RASBERY_NODAL_BATCH_WAIT_US`,
  `RASBERY_PPR_MODE`, `RASBERY_PC_MODE`, `RASBERY_GPU_RB_SWEEPS`, …).
  그 외는 위처럼 `export`로 통과한다.
- `--pin taskset`은 GPU 1장이면 전체 CPU를 그 프로세스에 준다(무해). 2장 이상에서만
  실제로 쪼갠다.

### 4.3 리필 없이 기존 형태로 재확인하고 싶을 때

```bash
python3 tools/run_single_gpu_batch.py --batch-width 64 --gpu 0 -- \
    ~/<BUILD>/RASBERY --rasi <64 decks> --raso <64 outputs> --batch-mode 64
```

이쪽도 이제 `[RASBERY][REFILL]`을 낸다(잡 = 폭이면 `refills: 0`).

### 4.4 읽어야 할 줄

```bash
grep -h "REFILL\]\|MULTI_GPU\]\|CAPTURE_ARBITER\|BATCH_OCCUPANCY" ~/w4out/m64work/*.log
```

수용 조건: `duplicates: 0`, `stale_tenants: 0`, `double_releases: 0`,
`alloc_in_capture: 0`, `graph_fallbacks: 0`, `rc: 0`, `fail_lines: 0`.
성능 판정은 `[RASBERY][MULTI_GPU][TOTAL].cases_per_hour`와 `tail_idle_s`로 한다.

---

## 5. 남은 것

| 항목 | 상태 | 왜 지금 아닌가 |
|---|---|---|
| **혼합 지오메트리 admission** | 미구현 | 두 아레나 모두 `compatible()`이 공유 불변 지오메트리와 **memcmp 동일**을 요구한다. 다른 지오메트리 덱은 슬롯을 못 얻고 CPU 폴백으로 떨어진다. 매니페스트에 지오메트리 해시를 넣어 코호트를 나누는 것이 다음 단계 — 잡 큐가 있으니 dispatcher 쪽 작업이다 |
| **슬롯별 아레나 리사이즈** | 미구현 | 아레나는 프로세스 시작 시 폭 M으로 사이징된다. 잡이 M보다 적게 남은 꼬리에서 M−k 슬롯이 VRAM만 잡고 논다. M64에서 실측 낭비를 먼저 재야 한다 |
| **1,280잡 대량 안정성** | 미실행 | 로컬은 12잡까지만 실행. 238에서 §4.2 형태로 확인해야 계획 Task 20 Step 8이 닫힌다 |
| **다중 GPU 실측** | 미실행 | 로컬 1장, 238 GPU0 전용. `--gpus 0,1,...`의 스케일링 효율 0.85 가정은 아직 측정값이 아니다 |
| **`k_audit_tenant_reset` 디바이스 커널** | 미구현 | 계획 §3.2의 디바이스측 감사. persistent 스케줄러(W3.7)가 종결됐으므로 4구조체가 디바이스에 상주하지 않는다. 현재의 호스트측 사후조건 감사가 같은 불변식을 같은 지점에서 검사한다 |
| **`refill_latency`의 의미 확장** | 의도적 | 지금은 "레인이 비어 있던 간격"이고, 덱 import는 테넌시 안에 계상된다. import를 분리하려면 Driver 생성자에 계측을 넣어야 한다 — Task 20의 판정 지표는 아니다 |
