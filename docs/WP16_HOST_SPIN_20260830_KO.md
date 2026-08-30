# WP16 — 호스트 스핀: 79 % CPU와 92 % 동기화가 동시에 참일 수는 없다 (2026-08-30, KO)

브랜치 `codex/exact-throughput-campaign`, 기준 `cd3630c`.
입력은 238에서의 관측 하나이고, 이 문서는 **노브 2개의 구현**과
**그 노브로 측정해야 할 런북**이다. 아직 측정치는 없다 — 이 문서의 §3·§4는
채워질 표이지 채워진 표가 아니다.

---

## 0. 요약

1. **관측(238).** batch 8 프로세스 × `RASBERY_OMP_THREADS=8`, 24 CPU에서
   호스트 CPU **~79 %**인데 XFER ledger는 프로세스 시간의 **~92 %가
   `cudaStreamSynchronize` 안**이라고 말한다. 둘 다 참이면 그 79 %는 일이 아니다.
2. **가설.** CUDA 기본 스케줄은 `cudaDeviceScheduleAuto`이고, Auto는
   **활성 컨텍스트 수 < 논리 코어 수일 때 스핀**한다 — 8-on-24가 정확히 그 모양이다.
   동시에 libgomp의 배리어 기본값도 능동 대기다. 그래서 CPU는 **일이 아니라
   폴링으로 바쁘고**, 8개 프로세스의 폴러가 24코어를 두고 서로 경합한다.
3. **노브 2개를 추가했다. 기본값에서는 아무것도 바뀌지 않는다.**
   * `RASBERY_CUDA_SYNC_MODE = auto | spin | yield | blocking`
     → `cudaSetDeviceFlags(cudaDeviceScheduleAuto/Spin/Yield/BlockingSync)`,
       **프로세스당 1회, CUDA 컨텍스트 생성 전에**.
       미설정이면 **CUDA 진입점을 아예 호출하지 않는다** (§1.2).
   * `RASBERY_OMP_WAIT = active | passive`
     → `OMP_WAIT_POLICY`(+ passive면 `GOMP_SPINCOUNT=0`)를 **OpenMP 재-exec 전에**
       export. 미설정이면 종전과 바이트 단위로 동일한 `PASSIVE`/`0` 비덮어쓰기.
4. **B0 by construction.** 두 노브 모두 피연산자·커널·런치 순서·스트림을 건드리지
   않는다. **이미 기다리던 펜스를 어떻게 기다리는가**만 바꾼다. 그래서 모든 arm의
   결과는 비트 단위로 같아야 하고, 같지 않으면 그것이 결함이다.
5. **측정해야 할 trade는 하나다.** `blocking`은 코어를 돌려주는 대신 동기화마다
   **깨우기 지연(wake-up latency)**을 낸다. 배치(8프로세스)에서는 되찾은 코어가
   이기고 단일 덱에서는 깨우기 지연이 이길 가능성이 높다 — §4가 그 확인이다.

---

## 1. 구현

### 1.1 어디에 걸었나

| 항목 | 위치 |
|---|---|
| 플래그 적용 | `src/CudaHostSchedule.cu` — `ApplyCudaHostSchedule()` |
| CPU 전용 스텁 | `src/CudaHostScheduleStub.cpp` (`rc:"no-cuda"`) |
| 호출 지점 | `src/main.cpp`, `rasberyPrepareOpenMPStartup(argv);` **바로 다음 줄** |
| OMP 훅 | `src/main.cpp` `rasberyPrepareOpenMPStartup()` 안, `execvp` **앞** |
| 영수증 | `[RASBERY][CUDA][SCHED]` 한 줄 (두 노브 모두) |
| 계약 시험 | `tools/test_cuda_sched_contract.py` |

**왜 main() 맨 위인가.** `cudaSetDeviceFlags`는 이 프로세스에 CUDA 컨텍스트가
**없는 동안에만** 유효하다. 이 트리에는 명시적 `cudaSetDevice`가 한 곳도 없어서
primary context는 `Drive()` 중에 **먼저 CUDA 런타임을 부른 백엔드가 만든다**
(`GpuPhysicsArenaCuda.cu:101`, `CudaXsReconBackend.cu:2817`,
`CudaBICGBackend.cu:3114`, `CudaPprBackend.cu:1709`, `CudaCramBackend.cu:833`).
따라서 "모든 컨텍스트 생성보다 앞"이 증명되는 위치는 `main()`의 맨 위 하나뿐이다.

**왜 `execvp` 앞인가.** libgomp는 `OMP_WAIT_POLICY`/`GOMP_SPINCOUNT`를
**라이브러리 생성자에서**, 즉 `main()` 진입 전에 읽는다. 이미 돌고 있는 런타임에
정책을 바꿔 넣을 방법은 없고, **아직 런타임을 시작하지 않은 프로세스**에 넣는 것뿐이다.
그 프로세스가 `rasberyPrepareOpenMPStartup()`의 재-exec 자식이다.

### 1.2 기본값이 "변화 없음"인 방식

`RASBERY_CUDA_SYNC_MODE`가 미설정이면 `ApplyCudaHostSchedule()`은
`cudaSetDeviceFlags`도 `cudaGetDeviceFlags`도 부르지 않고 즉시 반환한다.
`cudaSetDeviceFlags(cudaDeviceScheduleAuto)`가 이론상 no-op인 것과, 기본 경로가
**CUDA 런타임에 진입하지 않는 것**은 다른 주장이다. 후자만 증명 가능하고,
`--help`나 CPU 전용 덱, 드라이버가 깨진 호스트에서 차이가 난다.
명시적 `RASBERY_CUDA_SYNC_MODE=auto`는 반대로 **호출한다** — 그래서
`source:"default"`와 `source:"env"`가 영수증에서 구분된다.

오타(`blockng` 등)는 실패가 아니라 **기본 경로 + `source:"invalid"` + `note`**다.
1,280덱 웨이브를 시작 시점에 죽이는 것이 더 나쁜 실패이기 때문이고, 대신 그 행은
영수증에서 `auto` 행으로 표시되어 `blocking` 열에 잘못 집계되지 않는다.

### 1.3 영수증

```
[RASBERY][CUDA][SCHED] {"mode":"blocking","source":"env",
  "requested":"cudaDeviceScheduleBlockingSync","applied":"blocking","rc":"cudaSuccess",
  "omp_wait":"passive","omp_wait_source":"env",
  "omp_wait_policy":"PASSIVE","gomp_spincount":"0"}
```

`applied`는 **가정이 아니라 `cudaGetDeviceFlags`로 되읽은 값**이다.
너무 늦게 부른 arm은 `rc:"cudaErrorSetOnActiveProcess"`, `applied:"auto"`로 나오고,
그 행은 `blocking` 행이 아니다. 이 구분이 없으면 순서 결함은
**아무 오류도 내지 않고 처리량 숫자로만** 나타난다.

### 1.4 `active` arm에서 `GOMP_SPINCOUNT`를 지우는 이유

libgomp에서 스핀 카운트는 정책 단어를 **이긴다**. `GOMP_SPINCOUNT=0`은
"스핀하지 마라"이므로, `OMP_WAIT_POLICY=ACTIVE`와 함께 두면 ACTIVE는 이름뿐이다.
키를 **지우면** libgomp가 ACTIVE의 기본(무한 스핀)을 쓴다. 그래서 active arm은
`unsetenv("GOMP_SPINCOUNT")`이고, 이 삭제 자체가 `changed`를 세워 재-exec를 만든다.
`tools/run_single_gpu_batch.py`의 `DEFAULT_ENV`가 `PASSIVE`/`0`을 이미 export하므로,
덮어쓰지 않는 set은 no-op이 된다 — 계약 시험이 이 두 가지를 모두 지킨다.

---

## 2. 238 런북

### 2.0 공통

* 계산은 **로컬에서 하지 않는다**. 238에서 돌리고 출력은 `E:` 쪽 run 디렉터리.
* 두 노브 모두 **`DEFAULT_ENV`에 넣지 않았다**. 238 기준선 환경은
  `test/reference/batch_reference_env_238.json`에 키 단위로 고정되어 있고,
  기준선이 갖지 않은 하네스 기본값은 정확히 `RASBERY_PPR_MODE=master`가 저질렀던
  침묵의 편차다. 둘 다 평범한 `--set` 오버라이드로 전달한다.
* 매 arm마다 자식의 `[RASBERY][CUDA][SCHED]`와 디스패처의
  `[RASBERY][MULTI_GPU][ENV]`를 **로그에 남기고 표에 함께 적는다**.
  `applied`가 요청과 다르면 그 행은 데이터가 아니다.
* **B0 확인은 매 arm 필수.** 같은 덱의 keff/핀출력이 `auto` arm과 비트 단위로
  같아야 한다 (`tools/compare_keff.py`, `tools/gate_b_pin_rms.py`).
  다르면 노브의 결함이지 물리가 아니다.

### 2.1 배치: 8×M8 + MPS, v6 env × {auto, blocking, yield} × {active, passive}

6 arm. arm 하나당 명령은 이 한 줄이고 `--set` 두 개만 바뀐다.

```sh
python tools/run_multi_gpu_batch.py \
    --gpus 0 --procs-per-gpu 8 --batch-width 8 --mps \
    --jobs manifest.txt --pin taskset \
    --workdir "$RUN/wp16_cuda-${SCHED}_omp-${WAIT}" \
    --set RASBERY_CUDA_SYNC_MODE=${SCHED} \
    --set RASBERY_OMP_WAIT=${WAIT} \
    -- ./build/RASBERY
```

`SCHED ∈ {auto, blocking, yield}`, `WAIT ∈ {active, passive}`.
`auto`/`passive` 조합이 **현행 v6 기준선**이다(단, `auto`는 여기서 명시적으로
설정되므로 `source:"env"`로 찍힌다 — 진짜 무설정 기준선을 한 행 더 원하면
`--set` 두 개를 모두 빼고 한 번 더 돌린다).

측정 중 CPU는 별도 창에서:

```sh
mpstat -P ALL 5 > "$RUN/wp16_cuda-${SCHED}_omp-${WAIT}/mpstat.txt"
# 프로세스별 user+sys는 종료 시점에 한 번에:
for p in $(pgrep -f 'RASBERY --'); do cat /proc/$p/stat; done > .../proc_stat_pre.txt
```

프로세스별 user+sys 초는 각 자식을 `/usr/bin/time -v`로 감싸는 쪽이 정확하다
(디스패처가 자식을 직접 exec하므로 `--` 뒤를 `/usr/bin/time -v ./build/RASBERY`로
바꾸면 된다; 그 경우 `[RASBERY][MULTI_GPU][ENV]`는 그대로다).

**채울 표:**

| SCHED | OMP_WAIT | c/h | 평균 CPU % (mpstat) | user+sys s/proc | applied | B0 |
|---|---|---:|---:|---:|---|---|
| auto (무설정) | — | | | | | |
| auto | passive | | | | | |
| auto | active | | | | | |
| blocking | passive | | | | | |
| blocking | active | | | | | |
| yield | passive | | | | | |
| yield | active | | | | | |

**읽는 법.** 가설이 맞다면 `blocking`에서 **평균 CPU %가 79 %에서 크게 내려가고**
프로세스별 user 초가 줄면서 **c/h는 유지되거나 오른다**. CPU %만 내려가고 c/h도
같이 내려가면 스핀이 실제로 일을 하고 있었다는 뜻이고, 그때 가설은 기각이다.
`yield`는 그 둘 사이의 중간 arm으로, blocking의 깨우기 지연이 문제일 때 대안이다.

### 2.2 단일 덱: v6 ± blocking

```sh
python tools/run_single_gpu_batch.py --gpu 0 --batch-width 64 \
    --jobs one_deck.txt --workdir "$RUN/wp16_single_auto" -- ./build/RASBERY
python tools/run_single_gpu_batch.py --gpu 0 --batch-width 64 \
    --jobs one_deck.txt --workdir "$RUN/wp16_single_blocking" \
    --set RASBERY_CUDA_SYNC_MODE=blocking -- ./build/RASBERY
```

| arm | wall s | TOTAL DRIVER s | CPU % | user+sys s |
|---|---:|---:|---:|---:|
| v6 (무설정) | | | | |
| v6 + blocking | | | | |

**기대는 "같거나 약간 느림"이다.** 프로세스가 하나뿐이면 되찾을 코어가 없고,
동기화마다 붙는 깨우기 지연만 남는다. 그게 바로 측정해야 할 trade이고,
**단일에서의 손해 폭이 배치에서의 이득보다 작아야** 배치 기본값으로 승격할 수 있다.
단일 wall이 눈에 띄게 나빠지면 이 노브는 **배치 전용 arm**으로 남는다.

---

## 3. 결과

*(미측정 — §2를 돌린 뒤 채운다.)*

## 4. 판정

*(미측정.)*

승격 조건을 미리 적어 둔다. 이 셋을 모두 만족할 때만 `blocking`을
배치 프로파일의 기본으로 올린다:

1. 배치 c/h가 `auto` 대비 **하락하지 않는다**.
2. 배치 평균 CPU %가 유의미하게 내려간다(스핀이 일이 아니었다는 증거).
3. 모든 arm이 B0 — keff/핀출력이 `auto` arm과 비트 단위로 같다.
4. 단일 덱 wall의 열화가 배치 이득보다 작다. 아니면 배치 전용으로 남긴다.

---

## 5. 하지 않은 것

* **`spin`은 표에 넣지 않았다.** 구현은 되어 있고(`RASBERY_CUDA_SYNC_MODE=spin`)
  받아들여지지만, `auto`가 이 호스트 모양에서 이미 스핀이라는 것이 가설이므로
  `spin`은 기준선의 중복이다. 가설이 기각되면(§2.1을 읽는 법) 그때
  `auto`와 `spin`을 분리해 재는 것이 다음 실험이다.
* **`trajectory::kArmEnv`에 넣지 않았다.** 아직 arm이 아니라 측정 도구다.
  §4의 승격 조건을 통과한 뒤에 넣는다.
* **`cudaDeviceScheduleBlockingSync`를 스트림별로 나누지 않았다.** 이 플래그는
  프로세스(컨텍스트) 단위이고, 스트림별 대기 정책은 다른 메커니즘
  (`cudaEventBlockingSync` + 이벤트 대기)이다. 배치의 동기화가 전부
  `cudaStreamSynchronize`인 지금은 프로세스 단위로 충분하다.
