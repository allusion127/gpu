# WP10.6 — VRAM sawtooth, 그리고 소크가 믿어도 되는 VRAM 숫자

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | 238 GPU1 20세대 소크(`bd7a0d3`)가 보고한 VRAM 298 MB ↔ 47,000 MB sawtooth, 상관된 33 % 처리량 저하, RSS +24.30 MB/gen |
| 증거 | `E:\rasbery_runs\2026-08-30\238\pricing_388e8f2.md` 블록 (34b) |
| 판정 | **sawtooth는 이 프로세스의 사실이 아니다.** 소크가 **다른 보드**를 샘플링했다(§2). 46 GB와 33 % 저하는 같은 원인(§3)이며, arena는 세대마다 재건되지 **않는다**(§4) |
| 플래그 | `RASBERY_ARENA_PERSIST=1`(기본 `0`) — per-case device block free list |
| receipt | `[RASBERY][EVALUATOR][MEM]` — `vram_mb`, `vram_delta_mb`, `vram_live_mb`, `vram_pooled_mb`, `vram_high_water_mb`, `device_allocs`, `device_frees`, `device_pool_hits`, `device_pool_parks`, `device_blocks_live`, `arena_rebuilds`, `arena_persist`, `malloc_taken_mb`, `malloc_in_use_mb`, `malloc_retained_mb`, `malloc_readable` |
| 계약 테스트 | `tools/test_arena_persist_contract.py` (negative control 8종, 컴파일 하네스 10항목 × 2 arm) |
| 소스 | `src/GpuDeviceBlockPool.h`(신규), `src/CudaXsReconBackend.cu`, `src/GpuPhysicsArenaCuda.cu`, `src/EvaluatorServer.h`, `tools/soak_run.py` |
| 기준 덱 | KNGR(kngr_238), `nxyz = 8,451`, `NG = 2`, `NISO = 39`, `NXS = 11`, `N_ACTIVE = 9` |

> 이 문서가 철회하는 것은 소크의 **VRAM 소견**이지 소크가 아니다. 블록 (34b)의 exactness
> 결과(zero-receipt 16종 전부 0, `live_cases`=0, poison 20/20 설명됨, restart 0)는 그대로
> 유효하다. 무너지는 것은 **보드 단위 측정치에서 계산한 기울기**뿐이다.

---

## 1. 세대 하나의 수명 주기 (텍스트 다이어그램)

`tools/soak_run.py`의 한 세대와, 그것이 `EvaluatorServer.h`에서 무엇을 세우고 무엇을 부수는지.

```
soak_run.py                          RASBERY --evaluator --batch-mode 16
-----------                          ----------------------------------
프로세스 시작(1회) ................> CUDA context 스탠드업            [1회/프로세스]
                                     XsLibrary 파싱(AcquireXsLibrary) [1회, 캐시]
                                     GpuPhysicsArena::reserve()       [1회/프로세스]
                                       └ cudaMallocFromPoolAsync,
                                         geometry 1.390 MiB
                                       + library  5.851 MiB
                                       + slot    224.378 MiB x slots
                                     NodalArena                       [프로세스 싱글턴]
                                     g_flatxs_libs (content hash)     [프로세스 싱글턴]

for g in 0..N-1:
  build_generation(g)                 wave(g+1, 18 cases)
  ── 웨이브 시작 ──────────────>       lane 0..15 병렬:
                                        ┌ CaseContext {Geometry, Scheduler,
                                        │              XSSet, IO}      << 케이스당 >>
                                        │   XSSet::EnsureBackend()
                                        │     └ XsReconBackend::Impl   << 케이스당 >>
                                        │        ensure()  : cudaMalloc x N   ← §3
                                        │        xeEnsure(): cudaMalloc x 8
                                        │        solveFlatXs(): dev_ref, dev_pernode,
                                        │                       dev_nodes/off/cnt,
                                        │                       ndev_dbl/ndev_int
                                        │   arena slot 취득 (refill::tenancy)
                                        │   ... statepoints / outers ...
                                        │   ~Impl(): cudaFree x N            ← §3
                                        └ arena slot 반납, Driver 파괴
  ── 웨이브 종료 ──────────────       iowriter::flushLines()
  sample_vram(...)                     [WAVE] receipt
  sample_rss_mb(child pid)             [MEM]  receipt   ← reportMemory()
  (arena/context/library: 그대로 유지)

프로세스 종료 .......................> rasberyReleaseBatchArena()  [arena_releases=1]
```

**세대 경계에서 파괴되는 것은 아무것도 없다.** arena·CUDA context·XS 라이브러리·cohort 캐시는
모두 프로세스 수명이고, `reportProcess()`의 `arena_releases`(종료 시 1회 stamp)와
`arena_standups`가 이미 그 증인이다. 세대마다 반복되는 것은 **케이스당 `CaseContext`**,
즉 `XSSet` → `XsReconBackend::Impl`의 device block 12종뿐이다.

---

## 2. 298 MB 골짜기의 정체 — 소크가 다른 보드를 봤다

블록 (34b)의 실행 명령:

```
CUDA_VISIBLE_DEVICES=1 python3.11 tools/soak_run.py ... --generations 20 --width 16
```

`--gpu`는 **주어지지 않았다**. `soak_run.py`의 두 줄이 서로 다른 보드를 가리켰다:

```python
env.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)   # 이미 "1" → 자식은 GPU1
...
result.vram_mb = sample_vram_mb(args.gpu)          # args.gpu == "0" → nvidia-smi -i 0
```

`nvidia-smi -i`는 **보드 인덱스**이고 `CUDA_VISIBLE_DEVICES`를 무시한다. 그래서 자식은 GPU1에서
돌고, 소크는 20세대 내내 **GPU0**을 쟀다. 그리고 같은 시각 GPU0에서는 블록 35(8×M16+MPS 3회 +
8×M8 1회)와 블록 36(8×M16 재현 2회 + 단발 3회)이 돌고 있었다 — 블록 35의 모든 pre-check가
"GPU1 87–91 % 사용 중"을 기록했듯, 반대 방향의 오염도 내내 존재했다.

| 관측치 | 실제 정체 |
|---|---|
| ~298–299 MB (정상 상태) | **GPU0의 MPS 서버 컨텍스트**, 클라이언트 작업 없음. 유휴 GPU1의 1 MiB와 다른 이유가 이것이다 |
| 31,000–47,000 MB (스파이크) | **GPU0의 8프로세스 × M16 배치** (§3의 식) |
| 세대 0, 11–16, 19에만 스파이크 | 그 세대들의 벽시계가 블록 35/36의 배치 런과 겹친 구간 |
| 같은 세대의 33 % 처리량 저하 | 같은 배치의 **호스트측 경합**(CPU·PCIe·드라이버). 조정자가 이미 "GPU1이 바쁠 때 GPU0 타이밍을 믿지 말 것"으로 한 방향은 명문화했고, 양방향임을 놓쳤다 |

즉 **하나의 원인(옆 보드의 8×M16 배치)이 sawtooth와 처리량 저하를 동시에 만든다.** 블록 (34b)이
"하나의 사건이 둘을 함께 일으킨다"고 추정한 것은 옳았고, 그 사건이 이 프로세스 안에 있다고 본
것만 틀렸다.

---

## 3. 46 GB의 귀속 — 식

`XsReconBackend::Impl::ensure()`의 `dev_block` 레이아웃(노드당 double):

```
mic      = NXS x NISO x NG      = 11 x 39 x 2 = 858
mic_ssm  = NISO x NG^2          = 39 x 4      = 156
lmp      = NXS x NG             = 11 x 2      =  22
lmp_ssm  = NG^2                               =   4
iden     = NISO                               =  39
xs       = NXS x NG                           =  22
xs_ssm   = NG^2                               =   4
phif     = NG                                 =   2
                                        합계  = 1,107 double = 8,856 B/node
```

`solveFlatXs()`의 참조 트윈 `dev_ref`(노드당 double):

```
9 x mic(78) + msm(156) + 9 x lmp(2) + ssm(4) = 880 double = 7,040 B/node
```

KNGR `nxyz = 8,451`:

```
dev_block = 8,451 x 8,856 B = 74.84 MB
dev_ref   = 8,451 x 7,040 B = 59.49 MB
                     소계   = 134.3 MB   (+ ndev_dbl/ndev_int, xe_hist, dev_pernode,
                                            dev_nodes/off/cnt, dev_sdid/sx/sscale)
```

레이아웃 계산기의 슬롯 예산과 일치한다 — `tools/test_gpu_physics_arena_contract.py`가 인쇄하는
`per_slot = 224.378 MiB`. 따라서

```
프로세스 1개(width 16) = 16 x ~225 MiB  +  geometry 1.390 MiB
                                        +  library  5.851 MiB
                                        +  NodalArena/flatxs 싱글턴
                       ~= 4 ~ 6 GB

8 프로세스 x M16 (MPS) = 8 x (4 ~ 6 GB) = 32 ~ 48 GB
```

**관측된 31,000–47,000 MB가 이 구간이다.** 46 GB는 "무언가가 세대 중간에 한 자릿수 더 많이
할당했다"가 아니라 **8개의 이웃 프로세스가 각자 정상적으로 할당한 합**이다. width 16 한
프로세스가 46 GB에 도달하는 경로는 존재하지 않는다: 그러려면 케이스 ~230개가 동시에 살아
있어야 하고, `live_cases`는 매 세대 0이었다.

---

## 4. arena는 세대마다 재건되지 않는다 — 그리고 이제 그것이 receipt다

`GpuPhysicsArena.h:39`가 "`rasberyReleaseBatchArena()` — 웨이브마다: **의도적으로 하지 않는다**"라고
쓰여 있고, `reserve()`는 두 번째 호출을 거절한다. 재건 가설은 설계상 성립하지 않는다. 다만
(34b)는 그것을 **보드 메모리 그래프로 추론**할 수밖에 없었다. WP10.6은 그 추론을 숫자로 바꾼다:

- `arena_rebuilds` — **살아 있는** device 영역이 shape 변경으로 해제되고 다시 레이아웃된 횟수.
  최초 스탠드업은 재건이 아니다. 계측 지점 3곳:
  `CudaXsReconBackend.cu` `ensure()` regrow(`dev_block != nullptr`), `xsrecon.nodes.regrow`,
  `xsrecon.stream.regrow`, 그리고 `GpuPhysicsArenaCuda.cu`의 2회차 이상 `reserve()`.
- `vram_mb` / `vram_delta_mb` — **이 프로세스가** 잡고 있는 device 바이트(사용 중 + 반납 대기).
  보드가 아니라 프로세스의 회계이므로 이웃이 움직일 수 없다.

정상 상태 기대치: `arena_rebuilds == 0`, `vram_delta_mb ≈ 0`.

---

## 5. 고친 것

### 5.1 계측 (무조건, 게이트 없음)

`tools/soak_run.py`:

- `sampled_gpu(env, requested)` — 자식의 `CUDA_VISIBLE_DEVICES`를 보드 인덱스로 해석한다.
  `--gpu`는 그것이 비었을 때만 쓴다. **(34b)의 결함이 여기서 닫힌다.**
- `sample_vram(gpu, pid)` — `--query-compute-apps=pid,used_gpu_memory`로 **자식 pid의 행**을
  먼저 찾는다. 못 찾으면 보드 총량을 쓰되 `scope="board"`로 **표시**하고, 같은 보드의 다른
  compute app pid를 함께 센다.
- 프로세스 자신의 `[MEM] vram_mb`가 있으면 그것이 **1순위**다(`scope="receipt"`). RSS는 반대로
  바깥 측정이 1순위인데, 이유가 대칭이다: RSS의 바깥 측정은 대상과 같은 방향으로 틀릴 수 없고,
  VRAM의 바깥 측정은 **바로 그것이 틀렸던 쪽**이다.
- 리포트에 `gpu_sampled`, `gpu_requested`, `vram_scopes`, 세대별 `vram_scope` /
  `vram_foreign_procs` / `vram_board_mb`가 실린다. 마크다운 표에 `scope` 열이 추가된다.
- **오염된 표본은 유죄판결하지 않는다**: `scope=="board"`이고 다른 tenant가 있으면 VRAM 기울기
  게이트를 건너뛰고(`growth.vram.gated=false`) 그 사실을 `problems`에 이름으로 적는다.

`src/EvaluatorServer.h` — `[RASBERY][EVALUATOR][MEM]`에 §메타데이터의 16개 필드 추가.
`malloc_retained_mb`는 glibc `mallinfo2()`의 `arena + hblkhd - uordblks`이며, (34b)가
"allocator를 다음에 보라"로 넘긴 질문에 **receipt가 직접 답하게** 한다. `mallinfo`가 아니라
`mallinfo2`인 이유는 전자의 필드가 `int`이고 4.5 GB를 조용히 감싸기 때문이다 — 감긴 카운터는
없는 카운터보다 나쁘다, 믿을 만해 보이므로.

### 5.2 per-case device block pooling (`RASBERY_ARENA_PERSIST=1`, 기본 off)

`src/GpuDeviceBlockPool.h`(신규, 순수 host + `__CUDACC__` 하의 얇은 래퍼 3개):

- `deviceBlockAlloc/Free` — **정확한 바이트 수**로 키가 걸린 free list. 분할·병합·best fit·반올림
  없음. `take(bytes)`는 정확히 `bytes`짜리를 돌려주거나 `nullptr`을 돌려준다. 74.8 MB 요청에
  74.9 MB 블록을 줄 수 있는 풀은 커널이 읽어도 되는 범위를 바꾸는 풀이다.
- `deviceBlockAllocOnce` — 프로세스 싱글턴(NodalArena, `g_flatxs_libs`, `GpuPhysicsArena` 블록)용.
  `vram_mb`가 전체 footprint가 되도록 **세지만** free list에는 절대 넣지 않는다.
- 풀에 등록되지 않은 포인터는 `give()`가 `false`를 돌려주어 **그대로 `cudaFree`로 간다**. 이
  헤더가 모르는 호출 지점은 이 헤더 때문에 깨지지 않는다.
- 풀 히트는 CUDA API를 **한 번도** 부르지 않으므로 `AllocWindow`를 열지 않는다. 드라이버에
  실제로 가는 경로에서만 연다(capture arbiter 계약 유지, `tools/test_capture_arbiter_contract.py`
  통과).

적용 지점: `CudaXsReconBackend.cu`의 `Impl` 소유 블록 22종(`dev_block`, `dev_fuel`, `dev_scalars`,
`dev_dep`, `dev_ref`, `dev_pernode`, `dev_nodes/off/cnt`, `dev_sdid/sx/sscale`,
`ndev_dbl/ndev_int`, `xe_*` 8종). 케이스당 ~24회의 **동기화하는** device API 호출이 정상 상태에서
0회가 된다 — width 16, 18 케이스/세대, 10k 세대면 ~4.3 M 회의 device-wide barrier다.

**B0 근거**: 할당 수명은 결과에서 관측 가능하지 않다. 풀 블록과 새 `cudaMalloc` 블록은 둘 다
초기화되지 않은 device 메모리이며(`cudaMalloc`도 0으로 채운다고 약속하지 않는다), 둘을 구분할 수
있는 코드는 자기가 쓰지 않은 메모리를 읽는 코드다 — 풀이 만드는 결함이 아니라 **드러내는** 결함이다.
그럼에도 기본을 `0`으로 둔 이유는 그것을 **주장이 아니라 238의 A/B로 증명**하기 위해서다(§6.3).

---

## 6. 238 GPU1 런북

### 6.1 사전 조건 — 이번에는 이것이 결과의 일부다

```bash
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv -i 0
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv -i 1
```

**두 보드 모두 비어 있어야 한다.** 소크는 이제 오염을 보고하지만, 오염된 런은 여전히 처리량
숫자를 쓸 수 없다(블록 35의 교훈, 양방향).

### 6.2 arm A — 기준 (`RASBERY_ARENA_PERSIST` 미설정)

```bash
CUDA_VISIBLE_DEVICES=1 python3.11 tools/soak_run.py \
  --deck kngr_238.json --workdir ~/gates/wp10_6/soak_off \
  --binary <build>/RASBERY --generations 20 --width 16 --gpu 1 \
  --report ~/gates/wp10_6/soak_off.json
```

`--gpu 1`을 **명시**한다. 이제 없어도 `CUDA_VISIBLE_DEVICES`가 이기지만, 명시가 리포트의
`gpu_requested`/`gpu_sampled`를 한 줄로 대조 가능하게 만든다.

### 6.3 arm B — 풀 켜기

```bash
CUDA_VISIBLE_DEVICES=1 RASBERY_ARENA_PERSIST=1 python3.11 tools/soak_run.py \
  --deck kngr_238.json --workdir ~/gates/wp10_6/soak_on \
  --binary <build>/RASBERY --generations 20 --width 16 --gpu 1 \
  --report ~/gates/wp10_6/soak_on.json
```

### 6.4 합격 기준

| 항목 | 기대 | 근거 |
|---|---|---|
| `vram_scopes` | `["receipt"]` (또는 `["process"]`) — **`"board"` 없음** | §5.1 |
| `[MEM] vram_mb` | 세대 간 **평평**, arena 크기(≈ 16 × 225 MiB + 7.2 MiB ≈ 3.6 GB) 근처. sawtooth 없음 | §3 |
| `vram_delta_mb` | 워밍업 이후 ≈ 0 | §4 |
| `arena_rebuilds` | **0** (모든 세대) | §4 |
| `device_allocs`/`device_frees` | arm A: 세대마다 ~430씩 증가 / arm B: 워밍업 후 **증가 멈춤** | §5.2 |
| `device_pool_hits` | arm A: 0 / arm B: 세대마다 ~430 | §5.2 |
| RSS 워밍업 후 기울기 | **≤ 8 MB/gen** | 소크 예산 |
| `malloc_retained_mb` | RSS 증가분의 대부분을 설명해야 함 → "leak이 아니라 high-water" | §5.1 |
| `worst_drift` | ≤ 3 % — 33 % 저하 소멸 | §2 (옆 보드를 비웠으므로) |
| zero-receipt 16종 | 전부 0, `live_cases` 0, poison 20/20 설명 | (34b)에서 이미 통과, 회귀 없어야 함 |

### 6.5 exactness 게이트 (arm B 채택 조건)

같은 덱 1개를 `RASBERY_ARENA_PERSIST` 없이 / `=1`로 각각 단발 실행하고 `--result full`의 h5를
`h5diff`로 비교한다. **바이트 동일**이어야 arm B가 기본으로 승격된다. 다르면 그것은 풀의 결함이
아니라 초기화되지 않은 device 메모리를 읽는 곳이 있다는 뜻이며, 그 지점을 찾는 것이 다음 작업이
된다 — 어느 쪽이든 풀은 켜지 않고 이 문서가 이유를 기록한다.

---

## 7. 철회 및 정정

- 블록 (34b)의 **"VRAM 2,160.84 MB/gen 증가 / sawtooth 298 MB ↔ 47,000 MB"** 소견은
  **철회한다**. 측정 대상이 이 프로세스가 아니었다(§2).
- 블록 (34b)의 **"worst_drift 33.19 %"** 는 사실이지만 **원인 귀속이 없다**. 원인은 옆 보드의
  호스트측 경합이며(§2), 조정자의 "GPU1이 바쁠 때 GPU0 타이밍 금지" 규칙을 **양방향으로**
  확장한다: *다른 보드가 바쁠 때 어느 쪽 처리량도 믿지 않는다.*
- 블록 (34b)의 **RSS +24.30 MB/gen(워밍업 후) / +4.15(후반)** 는 그대로 유효한 열린 항목이다.
  후반 기울기가 예산 안이라는 것은 high-water 수렴의 형태이고, `malloc_retained_mb`가 이제
  그것을 확인하거나 반증한다(§5.1). 컨테이너 카운터는 여전히 움직이지 않을 것으로 예상한다 —
  움직이는 것은 케이스당 host 배열을 16 스레드가 malloc/free 하며 만드는 per-thread arena의
  high-water이지, 자라는 캐시가 아니기 때문이다.
