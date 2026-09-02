# WP10.8 — persist 소크의 세 가지 판정을 되돌린다: RSS 기울기, `arena_rebuilds`, 사라진 18케이스

- 대상 브랜치: `codex/exact-throughput-campaign` (부모 `43f8174`)
- 증거: 238 GPU1, build `0054838` (WP19.2 / WP10.7 **이전**),
  `E:\rasbery_runs\2026-08-30\238\pricing_388e8f2.md` 블록 (38) Phase 2 arm A/B,
  그리고 `E:\rasbery_runs\2026-08-30\238\sigsegv_38\soak_on_arm_sigsegv_context.txt`
- 건드린 파일: `src/GpuDeviceBlockPool.h`, `src/EvaluatorServer.h`,
  `tools/soak_run.py`, `tools/run_multi_gpu_batch.py`, `tools/promotion_gate.py`,
  `tools/test_arena_persist_contract.py`, `tools/test_evaluator_mem_receipt_contract.py`,
  신규 `tools/test_device_block_pool_contract.py`,
  신규 `tools/test_restart_recovery_contract.py`
- **손대지 않은 파일**: `src/CudaXsReconBackend.*`, `src/FlatXsCtaKernel.cuh`,
  `src/FlatXsKernel.h`, `src/NodalKernel.h`, `src/Nodal.cpp`, `src/XsReconKernel.h`,
  `src/XSSet.cpp`, `src/GpuPhysicsArenaCuda.cu` — 이 파일들은 동시 진행 중인
  WP21-B2/C2 arm이 잡고 있다. XsRecon 쪽 호출부 개명은 커밋이 아니라
  `docs/patches/wp10_8_xsrecon.patch`로 낸다 (§3.4).

---

## 0. 요약

블록 (38) Phase 2 arm B(`RASBERY_ARENA_PERSIST=1`)의 FAIL 판정 세 줄은 이렇게
바뀐다.

| arm B가 말한 것 | 실제 | 고친 곳 |
|---|---|---|
| `RSS +115.97 MB/generation` (예산 8) | **SIGSEGV 재시작을 관통해서 fit한 기울기**. 같은 런의 second-half는 5.42 MB/gen으로 예산 안. 새 자식 프로세스의 cold ramp가 "post-warm-up" 창 안에 통째로 들어가 있었다 | `tools/soak_run.py:365` epoch 분절 |
| `arena_rebuilds`가 persist arm에서도 계속 오른다 → "persist가 rebuild를 막지 못한다" | 그 카운터는 **arena와 무관**했다. 케이스당 1회의 *per-instance device block 재배치*를 세고 있었고, 그래서 두 arm 모두 +17/gen이었다 | `src/GpuDeviceBlockPool.h:105/107/433`, `src/EvaluatorServer.h:1795` |
| `18/360 cases never reported` | 세대 1의 **18개 요청 전부**(16 candidate + poison + promote). 소크 루프가 죽은 자식을 재시작만 하고 재큐잉을 하지 않았다 | `tools/soak_run.py:1106` |

풀(pool) 자체는 RSS 증가의 범인이 **아니다**. 산술은 §1.2에 있고, 그 산술을
다음 런에서는 추정이 아니라 영수증으로 확인할 수 있게 `pool_bookkeeping_bytes`를
MEM에 넣었다. 별개로 WP10.6 free list는 **상한이 없었으므로**(정책이 없는 것도
정책이다) bytes/blocks/size-class depth 상한과 LRU 축출을 붙였다.

---

## 1. RSS +115.97 MB/generation — 무엇이 커졌는가

### 1.1 용의자: 풀의 host bookkeeping

WP10.6은 `XsReconBackend::Impl`의 케이스당 22개 device block을 exact-size free
list에 park한다. arm B에서 `device_pool_hits`는 세대 1의 39에서 세대 19의
6,695까지 올랐다 — 세대당 약 350건, width 16 + promote = 17 케이스이므로
**케이스당 약 21건**, 즉 22개 블록이 거의 전부 풀에서 나오고 있었다는 뜻이다.
용의자는 그 park 목록을 들고 있는 host 컨테이너다:

```
src/GpuDeviceBlockPool.h:145   std::unordered_map<const void*, Registration>        live;
src/GpuDeviceBlockPool.h:146   std::unordered_map<std::size_t, std::vector<Parked>> parked;
```

### 1.2 그 용의자를 저울에 올린다 (MEM 필드 산술)

세대당 park되는 블록 수는 17 케이스 × 22 블록 = **374**. 한 블록의 bookkeeping은

- `live` 항목: key `const void*` 8 B + `Registration` 16 B + 노드 오버헤드 ≈ 32 B → **56 B**
- `parked` 항목: `Parked{void*, uint64}` = **16 B**
- 사이즈 클래스 하나당 map 노드 ≈ 56 B, 클래스 수는 geometry로 결정되므로 22 근처에서 고정

즉 세대당 최대 374 × 72 B ≈ **26 KB**. `park`와 `take`가 균형을 이루면 이것조차
증가하지 않는다(`live`에서 지워지고 `parked`로 들어갈 뿐이다). 관측된 증가는
**115.97 MB/generation**이므로

```
26 KB / 115.97 MB  ≈  0.00022  =  0.022 %
```

**네 자릿수 차이다. 풀의 bookkeeping은 이 발견의 원인이 될 수 없다.**
`attribute_rss_growth`의 `ATTRIBUTION_SHARE = 0.05`(5 %) 기준으로는 "너무 작아서
용의자가 아님" 쪽에 들어간다.

같은 리포트가 두 번째 후보도 배제한다. "size key가 케이스마다 달라져서 park
목록이 무한히 자란다"면 **pooled VRAM이 함께 자라야** 한다. arm B의 raw VRAM
계열은 7774.0 → 5868.77 → 7782.0 → 5868.77 → 6039.10 이후 **남은 15세대 내내
6039.10으로 평탄**했다. 자라지 않았으므로 키는 어긋나지 않았고, park 목록은
working set 크기에서 수렴했다.

> 정직하게 남길 것: arm B의 **세대별 `[EVALUATOR][MEM]` 원본 줄은 E:로 회수되지
> 않았다.** 회수된 것은 리포트의 파생 기울기와 `sigsegv_38/` 컨텍스트 로그뿐이다.
> 따라서 위 산술은 리포트가 인용한 `device_pool_hits` / VRAM 계열 / 기울기로부터
> 재구성한 것이고, `malloc_taken/in_use/retained_mb`와 `cache_bytes.*`의 세대별
> 값은 **읽을 수 없었다**. 그것이 §3.1에서 영수증에 `pool_bookkeeping_bytes`,
> `pool_size_classes`, `pool_evictions`, `pool_evicted_mb`,
> `pool_park_refusals`를 추가한 이유다 — 다음 소크에서는 같은 질문이 추정이 아니라
> 뺄셈이 된다.

### 1.3 그러면 115.97은 무엇이었나 — 게이트 결함

arm B는 **세대 1에서 SIGSEGV로 evaluator가 죽고 attempt 2로 재시작했다.**
`sigsegv_38/soak_on_arm_sigsegv_context.txt`의 연속된 네 줄:

```
[RASBERY][CUDA][CAPTURE_RACE][RETRY] {"tag":"ppr.while","stage":"arm","cuda_error":901,...}
[RASBERY][CUDA][CAPTURE_RACE][RETRY] {"tag":"ppr.while","stage":"EndCapture(root)","cuda_error":901,...}
[RASBERY][MULTI_GPU][EVALUATOR][EXIT] {"returncode":-11}
[RASBERY][MULTI_GPU][EVALUATOR][START] {"command":[...],"attempt":2}
```

RSS는 `sample_rss_mb(session.pid)`로 세대 사이에 찍힌다. 재시작 뒤의 `session.pid`는
**다른 프로세스**이고, 그 프로세스는 라이브러리 로드·아레나 스탠드업·그래프 캐시
채우기를 처음부터 다시 한다. WP10.4/10.5가 도입한 `--warmup-generations`는 **런의
앞쪽** N세대를 버리는 규칙이므로, 런 중간에 시작된 warm-up은 하나도 버리지 못한다.
결과적으로 게이트는 "구 프로세스의 plateau, 그다음 신 프로세스의 cold→plateau
상승"을 하나의 직선으로 fit했다.

같은 리포트의 숫자가 이것을 확인해 준다.

| | arm A (재시작 없음) | arm B (세대 1에서 재시작) |
|---|---|---|
| post-warm-up 기울기 | 12.76 MB/gen | **115.97 MB/gen** |
| second-half 기울기 | 3.69 MB/gen | **5.42 MB/gen** |

second-half(세대 10–19)는 **전부 재시작 이후의 한 프로세스**이고 warm-up도 끝난
구간이다. 거기서 persist arm은 **5.42 MB/gen — 8 MB/gen 예산 안**이며, 재시작이
없던 arm A의 post-warm-up 12.76보다도 낮다. 즉 `RASBERY_ARENA_PERSIST=1`이 host
메모리를 arm A보다 더 새게 만든다는 증거는 이 런에 없다. 115.97은 **메모리에 대한
숫자가 아니라 프로세스 경계에 대한 숫자**였다.

---

## 2. `arena_rebuilds`가 실제로 센 것

### 2.1 증거

arm A: 세대마다 정확히 **+17** (17, 34, 51, …, 320). width 16 + promote 1 = 17
케이스이므로 **케이스당 정확히 1회**. arm B도 세대 1의 17에서 세대 19의 281까지
거의 같은 속도로 올랐다. **두 arm이 같다**는 것이 결정적이다 — persist 플래그가
바꾸는 것은 free/malloc이 드라이버에 도달하느냐뿐이고, 이 카운터는 그것과 무관하게
움직였다.

### 2.2 호출부

```
src/CudaXsReconBackend.cu:2748  Impl::ensure()          want_nxyz != nxyz 일 때
src/CudaXsReconBackend.cu:4074  node table regrow       n_nodes > nodes_cap 일 때
src/CudaXsReconBackend.cu:4094  stream array regrow     stream_len > stream_cap 일 때
src/GpuPhysicsArenaCuda.cu:197  두 번째 이후 stand-up   if (++standups > 1)
```

앞의 세 곳은 **per-instance** 영역이 shape 변화로 free + 재배치되는 사건이다.
`Impl`은 XSSet/Driver/deck마다 하나이므로, 케이스가 처음 실제 geometry로
`ensure()`를 부를 때 한 번 발생한다 — **케이스당 1회, 두 arm 모두**. 이것은 아레나
teardown이 아니다. persist는 그 사이트가 놓는 **블록**을 풀에 담을 뿐, 사이트가
발생하는 것을 막지 않으며 막겠다고 한 적도 없다.

### 2.3 고침 — 이름을 사실에 맞춘다

`src/GpuDeviceBlockPool.h`:

- `Stats::arena_rebuilds` → **`Stats::block_reshapes`** (105행). 세던 것은 그대로다.
- `noteBlockReshape()` (407행)가 새 철자. `noteArenaRebuild()` (418행)는 그것의
  **inline 별칭**으로 남긴다 — 두 개의 `.cu`가 그 이름으로 부르고 있고 이 헤더는
  그 호출부를 개명할 수 없기 때문이다. 동작은 완전히 동일하다.
- **`arena_standups` / `arena_teardowns`** (106–107행)는 등록 시 이미 들고 있던
  `poolable` 플래그에서 **파생**된다. 호출부의 협조가 전혀 필요 없다.
- `arenaRebuilds(Stats)` (433행) = `arena_teardowns`. 프로세스 수명 영역(nodal
  arena, flat-XS library, physics arena)은 한 번 잡고 종료 때만 놓으므로 **세대
  사이 영수증에서는 0이어야 하고**, 0이 아니면 그것이 바로 VRAM 톱니가 제기했던
  아레나 teardown이다.

`src/EvaluatorServer.h:1795`가 이제 `arena_rebuilds` 필드에 `arenaRebuilds(dev)`를
싣고, `block_reshapes`(1797행)와 `arena_standups`(1796행)를 나란히 낸다. 즉
**같은 이름이 이제 그 이름이 늘 약속하던 것을 답한다.**

---

## 3. 고친 것 전부

### 3.1 풀에 상한과 축출을 붙였다 (`src/GpuDeviceBlockPool.h`)

WP10.6은 스스로 "정책이 있는 캐시가 아니다"라고 썼다. 그것은 정확한 allocator이자
**상한 없는 컨테이너**다. 세대 0에서 park한 블록은 그 사이즈를 아무도 다시 묻지
않아도 프로세스 종료까지 붙들려 있고, 어떤 영수증도 그것을 말하지 못했다.

- 상한 세 개, 전부 환경변수: `RASBERY_ARENA_PERSIST_CAP_MB` (기본 8192, 0이면 끔),
  `RASBERY_ARENA_PERSIST_CAP_BLOCKS` (기본 4096),
  `RASBERY_ARENA_PERSIST_CLASS_DEPTH` (기본 64 = 최대 arena width의 4배).
- 축출은 **가장 오래 park된 것 우선, 사이즈 클래스를 가로질러**
  (`detail::Parked{block, seq}` 138행, `evictOneLocked` 256행). 정상 상태가 묻지
  않는 클래스가 나가고 working set은 남는다.
- 클래스 depth는 **넣기 전에** 검사한다. 이미 depth에 찬 클래스는 아무도 묻지 않는
  클래스이므로, 거기 넣고 나서 *다른* 클래스의 최고참을 축출하면 폭주하는 사이즈
  하나가 working set을 밀어내게 된다.
- 축출에는 device free가 필요한데 이 헤더는 순수 host다. CUDA 반쪽이
  `ensureBlockPoolReclaimer()`로 `cudaFree`를 등록한다. **reclaimer가 없으면 풀은
  상한을 넘기는 대신 park를 거부한다**(`park_refusals`) — reclaimer가 있어야만
  성립하는 상한은 상한이 아니다.
- `cudaFree`는 동기화 호출이므로 **락을 놓고** 부른다 (341–401행).
- exactness는 그대로다. `take()`는 여전히 `s.parked.find(bytes)`뿐이고
  `lower_bound`/`upper_bound`는 이 파일에 없다. 4095 요청이 4096 블록을 받는 일은
  없다.

### 3.2 영수증이 용의자를 저울에 올린다 (`src/EvaluatorServer.h:1782–1826`)

새 필드: `arena_standups`, `block_reshapes`, `device_blocks_pooled`,
`pool_cap_mb`, `pool_cap_blocks`, `pool_class_depth`, `pool_size_classes`,
`pool_evictions`, `pool_evicted_mb`, `pool_park_refusals`,
`pool_bookkeeping_bytes`, `pool_reclaimer`, 그리고 기존 `evictions{}` 안에
`blockpool` / `blockpool_refusals`.

`pool_size_classes`가 세대마다 오르면 그것이 "size key가 케이스마다 무언가를
싣고 있다"는 신호다. `pool_bookkeeping_bytes`가 있으면 §1.2의 산술을 다음 런에서
그대로 뺄셈으로 확인할 수 있다.

### 3.3 재시작을 관통하는 기울기를 금지했다 (`tools/soak_run.py`)

- `longest_epoch_segment()` (365행) — 한 프로세스 수명의 가장 긴 구간. 동률이면
  **나중** 구간(런 끝까지 살아남은 프로세스).
- `growth_slopes(series, warmup, epochs)`는 **세그먼트를 먼저 자르고 그다음 warm-up
  세대를 버린다.** 세대 2에서 선 자식에게는 자기 자신의 stand-up 계단이 있고,
  런의 0..N-1 세대를 버리는 것은 그 ramp를 하나도 버리지 못하기 때문이다.
- 새 리포트 필드: `segment_first_generation`, `segment_last_generation`,
  `segment_generations`, `crossed_a_restart`,
  `slope_across_restarts_mb_per_generation`(= 예전 게이트가 냈을 숫자. **게이트하지
  않고 나란히 출판한다** — 238 리포트와 비교하는 독자가 어느 숫자가 바뀌었는지
  볼 수 있어야 한다).
- 재시작을 건넜다면 게이트가 런의 일부만 봤다는 사실 자체를 problem으로 낸다.
  4세대를 fit하고 PASS한 게이트는 런이 산 증거보다 적은 증거로 통과한 게이트다.
- `attribute_rss_growth`는 이제 **게이트가 실제로 fit한 그 구간**의 MEM 영수증을
  가격한다(경계를 건너 컨테이너를 가격하지 않는다). `pool_bookkeeping_bytes`와
  `pool_size_classes`도 읽고, `block_reshapes`는 "케이스당 재배치이므로 오르는 것이
  정상"이라고 **명시**한다 — 238 리포트가 이 숫자를 잘못 읽은 자리다.

### 3.4 죽은 워커의 in-flight 케이스를 전부 다시 돌린다

**증거**: `sigsegv_38/soak_on_arm_sigsegv_context.txt`에서 attempt 2가 선 직후
harness가 보내는 첫 요청은 **`g0002c0000`**이다. 세대 1(`g0001*`)의 18개 요청은
다시 요청되지 않았다. `18 = 16 candidate + 1 poison + 1 promote` — **세대 하나 통째**.

원인은 두 드라이버의 비대칭이었다. `run_multi_gpu_batch._run_wave_chunk`는 WP8부터
죽은 자식의 미완 케이스를 **한 번** 재큐잉하고, 그 뒤에도 남은 것을 이름으로
보고한다. 같은 `EvaluatorSession` 클래스를 쓰는 `soak_run`의 세대 루프에는 그 경로가
없었다: `result.cases = len(outcome.cases)`를 기록하고, 자식을 재시작하고, 다음
세대로 갔다.

고침 (`tools/soak_run.py:1080–1140`):

- `case_names()` / `reported_names()` / `unreported()` — 요청과 영수증을 `key`와
  `output`(및 basename) 합집합으로 대조한다. `[EVALUATOR][CASE]`는 `key`를,
  Driver의 `[RASBERY][CASE]`는 `output`만 싣기 때문이다.
- 자식이 죽었고 미보고 요청이 있으면: 재시작 → **미보고분만** 다시 보낸다
  (`cases=missing`). 이미 영수증을 낸 케이스를 다시 돌리면 출력을 덮어쓰고 세대를
  중복 계수한다.
- wave id는 `(g+1)*1000 + attempt` — 재시작 뒤 같은 id는 낡은 영수증이 엉뚱한
  대기를 끝내게 한다(`_run_wave_chunk`가 이미 지키는 규칙).
- **한 번만.** 두 자식을 연달아 죽이는 세대는 독이 든 세대이고, 영원히 재시도하면
  한 후보가 캠페인을 멈춘다.
- 로그: `[RASBERY][SOAK][RESTART_RECOVERY]`(무엇을 다시 큐에 넣었는지),
  케이스마다 `[RASBERY][SOAK][CASE][RESTART_RECOVERED]`. cold 프로세스에서 다시
  돈 케이스는 wall time이 비교 가능하지 않으므로 보이지 않으면 안 된다.
- **세대별** `requested == reported` 단언. 불일치는 FAIL이고 **누락 id를 전부
  나열**한다. 런 합계만 보면 한 세대의 -4와 다른 세대의 +4가 상쇄된다.
- 리포트: `cases_accounted`, `cases_recovered`, `cases_missing`, 그리고 세대별
  `requested`/`epoch`/`recovered`/`missing`. markdown 표에 `req`·`epoch`·`recov`
  열이 붙는다.
- 복구에 성공한 세대는 **problem이 아니라 note**다. 잃은 것이 없고,
  `--expect-restarts`가 이미 "재시작 자체가 발견인가"를 말하는 플래그다. 여기서
  또 FAIL을 내면 그 플래그가 쓸모없어지고, 미래의 runner를 복구하지 않는 쪽으로
  민다.

`tools/run_multi_gpu_batch.py`에도 대칭으로:
`requeued_keys`, `WorkerResult.restart_recovered`,
`[RASBERY][MULTI_GPU][EVALUATOR][RESTART_RECOVERY]`,
`[RASBERY][MULTI_GPU][EVALUATOR][CASE][RESTART_RECOVERED]`.

### 3.5 승격 게이트 (`tools/promotion_gate.py`)

소크가 스스로 PASS라고 해도 다음 두 가지는 별도로 막는다.

- `cases_accounted is False` 또는 `cases_missing`이 비어 있지 않음 → **blocker**.
  세대 중간에 후보를 잃은 캠페인 위에서 올린 default는 "살아남은 후보"만 보고 올린
  default다.
- 어느 growth 블록이든 `crossed_a_restart is True` → **blocker**. 두 프로세스 수명에
  걸친 기울기는 새 자식의 re-warm을 잰 것이고, "읽을 수 없음"은 "통과"가 아니다.

### 3.6 XsRecon 호출부는 패치 파일로

`docs/patches/wp10_8_xsrecon.patch` — 세 줄
(`noteArenaRebuild()` → `noteBlockReshape()`, `CudaXsReconBackend.cu:2748/4074/4094`).
별칭이 남아 있으므로 **적용 전후 모두 트리가 컴파일되고 계약 테스트가 통과한다.**

---

## 4. 계약 테스트

| 테스트 | 결과 |
|---|---|
| `tools/test_device_block_pool_contract.py` (신규) | PASS (3 operator caps, 11 receipt fields, 14 compiled assertions × 2 arms, 8 negative controls) |
| `tools/test_restart_recovery_contract.py` (신규) | PASS (2 driven soaks — 복구 성공 1, 이름으로 보고된 손실 1 — 4 source contracts, 8 segmentation assertions, 8 negative controls) |
| `tools/test_arena_persist_contract.py` | PASS (9 receipt fields, 22 per-case blocks, 8 negative controls, 15 compiled ×2 arms) |
| `tools/test_evaluator_mem_receipt_contract.py` | PASS |
| `tools/test_evaluator_residency_contract.py` | PASS |
| `tools/test_evaluator_rolling_contract.py` | PASS |
| `tools/test_enum_alias_contract.py` | PASS |
| `tools/test_dependent_template_contract.py` | PASS |
| `tools/test_soak_run.py` (회귀) | PASS |
| `tools/test_soak_receipt_schema_contract.py` (회귀) | PASS |
| `tools/test_promotion_gate.py` (회귀) | PASS |
| `tools/test_multi_gpu_dispatch.py` (회귀) | PASS |

신규 두 개가 실제로 구동하는 것:

- **풀**: depth 상한이 reclaimer 없이도 유지되는지, 거부된 park가 `device_frees`로
  세어지는지, 상한을 붙인 뒤에도 4095/4097 요청이 4096 블록을 받지 않는지, 블록
  상한이 **사이즈 클래스를 가로질러 최고참**을 축출하는지(자기 클래스가 아니라),
  비워진 클래스가 `size_classes`에서 사라지는지, 200블록 bookkeeping이 1 MB 미만인지
  (= §1.2 산술 자체를 단언한다), `noteArenaRebuild()` 별칭이 `block_reshapes`로
  가는지, 프로세스 수명 영역의 stand-up/teardown이 `arena_standups`/`arena_rebuilds`로
  나뉘는지.
- **복구**: fake child를 `FAKE_RASBERY_POISON` + `FAKE_RASBERY_POISON_MARKER`로
  세대 중간에 한 번 죽여서 **전 케이스가 보고되고 recovered 표시가 붙는지**,
  이미 보고된 케이스를 다시 돌리지 않는지(`cases > requested` 금지), marker 없이
  재시도도 죽는 형태에서 **FAIL하고 누락 id를 세대별로 이름으로 내는지**,
  그리고 재시작 없는 런에서는 분절이 기존 숫자를 **바꾸지 않는지**.

---

## 5. 238 GPU1 런북 (20세대, 두 arm)

빌드는 이 커밋으로 새로 세운다 (WP10.7/WP19.2 포함, 0054838 아님).

```bash
# 238, GPU1이 idle인지 먼저 확인하고 시작한다.
BUILD=$HOME/gpu_dispatch_test_wp10_8/build_wp10_8
OUT=$HOME/gates/wp10_8

for ARM in off on; do
  W=$OUT/soak_$ARM
  mkdir -p "$W"
  ENVFLAGS=""
  [ "$ARM" = on ] && ENVFLAGS="RASBERY_ARENA_PERSIST=1"
  env CUDA_VISIBLE_DEVICES=1 RASBERY_GPU_FULL=1 $ENVFLAGS \
    python3.11 tools/soak_run.py \
      --deck $HOME/kngr_238/kngr_238.json \
      --workdir "$W" \
      --binary "$BUILD/RASBERY" \
      --generations 20 --width 16 \
      --gpu 1 \
      --rss-leak-mb 8 --vram-leak-mb 8 \
      --warmup-generations 2 \
      --expect-restarts 0 \
      --report "$W/soak_$ARM.json" \
    2>&1 | tee "$W/soak_$ARM.console"
done
```

### 통과 기준 — 전부 리포트에서 직접 읽는다

| 항목 | 필드 | 기준 |
|---|---|---|
| RSS 기울기 | `growth.rss.slope_mb_per_generation` | ≤ **8.0** MB/gen, 두 arm 모두 |
| 기울기가 온전한 런을 봤는가 | `growth.rss.crossed_a_restart` | **false**. true면 판정 자체가 무효이고 재시작 원인을 먼저 닫는다 |
| | `growth.rss.segment_generations` | **20** (= `--generations`) |
| 아레나 teardown | 세대별 `mem.arena_rebuilds` | **arm B에서 전 세대 0**. arm A는 케이스마다 free/alloc하므로 이 필드가 아니라 `device_frees`가 움직인다 |
| 케이스당 재배치 | 세대별 `mem.block_reshapes` | **+17/generation 부근이 정상**이며 실패 조건이 아니다 (§2). arm A와 arm B가 크게 다르면 그것이 새 발견이다 |
| 풀이 유계인가 | `mem.pool_size_classes` | 세대 2 이후 **평탄**. 계속 오르면 size key가 케이스별 값을 싣고 있다 |
| | `mem.pool_evictions`, `mem.pool_park_refusals` | **0**이 정상. 0이 아니면 working set이 상한을 넘은 것이므로 `RASBERY_ARENA_PERSIST_CAP_MB`를 올려 재측정한다 |
| | `mem.vram_pooled_mb` | 평탄 |
| | `mem.pool_bookkeeping_bytes` | ≪ RSS 증가분. 이 두 수의 비가 §1.2 산술이다 |
| 케이스 회계 | `cases_accounted` | **true** (`cases_reported == cases_requested`) |
| | `cases_missing` | **[]** |
| | `cases_recovered` | 0이 이상적. 0이 아니면 note로 남고, 그 세대의 c/h는 비교에서 뺀다 |
| GPU_FULL 계약 | `gpu_full_contract.contract_pass` | **true**, `flatxs_fallbacks`/`outer_fallbacks` 모두 0 (WP10.7이 닫은 자리) |
| 정확성 | 두 arm의 `--result full` 단일 deck h5diff | **0 differences**, digest 동일 (블록 38에서 `1f36e75dc00ed2b4`로 이미 PASS) |

### 판정 순서

1. `cases_accounted`가 false이면 **다른 숫자를 읽기 전에 멈춘다.** 후보를 잃은 런의
   throughput과 기울기는 누가 선택했는지 모르는 표본 위의 숫자다.
2. `crossed_a_restart`가 true이면 메모리 판정을 내리지 않는다. 재시작 원인
   (블록 38에서 3회 관측된 WP19 계열 capture-race SIGSEGV)을 먼저 닫는다. 이 빌드는
   `src/CrashReport.h`(WP19.2)를 포함하므로, 다시 나면 `[RASBERY][CRASH]` 한 줄이
   case/lane/slot/phase를 이름으로 말해 준다 — 0054838에는 없던 증거다.
3. 그 두 개가 깨끗할 때만 RSS/VRAM 기울기를 판정한다.
4. `tools/promotion_gate.py`는 1과 2를 blocker로 다시 확인한다 (§3.5).
