# WP13.1 — 사이트 원장: 이름 없는 7 GB와 6,100 sync에 이름을 붙인다 (2026-08-30, KO)

브랜치 `codex/exact-throughput-campaign`. 선행 문서는
`docs/WP13_HOST_DEVICE_TRAFFIC_20260830_KO.md`이고, 이 문서는 **그 문서가 닫지 못한 장부**를
닫기 위한 계측이다. 계측 외의 동작 변경은 없다.

---

## 0. 한 줄 요약

WP13은 소스만 읽어서 D2H 바이트를 0.8 % 안으로 닫았지만 **H2D는 절반이 남았다** —
11.7 GB / 54,918 copy 중 **7.0 GB / 24,952 copy**가 미귀속이고, 9,477 sync 중 **~6,100**이
이름이 없다(**653 µs × 6,100 = 4.0 s**, wall 11.2 s의 36 %). 남은 것들의 발화 빈도가
`canonicalElidesUpload/Download` 같은 **런타임 술어**로 정해지기 때문이고, nsys는 총계만 준다.

WP13.1은 `src/XferLedger.h`에 **래퍼**를 넣고 `src/` 전체의
`cudaMemcpy* / cuda*Synchronize`를 **169개 전부** 그 래퍼로 통과시킨다. 각 호출은
`file:function` + 버퍼 이름의 **사이트 태그**를 달고, `RASBERY_XFER_LEDGER=1`인 실행 끝에
`[RASBERY][XFER][LEDGER]`가 **사이트별 calls / bytes / sync ns**를 세 가지 정렬로 찍는다.
**한 번의 238 실행이 §4.2의 잔차와 §4.3의 sync 잔차를 동시에 확정한다.**

**등급 B0, 그리고 `trajectory::kArmEnv`에 넣지 않았다** — §2.

---

## 1. 무엇을 넣었나

### 1.1 래퍼 (`src/XferLedger.h`)

| 래퍼 | 감싸는 것 |
|---|---|
| `xfer::memcpyAsync(scope, leaf, dst, src, bytes, kind, stream)` | `cudaMemcpyAsync` |
| `xfer::memcpy(scope, leaf, dst, src, bytes, kind)` | `cudaMemcpy` (블로킹 — 경과 ns도 잰다) |
| `xfer::streamSync(scope, leaf, stream)` | `cudaStreamSynchronize` |
| `xfer::deviceSync(scope, leaf)` | `cudaDeviceSynchronize` |
| `xfer::eventSync(scope, leaf, ev)` | `cudaEventSynchronize` |
| `xfer::note(scope, leaf, dir, bytes)` | 래퍼를 쓸 수 **없는** 지점 (현재 0개) |

반환값은 감싼 CUDA 호출의 것 **그대로**다. 그래서 호출부의 `RASBERY_CUDA_TRY` /
`CUDA_CHECK`는 이전과 똑같은 것을 본다.

### 1.2 사이트 태그는 문자열 리터럴 두 개다

`scope` = `file:function`(필요하면 `(captured)`·`(hybrid)`·`(full)` 같은 arm 표기),
`leaf` = 버퍼 이름. 표는 **포인터로 키를 잡는다** — 리터럴은 정적 저장 기간을 가지므로
두 포인터의 해시가 O(1)이고, 200행 strcmp를 copy 경로에 놓지 않는다.
`tools/test_xfer_ledger_contract.py`가 "scope는 리터럴이고 자기 파일 이름으로 시작한다"를 강제한다.

### 1.3 계측 인구조사 — 169 사이트

| 파일 | 태그된 호출 |
|---|---:|
| `src/CudaXsReconBackend.cu` | **71** |
| `src/CudaOuterGraph.cu` | **50** |
| `src/CudaBICGBackend.cu` | **26** |
| `src/CudaCramBackend.cu` | 11 |
| `src/CudaPprBackend.cu` | 7 |
| `src/GpuPhysicsArenaCuda.cu` | 4 |
| **합** | **169** |

원본 grep은 174건이었고, 차이 5건은 **문자열 리터럴 안의 `cudaMemcpy(`**
(`fail("cudaMemcpy(nodal arena slot map)", rc)` 같은 에러 메시지)다. 계약 테스트의
스캐너는 주석과 문자열을 지우고 나서 본다 — 메시지 문구를 스캐너에 맞춰 고쳐 쓰게 만드는
검사는 검사가 아니다.

WP13 §3이 "지점"으로 셌던 것 중 **헬퍼 뒤에 숨어 있던 것들**이 이번에 갈라졌다:

- `DeviceBlock::upload` / `download` 하나가 **30개 호출처**를 대신 내고 있었다. 24.4 GB짜리
  flatxs micx/lmpx 다운로드(WP13 §3.1 `:3467-3471`)와 70.3 MB짜리 micx 업로드가 **같은 행**에
  들어가 있었다는 뜻이다. 이제 `flatxs micx mic` / `flatxs lmpx lmp` / `xsrecon micx mic` …
  으로 나뉜다.
- `BatchCore::pushOrSkip` / `push` / `push_pending` / `pushDeviceReadOnly` — CMFD의 outer당
  5종(WP13 §4.4 "횟수 기준 1위, 21,885 copy")이 여기서 각각의 행을 얻는다.
- `CudaCramBackend::h2d`, `CudaPprBackend`의 `h2d`/`d2h`/`syncStream`은 **이미 `name`을
  들고 있었다** — 그 `name`을 그대로 leaf로 쓴다. 열일곱 개 호출처가 인자 추가 없이 갈라졌다.
  (CRAM의 `"H2D mic"`만 predictor/corrector 두 곳에서 같은 리터럴이라 한 행으로 합쳐졌으므로
  `"H2D mic (predictor)"` / `"H2D mic (corrector)"`로 갈랐다. 바뀐 것은 실패 메시지 문구뿐이다.)

---

## 2. 왜 B0이고, 왜 `kArmEnv`에 없나

**구성상 B0.** 래퍼는 사이트가 원래 하던 CUDA 호출에 **같은 인자로, 같은 순서로, 같은
스트림에** 그대로 전달하고, 그 뒤에 카운터 산술만 한다. 데이터도, 순서도, 스트림도 건드리지
않으므로 궤적이 움직일 수 있는 경로가 없다. 계약 테스트 3번이 각 래퍼가 자기 이름의 CUDA
호출로 전달하는지를 검사하고, 음성 대조가 그 전달을 끊었을 때 잡히는지를 검사한다.

**꺼져 있을 때 분기 하나.** `RASBERY_XFER_LEDGER`가 없으면 사이트 표는 **읽지도 쓰지도
않는다**. 남는 비용은 (a) WP13이 이미 넣어 둔 프로세스 총계 relaxed atomic과 (b) 캐시된
`static bool` 하나에 대한 예측 가능한 분기다. **sync를 재는 시계 읽기는 그 분기 안쪽에만
있다** — 영수증이 자기가 재는 것을 느리게 만드는 원인이 되어서는 안 된다. 계약 테스트 4번이
이것을 고정한다.

**그래서 `trajectory::kArmEnv`에 넣지 않았다.** 궤적을 움직일 수 없는 knob을 case key에
넣으면 같은 실행 두 개를 서로 다른 캐시 항목으로 만든다 — `RASBERY_GPU_XFER_ELIDE`와
같은 이유다(WP13 §5).

### 2.1 총계의 의미가 바뀌었다 (좋은 쪽으로)

기존 `[RASBERY][XFER]`의 `d2h_calls/h2d_calls/syncs`는 **손으로 고른 부분집합**이었고
`covered` 필드가 그 목록이었다. 이제 래퍼가 유일한 계수 지점이므로 총계는 **단일덱 GPU
경로 전체**다. `covered`는 목록 대신 **파일 이름**을 말한다. 손으로 부르던
`xfer::countH2D/countD2H/countSync`는 전부 제거했고(이중 계수), 계약 테스트 7번이
재도입을 막는다.

---

## 3. 영수증이 볼 수 **없는** 것 — 먼저 말해 둔다

**CUDA 그래프에 기록된 copy는 replay마다가 아니라 capture 때 한 번 세어진다.**
해당 사이트는 scope에 `(captured)`를 달았다. 현재 그런 사이트는 **정확히 하나**다:

```
CudaBICGBackend.cu:enqueue_outer(captured):host_status   D2H  slots*64 B
```

`launch_outer`(그리고 `enqueue_sweeps`를 통한 sweep 그래프)가 `enqueue_outer`를 capture하므로,
이 행의 `calls`는 **capture 횟수**이고 replay는 보이지 않는다. WP13 §4.1은 이 지점에
**11,387 copy를 유도로** 귀속시켰다("이 값은 측정이 아니다"). 이번 영수증은 그 값을
**뺄셈으로 측정한다** — §5.3.

나머지 capture 창(`NodalArena::ensureGraph`, PPR의 while 그래프, nodal instance capture)은
커널만 담고 있고 그 안에 우리가 라우팅한 memcpy는 없다. 확인 방법은 소스이지 추정이 아니다:
capture 구간은 `enqueueKernels(lanes)` 하나이고, `memcpyAsyncOrFail`은 전부 그래프 launch의
바깥에서 불린다.

두 번째 한계: **D2D는 h2d/d2h 총계에 넣지 않는다.** CRAM의 BOS 스냅샷(21 MB × 4/dep)과 Xe
history 회전이 여기 해당하고, `dir:"d2d"` 행으로만 나온다. PCIe를 지나가지 않는 바이트를
PCIe 총계에 더하면 nsys와 어긋난다.

---

## 4. 영수증 읽는 법

`RASBERY_XFER_LEDGER=1`일 때 종료 직전에 네 줄이 나온다(단일·batch·evaluator 세 경로 전부).

```
[RASBERY][XFER][LEDGER] {"sites":N,"overflow":0,"h2d_calls":..,"h2d_bytes":..,
                         "d2h_calls":..,"d2h_bytes":..,"d2d_calls":..,"d2d_bytes":..,
                         "sync_calls":..,"sync_ns":..}
[RASBERY][XFER][LEDGER][BY_BYTES]   [{"site":"...","dir":"h2d","calls":N,"bytes":B,"ns":T}, ...]
[RASBERY][XFER][LEDGER][BY_CALLS]   [ ... ]
[RASBERY][XFER][LEDGER][BY_SYNC_NS] [ ... ]
```

- **`overflow`는 0이어야 한다.** 표는 512행 고정이고 넘치면 행을 버리는 대신 여기에 센다.
  0이 아니면 총계와 행의 합이 어긋나 있다는 뜻이므로 그 실행의 사이트 분해는 무효다.
- 세 정렬은 **같은 행 집합**이다. 바이트 병목(로드맵 1·3·7), 호출 횟수 병목(로드맵 4·5),
  그리고 **wall을 실제로 먹는 것**(로드맵 6, sync)을 각각 맨 위에 올리기 위한 것뿐이다.
- 한 줄이 아니라 네 줄인 이유: 200행을 한 줄에 담으면 nsys와 대조할 총계가 그 줄 한가운데에
  묻힌다.

---

## 5. 238 런북

로컬 계산 금지. 출력은 `E:`. IP는 어디에도 쓰지 않는다.

### 5.1 arm

`v4` = PROD(§1.1, `docs/PRICING_PROD_20260830_KO.md`) + `RASBERY_GPU_CRAM=1
RASBERY_GPU_CMFD_FUSE=15 RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_XE_TXN=1`.
**`v5` = `v4` + `RASBERY_GPU_XFER_ELIDE=1`** (WP13 §5의 순수 전송 소거). 이 문서에서 처음
이름을 붙인다.

```
# v5 + 원장
env -i RASBERY_PPR_MODE=master RASBERY_PC_MODE=decart RASBERY_GPU=1 \
  RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 \
  RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 \
  RASBERY_GPU_FLATXS=1 RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8 \
  RASBERY_GPU_WIEL_FOLD=chunked RASBERY_GPU_XE=1 \
  RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1 \
  RASBERY_OMP_THREADS=12 \
  RASBERY_GPU_CRAM=1 RASBERY_GPU_CMFD_FUSE=15 RASBERY_GPU_PPR=1 \
  RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_XE_TXN=1 \
  RASBERY_GPU_XFER_ELIDE=1 RASBERY_XFER_LEDGER=1 \
  <bin> <kngr cycle-1 deck>
```

**한 번이면 된다.** 이것은 A/B가 아니라 인구조사다. 타이밍을 인용할 실행이 아니므로
warm-up/hot 프로토콜도 필요 없다 — 다만 **wall은 인용하지 않는다**(sync마다 시계를 두 번
읽는다). 절감을 재려면 `RASBERY_XFER_LEDGER` **없이** v4/v5를 따로 돌린다.

원장을 v4(소거 OFF)에서도 한 번 받아 두면 `elided_calls`가 어느 사이트에서 나왔는지를
`by_calls` 두 장의 차이로 읽을 수 있다.

### 5.2 게이트 (전부 통과해야 B0 주장이 산다)

| 게이트 | 기준 |
|---|---|
| digest | **`1f36e75dc00ed2b4`** |
| outers | **`4377`** |
| h5diff | 0 |
| pin CSV | `cmp` 완전 동일 |
| 18지표 | Δ = 0 |
| `host_fallbacks` | 전 서브시스템 0 |
| `loop_arm` | `"device_graph"` |
| `txn_steps` | 1117 = `xe_device_steps` |
| `[RASBERY][XFER][LEDGER]` `overflow` | **0** |

**digest가 움직이면 계측이 궤적을 움직였다는 뜻이고, 그것은 §2의 "구성상 B0"가 틀렸다는
뜻이다.** 그 경우 이 커밋은 되돌린다.

### 5.3 nsys 대조 — 무엇이 맞아야 하고 무엇이 어긋나야 하나

기준값(WP13 §1, v4 단일 실행):

| 항목 | nsys |
|---|---:|
| H2D | **11.7 GB / 54,918 copy** |
| D2H | **28.6 GB / 55,183 copy** |
| `cudaStreamSynchronize` | **9,477 회 / 6.19 s** |

| 대조 | 기대 | 어긋나면 |
|---|---|---|
| `h2d_calls` (v4 원장) vs 54,918 | **일치해야 한다.** H2D 쪽에는 capture된 사이트가 없다 | 부족분은 우리가 못 찾은 H2D 사이트다. `by_calls`에서 outer(4,377)·sweep(18,627)의 배수를 찾는다 |
| `h2d_bytes` vs 11.7 GB | 일치 | 위와 같다 |
| `d2h_calls` vs 55,183 | **부족해야 한다.** 차이 = `enqueue_outer(captured):host_status`의 **replay 횟수** | 차이가 WP13 §4.1의 유도값 **11,387**과 가까우면 그 유도가 맞았다는 **측정**이다. 크게 벗어나면 `unroll = _ncmfd − iout` 모형이 틀린 것이다 |
| `d2h_bytes` vs 28.6 GB | 위와 같은 이유로 `host_status`(64 B × replay ≈ 0.7 MB)만큼 부족 — 즉 **사실상 일치** | 큰 차이는 미계측 D2H |
| `sync_calls` vs 9,477 | `cudaDeviceSynchronize` 2 사이트를 더한 값이므로 **9,477 + 소수** | — |
| `sync_ns` vs 6.19 s | 같은 자릿수. 시계는 호스트 wall이고 nsys는 API 시간이므로 정확히 같지는 않다 | — |

**그리고 이것이 이 커밋의 산출물이다:**

1. `by_bytes` 1위가 `CudaXsReconBackend.cu:DeviceBlock::download:flatxs micx mic` 계열
   합계 ≈ **24.4 GB**인지 (WP13 §4.4 바이트 1위의 검증).
2. `by_calls`에서 **§4.2의 7.0 GB / 24,952 copy가 어느 사이트로 가는지.** 가설은
   `CudaOuterGraph.cu`의 `runOneOuter:jnet bridge download` / `runOuterTail:jnet bridge upload`
   / `runOneOuter:flux upload`와 `CudaXsReconBackend.cu:solveNodal(full):jnet upload` —
   즉 canonical 바인딩이 잡히지 않은 outer들이다. **맞다면 이것은 새 기능이 아니라
   residency 버그이고, 로드맵 1번보다 먼저 볼 값어치가 있다.**
3. `by_sync_ns` 1위가 §4.3의 이름 없는 **~6,100 sync / 4.0 s**에 이름을 준다. 가장 유력한
   후보는 `CudaXsReconBackend.cu:solveNodal:final drain`과
   `CudaOuterGraph.cu:runOneOuter:pre-nodal drain`이다.

### 5.4 nsys 전/후

```
nsys profile --stats=true -o E:/<run>/xfer_ledger <위 커맨드에서 RASBERY_XFER_LEDGER 제외>
```

원장과 nsys를 **같은 실행에서** 뽑지 않는다 — 프로파일링 오버헤드(+60 %)가 sync ns를
오염시킨다. 대조는 `cuda_gpu_mem_time_sum`의 **호출 수**로 하고 절대 초는 인용하지 않는다.

---

## 6. 계약 테스트 (로컬에서 실행 가능, 컴파일 불필요)

```
python tools/test_xfer_ledger_contract.py
python tools/test_enum_alias_contract.py
python tools/test_dependent_template_contract.py
```

세 개 모두 이 커밋에서 통과했다. `test_xfer_ledger_contract.py`는 검사 8개와 **음성 대조
10개**를 갖는다 — 각 검사를 "그 검사가 지키는 성질을 깨뜨린 소스 사본"에 다시 걸어 실패하는지
확인한다. 실패할 수 없는 검사는 검사가 아니다.

| # | 검사 | 대조 |
|---|---|---|
| 1 | `src/` 어디에도 래퍼 밖 `cudaMemcpy*`/`cuda*Synchronize`가 없다 | PPR·CRAM에 raw 호출을 되돌려 넣는다 |
| 2 | 래퍼 헤더가 여섯 진입점을 선언한다 | `streamSync` 개명 |
| 3 | **각 래퍼가 자기 이름의 CUDA 호출로 전달한다 (B0)** | `memcpyAsync`가 copy를 삼키게 만든다 |
| 4 | 사이트 표가 `RASBERY_XFER_LEDGER`에 opt-in이다 | 무조건 기록 / 시계를 밖으로 끌어낸다 |
| 5 | 영수증이 총계 + 세 정렬을 낸다 | `BY_SYNC_NS` 제거 |
| 6 | `main.cpp`의 **세** 종료 경로가 전부 찍는다 | 하나를 주석 처리 |
| 7 | 어느 사이트도 총계를 손으로 더하지 않는다 (이중 계수) | `countH2D`를 하나 심는다 |
| 8 | 모든 래퍼 호출의 scope가 **리터럴**이고 자기 파일 이름으로 시작한다 | scope를 런타임 변수로 바꾼다 |

검사 1과 8의 스캐너는 **주석과 문자열 리터럴을 지우고** 본다. 검사 6은 텍스트가 아니라
코드를 센다 — 주석 처리된 호출은 찍히지 않는다.

`ALLOW` 목록은 **비어 있고, 그것이 옳은 상태다.** 래퍼를 쓸 수 없는 지점이 생기면
(경로 · 근거 · 그 줄의 식별 문자열)을 명시적으로 적어야 통과한다.

---

## 7. 로드맵에 남긴 것

WP13 §6의 순서는 그대로다. 이 커밋은 그 표의 **2번("§4.2의 H2D 잔차 확정 + §4.3의 sync
잔차")을 실행 가능하게 만든 것**이고, 1·3·4·5·6·7번은 하나도 건드리지 않았다.
다음 238 실행 **한 번**이면 2번이 닫힌다.
