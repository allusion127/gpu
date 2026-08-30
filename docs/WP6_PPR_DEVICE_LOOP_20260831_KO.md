# WP6 — PPR device loop / canonical input / batch arena

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `RASBERY_GPU_PPR` arm의 Picard 종료 판정, 입력 업로드, 핀 재구성, per-slot 할당 |
| 상위 계획 | `docs/GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` §WP6 단계 B·C·D·E |
| 선행 | 단계 A 완료 (238): `PPR_GPU` ON×2 결정론, Gate A **0.0000 / 35 statepoint 전부**, receipt `host_fallbacks 0` · `iterations 816` · `host_iterations 0` |
| 게이트 등급 | 단계 B **B0 (현 device arm 대비)** — fold의 association이 host fold와 동일 / 단계 C **B0 조건부**, 조건은 borrow의 건전성이고 그것은 `verify` 모드가 측정한다 / 단계 D **N1**, 원천은 `exp` 하나이고 나머지 순서는 전부 host의 것이다 / 전체는 host arm 대비 여전히 **N1**(`c502856`이 건너간 선) |
| 플래그 | `RASBERY_GPU_PPR=1`(기본 0) 안에서 `RASBERY_GPU_PPR_DEVICE_LOOP`(기본 **1**), `RASBERY_GPU_PPR_GRAPH`(기본 0), `RASBERY_GPU_PPR_CANONICAL`(기본 `off`; `1`=borrow, `verify`=borrow+대조), `RASBERY_GPU_PPR_RECON`(기본 0) |
| receipt | `[RASBERY][PPR_GPU]` **schema_version 2** — `loop_arm`, `host_syncs`, `host_syncs_per_statepoint`, `graph_launches`, `graph_builds`, `graph_refusal`, `canonical_mode`, `canonical_statepoints`, `canonical_mismatch`, `h2d_bytes`, `h2d_bytes_elided`, `d2h_bytes`, `recon_statepoints`, `pin_materializations`, `recon_repairs`, `recon_refusal`, `allocations`, `reallocations` |
| 계약 테스트 | `tools/test_ppr_gpu_contract.py` — 27 property, **negative control 38종** |
| 소스 | `src/CudaPprBackend.{h,cu}`, `src/CudaPprBackendStub.cpp`, `src/PprReconstructionKernel.cuh`(신규), `src/PPR.{h,cpp}`, `src/Driver.h`(PPR 블록·receipt) |
| 기준 덱 | KNGR, `nxyz = 8,451`, `nsurf = 26,692`, `NG = 2`, 집합체 88(쿼터), 핀 16×16 |
| 로컬에서 한 것 | 순수 python 계약(27 property + 음성대조 38), MSVC `/Zs`로 `src/PPR.cpp`·`src/CudaPprBackendStub.cpp`·`Driver.h`를 include하는 TU, 그리고 `CudaPprBackend.cu`(+`PprReconstructionKernel.cuh`)를 `tools/check_cuda_syntax.py`의 재작성 규칙 + graph/conditional shim으로 **`CUDART_VERSION` 12060·13000 양쪽** 문법 검사. **nvcc는 없다 — 238 컴파일이 첫 관문이다.** |

> 이 문서의 sync 수와 바이트 수는 **모델이 아니라 계약**이다. `tools/test_ppr_gpu_contract.py`가
> `CudaPprBackend.cu`의 `cudaStreamSynchronize` 호출 지점 수를 직접 세고(정확히 1개, 그것도
> 계수기를 올리는 헬퍼 안), 실행 중의 실측값은 `[PPR_GPU]`의 `host_syncs_per_statepoint` /
> `h2d_bytes` / `h2d_bytes_elided`가 스스로 인쇄한다.

---

## 1. 단계 A가 남긴 것

`c502856`의 device PPR은 `reset` + `drive`를 옮겼고, 238 단계 A가 그것을 통과시켰다.
남은 구멍은 두 개이고 **둘 다 산술이 아니라 관측**이다.

### 1.1 Picard 라운드마다 host가 멈춘다

`drive`의 한 라운드는 커널 8개다. 그 뒤에 `4 × 256` corner partial을 D2H하고
`cudaStreamSynchronize`를 한 다음, host가 네 합을 chunk 오름차순으로 접고
`RelativeChange`를 `1e-5`와 비교한다. 판정은 8 double이 아니라 **1 bit**인데, 그 1 bit를
보기 위해 스트림 전체가 라운드마다 직렬화된다.

단계 A 수신증이 그 수를 준다: `iterations 816` / `statepoints 35` = **statepoint당 23.31
라운드**. 여기에 마지막 sync 1개를 더해 **statepoint당 24.31 host sync**.

### 1.2 매 statepoint 1.94 MB를 다시 올린다

| 배열 | double | byte | 매 statepoint 바뀌는가 |
|---|---:|---:|---|
| `phif` | `nxyz·ng` = 16,902 | 135,216 | 예 — **단, device가 방금 썼다** |
| `phis` | `nsurf·ng` = 53,384 | 427,072 | 예 — **단, device가 방금 썼다** |
| `jnet` | `nsurf·ng` = 53,384 | 427,072 | 예 — **단, device가 방금 썼다** |
| `xsdf` | 16,902 | 135,216 | 예 |
| `xsrf` | 16,902 | 135,216 | 예 |
| `xsnf` | 16,902 | 135,216 | 예 |
| `xssm` | `nxyz·ng²` = 33,804 | 270,432 | 예 |
| `chif` | 16,902 | 135,216 | **아니오** — `_ref_generation`이 움직일 때만 |
| `crdf` | 16,902 | 135,216 | **아니오** — `RASBERY_PPR_CRDF` 없이는 전 구간 1.0 |
| **합** | | **1,935,872** | |

`phif/phis/jnet`은 outer segment가 device에서 만든 것이고, 그것을 host로 내려서 다시
올린다. `chif/crdf`는 아예 움직이지 않는다.

---

## 2. 단계 B — device convergence

### 2.1 fold는 host의 fold다 (이것이 B0 주장의 전부)

```
kCornerPartials<true><<<blocks_ch, 128>>>     고정 256 chunk, 현행 그대로
kCornerFoldAndCheck<<<1, 4>>>                 corner당 thread 1개
```

`kCornerFoldAndCheck`의 thread `t`는 `partials[t*nchunk + c]`를 **`c = 0 → nchunk-1`
오름차순으로** 0.0에서 시작한 double에 더한다. 그것은 "결정론적 reduction"이 아니라
**host가 D2H한 partial 위에서 돌던 바로 그 루프**다. 256-chunk 트리 reduction은 더 빠르고
합을 옮긴다 — 그래서 계약 테스트가 그 한 줄을 문자열로 붙잡는다.

그 다음 thread 0이 `PPR::drive`의 판정을 그대로 적용한다: 같은 `RelativeChange`
(`__host__ __device__` **하나**를 host_sync arm과 공유한다), 같은 `1e-5`, 같은 네 corner
논리곱. `iters`는 여기서만 증가하고, 통과하면 `converged`를 세운다.

### 2.2 남는 라운드가 아무것도 쓰지 않는다는 것 (`device_stream`)

`device_stream`은 `niter`개의 body를 그냥 다 enqueue한다. 대신 body의 **모든** 커널이
`template <bool kGuarded>`이고 첫 문장이

```cpp
if (kGuarded && pprHalted(x)) return;      // x.loop->converged != 0
```

이다. 라운드 *i*의 fold가 flag를 세우면 이미 enqueue된 *i+1..niter-1*의 커널은 전부
**쓰기 전에 반환한다.** 그러므로 상태는 라운드 *i*에서 얼어붙고, 그것이 host의 `break`가
남긴 상태다.

> **K 라운드마다 검사하는 배치는 채택하지 않았다.** 그것은 진짜 라운드를 더 돌리고, 그러면
> 결과가 움직인다(N1이 된다). 여기서는 **빈 라운드**를 돌린다 — launch는 낭비하고 정확성은
> 낭비하지 않는다. 계획 §WP6 단계 B의 "iteration 동일"을 문자 그대로 만족한다.

`reset()` 반쪽은 **같은 커널을 `<false>`로** 인스턴스화한다. 거기에는 루프가 없고, 아직
정의되지도 않은 flag를 읽는 reset은 그 자체가 버그다.

### 2.3 conditional WHILE (`device_graph`, opt-in)

`GpuOuterWhile.h`의 7-call 시퀀스를 그대로 따른다. arm node와 body의 마지막 node가 **같은
커널**이고, 술어는 `PPR::drive`의 루프 헤더 그 자체다:

```cpp
cudaGraphSetConditional(handle, (st->converged == 0 && st->iters < st->niter) ? 1u : 0u);
```

- arm에서 `iters=0, converged=0` → 1 → do-while의 첫 바퀴
- body 끝에서 fold가 갱신한 값으로 재평가 → host `for`문과 항 대 항으로 일치

**instantiate는 (deck, shape, 바인딩된 포인터 집합)당 1회다.** 키는 `DevCtx` 전체의
`memcmp`다 — capture는 kernel node parameter를 굽고, 단계 C의 borrow가 statepoint 사이에
켜지거나 꺼지면 `phif/phis/jnet`이 움직인다. shape만 키로 삼았다면 그 graph는 **유한하고
그럴듯한 값으로 틀린 버퍼**를 읽는다.

`reigv`는 그래서 `DevCtx`에서 빠져 device의 `PprLoopState`로 들어갔다. by-value로 두면
captured body가 **첫 statepoint의 고유값을 영원히 재생**한다. 계약 테스트가
`x.reigv`를 금지어로 둔다.

capture가 거절되면 (`stage: cudaGetErrorString`) `graph_refusal`에 이름이 남고 arm은
**`device_stream`으로 내려간다 — host로 내려가지 않는다.** capture 실패가 물리 판단처럼
보이면 안 된다.

### 2.4 census — statepoint 하나

| 항목 | `host_sync` (= `c502856`) | `device_stream` | `device_graph` |
|---|---:|---:|---:|
| host sync | **24.31** | **1** | **1** |
| partial D2H | 23.31 × 8,192 B = 190,955 B | **0** | **0** |
| kernel launch (drive) | 23.31 × 8 = 186 | 100 × 9 = **900** | **1 graph launch** |
| graph instantiate | 0 | 0 | **1 / (deck, shape)** |
| 라운드 수 | 23.31 | **23.31** | **23.31** |

`device_stream`이 launch를 늘리는 것은 사실이고 감출 것이 아니다. drive가 133.60 ms /
statepoint / 23.31 라운드 = 커널 하나당 약 716 µs이므로 launch overhead(약 5 µs)는
그 앞에서 3 % 미만이고, 없앤 것은 **23.31번의 직렬화**다. 그 교환이 실제로 이득인지는
§7.4의 wall이 답한다 — `device_graph`가 존재하는 이유가 그 launch 900개이고,
`graph_launches`가 그것을 인쇄한다.

---

## 3. 단계 C — canonical input

### 3.1 세 갈래, 그리고 하나만 조건부다

1. **`crdf`** — `RASBERY_PPR_CRDF` 없이는 전 구간 1.0. `_crdf_generation`은 **쓰기 지점에서**
   증가한다(`PPR::reset`과 `PPR::resetAndDriveGpu` 두 곳 모두 — 계약 테스트가 2를 센다).
   보정이 꺼져 있으면 카운터가 1에 멈추고 업로드는 1회로 끝난다.
2. **`chif`** — `XSSet::refGeneration()`. burnup 보간된 핵분열 스펙트럼은
   `PrecomputeBranchCoefficients`가 다시 만들 때 움직이지 statepoint마다 움직이지 않는다.
   generation이 `0`(caller가 보증 못 함)이면 **업로드한다** — 보수적인 방향.
3. **`phif/phis/jnet`** — outer segment의 canonical nodal set을 **빌린다.**

### 3.2 borrow의 전제와, 그것을 측정하는 방법

borrow는 "PPR 시점에 device 버퍼가 host 배열과 같은 바이트를 들고 있다"를 전제한다.
그것은 **이 arm의 성질이 아니라 outer segment 종료의 성질**이므로, 헤더가 주장하지 않고
`verify`가 측정한다:

```
RASBERY_GPU_PPR_CANONICAL=verify
  → host 배열을 (지금처럼) 자기 블록에 올리고
  → kCanonicalCompare가 borrowed vs uploaded를 __double_as_longlong으로 원소 비교
  → kFoldMismatch가 3×256 chunk 카운트를 접어 canonical_mismatch로 인쇄
```

비교는 **bitwise**다. `!=`는 두 NaN을 다르다고 하고, 허용오차는 더 쉬운 질문에 답한다.
`atomicAdd`는 이 TU에서 금지어이므로(계약 property 7) chunk별 카운트 + 1-thread fold다.

**all-or-nothing**은 세 층에서 각각 확인한다 — Driver의 제안(`seg.canonicalNodalBound()`),
`PPR::resetAndDriveGpu`의 `_canonical.complete()`, backend의 `have_set`. 한 층이 잊으면
조용하기 때문이다. 제안은 **statepoint마다 새로 하고, 안 하면 철회된다**: 침묵이 이전
statepoint의 제안을 남기면 host에서 outer를 돈 statepoint가 한 outer 낡은 device 버퍼를
읽는다.

### 3.3 바이트 — 무엇이 얼마나 줄어드는가

정상 상태(statepoint 2 이후, `chif`/`crdf`가 이미 올라가 있는 상태) 기준:

| 모드 | `h2d_bytes` / statepoint | 이 statepoint에서 elided | 감소 |
|---|---:|---:|---:|
| `c502856` (gate도 borrow도 없음) | 1,935,872 | 0 | — |
| generation gate만 (`canonical=off`) | 1,665,440 | 270,432 | **14.0 %** |
| `+ borrow` (`canonical=1`) | 676,080 (= xsdf/xsrf/xsnf 405,648 + xssm 270,432) | 1,259,792 | **65.1 %** |
| `verify` | 1,665,440 | 270,432 | 14.0 % — **측정 모드이지 성능 모드가 아니다** |

35 statepoint 누계(첫 statepoint는 `chif`/`crdf`를 올린다):

| 모드 | `h2d_bytes` | `h2d_bytes_elided` |
|---|---:|---:|
| `canonical=off` | 58,560,832 | 9,194,688 |
| `canonical=1` | 23,933,232 | 43,822,288 |

`off` 대비 누계 감소 **59.1 %**, 정상 상태 statepoint 기준 **65.1 %**.

> **계획의 단계 C 목표는 80 %이고 이 구현은 65.1 %다. 미달이며, 이유는 XS 블록이다.**
> `xsdf/xsrf/xsnf/xssm` 676,080 B가 남는데, 이것들은 `CanonicalRegion::LiveXs`로 device에
> 이미 있다. 그런데 오늘 PPR이 닿을 수 있는 경로가 없다: `CudaOuterSegment::canonicalNodalSet()`은
> `Flux/Jnet/Phis` 세 개만 publish하고(`CudaOuterGraph.cu:1092`), `XsReconBackend::canonicalBuffers()`가
> 돌려주는 것도 Driver가 넘긴 그 세 개뿐이다(`Driver.h:2011`, `CudaXsReconBackend.cu:4139`).
> LiveXs를 빌리려면 arena slot view를 PPR까지 끌어와야 하고, 그것은 `src/GpuCanonicalState.h`와
> segment API의 변경이다 — **이 커밋의 소유 범위 밖이고, 80 %는 그 변경과 함께 다시 청구한다.**
> 여기서 80 %를 주장하지 않는 편이 낫다.

---

## 4. 단계 D — `reconstructPinPower` 이식

### 4.1 왜 지금인가 (5.4 %가 아니라 9 MB다)

`floor_wall`은 재구성을 8.48 ms/statepoint, PPR의 **5.4 %**로 쟀고 `c502856`은 그것을
근거로 host에 남겼다. 바뀐 것은 둘이다.

1. 단계 B·C가 `reset + drive`를 줄이면 5.4 %는 커지는 지분이다.
2. **일곱 계수 배열이 D2H되는 유일한 이유가 이 host loop다.**

| 배열 | double | byte |
|---|---:|---:|
| `p`, `c`, `q` | 각 `nxyz·ng·15` = 253,530 | 각 2,028,240 |
| `l` | `nxyz·ng·9` = 152,118 | 1,216,944 |
| `a` | `nxyz·ng·8` = 135,216 | 1,081,728 |
| `phic` | `nxyz·ng·4` = 67,608 | 540,864 |
| `bt` | `nxyz·ng` = 16,902 | 135,216 |
| **합** | 1,132,434 | **9,059,472** |

9.06 MB는 이 arm이 **올리는 것 전부(1.94 MB)보다 크다.** 소비자를 옮기면 전송이 사라진다.

### 4.2 form function은 업로드가 아니라 registry다

`fmap`/`gmap`은 `std::map`으로 키가 잡힌 Chiffon depletion point 안에 있고, host는
(plane, assembly)마다 매 statepoint 새로 보간한다. **보간된 맵을 올리면**
`nz·nxya·(1+ng)·npina` double — KNGR 크기에서 수십 MB이고, 대체하려는 loop보다 비싸다.
그래서 반대로 한다:

- **한 번**: 모든 model의 모든 reference depletion point의 `_gmap`/`_fmap`을 slot으로
  평탄화해 올린다(라이브러리 데이터이므로 움직이지 않는다).
- **매 statepoint**: (plane, assembly)마다 `lo_slot`, `hi_slot`, `alpha` 세 값.
  `24 B × nz × nxya`.

보간 자체는 host의 식을 host의 순서로 device에서 계산한다:

```
gmap = g_lo + alpha * (g_hi - g_lo)         // PPR.cpp: loDpt._gmap[gi] + alpha * (hi - lo)
```

`lo`/`hi`/`alpha`를 고르는 `lower_bound` 걷기는 **host에 남는다.** `std::map` 순회이고,
여기서 틀리면 보이지 않기 때문이다.

### 4.3 무엇이 host의 순서이고 무엇이 아닌가

| 항목 | device | 이유 |
|---|---|---|
| overlap × group 누적 | **host 순서 그대로** | `power_integral`은 (overlap, g) 순으로 누적된다 — thread 하나가 한 핀을 전부 맡는 이유 |
| 정규화 partial | **host의 256 chunk**, 오름차순 fold | 정규화 상수는 **모든** 핀 출력에 곱해진다 |
| radial 합 | **k 오름차순** | host의 loop 순서 |
| Fq / FΔH | `fmax`, chunk 분할 자유 | max는 부동소수점에서 **정확히** 결합·교환법칙을 만족하고, `fmax`는 `std::max(a, NaN) == a`와 같이 NaN을 버린다 |
| `exp` | device libm | **N1의 원천이고 유일한 원천이다** |

NaN padding은 `__longlong_as_double(0x7ff8000000000000LL)` — `quiet_NaN()`의 표현 그대로다.
`nan("")`가 아니라 비트로 적는 이유는 핀 맵이 HDF5로 나가고 두 arm의 `h5diff`가
**payload를 비교**하기 때문이다.

### 4.4 D2H — light output에서 32 바이트

| 상황 | D2H / statepoint |
|---|---:|
| 현행 (`RECON=0`) | 9,059,472 (계수 7배열) |
| `RECON=1`, `pin_info` 꺼짐 | **32** (`inv_avg_power`, `frp`, `fqp`, spare) |
| `RECON=1`, `pin_info` 켜짐 | 32 + `nxya·nz·npina·8` (핀 맵) |
| `+ pin_flux` | 위 + `× ng` |

`materialize_pin`은 발명한 조건이 아니다. `Geometry::PinPower()`의 **독자는 하나**
(`IO.cpp`의 `if (d.print_opt.pin_info)`)이고, Driver가 그 플래그를 그대로 넘긴다.

### 4.5 계수를 안 내리는 대가, 그리고 그 수리

계수 7배열의 D2H를 없애려면 drive가 **재구성이 device에서 돌 것이라는 약속** 위에서
그것을 생략해야 한다. 약속이 깨지는 경로가 하나 있다 — 재구성 중의 CUDA 실패, 혹은
pointwise 모드를 요구하는 호출.

그때 host 배열은 **직전 statepoint의 계수**를 들고 있고, 그대로 host loop가 돌면
*지난 statepoint의 핀 출력을 이번 statepoint의 정규화로* 재구성한다. 전부 유한하고
전부 그럴듯하다. 그래서 수리가 선택이 아니다:

```cpp
if (_coeffs_device_only) {          // drive가 생략했다
    _gpu->noteReconRepair();        // 수신증에 센다
    reset(_reigv, _jnet, _phif, _phis);
    drive(_last_niter);             // host에서 다시 만든다
}
// 그 다음에야 host loop가 _p/_a/_c/_bt를 읽는다
```

수리는 정확하지만 **낭비한 drive**다. `recon_repairs`가 0이 아니면 그 arm은 생략하지
말았어야 한다 — 그 판단을 수신증이 대신 해 준다.

### 4.6 순 전송량 (KNGR, 정상 상태 statepoint, `pin_info` 꺼짐)

| | `c502856` | B·C 이후 | **+ D** |
|---|---:|---:|---:|
| H2D | 1,935,872 | 676,080 | 676,080 + 177,456 (xskf + plane 3종) = **853,536** |
| D2H | 9,059,472 + 190,955 (partial) | 9,059,472 | **32** |
| **합** | **11,186,299** | 9,735,552 | **853,568** |

`c502856` 대비 **−92.4 %**. 표의 D2H·H2D는 예측이고, `[PPR_GPU]`의 `h2d_bytes` /
`d2h_bytes`가 실측이다 — 어긋나면 수신증이 맞고 이 표가 틀렸다.

---

## 5. 단계 E — batch arena는 여전히 조건부다

계획은 `CudaPprArena`를 "M64 프로파일에서 PPR이 10 % 이상"일 때만 만들라고 한다.
그 전에 per-slot backend가 증명해야 하는 것은 **statepoint마다 할당하지 않는다**는 것
하나이고, 그것은 주장이 아니라 계수기가 되었다:

- `allocations` — 이 인스턴스가 낸 모든 `cudaMalloc`/`cudaMallocHost`. shape 하나면
  device 23 + pinned 3 = **26**이고, statepoint 수와 무관하다.
- `reallocations` — 두 번째 이후의 `ensureShape`. **`> 0`이면 statepoint 루프 안에서
  할당했다는 뜻이고, 그때가 `CudaPprArena`가 추측이 아니라 필요가 되는 시점이다.**

계약 테스트는 `ensureShape` 본문 밖에 `cudaMalloc`이 하나라도 있으면 실패한다.
backend·stream·모든 device 버퍼는 `PPR` 객체에 달려 있고 `PPR`은 `Driver::Drive()`의
지역변수이므로 `--batch-mode M`은 슬롯마다 자기 것을 갖는다(property 3, 변함없음).

---

## 6. 계약 테스트 — 27 property, 음성대조 38

`tools/test_ppr_gpu_contract.py`. property 1–10은 `c502856`의 것 그대로,
11–20이 WP6 단계 B/C/E, 21–27이 단계 D다. **음성대조가 이번에 추가되었다**: 각 항목이 소스를 한 군데 망가뜨리고
"문제가 하나도 보고되지 않으면" 실패한다 — 실패하는 것을 본 적 없는 계약 테스트는 주석이다.

| # | 대조군 | 무엇을 지키는가 |
|---|---|---|
| 1 | fallback 제거 | property 1 |
| 2 | device fold 역순 | 11 (association) |
| 3 | fold 커널 삭제 | 11 |
| 4 | body 커널 guard 제거 | 12 |
| 5 | reset을 `<true>`로 | 12 |
| 6 | 세지 않는 sync 추가 | 13 |
| 7 | `reigv`를 by-value로 | 14 |
| 8 | graph 키 무력화 | 15 |
| 9 | graph 거절이 host로 | 15 |
| 10 | borrow가 부분집합 허용 | 16 |
| 11 | borrow 기본 ON | 16 |
| 12 | 제안이 segment 무관 | 16 |
| 13 | verify가 `!=` 비교 | 17 |
| 14 | generation gate 영구 억제 | 18 |
| 15 | crdf bump를 쓰기에서 분리 | 18 |
| 16 | statepoint 경로의 할당 | 19 |
| 17 | realloc 계수기 삭제 | 19 |
| 18 | receipt 필드 손실 | 20 |
| 19 | schema 미증가 | 20 |
| 20 | stub 접근자 누락 | 8 (링크 오류는 GPU 빌드가 못 본다) |
| 21 | PPR knob을 `kArmEnv`에 | 10 |
| 22 | atomic reduction 복귀 | 7 |
| 23 | drive 없이 재구성 | 21 |
| 24 | `last_drive_ok` 미하강 | 21 |
| 25 | PPR이 자기 drive 플래그를 잊음 | 21 |
| 26 | 재구성 기본 ON | 22 |
| 27 | host 재구성이 fallback이 아님 | 22 |
| 28 | `materialize`를 상수로 | 23 |
| 29 | device NaN payload | 24 |
| 30 | 정규화 fold 역순 | 25 |
| 31 | `rsq2` 표류 | 25 |
| 32 | registry가 매 statepoint 업로드 | 26 |
| 33 | 핀 맵을 항상 내림 | 23 |
| 34 | 재구성 할당이 statepoint 경로로 | 19 |
| 35 | 계수를 무조건 생략 | 27 |
| 36 | 생략 후 수리 없음 | 27 |
| 37 | 수리가 host loop 뒤에 | 27 |
| 38 | 수신증이 수리 계수 손실 | 27 |

---

## 7. 238 runbook

로컬에 nvcc가 없다. **첫 관문은 238 컴파일이다.**

### 7.0 빌드

```bash
cmake --build build -j          # CudaPprBackend.cu 는 --fmad=false 로 컴파일된다
```

`CudaPprBackend.cu`는 CUDA 12.3+ 에서 conditional node API를 켠다(`RASBERY_HAS_PPR_WHILE`).
238은 CUDA 13이므로 `cudaGraphAddNode`의 13.0 시그니처 분기가 컴파일된다 —
**로컬에서 12060/13000 양쪽 문법 검사는 했지만 nvcc가 본 적은 없다.**

### 7.1 단일 production arm — B0 게이트가 먼저다

`RASBERY_GPU_PPR`은 기본 OFF이고 pin 데이터셋 이외에는 아무것도 바꾸지 않아야 한다.
그것이 이 arm의 첫 주장이므로 먼저 그것을 다시 확인한다.

```bash
export CUDA_VISIBLE_DEVICES=0
OUT=$PWD/wp6
# A: PPR OFF (현행 기준)
RASBERY_GPU_PPR=0                        <production arm> ...   # -> $OUT/a.h5, a.log
# B: PPR ON, 새 기본값(device_stream + canonical off)
RASBERY_GPU_PPR=1                        <production arm> ...   # -> $OUT/b1.h5, b1.log
RASBERY_GPU_PPR=1                        <production arm> ...   # -> $OUT/b2.h5  (ON x2)
```

합격 조건 — **전부**:

1. `h5diff -c a.h5 b1.h5` 에서 **pin 데이터셋을 제외한 모든 데이터셋이 0 차이**
   (pin power / pin flux / Fq / FdH 는 N1이므로 §7.2에서 따로 본다)
2. trajectory digest **`0d15abf29d222a02` / `4382`** 가 A와 B에서 동일
   (PPR은 `kArmEnv`에 없고 궤적을 먹이지 않는다 — 이 줄이 그것을 확인한다)
3. **ON×2 결정론**: `h5diff b1.h5 b2.h5` = 0 차이, 전 데이터셋
4. `[RASBERY][PPR_GPU]` 의 `host_fallbacks == 0`, `statepoints == 35`
5. `[RASBERY][PPR_GPU]` 의 **`reallocations == 0`**, 그리고 `allocations`가
   **statepoint 수를 바꿔도 같은 값**일 것(5 statepoint 덱과 35 statepoint 덱에서 동일).
   단계 D를 켜면 값 자체는 커진다(재구성 블록의 할당이 더해진다) — 불변량은 크기가
   아니라 **statepoint 수와 무관하다**는 것이다

1~5 중 하나라도 어긋나면 **여기서 멈춘다.** wall은 재지 않는다.

### 7.2 Gate A — pin / Fq / FdH (NaN-aware)

```bash
python3 tools/gate_a_compare.py --a $OUT/a.h5 --b $OUT/b1.h5 --nan-aware
```

`reconstructPinPower`는 유효 노드가 없는 핀에 **NaN을 채운다**(`PPR.cpp`,
`quiet_NaN`). 그러므로 비교는 (i) NaN 위치 집합이 A와 B에서 **정확히 같을 것**,
(ii) 나머지 원소에 대해서만 max relative difference를 잴 것. NaN 위치가 다르면
그것은 수치 차이가 아니라 **다른 재구성**이므로 델타를 보고할 것이 아니라 실패다.

단계 A가 남긴 기준: **Gate A 0.0000 / 35 statepoint 전부**. 단계 B는 fold의 association을
바꾸지 않으므로 **이 값이 유지되어야 한다** — 0.0000이 아니면 §2.1의 B0 주장이 거짓이다.

### 7.3 단계 B — arm 3종 대조

```bash
RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_DEVICE_LOOP=0                        <arm>  # host_sync (= c502856)
RASBERY_GPU_PPR=1                                                      <arm>  # device_stream (기본)
RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1                              <arm>  # device_graph
```

기록표:

| 항목 | host_sync | device_stream | device_graph |
|---|---:|---:|---:|
| `loop_arm` | `host_sync` | `device_stream` | `device_graph` |
| `iterations` | **816** | | |
| `host_syncs_per_statepoint` | ~24.3 | **1.000** | **1.000** |
| `graph_launches` / `graph_builds` | 0 / 0 | 0 / 0 | 35 / **1** |
| `graph_refusal` | `""` | `""` | `""` |
| `ppr_reset + ppr_drive` (ms/sp) | | | |
| h5diff vs host_sync arm | — | **0 / 644** | **0 / 644** |

**`iterations`가 셋 다 816이 아니면 채택하지 않는다.** 그것이 §2.2의 "빈 라운드"
주장 전체이고, 어긋나면 N1이며 그렇게 적어야 한다.
`graph_builds > 1`이면 `DevCtx`가 statepoint 사이에 움직였다는 뜻이므로 원인을 찾기 전에
`device_graph`를 채택하지 않는다.

채택 문턱(계획 §WP6 성능 게이트): **PPR phase(`ppr_reset + ppr_drive`) 25 % 이상 단축.**

### 7.4 단계 C — borrow의 건전성이 먼저, 바이트는 그 다음

```bash
# 1) 건전성 — 이것이 먼저다
RASBERY_GPU_PPR=1 RASBERY_GPU_SHARED_STATE=1 RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1 \
RASBERY_GPU_PPR_CANONICAL=verify   <arm>
```

합격 조건:

- `[RASBERY][PPR_GPU]` 의 **`canonical_mismatch == 0`**
- `canonical_statepoints == statepoints` (0이면 binding이 붙지 않은 것이고,
  `[RASBERY][OUTER_GPU]`의 `canonical_nodal=1`을 먼저 확인한다)

`canonical_mismatch > 0`이면 **borrow는 이 덱에서 건전하지 않다.** 그때는
`RASBERY_GPU_PPR_CANONICAL`을 켜지 않고, 어느 배열이 어긋났는지(§3.2의 세 launch를
`slot` 0/1/2로 분리해 두었다) 를 근거로 outer segment 종료 쪽에서 원인을 본다.

```bash
# 2) 바이트
RASBERY_GPU_PPR=1 ... RASBERY_GPU_PPR_CANONICAL=1   <arm>
```

| 항목 | `off` | `1` (borrow) |
|---|---:|---:|
| `h2d_bytes` (전체 35 sp) | 58,560,832 | 23,933,232 |
| `h2d_bytes_elided` | 9,194,688 | 43,822,288 |
| 누계 감소 | — | **59.1 %** |
| 정상 상태 statepoint 감소 | — | **65.1 %** |

계획의 80 % 문턱에는 **미달이고, §3.3이 그 이유와 남은 경로를 적어 두었다.**
위 표의 값은 예측이다 — 실측이 어긋나면 `h2d_bytes`가 맞고 이 표가 틀린 것이다.

### 7.4b 단계 D — 재구성

```bash
RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_RECON=1   <arm>
```

합격 조건 — **순서대로**:

1. `[RASBERY][PPR_GPU]` 의 `recon_statepoints == statepoints`, `recon_refusal == ""`.
   0이면 `recon_refusal`이 이유를 말한다(`form-function map size != npins^2` 등).
2. **`recon_repairs == 0`.** 0이 아니면 계수 D2H를 생략하지 말았어야 한다는 뜻이고,
   그 statepoint는 host drive를 한 번 더 돌았다.
3. `pin_materializations == (pin_info가 켜진 statepoint 수)`. 그보다 크면 인쇄하지 않는
   statepoint가 5 MB를 내리고 있다.
4. `d2h_bytes`가 `RECON=0` 대비 두 자릿수 배로 줄었다.
5. **Gate A (§7.2)**: pin power max relative difference, Fq/FΔH 델타.
   **NaN 위치 집합이 먼저 같아야 한다.** `exp`가 유일한 차이 원천이므로 여기서 나오는
   델타는 `reset+drive`의 것과 같은 크기여야 한다 — 자릿수가 다르면 §4.3의 순서 주장 중
   하나가 거짓이다.
6. `pin_info`가 켜진 statepoint에서 `pin_power` 데이터셋을 host arm과 비교한다
   (`h5diff -c`). N1이므로 0을 요구하지 않고 Gate A 문턱을 쓴다.

채택 문턱(계획 §WP6): **PPR 전체 40 % 이상 단축.** 그 40 %는 `ppr_recon`의 8.48 ms만으로는
나오지 않는다 — 9 MB D2H의 소멸까지 합쳐야 나오고, 그래서 4번이 5번만큼 중요하다.

### 7.5 wall — hot median of 3

```bash
for i in 1 2 3; do  RASBERY_GPU_PPR=1 <arm> ; done      # 새 기본값
for i in 1 2 3; do  RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_DEVICE_LOOP=0 <arm> ; done   # 현행 PPR-ON
```

runner가 재는 **현행 PPR-ON wall**이 기준선이다(PPR-OFF가 아니다 — 단계 B/C는 PPR-ON
안에서의 변경이다). 단일 16.9 s는 확인용이고 판정 기준으로 쓰지 않는다.

### 7.6 배치 — 값이 나오는 곳

```
8×M8 + MPS,  CUDA_VISIBLE_DEVICES=0,  RASBERY_GPU_PPR=1
```

기준 **878 c/h**(WP4 최고 arm) 대비. 채택 문턱 **+3 %**(WP5 문서 §6.4의 노이즈 규칙).
동시에 확인:

- 슬롯 8개 각각의 `[PPR_GPU]` 줄에서 `reallocations == 0`
- 슬롯마다 `allocations == 26` (프로세스 합이 아니라 **슬롯당**)
- `graph_builds`가 슬롯당 최대 1

여기서 PPR이 배치 GPU 시간의 **10 % 이상**이면 계획 §WP6 단계 E의 `CudaPprArena`가
비로소 근거를 얻는다. 미만이면 per-slot backend를 유지한다 — 그것도 결과다.

---

## 8. 롤백

| 무엇을 되돌리나 | 어떻게 |
|---|---|
| WP6 전체 | `RASBERY_GPU_PPR=0` (기본값) |
| 단계 B만 | `RASBERY_GPU_PPR_DEVICE_LOOP=0` → `c502856`의 host-sync 루프 |
| WHILE만 | `RASBERY_GPU_PPR_GRAPH` 미설정 (기본값) |
| 단계 C만 | `RASBERY_GPU_PPR_CANONICAL` 미설정 (기본값) |
| 단계 D만 | `RASBERY_GPU_PPR_RECON` 미설정 (기본값) |

`RASBERY_GPU_FULL=1`에서는 롤백 arm을 켠 채로 돌리지 않고 명시적 configuration error를
낸다(계획 §WP6 롤백 절). `RASBERY_GPU_PPR`이 꺼진 바이너리의 stdout은 이 커밋 이전과
동일하다 — receipt는 arm이 켜졌거나 `RASBERY_STATEPOINT_TELEMETRY`일 때만 인쇄된다.

---

## 9. 남은 구멍

1. **XS 블록이 여전히 매 statepoint 올라간다** (676,080 B). §3.3.
2. **`pointwise` 재구성(`use_quadrature=false`)과 `RASBERY_PPR_MODE=master`는 이식하지
   않았다.** 둘 다 다른 스킴이고, arm은 근사하지 않고 거절한다 — 다만 계수 D2H를 이미
   생략한 statepoint에서 거절하면 §4.5의 수리가 돌고 그것은 낭비다. production arm은
   quadrature이므로 실제로는 일어나지 않아야 하며, `recon_repairs`가 그것을 확인한다.
3. **`device_stream`은 launch를 900개로 늘린다.** `device_graph`가 그것을 1개로 만들지만
   capture 주장을 하나 더 얹는다. 어느 쪽이 이득인지는 §7.3/§7.5의 실측이 정한다.
4. **핀 맵을 안 내리면 `Geometry::PinPower()`는 낡은 채로 남는다.** 오늘 그것을 읽는
   것은 `IO.cpp` 하나뿐이고 Driver가 그 플래그를 그대로 넘기므로 안전하지만, **새 host
   독자가 생기면 이 규칙이 먼저 깨진다.** 계약 테스트 property 23이 IO의 조건문과
   Driver의 인자를 함께 붙잡는 이유다.
5. **nvcc가 이 파일들을 본 적이 없다.** 로컬 문법 검사는 shim 위에서 한 것이고,
   `__shared__`·launch configuration·레지스터 압력은 검사되지 않았다.
