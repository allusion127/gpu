# WP15 — micx/lmpx 잔류(residency): D2H 85 %를 지연 다운로드로 (2026-08-30, KO)

브랜치 `codex/exact-throughput-campaign`, 기준 `914f6b3`.
입력은 `docs/WP13_HOST_DEVICE_TRAFFIC_20260830_KO.md`의 **로드맵 1번**이고,
이 문서는 그 1번의 구현과, 같은 로드맵 2·3번에 대한 **인구조사 결과와 미구현 사유**다.

---

## 0. 요약

1. **flatxs micx/lmpx 다운로드를 없애지 않았다. 빚으로 바꿨다.**
   `RASBERY_GPU_MICX_RESIDENT=1`이면 `solveFlatXs`는 59.5 MB / 20 copy를
   내려받지 않고 **device에 두고**, host가 처음 그 배열을 만질 때
   `XSSet::EnsureMicxHost`가 **같은 copy 목록을 같은 offset에서** 한 번 실행한다.
   같은 바이트를, 다른 시점에. 그래서 B0다.
2. **reader 분류가 본체다.** `_micx`/`_lmpx`를 이름으로 부르는 XSSet 함수 17개 +
   값 접근자 2개를 전수 분류했다 — (a) 주소만 잡는 device 소비자 **5**,
   (b) 스칼라 11슬롯만 읽는 슬라이스 소비자 **7**, (c) 산란까지 읽는 전체 소비자 **7**.
   `tools/test_micx_resident_contract.py`의 M2는 이 분류를 **파일에 대한 전수 열거**로
   다시 세므로, 여기 없는 reader가 새로 생기면 게이트가 떨어진다.
3. **기대 절감: D2H 22.85 GB → 3.33 GB (−19.5 GB, 런 D2H의 68 %)**,
   `solveFlatXs:drain` 384회 중 **316회가 59.5 MB를 기다리지 않게 된다**.
   측정 D2H 28 GB/s 기준 **~0.70 s**, wall 11.34 s의 **6 %**.
4. **로드맵 2번(nodal consts 3.92 GB)은 B0로 지울 수 없다** — §4. 계측만 넣었다.
   **로드맵 3번(xe commit xs 1.66 GB)도 이번에 하지 않았다** — §5. 이유는 다르다.
5. `trajectory::kArmEnv`에 **넣지 않았다**. §6.

---

## 1. 사실 (238, v5, KNGR cycle-1, 35 statepoint, 4,377 outer, wall 11.34 s, `914f6b3`)

`[RASBERY][XFER][LEDGER]` 상위 지점 중 이 문서가 다루는 것:

| 지점 | dir | calls | bytes |
|---|---|---:|---:|
| `CudaXsReconBackend.cu:DeviceBlock::download:flatxs micx mic` | D2H | 3,456 | **18.22 GB** |
| `…:flatxs micx msm` | D2H | 384 | **4.05 GB** |
| `…:flatxs lmpx lmp` | D2H | 3,456 | 0.47 GB |
| `…:flatxs lmpx lsm` | D2H | 384 | 0.10 GB |
| **micx/lmpx 블록 합** | D2H | **7,680** | **22.85 GB** |
| `CudaXsReconBackend.cu:solveNodal:consts` | H2D | 9,675 | 3.92 GB |
| `CudaXsReconBackend.cu:DeviceBlock::download:xe commit xs` | D2H | 12,287 | 1.66 GB |
| `CudaXsReconBackend.cu:solveFlatXs:drain` | sync | 384 | **1.047 s** |

**flatxs 호출 수 F = 384**가 여기서 확정된다(3,456 / 9 = 384, 384 / 1 = 384).
WP13 §4.1이 바이트 항등식으로 역산한 `F ≈ 410`은 6 % 과대였다.
한 호출의 micx/lmpx 블록은 `9·(lmp+mic) + lsm + msm`
= 9·(135,216 + 5,273,424) + 270,432 + 10,546,848 = **59,495,040 B**.

`consts` 9,675 / 9 = **1,075**회가 `_const_generation` 전진 횟수이고,
같은 런의 Xe device step은 **1,117**이다. §4가 이 일치를 다룬다.

---

## 2. reader 분류표

`_micx` / `_lmpx`(라이브 블록)를 이름으로 부르는 지점 전수.
`_ref_micx` · `_micx_bos` · `_micx_generation` · `lib_micx` · `coeff_micx`는
**다른 배열**이므로 제외했다(각각 참조상태·BOS 스냅샷·세대카운터·라이브러리).

### (a) device 소비자 / 주소만 — 다운로드하지 않는다 (5)

| 지점 | 무엇을 하는가 | 왜 안전한가 |
|---|---|---|
| `MakeFlatXsHostView` | 11+11 주소를 view에 담는다 | 원소를 하나도 읽지 않는다. **EnsureMicxHost 자신이 목적지를 짓는 데 쓰는 view**이므로 여기 호출을 넣으면 무한 재귀다 |
| `TryUpdateFlatXSGpu` | 같은 주소를 page-lock, view를 device에 넘김 | 주소만. 성공 후 backend에서 **빚을 되읽어** 플래그를 세운다 |
| `PrepareXeDeviceCall` | xsrecon/Xe view에 11슬롯 주소 | `stage()`가 `micx_generation` 불일치일 때만 업로드하는데, 지연 solve 뒤에는 두 세대가 **구성상 일치**한다(`solveFlatXs`가 residency를 `micx_generation_next`로 올리고 `UpdateFlatXS`가 `_micx_generation`을 바로 그 값으로 올린다). 그래도 값싼 확인 한 줄을 두었다 |
| `Initialize` | `allocate()` | device 블록이 아직 없다 |
| `~XSSet` | `rasberyUnpinHost(base)` | 해제 시점, 원소를 읽지 않는다 |

### (b) 스칼라 슬라이스 — `MicxNeed::Scalars`, 49.0 MB (7)

산란 블록(`msm`+`lsm`, 10.5 MB)을 **읽지 않는** 소비자들. 전부 11개 스칼라
슬롯을 노드별로 축약(condense)한다.

| 지점 | 소비 형태 | 런당 발화 |
|---|---|---:|
| `PredictorStep` | 11슬롯 전체를 `_micx_bos`로 복사 | 34 (dep step) |
| `DepleteGpu` | CRAM predictor의 H2D 원본 4슬롯 | 34 (같은 epoch, no-op) |
| `CorrectorStepGpu` | CRAM corrector의 H2D 원본 4슬롯 | 34 |
| `CorrectorStep` (host fallback) | `eos_ptrs` 11슬롯 | 0 (device corrector 성공) |
| `DepleteNode` | 전이행렬 축약 11슬롯 | 0 (device CRAM) |
| `EvaluateEquilibriumXenon` | Xe 대수 축약 11슬롯 | 0 (device Xe) |
| `XSSet::micx()` 접근자 | `IO::WriteStepToResult`의 node-monitor 덤프, `NodeSpectralIndex`의 host arm | 0 (덤프 미사용, micWork 경유) |

**PROD에서 실제로 발화하는 것은 위 3개, 런당 ≈ 68회다.**

### (c) 전체 블록 — `MicxNeed::All`, 59.5 MB (7)

산란까지 읽거나 **쓰는** 지점. 쓰기가 여기 있는 이유는 §3의 두 번째 규칙이다.

| 지점 | 소비 형태 | 런당 발화 |
|---|---|---:|
| `unpackXS` | **writer** — 11슬롯 + 산란 | 0 (rodded 없음) |
| `ApplyBranchDeltaIdToNode` | **writer** — 델타 누적, 산란 포함 | 0 |
| `UpdateUnroddedNodeXS` | **writer** — host arm의 노드 본체 | 0 (device arm) |
| `Reconstruct` | 전체 재구성 | 0 |
| `ReconstructNode` | 노드 재구성, `_micx.xssm` 읽음 | 0 (rodded 없음) |
| `UpdateEquilibriumXenon` | host Xe 루프 + `RASBERY_XSRECON_DUMP` | 0 |
| `XSSet::micxssm()` 접근자 | 산란 원소 값 | 0 |

**PROD에서 0회다.** 즉 KNGR cycle-1에서 산란 블록(4.15 GB / 768 copy)은
**한 번도 host로 내려오지 않는다**. 이것이 슬라이스를 만든 이유다.

---

## 3. 구현 — `RASBERY_GPU_MICX_RESIDENT=1`

### 3.1 계약, 한 문장

> **빚이 남아 있는 동안 host 코드는 `_micx`/`_lmpx`를 읽지도 쓰지도 않는다.**

두 번째 규칙이 딸려 온다: **writer도 먼저 갚는다.** 늦게 도착한 다운로드가
host가 방금 쓴 열(column) 위에 이전 epoch 값을 덮기 때문이다. 이것이
`UpdateFlatXS`의 "runs after the device download so the device's whole-array
downloads cannot clobber these columns" 주석이 원래 지키던 순서이고,
지연 다운로드는 그 순서를 **호출 지점으로 옮긴 것**이다.

### 3.2 device 쪽 (`src/CudaXsReconBackend.cu` / `.h`)

- `Impl::micx_scalars_pending` / `micx_scatter_pending` — 빚. 절반씩 따로
  다는 이유는 §2(b)/(c)의 reader가 서로 다르기 때문이다.
- `solveFlatXs`: 20 copy 블록이 `!skip_micx_dl && !micx_resident`일 때만 발화.
  `micx_resident = rasberyGpuMicxResidentEnabled() && mark_micx_resident`.
  **`mark_micx_resident`를 곱한 것이 핵심이다.** false는 "곧 rodded 노드가 host
  열을 다시 쓴다"는 뜻이고, 그때 caller는 residency를 버려서(`resident_micx_generation = 0`)
  **다음 solve가 블록을 host에서 다시 올린다**. 그 업로드를 건너 살아남은 빚은
  업로드가 덮어버린 바이트로 갚히게 된다 — 이전 epoch의 값으로.
- `downloadFlatXsMicx(view, scalars, scatter)`: **같은 copy 목록, 같은 offset,
  같은 `Impl::download` 래퍼**. 그래서 실체화된 host 배열은 플래그-off arm이
  즉시 내려받았을 배열과 비트 동일하다. 끝에 `streamSync` — caller는 곧
  역참조할 host reader다.
- 업로드 두 지점(`solveFlatXs`의 micx 블록, `stage()`의 micx 블록)에
  **구성상 도달 불가능한 경보**를 달았다. 도달하면 그것은 "EnsureMicxHost를
  거치지 않는 host reader가 있다"는 뜻이고, 조용한 오답 대신 stderr가 나온다.

### 3.3 host 쪽 (`src/XSSet.cpp` / `.h`)

- `XSSet::EnsureMicxHost(MicxNeed)` — **빚을 갚는 유일한 지점**.
  - 빚이 없으면 **relaxed atomic load 두 번과 return**. flag-off arm은 출하된
    코드 그대로이고, A/B는 지연을 재지 bookkeeping을 재지 않는다.
  - 빚이 있으면 `#pragma omp critical`. reader 중에 `ReconstructNode`,
    `UpdateUnroddedNodeXS`, `DepleteNode`처럼 **OpenMP 루프 안의 노드 본체**가
    있기 때문에, 이것이 없으면 갚는 스레드가 다른 스레드가 쓰고 있는 버퍼로
    다운로드를 내린다. 첫 스레드가 갚고 나머지는 플래그가 이미 내려간 것을 본다:
    8,451 노드 루프에 다운로드 **1회**, 스레드당 1회가 아니다.
  - 플래그는 `mutable std::atomic<bool>`. `mutable`은 EnsureMicxHost가 const이기
    때문이고(교차단면 소비자는 전부 const다), `atomic`은 위 경합 때문이다.
  - 실패하면 **fail-open하지 않는다**. fail-open은 "host 본체를 대신 돌려라"인데,
    그 host 본체가 읽을 교차단면이 바로 device가 방금 넘기기를 거부한 것들이다.
    `[RASBERY][FAIL][micx]` 후 `abort()`.
- `TryUpdateFlatXSGpu`: 성공 후 **backend에서 플래그를 되읽는다**.
  자기 env 판독으로 추정하지 않는 이유는 backend가 이쪽이 열거하지 않는
  경우(`RASBERY_FLATXS_SKIP_MICX_DL`, `any_rodded`)에도 지연을 거절하기 때문이다.
- 값 접근자 `micx()` / `micxssm()`가 갚는다 — `IO.cpp`의 node-monitor 덤프와
  `NodeSpectralIndex`가 라이브 블록에 닿는 **유일한 경로**이므로, 호출처마다가
  아니라 여기서 한 번 막는 것이 옳다.

### 3.4 영수증 (`src/Driver.h`, `[RASBERY][MICX]`)

```
[RASBERY][MICX] {"schema_version":1,"slot":0,"arm":1,"resident_hits":384,
                 "lazy_downloads":0,"slice_downloads":68,
                 "materialised_per_hit":0.177,"bytes_saved":19521404928,
                 "mb_saved":18617.9,"nodal_const_uploads":9675,
                 "nodal_const_mb":3741.2}
```

- `resident_hits == 0` && `arm == 1` → **G0**. 플래그가 어떤 solve에도 닿지
  않았다는 뜻이고, 이 런에서 인용하는 절감은 전부 무효다.
- `materialised_per_hit` = `(lazy + slice) / resident_hits`가 이 기능의 측정값이다.
  384번 미루고 384번 실체화하면 copy를 **옮긴** 것이지 없앤 것이 아니다.
- `nodal_const_*`는 §4가 재기만 하고 고치지 않은 것이다.

---

## 4. 로드맵 2번 — nodal consts 9,675 copy / 3.92 GB: **B0로는 지울 수 없다**

인구조사 결과:

- 올라가는 것은 `eta1, eta2, m260, m251, m253, m262, m264, diagD, diagDI`
  9종 × `ndg = nxyz·NDIR·NG` = 50,706 double = **405,648 B/copy**.
  이미 `const_generation` 게이트가 걸려 있고, 9,675 / 9 = **1,075**회만 발화한다.
- 짓는 쪽은 **host**다: `Nodal::updateConstant`(`src/Nodal.cpp:100-148`),
  입력은 `xs.xsrf` / `xs.xsdf` + `hmesh`. `updateConstantsIfMoved`가
  `xs.macroXsGeneration()`로 게이트하고, 노드별 early-out을 통과한 노드가
  하나라도 있으면 `_const_generation`이 1 오른다.
- **1,075 ≈ 1,117 = `xe_device_steps`.** 이 전송은 업로드가 아니라
  **device→host→device 왕복의 꼬리**다: Xe device step이 macro xs를 바꾸고
  (§5의 `xe commit xs` 1.66 GB D2H), host가 그 xs로 계수 9종을 다시 짓고,
  3.65 MB를 되올린다. 로드맵 2번과 3번은 같은 고리의 두 쪽이다.

왜 못 지우는가:

| 시도 | 왜 안 되는가 |
|---|---|
| device에서 계수 계산 | `src/CudaNodalConstantKernel.h`에 **이미 있다**(Rev.7.1 Task 4). 그러나 **N1이지 B0가 아니다** — CUDA의 `exp`가 glibc 2.39와 이 body가 실제로 평가하는 인자의 3.34 %(물리 대역에서 5.34 %)에서 **정확히 1 ulp** 다르다. v3 freeze가 Task 22로 미룬 그 결정이다 |
| `ByteExactMirror` 소거(E2 방식) | 게이트가 이미 "바뀌었다"고 말한 뒤에 도달하는 지점이다. 9개 배열을 memcmp해서 hit이 날 일이 거의 없고, 매 발화마다 3.65 MB를 비교하는 비용만 남는다 |
| dirty 노드만 업로드 | 원리적으로 가능하다. `updateConstant`가 노드별 dirty 비트를 이미 돌려주므로 그 집합을 모으면 된다. **그러나 그 코드는 `src/Nodal.cpp`에 있고 이 작업의 파일 범위 밖이다.** 그리고 Xe가 움직이는 것은 연료 노드 전부이므로, 흩어진 노드의 6-double 조각 수천 개가 3.65 MB 한 번보다 빠르다는 보장이 없다 — **먼저 재야 한다** |

그래서 이번에 넣은 것은 `nodal_const_uploads` / `nodal_const_bytes` 두 계수기뿐이다.
`nodal_const_uploads / 9`가 `xe_device_steps`를 따라가는지가 다음 런에서 확인된다.
따라간다면 로드맵 2번의 올바른 형태는 "consts 업로드 제거"가 아니라
**"Xe commit이 host를 거치지 않게 하는 것"**이고, 그것은 로드맵 3번·6번과 같은 일이다.

---

## 5. 로드맵 3번 — `xe commit xs` 12,287 copy / 1.66 GB: **이번에 하지 않았다**

`drainXeCommit`(`CudaXsReconBackend.cu:2545` 부근)이 Xe step마다 내려받는 것은
`11·lmp + xs_ssm + iden[I135..I135+2]` = 13 copy / 1,960,632 B이고,
12,287 = 11 × 1,117이 그 중 스칼라 11슬롯이다.

**micx와 같은 처방(dirty flag + 지연)이 원리적으로 맞지만, `_xs`의 reader 인구가
`_micx`와 급이 다르다.** 이번 인구조사에서 확인한 host 소비자만:

- `Nodal::updateConstant` — `xsrf`, `xsdf` (§4의 그 고리)
- `CudaBICGBackend.cu:4636-4696 push_pending` — `xsnf`/`xsrf`/`xssm`을 **다시 H2D**
  (0.40 GB). 역시 device→host→device 왕복이다
- `NormFactor` / 출력 — `xskf`
- `XSSet::PredictorStep` — `_xs_bos` 스냅샷이 **11슬롯 전부**를 복사한다
- rod cusping의 `_xs` 블렌드, 결과 기록

즉 "몇 슬롯만 내려받으면 된다"는 축소는 **11슬롯 중 무엇을 host가 정말 읽는가**에
대한 전수 감사를 요구하고, 틀리면 조용한 오답이다. `_micx`에서 그 감사를 하는 데
이 문서의 §2가 들었다. 같은 크기의 작업을 `_xs`에 대해 별도 WP로 한다.

**다만 §4가 방향을 바꿔 놓는다.** `_xs`의 두 큰 host 소비자(nodal consts,
CMFD push_pending)가 **둘 다 곧바로 device로 되돌아가는 값**이라면, 축소해야 할
것은 다운로드의 폭이 아니라 왕복 자체다.

---

## 6. 등급과 case key

**B0, 그리고 `trajectory::kArmEnv`에 넣지 않았다.**

지연 다운로드는 **같은 copy 목록을, 같은 device offset에서, 같은 래퍼로** 실행한다.
바뀌는 것은 그 copy가 언제 일어나는가뿐이고, 어떤 host reader도 자기가 읽는
시점에는 즉시-다운로드 arm과 **같은 바이트**를 본다. 커널은 이 플래그를 볼 수
없다 — device 블록의 내용은 두 arm에서 동일하다.

움직일 수 없는 knob을 case key에 넣으면 같은 실행 두 개가 서로 다른 캐시 항목이
된다. `tools/test_micx_resident_contract.py`의 M1이 그 부재를 지킨다.

만약 A/B가 digest 차이를 낸다면 **플래그가 버그가 아니라**, EnsureMicxHost를
건너뛴 host reader가 버그다. 그리고 그때는 §3.2의 두 경보가 먼저 말한다.

---

## 7. 238 런북

로컬 계산 금지. 출력은 `E:`. IP는 어디에도 쓰지 않는다.

### 7.1 arm

```
# BASE (v5) — PROD env, 여기에 ledger를 켠다
env -i <v5 PROD env> RASBERY_XFER_LEDGER=1 <bin> <kngr cycle-1 deck>

# MICX = BASE + RASBERY_GPU_MICX_RESIDENT=1
```

교차 순서: warm-up 1 + hot 3, BASE/MICX 번갈아. median 채택.

### 7.2 게이트 (전부 통과해야 B0)

| 게이트 | 기준 |
|---|---|
| digest | `1f36e75dc00ed2b4` |
| outers | `4377` |
| h5diff | 0 differences |
| pin CSV | `cmp` 완전 동일 |
| 18지표 | Δ = 0 |
| `host_fallbacks` | 전 서브시스템 0 |
| `[RASBERY][ERROR][micx]` | **로그에 0회** — 한 번이라도 나오면 §2의 분류가 틀린 것이고 B0 주장은 철회다 |
| `[RASBERY][FAIL][micx]` | 0회 |

### 7.3 기대치 — 무엇이 어떻게 줄어야 하는가

`[RASBERY][XFER][LEDGER]`에서:

| 지점 | BASE | MICX 기대 | 유도 |
|---|---:|---:|---|
| `flatxs micx mic` (D2H calls) | 3,456 | **612** | 68 실체화 × 9 |
| `flatxs lmpx lmp` | 3,456 | **612** | 〃 |
| `flatxs micx msm` | 384 | **0** | §2(c)가 PROD에서 0회이므로 산란은 내려오지 않는다 |
| `flatxs lmpx lsm` | 384 | **0** | 〃 |
| micx/lmpx D2H bytes | 22.85 GB | **3.33 GB** | 68 × 49.0 MB |
| `solveFlatXs:drain` | 384 회 / 1.047 s | 384 회 / **~0.2 s** | 316회가 59.5 MB를 더 기다리지 않는다 |
| `downloadFlatXsMicx:micx drain` | — | **68 sync** | 새 지점 |

`[RASBERY][MICX]`에서: `resident_hits` **384**, `slice_downloads` **68**,
`lazy_downloads` **0**, `materialised_per_hit` **≈0.18**, `mb_saved` **≈18,600**.

> **68은 유도이지 측정이 아니다.** 34 depletion step × (PredictorStep 1 +
> CorrectorStepGpu 1)이다. 실측이 이보다 크면 §2에서 "PROD 0회"로 분류한
> reader 중 하나가 실제로는 발화하고 있다는 뜻이고, 어느 것인지는
> `lazy_downloads`(전체 블록을 요구한 쪽)와 `slice_downloads`의 비로 좁힌다.
> **`resident_hits`가 384보다 작으면** rodded 노드가 있는 statepoint가 있어
> `mark_micx_resident=false`가 나온 것이다 — 그 자체는 정상이다.

wall: D2H 28 GB/s(WP13 §6의 실측)로 19.5 GB ⇒ **~0.70 s**, 11.34 s의 **6.2 %**.
`drain`의 단축분은 그 안에 포함되며 별도로 더하지 않는다(같은 바이트다).

### 7.4 계약 테스트 (로컬, 컴파일 불필요)

```
python tools/test_micx_resident_contract.py     # 8 checks, 11 negative controls
python tools/test_enum_alias_contract.py
python tools/test_dependent_template_contract.py
python tools/test_xfer_ledger_contract.py
```

네 개 모두 이 커밋에서 통과한다.

> `tools/test_xfer_elide_contract.py`는 **`914f6b3` 시점에 이미 깨져 있다**
> (12 violations / 3 dead controls). 이 커밋 이전 트리에서 재현했다. 원인은
> WP13.1이 `uploadGuarded`/`pushDeviceReadOnly`에 leaf-name 인자를 추가하면서
> 그 테스트의 앵커가 함께 갱신되지 않은 것이고, 이 작업의 파일 범위 밖이다.

---

## 8. 미룬 것 — CRAM의 `H2D mic`을 D2D로

`CudaCramBackend.cu:896/999`가 depletion step마다 4슬롯 × 5.27 MB를
**host에서** 올린다(런당 272 copy / 1.37 GB). 그 host 복사본은 바로 이
flat-XS device 블록에서 내려온 것이다 — device → host → device.

**이 커밋은 생산자 쪽만 놓았다**: `XsReconBackend::micxResidentGeneration()`와
`Impl::dev_block + off_mic[xt]`가 그 블록을 가리킨다.
소비자 쪽(`cram::PredictorView`에 device 포인터 필드 + 두 backend가 다른
스트림이므로 event 하나)은 `src/CudaCramBackend.{h,cu}`를 고쳐야 하는데,
그 두 파일은 이 작업의 파일 범위 밖이다.

지우면 **H2D 1.37 GB + (이 작업이 새로 만드는) D2H 3.33 GB 중 68 × 21 MB**가
같이 사라진다. `kSlot = {XSAF, XSFF, XS2N, XS3N}`는 `CudaCramBackend.cu:55`의
private 상수이므로, XSSet에 그 목록을 **복제하지 않는 것**이 조건이다 —
복제한 순간 두 번째 의견이 생기고, 그것은 이 트리가 반복해서 피해 온 실수다.
