# WP14 — outer segment는 왜 매 outer마다 끝나는가 (2026-08-30, KO)

브랜치 `codex/exact-throughput-campaign`, 기준 커밋 `914f6b3`.
이 문서도 WP13과 같은 형태다: **본체는 분석**이고, 구현은 그 분석이 "컴파일러 없이
확실하다"고 판정한 만큼만이다. `docs/WP_PLAN_REVIEW_AND_TRACKER_20260831_KO.md`,
`docs/README_CAMPAIGN.md`, `docs/V5_FREEZE*`, `test/reference/*`는 건드리지 않았다.

---

## 0. 요약 (읽는 순서)

1. **segment가 budget 8을 못 채우는 것은 결함이 아니라 물리다.** kngr_238은 segment당
   평균 **3.30 outer**에서 나가는데, 그 종료의 88 %는 CMFD 결정이 슬롯을 **호스트 phase**
   (압도적으로 Xenon)로 넘기기 때문이다. 디바이스 술어로는 그 지점을 넘어갈 수 없다.
   §2의 표가 그 근거다.
2. **`escapes{}`는 그 이유를 말할 수 없었다.** Xenon / ThermalHydraulics / Search 세 갈래는
   모두 `escape = None`을 달고 나가므로(`src/CmfdOuterKernel.h`), 영수증에서 가장 큰
   버킷이 `"none"` 하나였고 그것은 아무 이름도 아니었다. 이번 커밋이 넣은
   **`exit_reasons{phase:count}`** 가 그 버킷을 쪼갠다 — B0, 세그먼트당 증분 1회.
3. **`exit observation` 3.727 s는 지연이 아니라 디바이스 실행이다.** 같은 위치의
   `host-free exit`가 5배 싼 것이 증거다(§3). 따라서 event+polling으로 바꿔도 회수되지
   않는다. 회수하려면 host enqueue와 device 실행을 **겹쳐야** 하고, 그것을 하는 물건은
   이미 트리에 있다 — conditional WHILE(`RASBERY_GPU_OUTER_GRAPH`, 기본 OFF).
4. 이번에 넣은 것: **`RASBERY_GPU_OUTER_SEGMENT_V2=1`** — segment exit의 **순수 랑데부
   소거** 두 개(§4). 예상 절감은 **작다(상한 ~12 ms, wall의 0.1 %)** 이고 그 숫자를
   숨기지 않는다. 큰 것은 §6의 로드맵이다.
5. 이번에 **넣지 않은 것**: exit mirror의 lazy 화(1.7 GB). 그 1.7 GB는 wall에 있지 않다
   (§5). 그리고 안전하게 하려면 host reader flush 지점 감사가 선행되어야 한다.

---

## 1. 기준 사실 (238, v5, 단일 KNGR deck, `914f6b3`)

`[RASBERY][XFER][LEDGER]` on. 35 statepoint, outer 4,377, wall 11.34 s.

env = `RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8
RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XE=1
RASBERY_GPU_XE_TXN=1 RASBERY_GPU_CMFD_FUSE=15 RASBERY_GPU_WIEL_FOLD=chunked
RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1`

| 항목 | 값 | 출처 |
|---|---:|---|
| wall | 11.34 s | — |
| `sync_calls` / `sync_ns` | 9,546 / **5.71 s (wall의 50 %)** | LEDGER 총계 |
| `runSegment:exit observation` | 4,221 회 / **3.727 s** (0.88 ms/회) | BY_SYNC_NS |
| `runOneOuter:pre-nodal drain` | 341 회 / 0.353 s (1.03 ms/회) | BY_SYNC_NS |
| `runSegment:host-free exit` | 1,255 회 / 0.199 s (**0.16 ms/회**) | BY_SYNC_NS |
| `runSegment:exit drain` | 1,326 회 / 1.7 ms (1.3 µs/회) | BY_SYNC_NS |
| `BatchCore::drain:launch` | 359 회 / 0.079 s | BY_SYNC_NS |
| `runOuterTail:segment exit word` (d2h) | 4,385 회 | BY_CALLS |
| jnet/dhat/phis exit mirror (d2h) | 3 × 1,326 회 × 427 KB = 1.7 GB | BY_BYTES |
| `issueFluxDownloads:out_phi` (d2h) | 4,670 회 / 631 MB | BY_BYTES |

### 1.1 여기서 바로 유도되는 구조

sync 사이트의 **호출 횟수만으로** segment 루프의 형태가 확정된다.

* `exit drain`은 `runSegment`의 마지막 랑데부이고 **segment마다 정확히 1회** →
  **segment = 1,326개**.
* `host-free exit`는 host-free segment마다 1회 → **host-free segment = 1,255개 (95 %)**.
* `pre-nodal drain`은 `!hostfree`인 outer마다 1회 → non-host-free outer = 341개.
* `exit observation`은 `i > 0`인 pass마다 1회 →
  **총 pass = 4,221 + 1,326 = 5,547**.
* pass는 outer를 커밋하거나(4,377) 나가거나(discovery) 둘 중 하나 →
  **discovery-only pass = 5,547 − 4,377 = 1,170**.
* 나머지 segment는 budget으로 끝난 것 → **budget 종료 = 1,326 − 1,170 = 156**.

> **segment당 outer = 4,377 / 1,326 = 3.30** (budget 8).
> 즉 "매 outer마다 나간다"가 아니라 **"budget의 41 %에서 나간다"** 이고,
> 그 88 %(1,170/1,326)는 디바이스가 exit word를 세워서 나간 것이다.
>
> `keep_exit_obs = hostfree && !outerHostFreeFull()`이 참(PROD는
> `RASBERY_GPU_OUTER_HOSTFREE_FULL`을 세우지 않는다)이므로, host-free arm임에도
> top-of-pass 관측은 유지된다 — 그게 4,221회의 정체다.

---

## 2. 종료 사유 표

`runSegment` 종료는 전부 `deviceOuterTransition`(`src/CudaOuterGraph.h`)의 6개 랭크에서
나온다. 그 위 3개는 `DeviceOuterProbe`(sweep이 세운 신호), 4번째는 CMFD 결정
(`cmfdOuterConvergence`, `src/CmfdOuterKernel.h`), 마지막이 budget이다.

| # | 종료 사유 | 판정 위치 | `next_phase` | `escape` | 이 deck에서의 빈도 |
|---|---|---|---|---|---|
| 1 | 비유한 (NaN/Inf) | `probe.nonfinite` | `failed` | `nonfinite` | 0 (fail-closed) |
| 2 | 음수 flux | `probe.negative_flux` | `failed` | `negative_flux` | 0 |
| 3 | Rayleigh hand-back | `probe.rayleigh` (sweep state 2) | `outer` | `rayleigh_fallback` | ~0 — 트리 주석의 실측 "12,000 outer 중 3" |
| 4 | cusping / material 변경 | `probe.material_changed` | `outer` | `material_changed` | **0** — host-free arm 진입 조건 `cusping_quiescent`가 이미 배제 |
| 5a | **CMFD 결정 → Xenon** | `xe_pending && (flux_converged \|\| xe_interim \|\| stall_sample)` | `xenon` | `none` (limit-cycle이면 `flux_limit_cycle_sample`) | **지배적.** `RASBERY_GPU_XE=1` + `RASBERY_STAGED_XE_TOL=1000`이 `xe_interim`을 "느슨하게 수렴한 flux"에서 발화시킨다. 3.3 outer 주기와 일치하는 유일한 갈래 |
| 5b | CMFD 결정 → T/H | `th_pending` (완전 수렴 후) | `thermal_hydraulics` | `none` | statepoint당 T/H pass 수 이하 |
| 5c | CMFD 결정 → Search | `search_pending` (settle gate 통과 후) | `search` | `none` | boron trial당 1 이하 |
| 5d | CMFD 결정 → 수렴 | 전부 수렴 | `normalize_flux_sign` | `flux_converged` | statepoint 해당 ≥1 → ≥35 |
| 5e | CMFD 결정 → 치명적 stall | `stall_events > MAX_FLUX_STALL_EVENTS \|\| !search_pending` | `failed` | `flux_stall_fatal` | 0 기대 |
| 6 | **segment budget** | `outer_in_segment + 1 >= budget` | `outer` | `segment_budget` | **156** (= 1,326 − 1,170, §1.1) |

**표에서 읽어야 하는 것.** 1,170개의 조기 종료 중 3·4번은 사실상 0이므로 **전부 5번**,
즉 **호스트 phase로의 인계**다. 이것은 "디바이스가 스스로 판정할 수 있는데 호스트가
물어보느라 나가는" 종류가 아니다 — Xe·T/H·search는 호스트가 소유한 phase이고,
`runSegment`가 반환되어야 실행된다. 그러므로

> **과제 (a)"디바이스가 exit 술어를 스스로 평가해서 segment를 이어가게 한다"는
> 이 deck에서 회수할 수 있는 outer가 없다.** 이어갈 수 있는 종료 사유가 없다.
> budget 종료 156건만이 스케줄링 인공물이고, 그건 이미 디바이스가 판정한다.

### 2.1 그래서 `exit_reasons{}`를 넣었다

`escapes{}`만으로는 5a/5b/5c가 전부 `"none"` 한 버킷이라 위 표를 **측정으로** 확인할
수 없었다. `DeviceOuterSegmentState::next_phase`는 exit 관측이 이미 내려받는 32 B 안에
있으므로, 비용은 segment당 bounds test 1 + relaxed increment 1이다 (B0, 무조건 ON).

동시에 pass 인구조사도 넣었다: `segment_passes`, `discovery_passes`,
`outers_per_segment`. stream arm에서는
`segment_passes − device_outers == discovery_passes`가 **항등식**이므로 영수증만으로
§1.1의 유도를 재확인할 수 있다.

---

## 3. `exit observation` 0.88 ms는 지연이 아니다

과제 지시는 "0.88 ms는 안쪽 커널보다 훨씬 길므로 대부분 launch/drain 지연"이라고
보았다. 영수증은 그 반대를 말한다.

`host-free exit`(0.16 ms)와 `exit observation`(0.88 ms)은 **스트림상 완전히 같은 위치**다:
둘 다 "직전 `runOneOuter`가 반환된 직후, 그 outer의 tail이 아직 in-flight인 스트림에
거는 synchronize"다. 지연이 원인이라면 둘은 같아야 한다. 5.5배 차이는 이렇게 설명된다.

* discovery로 끝난 1,170 segment에서는 `exit observation`이 먼저 스트림을 **비웠고**,
  그 뒤의 `host-free exit`는 빈 스트림을 본다(≈0).
* 따라서 `host-free exit`의 0.199 s는 사실상 budget 종료 **156건**의 몫이다 →
  156 × ≈1.28 ms. 이는 `exit observation`의 0.88 ms와 같은 자릿수이고,
  **직전 outer의 device 실행 시간**이다.

즉 `exit observation` 3.727 s는 **호스트가 GPU를 기다린 진짜 시간**이다.
event + pinned word polling으로 바꾸면 대기 지점만 옮길 뿐 회수량은 0이다
(exit word D2H가 스트림의 마지막 항목이므로, 그것이 도착하는 시각 = synchronize가
반환하는 시각).

### 3.1 wall 분해 — 상한은 2배이고 그것은 겹치기다

| | 값 | outer당 |
|---|---:|---:|
| wall | 11.34 s | 2.59 ms |
| sync (호스트가 GPU를 기다림) | 5.71 s | 1.30 ms |
| 나머지 (호스트 enqueue + host phase) | 5.63 s | 1.29 ms |

host-free arm에서 segment 루프는 **완전히 직렬**이다: pass i는 outer i−1의 exit를
관측해야 시작하고, 관측은 outer i−1의 전 작업을 기다린다. 그동안 디바이스는
호스트가 ~30개 enqueue helper를 도는 동안 놀고 있다. **완전한 파이프라이닝의 상한은
약 2배**이고, 그것을 하는 물건이 conditional WHILE이다 — segment의 outer 1..N을
**한 번의 `cudaGraphLaunch`** 로 만들어 호스트 enqueue를 없앤다.

`RASBERY_GPU_OUTER_HOSTFREE_FULL=1`(overrun 방식)은 같은 겹치기를 노리지만 budget
전체를 무조건 enqueue하므로, 이 deck에서는 segment당 4.7개의 no-op outer를 만든다.
호스트 enqueue 비용이 outer당 ~1.29 ms이므로 그 자체가 손해다 — 트리의 실측
("4.29 s → 7.45 s")과 일치한다. **overrun은 답이 아니고, WHILE만이 답이다.**

---

## 4. 이번 커밋: `RASBERY_GPU_OUTER_SEGMENT_V2=1`

`kArmEnv`에 넣지 **않았다**. `RASBERY_GPU_XFER_ELIDE`와 같은 논거다 — 궤적을 움직일 수
없는 노브를 case key에 올리면 같은 런 두 개에 대해 evaluator 캐시가 갈라진다.

두 소거 모두 **한 가지 사실**에서 나온다: 루프가 top-of-pass에서 `break` 했다는 것은
방금 `cudaStreamSynchronize(m.stream)`이 성공 반환했고 그 뒤로 이 스트림에 아무것도
enqueue되지 않았다는 뜻이다 → **스트림은 비어 있고, pinned exit word `m.h_seg`는
마지막으로 커밋된 outer의 transition이 쓴 32 B를 그대로 들고 있다.**
이 사실을 나르는 플래그가 `observed_exit`이다.

| # | 소거 | 조건 | 이 deck에서 | 근거 |
|---|---|---|---|---|
| V2-1 | `host-free exit`의 **두 번째 `cudaStreamSynchronize` 제거**. sweep accumulator(200 B)를 빈 스트림에서 blocking `cudaMemcpy`로 읽는다 — 그 blocking copy가 곧 랑데부다 | `segment_v2 && observed_exit && batch_width <= 1` | **~1,170회** | 같은 바이트, 같은 시점, 같은 순서. 호출 하나가 둘을 대신한다 |
| V2-2 | 종료 관측의 `exit segment state` **32 B D2H 제거**. `*m.h_seg`에서 가져온다 | `segment_v2 && observed_exit && m.h_seg != nullptr` | **~1,170회** | break 이후 스트림에 실린 것은 exit mirror 4개와 4 B halt clear뿐이고, `d_segments`를 쓰는 것은 없다 |

**batch에서는 이름으로 거부한다.** blocking `cudaMemcpy`는 legacy 스트림에서 발행되어
컨텍스트의 모든 blocking 스트림과 암묵 동기화한다. 단일 실행에서는 형제가 없어 무해하고,
batch에서는 형제 deck의 sweep 전체를 기다리게 된다 — 궤적 위험이 아니라 스케줄링
위험이며, 측정 대신 거부한다.

**repair pass는 플래그를 회수한다.** `m.sweep_host_continued`가 참이면 `runOuterTail`이
outer tail 하나를 스트림에 다시 얹고 그 transition이 `d_segments`를 쓴다. 그러면 위
사실이 거짓이 되므로, 그 자리에서 `observed_exit = false`로 되돌린다(우회 논증 금지).

### 4.1 예상 절감 — 작고, 숨기지 않는다

두 소거가 놓인 지점은 §3에서 **이미 ≈0으로 측정된** 경로다(빈 스트림 위의
synchronize, 32 B pinned D2H). 그러므로

> **예상 wall 절감 = 1,170 × (synchronize 고정비 + memcpyAsync 발행비) ≈ 1,170 × ~10 µs
> ≈ 12 ms, wall의 0.1 %.**
> 확실하게 줄어드는 것은 **횟수**다: `sync_calls` −1,170 (9,546 → 8,376, −12 %),
> `d2h_calls` −1,170.

이보다 큰 B0 절감은 `CudaOuterGraph.cu` 안에 남아 있지 않다. §3이 그 이유다.

---

## 5. 넣지 않은 것들과 그 이유

* **(c) exit mirror 3종 lazy 화 (1.7 GB).** 바이트는 크지만 **wall에는 없다**:
  mirror는 `host-free exit` 뒤에 enqueue되고 그 직후의 `exit drain`이 1,326회에
  1.7 ms(1.3 µs/회)다 — 즉 copy는 호스트가 `finish_cmfd_sweep_deferred`를 도는 동안
  DMA 엔진에서 끝난다. 게다가 안전한 lazy 화는 host reader flush 지점 감사가
  선행되어야 한다: PPR(`pin_power_reconstruction.reset`), `XSSet::NormalizeFluxSign`,
  `RASBERY_OUTER_TRACE`의 jnet 해시, 그리고 `!outer_on_device` host outer의
  `wiel`/`setls`. 하나라도 빠지면 그럴듯하고 틀린 답이 나온다. **WP14-next.**
* **(d) `sweep_in`/`sweep_halt`/`sweep_out` 합치기.** 셋은 서로 다른 디바이스 배열이고
  방향도 다르다(`sweep_halt`는 `uint32` 마스크 H2D, `sweep_in`은 `scalars` H2D,
  `sweep_out`은 같은 `scalars`의 D2H). 하나의 pinned 구조체로 합쳐지지 않는다.
  staging buffer + gather kernel로 가능하지만 절감은 ~8.6 K회 × ~5 µs ≈ 40 ms이고
  커널이 하나 늘어난다 — 비용 대비 이득이 §6-1보다 두 자릿수 작다.
* **(b) event + polling.** §3에서 회수량 0으로 판정. 구현하지 않았다.
* **run-ahead(관측 1-pass 지연).** 호스트 enqueue가 outer당 ~1.29 ms이므로 segment당
  no-op outer 1개가 곧 +1.29 ms이고, segment당 절감 상한(겹치기)과 같은 자릿수다.
  게다가 exit word에 outer 서수 스탬프가 필요하다 — **N1**(호스트 결정이 읽는 값의
  출처가 바뀐다). 별도 게이트로 남긴다.

---

## 6. 로드맵 (다음 사람에게)

1. **conditional WHILE의 A/B가 이 캠페인의 2배 레버다.** `RASBERY_GPU_OUTER_GRAPH=1`.
   이미 구현되어 있고(`src/GpuOuterWhile.h`), 이 deck의 조건을 만족한다
   (host-free 95 %, `batch_width = 1`, `budget = 8 ≥ 2`, 비-trace).
   기대: `sync_exit_observation` 4,221 → ~0, `graph_segments` ≈ 1,255,
   `iterations_per_launch` ≈ 2.3, `hostfree_enqueued − hostfree_outers` = 0.
   **N1이다** — capture-replay는 "호스트가 읽지 않는다"와 다른 종류의 정확성 주장
   (body의 노드 집합이 outer 1에서 동결된다)이므로 Gate A/B를 따로 받아야 한다.
   `graph_refusals{}`가 0이 아니면 그 이유부터 읽을 것.
2. exit mirror lazy 화 (§5, flush 지점 감사 선행).
3. `issueFluxDownloads:out_phi` 4,670회 / 631 MB — segment 단위로 줄일 수 있는지.
   `finish_cmfd_sweep_deferred`가 host `Geometry::Phif`를 실제로 읽는 시점 감사 필요.

---

## 7. 238 런북

### 7.1 계측만 (플래그 없이, 이번 커밋의 새 영수증 확인)

```
# v5 PROD env 그대로 + 레저 on
RASBERY_XFER_LEDGER=1 <v5 실행>
```

읽을 곳: `[RASBERY][OUTER_GPU]` 한 줄.

| 키 | 기대값 (914f6b3 기준 유도) |
|---|---|
| `segment_launches` | 1,326 |
| `device_outers` | 4,377 |
| `segment_passes` | 5,547 |
| `discovery_passes` | 1,170 |
| `outers_per_segment` | ≈ 3.30 |
| `budget_exits` | 156 |
| `exit_reasons{}` | `xenon`이 압도적, + `normalize_flux_sign` ≥ 35, `outer` = 156(+Rayleigh 극소), `thermal_hydraulics`/`search` 소수 |
| `escapes{}` | `none`이 5a/5b/5c의 합, `segment_budget` = 156 |
| **항등식** | `segment_passes − device_outers == discovery_passes` (stream arm에서 정확히 성립) |

이 한 줄이 §2의 표를 측정으로 확정한다. `xenon`이 압도적이 **아니면** §2의 추론이
틀린 것이고, 그 경우 §6-1의 우선순위도 다시 봐야 한다.

### 7.2 V2 A/B

```
# A (기준): v5 PROD env, RASBERY_XFER_LEDGER=1
# B (arm) : 위 + RASBERY_GPU_OUTER_SEGMENT_V2=1
```

| 확인 | A | B |
|---|---|---|
| digest / outer | `1f36e75dc00ed2b4` / 4377 | **동일** (B0 게이트) |
| h5diff | 0 | 0 |
| pin CSV `cmp` | identical | identical |
| `[XFER][LEDGER] sync_calls` | 9,546 | ≈ 8,376 (−1,170) |
| `[XFER][LEDGER] d2h_calls` | (기준) | −1,170 |
| `runSegment:host-free exit` 행 | 1,255회 | ≈ 85회 (budget 종료분만) |
| `v2_exit_syncs_elided` | 0 | ≈ 1,170 |
| `v2_state_d2h_elided` | 0 | ≈ 1,170 |
| `segment_v2_arm` | 0 | 1 |
| wall | 11.34 s | 11.3 s 내외 (**차이가 측정 노이즈 안이면 그것이 예상대로다**, §4.1) |

**digest가 움직이면 즉시 되돌리고 N1으로 보고할 것.** V2는 B0 주장이고, 그 주장이
틀렸다면 §4의 "스트림이 비어 있다"는 사실 중 하나가 거짓이라는 뜻이다.

### 7.3 (권장) §6-1의 WHILE A/B — 별도 게이트

```
# C: v5 PROD env + RASBERY_XFER_LEDGER=1 + RASBERY_GPU_OUTER_GRAPH=1
```

digest·h5diff·pin CSV는 같은 기준으로 받되, **N1로 분류하고 Gate A/B를 따로 통과시킬 것.**
기대 효과는 §3.1의 2배 상한 쪽이고, WP14의 V2와는 독립이다.

---

## 8. 계약 테스트

* `tools/test_outer_segment_v2_contract.py` (신규) — 11개 검사 + 10개 음성 대조.
  지키는 것: `exit_reasons`가 escape 히스토그램과 **같은 관측**에서 올라갈 것,
  `outerExitPhaseName`이 `DevicePhase` 전체를 덮을 것, `observed_exit`가 **오직**
  관측 break에서만 세워지고 repair pass에서 **회수될** 것, 두 소거가 모두
  플래그 AND `observed_exit`를 볼 것(그리고 sync 소거는 batch를 거부할 것),
  **플래그 OFF 경로가 그대로일 것**, 플래그가 `kArmEnv`에 **없을 것**,
  그리고 이 파일에 호스트 시계가 들어오지 않을 것.
* `tools/test_device_outer_state_machine.py` — D2H 사이트 상한을 18 → 19로 올렸다.
  새 사이트는 **랑데부를 추가한 것이 아니라 같은 200 B를 짧은 길로 읽는 것**이며,
  그 이유를 그 파일의 주석 형식대로 적어 두었다.
* 함께 돌린 것: `test_enum_alias_contract` PASS, `test_dependent_template_contract`
  PASS, `test_xfer_ledger_contract` PASS(신규 사이트도 래퍼를 통과 — `src/CudaOuterGraph.cu`의
  tagged call sites 50 → 51), `test_cmfd_outer_kernels_contract` / `test_batch_outer_budget_contract`
  / `test_outer_stream_sweep_contract` PASS.

---

## 9. 부록 — 스캔이 철자를 검사하고 있었다 (WP14 후속 커밋)

`914f6b3`(WP13.1 레저)에서 `CudaBICGBackend.cu`·`CudaXsReconBackend.cu`·
`CudaOuterGraph.cu`의 모든 `cudaMemcpy*`/`cuda*Synchronize`가 `src/XferLedger.h`의
래퍼를 지나가게 되었고, 가드 헬퍼 두 개는 **leaf 태그를 첫 인자로** 얻었다.
소스 스캔으로 동작하는 계약 테스트 셋이 그 철자를 따라가지 못해, **옳은 트리에 대해
위반을 보고**하거나 **아무것도 검사하지 않게** 되었다. 셋 다 약화 없이 복구했다.

| 테스트 | 실패하던 한 줄 이유 | 고친 방식 |
|---|---|---|
| `test_device_outer_state_machine.py` | `push(dhat_dev` 를 찾는데 소스는 `push("dhat", dhat_dev` — 레저가 `push`에 leaf 태그를 첫 인자로 넣었다 | 태그를 포함한 철자로 스캔(태그 **와** 목적지 둘 다 단언). 덤으로 `pushOrSkip(phi + m` stop 앵커와 주석을 stop으로 쓰던 앵커 3개도 되살렸다 |
| `test_device_outer_exactness_contract.py` | `cudaStreamSynchronize` 문자열을 찾는데 소스는 `rasbery::xfer::streamSync(...)` — 같은 호출, 다른 철자 | `SYNC_SPELLINGS` 두 철자를 모두 받는 `has_sync`/`sync_index`/`syncs_on` 도입. `d.stream` 인자 매칭은 래퍼가 스트림을 **세 번째**로 받으므로 경계 있는 정규식으로 |
| `test_xfer_elide_contract.py` | `pushDeviceReadOnly(<dst>` / `uploadGuarded(<dst>` 로 목적지를 캡처하는데 소스는 `("tag", <dst>` — 캡처가 문자열 리터럴에 걸려 **모든 버퍼가 "가드 없음"** 으로 읽혔다. 같은 이유로 C5의 "복사를 아직 발행하는가"도 거짓 | 목적지를 **선택적 태그 접두사**를 지나 매칭(`GUARD_TAG`), raw-copy 스캔은 래퍼 철자(`xfer::memcpyAsync(scope, leaf, dst, ...)`)를 **추가**, 죽어 있던 음성 대조 3개의 앵커를 현재 철자로 갱신 |

**약화하지 않았다는 근거 — 전부 mutate-and-fail로 확인했다.**

* `test_xfer_elide_contract.py`는 자체 음성 대조 하네스를 가지고 있고, 이제
  **6 checks / 8 negative controls 전부 적용되고 전부 발화**한다(이전: 3개 dead).
* `test_device_outer_exactness_contract.py`에는 **음성 대조 5개를 새로 넣었다**
  (`CONTROLS`, `run_controls()`). 순서를 지키는 규칙 3개에 대해 그 순서를 **제거한**
  트리로 각 검사를 돌리고, 통과하면 "규칙이 순서가 아니라 철자를 검사하고 있다"고
  **실패로 보고**한다. 대조가 파일에 닿지 못하면(앵커 이동) 그것도 실패다.
  변이는 디스크가 아니라 `read`를 갈아끼워 적용하므로 `src/` 아래는 쓰지 않는다.
  하네스 자체도 메타 검증했다: 앵커를 없애면 "measuring nothing"으로 실패,
  대조를 엉뚱한 검사에 물리면 역시 실패한다.
* `test_device_outer_state_machine.py`에는 `DEAD_ANCHORS`를 넣었다. `body_of`가
  **마커나 stop을 못 찾으면 그 사실 자체가 FAIL**이 된다 — 종전에는 마커 실종은
  빈 문자열(=조용한 통과), stop 실종은 파일 끝까지(=범위가 1,400줄 넓어진 규칙)였다.
  이 커밋이 되살린 stop 앵커 3개가 정확히 그 두 번째 형태였다.
  외부 하네스로 4개 변이(dhat 가드 제거, phi 태그 변경, dhat 태그 제거,
  `runOuterTail` 개명)를 넣어 **4개 모두 잡히는 것**을 확인했다.

`914f6b3`(WP14 이전) 대비 `tools/test_*.py` 전수 비교에서 **PASS→FAIL로 바뀐 것은
하나도 없고**, FAIL→PASS가 이 세 개다.
