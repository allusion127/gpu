# Task 10 part 5: 세그먼트가 호스트에게 "한 번 더?"를 묻지 않는다

브랜치 `codex/exact-throughput-campaign`, 기준 `feff7e7`
커밋 `f791f75`(전제 (a)/(c) + 스파이크 (7)), `d89fb1c`(WHILE),
`3369427`+`e85d0fb`(본문이 옮겨 간 계약 둘), `74be08d`(part 4 §6이 남긴 계약 둘)
측정 하드웨어: 로컬 WSL, GTX 1080 Ti(sm_61) 1장, nvcc 12.6 / gcc 13.3, driver 560.94
새 파일: `src/GpuOuterWhile.h`

---

## 0. 결론 먼저

part 4 §7은 네 단계를 남겼다. 셋을 닫고 넷째를 세웠다.

| # | part 4 §7 | 상태 |
|---|---|---|
| 1 | (b3) materialize mask를 캐시 키로 | 완료(part 4 §6) |
| 2 | (a) sweep 스테이징을 pinned로 | **완료** — §1 |
| 3 | `cudaGraphExecUpdate` + conditional node 합법성 | **측정 완료** — §2 |
| 4 | WHILE | **완료, 기본 OFF** — §3~§6 |

그리고 §7이 목록에 없던 하나를 닫았다: **캐시 미스가 nested capture를 여는 세 자리**
(part 4 §3(c)는 그중 하나만 알고 있었다).

`RASBERY_GPU_OUTER_GRAPH=1`, kngr_238 b8 S2:

```
in_body_host_syncs                  11,937 -> 1,038
sync_exit_observation               11,359 ->   460
hostfree_enqueued - hostfree_outers     21 ->     0
graph_launches 3,144   graph_iterations 8,369   iterations/launch 2.66
graph_instantiations    1   graph_warmup_misses 0
halted_outer_launches  24 ->     3
```

**wall은 로컬에서 말할 수 없다** — 교대 측정 세 쌍에서 그래프 arm이 세 번 다 빨랐지만,
같은 arm의 쌍 간 편차가 93 s다(§6). 238이 답한다. 그래서 플래그는 기본 OFF다.

---

## 1. 전제 (a) — 기록될 수 없던 스테이징 블록

`BatchCore::Slot::sweep_in` / `sweep_out`은 `std::vector<Slot>` 안의 값 배열이었고
(`src/CudaBICGBackend.cu:2562-2563` at `feff7e7`), 둘 다 outer당 한 번씩 도는
`cudaMemcpyAsync`의 끝점이다 — H2D 하나(`:4053`), D2H 하나(`:4081`).

pageable async copy는 두 가지를 하지 않는다. **비동기가 아니고**(드라이버가 자기 bounce
buffer로 staging하고 호출이 블록한다), **기록되지 않는다**(capturing stream에 올리면
거부되고 capture 전체를 데려간다). 두 번째가 WHILE의 전제였고, 첫 번째는 그것과 무관하게
hot path의 개선이다.

고침은 part 4 §3(a)가 적은 그대로다: `host_sweep_halt` 옆에 `cudaMallocHost` 블록을 하나
더 잡고(`BatchCore::host_sweep_scalars`, slot당 2레인), `sweep_in`을 그 블록을 가리키는
포인터로 만든다. **22개 사용처는 한 글자도 바뀌지 않았다** — 배열 이름은 그 모든 자리에서
이미 포인터였다.

바뀐 것 하나는 리셋이다. `acquireSlot`은 `sl = BatchCore::Slot{}`로 슬롯 전체를 리셋하고,
`Slot{}`은 두 포인터를 null로 만든다. 그래서 `bindSweepLanes(m)`이 리셋 **직후**에 서고,
리셋 감사(`batchSlotIsReset`)는 그 사실을 **믿는 대신 검사한다** — 바인딩이 없으면
`stale_tenants`가 올라간다.

### 1.1 게이트 (a) 단독

`RASBERY_pin` = feff7e7 + 이 변경만. kngr_238, 35 상태점, 12,017 outer, `h5diff -c` 644 object.

| 비교 | 결과 |
|---|---|
| S2 feature-off 동일성 (feff7e7 vs 신규) | **0** |
| S2 OFF vs ON b8 | **0** |
| S2 ON b8 x2 결정성 | **0** |
| S2 ON b8 (feff7e7 vs 신규) | **0** |
| X feature-off 동일성 | **0** |
| X OFF vs ON b8 | **0** |
| X ON b8 (feff7e7 vs 신규) | **0** |

`[TRAJECTORY]` digest `78e58de0db8b4484` / outers 12017, 9개 run 전부.
세그먼트 수신증 여섯은 part 3의 표와 **숫자 하나 다르지 않다**:
`in_body_host_syncs` 11,937 / `sync_pre_nodal` 504 / `hostfree_outers` 11,513 /
`hostfree_enqueued` 11,534 / `halted_outer_launches` 24 / `hostfree_repairs` 3.

---

## 2. 스파이크 (7) — conditional node를 가진 exec는 갱신되는가

part 4 §3(b)2가 "아직 측정되지 않았다"로 남긴 줄이다.
`tools/probe_while_body_capture.cu`에 서브프로브 (7) `exec_update_conditional`을 붙였다.

**질문의 모양이 중요하다.** "호출이 성공을 반환했는가"는 답이 아니다. 위상이 같고 body가
**다른 child graph**를 splice하는 root 두 개를 짓고, 첫 번째에서 instantiate한 exec를 두
번째로 update한 뒤 **replay한다**. `applied`는 새 child가 trip 수만큼 돌고 옛 child가 0일
때만 참이다 — 성공을 반환하고 아무것도 바꾸지 않는 update가 거부와 구별되지 않는 것이
이 줄이 존재하는 이유다.

로컬(12.6, sm_61, driver 560.94):

```
{"record":"exec_update_conditional","api_ok":true,"result":"success",
 "applied":true,"trips":7,"old_marks":0,"new_marks":7}
```

**합법이고, 실제로 갈아끼운다.** 그래서 §3의 WHILE은 update를 **쓰지 않는다** — 쓸 수
있었는데 쓰지 않은 것이고, 이유는 캐시가 더 단순하기 때문이다: 키로 instantiate를 캐시하면
"지금 update하려는 그 그래프가 stream arm이 launch했을 그래프인가"를 증명할 필요가 아예
없다.

여섯 줄 전부, 이 커밋 기준:

| 서브프로브 | 결과 |
|---|---|
| `capture_while` | true, trip 7/7, root node 2 |
| `graph_launch_in_capture` | **false** (`cudaErrorStreamCaptureUnsupported`) |
| `child_graph_node` | true, marks 7 |
| `fork_join_in_body` | true, marks 7 |
| `memcpy_in_body` | true |
| `exec_update_conditional` | **true / applied** |
| `device_launch_in_body` | 묻지 않음(로컬에서 hang, opt-in) |

---

## 3. 전제 (c) — 캐시 미스가 여는 capture는 셋이었다

part 4 §3(c)는 `launch_sweeps` 하나를 지목했다. 실제로는 셋이다:

```
src/CudaBICGBackend.cu   launch_outer   (BiCG outer graph)
src/CudaBICGBackend.cu   launch_sweeps  (sweep graph)
src/CudaXsReconBackend.cu solveNodal    (nodal drive)
```

셋 다 미스에서 `cudaStreamBeginCapture`를 **outer body가 캡처될 바로 그 스트림**에 연다.
nested capture는 거부되고, `CudaBICGBackend`의 기존 실패 경로는 거부에 대해 **캐시를
파괴하고 `use_graph=false`를 남긴 채 런의 나머지를 진행한다** — 원인에서 네 프레임 떨어진
조용한 절벽이다.

셋 다 이제 `rasbery::graphCaptureActive(stream)`을 **먼저 묻고**, 참이면 캡처 대신
**직접 enqueue한다**. 이것은 정확하다 — replay와 그것이 캡처된 enqueue는 같은 순서의 같은
커널이다 — 그리고 **센다**: `rasbery::graphWarmupMiss()` → 수신증
`graph_warmup_misses`, 게이트 0.

warm-up 규칙(세그먼트의 outer 0을 eager로 돌린다)이 그 수를 0으로 만드는 것이고,
이 가드는 warm-up이 깨졌을 때 그것이 **시체가 아니라 수신증**이 되게 하는 것이다.

---

## 4. WHILE

`RASBERY_GPU_OUTER_GRAPH=1`, 기본 OFF.

### 4.1 정지 규칙, 그리고 그것이 왜 스트림 루프의 것인가

```cuda
__global__ void k_outer_graph_cond(cudaGraphConditionalHandle h,
                                   const DeviceOuterSegmentState* seg,
                                   const std::uint32_t* halt, int slot) {
    const DeviceOuterSegmentState s = seg[slot];
    const unsigned int halted = (halt == nullptr) ? 0u : halt[slot];
    cudaGraphSetConditional(h, (s.exit == 0u && halted == 0u &&
                                s.outer_in_segment < s.budget) ? 1u : 0u);
}
```

스트림 루프는 `exit == 0`을 **관측한 뒤에만**, 그리고 `i < budget`일 때만 outer i를 돌린다.
transition 커널이 `outer_in_segment = i + 1`을 쓰므로 `outer_in_segment < budget`은
글자 그대로 `i + 1 < budget`이다.

**halt 항은 추가 규칙이 아니다.** 그것은 스트림 arm이 `budget`에 묶여 있다는 사실로부터
공짜로 얻던 것이다: 디바이스가 포기한 sweep은 halt를 올리고, 그 outer의 transition은
no-op이 되며, `outer_in_segment`가 **멈춘다**. halt 항이 없는 이 WHILE은 다른 답이 아니라
**끝나지 않는 답**이다. 항이 있으면 루프는 스트림 arm의 남은 pass들이 no-op이 되는
바로 그 지점에서 멈추고, 세그먼트 exit은 같은 `sweep_host_continued`를 찾아 같은 repair
pass를 돌린다.

**커널은 하나**다. root의 arm 노드와 body의 꼬리 노드가 같은 커널이다 — 두 질문은 세그먼트
수명의 같은 지점에서 묻는 같은 질문이고(둘 다 직전 outer의 transition 뒤), 정지 규칙을
두 번 적는 것은 다르게 적을 기회를 두 번 갖는 것이다.

### 4.2 outer 0이 밖에 있는 세 가지 이유

각각 하나만으로 충분하다.

1. **캐시가 warm이어야 한다** — §3. 미스가 body 안에서 일어나면 nested capture다.
2. **`cmfd_sweep_patch`는 세그먼트의 첫 outer를 patch하지 않는다**(part 3 §1a). outer 0은
   구조적으로 다른 body다.
3. **flux H2D는 세그먼트의 첫 outer에서만 발화한다** — 그 뒤로는 generation이 일치해
   elide된다. outer 1에서 캡처한 body에는 flux 노드가 **없고**, arm은 outer 1이 아직
   업로드를 필요로 하면 `flux_upload_live`로 **거부한다**. 소스 generation이 움직인
   업로드를 캡처하는 것이 이 기구가 절대 가질 수 없는 모양이다.

### 4.3 body는 스트림 arm의 body다

`runOneOuter`는 람다 하나다. 루프가 그것을 호출하고, capture가 그것을 기록한다.
추출은 기계적이다 — 같은 문장, 같은 순서 — 그리고 pass별 서두(exit 관측과
`hostfree_enqueued` bump, 둘 다 **루프**의 것이지 outer의 것이 아니다)는 호출부에 남았다.
계약(`tools/test_graph_splice_contract.py` 규칙 6)이 정의가 하나이고 두 호출자가 모두 그
이름을 부른다는 것을 고정한다.

### 4.4 거부, 이름으로

| 이유 | 언제 |
|---|---|
| `feature_off` | `RASBERY_GPU_OUTER_GRAPH` 미설정 |
| `unsupported_runtime` | conditional node 없는 CUDA로 빌드됨 |
| `not_hostfree` | 위 사다리가 이미 거부 (CY02의 `cusping_live` 포함) |
| `batch` | `rasberyBatchWidth() > 1` |
| `traced` | `RASBERY_OUTER_TRACE` |
| `budget_one` | b1 — eager outer 0 뒤에 캡처할 것이 없다 |
| `flux_upload_live` | §4.2(3) |
| `capture_failed` | 빌드가 거부 |

`batch`는 정책이지 누락이 아니다. **capture 창은 프로세스의 모든 allocation에 대해
배타적이고**(`GpuCaptureArbiter.h`, 그리고 그 헤더가 측정한 "스무 번 중 네 번이 1.8 s에
죽었다"), 이 창은 bucket capture가 아니라 **outer 하나만큼 넓다**. arbiter가 볼 수 있는
것들은 직렬화하지만, 배치의 존재 이유가 하나의 arena 위 M개 호스트 스레드다.

`capture_failed`는 **두 답을 가진다**. `record(body)` **이전**의 거부는 순수한 graph
plumbing이므로 호스트 상태가 움직이지 않았고 스트림 루프로 그냥 떨어진다. `record(body)`
**이후**는 하드 스톱이다: 그 시점에 body의 서른 개 enqueue 헬퍼는 호스트 호출로 이미
실행되어 CMFD 백엔드의 byte-exact upload shadow와 nodal 백엔드의 residency 선언을
움직였는데, 그 작업을 나를 예정이던 capture는 버려졌다. 스트림 arm에서 다시 돌리면
**바이트가 호스트를 떠난 적 없는 업로드를 elide한다** — 느린 답이 아니라 그럴듯한 틀린 답이다.

### 4.5 캐시, 그리고 69가 5가 된 이유

exec는 shape 키로 캐시된다 — `(budget, slot, nxyz, ng, nsurf, canonical, hostfree_full,
reigv_slot)`. `reigv_slot`이 키에 있는 것은 그것이 body가 굽는 주소 중 **arena의 것이
아닌 유일한 것**이기 때문이다: nodal 백엔드는 자기 device 블록을 첫 drive 안에서 잡고
nsurf가 바뀌면 다시 배치하므로, 기억된 슬롯은 해제된 메모리에 1/eigv를 publish한다.

`bindResidency`는 residency가 **실제로 바뀌었을 때만** 캐시를 버린다. 첫 측정이 그 조건문의
이유다: 무조건 clear로는 kngr_238이 **69번** instantiate했다 — `armOuterSegment`가 모든
SolveLoop과 ReconvergeFlux에서 도는데, 그 residency의 열두 포인터는 arena의 것이고 한 번도
움직이지 않았다. memcmp 하나가 그것을 **5**로 만든다.

---

## 5. 게이트

### 5.1 kngr_238 (35 상태점, 12,017 outer) — `h5diff -c`, 644 object

S2 = `RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1`
X = S2 + `RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1`
G = `RASBERY_GPU_OUTER_GRAPH=1`

| 비교 | 결과 |
|---|---|
| S2 feature-off 동일성 (`feff7e7` 바이너리 vs 신규) | **0** |
| S2 OFF vs ON b8 (스트림 arm) | **0** |
| S2 OFF vs **GRAPH** b8 | **0** |
| S2 GRAPH b8 x2 (결정성) | **0** |
| S2 스트림 b8 vs GRAPH b8 | **0** |
| S2 OFF vs GRAPH b1 | **0** |
| S2 OFF vs GRAPH b16 | **0** |
| S2 OFF vs GRAPH b8 + `HOSTFREE_FULL=1` | **0** |
| X feature-off 동일성 | **0** |
| X OFF vs GRAPH b8 | **0** |

diff 하네스가 실제로 민감한지는 part 4 §5.1이 이미 확인했다(같은 명령이 `geometry/hz`의
한 원소를 건드리면 `1 differing dataset`을 낸다). `h5diff -c -v`가 세는 대상은 644개다.

`[TRAJECTORY]` digest는 **11개 run 전부 `78e58de0db8b4484`, outers=12017**.

### 5.2 수신증 — 무엇이 움직였고 무엇이 움직이지 않았나

kngr_238, b8, S2. **왼쪽 열은 part 3/part 4가 기록한 값 그대로다.**

| 수신증 | 스트림 arm | **그래프 arm** |
|---|---|---|
| `segment_launches` | 3,214 | 3,214 |
| `device_outers` | 12,017 | 12,017 |
| `hostfree_segments` | 3,144 | 3,144 |
| `hostfree_outers` | 11,513 | 11,513 |
| `hostfree_enqueued` | 11,534 | **11,513** |
| `hostfree_enqueued − hostfree_outers` | 21 | **0** |
| `in_body_host_syncs` | 11,937 | **1,038** |
| `sync_exit_observation` | 11,359 | **460** |
| `sync_pre_nodal` | 504 | 504 |
| `halted_outer_launches` | 24 | **3** |
| `hostfree_repairs` | 3 | 3 |
| `graph_segments` / `graph_launches` | 0 | **3,144 / 3,144** |
| `graph_iterations` | 0 | **8,369** |
| `iterations_per_launch` | 0 | **2.66** |
| `graph_instantiations` | 0 | **1** |
| `graph_warmup_misses` | 0 | **0** |

읽는 법 넷:

* **`hostfree_enqueued − hostfree_outers` = 0.** part 3이 남긴 21개의 no-op outer가
  사라졌다. WHILE의 정의상 그렇다 — 디바이스가 매 iteration마다 정지 규칙을 평가하므로
  overrun이라는 것이 존재할 수 없다.
* **`halted_outer_launches` 24 → 3.** 같은 사실의 다른 얼굴이다. 남은 3은 디바이스가
  sweep을 포기한 세 outer이고, 그것은 overrun이 아니라 repair가 처리하는 것이다.
* **`in_body_host_syncs` 11,937 → 1,038.** 남은 1,038은 **그래프 세그먼트의 것이
  아니다**: 460(exit 관측) + 504(sync_pre_nodal) + 71(mirror drain) + 3(sweep 재발행)
  = 1,038이고, 전부 `sweep_wont_enqueue`로 host-free를 거부한 70개 세그먼트의 몫이다.
  그래프 세그먼트가 내는 in-body sync는 0이다.
* **`canonical_nodal_outers` / `flux_uploads_elided` / `device_flux_outers`가 줄어든
  것은 회귀가 아니다.** 그 셋은 호스트가 outer마다 세던 것이고, 그래프 arm에서는 호스트가
  outer 1에만 있었다. 12,041 → 3,720은 "호스트가 3,720번 지나갔다"는 뜻이고, 그것이 이
  arm의 요점이다.

다른 budget과 arm:

| run | `graph_launches` | `graph_iterations` | `iter/launch` | `instantiations` | `warmup_misses` | `enq − outers` |
|---|---|---|---|---|---|---|
| GRAPH b8 (x2, 동일) | 3,144 | 8,369 | 2.66 | 1 | 0 | 0 |
| GRAPH b16 | 2,747 | 8,595 | 3.13 | 1 | 0 | 0 |
| GRAPH b8 + FULL | 3,144 | 8,369 | 2.66 | 1 | 0 | 0 |
| GRAPH b8 + XSRECON/FLATXS | 3,144 | 8,369 | 2.66 | 1 | 0 | 0 |
| GRAPH b1 | 0 | 0 | — | 0 | 0 | 0 |

`HOSTFREE_FULL`이 그래프 arm에서 **무해해진다**는 것이 표의 세 번째 줄이다. 그 플래그가
없애던 것은 pass 꼭대기의 exit 관측인데, WHILE에는 pass가 없다. b1은 `budget_one`으로
11,946번 거부하고 그래프를 한 번도 세우지 않는다 — 의도한 결과다.

### 5.3 나머지 덱

각 덱마다 여섯 비교: feature-off 동일성, OFF vs GRAPH b1 / b8 / b16 / b8-FULL,
GRAPH b8 x2.

| 덱 | outer | 결과 |
|---|---|---|
| kngr3 (trimmed) | 662 | 여섯 전부 **0** |
| i-SMR CY01 | 200 | 여섯 전부 **0** |
| i-SMR CY02 (cusping) | 857 | 여섯 전부 **0** |

수신증(GRAPH b8):

| 덱 | `graph_segments` | `graph_iterations` | `iter/launch` | `inst` | `warmup` | `enq − outers` | 거부 |
|---|---|---|---|---|---|---|---|
| kngr3 | 370 | 268 | 0.72 | 1 | 0 | 0 | `not_hostfree`:5 |
| CY01 | 55 | 126 | 2.29 | 1 | 0 | 0 | `not_hostfree`:3 |
| CY02 | 24 | 52 | 2.17 | 1 | 0 | 0 | `not_hostfree`:159 |

CY02의 159는 host-free 사다리가 이미 낸 답이다(`sweep_wont_enqueue`:3,
`no_canonical_nodal`:156) — part 3과 part 4가 기록한 것과 **숫자 하나 다르지 않다**.
kngr3의 `iter/launch` 0.72는 이 덱이 세그먼트당 평균 1.7 outer밖에 돌지 않기 때문이고
(`negative_flux` 탈출 263회), WHILE이 0번 도는 경우가 흔하다는 뜻이다 — arm 커널이
"들어가지 마라"를 답하는 경우이고, 그것도 정확히 스트림 루프가 `break`하는 자리다.

### 5.4 배치 — 손대지 않았고, arm은 이름으로 거부한다

`--batch-mode 4`, 4덱. 다섯 비교 전부 d0..d3 **합 0**:

| 비교 | 결과 |
|---|---|
| base-OFF vs new-OFF | **0** |
| base-ON8 vs new-ON8 | **0** |
| new-OFF vs new-ON8 | **0** |
| new-OFF vs new-GRAPH8 | **0** |
| new-ON8 vs new-GRAPH8 | **0** |

그리고 그래프 arm은 **한 번도 서지 않았다**:

```
"graph_arm":1, "graph_launches":0, "graph_instantiations":0
"hostfree_refusals":{"not_stream_sweep":2818}
"graph_refusals":{"not_hostfree":2818}
```

**거부의 이름이 `batch`가 아니라 `not_hostfree`인 것이 정확하다.** 사다리가 host-free를
먼저 묻고, 배치는 거기서 `not_stream_sweep`으로 이미 떨어진다 — `batch` 버킷은 배치가
언젠가 host-free가 될 경우를 위한 **두 번째 울타리**이고, 오늘은 첫 번째가 먼저 잡는다.
어느 쪽이든 수신증이 그렇게 말한다.

`[CAPTURE_ARBITER]`의 `alloc_in_capture`는 **0**이다.

### 5.5 계약

| 항목 | 결과 |
|---|---|
| ctest | **12/12** |
| `tools/test_graph_splice_contract.py` (규칙 5 → 10) | **PASS** (spliced launch site 6, 허용된 bare launch 1) |
| `tools/test_device_outer_state_machine.py` | **PASS** |
| `tools/test_device_outer_exactness_contract.py` | **PASS** (10 불변식) |
| `tools/test_canonical_device_state_contract.py` | **PASS** |
| `tools/test_*.py` 전체 | `feff7e7`의 6 실패 → **4** (남은 넷은 이 작업이 건드리지 않는 기존 실패) |

계약에 붙은 규칙 다섯과 각각의 음성 대조:

| 규칙 | 되돌리면 |
|---|---|
| 6. body는 `runOneOuter` 하나이고 루프와 capture가 **둘 다** 그것을 부른다 | FAIL, 두 호출자 각각에 대해 |
| 7. arm은 `i == 1`에서만 서고, 세 백엔드는 capture 중 미스에 nested capture를 열지 않는다 | FAIL, 세 자리 각각에 대해 파일:줄과 함께 |
| 8. 정지 규칙은 exit / halt / budget **셋 다**를 읽고, 커널은 정확히 두 번 발사된다 | FAIL, 항 하나를 빼면 |
| 9. capture는 arbiter 창 안이고, `graphCapturePossible()`는 capture **앞**이며, arm은 배치를 이름으로 거부한다 | FAIL |
| 10. `cudaGraphAddNode`는 CUDA 13 가드를 가진다 | FAIL |

그리고 **계약 자신이 음성 대조를 통과하지 못한 자리가 넷 있었고, 넷 다 고쳤다**:
capture 가드를 파일 순서로 검사하면 두 번째 자리의 가드를 지워도 통과했고,
그래프 API 울타리를 "이 토큰이 람다 안에도 있는가"로 물으면 어디에 심어도 통과했고,
키 필드를 구조체 전체에서 찾으면 선언을 지워도 `operator==`가 언급하는 한 통과했고,
`want.halt`는 `want.halt_slot`의 접두사라 부분 문자열 검사가 짝의 반쪽만으로 만족했다.

---

## 6. wall

**이 절이 하는 일은 wall을 주장하지 않는 것이다.**

교대 측정(kngr_238, S2, b8, 스트림/그래프 번갈아 세 쌍, 각 run의 출력은 다음 run 전에
지운다 — 같은 디스크에 2 GB를 들고 재는 것이 part 4 §5.5가 겪은 일이다):

| 쌍 | 스트림 arm | 그래프 arm | 차 |
|---|---|---|---|
| 1 | 163.3 | 146.9 | −16.4 |
| 2 | 151.0 | 61.7 | −89.3 |
| 3 | 70.0 | 60.2 | −9.8 |

**그래프 arm이 세 쌍 모두에서 빠르다**, 그리고 그것이 이 표에서 읽을 수 있는 전부다.
같은 arm의 쌍 간 편차가 93 s(163.3 → 70.0)이므로 노이즈가 신호를 완전히 덮는다. 가장
조용한 쌍(3)에서 9.8 s / 14 %이고, 그것도 한 번의 관측이다.

그리고 이 세션에는 part 4에 없던 교란이 하나 더 있었다: **같은 상자에서 다른 캠페인의
`--batch-mode 4/8` 잡이 동시에 돌았다**(`nvidia-smi` 88 %, 게이트 run 하나가 226 s까지
늘어났다). 그래서 §5의 게이트 표는 유효하고 — 비트 동일성은 경합에 무관하다 — 이 표는
그렇지 않다.

**진짜 wall은 238에서 나와야 한다**(§7.4). 없앤 것이 무엇인지는 셀 수 있고(§5.2:
in-body sync 11,937 → 1,038, 호스트 enqueue 8,369회), 그것이 몇 초인지는 이 상자가
말할 수 없다.

---

## 8. 다음

1. **238 wall.** §7.4. 그 수가 `outerGraphEnabled()`의 기본값을 정한다.
2. **238 스파이크 (5)** — `device_launch_in_body`. 로컬에서는 hang이라 opt-in이고,
   sm_120 / CUDA 13에서는 답이 다를 수 있다. 다르면 nodal drive가 child graph가 아니라
   device-side launch가 될 수 있고, 그러면 body의 fork/join 한 쌍이 사라진다.
3. **`sync_pre_nodal` 504와 `sweep_wont_enqueue` 70.** 남은 in-body sync 1,038의
   대부분은 그래프 세그먼트가 아니라 **host-free를 거부한 70개 세그먼트**의 것이다.
   Wielandt warm-up이 그 거부의 이유이고(`canEnqueueDrive()`가 `_wiel_sweep >= 5`),
   그것을 줄이는 것은 이 작업이 아니라 W3의 것이다.
4. **`iterations_per_launch`를 올리는 것.** kngr3의 0.72는 WHILE이 0번 도는 세그먼트가
   흔하다는 뜻이고, 그런 세그먼트에서는 launch 하나가 순수한 오버헤드다. arm 커널이
   호스트에게 "이번엔 들어가지 않는다"를 미리 말할 방법은 없다(그것이 device state이므로)
   — 있다면 그것은 세그먼트 budget을 덱마다 고르는 쪽의 문제다.

---

## 7. 238이 확인해야 하는 것

로컬은 12.6 / sm_61이고 238은 13.0 / sm_120이다. 이 커밋들은 `cudaGraphAddNode`의
CUDA 13 시그니처를 가드하고(`src/GpuOuterWhile.h:230`), `cudaGraphInstantiate`의
3인자 형태만 쓴다 — 그래도 **conditional node의 성능은 아키텍처마다 다른 것이고,
로컬이 답할 수 없는 것이 정확히 그것이다.**

### 7.1 스파이크부터 (한 줄이 늘었다)

```bash
cd <repo>
nvcc -O3 -std=c++17 -arch=sm_120 -rdc=true \
     -o /tmp/pwbc tools/probe_while_body_capture.cu -lcudadevrt
CUDA_VISIBLE_DEVICES=0 timeout 300 /tmp/pwbc
CUDA_VISIBLE_DEVICES=0 RASBERY_PROBE_DEVICE_LAUNCH=1 timeout 300 /tmp/pwbc
```

읽어야 할 줄: `capture_while` / `child_graph_node` / `fork_join_in_body` /
`memcpy_in_body` / **`exec_update_conditional`** 이 전부 true인가, 그리고 두 번째
명령이 **끝나는가**(RC=124면 sm_120에서도 device launch는 hang이다).

### 7.2 빌드

```bash
cmake -S <repo> -B <build> -DCMAKE_BUILD_TYPE=Release \
      -DRASBERY_CUDA_ARCHITECTURES=120
cmake --build <build> -j
```

### 7.3 비트 동일성 + **진짜 wall**

세 arm을 비교한다: OFF / 스트림(ON) / 그래프(ON+GRAPH). 공통 환경:

```bash
export RASBERY_GPU=1 RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 \
       RASBERY_XE_ANDERSON=1 RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1
cd <decks>/kngr
```

```bash
# 1) OFF (기준)
unset RASBERY_GPU_OUTER RASBERY_GPU_OUTER_SEGMENT_MAX RASBERY_GPU_OUTER_GRAPH
./RASBERY --rasi kngr_238.json --raso off.h5 > off.log 2>&1

# 2) 스트림 arm, b8
export RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8
./RASBERY --rasi kngr_238.json --raso s8.h5 > s8.log 2>&1

# 3) 그래프 arm, b8 (x2, 결정성)
export RASBERY_GPU_OUTER_GRAPH=1
./RASBERY --rasi kngr_238.json --raso g8a.h5 > g8a.log 2>&1
./RASBERY --rasi kngr_238.json --raso g8b.h5 > g8b.log 2>&1

# 4) 그래프 arm, b16
RASBERY_GPU_OUTER_SEGMENT_MAX=16 ./RASBERY --rasi kngr_238.json --raso g16.h5 > g16.log 2>&1

# 5) S2 + XSRECON/FLATXS 위에서 다시 (OFF / 그래프 b8)
export RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1
unset RASBERY_GPU_OUTER RASBERY_GPU_OUTER_SEGMENT_MAX RASBERY_GPU_OUTER_GRAPH
./RASBERY --rasi kngr_238.json --raso xoff.h5 > xoff.log 2>&1
export RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8 RASBERY_GPU_OUTER_GRAPH=1
./RASBERY --rasi kngr_238.json --raso xg8.h5 > xg8.log 2>&1

# diff (644 object)
for p in "off s8" "off g8a" "g8a g8b" "s8 g8a" "off g16" "xoff xg8"; do
  set -- $p; echo -n "$1 vs $2: "; h5diff -c $1.h5 $2.h5 | grep -c "differences found"
done
```

**기대값: 여섯 줄 전부 0.** `[TRAJECTORY]` digest `78e58de0db8b4484`, outers 12017.

### 7.4 wall — 이것이 채택을 결정한다

`sync_exit_observation` 11,359개와 outer당 서른 개의 enqueue가 사라졌고, 로컬은
그것이 얼마인지 말할 수 없다(§6). 238에서 **교대로** 재고 세 쌍을 본다:

```bash
for i in 1 2 3; do
  unset RASBERY_GPU_OUTER_GRAPH; export RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8
  /usr/bin/time -f "stream $i %e" ./RASBERY --rasi kngr_238.json --raso /dev/shm/s.h5 > /dev/null 2>>wall.txt
  export RASBERY_GPU_OUTER_GRAPH=1
  /usr/bin/time -f "graph  $i %e" ./RASBERY --rasi kngr_238.json --raso /dev/shm/g.h5 > /dev/null 2>>wall.txt
done
```

현재 단일 프로덕션 wall은 **16.9 s**다. 판정 기준:

* **그래프 arm이 스트림 arm보다 빠르다** → 기본 ON 후보. 그때 바꿀 것은
  `outerGraphEnabled()`의 기본값 하나이고, 게이트 표는 그대로 다시 돌면 된다.
* **같다** → 그대로 opt-in. 이 arm이 없앤 것은 rendezvous 수이지 wall이 아니었다는
  뜻이고, 그러면 다음 후보는 W3의 idle window 측정이다(`CudaOuterGraph.cu` 본문 주석,
  "Whoever picks this up should measure that idle window on the 238 box FIRST").
* **느리다** → conditional node의 per-iteration 비용이 sm_120에서도 크다는 것이고,
  `graph_iterations`로 나눠서 그 비용을 직접 뽑을 수 있다. 그 수를 W0의 4.55 us와
  비교하는 것이 다음 결정의 입력이다.

### 7.5 읽어야 할 수신증

`[RASBERY][OUTER_GPU]`에서:

```
graph_arm=1
graph_segments == graph_launches            (세그먼트당 정확히 한 번)
graph_iterations / graph_launches           (iterations_per_launch, 로컬 2.66)
graph_instantiations                        (게이트: 한 줌. 로컬 5)
graph_warmup_misses                         (게이트: 0)
hostfree_enqueued - hostfree_outers          (게이트: 0)
in_body_host_syncs                          (로컬 11,937 -> 1,038)
graph_refusals{}                            (단일 런에서는 not_hostfree만)
```

그리고 `[RASBERY][CUDA][CAPTURE_ARBITER]`의 `alloc_in_capture` — **0이어야 한다.**
capture 창 안에서 백엔드가 할당했다는 뜻이고, warm-up 규칙이 그것을 막고 있다.

### 7.6 배치

배치에서는 arm이 이름으로 거부한다. 확인은 한 줄이다:

```bash
RASBERY_GPU_OUTER_GRAPH=1 ./RASBERY --rasi d0.json d1.json d2.json d3.json \
    --raso d0.h5 d1.h5 d2.h5 d3.h5 --batch-mode 4
grep -o 'graph_refusals:{[^}]*}' run.log   # -> {"batch":N}, graph_launches 0
```

