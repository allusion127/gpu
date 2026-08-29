# Task 10 part 4: conditional WHILE — 그 구멍은 하나가 아니었다

브랜치 `codex/exact-throughput-campaign`, 기준 `fe35621`
커밋 `72e55a7`(splice, §1-§5) + `<이 문서>`(nodal graph 캐시, §6)
측정 하드웨어: 로컬 WSL, GTX 1080 Ti(sm_61) 1장, nvcc 12.6 / gcc 13.3, driver 560.94
새 파일: `tools/probe_while_body_capture.cu`, `src/GpuGraphSplice.h`,
`tools/test_graph_splice_contract.py`

---

## 0. 결론 먼저

`docs/TASK10_HOSTFREE_OUTER_20260830_KO.md` §5는 열린 질문 하나로 끝난다 —
**capturing stream에 대한 `cudaGraphLaunch`가 child graph node로 기록되는가.**

**아니다.** 기록되지 않을 뿐 아니라, 거부가 국소적이지도 않다:

```
{"record":"graph_launch_in_capture","api_ok":false,
 "api_err":"cudaErrorStreamCaptureUnsupported","node_recorded":false,
 "body_nodes":[0,0],"replayed":false}
{"record":"error","what":"graph_launch.build ->
 cudaErrorStreamCaptureInvalidated: operation failed due to a previous error
 during capture"}
```

capture 전체가 무효가 되고, 그 전에 기록된 노드도 같이 사라진다. 그리고 §5가
"남은 구멍은 하나"라고 적은 것은 **틀렸다** — 같은 모양의 구멍이 **둘**이다.
nodal drive(`CudaXsReconBackend.cu:3365`)와 **CMFD sweep**(`CudaBICGBackend.cu:3734,
:3798`)이 모두 한 번 캡처해 두고 남은 런 내내 replay하는 graph를 launch한다.
sweep 쪽은 §5의 조사에 들어 있지 않았다.

첫 커밋이 한 일은 **두 구멍을 함께 메운 것**이고(§2), 두 번째가 그 다음에 온 전제 하나를
닫았다(§6 — 세그먼트마다 죽던 nodal graph, 그리고 그것이 데리고 있던, 아무도 세지 않던
host rendezvous). WHILE 자체는 아직 서지 않았고, 서지 못하게 하는 것이 무엇인지는 이제
추측이 아니라 목록이다(§3).

---

## 1. 측정 — `tools/probe_while_body_capture.cu`

W0의 `probe_conditional_graph.cu`는 제어 흐름을 정리했다(WHILE 합법, handle scope =
body graph, SWITCH는 12.8+, iteration당 4.55 us). 그것은 body를 **명시적 node API**로
지었다. 실제 outer body는 두 스트림에 걸친 enqueue 헬퍼 30여 개이고 **capture**로
지어질 것이므로, 이 스파이크는 capture 모양의 질문을 한다.

로컬(CUDA 12.6, sm_61, driver 560.94):

| 서브프로브 | 묻는 것 | 결과 |
|---|---|---|
| `capture_while` | root를 stream-capture하는 중에 WHILE을 지을 수 있는가 — `cudaStreamGetCaptureInfo` → `cudaGraphConditionalHandleCreate` → `cudaGraphAddNode` → `cudaStreamUpdateCaptureDependencies`, body는 **두 번째 스트림**의 `cudaStreamBeginCaptureToGraph` | **true**, trip 7/7, root node 2개 |
| `graph_launch_in_capture` | **그 구멍** | **false** — `cudaErrorStreamCaptureUnsupported`, node 증가 없음, capture 무효화 |
| `child_graph_node` | 소스 `cudaGraph_t`를 `cudaGraphAddChildGraphNode`로 capture에 splice | **true**, body 1→2 node, mark 7 = trip 7 |
| `fork_join_in_body` | conditional body 안에서 event로 두 번째 스트림에 fork/join | **true**, mark 7 = trip 7 |
| `memcpy_in_body` | conditional body 안의 pinned H2D memcpy node | **true** |
| `device_launch_in_body` | device-side `cudaGraphLaunch` | **묻지 않음** — §1.1 |

trip count를 device counter로 검증하는 것이 요점이다. 0번 도는 WHILE과 무한히 도는
WHILE은 둘 다 "합법"으로 보이고, 둘 다 Task 10에는 쓸 수 없다.

### 1.1 물어보지 않은 하나, 그리고 왜

`device_launch_in_body`(`cudaGraphInstantiateFlagDeviceLaunch` + 커널 안에서
`cudaGraphLaunch`)는 로컬에서 **거부하지 않고 멈춘다**. instantiate/upload/replay
어딘가에서 프로세스가 걸렸고 300 s `timeout`이 그것을 죽였다 — 그리고 그 뒤의
서브프로브 둘은 아예 돌지 못했다. hang은 refusal보다 나쁘다: 답 하나를 잃는 게 아니라
나머지를 전부 잃는다.

그래서 `RASBERY_PROBE_DEVICE_LAUNCH=1`일 때만 묻는다. **238(CUDA 13.0, sm_120)에서는
물어볼 값어치가 있고**, 답이 다를 수 있다 — `timeout` 아래에서, 그 run을 잃을 각오로.

### 1.2 238이 확인해야 하는 것

로컬은 12.6이고 238은 13.0이다. child graph splice와 fork/join은 12.6에서 이미
통과하므로 13.0에서 **깨질** 이유가 없지만, 다음 세 줄은 238에서 직접 나와야 한다:

```bash
nvcc -O3 -std=c++17 -arch=sm_120 -rdc=true \
     -o /tmp/pwbc tools/probe_while_body_capture.cu -lcudadevrt
CUDA_VISIBLE_DEVICES=0 timeout 300 /tmp/pwbc
CUDA_VISIBLE_DEVICES=0 RASBERY_PROBE_DEVICE_LAUNCH=1 timeout 300 /tmp/pwbc
```

읽어야 할 필드: `capture_while`, `child_graph_node`, `fork_join_in_body`,
`memcpy_in_body`가 전부 true인가. 그리고 두 번째 명령이 **끝나는가** —
`device_launch_in_body.asked=true`와 함께 `ok`가 무엇이든 나오면 그것이 답이고,
RC=124면 sm_120에서도 같은 hang이라는 것이 답이다.

---

## 2. 첫 커밋이 한 것 — `graphLaunchOrSplice`

`src/GpuGraphSplice.h` 하나와, 그것을 쓰는 두 자리다.

```
launch 자리                                          무엇이 되었나
CudaBICGBackend.cu:3734 / :3798    (sweep graph)     graphLaunchOrSplice
CudaBICGBackend.cu:3570 / :3643    (BiCG outer graph) graphLaunchOrSplice   ← §5.4
CudaXsReconBackend.cu:3365 / :3431 (nodal graph)     graphLaunchOrSplice
```

splice는 세 호출이고 **순서가 계약**이다:

```
cudaStreamGetCaptureInfo              커서가 지금 어디인가
cudaGraphAddChildGraphNode            그것에 의존하는 child
cudaStreamUpdateCaptureDependencies   커서를, 이제 child로
```

세 번째를 빠뜨리면 다음에 그 스트림에 기록되는 것이 child가 아니라 child가 의존하던
것에 의존한다 — sweep과 updjnet이 **동시에** 도는 그래프이고, 그것은 느린 답이 아니라
다른 답이다.

### 2.1 입장료 — 소스 그래프를 남긴다

`cudaGraphExec_t`는 `cudaGraph_t`로 되돌릴 수 없다. 두 캐시 모두 instantiate 다음 줄에서
소스를 destroy하고 있었고(관용적이고, 지금까지 옳았다), 이제 남긴다:

* `BatchCore::SweepGraph::src` (`CudaBICGBackend.cu`) — `destroyGraphCaches()`가 exec와
  **쌍으로** 파괴한다.
* `XsReconBackend::Impl::nodal_graph_src` — `dropNodalGraph()`와 소멸자가 쌍으로 파괴한다.

쌍이 아니면 key 변경이 한쪽만 떨어뜨리고, 그러면 splice되는 body가 stream arm이 launch하는
exec와 다른 것을 담는다. 이 기구가 절대 가질 수 없는 실패 모드가 그것이다.

### 2.2 켜지지 않은 런은 아무것도 내지 않는다

splice는 트리에서 가장 뜨거운 launch 자리에 앉는다 — outer당 sweep 하나, nodal 하나,
12,017 outer면 24,000번. `cudaStreamIsCapturing`은 싼 호출이지만 24,000번의 "싼"은
누군가 변호해야 하는 숫자이고, **capture를 한 번도 하지 않는 런은 그 질문을 받을 이유가
없다**.

그래서 프로세스 전역 플래그 `g_graph_capture_possible`가 앞에 선다. outer body capture가
한 번 올리고 **내리지 않는다**. "지금 capture가 열려 있다"가 아니라(그건 런타임이 스트림별로
답한다) "이 프로세스는 그 질문이 값어치 있는 프로세스다"이기 때문이다. 내릴 수 있는
플래그는 capture 창과 정확히 동기화되어야 하고, 그걸 틀리면 capture 안의
`cudaGraphLaunch` — 느린 답이 아니라 파괴된 capture다.

플래그가 내려가 있는 동안 이 경로는 cache-hot word의 relaxed load 하나와, 원래 있던
launch다.

---

## 3. WHILE이 아직 서지 못하는 이유 — 새로 드러난 세 전제

§5는 전제 셋을 이미 갖췄다고 적었고(body 안에 device 메모리를 읽는 host call 없음, H2D
node 집합 고정, nodal drive가 halt로 gating), 그것은 맞다. 코드를 실제로 훑어보니
**적혀 있지 않은 전제가 셋 더** 있었다. 전부 outer body를 capture 가능하게 만드는 조건이고,
전부 이 커밋 밖이다.

### (a) sweep 스테이징 블록이 pageable이다

`BatchCore::Slot::sweep_in` / `sweep_out`은 `std::vector<Slot>` 안의 값 배열이다
(`CudaBICGBackend.cu:2561-2562`). 그것을 `cudaMemcpyAsync`로 올리고 내린다:

```
CudaBICGBackend.cu:4031  H2D  sl.sweep_in   -> scalars[...]   (pageable)
CudaBICGBackend.cu:4060  D2H  scalars[...]  -> sl.sweep_out   (pageable)
CudaBICGBackend.cu:4036  H2D  &sl.eps       -> scalars[kEps]  (pageable, 휴면)
```

**pageable async copy는 capture에 기록되지 않는다.** 나머지 upload는 전부 pinned다 —
mask staging 셋, `d_slot_map`, `prepareDeviceSweeps`가 page-lock한 물리량 배열들 — 이
19개 double 블록만 아니다.

고치는 방법은 `host_sweep_halt` 옆의 `cudaMallocHost` 레인 블록으로 옮기는 것이고,
`sl.sweep_in`을 그 블록을 가리키는 포인터로 만들면 22개 사용처가 전부 그대로 컴파일된다.
**그 자체로 hot path의 pageable staging을 없애는 개선이고, 자기 몫의 bit-exactness
게이트가 필요하다.**

그리고 이것이 왜 WHILE의 전제인지가 중요하다: pinned가 되면 memcpy **node**가 되고,
node의 소스 주소는 캡처 때 baked되지만 **내용은 replay 때 읽힌다**. 세그먼트의 outer 0이
eager로 스테이징한 블록을 outer 1..N-1이 그대로 쓰게 되고, 세그먼트 안에서 움직이는 네
값(eigv/reigv/reigvs/errl2)은 이미 `cmfd_sweep_patch`가 device probe에서 덮어쓴다.
part 3이 patch 커널을 만든 이유가 여기서 두 번째로 값을 한다.

### (b) nodal graph가 세그먼트마다 죽는다 — **§6에서 고쳤다**

`g_key_materialize`(`CudaXsReconBackend.cu:3301`)는 그래프 키다. 그리고
`setCanonicalNodalSegmentMode(true)`가 mask를 0으로, 세그먼트 exit이 `Jnet|Phis`로 되돌린다
(`:3664`, `:3675-3676`). `setMaterializeMask`는 변경 시 그래프를 drop한다.

**즉 captured nodal graph는 세그먼트마다 두 번 파괴되고 다시 캡처된다.** part 3이 halt
gate에 대해 피한 바로 그 비용을, materialize mask는 아직 내고 있다.

WHILE에게 이것은 성능이 아니라 **수명** 문제다: child graph node는 `cudaGraph_t` 객체를
가리키고, 그 객체가 세그먼트 경계에서 파괴되면 이전 세그먼트에서 캡처한 outer body는
죽은 그래프를 가리킨다. 세 갈래가 있다:

1. **세그먼트마다 body를 다시 캡처한다.** capture는 싸지만 instantiate는 아니다 —
   1505 node에 3.24 ms(W0 측정)이고 kngr_238은 `segment_launches` 3,214이므로 10 s대다.
   60 s 런에서 그건 이득이 아니라 손해고, plan §12의 `graph_instantiation_wall_ms ≤ 250` 게이트를
   그대로 깬다.
2. **`cudaGraphExecUpdate`로 갱신한다.** topology가 같으면 싸다. conditional node를 가진
   exec에 대해 합법인지는 **아직 측정되지 않았다** — 다음 스파이크의 첫 줄이다.
3. **materialize mask를 drop 사유에서 빼고 캐시 키로 만든다.** mask는 두 값(0과 `Jnet|Phis`)
   밖에 갖지 않으므로 항목 두 개짜리 캐시다. 세그먼트당 재캡처가 **오늘 당장** 사라지므로,
   WHILE과 무관하게 값이 있고 따로 게이트할 수 있다.

3번이 옳았고 1번은 게이트가 막는다. **§6이 3번이다** — 그리고 세어 보니 비용은 여기서
적은 것보다 컸다: 재캡처마다 `cudaStreamSynchronize`가 딸려 있었고, 그것은 어떤
수신증에도 들어 있지 않았다.

### (c) 캐시 미스가 nested capture다

`launch_sweeps`는 miss에서 같은 스트림에 `cudaStreamBeginCapture`를 연다
(`CudaBICGBackend.cu:3766`). outer body의 capture가 열려 있는 동안 그것이 일어나면 치명적이다.

warm-up이 이것을 막는다 — 그리고 그 사실이 **WHILE의 모양을 정한다**:

> **세그먼트의 outer 0은 eager로 돌리고, body는 outer 1을 캡처한다.**

outer 0이 같은 `(nmax, depth, precision, lanes)`로 sweep 캐시를 데우고, 같은 drive로
(b)의 재캡처를 이미 치른다. outer 1에 이르면 두 백엔드 모두 warm이다. 그리고 outer 0은
어차피 달라야 한다 — `cmfd_sweep_patch`는 세그먼트의 첫 outer를 patch하지 않고
(part 3 §1a), flux H2D는 첫 outer에서만 발화한다(그 뒤로는 generation이 일치해 elide된다).
**body가 균일하려면 outer 0은 밖에 있어야 한다.**

---

## 4. 그래서 WHILE은 이렇게 생겼다 (확정된 부분)

```
armOuterSegment  →  hostfree 판정  →  graph 판정(새 사다리)
outer 0 : 지금 그대로, 스트림 위에서 eager
if graph arm:
    body capture:  cudaStreamBeginCapture(m.stream)
                   GetCaptureInfo → ConditionalHandleCreate → arm 커널
                   GetCaptureInfo → AddNode(WHILE) → UpdateCaptureDependencies
                   BeginCaptureToGraph(body_stream, body)
                       outer 1 을 통째로 -- sweep은 child node, nodal은 child node
                       k_outer_graph_cond   ← 조건
                   EndCapture(body) ; EndCapture(root) ; Instantiate
    cudaGraphLaunch(root_exec, m.stream)      ← outer 1..budget-1
else:
    지금의 for 루프
exit : 지금 그대로 (accum D2H, sync 1회, finishDeferredDrives, repair, mirrors)
```

**조건은 device state에 이미 있다.** `k_outer_transition`이 `seg.outer_in_segment`를
전진시키고 `seg.exit`을 latch하며, `seg.budget`은 `deviceOuterSegmentReset`이 쓴다. 따라서

```cuda
cudaGraphSetConditional(handle,
    (seg.exit == 0u && seg.outer_in_segment < seg.budget) ? 1u : 0u);
```

이고 — **budget이 capture key에 들어가지 않는다.** 같은 커널이 root의 arm 노드와 body의
꼬리 노드 양쪽에 쓰인다. 이것이 스트림 루프의 정지 규칙과 글자 그대로 같다는 점이
bit-exactness 논증의 전부다: 스트림 루프는 `for i in 0..budget-1`을 돌며 exit을 보면
break하고, WHILE은 outer 0을 뺀 나머지에 대해 같은 술어를 device에서 평가한다.

overrun은 존재하지 않는다. `hostfree_enqueued − hostfree_outers`가 정의상 0이 된다.

**수신증**: `graph_launches`, `graph_iterations`, `iterations/launch`,
`graph_instantiations`, `graph_warmup_miss`, 그리고 `in_body_host_syncs/outer → 0`.

**거부**: batch arm은 이름으로 거부한다(rendezvous arm에는 없앨 창이 없다). tracer는
거부한다 — part 3의 hostfree 사다리와 달리, 여기서는 tracer가 body 안에서 device 메모리를
동기화해 읽으므로 capture 자체가 불가능하다. CY02는 hostfree 사다리가 이미 거부한다.

---

## 5. 게이트 (첫 커밋)

이 커밋의 런타임 경로는 `g_graph_capture_possible`가 내려가 있는 동안
`cudaGraphLaunch(exec, stream)` 한 줄로 접히므로 동일성은 **구성상** 참이다. 표는 그것을
확인한다 — 그리고 구성상 참인 것을 확인하지 않는 습관이 part 3의 jnet race를 만들었다.

### 5.1 kngr_238 (35 상태점, 12,017 outer) — `h5diff -c`, 644 object

S2 = `RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1`
X = S2 + `RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1`

| 비교 | S2 | X |
|---|---|---|
| OFF(신규) vs OFF(`fe35621` 바이너리) — feature-off 동일성 | **0** | **0** |
| OFF vs ON b8 | **0** | **0** |
| ON b8 vs ON b8 (×2 결정성) | **0** | — |
| ON b8(신규) vs ON b8(`fe35621` 바이너리) | **0** | **0** |
| OFF vs ON b1 | **0** | — |
| OFF vs ON b16 | **0** | — |
| OFF vs ON b8 `HOSTFREE_FULL=1` | **0** | — |

`[TRAJECTORY]` digest는 **12개 run 전부 `78e58de0db8b4484`, outers=12017** — part 3이
기록한 값 그대로다.

**diff 하네스가 실제로 민감한지 확인했다**: `n_off.h5`를 복사해 `geometry/hz`의 첫 원소에
1.0을 더하면 같은 명령이 `1 differing dataset`을 낸다. 0이 나온다는 것이 "비교가
일어나지 않았다"는 뜻이 아니라는 확인이고, `h5diff -c -v`가 세는 비교 대상은 644개다.

### 5.2 수신증 — 아무것도 움직이지 않았다는 증거

part 3 §4.2의 표와 **같은 숫자여야** 하고, 같다(kngr_238, b8, S2):

| arm | `in_body_host_syncs` | `sync_pre_nodal` | `hostfree_outers` | `hostfree_enqueued` | `halted_outer_launches` | `hostfree_repairs` |
|---|---|---|---|---|---|---|
| 기본 | 11,937 | 504 | 11,513 | 11,534 | 24 | 3 |
| `HOSTFREE_FULL=1` | 1,038 | 504 | 11,513 | 25,152 | 13,642 | 3 |

`hostfree_enqueued − hostfree_outers` = 21(기본) / 13,639(FULL). **WHILE이 0으로 만들
숫자가 바로 이것이고, 이 커밋은 아직 건드리지 않았다.**

### 5.3 나머지 덱

각 덱마다 base-OFF / OFF / on1 / on8 / on8×2 / on16 / on8-FULL 여섯 비교.

| 덱 | outer | 결과 |
|---|---|---|
| kngr3 (trimmed) | 662 | 전부 **0** |
| i-SMR CY01 | 200 | 전부 **0** |
| i-SMR CY02 (cusping 774회) | 857 | 전부 **0** |
| 4-덱 batch (`--batch-mode 4`) | — | base-OFF vs OFF, base-ON8 vs ON8, OFF vs ON8 — d0..d3 전부 **0** |

거부 수신증도 part 3이 기록한 것과 같다:

```
CY02  hostfree_segments 24,  hostfree_refusals{"sweep_wont_enqueue":3,"no_canonical_nodal":156}
CY01  hostfree_segments 55,  hostfree_refusals{"sweep_wont_enqueue":3}
kngr3 hostfree_segments 370, hostfree_refusals{"sweep_wont_enqueue":5}
batch segment_launches 722 == device_outers 722 == host_outer_observations 722
```

### 5.4 계약

| 항목 | 결과 |
|---|---|
| ctest | **12/12** |
| `tools/test_graph_splice_contract.py` (신설) | **PASS** (spliced launch site 6, 허용된 bare launch 1 = nodal arena/batch 전용) |
| 계약의 음성 대조 — sweep launch 하나를 bare로 되돌리면 | **FAIL**, 파일:줄과 함께 |

계약이 실제로 잡은 것이 둘이다. (1) `launch_outer`의 graph launch 두 자리 — BiCG
**outer** 그래프이고 RESIDENT_SINGLE 세그먼트 경로에 있지 않지만, "오늘은 닿지 않는다"는
누가 이 줄을 고치지 않아도 참이 아니게 되는 종류의 전제라 함께 splice로 돌렸다.
(2) 문자열 리터럴 — `fail("cudaGraphLaunch(nodal arena)", rc)`는 launch가 아닌 줄에서
launch로 보인다. 계약이 주석과 문자열을 **둘 다** 지운 뒤에 읽는 이유다.

### 5.5 wall (로컬, base/new 교대 3쌍)

이 세션의 wall은 part 3의 것과 비교할 수 없다 — 같은 디스크에 run당 230 MB를 쓰면서
연달아 돌았다. 그래서 no-regression은 **교대 측정**으로만 물었다(kngr_238, S2, b8, 세 쌍
교대):

| 쌍 | `fe35621` | 신규 |
|---|---|---|
| 1 | 108.59 | 76.54 |
| 2 | 80.61 | 79.51 |
| 3 | 119.67 | 126.87 |

**이 표는 아무것도 말하지 않는다**, 그리고 그렇게 적는 것이 정직한 결론이다. 같은
바이너리의 쌍 간 편차가 40 s이므로 노이즈가 신호를 완전히 덮는다 — part 3이 ±3 s라고
적은 상자가 이 세션에는 없었다. 세 쌍 중 둘에서 신규가 빠르고 하나에서 느린 것도 그
말이다.

no-regression의 근거는 따라서 wall이 아니라 **구조**다: `g_graph_capture_possible`가
내려가 있는 동안 `graphLaunchOrSplice`는 relaxed atomic load 하나와 원래의
`cudaGraphLaunch(exec, stream)`이고, 그 플래그를 올리는 코드는 아직 없다. 그리고
§5.2의 수신증 12개가 part 3의 값과 전부 같다 — 그것이 "경로가 움직이지 않았다"의 실제 증거다.
**진짜 wall은 238에서 나와야 한다.**

---

## 6. 그리고 (b)를 먼저 고쳤다 — 세그먼트마다 죽던 nodal graph

§3(b)는 "captured nodal graph가 세그먼트마다 두 번 파괴되고 다시 캡처된다"를 **추론**으로
적었다. 세어 보니 추론이 아니었다. 두 번째 커밋(`72e55a7` 다음)이 그것을 센 뒤 고쳤다.

### 6.1 세는 것이 먼저였다 — `graph_captures`

nodal 수신증에는 `graph_launches`는 있고 **capture 횟수는 없었다**. 그런데 재캡처는
공짜가 아니다: `cudaStreamBeginCapture`가 idle 스트림을 요구하므로 그 앞에
`cudaStreamSynchronize(d.stream)`가 있고, 그것은 **host rendezvous**다. 그리고 그것은
세그먼트의 `in_body_host_syncs`에 **들어가지 않는다** — 그 카운터는 runner 자신의 drain만
센다.

즉 host-free 세그먼트가 `in_body_host_syncs: 0`을 보고하면서 nodal 백엔드 안에서
세그먼트당 한 번 호스트와 만나고 있었다. 재어 보면:

| 덱 | drives | `graph_captures` (이전) | segment_launches |
|---|---|---|---|
| kngr3 b8 | 662 | **379** | 375 |
| kngr_238 b8 | 12,041 | **3,282** | 3,214 |

세그먼트당 정확히 하나다.

### 6.2 원인은 키가 아니라 **키가 하나뿐이었다**는 것

`g_key_materialize`는 그래프 키로서 **옳다** — mask가 어떤 download NODE가 존재하는지를
정하므로, 아무도 보지 않을 때 캡처한 그래프를 누군가 볼 때 replay하면 stale한 Geometry
배열이다. 문제는 그 키가 **하나의 그래프에 대한 유효성 검사**였다는 점이다. mask는 두 값
사이를 세그먼트마다 왕복하므로, 유효성 검사는 매번 파괴를 뜻했다.

고침은 키를 **인덱스**로 만드는 것이다. `NodalGraphKey`(16개 필드, C++17이라
`operator==`는 손으로 쓴다)와 `std::vector<NodalGraph> nodal_graphs`가 생기고,
`solveNodal`은 동등성 검사 대신 **조회**를 한다. `setMaterializeMask`는 더 이상
`dropNodalGraph`를 부르지 않는다.

**안전 논증은 바뀌지 않았다.** 근거는 한 번도 "키가 일치했다"가 아니었다 — "이 그래프는
정확히 이 조건에서 캡처되었다"였고, 그것은 조회가 **확립**하는 것이고 동등성 검사는
근사할 뿐이었다. 키 필드 16개는 글자 그대로 옮겼다.

`dropNodalGraph()`는 이제 **모든** 항목을 파괴한다. 네 호출자 전부 topology 변경이라
캐시 전체가 무효이므로, "현재 그래프를 파괴한다"는 하나뿐이었기 때문에만 옳았다. 그리고
`nodal_graph`/`nodal_graph_src`는 캐시로의 **비소유 별칭**이 되었으므로 그것을 통해
파괴하면 double free다 — 계약이 그 철자를 금지한다(§6.4).

캐시에는 상한 8이 있다. 키 공간이 두 값보다 크다면 캐시를 통째로 버리는데, 그것이 바로
이 코드가 **이전에 하던 일**이므로 퇴화 경우가 현상보다 나빠질 수 없다.

### 6.3 결과

| 덱 / arm | `graph_captures` 이전 → 이후 | `graph_launches` |
|---|---|---|
| kngr3 b8 | 379 → **4** | 662 |
| kngr_238 b8 | 3,282 → **4** | 12,041 |
| kngr_238 b8 FULL | — → **4** | 25,659 |
| kngr_238 b1 / b16 | — → **3 / 4** | 12,020 / 12,065 |
| kngr_238 OFF | — → **2** | 12,017 |

OFF가 2인 것이 이 변경의 형태를 확인해 준다: 세그먼트가 없으면 mask가 왕복하지 않고,
따라서 고칠 것도 없었다.

**wall은 이 세션에서 여전히 말할 수 없다** — §5.5와 같은 이유로, 같은 상자에서 `b_on8`
96.1 s와 `n_on8a` 60.3 s가 나오지만 `b_off` 74.0 / `n_off` 77.7도 같이 나온다. 없앤 것은
세그먼트당 host rendezvous 하나와 instantiate 하나이고, **그것이 얼마인지는 238이 말해야
한다.**

### 6.4 게이트 (두 번째 커밋)

§5.1의 열 비교, §5.3의 세 덱 열여덟 비교, batch 세 비교 — **전부 0**.
`[TRAJECTORY]` digest는 12개 run 전부 `78e58de0db8b4484` / outers 12017.
세그먼트 수신증 여섯 개(§5.2)는 **숫자 하나 다르지 않다**. ctest 12/12.

계약에 규칙 넷이 붙었고 각각 음성 대조를 통과한다:

* `nodal_graph` / `nodal_graph_src`를 통한 파괴 금지(별칭이므로 double free),
* `dropNodalGraph`는 `e.exec`/`e.src`를 돌며 `nodal_graphs.clear()`,
* `setMaterializeMask`는 그래프를 drop하지 않는다 — 되돌리면 FAIL,
* `NodalGraphKey`는 materialize mask를 담는다, 캐시에는 상한이 있다.

---

## 7. 다음

순서가 정해져 있고, 각 단계가 자기 게이트를 가진다.

1. ~~**(b3)** materialize mask를 drop 사유에서 캐시 키로.~~ **완료** — §6.
2. **(a)** sweep 스테이징 블록(`Slot::sweep_in`/`sweep_out`)을 pinned로. hot path의
   pageable async copy 제거이자 WHILE의 전제.
   게이트: kngr_238 OFF/ON b8 S2+X bit-identical, ctest.
3. **`cudaGraphExecUpdate` + conditional node** 합법성 스파이크 (한 줄, 위 probe에 추가).
   §6이 (b)를 닫았으므로 child graph 객체는 이제 세그먼트를 넘어 살고, 이 질문은
   "body를 세그먼트마다 다시 만들어야 하는가"에서 "만들지 않아도 되는가"로 약해졌다 —
   그래도 물어야 한다.
4. **WHILE** — §4. 게이트: part 3의 표 전부 + `hostfree_enqueued − hostfree_outers → 0`,
   `graph_instantiations` warm-up 후 0, batch 거부 수신증.

### 238이 이 커밋에 대해 확인해야 하는 것

```bash
# 1) probe, sm_120 / CUDA 13.0
nvcc -O3 -std=c++17 -arch=sm_120 -rdc=true \
     -o /tmp/pwbc tools/probe_while_body_capture.cu -lcudadevrt
CUDA_VISIBLE_DEVICES=0 timeout 300 /tmp/pwbc
CUDA_VISIBLE_DEVICES=0 RASBERY_PROBE_DEVICE_LAUNCH=1 timeout 300 /tmp/pwbc

# 2) bit-exactness + 진짜 wall (로컬 상자는 이 세션 노이즈가 ±25 s였다)
#    kngr_238, S2 = NODAL+NODAL_FULL, X = S2 + XSRECON+FLATXS
#    OFF(신규) vs OFF(fe35621), OFF vs ON b8, ON b8 x2, ON b8(신규) vs ON b8(fe35621)
```

읽어야 할 수신증: `[OUTER_GPU]`의 `in_body_host_syncs`, `sync_pre_nodal`,
`hostfree_outers`, `hostfree_enqueued`, `halted_outer_launches`, `hostfree_repairs` —
§5.2의 여섯 값과 **같아야 한다**. 다르면 그것은 이 커밋이 아니라 상자의 차이이고, 어느
쪽인지는 같은 상자의 `fe35621` 실행이 답한다.
