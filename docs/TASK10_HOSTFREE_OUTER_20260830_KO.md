# Task 10 part 3: host-free outer — 관측을 세그먼트당 1회로

브랜치 `codex/exact-throughput-campaign`, 기준 `926497d`
커밋 `984e4ca`(mirror commit-at-issue), `1535daf`(host-free outer), `<이 문서>`
측정 하드웨어: 로컬 WSL, GTX 1080 Ti(sm_61) 1장, nvcc 12.6 / gcc 13.3

---

## 0. 결론 먼저

device outer segment는 outer당 **호스트 동기화를 2회** 지불하고 있었다.

| 사이트 | 왜 있었나 |
|---|---|
| `sync_pre_nodal` | `BICGCMFD::finishDrive`가 sweep의 스칼라 블록을 pinned 메모리에서 읽는다. outer마다 정확히 1회, 모든 arm·모든 budget에서 `device_outers`와 같았다 |
| `sync_exit_observation` | pass 상단에서 exit word를 본다. 이것이 있어야 세그먼트가 **멈출 수 있다** |

part 3은 첫 번째를 **세그먼트당 1회**로 내렸고(`RASBERY_GPU_OUTER_HOSTFREE`, 기본 ON),
두 번째까지 없애는 arm을 `RASBERY_GPU_OUTER_HOSTFREE_FULL`(기본 OFF)로 넣었다.
FULL arm에서 non-exit outer의 in-body 동기화는 **0**이다.

**FULL이 기본 OFF인 이유는 의심이 아니라 실측이다.** exit word를 보지 않으면 세그먼트가
멈출 수 없으므로 매번 budget 전부를 enqueue하고, 나머지는 halt gate가 no-op으로 만든다.
kngr3 budget 8에서 실제 outer 638개에 대해 no-op outer 2322개 — bit-exact이고 느리다
(4.29 s → 7.45 s). 이 낭비를 없애는 것이 conditional WHILE이고, **FULL arm은 WHILE이
존재하기 위한 전제**다.

---

## 1. 관측을 미루려면 같이 움직여야 했던 네 가지

### (a) 다음 sweep의 고유값 — probe에서

`BICGCMFD::setls`는 device-assembly arm에서 no-op이다(`_device_assembly_pending`만
기록하고 반환). 따라서 `stageSweepIO`가 스테이징하는 것들 중 **직전 outer의 관측이
만들어냈을 값**을 나르는 것은 eigv·reigv·reigvs·errl2 넷뿐이고, 나머지(epsl2, eshift,
budget들, 배열 포인터)는 세그먼트 내내 상수다.

`cmfd_sweep_patch`(CudaBICGBackend.cu)가 H2D 직후·graph 직전에 그 넷을 device probe에서
덮어쓴다. 산술은 호스트의 것을 그대로 쓴다 — `__ddiv_rn(1.0, eigv)`,
`__ddiv_rn(1.0, eigv + sm[kEshift])`.

**세그먼트의 첫 outer에서는 patch하지 않는다.** 거기서는 호스트의 eigv가 현재 값이고,
probe는 host outer 몇 개 전의 drive를 가리키고 있을 수 있다. 두 번째 outer부터 probe는
직전 outer의 verdict이고, 그것이 정확히 `finishDrive`가 호스트에 써 넣었을 값이다.

### (b) 카운터 — device에서 합산

`iter` / `_wiel_sweep` / `_bicg_iters`는 launch마다 `io.icmfd_done`으로 전진한다.
`cmfd_sweep_verdict`가 그 합을 직접 유지한다(`CmfdSweepProbeSink::Accum`).
`_wiel_sweep`이 이빨을 가진 쪽이다 — `canEnqueueDrive()`가 이것으로 게이팅되고
`resetIteration()`이 상태점마다 0으로 만든다.

누산기는 verdict **자신의 halt 게이트 뒤에** 있다. 이미 latch된 halt 뒤로 enqueue된
launch는 아무것도 하지 않았으므로 아무것도 더하지 않는다 — 이것이 재구성이 필요로 하는
정의 그대로다.

device가 drive를 포기한 경우(sweep state 0/2)에는 **그 launch의 스칼라 블록 전체**를
누산기에 복사해 둔다. 뒤에 enqueue된 outer들이 각자의 스테이징 블록을 같은 19개 double
위에 덮어쓰기 때문에, 그 시점이 유일하게 읽을 수 있는 순간이다.
`BICGCMFD::finishDeferredDrives`가 저장된 블록에서 `finishDrive`의 꼬리를 그대로 실행하고,
runner는 그 outer에게 tail을 준다(§2).

### (c) exit를 지난 outer들 — nodal drive의 halt

CMFD body kernel은 전부 halt를 읽고 반환한다. nodal drive만 그러지 못했다 — 호스트
호출이기 때문이다. FULL arm의 커널 다섯 개가 이제 `NodalView::halt`로 같은 word를 읽는다.

**정리가 아니라 필수다.** drive는 idempotent가 아니다 — 횡방향 누설을 jnet**에서** 만들고
마지막 phase가 jnet을 **쓴다**. 게이트 없이 overrun하면 자기 출력 위에서 다시 푼다.

cusping은 device word로 게이팅할 수 없다(Task 11). 따라서 `XSSet::RodCuspingQuiescent()`가
false인 덱은 per-outer arm을 유지한다 — 이름을 대면 i-SMR CY02다.

### (d) drain이 겸하고 있던 handover — 이것이 물었다

`sync_pre_nodal`은 sweep 관측의 비용이면서 **동시에** segment stream의 updjnet를 nodal
backend stream에 대해 순서 지우는 유일한 장치였다. 관측을 위해 그것을 없애자 순서도 같이
사라졌고, drive는 그때그때 resident인 jnet을 읽었다 — 보통은 이번 outer의 것(segment
stream이 앞서 있으므로), 가끔은 직전 outer의 것.

kngr3에서 **644개 중 51개 dataset, outer 1개 차이, budget 1을 포함한 모든 budget에서
동일**. budget에 무관하다는 것이 race의 서명이지 산술 변화의 서명이 아니다.

exactness invariant 4는 처음부터 "synchronise **또는 event**"를 허용했다. 이제
`cudaEventRecord`(segment stream, updjnet 뒤) + `XsReconBackend::waitOnSegmentEvent`
(backend stream)이고, outer의 cross-stream 쌍은 event 두 개에 호스트 0이다.

---

## 2. runner의 모양

outer의 **tail**(halt가 삼킨 updjnet 재발행부터 transition까지)이 `runOuterTail` 람다가
되었다. loop가 outer마다 호출하고, host-free 세그먼트의 repair pass가 device가 포기한
그 하나의 outer를 위해 한 번 더 호출한다. 문장은 그대로, 순서도 그대로다 —
12,000분의 3짜리 경로를 두 벌로 쓰지 않기 위해서다.

```
armOuterSegment  →  hostfree 판정(§3)  →  누산기 zero, nodal halt gate 설치
for i in 0..budget-1:
    [head]  flux / updpsi / mirrors / sweep enqueue / publish reigv / updjnet
            (hostfree가 아니면 여기서 sync + finishDrive)
    runOuterTail(i, ...)   nodal(+event) / cusping / upddhat / refresh / decision / transition
if hostfree:
    누산기 D2H → sync 1회 → finishDeferredDrives
    device가 포기했으면 runOuterTail(budget, ...) 한 번 더
[exit]  psi/dhat/jnet/phis mirror → halt clear → 관측 1회
```

`d_halt`는 이제 **세그먼트를 넘겨 살아남지 않는다**. 그럴 필요가 없었다 — body kernel만
읽었고 다음 세그먼트 진입이 지웠으니까. nodal halt gate가 그것을 보이게 만들었고
(gate는 graph key라서 세그먼트마다 떼면 세그먼트당 두 번 재인스턴스화한다), 세그먼트 사이의
host outer의 nodal drive가 같은 word를 읽는다.

---

## 3. host-free 판정 사다리 (`hostfree_refusals{}`)

세그먼트 진입 시 1회. 각 항목은 **세그먼트 안에서 움직일 수 없다는 사실**을 근거로 한다.

| 거부 | 근거 |
|---|---|
| `feature_off` | `RASBERY_GPU_OUTER_HOSTFREE=0` |
| `no_hooks` | part 3 훅 4종 중 하나라도 없음 |
| `not_stream_sweep` | rendezvous arm(기본 batch) — arm 시점 속성 |
| `sweep_wont_enqueue` | `canEnqueueDrive()` false. `_wiel_sweep`은 증가만 하고 `resetIteration()`은 상태점 경계 |
| `no_canonical_nodal` | `Nodal::TryDriveGpu`의 predicate. `rod_fraction`은 bank가 움직여야 변하고 bank move는 Search phase = 세그먼트 종료 |
| `cusping_live` | 같은 사실의 한 단계 위. fractional node 없음 + carry-over set 비어 있음이면 `ApplyRodCusping`은 아무것도 쓰지 않고 false를 반환한다 |

tracer는 **거부하지 않는다**. outertrace는 읽기만 하므로 traced host-free 세그먼트는
untraced와 같은 것을 계산한다. 대신 traced run은 race의 **부재**를 증명하지 못하므로
ON == OFF 게이트는 untraced로 돌린다.

---

## 4. 게이트

### 4.1 kngr_238 (35 상태점, 12,017 outer) — h5diff -c, 644 dataset

S2 = `RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1`
X  = S2 + `RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1` (238 호스트가 도는 형태)

| 비교 | S2 | X |
|---|---|---|
| OFF(신규) vs OFF(`926497d` 바이너리) — feature-off 동일성 | **0** | **0** |
| OFF vs ON b1 | **0** | **0** |
| OFF vs ON b8 | **0** | **0** |
| ON b8 vs ON b8 (×2 결정성) | **0** | **0** |
| OFF vs ON b16 | **0** | **0** |
| OFF vs ON b8 **FULL** (in-body 동기화 0) | **0** | **0** |
| OFF vs ON b8 `HOSTFREE=0` (part 3 이전 arm) | **0** | — |
| ON b8 vs ON b8(`926497d` 바이너리) | **0** | **0** |

`[TRAJECTORY]` digest는 **11개 run 전부 `78e58de0db8b4484`, outers=12017**.

### 4.2 수신증 — 동기화가 어디로 갔는가 (kngr_238, b8, S2)

| arm | `in_body_host_syncs` | /outer | `sync_pre_nodal` | `sync_exit_observation` | `halted_outer_launches` |
|---|---|---|---|---|---|
| per-outer (`HOSTFREE=0`) | 23,432 | 1.950 | 12,017 | 11,341 | 0 |
| 기본 (관측만 지연) | 11,937 | 0.993 | **504** | 11,359 | 24 |
| `HOSTFREE_FULL=1` | 1,038 | **0.086** | 460 | 460 | 13,642 |

`sync_pre_nodal` 12,017 → 504 (−96 %). 남은 504는 host-free를 거부한 outer들이다 —
`hostfree_refusals{"sweep_wont_enqueue":70}`, 즉 상태점마다의 Wielandt warm-up.

FULL arm에서 host-free outer 11,513개는 in-body 동기화가 **0**이다. 남은 1,038은
per-outer arm으로 떨어진 세그먼트의 것이다. 대가는 `hostfree_enqueued` 25,152 vs
`hostfree_outers` 11,513 — overrun 13,639개(= `halted_outer_launches` 13,642)이고
wall 61.0 s → 82.2 s다.

`hostfree_repairs: 3` — device가 sweep을 포기한(state 0/2) outer 3개에서 repair pass가
실제로 돌았고, 그 상태로 bit-exact다. 12,000분의 3짜리 경로가 게이트에서 실행됐다.

### 4.3 나머지 덱

| 덱 | 비교 | 결과 |
|---|---|---|
| kngr3 (3 상태점, 662 outer) | base-OFF / OFF / on1 / on8 / on8×2 / on16 / on8-FULL | 전부 **0** |
| i-SMR CY01 (200 outer) | 같은 6종 | 전부 **0** |
| i-SMR CY02 (857 outer, cusping 774회) | 같은 6종 | 전부 **0** |
| 4-덱 batch(`--batch-mode 4`) | OFF vs ON b8, OFF vs base — 덱 4개 | 전부 **0** |
| ctest | | **12/12** |
| Python 계약 | 신규 실패 | **없음** (기존 실패 4종은 `926497d`에서도 동일) |

CY02는 host-free를 **거부한다**: `hostfree_refusals{"no_canonical_nodal":156,
"sweep_wont_enqueue":3}`. fractional rod가 `Nodal::TryDriveGpu`를 먼저 막고, 그것이
막히면 arm 자체가 서지 않는다. 24개 세그먼트만 host-free로 돌았고 그 구간은 cusping이
정지 상태였다 — 그리고 결과는 bit-exact다.

batch의 슬롯별 수신증은 `device_outers == host_outer_observations`(budget 1, rendezvous
arm)로 **변하지 않았다**.

### 4.4 wall (로컬, 노이즈 ±3 s — no-regression 확인용)

| run | S2 | X |
|---|---|---|
| OFF | 61.34 | 64.20 |
| OFF (`926497d`) | 61.19 | 79.96 |
| ON b8 (기본) | 61.04 / 60.67 | 109.52 / 110.10 |
| ON b8 `HOSTFREE=0` | 61.16 | — |
| ON b8 FULL | 82.17 | — |
| ON b16 | 60.95 | — |

로컬 1080 Ti는 launch-bound라 기본 arm의 이득이 노이즈에 묻힌다. **실측은 238에서
나와야 한다** — 이 arm이 없애는 것은 rendezvous 자체가 아니라 host가 보는 동안 device가
비는 창이고, 그 창은 장치가 빠를수록 상대적으로 커진다.

---

## 5. 다음: conditional WHILE

FULL arm의 유일한 비용은 overrun이고, overrun의 원인은 "세그먼트가 자기 exit을 볼 수
없다"이다. WHILE은 그 결정을 device로 옮긴다 — trip count를 device가 정하면 overrun은
존재하지 않고 in-body 동기화도 0이다.

로컬 실측(`tools/probe_conditional_graph.cu`, CUDA 12.6 / sm_61): WHILE conditional node는
합법이고 런타임이 받는 handle scope는 **body graph**다. SWITCH는 12.8 미만에서 없고
nested-IF fallback은 iteration당 8.44 us(WHILE 자체는 4.55 us). 1505 node 인스턴스화 3.24 ms
(plan 게이트 250 ms). 12,017 outer × 4.55 us = 60 s 중 55 ms — **제어 흐름은 한 번도
문제였던 적이 없다.**

남은 것은 body를 캡처 가능하게 만드는 일이고, 이 커밋이 그 전제를 놓았다:

* body 안에 host가 device 메모리를 읽는 곳이 없다(FULL arm),
* H2D node 집합이 고정이다(W3 item 4 + 이번 patch kernel),
* nodal drive가 halt로 gating되므로 캡처된 body가 exit를 지나도 안전하다.

**남은 구멍은 하나**: nodal drive는 여전히 `cudaGraphLaunch`를 호출하는 **host call**이다.
capturing stream에 대한 graph launch가 child graph node로 기록되는지를 먼저 확인해야 하고,
아니라면 nodal 파이프라인을 segment stream 위로 직접 enqueue하는 경로가 필요하다.
