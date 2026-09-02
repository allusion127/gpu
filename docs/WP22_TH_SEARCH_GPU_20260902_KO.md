# WP22 — T/H 갱신과 임계 붕산 탐색의 디바이스 이식

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `XSSet::UpdateTH` / `XSSet::SolveTH` (0.70 s/statepoint) · `XSSet::SetBoron`의 노드별 쓰기 |
| 플래그 | `RASBERY_GPU_TH=1`(기본 off) · `RASBERY_GPU_SEARCH=1`(기본 off) · `RASBERY_TH_FORMS=0x…`(수축 마스크 override) |
| 게이트 등급 | T/H = **N1**(B0가 목표, 아직 미측정) · 붕산 적용 = **B0**(구성상) |
| 계약 테스트 | `tools/test_th_gpu_contract.py`(16 규칙) · `tools/test_search_gpu_contract.py`(13 규칙), 각 규칙마다 negative control |
| 동반 테스트 | `test_enum_alias_contract` · `test_dependent_template_contract` · `test_xfer_ledger_contract` · `test_gpu_full_fail_closed` · `test_cram_gpu_contract` · `test_statepoint_telemetry` — 전부 PASS |
| 신규 소스 | `src/ThKernel.h` · `ThReference.{h,cpp}` · `ThFormMine.h` · `ThFormMask.h` · `ThFormMiner.cpp` · `ThGpuReceipt.h` · `CudaThBackend.{h,cu}` · `CudaThBackendStub.cpp` · `SearchKernel.h` · `SearchGpuReceipt.h` |
| 수정 소스 | `src/XSSet.{h,cpp}` · `src/Driver.h` · `src/GpuFullContract.h` · `src/CudaXsReconBackend.{h,cu}` · `CudaXsReconBackendStub.cpp` · `CMakeLists.txt` · `tools/check_cuda_syntax.py` |
| 기준 덱 | KNGR `kngr_238.json`, `nxyz = 8,451`, `NG = 2` |
| 기준 digest | `1f36e75dc00ed2b4` / `4377` outers (플래그 off) |

> **이 문서가 주장하지 않는 것부터.** 두 아름 중 **시간을 옮기는 것은 T/H 하나뿐**이다.
> 계획 검토서가 붕산 탐색의 호스트 몫으로 적은 0.3–0.5 s는 secant 산술이 아니라
> `XSSet::BuildFlatXsStream`(≈1.0 s의 flatxs 호스트 준비)에 들어 있고, 이번 탐색 아름은
> **그것을 옮기지 않았다.** 탐색 아름이 실제로 없애는 것은 시행(trial)마다 발생하던
> `bppm` 한 배열의 H2D(=`nxyz × 8 B`)이며, 그것은 바이트 인구조사에서는 실재하지만
> 단일 덱 벽시계에서는 잡음 안이다. §7이 그 수치를 그대로 적는다.

---

## 1. T/H 갱신은 네 단계이고, 그중 둘만 병렬이다

`XSSet::UpdateTH`(src/XSSet.cpp)는 다음 네 단계다.

| 단계 | 내용 | 매핑 | 이유 |
|---|---|---|---|
| ① 노드 출력 | `power_density += xskf[ig][l] · Phif[l][ig]`, `× vol(l)` | **노드당 1레인** | 노드 간 의존 없음 |
| ② 직렬 폴드 | `total_power += node_power[lk]` (오름차순) → 부호 반전 → `total_raw_power` 재폴드 → `norm` → `total_area` | **1레인, 오름차순 그대로** | 부동소수 덧셈은 결합적이지 않다 |
| ③ 채널 스윕 | 입구 엔탈피를 축방향으로 운반, 매 노드에서 물성표 조회, `tmod`/`dmod`/`tful` 기록 | **채널당 1레인, 축방향은 직렬** | `h_cur(k)`가 `h_cur(k-1)`에 의존 |
| ④ 완화 + 지표 | `(1-w)·old + w·new` 3배열, `delta_Dop = max` | **노드당 1레인 + max 트리** | max는 결합적·정확 |

### 1.1 ②를 트리 리덕션으로 바꾸지 않은 이유

`norm = actual_power / total_raw_power`이고, 이 `norm`은 ③에서 **모든 노드의 출력에
곱해진다**. 즉 폴드 순서가 만드는 마지막 비트 차이는 스칼라 하나에 머무르지 않고 **노심
전체에 적용된다.** 블록 리덕션은 다른 double을 돌려주므로 이 아름의 B0 목표와 양립할 수
없다. `nxyz = 8,451`의 직렬 덧셈은 한 레인에서 마이크로초 단위이고, 0.70 s를 옮기는
길에서 그 값은 지불할 가치가 있다. `total_area`(중간면 nxy 폴드)도 같은 논증의 한 단계
아래다.

### 1.2 ③이 채널당 1레인인 이유

축방향은 **정의상 직렬**이다(엔탈피 운반). 그래서 병렬화 대상은 반경 방향 인덱스이고,
그것은 호스트 루프의 **바깥 인덱스가 이미 그것이었다**. WP21의 "노드 최내곽" 지침은
여기서 채널 최내곽으로 읽힌다: 한 warp의 32레인이 32개의 인접 반경 위치를 잡고, 각
레인이 같은 `k`를 동시에 읽으므로 `node_power[l + k·nxy]` 접근은 연속이다.

---

## 2. 수축 마스크 — 왜 상수 하나로는 안 되는가

`ThKernel.h`는 g++와 nvcc가 **같은 텍스트**로 컴파일하는 순수 바디다. 그 안의 모든
multiply-add는 형태를 **마스크에서 읽는다**. 사이트는 9비트/8개다.

| 비트 | 이름 | 식 |
|---|---|---|
| 0 | `TH_LERP_X0` | `z0 = z00 + fx·(z01 - z00)` |
| 1 | `TH_LERP_X1` | `z1 = z10 + fx·(z11 - z10)` |
| 2 | `TH_LERP_Y` | `z0 + fy·(z1 - z0)` |
| 3 | `TH_POWER_ACC` | `power_density += xskf·phif` |
| 4 | `TH_TFUEL` | `tmod + scale·GetTfuel(bu, lpd)` |
| 5–6 | `TH_RELAX` (2비트) | `(1-w)·old + w·new` — **세 형태**: 미융합 / `w·new` 융합 / `(1-w)·old` 융합 |
| 7 | `TH_TFUEL_LINEAR` | `rise · safe_lpd / first_lpd` |
| 8 | `TH_HAVG` | `0.5 · (h_cur + h_out)` |

`TH_RELAX`가 1비트가 아니라 **2비트**인 이유: 이 식은 곱 두 개의 합이므로 gcc는 둘 중
어느 쪽이든 융합할 수 있고, 세 번째 형태가 존재한다. 1비트 사이트였다면 호스트가 실제로
돌린 형태를 **표현할 수 없어** 채굴이 제거 불가능한 잔차를 보고했을 것이다.

### 2.1 실측값

이 저작 호스트(WSL2 / g++ 13.3)에서 `src/ThFormMiner.cpp`의 채굴을 그대로 돌린 결과:

| 빌드 플래그 | 채굴 마스크 | 잔차 | 해석 |
|---|---|---|---|
| `-O3 -march=native` | **`0x54`** | 0 (4개 시드 전부) | `LERP_Y`·`TFUEL` 융합, `POWER_ACC` 미융합, `RELAX`는 `(1-w)·old` 융합 |
| `-O3` (march 없음) | **`0x0`** | 0 (4개 시드 전부) | ISA에 FMA가 없으면 gcc는 **아무것도 수축하지 않는다** |

`TH_FORMS_DEFAULT = 0x54`는 그래서 **런이 의존하는 값이 아니라 수신증이 비교하는
대상**이다. 프로덕션 바이너리는 시작 시 자기 자신을 채굴한다 —
`CmfdOuterFormMiner.cpp`가 존재하는 것과 정확히 같은 이유이고, 그 문서가 기록하듯
베이크된 상수 하나는 그것이 측정된 기계의 기록일 뿐이다.

### 2.2 fixture가 사이트를 고정하지 못하고 있었다 — 세 건

첫 판본의 `thref::buildFixture`는 세 사이트를 **측정상 don't-care**로 남겼다. 사이트에
도달하지 못하는 fixture로 채굴한 마스크는 그 사이트를 고정하지 않는다.

1. **압력 축이 15.5를 정확히 포함**했다 → `fx = 0` → 두 x-lerp가 항등. 프로덕션
   `include/Database/mod_t.csv`도 15.5 부근 0.05 MPa 격자여서 **같은 방식으로 축퇴**돼
   있는데, 그렇기 때문에 fixture는 그러면 안 된다. 축을 13–17 MPa 5격자로 바꿔 15.5가
   구간 내부에 오게 했다.
2. **모든 노드의 선출력이 tf 표의 마지막 격자를 넘어갔다** → 클램프 → `fx = 1` → 그
   lerp도 항등이고, `GetTfuel`의 50 W/cm 미만 선형 연장은 **한 번도 도달되지 않았다**
   (주석은 도달한다고 적혀 있었다). 표를 900 W/cm까지 늘리고 노드의 12 %를 저출력으로
   두었다.
3. **`fuel_temp_rise_scale = 1.0`** → `tmod + 1.0·rise`는 어느 형태든 정확 → 사이트가
   don't-care. 프로덕션 기본값도 1.0이라 거기서는 진짜 don't-care지만, 배수를 지정하는
   덱은 아무도 고정하지 않은 사이트를 돌게 된다. fixture는 1.03을 쓴다.

`TH_HAVG`는 여전히 don't-care이고 앞으로도 그렇다 — `0.5·(a+b)`는 이진 부동소수에서
어느 철자든 정확하다. 이것은 산술의 성질이지 fixture의 결함이 아니며,
`mineStable()`이 마스크 일치가 아니라 **잔차**로 건전성을 보고하는 이유가 바로 이것이다.

---

## 3. 조각별 정확도 등급

| 조각 | 등급 | 이유 |
|---|---|---|
| 노드 출력 폴드 ① | **B0 목표** | `+ - × ÷`만. `TH_POWER_ACC` 1사이트, 마스크로 고정 |
| 직렬 폴드 ② | **B0 목표** | 1레인, 호스트와 같은 순서. 산술 자유도 없음 |
| 물성표 보간 | **B0 목표** | `milk::Table::Get` 전사. 이분 탐색은 정수 연산, clamp는 `std::clamp` 의미 그대로(NaN 포함) |
| `GetTfuel` | **B0 목표** | 표 조회 + 50 W/cm 선형 연장. 사이트 2개, 마스크로 고정 |
| 채널 스윕 ③ | **B0 목표** | 위 셋의 합성. 레인 간 통신 없음 |
| 완화 + `delta_Dop` ④ | **B0 목표** | 완화는 마스크 3형태, max는 결합적·정확 |
| 표 이탈 경고 (n_over/worst) | **B0** | 채널 순 오름차순 재주사로 호스트의 first-wins 타이브레이크를 그대로 재생 |
| 붕산 적용 (탐색 아름) | **B0 (구성상)** | 산술이 없다. 모든 노드가 같은 double을 받는 **저장** |

### 3.1 그런데 왜 문서 상단 등급은 N1인가

**도달 가능(reachable)은 측정(measured)이 아니다.** 위 표는 전부 "B0 목표"이고, 238이
`RASBERY_GPU_TH=1`로 `1f36e75dc00ed2b4` / `4377`을 재현하기 전까지 인용 가능한 등급은
**N1**이며 게이트는 Gate A / Gate B다. `src/ThGpuReceipt.h`의 `policy_note`가 게이트
스크립트가 읽는 형태로 같은 문장을 싣는다.

그리고 **채굴이 충분하지 않을 수 있는 이름 있는 이유**가 하나 있다 — WP7-C가 값을
치른 그것이다. 마스크는 `src/ThReference.cpp`, 즉 **인용문**에 대해 교정되는데
**인용문은 호출 지점이 아니다.** 프로덕션 철자는 `XSSet::SolveTH` 안에 있고 거기서는
`milk::Table::Get`과 `XSSet::GetTfuel`이 훨씬 큰 함수로 인라인되며, gcc의 수축 결정은
**그 문맥에서** 내려진다. 이것은 가설이 아니라 이 브랜치에서 측정된다: 같은 피연산자를
인용문의 **out-of-line** `refGetTfuel`과 **인라인된** `refChannelSweep`에 각각 물리면
`TH_LERP_X0` 비트가 다르게 고정된다(20k 스윕에서 17 대 0). `refChannelSweep`을 기준으로
쓴 것은 그 인라인 형상이 `SolveTH`의 것이기 때문이고, 238이 건전한 마스크로도 digest
이동을 측정한다면 **가장 먼저 볼 곳이 이 틈**이다(`src/XeFormAudit.h`가 Anderson 대수에
대해 기술하는 것과 같은 틈).

---

## 4. 탐색 아름 — 무엇이 옮겨졌고 무엇이 남았나

붕산 시행 하나는 세 가지다.

1. **secant / bracket** — 스칼라 둘 in, 스칼라 하나 out(`Scheduler.h`). **호스트에 남는다.**
   solve가 이미 호스트에 게시한 `k_eff`에 대한 O(1) 산술이므로 옮기면 나노초를 벌고
   커널 런치·스칼라 다운로드·bracket 로직의 두 번째 거처를 얻는다. 이 아름이 여기서
   지는 의무는 **부정형 성질** — propose 단계가 아무것도 전송하지 않는다 — 이고,
   `tools/test_search_gpu_contract.py`가 소스에 대해 그것을 잡아 둔다.
2. **노드별 붕산 쓰기** — `for l: bppm[l] = x`. **이것이 옮겨진다.** 같은 브로드캐스트를
   디바이스 커널이 flatxs 백엔드의 **상주** per-node 블록(`dev_pernode + 2·nxyz`)에 쓰고,
   그 백엔드 자신의 shadow(`mir_bppm`)를 커밋한다. 다음 `solveFlatXs`의
   `uploadGuarded("bppm", …)`가 그 복사를 건너뛴다.
3. **재구성 자체** — 이미 디바이스(`RASBERY_GPU_FLATXS`). 이 아름은 손대지 않는다.

### 4.1 호스트 미러는 계속 쓰인다 (그리고 그게 핵심이다)

`XSSet::BuildFlatXsStream`은 호스트에서 분기 스트림을 해결하며 노드마다 `_g.bppm(l)`을
읽어 붕산 좌표를 만든다. 즉 호스트 배열은 **바로 다음 `UpdateFlatXS`의 살아 있는
입력**이지 디바이스가 무의미하게 만든 사본이 아니다. 이 아름이 제거하는 것은 **전송**이지
배열이 아니다. 두 철자가 어긋나지 않도록 커널과 호스트 루프는 **같은 바디**
(`rasbery::search::searchBoronBroadcastNode`)를 통해 쓴다.

### 4.2 이 아름은 `RASBERY_GPU_XFER_ELIDE`와 합성되며, 그것을 감추지 않는다

`uploadGuarded`는 elide 플래그가 켜져 있을 때만 shadow를 참조한다. 꺼져 있으면 다음
`solveFlatXs`가 호스트 배열을 커널이 쓴 것 위에 올려 쓰므로 아름은 런치 하나를 쓰고
아무것도 아끼지 못한다. 그래서 `applyBoronDevice`는 그 경우 **이유를 붙여 decline**한다.
수신증이 `device_applies`를 `bytes_elided:0`과 나란히 찍어 독자가 앞의 숫자를 절감으로
오독하게 두는 것보다, fallback 하나와 문장 하나가 정직하다.

---

## 5. 예상 절감

단일 덱 KNGR v6 = 9.75 s, 그중 호스트 몫 ≈ 5 s.

| 항목 | 예상 | 근거 |
|---|---|---|
| T/H 호스트 산술 제거 | **−0.70 s** | 원장/텔레메트리의 `loop_wall.th_update` |
| T/H가 새로 지는 전송 | **+0.03 s 내외** | 갱신 1회당 H2D `xskf` 135 kB + `phif` 135 kB + 상태 3배열 203 kB + `burn` 34 kB = **507 kB**, D2H 상태 3배열 **203 kB** → 버스 **≈ 0.71 MB/갱신**(그 외에 디바이스 내부 D2D 스냅샷 203 kB) |
| T/H 커널 시간 | **+? (미측정)** | 채널 레인 8,451/nz 개, 표 조회가 지배. 238에서 `wall_ms`로 읽는다 |
| **T/H 순증** | **≈ −0.6 s (−6 %)** | 위 셋의 합. 측정 전 추정치임을 명시 |
| 붕산 H2D 제거 | **−(시행 수 × 67.6 kB)** | `nxyz·8 B = 67.6 kB`/시행. 시간으로는 밀리초 단위 |
| **탐색 순증(시간)** | **≈ 0** | 정직하게: 잡음 안이다 |
| 아직 남은 것 | **≈ 1.0 s** | `BuildFlatXsStream` — 계획 검토서가 탐색 몫으로 적은 0.3–0.5 s가 실제로 있는 곳. 다음 레버 |

> **왜 탐색 아름을 그래도 넣었는가.** 두 가지다. (a) 시행마다의 `bppm` 상행은 바이트
> 인구조사에서 실재하고, WP13의 원장이 그것을 세고 있었다. (b) propose 단계가 아무것도
> 전송하지 않는다는 **부정형 성질**은 지금 계약으로 못 박아 두지 않으면, 나중에 누군가
> secant에 디바이스 스칼라 조회를 넣는 순간 조용히 깨진다.

---

## 6. 남은 왕복 — 함의가 아니라 이름으로

| 왕복 | 크기/갱신 | 왜 아직 있는가 | 누가 없앨 수 있나 |
|---|---|---|---|
| `xskf` 상행 | `NG·nxyz·8 B` ≈ 67.6 kB | 정준 소유자가 live 거시 블록을 디바이스 주소로 **export하지 않는다** | `GpuCanonicalState.h`의 `LiveXs` 영역 + `RASBERY_GPU_SHARED_STATE` |
| `phif` 상행 | 동상 | 동상 (`Flux` 영역) | 동상 |
| `tful`/`tmod`/`dmod` 하행 | `3·nxyz·8 B` ≈ 203 kB | 소비자 `BuildFlatXsStream`이 **아직 호스트에 있다** | WP13 elision / flatxs 준비의 디바이스 이식 |

`ThBackend`의 `UpdateView`는 `xskf_device` / `phif_device` / `device_ready`를 **이미
받는다**. 오늘 XSSet이 거기에 null을 넣는 것은 export하는 쪽이 없기 때문이고,
`src/XSSet.cpp`의 `TryUpdateTHGpu`가 그 사실을 현장에 적어 둔다. 수신증은
`bytes_elided:0`을 0이 아닌 elision 시도 수와 나란히 찍는다 — "빌려 보려 했고 놓쳤다"는
"아낄 것이 없었다"와 전혀 다르게 읽힌다.

---

## 7. 수신증

### 7.1 `[RASBERY][TH_GPU]`

```
[RASBERY][TH_GPU] {"schema_version":1,"slot":0,"device":0,"arm":1,
  "th_updates":N,"device_updates":N,"host_fallbacks":0,"device_share":1.0,
  "channels":nxy,"nodes":8451,"bytes_elided":0,"bytes_h2d":…,"bytes_d2h":…,
  "wall_ms":…,"forms_mask":"0x54","policy_note":"…","status":"on"}
```

읽는 법 세 가지.

* **G0**: `th_updates == device_updates + host_fallbacks`는 회계 항등식이다. 플래그를
  켰는데 `device_updates:0`이면 "디바이스" 런과 "호스트" 런은 같은 런이고 거기서 잰
  모든 초는 무효다.
* **매핑**: `channels × (nodes/channels)`가 덱의 형상과 맞아야 한다. 스윕이 반경 위치당
  1레인이므로 `channels`가 덱의 `nxy`가 아니면 노심의 일부만 돌았다는 뜻이다.
* **수축 계약**: `forms_mask`는 커널이 실제로 돌린 마스크다. 아름이 한 번도 런치되지
  않았으면 `~`다 — 여기서 `thFormMask()`를 찍으면 **플래그를 켜지 않은 런에서 마스크를
  채굴**하게 되고 flag-off 로그에 `[RASBERY][FORMS]` 줄이 생긴다.

### 7.2 `[RASBERY][SEARCH_GPU]`

```
[RASBERY][SEARCH_GPU] {"schema_version":1,"slot":0,"arm":1,"applies":N,
  "device_applies":N,"host_fallbacks":0,"device_share":1.0,"proposals":N,
  "propose_transfers":0,"bytes_elided":…,"policy_note":"…"}
```

* `applies == proposals`여야 한다. 벌어지면 루프가 커밋한 시행을 단면적이 못 본 것이다.
* `propose_transfers`는 **0이어야 한다.** 0이 아니면 이 아름이 없애려던 왕복이 아름
  안에서 자란 것이다.
* 작은 아름일수록 수신증이 필요하다: 절감이 단일 덱 벽시계의 잡음 안이므로, 플래그가
  조용히 아무것도 안 하는 상황을 **알아챌 타이밍 신호가 없다.**

---

## 8. 238 런북

전제: `$BLD`는 CUDA 빌드 디렉터리, `$OUT`은 출력 루트(**로컬 계산 금지 · 출력은 E:**),
`$V6_ENV`는 v6 매니페스트의 환경 문자열.

### 8.1 A. T/H 아름 — digest 대조 (플래그 하나만 움직인다)

```bash
# 기준 (플래그 off)
env $V6_ENV RASBERY_STATEPOINT_TELEMETRY=1 \
    "$BLD/RASBERY" kngr_238.json -o "$OUT/wp22_a_off"

# T/H 아름
env $V6_ENV RASBERY_STATEPOINT_TELEMETRY=1 RASBERY_GPU_TH=1 \
    "$BLD/RASBERY" kngr_238.json -o "$OUT/wp22_a_th"

h5diff -c "$OUT/wp22_a_off/kngr.h5" "$OUT/wp22_a_th/kngr.h5"
```

판정:

1. `h5diff -c` **0 차이**이고 digest `1f36e75dc00ed2b4` / `4377` 불변 → **B0로 승격**하고
   `CudaThBackend.h`의 "REACHABLE IS NOT MEASURED" 문단과
   `ThGpuReceipt.h::kThGpuPolicyNote`를 그 측정으로 바꾼다.
2. 차이가 있으면 **N1 유지**, Gate A(`tools/gate_a_compare.py`) + Gate B(MASTER 대조)로
   간다. 그리고 §3.1의 인용문/호출지점 틈을 먼저 본다:
   `RASBERY_TH_FORMS`를 `0x00`, `0x54`, `0x57`, `0x1f3`로 스윕해 digest가 어느 값에서
   기준과 만나는지 확인한다. 만나는 값이 있으면 그것이 `SolveTH`의 인라인 문맥이 실제로
   돌린 마스크이고, `TH_FORMS_DEFAULT`가 아니라 **fixture**를 고쳐야 한다.
3. `[RASBERY][FORMS] {"mask":"TH_FORMS",…,"mined_sound":1}`을 확인한다.
   `mined_sound:0`이면 그 런에 대해서는 비트 동일성이 **성립하지 않는다**.

### 8.2 B. T/H 아름 — 단계 벽시계

```bash
grep -h SPTELEM "$OUT/wp22_a_off"/*.log | python tools/outer_profile.py --phase th_update
grep -h SPTELEM "$OUT/wp22_a_th"/*.log  | python tools/outer_profile.py --phase th_update
```

기대: `loop_wall.th_update`가 0.70 s → 0.1 s 이하. `nested_wall.flatxs`는 **불변**이어야
한다(이 아름은 `UpdateFlatXS`를 건드리지 않는다). 움직였다면 T/H 출력이 달라졌다는
뜻이고 그것은 §8.1 판정 2다.

### 8.3 C. 원장 바이트

```bash
env $V6_ENV RASBERY_GPU_TH=1 RASBERY_XFER_LEDGER=1 \
    "$BLD/RASBERY" kngr_238.json -o "$OUT/wp22_c_ledger"
grep -E "RASBERY\]\[XFER\]|CudaThBackend" "$OUT/wp22_c_ledger"/*.log
```

기대 행: `CudaThBackend.cu:solveTh:xskf` · `:phif` · `:tful` · `:tmod` · `:dmod` ·
`:snapshot`(D2D) · `:scalars`. 갱신 1회당 버스 ≈ 0.71 MB(H2D 507 kB + D2H 203 kB). `[RASBERY][TH_GPU]`의
`bytes_h2d`/`bytes_d2h`와 **일치해야 한다** — 두 수는 같은 런의 같은 복사를 두 곳에서
센 것이므로, 어긋나면 계측되지 않은 복사가 있다는 뜻이다.

### 8.4 D. 탐색 아름 — B0 확인과 elide 합성

```bash
# 탐색 아름은 XFER_ELIDE와 합성된다 (§4.2).  둘 다 켠다.
env $V6_ENV RASBERY_GPU_XFER_ELIDE=1 \
    "$BLD/RASBERY" kngr_238.json -o "$OUT/wp22_d_off"
env $V6_ENV RASBERY_GPU_XFER_ELIDE=1 RASBERY_GPU_SEARCH=1 \
    "$BLD/RASBERY" kngr_238.json -o "$OUT/wp22_d_search"

h5diff -c "$OUT/wp22_d_off/kngr.h5" "$OUT/wp22_d_search/kngr.h5"
```

판정: `h5diff -c` **0 차이**(등급이 구성상 B0이므로 이것은 확인이지 발견이 아니다) ·
`[RASBERY][SEARCH_GPU]`에서 `device_applies == applies` · `propose_transfers:0` ·
`bytes_elided == applies × 67,608`.

`RASBERY_GPU_XFER_ELIDE`를 끄고 `RASBERY_GPU_SEARCH=1`만 켠 대조군도 한 번 돌려
`host_fallbacks == applies`이고 status가 그 이유를 이름으로 말하는지 확인한다.

### 8.5 E. GPU_FULL 게이트

```bash
env $V6_ENV RASBERY_GPU_TH=1 RASBERY_GPU_FULL=1 \
    "$BLD/RASBERY" kngr_238.json -o "$OUT/wp22_e_full"
grep "RASBERY\]\[GPU_FULL\]" "$OUT/wp22_e_full"/*.log
```

기대: `th_fallbacks:0` · `search_fallbacks:0`(탐색 아름도 켰다면). 0이 아니면 아름이
어떤 statepoint에서 물러섰다는 뜻이고 `first_violation`이 어디인지 말한다.

### 8.6 F. 배치 처리량

```bash
python tools/run_multi_gpu_batch.py --decks 8 --workers 16 \
    --env "$V6_ENV RASBERY_GPU_TH=1" --repeat 3 --out "$OUT/wp22_f_batch"
```

median-of-3 c/h를 T/H off 대조군과 비교한다. **배치는 GPU 시간 바운드(sm 96 %)이므로
c/h가 크게 오를 것으로 기대하지 않는다.** 이 아름이 배치에서 사는 이유는 다른 것이다:
호스트 CPU가 73 % 바쁘고 그중 대부분이 CUDA 스핀이므로, statepoint당 0.70 s의 호스트
산술을 없애면 **코어가 풀린다.** 그래서 이 실행에서 읽을 숫자는 c/h가 아니라
호스트 CPU 점유율이다.

---

## 9. 빌드/도구 부수 변경

* `CMakeLists.txt`: `CudaThBackend.cu`를 CUDA 소스 목록과 `RASBERY_BITEXACT_CUDA_OPTS`
  (`--fmad=false`) 대상에, `CudaThBackendStub.cpp`를 non-CUDA 제외 목록에 추가.
  `ThReference.cpp`/`ThFormMiner.cpp`는 `src/*.cpp` glob으로 들어오며, RASBERY 타깃에는
  LTO가 걸려 있지 않으므로 인용문 TU의 분리가 유지된다(`CmfdOuterReference.cpp`와 동일).
* `tools/check_cuda_syntax.py`: `__syncthreads` · `cudaStreamWaitEvent` ·
  `cudaDeviceSynchronize` · `cudaEventSynchronize` 선언과 `std::isnan/isinf/isfinite`
  using, 그리고 `-DRASBERY_XFER_HAS_CUDA`를 추가했다. 이 게이트는 WP13.1이 복사를 원장
  래퍼로 라우팅한 이후 `CudaCramBackend.cu`에서 **깨져 있었다**(`rasbery::xfer::memcpyAsync`가
  호스트 컴파일에서 선언되지 않았기 때문). 지금은 `CudaThBackend.cu`와
  `CudaCramBackend.cu` 둘 다 통과한다. `CudaPprBackend.cu`는 런치 셰브론 재작성이
  템플릿 커널 이름에 걸려 여전히 실패하며, 그것은 이번 작업 범위 밖의 기존 결함이다.

---

## 10. 파일 목록과 각 파일이 지는 의무

| 파일 | 의무 |
|---|---|
| `src/ThKernel.h` | 순수 바디. g++/nvcc 공용, STL·예외·할당 없음. 모든 multiply-add가 마스크를 읽는다 |
| `src/ThReference.{h,cpp}` | 호스트 산술의 **축자 인용**. `ThKernel.h`를 절대 include하지 않는다 |
| `src/ThFormMine.h` | 채점·채굴. 양쪽을 다 보는 유일한 헤더 |
| `src/ThFormMask.h` / `ThFormMiner.cpp` | 프로덕션 마스크 해석. env override는 언제나 이긴다 |
| `src/ThGpuReceipt.h` | `[RASBERY][TH_GPU]` 필드와 등급 문자열 |
| `src/CudaThBackend.{h,cu}` / `…Stub.cpp` | 계약·비용 원장·등급 / 커널 4개와 버퍼 / CPU 전용 빌드 |
| `src/SearchKernel.h` | 노드당 저장 하나. 마스크 없음, 그 이유가 적혀 있음 |
| `src/SearchGpuReceipt.h` | `[RASBERY][SEARCH_GPU]` 필드와 B0 문자열 |
| `src/CudaXsReconBackend.{h,cu}` | 붕산 브로드캐스트 커널과 `applyBoronDevice` — **버퍼의 소유자에게** |
| `tools/test_th_gpu_contract.py` | 16 규칙 |
| `tools/test_search_gpu_contract.py` | 13 규칙 |
