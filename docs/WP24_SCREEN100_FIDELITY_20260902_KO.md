# WP24 — `RASBERY_FIDELITY=screen100`: 이름 붙은 수렴 프리셋

작성 2026-09-02 · 기준 tip `d515dde` (branch `codex/exact-throughput-campaign`)
· 181 컴파일 확인 완료 (WSL Ubuntu-24.04, gcc13 + nvcc 13.2, `sm_120`, Release, 링크까지 성공)
· **이 문서의 성능 수치는 전부 정적 추정이다. 238/181에서 측정한 것은 없다.**
  측정해야 할 것과 그 절차는 §7 런북에 있다.

---

## 1. 무엇을 만들었고, 왜 프리셋이어야 했나

사용자 지시는 GA 평가기의 정확도 목표를 프로덕션 봉투(1.905 pcm / 15.309 ppm /
AO 0.013 / pin 0.238 % / 0.80 %)에서 **MASTER 상대 |dkeff| ≤ 100 pcm, pin 오차
< 1 % (RMS·max)** 로 바꾸고, 그 목표에 맞게 모든 허용오차를 **일관되게** 설정한
이름 있는 프리셋을 만들라는 것이었다.

기존 A2 아암은 이름이 없다. `run_single_gpu_batch.DEFAULT_ENV`가 내보내는
`RASBERY_STAGED_FLUX_TOL=50`, `_XE_TOL=1000`, `_LOOSE_SETTLE=1` 세 줄이 전부이고,
어떤 영수증도 "이 숫자가 어느 아암에서 나왔나"에 답하지 못한다. 트리 자신이
이미 그 결함을 적어 두었다(`src/CaseFidelity.h`, "WHY AN `A2` DECLARATION MAY NOT
INVENT MULTIPLIERS"):

> `A2`는 **가족**이다. 50/1000이 측정된 아암이고 5/10은 다른 아암이며, 어느
> 쪽인지 말하지 않는 `A2` 영수증은 재현할 수 없는 숫자다.

스크리닝 프리셋을 같은 방식(환경변수 여섯 줄)으로 만들면 그 결함이 더 넓은
폭발반경으로 재발한다. **A2는 경로만 바꾸지만 screen100은 발표되는 답 자체를
바꾸기 때문이다.** 그래서 결과물은 표(`src/FidelityPreset.h`)이고, 트리의 다른
어떤 곳도 프리셋의 숫자를 적을 수 없다.

### 1.1 왜 승수만으로는 100 pcm 프리셋을 만들 수 없나 (핵심 논거)

스테이징에서 **폴리시 허용오차에 도달한 수렴만이 solve를 끝낼 수 있다**
(`Driver.h` `polishing` 래치, 4522-4560). 즉 발표되는 답은 언제나 폴리시
허용오차의 답이고, `RASBERY_STAGED_*` 승수는 아우터를 사줄 뿐 정확도 예산을
쓰지 않는다. 따라서 100 pcm 프리셋은 **프로덕션(폴리시) 숫자 자체를 움직여야
하며**, 이것이 이 표가 A2 아암이 한 번도 하지 않은 일이다.

---

## 2. 노브 표

`src/FidelityPreset.h` `kFidelityPresets`가 유일한 출처다. `strict`/`A2` 행은
현재 트리의 값을 **그대로 다시 적은 것**이고, `tools/test_fidelity_preset_contract.py`
가 양쪽을 소스에서 읽어 비교한다 — 표가 자기가 재진술한다고 주장하는 트리에서
멀어질 수 없다.

| 노브 | strict | A2 | **screen100** | 근거 |
|---|---|---|---|---|
| `staged_flux_mult` | 1.0 | 50.0 | **5.0** | 승수는 이미 완화된 폴리시 값에 곱해진다. §2.1 |
| `staged_xe_mult` | 1.0 | 1000.0 | **100.0** | loose Xe = 1e-5 × 100 = **1e-3**, A2 아암의 절대값과 동일 |
| `loose_settle` | false | true | **true** | 측정된 최대 단일 절감(5,675 → 4,614 아우터, −18.7 %) |
| `keff_tol_mult` | 1.0 | 1.0 | **10.0** → 1e-5 | 한 데케이드. 잔여오차 ~ρ/(1−ρ)×tol = 1–10 pcm/statepoint |
| `search_tol_mult` | 1.0 | 1.0 | **10.0** | 출하된 10:1 search:keff 비를 정확히 보존. 단 승수만으로는 **덱에 대해 무한대**다 — §2.8 |
| `search_tol_cap` | 0.0(없음) | 0.0(없음) | **1e-4 (절대)** | 승수 뒤에 min()으로 걸리는 절대 천장. 어떤 덱도 10 pcm을 넘겨 쓸 수 없다. §2.8 |
| `flux_l2_tol` | 1e-6 | 1e-6 | **1e-4** | errl2는 상대 핵분열원 변화 → 잔여 노드출력 0.01–0.1 % |
| `xe_tol` | 1e-6 | 1e-6 | **1e-5** | 한 데케이드만. 측정 datum: 1e-5 정지 시 keff 최대 8.9 pcm |
| `xe_oscillation_floor` | 1e-4 | 1e-4 | **1e-4 (고정)** | 100×tol로 떠다니면 1e-3 — 병리가 사는 자리. §2.2 |
| `cmfd_sweep_epsl2` | 1e-6 | 1e-6 | **1e-5** | flux_tol(1e-4) **이하**가 불변식. 1e-5는 미측정 보수값이고 1e-4도 적법하다. §2.3 |
| `rodcrit_search_cap` | 1e-5 | 1e-5 | **1e-5 (= 프로덕션, 의도된 no-op)** | 열은 존재하되 screen100은 **쓰지 않는다**. 미측정 + Gate B가 rod 덱에서 구조적으로 눈이 먼다. §2.4 |
| `rodcrit_search_floor_cusping` | 5e-5 | 5e-5 | **5e-5 (= 프로덕션, 의도된 no-op)** | 위와 같은 이유. §2.4 |
| `staged_search_margin` | 4.0(내장) | 4.0(내장) | **2.0** | `min()` 캡이 binding. §2.5 |
| `boron_bracket` | off | off | **ON** | 238 블록 33에서 −1.76 % 측정, **엄격 봉투로만** 기각됨. §2.6 |
| `carry_slope` | off | off | **off (고정)** | 측정 +8.11 % 아우터. 프리셋이 **명시적으로** 끈다. §2.7 |
| `warm_boron` | off | off | **off (고정)** | 효과가 무관한 CLI 플래그에 의존 → 한 이름 두 물리 |
| `max_trials` | 0 | 0 | **0 (캡 없음)** | 캡은 정확도를 완화하는 게 아니라 **포기**한다 |
| `statepoint_grid` | "" | "" | **"" (OFF)** | 요구사항 (4). 노브는 존재하되 꺼져 있다 |
| `TH_DOPPLER_TOLERANCE` | 1e-2 | 1e-2 | **1e-2 (불변)** | 이미 예산의 ~7.5 pcm. 3e-2면 ~22 pcm |
| `SEARCH_SETTLE_ITERS` | 2 | 2 | **2 (불변)** | 미정착 샘플 1개 = +112 pcm = 예산 전부 |
| `GA_FEEDBACK_PASSES` | 0 | 0 | **0 (건드리지 않음)** | 프로세스 FLOOR를 FeedbackLimited로 래치 → 승격 불가 |

### 2.1 왜 screen100의 스테이징 승수가 A2보다 **작은가**

오타가 아니다. 승수는 screen100이 **이미 움직인** 프로덕션 값에 곱해진다.
A2의 loose Xe는 1e-6 × 1000 = 1e-3이다. 그 **절대** loose 단계를 1e-5 베이스
위에서 유지하려면 승수는 1000이 아니라 100이다. 1000을 그대로 가져가면 loose
Xe가 1e-2가 되고, 캐스케이드가 한 스텝 만에 종료된 뒤 모든 폴리시 전이가
`prev_xe_change`를 무한대로 재무장한 채 캐스케이드 전체를 다시 돌린다
(`Driver::SolveLoop`의 폴리시 전이) — 절감이 아니라 스래시이고,
`staged_relapses`로 보인다.
WP9-D D3 폐기 규칙이 "relapses가 trials와 같은 자릿수"인 이유가 이것이다.

flux도 같은 논리: 5 × 1e-4 = 5e-4는 loose 단계를 폴리시보다 유용한 배수만큼
위에 두면서도 loose 합의가 무의미해지지는 않는 지점이다.

### 2.2 끊어야 했던 결합 — `XE_OSCILLATION_FLOOR`

`Driver.h`는 이것을 `100.0 * XE_EQUILIBRIUM_TOLERANCE`로 쓴다. Xe 허용오차를
1e-5로 완화하면 댐퍼 개입 바닥이 1e-4 → **1e-3**으로 조용히 떠오른다. 그런데
문서화된 병리(APR1400 cy01이 ~1e-3에서 80+ 스텝 진동)가 바로 거기 있다 — 댐퍼가
자기가 잡으려고 튜닝된 케이스에서 개입을 멈춘다. 그래서 screen100은 바닥을
**절대값 1e-4로 고정**한다. 계약 테스트가 이 항목을 별도로 검사한다.

**단 이것은 깨끗한 승리가 아니라 거래다(리뷰 수정).** `Driver.h`가 실제로 적어 둔
불변식은 절대값이 아니라 **비(ratio)** 다 — "자기 허용오차에서 아직 두 자릿수
떨어져 있는 반복만이 정체된 것이지 끝나가는 것이 아니다". 프로덕션 1e-6에 대해
바닥은 100배이고, screen100의 1e-5에 대해서는 **10배**다. 바닥이 넘어야 하는
측정된 지터 대역은 같은 주석의 i-SMR CY03 기록 — 1.46e-6으로 계약하고 1.65 /
1.92 / 3.64e-6을 읽은 세 번의 비수축, 즉 허용오차의 **최대 ~3.6배** — 이므로
screen100에서 남는 여유는 프로덕션의 ~28배가 아니라 **~2.8배**다. 그리고 바닥
아래에서의 오발이 CY03을, 재시작 사슬을 통해 CY04까지 **몇 pcm 헛되이** 움직인
기록이 있다. 1e-3으로 띄우는 것은 답이 아니다(그것이 APR1400 병리 그 자체다).
**~3e-4**면 ~30배를 회복하면서 병리보다 ~3배 아래에 머문다 — **세 번째로 스윕할
노브**다.

### 2.3 CMFD 스윕 종료(`_epsl2`) — 앞선 두 초안의 근거는 **산술이 거꾸로였다**

`_epsl2`는 아우터의 flux 허용오차와 **다른 숫자**다 — `BICGCMFD::drive()` 안
스윕 루프의 조기 탈출이다.

**앞선 초안의 주장(폐기).** "`flux_tol`과 같게 두면 탈출할 때마다 residual이
구조적으로 flux_tol 아래가 되어 `residual < flux_tol_now`가 자명하게 참이 되므로
10배 마진을 둔다." **코드는 정반대를 말한다.** `drive()`는 `errl2 < _epsl2`에서
break하고, 아니면 `_ncmfd = 5`를 소진한다. 아우터의 L2 절반을 그 두 갈래에 대고
읽으면:

| | break | 캡 소진 |
|---|---|---|
| `epsl2 == flux_tol` | `errl2 < flux_tol` → 아우터 L2 **통과** | `errl2 >= flux_tol` → **실패** |
| `epsl2 < flux_tol` | 자명하게 **통과** | `[epsl2, flux_tol)`에 떨어지면 **수렴하지 않았는데 통과** |

즉 `epsl2 == flux_tol`일 때 아우터의 L2 절반은 정확히 "스윕 루프가 수렴했는가"이며
**가능한 가장 엄격한 판정**이다. `epsl2`를 더 **조이면** L2 절반이 엄격해지는 게
아니라 **더 느슨해지고**, 동시에 내부 스윕 비용을 더 쓴다. 이 두 축 모두에서
`1e-4`가 `1e-5`를 지배한다. (출하 기본값이 `kProdCmfdSweepEpsl2 = kProdFluxL2Tol
= 1e-6`, 즉 1:1인 것도 같은 이야기다.)

**그럼에도 1e-5로 출발하는 진짜 이유.** 방어 가능한 거래는 위의 것이 아니라
다른 것이다: 아우터당 더 잘 수렴한 flux는 |dk| 절반을 **더 적은 아우터**로
만족시킬 수 있고, 비용 모형은 (아우터 수 × 아우터당 내부 비용)이므로 총합의
부호는 비만 보고 결정되지 않는다. **아무도 측정하지 않았다.** 그래서 정확도
방향으로 보수적인 1e-5에서 출발하되, **`1e-4`가 더 싸고 동시에 더 엄격한
방향임을 명시**하고 **screen100의 실측 아우터 수가 나오면 첫 번째로 스윕할
노브**로 남긴다. 계약 테스트의 술어도 이에 맞춰 `epsl2 <= flux_tol`로 고쳤다 —
살아남는 불변식은 "스윕이 아우터 판정이 요구하는 것보다 **느슨하게** 멈춰서는
안 된다"뿐이며, 1e-4로의 스윕을 금지하지 않는다.

### 2.4 프리셋을 조용히 무효화하는 함정 — RODCRIT 클램프

`Scheduler::criticalSearchTolerance()`는 RODCRIT에서
`max(min(tolerance_search, rodcrit_search_cap), rodcrit_search_floor)`를 반환한다.
`min()`이 완화를 **버린다**. `tolerance_search`만 올린 프리셋은 모든 rod-crit
덱(iSMR / CY 계열)에서 완전한 no-op이면서 영수증은 screen100이라고 말한다 —
CaseFidelity.h가 "선언보다 **더 정밀하게** 풀린 케이스"라고 부르는 방향의
결함이고, 증상은 "이유 없이 덱마다 재현되지 않는 speedup"뿐이다.

그래서 WP24는 클램프의 상수를 `Schedule::rodcrit_search_cap` 필드로 만들었다
(기본값이 `kRodCritSearchTol`이므로 프리셋 없이는 표현이 비트 단위로 동일).
호스트/디바이스 검색 파라미터 블록은
`tools/test_gpu_physics_interface_contract.py`가 필드 단위로 고정하므로
`DeviceScheduleParams`에도 쌍둥이를 추가했다(현재는 reset-only이며, 그 사실을
`TODO(WP24-device-search)` 마커로 남기고 계약 테스트가 그 마커를 고정한다 —
디바이스 검색이 켜지는 날 호스트는 프리셋으로 디바이스는 내장값으로 클램프하는
"한 solve 안 두 허용오차"가 되지 않도록).

### 2.4b 그리고 screen100은 그 두 열을 **쓰지 않는다** (리뷰 수정)

초안은 `rodcrit_search_cap` 1e-5 → 1e-4, cusping 바닥 5e-5 → 1e-4로 올렸다.
**되돌렸다.** 이 행이 인용하는 모든 측정은 **붕소 덱**에서 나왔다 — KNGR의
4,377 아우터, APR1400 기울기, −1.76 % 브래킷 결과. rod-crit 계열에는 측정이
하나도 없는데, 하필 그 덱들에서 screen100 Gate B는 **채점되는 스칼라 열 전부가
구조적으로 무력**하다:

| 열 | rod-crit 덱에서의 실체 |
|---|---|
| `delta_pcm` | 물리가 아니라 **검색 잔차**다(keff는 구조적으로 목표에 고정). 그리고 그 잔차를 묶는 것이 바로 이 캡이다 |
| `delta_ppm` | 덱이 고정한 붕소가 양쪽에 같으므로 **~0** (검색되는 것은 rod이지 붕소가 아니다) |
| `delta_ao` | screen100 봉투에서 **advisory**. 그런데 rod 오위치는 주로 **축방향 형상**으로 나타난다 — 잡아낼 수 있는 유일한 지표가 실패시키지 않는 지표다 |
| `delta_fqn/frn/fqp/frp` | 비교 도구는 계산하지만 **어느 봉투의 METRICS에도 없다** |

즉 `compare_master_rasbery.py --envelope screen100`이 iSMR/CY 덱에서
`GATE B scalars: PASS`를 출력하면서 **실패할 수 없는 열만 심판**할 수 있었다.
게이트가 가장 눈이 먼 곳에서 프리셋이 가장 느슨했다는 뜻이다. 조치는 둘이고
둘 다 산문이 아니라 검사다:

1. **행이 두 열을 프로덕션 값으로 되돌린다.** screen100은 rod-crit 검색에서
   **문서화된 no-op**이 된다 — 선언보다 정밀하게 푸는 안전한 방향이고,
   `[RASBERY][FIDELITY]`의 `resolved_entry0.search_tol`이 실제로 쓴 1e-5를
   출력하므로 no-op이 추론이 아니라 **보이는 사실**이다.
   `check_screen100`이 두 열이 `strict`와 같은지 검사하고, 음성 대조군이
   1e-4로 올린 행을 잡는다.
2. **`gate_b_envelope.report()`가 "구조적으로 고정된" 열만 심판했으면 PASS를
   거부한다**(`NOT SCORED`, exit 2). `compare_master_rasbery.py`는 조인된 모든
   statepoint에서 정확히 0.000인 델타 열을 그런 열로 신고한다 — 그것이 덱이
   양쪽에 고정한 양이라는 유일한 기계적 신호다.

**올리려면**: 먼저 rod-crit 계열을 그들 자신의 Gate B로 측정하고, 그 Gate B에
rod 덱에서 **실패할 수 있는** 열을 넣어라.

### 2.5 `staged_search_margin` 2.0

`Driver::SolveLoop`의 loose 허용오차 블록:

```
loose_keff_tol = has_search ? min(keff_tol × flux_mult, search_tol / margin) : …
```

캡이 **binding**이다. 출하 트리에서 `min(1e-6×50, 1e-5/4) = min(5e-5, 2.5e-6)
= 2.5e-6` — 즉 A2의 "50배 완화"는 |dk| 절반에서 실제로는 **2.5배**이고, 캡을
정하는 것은 `RASBERY_STAGED_FLUX_TOL`이 아니라 **검색 허용오차**다.
screen100에서 search_tol=1e-4, margin=2면 `min(1e-5×5, 5e-5) = 5e-5`. margin 1.0은
샘플을 검색 허용오차와 **같은** 자리에 놓는다 — 캡이 배제하려던 잡음 그 자체 —
이므로 2.0은 의도된 반걸음이지 끝점이 아니다.

**무엇을 사고 무엇을 쓰는지, 정직하게(리뷰 수정).**
`loose_flux_tol = max(loose_keff_tol, flux_tol × flux_mult) = max(2.5e-5 또는
5e-5, 5e-4) = 5e-4`로 **어느 쪽이든 같다.** 즉 margin이 움직이는 것은
`loose_keff_tol` 하나뿐이고, 2.5e-5 → 5e-5, **인수 2**다. 그 대가로 loose 샘플과
secant가 읽는 허용오차의 간격이 절반이 된다: 출하 관계는 **인수 4**이지 2가
아니다(초안의 "출하 트리와 같은 관계"라는 표현은 틀렸다). 그리고 이 트리의
rod-crit 기록이 바로 "자기 허용오차 근처에서 수렴한 flux로 keff를 읽는 검색은
튀고, 완화가 아낀 것보다 더 많은 아우터를 되쓴다"이다. 그래서 이것이
**두 번째로 스윕할 노브**이며, trial 수가 올라오면 내장 4.0으로 되돌리는 것이
보수적 선택이다.

### 2.6 `boron_bracket`을 켜는 근거

238, 2026-08-31, `bd7a0d3` 블록 33에서 WP9-D 검색 레버 다섯 개를 각각/전부
측정했다. 아우터를 줄인 것은 `RASBERY_SEARCH_BORON_BRACKET` 하나뿐이다:
4,377 → 4,300 (**−1.76 %**), `bisect=3`으로 브래킷이 실제로 걸린 것 확인,
단일 케이스 10.034 s vs 10.172 s. **기각 사유는 오직 엄격 봉투였다** — Gate A
keff 2.27 pcm (스크린 1.905 pcm), Gate B 1.945 pcm / 15.334 ppm.

**그런데 그 기각이 이 봉투에서 살아남지 못하는 이유는 "2.27이 100의 2 %라서"가
아니다(리뷰 수정).** 초안은 그렇게 적었고, 그것은 2.27 pcm를 예산에서 실제로
지출된 **정확도 오차**로 취급한 것이다.
`docs/A2_OUTER_REDUCTION_DESIGN_20260902_KO.md`는 다르게 말한다: `search_tol = 2e-5`
(2 pcm)에서 발표되는 keff는 근(root)이 아니라 **검색 경로의 함수**이고, 그 하나의
메커니즘이 boron_bracket·warm start·그 밖의 모든 수용 레버를 죽였다. 즉 1.905 pcm
스크린에 대한 2.27 pcm Gate A 편차는 **검색 경로 산물**이지 정확도가 아니다.
screen100에서 달라지는 것은 검색 허용오차가 (절대 캡이 걸린) **10 pcm**이라는 점
— 경로 산물이 아암이 요구하는 허용오차의 5배가 아니라 그 **안쪽**에 들어온다.
결론은 같고, 근거가 다르다.

**단, 이 플래그가 해소하지 못한 미해결 결함이 있다.** 8×M16 배치 + boron_bracket이
128 중 5개 워커 무성 실패(proc6, 오류 텍스트 없음, 재시도도 실패)를 냈고 같은
환경의 8×M8은 128/128 클린이었다. boron_bracket-OFF 8×M16 대조군은 실행된 적이
없고 포렌식은 블록 34다. **그 블록이 닫힐 때까지 screen100 배치는 M8로 고정한다.**

### 2.7 왜 프리셋이 자기 공간의 **모든** 노브를 주장하는가

`carry_slope`가 작동 예시다. 238이 +8.11 % 아우터로 측정했고(trials 130 vs
base ~120, 자기 D1 폐기 규칙 위반), `all_together`가 이를 물려받아 +15.8 %였다.
캠페인 셸이 이미 이 변수를 export한 상태에서 screen100을 실행하고 프리셋이 그
노브를 **언급하지 않으면**, 그 실행은 screen100이라고 말하는 영수증 아래에서
8 % 느리다. 그래서 표는 내장 기본값과 같은 값이라도 명시적으로 적는다.

같은 이유로 **이름 붙은 프리셋은 환경을 덮어쓴다** (defaults를 제공하는 게
아니라). 대안 — export된 노브가 이긴다 — 은 `DEFAULT_ENV` 안의 screen100 케이스가
50/1000(A2 아암의 승수)으로 풀리면서 영수증은 screen100이라고 말하는 것을
의미한다. 이름 없는 아암 결함의 재발이다. 요청의 명시적 노브
(`"staged_flux_tol": N`)만이 프리셋을 덮으며, `resolveCaseFidelity()`가 그것을
프리셋 **뒤에** 적용하므로 등가성 검사가 여전히 결과를 심판한다.

### 2.8 `search_tol_cap` — 승수는 **덱에 대해 무한대**다 (리뷰 수정)

`search_tol_mult`는 `schedule.tolerance_search`에 곱해지고, 그 값을 채우는 것은
`IO::ReadInput`이 읽는 **덱의** `search_tol`(또는 `search_pcm_tolerance × 1e-5`)이다.
따라서 "×10 = 10 pcm"은 덱이 내장값 1e-5를 말할 때만 참이다:

| 덱이 말한 값 | ×10 | 100 pcm 예산에서 |
|---|---|---|
| `search_tol: 1e-5` (내장) | 1e-4 = **10 pcm** | 10 % — 이 행의 ppm 산술이 가정하는 값 |
| `search_pcm_tolerance: 2` | 2e-4 = **20 pcm** | 20 % |
| `search_pcm_tolerance: 5` | 5e-4 = **50 pcm** | **예산의 절반, 노브 하나에서, 조용히** |

그리고 이 트리의 **지배적 리스크 발견 자체가 이 노브**다:
`docs/A2_OUTER_REDUCTION_DESIGN_20260902_KO.md`는 `search_tol = 2e-5`(2 pcm)에서
발표되는 keff가 근이 아니라 **검색 경로의 함수**임을 기록하고, 그 하나의
메커니즘이 boron_bracket·warm start·모든 수용 레버를 죽였다고 적는다.

더 나쁜 것은 계약 테스트가 그 사실을 볼 수 없었다는 점이다:
`check_screen100`은 `1.0e-5 × search_tol_mult`를 **내장값에서** 계산했으므로,
바이너리가 그 몇 배를 쓰는 동안 자기 가드와 ppm 교차검사를 둘 다 통과했다.

**수정**: 행이 절대 천장 `search_tol_cap`을 든다(`0.0` = 없음 = 출하 동작).
`Scheduler::criticalSearchTolerance(scale, cap)`이 **스케일 뒤, RODCRIT 클램프
앞**에 `min()`으로 건다(rod-crit 바닥의 `max()`가 마지막 말을 유지해야 하므로 —
바닥은 측정된 keff 잡음에서 검색을 빼내려고 있고, 천장이 그것을 도로 밀어 넣으면
안 된다). screen100은 `1e-4`를 든다: **어떤 덱도 10 pcm을 넘겨 쓸 수 없다.**
행이 RODCRIT 캡을 절대값으로 들어야 했던 것과 정확히 같은 이유이며, 계약 테스트는
이제 최악값을 **캡에서** 계산하고, 캡이 없는 승수 완화 자체를 거부한다
(`search_tol_cap = 0.0`으로 만든 음성 대조군이 그것을 확인한다).

`keff_tol_mult`가 맨 승수로 남는 이유는 대칭이 아니라 비대칭이다: 그것은
`rho/(1-rho)`를 통해 지출되고 **예산의 분수**로 검사되지 예산 자체로 검사되지
않는다.

---

## 3. 배선 — 여섯 소비자

값이 다섯 곳에만 도달하면 영수증 하나가 그 값에 대해 다른 의견을 갖는다.

| 소비자 | 위치 | 읽는 값 |
|---|---|---|
| 아우터 판정 (호스트+디바이스) | `Driver.h` `keff_tol` / `flux_tol` | `tolerance_keff × keff_tol_mult`, `max(keff_tol, flux_l2_tol)` |
| 임계 검색 | `Driver.h` `search_tol` | `criticalSearchTolerance(search_tol_mult)` + cap/floor |
| CMFD 스윕 탈출 | `Driver::Drive` `setEpsl2` | `ctx.tolerances.cmfd_sweep_epsl2` |
| Xe 캐스케이드 / 폴리시 | `xe_tol_now`, `loose_xe_tol`, 댐퍼 | `tol.xe_tol`, `tol.xe_oscillation_floor` |
| Anderson 무장 게이트 (호스트·디바이스 두 아암) | `picard < ctx.tolerances.xe_tol` | 같은 값 |
| 디바이스 Xe 요청 | `req.eq_tol` | `ctx.tolerances.xe_tol` |
| 임계 검색 **정책** (bracket / margin / carry / warm / trials) | `Driver::Run` `ctx.search_policy` | `_fidelity.searchPolicy()` |

**덱은 다시 쓰지 않는다.** `IO::ReadInput`이 파스 직후 덱의 정규 다이제스트를
접으므로, 허용오차를 덱에 써넣으면 케이스 키의 **DECK** 절반이 움직이고 한 노심의
두 fidelity가 두 노심처럼 보인다. 그래서 `keff`/`search`는 **승수**로,
나머지는 소비 지점의 **절대값**으로 적용한다.

전달 경로는 `SolverContext::tolerances`(`Driver::Drive`에서 `_fidelity`로부터
한 번 해석) — `fidelity`가 이미 그렇게 하는 것과 같은 이유다. SolveLoop가
static이므로 컨텍스트에 없는 정책은 SolveLoop 안의 static이 되어야 하고, 그것이
WP10.3이 한 커밋을 들여 제거한 래치다.

**소비자가 여섯이 아니라 일곱인 이유(리뷰 수정).** 첫 초안은 행의 허용오차
여덟 개만 케이스에 실었고 검색 노브 다섯 개는 `processSearchPolicy()` —
**환경**에서 프리셋을 읽는 프로세스 단위 static — 에 남겨두었다.
`SolverContext::search_policy`는 그 값으로 초기화된 뒤 **어디서도 재대입되지
않았다.** 결과:

* 평가기 소켓으로 `"fidelity_preset":"screen100"`을 요청한 GA 케이스는 행의
  허용오차를 받고 **환경의 검색 정책**을 받았다. `boron_bracket`은 행이 ON이라고
  말하는데 OFF, `staged_search_margin`은 내장 4.0으로 떨어져
  `loose_keff_tol = min(1e-5×5, 1e-4/4) = 2.5e-5` — 행이 의도한 5e-5가 아니다.
  §2.5가 margin 2.0으로 막으려던 클리핑 그 자체다.
* `RASBERY_SEARCH_CARRY_SLOPE`를 export한 캠페인 셸 안에서는 §2.7이 프리셋을
  만든 이유인 +8.11 % 레버가 켜진 채로 돌았다.
* `[RASBERY][FIDELITY]`가 그 다섯 개를 **표 행에서** 출력했으므로, 영수증이
  실행이 쓰지 않은 노브를 단언했다.
* 그리고 `armEnvValue`가 두 경로 모두에서 `RASBERY_FIDELITY="screen100"`을
  접었으므로 **케이스 키가 같았다** — 서로 다른 두 해에 대해 하나의 키, 즉
  캐시 미스가 아니라 **잘못된 히트**다.
* `promote`는 프리셋을 지우지만(§4) 지우는 것은 허용오차 절반뿐이었고, 검색
  절반은 프로세스 static에 그대로 남았다. 승격 재실행이 봉투 밖 검색 정책으로
  돌면서 영수증에는 `policy:"strict"`라고 적히는 경로였다.

**그리고 그 마지막 항목은 첫 수정이 닫지 못했다(2차 리뷰).** `searchPolicy()`가
행이 없을 때 `processSearchPolicy()`를 답했는데, **그 함수 자신이
`RASBERY_FIDELITY`를 해석한다**(`Scheduler.h`). 즉 "이 케이스는 프리셋이 없다"가
조용히 "이 케이스는 **프로세스의** 프리셋 검색 노브를 쓴다"를 의미했다. 런북이
지시하는 배치(§7.1: 프로세스를 `--set RASBERY_FIDELITY=screen100`으로 띄운다)에서
프리셋을 지우는 두 경로 — `op:"promote"`와 모든 `"fidelity":"strict"` 요청 —
는 곧 **수용 레인**이고, 그것들이 내장 허용오차 + screen100의 검색 정책
(boron_bracket **ON**, staged margin 2.0)으로 돌았다. 게다가 `armEnvValue()`가
프리셋에 `""`, 다섯 검색 이름에도 `""`(행이 값을 공급했으므로 raw 환경은 비어
있다)를 접었으므로, 그 payload는 **브래킷이 꺼진 프리셋 없는 프로세스의 진짜
strict 실행과 바이트 단위로 같다** — 하나의 케이스 키, 두 개의 해.

그래서 읽기를 **둘로 쪼갰다**: `environmentSearchPolicy()`는 행이 **없는** raw
환경 읽기(= WP24 이전 경로 그대로), `processSearchPolicy()`는 그것에 행을 얹은
프로세스의 답. 케이스는 행이 없으면 전자를 쓴다. 프리셋 없는 프로세스에서는 두
함수가 같은 비트를 답하므로 기존 실행은 하나도 움직이지 않고, screen100
프로세스에서는 지워진 프리셋이 실제로 **양쪽 절반 모두** 지워진다.
`check_cleared_preset`이 이것을 소스에서 고정하고, 음성 대조군이 옛 배선을
잡는다.

수정은 한 줄과 그 한 줄이 부르는 함수다: `CaseFidelity::searchPolicy()`가 행이
있으면 `presetSearchPolicy(row)`, 없으면 **`environmentSearchPolicy()`** 를 답하고, `Driver::Run`이 `ctx.tolerances` 바로 옆에서
`ctx.search_policy = _fidelity.searchPolicy();`로 실어 보낸다. 영수증은
`ctx.search_policy`에서 출력하고, `armEnvValue()`는 다섯 노브를 **유효 정책**에서
접는다(케이스가 아무것도 바꾸지 않았으면 원래의 raw 환경 문자열을 그대로 두므로
프리셋 없는 실행의 키는 바이트 단위로 불변).

---

## 4. 영수증·케이스 키·평가기

* **`[RASBERY][FIDELITY]`** — 새 줄, `{"schema_version":2,"preset":…,"policy":…,
  "acceptance_eligible":…,"knobs":{17개 노브 전부},"resolved_entry0":{…}}`.
  17개 노브는 **실행에서** 나온다: 허용오차는 `ctx.tolerances`, 검색 다섯 개는
  `ctx.search_policy`, 스테이징 승수는 `_fidelity` — 표 행이 아니다(요청의 명시
  노브가 행을 덮을 수 있으므로 행을 출력하면 틀린다).
  `resolved_entry0`은 스케줄 엔트리 0에 대해 **해석된 절대값**
  (`keff_tol` / `search_tol` / `flux_tol` / `xe_tol` / `loose_*`)이다:
  `keff_tol_mult`·`search_tol_mult`는 **덱이 말한** 값에 곱해지므로
  (`search_pcm_tolerance: 5`인 덱은 이미 5e-5, ×10이면 5e-4 = 50 pcm),
  승수만 적힌 영수증은 덱을 같이 들고 있지 않은 독자에게 아무 답도 주지 않는다.
  **프리셋이 지정된 경우에만**
  출력되므로 `RASBERY_FIDELITY`가 unset인 실행의 stdout은 바이트 단위로 이전과
  같다. `[CASE]` 줄이 "이게 어느 케이스이고 어떤 정책 단어를 달았나"에 답한다면
  이 줄은 "그 단어가 **어떤 숫자**인가"에 답한다 — A2 아암이 한 번도 답하지
  못한 질문이다.
* **`[RASBERY][CASE]`** — `schema_version` 6 → **7**, `fidelity_preset` 필드 추가
  (`"none"`이 기본, null이 아님).
* **light JSONL** (`BatchLightResult`) — `fidelity_preset` 필드. GA가 light
  스칼라를 서로 순위 매길 때 `policy:"A2"`만으로는 두 숫자가 비교 가능한지 알 수
  없기 때문이다.
* **`[RASBERY][EVALUATOR][CASE]`** — `fidelity_preset`, Driver 영수증에서. 요청의
  에코가 아니므로 잘못 적용된 프리셋은 불일치로 드러난다.
* **`[PHYSICS_MODE]`** (프로세스) — `fidelity_preset`.
* **평가기 hello** — `fidelity_presets`(이 빌드가 가진 행 목록),
  `fidelity_preset_default`.
* **`"none"`은 출력 어휘이자 입력 별칭이다(리뷰 수정).** `presetToken()`이 영수증
  네 곳에 `"none"`을 쓰는데 입력 어휘는 빈 문자열이었으므로, light 결과의 필드를
  다음 요청에 그대로 되먹이는 — 가장 자연스러운 — GA 경로가 바이너리가 방금
  출력한 값에 대해 "is not a preset this build knows" 거부를 받았다.
  `resolveCaseFidelity()`가 `"none"`을 빈 문자열로 정규화한다.
* **두 필드, 두 어휘, 그리고 겹친다.** `strict`/`A2`는 `fidelity`와
  `fidelity_preset` **양쪽 모두**에서 합법이고 `screen100`은 후자에서만,
  `L3coarse`는 전자에서만 합법이다. 즉 맞는 단어를 틀린 필드에 넣은 컨트롤러는
  screen100에 대해서는 **거부**를, strict/A2에 대해서는 **조용한 수락**을 받는다
  — 두 필드의 의미가 다른데도(`fidelity`는 케이스가 실제로 푸는 것과 등가성
  검사를 받는 **선언**, `fidelity_preset`은 17개 노브를 정하는 **행 이름**).
  평가기 hello 줄이 `fidelity_field_note`로 이것을 말한다.
* **JSON 요청** — `case` / `promote` / `wave`에서 `"fidelity_preset": "screen100"`.
  wave 레벨은 **기본값**(케이스가 자기 것을 말했으면 유지). `promote`는
  프리셋을 **지운다** — WP10.7이 `strict`/`full`/full-grid에 대해 펴는 논증 그대로.
* **케이스 키** — `RASBERY_FIDELITY`가 `trajectory::kArmEnv`에 들어갔고
  `armEnvValue()`가 per-case 분기를 얻었다. `tools/case_key.py`는 이 목록을
  `Driver.h`에서 읽으므로 Python 쪽 수정은 없다.
  **비용: 이 목록에 이름이 하나 추가되면 지금까지 계산된 모든 케이스 키가 바뀐다**
  (payload 한 줄 증가). WP9-D가 다섯 개를 한 번에 추가하며 만든 선례다. 캐시
  연속성이 필요한 GA 컨트롤러는 캐시 미스를 각오해야 한다 — 안전한 방향의 미스다.
* **`policy` 값은 여전히 `A2`.** `PhysicsFidelity`는 `kFidelityTraits[4]`로 닫힌
  4행 표이고 `case_key.py`·`exact_audit.py`·`test_result_fidelity_contract.py`가
  인덱스로 미러링한다. 다섯 번째 fidelity를 만들면 계약 세 개와 디스크 위의 모든
  매니페스트가 깨진다. screen100은 **StagedA2로 해석**되고, *어느* staged 아암인지는
  프리셋 이름이 답한다.

---

## 5. 승격 게이트와의 관계

`tools/promotion_gate.py`는 이미 grade `A2`/`L3coarse`를 기본값으로 승격시키기를
거부한다("A2/coarse는 MODE-SPECIFIC이며 strict 기본값을 대체하지 않는다").
screen100은 A2로 해석되므로 그 NEVER 판정에 걸린다. 승격 경로는 `op:"promote"`
— 프리셋을 지우고 strict/full/full-grid로 재실행 — 이며, 그 결과만 프로덕션
봉투에 대해 채점된다.

**그러나 "게이트 수정이 필요 없다"는 첫 초안의 주장은 두 군데에서 틀렸고, 둘 다
고쳤다(리뷰 수정).**

1. **A/B 위생 검사가 screen100과 프로덕션 A2를 구별하지 못했다.** 게이트의
   "두 아암이 다른 fidelity로 돌았다" 검사는 `base["policy"] != cand["policy"]`
   하나였는데, `policy`는 이 바이너리가 돌릴 수 있는 **모든** staged 아암에 대해
   `A2`다. 즉 50/1000 프로덕션 기준선에 대해 벤치마크한 screen100 후보가 "같은
   fidelity" 검사를 통과하고 그 처리량 이득이 한 아암의 것처럼 채점됐다. 정확히
   "A2는 **가족**이며, 가족 이름만 적힌 영수증은 아무도 재현할 수 없는 숫자"
   결함이 게이트하는 바로 그 도구 안에 남아 있던 것이다. 이제 `fidelity_preset`을
   `policy` 옆에서 비교한다(양쪽 다 필드가 없으면 WP24 이전 블록이므로 검사하지
   않는다 — 한쪽에만 없는 것은 불일치다).
2. **NEVER 판정은 `block["grade"]`, 즉 사람이 적은 필드에 걸려 있었다.** 이제
   아암의 영수증이 `none`/`strict`가 아닌 프리셋을 보고하는데 grade가
   `A2`/`L3coarse`가 아니면 블로커가 붙는다.

그 필드가 캠페인 영수증에 실제로 도달하도록 `tools/run_multi_gpu_batch.py`가
`[PHYSICS_MODE]`의 `fidelity_preset`을 per-case 항목·워커 영수증·후보 영수증·
캠페인 TOTAL(`fidelity_preset_measured`)에 함께 접는다.

**아암에 무엇을 적어야 하는가(리뷰 수정).** 게이트가 비교하는 것은 아암의
**스칼라** `fidelity_preset`이고, 하네스가 내보내는 것은 jobs-가중 **딕셔너리**
`fidelity_preset_measured`다. 둘을 잇는 규칙은 하나뿐이며 여기 적어 둔다:
**`fidelity_preset_measured`의 키가 하나면 그 키를 아암의 `fidelity_preset`으로
적는다. 키가 둘 이상이면 그 아암 자체가 혼합 아암이므로 A/B에 쓸 수 없다.**
그리고 `"unreported"` 키는 표 항목이 아니라 **캠페인 실패**다 — 그 값은 자식이
`fidelity_preset`을 아예 출력하지 않았다는 뜻이고, `check_run_receipts()`가 그런
실행을 이미 거부한다(§4의 라운드트립 검사).

**그리고 하네스는 자기가 요청한 프리셋을 자식이 실제로 돌렸는지 확인한다.**
이것이 없으면 다중 호스트에서 가장 흔한 운영 실수가 조용히 통과한다: 런북대로
`--set RASBERY_FIDELITY=screen100`을 주었는데 그 호스트의 바이너리가 WP24 이전
빌드인 경우(리빌드 안 된 호스트, PATH의 낡은 RASBERY). 그 바이너리는 변수를
무시하고 `DEFAULT_ENV`의 50/1000 A2 아암을 **프로덕션 폴리시 허용오차로** 풀며
`policy:"A2"`를 출력한다. `derive_declared_fidelity()`도 (행의 승수를 읽어) `A2`를
답하므로 정책 비교는 A2 대 A2로 **통과**하고, 캠페인은 프로덕션 허용오차의
처리량 숫자를 스크리닝 이름 아래 적재한다 — `src/main.cpp`의
`fidelityPresetEnvIsUnknown()` 거부가 막으려는 결함 그대로이며, 그 거부는
**낡은 바이너리가 돌고 있으므로** 도울 수 없다. 그래서
`check_run_receipts()`가 `receipt_preset(output)`을 자식 환경의
`RASBERY_FIDELITY`(없으면 `"none"`)와 **같은지** 검사하고, 필드 자체가 없으면
`CASE_REQUIRED_FIELDS`가 WP10.3 버전 거부에 쓰는 것과 같은 모양으로
**거부**한다("이 바이너리는 WP24 이전이라 받은 프리셋을 이행할 수 없다").

`exact_audit.derive_declared_fidelity()`가 `RASBERY_FIDELITY`를 배웠다. 이것이
없었다면 screen100 자식은 `A2`를 출력하는데 하네스는 `strict`를 기대해 **모든
실행이 정책 불일치로 무효화**된다 — 그 모듈이 없애려고 만들어진 WP4 결함의
그대로의 재커밋이다. 오타난 프리셋 이름은 파생 대신 `ValueError`를 던진다(바이너리가
`exit 2`로 거부하므로, 정책을 선언할 실행 자체가 없다). `--strict`가 지우는 키
목록에도 `RASBERY_FIDELITY`가 들어갔다.

---

## 6. Gate B 봉투 (요구사항 3)

WP24 이전에는 **두 Gate B 도구 모두 무조건 0을 반환했다.**
`compare_master_rasbery.py`에는 임계값 자체가 없었고 `gate_b_pin_rms.py`는 인자
두 개를 받아 한 줄을 출력했다. 프로덕션 봉투는 docs/ 안의 산문으로만 존재했다.
즉 "Gate B 통과"는 사람이 터미널의 숫자를 기억 속 설계문서 수치와 비교하는
일이었다.

`tools/gate_b_envelope.py`가 두 봉투를 숫자로 들고 있고, 두 도구가
`--envelope {production,screen100}`을 받으며 **위반 시 비-0으로 종료**한다.
**기본값은 `production`** 이므로 기존 호출이 조용히 느슨한 쪽을 얻는 일은 없다.

**네 열 모두 MASTER 상대 *절대* 한계이고, 프로덕션 기준선이 이미 그 안에 들어
있다.** 이것이 screen100 FAIL을 읽기 전에 반드시 읽어야 하는 문장이고, 첫 초안이
빠뜨린 문장이다. 프로덕션 RASBERY-vs-MASTER는 0이 아니므로, 같은 이름 아래에서
각 열이 허용하는 **새 오차**의 양은 서로 다르다.

**그리고 production 행은 *측정값*이 아니라 *합격선*이다(2차 리뷰).**
`docs/A2_OUTER_REDUCTION_20260829_KO.md` §5의 표에는 열이 **둘** 있다 —
`v2 기준`(v2 참조가 실제로 잰 값)과 `합격`(넘어서는 안 되는 선). 첫 초안은
**측정 열**을 가져왔다(1.905 / 15.309 / 0.238). 결과는 확인 가능하고 확인했다:
`docs/PRICING_PROD_20260830_KO.md`가 기록한 **채택된** 프로덕션 Gate B는
max|Δppm| = **15.334**, max|Δpcm| = 1.847, max|ΔAO| = 0.012이고 판정은 PASS다.
15.309 한계에 대고 돌리면 그 동결된 결과가 **exit 1**이다. `production`은
**기본** 봉투이므로, 기존의 모든 `compare_master_rasbery.py` 호출이 캠페인이 이미
채택한 실행에서 비-0을 내기 시작했을 것이다 — 모듈 docstring이 스스로 경고한
"정당한 실행이 반올림 차이로 실패하면 게이트는 첫날에 신뢰를 잃는다"가 기본값으로
출하되는 형태다. 그래서 행은 **합격 열**을 적는다. 측정값은 docs에 남는다.
계약 테스트는 채택된 프로덕션 Gate B 숫자를 그대로 `production` 봉투에 통과시키는
행위 검사를 갖는다.

| 지표 | production(합격선) | screen100 | **새 오차 여유** | 유도 |
|---|---|---|---|---|
| `keff_pcm` | 2.0 | **100** | 98.0 pcm | 지시 그대로 |
| `ppm` (CBC) | 15.4 | **33.5** | 18.1 ppm | **유도값.** 붕소 검색 statepoint는 \|dkeff\|가 없다(keff가 구조적으로 목표에 고정). 그래서 keff 예산을 −5.4 pcm/ppm 기울기로 ppm에 옮겨야 하는데, **옮겨야 하는 것은 한계가 아니라 증분**이다: 15.4 + 98.0/5.4 = 33.5 ppm |
| `ao` | 0.013 | **advisory (실패시키지 않음)** | — | keff 예산에서 축방향 오프셋으로 가는 방어 가능한 연결이 없다. 완결돼 보이려고 숫자를 지어내는 것이 게이트가 발화할 때 아무도 변호할 수 없는 숫자를 갖게 되는 경로다 |
| `pin_rms_pct` | 0.24 | **1.0** | 0.76 pp | 지시 |
| `pin_max_pct` | 0.80 | **1.0** | **0.20 pp — binding될 열** | 지시 |

**왜 ppm이 18.5가 아니라 33.5인가(리뷰 수정).** 100/5.4 = 18.5는 ppm 한계를
keff **한계**의 상(image)으로 읽은 값이다. 그런데 keff 한계는 ~0에서 재고 ppm
한계는 프로덕션 CBC 합격선에서 잰다. 18.5로 두면 keff에는 ~98 pcm의 새 오차를,
붕소에는 3.1 ppm(≈17 pcm 등가)을 허용하는 것이 되고 — screen100의 **자기 노브가
그보다 더 쓴다**: 검색 허용오차(절대 캡 1e-4) = 10 pcm = 1.85 ppm,
xe_tol 1e-5의 측정 8.9 pcm = 1.65 ppm, 합 3.5 ppm이 이미 3.1 ppm을 넘는다(연소
누적은 아직 세지도 않았다). `tools/test_fidelity_preset_contract.py`의
`check_screen100`이 이제 이 교차검사를 한다 — 이 패치의 두 절반이 다시는 붕소
예산에 대해 서로 다른 말을 할 수 없다.

**그 선택이 낳는 결과를 한 문장으로(리뷰 수정).** 임계붕소 덱에서는 `delta_pcm`이
구조적으로 ~0이고 모델-대-MASTER 반응도 오차 전부가 `delta_ppm`에 실린다. 따라서
**그런 덱에서 screen100의 실효 반응도 게이트는 33.5 ppm × 5.4 = ~181 pcm**,
즉 지시의 100 pcm의 약 1.8배이며, keff 열은 100 절대값에 묶여 있다. 내부적으로는
일관되지만(같은 예산을 각 열의 단위로 표현한 것), screen100을 지시와 대조하는
독자는 33.5 ppm을 "100 pcm"으로 읽기 쉽다. `gate_b_envelope.ENVELOPES["screen100"].note`
가 이 문장을 들고 있고 모든 판정에 함께 출력된다.

**`pin_max` 1.0 %는 측정이 아니라 지시다.** 그리고 그것이 남기는 실제 여유는
0.20 pp다. 더구나 측정된 staged 스캔에는 **프로덕션 폴리시 허용오차에서** 이미
1.12–1.16 % pin max에 있는 아암이 있고 screen100은 그 위에 폴리시를 더 푼다. 즉
**이 열은 검증된 것이 아니며, screen100 행은 덱별 Gate B를 기다리는 후보이지
캠페인 기본값이 아니다.** 이 숫자들은 `gate_b_envelope.ENVELOPES["screen100"].note`
안에 있고 두 도구가 판정할 때마다 출력하므로, FAIL을 읽는 사람이 "1 % 봉투의
절반"과 "0.20 pp 여유의 절반"을 혼동하지 않는다.

**경계.** 지시는 pin 오차 "< 1 %"라고 말하고 `verdict()`는 `<= limit`을 쓴다.
한계에 정확히 앉은 측정은 반올림 산물이지 위반이 아니라는 판단이며, 소스에서
발견하게 두지 않고 여기와 모듈 docstring에 적어 둔다.

**아무것도 재지 않은 실행은 PASS가 아니다.** `verdict()`는 호출자가 주지 않은
지표를 건너뛴다(붕소 검색이 없는 덱에는 ppm 열이 없고, 없는 것을 통과로 세는 건
그 자체가 결함이다) — 그래서 빈 측정은 "통과"한다. `scored()`가 실제로 심판된
지표를 돌려주고 `report()`가 그것이 비면 **`NOT SCORED`와 exit 2**를 낸다.
두 도구 어느 쪽도 봉투 전체를 재지 않으므로(`compare_master_rasbery.py`는 스칼라
셋, `gate_b_pin_rms.py`는 핀 둘), 판정 줄은 자기가 재지 **않은** 열도 이름으로
적는다 — 한쪽 도구만 인용한 "Gate B 통과"는 봉투의 절반이다.

**그리고 "잰 것"에는 *실패할 수 없는 것*이 포함되지 않는다(리뷰 수정).**
`report()`는 호출자가 **구조적으로 고정됨(pinned)** 이라고 신고한 열을 심판된
것으로 세지 않고, 심판된 열이 전부 pinned면 `NOT SCORED`(exit 2)를 낸다.
`compare_master_rasbery.py`는 조인된 모든 statepoint에서 **정확히 0.000**인 델타
열을 pinned로 신고한다 — rod-crit 덱의 `delta_ppm`(양쪽이 덱이 고정한 같은 붕소)이
그것이며, 그것이 덱이 양쪽에 고정한 양이라는 유일한 기계적 신호다. 검사는
**정확한 0 등호**여서 근사-0인 진짜 일치는 계속 채점된다(따라서 이 판정은 pinned
열을 **과소** 신고할 수는 있어도 없는 `NOT SCORED`를 만들지는 않는다).

**어느 봉투도 심판하지 않는 열: 피킹.** `compare_master_rasbery.py`는
`delta_fqn`/`delta_frn`/`delta_fqp`/`delta_frp`를 계산하지만 `METRICS`에는 없다.
AO가 advisory인 덱에서는 축·경 방향 피킹 이야기 전체가 무채점으로 남는다는 뜻이다.
`report()`가 매 판정마다 이 네 열을 이름으로 적는다.

### `max`의 정의 (명시 요구사항)

`max`는 **비교 대상 핀 모집단 전체에 대한 최대 |상대오차| (%)**, 즉 **최악의 핀
하나**다. 백분위수가 아니고 집합체 평균도 아니다. 모집단과 정규화는 이 도구가
줄곧 쓰던 것 그대로다: 양쪽을 각자의 양수-핀 평균으로 나누어(따라서 출력 **모양**을
비교하지 출력 레벨을 비교하지 않는다) MASTER 값이 0.05를 넘는 핀만 넣는다. 같은
수치에서 max는 RMS보다 엄격히 강한 검사이며, 실제로 binding될 쪽이다.

### `--all-steps`가 왜 필요한가

스테이징에서 발표 상태는 **언제나** 프로덕션 허용오차를 만족했다(POLISH가
발표 전에 복원). 따라서 staged 실행의 핀 오차는 수렴 오차가 아니라 **연소를 통한
궤적 발산**이고, 측정된 스캔은 그것을 기울기가 아니라 **절벽**으로 보여준다:
다섯 아암이 1.12–1.16 %에 있는 동안 이웃 아암들은 0.03–0.07 %에 있고, 그 중
하나의 편차는 **statepoint 28–33에만** 존재한다. BOC 전용 핀 게이트는 이것을 볼
수 없다.

**그러나 첫 초안의 `--all-steps`는 그 관찰을 잘못된 계측기로 바꿔 놓았다(리뷰
수정).** 그것은 PPI **하나**(문서화된 모든 호출에서 `kngr_mas_ppi_boc.txt`,
즉 BOC 분포)를 파싱해 **모든** statepoint를 그 하나에 대해 재고, 최악 스텝에서
판정을 취했다. BOC→EOC 핀 형상 재분포는 수 %에서 수십 %의 **실제 물리**이므로
`--all-steps --envelope screen100`은 **정상 실행에서도 반드시 FAIL**한다 —
프리셋과 아무 상관 없는 이유로. 항상 실패하는 게이트는 무시되고, 무시되는 게이트는
WP24가 대체한 무조건 `exit 0`과 운용상 같은 것이다. 그리고 §7.1이 바로 그 호출을
screen100 판정으로 처방하고 있었다.

그래서 규칙은 **같은 것끼리 아니면 채점하지 않는다**로 바뀌었다:

| | 무엇 | 판정에 들어가나 |
|---|---|---|
| **SCORED** | 자기 MASTER 기준(positional PPI는 `--step`의 기준, `--ppi-step STEP=PATH`로 추가)을 가진 statepoint | **예.** exit code는 이 모집단에서만 나온다 |
| **DRIFT** | `--all-steps`가 추가로 보고하는 나머지 statepoint, positional(BOC) PPI 기준 | **아니오.** 출력만 한다 |

`--all-steps`가 겨냥한 후기연소 실패 모드는 실재한다. 그것을 **채점**하려면
statepoint별 MASTER PPI가 필요하고 그게 `--ppi-step`이 받는 것이다. 그 전까지
드리프트 열이 읽을 수 있는 전부다. 인자 두 개짜리 기존 호출은 하던 그대로
step 0001을 채점하고, 하위 호환을 위해 `BOC pin: rms … max …` 줄도 계속 출력한다
(`docs/V5_FREEZE_20260830_KO.md`, `docs/WP20_GPU_FP32_20260831_KO.md`가 그 문자열을
인용한다).

---

## 7. 238 런북

전제: `RASBERY_FIDELITY`가 `kArmEnv`에 들어갔으므로 **캐시 연속성은 없다.**
모든 아암을 cold로 잡는다. boron_bracket 미해결 결함(§2.6) 때문에 **배치는 M8**.

### 7.1 KNGR 단일: strict vs screen100

```
# A) 기준선 (현행 프로덕션 A2 아암, 프리셋 없음 — 지금까지의 캠페인 그대로)
python tools/run_single_gpu_batch.py --gpu 0 --batch-width 1 \
    -- RASBERY --rasi <kngr_238.json> --raso E:/run/wp24/base --batch-mode 1

# B) screen100
python tools/run_single_gpu_batch.py --gpu 0 --batch-width 1 \
    --set RASBERY_FIDELITY=screen100 \
    --set-unset RASBERY_STAGED_FLUX_TOL --set-unset RASBERY_STAGED_XE_TOL \
    --set-unset RASBERY_STAGED_LOOSE_SETTLE \
    -- RASBERY --rasi <kngr_238.json> --raso E:/run/wp24/s100 --batch-mode 1
```

(B의 `--set-unset` 세 줄은 **필수다.** 첫 초안은 "엄밀히는 불필요"라고 적었지만
그렇지 않다: `armEnvValue()`는 케이스의 승수가 프로세스 기본값과 **같을 때** RAW
환경 텍스트를 접는다. `DEFAULT_ENV`의 50/1000/1이 남아 있는 screen100 실행은
5/100으로 풀면서 env 다이제스트에는 `"50"`/`"1000"`/`"1"`을 접으므로, 같은 해에
대해 세 노브를 지운 실행과 **다른 케이스 키**가 나온다. 보수적인 방향(캐시 미스이지
잘못된 히트는 아님)이지만 캐시를 조각내고 같은 아암의 두 영수증을 키로 비교할 수
없게 만든다. 그래서 런북에서 이 세 줄은 선택이 아니다.
`[RASBERY][FIDELITY]` 줄이 실제로 적용된 값을 확인해 준다.)

**기록할 것**

| 항목 | 어디서 |
|---|---|
| wall (s) | 하네스 요약 |
| 총 아우터 / statepoint 수 | `[RASBERY][CASE]` `outers`, `statepoints` |
| 아우터/sp | 위 둘의 비. 기준: KNGR **125/sp** (4,377 / 35) |
| `staged_relapses`, `trials`, `bisect` | `SPTELEM SUMMARY` |
| `search.exit != 1` statepoint 수 | 100 pcm 봉투가 인증할 수 없는 집합 |
| 적용된 노브 | `[RASBERY][FIDELITY]` 줄 (표 §2와 대조) |
| 해석된 절대 허용오차 | `[RASBERY][FIDELITY]` `resolved_entry0` — 덱이 말한 값 × 승수 |
| 유효 검색 정책 | `[RASBERY][FIDELITY]` `boron_bracket`/`staged_search_margin` 등 — 표 행이 아니라 **실행**에서 나온다 |
| `policy` / `fidelity_preset` | `[RASBERY][CASE]` — `A2` / `screen100` 이어야 함 |

**Gate B (screen100 봉투)**

```
python tools/compare_master_rasbery.py <MAS_SUM> E:/run/wp24/s100/*.h5 \
    -o E:/run/wp24/s100_gateb --envelope screen100        # exit code가 판정

# 핀: 판정은 자기 MASTER 기준을 가진 statepoint에서만 나온다.
# 기준이 BOC 하나뿐이면 판정도 BOC 하나이고, --all-steps는 드리프트 보고만 한다.
python tools/gate_b_pin_rms.py E:/run/wp24/s100/*.h5 <kngr_mas_ppi_boc.txt> \
    --all-steps --envelope screen100                      # exit code가 판정
# statepoint별 MASTER PPI가 있으면 그만큼 채점 대상이 늘어난다:
python tools/gate_b_pin_rms.py E:/run/wp24/s100/*.h5 <kngr_mas_ppi_boc.txt> \
    --ppi-step 0018=<mas_ppi_mid.txt> --ppi-step 0033=<mas_ppi_eoc.txt> \
    --all-steps --envelope screen100
```

채점: `max|delta_pcm| ≤ 100`, `max|delta_ppm| ≤ 33.5`, pin RMS·max `≤ 1.0 %`
**— 채점된(자기 기준을 가진) statepoint의 최악값에서**. `--all-steps`가 추가로
찍는 BOC 기준 드리프트 줄은 후기연소 이상을 **보는** 용도이지 채점 대상이 아니다
(BOC→EOC 재분포가 수 %~수십 %이므로 그것을 1 % 봉투로 재면 정상 실행도 반드시
FAIL한다). `delta_ao`는 보고만.

**두 도구를 모두 돌려야 한다.** 어느 쪽도 봉투 전체를 재지 않으므로(스칼라 셋 /
핀 둘) 한쪽만 인용한 "Gate B 통과"는 봉투의 절반이다. 판정 줄이 자기가 재지 않은
열을 이름으로 적어 주고, 아무 열도 재지 못했으면 `NOT SCORED`와 **exit 2**를 낸다.

**이 두 도구는 이제 위반 시 비-0으로 종료한다.** `set -e` 런북에 넣기 전에
알려진-정상 **프로덕션** 결과에 대해 한 번씩 돌려 볼 것 — 지금까지 눈으로만
비교하던 수치이므로, 정상 실행이 반올림으로 실패하면 첫날에 게이트 신뢰를 잃는다.

**Gate A (strict 대비, 정보용)**

```
python tools/gate_a_compare.py <base h5> <s100 h5> --keff-pcm 100 --pin-pct 1.0
```
strict 대비 궤적 차이의 크기를 알기 위한 것이지 채점이 아니다. screen100의
채점 기준은 MASTER 상대 Gate B다.

### 7.2 m64 배치 처리량: 8×M8, median-of-3

```
for i in 1 2 3; do
  python tools/run_multi_gpu_batch.py --workers 8 --batch-width 8 \
      --set RASBERY_FIDELITY=screen100 \
      --set-unset RASBERY_STAGED_FLUX_TOL --set-unset RASBERY_STAGED_XE_TOL \
      --set-unset RASBERY_STAGED_LOOSE_SETTLE \
      -- RASBERY --jobs <m64 manifest> --batch-mode 8 --result light
done
```

- **median-of-3의 c/h**를 base(프리셋 없음, 동일 8×M8) 대비 보고.
- 배치는 GPU-time-bound이므로 **c/h는 1/outers로 스케일**한다. 아우터 감소율과
  c/h 증가율이 크게 어긋나면 그것 자체가 조사 대상(스래시 또는 호스트 병목).
- **`--set-unset` 세 줄은 §7.1과 같은 이유로 여기서도 필수다(리뷰 수정).**
  초안은 이 아암에서 세 줄을 빠뜨렸는데, 그러면 5/100으로 푸는 실행이 env
  다이제스트에는 `DEFAULT_ENV`의 `"50"`/`"1000"`/`"1"`을 접으므로 §7.1 아암과
  **키가 비교되지 않는다**. 처리량 주장 전체가 이 아암에 걸려 있으므로 두 레시피는
  같아야 한다.
- **M16을 쓰지 말 것** (§2.6).
- 128/128 성공을 확인할 것. 무성 워커 실패가 나오면 boron_bracket 결함이
  M8까지 온 것이므로 즉시 중단하고 블록 34로 넘긴다.

### 7.3 회귀 확인 (계산 없이)

```
python tools/test_fidelity_preset_contract.py
python tools/test_case_fidelity_contract.py
python tools/test_case_key_contract.py
python tools/test_staged_tolerance.py
python tools/test_search_policy_contract.py
python tools/test_promotion_gate.py
python tools/test_soak_receipt_schema_contract.py
python tools/test_gpu_physics_interface_contract.py
python tools/test_harness_env_parity.py
python tools/test_enum_alias_contract.py
python tools/test_dependent_template_contract.py
```

### 7.4 첫 번째 스윕 아암: `screen100e4` (WP24.1)

§2.3이 "screen100의 실측 아우터 수가 나오면 **첫 번째로 스윕할 노브**"라고 지목한
`cmfd_sweep_epsl2` 1e-5 → 1e-4를, **소스를 고치지 않고** 환경변수 하나로 잡을 수
있게 표에 네 번째 행 `screen100e4`를 추가했다. 이 행은 `screen100`과 **이름과
`cmfd_sweep_epsl2` 외의 모든 열이 문자 그대로 동일**하다 —
`tools/test_fidelity_preset_contract.py`의 `check_sweep_arm`이 그것을 검사하므로,
`check_screen100`이 부모 행에 대해 증명하는 모든 불변식이 이 아암에 대해서도
구성상 성립한다. 1e-4는 `flux_l2_tol`과 **같은 값**이고(불변식 `epsl2 <= flux_tol`을
등호로 만족), 그 지점에서 아우터의 L2 절반은 정확히 "스윕 루프가 수렴했는가"가
된다 — §2.3 표의 가장 엄격한 읽기이자 출하 트리의 1:1 상태다. 아우터당 비용은
싸지지만 **아우터 수**가 늘 수 있고 그 곱의 부호는 비만 보고 결정되지 않으므로,
두 아암 모두에 대해 **outers/sp와 wall을 같이** 보고할 것. 한쪽만 인용하면 스윕이
물은 것과 다른 질문에 답하게 된다.

행이지 오버라이드 노브가 아닌 이유: 케이스 키는 프리셋의 **이름만** 접는다.
`screen100`이라는 이름 아래에서 epsl2를 움직이는 환경 노브는 서로 다른 두 스윕
종료값에 **하나의 케이스 키**를 주므로, 잘못된 캐시 HIT — `kArmEnv`가 절대 허용해서는
안 되는 단 하나의 방향 — 이 된다. 행 다이제스트 검사의 문구 그대로 "재튜닝은 새 아암이고
새 이름을 받을 자격이 있다"이며, `e4`라는 접미사가 어떤 노브를 어떤 값으로 옮겼는지
말한다. 실제로 `screen100`과 `screen100e4`는 env 다이제스트가 **다르므로** 두 아암은
캐시가 섞이지 않는다.

```
# C) screen100e4 — B와 완전히 같은 레시피에서 프리셋 이름만 바꾼다
python tools/run_single_gpu_batch.py --gpu 0 --batch-width 1 \
    --set RASBERY_FIDELITY=screen100e4 \
    --set-unset RASBERY_STAGED_FLUX_TOL --set-unset RASBERY_STAGED_XE_TOL \
    --set-unset RASBERY_STAGED_LOOSE_SETTLE \
    -- RASBERY --rasi <kngr_238.json> --raso E:/run/wp24/s100e4 --batch-mode 1
```

`--set-unset` 세 줄은 §7.1과 **같은 이유로** 여기서도 필수다(그래야 B와 C의 키가
비교된다). `[RASBERY][FIDELITY]` 영수증의 `fidelity_preset`이 `screen100e4`,
`resolved_entry0`의 스윕 종료값이 `1e-4`로 찍히는지 확인할 것. 채점 봉투는 **부모와
같다** — `tools/gate_b_envelope.py`가 이 이름을 `screen100` 봉투로 매핑하므로
(별칭이지 두 번째 봉투가 아니다) Gate B는 §7.1의 두 명령에서 `--envelope screen100e4`로
바꿔 그대로 돌린다. 스윕은 아암이 **얼마나 틀려도 되는지**를 바꾸지 않는다.
`screen100`과 마찬가지로 승격 대상이 아니다(§5).

### 7.5 두 번째 스윕 아암: `screen100x` — Xe **폴리시** 허용오차 1e-4 (WP24.2)

`screen100e4`가 CMFD 스윕 종료값을 스윕한다면, `screen100x`는 **같은 부모에서**
다른 노브 하나를 스윕한다: `xe_tol` 1e-5 → 1e-4. 두 아암은 **누적이 아니라 병렬**이다
(둘을 겹치면 측정된 아우터 델타가 두 변경의 합이 되어 스윕이 아니게 된다).
표 행은 `screen100`과 **이름과 `xe_tol` 외 모든 열이 문자 그대로 동일**하며,
`check_sweep_arm`이 이제 `{아암: 스윕 열}` 맵으로 두 아암 모두에 대해 그것을 검사한다.

**왜 아직 한 데케이드가 남아 있었는가.** Xe 술어는 `src/Driver.h:4871`의
`xe_change >= xe_tol_now` 하나다. `xe_tol_now`는 `:4405`에서 결정되는 **스테이징 쌍**이다
— `polishing`이 false인 동안은 느슨한 다리(`xe_tol × staged_xe_mult`), `:5024`에서
느슨한 단계가 합의되어 폴리시가 무장되면 `tol.xe_tol` 그 자체. 즉 screen100의 100×
스테이징이 푸는 것은 **느슨한 다리뿐**이고, 캐스케이드에 넘기는 폴리시 다리는 1e-5로
그대로다 — 프로파일이 실측한 정지점(~1e-4)보다 한 데케이드 아래. 그 대가는 한 스텝이
아니라, 느슨한 단계가 1e-3에서 빠져나온 뒤 **1e-5까지 두 번째 직렬 캐스케이드**를
다시 도는 것이고, 캐스케이드는 커밋된 모든 검색 시행과 T/H 갱신이 재무장한다.
폴리시 다리를 1e-4로 옮기면 그 데케이드가 사라지고, 느슨한 다리는 파생값 그대로
1e-4 × 100 = 1e-2로 따라 움직인다(행에 다시 적지 않는다).

**사라지지 않는 두 바닥 — 그래서 추정은 범위다.**

1. `xe_pending`의 `xe_count + xe_interim_count == 0` 항 (`src/Driver.h:4654-4656`).
   모든 캐스케이드의 **첫 Xe 스텝은 무조건** 취해진다. 허용오차가 아무리 느슨해져도
   228 캐스케이드는 최소 228 스텝을 쓴다.
2. `prev_inner` 센티널 (`src/Driver.h:4877`). 실제로 취해진 Xe 스텝은 `prev_inner`를
   `eigv + 1.0`으로 덮어써서 그 뒤 **최소 한 번의 완전한 flux 재수렴**을 강제한다.
   스텝당 아우터 비용에도 자체 바닥이 있다는 뜻이다.

이 두 바닥을 고정한 채 프로파일 산술을 돌리면 **xe 아우터 1847 → ~1400**,
**총 77 → ~64 outers/sp**. 이것은 **추정이며 측정이 아니다** — 측정이 이 아암의 존재 이유다.

**정확도 비용.** 1e-4는 캐스케이드 스텝 간 Xe 수밀도 변화의 허용오차이고, 곱해지는
것은 캐스케이드 구간의 평형-Xe 반응도 가치다: **statepoint당 ~0.3 pcm**. screen100
봉투의 100 pcm 대비 세 자릿수 아래이므로 스크리닝 노브이지 합격 노브가 아니다
(§5와 같이 승격 대상 아님).

**유지되는 불변식.** `xe_oscillation_floor`는 프로덕션 1e-4에 그대로 두어
`xe_tol <= xe_oscillation_floor`가 **등호로** 성립한다 — 이 노브의 **가장 느슨한
합법값**. 이보다 위로 올리면 진동 검출기가 잡음이라고 부르는 대역 안에서 캐스케이드가
수렴했다고 선언되어, 수렴 판정과 잡음 판정이 서로 다른 "수렴한 캐스케이드"를 갖게 된다.

**정확성 등급 A2, 자체 env/케이스 키 분기.** `armEnvValue()`는 프리셋의 **이름만**
접으므로 `screen100x`는 `screen100`·`screen100e4`와 키가 갈라지고 세 아암은 캐시가
섞이지 않는다. Gate B 봉투는 **부모와 같다** — `tools/gate_b_envelope.py`가 이 이름을
`screen100` 봉투로 **별칭** 매핑한다(두 번째 봉투가 아니다). 스윕은 아암이 **얼마나
빨리** 수렴하는지를 바꿀 뿐, **얼마나 틀려도 되는지**를 바꾸지 않는다.

#### 238 레시피 — KNGR 단일 warm 1회 + hot 3회, 네 아암

네 아암을 **같은 레시피에서 프리셋 이름만 바꿔** 돌린다. `--set-unset` 세 줄은
§7.1과 **같은 이유로** 모든 아암에서 필수다(그래야 아암 간 키가 비교된다).
아암 D는 C와 같은 행에 `RASBERY_A2_PREV_INNER=carry`를 더한 것으로, 위 바닥 (2)를
직접 겨냥한다 — 행이 아니라 **환경 노브**이므로 별도 이름 없이 케이스 키에 접힌다.

```
# A) strict
python tools/run_single_gpu_batch.py --gpu 0 --batch-width 1     --set RASBERY_FIDELITY=strict     --set-unset RASBERY_STAGED_FLUX_TOL --set-unset RASBERY_STAGED_XE_TOL     --set-unset RASBERY_STAGED_LOOSE_SETTLE     -- RASBERY --rasi <kngr_238.json> --raso E:/run/wp24/strict --batch-mode 1

# B) screen100          (RASBERY_FIDELITY=screen100,  --raso .../s100)
# C) screen100x         (RASBERY_FIDELITY=screen100x, --raso .../s100x)
# D) screen100x + carry (C에 --set RASBERY_A2_PREV_INNER=carry, --raso .../s100x_carry)
```

각 아암은 **warm 1회 + hot 3회**. warm은 버리고 hot 3회의 **중앙값**을 인용한다
(cold 캐시 편차가 아우터 델타보다 크면 스윕이 측정하는 것은 캐시다).

**읽을 영수증** (§7.1 표에 더해)

| 항목 | 어디서 | 무엇을 말하는가 |
|---|---|---|
| `outers_by_phase.xe` | 아우터 예산 영수증 | 이 아암이 겨냥한 바로 그 덩어리. 1847 → ~1400이 기대치 |
| `cascade.xe_steps_per_cascade` | 〃 | 캐스케이드당 스텝. 바닥 (1) 때문에 **1.0 아래로는 못 간다** |
| `cascade.residual == 0` | 〃 | 캐스케이드 항등식. 0이 아니면 위 두 수치의 해석 자체가 무효 |
| `settle.outers_loose` / `settle.outers_polish` | 〃 | 데케이드가 실제로 사라졌는지. polish 몫이 줄지 않으면 이 아암은 실패다 |
| `xe.budget_exhausted == 0` | 〃 | 예산 소진으로 끝난 캐스케이드가 있으면 허용오차가 아니라 예산이 판정을 내린 것 |
| `staged_relapses` | `SPTELEM SUMMARY` | 폴리시 진입 후 느슨한 단계로 되돌아간 횟수 |
| `counterfactual.loose_hit_rate` | 아우터 예산 영수증 | 느슨한 다리가 실제로 걸린 비율 — 스테이징이 살아 있다는 증거 |

**합격 판정**

```
python tools/compare_master_rasbery.py <MAS_SUM> E:/run/wp24/s100x/*.h5     -o E:/run/wp24/s100x_gateb --envelope screen100 --all-steps
python tools/gate_b_pin_rms.py E:/run/wp24/s100x/*.h5 <kngr_mas_ppi_boc.txt>     --all-steps --envelope screen100
```

기준: **100 pcm 이내, 핀 1 %** (`--envelope screen100x`도 같은 봉투로 해석되지만,
세 아암을 한 표로 비교할 때는 봉투 이름을 하나로 적어 두는 편이 읽기 쉽다).

**처리량**: **8×M8 median-of-3**을 인용 수치(c/h)로 삼고, **8×M16은 보조**로만 적는다
(§2.6의 boron_bracket 결함 때문에 M16은 인용 대상이 아니다).

**즉시 중단 신호**

- `flux_limit_retries != 0` — flux 한계 재시도가 발생하면 아우터 감소는 허용오차가
  아니라 한계 도달의 결과다.
- `staged_relapses > 0.2 × search_trials` — 폴리시 진입이 안정적이지 않다는 뜻이고,
  이 아암이 옮긴 것이 바로 폴리시 다리다.
- `cmfd.sweeps_per_outer > 4.4` — `_ncmfd = 5` 캡에 붙기 시작한 것이므로 아우터당
  비용이 스윕 캡에 지배되어 outers/sp 비교가 무의미해진다.

---

---

## 8. 예상 아우터 감소 — **추정이며, 측정이 아니다**

프로파일(`Driver.h` A2 주석 블록, `tools/outer_profile.py` on kngr_238): 12,017
아우터 / 35 statepoint 중 8,579(**71.4 %**)가 평형-Xe 캐스케이드, 1,508(12.5 %)이
임계 검색. 구조는 하나다 — 228개 캐스케이드 × 10.65 정착 Xe 스텝 × 스텝당 3.53
아우터의 flux 재수렴, 그리고 캐스케이드는 **모든** 커밋된 검색 시행과 T/H 갱신이
재무장한다. 붕소 시행 하나가 ~38 아우터를 쓰고 실행당 137개가 취해진다.
현행 캠페인 기준선은 4,377 아우터 / 35 sp = **125/sp** (m64: 8,415 / 51 = 165/sp).

| 레버 | 추정 효과 | 근거의 성격 |
|---|---|---|
| flux L2 2데케이드 + keff 1데케이드 | 스텝당 3.53 → ~1.5 아우터 | 정적 추론 |
| search 1데케이드 (+ rod cap/floor) | 커밋 시행 ~절반 → 재무장 캐스케이드 감소 | 정적 추론 |
| `staged_search_margin` 4 → 2 | loose |dk| 캡 2배 완화 (§2.5) | **미측정** (블록 33의 +0.11 %는 search_tol이 여전히 캡이었으므로 이를 bound하지 못한다) |
| Xe 1데케이드 | 캐스케이드당 스텝 1.5–2개 감소 | 정적 추론 |
| `_epsl2` 1e-6 → 1e-5 | 아우터 **수**가 아니라 아우터 **비용** −30~40 % | 정적 추론 |
| `boron_bracket` | **−1.76 %** | **측정** (238 블록 33) |

**종합 추정: KNGR 125/sp → 45–60, m64 165/sp → 60–80 (아우터 2.0–2.7배 감소),
여기에 `_epsl2`의 아우터당 비용 −30~40 %가 곱해진다.** 배치가 GPU-time-bound이므로
그것이 c/h 배수다. 다시 강조하지만 **이 표에서 측정된 값은 boron_bracket의
−1.76 % 하나뿐이다.**

### 8.1 재측정해야 하는 숨은 가정

`outerSegmentBudget`(기본 8)의 경제학이 프리셋 때문에 바뀐다. WP14는 세그먼트가
CMFD 결정이 슬롯을 HOST 단계(압도적으로 Xe)에 넘기기 때문에 ~3.30 아우터에서
멈춘다고 측정했다. flux 허용오차가 느슨해지면 `flux_converged`가 **더 일찍** 켜지고,
따라서 Xe가 더 일찍 발화하고, 따라서 세그먼트가 **더 짧아진다**. 디바이스-아우터
아암의 이점이 아우터 수와 같은 속도로 줄어들 수 있다. **현행 예산의 측정된 이득을
screen100 영수증으로 그대로 이월하지 말 것.**

---

## 9. 지배적 리스크

statepoint 하나의 오차가 아니라 **연소 사슬**이다. 모든 statepoint의 수렴 상태가
다음 statepoint의 초기조건이고, statepoint당 10–20 pcm의 허용오차 오차는 상쇄될
의무가 없다 — 연소 궤적을 편향시켜 EOC의 dkeff·CBC 드리프트로 나타날 수 있다.

따라서:

1. **100 pcm 게이트는 사이클 평균이 아니라 statepoint별로** 평가한다.
   `compare_master_rasbery.py`가 max|delta|를 쓰는 것이 그 뜻이다.
2. **핀 예산은 선형으로 쓸 수 없다.** §6의 절벽 참조. screen100은 KNGR 스윕에서
   채택해 다른 덱에 이식하는 것이 아니라 **덱별로 검증**해야 한다.
3. **재시작/셔플 statepoint**(`primeXeDamping`)가 완화된 Xe 허용오차의 역사적
   실패 지점이다 — 잘못된 축방향 분기(AO +0.314 vs MASTER −0.054). screen100을
   재시작 덱에 처음 돌릴 때는 댐퍼 트리거를 반드시 다시 확인한다. §2.2의 절대
   바닥 고정이 이 실패 모드를 겨냥한 것이지만, 겨냥했다는 것이 측정했다는 뜻은
   아니다.
4. **핀 헤드룸이 좁다 — 그리고 좁은 쪽은 CBC가 아니라 pin max다.** 봉투는 네 열
   모두 MASTER 상대 **절대** 한계이고 프로덕션 기준선이 이미 그 안에 있다(§6).
   CBC는 프로덕션 합격선 15.4에 98.0 pcm의 상 18.1 ppm을 더한 33.5 ppm이므로
   keff 열과 같은 예산을 받지만, `pin_max`는 지시가 1 %로 못 박아 합격선 0.80 % 위에
   **0.20 pp**만 남는다. 즉 실질 게이트는 붕소가 아니라 **핀 최대**이고, 그
   열의 1 %는 이 트리의 어떤 측정에서도 유도된 값이 아니다(측정된 staged 스캔에
   프로덕션 폴리시에서 이미 1.12–1.16 %인 아암이 있다). screen100 행은 그래서
   덱별 Gate B를 기다리는 **후보**다.
5. **트래커에 게이트로 올려야 하는 두 항목 — 산문이 아니라 조건이다(리뷰 수정).**
   * **8×M16 + boron_bracket의 128 중 5 무성 워커 실패**(§2.6). 행은
     `boron_bracket = ON`을 들고 있고 배치 폭 제약은 **주석과 런북에만** 있다.
     바이너리도 평가기도 계약 테스트도 8×M16 screen100 배치를 막지 않는다.
     블록 34(boron_bracket-OFF 8×M16 대조군)가 **screen100을 M16으로 쓰기 전의
     게이트**다.
   * **rod-crit 계열(iSMR / CY)의 자체 Gate B**(§2.4b). 지금은 행이 두 열을
     프로덕션 값으로 되돌려 두었으므로 안전하지만, 되돌린 이유 — 그 덱들에서
     채점되는 스칼라 열 전부가 구조적으로 무력하다는 것 — 은 그대로다.
     **rod 덱에서 실패할 수 있는 열을 가진 Gate B를 만드는 것이 그 두 열을
     올리기 전의 게이트다.**

---

## 10. 변경 파일

| 파일 | 내용 |
|---|---|
| `src/FidelityPreset.h` | **신규.** 프리셋 표, `SolveTolerances`, 환경 읽기, 미지 이름 판정 |
| `src/RunContract.h` | `processStagedFluxMult/XeMult`, `detectedPhysicsFidelity`가 프리셋을 본다 |
| `src/CaseFidelity.h` | `CaseFidelity::preset`/`presetSpec()`/`tolerances()`/**`searchPolicy()`**(행이 없으면 `environmentSearchPolicy()` — 지운 프리셋이 검색 절반도 지운다)/`presetToken()`; `FidelityRequest::preset`; `resolveCaseFidelity` 단계 0; strict×preset 모순 거부 |
| `src/Scheduler.h` | `Schedule::rodcrit_search_cap`; `criticalSearchTolerance(scale, **cap**)`; **`presetSearchPolicy()`**(행 → 검색 정책, per-case/per-process 공용); **`environmentSearchPolicy()`**(행이 **없는** raw 환경 읽기)와 `processSearchPolicy()`(= 그것 + 행)의 분리; **`kStagedSearchMarginBuiltIn`**(내장 4.0의 유일한 자리) |
| `src/Driver.h` | `SolverContext::tolerances` + **`ctx.search_policy = _fidelity.searchPolicy()`**; 일곱 소비자 배선; `kArmEnv` + `armEnvValue`(**검색 노브 다섯 개도 유효값으로 접는다**); `[RASBERY][FIDELITY]` schema 2(실행에서 출력 + `resolved_entry0`); `[CASE]` schema 7; 죽은 `setEpsl2(1.0e-6)` 삭제 |
| `src/EvaluatorServer.h` | `fidelity_preset` 파싱/wave 기본값/promote 클리어/hello/per-case 줄 |
| `src/main.cpp` | 미지 프리셋 거부; `[PHYSICS_MODE]`에 `fidelity_preset` |
| `include/chiffon/BatchLightResult.h` | light JSONL `fidelity_preset` |
| `src/GpuSlotControl.h` | `DeviceScheduleParams::rodcrit_search_cap` (호스트 쌍둥이, reset-only + `TODO(WP24-device-search)` 마커) |
| `tools/exact_audit.py` | 프리셋 표를 소스에서 **지연 로드**(import 시 읽지 않는다); `derive_declared_fidelity`; **`receipt_preset()`**; `NON_STRICT_ENV_KEYS` |
| `tools/gate_b_envelope.py` | **신규.** 두 봉투(production은 **합격 열**, screen100은 **기준선+증분**, ppm 33.5), `--envelope`, `headroom()`, `scored(..., pinned)`, `report(..., pinned)`(재지 않았거나 **실패할 수 없는 열만** 잰 실행은 `NOT SCORED`/exit 2), `UNJUDGED_COLUMNS` |
| `tools/gate_b_pin_rms.py` | argparse, `--envelope`, `--step`/**`--ppi-step STEP=PATH`**/`--all-steps`(**드리프트 보고 전용**), 비-0 종료, `BOC pin:` 줄 하위호환 |
| `tools/compare_master_rasbery.py` | `--envelope`, 판정, 비-0 종료, .md에 봉투 이름, 빈 열 ValueError 가드, **모든 행에서 정확히 0인 델타 열을 `pinned`로 신고** |
| `tools/case_key.py` | `effective_fidelity()`가 프리셋 행을 읽는다(`exact_audit.FIDELITY_PRESETS`) — 이것이 없으면 파이썬 미러가 screen100 실행마다 바이너리와 **다른 케이스 키**를 계산한다 |
| `tools/run_multi_gpu_batch.py` | `fidelity_preset`을 per-case·워커·후보·TOTAL 영수증에 접는다 |
| `tools/promotion_gate.py` | A/B 위생 검사가 `fidelity_preset`을 `policy` 옆에서 비교; 아암이 보고한 프리셋과 사람이 적은 `grade`의 교차검사 |
| `tools/test_promotion_gate.py` | 위 두 검사와 `receipt_preset()`·프리셋 라운드트립 감사의 **픽스처**(양쪽 부재는 통과, 한쪽 부재는 거부, 낡은 바이너리는 거부) |
| `tools/test_search_policy_contract.py` | 다섯 노브의 "한 자리"가 `environmentSearchPolicy()`로 이동한 것을 반영 |
| `tools/run_single_gpu_batch.py` | 오타난 프리셋의 `ValueError`를 하네스 메시지로; `LaunchPlan.declared_preset` + `declared_preset_from_env()`; **`check_run_receipts()`가 요청한 프리셋과 자식이 출력한 프리셋의 일치를 요구**(필드 부재 = WP24 이전 바이너리 = 거부) |
| `tools/fake_rasbery_child.py` | `RASBERY_FIDELITY`를 존중; `fidelity_preset` 출력; `[CASE]` schema 7 |
| `tools/test_fidelity_preset_contract.py` | **신규.** 7부(+ polish-invariant) + 25개 네거티브 컨트롤; WP24.2에서 `check_sweep_arm`이 `{아암: 스윕 열}` 맵으로 일반화(`screen100e4`→`cmfd_sweep_epsl2`, `screen100x`→`xe_tol`)되고 아암별 불변식·네거티브 컨트롤 3종 추가 |
| 기존 계약 테스트 9종 | WP24가 의도적으로 옮긴 핀을 이유와 함께 갱신 |
