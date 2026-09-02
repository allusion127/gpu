# A2 아우터 저감 설계서 — m64 165 outers/sp 를 절반으로

- 대상 브랜치: `codex/exact-throughput-campaign` @ `d515dde`
- 작성일: 2026-09-02
- 성격: 설계 문서. **빌드/실행 없음**, 소스 정독·git·정적 스캔만으로 작성.
- 정확성 어휘: **B0** = 비트 동일, **N1** = 결정론적 궤적 변경(Gate A/B 통과 필요), **A2** = 수용 허용오차(정책) 변경(Gate A/B + 별도 브랜치·롤백).
- Gate A(동결 궤적 대비): keff 1.905 pcm / boron 15.309 ppm / AO 0.013 / pin 1 %
- Gate B(MASTER 대비): pin RMS 0.238 % / max 0.80 %
- 프로덕션 환경: `RASBERY_STAGED_FLUX_TOL=50` `RASBERY_STAGED_XE_TOL=1000` `RASBERY_STAGED_LOOSE_SETTLE=1` `RASBERY_GPU_WIEL_FOLD=chunked`, XE_ANDERSON on(single), `OUTER_GRAPH` conditional WHILE, `PC_MODE=decart`

---

## §1. 아우터 예산 — 125/165 outers/sp 는 어디로 가는가

### 1.1 측정된 골격

| 항목 | KNGR 단일 덱 | m64 (APR1400 CY02 장전, GA 타깃) |
|---|---|---|
| statepoint | 35 | 51 |
| outers | 4,377 | 8,415 |
| outers / sp | 125.1 | 165.0 |
| CMFD sweeps | 18,627 | — |
| sweeps / outer | **4.256** (예산 5) | — |
| BiCGSTAB iters | 74,508 | — |
| iters / sweep | **4.000** (= 1+`_nmaxbicg`, 산술 유도값) | — |
| 단일 덱 wall | 9.75 s | — |
| 배치 8×M16+MPS | — | **1,321–1,440 c/h** (sm 96 %, 커널시간 이상치의 ≈92 %) |

**m64 에는 원인 귀속(cause attribution) 영수증이 존재하지 않는다.** 아래의 모든 분해는 KNGR 프로파일이며, m64 로의 이전은 statepoint 수·trial 수 스케일링에 의한 추정이다. 이것이 본 설계서 전체에서 가장 큰 단일 불확실성이다(§5-1).

### 1.2 원인별 분해 (프로덕션 설정, `docs/A2_OUTER_REDUCTION_20260829_KO.md` §4.3)

4,614 outers / 35 sp = 131.8/sp 기준:

| 버킷 | outers | 비중 |
|---|---|---|
| **xe** | 3,184 | **69.0 %** |
| search | 707 | 15.3 % |
| initial (SolveLoop 진입) | 347 | 7.5 % |
| settle | 290 | 6.3 % |
| th | 86 | 1.9 % |
| fallback | 0 | 0 % |

xe 버킷의 내부 구조: **228 cascades × 5.24 steps/cascade × 2.67 outers/step = 3,184** (0.2 % 이내로 닫힘).

### 1.3 cascade 항등식 — "search 15.3 %" 는 과소 계상이다

cascade 는 정확히 세 지점에서 재장전된다: SolveLoop 진입(`Driver.h:3844/3856`), 커밋된 T/H 스텝, 커밋된 search trial(둘 다 `Driver.h:4681`, 동시 발생 시 1회만 증가).

```
cascades = solve_loops + th_updates + search_trials − th_search_coincident
228      = 69         + 126        + 137           − 104
```

cascade 당 실측 13.96 outers 를 부과해 **인과적으로** 재귀속하면:

| 원인 | cascades | outers | 런 대비 |
|---|---|---|---|
| search trial | 137 | ~1,913 | **41.5 %** |
| SolveLoop 진입 | 69 | ~963 | 20.9 % |
| 비동시 T/H | 22 | ~307 | 6.7 % |

즉 boron trial 1회의 실제 청구서는 버킷상의 ~5 outers 가 아니라 **all-in ~14 outers** 다(스테이징 이전에는 ~38; `docs/WP9D_SEARCH_TRIALS_20260831_KO.md` §2.2 는 아직 38 을 인용하고 있어 낡았다).

### 1.4 두 개의 구조적 바닥

**(a) Xe 스텝의 2-outer 바닥.** `Driver.h:4426-4432` — 소스 확인:

```cpp
if (xe_change >= xe_tol_now || (xe_once_mode && !xe_once_done)) {
    prev_inner  = eigv + 1.0;
    clean_iters = 0;
    continue;
}
```

`prev_inner = eigv + 1.0` 오염은 다음 아우터의 `|prev_inner − eigv| < keff_tol_now` (`Driver.h:4126`, 확인) 를 구조적으로 실패시킨다. 1,192 스텝 중 ~964 가 허용오차를 넘지 못해 오염되므로 **≥1,928 outers(런의 ≥42 %)** 가 이 한 문장의 산물이다 — 실측 2.67 에 대해 바닥이 2.

동일 관용구가 네 지점에 있다: Xe 스텝(4429), interim(4441), settle gate(4489), polish 전이(4551).

**반론(설계 D2 제기, 판정단이 유효로 인정).** 프로덕션 스테이징에서 cascade 는 `prev_xe_change >= xe_tol_now = 1e-3` 인 동안만 계속된다. Xe worth ~2,500–3,000 pcm 에 대해 1e-3 상대 변화는 k 로 ~2.5e-5, 즉 loose keff_tol 5e-6 의 ~5배다. 따라서 **오염이 없었더라도 대다수 스텝에서 수렴 판정은 어차피 실패했을 것**이며, "≥42 % 가 순수 인공물" 이라는 주장은 허용오차 산술을 견디지 못한다. 실제 낭비는 XS 가 움직이지 않은 두 지점(settle gate, polish 전이)에 있다. 본 설계서는 이 반론을 채택하되 사실 확인은 무료 카운터로 넘긴다(S0).

**(b) 세그먼트 길이는 예산이 아니라 물리가 정한다.** 블록 19 디바이스 영수증: 4,377 outers / 1,326 segments = **3.301 outers/segment**. 종료 사유 — `flux_converged` 1,086 (81.9 %), `segment_budget` 124 (9.4 %), `negative_flux` 115 (8.7 %), `none` 1.

81.9 % 가 "CMFD 판정이 수렴했고 호스트가 Xe 스텝을 돌려야 해서" 끝난다. 따라서 `RASBERY_GPU_OUTER_SEGMENT_MAX` 를 8 이상으로 올리는 것은 **원리적으로 무익**하다. 유일한 구조적 해법은 Xe 핸드오프를 디바이스로 올리는 것인데, `Driver.h:4029-4033` 이 `s.xe_pending = 0; s.th_pending = 0; s.xe_interim_l2 = 0; s.xe_once_mode = 0` 으로 하드와이어하여 디바이스 결정기를 **의도적으로 굶기고** 있다. 그래서 디바이스는 RequeueOuter/Converged/Fatal 만 낼 수 있고 영수증에 `none:1` 이 찍힌다. (`CmfdOuterKernel.h:551` 의 Xenon 분기는 작성되어 있으나 도달 불가 — 소스 확인.)

### 1.5 아무도 건드리지 않은 두 번째 분모: sweeps/outer

`Driver.h:5023-5025` — **세 setter 각각의 유일한 호출부**임을 전수 확인:

```cpp
cmfd_solver.setNcmfd(5);
cmfd_solver.setEpsl2(1.0e-6);
cmfd_solver.setEshift(0.04);
```

`drive()` 가 반환하는 `errl2` 가 곧 아우터가 시험하는 `residual` 이다. A2 스테이징은 **소비자**(`flux_tol_now` = 5e-5, LOOSE)만 느슨하게 했고 **생산자**(`_epsl2` = 1e-6)는 그대로 두었다 — 즉 모든 LOOSE 아우터 안에 50× 의 생산자/소비자 간극이 있다. 실측 4.256 sweeps 대 예산 5 는 drive 가 대개 **허용오차가 아니라 예산으로** 끝난다는 뜻이다.

GPU 시간 ≈ outers × sweeps/outer × ~8 launches 이고 커널은 소격자 지연 한계(188 SM 상에서 34–133 blocks)이므로, **sweeps/outer 는 outers 와 곱해지는 독립 인자**다. 지금까지 캠페인은 첫 인자만 공격해 왔다.

### 1.6 영수증이 보여주지 않는 손실

`BICGCMFD.cpp:978` 부근의 음속 재시도 규칙 — 음의 플럭스를 낸 sweep 은 예산을 소모하지 않고 최대 `20*_ncmfd` = 100회까지 재시도된다. 이 sweep 들은 실제 런치이지만 **어떤 outer-count 영수증에도 나타나지 않는다.** `icmfd_done − sweeps_done` 은 `CmfdSweepIO` 에 이미 실려 있고 영수증까지만 도달하면 된다.

### 1.7 m64 로의 이전 (추정)

m64 는 statepoint 1.457×, outers 1.922× 다. 경계·trial 형 레버는 계수 1.457/1.922 = **0.758** 로 축소되고, cascade 내부 레버(Xe 스텝 수, sweeps/outer)는 거의 등가로 이전된다. 165 > 125 라는 사실 자체가 m64 가 statepoint 당 더 많은 cascade 내부 작업을 쓴다는 뜻이며 이는 경계형 레버에 불리하고 cascade 내부 레버에 유리하다 — **다만 이를 확정할 m64 영수증이 없다.**

---

## §2. 네 설계 — 순위, 판정 평균, 판정단이 지목한 최대 리스크

### 1위: D2 — 내부/외부 허용오차 결합(inexact-Newton) + 수용 재조정 · 평균 **7.83** (8 / 8 / 7.5)

`_epsl2` 를 `flux_tol_now / margin` 으로 스테이징(A2), nmax 사다리는 **구조적 불가로 판정하고 만들지 않음**(정확 그래프 키 + 배치 균일성 거부), settle gate·polish 전이의 오염 재조정, interim-Xe 를 이름으로 폐기.

- **최대 리스크(3인 합의): Gate A 경로 실패.** `search_tol` = 2e-5 = 2 pcm 이 Gate A 예산 1.905 pcm **보다 크므로**, 공표 keff 는 근(root)이 아니라 **탐색 경로의 함수**다. boron_bracket(실측 −1.76 % outers, 2.27 pcm 실패)과 warm start(2.36–2.40 pcm)가 같은 기전으로 죽었다.
- **부차 리스크:** 헤드라인 18 % 가 자체 산술을 못 견딤(polish 전이 절감이 자신의 Xe-worth 논증에 의해 상쇄됨). outers 와 sweeps/outer 를 **분리 가능한 인자로 곱함** — 실제로는 같은 residual 로 결합되어 있어 느슨한 drive 가 outers 를 되사올 수 있고, 본 트리에 선례가 있다(`_nmaxbicg` 6→4 가 i-SMR outers +6 %). GPU 시간 −38 % 는 CMFD 비중 54.6 % 를 무시한 합성으로 실제 −29 % 이하.
- **판정단이 보증한 부산물:** `RASBERY_BICG_NMAX` / `RASBERY_BICG_EPS` 가 `trajectory::kArmEnv` 에 **없다**(본 문서에서 49개 엔트리 전수 확인 — 없음). 두 상수는 BICGCMFD 생성자에서 읽히고 궤적을 움직이므로 **오늘 서로 다른 내부 예산의 두 런이 같은 case key 를 공유**하고 캐시된 답이 오배송될 수 있다. 채택 여부와 무관한 선행 커밋.

### 2위: D3 — 섭동 케이던스 붕괴 · 평균 **7.00** (7.5 / 6 / 7.5)

L1 정직한 Xe 후 재구동(오염 제거), L2 디바이스 상주 Xe 핸드오프, L3 boron 근의 EFPD 외삽, L4 T/H 지연, L0 수용점 종단 재수렴.

- **최대 리스크:** L1 이 자기 물리 논증이 기술하지 않는 모집단을 겨냥함. 오염은 `xe_change >= xe_tol_now` 일 때만 발화 — 즉 **XS 가 가장 크게 움직인 스텝**이며, 한 번의 drive 로 |Δk| 를 keff_tol 아래로 넣기 가장 어려운 스텝이다. 설계는 그 반대("k 가 keff_tol 보다 작게 움직인 스텝")를 가정해 50–75 % 적중률을 산정했다.
- **판정단 지적 오류:** "디바이스 트윈에는 오염이 없다 → 한 지점에서만 불일치" 논거는 범주 오류다. `CmfdOuterKernel.h:551` 의 Xenon 은 Xe 스텝 **이전의 디스패치**이지 스텝 이후 분기가 아니며, 해당 파일에 스텝 이후 Xe 분기는 존재하지 않는다(확인). L2 의 디바이스 Xe phase **본체는 미구현**이고, `drainXeCommit` 생략은 "호스트 배열이 권위" 라는 명시 불변식을 뒤집는다. L2 의 B0 주장은 자기모순이다 — `xe_pending` 을 풀면 디바이스 Xenon 경로가 곧 L1 이 된다.
- **가장 값진 부분: L0.** 수용된 boron 을 게이트보다 훨씬 낮은 허용오차로 종단 결정론적 재수렴(이미 `Driver.h:4741` 의 best-fallback 형상이 정확히 그 모양). **세 판정단 전원이 "이것을 먼저 배치하라"고 독립적으로 결론.**

### 3위: D1 — statepoint 시드 + Wielandt shift 스케줄 · 평균 **6.83** (7 / 6.5 / 7)

flux/keff/boron 의 1차 EFPD 외삽 + 선발 Xe 스텝 + residual 연동 적응 shift.

- **최대 리스크: 물리 배치 오류.** 시드가 DEPLETION 블록 **앞**에 놓이는데, `PredictorStep` 이 live `Phif` 를 `_flux_bos` 로 스냅샷해 거기서 감쇄한다. 수렴한 적 없는 외삽 형상으로 감쇄하면 **핵종 밀도가 움직인다** — 이는 궤적 변경(N1)이 아니라 답의 변경이며 Gate B 가 LOW 가 아니라 MEDIUM 이 된다. 한 줄 이동으로 교정 가능하나, 라인 정밀도에 신뢰 전부를 건 설계에서 가장 하중이 큰 라인이었다.
- **부차:** 선발 Xe 스텝이 `ApplyXeEquilibrium` 로 `iden[I135]/[Xe135]/[Xe135m]` 를 덮어쓰므로 `xenon_transient` 덱에서는 물리 변경이며 가드가 없다. boron 항(헤드라인의 절반)은 payoff 가 **연속이 아니라 이진**(외삽근이 2e-5 공 안에 들어가야만 trial 이 제거됨)인데 연속으로 가격되었다.
- **살아남는 것:** Wielandt shift 스케줄(~25줄; `BICGCMFD::eshift()` 접근자 기존재, `kEshift` 가 매 런치 `sweep_in` 스칼라 블록에 이미 탑승, `SweepGraphCapacity` 키 아님 → 재캡처 0·전송 0)과 Day 0 무비용 킬 테스트.

### 4위: D0 — FAA 플럭스 공간 안전장치 Anderson · 평균 **6.33** (6 / 6 / 7)

Xe-AA 안전장치 4종을 플럭스 반복자에 이식, norm lock 추가, 세그먼트 내부 디바이스 트랜잭션 6노드.

- **최대 리스크: 수용률 붕괴로 인한 실측 무효과.** CMFD 고유치 문제는 φ 에 대해 동차이므로 고정점 Jacobian 이 반복자 방향으로 단위 고유치를 가지고 차분 열이 그 방향에서 거의 영 — Anderson 이 병적으로 나쁜 고전적 경우다. 설계는 이를 **감지**하여 norm lock 을 발명했으나 하류 증상만 고치고 Gram 영방향은 그대로 둔다.
- **두 번째 리스크(정확성 분류 문제):** `prev_inner <- eigv_cand` 커밋은 **수용 판정 자체**(`flux_converged` 의 keff 절반)를 바꾼다. 어휘상 이는 N1 이 아니라 **A2** 이며, 동시에 outers 를 실제로 줄이는 유일한 기전이기도 하다 — 즉 측정된 절감이 "가속" 이 아니라 "느슨해진 수렴 판정" 일 수 있다.
- **자체 산술의 자기부정:** HALT_GUARD 로 세그먼트 종료 아우터에서 무조건 무효화(≈30 % 의 아우터 접근 불가) + 스스로 인정한 "3,184 Xe outers 중 2,384 는 오염 바닥이라 AA 가 못 건드림". 12 % 를 맞추려면 남은 것의 ~26 % 를 걷어야 한다. 정직한 중심값 5 %, 이는 설계 자신의 PARK 분기.
- **최고의 아이디어:** `enqueueOuterTransition` **이후** 배치 → HALT_GUARD 로 세그먼트 종료 아우터에서 자동 무효화 → **공표되는 어떤 플럭스도 외삽값이 아님.** Gate B LOW 주장을 방어 가능하게 만드는 유일한 장치.

---

## §3. 통합 프로그램 — 스파이크 순서, 플래그/영수증, 정확성 등급, 누적 효과

### 3.0 통합의 지배 원리

세 판정단(총 12표)이 설계와 무관하게 수렴한 결론 하나:

> **`search_tol` = 2e-5 (2 pcm) 가 Gate A 예산 1.905 pcm 보다 크므로, 공표 keff 가 탐색 경로의 함수인 한 어떤 초기추정·수용판정 레버도 측정 불가능하다.** boron_bracket(−1.76 % outers, 2.27 pcm), warm start(2.36–2.40 pcm) 는 세 번의 우연이 아니라 **하나의 실패 기전**이다.

따라서 프로그램의 첫 항목은 절감 레버가 아니라 **측정 가능성의 복원**이다. 이를 뒤로 미루면 이후 모든 스파이크의 Gate A 실패가 "경로 차이" 인지 "근 차이" 인지 판별 불가능해진다.

### 3.1 스파이크 순서 (각 ≤ 2일, GPU0 전용, 결과는 E:, 로컬 계산 금지)

| # | 스파이크 | 등급 | 킬 기준 |
|---|---|---|---|
| **S0** | 계측만 — 영수증 확보 | **B0** | m64 xe 버킷 < 50 % → 설계 전면 재조준 |
| **S1** | L0 수용점 종단 재수렴 | **N1**(+재동결) | outers 비용 > 4 % 또는 아츰 간 keff 산포가 1 pcm 밑으로 안 내려감 |
| **S2** | 무코드 노브 스윕 | **A2** | `staged_relapses > 0.2 × search_trials` |
| **S3** | `_epsl2` 스테이징 (search-blind 우선) | **A2** | K0: 예산소진 drive > 70 % → 사망 |
| **S4** | L1 정직한 재구동 (LOOSE 한정) | **N1** | S0 반사실 카운터 적중률 < 0.30 |
| **S5** | boron 근 EFPD 외삽 + statepoint 시드 | **N1** | 외삽 x0 오차가 현행 대비 3× 이상 안 줄면 |
| **S6** | Wielandt shift 2단 / FAA | 보류 | S3–S5 결과 확인 후 재평가 |

---

#### S0 — 계측 (Day 0, 무코드 + ~10줄 영수증, B0)

프로덕션 환경 + `RASBERY_STATEPOINT_TELEMETRY=1` 로 KNGR 1덱 + m64 1덱. **타이밍 아츰에서 분리**(A2 §6.3: 텔레메트리가 단일 덱 wall ~2.5 %, 배치는 파괴).

신설 영수증(모두 가법적, `schema_version` 유지):

- `cmfd_sweep_exits` — state 1(수렴) vs state 3(예산 소진) 분리. **S3 의 존폐를 결정하는 단 하나의 수치.** 반드시 S0 에서 확보한다. 설계 D2 는 이를 Day 2 패치에 배치했는데, 그러면 자신의 킬 기준을 볼 수 없다.
- `negative_retry_sweeps` = Σ(`icmfd_done − sweeps_done`) — 오늘 어떤 영수증에도 없는 순손실 버킷.
- `sweeps_per_outer` — statepoint 별.
- `bicg_early_exit_frac`, `overrun_iterations` — `BackendCounters` 에 이미 유지되나 읽히지 않음.
- `xe_first_outer_would_converge` — **반사실 카운터.** Xe 스텝 직전 `pre_xe_eigv` 를 포착하고, 오염된 다음 아우터에서 `|pre_xe_eigv − eigv| < keff_tol_now && residual < flux_tol_now` 를 **평가만 하고 사용하지 않는다.** 래더가 읽지 않으므로 digest 불변 = B0. **S4 를 작성하기 전에 S4 의 가격을 정확히 매긴다.**
- `poison_suppressed` / `poison_converged_next` — §1.4 의 미해소 반론을 직접 판정하는 쌍.

무런 오프라인 작업: 기존 영수증 필드(`search_first_x`, NO.= 라인의 PPM/EFPD, `search_iterations`)로 2점 EFPD 근 외삽을 35/51 statepoint 전체에 소급 적합 → **S5 의 가격을 런 없이 산정.**

동시 해소 항목: 4,614 대 4,377 outers 불일치(본 설계서의 모든 비중이 여기 의존), **m64 최초 원인 귀속.**

**킬:** m64 원인 분해에서 xe 버킷이 50 % 미만이면 §1 의 전제가 m64 로 이전되지 않는 것이므로 전면 재산정.

---

#### S1 — L0: 공표 답을 탐색 경로에서 분리 (N1 + 재동결)

`RASBERY_SEARCH_FINAL_RECONVERGE`. CONVERGED 분기(`Driver.h:4725` 부근)에 `|k_res| > search_tol/4` 일 때 종단 단계를 추가: 이미 측정된 기울기로 1회 secant 보정 → 적용 → `ReconvergeFlux(ctx, eigv, FALLBACK_RECONVERGE_ITER, keff_tol, flux_tol, total_outer)` → 재측정. 형상은 `Driver.h:4741` 의 best-fallback 을 그대로 재사용한다(닫힌 escape 집합: Xe 없음, T/H 없음, search 없음, 자체 디바이스 위임 보유).

- 비용: **+2 % outers** (오늘 fallback 버킷 = 0 이므로 순증).
- 효과: 모든 아츰의 landing ball 을 ~2 pcm → ~0.5 pcm 으로 축소.
- **거버넌스 사실:** L0 자체가 N1 이며 모든 statepoint 의 공표값을 움직인다. 따라서 **Gate A 를 통과하지 못할 개연성이 산술적으로 높고, 통과 여부와 무관하게 기준 궤적 재동결(re-freeze)을 수반한다.** 이는 정책 행위이므로 별도 브랜치·롤백 경로를 A2 와 동일하게 운용하고, **이후 모든 스파이크는 새 동결 궤적에 대해 게이트한다.**
- 부수 효과: 이미 작성되어 있고 0.37 pcm 로 기각된 `RASBERY_SEARCH_BORON_BRACKET`(−1.76 % outers)이 재측정 가능해진다.

**킬:** outers 비용 > 4 %, 또는 재수렴 후에도 아츰 간 keff 산포가 1 pcm 밑으로 안 내려가면 — 그 경우 `search_tol` 자체를 조이는 A2 로 전환 판단.

---

#### S2 — 무코드 노브 (A2, 솔버 코드 0줄)

**선행 커밋(무조건, 채택 여부 무관):** `RASBERY_BICG_NMAX` / `RASBERY_BICG_EPS` 를 `trajectory::kArmEnv` 에 추가하고 `tools/test_case_key_contract.py` 갱신. 본 문서에서 kArmEnv 49개 엔트리를 전수 확인했고 두 이름은 없다. 두 상수는 BICGCMFD 생성자에서 읽혀 궤적을 움직이므로 **오늘 서로 다른 내부 예산이 같은 case key 를 공유한다** — 아래 nmax A/B 자체가 캐시된 답을 받을 수 있는 실측 무결성 결함이다.

런:

1. `RASBERY_STAGED_XE_TOL` ∈ {3000, 10000} — 1000 까지만 스윕되었다. xe 버킷 전체(69 %)를 steps/cascade 로 곱한다. 이미 `kArmEnv` 소속이고 case key 에 반영된다.
2. `RASBERY_BICG_NMAX` ∈ {2, 4} — **`outers × cmfd_sweeps × (1+nmax)` 로 채점.** `bicg_iters` 는 `attempts × (1+_nmaxbicg)` 로 유도되는 산술값이지 측정값이 아니므로 채점에 쓰면 안 된다.
3. (S1 통과 시) `RASBERY_SEARCH_STAGED_MARGIN=1.0` — `STAGED_SEARCH_MARGIN=4.0` 때문에 `loose_keff_tol = min(keff_tol×50, search_tol/4) = 5e-6` 로 묶여 있어 **50× 승수는 구속되는 절반에서 실제 5× 에 불과하다.** margin 1.0 은 2e-5, 4배 느슨. trial 이 아니라 outers 를 제거하는 유일한 미시도 A2 노브.

**킬:** `staged_relapses > 0.2 × search_trials` — loose 단계가 다른 근을 쫓고 있다는 신호(현재 KNGR 기준 0).

---

#### S3 — `_epsl2` 스테이징 (A2, ~60줄, 디바이스 코드 0)

**K0 게이트(S0 결과로 판정):** `cmfd_sweep_exits` 에서 예산 소진(state 3) drive 가 70 % 를 넘으면 — 5 sweep 안에 1e-6 에 못 닿는 drive 는 5e-6 에도 대개 못 닿으므로 — **S3 는 사망**하고 프로그램은 S4 로 건너뛴다. 판정단 3인 중 1인이 지목한, D2 의 가장 취약한 하중 전제다.

통과 시 구현: `Driver.h:3987` 의 허용오차 결정 블록 옆에 한 줄

```cpp
if (inner_couple > 0.0) ctx.cmfd_solver.setEpsl2(flux_tol_now / inner_couple);
```

`RASBERY_INNER_COUPLE` 기본 0(=off). margin 이 장식이 아닌 이유: `_epsl2 == flux_tol_now` 로 두면 아우터의 residual 절반이 공허해져 `flux_converged` 가 keff 항만으로 붕괴한다. 기본 margin 10 → LOOSE 에서 5e-6(5× 느슨). 이는 `STAGED_SEARCH_MARGIN` 이 이미 같은 이유로 쓰는 패턴이다.

**주요 아츰은 search-blind 변형으로 한다**(판정단 3인 중 2인이 "fallback 이 아니라 primary 여야 한다"고 지적). `has_search && searchType == BORON` 인 수용 표본과 POLISH 단계 전체에서는 프로덕션 `_epsl2` 를 유지하고 나머지에서만 완화한다. secant 가 읽는 자릿수가 보존되므로 boron_bracket·warm start 를 죽인 기전이 발화하지 않는다.

전송 비용 **정확히 0**: `kEpsl2` 는 매 런치 무조건 업로드되는 `sweep_in` 스칼라 블록에 이미 탑승하며 `SweepGraphCapacity`(nmax, slots, precision, lanes) 의 키가 아니다 → 재캡처 0, 새 커널 0, `CmfdOuterKernel.h` 변경 0.

**채점 규칙:** `outers × cmfd_sweeps` 로만 채점. sweeps 단독 채점 금지 — 아우터당 덜 수렴한 플럭스가 outers 를 되사올 수 있고 이 트리에 선례가 있다(`_nmaxbicg` 6→4 → i-SMR outers +6 %).

**중단 신호:** `negative_flux` 세그먼트 탈출 > ~150/1,326, `negative_retry_sweeps` > cmfd_sweeps 의 15 %.

---

#### S4 — L1: 정직한 Xe 후 재구동, LOOSE 한정 (N1, ~40줄)

`RASBERY_XE_HONEST_REDRIVE`(기본 0). `Driver.h:4429` 의 `prev_inner = eigv + 1.0` 을 `pre_xe_eigv`(Anderson/Picard 호출 직전 포착)로 대체. 플래그 off 시 표현식은 리터럴 그대로 → digest 불변.

**`!polishing` 로 한정한다.** search 가 프로덕션 허용오차에서 실제로 읽는 모든 표본은 여전히 오염 강제 재구동 위에서 취해지므로 수용점의 정의가 오늘과 비트 동일해지고, Gate A 경로 차이 논거가 소멸한다. `staged_relapses = 0`(loose 단계가 polish 를 단 한 번도 실패시킨 적 없음)이 이 한정의 안전성 근거이며, 오염 스텝 대다수는 loose 단계에 있다.

**본체를 헤더에 인라인하지 말 것** — 별도 TU. 선례 `71092e2`: 그런 본체를 SolveLoop 에 접합하면 인라이닝·`-ffp-contract` 결정이 재구성되어 **플래그 OFF digest 가 움직였다**(22b9a3187bfb4beb/4566 outers → c1a5d9116df9edb3/4601).

계약 시험: `tools/test_staged_tolerance.py` 는 POLISH 오염(4551)을 고정하고 이 지점은 고정하지 않는다 — `tools/test_device_outer_state_machine.py` 에 플래그 조건부 등가 단언을 추가해야 한다.

**킬:** S0 의 `xe_first_outer_would_converge / xe_steps < 0.30`.

---

#### S5 — boron 근 EFPD 외삽 + statepoint flux 시드 (N1)

두 레버 모두 S0 에서 이미 무런 가격 산정되었다. S0 이 3× 개선을 못 보이면 착수하지 않는다.

- **boron 근:** `SearchCarry` 에 (last_root_ppm, last_root_efpd, prev_*) 추가, `carriedBoronSlope` 의 가드 3종(비유한 / 부호 / 비율 [0.5,2.0])을 **축자 재사용** + ppm 절대 클램프. 소비는 `Scheduler.h:610` 의 else 분기. `carry_slope`(+8.11 % 로 기각)와 다른 점: 그것은 dk/dx 를 고쳤고 secant 는 어차피 첫 스텝에서 재측정한다; 이것은 **영점**을 고친다. `RASBERY_SEARCH_CARRY_ROOT` 를 `kArmEnv`/`CaseKey.h` 에 등록한다.
- **flux 시드:** phi_{k−1}, phi_{k−2} 를 EFPD 횡좌표와 함께 보관해 1차 외삽. WP10.2 WarmState(+2.3 % 로 기각)와 다른 점: 그것은 **낯선** 플럭스를 Jnet=0/Phis=0/dhat=0 에 대해 썼고(O(1) 비정합), 이것은 **직전 statepoint 의 수렴된 전류·d-hat** 위에 이웃 플럭스를 쓴다(O(dt²)).
  - **배치 필수 교정(판정단 지적):** DEPLETION 블록 **앞이 아니라 `PredictorStep` 뒤, 첫 SolveLoop 앞**에 쓸 것. `PredictorStep` 이 live `Phif` 를 `_flux_bos` 로 스냅샷해 감쇄하므로, 앞에 두면 핵종 밀도가 움직여 N1 이 아니라 **답의 변경**이 된다.
  - 선발 Xe 스텝(`RASBERY_SP_SEED_XE`)은 `schedule.xenon_transient` 가드 필수 — transient 덱에서 I/Xe 는 대수적 함수가 아니라 감쇄 상태다.
  - 게이지 안전성: `NormFactor` 가 매 Xe·감쇄 스텝 전에 출력으로 재규격화하고, `wiel` 은 `errl2` 를 `gammad` 로 정규화하며, statepoint 마다 첫 5 sweep 은 warm-up 의 Rayleigh 분기(스케일 불변)를 탄다 — 따라서 **형상만 시드되고 스케일은 자유 게이지**다.

---

#### S6 — 보류 (Wielandt shift, FAA)

- **Wielandt shift 2단** (~25줄): `polishing` bool 로 LOOSE 0.04 / POLISH 0.02. 연속 residual 연동 스케줄은 지수 튜닝 리스크가 있어 채택하지 않는다. `eshift()` 접근자 기존재, `kEshift` 가 `sweep_in` 에 이미 탑승, 그래프 키 아님 → 배관 비용 0. **0 초과 클램프 필수**(`_eshift == 0.0` 이 다섯 지점에서 경로를 단락시킨다). `negative_flux` 탈출이 과조임의 실패 모드.
- **FAA(플럭스 공간 Anderson):** 11 person-days, 자체 산술로 중심값 5 %, 동차성으로 인한 수용률 붕괴 리스크, `prev_inner <- eigv_cand` 가 사실상 A2. **본 프로그램에서는 착수하지 않는다.** 유일하게 정당화되는 경로는 "S4 가 통과하고 수용률 > 80 % 가 확인된 뒤, `Driver.h:4429` 오염 자체를 제거하는 A2 의 사전 조건으로서" 이며, 그것은 별도 브랜치·별도 게이트 항목이다.
- **명시적 폐기 1 — `RASBERY_XE_INTERIM_L2`.** 계획 W3.6 의 첫 후보이며 **구조적 원인이 있는 실측 NO-GO** — interim 스텝은 합성 사상의 점이 아니므로 `Driver.h:4346` 의 Anderson 게이트(`flux_converged` 항)가 이력에서 축출하고, 제안 수가 1,472 → 664 로 붕괴하며 outers 가 12,017 → 14,332/15,850/17,755 로 상승한다. Xe Anderson 단독이 outers −51 % 를 지고 있으므로 그것을 굶기는 어떤 것도 순손실이다. 진단용 노브로만 유지하고 **계획에서 이름으로 삭선한다.**
- **명시적 폐기 2 — `RASBERY_GPU_OUTER_SEGMENT_MAX` 상향.** §1.4(b): 세그먼트 종료의 81.9 % 가 물리적 이유이므로 예산 상향은 원리적으로 무익하다.

### 3.2 누적 기대치와 m64 c/h 환산

각 값은 판정단 할인 후의 **중심값**이며 대괄호는 밴드다.

| 단계 | outers 인자 | 근거 |
|---|---|---|
| 기준 m64 | 165.0/sp | 8,415 outers / 51 sp |
| S1 (L0) | ×1.02 [1.01–1.04] | 의도적 비용 |
| S2 (XE_TOL 3k/10k) | ×0.92 [0.88–0.97] | steps/cascade 5.24 → ~4.4 |
| S3 (`_epsl2`) | ×0.98 [0.95–1.03] | outers 효과는 미미(주효과는 sweeps) |
| S4 (L1 loose) | ×0.94 [0.90–1.00] | 오염 스텝의 loose 부분 |
| S5 (근 외삽 + 시드) | ×0.955 [0.93–0.99] | trial 영점 + initial 버킷 |
| **누적** | **×0.826 [0.73–0.99]** | |

**outers/sp: 165 → ~136** (밴드 120–163).

sweeps/outer: S3 통과 시 4.256 → ~3.6(인자 0.85, 밴드 0.70–1.00), K0 에서 사망 시 인자 1.00.

GPU 시간 합성(CMFD 내부가 커널 시간의 54.6 %; outers 는 전부에, sweeps/outer 는 CMFD 몫에만 곱한다):

```
T/T0 = f_out × (0.546 × f_sweep + 0.454)
     = 0.826 × (0.546 × 0.85 + 0.454) = 0.826 × 0.918 = 0.758
```

배치 8×M16+MPS 는 GPU 시간 한계(sm 96 %, 커널시간 이상치의 ≈92 %)이므로 거의 1:1 환산된다:

| 시나리오 | f_out | f_sweep | T/T0 | m64 c/h (1,440 기준) |
|---|---|---|---|---|
| 비관 (S3 K0 사망, S4 무효) | 0.95 | 1.00 | 0.950 | ~1,515 |
| **중심** | **0.826** | **0.85** | **0.758** | **~1,900** |
| 낙관 | 0.73 | 0.70 | 0.610 | ~2,360 |

기준 하단 1,321 c/h 로 잡으면 중심값은 ~1,745 c/h.

**목표 대비 정직한 진술.** 이 프로그램은 **m64 165 → 82.5 outers/sp 의 반감을 달성하지 못한다.** 중심 시나리오는 반감이 요구하는 −50 % 대비 −17 % 이며, 처리량 이득의 상당 부분은 outers 가 아니라 sweeps/outer 라는 두 번째 분모에서 온다. 반감에 도달하려면 `Driver.h:4429` 오염 제거(≥42 % 가 걸린 유일한 블록) 또는 디바이스 상주 Xe 핸드오프가 필요하고, 둘 다 판정단이 가장 높은 리스크로 평가한 항목이며 본 프로그램의 사전 조건(S1, S4)이 통과한 뒤에야 착수 가능하다.

---

## §4. B0 로 남는 것, 그리고 스파이크별 Gate A/B 운용

### 4.1 모든 스파이크에서 B0 로 유지되는 것

1. **플래그 OFF 경로.** 각 신설 노브는 off 분기가 현행 리터럴이 되도록 조건부 표현식으로 작성한다. 구체적으로: 히스토리 벡터 미할당(플래그 뒤에서 할당 — WarmState 는 파싱 시점에 할당해서 실패했다), `PhifMutable()` 미호출(`_phif_generation` 불변 → 플럭스 업로드 생략 카운터 불변), `eshift()`/`setEpsl2()` 미대입(`io.eshift`/`io.epsl2` 가 런 전체에서 동일 바이트), `Scheduler.h:610` 기존 else 분기 유지, **새 환경변수 이름 미노출**(`envSetToken`/`env_digest` 바이트 동일).
2. **본체의 out-of-line 배치.** 헤더 인라인 금지(`71092e2` 선례). 플래그 OFF digest 가 컴파일러 결정 재구성만으로 움직인 실측 사례가 있다.
3. **디바이스 커널·그래프 노드 인구조사.** 새 노드 0, `nodes/sweep` 불변 → `tools/test_cmfd_fuse_contract.py` 무영향. S0–S5 어느 항목도 `.cu` 파일을 열지 않는다.
4. **아레나 레이아웃.** (S6 이후에나 해당) 신설 블록은 **마지막**에 배치해 기존 영역 오프셋 불변.
5. **b1/b8/graph digest 불변식.** 세 아츰이 서로 일치해야 한다. 불일치는 답이 세그먼트 예산의 함수가 되었다는 뜻이며 즉시 중단 신호다.
6. **PPR / CRAM / `WIEL_FOLD=chunked` / `PC_MODE=decart`** 아츰은 건드리지 않는다.
7. **S0 전체.** 반사실 카운터를 포함해 래더가 읽지 않는 계측만 추가하므로 digest 불변 — 이것이 S0 를 무료로 만든다.

### 4.2 스파이크별 게이트 실행 절차

**모든 스파이크 공통 4단계:**

1. **B0 증명 런.** 플래그 OFF 로 **동결 궤적 digest 를 비트 단위 재현.** 실패하면 가드가 잘못된 것이고 이하 전부 무의미 — 즉시 중단.
2. **아츰 런.** 플래그 ON, 프로덕션 환경 나머지 고정.
3. **Gate A** — 동결 궤적 대비 18개 지표: keff 1.905 pcm / boron 15.309 ppm / AO 0.013 / pin 1 %.
4. **Gate B** — MASTER 대비 pin RMS 0.238 % / max 0.80 %.

**중단 감시 4종(모든 아츰에서 기준선과 나란히 출력):**

| 감시 항목 | 기준선 | 중단 임계 |
|---|---|---|
| `negative_flux` 세그먼트 탈출 | 115 / 1,326 | > ~150 |
| `flux_limit_retries` | 0 | ≠ 0 |
| `staged_relapses` | 0 | > 0.2 × search_trials |
| `cmfd_sweeps / outers` | 4.256 | > 4.4 (S3 제외) |

추가로 `FLUX_STALL` / `SEARCH_EXHAUSTED` / `NO_PROPOSAL` 이 0 에서 이탈하면 원인 불문 중단. `search_bisect` 가 0 에서, `search_refused` 가 2 에서 이탈하면 S5 의 제안기가 방황하기 시작한 신호다.

**등급별 추가 절차:**

- **A2 (S2, S3):** 자체 브랜치 + 명시적 롤백 경로. 수용 허용오차가 움직이므로 채택 시 `[SUMMARY]` 에 A2 gate word 를 기록하고, 해당 노브가 `kArmEnv` 에 있어 `env_digest` 를 분기시키는지 확인한다(S2 의 선행 커밋이 정확히 이 결함을 고친다).
- **N1 (S1, S4, S5):** Gate A/B 만. 단 **S1 은 예외** — 통과·실패와 무관하게 기준 궤적 재동결을 수반하므로 A2 와 동일한 브랜치·롤백 규율을 적용하고, **S1 이후의 모든 스파이크는 새 동결 궤적에 대해 게이트한다.** 이 재동결은 문서화된 정책 행위여야 하며 조용히 일어나서는 안 된다.
- **텔레메트리 분리:** `RASBERY_STATEPOINT_TELEMETRY=1` 런은 절대 타이밍 아츰에 섞지 않는다(단일 덱 wall ~2.5 %, 배치는 파괴). m64 는 항상 2회 — 계측용 1회, wall/c-h 용 1회.

### 4.3 채점 규칙 (판정단이 반복 지적)

- `outers × cmfd_sweeps` 를 곱으로 채점. **sweeps 단독 금지.**
- `bicg_iters` 로 채점 금지 — `attempts × (1+_nmaxbicg)` 로 유도되는 산술값이다. nmax 를 재는 유일한 정직한 방법은 `outers × cmfd_sweeps × (1+nmax)`.
- Gate A 실패 시 trust cap / margin 을 되감아 pcm 을 회수하지 말 것 — 그것은 게이트를 피팅하는 것이다. S1(경로 분리)로 돌아간다.

---

## §5. 미해결 질문

1. **m64 원인 귀속이 존재하지 않는다.** §1 의 모든 비중은 KNGR 프로파일이다. m64 가 statepoint 당 165 outers 를 쓰는 이유가 (a) statepoint 당 trial 이 더 많아서인지 (b) cascade 가 더 길어서인지 알 수 없고, 두 답은 서로 다른 레버를 지목한다. **S0 이전에 어떤 것도 확정하지 말 것.**
2. **4,614 대 4,377 outers 불일치.** A2 귀속 프로파일과 블록 19 이후 모든 프로덕션 런 사이의 미해소 차이. 본 문서의 모든 백분율이 여기 의존한다.
3. **디스패치 카운트 불일치(인자 ~1.6).** nsys 기록은 내부 반복 ~47.5k(reduce_dot2_fused 47,925 / prepare_p_jacobi 47,483)를 보이는데 호스트 계수는 74,508 — 유효 2.55 대 구조적 4.0. 컬러 sweep 쪽에서도 379,027/74,508 = 5.09 로 유효 R~2.5 대 설정 4. `[CMFD][OCCUPANCY]` 의 `launches_per_iteration` 과 `[CMFD][GRAPH]` 의 `nodes_per_sweep`/`launches_per_outer` 를 프로덕션 환경 런에서 인용하기 전까지 **모든 노드 카운트 산술(85 nodes/sweep, ~362/outer, ~3.0M/m64 case)은 이 인자만큼 불확실하다.**
4. **음속 재시도 꼬리의 크기를 모른다.** `icmfd_done − sweeps_done` 이 어떤 영수증에도 없다. 이것이 두꺼우면 스테이징도 Anderson 도 건드리지 못하는 순손실 버킷이며, `_epsl2` 완화 이전에 먼저 고쳐야 한다.
5. **`cmfd_sweep_exits` state 1 vs 3 분리를 모른다** — S3 의 존폐를 결정하는 단일 수치. 4.256/5 = 85 % 예산 활용률은 "대부분 예산 소진" 을 시사하며, 그렇다면 `_epsl2` 완화는 sweep 을 거의 못 줄인다.
6. **case key 결함(본 문서에서 확인):** `RASBERY_BICG_NMAX` / `RASBERY_BICG_EPS` 가 `kArmEnv` 49개 엔트리에 없다. 채택 여부와 무관한 선행 커밋 대상.
7. **`_ncmfd` = 5 는 한 번도 스윕되지 않았고 env 노브가 없다.** 스윕하려면 먼저 `icmfd_budget = 20*_ncmfd` 를 분리해야 한다 — 아니면 예산 변경이 재시도 거동을 조용히 바꾼다.
8. **L0(S1)이 스스로 Gate A 를 통과할 수 있는가?** 산술상 재동결이 거의 확실하다. 재동결의 승인 주체와 절차가 정의되어 있지 않다.
9. **판정단 불일치(미해소):** polish 전이 절감(~275 outers)이 D2 자신의 Xe-worth 산술에 의해 상쇄되는가. 한 판정단은 "대부분 증발", 다른 둘은 언급이 없다. S0 의 `poison_suppressed` / `poison_converged_next` 쌍이 직접 판정한다.
10. **디바이스 Xe 핸드오프(L2)의 두 번째 호스트/디바이스 발산.** `xe_pending` 을 풀면, 안정된 스텝(xe_change < tol)에서 호스트는 같은 아우터 안에서 settle gate·search 로 흘러가는데 디바이스의 Xenon 액션은 항상 아우터를 끝내고 재구동을 강제한다. 오염 수정만으로는 위임 아츰이 호스트 아우터 열을 재현할 수 없다. L2 의 B0 주장도 이와 함께 재검토 대상이다.
11. **Xe Anderson 의 콜드스타트 잠식.** 스테이징이 cascade 를 10.65 → 5.24 스텝으로 줄이면서 depth-2 창이 채워지는 시점이 늦어졌다 — 228 개의 첫 스텝이 순수 Picard, 228 개의 둘째 스텝이 AA(1), 1,192 − 752 = 440 스텝은 제안조차 되지 않았다. 수용률은 depth 2 에서 이미 97.5 %(733/752) 로, 이는 **신뢰 영역이 아니라 depth 가 구속 조건**이라는 서명이다. `XE_ANDERSON_DEPTH` 3/4 및 cascade 간 이력 이월은 env 한 개짜리 A/B 인데 기록에 스윕이 없다.
12. **`RASBERY_SEARCH_STAGED_MARGIN=1.0` 이 미시도 A2 노브인가.** `STAGED_SEARCH_MARGIN=4.0` 때문에 프로덕션 50× 승수는 구속되는 절반에서 실제 5× 다. 이것이 f50 → f100 이 −1.7 % 에 그친 이유이며, margin 을 여는 것이 flux 승수를 더 키우는 것보다 직접적이다. 단 S1 이후에만 측정 가능하다.
