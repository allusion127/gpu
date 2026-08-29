# W3.6 A2 — 상태점당 outer 감축 결과 (Task 13a/13b 판정)

**작성일**: 2026-08-29 | **브랜치**: `a2/outer-reduction` (기준 `cad0c0f`)
**측정 환경**: 로컬 WSL, GTX 1080 Ti (sm_61), nvcc 12.6 / gcc 13.3, `kngr_238.json` 35상태
**참고**: 이 문서의 wall은 **1080 Ti 로컬값**이다. 238 서버(sm_120) 대비 절대값이 다르므로 **비율만** 유효하다.

---

## 1. 요약 (TL;DR)

| 항목 | 결과 | 판정 |
|---|---|---|
| **총 outer** | 12,017 → **4,614 (−61.6 %)**, 343.3 → **131.8 /상태** | **목표(≤6,000) 달성**, 계획 목표 100/상태에 근접 |
| **로컬 wall** | 62.2 → **33.0 s (1.88×)**, 3회 교차측정 |  |
| **Gate A** (vs `cad0c0f` 궤적) | Δkeff **2.20 pcm**, ΔCBC **0.52 ppm**, ΔAO **5.4e-4**, 핀 **0.13 %** | A2 스크린(5/5/0.01/1 %) **전부 통과** |
| **Task 13a (interim-Xe)** | 12,017 → **14,332~17,755 (악화)** | **NO-GO — 구조적 이유 있음(§3)** |
| **Task 13b (flux-space Anderson)** | 미착수 | **불필요 판정 보류** — 목표가 §4로 달성됨(§7) |
| 채택 후보 | `RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1` | 기본 OFF, Gate B 대기 |

**핵심 발견 하나**: 계획이 A2의 1순위로 지목한 interim-Xe는 **Xe Anderson과 적대적**이다(§3). 계획이 언급하지 않은 **허용오차 단계화**가 실제 레버였다.

---

## 2. 측정 먼저 — outer 프로파일 (`tools/outer_profile.py`, commit `bba5e25`)

`RASBERY_STATEPOINT_TELEMETRY=1` + `RASBERY_GPU_OUTER=1`(budget 8) 수신증을 접는 도구를 추가했다. **기준선(cad0c0f, S2 + OUTER b8 + Anderson Xe ON)**:

```
TOTAL OUTERS : 12,017   (343.3 per statepoint,  35 statepoints)
  initial        712     5.9 %
  xe            8,579   71.4 %   ← 평형-Xe 캐스케이드 재수렴
  th              198     1.6 %
  search        1,508   12.5 %
  settle        1,020    8.5 %   ← SEARCH_SETTLE_ITERS 게이트
  fallback          0     0.0 %
xe 재수렴 비용 : 3.53 outer / 정착 Xe 스텝
캐스케이드     : 228개, 10.65 스텝/캐스케이드,  Xe 스텝 2,428
segment escape : flux_converged 2,370 / segment_budget 572 / negative_flux 263 / flux_stall_fatal 6
```

**구조 해석 (두 숫자가 아니라 하나의 구조다)**: 캐스케이드는 **커밋된 search trial·T/H 갱신마다 재장전**된다. 따라서 붕산 trial 1점 = 약 **38 outer**의 Xe 재평형(10.65 × 3.53)이고, 런당 137 trial이 소비된다. MASTER는 상태점당 총 ~59 outer다.

**허용오차가 레버인 이유**: 이 8,579 outer는 전부 **곧 버려질 trial 붕산농도에서** flux를 |dk|<1e-6(1 pcm)·L2<1e-6까지, Xe를 상대변화 1e-6까지 수렴시키고 있었다. 그 결과의 소비자는 허용오차 **2e-5**의 secant 탐색이다 — 20배 느슨하다.

수렴 판정 기준(deck `kngr_238.json`): `eigv_tolerance = 1e-6`, `CMFD_FLUX_L2_TOLERANCE = 1e-6`, `XE_EQUILIBRIUM_TOLERANCE = 1e-6`, `search_tol = 2e-5`, `max_eigen_iterations = 100`.

---

## 3. Task 13a — interim-Xe: **NO-GO (측정)**

`RASBERY_XE_INTERIM_L2`는 이미 host·device 양쪽에 존재한다(`CmfdOuterKernel.h` 511–521, device로 mined). 봉쇄도 Task 9에서 제거되어 있다. 스캔 결과:

| 임계 | 총 outer | vs 기준 | wall (s) | xe_aa_proposed | 해석 |
|---|---:|---:|---:|---:|---|
| 기준 (off) | 12,017 | — | 62.2 | 1,472 | |
| 1e-3 | 15,850 | **+31.9 %** | 75.2 | 926 | 악화 |
| 1e-4 | 14,332 | **+19.3 %** | 71.3 | 664 | 악화 |
| 1e-5 | 17,755 | **+47.8 %** | 107.6 | 833 | 악화 |

**원인은 구조적이며 우연이 아니다.** interim 스텝은 **미수렴 flux 위에서** 발화한다. Anderson이 외삽하는 대상은 합성사상 `x → (x에서 flux 수렴) → 평형 Xe`이고, 미수렴 flux는 그 사상의 점이 아니다. 그래서 `Driver.h`의 Anderson 게이트(`flux_converged` 항)가 interim 스텝을 **history 밖의 순수 Picard 스텝**으로 강제한다. Xe Anderson은 단독으로 outer의 51 %를 담당하는 레버이므로(캠페인 §2), 그것을 굶기는 어떤 것도 순손해다. 제안 수가 1,472 → 664로 무너지는 것이 그 지표다.

**결론**: Task 13a는 **Anderson 채택 이전에 설계된 태스크**이며, 채택 이후에는 성립하지 않는다. 계획서의 Task 13a는 **철회**를 권고한다. (`RASBERY_XE_INTERIM_L2` 자체는 진단·A/B용으로 존치.)

---

## 4. 채택 후보 — 허용오차 단계화 (commit `c303b3a`, `e7b66a5`)

### 4.1 설계

루프에 **단계(stage)** 를 도입한다.

- **LOOSE 단계**: `keff_tol × RASBERY_STAGED_FLUX_TOL`, `flux_tol × 동일`, `XE_EQUILIBRIUM_TOLERANCE × RASBERY_STAGED_XE_TOL`.
- 모든 피드백이 **LOOSE에서** 수렴했다는 판정은 **탈출이 아니다.** 그것은 POLISH 전환의 트리거다: 생산 허용오차 복원 → `prev_inner` 오염(실제 재구동 강제) → **Xe 캐스케이드 재장전**(`prev_xe_change = inf`) → 정착 게이트 초기화 → **같은 질문을 다시** 던진다.
- **생산 허용오차에서 도달한 합의만이 solve를 끝낸다.**
- POLISH에서 탐색이 이견을 내면 trial을 하나 더 커밋하고 LOOSE로 되돌아간다(`staged_relapses` 계상). 즉 **outer는 잃을 수 있어도, 생산 허용오차를 한 번도 만족한 적 없는 상태를 발행할 수는 없다.**

**탐색 바닥(floor)**: 자기 허용오차보다 나쁘게 수렴한 flux에서 k_eff를 읽는 secant는 노이즈를 표본한다(이 코드베이스의 `rodcrit_search_floor`가 그 전례다). 그래서 탐색 실행 중 LOOSE keff 허용오차는 `search_tol / 4` 아래로 **캡**된다. 느슨함은 아무도 읽지 않는 자리에만 허용된다.

**정착 게이트 (`RASBERY_STAGED_LOOSE_SETTLE=1`, 별도 플래그)**: `SEARCH_SETTLE_ITERS` 게이트는 **탐색이 신뢰하는 표본**을 위해 2 outer를 쓴다. 단계화 이후 표본은 두 종류다 — LOOSE 표본(secant를 근방으로 몰기만 하고 전부 재검증됨)과 POLISH 표본(발행되는 것). 게이트는 후자에서만 값어치가 있다. 단계화가 주변을 줄이면서 이 버킷은 8.5 % → **21.1 %** 로 자라 있었다.

### 4.2 승수 스캔 (전부 35상태 full deck, Gate A는 `cad0c0f` 궤적 대비)

| arm | FLUX×/XE× / settle | 총 outer | /상태 | wall (s) | Δkeff (pcm) | Δppm | ΔAO | 핀 max % | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | — | 12,017 | 343.3 | 62.2 | — | — | — | — | 기준 |
| f5 | 5 / 1 / off | 9,871 | 282.0 | 57.7 | 3.23 | 0.64 | 0.0078 | **1.14** | 핀 초과 |
| f20 | 20 / 1 / off | 11,943 | 341.2 | 66.3 | — | — | — | — | 무효(스래시 279 캐스케이드) |
| x10 | 1 / 10 / off | 10,524 | 300.7 | 55.0 | 3.21 | 0.64 | 0.0078 | **1.14** | 핀 초과 |
| x100 | 1 / 100 / off | 9,949 | 284.3 | 55.5 | 3.21 | 0.64 | 0.0078 | **1.14** | 핀 초과 |
| f5x100 | 5 / 100 / off | 7,850 | 224.3 | 46.8 | 1.69 | 0.38 | 0.0078 | **1.14** | 핀 초과 |
| f5x1k | 5 / 1000 / off | 7,896 | 225.6 | 45.6 | 0.048 | 0.005 | 1.5e-4 | 0.05 | 통과 |
| f10x1k | 10 / 1000 / off | 7,132 | 203.8 | 44.0 | 0.046 | 0.003 | 1.4e-4 | 0.03 | 통과 |
| f20x1k | 20 / 1000 / off | 6,499 | 185.7 | 42.0 | 0.048 | 0.004 | 1.4e-4 | 0.03 | 통과 |
| f50x1k | 50 / 1000 / off | 5,675 | 162.1 | 38.2 | 0.51 | 0.07 | 4.2e-4 | 0.07 | 통과 |
| f100x1k | 100 / 1000 / off | 5,577 | 159.3 | 37.3 | 1.91 | 0.14 | 3.2e-4 | 0.06 | 통과(정확도 열위) |
| f5x1e4 | 5 / 10000 / off | 7,524 | 215.0 | 46.8 | 3.28 | 0.83 | 0.0079 | **1.16** | 핀 초과 |
| s_on20 | 20 / 1000 / **on** | 5,864 | 167.5 | 37.3 | 1.42 | 0.27 | 0.0068 | **1.12** | 핀 초과 |
| **s_on** | **50 / 1000 / on** | **4,614** | **131.8** | **33.0** | **2.20** | **0.52** | **5.4e-4** | **0.13** | **채택 후보** |

**비단조성에 대한 주석**: 핀 1.1 % 를 내는 arm들(f5, x100, f5x100, f5x1e4, s_on20)은 승수가 더 큰 arm들보다 오히려 나쁘다. 이는 매끄러운 정확도 손실이 아니라 **후기 연소(28–33상태)에서의 이산적 경로 분기**다(f5x100의 편차는 28–33에만 존재). 승수가 충분히 크면 LOOSE 단계가 Xe를 거의 건드리지 않고 **POLISH 캐스케이드가 정확한 작업을 전부 수행**하므로 궤적이 기준선으로 되돌아온다 — 이것이 x1000 영역이 x100 영역보다 **더 정확한** 이유다. 따라서 승수 선택은 "작을수록 안전"이 아니다.

### 4.3 채택 후보의 프로파일 (s_on)

```
TOTAL OUTERS : 4,614   (131.8 per statepoint)
  initial        347     7.5 %
  xe           3,184   69.0 %
  th               86     1.9 %
  search         707    15.3 %
  settle         290     6.3 %     (1,020 → 290)
  fallback         0     0.0 %
xe 재수렴 비용 : 2.67 outer/스텝 (3.53에서)
캐스케이드     : 228개, 5.24 스텝/캐스케이드 (10.65에서)
Xe Anderson    : 733/752 수용 (97.5 %)  ← 기준 95.1 %, 알고리즘 열화 없음
flux limit cycle: 0 (기준 6),  Xe budget exhausted: 0 (기준 2)
segment escape : flux_converged 1,163(82.4 %) / segment_budget 133 / negative_flux 115 / flux_stall_fatal 0
```

- **캐스케이드 수는 그대로(228)** 이고 **스텝 수가 반으로** 준다 — 즉 탐색 궤적을 바꾸지 않고 각 캐스케이드를 싸게 만들었다.
- **`staged_relapses = 0`**: LOOSE 단계의 합의가 POLISH를 **한 번도** 통과하지 못한 적이 없다. 느슨한 단계가 다른 근을 찾고 있지 않다는 직접 증거.
- `flux_stall_fatal` 6 → 0, `negative_flux` 263 → 115: escape 자체를 목표로 삼지 않았는데 함께 줄었다(segment 수가 3,211 → 1,411).

---

## 5. Gate A (로컬) — 전 항목

| 검사 | 결과 |
|---|---|
| **feature-off byte 동일성** (본 브랜치 바이너리, env 없음 vs `cad0c0f` 바이너리) | `h5diff` **500/500 데이터셋 0 differences**, outer 12,017 동일 |
| **run-to-run 결정론 ×2** (`det1`/`det2`, 동일 env) | `h5diff` **IDENTICAL** |
| **빌드 간 재현성** (커밋 3개 전 바이너리와 동일 arm) | `h5diff` **IDENTICAL** |
| **kngr_238 Gate A** (s_on vs base) | keff 2.196 pcm / ppm 0.520 / AO 5.4e-4 / 핀 0.129 % — **전부 스크린 내** |
| 편차 분포 | 상태 1–31 모두 |Δkeff| < 0.22 pcm, 핀 < 0.08 %. 32–35(EOC·자연EOC 구간)에서 −1.37/+1.74/−0.52/−2.20 pcm — **연소 궤적 누적**, 버그 시그니처 아님 |
| **trimmed kngr3** (3상태) | keff 0.050 pcm / ppm 0.003 / AO 0.0 / 핀 0.014 % — outer 662 → **292 (−55.9 %)** |
| **i-SMR CY01** (2상태) | keff 0.030 pcm / ppm 0.0 / AO 0.0 — outer 200 → **105 (−47.5 %)** |
| **i-SMR CY02** (2상태) | keff 1.440 pcm / ppm 0.0 / AO 1e-4 — outer 857 → **367 (−57.2 %)**. sp1은 restart+shuffle(primeXeDamping) 다근 구간이나 **AO 1e-4로 물리 분지 유지** |
| **배치 4덱 ↔ 단일 동일성** | on: d0–d3 **전부 IDENTICAL**. off(대조군)도 전부 IDENTICAL. 배치 wall 15.5 → **7.0 s** |
| **ctest** | 12/12 PASS |
| **계약 테스트** | 신규 `test_staged_tolerance.py` PASS. 실패는 브랜치 이전부터의 4건(`cmfd_fp32`, `ga_feedback_screen`, `ga_promotion_gate`, `nodal_constant_cache`)뿐 — 본 브랜치가 추가한 실패 **0** |

**로컬 wall (3회 교차측정, 노이즈 있음)**: off 59.96 / 62.32 / 62.26 s → on 32.58 / 33.03 / 33.09 s = **1.88×**.

---

## 6. Gate B 프로토콜 (238 러너용) — **로컬에서는 실행 불가**

MASTER 기준이 로컬에 없다. 아래를 238 GPU0에서 그대로 실행한다.

### 6.0 브랜치·빌드

```bash
git fetch && git checkout a2/outer-reduction     # 기준 cad0c0f
cmake -S <src> -B <bld> -DCMAKE_BUILD_TYPE=Release \
      -DRASBERY_ENABLE_CUDA=ON -DRASBERY_CUDA_ARCHITECTURES=120 \
      -DRASBERY_ENABLE_TESTS=ON
cmake --build <bld> -j
```

### 6.1 공통 env (S2 단일 생산 구성)

```bash
export RASBERY_GPU=1 RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1
export RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1
export RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1
export RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8
export RASBERY_OMP_THREADS=12
# RASBERY_XE_ANDERSON 은 설정하지 말 것 — 단일 실행 기본값 ON 이 검증 대상이다
```

### 6.2 arm 실행 (각 arm 3회, 교차 순서로. wall은 3회 중앙값)

```bash
D=~/t18decks/kngr; O=~/a2gate; mkdir -p $O; cd $D

# BASE — 현행 생산 (= cad0c0f 궤적)
for r in 1 2 3; do
  /usr/bin/time -f "%e" -o $O/base_r$r.wall \
    <bld>/RASBERY --rasi kngr_238.json --raso $O/base_r$r.h5 > $O/base_r$r.log 2>&1
done

# CAND — A2 채택 후보
export RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1
for r in 1 2 3; do
  /usr/bin/time -f "%e" -o $O/cand_r$r.wall \
    <bld>/RASBERY --rasi kngr_238.json --raso $O/cand_r$r.h5 > $O/cand_r$r.log 2>&1
done
unset RASBERY_STAGED_FLUX_TOL RASBERY_STAGED_XE_TOL RASBERY_STAGED_LOOSE_SETTLE

# 보조 arm (승수 민감도 — CAND 가 Gate B 를 통과하지 못할 때만 필요)
#   f20x1k : STAGED_FLUX_TOL=20 STAGED_XE_TOL=1000            (settle 플래그 없음)
#   f50x1k : STAGED_FLUX_TOL=50 STAGED_XE_TOL=1000            (settle 플래그 없음)
```

### 6.3 outer·telemetry 수신증 (타이밍과 **섞지 말 것**, 별도 실행)

```bash
RASBERY_STATEPOINT_TELEMETRY=1 <bld>/RASBERY --rasi kngr_238.json --raso $O/base_tel.h5 > $O/base_tel.log 2>&1
RASBERY_STATEPOINT_TELEMETRY=1 RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 \
  RASBERY_STAGED_LOOSE_SETTLE=1 \
  <bld>/RASBERY --rasi kngr_238.json --raso $O/cand_tel.h5 > $O/cand_tel.log 2>&1

python3 tools/outer_profile.py $O/base_tel.log $O/cand_tel.log     # A/B 표
python3 tools/outer_profile.py --json $O/cand_tel.log              # 기계판독 총계
```

**기대값(로컬 기준, 238에서도 outer 수는 동일해야 한다 — 결정론적)**: base 12,017 / cand **4,614**, `staged_relapses` 0, Anderson 수용률 ~97.5 %.
outer 수가 로컬과 다르면 **먼저 그것을 조사할 것** — deck 또는 env가 다르다는 뜻이다.

### 6.4 Gate A (238 자체 확인)

```bash
python3 tools/gate_a_compare.py $O/base_r1.h5 $O/cand_r1.h5 --per-step
python3 tools/gate_a_compare.py $O/base_r1.h5 $O/base_r2.h5        # 결정론: 전부 0 이어야 함
h5diff -q $O/cand_r1.h5 $O/cand_r2.h5 && echo DETERMINISTIC
```

### 6.5 **Gate B — MASTER 대비 (판정의 본체)**

```bash
python3 tools/compare_master_rasbery.py <MAS_SUM> $O/base_r1.h5 -o $O/master_base
python3 tools/compare_master_rasbery.py <MAS_SUM> $O/cand_r1.h5 -o $O/master_cand
# <MAS_SUM> = scratchpad kngr_mas_sum.txt (MASTER_vs_RASBERY_COMPARISON_20260824_KO.md 부록 A)
# BOC 핀은 기존 핀 스크립트(scratchpad make_v2_master_cmp.py + kngr_mas_ppi_boc.txt)로 별도 산출
```

**합격 envelope (v2 기준값)**

| 지표 | v2 기준 | 합격 |
|---|---|---|
| 반응도 max | 1.905 pcm | **≤ ~2 pcm** |
| CBC max | 15.309 ppm | **≤ ~15.3 ppm** |
| AO | 0.013 | **≤ ~0.013** |
| BOC 핀 RMS / max | 0.24 % / 0.78 % | **≤ 0.24 % / 0.8 %** |

**판정 규칙**: Gate A의 2.20 pcm 편차는 Anderson 채택 때와 **같은 구조의 질문**이다 — 궤적이 움직였을 때 그것이 MASTER **쪽으로** 움직였는지 **반대로** 움직였는지를 Gate B만이 답한다. Gate B가 v2 envelope 안이면 채택 + **v3 동결**; envelope 밖이면 승수를 낮춘 arm(f20x1k, f50x1k without settle)으로 재판정한다.

### 6.6 부수 게이트

```bash
# feature-off byte 동일성 (500 데이터셋 골든)
h5diff -q $O/base_r1.h5 <cad0c0f 바이너리 산출 h5>

# 배치 4덱 ↔ 단일 (per-deck 동일성)  — bfdecks / bfbatch.sh 계열
# M64 앵커: 배치 기본값 확인용. **배치에서 STAGED_* 를 켜지 말 것** —
#   Anderson 배치 OFF 와 같은 도착폭 기아 구조(캠페인 §8.2)에 걸린다.
#   배치 채택은 Phase 5 slot compaction 이후 별도 판정.
ctest --output-on-failure
for t in tools/test_*.py; do python3 $t; done   # 사전 실패 4건 외 신규 실패 0
```

### 6.7 wall 기대

로컬 1080 Ti: 62.2 → 33.0 s(1.88×). 238에서 현행 최고가 33.2 s이므로 **동일 비율이면 ~17.7 s = MASTER 27.2 s 대비 1.54×**. 실제로는 outer가 줄면 커널 발사 지연 비중이 커지므로 **1.6~1.9× 범위**를 기대값으로 기록하고, 미달 시 §7의 잔여 항목으로 넘긴다.

---

## 7. 롤백·잔여

**롤백**: 세 env를 설정하지 않는 것으로 끝난다(`RASBERY_STAGED_FLUX_TOL`, `RASBERY_STAGED_XE_TOL`, `RASBERY_STAGED_LOOSE_SETTLE`). feature-off byte 동일성이 `h5diff` 전 데이터셋으로 확인되어 있으므로 **코드 되돌림은 필요 없다**. 코드 단위 롤백이 필요하면 `git revert e7b66a5 c303b3a`(도구·계약 커밋은 무해하므로 존치 가능).

**Task 13b (flux-space Anderson) 판정 보류**: 계획의 목표(300 → 100/상태)는 131.8/상태로 **거의 달성**되었고, 남은 outer의 69 %는 여전히 Xe 캐스케이드다. flux-space Anderson은 그 안쪽의 flux 재수렴(2.67 outer/스텝)을 노리는데, **이미 2.67**이므로 상한이 캐스케이드당 5.24 스텝 × 1.67 = 약 **2,000 outer**다. 착수 전에 `docs/GPU_RASBERY_100_PERCENT_GPU_ASYNC_SCHEDULER_PLAN_REV7_1_KO.md` §13의 비용(3인일 스파이크 + device 커널)과 이 상한을 다시 견줄 것.

**다음 후보(측정 근거 순)**

1. **캐스케이드 스텝 수** — 5.24 스텝/캐스케이드 × 228 캐스케이드가 남은 outer의 대부분이다. Anderson depth(현재 m=2) 확대가 가장 싼 실험이다.
2. **search trial 137** — POLISH 표본만 정밀하므로 secant의 초기 브래킷을 더 공격적으로 잡을 여지가 생겼다.
3. **`negative_flux` escape 115** — segment 탈출의 8.2 %. outer를 직접 쓰지는 않으나 segment 재시작 비용이다.

---

## 부록 A. 재현 명령 (로컬)

```bash
git -C <repo> worktree add <repo>/../rasbery_gpu_wt_a2 cad0c0f -b a2/outer-reduction
~/a2sync.sh ~/rb_a2 && ~/bfbuild.sh ~/rb_a2 ~/rb_a2_b
~/a2run.sh ~/rb_a2_b/RASBERY ~/a2out base RASBERY_STATEPOINT_TELEMETRY=1
~/a2run.sh ~/rb_a2_b/RASBERY ~/a2out cand RASBERY_STATEPOINT_TELEMETRY=1 \
   RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_STAGED_LOOSE_SETTLE=1
python3 tools/outer_profile.py ~/a2out/base.log ~/a2out/cand.log
python3 tools/gate_a_compare.py ~/a2out/base.h5 ~/a2out/cand.h5 --per-step
```

## 부록 B. 커밋

| 커밋 | 내용 |
|---|---|
| `bba5e25` | `tools/outer_profile.py` — outer 귀속·escape 수신증 판독기 |
| `c303b3a` | 허용오차 단계화 (`RASBERY_STAGED_FLUX_TOL`, `RASBERY_STAGED_XE_TOL`), 기본 OFF |
| `e7b66a5` | LOOSE 단계 정착 게이트 생략 (`RASBERY_STAGED_LOOSE_SETTLE`), 기본 OFF |
| `5006604` | `tools/gate_a_compare.py` — Gate A 스크린 |
| `28abd99` | `tools/test_staged_tolerance.py` 계약 + `test_xe_frozen_mode.py` 동반 갱신 |
