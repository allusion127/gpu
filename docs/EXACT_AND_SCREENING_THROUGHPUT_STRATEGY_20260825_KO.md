# RASBERY 정확 경로 가속 및 대량 후보 처리량 전략

작성일: 2026-08-25
기준 브랜치: `codex/cmfd-gpu-assembly-drive-fusion-v2`

## 1. 목표의 해석

현재 단일 RTX PRO 6000에서 64개 APR1400 입력을 처리한 정상상태 실측은 213.9 cases/h이며, 동일 비교의 MASTER CPU 16병렬은 216–218 cases/h이다. 따라서 현재 정확 계산 경로는 MASTER W16과 거의 동등하지만, 한 장의 GPU에서 동일한 전체 물리 계산을 그대로 수행하면서 MASTER W16 대비 20배 이상의 처리량을 얻으려면 4,320 cases/h 이상이 필요하다.

이 목표는 두 종류로 분리한다.

1. **정확 물리 계산 처리량**: 모든 상태점, 피드백, 출력과 수렴 조건을 기존과 동일하게 수행한다.
2. **후보 생성 처리량**: 저비용 근사 screen으로 후보를 순위화한 뒤 선택된 survivor를 정확 경로로 다시 계산한다.

근사 screen 결과를 정확 계산 결과로 취급하지 않는다. 모든 screen receipt에는 `requires_exact_rerun=true`를 기록하고, 최종 채택값은 exact rerun에서만 생성한다.

## 2. 현재 한계

CMFD GPU operator assembly와 scalar graph fusion은 정상상태 M64 처리량을 약 196.4에서 213.9 cases/h로 높였다. 이 변경은 CPU setls를 제거하고 CMFD 도착 간격을 줄였지만, 남은 격차를 수십 배로 확대할 수 있는 종류의 최적화는 아니다.

정확 경로의 다음 큰 레버는 커널 미세 최적화가 아니라 **전체 outer iteration 수**다. APR1400 평형 Xe 상태에서는 하나의 Xe 갱신 뒤마다 flux를 다시 수렴시키므로, Xe–flux 고정점 반복이 전체 transport solve 횟수를 결정한다. 단, 기존 `RASBERY_XE_INTERIM_L2` 실험처럼 느슨한 flux에서 Xe를 조기에 갱신하면 다중 고정점 문제에서 다른 해로 이동할 수 있으므로 생산 경로에 사용할 수 없다.

## 3. 정확 경로: safeguarded Anderson acceleration

### 3.1 원칙

Anderson acceleration은 평형 Xe 고정점 `x = F(x)`의 최근 residual 이력을 사용해 다음 후보를 생성한다. 다음 안전 조건을 모두 적용한다.

- 기본값은 비활성이다.
- 기능이 꺼져 있으면 기존 `UpdateEquilibriumXenon(power, xe_relax)` 호출을 그대로 실행한다.
- 최종 수렴 판정은 가속된 step 크기가 아니라 원래 map의 raw residual `F(x)-x`로 수행한다.
- 비유한값, 특이한 최소제곱계, 과도한 step, residual 증가가 검출되면 즉시 기존 under-relaxation으로 복귀한다.
- 한 statepoint의 이력은 search/T-H/rod/boron/material 상태가 바뀔 때 폐기한다.
- 다른 Driver와 상태를 공유하지 않는다. 64개 동시 입력에서 각 인스턴스가 독립 이력을 가진다.
- 정확도 게이트는 기존 CPU/golden HDF5 및 MASTER 비교를 그대로 사용한다.

### 3.2 권장 설정

```bash
RASBERY_XE_ANDERSON=1
RASBERY_XE_ANDERSON_DEPTH=3
RASBERY_XE_ANDERSON_START=2
RASBERY_XE_ANDERSON_MAX_STEP=1.25
RASBERY_XE_ANDERSON_ACCEPT_RATIO=0.95
```

첫 생산 채택은 depth 2와 3만 비교한다. depth를 크게 하면 작은 least-squares 문제의 조건수가 나빠지고 이력 관리 비용만 늘 수 있으므로 기본 후보에 포함하지 않는다.

### 3.3 롤백

```bash
RASBERY_XE_ANDERSON=0
```

이 설정은 기존 fixed-point/oscillation detector/damping 경로로 즉시 복귀해야 한다.

## 4. 후보 생성 경로: fail-closed screen → exact

수십 배 처리량 목표는 다수 장전모형 또는 설계 후보의 **평가 파이프라인**에서 달성 가능성이 높다. 기존 feedback-limited screen을 다음 계약 아래 사용한다.

1. screen은 `RASBERY_BATCH_LIGHT_RESULT=1`과 함께만 실행한다.
2. `RASBERY_GA_FEEDBACK_PASSES=N`을 사용한 모든 row에 근사 physics mode를 기록한다.
3. batch width는 검증된 안전 상한 64를 초과하지 않는다.
4. objective별 순위 상관과 survivor recall을 기록한다.
5. survivor는 screen 출력을 재사용하지 않고 정확 입력부터 다시 실행한다.
6. exact survivor에서만 Anderson 가속을 선택적으로 A/B한다.
7. 전체 처리량은 `(screened candidates) / (screen wall + exact rerun wall)`로 계산하고 MASTER W16 처리량과 비교한다.

### 4.1 성능 게이트

```text
speedup_vs_master_w16 = effective_cases_per_hour / master_w16_cases_per_hour
```

- `master_w16_cases_per_hour`는 동일한 입력 집합과 출력 범위에서 측정한다.
- 목표 20배라면 최소 4,320–4,360 effective cases/h가 필요하다.
- 정확 survivor 실패, `requires_exact_rerun` 누락, fallback 발생, recall 기준 미달이면 전체 campaign을 무효 처리한다.

## 5. 검증 매트릭스

### 5.1 정확 경로

| Arm | Anderson | depth | 목적 |
|---|---:|---:|---|
| A | 0 | — | 기존 정확 기준선 |
| B | 1 | 2 | 보수적 가속 |
| C | 1 | 3 | 권장 후보 |
| D | 1 | 4 | 조건수/과적합 확인용 |

각 arm을 M1과 M64에서 최소 세 번 교차 실행한다. 다음을 모두 저장한다.

- wall, cases/h
- outer 수와 statepoint별 Xe 반복 수
- raw residual 및 accepted/rejected Anderson step 수
- fallback 이유별 횟수
- CMFD mean width와 arrival gap
- graph/drive/arena fallback
- HDF5 dataset 비교 결과

### 5.2 정확도 게이트

1. 기능 off 산출물이 현재 기준과 byte-identical인지 확인한다.
2. 기능 on은 우선 500/500 dataset byte 비교를 시도한다.
3. 반복 경로가 달라 byte identity가 성립하지 않으면 기존 허용 기준보다 엄격한 수치 게이트를 적용한다.
   - k-eff/CBC/AO/Fq/Fr
   - 상태점별 flux와 power norm
   - MASTER 비교 지표
4. raw residual이 기존 tolerance를 만족하지 않은 상태는 수렴으로 인정하지 않는다.
5. 가속 on/off가 다른 고정점으로 수렴하면 해당 설정을 즉시 폐기한다.

### 5.3 screen → exact 게이트

- screen-only 숫자를 정확 처리량으로 보고하지 않는다.
- top-k recall, Spearman 순위상관, exact survivor 성공률을 함께 보고한다.
- 최종 속도 배수는 screen과 exact rerun 시간을 모두 포함한다.
- 결과 receipt에 screen/exact 물리 모드와 입력·XS SHA-256을 포함한다.

## 6. 현실적인 성능 시나리오

| 구분 | 예상 효과 | 의미 |
|---|---:|---|
| CMFD persistent/cooperative kernel 단독 | 1.1–2배 범위의 후보 | dispatch 감소. 실측 전 예상이며 20배를 단독 달성하지 못함 |
| safeguarded Anderson 정확 경로 | outer 반복 감소량에 비례 | 정확도 게이트 통과 시 전체 정확 계산 단축 가능 |
| screen → exact 파이프라인 | 후보군과 survivor 비율에 따라 20배 이상 가능 | 근사 screen을 포함한 설계 탐색 처리량이며 최종 결과는 exact rerun |
| 다중 GPU/다중 노드 | 장치 수에 근접한 병렬 배수 | 한 장 GPU 제약 밖의 확장 경로 |

## 7. 채택 기준

정확 경로 변경은 다음을 모두 만족할 때만 기본값 후보가 된다.

- 모든 정확도 게이트 통과
- 다른 고정점 이동 없음
- fallback 및 reject telemetry가 설명 가능
- M1과 M64 중앙값 모두 개선
- 기존 경로로 즉시 롤백 가능

수십 배 주장은 다음 조건에서만 허용한다.

- 동일 후보군과 objective
- screen + exact 전체 wall 포함
- survivor exact rerun 완료
- MASTER W16 실측 처리량을 분모로 사용
- approximate/exact receipt가 명확히 분리됨

## 8. 결론

현재 정확 GPU 경로는 MASTER W16과 거의 동등한 단계다. 한 장 GPU에서 수십 배를 얻기 위해서는 커널 한두 개의 추가 융합이 아니라 transport solve 횟수를 안전하게 줄이는 고정점 가속과, 많은 후보를 저비용으로 거른 뒤 정확 재계산하는 계층형 실행 전략이 필요하다. 정확 경로에서는 safeguarded Anderson을 실험적으로 적용하고, 수십 배 목표는 fail-closed screen→exact campaign의 전체 처리량으로만 판정한다.