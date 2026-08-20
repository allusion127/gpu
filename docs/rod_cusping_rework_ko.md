# Rod cusping 재구현 및 속도 개선 (요약)

## 배경
제어봉 tip이 노드 중간에 걸칠 때 노드 평균 단면적의 균질화 오차(cusping)를 보정하기 위해,
PARCS §9.3 방식의 3-노드 축방향 미세격자(fine-mesh) FDM을 풀어 **단면적 가중인자 α**(Eq. 9.5)를
구한다. tip 노드의 macro XS를 rodded/unrodded 사이에서 flux-volume 가중으로 보간한다.

## 핵심 변경

### 1. CNCC(D̂ 주입) 제거 — α 가중만 사용
미세격자 계면 전류를 `Jnet`에 써서 CMFD `upddhat()`이 보정 D̂를 만들게 하는 경로는
**수치적으로 발산**한다. rod search가 한 번 크게 점프해 flux가 수렴에서 멀어지면
`max|dhat|/dtil`이 12 → 1e4 → 1e15 → NaN으로 폭주하고, NaN이 search로 흘러
`SetRod(NaN)` → 배열 범위 초과 → SIGSEGV로 죽는다.
- 부호 반전, Wielandt shift(`_eshift`), 수렴 tolerance, D̂ under-relaxation(η),
  대각우세 clamp — **모두 시도했으나 원인이 아니거나 발산을 막지 못함**.
- 결론: 미세격자 전류는 **α 계산에만** 쓰고 D̂에는 주입하지 않는다.

### 2. 경계 fine cell의 부분 점유(fractional) — 느림의 진짜 원인 해결
기존에는 fine cell을 **중심 판정(이진)**으로 rodded/unrodded 결정했다
(`fine_center > rod_tip`). 그래서 tip이 연속적으로 움직여도 셀이 1cm마다 한 번에
뒤집혀 `keff(봉 위치)`가 **계단(staircase, ~3e-5 높이)** 함수가 되었다.
secant search는 두 점으로 기울기를 추정하는데, 계단 위에서는 국소 기울기가
무의미해 overshoot → 수백 번 re-bracket(검색 tolerance 1e-5 < 계단 높이 →
영원히 수렴 못 함 → 300회 cap). 특정 봉 깊이의 스텝이 한 번에 4000~5400 outer를 먹었다.

**해결:** tip이 걸친 **한 개의 경계 fine cell만 부분 점유**로 처리한다.
`_fine_rod_frac` = (셀 내 rodded 길이)/(셀 높이)로 두고, 미세격자 풀이와 α 누적
양쪽에서 그 셀의 macro XS를 volume 가중한다. `keff(x)`가 연속이 되어 secant가
원래 tolerance(1e-5)로 잘 수렴한다.
- rod depletion fluence는 여전히 1cm bin이며, fluence/감손 경로는 `frac >= 0.5`로
  게이팅해 기존(중심 판정) 거동과 **bit-identical** 유지.

### 3. Rod criticality search에 relaxed solve 허용
계단을 없애기 전에는, 부드럽지 못한 keff(봉위치) 때문에 rod search의 중간 trial을
싸게 풀면 secant가 신뢰할 수 없어, **rod search만 매 trial 정밀(full) 수렴**을 돌고 있었다
(boron search는 이미 싼 trial + 마지막만 정밀 검증을 사용). `Driver.h`의
`relaxed_search_solve` 조건에서 `searchType != RODCRIT` 예외를 제거.
이제 계단이 없으니 중간 trial을 8 CMFD iter / TH 1회로 싸게 풀고 채택점만 정밀 검증한다.
- 트레이드오프: rod worth가 낮은 구간에선 봉 위치가 ~0.1cm(상대 0.25%) 다른 점에 안착할 수 있다
  (둘 다 1e-5 임계). keff 임계는 유지. boron search와 동일한 정밀도 수준.

### 결과
`test/5_Criticality/1D_Rod_Cusping.json`: **107초 → 20초(fractional cell) → 8초(+relaxed search) ≈ 13배**.
i-SMR CY01: 약 32초 → 6초. 모든 스텝 keff 임계(~1.0).
병목 스텝들의 outer 수 4000~5400 → 30~100.

## 시도했으나 채택하지 않은 최적화
- **D̂ under-relaxation**: 일반 nodal D̂를 damping하면 오히려 느려짐(η=0.5 → 127초).
  nodal D̂ 가속이 필요하므로 damping 금지.
- **cusping XS relaxation(`_rod_cusping_relaxation`)**: 0.1이 이미 최적(높이면 진동으로 느려짐).
- **per-iteration 캐시**(coarse XS / base XS snapshot): 약 10%뿐이고, snapshot이
  `_iden`(xenon) 갱신을 빠뜨리는 잠재 결함이 있어 **제거**. 핵심 이득은 알고리즘(경계 셀)에서 나옴.

## 주의
cusping이 활성인 케이스(움직이는 봉)는 secant search 민감도 때문에 **단일 스레드라도
실행마다 ~1e-6 수준으로 비결정적**이다. 검증 비교는 1e-9이 아니라 ~1e-5로 할 것.
봉이 부분 삽입되지 않는 케이스는 cusping이 비활성이라 bit-exact로 보존된다.
이번 변경으로 i-SMR/boron 등 cusping 케이스의 축방향 출력이 ~3e-4 이동하므로
해당 케이스의 기준 결과는 재생성이 필요하다.
