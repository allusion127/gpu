# RASBERY 입력 수렴 설정

RASBERY JSON 입력에서 사용자가 조정하는 수렴 관련 키는 아래 두 개만
남긴다.

```json
"convergence": {
  "max_eigen_iterations": 100,
  "eigv_tolerance": 1.0e-6
}
```

- `max_eigen_iterations`: CMFD+Nodal 고유치 반복의 최대 횟수
- `eigv_tolerance`: 고유치 반복 수렴 기준

두 키는 top-level `convergence` 블록에 둘 수 있고, 필요한 경우 개별
`schedule` 항목에도 둘 수 있다. `standard` schedule 항목에 넣은 값은 이후
schedule 기본값으로 이어진다.

## 내부 기본값

그 외 TH 피드백, 임계 탐색, secant 안정화에 쓰는 tolerance와 반복 제한은
입력에서 조정하지 않는다. 값은 `src/Scheduler.h`의 `constexpr` 기본값을
사용한다.

```text
critical search max iterations = 100
TH feedback max iterations     = 10
critical search tolerance      = 1.0e-5
TH feedback tolerance          = 1.0e-6
temperature search tolerance   = 0.01
boron search tolerance         = 0.01
rod search tolerance           = 0.01
minimum secant denominator     = 1.0e-12
```

## 더 이상 읽지 않는 키

기존 입력 파일에 아래 키가 남아 있어도 새 파서는 스케줄 수렴값으로 반영하지
않는다.

- `max_th_iterations`
- `th_tolerance`
- `search_max_iter`
- `search_tol`
- `search_pcm_tolerance`
- `min_secant_denom`
- `search_minimum_secant_denominator`
- `search_minimum_span`
- `search_slope_freeze_dx_threshold`
- `minimum_keff`
- `minimum_carry_slope`

임계 탐색의 물리적 조건인 `search`, `search_min`, `search_max`,
`search_target`은 이 정리 대상이 아니며 기존처럼 입력에서 사용한다.
