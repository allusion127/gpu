# 코드베이스 정리 요약 (2026-05-29)

여러 차례의 실험·테스트를 거치며 누적된 군더더기를, **동작(수치 결과)을 한 글자도 바꾸지 않은 채**
`Nodal.cpp/Nodal.h`(RASBERY 측 간결 스타일)와 `CHIFFON/Model.h`(CamelCase 공개 API + `_snake_case`
멤버) 스타일에 맞춰 통일·정리했다.

## 1. 작업 원칙

- **동작 보존이 최우선.** 모든 변경은 순수하게 구조/이름/주석 차원이며, 수치 표현·평가 순서·제어 흐름·
  알고리즘은 일절 건드리지 않았다.
- **회귀 검증 게이트.** 각 단계마다 빌드 + 회귀 테스트를 돌려 결과가 baseline과 동일함을 확인했다.
  회귀 스위트(`/.regress.sh`)는 다음을 `h5diff`로 비교한다.
  - CHIFFON 라이브러리 3종 (Importer/Interpolator/Model 의 HISTORY 보정 피팅 경로)
  - IISC solver 케이스 3종 (XSSet 의 history 보정 소비 경로)
  - pin-power probe (PPR 핀출력/플럭스 재구성)
  - 붕소 탐색 케이스 (Scheduler 임계 탐색)
  - i-SMR CY01 (Nodal/CMFD/BICG/Geometry/PPR/Scheduler/TH + 봉 탐색 전체 solver)
  - 허용오차: 결정적 경로 1e-9, 스레딩 노이즈가 있는 i-SMR/붕소 경로는 1e-6
    (OpenMP reduction 순서로 0 근처 reactivity만 흔들리므로 그 항목은 제외)

## 2. 단계별 변경 내용

### Batch 1 — 죽은/실험용 코드 제거
- `main.cpp`: `rasberyPrepareOpenMPStartup()`가 이미 수행하는 OpenMP `setenv` 중복 블록 제거,
  주석 처리된 디버그용 하드코딩 입력 블록 제거.
- `Scheduler.h`: 사용처 없는 빈 `THSettings` 구조체 제거, 사용처 없는 `SearchType::MODTEMP` 제거.
- `PPR.h`: 사용처 없는 `phic2` 매크로 제거.
- `Benchmark.h`: 주석 처리된 burnup 검증 블록 제거.
- `Importer.h`: 오타 `entier`→`entire`, 호출되지 않는 private 메서드 `ReadChix` 제거.

### Batch 2 — HISTORY_* 상수 난립 정리 (실험 잔재의 핵심)
- 길고 흩어져 있던 `HISTORY_VECTOR_*` (size_t sentinel) 13개 → 스코프 enum `Chiffon::Hv` 로 통합
  (예: `HISTORY_VECTOR_ROD_DEPLETION_FACTOR` → `Hv::ROD_DEPL`,
  `HISTORY_VECTOR_THERMAL_FLUX_FRACTION` → `Hv::THERM_FRAC`).
- `HISTORY_KIND_*` (int 태그) 13개 → 스코프 enum `Chiffon::Hk` 로 통합
  (예: `HISTORY_KIND_RHST_FLUENCE_WEIGHTED` → `Hk::RHST_FLU`).
- 4개 파일(Model/Importer/Interpolator/XSSet)에 걸쳐 **193개 사용처** 일괄 치환.
  내부 정수값은 그대로라 동작 동일. (enum class 대신 namespace+enum 을 쓴 이유는, `int kind` 필드와
  size_t 비교/산술에 캐스팅 없이 그대로 쓰여 동작 보존이 보장되기 때문.)

### Batch 3 — 불필요한 람다 인라인 / 중복 로직 병합
가장 더러웠던 부분. 핵심 hot-path는 직접, 나머지 파일은 파일별 병렬로 처리했고 전부 회귀 통과.
- **XSSet.cpp** `ApplyHistoryDeltasToNode`: 거의 동일한 ~120줄짜리 `currentAt`/`referenceAt`
  history-vector 평가 블록이 **두 번** 중복되던 것을, 공유 람다 `histCur`/`histRef` 로 합치고
  두 경로(가중 RHST / IISC 벡터)에서 호출하도록 변경 (~150줄 중복 제거).
  두 함수에 동일하게 정의돼 있던 `findCtype`/`findLoBurn`/`findHiBurn` 람다 3종 → 파일 정적 헬퍼로
  단일화. `Initialize`의 자잘한 1회용 람다 6종 → 파일 정적 헬퍼로, `flattenBranchDelta` 람다 → 멤버 함수.
- **Interpolator.h**: 거의 동일한 6개 `AddRhst*FitPoint` 함수의 공통 머리부분을
  `BuildHistoryMainReferenceResidual()` 헬퍼로 추출(분기별 가중치/키 로직은 그대로 보존),
  중복 `PointDensity` 구조체 2개를 1개로, `coefficientStride` 상수 중복 3곳을 파일 스코프 1개로.
- **Importer.h**: 거대 캡처 람다 `parseVectorIsotope`·`readVectorSettingsBlock` → 멤버 함수로 추출,
  bppm/tful/dmod 분기 설정 및 rod-depletion 파싱 중복 블록을 헬퍼로 단일화,
  4번씩 반복되던 isspace/toupper 문자 람다를 공유 헬퍼로.
- **Model.h**: `hvBurnMemWeight` 안의 `exponential`/`sourceDecay` 람다 인라인, `// region` 마커 제거.
- **Exporter.h**: 거의 동일한 7개 필드 접근 람다 → `SaveMemberField` 헬퍼 하나로.
- **IO.cpp**: JSON 다중 철자 키 폴백 패턴을 `FirstPresentKey`/`AnyKeyTrue` 헬퍼로 정리(우선순위 보존).
- **Geometry.cpp/.h**: 1/4 대칭 반복 패턴을 `quarterSpans()` 로 공유화, 미한정 `copy`→`std::copy`.
- **CMFD.cpp**: `upddhat` 세 분기의 공통 꼬리부분 통합. **BICGSolver.cpp**: 스타일 이질적인 `__restrict__` 제거.
- **PPR.cpp/.h**: 중복된 `jnetX`/`jnetY` → 방향 인자 하나 받는 `jnetDir(dir, ...)` 로 병합.
- **Scheduler.h**: `ProposeNextSearchPoint`의 RODCRIT/BORON 분기 중복 → `AdvanceSecantSearch()` 로 단일화.
- **Timer.h**: 자명한 멤버/메서드의 군더더기 `@brief` 와 인라인 본문 뒤 불필요한 세미콜론 제거.
- **milk.h / ReflectorSolver.h**: 이미 깔끔하거나(milk), 의도된 미완성(WIP) 코드(ReflectorSolver의
  주석 처리된 ADF 로직·예약 파라미터)라 그대로 보존.

### Batch 4 — 지나치게 긴 이름 단축
- `HistoryVector*` 자유 함수군 9종 → 짧은 `hv*` 이름 (62개 사용처). 예: `HistoryVectorRodFluenceCoordinate`
  → `hvRodFluCoord`, `HistoryVectorRoddedBurnupFraction` → `hvRoddedBurnFrac`. (Model.h 의 소문자 헬퍼
  `pidx`/`pidg`/`trim` 관례와 일치.)
- `Timer`: private 멤버 `start`/`end` → `_start`/`_end`.
- `BICGCMFD`: `_unshifted_diag`→`_udiag`, `unshifted_diag()`→`udiag()`, `setIterationLimit()`→`setIterLim()`.
- `Scheduler`: 탐색 상수 17개의 장황한 `kDefault` 접두사 제거
  (`kDefaultRodCriticalSearchTolerance`→`kRodCritSearchTol` 등).
- 이미 충분히 명료한 공개 API(`GetCrossSection`, `AddDepletionPoint`, milk 의 Vector/Matrix API,
  `nmaxswp`처럼 밑줄 없는 공개 멤버 관례를 따르는 `iter`)는 그대로 유지.

### Batch 5 — 포맷 통일
- 이번에 손댄 모든 파일에 저장소 `.clang-format`(LLVM 기반, 들여쓰기 4칸, ColumnLimit 0,
  PointerAlignment Left, 연속 대입/선언 정렬) 적용 → 새로 추가/병합된 코드까지 동일 스타일로 정렬.

## 3. 결과 요약

- 프로젝트 소스 합계 **약 19,560줄 → 19,194줄 (약 -366줄)**, 그 이상으로 중복·실험 잔재가 정리됨.
  - `XSSet.cpp` 3370→3203, `Interpolator.h` 1973→1891, `Importer.h` 2333→2279, `Model.h` 1367→1339 등.
- 람다/중복/긴 이름이 핵심 hot-path에서 대폭 줄어 가독성이 개선됨.
- **모든 단계에서 회귀 테스트 전 항목 통과 → 수치 결과는 정리 전과 동일.**

## 4. 참고

- 회귀 검증 스크립트: 저장소 루트의 `/.regress.sh` (정리 검증용 보조 스크립트).
- 동작 비교 baseline 산출물은 `/tmp/ras_base/` 에 보관(임시).
