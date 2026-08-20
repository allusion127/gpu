# CHIFFON - RASBERY 연결 구조

> 참고: 이 문서는 일반 interface와 일부 legacy keyed RHST 용어를 함께
> 설명한다. 현재 9-1_IISC 기준 방법론인 keyless SPCT+RDPL+RHST는
> `test/9-1_IISC/docs/keyless_iisc_rhst_methodology_ko.md`를 기준으로 본다.

이 문서는 CHIFFON에서 만든 cross-section library가 RASBERY 전노심 계산에
어떻게 연결되는지 정리한다. 목적은 두 코드 사이의 책임 경계, 데이터 흐름,
제어봉/분기/연소 상태가 어느 지점에서 반영되는지 한 번에 추적할 수 있게
하는 것이다.

## 역할 분담

CHIFFON은 group constant library를 만들고 보간 가능한 형태로 저장하는
쪽을 담당한다.

- HGC/branch/extra/rod-history 입력을 읽어 `Chiffon::Model`을 만든다.
- Reference depletion point, branch delta, history delta, isotope vector
  correction을 구성한다.
- HDF5 파일에 `CHIFFON_HDF5` format으로 저장한다.
- 저장된 HDF5를 다시 읽어 RASBERY가 사용할 `Model` 목록을 제공한다.

RASBERY는 전노심 상태를 소유하고, 각 노드의 상태에 맞는 단면적을 재구성한
뒤 CMFD/Nodal/PPR/depletion 계산을 수행한다.

- Geometry, schedule, TH/search/restart 상태를 관리한다.
- CHIFFON model을 `XSSet` 내부의 structure-of-arrays 형태로 평탄화한다.
- 각 노드의 burnup, boron, fuel temperature, moderator density, rod state에
  맞게 live XS 배열을 갱신한다.
- Rod cusping처럼 전노심 solver 직전에 필요한 local XS correction을 적용한다.

## 실행 경로

CHIFFON library를 새로 만들 때는 `src/main.cpp`에서 다음 경로를 탄다.

```text
RASBERY --chiffoni input.json --chiffono output.h5
  -> Chiffon::Isotope::Initialize(...)
  -> Chiffon::Importer::ReadInput(...)
  -> Chiffon::Exporter::SaveHDF(...)
```

전노심 계산에서 이미 만들어진 CHIFFON HDF5를 사용할 때는 다음 경로를 탄다.

```text
RASBERY --rasi core.json --raso result.h5
  -> IO::ReadInput(...)
  -> Geometry::Initialize(...)
  -> XSSet::Initialize(xs_path)
  -> Chiffon::Importer::LoadHDF(xs_path)
  -> XSSet internal flattening
  -> Driver schedule loop
```

입력 JSON에서는 `data.cross-section`이 CHIFFON HDF5 경로를 가리킨다. `IO`는
geometry를 먼저 초기화한 뒤 `XSSet::Initialize()`를 호출한다. 이는
`XSSet`이 `nxy`, `nz`, node volume, rod map 같은 geometry 정보를 사용해
library를 노드별 계산 구조로 옮겨야 하기 때문이다.

## CHIFFON 쪽 파일 책임

`include/chiffon/Model.h`는 CHIFFON의 중심 데이터 구조다.

- `CrossSection`은 macroscopic, microscopic, lumped microscopic XS를 담는다.
- `DepletionPoint`는 하나의 HGC 상태점과 isotope density, flux, branch
  metadata를 담는다.
- `DeltaCrossSection`은 branch/history correction을 다항식 또는 spline으로
  평가한다.
- `Model::FillCrossSection()`은 reference XS와 branch delta를 조합해 한 상태의
  XS와 isotope density를 만든다.

`include/chiffon/Importer.h`는 입력과 HDF5 library를 읽는다.

- CHIFFON JSON 입력에서 fuel file, branch, extra reference, rod-history 설정을
  읽는다. 입력 parser는 정해진 schema만 직접 읽으며, 같은 물리량에 대한 여러
  alias나 사용하지 않는 state field fallback은 두지 않는다.
- HDF5의 metadata, depletion points, unified branch delta registry를
  `Model`로 복원한다. 오래된 파일의 `history_deltas` group도 호환 경로로
  읽을 수 있다.
- isotope vector correction과 rod-history ctype convention을 해석한다.

현재 CHIFFON JSON에서 `ReadInput()`이 받는 주요 schema는 다음과 같다.

- `settings.discontinuity factor`
- `settings.bppm`, `settings.tful`, `settings.dmod`의 `apply`, `order`, `type`,
  선택적 `pre_remove`
- 선택적 `settings.spectral`의 `apply`, `isotopes`, 선택적 `isotope ratio`,
  `pre_remove`
- 선택적 `settings.rod history`의 `apply`, 선택적 `pre_remove`
- `fuels.<name>.main`, 선택적 `extra`, 선택적 `rod history`

`avg_tmod`, `rated_power` 같은 별도 state delta 입력은 현재 Importer에서 읽지
않는다. 그런 효과가 필요하면 RASBERY의 실제 state update나 branch 축 정의에
명시적으로 추가해야 한다.

`include/chiffon/Exporter.h`는 `Model`을 HDF5로 저장한다.

- depletion point와 delta cross-section을 flat dataset으로 저장한다.
- BPPM, TFUL, DMOD, SPCT, RHST correction을 `branch_deltas` registry에 같은
  `BranchDelta` payload 형식으로 저장한다.
- `milk::Vector` 데이터는 불필요한 `std::vector` 복사 없이 raw span을
  dataset buffer에 이어 붙인다.

`include/chiffon/Interpolator.h`는 branch/history correction fit을 담당한다.

- BPPM, fuel temperature, moderator density branch delta를 만든다.
- Extra HGC와 rod-history HGC로부터 SPCT/RHST branch correction을 fitting한다.
- SPCT는 isotope vector에서 계산한 scalar term을 branch 축으로 사용하고,
  RHST는 rodded burnup을 scalar branch 축으로 사용한다.
- 코드 내부에서는 SPCT fitting point를 `_spct_dpts`, RHST fitting point를
  `_rhst_dpts`로 저장한다. 입력 키 이름의 `extra`, `rod history`는 유지하되,
  계산 경로 이름은 branch type과 맞춘다.
- Reference와 correction의 기준 상태가 섞이지 않도록 ctype, burnup,
  trajectory ctype을 함께 사용한다.

`include/chiffon/Benchmark.h`와 `include/chiffon/ReflectorSolver.h`는 검증 출력과
reflector ADF 적용처럼 보조 역할을 담당한다.

## RASBERY 쪽 연결점

`src/XSSet.cpp`가 CHIFFON과 RASBERY 사이의 핵심 어댑터다. `XSSet::Initialize()`
는 `Chiffon::Importer::LoadHDF()`로 읽은 `Model` 목록을 보관한 뒤, solver에서
빠르게 접근할 수 있도록 주요 데이터를 평탄화한다.

대표적인 내부 배열은 다음과 같다.

- `_ref_lmpx`, `_ref_micx`: reference depletion point의 lumped/microscopic XS.
- `_lib_lmpx`, `_lib_micx`: live node XS 재구성에 필요한 library data.
- `_lib_coeff_lmpx`, `_lib_coeff_micx`: branch/history delta coefficient.
- `_node_delta_*`: 각 노드가 현재 사용할 delta table index와 interpolation
  정보를 가리키는 cache.
- `_xsdf`, `_xsaf`, `_xsnf`, `_xskf`, `_xssf`: CMFD/Nodal 계산에 직접 들어가는
  macroscopic live XS.
- `_micx`: depletion과 isotope correction에서 사용하는 microscopic live XS.

이 단계에서 CHIFFON의 객체 중심 자료구조는 RASBERY의 노드 중심 SoA 구조로
바뀐다. 전노심 반복 루프에서 `Model` 객체를 계속 탐색하지 않기 위한 선택이다.

CHIFFON HDF5의 `branch_deltas` registry는 RASBERY 내부에서 고정 branch slot으로
평탄화된다. 일반 branch는 현재 노드 상태에서 바로 branch 축 값을 얻고,
SPCT/RHST는 metadata를 이용해 축 값을 계산한다. SPCT의 축 값은 현재 isotope
density vector에서 만들어진 scalar이고, RHST의 축 값은 node별
`rodded_burnup`이다.

## 노드별 XS 갱신

노드별 XS 갱신은 대체로 다음 순서다.

```text
state update
  -> XSSet::PrecomputeBranchCoefficients(...)
  -> XSSet::UpdateFlatXS(...)
  -> solver uses XSSet live arrays
```

`PrecomputeBranchCoefficients()`는 각 노드의 ctype, burnup, branch state에 맞는
reference slot과 delta coefficient slot을 미리 고른다. 그 뒤
`UpdateFlatXS()`가 isotope density, branch correction, rod state를 반영해
live XS 배열을 다시 만든다.

제어봉이 없는 일반 노드는 unrodded `ctype = 0` reference를 기준으로 갱신된다.
제어봉이 들어간 노드는 `XSSet::UsesRodXS()`와 node별 `_ctyp`, `_rod_fraction`에
따라 rodded reference와 unrodded reference를 조합하거나 rodded XS를 사용한다.

## Rod Insertion과 Cusping

`IO`는 rod map과 rod profile을 읽어 `XSSet`에 전달한다. Schedule이나 search가
rod insertion depth를 바꾸면 `Driver`는 `XSSet::SetRod()`를 호출한다.

`SetRod()`는 다음 상태를 갱신한다.

- coarse node의 `_rod_fraction`
- coarse node의 `_ctyp`
- fine axial rod-state mesh의 `_fine_rod_type`
- live XS 배열

Rod cusping correction은 global solver를 새로 추가하지 않고, XS 갱신 직후의 local
보정으로 처리한다. `Driver`는 Nodal 계산 후 얻은 axial surface flux/current와
z 방향 transverse leakage 계수를 사용해 `XSSet::ApplyRodCusping()`을 호출한다.

```text
Nodal solve
  -> ApplyRodCusping(eigv, axial transverse leakage)
  -> CMFD dtil update
  -> Nodal/CMFD coupling update
```

cusping은 partially rodded tip node를 찾고, 그 node와 위/아래 node까지 총 세 개의
coarse axial node를 내부 고정 fine mesh로 나눈다. Fine mesh에서는
rodded 구간에 rodded XS를, unrodded 구간에 unrodded XS를 사용해 1-D axial FDM을
푼다. RHS에는 Nodal의 quadratic transverse leakage를 source에서 빼서 넣는다.
그 fine flux로 group-wise rodded flux-volume fraction을 구하고, 그 alpha로
unrodded/rodded XS를 섞어 coarse node XS를 갱신한다.

## Depletion 연결

Fuel depletion은 RASBERY의 `XSSet`이 isotope density와 burnup을 직접 갱신한다.
CHIFFON은 이때 필요한 microscopic XS와 depletion chain 정보를 제공한다.

Rod depletion은 CHIFFON의 `rod depletion` HGC에서 ctype별 `fluence -> delta XS`
함수로 만들어진다. RASBERY의 `XSSet`은 `_fine_rod_type`과 같은 full-core
fine axial layout에 `_fine_rod_fluence`를 저장하고, predictor/corrector depletion
단계에서 절대 flux와 dt로 fluence를 전진시킨다. Rod XS를 만들 때는 coarse node
안의 같은 ctype fine cells 평균 fluence로 rod-material delta XS를 적용한다.

## Interface Contract

두 코드 사이에서 특히 유지해야 할 convention은 다음과 같다.

- `ctype = 0`은 unrodded reference를 뜻한다.
- 양수 `ctype`은 rodded/reference/rod-history trajectory를 구분하는 제어봉
  type key로 사용된다.
- RASBERY의 coarse burnup은 MWd/t 단위이고, CHIFFON reference burn key는
  `burnup / 1000` convention을 함께 사용한다. 기존 변환 위치를 바꾸면 library
  lookup이 흔들릴 수 있다.
- Branch variable은 주로 `bppm`, `tfuel`, `dmod`이며, CHIFFON delta fitting과
  RASBERY live XS update가 같은 의미로 해석해야 한다.
- Rod insertion depth는 fuel 상단에서 아래 방향으로 잰다.
- Fine rod-state mesh의 axial 세분화 수는 내부 기본값 10을 사용한다.
- Fine rod fluence는 restart file의 `/fine_rod_fluence`에 저장된다.

## 이번 CHIFFON 정리 사항

이번 정리는 numerical behavior를 바꾸지 않는 범위에서 통일감과 가독성을 맞추는
쪽에 집중했다.

- `Importer.h`에서 isotope ratio power 구성, rod-history current ctype parsing,
  extra isotope XS list parsing에 있던 작은 one-off 람다를 명시적인 루프와
  조건문으로 정리했다.
- `Exporter.h`에서 별도의 local append 람다를 제거하고, 이미 존재하는
  `AppendFlat()` 경로를 사용하도록 통일했다.
- `Model.h`에서 branch delta 적용을 작은 람다로 감추지 않고, BPPM/TFUL/DMOD
  delta 적용 순서를 직접 드러냈다.
- `Interpolator.h`에서 trajectory reference search의 local scan 람다를 두 단계
  loop로 바꾸고, 사용하지 않는 rod-history 인자를 제거했다.
- `Interpolator.h`의 history/spectral fitting 블록에 남아 있는 람다는 여러
  fit data structure를 공유하는 큰 local procedure이다. 작은 helper로 흩어 놓으면
  correction fitting 흐름을 추적하기 어려워져서, 현재는 한 블록 안에서 유지한다.
