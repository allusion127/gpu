# RASBERY 코드 이해 및 CUDA 포팅 준비 문서

작성일: 2026-07-03  
대상 경로: `C:\WorkSpace\Personal_Research\RASBERRY_GPU\Rasbery-main\Rasbery-main`

## 0. 이 문서의 목적

이 문서는 RASBERY 소스코드를 처음 보는 사람이 다음을 이해하도록 돕기 위한 문서이다.

1. 현재 디렉토리에 어떤 파일과 디렉토리가 있는지
2. RASBERY가 입력을 받아 어떤 순서로 노심 계산을 수행하는지
3. 핵심 C++ 코드가 어떤 역할을 하는지
4. 계산 흐름을 pseudo code로 보면 어떤 구조인지
5. GPU CUDA로 옮길 때 무엇을 먼저 보고, 무엇을 조심해야 하는지
6. CUDA 포팅 후 어떤 validation/test를 우선 수행해야 하는지

이 저장소에는 반복 생성된 검증 데이터가 매우 많다. 전체 파일 수는 약 3,500개 이상이며, 특히 `test` 아래에는 같은 형식의 `.json`, `.HGC`, `.inp`, `.sum`, `.out`, `.csv`, `.png`가 대량으로 존재한다. 따라서 반복 산출물은 하나씩 나열하지 않고, 디렉토리와 파일 패턴 단위로 의미를 정리한다. 소스코드, 핵심 스크립트, 문서 파일은 파일별로 따로 설명한다.

## 1. 한 줄 요약

RASBERY는 JSON 입력 deck과 CHIFFON HDF5 단면 라이브러리를 읽고, 2-group neutron diffusion eigenvalue 문제를 CMFD, BiCGSTAB, Nodal 보정, rod cusping, thermal-hydraulic feedback, depletion predictor-corrector, pin power reconstruction 순서로 계산하는 C++23 노심 설계 nodal code이다.

CUDA 포팅 관점에서는 `IO`, `Geometry 초기화`, `Scheduler`, `CHIFFON import/export`는 CPU에 두고, `CMFD 계수 조립`, `A*x`, vector reduction, `Nodal` phase loop, `XSSet`의 SoA 기반 단면 재구성, `PPR` pin power reconstruction부터 GPU로 옮기는 전략이 가장 현실적이다.

## 2. 핵심 용어

| 용어 | 의미 |
|---|---|
| RASBERY | Reactor Analyzer for Statics, Burnup Evaluation and tRansient analYsis. 이 저장소의 노심 해석 실행 코드. |
| CHIFFON | DeCART2D HGC 같은 lattice 결과를 읽고 RASBERY용 HDF5 단면 라이브러리로 만드는 전처리/단면 라이브러리 계층. |
| HGC | Homogenized Group Constants로 추정된다. DeCART2D lattice 계산에서 나온 군정수 파일이며 CHIFFON의 원천 입력이다. |
| XS | Cross Section, 단면. 확산계수, 흡수, 산란, 핵분열, nu-fission 등. |
| CMFD | Coarse Mesh Finite Difference. 노드 평균 flux를 풀기 위한 coarse mesh diffusion 가속/해법. |
| BiCGSTAB | Biconjugate Gradient Stabilized. CMFD 선형계를 반복적으로 푸는 Krylov solver. |
| Nodal | SENM/FENM nodal method. 노드 내부 flux shape와 surface current/flux를 보정한다. |
| dtil | CMFD finite-difference coupling coefficient. 두 노드 사이 확산 연결 강도. |
| dhat | Nonlinear current correction coefficient. Nodal 계산과 CMFD 사이의 current mismatch를 보정한다. |
| PPR | Pin Power Reconstruction. 노드 평균 flux에서 pin-wise power와 `FRP/FQP`를 재구성한다. |
| TH | Thermal-Hydraulic feedback. 출력 분포로부터 연료/감속재 온도와 밀도를 갱신한다. |
| Depletion | 연소 계산. flux와 microscopic XS를 사용해 핵종 밀도와 burnup을 시간에 따라 갱신한다. |
| IISC/RHST/RDPL | CHIFFON/RASBERY의 spectral/rod-history 단면 보정 방법론. 현재 `XSSet.cpp` 런타임 단면 갱신과 `test/9-1_IISC` 검증에 직접 연결된다. |

## 3. 전체 실행 흐름

RASBERY executable은 두 종류의 작업을 수행한다.

| 실행 옵션 | 역할 |
|---|---|
| `--chiffoni input.json --chiffono output.h5` | CHIFFON 입력 JSON을 읽고 HDF5 단면 라이브러리를 만든다. |
| `--rasi input.json --raso output.h5` | RASBERY 노심 해석 JSON을 읽고 HDF5 결과를 만든다. |

가장 중요한 실행 흐름은 `--rasi/--raso`이다.

```text
main.cpp
  OpenMP 환경 설정
  CLI 옵션 파싱
  Chiffon::Isotope::Initialize(...)

  for each --chiffoni/--chiffono:
    Chiffon::Importer.ReadInput(...)
    Chiffon::Exporter::SaveHDF(...)

  for each --rasi/--raso:
    rasbery::Driver(input_json, output_h5).Drive()

Driver::Drive()
  Geometry, Scheduler, XSSet, IO 생성
  IO.ReadInput(input_json)
    JSON 파싱
    Geometry.Initialize(...)
    XSSet.Initialize(cross_section_h5)
    restart/shuffle/rod 설정 복원

  BICGCMFD, Nodal, PPR 생성
  XSSet.InitXS(...)
  result.h5 open

  for each schedule step:
    schedule 조건 적용
    depletion/derivative/rod 사전 작업
    SolveLoop(...)
    PPR pin power reconstruction
    IO.AddResult(...)
    IO.WriteStepToResult(...)
    필요 시 restart 저장

  result.h5 close
```

## 4. 초보자용 최상위 pseudo code

아래 pseudo code는 실제 C++ 문법이 아니라, 계산이 어떤 순서로 진행되는지 이해하기 위한 흐름이다.

```text
프로그램 시작:
  병렬 계산에 쓸 OpenMP thread 수를 설정한다.
  사용자가 준 입력 파일과 출력 파일 목록을 읽는다.

  단면 라이브러리 생성 요청이 있으면:
    HGC/JSON 기반 CHIFFON 입력을 읽는다.
    RASBERY가 쓸 수 있는 HDF5 단면 파일을 저장한다.

  노심 해석 요청이 있으면:
    각 입력 JSON마다 Driver를 실행한다.

Driver 실행:
  1. JSON 입력을 읽는다.
     - geometry: 노심 격자, 대칭, albedo, 축방향 mesh
     - core/batch: assembly 배치와 축방향 단면 ID
     - TH: 압력, 입구/출구 온도, 출력
     - schedule: 표준 계산, 연소, 제어봉, 미분계수, critical search
     - rod map/configuration: 제어봉 위치와 삽입 profile
     - data.cross-section: CHIFFON HDF5 단면 라이브러리

  2. Geometry를 만든다.
     - assembly map을 fine node map으로 바꾼다.
     - 각 node의 이웃 node를 계산한다.
     - 각 surface가 어떤 node 두 개를 연결하는지 만든다.
     - flux, current, surface flux, pin-power 배열을 할당한다.

  3. XSSet을 만든다.
     - HDF5 단면 라이브러리를 읽는다.
     - node마다 어떤 assembly/model을 쓸지 연결한다.
     - burnup 보간, branch delta, microscopic XS, isotope density 배열을 준비한다.

  4. schedule step을 하나씩 계산한다.
     if step이 depletion이면:
       predictor:
         현재 flux로 핵종을 임시 연소시킨다.
         burnup과 단면을 임시 갱신한다.
       predicted 상태에서 neutronics solve를 한다.
       corrector:
         BOS/EOS flux와 단면을 평균해서 핵종을 다시 계산한다.
         최종 burnup과 단면을 갱신한다.

     if step이 derivative이면:
       boron/temperature/density를 조금 바꾸고 단면을 갱신한다.

     if step이 rod insertion이면:
       rod 위치를 바꾸고, 해당 node의 rod fraction과 ctype을 갱신한다.
       flux/current를 초기화한다.

     마지막으로 항상 neutronics solve를 수행한다.
     pin power를 재구성한다.
     결과를 HDF5에 저장한다.
```

## 5. `SolveLoop` pseudo code

`Driver.h`의 `SolveLoop()`가 실제 계산의 중심이다.

```text
SolveLoop(ctx, eigv, schedule):
  if boron search 또는 rod critical search가 있으면:
    search 상태를 초기화한다.
    첫 guess를 단면/제어봉 상태에 반영한다.

  CMFD 반복 카운터 초기화
  dtil 계산

  반복:
    # 1. CMFD flux solve
    현재 flux로 fission source psi 계산
    현재 eigv로 CMFD matrix diag/cc/src 구성
    BiCGSTAB으로 A * flux = src 선형계 풀이
    Wielandt shift로 eigv와 residual 갱신

    # 2. Nodal correction
    flux에서 surface net current 계산
    Nodal solver가 surface flux/current를 재계산
    rod cusping이 필요한 node의 macro XS 보정
    Nodal current와 CMFD current mismatch로 dhat 갱신

    if flux/eigv가 아직 수렴하지 않았으면:
      다시 반복

    # 3. Critical search
    if boron 또는 rod search가 필요하면:
      현재 k_eff와 target k_eff 차이를 계산
      secant/probe/bracket 방식으로 다음 boron ppm 또는 rod step 제안
      새 boron/rod 상태를 XSSet에 반영

    # 4. TH feedback
    if TH feedback이 필요하면:
      현재 flux로 node power 계산
      enthalpy rise로 tmod/dmod/tfuel 계산
      새 온도/밀도에 맞춰 XS 갱신

    if flux, search, TH가 모두 수렴했으면:
      종료
```

핵심은 flux solve와 feedback이 완전히 분리된 nested loop가 아니라, 한 outer loop 안에서 같이 co-converge한다는 점이다.

## 6. 주요 데이터 구조

### 6.1 Geometry

`Geometry`는 계산 격자와 대부분의 큰 해석 배열을 소유한다.

대표 배열:

| 배열 | 의미 | 대략적 shape |
|---|---|---|
| `_phif` | node-averaged scalar flux | `nxyz * ng` |
| `_jnet` | surface net current | `LR * ng * NDIRMAX * nxyz`에 해당 |
| `_phis` | surface flux | `LR * ng * NDIRMAX * nxyz`에 해당 |
| `_psi` | fission source | `nxyz` |
| `_neib` | 3D neighbor index | `nxyz * 6` |
| `_lktosfc` | node side에서 surface index로 가는 map | `nxyz * 3 * 2` |
| `_lklr` | surface의 left/right node | `nsurf * 2` |
| `_hmesh` | 각 node의 x/y/z mesh size | `nxyz * 3` |
| `_vol` | node volume | `nxyz` |
| `_bppm`, `_tful`, `_tmod`, `_dmod` | boron, fuel temp, moderator temp/density | `nxyz` |
| `_rod_fraction` | node별 rod 점유율 | `nxyz` |
| `_pphif`, `_ppower` | pin flux/pin power | assembly, axial, group, pin index |

CUDA 관점:

- 좋은 점: 대부분 1차원 배열이고 index가 명확하다.
- 어려운 점: raw pointer와 class method access가 많다. GPU에서는 `DeviceGeometryView` 같은 POD view가 필요하다.

### 6.2 XSSet

`XSSet`은 단면과 핵종 상태를 관리한다. 현재 구조는 SoA, 즉 structure-of-arrays 방식이다.

대표 index:

```text
macro XS:          xs[ig * nxyz + l]
scatter XS:        xssm[(igs * ng + ige) * nxyz + l]
microscopic XS:    micx[(iso * ng + ig) * nxyz + l]
isotope density:   iden[iso * nxyz + l]
flux in Geometry:  phif[l * ng + ig]
```

주의할 점:

- `XSSet` 내부는 `std::vector`, `milk::Vector`, `std::map`, CHIFFON object가 섞여 있다.
- GPU 커널에 그대로 넣을 수 있는 것은 최종 평탄화된 배열뿐이다.
- CHIFFON object, HDF5, JSON, `std::map` 기반 로직은 host-only로 유지하는 편이 맞다.

## 7. 디렉토리 및 파일 구조

### 7.1 루트 디렉토리

| 경로 | 역할 |
|---|---|
| `.vscode/` | VS Code CMake/Ninja build, debug, extension 설정. Linux/WSL 중심 경로가 일부 들어 있다. |
| `docs/` | CMFD/Nodal, CHIFFON interface, IISC/RHST, rod cusping, 성능 수정 기록 등 설계/개발 문서. |
| `include/` | vendored dependency와 CHIFFON header-only 코드, depletion/TH 데이터베이스. |
| `src/` | RASBERY solver 본체 C++ 코드. |
| `test/` | CHIFFON/RASBERY validation 입력, reference, 결과, plot, generated deck. |
| `tools/` | HDF5/summary 비교와 validation plot용 Python 스크립트. |

### 7.2 루트 파일

| 파일 | 역할 |
|---|---|
| `README.md` | RASBERY 개요, Linux 의존성 설치, 테스트 안내. 현재 `Viewer/requirements.txt`, `test/Tests.txt` 등 실제 트리와 맞지 않는 안내가 일부 있다. |
| `CMakeLists.txt` | CMake build 정의. C++23, HDF5 필수, OpenMP 선택, `src/*.cpp` glob으로 `RASBERY` executable 생성. |
| `.regress.sh` | HDF5 baseline 회귀 비교 스크립트. `h5diff`로 `/tmp/ras_base`와 새 결과를 비교한다. 현재는 Linux 절대경로와 오래된 일부 test path를 사용하므로 CUDA 포팅 시 갱신 필요. |
| `.clang-format` | LLVM 기반 4칸 들여쓰기, ColumnLimit 0, pointer left alignment. |
| `.gitignore` | build 산출물, Python cache, HDF5, PNG, PDF 등을 제외. |
| `.gitattributes` | text auto, LF line ending. |
| `.cleanup_findings.txt` | 과거 코드 cleanup 후보 분석 결과. 특히 `XSSet.cpp`의 history vector lambda 중복과 verbose code를 지적한다. |
| `SENM.nb` | Mathematica notebook. 파일명상 SENM 수식/계수 유도 또는 검토용 notebook으로 보인다. |
| `RASBERY_CODE_UNDERSTANDING_AND_CUDA_PORTING.md` | 이 문서. |

## 8. `src/` 소스 파일별 설명

| 파일 | 역할 | CUDA 관점 |
|---|---|---|
| `main.cpp` | CLI 파싱, OpenMP 환경 설정, CHIFFON 변환, RASBERY Driver 실행. | GPU 대상 아님. CUDA build option과 실행 모드 선택은 여기에 들어갈 수 있다. |
| `Driver.h` | 전체 orchestration. `Drive()`와 `SolveLoop()`가 핵심. | host control loop로 유지. GPU kernel launch 순서를 제어하는 위치가 될 가능성이 높다. |
| `IO.h`, `IO.cpp` | JSON 입력 파싱, schedule 확장, restart/shuffle, rod map, HDF5/CSV 출력. | host-only. HDF5/JSON/std::filesystem은 GPU로 옮기지 않는다. |
| `Geometry.h`, `Geometry.cpp` | node, surface, neighbor, assembly index map 생성. flux/current/PPR 배열 소유. | 초기화는 CPU. 생성된 map과 배열 view를 GPU로 복사. |
| `XSSet.h`, `XSSet.cpp` | HDF5 단면 library 로딩, XS flatten, branch/history correction, depletion, TH, rod, cusping. | 가장 중요. SoA 배열 기반 hot loop는 GPU 후보. CHIFFON object와 branch metadata는 CPU에서 flatten 후 device view로 전달. |
| `Scheduler.h` | schedule type, search type, TH/search/depletion parameter, critical search state. | CPU control logic. 검색 알고리즘은 CPU 유지가 현실적. |
| `CMFD.h`, `CMFD.cpp` | CMFD coefficient와 matrix-vector element 계산. `dtil`, `dhat`, `diag`, `cc`, `src`, `psi`. | node/surface loop가 GPU 후보. |
| `BICGCMFD.h`, `BICGCMFD.cpp` | CMFD outer iteration, Wielandt shift, BiCGSTAB 호출, eigenvalue/residual 갱신. | kernel launch orchestration은 CPU. `updpsi`, `setls`, `source`, reduction은 GPU 후보. |
| `BICGSolver.h`, `BICGSolver.cpp` | BiCGSTAB solver, SSOR preconditioner, `A*x`. | `A*x`, vector update, dot reduction은 GPU 후보. SSOR sweep은 재설계 필요. |
| `Nodal.h`, `Nodal.cpp` | SENM/FENM nodal correction. transverse leakage, even/odd coefficient, surface flux/current 계산. | phase별 node/surface loop가 GPU 후보. phase 사이 barrier가 필요하므로 여러 kernel로 나누는 구조. |
| `PPR.h`, `PPR.cpp` | Pin power reconstruction. corner flux, source update, homogeneous/particular coefficient, pin power, FRP/FQP. | pin/assembly/node loop가 GPU 후보. corner iteration과 reductions는 별도 처리. |
| `pch.h` | 방향, 경계, 수치 상수, 물리 상수. | device/host 공용 상수 header로 유지 가능. |
| `Timer.h` | 단순 시간 측정 유틸리티. | GPU timing에는 CUDA event 별도 필요. |

## 9. 핵심 클래스별 상세 pseudo code

### 9.1 `IO::ReadInput`

```text
ReadInput(filepath):
  input_dir 계산
  JSON 파일 로드

  data.cross-section 경로 해석
  data.restart가 있으면 restart 파일 목록 등록

  ParseSchedule(config)
    default parameters 읽기
    TH default 읽기
    convergence default 읽기
    schedule 배열을 Scheduler entry로 확장

  if geometry block이 있으면:
    ng, xydivision, npins, hx, hy, hz, symmetry, albedo 읽기
    core map 읽기
    batch axial layer를 z-layer 배열로 확장
    shuffle entry가 있으면 restart에서 source assembly 정보 복원
  else if restart만 있으면:
    restart HDF5에서 geometry 복원
  else:
    error

  Geometry.Initialize(geometry_input)
  XSSet.Initialize(cross_section_h5)

  axial rod division 설정
  shuffle이 있으면 ApplyShuffle()
  rod configuration과 rod map을 XSSet.rod_groups에 저장
  rod profile matrix 생성

  restart가 있으면:
    burnup, history_ctype, rodded_fluence, isotope_density, TH state, flux 복원
```

### 9.2 `Geometry::Initialize`

```text
Initialize(GeometryInput):
  basic dimension 저장: ng, nz, ndivxy, npins
  symmetry와 albedo 저장

  core map을 보고 실제 계산 영역의 nx, ny 계산
  각 row/column의 시작/끝 node index 계산
  (i,j) -> 2D node index l map 생성

  2D neighbor map 생성: west/east/north/south
  reflecting boundary이면 neibrb에서 자기 자신으로 mirror 처리

  각 axial plane에 대해:
    3D node index lk = k*nxy + l2d
    3D neighbor map 생성
    hmesh와 volume 계산

  flux/current/source/TH/rod/PPR 배열 할당

  surface map 생성:
    각 surface가 left/right node 중 어느 것과 연결되는지 저장
    각 node side에서 surface index로 가는 lktosfc 생성

  assembly map 생성:
    assembly index la
    node -> assembly
    assembly local subnode -> node

  fuel flag 계산:
    batch layer ID가 'R'로 시작하면 reflector로 본다.

  active core axial range kbc/kec 계산
  pin power reconstruction 배열 할당
```

### 9.3 `XSSet::Initialize`와 `InitXS`

```text
XSSet.Initialize(xs_path):
  node-level storage 할당:
    comp, assembly, ctyp, burn, isotope density, rod fluence

  TH property table 로드:
    mod_cp, mod_h, mod_rho, mod_t, tf

  CHIFFON HDF5 단면 library 로드
  depletion decay/transition matrix 로드

  macro XS, microscopic XS, isotope density 배열 할당
  reference XS와 branch coefficient 배열 할당

  node마다 어떤 CHIFFON model을 사용할지 comp map 생성
  CHIFFON model의 reference depletion point와 branch delta를 flat SoA 배열로 변환
  node별 burnup bracket과 delta bracket lookup table 준비

XSSet.InitXS(bppm, tful, tmod, pressure):
  각 node의 bppm, tful, tmod, dmod 초기화
  각 node에서 CHIFFON model.GetCrossSection(...) 호출
  isotope density와 microscopic/lumped XS를 채운다
  PrecomputeBranchCoefficients()
  UpdateFlatXS()
```

### 9.4 `XSSet::UpdateFlatXS`

이 함수는 단면 갱신의 hot path이다.

```text
UpdateFlatXS(options):
  if flat coefficient 준비가 안 되어 있으면:
    느린 object 기반 Update() 수행
    return

  평균 moderator density/temperature 계산

  if 전체 node 갱신이고 reference restore가 필요하면:
    reference lumped/micro XS를 live 배열로 복사

  for each target node:
    if boron_difference mode:
      기존 boron delta를 빼고 새 boron delta를 더한다.
      H/B/O light isotope를 새 bppm/dmod 기준으로 refresh한다.
      continue

    if 일부 node만 갱신하면:
      해당 node reference XS 복원

    if rod XS를 쓰는 node이면:
      FillRodNodeXS(l)
      ApplyHistoryDeltasToNode(l)
    else:
      BPPM, TFUL, DMOD branch delta 적용
      IISC/RHST history delta 적용
      light isotope refresh

  if 전체 node:
    Reconstruct()
  else:
    ReconstructNode(l) for updated nodes
```

CUDA 관점에서 `UpdateFlatXS`는 좋은 후보지만 분기와 metadata 참조가 많다. 1차 포팅은 `Reconstruct()`처럼 단순 SoA accumulation부터 시작하는 것이 안전하다.

### 9.5 `XSSet::Reconstruct`

```text
Reconstruct():
  for each scalar XS type:
    if XSDF 또는 XSRF는 건너뜀
    dst = lumped XS
    for each isotope:
      for each group:
        for each node:
          dst[group,node] += micro_xs[iso,group,node] * isotope_density[iso,node]

  scatter matrix도 같은 방식으로 합산

  for each group,node:
    diffusion coefficient XSDF = 1 / (3 * transport XS)

  for each from_group,node:
    removal XSRF = absorption + outgoing scattering sum
```

이 함수는 CUDA 포팅 1순위 후보이다. 배열 접근이 규칙적이고, node별 독립 계산이 많다.

### 9.6 `BICGCMFD::drive`

```text
drive(eigv, flux):
  reigv = 1 / eigv
  reigvs = 1 / (eigv + eshift)

  for cmfd iteration:
    reigvdel = reigv - reigvs

    for each node:
      for each group:
        src[group,node] = chi[group,node] * psi[node] * reigvdel

    BiCGSTAB reset residual
    BiCGSTAB solve several times

    Wielandt update:
      새 psi 계산
      eigv, reigv, shifted reigvs, residual 갱신

    if shift가 있으면:
      shifted diag 갱신
      preconditioner 다시 factorize

    negative flux check
    if residual < tolerance:
      break
```

### 9.7 `BICGSolver::solve`

```text
BiCGSTAB solve:
  rho = dot(r0, r)
  beta = rho * alpha / (old_rho * omega)

  p = r + beta * (p - omega * v)
  y = M^-1 p                 # SSOR preconditioner
  v = A y

  alpha = rho / dot(r0, v)
  s = r - alpha * v

  z = M^-1 s                 # SSOR preconditioner
  t = A z

  omega = dot(s,t) / dot(t,t)
  phi = phi + alpha*y + omega*z
  r = s - omega*t
  residual = norm(t) / initial_residual
```

CUDA에서 `dot`, vector update, `A*y`, `A*z`는 GPU에 적합하다. 그러나 `M^-1`이 현재 SSOR forward/backward sweep이라 GPU에 적합하지 않다.

### 9.8 `BICGSolver::minv`

```text
SSOR preconditioner:
  forward sweep l = 0 to nxyz-1:
    tmp[l] = D^-1 * (b[l] - lower_neighbor_terms_already_computed)

  backward sweep l = nxyz-1 to 0:
    x[l] = D^-1 * (D*tmp[l] - upper_neighbor_terms_already_computed)
```

이 구조는 `l` 순서 의존성이 있다. CUDA에서는 모든 node를 동시에 처리하기 어렵다. 이 부분은 다음 중 하나로 재설계해야 한다.

- block-Jacobi preconditioner
- multi-color Gauss-Seidel/SSOR
- cuSPARSE SpMV + cuSPARSE/AmgX 계열 preconditioner
- CPU SSOR 유지 + GPU matvec 혼합
- 초기 포팅 단계에서는 preconditioner를 단순화하고 convergence 차이를 validation으로 확인

### 9.9 `Nodal::drive`

현재 `Nodal::drive()`는 하나의 OpenMP parallel region 안에서 phase별 barrier를 둔다.

```text
Nodal drive:
  phase 1: for each node
    updateConstant(lk)

  phase 2: for each node
    caltrlcff0(lk)

  phase 3: for each node
    caltrlcff12(lk)

  phase 4: for each node
    updateMatrix(lk)

  phase 5: for each node
    calculateEven(lk)

  phase 6: for each surface
    calculateJnet(ls)
```

CUDA에서는 phase마다 kernel을 나누고, kernel launch 사이의 global synchronization을 barrier로 사용하면 된다.

예상 CUDA 구조:

```text
kernel_updateConstant<<<...>>>
kernel_caltrlcff0<<<...>>>
kernel_caltrlcff12<<<...>>>
kernel_updateMatrix<<<...>>>
kernel_calculateEven<<<...>>>
kernel_calculateJnet<<<...>>>
```

주의:

- `ng=2` 하드코딩 형태의 작은 2x2 matrix 계산이 많다.
- `std::exp`, `sqrt` 같은 transcendental 연산이 많아 double precision GPU 성능 영향을 확인해야 한다.
- 결과가 CPU와 bitwise 같을 가능성은 낮고, tolerance 기반 비교가 필요하다.

### 9.10 `PPR::reconstructPinPower`

```text
PPR:
  reset:
    node/group별 buckling Bt 계산
    corner flux 초기화
    polynomial fitting coefficient c 계산
    axial leakage와 source q 계산

  drive(niter):
    반복:
      여러 번:
        for group,node:
          particular + homogeneous + projectFlux 계수 갱신
        source 갱신
      corner flux continuity 갱신
      corner flux sum 변화가 작으면 종료

  reconstructPinPower:
    for axial plane, assembly, pin:
      해당 pin이 속한 sub-node를 찾는다.
      homogeneous flux를 point 또는 quadrature로 평가한다.
      form function fmap/gmap을 burnup 보간한다.
      pin flux/power 저장

    nodal average power로 pin power normalize
    FRP = z-averaged radial pin power 최대값
    FQP = 3D pin power 최대값
```

CUDA 후보:

- pin별 reconstruction은 병렬성이 매우 높다.
- FRP/FQP는 reduction 필요.
- `updateCorner`는 stencil neighbor 참조와 반복 dependency가 있으므로 kernel 단계 분리 필요.

### 9.11 Depletion predictor-corrector

```text
PredictorStep:
  BOS 상태 저장:
    XS, microscopic XS, isotope density, burnup, flux, rod fluence
  BOS flux로 Deplete()
  rod material fluence 갱신
  burnup 임시 advance
  XS 갱신

CorrectorStep:
  BOS flux/rate와 EOS flux/rate를 평균하거나 EOS rate 사용
  for each node:
    microscopic XS를 flux-weighted condensation
    Bateman transition matrix 생성
    CRAM solver로 isotope density 계산
    Xe equilibrium이면 I/Xe를 equilibrium 값으로 덮어씀
    burnup, history ctype, rodded fluence 갱신
  rod material fluence 복원/갱신
  branch coefficient 재계산
  XS 갱신
```

CUDA 관점:

- node별 depletion은 독립이라 병렬화 가능하다.
- 하지만 각 node마다 `niso x niso` Bateman/CRAM dense solve가 들어간다.
- isotope 수가 현재 CHIFFON 기준 약 39종이므로, batched small dense solver 형태가 필요하다.
- 초기 CUDA 포팅에서는 depletion을 CPU에 남기고 flux solve만 GPU화하는 전략이 더 안전하다.

## 10. `include/` 구조

| 경로/파일 | 역할 |
|---|---|
| `include/milk.h` | 자체 header-only 수치 유틸리티. aligned `Vector/Matrix`, table interpolation, BLAS-like dot/copy/addScaled/multiply, LU solve/inverse, CRAM Bateman solver 포함. |
| `include/chiffon/Model.h` | isotope registry, XS enum, branch/history enum, `CrossSection`, `DeltaCrossSection`, `DepletionPoint`, `Model` 정의. |
| `include/chiffon/Importer.h` | CHIFFON JSON, HGC, HDF5 읽기. rod fluence 재구성, history tagging 등. |
| `include/chiffon/Exporter.h` | CHIFFON model을 HDF5 library로 저장. |
| `include/chiffon/Interpolator.h` | burnup/branch interpolation, IISC/RHST/RDPL fitting. |
| `include/chiffon/Benchmark.h` | CHIFFON 검증/비교 보조. |
| `include/chiffon/ReflectorSolver.h` | reflector 관련 보조 solver. |
| `include/Database/dep_decay.csv` | depletion decay matrix. |
| `include/Database/dep_trans.csv` | depletion transmutation matrix. |
| `include/Database/mod_cp.csv` | moderator heat capacity table. |
| `include/Database/mod_h.csv` | moderator enthalpy table. |
| `include/Database/mod_rho.csv` | moderator density table. |
| `include/Database/mod_t.csv` | moderator temperature table. |
| `include/Database/tf.csv` | fuel temperature increment table. |
| `include/highfive/` | vendored HighFive HDF5 C++ wrapper. |
| `include/nlohmann/` | vendored nlohmann JSON. |
| `include/plog/` | vendored plog logging. |

CUDA 관점:

- `highfive`, `nlohmann`, `plog`, CHIFFON importer/exporter는 host-only.
- `milk::Vector/Matrix/Solver`는 그대로 device 코드에서 쓰기 어렵다. device buffer와 POD view를 따로 만드는 것이 좋다.
- `dep_decay`, `dep_trans`, TH table은 GPU에 올릴 수 있지만, 1차 포팅에서는 CPU 유지가 안전하다.

## 11. `docs/` 문서 파일

| 파일 | 내용 |
|---|---|
| `docs/CHIFFON_manual.md` | CHIFFON 입력 매뉴얼. branch delta, state delta, isotope vector, rod depletion, burnup interpolation 예시. |
| `docs/chiffon-rasbery-interface.md` | CHIFFON과 RASBERY의 역할 분담, `--chiffoni`, `--rasi`, XSSet 연결, rod/cusping/depletion interface. |
| `docs/cmfd_nodal_preconditioner_ko.md` | CMFD, Nodal, SSOR preconditioner를 미니코어 예시로 설명. CUDA 포팅 시 solver 이해에 가장 유용한 기존 문서. |
| `docs/code_cleanup_summary_ko.md` | 2026-05-29 코드 정리 요약. |
| `docs/history_simplification_ko.md` | history correction 단순화, rod fluence HGC 재구성, 검증 요약. |
| `docs/iisc-implementation-ko.md` | IISC/RHST/RDPL 구현 상세. `XSSet.cpp` history correction 이해에 중요. |
| `docs/perf_and_cusping_fixes_ko.md` | 성능/수렴/rod cusping 수정 기록. OpenMP gate, TH relaxation, 실패한 red-black SSOR 시도 포함. |
| `docs/rasbery-input-tolerance.md` | JSON 입력에서 유지되는 convergence key 설명. |
| `docs/rod-depletion-prep.md` | fine rod-state mesh와 rod material depletion 구현 메모. |
| `docs/rod_cusping_rework_ko.md` | PARCS 방식 rod cusping 재구현과 속도 개선 요약. |
| `docs/Spectral and rod history correction.md` | spectral/rod-history correction의 legacy/keyed 방법론과 fitting 설명. 현재 기준 문서는 아니지만 배경 자료. |
| `docs/_gen_chain.py` | depletion chain SVG 생성 Python script. |
| `docs/depletion-chain.svg` | depletion/transmutation chain 그림. |

## 12. `tools/` 파일

| 파일 | 역할 |
|---|---|
| `tools/compare_keff.py` | RASBERY HDF5 `summary/keff`와 reference `.sum`의 keff를 비교한다. bias, RMS, max error를 `pcm` 성격의 `x1e5` 값으로 출력. |
| `tools/compare_micxs.py` | HDF5 `node_monitor`에 저장된 microscopic XS와 reference/baseline microscopic XS를 isotope별로 비교한다. |
| `tools/plot_ismr_validation.py` | `test/7_i-SMR_Validation`의 RASBERY HDF5와 reference `.sum`을 읽어 rod position, AO/FQP, assembly power/burnup plot을 만든다. |

## 13. `test/` 구조

### 13.1 전체 파일 유형

대표 확장자 개수는 다음과 같다.

| 확장자 | 개수 | 의미 |
|---|---:|---|
| `.json` | 약 1,300개 이상 | RASBERY solver deck 또는 CHIFFON library build deck |
| `.HGC` | 약 390개 | DeCART2D homogenized group constants, CHIFFON 원천 입력 |
| `.inp` | 약 360개 | DeCART2D 원 입력 deck |
| `.sum` | 약 350개 | DeCART2D/MASTER summary reference |
| `.out` | 약 340개 | DeCART2D 실행 로그와 echo |
| `.csv` | 약 290개 | summary, error metric, manifest |
| `.png` | 약 110개 | validation plot |
| `.BRCH` | 약 35개 | DeCART branch 관련 부산물로 추정. 대부분 0 byte라 현재 regression 핵심은 아님. |
| `.py` | 약 18개 | test generation, validation, plotting script |

### 13.2 `test/` 하위 디렉토리별 의미

| 디렉토리 | 목적 | 주요 파일/패턴 |
|---|---|---|
| `test/2_Single_Depletion/` | 단일 assembly depletion smoke test. | `CE16.json`, `CE16_Gd.json`, `WH17.json`, `WH17_Gd.json` |
| `test/3-1_Colinear/` | colinear XS 보간/indicator 비교. | `Base/Pu/PuXe/IISC` 계열 JSON, `LEU_base_0101.HGC`, 결과 CSV/PNG |
| `test/3-2_Colinear_HIGA/` | HIGA variant colinear 비교. | HIGA HGC, reference/PuXe 결과 CSV |
| `test/3-3_Colinear_TPGrid/` | HIGA/NonHIGA, moderator temperature 550-610 K, power 25-100%, Base/Pu/IISC grid sweep. | `manifest.csv`, `cases/`, `summaries/`, `.HGC/.inp/.sum/.out` |
| `test/3-4_HGC_Isotope_Diff_60BU/` | 60 MWd/kgHM HGC isotope density와 isotope ratio 차이 분석. | `compare_hgc_isotopes.py`, `collect_*`, 결과 CSV, README |
| `test/3_Full_Depletion/` | iSMR/assembly full depletion 기본 검증. | `iSMR_noTH.json`, `WH17_*`, `PLUS7_*`, `.sum`, `keff.csv` |
| `test/4_TH/` | 1D/3D thermal-hydraulic coupled validation. | `TH_1D.json`, `TH_3D.json`, reference `.sum`, 비교 CSV/PNG |
| `test/5_Criticality/` | boron search와 rod criticality search 검증. | `Boron_search.json`, `Rod_search.json`, `i-SMR_CY01_Boron.json`, `i-SMR_CY01_Rod.json` |
| `test/6_Restart_shuffle/` | restart와 fuel shuffle 후 burnup/isotope/XS 재배치 검증. | `Base.json`, `2D_Restart.json`, `2D_Shuffle.json`, `plot_shuffle_check.py` |
| `test/7_i-SMR_Validation/` | i-SMR cycle 1-4 full-core validation. | `i-SMR_CY01.json` to `CY04`, `Reference_input/`, `Reference_output/`, plot script |
| `test/8-1_PWR_Validation_YG03/` | YG03 PWR cycle 입력 deck. | `YG03_CY01.json`, `YG03_CY02.json` |
| `test/9-1_IISC/` | CE16 9-case IISC/RHST homogenization validation harness. | `run_iisc.py`, per-case generated input/summary/plot/docs |
| `test/CrossSections/` | CHIFFON library source와 DeCART2D reference set. | `1_Test_Assemblies`, `2_i-SMR_Validation`, `5_IISC_Test` 등 |
| `test/Tests.md` | 대표 build/run command 모음. i-SMR validation command 포함. |

### 13.3 `test/CrossSections/`

| 디렉토리 | 의미 |
|---|---|
| `1_Test_Assemblies/` | CE16/WH17/FA_A1 등 test assembly HGC와 CHIFFON JSON. |
| `2_i-SMR_Validation/` | i-SMR validation용 DeCART2D input과 HGC library source. |
| `3_PWR_Validation/` | PWR validation용 CHIFFON JSON skeleton. |
| `4_Colinear_Test_old/` | 오래된 colinear test set. |
| `5_IISC_Test/` | CE16 Plain/Gd/GdE, enrichment R4/R7/R10 등 IISC validation의 대형 원천 데이터. `.HGC`, `.inp`, `.out`, `.sum`, `.BRCH`가 대량 존재. |

## 14. Validation과 regression 전략

### 14.1 현재 validation 방식

현재 저장소는 두 축의 validation을 가진다.

1. HDF5 baseline regression
   - `.regress.sh`
   - RASBERY 실행 결과 `.h5`를 baseline과 `h5diff`로 비교
   - tolerance: 보통 `1e-9`, boron/i-SMR 일부 `1e-6`

2. 물리 reference 비교
   - `.sum` reference와 HDF5 결과 비교
   - `tools/compare_keff.py`
   - `tools/plot_ismr_validation.py`
   - `test/9-1_IISC/run_iisc.py`

### 14.2 CUDA 포팅 후 우선 테스트

추천 순서:

1. CPU baseline을 새로 만든다.
   - 같은 compiler, 같은 input, 같은 HDF5 output을 보관한다.
   - CUDA 결과는 이 baseline과 비교한다.

2. 작은 smoke test부터 실행한다.
   - `test/2_Single_Depletion/*.json`
   - `test/3_Full_Depletion/iSMR_noTH.json`
   - `test/4_TH/TH_1D.json`, `TH_3D.json`

3. critical search 검증.
   - `test/5_Criticality/Boron_search.json`
   - `test/5_Criticality/Rod_search.json`

4. restart/shuffle 검증.
   - `test/6_Restart_shuffle`
   - GPU memory indexing이나 assembly/node mapping 오류를 잘 잡는다.

5. i-SMR full-core validation.
   - `test/7_i-SMR_Validation/i-SMR_CY01.json`부터 시작
   - 이후 CY02-CY04

6. IISC/RHST validation.
   - `test/9-1_IISC/run_iisc.py --case CASE1_CE16_Plain_R4`
   - 이후 전체 9 case

7. 넓은 parameter sweep.
   - `test/3-3_Colinear_TPGrid`

### 14.3 GPU에서 허용오차를 재정의해야 하는 이유

GPU는 reduction 순서가 CPU와 달라질 가능성이 높다. 따라서 다음 값은 bitwise identical 비교를 기대하면 안 된다.

- eigenvalue residual
- dot product
- fission source sum
- norm factor
- TH total power
- PPR FRP/FQP max/reduction
- boron/rod search secant 경로

권장:

- CPU와 GPU HDF5 결과를 field별 tolerance로 비교한다.
- `summary/keff`, `summary/fqn/frn/fqp/frp`, `summary/ppm`, `summary/tf_avg/tm_avg/dm_avg`, `flux`, `xs`, `pin_power`를 따로 비교한다.
- `/summary/reactivity`처럼 near-zero derived field는 상대오차보다 절대오차 기준을 둔다.

## 15. CUDA 포팅 전략

### 15.1 CPU에 남길 부분

다음은 GPU로 옮겨도 이득이 작거나 구현 위험이 크므로 CPU에 유지한다.

| 부분 | 이유 |
|---|---|
| JSON parsing | `nlohmann::json`, 문자열, map 중심. 계산 병목 아님. |
| HDF5 read/write | HighFive/HDF5는 host I/O. GPU kernel 대상 아님. |
| CHIFFON importer/exporter | HGC parsing, model object, fitting, map/vector 중심. |
| Geometry initialization | 한 번만 실행되는 index map 생성. |
| Scheduler/search control | 분기와 상태 machine 중심. |
| Result aggregation/output | I/O와 summary formatting 중심. |

### 15.2 1차 CUDA 후보

| 후보 | 이유 | 예상 kernel |
|---|---|---|
| `BICGSolver::axb` | 이미 OpenMP 병렬. node별 독립 SpMV 형태. | one thread per node/group 또는 one block per node |
| vector update/dot | BiCGSTAB에서 반복 사용. | cuBLAS 또는 custom reduction |
| `CMFD::setls` | node별 diag/cc 조립. | one thread/block per node |
| `CMFD::upddtil/upddhat/updjnet` | surface별 독립. | one thread per surface/group |
| `BICGCMFD::updpsi`, source build | node/group별 독립. | one thread per node |
| `XSSet::Reconstruct` | SoA accumulation. | isotope/group/node tiled kernel |
| `XSSet::NormFactor`, `UpdateBurnup`, `UpdateTH` 일부 | node reduction과 node별 계산. | reduction + node kernel |
| `Nodal::drive` phases | phase 사이 barrier만 있으면 node/surface 독립. | phase별 kernel |
| `PPR::reconstructPinPower` | pin별 병렬성이 큼. | one thread/block per assembly-plane-pin |

### 15.3 2차 또는 재설계 후보

| 후보 | 어려움 | 가능한 접근 |
|---|---|---|
| `BICGSolver::minv` SSOR | forward/backward 순차 의존성 | block-Jacobi, multi-color, cuSPARSE, CPU fallback |
| `XSSet::UpdateFlatXS` 전체 | branch/history correction 분기와 metadata 많음 | 먼저 CPU에서 flattened delta를 만들고 GPU kernel은 단순 apply만 수행 |
| `XSSet::DepleteNode` | node별 small dense CRAM solve | batched small matrix solver, 또는 CPU 유지 |
| `ApplyRodCuspingStencil` | touched list, dynamic vector, small dense solve | CPU 유지 후 필요 시 batched kernel |
| `PPR::updateCorner` | iterative stencil update | phase kernel + convergence reduction |

### 15.4 권장 CUDA 아키텍처

직접 `Geometry`와 `XSSet` class를 CUDA kernel에 넣지 말고, GPU 전용 view를 만든다.

```cpp
struct DeviceGeometryView {
    int ng, nxy, nxyz, nsurf, nz, kbc, kec;
    const int* neib;
    const int* lklr;
    const int* idirlr;
    const int* lktosfc;
    const double* hmesh;
    const double* vol;
    double* phif;
    double* jnet;
    double* phis;
    double* psi;
};

struct DeviceXSView {
    int ng, nxyz;
    const double* xsdf;
    const double* xstf;
    const double* xsaf;
    const double* xsrf;
    const double* xsnf;
    const double* xskf;
    const double* xssm;
    const double* chif;
};

struct DeviceCMFDWorkspace {
    double* dtil;
    double* dhat;
    double* diag;
    double* cc;
    double* src;
    double* psi;
};
```

그 다음 host class는 다음 역할만 한다.

```text
CPU Driver:
  JSON/HDF5 읽기
  Geometry/XSSet 초기화
  device buffer allocate/copy

  반복:
    GPU kernel로 flux solve 일부 수행
    필요한 feedback/search는 CPU에서 판단
    변경된 XS/TH/rod state만 GPU로 sync

  결과를 CPU로 복사
  HDF5 저장
```

### 15.5 빌드 시스템 변경

현재 `CMakeLists.txt`는 `LANGUAGES C CXX`만 사용한다. CUDA 포팅 시 다음이 필요하다.

```cmake
project(RASBERY LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89) # 실제 GPU에 맞게 조정

target_sources(RASBERY PRIVATE
    src/cuda/DeviceBuffers.cu
    src/cuda/CMFDKernels.cu
    src/cuda/BICGKernels.cu
)

target_link_libraries(RASBERY PRIVATE CUDA::cudart CUDA::cublas)

target_compile_options(RASBERY PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
```

주의:

- `-march=native`를 CUDA compiler에 넘기면 안 된다.
- OpenMP와 CUDA build option을 분리해야 한다.
- Debug/Release build directory와 VS Code launch path가 현재 일부 불일치하므로 정리하면 좋다.

### 15.6 2-group 하드코딩 문제

현재 코드는 대부분 2-group 기준으로 작성되어 있다.

예:

- `BICGSolver.cpp`의 `l * 4`, `l * 2`
- `BICGCMFD.cpp`의 `xsnf(0,l) * flux(0,l) + xsnf(1,l) * flux(1,l)`
- `Nodal.cpp` 내부의 `double a[2][2]`, `coeff[2][2]`

CUDA 포팅 시 선택지가 있다.

1. 지금처럼 2-group 전용 GPU kernel을 만든다.
   - 가장 빠르고 현실적.
   - 현재 코드와 맞음.
   - 나중에 multi-group 확장 때 재작업 필요.

2. `ng` 일반화 kernel을 만든다.
   - 더 범용적.
   - 작은 2x2 matrix 최적화가 어려워짐.
   - 코드 수정 범위가 커짐.

현재 목표가 "기존 RASBERY를 CUDA로 먼저 구동"이라면 1번이 현실적이다.

## 16. CUDA 포팅 단계 제안

### Phase 0. 기준선 고정

- CPU build가 되는 환경을 정리한다.
- 작은 test와 i-SMR CY01의 CPU HDF5 baseline을 만든다.
- 비교 스크립트를 현재 경로에 맞게 정리한다.
- profiling을 수행해 실제 병목을 확인한다.

### Phase 1. GPU device buffer와 view 추가

- `DeviceGeometryView`
- `DeviceXSView`
- `DeviceCMFDWorkspace`
- host/device copy helper
- CUDA on/off runtime flag

이 단계에서는 계산 결과가 바뀌면 안 된다.

### Phase 2. CMFD matvec와 vector operation GPU화

- `BICGSolver::axb` GPU kernel
- dot product는 cuBLAS 또는 custom reduction
- vector update kernel

초기에는 preconditioner는 CPU 유지 또는 no-preconditioner/block-Jacobi로 비교한다.

### Phase 3. CMFD coefficient kernel

- `upddtil`
- `upddhat`
- `updjnet`
- `setls`
- `updpsi`
- source build

이 단계에서 flux solve 대부분이 GPU에 올라간다.

### Phase 4. Nodal phase GPU화

- `updateConstant`
- `caltrlcff0`
- `caltrlcff12`
- `updateMatrix`
- `calculateEven`
- `calculateJnet`

phase 사이에 kernel synchronization을 둔다.

### Phase 5. XS reconstruction과 TH 일부 GPU화

- `XSSet::Reconstruct`
- `NormFactor`
- `UpdateBurnup`
- `UpdateTH` node power kernel

`UpdateFlatXS` 전체와 history correction은 일단 CPU 유지 가능.

### Phase 6. PPR GPU화

- pin power reconstruction
- normalization reduction
- FRP/FQP reduction

### Phase 7. Depletion과 rod cusping 고도화

- depletion CRAM batched kernel 검토
- rod cusping small dense solve batched화 검토
- 필요하지 않으면 CPU 유지

## 17. 중요한 위험 지점

### 17.1 SSOR preconditioner

현재 BiCGSTAB의 `minv()`는 SSOR forward/backward sweep이다. GPU에서는 이 순차 의존성이 가장 큰 장애물이다. 단순히 OpenMP loop를 CUDA로 바꾸는 방식으로는 안 된다.

### 17.2 Floating point 재현성

GPU reduction 순서가 바뀌면 다음이 달라질 수 있다.

- keff 마지막 자리
- residual
- search iteration 수
- boron ppm
- rod critical position
- TH iteration 수

따라서 "bitwise identical"보다 "물리적으로 허용 가능한 tolerance"를 정의해야 한다.

### 17.3 Host object와 device code 분리

다음은 device kernel에서 직접 쓰면 안 된다.

- `std::vector`
- `std::map`
- `std::string`
- `std::filesystem`
- `std::format`
- `HighFive::File`
- `nlohmann::json`
- CHIFFON model object
- exception throw

GPU에는 plain pointer와 scalar 값만 넘기는 구조가 안전하다.

### 17.4 메모리 layout 불일치

현재 flux는 `phif[l * ng + ig]`이고 XS는 `xs[ig * nxyz + l]`이다.

즉 flux와 XS의 group/node 순서가 다르다.

```text
flux: node-major
XS:   group-major
```

CPU에서는 괜찮지만 GPU memory coalescing에는 영향을 준다. CUDA 포팅 시 다음 중 하나를 선택해야 한다.

- 기존 layout 유지: 결과 비교가 쉬움.
- GPU 전용 transposed layout 추가: 성능은 좋지만 sync와 버그 위험 증가.

초기 포팅은 기존 layout 유지가 안전하다.

### 17.5 Rod cusping의 dynamic list

`ApplyRodCusping`은 부분 삽입 node를 찾고, 3-node stencil을 만들고, 작은 dense system을 동적으로 구성한다. CUDA 1차 대상이 아니다.

### 17.6 Depletion CRAM

node별 독립성이 있지만 작은 dense matrix operation이 많다. GPU로 옮길 수는 있으나 solver 정확성과 핵종 보존 검증이 까다롭다.

## 18. 다음 작업에서 바로 보면 좋은 파일

코드 이해 순서:

1. `src/main.cpp`
2. `src/Driver.h`
3. `src/IO.cpp`
4. `src/Geometry.h`, `src/Geometry.cpp`
5. `src/XSSet.h`, `src/XSSet.cpp`
6. `src/CMFD.h`, `src/CMFD.cpp`
7. `src/BICGCMFD.cpp`
8. `src/BICGSolver.cpp`
9. `src/Nodal.cpp`
10. `src/PPR.cpp`

CUDA 준비 순서:

1. `docs/cmfd_nodal_preconditioner_ko.md`
2. `docs/chiffon-rasbery-interface.md`
3. `docs/perf_and_cusping_fixes_ko.md`
4. `src/BICGSolver.cpp`
5. `src/BICGCMFD.cpp`
6. `src/CMFD.cpp`
7. `src/Nodal.cpp`
8. `src/XSSet.cpp`
9. `src/PPR.cpp`

Validation 준비 순서:

1. `test/Tests.md`
2. `.regress.sh`
3. `tools/compare_keff.py`
4. `tools/compare_micxs.py`
5. `tools/plot_ismr_validation.py`
6. `test/7_i-SMR_Validation`
7. `test/9-1_IISC/run_iisc.py`

## 19. 결론

RASBERY는 이미 OpenMP로 여러 hot loop가 병렬화되어 있어 CUDA 후보를 찾기 쉽다. 하지만 "OpenMP loop를 CUDA kernel로 그대로 변환"하는 방식은 충분하지 않다. 특히 BiCGSTAB의 SSOR preconditioner, depletion CRAM, rod cusping은 알고리즘 구조상 GPU 이식 설계가 필요하다.

가장 안전한 포팅 방향은 다음이다.

```text
CPU:
  input, HDF5, geometry setup, CHIFFON, scheduler, search control, output

GPU 1차:
  CMFD coefficient
  A*x
  vector reductions/updates
  Nodal phase kernels
  XS reconstruct
  PPR pin reconstruction

GPU 2차:
  TH 일부
  depletion batched CRAM
  rod cusping batched solve
```

이 순서로 진행하면 작은 smoke test에서 full-core validation까지 단계적으로 검증할 수 있고, 코드 전체를 한 번에 바꾸는 위험을 줄일 수 있다.
