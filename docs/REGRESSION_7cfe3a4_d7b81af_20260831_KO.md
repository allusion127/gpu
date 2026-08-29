# 회귀 조사: `7cfe3a4` → `d7b81af`에서 flag-off 궤적이 움직였다 (2026-08-31)

| 항목 | 값 |
| --- | --- |
| 범위 | `7cfe3a4..d7b81af` (24 커밋) |
| 측정 arm | **arm X**(단일 `kngr_238`, 238 서버, production 환경) |
| 관측 | `7cfe3a4` → digest `22b9a3187bfb4beb`, outers **4566**<br>`d7b81af` → digest `c1a5d9116df9edb3`, outers **4601**<br>`h5diff -c` **435/644** 상이 |
| 규칙 | **B0 — feature-off byte identity.** 새 플래그는 전부 unset이었으므로 이 범위의 어떤 커밋이 *기본* 궤적을 옮겼다 |
| 결론 | **원인 후보 1개, 커밋 `71092e2` (WP7-C).** 나머지 22개 코드 커밋은 flag-off 경로에서 무해함이 소스로 확인됨 |
| 수정 | 본 커밋. `src/XeAndersonReference.cpp`는 `7cfe3a4`와 **byte identical**로 복원, 채굴 채널을 분리 |
| 확인 | 238에서 아래 §4 runbook. **§3의 한 줄 진단이 bisect보다 먼저다** |

---

## 1. 왜 "테스트 코드"가 production 궤적을 옮길 수 있는가

`RASBERY_XE_FORMS`는 **baked 상수가 아니다.** `src/XeFormMiner.cpp:42`의
`xeFormMask()`가 프로세스 시작 시 이 호스트에서 **직접 채굴(mine)** 하고, 그 값이
커널 인자로 들어간다:

- `src/CudaXsReconBackend.cu:2806` — `kXeDotStage1(..., xe::xeFormMask())`
- `src/CudaXsReconBackend.cu:2835` — `kXeCandidate(..., xe::xeFormMask(), ...)`
- `src/CudaXsReconBackend.cu:2967` — `const unsigned long long forms = xe::xeFormMask();`

arm X는 `RASBERY_GPU_XE=1 XE_ANDERSON=1 XSRECON=1`이므로 **이 경로는 켜져 있다.**
따라서 채굴 결과가 1비트라도 움직이면 device Xe Anderson의 내적/후보 수축(contraction)이
바뀌고, 마지막 비트 수준의 차이가 outer 수를 흔든다. 플래그는 하나도 건드리지 않은 채로.

관측된 크기(+35 outer, 0.77 %)는 정확히 이 종류의 섭동이다. 제어흐름 변경이었다면
훨씬 크게 벌어졌을 것이다.

---

## 2. 커밋별 판정

### 2.1 원인 후보 — `71092e2` "feat(WP7-C)"

**두 개의 문(door)을 동시에 열었다. 둘 다 gate 없이 기본값을 움직인다.**

#### (a) soundness 판정이 채널 간 공유되었다

- `src/XeFormMine.h` (71092e2 시점) — `scoreMask()`에 WP7-C 정규방정식 루프가
  **같은 `bad` 총계로** 추가됨. `mineStable()`은 `scoreMask(f, m) != 0`이면
  `sound = false`로 만든다.
- `src/GpuFormMask.h` `resolveCalibratedFormMask()` — 우선순위는
  ① env override ② **sound한 채굴값** ③ **unsound면 `XE_FORMS_DEFAULT`(0xd)**.
- 즉 **비트 5..12(= `RASBERY_GPU_XE_TXN` 전용, 기본 OFF)** 를 이 호스트에서
  채굴하지 못하면 `sound=false`가 되어, **아무도 건드리지 않은 비트 0..4까지**
  0xd로 강등된다. 238의 채굴값이 0xd가 아니면(형제 CMFD mask가
  `0x6`(dev) vs `0x7`(238)로 갈리는 것이 `src/GpuFormMask.h` 머리주석에 기록되어 있다)
  그 순간 device Xe 산술이 바뀐다.

#### (b) 참조 translation unit을 편집했다 — 이쪽이 더 직접적이다

`src/XeAndersonReference.cpp`가 별도 오브젝트로 존재하는 이유는 파일 자신이 적어둔
그대로다(`src/XeFormMine.h` 머리주석):

> this file pulls in the SHIPPED bodies (XeKernel.h), and the reference translation
> unit must never see them or **gcc common-subexpressions across the two and changes
> what the reference measures**.

71092e2는 바로 그 TU에

- `#include <cmath>`
- `buildFixture()` **내부**에 `std::sqrt`를 쓰는 33줄짜리 Gram 케이스 루프
- 새 export 함수 `refAlgebra()` 46줄

을 넣었다. 이 셋은 `refDot` / `refCandidate` **바로 앞**, 그리고 그 둘의 피연산자를
만들어내는 함수 **안**에 놓였다. 그 두 인용(quotation)이 바로 **비트 0..4가 측정되는
대상**이다. 분리 논거가 한 단계 아래에도 그대로 적용되는데 적용되지 않았다.

> 참고: 채굴은 shipped body의 host 컴파일 결과가 아니라 **참조 쪽 codegen**에
> 민감하다. shipped body는 `xsrMul`/`xsrFma`(`src/XsReconKernel.h:83`)가 gcc에서
> `asm volatile` 배리어를 쓰므로 재수축되지 않는다. 움직일 수 있는 건 참조 TU뿐이고,
> 71092e2가 편집한 것이 정확히 그 TU다.

#### (c) 부수 확인 — 나머지 71092e2 변경은 무해

- `src/XeKernel.h:157` `XE_FORMS_DEFAULT = 0xdull` **불변**.
- `src/Driver.h` `TryAndersonXeStepGpu()` 선두의
  `static const bool txn = rasberyGpuXeTxnEnabled(); if (txn && ...)` — 기본 OFF,
  round-tripping arm 본문 무변경.
- `src/CudaXsReconBackend.cu` — `xe_pairs`/`xe_bits` cudaMalloc 크기 증가, `xe_ctl`
  추가, `uploadXeTxnLayouts()`는 `rasberyGpuXeTxnEnabled()` 뒤. 값에 영향 없음.
- **다만 `tools/test_xe_anderson.py`는 이 커밋 이후 branch tip에서 이미 FAIL**
  (`xe_aa_proposed must be charged in exactly 2 place(s)` — TXN 경로가 3번째 charge
  site를 추가했다). 본 조사 범위 밖이지만 별도로 닫아야 한다.

### 2.2 무해 판정 — 나머지 코드 커밋

| 커밋 | 판정 | 근거 (file:line) |
| --- | --- | --- |
| `5883023` + `2a1161b` (WP9-A phase enum) | **무해** | 추가된 것은 `outer_timing::Scope` RAII와 `xsphase::Scope`의 세 번째 인자뿐(`src/Driver.h` PH_TH_UPDATE/PH_XE_STEP/PH_SEARCH_PROPOSE/PH_SEARCH_APPLY 스코프, `src/XSSet.cpp:3136` `xsphase::Scope(..., LB_FLATXS)`). `PH_COUNT`로 크기가 정해지는 배열은 `Counters::phase[]`/`phase0[]` — 순수 wall 누적. `2a1161b`가 고친 `PH_PPR_RESET = PH_UPDDHAT + 1` 중복 case는 **컴파일 실패**였으므로 그 사이 커밋들은 애초에 바이너리가 없다. `src/XSTiming.h` `Scope`는 armed가 아니면 clock을 읽지 않는다 |
| `3df4ea7` (WP9-D search) | **무해** | `src/Scheduler.h` `ProposeNextSearchPoint()`가 `SecantSearchParams params;`를 if 밖으로 올렸을 뿐 — BORON 분기는 RODCRIT 전용 필드(`enforce_rod_clamp`/`rod_max_step`/`use_bracket`)를 이전에도 기본값으로 두었다. `TallyProposal()`은 `method` 문자열만 읽는 카운터. `search_first_x`/`search_last_dx`는 어디서도 읽히지 않음 |
| `f9999a3` (WP10.1 test) | **무해** | `src/Scheduler.h`에서 WP9-D 필드 8개를 `// OUTPUT parameters` 아래로 **이동**만. `Schedule`은 필드 단위로 `DeviceSearchState`에 복사되므로 레이아웃 의존 없음 |
| `031ccbe` (WP10.2 warm start) | **무해** | `src/Driver.h` warm-start 블록 전체가 `if (!_warm_start_from.empty())` 안. `_warm_start_from`/`_warm_state_out`는 `setWarmStart()`로만 채워지고 CLI 기본은 빈 문자열. cold 경로의 `ResetFluxAndCurrents(1.0)`는 그대로. `[RASBERY][WARMSTART]` receipt도 `warm_status != "off" \|\| warm_save_status != "off"` 뒤 |
| `95481bc` / `f857274` (WP10.1 case key) | **무해** | `src/CaseKey.h:257` `deckPayload(const Json& config, ...)` — **const 참조**, `Json rest = config;`로 복사, 파싱된 deck을 변형하지 않는다. `src/IO.cpp:622`에서 파스 직후 호출. XS 라이브러리 digest는 별도 `std::ifstream` 바이트 스트림(`src/XSSet.cpp` `XsLibraryDigestOf`)이라 HDF5 핸들과 무관 |
| `f32e646` (WP8.2 cohort/PPR quadrature) | **무해** | `src/PPR.cpp` `buildPinQuadratureTable(ndiv, npins)`는 원 멤버 함수를 줄 단위로 옮긴 것이고 값은 `(ndivxy, npins)`만의 함수. 공유는 `shared_ptr<const>`. XS 라이브러리 캐시 키에 content digest가 추가됐지만 **캐시 키일 뿐**이며 파싱 결과는 동일 |
| `8437697` (WP8.1.5) | **무해** | `src/main.cpp:737` `batch_execution = (batch_width > 0 && !rasbery_inputs.empty()) \|\| evaluator_mode`. arm X는 `evaluator_mode=false` → `ExecutionMode::Single` 그대로. Anderson 기본값(mode 의존)은 arm X가 `RASBERY_XE_ANDERSON=1`로 명시하므로 `XeAndersonSource::Env` |
| `2c04a6e` (WP5-B/C FlatXS CTA) | **무해** | `src/CudaXsReconBackend.cu:3335` `static const bool cta = rasberyGpuFlatXsCtaEnabled();` — absent means off, else 분기가 원래의 `kernelFlatXs<<<grid,128>>>` 그대로 |
| `71092e2` 외 `f6f0b63` (WP7-B FUSE) | **무해** | `src/CudaBICGBackend.cu:261` `RASBERY_GPU_CMFD_FUSE` 기본 0 → `fuse_dot/dot2/wiel/sweep_pre` 전부 false, reference 커널 쌍이 그대로 발사. `fuse_retire`는 mask와 무관하게 할당되어 arena shape가 mask에 의존하지 않게 한 것 |
| `9dff6ff` / `666e123` / `1cdd724` (WP1 guards, receipts, evaluator) | **무해** | `RASBERY_GPU_FULL_GUARD`는 `src/GpuFullContract.h:125` `RASBERY_GPU_FULL`/`_STRICT` unset이면 tally만. `BICGSolver.cpp`/`CudaBICGBackend.cu`의 세 fallback 필드는 출력 전용 |
| `c502856` (Task 10 PPR GPU) | **무해** | `src/PPR.cpp:71` `if (_gpu == nullptr \|\| !_gpu->available()) return false;` — arm off면 `_reigv/_jnet/...` 대입 이전에 반환 |
| `8bd5112` (Task 16 CRAM) | **무해** | `XSSet::PredictorStep`/`CorrectorStep`의 host OMP 블록이 `if (!*_on_device)` 안으로 들어갔을 뿐, arm off면 host 본문 그대로 |
| `d7b81af` (arena `Comps` 제거) | **무해** | `Geometry::_comps`는 `new int[_nxyz]`로 할당만 되고 아무도 쓰지도 읽지도 않았다(`src/Geometry.h:170` 주석). arena region 제거로 뒤쪽 region offset이 밀리지만 모든 region은 import로 채워진다 |

### 2.3 범위 밖 확인 — `63f4cfb` (WP10.3, `d7b81af` **이후**)

같은 부류의 결함이 있는지 지시대로 확인했다. **없다.**

- `src/CaseFidelity.h` `processCaseFidelity()`가 읽는 것은
  `detail::stagedMultiplier("RASBERY_STAGED_FLUX_TOL")` / `_XE_TOL` —
  `src/RunContract.h:122`의 `atof` + `>= 1.0` clamp로, `SolveLoop`의 옛
  `static const` 람다와 **동일한 읽기**.
- `parseStagedLooseSettle()`는 옛 람다의 truthiness 테스트를 그대로 옮긴 것
  (`nullptr` → false, `""/0/off/OFF/false/FALSE` → false).
- "`strict`가 staged를 지운다" 규칙은 `resolveCaseFidelity()`에서
  `request.fidelity == "strict"`일 때만 발화한다. 아무것도 선언하지 않은
  단발 CLI 실행은 `FidelityRequest::empty()`이므로 `out = base`에서 끝난다.
- `src/Driver.h:4497` `ctx.fidelity = _fidelity;`로 Driver 값이 SolveLoop에 전달된다.
- case key 쪽 `armEnvValue()`는 "프로세스 기본과 같으면 raw env 문자열 그대로"라
  기존 키가 보존된다.

---

## 3. 238에서 **먼저** 할 한 줄 진단 (bisect보다 싸다)

채굴 mask는 런타임에 **stderr로 스스로 보고한다**(`src/GpuFormMask.h`
`resolveCalibratedFormMask`). 두 바이너리의 로그에서 이 한 줄을 비교하면
§2.1이 원인인지 아닌지가 즉시 결정된다.

```bash
# 각 바이너리의 arm X 로그에서
grep -h 'RASBERY\]\[FORMS' run_7cfe3a4.log run_d7b81af.log
# 예: [RASBERY][FORMS] {"mask":"XE_FORMS","value":"0x7","source":"mined","build_default":"0xd"}
```

판정:

| `value` / `source` 비교 | 결론 |
| --- | --- |
| 두 로그의 `value`가 **다르다** | **§2.1이 원인 확정.** 본 커밋의 수정으로 닫힌다 |
| `d7b81af` 쪽만 `source":"build_default"` | **§2.1(a) 경로 확정** — soundness 공유로 강등된 것 |
| `value`·`source` 모두 **같다** | §2.1은 원인이 아니다 → §4 bisect로 넘어간다. 그 경우 남는 가설은 헤더 추가로 인한 host 인라이닝/수축 변화이며, bisect가 커밋을 특정해 준다 |

`RASBERY_XE_FORMS=<value>`로 두 바이너리를 같은 mask에 고정해 재실행하면
"mask가 원인"이라는 명제를 A/B로 직접 확인할 수 있다.

---

## 4. 238 bisect runbook

### 4.1 arm X 환경 (이 값에서 한 글자도 바꾸지 않는다)

```
RASBERY_GPU=1 CMFD_SWEEP=1 CMFD_RESIDENT_SINGLE=1 NODAL=1 NODAL_FULL=1
XSRECON=1 FLATXS=1 WIEL_FOLD=chunked GPU_XE=1 XE_ANDERSON=1
STAGED_FLUX_TOL=50 STAGED_XE_TOL=1000 STAGED_LOOSE_SETTLE=1
GPU_OUTER=1 GPU_OUTER_SEGMENT_MAX=8
```

**이 범위에서 새로 생긴 플래그는 전부 unset으로 둔다**:
`RASBERY_GPU_XE_TXN`, `RASBERY_GPU_CMFD_FUSE`, `RASBERY_GPU_FLATXS_CTA`,
`RASBERY_GPU_FLATXS_CTA_THREADS`, `RASBERY_GPU_PPR`, `RASBERY_GPU_CRAM`,
`RASBERY_XSLIB_DIGEST`, `RASBERY_STATEPOINT_TELEMETRY`, `RASBERY_XE_FORMS`.
`RASBERY_GPU_FULL` / `RASBERY_GPU_STRICT`도 unset(가드는 tally만 하게).

> CRLF 함정: env 파일이 CRLF면 `RASBERY_XE_ANDERSON=1\r`이 되어 값이 인식되지 않고
> mode 기본으로 떨어진다(`src/Driver.h` `xeAndersonGate()`가 trim은 하지만,
> 다른 플래그는 하지 않는 것도 있다). env 파일은 LF로 유지한다.

### 4.2 빌드 포인트 (4개)

| # | 커밋 | 이 지점이 답하는 질문 |
| --- | --- | --- |
| B0 | `7cfe3a4` | 기준선 재현 (digest `22b9a3187bfb4beb`, outers `4566`) |
| B1 | `3df4ea7` | WP9-A phase enum + WP7-B FUSE + **WP7-C** + WP10.1/10.2 + WP9-D까지가 움직였는가 |
| B2 | `47161ed` | WP7-C **이전** 지점. B1이 움직이고 B2가 안 움직이면 원인은 `5883023`·`71092e2`·`95481bc`·`031ccbe`·`3df4ea7` 중 하나 |
| B3 | `2c04a6e` | FlatXS CTA dispatch site가 reference 경로를 건드렸는가 |
| B4 | `d7b81af` | 관측된 회귀 재현 (digest `c1a5d9116df9edb3`, outers `4601`) |

> 순서 주의: 히스토리 순서는 `7cfe3a4` → `47161ed` → `71092e2` → `3df4ea7` →
> `2c04a6e` → `d7b81af`. 즉 **B2(`47161ed`)가 B1(`3df4ea7`)보다 먼저**다.
> 실행은 B0 → B2 → B1 → B3 → B4 순으로 하는 것이 이분 폭을 가장 빨리 줄인다.

`5883023`~`2a1161b` 사이는 **컴파일되지 않는다**(`PH_PPR_RESET` 중복 case,
`2a1161b` 커밋 메시지 참조). 그 구간을 빌드 포인트로 잡지 말 것.

### 4.3 각 지점에서 하는 일 (지점당 1회, 동일 호스트·동일 GPU)

```bash
git checkout <COMMIT>
# 빌드는 캠페인 표준 절차 그대로 (WSL micromamba CUDA 12.6, 238 = sm120, GPU0만)
# 실행: 단일 kngr_238, arm X 환경, 결과 HDF5는 지점별 파일명으로

# 1) 궤적 receipt
grep -h 'TRAJECTORY' run_<COMMIT>.log
#    -> {"digest":"...","outers":...,"statepoints":...}

# 2) 채굴 mask receipt (§3)
grep -h 'RASBERY\]\[FORMS' run_<COMMIT>.log

# 3) 결과 비교
h5diff -c results_7cfe3a4.h5 results_<COMMIT>.h5 | tail -3
```

기록할 것: `digest`, `outers`, `statepoints`, `XE_FORMS` 의 `value`/`source`,
`h5diff` 상이 개수.

### 4.4 판정 표

| 지점 | digest == `22b9a3187bfb4beb` | 결론 |
| --- | --- | --- |
| B0 | 예 (필수) | 기준선 재현됨. 아니면 **호스트/빌드 환경이 먼저 바뀐 것**이고 커밋 조사는 무효 |
| B2 `47161ed` | 예 | 원인은 `5883023`..`d7b81af` 구간 |
| B2 | 아니오 | 원인은 `7cfe3a4`..`47161ed` 구간 (Task 10 PPR / Task 16 CRAM / WP1 guards / WP8-1) |
| B1 `3df4ea7` | 아니오 & B2 예 | 원인은 `5883023`·**`71092e2`**·`95481bc`·`031ccbe`·`3df4ea7` 중 하나 → §3의 FORMS 비교로 `71092e2` 확정/배제 |
| B3 `2c04a6e` | B1과 동일 | FlatXS CTA는 무관 (소스 판정과 일치) |
| B4 `d7b81af` | 아니오 (`c1a5d9116df9edb3`) | 관측 재현 |

### 4.5 수정 검증 (본 커밋)

```bash
git checkout <이 커밋>
# arm X 1회
grep -h 'TRAJECTORY' run_fix.log     # -> digest 22b9a3187bfb4beb, outers 4566 이어야 한다
grep -h 'RASBERY\]\[FORMS' run_fix.log   # -> 7cfe3a4의 value/source와 동일해야 한다
h5diff -c results_7cfe3a4.h5 results_fix.h5   # -> 0/644
```

세 줄이 모두 맞으면 B0 규칙이 복구된 것이다. 하나라도 어긋나면 §3에서 배제되지
않은 두 번째 가설(헤더 추가로 인한 host codegen 변화)이 남아 있는 것이고,
그때는 §4.4의 B2/B1 이분을 끝까지 진행한다.

---

## 5. 이 커밋이 바꾼 것

| 파일 | 변경 |
| --- | --- |
| `src/XeAndersonReference.cpp` | **`7cfe3a4`와 byte identical로 복원.** `<cmath>`, `buildFixture` 내 Gram 케이스 루프, `refAlgebra` 제거 |
| `src/XeAlgebraReference.cpp` | 신규 TU. `buildAlgebraFixture()`(자체 seed) + `refAlgebra()` 인용 |
| `src/XeAndersonReference.h` | `buildAlgebraFixture()` 선언과 분리 논거 |
| `src/XeFormMine.h` | `scoreMask` → `scoreShippedMask`(비트 0..4) + `scoreAlgebraMask`(비트 5..12)로 분리. 채널별 site 목록·pass budget·`descend()`. `mineStable(f, sound, algebra_sound)` |
| `src/XeFormMask.h` / `src/XeFormMiner.cpp` | `mineXeFormsOnThisHost(sound, algebra_sound)`. `resolveCalibratedFormMask`에 **shipped soundness만** 전달. algebra 채널이 unsound면 경고 1줄 |
| `test/xe_form_probe.cpp` | `buildMiningFixture()` 사용, 두 채널을 각각 check |
| `CMakeLists.txt` | `XeAndersonReference.cpp`를 명시적으로 나열하는 두 타깃에 `XeAlgebraReference.cpp` 추가 (main 타깃은 `src/*.cpp` GLOB) |
| `tools/test_xe_txn_contract.py` | WP7-C 인용의 위치를 새 TU로 갱신 |
| `tools/test_xe_forms_default_contract.py` | **신규.** 30 검사 / 음성대조 11 |

### 5.1 새 계약 테스트가 잠그는 규칙

1. `XE_FORMS_DEFAULT == 0xd`이고 비트 5 이상은 0이다.
2. shipped site는 비트 0/2/3/4에 그대로 있다.
3. `src/XeAndersonReference.cpp`에는 `refAlgebra`/`buildAlgebraFixture`/`alg_cases`/
   `f.alg`/`std::sqrt`/`<cmath>` 중 **어느 것도 없다** (주석 제외 코드 기준).
4. `scoreShippedMask`와 `scoreAlgebraMask`는 서로의 심볼을 참조하지 않는다.
5. `mineStable`에서 `sound`는 `scoreShippedMask`로만, `algebra_sound`는
   `scoreAlgebraMask`로만 쓰인다.
6. `resolveCalibratedFormMask`의 4번째 인자는 정확히 `sound`(병합 플래그 금지).
7. shipped 채널의 site 목록·방문 순서·pass budget(6)은 WP7-C 이전 그대로.
8. `XeAndersonReference.cpp`를 링크하는 CMake 타깃은 `XeAlgebraReference.cpp`도 링크한다.

```bash
python tools/test_xe_forms_default_contract.py
python tools/test_xe_txn_contract.py
python tools/test_enum_alias_contract.py
python tools/test_dependent_template_contract.py
```

### 5.2 아직 열려 있는 것

- **`tools/test_xe_anderson.py`가 branch tip에서 이미 FAIL**한다
  (`xe_aa_proposed must be charged in exactly 2 place(s)`; `71092e2`가
  `TryAndersonXeStepGpuTxn`에 3번째 charge site를 추가했다). 본 회귀와는 별개이며
  이 커밋에서 건드리지 않았다.
- §3의 FORMS 비교가 "두 로그가 같다"로 나오면 원인은 다른 곳이다. 남는 가설은
  `Driver.h`가 이 범위에서 얻은 5개 include(`CaseKey.h`·`GpuFullContract.h`·
  `WarmState.h`·`XSTiming.h`·`XsLibrary.h`)와 `XSSet.cpp`의 `Sha256.h`가 host
  인라이닝을 바꿔 수축이 달라지는 경우다. 이는 소스로 판정할 수 없고 §4의 이분으로만
  좁혀진다.
