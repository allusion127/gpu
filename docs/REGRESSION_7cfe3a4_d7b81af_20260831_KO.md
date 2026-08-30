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

---

## 6. 부록 (2026-08-30 추가): §3의 판정표는 틀렸다 — mask는 원인이 아니다

### 6.1 238이 실제로 준 세 줄

같은 호스트, 같은 arm-X 환경, `RASBERY_GPU_XE=1`, `XE_TXN` unset/0:

| 커밋 | `[RASBERY][FORMS]` XE_FORMS | arm-X digest / outers |
| --- | --- | --- |
| `7cfe3a4` | `{"value":"0xd","source":"mined_matches_default","mined":"0xd"}` | `22b9a3187bfb4beb` / 4566 |
| `d7b81af` | `{"value":"0xd2d","source":"mined","mined":"0xd2d"}` | `c1a5d9116df9edb3` / 4601 |
| `8919331` | `{"value":"0xd2d","source":"mined","mined":"0xd2d"}` | `c1a5d9116df9edb3` / 4601 (h5diff 435, 불변) |

`src/XeAndersonReference.cpp`는 `8919331`에서 `7cfe3a4`와 **byte identical**임이
확인되었다. 즉 §2.1(b)(참조 TU 편집) 가설은 이미 되돌려졌고, 되돌린 뒤에도 궤적은
움직인 채다.

### 6.2 `0xd2d`를 분해하면 §3의 판정 규칙이 무너진다

```
0xd2d = 0xd | (1<<5) | (1<<8) | (1<<11)
        ^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
     shipped        algebra (WP7-C: DET=1, G0=0, G1=1, PROJ=1)
```

**shipped 하위 마스크(비트 0..4)는 두 커밋에서 동일한 `0xd`다.** 움직인 것은
`RASBERY_GPU_XE_TXN` 전용 채널뿐이다. §3의 표는 `value`가 다르면 §2.1이 원인이라고
적었지만, `value`는 **두 채널의 합집합**이었고 합집합이 움직였다는 사실만으로는
production arm이 움직였는지 알 수 없다. 그 표의 첫 행은 폐기한다.

### 6.3 비트 5 이상을 소비하는 live-arm 코드는 없다 (소스 판정)

지시받은 대로 `xeFormMask()`의 모든 소비자를 훑었다. 결과는 **음성**이다.

| 소비자 | file:line | 읽는 것 |
| --- | --- | --- |
| `xeDotChunk` (`kXeDotStage1`) | `src/XeKernel.h:432,433` | `(forms >> XE_DOT_FIRST_BIT) & 3ull`, `(forms >> XE_DOT_THIRD_BIT) & 1ull` |
| `xeCandidateOrdinal` (`kXeCandidate`) | `src/XeKernel.h:336,337` | `(forms >> cand_bit) & 1ull`, `cand_bit ∈ {3,4}` |
| `xeAndersonFit` (`kXeAndersonSolve`, TXN 전용) | `src/XeKernel.h:614,618,620,623` | `xeSiteState(forms, XE_TXN_*_BIT)` |
| `auditAndersonFit` | `src/XeFormAudit.cpp:50` | full mask, `RASBERY_XE_FORMS_AUDIT` 뒤 |

`kXeCommit`/`kXeHistory`/`kXeEvaluate`는 mask를 **인자로도 받지 않는다**.
`forms != XE_FORMS_DEFAULT` / `popcount` / `switch(forms)` 형태의 전체-마스크 술어는
트리 어디에도 없다. `resolveCalibratedFormMask`는 mask 폭을 검사하지 않으므로
`XE_BIT_COUNT`가 5에서 13으로 커진 것도 값에 영향이 없다. 그리고 case key는
`trajectory::kArmEnv`를 **raw getenv**로 읽으므로(`src/Driver.h:611`) unset인
`RASBERY_XE_FORMS`는 두 커밋에서 똑같이 `null`이다 — 채굴값은 키에 들어가지 않는다.

**결론: `71092e2`는 §2.1(a)로도 (b)로도 arm-X 궤적을 움직이지 않았다.**
`7cfe3a4..d7b81af`의 회귀는 아직 **미귀속**이며, 남은 유일한 가설은 §5.2의 두 번째
항목(헤더 추가로 인한 host 인라이닝/수축 변화)이다. §4의 B0 → B2(`47161ed`) →
B1(`3df4ea7`) 이분을 끝까지 진행해야 한다. **B2가 이미 움직여 있으면 WP7-C는 완전히
배제되고 범위는 `7cfe3a4..47161ed`(Task 10 PPR / Task 16 CRAM / WP1 guards / WP8-1)로
바뀐다.**

### 6.4 이 커밋이 그럼에도 바꾼 것 — 인자에서 채널을 없앤다

소비자들이 각자 자기 필드만 좁혀 읽는다는 것은 **소스 사실이지 계약이 아니었다.**
`forms != XE_FORMS_DEFAULT` 한 줄, `__popcll(forms)` 한 줄, 또는 새 열거자가 shipped
비트 자리와 겹치는 순간, TXN 전용 채굴 결과가 플래그 하나 움직이지 않은 채 production
궤적을 바꾼다 — WP7-C가 soundness 채널로 이미 한 번 저지른 바로 그 모양이다.

| 파일 | 변경 |
| --- | --- |
| `src/XeKernel.h` | `XE_SHIPPED_FORMS`(비트 0..4) / `XE_ALGEBRA_FORMS`(비트 5..12). 리터럴이 아니라 `XE_TXN_DET_BIT`·`XE_BIT_COUNT`에서 유도 |
| `src/XeFormMask.h` / `src/XeFormMiner.cpp` | `xeShippedFormMask()` = `xeFormMask() & XE_SHIPPED_FORMS`. 재해결 없음, 두 번째 getenv 없음 |
| `src/CudaXsReconBackend.cu` | `xeDots`·`xeCandidate` 발사가 `xeShippedFormMask()`. `xeTransaction`만 full mask (호출 자체가 `rasberyGpuXeTxnEnabled()` 뒤) |
| `src/XeFormMiner.cpp` | `[RASBERY][FORMS] {"mask":"XE_FORMS","resolved":...,"shipped":...,"algebra":...,"live_arm":"shipped","txn_arm":"resolved","algebra_sound":n}` 한 줄 추가. 합집합 줄은 그대로 남는다 |
| `tools/test_xe_forms_shipped_split_contract.py` | **신규.** 33 검사 / 음성대조 18 |

### 6.5 채굴 `0xd2d` 자체는 건전한가 — 그렇다

`XE_FORMS_DEFAULT`의 비트 5 이상이 0인 것은 **측정이 아니라 측정의 부재**다
(`src/XeKernel.h` 주석이 그렇게 적어 두었다: 저작 호스트에 nvcc가 없어 WP7-C 사이트는
한 번도 측정된 적이 없다). 238은 그 네 사이트를 처음으로 실제 채굴했고, DET/G1/PROJ가
"첫 곱을 add에 fuse"(state 1), G0가 "fuse 없음"(state 0)으로 나온 것은 `-O3
-ffp-contract=fast`의 g++에서 전혀 이상하지 않다. `[WARN][FORMS]`도 뜨지 않았으므로
`algebra_sound=true`, 즉 네 사이트 모두 0 mismatch로 결정적이었다. **채굴은 정상이고
고칠 것이 없다.** 고칠 것은 그 값이 production 발사 인자에 실려 다니던 것이었다.

### 6.6 238 검증 (이 커밋)

```bash
# arm X 1회, XE_TXN unset
grep -h 'TRAJECTORY'          run_fix.log   # digest / outers
grep -h 'RASBERY\]\[FORMS'    run_fix.log   # 2줄: 합집합 + shipped/algebra 분해
h5diff -c results_7cfe3a4.h5 results_fix.h5 | tail -3
```

기대값과 판정:

| 관측 | 기대 | 어긋나면 |
| --- | --- | --- |
| `"shipped":"0xd"` | `7cfe3a4`의 `value`와 동일 | shipped 채널이 실제로 움직인 것 → §2.1이 되살아난다 |
| `"algebra":"0xd20"`, `"algebra_sound":1` | 238의 첫 실측 그대로 | 채굴 불안정, WP7-C 게이트 문제 |
| digest `22b9a3187bfb4beb` / outers `4566` / h5diff `0/644` | — | **예상대로 어긋난다.** §6.3이 맞다면 이 커밋은 궤적을 되돌리지 않는다. 그때는 §4의 B2/B1 이분이 유일한 다음 수이며, 이 커밋은 그 이분에서 mask 채널을 영구히 배제해 준 것이다 |

---

## 7. 원인 확정 (2026-08-31): `71092e2`가 옮긴 것은 **산술이 아니라 산술이 컴파일되는 함수**다

### 7.1 238이 새로 준 격리 관측

§6까지의 조사는 `7cfe3a4..d7b81af` 24커밋 구간에서 후보를 좁히지 못했다.
238이 `71092e2` **하나만** cherry-pick해 격리했다 (arm X, `RASBERY_GPU_XE_TXN` unset):

| 트리 | digest / outers | `[RASBERY][FORMS]` XE_FORMS |
| --- | --- | --- |
| `47161ed` | `22b9a3187bfb4beb` / **4566** | `0xd` |
| `47161ed` + `71092e2` (clean apply) | `c1a5d9116df9edb3` / **4601** | `0xd2d` |

`d7b81af`와 같은 값이다. **`71092e2` 단독이 원인이고, 나머지 23커밋은 무해**임이
소스 판정(§2.2)과 함께 실측으로도 확정되었다.

### 7.2 `71092e2`가 flag-off 경로에서 바꿀 수 있는 것 — 전수 판정

지시받은 6개 용의자를 순서대로 소스 판정했다. **여섯 모두 음성이다.**

| # | 용의자 | 판정 | 근거 (file:line) |
| --- | --- | --- | --- |
| 1 | `kXeHistory` (rotate+record+save 융합) | **음성 — split arm이 발사하지 않는다** | 유일한 발사점이 `src/CudaXsReconBackend.cu:3020` (`XsReconBackend::xeTransaction` 본문 내부). split arm은 `src/Driver.h:2758,2762,2766`에서 여전히 `XeGpuRotateHistory` / `XeGpuRecordColumn(aa.ncol)` / `XeGpuSaveEvaluation` 세 진입점을 순서대로 부른다. 회전 순서·기록 컬럼(`aa.ncol`, `ncol-1` 아님)·저장 시점(기록 뒤, 무조건)·`--aa.ncol`/`++aa.ncol` 모두 `47161ed`와 동일 — `tools/test_xe_split_arm_sequence_contract.py`가 이제 이 전부를 고정한다 |
| 2 | `xeCommit` / `drainXeCommit` | **음성 — 값 변경 없음** | `drainXeCommit`은 마지막 `return download(...)`가 `if (!download(...)) return false; countXeD2H(...); return true;`로 풀렸을 뿐(`src/CudaXsReconBackend.cu:2670`). `xeCommit`에 추가된 것은 `countXeD2H` 1회와 `xe_device_steps.fetch_add` 1회. "post-commit download failure retires the instance"는 `xeTransaction` 전용 문장이며 `xeCommit` 본문은 손대지 않았다 |
| 3 | `TryAndersonXeStepGpu`의 dispatch·수락/감쇠/reset-edge | **음성 — 본문 텍스트 무변경** | dispatch는 `src/Driver.h:2738-2740` 두 문장뿐이고 그 아래 `XSSet& xs = ctx.cross_sections;`부터 끝까지는 `47161ed`와 문자 단위로 동일(정규방정식 4식 `src/Driver.h:2808-2814`, 안전장치 4개, `RejectXeAnderson` 4곳 포함). 감쇠와 reset edge는 이 커밋이 건드리지 않았다 |
| 4 | `XSSet.cpp` Xe 진입점 | **음성 — 기존 진입점 서명·인자 무변경** | `XSSet::XeGpuTransaction`은 **새 함수**로 추가(`src/XSSet.cpp:4199`). `UpdateEquilibriumXenon` / `XeGpuEvaluate` / `XeGpuCommitPicard`의 서명도 본문도 diff에 없다. `PrepareXeDeviceCall`은 호출부가 5→6개로 늘었지만 부동소수 산술이 없는 포인터 배선 함수라 인라이닝이 값을 바꿀 수 없다 |
| 5 | `XeGpuReceipt.h` 카운터 | **음성 — 읽는 쪽이 없다** | 추가된 6필드(`xe_device_steps`/`txn_steps`/`txn_accepted`/`txn_declined`/`host_syncs`/`d2h_bytes`)는 `appendXeGpuReceiptFields`에서 출력될 뿐, 어떤 분기 조건에도 들어가지 않는다 |
| 6 | `xeDots`/`xeCandidate` 인자와 `XE_DOT_*` 상수 | **음성 — 상수 불변, 필드 폭 불변** | `XE_DOT_FIRST_BIT=0` / `XE_DOT_THIRD_BIT=2` / `XE_CAND1_BIT=3` / `XE_CAND2_BIT=4`는 그대로이고, 소비자는 각자 자기 필드만 좁혀 읽는다: `src/XeKernel.h:432,433`(`&3ull`, `&1ull`), `src/XeKernel.h:336`(`cand_bit = (ncol<=1)?XE_CAND1_BIT:XE_CAND2_BIT`, `&1ull`). §6.3의 결론 그대로이며 `c645124` 이후로는 발사 인자 자체가 `xeShippedFormMask()`(비트 0..4)다. 채굴값이 `0xd`→`0xd2d`로 움직였지만 **하위 5비트는 두 트리에서 같은 `0xd`** 이므로 production 산술은 같은 mask로 돌았다 |

### 7.3 그러면 남는 것은 하나뿐 — **main.cpp TU의 콜그래프 질량**

`src/Driver.h`를 include하는 `.cpp`는 `src/main.cpp` 하나뿐이고
(`src/EvaluatorServer.h`가 include하지만 그 헤더 역시 `main.cpp`가 먹는다),
`CMakeLists.txt`에 LTO는 없다. 즉 **솔버 전체가 하나의 TU**로 g++에 들어가고,
릴리스 플래그는 `-O3 -march=native`(`CMakeLists.txt:147`) — gcc의 C++ 기본
`-ffp-contract=fast`가 살아 있다.

그 TU 안에서 Xe 스텝은 **호출부가 정확히 하나씩인 static 멤버 함수의 사슬**이다:

```
SolveLoop            src/Driver.h:3965  -> TryAndersonXeStep
TryAndersonXeStep    src/Driver.h:2936  -> TryAndersonXeStepGpu
TryAndersonXeStepGpu src/Driver.h:2739  -> TryAndersonXeStepGpuTxn   <-- 71092e2가 추가
```

`-finline-functions-called-once`(-O1 이상 기본)는 호출부가 하나인 함수를
**크기와 무관하게** 호출자 안으로 접는다. 따라서 `71092e2`가 추가한
`TryAndersonXeStepGpuTxn` 약 110줄은 "옆에 놓인 죽은 코드"가 아니라
**`SolveLoop` 본문 안으로 들어간 110줄**이다. 그 결과 `SolveLoop`의 인라이닝·
스케줄링·수축 결정이 전부 새 몸통을 끼고 다시 계산된다.

그리고 그 `SolveLoop` 안에는 **배리어가 없는 host 산술**이 있다 —
split arm의 정규방정식 네 식(`src/Driver.h:2808-2814`):

```cpp
const double det = a * c - b * b;
gamma[0] = (c * p - b * q) / det;
gamma[1] = (a * q - b * p) / det;
proj     = gamma[0] * p + gamma[1] * q;
```

이들은 `xsrMul`/`xsrFma`(`src/XsReconKernel.h:83`의 `asm volatile` 배리어)를
쓰지 않는다. 어느 곱을 add에 fuse할지는 gcc가 그때그때 정하고, 그 선택은
**이 식들이 어느 함수 몸통 안에서 컴파일되느냐**에 달려 있다.
`71092e2` 자신의 커밋 메시지가 그 사실을 적어 두었다("g++ at -O3 may fuse
either"), 그리고 `src/Driver.h:2840-2852`의 audit 훅 주석은 이 호스트에서
그 gap이 실제로 벌어진 사례(호스트 181)를 이름으로 적어 두었다.

**+35 outer(0.77 %), h5diff 435/644 — 마지막 비트 섭동의 크기다.**
제어흐름이 바뀌었다면 이것보다 훨씬 크게 벌어졌을 것이다(§1).

### 7.4 대조군이 있다 — 같은 모양인데 움직이지 않은 훅

`8919331`은 **같은 함수의 한가운데에** 같은 모양의 훅을 넣었다
(`src/Driver.h:2853-2857`):

```cpp
static const bool audit = xe::xeFormAuditEnabled();
if (audit)
    xe::auditAndersonFit(dots, aa.ncol, XE_ANDERSON_MIN_GRAM, solved, ...);
```

그리고 §6.1의 실측대로 **궤적은 움직이지 않았다**(`d7b81af`와 동일한
`c1a5d9116df9edb3`/4601, h5diff 435 불변). 두 훅의 유일한 구조적 차이는:

| | 캐시된 bool | 호출되는 몸통이 있는 곳 | 궤적 |
| --- | --- | --- | --- |
| audit 훅 (`8919331`) | 예 | `src/XeFormAudit.cpp` — **다른 TU**, 불투명 호출 | **불변** |
| TXN 훅 (`71092e2`) | 예 | `src/Driver.h` — **같은 TU, 호출부 1개** → 인라인 | **이동** |

`src/XeFormAudit.h`의 머리주석이 "인라인이면 false negative"라고 이미
적어 둔 규칙이며, `71092e2`는 그 규칙을 한 단계 아래에서 어겼다.
**"ONE LINE, AT THE TOP"은 콜리가 불투명할 때만 참이다.**

### 7.5 hunk-bisect와의 대응 (238 러너와 반드시 일치해야 하는 표)

| hunk 묶음 | 소스 판정 | 예상 bisect 결과 |
| --- | --- | --- |
| **mask 파일** (`XeFormMine.h`, `XeAndersonReference.*`, `test/xe_form_probe.cpp`) | 무해 — 채굴 하위 5비트 불변(§6.2), 참조 TU는 `32ac308`에서 이미 원복 | 4566 유지 |
| **kernel 파일** (`XeKernel.h`, `CudaXsReconBackend.cu/.h`, `XeGpuReceipt.h`) | 무해 — device TU는 `--fmad=false`(`CMakeLists.txt:30`)이고 shipped body는 `xsrMul`/`xsrFma`로 고정. `XeKernel.h`가 새로 넣은 inline 함수들은 **main.cpp TU에서 아무도 참조하지 않으므로** cgraph에서 제거되어 인라이닝 예산에 들어가지 않는다 (참조자는 `CudaXsReconBackend.cu`, `XeFormAudit.cpp`, `XeFormMine.h`뿐) | 4566 유지 |
| **host 파일** (`Driver.h`, `XSSet.cpp/.h`, `CudaXsReconBackendStub.cpp`) | **원인** — `Driver.h`의 `TryAndersonXeStepGpuTxn`(+본문)과 그 유일 호출부 | **4601로 이동** |

`Driver.h` 안에서 다시 3개 hunk로 쪼갠다면:

| Driver.h hunk | 판정 |
| --- | --- |
| `kArmEnv`에 `"RASBERY_GPU_XE_TXN"` 추가 (`src/Driver.h:571`) | 무해 — 소비자는 `armEnvJson()`(receipt)과 `caseKeyProvenance()`(case key) 둘뿐, 물리 경로 없음. unset이면 두 트리 모두 `null` |
| `TryAndersonXeStepGpuTxn` 본문 (+110줄) | **원인** |
| dispatch 2줄 (`src/Driver.h:2738-2740`) | 원인의 **전달자** — 이 참조가 없으면 위 본문은 cgraph에서 제거되어 무해해진다 |

### 7.6 수정 (본 커밋)

| 파일 | 변경 |
| --- | --- |
| `src/Driver.h:56-63` | `RASBERY_NEVER_INLINE` — gcc `__attribute__((noinline, cold))`, MSVC `__declspec(noinline)`. 헤더 안에서만 살 수 있는 default-off arm이 audit 훅과 같은 **불투명성**을 사는 수단이며, **ON arm의 성능 힌트가 아니라 OFF arm의 정확성 계약**이라고 주석이 못박는다 |
| `src/Driver.h:2610` | `static RASBERY_NEVER_INLINE bool TryAndersonXeStepGpuTxn(...)`. 본문·의미·TXN=1 동작은 한 글자도 바꾸지 않았다 |
| `src/Driver.h:2727-2740` | dispatch 주석에 "ONE LINE은 콜리가 NEVER_INLINE일 때만 참"이라는 이유와 이 회귀의 digest를 적었다 |
| `tools/test_xe_split_arm_sequence_contract.py` | **신규.** 55 검사 / 음성대조 13 |
| `tools/test_xe_anderson.py`, `tools/test_xe_txn_contract.py` | 앵커가 `static bool ...Txn(`에 붙어 있었으므로 반환형 기준으로 옮김. 검사 내용 무변경 |

`cold`까지 붙인 이유: dispatch 분기를 unlikely로 표시해 `SolveLoop`의 블록
배치를 분기가 **없던** 시절에 더 가깝게 되돌린다. TXN=1 arm은 그대로 동작하며
(호스트 쪽 일은 request 구성 + 호출 + 카운터뿐, 산술은 전부 device),
`-Os`로 컴파일되는 것이 그 arm의 벽시계에 의미 있는 영향을 주지 않는다.

#### 7.6.1 새 계약 테스트가 잠그는 규칙

1. `RASBERY_NEVER_INLINE`이 gcc에서 실제로 `__attribute__((noinline, ...))`으로 전개된다 (빈 매크로 = 이름 붙은 주석).
2. `TryAndersonXeStepGpuTxn` 정의가 그 속성을 달고 있고, 호출부는 정확히 하나다.
3. 중립 대조군이 진짜 중립이다 — `auditAndersonFit`은 `XeFormAudit.h`에서 **선언만** 되고 `XeFormAudit.cpp`에서 정의된다.
4. dispatch는 두 문장뿐이고 `XSSet& xs = ctx.cross_sections;` 앞에 다른 것이 없다.
5. split arm의 device 호출 순서: evaluate → rotate → record → save → dots → candidate → commit, 각 1회.
6. 윈도 부기: 회전은 `aa.ncol == XE_ANDERSON_DEPTH` 가드 뒤 `--aa.ncol`, 기록은 `XeGpuRecordColumn(aa.ncol)`(≠ `ncol-1`), `++aa.ncol`은 기록과 저장 사이, 저장은 `if (aa.have_prev)` **밖**, `aa.have_prev = true`는 저장 뒤, dots/candidate는 `aa.ncol`.
7. split arm은 `XeGpuTransaction`/`XeTxnControl`/`xeAndersonFit`/`xeHistoryOrdinal` 등 TXN 기계를 언급하지 않고, `kXeHistory`·`kXeAndersonSolve`·`kXeCandidateTxn`·`kXeAndersonGate`·`kXeCommitTxn`은 `XsReconBackend::xeTransaction` 안에서만 발사된다.
8. 정규방정식 네 식이 문자 그대로 남아 있다 (`std::fma`나 `xsrMul`로 다시 쓰면 다른 반올림을 고정해 기준선 자체가 움직인다).

```bash
python tools/test_xe_split_arm_sequence_contract.py
python tools/test_xe_anderson.py
python tools/test_xe_txn_contract.py
python tools/test_xe_forms_default_contract.py
python tools/test_xe_forms_shipped_split_contract.py
python tools/test_xe_forms_audit_contract.py
python tools/test_xe_gpu_contract.py
python tools/test_enum_alias_contract.py
python tools/test_dependent_template_contract.py
python tools/test_telemetry_neutrality.py
```

### 7.7 238 검증 라인

arm X 환경은 §4.1 그대로, **`RASBERY_GPU_XE_TXN`은 unset**.

```bash
# (1) 수정본 -- B0 복구 여부
git checkout <이 커밋>
# 빌드: 캠페인 표준 (WSL micromamba CUDA 12.6, 238 = sm120, GPU0만)
grep -h 'TRAJECTORY'  run_fix.log     # -> digest 22b9a3187bfb4beb, outers 4566
grep -h 'FORMS'       run_fix.log     # -> shipped 0xd (algebra 0xd20은 움직여도 무방)
h5diff -c results_47161ed.h5 results_fix.h5 | tail -3   # -> 0/644

# (2) A/B 확증 -- 속성만 제거한 대조 빌드
#     src/Driver.h:2610 의 RASBERY_NEVER_INLINE 한 토큰만 지우고 재빌드
grep -h 'TRAJECTORY'  run_inlined.log # -> c1a5d9116df9edb3 / 4601 이 나와야 한다

# (3) TXN=1 이 여전히 작동하는지 (N1이어도 무방, 죽지만 않으면 된다)
#     arm X + RASBERY_GPU_XE_TXN=1
grep -h 'XE_GPU'      run_txn.log
#     txn_steps > 0 이고 txn_declined == 0 이어야 census가 한 arm의 것이다
```

판정:

| 관측 | 기대 | 어긋나면 |
| --- | --- | --- |
| (1) digest `22b9a3187bfb4beb` / outers `4566` / h5diff `0/644` | B0 복구 | §7.3이 맞더라도 잔여분이 남은 것 — 다음 수는 `TryAndersonXeStepGpuTxn` 본문을 자체 TU로 내보내 `overall_size` 기여까지 없애는 것(현재 fix는 인라인만 막고 cgraph 노드 자체는 남긴다) |
| (2) `c1a5d9116df9edb3` / `4601` | 인라이닝이 원인임을 A/B로 확정 | (2)도 4566이면 원인은 이 훅이 아니라 `71092e2`의 다른 hunk이며, §7.5의 3분할 bisect로 되돌아간다 |
| (3) `txn_steps > 0`, `txn_declined == 0` | TXN arm 생존 | `noinline/cold`가 arm을 깨뜨린 것이 아니라 arm이 declined하는 것 — `xeTransaction`의 refusal 조건을 본다 |

### 7.8 §5.2와 §6.3에 대한 정정

- §5.2의 "남는 가설: 헤더 추가로 인한 host 인라이닝/수축 변화"는 **맞았다.**
  다만 그 가설은 "include가 늘어서"라고 적혀 있었고, 실제 기전은 include가
  아니라 **호출부가 하나인 새 함수가 `SolveLoop` 안으로 접힌 것**이다.
  `71092e2`가 `Driver.h`에 추가한 include는 없다(`XeKernel.h`·`XeGpuReceipt.h`
  둘 다 이미 있었다).
- §6.3의 "비트 5 이상을 소비하는 live-arm 코드는 없다"는 **유효하다.**
  이번 판정은 그것을 뒤집지 않고, 그 음성 판정이 남긴 공백을 메운다.
