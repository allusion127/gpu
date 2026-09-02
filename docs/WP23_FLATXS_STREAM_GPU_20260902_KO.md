# WP23 — 분기 스트림 해석기(BuildFlatXsStream)와 노달 상수의 디바이스 이식

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `XSSet::BuildFlatXsStream`(≈1.0 s/단일 KNGR 런) · `XsReconBackend::solveNodal`의 상수 9배열 업로드(9,675회 / 3.9 GB) |
| 플래그 | `RASBERY_GPU_FLATXS_STREAM=1`(기본 off, `RASBERY_GPU_FLATXS`의 하위 아름) · `RASBERY_FLATXS_STREAM_STRIDE` · `RASBERY_FLATXS_STREAM_FORMS`(**채굴됨**, WP23.1) · `RASBERY_GPU_FLATXS_STREAM_LIBM=exact\|fast`(기본 `fast`, WP23.1) · `RASBERY_GPU_NODAL_CONSTS=1`(기본 off) |
| 게이트 등급 | 스트림 = **N1**(WP23.1 이후 이유 **하나**, §4) · 노달 상수 = **N1(측정됨)**(§6) |
| 계약 테스트 | `tools/test_flatxs_stream_contract.py`(16 규칙) · `tools/test_flatxs_stream_forms_contract.py`(12 규칙, WP23.1) · `tools/test_nodal_consts_gpu_contract.py`(13 규칙), 각 규칙마다 negative control |
| 동반 테스트 | `test_flatxs_cta_contract` · `test_micx_resident_contract` · `test_micx_layout_contract` · `test_xfer_ledger_contract` · `test_th_gpu_contract` · `test_gpu_full_fail_closed` · `test_enum_alias_contract` · `test_dependent_template_contract` — 전부 PASS |
| 신규 소스 | `src/FlatXsStreamKernel.h` · `src/FlatXsStreamReceipt.h` · `src/NodalConstsReceipt.h` · **WP23.1**: `src/FlatXsStreamExactMath.h` · `src/FlatXsStreamReference.{h,cpp}` · `src/FlatXsStreamFormMine.h` · `src/FlatXsStreamFormMask.h` · `src/FlatXsStreamFormMiner.cpp` · `test/flatxs_stream_form_probe.cpp` |
| 수정 소스 | `src/XSSet.{h,cpp}` · `src/FlatXsKernel.h` · `src/CudaXsReconBackend.{h,cu}` · `CudaXsReconBackendStub.cpp` · `src/Nodal.cpp` · `src/Driver.h` · `src/GpuFullContract.h` · **WP23.1**: `CMakeLists.txt` |
| 기준 덱 | KNGR `kngr_238.json`, `nxyz = 8,451`, `NG = 2` |
| 기준 digest | `1f36e75dc00ed2b4` / `4377` outers (플래그 off) |

> **이 문서가 주장하지 않는 것부터.** (1) 두 아름 모두 **아직 실행되지 않았다.** 저작 호스트에는
> CUDA 툴킷도 HDF5도 없어서 `src/CudaXsReconBackend.cu`와 `src/XSSet.cpp`는 **컴파일조차
> 검증되지 않았다.** 공유 바디(`FlatXsStreamKernel.h`)만 g++ `-fsyntax-only -Wall -Wextra`로
> 통과했고, 나머지는 계약 테스트의 정적 검사와 정독뿐이다. §7의 238 런북이 첫 실측이다.
> (2) 노달 상수 아름은 **업로드만** 없앤다. 호스트 스윕(`Nodal::updateConstantsIfMoved`)은 그대로
> 돌고, 그것이 CPU 폴백을 옳게 유지하는 것이자 §6의 ULP 자기검사를 가능하게 하는 것이다.
> (3) 스트림 아름은 `tful`/`dmod`의 **소비자 하나**를 디바이스로 옮길 뿐, WP22 수신증이 남긴
> `tful/tmod/dmod` D2H를 없애지 않는다. 다른 호스트 소비자(FillRodNodeXS, 커스핑, 소진)가 남아 있다.

---

## 1. 해석기 해부 — 노드 하나가 하는 네 가지

`XSSet::BuildFlatXsStream`(src/XSSet.cpp)은 한 statepoint의 **비삽입(unrodded) 노드 목록**을
`kernelFlatXsCta`가 소비하는 평탄 `(did, x, scale)` 적용 스트림으로 바꾼다. 노드당 순서는 넷이다.

| # | 단계 | 입력 | 출력 항목 수(상한) |
|---|---|---|---|
| ① | 스칼라 분기 3축 | `boron_dmod·wvfr·bppm·BORON_DENSITY_FACTOR`, `sqrt(tful)`, `dmod` + `_node_delta_{lo,hi,frac}[branch][l]`, `hw` | 축당 최대 4 (자체 lo/hi + 트윈 lo/hi) → **12** |
| ② | 2원소 작업공간 프로브 | ①의 스트림 접두부 + `_ref_micx[XSAF]` | 0 (값만 생산) |
| ③ | 자기 라이브러리 스펙트럴 히스토리 | `_lib->lib_spectral_history[comp[l]]`, `_iden`, `_burn`, `Phif`, 기준 궤적 | 항당 최대 2 |
| ④ | 히스토리 트윈 라이브러리 | `lib_history_partner[comp[l]]`의 항들, 가중 `hw` | 항당 최대 2 |

**①의 순서는 우발이 아니다.** 분기별로 "wu 아름 → 히스토리 트윈 아름"이고, 이는
`UpdateUnroddedNodeXS`가 `applyDelta`를 부르는 바로 그 순서다. 스트림은 순서대로 적용되므로
(FlatXsKernel.h의 결정성 계약) 이 순서를 바꾸면 다른 double이 나온다.

**②가 왜 있는가.** `NodeSpectralIndex`는 `_micx`가 아니라 **base+branch 상태**를 읽어야 한다.
`_micx`는 히스토리 보정이 이미 적용된 이전 스텝의 상태이고, 그걸 읽으면 보정이 자기 좌표로
되먹임된다. 호스트는 `NISO*ng` 스크래치를 만들고 그중 **두 원소만**(XSAF Pu-239/B-10 열) 채운다.
디바이스에서는 그 둘이 레지스터에 있다.

### 1.1 노드 간 의존성 — 없다. 확인함.

`_flatxs_node_apps[i]`는 하나의 OpenMP 반복이 쓰고 다른 어느 반복도 읽지 않는다.
`micprobe` / `p_did` / `hist`는 `thread_local`이고 노드마다 `clear()` 된다.
**노드를 가로지르는 러닝 누산기는 없다.** `BuildFlatXsStream`에서 유일한 직렬 구간은
마지막 연접(concatenation)인데, 그것은 노드 순서대로의 **복사**이고 산술을 도입하지 않는다.
→ 순서 의존 작업공간은 **노드 내부**에만 있으므로 **B0는 원리상 도달 가능**하다.
그럼에도 이 아름이 B0가 아닌 이유는 §4다(도달 가능성과 측정은 다르다).

### 1.2 1.0 s는 어디서 나오는가 — 귀속하지 않는다

WP22가 잰 것은 `SPTELEM nested_wall "xsrecon/flatxs" 1.9 s` 중 커널이 ~0.1 s라는 것뿐이고,
그 안에서 ①~④의 비중은 **측정되지 않았다.** 후보는 넷이며 문서는 이를 가설로만 적는다.

1. ③/④의 항마다 `findLoBurn`/`findHiBurn` 이분 탐색 + `referenceDensity` 2회 조회·보간,
2. ②의 `flatxsProbeMicElement` 2회(각각 노드의 분기 접두부 전체를 Horner로 재적용),
3. `std::vector` push_back과 `thread_local` 스크래치의 재할당·순회,
4. 마지막 직렬 연접(전체 스트림 ≈ `nodes × apps × 20 B`를 단일 스레드가 복사).

**아름은 1~3을 옮기고 4를 없앤다**(고정 슬롯 패킹이라 연접 자체가 사라진다). 4가 지배적이었다면
절감은 상한에 가깝고, 3이 지배적이었다면 절감은 그보다 작다. §7의 A/B가 그 답이다.

---

## 2. 패킹 — 왜 조밀(dense)하지 않은가

호스트는 조밀 스트림 + 배타 스캔(`node_off[i]` = 누적 길이)을 만든다. 디바이스에서 조밀 스캔을
하려면 2패스 count/scan이나 atomic bump이 필요하고, **둘 다 한 노드의 항목이 놓이는 순서를
바꾼다.** 그것은 CTA 커널의 결정성 계약이 금지하는 유일한 것이다.

그래서 이 아름은 패킹하지 않는다. 노드 `i`는 `[i*stride, i*stride + cnt_i)`를 소유하고
`node_off[i] = i*stride`를 **커널이 직접 쓴다.** 소비자는 이미 `node_off`/`node_cnt`를 읽고
연속성을 가정하지 않으므로 하류는 그대로다. 비용은 세 배열의 여유 공간뿐이다
(`stride × n_nodes`; KNGR에서 `stride ≈ 12 + 4·max_terms`).

`stride`는 **정적 상한**으로 계산된다(`XSSet::FlatXsStreamEligible`):
`4 × 3 + 2 × (자기 모델 항수 + 트윈 모델 항수)`. 따라서 `kRefusalCapacity`는
"드문 일"이 아니라 **도달 불가능**이고, 커널의 용량 검사는 절대 켜지지 않아야 하는 점검이다.

---

## 3. 좌표 형태 표 — 구현 20 / 거부 0, 그러나 등급은 형태가 정한다

Chiffon `SpectralCoordinate`는 열거자 20개다(6/10/11은 실험 빌드가 쓴 값이라 결번).
**20개 전부 디바이스 바디에 있다.** 거부 사다리는 살아 있지만, 오늘 거부하는 것은
"구현되지 않은 형태"가 아니라 **알 수 없는 열거자**뿐이다.

| 값 | 이름 | 좌표 산술 | libm | 등급 기여 |
|---|---|---|---|---|
| 0 | `Density` | `max(0, N)` | — | B0 가능 |
| 1 | `LogDensity` | `log(max(N, fl))` | **log** | **N1** |
| 2 | `ThermalWeighted` | `φ_th/φ · max(0,N)` | — | B0 가능 |
| 3 | `FastWeighted` | `φ_f/φ · max(0,N)` | — | B0 가능 |
| 4 | `FluxRatioInteraction` | `log(φ_th/φ_f)·(N−N_ref)` | **log** | **N1** |
| 5 | `SqrtDensity` | `sqrt(max(0,N))` | sqrt(정확반올림) | B0 가능 |
| 7 | `SpectralIndex` | `NodeSpectralIndex` | **log** | **N1** |
| 8 | `SpectralIndexInteraction` | `NodeSpectralIndex·(N−N_ref)` | **log** | **N1** |
| 9 | `RelativeBurnRatio` | `log(Na/Ra) − log(Nb/Rb)` | **log** | **N1** |
| 12/13/14 | `Bppm/Tful/DmodInteraction` | `(x_b − x_ref)·(N − N_ref)` | — | B0 가능 |
| 15 | `LogDeviationSquared` | `log(N/N_ref)²` | **log** | **N1** |
| 16 | `FissileFraction` | `Na/(Na+Nb)` | — | B0 가능 |
| 17 | `InverseRatio` | `N_ref/N` | — | B0 가능 |
| 18 | `CubeRootRatio` | `cbrt(N/N_ref)` | **cbrt** | **N1** |
| 19 | `SaturatingRatio` | `N/(1+N/N_ref)` | — | B0 가능 |
| 20/21/22 | `Bppm/Tful/DmodRodAge` | `(x_b − x_ref)·Φ_rod·SCALE` | — | B0 가능 (§3.1) |

**히트 수는 이 문서가 적지 않는다.** 어떤 형태를 덱이 쓰는지는 CHIFFON 라이브러리 파일의
성질이고, 저작 호스트에는 그 라이브러리가 없다. 수신증의 `forms_hit{...}`가 **그 형태를
가진 모델에 속한 노드 수**를 형태 이름별로 찍는다. 기대: KNGR/m64는 `LogDensity`를 거의
확실히 쓰므로 `libm_form_hit:1`이 예상되며, 그렇다면 §4의 이유 (b)가 살아 있다.

### 3.1 rod-age 3형태가 B0 가능한 이유는 구조적이다

`FineRodThermalFluenceAverage(l, ctype)`는 `ctype <= 0`이면 **첫 줄에서 0.0을 돌려준다.**
`BuildFlatXsStream`은 **비삽입 노드 목록으로만** 불리므로 호스트 해석기의 `currentCtype`은
언제나 0이고, 따라서 이 세 형태의 좌표는 `(x − x_ref) · 0.0 · SCALE`이다. 디바이스 바디의
`fsRodThermalFluence()`가 그 사실을 **이름 붙여** 담고 있다 — 삽입 노드가 이 바디에 들어오는
날 고쳐야 할 자리가 한 곳이 되도록.

같은 논증이 더 큰 단순화를 준다: `ctypeIndex == ctypeIndex0`, `referenceBase == referenceBase0`,
`referenceBurnups == ctype-0 키 목록`. 그래서 디바이스는 **모델당 ctype-0 행만** 평탄화한다.
삽입 노드가 이 바디에 닿으면 `kRefusalRodded`이지 틀린 답이 아니다.

---

## 4. 정확도 등급 — 조각별, 그리고 왜 B0가 아닌가

| 조각 | 등급 | 이유 |
|---|---|---|
| 스칼라 분기 ① | B0 **가능** | `+ − × sqrt`뿐. 수축 사이트 없음 |
| 프로브 ② | **B0** | `flatxsProbeMicElement`를 **그대로** 부른다. 마스크는 이미 채굴된 `FLATXS_FORMS = 0x3FF` |
| 소진 브래킷 탐색 | **B0** | `std::lower_bound` 의미를 정수 연산으로 전사. 반올림 자유도 없음 |
| 기준 밀도/조건 보간 | **B0**(WP23.1) | `acc += f·(hi−lo)` 3사이트. **마스크 채굴됨** → 이유 (a) 소멸(§4.1) |
| libm 없는 13형태 좌표 | B0 **가능** | `+ − × ÷ sqrt`뿐 |
| libm 7형태 좌표 | **N1** | glibc `log`/`cbrt` ≠ CUDA `log`/`cbrt`, ~1 ulp → 이유 (b) |
| 고정 슬롯 패킹 | **B0** | 순열이 아니라 **주소**만 바꾼다. 노드 내부 순서 불변 |

**WP23은 이유를 둘로 적었다. WP23.1이 하나를 갚고 하나를 측정했다.**

- **(a) 수축 마스크 — 갚음.** §4.1. 세 사이트는 이제 **채굴된다**. 수신증의
  `forms_source` / `forms_sound`가 이번 런의 마스크가 **이 호스트의 측정**인지를 말하고,
  `forms_sound == 0`일 때만 이유 (a)가 살아 있다.
- **(b) 일곱 형태가 libm을 부른다 — 남음.** §4.2. `FlatXsKernel.h`가 애초에 좌표를 호스트에
  남긴 이유가 이것이고, WP23.1은 그 문장을 **뒤집는 것이 아니라 숫자로 한정한다.**
  수신증의 `libm_form_hit`가 이번 런에 이 이유가 살아 있는지를 말한다.

두 이유는 **고쳐야 할 것이 달랐다**. 하나의 등급으로 합쳤다면 (a)가 갚였다는 사실도,
(b)가 남았다는 사실도 보이지 않았을 것이다.

### 4.1 수축 마스크 — 채굴됨 (WP23.1)

구조는 WP22의 T/H 채굴기와 **같다**. 다른 것은 무엇을 인용하느냐뿐이다.

| 파일 | 역할 |
|---|---|
| `src/FlatXsStreamReference.{h,cpp}` | 세 람다의 **축어 인용** + 픽스처. **자기 TU**. `FlatXsStreamKernel.h`를 절대 include하지 않는다 |
| `src/FlatXsStreamFormMine.h` | 출하 바디를 인용에 대해 채점(비트 단위) + 좌표 하강 |
| `src/FlatXsStreamFormMask.h` / `FlatXsStreamFormMiner.cpp` | 프로덕션 해석. `rasbery::gpu::resolveCalibratedFormMask` 우선순위 |
| `test/flatxs_stream_form_probe.cpp` | 게이트. 건전성·사이트 결정성·해석기 일치·libm 잔차 |

**마스크 레이아웃 — 사이트당 2비트, `XE_SITE_*` 인코딩 그대로.**

| 필드 | 오프셋 | 호스트 철자(`XSSet.cpp`) | 채굴값(WSL2 g++ 13.3 `-O3 -march=native`) |
|---|---|---|---|
| `FS_REFDENS` | 0 | `value += fraction * (hi − lo)` | `FS_SITE_P1` |
| `FS_REFDENS0` | 2 | `v += f * (hi − v)` | `FS_SITE_P1` |
| `FS_REFCOND` | 4 | `value += fraction * (hi − value)` | `FS_SITE_P1` |
| — | — | `FS_ALL` = `0x3f` | **`FLATXS_STREAM_FORMS = 0x15`, sound = 1** |

상태는 셋이다: `FS_SITE_NONE`(곱을 반올림한 뒤 더함) / `FS_SITE_P1`(`fma(f, d, acc)`) /
`FS_SITE_P2`(`fma(d, f, acc)`). **`P2`는 이 세 사이트에서 증명 가능한 don't-care다** — 곱이
하나뿐이므로 대안 철자는 인자를 바꾼 fma이고, IEEE-754 fma는 앞 두 인자에 대해 정확히
교환법칙을 만족한다. 그래서 건전성(`sound`)의 정의는 **"모든 씨앗이 0에 도달"**이지
"모든 씨앗이 같은 패턴"이 아니다(`ThFormMine.h`의 `TH_HAVG`와 같은 상황·같은 판정).
2비트를 유지하는 이유는 형제 마스크와의 인코딩 통일, 그리고 곱이 둘인 사이트가 나중에
생겨도 옆 필드를 재번호 매기지 않기 위해서다.

**측정 (저작 호스트, `rasbery_flatxs_stream_form_probe`):**

```
FLATXS_STREAM_FORMS mined 0x15 (shipped record 0x0), sound=1
  sites: FS_REFDENS=P1 FS_REFDENS0=P1 FS_REFCOND=P1
  FS_REFDENS   P1 -> NONE : 994 mismatching words
  FS_REFDENS0  P1 -> NONE : 1174 mismatching words
  FS_REFCOND   P1 -> NONE : 70 mismatching words
  resolver: mask 0x15 source mined sound 1 libm fast
```

세 사이트 모두 **결정적**이다(틀리게 놓으면 반드시 깨진다). `FS_REFCOND`가 70으로 가장 얇은
것은 물리다 — 증분 `f·(hi−value)`가 누산기 `value`에 비해 작을수록 합의 반올림이 뒤집힐 확률이
낮다. 픽스처의 노드 수(2048)는 **이 가장 얇은 사이트**가 결정적이 될 때까지 키운 값이지
가장 두꺼운 사이트에 맞춘 값이 아니다.

**인라이닝 문맥이 인용의 일부다.** WP22가 비싸게 배운 것: gcc는 어느 곱을 add에 접을지를
**인라이닝 문맥마다 다시 정한다.** 그래서 인용은 표현식만이 아니라 호출 그래프를 복제한다 —
`refBuildStream`(`#pragma omp parallel for schedule(dynamic, 64)`, 프로덕션과 같은 절)
→ `refResolveNode`(노드 하나) → `refResolveSpectralHistoryDeltas`(세 람다를 **한 함수에**).
그래서 이 TU는 **반드시 `-fopenmp`로 컴파일되어야 한다**(pragma가 무시되면 본문이 outline되지
않고, 그것이 바로 다른 마스크를 고정하는 차이다). CMake가 그렇게 링크하고 계약 테스트가 고정한다.

`kStreamFormsDefault = 0`은 **남는다.** 이제 런이 의존하는 값이 아니라 **출하 기록**이고,
수신증이 그것과 채굴값을 비교해 "이 호스트는 다르다"를 한 줄로 말하게 하는 용도다
(`TH_FORMS_DEFAULT`와 같은 취급).

### 4.2 libm — 측정했고, 남는다 (WP23.1)

`src/FlatXsStreamExactMath.h`는 `log`/`cbrt`를 **libm 없이** 계산한다: `+ − × ÷ fma`만으로 된
double-double 평가라, **g++와 nvcc가 같은 비트를 낸다(구성상).** `RASBERY_GPU_FLATXS_STREAM_LIBM=exact`가
일곱 형태의 모든 `log`/`cbrt`를 이것으로 바꾼다.

**그런데 그것은 B0가 요구하는 것이 아니다.** 플래그 off 경로는 여전히 **glibc**를 부르므로,
필요한 등식은 `device == glibc`이지 `device == host_exact`가 아니다. 그래서 잰다 —
덱이 실제로 넘기는 인수 범위에서 10⁶ 샘플:

| 함수 | mismatch / 10⁶ | max ulp | 누가 틀렸나 | 판정 근거 |
|---|---|---|---|---|
| `log` | **242** | 1 | **glibc가 242, 우리는 0** | double-double 값(≈2⁻¹⁰⁰)에 더 가까운 쪽 |
| `cbrt` | **538,003** | 3 | **glibc가 538,003, 우리는 0** | 동일 |

읽는 법. 우리 쪽은 샘플 전부에서 **정확히 반올림**되었고(`ours_not_CR == 0`,
`undecidable == 0`), 차이는 전부 **glibc가 정확 반올림이 아니라서** 생긴다. glibc의 dbl-64
`log`는 최악 ~0.519 ulp라 거의 항상 맞고 가끔 틀리며, `cbrt`는 애초에 그 수준이 아니다
(x86_64 `libm-test-ulps`가 1 ulp로 기록).

**따라서 기본값은 `fast`다.** 규칙대로다: mismatch가 0이 아니므로 기본은 `fast`, 아름은
이유 (b)로 **N1을 유지**한다. 동시에 `exact`는 **호스트에 더 가까운** 아름이다(로그 인수의
99.976 %에서 비트 일치). 238이 digest를 뜨기 전까지 기본을 바꾸지 않는 이유는 정확도가 아니라
**A/B의 무결성**이다 — 기본을 바꾸면 아무 플래그도 안 옮긴 런의 궤적이 움직인다.
`RASBERY_GPU_FLATXS_STREAM_LIBM`이 `trajectory::kArmEnv`에 있는 이유이기도 하다.

**B0로 가는 유일한 길**은 §8-6에 적었다: 정확 반올림이 아니라 **glibc의 알고리즘 자체를
재현**하는 것. 이 WP는 그것을 하지 않았다(라이선스·출처 문제가 코드 결정이 아니라 정책 결정이다).

---

## 5. 거부 사다리와 정직성

| 코드 | 언제 | 어디서 결정 | 대응 |
|---|---|---|---|
| `kRefusalForm` | 라이브러리가 미구현 열거자를 담음 | **호스트, 런당 1회** (`FlatXsStreamEligible`) | 형태 **이름**과 함께 stderr 1줄, 아름 전체 off, 호스트 해석기 |
| `kRefusalModel` | 모델의 기준 궤적이 평탄화되지 않음 | 호스트, 런당 1회 | 동일 |
| `kRefusalCapacity` | 노드가 `stride`를 넘김 | 커널(도달 불가, §2) | 콜 전체 거부 → 호스트 해석 후 **1회 재시도** |
| `kRefusalRodded` | 삽입 노드가 목록에 있음 | 커널(구성상 불가) | 동일 |
| 프로브 | `CHIFFON_PROBE_BLEND=pu|both` | 호스트, 런당 1회 | 아름 off (RoddedPuFraction 미이식) |

**모든 이유가 호스트에서 사전에 결정 가능하다**는 것이 이 설계의 요점이다. 그래서 커널의
사다리는 "일상 경로"가 아니라 **절대 켜지지 않아야 하는 점검**이고, 켜지면 호스트 사전검사와
디바이스가 불일치한다는 뜻이므로 **두 답을 꿰매지 않고 그 콜의 아름을 통째로 포기한다.**

재시도가 CPU 재구성 루프 전체(8,451 노드)로 떨어지지 않고 **호스트 해석 + 재호출**인 이유:
스트림 페이즈가 거부해도 solve의 나머지는 손대지 않았기 때문이다.

---

## 6. 노달 상수 아름 (`RASBERY_GPU_NODAL_CONSTS=1`)

**무엇을 없애는가.** `solveNodal`의 상수 9배열 H2D — KNGR 런에서 9,675회 / 3.9 GB
(`RASBERY_GPU_FP32`에서 1.96 GB). 대신 `xsdf` 한 배열의 업로드(그림자 보호)와
`nodalConstantCoefficients`를 노드×군 레인으로 부르는 커널 하나가 들어간다.

**무엇을 없애지 않는가.** 호스트 스윕. `Nodal::drive`는 `updateConstantsIfMoved()`를 여전히
`solveNodal` **앞에서** 부른다. 이유는 둘이다: CPU 폴백과 하이브리드 드라이브가
`_eta1 … _diagDI`를 읽고, ULP 자기검사가 호스트 답을 비교 대상으로 필요로 한다.

**왜 `CudaNodalConstantKernel.h`의 런처가 아닌가.** `CudaOuterGraph.cu`(1713~)가 그 런처가
비활성인 이유를 두 가지로 적어 두었다: (1) 입력(`SlotRegion::Xs`/`ConstantXs`)이 프로덕션
경로에서 **한 번도 쓰이지 않는다**, (2) 출력 패킹이 `kNcDiagDI = 7 / kNcDiagD = 8`인데
이 백엔드의 **독자는 `diagD = 7 / diagDI = 8`**이다 — 묶으면 모든 노드·방향에서 `D`와 `1/D`가
뒤바뀌고, 둘 다 유한하고 그럴듯하므로 아무것도 터지지 않는다. 그래서 이 아름은 **중복되면
안 되는 단 하나 — 산술(`nodal::nodalConstantCoefficients`) — 를 재사용하고**, 이 백엔드의
패킹으로 쓴다. 계약 테스트가 저장 순서를 고정한다.

**얼리아웃을 두지 않은 이유.** 호스트는 `xsrf/xsdf`가 안 움직인 노드를 건너뛴다. 커널은
generation이 움직였으면 전 노드를 다시 계산한다. 안 움직인 노드는 입력이 같고 바디가
결정적이므로 **이미 있던 값을 다시 쓴다.** 호스트 캐시가 사는 것은 시간이고 디바이스는 그게
필요 없다. 반면 디바이스에 캐시를 두면 아무도 일관성을 지켜 주지 않는 두 번째 캐시가 생긴다.

**등급 = N1, 그리고 그것은 측정값이다.** `test/nodal_constant_exp_probe.cu`가
sm_61에서 400만 인자를 훑어 `sqrt`는 전부 일치, `exp`는 **3.34 %에서 정확히 1 ulp** 차이를
쟀다(물리적 kp2 대역 안에서는 5.34 %). 그러므로 궤적을 움직이는 전이이고 digest 동등성이
아니라 Gate A/B로 값을 매긴다.

**자기검사.** 빌드마다가 아니라 **첫 빌드와 매 256번째** 빌드에서, 9배열 각각의 **연속 구간
최대 4,096원소**를 되읽어 호스트 값과 ULP 거리를 잰다. 전체 배열을 내리면 이 아름이 없애려는
바로 그 바이트를 도로 움직이게 되므로 구간이다. 연속 구간을 쓸 수 있는 이유는 두 레이아웃
모두에서 역사상이 정확하기 때문이다(AoS는 `c_grp == 1`이라 `e`가 곧 호스트 첨자, SoA는
`c_node == 1`이라 `e < nxyz`가 `(lk=e, dir=0, ig=0)`). FP32 아름에서는 호스트 값을
**아름의 폭으로** 내려 비교한다 — 축소 오차는 WP20.1이 이미 값을 매긴 것이고 이 아름의
것으로 보고하면 안 되기 때문이다.

읽는 법: `max_ulp <= 1` & `over_1ulp == 0`이면 위 스파이크가 덱에서 재현된 것.
그보다 크면 **다른 현상**이다 — `--fmad=false`가 막지 못한 수축, 낡은 `xsdf`, 레이아웃 불일치 —
그리고 그것은 Gate B 핀출력 미스로 발견하는 것보다 수신증에서 읽는 편이 훨씬 낫다.
표본이 없는 런은 `0`이 아니라 **`-1`**을 찍는다("표본 없음"과 "0 ulp"는 다른 사실이다).

### 6.1 블록 49가 찍은 `max_ulp:9545195`는 산술이 아니라 **경합**이었다

블록 49v2의 수신증: `checked_elems:184320, max_ulp:9545195, max_ulp_array:7,
over_1ulp:107279`(58.2 %) — **그런데 Gate A·Gate B는 통과했다.** 950만 ulp는 "exp가 1 ulp
다르다"가 아니다. 셋 중 어느 것인지를 코드에서 가렸다.

**배열 7은 `diagD = 4·xsdf / (hmesh·hmesh)`다.** exp도 sqrt도 없고, form 마스크 사이트도
없고, `--fmad=false` 아래 IEEE 나눗셈 하나다. 호스트와 디바이스가 **1 ulp도** 다를 수 없다.
그러므로 그 숫자는 애초에 산술에 대한 것이 아니었다. (참고: `CudaNodalConstantKernel.h`의
**아레나** 커널 enum은 7 = `diagDI`, 8 = `diagD`로 **반대**다. 이 백엔드의 패킹은 7 = `diagD`이고,
수신증은 이제 배열 이름을 함께 찍는다.)

**진짜 원인**: 커널은 `cudaStreamNonBlocking`으로 만든 `stream`에서 돌고, 자기검사의
`xfer::memcpy`는 **레거시 기본 스트림**의 블로킹 복사다. 레거시 스트림은 non-blocking
스트림과 **암묵적 순서를 갖지 않는다.** 즉 되읽기는 `kNodalConstsDevice`가 아직 쓰지 않은
메모리를 표본했고, 비교 대상은 **직전 generation의 계수**(또는 OFF 경로가 올려 둔 값)였다.
Gate A/B가 통과한 것은 이 아름의 **진짜 소비자**가 같은 스트림 위의 커널들이라 처음부터
빌드 뒤에 올바르게 정렬돼 있었기 때문이다 — 순서를 어긴 것은 **진단뿐**이었다.

**수정 세 가지** (`src/CudaXsReconBackend.cu`, `src/NodalConstsReceipt.h`):

1. 되읽기 전에 **빌드 스트림을 배수**한다(`xfer::streamSync(…, "selfcheck drain", stream)`).
   표본은 1,081빌드 중 5회뿐이므로 이 동기화는 아름의 경로가 아니라 **진단의 경로**에 있다.
2. **지표를 배열별·상대오차로** 바꾼다. ULP 거리는 0 근처에서 무한대이고 부호가 바뀌면
   의미가 없는데 이 계수들은 실제로 0을 넘나든다. `rel = |a−b| / max(|a|,|b|,floor)`이고
   **floor는 그 배열 자신의 표본 최대 크기 × 1e-12** — 누가 고른 상수가 아니다. ULP도 함께
   남긴다(스파이크의 경계가 ulp로 서술돼 있으므로).
3. `max_ulp_array`를 **저장하지 않고 유도**한다. 기존 코드는 compare-exchange **뒤에**
   다시 읽은 `worst >= max_ulp`로 저장했는데, 이것은 뒤의 검사가 최대값을 **동점으로 맞추기만
   해도** 참이 된다. 이제 9칸 배열별 표가 기록이고 두 스칼라는 그 argmax이며, 수신증은
   `by_array`로 표 전체를 찍는다.

`tools/test_nodal_consts_gpu_contract.py`의 규칙 13·14가 각각 부정 대조군과 함께 이 두
성질(배수 순서, 배열별 상대 지표)을 잡아 둔다. 규칙 13의 대조군은 `stream` 대신 `nullptr`을
동기화하는 것 — 즉 **버그 그 자체**다.

**재확인**: 238에서 `RASBERY_GPU_NODAL_CONSTS=1`을 다시 돌려
`[RASBERY][NODAL_CONSTS]`의 `max_ulp`가 **1 이하**, `over_1ulp`가 0에 가깝고,
`by_array`의 `diagD`·`diagDI` 항목이 `ulp:0` / `rel:0`인지 본다. `diagD`가 여전히 0이
아니면 그때는 경합이 아니라 **진짜 불일치**이므로 커널을 본다.

---

## 7. 238 런북

전제: `E:`에 출력, 로컬 계산 금지, GPU0만. v6 기준선은 `RASBERY_GPU_FLATXS_STREAM` /
`RASBERY_GPU_NODAL_CONSTS`를 **둘 다 unset**한 것이며 digest `1f36e75dc00ed2b4` / `4377`.

### 7.1 블록 A — 스트림 아름 단독 (단일 덱)

```
# A0 기준선(플래그 off) — SPTELEM 켜고 3회, 중앙값
RASBERY_SPTELEM=1 ./RASBERY kngr_238.json --out E:/rasbery_runs/<date>/238/wp23/a0_%d.h5

# A1 스트림 아름
RASBERY_GPU_FLATXS_STREAM=1 RASBERY_SPTELEM=1 \
  ./RASBERY kngr_238.json --out E:/rasbery_runs/<date>/238/wp23/a1_%d.h5
```

읽을 것, 순서대로:

1. `[RASBERY][FLATXS][STREAM]`의 **`device_calls`** — 0이면 G0 위반이고 A1의 모든 초는 무효.
   `calls == device_calls + host_calls`가 회계 항등식.
2. `forms_hit{...}`와 `libm_form_hit` — §4의 이유 (b)가 살아 있는지. **§3의 표를 이 값으로 채운다.**
3. `host_fallback_nodes` — 0이 아니면 stderr의 `[RASBERY][WARN][flatxs.stream]` 줄에 이름이 있다.
4. SPTELEM `nested_wall "xsrecon/flatxs"` — A0의 1.9 s가 A1에서 얼마가 되는가. **이것이 절감이다.**
5. `bytes_elided` vs `bytes_h2d` — 스트림은 양방향으로 사라지고, 대신 노드 열(`tful`이 신규,
   브래킷 표가 신규)이 WP13 그림자 아래 올라간다. **순증이 아님을 여기서 읽는다.**
6. `[RASBERY][XFER]` 사이트 원장 — `stream_did`/`stream_x`/`stream_scale` 행이 A1에서 사라져야 한다.

**정답 게이트**: `libm_form_hit == 0` **그리고** `forms_sound == 1`이면 digest는 **같아야 한다**
— 그때 아름은 **B0**이고, 다르면 그것은 버그이지 등급이 아니다. `libm_form_hit == 1`이면
digest 비교는 무의미하고 **Gate A**(FP64 호스트 궤적 대비, `tools/gate_a_compare.py`) +
**Gate B**(MASTER 대비 핀 RMS, `tools/gate_b_pin_rms.py`) + `h5diff`의 정성 확인만 유효하다.

### 7.1.1 A1 이전에 먼저 — 마스크 채굴 (WP23.1)

**빌드 호스트에서 30 ms.** 238에서 RASBERY를 돌리기 전에 게이트를 한 번 돌린다.

```
ctest -R flatxs_stream_form_probe --output-on-failure
#   또는 직접:  ./rasbery_flatxs_stream_form_probe
```

읽을 것:

1. `sound=1` — 아니면 **여기서 멈춘다.** 네 씨앗 중 하나도 0에 못 갔다는 뜻이고,
   그때는 이 빌드의 반올림 계약을 아무도 모른다. A1을 돌려봐야 digest 비교가 무의미하다.
2. **채굴된 마스크 값** — 저작 호스트(WSL2 / g++ 13.3 / `-march=native`)는 `0x15`.
   238(Xeon Gold 5317, gcc 14.3)이 **다른 값을 낼 수 있고 그것이 정상이다.**
   `CMFD_OUTER_FORMS`가 `0x6`(dev) vs `0x7`(238)이었던 것과 같은 종류의 차이다.
   런 로그의 `[RASBERY][FORMS] {"mask":"FLATXS_STREAM_FORMS", ...}` 한 줄이 그것을 말하고,
   `"source":"mined"`가 아니라 `"build_default_mining_failed"`면 (1)로 돌아간다.
3. 세 사이트의 `-> NONE` 카운트가 모두 **0보다 큰가** — 아니면 픽스처가 사이트를 안 집고
   있다는 뜻이므로 마스크가 아니라 픽스처를 고쳐야 한다.
4. `libm ... mismatch / max_ulp` — §4.2의 표가 이 호스트에서도 같은 모양인지.
   glibc 버전이 다르면 242가 움직일 수 있다. **움직여도 결론(기본 `fast`)은 안 바뀐다.**

그리고 A1의 수신증에서 다음 세 필드를 §4.1 표와 대조한다:

```
"forms_mask":"0x15","forms_source":"mined","forms_sound":1,"libm":"fast"
```

`forms_source`가 `"env"`인데 그럴 의도가 없었다면 셸에 `RASBERY_FLATXS_STREAM_FORMS`가
남아 있는 것이고, 그 런의 digest 비교는 무효다.

### 7.1.2 선택 — `exact` libm A/B

```
RASBERY_GPU_FLATXS_STREAM=1 RASBERY_GPU_FLATXS_STREAM_LIBM=exact RASBERY_SPTELEM=1   ./RASBERY kngr_238.json --out E:/rasbery_runs/<date>/238/wp23/a1x_%d.h5
```

기대: `libm_form_hit == 1`인 덱에서 A1과 A1x의 digest는 **다르다**(그것이 §4.2의 차이다).
값어치는 Gate A의 잔차가 **줄어드는지**에 있다 — 줄어들면 `exact`가 호스트에 더 가깝다는
§4.2의 주장이 실측으로 확인된 것이고, 기본값을 바꿀 근거가 생긴다. 안 줄어들면 그 근거가 없다.

### 7.2 블록 B — 노달 상수 아름 단독

```
RASBERY_GPU_NODAL_CONSTS=1 RASBERY_SPTELEM=1 \
  ./RASBERY kngr_238.json --out E:/rasbery_runs/<date>/238/wp23/b1_%d.h5
```

읽을 것: `[RASBERY][NODAL_CONSTS]`의 `device_builds`(G0), `uploads_elided`(9의 배수여야 함),
`bytes_elided`(A0의 `[RASBERY][MICX]` `nodal_const_bytes`와 맞아야 함), 그리고 **`max_ulp`**.
`max_ulp <= 1` & `over_1ulp == 0`이 아니면 **여기서 멈추고** §6의 세 후보를 본다.
등급이 N1(측정)이므로 게이트는 **Gate A/B 전용**이다.

### 7.3 블록 C — 두 아름 동시 + 배치

```
RASBERY_GPU_FLATXS_STREAM=1 RASBERY_GPU_NODAL_CONSTS=1 ... (단일, 3회 중앙값)

# 배치: 8×M16, 중앙값 3회, MPS 켜고
RASBERY_GPU_FLATXS_STREAM=1 python3 tools/run_multi_gpu_batch.py --decks 128 --slots 16 --procs 8 ...
```

**배치에서 무엇을 기대하는가.** 스트림 아름의 절감은 **호스트 코어**다. 단일 덱에서는 ~1.0 s의
벽시계지만, 8×M16에서는 그 1.0 s가 16개 슬롯 × 8 프로세스에서 호스트 CPU를 경쟁적으로 먹던
것이므로, 배치 처리량(c/h)의 개선이 단일 덱 비율보다 **클 수 있다.** 반대로 디바이스가
이미 포화라면 개선이 없을 수도 있다. 둘 다 가능하고, 그래서 배치를 따로 잰다.
노달 상수 아름은 **버스**를 비우므로 MPS 아래 여러 프로세스가 PCIe를 공유할 때가 유리하다.

### 7.4 실패 시 첫 확인

- `device_calls == 0` → stderr의 `[RASBERY][WARN][flatxs.stream]` 이름. 대부분 `form` 또는 `model`.
- digest가 움직였고 `libm_form_hit == 0` → 먼저 `forms_sound`를 본다.
  **`forms_sound == 1`이면 이유 (a)가 아니다** — 마스크는 이 호스트에서 인용과 비트 일치가
  검증된 값이므로, 원인은 다른 곳(패킹, 브래킷, 프로브)이다. 마스크를 훑는 것은 시간 낭비다.
  `forms_sound == 0`이면 채굴이 실패한 것이고, 그때만 `RASBERY_FLATXS_STREAM_FORMS`를
  `0x00 … 0x3f`(2비트×3사이트, 유효 상태 조합은 27가지)로 훑는다. digest가 복구되는 값이
  있으면 픽스처가 그 사이트를 못 집은 것이므로 **픽스처를 고친다**(마스크를 하드코딩하지 않는다).
- digest가 움직였고 `libm_form_hit == 1` → §4.2. 그것은 등급이지 버그가 아니다.
- `RASBERY_GPU_FULL=1`에서 케이스가 죽음 → 그게 정상이다. 아름이 폴백했다는 뜻이고
  `Subsystem::flatxs_stream` / `nodal_consts` 카운터가 어느 심(seam)인지 말한다.

---

## 8. 남은 부채 (다음 WP가 집을 것)

1. ~~**수축 마스크 채굴기.**~~ **WP23.1에서 갚음** — §4.1. 남은 것은 238에서 채굴값을
   한 번 확인하는 것뿐이고, 그것은 §7.1.1의 30 ms짜리 절차다.
2. **오프라인 리플레이 게이트.** `RASBERY_FLATXS_DUMP` 캡처는 **해석된 스트림**을 담지만
   스펙트럴 히스토리 라이브러리 표는 담지 않아서, 디바이스 바디를 호스트 해석기에 대해
   오프라인으로 채점할 수 없다. 캡처 포맷에 §3의 평탄화 표를 추가하는 것이 그 게이트의 전제다.
3. **삽입(rodded) 노드.** ctype-0 붕괴(§3.1)가 무효가 되므로 모델당 전 ctype 행과
   `FineRodThermalFluenceAverage`의 미세 메쉬 평균이 필요하다. 오늘은 `kRefusalRodded`.
4. **`tful/tmod/dmod` D2H.** 이 아름은 소비자 하나(`BuildFlatXsStream`)를 옮겼을 뿐이다.
   WP22 수신증이 남긴 잔여는 다른 호스트 소비자들이 남아 있는 한 그대로다.
5. **컴파일 검증.** `CudaXsReconBackend.cu`는 `tools/check_cuda_syntax.py`의 shim이 아직
   커버하지 못한다(graph API, `cudaHostRegister`, `cudaStreamBeginCapture`, `deviceBlockAlloc`,
   `FlatXsCtaKernel.cuh`의 device-only `#error`). 238에서 nvcc가 이 파일을 처음 본다.
   WP23.1이 이 파일에 넣은 변경(마스크 위임, `libm` 인자)은 shim 통과 개수를 **바꾸지 않았다**
   — HEAD와 동일한 202개 shim 결함, 새 오류 0.
6. **libm 이유 (b) — B0로 가는 유일한 길.** 정확 반올림은 답이 아니다(§4.2가 잰 대로,
   틀리는 쪽은 glibc다). 등식 `device == glibc`를 만들려면 **glibc의 알고리즘을 디바이스에
   재현**해야 한다. `cbrt`는 짧다(`sysdeps/ieee754/dbl-64/s_cbrt.c`, 다항식 + Newton, 표 없음);
   `log`는 128엔트리 표 + 다항(ARM optimized-routines). **코드 결정이 아니라 정책 결정**이므로
   여기서 하지 않았다 — glibc는 LGPL, optimized-routines는 MIT이고, 어느 쪽이든 이 트리에
   외부 수치 코드를 들이는 것은 승인이 필요한 일이다. 그 승인이 나면 `cbrt`부터가
   비용 대비 효과가 가장 크다(§4.2의 538,003 vs 242).
7. **`exact` 경로의 디바이스 실측.** `FlatXsStreamExactMath.h`는 g++에서만 돌아봤다.
   "구성상 nvcc와 같은 비트"는 IEEE-754가 `+ − × ÷ fma`에 주는 보장에서 따라오는 주장이지
   측정이 아니다. 238에서 `--fmad=false` TU로 컴파일된 뒤 §7.1.2가 첫 실측이다.
