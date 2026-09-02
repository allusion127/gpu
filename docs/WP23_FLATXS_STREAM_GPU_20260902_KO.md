# WP23 — 분기 스트림 해석기(BuildFlatXsStream)와 노달 상수의 디바이스 이식

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `XSSet::BuildFlatXsStream`(≈1.0 s/단일 KNGR 런) · `XsReconBackend::solveNodal`의 상수 9배열 업로드(9,675회 / 3.9 GB) |
| 플래그 | `RASBERY_GPU_FLATXS_STREAM=1`(기본 off, `RASBERY_GPU_FLATXS`의 하위 아름) · `RASBERY_FLATXS_STREAM_STRIDE` · `RASBERY_FLATXS_STREAM_FORMS` · `RASBERY_GPU_NODAL_CONSTS=1`(기본 off) |
| 게이트 등급 | 스트림 = **N1**(이유 둘, §4) · 노달 상수 = **N1(측정됨)**(§6) |
| 계약 테스트 | `tools/test_flatxs_stream_contract.py`(16 규칙) · `tools/test_nodal_consts_gpu_contract.py`(13 규칙), 각 규칙마다 negative control |
| 동반 테스트 | `test_flatxs_cta_contract` · `test_micx_resident_contract` · `test_micx_layout_contract` · `test_xfer_ledger_contract` · `test_th_gpu_contract` · `test_gpu_full_fail_closed` · `test_enum_alias_contract` · `test_dependent_template_contract` — 전부 PASS |
| 신규 소스 | `src/FlatXsStreamKernel.h` · `src/FlatXsStreamReceipt.h` · `src/NodalConstsReceipt.h` |
| 수정 소스 | `src/XSSet.{h,cpp}` · `src/FlatXsKernel.h` · `src/CudaXsReconBackend.{h,cu}` · `CudaXsReconBackendStub.cpp` · `src/Nodal.cpp` · `src/Driver.h` · `src/GpuFullContract.h` |
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
| 기준 밀도/조건 보간 | B0 **가능** | `acc += f·(hi−lo)` 3사이트. **마스크 미채굴** → 이유 (a) |
| libm 없는 13형태 좌표 | B0 **가능** | `+ − × ÷ sqrt`뿐 |
| libm 7형태 좌표 | **N1** | glibc `log`/`cbrt` ≠ CUDA `log`/`cbrt`, ~1 ulp → 이유 (b) |
| 고정 슬롯 패킹 | **B0** | 순열이 아니라 **주소**만 바꾼다. 노드 내부 순서 불변 |

**따라서 아름 전체는 N1이고, 이유는 독립적으로 둘이다.**

- **(a) 수축 마스크가 채굴되지 않았다.** `FS_REFDENS` / `FS_REFDENS0` / `FS_REFCOND`
  세 비트의 기본값은 `0`(아무것도 융합 안 함)이고, 그것은 gcc에 대한 **추측이지 측정이 아니다.**
  `src/ThFormMiner.cpp` 모양의 채굴기가 이 셋을 고정하기 전까지는 libm을 하나도 안 쓰는
  라이브러리에서도 B0가 아니다. `RASBERY_FLATXS_STREAM_FORMS=0x…`로 override 가능.
  **세 비트인 이유**: 같은 모양의 lerp라도 gcc는 문장마다 독립적으로 고른다(xsrecon 캠페인의 발견).
  `referenceDensity`(`value += fraction*(hi−lo)`)와 `referenceDensity0`(`v += f*(hi−v)`)은
  **같은 수를 다르게 쓴 것**이며, 합치는 것은 비트를 바꾸는 단순화라 합치지 않았다.
- **(b) 일곱 형태가 libm을 부른다.** `FlatXsKernel.h`가 애초에 좌표를 호스트에 남긴 이유가
  이것이고, 이 WP는 그 문장을 **뒤집는 것이 아니라 한정한다.** 수신증의 `libm_form_hit`가
  이번 런에 이 이유가 살아 있는지를 말한다.

두 이유는 **고쳐야 할 것이 다르다**. 하나의 등급으로 합치면 무엇을 갚아야 하는지가 사라진다.

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

**정답 게이트**: `libm_form_hit == 0`이면 digest 동등성을 **시도해 볼 수는 있다**(이유 (a)만 남고,
그것은 마스크가 우연히 맞았을 때 통과한다). `libm_form_hit == 1`이면 digest 비교는 무의미하고
**Gate A**(FP64 호스트 궤적 대비, `tools/gate_a_compare.py`) + **Gate B**(MASTER 대비 핀 RMS,
`tools/gate_b_pin_rms.py`) + `h5diff`의 정성 확인만 유효하다.

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
- digest가 움직였고 `libm_form_hit == 0` → **이유 (a)**. `RASBERY_FLATXS_STREAM_FORMS`를
  `0x0 … 0x7` 8가지로 훑어 digest가 복구되는 마스크가 있는지 본다. 있으면 그것이 채굴기가
  자동으로 찾아야 할 값이고, 없으면 다른 곳이 문제다.
- `RASBERY_GPU_FULL=1`에서 케이스가 죽음 → 그게 정상이다. 아름이 폴백했다는 뜻이고
  `Subsystem::flatxs_stream` / `nodal_consts` 카운터가 어느 심(seam)인지 말한다.

---

## 8. 남은 부채 (다음 WP가 집을 것)

1. **수축 마스크 채굴기.** `src/ThFormMiner.cpp` 모양으로 세 lerp 사이트를 프로덕션 철자에 대해
   고정해야 이유 (a)가 사라진다. 그전까지 libm 없는 덱에서도 B0를 주장할 수 없다.
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
