---
title: IISC 보정의 코드 구현
aliases:
  - IISC implementation
  - IISC/RHST/RDPL code implementation
  - isotope vector correction implementation
tags:
  - Rasbery
  - CHIFFON
  - IISC
  - cross-section
  - implementation
---

# IISC 보정의 코드 구현

이 문서는 **IISC**(isotope-vector 기반 spectral history 보정)가 CHIFFON
라이브러리 생성기와 RASBERY 노심 계산기 안에서 *실제로 어떻게 구현되어
있는지*를 코드 수준에서 정리한다. 물리적 동기·후보 선정·검증 결과 같은
**방법론** 관점은 다음 문서가 다룬다.

- 기준 방법론: [`test/9-1_IISC/docs/keyless_iisc_rhst_methodology_ko.md`](../test/9-1_IISC/docs/keyless_iisc_rhst_methodology_ko.md)
- 상세 방법론: [`test/9-1_IISC/docs/iisc_rhst_methodology_detailed_ko.md`](../test/9-1_IISC/docs/iisc_rhst_methodology_detailed_ko.md)

이 문서는 그 둘의 "코드 대응판"이다. 모든 수식은 코드의 실제 동작과 일치하도록
적었고, 본문의 `file:line`은 검증된 위치를 가리킨다.

> 이 문서는 과거의 `docs/isotope-vector-absolute-method.md`(trajectory-keyed
> `kind 4/5` + `trajectory_weight` 방식 설명)를 대체한다. 그 방식은 현재 코드에
> 더 이상 존재하지 않는다.

---

## 0. 한눈에 보기

전체 보정은 reference 단면적 위에 **독립적인 delta들의 단순 합**으로 얹힌다.

$$
\Sigma \;=\; \Sigma_{\mathrm{ref}}(B,c)
\;+\;\underbrace{\Delta\Sigma_{\mathrm{BPPM}}+\Delta\Sigma_{\mathrm{Tfuel}}+\Delta\Sigma_{\mathrm{Dmod}}}_{\text{instantaneous branch}}
\;+\;\Delta\Sigma_{\mathrm{RDPL}}
\;+\;\underbrace{\Delta\Sigma_{\mathrm{IISC}}+\Delta\Sigma_{\mathrm{IISC\text{-}RHST}}}_{\text{history (vector) 보정}}
$$

| 항 | 의미 | indicator (기준안) | 적용 노드 |
|---|---|---|---|
| `BPPM` / `Tfuel` / `Dmod` | instantaneous branch 보정 | boron 밀도 / $\sqrt{T_f}$ / $\rho_m$ | 전부 |
| `RDPL` | rod material 자체 depletion | $\widehat\Phi_{\mathrm{rod}}$ (rod fluence) | rodded |
| `IISC` | 일반 spectral history | $N_{\mathrm{Xe135}},\,N_{\mathrm{Pu239}},\,N_{\mathrm{U238}}/N_{\mathrm{Pu239}}$ | 전부 |
| `IISC-RHST` | rodded fuel burn-path residual | $\widehat\Phi_{\mathrm{rod}},\,N_{\mathrm{Pu240}},\,N_{\mathrm{Pu241}}$ | rodded history 보유 노드 |

역할 분담:

- **CHIFFON (offline fit)** — HGC를 읽어 각 delta의 계수를 적합한다.
  핵심 코드는 `include/chiffon/{Model,Interpolator,Importer,Exporter}.h`.
- **RASBERY (runtime)** — 노드별 상태(연소도, isotope 재고, rod 상태)에서
  delta를 평가해 단면적에 더한다. 핵심 코드는 `src/XSSet.{h,cpp}`, 상태
  저장은 `src/IO.cpp`.

---

## 1. 데이터 구조 (`Model.h`)

### 1.1 보정의 종류 키 — `Hk`

history 보정은 `HistoryDeltaCorrection::kind`(정수)로 종류를 구분한다.
값은 HDF on-disk 호환 때문에 고정이다.

```cpp
namespace Hk {
enum : int {
    VEC             = 4,   // 레거시 ctype-keyed spectral vector
    RHST_UNIT       = 7,   // rod-history residual vector (IISC-RHST)
    CTYPE_INDEP_VEC = 26   // ctype-independent IISC vector
};
}
```

런타임에서 적용되는 kind는 `CTYPE_INDEP_VEC`와 `RHST_UNIT` 둘뿐이다
([§5.2](#52-applyhistorydeltastonode)). keyless 설계상 **IISC는 항상
ctype-independent(`CTYPE_INDEP_VEC`)로 생성**되고, ctype-keyed `VEC`(kind 4)는
더 이상 만들어지지 않는다(입력에서 `ctype_independent: false`를 주면 import 단계에서
거부). `VEC` enum은 과거 HDF를 읽기 위한 호환용으로만 남아 있다([§6.1](#61-keyless-설계-iisc는-ctype-independent-ctype-key-미사용)).

### 1.2 합성 좌표 — `Hv`

isotope index가 아닌 특수 좌표(rod fluence, log 밀도, burnup 등)는
`SIZE_MAX` 근처의 sentinel 정수로 표시한다.

```cpp
namespace Hv {
enum : size_t {
    ROD_DEPL, LOG_XE, LOG_PU, LOG_U_PU, TOT_BURN, CUR_ROD_FRAC,
    FUEL_ROD_FLU, ROD_BURN_FRAC, FUEL_ROD_BURN, ROD_BURN,
    ROD_FLU, FAST_THERM, THERM_FRAC          // (실제 값은 SIZE_MAX - k)
};
inline constexpr double ROD_FLU_SCALE = 1.0e-22;
}
```

> **중요 (런타임 지원 범위).** CHIFFON 적합 단계에서는 13개 합성 좌표 전부가
> `Interpolator::PointDensity`에서 평가된다. 그러나 RASBERY 런타임
> (`ApplyHistoryDeltasToNode`)에서는 **`Hv::ROD_FLU` 하나만** 특수 처리되고,
> 나머지는 평가 시 0이 된다([§5.3](#53-좌표-평가자-histcur--histref)). 결과적으로
> *end-to-end로 동작하는 좌표는 실제 isotope 밀도와 `ROD_FLU`뿐*이다. 기준안의
> indicator(Xe135/Pu239/U238/Pu240/Pu241 + rod_fluence)는 모두 이 범위 안에 있다.

### 1.3 보정 객체 — `HistoryDeltaCorrection`

```cpp
struct HistoryDeltaCorrection {
    BranchDelta          delta;           // [ctype][burnup key] -> DeltaCrossSection
    BRANCHTYPE           branch_type = SPCT;
    int                  kind        = 4;  // Hk::VEC/RHST_UNIT/CTYPE_INDEP_VEC
    int                  state_field = 9;  // (직렬화 전용, 계산엔 미사용)
    std::vector<size_t>  vector_isotopes;  // 좌표 index 목록 (isotope or Hv::)
    std::vector<int>     vector_powers;    // 위 좌표에 곱해지는 거듭제곱
};
```

- `BranchDelta = unordered_map<int, map<int, DeltaCrossSection>>` —
  control-rod type(ctype) → 정수 burnup key → 계수.
- `DeltaCrossSection`은 거듭제곱 다항식(또는 spline) 계수를 Horner 형태로
  보관한다. IISC/RHST의 vector 항은 항당 `(상수, 기울기)` 2-계수 다항식이다.
- `state_field`는 HDF에 직렬화되고 reflector 평균의 메타데이터 동일성 비교에만
  쓰이며, 단면적 계산에는 전혀 읽히지 않는다.

### 1.4 라이브러리 점(`DepletionPoint`)의 history 메타데이터

```cpp
int    _ctyp;                 // 현재 control-rod type
int    _trajectory_ctyp;      // depletion trajectory의 ctype
bool   _trajectory_reference; // rod-history set의 main/nondepleted 기준점 여부
double _fuel_rod_fluence;     // rodded spectrum 아래 fuel 노출 [n/cm2]
double _rod_fluence;          // rod material 자체 fluence [n/cm2]
```

이 필드들은 Importer가 HGC를 태깅할 때 채우고([§4](#4-hgc-import-와-태깅-importerh)),
Interpolator가 좌표/residual을 만들 때 읽는다.

---

## 2. 좌표(coordinate) 정의

vector 보정 한 항은 `(vector_isotopes, vector_powers)`로 표현된다. 평가는
`Interpolator::EvalVectorTermFromAccessors`와
`EvalAbsoluteVectorTermFromAccessor`가 담당한다.

### 2.1 단일 isotope / 곱 항

거듭제곱이 모두 0 이상이면 절대 좌표의 곱이다.

$$
x \;=\; \prod_i N_i^{\,p_i}, \qquad p_i \ge 0 .
$$

기준안의 IISC는 $\{N_{\mathrm{Xe135}}\},\{N_{\mathrm{Pu239}}\}$ 처럼 1차 항을 쓴다
(각 isotope에 대해 단위 거듭제곱 벡터 하나씩 자동 생성, [§4.3](#43-vector-settings-파싱)).

### 2.2 비율(ratio) 항

거듭제곱 중 하나라도 음수면 비율 항으로 본다. 음수 거듭제곱 isotope는
분모로 간다.

$$
x \;=\; \frac{\prod_{p_i>0} N_i^{\,p_i}}{\prod_{p_j<0} N_j^{\,|p_j|}} .
$$

예: `["U238","Pu239"]` 비율은 powers `U238:+1, Pu239:-1`로 인코딩되어

$$
R_{U/P} \;=\; \frac{N_{\mathrm{U238}}}{N_{\mathrm{Pu239}}}
$$

가 된다. 분모가 너무 작으면 인위적 큰 비율을 막기 위해 0을 반환한다.

$$
\text{denominator} < \texttt{HISTORY\_RATIO\_DENOM\_EPS}=10^{-10}
\;\Longrightarrow\; x = 0 .
$$

### 2.3 2차 이상 곱 항은 reference 기준으로 centering

거듭제곱 합이 1보다 크고(2차 이상) 음수가 없는 *순수 곱* 항은, 1차 축에
흡수되지 않도록 reference 상태를 빼고 평가한다(`IsCenteredProductTerm`).

$$
x \;=\; \prod_i \bigl(N_i - N_{i,\mathrm{ref}}\bigr)^{p_i} .
$$

1차 항과 비율 항은 절대 좌표를 그대로 쓴다(위 §2.1, §2.2).

### 2.4 rod fluence 좌표

rod fluence는 선형 fit의 conditioning을 위해 $O(1)$로 스케일한다
(`hvRodFluCoord`).

$$
\widehat\Phi_{\mathrm{rod}} \;=\; 10^{-22}\,\Phi_{\mathrm{rod}}, \qquad
\texttt{ROD\_FLU\_SCALE}=10^{-22}.
$$

`Hv::ROD_FLU`는 rod material fluence(`_rod_fluence`), `Hv::FUEL_ROD_FLU`는
fuel 노출(`_fuel_rod_fluence`)에 같은 스케일을 적용한다. 런타임에서는
`ROD_FLU`만 평가된다.

### 2.5 합성 좌표 요약표

| 좌표 | 의미 | 적합(fit) | 런타임 |
|---|---|:---:|:---:|
| 실제 isotope index (`Xe135`,`Pu239`,`Pu240`,`Pu241`,`U238`,…) | isotope 밀도 | ✅ | ✅ |
| `ROD_FLU` | rod material fluence | ✅ | ✅ |
| `FUEL_ROD_FLU` | fuel rod 노출 | ✅ | ❌(0) |
| `ROD_DEPL` | rod fluence 재사용 좌표 | ✅ | ❌(0) |
| `LOG_XE`,`LOG_PU`,`LOG_U_PU` | log 밀도/비율 | ✅ | ❌(0) |
| `TOT_BURN`,`FUEL_ROD_BURN`,`ROD_BURN` | burnup 좌표 | ✅ | ❌(0) |
| `CUR_ROD_FRAC` | 현재 rod 삽입 지시자 | ✅ | ❌(0) |
| `ROD_BURN_FRAC` | rodded burnup 분율 | ⚠️ degenerate(=1) | ❌(0) |
| `FAST_THERM`,`THERM_FRAC` | flux 비/분율 | ✅ | ❌(0) |

❌(0) 좌표를 vector에 넣으면 런타임에서 그 항은 항상 0으로 평가되어 보정에
기여하지 않는다(에러 없음). 기준안은 ✅ 좌표만 사용한다.

---

## 3. CHIFFON offline fit (`Interpolator.h`)

`Interpolator::Interpolate(model, ...)`가 한 모델의 모든 delta를 순서대로
적합한다. 순서가 곧 residual 정의의 핵심이다.

```text
BPPM  ->  TFUL  ->  DMOD(pre_remove: bppm)  ->  RDPL
      ->  IISC(pre_remove: dmod; prior history 제거)
      ->  IISC-RHST(pre_remove: dmod[, rod_depletion]; prior IISC 제거)
```

### 3.1 instantaneous branch (BPPM / TFUL / DMOD)

각 branch는 HGC branch point와 같은 ctype/burnup의 reference 사이 XS 차이를
branch 변수에 대해 보간한다. 좌표는 각각 boron 밀도, $\sqrt{T_f}$, $\rho_m$.
DMOD는 이미 적합된 BPPM 효과를 먼저 빼고 남은 residual을 적합한다
(`pre_remove = ["bppm"]`).

이 1-변수 branch 적합은 `Interpolator::InterpolatePoly`(다항) 또는
`InterpolateSpline`(piecewise)이 담당한다. `InterpolatePoly`는 **제약 있는
최소제곱(normal-equation / KKT)** 으로 푼다. Vandermonde $V$($V_{ki}=x_k^{\,i}$)로
정규방정식 $V^{\mathsf T}V$를 만들고, 고정점(예: DMOD는 양 끝 $x$를 정확히
지나도록 `fixed_indices = {0, n}`)을 Lagrange 제약행 $C$로 묶은 KKT 행렬

$$
\begin{bmatrix} V^{\mathsf T}V & C^{\mathsf T}\\ C & 0\end{bmatrix}
$$

을 LU로 역행렬화해 계수를 얻는다. 제약이 없으면 보통의 LSQ
$(V^{\mathsf T}V)\,b = V^{\mathsf T}y$와 같다. 이 normal-equation 경로는 예전부터
그대로이며, 아래 §3.5의 isotope-vector 적합(QR)과는 **별개 루틴**이다.

### 3.2 RDPL (rod material depletion) — `FitRodDepletionDeltas`

rod material이 *연소된* 단면적과 *신선한* 단면적의 차이를 rod fluence 축으로
적합한다.

$$
\mathbf R_{\mathrm{RDPL}}(B,c) \;=\;
\Sigma_{\mathrm{rod,depl}}(B,c)\;-\;\Sigma_{\mathrm{rod,nondepl}}(B,c) .
$$

- **축**: `RodDepletionAxis = _rod_fluence`(유한할 때), 아니면 burnup.
- **paired-reference 경로**: `_trajectory_reference`(nondepleted) 점이 있으면
  같은 burnup에서 depleted − nondepleted를 만들고, $(x=0,\ \mathbf 0)$ zero anchor를
  추가한다.
- **fallback 경로**: nondepleted 점이 없으면 `|burnup|`이 가장 작은 점을 기준으로
  잡는다(이 경로에는 별도 zero anchor가 없다).
- **저장**: `_rod_depletion_delt[ctype][0]` — burnup key를 0으로 고정한 *rod fluence
  응답 테이블*이다(default `spline`, order 1).
- macroscopic XS는 비우고 micx/lmpx만 보관한다.

런타임 적용(`Model::ApplyRodDepletion`)은 `ctype>0 && fluence>0`일 때만,
fluence를 knot 범위로 clamp해서 더한다.

### 3.3 IISC vector — `AddCurrentSpectralFitPoint`

HGC `extra` 점 $j$와 *같은 ctype/burnup의 main reference* 사이 residual을 적합한다.

$$
\mathbf R^{(j)}_{\mathrm{IISC}}(B) \;=\;
\Sigma^{(j)}_{\mathrm{HGC}}(B)\;-\;\Sigma_{\mathrm{ref}}(B,c_{\mathrm{cur}})
\;-\;\Delta\Sigma_{\mathrm{pre}}\;-\;\sum_{\text{prior}}\Delta\Sigma_{\text{prior}} .
$$

- `pre_remove` 기본값은 `["dmod"]` → $\Delta\Sigma_{\mathrm{pre}}=\Delta\Sigma_{\mathrm{Dmod}}$.
- IISC 단계에서는 `_history_deltas`가 비어 있어 prior 항이 사실상 없다.
- macroscopic XS는 비운다.
- 저장 ctype: 항상 0 (keyless 설계상 IISC는 ctype-independent).

### 3.4 IISC-RHST vector — `AddRhstCurrentSpectralFitPoint`

`rod history` HGC 점을 *current-state* residual sample로 본다. 이미 적합된
branch / RDPL / IISC를 모두 제거한 잔차를 적합한다.

$$
\begin{aligned}
\mathbf R^{(j)}_{\mathrm{RHST}}(B) =\;&
\Sigma^{(j)}_{\mathrm{rod\,history}}(B)\;-\;\Sigma_{\mathrm{main\,ref}}(B,c_{\mathrm{cur}})\\
&-\;\Delta\Sigma_{\mathrm{Dmod}}^{(j)}
\;-\;\Delta\Sigma_{\mathrm{RDPL}}^{(j)}
\;-\;\Delta\Sigma_{\mathrm{IISC}}^{(j)} .
\end{aligned}
$$

- main reference는 `FillReferenceState`(burnup 보간 포함)로 만든다.
- prior IISC 제거는 `SubtractPriorHistoryDeltas`(이미 append된 IISC를 뺀다).
- `_trajectory_ctyp <= 0`인 점은 건너뛴다.
- 저장 ctype은 현재 ctype(pair key 없음).
- rod-history가 없는 SPCT 점에 대해서는
  `AddCurrentSpectralResidualZeroPoint`로 잔차 표면을 0으로 anchoring한다.

> **`pre_remove` 주의.** C++ `SPCT_SETTINGS`의 default `pre_remove`는 `["dmod"]`다.
> RHST 블록을 따로 주지 않으면 RHST는 SPCT 설정을 복사하므로 `["dmod"]`가 된다.
> 다만 기준안 `iisc_rhst_best`는 입력에서 `rhst spectral` 블록에
> `pre_remove=["dmod","rod_depletion"]`를 명시하므로 RHST 잔차에서 RDPL까지 뺀다.

### 3.5 vector 계수 적합 — `FitVectorTermCoefficients`

residual을 reference에서 0이 되도록 centering한 **선형 최소제곱**으로 푼다.
sample $j$(= zero anchor + 각 HGC residual point)에 대해

$$
\mathbf y^{(j)} \;\approx\; \sum_{\alpha} \mathbf C_\alpha
\bigl(x^{(j)}_\alpha - x^{(0)}_\alpha\bigr),
\qquad (x^{(0)},\mathbf y^{(0)}) = (x_{\mathrm{ref}},\,\mathbf 0)
$$

를 만족하는 CrossSection-값 계수 $\mathbf C_\alpha$를 찾는다.

- 첫 sample은 reference isotope vector의 **zero residual anchor**
  (`AddVectorReferenceZero`)다.
- design 행렬 $X$($n_{\mathrm{data}}\times n_{\mathrm{term}}$, 행 = sample,
  열 = centering·스케일된 $x_\alpha-x_{\mathrm{ref},\alpha}$)에서 분산 0인 feature와
  collinear feature를 먼저 떨군다.
- 남은 $X$를 **modified Gram–Schmidt QR**($X=QR$)로 분해하고
  $X^{+}=R^{-1}Q^{\mathsf T}$를 만들어 계수를 $\mathbf C = X^{+}\mathbf Y$로 얻는다
  ($\mathbf Y$는 sample별 residual CrossSection을 쌓은 것). 모든 XS 성분/group이
  같은 $x$좌표를 공유하므로 $X^{+}$ 하나를 전 성분에 적용한다.
- rank-deficient면 ridge 등 정규화를 쓰지 않고 해당 블록을 비운다(빈 결과 반환).
- 결과는 동치인 $c_0 + \sum_\alpha c_\alpha x_\alpha$ 형태로 저장한다
  ($c_0 = -\sum_\alpha c_\alpha x_{\mathrm{ref},\alpha}$). 항마다
  `(상수, 기울기)` 2-계수 `DeltaCrossSection`로 보관하고 상수항은 첫 항에만 둔다.
  macroscopic XS는 비운다.

> **QR ↔ normal-equation 동치.** QR 풀이는 정규방정식
> $(X^{\mathsf T}X)\,\mathbf C = X^{\mathsf T}\mathbf Y$와 **같은 최소제곱 해**다
> ($\min_{\mathbf C}\lVert X\mathbf C-\mathbf Y\rVert$). 정확산술에서 해 $\mathbf C$는
> 동일하고, QR은 조건수를 $\kappa(X)$로 다뤄 정규방정식의
> $\kappa(X^{\mathsf T}X)=\kappa(X)^2$ 제곱을 피하는 **수치 안정성 개선**일 뿐이다.
> 즉 "잔차 $\mathbf r = X\mathbf C-\mathbf Y$ 최소화"라는 목적은 그대로이고,
> $\mathbf r\to 0$은 sample 수 = 항 수인 determined일 때만이며 일반적인
> over-determined에서는 $\lVert\mathbf r\rVert$를 최소화한다(normal-equation도 동일).
> 이력: isotope-vector 적합은 도입(`926cc1d`)부터 QR이고, rank-deficient용
> ridge-normal-equation **fallback**이 `716409f`에서 제거되어 현재 QR-only다.
> 1-변수 branch 적합(§3.1)은 여전히 normal-equation/KKT를 쓴다.

XS 성분/group을 합친 index $q$에 대해, 저장된 한 표면은

$$
\Delta\Sigma_{q}(B,c,\mathbf z) \;=\; \sum_{\alpha} A_{q,\alpha}(B,c)\,
\bigl[z_\alpha - z_{\alpha,0}(B,c)\bigr]
$$

를 평가한다.

---

## 4. HGC import 와 태깅 (`Importer.h`)

### 4.1 입력 구조 (`ReadInput`)

```json
{
  "settings": {
    "discontinuity factor": true,
    "bppm":  { "apply": true, "order": 1, "type": "spline" },
    "tful":  { "apply": true, "order": 1, "type": "spline" },
    "dmod":  { "apply": true, "order": 1, "type": "spline", "pre_remove": ["bppm"] },

    "spectral": {                         /* IISC */
      "apply": true,
      "isotopes": ["Xe135", "Pu239"],
      "isotope ratio": [["U238", "Pu239"]],
      "ctype_independent": true,
      "pre_remove": ["dmod"]
    },
    "rhst spectral": {                    /* IISC-RHST */
      "apply": true,
      "isotopes": ["rod_fluence", "Pu240", "Pu241"],
      "pre_remove": ["dmod", "rod_depletion"]
    },
    "rod depletion": { "apply": true, "order": 1, "type": "spline" }
  },

  "fuels": {
    "W1": {
      "main":  "PLEU_base_0101.HGC",                    /* reference 연소 */
      "extra": ["PLEU_560_0101.HGC", "PLEU_600_0101.HGC", "PLEU_P50_0101.HGC"],  /* IISC fit 점 */
      "rod history": {
        "CR1": {
          "main":  "PLEU_ROD_0101.HGC",                 /* trajectory_reference */
          "extra": ["PLEU_ROD_560_0101.HGC", "PLEU_ROD_600_0101.HGC", "PLEU_ROD_P50_0101.HGC"]
        }
      },
      "rod depletion": {                                /* RDPL pair */
        "nondepleted": "PLEU_RODNONDEPL_0101.HGC",
        "depleted":    "PLEU_ROD_0101.HGC"
      }
    }
  }
}
```

- `main` → reference depletion set, `extra` → SPCT(IISC) 점,
  `rod history` → RHST 점, `rod depletion` → RDPL pair.
- RHST 벡터 설정 키는 `rhst spectral` / `rhst_spectral` /
  `rod history spectral` / `rod_history_spectral` 중 하나다. 이 키가 없으면
  RHST는 `spectral` 블록을 그대로 복사한다. (참고: `settings.rod history`라는
  키는 RHST *벡터 설정*으로 읽히지 않는다 — fuel 안의 `rod history`만 RHST
  fitting 점을 공급한다.)

### 4.2 RHST current/trajectory ctype 매핑 (`AppendHGCPoints`)

`rod history` 키의 숫자가 `trajectoryCtype`이 된다(`"CR1"` → 1). 점의 현재
ctype은 다음 규칙으로 정한다.

$$
\text{HGC ctype} = 0 \;\Rightarrow\; c_{\mathrm{cur}} = c_{\mathrm{traj}}, \qquad
\text{HGC ctype} = \text{CR*} \;\Rightarrow\; c_{\mathrm{cur}} = 0 .
$$

즉 rodded로 연소된 base reference는 rodded current state, 그 `CR*` branch는
rod-out current state로 해석한다. `current_ctype: "hgc"`(verbatim) 또는 정수로
이 매핑을 덮어쓸 수 있다. `main` 항목은 `_trajectory_reference = true`로 태깅한다.

### 4.3 vector settings 파싱 (`readVectorSettingsBlock`)

- `isotopes` 배열 → isotope마다 단위 거듭제곱 벡터 하나씩 생성.
- `isotope ratio`의 2-원소 항 → `+1/-1` 거듭제곱(분자/분모).
- `isotope product` / `cross term(s)` → 거듭제곱을 합산한 곱 항.
- `ctype_independent` 기본값 `true`(keyless), `pre_remove` 기본값 `["dmod"]`.
  `spectral` 블록에 `ctype_independent: false`를 주면 import 단계에서 거부된다.
- vector 보정에는 다항식 `order`/`type`이나 weighting 필드가 없다(있어도 무시).

isotope 이름은 `parseVectorIsotope`가 해석한다. 실제 isotope ID/기호 외에
`rod_fluence`, `log_pu239`, `current_rod_fraction` 같은 합성 좌표 별칭도 받는다.
단, [§2.5](#25-합성-좌표-요약표)의 ❌ 좌표 별칭은 런타임에서 0이 되므로 주의.

### 4.4 rod fluence 재구성 (`ReconstructFluenceFromHGC`)

rod fluence는 **외부 CSV 없이 각 HGC의 flux × time으로 재구성**한다.

$$
\mathrm{EFPD}\,[\mathrm{d}] = \frac{B\,[\mathrm{MWd/kgHM}]\times 1000}{q\,[\mathrm{W/gHM}]},
\qquad
\Phi = \sum_{\text{step}} \tfrac12\bigl(\phi_{k-1}+\phi_k\bigr)\,\Delta t,\ \ \Delta t = \Delta(\mathrm{EFPD})\times 86400\,\mathrm{s}.
$$

`AssignReconstructedFluence`: `_fuel_rod_fluence`는 재구성값, `_rod_fluence`는
nondepleted(신선한 absorber) 점이면 0, 아니면 재구성값.

---

## 5. RASBERY runtime 적용 (`XSSet.cpp`)

`UpdateFlatXS`가 노드별로 단면적을 다시 만든다. fuel 노드는 두 경로로 나뉜다.

### 5.1 노드 조립 경로

**Unrodded 노드** — `ApplyBranchDeltasToNode`

$$
\Sigma \leftarrow \Sigma_{\mathrm{ref}}(B,0)
+\Delta\Sigma_{\mathrm{BPPM}}+\Delta\Sigma_{\mathrm{Tfuel}}+\Delta\Sigma_{\mathrm{Dmod}}
+\Delta\Sigma_{\mathrm{IISC}}+\Delta\Sigma_{\mathrm{IISC\text{-}RHST}} .
$$

branch 좌표: boron 밀도 $=\rho_m\cdot w_{\mathrm{vfr}}\cdot \text{ppm}\cdot$`BORON_DENSITY_FACTOR`,
$\sqrt{T_f}$, $\rho_m$. 그다음 `ApplyHistoryDeltasToNode`로 history 항을 더한다.

**Rodded 노드** — `FillRodNodeXS` 이후 `ApplyHistoryDeltasToNode`

$$
\Sigma \leftarrow \Sigma_{\mathrm{ref}}(B,c_{\mathrm{cur}})
+\Delta\Sigma_{\mathrm{RDPL}}(c_{\mathrm{cur}},\widehat\Phi_{\mathrm{rod}})
+\Delta\Sigma_{\mathrm{IISC}}+\Delta\Sigma_{\mathrm{IISC\text{-}RHST}} .
$$

부분 삽입 노드는 fine-rod segment fraction으로 unrodded/rodded reference와 RDPL을
섞은 뒤 history 항을 더한다. **IISC와 IISC-RHST는 두 경로 모두에 적용된다.**

### 5.2 `ApplyHistoryDeltasToNode`

각 history 보정에 대해 kind로 분기한다.

```text
current_ctype    = UsesRodXS(l) ? _ctyp[l] : 0
trajectory_ctype = _history_ctyp[l]

kind == CTYPE_INDEP_VEC : storage_ctype = 0                 // 모든 노드 (IISC)
kind == RHST_UNIT       : if (trajectory_ctype <= 0) skip;
                          storage_ctype = current_ctype     // rodded history 노드만
else                    : skip                              // 레거시 VEC 등 (현재 fit은 생성 안 함)
```

이후 `(branch, storage_ctype, _burn[l])`로 `resolveDelta`가 burnup 양쪽 점을
bracket해 lo/hi 표면과 보간 분율을 찾고, 좌표 $x$를 평가해 더한다.

$$
\Delta\Sigma = (1-f)\,\Delta\Sigma_{\mathrm{lo}}(x) + f\,\Delta\Sigma_{\mathrm{hi}}(x),
\qquad f = \frac{B - B_{\mathrm{lo}}}{B_{\mathrm{hi}} - B_{\mathrm{lo}}}.
$$

> **rod 인출(withdrawn) 노드의 거동.** 현재 unrodded($c_{\mathrm{cur}}=0$)이지만
> rodded history($c_{\mathrm{traj}}>0$)가 있는 노드는 RHST가 `storage_ctype = 0`
> 표면으로 적용된다. 이때 rod fluence 좌표는 `FineRodFluenceAverage(l, 0) = 0`
> 이므로 0이 되고, RHST 잔차는 주로 $N_{\mathrm{Pu240}},N_{\mathrm{Pu241}}$ 차이와
> 표면 상수항이 담당한다.

### 5.3 좌표 평가자 `histCur` / `histRef`

```cpp
auto histCur = [&](size_t iso) -> double {
    if (iso == Hv::ROD_FLU)
        return hvRodFluCoord(FineRodFluenceAverage(l, current_ctype));
    const size_t idx = iso * stride + node;            // stride = nxyz
    return idx < _iden.size() ? _iden[idx] : 0.0;      // 실제 isotope 밀도
};
auto histRef = [&](size_t iso) -> double {
    if (iso == Hv::ROD_FLU) return 0.0;
    const size_t idx = iso * stride + node;
    return idx < _ref_iden.size() ? _ref_iden[idx] : 0.0;
};
```

- 현재 좌표는 `_iden`, reference 좌표는 `_ref_iden`(노드별 reference isotope 재고).
- `Hv::ROD_FLU`만 특수 처리. 다른 `Hv::` sentinel은 `iso*stride`가
  `_iden.size()`를 넘어 bounds check에 걸려 **조용히 0**이 된다
  ([§2.5](#25-합성-좌표-요약표) 참조).

### 5.4 `FineRodFluenceAverage`

fine 축방향 rod-state mesh에서, 요청 ctype과 일치하고 절반 이상 rodded인
($\texttt{\_fine\_rod\_frac} \ge 0.5$) cell의 fluence 산술 평균을 반환한다.
`ctype <= 0`이면 0.

---

## 6. 알려진 제약과 주의점 (footguns)

### 6.1 keyless 설계: IISC는 ctype-independent (ctype key 미사용)

이 보정은 **keyless로 마무리되었다.** trajectory/current pair key
($1000\,c_{\mathrm{traj}}+c_{\mathrm{cur}}$)는 생성되지 않고, 그것을 만들던
`HistoryCoefficientCType` helper도 제거되었다. RHST 잔차는 현재 ctype 표면에,
IISC는 ctype에 무관한 ctype 0 표면(`CTYPE_INDEP_VEC`)에 저장된다.

- `SPCT_SETTINGS`의 `ctype_independent` 기본값은 `true`이고, IISC fit은 항상
  `CTYPE_INDEP_VEC`를 생성한다.
- `spectral` 블록에 `ctype_independent: false`를 주면 import 단계에서 명확한
  에러로 거부된다(ctype-keyed IISC = 레거시 `VEC`는 더 이상 지원하지 않음).
- `VEC`(kind 4) enum은 과거 HDF 라이브러리를 읽기 위한 호환용으로만 남으며,
  런타임 apply 경로는 이를 건너뛴다.

### 6.2 런타임에서 실제 isotope + `ROD_FLU`만 평가

[§2.5](#25-합성-좌표-요약표)의 ❌ 좌표(예: `fuel_rod_fluence`, `current_rod_fraction`,
`log_*`, `total_burnup`)를 vector에 넣으면 런타임에서 0으로 평가되어 보정에
기여하지 않는다. 기준안 indicator는 모두 ✅ 좌표다.

### 6.3 `ROD_BURN_FRAC`은 degenerate

`ROD_BURN_FRAC` 좌표는 적합 단계에서 `hvRoddedBurnFrac(burnKey, burnKey)`로
계산되어 trajectory가 있으면 항상 1, 없으면 0인 지시자로 퇴화한다(진짜 분율이
아님). 게다가 런타임에서는 0이 된다.

### 6.4 rodded burnup은 더 이상 추적하지 않는다

과거의 노드별 `_rodded_burn`(rod-in 누적 연소도)은 어떤 단면적 계수에도 쓰이지
않아 **제거되었다.** keyless RHST는 rodded burnup이 아니라 `rod_fluence`와
$N_{\mathrm{Pu240}},N_{\mathrm{Pu241}}$로 rod burn-path를 설명한다. 이에 따라
restart/shuffle의 `/rodded_burnup` 데이터셋도 더 이상 쓰거나 읽지 않는다(과거
restart 파일에 있어도 무시).

### 6.5 Python 생성기에만 있는 키

`test/9-1_IISC/generate_iisc_case.py`가 내보내는 일부 키
(`history_unit_weighted`, `history_fraction_weighted`, `merge_rod_history_*`,
`history_directional_*`, `ctype_independent_spct`, `rhst spectral stages` 등)는
C++ Importer가 읽지 않는다(진단/manifest 전용). 코드 동작 기준으로 판단할 때는
무시해도 된다.

---

## 7. 런타임 상태 추적과 영속화

### 7.1 상태 갱신 (`UpdateBurnup`)

노드가 연소(`burn >= 1e-10`)하고 `UsesRodXS(l)`이면

$$
\texttt{\_history\_ctyp}[l] = \texttt{\_ctyp}[l], \qquad
\texttt{\_rodded\_fluence}[l] \mathrel{+}= \phi\,\Delta t\,f_{\mathrm{rod}}
$$

($f_{\mathrm{rod}} = \mathrm{clamp}(\text{rod fraction},0,1)$). fine-rod
fluence(`_fine_rod_fluence`)는 별도 fine mesh에서 누적된다.

### 7.2 restart / shuffle (`IO.cpp`)

저장/복원 데이터셋: `/history_ctype`, `/rodded_fluence`, `/fine_rod_fluence`.

- shuffle 재배치 시 `rodded_fluence`는 source 노드 평균이고, `history_ctype`은
  평균이 아니라 *첫 non-zero source* 값을 취한다.
- `fine_rod_fluence`는 shuffle 재배치 대상이 아니며 일반 restart로만 그대로 복원된다.
- (과거 `/rodded_burnup` 데이터셋은 제거되었다 — [§6.4](#64-rodded-burnup은-더-이상-추적하지-않는다).)

---

## 8. 전체 수식 요약

$$
\boxed{\;
\Sigma \;=\; \Sigma_{\mathrm{ref}}(B,c)
\;+\;\Delta\Sigma_{\mathrm{BPPM}}+\Delta\Sigma_{\mathrm{Tfuel}}+\Delta\Sigma_{\mathrm{Dmod}}
\;+\;\Delta\Sigma_{\mathrm{RDPL}}
\;+\;\Delta\Sigma_{\mathrm{IISC}}+\Delta\Sigma_{\mathrm{IISC\text{-}RHST}}
\;}
$$

- 모든 vector 항은 reference에서 0(zero anchor)이고 micx/lmpx만 보정한다.
- 적합 잔차는 *순차적으로* 정의된다(나중 항은 앞 항을 뺀 잔차에 적합).
- IISC indicator $\mathbf z_{\mathrm{IISC}}=[N_{\mathrm{Xe135}},N_{\mathrm{Pu239}},N_{\mathrm{U238}}/N_{\mathrm{Pu239}}]$,
  RHST indicator $\mathbf z_{\mathrm{RHST}}=[\widehat\Phi_{\mathrm{rod}},N_{\mathrm{Pu240}},N_{\mathrm{Pu241}}]$.

---

## 9. 코드 맵

| 항목 | 위치 |
|---|---|
| kind/coord enum, `HistoryDeltaCorrection`, `CorrectionComponent` | [`Model.h`](../include/chiffon/Model.h) |
| `EvalVectorTerm*`, `FitVectorTermCoefficients`, `FitRodDepletionDeltas` | [`Interpolator.h`](../include/chiffon/Interpolator.h) |
| `AddCurrentSpectralFitPoint`, `AddRhstCurrentSpectralFitPoint`, `Interpolate` | [`Interpolator.h`](../include/chiffon/Interpolator.h) |
| `PointDensity`(합성 좌표 적합 평가) | [`Interpolator.h`](../include/chiffon/Interpolator.h) |
| HGC 태깅, ctype 매핑, fluence 재구성, settings 파싱 | [`Importer.h`](../include/chiffon/Importer.h) |
| `branch_deltas`/`history_deltas`/`rod_depletion_delt` 직렬화 | [`Exporter.h`](../include/chiffon/Exporter.h) |
| `ApplyBranchDeltasToNode`, `FillRodNodeXS`, `ApplyHistoryDeltasToNode` | [`XSSet.cpp`](../src/XSSet.cpp) |
| `histCur`/`histRef`, `FineRodFluenceAverage` | [`XSSet.cpp`](../src/XSSet.cpp) |
| `UpdateBurnup`(상태 추적) | [`XSSet.cpp`](../src/XSSet.cpp) |
| restart/shuffle 영속화 | [`IO.cpp`](../src/IO.cpp) |
| 기준 입력/옵션 생성기 | [`generate_iisc_case.py`](../test/9-1_IISC/generate_iisc_case.py) |
