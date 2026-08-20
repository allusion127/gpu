# History 보정 엔진 단순화 (IISC / RHST / RDPL)

실험을 위해 난립했던 history 보정 프레임워크를 **최종 방법론 세 항(IISC, IISC-RHST, RDPL)만**
남기고 정리했다. 적용 방식은 BPPM/Tfuel/Dmod branch처럼 고정된 경로이고, 입력은 CSV 없이 HGC만
넣으면 되도록 단순화했다. 최종 방법론은 `test/9-1_IISC/docs/iisc_rhst_methodology_detailed_ko.md`
기준 그대로다.

$$
\Delta\Sigma = \Delta\Sigma_{\mathrm{BPPM}} + \Delta\Sigma_{\mathrm{Tfuel}} + \Delta\Sigma_{\mathrm{Dmod}}
             + \Delta\Sigma_{\mathrm{RDPL}} + \Delta\Sigma_{\mathrm{IISC}} + \Delta\Sigma_{\mathrm{IISC\text{-}RHST}}
$$

## 1. Stage A — 실험용 기계장치 제거 (동작 100% 동일)

엔진이 실제로 실행하는 보정 경로를 IISC + RHST + RDPL로 고정하고, 나머지 실험 변종을 전부 제거했다.
모든 단계는 기존 라이브러리(`iisc_rhst_best`/`spectral_full`/`traditional`) 출력과 **bit-identical**임을
회귀로 확인하며 진행했다. 총 약 **−1,385줄**.

| 파일 | 변화 | 제거/단순화 내용 |
|---|---:|---|
| `Interpolator.h` | 1973→1298 | fit 디스패처를 IISC(`CTYPE_INDEP_VEC`)+RHST(`RHST_UNIT`)+RDPL로 통일. `AddRhstFluenceWeighted/BurnupWeighted/TrajectoryBasis/DirectionalLinear/Offset/ScalarState` 등 fit 함수 7개와 merge/pair-key/stages/trajectory-ref 로직 제거. `StoredHistoryX`/`HistoryDeltaCTypeForPoint`/`SubtractPriorHistoryDeltas`를 IISC-only prior 경로로 축약. |
| `XSSet.cpp` | 3370→2968 | 런타임 312줄 `ApplyHistoryDeltasToNode` kind 분기를 ~78줄로 축소 (IISC=ctype0 surface, RHST=current-ctype surface; `rod_fluence`만 synthetic 좌표). |
| `Model.h` | 1367→1246 | `IsHistory*` predicate 9개, `hvBurnMemWeight`, 실험용 `Hk` kind 10개(STATE/RHST_OFFSET/RHST_FRAC/RHST_FLU/RHST_BURN*/RHST_TRAJ/RHST_DIR*) 제거. `VEC`/`RHST_UNIT`/`CTYPE_INDEP_VEC`만 유지. |
| `Importer.h` | — | `STATE_SETTINGS`/`RHST_SETTINGS` 구조체와 모든 실험용 SPCT 플래그(merge_*/history_*_weighted/directional/burnup-weight-shape) 파싱 제거. |

## 2. Stage B — 입력 단순화 + HGC에서 fluence 재구성 (의도된 ~수% 변화)

### 2.1 rod fluence를 HGC flux × time으로 재구성
기존에는 rod fluence를 외부 CSV(`rod_fluence_*.csv`, lattice summary의 NRN)로 따로 받아
CHIFFON 입력이 복잡했다. 이제는 HGC가 이미 가진 flux와 burnup으로 직접 적분한다.

$$
\mathrm{EFPD}_i = \frac{B_i \cdot 1000}{p_i}\,[\mathrm{d}], \qquad
\Phi_i = \sum_{j\le i} \tfrac{1}{2}\,(\phi_{j-1}+\phi_j)\,(\mathrm{EFPD}_j-\mathrm{EFPD}_{j-1})\cdot 86400
$$

- $\phi$: HGC `%FLUX`의 fast+thermal 평균 flux 합, $B$: burnup, $p$: power density(`AD_PDEN`).
- 검증: 재구성값이 summary NRN과 ~수% 이내로 일치 (예: PLEU_ROD 0.1 BU에서 6.85e19 vs NRN 6.68e19).
- 의미 분리는 그대로 유지: `_fuel_rod_fluence`(연료 노출)는 항상 flux×time, `_rod_fluence`(rod 물질
  fluence)는 nondepleted counterfactual(신품 흡수체)에서만 0. RDPL의 fluence-0 anchor는 코드에
  하드코딩되어 있어 nondepleted reference의 fluence 값과 무관하다.

### 2.2 입력 스키마 단순화 (CSV 제거)
이제 사용자는 카테고리별로 **HGC만** 넣으면 된다. CSV·중첩 fluence 설정 불필요.

```jsonc
"settings": {
  "spectral":      { "isotopes": ["Xe135","Pu239"], "isotope ratio": [["U238","Pu239"]] },  // IISC
  "rhst spectral": { "isotopes": ["rod_fluence","Pu240","Pu241"], "pre_remove": ["dmod","rod_depletion"] },  // RHST (rod_fluence 필수)
  "rod depletion": { "apply": true }                                                          // RDPL
},
"fuels": { "W1": {
  "rod history": {
    "CR1":              { "main":  "PLEU_ROD_0101.HGC" },
    "CR1_T560":         { "ctype": 1, "extra": ["PLEU_ROD_560_0101.HGC"] },
    "CR1_non_depleted": { "ctype": 1, "extra": ["PLEU_RODNONDEPL_0101.HGC"], "nondepleted": true }
  },
  "rod depletion": { "reference": "PLEU_RODNONDEPL_0101.HGC", "depleted": "PLEU_ROD_0101.HGC" }
}}
```

- **IISC / RHST의 isotope 벡터는 자유롭게 선택 가능**(개수·종류). 단 RHST에는 `rod_fluence`가 필수다.
- nondepleted counterfactual은 `"nondepleted": true`로 표시(또는 항목 이름에 `_non_depleted` 포함).
- 동작하는 예시: `test/9-1_IISC/cross_sections/iisc_rhst_best_simple.json` (CSV 0개, 기존 라이브러리와
  bit-identical하게 빌드됨).

### 2.3 HDF 라이브러리
포맷이 단순화되어 기존 `.h5`는 재생성해야 한다(요청대로 깨끗한 교체). 위 입력으로 생성하면 된다.

## 3. 검증

- **traditional / spectral_full**: solver 출력 정리 전과 **bit-identical** (fluence 비의존 → IISC/branch
  경로 무결성 확인).
- **iisc_rhst_best**: `test/9-1_IISC` 방법론 acceptance 통과.

| primary scenario | Max Δk [pcm] (재구성 fluence) | 문서 기준 |
|---|---:|---:|
| rod_to_unrod | 95.05 (acceptance <100) | 94.5 |
| unrod_to_rod | 28.19 | — |
| rod_depletion | 99.67 | 126 |
| rodded_T550 | 142.14 | 135 |
| rodded_P75 | 97.53 | 133 |

primary worst **142.14 < 150**, rod_to_unrod **95.05 < 100** → 두 acceptance 조건 모두 만족.
CSV→HGC fluence 전환의 영향은 acceptance 한도 안의 ~수% 수준이다.

## 4. 코드 대응 (갱신)

| 코드 영역 | 역할 |
|---|---|
| `Importer::ReconstructFluenceFromHGC` | HGC flux×time으로 rod fluence 재구성 (CSV 대체) |
| `Importer::AppendHGCPoints` (`nondepleted` 인자) | RHST/SPCT HGC 적재 + fluence 배정 |
| `Importer::AppendRodDepletionPairHGC` | RDPL reference(신품)/depleted pair 적재 |
| `Interpolator::Interpolate` | IISC + RHST_UNIT + RDPL 고정 fit |
| `Interpolator::FitRodDepletionDeltas` | RDPL: nondepl→depleted delta vs fluence (anchor 0) |
| `XSSet::ApplyHistoryDeltasToNode` | 런타임 IISC/RHST 적용 (RDPL은 `FillRodNodeXS`) |
| `Chiffon::Hk` (`VEC`/`RHST_UNIT`/`CTYPE_INDEP_VEC`) | 남은 보정 kind 태그 |
