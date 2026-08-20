> Legacy/keyed RHST note: this file records an earlier trajectory-keyed rod
> history correction discussion. The current keyless SPCT+RDPL+RHST method is
> documented in `test/9-1_IISC/docs/keyless_iisc_rhst_methodology_ko.md`.

## 1. Rationale
Micro depletion을 사용하는 현재 구조에서는 일반적인 의미의 spectral history, 즉 과거 spectrum이 isotope depletion을 통해 장기적으로 XS에 영향을 남기는 효과가 상당 부분 depletion solver 안에 이미 들어간다. 따라서 correction이 직접 잡아야 하는 것은 크게 두 가지로 보는 것이 자연스럽다.
1. 노심 운전 환경의 차이로부터 발생되는 spectrum의 차이가, micro XS에 주는 residual
2. Control rod가 삽입된 trajectory에서 연소되었기 때문에 main reference와 달라진 trajectory residual
그렇다면 실제로 residual은 어떤 핵종에서 얼마나 발생하는가? 다음과 같은 HGC 파일들을 바탕으로 80 MWD/kgHM 에서의 reference와 history branch 사이에서의 isotope density 차이를 비교해 보았다.
- main: `LEU_base_0101.HGC`, `REFERENCE`, ctype 0
- spectral history: `LEU_GT_560/600/P50_0101.HGC` 중 가장 변화가 큰 case 사용. 각각 Tmod=560K, 600K, P=50% 에서 depletion한 경우
- rod-history: `LEU_GT_ROD_0101.HGC`의 `CR1 REFERENCE` rodded depletion 후 rod를 뺀 branch

**Isotope Density 차이**

| burn | isotope     | spectral rel. | rod-history rel. |
| ---: | ----------- | ------------: | ---------------: |
|80|Pu239|+9.53%|+81.41%|
|80|U235|+7.99%|+52.58%|
|80|U238|-0.213%|-1.416%|
|80|Pu240|+3.83%|+26.12%|
|80|Pu242|-1.92%|-14.27%|
|80|U238/Pu239|8.90%|-45.66%|
|80|Pu239/U238|9.76%|+84.01%|
|80|U235/Pu239|1.41%|-15.89%|
|80|Xe135/Sm149|16.39%|-27.40%|

확인 결과, Pu239 그리고 U238과 Pu239의 ratio가 history effect에 의해서 매우 큰 영향을 받음을 알 수 있었으며, 그 외에 Pu240, Pu242 등도 영향을 받았다. 특히, Pu-239와 달리 U238과 Pu239의 비율은 spectral history에서는 큰 차이가 없었으나, rod가 관여할 경우 크게 차이가 나는 것을 알 수 있었다. (spectral history의 경우 positive 방향, rod-history의 경우 negative 방향) 따라서 Pu239 및 Pu239/U238 ratio가 유력한 후보로 채택되었다.
이에 더해, Xe135 역시 indicator로 채택되었는데, 이는 Xe135가 즉각적으로 power에 의한 영향을 반영할 수 있기 때문이며, Xe135는 그 자체로 흡수능이 매우 높은 독물질이므로 다른 원소에 미치는 영향을 고려해야 하기 때문이다.
최종적으로 Xe135, Pu239, U238/Pu239 조합은 서로 다른 물리를 나누어 담당한다.
- Xe135: instantaneous power / posion indicator
- Pu239: spectral indicator
- U238/Pu239: rod-history-sensitive indicator
이때 Isotope density는 RASBERY가 depletion module을 통해 별도로 계산하고, correction은 현재 계산된 isotope density coordinate를 이용해 XS residual만 보정한다.

## 2. Methodology
### 2.1 Coordinate definition
Spectral 과 rod history correction 모두 absolute isotope-density coordinate를 사용한다. Reference density와의 difference나 relative difference를 쓰지 않는다.
일반 isotope coordinate는 다음과 같다.
$$
x_i = N_i
$$
두 isotope으로 주어진 `isotope ratio`는 다음과 같이 정의한다.
$$
x_{A/B} = \frac{N_A}{N_B}
$$
분모가 너무 작을 때 ratio가 numerical noise를 증폭하지 않도록, 현재 구현은 $N_B < 10^{-10}$ 이면 해당 ratio coordinate를 0으로 둔다. 현재 correction에 사용하는 indicator vector는 다음과 같다.
Spectral correction:
$$
\mathbf{x}^{\mathrm{spec}}
= \left[
N_{\mathrm{Xe135}},
N_{\mathrm{Pu239}},
\frac{N_{\mathrm{U238}}}{N_{\mathrm{Pu239}}}
\right]
$$
Rod-history correction:
$$
\mathbf{x}^{\mathrm{rod}}
= \left[
N_{\mathrm{Pu239}},
\frac{N_{\mathrm{U238}}}{N_{\mathrm{Pu239}}}
\right]
$$
### 2.2 Total correction form
Runtime에서 node XS는 다음 형태로 계산된다.
$$
\sigma_{\mathrm{node}}
=
\sigma_{\mathrm{base}}(c, B)
+ \Delta\sigma_{\mathrm{branch}}(c, B, C_{B}, T_{f},\rho_{m})
+ \Delta\sigma_{\mathrm{spec}}(c, B; \mathbf{x}^{\mathrm{spec}})
+ \Delta\sigma_{\mathrm{rod}}(h, c, B_{\mathrm{rod}}; \mathbf{x}^{\mathrm{rod}})
$$
여기서
- $c$ : current control rod state
- $h$ : depletion trajectory control rod state
- $B$ : total burnup
- $B_{\mathrm{rod}}$: rodded burnup, 즉 rod가 삽입된 상태로 누적된 burnup
- $\sigma_{\mathrm{base}}(c, B)$: 현재 rod state의 main HGC reference XS
- $\Delta\sigma_{\mathrm{branch}}(c, B, C_{B}, T_{f},\rho_{m})$: boron, fuel temperature, mod density에 의한 correction
- $\Delta\sigma_{\mathrm{spec}}(c, B; \mathbf{x}^{\mathrm{spec}})$: Spectral history correction
- $\Delta\sigma_{\mathrm{rod}}(h, c, B_{\mathrm{rod}}; \mathbf{x}^{\mathrm{rod}})$: Rod history correction
여기서 $h$는 CR1 branch, CR2 branch 처럼 어떤 rod depletion trajectory에서 온 correction table을 사용할지 고르는 key를 의미한다.
### 2.3 Least-squares fitting
Spectral과 rod history correction은 모두 같은 linear least-squares 형태로 fitting한다. 하나의 burnup/ctype key에 대해 $n$개의 fitting point가 있다고 하자. 예를 들어 indicator로 $N_1$, $N_2$, $N_3/N_4$를 사용하면 correction model은
$$
\Delta\sigma
=
\beta_0
+
\beta_1 N_1
+
\beta_2 N_2
+
\beta_3 \frac{N_3}{N_4}
$$
이다. 여기서 $N_1,N_2,N_3,N_4$는 선택한 isotope density이다. 현재 spectral correction에서는 예를 들어 $N_1=N_{\mathrm{Xe135}}$, $N_2=N_{\mathrm{Pu239}}$, $N_3/N_4=N_{\mathrm{U238}}/N_{\mathrm{Pu239}}$에 해당한다.

Fitting을 위해 여러 개의 HGC 파일을 input으로 주었으므로, 각 fitting point마다 indicator 값이 존재한다. 이 값들을 행으로 모은 indicator matrix를 $\mathbf{X}$라고 쓰면
$$
\mathbf{X}
=
\begin{bmatrix}
1 & N_1^{(1)} & N_2^{(1)} & \left(N_3/N_4\right)^{(1)} \\
1 & N_1^{(2)} & N_2^{(2)} & \left(N_3/N_4\right)^{(2)} \\
\vdots & \vdots & \vdots & \vdots \\
1 & N_1^{(n)} & N_2^{(n)} & \left(N_3/N_4\right)^{(n)}
\end{bmatrix}
$$
로 표현할 수 있다. 첫 번째 column은 constant term $\beta_0$에 대응한다.

실제 XS residual은 group별, XS type별로 같은 indicator matrix에 대해 독립적으로 fitting된다. 예를 들어 absorption XS residual은
$$
\mathbf{y}_a
=
\begin{bmatrix}
\Delta\sigma_a^{(1)} \\
\Delta\sigma_a^{(2)} \\
\vdots \\
\Delta\sigma_a^{(n)}
\end{bmatrix}
$$
이고, fission XS residual과 nu-fission XS residual은 각각
$$
\mathbf{y}_f
=
\begin{bmatrix}
\Delta\sigma_f^{(1)} \\
\Delta\sigma_f^{(2)} \\
\vdots \\
\Delta\sigma_f^{(n)}
\end{bmatrix},
\qquad
\mathbf{y}_{\nu f}
=
\begin{bmatrix}
\Delta\nu\sigma_f^{(1)} \\
\Delta\nu\sigma_f^{(2)} \\
\vdots \\
\Delta\nu\sigma_f^{(n)}
\end{bmatrix}
$$
처럼 둔다. Scattering, transport, fission spectrum 등 다른 XS component도 같은 방식으로 같은 $\mathbf{X}$에 대해 독립적으로 fitting된다.

Absorption XS의 coefficient vector를
$$
\boldsymbol{\beta}_a
=
\begin{bmatrix}
\beta_{a,0} \\
\beta_{a,1} \\
\beta_{a,2} \\
\beta_{a,3}
\end{bmatrix}
$$
라고 하면,
$$
\mathbf{X}\boldsymbol{\beta}_a
\approx
\mathbf{y}_a
$$
를 least-squares sense로 푼다. Normal equation으로 쓰면 다음과 같다.
$$
\left(\mathbf{X}^T\mathbf{X}\right)\boldsymbol{\beta}_a
=
\mathbf{X}^T\mathbf{y}_a
$$
실제 구현에서는 각 column scale 차이가 큰 isotope density coordinate를 안정적으로 다루기 위해 column scaling을 적용한 뒤 선형 시스템을 푼다. Fitting된 coefficient는 `DeltaCrossSection` 형태로 저장되고, 실제 계산 시에는
$$
\Delta\sigma_a
=
\beta_{a,0}
+
\beta_{a,1}N_1
+
\beta_{a,2}N_2
+
\beta_{a,3}\frac{N_3}{N_4}
$$
를 평가한다. 다른 XS component도 같은 isotope indicator에 대해 자기 coefficient를 사용한다.

### 2.4 Spectral correction fitting
Spectral correction은 normal main-extra HGC set으로부터 fitting한다. Main reference point를 $\sigma(c,B)$, extra point 를 $\hat{\sigma}(c,B)$ 라고 하면 target residual은
$$
\mathbf{y}^{\mathrm{spec}}
=
\hat{\sigma}(c,B)
-
\sigma(c,B)
-
\sum_{q \, \in \, \mathrm{branch }}
\Delta\sigma_q(c,B)
$$
이다. 현재 입력에서는 `pre_remove = ["dmod"]`를 사용하므로, dmod branch로 이미 보정 가능한 residual을 먼저 제거한 뒤 fitting한다.
Linear model은 다음과 같다.
$$
\Delta\sigma_{\mathrm{spec}}
(c,B;\mathbf{x}^{\mathrm{spec}})
=
\boldsymbol{\beta}^{\mathrm{spec}}_0(c,B)
+
\sum_j
\boldsymbol{\beta}^{\mathrm{spec}}_j(c,B)
x_j^{\mathrm{spec}}
$$
여기서 coefficient key는 current ctype \(c\)에 의해 결정된다. 즉 `spectral` correction은 현재 rod state의 local spectrum/current-state residual을 담당하며, depletion trajectory \(h\)를 독립 변수로 갖지 않는다.

구현상 rodded trajectory HGC의 `extra`가 제공된 경우, \(h=c\ne0\)인 rodded-current 상태에서는 같은 current ctype \(c\)에 대한 rodded-basis spectral coefficient를 우선 사용할 수 있다. 하지만 이것은 `spectral` correction이 \(h\)에 의존한다는 뜻이 아니라, current state가 rodded일 때 사용할 더 적절한 local coefficient set을 고르는 것이다. Rod trajectory memory 자체는 별도의 `rod history` correction이 담당한다.

반대로 rodded depletion 후 rod-out이 된 경우에는

$$
h\ne0,\quad c=0
$$

이므로 spectral correction은 현재 state인 unrodded coefficient를 사용한다. 이 선택은 중요하다. Rod가 빠진 뒤의 instantaneous spectrum은 unrodded에 가깝고, rod trajectory memory는 별도의 `rod history` correction이 담당해야 하기 때문이다.

### 2.5 Rod-history correction fitting

`rod history` correction은 rodded depletion HGC set과 normal main HGC 사이의 residual을 fitting한다. Normal main reference를 \(m(c,B)\), rod-history reference를 \(r(h,c,B)\)라고 하자.

이때 trajectory offset target은

$$
\mathbf{y}^{\mathrm{rod}}
=
\sigma_r(h,c,B)
-
\sigma_m(c,B)
-
\sum_{q \in \mathrm{pre\_remove}}
\Delta\sigma_q(c,B)
-
\Delta\sigma_{\mathrm{spec}}(c,B;\mathbf{x}^{\mathrm{spec}}_r)
$$

이다.

즉 `rod history` correction은 main reference와 rod-history reference 사이의 remaining residual을 담당한다. 이 residual은 control rod insertion trajectory가 남긴 actinide/spectrum-memory effect를 나타낸다.

Fitting된 coefficient는 \((h,c,B)\) 또는 runtime에서는 \((h,c,B_{\mathrm{rod}})\) key로 저장된다. 따라서 \(h\)는 회귀식의 입력 feature가 아니라 coefficient family를 나누는 label이다. 예를 들어 `CR1`로 탄 history와 `CR2`로 탄 history가 동시에 주어지면, 두 trajectory는 서로 다른 rod worth와 spectrum perturbation을 만들 수 있으므로 같은 \(\mathbf{x}^{\mathrm{rod}}\) 값을 갖더라도 같은 coefficient를 공유하면 안 된다. 이 경우 \(h\)가 없으면 `CR1 -> unrodded`와 `CR2 -> unrodded` residual이 같은 table에 섞인다.

Linear model은 다음과 같다.

$$
\Delta\sigma_{\mathrm{rod}}
(h,c,B_{\mathrm{rod}};\mathbf{x}^{\mathrm{rod}})
=
\boldsymbol{\beta}^{\mathrm{rod}}_0(h,c,B_{\mathrm{rod}})
+
\sum_j
\boldsymbol{\beta}^{\mathrm{rod}}_j(h,c,B_{\mathrm{rod}})
x_j^{\mathrm{rod}}
$$

Runtime에서는 이 coefficient를 total burnup이 아니라 rodded burnup key로 lookup한다.

$$
\Delta\sigma_{\mathrm{rod}}
\left(
h,c,
\mathrm{round}(B_{\mathrm{rod}});
\mathbf{x}^{\mathrm{rod}}
\right)
$$

이는 80 burnup 상태라도 실제로 rod가 삽입된 상태로 탄 양이 40 burnup이면, rod-history correction은 40 burnup 부근의 rod trajectory data를 참조해야 한다는 뜻이다.

### 2.6 CHIFFON input schema

현재 CHIFFON input은 다음처럼 분리한다.

```json
{
  "settings": {
    "spectral": {
      "apply": true,
      "isotopes": ["Xe135", "Pu239"],
      "isotope ratio": [["U238", "Pu239"]],
      "pre_remove": ["dmod"]
    },
    "rod history": {
      "apply": true,
      "isotopes": ["U238", "Pu239"],
      "isotope ratio": [["U238", "Pu239"]],
      "pre_remove": ["dmod"]
    }
  }
}
```

HGC input도 normal main/extra와 rod-history main/extra를 분리한다.

```json
{
  "fuels": {
    "W1": {
      "main": "LEU_base_0101.HGC",
      "extra": [
        "LEU_GT_560_0101.HGC",
        "LEU_GT_600_0101.HGC",
        "LEU_GT_P50_0101.HGC"
      ],
      "rod history": {
        "CR1": {
          "main": "LEU_GT_ROD_0101.HGC",
          "extra": [
            "LEU_GT_ROD_560_0101.HGC",
            "LEU_GT_ROD_600_0101.HGC",
            "LEU_GT_ROD_P50_0101.HGC"
          ]
        }
      }
    }
  }
}
```

`rod history.main`은 rodded trajectory reference이고, `rod history.extra`는 같은 rodded trajectory 위에서 spectral/current coefficient를 fitting하기 위한 extra HGC이다.

## 3. Results

Benchmark는 rodded depletion 후 unrodded state로 전환되는 case를 포함한다. Error는
$$
\Delta k \times 10^5
=
(k_{\mathrm{calc}} - k_{\mathrm{ref}})\times10^5
$$
로 정의한다.

### 3.1 Rod-in depletion region

0--40 burnup 구간은 rod insertion 상태로 depletion되는 구간이다.

| Method | Max abs error (pcm) | Mean abs error (pcm) | RMS error (pcm) |
|---|---:|---:|---:|
| Traditional | 423 | 314.5 | 336.5 |
| Pu correction | 15 | 4.6 | 5.5 |
| Pu-Xe-U/Pu correction | 26 | 17.7 | 18.5 |

이 구간에서는 Pu-only correction도 이미 매우 잘 작동한다. Pu-Xe-U/Pu correction은 약간 더 보수적인 residual을 남기지만, 여전히 30 pcm 이내에 머문다.

### 3.2 Rodded-to-unrodded region

40--80 burnup 구간은 rodded depletion history를 가진 상태에서 rod가 빠진 unrodded current state를 계산하는 구간이다. 이 영역이 기존 방법론에서 가장 어려웠다.

| Method | Max abs error (pcm) | Mean abs error (pcm) | RMS error (pcm) |
|---|---:|---:|---:|
| Traditional | 1003 | 510.2 | 541.1 |
| Pu correction | 689 | 369.9 | 421.3 |
| Pu-Xe-U/Pu correction | 205 | 90.8 | 106.4 |

Traditional method는 rod-out 직후 1000 pcm 수준의 error가 발생한다. Pu-only correction은 일부 개선되지만, rod trajectory effect와 instantaneous current effect를 충분히 분리하지 못해 최대 689 pcm까지 남는다.

Pu-Xe-U/Pu correction은 rodded-to-unrodded 구간에서 max abs error를 205 pcm 수준으로 낮춘다. Mean abs error도 Traditional 대비 약

$$
\frac{510.2 - 90.8}{510.2} \approx 82.2\%
$$

감소한다. Pu-only 대비로도

$$
\frac{369.9 - 90.8}{369.9} \approx 75.5\%
$$

감소한다.

### 3.3 Interpretation

개선의 핵심은 두 correction을 분리한 점이다.

```text
spectral correction:
  current state의 isotope/spectrum residual 보정

rod history correction:
  rodded trajectory와 normal main trajectory 사이의 residual 보정
```

Rodded-to-unrodded case에서 current state는 unrodded이므로 spectral coefficient는 unrodded coefficient를 쓰는 것이 자연스럽다. 반면 rod가 삽입된 상태로 누적된 burnup memory는 `rod history` correction이 담당하며, 이때 burnup key는 total burnup이 아니라 `rodded_burn`이다.

따라서 최종 형태는 단순한 empirical fitting이라기보다, 다음 세 물리량을 분리해 반영한 correction이다.

1. Current spectrum/poison state: `Xe135`
2. Actinide buildup: `Pu239`
3. Rod-history-sensitive conversion state: `U238/Pu239`

이 분리가 rodded-to-unrodded case에서 기존 방법론 대비 큰 개선을 만든다.
