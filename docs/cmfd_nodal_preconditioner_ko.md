# CMFD · Nodal · Preconditioner 개념 정리 (미니코어 예시)

이 문서는 RASBERY의 노심 계산 핵심 3요소 — **CMFD**, **Nodal**, **Preconditioner** — 가 각각 무엇이고 코드의 어디에 해당하는지를, 아주 작은 **2×2 미니코어**를 예시로 행렬을 직접 써가며 설명합니다.

---

## 0. 무엇을 푸는가 — 중성자 확산 고유치 문제

정상상태 노심은 다군 중성자 확산 고유치 방정식을 만족합니다:

$$
-\nabla\cdot D_g\nabla\phi_g \;+\; \Sigma_{r,g}\phi_g
\;=\; \sum_{g'}\Sigma_{s,g'\to g}\phi_{g'} \;+\; \frac{1}{k}\,\chi_g\sum_{g'}\nu\Sigma_{f,g'}\phi_{g'}
$$

행렬로 쓰면

$$
\mathbf{M}\,\boldsymbol{\phi} \;=\; \frac{1}{k}\,\mathbf{F}\,\boldsymbol{\phi}
$$

- $\mathbf{M}$ : 손실(누설+제거−산란) 연산자
- $\mathbf{F}$ : 핵분열 생성 연산자
- $k$ : 고유치 = $k_\text{eff}$, $\boldsymbol{\phi}$ : 중성자속(고유벡터)

이걸 직접 풀기엔 너무 크고 비싸므로, **싼 저차(CMFD) 풀이 + 정확한 고차(Nodal) 보정**을 번갈아 돌립니다.

---

## 1. CMFD (Coarse-Mesh Finite Difference)

### 개념
노드(coarse cell)들 사이의 중성자 **흐름(net current)** 을 유한차분으로 근사합니다. 인접한 노드 $l,m$의 경계면을 지나는 net current는

$$
J_{l\to m} \;=\; -\underbrace{\widetilde{D}_{l,m}}_{\text{dtil}}(\phi_m-\phi_l)\;-\;\underbrace{\widehat{D}_{l,m}}_{\text{dhat}}(\phi_m+\phi_l)
$$

- $\widetilde{D}$ (**dtil**) : 확산계수·격자크기로 정해지는 **선형 FD 결합** (코드: `CMFD::upddtil`)
- $\widehat{D}$ (**dhat**) : Nodal이 주는 **비선형 보정**(CNCC). 처음엔 0 (코드: `CMFD::upddhat`)

각 노드에서 균형식(누설+제거 = 산란+분열)을 세우면 **7-point 스텐실**(3D) 행렬 $\mathbf{A}$가 만들어집니다. 코드에서는 `BICGCMFD::setls`가 $\mathbf{A}$의 대각(`diag`)·결합(`cc`)을 조립합니다.

### 미니코어 2×2 (4노드, 1군)

```
 ┌─────┬─────┐
 │  2  │  3  │      인접: 0–1, 2–3 (x방향)
 ├─────┼─────┤            0–2, 1–3 (y방향)
 │  0  │  1  │      바깥은 경계조건(누설)
 └─────┴─────┘
```

노드 $l$의 결합계수를 $c_{lm}=\widetilde{D}_{lm}A_{lm}$ (면적 포함)이라 하면, 손실행렬은

$$
\mathbf{A} \;=\;
\begin{bmatrix}
 a_{0} & -c_{01} & -c_{02} & 0 \\
 -c_{01} & a_{1} & 0 & -c_{13} \\
 -c_{02} & 0 & a_{2} & -c_{23} \\
 0 & -c_{13} & -c_{23} & a_{3}
\end{bmatrix},
\qquad
a_{l} = \Sigma_{r,l}V_l + \!\!\sum_{m\in\text{nbr}(l)}\!\! c_{lm} + (\text{경계누설})
$$

핵분열은 (1군이면) 대각행렬:

$$
\mathbf{F} = \mathrm{diag}(f_0,f_1,f_2,f_3),\qquad f_l=\nu\Sigma_{f,l}V_l
$$

성질: **희소(sparse)**, **대각우세(diagonally dominant)**, dhat=0이면 **대칭**.

### 고유치 외부 반복 (Wielandt/power)
분열원 $\boldsymbol{\psi}=\mathbf{F}\boldsymbol{\phi}$를 알면, 매 반복마다 **선형계**

$$
\mathbf{A}\,\boldsymbol{\phi} \;=\; \frac{1}{k}\,\boldsymbol{\psi} \;\equiv\; \mathbf{b}
$$

를 풀고 $k$를 갱신합니다 (코드 `BICGCMFD::drive` + `wiel`, Wielandt shift `_eshift=0.04`). **이 $\mathbf{A}\boldsymbol{\phi}=\mathbf{b}$ 선형계를 푸는 게 BiCGSTAB**이고, 거기서 preconditioner가 쓰입니다.

---

## 2. Nodal (Analytic Nodal Method, ANM)

### 개념
CMFD의 dtil은 "노드 중심 사이에서 flux가 **직선**으로 변한다"는 거친 가정입니다. 실제 flux는 지수/쌍곡선 모양입니다. **Nodal**은 노드 내부에서 가로방향 누설(transverse leakage)을 적분한 1D 확산방정식을 **해석적으로** 풀어 더 정확한 노드평균 flux와 **경계면 current** $J^\text{nodal}$를 얻습니다 (코드: `Nodal::drive` → `updateConstant`/`calculateEven`/`calculateJnet2n`; `updateConstant`의 $\sinh/\cosh$가 해석해의 핵심).

### dtil ↔ dhat 연결 (CNCC)
Nodal이 준 정확한 current $J^\text{nodal}$를 CMFD가 **같은 flux에서 재현**하도록 dhat을 정의합니다:

$$
\widehat{D}_{l,m} \;=\; \frac{-\,J^\text{nodal}_{l,m} \;-\; \widetilde{D}_{l,m}(\phi_m-\phi_l)}{\phi_m+\phi_l}
$$

이 dhat을 $\mathbf{A}$의 대각·비대각에 더하고(코드 `upddhat`) CMFD를 다시 풉니다.

### 외부 반복 = CMFD ↔ Nodal
```
   [CMFD 선형계 A φ = b 풀기 (BiCGSTAB)]  ← 빠른 저차 해
            │  flux/current 갱신
            ▼
   [Nodal: 해석적 current J_nodal 계산]    ← 정확한 고차 해
            │  dhat 갱신 → A 수정
            ▼
   (수렴할 때까지 반복; 코드 Driver::SolveLoop의 outer 루프)
```
즉 **CMFD = 실제로 역행렬을 푸는 빠른 가속기**, **Nodal = dhat으로 정확도를 끌어올리는 보정자**. 둘이 수렴하면 "Nodal 수준 정확도"를 "CMFD 수준 비용"으로 얻습니다.

---

## 3. Preconditioner (전처리자) — 여기서는 SSOR

### 왜 필요한가
$\mathbf{A}\boldsymbol{\phi}=\mathbf{b}$를 BiCGSTAB(Krylov 반복법)으로 풉니다. Krylov의 수렴 속도는 $\mathbf{A}$의 **조건수**에 좌우됩니다. **전처리자** $\mathbf{M}\approx\mathbf{A}$ (단, $\mathbf{M}^{-1}$가 싸야 함)를 써서

$$
\mathbf{M}^{-1}\mathbf{A}\,\boldsymbol{\phi} \;=\; \mathbf{M}^{-1}\mathbf{b}
$$

로 바꾸면 $\mathbf{M}^{-1}\mathbf{A}\approx\mathbf{I}$라 훨씬 빨리 수렴합니다. 매 BiCGSTAB 반복에서 $z=\mathbf{M}^{-1}r$를 한두 번 적용합니다.

### 여기서의 전처리자 = SSOR
RASBERY는 **SSOR(Symmetric Successive Over-Relaxation)** 전처리자를 씁니다. $\mathbf{A}=\mathbf{D}+\mathbf{L}+\mathbf{U}$ (대각 / 하삼각 / 상삼각)로 쪼개고

$$
\boxed{\;\mathbf{M}_\text{SSOR} \;=\; (\mathbf{D}+\mathbf{L})\,\mathbf{D}^{-1}\,(\mathbf{D}+\mathbf{U})\;}
$$

$z=\mathbf{M}^{-1}r$ 적용은 **두 번의 삼각 sweep**입니다 — 이게 코드의 **`BICGSolver::minv`** 입니다:

1. **전방 sweep**: $(\mathbf{D}+\mathbf{L})\,y=r$ 를 전방대입으로 풀기
   $$ y_l \;=\; \mathbf{D}_l^{-1}\Big(r_l-\!\!\sum_{m<l}\! c_{lm}\,y_m\Big) $$
2. **후방 sweep**: $(\mathbf{D}+\mathbf{U})\,z=\mathbf{D}\,y$ 를 후방대입으로 풀기
   $$ z_l \;=\; \mathbf{D}_l^{-1}\Big(\mathbf{D}_l\,y_l-\!\!\sum_{m>l}\! c_{lm}\,z_m\Big) $$

### 미니코어 2×2로 보는 SSOR

$$
\mathbf{D}=\begin{bmatrix}a_0&&&\\&a_1&&\\&&a_2&\\&&&a_3\end{bmatrix},\quad
\mathbf{L}=\begin{bmatrix}0&&&\\-c_{01}&0&&\\-c_{02}&0&0&\\0&-c_{13}&-c_{23}&0\end{bmatrix},\quad
\mathbf{U}=\mathbf{L}^{\mathsf T}
$$

전방대입 $(\mathbf{D}+\mathbf{L})y=r$:

$$
\begin{aligned}
y_0 &= r_0/a_0\\
y_1 &= (r_1 + c_{01}y_0)/a_1\\
y_2 &= (r_2 + c_{02}y_0)/a_2\\
y_3 &= (r_3 + c_{13}y_1 + c_{23}y_2)/a_3
\end{aligned}
$$

> **핵심**: $y_3$는 $y_1,y_2$를, $y_1$은 $y_0$를 필요로 합니다 → 노드를 **순서대로** 풀어야 하는 **본질적 직렬** 연산. 이것이 `minv`가 8-thread에서 병렬화되지 않는(SSOR가 serial bottleneck인) 이유입니다. (이번 세션에 red-black 2색 재정렬로 병렬화를 시도했으나, i-SMR이 작아 fork/join 오버헤드가 이득을 압도해 되돌렸습니다 — `perf_and_cusping_fixes_ko.md` §7.)

코드 매핑 (`BICGSolver::minv`):
```cpp
// 전방: tmp(l) = D^{-1} (b(l) - sum_{ln<l} cc(l,ln)*tmp(ln))
// 후방: x(l)   = D^{-1} (D*tmp(l) - sum_{ln>l} cc(l,ln)*x(ln))
```

---

## 4. 2군(2-group) 블록 구조 — 코드의 실제 모습

실제로는 노드마다 **2개 에너지군**이 있어, 각 $\phi_l$이 2-벡터이고 행렬의 각 "원소"가 **2×2 블록**입니다.

$$
\boldsymbol{\phi}_l=\begin{bmatrix}\phi_{l,1}\\ \phi_{l,2}\end{bmatrix},\qquad
\mathbf{D}_l=\begin{bmatrix}d_{11}&d_{12}\\ d_{21}&d_{22}\end{bmatrix}_l
$$

- **대각 블록 $\mathbf{D}_l$** (2×2): 노드 내 군간 결합(제거·산란·분열). 코드: `diag`(2×2), 그 역행렬 `dinv`(2×2, `dinv[l*4+0..3]`).
- **결합 블록** (이웃 노드): 확산 누설은 군 내에서만 일어나 사실상 군별 스칼라. 코드: `cc[l*12 + g*6 + idir*2 + lr]` (군 $g$, 방향 `idir`∈{x,y,z}, 좌우 `lr`).

그래서 `minv`의 전/후방 sweep은 노드별 **2×2 블록 전·후방대입**입니다:

$$
\mathbf{tmp}_l=\mathbf{D}_l^{-1}\Big(\mathbf{b}_l-\sum_{m<l}\mathbf{c}_{lm}\,\mathbf{tmp}_m\Big)\quad(\text{2-벡터, } \mathbf{D}_l^{-1}\text{는 2×2})
$$

---

## 5. 한눈에 보는 전체 흐름 (코드 대응)

```
Driver::SolveLoop  (스텝당 outer 반복)
│
├─ BICGCMFD::drive ────── CMFD 고유치 1스텝
│   ├─ setls         : A(=diag,cc) 조립  (dtil + dhat)
│   ├─ BICGSolver::solve : A φ = b  (BiCGSTAB)
│   │      └─ minv   : SSOR 전처리 (전방+후방 sweep)  ← Preconditioner
│   │      └─ axb    : 행렬-벡터곱 A·v
│   └─ wiel          : k_eff 갱신 (Wielandt shift)
│
├─ Nodal::drive ──────── 해석적 current J_nodal 계산   ← 고차 보정
│   └─ updateConstant(sinh/cosh) / calculateJnet2n
│
├─ upddhat           : dhat = (J_nodal − dtil·Δφ)/(Σφ)  → A 수정
│
├─ (cusping / 임계탐색 / TH 피드백 …)
│
└─ flux 수렴까지 반복
```

**요약**
- **CMFD** = 노드 사이를 유한차분으로 잇는 **희소 선형계**. 실제로 BiCGSTAB로 푸는 대상.
- **Nodal** = 노드 내부를 해석적으로 풀어 **dhat**(비선형 보정)을 제공 → CMFD를 Nodal 정확도로 끌어올림.
- **Preconditioner(SSOR)** = BiCGSTAB가 빨리 수렴하도록 $\mathbf{A}$를 근사 역변환($\mathbf{M}^{-1}$). 코드의 `minv`이며, 전/후방 삼각 sweep이라 본질적으로 직렬.
