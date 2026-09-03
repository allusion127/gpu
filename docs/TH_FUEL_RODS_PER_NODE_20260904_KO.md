# 노드당 연료봉 수(연료온도 선출력 분모) — 2026-09-04

## 결함
`XSSet::SolveTH`의 연료온도 선출력밀도는 `lpd = 1000 * P_node / (RODS * hz[k])`이며,
`RODS`가 리터럴 `62.0`으로 **세 곳**에 하드코딩되어 있었다(CPU 기준선 e76d40d에서 그대로 상속).

- `src/XSSet.cpp:6378` (호스트 본체)
- `src/ThKernel.h:436` (GPU 이식 본체)
- `src/ThReference.cpp:209` (form-mask 채굴용 축자 인용)

## 근거
| 노형 | 격자 | 연료봉/집합체 | ndivxy | 노드당 |
|---|---|---|---|---|
| APR1400 | 16×16 | 236 | 2 | **59** |
| i-SMR | 17×17 | 260 (MASTER depf `npin, nfrod (289-29)`) | 2 | **65** |

i-SMR 측정: RASBERY/MASTER 연료온도 **상승분** 비 1.047–1.062 ≈ 65/62 = 1.048.
→ tfavg 약 **+9 K** 편향(≈ −25 pcm). APR1400은 반대 방향(59/62 = 0.952)이다.

## 구현 (기본값은 B0 보존)
- `geometry.dimensions.nfrod` (별칭 `fuel rods per assembly`, `fuel_rods_per_assembly`) 파싱,
  `rods_per_node = nfrod / ndivxy²`. `Geometry::Initialize`에서 **한 번만** 해석하고
  `[RASBERY][TH][NFROD]` 영수증(값·source·deck_nfrod·ndivxy)을 찍는다.
- 환경변수 `RASBERY_TH_FUEL_RODS=legacy|deck|<숫자>`. **기본 `legacy`(=62)**,
  따라서 digest `1f36e75dc00ed2b4`/4377 outers는 바이트 동일하게 유지된다.
  `deck`인데 덱에 `nfrod`가 없으면 **거부**한다(npins²−안내관 추정은 하지 않는다).
- 리터럴 `62.0`은 `src/ThFuelRods.h::kLegacyFuelRodsPerNode` 한 곳에만 남는다
  (`tools/test_th_fuel_rods_contract.py`가 강제).
- 케이스 키: payload에 `th_fuel_rods`(유효 값) 한 줄 추가 → `kSchema` v2→v3,
  **모든 케이스 키가 한 번 이동**한다(캐시는 재키잉 필요). source는 접기지 않고
  `[RASBERY][CASE]`(schema_version 9)에만 인쇄된다.

## 별건 미해결 항목 — tf.csv의 연소도 의존성이 너무 약하다
KNGR(`docs/kngr_v2_vs_master.csv`) tfavg Δ는 BOC **−14.6 K**에서 EOC **+71 K**로 단조 증가한다.
분모 상수 하나로는 부호가 바뀌는 이 추세를 설명할 수 없다 — Tfuel 테이블의 연소도 축 자체가
따로 조사되어야 한다. 본 커밋은 그 항목을 **건드리지 않는다**.

## 238 확인 레시피
1. i-SMR CY01 statepoint 0을 `RASBERY_TH_FUEL_RODS=legacy` / `=deck`(nfrod=260) 두 팔로 실행.
   기대: `delta_tfavg` 6.2 K → ~0, keff는 약 +25 pcm 이동.
2. KNGR은 **deck 팔 단일 실행**으로 Gate A(keff/ppm) · Gate B(핀 RMS) 재확인.
3. 두 팔의 `[RASBERY][TH][NFROD]`와 `[RASBERY][CASE].th_fuel_rods`가 서로 다른지 확인.

분류: **A2 / N1** — 기본 팔은 산술이 불변(A2), deck 팔은 궤적을 바꾸므로 새 기준선이 필요(N1).
