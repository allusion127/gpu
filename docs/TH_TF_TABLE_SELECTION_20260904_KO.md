# 연료온도 테이블 선택 (WH / CE) — 2026-09-04

## 발견
`include/Database/tf.csv`(9 LPD × 10 연소도, ΔT = Tfuel − Tmod)는 MASTER **내장 WH형**
테이블(`isolth = 11`)이다 — 좌상단 모서리가 MASTER-3.0 매뉴얼의 `%DEF_TFT` 예제와 일치한다.
읽기: `XSSet::LoadTHTables` → `XSSet::GetTfuel`(`src/XSSet.h:516-526`, 사용 `src/XSSet.cpp:6378-6381`),
GPU 미러 `thGetTfuel`(`src/ThKernel.h`) 및 `src/ThReference`.

그러나 APR1400/KNGR 덱은 `isolth = 12`(**ABB-CE**)이고, 상승분의 **연소도 기울기가 약 5배 가파르다**:
한 주기 동안 MASTER 상승분은 **−29.6 %**, 본 테이블은 **−5.8 %**.
→ KNGR tfavg가 BOC **−14.6 K** → EOC **+71 K**로 흐르고, 이를 붕소 탐색이 흡수해
Gate B 잔차 **−15.3 ppm**으로 나타난다. (`docs/TH_FUEL_RODS_PER_NODE_20260904_KO.md` 미해결 항목의 답.)

## 구현 (기본값은 B0 보존)
- **덱 키** `"tf table"`(별칭 `tf_table`, `tfuel table`, `tfuel_table`) —
  `geometry.dimensions` **또는** `"default parameters"` 중 **한 곳**에만.
  값: `"wh"` | `"ce"` | `"file:<경로>"` | `%DEF_TFT`형 인라인 `{lpd:[…], bu:[…], dt:[[…]]}`.
  두 곳 모두 선언하면 **거부**한다(패배한 선언이 보이지 않으므로).
- **환경변수** `RASBERY_TH_TF_TABLE=legacy|deck|wh|ce|<경로>`. **기본 `legacy`**.
  `legacy`는 `wh`와 **같은 정체성**으로 해석되므로(tf.csv가 곧 WH 테이블),
  digest `1f36e75dc00ed2b4`/4377 outers는 바이트 동일하게 유지된다.
  `deck`인데 덱이 아무것도 선언하지 않으면 **거부**한다(추정하지 않는다).
- **경로**: `Geometry::Initialize`가 요청만 한 번 해석(`_tf_choice`) → `th::loadTfTable`이
  프로세스당 정체성별 **한 번** 로드/digest/영수증. CPU(`XSSet::_tf_table`) → GPU
  (`thgpu::TableView`는 `_tf_table`에서 채워짐) → `ThReference`가 모두 그 한 테이블을 본다.
  T/H form-mask 픽스처는 **자체 합성 테이블**을 쓰므로 선택과 무관하다(음성 대조로 고정).
- **영수증** `[RASBERY][TH][TFTABLE] {source, name, path, sha256, nlpd, nbu}`.
- **케이스 키**: payload에 `th_tf_table = <name>:<sha256>` 한 줄 추가 → `kSchema` **v3→v4**,
  `[RASBERY][CASE]` **schema_version 9→10**. **모든 케이스 키가 한 번 더 이동**한다(캐시 재키잉 필요).
  접히는 것은 **정체성(내용 digest)**이고 `source`는 인쇄만 된다.
- **`ce`는 아직 거부한다**: `include/Database/tf_ce.csv`는 트리에 **없고 추정본도 넣지 않았다**.
  요청 시 "CE table not yet regressed; run tools/fit_tf_table.py"로 크게 실패한다.

## `tools/fit_tf_table.py`
입력 CSV(헤더 필수, 별칭 허용): `efpd`, `node|plane|l|k`, `burnup|bu`[GWd/t],
`lpd`[W/cm] 또는 `power`+`hz`+`rods`, `tfuel|tf`[K], `tmod|tm`[K] — 181 러너가 MASTER
`$TF`/`$TM` 편집에서 생성한다. `--lpd-from power,hz,rods`는 SolveTH 철자 그대로
`lpd = 1000·P_node/(rods·hz)`를 계산한다(`hz`·`rods`는 상수도 가능).
적합: 미지수는 격자값 자체, 표본은 `milk::Table::Get`과 **동일한 이중선형 가중**(클램프 포함)으로
들어가고, 양방향 2차 차분(곡률) 벌점 + LPD 증가 단조 벌점(IRLS)을 건다.
셀별 커버리지·잔차 RMS/최대를 보고하고, **데이터가 없는 격자점은 `--extrapolate` 없이는 거부**한다.
`--self-test`: 알려진 CE형 테이블에서 표본을 생성해 재복원(현재 최악 knot 오차 **0.035 K** ≤ 0.5 K).

## 정확성 분류
기본 팔은 산술 불변 — **A2**. `deck`/`ce` 팔은 궤적을 바꾸므로 새 기준선 — **N1**.
합격 기준: 전 statepoint `|Δtfavg| ≤ 5 K`, `max|Δppm| ≤ 5`, `|Δfqp| ≤ 0.02`.

## 238 레시피 (tf_ce.csv가 생긴 뒤)
1. `tools/fit_tf_table.py <MASTER 편집>.csv --lpd-from power,hz,59 -o include/Database/tf_ce.csv`
   — 커버리지 표와 잔차를 커밋 메시지에 남긴다.
2. KNGR 단일 실행 2팔: `legacy` vs `deck`(덱에 `nfrod: 236` + `"tf table": "ce"`).
3. Gate A(keff/ppm)·Gate B(핀 RMS) + `docs/kngr_v2_vs_master.csv` 대비 tfavg 사다리.
4. 두 팔의 `[RASBERY][TH][TFTABLE]`와 `[RASBERY][CASE].th_tf_table`이 다른지 확인.
