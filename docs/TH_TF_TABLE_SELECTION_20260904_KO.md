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

---

## CE 표 회귀 결과 (2026-09-04)

`include/Database/tf_ce.csv` **생성 완료**. 소스: 181의 MASTER KNGR cycle-1 실행
`CodexBench\kngr_tf_edit_20260904\` (`isolth=12`, `%GEN_PIN 16 236`, `%GEN_THD 3983.0 MW`,
19×19×27, 연료평면 25×15.24 cm, 241 FA, 1093.15 EFPD / 노심평균 42.52 MWd/kgU까지 연소).
MAS_SUM sha256 `665f3af3e3831b33be479f3caaf42c5e7844d00ea7a609ca3cc21d85a9440ce3` —
표의 **모서리 셀**에 그대로 기록했다(아래 "출처 줄").

### MAS_RST에는 온도가 없다 (계획 변경의 근거)
매뉴얼 4.2절은 Record 23/24/25를 `nt*(R)` Tm/Dm/Tf로 광고하지만 **이 빌드는 쓰지 않는다.**
`tools/parse_master_rst.py`가 실제 레이아웃을 역공학해 문서화했다(리틀엔디안 Fortran
unformatted sequential, **REAL*8**, 마커 head==tail 175,325 레코드 전부 일치):

```
[  0] 256B × 115   MAS_INP 덱 에코(한 줄 = 한 레코드)
[115]   4B × 241   연료 FA 배치 ID (A4), %LPD_BCH row-major
[356]   8B, 56B(7×R8 노심 스칼라: tin/mflow/pload/boron), 12B
[359] FA 블록 × 241, 각 726 레코드
        +0                4B  FA 순번 마커
        평면군 × 25 (평면1 = 최하단 연료평면 = 전역 축방향 노드 2), 각 29 레코드:
          4 × [ 56B = 7×R8 : 노드 연소도 + 표면연소도 6 ]
              각 뒤에 [264B = 33×R8 핵종 수밀도] [40B = 5×R8 — **항상 0**]
          1 × [   4B = npin(16) ]
         16 × [ 128B = 16×R8 : 그 평면의 2-D 핀 연소도 ]
```

근거: (a) `nt*4`(38,988 B)·`nt*8`(77,976 B) 길이의 레코드가 **없다**; (b) 파일 전체 float64 중
[280,1500] 구간에 드는 값은 **3개뿐**이고 모두 노심 스칼라 레코드(tin 290.6, mflow 3480,
boron 3773.48)다 — 0/360/1080 EFPD 파일 모두 동일; (c) 온도가 들어갈 유일한 노드별 슬롯
(5×R8)이 **모든 노드·모든 statepoint에서 정확히 0**이다. 노드 **출력**도 없다(Part 3의
축방향 노드 출력·2-D 핀 출력 미기록; 존재하는 16×16 배열은 연소도다).
→ **3-D 노드 단위 회귀는 불가능**하다.

**검증**(`--validate MAS_OUT`): restart의 3-D 연소도를 두 방향으로 접어 MAS_OUT과 대조한다.

- 반경: 241 FA 축방향 평균 vs `$B2D` → **max |ΔBu| = 0.0005 MWd/kgU**
- 축방향: 25 평면 반경 평균 vs `$B1D` → **max |ΔBu| = 0.0001 MWd/kgU**
- 노드 평균 연소도 14.003 / 42.008 (360 / 1080 EFPD) vs MAS_SUM CYC-BU 14.0027 / 42.0082

두 방향을 **모두** 걸어야 (FA, 평면) 인덱스가 전치되거나 축이 뒤집힌 경우가 걸러진다.

### 적합 데이터셋 (`tools/build_tf_dataset.py`)
MASTER가 연료온도를 인쇄하는 **유일한** 곳은 MAS_OUT의 `$FB2D_n`(FA별 축방향 평균
Tm/Tf/Dm)이다. 여기에 `$P2D_n`(FA 상대출력)과 `$B2D_n`(FA 연소도)을 붙여 (FA, statepoint)당
한 표본을 만든다:

```
P_FA[kW] = 1000 · 3983 MW · relpow / 241
lpd      = 1000 · P_FA / (nfrod · H),   nfrod = 236,  H = 381 cm    (SolveTH 철자)
```

**9,640 표본 / 40 statepoint**, lpd **38.4 – 353.9 W/cm**(평균 183.80 = 노심평균과 일치),
bu **0 – 50.19 MWd/kgU**, dT **87.4 – 686.4 K**.

**한계(명시)**: 표본 dT는 25개 축방향 평면에 대한 `mean_k dT(bu_k, lpd_k)`이지
`dT(평균 bu, 평균 lpd)`가 아니다. dT는 LPD에 거의 선형이라 이 Jensen 간극은 축방향 출력
분산의 2차항이지만 0은 아니며, 표가 볼록한 고 LPD 쪽에서 적합을 약간 낮춘다. 또 축방향
평균 탓에 표본이 ~354 W/cm까지만 닿는다(3-D 노드는 그 위로 간다).

### 적합 (`tools/fit_tf_table.py`, 신규 옵션 2개)

```
tools/build_tf_dataset.py MAS_OUT --restart MAS_RST.APRQ_01_0360.00 -o edits.csv
tools/fit_tf_table.py edits.csv --lpd-from power,381,236 \
    --relative-to include/Database/tf.csv --smooth 1e-1 \
    --origin-continuation --extrapolate \
    --provenance "regressed-from=... mas_sum_sha256=<64hex> ..." \
    -o include/Database/tf_ce.csv
```

- **`--relative-to` = 커버리지 밖에서 WH 형상으로 고정.** 노심은 타면서 평탄해지므로 고 LPD는
  저연소도에서만, 고연소도는 중간 LPD에서만 일어난다 → 90 격자점 중 **41개가 표본 0**.
  dT를 직접 외삽하면 선형 폭주다. 대신 **비 `dT / tf.csv(bu,lpd)`를 적합**하고 다시 곱한다:
  비는 O(1)이고 완만해서 같은 곡률 평활자가 빈 격자점을 "여기서는 WH 형상을 유지"로 옮긴다.
  측정이 없는 곳에서 정직한 답이며, 요청된 *clamp to the WH table's shape*를 후처리가 아니라
  **추정량 자체**로 구현한 것이다. 적합된 비 범위 **0.664 – 1.580**.
- **`--origin-continuation`.** `GetTfuel`(`src/XSSet.h:516-526`)은 50 W/cm 미만을 Table::Get의
  클램프가 아니라 `rise·lpd/50`으로 잇는다. 적합이 클램프하면 첫 열에 틀린 구속이 걸리는데,
  첫 열이 바로 모든 집합체의 축방향 끝단이 읽는 열(= 노드 평균 tfavg의 주 기여)이다.
  최대 잔차가 14.04 → **11.89 K**로 떨어진다.
- **`--extrapolate` 사용함**, 위 41개 격자점 때문에. 커버리지(행 = 연소도, 열 = LPD):

```
  bu    0.0:   860  1048   652   540   476   376   161     1     0
  bu   5.66:   908  1400  1128   848   692   486   170     0     0
  bu  10.27:   112   588   844   652   513   284    55     0     0
  bu  12.86:     0   372   676   724   663   252     9     0     0
  bu  17.15:     0   384   801  1536  1278   159     0     0     0
  bu  24.06:     0   172   765  1860  1280    13     0     0     0
  bu   28.5:     0     0  1000  2072  1072     0     0     0     0
  bu  36.66:     0     0  1822  2938  1116     0     0     0     0
  bu  49.21:     0     0  1332  1816   484     0     0     0     0
  bu   60.0:     0     0   170   170     0     0     0     0     0
```

즉 **외삽된 셀 = LPD 400/450 열의 대부분 + 고연소도 행의 저 LPD 끝(50·100) + bu 0/5.66 행의
최고 LPD**. 이들은 전부 *WH 형상 × 이웃에서 이어진 비*이지 독립적인 주장이 아니다.
(bu 60 행과 LPD 400 열은 클램프 덕에 bu 50.19 · lpd 353.9 표본이 약한 가중치로 닿는다.)

**잔차: RMS 2.2252 K, max 11.8870 K** (표본 dT 87–686 K 대비 RMS 0.4 %). LPD 단조성 위반 **0건**.
최대 잔차는 최저출력 집합체(38–55 W/cm)에 몰린다 — 자료는 원점을 지나는 직선에 가까운데
표에는 50 W/cm 마디가 있어 생기는 꺾임이다.

### CE vs WH (200 W/cm, 연소도 방향) — 예측과 일치

| bu [MWd/kgU] | CE (tf_ce.csv) | WH (tf.csv) |
|---|---|---|
| 0         | 417.36 | 324.19 |
| 5.66      | 382.65 | 332.87 |
| 12.86     | 334.03 | 317.34 |
| 16 (보간) | 293.7  | 306.1  |
| 17.15     | 278.94 | 302.01 |
| 24.06     | 258.37 | 279.70 |
| 36.66     | 249.20 | 247.86 |

**16 MWd/kgU 구간에서 CE는 −29.6 %, WH는 −5.6 %** — `src/ThTfTable.h`가 적어둔 "약 5배 가파른
연소도 기울기"(29.6 % vs 5.8 %)를 **독립적으로 재현**했다. 원자료로도 같다: 200 W/cm 근방
표본의 dT/WH 비가 bu 0에서 1.296 → bu 16.6에서 0.954다. LPD 방향의 구조도 다르다 — CE는
dT/LPD 기울기가 2.28(38 W/cm) → 1.94(354 W/cm)로 **떨어지고**(오목), WH는 1.60 → 1.66으로
**오른다**(볼록).

### 출처 줄 (모서리 셀)
`milk::Table::ParseFromCSV`(`include/milk.h`)는 **첫 줄을 x축으로** 읽고 그 줄의 **0번 셀만**
무시한다. 따라서 `#` 주석 줄을 앞에 두면 그것이 헤더로 먹혀 로드가 깨진다. 출처는 모서리
셀에 넣는 수밖에 없고(콤마 금지), `fit_tf_table.write_table`이 이를 강제한다:

```
Bu/LPD regressed-from=MASTER/APRQ_01 run=kngr_tf_edit_20260904 isolth=12 mas_sum_sha256=665f3a… tool=tools/fit_tf_table.py,50,100,…
```

`write_table`은 **바이트로 LF**를 쓴다(`Path.write_text`는 Windows에서 CRLF로 번역한다).
표의 정체성은 파일 **바이트**의 sha256(`[RASBERY][TH][TFTABLE]`·`tools/case_key.py`)이므로
CRLF 사본은 다른 키가 된다.

### 계약 시험
`tools/test_tf_ce_table_contract.py`(신규): 존재 · tf.csv와 동일 마디 · 양축 강증가 · 양수 ·
**tf.csv와 값이 같지 않을 것** · LPD 단조 · 출처 줄(run + `mas_sum_sha256=<64hex>` + isolth=12 +
도구명) · LF · `milk::Table::ParseFromCSV` 규칙 그대로의 파싱 · `tools/case_key.py`가 이제 `ce`를
**거부하지 않고** 파일 바이트 sha256으로 키잉하며 wh와 **다른** 키를 낼 것 · 200 W/cm에서 CE
기울기가 WH의 4배 이상일 것.

`tools/test_th_tf_table_contract.py`: 5번 주장을 "파일이 **없을 때**의 거부 경로가 살아있을 것"
으로 바꾸고, tf.csv 바이트 sha256(`cb722543…`)을 **핀**으로 추가했다(B0 기준선 보호).

동시 통과: `test_enum_alias_contract`, `test_dependent_template_contract`,
`fit_tf_table.py --self-test`(최악 knot 오차 0.035 K).

### 238 레시피 (이제 실행 가능)
1. KNGR 단일 `legacy` 실행 = 기준팔. 덱팔 = `nfrod: 236` + `"tf table": "ce"` +
   `RASBERY_TH_TF_TABLE=deck`.
2. 두 팔의 `[RASBERY][TH][TFTABLE]`(name/sha256)과 `[RASBERY][CASE].th_tf_table`이 **다른지**
   먼저 확인 — 같으면 표가 바뀌지 않은 것이다.
3. Gate A(keff/ppm) · Gate B(핀 RMS) + `docs/kngr_v2_vs_master.csv` 대비 tfavg 사다리.
4. 합격 기준: 전 statepoint **|Δtfavg| ≤ 5 K**, **max|Δppm| ≤ 5**, **|Δfqp| ≤ 0.02**.
5. 덱팔은 궤적을 바꾸므로 **N1**(새 기준선), 기본팔은 산술 불변 **A2** — 위 §정확성 분류대로.

### 미해결로 남기는 것
- **Jensen 편향**: 축방향 평균에서 온다. 없애려면 3-D 노드 온도가 필요하고, 그러려면 MASTER를
  3-D 피드백 맵을 실제로 인쇄하는 옵션(또는 Record 23/24/25를 쓰는 restart 빌드)으로 다시
  돌려야 한다. 이번 실행의 `iprfb=2`는 MAS_OUT에 `$FB3D`류 블록을 만들지 않았다 —
  `$FB1D`/`$FB2D`뿐이다.
- **41개 외삽 격자점**은 WH 형상을 물려받았을 뿐 측정이 아니다. KNGR 노드가 실제로 그 구석을
  밟는지 Gate 실행의 (bu, lpd) 히스토그램으로 확인할 것.
