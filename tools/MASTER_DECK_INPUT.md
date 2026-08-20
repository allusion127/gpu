# MASTER 덱을 RASBERY 입력으로 쓰기

`tools/master2rasi.py`는 MASTER 카드 덱(`%GEN_DIM`, `%LPD_BCH`, `%EXE_DEP` …)을
RASBERY 케이스 JSON으로 번역한다. `tools/rasberry-masi`는 번역과 실행을 한 줄로
묶은 래퍼다. C++ 소스는 건드리지 않는다 — 기존 `RASBERY --rasi`가 이미 받는
JSON을 만들어 낼 뿐이다.

```
rasberry-masi depf_01.inp --xs ismr_higa.h5 -o out.h5 --fold quarter --tfuel 900
```

---

## 1. 설계 계약

### 1.1 덱 불변 (deck invariance)

**같은 덱 파일 하나가 MASTER와 RASBERY 양쪽에서 무수정으로 돈다.** RASBERY에만
필요한 정보(XS 라이브러리 경로, fold 옵션, 고정 연료온도)는

1. **CLI 인자가 1순위** — `--xs`, `--fold`, `--tfuel`
2. 덱에 적어야 한다면 **`!RASI` 주석 지시자만** 사용

덱 본문에 RASBERY 전용 카드를 추가하는 것은 금지한다.

MASTER 4.0 매뉴얼(입력 문법 규칙)은 `#`와 `!`를 **행 어디서나 주석 시작 문자**로
규정한다. 따라서 `!RASI ...` 행은 MASTER가 읽기 전에 버린다. 실측 확인:

| 확인 | 결과 |
|---|---|
| MASTER가 스캔한 덱 사본 `MAS_INP0`의 `RASI` 출현 횟수 | **0** |
| `!RASI` 3줄 포함 덱 vs 원본 덱의 `MAS_INP0` | **바이트 동일** |
| MASTER 실행 결과 | 정상 종료, 24 스텝 |

지원 지시자:

```
!RASI xs=ismr_higa.h5          # 교차단면 라이브러리
!RASI tfuel=900.0              # 고정 연료온도 [K]
!RASI fold=quarter             # none | quarter
!RASI boron=0.0                # 초기 붕소 [ppm]
!RASI xenon=equilibrium        # transient | equilibrium (§7.1, 혼재 덱 전용)
!RASI alias FA_XX=MYMODEL      # 조성 이름 → 라이브러리 모델 이름
```

CLI 인자가 주어지면 지시자를 덮어쓴다(단 `--boron`과 덱 값이 충돌하면 fail-closed).

### 1.2 fail-closed

번역기는 **추측하지 않는다.** 모든 카드는 번역 대상 / 무해(inert) / 거부 중
하나로 명시 분류되며, 표에 없는 카드를 만나면 즉시 종료한다. 카드 *값*도
마찬가지다 — 미지의 키워드나 범위 밖 플래그는 조용한 기본값이 아니라 오류다.

RASBERY 파서 자체는 관대해서 위험한 조용한 기본값이 여럿 있다(모르는 `search`
문자열 → `keff`, 모르는 `TH_mode` → `none`, `type` 없는 schedule 항목 → 이후 전체
무시, 축방향 스택 길이 불일치 → 검사 없이 범위 밖 접근). 번역기는 이것들을
**자기 쪽에서 먼저 검증**한다.

---

## 2. 카드 커버리지

`master2rasi.py --coverage <deck>`으로 덱별 실제 표를 뽑을 수 있다.

### 2.1 번역되는 카드

| 카드 | MASTER 필드 | RASBERY JSON |
|---|---|---|
| `%GEN_DIM` | `nx ny nz nbatch ncomp / ndim ngeo nsym ndivxy ndivz / ng` | `geometry.dimensions.ng`, `.xydivision`(=`ndivxy`), `.nx/.ny/.nz`(기록용) |
| `%GEN_GEO` | `wide height` + `zmesh(1..nz)` | `geometry.size.hx/hy`(=`wide`), `.hz`(RLE, **순서 반전**) |
| `%GEN_SYM` | `isymlx/ly/lz`, `isymrx/ry/rz` | `geometry.albedo` (0→0.5 진공, ±1→0.0 반사) |
| `%GEN_THD` | `power / tin trise tavg press / mflow hgfl` | `TH.rated power`(fold 시 ÷4), `.inlet temperature`(=`tin`+273.15), `.outlet temperature`(=`tin+trise`+273.15), `.global pressure`(bar÷10 → MPa), `default parameters.moderator_temperature`(=`tavg`+273.15) |
| `%GEN_FDB` | `tf tm dm` | `TH.TH_mode` — 셋 다 `off`→`none`, 셋 다 `on`→`steady` |
| `%GEN_PIN` | `icornf iweigh npin nfrod` | `geometry.dimensions.npins`(=\|`npin`\|) |
| `%LPD_BCH` | 배치 2-D 맵, `o`=노심 외부 | `core` (선행 공백→`"XX"`, 후행 공백→짧은 행) |
| `%LPD_B&C` | 배치별 `icomz(1..nz)` (**아래→위**) | `batch[b]` = `{id,count}` RLE (**위→아래**로 반전) |
| `%LPD_C&X` | `icomp nmxset ind [nfrods]` | 조성 번호 → 라이브러리 모델 이름 해석 |
| `%EXE_STD` | `isearch ixe ism pload` | `schedule[0]` = `{type:"standard", search:…}`, `default parameters.xenon`, 이후 `rate`=`pload`×100 |
| `%EXE_DEP` | `delt itg [tgkeff tgppm]` | `delt>0` → `{type:"depletion", steps:1, rate, time:delt}` |
| `%EDT_OPT` | `iwrst icob ippi …` | `schedule[].print` — `ippi≥1`→`pin-wise information`, `iwrst≥1`→마지막 스텝 `save` |
| `%ROD_CFG` | `idgr … matabs … crpos1 crpos2 ifgtp` | `rod configuration[g].ctype`(=`matabs`) + `--rod-profile` |
| `%ROD_MAP` | 제어봉 2-D 맵, `o`=봉 없음 | `rod map` (`o`→`"XX"`) |

### 2.2 받아들이되 버리는 카드 (inert)

RASBERY가 소비할 정보가 없는 카드다. 번역기는 조용히 통과시키고 커버리지 표에
사유를 적는다.

`%JOB_TYP` `%JOB_VER` `%JOB_TIT` `%JOB_IDE` `%JOB_MDL`
`%GEN_LMT` `%GEN_CDN` `%EDT_OUT` `%EDT_PIN` `%LPD_HFF`
`%COB_INP` `%MSC_MEM` `%DEF_MSC`

- `%LPD_HFF`(형상함수)는 무시된다. RASBERY의 핀출력 재구성은 XS 라이브러리에
  들어 있는 인자를 쓴다.

### 2.2b 출력은 없지만 값을 검사하는 카드 (checked)

JSON을 만들지 않지만 특정 설정이면 번역 자체가 틀리기 때문에, 토큰을 **실제로
읽는다.** 통과하거나 중단하거나 둘 중 하나다.

| 카드 | 읽는 필드 | 거부 조건 |
|---|---|---|
| `%JOB_HEX` | `ihex` | `ihex≠0` — 육각 노심. RASBERY 기하는 직교뿐이고 번역 경로가 없다 |
| `%GEN_MTH` | `ibndc` (3번째 필드) | `ibndc=1/2` — MASTER 고유의 알베도/경계선원 지정이 `%GEN_SYM`을 대체한다. 번역기는 `%GEN_SYM`만 읽으므로 덱이 요구한 것과 다른 경계조건을 내보내게 된다. 범위(0..2) 밖도 거부 |

이전 판은 둘 다 inert였다. 커버리지 표는 `%JOB_HEX`를 "checked, then inert"라고
적었지만 검사하는 코드가 없었고, `ibndc`는 이 문서가 스스로 발산원이라고
적어 놓고도 조용히 통과시켰다.

### 2.3 거부하는 카드 (fail-closed)

| 카드 | 사유 |
|---|---|
| `%EXE_ROD` | MASTER는 축 스택 **바닥 기준** `%ROD_CFG` 단위로 위치를 주고, 명시되지 않은 뱅크는 전인출로 되돌리며, 이름을 **접두사 매칭**한다. RASBERY의 `rod insertion`은 **활성노심 상단 기준 cm**이고 이 규칙이 하나도 없다 |
| `%EXE_CRS` | MASTER 제어봉 탐색의 뱅크 시퀀스를 RASBERY 봉 탐색이 단계별로 재현하지 못함 |
| `%EXE_RHO` | 반응도계수 실행 모드에 대응물 없음 |
| `%EXE_SHP` `%EXE_TRN` `%EXE_XED` | 형상맞춤 / 과도 / 제논동특성 |
| `%LPD_SHF` `%LPD_PUL` `%LPD_ASF` `%LPD_DB2` | 다주기 셔플, SFP 냉각, 비대칭 집합체, 예약 카드 |
| `%DEF_ETS` `%DEF_TFT` | 외부선원, 사용자 연료온도 표 |
| `%DET_CFG` `%DET_MAP` `%GEN_DCH` `%ROD_COR` | 검출기, 붕괴열, 코너봉(육각 전용) |
| `%THC_FLO` `%THC_TIN` | 채널별 유량/입구온도 |
| `%TRN_*` `%XED_*` | 과도/제논동특성 그룹 |
| `%XSM_*` | 교차단면 수정 — 라이브러리 생성 단계(CHIFFON)에서 해야 함 |

`%EXE_DEP`도 `delt < 0`(목표 도달까지 자동 감손)이면 거부한다. 스텝 수가 실행
전에 정해지지 않아 RASBERY 스케줄로 옮길 수 없다. 덱에서 명시 스텝으로 펴야 한다.

### 2.4 카드 이름과 필드 수

- **카드 이름은 양쪽이 고정이다.** `%GEN_DIM1`처럼 접미 오타가 붙으면 예전에는
  앞 6글자가 `%GEN_DIM`으로 매칭되고 남은 `1`이 첫 본문 토큰이 되어, `nx`부터
  한 칸씩 밀린 덱이 **조용히** 번역됐다. 지금은 카드 이름 뒤에 공백이나 줄 끝이
  와야 하고, 아니면 `unrecognised card syntax`로 중단한다.
- **고정폭 카드는 필드 수가 정확해야 한다.** 남는 토큰은 읽히지 않으므로
  오타가 유효 입력처럼 통과한다.

  | 카드 | 필드 수 |
  |---|---|
  | `%GEN_DIM` | 정확히 11 |
  | `%GEN_SYM` | 정확히 6 |
  | `%GEN_PIN` | 정확히 4 |
  | `%GEN_THD` | 5 또는 7 (`mflow hgfl` 유무) |
  | `%EXE_STD` | 정확히 4 |
  | `%EXE_ROD` 행 | 정확히 2 |
  | `%ROD_CFG` 행 | 정확히 11 |

- **수는 유한해야 한다.** `nan`/`inf`/`-Infinity`는 파이썬 `float()`이 받아
  주지만 JSON이 아니고, RASBERY는 그 상태점을 조용히 오염된 채로 쓴다. 덱
  경계에서 거부하고, 출력 직전에도 `json.dumps(allow_nan=False)`로 다시 막는다.
- **`%GEN_THD`는 물리적으로도 검사한다** — 압력 > 0, 온도 상승 ≥ 0,
  입구·평균·출구 온도가 절대영도 위.
- **`%GEN_SYM`은 세 축 모두 `0`/`-1`/`+1`만 받는다.** 예전에는 z축만 "0이 아니면
  반사"였고, 오타(`2`, `10`)가 반사 경계로 조용히 들어갔다.

---

## 3. 사전 호환성 검사

번역 **전에** 덱을 진단한다. 막히는 항목을 한 번에 모두 보고하고, 막히면 exit 1.

```
master2rasi.py depf_01.inp --check
rasberry-masi depf_01.inp --check          # 동일
```

보고 항목:

1. **카드** — 번역/inert/거부 분류와 사유
2. **기하** — 집합체 수, 축 노드, `ngeo`/`ndim` 경고
3. **장전 맵과 fold** — `--fold quarter` 가능 여부(팔분대칭 검증), `--fold none`
   가능 여부(공백이 선행/후행뿐인지)
4. **덱이 담을 수 없는 입력** — XS 라이브러리 유무, 모델 이름 해석 결과,
   연료온도 유무
5. **판정** — `TRANSLATABLE as invoked` 또는 `BLOCKED` + 사유 목록

### 3.1 종료 코드와 `assume`

번역 기록은 두 등급이다.

- `note` — 출력이 덱만으로 완전히 결정된다. 출처 기록일 뿐이다.
- `ASSUME` — 덱이 말하지 않은 것을 번역기가 **물리적 기본값으로 채웠다**(또는
  검증을 건너뛰었다). 출력은 합리적일지언정 추측이다.

| 코드 | 뜻 |
|---|---|
| 0 | 번역 성공, 추측 없음 |
| 1 | 덱 거부 (`DeckError`) |
| 2 | 덱 파일을 읽을 수 없음 / 래퍼 사용법 오류 |
| **3** | **JSON은 썼지만 `ASSUME`이 하나 이상 있었다** |

`ASSUME`이 붙는 경우: `%GEN_SYM` 부재(6면 진공 가정), XS 라이브러리 모델 이름을
읽지 못해 이름 대응을 검증하지 못함, `ism`이 `ixe`와 달라 사마륨 요구를 버림,
`%EXE_STD` 없이 `delt=0` `%EXE_DEP`를 초기 상태로 삼음, 궤환이 켜진 덱에서
`%GEN_THD mflow/hgfl`을 버림(§7.2).

`--allow-assumptions`를 주면 목록을 그대로 찍고 0으로 끝난다. `rasberry-masi`는
코드 3을 그대로 전파하며, 받아들이기 전에는 **풀지 않는다**(JSON은 이미 쓰여
있으니 눈으로 확인하고 다시 부르면 된다).

---

## 4. 대칭 처리

기하 모드는 **셋**이다.

| `%GEN_DIM ngeo` | `--fold` | 결과 |
|---|---|---|
| 1 (전노심) | `none` | 덱 그대로. `{angle:360, mirror:false, divided:false}`, 정격출력 ×1 |
| 1 (전노심) | `quarter` | 팔분대칭 검증 후 1/4로 접음. `{90, true, true}`, ×1/4 |
| 4 (이미 1/4) | `none` | 이미 접힌 노심으로 번역(§4.1). `{90, true, divided}`, ×1/4 |
| 4 (이미 1/4) | `quarter` | **거부** — 두 번 접힌다 |
| 그 외 | — | **거부** — RASBERY `symmetry`에 1/n 섹터 표현이 없다 |

### 4.1 이미 접힌 덱 (`ngeo≠1`)

예전에는 모드가 둘뿐이었다. `ngeo=4` 덱은 어느 쪽에도 안 맞아 **전노심 분기로
떨어졌고**, 1/4 맵이 `{angle:360, mirror:false}` + 전노심 정격출력으로 방출됐다.
**출력밀도 4배 오류**이며, 실행은 정상 종료하고 그럴듯한 k-eff를 낸다.

지금은 모든 값을 카드에서 **유도**한다(추측하지 않는다).

| 출력 | 유도 근거 |
|---|---|
| `symmetry.angle` = 360/ngeo | `%GEN_DIM ngeo` |
| `TH.rated power` ×1/ngeo | 같음. RASBERY의 정격출력은 **모델링된 섹터**의 출력이다 |
| `symmetry.mirror` | 1/4 맵이 주대각에 대해 대칭인지 실제로 검사 |
| `center assembly divided` | `%GEN_SYM isymlx`: `-1`(집합체 중앙을 지나는 대칭면)→`true`, `+1`(집합체 경계)→`false` |
| `albedo west/north` = 0.0 | `%GEN_SYM isymlx/isymly = ±1`에서 그대로 나온다 |

유도가 안 되면 거부한다 — `%GEN_SYM` 부재, 절단면이 대칭면으로 선언되지 않음
(`isymlx=0`), `isymlx≠isymly`(RASBERY는 축별 플래그가 없다), 대각 비대칭,
`ngeo∉{1,4}`.

fold 전 **팔분대칭을 검증**한다 — 좌우 거울, 상하 거울, 주대각 전치 셋 다.
하나라도 깨지면 위치를 찍어 보고하고 **거부**한다(조용히 정보를 버리지 않는다).

검증을 통과하면 중심행/중심열을 포함한 남동 사분면을 취하고, 후행 공백을 잘라
RASBERY의 톱니형 행으로 만든다. 출력에는 `{"angle":90, "mirror":true,
"center assembly divided":true}`, 알베도 west/north=0.0, 정격출력 ÷4가 붙는다.

`--fold none`(기본)은 덱 그대로 번역한다. RASBERY의 `core` 배열은 빈 자리를
**선행 `"XX"` 연속** 또는 **짧은 행**으로만 표현할 수 있다(중간에 낀 `"XX"`는
그 행 전체를 조용히 어긋나게 한다). 그래서 중간 공백이 있으면 거부한다.

---

## 5. 동일 덱 이중 실행 (MASTER + RASBERY)

목적: **벤치마크·DB에서 기준값(MASTER)과 고속값(RASBERY)을 같은 입력으로** 뽑는다.

```
# (a) 기준값 — MASTER
copy  ..\ISMR_CY01_PROLOG1\MAS_XSL  MAS_XSL
copy  ..\ISMR_CY01_PROLOG1\MAS_HFF  MAS_HFF
copy  depf_01.inp  MAS_INP
..\CODE\BIN\master4.0m4_r1.exe
move  MAS_SUM  depf_01.sum

# (b) 고속값 — RASBERY, 같은 파일
rasberry-masi depf_01.inp --xs ismr_higa.h5 -o out.h5 --fold quarter --tfuel 900
```

`depf_01.inp`는 두 실행 사이에 **한 글자도 바뀌지 않는다.** `!RASI` 지시자를
넣어 두면 (b)의 CLI 인자도 생략할 수 있고, (a)는 그 줄들을 주석으로 버린다.

### 5.1 교차단면 라이브러리는 코드별로 별도다

두 코드는 같은 물리 데이터를 **다른 파일 형식**으로 읽는다. 같은 HGC 계보에서
나왔다는 점을 반드시 확인하고 쓸 것 — 번역기는 이것을 검사할 수 없다.

| | MASTER | RASBERY |
|---|---|---|
| 라이브러리 파일 | `MAS_XSL` (텍스트) | `*.h5` (CHIFFON HDF5) |
| 형상함수 | `MAS_HFF` (별도 파일) | 같은 `.h5`에 포함 |
| 생성 도구 | PROLOG (`prolog41m4.exe`) | CHIFFON (`RASBERY --chiffoni`) |
| 생성 입력 | `PRO_FA_*.INP` | `*_xs.json` |
| **공통 상류** | DeCART2D HGC (`dec_FA_*.HGC`) | 같은 HGC (`dec_FA_*.HGC`) |
| 조성 이름 | `FA_A1`, `REF_R1`, `REF_AXIAL_B` … | `A1`, `R1`, `RB` … |
| 이름 대응 | `%LPD_C&X`의 `nmxset` | 번역기가 정규화(§6) |
| 덱이 경로를 지정? | 함 (`%JOB_TYP`의 `xsl`/`hff`) | **못 함** → `--xs` |

`MAS_XSL`은 `MAS_REF` + 각 집합체 `PRO_FA_*.XSD`를 이어붙여 만든다
(`xsgen.bat`). 덱이 참조하는 조성이 하나라도 빠지면 MASTER는
`forrtl: severe (24): end-of-file during read, unit 2, file ...MAS_XSL`로 죽는다.

### 5.2 실측 대조 (iSMR_HIGA CY-01, ARO, 전노심 11×11×26)

같은 `depf_01.inp` 한 파일. MASTER는 로컬 Windows 실행(30.6 s), RASBERY는
`rasberry-masi`로 1/4 노심 번역 후 CPU 실행(1.6 s).

| 스텝 | EFPD | MASTER k-eff | RASBERY k-eff | Δk [pcm] |
|---:|---:|---:|---:|---:|
| 1 | 0.000 | 1.032934 | 1.034084 | +115.0 |
| 4 | 37.392 | 1.004667 | 1.005531 | +86.4 |
| 12 | 336.527 | 1.010306 | 1.011122 | +81.6 |
| 20 | 635.663 | 1.007720 | 1.009185 | +146.5 |
| 24 | 749.708 | 1.002826 | 1.005963 | +313.7 |

전 24스텝 Δk = **+71 … +314 pcm**, EFPD 격자 일치 오차 < 5e-4 일.
BOC +115 pcm은 내장 i-SMR 검증 케이스의 −81 pcm과 같은 대역이고, EOC 쪽 드리프트는
감손/반사체 처리 차이로 남아 있는 기존 과제다(번역 오류가 아니다 — §7 참조).

---

## 6. 조성 이름 대응

MASTER `%LPD_C&X`의 XS 세트 이름과 CHIFFON 라이브러리의 모델 이름은 다르다.
번역기는 순서대로 시도하고, **후보가 라이브러리에 실재할 때만** 채택한다.

1. `!RASI alias` / `--alias`로 준 명시 대응
2. 이름 그대로
3. 정규화 규칙 — `REF_AXIAL_B`→`RB`, `REF_AXIAL_T`→`RT`, `FA_*`→`*`, `REF_*`→`*`

모두 실패하면 시도한 후보와 라이브러리의 실제 모델 목록을 찍고 종료한다.
`--xs`를 주면 라이브러리를 열어 검증하므로(h5py 필요) 오탈자가 실행 전에 잡힌다.

축 반사체 모델 이름은 **`R`로 시작해야 한다.** RASBERY는 배치 층 이름의 첫 글자가
`R`인지로 연료/반사체를 가르고, 여기서 활성노심 범위(연소·첨두·봉 삽입 깊이)가
정해진다.

번역기는 이 불변식을 **채택한 후보에 대해 실제로 검사한다.** `REF_*` 규칙이
`R`로 시작하지 않는 이름을 내면(예: 라이브러리에 `RB`가 없어
`REF_AXIAL_B`→`AXIAL_B`가 채택되는 경우) 중단한다 — 그대로 두면 반사체가 연료로
적재되고 활성노심 범위가 어긋난다.

---

## 7. 제약

**덱이 담을 수 없어 반드시 밖에서 줘야 하는 것**

- **교차단면 라이브러리** (`--xs` 또는 `!RASI xs=`). MASTER 덱은 `MAS_XSL`만
  가리킬 수 있고 CHIFFON `.h5`를 지정할 문법이 없다.
- **고정 연료온도** (`--tfuel` 또는 `!RASI tfuel=`). `%GEN_FDB`가 연료 궤환을
  끄면 MASTER는 연료를 **라이브러리 기준점**에 묶어 두는데, 그 값은 어느 카드에도
  없다. DeCART 기준 분기값(iSMR_HIGA는 900 K)을 주면 된다. 궤환이 켜져 있어도
  초기값으로 필요하다.

### 7.1 상태점별 제논 (`%EXE_STD ixe`)

MASTER는 `ixe`를 **실행 상태마다** 준다. RASBERY는 `default parameters.xenon`
**하나**를 케이스 전체에 적용한다.

예전에는 마지막 `%EXE_STD`의 값이 그대로 전역 설정이 됐다. 감손 상태들이
`eq`(평형)이고 마지막 EOC 재편집만 `tr`(과도)인 덱은 **전 주기가 과도 제논으로**
감손됐다 — 소급 적용이고, 아무 말도 없었다.

지금은 모드가 섞이면 거부한다. 어느 쪽을 쓸지 알고 있다면 명시하라.

```
master2rasi.py deck.inp --xenon equilibrium ...      # 또는 !RASI xenon=equilibrium
```

덱이 실제로 요구하지 않는 모드를 주면 거부한다. MASTER가 대부분의 상태점에서
무엇을 돌렸는지는 MAS_SUM의 `SUMMARY EDIT : OPTIONS` 열(XE/SM)에 있다.

### 7.2 `%GEN_THD`의 `mflow`/`hgfl`

읽고 나서 **버려졌다**. RASBERY의 `TH` 블록에는 유량 입력이 없다 — 정격출력과
입·출구 엔탈피 차로 노심 유량을 스스로 정한다.

- `%GEN_FDB` 궤환 **off** (`TH_mode="none"`): T/H 계산 자체가 없으므로 무의미.
  `note`로 기록.
- 궤환 **on** (`TH_mode="steady"`): 덱이 말한 유량과 다른 유량으로 T/H가 돈다.
  `ASSUME`으로 올리고 **exit 3**. `power/(h_out − h_in)`이 덱의 `mflow`를
  재현하는지 확인한 뒤 `--allow-assumptions`로 받아들여라.

**번역되지만 관례가 끼는 것**

- `%EDT_OPT`의 `iwrst`는 MASTER에서 **매 편집점마다** 재시작 파일을 쓴다(24개의
  `MAS_RST.*`). 번역기는 **마지막 스텝만** `save`한다 — 출력 크기 때문이며
  `--save-every`로 되돌린다.
- MASTER는 `%EXE_STD` 잡케이스와 뒤따르는 `delt=0` `%EXE_DEP`을 각각 실행하지만
  둘 다 0 EFPD의 같은 상태다. 번역기는 `standard` 항목 하나로 낸다(MASTER 요약의
  24개 상태와 일치).
- 끝에 붙은 `%EDT_OPT`만 있는 잡케이스는 MASTER 요약의 중복 재편집 행(NO. 25)을
  만든다. 상태를 진행시키지 않으므로 스케줄 항목을 내지 않는다.
- `steps:N` 묶음의 `print`는 RASBERY에서 **첫 항목에만** 적용된다. 그래서 번역기는
  `%EXE_DEP` 하나당 `steps:1` 항목 하나를 낸다(19개 동일 스텝도 19개 항목).

**물리·수치상 알려진 제약**

- **CBC(임계붕소) 탐색은 라이브러리의 붕소 분기가 살아 있어야 한다.** iSMR_HIGA
  라이브러리는 DeCART `EDIT/isotope` 카드에 B-10(5000)이 없어 붕소 분기 좌표가
  축퇴돼 있고, CHIFFON이 이를 감지해 분기를 버린다(모델당 경고 1줄). 이 상태에서
  `isearch=boron` 덱을 번역해 돌리면 붕소를 바꿔도 반응도가 움직이지 않는다.
  해결하려면 DeCART FA 입력에 `5000`을 넣어 재생성해야 한다.
- `%GEN_FDB`가 일부만 `on`이면 거부한다(예: FTC 계산의 `on off off`). RASBERY의
  T/H 궤환은 전부-아니면-전무다.
- 알베도 진공면을 `0.5`로 쓴다. 검증된 i-SMR 관례를 따른 것이며 완전 Marshak
  진공(`1.0`)이 아니다.
- 축방향 `ndivz`는 MASTER 4.0에서 "Not working"이고 RASBERY에도 대응물이 없다.
- 육각 노심은 지원하지 않는다.

---

## 8. 파일

```
tools/master2rasi.py        MASTER 덱 → RASBERY 케이스 JSON (파이썬, 의존성 없음;
                            h5py 있으면 XS 모델 이름 검증)
tools/rasberry-masi         번역 + 실행 래퍼
tools/MASTER_DECK_INPUT.md  이 문서
```

`master2rasi.py --help` / `rasberry-masi --help`에 전체 옵션이 있다.
