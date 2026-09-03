# GPU RASBERY vs MASTER — i-SMR 실행 비교 런북 (2026-09-03)

이 문서는 두 개의 런북을 담는다.

* **A. 238 `~/gates/block54.sh`** — GPU RASBERY 정확도(앵커 A/B/C) + 단일 속도 + 배치 처리량.
  기존 체인(`block51v2 → block52 → block53`)의 **`block53_DONE` 이후에 이어 붙인다.**
* **B. 181 MASTER 런북** — MASTER 벽시계(단일 median-of-10, W16 cases/h)와 핀(MAS_PPI) 재생성.

그리고 두 런북의 **전제**가 되는 대칭 규약 판정(§1)과 자산 스테이징 현황(§2)을 먼저 적는다.

이 문서에 적힌 명령은 **아직 실행되지 않았다.** 238의 GPU 체인(block51v2→52→53)이 작업 시점에
가동 중이었으므로 어떤 RASBERY 실행도 시작하지 않았고, 181은 유휴가 아니었으므로 MASTER를
실행하지 않았다(§2.3의 blocker). 스테이징(파일 복사)만 완료되어 있다.

---

## 1. 대칭 규약 판정: **ROTATIONAL** (덱 5종 변경 필요)

### 1.1 판정

**MASTER i-SMR 기준해는 중심 집합체를 반으로 가르는 90도 회전(rotational) 1/4 노심이다.**
RASBERY 용어로 `mirror: false`. 커밋 `6c0d068`이 옳고, `test/7_i-SMR_Validation`의 덱 5종이
모두 틀렸다. 확신도 0.93.

주의: 과제에서 지목된 아카이브 쌍 `ismr_retro_{mir,rot}`만으로는 **판정할 수 없다.** 그 덱의
1/4 맵은 전치 불변이라 두 fold가 수학적으로 같은 문제를 푼다(24 스텝 keff가 소수점 6자리까지
동일). 판정 근거는 MASTER `.sum`의 출력 구조와 MASTER 카드 덱이다.

### 1.2 근거

1. **아카이브 쌍은 축퇴(degenerate)다.** `ismr_retro_mir` / `ismr_retro_rot`의 덱 차이는
   `"mirror"` 한 바이트뿐이고, `run.log`의 24개 keff가 전부 동일하다. 차이는 `run.err`의 dhat
   진단 반올림(max|dhat/dtil| 2.33553 vs 2.33742)과 `out_pinpower.csv` 46바이트뿐.
   `src/Geometry.cpp:168-177`이 "두 fold는 사분면 장전이 대각 대칭일 때만 일치한다"고 명시한다.
   아카이브 게이트 `out_rotfold_gates.txt`가 이를 정량화한다 — CY01은 max|dkeff| 0.0004 pcm,
   최악 노드 출력 9.9e-05 %; **CY02/03/04는 6.83 / 7.50 / 6.14 pcm, 최악 노드 출력
   5.56 / 4.50 / 4.18 %.** 판별 케이스는 CY01이 아니라 CY02–04다.

2. **결정적 증거 — MASTER `.sum` SUMMARY EDIT 5의 출력 구조.**
   네 개의 `Reference_output/depf_0N.sum`에서 집합체 출력/연소도/k-inf 맵을 파싱해, 모든
   전치 쌍 (Y_r,X_c) vs (Y_c,X_r)을 **중심선 쌍**(행 Y5 또는 열 E)과 **내부 쌍**으로 나눴다.

   | 덱 | 중심선 쌍 | 내부 쌍 최대 |d| |
   |---|---|---|
   | depf_02 | 100쌍, 출력·연소도·k-inf 전부 **0.00000 %** | 출력 0.276 % (NO.=13, 6J 0.8347 vs 9F 0.8324), 연소도 0.157 %, k-inf 0.039 % |
   | depf_03 | 104쌍, 전부 0.00000 % | 출력 0.158 %, 연소도 0.149 % (NO.=1, 7J 22.739 vs 9G 22.773 MWd/kg), k-inf 0.029 % |
   | depf_04 | 104쌍, 전부 0.00000 % | 출력 0.494 % (NO.=19, 7J 0.6072 vs 9G 0.6042), 연소도 0.175 %, k-inf 0.029 % |
   | depf_01 | (신연료, 대각 대칭 장전) 전부 0 — 축퇴 |

   **연소도 줄이 결정타다.** 집합체 연소도는 반복 계산으로 수렴하는 양이 아니라 셔플로
   실려 들어오는 양이다. MASTER의 내부 전치 쌍은 실제로 다르므로(0.034–0.036 MWd/kg, 인쇄
   해상도의 수십 배) MASTER 모델은 집합체 하위 수준에서 대각 대칭이 **아니다.** 그런데
   중심선 전치 쌍 — (Y5,F)/(Y6,E), (Y5,G)/(Y7,E), (Y5,H)/(Y8,E), (Y5,J)/(Y9,E) — 은 네 덱,
   모든 상태점, 세 양 모두에서 비트 단위로 동일하다. **mirror fold라면 그 둘은 서로 다른 두
   집합체이고 내부 쌍과 똑같이 달라야 한다.** rotational fold(중심 집합체 분할)에서는 그 둘이
   한 물리 집합체의 두 반쪽이므로 MASTER가 하나만 계산하고 집합체 평균을 두 자리에 인쇄한다.
   파일이 보여주는 것이 후자다.

3. **MASTER 카드 덱이 같은 말을 한다.** `depf_0N.inp` 넷 모두 `%GEN_DIM "3 4 1 2 2"`
   (ndim=3, ngeo=4, **nsym=1**, ndivxy=2, ndivz=2), `depf_01.inp`의
   `%GEN_SYM "-1 -1 0 / 0 0 0"`(두 절단면 모두 집합체 중앙 통과).
   `src/Geometry.cpp:169`가 이 카드를 그대로 지목한다: *"mirror: false — 90-degree
   ROTATIONAL fold (MASTER's %GEN_DIM nsym=1)"*.

4. **RASBERY 실행도 rotfold에서만 같은 지문을 낸다.** 아카이브 CY02 출력(mirror=`rf_a3`,
   rot=`rf_b2`, 덱은 flag 하나만 다름)에 같은 전치 쌍 검사:
   mirror는 중심선 0.0000 %, 내부 0.0000 % — 완전 대각 대칭이라 MASTER의 0이 아닌 내부
   비대칭과 모순. rot은 중심선 출력 1.25/3.59/3.92 %, 연소도 26.3/17.4/11.9 %(진짜 집합체
   내 반쪽 분할; step1 5F/6E 연소도 12.008 vs 16.295, 평균 14.151 = mirror의 단일값 14.1516),
   내부는 출력 0.026/0.096/0.097 %, 연소도 0/0.029/0.047 % — MASTER 잔여 내부 비대칭과 같은 자리수.

5. **APR1400 수용 기록의 방증.** `out_rotfold_gates_c.txt` C5절, cy02 vs MASTER: 중심선 17좌석
   집합체 출력 오차가 mirror rms 2.88 % / max +6.28 %(BOC), 3.76 % / +7.76 %(450 EFPD)에서
   rot(중심선 쌍 평균) 1.44 % / -2.74 %, 0.166 % / +0.364 %로 떨어진다. 내부 52좌석은 거의
   그대로(1.70→1.41 %, 1.24→0.27 %). 개선이 두 fold가 다른 좌석에만 몰린다.

6. **정직한 유보 — i-SMR 자체에서는 정확도 이득이 관측되지 않는다.** 두 아카이브 CY02 실행을
   `depf_02.sum`에 대해 22 연료좌석 EFPD 조인(781.21474099 d 오프셋)으로 재비교하면 전체 rms는
   BOC 1.97 %(mir) vs 2.29 %(rot), 314 EFPD 5.24 vs 5.21 %, 696 EFPD 1.91 vs 1.87 %. 둘 다
   중기 F6 좌석에서 -14.7 %로 정점을 찍는 **공통 편향**에 지배되며, 이는 fold가 아니라
   단면적/물리 불일치다. 즉 이 변경은 **규약 정정이지 측정된 정확도 개선이 아니다.**

### 1.3 덱 변경 — 리포지터리에 커밋할 수 없다 (중요)

`geometry.symmetry.mirror`를 `true` → `false`로 5개 파일에서 바꾼다:

```
test/7_i-SMR_Validation/i-SMR_CY01.json
test/7_i-SMR_Validation/i-SMR_CY02.json
test/7_i-SMR_Validation/i-SMR_CY03.json
test/7_i-SMR_Validation/i-SMR_CY04.json
test/7_i-SMR_Validation/cy02_step1.json
```

**이 변경은 커밋에 들어가지 않는다.** `.gitignore:82`가 `/test/**`를 제외한다 — `test/`는
대역 외로 배포되는 ~3.9 GB 픽스처 코퍼스이고, 추적되는 것은 `test/*.cpp|*.h|*.cu`와
`test/reference/*.json`뿐이다(`git ls-files test/` = 36개, i-SMR 덱은 0개).
따라서 **이 변경은 픽스처가 놓인 각 호스트에서 개별로 적용해야 하는 스테이징 단계**이며,
아래 §3.1의 block54 사전조건에 그대로 들어 있다. 로컬 워크스테이션에는 이미 적용되어 있다.

적용/검증 명령(어느 호스트에서나):

```sh
cd <deck dir>
for f in i-SMR_CY01.json i-SMR_CY02.json i-SMR_CY03.json i-SMR_CY04.json cy02_step1.json; do
  sed -i 's/"mirror": true/"mirror": false/' "$f"
done
grep -H '"mirror"' *.json      # 5줄 전부 false 여야 한다
```

리포지터리 쪽 검증은 `tools/test_ismr_tools_contract.py`의 `fold_contract()`가 맡는다.
픽스처가 없는 클론에서는 NOTE를 찍고 넘어가고, 픽스처가 있는 호스트에서는 실패한다.

### 1.4 flag를 뒤집기 전에 계획해야 할 것

* **전제조건은 이미 충족.** `Geometry::Initialize`는 rotfold에서 두 절단면이 같은 경계가
  아니면 throw한다. 덱 5종 모두 albedo west=0.0, north=0.0이고 hx==hy==21.54712라 두 가드
  모두 걸리지 않는다(계약 시험이 이것도 확인한다).
* **CY01은 물리적으로 무변화**(0.0004 pcm)지만 **CY02/03/04는 최대 7.5 pcm, 노드 출력 5.6 %
  움직인다.** mirror 시대의 기준선을 키로 쓰는 것 — `.regress_baseline`, `.golden_cpu`,
  재시작 체인 `restart_21/22/23`, 저장된 h5diff 기준 — 은 전부 다시 떠야 한다.
  **CY01만 뒤집고 02–04를 두면 재시작 체인이 내부적으로 불일치한다.** 반드시 CY01→CY04를
  순서대로 다시 돌려 재시작 파일을 재생성한다(block54가 그렇게 한다).
  다행히 리포에는 커밋된 기준선 파일이 없다(`.regress_baseline`, `.golden_cpu` 둘 다 미추적,
  `test/reference/validation_baseline_manifest_v5.json`에도 i-SMR keff 기준선은 없다). 재생성
  대상은 호스트 로컬 산출물뿐이다.
* **비교 도구는 중심선 쌍을 아직 모른다 (미해결, §7-1).** rotfold에서 (Y5,X_c)와 (Y_c,E)는
  한 집합체의 두 반쪽이고 출력 3.9 %, 연소도 26 %까지 갈라지는데 MASTER는 집합체 평균 하나를
  인쇄한다. `tools/compare_master_rasbery.py`의 **스칼라** 컬럼은 영향이 없지만
  `tools/plot_ismr_validation.py`의 집합체 패널은 raw 좌석을 비교하므로 **올바른 rotational
  실행이 틀린 mirror 실행보다 나빠 보인다.** 핀 쪽은 이번 커밋에서 해결했다(§1.5).
* **`tools/master2rasi.py`도 같은 방식으로 틀렸다 (미해결, §7-2).**
  `already_folded_core()`가 ngeo=4 MASTER 덱에 대해 무조건 `"mirror": True`를 내보내고,
  `check_diagonal_symmetry()`가 전치 불변이 아닌 1/4 맵을 거부한다. 이 MASTER 덱들은 nsym=1을
  선언하므로 변환기는 `mirror: False`를 내야 하고 대각 대칭 거부는 없어져야 한다.
* **알려진 잔여 근사** (`src/Geometry.cpp:311-316`): rotfold에서 `_neibrb`는 mirror 폐쇄를
  유지한다(PPR의 3x3 핀 스텐실이 회전 이웃을 표현하지 못함). 절단선 근처 핀 재구성이 근사로
  남는다. **HIGA 핀 앵커에는 영향이 없다** — 그 장전은 대각 대칭이라 fold가 항등이다.
* **HIGA 덱의 mirror는 건드리지 않는다.** 거기서는 구조적으로 무변화다.

### 1.5 이번 커밋이 고친 것 (핀 절단선 반쪽 핀)

`center assembly divided` 덱은 두 절단선 위에 반쪽(모서리는 1/4) 집합체를 갖는다. `.h5`에서
없는 반쪽은 비어 있고 **절단선 위의 핀들은 MASTER가 인쇄하는 같은 핀의 절반 출력을 갖는다.**
HIGA BOC 앵커에서 실측:

| | 절단선 9좌석 rms | 전 노심 rms / max |
|---|---|---|
| 보정 없음 | 15–21 % | **9.513 % / 50.14 %** |
| `--half-pin-correct` | 0.43–0.66 % | **4.663 % / 24.83 %** |

보정 후 남는 6.7–8.1 %는 반사체 인접 좌석 (0,4)(3,3)(4,0)(4,1)에만 몰려 있다 — 메모리에 기록된
7.3–9.4 % 반사체 잔차 그대로다. 즉 **보정 없이 읽은 9.5 %의 절반은 fold의 산술 인공물이며,
핀 출력 오차의 단위로 인쇄된다.** 기본값은 OFF다(KNGR의 동결 production 수치를 움직일 수
없으므로). 절단 좌석이 검출되었는데 플래그가 없으면 도구가 경고를 찍는다.

---

## 2. 자산 스테이징 현황

### 2.1 238 — `~/gates/ismr/` (91 MB, 원격 `find`+`du -sh`로 확인)

```
~/gates/ismr/i-SMR_Validation.h5        94,927,853 B   (로컬 E:/rasbery_scratch_h5/decks/ 에서 복사)
~/gates/ismr/test_7/                    리포 test/7_i-SMR_Validation/ 전체
                                        i-SMR_CY01..04.json, cy02_step1.json,
                                        Reference_input/depf_0{1..4}.inp,
                                        Reference_output/depf_0{1..4}.sum
~/gates/ismr/higa_cy01/                 iSMR_HIGA_CY01.json, ismr_higa_xs.json,
                                        ismr_higa_xs_bppmON.json
                                        (BENCH_20260810/rasberry_c2/inputs/ 에서;
                                         BENCH_20260810/comparison/rasberry/ 사본과 바이트 동일)
```

h5 내용 확인(로컬 miniconda3 h5py 3.16.0; 루트에는 attrs가 없고 `Metadata` 그룹에 있다):
`Metadata/format = 'CHIFFON_HDF5'`, `Metadata/version = '3.0.0'`, `Metadata/num_models = 16`,
`Models/`에 `Model_0..Model_15`. **238에 이미 있던 2.4.0 사본이 아니라 필요한 3.0.0 자산이 맞다.**
sha256 일치 확인됨.

238에서 새로 만든 것은 격리된 `~/gates/ismr/` 디렉터리뿐이다. 가동 중인 GPU 체인
(block51v2→52→53), 다른 사용자 작업, GPU0/GPU1은 조회하지도 건드리지도 않았다.

### 2.2 181 — `C:/Users/kmk/CodexBench/ismr_master_20260903/`

로컬 `iSMR_HIGA/MASTER/`는 ISMR-CY01용 **완결된 MASTER 실행 세트**를 갖고 있다.
`depf_01.bat`를 읽어 확인한 최소 입력 집합은 정확히 `{depf_01.inp, depf_01.bat, MAS_XSL, MAS_HFF}`
이며 빠진 것이 없다(.bat는 `copy ..\ISMR_CY01_PROLOG1\MAS_XSL MAS_XSL`,
`copy ..\ISMR_CY01_PROLOG1\MAS_HFF MAS_HFF`, `copy depf_01.inp MAS_INP.` 후 exe 호출).

.bat가 의존하는 형제 디렉터리 상대경로 구조를 보존해 복사했다(공백/괄호가 있는
`"ISMR-CY01 -_ARO (REF)"`는 `ISMR-CY01_ARO_REF`로 개명; `ISMR_CY01_PROLOG1`은
`..\ISMR_CY01_PROLOG1\...` 상대참조가 살아야 하므로 원래 이름 유지):

```
ISMR-CY01_ARO_REF/depf_01.inp     7,841 B
ISMR-CY01_ARO_REF/depf_01.bat       374 B
ISMR_CY01_PROLOG1/MAS_XSL       450,088 B
ISMR_CY01_PROLOG1/MAS_HFF    82,662,255 B
```

181에서 `dir /s`로 크기 전부 일치 확인(합계 83,120,558 B). **실행하지 않았다 — 복사만.**

스테이징되지 않은 참조 산출물(로컬에만 있음, §5에서 필요해지면 복사):
`ISMR-CY01 -_ARO (REF)/depf_01.sum`(84,387 B), `MAS_PPI.SMF_01_*`(24개 × 11,571,055 B),
`MAS_RST.SMF_01_*`, `depf_01.out`, `fort.119`.

### 2.3 blocker (작업 시점)

1. **181은 유휴가 아니다.** `tasklist | findstr master`가 `master4.0m4_r1.exe` 16개 동시
   실행(각 ~131 MB RSS, 세션 `Services`)을 보였고, 수 초 간격 두 번의 호출 사이에 PID가 전부
   교체된다 — 짧은 MASTER 실행을 계속 생성/회수하는 자동 하네스가 지금 돌고 있다는 뜻.
   CPU 부하 87 %(PowerShell `Get-CimInstance Win32_Processor`; **181 이미지에 `wmic`이 없다**).
   메모리에 기록된 receipt 하네스
   `C:\Users\kmk\CodexBench\codex_gpu_master_compare_20260728\master_reference_20260728T053326Z`
   로 보인다. **지금 단일 스레드 MASTER 타이밍을 걸면 CPU를 경합해 벽시계 측정이 무효가 된다.**
   181이 유휴가 될 때까지 기다리거나, 잡음을 감수한다는 명시적 승인을 받아야 한다.
2. **181의 `depf_01.bat`이 스테이징되지 않은 `..\CODE\BIN\master4.0m4_r1.exe`를 가리킨다**
   (입력만 복사하라는 지시 범위 때문). 실행 전에 그 줄을 절대경로
   `D:\DeCART_MASTER\BIN\master4.0m4_r1.exe`로 고치거나 `CODE\BIN` 형제 폴더를 만들어야 한다.
3. **`wmic`은 181에 없다.** wmic을 가정한 스크립트는 실패한다. CIM을 쓴다.

---

## 3. A. 238 런북 — `~/gates/block54.sh`

### 3.1 사전조건 (block54.sh를 놓기 전에 사람이 한 번 한다)

> **상태 (2026-09-03): (a)(b)(c-1)은 이미 완료되어 있고 block54의 precheck가 `fail=0`으로
> 통과한다.** 아래 절차는 재현·감사용으로 남긴다. 실제로 놓인 것:
>
> * `~/gates/ismr/rasbery_gpu/{tools,src}` — 커밋 `d2b675d`의 `tools/*.py` 138개와
>   `src/*.{h,cpp,cu}` 119개. `src`가 필요한 이유는 `tools/case_key.py`가 `src/Driver.h`,
>   `src/IO.cpp`, `src/XSSet.cpp`, `src/FidelityPreset.h`, `src/CudaXsReconBackend.cu`,
>   `src/main.cpp`를 **읽어서** arm 목록과 경로 정규화를 얻기 때문이다(복사본을 두지 않는다는
>   그 파일의 설계). 빌드는 하지 않았다.
> * `~/gates/ismr/higa_cy01/higa_depf_01.sum` (84,387 B) 와
>   `~/gates/ismr/higa_cy01/higa_mas_ppi_boc.txt` (11,571,055 B, 로컬
>   `MAS_PPI.SMF_01_0000.00` = BOC) — 앵커 C의 핀 비교 대상.
> * `~/gates/ismr/test_7`의 덱 5종을 §1.3대로 `mirror: false`로 뒤집었다(5줄 전부 확인).
> * `~/gates/ismr` 합계 106 MB.
>
> 검증까지 마친 것: `case_key.py` import OK, `make_ismr_screening_set.py`가 수정된 CY02에서
> **24개를 생성하고 24개 deck digest가 전부 서로 다르다**(rod 12 / shuffle 8 / boron 4),
> `gate_b_pin_rms.py`가 `--lattice/--origin/--half-pin-correct/--col-labels`를 노출한다.
> 남은 것은 §3.1 (c-2)의 **HIGA CHIFFON 3.0.0 라이브러리 생성**뿐이고, 그것은 block54가
> 스스로 단계 0에서 한다(RASBERY 바이너리를 쓰지만 GPU는 쓰지 않는다).

```sh
eval $(awk '/Host gpu2-6000/{f=1} f&&/HostName/{h=$2} f&&/Port/{p=$2} f&&/User/{u=$2} f&&/^ *$/&&f{exit} END{print "H="h";P="p";U="u}' ~/.ssh/config)
```

**(a) 새 도구가 담긴 소스 트리를 238에 놓는다.**
238의 어떤 트리에도 `tools/gate_b_envelope.py`가 **없다** — 모두 WP24 이전이다. 그리고
`tools/case_key.py`는 `src/Driver.h`와 `src/IO.cpp`를 읽으므로 tools 디렉터리만으로는 안 된다.
**소스 체크아웃이 필요하다**(빌드는 필요 없다):

```sh
# 워크스테이션에서 먼저 브랜치를 푸시한다 (양쪽 remote가 같은 GitHub 저장소다)
git push origin codex/exact-throughput-campaign

ssh -o BatchMode=yes -p $P $U@$H '
  set -e
  cd ~/gpu_dispatch_test_2c64f24
  git fetch origin codex/exact-throughput-campaign
  git worktree add ~/gates/ismr/rasbery_gpu d2b675d
  ls ~/gates/ismr/rasbery_gpu/tools/gate_b_envelope.py \
     ~/gates/ismr/rasbery_gpu/tools/make_ismr_screening_set.py
'
```

**(b) 덱 5종의 fold 규약을 고친다 (§1.3).**

```sh
ssh -o BatchMode=yes -p $P $U@$H '
  cd ~/gates/ismr/test_7
  for f in i-SMR_CY0*.json cy02_step1.json; do sed -i "s/\"mirror\": true/\"mirror\": false/" "$f"; done
  grep -H "\"mirror\"" *.json
'
```

**(c) 앵커 C(HIGA)에 필요한 두 자산을 마저 올린다.**
현재 `~/gates/ismr/higa_cy01/`에는 json 3개뿐이다. MASTER 기준 산출물과 3.0.0 라이브러리가 없다.

```sh
# c-1. 로컬 → 238: MASTER .sum 과 BOC 핀
scp -P $P "C:/Users/MK/Desktop/CT&RPL/4_Codes/0_GPU/iSMR_HIGA/MASTER/ISMR-CY01 -_ARO (REF)/depf_01.sum" \
    $U@$H:~/gates/ismr/higa_cy01/higa_depf_01.sum
scp -P $P "C:/Users/MK/Desktop/CT&RPL/4_Codes/0_GPU/iSMR_HIGA/MASTER/ISMR-CY01 -_ARO (REF)/MAS_PPI.SMF_01_0000.00" \
    $U@$H:~/gates/ismr/higa_cy01/higa_mas_ppi_boc.txt
```

```sh
# c-2. 238에서 HIGA CHIFFON 3.0.0 라이브러리 생성.
#      ~/ismr_bench_20260810/input/higa/ismr_higa.h5 는 2.4.0 이라 현 바이너리가 abort 한다.
#      hgc 30개가 ~/ismr_bench_20260810/input/higa/hgc 에 있다(확인됨).
#      ismr_higa_xs.json 의 `settings` 블록을 지우고 넣어야 한다(3.0.0은 붕소 축퇴를 자동 검출).
#      *** 이 단계는 GPU를 쓰지 않지만 RASBERY 바이너리를 실행한다.
#          체인이 도는 동안에는 하지 말고 block54 안에서 하거나 체인 종료 후 수동 실행한다. ***
```

**(d) 사전조건 점검** — block54.sh가 시작할 때 스스로 다시 확인한다(아래 PRECHECK 블록).

### 3.2 `~/gates/block54.sh` — 전문

```bash
#!/bin/bash
# Block 54: i-SMR accuracy (anchors A/B/C) + single-run speed + 24-case batch
# throughput, GPU RASBERY vs MASTER.  Runs AFTER block53_DONE.
#
# Anchors, and what each one is FOR (docs/ISMR_MASTER_COMPARISON_RUNBOOK_20260903_KO.md):
#   A  i-SMR_CY01 step1 ARO, ONE statepoint, vs depf_01.sum row 1 (SEARCH=1).
#      *** CY01 rows 2+ are rod-critical in MASTER and must NOT be quoted as a
#          keff comparison.  Only efpd 0.000 of cmp_CY01 is anchor A. ***
#   B  CY02/03/04 rod-critical, WHOLE cycle.  The metric is delta_rod_cm_<BANK>,
#      not keff: MASTER prints K-EFF == 1.000000 and both sides state 0.00 ppm,
#      so the two scalar columns are a search residual and a constant minus
#      itself.  Bars: per-bank max|d| <= 5 cm, rms <= 2 cm.
#   C  iSMR_HIGA CY01, 24 steps.  The ONLY pin-comparable anchor: the built-in
#      depf_0N.sum have PPI=0 at every step, so they carry no pin data at all.
#
# The chain order CY01 -> CY02 -> CY03 -> CY04 is load-bearing: 02/03/04 restart
# off their predecessor, and the decks were just re-folded to `mirror: false`
# (worth up to 7.5 pcm on 02-04), so every restart file has to be regenerated
# under the new fold in one pass.
set -u

G=/home/kmk/gates/block54
W=/home/kmk/gates/ismr/rasbery_gpu          # SOURCE tree at d2b675d (tools only)
BT=/home/kmk/gpu_dispatch_test_b69d7d4      # tree the BINARY and the batch runner come from
B=$BT/build_b69d7d4/RASBERY                 # the tip block51_v2/52/53 are measuring.
# ^ If the chain rebuilds before block54 fires, point both lines at the new tree
#   and re-run the precheck: an i-SMR number taken on a different binary than the
#   throughput campaign's tip cannot be quoted beside it.
I=/home/kmk/gates/ismr
LIB=$I/i-SMR_Validation.h5
D7=$I/test_7
HIGA=$I/higa_cy01
PY=python3.11

mkdir -p "$G"
exec > >(tee "$G/self_run.log") 2>&1
# Every exit path leaves a marker.  The poller reads the markers, not the log:
# a precheck refusal that left no file at all would look like "still running"
# forever.  block54_DONE is touched at the very end; anything short of that --
# the precheck refusal below, an unhandled failure, a kill -- lands here.
trap '[ -e /home/kmk/gates/block54_DONE ] || touch /home/kmk/gates/block54_FAILED' EXIT
rm -f /home/kmk/gates/block54_DONE /home/kmk/gates/block54_FAILED
echo "[block54] start $(date -u +%FT%TZ) binary=$B tools=$W"

##############################################################################
# PRECHECK.  Every one of these has been a real absence on this host.
##############################################################################
fail=0
for f in "$B" "$LIB" "$W/tools/gate_b_envelope.py" "$W/tools/compare_master_rasbery.py" \
         "$W/tools/gate_b_pin_rms.py" "$W/tools/gate_a_compare.py" \
         "$W/tools/make_ismr_screening_set.py" "$BT/tools/run_multi_gpu_batch.py" \
         "$D7/i-SMR_CY01.json" "$D7/Reference_output/depf_01.sum" \
         "$HIGA/iSMR_HIGA_CY01.json"; do
  [ -e "$f" ] || { echo "[block54] MISSING $f"; fail=1; }
done
# The fold convention is a PRECONDITION, not a preference: a mirror-folded deck
# scored against these rotational references is a 7.5 pcm error nobody can see.
if grep -l '"mirror": true' $D7/i-SMR_CY0*.json $D7/cy02_step1.json 2>/dev/null | grep -q .; then
  echo "[block54] *** decks still say mirror:true -- see runbook Sec 1.3 ***"; fail=1
fi
[ "$fail" = "0" ] || { echo "[block54] PRECHECK FAILED"; exit 1; }
echo "[block54] precheck ok"

##############################################################################
# Environment.  v5 PROD + v6 additions (runbook Sec 3.1 / plan Sec 3.1).
# CMFD_BLOCK=64 is SINGLE-DECK ONLY and OUTER_SEGMENT_V2 is dropped from the
# batch arm: measured flat and -1.0 % respectively in the batch shape.
##############################################################################
BASE=(PATH=$PATH HOME=$HOME LD_LIBRARY_PATH=/home/kmk/codex/tools/gcc13/lib
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID)

V6=(RASBERY_PPR_MODE=master RASBERY_PC_MODE=decart RASBERY_GPU=1
RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 RASBERY_GPU_NODAL=1
RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1
RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8 RASBERY_GPU_WIEL_FOLD=chunked
RASBERY_GPU_XE=1 RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000
RASBERY_STAGED_LOOSE_SETTLE=1 RASBERY_OMP_THREADS=12 RASBERY_GPU_CRAM=1
RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_OUTER_GRAPH=1
RASBERY_GPU_MICX_RESIDENT=1 RASBERY_GPU_XFER_ELIDE=1 RASBERY_GPU_OUTER_SEGMENT_V2=1)

SINGLE=("${V6[@]}" RASBERY_GPU_CMFD_BLOCK=64)
BATCH=(RASBERY_PPR_MODE=master RASBERY_PC_MODE=decart RASBERY_GPU=1
RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 RASBERY_GPU_NODAL=1
RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1
RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8 RASBERY_GPU_WIEL_FOLD=chunked
RASBERY_GPU_XE=1 RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000
RASBERY_STAGED_LOOSE_SETTLE=1 RASBERY_OMP_THREADS=12 RASBERY_GPU_CRAM=1
RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_OUTER_GRAPH=1
RASBERY_GPU_MICX_RESIDENT=1 RASBERY_GPU_XFER_ELIDE=1)
# The CPU twin is the SAME arm with the device off, so a difference is the
# device and not the configuration.
CPU=("${SINGLE[@]}")

##############################################################################
# Helpers (same shapes as block52.sh, so the logs read the same way).
##############################################################################
wait_gpu0_free() {
  local n=0
  while true; do
    if [ -z "$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader -i 0 2>/dev/null | grep -v '^nvidia-cuda-mps-server$')" ]; then n=$((n+1)); else n=0; fi
    [ $n -ge 3 ] && break
    echo "[wait0] $(date -u +%T) GPU0 busy"; sleep 30
  done
}
timed_run() {
  local log=$1; shift
  local t0 t1 rc
  t0=$(date +%s.%N); "$@" > "$log" 2>&1; rc=$?; t1=$(date +%s.%N)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}'
  return $rc
}
median3() { printf '%s\n' "$1" "$2" "$3" | sort -n | sed -n 2p; }

##############################################################################
# 0.  HIGA CHIFFON 3.0.0 library.  ~/ismr_bench_20260810/input/higa/ismr_higa.h5
#     is 2.4.0 and the current binary aborts on it.  The `settings` block has to
#     come out first: 3.0.0 detects the degenerate boron branch by itself.
##############################################################################
HLIB=$G/ismr_higa_300.h5
if [ ! -f "$HLIB" ]; then
  $PY - "$HIGA/ismr_higa_xs.json" "$G/ismr_higa_xs_nosettings.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8-sig"))
d.pop("settings", None)
json.dump(d, open(sys.argv[2], "w", encoding="utf-8"), indent=2)
print("settings block removed" if True else "")
PYEOF
  ( cd "$HIGA" && env -i "${BASE[@]}" "$B" \
      --chiffoni "$G/ismr_higa_xs_nosettings.json" --chiffono "$HLIB" ) \
      > "$G/higa_lib.log" 2>&1
  echo "[block54] HIGA 3.0.0 library rc=$? -> $HLIB"
fi

##############################################################################
# 1.  ACCURACY, anchors A and B.  Chain order, one GPU run per cycle, then two
#     more repeats of CY02 for determinism, then the CPU twin.
##############################################################################
AW=$G/acc; mkdir -p "$AW"; cd "$AW"
cp "$D7"/i-SMR_CY0*.json "$D7/cy02_step1.json" .
cp -r "$D7/Reference_output" .
ln -sf "$LIB" ./i-SMR_Validation.h5
wait_gpu0_free

for N in 01 02 03 04; do
  wall=$(timed_run "$AW/log_CY${N}_gpu.txt" env -i "${BASE[@]}" "${SINGLE[@]}" \
    RASBERY_STATEPOINT_TELEMETRY=1 "$B" --rasi "i-SMR_CY$N.json" --raso "out_CY${N}_gpu.h5")
  echo "acc arm=gpu cycle=$N wall=${wall}s rc=$?"
done

# GPU determinism: THREE runs of the same deck, h5diff between all three pairs.
# A 40.6 pcm GPU non-reproducibility was observed on CY02 step1 on this host
# before, so this is a precondition of quoting any accuracy number, not a nicety.
for r in 2 3; do
  wall=$(timed_run "$AW/log_CY02_det$r.txt" env -i "${BASE[@]}" "${SINGLE[@]}" \
    "$B" --rasi i-SMR_CY02.json --raso "out_CY02_det$r.h5")
  echo "acc arm=det cycle=02 run=$r wall=${wall}s rc=$?"
done
for pair in "gpu:det2" "gpu:det3" "det2:det3"; do
  a=${pair%%:*}; b=${pair##*:}
  h5diff -c "out_CY02_${a}.h5" "out_CY02_${b}.h5" > "$AW/h5diff_${a}_${b}.txt" 2>&1
  echo "[block54] h5diff CY02 $a vs $b lines=$(wc -l < "$AW/h5diff_${a}_${b}.txt") (expect 0)"
done

# CPU twin (RASBERY_GPU=0), same arm otherwise, for Gate A.
for N in 01 02; do
  wall=$(timed_run "$AW/log_CY${N}_cpu.txt" env -i "${BASE[@]}" "${CPU[@]}" RASBERY_GPU=0 \
    "$B" --rasi "i-SMR_CY$N.json" --raso "out_CY${N}_cpu.h5")
  echo "acc arm=cpu cycle=$N wall=${wall}s rc=$?"
done

echo "[block54] === GATE A (GPU vs CPU twin) ==="
for N in 01 02; do
  $PY "$W/tools/gate_a_compare.py" "out_CY${N}_cpu.h5" "out_CY${N}_gpu.h5" \
    > "$G/gate_a_CY$N.log" 2>&1
  echo "gate_a CY$N rc=$?"; tail -12 "$G/gate_a_CY$N.log"
done

##############################################################################
# 2.  GATE B scalars + ROD (anchors A and B).
#     --efpd-offset auto: CY02/03/04 restart, so RASBERY carries cycle-1 burnup
#     forward while depf_0N.sum restarts its EFPD column at 0.
##############################################################################
echo "[block54] === GATE B vs MASTER ==="
for N in 01 02 03 04; do
  OFF=auto; [ "$N" = "01" ] && OFF=0
  $PY "$W/tools/compare_master_rasbery.py" "Reference_output/depf_$N.sum" \
      "out_CY${N}_gpu.h5" -o "$G/cmp_CY$N" --efpd-offset $OFF \
      --rod-max-cm 5.0 --rod-rms-cm 2.0 > "$G/cmp_CY$N.log" 2>&1
  echo "compare CY$N rc=$?"
  tail -30 "$G/cmp_CY$N.log"
done
echo "[block54] *** ANCHOR A IS cmp_CY01 ROW efpd=0.000 ONLY. MASTER's CY01"
echo "[block54]     rows 2+ are rod-critical (SEARCH=5); do not quote them. ***"
echo "[block54] *** ANCHOR B IS the delta_rod_cm_* columns of cmp_CY02/03/04,"
echo "[block54]     not delta_pcm and not delta_ppm. ***"
echo "[block54] === boron axis check (both sides must be 0.00 ppm all cycle) ==="
for N in 02 03 04; do
  echo -n "CY$N max|delta_ppm| = "; grep -o 'max|delta_ppm| = [0-9.]*' "$G/cmp_CY$N.log" | head -1
done

##############################################################################
# 3.  ACCURACY, anchor C: HIGA CY01, 24 steps, and the ONLY pin comparison.
##############################################################################
HW=$G/higa; mkdir -p "$HW"; cd "$HW"
cp "$HIGA/iSMR_HIGA_CY01.json" .
if [ -f "$HLIB" ]; then
  wait_gpu0_free
  wall=$(timed_run "$HW/log_higa_gpu.txt" env -i "${BASE[@]}" "${SINGLE[@]}" \
    RASBERY_STATEPOINT_TELEMETRY=1 "$B" --rasi iSMR_HIGA_CY01.json --raso out_higa_gpu.h5)
  echo "acc arm=gpu cycle=higa wall=${wall}s rc=$?"
  wall=$(timed_run "$HW/log_higa_cpu.txt" env -i "${BASE[@]}" "${CPU[@]}" RASBERY_GPU=0 \
    "$B" --rasi iSMR_HIGA_CY01.json --raso out_higa_cpu.h5)
  echo "acc arm=cpu cycle=higa wall=${wall}s rc=$?"
  $PY "$W/tools/gate_a_compare.py" out_higa_cpu.h5 out_higa_gpu.h5 > "$G/gate_a_higa.log" 2>&1
  tail -12 "$G/gate_a_higa.log"

  if [ -f "$HIGA/higa_depf_01.sum" ]; then
    $PY "$W/tools/compare_master_rasbery.py" "$HIGA/higa_depf_01.sum" out_higa_gpu.h5 \
        -o "$G/cmp_higa" > "$G/cmp_higa.log" 2>&1
    echo "compare higa rc=$?"; tail -25 "$G/cmp_higa.log"
  else
    echo "[block54] SKIP HIGA scalars -- $HIGA/higa_depf_01.sum not staged (runbook Sec 3.1c)"
  fi

  # THE PIN GATE.  --lattice 17 because an i-SMR MAS_PPI is 24 x 17 x 17 = 6936
  # numbers per seat and the 16x16 default reshapes 6912 of them into 27
  # imaginary planes.  --origin E5 because the i-SMR full-core PPI centre is E5,
  # not KNGR's J9.  --half-pin-correct because the two cut lines hold half
  # assemblies whose cut-line pins carry half the power MASTER prints for the
  # same pins (9.513 % / 50.14 % uncorrected vs 4.663 % / 24.83 % corrected on
  # this very anchor).  The column alphabet falls back to plain A-Z by itself:
  # i-SMR's ninth column really is `I`, which the PWR alphabet skips.
  if [ -f "$HIGA/higa_mas_ppi_boc.txt" ]; then
    $PY "$W/tools/gate_b_pin_rms.py" out_higa_gpu.h5 "$HIGA/higa_mas_ppi_boc.txt" \
        --lattice 17 --origin E5 --half-pin-correct --envelope production \
        > "$G/gate_b_pin_higa.log" 2>&1
    echo "gate_b_pin higa rc=$?"; cat "$G/gate_b_pin_higa.log"
    # The uncorrected reading, REPORTED so the artefact stays visible and nobody
    # rediscovers it as a physics finding.
    $PY "$W/tools/gate_b_pin_rms.py" out_higa_gpu.h5 "$HIGA/higa_mas_ppi_boc.txt" \
        --lattice 17 --origin E5 --envelope production \
        > "$G/gate_b_pin_higa_uncorrected.log" 2>&1
    grep -E "WARNING|^BOC pin" "$G/gate_b_pin_higa_uncorrected.log"
  else
    echo "[block54] SKIP HIGA pin -- $HIGA/higa_mas_ppi_boc.txt not staged (runbook Sec 3.1c)"
  fi
else
  echo "[block54] SKIP anchor C -- no HIGA 3.0.0 library at $HLIB"
fi

##############################################################################
# 4.  SINGLE-RUN SPEED.  warm-up 1 + 3 hot, GPU and CPU interleaved per deck so
#     a machine drift hits both arms, median of the 3 hot runs.
#     Speed-up = median(CPU) / median(GPU) on the SAME deck.
##############################################################################
SW=$G/speed; mkdir -p "$SW"; cd "$SW"
cp "$D7"/i-SMR_CY01.json "$D7"/i-SMR_CY02.json "$D7/cy02_step1.json" .
cp "$HIGA/iSMR_HIGA_CY01.json" . 2>/dev/null
ln -sf "$LIB" ./i-SMR_Validation.h5
# CY02 restarts, so it needs its predecessor's restart file beside it.
cp "$AW"/restart_*.h5 . 2>/dev/null

SPEED_DECKS=(cy02_step1.json i-SMR_CY02.json i-SMR_CY01.json)
[ -f "$HLIB" ] && SPEED_DECKS+=(iSMR_HIGA_CY01.json)

wait_gpu0_free
for r in w 1 2 3; do
  for deck in "${SPEED_DECKS[@]}"; do
    tag=$(basename "$deck" .json)
    for arm in gpu cpu; do
      if [ "$arm" = "gpu" ]; then AENV=("${SINGLE[@]}"); else AENV=("${CPU[@]}" RASBERY_GPU=0); fi
      wall=$(timed_run "$SW/${tag}_${arm}_$r.log" env -i "${BASE[@]}" "${AENV[@]}" \
        "$B" --rasi "$deck" --raso "$SW/${tag}_${arm}_$r.h5")
      echo "speed deck=$tag arm=$arm run=$r wall=${wall}s rc=$?"
    done
  done
done

echo "[block54] === SINGLE-RUN SPEED SUMMARY ==="
for deck in "${SPEED_DECKS[@]}"; do
  tag=$(basename "$deck" .json)
  for arm in gpu cpu; do
    w1=$(grep -m1 "^speed deck=$tag arm=$arm run=1 " "$G/self_run.log" | grep -oE 'wall=[0-9.]+' | cut -d= -f2)
    w2=$(grep -m1 "^speed deck=$tag arm=$arm run=2 " "$G/self_run.log" | grep -oE 'wall=[0-9.]+' | cut -d= -f2)
    w3=$(grep -m1 "^speed deck=$tag arm=$arm run=3 " "$G/self_run.log" | grep -oE 'wall=[0-9.]+' | cut -d= -f2)
    if [ -n "$w1" ] && [ -n "$w2" ] && [ -n "$w3" ]; then
      eval "MED_${arm}=$(median3 "$w1" "$w2" "$w3")"
    else
      eval "MED_${arm}="
    fi
  done
  if [ -n "${MED_gpu:-}" ] && [ -n "${MED_cpu:-}" ]; then
    echo "speed_summary deck=$tag median_gpu_s=$MED_gpu median_cpu_s=$MED_cpu speedup=$(awk -v c="$MED_cpu" -v g="$MED_gpu" 'BEGIN{printf "%.2f", c/g}')x"
  else
    echo "speed_summary deck=$tag INCOMPLETE"
  fi
done

##############################################################################
# 5.  BATCH THROUGHPUT.  24-case i-SMR screening set, 8xM8 + MPS, median-of-3.
#     The set is generated here, not checked in: it is derived from CY02 and its
#     distinctness has to be verified against THIS deck.  The generator refuses
#     to write a set with a case_key collision -- a duplicate deck inside a
#     throughput measurement is a cache hit reported as a solved case.
##############################################################################
BW=$G/batch; mkdir -p "$BW/out"
cd "$W"
$PY tools/make_ismr_screening_set.py "$D7/i-SMR_CY02.json" -o "$BW/decks" \
    --manifest "$BW/ismr24.txt" > "$G/make_set.log" 2>&1
rc=$?
cat "$G/make_set.log"
if [ "$rc" != "0" ]; then
  echo "[block54] screening set generation FAILED -- no batch number"
else
  # The generated decks name the library by the base deck's relative name, so
  # the library has to sit beside them.
  ln -sf "$LIB" "$BW/decks/i-SMR_Validation.h5"
  cp "$AW"/restart_*.h5 "$BW/decks/" 2>/dev/null
  $PY - "$BW/ismr24.txt" "$BW/out" <<'PYEOF'
import sys, pathlib
mf, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = []
for line in mf.read_text().splitlines():
    if not line.strip():
        continue
    deck, _h5 = line.split()[:2]
    lines.append(f"{deck} {out / (pathlib.Path(deck).stem + '.h5')}")
mf.write_text("\n".join(lines) + "\n")
print(f"manifest rewritten: {len(lines)} jobs -> {out}")
PYEOF
  BSETS=(); for kv in "${BATCH[@]}"; do BSETS+=(--set "$kv"); done
  for r in 1 2 3; do
    wait_gpu0_free
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,clocks.sm,power.draw,temperature.gpu \
      --format=csv -l 2 -i 0 > "$BW/run${r}_telemetry.csv" &
    TP=$!
    env -i PATH=$PATH HOME=$HOME LD_LIBRARY_PATH=/home/kmk/codex/tools/gcc13/lib \
      CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
      timeout 7200 $PY "$BT/tools/run_multi_gpu_batch.py" --gpus 0 --procs-per-gpu 8 \
        --batch-width 8 --claim auto --result light --jobs "$BW/ismr24.txt" \
        --cwd "$BW/decks" --workdir "$BW/run$r" --pin taskset --mps "${BSETS[@]}" \
        -- "$B" > "$BW/run$r.log" 2>&1
    echo "[block54] batch run$r rc=$?"
    kill $TP 2>/dev/null
    grep -E "MULTI_GPU..(TOTAL|FAIL)" "$BW/run$r.log" | cut -c1-500
  done
  c1=$(grep -m1 -o '"cases_per_hour":[0-9.]*' "$BW/run1.log" | cut -d: -f2)
  c2=$(grep -m1 -o '"cases_per_hour":[0-9.]*' "$BW/run2.log" | cut -d: -f2)
  c3=$(grep -m1 -o '"cases_per_hour":[0-9.]*' "$BW/run3.log" | cut -d: -f2)
  if [ -n "$c1" ] && [ -n "$c2" ] && [ -n "$c3" ]; then
    echo "[block54] BATCH 8xM8+MPS i-SMR24 median cases_per_hour=$(median3 "$c1" "$c2" "$c3") (runs: $c1 $c2 $c3)"
  else
    echo "[block54] BATCH cases_per_hour NOT PARSED (runs: '$c1' '$c2' '$c3')"
  fi
fi

##############################################################################
# 6.  Summary and the DONE marker.
##############################################################################
echo "[block54] === SUMMARY ==="
grep -E "^acc |^speed_summary |^gate_a |^compare |^gate_b_pin |^\[block54\] (h5diff|BATCH)" "$G/self_run.log"
echo "[block54] end $(date -u +%FT%TZ)"
touch /home/kmk/gates/block54_DONE
```

### 3.3 체인에 잇기 (`block53_DONE` 이후)

현재 poller(`wait_clean_then_run_51.sh`)는 block53에서 끝난다. block54를 이어 붙이려면 별도의
얇은 poller를 detach로 띄운다. **기존 poller나 block51/52/53 스크립트는 건드리지 않는다.**

```bash
cat > ~/gates/wait_block53_then_run_54.sh <<'EOF'
#!/bin/bash
# Chain tail: run block54 once block53 reaches DONE.  Same clean-check policy as
# wait_clean_then_run_51.sh -- loadavg<4, GPU1 idle, other users' jobs absent,
# 3 consecutive 60 s clean checks -- applied AFTER block53_DONE appears, so this
# never races the block that is still running.
set -u
G=~/gates
LOG=$G/wait_block53_then_run_54.log
NEED=3
consec=0
echo "[waitb53] poller start $(date -u +%FT%TZ) pid=$$" >> "$LOG"

while [ ! -f "$G/block53_DONE" ]; do
  if [ -f "$G/block53_FAILED" ]; then
    echo "[waitb53] block53 FAILED -- block54 not started" >> "$LOG"; exit 1
  fi
  sleep 60
done
echo "[waitb53] block53_DONE seen at $(date -u +%FT%TZ)" >> "$LOG"

while true; do
  load1=$(cut -d' ' -f1 /proc/loadavg)
  load_ok=$(awk -v l="$load1" 'BEGIN{print (l<4.0)?1:0}')
  gpu1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 2>/dev/null)
  excl=$(ps -eo args 2>/dev/null | grep -E 'lpopt_ws|v31_refit|train_v3|foldacct|a5lr|scratch/' | grep -v grep)
  if [ "$load_ok" = "1" ] && [ -z "$gpu1" ] && [ -z "$excl" ]; then consec=$((consec+1)); else consec=0; fi
  echo "[waitb53] $(date -u +%FT%TZ) load1=$load1 gpu1=$([ -n "$gpu1" ] && echo busy || echo idle) consec=$consec/$NEED" >> "$LOG"
  [ "$consec" -ge "$NEED" ] && break
  sleep 60
done

echo "[chain54] launching block54 at $(date -u +%FT%TZ)" >> "$LOG"
bash "$G/block54.sh" > "$G/block54_run.log" 2>&1
rc=$?
echo "[chain54] block54 rc=$rc at $(date -u +%FT%TZ)" >> "$LOG"
if [ "$rc" -eq 0 ]; then touch "$G/block54_DONE"; else touch "$G/block54_FAILED"; fi
EOF
chmod +x ~/gates/wait_block53_then_run_54.sh ~/gates/block54.sh
nohup ~/gates/wait_block53_then_run_54.sh > /dev/null 2>&1 &
```

`block54.sh`가 마지막 줄에서 스스로 `block54_DONE`을 만들고, poller도 rc==0일 때 만든다
(둘 다 있어도 무해하다 — `touch`는 멱등). rc!=0이면 `block54_FAILED`가 남는다.

---

## 4. 합격선과 판독법

| 항목 | 도구 | 합격선 | 근거 |
|---|---|---|---|
| Gate A (GPU vs CPU 쌍) | `gate_a_compare.py` | 5 pcm / 5 ppm / 0.01 AO / 1 % 핀 | A2 스크리닝 기본값 |
| GPU 결정성 | `h5diff -c` × 3쌍 | **lines = 0** | 과거 CY02 step1에서 40.6 pcm 비재현성 관측 이력 |
| 앵커 A (CY01 step1) | `cmp_CY01` efpd=0.000 행 | production envelope (2.0 pcm / 15.4 ppm / 0.013 AO) | `gate_b_envelope.py` |
| 앵커 B (CY02/03/04) | `cmp_CY0N`의 `delta_rod_cm_*` | bank당 max|Δ| ≤ 5 cm, rms ≤ 2 cm | 계획 §2 |
| 앵커 C 스칼라 (HIGA) | `cmp_higa` | production envelope | 〃 |
| 앵커 C 핀 (HIGA BOC) | `gate_b_pin_rms.py --lattice 17 --origin E5 --half-pin-correct` | production 0.24 % rms / 0.80 % max | 〃 |
| 배치 처리량 | `run_multi_gpu_batch.py` 8xM8+MPS | 비교 대상은 181 MASTER W16 cases/h | 계획 §4 |

**판독할 때 틀리기 쉬운 세 가지:**

1. **CY01은 step1 한 점만 앵커다.** MASTER는 row 2부터 봉으로 임계를 잡는다. 덱을 rod-search로
   고쳐 맞추지 말 것 — 재시작 체인 전체가 무효화된다.
2. **CY02–04의 지표는 keff가 아니라 rod cm이다.** `summary/rod_step`은 cm이 아니다(관측값
   0.78–2.13). 변환은 `pos_cm = 240.0 − rods/insertions` 하나뿐이다.
   두 사이드 다 전 스텝 0.00 ppm이므로 **i-SMR에는 임계붕소 비교점이 없다** — 붕소 축은
   "0 ppm 유지 확인"으로만 보고한다. `compare_master_rasbery.py`는 이 컬럼을 `pinned`로
   표시하고, 판정 가능한 컬럼이 pinned뿐이면 PASS 대신 NOT SCORED(exit 2)를 낸다.
3. **핀 수치는 `--half-pin-correct` 유무로 2배 차이 난다** (§1.5). 보정 없는 9.513 %는
   물리가 아니라 fold의 산술이다. block54는 두 값을 모두 찍되 판정은 보정본으로 한다.

참고 실측(아카이브 `rf_b2/out_CY02.h5` × `depf_02.sum`, 이번 도구로 스모크 실행):
21 상태점 조인, `max|delta_pcm| = 4.814`, `max|delta_ao| = 0.117`, `max|delta_ppm| = 0.000`(pinned),
`delta_rod_cm`: R3 max 39.522 / rms 29.535, R4 max 39.522 / rms 30.606, R2 max 13.346 / rms 3.933,
S1/S2/S3/S4/S5/S6/R1은 양쪽 모두 240 cm 고정(pinned). **이것은 구 라이브러리로 돈 회전-fold
실험 산출물이므로 FAIL이 정상이며, 도구가 끝까지 도는지와 컬럼이 의미를 갖는지를 보여주는
용도다. 수용 수치로 인용하지 말 것.**

---

## 5. B. 181 MASTER 런북

### 5.1 무엇을 왜 재실행하는가

`.sum`으로 이미 존재하는 것: keff, AO, FQ/FR, 집합체 출력·연소도, 봉 위치. **재실행이 필요한
것은 두 가지뿐이다.**

1. **MASTER 벽시계.** `.sum`에 타이밍 줄이 없다. 기존에 인용되던 30.6 s와 43.8–54.6 s 두 값은
   서로 모순이고 HIGA 덱의 것이다 — **인용 금지.** 새로 측정한다.
2. **내장 덱의 핀(MAS_PPI).** `Reference_output/depf_0N.sum`은 전 스텝 `PPI=0`이라 핀 데이터가
   아예 없다. **단, 이것은 지금 실행할 수 없다** — 181에 i-SMR용 MASTER XS(MAS_XSL/MAS_HFF)가
   없기 때문이다(§5.5). 지금 실행 가능한 것은 **HIGA CY01**뿐이고, HIGA는 MAS_PPI가 이미
   로컬에 24개 다 있으므로 핀 재생성도 필요 없다. 즉 **지금 181에서 할 일은 벽시계 측정 하나다.**

### 5.2 실행 전 조건 (전부 충족되어야 한다)

```bat
:: 1) MASTER/DeCART 계열 프로세스가 하나도 없어야 한다
ssh -o BatchMode=yes 181 "tasklist | findstr /i \"master decart prolog promarx\""
::    -> 무결과여야 한다. 작업 시점에는 master4.0m4_r1.exe 가 16개 돌고 있었다(§2.3).

:: 2) CPU 부하 < 10 %.  wmic 은 이 이미지에 없다 -- CIM 을 쓴다.
ssh -o BatchMode=yes 181 "powershell -NoProfile -Command \"(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average\""

:: 3) 출력은 C: 로 낸다.  D: 여유 7.1 GB / 930 GB 로 사실상 가득 찼다.
```

### 5.3 실행 준비 (스테이징된 입력을 실행 가능하게 만든다)

입력은 §2.2대로 이미 `C:\Users\kmk\CodexBench\ismr_master_20260903\`에 있다. 남은 것은
`.bat`의 상대경로 문제 하나다(§2.3-2).

```bat
:: exe 를 형제 CODE\BIN 에 두어 depf_01.bat 를 그대로 쓴다 (권장: .bat 를 수정하지 않는다)
ssh -o BatchMode=yes 181 "mkdir C:\Users\kmk\CodexBench\ismr_master_20260903\CODE\BIN"
ssh -o BatchMode=yes 181 "copy D:\DeCART_MASTER\BIN\master4.0m4_r1.exe C:\Users\kmk\CodexBench\ismr_master_20260903\CODE\BIN\"
ssh -o BatchMode=yes 181 "dir C:\Users\kmk\CodexBench\ismr_master_20260903\CODE\BIN"
```

대안(형제 폴더를 만들지 않는 쪽): `depf_01.bat`의 마지막 줄
`..\CODE\BIN\master4.0m4_r1.exe`를 **절대경로**로 바꾼다. 스테이징된 `.bat`는 상대경로만
가지고 있고 그 형제 `CODE\BIN`은 복사되지 않았으므로, 둘 중 하나를 반드시 해야 실행된다.

```bat
ssh -o BatchMode=yes 181 "powershell -NoProfile -Command \"$p='C:\Users\kmk\CodexBench\ismr_master_20260903\ISMR-CY01_ARO_REF\depf_01.bat'; (Get-Content $p) -replace '\.\.\\CODE\\BIN\\master4\.0m4_r1\.exe', 'D:\DeCART_MASTER\BIN\master4.0m4_r1.exe' | Set-Content -Encoding ascii $p\""
ssh -o BatchMode=yes 181 "type C:\Users\kmk\CodexBench\ismr_master_20260903\ISMR-CY01_ARO_REF\depf_01.bat"
::   -> 마지막 줄이 D:\DeCART_MASTER\BIN\master4.0m4_r1.exe 여야 한다
```

**`master30_release.exe`를 쓰지 않는다.** 생산 exe는
`D:\DeCART_MASTER\BIN\master4.0m4_r1.exe`(SHA256 418362B3…0848B2, v4.00 MOD3, 단일 스레드)다.

### 5.4 타이밍 프로토콜

`Measure-Command`를 쓰지 않는다. 기존 receipt 하네스
(`C:\Users\kmk\CodexBench\codex_gpu_master_compare_20260728\master_reference_20260728T053326Z`,
`run_receipt.json` 스키마, `processor_affinity 0x1`) 레이아웃을 그대로 복제한다.

**단일 (single):** 케이스 폴더를 웨이브마다 새로 만들고, 1 프로세스를 코어 0에 고정
(`start /affinity 1`), **warm-up 1회 + 측정 10회, median**.

```bat
ssh -o BatchMode=yes 181 "cd /d C:\Users\kmk\CodexBench\ismr_master_20260903\ISMR-CY01_ARO_REF & depf_01.bat"
:: 각 반복마다: 케이스 폴더 정리 -> depf_01.bat -> run_receipt.json{wall_s, sha256(MAS_SUM), exe sha256}
```

**W16:** 16 프로세스 × `threads_per_task=1`, 각 프로세스는 자기 케이스 폴더에서 실행
(MAS_INP/MAS_XSL/MAS_HFF가 CWD 고정 파일명이라 폴더를 공유하면 서로 덮어쓴다),
**warm-up 1 웨이브 + 측정 10 웨이브 → cases/h.**

**유효성 판정:** 웨이브 간 `MAS_SUM`의 sha256이 전부 동일해야 한다. 하나라도 다르면 그 웨이브의
벽시계는 버린다(같은 계산을 하지 않은 것이다).

**혼용 금지:** 7월 캠페인 하드웨어 receipt는 논리 프로세서 16개(SMT off)에서 떴고 현재는 32개다.
**APR1400 기존 곡선(L1 33.28 → L16 202.31 c/h, 6.08×)은 재베이스라인 없이 i-SMR과 섞어 쓸 수
없다.** i-SMR W16 cases/h는 i-SMR 단독으로 새로 세운다.

### 5.5 지금 181에서 할 수 없는 것 (갭)

| 갭 | 상태 | 획득 절차 |
|---|---|---|
| 내장 i-SMR MASTER XS (MAS_XSL/MAS_HFF) | 없음 (SMR_H2O는 16×16 IFBA, 나머지는 YG3) | 리포 `test/CrossSections/2_i-SMR_Validation/`의 11 FA + 11 반사체 HGC를 181로 옮겨 `prolog41m4.exe`(또는 `C:\DeCART_MASTER\BIN\PROMARX1.0m2.exe`)로 `.XSD/.FFL` 생성 → `xsgen.bat` 패턴으로 결합 |
| 내장 덱 핀(MAS_PPI) | 없음 (`.sum` 전 스텝 PPI=0) | 위 XS가 선행. 그 뒤 `depf_02.inp`에 PPI 옵션을 켜고 재실행 |
| PROMARX i-SMR 전용 반사체 XS | 없음 (현재 YG3 반사체 v3.0 재사용) | 181 `D:\codex\decart_xesm_ab_axial_20260820`의 i-SMR 축방향 반사체 DeCART2D 결과를 PROMARX1.0m2로 변환. **미해결 반사체 인접 핀 잔차(§1.5의 6.7–8.1 %, Y7X9/Y9X7)의 최우선 가설이므로 별건으로 추적한다.** |

---

## 6. 산출물

**238** (`~/gates/block54/`): `self_run.log`(전체 전사),
`cmp_CY0{1..4}.{csv,md,log}`(스칼라 + `delta_rod_cm_*`), `gate_a_CY0{1,2}.log`,
`acc/h5diff_*.txt`(결정성), `gate_b_pin_higa.log` / `gate_b_pin_higa_uncorrected.log`,
`cmp_higa.{csv,md,log}`, `speed/*.log`(warm+3hot), `batch/run{1,2,3}.log` + `*_telemetry.csv`,
`batch/decks/`(24 후보) + `batch/ismr24.txt`, `~/gates/block54_DONE`.

**181**: 케이스별 `MAS_SUM`(EDIT 2/3/5), `run_receipt.json`(`wall_s`, sha256),
W16 웨이브 요약 cases/h.

---

## 7. 미해결 (이번 커밋 범위 밖, 순서대로)

1. **중심선 쌍 평균이 비교 도구에 없다.** rotfold에서 (Y5,X_c)와 (Y_c,E)는 한 집합체의 두
   반쪽인데 `tools/plot_ismr_validation.py`는 raw 좌석을 비교한다 →
   **올바른 rotational 실행이 틀린 mirror 실행보다 나빠 보인다.** APR1400
   `out_rotfold_gates_c.txt`가 쓴 것과 같은 쌍 평균을 넣어야 한다. 핀 쪽 등가 문제는
   `--half-pin-correct`로 해결됨(§1.5).
2. **`tools/master2rasi.py`가 같은 방식으로 틀렸다.** `already_folded_core()`가 ngeo=4에
   무조건 `mirror: True`를 내고 `check_diagonal_symmetry()`가 비-전치불변 맵을 거부한다.
   이 덱들은 nsym=1을 선언하므로 `mirror: False`를 내야 하고, 대각 대칭 거부는 mirror 근사를
   정직하게 유지하려던 장치이므로 함께 없어져야 한다.
3. **`_neibrb`의 rotfold 근사** (`src/Geometry.cpp:311-316`): PPR의 3x3 핀 스텐실이 회전 이웃을
   표현하지 못해 절단선 근처 핀 재구성이 근사로 남는다. HIGA 앵커에는 영향 없음(대각 대칭).
4. **반사체 인접 핀 잔차 6.7–8.1 %** — §5.5의 PROMARX i-SMR 반사체 XS가 최우선 가설.
5. **181 유휴 대기** — §2.3-1. 유휴 확인 전 MASTER 타이밍 측정은 무효다.

## 부록 A. 181 MASTER 실측 (2026-09-03)

- 스테이징된 `ISMR_CY01_PROLOG1/MAS_XSL`(450,088 B, 3/25 수정)은 1/6 기준 출력보다 나중에 덮어써진 **절단 파일**이라 `SCANING MASTER INPUT`에서 unit 2 EOF로 중단됨(exe·입력 해시는 로컬과 동일, 줄바꿈·ssh 환경·런처 패턴 무관 — APR1400 대조 덱은 같은 세션에서 111.5 s 정상 완료).
- 복구: `xsgen.bat` 레시피 그대로 재조립(MAS_REF + A1..A8,AC `PRO_FA_*.XSD` → MAS_XSL 3,721,440 B; `.FFL` → MAS_HFF 6,358,635 B). 원본은 `.broken_*`로 보존.
- 검증: CY01 24스텝 완주, step-1 K-EFF 1.032934 vs 기준 1.032935(0.1 pcm), MAS_SUM 크기 84,387 B 기준과 동일, 10회 MAS_SUM sha256 전부 동일.
- **단일 wall(warm 1 + 10회, affinity 0x1): median 26.26 s** (min 25.55 / max 26.51).
- W16 cases/h: 미측정(16 동시 프로세스 기동이 로컬 권한 분류기에 차단 — 사용자 허용 후 재시도). 수신증: `E:\rasbery_runs\2026-09-03\ismr_master\181\receipt.txt`.

---

## 8. Anchor B reactivity protocol (block54c)

§3.2 §2의 앵커 B는 `delta_rod_cm_*` (bank cm 차이)로만 보고된다 — CY02/03/04는 양쪽 다
rod-critical(MASTER K-EFF == 1.000000, 양쪽 0.00 ppm)이라 `delta_pcm`/`delta_ppm`은 물리
정보가 없다(§3의 `compare_master_rasbery.py` 모듈 docstring 참고). 이 rod cm 격차를
**pcm 반응도**로 바꾸는 절차가 block54c다: RASBERY 쪽 로드-탐색을 끄고 MASTER가 수렴한
정확한 cm에 봉을 고정한 뒤 그 형상에서 RASBERY의 keff를 그대로 읽는다 — 붕소가 양쪽 다
0 ppm이므로 그 keff의 (k-1)/k × 1e5는 순수하게 봉 위치 격차가 가진 반응도다.

새 도구 둘 (둘 다 `tools/compare_master_rasbery.py`의 `.sum`/`.h5` 파서를 재사용하며,
`.sum` 파싱을 다시 구현하지 않는다):

* `tools/make_ismr_fixed_rod_deck.py <deck.json> <depf_0N.sum> -o <out.json>` — 스케줄의
  선두 `"standard"` 엔트리 `"search"`를 `"rod"` → `"keff"`로 강제하고(src/IO.cpp:34,487),
  통계점마다 `{"type": "rod insertion", "<BANK>": <insertion_cm>, ...}` 엔트리를
  (src/IO.cpp:502-528, 실사용 예시 `test/3-1_Colinear/Base_Rasbery.json:69`) 그 통계점
  바로 앞에 끼워 넣는다. `insertion = 240 − cm`. 덱의 통계점 전개 순서(선두 standard +
  각 depletion 엔트리의 `steps`번)와 `.sum` EDIT 1의 EFPD 오름차순 행 개수가 정확히
  같아야 하며(CY02 21==21, CY03 22==22, CY04 22==22, 2026-09-03 확인), 다르면 즉시
  거부한다 — 통계점을 하나씩 밀어서 잘못 짝짓는 것보다 낫다. R2/R4가 `.sum`에 없는
  경우에만 `apply_overlap(r3_cm)`(R2=R3+120, R4=R3-120, [0, 240] 클램프)로 대체한다;
  `.sum`이 직접 준 값이 항상 이 공식보다 우선한다.
* `tools/ismr_rod_reactivity.py <depf_0N.sum> <out_fixedrod.h5> [--orig-h5 <원래 rod-search h5>]`
  — 고정-로드 실행의 `summary/keff`를 읽어 통계점마다 `delta_rho_pcm = (k-1)/k × 1e5`를
  찍고 전체 rms/max를 낸다. `--orig-h5`를 주면 원래 rod-search 실행의 `rods/insertions`
  (`pos_cm = 240 − insertion`)와 MASTER cm의 차이도 `delta_rod_cm_<BANK>`로 같이 낸다.

로컬에서는 파이썬 도구만 돈다(`tools/test_ismr_fixed_rod_contract.py`) — 아래 GPU 실행
명령은 238에서만 돌리며, 이 세션에서는 실행하지 않았다.

```bash
# block54c: anchor B rod-position gap -> pcm, CY02 (CY03/CY04는 0N만 바꿔 그대로 반복)
cd "$AW"   # block54 §3.2의 acc 작업 디렉터리, i-SMR_CY0N.json + Reference_output/이 이미 있음

$PY "$W/tools/make_ismr_fixed_rod_deck.py" \
    "$D7/i-SMR_CY02.json" "$D7/Reference_output/depf_02.sum" \
    -o i-SMR_CY02_fixedrod.json

env -i "${BASE[@]}" "${SINGLE[@]}" RASBERY_STATEPOINT_TELEMETRY=1 "$B" \
    --rasi i-SMR_CY02_fixedrod.json --raso out_CY02_fixedrod.h5

$PY "$W/tools/ismr_rod_reactivity.py" \
    "$D7/Reference_output/depf_02.sum" out_CY02_fixedrod.h5 \
    --orig-h5 out_CY02_gpu.h5 -o "$G/rho_CY02" | tee "$G/rho_CY02.log"

# CY03: i-SMR_CY02 -> i-SMR_CY03, depf_02.sum -> depf_03.sum, out_CY02_* -> out_CY03_*, rho_CY02 -> rho_CY03
# CY04: 위와 동일하게 04로 치환
```

`$W`, `$D7`, `$G`, `$B`, `$PY`, `${BASE[@]}`, `${SINGLE[@]}`는 §3.2 block54.sh가 이미
정의한 것과 동일한 변수다(단일-실행 인자는 block54 §1의 `${SINGLE[@]}` GPU arm을 그대로
쓴다 — 배치 arm이 아니다). `out_CY0N_gpu.h5`는 block54 §1이 이미 만든 원래 rod-search
실행 산출물이므로, `--orig-h5`를 위해 다시 돌릴 필요가 없다.

---

## 9. 물리 집합체 출력/연소도 맵 비교 (§7 항목 1)

`tools/compare_assembly_maps.py` (신규)가 §7 항목 1(중심선 쌍 평균이 비교 도구에
없다)을 채운다. MASTER `.sum`의 SUMMARY EDIT 5 출력/연소도 맵을 파싱하고, RASBERY
`.h5`의 `steps/*/assembly/{power,burn}`을 90도 회전 폴드(`ndivxy=2, symdiv=true`)로
**물리 집합체** 단위로 접어(1/4맵 슬롯 (0,c)와 (c,0), c>0 는 한 물리 집합체의 두
반쪽 — `docs/ROTATIONAL_SHUFFLE_FIX_20260904_KO.md`, `src/IO.cpp:2600`의 `gather`
주석) 같은 물리 집합체끼리 대조한다. 중심(0,0)은 자신의 1/4을 그대로 쓰고
(`src/IO.cpp:2603-2606`가 자기 자신을 3번 더 회전시켜 나머지를 채우므로 분할 불필요),
`--mirror`(APR1400 KNGR)는 폴드를 끈다 — i-SMR 5개 덱은 `mirror: false`(§1)이므로
기본값(폴드 ON)을 그대로 쓴다. 상태점마다 CSV 한 줄/물리 집합체(`master`, `rasbery`,
`delta`, 두 반쪽 원값과 그 차이인 `sym_*` 정보 컬럼)와, 출력(%)·연소도(GWd/t) 맵 전체
rms/max를 MD로 낸다. 검증은 `tools/test_assembly_map_fold_contract.py`(합성 3x3 맵
및 실제 `depf_01.sum` F5=E6 스모크)와 `tools/test_ismr_tools_contract.py`의 fold
계약이 맡는다. `.sum` 파서는 `tools/compare_master_rasbery.py`를 재사용한다
(LAST-EFPD-WINS, `--efpd-offset auto` 재사용).

238에서 block54 §1이 만든 `acc/out_CY0N_gpu.h5`에 대해 (CY02/03/04는 리스타트이므로
`--efpd-offset auto`, CY01은 생략):

```sh
cd ~/gates/block54/acc
for N in 01 02 03 04; do
  OFF=auto; [ "$N" = "01" ] && OFF=0
  python3.11 ~/gates/ismr/rasbery_gpu/tools/compare_assembly_maps.py \
      ~/gates/ismr/test_7/Reference_output/depf_$N.sum \
      out_CY${N}_gpu.h5 -o ~/gates/block54/amap_CY$N --efpd-offset $OFF
done
```

이 도구는 §7 항목 1을 코드로는 닫지만, 238에서는 아직 실행하지 않았다(이 세션의
스코프는 python 도구/테스트뿐 — 솔버도 컴파일러도 이 PC에서 돌리지 않는다).
