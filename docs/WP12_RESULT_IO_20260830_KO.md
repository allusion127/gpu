# WP12 — io_wall의 나머지 절반: 핀출력 CSV 직렬화를 solver 스레드에서 내린다 (2026-08-30)

문서 메타데이터

| 항목 | 값 |
|---|---|
| 계기 | v4 후보 단일 실측(238, KNGR cycle-1, 35 상태점, 14.56 s, `[RASBERY][SPTELEM][SUMMARY]`): **io_wall ≈ 2.89 s (19.9 %)**, 그 대부분이 `result_write`. CRAM+search+TH+PPR+Xe를 **합친 것보다 크다** |
| 다음 후보 | nested_wall flatxs 1.92 s (13 %), startup init+library 0.92 s (6.3 %) — 이 문서 §5는 후자의 **계측 계획만** 담는다 |
| 구현 위치 | `src/IoWriter.h`(게이트·side task·수신증), `src/IO.cpp`(스냅샷 타입·호출부), `src/main.cpp`(수신증 인쇄) |
| 플래그 | `RASBERY_RESULT_ASYNC=1` (기본 **OFF**) |
| 등급 | **B0 예상** (§4) |
| 로컬에서 한 것 | 소스 계약 3종 통과. **이 PC에는 컴파일러가 없다 — 빌드도 실행도 하지 않았고 성능 수치도 없다** |
| 238에서만 가능한 것 | §6 전부 |

---

## 0. 한 줄 결론

**HDF5는 이미 solver 스레드 밖에 있었다. 밖에 없던 것은 핀출력 CSV다** —
`--result full` 한 케이스당 **~119 MB**의 텍스트를 `ostream <<`로 double 하나씩,
`PH_RESULT_WRITE` 스코프 안에서, solver 스레드가 직접 찍고 있었다.
WP12는 그 직렬화를 **값 스냅샷 + side task**로 만들어 이미 존재하는 writer 스레드에
넘긴다. 출력 바이트는 같은 코드가 같은 값에서 같은 순서로 만든다.

---

## 1. 결과 경로를 읽은 결과 (과제 (1)의 답)

### 1.1 어디서 재는가

| 계측 | 위치 | 무엇 |
|---|---|---|
| `PH_RESULT_ADD` | `src/Driver.h:180` 열거, `:5320` 스코프 | `IO::AddResult` — 모든 모드가 내는 패킹 비용 |
| `PH_RESULT_WRITE` | `src/Driver.h:181` 열거, `:5329`~`:5354` 스코프 | 선택된 출력 모드가 **그 위에** 얹는 비용 |
| `io_wall` | `src/Driver.h:5356` (`step_io_seconds`), `:5383`, 합산 `:5358` | 위 둘의 합. `[RASBERY][SPTELEM]`/`[SUMMARY]`가 인쇄 |

`io_start`는 `AddResult` 직전, 종료는 `WriteStepToResult`(또는 `BatchLightResult::Write`)
직후다. 즉 **io_wall은 상태점마다 닫히는 구간이고, 끝판 일괄 쓰기가 아니다.**

### 1.2 상태점별인가 끝판인가 — 상태점별이다

* `IO::OpenResult` (`src/IO.cpp:1569`) — 실행 1회. `/geometry/*` 15개 데이터셋.
* `IO::WriteStepToResult` (`src/IO.cpp:1650`) — **상태점마다 1회**.
  `steps/NNNN/assembly/*`, `steps/NNNN/axial/*`, `steps/NNNN/node/{power,burnup,kinf,flux}`,
  (`pin_info`일 때) `steps/NNNN/pin_power`, (`pin_flux`일 때) `steps/NNNN/pin_flux`,
  (`xs_info`일 때) `steps/NNNN/xs/*`, (`node_monitor`일 때) `steps/NNNN/node_monitor/*`.
* `IO::CloseResult` (`src/IO.cpp:2097`) — 실행 1회. `summary/*` 27개 + `rods/*`.

**파일은 상태점마다 다시 열리지 않는다.** `HighFive::File`은 `FileSession`이 소유하고
(`src/IO.h:63`) `OpenResult`가 만든 뒤 `CloseResult`가 닫는다. **단 하나 예외가
핀출력 CSV**로, 그것은 상태점마다 `std::ofstream`을 `ios::app`으로 새로 연다.

### 1.3 압축·청킹 — 없다

`Node::createDataSet`(`src/IoWriter.h`)는 `HighFive::DataSpace`만 만들고
`DataSetCreateProps`를 전혀 넘기지 않는다. **gzip 필터도, shuffle도, 청킹도 없다** —
전 데이터셋이 contiguous·비압축이다. 따라서 과제 (2)(b)의 "gzip 끄기 / 청크 키우기"는
**끌 것이 없다**. 데이터셋별 attribute 재작성도 없다(attribute를 아예 쓰지 않는다).

### 1.4 solver 스레드에서 동기인가 — HDF5는 **아니다**, CSV는 **그렇다**

`src/IoWriter.h`의 writer 스레드가 2026-08-27부터 **기본**(`RASBERY_IO_WRITER` 미설정 →
`thread`)이다. `Recorder`가 각 HDF5 호출을 인자 소유 복사본에 대한 클로저로 기록하고
bounded MPSC 큐(기본 64 배치 / 512 MB)에 넘기며, `CloseResult`가 `FenceJobWrites()`로
합류한다. 즉 **과제 (2)(a)는 HDF5에 대해서는 이미 구현되어 있다.**

남은 동기 구간은 하나뿐이다 — `src/IO.cpp`의 핀출력 CSV
(패치 전 `IO.cpp:1689`의 `std::ofstream csv(...)`부터 `:1762`까지). 이것은 HDF5가
아니므로 writer 스레드 채택이 손대지 않았고, `Driver.h:4703`이 스스로 그 크기를
적어 두었다 — **"pin-off drops the ~119 MB/case pin-power CSV"**.

### 1.5 크기

`pin_power` = `nz × nxya × npins²` double, `pin_flux` = 거기에 `× ng`.
CSV는 같은 배열을 **텍스트로** 두 번(Z평균 1회 + 축방향 평면 `kec-kbc`회) 찍는다.
`csv << val`는 값당 `num_put` 호출 하나 — 케이스당 ~119 MB / 값당 ~9 B ≈ **1천만 회
이상의 double→텍스트 변환이 solver 스레드 위에서** 일어난다. 이것이 `result_write`가
`result_add`보다 크고 io_wall이 2.89 s인 이유다.

### 1.6 배치 `--result light`와의 차이

`ResultMode`는 셋이다(`src/Driver.h:4702`~`4708`).

| 모드 | `WriteStepToResult` | 핀 CSV | HDF5 결과파일 |
|---|---|---|---|
| `full` (기본) | 호출 | **쓴다 (~119 MB)** | 전부 |
| `pin-off` | 호출 | 안 씀(`print_opt.pin_info=false`로 덮어씀) | `pin_power`/`pin_flux` 제외 |
| `light` | **호출 안 함** | 안 씀 | **없음** — `BatchLightResult::Write`의 스칼라 JSONL 한 줄 |

`light`는 **출력 형태 스위치**이지 물리 스위치가 아니다(`tools/exact_audit.py` 참조).
WP12는 `full`에서만 값을 낸다 — `pin-off`/`light`에는 옮길 직렬화가 없다.

---

## 2. 무엇을 구현했는가

### 2.1 `src/IoWriter.h`

1. **게이트.** `resultAsyncRequested()` / `resultAsyncEnabled()` / `resultIoModeName()`.
   `RASBERY_RESULT_ASYNC`가 미설정·빈 값·`0`이면 **sync**(기존 경로). `1|true|on|yes`이면
   async. 단 `RASBERY_IO_WRITER=inline`에서는 넘길 스레드가 없으므로 **sync로 강등**하고
   수신증에 `sync`라고 적는다 — 골든이 얼어붙지 않은 스레드를 여기서 새로 만들지 않는다.
2. **`Batch::tasks`.** `std::vector<std::function<void()>>`. `ops`(HDF5)와 **분리**한다:
   `ReplayCtx`도 핸들 슬롯도 필요 없고, 무엇보다 `Hdf5Guard` **밖에서** 돌아야 한다.
3. **`replay()`가 tasks를 먼저, 가드 밖에서 실행.** 15 MB짜리 텍스트 포맷 동안
   프로세스 전역 HDF5 락을 쥐고 있으면 이 파일이 없애려던 직렬화를 나머지 63 덱에게
   그대로 돌려주는 셈이다. HDF5 op과의 순서는 **다른 파일이므로 무관**하고,
   task끼리의 순서는 큐의 FIFO가 보장한다(한 Driver 스레드가 프로그램 순서로 넣는다).
4. **`Recorder::pushSideTask(F&&, bytes)`.** inline 모드면 즉시 실행(= 패치 전 호출).
   thread 모드면 배치에 싣고 **payload를 `_batch.bytes`에 계상** — 큐의 바이트 경계가
   task에도 걸린다.
5. **`Recorder::submit()`의 조기 반환 수정.** `ops.empty()`만 보던 것을
   `ops.empty() && tasks.empty()`로. task만 실린 배치가 조용히 버려지지 않는다.
6. **`ResultIoCounters` + `reportResultIo()`** — §3.

### 2.2 `src/IO.cpp`

`PinPowerCsvRecord`(익명 네임스페이스, `IO.cpp:1440`) — **값 타입**이다.
필드는 `path / append / step / efpd / nz,nxya,npins,npina,nxa,nya,kbc,kec /
hz[nz] / ijtola[nya*nxa] / pin_data[nz*nxya*npina]`. **참조도 포인터도 없다.**

`PinPowerCsvRecord::Emit() const`는 패치 전 블록을 **그대로** 옮긴 것이고 치환은 둘뿐이다:

* `g.hz(k)` → `hz[k]`
* `g.ijtola(ia, ja)` → `ijtola[ja * nxa + ia]`

manipulator(`std::setprecision(6)`), `<<` 호출 순서, `std::format` 문자열, 개행 위치가
모두 동일하다. 이 트리에는 `std::locale::global` 호출이 하나도 없으므로 두 경로가
같은 `num_put`을 쓴다.

호출부(`IO.cpp` `WriteStepToResult` 안):

```cpp
PinPowerCsvRecord csv_record;   // 숫자 memcpy — 싸다
...
csv_record.pin_data = std::move(pin_data);   // HDF5 write_raw가 이미 자기 복사본을 떴다
_pin_power_csv_started = true;               // solver의 질문이므로 두 모드 모두 여기서 전진
if (iowriter::resultAsyncEnabled())
    rec.pushSideTask([record = std::move(csv_record)]() { record.Emit(); }, csv_bytes);
else
    csv_record.Emit();                       // feature-off = 예전 그 자리, 그 스레드
```

`pin_data`는 **복사가 아니라 이동**한다 — HDF5 `write_raw`가 이미 배치용 복사본을
떴으므로 남은 독자가 없다. 즉 async가 켜져도 **추가 대용량 복사는 0**이다.

### 2.3 `src/main.cpp`

`reportSummary` 세 호출부 옆에 `reportResultIo(std::cout)`를 붙였다.

---

## 3. 수신증

```
[RASBERY][RESULT_IO] {"mode":"async","records":8,"bytes":124780544,
                      "writer_wall_ms":2140.5,"solver_blocked_ms":0.0,"queue_max_depth":3}
```

| 필드 | 의미 |
|---|---|
| `mode` | `async` \| `sync`. `RASBERY_RESULT_ASYNC=1`이어도 writer가 inline이면 `sync` |
| `records` | 직렬화된 결과 레코드 수 = 핀출력 CSV 블록 수 = `pin_info` 상태점 수 |
| `bytes` | 그 레코드들이 디스크에 **실제로 덧붙인** 바이트. `<<` 호출을 세지 않고 `file_size` 차이로 잰다 |
| `writer_wall_ms` | emitter 안에서 보낸 시간. **모드와 무관하게 같은 양**을 재므로 A/B가 뺄셈이 된다 (sync면 solver 스레드에서, async면 writer 스레드에서) |
| `solver_blocked_ms` | 가득 찬 큐를 기다린 시간(`iowriter::counters().block_ns`) |
| `queue_max_depth` | 큐 최고 수위(`counters().max_depth`) |

`solver_blocked_ms`/`queue_max_depth`는 writer 큐 카운터를 **읽는다**. 결과 레코드가
HDF5 배치와 같은 bounded 큐를 타므로 그 숫자가 곧 답이고, 별도로 세는 것은 서로
어긋날 기회를 하나 더 만드는 것뿐이다.

---

## 4. 정확성 등급 — B0 예상

| 주장 | 근거 |
|---|---|
| **궤적 불변** | writer는 solver가 소유한 어떤 것도 읽지 않는다. 스냅샷은 값 전용이고 emitter는 `g.`/`d.`/`xs.`/`_pin_power_csv_*`를 한 번도 쓰지 않는다(`tools/test_result_async_contract.py`가 강제). solver 쪽에 되돌려 쓰는 경로가 없으므로 digest·outers는 정의상 불변 |
| **HDF5 바이트 불변** | WP12는 HDF5 경로를 **전혀 건드리지 않았다**. op 순서·핸들 슬롯·payload가 그대로 |
| **CSV 바이트 불변** | 같은 코드, 같은 manipulator, 같은 값, 같은 순서. 상태점 간 순서는 큐 FIFO |
| **feature-off** | 스냅샷도 안 만들고 task도 안 만들고 emitter가 예전 그 자리에서 돈다. 두 번째 구현이 아니라 **같은 함수** |

**알려진 실패 경로 차이 하나** (성공 경로에는 영향 없음): CSV 열기 실패 시,
sync는 예전처럼 solver 스레드에서 던진다. async는 writer 스레드에서 던지고
`replay()`가 세션을 poison → `[RASBERY][IO_WRITER][FAIL]`이 나가고 다음
`ThrowIfWritesFailed()`/`CloseResult()`의 fence가 **그 잡(job)의** 예외로 바꾼다.
잡은 어느 쪽이든 실패한다. 또한 async에서는 `_pin_power_csv_started`가 열기 실패
전에 이미 true가 된다 — 실패 경로에서만 관찰 가능한 차이다.

---

## 5. startup 0.92 s — 계측 계획만 (과제 (3), 코드 변경 없음)

**추가 계측이 필요 없다. 이미 있다.**

* `init_seconds` = `Driver.h:4933` — `driver_start`부터 `OpenResult` 직후까지 전부
  (덱 파싱 + geometry 빌드 + XS 라이브러리 확보 + nxyz 크기 할당 + 결과파일 생성).
  `[TIMING] Init+IO`로도 인쇄.
* `library_seconds` = `Driver.h:4724`~`4739` — **`IO::ReadInput` 하나**를 감싼다.
  즉 `init_seconds` ⊃ `library_seconds`이고, 둘의 차가 solver 객체 생성과 `OpenResult`다.
* **분해는 `[RASBERY][READINPUT]`이 이미 인쇄한다** (`src/IO.cpp:1075` 근방):
  `{"deck_s","geometry_s","xs_s","rest_s","total_s"}` —
  각각 JSON 덱 파싱 / geometry 빌드 / `XSSet::Initialize`(캐시된 파스 + nxyz 할당) /
  rod 맵·shuffle·restart 복원.
* **XS 라이브러리 HDF5 읽기의 상각 여부는 `[RASBERY][XSLIB_CACHE]`가 답한다**
  (`src/XSSet.cpp:1111`): `{"loads","hits",...}`.

**상각 판정 (소스 기준):**

* `AcquireXsLibrary`(`src/XsLibrary.h:216`)는 **프로세스 전역** 캐시이고 키가
  (정규화 경로, size, mtime, ng)다. 34 MB CHIFFON 파스는 **프로세스당 1회**이고,
  둘째 케이스부터는 mutex 하나 + `shared_ptr` 복사다. 따라서 **영속 evaluator와
  배치 모드에서는 `xs_s`의 파스 절반이 이미 상각되어 있다.**
* 상각되지 **않는** 것: JSON 덱 파싱(`deck_s`), geometry 빌드(`geometry_s`),
  nxyz 크기 할당(`xs_s`의 나머지 절반), restart/shuffle(`rest_s`), `OpenResult`.
  이것들은 케이스 고유 상태이므로 구조상 케이스마다 낸다.
* **따라서 단일 덱 1회 실행(= v4 후보 실측)에서는 상각이 0이다.** 0.92 s는 통째로
  한 번 낸 값이고, 이 숫자만으로 "startup을 줄여야 한다"고 말할 수 없다 —
  40× 캠페인의 단위는 케이스 처리량이고 거기서는 `xs_s`의 파스 몫이 이미 사라진다.

**238에서 할 일(관측만):** v4 후보 env로 단일 실행 1회, `[RASBERY][READINPUT]`과
`[RASBERY][XSLIB_CACHE]` 한 줄씩을 로그에서 뽑아 `deck_s / geometry_s / xs_s / rest_s`의
0.92 s 분해를 기록한다. 그 분해가 `xs_s`에 몰려 있으면 상각이 답이고(= 영속 evaluator
경로가 이미 해결), `geometry_s`/`deck_s`에 몰려 있으면 그때 비로소 레버를 고민한다.

---

## 6. 238 런북

### 6.1 A/B 한 쌍

PROD env(§1.1 of `PRICING_PROD_20260830_KO.md`) + v4 후보 다섯 개
(`RASBERY_GPU_CRAM=1`, `RASBERY_GPU_CMFD_FUSE=15`, `RASBERY_GPU_PPR=1`,
`RASBERY_GPU_PPR_GRAPH=1`, `RASBERY_GPU_XE_TXN=1`), **`--result full`**,
KNGR cycle-1, 35 상태점.

| arm | 추가 env | 출력 디렉터리 |
|---|---|---|
| (A) 기준 | 없음 | `$OUT/wp12_off` |
| (B) WP12 | `RASBERY_RESULT_ASYNC=1` | `$OUT/wp12_on` |

각 arm: warm-up 1 + hot 3, 교차 순서(A,B,A,B,...). 출력은 **E: 아래**에.
`RASBERY_STATEPOINT_TELEMETRY=1`은 io_wall을 읽는 **계측 실행에만** 켜고,
wall median을 뽑는 타이밍 실행에서는 끈다(PROD 규약과 동일).

### 6.2 게이트

| # | 확인 | 기대 |
|---|---|---|
| G1 | `h5diff -c $OUT/wp12_off/<stem>.h5 $OUT/wp12_on/<stem>.h5` | **0 차이** |
| G2 | `cmp $OUT/wp12_off/<stem>_pinpower.csv $OUT/wp12_on/<stem>_pinpower.csv` | **동일** (h5diff가 CSV를 보지 않는다 — 이 줄이 WP12의 진짜 게이트다) |
| G3 | digest | **`1f36e75dc00ed2b4`** 불변 |
| G4 | outers | **4377** 불변 |
| G5 | `[RASBERY][IO_WRITER][SUMMARY]`의 `failures` / `skipped` | 둘 다 **0** |
| G6 | `[RASBERY][RESULT_IO]` | (A) `mode:"sync"`, (B) `mode:"async"`, 양쪽 `records`/`bytes` **동일** |
| G7 | `[RASBERY][SPTELEM][SUMMARY]`의 `io_wall` | (B)가 (A)보다 작아야 한다 |
| G8 | wall median | (B) ≤ (A). Δ를 보고한다 |

**G6의 `records`/`bytes`가 두 arm에서 같다는 것이, 옮긴 것이 같은 양의 같은 일이라는
가장 싼 증거다.** 다르면 G2를 보기 전에 멈춘다.

### 6.3 보고 형식

```
WP12 A/B (238, v4 후보 env, --result full, 35 sp)
  (A) RESULT_ASYNC off : wall <median> s, io_wall <x> s, RESULT_IO mode=sync  writer_wall_ms=<w>
  (B) RESULT_ASYNC on  : wall <median> s, io_wall <y> s, RESULT_IO mode=async writer_wall_ms=<w'> solver_blocked_ms=<b> queue_max_depth=<q>
  h5diff 0 : Y/N     pinpower.csv cmp : same/diff
  digest 1f36e75dc00ed2b4 : Y/N     outers 4377 : Y/N
  Δwall = <B-A> s
```

### 6.4 기대치와 그 한계

`writer_wall_ms`가 CSV 직렬화의 절대 비용이다. 이상적으로 io_wall이 그만큼 줄지만
**상한은 solver가 그동안 실제로 할 일이 있느냐**로 정해진다. 마지막 상태점의 CSV는
숨길 solve가 남아 있지 않으므로 `CloseResult`의 fence에서 그대로 드러난다 —
즉 기대 절감은 `writer_wall_ms × (records-1)/records`가 상한이고, 큐가 차서
`solver_blocked_ms > 0`이면 그만큼 더 깎인다.
`RASBERY_IO_WRITER_QUEUE_MB`(기본 512 MB)가 좁으면 그것부터 넓힌다.

---

## 7. 계약 시험

| 시험 | 결과 |
|---|---|
| `tools/test_result_async_contract.py` (신규) | **PASS** (4 스캔, 4 음성 대조군) |
| `tools/test_enum_alias_contract.py` | **PASS** (127 파일, 66 enum) |
| `tools/test_dependent_template_contract.py` | **PASS** (125 파일) |

신규 시험이 강제하는 네 가지:

1. **스냅샷이 새지 않는다** — `PinPowerCsvRecord`에 참조/포인터/solver 타입이 없고,
   `Emit()`이 `g.`/`d.`/`xs.`/`_pin_power_csv_*`/`_result_session`을 쓰지 않는다.
2. **큐가 유계다** — `pushSideTask`가 payload를 배치 바이트에 계상하고,
   `Queue::submit`이 두 경계를 모두 보며 `_not_full`에서 블록하고,
   `replay()`가 task를 **`Hdf5Guard` 앞에서** 돌리고, task만 실린 배치가 버려지지 않는다.
3. **닫기 전에 합류한다** — `CloseResult`가 `_result_session.reset()` **전에**
   `FenceJobWrites()`를 부른다.
4. **feature-off가 옛 경로다** — 게이트의 기본이 false이고, sync/async 두 갈래가
   **같은** `Emit()`을 부르며, `WriteStepToResult`가 더는 스스로 `std::ofstream`을
   열지 않는다.

각 스캔은 그것을 위반하는 합성 소스(음성 대조군)에 대해 반드시 실패해야 하고,
실패하지 못하면 이 시험 자체가 FAIL로 보고한다.
