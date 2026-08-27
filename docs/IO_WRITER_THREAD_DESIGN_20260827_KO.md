# 전용 HDF5 writer 스레드 설계 — 배치 I/O 직렬화 해소 (`RASBERY_IO_WRITER`)

**작성일**: 2026-08-27 | **브랜치**: `codex/exact-throughput-campaign` | **대상**: 배치 처리량 (M64 214.8 → 250~300 cases/h 목표)

> **채택 상태 (2026-08-27): 기본값 = `thread`, 전 실행 모드 공통.**
> 238 검증에서 시도한 **모든 구성에서 byte-identical**(단일덱 500/500, M64 **45,312/45,312 데이터셋**, restart 스냅샷 포함) + M64 **+0.6 %**. 따라서 `inline`은 더 이상 "안전한 선택"이 아니라 **레거시 경로**이며, `RASBERY_IO_WRITER=inline`로만 도달한다(bisect·A/B용). 오타 값은 경고 후 **기본값(thread)** 으로 떨어진다 — 골든이 동결된 경로가 기본값이기 때문.
> 수신증에 **provenance**가 추가되었다: `{"mode":"thread","mode_source":"default"}` / `"env"`. `thread(default)`·`thread(env)`·`inline(env)` 세 실행이 로그에서 구분되지 않으면 A/B가 무효이므로, config·summary **양쪽** 수신증에 모두 실린다.
> 관련: `docs/CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md` §8(채택 결정), `test/reference/validation_*_v2.json`(baseline 재동결 준비).

## 1. 문제 — 왜 지금 I/O가 1순위 지렛대인가

- HDF5 1.10.x 빌드는 thread-safe가 아니므로 `Chiffon::Hdf5Guard`(프로세스 전역 recursive mutex)가 **모든** 런타임 진입을 직렬화한다. `--batch-mode 64`가 성립하는 근거이지만, 그 비용을 전부 **solver 스레드**에 청구한다.
- 수신증 실측: `[RASBERY][HDF5][LOCK]` 획득당 ~734 ms, 누적 대기 수백만 ms.
- SPTELEM 위상 분해(M64): `io_wall`이 단일덱 ~3 %에서 **~40 %/케이스**로 폭증. Anderson의 outer −38 % 감축이 처리량에 반영되지 않은 원인이 바로 이것(`CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md` §3).

즉 **배치 wall은 compute가 아니라 I/O 직렬화가 통제**한다.

## 2. 설계 — 락을 키우지 않고 소유권을 옮긴다

락은 64개 스레드가 비-thread-safe 라이브러리 안에서 *교대*하게 만든다. 전용 스레드는 **오직 한 스레드만** 쓰기로 진입하게 만든다.

```
Driver 스레드 (×64)                     writer 스레드 (×1)
  ── 계산 ──┐                            ┌──────────────────────┐
            │ record(소유 복사본)        │ Hdf5Guard 1회/batch  │
            ├──> [ 경계 MPSC FIFO 큐 ] ──┤ ops를 기록 순서대로   │
  ── 계산 ──┘        (개수+바이트 상한)   │ 그대로 replay        │
                                         └──────────────────────┘
```

- 총 HDF5 작업량은 줄지 않는다(같은 호출, 같은 순서). **다른 63덱의 compute와 겹쳐질 뿐**이다.
- 읽기(덱 파싱, XSLIB 로드, restart 로드, shuffle)는 그대로 Driver 스레드에서 `Hdf5Guard` 아래 수행된다. 따라서 writer 스레드도 **batch당 1회 같은 guard를 잡는다** — 런타임 안의 스레드는 여전히 정확히 하나이고, 달라진 것은 **"solver가 직접 기다리느냐, writer가 대신 기다리느냐"**뿐이다.
- **주의**: `[HDF5][LOCK].acquires`는 줄지 않는다(inline도 이미 쓰기 함수당 1회였다 — 실측 54 vs 54). 이 변경의 신호는 획득 횟수가 아니라 **SPTELEM `io_wall`**이다(로컬 단일덱 실측 0.0151 s → 0.0018 s).

### 2.1 구현 형태 — 직렬화기가 아니라 **호출 녹화**

`src/IoWriter.h`가 `HighFive::File&`/`HighFive::Group`을 대체하는 프록시(`iowriter::Node`)를 제공한다. 호출부는 전부 `auto`를 쓰고 있었으므로 텍스트 변경이 거의 없다.

| 호출 | inline 모드 | thread 모드 |
|---|---|---|
| `parent.createGroup(name)` | 즉시 실행 | op 기록, 슬롯 k 예약 |
| `parent.createDataSet(name, value)` | 즉시 실행 | `payload = value` **소유 복사** 후 op 기록 |
| `parent.createDataSet<T>(name, dims)` | 즉시 실행 | 생성 op 기록(슬롯 예약) |
| `.write_raw(ptr)` | 즉시 실행 | `_count`개 원소를 소유 버퍼로 복사 후 **별도** op 기록 |

`HighFive::DataSpace`는 생성자가 `H5Screate_simple`을 호출하므로 **Driver 스레드에서 만들면 안 된다**. 그래서 호출부는 `HighFive::DataSpace` 대신 순수 숫자 타입 `iowriter::Dims`를 넘기고, DataSpace는 실제로 데이터셋이 만들어지는 곳(inline이면 recorder guard 아래, thread면 writer 스레드)에서 구성된다.

### 2.2 byte-identity 논증 (계약의 핵심)

replay는 **inline 호출열 그 자체**를 나중에 다른 스레드에서 실행하는 것이다.

1. **순서**: op는 기록 순서대로 replay되고, 한 파일의 batch는 제출 순서대로 replay된다(FIFO 큐 1개 + 파일당 기록자 1개 = 그 덱의 Driver 스레드). 따라서 모든 그룹의 link 삽입 순서가 inline과 동일하다.
2. **핸들**: 생성 op는 결과 핸들을 `push_back`하므로 k번째 생성은 항상 슬롯 k에 들어간다. 자식 op는 부모를 **슬롯 번호로** 지목한다 — 경로 재유도가 아니라 `parent.createDataSet(...)` 그대로다.
3. **값**: payload는 기록 시점에 복사된다. solver는 반환 즉시 Geometry/XSSet을 변경해도 무방하다.
4. **생성/쓰기 분리**: inline에서 `createDataSet<T>(name, space)`는 `write_raw`가 뒤따르지 않아도 link를 만든다. 녹화도 같아야 하므로 생성과 `write_raw`를 별개 op로 둔다.

수용 시험은 기존 bit-golden 게이트다(단일덱 500/500 데이터셋, 배치 45,312/45,312 — 전 구성 통과가 채택 근거). `RASBERY_IO_WRITER=inline`(레거시 경로, 이제 명시적으로만 선택)은 문자 그대로 종전 경로이므로 자명하게 불변이다.

**적용 범위 — 성공 경로에 한정.** 위 논증은 모든 op가 성공적으로 replay된 실행에 대한 것이다. **오류 경로의 부분 파일은 두 모드가 다를 수 있고, 그것이 의도된 설계다**: inline은 예외가 난 그 지점까지 쓰고 멈추는 반면, thread 모드는 세션을 poison하고 **그 파일의 이후 batch를 전부 건너뛴다**(§2.5). 실패한 job의 산출물은 어느 모드에서도 유효하지 않으므로 게이트 대상이 아니다 — 대신 그 job이 반드시 **loud하게 실패**하는 것이 계약이다.

### 2.5 실패 격리 — poison된 세션은 흡수성(absorbing)

파일 생성이 실패하면(권한 없는 출력 디렉터리 등) 그 세션은 `failed`로 표시된다. 이후 그 세션 앞으로 제출된 batch는 **replay되지 않고 건너뛰어진다**. 그렇게 하지 않으면 열리지 않은 파일을 역참조해 **writer 스레드가 SIGSEGV로 죽고 64덱 전부가 함께 죽는다**. 건너뛴 batch는 수신증의 `skipped`로 계수되고(조용한 손실이 아니다), 최초 실패는 job id와 함께 즉시 보고되며, `CloseResult()`의 fence가 그것을 **그 job의 예외**로 되돌린다. 다층 방어로 op를 열지 않는 batch는 replay 진입 시 `file` 널 여부를 먼저 검사하고, 슬롯 참조는 전부 경계 검사(`groupAt`/`dataSetAt`/`fileOf`)를 거친다.

### 2.3 교착 없음

유일한 blocking edge는 `solver → 가득 찬 큐`이고, writer는 solver를 기다리지 않는다(큐와 `Hdf5Guard`만 기다리며, solver는 큐 없이도 guard를 놓는다). thread 모드의 recorder는 **어떤 HDF5 락도 잡지 않으므로**, 큐에서 블록된 Driver가 guard를 쥔 채 멈추는 상황이 없다.

### 2.4 restart 읽기 경쟁 — 조사 결과

**같은 실행 안에서 자기가 쓴 restart를 되읽는 경로는 없다.** 근거:

- restart **입력**은 덱의 `"restart"` 블록에서 오고, 전부 `IO::ReadInput()` 안에서 읽힌다 — 즉 그 job이 첫 쓰기를 큐에 넣기 **전**이다. (`LoadGeometryFromRestart`, EFPD 복원, shuffle 소스 읽기 모두 `ReadInput`/`ApplyShuffle` 경로.)
- restart **출력** 경로는 `Driver::RestartPath()`가 OUTPUT 경로(`<dir>/<stem>_restart_<step>.h5`)에서 유도한다(plan Rev.4 §7). 덱들이 입력 파일이나 출력 상위 디렉터리를 공유해도 restart 네임스페이스는 job-local이다.
- 따라서 writer가 쓰는 중인 파일을 다른 job이 읽는 일은 구조적으로 발생하지 않는다. 그럼에도 읽기는 여전히 `Hdf5Guard` 아래에서 수행되므로, 설령 그런 덱이 작성되더라도 라이브러리 수준의 경쟁은 없다(관측되는 것은 "아직 안 쓰인 파일"이지 "찢어진 파일"이 아니다).

## 3. 계약 (`tools/test_io_writer_contract.py`)

h5diff가 볼 수 **없는** 성질만 정적으로 고정한다.

1. **env 게이트 = 2모드 × 2 provenance** (채택 후 갱신): 미설정/빈 값 → **thread(default)**, `thread` → thread(env), `inline` → inline(env), 그 외 → 경고 후 **thread(default)**. 함수 지역 static 1회 캐시(`resolution()`), `mode()`/`modeSource()`는 그 1개 답을 보는 얇은 접근자, `RASBERY_IO_WRITER` 독자는 `IoWriter.h` 한 곳뿐. config·summary 수신증 양쪽이 `mode_source`를 싣는다.
2. **writer가 쓰기 전부를 소유**: `IO.cpp`에 `HighFive::Group/DataSet/DataSpace`와 `_result_file`이 하나도 남아 있지 않고, 남은 `HighFive::File`은 전부 `ReadOnly`. replay 사이트는 1곳이며 `Hdf5Guard`를 batch 전체에 대해 잡고, `ReplayCtx`는 guard 스코프 **안에서** 생성·소멸한다(Group/DataSet 소멸자도 런타임 재진입).
3. **순서·소유권**: 슬롯 `push_back` 순서, 생성/`write_raw` 분리, payload 값 복사(참조 캡처 금지), FIFO(`front`/`pop_front`).
4. **경계 큐**: 개수·바이트 **양쪽** 상한, 초과 시 대기(드롭 없음), 대기 시간을 `enqueue_block_ms`에 청구, high-water 기록. 단일 batch가 바이트 상한을 넘어도 교착하지 않음(`|| _queue.empty()`).
5. **종료·실패**: drain 후 join, 실패는 job id와 함께 `[RASBERY][IO_WRITER][FAIL]`로 즉시 보고 + 세션에 표시 → `CloseResult()`의 fence에서 **그 job의 예외**로 재발생, 프로세스 종료코드에도 반영. 두 main() 분기 모두 config/summary 수신증 방출.
6. **SPTELEM**: 두 수신증이 line sink를 통과하고, append와 write가 같은 mutex 아래(라인 분할 불가).

## 4. 수신증

```
[RASBERY][IO_WRITER] {"mode":"thread","mode_source":"default","queue_limit":64,
  "queue_bytes":536870912}
...
[RASBERY][HDF5][LOCK] {"acquires":...,"wait_ms":...}
[RASBERY][IO_WRITER][SUMMARY] {"mode":"thread","mode_source":"default","requests":...,
  "ops":...,"bytes":...,"max_queue_depth":...,"max_queue_bytes":...,
  "enqueue_block_ms":...,"writer_busy_ms":...,"failures":0,"skipped":0}
```

`[BATCH_HOST]`(선언된 구성, 첫 덱 이전) / `[BATCH_HOST][PIN]`(최종 카운터, teardown)과 같은 계열 규칙이다. 읽는 법:

- `mode_source`는 **그 값이 어디서 왔는지**다. `thread`+`default`는 프로덕션 실행, `thread`+`env`는 명시적 A/B 팔, `inline`+`env`는 레거시 비교 팔. baseline 재동결 실행은 반드시 `mode_source="default"`여야 한다(env가 붙은 실행은 실험이지 기준이 아니다).
- `requests=0`인데 `mode="thread"`면 → 게이트가 실제로 켜지지 않았거나 쓰기가 없었다. A/B의 thread 팔에서 이 값이 0이면 그 측정은 무효다.
- `enqueue_block_ms`가 크면 → writer가 병목(큐 상한 상향 또는 I/O 자체가 한계).
- `writer_busy_ms` ≈ 총 wall 이면 → HDF5 작업량 자체가 한계이고, 겹침으로 얻을 것이 남지 않았다(그때는 데이터셋 축소/압축이 다음 지렛대).
- `failures` > 0 이면 그 실행은 무효다.

## 5. 게이트 계획 (238 검증 에이전트용)

| 게이트 | 내용 | 합격 기준 |
|---|---|---|
| **G0** | 계약 테스트 `python3 tools/test_io_writer_contract.py` | PASS |
| **G1** | 레거시 arm 회귀: `RASBERY_IO_WRITER=inline`로 기존 단일덱 골든 | 500/500 데이터셋 Δ=0 (**채택 후 이 arm은 기본값이 아니라 명시적 env로만 도달**) |
| **G1b** | **기본값 provenance**: env 미설정 실행 | `mode="thread"`, `mode_source="default"`, `failures=0`·`skipped=0` |
| **G2** | **thread byte-identity (단일덱)**: 같은 덱을 `inline`/`thread`로 각각 실행 후 `h5diff` | out.h5 및 모든 `*_restart_*.h5` 차이 0 |
| **G3** | **thread byte-identity (배치)**: M64 골든 세트 | 708/708 데이터셋 Δ=0, FAIL 0/64 |
| **G4** | **처리량 A/B (M64)**: 동일 덱 세트를 `inline` vs `thread` | thread ≥ inline; 목표 250~300 c/h. `[HDF5][LOCK].acquires` 대폭 감소 + `[IO_WRITER][SUMMARY].failures=0` 동반 확인 |
| **G5** | 수신증 무결성 | 두 실행 모두 config/summary 1줄씩, `mode`가 의도한 값, `failures=0`·`skipped=0` |
| **G6** | **음성 게이트**: 1덱의 `--raso`를 쓰기 불가 디렉터리로 | exit≠0, `[IO_WRITER][FAIL]`에 job id, 크래시/행 없음, 나머지 63덱 무영향 |

**주의 1**: G4는 SPTELEM을 끈 상태로 측정한다(telemetry 자체가 배치에서 I/O 경합을 증폭해 처리량을 왜곡한다).

**주의 2 — `cmp`를 동일성 판정에 쓰지 말 것.** HDF5 object header는 기본값(`H5Pset_obj_track_times` = TRUE)으로 **생성/수정 시각을 파일에 기록**한다. 실측: 같은 `inline` 바이너리로 3초 간격 재실행한 두 파일도 오프셋 55(root group `OHDR` 뒤 4×4-byte epoch)에서 raw 바이트가 다르다. **판정은 `h5diff`(데이터셋 단위) — 기존 골든 게이트가 쓰는 그 기준 — 으로만 한다.**

### 5.1 로컬 사전 검증 (CPU-only 빌드, 2026-08-27)

CUDA 없는 g++ 13.3 빌드에서 `--batch-mode`는 막혀 있으므로 배치 동시성은 238에서 확인해야 한다. 그 전 단계까지는 다음이 통과했다.

| 시나리오 | 결과 |
|---|---|
| 단일덱 (node monitor 32노드 + xs-info + restart save) | `out.h5`, `restart_1.h5` **h5diff 0 차이** |
| 단일덱 + 핀출력 (pin_power, pin_flux, `*_pinpower.csv`), 3 스텝 | `pin.h5` / `restart_1.h5` / CSV **모두 동일** |
| 한 프로세스 6덱 직렬 (세션 6개가 한 writer를 통과) | 6개 결과 파일 + restart, **h5diff 0 차이** (파일당 객체 4,912개) |
| 백프레셔 (`RASBERY_IO_WRITER_QUEUE=1`, `_QUEUE_MB=1`) | 결과 동일, `enqueue_block_ms=149.5`로 청구, `requests/ops/bytes`는 기본 큐와 완전히 동일 |
| 계약 테스트 | `tools/test_io_writer_contract.py` PASS |
| **음성 게이트** — 쓰기 불가 출력 디렉터리(0555) | exit=1, `[IO_WRITER][FAIL]`에 job id·경로·사유, **크래시/행 없음** |
| **음성 게이트 (6덱 중 1덱 불량)** | 불량 덱만 실패, **나머지 5덱 전부 정상 산출·기준과 h5diff 0 차이** |

부수 관측: 8덱을 한 프로세스에서 동시에 띄우면 `[HDF5][LOCK].wait_ms`가 24회 획득에 **8.4~9.6 s**에 달한다. 이 대기는 statepoint 쓰기가 아니라 **94 MB XSLIB 읽기 동시 진입**이다 — writer 스레드가 손대지 않는 부분이므로, G4에서 기대 이득이 SPTELEM `io_wall` 비중(≈40 %)을 넘지 못하는 것이 정상이다. XSLIB 캐시 트랙(plan §14)이 그 나머지 절반을 겨냥한다.

## 6. 위험과 완화

| 위험 | 완화 |
|---|---|
| writer 1개가 새 직렬화 지점이 된다 | 총 HDF5 작업량은 불변이고 compute와 겹친다. `writer_busy_ms`/`enqueue_block_ms`가 이 경우를 직접 폭로한다. |
| 큐 메모리(핀파워/핀플럭스 blocks) | 개수(64) **와** 바이트(512 MiB) 이중 상한. `RASBERY_IO_WRITER_QUEUE`, `RASBERY_IO_WRITER_QUEUE_MB`로 조정. |
| `CloseResult()`의 fence가 job 말미를 직렬화 | job당 1회. 실패를 그 job의 종료코드로 되돌리는 대가이며, writer가 병목이면 어차피 모두가 기다린다. |
| payload 복사 비용 | statepoint당 수 MB memcpy. HDF5 쓰기 비용 대비 무시 가능하며, 대신 solver가 즉시 복귀한다. |
| MSVC | `std::jthread` 미사용(평범한 `std::thread` + 명시적 join), gcc 전용 구문 없음. |
