# WP8 단계 1.5 / 2: 워커가 다시 프로세스가 되고, 코호트라는 수명이 생긴다

브랜치 `codex/exact-throughput-campaign`, 기준 `3df4ea7`
커밋 `8437697`(단계 1.5 dispatcher), `2a1161b`(WP9-A enum 수정), `f32e646`(단계 2)
새 파일: `src/CohortKey.h`, `src/CohortContext.h`, `src/PprQuadrature.h`,
`tools/fake_rasbery_child.py`, `tools/test_cohort_context_contract.py`

로컬에서 수행한 것: 순수 파이썬 계약 테스트와 MSVC `/Zs` 문법 검증
(`PPR.cpp`, `XSSet.cpp`, `IO.cpp`, `main.cpp`), 그리고 CohortKey/CohortContext를
실제로 **컴파일해서 돌린** 동작 검증(§3.3).
**성능 수치는 하나도 없다. 전부 238이 재야 한다(§4).**

---

## 0. 결론 먼저

| 항목 | 상태 |
|---|---|
| 단계 1.5 — `run_multi_gpu_batch.py`가 지속 evaluator를 쓴다 | 완료(`8437697`) |
| worker 사망 시 재시작 + wave 잔여분 1회 재큐 | 완료 — 음성 대조군 포함 |
| 튜너 calibration도 evaluator를 통과 | 완료 |
| `--no-evaluator`(구 형태) = 명명된 wall 대조군 | 완료 |
| 단계 2 — CohortKey + CohortContext + 카운터 | 완료(`f32e646`) |
| 단계 2 — 공유되는 실제 상태 | **PPR quadrature 하나뿐**(§2.3에 이유) |
| 단계 2 — Geometry 맵 공유 | **미착수, 그리고 왜 아직 아닌지 명시**(§2.3) |
| XSLIB 캐시 키 content digest | 완료 |
| 단계 3 — mutable buffer reuse | **미착수. 전제가 성립하지 않는다**(§5) |
| c/h, 세대별 wall, cohort_hits, restart 시험 | **미측정 — 238 필요** |

### 0.1 이 레버의 정직한 크기, 다시

단계 1 문서 §0.1이 한 말이 그대로 유효하다: `outside_drive`(프로세스 이미지 +
로더 + CUDA 컨텍스트 + teardown)는 1.75–4.92 s이고 **프로세스당 1회**다.
단계 1이 그것을 지울 수 있는 모드를 만들었고, **단계 1.5가 실제로 지운다** —
그 전까지 `tools/run_multi_gpu_batch.py`는 청크마다 `subprocess.run`을 돌렸으므로
캠페인은 여전히 `outside_drive × 청크수`를 냈다. 이제 `outside_drive × 워커수`다.

단계 2가 지우는 것은 그보다 훨씬 작다. **PPR quadrature 테이블 하나**,
M64 wave에서 64번 짓던 것을 1번 짓는다. 크기는 `npins² × ndiv² × 9 × 17` double
언저리(17×17 pin, ndiv=2면 약 1.4 MB)이고 빌드는 밀리초다.
**이것이 5 %를 만들 것이라고 주장하지 않는다.** 단계 2가 실제로 만든 것은
*키와 회계*이고, 그 위에 Geometry 맵을 올리는 것이 값어치의 대부분이다(§2.3).

---

## 1. 단계 1.5 — 디스패처

### 1.1 워커가 무엇이 되었나

`run_multi_gpu_batch.py`의 워커는 **청크마다 새 RASBERY 프로세스를 띄우는
시퀀스**였다. 이제 `(GPU, K-slot)`마다 **지속 `RASBERY --evaluator-jsonl -`
하나**이고, 청크는 그 stdin에 `{"op":"wave", ...}` 한 줄로 들어간다.

바뀌지 않은 것: 물리, 큐, claim 정책, 청크 회계, 수신증 감사, MPS, 튜너의
선택 규칙, `[MULTI_GPU][PROC]`/`[GPU]`/`[TOTAL]`의 기존 필드.
바뀐 것: **프로세스 이미지의 개수**.

그래서 receipt가 `images`와 `waves`를 **따로** 낸다
(`tools/run_multi_gpu_batch.py`, `[PROC].processes` / `[PROC].waves`,
`[TOTAL].images` / `[TOTAL].waves`). `--no-evaluator`에서는 둘이 같고,
그 등식이 깨지는 폭이 이 모드가 산 것의 전부다. 한 필드로 합쳐 놓았다면
레버가 당겨졌는지 아닌지를 로그에서 읽을 수 없다.

### 1.2 wave가 batch 분기의 수신증을 낸다

`check_run_receipts`(`tools/run_single_gpu_batch.py`)는 `[RASBERY][BATCH_HOST]`를
읽어 "multi-instance batch 분기가 실제로 돌았는가, 요청한 host_threads로
돌았는가"를 판정한다. evaluator는 그 태그를 내지 않았으므로, 디스패처가 보낸
**모든 wave가 "batch 분기가 돌지 않았다"로 감사될** 참이었다.

한 모드를 위해 감사를 약화시키는 것이 그 모드가 감사되지 않게 되는 경로다.
그래서 반대로 했다: `EvaluatorServer::runWave`가 `WAVE_START` 바로 뒤에
`main.cpp:1118`과 **같은 필드, 같은 `legacy_pinning_criterion` 규칙**으로
`[RASBERY][BATCH_HOST]`를 wave마다 낸다.

`[PHYSICS_MODE]`는 다르다 — 프로세스당 한 번, 첫 요청을 읽기 전에 인쇄된다.
그래서 디스패처는 자식의 **preamble**(READY 이전 출력)을 보관하고 wave 텍스트
앞에 붙여서 감사한다(`session.preamble`). **fidelity는 프로세스 속성이고,
워커가 이제 프로세스이므로, fidelity는 워커 속성이다.** 한 캠페인에서
strict와 A2를 섞으려면 evaluator 두 개이지 wave 두 개가 아니다.

### 1.3 실패 격리는 두 층이고, 새로 생긴 것은 두 번째뿐이다

| 층 | 누가 처리하나 | 디스패처가 하는 일 |
|---|---|---|
| 케이스 하나가 던진다 | evaluator 내부(`runOneCase`가 잡는다) | `[EVALUATOR][CASE].status="failed"`로 도착. **아무것도 하지 않는다** — 워커를 교체하지 않는다. 후보 하나가 나쁠 때마다 프로세스 stand-up을 내는 것은 GA에서 재앙이다 |
| 프로세스가 죽는다 | **디스패처만 가능** | EOF가 사망 신호다. 수신증을 낸 케이스는 회계되고, 나머지는 아무도 회계하지 않는다 |

두 번째의 규칙:

1. 수신증을 낸 것은 끝난 것.
2. 나머지는 **새 자식에 1회 재큐**.
3. 그것도 죽으면 아직 회계되지 않은 케이스를 **이름으로 실패 보고**.

**1회이고 될 때까지가 아니다.** 자식을 둘 연속 죽이는 청크에는 독이 든 후보가
있고, 무한 재시도는 나쁜 후보 하나를 멈춘 캠페인으로 바꾼다.
`--evaluator-max-restarts 0`은 교체를 끄고, 그 워커는 **claim을 멈춘다** —
남은 큐는 프로세스가 살아 있는 워커가 훔치는 편이 낫다.

수신증: `[MULTI_GPU][EVALUATOR][FATAL]`이 죽은 wave마다 한 줄, `completed`와
**이름이 든 `unfinished` 목록**을 싣는다. 개수만 적은 사망 보고는 운영자에게
매니페스트와 디렉터리를 diff시키는 것과 같다.

### 1.4 튜너

`_measure_candidate`의 calibration wave도 `evaluator=args.evaluator`를 받는다.
청크마다 프로세스를 내는 형태에서 잰 K는, 그 비용을 내지 않을 캠페인의 K가
아니다 — 프로세스당 고정비(2.56 GB VRAM, `outside_drive` 1.75–4.92 s)가 바로
K 분할이 곱하는 항이고, 워커당 한 번 내는 후보와 청크당 한 번 내는 후보는
서로 다른 무릎에 앉는다.

### 1.5 계약 테스트와 음성 대조군

`tools/fake_rasbery_child.py`는 **두 자식 형태를 하나의 수신증 코드로** 말한다.
두 테스트가 각자의 fake를 들면 서로 갈라지고, 한 모드가 다른 모드는 절대 내지
않는 수신증에 대해 통과하게 된다.

| 대조군 | 무엇을 몰아붙이나 |
|---|---|
| `FAKE_RASBERY_FAIL=<substr>` | 케이스 하나 실패, **프로세스 생존**. restarts=0, processes=1, 나머지 5개 출력 존재 |
| `FAKE_RASBERY_POISON` + `_MARKER` | 첫 이미지만 죽는다 → 재시작 → 재큐 → 전부 완료. `restarts=1`, `processes=2`, `unfinished=[d2..d5]`, 6/6 출력 |
| `FAKE_RASBERY_POISON`(마커 없음) | 두 번 죽는다 → 재큐 소진 → `failed_cases=[d2..d5]`, `[MULTI_GPU][FAIL]`에 "never evaluated", 출력은 d0/d1뿐 |
| `--evaluator-max-restarts 0` | 교체 없음, claim 중단(3청크 중 2 wave) |
| `FAKE_RASBERY_NO_READY=1` | READY 도달 실패 → 로그 경로를 가리키는 거부 |
| `--no-evaluator` | **images == waves.** 이것이 깨지면 위의 `images < waves` 검사가 공허하게 통과한 것이다 |

---

## 2. 단계 2 — 코호트

### 2.1 키 (`src/CohortKey.h`)

| 들어가는 것 | 왜 |
|---|---|
| `ng, nz, ndivxy, npins` | 모든 맵이 이것으로 크기가 정해진다 |
| `hx, hy, hz[]` | 메시 |
| `symang, symopt, symdiv` | 대칭 접힘 |
| `albedo[6]` | 경계조건 |
| **core 점유 마스크** | Geometry의 core 스캔은 `"XX"`를 건너뛴다 |
| 라이브러리 **content digest**, `ng` | `ng`로 크기가 정해지는 것은 geometry만의 함수가 아니다 |

**마스크이고 맵이 아니다.** 인덱스/이웃 맵은 어느 격자 위치가 *점유되었는가*의
함수이고 *무엇이 점유했는가*의 함수가 아니다. 그래서 조립체 종류를 섞는 GA
세대는 **하나의 코호트**다 — 그것이 최적화할 값어치가 있는 경우다.
맵을 해싱했다면 후보마다 코호트가 하나씩이고 레버는 정확히 0이며 아무 수신증도
그렇다고 말하지 않는다. 치수만 해싱했다면 구멍 있는 노심이 구멍 없는 노심의
맵을 쓰게 되는데, 그것은 느린 답이 아니라 **틀린 답**이다.

`batch`는 **들어가지 않는다**. Geometry가 그것에서 파생하는 것
(`_is_fuel`, `_kbc`, `_kec`, `_hzcore`)은 후보가 다시 채우는 케이스 상태이지
코호트가 공유하는 topology가 아니다(`src/EvaluatorContext.h:30-47`이 같은 여섯
필드를 이미 지목한다).

**대칭 필드는 들어가되 정규화하지 않는다.** casekey는 궤도로 접는다 — 등거리
변환으로 관련된 두 패턴은 같은 물리이므로. 코호트는 그러면 **안 된다**:
공유하는 상태가 *격자 위치로 색인*되고, 노심과 그 전치는 서로 다른 인덱스 맵을
통과하는 같은 물리다. 전치를 가로질러 이웃 맵을 공유하는 것은 조용한 aliasing
버그다.

**`GeometryInput`으로 키를 만들지 덱 JSON으로 만들지 않는다.** `GeometryInput`이
`Geometry::Initialize`가 실제로 받는 인자다 — shuffle resolver가 `core`를
제자리에서 다시 쓴 **뒤**, restart fallback이 geometry 블록을 복구한 **뒤**.
`config["geometry"]`에서 읽은 키는 둘 다 놓친다.

### 2.2 상태 (`src/CohortContext.h`)

`cohort::Context`의 **모든 멤버가 `const`**다. 그것이 안전성 논증의 전부다:
이 객체는 최대 M개의 Driver 스레드가 사이에 락 없이 읽는다. `mutable` 멤버
하나가 데이터 레이스이고, 증상은 아무도 재현할 수 없는 일정으로 움직이는
digest다. quadrature를 `once_flag` 뒤에서 lazy하게 채우지 않고 **빌더에서
즉시** 짓는 이유도 그것이다 — lazy하게 채워지는 멤버는 mutable 멤버다.

### 2.3 오늘 실제로 공유되는 것, 그리고 왜 더는 아닌가

**PPR pin-power quadrature 테이블 하나뿐이다.** `(ndivxy, npins)` 외에는 아무것도
읽지 않고, Driver마다 = 케이스마다 bit-identical하게 다시 지어지고 있었다.

Geometry의 이웃/표면/조립체 맵은 **여기 없다**. 이유는 신중함이 아니라 구체적이다:

1. `Geometry::Initialize`는 모든 배열을 bare `new[]`로 할당하고 `delete[]`
   프롤로그가 없다. 한 번 돌 수는 있어도 **다시 돌 수 없다** — 두 번 부르면
   전부 누수된다. 공유 Geometry는 아무도 재초기화하지 않는 Geometry여야 한다.
2. 접근자가 mutable `int&` / `double&`를 반환한다. 오늘 아무도 쓰지 않는다는
   것은 *현재 호출자에 대한 사실*이지 타입 수준 보장이 아니다.
   `Phif()` / `PhifMutable()` 쌍(`Geometry.h:409-431`)이 유일하게 굳혀진 필드이고,
   그것이 나머지를 굳히는 본보기다.

둘 다 bit-identity 게이트가 달린 진짜 리팩터다. 그 둘을 하기 전에 공유를
주장하는 것은 **안전성 없이 레버를 주장하는 것**이다.
그래서 수신증이 `cohort_builds`/`cohort_hits`와 `geometry_builds`를 **둘 다** 낸다:
전자는 코호트 수, 후자는 여전히 `cases`. 한 숫자가 두 사실을 대신하게 두지 않는다.

### 2.4 XSLIB 캐시 키

`(path, size, mtime, ng)`였다. 일회성 프로세스에서는 옳다 — 프로세스가 라이브러리
교체 작업보다 짧으니까. **evaluator에서는 틀리고, 조용히 틀린다**: 교체 이후의
모든 케이스가 옛 파스를 재사용하고, 완벽히 자기일관적인 수를 내며, 어떤 수신증도
그것이 어느 라이브러리에서 왔는지 말하지 않는다.

이제 키가 파일 바이트의 SHA-256을 싣는다. `CaseKey`가 쓰는 **같은 transform**
(`include/chiffon/Sha256.h`)이고, `BatchLightResult::Sha256FileCached`는 **아니다** —
그쪽은 경로만으로 memoise하고 만료되지 않으므로 정확히 지우려는 그 staleness다.

**아직 열려 있는 것을 이름으로 적는다.** digest 자체는 `(path, size, mtime)`으로
memoise된다(M개 케이스가 34 MB를 M번 읽지 않기 위해). 크기가 같고 mtime 해상도
안에서 일어난 교체는 여전히 낡은 digest를 돌려준다. 그 구멍은 전보다 엄격히
작고(이제 크기도 같아야 한다), `RASBERY_XSLIB_DIGEST=always`가 acquisition당
파일 읽기 하나로 완전히 닫는다. `digest_computes`가 memo가 작동한다는 증인이다.

### 2.5 관련 없는 수정 하나 (`2a1161b`)

`main.cpp`를 컴파일해 보다가 발견했다. `src/Driver.h`의
`PH_PPR_RESET = PH_UPDDHAT + 1`은 `12052df`가 썼을 때 옳았다. `5883023`(WP9-A)이
그 사이에 SolveLoop phase 넷을 끼워 넣고 initialiser를 그대로 두었다.
그때부터 floor phase 여섯이 loop phase를 **일대일로 aliasing**했고,
`phaseName()`의 switch에 중복 case label이 넷 생겼다 — 중복 case는 ill-formed이고,
**`Driver.h`를 include하는 모든 translation unit이 컴파일에 실패한다.**
`PH_COUNT`도 넷 모자랐다.

initialiser를 지우면 `PH_SEARCH_APPLY` 다음에 붙는다 — 그 위 블록 주석이 줄곧
말해 온 바로 그것이다. **`5883023` 이후 트리는 빌드된 적이 없다.**
238에서 무엇을 하든 이 커밋이 먼저 들어가야 한다.

### 2.6 계약 테스트 (`tools/test_cohort_context_contract.py`)

정적 스캔 + 음성 대조군 11개, 그리고 **컴파일 반쪽**. 컴파일러가 있으면
`CohortKey.h`/`CohortContext.h`를 실제로 빌드해 진짜 digest 위에서 판정한다
(`GeometryInput`만 stub — 실제 `Geometry.h`는 pch → highfive → HDF5 C 헤더까지
끌고 오는데, 그것은 키의 성질이 아니라 빌드 시스템의 성질이다. stub이 실제
struct에서 표류하면 필드 목록 비교가 실패한다).

로컬 결과 (MSVC 14.50, `/std:c++20`), 전부 통과:

| 성질 | 기대 | 이유 |
|---|---|---|
| `permuted` | 같은 코호트 | **이것이 레버다.** 없으면 단계 2는 아무것도 사지 않고 다른 모든 수는 그대로다 |
| `holed` | 다른 코호트 | 점유가 바뀌면 맵이 바뀐다 |
| `albedo` / `mesh` / `library` | 다른 코호트 | |
| `ragged` | 같은 코호트 | 공백 차이로 코호트가 갈라지면 안 된다 |
| `batch` | 키에 안 들어감 | 후보가 움직이는 것 |
| `same_object` / `isolated` | 객체 동일 / 분리 | cross-cohort isolation |
| `quadrature_shared` | 두 코호트가 테이블 하나 | `(ndivxy, npins)`의 순함수 |
| `builds=2, hits=1, quad_builds=1` | | 회계 |

---

## 3. 로컬에서 한 것, 그리고 하지 않은 것

### 3.1 한 것

```bash
python tools/test_multi_gpu_dispatch.py        # 단계 1.5 + 음성 대조군 6개
python tools/test_fleet_tuner.py               # 두 형태 end-to-end
python tools/test_evaluator_contract.py        # + BATCH_HOST/프로토콜 계약
python tools/test_cohort_context_contract.py   # 정적 + 컴파일 반쪽
python tools/test_xslib_cache_contract.py
python tools/test_case_key_contract.py
```

MSVC `/Zs`(구문만): `src/PPR.cpp`, `src/XSSet.cpp` 통과.
`src/IO.cpp`, `src/main.cpp`는 `/D_inline=...`이 필요하다 — `IoWriter.h`의 멤버
`_inline`이 MSVC 키워드와 충돌한다(238의 WSL gcc 빌드에는 없는 문제).
그 정의를 주면 둘 다 통과.

### 3.2 하지 않은 것 — 정직하게

- **CUDA를 포함한 전체 빌드.** 로컬에 CUDA 툴체인이 없다.
- **수치 하나도.** digest, c/h, wall, VRAM, 전부 238이다.
- **PPR quadrature 공유의 bit-identity.** 산술은 줄 단위로 안 건드렸지만,
  그 주장을 확인하는 것은 `[TRAJECTORY].digest` 비교이지 diff가 아니다.

### 3.3 이 트리에서 이미 실패하던 계약 테스트 (내 것이 아니다)

기준 `3df4ea7`에서 `test_xe_anderson`, `test_gpu_phase_scheduler_contract`,
`test_ga_promotion_gate`, `test_cram_gpu_contract` 외 다수가 이미 실패한다.
`2c04a6e`(다른 작업자의 WP5-B/C)가 `test_cmfd_outer_kernels_contract`,
`test_nodal_constant_gpu_contract`, `test_ppr_gpu_contract` 셋을 새로 깼다
(worktree 대조로 확인: 이 셋은 `8437697`에서 통과, `2c04a6e`에서 실패).
**이 커밋들은 새 실패를 만들지 않았다.**

---

## 4. 238 런북

**전부 GPU0만, `CUDA_VISIBLE_DEVICES=0`을 자식까지 고정.**
텔레메트리 실행과 wall 측정 실행을 섞지 말 것. 첫 실행은 warm-up으로 버릴 것.

### 4.0 먼저 빌드가 되는지 확인한다

`2a1161b` 이전 트리는 컴파일되지 않는다(§2.5). 이것이 통과하지 않으면
아래는 전부 무의미하다.

```bash
cmake -S . -B build -DRASBERY_ENABLE_CUDA=ON -DRASBERY_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 4.1 feature-off identity (B0, 먼저)

`--no-evaluator`는 옛 경로를 그대로 탄다. 단계 1.5 이전 tip과 바이트 동일해야
한다 — 깨지면 dispatcher 리팩터가 청크 경로를 건드린 것이다.

```bash
python tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 1 --batch-width 64 \
    --jobs g1.jobs --result light --no-evaluator --workdir off_new -- ./build/RASBERY \
    > off.new.log
diff <(grep -E '^\[RASBERY\]\[(TRAJECTORY|PHYSICS_MODE)\]' off.old.log) \
     <(grep -E '^\[RASBERY\]\[(TRAJECTORY|PHYSICS_MODE)\]' off.new.log)   # 기대: 빈 diff
```

### 4.2 본 실험 — 8 × M8 + MPS, 두 형태

큐는 같은 매니페스트(64 케이스 권장 — `--claim`이 워커당 여러 청크를 내야
`images < waves`가 의미를 가진다). 각 arm 5회, median과 p10/p90, arm 순서 교대.

```bash
# A. 지속 evaluator (기본값)
python tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 8 --total-width 64 \
    --mps --jobs g1.jobs --result light --claim 8 \
    --workdir wp8b_eval --cwd /path/to/decks -- ./build/RASBERY

# B. 대조군 — 청크마다 프로세스
python tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 8 --total-width 64 \
    --mps --jobs g1.jobs --result light --claim 8 --no-evaluator \
    --workdir wp8b_chunk --cwd /path/to/decks -- ./build/RASBERY
```

### 4.3 판정표 — 이 표의 칸이 채워지기 전에는 어떤 주장도 없다

| 지표 | 출처 | 채택 기준 |
|---|---|---|
| 케이스별 digest, A vs B | A: `[EVALUATOR][CASE].digest` / B: `[TRAJECTORY].digest` | **multiset 동일. B0.** 하나라도 다르면 채택 중단 |
| `images` / `waves` | `[MULTI_GPU][TOTAL]` | A: `images == 8`(워커 수), `waves == 청크 수`. B: `images == waves` |
| 제거된 프로세스 stand-up | `(B.images − A.images) × outside_drive` | 보고. 이것이 레버의 산술적 크기다 |
| 전체 c/h | `[TOTAL].cases_per_hour` | 계획 WP8 게이트: **≥ +5 %**. 3 % 이하는 노이즈 |
| 세대별/wave별 wall | `[EVALUATOR][WAVE].wall_s` | 첫 wave와 이후 wave의 차이가 stand-up 비용이다 |
| `cohort_builds` | `[EVALUATOR]`, `[COHORT]` | **1**. 케이스 수·wave 수와 무관해야 한다 |
| `cohort_hits` | 동상 | `cases − 1` |
| `ppr_quadrature_builds` | `[EVALUATOR]`, `[COHORT]` | **1** (M64 wave에서 64가 아니라) |
| `xslib_loads` | `[XSLIB_CACHE]`, `[EVALUATOR]` | **워커당 1** |
| `xslib_digest_computes` | `[XSLIB_CACHE]` | **워커당 1.** 케이스 수와 함께 자라면 memo가 죽은 것이고 케이스마다 34 MB를 읽고 있다 |
| `slot_duplicates` / `stale_tenants` / `double_releases` | `[EVALUATOR]`, `[REFILL]` | **전부 0.** 아니면 그 측정은 무효 |
| `pin_live_ranges_between_waves` | `[EVALUATOR]` | **0** |
| `arena_releases` | `[EVALUATOR]` | 워커당 **1** |
| `fatal_waves` / `evaluator_restarts` | `[TOTAL]` | 정상 실행에서 **0** |
| VRAM 피크 | `nvidia-smi` | B와 같아야 한다 — 단계 2는 device에 아무것도 더하지 않는다 |
| RSS | `/usr/bin/time -v` | A가 B보다 크지 **않아야** 한다(코호트는 1.4 MB, 지운 것은 63 × 1.4 MB) |

### 4.4 restart 시험 (별도 실행, 2분)

독이 든 케이스 하나로 워커를 죽이고, 나머지가 재큐되어 끝나는지 본다.
실제 독은 존재하지 않는 덱이면 충분하지 않다 — 그것은 **케이스 실패**이지
프로세스 사망이 아니다(그리고 그 구분이 §1.3의 요점이다). 프로세스를 죽이려면
`RASBERY_GPU_FULL=1`로 fail-closed를 유발하거나, arena가 서지 못할 폭을 요구한다.

```bash
# 케이스 실패(프로세스 생존)만 확인하는 값싼 버전
python tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 1 --batch-width 8 \
    --jobs g_bad.jobs --result light --workdir wp8b_fail -- ./build/RASBERY
```

기대: `[EVALUATOR][CASE]` 한 줄이 `"status":"failed"`, `[PROC].restarts == 0`,
`[PROC].processes == 1`, 나머지 케이스 출력 전부 존재, rc != 0,
`[MULTI_GPU][FAIL]`에 그 덱 이름.

프로세스 사망 경로는 로컬 계약 테스트가 fake child로 이미 몰아붙였다(§1.5).
238에서 굳이 재현할 필요는 없고, **자연 발생하면 `[EVALUATOR][FATAL]` 줄이
그것을 설명해야 한다.** 설명하지 못하면 그것이 결함이다.

### 4.5 cross-cohort isolation (30분)

서로 다른 geometry 두 덱군을 **번갈아** 한 evaluator에 보낸다
(계획 WP8 isolation 항목 4). 매니페스트를 A/B/A/B로 짜면 된다.

기대: `cohort_builds == 2`, `cohorts == 2`, 각 덱의 digest가 단일 코호트
실행에서와 동일, `ppr_quadrature_builds`는 두 코호트의 `(ndivxy, npins)`가
같으면 **1**.

---

## 5. 단계 3은 왜 시작하지 않았나

계획 단계 3은 `CaseContext::reset(candidate)` — nxyz 크기 XS/isotope/work 배열의
capacity를 유지하고 값만 다시 채우기, generation poison으로 reset coverage 검사.

**그 전제가 단계 2에서 성립하지 않는다고 밝혀졌다.** 재사용할 버퍼의 대부분은
`Geometry`와 `XSSet`이 소유하고, `Geometry::Initialize`는 재실행이 불가능하며
(§2.3), 접근자는 mutable 참조를 준다. 버퍼를 재사용하되 그 둘을 고치지 않는 것은
정확히 §2.3이 경고하는 "안전성 없이 레버를 주장하기"다.

단계 3 이전에 해야 할 순서:

1. `Geometry`를 RAII 컨테이너로 — `Initialize`에 `delete[]` 프롤로그 또는
   `std::vector`/`milk::Vector` 소유권. bit-identity 게이트 하나.
2. 접근자를 `Phif()`/`PhifMutable()` 본보기대로 굳히기 — 그래야 공유가
   컴파일러에게 검사받는다.
3. 그 다음에야 `CohortContext`가 Geometry 맵을 실제로 들 수 있고,
   `CaseContext::reset`이 무엇을 다시 채워야 하는지가 타입으로 정의된다.

`geometry_builds == cases`가 그 셋이 아직 없다는 기계 판독 가능한 증거로
수신증에 남아 있다.

---

## 6. 남은 구멍

1. **단계 1의 §8 구멍 중 닫힌 것**: `run_multi_gpu_batch.py` 미개조(닫힘, `8437697`),
   worker 교체 부재(닫힘 — 단, 판정은 "프로세스가 죽었다"이지
   "CUDA 상태가 오염되었다"가 아니다. 후자를 판정할 신호는 여전히 없다),
   XSLIB 키(닫힘, `f32e646`).
2. **`fidelity`는 여전히 프로세스 속성**이고, 이제 워커 속성이다. 한 캠페인에서
   두 fidelity를 원하면 evaluator 두 벌이다.
3. **wave 타임아웃이 없다.** 사망은 EOF로 감지하지만 *멈춘* 자식은 감지하지
   않는다. calibration의 `--tune-budget-s`는 claim 전에만 검사되므로 멈춘 wave를
   끊지 못한다.
4. **`Driver.h`의 `p.xslib_digest`는 여전히 `BatchLightResult::Sha256FileCached`**를
   쓴다 — 경로만으로 memoise하고 만료되지 않는 그 캐시다. 장수명 프로세스에서
   case key가 낡은 라이브러리 digest를 실을 수 있다. WP10.1의 문제이고,
   고치려면 `XsLibraryContentDigest`(이 커밋이 노출했다)로 바꾸면 된다.
5. **`PPR::_isfuel_stage`가 크기로 키잉된다**(`PPR.cpp:97`). 오늘은 PPR 객체가
   Driver마다 하나라 안전하지만, 언젠가 PPR을 코호트로 올리면 같은 geometry의
   다른 loading pattern이 낡은 fuel mask를 재사용한다. 그때 고칠 것.
6. **`Geometry::_comps`는 할당되고 한 번도 쓰이지 않으면서 device 업로드
   목록에 있다**(`GpuPhysicsArenaLayout.h:163`, `GeometryRegion::Comps`).
   초기화되지 않은 메모리를 업로드하고 있다. 내 작업이 아니지만 적어 둔다.

---

## 7. 파일 색인

| 파일 | 무엇 |
|---|---|
| `tools/run_multi_gpu_batch.py` | `EvaluatorSession`(pipe 프로토콜·재시작), `_run_wave_chunk`(재큐 1회), `--no-evaluator`, `--evaluator-max-restarts`, `images`/`waves` 수신증 |
| `tools/fake_rasbery_child.py` (신규) | 두 자식 형태를 하나의 수신증 코드로. `FAKE_RASBERY_*` 음성 대조군 knob |
| `src/EvaluatorServer.h` | wave당 `[BATCH_HOST]`, 프로세스 수신증의 cohort/digest 필드 |
| `src/CohortKey.h` (신규) | `geometryPayload` / `occupancyMask` / `keyOf` |
| `src/CohortContext.h` (신규) | `Descriptor` / `Context`(전부 const) / `acquire` / `acquirePinQuadrature` / `[COHORT]` |
| `src/PprQuadrature.h` (신규) | `QuadPoint`/`PinOverlap`/`PinQuadInfo`/`PinQuadTable`, PPR.h에서 분리 |
| `src/PPR.{h,cpp}` | 테이블을 소유하지 않고 빌린다. 산술 무변경 |
| `src/IO.{h,cpp}` | `geometry_input`이 확정된 자리에서 코호트 획득 |
| `src/XsLibrary.h`, `src/XSSet.cpp` | content digest 키, `digest_computes`, `RASBERY_XSLIB_DIGEST` |
| `src/main.cpp` | 세 teardown 블록 전부에 `[COHORT]` |
| `src/Driver.h` | `2a1161b` — phase enum 충돌 수정 |
| `tools/test_cohort_context_contract.py` (신규) | 정적 7부 + 음성 대조군 11 + 컴파일 반쪽 13성질 |
