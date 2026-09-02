# WP10.7 — 입장(admission) 시점의 device residency 확립

- 대상 브랜치: `codex/exact-throughput-campaign` (이 커밋의 부모 `e9b2f7b`)
- 증거: 238 GPU1, build `0054838`, `E:\rasbery_runs\2026-08-30\238\pricing_388e8f2.md` 블록 (38) Phase 2 (arm A)
- 건드린 파일: `src/GpuFullContract.h`, `src/Driver.h`, `src/EvaluatorServer.h`,
  `tools/soak_run.py`, `tools/promotion_gate.py`, `tools/fake_rasbery_child.py`,
  `tools/test_evaluator_residency_contract.py`,
  그리고 후속 커밋 WP10.7b에서 `src/XSSet.cpp` (§2.7)
- 손대지 않은 파일: `src/XSSet.h`, `src/CudaXsReconBackend.*`, nodal kernel 헤더

---

## 1. 메커니즘 — 무엇이 건너뛰어졌는가

### 1.1 증거

238 GPU1에서 20세대 × width 16, PROD + `RASBERY_GPU_FULL=1`, arm A
(`RASBERY_ARENA_PERSIST` unset) 소크가 **13건**의 fail-closed 사망을 냈다.
run-level 영수증은 이렇게 끝났다.

```
{"gpu_full":true,"cmfd_fallbacks":0,"outer_fallbacks":9,"nodal_fallbacks":0,
 "flatxs_fallbacks":4,"xe_fallbacks":0,"ppr_fallbacks":0,"cram_fallbacks":0,
 "allowed_refusals":{"wielandt_warmup":0},"contract_pass":false,
 "first_violation":"subsystem=outer site=Driver: outer segment pre-arm
                    reason=no_residency"}
```

- **flatxs 4건** — 세대 6/9/12의 `promote` 스텝과 세대 19의 case 14.
  네 건 모두 `statepoints:0, outers:0, th_updates:0`, `wall_s` 0.18–0.43 s.
  **0.2초는 deck 파싱 하나의 시간**이므로, 이 케이스들은 `InitXS`가 부르는
  **첫 `XSSet::UpdateFlatXS`에서 죽었고 statepoint를 한 번도 돌지 않았다.**
- **outer 9건** — `subsystem=outer ... reason=no_residency` (세대 6, 9, 12,
  15×3, 16, 18×2). 이쪽이 오히려 더 큰 범주인데, 소크의 판정은 이 숫자를
  **읽지 않았다** (§4.3).
- `arena_rebuilds` 증분은 세대마다 정확히 +17 = 16 케이스 + promote 1개,
  즉 **케이스당 정확히 한 번의 live region free + 재배치**.

### 1.2 건너뛴 단계 (file:line)

**`src/Driver.h:5015`(현재 트리의 `establishDeviceResidency` 호출 자리) —
그 호출이 없던 자리.** `Driver::Run`은 deck 파싱 직후 `gpu::outerGpuEnabled()` 아래에서
outer segment만 세우고(`gpu::rasberyStandUpOuterSegment`, 반환값 폐기),
곧바로 `cross_sections.InitXS(...)`로 넘어갔다.

**FlatXS 팔에는 입장 문(door)이 아예 없었다.** `XsReconBackend`는
`XSSet::TryUpdateFlatXSGpu` 안에서 **first touch**로 생성된다
(`src/XSSet.cpp:3261-3271`). 즉 "이 케이스의 flat-XS device residency가
존재하는가"를 **처음으로 묻는 주체가, 그것이 없다는 이유로 케이스를 죽이는
fail-closed 가드 자신**(`src/XSSet.cpp:3442`)이었다. 그리고 그 가드는 이유를
말할 수 없다 — 백엔드 자신의 이유(`XsReconBackend::status()`:
`"no CUDA device: ..."`, `"stream: ..."`, `"scalar buffer allocation failed"`)는
**프로세스 전역 `std::call_once` 경고**로만 나간다(`src/XSSet.cpp:3265-3271`).
20세대를 사는 상주 evaluator에서는 **첫 번째 케이스만 이유를 찍고 나머지 셋은
아무것도 찍지 않는다.** 16k줄 arm-A 로그가 정확히 그 상태다.

**두 번째 자리: `src/Driver.h:2427` 과 `src/Driver.h:3833`.** 고치기 전에는
두 곳 모두 `if (gpu_outer_may_arm) armOuterSegment(ctx, eigv, residual);`
한 줄이었고 — **반환값이 버려졌다.** `armOuterSegment` → `rasberyBindOuterResidency` →
`CudaOuterSegment::bindResidency`는 진입하자마자
`_impl->residency_bound = false`로 내리고(`src/CudaOuterGraph.cu:1118`)
실패 이유를 `_impl->status`에 남긴 뒤 false를 돌려준다. 호출자가 이를 읽지
않으므로, 한 줄 아래의 post-arm 사다리가 같은 사실을
`residency_bound == 0` → **`no_residency`** 라는 총칭으로 **재유도**한다.
9건이 죽은 그 문자열은 **증상이고, 원인은 버려졌다.**

### 1.3 가설 판정

| 가설 | 판정 | 근거 (이 트리에서) |
|---|---|---|
| (a) promote가 다른 CaseFidelity/StatepointGrid를 갖고 residency plan이 재구축되지 않음 | **기각** | 어떤 device 팔도 `CaseFidelity`를 읽지 않는다. `_fidelity`는 `ReadInput`(burnup grid)과 `SolverContext`(tolerance)까지만 간다. `TryUpdateFlatXSGpu`도 `armOuterSegment`도 볼 수 없다. 그리고 네 번째 사망 `g0019c0014`는 **일반 strict 케이스**로, 형제 15개와 grid가 같다 |
| (b) WP18 slot 리셋이 첫 입장만 세우는 residency 플래그를 지움 | **기각** | FlatXS 백엔드는 XSSet 당 · Driver 당 · 케이스 당이라 지울 per-slot 플래그 자체가 없다. outer 쪽도 `armOuterSegment`가 **모든** SolveLoop/ReconvergeFlux 진입마다 재바인드한다(첫 진입만이 아님) |
| (c) no-persist 팔에서 케이스마다 device block을 해제/재배치하고, 다음 입장이 그 확립을 기다리지 않음 | **채택** | residency 확립이 **문 없는 first-touch**다: 사전 확인도, 이유도, 영수증도 없다. `arena_rebuilds` = 케이스당 정확히 1 (§1.1) 이 그 위에 얹힌다. 3/4가 promote 스텝인 것도 이와 정합적이다 — promote는 wave의 **마지막** job(16 lanes에 18 jobs)이라 **항상 재활용된 lane**에서 돈다 |
| (d) outer/no_residency 9건이 같은 뿌리 | **같은 부류, 다른 기전** | 둘 다 "확립 단계의 실패가 이름을 갖지 못한다"이지만, outer 쪽은 §1.2의 버려진 반환값이다. 이번 커밋에서 둘 다 닫는다 |

**타이밍이 판정을 굳힌다.** 0.18–0.43 s는 첫 `UpdateFlatXS`이고, 그 시점에
residency를 확립하려 시도한 코드는 그 호출 자신뿐이었다.

---

## 2. 고친 것

### 2.1 입장 문 — `Driver::establishDeviceResidency` (`src/Driver.h`)

- **케이스마다 무조건 1회**, physics 이전, `InitXS` **앞**에서 실행.
  `static bool` / `std::call_once` / lane·slot 키 **없음** — 재활용된 slot과
  새 slot이 같은 호출을 지나간다 (계약 테스트가 이를 잡는다).
- 확립은 `XSSet::EnsureBackend()`(`src/XSSet.h:705`)로 간다. 이 함수는 구조상
  멱등이므로 "확립"과 "재확립"이 한 호출이다.
- **영수증 `[RASBERY][RESIDENCY]`** 를 케이스마다 찍는다 — 성공에도 찍는다.
  "팔이 켜져 있고 확립됐다"와 "묻지도 않았다"가 같아 보이면 안 된다는,
  이 트리의 G0 규칙 그대로. 상주 프로세스의 N번째 케이스도 첫 케이스만큼
  진단 가능해진다 (`call_once` 경고가 못 갖는 성질).
- **fail-closed 유지, 다만 이유를 지목한다.** 요청된 팔의 residency를 확립할
  수 없으면 게이트 하에서 **문에서** 던진다 —
  `RASBERY_GPU_FULL_REQUIRE_RESIDENCY(FlatXs, "Driver: admission residency",
  <XsReconBackend::status()>)`.

### 2.2 `armOuterSegment`의 반환값을 읽는다 (`src/Driver.h` 두 호출부)

```
const bool gpu_outer_arm_ok = gpu_outer_may_arm && armOuterSegment(ctx, eigv, residual);
if (gpu_outer_may_arm && !gpu_outer_arm_ok)
    gpufull::nameFirstFallback(gpufull::Subsystem::Outer, "Driver: outer segment arm",
                               gpu::rasberyOuterSegment(gpu_outer_claim.slot).status().c_str());
```

**사다리가 여전히 결정한다.** 아래의 `RASBERY_GPU_FULL_GUARD_ALLOWED`는 그대로
던지고 카운터도 그대로다 — 이번 변경은 **동작 변경이 아니라 진단**이다.
실패한 단계가 먼저 이름을 갖고, 영수증이 증상 대신 원인을 싣는다.

### 2.3 게이트 프리미티브 (`src/GpuFullContract.h`)

`gpufull::requireResidency` / `RASBERY_GPU_FULL_REQUIRE_RESIDENCY`.
seam 가드와 다른 점은 **카운트 규칙** 하나다.

| | 게이트 ON | 게이트 OFF |
|---|---|---|
| 문 | count 1 + throw (케이스는 seam에 도달하지 않음) | **name only, count 0** |
| seam(기존) | 도달하지 않음 | 종전대로 count 1 |

→ **두 팔 모두에서 per-subsystem 카운트가 종전과 같다.** OFF 팔에서 문이 세면
집계가 두 배가 되고 캠페인이 이 숫자들을 읽는 feature-off identity가 깨진다.

### 2.4 `first_violation` = 진짜로 첫 번째 (`src/GpuFullContract.h`)

종전에는 프로세스 전역 문자열 하나에 대한 **CAS 승자**였고, CAS는 *문자열을
만드는 곳*에서 잡힌다 — 사건이 일어난 곳이 아니라. 16개 lane이 마이크로초
간격으로 fallback하면, `count()`와 문자열 사이에서 스케줄 아웃된 스레드가
**자기 사건이 먼저였는데도** 래치를 잃는다. 238 arm-A에서 flatxs와 outer가
**같은 세대 6**에 함께 터졌고, 영수증에는 둘의 순서를 가릴 방법이 없었다.

- `count()`가 **사건 지점에서** 단조 ordinal을 찍는다 (relaxed fetch_add 1회 +
  thread-local store 1회).
- `nameFirstFallback()`이 그 스탬프를 **소비**하고(`takePendingOrdinal`,
  소비가 핵심 — 남은 스탬프는 나중 site를 이른 ordinal로 기록하게 만든다),
  subsystem별 첫 site 슬롯에 기록한다.
- `firstViolation()`은 subsystem 테이블에서 **최소 ordinal**을 골라 만든다.
- 영수증에 `first_violation_seq`와 `violations{}`가 붙는다:

```
"first_violation_seq":1,
"violations":{"cmfd":{"count":0,"seq":0,"site":null,"reason":null},
              "outer":{"count":9,"seq":2,"site":"Driver: outer segment pre-arm",
                       "reason":"no_residency"},
              ...,
              "flatxs":{"count":4,"seq":1,"site":"XSSet::UpdateFlatXS",
                        "reason":"declined"}, ...}
```

이 로그의 형태(flatxs 4 / outer 9)는 계약 테스트가 **컴파일해서 실행**하며
확인한다 — 238 인터리브(flatxs 사건이 먼저, outer 문자열이 먼저)를 재현하고
`first_violation`이 flatxs를 지목하는지 본다.

### 2.5 `promote`는 한 개의 문으로 (`src/EvaluatorServer.h`)

`op:"promote"`에는 문이 **두 개**였다. rolling 모드에서는 읽는 즉시
`processCaseFidelity()`에 대해 해석되고, wave 모드에서는 `wave` 라인을 기다려
**그 wave의 선언**에 대해 해석됐다 — 같은 요청이 모드에 따라 다른 뜻이었다.

구체적 구멍: promote는 파싱 시점에 `strict` / `full` 출력 / full burnup grid
**셋만** 뒤집고 `flux_mult` · `xe_mult` · `loose_settle`은 뒤집지 않는다.
따라서 staged tolerance를 선언한 wave가 그 안의 promotion에 그것을 물려주었고,
**acceptance lane의 재평가**(RunContract.h 기준 `acceptance_eligible`인
유일한 행)가 `policy:"strict"`를 보고하면서 screening tolerance로 수렴했다.

→ `applyWaveFidelityDefault(wave, request, promoted)`가 `promoted`면 즉시
반환한다. wave를 거부하지 않는 이유: 혼합 wave는 WP10.3에서 설계상 합법이고,
screening 세대 안의 promotion이야말로 이 op가 쓰여진 바로 그 경우다.

### 2.6 하네스 (`tools/soak_run.py`, `tools/promotion_gate.py`)

- `GPU_FULL_FALLBACKS`에 **`outer_fallbacks` 추가.** arm A는 `outer_fallbacks:9`
  — flatxs 4보다 큰 범주 — 를 찍었는데, 소크가 읽지 않은 유일한 카운터가
  그것이었다.
- 소크 리포트에 `gpu_full_contract` 블록(`contract_pass`, `first_violation`,
  `first_violation_seq`, `violations`, `allowed_refusals`). 게이트 하에서
  `contract_pass:false`면 **problem**으로 올린다 — 카운터에서 재유도하는 대신
  영수증 자신의 판정을 읽는다 (미래의 count-only seam이 빠져나가지 못하게).
- `promotion_gate.py`: 소크가 pass여도 `gpu_full_contract.contract_pass`가
  false면 default-ON 승격을 **차단**한다. GPU 팔이 CPU로 떨어진 런에서 승격된
  플래그는 CPU 숫자로 승격된 플래그다. (필드가 없는 옛 블록은 무해 — 이 파일의
  "언급되지 않은 것은 가정하지 않는다" 규칙 그대로.)
- `tools/fake_rasbery_child.py`가 `outer_fallbacks`를 찍는다. 실제 바이너리가
  찍는데 스탠드인이 찍지 않는 필드는 이 트리의 어떤 테스트도 몰지 않는
  필드이고, 그것이 arm A가 `outer_fallbacks:9`에 도달하고도 소크 판정이
  그 숫자를 보지 않은 경로다.

### 2.7 `src/XSSet.cpp` — WP10.7b에서 적용됨

처음에는 WP21-B/C가 같은 파일을 편집 중이라 `docs/patches/wp10_7_xsset.patch`로
넘겼다. 그 잠금이 풀린 뒤 **WP10.7b에서 적용하고 패치 파일은 삭제했다.**

내용: `XSSet::UpdateFlatXS`의 fail-closed 가드가 백엔드 자신의 `status()`를
이유로 싣는다. WP10.7의 필수 경로는 아니다 — 게이트 하에서는 문이 이미 그보다
먼저 진짜 이유로 던진다 — **게이트 OFF 팔**(영수증이 전부인 곳)과 문이
`ready`라고 답한 뒤 **런 중간에 오는 decline**을 위한 나머지 절반이다.

이유 문자열은 가드 호출 안이 아니라 seam 주석 **위**에서 만든다.
`tools/test_gpu_full_fail_closed.py`의 WINDOW(seam 앵커 ±8줄)가 "가드가 다른
분기로 흘러가지 않았다"를 고정하고 있고, 주석을 담으려고 그 창을 넓히는 것은
그 성질을 조판과 맞바꾸는 일이다.

---

## 3. 계약 테스트

`tools/test_evaluator_residency_contract.py` (8개 속성, 전부 negative control
동반, 두 개의 **live** 반쪽):

1. 입장 문이 존재하고 **무조건**이다 (latch 스캔은 주석·문자열을 마스킹한
   본문에서 수행 — 주석이 `std::call_once`라는 단어를 말해서 통과/실패가
   갈리면 안 된다).
2. 문이 `InitXS` **앞**에서 호출된다.
3. **live**: 출하되는 `establishDeviceResidency` 본문을 스텁과 함께 컴파일해
   실행 — 팔 OFF면 침묵, 확립되면 영수증, 게이트 OFF면 name하되 count하지 않음,
   그리고 이유 문자열이 백엔드의 것.
4. `armOuterSegment`의 반환값이 **두 호출부 모두**에서 읽힌다.
5. `promote`가 한 문을 지난다 (wave default 조기 반환 + rolling이
   `processCaseFidelity()`에 대해 해석 + promote가 세 tolerance 필드를 스스로
   뒤집지 **않는다**).
6. **live**: 238 인터리브에서 `first_violation`이 flatxs를 지목,
   `first_violation_seq == 1`, flatxs 4 / outer 9, site가 첫 사건의 것.
7. 영수증이 subsystem마다 count·seq·site·reason을 싣는다.
8. 소크가 7개 subsystem 전부와 영수증의 `contract_pass`를 읽는다.

**로컬 결과 (전부 PASS)**

```
test_evaluator_residency_contract    PASS (live halves driven)
test_gpu_full_fail_closed            PASS
test_evaluator_rolling_contract      PASS
test_evaluator_mem_receipt_contract  PASS
test_soak_receipt_schema_contract    PASS
test_enum_alias_contract             PASS
test_dependent_template_contract     PASS
test_evaluator_contract / test_soak_run / test_promotion_gate /
test_ppr_gpu_master_mode_contract / test_receipt_counters_live /
test_case_fidelity_contract / test_result_fidelity_contract /
test_device_outer_exactness_contract / test_arena_persist_contract /
test_batch_refill_contract / test_v5_defaults_contract /
test_exact_only_contract / test_harness_env_parity            PASS
```

돌연변이 검증(수동): ordinal 스탬프 삭제, promote 조기 반환 삭제,
`armOuterSegment` 반환값 폐기 복원, `firstViolation`을 래치로 되돌리기 —
네 가지 모두 계약 테스트가 잡았고, live 영수증 검사가 마지막 것을 잡았다.

---

## 4. 238 GPU1 런북

로컬 계산 금지. 아래는 238에서, 출력은 `E:`(호스트 쪽 `~/gates/wp10_7/`)로.

### 4.1 빌드

```bash
# 238, WSL micromamba CUDA 12.6 환경
cd ~/rasbery && git fetch && git checkout <이 커밋의 해시>
cmake -S . -B build-wp10_7 -DCMAKE_BUILD_TYPE=Release
cmake --build build-wp10_7 -j
ctest --test-dir build-wp10_7 --output-on-failure
python3 tools/test_evaluator_residency_contract.py
```

`ctest` 그린 + 계약 테스트 PASS 전에는 소크를 돌리지 않는다.

### 4.2 20세대 소크, 두 팔

GPU1만 쓴다(`CUDA_VISIBLE_DEVICES=1`). GPU0에 다른 작업이 있으면 throughput
숫자는 오염된다 — 블록 (38)(c)가 그 사례다. **RSS/카운터 결론은 영향 없음.**

```bash
GATES=~/gates/wp10_7 && mkdir -p $GATES

# 공통 env (PROD + GPU_FULL)
common() {
  export CUDA_VISIBLE_DEVICES=1
  export RASBERY_GPU=1 RASBERY_GPU_FULL=1
  # PROD 팔 세트는 v6 env 파일 그대로
  source ~/gates/env_v6.sh
}

# arm A -- RASBERY_ARENA_PERSIST unset
( common; unset RASBERY_ARENA_PERSIST
  python3 tools/soak_run.py --generations 20 --width 16 \
      --workdir $GATES/soak_off --report $GATES/soak_off.json \
      > $GATES/soak_off.log 2>&1 )

# arm B -- RASBERY_ARENA_PERSIST=1
( common; export RASBERY_ARENA_PERSIST=1
  python3 tools/soak_run.py --generations 20 --width 16 \
      --workdir $GATES/soak_on --report $GATES/soak_on.json \
      > $GATES/soak_on.log 2>&1 )
```

### 4.3 게이트 — 두 팔 모두에서

| 지표 | 요구값 | 어디서 읽나 |
|---|---|---|
| `zero_receipts.flatxs_fallbacks` | **0** | `soak_*.json` |
| `zero_receipts.outer_fallbacks` | **0** | `soak_*.json` (WP10.7에서 새로 assert) |
| `gpu_full_contract.contract_pass` | **true** | `soak_*.json` |
| `gpu_full_contract.first_violation` | **null** | 같음 |
| `gpu_full_contract.violations.*.count` | 전부 **0** | 같음 |
| `[RASBERY][GPU_FULL][VIOLATION] ... reason=no_residency` | 로그에 **0건** | `soak_*.log` |
| `cases_reported` | `cases_requested`와 같음 (poison 20 제외) | `soak_*.json` |
| `restarts` | **0** | `soak_*.json` |
| RSS 기울기 | post-warmup **≤ 8.0 MB/gen** | `growth` 블록 |
| VRAM 기울기 | **0.0 MB/gen** (arm A 기준선 유지) | `growth` 블록 |
| arm B 추가 | `device_pool_hits` 증가, `arena_rebuilds` 평탄 | per-generation `mem` |

빠른 확인:

```bash
for a in off on; do
  echo "== arm $a"
  python3 - "$GATES/soak_$a.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
c=r.get("gpu_full_contract",{})
print(" pass:",r["pass"],
      "flatxs:",r["zero_receipts"].get("flatxs_fallbacks"),
      "outer:",r["zero_receipts"].get("outer_fallbacks"),
      "contract_pass:",c.get("contract_pass"),
      "first:",c.get("first_violation"))
print(" rss slope:",r["growth"])
for p in r["problems"]: print("  problem:",p)
PY
  grep -c "no_residency" "$GATES/soak_$a.log" || true
  grep -c "\[RASBERY\]\[RESIDENCY\]" "$GATES/soak_$a.log" || true
done
```

`[RASBERY][RESIDENCY]` 줄 수는 **실행된 케이스 수와 같아야 한다**(세대당 17,
20세대 = 340 + isolation recheck). 이보다 적으면 문을 건너뛴 입장이 있다는
뜻이고, 그것이 이 work package가 닫으려는 바로 그 상태다.

### 4.4 RSS 기울기 보고

arm A의 종전 결과는 post-warmup **12.76 MB/gen**(예산 8.0 초과), second-half
3.69 MB/gen — 증가가 런 앞쪽에 몰려 있었다. WP10.7은 RSS를 겨냥한 변경이
아니므로 **개선을 주장하지 않는다.** 보고 형식만 고정한다:

```
arm A: post_warmup <x> MB/gen, second_half <y> MB/gen, budget 8.0  -> PASS/FAIL
arm B: post_warmup <x> MB/gen, second_half <y> MB/gen, budget 8.0  -> PASS/FAIL
```

RSS가 여전히 FAIL이면 그것은 **별건**으로 남기고, 이 블록의 판정은
flatxs/outer/contract_pass 세 줄로 내린다. 두 문제를 한 판정에 섞으면 어느
쪽도 종결되지 않는다.

### 4.5 회귀 확인 (feature-off identity)

`RASBERY_GPU_FULL` unset · 모든 GPU 팔 unset 으로 단일 deck 1회:
`[RASBERY][RESIDENCY]` 줄이 **나오지 않아야** 하고
(`establishDeviceResidency`의 첫 두 predicate),
trajectory digest가 이 커밋 이전과 **동일해야** 한다.
