# WP 계획 검토와 추적표 — `GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md`

| 항목 | 값 |
|---|---|
| 검토 대상 | `docs/GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` (1,658행, 원본과 verbatim) |
| 계획의 고정 SHA | `8b8f18e1c837121bd7873b43c207dafeda632edd` |
| 검토 시점 트리 | `codex/exact-throughput-campaign` @ `8bd5112` (= `8b8f18e` + Task 16 GPU CRAM). 검토 도중 하네스 커밋 `7099e54`가 착지했고, 아래 하네스 인용은 그 이후 줄번호다. |
| 검토 방법 | 정적 소스 검토만. **이 검토에서 서버 238 실행·빌드·프로파일은 수행하지 않았다.** |
| 작성 | 2026-08-31 KST |
| 갱신 | 2026-08-31 KST — R5·F9·F10·F11 종결. `5ccf879`(하네스 감사 전환), `666e123`(죽은 receipt 필드 배선). 여전히 **로컬 빌드 없음**: §7과 §8-7)이 238에서 확인할 것을 적는다. |
| 갱신 | 2026-09-01 KST — **R2 종결**(소스의 9칸 채택, 파서는 칸 수를 세지 않는다). WP3 계약 하드닝 `79c1880`, WP4 튜너 `6ad1de7`. 238 L5 행렬 실측이 들어왔다(docs/W4_L5 §4.8): control 578 c/h로 원시 582 재현, 8×M8+MPS **878 c/h = 1.519×**. §9에 WP3 가격 평가 runbook을 적었다. 여전히 **로컬 빌드 없음**. |

> 이 문서는 두 가지를 한다. (1) 계획 §4.3/§4.4가 트리에 대해 주장하는 사실을 `file:line`으로 확인하거나 반박한다. (2) WP0–WP11 추적표를 만든다. 추적표는 이후 계속 갱신한다.

---

## 0. 한 줄 요약

계획의 핵심 판단 — **“다음 이득은 새 커널이 아니라, 이미 착지한 기능의 가격 평가와 계약 정리에서 나온다”** — 는 소스와 일치한다. 다만 계획이 “검증·하드닝”이라고 부른 WP3은 트리에서 **이미 계약 테스트까지 존재**하고, 계획이 언급하지 않은 **GPU CRAM(WP9-B)이 이미 착지**했으며, 계획의 모든 M64 절대값과 §3.3 Amdahl 지분은 **하네스 결함(24 lane 캡, 115.6 vs 582 c/h)이 있던 dispatcher로 측정**되었다. 마지막 항목이 가장 크다: **WP5가 “FlatXS 39.9%가 최우선”이라고 결정하기 전에 §3.3을 고친 하네스에서 다시 재야 한다.**

---

## 1. §4.4(CMFD compaction) 검증 — 계획이 옳고, 트리는 계획보다 더 나가 있다

계획 §4.4의 전제부터 확인했다. 종합 보고서는 실제로 compaction을 미착수로 분류한다: `docs/GPU_RASBERY_PERFORMANCE_AND_ARCHITECTURE_REPORT_20260830_KO.md:556`(“compaction 미착수”)와 §7.2 `:739`, `:743`(“아직 손대지 않은 것”). 소스는 그렇지 않다.

**F1. `RASBERY_GPU_CMFD_COMPACT`은 존재하고 기본 OFF다.**
`src/CudaBICGBackend.cu:365`(“ACTIVE-SLOT COMPACTION (RASBERY_GPU_CMFD_COMPACT, default OFF)”), 게이트 본체 `src/CudaBICGBackend.cu:533-537`. → 계획 §4.3 행 **확인**.

**F2. logical→physical slot map이 있다.**
매크로 `RASBERY_CMFD_SLOT(m)` — `src/CudaBICGBackend.cu:401-408`(인자 `const int* __restrict__ slot_map, const int lanes`). pinned host buffer + 안정 device pointer는 `src/CudaBICGBackend.cu:2905-2914`에서 할당되고(`h_slot_map[i] = i % S`가 OFF의 항등 초기화), `:3239`에서 내용만 async H2D 되며, `:2973-2976`에서 해제된다. **계획 §WP3 “구현 세부”의 “graph 인자 주소는 바뀌지 않고 내용만 갱신한다”는 이미 그렇게 되어 있다.** → **확인**.

**F3. bucket graph가 있다.**
`cmfdBucketIndex()` `src/CudaBICGBackend.cu:423-428`, `g_cmfd_bucket_graphs` `:545`, 9칸 히스토그램 `:546-551`, bucket 변경 시 graph 인스턴스 무효화 논리 `:3609-3616`. → **확인**.

**F4. `[RASBERY][CMFD][COMPACT]` receipt가 있고 항상 인쇄된다.**
`src/CudaBICGBackend.cu:5620-5645`. 필드: `enabled`, `logical_drives`, `physical_slot_blocks`, `padding_blocks`, `padding_fraction`, `bucket_graphs`, `bucket_histogram[9]`. 주석이 명시하듯 **ON/OFF 무관하게 인쇄**되므로 “상금의 크기”를 먼저 볼 수 있다. → **확인**.

**F5. 계약 테스트는 이미 있고, 계획이 부른 이름과 다르다.**
계획 §WP3 변경 파일은 `tools/test_cmfd_compaction_contract.py`(수정)와 `test/cmfd_compaction_runtime.cpp`(생성)를 적었지만, 트리에는 **`tools/test_cmfd_slot_compaction_contract.py`**와 **`test/cmfd_slot_compaction_replay.cu`**가 이미 있다. 그리고 그 테스트는 계획 §WP3 “테스트 우선 절차” 1번을 **이미 수행한다**: 모든 `__global__` 커널이 `RASBERY_CMFD_SLOT`로 슬롯을 해석하는지(`tools/test_cmfd_slot_compaction_contract.py:145-146`), 가드가 첫 문장이며 모든 `__syncthreads`/shared 쓰기보다 앞서는지(`:157`), 네 개의 per-slot mask가 매핑된 슬롯으로 인덱싱되는지, `gridDim.x`가 손대지지 않았는지, OFF가 완전 항등인지, bucket ladder가 스케줄러의 그것과 같은지, receipt 필드가 있는지(`:247-248`) — 테스트 docstring 1–7번.

> **WP3 재분류(권고):** WP3은 “검증·하드닝”이 아니라 **“238에서 가격 평가”**가 남은 유일한 작업이다. 남은 코드 작업은 계획 §WP3 “테스트 우선 절차” 2–4번(비연속 physical slot `{1,4,7,31}` 강제 runtime 테스트, bucket 경계 전수, tenancy 교체 후 stale map) 뿐이고, 그 셋은 `test/cmfd_slot_compaction_replay.cu`를 확장하면 된다.

---

## 2. §4.3 행별 검증

| 계획 §4.3의 주장 | 판정 | 근거 |
|---|---|---|
| XSLIB host cache 기본 ON, single-flight | **확인** | `src/XSSet.cpp:877-882` — `RASBERY_XSLIB_CACHE`가 정확히 `"0"`일 때만 비활성. 캐시 키는 `(path, file_size, mtime, ng)` (`src/XSSet.cpp:~862-874`, `AcquireXsLibrary` `:887-897`). WP8이 지적한 약점 그대로다. |
| writer thread / light output 구현·측정 | **확인** | `src/IoWriter.h`, `include/chiffon/BatchLightResult.h` |
| host-free outer 구현·채택 | **확인** | `src/CudaOuterGraph.cu:1689-1721` host-free 사다리와 `hostfree_refusals` receipt |
| conditional WHILE 구현, 기본 OFF | **확인** | `src/GpuOuterWhile.h:89-100` — “RASBERY_GPU_OUTER_GRAPH -- DEFAULT OFF, and that is a gate decision” |
| GPU PPR 구현, N1, 기본 OFF, fail-open | **확인** | 기본 OFF `src/CudaPprBackend.cu:799-801`; fail-open `src/CudaPprBackend.cu:725-733`(`[RASBERY][PPR_GPU][WARN] … falling back to host PPR`); host seam(WP1 이전) `src/Driver.h`의 `if (!ppr_on_device)` |
| K-process/GPU + MPS launcher 구현 | **확인** | `tools/run_multi_gpu_batch.py` (단, §7 참조 — 지금 다른 작업자가 수정 중) |
| CMFD active-slot compaction 소스에 있고 기본 OFF | **확인** | F1–F5 |
| immediate refill 구현 | **확인** | `src/BatchRefill.h`, `src/main.cpp`의 `refill::ledger()` |
| persistent evaluator = 문서만 | **확인** | `src/EvaluatorContext.h`만 존재 |
| FlatXS cooperative kernel 없음 | **확인** | `src/FlatXsKernel.h`에 CTA 협업 본문 없음 |
| PPR device convergence/reconstruct 없음 | **확인** | `src/CudaPprBackend.cu`는 매 반복 corner partial D2H + `cudaStreamSynchronize` |
| result-mode / fidelity 분리 불완전 | **확인 (WP1이 해결)** | §3 참조 |
| **GPU full fail-closed 없음** | **부분 반박** | 아래 F6 |
| **(행 없음) GPU CRAM** | **누락** | 아래 F7 |

**F6. “GPU full fail-closed 없음”은 한 곳에서 틀렸다.**
CMFD/BiCGSTAB 백엔드는 **stand-up에서 이미 fail-closed**다: `src/BICGSolver.cpp:86-90`가 `RASBERY_GPU`가 켜져 있는데 백엔드가 없으면 `throw std::runtime_error("RASBERY_GPU requested but unavailable: " + status)`. 계획의 표는 “없음”이라고 적었지만, 정확히는 **“stand-up은 닫혀 있고, per-drive는 열려 있었다”**이다. per-drive 쪽 — `src/BICGCMFD.cpp`의 `drive()`가 `driveDeviceSweeps()` 실패 시 순정 host BiCGSTAB 루프로 떨어지는 자리 — 는 실제로 fail-open이었고 **아무 카운터도 없었다**. WP1이 그 자리를 닫았다(§3).

**F7. §4.3에 GPU CRAM 행이 없다 — 고정 SHA 이후에 착지했기 때문이다.**
`8bd5112`(“feat(Task 16): CRAM Bateman depletion on the device, behind RASBERY_GPU_CRAM”)는 계획의 분석 SHA `8b8f18e` 다음 커밋이다. 이것은 **계획 WP9 단계 B가 조기 인도된 것**이며, WP9의 “재프로파일에서 10% 이상인 항목만 구현한다”는 원칙은 CRAM에 대해서는 이미 무의미하다 — 코드는 있고, **가격만 없다**.
- 기본 OFF: `src/CudaCramBackend.cu:777-780`
- fail-open: `src/CudaCramBackend.cu:647-656`(`[RASBERY][CRAM_GPU][WARN] … falling back to host depletion`), host seam `src/XSSet.cpp`의 `++_cram_host_fallbacks` 두 곳
- **등급은 N1이 아니라 “궤적을 움직인다”**: `src/Driver.h:396-405`이 `RASBERY_GPU_CRAM`을 `kArmEnv`에 **일부러 넣은** 이유를 적어 놓았다 — 장치 depletion이 쓴 동위원소 재고를 다음 상태점이 읽으므로 keff·임계붕소·AO가 첫 상태점 이후 전부 움직인다. PPR(`kArmEnv`에서 일부러 제외)과 정확히 대비되는 반례다.

> **권고:** §4.3에 CRAM 행을 추가하고, WP2 채택 매트릭스에 `RASBERY_GPU_CRAM=1/0`을 **6번째 대상 기능**으로 넣는다. WP9 단계 B는 “구현”에서 “가격 평가 + N1 게이트”로 재분류한다.

---

## 3. WP1에서 실제로 발견한 것 (구현과 함께 종결)

**F8. `light` 출력이 screening으로 분류되던 자리와, 그 대가가 양방향이었다는 것.**
WP1 이전 `src/main.cpp`는 `const bool screening = (ga_feedback_passes > 0) || light_result;`였다. 계획 §6.2·§10 1행이 지적한 그대로다. 그런데 대가는 **두 방향**으로 지불되고 있었다.
1. strict/light 실행이 acceptance 감사에서 `full_hdf5:false`로 무효화된다 — `tools/run_single_gpu_batch.py:42-47`의 `EXACT_PHYSICS_MODE`.
2. 그리고 하네스는 그 문제를 **회피**하려고 light 청크에 대해 *screening receipt를 기대*한다 — `tools/run_single_gpu_batch.py:59-69`의 `SCREENING_PHYSICS_MODE` / `expected_physics_mode()`, 호출부 `:408`. 즉 **screening이 아닌 실행이 낸 “screening이다”라는 receipt가 통과**하고 있었다. exact-only 계약이 막으려던 바로 그 실패를 반대편에서 만들어낸 셈이다.

또한 같은 트리 안에서 **두 receipt가 이미 서로 모순**이었다: JSONL 쪽 `include/chiffon/BatchLightResult.h`의 `Write()`는 feedback pass가 0이면 `"physics_mode":"exact"`를 적는데, stdout 쪽은 같은 실행을 `ga_screen_feedback_limited`라고 적었다.

**F9. 세 개의 receipt 필드가 인쇄되지만 한 번도 증가하지 않는다.**
`BackendCounters::xs_cpu_fallbacks`(선언 `src/CudaBICGBackend.h:53`, 인쇄 `src/BICGSolver.cpp:108`), `cmfd_cpu_fallbacks`(`:55` / `:110`), `nodal_cpu_fallbacks`(`:84` / `:131`) — **소스 어디에도 `++`가 없다.** 늘 0을 보고한다. WP2가 “기능 flag가 켜짐이 아니라 engagement receipt가 0보다 큰 실행만 유효하다”를 이 필드들로 판정하려 했다면, 그 판정은 상수를 읽고 있었다. WP1의 `[RASBERY][GPU_FULL]`이 cmfd/nodal/flatxs/xe/ppr/cram/outer를 실제로 센다. ~~**남은 작업: 세 죽은 필드를 배선하거나 삭제한다(WP2).**~~

> **종결 (`666e123`).** 배선했다. 삭제가 아니라 **하나의 진실을 읽게** 했다 — 세 필드는 WP1 가드가 이미 세고 있는 seam tally(`gpufull::fallbacks`)에서 채워진다. 자리:
> `src/BICGSolver.cpp:129-133`(비-arena 판)과 `src/CudaBICGBackend.cu:5671-5675`(arena 판, 그리고 `:5678-5680`에 세 필드를 **처음으로** 인쇄). 두 판 모두 고친 이유: `BICGSolver::~BICGSolver`는 arena 실행에서 조기 반환하므로(`src/BICGSolver.cpp:95-101`) **M64는 비-arena 판을 한 줄도 인쇄하지 않는다.** 한 판만 고치면 캠페인이 실제로 재는 팔에서 보이지 않는다.
> 매핑: `cmfd_cpu_fallbacks`←`Subsystem::Cmfd`, `nodal_cpu_fallbacks`←`Nodal`, `xs_cpu_fallbacks`←`FlatXs+Xe+Cram`(XSSet의 세 드라이브).
> **두 번째 카운터를 seam에 심지 않은 것은 결정이다** — 한 사건을 두 tally가 세면 어긋날 수 있고, 그 어긋남은 두 receipt 어디에도 보이지 않는다(F13이 적은 규칙).
> 계약: `tools/test_receipt_counters_live.py`. `BackendCounters`의 모든 `*fallbacks*` 필드가 `src/` 안에 write site를 갖는지 전수 스캔하고, 음성 대조군 셋(없는 필드 이름은 0건, `gf::fallbacks(` 줄을 지우면 세 필드가 다시 죽어야 함)을 붙인다. 아직 죽은 네 필드(`xs_gpu_calls`, `nodal_gpu_calls`, `th_gpu_calls`, `depletion_gpu_calls` — 전부 “engage 했다”의 양성 쪽)는 테스트의 `KNOWN_DEAD`에 **이유와 함께 열거**했다: WP2가 배선하면 목록이 눈에 보이게 줄어든다.

**F10. 융합 Xe 팔(`RASBERY_GPU_XSRECON`)의 fallback은 카운터가 전혀 없었다.**
분할 팔은 `src/XSSet.cpp`의 `tally.host_fallbacks`로 세지만, 융합 팔은 “any failure falls through to the unchanged CPU loop” 주석만 있고 아무것도 세지 않았다. 즉 `[RASBERY][XE_GPU]`의 `device_updates=0`을 보고도 “팔이 꺼졌나 거절했나”를 구분할 수 없었다. WP1이 두 자리 모두 가드했다.

> **종결 (`666e123`).** 융합 팔에 **자기 삼중항**을 줬다: `fused_updates` / `fused_device_updates` / `fused_host_fallbacks` — 선언 `src/XeGpuReceipt.h:36-55`, 인쇄 `:88-93`, 청구 `src/XSSet.cpp:4136-4149`.
> **`host_fallbacks`에 합치지 않은 것이 요점이다.** `XSSet::UpdateEquilibriumXenon`은 분할 팔이 거절하면 **이어서** 융합 팔을 시도한다(`src/XSSet.cpp:4116-4149`). 합쳤다면 Xe 스텝 하나에 두 번 청구되어 분할 팔의 회계 항등식 `xe_updates == device_updates + host_fallbacks`가 깨진다. 삼중항을 따로 두면 **두 팔 각각이 자기 receipt 안에서 항등식**으로 남는다.
> `fused_updates`는 시도 **전에** 청구한다(분할 팔이 `xe_updates`를 그렇게 하는 것과 같은 이유). 계약 테스트가 소스 줄 순서로 그것을 붙든다.

**F11. outer segment의 두 arm 실패는 이름조차 없다.**
`src/Driver.h`의 `armOuterSegment`에서 `if (!view.valid) return false;`(`src/Driver.h:1570`)와 `if (!gpu::rasberyBindOuterResidency(residency)) return false;`(`src/Driver.h:1738`)는 `noteOuterSegmentRefusal`을 부르지 않고 반환한다. 따라서 `[OUTER_GPU].refusals`에 사유가 남지 않는다. WP1의 pre-arm 가드가 GPU full에서 이를 치명적으로 만들지만, ~~**사유는 여전히 무명**이다.~~

> **종결 (`666e123`).** 둘 다 이름을 얻었다. `view.valid == false` → `OuterSegmentRefusal::NoResidency`(`src/Driver.h:1570-1580`), `rasberyBindOuterResidency` 실패 → `OuterSegmentRefusal::Unbound`(`src/Driver.h:1756-1765`). enum은 손대지 않았다 — 두 값 모두 이미 있었고 뜻이 정확히 맞는다(`src/CudaOuterGraph.h:489`, `:483`).
> **왜 이 둘이 살아남았는지가 계약 테스트의 설계를 정했다.** 둘 다 *자기 이름을 부르는* 다른 거절(`no_arena`, `sweep_not_resident`) 바로 몇 줄 아래에 있다. 함수를 훑는 사람 눈에는 note가 사방에 보이고, 고정 창(window) 스캔도 **이웃의 note로 만족**한다. 그래서 `tools/test_receipt_counters_live.py`는 창이 아니라 **직전 `return`까지**를 구간으로 잡는다. 음성 대조군: note 줄을 전부 지우면 feature-off를 뺀 세 개 return이 전부 보고되어야 한다.
> 주석 안의 “~~return false~~” 인용을 세지 않도록 스캐너는 줄 수를 보존한 채 `//` 주석을 지운 뒤 읽는다.

→ **WP7 단계 A의 “필수 receipt” 목록에 이 둘이 들어간다(이제 실제로 인쇄된다).**

**F12. L3coarse는 바이너리가 탐지할 수 없다.**
계획 §6.2는 fidelity를 “실제 env/deck에서 도출”하라고 하지만, `tools/make_screening_deck.py`는 **덱의 연소도 격자만 다시 쓴다**(`:418-419`, `COARSE_BURNUPS`) — 환경변수를 세우지 않는다(`:109-111`에 적힌 env 블록은 A2 knob이지 coarse가 아니다). 그리고 `[PHYSICS_MODE]` receipt는 어떤 덱도 파싱하기 전에 나온다. → WP1은 `RASBERY_PHYSICS_FIDELITY` **선언 채널**을 두되, **선언은 오직 더 거칠게만 만들 수 있다**는 규칙(rank max)으로 닫았다. 덱에서 도출하는 per-case 검사는 WP0의 manifest parser에 속한다.

**F13. §6.3 receipt의 두 필드는 여기서 낼 수 없다.**
계획 §6.3 예시 receipt의 `graph_fallbacks`는 이미 **세 곳**이 같은 키 이름으로 내고 있고(`[CUDA][BACKEND_COUNTERS]`, `[NODAL][GPU]`, `[NODAL][BATCH]`), `tools/run_single_gpu_batch.py:38`의 `GRAPH_FALLBACK_COUNTER` 정규식이 이미 셋 다 훑는다. `[GPU_FULL]`에서 한 번 더 내면 하나의 사실에 숫자가 둘이 된다. `mid_iteration_materializations`는 WP5의 XS ownership이 들어오기 전까지 생산자가 없다. → 둘 다 `[GPU_FULL]`에서 **의도적으로 제외**하고 그 이유를 `src/GpuFullContract.h`에 적었다.

**F14. §6.3 4항(실패 케이스의 임시 출력 삭제)은 WP1에 넣지 않았다.**
결과 HDF5는 `IoWriter`가 쓰고, main.cpp이 만들지 않은 경로를 지우는 것은 위험하다. → 잔여 작업(WP1.1): `IoWriter`가 `<out>.partial`에 쓰고 성공 시 atomic rename. 계획 §10의 “출력 파일 충돌 → canonical path uniqueness, temp+atomic rename”이 어차피 같은 것을 요구한다.

---

## 4. 계획과의 이견·위험 (구체적으로)

**R1. §3.2/§3.3의 모든 M64 숫자는 결함 있는 하네스에서 측정되었다. — 가장 큰 위험.**
dispatcher가 Driver lane을 호스트 코어 수(24)로 제한했고, 바이너리 자신의 기준 실행은 24코어에서 64 lane을 돌린다. 같은 바이너리에서 **115.6 c/h 대 582 c/h, 5.0×**. 근거: `tools/run_single_gpu_batch.py:15`, `:87-88`, `:206`, `:523-527` (커밋 `7099e54` "fix(harness): make both batch harnesses reproduce the raw 238 reference line"에 포함). 문서 자신이 “세 번의 이전 sweep이 이것을 데이터로 보고했다”고 적고 있다.
→ **결과:** §3.3의 `FlatXS 39.9% / Xe 26.9% / CMFD 19% / Nodal 8.6%`는 **도착 폭이 14.5로 굶던 실행의 커널 지분**이다. 폭이 채워지면 지분은 반드시 이동한다. **WP5가 “FlatXS 최우선”을 확정하기 전에 §3.3을 고친 하네스에서 다시 재야 한다.** 이것을 WP5의 blocking dependency로 추적표에 넣었다.

> **R1 갱신 (2026-09-01).** 하네스 쪽은 끝났다: 238에서 dispatcher control 1×M64가
> **578 c/h**로 원시 라인 **582**를 −0.7%로 재현했다(docs/W4_L5 §4.8). 5.0배 결함은
> 재발하지 않는다. 그러나 §3.3의 지분은 **아직 다시 재지 않았다** — 그리고 이제
> 재야 할 실행이 하나 더 늘었다: `width_fill`이 0.28에서 **0.41**로 오른
> 8×M8+MPS arm이다. 폭이 두 배 가까이 찬 실행에서 FlatXS 지분이 그대로인지가
> WP5의 전제이고, 그 측정 없이 “FlatXS 39.9%”를 인용하는 것은 여전히 굶던 실행을
> 인용하는 것이다.

**R2. WP3 bucket ladder: 계획 7칸, 트리 9칸.**
계획 §WP3 구현 세부는 “bucket은 `1,2,4,8,16,32,64`로 제한”이라 적었다. 트리는 **`1,2,4,8,16,24,32,48,64` 9칸**이다 — `src/CudaBICGBackend.cu:424`, 히스토그램 배열 크기 9 `:547`, receipt의 `bucket_histogram` 길이 9 `:5638-5644`. 그리고 계약 테스트는 “bucket ladder는 스케줄러의 사다리이지 두 번째 사다리가 아니다”를 명시적으로 붙들고 있다(테스트 docstring 6번).
**R2 종결 — 트리의 9칸을 채택하고, 아무 파서도 칸 수를 세지 않는다 (`79c1880`).**

선택지 (a)를 택한다: **소스가 진실이다.** 근거는 두 가지다.

1. 24와 48은 238 M64의 실제 도착 폭 14.5 부근을 덮는 rung이다. 계획의 7칸으로 줄이면
   14.5는 16 대신 16, 20은 32로 올라가 **WP3이 지금 가격을 매기려는 padding이 늘어난다.**
   사다리를 짧게 만드는 것은 receipt를 짧게 만들지 padding을 줄이지 않는다.
2. 사다리는 네 곳에 있고(`src/GpuPhaseScheduler.h:69` `kDispatchBuckets`,
   `src/CudaBICGBackend.cu:418` `kBuckets`, `src/CudaXsReconBackend.cu:1310` `kBuckets`,
   `test/cmfd_slot_compaction_replay.cu:127`) 히스토그램 배열 두 개가 그 **길이**에
   묶여 있다(`CudaBICGBackend.cu:548`, `CudaXsReconBackend.cu:359`). 스케줄러 쪽을 7로
   바꾸는 것은 커널 코드 변경이고, WP3의 지시는 “커널 코드를 바꾸지 말 것”이다.

**계약으로 고정한 방식이 요점이다 — 9를 적지 않았다.**
`tools/test_cmfd_slot_compaction_contract.py`의 `scan_ladder()`는 스케줄러의 리스트를
읽어 나머지 세 복사본과 비교하고, `kDispatchBucketCount`와 두 히스토그램 배열 크기를 그
**길이**와 비교한다. 9를 상수로 적는 것은 계획이 7을 적은 것과 같은 실수이며,
`cmfdBucketIndex()`가 `len-1`을 반환할 수 있으므로 짧은 배열은 **범위 밖 atomic 증가**다.
음성 대조군 두 개가 붙어 있다: `.cu`의 사다리를 계획의 7칸으로 줄이면 FAIL,
히스토그램만 7로 줄여도 FAIL.

**파서는 관대하되 침묵하지 않는다.** `tools/cmfd_compact_receipt.py`가 사다리를 소스에서
읽어 receipt의 히스토그램과 zip한다. 길이가 다르면 **mismatch로 보고하고 원본 그대로
돌려준다** — 잘라 맞추지도, 0으로 채워 맞추지도 않는다. 조용히 맞추는 것이 파서와
receipt가 한 캠페인 내내 어긋나는 방식이다. `padding_fraction`은 블록 수에서 **다시
계산**한다(세 숫자가 한 사실이므로 불일치는 반올림이 아니라 발견이다).
`python3 tools/cmfd_compact_receipt.py <log>`가 WP3 가격표를 인쇄한다.


**R3. WP3의 등급이 문서 간에 다르다.**
새 계획은 CMFD compaction을 **B0**로 적는다(§WP3 정확성 게이트 “B0가 목표다”, §9 표). GA evaluator 계획은 같은 것을 **N1**으로 적는다(`docs/GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md:28`, `:868`).
→ **권고:** B0를 *목표*로 두고, 238의 첫 ON×3 digest + `h5diff -r`를 판정자로 삼는다. digest가 움직이면 그 자리에서 N1로 기록하고 Gate B를 태운다. 두 문서 중 하나를 지금 고치는 것보다 측정이 싸다.

**R4. WP0/WP2/WP4가 모두 하네스 파일을 건드린다 — 지금 다른 작업자가 그 파일들을 수정 중이다.**
이 검토 도중 그 작업의 1차분이 `7099e54`로 착지했다(두 하네스 + `tools/test_multi_gpu_dispatch.py`). 아직 워킹트리에 남은 것: `?? tools/test_harness_env_parity.py`, `?? test/reference/batch_reference_env_238.json`.
→ **순서 강제:** 하네스 env-parity 작업이 커밋된 뒤에 WP0의 `run_exact_throughput_matrix.py`/`parse_perf_receipts.py`를 얹는다. WP4의 tuner는 그 위에.

**R5. WP1의 acceptance 감사 전환은 하네스 파일을 고쳐야 완결된다.** — **종결 (`5ccf879`).** 아래는 그때 적은 요구사항이고, 실제로 그렇게 했다. 종결 내역은 이 절 끝에 붙였다.
바이너리 쪽은 끝났다(`policy`/`acceptance_eligible` 필드 + `tools/exact_audit.py`). 그러나 감사 자체는 `tools/run_single_gpu_batch.py`에 있고 그 파일은 내 소유가 아니다. **필요한 변경은 정확히 다음 한 덩어리다:**

```python
# tools/run_single_gpu_batch.py — WP1: 감사를 출력 모드가 아니라 fidelity에 건다
from exact_audit import audit_physics_mode          # tools/ 를 sys.path 에 넣고

# EXACT_PHYSICS_MODE / SCREENING_PHYSICS_MODE / expected_physics_mode() 를 삭제하고
# check_run_receipts() 안의
#     problems = check_physics_mode(output, expected_physics_mode(plan.result_mode))
# 를
      problems = audit_physics_mode(output)          # result_mode 는 더 이상 인자가 아니다
# 로 바꾼다. LaunchPlan.result_mode 는 보고용으로만 남긴다.
```

`tools/run_multi_gpu_batch.py:921`, `:1057-1071`의 “light wave는 `RASBERY_ALLOW_SCREENING=1`이 필요하다” 거부도 함께 제거해야 한다 — WP1 이후 바이너리는 그것을 요구하지 않는다(무해하지만 이제 틀린 문장이다).

**R5 종결 내역 (`5ccf879`).**

| 자리 | 무엇을 했나 |
|---|---|
| `tools/run_single_gpu_batch.py:30-32` | `sys.path`에 `tools/`를 넣고 `from exact_audit import audit_physics_mode` |
| `tools/run_single_gpu_batch.py:38-52` | `EXACT_PHYSICS_MODE` / `SCREENING_PHYSICS_MODE` / `expected_physics_mode()` / `check_physics_mode()` **전부 삭제**. 그 자리에는 무엇이 왜 사라졌는지가 남아 있다 |
| `tools/run_single_gpu_batch.py:351-355` | `check_run_receipts()`가 `audit_physics_mode(output)` — **`result_mode`는 더 이상 인자가 아니다** |
| `tools/run_single_gpu_batch.py:217-222` | `LaunchPlan.result_mode`는 **보고 전용**이라고 명시 |
| `tools/run_multi_gpu_batch.py:1058-1068` | light wave 사전 거부 **삭제**. 바이너리가 더는 거절하지 않으며, 실행 파일보다 엄한 가드는 **통과했을 실행을 거절한다** |
| `tools/run_single_gpu_batch.py:517-518`, `tools/run_multi_gpu_batch.py:918-922` | `--result` 도움말에서 “light는 `RASBERY_ALLOW_SCREENING=1`이 추가로 필요” 삭제 |

`RASBERY_ALLOW_SCREENING`의 뜻은 **하나로 유지**한다 — “strict가 아닌 fidelity를 허용한다”. 그리고 여전히 **운영자의 것**이다: `tools/test_harness_env_parity.py:134-140`이 두 하네스 중 어느 쪽도 스스로에게 그 권한을 주지 못하게 붙든다.

계약 테스트(음성 대조군 포함):

- `tools/test_multi_gpu_dispatch.py:435-497` — 삭제된 네 이름이 **돌아오면 FAIL**(`hasattr` 음성 대조군), `check_run_receipts`가 `audit_physics_mode(output)`를 인자 없이 부르는지, strict/light가 **통과**하고 coarse·feedback-limited·pre-WP1 receipt가 **거절**되는지
- `tools/test_multi_gpu_dispatch.py:539-567` — light wave가 권한 **없이/있게/거짓값으로** 모두 rc=0, 그리고 거부 문구가 dispatcher에 **남아 있지 않은지**
- `tools/test_exact_only_contract.py:71-119` — 감사가 fidelity로 옮겨간 뒤에도 exact-only 계약이 성립하는지(“scalar 출력이라는 이유로 strict 실행을 무효화하면 FAIL”이 추가되었다)

`tools/test_*.py` 실패 집합은 `9dff6ff`와 동일한 **사전 실패 9개**.

**R6. WP1 §변경 파일과 실제 구현의 의도적 차이.**
계획은 `RunContract.{h,cpp}`, `GpuFullContract.{h,cpp}`, `CMakeLists.txt`, 그리고 `CudaBICGBackend/{XsRecon,Ppr}Backend.{h,cu}` 수정을 적었다. 실제로는 **헤더 온리**, CMake 무변경, `.cu` 무변경으로 구현했다.
- **이유 1 (링크):** `CMakeLists.txt:34-35`는 RASBERY 타깃에 대해 `src/*.cpp`를 glob 하지만, **15개 남짓의 테스트 실행 파일은 개별 소스를 나열해 링크한다**(예: `CMakeLists.txt:197-201`이 `CudaXsReconBackend.cu` + `XeAndersonReference.cpp` + `XeFormMiner.cpp`). 카운터를 `.cpp`에 두면 그 `add_executable` 전부를 고쳐야 하고, **이 기계에서는 그 어느 것도 빌드로 검증할 수 없다.** 인라인 함수 안의 함수 지역 static은 표준이 프로그램당 한 인스턴스를 보장하므로 링크 목록을 건드리지 않는다.
- **이유 2 (자리):** 가드는 **host 쪽 seam**, 즉 CPU 본문이 실제로 시작되는 지점에 둔다. `.cu` 안의 수백 개 `return false`는 전부 그 seam 중 하나로 올라온다(부록 A). seam에 두면 하나의 스캔 가능한 목록이 되고, `.cu`를 빌드 없이 수술하지 않아도 된다.

**R7. `[RASBERY][PHYSICS_MODE]`의 `physics_mode` 값은 일부러 옛 어휘를 유지했다.**
strict와 A2 **둘 다** 지금까지 `full_exact_nodal`을 찍어 왔고 238 하네스가 그 문자열을 감사한다. 이 필드를 새 정책 단어로 돌리면 **디스크의 모든 A2 매니페스트가 무효**가 된다. 그래서 fidelity는 새 필드(`policy`, `physics_fidelity`)에만 있고, `physics_mode`에 새로 생긴 값은 `ga_screen_coarse10` 하나뿐이다(WP1 이전 바이너리는 coarse 덱을 보고할 수 없었으므로 저장된 arm이 그 값을 가질 수 없다).

**R8. 계획 §11-2 “배치에서 outer segment budget 8을 재도입하지 않는다”는 트리 상태와 일치한다.** 별도 조치 불필요. (메모리의 “budget 8 in batch는 폭 붕괴로 폐기” 기록과 같다.)

---

## 5. WP 추적표

상태 어휘: **done**(코드+게이트 완료) / **landed-unpriced**(코드 있음, 238 가격 없음) / **in-progress** / **not-started**.

| WP | 범위 | 상태 | 소유 | 다음 게이트 | 차단 의존성 |
|---|---|---|---|---|---|
| **WP0** | 재현 가능한 baseline + 자동 판정 하네스 | **not-started** | 하네스 담당 | `tools/test_perf_manifest_contract.py`가 mode/fidelity 불일치 arm 비교를 거부 | R4: `run_{single,multi}_gpu_batch.py` env-parity 작업 커밋 |
| **WP1** | 출력/충실도 분리 + GPU full fail-closed | **done (코드·계약·하네스)** — 바이너리 `9dff6ff`, 하네스 감사 전환 `5ccf879`, 죽은 receipt 필드 `666e123` | 본 작업 | 238에서 §8 runbook **7줄** 통과 + 오버헤드 <1% | ~~R5~~ 종결. **238 빌드만 남았다** |
| **WP2** | 착지 기능 238 재가격 | **not-started** (단, 판정 근거는 준비됨 — F9/F10 종결로 engagement receipt가 상수를 읽지 않는다) | 하네스 담당 + 본 작업 | 6개 기능 각각 `adopt`/`keep-opt-in`/`reject`. **판정은 `[GPU_FULL].*_fallbacks` + `[BACKEND_COUNTERS].{xs,cmfd,nodal}_cpu_fallbacks` + `[XE_GPU].fused_*`로 한다** | WP0, WP1(모드 일치 비교가 성립해야 함), R4. **잔여 F9: 네 개의 `*_gpu_calls`(`xs`,`nodal`,`th`,`depletion`)는 아직 죽어 있다** — `tools/test_receipt_counters_live.py`의 `KNOWN_DEAD`에 이유와 함께 열거되어 있고, 배선하면 목록이 줄어든다 |
| **WP3** | CMFD compaction 검증·채택 | **landed-unpriced, 계약 하드닝 done** (`79c1880`: 커널 12개의 outer 계열까지 스캔 확대 + 15개 음성 대조군, R2 종결) | 본 작업 | **238 가격 평가만 남았다 — §9 runbook**. M64 mode-matched +5%, padding 유의 감소, ON×2 digest 동일 | WP2, ~~R2~~ 종결, R4. **코드 작업 없음** |
| **WP4** | 단일 GPU K-process 자동 튜닝 | **priced + tuner landed** (`6ad1de7`) — 238 8-arm 행렬 실측(docs/W4_L5 §4.8), `--procs-per-gpu auto`/`--tune-from` 착지 | 하네스 담당 + 본 작업 | ~~K=2가 1×64 대비 ≥1.05×~~ **통과: 2×M32 = 1.121× (MPS 없음)**. 최고 arm 8×M8+MPS = **878 c/h = 1.519×**. 다음: 12×M6/16×M4로 무릎 확정, 3연속 wave 회귀 없음 | ~~R1~~ 해소(§4.7 결함 재발 없음: control 578 vs 원시 582), R4 |
| **WP5** | FlatXS CTA-per-node + XS residency | **not-started** | 본 작업 | 단계 A(ptxas/ncu spill 증명) → 분기 | **R1: 재측정은 이제 가능해졌다** — dispatcher control이 원시 라인을 ±5%로 재현했다(578 vs 582). 남은 것은 §3.3 커널 지분을 **폭이 찬 실행에서 다시 재는 것**이고, 그 실행은 8×M8+MPS다(width_fill 0.28→0.41). WP3+WP4 |
| **WP6** | GPU PPR device loop/reconstruct/canonical | **부분 landed-unpriced** (`c502856` 장치 PPR, 기본 OFF·fail-open) | 본 작업 | 단계 A: `ppr_device_calls == statepoints`, `ppr_host_calls == 0` (GPU full), N1 Gate A/B | WP2(PPR 1차 A/B), WP1(fail-closed로 engagement 증명) |
| **WP7** | 단일 launch/sync 축소 + Xe transaction | 단계 A **부분 landed-unpriced** (WHILE 구현·기본 OFF) / **단계 B 코드·계약 done, 미가격** — census + `RASBERY_GPU_CMFD_FUSE` 4비트(기본 0), `[RASBERY][CMFD][GRAPH]` receipt, `tools/test_cmfd_fuse_contract.py`, `docs/WP7_CMFD_GRAPH_CENSUS_20260831_KO.md` / 단계 C **not-started** | 본 작업 | 단계 A: 단일 WHILE ≥5% (교대 5쌍). 필수 receipt에 `refusals{no_residency}` / `refusals{unbound}` 포함. **단계 B: 238에서 mask별 h5diff 0/644 + digest `0d15abf29d222a02`/`4382` 동일(B0)이 먼저, 그 다음 단일 wall ≥3%(16.9 s 기준) — 미달이면 기본값 유지.** census 실측: outer 90 node, sweep 98 node/sweep → `FUSE=7`에서 78 / 85(−13.3%) | WP2. ~~F11~~ 종결(`666e123`). **단계 B는 238 컴파일이 첫 관문**(로컬에 nvcc 없음) |
| **WP8** | 장수명 GA evaluator | **not-started** | 본 작업 | 단계 1: `--evaluator-jsonl` wave 프로토콜 + A-B-A 동일성 | WP1+WP2 안정화 |
| **WP9** | 잔여 상태점 바닥 (CRAM/TH/search) | **B는 landed-unpriced** (`8bd5112` GPU CRAM), TH/search **not-started** | 본 작업 | CRAM: N1 Gate A/B + `[CRAM_GPU].host_fallbacks == 0` (GPU full) | WP2에 `RASBERY_GPU_CRAM` arm 추가 (F7) |
| **WP10** | GA 중복 캐시·warm-start·multi-fidelity | **not-started** | GA 담당 | `CandidateKey` 계약 테스트 + 실제 hit rate 측정 | WP8 |
| **WP11** | soak / v4 freeze / 기본값 승격 | **not-started** | 전원 | 10,000 case soak에서 §11 receipt 전부 0 | 전 항목 |

### 5.1 상태 정의에 대한 주의

“landed-unpriced”는 **계획이 그렇게 부르라고 한 상태**다(§4.3 제목). 이 표에서 그것은 **“새로 구현하지 말 것”**을 뜻한다 — WP3과 WP9-B가 특히 그렇다.

---

## 6. WP1 구현 요약 (`file:line`)

### 6.1 새 파일

| 파일 | 내용 |
|---|---|
| `src/RunContract.h` | `PhysicsFidelity{FullExact,StagedA2,Coarse10State,FeedbackLimited}`, 단일 진실 표 `kFidelityTraits[4]`(`:80-90` 부근), A2 탐지(Driver.h와 **같은 읽기**), `RASBERY_PHYSICS_FIDELITY` 선언 채널(더 거칠게만) |
| `src/GpuFullContract.h` | `Subsystem{Cmfd,Outer,Nodal,FlatXs,Xe,Ppr,Cram}`, `Violation : std::runtime_error`(site를 이름 짓는다), `required()`(**기본 OFF**, `RASBERY_GPU_FULL` / 별칭 `RASBERY_GPU_STRICT`, 한 번만 읽음), `note()`/`noteIf()`/`count()`, `appendReceiptFields()`, 매크로 `RASBERY_GPU_FULL_GUARD(_IF)` / `RASBERY_GPU_FULL_COUNT` |
| `tools/exact_audit.py` | fidelity에 건 acceptance 감사. `audit_physics_mode(output, require={"strict"})`. `result_mode`는 읽되 **절대 판정하지 않는다** |
| `tools/test_result_fidelity_contract.py` | WP1(a)/(c) 계약. `kFidelityTraits` 표 파싱 + 음성 대조군, Driver.h와 RunContract.h의 A2 술어 고정, main.cpp의 `screening`이 fidelity에서 나오는지, receipt 10개 필드, 감사의 12개 음성 대조군 |
| `tools/test_gpu_full_fail_closed.py` | WP1(b) 계약. 13개 fallback seam 전수 스캔 + seam마다 음성 대조군, count-only 예외 allowlist, per-case catch 두 개, receipt 양쪽 분기, `kArmEnv` 부재 |

### 6.2 가드를 심은 자리 (13곳)

| seam | 위치 | 이전 상태 |
|---|---|---|
| CMFD per-drive | `src/BICGCMFD.cpp:876` | 카운터 없음 (F6) |
| Nodal | `src/Nodal.cpp:857` | 카운터 없음 (F9) |
| FlatXS | `src/XSSet.cpp:3092` | 카운터 없음 |
| Xe 분할 | `src/XSSet.cpp:4131` | `tally.host_fallbacks`만 |
| Xe 융합 | `src/XSSet.cpp:4146` | **아무것도 없음** (F10) |
| CRAM predictor | `src/XSSet.cpp:4545` | `_cram_host_fallbacks` |
| CRAM corrector | `src/XSSet.cpp:4693` | `_cram_host_fallbacks` |
| outer pre-arm (ReconvergeFlux) | `src/Driver.h:1980` | `[OUTER_GPU].refusals` |
| outer 비유한 escape | `src/Driver.h:2042` | 없음 |
| outer 실패 (ReconvergeFlux) | `src/Driver.h:2062` | `refusals{launch_failed}` |
| outer pre-arm (SolveLoop) | `src/Driver.h:3117` | `[OUTER_GPU].refusals` |
| outer 실패 (SolveLoop) | `src/Driver.h:3237` | `refusals{launch_failed}` |
| PPR | `src/Driver.h:4222` | `ppr_host_statepoints` |
| **(count-only)** segment 안의 CMFD enqueue 거절 | `src/Driver.h:1229` | **아무것도 없음** — 살아 있는 segment 안이라 throw 불가 |

### 6.3 receipt 변경 (`src/main.cpp`)

- `[RASBERY][PHYSICS_MODE]`(`:644`)에 추가: `physics_fidelity`, `policy`, `acceptance_eligible`, `requires_exact_rerun`, `fidelity_declared`, `gpu_full`. 기존 필드는 값·의미 유지, 단 **light 실행의 `physics_mode`/`screening`이 바뀐다**(그것이 결함이었다).
- `screening` 정의: `:590` — `rasbery::fidelityIsScreening(fidelity)`.
- 알 수 없는 `RASBERY_PHYSICS_FIDELITY` 거부: `:577-588`.
- 새 `[RASBERY][GPU_FULL]` receipt: batch `:955`, serial `:1083` — **게이트 ON/OFF 무관하게** 인쇄.
- `--help`: light가 screening이라는 문장 삭제, `RASBERY_PHYSICS_FIDELITY`와 `RASBERY_GPU_FULL` 문서화(`:400-418`).

### 6.4 바이트 동일성

- HDF5 쓰기 경로는 손대지 않았다.
- `[TRAJECTORY]` digest는 상태점 데이터만 접는다(`src/Driver.h:~370-380`). `RASBERY_GPU_FULL`은 **의도적으로 `kArmEnv`에 넣지 않았다** — 완주한 실행은 게이트가 없을 때와 같은 경로를 탔기 때문이다(PPR에 대해 리스트가 이미 적어 둔 논리와 같고, CRAM에 대해 적어 둔 논리의 반대). 따라서 `env` 필드도 불변.
- 기본 OFF에서 가드는 **완화된 원자 증가 하나**이며, 그 자리는 이미 CPU 물리 본문 전체를 시작하려던 참이었다.
- stdout은 필드가 늘어난다.

### 6.5 로컬에서 실행한 검증

- `tools/test_*.py` 전량. 실패 집합이 `8bd5112` baseline(별도 worktree)과 **완전히 동일한 9개**(전부 사전 실패: C++ 컴파일러 부재 2건, anchor drift 7건). 신규 실패 0.
- 신규 두 계약: 구현 전 FAIL → 구현 후 PASS(테스트 우선 절차 이행).
- `src/main.cpp`, `src/Driver.h`, `src/Nodal.cpp`, `src/XSSet.cpp`, `src/BICGCMFD.cpp`의 중괄호/괄호 균형이 baseline과 **동일**.

---

## 7. 238 빌드로만 확인 가능한 것 (명시)

로컬에서는 컴파일하지 않았다(사용자 규칙: 로컬 계산 금지, 그리고 PATH에 C++ 컴파일러가 없다). **238에서 확인해야 하는 것:**

1. **컴파일 자체.** 특히 (a) `src/GpuFullContract.h`의 함수 지역 static `std::array<std::atomic<unsigned long long>, N>`가 `nvcc`가 컴파일하는 TU에서도 문제 없는지 — `Driver.h`가 이 헤더를 포함하고 `Driver.h`는 `main.cpp`(host)만 포함하지만, 향후 `.cu`에서 포함될 때를 위한 확인, (b) `RASBERY_GPU_FULL_GUARD_IF` 안에서 쉼표 없는 조건식만 쓰였는지(매크로 인자 파싱).
2. **`ctest --test-dir "$BLD" --output-on-failure`가 baseline과 같은 결과**인지. 특히 `test/` 아래 개별 소스를 링크하는 실행 파일들이 여전히 링크되는지(헤더 온리 선택의 근거가 맞는지).
3. **feature-off 바이트 동일성.** 게이트 없이 같은 덱을 돌려 `h5diff -r`가 비고, `[TRAJECTORY]` digest가 이전 바이너리와 같은지.
4. **fail-closed가 실제로 케이스 단위인지.** 4덱 배치에서 한 덱만 위반시켜 나머지 3덱이 완주하는지(계획 §WP1 테스트 우선 절차 3번).
5. **오버헤드 <1%** (계획 §WP1 성능 판정). 단일과 M64 각각, 게이트 OFF에서.
6. **`RASBERY_GPU_FULL=1`에서 현재 arm이 실제로 통과하는지.** 통과하지 못하면 그것이 이 WP의 첫 발견이다 — 어느 팔이 조용히 CPU였는지 receipt가 site로 말한다.

**F9/F10/F11(`666e123`)이 추가한 238 전용 확인:**

- **(7)** **`src/CudaBICGBackend.cu`가 이제 `GpuFullContract.h`를 포함한다.** §7-1(a)가 “향후 `.cu`에서 포함될 때를 위한 확인”이라고 적어 둔 그 일이 **지금 일어났다.** 함수 지역 static `std::array<std::atomic<unsigned long long>, N>`가 nvcc의 host pass에서 컴파일되고, 그 함수 지역 static이 host `.cpp` TU들과 **한 인스턴스로 링크되는지**가 이 커밋의 첫 컴파일 위험이다. (헤더에 `__device__` 코드는 없고 device 코드에서 호출되지도 않는다.)
- **(8)** **`[XE_GPU]` receipt에 필드 3개가 늘었다.** 저장된 매니페스트를 키 단위로 읽는 파서는 무해하지만, **필드 수를 세는** 파서가 있으면 거기서 깨진다. `tools/test_xe_gpu_contract.py`는 통과한다.
- **(9)** **카운터가 실제로 움직이는지.** 소스 스캔은 “쓰는 자리가 있다”까지만 증명한다. 값이 0에서 벗어나는 것은 §8-7)에서만 확인된다.

---

## 8. 238 runbook — WP1 acceptance (7단계)

```bash
export REPO=$HOME/gpu
export BLD=$HOME/build/rasbery-wp1-sm120
export DATA=$HOME/kngr_238
export OUT=$HOME/bench/rasbery-wp1
mkdir -p "$OUT"

# 0) 빌드 (계획 §WP0 기준 빌드와 같은 플래그)
cmake -S "$REPO" -B "$BLD" -DCMAKE_BUILD_TYPE=Release \
  -DRASBERY_ENABLE_CUDA=ON -DRASBERY_CUDA_ARCHITECTURES=120 \
  -DRASBERY_ENABLE_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$BLD" -j
ctest --test-dir "$BLD" --output-on-failure 2>&1 | tee "$OUT/ctest.log"

# 0b) 소스 계약 (하드웨어 불필요, 하지만 같은 트리에서)
python3 "$REPO/tools/test_result_fidelity_contract.py"
python3 "$REPO/tools/test_gpu_full_fail_closed.py"
python3 "$REPO/tools/test_result_mode_contract.py"
python3 "$REPO/tools/test_exact_only_contract.py"

# 1) ACCEPTANCE 1 -- light + strict 가 더 이상 screening 이 아니다
#    기대: policy=strict, screening=false, acceptance_eligible=true,
#          result_mode=light, full_hdf5=false, 그리고 RASBERY_ALLOW_SCREENING 없이 시작
env -u RASBERY_ALLOW_SCREENING -u RASBERY_STAGED_FLUX_TOL -u RASBERY_STAGED_XE_TOL \
    -u RASBERY_GA_FEEDBACK_PASSES CUDA_VISIBLE_DEVICES=0 \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238.json" --raso "$OUT/light_strict.h5" --result light \
  2>&1 | tee "$OUT/a1.log" | grep -E '\[PHYSICS_MODE\]|\[EXACT_ONLY\]'

# 2) ACCEPTANCE 2 -- coarse 선언은 output 과 무관하게 screening 이다
#    기대: policy=L3coarse, screening=true, acceptance_eligible=false, 종료코드 2
env -u RASBERY_ALLOW_SCREENING CUDA_VISIBLE_DEVICES=0 RASBERY_PHYSICS_FIDELITY=L3coarse \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238.json" --raso "$OUT/coarse_full.h5" --result full \
  2>&1 | tee "$OUT/a2.log" | grep -E '\[EXACT_ONLY\]'; echo "exit=$?"
#    선언은 더 거칠게만: A2 실행에 strict 를 선언해도 policy 는 A2 여야 한다
env RASBERY_ALLOW_SCREENING=1 CUDA_VISIBLE_DEVICES=0 \
    RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000 RASBERY_PHYSICS_FIDELITY=strict \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238.json" --raso "$OUT/a2_decl.h5" --result light \
  2>&1 | grep -o '"policy":"[^"]*"'      # 기대: "policy":"A2"

# 3) ACCEPTANCE 3 -- GPU full 에서 모든 fallback 이 0
#    기대: [RASBERY][GPU_FULL] 의 *_fallbacks 전부 0, contract_pass:true, 종료코드 0
source "$REPO/docs/238_gpu_arms.env" 2>/dev/null || {
  export RASBERY_GPU=1 RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 \
         RASBERY_GPU_NODAL=1 RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 \
         RASBERY_GPU_FLATXS=1 RASBERY_GPU_WIEL_FOLD=chunked RASBERY_GPU_XE=1 \
         RASBERY_IO_WRITER=thread; }
CUDA_VISIBLE_DEVICES=0 RASBERY_GPU_FULL=1 \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238.json" --raso "$OUT/gpufull.h5" --result full \
  2>&1 | tee "$OUT/a3.log" | grep -E '\[GPU_FULL\]|\[GPU_FULL\]\[VIOLATION\]|\[RASBERY\]\[FAIL\]'

# 4) ACCEPTANCE 4 -- feature-off 는 이전 바이너리와 B0
#    (이전 8bd5112 빌드의 같은 arm 결과를 $OUT/base.h5 로 미리 확보해 둘 것)
CUDA_VISIBLE_DEVICES=0 "$BLD/RASBERY" --rasi "$DATA/kngr_238.json" \
  --raso "$OUT/wp1_off.h5" --result full 2>&1 | tee "$OUT/a4.log"
h5diff -r "$OUT/base.h5" "$OUT/wp1_off.h5" && echo "B0 OK"
diff <(grep -o '\[RASBERY\]\[TRAJECTORY\].*' "$OUT/base.log") \
     <(grep -o '\[RASBERY\]\[TRAJECTORY\].*' "$OUT/a4.log") && echo "digest OK"

# 5) ACCEPTANCE 5 -- 위반은 케이스 단위이고 배치를 죽이지 않는다
#    4덱 중 하나만 ng!=2 또는 rod-fraction 덱으로 바꿔 device 를 강제 거절시킨다.
#    기대: [RASBERY][FAIL] 이 그 덱 하나만, 나머지 3덱의 h5 는 정상 생성
CUDA_VISIBLE_DEVICES=0 RASBERY_GPU_FULL=1 \
  "$BLD/RASBERY" --jobs "$OUT/m4_one_bad.jobs" --batch-mode 4 --result full \
  2>&1 | tee "$OUT/a5.log" | grep -E '\[RASBERY\]\[FAIL\]|\[GPU_FULL\]'
ls -l "$OUT"/m4_*.h5     # 기대: 3개

# 6) 성능 판정 -- 계약 계측 오버헤드 <1% (게이트 OFF, 교대 5쌍)
#    A = 8bd5112 바이너리, B = WP1 바이너리, 같은 arm/같은 result mode
#    warm-up 1회 버리고 A1 B1 A2 B2 ... A5 B5, median/p10/p90

# ===========================================================================
# 7) ACCEPTANCE 6 -- F9/F10/F11: 죽어 있던 카운터가 실제로 움직인다
# ===========================================================================
#    소스 스캔(tools/test_receipt_counters_live.py)은 "쓰는 자리가 있다"까지만
#    증명한다.  0 에서 벗어나는 것은 여기서만 확인된다.  그리고 그 확인은
#    반드시 두 방향이어야 한다: 정상 arm 에서 0 이고, 강제 fallback 에서 0 이
#    아니어야 한다.  둘 중 하나만 보면 "항상 0" 과 "항상 1" 을 구분하지 못한다.

# 7a) 정상 arm: 세 필드 전부 0, 그리고 융합 Xe 삼중항이 합이 맞는다
source "$REPO/docs/238_gpu_arms.env" 2>/dev/null || true
CUDA_VISIBLE_DEVICES=0 \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238.json" --raso "$OUT/f9_ok.h5" --result full \
  2>&1 | tee "$OUT/f9_ok.log" \
  | grep -oE '"(xs|cmfd|nodal)_cpu_fallbacks":[0-9]+|"fused_[a-z_]+":[0-9]+'
#    기대: xs/cmfd/nodal_cpu_fallbacks 전부 0
#          fused_updates == fused_device_updates + fused_host_fallbacks (항등식)
#    주의: --batch-mode 없이 RASBERY_GPU_CMFD_RESIDENT_SINGLE 도 없으면 비-arena
#          판(src/BICGSolver.cpp)이, 있으면 arena 판(CudaBICGBackend.cu)이 찍힌다.
#          두 판 모두 세 필드를 인쇄해야 한다 -- 한쪽만 나오면 그것이 결함이다.

# 7b) 강제 fallback 하나: nodal 이 가장 싸다.
#     ACCEPTANCE 5 가 이미 쓰는 rod-fraction(또는 ng!=2) 덱은 TryDriveGpu 를
#     확실히 거절시킨다.  게이트는 OFF 로 둔다 -- 여기서 보려는 것은 실패가
#     아니라 "숫자가 움직이는가" 다.
env -u RASBERY_GPU_FULL -u RASBERY_GPU_STRICT CUDA_VISIBLE_DEVICES=0 \
    RASBERY_GPU=1 RASBERY_GPU_NODAL=1 \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238_rodfrac.json" --raso "$OUT/f9_fb.h5" \
  --result full 2>&1 | tee "$OUT/f9_fb.log" \
  | grep -oE '"nodal_cpu_fallbacks":[0-9]+|"nodal_fallbacks":[0-9]+|"contract_pass":[a-z]+'
#    기대: [BACKEND_COUNTERS].nodal_cpu_fallbacks > 0
#          그리고 [GPU_FULL].nodal_fallbacks 와 **같은 값**  <- 하나의 진실
#          contract_pass:false
#    두 숫자가 다르면 F9 의 배선이 두 번째 tally 가 되어 버린 것이다.

# 7c) 같은 실행을 게이트 ON 으로: 케이스가 실패해야 한다
CUDA_VISIBLE_DEVICES=0 RASBERY_GPU=1 RASBERY_GPU_NODAL=1 RASBERY_GPU_FULL=1 \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238_rodfrac.json" --raso "$OUT/f9_gate.h5" \
  --result full 2>&1 | grep -E '\[GPU_FULL\]\[VIOLATION\]|\[RASBERY\]\[FAIL\]'; echo "exit=$?"
#    기대: subsystem=nodal site=Nodal::drive 가 이름으로 찍히고 종료코드 != 0

# 7d) 융합 Xe 팔(F10): 분할 팔을 끄고 융합 팔만 켠다
env -u RASBERY_GPU_XE CUDA_VISIBLE_DEVICES=0 \
    RASBERY_GPU=1 RASBERY_GPU_XSRECON=1 \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238.json" --raso "$OUT/f10.h5" --result full \
  2>&1 | grep -o '\[RASBERY\]\[XE_GPU\].*'
#    기대: fused_updates > 0 (팔이 실제로 불렸다)
#          fused_device_updates > 0 이면 engage, 0 이면 "켜졌지만 전부 거절"
#          -- WP1 이전에는 이 둘을 구분할 방법이 아예 없었다.

# 7e) outer 거절 사유(F11): 무명 거절이 남지 않는다
env -u RASBERY_GPU_CMFD_SWEEP CUDA_VISIBLE_DEVICES=0 \
    RASBERY_GPU=1 RASBERY_GPU_OUTER=1 \
  "$BLD/RASBERY" --rasi "$DATA/kngr_238.json" --raso "$OUT/f11.h5" --result full \
  2>&1 | grep -o '\[RASBERY\]\[OUTER_GPU\].*'
#    기대: device_outers:0 이면서 refusals{} 가 **비어 있지 않다**.
#    판정 규칙(이것이 F11 의 계약이다): 세그먼트가 하나도 안 섰는데
#    sum(refusals.values()) == 0 이면 어딘가에 아직 무명 거절이 있다.
#    no_residency / unbound 는 이 arm 으로는 나오지 않는다(sweep_not_resident 가
#    먼저 잡는다).  둘은 arena/bind 가 실패해야 나오므로 여기서는 "이름이 있는
#    사유로 전부 설명되는가" 만 본다.
```

> **주의:** 3)의 `[GPU_FULL]`이 0이 아니면 **그것이 결과다.** 그 실행은 A/B의 기준선으로 쓸 수 없고, `site=`가 어느 팔이 조용히 CPU였는지 말한다. 그 경우 WP2의 대상 기능 표에 “그 팔은 238에서 실제로 engage 하지 않는다”를 먼저 기록한다.

> **7)에 대한 주의:** 7b)의 두 숫자(`[BACKEND_COUNTERS].nodal_cpu_fallbacks`와 `[GPU_FULL].nodal_fallbacks`)가 **다르면** F9의 수정이 목적을 잃은 것이다. 두 필드는 같은 원자(`gpufull::detail::counter`)를 읽도록 배선했고, 다르다는 것은 어딘가에서 두 번째 tally가 생겼다는 뜻이다 — F13이 금지한 바로 그 상태.

---

## 9. 238 runbook — WP3 CMFD compaction 가격 평가

WP3에 남은 것은 코드가 아니라 **가격**이다(§1 재분류, §5 표). 아래는 그 측정이다.

전제: §8-0)의 빌드, `docs/W4_L5_MULTIPROC_PER_GPU_20260901_KO.md` §4.1의 환경,
§4.2의 64잡 매니페스트. **하네스 없이 재지 말 것** — 환경 불일치는 이 캠페인에서 오류도
FAIL 줄도 남기지 않는 유일한 결함이다(W4_L5 §4.7).

```bash
export REPO=$HOME/gpu; export B=$HOME/build/rasbery-wp3-sm120
export O=$HOME/bench/wp3; mkdir -p $O
export RASBERY_ALLOW_SCREENING=1          # --result light (권한은 운영자의 것)
cd $REPO
python3 tools/test_cmfd_slot_compaction_contract.py   # 하드웨어 불필요, 같은 트리에서
python3 tools/test_cmfd_slot_compaction_contract.py --run   # nvcc replay 포함
```

> **`RASBERY_GPU_CMFD_COMPACT`는 `kArmEnv`에 없다** (`src/Driver.h:415-443`). 그것은
> 선언이다: 이 기능은 B0이며 궤적을 움직이지 않는다. 따라서 `[TRAJECTORY]` digest의
> `env` 필드가 ON/OFF에서 **같고**, digest를 직접 비교할 수 있다. **digest가 움직이면
> 그 선언이 틀린 것이고, 그 자리에서 N1로 재분류하고 Gate B를 태운다**(R3).
> 그리고 `DEFAULT_ENV`가 주지 않는 키다 — **반드시 `--set`으로 준다.** export도
> 상속되기는 하지만(자식은 `os.environ`을 물려받는다) 그때 A/B의 arm은 **셸의 상태**가
> 되고, 두 arm을 한 루프에서 도는 이 runbook에서 그것은 정확히 W4_L5 §5 함정 10이다.
> `--set`으로 주면 `[MULTI_GPU][ENV]`와 `[PROC].env`가 **무엇이 실제로 갔는지** 적는다.

### 9.1 상금부터 본다 (GPU 시간 0)

`[RASBERY][CMFD][COMPACT]`는 **ON/OFF 무관하게 인쇄된다**(F4). 그러므로 OFF 실행 하나로
“고칠 것이 얼마나 있는가”를 먼저 읽는다 — padding이 이미 작다면 나머지 단계는 돌릴
필요가 없다.

```bash
python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 1 --batch-width 64 \
    --claim auto --result light --jobs $O/m64.txt --cwd ~/t18decks/kngr \
    --workdir $O/w/prize --pin taskset -- $B/RASBERY > $O/prize.out 2>&1
python3 tools/cmfd_compact_receipt.py $O/w/prize/*.log
```

기대 출력(사다리는 소스에서 읽는다 — 9칸, R2):

```text
# bucket ladder from source: [1, 2, 4, 8, 16, 24, 32, 48, 64]
...: enabled=0 logical_drives=N physical=N padding=N padding_fraction=0.xxxx ...
    buckets: <=1=.., <=2=.., <=4=.., <=8=.., <=16=.., <=24=.., <=32=.., <=48=.., <=64=..
```

**판정 규칙:** `padding_fraction`이 0.10 미만이면 WP3의 상한은 10 %이고, 아래 A/B의
목표(+5 %)는 사실상 도달 불가능하다 — 그 경우 **그 숫자를 적고 WP3을 `reject`로 닫는다.**
`bucket_histogram`은 어느 rung에 무게가 있는지 말한다: 14.5 부근이면 `<=16`과 `<=24`가
무거워야 한다. `<=64`가 무겁다면 도착 폭이 이미 차 있다는 뜻이고, 그것도 결과다.

### 9.2 A/B — 1×M64 (control 폭)

교대 5쌍, 워밍업 1회 버림. **arm마다 `--workdir`를 다르게** (W4_L5 §5 함정 5).

```bash
for i in 1 2 3 4 5; do
  for v in 0 1; do
    python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 1 --batch-width 64 \
        --claim auto --result light --jobs $O/m64.txt --cwd ~/t18decks/kngr \
        --workdir $O/w/m64_c${v}_${i} --pin taskset \
        --set RASBERY_GPU_CMFD_COMPACT=$v \
        -- $B/RASBERY > $O/m64_c${v}_${i}.out 2>&1
    grep -h "MULTI_GPU\]\[TOTAL\]" $O/m64_c${v}_${i}.out
  done
done
grep -ho '"RASBERY_GPU_CMFD_COMPACT":"[01]"' $O/m64_c*_1.out | sort | uniq -c   # env 확인
```

### 9.3 A/B — 8×M8 + MPS (폭이 찬 arm)

**이 arm이 WP3의 진짜 시험이다.** W4_L5 §4.8이 `width_fill`을 0.28 → 0.41로 올렸다.
compaction이 지우는 것은 **선언 폭과 도착 폭의 차이**이므로, 도착 폭이 이미 오른 arm에서
상금은 **작아져 있어야 한다.** 두 arm의 `padding_fraction`을 나란히 적는 것이 이 WP의
가장 중요한 한 줄이다 — WP3과 WP4가 같은 비용을 두 방법으로 지우기 때문이다(계획 §3).

```bash
for i in 1 2 3 4 5; do
  for v in 0 1; do
    python3 tools/run_multi_gpu_batch.py --gpus 0 --procs-per-gpu 8 --total-width 64 \
        --claim auto --result light --mps --jobs $O/m64.txt --cwd ~/t18decks/kngr \
        --workdir $O/w/m8_c${v}_${i} --pin taskset \
        --set RASBERY_GPU_CMFD_COMPACT=$v \
        -- $B/RASBERY > $O/m8_c${v}_${i}.out 2>&1
    grep -h "MULTI_GPU\]\[TOTAL\]\|MULTI_GPU\]\[MPS\]" $O/m8_c${v}_${i}.out
  done
  pgrep -a nvidia-cuda-mps        # arm 사이 잔존 확인. 비어 있어야 한다
done
python3 tools/cmfd_compact_receipt.py $O/w/m8_c0_1/*.log $O/w/m8_c1_1/*.log
```

`[MPS].active:true`가 아닌 arm은 무효다(W4_L5 §4.5).

### 9.4 B0 — OFF는 이전 바이너리와 바이트 동일

compaction OFF는 **완전 항등**이어야 한다(계약 테스트 5번). 그것을 소스가 아니라
디스크에서 확인한다. `$O/base.h5`는 `3e9e3e5` 빌드의 같은 arm 결과.

```bash
CUDA_VISIBLE_DEVICES=0 $B/RASBERY --rasi $DATA/kngr_238.json \
    --raso $O/off.h5 --result full 2>&1 | tee $O/off.log
h5diff -r $O/base.h5 $O/off.h5 && echo "B0 OK (identity-off)"
diff <(grep -o '\[RASBERY\]\[TRAJECTORY\].*' $O/base.log) \
     <(grep -o '\[RASBERY\]\[TRAJECTORY\].*' $O/off.log) && echo "digest OK"
```

### 9.5 ON×2 digest 동일 — 그리고 ON vs OFF도

```bash
for i in 1 2; do
  CUDA_VISIBLE_DEVICES=0 RASBERY_GPU_CMFD_COMPACT=1 $B/RASBERY \
      --rasi $DATA/kngr_238.json --raso $O/on_$i.h5 --result full \
      2>&1 | tee $O/on_$i.log | grep -o '\[RASBERY\]\[TRAJECTORY\].*'
done
diff <(grep -o '\[RASBERY\]\[TRAJECTORY\].*' $O/on_1.log) \
     <(grep -o '\[RASBERY\]\[TRAJECTORY\].*' $O/on_2.log) && echo "ON x2 digest OK"
h5diff -r $O/on_1.h5 $O/on_2.h5 && echo "ON x2 bytes OK"
# 그리고 B0의 본체: ON 이 OFF 와 같은가
h5diff -r $O/off.h5  $O/on_1.h5 && echo "B0 OK (ON == OFF)"
diff <(grep -o '\[RASBERY\]\[TRAJECTORY\].*' $O/off.log) \
     <(grep -o '\[RASBERY\]\[TRAJECTORY\].*' $O/on_1.log) && echo "B0 digest OK"
```

> ON×2를 먼저 보는 이유: **ON 경로가 비결정적이면 ON≠OFF는 compaction의 증거가 아니다.**
> 계획은 ON×3을 적었다. ×2가 통과하고 ×3이 필요해지는 경우는 하나뿐 — ON×2가 갈렸을 때이며,
> 그때는 개수를 늘리는 것이 아니라 **비결정성을 먼저 보고한다.**

### 9.6 판정표

| 항목 | 어디서 | 채택 기준 |
|---|---|---|
| 상금 | §9.1 `padding_fraction` (OFF) | <0.10이면 `reject`. 그 숫자를 적고 닫는다 |
| 1×M64 처리량 | §9.2 `[TOTAL].cases_per_hour` 중앙값 | ON/OFF ≥ **+5 %** |
| 8×M8+MPS 처리량 | §9.3 동일 | ON/OFF ≥ +5 %. **여기서 사라지면 WP4가 이미 그 비용을 지운 것이다** |
| padding 감소 | §9.1/§9.3 두 arm의 `padding_fraction` | ON에서 유의하게 감소 |
| graph 폭증 없음 | `bucket_graphs` | bucket 수(9)를 넘어 계속 증가하지 않는다 |
| B0 | §9.4, §9.5 | `h5diff -r` 비고 `[TRAJECTORY]` digest 동일 — OFF vs 이전 바이너리, **그리고 ON vs OFF** |
| 결정성 | §9.5 | ON×2 digest·바이트 동일 |
| 감사 | 모든 arm | `rc=0`, `fail_lines=0`, `duplicates/stale_tenants=0` |

digest가 움직이면 **그 자리에서 N1로 기록하고 Gate B를 태운다**(R3). 성능 숫자는
그 다음이다.

### 9.7 이 runbook이 로컬에서 확인할 수 없는 것

- padding의 실제 크기. 소스는 “세는 자리가 있다”까지만 말한다.
- ON 경로의 정확성. 계약 테스트는 **정적**이다 — `--run`의 replay는 nvcc가 있어야 하고,
  비연속 physical slot `{1,4,7,31}` 강제와 tenancy 교체 후 stale map은 계획 §WP3
  “테스트 우선 절차” 2·4번이며 `test/cmfd_slot_compaction_replay.cu`를 확장해야 한다.
  **이 트리에는 아직 없다.**
- bucket graph 인스턴스 수의 실제 추이.

---

## 부록 A — 상위 fail-open seam 목록 (WP1 가드의 근거)

`.cu` 내부의 수백 개 `return false`는 전부 아래 중 하나로 올라온다. 그래서 가드는 여기에만 있다.

| 서브시스템 | 내부 거절이 모이는 지점 | 가드 |
|---|---|---|
| CMFD | `BICGCMFD::drive` — `driveDeviceSweeps()` false → 순정 host BiCGSTAB | `src/BICGCMFD.cpp:876` |
| Nodal | `Nodal::drive` — `TryDriveGpu()` false → `driveBody()` | `src/Nodal.cpp:857` |
| FlatXS | `XSSet::UpdateFlatXS` — device arm 거절 → reference 루프 | `src/XSSet.cpp:3092` |
| Xe | `XSSet::UpdateEquilibriumXenon` — 분할/융합 두 팔 | `src/XSSet.cpp:4131`, `:4146` |
| CRAM | `XSSet::Deplete` / `PredictorCorrectorStep` | `src/XSSet.cpp:4545`, `:4693` |
| Outer | `Driver::ReconvergeFlux` / `SolveLoop` — pre-arm 실패, 비유한 escape, runSegment 실패 | `src/Driver.h:1980`, `:2042`, `:2062`, `:3117`, `:3237` |
| PPR | `Driver` 상태점 루프 — `resetAndDriveGpu()` false → host reset+drive | `src/Driver.h:4222` |
| Outer (count-only) | `Driver::outerSweepEnqueueHook` — 살아 있는 segment 안의 blocking drive | `src/Driver.h:1229` |

`src/BICGSolver.cpp:86-90`(stand-up)은 **이미 fail-closed**이며, `tools/test_gpu_full_fail_closed.py`가 그 `throw`의 존재를 붙든다 — 그것이 fallback으로 바뀌면 seam 스캔은 볼 수 없기 때문이다(seam 자체가 생기지 않는다).
