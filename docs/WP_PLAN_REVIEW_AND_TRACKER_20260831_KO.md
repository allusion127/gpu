# WP 계획 검토와 추적표 — `GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md`

| 항목 | 값 |
|---|---|
| 검토 대상 | `docs/GPU_RASBERY_BOTTLENECK_PARALLEL_ACCELERATION_IMPLEMENTATION_PLAN_20260830_KO.md` (1,658행, 원본과 verbatim) |
| 계획의 고정 SHA | `8b8f18e1c837121bd7873b43c207dafeda632edd` |
| 검토 시점 트리 | `codex/exact-throughput-campaign` @ `8bd5112` (= `8b8f18e` + Task 16 GPU CRAM). 검토 도중 하네스 커밋 `7099e54`가 착지했고, 아래 하네스 인용은 그 이후 줄번호다. |
| 검토 방법 | 정적 소스 검토만. **이 검토에서 서버 238 실행·빌드·프로파일은 수행하지 않았다.** |
| 작성 | 2026-08-31 KST |

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
`BackendCounters::xs_cpu_fallbacks`(선언 `src/CudaBICGBackend.h:53`, 인쇄 `src/BICGSolver.cpp:108`), `cmfd_cpu_fallbacks`(`:55` / `:110`), `nodal_cpu_fallbacks`(`:84` / `:131`) — **소스 어디에도 `++`가 없다.** 늘 0을 보고한다. WP2가 “기능 flag가 켜짐이 아니라 engagement receipt가 0보다 큰 실행만 유효하다”를 이 필드들로 판정하려 했다면, 그 판정은 상수를 읽고 있었다. WP1의 `[RASBERY][GPU_FULL]`이 cmfd/nodal/flatxs/xe/ppr/cram/outer를 실제로 센다. **남은 작업: 세 죽은 필드를 배선하거나 삭제한다(WP2).**

**F10. 융합 Xe 팔(`RASBERY_GPU_XSRECON`)의 fallback은 카운터가 전혀 없었다.**
분할 팔은 `src/XSSet.cpp`의 `tally.host_fallbacks`로 세지만, 융합 팔은 “any failure falls through to the unchanged CPU loop” 주석만 있고 아무것도 세지 않았다. 즉 `[RASBERY][XE_GPU]`의 `device_updates=0`을 보고도 “팔이 꺼졌나 거절했나”를 구분할 수 없었다. WP1이 두 자리 모두 가드했다.

**F11. outer segment의 두 arm 실패는 이름조차 없다.**
`src/Driver.h`의 `armOuterSegment`에서 `if (!view.valid) return false;`(`src/Driver.h:1570`)와 `if (!gpu::rasberyBindOuterResidency(residency)) return false;`(`src/Driver.h:1738`)는 `noteOuterSegmentRefusal`을 부르지 않고 반환한다. 따라서 `[OUTER_GPU].refusals`에 사유가 남지 않는다. WP1의 pre-arm 가드가 GPU full에서 이를 치명적으로 만들지만, **사유는 여전히 무명**이다. → **WP7 단계 A의 “필수 receipt” 목록에 이 둘을 추가할 것.**

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

**R2. WP3 bucket ladder: 계획 7칸, 트리 9칸.**
계획 §WP3 구현 세부는 “bucket은 `1,2,4,8,16,32,64`로 제한”이라 적었다. 트리는 **`1,2,4,8,16,24,32,48,64` 9칸**이다 — `src/CudaBICGBackend.cu:424`, 히스토그램 배열 크기 9 `:547`, receipt의 `bucket_histogram` 길이 9 `:5638-5644`. 그리고 계약 테스트는 “bucket ladder는 스케줄러의 사다리이지 두 번째 사다리가 아니다”를 명시적으로 붙들고 있다(테스트 docstring 6번).
→ **결정 필요:** (a) 트리의 9칸을 채택하고 계획 문장을 고친다(권고 — 24/48은 M64의 실제 도착 폭 14.5 부근을 훨씬 잘 덮는다), 또는 (b) 7칸으로 줄이고 스케줄러 사다리도 함께 바꾼다. **파서를 계획대로 7칸으로 쓰면 receipt와 어긋난다.**

**R3. WP3의 등급이 문서 간에 다르다.**
새 계획은 CMFD compaction을 **B0**로 적는다(§WP3 정확성 게이트 “B0가 목표다”, §9 표). GA evaluator 계획은 같은 것을 **N1**으로 적는다(`docs/GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md:28`, `:868`).
→ **권고:** B0를 *목표*로 두고, 238의 첫 ON×3 digest + `h5diff -r`를 판정자로 삼는다. digest가 움직이면 그 자리에서 N1로 기록하고 Gate B를 태운다. 두 문서 중 하나를 지금 고치는 것보다 측정이 싸다.

**R4. WP0/WP2/WP4가 모두 하네스 파일을 건드린다 — 지금 다른 작업자가 그 파일들을 수정 중이다.**
이 검토 도중 그 작업의 1차분이 `7099e54`로 착지했다(두 하네스 + `tools/test_multi_gpu_dispatch.py`). 아직 워킹트리에 남은 것: `?? tools/test_harness_env_parity.py`, `?? test/reference/batch_reference_env_238.json`.
→ **순서 강제:** 하네스 env-parity 작업이 커밋된 뒤에 WP0의 `run_exact_throughput_matrix.py`/`parse_perf_receipts.py`를 얹는다. WP4의 tuner는 그 위에.

**R5. WP1의 acceptance 감사 전환은 하네스 파일을 고쳐야 완결된다 — 이번 커밋에 포함하지 않았다.**
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
| **WP1** | 출력/충실도 분리 + GPU full fail-closed | **done (코드·계약)** / 하네스 절반 **not-started** | 본 작업 | 238에서 §8 runbook 4줄 통과 + 오버헤드 <1% | R5(하네스 감사 전환), 238 빌드 |
| **WP2** | 착지 기능 238 재가격 | **not-started** | 하네스 담당 + 본 작업 | 6개 기능 각각 `adopt`/`keep-opt-in`/`reject` | WP0, WP1(모드 일치 비교가 성립해야 함), R4 |
| **WP3** | CMFD compaction 검증·채택 | **landed-unpriced** (계약 테스트까지 존재, F1–F5) | 본 작업 | M64 mode-matched +5%, padding −30%, ON×3 digest 동일 | WP2, R2(ladder 결정), R4 |
| **WP4** | 단일 GPU K-process 자동 튜닝 | **landed-unpriced** (launcher만) | 하네스 담당 | K=2가 1×64 대비 ≥1.05× | WP0, R1(하네스 결함 수정), R4 |
| **WP5** | FlatXS CTA-per-node + XS residency | **not-started** | 본 작업 | 단계 A(ptxas/ncu spill 증명) → 분기 | **R1: 고친 하네스에서 §3.3 재측정**, WP3+WP4 |
| **WP6** | GPU PPR device loop/reconstruct/canonical | **부분 landed-unpriced** (`c502856` 장치 PPR, 기본 OFF·fail-open) | 본 작업 | 단계 A: `ppr_device_calls == statepoints`, `ppr_host_calls == 0` (GPU full), N1 Gate A/B | WP2(PPR 1차 A/B), WP1(fail-closed로 engagement 증명) |
| **WP7** | 단일 launch/sync 축소 + Xe transaction | **부분 landed-unpriced** (WHILE 구현·기본 OFF) | 본 작업 | 단계 A: 단일 WHILE ≥5% (교대 5쌍) | WP2, F11(무명 refusal 두 개를 receipt에 추가) |
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

---

## 8. 238 runbook — WP1 acceptance

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
```

> **주의:** 3)의 `[GPU_FULL]`이 0이 아니면 **그것이 결과다.** 그 실행은 A/B의 기준선으로 쓸 수 없고, `site=`가 어느 팔이 조용히 CPU였는지 말한다. 그 경우 WP2의 대상 기능 표에 “그 팔은 238에서 실제로 engage 하지 않는다”를 먼저 기록한다.

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
