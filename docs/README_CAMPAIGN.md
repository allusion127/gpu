# GPU RASBERY 가속 캠페인 — 문서 색인

**브랜치** `codex/exact-throughput-campaign` · **최신 팁** `bd7a0d3`(WP18) · **v5 동결** `b903225` · **v6 후보** `d68de15`
**대상** APR1400/KNGR CY1 PSAR 1주기(`kngr_238.json`, 35상태 자연 EOC, 1/4 노심 8,451 노드, NG=2)
**기준기** MASTER (CPU) — 단일 27.2 s / W16 배치 217 case/h

> 저장소 루트의 `README.md`는 상위 RASBERY/HYUSENM 프로젝트의 것이다. **이 파일이 캠페인
> 문서의 색인**이다.

---

## 여기서 시작한다

| 무엇을 알고 싶은가 | 읽을 것 |
|---|---|
| **지금 얼마나 빠른가, 어디가 병목인가, 코드가 어떻게 생겼는가** | **[`GPU_RASBERY_PERFORMANCE_AND_ARCHITECTURE_REPORT_20260830_KO.md`](GPU_RASBERY_PERFORMANCE_AND_ARCHITECTURE_REPORT_20260830_KO.md)** ← **종합 보고** |
| 무엇을 어떤 순서로 만들 것인가 (Task 0–28, Wave) | [`GPU_RASBERY_100_PERCENT_GPU_ASYNC_SCHEDULER_PLAN_REV7_1_KO.md`](GPU_RASBERY_100_PERCENT_GPU_ASYNC_SCHEDULER_PLAN_REV7_1_KO.md) |
| 20×는 정말 가능한가, GPU 몇 장이 필요한가 | [`GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md`](GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md) §0.2, §4 |
| **방금 무엇이 채택되고 무엇이 기각됐는가 (최신 실측)** | **[`PRICING_PROD_PHASE2_20260831_KO.md`](PRICING_PROD_PHASE2_20260831_KO.md)** ← **2단계 가격 평가(v5 이후)** |
| 지금 무엇을 어떤 env로 돌려야 하는가 | **[`PRICING_PROD_PHASE2_20260831_KO.md`](PRICING_PROD_PHASE2_20260831_KO.md) §10** (동결 arm은 [`V5_FREEZE_20260830_KO.md`](V5_FREEZE_20260830_KO.md) §1, v3 env는 [`V3_FREEZE_20260829_KO.md`](V3_FREEZE_20260829_KO.md) §2에 그대로 있다) |
| 답이 맞는지 어떻게 아는가 | 종합 보고 §5 + [`MASTER_vs_RASBERY_COMPARISON_20260824_KO.md`](MASTER_vs_RASBERY_COMPARISON_20260824_KO.md) |

**한 줄 현황**: 단일덱 **9.81 s = MASTER 27.2 s 대비 2.77×**, 배치 **m64 1,246 c/h =
MASTER W16 217 c/h 대비 5.7×**(8×M16+MPS auto, 128잡 매니페스트), 그리고 **같은 덱으로 잰
KNGR×64 배치 2,016 c/h = 1.786 s/case** — 팁 `bd7a0d3`, 측정 arm은 `d68de15`의
**v6 + `CMFD_BLOCK=64`**(238 `1f36e75dc00ed2b4`/4377, 2026-08-31).
PROD 기준선 16.658 s에서 **−6.850 s(−41.1 %)**이고, **v5(11.189 s) 이후의 −1.381 s는 전부 B0다** —
`OUTER_GRAPH`(그래프로 캡처한 WHILE 루프가 **h5diff 0으로 비트가 같았다**, 런북이 가정한 N1보다 강하다) ·
`MICX_RESIDENT`(micx D2H 18.69 → 3.31 GB, **배치 +19.3 %**) · `WP15.1`(CRAM mic H2D **0 바이트**) ·
`CMFD_BLOCK=64`(같은 일을 더 많은 더 작은 블록으로, `vector_blocks` 67 → 265).
정확도는 v2 대비 **비열화 그대로**(반응도 1.847 pcm / CBC 15.334 ppm / AO 0.012 /
BOC 핀 RMS 0.238 % · max 0.80 %). **병목의 이름이 다시 바뀌었다** — PCIe는 Gen4 x16의 25–35 %만 쓰고,
배치 CPU 74 %는 실작업이 아니라 `cudaStreamSynchronize` 스핀이며, 배치는 폭 기아가 아니라
**GPU 시간 바운드**다(WP18이 꼬리를 0으로 만들고도 c/h가 내려갔다). **남은 큰 레버는 A2 outer 감축 하나다.**
→ **[`PRICING_PROD_PHASE2_20260831_KO.md`](PRICING_PROD_PHASE2_20260831_KO.md)** ·
[`V5_FREEZE_20260830_KO.md`](V5_FREEZE_20260830_KO.md) ·
[`PRICING_PROD_20260830_KO.md`](PRICING_PROD_20260830_KO.md)

---

## 1. 계획 문서 (무엇을 만들 것인가)

| 문서 | 줄 | 내용 |
|---|---:|---|
| [`GPU_RASBERY_100_PERCENT_GPU_ASYNC_SCHEDULER_PLAN_REV7_1_KO.md`](GPU_RASBERY_100_PERCENT_GPU_ASYNC_SCHEDULER_PLAN_REV7_1_KO.md) | 2,607 | **마스터 계획.** Task 0–28, Wave W0–W5+, 게이트 등급 B0/N1/A2 정의, M1/M2 마일스톤, §13 성능 목표·호스트 예산, §14 완료 정의. Task 18a–18d 수신증 포함 |
| [`GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md`](GPU_RASBERY_GA_EVALUATOR_PLAN_20260831_KO.md) | 981 | **케이스 비용 모델**(`t_case = T_process + T_percase + N_sp·c + N_outer·d`), 레버 **L1–L8**, **정직한 상한**(공학 1,600–2,100 c/h), evaluator 계약, GA 2단 파이프라인 |
| [`GPU_RASBERY_EXACT_THROUGHPUT_ACCELERATION_PLAN_REV4_KO.md`](GPU_RASBERY_EXACT_THROUGHPUT_ACCELERATION_PLAN_REV4_KO.md) | 1,068 | Rev.7.1의 전신. **§14 = XSLIB 캐시 설계**(Rev.7.1 §14가 아니다), §13 = 멀티GPU 설계 |
| [`PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md`](PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md) | 1,165 | persistent/cooperative CMFD + **slot compaction** 설계. persistent는 W0에서 폐쇄, compaction은 살아 있다 |
| [`SINGLE_GPU_BATCH_OPTIMIZATION_PLAN_20260824_KO.md`](SINGLE_GPU_BATCH_OPTIMIZATION_PLAN_20260824_KO.md) | 158 | 단일 GPU 동시입력(배치) 최초 계획 |

## 2. 결과·판정 보고 (무엇이 측정되었는가)

| 문서 | 줄 | 판정 |
|---|---:|---|
| **[`V5_FREEZE_20260830_KO.md`](V5_FREEZE_20260830_KO.md)** | — | **v5 동결(2026-08-30).** 두 호스트 게이트 표 6행, env 전문, 기본값이 된 넷과 되지 않은 둘의 이유, `case_key`가 갈리는 자리, 단일 11.189 s / 배치 981.7 c/h, 기각된 warm-start, 재현 절차(`gate_b_pin_rms.py` 포함), 아직 갚지 않은 확인 실행 1건. 매니페스트 `test/reference/validation_baseline_manifest_v5.json`(`frozen:true`) |
| **[`PRICING_PROD_PHASE2_20260831_KO.md`](PRICING_PROD_PHASE2_20260831_KO.md)** | — | **238 2단계 가격 평가(2026-08-31, 블록 19–33)**: 채택 레버 사다리 **16.658 → 14.378 → 11.952 → 11.189 → 10.206 → 9.808 s**(단계별 등급 포함), v6 스택 구성요소별 가격표, 전송 원장(D2H −75 %, sync 호출 −40 %), **WP17 커널 점유율 인구조사**(상위 15 중 14가 2×SM 미만), **배치 진실표**(형상·덱 무게 1.92×·KNGR×64 2,016 c/h·MPS 1.98× vs 폭 1.92×·CPU 스핀 74→40 %), **기각·폐쇄 목록**(persistent `c_barrier` 0.788 µs NO-GO · rolling refill · WP9-D · claim 청크 · 8프로세스 초과 형상), 보고서가 열어 둔 구멍 8건, 다음 레버 넷. 후보 매니페스트 `test/reference/validation_baseline_manifest_v6_candidate.json` |
| **[`PRICING_PROD_20260830_KO.md`](PRICING_PROD_20260830_KO.md)** | — | **238 PROD env 가격 평가(2026-08-30)**: PPR(WP6-F)·CRAM(Task 16)·FUSE(WP7-B)·XE_TXN(WP7-C) 기능별 등급·게이트·채택 판정, v4 전체 스택 **14.378 s = 1.89×**, v4 후보 env, 폐기된 수(arm-X env 아티팩트 · 181에서 채굴된 `0xadd` pin), **phase 분해 + nsys 커널 프로파일**(두 계기가 어긋난다), 다음 레버 셋, 교훈 |
| **[`GPU_RASBERY_PERFORMANCE_AND_ARCHITECTURE_REPORT_20260830_KO.md`](GPU_RASBERY_PERFORMANCE_AND_ARCHITECTURE_REPORT_20260830_KO.md)** | 1,080 | **본 캠페인 종합**: 단계별 성능 사다리, nsys 커널/API/osrt 표, 케이스 비용 모델, 병목 원장, 게이트 체계, 결함 19건, 상한, 재현 명령 |
| [`V3_FREEZE_20260829_KO.md`](V3_FREEZE_20260829_KO.md) | 307 | **v3 생산 arm 정의**, 구성요소별 게이트 등급, 계기 중립성, 동결 절차·롤백 |
| [`A2_OUTER_REDUCTION_20260829_KO.md`](A2_OUTER_REDUCTION_20260829_KO.md) | 291 | **A2 = 최대 단일 레버.** 승수 스캔 14 arm, outer −61.6 %, Task 13a **NO-GO**, Gate B 프로토콜 |
| [`CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md`](CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md) | 150 | Xe Anderson **채택**(정확도 근거), 폭 96/128 **기각**, FP32 +2.6 %, **배치 도착 폭 기아** 진단, 기본값 확정 |
| [`MASTER_vs_RASBERY_COMPARISON_20260824_KO.md`](MASTER_vs_RASBERY_COMPARISON_20260824_KO.md) | 165 | **Gate B 기준값**의 출처 |
| [`GPU_RASBERY_METHODOLOGY_BENCHMARK_20260824_KO.md`](GPU_RASBERY_METHODOLOGY_BENCHMARK_20260824_KO.md) | 235 | 방법론·벤치마크 전반 |
| [`CAMPAIGN_PHASE1_PHASE2_REPORT_20260826_KO.md`](CAMPAIGN_PHASE1_PHASE2_REPORT_20260826_KO.md) | 97 | Rev.4 Phase 0–2 결과 |

## 3. Task 10 — device outer 세그먼트에서 conditional WHILE까지

읽는 순서가 곧 구현 순서다. 각 문서가 **다음 문서가 닫은 구멍**을 열어 둔 채 끝난다.

| 문서 | 줄 | 무엇을 닫았나 |
|---|---:|---|
| [`TASK10_HOSTFREE_OUTER_20260830_KO.md`](TASK10_HOSTFREE_OUTER_20260830_KO.md) | 231 | **host-free outer** — 관측을 세그먼트당 1회로. `sync_pre_nodal` 12,017 → 504. 같이 움직여야 했던 네 writer. 남긴 질문: "capture 중 `cudaGraphLaunch`가 child node로 기록되는가" |
| [`TASK10_CONDITIONAL_WHILE_20260831_KO.md`](TASK10_CONDITIONAL_WHILE_20260831_KO.md) | 475 | **답은 "아니다"**, 그리고 같은 구멍이 **둘**이었다. `GpuGraphSplice.h`, 새로 드러난 전제 셋, **nodal 그래프 캐시**(재캡처 3,282 → 4) |
| [`TASK10_OUTER_WHILE_20260901_KO.md`](TASK10_OUTER_WHILE_20260901_KO.md) | 564 | **conditional WHILE.** 정지 규칙, pinned 스테이징, `exec_update_conditional` 스파이크, 캐시 미스가 여는 capture **셋**. in-body sync 11,937 → 1,038, overrun → **0**. **238 wall이 기본값을 정한다** |

## 4. 배치 처리량 — 리필 · 멀티GPU · 폭

| 문서 | 줄 | 내용 |
|---|---:|---|
| [`W4_LITE_REFILL_MULTIGPU_20260830_KO.md`](W4_LITE_REFILL_MULTIGPU_20260830_KO.md) | 304 | **즉시 슬롯 리필**(이미 동작하고 있었다 — 추가된 것은 **계약**), `--jobs` 매니페스트, 테넌시 감사, **멀티GPU dispatcher**(GPU당 프로세스 + flock 공유 큐) |
| [`W4_L5_MULTIPROC_PER_GPU_20260901_KO.md`](W4_L5_MULTIPROC_PER_GPU_20260901_KO.md) | 542 | **레버 L5 — GPU당 K 프로세스.** `src/`를 한 줄도 건드리지 않는 폭 레버. MPS 수명주기, VRAM 가드, **판정 지표는 `mean_width`가 아니라 `width_fill`** |
| [`IO_WRITER_THREAD_DESIGN_20260827_KO.md`](IO_WRITER_THREAD_DESIGN_20260827_KO.md) | 149 | **HDF5 writer 스레드.** byte-identity 논증, 실패 격리(poison 세션은 흡수성), 게이트 G0–G6. **채택 완료 — 기본값 `thread`** |

## 5. 서브시스템 이식 기록

| 문서 | 줄 | 대상 |
|---|---:|---|
| [`CMFD_GPU_ASSEMBLY_AND_DRIVE_FUSION_20260825_KO.md`](CMFD_GPU_ASSEMBLY_AND_DRIVE_FUSION_20260825_KO.md) | 655 | CMFD 연산자 조립 + BiCG drive 융합 |
| [`CMFD_FUSION_VALIDATION_20260825_KO.md`](CMFD_FUSION_VALIDATION_20260825_KO.md) | 87 | 위의 검증 |
| [`NODAL_GPU_DESIGN_20260821_KO.md`](NODAL_GPU_DESIGN_20260821_KO.md) | 63 | Nodal(SENM) 이식 정찰 |
| [`NODAL_GPU_REFACTOR_RECEIPT_20260824_KO.md`](NODAL_GPU_REFACTOR_RECEIPT_20260824_KO.md) | 66 | Nodal 리팩토링 영수증 |
| [`NODAL_REFACTOR_VALIDATION_BOTTLENECK_20260825_KO.md`](NODAL_REFACTOR_VALIDATION_BOTTLENECK_20260825_KO.md) | 123 | Nodal 검증 + 병목 분석 |
| [`GPU_NODAL_SUITABILITY_AND_SPEED_ROADMAP_20260824_KO.md`](GPU_NODAL_SUITABILITY_AND_SPEED_ROADMAP_20260824_KO.md) | 135 | 노달의 GPU 적합성 진단 |
| [`XSRECON_GPU_20260821_KO.md`](XSRECON_GPU_20260821_KO.md) | 129 | 평형 Xe / XS 재구성 이식 |
| [`FLATXS_GPU_20260821_KO.md`](FLATXS_GPU_20260821_KO.md) | 135 | `UpdateFlatXS`(분기 Horner) 이식 |
| [`cmfd_nodal_preconditioner_ko.md`](cmfd_nodal_preconditioner_ko.md) | 214 | CMFD·Nodal·전처리기 개념 (미니코어 예시) |

## 6. 배경 · 참조

| 문서 | 줄 | 내용 |
|---|---:|---|
| [`REV00_CODE_UNDERSTANDING_AND_CUDA_PORTING.md`](REV00_CODE_UNDERSTANDING_AND_CUDA_PORTING.md) | 1,155 | 캠페인 이전의 코드 이해 · CUDA 포팅 준비 |
| [`GPU_ACCEL_LITERATURE_SURVEY_20260821_KO.md`](GPU_ACCEL_LITERATURE_SURVEY_20260821_KO.md) | 127 | 노심해석 코드 GPU 가속 문헌조사 |
| [`CONSISTENCY_FINDINGS_20260821.md`](CONSISTENCY_FINDINGS_20260821.md) | 162 | 일관성 조사와 처분 (no-op env 플래그 3종 제거 근거 포함) |
| [`chiffon-rasbery-interface.md`](chiffon-rasbery-interface.md) / [`CHIFFON_manual.md`](CHIFFON_manual.md) | 238 / 238 | CHIFFON 단면 라이브러리 연결 구조 · 입력 매뉴얼 |
| [`rasbery-input-tolerance.md`](rasbery-input-tolerance.md) | 55 | 입력 수렴 설정 |
| [`REV00_MASTER_REFERENCE_PROVENANCE.md`](REV00_MASTER_REFERENCE_PROVENANCE.md) | 249 | MASTER 참조값 출처 |
| [`DEPLETION_PC_REFLECTOR_XS_SURVEY_20260822_KO.md`](DEPLETION_PC_REFLECTOR_XS_SURVEY_20260822_KO.md) | 32 | 감손 PC 스킴 · 반사체 XS 문헌조사 |
| [`PATCH_HARNESS_REAPPLY_WARNING.md`](PATCH_HARNESS_REAPPLY_WARNING.md) | 62 | ⚠ 이 트리에 `02_variants/patch_harness`를 재적용하지 말 것 |

## 7. 물리 · 이력 (캠페인 이전, 참고용)

`iisc-implementation-ko.md`, `history_simplification_ko.md`,
`Spectral and rod history correction.md`, `rod_cusping_rework_ko.md`,
`rod-depletion-prep.md`, `perf_and_cusping_fixes_ko.md`, `code_cleanup_summary_ko.md`

---

## 부록: 용어와 규약

### 게이트 등급

| 등급 | 주장 | 게이트 |
|---|---|---|
| **B0** | 이전 경로와 **비트가 같다** | feature-off byte 동일성 + 결정론 ×2 + `[TRAJECTORY] digest`. Gate A/B 불필요 |
| **N1** | 부동소수 **결합 순서**가 바뀐다 | Gate A(이동 **크기**) + Gate B(MASTER **쪽**인가) + 구성요소별 추가 게이트 |
| **A2** | 수렴 **도중** 무엇으로 수렴하는지가 바뀐다 | N1 전부 + 불변식 계약 + 별도 브랜치 + 명시적 롤백 |

### 번호 체계 — 섞지 말 것

- **Task 0–28 / Wave W0–W5+** — Rev.7.1 계획의 **구현 순서**
- **L1–L8** — GA evaluator 계획의 **레버 번호** (`W4_L5_…` 문서의 "L5"가 이것이다)
- **W1–W4** — GA evaluator 계획의 **자체 웨이브 축** (Rev.7.1의 W와 별개)

### 실행 규약

- **238 GPU0 전용.** 하네스는 자식에게 `CUDA_VISIBLE_DEVICES=0`을 강제한다.
- **타이밍과 텔레메트리를 섞지 말 것** — `RASBERY_STATEPOINT_TELEMETRY`는 배치에서 I/O
  경합을 증폭해 처리량을 왜곡한다.
- **동일성 판정에 `cmp`를 쓰지 말 것** — HDF5 object header가 생성/수정 시각을 기록한다.
  판정은 `h5diff`(데이터셋 단위)로만.
- **arm당 hot run ≥ 3회, warm-up 1회 제외, 교차 순서, median 보고.**
- **단일덱 arm과 배치 arm은 이제 env가 다르다.** `RASBERY_GPU_CMFD_BLOCK=64`(배치 −0.3 %)와
  `RASBERY_GPU_OUTER_SEGMENT_V2=1`(배치 −1.0 %)은 **단일덱 전용**이고,
  `RASBERY_CUDA_SYNC_MODE=blocking`은 **배치 옵션**이다(단일덱에서는 +8.7 % 느리다).
- **`RASBERY_XE_ANDERSON`은 모드 의존 기본값**(단일 ON / 배치 OFF)이다. 배치 arm에서
  켜려면 **반드시 명시적으로 export**하고, 빈 값은 OFF가 아니라 "요청 없음"이다.
- **골든 데이터셋 수**: 단일 500(v1/v2 기록) / 644(로컬 트리 실측) / 배치 M64 708 /
  writer 게이트 45,312. **어느 쪽을 세었는지가 문제다** — v3 동결이 238의 실제 수를 기록한다.
