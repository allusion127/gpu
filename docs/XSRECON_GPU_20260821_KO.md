# 평형 Xe/XS 재구성 GPU 이식 — 구현·비트 동일 검증·성능 판정 (2026-08-21)

브랜치 `codex/hotpath-opt` (canon `6cf39f1` 위, 커밋 체인 `be6cfc2` → `560ed0d` →
`6ce4bb7` → `e53f8fa` → `0d92085` → 본 문서 커밋). 실행 장비 GPU238 GPU0
(RTX PRO 6000 Blackwell, 논리 CPU 24), 빌드 레시피 = attestation_20260821.json
attempt_2 + `-DRASBERY_ENABLE_TESTS=ON`.

## 1. 무엇을 이식했나

`XSSet::UpdateEquilibriumXenon`의 연료 노드 루프 전체 — 미시 XS 1군 condense,
Xe-135 평형 덮어쓰기, 감쇠 적용, `ReconstructNode` — 를 하나의 융합 CUDA 커널로
(`RASBERY_GPU_XSRECON`, 기본 off, 실패 시 stderr 경고 1회 후 CPU 폴백).

- 공유 본체 `src/XsReconKernel.h`: 호스트 빌드가 곧 기준. 포인터 뷰(BatchView)
  경계라 배열이 device 상주/생산으로 바뀌어도 본체 무변경.
- 백엔드 `src/CudaXsReconBackend.{h,cu}` + 스텁: 감손 탐사 하니스 관례 계승
  (인스턴스당 스트림, grow-only 블록, `RASBERY_CUDA_TRY`, G0 영수증
  `[RASBERY][XSRECON][GPU] {"nodes_solved":N}`).
- `_micx`/`_lmpx`는 device 상주, 호스트 재작성 시에만 세대 카운터로 재업로드
  (`XSSet::_micx_generation`, `UpdateFlatXS`/`Update`가 bump; cusping은 `_xs`만
  블렌드하므로 무관). 호출당 전송: phif/iden/xs 업로드, xs+iden 3행 다운로드.
- CPU 스레드=노드 매핑으로 SoA 접근이 전부 coalesce — CPU가 느린 실제 원인이던
  "노드 고정 시 stride 접근의 캐시라인 8배 증폭"이 구조적으로 소멸. warp
  divergence는 이 커널의 요인이 아님(분기 인벤토리: 재질 분기는 정적 연료 목록
  compaction으로 소거, 나머지는 균일 가드 뿐).

## 2. 계측이 먼저다 — RASBERY_XS_TIMING (R8 해소)

M1 영수증의 "depletion 39.34%"가 라벨 오류였던 전례(감손 프로브 R8) 때문에,
이식 전에 독립 phase 계측을 shipped binary에 넣었다(`src/XSTiming.h`,
버킷: eqxe/condense/recon, flatxs/rodded/unrodded, update_th, set_boron,
update_burnup; OFF 시 시계 미접촉). 프로파일 덱 실측으로 eqxe가 실제 지배
블록임을 확인한 뒤 착수했다.

## 3. 비트 동일 검증 — 세 층, 그리고 핵심 발견

**최종 상태: `RASBERY_GPU_XSRECON=1`로 전체 덱(824 outer, TH 10회, 보론 12회,
Xe 100호출)이 CPU 팔과 HDF5 바이트 동일**(실행 시각 기록 114 B 제외 — CPU끼리
반복 실행 대조군과 동일 수준). 호출별 출력 해시 100×13 성분 전부 일치.

### 3.1 핵심 발견: 소스 인용은 기준이 될 수 없다

gcc `-ffp-contract=fast`(기본)는 **같은 문장을 번역 단위마다 다르게 융합하고,
단일 사용 임시변수를 가로질러도 융합한다.** 합성 데이터 하니스가 세 컴파일러
(WSL g++13, conda g++14.3, nvcc 13.0)에서 전부 PASS인데 실덱은 1 ULP씩
갈라졌던 이유다. 프로덕션 `XSSet.cpp`의 실제 축약 형태는 캡처에서 채굴해야
했다:

| 표현식 | 프로덕션 gcc의 실제 형태 | 근거 |
|---|---|---|
| condense `sum += mic·φ` | **비융합** (곱 반올림 후 덧셈) | form_probe c=0: I135 6025/6025 |
| fissSource `+= fRate·dep` | **비융합** | form_probe f=0 |
| Xeeq 분자 `λI·Ieq + fissXe` | **FMA** | form_probe n bit0 |
| Xeeq 분모 `λXe + sigaXe` | **FMA — `sigaXe` 임시변수를 가로질러 `fma(cond,Σφ,λXe)`로 융합** | form_probe n bit1 (6025/6025 완성 조합) |
| ReconstructNode `val += mic·iden` (스칼라+산란) | **FMA** | form_probe r=1: 12050/12050 |

커널 고정 방법: FMA 자리는 `xsrFma`(양쪽 하드웨어 FMA), 비융합 자리는
`xsrMul`(device는 `--fmad=false`로, host는 asm 배리어로 재융합 차단 — 명명
임시변수만으로는 배리어가 아니다).

### 3.2 검증 도구 (전부 env 게이트, 기본 무비용)

1. `RASBERY_XSRECON_DEBUG_HASH` — 호출별 FNV 해시 13성분. 최초 분기 호출과
   갈라진 배열을 특정(전부 call=1, iden I135/Xe135m 행 기원으로 국소화).
2. `RASBERY_XSRECON_DUMP=<p>` — 첫 Xe 호출의 입출력 전체 캡처(75 MB).
   **한계: 첫 호출만. 감쇠(relax<1) 호출 캡처엔 호출 선택 기능 추가 필요.**
3. `test/xsrecon_replay.cpp` — 캡처 위에서 공유 본체 리플레이, 프로덕션 출력과
   요소별 ULP. 최종 PASS.
4. `test/xsrecon_form_probe.cpp` — 융합/비융합 전 조합 전수 채점. §3.1의 표가
   이 도구의 출력이다.
5. 합성 하니스(host/device) — 기준을 §3.1 형태로 고정 후 PASS. 인용-기준의
   불충실성이 판명되었으므로 형태의 진실원은 캡처 프로브다(하니스 주석에 명시).

### 3.3 미결

- **감쇠 경로(relax<1)의 형태는 캡처 미검증** — 이 덱은 감쇠 미발생. 진동 덱
  (iSMR CY03/04류)에서 캡처 후 프로브 확장 필요. 현재는 하니스 정합 FMA 형태.
- MSVC 호스트 경로의 `xsrMul`은 volatile 폴백(성능 무관, Windows 개발 빌드용).

## 4. 성능 판정 (24코어 호스트 기준 — 64코어 호스트와 다를 수 있음)

| 측정 | CPU 팔 | GPU 팔 | 비고 |
|---|---|---|---|
| M1 eqxe (100호출) | 0.312 s | 0.355 s | M1은 24코어가 한 인스턴스에 전부 붙어 CPU가 이김. 호출당 ~3 MB 전송이 상쇄 |
| M64 eqxe (6,400호출) | **559.1 s** | **35.0 s = 16.0×** | oversubscription에서 호출자 스레드 관점 대기 포함 |
| M64 벽시계 | 1:27.00 | **1:27.00 — 동일** | 아래 §4.1 |
| M64 물리 | — | 64/64 출력 시각기록 수준 차이뿐 | 비트 동일 유지 |
| M64 CPU 사용 | 1712% (17.1/24코어) | 1261% (12.6/24) | 양팔 다 미포화 |

### 4.1 벽시계가 안 움직인 이유 — 지연 한계(latency-bound)

- GPU 사용률 실측 평균 ~20%, 피크 ~32% → **용량 한계 아님**.
- CPU도 미포화 → **CPU 한계 아님**.
- 남는 것은 **outer 반복당 왕복 지연**: 인스턴스당 86.7 s / 824 outer ≈ 105 ms.
  CMFD 아레나 배치 형성(도착 간격 EWMA 7.8~11.3 ms) + 스트림 동기화(런치당) +
  호스트 직렬 구간이 파이프라인 버블을 만들고, eqxe를 16× 줄여도 그 시간이
  버블로 흡수된다. 부수 관찰: xsrecon이 인스턴스 도착을 동기화해 아레나 평균
  폭이 10.8→18.6으로 커졌다(런치 21,760→12,645).
- `RASBERY_BATCH_WAIT_US` 스윕(0/2000/8000 → 폭 18.6/35.4/46.1)은 벽시계를
  87→93→105 s로 **악화**시켰다: GPU가 놀고 있으므로 폭을 키우는 대기는 지연만
  추가한다. **NO-GO** (물리는 전 스윕에서 동일 — 배치 폭의 비트 안전성 실측
  확인이라는 부산물).

### 4.2 다음 지렛대 (순위)

1. **outer 루프의 디바이스 상주화(Phase 2)**: `_xs`를 device에 남겨 CMFD가
   직접 소비(호출당 xs 왕복 1.6 MB 소거), 나아가 nodal coupling까지 device
   잔류(§5.4(A) rank 1·3의 원래 조건). BatchView 경계는 이를 위해 설계됨.
   지연 한계인 지금, 왕복 자체를 없애는 것이 유일하게 벽시계를 움직인다.
2. **동기점 축소**: xsrecon solve의 per-call `cudaStreamSynchronize`를 CMFD
   아레나 흐름과 겹치기(이벤트 기반).
3. UpdateFlatXS(branch Horner) 커널화 — flatxs 310 s(M64 스레드합)가 다음 큰
   CPU 블록. §3의 형태 채굴 방법론 재사용.
4. 64코어 호스트에서 재측정 — 이 호스트의 24코어 결론이 그대로 이전되지 않는다.

## 5. 재현 명령

```bash
# 게이트 일괄 (build 디렉터리에서)
./rasbery_xsrecon_consistency 8451
CUDA_VISIBLE_DEVICES=<GPU0-UUID> ./rasbery_xsrecon_device_consistency 4096
# 캡처 → 리플레이 → 형태 프로브
RASBERY_XSRECON_DUMP=$PWD/cap ./RASBERY --rasi deck.json --raso out.h5
./replay $PWD/cap && ./form_probe $PWD/cap
# 전체 덱 A/B (해시 스트림 diff 0 + h5 시각기록 차이만이 합격)
RASBERY_XSRECON_DEBUG_HASH=1 ./RASBERY --rasi deck.json --raso a.h5 2> h_cpu.log
RASBERY_XSRECON_DEBUG_HASH=1 RASBERY_GPU_XSRECON=1 ./RASBERY --rasi deck.json --raso b.h5 2> h_gpu.log
diff h_cpu.log h_gpu.log && cmp -l a.h5 b.h5 | wc -l
```
