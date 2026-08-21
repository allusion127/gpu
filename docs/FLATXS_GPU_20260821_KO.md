# UpdateFlatXS(분기 Horner) GPU 이식 — 병목 재점검·비트 동일 검증·성능 판정 (2026-08-21)

브랜치 `codex/hotpath-opt` (xsrecon 체인 `3a3dc15` 위). 실행 장비 GPU238 GPU0
(RTX PRO 6000 Blackwell, 논리 CPU 24). 스크리닝(GA) 없는 순수 계산 속도
캠페인의 3번 지렛대(§4.2 of XSRECON_GPU_20260821_KO.md) 실행 기록.

## 1. 병목 재점검 (착수 전 실측, M64 프로파일 덱)

xsrecon+CMFD스윕+핀 고정 이후(벽시계 63.17 s)의 XS phase 스레드합:

| 블록 | CPU팔 스레드합 | 비고 |
|---|---|---|
| eqxe (Xe 재구성) | 31.4 s | 이미 GPU (nodes 38.56M) |
| **flatxs** | **321.4 s** | **남은 지배 블록. 전부 unrodded** |
| set_boron | 163.5 s | SetBoron = bppm 대입 + UpdateFlatXS 호출뿐 → 사실상 flatxs 중복 계상 |
| update_th | 140.0 s | 내부 UpdateFlatXS 포함; TH 솔브 자체는 ~6 s |
| update_burnup | 0 | 이 덱은 감손 상태점 없음 |

**판정: set_boron/update_th 버킷은 내부 UpdateFlatXS를 포함하므로, flatxs
하나가 남은 XS 측 CPU 비용의 사실상 전부다.** 포화 상태: GPU sm 평균 43.6%,
CPU 15.6/24 — 양쪽 미포화 = 지연 한계 구조 지속.

## 2. 무엇을 이식했나

`XSSet::UpdateFlatXS`의 unrodded 노드 경로 전체(`UpdateUnroddedNodeXS`) —
참조 gather → 분기/스펙트럴 히스토리 델타의 Horner 적용 → 경수 동위원소
갱신 → SoA scatter → 거시 XS 재조립 — 를 하나의 CUDA 커널로
(`RASBERY_GPU_FLATXS`, 기본 off, 실패 시 CPU 폴백).

**설계의 핵심 = 호스트 resolve / 디바이스 apply 분리.** 스펙트럴 히스토리
좌표 계산은 `std::log`를 쓰는데 glibc log는 correctly-rounded가 아니고
CUDA와 다르므로, 좌표를 디바이스에서 계산하는 설계는 검증 불가능하다.
그래서:

- **호스트**: `BuildFlatXsStream`이 노드별 applyDelta 호출 전부를 CPU 루프와
  동일한 순서의 (did, x, scale) 평탄 스트림으로 resolve. 히스토리 resolve는
  CPU 팔이 쓰는 바로 그 `ResolveSpectralHistoryDeltas`를 호출(코드 공유 =
  분기 원천 차단). `NodeSpectralIndex`가 읽는 워크스페이스 2원소(XSAF
  Pu-239/B-10 열군)는 분기 프리픽스를 채굴 형태로 2원소만 재생하는 프로브
  (`flatxsProbeMicElement`)로 공급.
- **디바이스**: `FlatXsKernel.h` 공유 본체(스레드=노드)가 스트림을 적용.
  CPU가 느리던 실제 원인(노드 고정 시 SoA stride 접근의 캐시라인 8배 증폭)이
  coalesce로 구조 소멸 — xsrecon과 동일 논리.

### 2.1 상주·공유 설계 (전송량이 성능의 전부)

| 데이터 | 정책 |
|---|---|
| 계수 테이블(`_lib_coeff_*`, ~7 MB) | **프로세스 전역 1회 업로드**, 내용 FNV 해시로 64 인스턴스 공유 |
| 참조 블록(`_ref_lmpx/_ref_micx`, ~60 MB/인스턴스) | 인스턴스 상주, `_ref_generation`(PrecomputeBranchCoefficients가 유일한 재작성자) 변경 시만 재업로드 |
| 라이브 `_micx/_lmpx` | **xsrecon 백엔드와 같은 디바이스 블록 공유.** flatxs가 생산 후 전체 다운로드 → 호스트=디바이스 비트 동일 → **세대 카운터를 선반영해 다음 xsrecon 호출의 ~70 MB 재업로드 소거** (rodded 노드가 뒤따르면 무효화) |
| xs·iden | 호출당 whole 업로드(타깃 외 열 왕복 보존), 다운로드는 xs 전체 + iden H-1/B-10/O-16 3행(레지스트리상 0..2 연속) |
| 스트림/노드 목록 | 호출당 업로드, grow-only 버퍼 |

rodded 노드는 CPU 경로 유지(다운로드 **후** 실행 — 전체 배열 다운로드가
rodded 열을 덮어쓰지 못하게 순서 고정), 그 호출은 세대 선반영을 포기한다.

## 3. 비트 동일 검증 — 형태 채굴 방법론 재사용

xsrecon의 교훈(소스 인용은 축약 형태의 기준이 될 수 없다) 그대로, 검증은
프로덕션 캡처 기반:

1. `RASBERY_FLATXS_DUMP=<p>` — 첫 all-unrodded 호출의 전체 입력(+resolve된
   스트림, 131 MB)과 CPU 팔 출력(61 MB) 캡처. 캡처 시 CPU 강제(출력이 곧
   ground truth).
2. `test/flatxs_replay.cpp` — 캡처 위에서 공유 본체 리플레이 + **1024개
   축약 마스크 전수 스윕**(10 사이트: 4 Horner + 4 accumulate + 거시 2).
3. `test/flatxs_device_replay.cu` — 같은 캡처를 --fmad=false 디바이스
   빌드로, 요소별 ULP.

**결과**: 채굴 마스크 = **0x3FF(전 사이트 융합)** — xsrecon과 달리 이
함수에선 gcc가 전부 융합했다. 스윕 정확일치 집합 {0x3F1..0x3FF 홀수}에서
비트 1/2/3(lmp accumulate, lmp-scatter Horner/accumulate)은 이 라이브러리가
행사하지 않아 **미구속**(다른 라이브러리에선 스윕 재실행 필요). 게이트:

| 게이트 | 결과 |
|---|---|
| 호스트 리플레이 (8451 노드, 스트림 18,075) | **0 ULP PASS** |
| 디바이스 리플레이 (--fmad=false) | **0 ULP PASS** |
| M1 전체 덱 A/B (24 flatxs 호출 전 GPU, nodes 202,824) | **h5 차이 57 B = CPU-vs-CPU 대조군과 동일 오프셋의 시각기록뿐** |
| M64 물리 (64 덱) | 기준 대비 대차이 파일 0 |

함정 기록: 디바이스 리플레이 최초 FAIL(연료노드 25% 대차이)은 커널이 아니라
**테스트 도구가 디바이스 뷰에 `has_coeff_micx`를 복사하지 않아** micx 델타가
전부 스킵된 것 — `--one <idx> <max_deltas>` 스트림 이분법으로 국소화했다.
프로덕션 백엔드는 뷰를 통째로 복사하므로 해당 없음.

## 4. 성능 판정 (M64 프로파일 덱, 24코어 호스트)

| 측정 | 이전 (xsrecon+스윕+핀) | flatxs GPU 추가 | 비고 |
|---|---|---|---|
| flatxs 스레드합 | 321.4 s | **24.2 s = 13.3×** | 전량 GPU (12,980,736 노드) |
| set_boron / update_th | 163.5 / 140.0 s | 11.1 / 7.9 s | 내부 flatxs 소거 효과 |
| eqxe (GPU) | 31.4 s | 19.4 s | 세대 선반영으로 micx 재업로드 소거 |
| **XS 4블록 합계** | ~656 s | **~62.6 s** | CPU에서 XS 계산이 사실상 소멸 |
| **벽시계** | **63.17 s** | **59.73 s (−5.4%)** | 아래 §4.1 |
| GPU sm 평균/피크 | 43.6% / 67% | 46.4% / 78% | |
| CPU | 15.6 / 24 | 14.1 / 24 | |

누적(단일상태 M64): 87.0 → 59.73 s = **1.457×**, 전 단계 물리 비트 동일.

### 4.1 벽시계가 5.4%만 움직인 이유 — 지연 한계 재확인

XS 스레드합 ~590 s를 지웠는데 벽시계는 3.4 s만 줄었다. GPU 46%·CPU 14/24
양쪽 미포화 그대로 — 아낀 CPU 시간이 outer 반복당 왕복 버블로 흡수되는
구조(xsrecon 때와 동일 진단)가 유지된다. **이제 CPU 측 XS 블록이 소멸했으므로
남은 벽시계의 주인은 Nodal(호스트 단일스레드)·CMFD 호스트 구간·왕복
버블이고, 순위 1 지렛대 "outer 루프 디바이스 상주화"의 선행 조건(_xs가
디바이스에서 생산됨)이 이번 이식으로 마련되었다** — flatxs가 이미 디바이스
`_xs`를 생산하므로, CMFD가 그것을 직접 소비하면 호출당 xs 왕복이 사라진다.

## 5. 잔여·주의

- **rodded 덱 미검증**: 프로파일 덱은 rodded 0. CY02 T4/T6 LP처럼 rodded
  상태가 있는 덱에서 A/B(분할 경로 + 세대 무효화) 확인 후 프로덕션 적용.
- **partial-node 호출**(options.nodes) 경로는 구현되어 있으나 이 덱에선
  전부 full 호출이라 미행사.
- 스윕 미구속 비트 1/2/3 — lmp-scatter 델타를 행사하는 라이브러리에선 캡처
  후 스윕 재실행.
- CY02 51상태 실전 워크로드 재측정(현재 수치는 단일상태 M64).

## 6. 재현 명령

```bash
# 캡처 → 리플레이 → 스윕 → 디바이스 리플레이
RASBERY_FLATXS_DUMP=$PWD/fx ./RASBERY --batch-mode 1 --rasi deck.json --raso o.h5
./rasbery_flatxs_replay $PWD/fx            # PASS 필수
./rasbery_flatxs_replay $PWD/fx --sweep    # 마스크 재채굴
CUDA_VISIBLE_DEVICES=0 ./rasbery_flatxs_device_replay $PWD/fx
# 전체 덱 A/B (h5 차이 = CPU-vs-CPU 대조군 시각기록 바이트 수와 동일해야 합격)
RASBERY_GPU=1 RASBERY_GPU_CMFD_SWEEP=1 ./RASBERY ... --raso cpu.h5
RASBERY_GPU=1 RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_FLATXS=1 RASBERY_GPU_XSRECON=1 \
  ./RASBERY ... --raso gpu.h5
cmp -l cpu.h5 gpu.h5 | wc -l
```
