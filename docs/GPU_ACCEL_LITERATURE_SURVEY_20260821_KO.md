# 원자로 노심해석 코드 GPU 가속 최신 연구 동향 조사 (2026-08-21)

조사 범위: 2011–2026 공개 문헌 (KNS/M&C/PHYSOR 학회, PNE/ANE/NET/NED/NSE/CPC 저널).
모든 인용은 웹에서 서지 실존을 확인한 문헌이며, 확인 불가 항목은 "추정"으로 명시.
KNS 전문 3편은 원문 PDF를 직접 열람하여 수치 검증.

## 1. Nodal 확산/SP3 솔버의 GPU 가속

### 1.1 VANGARD (서울대, SP3 SENM 핀단위 nodal) — 가장 직접적인 벤치마크 대상

문헌 계보 (모두 실존 확인):
- Jeon, Choi, Joo, "Feasibility of Fast Pinwise Core Simulation Using GPUs," Trans. KNS Autumn 2020 (원문 열람)
- Jeon, Hong, Choi, Joo, "GPU Acceleration of the Prototype Pinwise Core Analysis Code VANGARD," M&C 2021, DOI 10.13182/M&C21-33727
- Jeon, Hong, Choi, Joo, "Methods and performance of a GPU-based pinwise two-step nodal code VANGARD," Progress in Nuclear Energy 156, 104528 (2023)
- Jeon, Kochunas, "Verification and Validation of the VANGARD Pinwise Nodal Code with Analyses of BEAVRS," NSE 200 sup1 (2024)

방법론 핵심 (KNS 2020 원문 기준):
- 커널 분해: one-node SP3 SENM 커널, **스레드 = (셀, 에너지군)**, 군 Jacobi + 셀 red-black ordering — 2-node problem 병렬이 아니라 one-node 정식화로 2-node 문제 자체를 회피.
- 차수 하이브리드: 반경 2차 + 축방향 4차 전개.
- 혼합정밀도: 반경 FP32 / 축방향 산술 FP64 + 쌍곡함수만 FP32(SFU). 완전 FP32는 축방향 κ 대에서 발산 보고. 혼합은 완전 FP32 대비 15% 손실.
- 전송: 초기 이식에서 outgoing current D2H가 nodal 시간의 40% → **CMFD 결합계수(D̂)를 GPU에서 직접 계산해 복사 제거**.
- PNE 2023: 적응형 저차 전개, two-level 표면류 보정, XS 군정수 압축 후 GPU 전량 상주, CRAM 대량병렬 감손.

정량: NEACRP 전노심에서 nodal 커널 기준 RTX 2080 Ti 1장 = Xeon E5-2630 v4 1코어 대비 210–213×, 20코어 대비 19.5–19.9× (전체 시간 9.5×, T/H CPU 잔류). CPU-GPU CBC 완전 일치, 핀출력 최대 상대차 <0.005% (비트 동일은 아님). PNE 2023: APR1400/AP1000 3D 핀단위 주기감손을 소비자 GPU 1장으로 3분 이내.

시사점: RASBERY SENM 커널과 동일 계열의 검증된 선행. 혼합정밀도는 비트 동일성 요구와 충돌 → 속도 참고, 정밀도 전략 불채택.

### 1.2 Cuda_NEM (브라질 COPPE/UFRJ, 4차 NEM)

- Mendes, Heimlich, de Lima, Silva, NED 416, 112751 (2023/2024). 4차 NEM + 2차 횡누설 CUDA 병렬화, CNFR 대비 6–8×. 비교 기준 CPU 사양 미확정(단일코어로 추정 — 부분 추정), 일부 GT 1030 기준 완만한 이득. 커널 융합·메모리 배치 최적화 부재 시의 하한선 사례.

### 1.3 기타

- SCOPE2 GPU (일본 NFI): Kodama, Tatsumi, Ohoka, M&C 2011 — 핀단위 SP3 최초기 GPU 탐색.
- STORK (중국): Yu Lulin 등, 原子能科学技术 58(3), 662-671 (2024) — 층별 2D MOC(69군) + 3D pin-by-pin SP3 멀티 GPU. C5G7/VERA 검증. CPU 대비 speedup 미공개.
- KOMODO/ADPRES, PARCS: GPU 이식 공개 문헌 없음 — 오픈소스 nodal GPU화는 공백 영역.
- NuDEAL (SNU/미시간/KAERI, arXiv 2026): HFEM 응답행렬 red-black 독립 해법, cuBLAS batched LU(cublasDgetrfBatched). RTX 4070 Ti 1장이 Griffin 96코어와 경쟁(C5G7 3D).

## 2. CMFD/저차 가속의 GPU화

### 2.1 Song et al. (하얼빈공정대, Frontiers in Energy Research 8:124, 2020) — GPU 선형해법 직접 비교

- MG CMFD→FG CMFD two-level 구조 전체 GPU 상주. 선형해법 3종 비교: Jacobi(병렬성 최상·수렴 최악), **red-black SOR(균형·최종 채택)**, preconditioned GMRES(수렴 최상·GPU 오버헤드 큼). 균질화 셀당 1스레드, 조립 행당 1스레드, 전 구간 FP64.
- GTX 1080 Ti vs i9-7900X: CMFD 가속 반복수 ~100×↓, GPU-MOC ~25×, GPU-CMFD(SOR) ~13×, 조합 ~2,400× (비가속 CPU-MOC 대비).
- 시사점: **조대메쉬 CMFD 규모에선 Krylov보다 red-black SOR이 GPU 총비용 우위** — BiCG 아레나와 A/B 벤치 후보.

### 2.2 STREAM3D-GPU (UNIST) — "CPU-GPU 결과 동일성" 철학

- Dzianisau, Lee, Trans. KNS Spring 2024 (원문 열람); CPC 109952 (2025); NSE 2025 (DOI 10.1080/00295639.2025.2597646).
- OpenACC, 축방향 평면 분해로 평면별 최소 데이터만 비동기 왕복. 2×EPYC 7543(64코어) vs 8×RTX A5000: C5G7 MOC 40.2×/전체 17.0×; OPR-1000 72군+T/H MOC 15.4×/전체 19.0×. **CPU-GPU k_eff 완전 동일(동일 알고리즘·동일 연산순서 유지 명시)** — RASBERY 비트 동일성 전략과 같은 철학. NSE 2025: CMFD 실행시간 최대 22×↓, BEAVRS 감손 5 pcm 이내. CMFD 내부 해법 종류 미확인.

### 2.3 nTRACER GPU (서울대)

- Choi, Kang, Joo, KNS Autumn 2018; Choi, Kang, Lee, Joo, PNE 133, 103631 (2021).
- MOC·CMFD·1D 축방향 GPU 이식, 혼합정밀도 광범위(소비자 GPU FP32 우위 활용 명시), Jacobi 군 스위핑, CPU-GPU 동시성. VERA #5: 16코어 대비 10–13×, ray tracing 한정 ~70× (RTX 2080 Ti).

### 2.4 Batched 선형해법·multi-state — RASBERY 아레나의 최근접 선행

- **nTRACER batched CRAM** — Lee, Jae, Choi, Kang, Joo, Trans. KNS Spring 2020 (원문 열람): 수십만 개 동일 희소패턴(비영 3,411) 소형계를 **시스템당 1스레드 batched BiCGSTAB+Jacobi**로. **인덱스 벡터는 constant memory에 1부(패턴 공유), 원소는 non-zero-major(같은 비영위치의 전 시스템 원소 연속 배치)로 완전 코얼레싱**. CRAM 특성상 반복수 균일 → warp divergence 소멸.
- OpenMOC 기반 SMR 3D MOC — Qufei Song 외(SJTU), PNE 192, 106109 (2025/2026): CMFD GPU 구현, 트랙 batch·on-the-fly 축방향 ray tracing으로 메모리 98.5%↓, flat-source 단일스레드 CPU 대비 최대 1015.8×, NuScale 전노심 단일 GPU.
- 다중 노심상태(LP) 동시 GPU 평가: 제목만 확인된 문헌 1건(서지 상세 미확인 — 추정). **64-인스턴스급 배치 nodal 시뮬레이션 공개 사례는 사실상 공백.**

### 2.5 CUDA graph / Wielandt / red-black

- CUDA graph: 원자로 코드 적용 공개 문헌 미발견 → RASBERY의 스윕 그래프는 문헌상 신규성.
- Wielandt: GPU 특화 처리 문헌 미발견(CPU 이론연구만: 공간의존 Wielandt 반복수 46%↓, NSE 2017 미시간 계열).
- Red-black: VANGARD(셀)·Song(SOR)·NuDEAL(응답행렬) — GPU nodal/CMFD 표준 패턴으로 수렴.

## 3. 단면(XS) feedback 처리의 GPU화

### 3.1 nTRACER XS 전면 오프로드 (KNS 2020 원문 열람) — 전송 최소화의 정석

- 주기감손에서 XS 루틴 CPU 잔류 시 전체의 ~60% 점유 규명 → 전면 이식 결론.
- 중첩 AoS를 단일 연속 1D 배열+변위 벡터로 SoA화. (영역,군)당 1스레드.
- APR1400 2D 쿼터코어 주기감손 8h(20코어) → 1.5h (RTX 2080 Ti 1장, GPU 메모리 8.5GB). XS 루틴 시간 CPU 대비 15% 이하로.
- 후속: Lee, Kim, Joo, PNE 165, 104928 (2023).

### 3.2 VANGARD XS 압축·디바이스 상주 (PNE 2023)

- branch/history 군정수 압축 후 GPU 전량 상주, 매크로 XS 재구성·CRAM·중성자속 전부 온디바이스 — **주기 계산 중 CPU-GPU 대량 전송이 없는 구조**가 3분 주기감손의 전제.
- **Horner 다항 XS 재구성의 GPU 구현을 명시한 문헌은 미발견** — 문헌은 테이블 보간 계열 지배적. CHIFFON식 Horner 재구성 GPU화는 공백 = RASBERY의 차별점(논문화 가치).

### 3.3 반례·보조

- RAST-K FR 감손 GPU (Dzianisau & Lee, KNS Spring 2020, 원문 열람): CRAM 행렬연산만 GPU, 역행렬 CPU 잔류 → 매 스텝 왕복 전송, 스텝당 18% 개선에 그침. **부분 오프로드+빈번 전송이 실패하는 전형 반례.**
- Heimlich 계열: ANE 91 (2016) 순차 대비 >200×; ANE 118 (2018).
- PRAGMA (서울대 GPU MC): ANE 160 (2021); NET 55 (2023) — 연속에너지 XS 포함 전 데이터 디바이스 상주 원칙. 24장 소비자 GPU로 100억 히스토리/15분급.
- Shift (ORNL): Hamilton, Evans, ANE 128 (2019) — CE XS GPU 상주(ExaSMR). MPACT 생산급 GPU 포트 공개 문헌 미발견(추정).

## 4. 교차 관찰

1. **정밀도**: 소비자 GPU 시대엔 FP32/혼합 지배적(nTRACER·VANGARD); 재현성 중시 코드(STREAM3D-GPU 등)는 FP64+동일 알고리즘으로 CPU-GPU 완전 일치. 2군 nodal+CMFD는 연산량이 작아 RTX PRO 6000의 낮은 FP64 비율이 병목이 아닐 가능성 큼(실측: RASBERY 커널 합 3.5s/57.5s — 메모리·지연 바운드).
2. **전송**: 성공 공통분모 = 디바이스 완전 상주 + D̂·환산까지 온디바이스. 실패 공통분모 = 반복 루프 내 부분 왕복(RAST-K, VANGARD 초기 40%).
3. **비교 기준**: 단일코어 대비(210×, 1015×)와 다코어 대비(10–20×)가 혼재 — 인용 시 반드시 병기.

## 5. RASBERY 벤치마킹 권고 (우선순위)

| # | 서브시스템 | 차용 기법 | 출처 |
|---|---|---|---|
| 1 | CMFD BiCG 배치 아레나 | 고정 희소패턴 공유: 인덱스 1부 constant/read-only cache, 원소 instance-major 배치로 64 인스턴스 코얼레싱, 반복수 균일화 | nTRACER batched CRAM (KNS 2020) |
| 2 | SENM nodal 커널 | (노드,군)당 1스레드 + 군 Jacobi + red-black; κ·sinh/cosh·계수 레지스터 지역화; **D̂/결합계수 온디바이스 계산으로 D2H 제거** | VANGARD KNS 2020/PNE 2023 |
| 3 | CMFD 해법 대안 | 2군 조대메쉬에선 batched red-black SOR이 Krylov를 이길 수 있음 — A/B 벤치(둘 다 고정 반복순서라 비트 동일 유지 가능) | Song et al. 2020 |
| 4 | 비트 동일성 규약 | "동일 알고리즘·동일 연산순서 → k_eff 완전 동일" 검증 프로토콜, reduction 순서 고정. FP32/혼합 불채택 | STREAM3D-GPU; VANGARD |
| 5 | CHIFFON XS Horner | 계수 압축 후 전량 상주, (노드,군)당 1스레드 Horner, SoA+변위 평탄화, 루프 내 왕복 금지 | VANGARD PNE 2023 + nTRACER (반례: RAST-K) |
| 6 | 64 인스턴스 배치 | 다중 상태 동시 GPU 평가·CUDA graph 적용 모두 문헌 공백 → **아레나+그래프 조합은 신규성 주장 가능** | §4 공백 분석 |

요약: ① nTRACER batched 레이아웃을 아레나에, ② VANGARD 커널 구조로 SENM 이식(D̂ 온디바이스까지), ③ red-black SOR 대조군, ④ STREAM3D식 동일-연산순서 규약의 회귀 테스트화.

### 확인 불가/추정 명시
Cuda_NEM 비교 기준 CPU 사양, STREAM3D-GPU NSE 2025의 CMFD 내부 해법, MPACT 생산급 GPU 포트 부재 판단, LP 최적화 GPU 논문 서지 — 4건은 추정/미확정.

### 주요 출처 링크
VANGARD PNE: https://www.sciencedirect.com/science/article/abs/pii/S0149197022004024
VANGARD KNS 2020: https://www.kns.org/files/pre_paper/44/20A-199-전서윤.pdf
nTRACER PNE 2021: https://www.sciencedirect.com/science/article/abs/pii/S0149197021000032
nTRACER 감손 KNS 2020: https://www.kns.org/files/pre_paper/43/20S-179-이한규.pdf
Song 2020: https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2020.00124/full
STREAM3D-GPU KNS 2024: https://www.kns.org/files/pre_paper/51/24S-051-Siarhei.pdf
STREAM3D-GPU CMFD NSE 2025: https://www.tandfonline.com/doi/full/10.1080/00295639.2025.2597646
Cuda_NEM NED: https://www.sciencedirect.com/science/article/abs/pii/S0029549323006003
SMR GPU-3D-MOC PNE: https://www.sciencedirect.com/science/article/abs/pii/S0149197025005074
STORK 2024: https://yznkxjs.xml-journal.net/en/article/doi/10.7538/yzk.2023.youxian.0657
Shift ANE 2019: https://doi.org/10.1016/j.anucene.2019.01.012
PRAGMA NET 2023: https://www.sciencedirect.com/science/article/pii/S1738573323001985
스케일러블 GPU 감손 PNE 2023: https://doi.org/10.1016/j.pnucene.2023.104928
RAST-K GPU KNS 2020: https://www.kns.org/files/pre_paper/43/20S-398-Siarhei-Dzianisau.pdf
NuDEAL arXiv 2026: https://arxiv.org/html/2607.01591
