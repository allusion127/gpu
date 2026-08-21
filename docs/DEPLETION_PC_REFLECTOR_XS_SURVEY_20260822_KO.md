# 감손 PC 스킴·반사체 XS 생성 방법론 문헌조사 (2026-08-22)

웹 검증 문헌만 인용(불확실 항목 "추정" 명시). 대상: RASBERY(2군 nodal, CRAM8 + 고전 PC)와 반사체 XS 신버전(PROMARX식 1D 스펙트럼 기하).

## A. 감손 Predictor-Corrector

핵심 문헌 계보:
- Isotalo & Aarnio, ANE 38 (2011) ×2 — LE/QI(수송해 추가 없이 시간차수 상승), substep(구간 내 반응률 갱신, 비용=CRAM 호출뿐)
- Isotalo & Sahlberg, NSE 179 (2015) — 스킴 체계 비교, **LE/QI+substep 최상위**
- Dufek/Kotlyar/Shwageraus, ANE 60 (2013) — SIE(무조건 안정, Xe/Gd 진동 제거); Kotlyar & Shwageraus ANE 92 (2016) substep 결합
- Josey/Forget/Smith, JCP 350 (2017) — 고차법(EPC-RK4/CF4): Gd-157 기준 2차 PC 대비 수송해 2배 이상 절감
- Pusa, NSE 169 (2011)/NSE 182 (2016) — CRAM 14·16 계수, **IPF 형식으로 고차(≤48) 안정화**
- 채택 실적: Serpent 2(CE/LI 기본, LE/QI·substep·SIE 옵션), OpenMC(Romano ANE 152 (2021) — 전 스킴 라이브러리), Shift(Davidson ANE 114 (2018) — LE/QI 계열), CASMO5(Gd 전용 2차 감손, Rhodes PHYSOR-2006), VANGARD(고차 CRAM + NIBC: Jeon ANE 192 (2023) 핀출력 8%→2.7%)

GPU 관점: 수송해가 지배 비용 → 수송해를 늘리지 않는 고차화(LE/QI, substep)가 정석이고 CRAM 추가 호출은 배치로 흡수(VANGARD PNE 156 (2023): 고차 CRAM으로 10⁶년급 스텝, 3분 주기감손).

**적용 권고 1순위: LE/QI + substep (corrector를 '밀도 평균'→QI 보간으로 교체 포함) + CRAM order 16 IPF 상향.**
- 스텝당 수송해 2회 유지 → 동일 정확도에서 감손 스텝 수 절반 목표 가능(배수는 문제 의존 — Serpent 참조해와 스텝 수렴성 시험 필수)
- 구현: 이전 스텝 반응률 저장(LE) + corrector 2차 보간(QI) + 스텝당 CRAM 4~8회(substep)
- 진동 관찰 시 SIE(+substep) 전환. 주의: 수렴 경로 변화 = 비트 계보와 별개 검증 체계 필요(정확도 게이트).

## B. 반사체 XS 생성

계보: Koebke(1981)/Smith PNE 17 (1986) 등가이론 → EDF Lefebvre-Lebigot 1D(Marguet 2018; Clerc arXiv:1405.2659) → PARAGON(WCAP-16045-NP) → **2D colorset**: Tahara/Kanagawa/Sekimoto JNST 37 (2000) — **1D B/R 상수는 코너부 집합체 출력 3~4% 오차, 2D 상수로 해소** → SIMULATE5 submesh 위치의존 상수(Bahadir ANFM V 2015, BEAVRS 실증) → MC 기반(Fridman & Leppänen ANE 38 (2011); Park & Park Energies 18 (2025) — McCARD/MASTER 한빛3 다주기, MACAO DF).
최신 등가 개선: Machach/Hébert/Dall'Osso NSE 199 (2025) — inscatter 확산계수 + DF 재규격화. albedo vs 명시노드: Petkov & Mittag PNE (2006). 실무 함정 총설: Smith PNE 101 (2017).

**적용 권고 1순위: 2D colorset(집합체+baffle+reflector 실기하, 코너 명시) 기반 위치의존 반사체 상수 + 2D DF.**
- 기대 이득: 주변부/코너 출력 오차 수 %p 감소 — **iSMR radial 잔차 미귀속 문제의 최우선 검증 가설**과 정합
- 2순위: inscatter D + DF 재규격화(NSE 199 2025); 3순위: baffle/barrel/pad 다중영역
- 검증 절차: Serpent/McCARD 동일 colorset 참조 계면류·DF 대조 → 노심추적 주변 집합체 출력 최종 확인

(조사 에이전트 전체 보고서의 요약판 — 서지 상세·추정 표기 목록은 조사 원문 참조)
