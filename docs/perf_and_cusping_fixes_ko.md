# 성능·수렴·Cusping 수정 정리 (2026-05-30)

이 문서는 한 세션에서 적용한 정확성/성능 변경을 정리합니다. 핵심 메시지: **옛 코드가 빨랐던 건 일부 물리(특히 TH/Doppler 피드백)가 꺼져 있었기 때문이고, 지금은 올바른 결과를 내면서 다시 빠르게 만들었습니다.**

누적 결과(1-thread 안정 측정):

| 케이스 | 시작 | TH 최적화 후 | **솔버/CRAM 최적화 후(최종)** | 총 개선 |
|---|---|---|---|---|
| keffonly (keff 탐색) | 2025 outer / 16.5s | 1066 / 10.3s | **1058 / 9.41s** | **−43%** |
| i-SMR CY01 (rod 탐색) | 1936 / 18.0s | 1289 / 12.3s | **1372 / 11.48s** | **−36%** |

(8-thread: keffonly 7.1s, i-SMR 9.5s. i-SMR은 936노드로 작아 fine-grained 병렬 효율이 낮음 — 아래 §6·§7 참고.)

---

## 1. 제어봉 cusping 희석 버그 (정확성)

`_rod_cusping_relaxation`은 수렴 완화 노브가 아니라 **cusp을 영구 희석**시키고 있었습니다. `ApplyRodCusping`이 매 outer 반복마다 cusp 노드를 base로 리셋한 뒤 블렌딩 `_xs = r·cusp + (1−r)·old_xs`를 수행 → `old_xs`가 항상 base → 수렴점이 `base + r·(cusp−base)` = cusp의 r배(기본 0.1 = 10%)만 적용. 증거: relax에 따라 수렴 제어봉 위치가 최대 4cm(40%) 이동.

- **수정**: `ApplyRodCusping`을 누적형 고정점으로 재배열(스텐실 먼저 → 집합을 떠난 노드만 base 리셋). 기본 `_rod_cusping_relaxation = 1.0`(완전 cusp 직접 적용). relax<1은 premature flux 수렴으로 여전히 희석되므로 1.0이 정답이자 최속.
- 검증: relax=1.0은 i-SMR/1D_Rod_Cusping 모두 안정. 기존 "검증된" 베이스라인은 사실 10% 희석본이었으므로 PARCS 레퍼런스로 재검증 권장.

## 2. Rod-crit 탐색 허용오차 바닥값 (속도)

fractional cell로 staircase를 없앴지만 fine-cell 경계 kink가 ~3e-5 keff 노이즈를 남겨, 1e-5 탐색 허용오차가 노이즈 안에서 진동(스텝당 8–13 trial). cusping 활성 시 `rodcrit_search_floor = 5e-5`로 바닥값 설정 → 스텝당 2–5 trial. IISC 등 비-cusping 케이스는 게이트로 영향 없음(비트 동일).

## 3. OMP 게이트 통합 (속도, 큰 노심)

Nodal(>4096)/BICG(>2048)/리덕션(>2000) 게이트를 단일 런타임 노브 `rasbery_omp_gate`(기본 1024, env `RASBERY_OMP_GATE`)로 통합. 빅 케이스(3744노드, min-of-3): 직렬 111s → 게이트 1024 67.7s = **1.65배**. i-SMR(936<1024)은 직렬 유지(작은 케이스는 병렬이 손해). 주의: 이 박스의 8스레드 wall-time은 변동이 커서 반드시 min-of-3.

## 4. 제논 평형 기본값 (편의)

`xenon_transient` 기본값을 평형(false)으로 변경(Scheduler.h, IO.cpp). `"xenon":"transient"`로 옵트인. 단, **step-2 outer 급증은 제논 모드와 무관**(transient/equilibrium 비트 동일)함을 확인 — 첫 연소 스텝의 BOC 콜드스타트가 원인.

## 5. TH/Doppler 피드백 — outer "폭증"의 진실 (정확성 + 속도)

옛 커밋 대비 outer가 ~5배(keffonly 390→2025)였던 이유를 git worktree 이분법으로 추적 → **7a23d53 "Refresh HGC and cleanup"**.

**근본 원인**: 고유치 해가 **음수 flux**(고유벡터 부호 모호성)를 반환 → `SolveTH`의 `norm = (total_raw_power>0)?…:0` 가드가 출력을 0으로 → **Doppler/감속재 피드백이 통째로 무시됨**. 옛 코드(V1.0)는 이렇게 피드백이 꺼진 채로 빨랐음(keff 1.0513 = 무피드백, 물리적으로 틀림). 7a23d53이 `UpdateTH`에 `node_power` 부호 뒤집기 블록을 추가해 피드백을 켰고(keff 1.0387, 올바름) 그래서 느려진 것. **솔버 회귀 아님** — TH를 끄면(thmode none) 현재 코드도 441 outer로 옛 390과 동일.

> `UpdateTH`의 `node_power` 부호 블록은 **삭제 금지**(피드백 비활성화됨). 근본 정리는 솔버에서 고유벡터 부호를 양수로 정규화하는 것.

**적용한 가속(모두 결과 보존)**:
1. **TH under-relaxation** `_th_relaxation=0.85`(env `RASBERY_TH_RELAX`): `SolveTH`가 온도를 직접 덮어써서 undamped Picard가 진동(delta_Dop 0.41→0.59). `UpdateTH`에서 SolveTH 이전 온도(tful/tmod/dmod) 스냅샷과 블렌딩 → 같은 고정점에 절반 반복으로 수렴. 최적 ~0.85(keff·rod 둘 다 이득).
2. **Predictor BOS solve 생략**(Driver.h depletion 루프): 스텝당 3 solve는 BOS / N^P(예측) / N^C(보정). "Final"은 N^C 해(탐색·PPR·보고가 여기) — corrector solve(N^P)와 다른 조성이라 중복 아님. 진짜 중복은 predictor solve(BOS_k = EOS_{k−1}, 이전 스텝 final이 이미 수렴). isub==0에서 생략하고 이월 flux 재사용(PredictorStep이 `_g.Phif()`를 BOS로 스냅샷). ~9%.
3. **Doppler 허용오차 1e-3 → 1e-2**(Driver.h `TH_DOPPLER_TOLERANCE`): 1e-3(~0.9K)은 과도하게 빡빡. 1e-2에서 power는 rtol 1e-3 내 0 diff, 추가 −19~25%.

TH 피드백은 스텝/solve 간 온도 warm-start가 이미 동작(첫 delta_Dop 0.33→0.18→0.07 감소)하므로 cold-restart 이득은 없음. 추가 여지: TH 고정점 Anderson 가속.

## 6. 솔버·CRAM 미세 최적화 (속도, 실제 프로파일 기반)

실측 프로파일(i-SMR, 1-thread) 상위 핫스팟: XS 재구성(`ApplyBranchDeltaIdToNode`) ~27%, 선형솔버(`axb`+`minv`+`solve`) ~26%, CRAM 연소(`solveBatemanCRAM`+`hypot`) ~13%, Nodal ~11%. 이를 근거로 적용:

1. **CRAM `std::abs(complex)` → `detail::magnitude()`** (include/milk.h 4사이트). 프로파일에서 `hypot`이 5.6%였는데, CRAM Gauss-Seidel 잔차/대각 검사가 glibc `std::abs(complex)`(=overflow-safe `hypot`)를 호출. 라이브러리 자체 `magnitude()`(정상 범위는 `sqrt(r²+i²)`, 극단만 hypot로 fallback — 오버플로 가드 유지)로 교체. **1-thread −6%** (keffonly 10.3→9.65s). 결과는 ~1e-9 내(FP 재결합).
2. **BiCGSTAB 내부 반복 `_nmaxbicg` 6 → 4** (src/BICGCMFD.cpp). `setIterLim`은 호출되지 않는 dead code라 6이 사실상 하드코딩이었음. 0.1 상대잔차에서 보통 일찍 탈출하므로 4로 충분. outer는 i-SMR에서 ~6% 늘지만 inner sweep 절감이 더 커서 **순이득**(keffonly 9.65→9.41s, i-SMR 11.9→11.47s).
3. **OMP 게이트 1024 → 256** (src/main.cpp). i-SMR(936노드)에서 `axb`/Nodal/리덕션을 병렬화 → 8-thread에서 약 5% (10.6→10.1s). 작은 케이스도 병렬이 도움(옛 메모리의 "작은 케이스 병렬 손해"는 더 작은 크기 한정이었음을 실측으로 정정).

## 7. red-black SSOR 병렬화 — 시도했으나 실패 (기록용)

`minv`(SSOR 전후방 sweep)가 유일한 본질적 직렬 병목이라, 7-point 스텐실의 bipartite성을 이용한 **red-black(2-color) SSOR**를 구현했습니다(BFS 2-coloring → red/black 4단계 완전병렬 sweep). **수렴은 정상**(keff 정확, outer 오히려 1058→1026 감소)이었으나 **속도는 전부 악화**: i-SMR 8-thread 3배 느림(7.2→22.6s), 1-thread도 인디렉션/캐시로 느림, 큰 케이스(3744)도 lex 67s 대비 red-black 80s. 원인: i-SMR이 작아 per-color fork/join + `_red[i]` 인디렉션 비용이 병렬 이득을 압도. → **전면 되돌림.** 결론: 이 코드/문제크기에서 SSOR 병렬화는 이득 없음(진짜 병렬 확장은 multicolor 재정렬 대규모 재작성 필요하나 ROI 낮음). i-SMR(936노드)은 fine-grained 병렬에 너무 작아 8-thread 효율이 본질적으로 제한됨(XS 재구성·CRAM만 병렬 이득, 선형솔버는 작은 루프 fork/join 한계).

---

## 진단 env 노브 (이 세션 추가)
- `RASBERY_OMP_GATE` — 솔버 병렬 게이트(기본 256)
- `RASBERY_TH_RELAX` — TH 피드백 under-relaxation(기본 0.85)
- `RASBERY_CUSP_RELAX` — cusp relaxation(기본 1.0; ≤2.0 허용, 실험용)
- `RASBERY_SL_TRACE` — SolveLoop별 outer/TH + TH iter별 delta_Dop 추적
- NO.= 출력 줄에 스텝별 wall time `t=..s` 추가
