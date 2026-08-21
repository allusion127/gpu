# Nodal(SENM) GPU 이식 설계 — 착수 전 정찰 결과 (2026-08-21)

실측 근거(RASBERY_OUTER_TIMING, M64 프로파일 덱 57.45s):
outer당 직렬 경로 = drive 37.3ms(아레나 대기 포함) + **nodal 14.5ms(호스트 계산, 스레드합 762s)** +
setls 1.9ms + 기타 1.3ms. XS 블록은 GPU화 종결(eqxe 15.9s + flatxs 10.7s 스레드합뿐).
Nodal이 남은 최대 순수 CPU 블록이다.

## 1. 이식 가능성 판정: GO (형태 채굴 규모가 관건)

`Nodal.cpp` 정독 결과:

1. **초월함수는 단 한 곳** — `updateConstant()`의 `exp(kp)`(+sqrt). 그리고 이미
   `_constant_xsrf/_xsdf` 셰도로 **XS 변경 시에만 재계산**된다(outer당 no-op).
   glibc exp는 CUDA와 반올림이 다르므로 **updateConstant는 호스트 잔류**가 설계다
   — flatxs의 resolve/apply 분리와 동일한 원리. 케이스당 ~수십 회만 실행되므로
   비용 무시 가능, 결과 12개 상수 배열(~5MB)을 세대 게이트로 업로드.
2. 나머지 5개 페이즈(caltrlcff0, caltrlcff12, updateMatrix, calculateEven,
   calculateJnet)는 **순수 +,-,*,/ 산술 + 2×2 행렬해**뿐 → 형태 채굴로 비트 동일
   가능(ng=2 고정).
3. `drive()`가 이미 6개 배리어-분리 omp for = **커널 6개(호스트 1 + 디바이스 5)로
   1:1 매핑**. 페이즈 간 이웃 읽기는 배리어 뒤에서만 일어나므로 커널 경계가 곧
   동기 경계.
4. 문헌 정합: VANGARD(KNS 2020)가 같은 구조를 (셀,군) 스레드 + red-black으로
   이식하고 **D̂를 온디바이스 계산해 D2H 40%를 제거**했다. 우리 대응물 =
   upddhat까지 디바이스로 밀면 jnet 다운로드도 사라진다.

## 2. 설계 (RASBERY_GPU_NODAL)

- **백엔드**: XsReconBackend Impl에 통합(디바이스 `_xs` 재사용이 핵심 —
  updateMatrix가 매 outer 읽는 xssm/chif/xsnf/xsrf가 이미 flatxs 산출물로
  디바이스 상주. chif는 ref_generation에 실어 업로드).
- **상주**: Nodal의 전 배열(trlcff/eta/m/mu/tau/matM*/dsncff)을 디바이스 상주.
  호스트에는 updateConstant 산출 12개 배열만 존재(변경 시 업로드).
- **호출당 전송**: ↑ jnet(0.4MB)+flux(0.14MB)+reigv, ↓ jnet+phis(0.8MB)
  (+cusping 덱은 trlcff 1.2MB). **2단계에서 upddhat도 커널화하면 jnet 왕복까지
  소거**(CMFD dtil/dhat이 디바이스 생산 → setls 조립도 디바이스 후보 = outer 루프
  완전 상주화의 마지막 조각).
- **스레드 매핑**: 페이즈 1·2·4·5 = 노드당 1스레드(ng=2 내부 언롤), 페이즈 6 =
  표면당 1스레드(1n/2n 분기 — 경계 표면은 워프 소수라 divergence 무시 가능).
- **검증**: flatxs 방법론 그대로 — RASBERY_NODAL_DUMP로 drive() 1회의 입출력 +
  페이즈별 중간 배열 캡처 → 페이즈별 리플레이 게이트 → 형태 스윕.
  **사이트 수가 관건**: calculateJnet2n 하나가 ~25개 축약 사이트라 전수 스윕
  불가 → 페이즈별·배열별 분할 채굴(캡처된 페이즈 입력에서 시작해 페이즈 단위로
  0-ULP 달성 후 다음 페이즈) + 첫 발산 요소 국소화로 사이트를 이분 탐색.
  예상 규모: flatxs 캠페인(1일)의 1.5~2배.

## 3. 기대 효과 (실측 산술)

- nodal 14.5ms/outer 소거 → M64 벽시계 57.5 → ~46s 상당(−20%). 단 지연 한계
  구조상 일부는 drive 대기에 흡수될 수 있음 — 대신 인스턴스들의 아레나 도착이
  촘촘해져 배치 폭이 커지는 부수 효과(xsrecon 때 10.8→18.6 실측)가 있다.
- 2단계(upddhat/setls 디바이스 + jnet 상주)까지 가면 outer당 호스트 직렬 잔여는
  수렴판정+제어뿐 — 문헌상 "완전 상주" 성공 사례들(VANGARD·PRAGMA)의 구조에
  도달한다.

## 4. 선행 조건·주의

- canon 산술 불변: Nodal.cpp의 기존 형태를 바꾸면 안 됨(비트 계보 단절). 공유
  본체는 새 헤더(NodalKernel.h)로 추출하되 **호스트 경로는 기존 코드 유지**,
  캡처가 기준.
- `calculateJnet1n`의 albedo 경계, `trlcffbyintg`의 hmesh==0 분기 — 정확히 복제.
- 프로파일 덱은 반사체 경계 albedo 사용 → 1n 경로도 캡처에 나타남(검증 커버 ✓).
- cusping 덱(rodded)은 trlcff 다운로드 필요 — flatxs와 같은 rodded 게이트.
