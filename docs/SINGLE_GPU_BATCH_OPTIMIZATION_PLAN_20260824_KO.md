# GPU RASBERY 단일 GPU 동시입력 최적화 계획 및 1차 적용 내역

작성일: 2026-08-24  
대상 브랜치: `codex/single-gpu-batch-dispatch`  
기준 브랜치: `codex/hotpath-opt`

## 1. 목표와 제약

목표는 RTX PRO 6000 한 장만 사용하면서 동일 APR1400 기하를 공유하는 다수 입력을 비동기적으로 공급하고, GPU 배치 아레나의 실효 폭을 높여 MASTER W16 처리량을 넘어서는 것이다.

고정 제약은 다음과 같다.

- GPU는 한 장만 사용한다. 실행 시 `CUDA_VISIBLE_DEVICES=0`으로 고정한다.
- Driver/Scheduler의 서치, 감손, T/H, 제어 분기는 CPU에 남긴다.
- 모든 코드를 무조건 GPU에 상주시켜 큰 상태기계를 만들지 않는다.
- 노드·표면 단위 대량 계산만 GPU 또는 캐시 친화적인 연속 메모리 루프로 보낸다.
- 단일 입력 및 배치 입력의 물리 결과는 기존 골든 출력과 동일해야 한다.
- 성능 변경은 정확도 변경과 분리해 커밋하고, CUDA 폴백 또는 그래프 실패가 한 건이라도 발생하면 성능 수치를 채택하지 않는다.

## 2. 기준선과 병목

보고서 기준 M64 처리량은 169.2 cases/h이고 MASTER W16은 216–218 cases/h이다. 현재 GPU 경로는 MASTER 병렬 기준보다 약 22% 낮다. CMFD 평균 집계 폭은 22.7/64, 노달 아레나 평균 폭은 6.3이며, 고정 대기를 추가해 폭만 키우는 방식은 wall time을 줄이지 못했다.

단일 실행 프로파일에서 CPU 잔류 CMFD 연산은 `setls` 7.48 s, `upddhat` 4.63 s, `updjnet` 1.50 s이다. 이 경로는 GPU 커널 자체의 계산량뿐 아니라 각 Driver가 다음 GPU 랑데부에 도착하는 시차를 만든다. 따라서 첫 단계는 전면 GPU 상주가 아니라 **CPU 핫패스 단축과 Driver 워커 수 제어**이다.

## 3. 실행 구조

### 3.1 CPU 역할

- 입력별 Driver/Scheduler 상태기계
- 붕소·제어봉 서치, 자연 EOC, Xe 진동 제어
- CRAM/PC 지휘와 분기 많은 T/H 작업
- HDF5 호출과 결과 수명 관리

### 3.2 GPU 역할

- 동일 기하의 여러 입력을 `grid.y = active slots`로 묶은 CMFD BiCGSTAB
- 디바이스 Wielandt sweep
- XSRECON, FLATXS, Nodal FULL의 노드·표면 병렬 연산
- 슬롯별 halt/active 마스크를 이용한 ragged convergence 처리

### 3.3 분배 정책

CUDA arena 폭과 CPU Driver 워커 수를 서로 다른 자원으로 취급한다.

- arena 폭: 최대 동시 상태 수와 GPU 그리드의 슬롯 축
- host worker 수: CPU에서 실제로 진행되는 Driver 수
- 남은 입력: OpenMP `schedule(dynamic, 1)` 작업 큐에서 워커 종료 즉시 재배정

24 CPU 스레드 서버의 64입력 초기 후보는 24, 36, 48, 64 워커이다. 각각 CPU 배수 1.0, 1.5, 2.0, legacy에 해당한다. 성능 수치 없이 하나를 생산 기본값으로 고정하지 않고 동일 덱으로 A/B한다.

## 4. 1차 적용 사항

### 4.1 CMFD 불변 기하 캐시

`CMFD` 생성 시 다음 불변 데이터를 연속 배열에 한 번만 구성한다.

- 표면별 좌·우 노드와 방향
- 노드·방향·면별 surface 및 neighbor 인덱스
- 노드별 mesh 길이, 면적, 체적
- 경계별 albedo

그 결과 `setls`, `upddtil`, `upddhat`, `updjnet`, `updpsi`, 희귀 Rayleigh fallback의 `axb`가 반복마다 `Geometry` 인덱서와 포인터 체인을 다시 조회하지 않는다. flux, jnet, XS, dtil, dhat, diag, cc는 캐시하지 않으므로 상태점별 물리값은 기존과 동일하게 갱신된다.

`setls`는 기존의 그룹 순서, 왼쪽 방향 역순, 오른쪽 방향 정순 및 각 대각 원소의 덧셈 순서를 유지하면서 `diag_l`과 `cc_l` 연속 포인터를 사용하도록 변경했다. 산술 재결합은 하지 않았다.

### 4.2 단일 GPU 실행 프로필

`tools/run_single_gpu_batch.py`를 추가했다.

- 기본적으로 GPU 0 한 장만 노출한다.
- 보고서의 표준 GPU 스위치를 한 번에 설정한다.
- `--rasi` 입력 수와 `--batch-mode` 폭을 확인한다.
- 보이는 CPU affinity를 기준으로 `RASBERY_BATCH_HOST_THREADS`를 수치로 계산한다.
- arena 폭은 유지하고 Driver만 제한한다.
- 실행 전 `[RASBERY][SINGLE_GPU_PROFILE]` JSON receipt를 출력한다.
- `--worker-factor 1.0|1.5|2.0` 또는 `--host-workers legacy|N`으로 A/B할 수 있다.

예시:

```bash
python tools/run_single_gpu_batch.py \
  --batch-width 64 \
  --gpu 0 \
  --host-workers auto \
  --worker-factor 1.0 \
  -- \
  ./RASBERY \
  --rasi deck_00.json deck_01.json ... deck_63.json \
  --raso out_00.h5 out_01.h5 ... out_63.h5 \
  --batch-mode 64
```

24 CPU 스레드가 보이면 위 명령은 arena 64슬롯과 host worker 24개를 사용한다. `--worker-factor 1.5`는 36개, `2.0`은 48개, `--host-workers legacy`는 64개를 사용한다.

### 4.3 1차 검증

- 수정된 `CMFD.cpp`는 mock `Geometry/XSSet`을 사용한 C++17 `-Wall -Wextra -Werror -fsyntax-only` 검사를 통과했다.
- `tools/test_cmfd_geometry_cache.py`는 핫 함수에 Geometry 재조회가 남지 않았는지 검사하고, 합성 2군 문제에서 기존 수식과 캐시 수식의 double 비트를 비교한다.
- `tools/test_single_gpu_batch_profile.py`는 입력 수, batch 폭, CPU affinity, worker factor, GPU 고정 및 환경변수 영수증 계약을 검사한다.

이 검증은 소스·산술 계약 검사이다. 생산 CUDA 빌드, APR1400 골든 HDF5 및 처리량 측정은 238 서버에서 별도로 수행해야 한다.

## 5. 다음 구현 단계

### 단계 A — 1차 패치 서버 검증

1. CPU 골든 대비 Tier 3 전체 HDF5 비교
2. M1, M64 각각 3회 이상 반복
3. worker 24/36/48/64 매트릭스 측정
4. `RASBERY_BATCH_WAIT_US=0`을 기준으로 25/50/100 µs만 제한적으로 측정
5. 아래 영수증을 모두 저장
   - wall 및 cases/h
   - CMFD/Nodal mean width와 histogram
   - HDF5 lock wait
   - H2D/D2H bytes 및 graph fallback
   - `outer_timing`, `xsphase`, GPU utilization

채택 조건은 골든 출력 일치, fallback 0, graph failure 0이며 처리량은 최소 218 cases/h 초과이다.

### 단계 B — CMFD operator assembly 배치 GPU화

전면 Driver 상주 대신 `upddhat + setls + updjnet`만 별도 배치 연산으로 만든다.

- Geometry surface/node map은 arena 공용 GPU 메모리에 1회 상주시킨다.
- 입력별 flux, jnet, XS 및 dhat 진단 상태만 슬롯 스트라이드로 전달한다.
- `setls`가 만든 diag/cc는 BiCGSTAB가 바로 소비하므로 CPU로 되돌리지 않는다.
- CPU 소비자가 필요한 jnet/dhat 최소집합만 경계에서 회수한다.
- dhat guard 통계와 clamp 옵트인은 슬롯별 정수/최댓값 reduction으로 유지한다.
- 기존 CPU 구현은 fail-open 폴백과 골든 생성 경로로 남긴다.

이 단계의 목적은 M64에서 보고된 대규모 diag/cc 왕복과 CPU 도착 스큐를 동시에 줄이는 것이다.

### 단계 C — Wielandt finalize의 독립 합 병렬화

현재 `err`, `gammad`, `gamman`은 서로 독립이지만 한 스레드가 세 개를 함께 순차 폴드한다. 각 합의 `l=0..nxyz-1` 순서를 그대로 유지한 채 슬롯당 3개 스레드가 각각 한 합을 담당하고, thread 0이 eigenvalue 업데이트를 수행하도록 변경할 수 있다. 각 합 내부의 operand pairing은 바뀌지 않으므로 결정론을 유지할 수 있다. 이 변경은 CUDA replay와 500-dataset bit gate 후 별도 커밋으로 반영한다.

### 단계 D — 출력 경합 분리

현재 HDF5 1.10 런타임은 프로세스 전역 직렬화가 필요하다. 라이브 객체를 writer thread로 넘기는 방식은 수명 문제를 만들 수 있으므로, 먼저 상태점 결과를 소유권이 명확한 메모리 스냅샷으로 직렬화하는 경계를 만든다. 그 다음 단일 writer 큐 또는 thread-safe HDF5 1.14 빌드를 A/B한다. 계산 스레드가 HDF5 mutex 안에서 장시간 대기하지 않는 것이 목표다.

## 6. 성능 판정 기준

| 단계 | 정확도 게이트 | 최소 처리량 게이트 | 주요 추가 지표 |
|---|---:|---:|---|
| 1차 캐시/워커 분배 | CPU 골든 HDF5 동일 | >218 cases/h | HDF5 wait, CMFD width, host RSS |
| CMFD operator GPU화 | 500/500 dataset byte 동일 | >240 cases/h | H2D bytes, setls/dhat/jnet 시간 |
| finalize/HDF5 개선 | 동일 | >260 cases/h | graph nodes, writer queue stall |

처리량 목표는 각 단계의 채택 기준이며 보장 수치가 아니다. 실제 RTX PRO 6000 GPU0에서 측정된 중앙값만 보고서에 반영한다.

## 7. 금지 사항

- 정확도 완화, 반복 수 축소 또는 GA screening 근사값을 생산 벤치마크와 혼합하지 않는다.
- GPU occupancy를 높이기 위해 입력을 lock-step으로 강제하지 않는다.
- 고정 linger로 mean width만 높이고 wall time이 악화된 결과를 채택하지 않는다.
- 측정 없이 worker 수, CUDA block 크기 또는 iteration batch를 기본값으로 고정하지 않는다.
- MASTER 대비 가속도는 동일 물리 모드, 동일 출력 범위, 동일 입력 집합으로만 계산한다.
