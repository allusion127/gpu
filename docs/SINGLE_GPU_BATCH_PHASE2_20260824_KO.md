# 단일 GPU 동시입력 최적화 2차 적용 메모

작성일: 2026-08-24  
대상: `codex/single-gpu-batch-dispatch`

## 목적

RTX PRO 6000 한 장과 동일 APR1400 기하의 다중 입력 조건에서 CPU Driver가 GPU 배치 랑데부에 도착하는 편차를 줄이고, CUDA arena 폭과 호스트 워커 수를 독립적으로 조절한다. 모든 상태기계를 GPU에 상주시켜 복잡도를 키우지 않고, 반복 횟수가 큰 CMFD 노드·표면 연산과 입력 공급 경로만 최적화한다.

## 2차 코드 변경

### APR1400 2군 CMFD 고정 경로

`CMFD::upddhat`, `CMFD::updjnet`, `CMFD::updpsi`에 `ng == 2` 고정 횟수 경로를 추가했다. APR1400 생산 덱의 두 군을 명시적으로 처리해 반복마다 군 수 조회, stride 계산과 accessor 호출을 줄인다. 다른 군 구조는 기존 generic loop를 그대로 사용한다.

- `upddhat`: surface-group flat index를 한 번 계산하고 dtil/dhat/jnet을 직접 접근한다.
- `updjnet`: 두 군을 같은 순서로 명시적으로 갱신한다.
- `updpsi`: `xsnfData()`의 group-major SoA와 node-major flux를 직접 결합한다.
- dhat의 fsum guard, finite 검사, envelope 통계와 clamp opt-in 의미는 바꾸지 않았다.
- `setls`는 기하 캐시를 사용하되 기존 group·direction·대각 누산 순서를 유지한다.

### 자동 워커 매트릭스

`tools/benchmark_single_gpu_matrix.py`를 추가했다. 한 GPU와 한 CUDA arena 폭을 고정하고 host Driver worker만 24/36/48/64로 바꿔 교차 실행한다.

- 각 configuration을 한꺼번에 실행하지 않고 repeat 안에서 교차해 GPU clock·온도·실행 순서 편향을 줄인다.
- 각 trial마다 독립된 `--raso` 경로와 전체 로그를 생성한다.
- launch profile, batch occupancy, backend counter를 파싱한다.
- telemetry 누락, graph fallback 발생, graph launch 또는 batch launch가 0이면 해당 run을 무효 처리한다.
- `manifest.json`, `runs.csv`, `summary.json`을 저장한다.
- 유효 run의 median cases/h로 최적 worker 수를 선택하고 MASTER W16 218 cases/h gate를 평가한다.

실행 예:

```bash
python tools/benchmark_single_gpu_matrix.py \
  --batch-width 64 \
  --gpu 0 \
  --workers 24,36,48,64 \
  --warmups 1 \
  --repeats 3 \
  --master-cases-per-hour 218 \
  --throughput-gate 218 \
  --output-root bench/apr1400_single_gpu_20260824 \
  -- \
  ./RASBERY \
  --rasi deck_00.json deck_01.json ... deck_63.json \
  --raso golden_00.h5 golden_01.h5 ... golden_63.h5 \
  --batch-mode 64
```

## 검증 상태

현재 환경에서 완료한 검증:

```text
cmfd geometry cache: PASS
single gpu batch profile: PASS
single gpu benchmark matrix: PASS
CMFD C++17 syntax contract: PASS
```

계약 테스트는 합성 2군 문제에서 기존 loop와 고정 경로의 double bit pattern을 비교하며, left boundary, right boundary, interior surface 및 guarded dhat update를 포함한다.

아직 완료하지 않은 생산 게이트:

- CUDA 13 / `sm_120` 실제 빌드
- APR1400 CPU golden 전체 HDF5 비교
- RTX PRO 6000 GPU0 M1/M64 반복 벤치마크
- Nsight Systems/Compute 측정

따라서 현재 커밋은 성능을 개선할 구조와 재현 가능한 측정 경로를 구현한 것이며, MASTER보다 빠르다는 수치는 아직 주장하지 않는다.

## 다음 적용 순서

1. 24/36/48/64 워커 매트릭스로 단일 GPU 최적 host 공급 폭을 결정한다.
2. 정확도 gate와 218 cases/h gate를 모두 통과한 구성을 기준선으로 고정한다.
3. 다음 코드 단계에서는 전체 Driver가 아니라 `upddhat + setls + updjnet` operator assembly만 배치 GPU화한다.
4. diag/cc는 BiCGSTAB가 즉시 소비하게 하고, CPU에는 필요한 jnet/dhat 진단 최소값만 반환한다.
5. 이후 `cmfd_wiel_finalize`의 세 독립 합을 각자 기존 `l=0..nxyz-1` 순서로 계산하게 분리해 결정론을 유지한다.
6. 마지막으로 HDF5 출력 경합을 계산 경로에서 분리한다.

## 채택 기준

| 단계 | 정확도 | 처리량 | 실패 조건 |
|---|---|---:|---|
| 2차 host 경로/워커 매트릭스 | CPU golden HDF5 동일 | >218 cases/h | telemetry 누락 또는 graph fallback |
| CMFD operator assembly GPU화 | 500/500 dataset byte 동일 | >240 cases/h | H2D 증가 또는 CPU fallback |
| finalize/HDF5 개선 | 동일 | >260 cases/h | 출력 누락 또는 writer stall 증가 |

처리량은 목표/채택 기준이며 보장 수치가 아니다. 실제 GPU0 반복 측정 중앙값만 최종 보고서에 반영한다.
