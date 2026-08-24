# RASBERY 단일 GPU 동시입력 워커 매트릭스 실행 절차

대상 브랜치: `codex/single-gpu-batch-dispatch`  
대상 장치: GPU 0 한 장  
비교 대상: 동일 APR1400 입력 64개, 동일 출력 범위, 동일 물리 옵션

## 1. 목적

CUDA 아레나 폭과 CPU Driver 워커 수를 분리하여, GPU 슬롯 수를 줄이지 않은 상태에서 호스트 과가입과 입력 도착 스큐를 줄인다. 비교 중 변경하는 변수는 `RASBERY_BATCH_HOST_THREADS` 하나뿐이다.

- 고정: GPU 0, 입력 64개, `--batch-mode 64`, CUDA/물리 옵션, 출력 데이터셋
- 변화: host worker 24, 36, 48, 64
- 판정: 유효 실행 3회의 중앙값 cases/h
- 무효: 비정상 종료, telemetry 누락, graph fallback 1건 이상

## 2. 빌드 및 정적 검증

```bash
git checkout codex/single-gpu-batch-dispatch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 24

python tools/test_cmfd_geometry_cache.py
python tools/test_single_gpu_batch_profile.py
python tools/test_benchmark_single_gpu_matrix.py
```

생산 서버에서는 기존 CUDA 13 및 `sm_120` 설정을 그대로 사용한다.

## 3. 정확도 골든 생성

성능 실행과 별도로 CPU 골든 HDF5를 먼저 생성한다. GPU 매트릭스의 모든 출력은 동일 상태점과 동일 데이터셋을 골든과 비교한다.

필수 비교 항목:

- `keff`, 보론 및 제어봉 결과
- 2군 flux 및 pin/assembly power
- TH 상태, depletion/decay 결과
- 모든 상태점과 전체 HDF5 데이터셋
- NaN/Inf, 음의 flux 및 solver flag

## 4. 워커 매트릭스 실행

아래의 `deck_00.json ... deck_63.json`과 출력명은 실제 APR1400 파일 목록으로 치환한다.

```bash
python tools/benchmark_single_gpu_matrix.py \
  --batch-width 64 \
  --gpu 0 \
  --workers 24,36,48,64 \
  --warmups 1 \
  --repeats 3 \
  --timeout 14400 \
  --output-root benchmark/apr1400_gpu0_workers \
  -- \
  ./build/RASBERY \
  --rasi deck_00.json deck_01.json deck_02.json deck_03.json \
         deck_04.json deck_05.json deck_06.json deck_07.json \
         deck_08.json deck_09.json deck_10.json deck_11.json \
         deck_12.json deck_13.json deck_14.json deck_15.json \
         deck_16.json deck_17.json deck_18.json deck_19.json \
         deck_20.json deck_21.json deck_22.json deck_23.json \
         deck_24.json deck_25.json deck_26.json deck_27.json \
         deck_28.json deck_29.json deck_30.json deck_31.json \
         deck_32.json deck_33.json deck_34.json deck_35.json \
         deck_36.json deck_37.json deck_38.json deck_39.json \
         deck_40.json deck_41.json deck_42.json deck_43.json \
         deck_44.json deck_45.json deck_46.json deck_47.json \
         deck_48.json deck_49.json deck_50.json deck_51.json \
         deck_52.json deck_53.json deck_54.json deck_55.json \
         deck_56.json deck_57.json deck_58.json deck_59.json \
         deck_60.json deck_61.json deck_62.json deck_63.json \
  --raso out_00.h5 out_01.h5 out_02.h5 out_03.h5 \
         out_04.h5 out_05.h5 out_06.h5 out_07.h5 \
         out_08.h5 out_09.h5 out_10.h5 out_11.h5 \
         out_12.h5 out_13.h5 out_14.h5 out_15.h5 \
         out_16.h5 out_17.h5 out_18.h5 out_19.h5 \
         out_20.h5 out_21.h5 out_22.h5 out_23.h5 \
         out_24.h5 out_25.h5 out_26.h5 out_27.h5 \
         out_28.h5 out_29.h5 out_30.h5 out_31.h5 \
         out_32.h5 out_33.h5 out_34.h5 out_35.h5 \
         out_36.h5 out_37.h5 out_38.h5 out_39.h5 \
         out_40.h5 out_41.h5 out_42.h5 out_43.h5 \
         out_44.h5 out_45.h5 out_46.h5 out_47.h5 \
         out_48.h5 out_49.h5 out_50.h5 out_51.h5 \
         out_52.h5 out_53.h5 out_54.h5 out_55.h5 \
         out_56.h5 out_57.h5 out_58.h5 out_59.h5 \
         out_60.h5 out_61.h5 out_62.h5 out_63.h5 \
  --batch-mode 64
```

## 5. 생성 결과

`--output-root` 아래에 다음 파일이 생성된다.

- `manifest.json`: 고정 조건과 전체 명령
- `runs.csv`: 각 실행 wall time, cases/h, 유효성, telemetry
- `summary.json`: 워커별 중앙값과 최적 유효 구성
- `<phase>-<round>-<worker>/run.log`: 원본 stdout/stderr
- `<phase>-<round>-<worker>/outputs/`: 실행별 독립 HDF5

## 6. 채택 기준

1. 전체 HDF5 골든 비교 통과
2. `graph_fallbacks == 0`
3. CUDA 또는 solver fatal flag 0
4. 동일 워커 구성의 유효 측정 3회 확보
5. MASTER W16의 216–218 cases/h를 중앙값으로 초과
6. 최고 처리량과 차이가 2% 이내면 더 적은 워커 구성을 선택

첫 매트릭스에서 24/36/48/64 중 최고값을 찾은 후, 그 주변만 세분화한다. 예를 들어 36이 최고면 30/36/42를 추가 측정한다. 고정 `RASBERY_BATCH_WAIT_US`는 워커 수를 결정한 뒤 0/25/50/100 us 순서로 별도 측정하며 두 변수를 한 번에 바꾸지 않는다.

## 7. 다음 코드 단계의 진입 조건

워커 분배만으로 MASTER를 넘지 못하거나 CMFD/Nodal 평균 batch width가 낮게 유지될 때 다음 단계로 이동한다.

- `upddhat + setls + updjnet` 배치 GPU operator assembly
- `diag/cc`를 BiCGSTAB 소비 시점까지 device-resident로 유지
- 슬롯별 dhat guard 및 최대 defect reduction
- HDF5 스냅샷 소유권 경계와 단일 writer queue

각 단계는 별도 커밋과 별도 골든/처리량 게이트를 사용한다.
