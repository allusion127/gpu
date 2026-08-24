# GPU RASBERY 노달 CUDA 리팩토링 적용 영수증

작성일: 2026-08-24
기준: `codex/single-gpu-batch-dispatch-v2`
설계 근거: `GPU_NODAL_SUITABILITY_AND_SPEED_ROADMAP_20260824_KO.md`

## 적용 범위

이번 변경은 노달 계산 전체를 GPU 상태기계로 옮기지 않는다. CPU에는 Driver/Scheduler, 감손·서치·T/H 분기와 `updateConstant`의 초월함수 계산을 유지하고, 이미 FULL CUDA 경로에 들어온 대량 노드 연산과 배치 전송 고정비만 줄인다.

### 1. Mat + Even 동일 노드 커널 융합

기존 FULL 노달 그래프는 다음 5개 계산 커널을 실행한다.

```text
Trl0 -> Trl12 -> Mat -> Even -> Jnet
```

`Mat(lk)`가 생산한 `matM/matMI/mu/tau`는 `Even(lk)`가 같은 노드에서 바로 소비하며 다른 노드의 중간값을 읽지 않는다. 따라서 한 CUDA thread에서 두 기존 host/device 공유 본체를 순서대로 호출하는 `kNodalMatEven`을 추가했다. 부동소수점 표현식, FMA mask와 노드별 실행 순서는 바꾸지 않는다.

다음 경계는 유지한다.

- `Trl0 -> Trl12`: Trl12가 이웃 노드의 Trl0 결과를 읽으므로 grid-wide 경계 필요
- `MatEven -> Jnet`: Jnet이 표면의 좌·우 노드 결과를 읽으므로 grid-wide 경계 필요

기본은 융합이며 아래 설정으로 즉시 기존 5커널 경로로 돌아간다.

```bash
RASBERY_GPU_NODAL_FUSE_MAT_EVEN=0
```

### 2. NodalArena XS byte residency

기존 아레나는 별도 device allocation을 사용하므로 `hoststateGeneration()`을 residency key로 쓸 수 없었다. 그 결과 `xsrf`, `xsnf`, `xssm` 약 540 KiB/drive/slot을 항상 업로드했다.

새 경로는 각 슬롯에 마지막으로 **성공적으로 drain된 H2D 바이트**를 보관한다. 다음 입력이 `memcmp` 기준 완전히 동일할 때만 DMA를 생략한다. `double` 비교가 아니라 byte 비교이므로 signed zero, NaN payload와 최하위 비트까지 구분한다. shadow는 `cudaStreamSynchronize` 성공 후에만 갱신한다.

XS 업로드 여부는 batch마다 임시 container를 만들지 않고 기존 슬롯 객체의 3개 boolean에 기록한다. 따라서 전송 생략 최적화가 host allocator 호출을 새로 만들지 않는다.

기본은 활성화이며 아래 설정으로 기존 무조건 업로드 경로를 사용한다.

```bash
RASBERY_GPU_NODAL_XS_MIRROR=0
```

추가 영수증:

```text
mat_even_fused
xs_mirror
xs_h2d_bytes
xs_h2d_skipped_bytes
```

## 검증 상태

현재 환경에서 완료 가능한 검증:

- transfer mirror C++20 compile/run 계약
- source transformer idempotence 및 anchor drift fail-closed
- allocation-free hot-path transformer 계약
- phase dependency와 rollback 경로 정적 계약
- Python syntax 검사
- `git diff --check`

RTX PRO 6000 Blackwell의 CUDA 13 생산 빌드, 500/500 HDF5 bit gate와 M64 처리량은 서버 238에서 별도로 수행해야 한다. 따라서 이 커밋은 실제 속도 향상을 주장하지 않으며 Draft PR 상태로 유지한다.
