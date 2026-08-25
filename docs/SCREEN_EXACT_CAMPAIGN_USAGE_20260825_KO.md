# MASTER W16 대비 screen → exact 캠페인 실행 절차

## 1. 전제

이 절차가 산정하는 값은 **대량 설계 후보의 실효 처리량**이다. screen 단계는 근사 계산이며 최종 물리 결과가 아니다. 선택된 survivor의 exact 계산이 모두 완료된 경우에만 캠페인을 유효하다고 판정한다.

검증된 단일 GPU batch 상한은 64이므로 `--batch-width`는 64 이하로 둔다.

## 2. manifest

JSON 배열 또는 JSONL을 사용한다.

```json
[
  {"id": "loading_0000", "input": "decks/loading_0000.json"},
  {"id": "loading_0001", "input": "decks/loading_0001.json"},
  {"id": "loading_0002", "input": "decks/loading_0002.json"}
]
```

각 input 경로는 서로 달라야 한다. 같은 경로를 두 슬롯에서 동시에 사용하는 것은 output/host-registration 수명 문제를 만들 수 있으므로 실행기가 fail-closed로 거부한다.

## 3. 실행 예

아래 예시는 score가 큰 후보를 상위 16개 선택한다.

```bash
python tools/run_screen_exact_campaign.py \
  --manifest campaign/manifest.json \
  --workdir campaign/run_001 \
  --executable ./build/RASBERY \
  --batch-width 64 \
  --screen-feedback-passes 2 \
  --survivors 16 \
  --score-key objective.total \
  --maximize \
  --master-w16-cases-per-hour 217 \
  --target-speedup 20 \
  --timeout 14400 \
  --common-arg=--chiffoni \
  --common-arg=/data/chiffon.h5
```

`--common-arg`는 RASBERY 실행 파일에 그대로 전달된다. 값이 `-`로 시작하거나 공백을 포함하면 `--common-arg=VALUE` 형식을 사용한다.

exact survivor에서 실험적인 Xe Anderson 경로를 A/B하려면 다음을 추가한다.

```bash
--exact-xe-anderson
```

단, 실행 파일에 해당 기능이 구현되어 있지 않으면 이 옵션은 환경변수만 설정하므로 기존 exact 경로가 실행된다. 기능 실가동 여부는 실행 로그와 전용 telemetry로 별도 확인한다.

## 4. 산출물

```text
campaign/run_001/
  campaign_receipt.json
  screen_ranking.json
  logs/
    screen_0001.log
    ...
    exact_0001.log
  screen/
    loading_0000.jsonl
    ...
  exact/
    <selected survivor>.h5
```

`campaign_receipt.json`의 핵심 필드는 다음과 같다.

```json
{
  "physics_mode": "screen_then_exact",
  "valid": true,
  "requires_exact_rerun": false,
  "target_met": true,
  "performance": {
    "effective_candidate_cases_per_hour": 0.0,
    "master_w16_cases_per_hour": 217.0,
    "speedup_vs_master_w16": 0.0,
    "target_speedup": 20.0
  }
}
```

`valid=true`는 exact survivor 출력이 모두 존재하고 검증기를 통과했다는 뜻이다. `target_met=true`는 정확성 판정과 별개로 목표 처리량을 넘었다는 뜻이다.

## 5. 종료 코드

| 종료 코드 | 의미 |
|---:|---|
| 0 | exact 캠페인 유효, 처리량 목표 통과 |
| 2 | 입력·screen receipt·exact 출력·validator 중 하나가 실패하여 캠페인 무효 |
| 3 | exact 캠페인은 유효하지만 MASTER W16 대비 목표 배수 미달 |

CI나 최적화 파이프라인에서는 종료 코드 2와 3을 모두 실패로 취급한다. 성능 목표를 낮춰 성공으로 보이게 하지 말고, receipt의 실제 wall과 baseline을 보존한다.

## 6. 외부 exact validator

HDF5 구조·필수 dataset·물리 허용오차를 검사하는 별도 프로그램이 있으면 다음과 같이 연결한다.

```bash
--exact-validator 'python tools/validate_exact.py --input {input} --output {output}'
```

각 survivor에 대해 validator가 0으로 종료해야 캠페인이 유효하다. 단순히 파일이 생성되었다는 사실만으로 물리 검증을 대체하지 않는다.

## 7. 필수 보고 지표

- screen 후보 수와 survivor 수
- screen wall, exact wall, 전체 wall
- effective candidate cases/h
- exact survivor cases/h
- MASTER W16 실측 cases/h
- speedup vs MASTER W16
- exact 성공률
- objective별 Spearman 상관과 top-k recall
- screen과 exact의 실행 파일 및 입력 SHA-256
- GPU fallback, graph failure, 비유한값 발생 수

## 8. 해석 제한

screen 256개와 exact 16개의 총 wall로 계산한 후보 처리량이 MASTER W16보다 20배 빠르더라도, 이는 256개 모두를 exact 계산한 처리량이 아니다. 최종 설계 채택값은 exact survivor 결과만 사용한다. 이 구분이 유지될 때만 screen → exact 방식의 수십 배 가속 주장이 재현 가능하고 감사 가능하다.