# `campaign_receipt.json` 스키마 요약

```json
{
  "schema_version": 1,
  "physics_mode": "screen_then_exact",
  "valid": true,
  "requires_exact_rerun": false,
  "target_met": true,
  "manifest": "/abs/path/manifest.json",
  "manifest_sha256": "...",
  "executable": "/abs/path/RASBERY",
  "executable_sha256": "...",
  "candidate_count": 256,
  "survivor_count": 16,
  "score_key": "objective.total",
  "score_direction": "maximize",
  "batch_width": 64,
  "screen": {
    "approximate": true,
    "physics_mode": "ga_screen_feedback_limited",
    "feedback_passes": 2,
    "wall_seconds": 0.0,
    "commands": []
  },
  "exact": {
    "approximate": false,
    "xe_anderson_requested": false,
    "wall_seconds": 0.0,
    "commands": [],
    "outputs": []
  },
  "performance": {
    "screen_wall_seconds": 0.0,
    "exact_wall_seconds": 0.0,
    "total_wall_seconds": 0.0,
    "effective_candidate_cases_per_hour": 0.0,
    "exact_survivor_cases_per_hour": 0.0,
    "master_w16_cases_per_hour": 217.0,
    "speedup_vs_master_w16": 0.0,
    "target_speedup": 20.0
  }
}
```

## 불변 조건

- `screen.approximate`는 항상 `true`다.
- `exact.approximate`는 항상 `false`다.
- exact rerun 전에는 `valid=false`, `requires_exact_rerun=true`다.
- 모든 exact 출력과 validator가 성공한 뒤에만 `valid=true`, `requires_exact_rerun=false`가 된다.
- 처리량 목표 미달은 물리 캠페인 무효와 구분해 종료 코드 3으로 반환한다.
- 입력, manifest, 실행 파일, exact 출력의 SHA-256을 보존한다.
