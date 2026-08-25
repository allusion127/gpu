# Screen → Exact campaign tool

- 실행기: `run_screen_exact_campaign.py`
- 계약 테스트: `test_screen_exact_campaign.py`
- 방법론: `../docs/EXACT_AND_SCREENING_THROUGHPUT_STRATEGY_20260825_KO.md`
- 실행 절차: `../docs/SCREEN_EXACT_CAMPAIGN_USAGE_20260825_KO.md`

빠른 계약 검사:

```bash
python tools/test_screen_exact_campaign.py
```

이 도구의 처리량은 approximate screen과 selected exact rerun을 합친 **설계 후보 처리량**이다. screen 결과 자체는 최종 물리 결과가 아니다.