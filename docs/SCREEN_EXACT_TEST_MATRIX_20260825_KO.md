# Screen → Exact 테스트 매트릭스

| 항목 | 기대 결과 |
|---|---|
| width 1/2/64 | 실행 허용 |
| width 0/65 | 종료 코드 2 |
| 서로 다른 입력 경로 | 실행 허용 |
| 중복 입력 경로 | 종료 코드 2 |
| screen mode 정상 | score 파싱 |
| screen mode 누락/오류 | 종료 코드 2 |
| requires_exact_rerun 누락/false | 종료 코드 2 |
| finite numeric score | 정렬 허용 |
| NaN/Inf/boolean score | 종료 코드 2 |
| exact 파일 생성 및 validator 0 | 캠페인 valid |
| exact 파일 누락/빈 파일 | 종료 코드 2 |
| validator non-zero | 종료 코드 2 |
| valid + target 달성 | 종료 코드 0 |
| valid + target 미달 | 종료 코드 3 |
