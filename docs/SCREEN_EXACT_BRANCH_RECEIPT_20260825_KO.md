# Screen → Exact MASTER W16 처리량 브랜치 적용 영수증

작성일: 2026-08-25
브랜치: `codex/screen-exact-master-w16-throughput`
기준: `codex/cmfd-gpu-assembly-drive-fusion-v2`

## 추가 파일

- `tools/run_screen_exact_campaign.py`
- `tools/test_screen_exact_campaign.py`
- `tools/README_SCREEN_EXACT.md`
- `docs/EXACT_AND_SCREENING_THROUGHPUT_STRATEGY_20260825_KO.md`
- `docs/SCREEN_EXACT_CAMPAIGN_USAGE_20260825_KO.md`
- `docs/SCREEN_EXACT_ACCEPTANCE_CHECKLIST_20260825_KO.md`
- `docs/SCREEN_EXACT_RECEIPT_SCHEMA_20260825_KO.md`
- `docs/superpowers/plans/2026-08-25-screen-exact-master-w16-throughput.md`
- `.github/workflows/screen-exact-throughput-contracts.yml`

## 코드 계약

1. Screen 결과는 순위화에만 사용한다.
2. Screen receipt가 근사 물리 모드와 exact 재계산 의무를 표시하지 않으면 즉시 실패한다.
3. Survivor는 원 입력에서 exact 경로로 다시 실행한다.
4. Approximation 환경변수는 exact 단계에서 제거한다.
5. exact 출력과 선택적 validator가 모두 성공해야 캠페인이 유효하다.
6. batch width는 검증된 상한 64를 넘을 수 없다.
7. 같은 입력 경로를 여러 슬롯에서 동시에 사용하지 못한다.
8. MASTER W16 대비 배수는 screen과 exact wall을 모두 포함한다.
9. 목표 미달과 물리 캠페인 무효를 서로 다른 종료 코드로 구분한다.

## CI 계약

- Python syntax compile
- fake RASBERY를 이용한 screen→exact end-to-end 실행
- 상위 survivor 선택
- exact 출력 생성 확인
- 목표 미달 종료 코드 확인
- 중복 input fail-closed 확인
- `git diff --check`

## 서버 게이트

실제 수십 배 주장을 채택하기 전 다음을 수행한다.

- 동일 후보군의 MASTER W16 처리량 측정
- screen과 exact를 포함한 최소 3회 교차 실행
- exact survivor 100% 성공
- HDF5 및 MASTER 비교 게이트 통과
- objective Spearman 상관과 top-k recall 보고
- fallback 및 비유한값 0
- `campaign_receipt.json`의 `valid=true`, `requires_exact_rerun=false`, `target_met=true` 확인

이 브랜치는 단일 후보의 exact 계산이 MASTER W16보다 수십 배 빠르다고 주장하지 않는다. 다수 설계 후보를 screen하고 survivor를 정확 재계산하는 전체 파이프라인의 실효 후보 처리량을 재현 가능하게 산정한다.