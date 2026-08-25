# Screen → Exact 캠페인 채택 체크리스트

## 실행 전

- [ ] MASTER W16 기준이 동일 후보군·동일 출력 범위에서 측정됨
- [ ] 후보별 입력 경로가 고유함
- [ ] batch width가 64 이하임
- [ ] screen objective와 survivor 수가 사전에 고정됨
- [ ] exact validator와 정확도 허용 기준이 사전에 고정됨

## Screen

- [ ] 모든 row의 `physics_mode`가 `ga_screen_feedback_limited`임
- [ ] 모든 row의 `requires_exact_rerun`이 `true`임
- [ ] 비유한값과 실패 후보가 별도로 기록됨
- [ ] screen 결과가 최종 HDF5 또는 승인 결과로 복사되지 않음

## Exact survivor

- [ ] approximation 환경변수가 제거됨
- [ ] 모든 survivor를 원 입력에서 다시 실행함
- [ ] 모든 exact 출력이 비어 있지 않음
- [ ] HDF5 필수 dataset 검사를 통과함
- [ ] k-eff/CBC/AO/Fq/Fr 및 MASTER 비교 게이트를 통과함
- [ ] graph/arena/drive fallback이 0임

## 성능

- [ ] screen wall과 exact wall을 모두 포함함
- [ ] `effective_candidate_cases_per_hour = Ncandidate*3600/(screen+exact wall)`을 사용함
- [ ] MASTER W16 실측 cases/h를 분모로 사용함
- [ ] 콜드 스타트와 점유 런을 제거하는 기준이 사전에 정의됨
- [ ] 최소 3회 교차 실행의 중앙값을 사용함
- [ ] target 20×이면 4,320–4,360 effective cases/h 이상인지 확인함

## 품질

- [ ] objective별 Spearman 상관을 기록함
- [ ] top-k recall을 기록함
- [ ] exact survivor 성공률이 100%임
- [ ] 입력·실행 파일·XS SHA-256을 보존함
- [ ] `campaign_receipt.json`의 `valid=true`, `requires_exact_rerun=false`, `target_met=true`를 확인함

하나라도 충족하지 못하면 수십 배 처리량 주장을 채택하지 않는다.