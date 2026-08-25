# Screen → Exact 안전 경계

- 실행기는 shell command를 사용하지 않고 argv 배열로 RASBERY를 호출한다.
- 외부 validator만 명시적인 문자열 템플릿을 `shlex.split`으로 분리한다.
- 기존 output이 있으면 해당 stage 실행 직전에 삭제하여 stale 성공을 방지한다.
- exact 단계는 screen 전용 환경변수를 제거한다.
- candidate input SHA-256과 exact output SHA-256을 receipt에 기록한다.
- 중복 input 경로는 host registration 및 동시 파일 접근 충돌을 방지하기 위해 거부한다.
- timeout, non-zero return code, 누락 receipt, 잘못된 physics mode, 비유한 score, 누락 exact output은 모두 캠페인 무효다.
- 목표 처리량 미달은 물리 실패와 분리하되 CI에서는 실패로 사용할 수 있도록 종료 코드 3을 반환한다.
