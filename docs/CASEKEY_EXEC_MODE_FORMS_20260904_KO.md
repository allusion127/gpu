# 케이스 키 v2 — 실행 모드 · 실효 Xe 암 · 수축 마스크 (2026-09-04)

## 1. 구멍 (238에서 관측)

같은 덱 · 같은 환경변수로 돌린 단일 실행과 `--batch-mode 1` 실행이
**같은 case_key / env_digest** 를 발행하면서 **다른 궤적**으로 수렴했다.

    단일        digest 1f36e75dc00ed2b4 / outers 4377
    batch-mode 1 digest 4c663ff538b28f82 / outers 7087

원인은 `--batch-mode` 가 **argv 플래그**라는 것. 이 플래그가 Xe Anderson 채택
기본값을 뒤집는데(`src/Driver.h` `xeAndersonGate`, 단일=ON / 배치=OFF),
`RASBERY_XE_ANDERSON` 은 미설정이라 env 절반은 양쪽 모두 `~` 로 같았다.
그 위에 두 구멍이 더 있었다.

* `RASBERY_GPU_XE_TXN` 의 **실효** 상태(기본 ON, 스텁 빌드는 항상 OFF)도 미포함.
* 호스트가 **채굴한** 수축 마스크(WSL 0x6 / 238 0x7)는 어떤 환경 문자열로도
  표현되지 않는데 생산 산술의 반올림을 고른다 — 키가 전혀 보지 못했다.

## 2. 수정

`casekey::Provenance` 와 `payloadOf` 가 `code_sha` **뒤에** 네 줄을 추가한다.

    exec_mode    single | batch          (executionModeName())
    xe_anderson  on | off                (xeAnderson(), 실효 상태)
    xe_txn       on | off                (rasberyGpuXeTxnEnabled(), 실효 상태)
    forms        sha256(forms payload)   (아래 3절) 또는 ~

`kSchema` = `rasbery-case-key/v2`, `[RASBERY][CASE]` `schema_version` = **8**,
`[RASBERY][TRAJECTORY]` `schema_version` = **2**.

**기존 키는 전부 바뀐다.** 값이 아니라 줄이 늘었기 때문에 AA 기본값 단일 실행도
예외가 아니다. v1 캐시는 병합이 아니라 **재키잉** 대상이다.

**source 이름은 키에 접지 않는다.** `RASBERY_XE_ANDERSON` 은 이미 `kArmEnv`
항목이라 "누가 요청했는가"는 env 줄에 들어 있다. 그래서 배치 하니스
(`tools/run_single_gpu_batch.py` `DEFAULT_ENV` 가 `RASBERY_XE_ANDERSON=1` 을
명시적으로 내보낸다 — 238 영수증의 anderson_accepted ~38k 가 그것이다)의 배치
케이스와 단일 실행은 **`exec_mode` 한 줄만** 다르다. 실효 Xe 암이 같기 때문이다.
맨손 `--batch-mode 1` 만 `xe_anderson` 까지 갈라진다 — 그게 닫은 결함이다.

## 3. 수축 마스크 레지스트리

`src/GpuFormMask.h` 에 프로세스 전역 레지스트리. 두 리졸버
(`resolveFormMask` / `resolveCalibratedFormMask`)가 `(name, value, source,
mined_sound)` 을 한 번씩 등록한다. 영수증은 **삽입 순서**, 다이제스트는
**이름 정렬** 사본(스케줄링 사실이 키를 움직이면 안 되므로).

`Driver::primeFormMasks()` 가 케이스 키 계산 **직전에** 네 채널
(`xeFormMask`, `xeHostFormMask`, `thFormMask`, `streamFormMask`,
`cmfdOuterFormsRuntime`)을 해소한다. 그러지 않으면 evaluator 서버(한 프로세스
64 케이스)에서 1번 케이스는 빈 레지스트리로, 2번은 채워진 레지스트리로 키를
만든다. `formsPayloadFrozen()` 은 첫 호출에서 **래치**되고, 이후에 해소되는
채널은 stderr 경고로 남는다.

## 4. 새 영수증 필드

`[RASBERY][CASE]`: `exec_mode`, `xe_anderson`, `xe_anderson_source`,
`xe_txn`, `xe_txn_source`, `forms_digest`, `forms_pin`, `forms`(배열).
`[RASBERY][TRAJECTORY]`: `env` 는 그대로 원문 유지, 옆에 `resolved` 객체가
같은 필드를 싣는다(`armEnvJson()` 은 test_telemetry_neutrality 가 원문 보고로
묶어 두었으므로 그 안에 파생값을 넣지 않는다).

## 5. 새 노브

    RASBERY_FORMS_PIN = default | mined      (기본 mined = 기존 동작)

`default` 는 모든 채널을 **빌드 기본값**으로 강제하고 source 를
`pinned_default` 로 찍는다. 우선순위:

    채널별 env 오버라이드  >  FORMS_PIN=default  >  채굴값  >  빌드 기본값

호스트마다 채굴값이 갈리는 상황에서 "출하된 반올림"을 양쪽에서 얻는 스위치다.

## 6. 238 러너가 재검증할 것

1. 같은 덱 · 같은 env 로 단일 실행과 `--batch-mode 1` 을 돌려
   `[RASBERY][CASE]` 의 `case_key` 가 **달라지는지**, 그리고 `exec_mode` /
   `xe_anderson` 이 각각 `single`/`on`, `batch`/`off` 로 찍히는지.
2. 하니스 배치(`DEFAULT_ENV`, AA=1)와 단일 실행은 payload 가 `exec_mode` 한
   줄만 다른지 — `tools/case_key.py <deck> --exec-mode batch --payload` 와
   `--exec-mode single --payload` 를 diff.
3. `RASBERY_FORMS_PIN=default` 로 한 번 더 돌려 `forms_digest` 가 기본 실행과
   **다른지**, `forms` 배열의 모든 채널 source 가 `pinned_default` 인지.
4. `tools/case_key.py` 로 키를 재현할 때는 실행의 `forms_digest` 를
   `--forms-digest` 로 넘겨야 한다(채굴값은 도구가 유도할 수 없다).
