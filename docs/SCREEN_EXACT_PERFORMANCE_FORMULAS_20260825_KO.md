# Screen → Exact 처리량 계산식

후보 수를 `N`, screen wall을 `Ts`, exact survivor wall을 `Te`, MASTER W16 처리량을 `Pm`이라고 한다.

```text
Ttotal = Ts + Te
Peffective = N * 3600 / Ttotal
SpeedupW16 = Peffective / Pm
```

예를 들어 MASTER W16이 217 cases/h일 때 20배 게이트는 다음과 같다.

```text
Peffective >= 20 * 217 = 4,340 cases/h
```

survivor 수를 `K`라고 하면 exact survivor 자체의 관측 처리량은 다음과 같이 별도 기록한다.

```text
Pexact_survivor = K * 3600 / Te
```

`Peffective`는 N개 후보 모두를 정확 계산한 속도가 아니라, N개를 순위화하고 K개를 정확 재계산하는 설계 탐색 파이프라인의 처리량이다. 이 둘을 같은 지표로 보고하지 않는다.