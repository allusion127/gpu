# Rod Depletion 구현 메모

이 문서는 제어봉 물질 depletion을 위해 사용하는 fine axial rod-state와
fluence 저장 흐름을 정리한다. Fuel burnup과 별개로, rod material은 누적
fluence `flux * time`을 좌표로 하는 CHIFFON delta XS를 사용한다.

## Fine Rod-State Mesh

rod-state mesh는 노심 전체의 횡방향 노드 구조를 그대로 사용하고, axial
방향만 내부 고정 fine division으로 세분화한다.

```text
fine axial planes = nz * rod_division
fine rod cells    = nxy * nz * rod_division
```

fine rod-state index는 다음과 같다.

```text
fine_k = k * rod_division + m
idx    = fine_k * nxy + l2d
```

여기서 `l2d`는 횡방향 노드 index, `k`는 coarse axial node index,
`m`은 coarse node `k` 내부의 fine axial subdivision index이다.

이 배치는 같은 fine axial plane의 모든 횡방향 노드가 연속하도록 만든다.
따라서 추후 axial rod depletion sweep이나 삽입 overlap 갱신을 할 때
자연스럽게 접근할 수 있다.

## 저장 필드

현재 `XSSet`은 fine rod-state 배열 두 개를 소유한다.

```text
_fine_rod_type   [nxy * nz * rod_division]
_fine_rod_fluence[nxy * nz * rod_division]
```

`_fine_rod_type`은 현재 제어봉 삽입 상태에서 다시 계산된다. 값이 0이면
해당 fine cell에 제어봉이 없다는 뜻이고, 양수이면 그 제어봉의 `ctype`을
뜻한다.

`_fine_rod_fluence`는 같은 fine cell layout으로 저장된다. Predictor 단계에서는
BOS flux로 provisional fluence를 전진시키고, corrector 단계에서는 BOS 값을
복구한 뒤 BOS/EOS 평균 절대 flux로 다시 전진시킨다.

## Rod Insertion Overlap

`XSSet::SetRod()`는 기존 coarse node 기반 `rod_fraction`을 계속 계산한다.
그 뒤 `RebuildFineRodOccupancy()`를 호출하여 같은 삽입 깊이 convention을
fine axial rod-state mesh에 적용한다.

삽입 깊이는 fuel 영역 상단에서 아래 방향으로 잰다. 각 rod group과
transverse rod-map node에 대해, fine axial cell의 top이 rod tip보다
아래에 있으면 그 fine cell을 제어봉 점유 상태로 표시한다.

```text
fine_top > rod_tip
```

이는 기존 coarse overlap convention과 연결된다.

```text
overlap      = node_top - max(rod_tip, node_bot)
rod_fraction = clamp(overlap / node_height, 0, 1)
```

따라서 fine rod-state 저장소는 현재 coarse rod-fraction 방식과 미래의
fine rod-material depletion 모델 사이의 직접적인 연결부 역할을 한다.

## Rod XS 보정

CHIFFON 입력의 `rod depletion` HGC는 ctype별 `fluence -> delta XS` fitting
자료로 읽힌다. HGC의 burnup scalar 자리를 fluence 좌표로 사용하며, ctype별로
가장 작은 fluence 점을 기준 XS로 잡아 delta를 만든다.

RASBERY는 rodded node XS를 만들 때 현재 coarse node 안에서 같은 ctype을 가진
fine rod cells의 평균 fluence를 계산하고, 그 값으로 rod depletion delta XS를
추가한다. Rod cusping용 rodded XS도 같은 fluence 평균을 사용한다.

## 현재 제한

- Rod movement는 현재 geometry에서 occupancy를 다시 만들 뿐, 별도의
  rod-material inventory를 rod 방향으로 이동시키지 않는다.
- Fine rod fluence는 restart file의 `/fine_rod_fluence`로 저장/복구된다.

## 정리 사항

rod cusping과 rod depletion 스캐폴드는 수치 데이터 흐름을 한 번에 추적하기
쉽도록 정리했다.

- Fine rod indexing/storage용 one-off helper를 제거하고, overlap을 다시
  계산하는 루프 안에 fine rod-state index 식을 직접 드러냈다.
- Fine rod-state 배열의 unused public accessor를 제거했다. restart,
  diagnostics, depletion module에서 실제로 필요해질 때 다시 노출하는 편이
  낫다.
- Rod cusping FDM assembly와 homogenization loop의 작은 람다를 제거하고,
  matrix row와 SoA offset을 사용하는 루프 안에서 직접 계산하도록 바꿨다.
- Dense local linear solve, rodded/unrodded macro XS fetch, cusping stencil
  적용, fine rod occupancy rebuild, fine rod fluence update처럼 수치적
  경계가 분명한 helper만 남겼다.
