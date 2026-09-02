# WP21-B/C — flatxs 스토어와 nodal 로드의 코얼레싱: 인벤토리와 **귀속된 미변환**

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | `kernelFlatXsCta`의 스토어(st 25.2) · `kNodalJnet<0>`의 로드(ld 16.7) |
| 근거 | 238 ncu 프로파일, `E:\rasbery_runs\2026-08-30\238\pricing_388e8f2.md` **블록 39** |
| 기준 덱 | KNGR `kngr_238.json`, `nxyz = 8,451`, `nsurf = 26,692`, `NG = 2` |
| 게이트 등급 | **B0** — 이 WP는 **주소를 하나도 바꾸지 않는다**. 규약에 이름을 붙이고, 수신증에 적고, 계약으로 고정한다 |
| 판정 | digest **`1f36e75dc00ed2b4` / `4377`** 불변 + `h5diff -c` 0 + 핀 CSV Δ=0 |
| 계약 테스트 | `tools/test_micx_layout_contract.py` · `tools/test_nodal_soa_contract.py` (순수 python, negative control 각 9/8종) |
| 소스 | `src/FlatXsKernel.h` · `src/FlatXsCtaKernel.cuh` · `src/CudaXsReconBackend.cu` |
| 결론 한 줄 | **WP21-B의 전제가 뒤집혀 있었다.** 블록은 이미 SoA였고, 25.2는 레이아웃 결함이 아니라 **CTA-per-node라는 병렬화 축과 SoA의 구조적 불일치**다 |

> **이 문서가 주장하지 않는 것부터.** WP21-A가 CMFD에서 한 일(순수 순열, 70개 인덱스
> 지점)을 여기서 되풀이하지 않았다. **되풀이할 대상이 없었기 때문이다.** flatxs 블록은
> 이미 노드-최내측이고, nodal의 노드-메이저 배열들은 **이 캠페인이 편집할 수 없는 파일**
> (`src/NodalKernel.h`의 host/device 공유 본문과 그 호스트 참조 `src/Nodal.cpp`)에 인덱스
> 지점을 전부 두고 있다. 그래서 이 WP의 산출물은 **인벤토리·귀속·고정**이고, 미변환은
> 숨기지 않고 §4에 귀속해 둔다.

---

## 1. 관측 — 블록 39가 남긴 두 개의 최악값

`ncu --launch-skip 10 --launch-count 5`, 커널별, `.ratio` 서브메트릭
(8바이트 접근의 이상값 = 2, WP21-A §2와 같은 8-스레드 wavefront 모델):

| 커널 | ld sectors/req | st sectors/req | dram %peak | occupancy | 그리드 |
|---|---:|---:|---:|---:|---|
| `kernelFlatXsCta` | 7.8 | **25.2** | **23.1** | 62.4 % | nxyz CTA × 128 스레드 |
| `kNodalJnet<0>` | **16.7** | 7.7 | — | — | nsurf 스레드 |

`kernelFlatXsCta`는 **블록 39에서 유일하게 bandwidth를 실제로 먹는 커널**이다(WP21-A의
CMFD 다섯 커널은 dram 1~7 %). `kNodalJnet`의 16.7은 트리 전체 최악 로드다.

---

## 2. WP21-B — `kernelFlatXsCta`의 st 25.2

### 2.1 전제가 뒤집혀 있었다

캠페인 브리프는 이 커널을 **thread-per-node**로, 블록을 **`[node][iso][xs]…` 노드-메이저
1,107 double**로 기술하고, "노드-최내측 SoA로 바꾸라"고 지시했다. 트리를 읽으면 셋 다
사실이 아니다:

| 브리프 | 실제 (`4b00d09` 트리) |
|---|---|
| thread-per-node | `kernelFlatXsCta`는 **CTA-per-node**다. `blockIdx.x`가 노드, `threadIdx.x`가 **컴포넌트 서수 q** (`src/FlatXsCtaKernel.cuh` (P1)) |
| `[node][iso][xs]` 노드-메이저 | **이미 컴포넌트-메이저·노드-최내측**이다. `c*nxyz + l` (`FlatXsView`의 인덱싱 주석, `XSSet::micx()`, CRAM D2D fill 전부 동의) |
| 노드당 1,107 double | 노드당 **880**이다: `N_ACTIVE*(NG+NMIC) + NLSM + NMSM = 9*(2+78) + 4 + 156`. FP32 arm에서는 880 float |

즉 **브리프가 요구한 종착 상태는 이미 트리의 상태**였다.

### 2.2 그래서 25.2는 어디서 오는가 — 그리고 왜 순열로는 못 고치는가

CTA-per-node에서 노드 `l`은 CTA 전체에 고정이고 워프의 32 레인은 서수 `q, q+1, …`을 든다.
`c*nxyz + l` 아래에서 연속 레인의 주소는 **`nxyz * elem_bytes` 만큼 떨어진다** → 레인마다
자기 섹터 하나 → 요청당 최대 32섹터. 스캐터 단계는 노드당 880회가 **전부** 이 형태다.

모델과 실측:

| 접근 | 형태 | 레인 간 stride | 모델(섹터/요청) | 실측 |
|---|---|---:|---:|---:|
| 스캐터 (`fxsStoreMic/Msm/Lmp/Lsm`) | `c*nxyz + l`, c = 레인 | 8·nxyz B | ~32 (완전 분산) | **25.2** |
| 게더 (`fxsRefMic/…`) | 동일 | 8·nxyz B | ~32 | — |
| 델타 계수 (`cdata[(base+p)*NMIC + e]`) | e = 레인, **stride 1** | 8 B | **2** | — |

로드가 7.8로 "덜 나쁜" 이유가 여기 있다: 노드당 게더는 880회지만 **계수 로드는
`node_cnt × nord × 880`회**(KNGR 덱에서 노드당 수천 회)이고 그것들은 완전 연속이다.
평균이 계수 쪽으로 끌려간다. **스토어에는 그런 희석이 없다** — 스토어는 880회 전부가
분산 형태다. 모델이 관측을 재현한다.

**순열은 전치이고, 전치는 두 축 사이에서 제로섬이다.** CTA arm의 컴포넌트 워크를
코얼레스하는 유일한 순서는 노드-메이저 `l*880 + q`이고, 그러면:

| 소비자 | 병렬화 축 | `c*nxyz+l` (현재) | `l*880+c` (뒤집으면) |
|---|---|---:|---:|
| `kernelFlatXsCta` (CTA-per-node) | 컴포넌트 | **32** | 2 |
| `kernelFlatXs` (thread-per-node, 참조 arm) | 노드 | 2 | **32** |
| CRAM D2D fill/widen (WP15.1/WP20.1) | 노드 | 2 | **32** |
| Xe commit xs / xsrecon condense | 노드 | 2 | **32** |
| `XSSet::micx/refMicx/micxssm` (호스트) | 노드 | 캐시-친화 | 파괴됨 |
| D2H 다운로드 / H2D 재업로드 | — | 직선 복사 | **전치 커널 필요** |

중간값도 사지 않는다. 노드를 W개씩 묶는 타일 SoA `[l/W][c][l%W]`를 넣어 보면: W=8이면
CTA arm의 레인 간 stride는 64 B로 여전히 레인당 1섹터(개선 0)이고, W=2면 CTA arm이 16섹터로
절반이 되는 대신 thread-per-node 소비자가 8→16섹터로 두 배가 된다. **정확히 제로섬이다.**

CTA를 노드 V개로 타일링하는 안(레인 = `(q, j)`, j가 노드)은 스토어를 고치지만 (a) 델타
스트림이 노드마다 달라 워프가 V-way로 발산하고, (b) 지금 **로드의 대부분을 차지하는**
연속 계수 읽기가 stride V로 흩어지며, (c) 공유 워크스페이스가 V배(FP64 7,352 B → V=8에서
58 KB/CTA)가 되어 occupancy가 무너진다. **스토어 12배를 사려고 로드와 occupancy를 판다.**

### 2.3 그래서 무엇을 바꿨나 — 규약에 이름을 붙였다

`src/FlatXsKernel.h`에 `rasbery::flatxs::block_layout`을 넣었다. WP21-A의
`rasbery::cmfd_layout`과 같은 형태다:

```cpp
namespace block_layout {
constexpr bool kNodeInnermost = true;
constexpr int  kLayoutVersion = kNodeInnermost ? 2 : 1;   // 1 = AoS, 2 = SoA
inline const char* name() { return kNodeInnermost ? "soa" : "aos"; }
RASBERY_XSR_HD constexpr int elem(int nxyz, int l, int c, int per_node) {
    return kNodeInnermost ? c * nxyz + l : l * per_node + c;
}
RASBERY_XSR_HD constexpr int lmp(int nxyz,int l,int ig){ return elem(nxyz,l,ig,NG);   }
RASBERY_XSR_HD constexpr int lsm(int nxyz,int l,int e ){ return elem(nxyz,l,e ,NLSM); }
RASBERY_XSR_HD constexpr int mic(int nxyz,int l,int e ){ return elem(nxyz,l,e ,NMIC); }
RASBERY_XSR_HD constexpr int msm(int nxyz,int l,int e ){ return elem(nxyz,l,e ,NMSM); }
}
```

- **`int`를 유지한다.** WP20.1의 여덟 폭 접근자(`fxsRefMic` 등)가 `int e`를 받고 모든 호출
  지점이 `int`를 넘겼다. `long long`으로 넓히면 주소 계산 명령열이 달라진다 —
  이 WP는 주소를 안 바꾼다고 주장하므로 타입도 안 바꾼다. 최대 곱은
  `NMIC*nxyz = 78 × 8,451 = 659,178`.
- **AoS 가지를 살려 둔다.** WP21-A와 같은 이유: 레이아웃 질문을 이분 가능하게 만든다.
  단, 채택된 arm이 아니다 — 뒤집으려면 `XSSet.h`의 호스트 접근자와 CRAM D2D fill도 같이
  뒤집어야 하고, 계약 테스트가 그렇게 적어 둔다.
- **16개 호출 지점**이 헬퍼 경유로 바뀌었다(두 본문에 8개씩). 반환 식은 이전 인라인 철자와
  **문자 단위로 같다** — g++ 13.3 아래 host shim 실행으로 확인했다(§6.0).
- **`xs` / `xs_ssm` / `iden`은 헬퍼에 넣지 않았다.** 이 셋은 FP64 거시 권위이고 nodal 드라이브·
  CMFD 연산자·호스트가 같이 읽는다. 네 블록의 일부가 아니며, 헬퍼로 덮으면 "블록과 같이
  순열해도 된다"는 오해를 초대한다. 계약 테스트가 이 배제를 강제한다.

### 2.4 로드 경로 (ld 7.8) — WP13 `flatxs.inputs`는 이미 SoA다

브리프는 "노드별 좌표/분기 입력이 AoS일 것"이라고 봤다. 아니다:

| 입력 | 인덱스 | 노드당 | 판정 |
|---|---|---:|---|
| `wvfr` / `dmod` / `bppm` | `[l]` | 1 | **이미 이상적**(섹터 2). WP21-A의 `psi`/`vol`과 같은 형태 |
| `nodes` / `node_off` / `node_cnt` | `[i]` | 1 | 이미 이상적 |
| `stream_did` / `stream_x` / `stream_scale` | `[s]` | — | CTA arm에서 **블록-유니폼 브로드캐스트**(전 레인이 같은 주소) → 1~2섹터 |
| `deltas[did]` (`DeltaMeta`, int 5개) | AoS 구조체 | — | 역시 블록-유니폼 브로드캐스트. 치환이 사지 않는다 |
| `coeff_lmp/lsm/mic/msm` | `[row*N + e]`, e = 레인 | — | **stride 1, 이미 최적**. 이것이 CTA arm이 존재하는 이유다 |

**변환할 것이 없다.** ld 7.8은 §2.2의 게더(880회 분산)와 계수 로드(수천 회 연속)의
가중 평균이고, 게더 쪽은 스토어와 같은 전치 문제다.

---

## 3. WP21-B가 남긴 것 — 수신증과 계약

### 3.1 수신증

`src/CudaXsReconBackend.cu`가 프로세스당 한 번, 무조건 찍는다:

```
[RASBERY][MICX][LAYOUT] {"layout":"soa","layout_version":2,"elem_bytes":8,
                         "components_per_node":880,"cta_arm":1}
```

- **자기 태그다.** `Driver.h`가 `[RASBERY][MICX]`로 WP15 잔류 수신증을 찍는데 그 파일은
  이 캠페인의 것이 아니다. 세 번째 태그 세그먼트를 붙이면 `[RASBERY][MICX]` grep이 둘 다
  찾으면서 잔류 라인의 스키마는 건드리지 않는다.
- **캡처 키에는 안 넣었다.** WP21-A와 같은 판정: 레이아웃은 컴파일 타임 상수라 한 프로세스가
  두 값을 못 든다. 키에 넣으면 자기 자신과 비교하는 죽은 필드가 된다. 구분이 실제로 필요한
  곳 — 프로파일/digest/센서스를 **엉뚱한 커널 본문에 대고 읽는 것** — 은 수신증이다.
- `elem_bytes`가 같은 줄에 있는 이유: 주소는 **순서 AND 폭**이고 WP20.1이 폭을 arm으로 만들었다.

### 3.2 `tools/test_micx_layout_contract.py`

7개 규칙 + negative control 9종. 고정하는 것: 레이아웃의 단일 정의, 헬퍼가 **옛 식 그대로**일
것, 두 본문 모두 인라인 철자를 안 쓸 것, 여덟 폭 접근자가 여전히 블록의 유일한 철자일 것,
거시 배열이 헬퍼에 안 들어갈 것, `XSSet`의 호스트 접근자가 같은 순서일 것, 수신증이 리터럴이
아니라 `block_layout::` 심볼을 인용할 것.

가장 중요한 규칙은 세 번째다. 이 블록은 **주소로** CRAM 백엔드에 넘어가고
(`XSSet::FillCramMicDevice` → `CudaCramBackend`의 D2D fill), 열한 개 호스트 `_micx` 벡터에
앨리어스되고, 그 벡터로 D2H 내려오고, 세대 미스에 다시 올라간다. **한 소비자만 순서를 달리
알아도 유한하고 그럴듯하고 틀린 단면적**이 나오며 아무 데서도 오류가 안 난다.

---

## 4. WP21-C — `kNodalJnet<0>`의 ld 16.7

### 4.1 인벤토리 — 이 커널이 만지는 모든 디바이스 배열

스레드 = **서페이스 `ls`** (`const int ls = blockIdx.x*blockDim.x + threadIdx.x`).
노드 id는 `lkl = v.lklr[ls*NLR+0]`, `lkr = v.lklr[ls*NLR+1]`로 **게더**되고,
`lkd = lk*NDIR + idir`가 거의 모든 로드의 베이스다.

#### 4.1.1 노드-메이저 AoS — 이것이 16.7의 원인이다

| 배열 | 클래스 | 인덱스 | 노드당 | 레인 간 stride (lk 연속 시) | 모델 섹터 | jnet가 읽는 횟수 |
|---|---|---|---:|---:|---:|---:|
| `eta1` | updateConstant 상수 | `[(lk*NDIR+idir)*NG + ig]` | 6 | 48 B | **8~12** | 11 |
| `eta2` | 〃 | 〃 | 6 | 48 B | 8~12 | 5 |
| `diagD` | 〃 | 〃 | 6 | 48 B | 8~12 | 8 |
| `m251`, `m253` | 〃 | 〃 | 6 | 48 B | 8~12 | 각 1 |
| `trlcff1` | 워킹 | 〃 | 6 | 48 B | 8~12 | 11 |
| `dsncff2`, `dsncff4`, `dsncff6` | 워킹 | 〃 | 6 | 48 B | 8~12 | 각 10 |
| `mu`, `tau` | 워킹 (per-node-dir 행렬) | `[(lk*NDIR+idir)*NG2 + j*NG+i]` | 12 | 96 B | **8~16** | 18 / 20 |
| `matMI` | 워킹 (per-node 행렬) | `[lk*NG2 + j*NG + i]` | 4 | 32 B | 8 | 10 |
| `hmesh` | 지오메트리 | `[lk*NDIR + dir]` | 3 | 24 B | 8 | 5 |

이 아홉 클래스가 커널 로드의 대부분이다. **노드-최내측으로 옮기면**
(`(idir*NG+ig)*nxyz + lk` 등) 레인 간 stride가 8 B가 되어 모델 섹터가 **2**로 간다 —
`lk`가 `ls`에 대해 연속인 방향(X)에서. Y/Z 방향 서페이스에서는 `lk`가 `nx`/`nxy` 간격으로
뛰므로 게더로 남고, 순열의 이득은 방향별로 다르다. **이것이 이 변환의 기대치를 ~2가 아니라
~3으로 적는 이유다.**

#### 4.1.2 움직일 수 없는 것 — 그리고 그 이유

| 배열 | 인덱스 | 왜 그대로인가 |
|---|---|---|
| `flux` (= CMFD `phi`) | `[lk*NG + ig]` | **정본 핸드오프.** 공유 모드에서 이 포인터가 곧 CMFD 백엔드의 버퍼다(`GpuCanonicalState.h`: "the same bytes both backends already index, not a transposed copy"). **WP21-A가 CMFD 쪽 `phi`/`src`/Krylov를 `[2l+ig]`에 그대로 둔 것이 바로 이 규약을 지키기 위해서다.** 옮기려면 `CudaBICGBackend.cu`가 필요하고 그 파일은 이 캠페인의 것이 아니다 |
| `jnet`, `phis` | `[ls*NG + ig]` | 같은 이유 + `Geometry::Jnet`/`Phif`가 D2H 목적지다. 서페이스-인덱스이므로 이 커널에서는 이미 stride 16 B(모델 4)이고 최악값도 아니다 |
| `lklr`, `idirlr`, `sgnlr` | `[ls*NLR + side]` (int) | 서페이스-인덱스 지오메트리. `[side*nsurf + ls]`로 옮기면 4→2섹터를 사지만, 인덱스 지점이 `NodalKernel.h`에 있고(§4.2) 이득은 커널 로드의 몇 %다 |
| `albedo` | `[dir*NLR + side]` | 6개짜리 테이블. 워프 전체가 같은 주소 → 브로드캐스트 |
| `xsrf`/`xsnf`/`xssm`/`chif` | `[ig*nxyz + lk]` | **이미 그룹-메이저 SoA다.** flatxs 블록과 같은 규약을 공유한다 |

### 4.2 왜 변환하지 않았는가 — 귀속

세 가지가 겹친다. 셋 다 기술적 사실이고 셋 다 이 WP의 권한 밖이다.

1. **인덱스 지점이 전부 `src/NodalKernel.h`에 있다.** 그 파일은 **host/device 공유 본문**이고,
   `src/Nodal.cpp:884/958`이 자기 호스트 배열 위에서 **같은 함수들을 프로덕션 경로로 돌린다**.
   컴파일 타임 상수를 뒤집으면 호스트 arm이 자기 배열을 틀린 순서로 읽는다. WP21-A가 이
   문제를 겪지 않은 이유는 CMFD의 CPU 참조 솔버가 **다른 파일의 다른 함수**(`BICGSolver.cpp`)여서
   디바이스 쪽만 pack lane으로 치환하면 됐기 때문이다. 여기서는 그 분리가 없다.

2. **따라서 안전한 형태는 뷰가 stride를 나르는 것뿐이다** — `NodalViewT`에 클래스별
   `(a, b)` stride 쌍을 두고 인덱스를 `lk*a + c*b`로 쓰며, **구조체 기본값을 현재 AoS 값으로**
   두어 `Nodal::MakeView()`·리플레이 도구·`test/canonical_state.cpp`가 한 글자도 안 바뀌게 하고,
   디바이스 뷰 빌더만 SoA를 세운다. 분기가 아니라 IMAD 하나이므로 latency-bound 커널에서는
   무시할 만하다. **그러나 그것은 `NodalKernel.h`의 모든 인덱스 지점을 고치는 일**이고,
   그 파일은 이 작업 패키지에 배정되지 않았다.

3. **그리고 소비자 정리가 남는다.** 아홉 상수는 세대 변화마다 H2D로 올라오고
   (`Nodal`의 호스트 배열 → pack lane 필요), `trlcff0`/`trlcff2`/`matM`은 하이브리드 arm
   (`RASBERY_GPU_NODAL_FULL` 미설정)에서 드라이브 도중 D2H로 `Nodal`의 FP64 호스트 배열로
   내려간다 — 그 세 복사도 unpack lane이 필요하다. FP32 arm의 좁힘 커널은 원소별이라 무관하고,
   아레나의 dense rebase(`nodalSlotView`)는 배열 **개수**만 쓰므로 순열에 무관하다.

**컴파일러도 GPU도 없는 세션에서 이 셋을 한꺼번에 하는 것은 조용한 인덱스 오류의 정확한
조건이다.** 이 트리가 가장 두려워하는 실패 형태 — 유한하고 그럴듯하고 틀린 — 이고,
digest가 그것을 잡더라도 원인 이분은 인덱스 지점 수십 개 위에서 해야 한다. 그래서
**미변환으로 귀속하고, 변환 계획을 위에 적어 둔다.**

### 4.3 그래서 무엇을 바꿨나 — 두 순서를 수신증에 적었다

`[RASBERY][NODAL][GPU]`에 두 필드가 붙었다:

```
,"canonical_layout":"element","private_layout":"node_major"
```

- `canonical_layout`은 **양면 불변식**이다. `"element"`는 `flux [lk*NG+ig]` /
  `jnet·phis [ls*NG+ig]`를 뜻하고, WP21-A가 CMFD 쪽을 `[2l+ig]`에 남긴 것과 **짝을 이룬다.**
  한쪽만 움직이면 두 백엔드가 같은 바이트를 다른 지도로 읽는다.
- `private_layout`은 **미변환의 상설 진술**이다. `"node_major"`가 찍혀 있는 한 §4.1.1의 16.7은
  설명된 잔차이고, 누군가 변환하면 이 문자열이 먼저 바뀐다.

### 4.4 `tools/test_nodal_soa_contract.py`

WP21-A와 WP21-C 사이의 **핸드오프 규약**을 양쪽에서 고정한다. 이것이 현재 아무도 지키지 않는
실질적 불변식이다: WP21-A는 `cc`/`diag`/`neighbors`/`face_area`를 SoA로 옮기면서
`phi`/`dtil`/`dhat`를 **의도적으로** 남겼는데, 그 이유가 `docs/WP21_A_...` §2.2에만 적혀 있고
nodal 쪽에서는 아무 데도 강제되지 않는다. 누가 나중에 `phi`를 순열하면 nodal은 그것을
`flux[lk*NG+ig]`로 읽는다 — 조용히.

고정하는 것: 정본 세 영역의 인덱스 형태(양쪽 파일에서), `cmfd_layout` 헬퍼가 정본 배열을
**덮지 않을 것**, `canonicalFromSlotView`의 레이아웃 주석이 살아 있을 것, nodal 수신증의 두
필드, 그리고 §4.1.1의 미변환 인벤토리가 이 문서에 남아 있을 것.

---

## 5. 왜 B0인가 — 논증

WP21-A의 다섯 조건보다 짧다. **주소를 하나도 안 바꿨기 때문이다.**

1. **값이 안 바뀐다.** 읽히는 double은 같은 주소의 같은 double이다.
2. **식이 안 바뀐다.** `block_layout::mic(nxyz, l, e)`는 `e * nxyz + l`로 **인라인 확장되는
   constexpr**이고, 반환 타입도 호출 지점의 타입도 그대로 `int`다. 산술 표현식은 한 글자도
   재타이핑되지 않았다 — `pol.ma(...)` 사이트는 전부 손대지 않았고 `FLATXS_FORMS`/`NODAL_FORMS`
   마이닝 마스크가 기술하는 코드가 그대로 돈다.
3. **누산 순서가 안 바뀐다.** (P1) 레인 소유권, (P2) 동위원소 폴드, (P3) 유니폼 재계산 전부 불변.
4. **전송이 안 바뀐다.** pack lane이 없다 → `[RASBERY][XFER]` 총계가 바이트/호출 모두 불변.
5. **새 코드는 수신증 두 줄과 문자열 상수 둘뿐**이고, 둘 다 solve 경로 밖(정적 소멸 시점과
   드라이브 카운터 옆)이다.

그래서 이 변경은 **플래그 뒤에 두지 않았다.** 게이트가 곧 증명이다.

---

## 6. 238 runbook

로컬에 nvcc가 없다. **첫 관문은 238 컴파일이다.** GPU0만 사용한다.

```bash
export CUDA_VISIBLE_DEVICES=0
V6_ENV='RASBERY_PPR_MODE=master RASBERY_PC_MODE=decart RASBERY_GPU=1
RASBERY_GPU_CMFD_SWEEP=1 RASBERY_GPU_CMFD_RESIDENT_SINGLE=1 RASBERY_GPU_NODAL=1
RASBERY_GPU_NODAL_FULL=1 RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1
RASBERY_GPU_OUTER=1 RASBERY_GPU_OUTER_SEGMENT_MAX=8 RASBERY_GPU_WIEL_FOLD=chunked
RASBERY_GPU_XE=1 RASBERY_STAGED_FLUX_TOL=50 RASBERY_STAGED_XE_TOL=1000
RASBERY_STAGED_LOOSE_SETTLE=1 RASBERY_OMP_THREADS=12 RASBERY_GPU_CRAM=1
RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1 RASBERY_GPU_CMFD_FUSE=15
RASBERY_GPU_XE_TXN=1 RASBERY_RESULT_ASYNC=1 RASBERY_GPU_FLATXS_CTA=1
RASBERY_GPU_OUTER_GRAPH=1 RASBERY_GPU_MICX_RESIDENT=1 RASBERY_GPU_XFER_ELIDE=1
RASBERY_GPU_OUTER_SEGMENT_V2=1 RASBERY_GPU_CMFD_BLOCK=64'
```

### 6.0 선행 — 계약 게이트 (로컬에서 이미 통과)

```bash
python3 tools/test_micx_layout_contract.py
python3 tools/test_nodal_soa_contract.py
python3 tools/test_micx_resident_contract.py
python3 tools/test_gpu_fp32_contract.py
python3 tools/test_xfer_ledger_contract.py
python3 tools/test_flatxs_cta_contract.py
python3 tools/test_enum_alias_contract.py
python3 tools/test_dependent_template_contract.py
python3 tools/test_cram_gpu_contract.py
python3 tools/test_cmfd_soa_contract.py
python3 tools/test_capture_standup_isolation_contract.py
ctest   # 12/12
```

헬퍼가 옛 식과 **문자 단위로 같다**는 주장은 g++ 13.3 host shim으로 이미 실행되었다:
`block_layout::{lmp,lsm,mic,msm}(8451, l, c) == c*8451 + l`을 네 컴포넌트 공간 전체와
`l = 0,977,…`에 대해 전수 비교, 0 불일치. 238에서 실행 가능한 증명은
`test/flatxs_device_replay.cu --cta`(0-ULP)다.

### 6.1 digest 불변 (**이것이 먼저다**)

```bash
env $V6_ENV <production arm> ... -o "$OUT/a_bc"
```

합격 조건 — **전부**:

- digest **`1f36e75dc00ed2b4` / `4377`**
- 직전 v6 산출물 대비 `h5diff -c` **0 차이**
- `[RASBERY][MICX][LAYOUT]`가 `"layout":"soa","layout_version":2`
- `[RASBERY][NODAL][GPU]`가 `"canonical_layout":"element","private_layout":"node_major"`
- `[RASBERY][NODAL_ARENA]`가 `"canon_locked":1` (WP19.2b)
- `[RASBERY][XFER]` 총계가 **바이트/호출 모두 불변**
- CSV(핀 파워/AO/keff) 전 항목 Δ = 0

**digest가 움직이면 이 WP는 순열조차 아니었다는 뜻이다** — 주소를 안 바꿨다고 주장했으므로
움직일 이유가 없고, 움직였다면 `block_layout::elem()`의 인라인이 의심 1순위다.
`kNodeInnermost = false`로 빌드하면 트리가 노드-메이저 주소로 가므로(호스트 접근자와 CRAM
fill도 같이 뒤집어야 한다) 이분은 가능하지만 **그것은 지원되는 arm이 아니다.**

### 6.2 ncu 재측정 — 블록 39와 같은 디렉티브

```bash
for k in kernelFlatXsCta kernelFlatXs kNodalJnet; do
  ncu --kernel-name "regex:$k" --launch-skip 10 --launch-count 5 \
      --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
      --csv env $V6_ENV <production arm> ... > "$OUT/ncu_$k.csv"
done
```

**기대치를 정직하게 적는다:**

| 커널 | ld 전 | **ld 후 기대** | st 전 | **st 후 기대** |
|---|---:|---:|---:|---:|
| `kernelFlatXsCta` | 7.8 | **7.8 (불변)** | 25.2 | **25.2 (불변)** |
| `kNodalJnet<0>` | 16.7 | **16.7 (불변)** | 7.7 | **7.7 (불변)** |

**이 표의 전부가 "불변"이라는 것이 이 측정의 요점이다.** 이 WP는 주소를 안 바꿨으므로
섹터가 움직이면 뭔가 다른 일이 일어난 것이고, 그때는 §6.1의 digest가 이미 깨져 있어야 한다 —
두 게이트가 서로를 검증한다.

**§2.2가 예측하는 값은 별도 arm에서만 나온다.** `RASBERY_GPU_FLATXS_CTA` 없이 돌리면
`kernelFlatXsCta` 대신 thread-per-node `kernelFlatXs`가 뛰고, 그 커널의 스토어는 같은
바이트를 **st ~2**로 쓴다. 그것이 §2.2의 제로섬 주장에 대한 **실측 확인**이다:

```bash
env $V6_ENV RASBERY_GPU_FLATXS_CTA=1 <arm> ... -o "$OUT/c_cta"     # 기준
env ${V6_ENV/RASBERY_GPU_FLATXS_CTA=1/} <arm> ... -o "$OUT/c_thread"  # CTA off
```

합격 조건: 두 arm의 digest 동일 + `h5diff -c` 0 (CTA arm은 B0로 설계되었다) + `kernelFlatXs`의
**st ≈ 2**. 그러면 25.2의 귀속이 관측으로 지지되고, **CTA arm을 채택할지**가 처음으로
측정 가능한 질문이 된다(공유 워크스페이스 occupancy 對 스토어 섹터). 그 A/B는 이 WP의
산출물이 아니라 그 다음 작업 패키지의 것이다.

### 6.3 단일덱 wall — 워밍업 1 + hot 3

```bash
for r in w 1 2 3; do env $V6_ENV <production arm> ... -o "$OUT/b_$r"; done
```

**기대치: 0.** 주소를 안 바꿨다. 이 줄이 움직이면 측정 노이즈이거나 §6.1이 깨진 것이다.

### 6.4 배치 — 기준선 확인

```bash
python3.11 tools/run_multi_gpu_batch.py --set "$V6_ENV" ...   # 8 procs x M16 + MPS
```

기준: **1,321 c/h**(블록 37/38의 클린 V2 baseline 1,320.8~1,326.7). **기대치도 1,321이다.**
이 WP는 성능 변경이 아니다 — 이 줄은 회귀가 없다는 것만 말한다.

---

## 7. 미착수 — 다음 작업 패키지가 손댈 곳

1. **`NodalKernel.h`의 stride-in-view 변환** (§4.2의 형태 2). 아홉 상수 + 열두 워킹 배열을
   노드-최내측으로. 기대치 `kNodalJnet` ld **16.7 → ~3**, 그리고 같은 순열이
   `kNodalTrl0`/`kNodalTrl12`/`kNodalMat`/`kNodalEven`/`kNodalMatEven`(전부 thread-per-node)의
   로드도 같이 고친다 — jnet보다 이쪽이 더 큰 몫일 수 있다. 필요한 것:
   `NodalKernel.h` + `Nodal.cpp`(또는 pack lane) + 아홉 상수의 H2D + 하이브리드 arm의 세 D2H.
2. **`kernelFlatXsCta` 채택 재판정** (§6.2의 c_cta 對 c_thread). 25.2가 구조적이라면 질문은
   "고칠 수 있나"가 아니라 "이 arm이 값을 하나"이고, 그 답은 아직 측정되지 않았다.
3. **`lklr`/`idirlr`/`sgnlr`의 서페이스-SoA** (§4.1.2). 4→2섹터, 작다. 1번과 같은 파일에서
   같이 하는 것이 맞다.
4. **WP21-A가 남긴 리덕션의 6.55/7.05** — 여전히 미해결. 이 WP는 손대지 않았다.
