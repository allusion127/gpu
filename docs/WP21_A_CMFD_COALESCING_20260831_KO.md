# WP21-A — CMFD/BiCG 노드-최내측(SoA) 레이아웃: 언코얼레스드 접근의 제거

## 문서 메타데이터

| 항목 | 값 |
|---|---|
| 대상 | CMFD 연산자 배열의 저장 순서 (`diag`/`dinv`/`udiag`/`cc` + `neighbors`/`node_surface`/`face_area`) |
| 플래그 | **없음 — 무조건**. 레이아웃은 컴파일 타임 상수 `cmfd_layout::kNodeInnermost` |
| 게이트 등급 | **B0** — 값도 연산도 연산 순서도 움직이지 않는다. 저장 주소만 바뀐다 |
| 판정 | digest **`1f36e75dc00ed2b4` / `4377`** 불변 + `h5diff -c` 0 차이 |
| 계약 테스트 | `tools/test_cmfd_soa_contract.py` (순수 python, negative control 9종) |
| 동반 수정 | `tools/test_cmfd_fuse_contract.py`(bit 4 추가로 default/노드 모델 갱신) · `tools/test_cmfd_assembly_kernel.py`(비교를 레이아웃 헬퍼 경유로) |
| 소스 | `src/CmfdAssemblyKernel.h` · `src/CudaBICGBackend.cu` |
| 근거 | 238 ncu 프로파일, `E:\rasbery_runs\2026-08-30\238\pricing_388e8f2.md` **블록 39 (a2)** |
| 기준 덱 | KNGR `kngr_238.json`, `nxyz = 8,451`, `NG = 2`, `n = 16,902` |
| 부수 항목 | FUSE **bit 4 (`kFuseNorm`)** — 잔차 노름 stage 2 접기. **기본 OFF**, 가격 미측정 |

> **이 문서가 주장하지 않는 것부터.** 블록 39는 이 커널들이 전부 **latency-bound**라고
> 못박았다 — dram은 피크의 1~7 %, warps active는 grid 133×64에서 **4 %**. 8,451 노드는
> 188 SM에 비해 그냥 작다. 따라서 **코얼레싱을 고쳐도 단일덱 wall은 크게 안 움직인다.**
> 다섯 커널의 GPU 시간 합은 단일 실행에서 ~2.4 s이고, 섹터 수를 줄여도 지연은 그대로다.
> 이 변경이 실제로 값을 내는 곳은 **배치**다: 8개 MPS 클라이언트가 같은 L2/DRAM 섹터를
> 두고 경합할 때, 요청당 섹터 수는 곧 **경합량**이다. 그래서 채택 판정선은 단일 wall이
> 아니라 8×M16 c/h(기준 1,321)다.

---

## 1. 관측 — 블록 39가 무엇을 쟀는가

`ncu --launch-skip 10 --launch-count 5`, 커널별, `.ratio` 서브메트릭 (8바이트 접근의 이상값 = 2):

| 커널 | ld sectors/req | st sectors/req | dram %peak | warps active %peak | grid |
|---|---:|---:|---:|---:|---:|
| `colored_block_sweep` | **6.55** | **6.90** | 4.44 | 4.03 | 133 |
| `matvec_two_group` | **9.92** | **6.38** | 6.93 | 4.14 | 133 |
| `prepare_p_jacobi` | **6.33** | **6.38** | 3.24 | 4.15 | 133 |
| `reduce_dot_stage1` | **6.55** | 1.00 | 1.40 | 16.5 | 17 |
| `reduce_dot2_fused` | **7.05** | 1.00 | 1.59 | 14.5 | 17 |
| `update_solution` | 2.27 | 3.19 | 2.90 | 5.87 | 265 |

`update_solution`만 정상이다. 그리고 그것이 **유일하게 원소 인덱스 `i`로 도는 커널**이다
(`i = blockIdx.x*blockDim.x + threadIdx.x`, `i < n`). 나머지는 전부 **노드 인덱스 `l`**로 돌고
(`docs/WP17_CMFD_OCCUPANCY_20260830_KO.md` §partition, 원소 `i = 2l+ig`), 그 커널들이 읽는
배열은 전부 **노드-메이저**로 패킹되어 있었다. 이것이 상관이 아니라 **원인**이라는 것이 이
문서의 주장이다.

---

## 2. 인벤토리 — 다섯 커널이 만지는 모든 디바이스 배열

측정 모델: ncu가 64비트 전역 접근을 **8스레드 단위(wavefront)** 로 쪼개므로 요청당 이상값은
`8×8 B = 64 B = 2섹터`다. 아래 "예상 섹터"는 **연속 8스레드가 닿는 서로 다른 32 B 섹터 수**다
(32비트 `int` 접근은 16스레드/요청이므로 이상값 2, 최대 16).

### 2.1 변환한 것 — 노드-메이저였고, 이제 컴포넌트-메이저다

| 배열 | 원소/노드 | 이전 인덱스 | 스레드 간 stride | 예상 섹터 (전) | 새 인덱스 | 예상 섹터 (후) |
|---|---:|---|---:|---:|---|---:|
| `diag`, `dinv`, `udiag` | 4 | `[4l+k]` | 32 B | **8** | `mat(nxyz,l,k) = k*nxyz+l` | **2** |
| `cc` | 12 | `[12l+j]` | 96 B | **8** | `cpl(nxyz,l,j) = j*nxyz+l` | **2** |
| `diag_f`, `dinv_f`, `cc_f` | 4 / 12 | 위와 동일 | 32 / 96 B | **8** | 위와 동일 | **2** |
| `neighbors` | 6 (`int`) | `[6l+s]` | 24 B | **12** | `face(nxyz,l,s) = s*nxyz+l` | **2** |
| `assembly_node_surface` | 6 (`int`) | `[6l+f]` | 24 B | **12** | `face(...)` | **2** |
| `assembly_face_area` | 3 | `[3l+d]` | 24 B | **8** | `dir(nxyz,l,d) = d*nxyz+l` | **2** |

### 2.2 일부러 **안** 건드린 것 — 그리고 그 이유

| 배열 | 원소/노드 | 인덱스 | 스레드 간 stride | 섹터 | 왜 그대로인가 |
|---|---:|---|---:|---:|---|
| `phi`, `src`, `r`, `r0`, `p`, `v`, `s`, `t`, `y`, `z`, `ax` (+ `_f` 쌍둥이) | 2 | `[2l+ig]` | 16 B | **4** | **리덕션 피연산자.** dot은 원소 인덱스 `i ∈ [0,n)` 위에서 `chunk = ceil(n/gridDim.x)`로 고정 분할되고 256-lane 고정 트리로 접는다. 벡터를 치환하면 **분할이 다시 쪼개지고 덧셈 순서가 움직인다 — B0가 아니라 N1이다.** 배치 비트동일성이 통째로 이 한 줄 위에 서 있다 |
| `phi` | 2 | `[l*ng+ig]` | 16 B | **4** | 추가로 **포인터 핸드오프**다. `CudaBatchArena::CmfdResidentView`가 nodal/PPR/outer-segment에 그대로 넘긴다(`CudaBICGBackend.h`의 "WHY THIS IS A HANDOFF AND NOT A COPY"). 바꾸면 잠긴 파일의 소비자를 다 고쳐야 하고, 그 대가로 섹터 4→2 하나를 산다 |
| `dtil`, `dhat` | — | `[ls*ng+ig]` | — (게더) | — | 서페이스 id로의 **비정형 게더**다. 노드 stride가 아니므로 치환이 사지 않는다. 그리고 이것도 `CmfdResidentView`의 핸드오프 대상이다 |
| `psi`, `vol` | 1 | `[l]` | 8 B | **2** | 이미 이상적이다 |
| `colors` | 1 (`int`) | `[l]` | 4 B | **2** | 이미 이상적이다 |
| `chif`, `xsnf`, `xsrf`, `xssm` | — | `[ig*nxyz+l]` | 8 B | **2** | **이미 그룹-메이저 SoA다.** 이 캠페인이 고치려던 바로 그 형태였다 |
| `scalars` | — | `[m*kScalarCount+slot]` | 0 (브로드캐스트) | **1~2** | 워프 전체가 같은 주소를 읽는다 |

### 2.3 대조 — 모델 대 실측

| 커널 | 모델(ld) | 실측(ld) | 모델(st) | 실측(st) | 변환 후 모델(ld) |
|---|---:|---:|---:|---:|---:|
| `matvec_two_group` | 7.3 | 9.92 | 4.0 | 6.38 | **2.9** |
| `colored_block_sweep` | 7.1 | 6.55 | 4.0 | 6.90 | **2.9** |
| `prepare_p_jacobi` | 5.6 | 6.33 | 4.0 | 6.38 | **3.4** |

**로드 열은 모델이 ±15 % 안에서 맞는다.** 그러므로 "AoS stride가 원인"이라는 귀속은
관측으로 지지된다. **스토어 열은 모델이 1.5~1.7배 과소예측한다.** 정직하게 남긴다:

- `colored_block_sweep`은 설명이 있다. `colors[l] != target_color`가 워프의 절반을
  되돌려보내므로, **활성 8레인이 16노드 폭(256 B)에 흩어진다** → 8섹터. 실측 6.90과
  맞는 방향이다. **그래서 이 커널의 스토어는 SoA 이후에도 2로 가지 않는다** — 색 필터가
  만드는 부분워프는 레이아웃이 아니라 색칠의 성질이다. 기대치는 ~4다.
- `prepare_p_jacobi`/`matvec_two_group`의 스토어 잔차는 **미귀속**이다. 두 커널 다 색
  필터가 없고 저장은 `[2l+0]`/`[2l+1]` 둘뿐이라 모델상 4가 상한이다. ncu의 스토어 요청
  granularity이거나 write-allocate 섹터일 수 있으나 **추정으로 적지 않는다.** 변환 후
  ncu가 이 열을 어떻게 움직이는지가 §6.4의 관측 항목이다.

`reduce_dot_stage1`의 ld 6.55 / `reduce_dot2_fused`의 7.05도 **미해결로 남긴다.** 이 커널들은
연속 스레드가 `am[begin + tid]`로 **완전 연속**을 읽는다(모델 2). `begin = blockIdx.x * chunk`,
`chunk = 995`가 32의 배수가 아니라 정렬이 어긋나는 것만으로는 6.55가 나오지 않는다.
**그리고 §2.2가 말하듯 이 커널들은 손댈 수 없다** — 피연산자를 치환하면 합이 달라진다.
WP21-B 이후의 후보로 남긴다.

---

## 3. 무엇이 바뀌었나

### 3.1 레이아웃의 유일한 정의 — `src/CmfdAssemblyKernel.h`

```cpp
namespace rasbery::cmfd_layout {
constexpr bool kNodeInnermost = true;
constexpr int  kLayoutVersion = kNodeInnermost ? 2 : 1;   // 1 = AoS, 2 = SoA
inline long long component(int nxyz, int l, int k, int per_node) {
    return kNodeInnermost ? (long long)k * nxyz + l : (long long)l * per_node + k;
}
inline long long mat (int nxyz,int l,int k){ return component(nxyz,l,k, 4); }
inline long long cpl (int nxyz,int l,int j){ return component(nxyz,l,j,12); }
inline long long face(int nxyz,int l,int f){ return component(nxyz,l,f, 6); }
inline long long dir (int nxyz,int l,int d){ return component(nxyz,l,d, 3); }
}
```

**컴파일 타임 상수인 이유.** 런타임 스위치는 (a) 모든 연산자 접근마다 분기를 얹고,
(b) 한 프로세스가 두 레이아웃을 동시에 들 수 있게 만든다. 컴파일 타임이면 두 가지(a)(b)가
동시에 사라지고, AoS 가지가 **살아 있는 참조**로 남아 이분(bisect)이 가능하다.

### 3.2 커널 — 산술은 한 글자도 안 건드렸다

70개 인덱스 지점이 헬퍼 경유로 바뀌었다. 바뀐 것은 **주소뿐**이다: `dm[4*l+0]`이
`dm[cmfd_layout::mat(nxyz, l, 0)]`이 되었을 뿐, `y0 = dm[…]*x0 + dm[…]*x1`이라는 **식은
글자 그대로 같다.** 이것이 중요한 이유는 가독성이 아니다 — nvcc는 `x + a*b`를 FMA로 접으므로,
식을 **다시 타이핑하면** 대수적으로 같아도 반올림이 달라질 수 있다. 그래서 인덱스만 갈았다.

대상: `matvec_two_group` · `begin_outer_fused` · `prepare_p_jacobi` · `colored_block_sweep` ·
`update_s_jacobi` · `persistentColourSweepNode` · `persistentMatvecNode` ·
`bicg_iteration_persistent` · `cmfd_updls` · 그리고 FP32 쌍둥이 전부
(`refresh_operator_mirror_f32` · `begin_outer_fused_f32` · `matvec_two_group_f32` ·
`colored_block_sweep_f32` · `prepare_p_jacobi_f32` · `update_s_jacobi_f32`).

WP20이 좁힌 FP32 내부 루프는 **같은 치환을 그대로 받는다.** `diag_f`/`cc_f`/`dinv_f`는
double 원본과 **같은 stride(`mat_stride`/`cpl_stride`)** 를 쓰므로 두 정밀도가 한 규약을
공유한다 — 정밀도별로 레이아웃이 갈리는 상황은 만들지 않았다.

### 3.3 생산자 — 전부 처리했다

| 생산자 | 처리 |
|---|---|
| `cmfd_assembly::assembleNode2G` (디바이스 조립) | `View`의 출력 3종과 입력 2종을 헬퍼 경유로. 노드별 베이스 포인터(`diag_l`/`cc_l`/`udiag_l`)는 사라졌고 컴포넌트 인덱스가 노드를 나른다 |
| `cmfd_updls` (Wielandt 재기록) | `l*4 + ige*2+igs` → `mat(nxyz, l, ige*2+igs)` |
| `refresh_operator_mirror_f32` (FP32 미러) | double과 float 양쪽을 같은 헬퍼로 |
| `BatchCore::init` 지오메트리 업로드 | `soa_neighbors`/`soa_node_surface`/`soa_face_area`를 **아레나 생성 시 한 번** 만들어 올린다. 호스트 사본은 노드-메이저로 남는다(그리디 색칠, `compatible()`의 토폴로지 가드, `Geometry`가 그 순서로 읽는다) |
| `issueUploads` H2D (`diag`, `cc`) | 슬롯별 **pinned pack lane** 경유 |
| `issueSweepUploads` H2D (`udiag`, `diag`, `cc`) | 같은 lane 경유 |
| `issueExceptionalOperatorDownloads` D2H | lane으로 당긴 뒤 **드레인 이후에** 호스트 노드-메이저로 흩뿌린다 |

### 3.4 pack lane — 호스트 배열은 왜 안 바꿨나

`CMFD::_diag`/`_cc`와 `BICGCMFD`의 `_udiag`는 **CPU 참조 솔버**(`src/BICGSolver.cpp`)가
노드-메이저로 읽는다. 그것은 우리 파일이 아니고, 무엇보다 **골든 참조**다. 그래서
치환은 pinned lane에서 일어난다:

```
packMat(lane, m, host_aos) : host[l*4+k]  -> lane[mat(nxyz,l,k)]
packCpl(m, host_aos)       : host[l*12+j] -> lane[cpl(nxyz,l,j)]
unpackMat / unpackCpl      : 그 역
```

- **바이트 수는 안 움직인다.** lane은 정확히 `matrix_count`/`coupling_count` double이고,
  `xfer::memcpyAsync`의 사이트명·leaf명·크기가 전부 그대로다 → **XFER 원장이 안 움직인다.**
- **슬롯당 한 lane**이다. 공유 lane 하나면 슬롯 `m`의 DMA가 아직 안 끝났는데 `m+1`이
  소스를 덮어쓴다. `cudaMemcpyAsync`이므로 그것은 정의되지 않은 동작이다.
- **lane은 3개**다(`diag`/`udiag`/`cc`). 호스트 조립 경로에서 셋이 동시에 in-flight다.
- 비용: 슬롯당 `(2*4 + 12) * nxyz` double ≈ **1.1 MB**, 16슬롯 17 MB. 그리고 **v6가 도는
  경로에서는 한 바이트도 복사되지 않는다** — 디바이스 조립이 켜져 있으면 연산자는 PCIe를
  아예 건너지 않는다(`cmfd_diag_h2d_elided_bytes`).

### 3.5 손대지 않아도 됐던 것

- **WP13 elision shadow.** 미러는 *호스트* 버퍼를 섀도잉한다("호스트 배열이 지난 push
  이후 바뀌었나"). 그 질문은 DMA가 어떤 순서로 봤든 무관하다. `pushOrSkip`은 선택적
  `dma_source`를 받고, **미러는 계속 `host_buffer`를 섀도잉한다.**
- **잠긴 파일의 소비자.** `CmfdResidentView`가 내보내는 것은 `phi`/`psi`/`dtil`/`dhat`/`xsnf`
  뿐이다 — **하나도 안 바꿨다.** nodal·PPR·outer-segment 쪽에 접근자도 전치 뷰도
  필요 없었고, `docs/patches/`에 넣을 hunk도 없다.
- **`mat_stride`/`cpl_stride`/`vec_stride`.** 슬롯 stride는 배열 크기이지 내부 순서가
  아니다. 배치 아레나의 레인별 일관성은 자동으로 유지된다 — 모든 슬롯이 같은 `nxyz`를
  쓰므로 `k*nxyz + l`은 슬롯마다 같은 형태다.

### 3.6 수신증

`[RASBERY][CMFD][GRAPH]`에 두 필드가 붙었다:

```
,"layout":"soa","layout_version":2
```

**그래프 캡처 키에는 넣지 않았다.** 키는 *한 프로세스 안에서* 두 캡처를 구분하는 물건이고,
레이아웃은 컴파일 타임 상수라 한 프로세스가 두 값을 가질 수 없다. 키에 넣으면 항상 같은
값을 비교하는 죽은 필드가 된다. 구분이 실제로 필요한 곳 — 프로파일/digest/센서스를
**엉뚱한 커널 본문에 대고 읽는 것** — 은 수신증이고, 그래서 수신증에 넣었다.
(`GpuCaptureArbiter.h`는 WP19.2가 동시에 편집 중이라 애초에 손대지 않았다.)

---

## 4. 왜 B0인가 — 논증

1. **값이 안 바뀐다.** `(l, k)`에 대해 읽히는 double은 같은 double이다. 순열은 전단사다.
2. **식이 안 바뀐다.** 산술 표현식은 글자 그대로 복사되었다. 같은 TU, 같은 플래그 →
   nvcc의 contraction 결정이 같다.
3. **누산 순서가 안 바뀐다.** `assembleNode2G`의 좌면 Z,Y,X / 우면 X,Y,Z 순서, `mulAdd`/
   `roundedMul`의 mined form, `cmfd_updls`의 `fma(-c2, vl, um[idx])` — 전부 그대로다.
4. **리덕션 분할이 안 움직인다.** 리덕션 피연산자를 **하나도 치환하지 않았다**(§2.2).
   `chunk`, per-thread stride, 256-lane 트리, strict ascending fold 전부 불변.
5. **커널 간 의존이 안 바뀐다.** 노드별 쓰기는 여전히 disjoint하고, 색 스윕의 그리드 폭
   배리어는 여전히 커널 경계다.

이 다섯이 전부 참이면 출력은 **비트 단위로 같아야 한다.** 그것을 확인하는 것이 digest
`1f36e75dc00ed2b4` / `4377`이고, 그래서 이 변경은 **플래그 뒤에 두지 않았다** — 플래그는
"측정 없이는 못 믿겠다"는 뜻인데, 여기서는 게이트가 곧 증명이다.

---

## 5. 부수 항목 — FUSE bit 4 (`kFuseNorm`)

`reduce_norm_accumulate_stage2`는 **1블록 × 1스레드**로 BiCGSTAB 이터레이션마다 발사되고,
블록 39의 nsys 집계에서 **전체 커널 시간의 3.7 %** (47k 발사)를 먹는다. 사실상 순수 dispatch다.
WP7 stage B는 이것을 남겨두었고, 그 이유를 소스에 적어두었다: stage 1은 `halt`로,
stage 2는 `active`로 게이트하며 **그 차이가 곧 over-run 텔레메트리**라는 것.

`reduce_norm_accumulate_fused`는 그 반론을 정면으로 처리한다:

```cpp
RASBERY_CMFD_SLOT(m);
if (halt[m] != 0u) {                       // 가드가 아니라, stage 2의 halted path 자체
    if (blockIdx.x == 0u && threadIdx.x == 0u && active[m] != 0u)
        ++counters[m * kCounterSlots + kOverrunCount];
    return;
}
... stage 1 본문 그대로 ...
if (atomicInc(retire + m, gridDim.x - 1u) != gridDim.x - 1u) return;
if (active[m] == 0u) return;               // stage 2의 첫 테스트, 배리어 뒤에서
... strict ascending fold, sqrt, accumulate_iteration_active ...
```

네 경우 전부 두-노드 형태와 관측이 같다: (halted, active) → 카운터 1증가; (halted, inactive)
→ 아무것도; (live, inactive) → stage 1만 돌고 partial만 씀(아무도 안 읽음); (live, active)
→ 접고 누산. FP32 쌍둥이(`reduce_norm_accumulate_fused_f32`)도 같다.

**기본 OFF다.** 소스의 규칙("a default is a claim")을 따른다: mask 15는 양쪽 호스트에서
B0로 측정되고 0.2 s 채택 임계를 넘긴 값이다. bit 4는 텍스트 논증으로 B0지만 **238에서
가격을 안 재봤다.** `kFuseAllBits`에는 들어가고(그래서 `RASBERY_GPU_CMFD_FUSE=31`이
파싱된다) `kFuseDefaultMask`에는 안 들어간다.

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

### 6.0 선행 — 계약 게이트

```bash
python3 tools/test_cmfd_soa_contract.py
python3 tools/test_cmfd_occupancy_contract.py
python3 tools/test_cmfd_fuse_contract.py
python3 tools/test_cmfd_fp32_contract.py
python3 tools/test_gpu_fp32_contract.py
python3 tools/test_xfer_ledger_contract.py
python3 tools/test_enum_alias_contract.py
python3 tools/test_dependent_template_contract.py
python3 tools/test_cmfd_assembly_kernel.py   # 컴파일러 필요 — 238에서만 돈다
ctest   # 12/12
```

`test_cmfd_assembly_kernel.py`가 **이 변경의 실행 가능한 증명**이다: 레거시 CPU 루프를
노드-메이저 그대로 두고, 그 출력을 `cmfd_layout` 헬퍼로 치환한 뒤 조립 커널의 출력과
**바이트 단위로** 비교한다. 값은 같고 주소만 움직였다는 주장이 여기서 실행된다.

### 6.1 digest 불변 (**이것이 먼저다**)

```bash
env $V6_ENV <production arm> ... -o "$OUT/a_soa"
```

합격 조건 — **전부**:

- digest **`1f36e75dc00ed2b4` / `4377`** (WP20의 feature-off 기준값과 같다)
- 직전 v6 산출물 대비 `h5diff -c` **0 차이**
- `[RASBERY][CMFD][GRAPH]`가 `"layout":"soa","layout_version":2`
- `[RASBERY][XFER]` 총계가 **바이트/호출 모두 불변** — pack lane은 순열이지 전송이 아니다
- CSV(핀 파워/AO/keff) 전 항목 Δ = 0

**하나라도 어긋나면 성능은 재지 않는다.** digest가 움직이면 이 변경은 순열이 아니었다는
뜻이고, 그때 필요한 것은 튜닝이 아니라 이분이다: `cmfd_layout::kNodeInnermost = false`로
빌드하면 트리 전체가 WP21-A 이전의 주소로 돌아간다 — 그것이 AoS 가지를 살려둔 이유다.

### 6.2 ncu 재측정 — 다섯 커널, 전/후

블록 39와 **같은 디렉티브**로 재현한다(그래야 비교가 성립한다):

```bash
for k in colored_block_sweep matvec_two_group prepare_p_jacobi \
         reduce_dot_stage1 reduce_dot2_fused update_solution; do
  ncu --kernel-name "regex:$k" --launch-skip 10 --launch-count 5 \
      --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
      --csv env $V6_ENV <production arm> ... > "$OUT/ncu_$k.csv"
done
```

기대치(§2.3의 모델):

| 커널 | ld 전 | **ld 후 기대** | st 전 | **st 후 기대** |
|---|---:|---:|---:|---:|
| `matvec_two_group` | 9.92 | **~3** | 6.38 | 미귀속 잔차만 남음 |
| `colored_block_sweep` | 6.55 | **~3** | 6.90 | **~4** (색 필터의 부분워프, 레이아웃 무관) |
| `prepare_p_jacobi` | 6.33 | **~3.4** | 6.38 | 미귀속 잔차만 남음 |
| `reduce_dot_stage1` | 6.55 | **6.55 (불변)** | 1.00 | 1.00 |
| `reduce_dot2_fused` | 7.05 | **7.05 (불변)** | 1.00 | 1.00 |
| `update_solution` | 2.27 | **2.27 (불변)** | 3.19 | 3.19 |

**뒤 세 줄이 불변이어야 한다는 것이 이 측정의 절반이다.** 움직였다면 리덕션 피연산자를
건드린 것이고, 그러면 §6.1의 digest가 이미 깨져 있어야 한다 — 두 게이트가 서로를 검증한다.

### 6.3 단일덱 wall — 워밍업 1 + hot 3

```bash
for r in w 1 2 3; do env $V6_ENV <production arm> ... -o "$OUT/b_$r"; done
```

**기대치를 미리 적는다: 작다.** 다섯 커널은 단일 실행에서 GPU 시간 ~2.4 s이고 전부
latency-bound다(dram 1~7 %, warps active 4 %). 섹터를 줄여도 지연은 그대로이므로
**단일덱 wall 이득은 0~3 % 범위**로 본다. 이 숫자로 채택을 판정하지 않는다.

### 6.4 배치 — 채택 판정선

```bash
python3.11 tools/run_multi_gpu_batch.py --set "$V6_ENV" ...   # 8 procs x M16 + MPS(12%)
```

기준: 블록 37/38의 클린 V2 baseline **1,320.8~1,326.7 c/h**, 블록 39 재측정 1,291.7 c/h.

**채택 조건**: digest/h5diff/CSV가 §6.1을 통과하고 8×M16 c/h가 **1,321 대비 유의하게
높을 것**. 여기가 이득이 나올 곳인 이유는 §0에 적었다 — 요청당 섹터는 곧 8 클라이언트가
L2/DRAM에 거는 경합량이고, MPS 아래에서 그 경합은 실제 wall이다.

### 6.5 FUSE bit 4 가격 측정 (별도 arm)

```bash
env $V6_ENV RASBERY_GPU_CMFD_FUSE=15 <production arm> ... -o "$OUT/f15"   # 기준
env $V6_ENV RASBERY_GPU_CMFD_FUSE=31 <production arm> ... -o "$OUT/f31"   # bit 4
```

합격 조건: `h5diff -c` **0 차이** + digest 동일 + `[BACKEND_COUNTERS]`의
`overrun_iterations`가 **두 arm에서 같을 것**(이것이 §5 논증의 실측 확인이다) +
`[RASBERY][CMFD][GRAPH]`의 `launches_per_outer`가 이터레이션당 1 감소.
채택은 0.2 s 임계를 따른다 — 그 아래면 bit 4는 armed·OFF로 남는다.

---

## 7. 미착수 — 다음 작업 패키지가 손댈 곳

1. **리덕션의 6.55/7.05.** 완전 연속 접근인데 왜 3배가 나오는지 미해결. 피연산자는
   못 건드리므로(§2.2) 남는 후보는 `chunk`의 32-정렬(패딩된 `begin`)뿐이고, 그것도
   분할을 바꾸면 N1이다. **정렬만 바꾸고 분할은 유지하는 형태**가 있는지가 WP21-B 질문.
2. **`kernelFlatXsCta`의 st 25.2** — 블록 39의 최악값이고, **유일하게 bandwidth-relevant한
   커널**(dram 23.1 %, occupancy 62.4 %)이다. WP20.1 담당 영역.
3. **`kNodalJnet`의 ld 16.7** — 블록 39의 최악 로드. nodal 쪽.
4. **`kPredictor`/`kCorrector`** — 블록 39가 "가장 중요한 발견"으로 지목한 것.
   dram 0.45 %인데 발사당 4.4 ms. 코얼레싱 문제가 아니라 직렬화/점유율 문제다.
5. **`phi`의 4섹터.** 지금은 nodal/PPR 핸드오프 때문에 못 움직인다. 소비자 쪽까지
   같이 옮기는 작업 패키지가 서면 그때 2로 간다.
