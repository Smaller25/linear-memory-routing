# Linear-Memory-Routing — 중간 보고서 (2026-06-22)

지금까지의 프로젝트를 종합한다. 초기 가설, 방법론 계열, 각 실험 트랙과 그 결과, 현재 위치를
담았다. 실험별 상세는 `report/0001`–`0012`에, 운영 상태는 `SESSION_HANDOFF.md`와
`lmr/mosc/README.md`에 있다. 이 문서는 그것들을 잇는 단일 서사다.

---

## 1. 초기 가설과 동기

선형 순환 시퀀스 모델(Mamba-2, Gated DeltaNet 등)은 O(1) 상태로 동작하지만, **고정 크기 순환 상태가
포화한다**는 한계가 있고, **동시에 등장하는 사실들을 하나의 상태 행렬에 뒤섞는다**. 실패 양상은 두
축으로 갈린다.

- **시간 축(포화):** 긴 맥락을 지나는 동안 고정 상태가 과거 정보를 덮어쓴다 → 길이가 길어지면
  Needle-in-a-Haystack / passkey에서 실패한다.
- **공간 축(간섭):** 여러 사실이 한 상태에 누적되며 서로를 덮어쓴다 → multi-key 회상(MQAR)에서
  실패한다.

**핵심 가설.** 시퀀스를 따라 **순환 상태의 체크포인트를 캐싱**하고, 질의 시점에 그중에서 고르는 작은
**학습된 read-out 라우터**를 붙이면, 모델은 동작에 쓰는 상태를 키우지 않고도 **고정 상태의 회상 한계를
넘어선다**. 라이브 상태가 잃어버린 정보를 되살리는 방식이다.

두 레퍼런스가 이 두 축을 잡아준다. **Memory Caching**(시간 축 체크포인트)과
**Mixture-of-Memories / MoM**(간섭을 위한 병렬 메모리)이다.

---

## 2. 두 트랙

| | **Frozen 리트로핏** (Track A, 보고서 0001–0011) | **From-scratch 공동학습** (Track B, 보고서 0012) |
|---|---|---|
| 아이디어 | 사전학습 모델을 freeze하고, 캐싱된 상태 위의 read-out 라우터만 학습 | 캐싱·라우팅을 내장한 작은 모델을 end-to-end로 학습 |
| 백본 | Mamba-2 1.3B/2.7B, GDN 1.3B (사전학습 → FLA) | GDN-2 (2층, from scratch) |
| 목표 | single-needle long-context (포화) | multi-key 회상 (간섭) |
| 성과 | **검증된 net win** | **multi-key 해결 (이번 세션)** |

---

## 3. Track A — frozen 리트로핏 (보고서 0001–0011)

세그먼트별 순환 상태 캐시 위에서 시도한 read-out 메커니즘들:

- **RM** (학습 없는 read mix) — neutral-to-negative; frozen은 학습된 것과 다르다 (0002).
- **GRM** (학습된 게이트) — RM의 붕괴를 회복하지만, 학습한 세그먼트 수의 ~2배까지만 일반화 (0003, 0004).
- **SSC** (스냅샷에 대한 hard top-k 선택) — 여기서 win이 나왔다.

**핵심 결과.**

- **포화 길이에서의 net win.** 자연어 passkey @8k: vanilla→+SSC = **0.738→0.986 (+0.25) @mamba2-1.3b**,
  **0.500→1.000 (+0.50) @2.7b**. 이 win은 **모델이 클수록 커진다** (0005, 0007).
- **표준 벤치마크로 전이.** passkey로 학습한 SSC가 RULER `niah_single`에서 **zero-shot**으로
  vanilla를 이긴다 (+0.06–0.18 @4k/8k) (0009).
- **오직 hard top-k만 일반화.** RM / GRM / AoM(dense) / MoM-slot-merge / hierarchical은 long-context에서
  모두 붕괴하고, 희소한 hard 선택만 살아남는다 (k∈[2,8]; k=1은 불안정) (0008, 0009).
- **GDN**은 long-context 회상이 강한 백본이지만(vanilla passkey 8k≈0.93 vs mamba2 0.74), GDN의 **라우터
  학습은 커널에 막혔다**. head_dim=256에서 chunk-bwd 공유메모리가 A100 한도를 넘고 tilelang 공백이
  있다 (0006).

**정직한 음성 결과 (Track A의 경계).**

- **multi-key는 범위 밖 (0010).** multi-key로 SSC를 학습시키면 수렴하지 않았다. 근본 원인:
  *"MC는 **사실**마다가 아니라 **세그먼트**마다 상태 하나를 캐싱한다; 한 세그먼트를 공유하는 키들은
  구분되지 않는다."* 이것은 *포화*가 아니라 *간섭* 영역이다.
- **constant-memory가 아니다 (0011).** win은 O(N) 스냅샷이 거의 다 있어야 나온다; 캐시를 B로 제한하면
  회상이 B/N에 비례해 저하된다. top-k는 *읽기*를 O(N·k)로 줄이지만 *캐시*는 O(N)으로 남는다 → SSC는
  RNN↔attention 스펙트럼 위의 압축 캐시 한 점이지, constant-memory 선형 모델이 아니다.

---

## 4. Track B — from-scratch Dynamic-MoSC (보고서 0012, 이번 세션)

**아이디어.** 0010의 간섭 영역을 공략하려고, **내용에 따라 적응하는 세그먼트 경계**(세그먼트가
256토큰 단위가 아니라 ≈사실 단위가 되도록)와 Track A의 hard-top-k 세그먼트 캐시 read-out을 함께 두고
from scratch로 공동학습한다. 백본은 **GDN-2** (`fla.layers.gdn2`; gated-delta의 일반형 — 스칼라 게이트
→ Gated DeltaNet v1, 벡터 게이트 → KDA). 태스크는 Zoology 충실 구현 **MQAR**.

### Phase 0 — 세그먼트 라우팅이 multi-key를 풀긴 푸는가? (recall vs #kv)
| 모델 | kv4 | kv8 | kv16 | kv32 | kv64 | kv128 |
|-------|----|----|-----|-----|-----|------|
| vanilla GDN-2 | 1.00 | 1.00 | 0.93 | 0.55 | 0.28 | — |
| **oracle** (사실 단위 경계) | 1.00 | 1.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| fixed / surprisal | 1.00 | 1.00 | ~0.95 | ~0.6 | ~0.3 | ~0.15 |

→ **PASS.** 사실 단위 경계만 있으면 세그먼트 캐시 + hard-top-k read가 multi-key를 **푼다**(0010
영역). 다만 win은 전부 **경계 배치**에서 온다. fixed/surprisal은 vanilla 수준에 그친다.

### Phase 1 — oracle 경계는 학습 가능한가?
모델 자신의 hidden 상태 위에 올린 `Linear(d,1)` 헤드를 oracle 위치로 distill했다.

- 첫 시도는 실패했다(vanilla보다 못함). 원인은 **커리큘럼 과적합**으로 진단됐다(kv 4/8만으로 학습하니
  *위치* 편향을 익혀, kv와 무관하게 경계를 ~8개만 쳤다). **커널 버그가 아니다**(fla gated-delta op
  테스트가 sm_120에서 통과, 10/10).
- 커리큘럼을 **train-kv 4–64**로 넓히자 해결됐다: **learned ≈ oracle**(kv128 0.998), 학습에서 보지
  못한 kv 수까지 일반화. → **경계는 학습 가능하다.**

### true-state 타당성 검증 — 확정됨
read-out은 세그먼트를 pooled-hidden **proxy**(attention-lite)로 요약하고 있었다. 그래서 win이 *진짜
순환 상태 회상*인지가 미해결로 남아 있었다. 각 경계의 **진짜 GDN-2 순환 상태**를 뱅크에 담아 다시 돌렸다
(`backbone.run_segmented`):

| 모델 | kv16 | kv32 | kv64 | kv128 |
|-------|-----|-----|-----|------|
| oracle (proxy)      | 1.00 | 1.00 | 1.00 | 1.00 |
| **oracle (true-state)** | 1.00 | 0.999 | **0.998** | **0.992** |
| fixed (proxy)       | 0.97 | 0.64 | 0.34 | 0.17 |
| **fixed (true-state)**  | 0.999 | 0.988 | **0.882** | **0.468** |

→ (1) **multi-key win은 진짜 순환 상태 회상이다.** pooled-hidden attention 아티팩트가 아니다. 핵심
caveat이 해소됐다. (2) **진짜 순환 상태 자체가 큰 지렛대다**: 경계가 조잡한 `fixed`인데도
kv64에서 0.34→0.88, kv128에서 0.17→0.47로 뛴다(proxy는 경계 배치의 중요성을 과장했다). 경계 배치는
가장 밀도가 높은 지점(kv128)에서만 여전히 결정적이다.

---

## 5. 과정 메모 (값비싸게 얻은 것; 재발견 금지)

- **서버 이전.** A100(VESSL) → 2× RTX PRO 6000 Blackwell(sm_120, Slurm). GPU를 감지해 분기하는 환경
  (`scripts/setup_env.sh`) + conda 환경 **`sh_routing`** + Slurm 엔트리포인트
  (`scripts/sh_slurm_run.sh`)를 만들었고, A100과 Blackwell 경로를 둘 다 살려뒀다. GPU 작업은 모두
  `sbatch`로 돈다.
- **fla 0.5.1 → 0.5.2** 로 **GDN-2**를 들여왔다(`fla/ops/gdn2`, `fla/layers/gdn2.py`); transformers
  5.12용 `_tied_weights_keys` 패치를 재적용했고, frozen 트랙 게이트는 그대로 통과한다(18/18).
- **Blackwell 커널은 멀쩡하다.** Mamba-3는 `mamba_ssm`(SISO 커널)이 필요해 미뤘다; GDN-2 / KDA /
  GDN-v1은 순수 Triton이라 sm_120에서 from scratch로 학습된다(tilelang 불필요). tilelang은
  py3.13/Blackwell에서 못 쓰지만, GDN-2 경로엔 상관없다.
- **MQAR 함정.** 높은 lr(3e-3) + grad-clip + 쉬운 kv와 어려운 kv를 함께 담은 커리큘럼이 필요하다;
  kv 16/32만으로 곧장 학습하면 loss가 랜덤에 박힌다(버그가 아니라 커리큘럼). 레거시 MoCMMixer 트레이너는
  fla 0.5.2에서 학습되지 않는다(경로 밖).

---

## 6. 현재 위치

**확립된 것.**

- Track A: frozen 모델의 캐싱된 상태 위에 학습한 hard-top-k read-out은 single-needle long-context에서
  크기에 따라 커지는 실질 net win을 주고 RULER로 전이한다 — 다만 single-needle 전용이고
  constant-memory가 아니다.
- Track B: Track A가 닿지 못한 multi-key 영역이, GDN-2 + **진짜** 캐싱 순환 상태 위의 hard-top-k
  read-out으로, **경계가 사실 단위일 때** from scratch로 **풀린다**. 그리고 그 경계는 **학습
  가능하다**(learned ≈ oracle, 일반화 포함).

**열린 것 (다음 마일스톤).**

1. **비지도 경계** — oracle distill 없이, 태스크 loss / budget 페널티만으로 사실 단위 경계를 학습.
   (distill은 신호가 존재함을 보였고, 이건 그 신호가 비지도로 학습 가능함을 보이는 단계다.)
2. **learned × true-state 결합** (지금까지 따로 측정).
3. **true-state read-out 스케일** — `run_segmented`는 세그먼트별 순차 루프라 ~5–6× 느리다; batched /
   intermediate-state 커널이 필요하다.
4. **일반화** — single-needle long-context(passkey/RULER), 병렬 풀 라우팅(Mixture of Segment-Cache),
   더 긴 맥락(16k/32k).

**보고서 매핑:** Track A = 0001–0011; Track B = 0012; 이 문서 = 종합본.
