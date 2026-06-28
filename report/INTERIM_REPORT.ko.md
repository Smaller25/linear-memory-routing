# Linear-Memory-Routing — 중간 보고서 (2026-06-28 갱신)

report 0016까지의 결과를 묶었다. 목표와 실험 세팅, 결과를 한자리에 담았다. 실험별 상세는
`report/0001`–`0016`에 있고, 이 문서는 그것들을 잇는 단일 서사다. (0012까지만 다루던 2026-06-22
버전을 대체한다.)

---

## 1. 목표와 가설
선형 순환 모델(Mamba-2, Gated DeltaNet/GDN-2, KDA)은 O(1) 상태로 동작하지만, 고정 상태가 (a) 긴
맥락에서 **포화**하고 (b) **동시에 등장하는 사실들을 뒤섞는다**. 실패는 두 축으로 갈린다.
- **시간 축(포화):** 과거 정보가 덮어써진다 → long-context single-needle retrieval에서 실패.
- **공간 축(간섭):** 여러 사실이 한 상태에 섞인다 → multi-key recall에서 실패.

**가설.** 시퀀스를 따라 순환 상태 체크포인트를 캐싱하고, 그 위에 작은 **학습된 hard-top-k read-out
라우터**를 붙이면, 동작에 쓰는 상태를 키우지 않고도 **고정 상태의 recall 한계를 넘는다**.

**두 트랙** (한 체크포인트로 합칠 수 없다 — 합성·from-scratch vs 실제·사전학습):
| | Track A — frozen 리트로핏 (0001–0011, 0014) | Track B — from-scratch 공동학습 (0012–0016) |
|---|---|---|
| 아이디어 | 사전학습 LM을 freeze, read-out 라우터만 학습 | 캐싱·라우팅을 내장한 작은 모델을 end-to-end 학습 |
| 목표 | single-needle long-context retrieval | multi-key recall + 적응적 분절 |

---

## 2. 실험 세팅
- **모델.** Track A: 사전학습 `state-spaces/mamba2-{370m,1.3b,2.7b}`, GDN-1.3b → FLA, freeze, + ~30M
  SSC 라우터. Track B: from-scratch GDN-2 (d_model 256, 4층, ~6M).
- **벤치마크 (표준 스위트).**
  - **MQAR** (Zoology 충실, `lmr/tasks/mqar.py`) — 토큰 단위 multi-key recall; 통제용 in-house probe.
    **불규칙 변종** (`make_mqar_gapped`): 사실 사이에 랜덤 filler → 비주기 위치(adaptivity 테스트).
  - **RULER** (NVIDIA vendoring, 실제 텍스트 long-context) — `niah_single`, `niah_multikey`; free
    generation + 공식 string-match.
  - **flip-flop** (FFLM, `lmr/tasks/flipflop.py`) — state-tracking do-no-harm 체크.
  - *권고(미실행):* **Selective Copying** + **MAD noisy/fuzzy recall** — 우리 MQAR+filler probe의 표준
    대응물; 신뢰성 위해 채택 (`notes/adaptive-boundary-research.md`).
- **레시피.** lr 3e-3, AdamW (wd 0.1, β 0.9/0.95), grad-clip 1.0; MQAR은 쉬운 kv와 어려운 kv를 함께 담은
  커리큘럼이 필요하다(어려운 kv만으로 곧장 학습하면 부트스트랩에 실패한다).
- **하드웨어.** A100(VESSL) → 2× RTX PRO 6000 Blackwell(sm_120)로 이전, Slurm, conda 환경 `sh_routing`
  (torch 2.11+cu128, FLA 0.5.2). GPU 작업은 모두 `sbatch scripts/sh_slurm_run.sh`.
- **read-out 모드** (`lmr/mosc/`): `fixed`(매 C), `oracle`(사실마다 경계), `surprisal`, `learned`(경계
  head를 oracle로 distill), `unsup`/`unsup_ste`(oracle 없음). 세그먼트 요약은 **경계 토큰의 hidden**으로
  잡는다(평균-풀링은 fact를 희석한다 — 0015/§4 참고). 진짜 순환 상태 캐시는 `backbone.run_segmented`.

---

## 3. 결과

### Track A — frozen 리트로핏
- **SSC 순승, single-needle (0005/0007).** passkey @8k vanilla→+SSC: **0.738→0.986 (+0.25) @1.3b**,
  **0.500→1.000 (+0.50) @2.7b** — 모델이 클수록 이득이 커진다.
- **RULER로 전이 (0009).** passkey로 학습한 SSC가 RULER `niah_single`에 zero-shot으로 통한다(+0.06–0.18).
- **오직 hard top-k만 일반화 (0008).** RM/GRM/AoM/MoM-merge/hierarchical은 붕괴; SSC k∈[2,8].
- **RULER free generation, mamba2-370m (0014, 이번 세션).** 공식 메트릭, matched n=30:
  niah_single @2048 **vanilla 0.00 → +SSC 36.7** (zero-shot); multikey 0.00 → 3.3. 리트로핏이
  **실제 free-gen 프로토콜에서 2048까지 순승**한다. 단서: 370m은 ~2k 사전학습이라 그 너머는 OOD다(4k/8k
  바닥). **+SSC free-gen은 2k를 넘으면 너무 느리다**(4h 타임아웃 — decode 루프가 매 토큰 캐시 세그먼트를
  재실행한다; batched 커널이 필요하다).
- **음성 결과.** multi-key는 범위 밖이다 (0010). **상수 메모리(constant memory)가 아니다** — ~전체 O(N)
  스냅샷이 필요하고, 캐시를 B로 제한하면 ∝ B/N로 저하한다 (0011).

### Track B — from-scratch Dynamic-MoSC
- **multi-key 해결·스케일 (0012/0013).** GDN-2 + hard-top-k 세그먼트 캐시 read-out: vanilla는 포화하지만
  (규칙 0.92@kv64→0.002@kv512), **oracle·learned 경계는 kv512까지 ~1.0**이고, 학습한 kv≤128 범위
  너머로 일반화한다. 경계 precision/recall ~1.0.
- **진짜 순환 상태로 확정 (0012).** pooled-hidden proxy를 각 경계의 실제 GDN-2 상태로 바꿔도 oracle은
  ~1.0을 유지한다(kv128 0.99) — win이 pooled activation에 대한 attention이 아니라 **진짜 상태 recall**
  이라는 뜻이다. 진짜 상태 자체가 강력한 지렛대다(fixed 경계 0.34→0.88 @kv64).
- **Adaptivity — 결정적 테스트 (0015).** 규칙 MQAR은 **degenerate**다: fixed `chunk=2` == oracle == 1.0
  (사실이 고정 주기). **불규칙** MQAR에선 둘이 갈린다:

  | 불규칙 | kv64 | kv128 | kv256 | kv512 |
  |---|---|---|---|---|
  | vanilla | 0.82 | 0.30 | 0.06 | 0.01 |
  | **fixed chunk=2** | **0.00** | **0.00** | **0.00** | **0.00** |
  | oracle | 1.00 | 1.00 | 1.00 | 0.97 |
  | **learned** | ~1.0 | ~1.0 | ~1.0 | **1.00** |

  고정 stride는 **실패**한다. learned head는 oracle과 동급이고(precision/recall **1.0**), 세그먼트 길이가
  **진짜 분포**(median 4@kv64→2@kv512, 랜덤 gap을 따라간다)를 따른다 — 규칙의 스파이크-at-2와 대조된다.
  → 경계는 **content-adaptive하며 학습된다** — *감독 하에서*.
- **unsupervised 경계 학습은 3방식 모두 실패 (0016).** oracle 없이는: **soft** landmark-attention은
  recall ~1.0지만 full attention을 우회하고(경계가 안 생긴다), **STE hard-cache + L1**은 0으로 붕괴하며,
  **warm-start**(distill 후 oracle 제거)는 **완전히 drift**한다 — threshold-free top-k(p)∩facts = **0.00**
  vs supervised **1.00**. 경계 신호는 지속적 감독을 요구한다. unsupervised 이산 경계 발견은 열린 문제다.
- **state-tracking do-no-harm (flip-flop).** vanilla GDN-2와 Dynamic-MoSC(fixed) 둘 다 n_instr
  128/256/512에서 1.00 — 분절이 native state-tracking을 깨지 않는다.

## 4. 정직한 범위 / 보이지 못한 것
- **adaptivity는 supervised 한정이다.** learned per-fact 경계(0015)는 oracle distillation을 요구하고,
  unsupervised 발견은 실패했다(0016). 이게 핵심 미해결이다.
- **sparse-cache 예산이 주장의 전제다.** full(soft) attention을 주면 이 recall 태스크는 trivially
  풀려서 caching에 대해 아무것도 증명하지 못한다 — 모든 비교는 캐시/read 예산을 고정해야 한다(0016).
- **소규모다.** Track B는 합성 MQAR 위 ~6M from-scratch; Track A RULER은 370m(2k 너머 OOD).
- **free-gen + read-out은 2k를 넘으면 느리다** (RULER 4k/8k +SSC 타임아웃) — batched/intermediate-state
  커널이 필요하다.
- Mamba-3은 보류했다(`mamba_ssm` SISO 커널 필요); GDN 라우터 *학습*은 head_dim=256에서 커널에 막힌다.

## 5. 현재 위치
**확립된 것.** (A) frozen 모델의 캐싱된 상태에 학습한 hard-top-k read-out은 single-needle
long-context에서 크기 따라 커지는 순승을 주고, RULER로 전이하며, 실제 텍스트 free-gen에서도 2048에서
순승한다 — 다만 single-needle 전용이고 상수 메모리가 아니다. (B) multi-key recall은 GDN-2 + **진짜** 캐싱
상태 위의 read-out으로, 경계가 사실 단위일 때 from scratch로 풀리고, 그 경계는 **감독 하에서 학습되며
content-adaptive**하다.

**열린 것.** unsupervised 경계 학습; free-gen+read-out을 2k 너머로 스케일(커널); supervised method를
표준 벤치마크(Selective Copying / MAD)로 검증; 진짜 per-row 순환 상태 캐시; LongBench.

**보고서 맵.** Track A = 0001–0011, 0014; Track B = 0012, 0013, 0015, 0016; figure = artifact
`seg-length-dist`, `adaptive-boundaries`; 이 문서 = 종합본.
