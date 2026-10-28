# Linear-memory-routing — 확립된 사실 (facts only)

_2026-07-09. 증거로 뒷받침되는 사실만. 가설·제안·추천은 제외 (그건 `adaptive-boundary-research.md`).
출처 태그: report 번호(standalone), long-gdn 문서/job, SSM_Rank_Analysis. 재현 1회뿐인 건 (single-run)로 표시._

---

## A. Track A — frozen GDN + SSC read-out (retrofit)

- **A1. Single-needle 순승, 크기 스케일링.** passkey@8k vanilla→+SSC: 0.738→0.986 (1.3B), 0.500→1.000 (2.7B). [0005/0007]
- **A2. RULER single-needle length 무관 개선.** vanilla→+SSC: 4K 0.701→0.957, 32K 0.365→0.868, 128K 0.188→0.910, **512K 0.125→0.969**. 16K 이후 vanilla 단조 붕괴, SSC 0.91+ 유지. [long-gdn README]
- **A3. hard top-k만 일반화.** RM/GRM/AoM/MoM-merge/hierarchical read-out은 붕괴; SSC k∈[2,8]만 통함. [0008]
- **A4. 상수 메모리가 아님.** ~전체 O(N) state 스냅샷 필요; 캐시를 B로 제한하면 recall ∝ B/N로 저하. [0011]
- **A5. Multi-key 실패 (vanilla보다 나쁨, short-L).** niah_multikey_1 @4K: vanilla 0.752 vs +SSC 0.424. multivalue @8K: vanilla 0.648 vs SSC 0.215. 65K+에서만 근소 역전(single_3 0.195→0.219). [MULTINIAH_PARETO]
- **A6. Multi-key 12+ variant 전부 실패.** F1-A(supervised routing)/C(pointer-copy)/D(temp)/F(pooling mean·max·attn)/G(chunk-size) 모두 multi@4K < 0.65 (best 0.441). 결론: frozen LM head + chunk-mean-pool descriptor + per-layer SSC 조합이 부적합. [MULTIKEY_ATTACK_PLAN]
- **A7. Multi-key routing이 길이와 함께 붕괴.** any-layer hit rate 4K 0.95 → 16K 0.65 → 32K 0.40. router prob on needle 0.192 → 0.057 → 0.028. single은 any-hit 1.0 전 길이 유지. [7Q]
- **A8. Descriptor는 cos-sim으로 안 보이나 선형 분리됨.** disc_ratio(cos-sim): single median 0.945 / multi **1.002(random)**. linear-probe AUC: single **1.000**(24층 중 21층) / multi median **0.822**(max 0.917). L2-normalize 후 AUC 불변 → **신호는 magnitude 아닌 direction**. [DESCRIPTOR_SIGNAL_FINDINGS]

## B. Track B — from-scratch GDN-2 (standalone, 0012–0020)

- **B1. Multi-key는 oracle/learned 경계면 from-scratch로 풀림.** GDN-2 + hard-top-k segment-cache read-out: oracle·learned 경계 kv512까지 ~1.0 (vanilla는 규칙 0.92@kv64→0.002@kv512). 경계 precision/recall ~1.0. [0012/0013]
- **B2. 진짜 순환 상태로도 유지.** pooled-hidden proxy를 실제 GDN-2 state로 바꿔도 oracle ~1.0 (kv128 0.99). [0012]
- **B3. Adaptivity는 supervised에서만.** 규칙 MQAR degenerate(fixed chunk=2 == oracle == 1.0). 불규칙 MQAR: fixed=0.00, oracle/learned ~1.0, learned가 위치 복원(prec/rec/top-k 1.0). [0015]
- **B4. Unsupervised 경계 발견 3방식 모두 실패.** soft(경계 안 생김, full-attn 우회) / STE+L1(0으로 붕괴) / warm-start(oracle 제거 후 drift). threshold-free top-k(p)∩facts = **0.00** vs supervised **1.00**. [0016]
- **B5. Selective Copying: read-out 순승, 단 무제한 캐시면 adaptivity 불필요.** vanilla M128 0.592 → fixed/oracle/learned 모두 ~0.99. learned 위치 복원 1.00. **fixed chunk=2도 ~0.99** (과분할 공짜). [0017]
- **B6. 축1 정보밀도: 신호는 실재하나 co-train에 약함.** task-학습 백본의 raw surprisal이 fact를 top-k(p) **0.40** 랭크(0016의 0.00 대비). quantile firing으로 hard 캐시 켜면 **0.40→0.01 drift**. entropy 0.2–0.3, cosdist ~random. [0018]
- **B7. 축2 hierarchical 재압축: 압축이 drop에 짐.** oracle 경계, 예산 B: capped-64 0.98/0.49/0.24 (∝B/N 재현), **hier-64 0.73/0.26/0.03 < capped**. merge가 fact 파괴 → exact recall엔 손해. [0019]
- **B8. 축3 salience-gated retention (상수 메모리): oracle만 작동.** filler128: oracle 0.97/0.59/0.13/0.03 (부트스트랩된 vanilla 0.82/0.30/0.06/0.01 대비 +0.11~0.19). none·surprisal은 **0/3 부트스트랩 실패**. [0020]
- **B9. from-scratch MQAR 부트스트랩은 GPU-비결정 knife-edge.** 동일 config가 loss 8.32(실패) vs 0.000(성공)으로 갈림 (job 1249 vs 1260). [0020]

## C. DSC — long-gdn from-scratch (GDN-2 370M + chunk-cache + top-k=2 routing)

- **C1. 1B 토큰: DSC가 vanilla 압도.** single_1 4K +160%(0.052 vs 0.020), 32K +38%. multikey 4K +55%, 32K +64%, 65K +114%. [DSC_PROBLEM_STATUS]
- **C2. 5B 토큰: single_1 퇴행 (vanilla 아래로).** single_1 32K 0.029→0.013 (-55%), vanilla는 0.021→0.029(+38%). "더 학습했는데 성능 하락". multikey는 5B에서 near-floor(양쪽 0.02~0.08). [DSC_PROBLEM_STATUS]
- **C3. 원인 확정: alpha combine-gate 폐쇄.** `o = o_main + alpha·o_mem`에서 alpha 1B **0.538** → 5B **0.270** (-50%, 16층 전부). backbone·router weight(45.0↔45.1)·descriptor·loss 전부 정상. 다른 가설(descriptor degeneration, state washout, router decay) 기각. [DSC_PROBLEM_STATUS, 가설 E]
- **C4. DSC RULER 절대값 near-floor (0.02~0.08).** from-scratch 370M을 1~5B 토큰만 학습한 약한 base + alpha collapse. (Track A frozen GDN-1.3B은 4K 0.957.) [DSC_PROBLEM_STATUS]
- **C5. v0→v3 정보 누출 3건 후 수정.** v1 gather/mask leak, v2 `seg_query.mean` leak → v3에서 인과 보장. loss 정상, 코드 버그 없음. [DSC_INCIDENT_POSTMORTEM]
- **C6. v4는 alpha를 안 건드림.** v4 = learnable attention chunk-summary + router entropy-bias (HiLS). `dsc_combine_alpha` additive gate 구조 v3와 동일. [dsc/v4/CHANGES, v4/dsc.py]
- **C7. DSC ~4.3× 느림.** chunk sequential scan + cached-state overhead (16.7K vs 72.5K tok/s/GPU). [DSC_PROBLEM_STATUS]

## D. State 용량 / 신호 (SSM_Rank_Analysis)

- **D1. SSM state의 effective rank는 context 길이 따라 T*에서 포화.** head별 상이(Type A/B/C). state injection이 oracle retrieval 부분 복제. [SSM_Rank_Analysis]
- **D2. within-chunk는 포화 근처가 아님.** mamba2 state 64×128, eRank 용량 ~64. kv 2→64쌍에서 eRank 2.8→~7. → **256 토큰 chunk 안 소수 key는 eRank 한 자리 ≪ 용량.** [worked_example_S1_D1]
- **D3. 밀도-기반 동적 chunking은 자연어로 전이 안 됨.** 합성(반복): chunk-length vs density Spearman ρ epiplexity **0.94** / eRank 0.73. 자연어 passage: eRank ρ **−0.008**, epiplexity ρ **NaN**. [dynamic_chunking_by_density]
- **D4. Info-capacity 신호 매트릭스는 계산됨, 승자 분석은 pending.** S1 eRank/S2 pred-entropy/S3 epiplexity/S4 decodable/S5 Rényi/S6 intrinsic-dim × MQAR/NL/state-tracking. §7 분석 미완. [information_capacity_signals]

## E. Multi-key 진단 (종합 — 사실 수준)

- **E1. Multi-key 실패는 within-chunk capacity 한계가 아니다.** within-chunk state는 미포화(D2), 즉 소수 key는 chunk state에 구별되어 남음. [D2 + A5–A8 종합]
- **E2. 실패 지점은 두 mechanism.** (a) **cross-chunk routing**: query-independent + pooled descriptor가 긴 문맥에서 queried key의 chunk를 못 고름 (A7: any-hit 0.95→0.40). (b) **within-chunk 추출**: 맞는 chunk를 retrieve해도 READ가 pooling으로 queried key를 못 뽑음. [7Q H2 + H2-alt]
- **E3. Descriptor 병목 = chunk를 1벡터로 collapse (WRITE).** 시도된 모든 pooling(mean/max/attn)이 →1벡터, per-key 구별 소실. multi-vector/late-interaction은 미시도. [A6, A8]
- **E4. Exact multi-key의 *총* 저장은 ~O(#facts).** 압축(hier)이 drop에 짐(B7), frozen이 상수 메모리 아님(A4). ※ 이는 시퀀스 전체 저장량 얘기이며, within-chunk 해상도(D2/E1)와는 별개 축.

## F. 반복 관측된 cross-cutting 사실

- **F1. "추가한 memory 경로가 next-token 목표에 보상 못 받아 붕괴"가 독립 3곳에서 관측.** 축1 surprisal drift 0.40→0.01(B6), 축3 salience 부트스트랩 실패(B8), DSC alpha 0.54→0.27(C3).
- **F2. 이 태스크들은 예산/제약 없으면 비차별적.** full soft-attention이나 무제한 캐시를 주면 trivially 풀림 (0016 soft, 0017 fixed==oracle). where-to-cut 주장은 캐시/read 예산 고정 필요.
- **F3. 선행연구 게재 결과 (참고).** Memory Caching(Behrouz, inference O(L)) REJECT → 같은 팀 MoM(memory=substrate, inference O(1)) ACCEPT; Log-Linear ACCEPT. [OPENREVIEW_LESSONS]

---

_이 문서는 사실만 담는다. 진행 중 가설(multi-vector + late-interaction으로 E2/E3 공략 등)·설계·추천은 `notes/adaptive-boundary-research.md` 및 각 report 참조._
