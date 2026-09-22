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
- **E3. Descriptor 병목 = chunk를 1벡터로 collapse (WRITE).** 시도된 모든 pooling(mean/max/attn)이 →1벡터, per-key 구별 소실. [A6, A8] **갱신 2026-09-22**: multi-vector/late-interaction을 시도했고 **이긴다** — 32토큰 블록 8개 + best-block 채점으로 적중률 0.325 → 0.450(p=0.0059), 메모리 +6%, 파라미터 0개. m에 내부 최적점(8)이 있어 per-key 벡터로 가는 길이 아니다. [G6, G7]
- **E4. Exact multi-key의 *총* 저장은 ~O(#facts).** 압축(hier)이 drop에 짐(B7), frozen이 상수 메모리 아님(A4). ※ 이는 시퀀스 전체 저장량 얘기이며, within-chunk 해상도(D2/E1)와는 별개 축.

## F. 반복 관측된 cross-cutting 사실

- **F1. "추가한 memory 경로가 next-token 목표에 보상 못 받아 붕괴"가 독립 3곳에서 관측.** 축1 surprisal drift 0.40→0.01(B6), 축3 salience 부트스트랩 실패(B8), DSC alpha 0.54→0.27(C3).
- **F2. 이 태스크들은 예산/제약 없으면 비차별적.** full soft-attention이나 무제한 캐시를 주면 trivially 풀림 (0016 soft, 0017 fixed==oracle). where-to-cut 주장은 캐시/read 예산 고정 필요.
- **F3. 선행연구 게재 결과 (참고).** Memory Caching(Behrouz, inference O(L)) REJECT → 같은 팀 MoM(memory=substrate, inference O(1)) ACCEPT; Log-Linear ACCEPT. [OPENREVIEW_LESSONS]

## G. MC-SSC(mean-pool) GDN-2 370M @8K diverse-key — long-gdn 캠페인 (0027, 2026-08-19~09-11)

_0024는 length 2048에서 "routing이 지배 병목"을 oracle 주입으로 확립했다. 아래는 8192(후보 세그먼트 ~31개)에서의 측정과, 처음으로 이긴 개입 둘. 프로토콜: 시드 42/43/44, 셀당 50문항, topk 2, gate native. 팔 비교는 전부 짝지은 정확 부호검정._

- **G1. 캐시를 붙이면 순손실이다 (고정 프로토콜, 학습량 일치).** 8192·시드 42/43/44·셀당 50문항에서 학습량을 맞춘 vanilla GDN-2 30B **11.0** vs MC-SSC 30B **2.7**. 짝지은 300문항 **3획득/28상실, p=0.0000**. 손실은 **N=4에 집중**된다(17.3 → 2.0, 세 시드 모두 p≤0.031). N=16은 4.7 vs 3.3으로 양쪽 바닥. 즉 캐시는 이미 실패하던 모델을 더 망치는 게 아니라 **작동하던 모델을 깨뜨린다.** ※ 이전 판의 "vanilla 18.7"은 다른 벤치마크·다른 체크포인트 수치였고 **철회**. 첫 대조는 vanilla를 5B로 잡아 학습량이 6배 어긋나 있었다. [0027]
- **G2. state는 답을 갖고 있다.** gold를 전 레이어 top-1에 강제하면 **74.0**이고 기준선 대비 **214획득/0상실** — 한 문항도 잃지 않는다. 0024의 oracle 주입 결과가 8K에서도 성립. native 라우팅 적중률 **0.050**(고정 프로토콜 300문항, 우연 ≈0.065), 레이어 간 선택 일치도 0.00. [0027]
- **G3. 동결 모델에 dense를 강제하면 0.0점** (기준선 대비 0획득/8상실, p=0.0078). 셀당 1500초로 top-2의 430초 대비 3.5배 — 희소 읽기가 곧 효율 주장이라는 점이 시간으로도 나온다. 단 이는 top-2로 학습된 모델을 추론 때 dense로 민 것이라 **dense 아키텍처의 품질이 아니다.** Log-Linear·DLA는 모든 상태를 학습된 가중으로 읽고 multi-key에서 크게 이득(DLA on Gated DeltaNet: MK-NIAH-1 19.1 → 49.3 @4K). **희소하게 읽는 것이 효율 주장이고, 희소하게 읽는 것이 깨진다.** [0027, DLA Table 3]
- **G4. 학습 가능한 gold 신호는 L0·L1에만 (H6 답).** 레이어별 라우터 적합 AUC: L0 **0.742**, L1 0.706, L2~L15 0.51~0.59 = 같은 캡처에서 재적합한 원본 선형 connector(0.507~0.554)와 구분 불가. 3M MLP와 단일 선형 사상이 같은 점수 → 용량 부족이 아니라 `h`에 정보가 없다. [0027]
- **G5. 라우팅 공유가 이긴다.** L0에서 한 번 고르고 16층이 그 인덱스를 재사용(gate 무게는 층별 유지) → **동일 라우터로 N=4가 4 → 18**(짝지은 8획득/1상실, p=0.039). 추론 시 라우팅 계산 1/16. 근거는 G4 + "읽기는 gold가 모든 층에 있어야 작동"(oracle 16/16 = 82 vs ~3/16 = 4). [0027]
- **G6. 하위 블록 max-sim이 이긴다 — E3의 미시도 칸.** 32토큰 블록 8개를 저장하고 가장 잘 맞는 블록으로 채점: 적중률 **0.325 → 0.450**(깨끗한 2시드, 51획득/26상실, p=0.0059, 4/4 셀 양수). 세그먼트 메모리 **+6%**(state 262k값 대비 descriptor 2048값), 파라미터 0개. [0027]
- **G7. m은 내부 최적점이 있다 (8).** m = 1/2/4/8/16/32에서 적중률 32.5 / 35.0 / 35.0 / **44.0** / 36.0 / 35.0. 단조가 아니다 — 8토큰 블록은 평균 자체가 시끄럽다. 8 > {1,4} 확립(p=0.0117 / 0.0300), 8 > {16,32}는 방향만. **따라서 "finer chunks"류가 아니고, RESEARCH_STATE가 기각한 per-key 벡터도 아니다.** 32블록 새 캡처를 오프라인 coarsen한 m=8이 44.0으로 배포 라우터 45.0을 재현. [0027]
- **G8. 합산 헤드라인 2.7 → 13.0, 그러나 캐시 없는 모델을 못 넘는다.** 짝지은 300문항 37획득/6상실, **p=0.0000**, oracle 74.0의 17.6%. 그런데 같은 프로토콜의 **vanilla 30B 11.0과 견주면 30획득/24상실, p=0.50 — 구별되지 않는다.** 개입은 캐시가 입힌 손해를 되돌릴 뿐 그 이상을 사지 못한다. [0027]
- **G9. 읽기는 gate 무게의 문턱이다 (H7 답).** gold 무게 0.123이면 정답, 0.046이면 오답, oracle 0.536. 공유 top-2에서 gold 옆에 뽑힌 세그먼트가 더 큰 무게를 받는 경우가 흔하다(0.242 vs 0.123). k를 넓히면 적중률 3배·점수 18 → 10. [0027]
- **G10. 남은 병목은 읽기. 완벽한 선택도 절반만 메운다.** 분해 결과 선택 0.450→**0.640**, 변환 0.295→**0.328**(N=4) / 0.095→**0.231**(N=16), oracle 변환은 0.67/0.69. [0027]
- **G11. 문서 쪽 학습 파라미터는 전부 진다.** 4 init 짝지음: fixed max-sim **0.338** vs per-head 사영 0.314(4/4 패, t=3.07, 262k) vs 학습된 블록 어텐션 pooling 0.223(2k) vs 평균·max 혼합 0.315(전역 스칼라 1개) / 0.306(헤드별) / 0.337(질의조건부). **이득을 낸 변형이 하나도 없다.** [0027]
- **G12. 혼합은 끝점을 포함해도 안전하지 않다.** a=0이 평균과 1.2e-7, a=1이 max와 정확히 일치하는데도 적합된 a는 0.56에 앉아 양쪽 다 못 이긴다. max는 few_needles를 +0.104(4/4, t=3.8) 이기고 many_needles를 −0.039(1/4) 진다 — **두 체제가 반대 집계를 원하고 전역 스칼라 하나로는 못 맞춘다.** 질의조건부로 만들면 적응이 아니라 max 쪽으로 더 간다(many에서 0/4). [0027]
- **G13. gate 마진은 두 범위 모두 실패.** 균일 마진 13.5 → 3.5(p=0.0000); 답이 생성되는 마지막 세그먼트로 좁혀도 18.5 → 10.0. 질의 위치 무게 분포는 oracle급이었는데 N=16이 0.0으로 붕괴 = 프리필 오염. [0027]
- **G14. 그 외 기각.** k 확대 단독, 공유 소스를 L0에서 이동, L0+L1 앙상블, needle 보조손실, 8K 정합 학습셋(적중률 22.3%→17.0%, p=0.0070), 학습 없는 어휘 매칭(GDN의 `k`는 문맥적이라 chance). [0027]
- **G15. 76.7M에서 고장 모양 재현.** 비임베딩 52.1M, 친칠라 1.53B 토큰, 앵커 레시피 유지. oracle **20.0** vs 무개입 **0.33**(60배, 370M은 32배), 적중률 0.040, 레이어 일치도 0.00. ppl은 학습 길이 4배까지 유지(4K 27.2 / 8K 26.5 / 16K 30.8)이므로 외삽 탓이 아니다. **단 전이 제약**: 학습 가능 신호가 50M은 7개 층(L0~L6 0.75~0.78), 370M은 2개 층. 라우팅 공유 이득을 작은 쪽에서 과대평가할 수 있다. [0027]
- **G16. 프로토콜 드리프트가 한 달을 비교 불가로 만들었다.** 같은 무개입 기준선이 4.0(1시드) / 2.3(3시드) / 3.0(2 깨끗한 시드), oracle이 82와 74. **팔을 돌리기 전에 프로토콜을 고정하고 base·oracle을 같은 런에 넣을 것.** [0027]
- **G17. 학습을 포함한 오프라인 프로브의 분해능은 ~0.02 (n=196).** forward 출력이 1.2e-7까지 동일한 모형을 두 번 적합하면 0.311과 0.327이 나온다 — fp 오차가 8000 AdamW 스텝에 누적돼 독립 추출이 된다(500스텝 뒤 가중치 차 2.5e-2). 0.021 차이로 내린 판정 하나를 철회했다. **판정용 변형은 4 init 이상, 평균과 폭을 함께 보고.** [0027]

- **G18. MC-SSC는 용량 추가가 아니라 순환 분절이다.** `_segment_gdn2_batched`가 모든 세그먼트를 `initial_state=None`으로 스캔하고 `gdn2_ssc_forward`는 `independent` 외의 모드를 거부한다. 따라서 위치 t는 자기 256토큰만 보고, 세그먼트를 잇는 유일한 다리가 top-k 읽기다. 다리가 우연 수준이면 남는 것은 256토큰 모델이고, 그게 G1의 기제다. [0027, dsc/mc_gdn2/ssc.py]
- **G19. 트랙 1(surprisal 가중 descriptor) 기각 — 전제가 거짓.** 정답 토큰의 surprisal이 같은 세그먼트 나머지 대비 **0.889배**(6셀 전부 1.0 미만), 세그먼트 내 순위 **0.456**(가운데). τ를 올릴수록 정답 토큰의 블록 내 가중치 몫이 0.218 → 0.096으로 **줄어든다.** needle이 템플릿 문장이라 에세이 산문보다 예측하기 쉽다. 라우터 적합 전에 캡처 4.5분 + CPU로 판정. [0027 §8, probe_surprisal_premise.py]
- **G20. 트랙 2(학습 prefix) 기각 — 두 이유.** 8,192 파라미터를 2,000스텝 학습해 손실 2.76 → 2.83으로 **변화 없음**(train top1 0.100 → 0.090). 그리고 held-out **ppl +8.1%**로 ppl 게이트에 걸렸다. 라우팅은 못 고치고 언어모델링만 망친 순손실. 프롬프트가 질의를 모르므로 세그먼트마다 같은 벡터가 들어가 조건부 선택이 불가능한 것이 원인으로 보인다. [0027 §8]
- **G21. 동결 백본 위 학습 파라미터는 7/7 패배.** 사영 262k, 어텐션 pooling 2k, 혼합 1개·헤드별 16개·질의조건부 D+1개, prefix 8k. 이긴 둘은 파라미터 0개(라우팅 공유, max-sim). [0027 §5, §8]


- **G22. chained 추론 대조는 판정에 실패했다 (교란).** `checkpoint_mode=chained`로 추론만 바꾸면 두 팔 모두 **6셀 전부 0.00**(base 대비 0획득/8상실 p=0.0078, vanilla30b 대비 0/33 p=0.0000, maxsim+chained는 maxsim 대비 0/39). 체크포인트가 `initial_state=None`으로 학습돼 경계를 넘는 상태를 본 적이 없으므로 이 붕괴가 분절 탓인지 분포 이동 탓인지 **구분되지 않는다.** 실험 설계 시 "올라가면 증거, 내려가면 미결"로 미리 적어둔 그대로다. **chained 판정은 from-scratch 학습 외에 길이 없다.** 부수: chained가 셀당 ~460초로 independent ~430초와 거의 같다 — 8K에 세그먼트가 32개뿐이라 배치 병렬 이득이 작고, 따라서 from-scratch에서 속도는 장애물이 아니다. [0027, run_chained_pod.sh]
- **G23. 이 프로토콜의 점수는 바닥에서 분해능이 거의 없다.** vanilla 11.0 / MC 2.7 / chained 0.0 / dense 0.0이 모두 하위 구간에 눌려 있어 2.7과 4.0을 구별할 수 없다(셀당 50문항, 1문항 = 2점, 검출 하한 12pp). 바닥 근처 팔은 점수가 아니라 **적중률로 먼저 스크리닝**할 것. chained를 점수부터 잰 것이 이 규칙을 어긴 사례다. [0027]


---

_이 문서는 사실만 담는다. 진행 중 가설(multi-vector + late-interaction으로 E2/E3 공략 등)·설계·추천은 `notes/adaptive-boundary-research.md` 및 각 report 참조._
