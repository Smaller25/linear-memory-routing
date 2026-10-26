# 0025 실험 스펙 — MC-SSC router 진단: router는 needle 검출기인가, key 판별기인가

> 대상: Claude Code. 0024(`lmr/analysis/260725_mc_niah_analysis/`)의 E1 인프라를 재사용한다.
> **전부 재학습 없음.** X1–X4는 추론/후처리만. GPU 학습 job 제출 금지.
>
> **rev2 (2026-07-27) 변경점**
> - X1에 `u_t := q_t` 재계산 추가 (무료, 가장 강한 단일 개입 후보)
> - **X2 신설**: multi-query / multivalue의 key별 hit@2 — 새 Tier 1. 새 데이터 생성 불필요
> - 구 X2(off-template probe) → **X3**로 이동, 순서도 뒤로
> - **"routing 공간 분리"를 처방 목록에서 삭제** (rev1의 논리 오류 — §0.3 참조)
> - §7의 multiquery/multivalue 제외 조항 삭제

## 목차

- [0. 배경과 판별 대상](#0-배경과-판별-대상)
  - [0.1 세 가설](#01-세-가설)
  - [0.2 분기별 처방](#02-분기별-처방)
  - [0.3 rev1에서 철회한 처방 — "routing 공간 분리"](#03-rev1에서-철회한-처방--routing-공간-분리)
  - [0.4 실행 순서](#04-실행-순서)
- [1. 환경·코드 고정](#1-환경코드-고정)
- [2. X1 — descriptor-only 예측 + `u_t := q_t` 재계산](#2-x1--descriptor-only-예측--u_t--q_t-재계산)
- [3. X2 — multi-query / multivalue key별 hit@2 (신설, Tier 1)](#3-x2--multi-query--multivalue-key별-hit2-신설-tier-1)
- [4. X3 — off-template anomaly probe (최종 갈림길)](#4-x3--off-template-anomaly-probe-최종-갈림길)
- [5. X4 — random-routing MC ablation](#5-x4--random-routing-mc-ablation)
- [6. X5 — sub-block MaxSim P-sweep (조건부)](#6-x5--sub-block-maxsim-p-sweep-조건부)
- [7. 판정 규칙과 산출물](#7-판정-규칙과-산출물)
- [8. 하지 말 것](#8-하지-말-것)

---

## 0. 배경과 판별 대상

0024 결론: MC-SSC(mean-pool descriptor) GDN2-370M의 multi-key NIAH 실패는 routing이
지배적 병목. oracle 주입으로 4/4 셀 개선(최대 0.062→0.562).

미확정 문제는 **router가 무엇에 반응하는지**다. 상위 주장은 다음 한 문장이고, 이 스펙의
목표는 이것을 확정한 뒤 두 하위 가설을 가르는 것이다.

> **상위 주장 (M)**: router는 needle 검출기이지 key 판별기가 아니다.
> "이 segment에 haystack과 다른 내용이 있는가"는 풀고, "그 중 어느 key인가"는 정보가 없다.

### 0.1 세 가설

| ID | 가설 | single hit@2 = 0.93–1.00 | `amongkeys` ≈ 0.5 |
|---|---|---|---|
| H_blind | 위치·norm만 사용, query 무시 | 설명 못 함 → **전역 기각** | 설명함 |
| H_outlier | descriptor 중심에서 먼 segment 선택 | 설명함 | 설명함 |
| H_template | query의 공통 템플릿 성분만 매칭, 판별 토큰(`X`) 미사용 | 설명함 | 설명함 |

H_blind는 single hit@2가 chance(0.286)의 3.2–3.5×라는 점에서 전역적으로 기각.
단 **layer 9–13 평탄 구간(paired_S≈0.19, paired_D≈0.25 고정)** 은 국소적 H_blind일 수
있으므로 layer별로 확인한다(X1).

H_outlier와 H_template은 둘 다 (M)을 함의한다. 따라서 **X2는 (M)을 검정하고, X3가
둘을 가른다.** 순서가 이렇게 되는 이유는 X2가 기존 RULER 태스크만으로 되고 X3는 데이터
생성이 필요하기 때문이다.

### 0.2 분기별 처방

| 판정 | 처방 (순서대로) | X5(P-sweep) |
|---|---|---|
| **H_outlier** | ① `u_t := q_t` 로 묶기 (X1에서 무료 검증) → ② contrastive router supervision (핵심) → ③ descriptor 기하 교정 (아래) | 실행 안 함 |
| **H_template** | ① sub-block MaxSim (X5가 메인) → ② checkpoint granularity 통합 → ③ contrastive supervision 보조 | **메인 실험으로 승격** |

**descriptor 기하 교정에 대해.** 0024에서 `qk_align ≈ 0` (−0.04~+0.06)인데, d=64 랜덤
방향의 코사인 표준편차가 `1/√64 = 0.125`이므로 **관측값은 잡음보다도 작다.** 그런데 읽기는
작동한다(D 조건 read-cos ≥ 0.98). 설명 가설: GDN2는 delta rule 계열이라 state가
`(I − β_t k_t k_tᵀ)` 곱으로 이전 key 성분을 지워가며 쓰므로, `v_{t*}`를 꺼내는 실제 읽기
방향은 raw `k_{t*}`가 아니라 후속 key들에 의해 deflate된 방향이다. 즉 **descriptor를 raw
key 평균으로 두는 것은 backbone의 읽기 기하와 다른 공간에서 매칭하는 것**이다.
→ 처방은 "전용 projection 추가"가 아니라 WY/UT 표현 기준으로 descriptor를 구성하는 것.
이 가설의 검정은 본 스펙 범위 밖(후속), 여기서는 근거만 기록한다.

### 0.3 rev1에서 철회한 처방 — "routing 공간 분리"

rev1은 H_outlier의 처방으로 "routing 공간 분리 + contrastive supervision"을 적었다.
**앞부분은 철회한다.** 두 개의 다른 축을 하나로 뭉갠 오류였다.

- **축 A**: router query `u_t` vs retrieval query `q_t` (논문 Table 5 `- Shared u and q` 행)
- **축 B**: descriptor를 `W_K`(state에 쓰는 키)로 만들 것인가, 전용 projection으로 만들 것인가

rev1이 근거로 든 `qk_align ≈ 0`은 **축 B**에 대한 관측인데, 논문 ablation 행(축 A)을 끌어와
"분리"로 묶었다.

H_outlier 하에서 축 A를 분리하는 것은 처방이 아니라 악화 요인이다.

1. outlier 고르기는 LM loss의 **퇴화 최적해**다. 파라미터를 더 주면 그 해를 찾을 자유도만 늘어난다.
2. top-k가 미선택 segment의 gradient를 차단하므로 **새 파라미터에 갈 학습 신호가 없다.**
   자유도 확대는 신호가 있을 때만 의미가 있다.

**반대 방향이 맞다.** 논문에 이미 대안 파라미터화로 `u_t = q_t`가 적혀 있다. D 조건 읽기가
0.98 이상으로 온전하므로 `q_t`는 key 정체성 정보를 갖고 있다. router가 그것을 그대로 쓰면
작동하는 판별 신호를 물려받는다. 재학습 없이 점수만 다시 계산해 검증 가능 → **X1에 포함.**

축 B는 "분리"가 아니라 §0.2의 기하 교정으로 다룬다.

### 0.4 실행 순서

```
X1 (반나절) ─┬─→ X2 (1일)  ──→ X3 (1일, 갈림길) ──→ H_template면 X5
             └─→ X4 (병렬, 아무 때나)
```

---

## 1. 환경·코드 고정

```
long-gdn worktree : e71713e
fla               : 4b02d15d (이후 버전은 GLA API 비호환 — pin 유지)
모델              : mc-gdn2-370m-*-v2-meanpool, 5B / 30B ckpt
                    16 layers, topk=2, chunk 256, descriptor = L2-norm key mean-pool
작업 디렉토리     : lmr/analysis/260725_mc_niah_analysis/
```

**시작 전 확인 (추측하지 말고 읽을 것):**

1. `data.py`의 `prepare-a` / `prepare-b` 서브커맨드 실제 시그니처
2. `forward_with_diagnostics`에서 `route_indices` / `route_scores`를 계산·저장하는 지점 —
   descriptor `c_i`, router query `u_t`, retrieval query `q_t`를 그 자리에서 함께 덤프 가능한지
3. `results/` 하위 JSON 스키마 (E1 출력 키 이름)
4. vendored RULER(`src/ruler`)에 `niah_multiquery`, `niah_multivalue` 태스크가 있는지,
   needle 개수 파라미터가 노출되는지 (X2의 전제)
5. `MemoryCachingGDN2Layer._project` 재사용 가능 여부 (E3 경로)
6. bf16 재실행 노이즈 ~2.3pp. seed 고정. 셀당 결론 마진이 3pp 이내면 유의하지 않은 것으로 처리

Dataset B는 gold/distractor를 segment 1–5에만 배치한다(0024 한계). 신규 데이터도 동일
규약을 따를 것 — 비교 가능성 유지.

---

## 2. X1 — descriptor-only 예측 + `u_t := q_t` 재계산

**목적** (두 개):
- (a) layer별로 router가 query를 사용하는지 판정 → 국소 H_blind 검정
- (b) `u_t`를 `q_t`로 교체하면 hit@2가 오르는지 → §0.3의 처방 ①을 무료로 검증

추가 forward pass 0회. E1이 이미 answer position에서 16 layer의 routing score를 계산하므로,
그 지점에서 `c_i`(정규화 전/후), `u_t`, `q_t`, top-2 인덱스를 덤프하도록 확장한다.

### 2a. descriptor-only 예측

query를 전혀 쓰지 않는 특징 3개:

```
pos_i     = i / (s-1)                       # 정규화 위치
norm_i    = ||c_i||_2                       # 정규화 전 descriptor norm
outlier_i = 1 - <ĉ_i, mean_j(ĉ_j)>          # descriptor 중심에서의 거리 (H_outlier 직접 대응)
```

측정:
1. **선택 예측**: 위 3개로 실제 top-2 집합 예측. layer별 Jaccard. 각 특징 단독 argmax-2도 함께
2. **점수 회귀**: `r_t^(i) ~ pos_i + norm_i + outlier_i` 의 R², layer별
3. **descriptor Gram off-diagonal 평균** `mean_{i≠j} <ĉ_i, ĉ_j>`, layer별 (공짜)

판정:
- 선택 예측 정확 / R² ≥ 0.9 → 그 layer는 query 미사용 (국소 H_blind)
- layer 9–13에서 성립하고 14–15에서 불성립할 것으로 **예상**. 다르면 그것이 결과이므로 그대로 보고
- Gram off-diagonal ≥ 0.8 → anisotropy 확정. **X1c 추가 실행**: descriptor centering
  (`c_i - mean_j c_j`) 또는 top-1 PC 제거 후 hit@2 재계산. 예측 개선폭 `1/(1-⟨cos⟩)` 배

### 2b. `u_t := q_t` 재계산 (rev2 신규)

같은 덤프에서 router query만 retrieval query로 교체해 점수를 다시 매기고 hit@2를 재계산한다.
descriptor는 그대로. 조건 4개: `stock` / `u=q` / `stock+centering` / `u=q+centering`.

- 대상: paired-S, paired-D, single, multikey. 5B/30B. layer별 + best-layer
- **`u=q`에서 paired-D hit@2가 유의하게 오르면 그것만으로 저비용 개선안이 확보된다.**
  이 경우 후속 설계의 출발점이 바뀌므로 즉시 보고
- 차원 불일치 가능성 있음(`u_t ∈ R^d` vs `q_t`의 head 차원). 확인 후 head별 평균 또는
  head 단위 점수 최대값 중 무엇을 썼는지 반드시 기록

### 출력

`results/x1_router_probe.json`
```
{layer: {"pred_jaccard": float, "r2": float, "gram_offdiag": float,
         "feat_r2": {"pos":…, "norm":…, "outlier":…},
         "hit2": {"stock":…, "u_eq_q":…, "stock_centered":…, "u_eq_q_centered":…}}}
```
그림: layer축 2패널 — (R², gram_offdiag) / (4조건 hit@2). 0024 Figure 2 곡선을 같은 x축에 겹칠 것.

---

## 3. X2 — multi-query / multivalue key별 hit@2 (신설, Tier 1)

**목적**: 상위 주장 (M) "router는 needle 검출기이고 key 판별기가 아니다"를 직접 검정.
**새 데이터 생성 불필요 — 기존 RULER 태스크를 쓴다.**

### 근거

multi-key와 multi-query는 **needle 개수와 배치가 같고 질문만 다르다.**
multi-key는 여러 needle 중 하나만 묻고(정답 segment 1개), multi-query는 전부 묻는다
(정답 segment N개). (M)이 참이면 sharp prediction:

```
hit@2(multi-query) ≈ hit@2(single)  ≫  hit@2(multi-key)
```

needle 검출만 하면 되는 과제에서는 잘 되고, key를 판별해야 하는 과제에서만 무너진다는 것.
컨텍스트가 사실상 동일하므로 차이는 전적으로 "어느 needle을 원하는가"에서 온다.

multivalue(key 1개, value 여러 개, 전부 정답)도 같은 논리로 포함한다.

**주의: H_outlier와 H_template을 가르지는 못한다.** 둘 다 "잘 될 것"으로 예측한다.
그건 X3의 일이다. X2는 (M)의 검정이고, (M)이 깨지면 X3 이하 전체를 재설계해야 한다.

### 측정

**최종 정확도로 재지 말 것.** top-k=2가 needle N개를 덮지 못해 무조건 0 근처가 나오고,
생성 형식과 `string_match_all`이 섞인다. E1 기계로 routing 레벨에서 잰다.

- **질문된 key별 hit@2**: 각 needle에 대해 그 needle의 segment가 top-2에 들어가는가.
  needle별로 따로 집계한 뒤 평균 (macro)
- 보조: `hit@k` with **k를 needle 수에 맞춰 올린 조건** (추론 시만, 재학습 없음).
  k 상한이 병목인지 분리
- needle 수 스윕: 2, 4 (RULER 파라미터가 노출되면). 안 되면 기본값 그대로
- 대조군: 같은 길이·같은 needle 수의 multi-key. **needle 수를 맞추는 것이 이 실험의 핵심**
- 길이: 2K 기본. **4K, 8K 추가** — 0024가 2K만 측정해서 1절 (다)의 4K→8K 붕괴에 대한
  직접 설명이 없다. multi-key hit@2가 8K에서 4K보다 떨어지면 "needle 검출 자체의 precision이
  segment 수에 따라 붕괴"라는 별도 결론이 나온다
- 모델: 5B/30B. layer별 + best-layer

### 판정

| 관측 | 결론 |
|---|---|
| multi-query hit@2 ≈ single ≫ multi-key | **(M) 확정.** X3로 진행 |
| multi-query hit@2 ≈ multi-key (둘 다 낮음) | **(M) 기각.** router가 needle 검출도 못 하는 것 → 문제는 key 판별이 아니라 descriptor 전반. X1c(centering)와 기하 교정이 최우선으로 승격되고 X3/X5는 보류 |
| multi-query가 single보다 낮되 multi-key보다 높음 | 부분 지지. needle 수 스윕으로 k 상한 효과와 분리 |

### 출력

`results/x2_multiquery.json` + (태스크 × 길이 × 모델) 표. multi-key 대조를 같은 표에.

---

## 4. X3 — off-template anomaly probe (최종 갈림길)

**X2에서 (M)이 확정된 경우에만 실행.** H_outlier vs H_template 판별.

### 데이터

Dataset B(paired, 32쌍) 생성기를 확장. 기존 multi 예제에 **템플릿 없는 이상 문장** 1개 추가.

- needle: 기존 RULER 템플릿 유지 (실제 문자열은 코드에서 확인)
- **off-template anomaly**: 맥락 없는 UUID 또는 랜덤 영숫자 한 줄. 질의어와 어휘 겹침 0,
  템플릿 구조 없음. haystack과는 분포적으로 명확히 다름
- 배치: gold / distractor / anomaly를 서로 다른 segment에 (D 조건 준용). anomaly도 1–5 범위
- 쌍맞춤 유지: anomaly 유무만 다른 쌍, 나머지 토큰 단위 동일

### 측정

anomaly segment의 `hit@2`, layer별. 대조군은 같은 위치의 haystack-only segment.

### 판정

| 관측 | 결론 | 다음 |
|---|---|---|
| anomaly가 gold와 경합 | **H_outlier** | §0.2 H_outlier 처방. **X5 실행 안 함** |
| anomaly 무시, gold/distractor만 선택 | **H_template** | **X5를 메인으로 승격.** X3b 실행 |
| 전부 chance | 재검토 — X1·X2와 교차 확인 | — |

n=32, 마진 3–4 샘플. 방향성 해석. 두 모델 모두.

### X3b — paired query-swap (H_template인 경우만)

같은 context에 정당한 query가 두 개 존재한다(gold key 질의 / distractor key 질의).
descriptor 완전 동일, query만 다르고 정답 segment는 다르다. top-2가 **바뀌지 않으면**
판별 부분공간 미사용의 직접 증거. within-context 대조라 예제 간 분산이 제거되고 n=32로 결론.

측정: 두 query의 top-2 Jaccard, 각각의 hit@2. **X1b의 `u=q` 조건에서도 같이 측정할 것** —
`u=q`로 바꿨을 때 Jaccard가 낮아지면(선택이 query에 따라 갈리면) 처방 ①의 직접 증거가 된다.

---

## 5. X4 — random-routing MC ablation

**목적**: 앵커 표에서 MC의 장문 single-needle 이득이 routing 품질에서 오는지, segment당
write 부하가 256으로 상한되는 효과인지 분리. **X1–X3와 독립, 병렬 실행 가능.**

### 근거

MC 5B의 S-NIAH-1은 8K→32K에서 24/30/34로 회복되고 vanilla 5B는 16/4/0으로 붕괴한다.
32K에서 vanilla는 state 하나가 32K를 흡수하고 MC는 각 state가 256만 쓴다.
**routing이 완전 무작위여도 읽어낸 state가 더 깨끗하다.** 이 ablation 없이는 "MC의 이득이
routing에서 온다"는 문장을 쓸 수 없다. 현재 앵커 세트의 최대 공백.

### 구현

추론 경로에서 top-k 선택만 교체. 가중치 재학습 없음.

- `random`: 과거 segment에서 k개 균등 샘플 (seed 고정, 5 seed)
- `recent`: 항상 가장 최근 k개 (위치 baseline)
- `stock`: 변경 없음 (sanity — `gold=None`일 때 byte-identical, 0024 E2와 동일 규약)

gate는 0024 E2와 동일 규약: `score = max(선택 최대, online)`.

### 측정

S-NIAH-1, MK-NIAH-1 @ {1K, 2K, 4K, 8K, 16K, 32K}, 5B/30B.

병기할 것: `chance-routing 상한 = 2/(N-1) × conditional_EM`, `N = ctx/256`,
`conditional_EM`은 0024 oracle 범위 0.31–0.56 → 0.6으로 상한 근사.
관측/상한 비율이 길이에 따라 커지면 routing이 실제로 작동하는 증거.

### 출력

`results/x4_random_routing.json` + `stock / random / recent / vanilla` 4행 × 6열 표(markdown),
길이별 배율 열 포함.

---

## 6. X5 — sub-block MaxSim P-sweep (조건부)

**X3가 H_template으로 판정된 경우에만 실행.**

**목적**: index granularity만 세분화해 paired-D hit@2가 회복되는지. state는 256 그대로,
새 파라미터 0개, 재학습 없음.

### 구현

각 segment의 256개 key를 `P`개 sub-block으로 stride 분할, 각각 mean-pool + L2 정규화.

```
s_i = max_{p ∈ [P]} <û_t, ĉ_ip>
```

`P ∈ {1, 2, 4, 8, 16, 32}`. **P=32까지 반드시 포함.**
X1c에서 centering이 효과 있었으면 `centering × P` 2차원으로 확장.

### 예측 (곡선 형태 자체가 결과이므로 스펙에 명시)

sub-block 길이 `C' = 256/P`일 때:

```
signal(판별 토큰 1–2개)                      ∝ 1/C'
noise(haystack C'-1개 key 평균의 segment간 변동) ∝ 1/√C'
⇒ SNR ∝ 1/√C',  즉 P 대비 개선은 √P 배
```

MaxSim이 P개 노이즈의 최대를 취하는 `√(2 ln P)` 페널티를 감안하면 순 개선은 약
`√P / √(2 ln P)`. **P=8 실패가 기본 예측이다.**

측정 곡선을 예측 곡선과 겹쳐 그린다. 정량 법칙이 되므로 단순 ablation보다 기여도가 높다.

### Kill-check

> **P=32에서도 paired-D best-layer hit@2가 0.6을 넘지 못하면 index granularity 가설 기각.**
> → contrastive supervision + 기하 교정으로 전환.

(P=8 기준으로 판단하지 말 것 — 위 √P 스케일링 때문에 실패가 기본 예측이다.)

### 출력

`results/x5_psweep.json`, 그림: P축 × hit@2, 예측 곡선 overlay, paired-S/D 분리.

---

## 7. 판정 규칙과 산출물

### 공통 규약

- best-layer 방식은 0024와 동일 유지 (layer별 보고 + best-layer 요약)
- **chance level을 데이터셋마다 따로 계산해 병기.** 0024에서 multikey 0.486,
  single/paired 0.286–0.288. 신규 데이터·신규 길이는 다시 계산할 것.
  multi-query는 정답 segment가 여러 개이므로 chance 계산식이 다르다 — 반드시 재도출
- n이 작다(16–32). 절대값 주장 금지, 방향성으로 서술
- bf16 노이즈 2.3pp 이내 차이는 결론에 쓰지 않음
- 모든 수치는 JSON에 먼저 쓰고 보고서는 JSON만 참조 (하드코딩 금지)

### 보고서

`report/0025.md` (영문, 수치·재현 커맨드 원본) + `report/0025_ko.md` (한국어 요약).
0024 형식 — **맨 앞에 목차**, 한 줄 결론, 셋업, 실험별 절, 판정표, 한계, 재현.

판정표:

| 항목 | 판정 | 근거 | 함의 |
|---|---|---|---|
| H_blind (layer 9–13 국소) | | X1a R², 선택 예측 | |
| **(M) router = needle 검출기** | | X2 multi-query vs multi-key | |
| H_outlier | | X3 anomaly hit@2 | |
| H_template | | X3 + X3b Jaccard | |
| `u_t := q_t` 개선 | | X1b 4조건 hit@2 | |
| anisotropy | | X1a Gram, X1c centering | |
| MC 이득 = routing | | X4 random vs stock | |

### 최종 산출

1. 위 판정표
2. X4 표 (chance 상한 병기) — 앵커 공백 메움
3. 다음 설계 권고 1페이지. §0.2 분기표를 실측으로 채운 형태

---

## 8. 하지 말 것

- **학습 job 제출 금지.** X1–X4 전부 추론/후처리다. 재학습이 필요해 보이면 멈추고 보고
- **"routing 공간 분리"를 처방으로 제안하지 말 것.** §0.3에서 철회했다.
  `u_t`와 `q_t`는 분리가 아니라 **묶는** 방향을 검증한다
- fla 버전 올리지 말 것 (GLA API 비호환)
- Dataset B의 gold/distractor segment 범위(1–5)를 바꾸지 말 것 — 0024와 비교 불가해짐
- 0024의 E2 oracle 결과를 재측정하지 말 것 (`e2_oracle.json` 사용.
  `e2_join`은 정보량 부족으로 사용 안 함)
- anti-oracle 구현하지 말 것 — X1/X2/X3가 더 싸고 정보량이 많아 대체됨
- X2를 최종 정확도로 재지 말 것 (§3 참조). routing 레벨에서 잰다
- 위치 정보를 특징에서 빼지 말 것. `pos_i` 단독 R²가 높을 가능성이 실제 후보다
- 결과가 예상과 다를 때 예상에 맞추려고 지표를 바꾸지 말 것. 다른 결과는 그대로 보고