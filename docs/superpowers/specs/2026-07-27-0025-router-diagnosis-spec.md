# 0025 실험 스펙 — MC-SSC router 진단: H_blind / H_outlier / H_template 판별

> 대상: Claude Code. 0024(`lmr/analysis/260725_mc_niah_analysis/`)의 E1 인프라를 재사용한다.
> **전부 재학습 없음.** X1–X3는 추론/후처리만. GPU 학습 job 제출 금지.

## 목차

- [0. 배경과 판별 대상](#0-배경과-판별-대상)
- [1. 환경·코드 고정](#1-환경코드-고정)
- [2. X1 — descriptor-only 예측 (H_blind 검정)](#2-x1--descriptor-only-예측-h_blind-검정)
- [3. X2 — off-template anomaly probe (갈림길)](#3-x2--off-template-anomaly-probe-갈림길)
- [4. X3 — random-routing MC ablation](#4-x3--random-routing-mc-ablation)
- [5. X4 — sub-block MaxSim P-sweep (조건부)](#5-x4--sub-block-maxsim-p-sweep-조건부)
- [6. 판정 규칙과 산출물](#6-판정-규칙과-산출물)
- [7. 하지 말 것](#7-하지-말-것)

---

## 0. 배경과 판별 대상

0024 결론: MC-SSC(mean-pool descriptor) GDN2-370M의 multi-key NIAH 실패는 routing이
지배적 병목. oracle 주입으로 4/4 셀 개선(최대 0.062→0.562).

다만 **router가 무엇에 반응하는지**가 미확정이다. 세 가설이 현재 데이터를 모두 설명한다.

| ID | 가설 | single hit@2 = 0.93–1.00 | `amongkeys` ≈ 0.5 |
|---|---|---|---|
| H_blind | 위치·norm만 사용, query 무시 | 설명 못 함 | 설명함 |
| H_outlier | descriptor 중심에서 먼 segment 선택 | 설명함 | 설명함 |
| H_template | query의 공통 템플릿 성분만 매칭, 판별 토큰(`X`) 미사용 | 설명함 | 설명함 |

H_blind는 single hit@2가 chance(0.286)의 3.2–3.5×라는 점에서 전역적으로는 이미 기각.
단 **layer 9–13 평탄 구간(paired_S≈0.19, paired_D≈0.25 고정)** 은 국소적 H_blind일 수
있으므로 layer별로 확인한다.

처방이 갈린다:
- **H_outlier** → routing 공간이 query와 연결되지 않음. descriptor 세분화는 무의미.
  routing projection 분리 + contrastive supervision으로 감.
- **H_template** → 판별 신호는 존재하고 해상도만 부족. sub-block MaxSim(X4)이 메인.

우선순위: **X1 → X2 → (분기) → X3는 병렬로 아무 때나.**

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
2. E1 스크립트에서 layer별 routing score를 계산·저장하는 지점 — descriptor `c_i`와
   router query `u_t`를 그 자리에서 함께 덤프할 수 있는지
3. `results/` 하위 JSON 스키마 (E1 출력 키 이름)
4. `chunk_gdn2` 경로 (E3에서 state 재구성에 쓴 것) — X3에서 재사용 가능한지
5. bf16 재실행 노이즈 ~2.3pp. seed 고정하고, 셀당 결론 마진이 3pp 이내면 유의하지 않은
   것으로 처리

Dataset B는 gold/distractor를 segment 1–5에만 배치한다(0024 한계). X2의 신규 데이터도
동일 규약을 따를 것 — 비교 가능성 유지.

---

## 2. X1 — descriptor-only 예측 (H_blind 검정)

**목적**: layer별로 router가 query를 사용하는지 판정. 추가 forward pass 0회.

### 구현

E1이 이미 answer position에서 16 layer의 routing score `r_t^(i)`를 계산한다.
그 지점에서 `c_i` (segment descriptor, L2 정규화 전/후 둘 다), `u_t`, 선택된 top-2
인덱스를 덤프하도록 확장한다.

query를 전혀 쓰지 않는 특징 3개를 만든다.

```
pos_i       = i / (s-1)                       # 정규화 위치
norm_i      = ||c_i||_2                       # 정규화 전 descriptor norm
outlier_i   = 1 - <ĉ_i, mean_j(ĉ_j)>          # descriptor 중심에서의 거리
```

두 개를 측정한다.

1. **선택 예측**: 위 3개 특징만으로 실제 top-2 집합을 예측 (logistic regression 또는
   단순히 각 특징 단독 argmax-2와의 Jaccard). layer별로 보고.
2. **점수 회귀**: `r_t^(i) ~ pos_i + norm_i + outlier_i` 의 R².

동시에 공짜로 나오는 것 — 반드시 같이 기록:

3. **descriptor Gram off-diagonal 평균** `mean_{i≠j} <ĉ_i, ĉ_j>`, layer별.

### 판정

- 선택 예측 정확도 높음 / R² ≥ 0.9 → 그 layer는 query 미사용 (국소 H_blind)
- layer 9–13에서 위가 성립하고 layer 14–15에서 성립하지 않을 것으로 **예상**.
  예상과 다르면 그것 자체가 결과이므로 그대로 보고.
- Gram off-diagonal ≥ 0.8 → anisotropy 확정. 이 경우 **X1b**를 추가 실행:
  descriptor centering(`c_i - mean_j c_j`) 또는 top-1 PC 제거 후 hit@2 재계산.
  centering만으로 paired-D hit@2가 오르면 그 자체가 저비용 개선안.

### 출력

`results/x1_descriptor_only.json`
```
{layer: {"pred_jaccard": float, "r2": float, "gram_offdiag": float,
         "feat_r2_single": {"pos":…, "norm":…, "outlier":…}}}
```
그림: layer축 × (R², gram_offdiag) 2패널. 0024 Figure 2의 hit@2 곡선을 같은 x축에 겹칠 것.

---

## 3. X2 — off-template anomaly probe (갈림길)

**목적**: H_outlier vs H_template 판별. 이 결과가 이후 설계를 결정한다.

### 데이터

Dataset B(paired, 32쌍) 생성기를 확장한다. 기존 multi 예제에 **템플릿 없는 이상 문장**
1개를 추가 삽입한다.

- needle: 기존 RULER 템플릿 유지 (`The special magic number for {key} is {value}.`
  — 실제 문자열은 코드에서 확인)
- **off-template anomaly**: 맥락 없는 UUID 또는 랜덤 영숫자 한 줄. 질의어와 어휘 겹침
  0, 템플릿 구조 없음. haystack과는 분포적으로 명확히 다름.
- 배치: gold / distractor / anomaly를 **서로 다른 segment**에 (D 조건 준용).
  anomaly segment도 1–5 범위 내.
- 쌍맞춤 유지: anomaly 유무만 다른 쌍을 만들어 토큰 단위로 나머지 동일하게.

### 측정

anomaly segment가 top-2에 들어가는 비율 `hit@2(anomaly)`, layer별. 대조군은 같은
위치의 haystack-only segment.

### 판정

| 관측 | 결론 | 다음 설계 |
|---|---|---|
| `hit@2(anomaly)` 유의하게 높음 (gold와 경합) | **H_outlier** | routing 공간 분리 + contrastive supervision. X4 실행 안 함 |
| anomaly 무시, gold/distractor만 선택 | **H_template** | X4(P-sweep)를 메인 실험으로 승격 |
| 둘 다 아님 (전부 chance) | 재검토 — X1 결과와 교차 확인 | — |

n=32이므로 마진 3–4 샘플. 방향성 해석. 두 모델(5B/30B) 모두 측정.

### X2b (H_template로 판정된 경우에만)

**paired query-swap.** 같은 context에 대해 두 개의 정당한 query가 존재한다 —
gold key를 묻는 것, distractor key를 묻는 것. descriptor는 완전히 동일하고 query만
다르다. 정답 segment는 서로 다르다.

top-2 집합이 **바뀌지 않으면** router가 판별 부분공간을 사용하지 않는다는 직접 증거.
within-context 대조이므로 예제 간 분산이 제거되고 n=32로 결론이 난다.
측정: 두 query의 top-2 Jaccard, 그리고 각각의 hit@2.

### 출력

`results/x2_offtemplate_probe.json`, `results/x2b_query_swap.json` (조건부)

---

## 4. X3 — random-routing MC ablation

**목적**: 현재 앵커 표에서 MC의 장문 single-needle 이득이 routing 품질에서 오는지,
아니면 segment당 write 부하가 256으로 상한되는 효과인지 분리. **X1/X2와 독립이므로
병렬 실행 가능.**

### 근거

현재 앵커에서 MC 5B의 S-NIAH-1은 8K→32K에서 24/30/34로 회복되고 vanilla 5B는
16/4/0으로 붕괴한다. 그런데 32K에서 vanilla는 state 하나가 32K를 흡수하고 MC는 각
state가 256만 쓴다. **routing이 완전 무작위여도 읽어낸 state가 더 깨끗하다.**
이 ablation이 없으면 "MC의 이득이 routing에서 온다"는 문장을 쓸 수 없다.

### 구현

추론 경로에서 top-k 선택만 교체. 가중치 재학습 없음.

- `random`: 과거 segment에서 k개 균등 샘플 (seed 고정, 5 seed)
- `recent`: 항상 가장 최근 k개 (위치 baseline)
- `stock`: 변경 없음 (sanity — byte-identical 확인, 0024 E2와 동일 규약)

gate `γ`는 oracle 개입(0024 E2)과 동일 규약으로 처리: 선택된 memory가 gate에서
불리해지지 않도록 `score = max(선택 최대, online)`.

### 측정

S-NIAH-1, MK-NIAH-1 @ {1K, 2K, 4K, 8K, 16K, 32K}, 5B/30B.

### 해석 보조

`chance-routing 상한 = 2/(N-1) × conditional_EM` 을 같이 계산해 표에 병기한다.
`N = ctx/256`, `conditional_EM`은 0024 oracle 범위 0.31–0.56 사용(0.6으로 상한 근사).
관측 / 상한 비율이 길이에 따라 커지면 routing이 실제로 작동하는 증거.

### 출력

`results/x3_random_routing.json` + `stock / random / recent / vanilla`
4행 × 6열 표 (markdown), 길이별 배율 열 포함.

---

## 5. X4 — sub-block MaxSim P-sweep (조건부)

**X2가 H_template로 판정된 경우에만 실행.**

**목적**: index granularity만 세분화해 paired-D hit@2가 회복되는지. state는 256 그대로,
새 파라미터 0개, 재학습 없음.

### 구현

각 segment의 256개 key를 `P`개 sub-block으로 stride 분할, 각각 mean-pool + L2 정규화.

```
s_i = max_{p ∈ [P]} <û_t, ĉ_ip>
```

`P ∈ {1, 2, 4, 8, 16, 32}`. **P=32까지 반드시 포함.**

### 예측 (스펙에 명시하는 이유: 곡선 형태 자체가 결과)

sub-block 길이 `C' = 256/P`일 때 판별 SNR:

```
signal(판별 토큰 1–2개) ∝ 1/C'
noise(haystack C'-1개 key 평균의 segment간 변동) ∝ 1/√C'
⇒ SNR ∝ 1/√C' , 즉 P 대비 개선은 √P 배
```

MaxSim이 P개 노이즈의 최대를 취하는 `√(2 ln P)` 페널티를 감안하면 **P=8 실패가
기본 예측**이다. P=1 대비 순 개선은 대략 `√P / √(2 ln P)`.

측정 hit@2 곡선을 이 예측 곡선과 겹쳐 그린다. 정량 법칙이 되므로 단순 ablation보다
기여도가 높다.

### Kill-check

> **P=32에서도 paired-D best-layer hit@2가 0.6을 넘지 못하면 index granularity 가설
> 기각.** routing 공간 분리 / supervision 방향으로 전환.

(P=8 기준으로 판단하지 말 것 — 위 √P 스케일링 때문에 실패가 기본 예측이다.)

### 출력

`results/x4_psweep.json`, 그림: P축 × hit@2, 예측 곡선 overlay, paired-S/D 분리.

---

## 6. 판정 규칙과 산출물

### 공통 규약

- best-layer 방식은 0024와 동일하게 유지 (layer별 보고 + best-layer 요약)
- **chance level을 데이터셋마다 따로 계산해 병기.** 0024에서 multikey 0.486,
  single/paired 0.286–0.288. 신규 X2 데이터는 다시 계산할 것
- n이 작다(16–32). 절대값 주장 금지, 4/4 셀 단조성 같은 방향성으로 서술
- bf16 노이즈 2.3pp 이내 차이는 결론에 쓰지 않음
- 모든 수치는 JSON에 먼저 쓰고, 보고서는 JSON만 참조 (하드코딩 금지)

### 보고서

`report/0025.md` (영문, 수치·재현 커맨드 원본) + `report/0025_ko.md` (한국어 요약).
0024 형식 따를 것 — **맨 앞에 목차**, 한 줄 결론, 셋업, 실험별 절, 판정표, 한계, 재현.

판정표는 이 형식으로:

| 가설 | 판정 | 핵심 근거 | 함의 |
|---|---|---|---|
| H_blind (layer 9–13) | | X1 R², 선택 예측 | |
| H_outlier | | X2 hit@2(anomaly) | |
| H_template | | X2 + X2b Jaccard | |
| MC 이득 = routing | | X3 random vs stock | |

### 최종 산출

1. 위 판정표
2. X3 표 (chance 상한 병기) — 앵커 공백 메움
3. 다음 설계 권고 1페이지: X2 분기 결과에 따라 (a) routing 공간 분리 + contrastive
   supervision, 또는 (b) sub-block descriptor + checkpoint granularity 통합

---

## 7. 하지 말 것

- **학습 job 제출 금지.** X1–X4 전부 추론/후처리다. 재학습이 필요해 보이면 멈추고 보고
- fla 버전 올리지 말 것 (GLA API 비호환)
- Dataset B의 gold/distractor segment 범위(1–5)를 바꾸지 말 것 — 0024와 비교 불가해짐
- 0024의 E2 oracle 결과를 재측정하지 말 것 (이미 있음, `e2_oracle.json` 사용.
  `e2_join`은 정보량 부족으로 사용 안 함)
- anti-oracle 구현하지 말 것 — X1/X2가 더 싸고 정보량이 많아 대체됨
- 위치 정보를 특징에서 빼지 말 것. `pos_i` 단독 R²가 높을 가능성이 실제 후보다
- 결과가 예상과 다를 때 예상에 맞추려고 지표를 바꾸지 말 것. 다른 결과는 그대로 보고
