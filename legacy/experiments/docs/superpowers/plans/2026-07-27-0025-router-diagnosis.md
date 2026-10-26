# 0025 Router 진단 (X1–X4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자 스펙 `docs/superpowers/specs/2026-07-27-0025-router-diagnosis-spec.md`대로 H_blind/H_outlier/H_template 판별(X1, X2) + random-routing ablation(X3) + 조건부 P-sweep(X4)을 실행하고 report/0025 작성.

**Architecture:** 0024 인프라(`lmr/analysis/260725_mc_niah_analysis/`) 재사용. 핵심 설계: **X1은 GPU 덤프 1회 + CPU 후처리**(u_t와 32-sub-block 원시 descriptor를 덤프하면 X1·X1b·X4가 전부 CPU 재계산이 됨), X2는 신규 paired 데이터 + 1-pass routing 측정, X3는 **segment-cached incremental generation 엔진**(신규; independent checkpoint 모드라 수학적으로 full forward와 동치)으로 6개 길이 그리드를 감당.

**Tech Stack:** 기존 스택 그대로 (pinned worktree e71713e, fla 4b02d15d, sh_infocap python, SLURM sbatch).

## Global Constraints

- **사용자 스펙이 최상위 문서** — 이 플랜과 충돌하면 스펙이 이긴다. 스펙 §7 "하지 말 것" 전부 준수 (학습 job 금지, fla 업그레이드 금지, Dataset B segment 1–5 규약 유지, E2 재측정 금지, anti-oracle 금지, pos_i 특징 제거 금지, 지표 사후 변경 금지)
- GPU는 sbatch만 (`-p main --gres=gpu:rtx6000:1`); PY=/data2/sohyung/conda-envs/sh_infocap/bin/python; env_common.sh source (pinned fla PYTHONPATH)
- 코드 위치: `lmr/analysis/260725_mc_niah_analysis/` (기존 폴더에 x1_*.py, x2_*.py, x3_*.py, x4_*.py 추가). 기존 파일 수정은 최소화 — routing_stats.py/oracle.py/paired_gen.py는 **읽기 재사용** 위주, 수정 시 기존 e1/e2/e3 재현성 불변 유지
- 결과: `results/x*.json`(repo+`$MC_OUT/results/`), 대용량 덤프는 `$MC_OUT`만. **보고서는 JSON만 참조 (수치 하드코딩 금지)**
- 모든 수치 관례는 0024와 동일: 데이터셋별 chance 병기, best-layer + layer별, bf16 노이즈 2.3pp 이내 결론 금지, n 작음 → 방향성 서술
- seed 고정: X3 random은 seed {0,1,2,3,4}
- 커밋 트레일러: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; .sh/.sbatch는 git add -f

**검증된 코드 사실 (스펙 §1 체크리스트 수행 결과 — 구현자는 신뢰):**
- `data.py` CLI: `prepare-a --num-samples N` (length=2048 함수 인자는 CLI 미노출 — X3에서 노출 필요), `prepare-b --n-pairs N`. `annotate(text, tok)` → needles/gold_seg/n_seg/cur_seg 등
- `routing_stats.py`: `capture_hidden(model, ids)` → layer별 attn 입력 [T,D]; `routing_scores_at(attn, h, t)` 내부에서 `rk=F.normalize(k)`, `summaries=segment_key_sums(rk, 256)` [1,N,H,K], `u=connector(h[t])` [1,1,H,K], score=einsum. **여기가 X1 덤프 지점** — 같은 수식으로 별도 스크립트에서 재계산해 덤프
- e1 JSON 스키마: `results[model][dataset] = {per_layer:[{layer,hit_at_2,gold_rank_mean,amongkeys_acc,amongkeys_n,n_eligible,n_total}], n_total, n_eligible, mean_cur_seg, chance_hit2, best_layer, best_layer_hit_at_2}`
- `oracle.py`: `_make_oracle_class()`가 mc_ssc forward 복사본 + 주입 지점 패턴 제공 — X3 selection override의 템플릿. `patch_oracle(model)` 패턴으로 ssc 교체, stock 시 byte-identical 검증 함수 `_sanity_check_baseline_matches_unpatched` 존재
- `fidelity.py._scan`: chunk_gdn2 호출 규약 (initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False, cu_seqlens=None)
- MC config: independent checkpoint mode → **각 segment의 online 경로는 zero state에서 시작** (segment 간 정보는 오직 cached read로만 전달) — X3 incremental generation의 수학적 동치성 근거
- Dataset B: `$MC_OUT/data/paired/{S,D}.jsonl`, 토큰 정렬 쌍 (0024 Task 4 fix 이후), `_neutralize`/`_fit_filler`로 길이 일치 filler 치환 가능
- 앵커 (X3 vanilla 행 + stock 참조): collaborator 4-way 문서 — vanilla-5B S-NIAH-1 @1K..32K = 100/90/54/16/4/0, MK-NIAH-1 = 18/20/12/4/2/2; MC-5B S1 = 86/60/36/24/30/34, MK1 = 4/2/12/6/2/0; MC-30B S1 = 100/92/74/50/28/16, MK1 = 36/32/34/4/4/2. vanilla-30B는 앵커 없음(각주 처리)

---

### Task 1: X1 GPU 덤프 (`x1_dump.py`) — u_t + sub-block descriptor 저장

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/x1_dump.py`
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/x1.sbatch` (e1.sbatch 복제, 실행줄 교체, `-t 01:00:00`)

**Interfaces:**
- Produces: `$MC_OUT/x1_dump/{model}/{dataset}/{idx}.npz` — 키: `u` [L,H,K] fp16 (answer position, L=16 layers), `csub_raw` [L,N,32,H,K] fp16 (segment별 32개 contiguous sub-block의 **정규화 전 key mean-pool**; rk=normalize(k) 후 블록 평균 — 즉 "L2-norm된 key들의 블록 mean-pool", full-segment c_i는 32블록 평균으로 정확 복원됨), `stock_scores` [L,N] fp32 (routing_scores_at와 동일 수식, 미래/현재 -inf), `meta` (gold_seg, cur_seg, n_seg, eligible, key_segs, distractor_segs?, pair_id?, condition?)
- 대상: 2 모델(mc-5B, mc-30B) × 4 데이터셋(niah_single_1, niah_multikey_1, paired_S_multi, paired_D_multi — E1과 동일 로딩 `_rows_for` 재사용)

- [ ] **Step 1: x1_dump.py 작성.** `routing_stats.capture_hidden` + `attn._project` 재사용. 핵심:

```python
# per layer i, attn:
q, k, v, g, b, w = attn._project(h.unsqueeze(0))
rk = F.normalize(k.float(), p=2, dim=-1)              # [1,T,H,K] float32
T = rk.shape[1]; n_seg = (T + 255) // 256
csub = torch.zeros(n_seg, 32, rk.shape[2], rk.shape[3])
for s in range(n_seg):
    seg = rk[0, s*256:min((s+1)*256, T)]              # [<=256,H,K]
    C = seg.shape[0]; P = 32
    bounds = [round(p*C/P) for p in range(P+1)]        # 마지막 partial seg도 32분할
    for p in range(P):
        blk = seg[bounds[p]:bounds[p+1]]
        csub[s, p] = blk.mean(0) if blk.shape[0] else 0
u_t = attn.ssc.connector(h[T-1:T].unsqueeze(0)).view(attn.ssc.num_heads, attn.ssc.head_qk_dim)
```
`stock_scores`는 `routing_scores_at(attn, h, T-1)` 그대로 호출해 저장. **일관성 검증 내장**: full c_i(=csub 평균, partial seg는 블록 크기 가중 평균 — bounds 기반 가중이므로 단순 평균이 아니라 `seg.mean(0)`을 별도 저장·비교하거나, 재구성 오차 <1e-3 assert.

- [ ] **Step 2: sbatch 제출·완료 대기** (두 모델 순차, ~수 분). 로그에서 assert 통과 확인
- [ ] **Step 3: 덤프 무결성 검증** — 재구성한 stock top-2가 e1 산출과 일치하는지: 각 dataset의 layer별 hit@2를 덤프에서 재계산해 `results/e1_routing.json`의 per_layer.hit_at_2와 비교 (bf16 노이즈 감안 ±3pp; best-layer는 일치해야 함). 검증 스크립트 출력물을 리포트에 포함
- [ ] **Step 4: 커밋** (`mc-niah 0025: X1 dump job (u_t + 32-subblock descriptors)`)

---

### Task 2: X1 CPU 분석 (`x1_analyze.py`) — H_blind 검정 + X1b

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/x1_analyze.py`
- Test: `tests/lmr/test_mc_niah_x1.py` (합성 텐서로 특징·R²·Jaccard 계산 검증, CPU)

**Interfaces:**
- Consumes: Task 1 덤프
- Produces: `results/x1_descriptor_only.json` (repo+MC_OUT) — 스펙 §2 스키마 + 확장: `{model: {dataset: {layer: {pred_jaccard: {pos,norm,outlier,combined}, r2, gram_offdiag, feat_r2_single:{pos,norm,outlier}}}}}` + `x1b` 키 (조건부); 그림 `results/x1_descriptor_only.png` — layer축 × (R², gram_offdiag) 2패널 + e1 hit@2 곡선 overlay (스펙 요구)

- [ ] **Step 1: 실패하는 테스트 작성** — 합성 케이스: (i) scores가 정확히 pos_i의 선형함수면 r2≈1, pos의 pred_jaccard=1; (ii) scores가 u·ĉ로 만들어지고 특징과 무관하면 r2 낮음; (iii) gram_offdiag 계산이 손계산과 일치
- [ ] **Step 2: 구현.** 특징 (descriptor는 [H,K] → **H·K로 flatten**):

```python
c_full_raw = csub.mean(axis=1)                        # [N,H,K] 검증된 재구성 (블록 가중 주의)
c_flat = c_full_raw.reshape(N, -1)                    # 정규화 전
c_hat = c_flat / (norm(c_flat, axis=1, keepdims=True) + 1e-8)
pos = arange(N) / max(N-1, 1)
nrm = norm(c_flat, axis=1)
outlier = 1 - c_hat @ c_hat.mean(0) / (norm(c_hat.mean(0)) + 1e-8)
gram_offdiag = (c_hat @ c_hat.T)[off_diagonal].mean()
```
- eligible segment만 사용 (cur_seg 이전). ineligible 샘플 제외 규약은 e1과 동일
- **점수 회귀 R²**: 샘플 내 z-score한 stock_scores를 (sample×eligible seg) 풀링, `lstsq([1,pos,norm,outlier])`. feat_r2_single은 단일 특징 회귀
- **선택 예측**: 특징 단독 argmax-2와 실제 top-2의 Jaccard (pos는 큰 쪽/작은 쪽 둘 다 시도해 좋은 쪽 방향을 기록 — 방향도 결과), combined = 회귀 적합 점수의 argmax-2
- **X1b (조건부: 어떤 layer든 gram_offdiag ≥ 0.8)**: centering(`ĉ_i - mean_j ĉ_j`) 및 top-1 PC 제거(샘플별 eligible ĉ SVD) 후 `<û, ·>` argmax-2로 hit@2 재계산 (u도 normalize — 순위 불변이므로 raw u와 동일), 4 데이터셋 layer별. **덤프만으로 계산, GPU 불필요**
- [ ] **Step 3: 테스트 통과 + 실행 + 그림 생성**
- [ ] **Step 4: 판정 기록** — 스펙 §2 판정 규칙대로 layer 9–13 vs 14–15 국소 H_blind 여부를 JSON `verdict_notes`에 서술. 예상과 달라도 그대로 기록
- [ ] **Step 5: 커밋** (results 포함)

---

### Task 3: X2 off-template anomaly probe (+조건부 X2b)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/x2_data.py` (paired_gen 재사용 확장)
- Create: `lmr/analysis/260725_mc_niah_analysis/x2_probe.py` (1-pass routing 측정)
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/x2.sbatch`
- Test: `tests/lmr/test_mc_niah_x2.py`

**Interfaces:**
- Produces: `$MC_OUT/data/paired/X2.jsonl` (32쌍×2행: variant ∈ {anom, noanom}), 조건부 `X2B.jsonl` (32쌍×2행: query ∈ {gold, distractor}); `results/x2_offtemplate_probe.json`, 조건부 `results/x2b_query_swap.json`

- [ ] **Step 1: 실패하는 테스트** — X2 데이터 불변식: 쌍은 anomaly 문장 자리만 다르고(길이 일치 filler 치환, `_fit_filler` 재사용) 나머지 토큰 동일; gold/distractor/anomaly 전부 서로 다른 segment (D 준용), 전부 segment 1–5; anomaly 문장은 needle 템플릿·질의어와 어휘 겹침 0 (UUID 형식 `"a3f8-0b2e-..."` 랜덤 영숫자 1줄, 정규식으로 needle 패턴 미매치 assert)
- [ ] **Step 2: x2_data.py 구현** — `paired_gen.build_pairs`의 D-조건 로직 재사용해 multi 컨텍스트 구성 후, 별도 segment에 anomaly 삽입(anom) / 같은 자리 filler(noanom). `annotate` 확장 불필요 — anomaly seg 인덱스는 생성 시 기록하고 재토크나이즈로 실측 검증. 32쌍 생성
- [ ] **Step 3: x2_probe.py** — e1과 동일한 1-pass: `capture_hidden` + `routing_scores_at`, layer별로 anomaly seg ∈ top-2 비율 (anom 행), 대조군 = noanom 쌍의 같은 seg 인덱스 hit율, gold hit@2도 병기. 두 모델. chance 재계산(스펙 §6). sbatch 제출 (~수 분)
- [ ] **Step 4: 판정** — 스펙 §3 표 그대로: anomaly 경합 → H_outlier / anomaly 무시 → H_template / 둘 다 chance → X1과 교차 확인. n=32 마진 3–4 샘플 방향성. **판정 결과를 coordinator에 보고하고 X2b/X4 실행 여부 지시받기** (구현자가 스스로 분기하지 말 것)
- [ ] **Step 5 (X2b, H_template 판정 시에만)**: X2B.jsonl — 같은 D-조건 컨텍스트에 질문만 gold key ↔ 특정 distractor key로 교체한 쌍. 측정: 두 query의 top-2 Jaccard(layer별) + 각자 hit@2. 1-pass, 같은 sbatch 패턴
- [ ] **Step 6: 커밋** (data 스크립트/테스트/결과)

---

### Task 4: X3 random-routing ablation (헤비 태스크)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/x3_gen.py` (incremental generation 엔진 + selection override)
- Create: `lmr/analysis/260725_mc_niah_analysis/x3_run.py` (그리드 드라이버, skip-existing 머지)
- Modify: `lmr/analysis/260725_mc_niah_analysis/data.py` (prepare-a에 `--length`/`--tasks` CLI 노출만)
- Create: `lmr/analysis/260725_mc_niah_analysis/sbatch/x3.sbatch` (MODEL·TASK env로 분할)
- Test: `tests/lmr/test_mc_niah_x3.py` (selection override 순수 로직 CPU 테스트)

**Interfaces:**
- Produces: `results/x3_random_routing.json` — `{model: {task: {length: {"stock": s, "recent": s, "random": {"mean": s, "per_seed": [5], }, "chance_upper": 2/(N-1)*0.6}}}}` + `results/x3_table.md` (4행: stock/random/recent/vanilla-anchor × 6열 + 배율)

- [ ] **Step 1: 데이터 준비** — `data.py prepare-a --length {1024,2048,4096,8192,16384,32768} --tasks niah_single_1 niah_multikey_1 --num-samples 50` (CLI 노출 후 CPU 실행; 2048은 기존 재사용)
- [ ] **Step 2: incremental generation 엔진.** independent checkpoint 모드라 **수학적으로 full forward와 동치**인 경로: layer별로 (완결 segment의 memory [H,K,V], summary c_i [H,K])를 freeze하고, 매 스텝 현재 partial segment만 전 layer 재실행 + frozen summaries로 top-k + frozen memories로 read. 프리필도 segment 단위 순차 처리(동일 코드 경로). oracle.py의 forward 복사본을 기반으로 `selection_mode ∈ {stock, random, recent}` 파라미터화:

```python
if mode == "stock":  top_indices = topk(scores)          # 변경 없음
elif mode == "recent": top_indices = [N-1, N-2][:k]      # 최근 k개
elif mode == "random": top_indices = rng.sample(range(N), min(k, N))
# random/recent의 score는 E2 규약: max(선택된 것들의 stock score 최대, online) — gate 불리 방지
```
선택 override 로직은 순수 함수로 분리해 CPU 테스트 (eligibility: 과거 segment만, N=0이면 online만)
- [ ] **Step 3: 동치성 검증 게이트 (필수, 그리드 전에)** — 2048에서 8샘플: incremental(stock) greedy 토큰열 == 기존 `gen_eval.greedy_generate`(full re-forward) 토큰열. 전부 일치해야 진행; argmax flip 발생 시 최대 logit 오차와 함께 보고하고 **중단·coordinator 보고** (프로토콜 결정 필요). 추가: patched-stock == unpatched (E2 `_sanity_check` 재사용, 2샘플)
- [ ] **Step 4: 그리드 실행** — 2 모델 × 2 태스크 × 6 길이 × {stock, recent, random×5seed} = 168셀, 50샘플, n_gen=128. sbatch를 (model×task) 4개 job으로 분할, 각 job 내 길이 오름차순 + skip-existing 머지 (JSON에 셀 단위 기록, 재제출 안전). `-t 05:50:00`, 예상 각 2–4h. **32K에서 첫 셀 완료 시점의 per-sample 시간을 로그로 확인하고, 잔여 예산 초과가 예상되면 32K/16K만 25샘플로 감축 후 JSON meta에 기록** (프로토콜 변경은 셀 단위로 균일하게: stock/recent/random 같은 n)
- [ ] **Step 5: 표 생성** — `chance_upper = 2/(N-1) × 0.6` 병기 (N=ctx/256), 관측/상한 배율 열, vanilla 행은 앵커 인용 (5B만; 30B는 앵커 없음 각주). 배율이 길이에 따라 커지는지가 판정 신호 (스펙 §4)
- [ ] **Step 6: 커밋** (코드 먼저, 결과 나중 — 그리드가 길어 중간 커밋 허용: 셀 단위 머지라 부분 결과도 유효)

---

### Task 5 (조건부: X2 = H_template일 때만): X4 sub-block MaxSim P-sweep

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/x4_psweep.py` (CPU only — Task 1 덤프 재사용)
- Test: `tests/lmr/test_mc_niah_x4.py`

**Interfaces:**
- Consumes: Task 1 덤프 (`csub_raw` [L,N,32,H,K], `u`)
- Produces: `results/x4_psweep.json` + `results/x4_psweep.png` (P축 hit@2, paired-S/D 분리, 예측 곡선 overlay, stock 참조선)

- [ ] **Step 1: 실패하는 테스트** — 합성: 판별 블록 하나에만 신호를 심으면 P가 클수록 hit 상승; P=1이 full-segment 정규화 descriptor와 일치
- [ ] **Step 2: 구현** — P ∈ {1,2,4,8,16,32}: 32블록을 P그룹으로 재평균(**블록 크기 가중** — partial segment 주의) → 그룹별 L2 정규화 ĉ_ip → `s_i = max_p <û_t, ĉ_ip>` → top-2 → paired S/D hit@2, layer별+best-layer. 예측 곡선 `g(P)=√P/max(1,√(2lnP))`, g(1)=1 정규화, 트윈축 overlay. **P=32 필수 포함**
- [ ] **Step 3: Kill-check 기록** — P=32 paired-D best-layer hit@2 > 0.6 여부를 JSON `kill_check`에 명시
- [ ] **Step 4: 커밋**

---

### Task 6: report/0025.md + 0025_ko.md + 마무리

**Files:**
- Create: `report/0025.md` (영문), `report/0025_ko.md` (한국어 요약), `report/0025_figs/`
- Modify: `report/README.md` (인덱스 1줄)

- [ ] **Step 1:** 스펙 §6 형식 그대로 — 맨 앞 목차, 한 줄 결론, 셋업(고정 해시 명기), X1/X2(/X2b)/X3(/X4) 절, **판정표(스펙 §6 4행 형식)**, X3 표(chance 상한 병기), 다음 설계 권고 1페이지 (X2 분기 결과에 따라 (a) routing 공간 분리+contrastive supervision 또는 (b) sub-block descriptor+checkpoint granularity), 한계, 재현 커맨드. **모든 수치는 results/*.json에서 읽어 기입** — 스크립트로 대조 후 작성
- [ ] **Step 2:** 리뷰 subagent의 수치 전수 대조 후 커밋; 프로젝트 메모리 갱신 (coordinator)

## Self-Review

- 스펙 커버리지: §2→Task1+2, §3→Task3, §4→Task4, §5→Task5(조건부), §6→Task6, §7 금지사항→Global Constraints. X1→X2→분기, X3 독립 — 실행 순서 1,2,3,4,(5),6으로 스펙 우선순위 충족 (X3는 X1/X2 뒤에 두지만 결과 독립이므로 순서 무해)
- Placeholder: 핵심 수식·override·덤프 코드 제공, 나머지는 기존 파일 참조로 지정 (구현자가 해당 파일을 읽는 것이 전제 — 파일명·함수명 명시함)
- 타입 일관성: 덤프 npz 키(u/csub_raw/stock_scores/meta)를 Task 2·5가 동일 명칭으로 소비; x3 JSON 스키마 명시
