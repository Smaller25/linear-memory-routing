# Lab-meeting figures — status & specs (2026-07-13)

Per-figure: **status**, best form, where the figure/raw-data is, settings, and the read-off. Folder:
`~/labmeeting_figures/` (`candidates/` = existing PNGs pulled from prior experiments; `code/` = scripts).

**Status legend** — 🟢 있음(기존 그대로 or 미세수정) · 🟡 재생성 가능(raw data 저장돼 있음) · 🔴 생성 필요(새 실험/로깅) · ⬜ 개념도(데이터 불필요) · 🗓️ 계획(아직 안 돌림, "결과인 척" 금지)

## Evidence-grade summary (교수님이 한눈에 볼 수 있게)
- **결과 있음**: F3🟢(생성완료), F5🟡(부분), F6/F9🟡(erank 관련 기존 플롯 다수).
- **개념도**: F1, F2, F11 ⬜.
- **생성/로깅 필요**: F7, F8 🔴 (per-step aₜ,Bₜ 로깅 필요), F12 🔴(싼 실험).
- **문헌 재플롯**: F4 (DynLA Table 3 숫자 필요).
- **계획(아이디어)**: F10, 그리고 섹션 3 전부 🗓️.

---

## F1 ⬜ 개념도 — memory 위계 (agentic vs architecture-state)
- best: 2단 다이어그램, "검색해 context에 넣어도 같은 고정 state로 재압축 → context 내 회수 실패는 agentic이 원리적으로 못 건드림" 한 문장.
- source: 데이터 불필요. **새로 그림(draw.io/PPT)** 권장. 코드 대상 아님.

## F2 ⬜ 개념도 — 2×2 지형도 (boundary 정책 × 용량 처리)
- best: x=고정↔동적, y=파괴적merge↔비파괴cache. Log-linear=(고정,merge), DynLA=(동적,merge), MC=(고정,cache), **(동적,cache)=빈칸=본인**.
- source: 데이터 불필요. 다이어그램. 판독: "chunking이 주장이 아니라 지형의 빈자리".

## F3 🟢 생성완료 — additive merge가 delta 대수 위반 (본인 논증)
- **PNG: `F3_delta_toy.png`** / 코드: `code/F3_delta_toy.py` (CPU, 모델 불필요).
- 세팅: d=16, delta rule(β=1, unit key). k→A 저장 후 같은 k→B 갱신. (a) 단일 연속 state vs (b) 2-세그먼트 additive read.
- **결과: single A=0.00/B=1.00 (A 정상 삭제), additive A=1.00/B=1.00 (삭제된 A 부활).**
- 판독: additive는 어떤 스칼라 가중을 줘도 지운 값을 못 없앰 → merge 아닌 cache. **"본인 분석"임을 슬라이드에 명시.**

## F4 🔴 문헌 재플롯 — DynLA Table 3 (Mamba-2 vs GDN, S-NIAH/MK/MQ/MV)
- best: grouped bar, 출처 명기. 판독: GDN이 다 이기는데 MV만 역전(14.8 vs 19.4) = erase의 양날 = F3의 실증판.
- source: **논문 Table 3 숫자 필요** (본인 실험 아님). 숫자 주시면 재플롯.

## F5 🟡 재생성 가능(부분) — proxy: token-level 실패 → state-level 가능성
- best: task별 accuracy bar, series={token-level signal, state-level(erank/epiplexity), 고정분할 baseline}, **동일 checkpoint budget** 캡션. raw 있으면 budget K vs acc Pareto가 더 강함.
- **기존 후보**:
  - `candidates/F5_multikey_failure_evidence.png`, `F5_single_vs_multi_4k.png` — descriptor(token/pooled)-level 신호가 multi에서 무너짐 (long-gdn, GDN-1.3B).
  - `candidates/F5_signal_matrix_delta.png`, `F5_F9_signal_trajectories.png` — state-level 신호(S1 eRank/S2 entropy/S3 epiplexity) × data (SSM_Rank).
  - `candidates/F5_epiplexity_chunk_by_density.png` + `F5_natural_passage_chunks.png` — epiplexity가 밀도-신호로 유효(합성 ρ0.94) but 자연어 degenerate.
- **주의(정직)**: 우리 token-level 실패 증거는 대부분 **from-scratch MQAR density(0018)** + **frozen descriptor(long-gdn)**로 나뉘어 있고, "frozen+SSC 단일 셋업에서 token vs state를 한 판에" 그린 깨끗한 plot은 **아직 없음** → 재생성 시 assemble 필요.
- raw data: `SSM_Rank_Analysis/notebooks/capacity_results/*.json`, `linear-memory-routing/report/0016,0018` 수치.

## F6 🟢 생성완료 (VESSL A100) — erank 킬러 플롯 (r̄ vs erank, plain GDN2-370m 6B)
- best: head별 산점도 + 이론곡선 y=min(64, e/(1−x)) 오버레이, Type 색분.
- **기존 후보(근사)**:
  - `candidates/F6_erank_vs_MQARload.png` — eRank vs MQAR load (양 모델).
  - `candidates/F6_F9_load_vs_horizon.png` — **eRank↑인데 recall↓ (anti-correlation)** = "erank≠capacity" 직접 증거 (SSM_Rank §2).
  - `candidates/F6_alt_a_rank.png`, `F6_per_head_change.png` — head별 rank / 변화.
- **주의**: 정확한 "r̄ vs erank + 이론곡선" 산점도는 **아직 없음**(교수님께 보일 핵심). 지금 돌리는 분석(decay vs geometry)이 이걸 채울 것. `utils.effective_rank`가 entropy 정의인지 확인해 곡선 상수 맞출 것.
- raw data: forward에서 (aₜ,Bₜ) 로깅 필요 → 아래 F7과 공유.

## F7 🔴 생성 필요 — 반사실 분해 (decay vs key-anisotropy)
- best: 2×2 bar {실제 decay, aₜ=1} × {실제 key, 등방 랜덤 key}의 erank.
- source: **per-step (aₜ,Bₜ,xₜ) 로깅 후 state 재구성(학습 불필요)**. 로깅 코드 신규. `capacity_utils.get_ssm_states`는 최종 state만 주므로 mixer 내부 로깅 추가 필요.
- 판독: decay 몫과 key 몫이 가법 분리 → F6의 인과판.

## F8 🔴 생성 필요 — head별 Bₜ 코사인 유사도 히스토그램 + 등방 null
- source: F7과 동일 로깅(Bₜ) 재사용. 판독: 오른쪽 치우침 = key 뭉침 직접 증거.

## F9 🟡 재생성 가능(부분) — erank(Sₜ) vs 위치 + 그 위치 needle recall
- best: x=위치 t, y1=erank(Sₜ), y2=심은 needle 최종 recall. 정렬 = "진짜 용량인가" 상관 증거.
- 기존 후보: `candidates/F5_F9_signal_trajectories.png`(erank over position), `F9_state_capacity_sweep.png`, `F6_F9_load_vs_horizon.png`.
- **주의**: recall-by-position 오버레이는 **새 측정 필요**(needle을 위치별로 심고 recall). erank 궤적은 있음.

## F10 🗓️ 계획 — fully-span 가설 (원본 key vs whitening key → recall)
- 아직 안 돌림. **"다음 실험"으로 제시** (결과인 척 금지). 학습 불필요(재구성).

## F11 ⬜ 개념도 — hot state(GPU 고정예산) / cold dictionary(host) 계층
- best: spill은 write 길목에서 복사(state 역복호 아님) 화살표. Transformer KV-cache 계층화를 bounded-state에 이식.

## F12 🔴 생성 필요(싼 상한) — state only vs state + oracle dictionary (MK/MV NIAH)
- best: bar. 판독: headroom 크기 = 이 방향 가치 상한. 1.3B로 충분. "정말 가능한가?"의 정직한 첫 답.
- source: 신규 실험. frozen 모델 + needle KV를 따로 저장해 read에 합침. eval 인프라 재사용(eval_ruler류).

---

## 용어 정정 (Fable 지적 반영)
- **epiplexity = 압축가능(구조적) 정보**. UUID/전화번호 같은 exact-recall 항목은 오히려 **time-bounded entropy(압축불가 잔여)** 쪽. dictionary 후보 신호 = **epiplexity 낮고 entropy 높은** 쪽. F5/섹션3 서술 시 이 방향으로.
- LongMemEval: 1.3B 순수 LM엔 무리 → **knowledge-update 카테고리만 MQAR-style synthetic으로 축소 이식**하면 소형 모델로 검증 가능.

---
## F8 🟢 생성완료 (VESSL A100) — "실제로 뭉치는지" 직접 확인 (plain GDN2-370m 6B)
- PNG: `F8_concentration.png` / code: `code/F8_concentration.py`.
- (좌) state 특이값 스펙트럼: 실제 head vs random Gaussian state — top σ 점유율 head0 **29.3%** vs 등방 **1.8%** → 상태가 소수 방향 집중(뭉침).
- (우) key 쌍 코사인: 실제 **0.605** vs isotropic null **0.004** → key가 같은 방향으로 뭉침(저rank state의 원인).
- 판독: F6의 erank-gap(추론)을 직접 시각화. 단, key-cosine은 일반 LM hidden anisotropy도 반영(등방 null은 극단) → state SV 집중이 더 깨끗한 지표.
