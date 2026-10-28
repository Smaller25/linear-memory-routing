# 0025 Router 진단 rev2 (X1–X5, VESSL 실행) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **이 플랜은 `2026-07-27-0025-router-diagnosis.md`(rev1 플랜)를 대체한다.** rev1 플랜은 기록용으로 유지.

**Goal:** 사용자 스펙 rev2 (`docs/superpowers/specs/2026-07-27-0025-router-diagnosis-spec.md`) 실행 — 상위 주장 (M) 검정(X2) + H_blind/H_outlier/H_template 판별(X1/X3) + `u_t:=q_t` 검증(X1b) + random-routing ablation(X4) + 조건부 P-sweep(X5). **GPU 작업 전부 VESSL에서** (사용자 지시).

**Architecture:** GPU = VESSL 워크스페이스(SSH, `~/sohyung2.pem`), CPU 전처리·후처리 = greenbeard 로컬 (dynmc 교훈: 전처리는 로컬 96코어). 데이터는 로컬 생성 → rsync 업로드; 덤프/결과는 rsync 회수 → repo `results/` + `/data2/sohyung/mc_niah/`. X1 덤프에 `q_t`를 추가하면 X1a/X1b/X1c/X5가 전부 로컬 CPU 후처리가 된다.

**Tech Stack:** 기존 + VESSL (persistent `/root/smaller` geesefs, 컨테이너 재시작 대비 idempotent bootstrap).

## Global Constraints

- **사용자 스펙 rev2가 최상위** — 충돌 시 스펙 우선. §8 "하지 말 것" 전부: 학습 job 금지, **"routing 공간 분리" 처방 언급 금지**(§0.3 철회 — u_t와 q_t는 묶는 방향), fla 업그레이드 금지, Dataset B segment 1–5 유지, E2 재측정 금지, pos_i 제거 금지, 지표 사후 변경 금지
- **GPU 실행은 전부 VESSL** (사용자 지시 "전부다"): greenbeard sbatch 사용 금지. greenbeard 로컬은 CPU 작업(데이터 생성·분석·rsync)만
- VESSL 접속: `ssh -i /home/sohyung/sohyung2.pem` + 사용자 제공 endpoint (T0에서 확정). 영구 저장은 `/root/smaller`만; **geesefs에 스크래치 쓰기 금지** (TMPDIR/TRITON_CACHE/HF_HOME은 컨테이너 로컬 디스크), 컨테이너 재시작으로 pip·ssh키 소실 가능 → bootstrap은 idempotent, 장기 job은 셀 단위 재개 가능해야 함
- 코드는 이 repo 브랜치 `sh/mc-niah-analysis`에 커밋 → VESSL에서 pull (양방향 코드 이동은 git으로만; 결과 JSON/덤프만 rsync)
- 수치 규약 (스펙 §7): 데이터셋·길이별 chance 재계산 (multi-query는 정답 segment 복수 → 식 재도출), best-layer + layer별, bf16 노이즈 2.3pp, n 작음 → 방향성, JSON 먼저 쓰고 보고서는 JSON만 참조
- X4 random seed {0..4}; 커밋 트레일러 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; .sh는 git add -f

**검증된 사실 (스펙 §1 체크 완료):**
- vendored RULER에 `niah_multiquery`(k=1,q=4 → `num_needle_k=max(k,q)` 클램프로 **key 4·전부 질의**)와 `niah_multivalue`(k=1,v=4) 존재 — multikey_1(k=4,q=1)과 needle 수 정합. needle 수 스윕은 `niah.py` 직접 호출(`--num_needle_q 2`)로 가능
- `routing_scores_at` 수식/덤프 지점, e1 JSON 스키마, `_project` 재사용, oracle.py override 패턴: rev1 플랜의 "검증된 코드 사실" 블록 그대로 유효
- q_t는 `_project`의 첫 반환값 — kernel 규약대로 L2 normalize 후 사용; u와 같은 [H,K] (num_v_heads×head_qk_dim) → **차원 일치, head별 내적 합 방식 stock과 동일하게 적용** (스펙 §2b의 "차원 불일치 가능성" 항목에 이 확인 결과를 기록할 것)
- 미커밋 파일 `x1_dump.py`/`sbatch/x1.sbatch`(rev1 산출, 실행 안 됨)가 워킹트리에 있음 — T1 구현자는 x1_dump.py를 rev2로 수정(q 추가)해 재사용, sbatch는 삭제하지 말고 두되 사용하지 않음
- 앵커 표 (X4 vanilla 행 인용): rev1 플랜 Global Constraints의 앵커 블록 그대로

---

### Task 0: VESSL 부트스트랩 + smoke

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/vessl/bootstrap.sh` (워크스페이스에서 실행, idempotent)
- Create: `lmr/analysis/260725_mc_niah_analysis/vessl/run_remote.sh` (로컬에서 ssh 실행 래퍼: `run_remote.sh <스크립트> [args...]` → nohup + 로그 + PID/마커, `--poll` 옵션)
- Create: `lmr/analysis/260725_mc_niah_analysis/vessl/env_vessl.sh` (VESSL 쪽 env: PYTHONPATH=pydeps:worktree:worktree/dsc, HF_HOME=/tmp/hf_home, TMPDIR/TRITON_CACHE=/tmp, MC_OUT=/root/smaller/mc_niah)

- [ ] **Step 1 (BLOCKER 해소):** 사용자에게 endpoint 확인 — coordinator가 전달. 접속 검증: `ssh -i /home/sohyung/sohyung2.pem <endpoint> "hostname; nvidia-smi -L"` (GPU 종류 기록)
- [ ] **Step 2: bootstrap.sh** — /root/smaller/mc_niah/{code,data,results,logs,ckpts,pydeps} 구성:
  - lmr repo: `/root/smaller/mc_niah/code/linear-memory-routing` clone/pull 브랜치 `sh/mc-niah-analysis` (토큰 `/root/smaller/.gh_token_new` 존재 확인, 없으면 BLOCKED 보고)
  - long-gdn: public clone → `git checkout e71713e` (worktree 아닌 일반 checkout)
  - fla: clone → `git checkout 4b02d15d` → `pip install --no-deps --target /root/smaller/mc_niah/pydeps .` (영구 저장이라 1회)
  - 휘발 deps 매 부팅 재설치: `pip install -q einops transformers huggingface_hub numpy matplotlib` (이미지에 torch 전제 — 버전 기록)
  - ckpt: 3개 .pth를 `/root/smaller/mc_niah/ckpts/`에 1회 다운로드(hf_hub_download → geesefs로 이동), 부팅 시 로컬 /tmp/hf_home로 복사하는 함수 포함
- [ ] **Step 3: smoke** — 0024 `smoke.py`를 VESSL 경로로 실행 (load_mc.py의 WORKTREE/ckpt 경로를 env로 오버라이드 가능하게 소폭 수정: `MC_LONGGDN_WORKTREE` env는 이미 지원, ckpt는 `MC_CKPT_DIR` env 지원 추가 — hf_hub_download 대신 로컬 파일 우선). 기대: `[smoke] ALL OK`. triton 커널이 VESSL GPU(아키텍처 확인)에서 컴파일되는지가 핵심 리스크
- [ ] **Step 4: 커밋** (vessl/ 스크립트 + load_mc.py 수정)

### Task 1: X1 덤프 (rev2: q_t 추가) — VESSL GPU

- rev1 T1과 동일하되: (i) npz에 `q` [16,H,K] fp16 추가 (answer position, `_project` q를 kernel 규약 L2-norm), (ii) 실행은 `run_remote.sh x1_dump.py --model ...`, (iii) 완료 후 덤프를 `/data2/sohyung/mc_niah/x1_dump/`로 rsync 회수, (iv) e1 대조 검증(±3pp, best-layer 일치)은 로컬 CPU에서
- 출력: `$MC_OUT/x1_dump/{model}/{dataset}/{idx}.npz` — 키 `u`, `q`, `csub_raw`, `c_full`, `stock_scores`, meta

### Task 2: X1 분석 (2a+2b+X1c) — 로컬 CPU

- rev1 T2의 2a(특징 Jaccard/R²/gram) 유지 + **2b 신규**: 같은 덤프에서 `u→q` 교체 재채점, 4조건 hit@2 {stock, u_eq_q, stock_centered, u_eq_q_centered} × 4 데이터셋 × 2 모델 × layer별. centering은 X1c 조건(gram≥0.8) 성립 시 4조건 전부, 아니면 stock/u_eq_q 2조건 + X1c 생략 사유 기록
- **`u=q`로 paired-D hit@2 유의 상승 시 즉시 coordinator 보고** (스펙: 후속 설계 출발점 변경)
- 출력: `results/x1_router_probe.json` (스펙 rev2 스키마), 그림 2패널 (R²·gram / 4조건 hit@2, e1 곡선 overlay)
- 테스트: rev1 T2 테스트 + u=q 재채점 로직 합성 검증 (신호를 q에만 심으면 u_eq_q hit 상승)

### Task 3: X2 multi-query/multivalue (신설 Tier 1) — 데이터 로컬, probe VESSL

- **데이터 (로컬 CPU)**: `data.py prepare-a` 확장 호출로 `niah_multiquery`, `niah_multivalue`, `niah_multikey_1` @ {2048, 4096, 8192} 각 50샘플 (multikey 2048은 기존 재사용, 4K/8K 신규 — X2 대조군 겸 스펙의 "4K→8K 붕괴 직접 설명"). needle 수 스윕: `niah.py` 직접 호출로 multiquery q=2 버전 추가 (가능하면)
- **annotate 확장** (`data.py`): multi-query("for X, Y, and Z" 리스트 파싱 → queried keys 복수, gold_segs 집합), multivalue(key 1개, value 4개 각각의 seg). 기존 단수 경로 회귀 없음 (기존 테스트 유지 + 신규 테스트)
- **probe (VESSL GPU)**: E1 기계 재사용, 신규 지표 — **질문된 key별 hit@2 macro 평균** (needle별 "그 needle의 seg ∈ top-2" 집계 후 평균), 보조 **hit@k, k=needle 수** (추론 시 topk만 상향 — routing 점수는 그대로, top-k 집합만 크게), layer별+best-layer, 5B/30B
- **chance 재도출** (multi-gold): needle m의 hit@2 chance = 2/n_past (needle별 동일) → macro chance = mean(2/n_past); hit@k chance = k/n_past. JSON에 식과 값 병기
- 출력: `results/x2_multiquery.json` + (태스크×길이×모델) 표. **판정 (M)을 coordinator에 보고하고 X3 진행 여부 지시 대기** (스펙 §3 판정표: multiquery≈single≫multikey → (M) 확정 / multiquery≈multikey → (M) 기각 → X1c·기하교정 승격, X3/X5 보류)

### Task 4: X3 anomaly probe (X2에서 (M) 확정 시에만) + X3b — VESSL GPU

- rev1 T3(구 X2)와 동일 설계: X2.jsonl 32쌍 (anom/noanom, D 준용, `_fit_filler` 길이 일치, UUID 이상 문장), hit@2(anomaly) vs 대조군, 판정 H_outlier/H_template
- **X3b (H_template 시)**: query-swap 쌍 + top-2 Jaccard/hit@2, **stock과 u=q 두 조건 모두** (스펙 rev2: u=q에서 Jaccard가 낮아지면 처방① 직접 증거)
- 출력: `results/x3_offtemplate_probe.json`, `results/x3b_query_swap.json`

### Task 5: X4 random-routing ablation — VESSL GPU (헤비)

- rev1 T4 설계 그대로 (incremental generation 엔진 + 동치성 게이트 + selection override stock/random×5seed/recent + 6길이 그리드 50샘플 n_gen=128), 단:
  - 실행은 VESSL: (model×task) 4개 청크를 `run_remote.sh`로 순차/병렬(GPU 수에 따라), **셀 단위 skip-existing 머지 필수** (컨테이너 재시작 내성) + 로컬 watchdog(선택: dynmc ensure 패턴 — 3분마다 ssh로 진행 마커 확인·재기동)
  - 데이터: 6길이 prepare-a 로컬 생성 → 업로드 (X2와 공유: multikey 4K/8K 재사용)
  - 동치성 게이트: incremental(stock) == full re-forward greedy 토큰열, 2048 8샘플, VESSL에서 실행
- 출력: `results/x4_random_routing.json` + `results/x4_table.md` (stock/random/recent/vanilla-앵커 4행 × 6열 + chance 상한 `2/(N-1)×0.6` + 배율 열)

### Task 6: X5 P-sweep (X3=H_template 시에만) — 로컬 CPU

- rev1 T5 그대로 (T1 덤프의 csub_raw 재평균 → P∈{1,2,4,8,16,32}, 예측 곡선 √P/√(2lnP) overlay, kill-check P=32 D-hit@2 0.6) + **X1c에서 centering 효과 확인 시 centering×P 2차원**
- 출력: `results/x5_psweep.json` + 그림

### Task 7: report/0025.md + 0025_ko.md + 마무리

- 스펙 §7 판정표 **7행** (H_blind 국소 / (M) / H_outlier / H_template / u=q 개선 / anisotropy / MC 이득=routing), X4 표, 다음 설계 권고 1페이지(§0.2 분기표를 실측으로 채움 — §0.3 철회 처방 언급 금지), 한계, 재현(VESSL 커맨드). 리뷰 subagent 수치 전수 대조. README 인덱스, 프로젝트 메모리 갱신

## Self-Review
- 스펙 rev2 커버리지: §2→T1+T2, §3→T3, §4→T4, §5→T5, §6→T6, §7→T7, §8→Global Constraints. 실행 순서 T0→T1→T2→T3→(분기)→T4→T5→(T6)→T7; X4(T5)는 독립이므로 분기 대기 없이 T3 후 착수 가능
- VESSL 제약 반영: idempotent bootstrap, geesefs 스크래치 금지, 셀 단위 재개, 코드는 git 경유
- 타입 일관성: 덤프 키에 q 추가가 T2/T6 소비처에 명시; x1 출력 파일명 rev2 스키마(`x1_router_probe.json`)로 통일
