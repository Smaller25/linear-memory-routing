# Handoff: linear-memory-routing (0024 → 0025 → 0026) — 2026-07-28

작성 시점 기준 인수인계 문서. **당신(Codex)이 이 작업들을 이어받습니다.** Claude를 다시 호출하지 않아도
되도록, 배경·현재 진행 중인 작업·정확한 재개 커맨드·환경 함정을 모두 담았습니다.

---

## 0. 한눈에 보기

| 실험 | 상태 | 산출물 |
|---|---|---|
| **0024** MC-SSC multi-NIAH 실패 원인 | **완료** | `report/0024.md`, `report/0024_kr.md` |
| **0025** router 진단 (X1/X2/X4) | **X1·X2 완료, X4 마무리 중** | `report/0025_ko.md` (X4 절은 placeholder), `results/x1_router_probe.json`, `results/x2_*.json` |
| **0026** SRLA 구현 | **코드 완료·리뷰 2라운드 통과, GPU 미실행** | 브랜치 `sh/srla` @ `33a387fa` |
| 부가 | 30B lm-eval 비교 완료 | `/data2/sohyung/mc_niah/side_eval/SUMMARY_30B.md` |

**주 브랜치**: `sh/mc-niah-analysis` (0024/0025 + 스펙/플랜), origin에 push됨.
**0026 브랜치**: `sh/srla`, worktree `/home/sohyung/linear-memory-routing/.claude/worktrees/agent-a6df3d8c035245a74`, **push 안 됨**.

---

## 1. 지금 이 순간 돌고 있는 작업 (상세)

### 1-A. 0025 X4 그리드 — VESSL A100 (거의 끝)

**무엇**: random-routing ablation. "MC의 장문 이득이 routing 품질에서 오는가, 아니면 segment당
write 부하가 256으로 상한되는 효과인가"를 가른다. 4 mode 계열(stock / recent / random×5 seed) ×
2 모델(mc-5B, mc-30B) × 2 태스크(niah_single_1, niah_multikey_1) × 6 길이(1K~32K) = **168 셀**,
셀당 50샘플 greedy 생성(n_gen=128).

**현재**: **163/168 셀 완료**, 3개 프로세스 잔여, finalizer 워치독 살아 있음.

**구조 (중요)**:
- 셀 단위로 `/root/smaller/mc_niah/results/x4_raw.json`에 즉시 저장 → **skip-existing 재개 가능**
- GPU util이 단일 job에서 20%밖에 안 나와(순차 greedy decode = latency-bound) `(model,task,length)`
  분할로 **6개 병렬** 실행 중. raw JSON 쓰기는 원자적(`os.replace`)
- 병렬 슬라이스는 aggregate의 `(model,task)` 항목을 자기 길이만으로 덮어쓰므로, **finalizer
  워치독**(`/root/x4_finalize_watch.sh`, 로그 `logs/x4_finalize.log`)이 전부 끝난 뒤 4개 조합을
  전체 길이로 재실행(전부 skip, aggregate만 재생성)하고 `x4_table.py`를 돌린다 → 로그에
  `[fin] FINALIZE DONE`이 찍히면 완료

**당신이 할 일**:
1. `ssh -o BatchMode=yes -i /home/sohyung/sohyung2.pem -p 32273 root@betelgeuse.cloud.vessl.ai`로
   `grep -c FINALIZE /root/smaller/mc_niah/logs/x4_finalize.log` 확인 (안 뜨면 대기)
2. 완료 후 결과 회수:
   `rsync -a -e "ssh -i /home/sohyung/sohyung2.pem -p 32273" root@betelgeuse.cloud.vessl.ai:/root/smaller/mc_niah/results/x4_{random_routing.json,table.md} /data2/sohyung/mc_niah/results/`
   그리고 repo `lmr/analysis/260725_mc_niah_analysis/results/`에도 복사 후 커밋
3. **`report/0025_ko.md`의 "X4 — random-routing MC ablation (실행 중)" 절을 실측으로 교체**하고,
   판정표 7행 중 "MC 이득 = routing" 행을 채운다. 판정 규칙: `stock ≫ random`이면 routing이
   실제로 작동, `stock ≈ random ≫ vanilla`면 이득은 routing이 아니라 segment당 write 상한 효과.
   `chance 상한 = 2/(N-1) × 0.6` (N=ctx/256) 병기. **수치는 JSON에서 읽어 기입, 하드코딩 금지**
4. 그 다음 `report/0025.md`(영문 원본)를 작성 — 지금은 `0025_ko.md`만 있고 영문은 "X4 완료 후
   작성 예정"으로 명시돼 있다

**주의**: VESSL 컨테이너는 재시작되면 `/root/.ssh`가 초기화된다. ssh가 `Permission denied
(publickey)`면 **사용자에게 웹 터미널에서 아래를 실행하도록 요청**해야 한다(당신이 할 수 없음):
```
mkdir -p /root/.ssh && cat /root/smaller/ssh_authorized_keys_sohyung2 >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
```
그 뒤 `bash /root/smaller/mc_niah/t0_local.sh`로 환경 복원(idempotent, 수 분).

### 1-B. 0026 backward gate — 로컬 SLURM (큐 대기)

**무엇**: SRLA 학습 전 **차단 게이트**. 두 가지를 확인한다:
- **gate A**: routing loss(gold chunk CE)로 `W_q`, `W_desc`에 유한·비영 gradient가 오는가 (state를
  detach하므로 커널 backward 불필요)
- **gate B**: LM loss가 `chunk_gdn2` **backward**를 통과하는가 — 로컬은 sm_120 Blackwell이고
  0024/0025에서 검증된 건 forward/eval뿐이다. MC 모델 자체는 H200에서 학습됨
- **scale diagnostics**: `per_head_ratio_read_over_online` — `--fusion-scale`을 정하는 숫자

**현재**: **job 2436 `PD (Resources)`** — 다른 사용자(junho)의 6h job 2개가 gpu01의 GPU 2장을
모두 점유 중. 자리가 나면 자동 시작된다.

**당신이 할 일**:
1. `squeue -u sohyung`으로 2436 상태 확인
2. 완료 후 **`/data2/sohyung/srla/results/srla_backward_gate.json`을 읽고 `fla_pin` 블록을 먼저 확인**
   (노드에서 pin이 검증되는 유일한 지점), 그 다음 `verdict.blocking_issues == []` 확인
3. **gate B 결과에 따라 분기**:
   - **통과** → 아래 §2의 primary 학습 arm 실행 가능 (단, 사용자 승인 후)
   - **실패** → `--no-lm-loss --detach-bank`로 축소(그러면 `W_q`+`W_desc`만 학습, `Φ_align`은
     학습 불가). **"LM-loss 단계는 A100/H200 필요"를 명시적으로 보고**하고 조용히 누락하지 말 것.
     그리고 hybrid descriptor arm은 이 경우 `W_q`만 학습되어 **아예 학습 불가**임을 기록
4. `scale_diagnostics.per_head_ratio_read_over_online`이 O(1)이 아니면 `--fusion-scale`을 조정하거나
   `--align-prenorm rms`를 쓴다 (후자는 identity-at-init을 포기하므로 `top_k=0` parity 재확인 필요)

**재제출 커맨드** (worktree에서):
```
cd /home/sohyung/linear-memory-routing/.claude/worktrees/agent-a6df3d8c035245a74
sbatch sbatch/srla_gate.sbatch
```

### 1-C. CPU bilinear probe — **완료됨 (결과 미해석)**

**무엇**: GPU를 쓰기 전 CPU만으로 0026의 핵심 베팅을 판별하는 사전 검사.

배경: SRLA의 점수식은 랭킹 관점에서 `q_tᵀ M f_m` (`M = W_qᵀW_desc`, rank ≤ R)이다. 0025는 이
계열의 **단 한 점 `M = I`만** 시험해 기각했다(paired-D 0.563→0.500 / 0.500→0.500). "학습된 M이
존재하는가"는 미검증. 0025 X1 덤프에 필요한 재료가 다 있으므로(answer-position `q_t`, descriptor
bank `c_full`, gold index; 264 npz) CPU에서 rank-R bilinear를 gold CE로 적합해 **held-out**에서 잰다.

**현재**: 산출물이 이미 생성돼 있다 —
`/data2/sohyung/mc_niah/results/probe_bilinear_router.{json,png}` (키: `meta`, `results`, `summary`).
repo 쪽 results 디렉토리와 커밋 여부는 확인 필요.

**당신이 할 일**:
1. `probe_bilinear_router.json`의 `summary`를 읽고 **결론을 판정**: paired_D_multi와
   niah_multikey_1에서 **held-out test hit@2**가 (a) `M=I` 기준선, (b) stock scorer, (c) chance를
   넘는가. 셀당 n이 16~50이므로 **SE/CI를 반드시 함께 보고**하고, test 샘플이 8개 미만인 셀은
   "inconclusive"로 처리 (점 추정치를 결론처럼 쓰지 말 것)
2. 판정에 따라 사용자에게 권고:
   - **넘지 못함** → 0026의 `average`-descriptor arm은 실패 예측. GPU 예산을 gate B → `Φ_align`/
     `h_state`(hybrid) 쪽으로 돌리는 게 낫다
   - **넘음** → primary 학습 arm 실행이 정당화됨
3. 결과물이 repo에 커밋 안 돼 있으면 커밋 (`lmr/analysis/260725_mc_niah_analysis/results/`)

---

## 2. 0026 SRLA — 코드 상태와 실행 계획

**브랜치**: `sh/srla` @ `33a387fa` (worktree
`/home/sohyung/linear-memory-routing/.claude/worktrees/agent-a6df3d8c035245a74`), **push 안 됨**.
22개 신규 파일, 기존 파일 수정 0. 테스트 **215 passed, 1 skipped**.

**테스트 커맨드** (repo 루트에서 bare `pytest -q`를 쓰면 vendored fla suite까지 수집돼 1500+ 실패가
난다 — 반드시 파일 지정):
```
cd <worktree>
/data2/sohyung/conda-envs/sh_infocap/bin/python -m pytest \
  tests/test_srla_modules.py tests/test_srla_fusion.py tests/test_srla_train.py \
  tests/test_srla_eval.py tests/test_srla_gate.py tests/test_srla_pin.py -q
```

**구성** (스펙 `docs/superpowers/specs/2026-07-28-0026-retrieval-improvement.md` rev2 기준):
`src/modules/{descriptor,router,cache}.py`, `src/models/fusion_wrapper.py`,
`src/models/{toy_backbone,backbone_loader}.py`, `train_router.py`, `eval_srla.py`, `bench_ttft.py`,
`srla_backward_gate.py`, `sbatch/*`, `docs/0026_design_notes.md`.

**리뷰에서 확정된 실행 순서와 primary 설정** (사용자 승인 후):
```
# 1) BLOCKING: gate (1-B 참조)
sbatch sbatch/srla_gate.sbatch

# 2) PRIMARY 학습 arm — 0025가 기각한 지점에서 정확히 출발
RUN=/data2/sohyung/srla/runs/run01 STEPS=4000 INIT=identity sbatch sbatch/srla_train.sbatch
#  == --backbone mc-30B --chunk-size 256 --seq-len 2048 --top-k 2
#     --router-init identity --route-dim 0 --l2-normalize-keys
#     --descriptor average --align key_linear --align-prenorm none
#     --fusion-scale <gate의 ratio_read_over_online에서 결정>
#     --router-loss ce --lambda-csr 1.0 --steps 4000 --resume auto
#  gate B 실패 시: EXTRA="--no-lm-loss --detach-bank"

# 3) SECONDARY (primary에서 신호가 보일 때만, init 비용 측정용)
RUN=/data2/sohyung/srla/runs/run02 INIT=xavier sbatch sbatch/srla_train.sbatch

# 4) 평가 — 4개 arm: frozen / untrained / untrained-identity / trained
CKPT=/data2/sohyung/srla/runs/run01/ckpt.pt sbatch sbatch/srla_eval.sbatch
sbatch --exclusive sbatch/srla_ttft.sbatch
```

**해석 규칙 (반드시 지킬 것)**:
- **핵심 비교는 `trained` vs `untrained-identity`** — `frozen`이나 `untrained`(랜덤 router)와의
  비교가 아니다. 이 쌍만이 "M을 학습한 것이 도움이 됐는가"를 분리한다
- `--n-gen 32`는 샘플당 32번 prefill을 유발한다(미해결 비효율). 5길이 × 4arm 스윕 전에 **8 정도로
  낮출 것**
- 학습과 평가에 **동일한** `--chunk-size --top-k --tau --descriptor --l2-normalize-keys
  --fusion-scale --persist-fusion`을 넘겨야 한다 (checkpoint에서 자동 채택하되 명시적 모순은 치명적
  에러로 처리됨)

---

## 3. 사용자 결정 대기 중인 항목 (당신이 임의로 정하지 말 것)

1. **`--l2-normalize-keys`를 기본값으로 둘지** — 스펙 §2B 문구는 raw key 평균(`k̄_m = mean k_t`)인데
   구현 기본값은 정규화된 키 평균. 근거: `chunk_gdn2`가 커널 내부에서 k를 L2 정규화하므로 **실제로
   `S_m`에 기록되는 키는 정규화된 키**이며, raw key descriptor는 state가 본 적 없는 벡터를 요약한다.
   또 MC-SSC(0024/0025 대상 모델)도 정규화 키 mean-pool을 썼으므로 `untrained-identity` 대조군이
   0025를 재현하려면 켜져 있어야 한다. `--no-l2-normalize-keys`로 문구 그대로도 가능.
   → **사용자 확인 필요** (Claude가 이미 물었고 답변 대기 중)
2. **학습 실행 승인** — gate 결과를 보고한 뒤 승인받고 시작
3. **hybrid descriptor arm의 출발점** — identity init이 구조적으로 불가(`3HK ≠ HK`)해서 null 결과가
   교란된다. 온전한 대안 둘: (a) 선행 `"mean"` 컴포넌트를 신설해 `[I|0|0|0]`로 초기화,
   (b) `--components state` 단독(feature_dim = H·K로 identity 도달 가능). **둘 다 미구현이며,
   gate B가 hybrid arm의 학습 가능성 자체를 결정하므로 그 전에 만들지 말 것**

---

## 4. 환경 함정 (전부 실제로 겪은 것들)

### fla 버전 pin — 가장 위험한 함정
- 이 저장소에는 **vendored `fla/` 0.5.2**가 있고, conda site-packages에는 **0.5.1로 라벨링됐지만
  실제 파일은 0.5.2와 md5 동일한** 사본이 있다. 즉 **버전 문자열 검사만으로는 속는다**
- pinned 커널(`long-gdn` @ `e71713e`)은 `chunk_gla_fwd_o_gk(..., use_exp2=..., transpose_state_layout=...)`를
  호출하므로, 0.5.2가 잡히면 **TypeError로 확실히 죽는다**
- 정답: **fla `4b02d15d`** (설치본 `/data2/sohyung/mc_niah/pydeps`). 검증은 버전 + `use_exp2`/
  `transpose_state_layout` 마커 존재 + `fla.__file__`이 pydeps 아래인지 **세 가지 모두**
- `import fla`가 이미 일어난 뒤에는 `sys.path` 재정렬로 되돌릴 수 없다 (0026 코드가 이 경우 큰 소리로 실패)

### 로컬(greenbeard)
- GPU 작업은 **반드시 `sbatch -p main --gres=gpu:rtx6000:1`**, 6h 캡. 로그인 노드에서 직접 CUDA 금지
- `env_common.sh` (0025) / `sbatch/srla_env.sh` (0026)를 source
- python: `/data2/sohyung/conda-envs/sh_infocap/bin/python`
- **root 디스크가 꽉 찼다** → 대용량은 전부 `/data2/sohyung/`
- 다른 사용자와 공유한다 (지금 junho의 job이 GPU 2장 점유 중)
- `node`는 기본 PATH에 없다 → `/home/sohyung/.nvm/versions/node/v24.18.0/bin`

### VESSL
- 접속: `ssh -i /home/sohyung/sohyung2.pem -p 32273 root@betelgeuse.cloud.vessl.ai`
- **컨테이너 재시작 시 `/root/.ssh`·pip 전부 소실** (`/root/smaller`만 영구)
- **geesefs(`/root/smaller`)에서 git clone/checkout 금지** — 소파일 대량 쓰기로 실패·디렉토리 증발.
  코드는 컨테이너 로컬(`/root/work`), geesefs에는 **bundle/tar/ckpt/결과만**
- private + git-lfs repo는 `vendor/long-gdn-e71713e.bundle`에서 클론 + `GIT_LFS_SKIP_SMUDGE=1`
- HF_HOME/TMPDIR/TRITON_CACHE는 반드시 컨테이너 로컬(`/tmp`)
- pip 리스트(검증본): `einops transformers huggingface_hub numpy matplotlib lightning sentencepiece
  pytest tenacity nltk wonderwords pyyaml html2text` + `python -m nltk.downloader punkt punkt_tab`
- 복원 스크립트: `bash /root/smaller/mc_niah/t0_local.sh` (idempotent)

### 데이터
- 로컬: `/data2/sohyung/mc_niah/data/{2048,...}/`, `data/paired/{S,D}.jsonl`, X1 덤프
  `/data2/sohyung/mc_niah/x1_dump/` (264 npz, 3.8GB)
- VESSL: `/root/smaller/mc_niah/data/` (1K~32K 전부 생성됨)
- **전처리는 로컬 96코어에서, GPU 서버 켜기 전에** (GPU 시간 낭비 방지 — 사용자 지시)

---

## 5. 배경: 0024/0025가 확정한 것 (0026의 근거)

- **0024**: MC-SSC(mean-pool descriptor) GDN2-370M의 multi-key NIAH 실패는 **routing 병목**.
  oracle로 gold chunk를 top-k에 강제 주입하면 4/4 셀 개선 (mc-30B/D: 0.062→0.562). write는 온전
  (b1_after 0.67~0.97), read는 같은 segment 키 충돌(S 조건)에서만 drift
- **0025**: router가 **query와 거의 무관**하게 점수를 낸다 (query 없는 descriptor 특징만으로
  R²≥0.9, layer 2~13). descriptor Gram off-diagonal **0.96~1.0**(극단적 anisotropy).
  값싼 처방 전부 무효: `u:=q` (paired-D flat, paired-S −18.75pp), centering(대수적 no-op 증명),
  PC1 제거(무효~악화). multiquery/multivalue 검정으로 **상위 주장 (M) "router는 needle 검출기다"
  기각** — needle이 여러 개면 "찾기" 자체가 무너진다
- **함의**: 남은 처방은 (a) **학습된 contrastive router supervision**, (b) **backbone 읽기 기하에
  맞춘 descriptor**(GDN2는 delta rule이라 state는 `(I − β k kᵀ)` 곱으로 쓰이므로 `v_{t*}`를 꺼내는
  방향은 raw `k_{t*}`가 아니다). 0026의 `W_desc`/`W_q` 학습과 `Φ_align`이 정확히 이 두 자리

---

## 6. 작업 규약

- 커밋 트레일러: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (당신이 커밋할 때는 당신
  자신을 명시해도 됨 — 다만 기존 커밋 스타일과 일관되게)
- `.sh`/`.sbatch`는 repo `.gitignore`가 `*.sh`를 무시하므로 **`git add -f`**
- 수치는 **JSON에 먼저 쓰고 보고서는 JSON만 참조** (하드코딩 금지)
- 작은 n(16~50)에서는 절대값 주장 금지, 방향성으로 서술. bf16 재실행 노이즈 ~2.3pp 이내 차이는
  결론에 쓰지 않음
- 데이터셋마다 chance level을 따로 계산해 병기 (0025: single 0.288, multikey 0.486, paired 0.286)
- 진행 기록: `.superpowers/sdd/progress.md` (git-ignored 로컬 스크래치)
- **`sh/srla`는 push 안 된 상태** — push 전 사용자 확인
