# 실험 기록

이 저장소(linear-memory-routing)에서 돌린 실험을 한곳에 모은 기록이다. 새 실험은 끝나는 대로 같은 형식으로 아래에 덧붙인다.

- 형식: 실험마다 **목적 / 세팅 / 결과 (raw) / 결과 (해석) / 경로**.
- 숫자는 로그, `train.jsonl`, wandb, 보고서에서 그대로 옮겼다. 출처에 해석이 없을 때는 정리할 때 붙인 해석에 "(에이전트 해석)"을 달았다.
- G = 2026-06~08 greenbeard 실험(`logs/`, `legacy/experiments/report/`). E = 2026-09 이후 mslm 코드베이스 실험(대부분 KISTI).
- report 0001–0011, 0022/0023, 0025는 클라우드 A100/VESSL에서 돌아서 여기에 없다. 원문은 `legacy/experiments/report/`.
- 2026-11-05 정리.

## 전체 목록

| ID | 날짜 | 실험 | 서버 | 대표 결과 |
|---|---|---|---|---|
| G1 | 06-19 | 환경·커널·백본 검증 | greenbeard | GDN/KDA/GDN2 학습 OK, Mamba3 커널 없음 |
| G2 | 06-19 | MQAR 첫 시도(train-kv 16/32, lr 탐색) | greenbeard | 기본 lr 전부 실패, lr 3e-3에서 oracle kv128 0.998 |
| G3 | 06-20 | vanilla·MoCM 부트스트랩 실패 진단 | greenbeard | loss 8.32(무작위) 고정 |
| G4 | 06-20 | Phase 0: 쉬운 커리큘럼 경계 비교 | greenbeard | kv64: vanilla 0.283, oracle 0.999, fixed 0.340 |
| G5 | 06-20 | Phase 1: learned 경계 헤드 | greenbeard | kv4–64 학습 시 kv128 0.998 |
| G6 | 06-20~22 | true-state 캐시 검증 | greenbeard | oracle kv128 0.992, fixed kv64 0.882 |
| G7 | 06-24 | predict_ruler smoke | greenbeard | native greedy와 일치 |
| G8 | 06-25~26 | d256/4L MQAR + 경계 분포 | greenbeard | learned kv512 0.999, 경계 중앙값 2 |
| G9 | 06-26 | flip-flop do-no-harm | greenbeard | vanilla·mosc 모두 1.000 |
| G10 | 06-26 | RULER, frozen mamba2-370m + SSC | greenbeard | niah_single@2K 0.00 → 36.67 |
| G11 | 06-26 | 불규칙 MQAR 적응 경계 | greenbeard | kv512: fixed2 0.000, learned 0.997 |
| G12 | 06-27~28 | 비지도 경계 학습 3종 | greenbeard | STE 붕괴, warm-start 0.00 |
| G13 | 06-28 | Selective Copying | greenbeard | M128: vanilla 0.592 → 0.99대 |
| G14 | 06-28 | axis-1 정보밀도 경계 | greenbeard | surprisal 0.39–0.41 → quantile 0.01 |
| G15 | 06-28 | axis-2 캐시 상한 vs 계층 재압축 | greenbeard | 버리기 > 합치기 |
| G16 | 07-09 | salience-gated retention | greenbeard | oracle kv64 0.966, 학습 신호 6/6 실패 |
| G17 | 07-09 | descriptor 기전 probe | greenbeard | 무작위 근처, 결론 불가 |
| G18 | 07-09 | frozen 370M RULER multikey | greenbeard | multikey@2K 0.452 → 0.129 |
| G19 | 07-10 | query-conditioned routing probe | greenbeard | key 구간 중 정답 0.267(무작위 0.128) |
| G20 | 07-10 | chunk 크기 sweep | greenbeard | 0.200/0.133/0.133, 무효 처리 |
| G21 | 07-10 | qa_1 평가 | greenbeard | 0.531 → 0.413, 무효 처리 |
| G22 | 07-17 | DynMC G0/G1/G1b | greenbeard | random vs fixed ppl −3.31% PASS |
| GA | 07-25~26 | 0024 MC-SSC 370M 실패 분해 | greenbeard | oracle 주입 0.062 → 0.562 |
| GB | 07-30~08 | normbias·ssketch·RULER·probe | greenbeard | RULER 평균 vanilla 27.2, meank4 36.2 |
| E0 | 07~09 | MC 라우팅 진단 요약(보류 문서) | greenbeard 등 | 370M 8K NIAH learned 3.0, oracle 74.0 |
| E1 | 10-26 | mslm 새 코드베이스 smoke | greenbeard | 재개 일치, oracle hit 1.0 |
| E2 | 10-29 | span-copy 라벨 통계 | CPU | n=8 창당 30.7개 |
| E3 | 10-29 | KISTI bootstrap + smoke | KISTI | SMOKE_OK |
| E4 | 10-29~30 | 50M vanilla vs MC, seed 0 | KISTI | 같은 step에서 MC val loss +0.049, NODE_FAIL로 중단 |
| E5 | 11-02 | 라우팅 모니터 문항 | KISTI | 64문항, 무작위 hit 13.68% |
| E6 | 11-02~03 | Stage 1a: span-copy CE 가중치 | KISTI | 최고 w0.1 hit 21.3%(무작위 +7.6pp), val loss +0.044, 기준 미달 |
| E7 | 11-05~06 | Stage 1c: detach router | KISTI | 통과. maxsim hit 36.6%(무작위 +23pp), val loss mon과 같음(2.947) |
| E8 | 11-10~11 | Stage 2 round 1: router descriptor 6종 + seed 1 재현 | KISTI | subblock32 46.7%, subblock16 42.7% 통과(기준 38.2%). 나머지 탈락 |
| E10 | 11-11~12 | Q3 round 2: subblock16·32·64 × seed 3 + NIAH string_match | KISTI | subblock64 48.2% > 32 44.3% > 16 37.7% > 8 33.0%(ruler3, seed 3 평균). string_match는 oracle까지 0~4%라 50M에서 변별 불가 |
| E9 | 11-10~11 | Q2 gate 분리 기여 + seed 2 재현 | KISTI | softgate hit 평균 42.5%(seed 차 18pp)지만 val loss +0.05. sep_softmax는 LM 손실을 지킨다 |

---

## I. greenbeard 실험 (2026-06 ~ 08)

**범위와 공통 설정**

- **서버.** greenbeard 연구실 서버 = `aigpu0423` (SLURM 노드 `gpu01`), GPU = NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 95 GB). 모든 잡은 `-p main --gres=gpu:rtx6000:1`. 잡 진입점 `scripts/sh_slurm_run.sh`(현재 위치: `legacy/experiments/scripts/sh_slurm_run.sh`, job-name 기본 `sh_routing`, 6h 제한) → 로그 `logs/sh_<job-name>_<jobid>.{out,err}`.
- **주 출처.** `/home/sohyung/linear-memory-routing/logs/` 276개 파일(= 138 잡, jobid 1071–1965, 2026-06-19 ~ 07-17). 결과 숫자는 모두 이 로그에서 그대로 옮겼다.
- **해석 출처.** `legacy/experiments/report/0012–0021.md`, `report/README.md`, `report/RESEARCH_STATE.md`, `report/INTERIM_REPORT.md`, `notes/*.md`, `lmr/mosc/README.md`, wait/submit 스크립트 `scripts/_sh_*.sh`, 계획서 `/home/sohyung/COG-second-brain/04-projects/linear-attention-multistate/planning/261029_from-scratch-routing-ce-plan.md`, `261023_restart-review-and-ipu-plan.md`.
- **logs/ 바깥 greenbeard 실행**(GA, GB. 10월 mslm smoke는 E1): `/data2/sohyung/mc_niah/logs/`(0024 분석과 8월 ssketch 계열), `/data2/sohyung/mslm_smoke/`(10월 mslm smoke). 요약만 옮겼다.
- **greenbeard가 아닌 것**(제외): report 0001–0011(클라우드 A100/VESSL, frozen SSC 원조 결과), 0022/0023(VESSL A100), 0025 X1/X2/X4(GPU는 VESSL), `lmr/dynmc/configs/0024.json`(340M 15B, `/root/smaller/...` 경로 = 원격). `/data2/sohyung/chunking_run_{1437,1438,1512}.log`는 다른 프로젝트(`SSM_Rank_Analysis` 노트북)라 제외.
- **코드 위치 주의.** 로그가 부른 `lmr.mosc.*`, `lmr.analysis.*`, `lmr.scripts.*`, `lmr/dynmc/*`의 소스는 지금 `legacy/experiments/lmr/` 아래에 있다(루트 `lmr/`에는 `__pycache__`만 남음).
- 표기: `acc` = recall accuracy(답 위치 정확도). "loss 8.32"는 MQAR 무작위 수준(= 학습 실패). `(에이전트 해석)` 표시가 없는 해석은 보고서/노트 원문을 옮긴 것이다.


**공통 설정 (from-scratch MQAR 계열)**
- 백본: FLA `GatedDeltaNet2`(GDN-2), env `sh_routing`(torch 2.11+cu128, fla 0.5.2). 레시피 lr 3e-3, wd 0.1, betas (0.9,0.95), grad-clip 1.0 (report 0012).
- 모델: `gdn2` = vanilla GDN-2, `mosc` = GDN-2 + 세그먼트 캐시 + hard top-k read-out(Dynamic-MoSC). `--chunk-mode` ∈ fixed / oracle / surprisal / learned / unsup / unsup_ste / density.
- 과제: Zoology식 MQAR(`lmr/tasks/mqar.py`). `--ctx-filler N` = 불규칙 MQAR(`make_mqar_gapped`, (k,v) 사이 무작위 길이 빈칸). `--task selcopy` = Selective Copying.
- 시드: 별도 표기가 없으면 기본 시드 1개(seed 인자 미지정).


### G1. 환경·커널·백본 검증 (2026-06-19 ~ 06-20)

- **목적.** 새 서버(Blackwell sm_120)에서 from-scratch 백본이 학습되는지, frozen-track 테스트와 GDN-2 chunk 커널이 정상인지 확인.
- **세팅/결과(raw).**
  - 1071 (06-19 17:15) `pytest tests/lmr/test_segment_runner_gdn.py` → **3 passed** (single_block_matches_fla, mc_rm_reduces_to_vanilla_single_segment, multi_segment_runs_and_adds_memory).
  - 1072 (17:36) 실패: `~cuda-nvcc_activate.sh: line 41: NVCC_PREPEND_FLAGS: unbound variable` (출력 없음).
  - 1073 (17:42) `verify_fromscratch.py`(scratchpad): `[FAIL] Mamba3 RuntimeError: Mamba-3 SISO kernels are unavailable`; `[OK] GatedDeltaNet params=2.8M loss=6.9478 grad_norm=2.430`; `[OK] KDA params=2.4M loss=6.9803 grad_norm=2.676`.
  - 1074 (17:55) `scripts/sh_check_backbones.py` + frozen-track pytest: Mamba3 FAIL, GatedDeltaNet OK(loss 6.9385), KDA OK(6.9615), **GDN2(layer) OK params=0.5M loss=0.0599**; pytest **18 passed**.
  - 1091 (06-20 01:43) `scripts/sh_check_gdn2_kernel.py`: FORWARD/BACKWARD 전부 `max_abs=nan ... MISMATCH`인데 마지막 줄은 `VERDICT: chunk kernel matches naive (gradients OK)` — 검사 스크립트 자체 이상.
  - 1092 (01:52) `pytest tests/ops/test_gated_delta.py -k "test_chunk and not varlen ..."` → PASSED (o diff 8.4e-5, dk 3.9e-3 등).
- **해석.** report 0012: "fla's gated-delta op test passes on sm_120 (10/10, fwd+bwd)" → 초기 학습 실패는 커널 버그가 아니라 커리큘럼 문제.
- **경로.** `logs/sh_sh_routing_{1071..1074,1091,1092}.*`, `legacy/experiments/scripts/sh_check_backbones.py`, `sh_check_gdn2_kernel.py`, `logs_sh_env_setup.txt`, `logs_sh_mamba2_build.txt`(causal-conv1d·mamba-ssm 휠 빌드 실패 기록).

### G2. MQAR 첫 시도: train-kv 16/32, lr 탐색 (2026-06-19 17:58 ~ 06-20 01:32)

- **목적.** Dynamic-MoSC(세그먼트 캐시 + hard top-k)가 multi-key 회상을 푸는지 첫 비교. 0010(frozen SSC는 multi-key 실패)의 regime을 from-scratch로 공략.
- **세팅.** GDN-2 2층 d_model 128, batch 64, 3000 step, train-kv {16,32}, eval-kv 16–128. 1076–1078은 로그에 lr 표기 없음(기본 lr, report 0012 기준 3e-4로 추정); 1081 이후 `lr=0.003`. 시드 1개.
- **결과(raw).**

| job | 모델 | lr | 최종 loss | kv16 | kv32 | kv64 | kv128 |
|---|---|---|---|---|---|---|---|
| 1075 | smoke 3종(gdn2/oracle/surprisal pools=4, 10 step) | – | 9.2 | 0.000 | 0.000 | – | – |
| 1076 | gdn2 | 기본 | 8.3308 | 0.000 | 0.000 | 0.001 | – |
| 1077 | mosc oracle | 기본 | 8.3295 | 0.000 | 0.000 | 0.001 | – |
| 1078 | mosc fixed C=64 | 기본 | 8.3188 | 0.001 | 0.000 | 0.000 | 0.000 |
| 1079 / 1080 | surprisal / gdn2 | – | **취소**(`CANCELLED ... DUE to SIGNAL Terminated`) | | | | |
| 1081 | gdn2 | 3e-3 | 8.3205 | 0.000 | 0.000 | 0.001 | 0.000 |
| 1082 | mosc oracle | 3e-3 | 6.0347 | 0.049 | 0.015 | 0.004 | 0.002 |
| 1083 | mosc fixed C=64 | 3e-3 | 0.0010 | 1.000 | 0.998 | 0.915 | 0.411 |
| 1084 | mosc surprisal | 3e-3 | 0.0014 | 1.000 | 0.997 | 0.882 | 0.436 |
| 1085 | mosc oracle (재실행) | 3e-3 | 0.0002 | 1.000 | 0.999 | 1.000 | 0.998 |

- **해석.** report 0012: "high lr is required — at 3e-4 every model pins at random loss". (에이전트 해석) 같은 설정의 oracle이 1082는 6.03에서 멈추고 1085는 750 step에 상전이 → from-scratch MQAR 부트스트랩이 실행마다 갈린다(0020의 "knife-edge" 관찰과 같은 현상). vanilla는 kv16/32 직행에서 한 번도 학습되지 않았다.
- **경로.** `logs/sh_sh_routing_{1075..1085}.*`, 코드 `legacy/experiments/lmr/mosc/train_mqar.py`.

### G3. vanilla·MoCM 부트스트랩 실패 진단 (2026-06-20 01:31–01:40)

- **목적.** vanilla GDN-2가 왜 MQAR을 못 배우는지(용량? 헤드 수? 층 수? 스텝?) 및 기존 MoCM 학습기 확인.
- **세팅/결과(raw).** 모두 train-kv 16/32, eval 16–128.
  - 1086 gdn2 num-heads 8, 8000 step: params 3.06M, loss 8.3187, acc 전부 0.000.
  - 1087 gdn2 8000 step: loss 8.3188, 전부 0.000.
  - 1090 gdn2 n-layers 4, 6000 step: params 2.63M, loss 8.3197, 전부 0.000.
  - 1088 `lmr.scripts.train_mocm_mqar --num-memories 1` 4000 step: params 2,296,592, loss 8.3199, recall 0.001/0.000/0.000/0.000.
  - 1089 MoCM M=4: params 2,497,064, loss 8.3230, recall 0.001/0/0/0.
- **해석.** report 0012: "All-models-fail-to-learn at first was curriculum (training straight on kv 16/32 fails to bootstrap), not a kernel bug". `lmr/mosc/README.md`: "legacy train_mocm_mqar.py (MoCMMixer) does NOT learn under fla 0.5.2 (silent, likely API drift)".
- **경로.** `logs/sh_sh_routing_{1086..1090}.*`.

### G4. Phase 0 — 쉬운 커리큘럼에서 경계 방식 비교 (report 0012, 2026-06-20 01:54–02:01)

- **목적.** 세그먼트 수준 라우팅이 multi-key를 "애초에" 풀 수 있는가(recall vs #kv).
- **세팅.** d128, 2층, batch 64, 6000 step, lr 3e-3, train-kv {4,8}, eval 4–64(128). 시드 1개.
- **결과(raw).**

| job | 모델 | loss | kv4 | kv8 | kv16 | kv32 | kv64 | kv128 |
|---|---|---|---|---|---|---|---|---|
| 1093 | vanilla gdn2 | 0.0003 | 0.999 | 0.999 | 0.929 | 0.546 | 0.283 | – |
| 1096 | mosc oracle | 0.0000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.997 |
| 1097 | mosc fixed C=64 | 0.0004 | 1.000 | 1.000 | 0.967 | 0.640 | 0.340 | 0.173 |
| 1098 | mosc surprisal | 0.0003 | 1.000 | 1.000 | 0.905 | 0.503 | 0.248 | 0.124 |
| 1094 | MoCM M=4 | 8.3302 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | – |
| 1095 | MoCM M=1 | 8.3286 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | – |

- **해석.** 0012: "PASS — but the whole win is boundary placement." oracle(사실마다 세그먼트)는 kv128 0.997, fixed는 vanilla를 겨우 넘고 surprisal(NLL 피크)은 vanilla 수준 → 문제는 "oracle에 가까운 경계를 학습할 수 있나"로 좁혀짐.
- **경로.** `logs/sh_sh_routing_{1093..1098}.*`, `legacy/experiments/report/0012.md`.

### G5. Phase 1 — learned 경계 헤드 (report 0012, 2026-06-20 06:33–08:15)

- **목적.** 은닉 상태 위 `Linear(d,1)` 경계 헤드를 oracle 위치로 증류(`--boundary-distill 1.0`, pos-weighted BCE)하면 oracle 수준 회상을 재현하나.
- **세팅.** d128/2층, batch 64, lr 3e-3. 1100–1102: train-kv 4/8, 6000 step. 1103/1104: train-kv 4–64, 8000 step. 1099: 30 step smoke(증류 없음).
- **결과(raw).**

| job | 설정 | loss | kv4 | kv8 | kv16 | kv32 | kv64 | kv128 |
|---|---|---|---|---|---|---|---|---|
| 1099 | learned smoke 30 step | 10.66 | 0.001 | 0.001 | – | – | – | – |
| 1100 | learned, kv4/8 | 0.0467 | 0.999 | 1.000 | 0.501 | 0.250 | 0.126 | 0.063 |
| 1101 | learned, kv4/8 (재실행) | 0.0160 | 1.000 | 0.999 | 0.522 | 0.283 | 0.505 | 0.727 |
| 1102 | learned, kv4/8 (+thr sweep) | 0.0376 | 1.000 | 1.000 | 0.611 | 0.306 | 0.154 | 0.076 |
| 1103 | **learned, kv4–64, 8000** | 0.0142 | 0.995 | 0.999 | 0.998 | 0.998 | 0.999 | **0.998** |
| 1104 | oracle, kv4–64, 8000 | 0.0000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

경계 품질(pred/seq, precision, recall):
- 1101: kv4 4.6/0.87/1.00, kv8 8.2/0.98/1.00, kv16 8.2/1.00/0.52, kv32 10.3/0.96/0.31, kv64 36.5/0.97/0.56, kv128 100.6/0.99/0.78.
- 1102: kv16 10.0/1.00/0.62, kv32 9.8/1.00/0.31, kv64 9.8/1.00/0.15, kv128 9.9/1.00/0.08. threshold sweep thr 0.50→0.05: kv16 0.61→0.78, kv128 0.08→0.10.
- 1103: kv4 4.8/0.84/1.00, kv8 8.7/0.92, kv16 16.9/0.95, kv32 33.0/0.97, kv64 65.2/0.98, kv128 128.7/0.99 (recall 전부 1.00); thr sweep 전부 1.00.

- **해석.** 0012: 첫 시도 실패는 "curriculum overfit, not threshold" — kv4/8로만 학습하면 kv와 무관하게 ~10개만 발화하는 위치 편향을 학습. train-kv 4–64로 넓히자 경계 수가 kv를 따라가고 학습 밖 kv128도 일반화 → "Boundaries are learnable". (에이전트 해석) 1101은 같은 명령인데 kv64/128 recall이 오히려 0.505/0.727로 반등 — 1102와 결과가 다르고 보고서는 1102 숫자(0.61/0.31/0.15/0.08)를 인용한다. 코드 변경 사이의 중간 실행으로 보인다.
- **경로.** `logs/sh_sh_routing_{1099..1104}.*`.

### G6. true recurrent-state 캐시 검증 (report 0012, 2026-06-20 ~ 06-22)

- **목적.** read-out이 pooled-hidden 프록시(attention-lite)가 아니라 진짜 GDN-2 재귀 상태를 캐시해도 이기는가.
- **세팅/결과(raw).**
  - 1105 (06-20 08:28) `probe_truestate.py`: `max|full - segmented| = 6.375e-04`, state finite, std 0.022, seg-to-seg diff 9.660e-03.
  - 1136 (06-21) `--true-state` smoke 30 step: params 3.44M, acc 0.000/0.000/0.001.
  - 1150 (06-22 10:15) oracle `--true-state`, train-kv 4–64, 6000 step, loss 0.0022: kv4–128 = 1.000 / 1.000 / 1.000 / 0.999 / 0.998 / **0.992**.
  - 1151 (09:57) fixed C=64 `--true-state`, loss 0.0002: 1.000 / 1.000 / 0.999 / 0.988 / **0.882** / **0.468**.
- **해석.** 0012: "oracle-true-state ≈ 1.0 → the multi-key win is genuine recurrent-state recall"; "The true recurrent state is a strong lever on its own" (fixed 프록시 0.34→true-state 0.88 @kv64). true-state는 순차 실행이라 ~5–6배 느림(1150은 25분).
- **경로.** `logs/sh_sh_routing_{1105,1136,1150,1151}.*`.

### G7. predict_ruler smoke (2026-06-24 13:27)

- **목적.** frozen 모델 free-generation 루프에 read-out을 넣은 `predict_ruler`의 정합성 확인.
- **결과(raw, job 1194).** `[naive] one-seg+RM vs native greedy: MATCH` (8토큰 동일 `[28, 652, 876, 949, 741, 882, 148, 673]`), multi-seg(chunk=16) 실행 OK.
- **경로.** `logs/sh_sh_routing_1194.*`, 스크립트는 당시 scratchpad(`smoke_predict_ruler.py`, 저장소에 없음).

### G8. d256/4L 통합 MQAR 프로토콜 + 학습된 경계 분포 (report 0013, 0015, 2026-06-25 ~ 06-26)

- **목적.** 모델을 키워(d256, 4층, 4헤드, ~6M) kv 512까지 일반화 확인, 학습된 세그먼트 길이 분포 덤프, 정규 MQAR의 퇴화(fixed chunk=2 = oracle) 확인.
- **세팅.** 8000 step, batch 64, lr 3e-3, train-kv 4–128(vanilla 일부는 4–32), eval-kv 64/128/256/512, `--ctx-filler` 없음(정규 MQAR).
- **결과(raw).**

| job | 날짜 | 모델 (train-kv) | loss | kv64 | kv128 | kv256 | kv512 |
|---|---|---|---|---|---|---|---|
| 1206 | 06-25 | vanilla (4–128) | 8.3202 | 0.001 | 0.000 | 0.000 | 0.000 |
| 1233 | 06-26 | vanilla (4–32) | 0.0000 | 0.921 | 0.319 | 0.026 | 0.002 |
| 1207 | 06-25 | oracle (4–128) | 0.0000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1208 | 06-25 | learned (4–128) | 0.0170 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1242 | 06-26 | learned (4–128) + `--dump-boundaries` | 0.0297 | 1.000 | 1.000 | 1.000 | 0.999 |
| 1247 | 06-26 | fixed chunk=2 (4–128) | 0.0001 | 1.000 | 1.000 | 1.000 | 0.999 |

경계 품질: 1208 pred/seq 64.8/129.5/257.0/513.2, precision 0.99/0.99/1.00/1.00, recall 1.00; 1242 64.8/129.5/257.0/513.3 동일. thr sweep 전부 1.00.

0013 세그먼트 길이 분포(1242 덤프): kv64 seg/row 66.3, 중앙값 2, 평균 3.9, len-2 비율 98.5%, 질의 영역 세그먼트 128; kv128 130.2/2/3.9/99.2%/256; kv256 258.2/2/4.0/99.6%/512; kv512 514.1/2/4.0/99.8%/1024.

- **해석.** 0013: "The median learned segment is length 2 at every kv" = (key,value) 한 쌍, 개수가 #facts를 따라감(학습 밖 kv256/512 포함). "Vanilla trained on the full kv 4–128 mix does NOT converge". 0015: fixed chunk=2가 oracle과 같음 → "on regular MQAR, adaptivity is neither needed nor tested" → 불규칙 MQAR(§11)로 이동.
- **경로.** `logs/sh_sh_routing_{1206,1207,1208,1233,1242,1247}.*`, 덤프 `_workspace/seglen_learned.npz`, 그림 artifact `seg-length-dist`.

### G9. flip-flop 상태추적 do-no-harm (2026-06-26 11:19)

- **목적.** 세그먼트 분할이 기본 상태추적 능력을 깨지 않는지.
- **세팅.** `lmr.mosc.train_flipflop`, d256/4L/4헤드, train-len 128, eval n_instr 128/256/512, 4000 step.
- **결과(raw).** 1240 gdn2(1.85M): 1.000/1.000/1.000, loss 0.0000. 1241 mosc fixed C=64(1.95M): 1.000/1.000/1.000, loss 0.0000.
- **해석.** INTERIM_REPORT: "segmenting does not break native state-tracking."
- **경로.** `logs/sh_sh_routing_{1240,1241}.*`, `legacy/experiments/lmr/tasks/flipflop.py`.

### G10. RULER free-generation, frozen mamba2-370m + SSC (report 0014, 2026-06-26)

- **목적.** frozen-retrofit 트랙을 공식 RULER 프로토콜(greedy free-gen 128토큰 + 공식 `string_match_all`)로 검증.
- **세팅.** 백본 `state-spaces/mamba2-370m`(FLA 변환, bf16, 동결). SSC 라우터 k=4, low-rank 64, chunk 256, passkey 학습(500 step, aux 1e-4) → `ckpt/ssc_370m.pt`. 과제 niah_single_1, niah_multikey_2. 스크립트 `scripts/sh_ruler_pipeline.sh` → `sh_ruler_eval.sh` → `sh_ruler_compare.sh`(legacy/experiments/scripts).
- **결과(raw).**
  - 1239 (08:07) **OOM**: `[1/4] train SSC router` 단계, train_len 2048, `CUDA out of memory. Tried to allocate 2.00 GiB` (90.41 GiB 할당).
  - 1243 (11:24) train_len 1024로 줄였으나 Triton `get_cache_manager` 예외로 **크래시**(0014: 루트 디스크 ~95%, 캐시 정리로 해결).
  - 1244 (13:04) SSC 학습 성공(params 15,728,640, train_len 1024, 4 seg, step 499 loss 0.0524, train-acc 1.000). passkey 표: 512 vanilla/RM/SSC 1.000/1.000/1.000; 1024 1.000/1.000/1.000; 2048 1.000/0.833/0.986(Δ −0.014). RULER 데이터 준비 **실패** `ModuleNotFoundError: No module named 'nltk'` → 예측 단계 전부 skip.
  - 1256 (14:05) 데이터 준비 다시 실패(validation.jsonl not created) → skip.
  - 1257 (14:36→18:10) 데이터 준비 성공(4096/8192 × 2과제 × 50). vanilla 인라인 점수 4개 셀 전부 0.00(n=50) — 0014는 인라인 metric 버그라 명시, 공식 eval로 niah_single@4k 28.1(n=32 다른 샘플셋)이라 기록. +SSC 단계에서 **TIME LIMIT 취소**.
  - 1264 (21:07→22:15) matched 비교 n=30, 공식 metric:

| length | task | vanilla | +SSC |
|---|---|---|---|
| 2048 | niah_single_1 | 0.00 | **36.67** |
| 2048 | niah_multikey_2 | 0.00 | 3.33 |
| 4096 | niah_single_1 | 0.00 | (시간초과) |
| 4096 | niah_multikey_2 | 0.00 | (시간초과) |

  마지막에 `CANCELLED ... DUE TO TIME LIMIT`.
- **해석.** 0014: "+SSC gives a real net win on the standard free-gen protocol" (2048 single, zero-shot), multikey는 바닥(0010과 일치). 4k/8k +SSC는 디코드 루프에서 캐시 세그먼트를 매 토큰 재실행해 4h 제한 초과 — 배치 read-out 커널 필요. RESEARCH_STATE는 이 단일 needle 이득이 저정보 filler로 부풀려졌을 수 있다고 단서를 단다.
- **경로.** `logs/sh_sh_routing_{1239,1243,1244,1256,1257,1264}.*`, `ckpt/ssc_370m.pt`, `data/ruler/{2048,4096,8192}/*/pred.jsonl`, `data/ruler/summary.json`. 미실행 스크립트: `scripts/sh_ruler_370m.sh`(job-name sh_ruler, 해당 로그 없음).

### G11. 불규칙 MQAR: 적응 경계가 필요하고 학습된다 (report 0015, 2026-06-26)

- **목적.** 정규 MQAR의 퇴화를 없애 "사실 위치를 찾았나 vs 주기 2를 배웠나"를 가린다.
- **세팅.** `--ctx-filler`(64 또는 128)로 (k,v) 사이 무작위 빈칸, d256/4L, lr 3e-3. 최종(1260–1263): 8000 step, batch 64, train-kv 4–128(vanilla 4–32), eval 64–512.
- **결과(raw).**
  - 1248 learned smoke(filler 64, 60 step): acc 0, pred/seq 47.2/60.9, precision 0.10/0.14.
  - **1차 시도 1249–1252 (전부 0)**: 1249 vanilla loss 8.3239; 1250 fixed2 8.3177; 1251 oracle 8.3178; 1252 learned loss 8.7355(경계는 precision 0.73–0.99/recall 0.99–1.00인데 acc 0.000).
  - 디버그(filler 64, 3000 step, train 4/8/16, eval 8/16/32): 1254 vanilla 0.998/0.991/0.801(loss 0.0007); 1255 oracle **0.002/0.001/0.000(loss 7.5916)**; 1258 oracle 빈칸 없음 0.991/0.985/0.625; 1259 oracle filler 64(수정 후) 0.998/0.994/0.878.
  - **최종(filler 128):**

| job | 모델 | loss | kv64 | kv128 | kv256 | kv512 |
|---|---|---|---|---|---|---|
| 1260 | vanilla (4–32) | 0.0000 | 0.821 | 0.299 | 0.062 | 0.012 |
| 1261 | fixed chunk=2 | 7.6124 | 0.001 | 0.001 | 0.001 | 0.000 |
| 1262 | oracle | 0.0002 | 1.000 | 0.999 | 0.998 | 0.969 |
| 1263 | learned (+dump) | 0.0072 | 1.000 | 0.999 | 0.999 | 0.997 |

  1263 경계: pred/seq 64.3/128.3/256.3/512.2, precision = recall = 1.00 전 kv; thr sweep 전부 1.00.
  0015 길이 분포: 불규칙 kv64 중앙값 4(2–11), kv128 3(2–9), kv256 2(2–7), kv512 2(2–6).
- **해석.** 0015: "A fixed stride collapses on irregular facts (0.00) while the learned head matches the oracle (~1.0)". 고친 버그 2개: (1) "Context filler must be blank, not random"(무작위 filler가 가짜 key/value 주입 → 모두 0.000; 1249–1252가 이 버그), (2) "Segment summary = boundary-token hidden, not mean-pool"(oracle 0.000 → ~1.0; 1255→1259). 단 0020은 1249(loss 8.32)와 1260(loss 0.000)을 "같은 설정의 GPU 비결정적 부트스트랩"으로 다시 인용한다 — (에이전트 해석) 두 잡 사이에 filler 생성 코드가 바뀌었으므로 0015의 설명(버그 수정)이 더 직접적이다. 여전히 oracle 증류 지도 필요.
- **경로.** `logs/sh_sh_routing_{1248..1263}.*`, `_workspace/seglen_irregular.npz`, artifact `adaptive-boundaries`.

### G12. 비지도 경계 학습 3종 (report 0016, 2026-06-27 ~ 06-28)

- **목적.** oracle 없이 경계를 배울 수 있나.
- **세팅.** 불규칙 MQAR filler 128, d256/4L, 8000 step, batch 48, train-kv 4–64, eval 64/128/256. 시드 1개.
- **결과(raw).**

| job | 방식 / 손잡이 | loss | kv64 | kv128 | kv256 | pred/seq(kv64/128/256) | top-k(p) |
|---|---|---|---|---|---|---|---|
| 1268 | soft smoke (b 0.02, 40 step, filler 64) | 9.26 | 0.000 | 0.000 | – | 31.3/56.1 (kv8/16) | – |
| 1269 | **soft** `unsup` budget 0.02 | 0.0000 | 0.999 | 0.999 | 0.987 | 0.0/0.0/0.0 | – |
| 1270 | soft budget 0.05 | 0.0000 | 0.999 | 0.993 | 0.934 | 0.0/0.0/0.0 | – |
| 1272 | STE smoke (b 0.01) | 9.26 | 0.000 | 0.000 | – | – | – |
| 1274 | **STE** `unsup_ste` budget 0.005 | 8.3253 | 0.000 | 0.000 | 0.000 | 0.0/0.1/0.3 | – |
| 1275 | STE budget 0.02 | 8.3248 | 0.001 | 0.000 | 0.000 | 0.0/0.0/0.1 | – |
| 1276 | STE budget 0.05 | 8.3237 | 0.001 | 0.000 | 0.000 | 0.0/0.0/0.0 | – |
| 1277 | warm-start smoke (300 step) | 8.79 | 0.000 | 0.000 | – | – | – |
| 1278 | **warm-start** warmup 3000, budget 0.0 | 0.0000 | 0.980 | 0.630 | 0.197 | 0.0/0.0/0.1 | – |
| 1279 | warm-start budget 0.005 | 0.0000 | 0.984 | 0.676 | 0.209 | 0.0/0.0/0.0 | – |
| 1280 | warm-start budget 0.0 (top-k 지표 추가) | 0.0000 | 0.980 | 0.630 | 0.197 | 0.0/0.0/0.1 | **0.00/0.00/0.01** |
| 1281 | 지도 learned (기준) | 0.0013 | 0.999 | 0.997 | 0.995 | 64.3/129.0/256.7 | **1.00/1.00/1.00** |

- **해석.** 0016: soft = "recall is real but via full attention over all tokens — boundaries never form"; STE+L1 = "L1 has no floor → p→0 → empty cache → no recall gradient"; warm-start = oracle 제거 후 "fully drifted"(top-k(p)∩facts 0.00 vs 지도 1.00). 결론: "The 'where to cut' boundary signal is learnable only with continuous oracle supervision." 또 full-attention이면 과제가 쉽게 풀리므로 캐싱 주장은 캐시/읽기 예산을 고정해야 함.
- **경로.** `logs/sh_sh_routing_{1268..1281}.*`, 덤프 `_workspace/seglen_unsup_b02.npz`, `seglen_ste_b005.npz`, `seglen_warmstart_b0.npz`.

### G13. Selective Copying (report 0017, 2026-06-28 05:56–06:03)

- **목적.** 지도 방법을 외부 표준 벤치마크(Gu & Dao 2023 Selective Copying)에서 검증.
- **세팅.** `--task selcopy --vocab 256 --scatter-mult 8`(데이터 M개 + 노이즈 8M개), d256/4L/4헤드(~2M), train M 4–64, eval M 16/32/64/128, 8000 step, batch 48. 제출 순서 `scripts/_sh_selcopy_resubmit.sh`(QOS 5잡 제한), 결과 수집 `_sh_selcopy_wait.sh`.
- **결과(raw).**

| job | 모델 | params | loss | M16 | M32 | M64 | M128 |
|---|---|---|---|---|---|---|---|
| 1285 | vanilla | 1.98M | 0.0000 | 1.000 | 0.998 | 0.989 | **0.592** |
| 1286 | fixed chunk=2 | 2.08M | 0.0003 | 0.999 | 0.998 | 0.997 | 0.994 |
| 1287 | oracle | 2.08M | 0.0003 | 1.000 | 1.000 | 0.999 | 0.998 |
| 1288 | learned | 2.08M | 0.0002 | 1.000 | 0.999 | 0.999 | 0.996 |

1288 경계: pred/seq 16.0/32.0/64.0/128.0, precision = recall = top-k(p) = 1.00 전 M. 길이 분포(0017) @M128 중앙값 6, 평균 8.9, 최대 164.
- **해석.** 0017: (1) read-out이 vanilla 붕괴(0.592)를 ~1.0으로, learned 헤드가 데이터 위치를 완벽 복원. (2) 그러나 "with an unconstrained cache, Selective Copying does NOT require adaptivity" — fixed chunk=2도 0.994. 적응성 필요성은 캐시 예산 아래서만 드러남(→ axis-2).
- **경로.** `logs/sh_selcopy-{vanilla_1285,fixed_1286,oracle_1287,learned_1288}.*`, `ckpt/selcopy_learned_seglens.npz`, `legacy/experiments/report/figures/selcopy_seglen.html`, artifact https://claude.ai/code/artifact/0ec9952f-0a27-47a0-a63f-17a19fd48b76, `legacy/experiments/scripts/sh_selcopy.sh`.

### G14. axis-1 정보밀도 경계 (report 0018, 2026-06-28 06:03–06:25)

- **목적.** 분류기 학습 대신 백본 자신의 정보밀도 신호(surprisal / 예측 엔트로피 / 인접 상태 cos 거리)를 목표 발화율로 임계화(oracle 없음).
- **세팅.** `--chunk-mode density --target-rate 0.15 --budget 1.0`, 불규칙 MQAR filler 128, d256/4L, train-kv 4–64, eval 64/128/256, 8000 step, batch 48. 1289–1291 = fixed-0.5 임계(제출 `_sh_density_resubmit.sh`, selcopy 뒤로 게이트), 1292–1294 = quantile 발화(기본값 변경 후 재실행).
- **결과(raw).**

| job | 신호 / 발화 | loss | recall(kv64/128/256) | pred/seq | precision | top-k(p) |
|---|---|---|---|---|---|---|
| 1289 | surprisal, 0.5 | 8.3238 | 0.000/0.000/0.000 | 0.0/0.0/0.0 | 0.00 | **0.39/0.40/0.41** |
| 1290 | entropy, 0.5 | 8.3236 | 0.000/0.000/0.000 | 0.0 | 0.00 | 0.28/0.22/0.19 |
| 1291 | cosdist, 0.5 | 8.3245 | 0.000/0.000/0.000 | 1.0/1.2/1.5 | 0.02/0.08/0.11 | 0.09/0.12/0.14 |
| 1292 | surprisal, quantile | 5.9527 | 0.001/0.000/0.000 | 76.8/133.9/249.9 | 0.01/0.01/0.02 | 0.01/0.01/0.02 |
| 1293 | entropy, quantile | 3.7047 | 0.003/0.001/0.001 | 76.8/133.8/249.9 | 0.11/0.12/0.16 | 0.12/0.13/0.15 |
| 1294 | cosdist, quantile | 8.3258 | 0.000/0.000/0.000 | 77.0/134.0/249.9 | 0.07/0.09/0.12 | 0.06/0.08/0.12 |

(보고서 0018 표는 entropy·cosdist 0.5 행을 "0.2–0.3", "0.1"로 요약.)
- **해석.** 0018: "The intrinsic signal is real but partial" — 과제 학습된 백본의 surprisal이 사실을 0.40 순위(무작위 ~0.16의 ~2.5배)로 찾지만 0.5 임계에선 경계 0개 → 캐시 비어 recall 0. quantile로 캐시를 켜면 "the surprisal ranking collapses 0.40 → 0.01" — 신호를 하드 선택자로 쓰는 순간 공동학습이 상관을 지운다(자기파괴). 다음 후보: frozen/stop-grad 신호, 재구성 앵커.
- **경로.** `logs/sh_density-{surprisal,entropy,cosdist}_{1289..1294}.*`, `ckpt/density_surprisal_seglens.npz`, `scripts/_sh_density_{resubmit,wait}.sh`, `legacy/experiments/scripts/sh_density.sh`, 코드 `lmr/mosc/mosc_model.py::_density`.

### G15. axis-2 계층 재압축 vs 캐시 상한 (report 0019, 2026-06-28 06:28–06:52)

- **목적.** 캐시를 B 슬롯으로 제한할 때, 오래된 세그먼트를 버리는 것(capped)보다 학습된 인접쌍 merge로 압축(hier)하는 게 나은가 — 상수 메모리 가능성.
- **세팅.** `_sh_axis2_submit.sh`: COMMON = mosc, filler 128, d256/4L/4헤드, train-kv 4–64, eval 64/128/256, 8000 step, batch 48. 경계: oracle 또는 density(surprisal, rate 0.15). 정책: full / capped-32 / capped-64 / hier-32 / hier-64.
- **결과(raw).**

| job | 경계 | 정책 | params | loss | kv64 | kv128 | kv256 |
|---|---|---|---|---|---|---|---|
| 1295 | oracle | full | 6.14M | 0.0000 | 0.999 | 0.988 | 0.873 |
| 1296 | oracle | capped-32 | 6.14M | 0.0000 | 0.513 | 0.256 | 0.122 |
| 1297 | oracle | capped-64 | 6.14M | 0.0002 | 0.984 | 0.491 | 0.236 |
| 1298 | oracle | hier-32 | 6.28M | 0.0005 | 0.394 | 0.109 | 0.015 |
| 1299 | oracle | hier-64 | 6.28M | 0.0005 | 0.732 | 0.263 | 0.032 |
| 1300 | density | full | 6.14M | 5.9527 | 0.001 | 0.000 | 0.000 |
| 1301 | density | capped-32 | 6.14M | 8.3253 | 0.000 | 0.000 | 0.000 |
| 1302 | density | capped-64 | 6.14M | 0.0019 | 0.201 | 0.005 | 0.002 |
| 1303 | density | hier-32 | 6.28M | 8.3292 | 0.000 | 0.000 | 0.000 |
| 1304 | density | hier-64 | 6.28M | 0.0137 | 0.219 | 0.000 | 0.014 |

density 행 top-k(p): 1300 0.01/0.01/0.02, 1301 0.00, 1302 0.01/0.01/0.00, 1303 0.00, 1304 0.09/0.12/0.14. (1300은 1292와 출력이 동일 — full 정책 = 기본.)
- **해석.** 0019: capped는 0011의 ∝B/N 열화 재현; "Hierarchical compression is WORSE than dropping" — 두 사실을 한 슬롯에 섞으면 둘 다 복원 불가. density 경계에선 정책과 무관하게 ~0. 결론 "exact multi-key recall is fundamentally ~O(N) in the number of facts"; 압축은 손실 허용 과제에서만.
- **경로.** `logs/sh_axis2-{oracle,density}-{full,capped-32,capped-64,hier-32,hier-64}_{1295..1304}.*`, `scripts/_sh_axis2_{submit,wait}.sh`, 코드 `lmr/mosc/mosc_model.py::_apply_cache_budget`.

### G16. salience-gated retention (상수 메모리) (report 0020, 2026-07-09 04:14–07:09)

- **목적.** 캐시 없이(상수 메모리) filler의 쓰기를 약하게 해 고정 상태에 needle을 남길 수 있나. salience 신호: none / oracle / surprisal.
- **세팅.** vanilla `gdn2` + `--salience`(`GDN2LM.forward(salience_gate)`), d256/4L, train-kv 4–64, 8000 step. 세 묶음:
  - `salience-*` (1583–1585): filler 256, batch 48, eval 16/64/128/256.
  - `salience128-*` (1591–1593): filler 128, batch 48, eval 64/128/256/512.
  - `sal2-*` (1597–1605): filler 128, **batch 64**, 시드 0/1/2(oracle은 시드 0만), `_sh_sal2_run.sh`.
- **결과(raw).**

| job | 묶음 | salience | seed | loss | 첫째 kv | | | |
|---|---|---|---|---|---|---|---|---|
| 1583 | f256 | none | – | 8.3147 | kv16 0.000 | kv64 0.000 | kv128 0.000 | kv256 0.000 |
| 1584 | f256 | oracle | – | 0.0000 | kv16 1.000 | kv64 0.978 | kv128 0.631 | kv256 0.124 |
| 1585 | f256 | surprisal | – | 8.3125 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1591 | f128 | none | – | 8.3240 | kv64 0.000 | kv128 0.000 | kv256 0.000 | kv512 0.000 |
| 1592 | f128 | oracle | – | 0.0000 | kv64 0.929 | kv128 0.487 | kv256 0.113 | kv512 0.025 |
| 1593 | f128 | surprisal | – | 8.3239 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1597/1598/1599 | sal2 | none | 0/1/2 | 8.3234/8.3220/8.3216 | 0.000 전부 | | | |
| 1600/1601/1604 | sal2 | surprisal | 0/1/2 | 8.3219/8.3225/8.3242 | 0.000 전부 | | | |
| 1605 | sal2 | oracle | 0 | 0.0000 | kv64 0.966 | kv128 0.591 | kv256 0.129 | kv512 0.031 |

- **해석.** 0020: (1) "The mechanism is sound (oracle)" — 캐시 없이 부트스트랩·회상, 0015의 부트스트랩된 vanilla(0.82/0.30/0.06/0.01)보다 +0.11–0.19, 그래도 용량 한계. (2) surprisal은 "chicken-and-egg" — 모델이 부트스트랩 못 하니 surprisal이 무의미. (3) "From-scratch MQAR bootstrap is a GPU-nondeterministic knife-edge" → 공정한 시험은 frozen 사전학습 모델(long-gdn GDN-1.3B)에서 하자고 권고.
- **경로.** `logs/sh_salience-*_158{3,4,5}.*`, `sh_salience128-*_159{1,2,3}.*`, `sh_sal2-*_{1597..1605}.*`, `scripts/_sh_salience_wait.sh`, `_sh_salience128_wait.sh`, `_sh_sal2_run.sh`, `legacy/experiments/scripts/sh_salience.sh`.

### G17. descriptor 기전 probe (2026-07-09 09:21–09:22)

- **목적.** multi-key 실패가 pooled-chunk descriptor 때문인지(위치별 vs 풀링 묘사) from-scratch 모델에서 측정.
- **세팅.** `lmr.analysis.descriptor_probe`, train-kv 4/8/16, eval 16/32/64, filler 64, chunk 16, 5000 step, seed 0/1.
- **결과(raw).**

| seed | final loss | kv | #q | #chk | pp_localize | pool_chunkhit | pp_decode | pool_decode |
|---|---|---|---|---|---|---|---|---|
| 0 (1610) | 0.0183 | 16 | 2048 | 6 | 0.010 | 0.112 | 0.008 | 0.015 |
| | | 32 | 4096 | 8 | 0.006 | 0.115 | 0.006 | 0.011 |
| | | 64 | 8192 | 12 | 0.008 | 0.121 | 0.007 | 0.007 |
| 1 (1611) | 0.0011 | 16 | 2048 | 6 | 0.035 | 0.194 | 0.005 | 0.026 |
| | | 32 | 4096 | 8 | 0.013 | 0.226 | 0.004 | 0.016 |
| | | 64 | 8192 | 12 | 0.009 | 0.219 | 0.002 | 0.010 |

- **해석.** 0021: "The earlier from-scratch raw-dot probe (1610/1611) was inconclusive (near-random everywhere): it scored by raw query·hidden ... Superseded." → 학습된 라우터를 쓰는 routing probe(§19)로 대체.
- **경로.** `logs/sh_descprobe-s{0_1610,1_1611}.*`, `scripts/_sh_descprobe_wait.sh`, `legacy/experiments/lmr/analysis/descriptor_probe.py`, `notes/descriptor-hypotheses.md`.

### G18. frozen mamba2-370m RULER multikey 재측정 (report 0021, 2026-07-09)

- **목적.** frozen SSC가 multi-key에서 진짜로 손해인지 이 서버의 실제 frozen 모델로 확인.
- **세팅.** `lmr.scripts.eval_ruler --arch mamba2 --model state-spaces/mamba2-370m --heads ckpt/ssc_370m.pt --variant ssc --low-rank-dim 64 --topk 2 --chunk-size 256`, niah_single_1 / niah_multikey_2, 2048/4096, max-examples 50, bf16 (teacher-forced 계열 eval_ruler, free-gen 아님).
- **결과(raw).**
  - 1612 (09:22) **실패**: `--model` 누락 → 기본 모델 차원과 불일치 `size mismatch for 0.router.0.weight: ... [64, 1024] from checkpoint ... [64, 2048] in current model`.
  - 1614 (11:31):

| task | len | vanilla | +RM | +SSC |
|---|---|---|---|---|
| niah_single_1 | 2048 | 0.691 | 0.681 | 0.734 |
| niah_single_1 | 4096 | 0.695 | 0.105 | 0.779 |
| niah_multikey_2 | 2048 | 0.452 | 0.419 | **0.129** |
| niah_multikey_2 | 4096 | 0.165 | 0.072 | 0.113 |

- **해석.** 0021: "SSC helps single-needle and HURTS multi-key — vanilla 0.452 → +SSC 0.129 at 2048 (−0.32)". +RM은 더 나쁨.
- **경로.** `logs/sh_frozen-mk-eval_1612.*`, `logs/sh_frozen-mk2_1614.*`, `scripts/_sh_frozenmk_wait.sh`, `_sh_frozenmk2_wait.sh`.

### G19. query-conditioned routing probe (report 0021, 2026-07-10 04:19–05:18)

- **목적.** 학습된 SSC 라우터 점수(`router(x_answer)·descriptor_chunk`)로, 질의된 key의 청크가 1등인지(전체 청크 중 / key 청크 K개 중) 층별 측정 — single은 되고 multi는 왜 안 되나.
- **세팅.** `lmr.analysis.routing_probe`, frozen mamba2-370m + `ckpt/ssc_370m.pt`, chunk 256, topk 2, 2048/4096, max-examples 50(실제 used=30).
- **결과(raw).**
  - 1628 **크래시** `ValueError: max() arg is an empty sequence`.
  - 1629 single 두 셀만 출력 후 **크래시** `IndexError: index 7 is out of bounds for dimension 0 with size 7` (routing_probe.py:155, key 청크 인덱스).
  - 1631 완료:

| task@len | used | K(평균 key청크) | chance | top1 ALL (mean / best) | top1 KEY (mean / best) |
|---|---|---|---|---|---|
| niah_single_1@2048 | 30 | 1.0 | 1.000 | 0.257 / 1.000 | 1.000 / 1.000 |
| niah_single_1@4096 | 30 | 1.0 | 1.000 | 0.204 / 0.967 | 1.000 / 1.000 |
| niah_multikey_2@2048 | 30 | 7.8 | 0.128 | 0.162 / 0.267 | 0.162 / **0.267** |
| niah_multikey_2@4096 | 30 | 15.9 | 0.063 | 0.074 / 0.179 | 0.074 / **0.179** |

- **해석.** 0021: "Multi fails because the router cannot pick the queried key among the K key-chunks" (best-layer ~2× chance), "The confusion is between KEY-chunks, not with filler" (`top1_amongkeys == top1_all`) → H2(질의 무관 descriptor) **확인**, H6(층 특화) 확인. RESEARCH_STATE: 이것이 유일한 in-distribution 측정이며 핵심 발견 #3.
- **경로.** `logs/sh_routing-probe_1628.*`, `sh_routing-probe2_1629.*`, `sh_routing-probe3_1631.*`, `ckpt/routing_probe.json`, `scripts/_sh_routingprobe_wait.sh`, `_sh_routingprobe2_wait.sh`, `_sh_rp3_wait.sh`, `legacy/experiments/lmr/analysis/routing_probe.py`.

### G20. chunk 크기 sweep (qcroute) (2026-07-10 05:25–10:14)

- **목적.** 청크를 잘게 하면 질의 조건 라우팅이 나아지나(`_sh_qcroute_wait.sh` 제목: "QUERY-CONDITIONED ROUTING PROTOTYPE (chunk-size sweep, multikey_2 @2048)", 기준 chunk 256 best 0.27 / chance 0.13).
- **세팅.** §19와 같으나 `--chunk-size` 128/64/32, 2048만, max-examples 30. 헤드는 chunk 256으로 학습된 그대로.
- **결과(raw).**

| job | chunk | single_1 top1 KEY (mean/best) | single ALL (mean/best) | multikey_2 K | chance | multikey ALL (mean/best) | multikey KEY (mean/best) |
|---|---|---|---|---|---|---|---|
| 1632 | 128 | 1.000 / 1.000 | 0.195 / 1.000 | 14.8 | 0.067 | 0.073 / 0.200 | 0.073 / 0.200 |
| 1633 | 64 | 1.000 / 1.000 | 0.174 / 1.000 | 28.9 | 0.035 | 0.039 / 0.133 | 0.039 / 0.133 |
| 1634 | 32 | 1.000 / 1.000 | 0.148 / 0.967 | 56.0 | 0.018 | 0.018 / 0.133 | 0.026 / 0.133 |

- **해석.** RESEARCH_STATE "Confounded / discarded": "the head was trained at chunk 256, so other sizes are OOD; also finer chunks just multiply candidates. Inconclusive."
- **경로.** `logs/sh_qcroute-c{128_1632,64_1633,32_1634}.*`, `ckpt/routing_probe_c{128,64,32}.json`, `scripts/_sh_qcroute_wait.sh`.

### G21. qa_1 자연어 QA 평가 (2026-07-10 06:35–07:25)

- **목적.** passkey로 학습한 SSC가 자연어 QA(RULER qa_1, SQuAD)에도 도움이 되나.
- **세팅.** §18과 같은 eval_ruler, `--tasks qa_1`, 2048/4096, max-examples 50.
- **결과(raw, job 1635).** qa_1@2048: vanilla 0.531, +RM 0.469, +SSC 0.413. qa_1@4096: 0.503 / 0.063 / 0.357.
- **해석.** RESEARCH_STATE: "the head is passkey-trained, so natural QA is OOD for it. Tells us nothing about the method on natural language." (폐기)
- **경로.** `logs/sh_qa1-eval_1635.*`, `scripts/_sh_qa1_wait.sh`.

### G— effrank-spike (날짜 미상, 로그 없음)

- `scripts/_sh_effrank_wait.sh`는 `logs/sh_effrank-spike_*.out`에서 `T*`(유효 rank 포화점)를 기다리지만 **logs/에 해당 파일이 없다**. 실행되지 않았거나 로그가 지워졌다. 관련 코드 `legacy/experiments/lmr/analysis/gdn2_effective_rank.py`; `notes/adaptive-boundary-research.md`는 effective rank 트리거를 "DEPRIORITIZED"로 기록.

### G22. DynMC: 신호 진단 G0 + 46M 사전학습 G1/G1b (2026-07-17)

- **목적.** DynMC(동적 경계 + 캐시 예산 32의 from-scratch LM) 계획의 관문 확인. G0(`exp0_signal_diag.py` docstring): 4개 경계 신호(drift/erasure/saturation/surprise)가 사실상 같은 신호인지 — 전 쌍 Jaccard > 0.9면 STOP, 0.5–0.9면 진행 + regime 분석, < 0.5면 청신호. G1(`g1_eval.py`): 무작위 분할로 학습한 모델이 고정 256 분할 모델 대비 held-out ppl 열화 ≤ 3%인지(대각 비교).
- **세팅.**
  - 모델: `lmr.dynmc` 46M 설정(params total 63.7M, non-emb 47.3M), dynmc=true, cache_budget 32, fixed_seg_len 256, seg_mode random vs fixed. lr 4e-4, wd 0.01, warmup 200, clip 1.0, **500M 토큰**(1907 step × 262,144 토큰), seed 1234, FineWeb 토큰화 데이터 `/data2/sohyung/dynmc/tokenized`(uniform plan 600M).
  - G1: ctx 4096, rows_per_micro 16, grad_accum 4 (`g1_46m_{random,fixed}_local.json`).
  - G1b: ctx 16384, merge_docs_per_row, rows 2 × accum 8 (`g1b_46m_{random,fixed}_16k.json`).
  - G0: `m-a-p/1.3B-100B-GatedDeltaNet-pure`(사전학습 GDN, 기존 mosc ckpt가 없어 대체), ctx 8192, n-seq 32, target-seg 256.
- **결과(raw).**
  - smoke: 1942 **실패** `AssertionError: u_proj grad missing/nonfinite`; 1943 ALL OK(loss 10.471→10.453, signal drift 평균 2201025.25로 비정상 큼); 1947 **실패** `NameError: name 'test_read_equivalence' is not defined`; 1948 ALL OK(read equivalence max|Δ|=0.000e+00, drift 평균 1.1906).
  - G0: 1946 **실패** `ModuleNotFoundError: No module named 'scipy'`; 1951 결과:

| Jaccard (matched density 0.25/chunk) | drift | erasure | saturation | surprise |
|---|---|---|---|---|
| drift | 1.000 | 0.218 | 0.157 | 0.583 |
| erasure | 0.218 | 1.000 | 0.139 | 0.293 |
| saturation | 0.157 | 0.139 | 1.000 | 0.147 |
| surprise | 0.583 | 0.293 | 0.147 | 1.000 |

  Pearson drift–surprise 0.740, drift–saturation −0.529, saturation–surprise −0.306; Spearman drift–surprise 0.726. `G0_VERDICT=PROCEED_WITH_REGIME_ANALYSIS (off-diag Jaccard min 0.139 max 0.583)`.
  - G1 학습: 1944/1945 **취소**(시작 직후 SIGNAL Terminated). 1949 random: step 0 loss 10.471 → step 1906 loss 3.7760(step 1900 3.6913), ~83.5k tok/s, MFU ~0.095, ckpt `/data2/sohyung/dynmc/runs/0024pre_random/ckpt_{250M,499M}`. 1950 fixed: 10.470 → 3.8087(step 1900 3.7239), ~72.3k tok/s.
  - G1 평가: 1953 **실패**(`ckpt_500M/state_rank0.pt` 없음 — 실제 이름 ckpt_499M), 1954 **실패**(`AssertionError`), 1955:

| 모델 \ eval 분할 | random | fixed |
|---|---|---|
| random 모델 | **45.139** | 46.308 |
| fixed 모델 | 46.333 | **46.686** |

  `G1: random 45.139 vs fixed 46.686 → delta -3.31% → PASS` (0.3M 토큰, holdout docs 44,024 / 50M 토큰).
  - G1b 학습: 1962 random 16K: 10.469 → 3.9794(step 1900 3.9040); 1963 fixed 16K: 10.469 → 4.0093(step 1900 3.9314).
  - G1b 평가 1965(`--merge-docs --n-rows 32`, 0.5M 토큰): random/random **52.959**, random/fixed 53.733, fixed/random 54.081, fixed/fixed **54.595** → `delta -3.00% → PASS`.
- **해석.** G0는 스크립트 판정 기준상 "진행 + regime 분석"(최대 Jaccard 0.583 = drift–surprise). G1/G1b는 무작위 분할 모델이 고정 분할보다 ppl이 오히려 3% 낮아 기준(열화 ≤3%) 통과. `261023_restart-review-and-ipu-plan.md` 표: "DynMC | from-scratch 46M | NIAH 미측정 | 중단". (에이전트 해석) G1은 ppl만 봤고 회상/NIAH는 측정되지 않았으며, 340M 15B 본학습(`configs/0024.json`)은 greenbeard에서 돌지 않았다.
- **경로.** `logs/sh_sh_routing_{1942..1965}.*`, 코드 `legacy/experiments/lmr/dynmc/{train,g1_eval,exp0_signal_diag,smoke_test}.py`, 설정 `legacy/experiments/lmr/dynmc/configs/g1*_46m_*.json`, 체크포인트 `/data2/sohyung/dynmc/runs/{0024pre_random,0024pre_fixed,g1b_random,g1b_fixed}/ckpt_{250M,499M}`, G0 출력 `_workspace/dynmc/exp0`, smoke `_workspace/dynmc_smoke/`.

---

### GA. 0024 — MC-SSC GDN2-370M multi-NIAH 실패 분해 (greenbeard, 2026-07-25 ~ 07-26, logs/ 밖)

- **목적.** end-to-end 학습된 MC-SSC(mean-pool descriptor) 370M이 single NIAH는 되고 multi-key는 무너지는 원인을 write → read → route → generation으로 분해.
- **세팅.** 모델 `LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool`의 5B/30B 체크포인트(16층, topk 2, chunk 256), 코드 long-gdn @e71713e worktree, fla 4b02d15d, env `sh_infocap`. 데이터 A: RULER niah_single_1 / niah_multikey_1 @2048 × 50; 데이터 B: paired 16쌍 × 조건 S(같은 세그먼트 distractor) / D(다른 세그먼트). 추론만(학습 없음).
- **결과(raw, report 0024 및 로그).**
  - E1 hit@2 best-layer: mc-5B single 0.933 / multikey 0.660 / paired_S 0.812 / paired_D 0.562; mc-30B 1.000 / 0.809 / 0.750 / 0.500. chance: single 0.288, multikey 0.486, paired 0.286.
  - E2 oracle 주입(n=16): mc-5B S 0.125→0.312, D 0.188→0.500; mc-30B S 0.000→0.312, D 0.062→**0.562**. (로그 `e2_mc_e2_5B_2309.log`: `mc-5B/D: baseline=0.188 oracle=0.500`; `e2_mc_e2_30B_2310.log`: `mc-30B/D: baseline=0.062 oracle=0.562`)
  - E3: b1_after 0.78–0.82(single≈multi), b2_read_cos_vs_single S 0.974 / D 0.995–0.996(층 평균), S는 L15에서 0.911/0.903.
  - 실패 잡: smoke_2282 `TypeError: chunk_gla_fwd_o_gk() got an unexpected keyword argument 'use_exp2'`(fla API 불일치 → fla 고정으로 해결).
- **해석.** 0024: "routing is the causal bottleneck"; read-side 손상은 S(key 충돌)에서만; oracle이 포화하지 않으므로(0.31–0.56) 게이트/생성 잔여 손실 존재.
- **경로.** `/data2/sohyung/mc_niah/logs/{smoke_2282..2285,e1_2287,e1_2288,e1_2312,e1_2313,e2_smoke_2308,e2_mc_e2_5B_2309,e2_mc_e2_30B_2310,e3_mc_e3_5B_2314,e3_mc_e3_30B_2315}.log`, 결과 `legacy/experiments/lmr/analysis/260725_mc_niah_analysis/results/{e1_routing,e2_oracle,e3_fidelity}.json`, sbatch `.../sbatch/e{1,2,3}.sbatch`, 보고서 `legacy/experiments/report/0024.md`, `0024_kr.md`.
- 참고: 0025(X1/X2/X4 router 진단)의 GPU 부분은 VESSL에서 실행(계획서 rev2: "GPU = VESSL, CPU 전처리·후처리 = greenbeard"). 루트 `lmr/analysis/260725_mc_niah_analysis/results/`의 `probe_bilinear_router.json`(07-28), `x4_*.json`(07-29)은 결과 사본이다.

### GB. 7월 말 ~ 8월 MC 라우팅 진단·ssketch 계열 (greenbeard, logs/ 밖, 요약만)

모두 `/data2/sohyung/mc_niah/logs/`, host aigpu0423. 해석은 `261023_restart-review-and-ipu-plan.md`(10-23) 요약을 따른다. 상세 보고서는 `/data2/sohyung/worktrees/lmr-mc-routing-tracks/report/`(이번 조사 범위 밖).

| 로그 | 날짜 | 내용 | raw 숫자 |
|---|---|---|---|
| normbias_2528 | 07-30 | mc-5B 라우터 점수 정규화 probe(n_rows 16) | best-layer hit@2 raw 0.4375(L4), frob 0.4375(L13), meank 0.5625(L14) |
| ssketch_smoke_2726/2731/2734/2748 | 08-02 | ssketch descriptor parity/smoke | 2748: "[smoke] 전부 통과" |
| ssketch_2735 | 08-02 | mc-5B descriptor=ssketch r8 hit@2 | single 0.911, multikey 0.638, multivalue 0.62, paired_S 0.625, paired_D 0.4375 |
| ssketch_2739 | 08-02 | mc-5B descriptor=meank, router_query 기본 | single 0.933, multikey 0.660, multivalue 0.76, paired_S 0.859, paired_D 0.766 (n=64) |
| ssketch_2740 | 08-02 | router_query=q | **취소**(SIGNAL Terminated) |
| ssketch_ruler_2870/2943, meank4_ruler_2871/2944, vanilla_ruler_2945, ssketch_s{1,2}_ruler_3025/3026 | 08-03~05 | 5B 체크포인트 RULER 16셀(S-NIAH 등 × 1K–8K, 셀당 50) | 평균(에이전트 계산): vanilla 27.2(8K 4.0), meank topk4 36.2(8K 7.5), ssketch s0 37.9 / s1 26.9 / s2 28.5 → 3시드 평균 31.1. 노트: "ssketch 5B 31.1 vs meank 36.3 vs vanilla 27.3. 자기 규칙 hit@2 4.1 → 실패" |
| ppl_lmeval_3031 | 08-06 | vanilla / meank / ssketch s0–s2 ppl·lm-eval | 결과 json만 기록(숫자 미확인) |
| gates_3036 | 08-06 | mc4-30B meank / ssketch r0·r8 게이트 진단 | 결과 json만. 노트: "게이트 상한 측정: multikey 28→56 중 +24pp가 섞는 비중에서 나옴" |
| diag2x2_3091, diag5b_3108 | 08-07 | 8K 진단 | diag5b ssketch-5B r8 @8192: single hit@2 1.0, multikey 0.2, multivalue 0.2 |
| probe8k_3117/3118 | 08-08 | 8K x1 dump + probe | 3118 **TIME LIMIT 취소**(probe 일부 완료) |
| mv_cov_3173 | 08-09 | mc4-30B meank @8192 | single 0.898(L1), multikey **0.06**(L2), multivalue 0.4(L13) |

(에이전트 해석) 이 묶음은 저장소 `logs/`에 없고 별도 worktree 코드로 돌았다. 해석 권위 문서는 해당 worktree의 report(0027 등)이며, 10-23 노트는 "8K 3.0→18.5" 헤드라인을 철회하고 "vanilla 11.0, MC 원본 2.7, MC+공유+max-sim 13.0(p=0.50), oracle 74.0"으로 정정했다고 적는다(이 수치의 실행 서버는 노트에 명시 없음).


### G 실패·크래시 잡 일람 (logs/)


| job | 그룹 | 원인 |
|---|---|---|
| 1072 | 환경 | `NVCC_PREPEND_FLAGS: unbound variable` (conda activate 스크립트) |
| 1073/1074 일부 | 백본 검증 | Mamba3 SISO 커널 없음 |
| 1079, 1080 | MQAR 첫 시도 | 사용자 취소(SIGNAL Terminated) |
| 1076–1078, 1081, 1086–1090, 1094, 1095, 1206 | MQAR | 학습 실패(loss ~8.32 고정, 오류 아님) |
| 1239 | RULER | CUDA OOM(SSC 학습 train_len 2048) |
| 1243 | RULER | Triton cache manager 예외(디스크) |
| 1244, 1256 | RULER | RULER 데이터 생성 실패(`nltk` 없음) |
| 1257, 1264 | RULER | +SSC free-gen 6h/4h TIME LIMIT |
| 1249–1252 | 불규칙 MQAR | 무작위 filler 버그로 전부 0 |
| 1255 | 불규칙 MQAR | mean-pool 요약 버그로 oracle 실패(loss 7.59) |
| 1261 | 불규칙 MQAR | fixed chunk=2 실패(설계상 예상된 결과) |
| 1274–1276 | 비지도 | STE 붕괴(loss 8.32) |
| 1612 | frozen eval | `--model` 누락 → state_dict size mismatch |
| 1628 / 1629 | routing probe | ValueError(빈 max) / IndexError(key 청크 인덱스) |
| 1942, 1947 | DynMC smoke | u_proj grad 없음 / NameError |
| 1944, 1945 | DynMC G1 | 사용자 취소 |
| 1946 | DynMC G0 | scipy 없음 |
| 1953, 1954 | DynMC G1 eval | ckpt 경로(500M→499M) / AssertionError |


---

## II. mslm 코드베이스 실험 (2026-09 ~ )

### E0. [이전 결과 인용] MC 라우팅 실패 진단 (2026-07 ~ 2026-09, 보류 시점 요약)

**목적**
- 2026-09-30 보류 문서 `260930_SHELVED-routing-diagnosis-and-options.md`의 요약이다. 이 세션이 11-02에 읽고 실험 가치를 판단하는 근거로 썼다.
- 질문: MC-SSC(meanpool) 라우터는 왜 NIAH에서 정답 구간을 못 고르는가?

**세팅**
- GDN2-370M, 30B 학습 ckpt, backbone 동결. 8K diverse-key NIAH, 짝지은 200문항.
- 50M 사다리 첫 칸도 함께 측정했다.

**3. 결과 raw data** (문서에서 그대로 옮김)
- 8K diverse-key NIAH, 370M 동결: learned 3.0 → 18.5(짝지은 200문항, p=0.0000), oracle 74.0.
  - 18.5는 파라미터를 하나도 추가하지 않은 두 개입으로 얻었다: 라우팅 공유(L0 결정을 16층이 재사용) + 하위 블록 max-sim(m=8).
  - 오프라인 라우터 CE(backbone 동결)는 L0·L1에서 작동했다. 이것이 18.5의 출처다.
- 50M 사다리: oracle 20.0 vs 무개입 0.33. 신호가 있는 층이 7개다.
  - **미해결:** 최근 고정 프로토콜로 잰 50M oracle은 0.00으로, 위 20.0과 어긋난다.
- 진단 수치
  - router score → gold AUC 0.46, 질의 의존 이득 +0.026
  - descriptor Gram 비대각 0.958~0.9998, 중간층 위치 R² 0.985, 답 span은 벡터의 2%
  - oracle을 16/16층에 주면 82, 3/16층에 주면 4
  - gate 무게 = 라우팅 점수: gold 0.123이면 되고 0.046이면 안 된다
  - dense 0점, k=8 18→10, 2K 상한에서 이득 28pp 중 24pp가 섞기에서 나옴
  - ssketch(state를 주소로 사용) 4.1 vs key 규칙 98.0
- 계획서 C4 행: 얼린 모델에서 max-sim 13.0 vs vanilla 11.0, p=0.50(차이 없음).
- 선점 확인 문서: Latent RAG 2×2에서 oracle 라우터 + hard 읽기 92% vs soft 8%.
- 백본 학습 시도 기록

| 시도 | 결과 |
|---|---|
| DSC v3/v4 from-scratch, LM 손실만 | conversion 1.0, chunk 0·1 편향, 5B에서 퇴행 |
| A2 memory-only FT 100M | single 14.4배, multikey_1 회복 없음 |
| A2 1B 연장 | −90% 퇴행 |
| ReLU 게이트 FT, LM 손실, 4K, 500M | 8K에서 5연속 0, gold-hit 0.77→0.41 |

  - 10-23 재검토 문서는 A2의 "14.4배"와 "−90%"를 측정 무효로 판정했다.

**결과 (해석 (당시 결론))**
- 원인을 넷으로 정리했다: (1) 선택하지 않은 구간에는 기울기가 0, (2) meanpool descriptor가 구간끼리 거의 같음, (3) 신호가 L0·L1에만 있음, (4) 고르는 점수와 섞는 무게가 같은 값.
- 이 진단이 from-scratch span-copy CE + sep gate 계획(261029)의 동기가 됐다.
- 11-02에 이 세션은 "oracle 74 vs learned 3 격차가 크니 stage 1은 돌릴 가치가 있다"고 판단했다.

**경로**
- `/home/sohyung/COG-second-brain/04-projects/linear-attention-multistate/planning/260930_SHELVED-routing-diagnosis-and-options.md`
- 원 수치 출처
  - `resources/260907_SUMMARY-diverse-key-journey.md`
  - `~/long-gdn/MC_ROUTING_INTEGRATED_REPORT_KO.md`
  - `resources/260911_ladder-50m-results.md`
- 이전 보고서 0001~0026은 `legacy/`(repo의 report/)로 옮겼다. 제목만에서 확인했다.

---

### E1. greenbeard smoke test: 새 mslm 코드베이스 (2026-10-26)

**목적**
- 새 from-scratch 코드베이스(GDN-2 + MC-SSC, PR #3)가 학습, 체크포인트 재개, 평가(learned/oracle), 커널 정확도를 통과하는지 확인한다.

**세팅**
- 장비: greenbeard SLURM, partition main, `--gres=gpu:rtx6000:1`. GPU는 RTX PRO 6000 Blackwell.
- config: `configs/smoke_greenbeard.env`
  - 50M 프리셋. MC 76,667,360 params, vanilla 71,948,768 params
  - tokens_per_step 16384, accum 2, total_steps 6104, warmup 61
- 데이터와 평가
  - smoke용 FineWeb-Edu 소량을 사전 토큰화했다.
  - RULER 문항은 길이 2048, 과제마다 n=2.
- job
  - 6775 `sh_mslm-train`: MC 12 step 연속 학습
  - 6776 `sh_mslm-smokeB`: MC 6+6 step 재개, vanilla 4 step + eval, chained 2 step
  - 6784 `sh_mslm-smokeC`: gpu selftest
  - 6785 `sh_mslm-eval`: MC learned/oracle 평가

**결과 (raw)**
- 연속 12 step(6775): step 12 loss 9.72903, val_loss 9.61061, 45,391 tok/s, 10.13 GB.
- 재개 6+6 step(6776): step 12 loss 9.72913, val_loss 9.61066, 약 46.5k tok/s.
  - 연속 학습과 1e-4 이내로 같다.
- vanilla 4 step: loss 10.30484, val 10.20201, 약 88k tok/s, 8.29 GB.
  - vanilla eval: 모든 과제 `string_match_all` 0.0(n=2).
- 커널 검사
  - `chunk_gdn2` vs naive: out rel err 1.49e-03, state rel err 1.13e-03
  - gather_read triton vs torch: rel err 2.23e-07
  - MC fwd/bwd bf16: loss 7.0527, grad norm 2.684 → `SELFTEST_OK`
  - 처음 smokeB의 selftest는 FLCE `view size is not compatible`로 실패했다. `.contiguous()` 수정(d16803c3) 후 smokeC에서 통과했다.
- MC eval(길이 2048, n=2, string_match_all은 전부 0.0)

| task | learned routing_hit | random_hit | oracle routing_hit |
|---|---|---|---|
| niah_single_1 | 0.53125 | 0.5714 | 1.0 |
| niah_single_2 | 1.0 | 1.0 | 1.0 |
| niah_single_3 | 0.6875 | 0.5714 | 1.0 |
| niah_multikey_1 | 0.84375 | 0.7857 | 1.0 |
| niah_multikey_2 | 0.65625 | 0.5714 | 1.0 |
| niah_multikey_3 | 0.59375 | 0.5714 | 1.0 |
| niah_multivalue | 1.0 | 0.9857 | 1.0 |

- CPU tiny smoke: losses [5.7454, 5.419, 5.1113], reload_loss_equal True, oracle_picks_gold True.
- 초기하분포 무작위 기준선: 사양서는 23.9%였는데 코드 값은 24.5%였다. 후보 32개면 23.8%가 된다.

**결과 (해석)**
- 학습, 재개, 평가, oracle 경로가 모두 동작한다. oracle hit는 1.0이다.
- 확인하지 못한 것: 8K/0.5M 배치 규모, multi-GPU/노드, GH200.
- 이후 PR #3 → 사용자가 병합.

**경로**
- 로그: `/data2/sohyung/mslm_smoke/slurm_logs/`
  - `sh_mslm-train_6775.*`, `sh_mslm-smokeB_6776.*`, `sh_mslm-smokeC_6784.*`, `sh_mslm-eval_6785.*`
- run: `/data2/sohyung/mslm_smoke/runs/smoke_gb_{a,b,vanilla,chained}`
  - eval: `smoke_gb_a/eval/step_0000012/items.jsonl`
- 코드 worktree: `/data2/sohyung/worktrees/lmr-ms-pretrain-base`(branch `sh/ms-pretrain-base`)
- env: `/data2/sohyung/conda-envs/sh_mslm`
- 10-29 greenbeard 파이프라인 시험 job 6969~6972는 GPU에서 돌기 전에 사용자 요청으로 취소했다. 결과는 없다.

---

### E2. span 복사 라벨 통계, 0단계 (2026-10-29)

**목적**
- 자연 텍스트(FineWeb-Edu)에서 span-copy 라벨이 8K 창마다 충분히 나오는지 본다.
- 그 결과로 n-gram 길이 n을 정한다.

**세팅**
- greenbeard smoke 데이터 val 1M 토큰, 8K 창 122개, n ∈ {8, 16, 32}. CPU에서 쟀다.
- 구현 PR #8을 낸 다른 세션이 측정했고, 결과는 계획서에 적혀 있다. 이 세션은에서 읽었다.

**결과 (raw)**

| n | 창당 라벨 평균 | 중앙값 | 10개 미만 창 | 구간 거리 사분위 |
|---|---|---|---|---|
| 8 | 30.7 | 10 | 48% | 2 / 3 / 5 |
| 16 | 11.3 | 0 | 83% | 2 / 2 / 6 |
| 32 | 5.7 | 0 | 92% | 2 / 2 / 6 |

**결과 (해석)**
- 기본값을 n=8로 정했다(`COPY_NGRAM=8`, `COPY_MIN_GAP=2`, `COPY_MAX_GOLD=4`).
- step당 라벨은 약 2,000개다. 다만 창의 절반이 10개 미만이고 거리가 2~5구간으로 가까워 경계선이다.
- 11-02 세션 판단: 라벨이 드물고 가까워서 "최근 구간을 고르라"는 버릇을 배울 위험이 있다. 그래서 판정 지표를 8K RULER 적중률로 뒀다.
- KISTI 전체 val(20M)로 다시 재는 일은 아직 안 했다.

**경로**
- 계획서: `/home/sohyung/COG-second-brain/04-projects/linear-attention-multistate/planning/261029_from-scratch-routing-ce-plan.md` §4 0단계
- 재현 명령: `CONFIG=configs/mc_ce_50m.env python -m mslm.data.span_copy --stats --windows 2000`

---

### E3. KISTI bootstrap GPU 자가시험 + 파이프라인 smoke (2026-10-29)

**목적**
- KISTI 한강 GH200(aarch64)에서 env, 커널, 학습, 재개, 평가가 도는지 확인한다.

**세팅**
- bootstrap: salloc으로 node1603, GH200 4장. env `sh_mslm`(torch 2.9.1+cu128, triton 3.5.1, fla 4b02d15d, transformers 4.57.6).
- smoke job 1178127: `configs/smoke_kisti.env`, GPU 1장. 순서는 selftest → 학습 → 재개 → eval.
- prepare_data 1177817: 7분 25초. prepare_eval 1177818: 2분 54초.

**결과 (raw)**
- node1603 GPU 시험
  - `chunk_gdn2` vs naive: out rel err 1.49e-03, state 1.14e-03
  - gather_read rel err 2.24e-07
  - MC fwd/bwd bf16: loss 7.0526, grad norm 2.684 → `[ok] GPU kernels`
- 로그인 노드(glogin03)의 GPU는 `CUDA-capable device(s) is/are busy or unavailable`로 실패했다. 계산용으로 막혀 있다(PR #6에서 건너뛰게 수정).
- smoke 1178127: COMPLETED(5분 08초). `SMOKE_OK`가 생겼고 train.jsonl은 `{"event": "end", "step": 30, "tokens": 1966080}`.
- natStr는 비독점 GPU 작업의 `--cpus-per-task`를 거부했다(PR #6에서 수정).

**결과 (해석)**
- KISTI GPU 경로 전체가 통과했다. 이후 본학습(E4)을 제출했다.

**경로**
- `/scratch/t512a03/mslm/runs/smoke_mc_50m/`(SMOKE_OK, ckpt step 10/20/30)
- 로그: `/scratch/t512a03/mslm/slurm_logs/`
- repo: `/home01/t512a03/linear-memory-routing`

---

### E4. 50M vanilla vs MC from-scratch, seed 0, 10B 토큰 목표 (2026-10-29 ~ 10-30)

**목적**
- MC-SSC(설정 2, CE 없음)가 같은 조건의 vanilla GDN-2(설정 1)와 비교해 LM loss와 긴 문맥 기억이 어떤지 본다.
- seed 3개로 계획했다.

**세팅**
- 모델: 50M(16층, d=384, 6 head × 128, FFN 768). `configs/vanilla_50m.env`와 `configs/mc_50m.env`는 `MODEL_ARCH`만 다르다.
- MC: chunk 256, top-4, meanpool/dot, softmax gate, independent, `MC_AUX_LOSS=none`.
- 데이터: FineWeb-Edu sample-10BT, TinyLlama v1.1 토크나이저. ctx 8192, step당 524,288 토큰.
- 학습: total_steps 19,074(10B 토큰), warmup 191. AdamW, LR 4e-4, cosine, wd 0.1, bf16.
- 자원: KISTI natStr, 노드 1개(GH200 4장), `--exclusive`, SB_TIME 24h. val과 ckpt는 500 step마다, KEEP_CKPTS=3.
- job
  - 1178128: vanilla s0
  - 1178445: mc s0
  - 1178129, 1178130: vanilla s1, s2 — 사용자가 46분 만에 취소
- wandb 업로드는 로그인 노드에서 사후 동기화로 했다(entity ks000225, project linear-memory-routing).

**결과 (raw)**
- sacct

| JobID | JobName | State | Start | End | Elapsed |
|---|---|---|---|---|---|
| 1178128 | mslm-vanilla_50m_s0 | NODE_FAIL | 2026-10-29T17:20:59 | 2026-10-30T05:00:30 | 11:39:31 |
| 1178129 | mslm-vanilla_50m_s1 | CANCELLED | 2026-10-29T17:20:59 | 2026-10-29T18:07:17 | 00:46:18 |
| 1178130 | mslm-vanilla_50m_s2 | CANCELLED | 2026-10-29T17:20:59 | 2026-10-29T18:07:17 | 00:46:18 |
| 1178445 | mslm-mc_50m_s0 | NODE_FAIL | 2026-10-29T18:13:43 | 2026-10-30T05:00:30 | 10:46:47 |

- 마지막 학습 줄
  - vanilla s0: step 16200, loss 2.81714
  - mc s0: step 11410, loss 2.90245
  - vanilla s1: step 990, loss 3.54935
  - vanilla s2: step 1030, loss 3.51904
- wandb summary
  - mc s0: step 11400, train/loss 2.90809, val/loss 2.82039, 171,451 tok/s
  - vanilla s0: step 16190, train/loss 2.76724, val/loss 2.71096, 223,507 tok/s
  - 마지막 기록은 둘 다 10-29 19:04 UTC(= 10-30 04:04 KST)
- val loss, 같은 step 비교(wandb history)

| step | vanilla | mc | mc − vanilla |
|---|---|---|---|
| 500 | 3.9502 | 4.0808 | +0.1306 |
| 1000 | 3.4582 | 3.5703 | +0.1121 |
| 2000 | 3.1580 | 3.2244 | +0.0665 |
| 3000 | 3.0385 | 3.0948 | +0.0564 |
| 4000 | 2.9682 | 3.0203 | +0.0521 |
| 5000 | 2.9196 | 2.9691 | +0.0495 |
| 6000 | 2.8814 | 2.9316 | +0.0501 |
| 7000 | 2.8522 | 2.9011 | +0.0489 |
| 8000 | 2.8254 | 2.8751 | +0.0496 |
| 9000 | 2.8073 | 2.8561 | +0.0488 |
| 10000 | 2.7882 | 2.8373 | +0.0491 |
| 11000 | 2.7718 | 2.8204 | +0.0486 |

- 초반 mc s0: step 40 loss 8.37, step 210 loss 4.95. 사용자 turm 화면에서 시작 11분 43초 시점.
- RULER 평가는 하지 않았다(학습 미완료).

**결과 (해석)**
- MC는 vanilla보다 val loss가 약 +0.049 높다. step 4000 이후로 차이가 고정됐다. 처리 속도는 약 23% 느리다.
- 두 run이 같은 시각에 멈췄으므로 24h 한도가 원인이 아니다. sacct로 NODE_FAIL을 확인했다.
- 11-02 결정
  - 50M seed 1·2(약 230 GPU-h)는 빼기를 권했다. 계획서는 3-seed 검정을 370M에서 하기 때문이다.
  - seed 0 10B 마무리(약 36 GPU-h)는 선택 사항으로 남겼다. 아직 재개하지 않았다.
  - 3B 시점 ckpt는 KEEP_CKPTS=3 때문에 지워졌다. 그래서 설정 2를 다시 돌렸다(E6의 mon).

**경로**
- run: `/scratch/t512a03/mslm/runs/{vanilla,mc}_50m_ctx8192_s0/`
  - vanilla ckpt step_15000/15500/16000
  - mc ckpt step_10000/10500/11000
- `vanilla_50m_ctx8192_s1`(ckpt 500), `_s2`(ckpt 500, 1000)
- wandb: `ks000225/linear-memory-routing`, runs `vanilla_50m_ctx8192_s0`, `mc_50m_ctx8192_s0`(forge.coreweave.com/wandb/...)
- 로그: `/scratch/t512a03/mslm/slurm_logs/mslm-{vanilla,mc}_50m_s0_*`
- 조회 스크립트: scratchpad `wb_read.py`, `wb_hist.py`

---

### E5. RULER 라우팅 모니터 문항 생성 (2026-11-02)

**목적**
- 학습 중 `ruler_hit`를 재는 고정 문항을 만든다. 최종 평가 seed 42와는 다른 seed를 쓴다.

**세팅**
- `EVAL_DATA_SEED=7 EVAL_TASKS=niah_multikey_1 EVAL_LENGTHS=8192 EVAL_NUM_SAMPLES=64`
- prepare_eval: `CONFIG=configs/mc_ce_50m.env`

**결과 (raw)**
- 64문항이 생겼고 meta.json의 seed는 7이다.
- 연구실 ACL 때문에 nltk punkt_tab 로드가 실패했다. `chmod -R go-w .../hf_cache/nltk_data`로 해결했다.

**결과 (해석)**
- E6와 E7의 `ruler_hit` 측정 기준이다. 무작위 기준선은 13.68%다(E6 로그의 ruler_random_hit 0.1368).

**경로**
- `/scratch/t512a03/mslm/data/route_monitor/TinyLlama__TinyLlama_v1.1/8192/niah_multikey_1/validation.jsonl`

---

### E6. Stage 1a: MC 재실행(mon) + span-copy CE 가중치 스윕 (2026-11-02 21:0x ~ 11-03 19:45)

**목적**
- LM 손실만 쓰면 라우터가 정답 구간을 못 고르는가(설정 2)?
- span-copy CE를 걸면 적중률이 무작위보다 확실히 오르는가(설정 3)?
- 통과 기준(3B 토큰 시점)
  - RULER 적중률 ≥ 무작위 +15pp, 그리고 설정 2보다 높을 것
  - val loss는 설정 2 대비 +0.02 이내
- 중단 기준: 무작위 +5pp 이내.

**세팅**
- 50M, ctx 8192, seed 0. `MAX_STEPS=5700`(약 3B 토큰, 2,988,441,600 토큰). LR 일정은 10B 기준(total_steps 19074, warmup 191).
- pack 실행: 노드 1개(node1345), GPU 4장에 run 하나씩. 각 run은 GPU 1장, accum 64, MICRO_BATCH 1. `sample_ids` 덕분에 데이터 순서는 4-GPU run과 같다.
- runs

| run | config | 바꾼 변수 |
|---|---|---|
| `mc_mon_50m_ctx8192_s0` | `configs/mc_50m.env` | RUN_TAG=mon (설정 2, aux 없음) |
| `mc_ce_w003_50m_ctx8192_s0` | `configs/mc_ce_50m.env` | MC_AUX_LOSS=span_copy_ce, MC_AUX_WEIGHT=0.03 |
| `mc_ce_w01_50m_ctx8192_s0` | 〃 | MC_AUX_WEIGHT=0.1 |
| `mc_ce_w03_50m_ctx8192_s0` | 〃 | MC_AUX_WEIGHT=0.3 |

- 모든 run에 `ROUTE_MONITOR_FILE`(E5)을 줬다. VAL_EVERY 500마다 val_loss, copy_hit(val 텍스트 span-copy 위치, 326 positions), ruler_hit(64문항, 마지막 프롬프트 토큰, teacher-forced)를 기록했다.
- job
  - 1200628 `mslm-pack-stage1a`: 24h
  - 1201020 `mslm-pack-stage1a_tail`: `afterany:1200628`, 3h. 실행됐지만 이어 받을 step이 없어 바로 끝났다.
- 계획서의 설정 4·5(sepgate, maxsim)는 1a 이후에 낼 예정이었고, 아직 내지 않았다.

**결과 (raw)**
- 시작 시점(step 10/20)

| run | step | loss | aux_loss | tok/s | mem |
|---|---|---|---|---|---|
| mon | 20 | 9.4499 | 0.0 | 45,380 | 10.18 GB |
| ce_w003 | 10 | 10.04108 | 2.13593 | 34,329 | 10.19 GB |
| ce_w01 | 10 | 10.04121 | 2.1352 | 34,357 | 10.19 GB |
| ce_w03 | 10 | 10.04198 | 2.13453 | 34,510 | 10.19 GB |

  - 예상 시간: mc 3B 토큰에 약 18.4h, ce는 약 24.3h. mon이 끝난 뒤 ce 속도가 약 37k tok/s로 올라 24h 안에 끝났다.
- 중간 기록(11-03 17:02)
  - mon: step 5700 loss 3.03092, 45,627 tok/s, 종료
  - ce_w003: step 4990 loss 3.02926, aux 1.21183
  - ce_w01: step 5000 loss 3.11125, aux 1.14629
  - ce_w03: step 5020 loss 3.22341, aux 1.38677
- 최종 eval 기록(마지막 4회). 열은 step, val_loss, copy_hit, ruler_hit, ruler_hit_by_layer(L0..L15).
  - copy_random_hit 0.3717, ruler_random_hit 0.1368

```
== mon end 5700
4500 3.00487 0.601 0.1141 [0.032, 0.095, 0.127, 0.079, 0.238, 0.143, 0.111, 0.111, 0.095, 0.111, 0.111, 0.111, 0.111, 0.111, 0.127, 0.111]
5000 2.98351 0.5993 0.1022 [0.048, 0.095, 0.095, 0.048, 0.063, 0.127, 0.127, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.127, 0.127]
5500 2.95593 0.6007 0.1161 [0.079, 0.175, 0.111, 0.127, 0.079, 0.143, 0.111, 0.111, 0.111, 0.127, 0.111, 0.111, 0.111, 0.111, 0.127, 0.111]
5700 2.95184 0.5978 0.1121 [0.048, 0.19, 0.111, 0.063, 0.048, 0.175, 0.127, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.127, 0.127]
== ce_w003 end 5700
4500 3.00368 0.6597 0.1657 [0.222, 0.254, 0.254, 0.333, 0.159, 0.159, 0.127, 0.127, 0.095, 0.095, 0.127, 0.127, 0.143, 0.127, 0.175, 0.127]
5000 2.98264 0.7475 0.1498 [0.175, 0.238, 0.19, 0.19, 0.127, 0.159, 0.127, 0.19, 0.143, 0.111, 0.143, 0.111, 0.111, 0.127, 0.127, 0.127]
5500 2.95854 0.6745 0.1488 [0.159, 0.19, 0.19, 0.175, 0.143, 0.159, 0.127, 0.206, 0.127, 0.095, 0.127, 0.127, 0.127, 0.127, 0.19, 0.111]
5700 2.95301 0.7243 0.1706 [0.238, 0.19, 0.27, 0.206, 0.222, 0.19, 0.175, 0.206, 0.143, 0.159, 0.127, 0.127, 0.127, 0.095, 0.143, 0.111]
== ce_w01 end 5700
4500 3.04908 0.7456 0.2044 [0.127, 0.143, 0.286, 0.333, 0.254, 0.333, 0.159, 0.206, 0.222, 0.206, 0.159, 0.19, 0.206, 0.143, 0.143, 0.159]
5000 3.02776 0.7214 0.2063 [0.159, 0.238, 0.254, 0.365, 0.222, 0.317, 0.19, 0.222, 0.222, 0.238, 0.127, 0.095, 0.175, 0.159, 0.159, 0.159]
5500 3.00368 0.7211 0.1885 [0.143, 0.286, 0.19, 0.206, 0.159, 0.317, 0.143, 0.19, 0.175, 0.286, 0.175, 0.143, 0.143, 0.127, 0.159, 0.175]
5700 2.99596 0.7609 0.2431 [0.19, 0.222, 0.254, 0.286, 0.238, 0.46, 0.175, 0.302, 0.238, 0.317, 0.254, 0.175, 0.19, 0.19, 0.222, 0.175]
== ce_w03 end 5700
4500 3.15527 0.7262 0.1964 [0.159, 0.127, 0.175, 0.222, 0.143, 0.254, 0.159, 0.286, 0.286, 0.238, 0.238, 0.159, 0.19, 0.127, 0.19, 0.19]
5000 3.13397 0.74 0.1687 [0.222, 0.175, 0.175, 0.127, 0.079, 0.143, 0.143, 0.206, 0.222, 0.19, 0.127, 0.143, 0.143, 0.175, 0.238, 0.19]
5500 3.11348 0.7573 0.1885 [0.127, 0.19, 0.159, 0.175, 0.143, 0.317, 0.238, 0.333, 0.238, 0.19, 0.111, 0.127, 0.111, 0.159, 0.222, 0.175]
5700 3.10123 0.7694 0.2044 [0.127, 0.127, 0.143, 0.159, 0.127, 0.302, 0.254, 0.333, 0.27, 0.333, 0.175, 0.206, 0.159, 0.206, 0.19, 0.159]
```

- 세션이 보고한 요약표(11-04 최종). "최근 3회 평균"은 5000·5500·5700 step의 ruler_hit 평균이다.

| run | NIAH ruler_hit 최근 3회 평균 (무작위 13.7%) | 5700 step | copy_hit (무작위 37%) | 최종 val loss (mon 대비) |
|---|---|---|---|---|
| mon | 11.0% (−2.7pp) | 11.2% | 60% | 2.952 (기준) |
| ce_w003 | 15.6% (+2.0pp) | 17.1% | 72% | 2.953 (+0.001) |
| ce_w01 | 21.3% (+7.6pp) | 24.3% (+10.6pp) | 76% | 2.996 (+0.044) |
| ce_w03 | 18.7% (+5.0pp) | 20.4% | 77% | 3.101 (+0.149) |

  - 11-03 중간 보고에서 ce_w03의 최고값은 27.9%였고, 세션은 이를 일시적으로 튄 값으로 봤다.

**결과 (해석 (당시 결론과 사용자 결정))**
- **통과 기준을 넘은 설정은 없다.**
  - 무작위 +15pp(약 29%)를 넘은 설정이 없다.
  - val loss 기준(+0.02 이내)은 w003만 만족한다.
  - w003은 중단 조건(+5pp 이내)에 걸리고, w03은 경계, w01은 5~15pp 사이 애매한 구간이다.
- 기존 MC router(mon)는 NIAH에서 무작위보다 낮다(11% < 13.7%). 370M의 "oracle 74 vs learned 3" 현상이 50M from-scratch에서도 재현됐다(C1 지지).
- CE를 걸면 개선된다(w01은 약 2배). 하지만 hit를 올린 만큼 LM loss를 잃는다.
- copy_hit는 60→76%로 오르지만 NIAH hit는 21%에 머문다. 학습 과제를 NIAH로 일반화하지 못한다.
- 층별로는 w01 최종에서 L5 46%, L7·L9 30% 이상이다. 다만 64문항이라 노이즈가 몇 pp 있다.
- 원인 가설
  - (1) meanpool descriptor 희석
  - (2) 라벨과 NIAH 불일치(가깝고 드문 라벨)
  - (3) CE gradient가 backbone q/k로 흘러 LM과 경쟁
- 사용자 결정
  - "여기서 멈추고 논의" → 1b(sepgate, maxsim)는 내지 않았다.
  - "관련된 다른 CE 연구들 참고 + WSS(2610.25621)" → 세션이 조사했다. WSS, MSA, DeepSeek-V3.2 DSA, SeerAttention을 보니 router 입력을 detach하거나 base를 얼리고, router 전용 Q/K와 cosine×γ를 쓴다.
  - 이를 근거로 "router 분리(detach) 변형"을 제안했고, 사용자가 "구현 진행"이라고 답했다 → E7.

**경로**
- run: `/scratch/t512a03/mslm/runs/mc_{mon,ce_w003,ce_w01,ce_w03}_50m_ctx8192_s0/`(train.jsonl, ckpt)
- pack 계획 파일: `/scratch/t512a03/mslm/jobs/packs/stage1a_20261102_195428.tsv`, `stage1a_tail_20261102_211635.tsv`
- 로그
  - `/scratch/t512a03/mslm/slurm_logs/mslm-pack-stage1a_1200628.out`, `mslm-pack-stage1a_tail_1201020.{out,err}`
  - run별: `/scratch/t512a03/mslm/jobs/packs/<run>_<jobid>.log`
- wandb: `https://forge.coreweave.com/wandb/ks000225/linear-memory-routing/runs/mc_mon_50m_ctx8192_s0`, `.../mc_ce_w003_50m_ctx8192_s0`(w01, w03도 같은 형식). 계산 노드에서 직접 올렸다.
- 코드: PR #8(span_copy_ce, sep_softmax, subblock8/maxsim), #9(wandb hit 지표), #10(run_idea.sh/configs), #11~#13(run_pack.sh, LINES 버그 수정, AFTER=)

---

### E7. Stage 1c: detached router (2026-11-05 ~ 11-06)

**목적**
- 1a에서 본 trade-off(hit↑ ⇔ LM loss↑)가 CE gradient와 backbone의 경쟁 때문인지 가린다(가설 3).
- router를 backbone과 분리(detach)하면 다음 두 가지가 되는지 본다.
  - LM loss가 mon과 같게 유지된다.
  - NIAH hit가 w01의 21%를 넘는다.
- 판정 기준
  - val loss ≈ mon(2.952)이면 detach가 trade-off를 없앤 것이다.
  - 그 상태에서 NIAH hit > 21%면 성공이다.
  - hit가 20% 근처에 머물면 문제는 라벨 쪽이다.

**세팅**
- 새 router `MC_ROUTER=detached_cos`(PR #14, `mslm/models/registry.py`)
  - 입력은 hidden을 `detach()`한 값이고, 그 위에 router 전용 W_q/W_k를 둔다.
  - 점수는 γ × head 평균 cosine. γ는 14에서 시작하고 상한은 100. W_k는 W_q와 같은 값으로 초기화한다.
  - descriptor는 router key를 meanpool하거나 subblock8 최댓값을 쓴다.
  - gate는 sep_softmax(backbone descriptor)이고 loss는 span_copy_ce.
  - 테스트 454개 통과. aux CE gradient는 router 파라미터에만 가고, LM gradient는 router에 가지 않는다.
- config: `configs/mc_ce_detach_50m.env`, `configs/mc_ce_detach_maxsim_50m.env`
- runs(50M, seed 0, MAX_STEPS 5700, ROUTE_MONITOR_FILE은 E5)

| run | router | CE 가중치 |
|---|---|---|
| `mc_detach_w1_50m_ctx8192_s0` | detach, meanpool | 1.0 |
| `mc_detach_w03_50m_ctx8192_s0` | detach, meanpool | 0.3 |
| `mc_detach_maxsim_w1_50m_ctx8192_s0` | detach, subblock8 max | 1.0 |
| `mc_detach_maxsim_w03_50m_ctx8192_s0` | detach, subblock8 max | 0.3 |

- 1차 시도: job 1217399(node1445, MICRO_BATCH=1, accum 64) + tail 1217400.
  - 사용자가 "너무 느린거아냐? GPU 더 써도 되는데"라고 해서 약 15분 만에 취소했다.
- 2차 시도: job 1217603 + tail 1217604(`afterany`, 3h). **MICRO_BATCH=4, accum 16**. step당 64 시퀀스는 그대로다.

**결과 (raw)**
- 1차(MICRO_BATCH=1), step 20

| run | loss | aux_loss | tok/s | mem |
|---|---|---|---|---|
| detach_w1 | 9.418 | 1.91621 | 32,947 | 11.6 GB |
| detach_w03 | 9.4179 | 1.91615 | 33,051 | 11.6 GB |
| detach_maxsim_w1 | 9.41809 | 1.73346 | 32,554 | 11.71 GB |
| detach_maxsim_w03 | 9.41793 | 1.73359 | 32,637 | 11.71 GB |
| (참고) 1a ce_w01 step 20 | 9.45059 | 2.03818 | 36,929 | 10.19 GB |

- 2차(MICRO_BATCH=4), step 20

| run | loss | aux_loss | tok/s | mem |
|---|---|---|---|---|
| detach_w1 | 9.41801 | 2.35101 | 48,510 | 41.14 GB |
| detach_w03 | 9.4179 | 2.35086 | 48,497 | 41.14 GB |
| detach_maxsim_w1 | 9.41811 | 2.17728 | 47,702 | 41.51 GB |
| detach_maxsim_w03 | 9.4179 | 2.17737 | 47,820 | 41.51 GB |

- 예상 시간: 3B 토큰에 약 17h. 완료는 11/6 05시쯤으로 봤다(1차 조건이었다면 약 25h).
- 4개 run 모두 step 5700에서 `end` 이벤트로 정상 종료했다. tail 1217604는 이어 받을 step이 없어 바로 끝났다(`end`가 두 번 기록됨).
- 최종 요약(11-10에 train.jsonl에서 읽음). "최근 3회"는 5000·5500·5700 step 평균이다. 비교용으로 E6 run을 같은 스크립트로 다시 읽어 함께 적었다.

| run | val_loss 5700 | ruler_hit 5700 | ruler_hit 최근 3회 | copy_hit 5700 | copy_random_hit |
|---|---|---|---|---|---|
| (E6) mon | 2.9518 | 0.1121 | 0.1101 | 0.5978 | 0.3717 |
| (E6) ce_w003 | 2.9530 | 0.1706 | 0.1564 | 0.7243 | 0.3717 |
| (E6) ce_w01 | 2.9960 | 0.2431 | 0.2126 | 0.7609 | 0.3717 |
| (E6) ce_w03 | 3.1012 | 0.2044 | 0.1872 | 0.7694 | 0.3717 |
| detach_w1 | 2.9501 | 0.2718 | 0.2556 | 0.7754 | 0.4014 |
| detach_w03 | 2.9498 | 0.3085 | 0.2887 | 0.7852 | 0.4014 |
| detach_maxsim_w1 | 2.9470 | 0.3800 | 0.3664 | 0.8547 | 0.4014 |
| detach_maxsim_w03 | 2.9478 | 0.3889 | 0.3628 | 0.8484 | 0.4014 |

  - ruler_random_hit는 모든 run에서 0.1368, ruler_items 64, ROUTE_MONITOR_FILE도 같다(`.../route_monitor/TinyLlama__TinyLlama_v1.1/8192/niah_multikey_1/validation.jsonl`).
  - **copy_hit는 E6와 직접 비교할 수 없다.** copy_positions가 E6 326개, E7 1240개다. MICRO_BATCH=4라 val 묶음에 시퀀스가 더 들어간 것으로 보인다(확인 안 함). 그래서 copy_random_hit도 0.3717과 0.4014로 다르다.
  - 마지막 학습 기록의 aux_loss: detach_w1 1.540, detach_w03 1.542, maxsim_w1 1.201, maxsim_w03 1.198.
- 궤적. 열은 step:val_loss/ruler_hit/copy_hit.

```
detach_w1         500:4.051/0.184/0.737 1000:3.550/0.255/0.715 1500:3.334/0.236/0.741 2000:3.222/0.188/0.746 2500:3.149/0.181/0.748 3000:3.102/0.204/0.777 3500:3.061/0.201/0.764 4000:3.026/0.241/0.777 4500:3.003/0.235/0.774 5000:2.977/0.259/0.776 5500:2.957/0.236/0.772 5700:2.950/0.272/0.775
detach_w03        500:4.055/0.171/0.733 1000:3.548/0.228/0.718 1500:3.336/0.220/0.748 2000:3.223/0.214/0.756 2500:3.147/0.228/0.762 3000:3.099/0.249/0.776 3500:3.059/0.251/0.766 4000:3.026/0.281/0.793 4500:3.001/0.280/0.774 5000:2.975/0.274/0.787 5500:2.956/0.284/0.776 5700:2.950/0.308/0.785
detach_maxsim_w1  500:4.052/0.264/0.803 1000:3.546/0.226/0.809 1500:3.332/0.324/0.823 2000:3.219/0.246/0.831 2500:3.143/0.349/0.835 3000:3.093/0.315/0.841 3500:3.055/0.375/0.843 4000:3.023/0.444/0.854 4500:2.996/0.341/0.850 5000:2.973/0.368/0.852 5500:2.952/0.351/0.849 5700:2.947/0.380/0.855
detach_maxsim_w03 500:4.054/0.222/0.796 1000:3.547/0.251/0.813 1500:3.333/0.276/0.829 2000:3.220/0.299/0.822 2500:3.148/0.284/0.830 3000:3.097/0.293/0.832 3500:3.058/0.402/0.835 4000:3.023/0.420/0.846 4500:2.998/0.423/0.840 5000:2.974/0.336/0.842 5500:2.954/0.363/0.844 5700:2.948/0.389/0.848
(E6) mon          500:4.102/0.110/0.507 1000:3.585/0.108/0.563 1500:3.363/0.103/0.564 2000:3.237/0.105/0.564 2500:3.162/0.110/0.574 3000:3.107/0.113/0.585 3500:3.062/0.107/0.586 4000:3.034/0.118/0.595 4500:3.005/0.114/0.601 5000:2.984/0.102/0.599 5500:2.956/0.116/0.601 5700:2.952/0.112/0.598
```

- step 5700 층별 ruler_hit(L0..L15)

```
detach_w1         0.29 0.41 0.29 0.17 0.36 0.27 0.25 0.21 0.24 0.24 0.17 0.14 0.32 0.19 0.38 0.41
detach_w03        0.22 0.30 0.40 0.49 0.32 0.27 0.25 0.30 0.33 0.22 0.21 0.22 0.30 0.33 0.43 0.33
detach_maxsim_w1  0.62 0.21 0.22 0.33 0.32 0.30 0.16 0.65 0.62 0.40 0.16 0.16 0.24 0.48 0.59 0.64
detach_maxsim_w03 0.27 0.13 0.19 0.29 0.25 0.27 0.33 0.46 0.33 0.56 0.35 0.25 0.24 0.73 0.79 0.78
```

**결과 (해석 (11-10, 에이전트 해석))**
- **판정 기준을 모두 넘었다.**
  - val loss: 네 run 모두 2.947~2.950으로 mon(2.952)과 같거나 약간 낮다. 1a의 trade-off(w01 +0.044, w03 +0.149)가 사라졌다. 가설 (3), 즉 CE gradient가 backbone과 경쟁한다는 가설을 지지한다.
  - NIAH ruler_hit(최근 3회): detach 25.6%/28.9%, maxsim 36.6%/36.3%. 모두 w01의 21.3%보다 높다.
  - E6의 통과 기준(무작위 +15pp, 약 28.7%)으로 보면 maxsim 두 run(+23pp)과 detach_w03(+15.2pp)이 넘는다.
- **descriptor(④)가 hit를 크게 바꾼다.** 같은 detach 조건에서 meanpool → subblock8 max로 바꾸면 최근 3회 기준 +7~11pp다. 궤적에서도 step 1500 이후 거의 항상 maxsim이 위에 있다. 학습 조건에서 ④ 효과를 보인 첫 결과다.
- CE 가중치 1.0과 0.3은 차이가 작다. detach에서는 LM 쪽 손해가 없으니 가중치는 2차 변수다.
- maxsim은 깊은 층(L13~L15, w03에서 0.73~0.79)과 L7~L9에서 hit가 높다. meanpool detach는 층별 편차가 작다.
- **주의할 점**
  - seed 1개, 64문항이다. 문항 하나가 1.6pp이고 이웃 eval 사이에서도 ±5~10pp 흔들린다(maxsim_w1 4000:0.444 → 4500:0.341).
  - ruler_hit는 teacher-forced 라우팅 적중률이다. NIAH 정답 생성(string_match)은 재지 않았다.
  - mon 대비 바뀐 것이 두 가지다(router detach와 sep_softmax gate). sep_softmax만 단독으로 켠 run(1b)은 없으므로, gate 분리의 기여는 따로 분리되지 않았다.
  - copy_hit는 E6와 평가 위치 수가 달라 비교하지 않는다.

**결과 (해석 (당시, 실행 직후))**
- CE 가중치(1.0과 0.3)와 상관없이 LM loss가 소수 넷째 자리까지 같다. GPU에서도 detach가 의도대로 동작한다는 첫 확인이다.
- aux CE 시작값이 1a보다 낮다(1.92/1.73 vs 2.04). 초기 cosine 순위 효과이고 성능 근거는 아니다.
- MICRO_BATCH=4의 aux 수치가 다른 것은 묶음 평균 방식 때문이다. 학습 자체는 같다.
- 1차 속도 저하의 원인: MICRO_BATCH=1에서 GPU 활용이 약 2%(약 15 TFLOPS)였다. 앞으로 50M pack은 MICRO_BATCH=4를 기본으로 권했다.
  - 1a도 MICRO_BATCH=1이었다. 4였다면 13~17h에 끝났을 것으로 추정했다.
- 노드를 늘리려면 다중 노드 pack 실행기가 필요하다. t512a03 계정의 5칸이 꽉 차서(다른 분의 vj* 작업 3개) 미뤘다.

**경로**
- run: `/scratch/t512a03/mslm/runs/mc_detach{,_maxsim}_{w1,w03}_50m_ctx8192_s0/`
  - 1차 취소분: `/scratch/t512a03/mslm/runs/_cancelled_1217399/`
- pack 계획 파일: `/scratch/t512a03/mslm/jobs/packs/stage1c_20261105_114415.tsv`, `stage1c_tail_20261105_114415.tsv`
  - 1차: `stage1c_20261105_112214.tsv`, `stage1c_tail_20261105_112215.tsv`
  - dry-run: `stage1c_20261105_103442.tsv`
- 로그: `/scratch/t512a03/mslm/slurm_logs/mslm-pack-stage1c_1217603.out`(1차는 `_1217399.out`)
- wandb: `ks000225/linear-memory-routing`, runs `mc_detach_*`
- 코드: PR #14(`detached_cos` router, ideas `detach` / `detach_maxsim`)

---

### E8. Stage 2 round 1: router descriptor 비교 (2026-11-10 ~ 11-11)

**목적**
- focus module을 ④ descriptor로 확정했다(사용자 "④로 확정", 11-10). 근거는 다음 두 가지다.
  - E7: 같은 detach 조건에서 meanpool → subblock8 max로 바꾸면 NIAH hit +7~11pp.
  - 0025 bilinear probe(370M frozen): probe hit는 높지만 query를 섞어도 거의 같다(real − shuffled-q 0~0.09). descriptor에 query에 따른 구분 정보가 적다.
- E7 틀(detach router, sep_softmax, CE 0.3)을 고정하고 router가 쓰는 descriptor만 바꿔, 어떤 요약이 NIAH hit를 가장 많이 올리는지 본다.
- seed 1 재현 2개로 seed noise 크기를 잰다.
- 판정: ruler_hit 최근 3회 평균. subblock8(E7 36.3%)보다 seed noise 이상 높으면 round 2(seed 0·1·2 + string_match)로 올린다.

**세팅**
- 코드: PR #17
  - `MC_ROUTER_DESCRIPTOR`: 전용 key를 쓰는 라우터의 descriptor만 따로 고른다. 게이트용 backbone descriptor는 `MC_DESCRIPTOR=meanpool` 그대로다.
  - `MC_ROUTER_DESCRIPTOR=subblock8`은 E7 detach_maxsim과 같은 모델이다(테스트로 확인).
  - 학습형 descriptor(mma, keyconv)는 meanpool과 같게 초기화하고, CE gradient만 받는다.
- config: `configs/mc_ce_detach_desc_50m.env`(idea `detach_desc`). 공통: 50M, ctx 8192, MAX_STEPS 5700, MICRO_BATCH 4, ROUTE_MONITOR_FILE은 E5.

| pack | run | descriptor | 겨냥하는 실패 |
|---|---|---|---|
| stage2a | `mc_detach_w03_50m_ctx8192_s1` | meanpool (E7 detach_w03, seed 1) | 재현 |
| stage2a | `mc_detach_maxsim_w03_50m_ctx8192_s1` | subblock8 (E7 maxsim_w03, seed 1) | 재현 |
| stage2a | `mc_desc_sb16_50m_ctx8192_s0` | subblock16 (16토큰 단위) | 희석 |
| stage2a | `mc_desc_sb32_50m_ctx8192_s0` | subblock32 (8토큰 단위) | 희석 |
| stage2b | `mc_desc_boundary_50m_ctx8192_s0` | segment 마지막 토큰 key (G11) | 정보를 한 점에 모음 |
| stage2b | `mc_desc_centered_50m_ctx8192_s0` | meanpool − 이전 segment 평균 | 공통/위치 성분(Gram 0.958~0.9998) |
| stage2b | `mc_desc_mma_50m_ctx8192_s0` | max/min/avg → linear (SeerAttention) | 희석 |
| stage2b | `mc_desc_keyconv_50m_ctx8192_s0` | causal depthwise conv(4) 후 meanpool (FlashMoBA) | 주변 신호 모으기 |

- job: 1249901 `mslm-pack-stage2a`(node1764), 1249902 `mslm-pack-stage2b`(node1722), 각 24h. 11-10 14:09 KST 제출, 약 15:20 학습 시작. 계정 5칸이 차서 tail 작업은 내지 않았다.
- 속도: run당 45~50k tok/s(keyconv가 가장 느림, 45.7k). 예상 종료 11-11 08시쯤.

**결과 (raw)**
- `scripts/harvest.py --traj`, 11-11 09:2x KST. ruler3 = 5000·5500·5700 step ruler_hit 평균. ruler_random_hit 0.1368, copy_random_hit 0.4014. keyconv는 다른 run보다 느려 11-11 10:1x에 끝난 뒤 다시 읽었다(층별 값은 5500 step 것).

| run | val 5700 | ruler 5700 | ruler3 | copy 5700 | aux(마지막) |
|---|---|---|---|---|---|
| (E7) meanpool s0 `mc_detach_w03_50m_ctx8192_s0` | 2.9498 | 0.3085 | 0.2887 | 0.7852 | 1.542 |
| meanpool s1 `mc_detach_w03_50m_ctx8192_s1` | 2.9405 | 0.3085 | 0.3078 | 0.7681 | 1.577 |
| (E7) subblock8 s0 `mc_detach_maxsim_w03_50m_ctx8192_s0` | 2.9478 | 0.3889 | 0.3628 | 0.8484 | 1.198 |
| subblock8 s1 `mc_detach_maxsim_w03_50m_ctx8192_s1` | 2.9365 | 0.3740 | 0.3416 | 0.8474 | 1.271 |
| subblock16 `mc_desc_sb16_50m_ctx8192_s0` | 2.9471 | 0.4931 | 0.4269 | 0.8843 | 1.013 |
| subblock32 `mc_desc_sb32_50m_ctx8192_s0` | 2.9418 | 0.5268 | 0.4673 | 0.8993 | 0.823 |
| boundary `mc_desc_boundary_50m_ctx8192_s0` | 2.9582 | 0.1488 | 0.1412 | 0.5675 | 2.337 |
| centered `mc_desc_centered_50m_ctx8192_s0` | 2.9495 | 0.2470 | 0.2335 | 0.7948 | 1.596 |
| mma `mc_desc_mma_50m_ctx8192_s0` | 2.9481 | 0.3433 | 0.2742 | 0.7717 | 1.442 |
| keyconv `mc_desc_keyconv_50m_ctx8192_s0` | 2.9430 | 0.2679 | 0.2321 | 0.7753 | 1.536 |

- step 5700 층별 ruler_hit(L0..L15)

```
meanpool s1   0.29 0.32 0.29 0.32 0.35 0.32 0.24 0.35 0.30 0.32 0.32 0.25 0.25 0.29 0.38 0.36
subblock8 s1  0.19 0.32 0.17 0.38 0.48 0.24 0.25 0.21 0.25 0.51 0.48 0.43 0.38 0.49 0.68 0.52
subblock16    0.22 0.27 0.19 0.30 0.30 0.41 0.44 0.49 0.59 0.71 0.48 0.56 0.56 0.64 0.86 0.87
subblock32    0.51 0.35 0.52 0.43 0.59 0.43 0.40 0.67 0.71 0.52 0.51 0.32 0.33 0.65 0.73 0.76
boundary      0.14 0.13 0.14 0.19 0.14 0.16 0.13 0.14 0.08 0.17 0.14 0.17 0.13 0.13 0.16 0.22
centered      0.24 0.22 0.25 0.17 0.30 0.29 0.30 0.44 0.27 0.22 0.16 0.14 0.13 0.21 0.27 0.33
mma           0.25 0.16 0.35 0.35 0.21 0.36 0.35 0.30 0.48 0.33 0.41 0.40 0.51 0.27 0.41 0.35
keyconv(5500) 0.11 0.17 0.24 0.24 0.21 0.24 0.21 0.19 0.27 0.14 0.21 0.24 0.25 0.21 0.30 0.32
```

- 궤적(step:val/ruler/copy)은 wandb와 `harvest.py --traj`에 있다. 주요 값: subblock32 ruler 0.202(500) → 0.459(2000) → 0.588(4500) → 0.527(5700). boundary는 전 구간 0.12~0.17.

**결과 (해석)** (11-11, QUEUE.md Q1 판정 규칙 적용)
- seed noise: meanpool |0.2887−0.3078| = 0.019, subblock8 |0.3628−0.3416| = 0.021 → 평균 0.020. 기준 = subblock8 seed 평균 0.352 + max(0.020, 0.03) = **0.382**.
- 통과: **subblock32 0.467(+11.5pp), subblock16 0.427(+7.5pp)** → Q3로. 하위 블록이 잘수록(32 → 16 → 8토큰) hit가 단조 증가하고 copy_hit도 같이 오른다(0.848 → 0.884 → 0.899). 포화 여부를 보려고 4토큰 단위 subblock64를 Q3에 추가했다(PR #25).
- 탈락: mma 0.274, centered 0.234, keyconv 0.232, boundary 0.141(무작위 수준). 학습형 요약(mma, keyconv)은 meanpool(0.298)보다도 낮아 Q7(NSA MLP)의 조건을 만족하지 못한다 → Q7 건너뜀.
- val loss는 모든 run이 2.937~2.958로 같은 범위다. descriptor는 LM 손실에 영향이 없다(detach 틀 유지).
- 깊은 층(L13~L15)의 hit가 subblock16에서 0.64~0.87로 가장 높다. Q5(층 공유)의 source 층 후보.
- 주의: seed 1개(subblock16·32), 64문항. 판정은 Q3(seed 3개 + string_match)에서 확정한다.

**경로**
- run: `/scratch/t512a03/mslm/runs/<run>/`
- pack 계획 파일: `/scratch/t512a03/mslm/jobs/packs/stage2a_20261110_140922.tsv`, `stage2b_20261110_140923.tsv`
- 로그: `/scratch/t512a03/mslm/slurm_logs/mslm-pack-stage2{a,b}_<jobid>.out`, run별 `/scratch/t512a03/mslm/jobs/packs/<run>_<jobid>.log`
- 코드: PR #17

---

### E9. Q2: gate 분리 기여 + seed 2 재현 (2026-11-10 ~ 11-11)

**목적**
- E7 주의점: mon과 detach run은 router detach와 sep_softmax gate 두 가지가 함께 달랐다. gate만 원래대로(`MC_GATE=softmax`, router 점수를 gate로 재사용) 돌려 sep_softmax의 기여를 분리한다. router 입력은 계속 detach라 backbone은 CE를 받지 않는다(LM gradient는 router 파라미터까지 간다).
- 같은 pack에 Q3에 필요한 seed 2 재현(subblock8, meanpool)을 넣어 노드 GPU 4장을 채웠다.

**세팅**
- 50M, ctx 8192, MAX_STEPS 5700, MICRO_BATCH 4, ROUTE_MONITOR_FILE은 E5, CE 0.3. 새 코드 없음(PR #17까지). 로컬 tiny 모델로 detached_cos + softmax gate의 forward/backward를 확인했다.

| run | 바꾼 것 |
|---|---|
| `mc_detach_maxsim_softgate_50m_ctx8192_s0` | detach_maxsim + `MC_GATE=softmax` |
| `mc_detach_maxsim_softgate_50m_ctx8192_s1` | 위와 같음, seed 1 |
| `mc_detach_maxsim_w03_50m_ctx8192_s2` | E7 detach_maxsim_w03, seed 2 |
| `mc_detach_w03_50m_ctx8192_s2` | E7 detach_w03, seed 2 |

- job: 1251866 `mslm-pack-stage2c`(node1766, 24h), 11-10 17:31 KST 제출, 약 21:10 학습 시작. run당 48~49k tok/s.

**결과 (raw)**
- `scripts/harvest.py --traj`, 11-11 13:5x KST. ruler3 = 5000·5500·5700 step 평균. ruler_random_hit 0.1368.

| run | val 5700 | ruler 5700 | ruler3 | copy 5700 | aux(마지막) |
|---|---|---|---|---|---|
| softgate s0 `mc_detach_maxsim_softgate_50m_ctx8192_s0` | 2.9958 | 0.3889 | 0.3373 | 0.8529 | 1.262 |
| softgate s1 `mc_detach_maxsim_softgate_50m_ctx8192_s1` | 2.9941 | 0.5050 | 0.5129 | 0.8459 | 1.361 |
| subblock8 s2 `mc_detach_maxsim_w03_50m_ctx8192_s2` | 2.9353 | 0.2589 | 0.2854 | 0.8597 | 1.136 |
| meanpool s2 `mc_detach_w03_50m_ctx8192_s2` | 2.9406 | 0.2589 | 0.2493 | 0.7706 | 1.497 |

- 비교용 seed 3개 요약(sep_softmax gate)
  - subblock8 ruler3: s0 0.3628, s1 0.3416, s2 0.2854 → 평균 0.330, 표본 표준편차 0.040. val 2.948 / 2.937 / 2.935.
  - meanpool ruler3: s0 0.2887, s1 0.3078, s2 0.2493 → 평균 0.282, 표준편차 0.030. val 2.950 / 2.941 / 2.941.
- softgate val loss 궤적은 처음부터 E7보다 높다: 500 step +0.07(4.120 vs 4.054), 2000 +0.07, 5700 +0.05(2.996 vs 2.948).
- step 5700 층별 ruler_hit(L0..L15)

```
softgate s0    0.35 0.46 0.43 0.46 0.49 0.30 0.22 0.25 0.19 0.40 0.14 0.25 0.35 0.40 0.70 0.82
softgate s1    0.44 0.44 0.52 0.33 0.48 0.56 0.51 0.57 0.64 0.68 0.51 0.51 0.40 0.25 0.65 0.59
subblock8 s2   0.11 0.30 0.13 0.19 0.11 0.17 0.11 0.19 0.14 0.24 0.25 0.14 0.43 0.33 0.68 0.60
meanpool s2    0.24 0.14 0.16 0.13 0.13 0.21 0.32 0.17 0.16 0.22 0.22 0.29 0.44 0.33 0.56 0.43
```

**결과 (해석)** (11-11, 에이전트 해석)
- **sep_softmax의 역할은 LM 손실 보호다.** gate만 softmax(router 점수 재사용)로 바꾸면 val loss가 +0.05 나빠진다(seed 2개 모두 2.994~2.996). backbone은 계속 detach라, LM gradient가 router 파라미터로 가면서 router가 LM 쪽으로 끌려가는 효과로 보인다.
- **hit는 softgate가 평균으로 더 높지만 불안정하다.** 0.337 / 0.513(평균 0.425, seed 차 18pp). sep_softmax subblock8 3 seed 평균 0.330보다 높지만, seed 2개뿐이고 퍼짐이 커서 확정할 수 없다. 1a(E6)처럼 hit와 LM 손실이 다시 맞바뀌는 구조다.
- E7의 "detach가 trade-off를 없앴다"는 결론은 **detach + sep_softmax 조합**에 대한 것이다. 기본 틀은 sep_softmax로 유지한다(QUEUE 규칙: 기록만, 다음 항목 막지 않음).
- **seed noise가 E8 추정(2pp)보다 크다.** subblock8 3 seed 표준편차 4.0pp, meanpool 3.0pp. Q3 판정은 seed 3개 평균과 이 표준편차로 한다. 그래도 subblock32(E8 seed 0, 0.467)는 subblock8 평균보다 3σ 이상 높다.

**경로**
- run: `/scratch/t512a03/mslm/runs/<run>/`
- pack 계획 파일: `/scratch/t512a03/mslm/jobs/packs/stage2c_20261110_173146.tsv`
- 로그: `/scratch/t512a03/mslm/slurm_logs/mslm-pack-stage2c_1251866.out`, run별 `/scratch/t512a03/mslm/jobs/packs/<run>_1251866.log`

---

### E10. Q3 round 2: subblock32·16 seed 반복 + subblock64 (2026-11-11 ~ 11-13)

**목적**
- E8 통과 후보(subblock32, subblock16)를 seed 3개로 확인하고 NIAH `string_match`로 최종 best_desc를 정한다.
- 하위 블록 크기가 작을수록 hit가 올라가는 추세가 4토큰 단위(subblock64)에서도 이어지는지 본다.

**세팅**
- E8과 같은 틀(`configs/mc_ce_detach_desc_50m.env`, CE 0.3, MICRO_BATCH 4, MAX_STEPS 5700).
- stage3a(job 1256568, 11-11 09:28 KST 제출): `mc_desc_sb32_50m_ctx8192_s{1,2}`, `mc_desc_sb16_50m_ctx8192_s{1,2}`
- stage3b(job 1256768, 11-11 10:12 KST 제출): `mc_desc_sb64_50m_ctx8192_s{0,1,2}` (`detach_desc:RUN_TAG=desc_sb64,MC_ROUTER_DESCRIPTOR=subblock64[,SEED=n]`)
- subblock8 seed 2는 E9 pack에서 학습 중이다.
- 학습이 끝나면 평가 pack(`PACK_JOB=eval`, `EVAL_TASKS=niah_single_2,niah_multikey_1,niah_multikey_2,niah_multivalue EVAL_LENGTHS=8192`). 평가 문항은 `/scratch/t512a03/mslm/data/eval_ruler/.../8192/`에 이미 있다.

**결과 (raw)**
- 학습: `scripts/harvest.py`, 11-13 KST. ruler3 = 5000·5500·5700 step ruler_hit 평균, ruler_random_hit 0.1368. s0는 E8(subblock16·32) 또는 이번 stage3b(subblock64) run이다.

| descriptor | ruler3 s0 / s1 / s2 | 평균 | 표준편차 | val 5700 s0 / s1 / s2 | copy 5700 s0 / s1 / s2 |
|---|---|---|---|---|---|
| subblock8 (E7·E8·E9) | 0.3628 / 0.3416 / 0.2854 | 0.330 | 0.040 | 2.9478 / 2.9365 / 2.9353 | 0.848 / 0.847 / 0.860 |
| subblock16 | 0.4269 / 0.3810 / 0.3224 | 0.377 | 0.052 | 2.9471 / 2.9350 / 2.9316 | 0.884 / 0.885 / 0.885 |
| subblock32 | 0.4673 / 0.4190 / 0.4431 | 0.443 | 0.024 | 2.9418 / 2.9336 / 2.9306 | 0.899 / 0.904 / 0.897 |
| subblock64 | 0.5125 / 0.4547 / 0.4798 | 0.482 | 0.029 | 2.9413 / 2.9335 / 2.9329 | 0.914 / 0.908 / 0.915 |

- 평가 evalA(job 1260946, hg3): RULER string_match_all(%), 8192 토큰, 과제당 100문항, step 5700 ckpt. learned / oracle 라우팅.

| run | niah_single_2 | niah_multikey_1 | niah_multikey_2 | niah_multivalue |
|---|---|---|---|---|
| mon `mc_mon_50m_ctx8192_s0` | 1.0 / 1.0 | 2.0 / 1.0 | 0.0 / 0.0 | 0.0 / 4.0 |
| meanpool `mc_detach_w03_50m_ctx8192_s0` | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.8 / 0.8 |
| subblock8 `mc_detach_maxsim_w03_50m_ctx8192_s0` | 0.0 / 0.0 | 3.0 / 3.0 | 0.0 / 0.0 | 0.8 / 1.0 |
| subblock32 `mc_desc_sb32_50m_ctx8192_s0` | 1.0 / 0.0 | 1.0 / 2.0 | 0.0 / 0.0 | 0.0 / 0.2 |

- evalB(ce_w01, subblock16, softgate, mma)는 hg2 보류로 취소한 뒤 아직 내지 않았다.

**결과 (해석)** (11-13, 에이전트 해석)
- **router hit는 하위 블록이 작을수록 계속 오른다.** 8 → 16 → 32 → 64(32 → 16 → 8 → 4토큰): 33.0 → 37.7 → 44.3 → 48.2%. subblock64가 subblock32보다 +3.9pp로, 두 표준편차(2.4, 2.9pp)보다 크다. QUEUE Q3 규칙(ruler_hit 기준)으로 **best_desc = subblock64**. 아직 포화하지 않았다.
- val loss는 모두 2.93~2.95로 같다. copy_hit도 하위 블록이 작을수록 오른다(0.85 → 0.91).
- **string_match는 50M·3B 토큰에서 변별력이 없다.** 모든 run이 0~4%이고 oracle 라우팅(정답 구간 강제)도 learned와 같은 수준이다. 라우팅이 아니라 생성 능력 자체가 병목이다. Q3 규칙의 "string_match 평균" 항목은 이 규모에서 판정에 쓸 수 없다 → 사용자 확인 필요.

**경로**
- pack 계획 파일: `/scratch/t512a03/mslm/jobs/packs/stage3a_20261111_092839.tsv`, `stage3b_20261111_101232.tsv`
- 로그: `/scratch/t512a03/mslm/slurm_logs/mslm-pack-stage3{a_1256568,b_1256768}.out`, 평가 `mslm-pack-evalA_1260946.out`
- 평가 결과: `/scratch/t512a03/mslm/runs/<run>/eval/step_0005700/{summary.json,items.jsonl}`

---

### E부록. 계획만 있고 실행하지 않은 것, 비용 추정

- **계획만 있고 실행하지 않은 것**
  - 1b(sepgate, maxsim × 가중치 0.1/0.03): 사용자 결정으로 보류
  - 370M stage 2(설정 1~5, 30B 토큰, run당 약 360 GPU-h 추정)
  - 설정 6(MARCH식 dense_chained)
  - share, maxsim_share_nolabel, chained, chunk64
  - 50M seed 1·2
  - E4 seed 0 10B 마무리와 RULER 최종 평가
- **결과가 없는 로그 목록**
  - 11-05에서 로컬 `/home/sohyung/linear-memory-routing/logs/`에 다음 파일 276개가 있는 것만 확인했다: `sh_axis2-*`, `sh_density-*`, `sh_descprobe-*`, `sh_frozen-mk*`, `sh_qa1-eval`, `sh_qcroute-c{32,64,128}`, `sh_routing-probe{,2,3}`, `sh_sal2-*`, `sh_salience128-*`.
  - 이 세션은 그 내용을 읽거나 논의하지 않았다. 다른 세션의 실험이거나 legacy 실험으로 보인다.
  - legacy 보고서 0001~0026은 제목만 확인했다. 예: 0018 Axis-1 density, 0019 Axis-2, 0020 salience, 0024 routing 병목.
- **처리량과 비용 추정(세션 계산)**
  - 50M을 노드 1개로 돌리면 vanilla 224k tok/s, mc 171k tok/s다.
  - 3B 토큰 기준 비용은 vanilla 약 15 GPU-h, mc 약 20 GPU-h, maxsim 약 30 GPU-h(추정)다.
  - 연구실 예산은 30,000 GPU-h(11/22까지)이고, 실제 병목은 계정당 작업 5칸이다.

