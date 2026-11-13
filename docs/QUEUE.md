# 실험 큐 (우선순위순)

자동 진행(orchestrator, [ORCHESTRATOR.md](ORCHESTRATOR.md))은 이 파일을 위에서부터 읽어 처리한다.
상태를 바꿀 때마다 이 파일을 PR로 갱신하고, 결과는 [EXPERIMENTS.md](EXPERIMENTS.md)에 적는다.
짧은 요약판은 Notion [MC router 실험 큐](https://app.notion.com/p/4145fe4de48381639383fe986d2bd18d)에 같이 갱신한다(wandb 링크 포함).

- 상태: `대기` / `제출` / `학습중` / `평가중` / `완료` / `보류(사유)` / `건너뜀(사유)`
- 조건: 이 항목을 내기 전에 만족해야 하는 것. 조건이 결과에 따라 갈리면 판정 규칙을 따른다.
- 공통 틀(별도 표기 없으면): 50M, ctx 8192, `MAX_STEPS=5700`(약 3B 토큰), `MICRO_BATCH=4`, `ROUTE_MONITOR_FILE`=E5 문항,
  router `detached_cos`, gate `sep_softmax`, `span_copy_ce` 가중치 0.3. 모든 run은 wandb(`ks000225/linear-memory-routing`)에 올라간다.
- 비교 지표: round 1 계열은 `ruler_hit` 최근 3회 평균(5000·5500·5700 step). 최종 판정은 NIAH `string_match`.
- **best_desc**: Q1·Q3 결과로 정해지는 최고 router descriptor. 정해지면 아래 표의 `best_desc` 자리에 실제 이름을 적는다.

## 운영 설정

| 항목 | 값 |
|---|---|
| 동시에 쓸 작업 칸 | 계정마다 최대 3: hg3(t512a03)과 hg2(t512a02, 11-11 사용자 허락). 각 계정 5칸은 연구실 공용이라 그 계정에서 다른 분 작업이 4칸 이상이면 새로 내지 않는다 |
| 작업 하나 | 노드 1개(GPU 4장), run 4개(`scripts/run_pack.sh`) |
| 사용자 확인이 필요한 것 | 370M 이상 학습, GPU 예산 25k GPU-h 초과 예상, 새 module 축(아래 표 밖) 추가 |
| 금지 | MQAR 실험, 로그인 노드에서 학습, 비밀번호·API 키를 파일에 쓰기 |

## 큐

| 순위 | ID | 내용 | module | 조건 | run (pack 명세) | 판정 규칙 | 상태 |
|---|---|---|---|---|---|---|---|
| 1 | Q1 (E8) | descriptor 6종 + seed 1 재현 2개 | ④ | — | stage2a: `detach:RUN_TAG=detach_w03,MC_AUX_WEIGHT=0.3,SEED=1` `detach_maxsim:RUN_TAG=detach_maxsim_w03,MC_AUX_WEIGHT=0.3,SEED=1` `detach_desc:RUN_TAG=desc_sb16,MC_ROUTER_DESCRIPTOR=subblock16` `detach_desc:RUN_TAG=desc_sb32,MC_ROUTER_DESCRIPTOR=subblock32` / stage2b: `detach_desc:RUN_TAG=desc_{boundary,centered,mma,keyconv},MC_ROUTER_DESCRIPTOR=<같은 이름>` | noise = seed 0·1 차이의 절댓값 평균(meanpool, subblock8). subblock8 seed 평균 + max(noise, 3pp) 를 넘는 후보를 상위 2개까지 Q3 로. 없으면 best_desc=subblock8 | 완료 (E8). 기준 38.2% → subblock32 46.7%, subblock16 42.7% 통과. best_desc 후보 = subblock32, subblock16 |
| 2 | Q2 | gate 분리 기여 분리(E7 주의점) | ⑨ | Q1 제출 후 칸이 비면 | `detach_maxsim:RUN_TAG=detach_maxsim_softgate,MC_AUX_WEIGHT=0.3,MC_GATE=softmax` (router 점수를 gate로 재사용, backbone은 계속 detach) | val loss와 ruler_hit를 E7 detach_maxsim_w03과 비교. 기록만 하고 다음 항목은 막지 않는다 | 완료 (E9). softgate hit 평균 42.5%(0.337/0.513)·val +0.05 → sep_softmax 유지. 함께 넣은 seed 2: subblock8 0.285, meanpool 0.249. 원래 메모: softgate seed 0·1 + Q3용 subblock8·meanpool seed 2를 같은 pack에) |
| 3 | Q3 | round 2: 상위 후보 seed 반복 + string_match | ④ | Q1 완료 | Q1 상위 ≤2개 × seed 0·1·2 중 없는 것, subblock8 seed 2. 학습 끝난 run마다 평가: `EVAL_TASKS=niah_single_2,niah_multikey_1,niah_multikey_2,niah_multivalue EVAL_LENGTHS=8192 EVAL_NUM_SAMPLES=100`(`scripts/eval.sh`, `slurm/eval.sbatch`). 같은 평가를 E6 mon, E7 4개 run에도 | seed 3개 평균 ruler_hit와 learned string_match 평균이 가장 높은 것을 best_desc 로. 차이가 seed 표준편차보다 작으면 단순한 쪽(subblock 계열 → 적은 파라미터) | 완료 (E10). ruler3 seed 3 평균: sb64 48.2% > sb32 44.3% > sb16 37.7% > sb8 33.0% → **best_desc = subblock64**. string_match는 oracle까지 0~4%라 50M에서 판정 불가(사용자 확인 대기) |
| 4 | Q4 | chunk 크기 | ① | Q3 완료 | `detach_desc:RUN_TAG=desc_best_c64,MC_ROUTER_DESCRIPTOR=best_desc,MC_CHUNK=64` `…_c128,MC_CHUNK=128` (subblock 계열이면 chunk가 m으로 나뉘는지 확인) | ruler_hit가 best_desc(256)보다 3pp 이상 높고 tok/s 저하 ≤30% 면 채택 후보. 결과는 Q9 설정에 반영 | 학습중 (E12, job 1273115 stage4: subblock64 + MC_CHUNK=64 / 128. Q6와 같은 pack) |
| 5 | Q5 | 층 배치: 한 층이 고른 구간을 공유 | ⑪ | Q3 완료 | best_desc 기준 `MC_SHARE_FROM=<Q3 best run에서 ruler_hit 최고 층>`, `MC_SHARE_FROM=0` 두 run | ruler_hit와 val loss가 best_desc 대비 같거나 좋으면 채택 후보(계산 절약) | 대기 |
| 6 | Q6 | top-k | ⑦ | Q3 완료 | best_desc 기준 `MC_TOPK=2`, `MC_TOPK=8` | k별 ruler_hit와 val loss를 기록. 다음 항목은 막지 않는다 | 학습중 (E12, job 1273115 stage4: subblock64 + MC_TOPK=2 / 8) |
| 7 | Q7 | 학습형 compression MLP(NSA) | ④ | Q1에서 mma 또는 keyconv가 subblock8보다 높을 때만 | 새 descriptor `nsa_mlp`(block 안 위치 임베딩 + 2층 MLP, 초기값 meanpool) 구현 → PR(테스트 포함) → 1 run | Q1 규칙과 같음. 조건 불만족이면 `건너뜀` | 건너뜀 (mma 27.4%, keyconv 23.2% < subblock8 35.2%) |
| 8 | Q8 | router 학습 신호: teacher KL / dense warm-up | ⑩ | Q3 완료 | 새 aux_loss 구현(teacher = 같은 batch의 dense GRM γ 또는 oracle 분포) → PR → 2 run | ruler_hit가 best_desc보다 3pp 이상 높으면 채택 후보 | 대기 |
| 9 | Q9 | 370M 확장: 최종 설정 + mon | 전체 | Q3~Q6 완료, **사용자 확인** | `SIZE=370m` 최종 설정과 mc(mon), 각각 2노드 × `TRAIN_PARTS` | string_match(4K·8K, 7과제)로 최종 비교 | 학습중 (E11, 11-13 사용자 확인 "괜찮은 아이디어 370M으로"). 1노드 4 GPU, 10B 토큰, MICRO_BATCH 2. subblock64: job 1272512(part 1). mon(mc_370m RUN_TAG=mon): job 1273114. part 2는 첫 tok/s를 보고 `AFTER=`로 이어 건다. 끝나면 string_match(learned·oracle)로 평가 |

## 변경 기록

- 2026-11-10: 처음 작성. focus module ④ 확정(사용자), E8 제출 후.
- 2026-11-10: Q1 두 pack 모두 학습 시작.
- 2026-11-10: 칸 하나가 비어 Q2 제출. 노드 GPU 4장을 채우려고 Q3에 필요한 seed 2 재현 2개를 같이 넣음.
- 2026-11-10: Q2 학습 시작.
- 2026-11-11: Q1 완료(E8). subblock32·16 통과, Q7 건너뜀. Q3 stage3a 제출, subblock64 추가(PR #25), stage3b 제출.
- 2026-11-11: hg2(t512a02)도 쓰기로 함(사용자). 평가 pack evalA·evalB를 hg2에 제출.
- 2026-11-11: 사용자가 hg2를 다른 작업에 쓰려고 비워 달라고 함 → evalA·evalB(대기 중, 실행 전) 취소. **사용자가 다시 말할 때까지 hg2에는 내지 않는다.** 평가는 hg3 칸이 날 때 낸다.
- 2026-11-13: mon 370M(job 1273114)과 Q4·Q6 50M pack stage4(job 1273115) 제출. Q5는 다음 칸에.
- 2026-11-13: 사용자가 Q9(370M) 진행을 지시. subblock64 370M part 1 제출(job 1272512). Q4~Q6(50M)는 칸이 남을 때.
- 2026-11-13: Q3 완료(E10). best_desc = subblock64(ruler_hit 기준, 추세 미포화). string_match는 50M에서 oracle까지 0~4%라 평가 지표로 쓸 수 없음 → 사용자에게 물음. 다음은 Q4~Q6(best_desc = subblock64).
- 2026-11-11: hg3에 평가 evalA 제출(job 1260946: mon, E7 meanpool·subblock8 s0, subblock32 s0. 8K, niah_single_2·multikey_1·multikey_2·multivalue). evalB(ce_w01, sb16, softgate, mma)는 다음 칸에.
- 2026-11-11: Q2 완료(E9). sep_softmax 유지. subblock8 3 seed 표준편차 4pp(Q3 판정에 사용).
