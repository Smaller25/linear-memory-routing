# linear-memory-routing 현재 작업 상태

작성일: 2026-07-30  
기준 브랜치: `sh/mc-niah-analysis`

## 1. 한눈에 보는 현재 상태

현재 프로젝트는 **0025 router 진단 실험을 마무리하고, 0026 SRLA 학습 실험을 시작하기 직전인 상태**다.

- 0024 분석: 완료
- 0025 X1/X2 분석: 완료
- 0025 X4 random-routing ablation: 완료
- 0026 SRLA 코드: 구현 및 테스트 완료
- 0026 GPU 학습: 아직 실행하지 않음
- 0026 backward gate: 실행 결과 확인 필요

현재 브랜치의 최신 커밋은 `1f98e708`이며, 0024·0025·0026의 진행 상황과 재개 방법을 정리한 인수인계 문서다.

## 2. 0025의 최신 결과: X4 random-routing ablation

X4의 목적은 MC 모델의 장문 성능 향상이 다음 중 무엇 때문인지 구분하는 것이었다.

1. 실제로 올바른 segment를 선택하는 routing 품질
2. 각 segment의 write 부담이 256토큰으로 제한되는 구조적 효과

이를 위해 기존 `stock` routing과 `recent`, `random` routing을 비교했다. 설정은 다음과 같다.

- 모델: `mc-5B`, `mc-30B`
- 태스크: `niah_single_1`, `niah_multikey_1`
- context length: 1K, 2K, 4K, 8K, 16K, 32K
- `chunk=256`, `topk=2`
- random seed: 0–4
- 셀당 샘플 수: 50

대표적인 32K 결과는 다음과 같다.

| 모델/태스크 | stock | random |
|---|---:|---:|
| mc-5B single-needle | 58% | 0% |
| mc-30B single-needle | 18% | 0% |

1K부터 32K까지 전반적으로 `stock ≫ random` 패턴이 반복되었다. 따라서 MC의 장문 single-needle 이득은 segment 분할만으로 설명되지 않으며, **학습된 routing이 성능에 실제로 기여한다**고 판정했다.

반면 `multikey` 태스크는 장문에서 stock routing 자체도 성능이 낮았다. 즉, 여러 key를 처리하는 문제에는 routing 외에 descriptor 또는 supervision 관련 병목이 남아 있다.

상세 결과는 [0025 한국어 보고서](0025_ko.md)의 X4 절에 있다.

## 3. 0025에서 확인된 실패와 남은 가설

다음과 같은 저비용 수정은 효과가 없거나 성능을 악화시켰다.

- router query를 `u_t := q_t`로 바꾸기
- descriptor centering
- top-1 principal component 제거
- 단순한 anisotropy 보정

따라서 현재 남은 주요 해결 방향은 재학습 기반의 두 가지다.

### 3.1 Contrastive router supervision

현재 LM loss만으로는 올바른 chunk를 직접 선택하도록 router를 충분히 학습시키기 어렵다. 정답 chunk 정보를 routing loss로 직접 제공해 `W_q`와 `W_desc`가 query–descriptor 대응을 학습하도록 하는 방향이다.

### 3.2 GDN2 read geometry에 맞춘 descriptor

현재 descriptor는 raw key의 평균을 사용한다. 그러나 GDN2의 state read는 단순히 특정 raw key 방향을 읽는 것이 아니라 이후 key들의 영향까지 반영한 방향일 가능성이 있다. 따라서 backbone이 실제로 정보를 읽는 기하에 맞춰 descriptor를 다시 구성하는 방향이 남아 있다.

## 4. 0026 SRLA의 현재 상태

0026 SRLA는 학습 가능한 retrieval router와 descriptor를 추가하는 실험이다.

구현은 별도 브랜치 `sh/srla`의 worktree에 준비되어 있으며, 인수인계 문서 기준으로 다음 항목이 구현되어 있다.

- descriptor 모듈
- router 모듈
- cache 모듈
- fusion wrapper
- toy backbone 및 backbone loader
- 학습 스크립트
- 평가 스크립트
- TTFT benchmark
- backward gate
- SLURM 실행 스크립트

테스트는 지정된 SRLA 테스트 묶음 기준으로 `215 passed, 1 skipped` 상태다. 다만 GPU에서 실제 학습을 시작하기 전 반드시 backward gate를 통과시켜야 한다.

## 5. 학습 전에 확인해야 할 backward gate

backward gate는 두 가지를 확인한다.

### Gate A: routing loss gradient

정답 chunk cross-entropy가 `W_q`와 `W_desc`에 유한하고 0이 아닌 gradient를 전달하는지 확인한다.

### Gate B: LM loss backward

LM loss가 `chunk_gdn2` backward를 통과하는지 확인한다. 로컬 GPU 환경에서는 forward/eval만 검증된 상태이므로, 학습 전에 별도 확인이 필요하다.

gate 결과에 따른 분기는 다음과 같다.

- 통과: primary SRLA 학습 실행 가능
- Gate B 실패: `--no-lm-loss --detach-bank` 축소 설정 검토
- Gate B 실패 시: LM-loss 학습은 A100/H200급 환경이 필요하다고 명시해야 하며, hybrid descriptor arm도 학습 가능 여부를 다시 판단해야 함

학습은 gate 결과를 확인하고 사용자 승인을 받은 뒤 실행하도록 되어 있다.

## 6. CPU bilinear probe

GPU 예산을 사용하기 전에, 학습된 bilinear form이 기존 `M=I` 방식보다 나아질 가능성이 있는지 CPU에서 확인하는 probe가 준비되어 있다.

파일:

- [probe_bilinear_router.py](../lmr/analysis/260725_mc_niah_analysis/probe_bilinear_router.py)

이 probe는 rank 제한이 있는 bilinear router를 학습하고, 다음 기준과 비교한다.

- `M=I` 기준선
- 기존 stock scorer
- chance
- 위치 prior
- query를 섞은 `shuffled_q` 대조군

결과 JSON은 인수인계 문서에서 언급된 `/data2/sohyung/mc_niah/results/` 아래에 생성된 것으로 기록되어 있으나, 현재 저장소에는 결과 해석과 커밋 여부를 확인하는 작업이 남아 있다.

## 7. 현재 작업 트리와 미완료 작업

현재 `sh/mc-niah-analysis` 브랜치에는 다음 미커밋 변경이 있다.

- `report/0025_ko.md`: X4 완료 결과 반영
- `lmr/analysis/260725_mc_niah_analysis/probe_bilinear_router.py`: CPU bilinear probe 신규 파일
- `.claude/`: 로컬 작업 관련 신규 파일

다음 작업 순서는 다음과 같다.

1. CPU bilinear probe 결과 JSON을 읽고 held-out 성능과 대조군을 비교한다.
2. 로컬 SLURM backward gate의 상태와 결과를 확인한다.
3. `fla` 버전과 경로가 올바른지 확인한다.
4. gate 통과 후에만 SRLA primary 학습을 사용자 승인 하에 제출한다.
5. 학습 결과는 `trained`와 `untrained-identity`를 핵심 비교로 평가한다.

## 8. 주의사항

- `sh/srla` 브랜치는 아직 push되지 않았다.
- SRLA 학습은 backward gate 결과를 보기 전 실행하면 안 된다.
- 비교 시 핵심은 `trained` 대 `untrained-identity`이며, 단순 random router와의 비교가 아니다.
- 로컬 GPU 작업은 로그인 노드에서 직접 CUDA를 실행하지 말고 SLURM을 사용해야 한다.
- 대용량 결과와 checkpoint는 `/data2/sohyung/` 아래에 저장해야 한다.
- `fla`는 pinned kernel과 호환되는 특정 커밋을 사용해야 하며, 단순 버전 문자열만으로 판단하면 안 된다.

## 결론

0025의 최신 실험은 **MC의 single-needle 장문 성능이 실제 routing 품질에 의존한다**는 점을 확인했다. 이제 연구의 중심은 단순한 후처리 보정이 아니라, 정답 chunk supervision과 GDN2 read geometry를 반영한 descriptor를 학습하는 것으로 이동했다.

0026 SRLA 구현은 준비되어 있지만, 아직 실제 학습 결과는 없다. 따라서 현재 가장 중요한 다음 단계는 **backward gate 실행 및 결과 판정**이다.
