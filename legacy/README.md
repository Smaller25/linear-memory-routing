# legacy — 지난 실험 기록과 코드 (읽기 전용)

2026-10-26에 저장소를 "multi-state linear attention 사전학습·평가" 구조로 다시 짰다.
그 전 내용은 지우지 않고 모두 `git mv` 로 여기에 옮겼다(파일 history 보존: `git log --follow <path>`).

> 여기 코드는 경로가 바뀌어서 **그대로는 실행되지 않는다**. 옛 상태 그대로 돌려 보려면
> 이동 직전 커밋 `42b57ce9` 를 따로 checkout 한다: `git worktree add ../lmr-old 42b57ce9`

## 무엇이 어디로 갔나

| 옛 위치 | 새 위치 | 내용 |
|---|---|---|
| `fla/` | `legacy/fla_fork/fla/` | 이 저장소가 원래 들고 있던 flash-linear-attention 0.5.2 사본(GDN-2 포함). 루트에 있으면 pip 로 고정 설치한 fla@4b02d15d 를 가려서 옮겼다 |
| `tests/`, `benchmarks/`, `utils/`, `examples/`, `images/`, `.github/` | `legacy/fla_fork/…` | fla 업스트림의 테스트·벤치마크·CI(워크플로가 더 이상 돌지 않음) |
| `setup.py`, `pyproject.toml`, `README.md`, `ENVs.md`, `FAQs.md`, `AUTHORS`, `CITATION.cff`, `CONTRIBUTING.md`, `.flake8`, `.pre-commit-config.yaml` | `legacy/fla_fork/` | fla 업스트림 패키징·문서 |
| `legacy/training/` (fla 의 옛 flame 학습 코드) | `legacy/fla_fork/upstream_legacy_training/` | 업스트림이 원래 legacy 로 둔 것 |
| `lmr/` | `legacy/experiments/lmr/` | 지난 실험 코드: MC 어댑터(Mamba2/GDN), MoSC·DynMC(학습형 경계), 라우팅 분석(`analysis/`), MQAR·passkey 학습, RULER 예측 스크립트 |
| `report/` | `legacy/experiments/report/` | 실험 보고서 0001–0026(아래 표), 중간 보고서, 랩미팅 그림 |
| `notes/` | `legacy/experiments/notes/` | 확정 사실, 실험 일지, descriptor 가설, 경계 연구 메모 |
| `docs/superpowers/` | `legacy/experiments/docs/` | 0024–0026 설계(spec)·계획(plan) 문서 |
| `evals/` | `legacy/experiments/evals/` | fla 계열 lm-eval harness·PPL 스크립트 |
| `scripts/` | `legacy/experiments/scripts/` | 옛 SLURM/VESSL 실행 스크립트, mamba 기반 RULER 하네스(`ruler.py`) |
| `HANDOFF_TO_CODEX.md`, `SESSION_HANDOFF.md` | `legacy/experiments/` | 세션 인계 메모 |
| `src/ruler/` | `third_party/ruler/` (**계속 사용**) | NVIDIA RULER 생성기·채점 함수 사본(Apache-2.0) |

## 지난 실험에서 배운 것 → 새 구조에 반영한 것

| 교훈 | 새 구조 |
|---|---|
| 학습 문제와 평가 문제가 같은 파일이었다 | 학습은 FineWeb-Edu `.bin` shard, 평가는 RULER `jsonl`(Paul Graham essay·노이즈 haystack). 폴더도 분리하고 `ruler_prepare` 가 겹치면 멈춘다 |
| teacher-forced 채점이 공식 채점과 크게 달랐다 | 평가는 RULER 공식대로 `input + answer_prefix` 를 넣고 greedy 생성 → `string_match_all` 만 쓴다 |
| 옛 RULER 하네스는 answer_prefix 를 붙이지 않았다 | RULER `call_api.py` 와 같이 붙인다(`EVAL_ANSWER_PREFIX=1`) |
| tokenizer 가 맞지 않았다 | 학습·평가 모두 `TOKENIZER` 하나. 체크포인트에 tokenizer 이름을 저장하고 다르면 평가가 멈춘다 |
| 대조군·시드·짝 비교가 없었다 | vanilla config 를 같이 두고, 평가 문항을 고정 시드로 만들어 문항 단위 결과(`items.jsonl`)를 남긴다. `mslm.eval.compare` 가 시드 평균 후 짝지은 비교를 한다 |
| fla 버전에 따라 `use_exp2` 오류 | fla 를 commit `4b02d15d` 로 고정 설치(`setup/install_env.sh`), 재설치 전 `site-packages/fla` 삭제 |

## report 번호별 한 줄 요약

| 번호 | 날짜 | 한 줄 요약 |
|---|---|---|
| 0001 | 06-15 | FLA Mamba2 위에 Memory Caching 뼈대를 세움 |
| 0002 | 06-15 | A100 검증 + 학습 없는 MC-RM(Phase 0): RM 이 성능을 망가뜨림 |
| 0003 | 06-15 | GRM 게이트를 학습하면 RM 이 망친 것을 되찾음 |
| 0004 | 06-15 | 더 긴 문맥 GRM + 메모리 가벼운 장문 평가 |
| 0005 | 06-17 | SSC 라우팅이 8k 에서 처음으로 vanilla 를 넘음 |
| 0006 | 06-17 | 두 번째 backbone(Gated DeltaNet) + MoM/AoM/low-rank/계층 결합 방식 추가 |
| 0007 | 06-17 | 모델이 커질수록 SSC 의 이득이 커짐 |
| 0008 | 06-17 | 결합 방식 비교: 장문으로 일반화되는 것은 hard top-k(SSC)뿐 |
| 0009 | 06-17 | SSC top-k sweep + RULER 확인 |
| 0010 | 06-17 | 다중 key 회상에서는 SSC 이득이 사라짐 |
| 0011 | 06-18 | 캐시 크기를 제한하면 SSC 이득도 사라짐(상수 메모리가 아님) |
| 0012 | 06-22 | from-scratch Dynamic-MoSC: 학습형 경계가 다중 key 회상을 풂 |
| 0013 | 06-22 | 학습된 구간 길이 분포: head 가 사실마다 구간을 나눔 |
| 0014 | 06-22 | 고정 mamba2-370m 의 RULER 자유 생성 평가 |
| 0015 | 06-22 | 불규칙 MQAR: 고정 간격은 실패, 학습형 경계 ≈ oracle |
| 0016 | 06-28 | 비지도 경계 학습은 세 방식 모두 실패, 지도 신호가 필요 |
| 0017 | 06-28 | Selective Copying 지도 학습: read-out 이득, 경계 복원, 적응성엔 예산 필요 |
| 0018 | 06-28 | 정보 밀도 기반 구간 나누기: 신호는 있으나 약해서 메커니즘이 못 됨 |
| 0019 | 06-28 | 계층 재압축: 손실 병합은 정확 회상에서 버리기보다 못함 |
| 0020 | 07-09 | 중요도 게이트 보존: oracle 은 되지만 자기지도 신호는 안 되고, from-scratch 는 잡음이 큼 |
| 0021 | 07-09 | 실제 고정 모델에서도 다중 key 실패 확인, 실패 위치 국소화 |
| 0022 | 07-15 | 비상관 쓰기(아이디어 A)가 다중 key 회상을 돕는가 — 장난감 실험 |
| 0023 | 07-15 | 비상관 쓰기는 decay 있는 실제 GDN-2 로 옮겨지지 않음 |
| 0024 | 07-26 | MC-SSC(mean-pool) GDN2-370M 의 multi-NIAH 실패: 라우팅이 주된 병목 |
| 0025 | 07-28 | 라우터 진단: 라우터는 key 판별기가 아니라 needle 검출기에 가까움(X4 무작위 라우팅 대조 포함) |
| 0026 | 07-30 | 현재 상태 정리 + SRLA(이중선형 라우터) go/no-go CPU probe |
| 0024–0026 통합 | 07-30 | 0024–0026 통합 보고서: 실패 분석과 SRLA 후속 설계 |
