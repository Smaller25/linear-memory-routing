# MC-SSC GDN2 Multi-NIAH 실패 원인 분석 — 설계 spec

- **날짜**: 2026-07-25
- **브랜치**: `sh/mc-niah-analysis` (base: `gdn-base-and-mechanisms`)
- **코드 위치**: `lmr/analysis/260725_mc_niah_analysis/`

## 1. 목적

Memory Caching(mean-pool descriptor) SSC를 얹어 end-to-end 학습한 GDN-2 370M이
single-NIAH 대비 multi-NIAH(MK-NIAH-1)에서 크게 실패하는 원인을,
**write → read → route → 생성** 4단계 인과 사슬로 분해해 지목한다.
결론은 후속 memory-routing 알고리즘 개선의 근거가 된다.

배경 수치 (collaborator 성적표, 5B ckpt, RULER 표준 프로토콜 @2048):

| 태스크 | vanilla | MC v2 |
|---|---|---|
| S-NIAH-1 | 88 | 60 |
| MK-NIAH-1 | 20 | 2 |

## 2. 대상 모델 (3개)

| 이름 | HF repo / 파일 | config |
|---|---|---|
| MC-25B | `LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool` / `checkpoint-25B-model-ckpt.pth` | `mc_370M` |
| MC-5B | 같은 repo / `checkpoint-5B-model-ckpt.pth` | `mc_370M` |
| Vanilla-5B | `LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla` / `checkpoint-5B-model-ckpt.pth` | `gdn2_370M` |

- MC 설정: topk=2, mc_chunk_size=256, descriptor = **mean-pool of L2-normalized keys**
  (commit `4ef4942` 이후 코드), 16 layers, tokenizer `TinyLlama/TinyLlama_v1.1`.
- Vanilla는 single-state라 chunk 분석 불가 — **E0 성능 앵커 전용**.
- 모델 정의 코드: `gyunggyung/long-gdn` **origin/main @ `e71713e`** (mean-pool 학습 코드와 일치)를
  별도 **git worktree**로 꺼내 PYTHONPATH import. 본 checkout은 건드리지 않고 커밋 해시를
  재현성 기록으로 남긴다. (vendoring은 커널 복제 부담 + 원본과 어긋날 위험으로 기각)

## 3. 데이터 (길이 2048 고정 = 8 segments × 256)

- **Dataset A (앵커, RULER 표준)**: vendored NVIDIA RULER(`src/ruler`)로
  `niah_single_1`, `niah_multikey_1` @2048 각 **50 샘플**.
  free generation + `string_match_all` — 성적표와 동일 프로토콜.
- **Dataset B (paired-controlled)**: 같은 needle(key·magic number)·같은 depth·같은 haystack로
  single/multi 쌍 **32쌍**. multi 쪽 distractor 배치를 통제:
  - **조건 S (16쌍)**: distractor key를 gold needle과 같은 segment 안에 강제 삽입
    (segment 내 token 거리 기록)
  - **조건 D (16쌍)**: distractor 전부 다른 segment에 삽입 (gold segment는 needle 1개만)
- S/D 통제는 E1 조건별 집계·E2·E3 전용. E0 앵커 비교는 Dataset A만 사용 (오염 방지).

조건별 가설 분리:

| 조건 | routing 난이도 | state 난이도 | 실패 시 결론 |
|---|---|---|---|
| D | 높음 (descriptor가 chunk 특정해야) | 낮음 (binding 깨끗) | routing 문제 (mean-pool 식별력) |
| S | 낮음 (그 segment가 key 전부 보유) | 높음 (한 state에 binding 2개) | state 간섭 / read 구분 실패 |

## 4. 실험

### E0 — 앵커 재현
3 모델 × {single_1, multikey_1} @2048, Dataset A, RULER 표준 스코어.
성적표(MC 60/2, vanilla 88/20)와 대략 일치 확인 — 이후 분석의 전제 검증.

### E1 — Routing 정확도 (질문 1: 정답 chunk를 찾는가)
MC 모델만. `forward_with_diagnostics`의 `route_indices`/`route_scores`를
answer position에서 layer별(16개) 수집:
- `hit@topk`: 정답 chunk ∈ 선택된 top-2 비율
- `gold rank`: route_scores 내 정답 chunk 순위 (top1_all)
- multi 전용 `top1_amongkeys`: key 보유 chunk들 중 질의된 key의 chunk가 1등인가 (chance=1/K)
- 샘플별 routing hit ↔ 최종 정답 여부 상관 (routing이 맞아도 틀리는가?)
- Dataset A로 전체 집계 + Dataset B로 S/D 조건별 집계

### E2 — Oracle routing 개입 (인과 주실험)
SSC forward를 subclass해 `forced_indices` 옵션 추가: **전 layer에서 정답 chunk를
top-k에 강제 주입** 후 재생성 → 점수 회복량 측정 (Dataset B, S/D별).
- 회복 O → routing이 병목 (D에서만 회복되면 개선 알고리즘의 적용 범위도 정량화됨)
- 회복 X → 병목은 downstream (E3가 write/read 중 어디인지 지목)
- anti-oracle은 **하지 않는다** (사용자 결정).

### E3 — Write/Read fidelity 프로파일 (질문 2: state decoding 실패인가)
생성 없이 prompt 1-pass forward만 사용. 각 layer의 실제 투영값
(`MemoryCachingGDN2Layer._project` 재사용, layer 입력은 forward hook 수집)으로,
읽기 연산의 선형 분해 `r = q·M ≈ (q·k_needle)·v_needle + 간섭항`의 각 인자를 측정:

**b-1 Write fidelity** — needle 정답 토큰 위치 t*, gold segment i*=t*//256:
- `v̂ = norm(k_t*)·M` 을 두 시점에서: ① needle 직후 state (segment prefix만 scan; 보존도 상한),
  ② segment 최종 state M_i* (실제 cache 저장본)
- 지표: cos(v̂, v_t*) per head → layer 평균.
  ①→② 하락 = segment 내 decay/overwrite; single→multi 하락 = needle 간 write 간섭
- 조건 S에서 needle 간 거리·key 유사도별 하락 곡선 (erase 반경)

**b-2 Read fidelity** — answer position T_a의 실제 query q_Ta:
- 정렬도 cos(q_Ta, k_t*) — query가 needle key 방향을 가리키는가
- 읽기 결과 r = q_Ta·M_i* 를 paired single의 r과 cosine 비교
- r에서 (q·k)v_needle 성분 vs 간섭 잔차 크기 비율

state 재구성은 학습과 동일한 `chunk_gdn2` + `scan_segments` 경로 사용 (수치 일치 보장).
대상: MC-25B, MC-5B. 산출: layer(16) × 지표 프로파일, single vs multi 오버레이, S/D 층화.

### 판정표 (최종 결론 프레임)

| b-1 write | b-2 read | E1 route | E2 oracle 회복 | 결론 |
|---|---|---|---|---|
| ✗ | – | – | ✗ | write-time 간섭 — state 용량/erase 문제 |
| ✓ | ✗ | – | ✗ | query가 binding 못 꺼냄 — readout 실패 |
| ✓ | ✓ | ✗ | ✓ | **routing 실패** — descriptor/router 개선으로 해결 가능 |
| ✓ | ✓ | ✓ | – | gate 혼합/online-path 노이즈 문제 |

## 5. 실행 환경

- greenbeard SLURM: `sbatch` (partition `main`, `--gres=gpu:rtx6000:1`, 6h 제한).
  노드 직접 CUDA 실행 금지.
- env `sh_routing` (py3.11, torch 2.9.1+cu128), `HF_HOME=/data2/sohyung/hf_home`,
  `PYTHONPATH` = lmr repo + long-gdn worktree.
- 사전 smoke test: dsc `gdn2_ops` triton 커널의 sm_120 동작 + MC ckpt 로드 + 짧은 forward.

## 6. 산출물

- 코드: `lmr/analysis/260725_mc_niah_analysis/`
  (`data.py`, `load_mc.py`, `eval_niah.py`, `routing_stats.py`, `oracle.py`, `fidelity.py`, sbatch 러너)
- 결과: `logs/mc_niah/` JSON + 그림
- 보고서: `report/` 관례에 따라 번호 붙인 md 1편 (판정표 기반 결론)

## 7. 리스크 및 완화

| 리스크 | 완화 |
|---|---|
| gdn2_ops 커널 sm_120 비호환 | 1.3B 평가 성공 전례 있음; smoke test 먼저 |
| MC wrapper `use_cache` 미지원 → 토큰마다 re-prefill로 생성 느림 | 2K·370M·50샘플이면 감당; 초과 시 샘플 축소 |
| `mc_370M` config가 pinned commit에 없거나 이름 상이 | worktree 준비 단계에서 config 존재 검증 |
| E0 재현치가 성적표와 크게 어긋남 | 원인 규명 전 E1+ 진행 보류, 사용자 보고 |

## 8. 범위 제외

- anti-oracle 대조군 (사용자 결정으로 제외)
- multiquery/multivalue/multikey_2·3 — multikey_1 결론 후 후속
- vanilla 모델의 내부 분석 (성능 앵커만)
- 개선 알고리즘 구현 자체 (본 분석의 후속 작업)
