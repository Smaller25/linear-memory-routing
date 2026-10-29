# linear-memory-routing — multi-state linear attention 사전학습·평가

GDN-2 위에 **Memory Caching(MC)** 을 얹은 언어모델을 처음부터(from scratch) 학습하고,
RULER NIAH 로 공정하게 평가하는 저장소다.

MC 는 `MC_CHUNK`(기본 256) 토큰마다 GDN-2 state 를 하나씩 저장해 두고,
토큰마다 라우터가 고른 top-k 구간 state 를 지금 state 와 게이트로 섞어 읽는다
(논문 *Memory Caching: RNNs with Growing Memory*, arXiv:2602.24281 의 SSC).

지난 실험 코드와 보고서 0001–0026 은 [`legacy/`](legacy/README.md) 에 있다.

---

## 한눈에 보기

```
configs/          실행 변수(.env). base.env 가 모든 변수를 선언하고, 나머지는 바꿀 것만 적는다
scripts/          submit_pipeline.sh(전체 연쇄 제출) · status.sh(진행 확인) · train.sh · eval.sh · smoke.sh · prepare_*.sh · sbatch_cmd.sh · show_config.sh
slurm/            *.sbatch — scripts/*.sh 를 감싼다(직접 sbatch 하지 말고 sbatch_cmd.sh 가 만든 명령 사용)
setup/            kisti_bootstrap.sh(처음 한 번) · environment.yml(conda env 정의) · install_env.sh · check_env.sh · download_assets.sh · scratch_keepalive.sh
mslm/models/      GDN-2(gdn2.py), MC-SSC(mc.py), LM(lm.py), 아이디어 꽂는 곳(registry.py)
mslm/kernels/     GDN-2 Triton 커널(NVIDIA, 비상업 라이선스), 구간 읽기 커널, CPU 참조 구현
mslm/data/        FineWeb-Edu 사전 토큰화, 재개 가능한 샘플러
mslm/eval/        RULER 문항 생성, 공식 생성 채점 + 라우팅 진단, 짝지은 비교
mslm/train.py     학습 루프
control_tower.ipynb  관제 노트북(같은 config 를 읽고 쓰기, 코드 보기, sbatch 명령 만들기, smoke)
third_party/ruler NVIDIA RULER 생성기·채점 함수 사본
tests/            CPU 테스트(설정 동기화, 노트북, 모델, 평가 지표)
legacy/           지난 실험(읽기 전용)
```

## 무엇을 기준으로 삼았나

MC 학습 코드는 이 저장소 밖 세 곳에 있었다. 이렇게 골랐다.

| 가져온 것 | 출처 | 이유 |
|---|---|---|
| GDN-2 커널 `gdn2_ops` (그대로 복사) | `long-gdn/dsc/lit_gpt/gdn2_ops` | 지금까지 370M 학습이 실제로 이 커널 + fla@4b02d15d 로 돌았다. fla@4b02d15d 에는 GDN-2 가 없고, 이 커널은 그 버전의 fla 내부 함수(`use_exp2` 등)에 묶여 있다 |
| GDN-2 층 구조·초기화 | `long-gdn/dsc/lit_gpt/gdn2.py`, `model.py`(mamba_init) | 파라미터 이름까지 같게 다시 썼다(나중에 옛 체크포인트를 읽을 여지를 남김). lit_gpt 의 rotary·KV cache·DSC 분기 등 이 연구에 필요 없는 부분은 버렸다 |
| SSC 수식(meanpool descriptor, `W_u` 라우터, online+top-k 결합 softmax) | `long-gdn/dsc/mc_baseline/mc_ssc.py`, `mc_gdn2/ssc.py` | 논문 식 16–17 을 그대로 따르는, 검증된 구현. 합(sum) 대신 평균(mean)인 이유도 그대로 주석에 남겼다 |
| 구간 읽기 Triton 커널 (그대로 복사) | `long-gdn/dsc/mc_baseline/cached_memory_read.py` → `mslm/kernels/gather_read.py` | `[B,T,k,H,K,V]` 중간 텐서를 만들지 않는 속도 최적화 커널 |
| `chained` 연결 방식 | 워크트리 `lmr-mc-routing-tracks/dsc/mc_gdn2/ssc.py` | 메인 dsc 에는 없는 논문 §3.4 의 다른 방식. 등록부의 한 항목으로 넣었다 |
| 50M 크기 | 같은 워크트리의 `mc_50M` (d=384, 6 head×128, ffn 768, 16층, topk=4) | |
| `long-gdn/linear-memory-routing` | 가져오지 않음 | 이 저장소의 옛 사본이라 새로 가져올 것이 없었다 |

다른 저장소는 읽기만 했고 수정하지 않았다.

## 크기 프리셋 (`MODEL_SIZE`)

| 이름 | 층 | d_model | GDN-2 head × dim | FFN | 파라미터(vanilla / mc) |
|---|---|---|---|---|---|
| `50m` | 16 | 384 | 6 × 128 | 768 | 71.9M / 76.7M (임베딩 제외 59.7M / 64.4M) |
| `370m` | 16 | 1024 | 16 × 128 | 2048 | 380.6M / 414.2M |
| `tiny` | 2 | 64 | 2 × 32 | 128 | CPU 테스트용 |

## 학습 레시피 (기본값, `configs/base.env`)

문맥 `SEQ_LEN=8192`, step 당 `GLOBAL_BATCH_TOKENS=524288`(0.5M), AdamW(β 0.9/0.95, wd 0.1, 1차원 파라미터·`A_log`·`dt_bias` 제외),
LR `4e-4`, cosine(끝 `LR×0.1`), warmup 1%, grad clip 1.0, bf16 autocast.
누적 횟수는 `GLOBAL_BATCH_TOKENS / (SEQ_LEN × MICRO_BATCH × GPU 수)` 로 자동 계산된다.
데이터는 FineWeb-Edu(`sample-10BT` 기본), 토크나이저는 TinyLlama v1.1.

## 평가 원칙

- **공식 생성 채점만**: RULER `call_api.py` 처럼 `input + answer_prefix` 를 넣고 greedy 생성(최대 128 토큰, EOS 에서 멈춤) →
  RULER postprocess → `string_match_all`. teacher-forced 채점은 하지 않는다.
- **같은 토크나이저**: 평가 문항도 `TOKENIZER` 로 길이를 맞춰 만들고, 체크포인트에 저장된 토크나이저와 다르면 멈춘다.
- **데이터 분리**: 학습은 FineWeb-Edu `.bin`, 평가는 Paul Graham essay·노이즈 haystack 으로 만든 `jsonl`. 폴더도 다르다.
- **대조군·시드·짝 비교**: vanilla config 를 같이 돌리고, 시드 0·1·2 로 3번씩. 평가 문항은 `EVAL_DATA_SEED` 로 고정되어
  모든 모델이 같은 문항을 푼다. `items.jsonl` 로 문항 단위 짝 비교(`python -m mslm.eval.compare`).
- **라우팅 진단**(MC): 마지막 프롬프트 토큰에서 층마다 정답 구간(바늘 값이 든 구간)을 골랐는지(hit), 정답 구간 순위,
  그리고 문항마다 초기하분포로 계산한 무작위 기준선 `1 − C(N−G,k)/C(N,k)` 를 같이 남긴다.
- **oracle 라우팅**(`EVAL_ROUTE=oracle`): 정답 구간을 top-k 에 강제로 넣는다. 평가 전용. 라우팅이 병목인지 가늠하는 상한.
- **MQAR 은 쓰지 않는다.**

---

## KISTI 한강에서 처음부터 끝까지

> 확인한 문서: <https://docs-ksc.gitbook.io/hangang-user-guide> (2026-10 기준).
> 아직 직접 확인하지 못한 값은 아래 [확인 필요](#kisti-에서-확인이-필요한-것) 에 모아 두었다.

### 0. 알아 둘 규칙

- 로그인: `ssh <ID>@hangang.ksc.re.kr` (OTP 6자리 → 비밀번호). 다른 로그인 노드로 붙으면 `ssh glogin03` 으로 옮긴다.
  파일 전송·큰 다운로드는 데이터무버 `hangang-dm.ksc.re.kr`(sftp).
- 끊겨도 작업이 살아 있도록 tmux 안에서 일한다: `module load tmux` → `tmux new -s work`(다음부터 `tmux attach -t work`,
  빠져나오기 `Ctrl-b d`). 셸 자동 종료(`TMOUT`)는 `~/.bashrc` 에 `unset TMOUT` 을 넣어 끈다.
- GPU 노드: GH200 4개/노드, **aarch64(ARM)**. 배정 큐 `natStr`, GPU 요청은 `--gres=gpu:N`.
- sbatch 에 `--comment="field=<분야>;appl=<프로그램>"` 가 **필수**. 허용값은 로그인 노드에서 `showappl`. 기본값은 `field=nlp;appl=pytorch`(`SB_COMMENT`).
- **작업은 `/scratch/$USER` 아래에서만 제출된다.** `sbatch_cmd.sh` 가 만든 명령이 알아서 `cd $SB_SUBMIT_DIR` 한다.
- 저장소 위치: 코드·conda env 는 홈(`/home01/$USER`, 128GB, 백업 있음), 데이터·체크포인트·HF 캐시는 `/scratch/$USER/mslm`
  (80TB 그룹, 백업 없음, **15일 동안 접근 안 한 파일 자동 삭제**).
- 계산 노드는 인터넷이 안 된다고 가정한다. 다운로드는 로그인/데이터무버 노드에서, 계산 노드에서는 HF offline 모드.

### 1. 접속 (매번)

```bash
ssh hangang
```

```bash
ssh glogin03
```

```bash
module load tmux && tmux new -s work
```

다음부터는 `tmux attach -t work` 로 다시 붙는다.

### 2. 처음 한 번: 코드 받기 + 부트스트랩 (로그인 노드, 인터넷 필요)

```bash
cd ~ && git clone https://github.com/Smaller25/linear-memory-routing.git && cd linear-memory-routing && bash setup/kisti_bootstrap.sh
```

`setup/kisti_bootstrap.sh` 가 순서대로 한다. 끝난 단계는 건너뛰므로 끊기면 같은 명령을 다시 치면 된다.

| 단계 | 하는 일 |
|---|---|
| 1 | `uname -m` 이 `aarch64` 인지(아니면 멈춤: 그 노드에서 깐 env 는 GPU 노드에서 안 돈다) |
| 2 | `showappl` 에 `SB_COMMENT`(`field=nlp;appl=pytorch`) 값이 있는지(없으면 경고) |
| 3 | conda env `sh_mslm` 설치(`setup/install_env.sh`). 이미 있고 torch·fla 커밋이 맞으면 건너뜀 |
| 4 | 토크나이저·FineWeb-Edu sample-10BT parquet(약 28GB)·nltk 를 `/scratch/$USER/mslm/hf_cache` 로 받기 |
| 5 | CPU 점검(`setup/check_env.sh`) |

conda env 는 `setup/environment.yml` 하나에 정의돼 있다(python 3.11, torch 2.9.1+cu128, triton 3.5.1 등 버전 고정,
aarch64 wheel 존재 확인함). fla 만 `--no-deps` 로 따로 깐다(의존성까지 깔면 PyPI fla-core 가 커밋 코드를 덮어쓴다).
손으로 만들 때:

```bash
conda env create -f setup/environment.yml && conda activate sh_mslm && pip install --no-deps "git+https://github.com/fla-org/flash-linear-attention@4b02d15d6a68700181b180235be62a9fb95d2a38"
```

### 3. 제출: 준비 → smoke → 본학습

```bash
cd ~/linear-memory-routing && MAIN=1 bash scripts/submit_pipeline.sh
```

```
prepare_data (사전 토큰화) ──→ smoke (작업 1개: GPU 점검 → 20 step → 재개해서 30 step → 평가) ──→ 본학습 vanilla/MC × 시드 0,1,2
prepare_eval (본평가 문항, 독립)
```

- 앞 작업이 실패하면 뒤 작업은 같이 취소된다(`--kill-on-invalid-dep`). smoke 가 통과해야 본학습이 시작된다.
- **같은 명령을 다시 치면 이어서 진행한다.** 끝난 단계(학습 데이터 `meta.json`, 평가 문항, smoke 의 `SMOKE_OK`, 끝난 run)와
  큐에 이미 있는 작업은 건너뛰고, 시간 초과로 끊긴 run 은 다시 내서 마지막 체크포인트부터 잇는다.
- 사용자당 제출 수 한도에 걸리면 거기서 멈추고 남은 것을 알려 준다. 앞 작업이 끝난 뒤 같은 명령을 다시 친다.
- 무엇을 낼지 먼저 보려면 `DRY_RUN=1` 을 붙인다. smoke 만 하려면 `MAIN=1` 을 뺀다.
- 낸 작업 목록은 `/scratch/$USER/mslm/jobs/pipeline_<시각>.txt`, 로그는 `/scratch/$USER/mslm/slurm_logs/`.

### 4. 진행 확인

```bash
bash scripts/status.sh
```

내 작업 큐, run 별 마지막 step·loss·val loss·평가 여부, 데이터 준비 상태, 최근 에러 로그를 한 화면에 보여 준다.

### 5. 한 단계씩 직접 낼 때

`scripts/sbatch_cmd.sh <job>` 은 config 로 sbatch 명령을 만들어 **출력만** 한다. `| bash` 를 붙이면 바로 제출된다.
job 은 `prepare_data`, `prepare_eval`, `selftest`, `smoke`, `train`, `eval`.

```bash
CONFIG=configs/mc_50m.env SEED=1 bash scripts/sbatch_cmd.sh train
```

GPU 노드를 직접 잡아 점검할 때는 `/scratch` 에서 salloc 한다.

```bash
cd /scratch/$USER && salloc --partition=natStr --nodes=1 --gres=gpu:1 --time=00:30:00 --comment="field=nlp;appl=pytorch"
```

### 6. 평가와 비교

평가는 GPU 1장이면 된다.

```bash
for seed in 0 1 2; do CONFIG=configs/mc_50m.env SEED=$seed SB_GPU_OPT=--gres=gpu:1 SB_EXCLUSIVE=0 bash scripts/sbatch_cmd.sh eval; done
```

vanilla 와 MC(learned)를 문항 단위로 짝지어 비교(시드 평균):

```bash
python -m mslm.eval.compare --a /scratch/$USER/mslm/runs/vanilla_50m_ctx8192_s{0,1,2}/eval/step_0019074 --b /scratch/$USER/mslm/runs/mc_50m_ctx8192_s{0,1,2}/eval/step_0019074 --route-a none --route-b learned
```

(`step_…` 는 실제 마지막 체크포인트 이름으로 바꾼다.)

### 7. /scratch 15일 삭제 대비

- 접근 시각 갱신: `CONFIG=configs/mc_50m.env bash setup/scratch_keepalive.sh`
- 홈으로 마지막 체크포인트 백업: `BACKUP_ROOT` 를 `/home01/$USER/mslm_backup` 등으로 정하면 학습이 저장할 때마다 마지막 1개를 복사한다.
  수동으로는 `BACKUP=1` 을 붙여 keepalive 스크립트를 실행한다. 홈 용량(128GB)에 주의.
- 연구실 서버로 가져오기: `sftp <ID>@hangang-dm.ksc.re.kr` 후 `get -r /scratch/<ID>/mslm/runs/<run>/ckpt`.
- 삭제 대상은 이름 앞에 `ToBeDelete_` 가 붙는다(20–30일 뒤 일괄 삭제). keepalive 스크립트가 보이면 경고한다.

---

## 변수 바꾸는 법

- 영구: `configs/<이름>.env` 에서 `KEY=value  # 설명` 줄을 고친다(노트북 `tower.set_var` 도 같은 파일을 고친다).
- 이번 한 번: 명령 앞에 붙인다. `CONFIG=configs/mc_50m.env SEED=1 MAX_TOKENS=1e9 bash scripts/train.sh`
- 우선순위: 명령 앞 변수 > config 파일 > `base.env`. `base.env` 순서대로 계산되므로 `SCRATCH_ROOT` 를 바꾸면
  `DATA_ROOT`·`RUNS_ROOT` 등도 따라 바뀐다.
- 최종 값 확인: `CONFIG=configs/mc_50m.env SEED=1 bash scripts/show_config.sh`
- 새 변수를 만들 때는 반드시 `base.env` 에 한 줄 주석과 함께 선언한다(테스트가 검사).

## 아이디어를 꽂는 법

MC 층은 다섯 부품으로 조립된다. `mslm/models/registry.py` 에 함수/클래스를 `@register(kind, name)` 으로 등록하고
config 변수를 그 이름으로 바꾸면 끝이다. 다른 코드는 손대지 않는다.

| 부품 | 변수 | 기본값 | 예: 후보 아이디어 |
|---|---|---|---|
| descriptor (구간 요약) | `MC_DESCRIPTOR` | `meanpool` | max-sim, span 요약 |
| router (점수) | `MC_ROUTER` | `dot` | 라우팅 공유, bilinear |
| gate (섞기) | `MC_GATE` | `softmax` | 게이트 분리 |
| connect (구간 연결) | `MC_CONNECT` | `independent` | `chained`(이미 있음), 구간 길이는 `MC_CHUNK`(예: 64) |
| aux_loss (라우터 보조 손실) | `MC_AUX_LOSS`, `MC_AUX_WEIGHT` | `none` | 라우팅 CE(`gold_ce` 예시 있음: `batch_meta` 에 정답 구간이 있을 때) |

노트북의 "아이디어별 구현 코드" 셀이 지금 config 가 쓰는 부품의 소스를 그대로 보여 준다.

## 관제 노트북

`control_tower.ipynb` — 저장소 루트에서 연다. config 표 보기·고치기, 부품 소스 보기, sbatch 명령 미리 보기, CPU smoke,
(GPU 노드에서) 단일 GPU smoke, 학습 로그·평가 요약 보기. **제출은 하지 않는다.**
노트북은 `tools/build_notebook.py` 로 만들며, `tests/test_notebook.py` 가 노트북과 셸이 같은 명령을 만드는지 검사한다.

## 테스트

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q tests
```

## KISTI 에서 확인이 필요한 것

- `TODO(확인 필요)` `natStr` 큐의 최대 시간(`SB_TIME=24:00:00` 이 거부되면 줄인다), `--account` 같은 과금 옵션이 필요한지.
- `SB_COMMENT` 의 `field=nlp;appl=pytorch` 가 `showappl` 목록에 맞는지(다중 GPU 면 `appl=pytorch-ddp` 일 수도).
- 계산 노드 인터넷 여부(문서에 명시 없음). 막혀 있다고 가정했다.
- 사용자당 제출·실행 작업 수 한도. 한도에 걸려도 `submit_pipeline.sh` 를 다시 치면 남은 것을 낸다.
- 비독점 1-GPU 할당(`--gres=gpu:1` 에 `--exclusive` 없이)이 허용되는지. 안 되면 smoke·평가도 4 GPU 노드를 통째로 쓴다.
- pip torch 2.9.1+cu128 aarch64 wheel 이 GH200 드라이버에서 실제로 도는지(wheel 존재만 확인). 안 되면 `TORCH_CUDA=cu129`, 그다음
  KISTI 컨테이너(`/apps/common/kisti-container/images/pytorch-25.03-py3-aarch64.sif`) 안에서 fla@4b02d15d 만 설치하는 방법.
- 다중 노드 학습: pip torch 에 들어 있는 NCCL 이 Slingshot(aws-ofi-nccl 플러그인)과 맞는지. KISTI 안내대로 단일 노드는
  `NCCL_NET_PLUGIN=none`, 다중 노드는 `NCCL_PROTO=^LL128` 을 `slurm/train.sbatch` 에 넣어 두었다. 다중 노드 전에 2노드 짧은 시험 필수.
- 홈 경로·용량: 문서마다 `/home01` 128GB(유저) / 64GB(그룹)로 다르다. `lfs quota -h /home01` 로 확인.
- 데이터무버에서 conda/huggingface_hub 를 쓸 수 있는지.

## 라이선스

- `mslm/kernels/gdn2_ops/`: NVIDIA Source Code License-NC(비상업 연구용). 같은 폴더의 `LICENSE` 참고.
- `third_party/ruler/`: Apache-2.0 (NVIDIA RULER).
- 그 밖의 코드: 저장소 `LICENSE`(MIT).
