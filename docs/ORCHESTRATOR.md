# 실험 자동 진행 절차 (orchestrator)

[QUEUE.md](QUEUE.md)의 항목을 우선순위대로 처리하는 Claude Code 세션의 절차다. 세션은 `/loop`(동적 간격)로 이 문서의
"한 번의 tick"을 반복한다. 사람이 단계마다 지시하지 않아도, 작업 칸이 비면 다음 항목을 내고 끝난 결과를 기록한다.

## 원칙

- **기록이 먼저다.** run이 끝나면 다음 항목을 내기 전에 [EXPERIMENTS.md](EXPERIMENTS.md)(목적 / 세팅 / 결과 raw / 해석 / 경로)와
  QUEUE.md 상태를 PR로 갱신하고 병합한다. 숫자는 `scripts/harvest.py` 출력에서 그대로 옮긴다.
- **wandb.** 학습 run은 `scripts/train.sh`가, 평가 결과는 `scripts/eval.sh`가 wandb(`WANDB_PROJECT`)로 올린다.
  run을 기록할 때 `harvest.py`의 `wandb=synced`를 확인하고, 아니면 로그인 노드에서
  `python -m mslm.wandb_sync <run_dir>`로 다시 올린다(키는 `wandb login`으로만, 파일에 쓰지 않는다).
- **판정은 QUEUE.md의 규칙대로.** 규칙이 가리지 못하는 경우(조건이 애매함, 결과가 기대와 반대, 실패가 반복됨)는
  추측해서 진행하지 않고 사용자에게 묻는다. 새 module 축, 370M 이상, 예산 초과도 사용자 확인 후에만.
- **고칠 코드가 생기면** 브랜치 → 테스트 추가 → 전체 테스트 통과 → PR → 충돌 없으면 병합 → KISTI에서 `git pull --ff-only`.
- 금지: MQAR, 로그인 노드 학습, 다른 연구실 멤버의 tmux 세션 조작, 비밀번호·키 기록, 다른 저장소 수정.

## KISTI 접근

- 계정 두 개를 쓴다: **hg3 = t512a03**(기본, wandb 로그인됨), **hg2 = t512a02**(칸이 없을 때, 11-11 사용자 허락).
  - hg2 작업은 결과를 hg3의 run 폴더에 쓴다. hg2에서 낼 때 항상 앞에 붙인다:
    `RUNS_ROOT=/scratch/t512a03/mslm/runs DATA_ROOT=/scratch/t512a03/mslm/data WANDB_PROJECT=`
    (두 계정이 서로의 /scratch/<계정>/mslm 에 쓸 수 있다. ACL 확인함.)
  - hg2에는 wandb 로그인이 없다. hg2 run은 tick마다 hg3 pane에서
    `python -m mslm.wandb_sync /scratch/t512a03/mslm/runs/<run>` 으로 올린다(이미 올린 step은 건너뜀, 평가 결과도 같이 올라감).
  - hg2 pane에서 명령은 `( unset CONDA_ENVS_PATH; eval "$(conda shell.bash hook)"; cd ~/linear-memory-routing && … )`.
    시스템 python3는 오래돼서 conda hook 없이는 `run_pack.sh`가 실패한다. `git log` 등은 `GIT_PAGER=cat`.

- 제출·`squeue`: t512a03 pane (`tmux -S /tmp/hangang-tmux/sock`, target `sohyung`). Steering_vector_operator 세션과
  함께 쓰므로 마지막 줄이 빈 프롬프트일 때만 입력하고, `clear`·Ctrl-c는 쓰지 않는다. 명령은
  `( unset CONDA_ENVS_PATH; cd ~/linear-memory-routing && … ); echo @@ORC<n>` 형태로 보내고 marker까지 잘라 읽는다.
- 읽기 전용 확인(로그 tail): t512a02 pane. `harvest.py`는 t512a03 홈에 있어 t512a02에서 읽을 수 없으니 t512a03 pane에서 실행한다. t512a02 (`tmux -S /tmp/hangang-t512a02/sock`, target `sohyung`).
  t512a03의 작업은 squeue에서 보이지 않는다.
- run 폴더: `/scratch/t512a03/mslm/runs/`. 모니터 문항: `ROUTE_MONITOR_FILE=/scratch/t512a03/mslm/data/route_monitor/TinyLlama__TinyLlama_v1.1/8192/niah_multikey_1/validation.jsonl`.
- pane에 "Hangang login ... is down"이 보이면 아무것도 하지 않고 사용자에게 알린다(OTP 재로그인 필요).

## 한 번의 tick

1. **상태 읽기.** t512a03에서 `squeue -u t512a03 -o '%.10i %.14j %.9T %.10M %.6D %R'`.
   우리 작업(`mslm-*`)과 다른 분 작업 수를 센다. QUEUE.md에서 상태가 `제출`/`학습중`/`평가중`인 항목의 run을
   t512a03 pane에서 `python3 ~/linear-memory-routing/scripts/harvest.py /scratch/t512a03/mslm/runs '<패턴>'`으로 읽는다.
2. **끝난 것 처리.**
   - 작업이 사라졌는데 run이 `end`가 아니면: 24시간 초과면 같은 `run_pack.sh` 명령을 다시 낸다(체크포인트에서 이어짐).
     그 밖의 실패는 run별 로그(`/scratch/t512a03/mslm/jobs/packs/<run>_<jobid>.log`)를 읽고, 원인이 코드면 고치고,
     같은 run이 두 번 실패하면 사용자에게 묻는다.
   - 모든 run이 `end`면 `harvest.py --traj`로 결과를 읽어 EXPERIMENTS.md에 기록하고, 판정 규칙을 적용해 QUEUE.md를 갱신한다
     (다음 항목의 `best_desc` 같은 자리 채우기 포함). 평가가 필요한 항목은 상태를 `평가중`으로 두고 평가 pack을 낸다.
3. **다음 항목 내기.** 계정마다 남은 칸 = min(3 − 우리 작업 수, 5 − 전체 작업 수). 그 계정에서 다른 분 작업이 4칸 이상이면 0. hg3를 먼저 채우고 모자라면 hg2.
   QUEUE.md를 위에서부터 보며 조건을 만족하는 `대기` 항목을 칸 수만큼 낸다. 한 항목의 run이 4개를 넘으면 pack을 나눈다.
   - 학습: `MICRO_BATCH=4 MAX_STEPS=5700 ROUTE_MONITOR_FILE=… bash scripts/run_pack.sh <pack> <spec> …`
   - 평가: `PACK_JOB=eval SB_TIME=06:00:00 EVAL_TASKS=… EVAL_LENGTHS=8192 bash scripts/run_pack.sh <pack> <같은 spec> …`.
     평가 문항이 없으면 먼저 `CONFIG=configs/mc_50m.env EVAL_LENGTHS=8192 bash scripts/sbatch_cmd.sh prepare_eval`로 만든다
     (여러 평가가 동시에 같은 문항 파일을 만들지 않게).
   - 내기 전에 `DRY_RUN=1`로 run 이름과 덮어쓴 값을 확인한다. 낸 뒤 job id와 plan 파일을 QUEUE.md와 EXPERIMENTS.md(새 E 항목,
     결과는 "아직 없음")에 적고 PR로 병합한다.
4. **Notion 갱신.** QUEUE.md 상태나 결과가 바뀌었으면 Notion 페이지
   [MC router 실험 큐](https://app.notion.com/p/4145fe4de48381639383fe986d2bd18d)도 같이 고친다(Notion MCP `notion-update-page`,
   `update_content`로 바뀐 줄만). 맨 위 "지금" 한 줄, 큐 표의 상태 칸, 결과 표, 변경 기록 한 줄. 새 run은 wandb 링크
   (`https://forge.coreweave.com/wandb/ks000225/linear-memory-routing/runs/<run>`)를 단다. 문장은 짧게, 표 위주로.
5. **알리기.** 항목이 끝났거나 판정이 났거나 사용자 확인이 필요하면 한두 줄로 보고한다. 아무 변화가 없으면 조용히 넘긴다.
6. **다음 깨우기.** 진행 중인 학습이 있으면 다음 평가 지점(500 step, 약 1.5시간)이나 예상 종료에 맞춰 1~3시간 뒤,
   대기열에서 기다리는 중이면 1시간 뒤. 큐가 비고 진행 중인 작업도 없으면 루프를 멈추고 최종 요약을 보고한다.

## 진행 중 판단 기준

- 학습 중간(500 step마다)에는 loss 발산(NaN, val loss가 E7 같은 step보다 0.1 이상 높음)만 본다. 발산이면 그 pack을 취소하고
  사용자에게 알린다. hit는 끝날 때만 판정한다(중간 값은 ±10pp 흔들림).
- 예상 시간: 50M 5700 step pack은 약 17시간(MICRO_BATCH=4). subblock32 같은 큰 descriptor는 더 걸릴 수 있으니
  첫 tok/s를 보고 24시간을 넘을 것 같으면 같은 명령을 `AFTER=<jobid>`로 미리 이어 건다(칸이 있을 때).
