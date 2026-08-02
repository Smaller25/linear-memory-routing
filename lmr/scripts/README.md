# `lmr/scripts/` — 학습 / 평가 진입점

Track A(frozen backbone + read-out head) 실험의 실행 스크립트다. 성격에 따라 두 갈래로 나눠 둔다.

## 구조

```
scripts/
  eval_recall.py       공유 채점 라이브러리 (실행 파일 아님)
  convert_mamba2.py    체크포인트 변환 유틸 (학습·평가 공통 준비 단계)
  train/               학습 진입점
  eval/                평가·추론 진입점
```

## `train/` — 학습

backbone은 얼리고 read-out/router head만 학습한다.

| 스크립트 | 무엇 |
|---|---|
| `train_variant.py` | SSC/GRM/AoM/MoM 변형 head 학습 (MQAR 합성 데이터) |
| `train_grm_passkey.py` | passkey 과제로 head 학습, 다중 세그먼트 |
| `train_mocm_mqar.py` | MoCM 믹서 학습 (legacy — fla 0.5.2에서 학습이 안 된다는 기록이 `report/0012.md`에 있다) |

```bash
python -m lmr.scripts.train.train_variant --variant grm --chunk-size 256 --steps 2000
```

## `eval/` — 평가·추론

학습된 head를 얹어 recall / RULER / passkey를 측정한다.

| 스크립트 | 무엇 |
|---|---|
| `eval_recall`(상위) | 채점 함수 `score` / `score_hidden` / `evaluate` 제공 |
| `eval_long.py` | 장문맥 recall 길이 스윕 |
| `eval_ruler.py` | RULER 과제 |
| `eval_real.py` | 변환된 실제 체크포인트 평가 |
| `eval_text_passkey.py` | 텍스트 passkey |
| `predict_ruler.py` | greedy free generation으로 RULER 예측 |
| `run_baseline.py` | head 없는 vanilla 기준선 |

```bash
python -m lmr.scripts.eval.eval_long --arch gdn --heads ckpt/ssc_heads.pt --variant ssc
```

## 왜 `eval_recall.py`는 `eval/`로 안 옮겼나

실행 스크립트가 아니라 **채점 라이브러리**다. `train/`과 `eval/` 양쪽에서 쓰고, 저장소 밖에서도 참조한다 — `long-gdn/dsc/track_a/scripts/`의 `eval_long.py`와 `run_baseline.py`가 `from lmr.scripts.eval_recall import ...` 로 직접 가져간다. 경로를 바꾸면 그쪽이 깨지므로 `lmr/scripts/eval_recall.py` 위치를 고정한다.

`convert_mamba2.py`도 같은 이유로 상위에 둔다. 학습 전 준비 단계이자 평가에서도 쓰는 변환 유틸이라 한쪽에 넣으면 오해를 부른다.
