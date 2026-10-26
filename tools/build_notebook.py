"""Regenerate control_tower.ipynb (keeps the notebook free of hand-edited logic).

    python tools/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 관제탑 (control tower)\n\n"
       "`configs/*.env` 를 **그대로** 읽고 쓴다. 셸 스크립트(`scripts/*.sh`)도 같은 파일을 읽으므로 둘이 어긋나지 않는다 "
       "(`tests/test_config_parity.py`, `tests/test_notebook.py` 가 확인).\n\n"
       "- 이 노트북은 **제출하지 않는다**. 만든 sbatch 명령을 복사해 터미널에서 직접 실행한다.\n"
       "- KISTI: 작업 제출은 `/scratch/$USER` 아래에서만 된다(명령에 `cd` 가 들어 있다)."),
    code("import os, sys, json\n"
         "REPO = os.path.abspath(os.getcwd())\n"
         "sys.path.insert(0, REPO)\n"
         "from mslm import tower\n"
         "CONFIG = 'configs/mc_50m.env'   # 바꿀 config\n"
         "OVERRIDES = {}                  # 이번 한 번만 바꿀 값, 예: {'SEED': '1', 'MAX_TOKENS': '1e9'}\n"
         "tower.list_configs()"),
    md("## 1. 변수 보기\n`from` 열: base(기본값) / config(이 파일이 바꿈) / override(OVERRIDES)."),
    code("tower.show(CONFIG, OVERRIDES)"),
    md("## 2. 변수 바꾸기\n"
       "- 파일에 영구 반영: `tower.set_var(CONFIG, 'MAX_TOKENS', '20e9')` (셸 스크립트도 바로 이 값을 쓴다)\n"
       "- 이번 제출에만: 위 `OVERRIDES` 에 넣는다 → 아래 명령에 `KEY=value` 로 붙는다."),
    code("# tower.set_var(CONFIG, 'MAX_TOKENS', '20e9')\n"
         "tower.show(CONFIG, OVERRIDES, section='train')"),
    md("## 3. 아이디어(모듈)별 구현 코드\n"
       "MC 층은 다섯 부품으로 조립된다. 새 아이디어는 `mslm/models/registry.py` 에 `@register(kind, name)` 으로 "
       "등록하고 config 변수(`MC_DESCRIPTOR`, `MC_ROUTER`, `MC_GATE`, `MC_CONNECT`, `MC_AUX_LOSS`)를 그 이름으로 바꾼다."),
    code("print('사용 중:', tower.parts(CONFIG, OVERRIDES))\n"
         "print('등록됨:', tower.available_parts())"),
    code("for kind, name in tower.parts(CONFIG, OVERRIDES).items():\n"
         "    print('=' * 20, kind, '=', name)\n"
         "    print(tower.source_of(kind, name))"),
    md("## 4. 실행 명령 미리 보기 (직접 복사해서 실행)"),
    code("for job in ['prepare_data', 'prepare_eval', 'train', 'eval']:\n"
         "    print(f'# {job}')\n"
         "    print(tower.sbatch_command(CONFIG, job, OVERRIDES))\n"
         "print('# GPU 노드에서 바로(대화형):')\n"
         "print(tower.shell_command(CONFIG, 'train', OVERRIDES))"),
    md("## 5. 연기 시험(smoke)\nCPU: tiny 크기로 몇 step 학습 → 저장·재로드 일치 → oracle 라우팅 확인. 몇 초면 끝난다."),
    code("tower.smoke(CONFIG, {**OVERRIDES, 'MODEL_SIZE': 'tiny', 'MC_CHUNK': '16', 'VOCAB_SIZE': '256'})"),
    md("단일 GPU: **GPU 가 할당된 노드에서만** `RUN_GPU = True` 로 바꿔 실행한다 "
       "(KISTI MyKSC Jupyter 또는 salloc 노드. 연구실 greenbeard 에서는 노드에서 직접 GPU 사용 금지 → sbatch 사용)."),
    code("RUN_GPU = False\n"
         "if RUN_GPU:\n"
         "    print(tower.smoke(CONFIG, OVERRIDES, device='cuda', steps=3, seq_len=2048))"),
    md("## 6. 진행 상황"),
    code("print(tower.run_dir(CONFIG, OVERRIDES))\n"
         "for r in tower.train_log(CONFIG, OVERRIDES, last=10):\n"
         "    print(r)\n"
         "for name, s in tower.eval_summaries(CONFIG, OVERRIDES).items():\n"
         "    print(name, json.dumps(s['results'], ensure_ascii=False)[:2000])"),
]
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"          # deterministic ids -> stable diffs
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).resolve().parents[1] / "control_tower.ipynb"
nbf.write(nb, out)
print("wrote", out)
