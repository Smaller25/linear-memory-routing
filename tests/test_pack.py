"""scripts/run_pack.sh (DRY_RUN) and scripts/pack_train.sh with a stub trainer: one GPU per run,
per-run overrides stay in their own run, finished runs are dropped. Never touches a real queue."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def env_for(tmp_path, **extra):
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    for name in ("squeue", "sbatch"):
        (stub / name).write_text("#!/bin/sh\nexit 0\n")
        (stub / name).chmod(0o755)
    return {"PATH": f"{stub}:{os.environ['PATH']}", "USER": "u", "HOME": str(tmp_path),
            "SCRATCH_ROOT": str(tmp_path / "scratch"), "DRY_RUN": "1",
            "LINES": "50", "COLUMNS": "200", **extra}  # login shells export these


def pack(tmp_path, *args, check=True, **extra):
    return subprocess.run(["bash", "scripts/run_pack.sh", *args], cwd=REPO, env=env_for(tmp_path, **extra),
                          capture_output=True, text=True, check=check)


def test_dry_run_lists_runs_and_one_job(tmp_path):
    out = pack(tmp_path, "s1", "mc:RUN_TAG=mon", "ce:RUN_TAG=ce_w003,MC_AUX_WEIGHT=0.03", MAX_STEPS="5700").stdout
    lines = out.split("[dry] pack s1")[1].splitlines()
    runs = [ln.strip().split("\t") for ln in lines[1:3]]
    assert [r[0] for r in runs] == ["mc_mon_50m_ctx8192_s0", "mc_ce_w003_50m_ctx8192_s0"]
    assert runs[0][1].endswith("configs/mc_50m.env") and runs[1][2] == "RUN_TAG=ce_w003 MC_AUX_WEIGHT=0.03"
    cmd = lines[3]
    assert "slurm/pack.sbatch" in cmd and "--job-name=mslm-pack-s1 " in cmd and "MAX_STEPS=5700" in cmd
    assert "MC_AUX_WEIGHT" not in cmd  # per-run override stays in the plan file


def test_rejects_same_folder_and_too_many_runs(tmp_path):
    r = pack(tmp_path, "s1", "ce", "ce", check=False)
    assert r.returncode != 0 and "같은 run 폴더" in r.stderr
    specs = [f"ce:RUN_TAG=t{i}" for i in range(5)]
    r = pack(tmp_path, "s1", *specs, check=False)
    assert r.returncode != 0 and "GPU 4개" in r.stderr


def test_finished_run_is_dropped(tmp_path):
    done = tmp_path / "scratch" / "runs" / "mc_mon_50m_ctx8192_s0"
    done.mkdir(parents=True)
    (done / "train.jsonl").write_text('{"event": "start", "total_steps": 9, "stop_step": 5}\n'
                                      '{"event": "end", "step": 5}\n')
    out = pack(tmp_path, "s1", "mc:RUN_TAG=mon", "ce").stdout
    assert "학습 끝남  mc_mon_50m_ctx8192_s0" in out
    assert "mc_mon" not in out.split("[dry] pack")[1]


@pytest.mark.parametrize("visible,expect", [(None, ["0", "1"]), ("4,6", ["4", "6"])])
def test_pack_train_gives_each_run_its_gpu(tmp_path, visible, expect):
    plan = tmp_path / "p.tsv"
    plan.write_text("a\t/c/a.env\tRUN_TAG=x MC_AUX_WEIGHT=0.3\nb\t/c/b.env\t\n")
    stub = tmp_path / "train.sh"
    stub.write_text('echo "$CUDA_VISIBLE_DEVICES $NGPU_PER_NODE $CONFIG ${MC_AUX_WEIGHT:-none}"\n'
                    '[ "$CONFIG" != /c/b.env ] || exit 3\n')
    env = {"PATH": os.environ["PATH"], "PACK_FILE": str(plan), "PACK_TRAIN": f"bash {stub}"}
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = visible
    r = subprocess.run(["bash", "scripts/pack_train.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "실패  b (exit 3" in r.stdout and "끝남  a" in r.stdout
    assert (tmp_path / "a_local.log").read_text().split() == [expect[0], "1", "/c/a.env", "0.3"]
    assert (tmp_path / "b_local.log").read_text().split() == [expect[1], "1", "/c/b.env", "none"]
