"""scripts/submit_pipeline.sh in DRY_RUN mode: right jobs, right order, right dependencies.
squeue/sbatch are stubbed so the test never touches a real queue."""

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(tmp_path, **extra):
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    for name in ("squeue", "sbatch"):
        (stub / name).write_text("#!/bin/sh\nexit 0\n")
        (stub / name).chmod(0o755)
    env = {"PATH": f"{stub}:{os.environ['PATH']}", "USER": "u", "HOME": str(tmp_path),
           "SCRATCH_ROOT": str(tmp_path / "scratch"), "DRY_RUN": "1", **extra}
    out = subprocess.run(["bash", "scripts/submit_pipeline.sh"], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout
    return [ln.split()[1] for ln in out.splitlines() if ln.startswith("[dry]")], out


def test_default_is_prepare_and_smoke(tmp_path):
    names, out = run(tmp_path)
    assert names == ["prepare_data", "prepare_eval", "smoke"]
    assert "--dependency=afterok:<prepare_data> --kill-on-invalid-dep=yes" in out
    assert "slurm/smoke.sbatch" in out and "--job-name=mslm-smoke " in out


def test_main_runs_wait_for_smoke(tmp_path):
    names, out = run(tmp_path, MAIN="1", SEEDS="0 1")
    assert names == ["prepare_data", "prepare_eval", "smoke",
                     "vanilla_50m_s0", "vanilla_50m_s1", "mc_50m_s0", "mc_50m_s1"]
    main_lines = [ln for ln in out.splitlines() if "slurm/train.sbatch" in ln]
    assert len(main_lines) == 4 and all("--dependency=afterok:<smoke>" in ln for ln in main_lines)
    assert "SEED=1" in main_lines[1]


def test_finished_steps_are_skipped(tmp_path):
    data = tmp_path / "scratch" / "data"
    (data / "fineweb_edu_tinyllama").mkdir(parents=True)
    (data / "fineweb_edu_tinyllama" / "meta.json").write_text("{}")
    (data / "eval_ruler" / "tok").mkdir(parents=True)
    (data / "eval_ruler" / "tok" / "meta.json").write_text("{}")
    smoke = tmp_path / "scratch" / "runs" / "smoke_mc_50m"
    smoke.mkdir(parents=True)
    (smoke / "SMOKE_OK").write_text("ok")
    done = tmp_path / "scratch" / "runs" / "mc_50m_ctx8192_s0"
    done.mkdir(parents=True)
    (done / "train.jsonl").write_text('{"event": "start", "total_steps": 5}\n{"event": "end", "step": 5}\n')
    names, out = run(tmp_path, MAIN="1", SEEDS="0")
    assert names == ["vanilla_50m_s0"]
    assert "--dependency" not in out.split("[dry] vanilla_50m_s0")[1].splitlines()[1]
