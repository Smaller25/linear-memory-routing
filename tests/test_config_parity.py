"""The shell loader (scripts/lib/config.sh) and the Python loader (mslm/envfile.py,
used by the notebook) must resolve every config identically, with and without
overrides, and build the identical sbatch command."""

import os
import subprocess
from pathlib import Path

import pytest

from mslm import envfile, tower

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted(str(p.relative_to(REPO)) for p in (REPO / "configs").glob("*.env"))
OVERRIDE_SETS = [{}, {"SEED": "2", "MAX_TOKENS": "1e9"}, {"SCRATCH_ROOT": "/tmp/x y", "SB_COMMENT": "field=nlp;appl=pytorch"}]


def bash(cmd: str, extra: dict) -> str:
    env = {"PATH": os.environ["PATH"], "USER": os.environ.get("USER", "u"), "HOME": os.environ.get("HOME", "/")}
    env.update(extra)
    return subprocess.run(["bash", "-c", f"source scripts/lib/config.sh; {cmd}"], cwd=REPO, env=env,
                          capture_output=True, text=True, check=True).stdout


def py_env(extra):
    return {"USER": os.environ.get("USER", "u"), "HOME": os.environ.get("HOME", "/"), **extra}


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("ov", OVERRIDE_SETS)
def test_resolve_matches_shell(config, ov):
    sh = bash(f"mslm_load_config {config} && mslm_print_config", ov)
    py = envfile.resolve(REPO / config, py_env(ov))
    assert sh == "".join(f"{k}={v}\n" for k, v in py.items())


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("ov", OVERRIDE_SETS)
@pytest.mark.parametrize("job", ["train", "eval"])
def test_sbatch_command_matches_shell(config, ov, job, monkeypatch):
    sh = bash(f"mslm_load_config {config} && mslm_sbatch_cmd {job}", ov).strip()
    for k in ("USER", "HOME"):
        monkeypatch.setenv(k, py_env({})[k])
    assert sh == tower.sbatch_command(config, job, ov)


def test_every_config_key_declared_and_commented():
    base = envfile.parse_env_file(envfile.BASE_ENV)
    assert all(ln.comment for ln in base.lines), "every base.env variable needs a one-line comment"
    for c in CONFIGS:
        for ln in envfile.parse_env_file(REPO / c).lines:
            assert ln.key in base.keys, f"{c}: {ln.key} not declared in base.env"


def test_no_slurm_reserved_names():
    # SLURM_* are set by SLURM inside jobs and would be mistaken for overrides
    assert not [k for k in envfile.declared_keys() if k.startswith("SLURM_")]


def test_set_value_roundtrip(tmp_path):
    p = tmp_path / "x.env"
    p.write_text("SEED=0                                 # seed\n")
    envfile.set_value(p, "SEED", "3")
    envfile.set_value(p, "MAX_TOKENS", "5e9")
    vals = envfile.resolve(p, py_env({}))
    assert vals["SEED"] == "3" and vals["MAX_TOKENS"] == "5e9"
    assert "# seed" in p.read_text()
