"""Helpers behind control_tower.ipynb. Everything reads/writes configs/*.env —
the same files the shell scripts load — so the notebook and the scripts
cannot drift (tests/test_config_parity.py and tests/test_notebook.py check it).
"""

from __future__ import annotations

import inspect
import os
import re
import tempfile
import time
from pathlib import Path

from mslm import envfile
from mslm.envfile import REPO_ROOT


def config_path(config: str | os.PathLike) -> Path:
    p = Path(config)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p


def list_configs() -> list[str]:
    return sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "configs").glob("*.env"))


def clean_environ(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Process env without stray config keys, plus explicit overrides."""
    env = {k: v for k, v in os.environ.items() if k not in set(envfile.declared_keys())}
    env.update({k: str(v) for k, v in (overrides or {}).items()})
    return env


def resolve(config, overrides: dict[str, str] | None = None) -> dict[str, str]:
    return envfile.resolve(config_path(config), clean_environ(overrides))


def table(config, overrides: dict[str, str] | None = None, section: str | None = None):
    """Rows of (section, key, value, source, comment) for display."""
    cfg = config_path(config)
    in_file = {ln.key for ln in envfile.parse_env_file(cfg).lines} if cfg != envfile.BASE_ENV else set()
    values = resolve(cfg, overrides)
    rows = []
    for ln in envfile.describe():
        if section and ln.section != section:
            continue
        src = "override" if overrides and ln.key in overrides else ("config" if ln.key in in_file else "base")
        rows.append((ln.section, ln.key, values[ln.key], src, ln.comment))
    return rows


def show(config, overrides=None, section=None):
    try:
        import pandas as pd
        return pd.DataFrame(table(config, overrides, section),
                            columns=["section", "key", "value", "from", "comment"]).set_index("key")
    except ImportError:
        for r in table(config, overrides, section):
            print(f"[{r[0]}] {r[1]}={r[2]}  ({r[3]})  # {r[4]}")


def set_var(config, key: str, value) -> None:
    """Write ``key=value`` into the config file itself (shell scripts see it too)."""
    envfile.set_value(config_path(config), key, str(value))


# ---------------------------------------------------------------- sbatch command
_SAFE = re.compile(r"^[A-Za-z0-9_./:=,@+%-]+$")


def shell_quote(v: str) -> str:
    """Same rule as scripts/lib/config.sh:mslm_quote."""
    return v if _SAFE.match(v) else "'" + v.replace("'", "'\\''") + "'"


def sbatch_command(config, job: str = "train", overrides: dict[str, str] | None = None) -> str:
    """The exact command scripts/sbatch_cmd.sh prints. Shown, never submitted."""
    if not (REPO_ROOT / "slurm" / f"{job}.sbatch").exists():
        raise FileNotFoundError(f"slurm/{job}.sbatch")
    cfg = config_path(config).resolve()
    v = resolve(cfg, overrides)
    q = shell_quote
    keys = envfile.declared_keys()
    env_part = "".join(f" {k}={q(str(overrides[k]))}" for k in keys if overrides and k in overrides)
    opts = (f"--job-name={q(v['SB_JOB_NAME'] + '-' + job)} --partition={q(v['SB_PARTITION'])}"
            f" --nodes={v['SB_NODES']} --ntasks-per-node=1 {v['SB_GPU_OPT']} --cpus-per-task={v['SB_CPUS_PER_TASK']}"
            f" --time={v['SB_TIME']}")
    if v["SB_COMMENT"]:
        opts += f" --comment={q(v['SB_COMMENT'])}"
    if v["SB_EXCLUSIVE"] == "1":
        opts += " --exclusive"
    if v["SB_EXTRA"]:
        opts += f" {v['SB_EXTRA']}"
    opts += f" --output={q(v['SB_LOG_DIR'] + '/%x_%j.out')} --error={q(v['SB_LOG_DIR'] + '/%x_%j.err')}"
    return (f"mkdir -p {q(v['SB_SUBMIT_DIR'])} {q(v['SB_LOG_DIR'])} && cd {q(v['SB_SUBMIT_DIR'])} && "
            f"MSLM_REPO={q(str(REPO_ROOT))} CONFIG={q(str(cfg))}{env_part} sbatch {opts} "
            f"{q(str(REPO_ROOT / 'slurm' / f'{job}.sbatch'))}")


# ---------------------------------------------------------------- idea modules
def parts(config, overrides=None) -> dict[str, str]:
    """Which registered part each MC plug point uses under this config."""
    v = resolve(config, overrides)
    return {"descriptor": v["MC_DESCRIPTOR"], "router": v["MC_ROUTER"], "gate": v["MC_GATE"],
            "connect": v["MC_CONNECT"], "aux_loss": v["MC_AUX_LOSS"]}


def available_parts() -> dict[str, list[str]]:
    from mslm.models import registry
    return {k: sorted(t) for k, t in registry._KINDS.items()}


def source_of(kind: str, name: str) -> str:
    from mslm.models import registry
    obj = registry.get(kind, name)
    return f"# {inspect.getsourcefile(obj)}:{inspect.getsourcelines(obj)[1]}\n" + inspect.getsource(obj)


# ---------------------------------------------------------------- smoke test
def smoke(config, overrides=None, device: str = "cpu", steps: int = 3, seq_len: int | None = None):
    """Build the configured model, train a few steps on random tokens, save,
    reload, check the reloaded model gives the same loss, and run one
    oracle-routed forward. Small sizes only (MODEL_SIZE=tiny for CPU)."""
    import torch

    from mslm.models import LM, ModelConfig
    from mslm.settings import Settings
    from mslm.train import lr_at, param_groups

    s = Settings(resolve(config, overrides))
    cfg = ModelConfig.from_settings(s)
    T = seq_len or min(s.int("SEQ_LEN"), 4 * cfg.mc.chunk)
    dev = torch.device(device)
    torch.manual_seed(s.int("SEED"))
    model = LM(cfg).to(dev)
    opt = torch.optim.AdamW(param_groups(model, s.float("WEIGHT_DECAY")), lr=s.float("LR"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    x = torch.randint(0, cfg.vocab_size, (2, T + 1), device=dev)
    log = []
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, 1, s.float("LR"), s.float("MIN_LR_RATIO"))
        with amp:
            out = model(x[:, :-1], x[:, 1:])
        out["loss"].backward()
        opt.step()
        opt.zero_grad()
        log.append(round(out["lm_loss"].item(), 4))
    with tempfile.TemporaryDirectory() as d:
        torch.save({"model": model.state_dict(), "model_config": cfg.to_json()}, f"{d}/ck.pt")
        m2 = LM(ModelConfig.from_dict(__import__("json").loads(torch.load(f"{d}/ck.pt")["model_config"]))).to(dev)
        m2.load_state_dict(torch.load(f"{d}/ck.pt", map_location=dev)["model"])
    model.eval()
    m2.eval()
    with torch.no_grad(), amp:
        a = model(x[:, :-1], x[:, 1:])["lm_loss"].item()
        b = m2(x[:, :-1], x[:, 1:])["lm_loss"].item()
        oracle = None
        if cfg.arch == "mc":
            model.set_routing(oracle_segments=torch.tensor([[0], [0]], device=dev), capture=True)
            model(x[:, :-1])
            rec = model.routing_records()[0]
            last = rec.indices[:, -1]
            oracle = bool((last == 0).any(-1).all())
            model.set_routing()
    return {"device": device, "params": model.num_params(), "seq_len": T, "losses": log,
            "reload_loss_equal": abs(a - b) < 1e-5, "oracle_picks_gold": oracle,
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- monitoring
def run_dir(config, overrides=None) -> str:
    from mslm.settings import Settings
    return Settings(resolve(config, overrides)).run_dir()


def train_log(config, overrides=None, last: int = 20) -> list[dict]:
    """Last lines of <run>/train.jsonl (works while the job is running)."""
    import json
    path = os.path.join(run_dir(config, overrides), "train.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path).read().splitlines()[-last:]]


def eval_summaries(config, overrides=None) -> dict[str, dict]:
    import glob
    import json
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir(config, overrides), "eval", "*", "summary.json"))):
        out[os.path.basename(os.path.dirname(p))] = json.load(open(p))
    return out


def shell_command(config, script: str = "train", overrides=None) -> str:
    """Equivalent direct (non-SLURM) command, e.g. for an interactive GPU node."""
    env = "".join(f"{k}={shell_quote(str(v))} " for k, v in (overrides or {}).items())
    return f"CONFIG={shell_quote(str(config))} {env}bash scripts/{script}.sh"
