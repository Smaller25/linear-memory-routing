"""mslm.wandb_sync: train.jsonl -> wandb, with a stand-in wandb module (no network, no key)."""

import json
import sys
import types

from mslm import wandb_sync


class FakeRun:
    def __init__(self, step):
        self.step, self.logged, self.config, self.finished = step, [], types.SimpleNamespace(), False
        self.config.update = lambda d, allow_val_change=False: self.config.__dict__.update(d)

    def log(self, m, step):
        self.logged.append((step, m))

    def finish(self):
        self.finished = True


def fake_wandb(monkeypatch, start_step=0):
    calls = {}

    def init(**kw):
        calls.update(kw)
        calls["run"] = FakeRun(start_step)
        return calls["run"]

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    return calls


def write_run(tmp_path, lines):
    d = tmp_path / "mc_50m_ctx8192_s0"
    d.mkdir()
    (d / "config.env").write_text('MODEL_ARCH="mc"\nSEED=0\n')
    (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    return d


LINES = [
    {"event": "start", "step": 0, "total_steps": 30, "warmup": 1},
    {"step": 10, "tokens": 100, "loss": 5.0, "aux_loss": 0.0, "lr": 1e-3, "grad_norm": 1.0, "tok_per_s": 9, "mem_gb": 1},
    {"step": 10, "tokens": 100, "val_loss": 5.5},
    {"event": "save", "step": 10, "ckpt": "x"},
    {"step": 20, "tokens": 200, "loss": 4.0, "aux_loss": 0.0, "lr": 1e-3, "grad_norm": 1.0, "tok_per_s": 9, "mem_gb": 1},
]


def test_uploads_metrics_with_run_name_and_config(tmp_path, monkeypatch):
    calls = fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_ENTITY", "")
    wandb_sync.sync(str(write_run(tmp_path, LINES)))
    run = calls["run"]
    assert calls["id"] == calls["name"] == "mc_50m_ctx8192_s0" and calls["resume"] == "allow"
    assert calls["project"] == "proj" and calls["entity"] is None
    assert calls["config"] == {"MODEL_ARCH": "mc", "SEED": 0}
    assert run.config.total_steps == 30
    assert [s for s, _ in run.logged] == [10, 10, 20]
    assert run.logged[0][1]["train/loss"] == 5.0 and run.logged[1][1] == {"val/loss": 5.5, "tokens": 100}
    assert run.finished


def test_resumed_run_skips_steps_wandb_already_has(tmp_path, monkeypatch):
    calls = fake_wandb(monkeypatch, start_step=11)
    wandb_sync.sync(str(write_run(tmp_path, LINES)))
    assert [s for s, _ in calls["run"].logged] == [20]


def test_tail_waits_for_complete_lines(tmp_path):
    p = tmp_path / "train.jsonl"
    tail = wandb_sync.JsonlTail(str(p))
    assert tail.read() == []
    p.write_text('{"step": 1}\n{"step": 2, "lo')
    assert tail.read() == [{"step": 1}]
    with open(p, "a") as f:
        f.write('ss": 3.0}\n')
    assert tail.read() == [{"step": 2, "loss": 3.0}]


def test_val_routing_metrics_are_uploaded():
    from mslm.wandb_sync import record_to_metrics
    rec = {"step": 500, "tokens": 9, "val_loss": 3.2, "copy_hit": 0.4, "copy_random_hit": 0.1,
           "copy_hit_by_layer": [0.3, 0.5], "copy_positions": 120, "ruler_hit": 0.2, "ruler_hit_by_layer": [0.1, 0.3]}
    m = record_to_metrics(rec)
    assert m["val/loss"] == 3.2 and m["val/copy_hit"] == 0.4 and m["val/copy_positions"] == 120
    assert m["val/copy_hit/L00"] == 0.3 and m["val/copy_hit/L01"] == 0.5 and m["val/ruler_hit/L01"] == 0.3
    assert record_to_metrics({"step": 1, "tokens": 1, "val_loss": 2.0}) == {"val/loss": 2.0, "tokens": 1}
