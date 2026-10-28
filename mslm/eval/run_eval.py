"""RULER NIAH evaluation with official generation scoring + routing diagnostics.

For every item: prompt = input + answer_prefix (as RULER's call_api.py does),
greedy decoding up to the task's tokens_to_generate (stop at EOS), RULER
postprocess, string_match_all. For MC models it also records, at the last
prompt token, which segments each layer routed to versus the gold segments
(the segments holding the needle values), the gold segment's rank, and the
per-item hypergeometric random-routing baseline.

Route modes: ``learned`` (the model's router) and ``oracle`` (gold segments
forced into the top-k; eval only). Results:
    <out>/items.jsonl    one line per (route, length, task, item) — for paired comparison
    <out>/summary.json   per (route, length, task) scores and routing stats
"""

from __future__ import annotations

import json
import os
import time

import torch
import yaml

from mslm.envfile import REPO_ROOT
from mslm.eval.metrics import item_score, postprocess_pred, routing_stats, string_match_all
from mslm.eval.ruler_prepare import task_file
from mslm.models import LM, ModelConfig
from mslm.settings import Settings


def resolve_ckpt(s: Settings) -> str:
    c = s.str("EVAL_CKPT")
    if c != "latest":
        return c
    d = os.path.join(s.run_dir(), "ckpt")
    with open(os.path.join(d, "latest")) as f:
        return os.path.join(d, f.read().strip())


def load_model(path: str, device, overrides: dict | None = None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = dict(ck["model_config"])
    for k, v in (overrides or {}).items():
        mc["mc"][k] = v
    cfg = ModelConfig.from_dict(mc)
    model = LM(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["state"]


def gold_token_positions(tok, prompt: str, answers: list[str]):
    enc = tok(prompt, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    pos = []
    for a in answers:
        c = prompt.find(a)
        if c < 0:
            continue
        pos.append(next(i for i, (s, e) in enumerate(offs) if e > c and e > s))
    return ids, pos


@torch.no_grad()
def generate(model, ids, max_new, eos_id, device, amp, oracle=None, capture=False):
    """Greedy decoding by full re-forward (exact for these recurrent models)."""
    x = torch.tensor([ids], device=device)
    out, records = [], None
    for step in range(max_new):
        model.set_routing(oracle_segments=oracle, capture=capture and step == 0)
        with amp:
            logits = model(x)[0, -1]
        if step == 0 and capture:
            records = [(r.scores[0, -1].float().cpu(), r.indices[0, -1].cpu()) for r in model.routing_records()]
        nxt = int(logits.argmax())
        if nxt == eos_id:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    model.set_routing()
    return out, records


def main():
    s = Settings.from_env()
    from transformers import AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    ckpt = resolve_ckpt(s)
    model, state = load_model(ckpt, device)
    if state.get("tokenizer") != s.str("TOKENIZER"):
        raise ValueError(f"checkpoint tokenizer {state.get('tokenizer')} != TOKENIZER {s.str('TOKENIZER')}")
    tok = AutoTokenizer.from_pretrained(s.str("TOKENIZER"))
    cfg = model.cfg
    out_dir = s.str("EVAL_OUT") or os.path.join(os.path.dirname(os.path.dirname(ckpt)), "eval",
                                                 os.path.basename(ckpt).removesuffix(".pt"))
    os.makedirs(out_dir, exist_ok=True)
    from importlib import util
    spec = util.spec_from_file_location("ruler_constants", REPO_ROOT / "third_party/ruler/gen/synthetic/constants.py")
    consts = util.module_from_spec(spec)
    spec.loader.exec_module(consts)
    task_yaml = yaml.safe_load(open(REPO_ROOT / "third_party/ruler/synthetic.yaml"))

    routes = s.list("EVAL_ROUTE") if cfg.arch == "mc" else ["none"]
    limit = s.int("EVAL_LIMIT")
    items_path = os.path.join(out_dir, "items.jsonl")
    summary = {"ckpt": ckpt, "arch": cfg.arch, "chunk": cfg.mc.chunk, "topk": cfg.mc.topk,
               "tokenizer": s.str("TOKENIZER"), "answer_prefix": s.bool("EVAL_ANSWER_PREFIX"), "results": []}
    t0 = time.time()
    with open(items_path, "w") as fout:
        for length in [int(x) for x in s.list("EVAL_LENGTHS")]:
            for task in s.list("EVAL_TASKS"):
                path = task_file(s.str("EVAL_DATA_DIR"), s.str("TOKENIZER"), length, task)
                items = [json.loads(l) for l in open(path)]
                items = items[:limit] if limit > 0 else items
                base_task = task_yaml[task]["task"]
                max_new = s.int("EVAL_MAX_NEW_TOKENS") or consts.TASKS[base_task]["tokens_to_generate"]
                for route in routes:
                    preds, refs, rstats = [], [], []
                    for it in items:
                        prompt = it["input"] + (it.get("answer_prefix", "") if s.bool("EVAL_ANSWER_PREFIX") else "")
                        ids, gpos = gold_token_positions(tok, prompt, it["outputs"])
                        gold = sorted({p // cfg.mc.chunk for p in gpos})
                        oracle = None
                        if route == "oracle":
                            oracle = torch.tensor([gold or [-1]], device=device)
                        gen, recs = generate(model, ids, max_new, tok.eos_token_id, device, amp,
                                             oracle=oracle, capture=cfg.arch == "mc")
                        pred = postprocess_pred(tok.decode(gen, skip_special_tokens=True))
                        layers = [routing_stats(sc, ch, gold) for sc, ch in (recs or [])]
                        rec = {"route": route, "length": length, "task": task, "index": it.get("index"),
                               "prompt_tokens": len(ids), "pred": pred, "answers": it["outputs"],
                               "score": item_score(pred, it["outputs"]), "gold_segments": gold,
                               "gold_token_pos": gpos, "routing": layers}
                        fout.write(json.dumps(rec) + "\n")
                        fout.flush()
                        preds.append(pred)
                        refs.append(it["outputs"])
                        rstats.extend(l for l in layers if l["hit"] is not None)
                    res = {"route": route, "length": length, "task": task, "n": len(items),
                           "string_match_all": string_match_all(preds, refs)}
                    if rstats:
                        res.update(routing_hit=sum(r["hit"] for r in rstats) / len(rstats),
                                   random_hit=sum(r["random_hit"] for r in rstats) / len(rstats),
                                   mean_gold_rank=sum(r["gold_rank"] for r in rstats) / len(rstats))
                    summary["results"].append(res)
                    print(json.dumps(res), f"[{time.time() - t0:.0f}s]", flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {out_dir}")


if __name__ == "__main__":
    main()
