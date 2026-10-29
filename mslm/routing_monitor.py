"""Routing hit rates measured during training (cheap, no generation).

copy_hit_counts: on validation text, at every span-copy labelled position, did
each layer's top-k contain a gold segment, and what would uniform random top-k
give (1 - C(N-G,k)/C(N,k) with N = eligible past segments).

RulerMonitor: a fixed set of RULER items (generated with a seed DIFFERENT from
the final evaluation's EVAL_DATA_SEED), one teacher-forced forward of the
prompt each, routing at the last prompt token versus the needle segments —
the same statistic run_eval reports, so the two are comparable.
"""

from __future__ import annotations

import json
import os

import torch

from mslm.eval.metrics import routing_stats


def random_hit(n_cand: torch.Tensor, n_gold: torch.Tensor, k: int) -> torch.Tensor:
    """Vectorised hypergeometric P(at least one gold among k uniform picks)."""
    kk = torch.minimum(n_cand, torch.full_like(n_cand, k)).float()
    miss = torch.ones_like(kk)
    for i in range(k):
        num = (n_cand - n_gold - i).float()
        den = (n_cand - i).float()
        step = torch.where(i < kk, num.clamp(min=0) / den.clamp(min=1), torch.ones_like(kk))
        miss = miss * step
    return torch.where(n_gold > 0, 1 - miss, torch.zeros_like(miss))


def copy_hit_counts(records, gold: torch.Tensor, chunk: int) -> torch.Tensor:
    """[L,3] sums of (hit, random_hit, n_positions) over labelled positions, per MC layer."""
    out = []
    for rec in records:
        g = gold.to(rec.indices.device)[:, :rec.indices.shape[1]]
        b, t = torch.nonzero(g[..., 0] >= 0, as_tuple=True)
        if b.numel() == 0:
            out.append(torch.zeros(3, device=rec.indices.device))
            continue
        gi, ch = g[b, t], rec.indices[b, t]                          # [M,G], [M,R]
        hit = ((ch.unsqueeze(-1) == gi.unsqueeze(-2)) & (gi.unsqueeze(-2) >= 0) & (ch.unsqueeze(-1) >= 0)).any(-1).any(-1)
        n_cand = rec.seg_of_token.to(t.device)[t]
        rnd = random_hit(n_cand, (gi >= 0).sum(-1), ch.shape[-1])
        out.append(torch.stack([hit.float().sum(), rnd.sum(), torch.tensor(float(b.numel()), device=hit.device)]))
    return torch.stack(out)


class RulerMonitor:
    def __init__(self, path: str, n_items: int, tokenizer: str, final_eval_seed: int, chunk: int):
        meta = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))), "meta.json")
        if os.path.exists(meta):
            seed = json.load(open(meta)).get("seed")
            if seed == final_eval_seed:
                raise ValueError(f"ROUTE_MONITOR_FILE was generated with seed {seed} = EVAL_DATA_SEED; "
                                 f"monitor items must differ from the final evaluation items")
        else:
            print(f"[route-monitor] WARNING no meta.json next to {path}; cannot check the item seed", flush=True)
        from transformers import AutoTokenizer

        from mslm.eval.run_eval import gold_token_positions
        tok = AutoTokenizer.from_pretrained(tokenizer)
        self.items = []
        for line in list(open(path))[:n_items]:
            it = json.loads(line)
            prompt = it["input"] + it.get("answer_prefix", "")
            ids, gpos = gold_token_positions(tok, prompt, it["outputs"])
            self.items.append((ids, sorted({p // chunk for p in gpos})))

    @torch.no_grad()
    def __call__(self, model, device, amp) -> dict:
        was_training = model.training
        model.eval()
        per_layer = None
        for ids, gold in self.items:
            model.set_routing(capture=True)
            with amp:
                model(torch.tensor([ids], device=device))
            stats = [routing_stats(r.scores[0, -1].float().cpu(), r.indices[0, -1].cpu(), gold)
                     for r in model.routing_records()]
            if per_layer is None:
                per_layer = [[] for _ in stats]
            for acc, st in zip(per_layer, stats):
                if st["hit"] is not None:
                    acc.append(st)
        model.set_routing()
        model.train(was_training)
        hit = [sum(s["hit"] for s in acc) / max(len(acc), 1) for acc in per_layer]
        rnd = [sum(s["random_hit"] for s in acc) / max(len(acc), 1) for acc in per_layer]
        return {"ruler_hit": round(sum(hit) / len(hit), 4), "ruler_random_hit": round(sum(rnd) / len(rnd), 4),
                "ruler_hit_by_layer": [round(h, 3) for h in hit], "ruler_items": len(self.items)}
