"""ssketch descriptor 계산을 pinned worktree 위에 얹기 위한 다리.

왜 importlib 인가
-----------------
`load_mc.bootstrap()`은 **pinned** long-gdn worktree
(`MC_LONGGDN_WORKTREE`, 기본 e71713e)를 `sys.path` 맨 앞에 넣는다. 모델
로드·커널이 전부 거기서 온다(프로토콜 CONSTANT 4). 그런데 `dsc/mc_sketch/`는
새 실험 worktree에만 있다. 두 worktree를 동시에 `sys.path`에 넣으면 `dsc`
패키지가 먼저 온 쪽으로 고정돼 `import dsc.mc_sketch`가 실패한다.

그래서 `mc_sketch/sketch.py`와 `mc_sketch/scoring.py`는 **의존성이 전혀 없는
순수 torch 파일**로 작성되어 있고, 여기서는 그 두 파일만 파일 경로로 직접
로드한다. 결과적으로 모델·커널은 pinned worktree 것을 쓰고, 점수 수식만 새
모듈에서 가져온다 — 핀이 흔들리지 않는다.

경로는 `MC_SKETCH_ROOT`로 덮어쓸 수 있다.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

DEFAULT_SKETCH_ROOT = os.environ.get(
    "MC_SKETCH_ROOT",
    "/home/sohyung/long-gdn/.sh_exp_worktrees/ssketch-router-descriptor/dsc/mc_sketch",
)

_MODS: dict[str, object] = {}


def _load(name: str):
    """`<MC_SKETCH_ROOT>/<name>.py`를 고유 모듈명으로 로드(패키지 import 아님)."""
    if name in _MODS:
        return _MODS[name]
    root = os.environ.get("MC_SKETCH_ROOT", DEFAULT_SKETCH_ROOT)
    path = os.path.join(root, f"{name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"mc_sketch/{name}.py not found at {path}. "
            "Set MC_SKETCH_ROOT to <long-gdn worktree>/dsc/mc_sketch."
        )
    modname = f"_mc_sketch_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    _MODS[name] = mod
    return mod


def sketch_module():
    return _load("sketch")


def scoring_module():
    return _load("scoring")


def sketch_source_root() -> str:
    return os.environ.get("MC_SKETCH_ROOT", DEFAULT_SKETCH_ROOT)


def make_P(d_v: int, rank: int, seed: int, device="cuda") -> torch.Tensor:
    """`[d_v, rank]` fp32. rank == d_v 면 항등행렬(full-state arm)."""
    return sketch_module().make_sketch(d_v, rank, seed=seed, device=device,
                                       dtype=torch.float32)


@torch.no_grad()
def segment_states(attn, hidden, chunk_size):
    """한 layer의 청크별 최종 상태 S_m을 얻는다.

    pinned worktree의 `_segment_gdn2_batched`(원본 무수정)를 그대로 태운다 —
    §3.4 Independent Compressors 배치 스캔이라 모델 forward와 같은 수식이다.

    성능 주의: RULER @2048 샘플의 실제 토큰 수는 256의 배수가 아니다
    (실측 1186~1903). 그래서 `_segment_gdn2_batched`는 배치 경로가 아니라
    **순차 fallback**(`_segment_gdn2_sequential`)을 탄다 — 결과는 동일하지만
    layer당 chunk_gdn2 호출이 N번이라 meank arm보다 느리다. sbatch를
    arm×모델 단위로 쪼개고 3시간을 잡아둔 이유가 이것이다.

    Args:
        attn: MemoryCachingGDN2Layer
        hidden: [T, D] 또는 [1, T, D]
    Returns:
        (online_output [1,T,H,V], memories [1,N,H,K,V])
    """
    from dsc.mc_gdn2.ssc import _segment_gdn2_batched
    from dsc.lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2

    hb = hidden.unsqueeze(0) if hidden.ndim == 2 else hidden
    q, k, v, g, b, w = attn._project(hb)
    online_output, memories = _segment_gdn2_batched(
        q, k, v, g, b, w, chunk_size=chunk_size, chunk_gdn2_fn=chunk_gdn2)
    return q, online_output, memories


@torch.no_grad()
def routing_scores_ssketch(attn, hidden, t, *, P, router_query="q",
                           online_score="read_norm", chunk_size=None):
    """t 위치의 과거 segment별 ssketch routing score [n_seg] + online 점수.

    score(t,m) = || concat_h( D_{m,h}^T u_{t,h} ) ||_2,  D_m = S_m @ P (fp32)

    router_query="q" 는 커널이 실제로 읽을 때 쓰는 바로 그 벡터
    (L2 정규화 + read_scale) 를 그대로 쓴다. GDN2SSC는 normalize_queries=True
    이므로 이 정합성을 깨면 "고르는 기준 = 실제 읽는 값" 주장이 성립하지 않는다.
    """
    sc = scoring_module()
    chunk_size = chunk_size or attn.ssc.chunk_size
    q, online_output, memories = segment_states(attn, hidden, chunk_size)

    D = sc.sketch_descriptors(memories, P)                       # [1,N,H,K,r]
    if router_query == "q":
        u = sc.make_router_query(q[:, t:t + 1],
                                 normalize_queries=attn.ssc.normalize_queries,
                                 scale=attn.ssc.read_scale)
    elif router_query == "wu":
        hb = hidden.unsqueeze(0) if hidden.ndim == 2 else hidden
        u = attn.ssc.connector(hb[:, t:t + 1]).view(
            1, 1, attn.ssc.num_heads, attn.ssc.head_qk_dim).float()
    else:
        raise ValueError(f"unknown router_query={router_query!r}")

    scores = sc.past_sketch_scores(u, D)[0, 0]                   # [N]

    d_v = online_output.shape[-1]
    rank = P.shape[1]
    if online_score == "read_norm":
        on = sc.online_read_norm_score(online_output[:, t:t + 1], P)[0, 0]
    elif online_score == "raw_norm":
        scores = scores * sc.raw_norm_past_correction(d_v, rank)
        on = sc.online_raw_norm_score(online_output[:, t:t + 1])[0, 0]
    elif online_score == "legacy":
        # 단위가 안 맞는 ablation. mc_ssc의 causal mean-pool 키 online 점수.
        import torch.nn.functional as F
        from dsc.mc_baseline.mc_ssc import causal_online_key_sums
        hb = hidden.unsqueeze(0) if hidden.ndim == 2 else hidden
        _, k, *_ = attn._project(hb)
        rk = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
        osum = causal_online_key_sums(rk, chunk_size)[:, t:t + 1]
        on = torch.einsum("bthk,bthk->bt", u.float(), osum.float())[0, 0]
    else:
        raise ValueError(f"unknown online_score={online_score!r}")

    cur_seg = t // chunk_size
    scores = scores.clone()
    scores[cur_seg:] = float("-inf")
    return scores, float(on)
