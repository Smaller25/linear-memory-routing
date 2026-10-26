"""X1 GPU dump: per-(model,dataset,sample) npz of connector output u_t,
answer-position retrieval query q_t, 32-subblock normalized-key descriptors,
full-segment descriptors, and stock routing scores — everything X1 (H_blind
test + rev2 u_t:=q_t probe) and later X-tasks need, recomputed with the
*exact* math E1 uses (routing_stats.routing_scores_at) so the dump can be
cross-checked against results/e1_routing.json.

추가 forward pass 없음: E1과 동일하게 capture_hidden으로 한 번만 forward하고,
그 hidden에서 layer별 attn._project를 재사용해 특징을 뽑는다.

rev2 (§0.3, §2b): u_t := q_t 처방을 무료로 검증하기 위해 attn._project의
retrieval query(첫 번째 반환값)를 answer position에서 뽑아 커널 규약대로
L2-정규화(F.normalize(q.float(), p=2, dim=-1))해 u와 동일 레이아웃으로 저장.

출력: $MC_OUT/x1_dump/{model}/{dataset}/{ri}.npz
  (ri = enumerate() 루프 위치, 항상 유일. 벤더링된 RULER의 niah.py에 변수
   섀도잉 버그가 있어 r["sample_id"](RULER 원본 "index" 필드)가 실제로는
   answer 문자열의 char offset이라 유일하지 않을 수 있음 -> 파일명 키로
   쓸 수 없음. sample_id는 meta.json에만 보존.)
  u            [L,H,K]      fp16  — answer position(T-1) connector output
  q            [L,H,K]      fp16  — answer position(T-1) retrieval query
                                    (attn._project 첫 반환값), L2-정규화
  csub_raw     [L,N,32,H,K] fp16  — L2-정규화된 key의 segment별 32 contiguous
                                    sub-block mean-pool (block sizes via
                                    round(p*C/32) boundaries; partial last
                                    segment still gets 32 blocks; empty block
                                    (C<32일 때 일부 p) -> 0)
  c_full       [L,N,H,K]    fp16  — seg.mean(0) 직접 계산 (재구성 검증용 정답)
  stock_scores [L,N]        fp32  — routing_scores_at(attn, h, T-1) verbatim
  (+ {ri}.meta.json sidecar: gold_seg, cur_seg, n_seg, n_tok, eligible,
     key_segs, sample_id (RULER "index", NOT guaranteed unique — see above),
     pair_id?, condition?)

일관성 검증: 저장된(=fp16 캐스팅된) csub_raw를 블록 크기 가중 평균으로 합쳐
c_full을 재구성 -> max-abs diff < 2e-3 아니면 즉시 assert 실패 (fail loudly).
"""
import argparse, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc
from routing_stats import DATASETS, _rows_for, capture_hidden, routing_scores_at
import data as mcdata

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
CHUNK, TOPK = load_mc.CHUNK, load_mc.TOPK
NSUB = 32
RECON_TOL = 2e-3


def _block_bounds(C, P=NSUB):
    return [round(p * C / P) for p in range(P + 1)]


@torch.no_grad()
def dump_sample(model, tok, input_text, layers):
    """(u, csub_raw, c_full, stock_scores) numpy arrays + meta dict for one sample."""
    ann = mcdata.annotate(input_text, tok)
    ids = torch.tensor([tok(input_text, add_special_tokens=False).input_ids],
                       device="cuda")
    T = ids.shape[1]
    assert T == ann["n_tok"], f"tokenization mismatch: T={T} ann.n_tok={ann['n_tok']}"
    t = T - 1
    cur_seg = t // CHUNK
    eligible = ann["gold_seg"] < cur_seg
    key_segs = sorted({n["seg"] for n in ann["needles"]})

    hiddens = capture_hidden(model, ids)
    n_seg_ref = ann["n_seg"]
    seg_lens = [min(CHUNK, T - s * CHUNK) for s in range(n_seg_ref)]
    L = len(layers)

    u_arr = q_arr = csub_arr = cfull_arr = scores_arr = None

    for i, attn in layers:
        h = hiddens[i]                                    # [T,D] bf16 cuda
        hb = h.unsqueeze(0)
        qproj, k, v, g, b, w = attn._project(hb)
        rk = F.normalize(k.float(), p=2, dim=-1)           # [1,T,H,K] fp32
        T_ = rk.shape[1]
        n_seg = (T_ + CHUNK - 1) // CHUNK
        assert n_seg == n_seg_ref, f"n_seg mismatch layer={i}: {n_seg} vs ann={n_seg_ref}"
        H_, K_ = rk.shape[2], rk.shape[3]

        if u_arr is None:
            u_arr = np.zeros((L, H_, K_), dtype=np.float16)
            q_arr = np.zeros((L, H_, K_), dtype=np.float16)
            csub_arr = np.zeros((L, n_seg, NSUB, H_, K_), dtype=np.float16)
            cfull_arr = np.zeros((L, n_seg, H_, K_), dtype=np.float16)
            scores_arr = np.zeros((L, n_seg), dtype=np.float32)

        for s in range(n_seg):
            seg = rk[0, s * CHUNK: min((s + 1) * CHUNK, T_)]   # [<=256,H,K]
            C = seg.shape[0]
            assert C == seg_lens[s], f"segment length mismatch layer={i} s={s}: {C} vs {seg_lens[s]}"
            bounds = _block_bounds(C, NSUB)
            for p in range(NSUB):
                blk = seg[bounds[p]:bounds[p + 1]]
                if blk.shape[0]:
                    csub_arr[i, s, p] = blk.mean(0).to(torch.float16).cpu().numpy()
                # else stays 0
            cfull_arr[i, s] = seg.mean(0).to(torch.float16).cpu().numpy()

        u_t = attn.ssc.connector(hb[:, T_ - 1:T_]).view(
            attn.ssc.num_heads, attn.ssc.head_qk_dim)
        u_arr[i] = u_t.detach().to(torch.float16).cpu().numpy()

        # rev2: retrieval query at the answer position, kernel-convention
        # L2-normalized (same normalization the keys get above), same [H,K]
        # layout as u — this is the u_t := q_t probe's raw ingredient.
        rq_t = F.normalize(qproj[0, T_ - 1].float(), p=2, dim=-1)   # [H,K]
        q_arr[i] = rq_t.detach().to(torch.float16).cpu().numpy()

        stock = routing_scores_at(attn, h, T_ - 1)          # [n_seg] fp32, future/cur -inf
        scores_arr[i] = stock.float().cpu().numpy()

    meta = {"gold_seg": ann["gold_seg"], "cur_seg": cur_seg, "n_seg": n_seg_ref,
            "n_tok": T, "eligible": eligible, "key_segs": key_segs}
    return u_arr, q_arr, csub_arr, cfull_arr, scores_arr, meta, seg_lens


def _check_reconstruction(csub_arr, cfull_arr, seg_lens):
    """저장될 fp16 값(csub_arr/cfull_arr) 그 자체로 재구성 오차를 계산.

    블록 크기 가중 평균으로 csub_arr의 32 블록을 합쳐 cfull_arr를 재구성한다.
    반환: max-abs diff (float)."""
    L, N = cfull_arr.shape[0], cfull_arr.shape[1]
    csub_f = csub_arr.astype(np.float32)
    cfull_f = cfull_arr.astype(np.float32)
    max_diff = 0.0
    for s in range(N):
        C = seg_lens[s]
        bounds = _block_bounds(C, NSUB)
        weights = np.array([bounds[p + 1] - bounds[p] for p in range(NSUB)],
                           dtype=np.float32)
        recon = (csub_f[:, s] * weights[None, :, None, None]).sum(1) / C  # [L,H,K]
        diff = np.abs(recon - cfull_f[:, s]).max()
        max_diff = max(max_diff, float(diff))
    return max_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mc-5B", "mc-30B"])
    a = ap.parse_args()

    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(a.model)
    layers = load_mc.mc_layers(model)
    print(f"[x1_dump] {a.model}: n_layers={len(layers)}", flush=True)

    for dsname in DATASETS:
        rows = _rows_for(dsname)
        out_dir = os.path.join(MC_OUT, "x1_dump", a.model, dsname)
        os.makedirs(out_dir, exist_ok=True)
        index = []
        for ri, r in enumerate(rows):
            try:
                u_arr, q_arr, csub_arr, cfull_arr, scores_arr, meta, seg_lens = dump_sample(
                    model, tok, r["input"], layers)
            except Exception as e:
                print(f"[x1_dump][warn] {dsname} sample {r['sample_id']} failed: {e}",
                      flush=True)
                continue

            max_diff = _check_reconstruction(csub_arr, cfull_arr, seg_lens)
            assert max_diff < RECON_TOL, (
                f"[x1_dump] RECONSTRUCTION FAILED {a.model}/{dsname} "
                f"sample={r['sample_id']}: max_abs_diff={max_diff} >= {RECON_TOL}")

            if "pair_id" in r:
                meta["pair_id"] = r["pair_id"]
                meta["condition"] = r["condition"]
            meta["sample_id"] = r["sample_id"]

            # File key = enumeration position `ri`, NOT r["sample_id"].
            # Vendored RULER (src/ruler/gen/synthetic/niah.py:~281) has a
            # variable-shadowing bug: the "index" field it writes is
            # actually a char offset (input_text.find(answer[0])), not the
            # loop index, so it collides across samples. `ri` is always
            # unique per (model,dataset) run. sample_id is preserved in meta
            # for provenance/joins.
            idx = ri
            npz_path = os.path.join(out_dir, f"{idx}.npz")
            np.savez(npz_path, u=u_arr, q=q_arr, csub_raw=csub_arr, c_full=cfull_arr,
                     stock_scores=scores_arr)
            with open(os.path.join(out_dir, f"{idx}.meta.json"), "w") as f:
                json.dump(meta, f)
            index.append(idx)
            print(f"[x1_dump] {a.model}/{dsname} {ri+1}/{len(rows)} idx={idx} "
                  f"sample_id={r['sample_id']} recon_max_diff={max_diff:.2e}", flush=True)

        with open(os.path.join(out_dir, "_index.json"), "w") as f:
            json.dump({"n_samples": len(index), "file_index": index}, f)
        print(f"[x1_dump] {a.model}/{dsname}: wrote {len(index)}/{len(rows)} samples",
              flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"[x1_dump] {a.model}: ALL DONE", flush=True)


if __name__ == "__main__":
    main()
