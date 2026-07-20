# -*- coding: utf-8 -*-
"""G1용 FineWeb-Edu 로컬 준비 (aigpu0423): 다운로드 + Mistral tokenize.

VESSL 체인과 독립 — G1(46M random vs fixed)을 로컬에서 먼저 돌리기 위함.
출력: /data2/sohyung/dynmc/tokenized/fineweb-{tokens.bin,doclens.npy}
"""
from __future__ import annotations

import os
import time

import numpy as np

BASE = "/data2/sohyung/dynmc"
RAW = f"{BASE}/raw"
OUT = f"{BASE}/tokenized"
CAP_TOKENS = 650_000_000


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/.done-fineweb"):
        print("already done")
        return

    from huggingface_hub import snapshot_download
    snapshot_download("mistralai/Mistral-7B-v0.1", repo_type="model",
                      allow_patterns=["tokenizer.json", "tokenizer_config.json",
                                      "tokenizer.model", "special_tokens_map.json"],
                      local_dir=f"{RAW}/tokenizer_mistral")
    snapshot_download("HuggingFaceFW/fineweb-edu", repo_type="dataset",
                      allow_patterns=["sample/10BT/000_00000.parquet",
                                      "sample/10BT/001_00000.parquet"],
                      local_dir=f"{RAW}/fineweb_edu", max_workers=8)
    print("downloads done", flush=True)

    from tokenizers import Tokenizer
    import pyarrow.parquet as pq
    tok = Tokenizer.from_file(f"{RAW}/tokenizer_mistral/tokenizer.json")
    probe = tok.encode("hello").ids
    print(f"BOS probe: {probe[:3]} (expect leading 1)", flush=True)
    manual_bos = probe[0] != 1

    import glob
    files = sorted(glob.glob(f"{RAW}/fineweb_edu/sample/10BT/*.parquet"))
    lens, n_tok, t0 = [], 0, time.time()
    with open(f"{OUT}/fineweb-tokens.bin", "wb") as fout:
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
                texts = batch.column("text").to_pylist()
                for e in tok.encode_batch(texts):
                    ids = ([1] + e.ids) if manual_bos else e.ids
                    fout.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    lens.append(len(ids))
                    n_tok += len(ids)
                if n_tok >= CAP_TOKENS:
                    break
            print(f"{os.path.basename(path)}: cumulative {n_tok/1e9:.3f}B "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if n_tok >= CAP_TOKENS:
                break
    np.save(f"{OUT}/fineweb-doclens.npy", np.asarray(lens, dtype=np.uint32))
    open(f"{OUT}/.done-fineweb", "w").close()
    print(f"FINEWEB_LOCAL_COMPLETE docs={len(lens)} tokens={n_tok/1e9:.3f}B", flush=True)


if __name__ == "__main__":
    main()
