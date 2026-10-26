# -*- coding: utf-8 -*-
"""G0용 SlimPajama validation 로컬 준비: venketh val chunk1 다운로드 + tokenize.

출력: /data2/sohyung/dynmc/tokenized/val-tokens.bin (+doclens) — exp0와
trigger 캘리브레이션(§2.5)이 이 파일을 사용.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

BASE = "/data2/sohyung/dynmc"
RAW = f"{BASE}/raw"
OUT = f"{BASE}/tokenized"
CAP_TOKENS = 60_000_000  # calibration 10M + G0 + 여유


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/.done-val"):
        print("already done")
        return
    from huggingface_hub import snapshot_download
    snapshot_download("venketh/SlimPajama-62B", repo_type="dataset",
                      allow_patterns=["validation/chunk1.jsonl.zst"],
                      local_dir=f"{RAW}/slimpajama_val")
    snapshot_download("mistralai/Mistral-7B-v0.1", repo_type="model",
                      allow_patterns=["tokenizer.json"],
                      local_dir=f"{RAW}/tokenizer_mistral")

    import zstandard as zstd
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{RAW}/tokenizer_mistral/tokenizer.json")
    manual_bos = tok.encode("hello").ids[0] != 1

    lens, n_tok = [], 0
    with open(f"{OUT}/val-tokens.bin", "wb") as fout, \
         open(f"{RAW}/slimpajama_val/validation/chunk1.jsonl.zst", "rb") as fh:
        reader = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        texts = []
        for line in reader:
            texts.append(json.loads(line)["text"])
            if len(texts) >= 2000:
                for e in tok.encode_batch(texts):
                    ids = ([1] + e.ids) if manual_bos else e.ids
                    fout.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    lens.append(len(ids)); n_tok += len(ids)
                texts = []
                if n_tok >= CAP_TOKENS:
                    break
        if texts and n_tok < CAP_TOKENS:
            for e in tok.encode_batch(texts):
                ids = ([1] + e.ids) if manual_bos else e.ids
                fout.write(np.asarray(ids, dtype=np.uint16).tobytes())
                lens.append(len(ids)); n_tok += len(ids)
    np.save(f"{OUT}/val-doclens.npy", np.asarray(lens, dtype=np.uint32))
    open(f"{OUT}/.done-val", "w").close()
    print(f"VAL_LOCAL_COMPLETE docs={len(lens)} tokens={n_tok/1e6:.1f}M", flush=True)


if __name__ == "__main__":
    main()
