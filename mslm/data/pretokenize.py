"""Pre-tokenise FineWeb-Edu into flat uint16 token shards.

Output (TRAIN_DATA_DIR):
    val_000.bin            first PRETOK_VAL_TOKENS tokens (held out, never trained on)
    train_000.bin, ...     PRETOK_SHARD_TOKENS tokens each
    meta.json              tokenizer, vocab, bos/eos ids, token counts, source

Each document is written as [BOS] + tokens + [EOS], the same way an
evaluation prompt is encoded (HF tokenizer default adds BOS).

Reads the parquet files of the dataset from the local HF cache in sorted
order (download them first with setup/download_assets.sh on a node with
internet); never touches the network.

    python -m mslm.data.pretokenize          # values come from the exported config
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np

from mslm.settings import Settings

_TOK = None


def _init(name):
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(name)


def _encode(texts):
    ids = _TOK(texts, add_special_tokens=False)["input_ids"]
    bos, eos = _TOK.bos_token_id, _TOK.eos_token_id
    return [np.asarray([bos] + x + [eos], dtype=np.uint16) for x in ids]


def local_parquet_files(dataset: str, config: str) -> list[str]:
    """Parquet files of ``config`` already in the HF cache (sorted, offline)."""
    import glob
    from huggingface_hub import snapshot_download
    root = snapshot_download(dataset, repo_type="dataset", local_files_only=True)
    prefix = f"sample/{config.split('-', 1)[1]}" if config.startswith("sample-") else "data"
    files = sorted(glob.glob(os.path.join(root, prefix, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}/{prefix}; run setup/download_assets.sh first")
    return files


def _batches(files, n):
    import pyarrow.parquet as pq
    for path in files:
        for rb in pq.ParquetFile(path).iter_batches(batch_size=n, columns=["text"]):
            yield rb.column(0).to_pylist()


class ShardWriter:
    def __init__(self, out_dir, prefix, shard_tokens):
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, shard_tokens
        self.idx, self.n_in_shard, self.total, self.f = 0, 0, 0, None
        self.files = []

    def write(self, arr):
        while len(arr):
            if self.f is None:
                name = f"{self.prefix}_{self.idx:03d}.bin"
                self.f = open(os.path.join(self.out_dir, name + ".tmp"), "wb")
                self.files.append(name)
            take = min(len(arr), self.shard_tokens - self.n_in_shard)
            arr[:take].tofile(self.f)
            self.n_in_shard += take
            self.total += take
            arr = arr[take:]
            if self.n_in_shard >= self.shard_tokens:
                self.close()

    def close(self):
        if self.f is not None:
            self.f.close()
            name = self.files[-1]
            os.replace(os.path.join(self.out_dir, name + ".tmp"), os.path.join(self.out_dir, name))
            self.f, self.n_in_shard = None, 0
            self.idx += 1


def main():
    s = Settings.from_env()
    out = s.str("TRAIN_DATA_DIR")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "meta.json")):
        print(f"[pretok] {out}/meta.json exists; delete the folder to rebuild")
        return
    from transformers import AutoTokenizer
    tok_name = s.str("TOKENIZER")
    tok = AutoTokenizer.from_pretrained(tok_name)
    if len(tok) > 65535:
        raise ValueError("uint16 shards need vocab <= 65535")
    files = local_parquet_files(s.str("HF_DATASET"), s.str("HF_DATASET_CONFIG"))
    print(f"[pretok] {len(files)} parquet files, tokenizer {tok_name}", flush=True)
    max_tokens, val_tokens = s.int("PRETOK_MAX_TOKENS"), s.int("PRETOK_VAL_TOKENS")
    val = ShardWriter(out, "val", val_tokens)
    train = ShardWriter(out, "train", s.int("PRETOK_SHARD_TOKENS"))
    t0, docs = time.time(), 0
    with Pool(s.int("PRETOK_WORKERS"), initializer=_init, initargs=(tok_name,)) as pool:
        for encoded in pool.imap(_encode, _batches(files, 256), chunksize=4):
            for arr in encoded:
                docs += 1
                # whole documents go to val until it is full, so val/train never share a document
                if val.total < val_tokens:
                    val.write(arr[: val_tokens - val.total])
                else:
                    train.write(arr[: max_tokens - train.total])
            if train.total >= max_tokens:
                break
            if docs % 100000 < 256:
                rate = (val.total + train.total) / (time.time() - t0)
                print(f"[pretok] docs={docs} train={train.total/1e9:.3f}B val={val.total/1e6:.1f}M {rate/1e6:.2f}M tok/s", flush=True)
    val.close()
    train.close()
    meta = dict(tokenizer=tok_name, vocab_size=len(tok), bos_id=tok.bos_token_id, eos_id=tok.eos_token_id,
                dtype="uint16", source=f"{s.str('HF_DATASET')}:{s.str('HF_DATASET_CONFIG')}",
                train_files=train.files, val_files=val.files, train_tokens=train.total,
                val_tokens=val.total, documents=docs)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pretok] done: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
