# -*- coding: utf-8 -*-
"""Packed-document dataloader over the tokenized SlimPajama binaries.

Input format (produced by dynmc/tokenize_slimpajama.py on the VESSL CPU node):
    tokens-{shard:05d}.bin   uint16 token stream, docs concatenated, BOS-first
    doclens-{shard:05d}.npy  uint32 per-doc token counts
    sources-{shard:05d}.npy  uint8 per-doc source ids
    (fineweb-tokens.bin / fineweb-doclens.npy for the 46M validation runs)

An "epoch plan" (built by build_upsample_plan.py, task #2) is a uint64 array of
global doc ids, in sampling order, possibly with repeats (length upsampling).
This loader walks the plan, concatenates docs into a token stream, cuts it into
ctx-length rows, and tracks intra-row document boundaries. Doc pieces created
by a row cut are treated as independent documents (state reset at the cut —
unavoidable under row packing; consistent for all runs being compared).

Yields per micro-batch (see train.py):
    input_ids [rows, ctx] int64
    labels    [rows, ctx] int64 (-100 at each doc piece's last token)
    doc_lens_per_row: list[list[int]] — for segmenting.build_batch_segments
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch


class TokenizedCorpus:
    """Memory-mapped view over all shards + global doc index."""

    def __init__(self, data_dir: str, prefix: str = ""):
        pat_bin = os.path.join(data_dir, f"{prefix}tokens-*.bin")
        self.bins = sorted(glob.glob(pat_bin))
        if not self.bins:  # 단일 파일 형식 (val/fineweb)
            single = os.path.join(data_dir, f"{prefix}tokens.bin")
            assert os.path.exists(single), f"no tokens under {data_dir} ({prefix})"
            self.bins = [single]
        self.maps = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.bins]
        lens, srcs, shard_of, offs = [], [], [], []
        for si, p in enumerate(self.bins):
            stem = p.replace("tokens", "doclens").replace(".bin", ".npy")
            dl = np.load(stem)
            src_p = p.replace("tokens", "sources").replace(".bin", ".npy")
            src = np.load(src_p) if os.path.exists(src_p) else np.full(len(dl), 255, np.uint8)
            off = np.zeros(len(dl), dtype=np.uint64)
            np.cumsum(dl[:-1], out=off[1:])
            lens.append(dl); srcs.append(src); offs.append(off)
            shard_of.append(np.full(len(dl), si, dtype=np.uint16))
        self.doc_len = np.concatenate(lens)          # [D] uint32
        self.doc_src = np.concatenate(srcs)          # [D] uint8
        self.doc_off = np.concatenate(offs)          # [D] uint64 (shard-local)
        self.doc_shard = np.concatenate(shard_of)    # [D] uint16
        self.num_docs = len(self.doc_len)
        self.total_tokens = int(self.doc_len.sum())

    def doc_tokens(self, doc_id: int) -> np.ndarray:
        m = self.maps[self.doc_shard[doc_id]]
        o = int(self.doc_off[doc_id])
        return m[o:o + int(self.doc_len[doc_id])]


class PackedDocIterator:
    """Walk an epoch plan, emit ctx-length rows with doc-boundary metadata.

    Resumable: state is (plan_pos, carry) — we only checkpoint plan_pos and
    re-derive by fast-forwarding rows (cheap, index-only).
    """

    def __init__(self, corpus: TokenizedCorpus, plan: np.ndarray, ctx: int,
                 start_row: int = 0):
        self.corpus = corpus
        self.plan = plan
        self.ctx = ctx
        self.pos = 0            # index into plan
        self.carry: list[tuple[int, int, int]] = []  # (doc_id, start, len) pending
        if start_row:
            self._skip_rows(start_row)

    def _next_pieces(self):
        """Generator of (doc_id, start, length) covering the plan in order."""
        while self.pos < len(self.plan):
            d = int(self.plan[self.pos])
            self.pos += 1
            yield (d, 0, int(self.corpus.doc_len[d]))

    def _skip_rows(self, n: int):
        for _ in range(n):
            self.next_row(materialize=False)

    def next_row(self, materialize: bool = True):
        """One ctx-length row. Returns (ids int64 [ctx], doc_lens list[int]) or None."""
        need = self.ctx
        pieces: list[tuple[int, int, int]] = []
        while need > 0:
            if self.carry:
                d, s, ln = self.carry.pop()
            else:
                try:
                    d, s, ln = next(self._gen)
                except AttributeError:
                    self._gen = self._next_pieces()
                    continue
                except StopIteration:
                    break
            take = min(ln, need)
            pieces.append((d, s, take))
            need -= take
            if take < ln:
                self.carry.append((d, s + take, ln - take))
        if need > 0:  # plan exhausted; drop incomplete row
            return None
        doc_lens = [p[2] for p in pieces]
        if not materialize:
            return (None, doc_lens)
        ids = np.empty(self.ctx, dtype=np.int64)
        w = 0
        for d, s, ln in pieces:
            ids[w:w + ln] = self.corpus.doc_tokens(d)[s:s + ln]
            w += ln
        return (ids, doc_lens)


def make_batch(it: PackedDocIterator, rows: int):
    """rows × next_row → tensors + doc metadata; None when the plan runs out."""
    id_rows, doc_lens_rows = [], []
    for _ in range(rows):
        r = it.next_row()
        if r is None:
            return None
        id_rows.append(r[0])
        doc_lens_rows.append(r[1])
    input_ids = torch.from_numpy(np.stack(id_rows))          # [rows, ctx]
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    for ri, dls in enumerate(doc_lens_rows):                 # doc 마지막 토큰 마스킹
        end = 0
        for dl in dls:
            end += dl
            labels[ri, end - 1] = -100
    return {"input_ids": input_ids, "labels": labels,
            "doc_lens_per_row": doc_lens_rows}


def uniform_plan(corpus: TokenizedCorpus, target_tokens: int, seed: int = 0) -> np.ndarray:
    """단순 셔플 plan (46M validation용 등). 본 실험은 build_upsample_plan.py 사용."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(corpus.num_docs)
    csum = np.cumsum(corpus.doc_len[order].astype(np.int64))
    cut = int(np.searchsorted(csum, target_tokens)) + 1
    return order[:cut].astype(np.uint64)
