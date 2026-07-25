import importlib.util, os, sys
import pytest

ANA = os.path.join(os.path.dirname(__file__), "..", "..",
                   "lmr", "analysis", "260725_mc_niah_analysis")
sys.path.insert(0, os.path.abspath(ANA))
import data as mcdata  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")


def test_annotate_finds_gold_needle(tok):
    filler = "The grass is green. The sky is blue. " * 120
    needle = "One of the special magic numbers for apple-pie is: 7301562."
    text = (filler[:2000] + " " + needle + " " + filler[2000:4000]
            + "\nWhat is the special magic number for apple-pie mentioned in the provided text?")
    ann = mcdata.annotate(text, tok)
    assert ann["query_key"] == "apple-pie"
    assert len(ann["needles"]) == 1
    n = ann["needles"][0]
    assert n["value"] == "7301562"
    # 토큰 위치가 실제 needle 문자 위치와 일치하는 segment를 가리킴
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    char_pos = text.find("7301562")
    tok_idx = next(i for i, (s, e) in enumerate(enc.offset_mapping) if s <= char_pos < e)
    assert n["seg"] == tok_idx // 256 == ann["gold_seg"]


def test_annotate_multikey_picks_queried(tok):
    needles = [f"One of the special magic numbers for key-{i} is: 100000{i}." for i in range(4)]
    filler = "The grass is green. " * 60
    text = (" ".join([filler, needles[0], filler, needles[1], filler, needles[2],
                      filler, needles[3], filler])
            + "\nWhat is the special magic number for key-2 mentioned in the provided text?")
    ann = mcdata.annotate(text, tok)
    assert ann["query_key"] == "key-2"
    assert len(ann["needles"]) == 4
    assert ann["gold_seg"] == next(n["seg"] for n in ann["needles"] if n["key"] == "key-2")


def test_paired_gen_invariants(tok):
    import paired_gen
    rows = paired_gen.build_pairs(tok, n_pairs=3, condition="S", seed=7) \
         + paired_gen.build_pairs(tok, n_pairs=3, condition="D", seed=7)
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["condition"], r["pair_id"]), {})[r["variant"]] = r
    assert len(by_pair) == 6
    for (cond, _), pair in by_pair.items():
        s, m = pair["single"], pair["multi"]
        # 쌍 불변식: 같은 needle/answer/gold segment
        assert s["needle_key"] == m["needle_key"] and s["outputs"] == m["outputs"]
        assert s["gold_seg"] == m["gold_seg"]
        # annotate로 실측 재검증
        ann_m = mcdata.annotate(m["input"], tok)
        assert ann_m["gold_seg"] == m["gold_seg"]
        assert ann_m["query_key"] == m["needle_key"]
        assert len(ann_m["needles"]) == 4          # gold + distractor 3
        dsegs = sorted(n["seg"] for n in ann_m["needles"] if n["key"] != m["needle_key"])
        assert dsegs == sorted(m["distractor_segs"])
        if cond == "S":
            assert m["gold_seg"] in m["distractor_segs"]      # 1개는 같은 segment
            assert m["codist_tok_dist"] is not None
        else:
            assert m["gold_seg"] not in m["distractor_segs"]  # 전부 다른 segment
        ann_s = mcdata.annotate(s["input"], tok)
        assert len(ann_s["needles"]) == 1
        # 길이 제약: 전체 ≤ 2048-128
        assert ann_s["n_tok"] <= 1920 and ann_m["n_tok"] <= 1920


def test_paired_gen_shared_haystack(tok):
    """single/multi가 같은 haystack을 공유하는지: 전체 토큰 수 일치, gold 위치·segment
    완전 동일, distractor 문장이 들어가는 span을 제외한 나머지 토큰이 완전히 동일한지."""
    import paired_gen
    rows = paired_gen.build_pairs(tok, n_pairs=3, condition="S", seed=11) \
         + paired_gen.build_pairs(tok, n_pairs=3, condition="D", seed=11)
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["condition"], r["pair_id"]), {})[r["variant"]] = r
    assert len(by_pair) == 6

    def tok_at(offsets, p):
        return next(i for i, (s, e) in enumerate(offsets) if s <= p < e)

    for (_cond, _pid), pair in by_pair.items():
        s, m = pair["single"], pair["multi"]
        enc_s = tok(s["input"], return_offsets_mapping=True, add_special_tokens=False)
        enc_m = tok(m["input"], return_offsets_mapping=True, add_special_tokens=False)
        # (a) 전체 토큰 수 완전 일치
        assert len(enc_s.input_ids) == len(enc_m.input_ids)

        ann_s = mcdata.annotate(s["input"], tok)
        ann_m = mcdata.annotate(m["input"], tok)
        gold_s = next(n for n in ann_s["needles"] if n["key"] == s["needle_key"])
        gold_m = next(n for n in ann_m["needles"] if n["key"] == m["needle_key"])
        # (b) gold의 실제 토큰 시작 위치·segment가 single/multi 간 완전 동일
        assert gold_s["tok_start"] == gold_m["tok_start"]
        assert gold_s["seg"] == gold_m["seg"]

        # (c) distractor 문장이 차지하는 토큰 span을 제외하면 나머지 토큰 시퀀스가 완전 동일
        distractor_matches = [mm for mm in mcdata.NEEDLE_RE.finditer(m["input"])
                               if mm.group(1) != m["needle_key"]]
        assert len(distractor_matches) == len(m["distractor_segs"])
        excluded = set()
        for mm in distractor_matches:
            lo = tok_at(enc_m.offset_mapping, mm.start())
            hi = tok_at(enc_m.offset_mapping, mm.end() - 1)
            excluded.update(range(lo, hi + 1))
        outside = [i for i in range(len(enc_m.input_ids)) if i not in excluded]
        assert outside, "no tokens outside distractor spans to compare"
        for i in outside:
            assert enc_s.input_ids[i] == enc_m.input_ids[i], (
                f"token mismatch outside distractor span at index {i}")
