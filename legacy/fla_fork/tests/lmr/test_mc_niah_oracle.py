import os, sys
import torch

ANA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "lmr", "analysis", "260725_mc_niah_analysis"))
sys.path.insert(0, ANA)
from oracle import inject_gold  # noqa: E402


def test_inject_gold_forces_missing_gold():
    # T=3 tokens, k=2. segment_ids: 토큰이 속한 segment
    top_idx = torch.tensor([[[0, 1], [0, 1], [2, 0]]])          # [1,3,2]
    top_sc = torch.tensor([[[5.0, 3.0], [5.0, 3.0], [4.0, 2.0]]])
    online = torch.tensor([[1.0, 6.0, 1.0]])
    seg_ids = torch.tensor([3, 3, 4])
    idx, sc = inject_gold(top_idx, top_sc, online, gold=2, segment_ids=seg_ids)
    # t=0: gold(2) 미포함 → 마지막 슬롯 교체, score = max(top.max, online) = 5
    assert idx[0, 0].tolist() == [0, 2] and sc[0, 0, 1] == 5.0
    # t=1: online이 최고(6) → 강제 슬롯 score도 6 (gate에서 열세 방지)
    assert idx[0, 1].tolist() == [0, 2] and sc[0, 1, 1] == 6.0
    # t=2: gold 이미 선택됨 → 불변
    assert idx[0, 2].tolist() == [2, 0] and torch.equal(sc[0, 2], top_sc[0, 2])


def test_inject_gold_respects_eligibility():
    top_idx = torch.tensor([[[0, 1]]]); top_sc = torch.tensor([[[5.0, 3.0]]])
    online = torch.tensor([[1.0]])
    idx, sc = inject_gold(top_idx, top_sc, online, gold=2,
                          segment_ids=torch.tensor([2]))   # gold==현재 seg → 미래/현재는 불가
    assert idx[0, 0].tolist() == [0, 1] and torch.equal(sc, top_sc)
