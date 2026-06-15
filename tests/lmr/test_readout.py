# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the RM / GRM / SSC read-out heads."""

import torch

from lmr.readout import (
    GatedResidualMemory,
    ResidualMemory,
    SparseSelectiveCaching,
    build_readout,
)


def _setup(b=2, seqlen=6, h=4, p=8, n=16, num_ckpt=3, hidden=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    y_main = torch.randn(b, seqlen, h, p, generator=g)
    y_cached = [torch.randn(b, seqlen, h, p, generator=g) for _ in range(num_ckpt)]
    x = torch.randn(b, seqlen, hidden, generator=g)
    dd = h * n
    descriptors = torch.randn(b, num_ckpt, dd, generator=g)
    return y_main, y_cached, x, descriptors, hidden, dd


def test_rm_no_cache_is_identity():
    rm = ResidualMemory()
    y_main, _, x, descr, _, _ = _setup()
    y, aux = rm(y_main, [], x, None)
    assert aux is None
    assert torch.equal(y, y_main)


def test_rm_is_plain_sum():
    rm = ResidualMemory()
    y_main, y_cached, x, descr, _, _ = _setup()
    y, _ = rm(y_main, y_cached, x, descr)
    assert torch.allclose(y, y_main + sum(y_cached))


def test_grm_shape_and_gate_range():
    y_main, y_cached, x, descr, hidden, dd = _setup()
    grm = GatedResidualMemory(hidden, dd)
    y, aux = grm(y_main, y_cached, x, descr)
    assert aux is None
    assert y.shape == y_main.shape


def test_ssc_topk_and_aux():
    y_main, y_cached, x, descr, hidden, dd = _setup()
    ssc = SparseSelectiveCaching(hidden, dd, topk=2)
    y, aux = ssc(y_main, y_cached, x, descr)
    assert y.shape == y_main.shape
    assert aux is not None and aux.ndim == 0 and torch.isfinite(aux)


def test_build_readout():
    assert isinstance(build_readout("rm", 32, 64), ResidualMemory)
    assert isinstance(build_readout("grm", 32, 64), GatedResidualMemory)
    assert isinstance(build_readout("ssc", 32, 64, topk=2), SparseSelectiveCaching)
