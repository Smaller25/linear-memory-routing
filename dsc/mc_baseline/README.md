# Memory Caching SSC — 공통 canonical core

논문 **Memory Caching: RNNs with Growing Memory**의 Sparse Selective Caching
(SSC) 식 (16)–(17)을 backbone 독립적으로 구현한다. 최신 공개본 기준은
[arXiv:2602.24281](https://arxiv.org/abs/2602.24281)이다.

> 주의: `arXiv:2506.04761`은 Memory Caching이 아니라 **Log-Linear Attention**이다.
> OpenReview 원 제출 ID는 `R3EJ2IjgOI`, 수정본은 `B5SkWFRE8U`로 확인된다.

## 구현 식

```text
u_t = x_t W_u
c_i = sum_{j in segment i} k_j
r_t^i = <u_t, c_i>
R_t = TopK({r_t^i | i < current_segment(t)})
y_t = gamma_t^online M_t(q_t)
      + sum_{i in R_t} gamma_t^i M_i(q_t)
```

`gamma`는 online + 선택된 cache 전체에 softmax한다. 현재 segment의 gate는
미래 토큰을 보지 않도록 `k`의 segment-local cumulative sum을 사용한다. 과거 cache
read는 GDN state를 current chunk로 다시 돌리지 않고 행렬 메모리 `M_i(q_t)`를 직접
평가한다.

## 파일

| 파일 | 역할 |
|---|---|
| `mc_ssc.py` | 학습 가능한 `W_u`, key-sum descriptor, per-token top-k, direct readout |
| `segment_checkpoint.py` | independent compressor / continuous checkpoint 두 모드 |
| `diagnostics.py` | online/cache weight, routing entropy, output norm |
| `tests/` | 수식, shape, gradient, strict causality, GDN adapter smoke test |

GDN별 구현은 [`../mc_gdn1/`](../mc_gdn1/)과
[`../mc_gdn2/`](../mc_gdn2/)에 분리되어 있다. 이 폴더는 다른 MC variant
(Residual/GRM/Soup)를 구현하지 않는다.

## 검증

```bash
python -m dsc.mc_baseline.tests.test_shape
python -m dsc.mc_baseline.tests.test_causality
```

상세 감사 결과와 DSC 버전 비교는
[`../docs/MC_TO_DSC_V3_PLUS.md`](../docs/MC_TO_DSC_V3_PLUS.md), 재현·merge 절차는
[`../docs/MC_SSC_REPRO_AND_MERGE.md`](../docs/MC_SSC_REPRO_AND_MERGE.md)를 따른다.
