# Memory Caching for GDN-2

Canonical GDN-2의 `[B,T,H,K]` query/key와 `[B,H,K,V]` recurrent state를
공통 SSC/GRM/Memory Soup 식에 연결한다.

- `ssc.py`: pre-projected tensor adapter. 원래 `chunk_gdn2`로 각 segment를 압축하고
  cache state를 직접 `M_i(q_t)`로 읽는다.
- `grm.py`: dense GRM adapter. 완료된 모든 segment를 softmax 가중합한다.
  GDN-2에서는 Memory Soup와 수학적으로 동일하므로 두 이름이 이 경로를 공유한다.
- `layer.py`: 검증된 기존 SSC v2 wrapper. GRM/Soup 추가로 수정하지 않는다.
- `dense_layer.py`: GRM/Soup 전용으로 분리된 full-sequence wrapper.
- GDN-2 kernel과 동일하게 cached read query에 L2 normalization과 `1/sqrt(K)` scale 적용.
- 추가 학습 파라미터: variant별 connector `W_u` 하나.

```python
from dsc.mc_gdn2 import MemoryCachingGDN2Layer
from dsc.mc_gdn2.dense_layer import DenseMemoryCachingGDN2Layer

ssc = MemoryCachingGDN2Layer(
    base_gdn2_layer,
    topk=2,
    chunk_size=256,
)
grm = DenseMemoryCachingGDN2Layer(
    base_gdn2_layer,
    variant="grm",
    chunk_size=256,
)
soup = DenseMemoryCachingGDN2Layer(
    base_gdn2_layer,
    variant="memory_soup",
    chunk_size=256,
)
```

현재 wrapper는 unpadded full-sequence 학습/평가 경로만 허용한다. LIVE GDN-2 training
체크포인트는 수정하지 않았다. GRM/Memory Soup 구현과 사용법의 상세 설명은
루트의 `MC_GRM_MEMORY_SOUP_GDN2_V2_KO.md`를 참고한다.
