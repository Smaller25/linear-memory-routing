# Vendored: NVIDIA RULER

`gen/` (synthetic data generators, tokenizer/template/manifest helpers) and
`eval_metrics.py` are vendored from NVIDIA's RULER benchmark (Apache-2.0):

- `gen/`           ← RULER `scripts/data/` (prepare.py, tokenizer.py,
                     manifest_utils.py, template.py, synthetic/*)
- `synthetic.yaml` ← RULER `scripts/synthetic.yaml` (task definitions)
- `eval_metrics.py`← RULER `scripts/eval/synthetic/constants.py` (string-match
                     metric functions + task→metric map)

Source: https://github.com/NVIDIA/RULER
Paper: Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context
Language Models?" (2024), https://arxiv.org/abs/2404.06654

Changes from upstream:
- `gen/synthetic/json/english_words.json` is the real 8.5 MB word list fetched
  from Git-LFS media (the shallow clone only had the LFS pointer).
- RULER's prediction clients (`scripts/pred/*`, which depend on vllm/nemo/trt)
  and the nemo-based `evaluate.py` are NOT vendored — `scripts/ruler.py`
  provides a mamba-ssm generation backend and applies the vendored metrics
  directly, avoiding the heavy inference/eval dependencies.

The generators are invoked as subprocesses by `scripts/ruler.py` (matching
RULER's own prepare.py design), so their internal relative imports are
unchanged.
