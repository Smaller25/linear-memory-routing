# Linear Memory Routing (`lmr`)

Memory Caching (MC) on top of FLA linear-attention models. Linear models compress a sequence
into a fixed-size recurrent state and under-recall; MC adds memory capacity **without leaving
linear complexity** by caching frozen recurrent-state checkpoints across segments and combining
them at read-out.

## How it works (Mamba2)

The model is run in segments of `chunk_size` tokens. Each segment runs from a **zero** SSM state;
at each boundary the final state is **frozen and cached**. For later segments, each cached
checkpoint's contribution to the output is recovered by re-running the scan on the *current*
segment with **input zeroed** and the checkpoint injected as the initial state — exact by
linearity of the SSD scan in `(input, state)`:

```
y_RM = scan(x_s, init=0) + Σ_{i<s} scan(x=0, init=h_L^(i))     # summed pre-norm/gate
```

Read-out heads (`lmr/readout.py`):
- **RM**  — plain residual sum. *Training-free, no parameters.*
- **GRM** — per-token sigmoid gate `γ = σ(⟨u_t, meanpool(h_L^(i))⟩)`, `u_t = x_t W_u`.
- **SSC** — top-k router over checkpoints + Switch-style load-balance aux loss.

A **MoM** (Mixture-of-Memories) comparison arm is wired via `lmr/mom_adapter.py`.

## Layout
- `ssd_scan.py` — pure-torch SSD scan with `initial_states` (CPU reference; the RM substrate).
- `segment_runner.py` — segment loop over an FLA `Mamba2ForCausalLM`; CPU (`naive`) or CUDA
  (`mamba_chunk_scan_combined`) backend.
- `readout.py` — RM / GRM / SSC heads.
- `converter.py` — `state-spaces/mamba2-*` → FLA `Mamba2ForCausalLM` (+ GPU logit-match).
- `state_utils.py`, `mom_adapter.py`, `tasks/` (MQAR + passkey generators), `scripts/`.

## Verify (CPU, no GPU)
```
pytest tests/lmr -q       # scan linearity, RM≡vanilla at N=1, runner==FLA forward, readout shapes
```
GPU/local only: `lmr/scripts/convert_mamba2.py` (logit-match), `lmr/scripts/run_baseline.py`
(vanilla vs +MC-RM recall curves), `train_variant.py`, `eval_real.py`.
