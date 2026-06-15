# Experiment log — linear-memory-routing

| date | phase | what | status |
|------|-------|------|--------|
| 2026-06-15 | 0 (scaffold) | MC library (`lmr/`): ssd_scan, segment_runner, readout (RM/GRM/SSC), converter, MoM adapter, MQAR/passkey gens, scripts. CPU tests for scan linearity, RM≡vanilla@N=1, runner==FLA forward, readout shapes. | code complete; CPU tests green; GPU experiments pending local A100 |
| 2026-06-15 | 0 (A100 validate) | Ran GPU tests (14 pass, 1 non-blocking MoM-arm fail) + converter logit-match (FLA≡mamba_ssm: argmax 100%, r=1.0) + Phase-0 baseline. Fixed 3 bugs: `_tied_weights_keys` list→dict (transformers 5.12), vocab pad 50277→50288, atol 1e-3→1e-2. **Training-free MC-RM neutral-to-negative** (mqar Δ0; passkey@2048 0.253→0.022; chunk64 MQAR Δ−0.081) — frozen model not trained for segmented/summed readout + synthetic tasks OOD for pretrained LM. → Phase 1 (train GRM/SSC) + real-text passkey. | verified; see `report/0002.md` |
