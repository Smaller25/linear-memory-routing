# Running the MC-GDN2 routing experiments on a single A100

Everything here fits on one A100 80GB. The three things you can run are a
370M evaluation (hours), a 50M from-scratch pretrain (about a day per arm),
and a routing-hit measurement (minutes). Setup to first number is roughly an
hour, most of it downloading.

## The box

The original runs were on a VESSL pod, `betelgeuse.cloud.vessl.ai`, A100-SXM4
80GB, one GPU. **The image tag was never written down.** What matters is the
fingerprint, measured on the box that produced every number in `report/`:

| | |
|---|---|
| python | 3.13, at `/opt/conda/bin/python` |
| torch | 2.9.1+cu128 |
| triton | 3.5.1 |
| transformers | 5.17.0 |
| GPU | A100-SXM4-80GB |

Any CUDA 12.x image with torch >= 2.7 should work; nothing in the code is
pinned to 2.9. `setup_a100.sh` prints the versions it ends up with, so compare
against the table before trusting a divergent result.

Two directories matter and they are not interchangeable. Put the repo and
every cache on **container-local disk** — network mounts are slow for caches
and unreliable for git. Put checkpoints and results on a **mount that outlives
the container**. A pod that kept checkpoints locally took 22 GPU-hours of
training with it when it stopped.

## Setup

```bash
git clone -b sh/mc-routing-tracks https://github.com/Smaller25/linear-memory-routing
cd linear-memory-routing

export REPO=$PWD
export PERSIST=/root/smaller/mc      # must survive the container
export CACHE=/root/cache             # container-local is fine

bash dsc/scripts/setup_a100.sh
bash dsc/scripts/fetch_a100.sh       # 20 GB, no GPU needed
```

`setup_a100.sh` ends by printing whether `chunk_gla_fwd_o_gk` has `use_exp2`
and `transpose_state_layout`. If either says MISSING, stop — nothing will run,
see below.

Every script needs three paths importable:

```bash
export PYTHONPATH=$REPO:$REPO/dsc:$REPO/src
```

`$REPO/src` is the one people forget. It holds the vendored RULER generator,
and without it the evaluation dies about 15 seconds in on
`ModuleNotFoundError: ruler`.

## Running things

**Routing hit rate** — the cheapest useful measurement, a few minutes, no
generation at all. It asks whether the top-k segment read includes the segment
holding the needle.

```bash
python dsc/scripts/measure_hit.py \
  --ckpt $PERSIST/ckpts/LLM-OS-Models2_mc-gdn2-370m-fineweb-edu-30b-v2-meanpool/checkpoint-30B-model-ckpt.pth \
  --config-name mc_370M --data-root $PERSIST/dk_data \
  --out $PERSIST/out/hit.jsonl \
  --arm native --topk 2 --cells 8192:4 8192:16 --seeds 42 43 44 \
  --max-samples 50 --batch-size 8 --device cuda
```

`--arm shared` needs a trained MLP router directory; use `native` for a model
that has not had one fitted.

Compare the reported hit against chance, which the script does not print:
chance is `mean(min(topk, n_segments) / n_segments)` over items, about 0.065
for topk 2 at 8192. A hit rate of 0.067 is not a low score, it is chance.

**Diverse-key NIAH** — free generation, RULER official `string_match`. About
15-18 minutes per cell of 50 items at 8192.

```bash
python dsc/scripts/diverse_key_niah_eval.py \
  --backend lit_gpt --ckpt <ckpt.pth> --config-name mc_370M \
  --data-root $PERSIST/dk_data --lengths 8192 --needles 4 16 --seeds 42 43 44 \
  --max-examples 50 --batch-size 8 --device cuda \
  --model-label mc30b --gate-label base --out-dir $PERSIST/out/base
```

Add `--oracle-routing` for the ceiling arm: routing is replaced by the segment
that actually holds the needle, everything else identical. Run it every time
you run a base arm. A base score near zero means nothing on its own — the
oracle is what tells you whether the readout has any resolution at this scale.

Pair the checkpoint with the right config or nothing loads: `mc_370M` for the
MC checkpoints, `gdn2_370M` for the vanilla ones, `mc_50M` for the ladder. The
eval hard-fails on a state-dict mismatch rather than loading partially, which
is deliberate — a partially loaded model still produces plausible scores.

The full protocol table (vanilla / MC / oracle / dense / chained arms in one
pass) is `dsc/scripts/run_protocol_baseline_pod.sh`.

**50M from-scratch pretrain** — about 22 hours per arm at 15K tokens/s on an
otherwise idle A100. Two arms, sequential.

```bash
bash dsc/scripts/run_fromscratch_chained_pod.sh
```

The two arms differ in exactly one config field, `mc_checkpoint_mode`:
`independent` resets the recurrent state every 256 tokens, which is the
deployed behaviour, and `chained` threads it across segments. Everything else
is the 370M anchor's recipe — global batch 128 x 4096, LR 4e-4, warmup 1%,
1.53B tokens (20 per total parameter), TinyLlama tokenizer, FineWeb-Edu.

`MICRO_BATCH_SIZE=2` keeps peak allocation near 12 GB instead of the 19.6 GB
that micro 4 measured, so a co-tenant on the same GPU does not take the run
down. Global batch is unchanged; gradient accumulation absorbs it.

wandb is optional. Write the key to `/root/.wandb_key` (chmod 600) and the run
logs to project `llm_next_gen`. No key means `WANDB_MODE=disabled` and the run
continues rather than dying 24 hours in.

The scripts hardcode `/root/work/lmr` in a few places. Either clone there or
`sed -i 's#/root/work/lmr#'"$REPO"'#g' dsc/scripts/run_*.sh`.

## What a working run looks like

From the 370M anchor, fixed protocol, 8192 tokens, 50 items per cell, seeds
42/43/44, topk 2:

| arm | N=4 | N=16 | pooled |
|---|---|---|---|
| vanilla 30B | 17.3 | 4.7 | 11.00 |
| MC-SSC 30B | 2.0 | 3.3 | 2.67 |
| oracle routing | 76.7 | 71.3 | 74.00 |

Routing hit for the MC 30B base arm is about 0.05 at layer 0 and 0.067
averaged over layers, against a chance of 0.065.

From the 50M from-scratch `independent` arm, 1.53B tokens: final validation
loss 3.287 / 3.254 / 3.333 / 3.410 at 4096 / 8192 / 12288 / 16384, and NIAH
0.00 for both base and oracle. That zero is not a result about routing. The
model emits prose and never attempts a number — it cannot follow the
instruction format at this size and token budget, which puts free-generation
NIAH below its floor. Read the generated text in
`<out-dir>/<label>/per_sample/*.jsonl` before drawing any conclusion from a
score, at any scale.

## Things that cost us time

**The fla version decides whether anything runs at all.** `chunk_gdn2.py`
calls `chunk_gla_fwd_o_gk(use_exp2=True, transpose_state_layout=...)`. pypi
`flash-linear-attention==0.5.1` does not have those kwargs, and neither does
the `fla/` vendored at the head of this fork. Commit `4b02d15d` does — that is
the pin, and `setup_a100.sh` installs it. The symptom is
`TypeError: chunk_gla_fwd_o_gk() got an unexpected keyword argument 'use_exp2'`
about 15 seconds into any forward pass. If you reinstall fla by hand, delete
`site-packages/fla` first: a stale `fla/utils.py` left beside `fla/utils/`
produces a circular import that looks unrelated.

**Never `pgrep -f` or `pkill -f` a script name over ssh.** The pattern matches
the ssh command line carrying it, so the check finds itself and a kill finds
its own session. This bit us three times, once killing the monitoring session
and once stalling a run 27 minutes. Use an explicit completion marker in the
log and grep for that.

**The eval buffers per-sample output until a cell ends.** An empty
`per_sample/*.jsonl` 15 minutes in is normal, not a hang. Progress prints only
at `[cell]` lines. Check `/proc/<pid>/fdinfo` if you need to know it is alive.

**A cell that overwrites its checkpoint.** `pretrain.py` writes only
`latest-model-ckpt.pth`, replaced at every save. To evaluate mid-training, copy
it first; evaluating the live file can read a half-written one.

**Pull results off the box as they appear.** Ephemeral containers take
everything. Checkpoints are re-downloadable from HuggingFace and FineWeb-Edu
is re-downloadable, but a training run is not.
