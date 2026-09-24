 # Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import chunked_cross_entropy, num_parameters
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from lit_gpt import FusedCrossEntropyLoss
import random
import os
import argparse
from data import get_stateful_stream_tok_dataset
import time
import torch.multiprocessing as mp
import shutil
from distutils.dir_util import copy_tree

_TRAIN_START_TIME = time.time()

os.environ["TRITON_CACHE_MANAGER"] = "cache:ParallelFileCacheManager"


def _hf_upload_checkpoint(local_path, repo_id, token, blocking=True, run_dir=None):
    """Upload a checkpoint file (and optional run metadata) to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)
        # Ensure repo exists (public by default — user policy: all HF uploads are public).
        create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=token)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=repo_id,
            repo_type="model",
        )
        # Also push a snapshot of the run directory (config + bash scripts).
        if run_dir is not None and os.path.isdir(run_dir):
            for fname in ("lit_gpt", "bash_scripts"):
                src = os.path.join(run_dir, fname)
                if os.path.isdir(src):
                    api.upload_folder(
                        folder_path=src,
                        path_in_repo=fname,
                        repo_id=repo_id,
                        repo_type="model",
                        ignore_patterns=["*.pth", "*.pt", "*.bin"],
                    )
        print(f"[HF] Uploaded {local_path} -> {repo_id}", flush=True)
        return True
    except Exception as e:
        print(f"[HF] Upload failed for {local_path}: {e}", flush=True)
        return False


def _hf_save_model_hf_format(args, state, fabric, repo_id, token, final=False):
    """Save model in HF format (safetensors) for downstream loading."""
    try:
        from huggingface_hub import HfApi
        from transformers import AutoConfig, AutoTokenizer
        # Use lit_gpt's built-in HF conversion if available, else skip.
        save_dir = os.path.join(args.out_dir, f"hf_format_{'final' if final else 'step' + str(state['step_count'])}")
        os.makedirs(save_dir, exist_ok=True)
        # Write a minimal config.json
        cfg = state["model"].config
        cfg_dict = {
            "architectures": ["GatedDeltaNet2ForCausalLM"],
            "model_type": "gdn2",
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "vocab_size": cfg.padded_vocab_size,
            "block_size": cfg.block_size,
            "intermediate_size": cfg.intermediate_size,
            "gdn2_per_layer": cfg.gdn2_per_layer,
            "local_window": cfg.local_window,
            "torch_dtype": "bfloat16",
        }
        import json
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=save_dir,
            path_in_repo=f"checkpoint-{'final' if final else state['step_count']}/hf_format",
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["*.pt", "*.bin", "*.safetensors"],
        )
        return True
    except Exception as e:
        print(f"[HF] HF-format save failed: {e}", flush=True)
        return False


def main(args):
    if args.debug:
        wandb_logger = WandbLogger(project="llm_next_gen", mode='disabled', name=args.exp_name, id=args.exp_name, save_dir=args.wandb_dir, dir=args.wandb_dir, version=args.exp_name, group="debug")
    else:
        wandb_logger = WandbLogger(project="llm_next_gen", name=args.exp_name, id=args.exp_name, save_dir=args.wandb_dir, dir=args.wandb_dir, version=args.exp_name, group=args.exp_group)

    if args.interactive_job:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full")
    else:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy='HYBRID_SHARD')
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed", loggers=[wandb_logger])
    fabric.launch()
    # fix seed in the very beginning
    fabric.seed_everything(args.seed)  # same seed for every process to init model (FSDP)
    fabric.print("##### Infra Details #####")
    fabric.print(f"Number of Nodes: {args.nodes}")
    fabric.print(f"Number of GPUs: {fabric.world_size}")
    fabric.print("##### Training Details #####")
    fabric.print(f"Maximum number of training tokens: {args.max_tokens}")
    fabric.print(f"Maximum training time: {args.actual_train_time/60.0} min")
    fabric.print(f"Micro batch size: {args.micro_batch_size}")
    fabric.print(f"Batch size: {args.batch_size}")
         
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME], device=fabric.device, dtype=torch.int64)
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()    
    if fabric.global_rank == 0:
        fabric.print(args)
    fabric.logger.log_hyperparams(args)
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=args.log_iter_interval)

    if os.path.exists(args.out_dir):
        args.resume = True
        print('Resuming from {}'.format(args.out_dir))
    else:
        if fabric.global_rank == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            target_litgpt_save_dir = os.path.join(args.out_dir, 'lit_gpt')
            target_bash_scripts_save_dir = os.path.join(args.out_dir, 'bash_scripts')
            target_pretrain_file = os.path.join(args.out_dir, 'pretrain.py')
            for d in (target_litgpt_save_dir, target_bash_scripts_save_dir):
                os.makedirs(d, exist_ok=True)
            # Snapshot the running lit_gpt + pretrain.py for reproducibility, if available.
            src_lit_gpt = os.path.join(wd, 'lit_gpt')
            if os.path.isdir(src_lit_gpt) and not args.debug:
                try:
                    copy_tree(src_lit_gpt, target_litgpt_save_dir)
                    shutil.copyfile(os.path.join(wd, "pretrain.py"), target_pretrain_file)
                except Exception as e:
                    fabric.print(f"[warn] code snapshot skipped: {e}")                
    
    over = {}
    for pair in (p for p in args.config_overrides.split(",") if p):
        if "=" not in pair:
            raise ValueError(f"--config_overrides expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        over[key] = value
    config = Config.from_name(args.model_name, **over)
    if over:
        fabric.print(f"config overrides: {over}")
    # Loud effective-batch check: micro x accum x world must reproduce the
    # recipe's global batch (paper-matched 128x4k arm => 128 seqs = 524,288
    # tokens per optimizer step) regardless of any --micro_batch_size override.
    effective_batch_seqs = args.micro_batch_size * args.gradient_accumulation_steps * fabric.world_size
    fabric.print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    fabric.print(
        f"Effective global batch: {args.micro_batch_size} micro x "
        f"{args.gradient_accumulation_steps} accum x {fabric.world_size} ranks = "
        f"{effective_batch_seqs} seqs = {effective_batch_seqs * config.block_size:,} tokens/step"
    )
    if args.use_stream_tok:
        tokenizer_source = args.tokenizer_path if args.tokenizer_path else args.tokenizer_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 999999999
        # B-4 fix: wire --seed into the streaming shuffle (was epoch+rank only).
        train_dataloader = get_stateful_stream_tok_dataset(corpus_name=args.corpus_name, path=args.train_data_dir_raw, split='train', tokenizer=tokenizer, block_size=config.block_size+1, rank=fabric.global_rank, world_size=fabric.world_size, batch_size=args.micro_batch_size, num_workers=args.train_num_workers, seed=args.seed)
        val_dataloader = None
        if args.val_data_dir_raw:
            try:
                val_dataloader = get_stateful_stream_tok_dataset(corpus_name=args.corpus_name, path=args.val_data_dir_raw, split=args.val_type, tokenizer=tokenizer, block_size=16384+1, rank=fabric.global_rank, world_size=fabric.world_size, batch_size=args.micro_batch_size // 2, num_workers=args.val_num_workers)
            except Exception as e:
                fabric.print(f"[warn] val dataset load failed, continuing without validation: {e}")
                val_dataloader = None

    else:
        train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        seed=args.seed,
        )
        if val_dataloader is None:
            train_dataloader = fabric.setup_dataloaders(train_dataloader)
        else:
            train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if args.val_type != 'val_sampled':
        val_dataloader = None
        
    if fabric.global_rank == 0:
        fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))

    if getattr(args, "init_from_ckpt", ""):
        # Finetuning init: load MODEL WEIGHTS ONLY from a foreign checkpoint
        # (e.g. an SSC 30B ckpt converted to the ReLU key layout). Optimizer,
        # scheduler, and step count start fresh; --resume still owns full
        # training-state restarts from out_dir and takes effect afterwards.
        sd = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"init_from_ckpt mismatch: {len(missing)} missing "
                f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
                f"(e.g. {unexpected[:3]}) — convert the checkpoint to this "
                "config's key layout first (see convert_ssc_ckpt_to_relu.py)")
        fabric.print(f"Initialized model weights from {args.init_from_ckpt}")


    if fabric.global_rank == 0:
        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        # we ignore the embedding & lm head parameter, which is standard 
        fabric.print(f"Total parameters {num_parameters(model.transformer.h):,}")
        fabric.print(model)
    
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), fused=True
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": args.hparams, "iter_num": 0, "step_count": 0}

    if args.resume:
        try:
            resume = os.path.join(args.out_dir, "latest-model-ckpt.pth")
            if fabric.global_rank == 0:
                fabric.print(f"Resuming training from {resume}")
            fabric.load(resume, state)
            fabric.print(f"Successfully resumed from {resume}")
        except:
            fabric.print(f"Failed to resume from {resume}")
            args.resume = False
    train_time = time.perf_counter()
    train(args, _TRAIN_START_TIME, fabric, state, train_dataloader, val_dataloader, monitor, args.resume)
    if fabric.global_rank == 0:
        fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        if fabric.global_rank == 0:
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(args, _TRAIN_START_TIME, fabric, state, train_dataloader, val_dataloader, monitor, resume):
    
    model = state["model"]
    optimizer = state["optimizer"]

    total_lengths = 0
    total_t0 = time.perf_counter()    
    max_tokens_per_device = args.max_tokens // fabric.world_size
    tokens_per_iter = args.micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    warmup_iters = args.warmup_tokens // fabric.world_size // tokens_per_iter
    initial_iter = state["iter_num"]
    curr_iter = 0
    loss_func = FusedCrossEntropyLoss()    

    if resume:
        if args.use_stream_tok:
            try:
                if fabric.world_size <= 1:
                    data_state_path = os.path.join(args.out_dir,"latest-data-state-ckpt.pth")
                else:
                    data_state_path = os.path.join(args.out_dir,f"latest-data-states-rank-{fabric.global_rank}-ckpt.pth")
                train_dataloader.load_state_dict(torch.load(data_state_path))                                                
                if fabric.global_rank == 0:
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
                resume = False
            except:
                fabric.print(f"Failed to resume dataloader from {args.out_dir}")
                raise KeyError("Failed to resume dataloader.. Please retrain from scratch.")

    tokens = 0
    train_t0 = time.perf_counter()
    
    if args.eval_before_training:
        fabric.print("Do validation before training:")
        val_loss = validate(args, fabric, model, val_dataloader, None)
        for i in range(args.num_extrapol):
            if fabric.global_rank == 0:
                fabric.print(f"step {state['iter_num']} {i+1} x: val loss {val_loss[i]:.4f}")
    
    def save_checkpoint(final=False, milestone_tokens_b=None):
        name = 'latest' if not final else 'final'
        checkpoint_path = os.path.join(args.out_dir,f"{name}-model-ckpt.pth")
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        if not final:
            fabric.save(checkpoint_path, state)
        else:
            # we are not interested in the optimizer state for the final checkpoint
            state['optimizer'] = None
            fabric.save(checkpoint_path, state)

        # Milestone (incremental) save: keep a copy at each token boundary (e.g., every 1B tokens).
        milestone_local_path = None
        if milestone_tokens_b is not None:
            milestone_local_path = os.path.join(args.out_dir, f"checkpoint-{milestone_tokens_b}B-model-ckpt.pth")
            if fabric.global_rank == 0:
                shutil.copy2(checkpoint_path, milestone_local_path)
                fabric.print(f"Milestone checkpoint saved: {milestone_local_path}")

        if args.use_stream_tok and not final:
            if fabric.world_size <= 1:
                checkpoint_path = os.path.join(args.out_dir,f"latest-data-state-ckpt.pth")
            else:
                checkpoint_path = os.path.join(args.out_dir,f"latest-data-states-rank-{fabric.global_rank}-ckpt.pth")
            torch.save(train_dataloader.state_dict(), checkpoint_path)
            fabric.print(f"Dataloader state checkpoint saved")

        # Push to HF Hub (rank 0 only). At milestone boundaries or for the final ckpt.
        if args.hf_upload and fabric.global_rank == 0 and args.hf_repo_id:
            upload_path = milestone_local_path if milestone_local_path else checkpoint_path
            if args.hf_upload_blocking:
                _hf_upload_checkpoint(upload_path, args.hf_repo_id, args.hf_token, blocking=True, run_dir=args.out_dir)
            else:
                import threading
                threading.Thread(
                    target=_hf_upload_checkpoint,
                    args=(upload_path, args.hf_repo_id, args.hf_token, True, args.out_dir),
                    daemon=True,
                ).start()

    for train_data in train_dataloader:
        # per gpu
        tokens += model.config.block_size * args.micro_batch_size
        if resume and not args.use_stream_tok:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                if fabric.global_rank == 0:
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))

        if state["iter_num"] >= max_iters:
            break
    
        iter_t0 = time.perf_counter()
        if args.use_stream_tok:
            input_ids = train_data['input_ids'][:, 0 : model.config.block_size].contiguous().to(fabric.device)
            targets = train_data['labels'][:, 1 : model.config.block_size + 1].contiguous().to(fabric.device)
        else:
            input_ids = train_data[:, 0 : model.config.block_size].contiguous()
            targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

        lr = get_lr(args, state["iter_num"], warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        is_accumulating = (state["iter_num"] + 1) % args.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            # Check if loss is NaN
            if torch.isnan(loss):
                # Create a debug directory if it doesn't exist
                debug_dir = "./logs/debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save the relevant tensors
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save({
                    'input_ids': input_ids,
                    'logits': logits,
                    'targets': targets,
                    'loss': loss
                }, os.path.join(debug_dir, f'nan_tensors_{timestamp}.pt'))
                
                print(f"NaN loss detected! Tensors saved to {debug_dir}/nan_tensors_{timestamp}.pt")
            fabric.backward(loss / args.gradient_accumulation_steps)


        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        if fabric.global_rank == 0 and state["iter_num"] % 10 == 0:
            total_tokens = model.config.block_size * state["iter_num"] * args.micro_batch_size * fabric.world_size / 1e9
            fabric.print(
                    f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                    f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                    f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
                    f" total training throughput {tokens / (t1 - train_t0) / 1e3:.2f}K tokens/s per GPU."
                    f" total trained tokens: {total_tokens} B tokens"
                    f" peak memory allocate {torch.cuda.memory_stats(0)['allocated_bytes.all.peak'] / 1e9} GB"
                )           
            
        estimated_flops = 1
        monitor.on_train_batch_end(
            state["iter_num"] * args.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )        

        # Exiting based on duration. 
        # credits: https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L985-L998
        if not is_accumulating and args.actual_train_time:
            train_time = (time.time() - _TRAIN_START_TIME)
            # start monitoring sync
            done_cuda = torch.tensor([train_time > args.actual_train_time], device=fabric.device, dtype=torch.int)
            # force synchronization (single-GPU runs have no process group:
            # calling all_reduce there raises and kills the run WITHOUT the
            # clean save this block exists to perform — observed on segment
            # 4239, which died at 5.5h and resumed 30M tokens back).
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                fabric.print(f"Training time {train_time/60.0} min, Reach time limit. Exiting ...")
                save_checkpoint()
                sys.exit()

        if not is_accumulating and state["step_count"] % args.save_step_interval == 0:
            # Detect 1B-token boundary crossings for milestone ckpt + HF upload.
            # iter_num counts forward passes; each forward sees micro_batch * block_size tokens per GPU.
            total_trained_tokens = state["iter_num"] * args.micro_batch_size * model.config.block_size * fabric.world_size
            milestone_tokens_b = None

            # Custom milestone list (e.g., "100000000,1000000000,5000000000" for 100M/1B/5B).
            # When set, this takes precedence over the regular interval-based milestones.
            custom_milestones = getattr(train, "_custom_milestones", None)
            if custom_milestones is None and getattr(args, "milestone_tokens_csv", ""):
                custom_milestones = sorted(int(x) for x in args.milestone_tokens_csv.split(",") if x.strip())
                train._custom_milestones = custom_milestones
                train._custom_milestone_idx = 0

            if custom_milestones:
                idx = getattr(train, "_custom_milestone_idx", 0)
                while idx < len(custom_milestones) and total_trained_tokens >= custom_milestones[idx]:
                    mt = custom_milestones[idx]
                    milestone_tokens_b = mt // 1_000_000_000 if mt >= 1_000_000_000 else None
                    # Use fractional label for sub-B milestones (e.g., checkpoint-100M).
                    if milestone_tokens_b is None:
                        milestone_label = f"{mt // 1_000_000}M"
                        # Override save_checkpoint's naming with a custom path.
                        save_checkpoint()
                        if fabric.global_rank == 0:
                            src = os.path.join(args.out_dir, "latest-model-ckpt.pth")
                            dst = os.path.join(args.out_dir, f"checkpoint-{milestone_label}-model-ckpt.pth")
                            shutil.copy2(src, dst)
                            fabric.print(f"Sub-B milestone checkpoint saved: {dst}")
                            if args.hf_upload and args.hf_repo_id:
                                if args.hf_upload_blocking:
                                    _hf_upload_checkpoint(dst, args.hf_repo_id, args.hf_token, blocking=True, run_dir=args.out_dir)
                                else:
                                    import threading
                                    threading.Thread(
                                        target=_hf_upload_checkpoint,
                                        args=(dst, args.hf_repo_id, args.hf_token, True, args.out_dir),
                                        daemon=True,
                                    ).start()
                    else:
                        save_checkpoint(milestone_tokens_b=milestone_tokens_b)
                    idx += 1
                train._custom_milestone_idx = idx
                if milestone_tokens_b is None and state["step_count"] % max(args.save_step_interval, 1) == 0:
                    # Still save the latest ckpt for resume, no milestone copy.
                    save_checkpoint()
            elif args.hf_upload and args.hf_repo_id and args.hf_upload_interval_tokens > 0:
                boundary = total_trained_tokens // args.hf_upload_interval_tokens
                last = getattr(train, "_last_hf_boundary", 0)
                if boundary > last and boundary >= 1:
                    # Walk every crossed boundary. Milestone filename uses ACTUAL token billions
                    # (e.g., interval=5B → checkpoint-5B, 10B, ..., 100B), not the boundary index.
                    interval_b = args.hf_upload_interval_tokens // 1_000_000_000
                    for b in range(max(last, 1), int(boundary) + 1):
                        milestone_tokens_b = b * interval_b
                        save_checkpoint(milestone_tokens_b=milestone_tokens_b)
                    train._last_hf_boundary = int(boundary)
                else:
                    save_checkpoint()
            else:
                save_checkpoint()

        # First save ckpt then do eval in case ckpt is not saved in time
        if val_dataloader is not None and not is_accumulating and state["step_count"] % args.eval_step_interval == 0:            
            t0 = time.perf_counter()
            val_loss = validate(args, fabric, model, val_dataloader, args.eval_iters)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            for i in range(args.num_extrapol):
                if fabric.global_rank == 0:
                    fabric.print(f"step {state['iter_num']} {i+1} x: val loss {val_loss[i]:.4f}, val time: {t1 * 1000:.2f}ms")        
                    fabric.log_dict({"metric/val_loss@"+str(i+1)+"x": val_loss[i].item()}, state["step_count"])
                    fabric.log_dict({"metric/val_ppl@"+str(i+1)+"x": math.exp(val_loss[i].item())}, state["step_count"])

            fabric.barrier()
    
    save_checkpoint(final=True)

# each gpu will run validation on the entire val_dataset to avoid headache.
@torch.no_grad()
def validate(args, fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters=100) -> torch.Tensor:
    if fabric.global_rank == 0:
        fabric.print("Validating ...")
    model.eval()
    if eval_iters is None:
        eval_iters = args.eval_iters
    losses = torch.zeros(args.num_extrapol, device=fabric.device, dtype=torch.float64)
    num_sample = 0
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        num_sample += 1
        for i, length in enumerate([4096, 8192, 12288, 16384]):   #[2048, 4096, 8192, 16384]
            if args.use_stream_tok:
                input_ids = val_data['input_ids'][:, 0:length].contiguous().to(fabric.device)
                targets = val_data['labels'][:, 1:length + 1].contiguous().to(fabric.device)
            else:
                input_ids = val_data[:, 0 : length].contiguous()
                targets = val_data[:, 1 : length + 1].contiguous()
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            # running average to avoid overflow
            losses[i] += (loss.item() - losses[i]) / num_sample
    fabric.print(f"Validation loss: {losses}")
    model.train()
    return losses


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        #filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        filenames = sorted(glob.glob(os.path.join(data_dir,f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)
        if split != "train":
            n_chunks = - (8 // -nodes) # ceil division
        else:
            n_chunks = 8
        dataset = PackedDataset(
            filenames,
            n_chunks=n_chunks,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=- (batch_size // -2), # ceil division
            block_size=  16384 + 1, #num_extrapol * block_size + 1, # val 4* extrapolation
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(args, it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    devices = torch.cuda.device_count() or 1
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='LLM Training')
    group = parser.add_argument_group('hyperparameters')
    group.add_argument('--output_root', default='', type=str, help='output root directory')
    group.add_argument('--wandb_dir', default='', type=str, help='wandb directory')
    group.add_argument('--train_data_dir', default='', type=str, help='training data directory')
    group.add_argument('--corpus_name', default='slimpajama', type=str, help='corpus name')
    group.add_argument('--train_data_dir_raw', default='', type=str, help='training data directory (raw file for stream tok)')
    group.add_argument('--val_data_dir', default='', type=str, help='validation data directory')
    group.add_argument('--val_data_dir_raw', default='', type=str, help='validation data directory (raw file for stream tok)')
    group.add_argument('--model_name', default='Samba_421M', type=str, help='model name')
    group.add_argument('--config_overrides', default='', type=str,
                       help="comma-separated key=value applied to the model "
                            "config, e.g. mc_checkpoint_mode=chained. Values "
                            "are cast to int or float when they parse as one, "
                            "so a mode name stays a string.")
    group.add_argument('--exp_name', default='', type=str, help='experiment name')
    group.add_argument('--exp_group', default='', type=str, help='experiment group name')
    group.add_argument('--train_config', default='tsz512x4k_20B', type=str, help='training config')
    group.add_argument('--resume', action='store_true', default=False, help='resume flag')
    group.add_argument('--debug', action='store_true', default=False, help='debug flag')
    group.add_argument('--interactive_job', action='store_true', default=False, help='debug flag')
    group.add_argument('--use_stream_tok', action='store_true', default=False, help='resume flag')
    group.add_argument('--tokenizer_name', type=str, default='TinyLlama/TinyLlama_v1.1')
    group.add_argument('--tokenizer_path', type=str, default='/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision/datasets/llm_next_gen/tokenizers/TinyLlama/TinyLlama_v1.1')
    group.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    group.add_argument('--total_evals', type=int, default=400, help='total number of evals')
    group.add_argument('--eval_iters', type=int, default=15, help='number of evaluation iterations')
    group.add_argument('--log_step_interval', type=int, default=10, help='log_step_interval')
    group.add_argument('--save_step_interval', type=int, default=1000, help='save_step_interval')
    group.add_argument('--eval_step_interval', type=int, default=1000, help='eval_step_interval')
    group.add_argument('--seed', type=int, default=3407, help='seed')
    group.add_argument('--init_from_ckpt', default='', type=str,
                       help='initialize MODEL WEIGHTS from this checkpoint '
                            '(strict; finetuning init — optimizer/step start '
                            'fresh, unlike --resume)')
    group.add_argument('--num_extrapol', type=int, default=4, help='num_extrapol')
    group.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
    group.add_argument('--beta1', type=float, default=0.9, help='beta1')
    group.add_argument('--beta2', type=float, default=0.95, help='beta2')
    group.add_argument('--grad_clip', type=float, default=1.0, help='gradient clip')
    group.add_argument('--val_type', default='val_sampled', type=str, help='choose between val_sampled and val for validation type')
    group.add_argument('--eval_before_training', action='store_true', default=False, help='do validation before the training starts')
    group.add_argument('--nnodes', type=int, default=None, help='number of nodes')
    group.add_argument('--train_num_workers', type=int, default=8)
    group.add_argument('--val_num_workers', type=int, default=1)
    group.add_argument('--actual_train_time', type=int, default=235, help='actual training time in mins')
    group.add_argument('--micro_batch_size', type=int, default=0, help='micro batch size')
    # B-2 fix: token budget was inferred by substring matching on
    # train_config+exp_name, so an exp_name containing e.g. "30B" silently
    # hijacked the budget. --max_tokens sets it directly and wins over matching.
    group.add_argument('--max_tokens', type=int, default=0, help='training token budget; overrides substring inference from train_config (0 = infer)')
    # B-3 fix companion: explicit global batch override (0 = infer from train_config).
    group.add_argument('--global_batch_size', type=int, default=0, help='global batch size in sequences; overrides substring inference from train_config (0 = infer)')

    hf_group = parser.add_argument_group('huggingface_hub upload')
    hf_group.add_argument('--hf_upload', action='store_true', default=False, help='enable HF Hub upload on milestone saves')
    hf_group.add_argument('--no-hf_upload', dest='hf_upload', action='store_false')
    hf_group.add_argument('--hf_repo_id', default='', type=str, help='target HF Hub repo (user/name)')
    hf_group.add_argument('--hf_token', default=os.getenv('HF_TOKEN', ''), type=str, help='HF token (defaults to env HF_TOKEN)')
    hf_group.add_argument('--hf_upload_interval_tokens', default=1_000_000_000, type=int, help='push to HF every N trained tokens (default 1B)')
    hf_group.add_argument('--milestone_tokens_csv', default='', type=str, help='Comma-separated custom milestone token counts (e.g., "100000000,1000000000,5000000000" for 100M/1B/5B). When set, overrides hf_upload_interval_tokens.')
    hf_group.add_argument('--hf_upload_blocking', action='store_true', default=False, help='block training until upload finishes')

    args = parser.parse_args()
    name = args.train_config +"_" + args.exp_name
    args.out_dir = args.output_root + '/outputs/' + name
    args.wandb_dir = args.output_root + '/wandb/' + name

    train_data_config = [("train_slim", 1.0)]
    val_data_config = [("validation", 1.0)]

    nodes = int(os.getenv("SLURM_NNODES", "1"))
    args.nodes = nodes

    micro_batch_size = 8

    # B-2 fix: infer the token budget from train_config ONLY (never from
    # exp_name — "30B" inside an experiment name used to silently hijack the
    # budget), and let an explicit --max_tokens win outright.
    budget_key = args.train_config
    if args.max_tokens > 0:
        max_tokens = args.max_tokens
    elif "20B" in budget_key:
        max_tokens = int(1e11) // 5 # 20 billion
    elif "100B" in budget_key:
        max_tokens = int(1e11) # 100 billion
    elif "50B" in budget_key:
        max_tokens = int(1e11) // 2 # 50 billion
    elif "30B" in budget_key:
        max_tokens = int(3e10) # 30 b
    elif "15B" in budget_key:
        max_tokens = int(3e10) // 2 # 15 b
    elif "10B" in budget_key:
        max_tokens = int(1e10) # 10 billion
    elif "5B" in budget_key:
        max_tokens = int(5e9) # 5 billion
    elif "1B" in budget_key:
        max_tokens = int(1e9) # 1 billion
    elif "50M" in budget_key:
        max_tokens = int(5e7) # 50 million (sanity check)
    else:
        raise ValueError("Unknown training token config")

    if "512x4k" in budget_key:
        micro_batch_size = 8
        global_batch_size = 512 // nodes
    elif "1024x4k" in budget_key:
        micro_batch_size = 8
        global_batch_size = 1024 // nodes

    elif "128x4k" in budget_key:
        # Paper-matched: global batch 0.5M tokens = 128 seqs @ 4K
        micro_batch_size = 8
        global_batch_size = 128 // nodes

    elif "256x8k" in budget_key:
        #8k
        global_batch_size = 256 // nodes
        micro_batch_size = 8

    elif "128x16k" in budget_key:
        #16k
        global_batch_size = 128 // nodes
        micro_batch_size = 4

    elif "64x32k" in budget_key:
        #32k
        global_batch_size = 64 // nodes
        micro_batch_size = 2

    elif "1024x2k" in budget_key:
        #2k
        global_batch_size = 1024 // nodes
        micro_batch_size = 32
    elif args.global_batch_size <= 0:
        # B-3 fix: this chain used to have no else, so an unmatched
        # train_config crashed later with an opaque NameError (or worse,
        # could silently reuse a stale value). Fail loudly instead.
        raise ValueError(
            f"Unknown batch shape in train_config={args.train_config!r}; "
            "use one of 512x4k/1024x4k/128x4k/256x8k/128x16k/64x32k/1024x2k "
            "or pass --global_batch_size explicitly"
        )

    if args.global_batch_size > 0:
        global_batch_size = args.global_batch_size // nodes

    if "1.3B" in name and "370M" not in name:
        micro_batch_size = micro_batch_size // 2
    
    micro_batch_size = max(1, micro_batch_size)
    args.min_lr = args.learning_rate / 10
    args.batch_size = global_batch_size // devices

    # Resolve the CLI --micro_batch_size override BEFORE computing gradient
    # accumulation. Grad accum used to be derived from the inferred local
    # micro_batch_size above, so reducing --micro_batch_size (e.g. 8 -> 4 on
    # A100 80GB) kept accum fixed and silently halved the effective global
    # batch (0.5M -> 0.25M tokens) while the token budget still added up.
    if args.micro_batch_size == 0:
        args.micro_batch_size = micro_batch_size
    micro_batch_size = args.micro_batch_size

    assert args.batch_size % args.micro_batch_size == 0, (
        f"per-device batch_size {args.batch_size} is not divisible by "
        f"micro_batch_size {args.micro_batch_size}; the effective global "
        f"batch would silently shrink"
    )
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    assert gradient_accumulation_steps > 0
    log_iter_interval = args.log_step_interval * gradient_accumulation_steps
    args.gradient_accumulation_steps = gradient_accumulation_steps

    args.actual_train_time = args.actual_train_time * 60 # convert to seconds
    args.warmup_tokens = int(max_tokens * 0.01)
    args.max_tokens = max_tokens
    args.log_iter_interval= log_iter_interval
    hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    args.hparams = hparams
    main(args)
