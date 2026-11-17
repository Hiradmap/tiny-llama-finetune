"""
train.py — Production Training Pipeline for Tiny_Llama

=============================================================================
PRODUCTION TRAINING ARCHITECTURE & RECIPE
=============================================================================
This script brings together all the validated training dynamics from Tiny_GPT2
applied directly to the modern Tiny_Llama architecture:

  1. "Dual-Optimizer Coordination":
     - Muon (lr=0.02) updates all internal 2D weight matrices via orthogonalization.
     - AdamW (lr=6e-4) updates the embedding table and 1D RMSNorm parameters.

  2. "Warmup-Stable-Decay (WSD) Schedule":
     - Warmup (0 -> 200 steps): Linear ramp preventing early gradient shocks.
     - Stable (200 -> 6,000 steps): Constant peak learning rate for sustained exploration.
     - Decay (6,000 -> 10,000 steps): Smooth cosine cooldown annealing learning rates
       down to 1e-5 to settle into the sharpest loss basin.

  3. "Automatic Mixed Precision (AMP)":
     - Utilizes bfloat16 natively on supported GPUs (or float16 with GradScaler).
     - Halves VRAM memory bandwidth while maintaining full float32 dynamic range.

  4. "Gradient Accumulation (Effective Batch Size = 64)":
     - Processes 4 micro-batches of size 16 before executing a weight update step.
     - Provides smooth, low-variance gradients that allow Muon to utilize aggressive
       learning rates (0.02) without numerical divergence.

  5. "Model Checkpointing & Evaluation":
     - Evaluates validation loss across 20 sample batches every 500 steps.
     - Automatically saves 'checkpoints/best_model.pt' whenever validation loss improves.
=============================================================================
"""

import argparse
import math
import os
import time
from typing import Tuple

import tiktoken
import torch

from benchmark import BenchmarkTracker, evaluate_generation_quality, save_results
from dataset import MemmapDataLoader
from model import LlamaConfig, TinyLlama
from muon import setup_optimizers

# ===========================================================================
# 1. COMMAND LINE HYPERPARAMETERS
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tiny_Llama with Muon + AdamW and WSD schedule.")
    
    # Experiment metadata
    parser.add_argument("--version", type=str, default="v01_llama", help="Experiment identifier for tracking.")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing train.bin and val.bin.")
    
    # Batch sizing & Context
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size per GPU step.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Number of micro-batches to accumulate.")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence context length in tokens.")
    
    # Training schedule
    parser.add_argument("--max_steps", type=int, default=10000, help="Total optimizer update steps.")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Linear warmup steps from 0 to peak LR.")
    parser.add_argument("--stable_steps", type=int, default=6000, help="Steps to hold learning rate constant.")
    
    # Optimizer learning rates
    parser.add_argument("--max_muon_lr", type=float, default=0.02, help="Peak learning rate for Muon optimizer.")
    parser.add_argument("--min_muon_lr", type=float, default=1e-5, help="Minimum learning rate for Muon after decay.")
    parser.add_argument("--max_adamw_lr", type=float, default=6e-4, help="Peak learning rate for AdamW optimizer.")
    parser.add_argument("--min_adamw_lr", type=float, default=1e-5, help="Minimum learning rate for AdamW after decay.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="L2 weight decay for AdamW embeddings.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient L2 norm for clipping.")
    
    # Checkpointing & Evaluation
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation and checkpoint check frequency.")
    parser.add_argument("--eval_steps", type=int, default=20, help="Number of batches to evaluate per checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to export benchmark metrics.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from (e.g. checkpoints/checkpoint_last.pt).")
    parser.add_argument("--seed", type=int, default=1337, help="Master random seed for reproducibility.")
    
    return parser.parse_args()


# ===========================================================================
# 2. WSD (WARMUP-STABLE-DECAY) LEARNING RATE SCHEDULE
# ===========================================================================

def get_wsd_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    stable_steps: int,
    max_steps: int
) -> float:
    """
    Computes the learning rate multiplier for step `step` using Warmup-Stable-Decay.
    
    Phases:
      1. Warmup:  lr = max_lr * (step + 1) / warmup_steps
      2. Stable:  lr = max_lr
      3. Decay:   lr = min_lr + 0.5 * (1 + cos(pi * ratio)) * (max_lr - min_lr)
    """
    if step < warmup_steps:
        # Linear warmup phase
        return max_lr * ((step + 1) / max(1, warmup_steps))
    elif step < stable_steps:
        # Stable phase (flat learning rate)
        return max_lr
    elif step >= max_steps:
        # End of training floor
        return min_lr
    else:
        # Cosine decay cooldown phase
        decay_ratio = (step - stable_steps) / max(1, (max_steps - stable_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


# ===========================================================================
# 3. LOSS EVALUATION FUNCTION
# ===========================================================================

@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_loader: MemmapDataLoader,
    val_loader: MemmapDataLoader,
    eval_steps: int,
    device: str,
    amp_dtype: torch.dtype,
    use_amp: bool
) -> Tuple[float, float]:
    """
    Averaging loss over `eval_steps` micro-batches on both train and validation splits.
    Disables autograd tracking and puts the model into eval() mode.
    """
    model.eval()
    losses = {}
    loaders = {"train": train_loader, "val": val_loader}

    for split, loader in loaders.items():
        loss_sum = 0.0
        for _ in range(eval_steps):
            x, y = loader.get_batch()
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            loss_sum += loss.item()
        losses[split] = loss_sum / eval_steps

    model.train()
    return losses["train"], losses["val"]


# ===========================================================================
# 4. MAIN TRAINING PIPELINE
# ===========================================================================

def main():
    args = parse_args()
    
    # Step 1: Set seeds for deterministic training
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    # Step 2: Configure Mixed Precision
    # bfloat16 is preferred if supported by GPU hardware (Ampere / Ada / Hopper / Blackwell)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = (device == "cuda")
    # GradScaler is required for float16 (to prevent underflow), but unnecessary for bfloat16
    scaler = torch.amp.GradScaler(device, enabled=(use_amp and not use_bf16))

    # Ensure output directories exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 75)
    print("  Tiny_Llama Production Training Engine")
    print("=" * 75)
    print(f"Device:                   {device_name} ({device})")
    print(f"Mixed Precision:          {'bfloat16 (Native)' if use_bf16 else 'float16 + GradScaler' if use_amp else 'float32 (CPU)'}")
    print(f"Micro-Batch Size:         {args.batch_size}")
    print(f"Gradient Accumulation:    {args.grad_accum_steps} steps")
    print(f"Effective Batch Size:     {args.batch_size * args.grad_accum_steps} sequences ({args.batch_size * args.grad_accum_steps * args.block_size:,} tokens/step)")
    print(f"Total Steps:              {args.max_steps:,} optimizer steps")
    print(f"Schedule:                 Warmup: {args.warmup_steps} | Stable: {args.stable_steps} | Decay: {args.max_steps - args.stable_steps}")
    print("=" * 75)

    # Step 3: Initialize Memory-Mapped Data Loaders
    train_loader = MemmapDataLoader(
        data_dir=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device
    )
    val_loader = MemmapDataLoader(
        data_dir=args.data_dir,
        split="val",
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device
    )

    # Step 4: Instantiate Modern LLaMA Model Architecture
    config = LlamaConfig(block_size=args.block_size)
    model = TinyLlama(config).to(device)
    total_params = model.get_num_params()
    print(f"Model Architecture:       TinyLlama ({total_params:,} parameters ~{total_params / 1e6:.2f}M)")

    # Step 5: Setup Dual Optimizers (Muon for 2D weights, AdamW for embeddings & norms)
    optimizer_muon, optimizer_adamw = setup_optimizers(
        model=model,
        lr_muon=args.max_muon_lr,
        lr_adamw=args.max_adamw_lr,
        weight_decay=args.weight_decay
    )

    # Step 6: Initialize Benchmark Observer Tracker
    tracker = BenchmarkTracker(version=args.version, model=model, config=config)
    tracker.on_train_start()

    best_val_loss = float("inf")
    tokenizer = tiktoken.get_encoding("gpt2")
    start_step = 0

    # Step 7: Handle Resumption or Step 0 baseline evaluation
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint '{args.resume}' not found!")
        print(f"\nResuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_muon_state_dict" in checkpoint:
            optimizer_muon.load_state_dict(checkpoint["optimizer_muon_state_dict"])
        if "optimizer_adamw_state_dict" in checkpoint:
            optimizer_adamw.load_state_dict(checkpoint["optimizer_adamw_state_dict"])
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        train_loss_resumed = checkpoint.get("train_loss", 0.0)
        tracker.on_eval(start_step, train_loss_resumed, best_val_loss)
        print(f"-> Successfully restored Step {start_step} | Best Val Loss: {best_val_loss:.4f} | Resumed Train Loss: {train_loss_resumed:.4f}")
    else:
        print("\nEvaluating initial baseline before training (Step 0)...")
        train_loss_0, val_loss_0 = estimate_loss(
            model, train_loader, val_loader, args.eval_steps, device, amp_dtype, use_amp
        )
        tracker.on_eval(0, train_loss_0, val_loss_0)
        print(f"Step 0 | Train Loss: {train_loss_0:.4f} | Val Loss: {val_loss_0:.4f} | Val PPL: {math.exp(min(val_loss_0, 20.0)):.2f}")

    start_time = time.time()
    step_start_time = time.time()

    # =======================================================================
    # MAIN OPTIMIZATION LOOP
    # =======================================================================
    for step in range(start_step, args.max_steps):
        # A) Calculate dynamic learning rates from WSD schedule
        lr_muon = get_wsd_lr(
            step, args.max_muon_lr, args.min_muon_lr, args.warmup_steps, args.stable_steps, args.max_steps
        )
        lr_adamw = get_wsd_lr(
            step, args.max_adamw_lr, args.min_adamw_lr, args.warmup_steps, args.stable_steps, args.max_steps
        )

        for param_group in optimizer_muon.param_groups:
            param_group["lr"] = lr_muon
        for param_group in optimizer_adamw.param_groups:
            param_group["lr"] = lr_adamw

        # B) Clear accumulated gradients
        optimizer_muon.zero_grad(set_to_none=True)
        optimizer_adamw.zero_grad(set_to_none=True)
        accum_loss = 0.0

        # C) Gradient Accumulation Inner Loop
        for micro_step in range(args.grad_accum_steps):
            x, y = train_loader.get_batch()
            
            # Forward pass under mixed precision
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                # Scale loss by accumulation steps so gradients average correctly
                loss = loss / args.grad_accum_steps
                
            accum_loss += loss.detach().item()
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

        # D) Unscale gradients & Clip gradient norm because of accuracy needed in the optimizer step
        scaler.unscale_(optimizer_muon)
        scaler.unscale_(optimizer_adamw)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()

        # E) Optimizer Step: Muon (orthogonalized) + AdamW
        scaler.step(optimizer_muon)
        scaler.step(optimizer_adamw)
        scaler.update()

        # F) Record Step Metrics
        tracker.on_step(step, accum_loss, grad_norm, lr_muon, lr_adamw)

        # G) Terminal Progress Logging (Every 10 steps)
        if (step + 1) % 10 == 0 or step == start_step:
            step_time = (time.time() - step_start_time) / (10 if step > start_step else 1)
            tokens_per_sec = (args.batch_size * args.grad_accum_steps * args.block_size) / max(step_time, 1e-6)
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            print(
                f"Step {step+1:5d}/{args.max_steps} | "
                f"Loss: {accum_loss:.4f} | "
                f"Norm: {grad_norm:.2f} | "
                f"Muon LR: {lr_muon:.4f} | "
                f"AdamW LR: {lr_adamw:.2e} | "
                f"Speed: {tokens_per_sec:,.0f} tok/s | "
                f"VRAM: {vram_mb:.0f}MB"
            )
            step_start_time = time.time()

        # H) Periodic Evaluation & Checkpoint Saving
        if (step + 1) % args.eval_interval == 0 or (step + 1) == args.max_steps:
            print("-" * 75)
            eval_start = time.time()
            train_loss, val_loss = estimate_loss(
                model, train_loader, val_loader, args.eval_steps, device, amp_dtype, use_amp
            )
            eval_time = time.time() - eval_start
            tracker.on_eval(step + 1, train_loss, val_loss)
            
            val_ppl = math.exp(min(val_loss, 20.0))
            print(
                f"[EVAL] Step {step+1:5d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Perplexity: {val_ppl:.2f} | "
                f"Eval Time: {eval_time:.2f}s"
            )

            # Save 'best_model.pt' if validation loss achieved a new all-time low
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save({
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "optimizer_muon_state_dict": optimizer_muon.state_dict(),
                    "optimizer_adamw_state_dict": optimizer_adamw.state_dict(),
                }, best_ckpt_path)
                print(f"🌟 New best validation loss! Saved checkpoint to: {best_ckpt_path}")

            # Save 'checkpoint_last.pt' for crash recovery and training resumption
            last_ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint_last.pt")
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "optimizer_muon_state_dict": optimizer_muon.state_dict(),
                "optimizer_adamw_state_dict": optimizer_adamw.state_dict(),
            }, last_ckpt_path)
            print("-" * 75)
            step_start_time = time.time()

    # Step 8: Finalize Training & Report
    tracker.on_train_end()
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"Training Complete in {total_elapsed / 60:.2f} minutes.")
    print(f"Best Validation Loss Achieved: {best_val_loss:.4f}")
    print("=" * 75)

    # Step 9: Sample Qualitative Generation
    print("\nSampling qualitative text generation from trained model...")
    gen_metrics = evaluate_generation_quality(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt="To be, or not to be, that is the question:"
    )
    print(f"Prompt:     '{gen_metrics['prompt']}'")
    print(f"Generated:  '{gen_metrics['sample']}'")
    print(f"Speed:      {gen_metrics['tokens_per_sec']} tokens/sec")

    # Step 10: Export JSON Benchmark File
    save_results(tracker, gen_metrics, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
