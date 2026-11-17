"""
train_sft.py — Supervised Fine-Tuning (Full-Weight) for Tiny_Llama

=============================================================================
FULL-WEIGHT SFT WORKFLOW:
=============================================================================
1. Restore Pretrained Weights:
   - Loads base checkpoint `checkpoints/base_model.pt` (Step 4,500 base model).
   - All 50.7M parameters are trainable (requires_grad=True).

2. Optimizer & Learning Rate:
   - Standard AdamW optimizer with a gentle learning rate (5e-5).
   - Linear warmup (20% of first epoch) + Cosine Decay down to 1e-6.
   - Low learning rate prevents "catastrophic forgetting" of base pretraining.

3. Target-Masked Cross-Entropy Loss:
   - Targets at prompt positions are set to -1 (`ignore_index = -1`).
   - PyTorch's F.cross_entropy skips tokens where target == -1,
     ensuring gradients ONLY optimize the model's ability to answer instructions!

4. Validation & Early Checkpointing:
   - Calculates validation loss at the end of each epoch.
   - Automatically saves the best model to `checkpoints/sft_best.pt`.
=============================================================================
"""

import argparse
import math
import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_sft import create_instruction_dataloaders
from model import LlamaConfig, TinyLlama


def parse_args():
    parser = argparse.ArgumentParser(description="Full-Weight Supervised Fine-Tuning for Tiny_Llama.")
    
    # Paths
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/base_model.pt",
                        help="Path to pretrained base model weights.")
    parser.add_argument("--data_path", type=str, default="data/instruction-data.json",
                        help="Path to instruction dataset JSON.")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save fine-tuned checkpoints.")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per forward step.")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps (effective BS = 16).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate for AdamW.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Floor learning rate after cosine decay.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 regularization factor.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping L2 norm threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()


def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, val_loader: DataLoader, device: str, amp_dtype: torch.dtype) -> float:
    """Evaluates cross-entropy loss over validation instruction dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits, loss = model(x, targets=y)
            # Compute token-weighted sum for exact epoch averaging
            loss_sum = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction="sum")
        
        valid_tokens = (y != -1).sum().item()
        total_loss += loss_sum.item()
        total_tokens += valid_tokens

    model.train()
    return total_loss / max(1, total_tokens)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print("=" * 75)
    print("  Tiny_Llama Full-Weight Supervised Fine-Tuning (SFT)")
    print("=" * 75)
    print(f"Device:             {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'} ({device})")
    print(f"Base Checkpoint:    {args.base_checkpoint}")
    print(f"Dataset:            {args.data_path}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch Size:         {args.batch_size} (effective: {args.batch_size * args.grad_accum_steps})")
    print(f"Peak Learning Rate: {args.lr}")
    print("=" * 75)

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Pretrained Model
    if not os.path.exists(args.base_checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.base_checkpoint}")

    print(f"Loading weights from {args.base_checkpoint}...")
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    
    # Retrieve config from checkpoint or instantiate default
    config = checkpoint.get("config", LlamaConfig())
    model = TinyLlama(config).to(device)
    
    # Load state dict
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    print(f"Restored base model ({model.get_num_params():,} total parameters).")

    # Ensure all parameters have gradients enabled for Full Fine-Tuning
    for p in model.parameters():
        p.requires_grad = True

    # 3. Create Instruction DataLoaders
    train_loader, val_loader = create_instruction_dataloaders(
        json_path=args.data_path,
        batch_size=args.batch_size,
        train_split_ratio=0.85,
        max_length=config.block_size,
        seed=args.seed
    )
    print(f"Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Val samples:   {len(val_loader.dataset)} ({len(val_loader)} batches)")

    # 4. Optimizer & Schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scaler = torch.amp.GradScaler(device, enabled=(amp_dtype == torch.float16))

    # Evaluate initial baseline validation loss before any fine-tuning
    print("\nEvaluating initial base model loss on instruction validation set...")
    init_val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
    print(f"Base Model Initial Val Loss: {init_val_loss:.4f} (Perplexity: {math.exp(min(init_val_loss, 20.0)):.2f})\n")

    best_val_loss = init_val_loss
    best_ckpt_path = os.path.join(args.output_dir, "sft_best.pt")

    global_step = 0
    start_time = time.time()

    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Mixed precision forward pass
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits, loss = model(x, targets=y)
                loss = loss / args.grad_accum_steps

            # Backward pass
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum_steps

            # Gradient accumulation step
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                # Update learning rate via Cosine Schedule
                current_lr = get_cosine_lr(global_step, total_steps, warmup_steps, args.lr, args.min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

                # Unscale & gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 20 == 0:
                    print(f"Epoch {epoch}/{args.epochs} | Step {global_step}/{total_steps} | "
                          f"Loss: {loss.item() * args.grad_accum_steps:.4f} | LR: {current_lr:.2e}")

        # End of Epoch Evaluation
        val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
        val_ppl = math.exp(min(val_loss, 20.0))
        avg_train_loss = epoch_loss / len(train_loader)

        print("-" * 75)
        print(f"[*] Epoch {epoch}/{args.epochs} Summary | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Perplexity: {val_ppl:.2f}")

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "train_loss": avg_train_loss,
                "optimizer_state_dict": optimizer.state_dict(),
            }, best_ckpt_path)
            print(f"🌟 Saved new best SFT model to: {best_ckpt_path}")
        print("-" * 75)

    total_time = time.time() - start_time
    print(f"\n[DONE] Fine-tuning finished in {total_time / 60:.1f} minutes!")
    print(f"Best Val Loss: {best_val_loss:.4f} (Saved at {best_ckpt_path})")


if __name__ == "__main__":
    main()
