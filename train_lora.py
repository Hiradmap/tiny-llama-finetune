"""
train_lora.py — Parameter-Efficient Fine-Tuning (LoRA) for Tiny_Llama

=============================================================================
LoRA FINE-TUNING WORKFLOW (v3):
=============================================================================
1. Load & Freeze Pretrained Base Model:
   - Restores `checkpoints/base_model.pt` (~50.7M parameters).
   - Injects trainable low-rank adapters (A, B) into Attention & SwiGLU layers.
   - Freezes 100% of original pretrained weights (no catastrophic forgetting!).

2. Optimizer:
   - AdamW optimizes ONLY the ~1.45M trainable LoRA parameters (~2.8% of model).
   - Higher learning rate (2e-4) is effective because only adapter matrices update.

3. Lightweight Checkpointing:
   - Saves ONLY the adapter weights to `checkpoints/lora_v3_best.pt` (~5-6 MB vs 580 MB!).
=============================================================================
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from dataset_sft import create_instruction_dataloaders
from lora import DEFAULT_TARGET_MODULES, inject_lora, save_lora_checkpoint
from model import LlamaConfig, TinyLlama
from train_sft import evaluate_loss, get_cosine_lr


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Supervised Fine-Tuning for Tiny_Llama.")

    # Paths
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/base_model.pt",
                        help="Path to pretrained base model weights.")
    parser.add_argument("--data_path", type=str, default="data/instruction-data-v2.json",
                        help="Path to instruction dataset JSON.")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save LoRA checkpoints.")
    parser.add_argument("--save_name", type=str, default="lora_v3_best.pt",
                        help="Filename for saved LoRA adapter checkpoint.")
    parser.add_argument("--results_path", type=str, default="results/lora_v3_metrics.json",
                        help="JSON path to export training metrics.")

    # LoRA Hyperparameters
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA scaling factor alpha.")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout probability.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per forward step.")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate for LoRA adapters.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate after decay.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 weight decay.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint}")

    # 1. Load pretrained base model
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", LlamaConfig())
    model = TinyLlama(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Inject LoRA adapters and freeze base weights
    lora_config = {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": list(DEFAULT_TARGET_MODULES),
    }
    trainable_params, total_params = inject_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=DEFAULT_TARGET_MODULES,
    )
    trainable_pct = 100.0 * trainable_params / total_params

    print("=" * 75)
    print("  Tiny_Llama LoRA Supervised Fine-Tuning (v3)")
    print("=" * 75)
    print(f"Device:             {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'} ({device})")
    print(f"Base Checkpoint:    {args.base_checkpoint}")
    print(f"Dataset:            {args.data_path}")
    print(f"LoRA Config:        rank={args.rank}, alpha={args.alpha}, dropout={args.dropout}")
    print(f"Total Parameters:   {total_params:,}")
    print(f"Trainable Params:   {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"Epochs / LR:        {args.epochs} epochs | Peak LR={args.lr}")
    print("=" * 75)

    # 3. DataLoaders (identical split seed for fair comparison with v2 Full SFT)
    train_loader, val_loader = create_instruction_dataloaders(
        json_path=args.data_path,
        batch_size=args.batch_size,
        train_split_ratio=0.85,
        max_length=config.block_size,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Val samples:   {len(val_loader.dataset)} ({len(val_loader)} batches)")

    # 4. Optimize ONLY trainable LoRA parameters
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_param_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_steps)
    log_interval = 100 if total_steps > 500 else 20

    scaler = torch.amp.GradScaler(device, enabled=(amp_dtype == torch.float16))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\nEvaluating initial base+LoRA model loss on instruction validation set...")
    init_val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
    print(f"Initial Val Loss: {init_val_loss:.4f} (Perplexity: {math.exp(min(init_val_loss, 20.0)):.2f})\n")

    best_val_loss = init_val_loss
    best_ckpt_path = os.path.join(args.output_dir, args.save_name)

    global_step = 0
    start_time = time.time()
    epoch_history = []

    # 5. LoRA Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits, loss = model(x, targets=y)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                current_lr = get_cosine_lr(global_step, total_steps, warmup_steps, args.lr, args.min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_param_list, args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_interval == 0:
                    print(f"Epoch {epoch}/{args.epochs} | Step {global_step}/{total_steps} | "
                          f"Loss: {loss.item() * args.grad_accum_steps:.4f} | LR: {current_lr:.2e}")

        val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
        val_ppl = math.exp(min(val_loss, 20.0))
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_history.append({
            "epoch": epoch,
            "step": global_step,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
        })

        print("-" * 75)
        print(f"[*] Epoch {epoch}/{args.epochs} Summary | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Perplexity: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_lora_checkpoint(
                path=best_ckpt_path,
                model=model,
                model_config=config,
                lora_config=lora_config,
                base_checkpoint=args.base_checkpoint,
                epoch=epoch,
                global_step=global_step,
                val_loss=val_loss,
                train_loss=avg_train_loss,
            )
            ckpt_size_mb = os.path.getsize(best_ckpt_path) / (1024 * 1024)
            print(f"🌟 Saved new best LoRA adapter ({ckpt_size_mb:.2f} MB) to: {best_ckpt_path}")
        print("-" * 75)

    total_time = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    print(f"\n[DONE] LoRA Fine-tuning finished in {total_time / 60:.1f} minutes! Peak VRAM: {peak_vram_mb:.1f} MB")
    print(f"Best Val Loss: {best_val_loss:.4f} (Saved at {best_ckpt_path})")

    if args.results_path:
        os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
        ckpt_size_mb = os.path.getsize(best_ckpt_path) / (1024 * 1024) if os.path.exists(best_ckpt_path) else 0.0
        metrics = {
            "method": f"LoRA (r={args.rank}, alpha={args.alpha})",
            "data_path": args.data_path,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_pct": round(trainable_pct, 2),
            "epochs": args.epochs,
            "lr": args.lr,
            "init_val_loss": round(init_val_loss, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_perplexity": round(math.exp(min(best_val_loss, 20.0)), 2),
            "training_time_sec": round(total_time, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "checkpoint_path": best_ckpt_path,
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "history": epoch_history,
        }
        with open(args.results_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Exported metrics to: {args.results_path}")


if __name__ == "__main__":
    main()
