"""
compare_versions.py — Side-by-Side Progress Evaluator Across All Versions

Evaluates every saved version without modifying or deleting any checkpoints:
  - v0: Base Pretrained Model           (checkpoints/base_model.pt)
  - v1: Full-Weight SFT (1.1k samples)  (checkpoints/sft_best.pt)
  - v2: Full-Weight SFT (15k samples)   (checkpoints/sft_v2_best.pt)
  - v3: LoRA r=16 SFT (15k samples)     (checkpoints/lora_v3_best.pt)

Measures held-out v2 validation loss/perplexity, file size, and qualitative
responses across multiple prompts, exporting results to results/version_comparison.json.
"""

import json
import math
import os

import tiktoken
import torch

from chat_sft import generate_response
from dataset_sft import create_instruction_dataloaders
from lora import load_model_from_checkpoint
from train_sft import evaluate_loss


VERSIONS = [
    ("v0_base", "v0: Base Pretrained Model", "checkpoints/base_model.pt"),
    ("v1_full_sft_1k", "v1: Full SFT (1.1k samples)", "checkpoints/sft_best.pt"),
    ("v2_full_sft_15k", "v2: Full SFT (15k samples)", "checkpoints/sft_v2_best.pt"),
    ("v3_lora_15k", "v3: LoRA r=16 (15k samples)", "checkpoints/lora_v3_best.pt"),
]

TEST_PROMPTS = [
    "What is machine learning?",
    "Give three tips for staying healthy.",
    "Convert 45 kilometers to meters.",
    "Explain why the sky is blue in simple terms.",
    "Write a short two-sentence story about a robot.",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = tiktoken.get_encoding("gpt2")

    _, val_loader = create_instruction_dataloaders(
        json_path="data/instruction-data-v2.json",
        batch_size=8,
        train_split_ratio=0.85,
        max_length=256,
        seed=42,
    )

    report = {"versions": [], "prompts": TEST_PROMPTS}

    print("=" * 85)
    print("  Tiny_Llama Fine-Tuning Version Comparison (v0 -> v1 -> v2 -> v3)")
    print("=" * 85)

    for key, label, ckpt_path in VERSIONS:
        if not os.path.exists(ckpt_path):
            print(f"[Skip] {label}: missing {ckpt_path}")
            continue

        torch.manual_seed(42)
        model, ckpt = load_model_from_checkpoint(ckpt_path, device=device, merge_lora=True)
        model.eval()

        val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
        val_ppl = math.exp(min(val_loss, 20.0))
        ckpt_size_mb = os.path.getsize(os.path.realpath(ckpt_path)) / (1024 * 1024)

        print(f"\n--- {label} ({ckpt_path}) ---")
        print(f"Checkpoint Size: {ckpt_size_mb:.2f} MB | v2 Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

        samples = {}
        for prompt in TEST_PROMPTS:
            torch.manual_seed(42)
            ans = generate_response(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=100,
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.25,
            )
            samples[prompt] = ans
            print(f"  Q: {prompt}")
            print(f"  A: {ans}\n")

        # Also compare v1 with old sampling (temp=0.7, rep=1.15) vs tuned (temp=0.3, rep=1.25)
        extra_note = None
        if key == "v1_full_sft_1k":
            torch.manual_seed(42)
            old_ans = generate_response(
                model,
                tokenizer,
                "What is machine learning?",
                device=device,
                max_new_tokens=100,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15,
            )
            extra_note = {"old_sampling_temp0.7_rep1.15": old_ans}

        report["versions"].append({
            "key": key,
            "label": label,
            "checkpoint": ckpt_path,
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "v2_val_loss": round(val_loss, 4),
            "v2_val_perplexity": round(val_ppl, 2),
            "samples_tuned_sampling": samples,
            **({"sampling_comparison": extra_note} if extra_note else {}),
        })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    out_path = "results/version_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Saved full comparison report to {out_path}")


if __name__ == "__main__":
    main()
