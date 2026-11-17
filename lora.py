"""
lora.py — Low-Rank Adaptation (LoRA) Implemented from Scratch in Pure PyTorch

=============================================================================
HOW LoRA WORKS MATHEMATICALLY:
=============================================================================
In Full-Weight Fine-Tuning, we update every weight matrix W_0 (d_out x d_in):
    W = W_0 + Delta_W

Hu et al. (2021) observed that weight updates during fine-tuning have a low
"intrinsic rank". Instead of updating all d_out * d_in parameters in W_0:
  1. FREEZE the pretrained weight matrix W_0 (requires_grad = False).
  2. Inject two tiny trainable low-rank matrices in parallel:
       A of shape (r, d_in)   -- initialized with Kaiming uniform
       B of shape (d_out, r)  -- initialized to ALL ZEROS
     where rank r << min(d_in, d_out) (e.g. r = 16 vs d = 512).
  3. Forward pass computes:
       y = W_0(x) + (alpha / r) * B(A(dropout(x)))

Why B = 0 at initialization?
  Because Delta_W = B @ A = 0 at step 0, the model starts with the EXACT same
  outputs as the pretrained base model, preventing early random noise shocks.

Benefits demonstrated in this project:
  - Trainable parameters drop from 50.7M (100%) to ~1.5M (~2.9%).
  - Saved checkpoint drops from ~500 MB (full weights + optimizer) to ~6 MB!
  - Zero inference overhead if merged back into W_0 after training.
=============================================================================
"""

import math
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model import LlamaConfig, TinyLlama


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that freezes the original weights
    and adds a trainable low-rank bypass (B @ A).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Freeze the original pretrained linear weights and bias
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Low-rank adapter matrices: A projects down (d_in -> r), B projects up (r -> d_out)
        self.lora_A = nn.Parameter(torch.empty((rank, in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank), device=device, dtype=dtype))

        # Standard LoRA initialization: Kaiming uniform for A, zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # F.linear automatically handles autocast mixed-precision casting
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B) * self.scaling
        return base_out + lora_out

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Merges delta_W = scaling * (B @ A) permanently into base_layer for zero-overhead inference."""
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.add_(delta_w.to(self.base_layer.weight.dtype))
        return self.base_layer


def inject_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
) -> Tuple[int, int]:
    """
    Freezes all base parameters in `model` and wraps matching Linear layers
    with LoRALinear adapters.

    Returns:
        (trainable_params, total_params)
    """
    target_set = set(target_modules)

    # 1. Freeze entire base model first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Recursively replace matching nn.Linear modules
    def _replace_modules(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if name in target_set and isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            else:
                _replace_modules(child)

    _replace_modules(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params


def merge_lora_weights(model: nn.Module) -> None:
    """Merges all LoRALinear layers back into standard nn.Linear layers in-place."""
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            setattr(model, name, child.merge())
        else:
            merge_lora_weights(child)


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extracts ONLY trainable LoRA adapter weights (lora_A, lora_B) from the model."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and ("lora_A" in name or "lora_B" in name)
    }


def save_lora_checkpoint(
    path: str,
    model: TinyLlama,
    model_config: LlamaConfig,
    lora_config: dict,
    base_checkpoint: str,
    epoch: int,
    global_step: int,
    val_loss: float,
    train_loss: float,
) -> None:
    """Saves a lightweight LoRA adapter checkpoint (only a few MB)."""
    torch.save(
        {
            "checkpoint_type": "lora",
            "lora_state_dict": get_lora_state_dict(model),
            "lora_config": lora_config,
            "config": model_config,
            "base_checkpoint": base_checkpoint,
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "train_loss": train_loss,
        },
        path,
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    base_checkpoint_override: Optional[str] = None,
    merge_lora: bool = True,
) -> Tuple[TinyLlama, dict]:
    """
    Unified loader that seamlessly restores either:
      - A Full-Weight checkpoint (base_model.pt, sft_best.pt, sft_v2_best.pt)
      - A lightweight LoRA adapter checkpoint (lora_v3_best.pt)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", LlamaConfig())

    if ckpt.get("checkpoint_type") == "lora" or "lora_state_dict" in ckpt:
        base_path = base_checkpoint_override or ckpt.get("base_checkpoint", "checkpoints/base_model.pt")
        if not os.path.exists(base_path):
            # Fallback relative to checkpoint directory
            alt_path = os.path.join(os.path.dirname(checkpoint_path), "base_model.pt")
            if os.path.exists(alt_path):
                base_path = alt_path
            else:
                raise FileNotFoundError(f"Base checkpoint for LoRA not found: {base_path}")

        base_ckpt = torch.load(base_path, map_location=device, weights_only=False)
        model = TinyLlama(config).to(device)
        model.load_state_dict(base_ckpt["model_state_dict"])

        lora_cfg = ckpt["lora_config"]
        inject_lora(
            model,
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg.get("dropout", 0.0),
            target_modules=lora_cfg.get("target_modules", DEFAULT_TARGET_MODULES),
        )
        missing, unexpected = model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys: {unexpected}")

        if merge_lora:
            merge_lora_weights(model)
        return model, ckpt

    model = TinyLlama(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt
