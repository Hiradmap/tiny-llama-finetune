"""
benchmark.py — Standardized Benchmarking System for Tiny_Llama

=============================================================================
THE PASSIVE BENCHMARK OBSERVER
=============================================================================
A standardized benchmarking tracker designed to observe and record training
dynamics without altering or slowing down the core training loop:

Key Metrics Tracked:
  - "Per-Step Training Loss": Enables plotting smooth loss convergence curves.
  - "Validation Loss & Perplexity": Evaluated periodically on held-out data.
  - "Wall-Clock Time to Milestones":
      - 4.50 Target: Baseline convergence threshold.
      - 3.80 Stretch Target: Modern architectural convergence threshold.
  - "Training Throughput": Measures tokens/second and step duration.
  - "Hardware Utilization": Tracks peak GPU VRAM allocation.
  - "Sample Generation": Captures qualitative text generation at run completion.
=============================================================================
"""

import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from model import TinyLlama

# ===========================================================================
# 1. BENCHMARK TRACKER CLASS
# ===========================================================================

class BenchmarkTracker:
    """
    Passive observer that records metrics throughout the training lifecycle.
    
    Hook points:
      - on_train_start(): Initializes timers and resets GPU memory counters.
      - on_step(): Records per-step loss, gradient norms, and learning rates.
      - on_eval(): Evaluates validation loss, perplexity, and milestone targets.
      - on_train_end(): Finalizes total wall-clock time and peak VRAM.
    """
    TARGET_LOSS = 4.5
    STRETCH_TARGET = 3.8

    def __init__(self, version: str, model: torch.nn.Module, config: Any):
        self.version = version
        self.model = model
        self.config = config
        self.total_params = sum(p.numel() for p in model.parameters())

        # Storage lists for time-series metrics
        self.eval_history: List[Dict[str, Any]] = []
        self.step_losses: List[float] = []
        self.grad_norms: List[float] = []
        self.muon_lrs: List[float] = []
        self.adamw_lrs: List[float] = []
        self.steps_completed = 0

        # Wall-clock timing
        self.train_start_time: Optional[float] = None
        self.total_training_time: Optional[float] = None
        self.time_to_target: Optional[float] = None
        self.time_to_stretch: Optional[float] = None

        # GPU memory footprint
        self.peak_vram_mb: Optional[float] = None

    def on_train_start(self):
        """Called immediately before the first training iteration."""
        self.train_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        lr_muon: Optional[float] = None,
        lr_adamw: Optional[float] = None
    ):
        """Called after each optimizer update step."""
        self.steps_completed = step + 1
        self.step_losses.append(loss)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        if lr_muon is not None:
            self.muon_lrs.append(lr_muon)
        if lr_adamw is not None:
            self.adamw_lrs.append(lr_adamw)

    def on_eval(self, step: int, train_loss: float, val_loss: float):
        """Called during periodic evaluation checkpoints."""
        elapsed = time.time() - (self.train_start_time or time.time())
        val_ppl = math.exp(min(val_loss, 20.0))

        # Check milestone threshold achievements
        if val_loss <= self.TARGET_LOSS and self.time_to_target is None:
            self.time_to_target = elapsed

        if val_loss <= self.STRETCH_TARGET and self.time_to_stretch is None:
            self.time_to_stretch = elapsed

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        entry = {
            "step": step,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
            "elapsed_seconds": round(elapsed, 2),
            "vram_mb": round(vram_mb, 1)
        }
        self.eval_history.append(entry)

    def on_train_end(self):
        """Called after the training loop terminates."""
        if self.train_start_time is not None:
            self.total_training_time = time.time() - self.train_start_time
        if torch.cuda.is_available():
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)


# ===========================================================================
# 2. GENERATION QUALITY EVALUATION
# ===========================================================================

def evaluate_generation_quality(
    model: TinyLlama,
    tokenizer: Any,
    device: str,
    prompt: str = "To be, or not to be, that is the question:",
    max_new_tokens: int = 40
) -> Dict[str, Any]:
    """Generates a text completion sample to record qualitative linguistic coherence."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    start = time.time()
    with torch.no_grad():
        out_tokens = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    gen_time = time.time() - start

    generated_text = tokenizer.decode(out_tokens[0].tolist())
    tokens_generated = len(out_tokens[0]) - len(tokens)
    gen_speed = tokens_generated / max(gen_time, 1e-6)

    return {
        "prompt": prompt,
        "sample": generated_text,
        "generation_time_sec": round(gen_time, 3),
        "tokens_per_sec": round(gen_speed, 1)
    }


# ===========================================================================
# 3. RESULTS EXPORT (JSON)
# ===========================================================================

def save_results(
    tracker: BenchmarkTracker,
    gen_metrics: Dict[str, Any],
    results_dir: str = "results"
) -> str:
    """Exports structured benchmark metrics and configuration to JSON."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"benchmark_{tracker.version}_{timestamp}.json")

    best_val_loss = min((e["val_loss"] for e in tracker.eval_history), default=None)
    final_val_loss = tracker.eval_history[-1]["val_loss"] if tracker.eval_history else None

    data = {
        "model_architecture": "Tiny_Llama",
        "version": tracker.version,
        "timestamp": timestamp,
        "parameters": {
            "total": tracker.total_params,
            "block_size": tracker.config.block_size,
            "vocab_size": tracker.config.vocab_size,
            "n_layer": tracker.config.n_layer,
            "n_head": tracker.config.n_head,
            "n_embd": tracker.config.n_embd,
            "intermediate_size": tracker.config.intermediate_size,
        },
        "performance": {
            "total_steps": tracker.steps_completed,
            "total_time_seconds": round(tracker.total_training_time or 0, 2),
            "peak_vram_mb": round(tracker.peak_vram_mb or 0, 1),
            "time_to_target_4_5": round(tracker.time_to_target, 2) if tracker.time_to_target else None,
            "time_to_stretch_3_8": round(tracker.time_to_stretch, 2) if tracker.time_to_stretch else None,
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss
        },
        "eval_history": tracker.eval_history,
        "sample_generation": gen_metrics
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[Benchmark] Results saved successfully to: {filepath}")
    return filepath
