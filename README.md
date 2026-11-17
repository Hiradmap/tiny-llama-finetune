# Tiny_Llama Instruction Fine-Tuning (Full SFT & LoRA)

> **Versioned Supervised Fine-Tuning (SFT) & LoRA from Scratch on a 50.7M Parameter LLaMA Architecture**  
> *Demonstrates the complete progression from a raw pretrained base model (`v0`) through 1.1k Full SFT (`v1`), 15k Expanded Full SFT (`v2`), and Parameter-Efficient LoRA (`v3`) in pure PyTorch.*

---

## 1. Version Progression & Benchmark Results

All versions start from the exact same pretrained `Tiny_Llama` base checkpoint (`base_model.pt`, 50.67M parameters, pretraining stopped at Step 4,500) and are evaluated on the same held-out 2,250-sample validation split (`data/instruction-data-v2.json`).

| Version | Method | Dataset | Trainable Params | Peak VRAM | Checkpoint Size | Val Loss (v2) | Val Perplexity |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`v0`** | Base Pretrained Model | FineWeb-Edu (Pretrain) | — | — | `484.94 MB` | `3.8099` | `45.15` |
| **`v1`** | Full-Weight SFT | `instruction-data.json` (1.1k) | `50,669,056` (100%) | ~3.1 GB | `579.97 MB` | `3.8145` | `45.36` |
| **`v2`** | Full-Weight SFT | `instruction-data-v2.json` (15k) | `50,669,056` (100%) | `3,180.8 MB` | `579.97 MB` | **`3.3820`** | **`29.43`** |
| **`v3`** | **LoRA (`r=16, alpha=32`)** | `instruction-data-v2.json` (15k) | **`1,236,992` (2.38%)** | **`2,542.3 MB`** | **`4.75 MB`** | `3.4227` | `30.65` |

### Key Takeaways Observed
1. **Why `v1` (1.1k) didn't improve `v2` val loss:** With only 1,100 samples, the model learned `<|endoftext|>` stop behavior (stopping instead of rambling), but overfit to ultra-short responses (`3.8145` val loss on unseen instructions).
2. **Data Scaling (`v1` $\rightarrow$ `v2`):** Expanding to 15,000 deduplicated Alpaca Cleaned + Dolly 15k samples (all filtered to fit cleanly inside the 256-token context window) dropped validation perplexity from **`45.15` $\rightarrow$ `29.43`** and produced multi-sentence, structured answers.
3. **LoRA Efficiency (`v2` vs `v3`):** Training just **2.38%** of parameters (`1.24M` low-rank adapter weights) achieved nearly identical perplexity (`30.65` vs `29.43`), reduced peak VRAM by **~638 MB**, and shrank the saved checkpoint by **122$\times$** (`4.75 MB` vs `579.97 MB`).
4. **Sampling Calibration:** Lowering `temperature` from `0.7` $\rightarrow$ `0.3` and raising `repetition_penalty` from `1.15` $\rightarrow$ `1.25` substantially reduces repetition loops in 50M-parameter models.

---

## 2. Qualitative Comparison (`"What is machine learning?"`)

* **`v0` (Base Model):** *"The process of converting machine learning to machine learning is called the “learning” method... There are two types of machine learning: - Machine Learning: This type of computer programming language..."* (Rambles like a web article, hits token limit)
* **`v1` (Full SFT 1.1k):** *"The machine learning process is called machine learning."* (Stops cleanly, too brief)
* **`v2` (Full SFT 15k):** *"Machine Learning is a type of computer-based programming language used to learn and analyze data, such as data or data from various sources... It is also commonly used for applications like data analysis, data processing, and data management."* (Coherent explanation + clean stop)
* **`v3` (LoRA `r=16`, 15k):** *"Machine Learning is a type of machine learning, which involves using machine learning to create and analyze data from various sources. It can be used for many applications..."* (Matches instruction style using only 4.75 MB of adapter weights)

*(Full per-prompt outputs across all 4 versions are saved in [`results/version_comparison.json`](results/version_comparison.json).)*

---

## 3. The Mechanics Implemented from Scratch

### Prompt Template & Response Loss Masking (`dataset_sft.py`)
Each sample is formatted into the Alpaca-style instruction prompt, and prompt tokens are masked with `-1` so `F.cross_entropy(..., ignore_index=-1)` computes gradients **strictly on the assistant's response**:
$$\mathcal{L} = -\frac{1}{N_{\text{response}}} \sum_{i \in \text{response}} \log P(x_i \mid x_{<i})$$

### Low-Rank Adaptation (`lora.py`)
`LoRALinear` freezes pretrained weights $W_0$ and injects trainable matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ ($B = 0$ at initialization):
$$y = W_0 x + \frac{\alpha}{r} B(A(\text{dropout}(x)))$$
Supports targeting `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, saving/loading lightweight adapter-only checkpoints, and merging $\Delta W$ back into $W_0$ for zero-overhead inference.

---

## 4. Project Structure

```
Tiny_Llama_Finetune/
├── data/
│   ├── instruction-data.json       # v1 dataset (1,100 samples)
│   └── instruction-data-v2.json    # v2 dataset (15,000 Alpaca Cleaned + Dolly 15k <= 256 tok)
├── checkpoints/                    # Preserved versioned checkpoints (.pt gitignored)
│   ├── base_model.pt               # v0 base model
│   ├── sft_best.pt                 # v1 Full SFT (1.1k)
│   ├── sft_v2_best.pt              # v2 Full SFT (15k)
│   └── lora_v3_best.pt             # v3 LoRA adapter (15k, 4.75 MB)
├── results/
│   ├── sft_v2_metrics.json         # v2 training curve & hardware metrics
│   ├── lora_v3_metrics.json        # v3 LoRA training curve & hardware metrics
│   └── version_comparison.json     # Side-by-side eval across v0 -> v3
├── model.py                        # TinyLlama architecture (RoPE, RMSNorm, SwiGLU, QK-Norm)
├── dataset_sft.py                  # Instruction formatter, target masking & dynamic padding collator
├── prepare_dataset_v2.py           # Downloads, deduplicates & length-filters v2 dataset
├── train_sft.py                    # Full-Weight SFT training script
├── lora.py                         # Pure-PyTorch LoRA layers, injector, saver & weight merger
├── train_lora.py                   # LoRA SFT training script
├── chat_sft.py                     # Unified interactive CLI (supports Full SFT & LoRA adapters)
└── compare_versions.py             # Automated v0/v1/v2/v3 evaluation & comparison script
```

---

## 5. Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Build the 15k v2 dataset
python prepare_dataset_v2.py

# 3. Run v2 Full-Weight SFT
python train_sft.py --data_path data/instruction-data-v2.json --save_name sft_v2_best.pt --epochs 3

# 4. Run v3 LoRA Fine-Tuning
python train_lora.py --epochs 2 --batch_size 8 --grad_accum_steps 2 --rank 16 --alpha 32 --lr 2e-4

# 5. Chat with any version (default: checkpoints/sft_v2_best.pt, temp=0.3, rep_penalty=1.25)
python chat_sft.py --checkpoint checkpoints/sft_v2_best.pt
python chat_sft.py --checkpoint checkpoints/lora_v3_best.pt

# 6. Run side-by-side comparison across all versions
python compare_versions.py
```
