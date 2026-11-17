# Tiny_Llama Instruction Fine-Tuning (Full-Weight SFT)

> **Supervised Fine-Tuning (SFT) of a 50.7M Parameter LLaMA Architecture from Scratch in Pure PyTorch**  
> *Adapts a pretrained base causal language model into an instruction-following assistant using prompt formatting and response loss masking.*

---

## 1. Overview

This repository demonstrates **Full-Weight Supervised Fine-Tuning (SFT)** applied to `Tiny_Llama` (~50.7M parameters). 
While pretraining teaches the base language model vocabulary, grammar, and world representations through raw next-token prediction, **fine-tuning teaches the model how to follow instructions and respond as a helpful assistant**.

### Key Architectural Upgrades
- **Modern Transformer Architecture**: RoPE (Rotary Position Embeddings), RMSNorm (pre-normalization), SwiGLU gated activations, QK-Norm, and Flash Attention (`F.scaled_dot_product_attention`).
- **Target Masking (`ignore_index = -1`)**: The loss function strictly evaluates the assistant's response tokens; instruction and prompt tokens are ignored during backpropagation.
- **Pure PyTorch Implementation**: Built cleanly from first principles without heavy third-party abstractions.

---

## 2. The Mechanics of SFT (Prompt Template & Loss Masking)

### Prompt Template
Each training sample is structured using the Alpaca-style instruction format:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response} <|endoftext|>
```

### Response Loss Masking
In causal language modeling, tokens at position $t$ predict tokens at position $t+1$. During fine-tuning, computing loss on the user's prompt degrades instruction adherence and forces the model to memorize user prompts. 

We apply dynamic target masking:
$$\mathcal{L} = -\frac{1}{N_{\text{response}}} \sum_{i \in \text{response}} \log P(x_i \mid x_{<i})$$
Prompt and padding tokens in the target tensor are assigned `-1`, which PyTorch's `F.cross_entropy(..., ignore_index=-1)` ignores during gradient calculations.

---

## 3. Project Structure

```
Tiny_Llama_Finetune/
├── data/
│   └── instruction-data.json    # 1,100 curated Alpaca-style instruction pairs
├── checkpoints/
│   ├── base_model.pt            # Pretrained base checkpoint (symlink / download)
│   └── sft_best.pt              # Fine-tuned assistant checkpoint
├── model.py                     # Full TinyLlama architecture (RMSNorm, RoPE, SwiGLU)
├── dataset_sft.py               # Dataset loader, prompt formatter & target masking collator
├── train_sft.py                 # Full-weight SFT training loop (AdamW + Cosine schedule)
├── chat_sft.py                  # Interactive CLI assistant inference engine
├── requirements.txt             # Minimal dependencies (torch, tiktoken, numpy)
└── README.md                    # Project documentation
```

---

## 4. Quickstart

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/Tiny_Llama_Finetune.git
cd Tiny_Llama_Finetune

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Fine-Tuning
```bash
python train_sft.py --epochs 5 --batch_size 8 --grad_accum_steps 2 --lr 5e-5
```

Key arguments:
- `--base_checkpoint`: Path to pretrained base weights (default: `checkpoints/base_model.pt`).
- `--epochs`: Number of training passes over the instruction dataset (default: `5`).
- `--batch_size`: Micro-batch size (default: `8`).
- `--grad_accum_steps`: Gradient accumulation steps (default: `2`, effective batch size: `16`).
- `--lr`: Peak learning rate for AdamW (default: `5e-5` with linear warmup and cosine decay).

### 3. Interactive Chat
Once fine-tuning completes, chat with your model:
```bash
# Interactive conversation loop
python chat_sft.py

# Single prompt test
python chat_sft.py --prompt "Explain photosynthesis in simple terms."
```

---

## 5. Training Dynamics

- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $=0.01$).
- **Learning Rate Schedule**: Linear warmup (10% of steps) followed by Cosine Annealing to $10^{-6}$.
- **Mixed Precision**: Native `bfloat16` / `float16` with automatic gradient scaling.
- **Hardware Footprint**: Easily trains in $< 3$ minutes on a single 6 GB GPU (RTX 3050 Laptop).
