"""
generate.py — Interactive Text Generation CLI for Tiny_Llama

=============================================================================
HOW AUTOREGRESSIVE GENERATION & SAMPLING WORK
=============================================================================
Autoregressive language models predict one token at a time:
  1. Input prompt tokens [x_1, x_2, ..., x_t] are passed through Tiny_Llama.
  2. The model outputs unnormalized logits across all 50,304 vocabulary tokens.
  3. Sampling controls modify logits to balance creativity and coherence:
     - "Temperature (T)": Rescales logits (logits / T). Lower T (<1.0) sharpens
       probabilities towards the most likely tokens; higher T (>1.0) flattens them.
     - "Repetition Penalty": Penalizes tokens that have already been generated,
       preventing degenerate repeating loops.
     - "Top-K Filtering": Truncates the distribution to keep only the top K
       most probable tokens, eliminating low-probability tail nonsense.
     - "Top-P (Nucleus) Filtering": Dynamically selects the smallest set of
       tokens whose cumulative probability exceeds P (e.g., 0.90).
  4. A token is sampled from the filtered distribution and appended to the prompt.
  5. The loop repeats until max_new_tokens is reached.
=============================================================================
"""

import argparse
import os
import sys
import time
from typing import Tuple

import tiktoken
import torch

from model import LlamaConfig, TinyLlama

# ===========================================================================
# 1. COMMAND LINE PARSING
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with trained Tiny_Llama.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for single-shot generation.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive multi-turn REPL.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Softmax temperature (0.0 = greedy).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K filtering threshold (0 = disabled).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling threshold (1.0 = disabled).")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty multiplier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling reproducibility.")
    return parser.parse_args()


# ===========================================================================
# 2. MODEL LOADER
# ===========================================================================

def load_model(checkpoint_path: str, device: str) -> Tuple[TinyLlama, LlamaConfig]:
    """Loads model weights and configuration from a saved checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found!\n"
            f"Please train the model first by running: python train.py"
        )

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", LlamaConfig())
    model = TinyLlama(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_loss = checkpoint.get("val_loss", None)
    step = checkpoint.get("step", None)
    if step is not None:
        loss_info = f" (Val Loss: {val_loss:.4f})" if val_loss is not None else ""
        print(f"Loaded checkpoint saved at Step {step}{loss_info}")

    return model, config


# ===========================================================================
# 3. TEXT GENERATION PIPELINE
# ===========================================================================

def generate_text(
    model: TinyLlama,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> Tuple[str, float, float]:
    """
    Encodes text prompt, generates new tokens autoregressively, and decodes output.
    """
    # Step 1: Tokenize prompt into BPE token integers
    tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    if len(tokens) == 0:
        tokens = [tokenizer.eot_token]
        
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    # Step 2: Autoregressive token generation loop
    start = time.time()
    with torch.no_grad():
        out_tokens = model.generate(
            idx=idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
            repetition_penalty=repetition_penalty
        )
    elapsed = time.time() - start

    # Step 3: Decode generated tokens back to readable string
    output_token_list = out_tokens[0].tolist() # we take the first row because the output is batch_size * sequence_length
    gen_tokens_count = len(output_token_list) - len(tokens)
    speed = gen_tokens_count / max(elapsed, 1e-6)

    generated_text = tokenizer.decode(output_token_list) # decode the tokens back to readable string
    return generated_text, speed, elapsed


# ===========================================================================
# 4. MAIN ENTRY POINT
# ===========================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    try:
        model, config = load_model(args.checkpoint, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 65)
    print("  Tiny_Llama Interactive Inference Engine")
    print(f"  Device: {device} | Params: {model.get_num_params():,} | Context: {config.block_size}")
    print(f"  Sampling: Temp={args.temperature} | Top-K={args.top_k} | Top-P={args.top_p} | RepPenalty={args.repetition_penalty}")
    print("=" * 65)

    # Interactive REPL mode
    if args.interactive or args.prompt is None:
        print("\nEntering interactive mode. Type your prompt and press Enter (or 'quit' / 'exit' to stop).\n")
        while True:
            try:
                prompt = input(">>> Prompt: ").strip()
                if prompt.lower() in ["quit", "exit", "q"]:
                    print("Exiting.")
                    break
                if not prompt:
                    continue

                output, speed, elapsed = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
                print("\n" + output)
                print(f"\n[Generated {args.max_new_tokens} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)]\n" + "-" * 65)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
    else:
        # Single-shot prompt generation
        output, speed, elapsed = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        print("\n" + output)
        print(f"\n[Generated {args.max_new_tokens} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)]")


if __name__ == "__main__":
    main()
