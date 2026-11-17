"""
chat_sft.py — Interactive Inference & Side-by-Side Comparison for Fine-Tuned Tiny_Llama

=============================================================================
HOW INFERENCE WORKS FOR INSTRUCTION-TUNED MODELS:
=============================================================================
1. Wraps raw user text in the exact prompt template used during SFT training:
   Below is an instruction that describes a task. Write a response that appropriately completes the request.

   ### Instruction:
   {user_prompt}

   ### Response:
   
2. The model generates tokens until:
   - It hits the <|endoftext|> token (ID: 50256)
   - OR reaches max_new_tokens
3. Extracts and displays only the Assistant's answer cleanly in the terminal.
=============================================================================
"""

import argparse
import os
import sys

import tiktoken
import torch

from lora import load_model_from_checkpoint
from model import TinyLlama


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with instruction-tuned Tiny_Llama.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_v2_best.pt",
                        help="Path to SFT or LoRA model checkpoint (e.g. checkpoints/sft_best.pt, checkpoints/sft_v2_best.pt, checkpoints/lora_v3_best.pt).")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Optional override for base checkpoint when loading a LoRA adapter.")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (lower = sharper, less rambling).")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K threshold.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus threshold.")
    parser.add_argument("--repetition_penalty", type=float, default=1.25, help="Repetition penalty.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt mode.")
    return parser.parse_args()


def generate_response(
    model: TinyLlama,
    tokenizer: tiktoken.Encoding,
    instruction: str,
    device: str,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.25
) -> str:
    # Build prompt matching training template
    prompt_str = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
        "### Response:\n"
    )

    prompt_tokens = tokenizer.encode(prompt_str)
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Generation loop
    out_tokens = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    # Decode newly generated tokens only
    gen_tokens = out_tokens[0, len(prompt_tokens):].tolist()

    # Truncate at EOT token if generated
    eot_token = tokenizer.eot_token
    if eot_token in gen_tokens:
        gen_tokens = gen_tokens[:gen_tokens.index(eot_token)]

    return tokenizer.decode(gen_tokens).strip()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        print("Please train the model first with `python train_sft.py` or `python train_lora.py`.")
        sys.exit(1)

    print(f"Loading model from: {args.checkpoint}...")
    model, ckpt = load_model_from_checkpoint(
        args.checkpoint,
        device=device,
        base_checkpoint_override=args.base_checkpoint,
        merge_lora=True,
    )
    model.eval()
    ckpt_kind = "LoRA Adapter (Merged)" if ckpt.get("checkpoint_type") == "lora" else "Full Weights"

    tokenizer = tiktoken.get_encoding("gpt2")

    print("\n" + "=" * 65)
    print(f"  Tiny_Llama SFT Interactive Chatbot [{ckpt_kind}]")
    print(f"  Sampling: Temp={args.temperature} | Top-K={args.top_k} | RepPenalty={args.repetition_penalty}")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 65 + "\n")

    if args.prompt:
        answer = generate_response(
            model, tokenizer, args.prompt, device,
            args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty
        )
        print(f"User:      {args.prompt}")
        print(f"Assistant: {answer}\n")
        return

    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        answer = generate_response(
            model, tokenizer, user_input, device,
            args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty
        )
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
