"""
prepare_dataset_v2.py — Expanded Instruction Dataset Builder (v2)

Downloads and curates high-quality instruction-following samples from:
  1. yahma/alpaca-cleaned (cleaned GPT-4/human instruction pairs)
  2. databricks/databricks-dolly-15k (human-written instruction pairs)

Why filter by token length <= 256?
  Tiny_Llama has a context window (`block_size`) of 256 tokens.
  If a training example is truncated mid-answer, the model never sees the
  <|endoftext|> stop token for that example, encouraging rambling or cut-off answers.
  Filtering to complete sequences <= 256 tokens ensures every training sample
  teaches complete, well-bounded responses.

Preserves `data/instruction-data.json` (v1) untouched and outputs
`data/instruction-data-v2.json`.
"""

import argparse
import json
import os
import random
import urllib.request

import tiktoken

from dataset_sft import format_prompt


ALPACA_URL = "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json"
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def download_if_missing(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        print(f"[Cache] Found existing {dest_path}")
        return
    print(f"[Download] Fetching {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out_f:
        out_f.write(resp.read())
    print(f"[Download] Saved {os.path.getsize(dest_path):,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Build expanded v2 instruction dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max total sequence length (prompt + response + EOT).")
    parser.add_argument("--max_samples", type=int, default=15000, help="Maximum number of samples in v2 dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    raw_alpaca = os.path.join(args.data_dir, "raw_alpaca_cleaned.json")
    raw_dolly = os.path.join(args.data_dir, "raw_dolly_15k.jsonl")
    out_v2 = os.path.join(args.data_dir, "instruction-data-v2.json")

    download_if_missing(ALPACA_URL, raw_alpaca)
    download_if_missing(DOLLY_URL, raw_dolly)

    # Load and normalize schemas to {instruction, input, output, source}
    candidates = []

    with open(raw_alpaca, "r", encoding="utf-8") as f:
        alpaca_data = json.load(f)
    for item in alpaca_data:
        inst = (item.get("instruction") or "").strip()
        inp = (item.get("input") or "").strip()
        out = (item.get("output") or "").strip()
        if inst and out:
            candidates.append({"instruction": inst, "input": inp, "output": out, "source": "alpaca_cleaned"})

    with open(raw_dolly, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            inst = (item.get("instruction") or "").strip()
            inp = (item.get("context") or "").strip()
            out = (item.get("response") or "").strip()
            if inst and out:
                candidates.append({"instruction": inst, "input": inp, "output": out, "source": "dolly_15k"})

    print(f"\nLoaded {len(candidates):,} raw candidates (Alpaca Cleaned + Dolly 15k).")

    # Filter by tokenized sequence length <= max_tokens
    enc = tiktoken.get_encoding("gpt2")
    filtered = []
    seen_instructions = set()

    for entry in candidates:
        # Deduplicate exact instructions
        key = (entry["instruction"].lower(), entry["input"].lower())
        if key in seen_instructions:
            continue
        seen_instructions.add(key)

        prompt_str, response_str = format_prompt(entry)
        total_len = len(enc.encode(prompt_str)) + len(enc.encode(response_str)) + 1  # +1 for EOT
        if total_len <= args.max_tokens:
            filtered.append({
                "instruction": entry["instruction"],
                "input": entry["input"],
                "output": entry["output"],
            })

    print(f"Filtered to {len(filtered):,} deduplicated samples fitting within {args.max_tokens} tokens.")

    random.seed(args.seed)
    random.shuffle(filtered)
    if len(filtered) > args.max_samples:
        filtered = filtered[:args.max_samples]

    with open(out_v2, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {len(filtered):,} samples to {out_v2} ({os.path.getsize(out_v2) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
