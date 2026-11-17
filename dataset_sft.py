"""
dataset_sft.py — Supervised Instruction Fine-Tuning Dataset Loader for Tiny_Llama

=============================================================================
KEY CONCEPTS IN INSTRUCTION FINE-TUNING DATASETS:
=============================================================================
1. Prompt Formatting (Alpaca / Raschka Style):
   - With extra context input:
       Below is an instruction that describes a task, paired with an input that provides further context.
       Write a response that appropriately completes the request.

       ### Instruction:
       {instruction}

       ### Input:
       {input}

       ### Response:
       {output}
   - Without extra context:
       Below is an instruction that describes a task.
       Write a response that appropriately completes the request.

       ### Instruction:
       {instruction}

       ### Response:
       {output}

2. Target Masking (Loss Masking):
   In standard pretraining, every single token predicts the next token.
   In instruction fine-tuning, we DO NOT penalize the model for what the user wrote.
   We only compute loss on the ASSISTANT's response.
   All prompt tokens (instruction + input + '### Response:\n') are set to `ignore_index = -1`
   in the target labels tensor.

3. Padding & Batching:
   Sequences in a batch are padded to the longest sequence in that batch (or fixed block_size).
   Padding tokens in targets are also masked with `ignore_index = -1`.
=============================================================================
"""

import json
import os
from typing import Optional, Tuple

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


def format_prompt(entry: dict) -> Tuple[str, str]:
    """
    Formats an instruction entry into (prompt_prefix, response).
    
    Returns:
        prompt_text: Everything up to and including '### Response:\n'
        response_text: The target response text followed by <|endoftext|>
    """
    instruction = entry["instruction"].strip()
    input_text = entry.get("input", "").strip()
    response = entry["output"].strip()

    if input_text:
        prompt_text = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    return prompt_text, response


class InstructionDataset(Dataset):
    """
    PyTorch Dataset that loads instruction JSON data, tokenizes it, and builds
    inputs and target label tensors with prompt-token loss masking.
    """
    def __init__(
        self,
        json_path: str,
        tokenizer: Optional[tiktoken.Encoding] = None,
        max_length: int = 256
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer or tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token  # <|endoftext|> ID: 50256

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Instruction dataset not found at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.encoded_samples = []
        for entry in self.data:
            prompt_str, response_str = format_prompt(entry)
            
            # Tokenize prompt and response separately
            prompt_tokens = self.tokenizer.encode(prompt_str)
            # Response tokens include the End-Of-Text token so the model learns when to STOP generating!
            response_tokens = self.tokenizer.encode(response_str) + [self.eot_token]
            
            full_tokens = prompt_tokens + response_tokens
            
            # Truncate if sequence exceeds max_length
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[:self.max_length]

            # Input tokens: full sequence
            input_ids = torch.tensor(full_tokens, dtype=torch.long)

            # Targets:
            # - Prompt positions are masked with -1 (ignored by PyTorch F.cross_entropy)
            # - Response positions contain the actual token IDs
            targets = input_ids.clone()
            prompt_len = min(len(prompt_tokens), len(full_tokens))
            targets[:prompt_len] = -1

            self.encoded_samples.append({
                "input_ids": input_ids,
                "targets": targets,
                "prompt_len": prompt_len
            })

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int) -> dict:
        return self.encoded_samples[idx]


def custom_collate_fn(batch: list[dict], pad_token_id: int = 50256, ignore_index: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a list of variable-length samples into a padded batch.
    
    In causal next-token prediction:
      - x (input):  tokens [0 ... T-2]
      - y (target): tokens [1 ... T-1]  (shifted by 1)
      
    Padding:
      - x padded with pad_token_id (50256)
      - y padded with ignore_index (-1) so padding is not included in loss
    """
    # Find max length in this specific batch (dynamic padding saves compute)
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_x = []
    batch_y = []

    for item in batch:
        input_ids = item["input_ids"]
        targets = item["targets"]
        seq_len = len(input_ids)

        # Pad input_ids with pad_token_id
        padded_x = torch.full((max_len,), pad_token_id, dtype=torch.long)
        padded_x[:seq_len] = input_ids

        # Pad targets with ignore_index
        padded_y = torch.full((max_len,), ignore_index, dtype=torch.long)
        padded_y[:seq_len] = targets

        # Next-token prediction alignment:
        # x is [:-1], y is [1:]
        batch_x.append(padded_x[:-1])
        batch_y.append(padded_y[1:])

    return torch.stack(batch_x, dim=0), torch.stack(batch_y, dim=0)


def create_instruction_dataloaders(
    json_path: str,
    batch_size: int = 8,
    train_split_ratio: float = 0.85,
    max_length: int = 256,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the instruction dataset into train and validation sets,
    and returns PyTorch DataLoaders with custom collation.
    """
    full_dataset = InstructionDataset(json_path=json_path, max_length=max_length)
    
    total_len = len(full_dataset)
    train_len = int(total_len * train_split_ratio)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=False
    )

    return train_loader, val_loader


# ===========================================================================
# SANITY CHECK & VERIFICATION
# ===========================================================================
if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "data", "instruction-data.json")
    print(f"Loading from: {data_file}")
    train_dl, val_dl = create_instruction_dataloaders(data_file, batch_size=4)
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # Fetch one batch to verify shapes and masking
    for x, y in train_dl:
        print(f"\nBatch X shape: {x.shape} (dtype: {x.dtype})")
        print(f"Batch Y shape: {y.shape} (dtype: {y.dtype})")
        
        # Check masking: count how many tokens are masked (-1) vs active targets (> -1)
        masked_tokens = (y == -1).sum().item()
        active_tokens = (y != -1).sum().item()
        print(f"Total tokens in batch: {y.numel()}")
        print(f"Masked prompt/pad tokens (-1): {masked_tokens} ({masked_tokens / y.numel() * 100:.1f}%)")
        print(f"Active response targets:      {active_tokens} ({active_tokens / y.numel() * 100:.1f}%)")
        break
