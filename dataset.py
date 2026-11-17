"""
dataset.py — Memory-Mapped Fast Binary Data Loader for Tiny_Llama

=============================================================================
WHY MEMORY-MAPPED DATA LOADING?
=============================================================================
In typical NLP workflows, reading an entire multi-million token dataset into
system memory (RAM) is slow, consumes gigabytes of memory, and creates
unnecessary CPU-to-RAM copying bottlenecks.

`np.memmap` solves this by creating a direct pointer to the binary file on disk:
  1. "Zero-RAM Allocation": The OS treats the `.bin` file as virtual memory pages.
  2. "On-Demand Paging": Bytes are pulled from NVMe/SSD directly into OS page cache
     only when a batch index is accessed.
  3. "Instant Initialization": Loading 100M+ tokens takes ~0.001 seconds because
     no array parsing occurs at startup.
=============================================================================
"""

import os
from typing import Tuple

import numpy as np
import torch

# ===========================================================================
# 1. AUTOMATIC DATASET RESOLUTION & SYMLINKING
# ===========================================================================

def ensure_dataset_files(data_dir: str = '.') -> None:
    """
    Checks if 'train.bin' and 'val.bin' are present in the local directory.
    If missing, automatically creates a symbolic link to the shared pre-tokenized
    dataset located in '../Tiny_GPT2/'.
    
    Why symlinking?
    - Prevents storing multiple 200MB+ duplicate copies of binary files on disk.
    - Enables instantaneous plug-and-play execution across project workspaces.
    """
    for split in ['train', 'val']:
        target_path = os.path.join(data_dir, f'{split}.bin')
        
        # Check if the target file already exists in local directory
        if not os.path.exists(target_path):
            # Check the sibling Tiny_GPT2 project repository
            fallback_source = os.path.join('..', 'Tiny_GPT2', f'{split}.bin')
            if os.path.exists(fallback_source):
                try:
                    abs_source = os.path.abspath(fallback_source)
                    os.symlink(abs_source, target_path)
                    print(f"[Dataset] Linked {target_path} -> {abs_source}")
                except Exception as e:
                    print(f"[Dataset Warning] Could not create symlink for {target_path}: {e}")


# ===========================================================================
# 2. MEMORY-MAPPED DATA LOADER CLASS
# ===========================================================================

class MemmapDataLoader:
    """
    Memory-mapped DataLoader for pre-tokenized binary files (uint16).
    
    Data Layout on Disk:
      - Raw 16-bit unsigned integers (uint16).
      - Range: [0, 50,304), representing GPT-2 BPE token IDs.
      - Each token occupies exactly 2 bytes on disk.
    """
    def __init__(
        self,
        data_dir: str = '.',
        split: str = 'train',
        batch_size: int = 16,
        block_size: int = 256,
        device: str | None = None
    ):
        """
        Initializes the binary memory map.
        
        Args:
            data_dir (str): Directory containing 'train.bin' or 'val.bin'.
            split (str): Dataset partition ('train' or 'val').
            batch_size (int): Micro-batch size (number of sequences per batch).
            block_size (int): Context sequence length (number of tokens per sequence).
            device (str): Destination device ('cuda' or 'cpu').
        """
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Automatically resolve and symlink data if running in a fresh workspace
        ensure_dataset_files(self.data_dir)

        # Locate the binary token file
        bin_path = os.path.join(self.data_dir, f'{split}.bin')
        if not os.path.exists(bin_path):
            fallback_source = os.path.join('..', 'Tiny_GPT2', f'{split}.bin')
            if os.path.exists(fallback_source):
                bin_path = fallback_source
            else:
                raise FileNotFoundError(
                    f"Could not find '{split}.bin' in '{data_dir}' or in '../Tiny_GPT2/'!\n"
                    f"Please verify that tokenized data exists."
                )

        # Step 1: Open the binary file as a read-only numpy memory map
        # dtype=np.uint16 is required because each token ID was saved as a 2-byte integer
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')

        # Step 2: Determine the maximum safe starting index
        # We need (block_size + 1) contiguous tokens for input x and target y (shifted by 1)
        self.max_idx = len(self.data) - (self.block_size + 1)
        assert self.max_idx > 0, f"Dataset '{bin_path}' is too small ({len(self.data)} tokens) for block_size {self.block_size}!"

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a random micro-batch of input sequences and next-token target sequences.
        
        Slicing Logic:
          - For each random index i:
              input  x[j] = tokens[i     : i + block_size]
              target y[j] = tokens[i + 1 : i + 1 + block_size]
              
        Returns:
            x (torch.Tensor): Input token IDs of shape (batch_size, block_size) on device.
            y (torch.Tensor): Target token IDs of shape (batch_size, block_size) on device.
        """
        # Step 1: Pick random starting token offsets across the entire dataset
        # .tolist() converts to plain Python ints so numpy slicing avoids per-element
        # torch-scalar -> Python int conversion overhead on every loop iteration.
        ix = torch.randint(0, self.max_idx, (self.batch_size,)).tolist()

        # Step 2: Slice input x and target y directly from disk
        # We cast to np.int64 because PyTorch embedding layers (nn.Embedding) require 64-bit indices
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])

        # Step 3: Asynchronously transfer tensors to target device (GPU VRAM)
        if self.device != 'cpu':
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

        return x, y


# ===========================================================================
# 3. VERIFICATION & SANITY CHECK
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing MemmapDataLoader for Tiny_Llama")
    print("=" * 60)
    
    loader = MemmapDataLoader(data_dir='.', split='train', batch_size=4, block_size=32)
    x, y = loader.get_batch()
    
    print(f"Total tokens in memory map: {len(loader.data):,}")
    print(f"Input batch shape (x):      {x.shape} on {x.device} (dtype: {x.dtype})")
    print(f"Target batch shape (y):     {y.shape} on {y.device} (dtype: {y.dtype})")
    
    print("\nSample first row (x):")
    print(x[0].tolist())
    print("\nSample next-token target row (y):")
    print(y[0].tolist())
    print("\nDataset loader verified successfully!")
