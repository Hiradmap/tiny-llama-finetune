"""
model.py — Modern LLaMA Architecture for Tiny_Llama (2024–2026 Open LLM Standards)

=============================================================================
ARCHITECTURAL EVOLUTION: GPT-2 (2019) -> Tiny_Llama (2024–2026)
=============================================================================
While GPT-2 established the foundation of autoregressive decoder Transformers,
modern open-weight models (LLaMA 3, Mistral, Qwen 2.5, Gemma 2) introduced
pivotal architectural upgrades that dramatically improve training stability,
inference throughput, and sample efficiency:

  1. "Rotary Position Embeddings (RoPE)":
     Replaces absolute learned position tables (wpe). Encodes token position
     directly into Query and Key vectors via 2D rotation matrices. Naturally
     captures relative token distances (m - n) and supports length generalization.

  2. "RMSNorm (Root Mean Square Normalization)":
     Replaces standard LayerNorm by removing mean-centering and additive bias.
     Computes x / RMS(x) * gamma. ~15% faster with superior FP16/BF16 stability.

  3. "SwiGLU (Gated Linear Unit) Feed-Forward Network":
     Replaces standard 4x GELU MLPs with multiplicative gating:
     (SiLU(x @ W_gate) * (x @ W_up)) @ W_down. The gating mechanism allows the
     network to dynamically suppress or amplify information pathways.

  4. "QK-Norm (Query-Key RMSNorm)":
     Normalizes Query and Key head vectors with RMSNorm before computing dot
     products. Prevents attention logits from drifting/exploding during long runs.

  5. "Bias-Free Linear Layers (bias=False)":
     Removes additive bias vectors across all linear projections, reducing
     parameter overhead and preventing bias gradient drift.

  6. "Tensor Core Vocabulary Padding (50,304 tokens)":
     Pads the GPT-2 vocabulary (50,257) to the nearest multiple of 64.
     Aligns matrix dimensions to NVIDIA GPU 32-thread warps and Tensor Core tiles.
=============================================================================
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class LlamaConfig:
    """
    A frozen blueprint of every architectural hyperparameter for Tiny_Llama.
    
    Using a dataclass keeps all hyperparameters in one unified structure,
    making it trivial to scale the model up or down.
    
    Target Budget: ~51.5 Million Parameters for local GPU training.
    """
    # Context Length: Maximum number of tokens the model can process in a single sequence
    block_size: int = 256
    
    # Vocabulary Size: Padded from 50,257 to 50,304 (divisible by 64 for NVIDIA Tensor Core warp execution)
    vocab_size: int = 50304
    
    # Depth: Number of stacked Transformer blocks (layers)
    n_layer: int = 8
    
    # Attention Heads: Number of parallel attention heads inside each block
    n_head: int = 8
    
    # Width (d_model): The continuous vector dimension representing each token throughout the network
    n_embd: int = 512
    
    # SwiGLU Hidden Dimension: Intermediate dimension for the gated feed-forward block
    # Computed as ~ (8/3) * n_embd rounded to nearest multiple of 64: (8/3)*512 = 1365 -> 1344
    intermediate_size: int = 1344
    
    # RoPE Base Frequency: Controls the wavelength scaling for rotary position frequencies
    rope_theta: float = 10000.0
    
    # Numerical Epsilon: Small constant added to denominator in RMSNorm to avoid division by zero
    eps: float = 1e-6


# ===========================================================================
# 1. THE NORMALIZATION PHASE — RMSNorm (Root Mean Square Normalization)
# ===========================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    
    "Traditional LayerNorm calculates both the mean and standard deviation of
    inputs, shifting by the mean and dividing by std:
        LayerNorm(x) = ((x - mean(x)) / std(x)) * gamma + beta
        
    RMSNorm makes a crucial mathematical observation:
    The shift invariance (subtracting mean(x)) provides virtually zero regularization
    benefit, whereas the scaling invariance (dividing by root mean square) accounts
    for nearly 100% of training stability.
    
    By discarding mean-centering and additive bias (beta), RMSNorm:
      1. Drops ~15% of arithmetic operations per normalization layer.
      2. Removes additive drift from bias gradients.
      3. Operates with superior stability in mixed precision (bfloat16/float16)."
    
    Mathematical Formula:
        RMS(x) = sqrt( (1 / d) * sum_{i=1}^d x_i^2 + eps )
        RMSNorm(x) = (x / RMS(x)) * gamma
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Epsilon constant prevents division by zero when activations are all zeros
        self.eps = eps
        
        # Learnable scaling vector (gamma) of shape (dim,) initialized to 1.0 (identity scale)
        # "Unlike LayerNorm which has both weight (gamma) and bias (beta), RMSNorm only learns gamma."
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Compute the mean of squared activations along the last dimension (d_model or head_dim)
        # x shape: (..., dim) -> variance shape: (..., 1)
        variance = x.pow(2).mean(-1, keepdim=True) # pow: element-wise power. mean: compute the mean of the elements along the given dimension. keepdim: keep the dimension of the reduced dimension
        
        # Step 2: Compute reciprocal square root: 1.0 / sqrt(variance + eps)
        # Using torch.rsqrt is a single fused hardware instruction on modern GPUs (faster than sqrt + div)
        rsqrt = torch.rsqrt(variance + self.eps) # rsqrt: reciprocal square root
        
        # Step 3: Scale normalized inputs element-wise by the learnable parameter self.weight (gamma)
        # Broadcasting: (..., dim) * (..., 1) * (dim,) -> (..., dim)
        return x * rsqrt * self.weight


class ResidualProjection(nn.Linear):
    """Linear projection initialized at a smaller scale for residual additions."""


# ===========================================================================
# 2. POSITIONAL ENCODING — Rotary Position Embeddings (RoPE)
# ===========================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    "In legacy GPT-2, positional information was injected by looking up a learned
    absolute position embedding table (wpe) and adding it to token embeddings at Layer 0:
        x = token_embedding + position_embedding[pos]
        
    This had two major flaws:
      1. Absolute positions do not explicitly encode relative distances (token i vs token j).
      2. The model cannot easily generalize to sequence lengths beyond the training context.
      
    RoPE revolutionizes positional encoding by framing position as a 2D ROTATION.
    Instead of adding vectors, RoPE rotates the Query and Key vectors in the complex plane:
        Query_m = R_{theta, m} * Query
        Key_n   = R_{theta, n} * Key
        
    When Query at position m attends to Key at position n, the inner product is:
        <Query_m, Key_n> = (R_m q)^T (R_n k) = q^T (R_m^T R_n) k = q^T R_{n - m} k
        
    Notice that the result depends ONLY on the relative offset (n - m)!
    Relative token order is preserved naturally across all attention layers."
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        # dim: Dimension of each individual attention head (head_dim = n_embd // n_head = 64)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Step 1: Compute inverse angular frequencies for each 2D coordinate pair
        # Formula: theta_i = 1.0 / (theta ^ (2i / dim)) for i in [0, 1, 2, ..., dim/2 - 1]
        # High frequencies (small wavelengths) capture local grammar;
        # Low frequencies (long wavelengths) capture long-range discourse structure.
        # Shape: (dim / 2,) -> e.g., 32 frequencies for head_dim=64
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)) 
        
        # Register as a non-trainable persistent buffer so it moves with the model device
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Declare typed annotations so static type checkers (Pylance/pyright) know these
        # will always be Tensors by the time forward() runs, even though they are set
        # dynamically via register_buffer inside _build_cache.
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor

        # Step 2: Precompute cos and sin trigonometric lookup tables up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precomputes and caches cos and sin matrices for sequence positions [0 ... seq_len-1]."""
        # Position indices: [0, 1, 2, ..., seq_len - 1] of shape (seq_len,)
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Outer product: [seq_len, dim / 2]
        # Computes the rotation angle (m * theta_i) for every position m and frequency index i
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate frequencies across the full head dimension: [seq_len, dim]
        # Layout: [f_0, f_1, ..., f_{d/2-1}, f_0, f_1, ..., f_{d/2-1}]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape cache with batch and head singleton dimensions for automatic PyTorch broadcasting:
        # Target broadcasting shape: (1, 1, seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns cached (cos, sin) tensors sliced to the exact current sequence length."""
        # If input sequence exceeds existing cache size, dynamically extend cache
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
            
        # Slice cache from position 0 to seq_len and cast to current computation device and precision
        cos = self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the coordinates of vector x by 90 degrees in 2D pairs.
    
    Given vector x = [x1, x2], the 90-degree orthogonal rotation is [-x2, x1].
    
    Mathematical derivation:
      A 2D rotation of vector [x1, x2] by angle theta is:
        [x1*cos(theta) - x2*sin(theta), x1*sin(theta) + x2*cos(theta)]
      This can be rewritten in vectorized form as:
        x * cos(theta) + [-x2, x1] * sin(theta)
      where rotate_half(x) produces [-x2, x1].
    """
    # Slice the first half (x1) and second half (x2) along the head dimension
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    # Concatenate [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding rotation to Query or Key tensor x.
    
    Vectorized Euler Rotation:
        Rotated_x = (x * cos) + (rotate_half(x) * sin)
        
    Args:
        x (torch.Tensor): Query or Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos (torch.Tensor): Precomputed cosine angles of shape (1, 1, seq_len, head_dim)
        sin (torch.Tensor): Precomputed sine angles of shape (1, 1, seq_len, head_dim)
        
    Returns:
        torch.Tensor: Position-rotated tensor with identical shape and precision.
    """
    return (x * cos) + (rotate_half(x) * sin)


# ===========================================================================
# 3. THE COMMUNICATION PHASE — Causal Attention with QK-Norm & RoPE
# ===========================================================================

class LlamaAttention(nn.Module):
    """
    Modern Multi-Head Causal Self-Attention with QK-Norm, RoPE, and Flash Attention.
    
    "Attention is the COMMUNICATION mechanism of the Transformer. It allows each token
    in a sequence to look back at all previous tokens, assess their relevance, and
    gather contextual information to update its internal representation."
    
    Key Innovations in Modern Attention:
      1. "Bias-Free Projections": All linear projections use bias=False, eliminating
         parameter waste and stabilizing weight gradients.
      2. "QK-Norm (Query-Key Normalization)":
         Applies RMSNorm to Query and Key head vectors BEFORE computing dot products.
         Why this is critical: In standard attention, unnormalized Q and K vectors can
         grow in magnitude during training, driving dot products into the flat saturation
         zones of Softmax. This causes near-zero gradients and entropy collapse.
         QK-Norm mathematically bounds attention logits, guaranteeing smooth learning.
      3. "RoPE Rotation": Dynamically applies rotary embeddings to normalized Q and K.
      4. "Flash Attention (F.scaled_dot_product_attention)":
         Executes fused memory-efficient CUDA kernels directly in GPU SRAM.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # Validate that total embedding width divides evenly into head slices
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # e.g., 512 / 8 = 64 dimensions per head
        
        # --- Linear Projections (Bias-Free) ---
        # Query Projection: "What kind of information am I looking for?"
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Key Projection: "What kind of information do I contain?"
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Value Projection: "What actual content do I communicate if attended to?"
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Output Projection: Merges multi-head perspectives back into a single vector of size n_embd
        self.o_proj = ResidualProjection(config.n_embd, config.n_embd, bias=False)
        
        # --- QK-Norm Modules ---
        # Independent RMSNorm instances applied to 64-dim Query and Key head vectors
        self.q_norm = RMSNorm(self.head_dim, eps=config.eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.eps)
        
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        # Unpack input tensor dimensions:
        # B = Batch size (sequences processed in parallel)
        # T = Sequence length (number of tokens in context)
        # C = Channel / embedding dimension (n_embd = 512)
        B, T, C = x.shape
        
        # --- Step 1: Compute Query, Key, and Value projections ---
        # Shape of each: (B, T, C)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # --- Step 2: Reshape into Multi-Head format ---
        # .view(B, T, n_head, head_dim) divides the 512-dim vector into 8 heads of 64 dims.
        # .transpose(1, 2) swaps sequence length T and n_head dimensions.
        # Output shape: (B, n_head, T, head_dim)
        # "Each head now operates independently in its own 64-dimensional subspace,
        # learning to track distinct syntactic or semantic patterns."
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # --- Step 3: Apply QK-Norm ---
        # Normalizes each 64-dimensional Query and Key head vector
        # Bounds dot-product magnitudes so attention softmax never saturates
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # --- Step 4: Apply Rotary Position Embeddings (RoPE) ---
        # Injects relative token position coordinates by rotating Q and K in 2D pairs
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # --- Step 5: Flash Attention Computation ---
        # Fused GPU kernel computes: Attention = Softmax((Q @ K.T) / sqrt(head_dim)) @ V
        # is_causal=True enforces lower-triangular causal masking (tokens cannot see the future)
        # Computes directly in GPU SRAM without materializing the O(T^2) attention matrix in VRAM!
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )
        
        # --- Step 6: Recombine attention heads ---
        # Reverse transpose: (B, n_head, T, head_dim) -> (B, T, n_head, head_dim)
        # .contiguous() ensures tensor memory is contiguous before calling .view()
        # .view(B, T, C) flattens heads back into a single continuous vector of size n_embd
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # --- Step 7: Output projection ---
        # Projects concatenated multi-head summaries back through the output linear layer
        return self.o_proj(y)


# ===========================================================================
# 4. THE COMPUTATION PHASE — SwiGLU Gated Feed-Forward Network
# ===========================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU Gated Feed-Forward Network.
    
    "If Attention is about COMMUNICATION (tokens gathering context from each other),
    then SwiGLU is about COMPUTATION (each token processing and transforming what it learned)."
    
    Legacy GPT-2 used standard 2-layer MLPs with GELU activation:
        MLP(x) = GELU(x @ W_fc) @ W_proj
        
    SwiGLU replaces this with a Gated Linear Unit using SiLU activation:
        SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
        
    Why SwiGLU is significantly superior:
      1. "Multiplicative Gating": The gate projection acts as a continuous, differentiable
         switch that dynamically opens or closes feature channels.
      2. "SiLU Smoothness": SiLU(z) = z * sigmoid(z) provides smooth non-linear gradients
         with zero saturation in the positive domain.
      3. "FLOP Parity via 8/3 Scaling":
         Because SwiGLU uses 3 weight matrices instead of 2, we scale intermediate_size
         down to (8/3) * n_embd = 1,344 channels (instead of standard 4x = 2,048),
         delivering much higher representational capacity at the EXACT same FLOP cost!
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # Gate Projection: Computes the continuous gating activation signal
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        
        # Up Projection: Computes the candidate feature representation
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        
        # Down Projection: Compresses the gated feature representation back to n_embd
        self.down_proj = ResidualProjection(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Compute gate projection and apply SiLU activation: SiLU(x @ W_gate)
        gate = F.silu(self.gate_proj(x))
        
        # Step 2: Compute up projection candidate features: (x @ W_up)
        up = self.up_proj(x)
        
        # Step 3: Multiplicative element-wise gating followed by down projection
        # Formula: (gate * up) @ W_down
        return self.down_proj(gate * up)


# ===========================================================================
# 5. THE TRANSFORMER BLOCK — Pre-RMSNorm Residual Layer
# ===========================================================================

class LlamaBlock(nn.Module):
    """
    Standard Pre-RMSNorm LLaMA Transformer Block.
    
    "Each block represents one floor in the transformer skyscraper.
    Data enters, undergoes communication (Attention), then computation (SwiGLU),
    and ascends to the next floor."
    
    The Pre-RMSNorm Residual Architecture:
      - We normalize BEFORE each sub-layer rather than after.
      - The residual connections (x = x + ...) create a clean, unobstructed 'highway'
        where gradients can flow backwards without vanishing or exploding.
        
    Diagram:
      Residual Highway ──┬──> [RMSNorm] ──> [Attention] ──> (+) ──┬──> [RMSNorm] ──> [SwiGLU] ──> (+) ──> Next Layer
                         │                                   ▲     │                               ▲
                         └───────────────────────────────────┘     └───────────────────────────────┘
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # Normalization layer preceding Causal Attention
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.eps)
        
        # Attention sub-layer (Communication)
        self.self_attn = LlamaAttention(config)
        
        # Normalization layer preceding SwiGLU MLP
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.eps)
        
        # Feed-forward sub-layer (Computation)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        # Sub-Layer 1: Pre-RMSNorm -> Attention -> Add back to residual stream
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        
        # Sub-Layer 2: Pre-RMSNorm -> SwiGLU MLP -> Add back to residual stream
        x = x + self.mlp(self.post_attention_layernorm(x))
        
        return x


# ===========================================================================
# 6. FULL MODEL — TinyLlama Causal Language Model
# ===========================================================================

class TinyLlama(nn.Module):
    """
    Full Tiny_Llama Autoregressive Decoder Language Model.
    
    End-to-End Execution Flow:
      1. Token IDs: Integer sequence of shape (B, T) -> Token Embedding Table -> (B, T, n_embd)
      2. RoPE Frequencies: Precomputed (cos, sin) angle tables sliced to length T
      3. Transformer Stack: Passed sequentially through 8 Pre-RMSNorm LlamaBlocks
      4. Final Normalization: Normalized through a final RMSNorm layer
      5. LM Head: Projected back to vocabulary space (B, T, vocab_size)
      6. Loss Computation: If training targets are provided, computes cross-entropy loss
      
    Key Architectural Details:
      - "Weight Tying": Shares weights between the token embedding table and final lm_head.
        Saves ~25.7M parameters while improving token representation quality.
      - "Tensor Core Alignment": Vocabulary size 50,304 is a clean multiple of 64.
      - "Scaled Gaussian Init": Projection layers feeding into residuals are scaled
        by 1 / sqrt(2 * n_layer) to prevent activation variance explosion.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding Table: Maps discrete token IDs [0 ... 50303] to dense continuous vectors (n_embd=512)
        # "This is the model's vocabulary dictionary — learning a unique 512-dim embedding for every token."
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Rotary Embedding Precomputer (generates cos and sin angle caches)
        self.rotary_emb = RotaryEmbedding(
            dim=config.n_embd // config.n_head,
            max_seq_len=config.block_size,
            theta=config.rope_theta
        )
        
        # The stack of 8 Transformer Blocks (floors of the skyscraper)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)])
        
        # Final RMSNorm layer applied before language modeling head
        self.norm = RMSNorm(config.n_embd, eps=config.eps)
        
        # Language Modeling Head: Projects hidden vectors (n_embd) back to vocabulary scores (vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Tying: Share weights between token embedding and output projection
        # "Tokens that have similar embeddings should naturally receive similar probability logits."
        self.tok_embeddings.weight = self.lm_head.weight
        
        # Initialize all model weights with scaled Gaussian distribution
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Scaled Gaussian Weight Initialization.
        
        Standard layers: Gaussian distribution with mean=0.0, std=0.02.
        Residual projection layers:
          Scaled down by 1 / sqrt(2 * n_layer) to keep activation variance ~1.0
          across deep residual addition highways.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if isinstance(module, ResidualProjection):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            # RMSNorm scaling gains (gamma) start initialized to 1.0 (identity scaling)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward execution pass for training or inference.
        
        Args:
            idx (torch.Tensor): Conditioning token IDs of shape (B, T).
            targets (torch.Tensor, optional): Next-token target IDs of shape (B, T).
            
        Returns:
            logits (torch.Tensor): Unnormalized token logits of shape (B, T, vocab_size).
            loss (torch.Tensor, optional): Cross-entropy loss scalar (None if targets is None).
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"
        
        # Step 1: Look up token embeddings: (B, T) -> (B, T, n_embd)
        x = self.tok_embeddings(idx)
        
        # Step 2: Retrieve RoPE (cos, sin) frequency tensors matching sequence length T
        cos, sin = self.rotary_emb(T, device=idx.device, dtype=x.dtype)
        
        # Step 3: Pass sequentially through all 8 Transformer Blocks
        for layer in self.layers:
            x = layer(x, cos, sin)
            
        # Step 4: Apply final RMSNorm layer
        x = self.norm(x)
        
        # Step 5: Compute Logits & Loss
        if targets is not None:
            # Training Mode: Compute logits for ALL positions in the sequence
            logits = self.lm_head(x)  # Shape: (B, T, vocab_size)
            
            # Compute Cross-Entropy loss across all tokens in the batch
            # .view(-1, vocab_size) flattens (B, T) into a single list of tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference Mode: Only project logits for the FINAL token position T to save compute
            logits = self.lm_head(x[:, [-1], :])  # Shape: (B, 1, vocab_size)
            loss = None
            
        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculates total trainable parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Exclude shared embedding/lm_head table
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, Top-K, Top-P, and repetition penalty.
        
        Args:
            idx (torch.Tensor): Seed prompt token IDs of shape (B, T).
            max_new_tokens (int): Number of subsequent tokens to generate.
            temperature (float): Softmax temperature (0.0 = greedy, 1.0 = standard).
            top_k (int, optional): Top-K sampling cutoff threshold.
            top_p (float, optional): Nucleus sampling cumulative probability cutoff.
            repetition_penalty (float): Multiplicative penalty against repeated tokens.
            
        Returns:
            torch.Tensor: Full sequence containing prompt + generated tokens of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop context if sequence length exceeds maximum context window
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Step 1: Forward pass to obtain logits for the last token position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Shape: (B, vocab_size)

            # Defensive Mask: Ensure padded vocabulary tokens (> 50,256) are never sampled
            if logits.size(-1) > 50257:
                logits[:, 50257:] = -float('inf')
            
            # Step 2: Apply Repetition Penalty
            # Vectorized: gather logits at all seen token positions in one shot,
            # penalize them (divide positives, multiply negatives), scatter back.
            # This avoids per-token GPU->CPU round-trips from the old Python loop.
            if repetition_penalty != 1.0:
                score = logits.gather(1, idx)
                score = torch.where(
                    score < 0,
                    score * repetition_penalty,
                    score / repetition_penalty
                )
                logits.scatter_(1, idx, score)

            # Step 3: Temperature Scaling & Greedy Sampling
            if temperature == 0.0:
                # Greedy deterministic choice: pick the single highest probability token
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Divide logits by temperature: lower T sharpens distribution, higher T flattens it
                logits = logits / temperature
                
                # Step 4: Top-K Filtering
                # Keep only the top K highest scoring tokens; mask out the rest with -inf
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    
                # Step 5: Top-P (Nucleus) Filtering
                # Dynamically keep the smallest set of tokens whose cumulative probability >= top_p
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Create boolean mask for tokens exceeding top_p threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices right by 1 to always keep at least the single best token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float('Inf')
                    
                # Step 6: Convert filtered logits to probabilities and sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            # Step 7: Append newly generated token ID to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


# ===========================================================================
# SANITY CHECK & ARCHITECTURE VERIFICATION
# ===========================================================================

if __name__ == "__main__":
    config = LlamaConfig()
    model = TinyLlama(config)
    
    total_params = model.get_num_params()
    non_emb_params = model.get_num_params(non_embedding=True)
    
    print("=" * 70)
    print("  Tiny_Llama Architecture Verification")
    print("=" * 70)
    print(f"Total Parameters:        {total_params:,} (~{total_params / 1e6:.2f}M)")
    print(f"Non-Embedding Params:    {non_emb_params:,} (~{non_emb_params / 1e6:.2f}M)")
    print(f"Vocabulary Size:         {config.vocab_size:,} (Tensor Core aligned)")
    print(f"Context Length:          {config.block_size} tokens")
    print(f"Hidden Dimension:        {config.n_embd} (d_model)")
    print(f"SwiGLU Intermediate:     {config.intermediate_size} channels")
    print(f"Layers / Heads:          {config.n_layer} layers / {config.n_head} heads (head_dim={config.n_embd // config.n_head})")
    print("=" * 70)
    
    # Test forward pass with dummy batch
    dummy_x = torch.randint(0, config.vocab_size, (2, 64))
    dummy_y = torch.randint(0, config.vocab_size, (2, 64))
    
    logits, loss = model(dummy_x, dummy_y)
    print(f"Forward output logits shape:  {logits.shape}")
    print(f"Forward computed loss:        {loss.item():.4f}")
    
    # Test generation
    gen_out = model.generate(dummy_x[:1, :10], max_new_tokens=15, temperature=0.8)
    print(f"Generation output shape:      {gen_out.shape}")
    print("Architecture verified successfully!")
