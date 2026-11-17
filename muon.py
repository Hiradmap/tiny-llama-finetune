"""
muon.py — MomentUm Orthogonalized by Newton-Schulz Optimizer

=============================================================================
WHAT IS THE MUON OPTIMIZER?
=============================================================================
Muon is a matrix optimizer designed specifically for training the internal
2D weight matrices (Linear / Projection layers) of Transformer neural networks.

Why standard optimizers fall short on 2D matrices:
  - "SGD with Momentum": Treats all dimensions equally, leading to slow convergence
    in ill-conditioned loss landscapes.
  - "AdamW": Scales gradients element-wise using second moments (v_t), which works
    well for 1D vectors and embedding tables, but can distort the geometric
    structure of 2D transformation matrices.

How Muon works:
  1. Computes standard SGD Nesterov momentum buffer for each 2D weight matrix.
  2. Orthogonalizes the update matrix using 5 iterations of the Newton-Schulz
     polynomial algorithm: f(X) = a*X + b*(X @ X^T) @ X + c*((X @ X^T)^2) @ X.
  3. Scales the update according to the matrix aspect ratio: sqrt(max(1, M/N)).
  4. Takes a coordinate-free orthogonal step in parameter space!

References:
  - Keller Jordan's modded-nanogpt repository
  - "Muon: An Optimizer for Hidden Layers of Neural Networks"
=============================================================================
"""

from typing import Callable, List, Optional, Tuple, overload

import torch
import torch.nn as nn

from model import TinyLlama

# ===========================================================================
# 1. NEWTON-SCHULZ 5TH-ORDER POLYNOMIAL ORTHOGONALIZATION
# ===========================================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Computes the orthogonalized matrix (zeroth matrix power) of 2D matrix G using 
    a 5th-order Newton-Schulz iterative polynomial algorithm.

    Mathematical Intuition:
      - An orthogonal matrix U satisfies U @ U.T = I (all singular values are 1.0).
      - Singular Value Decomposition (SVD) computes this exactly (U @ S^0 @ V^T = U @ V^T),
        but SVD is prohibitively slow on GPUs.
      - Newton-Schulz iteration repeatedly applies a polynomial function to G to push 
        all singular values toward 1.0 in just 5 fast GPU matrix multiplications!
    
    Args:
        G (torch.Tensor): 2D gradient or momentum update matrix of shape (M, N).
        steps (int): Number of polynomial refinement iterations (default: 5).
        
    Returns:
        torch.Tensor: Orthogonalized update matrix with identical shape and dtype.
    """
    # Step 1: Validation — Muon strictly operates on 2D weight matrices
    assert len(G.shape) == 2, f"Expected 2D tensor for Newton-Schulz, got shape {G.shape}"

    # Step 2: Optimal 5th-order polynomial coefficients:
    # f(X) = a*X + b*(X @ X.T) @ X + c*((X @ X.T) @ (X @ X.T)) @ X
    # These exact constants (3.4445, -4.7750, 2.0315) maximize the convergence domain.
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Step 3: Precision handling
    # Float16 has narrow dynamic range and risks overflow/NaNs during matrix powers.
    # We maintain bfloat16 (which has wide exponent range) or cast to float32.
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()

    # Step 4: Dimension optimization (Operating on smaller square dimension)
    # If rows > cols (tall matrix), transpose X so shape becomes (N, M) with N < M.
    # Why? (X @ X.T) creates an (N x N) matrix. Smaller N speeds up GPU matrix multiplies!
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True

    # Step 5: Normalization step
    # Scale matrix so its Frobenius norm is ~1.0, ensuring initial singular values
    # fall safely within the polynomial convergence domain (< sqrt(3)).
    X = X / (X.norm() + 1e-7)

    # Step 6: 5-step Newton-Schulz polynomial iteration loop
    for _ in range(steps):
        # A has shape (N, N) where N <= M (smaller dimension)
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    # Step 7: Restore original orientation if transposed earlier
    if transposed:
        X = X.T

    # Step 8: Cast back to original tensor precision
    return X.type_as(G)


# ===========================================================================
# 2. MUON OPTIMIZER CLASS
# ===========================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon Custom PyTorch Optimizer (subclasses torch.optim.Optimizer).
    
    Optimizes 2D internal linear weight matrices using Nesterov momentum
    followed by Newton-Schulz matrix orthogonalization.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0
    ):
        """
        Args:
            params: Iterable of torch.Tensor parameters to optimize.
            lr (float): Base learning rate for Muon updates (default: 0.02).
            momentum (float): Nesterov momentum coefficient (default: 0.95).
            nesterov (bool): Whether to use Nesterov accelerated momentum lookahead (default: True).
            ns_steps (int): Newton-Schulz iteration refinement count (default: 5).
            weight_decay (float): L2 regularization weight decay factor (default: 0.0).
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step across all parameter groups."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over parameter groups
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']

            # Iterate over individual parameter tensors
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError('Muon optimizer does not support sparse gradients')

                state = self.state[p]
                
                # Lazy initialization of momentum velocity buffer on first step
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # Step 1: Apply L2 weight decay if specified: g = g + weight_decay * p
                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)

                # Step 2: Update momentum velocity: buf = momentum * buf + g
                buf.mul_(momentum).add_(g)

                # Step 3: Compute update vector (Nesterov vs Standard momentum)
                if nesterov:
                    # Nesterov lookahead: update = g + momentum * buf
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                # Step 4: Orthogonalize and update weight matrix
                if update.ndim == 2:
                    # A) Orthogonalize update matrix via Newton-Schulz
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    
                    # B) Aspect Ratio Scaling (Keller Jordan):
                    #    Scales update magnitude by sqrt(max(1, M/N)) to ensure tall layers
                    #    learn at the same effective pace as square layers.
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    
                    # C) In-place weight update: W = W - (lr * scale) * update
                    p.data.add_(update, alpha=-lr * scale)
                else:
                    # Fallback for non-2D tensors: standard SGD step
                    p.data.add_(update, alpha=-lr)

        return loss


# ===========================================================================
# 3. DUAL-OPTIMIZER SETUP & PARAMETER PARTITIONING
# ===========================================================================

def setup_optimizers(
    model: TinyLlama,
    lr_muon: float = 0.02,
    lr_adamw: float = 6e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    fused: bool = True
) -> Tuple[Muon, torch.optim.AdamW]:
    """
    Intelligently partitions model parameters into dual optimizers:
    
      1. "Muon Optimizer":
         - Assigned to: All internal 2D weight matrices (q_proj, k_proj, v_proj,
           o_proj, gate_proj, up_proj, down_proj).
         - Why: These layers benefit enormously from matrix orthogonalization.
         
      2. "AdamW Optimizer":
         - Group A (Weight Decay): Token embedding table & tied lm_head.
         - Group B (No Weight Decay): 1D RMSNorm gain vectors (gamma).
         - Why: 1D parameters and categorical embeddings require element-wise scaling.
    """
    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []

    # Identify token embedding parameter tensor ID
    embedding_param_ids = {id(p) for p in [model.tok_embeddings.weight]}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Rule 1: Embeddings go to AdamW with weight decay
        if id(param) in embedding_param_ids:
            adamw_decay_params.append(param)
        # Rule 2: Internal 2D linear weight matrices go to Muon
        elif param.ndim == 2:
            muon_params.append(param)
        # Rule 3: 1D parameters (RMSNorm scales) go to AdamW with NO weight decay
        else:
            adamw_no_decay_params.append(param)

    # Instantiate Muon optimizer
    optimizer_muon = Muon(
        muon_params,
        lr=lr_muon,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0
    )

    # Instantiate AdamW optimizer with separated decay groups
    adamw_groups = [
        {'params': adamw_decay_params, 'weight_decay': weight_decay},
        {'params': adamw_no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Enable fused CUDA AdamW if available for faster kernel execution
    use_fused = fused and torch.cuda.is_available()
    optimizer_adamw = torch.optim.AdamW(
        adamw_groups,
        lr=lr_adamw,
        betas=betas,
        eps=1e-8,
        fused=use_fused
    )

    return optimizer_muon, optimizer_adamw
