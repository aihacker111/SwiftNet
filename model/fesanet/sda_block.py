"""
sda_block.py — 4D Spatial Decay Attention Block (SDA-Block)
=============================================================

Designed for the final stage (Stage 3, 7×7 = 49 tokens) where full
attention is feasible and strong spatial reasoning is needed.

Adds a learnable spatial decay bias to standard MHSA:
    attn_bias[head, i, j] = -softplus(log_decay[head]) × dist(i, j)
where dist is the Manhattan (L1) distance between spatial positions i, j.

This encodes a soft "locality" prior: nearby tokens get higher attention
by default, but the model can override it by learning large Q·K scores.
Unlike absolute position embeddings, the decay generalizes to arbitrary
resolutions (distance is relative).

Pipeline per block:
    x [B, N, C]   N = H*W (at this stage typically 49)
      ↓ pre-norm (DyT)
      ↓ QKV projection
      ↓ attn = QK*scale + spatial_decay_bias  ← new
      ↓ softmax → V → proj
      ↓ residual + ls1
      ↓ DyT + SwiGLU + ls2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dyt import DyT
from .fda_block import SwiGLUFFN, _DropPath


# ---------------------------------------------------------------------------
# SpatialDecayAttention
# ---------------------------------------------------------------------------

class SpatialDecayAttention(nn.Module):
    """
    Standard MHSA augmented with per-head learnable spatial decay bias.

    The bias matrix B[h, i, j] = -softplus(λ[h]) × manhattan(i, j)
    is computed once per (H, W) shape and cached for the forward pass.

    Args:
        dim:        channel dimension
        num_heads:  attention heads
        attn_drop:  attention dropout
        proj_drop:  output projection dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable log-decay per head; init to log(1) = 0
        # After softplus: decay starts at log(2) ≈ 0.69, then model refines
        self.log_decay = nn.Parameter(torch.zeros(num_heads))

    @staticmethod
    def _manhattan_dist(H: int, W: int, device: torch.device) -> Tensor:
        """
        Compute Manhattan distance matrix for all pairs of (H×W) positions.
        Returns: [H*W, H*W]
        """
        # Grid coordinates
        rows = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(-1)  # [N]
        cols = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(-1)  # [N]
        dist = (rows.unsqueeze(1) - rows.unsqueeze(0)).abs() + \
               (cols.unsqueeze(1) - cols.unsqueeze(0)).abs()  # [N, N]
        return dist.float()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x: [B, N, C]  N = H*W
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        nh, hd  = self.num_heads, self.head_dim

        # QKV
        qkv = self.qkv(x).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                     # each [B, nh, N, hd]

        # Attention logits
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, N, N]

        # ── Spatial decay bias ─────────────────────────────────────────────
        dist  = self._manhattan_dist(H, W, x.device)   # [N, N]
        decay = F.softplus(self.log_decay)              # [nh]  — always positive
        bias  = -decay.view(nh, 1, 1) * dist.unsqueeze(0)   # [nh, N, N]
        attn  = attn + bias.unsqueeze(0)               # [B, nh, N, N]

        attn = self.attn_drop(attn.softmax(dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# SDABlock
# ---------------------------------------------------------------------------

class SDABlock(nn.Module):
    """
    4D Spatial Decay Attention Block.

    Args:
        dim:        channel dimension
        num_heads:  attention heads
        mlp_expand: FFN expansion ratio
        drop:       dropout rate
        drop_path:  stochastic depth rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_expand: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm1 = DyT(dim)
        self.norm2 = DyT(dim)

        self.sda = SpatialDecayAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=drop,
            proj_drop=drop,
        )

        self.ffn = SwiGLUFFN(dim=dim, expand=mlp_expand, dropout=drop)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.ls2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = x + self.drop_path(self.ls1 * self.sda(self.norm1(x), H, W))
        x = x + self.drop_path(self.ls2 * self.ffn(self.norm2(x)))
        return x
