"""
block.py — HybridBlock: Depthwise Conv (local) + Window Self-Attention (global)
================================================================================

Each HybridBlock:
    ┌─────────────────────────────────────┐
    │          LayerNorm                   │
    │              ↓                       │
    │    ┌─────────┴──────────┐            │
    │    ↓                    ↓            │
    │  DWConv              WindowAttn      │
    │  (local)             (global+RoPE)   │
    │    └─────────┬──────────┘            │
    │           add + ls1                  │
    │           Residual                   │
    │              ↓                       │
    │           SwiGLU FFN                 │
    │           Residual + ls2             │
    └─────────────────────────────────────┘

- DWConv captures local texture/edge inductive bias at O(N·k²)
- WindowAttn captures global context at O(N·W²), linear in N
- Shifted window alternation connects cross-window information
- Layer scale (1e-2) + DropPath for stable from-scratch training
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import WindowSelfAttention


# ---------------------------------------------------------------------------
# Local branch: Depthwise-Separable Conv
# ---------------------------------------------------------------------------

class DWConvBranch(nn.Module):
    """
    Depthwise-Separable conv for local feature extraction.
    BN (not LN) for fast inference on edge hardware.

    x [B,N,D] → 2D → DW 3×3 → PW 1×1 → BN → GELU → sequence
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw  = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2,
                             groups=dim, bias=False)
        self.pw  = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn  = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, D = x.shape
        out = x.permute(0, 2, 1).reshape(B, D, H, W)
        out = self.act(self.bn(self.pw(self.dw(out))))
        return out.flatten(2).transpose(1, 2)   # [B, N, D]


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN. hidden = round_up(2/3 * expand * dim, 32).
    Matches GELU FFN param count at same expand ratio.
    """

    def __init__(self, dim: int, expand: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expand * 2 / 3)
        hidden = ((hidden + 31) // 32) * 32

        self.w1   = nn.Linear(dim, hidden, bias=False)
        self.w2   = nn.Linear(dim, hidden, bias=False)
        self.w3   = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


# ---------------------------------------------------------------------------
# HybridBlock
# ---------------------------------------------------------------------------

class HybridBlock(nn.Module):
    """
    Core repeating block: DWConv (local) + WindowAttn (global) in parallel.

    Args:
        dim:         embedding dimension
        num_heads:   attention heads
        window_size: window tile size
        mlp_expand:  FFN expansion ratio
        rope_base:   RoPE base frequency
        block_idx:   global index — determines shift pattern
        drop:        dropout rate
        drop_path:   stochastic depth rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        mlp_expand: float = 4.0,
        rope_base: float = 100.0,
        block_idx: int = 0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.shift = (block_idx % 2 == 1)

        self.norm1 = nn.LayerNorm(dim)
        self.conv  = DWConvBranch(dim)
        self.attn  = WindowSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            rope_base=rope_base,
            attn_drop=drop,
            proj_drop=drop,
        )

        self.norm2     = nn.LayerNorm(dim)
        self.ffn       = SwiGLUFFN(dim=dim, expand=mlp_expand, dropout=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.ls2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """x: [B, N, D]  N = H*W"""
        x_norm = self.norm1(x)
        mixed  = self.conv(x_norm, H, W) + self.attn(x_norm, H, W, shift=self.shift)
        x      = x + self.drop_path(self.ls1 * mixed)
        x      = x + self.drop_path(self.ls2 * self.ffn(self.norm2(x)))
        return x