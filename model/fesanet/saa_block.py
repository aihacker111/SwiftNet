"""
saa_block.py — Semantic Anchor Attention Block (SAA-Block)
===========================================================

Designed for mid-level stages (Stage 2, 14×14) where semantic regions
emerge. Uses a small set of dynamic anchor tokens to compress global
context into O(n·m) attention (m << n).

Pipeline per block:
    x [B, N, C]
      ↓ pre-norm (DyT)
      │
      ├─ saliency = sigmoid(Linear(x)) → [B, N, 1]   (importance score)
      │
      ├─ anchor aggregation:
      │    Q = learnable anchor embeddings [B, m, C]
      │    K = x * saliency, V = x
      │    → anchors [B, m, C]    (O(n·m) cost)
      │
      ├─ anchor self-attention (intra-anchor):
      │    [B, m, C] → MHSA → [B, m, C]  (O(m²), m=16 so trivial)
      │
      └─ broadcast back:
           Q = x, K/V = anchors → [B, N, C]   (O(n·m) cost)
           + DW-conv(x) → local texture bias
      │
      ↓ residual + ls1
      ↓ DyT + SwiGLU + ls2

Why anchors?
  - Global average pooling loses spatial structure; CLS token has no
    spatial bias; anchors are dynamic (saliency-weighted) and can
    represent multiple semantic regions simultaneously.
  - Total attention cost: O(n·m + m²) ≈ O(n·m) for m=16, n=196 → 3136
    vs O(n²) = 38416 for full attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dyt import DyT
from .fda_block import SwiGLUFFN, _DropPath


# ---------------------------------------------------------------------------
# SemanticAnchorAttention
# ---------------------------------------------------------------------------

class SemanticAnchorAttention(nn.Module):
    """
    Saliency-driven anchor attention.

    Args:
        dim:        channel dimension
        num_heads:  attention heads
        num_anchors: number of anchor tokens m (default 16)
        attn_drop:  attention dropout
        proj_drop:  output projection dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_anchors: int = 16,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.num_anchors = num_anchors

        # Saliency scorer: maps each token to a scalar importance weight
        self.saliency = nn.Linear(dim, 1, bias=True)

        # Learnable anchor seed embeddings (expanded per batch via broadcast)
        self.anchor_seeds = nn.Parameter(torch.zeros(1, num_anchors, dim))
        nn.init.trunc_normal_(self.anchor_seeds, std=0.02)

        # Anchor aggregation (Q=anchors, K/V=tokens)
        self.q_agg = nn.Linear(dim, dim, bias=False)
        self.kv_agg = nn.Linear(dim, 2 * dim, bias=False)

        # Anchor self-attention
        self.q_self = nn.Linear(dim, dim, bias=False)
        self.kv_self = nn.Linear(dim, 2 * dim, bias=False)
        self.anchor_attn_drop = nn.Dropout(attn_drop)

        # Broadcast back (Q=tokens, K/V=anchors)
        self.q_bcast = nn.Linear(dim, dim, bias=False)
        self.kv_bcast = nn.Linear(dim, 2 * dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

        # DW conv for local bias
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def _mhsa(
        self,
        q_proj: nn.Linear,
        kv_proj: nn.Linear,
        q_in: Tensor,
        kv_in: Tensor,
        drop: nn.Dropout,
    ) -> Tensor:
        """Generic multi-head cross-attention. q_in: [B, Lq, C], kv_in: [B, Lk, C]"""
        B, Lq, C = q_in.shape
        nh, hd   = self.num_heads, self.head_dim

        q = q_proj(q_in).reshape(B, Lq, nh, hd).permute(0, 2, 1, 3)   # [B, nh, Lq, hd]

        kv = kv_proj(kv_in)
        Lk = kv_in.shape[1]
        k, v = kv.reshape(B, Lk, 2, nh, hd).permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = drop(attn.softmax(dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, Lq, C)
        return out

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x: [B, N, C]  N = H*W
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape

        # ── Saliency weights ──────────────────────────────────────────────
        sal = torch.sigmoid(self.saliency(x))      # [B, N, 1]
        x_sal = x * sal                            # emphasize salient tokens

        # ── Anchor aggregation: Q=anchor_seeds, K/V=x_sal ────────────────
        anchors = self.anchor_seeds.expand(B, -1, -1)  # [B, m, C]
        anchors = anchors + self._mhsa(
            self.q_agg, self.kv_agg, anchors, x_sal, self.anchor_attn_drop
        )  # [B, m, C]

        # ── Anchor self-attention ─────────────────────────────────────────
        anchors = anchors + self._mhsa(
            self.q_self, self.kv_self, anchors, anchors, self.anchor_attn_drop
        )  # [B, m, C]

        # ── Broadcast: Q=tokens, K/V=anchors ─────────────────────────────
        global_out = self._mhsa(
            self.q_bcast, self.kv_bcast, x, anchors, self.attn_drop
        )  # [B, N, C]

        # ── DW conv for local texture ─────────────────────────────────────
        x_2d  = x.reshape(B, H, W, C).permute(0, 3, 1, 2)   # [B, C, H, W]
        local = self.dw_conv(x_2d).permute(0, 2, 3, 1).reshape(B, N, C)

        out = self.proj_drop(self.proj(global_out + local))
        return out


# ---------------------------------------------------------------------------
# SAABlock
# ---------------------------------------------------------------------------

class SAABlock(nn.Module):
    """
    Semantic Anchor Attention Block.

    Args:
        dim:         channel dimension
        num_heads:   attention heads
        num_anchors: number of anchor tokens
        mlp_expand:  FFN expansion ratio
        drop:        dropout rate
        drop_path:   stochastic depth rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_anchors: int = 16,
        mlp_expand: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm1 = DyT(dim)
        self.norm2 = DyT(dim)

        self.saa = SemanticAnchorAttention(
            dim=dim,
            num_heads=num_heads,
            num_anchors=num_anchors,
            attn_drop=drop,
            proj_drop=drop,
        )

        self.ffn = SwiGLUFFN(dim=dim, expand=mlp_expand, dropout=drop)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.ls2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = x + self.drop_path(self.ls1 * self.saa(self.norm1(x), H, W))
        x = x + self.drop_path(self.ls2 * self.ffn(self.norm2(x)))
        return x
