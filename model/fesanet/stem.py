"""
stem.py — WAPE Stem: Window-Attended Patch Embedding
=====================================================

Pipeline (224×224 input → H/4 × W/4 token grid):

    Input [B, 3, 224, 224]
        ↓  Conv2d(3 → mid, 3×3, stride=2, pad=1)    112×112
        ↓  DyT(mid)
        ↓  LocalWindowAttention(ws=3, dim=mid)        112×112  ← inductive bias
        ↓  Conv2d(mid → dim, 3×3, stride=2, pad=1)    56×56
        ↓  flatten + LayerNorm
    Output [B, 56*56, dim]

Why window attention in the stem?
  - A 3×3 window (9 tokens) is essentially free: 9² = 81 ops/token
  - But it gives the model a local context over micro-patches before the
    first downsampling, helping preserve fine edge cues that pure conv
    stems tend to blur
  - Purely attention-based (no reparameterization / structural re-param)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dyt import DyT


# ---------------------------------------------------------------------------
# LocalWindowAttention
# ---------------------------------------------------------------------------

class LocalWindowAttention(nn.Module):
    """
    MHSA confined to non-overlapping windows of size ws×ws.

    Args:
        dim:       channel dimension
        num_heads: attention heads
        ws:        window size (default 3)
        attn_drop: attention dropout
        proj_drop: output projection dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ws: int = 3,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.ws        = ws

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x: [B, H*W, dim]
        Returns:
            [B, H*W, dim]
        """
        B, N, C = x.shape
        ws = self.ws

        # Pad to make H, W divisible by ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = x.reshape(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad last 2 spatial dims
            H_pad, W_pad = H + pad_h, W + pad_w
            x = x.reshape(B, H_pad * W_pad, C)
        else:
            H_pad, W_pad = H, W

        # Reshape into windows: [B, nWh, nWw, ws, ws, C]
        nWh = H_pad // ws
        nWw = W_pad // ws
        x = x.reshape(B, nWh, ws, nWw, ws, C).permute(0, 1, 3, 2, 4, 5)
        # [B, nWh*nWw, ws*ws, C]
        Bw = B * nWh * nWw
        x = x.reshape(Bw, ws * ws, C)

        # MHSA
        qkv = self.qkv(x).reshape(Bw, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                          # [3, Bw, nh, ws², hd]
        q, k, v = qkv.unbind(0)                                    # each [Bw, nh, ws², hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale              # [Bw, nh, ws², ws²]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(Bw, ws * ws, C)  # [Bw, ws², C]
        out = self.proj_drop(self.proj(out))

        # Unwindow: [Bw, ws², C] → [B, H_pad, W_pad, C]
        out = out.reshape(B, nWh, nWw, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H_pad, W_pad, C)

        # Crop padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :]

        return out.reshape(B, H * W, C)


# ---------------------------------------------------------------------------
# WAPEStem
# ---------------------------------------------------------------------------

class WAPEStem(nn.Module):
    """
    Window-Attended Patch Embedding stem.

    Two-stage downsampling 224→56 (4× total):
      1. Conv2d(3 → mid, 3×3, s=2) + DyT + LocalWindowAttention
      2. Conv2d(mid → dim, 3×3, s=2)
      → flatten + LayerNorm

    Args:
        in_chans:   input image channels (default 3)
        embed_dim:  output token dimension
        img_size:   assumed input resolution (for documentation only)
        ws:         window size for LocalWindowAttention
        num_heads:  attention heads in local window attention
        drop:       dropout rate
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 64,
        img_size: int = 224,
        ws: int = 3,
        num_heads: int = 4,
        drop: float = 0.0,
    ):
        super().__init__()
        mid = embed_dim // 2

        # Stage 1: 224 → 112
        self.conv1 = nn.Conv2d(in_chans, mid, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = DyT(mid)
        self.lwa   = LocalWindowAttention(dim=mid, num_heads=num_heads, ws=ws, proj_drop=drop)
        self.lwa_norm = DyT(mid)   # pre-norm for attention

        # Stage 2: 112 → 56
        self.conv2 = nn.Conv2d(mid, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim)  # final token norm (for downstream stages)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            tokens: [B, H/4 * W/4, embed_dim]
            Hout, Wout: spatial dimensions
        """
        B, _, H, W = x.shape

        # Stage 1
        x = self.conv1(x)                     # [B, mid, H/2, W/2]
        H1, W1 = x.shape[2], x.shape[3]

        # Apply DyT norm, then reshape for attention
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H1 * W1, x.shape[1])  # [B, N, mid]
        x_flat = self.norm1(x_flat)

        # Local window attention (with residual)
        x_flat = x_flat + self.lwa(self.lwa_norm(x_flat), H1, W1)
        x = x_flat.reshape(B, H1, W1, -1).permute(0, 3, 1, 2)  # [B, mid, H/2, W/2]

        # Stage 2
        x = self.conv2(x)                     # [B, embed_dim, H/4, W/4]
        H2, W2 = x.shape[2], x.shape[3]

        tokens = x.permute(0, 2, 3, 1).reshape(B, H2 * W2, -1)  # [B, N, embed_dim]
        tokens = self.norm2(tokens)

        return tokens, H2, W2
