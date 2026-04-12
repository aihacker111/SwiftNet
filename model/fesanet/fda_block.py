"""
fda_block.py — Frequency-Decomposed Attention Block (FDA-Block)
================================================================

Pipeline per block:
    x [B, N, C]  (N = H*W)
      ↓ pre-norm (DyT)
      ↓ [B, C, H, W]
      ↓ Haar DWT → LL [B,C,H/2,W/2]  +  detail [B,3C,H/2,W/2]
      │
      ├─ LL  →  AxialStripAttention(H/2, W/2)   → attn_out [B,C,H/2,W/2]
      │
      └─ detail → 1×1 mix (3C→C) + 3×3 DW-conv → conv_out [B,C,H/2,W/2]
      │
      ↓ sum: attn_out + conv_out → [B,C,H/2,W/2]
      ↓ Haar IDWT → [B, C, H, W]
      ↓ [B, N, C] + ls1 * residual
      ↓ DyT + SwiGLU FFN + ls2 * residual

AxialStripAttention (O(n√n)):
  - Shared QKV projection
  - Row-wise attention:  reshape [B*H, W, C] → MHSA over W tokens
  - Col-wise attention:  reshape [B*W, H, C] → MHSA over H tokens
  - Sum → linear out

Why frequency decomposition?
  - Semantic / low-freq content (LL) naturally benefits from long-range
    attention (smooth regions, object classes)
  - High-freq details (edges, textures) benefit from local convolution
  - DWT makes the split mathematically clean and zero-parameter
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dyt import DyT
from .dwt import dwt2d, idwt2d


# ---------------------------------------------------------------------------
# SwiGLU FFN (shared across all block types)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, expand: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expand * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# AxialStripAttention
# ---------------------------------------------------------------------------

class AxialStripAttention(nn.Module):
    """
    Factorized row + column self-attention.
    Complexity: O(H*W*(H+W)) = O(n^{3/2}) for square feature maps.

    Args:
        dim:       channel dimension
        num_heads: attention heads
        attn_drop: attention dropout
        proj_drop: output projection dropout
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

    def _attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Standard scaled dot-product. q/k/v: [*, nh, L, hd]"""
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = self.attn_drop(a.softmax(dim=-1))
        return a @ v  # [*, nh, L, hd]

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x: [B, C, H, W]  (spatial tensor, not flattened)
        Returns:
            [B, C, H, W]
        """
        B, C, _, _ = x.shape
        nh, hd = self.num_heads, self.head_dim

        # Flatten spatial → [B, H*W, C] for QKV projection
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        qkv = self.qkv(x_flat).reshape(B, H, W, 3, nh, hd)

        # Split Q, K, V — still in [B, H, W, nh, hd] form
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]

        # ── Row-wise attention: for each row, attend over W tokens ───────
        # [B, H, W, nh, hd] → [B*H, nh, W, hd]
        q_r = q.reshape(B * H, W, nh, hd).permute(0, 2, 1, 3)
        k_r = k.reshape(B * H, W, nh, hd).permute(0, 2, 1, 3)
        v_r = v.reshape(B * H, W, nh, hd).permute(0, 2, 1, 3)
        row_out = self._attn(q_r, k_r, v_r)                           # [B*H, nh, W, hd]
        row_out = row_out.permute(0, 2, 1, 3).reshape(B, H, W, C)    # [B, H, W, C]

        # ── Col-wise attention: for each col, attend over H tokens ───────
        # [B, H, W, nh, hd] → [B*W, nh, H, hd]
        q_c = q.permute(0, 2, 1, 3, 4).reshape(B * W, H, nh, hd).permute(0, 2, 1, 3)
        k_c = k.permute(0, 2, 1, 3, 4).reshape(B * W, H, nh, hd).permute(0, 2, 1, 3)
        v_c = v.permute(0, 2, 1, 3, 4).reshape(B * W, H, nh, hd).permute(0, 2, 1, 3)
        col_out = self._attn(q_c, k_c, v_c)                           # [B*W, nh, H, hd]
        col_out = col_out.permute(0, 2, 1, 3).reshape(B, W, H, C).permute(0, 2, 1, 3)
        # [B, H, W, C]

        # ── Fuse row + col ────────────────────────────────────────────────
        out = (row_out + col_out).reshape(B, H * W, C)
        out = self.proj_drop(self.proj(out))
        return out.reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# FDABlock
# ---------------------------------------------------------------------------

class FDABlock(nn.Module):
    """
    Frequency-Decomposed Attention Block.

    Args:
        dim:        channel dimension
        num_heads:  attention heads for AxialStripAttention
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

        # ── Pre-norms ─────────────────────────────────────────────────────
        self.norm1 = DyT(dim)
        self.norm2 = DyT(dim)

        # ── Attention on LL subband ───────────────────────────────────────
        self.axial_attn = AxialStripAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=drop,
            proj_drop=drop,
        )

        # ── Detail branch (LH + HL + HH → mixed → DW conv) ───────────────
        # 3 detail subbands → mix to C channels, then spatial DW conv
        self.detail_mix = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.detail_dw  = nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                                    groups=dim, bias=False)

        # ── FFN ───────────────────────────────────────────────────────────
        self.ffn = SwiGLUFFN(dim=dim, expand=mlp_expand, dropout=drop)

        # ── Stochastic depth ─────────────────────────────────────────────
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # ── Layer scale (init 1e-2, ConvNeXt-style for from-scratch) ─────
        self.ls1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.ls2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x: [B, H*W, dim]
        Returns:
            [B, H*W, dim]
        """
        B, N, C = x.shape
        residual = x

        # Pre-norm → spatial tensor
        x_n  = self.norm1(x)
        x_2d = x_n.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # ── DWT decomposition ─────────────────────────────────────────────
        LL, LH, HL, HH = dwt2d(x_2d)                          # each [B, C, H/2, W/2]
        h2, w2 = H // 2, W // 2

        # ── Attention on LL ───────────────────────────────────────────────
        attn_out = self.axial_attn(LL, h2, w2)                # [B, C, H/2, W/2]

        # ── Detail conv branch ────────────────────────────────────────────
        detail = torch.cat([LH, HL, HH], dim=1)               # [B, 3C, H/2, W/2]
        conv_out = self.detail_dw(self.detail_mix(detail))    # [B, C, H/2, W/2]

        # ── Fuse and IDWT ────────────────────────────────────────────────
        fused_LL = attn_out + conv_out
        x_2d_out = idwt2d(fused_LL, LH, HL, HH)              # [B, C, H, W]

        out_flat = x_2d_out.permute(0, 2, 3, 1).reshape(B, N, C)  # [B, N, C]

        # ── Residual + layer scale ────────────────────────────────────────
        x = residual + self.drop_path(self.ls1 * out_flat)

        # ── FFN ───────────────────────────────────────────────────────────
        x = x + self.drop_path(self.ls2 * self.ffn(self.norm2(x)))

        return x


# ---------------------------------------------------------------------------
# DropPath (local copy to avoid cross-module import cycles)
# ---------------------------------------------------------------------------

class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep)
        return x * mask / keep
