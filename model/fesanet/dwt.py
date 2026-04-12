"""
dwt.py — Haar 2-D Discrete Wavelet Transform (DWT) and Inverse (IDWT)
=======================================================================

Implements a single-level 2-D Haar DWT as a fixed (non-learnable) op:

    DWT(x) → (LL, LH, HL, HH)   each at H/2 × W/2
    IDWT(LL, LH, HL, HH) → x    reconstructs at H × W

Used in FDABlock to decompose feature maps into:
  - LL (low-low):   smooth, semantic content → feed to attention branch
  - LH, HL, HH:     edges and textures → feed to depthwise-conv branch

Properties:
  - Fixed weights, no parameters — zero overhead vs learned conv
  - Perfectly invertible (lossless reconstruction)
  - Halves spatial resolution, so attention cost on LL is 4× cheaper
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ── Haar filter banks ──────────────────────────────────────────────────────
#  low-pass  h = [1, 1] / sqrt(2)
#  high-pass g = [1,-1] / sqrt(2)

_SQRT2 = 2.0 ** 0.5


def _haar_filters(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return (low, high) 1-D Haar filters as [1, 1, 2] tensors."""
    lo = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / _SQRT2   # [2]
    hi = torch.tensor([1.0, -1.0], device=device, dtype=dtype) / _SQRT2  # [2]
    return lo.view(1, 1, 2), hi.view(1, 1, 2)


# ── DWT ────────────────────────────────────────────────────────────────────

def dwt2d(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Single-level 2-D Haar DWT.

    Args:
        x: [B, C, H, W]  (H and W must be even)

    Returns:
        (LL, LH, HL, HH) each [B, C, H/2, W/2]
          LL — low-low   (approximate / semantic)
          LH — low-high  (horizontal edges)
          HL — high-low  (vertical edges)
          HH — high-high (diagonals / noise)
    """
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, f"DWT requires even H, W. Got {H}×{W}."

    lo, hi = _haar_filters(x.device, x.dtype)  # [1, 1, 2]

    # Reshape to [B*C, 1, H, W] for channel-wise depthwise conv
    x_ = x.reshape(B * C, 1, H, W)

    # ── Row-wise filtering → [B*C, 1, H, W/2] ──────────────────────────
    x_lo = F.conv1d(x_.squeeze(1).reshape(B * C * H, 1, W), lo, stride=2)   # [B*C*H, 1, W/2]
    x_hi = F.conv1d(x_.squeeze(1).reshape(B * C * H, 1, W), hi, stride=2)

    x_lo = x_lo.reshape(B * C, H, W // 2)  # [B*C, H, W/2]
    x_hi = x_hi.reshape(B * C, H, W // 2)

    # ── Col-wise filtering (transpose, convolve rows, transpose back) ──
    # x_lo: [B*C, H, W/2] → treat H as sequence → [B*C*(W/2), 1, H]
    x_ll = F.conv1d(x_lo.permute(0, 2, 1).reshape(B * C * (W // 2), 1, H), lo, stride=2)
    x_lh = F.conv1d(x_lo.permute(0, 2, 1).reshape(B * C * (W // 2), 1, H), hi, stride=2)
    x_hl = F.conv1d(x_hi.permute(0, 2, 1).reshape(B * C * (W // 2), 1, H), lo, stride=2)
    x_hh = F.conv1d(x_hi.permute(0, 2, 1).reshape(B * C * (W // 2), 1, H), hi, stride=2)

    # reshape to [B*C, H/2, W/2] → [B, C, H/2, W/2]
    half_H = H // 2
    half_W = W // 2

    def _reshape(t: Tensor) -> Tensor:
        return t.reshape(B * C, half_W, half_H).permute(0, 2, 1).reshape(B, C, half_H, half_W)

    return _reshape(x_ll), _reshape(x_lh), _reshape(x_hl), _reshape(x_hh)


# ── IDWT ───────────────────────────────────────────────────────────────────

def idwt2d(LL: Tensor, LH: Tensor, HL: Tensor, HH: Tensor) -> Tensor:
    """
    Single-level 2-D Haar IDWT (reconstruction).

    Args:
        LL, LH, HL, HH: each [B, C, H/2, W/2]

    Returns:
        x: [B, C, H, W]
    """
    B, C, half_H, half_W = LL.shape
    H, W = half_H * 2, half_W * 2

    lo, hi = _haar_filters(LL.device, LL.dtype)

    def _col_idwt(lo_sub: Tensor, hi_sub: Tensor) -> Tensor:
        """Reconstruct along rows (height axis). Input: [B*C, H/2, W/2]."""
        # treat each W column separately: [B*C*(W/2), 1, H/2]
        n = B * C * half_W
        l_ = lo_sub.permute(0, 2, 1).reshape(n, 1, half_H)
        h_ = hi_sub.permute(0, 2, 1).reshape(n, 1, half_H)
        rec = F.conv_transpose1d(l_, lo, stride=2) + F.conv_transpose1d(h_, hi, stride=2)
        # [n, 1, H] → [B*C, W/2, H] → [B*C, H, W/2]
        return rec.reshape(B * C, half_W, H).permute(0, 2, 1)

    # ── Col reconstruct ────────────────────────────────────────────────
    lo_part = _col_idwt(
        LL.reshape(B * C, half_H, half_W),
        LH.reshape(B * C, half_H, half_W),
    )  # [B*C, H, W/2]
    hi_part = _col_idwt(
        HL.reshape(B * C, half_H, half_W),
        HH.reshape(B * C, half_H, half_W),
    )  # [B*C, H, W/2]

    # ── Row reconstruct ────────────────────────────────────────────────
    # treat each H row separately: [B*C*H, 1, W/2]
    n = B * C * H
    lo_r = lo_part.reshape(n, 1, half_W)
    hi_r = hi_part.reshape(n, 1, half_W)
    out = F.conv_transpose1d(lo_r, lo, stride=2) + F.conv_transpose1d(hi_r, hi, stride=2)
    # [B*C*H, 1, W] → [B, C, H, W]
    return out.reshape(B, C, H, W)
