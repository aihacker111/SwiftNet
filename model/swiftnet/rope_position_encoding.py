"""
rope.py — 2D Rotary Position Embedding
=======================================
Dựa trên DINOv3 RopePositionEmbedding, được điều chỉnh cho SWIFT-Net:
  - Hỗ trợ cache sin/cos để tránh tính lại mỗi forward pass
  - apply_rope_2d() xử lý tensor shape [B, N, nH, head_dim]
  - Tương thích với cả Wave Attention và Window Self-Attention

Lý thuyết (Su et al., 2021 + DINOv3):
  Với mỗi cặp chiều (2i, 2i+1):
    q'_{2i}   = q_{2i}   * cos(θ_i * pos) - q_{2i+1} * sin(θ_i * pos)
    q'_{2i+1} = q_{2i+1} * cos(θ_i * pos) + q_{2i}   * sin(θ_i * pos)

  Tương đương phép nhân số phức: (q_{2i} + j·q_{2i+1}) * e^{j·θ_i·pos}
  → bảo toàn norm ||q'|| = ||q||, chỉ rotate phase
  → attention score <q',k'> phụ thuộc vào relative position (pos_q - pos_k)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# 2D RoPE — trực tiếp từ DINOv3 với thêm cache
# ---------------------------------------------------------------------------

class RopePositionEmbedding(nn.Module):
    """
    2D RoPE không có learnable weights.
    Hỗ trợ hai cách khởi tạo:
      1) ``base`` (mặc định 100.0) — giống BERT/LLaMA
      2) ``min_period`` + ``max_period`` — explicit frequency range

    Args:
        embed_dim:        tổng embedding dim (phải chia hết cho 4 * num_heads)
        num_heads:        số attention heads
        base:             base để tính periods (dùng khi không có min/max_period)
        min_period:       period nhỏ nhất
        max_period:       period lớn nhất
        normalize_coords: cách normalize tọa độ
            "separate" — H và W normalize riêng (tốt cho non-square input)
            "max"      — normalize bởi max(H, W)
            "min"      — normalize bởi min(H, W)
        shift_coords:     data augmentation: dịch tọa độ ngẫu nhiên (training)
        jitter_coords:    data augmentation: jitter scale tọa độ (training)
        rescale_coords:   data augmentation: rescale toàn bộ (training)
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0, (
            f"embed_dim ({embed_dim}) phải chia hết cho 4*num_heads ({4*num_heads})"
        )

        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Cần truyền một trong: `base` hoặc `min_period`+`max_period`.")

        D_head = embed_dim // num_heads
        self.base             = base
        self.min_period       = min_period
        self.max_period       = max_period
        self.D_head           = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords     = shift_coords
        self.jitter_coords    = jitter_coords
        self.rescale_coords   = rescale_coords
        self.dtype            = dtype

        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

        # Cache: tránh tính lại khi H, W không đổi
        self._cache_key: tuple[int, int] | None = None
        self._cache_sin: Tensor | None = None
        self._cache_cos: Tensor | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        """
        Trả về (sin, cos) để apply_rope_2d() sử dụng.

        Returns:
            sin: [H*W, D_head]
            cos: [H*W, D_head]
        """
        # Cache hit — không tính lại trong inference
        if not self.training and self._cache_key == (H, W):
            return self._cache_sin, self._cache_cos  # type: ignore[return-value]

        device = self.periods.device
        dtype  = self.dtype
        dd     = {"device": device, "dtype": dtype}

        # ── Tọa độ trong [-1, +1] ────────────────────────────────────────
        if self.normalize_coords == "max":
            max_HW  = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "min":
            min_HW  = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW
        else:  # "separate"
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)       # [HW, 2]
        coords = 2.0 * coords - 1.0        # shift [0,1] → [-1,+1]

        # ── Data augmentation (chỉ training) ─────────────────────────────
        if self.training:
            if self.shift_coords is not None:
                shift_hw = torch.empty(2, **dd).uniform_(
                    -self.shift_coords, self.shift_coords
                )
                coords = coords + shift_hw[None, :]

            if self.jitter_coords is not None:
                jmax = math.log(self.jitter_coords)
                jitter_hw = torch.empty(2, **dd).uniform_(-jmax, jmax).exp()
                coords = coords * jitter_hw[None, :]

            if self.rescale_coords is not None:
                rmax = math.log(self.rescale_coords)
                rescale_hw = torch.empty(1, **dd).uniform_(-rmax, rmax).exp()
                coords = coords * rescale_hw

        # ── Tính angles và sin/cos ────────────────────────────────────────
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        # coords: [HW, 2], periods: [D//4] → angles: [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)        # [HW, D]  — duplicate để fill head_dim

        sin = torch.sin(angles)  # [HW, D_head]
        cos = torch.cos(angles)  # [HW, D_head]

        # Cache (chỉ lưu khi inference)
        if not self.training:
            self._cache_key = (H, W)
            self._cache_sin = sin
            self._cache_cos = cos

        return sin, cos

    def invalidate_cache(self) -> None:
        """Xóa cache — gọi khi chuyển sang input size mới."""
        self._cache_key = None
        self._cache_sin = None
        self._cache_cos = None

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype  = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype)
                / (self.D_head // 2)
            )
        else:
            base     = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )
            periods  = base ** exponents / base * self.max_period
        self.periods.data = periods


# ---------------------------------------------------------------------------
# apply_rope_2d — hàm utility dùng trong Attention modules
# ---------------------------------------------------------------------------

def apply_rope_2d(
    x: Tensor,
    sin: Tensor,
    cos: Tensor,
) -> Tensor:
    """
    Áp dụng 2D RoPE rotation vào query hoặc key tensor.

    Args:
        x:   [B, N, num_heads, head_dim]  — N = H*W
        sin: [N, head_dim]
        cos: [N, head_dim]

    Returns:
        x_rotated: [B, N, num_heads, head_dim]  — cùng shape với x

    Toán học:
        Chia head_dim thành cặp (x1, x2) ← (even, odd indices):
            x1' = x1 * cos - x2 * sin
            x2' = x1 * sin + x2 * cos
        Tương đương nhân số phức: (x1 + j·x2) × (cos + j·sin)
    """
    # Avoid step-2 ONNX Slice ops (not constant-foldable for opset >= 10).
    # Replace [::2] / [1::2] with reshape-into-pairs + index — all step-1.
    half_D = x.shape[-1] // 2  # head_dim // 2 — static Python int (set at model init)

    # sin/cos: [N, D] → [N, D//2, 2] → take pair-index 0 → [1, N, 1, D//2]
    sin_h = sin.reshape(-1, half_D, 2)[:, :, 0][None, :, None, :]
    cos_h = cos.reshape(-1, half_D, 2)[:, :, 0][None, :, None, :]

    # x: [B, N, nH, D] → [B, N, nH, D//2, 2]
    x_r = x.reshape(*x.shape[:-1], half_D, 2)
    x1 = x_r[..., 0]   # [B, N, nH, D//2] — even indices, was x[..., ::2]
    x2 = x_r[..., 1]   # [B, N, nH, D//2] — odd  indices, was x[..., 1::2]

    # Rotate và interleave
    out = torch.stack(
        [x1 * cos_h - x2 * sin_h,
         x1 * sin_h + x2 * cos_h],
        dim=-1,
    )  # [B, N, nH, D//2, 2]

    return out.flatten(-2)  # [B, N, nH, D]