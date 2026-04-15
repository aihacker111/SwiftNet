from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0

        self.D_head           = embed_dim // num_heads
        self.base             = base
        self.normalize_coords = normalize_coords
        self.dtype            = dtype

        self.register_buffer(
            "periods",
            torch.empty(self.D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

        self._cache_key: tuple[int, int] | None = None
        self._cache_sin: Tensor | None = None
        self._cache_cos: Tensor | None = None

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        if not self.training and self._cache_key == (H, W):
            return self._cache_sin, self._cache_cos  # type: ignore[return-value]

        device = self.periods.device
        dtype  = self.dtype
        dd     = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            s = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / s
            coords_w = torch.arange(0.5, W, **dd) / s
        elif self.normalize_coords == "min":
            s = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / s
            coords_w = torch.arange(0.5, W, **dd) / s
        else:
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)

        sin = torch.sin(angles)
        cos = torch.cos(angles)

        if not self.training:
            self._cache_key = (H, W)
            self._cache_sin = sin
            self._cache_cos = cos

        return sin, cos

    def invalidate_cache(self) -> None:
        self._cache_key = None
        self._cache_sin = None
        self._cache_cos = None

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype  = self.dtype
        periods = self.base ** (
            2 * torch.arange(self.D_head // 4, device=device, dtype=dtype)
            / (self.D_head // 2)
        )
        self.periods.data = periods


# def apply_rope_2d(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
#     """
#     x:   [B, N, nH, head_dim]
#     sin/cos: [N, head_dim]
#     """
#     half_D = x.shape[-1] // 2

#     sin_h = sin.reshape(-1, half_D, 2)[:, :, 0][None, :, None, :]
#     cos_h = cos.reshape(-1, half_D, 2)[:, :, 0][None, :, None, :]

#     x_r = x.reshape(*x.shape[:-1], half_D, 2)
#     x1  = x_r[..., 0]
#     x2  = x_r[..., 1]

#     out = torch.stack(
#         [x1 * cos_h - x2 * sin_h,
#          x1 * sin_h + x2 * cos_h],
#         dim=-1,
#     )
#     return out.flatten(-2)







def apply_rope_2d(x, sin, cos):
    B_or_Bw, N, nH, D = x.shape
    half_D = D // 2
    # tách rõ: nửa đầu = height encoding, nửa sau = width encoding
    x_h = x[..., :half_D]
    x_w = x[..., half_D:]
    
    sin_h = sin[:, :half_D]  # [N, half_D]
    cos_h = cos[:, :half_D]
    sin_w = sin[:, half_D:]
    cos_w = cos[:, half_D:]
    
    # rotate từng nửa
    def rotate(z, s, c):
        z1, z2 = z[..., ::2], z[..., 1::2]
        s = s[..., ::2][None, :, None, :]
        c = c[..., ::2][None, :, None, :]
        return torch.stack([z1*c - z2*s, z1*s + z2*c], dim=-1).flatten(-2)
    
    return torch.cat([rotate(x_h, sin_h, cos_h), rotate(x_w, sin_w, cos_w)], dim=-1)