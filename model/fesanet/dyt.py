"""
dyt.py — Dynamic Tanh (DyT) normalization
==========================================

Replaces LayerNorm with a learnable element-wise transform:
    out = weight * tanh(alpha * x) + bias

Benefits vs LayerNorm:
  - 2 ONNX ops (Mul + Tanh) vs 6 ops for LayerNorm
  - No reduce-mean / reduce-var → friendlier for fixed-shape inference
  - alpha learns the effective "scale" of the distribution per channel

Reference: "Transformers without Normalization" (Ma et al., 2025)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DyT(nn.Module):
    """
    Dynamic Tanh normalization layer — drop-in replacement for LayerNorm.

    Args:
        dim:        number of features (channels)
        init_alpha: initial value for the learnable scale alpha
    """

    def __init__(self, dim: int, init_alpha: float = 0.5):
        super().__init__()
        self.alpha  = nn.Parameter(torch.full((dim,), init_alpha))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., dim]
        return self.weight * torch.tanh(self.alpha * x) + self.bias

    def extra_repr(self) -> str:
        return f"dim={self.alpha.shape[0]}, init_alpha={self.alpha.data[0].item():.2f}"
