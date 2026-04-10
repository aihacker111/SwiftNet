"""
convolution.py — Kronecker-Factored Depthwise Convolution (KD-Conv)
====================================================================

Thay thế standard depthwise-separable conv bằng Kronecker decomposition:
    W ≈ A ⊗ B,  A ∈ R^{p×q},  B ∈ R^{r×s}

Tiết kiệm tham số: C_in * C_out * k² → (p*q + r*s) * k²

Lý thuyết (Kronecker-Van Loan):
    vec(W) = (A ⊗ B) vec(X)
    Hoặc tương đương: W x = A (B x) theo thứ tự reshaping

Ví dụ C=256, k=3:
    Standard conv:  256 * 256 * 9 = 589,824 params
    KD-Conv (r=16): (16*16 + 16*16) * 9 = 4,608 params  → giảm 128×

Cấu trúc KD-Conv block:
    input
      ├── KroneckerConv2d (depthwise theo Kronecker)
      ├── BatchNorm
      └── GELU activation
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KroneckerConv2d(nn.Module):
    """
    Convolution với weight tensor xấp xỉ bằng Kronecker product A ⊗ B.

    Với in_channels = p*r, out_channels = q*s:
        W_full [out, in, k, k] ≈ kron(A [p,q], B [r,s]) reshape về [out, in, k, k]

    Để tránh materialization W_full (tốn bộ nhớ), ta thực hiện:
        1. Reshape input từ [B, C_in, H, W] → [B, p, r, H, W]
        2. Linear B theo dim r: → [B, p, s, H, W]
        3. Linear A theo dim p: → [B, q, s, H, W]
        4. Reshape → [B, C_out, H, W]
        5. Depthwise spatial conv k×k

    Điều này tương đương W = A ⊗ B nhưng không cần ma trận lớn.

    Args:
        in_channels:  C_in (phải = p * r)
        out_channels: C_out (phải = q * s)
        kernel_size:  kích thước kernel spatial
        rank_p:       rank của factor A (p và q = out_channels // rank_p)
        stride:       stride cho spatial conv
        padding:      padding cho spatial conv
        bias:         có dùng bias không
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        rank_p: int = 16,
        stride: int = 1,
        padding: int | str = "same",
        bias: bool = False,
    ):
        super().__init__()
        assert in_channels  % rank_p == 0, \
            f"in_channels ({in_channels}) phải chia hết cho rank_p ({rank_p})"
        assert out_channels % rank_p == 0, \
            f"out_channels ({out_channels}) phải chia hết cho rank_p ({rank_p})"

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.rank_p       = rank_p
        self.p  = rank_p
        self.r  = in_channels  // rank_p   # B factor input dim
        self.q  = rank_p
        self.s  = out_channels // rank_p   # B factor output dim

        # Factor A: [p, q] — channel mixing (coarse)
        self.A = nn.Parameter(torch.empty(self.p, self.q))

        # Factor B: [r, s] — channel mixing (fine)
        self.B = nn.Parameter(torch.empty(self.r, self.s))

        # Spatial depthwise conv k×k cho C_out groups
        self.spatial_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=out_channels,   # depthwise
            bias=bias,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Khởi tạo theo Kaiming uniform cho tích Kronecker
        # Var[A⊗B] = Var[A] * Var[B] → mỗi factor dùng gain / sqrt(p)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spatial_conv.weight, mode="fan_out")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            [B, C_out, H, W]
        """
        B, C, H, W = x.shape

        # Reshape: [B, C_in, H, W] → [B, p, r, H, W]
        x = x.view(B, self.p, self.r, H, W)

        # Áp dụng factor B theo chiều r: [B, p, r, H, W] → [B, p, s, H, W]
        # einsum "bprhw, rs -> bpshw"
        x = torch.einsum("bprhw, rs -> bpshw", x, self.B)

        # Áp dụng factor A theo chiều p: [B, p, s, H, W] → [B, q, s, H, W]
        # einsum "bpshw, pq -> bqshw"
        x = torch.einsum("bpshw, pq -> bqshw", x, self.A)

        # Reshape về [B, C_out, H, W]
        x = x.reshape(B, self.out_channels, H, W)

        # Spatial depthwise conv
        return self.spatial_conv(x)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"rank_p={self.rank_p}, A={self.p}×{self.q}, B={self.r}×{self.s}"
        )


class KDConvBlock(nn.Module):
    """
    KD-Conv block đầy đủ: KroneckerConv2d + BN + GELU.

    Thường được dùng bên trong SWIFTBlock để xử lý local features
    song song với Wave Attention (global features).

    Args:
        dim:        số channels
        kernel_size: kích thước spatial kernel (mặc định 3)
        rank_p:     Kronecker rank (mặc định 16)
        expand_ratio: tỷ lệ expand channels trong block (mặc định 1.0)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        rank_p: int = 16,
        expand_ratio: float = 1.0,
    ):
        super().__init__()
        mid_dim = int(dim * expand_ratio)

        self.block = nn.Sequential(
            # Pointwise expand (nếu expand_ratio > 1)
            nn.Conv2d(dim, mid_dim, kernel_size=1, bias=False) if mid_dim != dim
            else nn.Identity(),

            # Kronecker depthwise conv
            KroneckerConv2d(
                mid_dim, mid_dim,
                kernel_size=kernel_size,
                rank_p=rank_p,
                padding="same",
            ),

            nn.BatchNorm2d(mid_dim),
            nn.GELU(),

            # Pointwise squeeze (nếu expand_ratio > 1)
            nn.Conv2d(mid_dim, dim, kernel_size=1, bias=False) if mid_dim != dim
            else nn.Identity(),
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x:    [B, N, D]  với N = H*W (sequence format)
            H, W: spatial dims
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        # Sequence → 2D feature map
        x_2d = x.permute(0, 2, 1).reshape(B, D, H, W)
        out  = self.block(x_2d)
        # 2D → Sequence
        return out.flatten(2).transpose(1, 2)  # [B, N, D]