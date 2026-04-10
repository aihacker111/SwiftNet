"""
block.py — SWIFTBlock: unit lặp lại cơ bản của SWIFT-Net
=========================================================

Mỗi SWIFTBlock chứa:
  ┌─────────────────────────────────────────────┐
  │             LayerNorm                        │
  │                 ↓                            │
  │    ┌────────────┼────────────┐               │
  │    ↓            ↓           ↓               │
  │  KD-Conv   WaveAttn     DPLR-SSM             │
  │  (local)   (global)     (long-range)         │
  │    └────────────┼────────────┘               │
  │                 ↓                            │
  │              GMS Fusion                      │
  │                 ↓                            │
  │          [late blocks only]                  │
  │           Window SA + RoPE                   │
  │                 ↓                            │
  │            Learnable gate                    │
  │          alpha*wave + beta*win               │
  │                 ↓                            │
  │           Residual + FFN                     │
  │               (×2/3)                         │
  └─────────────────────────────────────────────┘

Tại sao 3 branches song song?
  - KD-Conv: inductive bias cục bộ, translation invariance
  - WaveAttn: global context O(n log n)
  - DPLR-SSM: sequential/temporal structure (tốt cho dense prediction)
  Mỗi branch capture thông tin bổ sung, GMS Fusion học cách blend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention   import WaveAttentionWithRoPE, WindowSelfAttentionWithRoPE
from .conv import KDConvBlock
from .ssm         import DPLRStateSpaceModel
from .fusion      import GMSFusion


# ---------------------------------------------------------------------------
# SwiGLU FFN — tốt hơn GELU FFN với cùng số params
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020).
    out = (W_1 x) * sigmoid(β * W_1 x) ⊙ (W_2 x)
    Dùng hidden_dim = 2/3 * expand * dim để giữ số params tương đương GELU FFN.

    Args:
        dim:         input/output dimension
        expand:      expansion ratio (mặc định 4 → hidden = 2/3 * 4 * dim)
        dropout:     dropout rate
    """

    def __init__(self, dim: int, expand: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expand * 2 / 3)
        # Làm tròn lên bội số của 64 cho hardware alignment
        hidden = ((hidden + 63) // 64) * 64

        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w2 = nn.Linear(dim, hidden, bias=False)   # value
        self.w3 = nn.Linear(hidden, dim, bias=False)   # output
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# SWIFTBlock
# ---------------------------------------------------------------------------

class SWIFTBlock(nn.Module):
    """
    Core repeating block của SWIFT-Net.

    Args:
        dim:           embedding dimension
        num_heads:     số attention heads
        window_size:   kích thước window cho Window SA (late blocks)
        mlp_expand:    FFN expansion ratio
        d_state:       SSM state dimension
        ssm_rank:      SSM low-rank rank
        kd_rank:       Kronecker conv rank
        wavelet_levels: Haar pyramid levels
        num_rff:       Random Fourier Features count
        rope_base:     RoPE base frequency
        is_late:       True → thêm Window SA song song
        block_idx:     index để xác định shift pattern
        drop:          dropout rate
        drop_path:     stochastic depth rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        mlp_expand: float = 4.0,
        d_state: int = 16,
        ssm_rank: int = 1,
        kd_rank: int = 16,
        wavelet_levels: int = 2,
        num_rff: int = 64,
        rope_base: float = 100.0,
        is_late: bool = False,
        block_idx: int = 0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.is_late   = is_late
        self.shift     = (block_idx % 2 == 1)  # shift window mỗi block xen kẽ

        # ── Norms ─────────────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # ── Branch 1: KD-Conv (local features) ───────────────────────────
        self.kd_conv = KDConvBlock(
            dim=dim,
            kernel_size=3,
            rank_p=kd_rank,
        )

        # ── Branch 2: Wave Attention + RoPE (global, O(n log n)) ─────────
        self.wave_attn = WaveAttentionWithRoPE(
            dim=dim,
            num_heads=num_heads,
            num_rff=num_rff,
            wavelet_levels=wavelet_levels,
            rope_base=rope_base,
            attn_drop=drop,
            proj_drop=drop,
        )

        # ── Branch 3: DPLR-SSM (sequential long-range) ───────────────────
        self.ssm = DPLRStateSpaceModel(
            d_model=dim,
            d_state=d_state,
            rank=ssm_rank,
            dropout=drop,
        )
        self.ssm_norm = nn.LayerNorm(dim)

        # ── GMS Fusion (gộp 3 branches) ───────────────────────────────────
        self.fusion = GMSFusion(dim=dim, n_streams=3, dropout=drop)

        # ── Window SA chỉ cho late blocks ────────────────────────────────
        if is_late:
            self.win_attn = WindowSelfAttentionWithRoPE(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                rope_base=rope_base,
                attn_drop=drop,
                proj_drop=drop,
            )
            self.win_norm = nn.LayerNorm(dim)
            # Learnable gate: wave vs window SA
            # Init [0.5, 0.5] → model tự học balance
            self.attn_gate = nn.Parameter(torch.zeros(2))

        # ── FFN ───────────────────────────────────────────────────────────
        self.ffn = SwiGLUFFN(dim=dim, expand=mlp_expand, dropout=drop)

        # ── Stochastic Depth (DropPath) ───────────────────────────────────
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # ── Layer Scale (init 1e-5 → stable training) ────────────────────
        self.ls1 = nn.Parameter(1e-5 * torch.ones(dim))
        self.ls2 = nn.Parameter(1e-5 * torch.ones(dim))

    def _run_attn_branch(self, x: Tensor, H: int, W: int) -> Tensor:
        """Chạy attention (wave only hoặc wave + window SA với learnable gate)."""
        wave_out = self.wave_attn(x, H, W)

        if not self.is_late:
            return wave_out

        win_out = self.win_attn(self.win_norm(x), H, W, shift=self.shift)
        gates   = torch.softmax(self.attn_gate, dim=0)  # [2]
        return gates[0] * wave_out + gates[1] * win_out

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x:    [B, N, D]  với N = H*W
            H, W: spatial dimensions
        Returns:
            [B, N, D]
        """
        residual = x
        x_norm   = self.norm1(x)

        # ── 3 branches song song ──────────────────────────────────────────
        conv_out  = self.kd_conv(x_norm, H, W)
        attn_out  = self._run_attn_branch(x_norm, H, W)
        ssm_out   = self.ssm(self.ssm_norm(x_norm))

        # ── Fusion ────────────────────────────────────────────────────────
        fused = self.fusion([conv_out, attn_out, ssm_out])

        # ── Residual + Layer Scale ────────────────────────────────────────
        x = residual + self.drop_path(self.ls1 * fused)

        # ── FFN ───────────────────────────────────────────────────────────
        x = x + self.drop_path(self.ls2 * self.ffn(self.norm2(x)))

        return x


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """
    Stochastic Depth (Huang et al., 2016).
    Drop toàn bộ residual path với probability p trong training.
    Scale output bởi 1/(1-p) để giữ kỳ vọng không đổi.

    Args:
        drop_prob: xác suất drop
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep)
        return x * random_tensor / keep

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"