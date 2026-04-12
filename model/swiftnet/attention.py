"""
attention.py — Window Self-Attention with 2D RoPE
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope_position_encoding import RopePositionEmbedding, apply_rope_2d


class WindowSelfAttention(nn.Module):
    """
    Shifted Window Self-Attention + 2D RoPE.
    Cosine attention (stable on INT8 hardware, no sqrt(d) tuning needed).
    O(N · W²) — linear in sequence length.

    Args:
        dim:         embedding dimension
        num_heads:   number of heads
        window_size: window tile size (default 7)
        rope_base:   RoPE base frequency
        tau:         cosine attention temperature
        attn_drop:   attention dropout
        proj_drop:   output dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        rope_base: float = 100.0,
        tau: float = 0.01,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.window_size = window_size
        self.tau         = tau

        self.qkv       = nn.Linear(dim, dim * 3, bias=False)
        self.proj      = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = RopePositionEmbedding(
            embed_dim=dim,
            num_heads=num_heads,
            base=rope_base,
        )

    # ── Window partition helpers ──────────────────────────────────────────

    @staticmethod
    def _partition(x: Tensor, W: int) -> Tensor:
        """[B, H, Wi, D] → [B*nW, W², D]"""
        B, H, Wi, D = x.shape
        x = x.view(B, H // W, W, Wi // W, W, D)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, W * W, D)

    @staticmethod
    def _reverse(windows: Tensor, W: int, H: int, Wi: int) -> Tensor:
        """[B*nW, W², D] → [B, H, Wi, D]"""
        B = windows.shape[0] // ((H // W) * (Wi // W))
        x = windows.view(B, H // W, Wi // W, W, W, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, Wi, -1)

    # ─────────────────────────────────────────────────────────────────────

    def forward(self, x: Tensor, H: int, W_img: int, shift: bool = False) -> Tensor:
        """
        x:     [B, N, D]  N = H * W_img
        shift: True for odd blocks
        """
        B, N, D = x.shape
        W  = self.window_size
        nH = self.num_heads

        x2d = x.view(B, H, W_img, D)

        shift_size = W // 2 if shift else 0
        if shift:
            x2d = torch.roll(x2d, shifts=(-shift_size, -shift_size), dims=(1, 2))

        pad_b = (W - H     % W) % W
        pad_r = (W - W_img % W) % W
        if pad_b > 0 or pad_r > 0:
            x2d = F.pad(x2d, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = H + pad_b, W_img + pad_r

        x_win = self._partition(x2d, W)  # [B*nW, W², D]
        Bw    = x_win.shape[0]

        q, k, v = self.qkv(x_win).chunk(3, dim=-1)
        q = q.view(Bw, W * W, nH, self.head_dim)
        k = k.view(Bw, W * W, nH, self.head_dim)
        v = v.view(Bw, W * W, nH, self.head_dim)

        sin, cos = self.rope(H=W, W=W)
        q = apply_rope_2d(q, sin, cos)
        k = apply_rope_2d(k, sin, cos)

        # Cosine attention
        q = F.normalize(q, dim=-1).permute(0, 2, 1, 3)   # [Bw, nH, W², hd]
        k = F.normalize(k, dim=-1).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.tau
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, v)                        # [Bw, nH, W², hd]
        out = out.permute(0, 2, 1, 3).reshape(Bw, W * W, D)

        out = self._reverse(out, W, H_pad, W_pad)
        out = out[:, :H, :W_img, :]

        if shift:
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(1, 2))

        return self.proj_drop(self.proj(out.reshape(B, N, D)))