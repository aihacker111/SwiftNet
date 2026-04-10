"""
attention.py — Attention modules cho SWIFT-Net
===============================================

WaveAttentionWithRoPE
    Linear attention O(n log n) qua Haar wavelet pyramid + Random Fourier Features.
    RoPE được áp vào Q, K sau linear projection.

WindowSelfAttentionWithRoPE
    Window-based self-attention cho late blocks.
    Dùng cosine attention (không có phép chia sqrt(d) — ổn định trên INT8 hardware).
    Shift window xen kẽ để kết nối thông tin cross-window.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope_position_encoding import RopePositionEmbedding, apply_rope_2d


# ---------------------------------------------------------------------------
# Wave Attention với RoPE
# ---------------------------------------------------------------------------

class WaveAttentionWithRoPE(nn.Module):
    """
    Wavelet Linear Attention + 2D RoPE.

    Pipeline:
        x [B,N,D]
        → linear: Q, K, V  [B,N,D]
        → reshape: [B,N,nH,head_dim]
        → apply_rope_2d(Q), apply_rope_2d(K)  ← RoPE chỉ vào Q, K
        → flatten: [B*nH, N, head_dim]
        → Haar wavelet pyramid: K → {K_0, K_1, ..., K_J}
        → RFF feature map: phi(Q), phi(K_j)  [O(n·D_rff)]
        → linear attention mỗi level: phi(Q)·(phi(K_j)^T · V_j) / (phi(Q)·sum_phi_K_j)
        → weighted sum theo level_weights (learnable softmax)
        → projection

    Độ phức tạp: O(N · D_rff · log N) so với O(N² · D) của standard attention.

    Args:
        dim:           embedding dimension
        num_heads:     số attention heads
        num_rff:       số Random Fourier Features (D_rff, thường 64-128)
        wavelet_levels: số levels Haar pyramid (thường 2-3)
        rope_base:     RoPE base frequency
        attn_drop:     dropout sau attention
        proj_drop:     dropout sau projection
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_rff: int = 64,
        wavelet_levels: int = 2,
        rope_base: float = 100.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) phải chia hết cho num_heads ({num_heads})"

        self.num_heads     = num_heads
        self.head_dim      = dim // num_heads
        self.num_rff       = num_rff
        self.wavelet_levels = wavelet_levels

        # ── Projections ──────────────────────────────────────────────────
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # ── RoPE (chia sẻ cho Q và K) ────────────────────────────────────
        self.rope = RopePositionEmbedding(
            embed_dim=dim,
            num_heads=num_heads,
            base=rope_base,
        )

        # ── Random Fourier Features (frozen — không train) ───────────────
        # Theo Bochner's theorem: omega ~ N(0, 1/head_dim) để normalize scale
        omega = torch.randn(self.head_dim, num_rff) * (self.head_dim ** -0.5)
        bias  = torch.rand(num_rff) * 2.0 * math.pi
        self.register_buffer("omega", omega)  # [head_dim, D_rff]
        self.register_buffer("bias",  bias)   # [D_rff]

        # ── Learnable level weights (softmax-normalized) ─────────────────
        self.level_weights = nn.Parameter(
            torch.ones(wavelet_levels + 1) / (wavelet_levels + 1)
        )

    # ------------------------------------------------------------------

    def _rff_feature(self, x: Tensor) -> Tensor:
        """
        Random Fourier Feature map: phi(x) = sqrt(2/D) * cos(omega^T x + b)
        Xấp xỉ RBF kernel: k(x,y) ≈ phi(x)^T phi(y)

        Args:
            x: [..., head_dim]
        Returns:
            [..., num_rff]
        """
        proj = x @ self.omega + self.bias   # [..., D_rff]
        return (2.0 / self.num_rff) ** 0.5 * torch.cos(proj)

    def _haar_pyramid(
        self, x: Tensor, H: int, W: int
    ) -> list[Tensor]:
        """
        Tạo Haar wavelet pyramid từ sequence.
        Level 0 = bản gốc, level j = LL sub-band sau j lần pool.

        Args:
            x: [B*nH, N, head_dim]  với N = H*W
        Returns:
            list of [B*nH, N_j, head_dim] với N_j = (H/2^j)*(W/2^j)
        """
        BnH, N, D = x.shape
        levels = [x]
        cur  = x.view(BnH, H, W, D).permute(0, 3, 1, 2)  # [BnH, D, H, W]
        cur_H, cur_W = H, W

        for _ in range(self.wavelet_levels):
            new_H, new_W = cur_H // 2, cur_W // 2
            if new_H < 1 or new_W < 1:
                break
            # Average pool 2×2 ≈ Haar LL filter (low-pass)
            cur   = F.avg_pool2d(cur, kernel_size=2, stride=2)  # [BnH, D, H/2, W/2]
            cur_H, cur_W = new_H, new_W
            levels.append(cur.permute(0, 2, 3, 1).reshape(BnH, cur_H * cur_W, D))

        return levels  # list[Tensor]

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x:    [B, N, D]  với N = H*W
            H, W: spatial dims của patch grid
        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape
        nH = self.num_heads

        # ── RoPE sin/cos ─────────────────────────────────────────────────
        sin, cos = self.rope(H=H, W=W)  # [N, head_dim]

        # ── QKV projection ────────────────────────────────────────────────
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # mỗi cái [B, N, D]
        q = q.view(B, N, nH, self.head_dim)
        k = k.view(B, N, nH, self.head_dim)
        v = v.view(B, N, nH, self.head_dim)

        # ── Áp dụng RoPE vào Q và K (KHÔNG áp vào V) ────────────────────
        q = apply_rope_2d(q, sin, cos)   # [B, N, nH, head_dim]
        k = apply_rope_2d(k, sin, cos)

        # ── Reshape cho multi-head processing ────────────────────────────
        # [B, N, nH, D] → [B*nH, N, D]
        q = q.permute(0, 2, 1, 3).reshape(B * nH, N, self.head_dim)
        k = k.permute(0, 2, 1, 3).reshape(B * nH, N, self.head_dim)
        v = v.permute(0, 2, 1, 3).reshape(B * nH, N, self.head_dim)

        # ── Wavelet pyramid cho K và V ────────────────────────────────────
        k_levels = self._haar_pyramid(k, H, W)
        v_levels = self._haar_pyramid(v, H, W)

        level_w = torch.softmax(self.level_weights[: len(k_levels)], dim=0)

        # ── Linear attention qua các levels ──────────────────────────────
        # Kernel trick: Attn(Q,K,V) = phi(Q) · [phi(K)^T V] / phi(Q) · [phi(K)^T 1]
        q_feat = self._rff_feature(q)  # [B*nH, N, D_rff]
        out = torch.zeros_like(q)

        for j, (k_j, v_j) in enumerate(zip(k_levels, v_levels)):
            k_feat = self._rff_feature(k_j)                    # [B*nH, N_j, D_rff]
            kv     = torch.bmm(k_feat.transpose(1, 2), v_j)    # [B*nH, D_rff, head_dim]
            num    = torch.bmm(q_feat, kv)                      # [B*nH, N, head_dim]
            denom  = (q_feat * k_feat.sum(1, keepdim=True)).sum(-1, keepdim=True)
            denom  = denom.clamp(min=1e-6)
            out    = out + level_w[j] * (num / denom)

        out = self.attn_drop(out)

        # ── Reshape và project ────────────────────────────────────────────
        out = out.view(B, nH, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# Window Self-Attention với RoPE + Cosine Attention
# ---------------------------------------------------------------------------

class WindowSelfAttentionWithRoPE(nn.Module):
    """
    Window-based Self-Attention cho late blocks.

    Tại sao cosine attention thay vì standard softmax?
    - Cosine attn = softmax(q/||q|| · k/||k|| / tau)
    - Không phụ thuộc magnitude → ổn định hơn khi quantize INT8
    - Không cần tune 1/sqrt(d) scaling
    - Tham khảo: DeiT-III (Touvron et al., 2022)

    Shift window (Swin-style):
    - Block chẵn: regular window partition
    - Block lẻ: shifted partition (dịch W//2, W//2)
    → Thông tin cross-window được trao đổi qua các blocks

    Chi phí: O(N · W² · D) / W² = O(N · D), linear với sequence length.

    Args:
        dim:         embedding dimension
        num_heads:   số attention heads
        window_size: kích thước cửa sổ (mặc định 7)
        rope_base:   RoPE base frequency
        tau:         temperature cho cosine attention (mặc định 0.01)
        attn_drop:   dropout
        proj_drop:   dropout
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

        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RoPE riêng cho window (size cố định window_size × window_size)
        self.rope = RopePositionEmbedding(
            embed_dim=dim,
            num_heads=num_heads,
            base=rope_base,
        )

    # ------------------------------------------------------------------
    # Window partition utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _partition(x: Tensor, W: int) -> Tensor:
        """[B, H, W_img, D] → [B*num_win, W*W, D]"""
        B, H, Wi, D = x.shape
        x = x.view(B, H // W, W, Wi // W, W, D)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, W * W, D)

    @staticmethod
    def _reverse(windows: Tensor, W: int, H: int, Wi: int) -> Tensor:
        """[B*num_win, W*W, D] → [B, H, W_img, D]"""
        B = windows.shape[0] // (H // W * Wi // W)
        x = windows.view(B, H // W, Wi // W, W, W, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, Wi, -1)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, H: int, W_img: int, shift: bool = False) -> Tensor:
        """
        Args:
            x:      [B, N, D]  với N = H * W_img
            H:      chiều cao patch grid
            W_img:  chiều rộng patch grid
            shift:  True cho odd blocks (shifted window)
        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape
        W  = self.window_size
        nH = self.num_heads

        x_2d = x.view(B, H, W_img, D)

        # ── Shift ────────────────────────────────────────────────────────
        shift_size = W // 2 if shift else 0
        if shift:
            x_2d = torch.roll(x_2d, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # ── Padding để H, W_img chia hết cho window_size ─────────────────
        pad_b = (W - H     % W) % W
        pad_r = (W - W_img % W) % W
        if pad_b > 0 or pad_r > 0:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x_2d.shape[1], x_2d.shape[2]

        # ── Partition → windows ───────────────────────────────────────────
        x_win = self._partition(x_2d, W)  # [B*nW, W², D]
        Bw    = x_win.shape[0]

        # ── QKV ──────────────────────────────────────────────────────────
        q, k, v = self.qkv(x_win).chunk(3, dim=-1)
        q = q.view(Bw, W * W, nH, self.head_dim)
        k = k.view(Bw, W * W, nH, self.head_dim)
        v = v.view(Bw, W * W, nH, self.head_dim)

        # ── RoPE bên trong window (sin/cos cho W×W grid, được cache) ─────
        sin, cos = self.rope(H=W, W=W)
        q = apply_rope_2d(q, sin, cos)
        k = apply_rope_2d(k, sin, cos)

        # ── Cosine Attention ──────────────────────────────────────────────
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # [Bw, nH, W², head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.tau  # [Bw, nH, W², W²]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                              # [Bw, nH, W², head_dim]
        out = out.permute(0, 2, 1, 3).reshape(Bw, W * W, D)

        # ── Merge windows → 2D ───────────────────────────────────────────
        out = self._reverse(out, W, H_pad, W_pad)
        if pad_b > 0 or pad_r > 0:
            out = out[:, :H, :W_img, :]

        # ── Unshift ───────────────────────────────────────────────────────
        if shift:
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(1, 2))

        out = out.reshape(B, N, D)
        return self.proj_drop(self.proj(out))