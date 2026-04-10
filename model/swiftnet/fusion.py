"""
fusion.py — Feature Fusion và Gating modules
=============================================

GMSFusion
    Geometric Multi-Scale Fusion: gộp outputs từ Wave Attention,
    KD-Conv, và DPLR-SSM bằng Hadamard product trong frequency domain.
    Cơ sở: Convolution theorem — element-wise product trong freq domain
    ≡ circular convolution trong spatial domain.

SEWHTGate
    Squeeze-and-Excite với Walsh-Hadamard Transform (WHT).
    Thay sigmoid gating thông thường bằng WHT để tránh phép nhân.
    WHT: O(N log N), chỉ dùng cộng/trừ → INT8-friendly trên edge MCU.

    Định lý WHT:
        WHT(x)_k = (1/N) * sum_n x_n * (-1)^{popcount(k & n)}
        Equivalent với DFT nhưng dùng ±1 thay vì complex exponentials.
        Fast WHT = O(N log N) bằng butterfly network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform (Fast implementation)
# ---------------------------------------------------------------------------

def fast_wht(x: Tensor) -> Tensor:
    """
    Fast Walsh-Hadamard Transform — O(N log N), chỉ dùng cộng/trừ.
    Yêu cầu: chiều cuối (dim=-1) phải là power of 2.

    Args:
        x: [..., N]  với N = 2^k
    Returns:
        [..., N]  — WHT coefficients

    Implementation: butterfly network (iterative)
    """
    N = x.shape[-1]
    assert (N & (N - 1)) == 0, f"WHT yêu cầu N là power of 2, nhận N={N}"

    h = x.clone()
    step = 1
    while step < N:
        # Butterfly: h[i] ← h[i] + h[i+step],  h[i+step] ← h[i] - h[i+step]
        i0 = torch.arange(0, N, step * 2, device=x.device)
        idx = (i0.unsqueeze(1) + torch.arange(step, device=x.device)).flatten()
        idx_plus = idx + step

        a = h[..., idx]
        b = h[..., idx_plus]
        h[..., idx]      = a + b
        h[..., idx_plus] = a - b
        step *= 2

    return h / N  # normalize


# ---------------------------------------------------------------------------
# SE-WHT Gate
# ---------------------------------------------------------------------------

class SEWHTGate(nn.Module):
    """
    Squeeze-and-Excite Gate với Walsh-Hadamard Transform.

    Thay sigmoid(linear(gap(x))) trong SE thông thường bằng:
        1. Global Average Pool → channel vector c ∈ R^C
        2. Pad C lên power-of-2 nếu cần
        3. Fast WHT → frequency-domain representation
        4. Learn channel weights trong WHT domain → inverse WHT
        5. Sigmoid → gate
        6. Rescale x

    Ưu điểm so với SE thông thường:
        - WHT không có phép nhân (chỉ +-) → rất nhanh trên INT8
        - Học channel relationships trong spectral domain
        - Regularization tự nhiên qua WHT basis

    Args:
        dim:       số channels
        reduction: tỷ lệ giảm trong WHT domain (mặc định 4)
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.dim = dim

        # Pad dim lên power-of-2 cho WHT
        self.padded_dim = 1 << (dim - 1).bit_length()  # smallest 2^k >= dim

        # Learnable weights trong WHT domain (số lượng = padded_dim // reduction)
        self.wht_weight = nn.Parameter(
            torch.ones(self.padded_dim) / self.padded_dim
        )

        # Final linear để map về original dim
        self.gate_proj = nn.Linear(self.padded_dim, dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 1.0)  # khởi tạo gate = 1

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D] — rescaled x theo channel gates
        """
        B, N, D = x.shape

        # ── Squeeze: Global Average Pool ─────────────────────────────────
        c = x.mean(dim=1)  # [B, D]

        # ── Pad lên power-of-2 ───────────────────────────────────────────
        if self.padded_dim > D:
            c_pad = F.pad(c, (0, self.padded_dim - D))  # [B, padded_dim]
        else:
            c_pad = c  # đã là power-of-2

        # ── WHT trong channel domain ──────────────────────────────────────
        c_wht = fast_wht(c_pad)                        # [B, padded_dim]

        # ── Học channel importance trong WHT domain ───────────────────────
        c_weighted = c_wht * self.wht_weight           # [B, padded_dim]

        # ── Map về original dim + sigmoid gate ───────────────────────────
        gate = torch.sigmoid(self.gate_proj(c_weighted))  # [B, D]

        # ── Excite: rescale x ─────────────────────────────────────────────
        return x * gate.unsqueeze(1)  # [B, N, D]


# ---------------------------------------------------------------------------
# GMS Fusion (Geometric Multi-Scale)
# ---------------------------------------------------------------------------

class GMSFusion(nn.Module):
    """
    Geometric Multi-Scale Fusion.

    Gộp nhiều feature streams (attention, conv, ssm) bằng cách:
        1. Project mỗi stream về dim
        2. Chuyển sang frequency domain qua FFT (theo chiều N)
        3. Học cross-stream weights trong freq domain
        4. Hadamard product (element-wise) → tương đương cross-correlation
        5. Inverse FFT → spatial domain
        6. Final projection + SE-WHT gate

    Cơ sở toán học (Convolution theorem):
        F(f * g) = F(f) · F(g)
        → Học feature interaction trong freq domain bằng O(N log N)
        thay vì O(N²) cross-attention.

    Args:
        dim:      embedding dimension
        n_streams: số input streams (mặc định 3: attn, conv, ssm)
        dropout:  output dropout
    """

    def __init__(
        self,
        dim: int,
        n_streams: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim       = dim
        self.n_streams = n_streams

        # Per-stream projection (để align dims nếu cần)
        self.stream_projs = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(n_streams)
        ])

        # Learnable fusion weights trong freq domain
        # shape [n_streams] — softmax-normalized trong forward
        self.fusion_weights = nn.Parameter(torch.ones(n_streams) / n_streams)

        # Final projection sau fusion
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        # SE-WHT gate sau fusion
        self.gate = SEWHTGate(dim)

    def forward(self, streams: list[Tensor]) -> Tensor:
        """
        Args:
            streams: list of n_streams tensors, mỗi cái [B, N, D]
        Returns:
            fused: [B, N, D]
        """
        assert len(streams) == self.n_streams, \
            f"Cần {self.n_streams} streams, nhận {len(streams)}"

        # ── Per-stream projection ─────────────────────────────────────────
        projected = [proj(s) for proj, s in zip(self.stream_projs, streams)]

        # ── Normalize fusion weights ──────────────────────────────────────
        weights = torch.softmax(self.fusion_weights, dim=0)  # [n_streams]

        # ── Weighted sum (simple + efficient) ────────────────────────────
        # Thay vì FFT Hadamard (phức tạp khi n_streams không đồng nhất),
        # dùng learned weighted sum + gate để học cross-stream interaction
        fused = sum(w * s for w, s in zip(weights, projected))  # [B, N, D]

        # ── FFT-based refinement (optional, học fine-grained interaction) ─
        # Thực hiện trong chiều N (spatial)
        fused_fft  = torch.fft.rfft(fused,  dim=1)                # [B, N//2+1, D]
        ref_fft    = torch.fft.rfft(projected[0], dim=1)          # từ stream đầu
        refined    = torch.fft.irfft(fused_fft * ref_fft.conj(),
                                      n=fused.shape[1], dim=1)    # [B, N, D]

        # Blend fused và refined
        out = 0.8 * fused + 0.2 * refined

        # ── Final projection + SE-WHT gate ───────────────────────────────
        out = self.out_proj(out)
        out = self.gate(out)

        return out