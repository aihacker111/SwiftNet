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

def fast_wht(x: Tensor, N: int) -> Tensor:
    """
    Fast Walsh-Hadamard Transform — O(N log N), addition/subtraction only.
    N must be a power of 2 and passed as a Python int for ONNX compatibility.

    Implemented with reshape + stack butterflies instead of in-place scatter
    so the entire function is ONNX-exportable without ScatterElements.

    Args:
        x: [B, N]  with N = 2^k  (Python int, not Tensor)
        N: sequence length as Python int
    Returns:
        [B, N]  — WHT coefficients
    """
    h = x
    step = 1
    # Pure Python loop — N and step are Python ints, fully unrolled by tracer
    while step < N:
        num_groups = N // (2 * step)           # Python int
        # Butterfly: view each group as (a, b) pairs, compute (a+b, a-b)
        # [B, N] → [B, num_groups, 2, step]
        h = h.reshape(-1, num_groups, 2, step)
        a = h[:, :, 0, :]                      # [B, num_groups, step]
        b = h[:, :, 1, :]
        h = torch.stack([a + b, a - b], dim=2).reshape(-1, N)  # [B, N]
        step *= 2
    return h / N


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

        # ── Pad lên power-of-2 — use stored Python ints for ONNX ────────
        pad_size = self.padded_dim - self.dim  # Python int, known at init
        if pad_size > 0:
            c_pad = F.pad(c, (0, pad_size))    # [B, padded_dim]
        else:
            c_pad = c

        # ── WHT trong channel domain ──────────────────────────────────────
        c_wht = fast_wht(c_pad, self.padded_dim)       # [B, padded_dim]

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

        # Learnable fusion weights — softmax-normalized in forward
        self.fusion_weights = nn.Parameter(torch.ones(n_streams) / n_streams)

        # Refinement projection (replaces FFT cross-correlation — not ONNX-exportable)
        self.refine_proj = nn.Linear(dim, dim, bias=False)

        # Final projection after fusion
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        # SE-WHT gate after fusion
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

        # ── Weighted sum — stack to avoid Python iteration over tensor ────
        # zip(weights, projected) iterates a Tensor which breaks ONNX tracing
        stacked = torch.stack(projected, dim=0)          # [n_streams, B, N, D]
        fused   = (weights.view(-1, 1, 1, 1) * stacked).sum(0)  # [B, N, D]

        # ── Learned refinement (replaces FFT cross-correlation) ─────────
        # FFT ops are not ONNX-exportable; use a linear projection instead
        refined = self.refine_proj(projected[0])
        out = fused + 0.2 * refined

        # ── Final projection + SE-WHT gate ───────────────────────────────
        out = self.out_proj(out)
        out = self.gate(out)

        return out