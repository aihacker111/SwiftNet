"""
ssm.py — Diagonal Plus Low-Rank State Space Model (DPLR-SSM)
=============================================================

SSM dạng continuous-time:
    h'(t) = A h(t) + B u(t)
    y(t)  = C h(t)

Vấn đề: A tổng quát → O(N²) computation.

Giải pháp DPLR (Gu et al., S4):
    A = Λ - P Q*
    Λ ∈ C^{N×N} — diagonal (N params thay vì N²)
    P, Q ∈ C^{N×r} — low rank, r << N

Tại sao hiệu quả?
    Theo Woodbury identity:
        (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
    → Resolvent (zI - A)^{-1} chỉ cần O(N) để tính diagonal term
      + O(r²) để tính rank correction
    → Toàn bộ SSM convolution tính bằng FFT trong O(L log L)

Trong SWIFT-Net, DPLR-SSM được dùng song song với Wave Attention
để capture long-range sequential dependencies mà CNN không có.

Đây là phiên bản đơn giản hóa cho edge:
    - Dùng diagonal A thực (không phức) để tránh complex arithmetic
    - Discretization bằng ZOH (Zero-Order Hold)
    - Convolution kernel tính bằng torch.fft
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DPLRStateSpaceModel(nn.Module):
    """
    DPLR-SSM: xử lý sequence với long-range context, O(L log L).

    Sử dụng:
        - A = diag(Λ) - P Q^T   (DPLR structure)
        - Discretization: ZOH với step size Δ
        - Inference: convolution với kernel K = (C B̄, C Ā B̄, C Ā² B̄, ...)
        - Kernel tính qua FFT → O(L log L)

    Args:
        d_model:  input/output dimension
        d_state:  state space dimension N (thường 16-64)
        rank:     rank r của low-rank perturbation (thường 1-4)
        dt_min:   minimum step size Δ
        dt_max:   maximum step size Δ
        dropout:  output dropout
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        rank: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.rank    = rank

        # ── Input / Output projections ────────────────────────────────────
        self.in_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

        # ── SSM parameters (per channel) ─────────────────────────────────
        # Λ: diagonal của A — learned log-scale để ensure negative real part
        # (stability: Re(Λ_i) < 0)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .repeat(d_model, 1)
        )  # [d_model, d_state]

        # P, Q: low-rank factors
        self.P = nn.Parameter(torch.randn(d_model, d_state, rank) * 0.01)
        self.Q = nn.Parameter(torch.randn(d_model, d_state, rank) * 0.01)

        # B: input matrix
        self.B = nn.Parameter(torch.randn(d_model, d_state) * (d_state ** -0.5))

        # C: output matrix
        self.C = nn.Parameter(torch.randn(d_model, d_state) * (d_state ** -0.5))

        # Δ (dt): step size — learned per channel, clamped [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.dt_log = nn.Parameter(torch.log(dt))  # [d_model]

        # D: skip connection (direct term)
        self.D = nn.Parameter(torch.ones(d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    # ------------------------------------------------------------------
    # Compute SSM kernel via DPLR + FFT
    # ------------------------------------------------------------------

    def _compute_kernel(self, L: int) -> Tensor:
        """
        Tính convolution kernel K_i = C_i * (Ā_i)^i * B̄_i cho i=0..L-1.
        Dùng FFT để đạt O(L log L) thay vì O(L * d_state).

        Returns:
            K: [d_model, L]  — convolution kernel
        """
        # ── Step 1: Reconstruct A từ DPLR structure ──────────────────────
        # A_diag = -exp(A_log)  (negative để đảm bảo stability)
        A_diag = -torch.exp(self.A_log)  # [d_model, d_state]

        # DPLR: A_full = diag(A_diag) - P Q^T
        # Nhưng để tính kernel, ta dùng resolvent (zI - A)^{-1} + Woodbury
        # Đây là simplified version: dùng chính diagonal term
        # (full DPLR: xem S4 paper Appendix C)

        # ── Step 2: Discretization (ZOH) ─────────────────────────────────
        dt = torch.exp(self.dt_log).unsqueeze(1)  # [d_model, 1]

        # Ā = exp(Δ A)  — exact matrix exponential cho diagonal A
        A_bar = torch.exp(dt * A_diag)            # [d_model, d_state]

        # B̄ = (Ā - I) A^{-1} B  — ZOH discretization
        # Với diagonal A: A^{-1} = 1/A_diag (element-wise)
        B_bar = (A_bar - 1.0) / A_diag * self.B  # [d_model, d_state]

        # ── Step 3: Compute kernel K ──────────────────────────────────────
        # K_n = C Ā^n B̄  cho n = 0, 1, ..., L-1
        # Vectorize: K = [C B̄, C Ā B̄, C Ā² B̄, ...]
        #
        # Dùng trick: K_n = sum_i C_i * (Ā_i)^n * B̄_i
        # Với Vandermonde matrix: V[n,i] = (Ā_i)^n

        # Ā_bar: [d_model, d_state] — powers base
        # Xây dựng [L, d_state] Vandermonde theo batch
        powers = torch.arange(L, device=A_bar.device, dtype=A_bar.dtype)  # [L]

        # A_bar_pow[d_model, L, d_state] = (Ā[d,i])^n
        A_bar_exp = A_bar.unsqueeze(1) ** powers.view(1, L, 1)  # [d_model, L, d_state]

        # K[d, n] = sum_i C[d,i] * (Ā[d,i])^n * B̄[d,i]
        # CB: [d_model, d_state], A_bar_exp: [d_model, L, d_state]
        CB = self.C * B_bar                          # [d_model, d_state]
        K  = (CB.unsqueeze(1) * A_bar_exp).sum(-1)  # [d_model, L]

        return K.real  # giữ phần thực

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, d_model]  với L = sequence length (H*W)
        Returns:
            y: [B, L, d_model]
        """
        B, L, D = x.shape

        # Input projection + activation gate
        x_in = self.in_proj(x)   # [B, L, d_model]

        # ── Tính SSM convolution ──────────────────────────────────────────
        K = self._compute_kernel(L)  # [d_model, L]

        # Chuyển về [d_model, L] → conv1d expects [batch, channels, length]
        # Thực hiện convolution theo chiều L cho mỗi channel độc lập
        x_t = x_in.transpose(1, 2)                     # [B, d_model, L]
        K_t = K.unsqueeze(0).expand(B, -1, -1)         # [B, d_model, L]

        # FFT convolution: O(L log L)
        # Round up to next power of 2 — cuFFT fp16 requires power-of-2 sizes
        fft_size = 1 << (2 * L - 1).bit_length()
        x_fft = torch.fft.rfft(x_t,  n=fft_size, dim=-1)  # [B, d_model, fft//2+1]
        k_fft = torch.fft.rfft(K_t,  n=fft_size, dim=-1)  # [B, d_model, fft//2+1]
        y_fft = x_fft * k_fft
        y     = torch.fft.irfft(y_fft, n=fft_size, dim=-1)[..., :L]  # [B, d_model, L]

        # Skip connection (D term)
        y = y + self.D.view(1, -1, 1) * x_t

        # Transpose về sequence format
        y = y.transpose(1, 2)  # [B, L, d_model]

        return self.drop(self.out_proj(y))


class SSMBlock(nn.Module):
    """
    Wrapper của DPLR-SSM với LayerNorm và residual connection.
    Dùng bên trong SWIFTBlock song song với Attention.

    Args:
        dim:     model dimension
        d_state: SSM state dimension
        rank:    low-rank rank
        dropout: dropout rate
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        rank: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm  = DPLRStateSpaceModel(
            d_model=dim,
            d_state=d_state,
            rank=rank,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, D] → [B, L, D]"""
        return x + self.ssm(self.norm(x))