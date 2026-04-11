"""
ssm.py — Causal Depthwise Conv1d SSM (ONNX-compatible)
=======================================================

The original DPLR-SSM used a Vandermonde-based kernel computed dynamically from
the sequence length L (from x.shape[1]). PyTorch 2.2.x's ONNX exporter treats
that sequence dimension as dynamic, so F.conv1d with a kernel of shape
[d_model, 1, L_dynamic] fails with "kernel of unknown shape".

Fix: replace the dynamic kernel with a static nn.Parameter of fixed shape
[d_model, 1, kernel_size] where kernel_size is a Python int set at __init__.
The causal depthwise conv1d behaviour is identical — only the receptive field
changes from "full sequence" to a fixed length.  Global context is still
captured by the Wave Attention branch running in parallel.

ONNX safety:
  - F.pad(x_t, (self.kernel_size - 1, 0)): pad size is Python int → static
  - F.conv1d(x_pad, self.conv_weight, ...): kernel shape is static parameter → safe
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DPLRStateSpaceModel(nn.Module):
    """
    Causal depthwise Conv1d approximation of the DPLR-SSM.

    Kernel is a fixed-size nn.Parameter (shape [d_model, 1, kernel_size]),
    making all shapes static at ONNX trace time.

    Args:
        d_model:     input/output dimension
        d_state:     kept for API compatibility (not used in conv version)
        rank:        kept for API compatibility (not used in conv version)
        dt_min:      kept for API compatibility (not used)
        dt_max:      kept for API compatibility (not used)
        dropout:     output dropout
        kernel_size: causal conv kernel length — Python int, ONNX-safe
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        rank: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.0,
        kernel_size: int = 31,
    ):
        super().__init__()
        self.d_model     = d_model
        self.kernel_size = kernel_size  # Python int — static for ONNX

        # ── Input / Output projections ────────────────────────────────────
        self.in_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

        # ── Causal depthwise conv kernel ─────────────────────────────────
        # Static shape [d_model, 1, kernel_size] — kernel_size is a Python int
        # from __init__, so F.conv1d sees a fixed-shape weight → ONNX-safe.
        self.conv_weight = nn.Parameter(
            torch.randn(d_model, 1, kernel_size) * (kernel_size ** -0.5)
        )

        # ── Skip connection (D matrix equivalent) ────────────────────────
        self.D = nn.Parameter(torch.ones(d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # Initialise conv to a causal exponential decay (mimics SSM impulse response)
        with torch.no_grad():
            k = self.kernel_size
            decay = torch.exp(
                -torch.arange(k, dtype=torch.float32) * (2.0 / k)
            )
            self.conv_weight.data[:, 0, :] = decay / decay.sum()

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            y: [B, L, d_model]
        """
        x_in = self.in_proj(x)               # [B, L, d_model]
        x_t  = x_in.transpose(1, 2)          # [B, d_model, L]

        # Causal left-padding: size is (kernel_size - 1) — a static Python int
        x_pad = F.pad(x_t, (self.kernel_size - 1, 0))   # [B, d_model, L+k-1]
        y = F.conv1d(x_pad, self.conv_weight, groups=self.d_model)  # [B, d_model, L]

        # Skip connection (direct term)
        y = y + self.D.view(1, -1, 1) * x_t  # [B, d_model, L]

        y = y.transpose(1, 2)                 # [B, L, d_model]
        return self.drop(self.out_proj(y))


class SSMBlock(nn.Module):
    """
    Wrapper of DPLRStateSpaceModel with LayerNorm and residual connection.

    Args:
        dim:         model dimension
        d_state:     SSM state dimension (API compat)
        rank:        low-rank rank (API compat)
        dropout:     dropout rate
        kernel_size: causal conv kernel size
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        rank: int = 1,
        dropout: float = 0.0,
        kernel_size: int = 31,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm  = DPLRStateSpaceModel(
            d_model=dim,
            d_state=d_state,
            rank=rank,
            dropout=dropout,
            kernel_size=kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, D] → [B, L, D]"""
        return x + self.ssm(self.norm(x))
