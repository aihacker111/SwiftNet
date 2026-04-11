from dataclasses import dataclass, field
@dataclass
class SWIFTNetConfig:
    """Toàn bộ hyperparameters của SWIFTNet."""

    # ── Patch embedding ───────────────────────────────────────────────────
    in_channels:    int   = 3
    patch_size:     int   = 4      # stride của patch embedding

    # ── Stage dims & depths ────────────────────────────────────────────────
    dims:    list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    depths:  list[int] = field(default_factory=lambda: [2, 2, 6, 2])

    # ── Attention ─────────────────────────────────────────────────────────
    num_heads:     list[int] = field(default_factory=lambda: [2, 4, 8, 16])
    window_size:   int  = 7
    num_rff:       int  = 64
    wavelet_levels: int = 2
    rope_base:     float = 100.0

    # ── Conv ──────────────────────────────────────────────────────────────
    kd_rank:  int = 16

    # ── SSM ───────────────────────────────────────────────────────────────
    d_state:       int = 16
    ssm_rank:      int = 1
    ssm_kernel_size: int = 31   # causal conv kernel length (static → ONNX-safe)

    # ── FFN ───────────────────────────────────────────────────────────────
    mlp_expand: float = 4.0

    # ── Regularization ────────────────────────────────────────────────────
    drop_rate:      float = 0.0
    drop_path_rate: float = 0.1

    # ── Late blocks: tỷ lệ blocks cuối dùng Window SA ────────────────────
    late_ratio: float = 0.4   # 40% blocks cuối mỗi stage

    # ── Head ──────────────────────────────────────────────────────────────
    num_classes:  int  = 1000
    distillation: bool = False
    global_pool:  str  = "avg"
