from dataclasses import dataclass, field


@dataclass
class SWIFTNetConfig:
    # ── Stem ──────────────────────────────────────────────────────────────
    in_channels: int = 3

    # ── Stage dims & depths ────────────────────────────────────────────────
    dims:   list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    depths: list[int] = field(default_factory=lambda: [2, 2, 6, 2])

    # ── Attention ─────────────────────────────────────────────────────────
    num_heads:   list[int] = field(default_factory=lambda: [2, 4, 8, 16])
    window_size: int   = 7
    rope_base:   float = 100.0

    # ── FFN ───────────────────────────────────────────────────────────────
    mlp_expand: float = 4.0

    # ── Regularization ────────────────────────────────────────────────────
    drop_rate:      float = 0.0
    drop_path_rate: float = 0.1

    # ── Head ──────────────────────────────────────────────────────────────
    num_classes:  int  = 1000
    distillation: bool = False
    global_pool:  str  = "avg"