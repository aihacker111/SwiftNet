"""
config.py — FESANetConfig dataclass
====================================

Centralizes all hyperparameters for the FESA-Net architecture.
Factory functions in fesa_net.py create pre-filled instances.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class FESANetConfig:
    """
    Configuration for FESANet.

    Stage layout (4 stages):
        Stage 0: FDA blocks,  56×56  (local + frequency attention)
        Stage 1: FDA blocks,  28×28  (local + frequency attention)
        Stage 2: SAA blocks,  14×14  (semantic anchor attention)
        Stage 3: SDA blocks,   7×7   (full spatial decay attention)

    Args:
        img_size:       input image resolution (assumes square)
        in_chans:       input image channels
        num_classes:    number of output classes
        dims:           channel dims per stage [d0, d1, d2, d3]
        depths:         number of blocks per stage [n0, n1, n2, n3]
        num_heads:      attention heads per stage [h0, h1, h2, h3]
        num_anchors:    anchor tokens in SAA stage (Stage 2)
        mlp_expand:     FFN expansion ratio (all stages)
        stem_ws:        window size for WAPEStem local attention
        stem_heads:     attention heads in WAPEStem
        drop:           dropout rate
        drop_path:      max stochastic depth rate (linearly scaled per block)
        init_alpha:     DyT initial alpha
    """

    img_size:    int        = 224
    in_chans:    int        = 3
    num_classes: int        = 1000

    dims:        List[int]  = field(default_factory=lambda: [64, 128, 256, 512])
    depths:      List[int]  = field(default_factory=lambda: [2, 2, 8, 2])
    num_heads:   List[int]  = field(default_factory=lambda: [4, 8, 16, 16])

    num_anchors: int        = 16
    mlp_expand:  float      = 4.0
    stem_ws:     int        = 3
    stem_heads:  int        = 4

    drop:         float      = 0.0
    drop_path:    float      = 0.1
    init_alpha:   float      = 0.5
    distillation: bool       = False

    def __post_init__(self):
        assert len(self.dims) == 4, "Must have exactly 4 stages."
        assert len(self.depths) == 4
        assert len(self.num_heads) == 4
