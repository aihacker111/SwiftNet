"""
fesa_net.py — FESA-Net: Frequency-Enhanced Semantic Anchor Network
===================================================================

A pure-transformer backbone (no structural reparameterization) combining
four complementary attention mechanisms across four stages:

    [224×224 input]
         ↓  WAPEStem (Conv + LocalWindowAttn + Conv)  →  56×56 tokens
         ↓  Stage 0: FDA × d0  (frequency-decomposed axial attention, 56²)
         ↓  PatchMerge (2× downsample)                →  28×28
         ↓  Stage 1: FDA × d1  (28²)
         ↓  PatchMerge                                →  14×14
         ↓  Stage 2: SAA × d2  (semantic anchor attention, 14²)
         ↓  PatchMerge                                →   7×7
         ↓  Stage 3: SDA × d3  (spatial decay full-attention, 7²)
         ↓  DyT + GAP → Linear(num_classes)

Factory functions (timm-compatible):
    fesa_net_tiny  — ~10M params
    fesa_net_small — ~22M params
    fesa_net_base  — ~40M params

Each factory registers the model with timm if timm is available.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fesanet.config    import FESANetConfig
from fesanet.dyt       import DyT
from fesanet.stem      import WAPEStem
from fesanet.fda_block import FDABlock
from fesanet.saa_block import SAABlock
from fesanet.sda_block import SDABlock


# ---------------------------------------------------------------------------
# PatchMerge — 2× spatial downsampling with channel expansion
# ---------------------------------------------------------------------------

class PatchMerge(nn.Module):
    """
    2× spatial downsampling via pixel-unshuffle + linear projection.

    Concatenates 2×2 neighbouring tokens → projects to out_dim.
    Equivalent to the Swin Transformer PatchMerging layer but expressed
    as a single linear, which keeps the ONNX graph cleaner.

    Args:
        in_dim:  input channel dimension
        out_dim: output channel dimension (usually 2× in_dim)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.proj = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        """
        Args:
            x: [B, H*W, C]
        Returns:
            merged: [B, H/2 * W/2, out_dim]
            H//2, W//2
        """
        B, N, C = x.shape
        assert H % 2 == 0 and W % 2 == 0

        x = x.reshape(B, H, W, C)
        # Gather 2×2 patches
        x0 = x[:, 0::2, 0::2, :]   # top-left
        x1 = x[:, 1::2, 0::2, :]   # bottom-left
        x2 = x[:, 0::2, 1::2, :]   # top-right
        x3 = x[:, 1::2, 1::2, :]   # bottom-right
        merged = torch.cat([x0, x1, x2, x3], dim=-1)          # [B, H/2, W/2, 4C]
        merged = merged.reshape(B, (H // 2) * (W // 2), 4 * C)
        merged = self.proj(self.norm(merged))                  # [B, H/2*W/2, out_dim]
        return merged, H // 2, W // 2


# ---------------------------------------------------------------------------
# FESANet backbone
# ---------------------------------------------------------------------------

class FESANet(nn.Module):
    """
    FESA-Net: Frequency-Enhanced Semantic Anchor Network.

    Args:
        cfg: FESANetConfig instance
    """

    def __init__(self, cfg: FESANetConfig):
        super().__init__()
        self.cfg = cfg
        dims     = cfg.dims
        depths   = cfg.depths
        heads    = cfg.num_heads

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = WAPEStem(
            in_chans=cfg.in_chans,
            embed_dim=dims[0],
            img_size=cfg.img_size,
            ws=cfg.stem_ws,
            num_heads=cfg.stem_heads,
            drop=cfg.drop,
        )

        # ── Build stages ──────────────────────────────────────────────────
        # Linearly scale drop_path across all blocks
        total_blocks = sum(depths)
        block_idx    = 0
        dp_rates: List[float] = []
        for d in depths:
            for _ in range(d):
                dp_rates.append(cfg.drop_path * block_idx / max(total_blocks - 1, 1))
                block_idx += 1

        dp_idx = 0

        # Stage 0: FDA blocks at 56×56
        self.stage0 = nn.ModuleList([
            FDABlock(
                dim=dims[0],
                num_heads=heads[0],
                mlp_expand=cfg.mlp_expand,
                drop=cfg.drop,
                drop_path=dp_rates[dp_idx + i],
            )
            for i in range(depths[0])
        ])
        dp_idx += depths[0]
        self.merge0 = PatchMerge(dims[0], dims[1])

        # Stage 1: FDA blocks at 28×28
        self.stage1 = nn.ModuleList([
            FDABlock(
                dim=dims[1],
                num_heads=heads[1],
                mlp_expand=cfg.mlp_expand,
                drop=cfg.drop,
                drop_path=dp_rates[dp_idx + i],
            )
            for i in range(depths[1])
        ])
        dp_idx += depths[1]
        self.merge1 = PatchMerge(dims[1], dims[2])

        # Stage 2: SAA blocks at 14×14
        self.stage2 = nn.ModuleList([
            SAABlock(
                dim=dims[2],
                num_heads=heads[2],
                num_anchors=cfg.num_anchors,
                mlp_expand=cfg.mlp_expand,
                drop=cfg.drop,
                drop_path=dp_rates[dp_idx + i],
            )
            for i in range(depths[2])
        ])
        dp_idx += depths[2]
        self.merge2 = PatchMerge(dims[2], dims[3])

        # Stage 3: SDA blocks at 7×7
        self.stage3 = nn.ModuleList([
            SDABlock(
                dim=dims[3],
                num_heads=heads[3],
                mlp_expand=cfg.mlp_expand,
                drop=cfg.drop,
                drop_path=dp_rates[dp_idx + i],
            )
            for i in range(depths[3])
        ])

        # ── Head(s) ───────────────────────────────────────────────────────
        self.norm_out     = DyT(dims[3], init_alpha=cfg.init_alpha)
        self.head         = nn.Linear(dims[3], cfg.num_classes) if cfg.num_classes > 0 else nn.Identity()
        # Distillation head: separate linear for teacher signal (DeiT-style)
        self.head_dist    = nn.Linear(dims[3], cfg.num_classes) if cfg.distillation else None
        self.distillation = cfg.distillation

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def num_features(self) -> int:
        return self.cfg.dims[-1]

    def count_parameters(self) -> dict:
        """Return parameter counts split by component."""
        groups = {
            "stem":   self.stem,
            "stage0": self.stage0,
            "stage1": self.stage1,
            "stage2": self.stage2,
            "stage3": self.stage3,
            "head":   self.head,
        }
        counts = {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in groups.items()
        }
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def forward_features(self, x: Tensor) -> Tensor:
        """Extract features without the classification head."""
        # Stem
        x, H, W = self.stem(x)       # [B, 56*56, d0]

        # Stage 0 (FDA, 56²)
        for blk in self.stage0:
            x = blk(x, H, W)
        x, H, W = self.merge0(x, H, W)  # → 28²

        # Stage 1 (FDA, 28²)
        for blk in self.stage1:
            x = blk(x, H, W)
        x, H, W = self.merge1(x, H, W)  # → 14²

        # Stage 2 (SAA, 14²)
        for blk in self.stage2:
            x = blk(x, H, W)
        x, H, W = self.merge2(x, H, W)  # → 7²

        # Stage 3 (SDA, 7²)
        for blk in self.stage3:
            x = blk(x, H, W)

        # Global average pooling
        x = self.norm_out(x)          # [B, 49, d3]
        x = x.mean(dim=1)             # [B, d3]
        return x

    def forward(self, x: Tensor):
        feat = self.forward_features(x)
        cls_logits = self.head(feat)

        if self.distillation:
            dist_logits = self.head_dist(feat)
            if self.training:
                # Return tuple during training so the loss can use both signals
                return cls_logits, dist_logits
            else:
                # Average the two heads at inference (DeiT-style)
                return (cls_logits + dist_logits) / 2.0

        return cls_logits


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _build(cfg: FESANetConfig, **kwargs) -> FESANet:
    from dataclasses import fields, asdict
    cfg_dict = asdict(cfg)
    cfg_dict.update(kwargs)
    valid = {f.name for f in fields(FESANetConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid}
    return FESANet(FESANetConfig(**filtered))


def fesa_net_tiny(pretrained: bool = False, **kwargs) -> FESANet:
    """
    FESA-Net Tiny  (~10 M params)
    dims=[32,64,128,256], depths=[2,2,4,2]
    """
    cfg = FESANetConfig(
        dims=[32, 64, 128, 256],
        depths=[2, 2, 4, 2],
        num_heads=[4, 4, 8, 8],
        drop_path=0.05,
    )
    return _build(cfg, **kwargs)


def fesa_net_small(pretrained: bool = False, **kwargs) -> FESANet:
    """
    FESA-Net Small  (~22 M params)
    dims=[48,96,192,384], depths=[2,2,6,2]
    """
    cfg = FESANetConfig(
        dims=[48, 96, 192, 384],
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 16],
        drop_path=0.1,
    )
    return _build(cfg, **kwargs)


def fesa_net_base(pretrained: bool = False, **kwargs) -> FESANet:
    """
    FESA-Net Base  (~40 M params)
    dims=[64,128,256,512], depths=[2,2,8,2]
    """
    cfg = FESANetConfig(
        dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 8, 16, 16],
        drop_path=0.2,
    )
    return _build(cfg, **kwargs)


import timm
from timm.models import register_model

@register_model
def fesa_net_tiny_224(pretrained: bool = False, **kwargs):
    return fesa_net_tiny(pretrained=pretrained, **kwargs)

@register_model
def fesa_net_small_224(pretrained: bool = False, **kwargs):
    return fesa_net_small(pretrained=pretrained, **kwargs)

@register_model
def fesa_net_base_224(pretrained: bool = False, **kwargs):
    return fesa_net_base(pretrained=pretrained, **kwargs)
        
if __name__ == "__main__":
    from timm.models import create_model

    model = create_model("fesa_net_small_224", num_classes=1000)
    model.eval()
    print(model)

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("output shape:", out.shape)

    # Distillation training demo
    model_dist = create_model("fesa_net_small_224", num_classes=1000, distillation=True)
    model_dist.train()
    cls_out, dist_out = model_dist(x)
    print("distillation cls shape:", cls_out.shape, " dist shape:", dist_out.shape)

    print("\nParameter counts:")
    for k, v in model.count_parameters().items():
        print(f"  {k}: {v:,}")