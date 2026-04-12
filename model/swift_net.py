"""
model.py — SWIFTNet  (CNN stem + Hybrid Transformer backbone)
=============================================================

Architecture:
    Input [B, 3, H, W]
        ↓
    ConvStem  (4× downsampling, local feature extraction)
        ↓
    Stage 0  dim=C,   H/4  × W/4
    Stage 1  dim=2C,  H/8  × W/8    ← PatchMerge
    Stage 2  dim=4C,  H/16 × W/16   ← PatchMerge
    Stage 3  dim=8C,  H/32 × W/32   ← PatchMerge
        ↓
    Global Average Pool
        ↓
    Head

ConvStem design:
    3×3 conv s2 → BN → GELU          (coarse edges, stride 2)
    3×3 DW conv + 1×1 PW → BN → GELU (local refinement, stride 1)
    3×3 conv s2 → BN                  (spatial downsampling to 1/4)

This gives the model a strong local prior before attention, avoids the
information loss of a single strided conv, and is INT8-friendly (BN fuses
into conv at deployment).

Factory (timm-registered):
    swift_net_tiny   ~6M  params
    swift_net_small  ~15M params
    swift_net_base   ~30M params
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.layers import trunc_normal_
from timm.models import register_model, build_model_with_cfg

from .swiftnet.block  import HybridBlock
from .swiftnet.config import SWIFTNetConfig


# ---------------------------------------------------------------------------
# ConvStem  (replaces WaveletPatchEmbed)
# ---------------------------------------------------------------------------

class ConvStem(nn.Module):
    """
    4× downsampling CNN stem with strong local inductive bias.

        3×3 conv s=2 → BN → GELU
        3×3 DW  s=1  → 1×1 PW → BN → GELU
        3×3 conv s=2 → BN
        Flatten → LayerNorm

    No bias in conv layers — BN handles centering.
    No BatchNorm in the final step — LN on tokens is DDP-safe without SyncBN.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        mid = embed_dim // 2

        self.stem = nn.Sequential(
            # Stage-1: strided conv (coarse features)
            nn.Conv2d(in_channels, mid, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),

            # Stage-2: DW + PW (local refinement, same resolution)
            nn.Conv2d(mid, mid, 3, stride=1, padding=1, groups=mid, bias=False),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),

            # Stage-3: strided conv (reach 1/4 resolution)
            nn.Conv2d(mid, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        """
        Returns:
            tokens: [B, H'*W', embed_dim]
            H', W': spatial grid dims
        """
        feat = self.stem(x)                              # [B, C, H/4, W/4]
        H, W = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).transpose(1, 2)         # [B, N, C]
        return self.norm(tokens), H, W


# ---------------------------------------------------------------------------
# Patch Merging (2× downsampling between stages)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """
    Strided Conv2d 2× downsampling + channel doubling.
    LN before and after for stable training.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.proj  = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        B, N, C = x.shape
        x   = self.norm1(x)
        x2d = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x2d = self.proj(x2d)
        Hn, Wn = x2d.shape[2], x2d.shape[3]
        x   = self.norm2(x2d.flatten(2).transpose(1, 2))
        return x, Hn, Wn


# ---------------------------------------------------------------------------
# Distillation-aware classifier head
# ---------------------------------------------------------------------------

class SWIFTNetHead(nn.Module):
    """
    Dual-head: (cls_logits, dist_logits) in training, averaged in eval.
    Call .fuse() before deployment to merge into a single linear.
    """

    def __init__(self, dim: int, num_classes: int, distillation: bool = False, drop: float = 0.0):
        super().__init__()
        self.distillation = distillation
        self.drop  = nn.Dropout(drop)
        self.head  = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.hdist = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x  = self.drop(x)
        c  = self.head(x)
        d  = self.hdist(x)
        if self.training and self.distillation and not torch.jit.is_scripting():
            return c, d
        return (c + d) / 2

    @torch.no_grad()
    def fuse(self) -> nn.Linear | nn.Identity:
        if not isinstance(self.head, nn.Linear):
            return nn.Identity()
        w = (self.head.weight + self.hdist.weight) / 2
        b = (self.head.bias   + self.hdist.bias)   / 2
        m = nn.Linear(w.size(1), w.size(0), device=w.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# ---------------------------------------------------------------------------
# SWIFTNet
# ---------------------------------------------------------------------------

class SWIFTNet(nn.Module):
    """
    SWIFT-Net: ConvStem + Staged HybridBlocks (DWConv + WindowAttn).

    Timm-compatible:
        - forward_features() → [B, N, C] pre-pool tokens
        - forward_head()     → logits
        - feature_info       for feature pyramid / dense tasks
        - Distillation dual-head
    """

    def __init__(self, config: Optional[SWIFTNetConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            valid  = {f.name for f in dataclasses.fields(SWIFTNetConfig)}
            config = SWIFTNetConfig(**{k: v for k, v in kwargs.items() if k in valid})

        self.config       = config
        self.num_stages   = len(config.dims)
        self.global_pool  = config.global_pool
        self.num_features = config.dims[-1]

        # feature_info: stem is 4× reduction; each merge doubles it
        stride = 4
        self.feature_info = []
        for i, dim in enumerate(config.dims):
            self.feature_info.append(dict(num_chs=dim, reduction=stride, module=f"stages.{i}"))
            stride *= 2

        # ── Stem ─────────────────────────────────────────────────────────
        self.stem = ConvStem(
            in_channels=config.in_channels,
            embed_dim=config.dims[0],
        )

        # ── Stochastic depth schedule ─────────────────────────────────────
        total  = sum(config.depths)
        dpr    = [x.item() for x in torch.linspace(0, config.drop_path_rate, total)]

        # ── Stages + mergers ──────────────────────────────────────────────
        self.stages  = nn.ModuleList()
        self.mergers = nn.ModuleList()
        self._build_stages(config, dpr)

        self.norm = nn.LayerNorm(self.num_features)

        self.head = SWIFTNetHead(
            dim=self.num_features,
            num_classes=config.num_classes,
            distillation=config.distillation,
            drop=config.drop_rate,
        )

        self.apply(self._init_weights)

    # ------------------------------------------------------------------

    def _build_stages(self, cfg: SWIFTNetConfig, dpr: list[float]) -> None:
        g = 0  # global block index
        for si in range(self.num_stages):
            dim   = cfg.dims[si]
            depth = cfg.depths[si]
            nH    = cfg.num_heads[si]

            blocks = nn.ModuleList([
                HybridBlock(
                    dim=dim,
                    num_heads=nH,
                    window_size=cfg.window_size,
                    mlp_expand=cfg.mlp_expand,
                    rope_base=cfg.rope_base,
                    block_idx=g + bi,
                    drop=cfg.drop_rate,
                    drop_path=dpr[g + bi],
                )
                for bi in range(depth)
            ])
            self.stages.append(blocks)
            g += depth

            if si < self.num_stages - 1:
                self.mergers.append(PatchMerging(dim, cfg.dims[si + 1]))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward_features(self, x: Tensor) -> Tensor:
        x, H, W = self.stem(x)
        for si, stage in enumerate(self.stages):
            for block in stage:
                x = block(x, H, W)
            if si < self.num_stages - 1:
                x, H, W = self.mergers[si](x, H, W)
        return self.norm(x)

    def forward_head(self, x: Tensor) -> Tensor:
        if self.global_pool == "avg":
            x = x.mean(dim=1)
        return self.head(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))

    # ------------------------------------------------------------------

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        """Return per-stage token tensors for dense prediction."""
        maps = []
        x, H, W = self.stem(x)
        for si, stage in enumerate(self.stages):
            for block in stage:
                x = block(x, H, W)
            maps.append(x)
            if si < self.num_stages - 1:
                x, H, W = self.mergers[si](x, H, W)
        return maps

    def count_parameters(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        out = {"total": total, "trainable": trainable,
               "stem": sum(p.numel() for p in self.stem.parameters())}
        for i, s in enumerate(self.stages):
            out[f"stage_{i}"] = sum(p.numel() for p in s.parameters())
        out["head"] = sum(p.numel() for p in self.head.parameters())
        return out

    @torch.no_grad()
    def fuse(self) -> None:
        self.head = self.head.fuse()


# ---------------------------------------------------------------------------
# timm factory
# ---------------------------------------------------------------------------

def _create(variant: str, pretrained: bool = False, **kw) -> SWIFTNet:
    return build_model_with_cfg(SWIFTNet, variant, pretrained, **kw)


@register_model
def swift_net_tiny(pretrained: bool = False, **kwargs) -> SWIFTNet:
    """~6M params — ultra-lightweight mobile."""
    args = dict(
        dims=[32, 64, 128, 256],
        depths=[2, 2, 6, 2],
        num_heads=[1, 2, 4, 8],
        drop_path_rate=0.05,
    )
    return _create("swift_net_tiny", pretrained, **{**args, **kwargs})


@register_model
def swift_net_small(pretrained: bool = False, **kwargs) -> SWIFTNet:
    """~15M params — balanced accuracy / efficiency."""
    args = dict(
        dims=[48, 96, 192, 384],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 12],
        drop_path_rate=0.1,
    )
    return _create("swift_net_small", pretrained, **{**args, **kwargs})


@register_model
def swift_net_base(pretrained: bool = False, **kwargs) -> SWIFTNet:
    """~30M params — high accuracy, server-class edge."""
    args = dict(
        dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
    )
    return _create("swift_net_base", pretrained, **{**args, **kwargs})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from timm.models import create_model

    model = create_model("swift_net_tiny", num_classes=1000)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("output:", out.shape)

    print("\nParameter counts:")
    for k, v in model.count_parameters().items():
        print(f"  {k}: {v:,}")

    # Distillation
    m2 = create_model("swift_net_tiny", num_classes=1000, distillation=True)
    m2.train()
    cls_out, dist_out = m2(x)
    print("distillation shapes:", cls_out.shape, dist_out.shape)