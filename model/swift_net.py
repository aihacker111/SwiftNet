"""
model.py — SWIFTNet Backbone
============================

Kiến trúc tổng thể:
    Input image [B, 3, H, W]
        ↓
    WaveletPatchEmbed  — DWT + Conv patch embedding
        ↓
    N × SWIFTBlock     — xen kẽ early/late blocks
        ↓
    Global Average Pool
        ↓
    Head (classifier / feature extractor)

Stage-based design (giống EfficientNet / Swin):
    Stage 0: 56×56, dim=C
    Stage 1: 28×28, dim=2C
    Stage 2: 14×14, dim=4C
    Stage 3:  7×7,  dim=8C

Patch embedding dùng DWT để nắm bắt multi-frequency từ đầu.

Factory functions (timm-registered):
    swift_net_tiny  — ~3M params (mobile-first)
    swift_net_small — ~8M params (balanced)
    swift_net_base  — ~20M params (high accuracy)

Timm integration:
    - @register_model + build_model_with_cfg for pretrained weight loading
    - feature_info for feature pyramid / dense prediction
    - Distillation dual-head (train: (cls, dist), eval: averaged)
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

from .swiftnet.block import SWIFTBlock
from .swiftnet.config import SWIFTNetConfig


# ---------------------------------------------------------------------------
# Wavelet Patch Embedding
# ---------------------------------------------------------------------------

class WaveletPatchEmbed(nn.Module):
    """
    Patch embedding với Haar DWT preprocessing.

    Pipeline:
        input [B, 3, H, W]
        → Haar DWT → LL, LH, HL, HH sub-bands  [B, 3*4, H/2, W/2]
        → Conv2d patch embedding [B, dim, H/patch_size, W/patch_size]
        → Flatten → LayerNorm

    Lý do dùng DWT thay vì plain Conv:
        - Sub-bands capture explicit frequency information
        - LL: smooth/low-freq (giống downsampling)
        - LH, HL: edges (horizontal/vertical)
        - HH: diagonal, texture
        - Giúp model "thấy" multi-scale ngay từ đầu

    Args:
        in_channels: số channels input (thường 3 cho RGB)
        embed_dim:   output embedding dimension
        patch_size:  spatial stride
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size

        # Haar DWT filters (fixed, không train)
        # Lo = [1,  1] / sqrt(2)
        # Hi = [1, -1] / sqrt(2)
        s = 0.5 ** 0.5
        ll = torch.tensor([[s, s], [s,  s]])   # LL
        lh = torch.tensor([[s, s], [-s, -s]])  # LH (horizontal edge)
        hl = torch.tensor([[s, -s], [s, -s]])  # HL (vertical edge)
        hh = torch.tensor([[s, -s], [-s, s]])  # HH (diagonal)

        # Stack thành conv weight: [4, 1, 2, 2]
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        # Expand cho in_channels: [4*in_channels, in_channels, 2, 2]
        # Dùng depthwise: mỗi channel riêng
        dwt_weight = filters.repeat(in_channels, 1, 1, 1)  # [4*C, 1, 2, 2]

        self.register_buffer("dwt_weight", dwt_weight)
        self.in_channels = in_channels

        # Patch conv: input = 4*in_channels sub-bands → embed_dim
        dwt_channels = 4 * in_channels
        stride = patch_size // 2  # DWT đã giảm 2× rồi

        # Stronger stem: 3×3 conv extracts local features from DWT sub-bands,
        # then a strided conv downsamples to the patch grid.
        # No BatchNorm here — BN is DDP-unfriendly when each GPU has small batch
        # (stats diverge per-GPU without SyncBN, causing gradient instability).
        # The trailing LayerNorm on the flattened tokens handles normalisation.
        self.proj = nn.Sequential(
            nn.Conv2d(dwt_channels, embed_dim, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=stride, stride=stride,
                      padding=0, bias=False),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            tokens: [B, H'*W', embed_dim]
            H', W': spatial dims của token grid
        """
        B, C, H, W = x.shape

        # ── Haar DWT (depthwise, stride=2) ───────────────────────────────
        # [B, C, H, W] → [B, 4*C, H/2, W/2]
        dwt = F.conv2d(
            x, self.dwt_weight,
            stride=2, padding=0,
            groups=self.in_channels,
        )

        # ── Patch embedding conv ───────────────────────────────────────────
        tokens = self.proj(dwt)      # [B, embed_dim, H', W']
        H_out, W_out = tokens.shape[2], tokens.shape[3]

        # ── Flatten + norm ────────────────────────────────────────────────
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        tokens = self.norm(tokens)

        return tokens, H_out, W_out


# ---------------------------------------------------------------------------
# Patch Merging (downsampling giữa stages)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """
    Giảm resolution 2× và tăng channels 2×.
    Dùng Conv2d stride=2 thay vì concatenation (hiệu quả hơn cho edge).

    Args:
        in_dim:  input channels
        out_dim: output channels (thường = 2 * in_dim)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm  = nn.LayerNorm(in_dim)
        self.proj  = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        """
        Args:
            x:    [B, N, C]  với N = H*W
            H, W: spatial dims
        Returns:
            x_merged: [B, H/2 * W/2, 2C]
            H//2, W//2
        """
        B, N, C = x.shape
        x = self.norm(x)

        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x_2d = self.proj(x_2d)                           # [B, 2C, H/2, W/2]

        H_new, W_new = x_2d.shape[2], x_2d.shape[3]
        x = x_2d.flatten(2).transpose(1, 2)              # [B, H/2*W/2, 2C]
        x = self.norm2(x)

        return x, H_new, W_new


# ---------------------------------------------------------------------------
# Distillation-aware classifier head
# ---------------------------------------------------------------------------

class SWIFTNetClassifier(nn.Module):
    """
    Dual-head classifier supporting knowledge distillation.

    During training with distillation=True: returns (cls_logits, dist_logits).
    During eval (or distillation=False):    returns averaged logits.

    Args:
        dim:          input feature dimension
        num_classes:  number of output classes (0 = Identity)
        distillation: enable distillation head
        drop:         dropout rate before the linear layers
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        distillation: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.distillation = distillation
        self.head_drop    = nn.Dropout(drop)
        self.head      = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x  = self.head_drop(x)
        x1 = self.head(x)
        x2 = self.head_dist(x)
        if self.training and self.distillation and not torch.jit.is_scripting():
            return x1, x2
        return (x1 + x2) / 2

    @torch.no_grad()
    def fuse(self) -> nn.Linear | nn.Identity:
        """Merge the two linear heads into one (for deployment)."""
        if not self.num_classes > 0:
            return nn.Identity()
        head      = self.head
        head_dist = self.head_dist
        w = (head.weight + head_dist.weight) / 2
        b = (head.bias   + head_dist.bias)   / 2
        m = nn.Linear(w.size(1), w.size(0), device=head.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# ---------------------------------------------------------------------------
# SWIFTNet Backbone
# ---------------------------------------------------------------------------

class SWIFTNet(nn.Module):
    """
    SWIFT-Net Backbone: State-space Wavelet-Integrated Fast Transformer.

    Timm-compatible:
        - Accepts a SWIFTNetConfig *or* flat kwargs (for build_model_with_cfg)
        - Exposes num_features and feature_info for feature pyramid usage
        - forward_features() returns token tensor [B, N, C] (pre-pool)
        - forward_head()     applies global pool + classification head
        - Distillation dual-head via SWIFTNetClassifier

    Args:
        config: SWIFTNetConfig instance (or None — build from **kwargs)
        **kwargs: flat params forwarded to SWIFTNetConfig when config is None
    """

    def __init__(self, config: Optional[SWIFTNetConfig] = None, **kwargs):
        super().__init__()

        # ── Build config from flat kwargs when called via build_model_with_cfg
        if config is None:
            valid = {f.name for f in dataclasses.fields(SWIFTNetConfig)}
            config = SWIFTNetConfig(**{k: v for k, v in kwargs.items() if k in valid})

        self.config      = config
        self.num_stages  = len(config.dims)
        self.global_pool = config.global_pool
        self.num_features = config.dims[-1]

        # ── feature_info (timm convention) ───────────────────────────────
        # patch_embed reduces by patch_size; each PatchMerging reduces by 2×
        stride = config.patch_size
        self.feature_info = []
        for i, dim in enumerate(config.dims):
            self.feature_info.append(
                dict(num_chs=dim, reduction=stride, module=f"stages.{i}")
            )
            stride *= 2  # next stage is 2× lower resolution

        # ── Patch Embedding ───────────────────────────────────────────────
        self.patch_embed = WaveletPatchEmbed(
            in_channels=config.in_channels,
            embed_dim=config.dims[0],
            patch_size=config.patch_size,
        )

        # ── Stochastic depth decay rule ───────────────────────────────────
        total_blocks = sum(config.depths)
        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, total_blocks)
        ]

        # ── Build stages + patch merging ──────────────────────────────────
        self.stages  = nn.ModuleList()
        self.mergers = nn.ModuleList()
        self._rebuild_stages(config, dpr)

        # ── Final norm ────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(self.num_features)

        # ── Classification head (with optional distillation) ──────────────
        self.head = SWIFTNetClassifier(
            dim=self.num_features,
            num_classes=config.num_classes,
            distillation=config.distillation,
            drop=config.drop_rate,
        )

        # ── Init weights ─────────────────────────────────────────────────
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Stage builder
    # ------------------------------------------------------------------

    def _rebuild_stages(self, config: SWIFTNetConfig, dpr: list[float]) -> None:
        """Rebuild stages với global block index đúng."""
        self.stages  = nn.ModuleList()
        self.mergers = nn.ModuleList()
        block_idx_global = 0

        for stage_i in range(self.num_stages):
            dim    = config.dims[stage_i]
            depth  = config.depths[stage_i]
            nheads = config.num_heads[stage_i]
            n_late = max(1, int(depth * config.late_ratio))

            blocks = nn.ModuleList()
            for block_i in range(depth):
                blocks.append(SWIFTBlock(
                    dim=dim,
                    num_heads=nheads,
                    window_size=config.window_size,
                    mlp_expand=config.mlp_expand,
                    d_state=config.d_state,
                    ssm_rank=config.ssm_rank,
                    ssm_kernel_size=config.ssm_kernel_size,
                    kd_rank=config.kd_rank,
                    wavelet_levels=config.wavelet_levels,
                    num_rff=config.num_rff,
                    rope_base=config.rope_base,
                    is_late=(block_i >= depth - n_late),
                    block_idx=block_idx_global,
                    drop=config.drop_rate,
                    drop_path=dpr[block_idx_global],
                ))
                block_idx_global += 1

            self.stages.append(blocks)

            if stage_i < self.num_stages - 1:
                self.mergers.append(
                    PatchMerging(dim, config.dims[stage_i + 1])
                )

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Extract token features — without pooling or head.

        Args:
            x: [B, C, H, W]
        Returns:
            tokens: [B, N, last_dim]  — after final LayerNorm
        """
        x, H, W = self.patch_embed(x)  # [B, N, d0]

        for stage_i, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x, H, W)

            if stage_i < self.num_stages - 1:
                x, H, W = self.mergers[stage_i](x, H, W)

        return self.norm(x)  # [B, N, last_dim]

    def forward_head(self, x: Tensor) -> Tensor:
        """
        Apply global pool then classification head.

        Args:
            x: [B, N, last_dim]  — output of forward_features
        Returns:
            logits: [B, num_classes]  (or tuple during distillation training)
        """
        if self.global_pool == "avg":
            x = x.mean(dim=1)  # [B, last_dim]
        return self.head(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Full forward pass.

        Args:
            x: [B, 3, H, W]
        Returns:
            logits: [B, num_classes]
            During distillation training: (cls_logits, dist_logits)
        """
        return self.forward_head(self.forward_features(x))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        """Đếm params của từng module."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        stage_params = {}
        for i, stage in enumerate(self.stages):
            stage_params[f"stage_{i}"] = sum(p.numel() for p in stage.parameters())

        return {
            "total":       total,
            "trainable":   trainable,
            "patch_embed": sum(p.numel() for p in self.patch_embed.parameters()),
            **stage_params,
            "head":        sum(p.numel() for p in self.head.parameters()),
        }

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        """
        Trả về feature maps của từng stage (dùng cho dense prediction).

        Args:
            x: [B, C, H, W]
        Returns:
            list of [B, N_i, dim_i] cho mỗi stage
        """
        maps = []
        x, H, W = self.patch_embed(x)

        for stage_i, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x, H, W)
            maps.append(x)

            if stage_i < self.num_stages - 1:
                x, H, W = self.mergers[stage_i](x, H, W)

        return maps

    @torch.no_grad()
    def fuse(self) -> None:
        """Fuse the distillation dual-head into a single linear layer."""
        fused = self.head.fuse()
        self.head = fused


# ---------------------------------------------------------------------------
# timm helper
# ---------------------------------------------------------------------------

def _create_swift_net(variant: str, pretrained: bool = False, **kwargs) -> SWIFTNet:
    return build_model_with_cfg(SWIFTNet, variant, pretrained, **kwargs)


# ---------------------------------------------------------------------------
# Factory functions — registered with timm
# ---------------------------------------------------------------------------

@register_model
def swift_net_tiny(pretrained: bool = False, **kwargs) -> SWIFTNet:
    """SWIFT-Net Tiny — ~3M params. Target: mobile-first."""
    model_args = dict(
        dims=[32, 64, 128, 256],
        depths=[1, 2, 4, 1],
        num_heads=[1, 2, 4, 8],
        d_state=8,
        kd_rank=8,
        num_rff=32,
        wavelet_levels=1,
        drop_path_rate=0.05,
    )
    return _create_swift_net("swift_net_tiny", pretrained, **{**model_args, **kwargs})


@register_model
def swift_net_small(pretrained: bool = False, **kwargs) -> SWIFTNet:
    """SWIFT-Net Small — ~8M params. Target: balanced accuracy/efficiency."""
    model_args = dict(
        dims=[48, 96, 192, 384],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 12],
        d_state=16,
        kd_rank=16,
        num_rff=64,
        wavelet_levels=2,
        drop_path_rate=0.1,
    )
    return _create_swift_net("swift_net_small", pretrained, **{**model_args, **kwargs})


@register_model
def swift_net_base(pretrained: bool = False, **kwargs) -> SWIFTNet:
    """SWIFT-Net Base — ~20M params. Target: high accuracy, server-class edge."""
    model_args = dict(
        dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[2, 4, 8, 16],
        d_state=32,
        kd_rank=16,
        num_rff=128,
        wavelet_levels=2,
        drop_path_rate=0.2,
    )
    return _create_swift_net("swift_net_base", pretrained, **{**model_args, **kwargs})


if __name__ == "__main__":
    from timm.models import create_model

    model = create_model("swift_net_tiny", num_classes=1000)
    model.eval()
    print(model)

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("output shape:", out.shape)

    # Distillation training demo
    model_dist = create_model("swift_net_tiny", num_classes=1000, distillation=True)
    model_dist.train()
    cls_out, dist_out = model_dist(x)
    print("distillation cls shape:", cls_out.shape, " dist shape:", dist_out.shape)

    print("\nParameter counts:")
    for k, v in model.count_parameters().items():
        print(f"  {k}: {v:,}")
