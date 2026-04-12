# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from https://github.com/facebookresearch/dinov3 for LW-ViT / SWIFTNet classification.

import re
import logging
from collections import defaultdict

logger = logging.getLogger("lw_vit")


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Supports both standard ViT naming (blocks.N) and SWIFTNet stage naming (stages.N.N).

      - patch_embed.* / cls_token / reg_tokens  → layer 0  (lowest LR)
      - blocks.{i}.*   (ViT flat)               → layer i+1
      - stages.{s}.{b} (SWIFTNet nested)        → layer sum(depths[:s]) + b + 1
      - head.* / norm.*                         → layer num_layers+1 (full LR)
    """
    layer_id = num_layers + 1  # default: full lr (no decay)

    if "patch_embed" in name or "cls_token" in name or "reg_tokens" in name:
        layer_id = 0
    elif ".blocks." in name:
        # ViT flat naming: backbone.blocks.N.* or blocks.N.*
        layer_id = int(name[name.find(".blocks."):].split(".")[2]) + 1
    elif "stages." in name:
        # SWIFTNet nested naming: stages.{stage_i}.{block_i}.*
        m = re.search(r"stages\.(\d+)", name)
        if m:
            # Map each stage to a layer group (stage 0 → layer 1, stage 3 → layer 4)
            stage_i = int(m.group(1))
            layer_id = stage_i + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    """
    Build per-parameter optimizer groups with layerwise LR decay and selective weight decay.
    Supports both ViT (flat blocks) and SWIFTNet (nested stages) architectures.

    Rules:
      - No weight decay on: biases, LayerNorm/norm params, tokens & positional embeddings.
      - Layerwise LR decay from the last block back to the patch embedding.
      - patch_embed gets an additional lr_mult (set to 1.0 to disable extra suppression).
    """
    if hasattr(model, "stages"):
        # SWIFTNet: stage-based architecture
        n_blocks = len(model.stages)
        logger.info(f"SWIFTNet detected: {n_blocks} stages for LLRD.")
    elif hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        n_blocks = len(model.backbone.blocks)
    elif hasattr(model, "blocks"):
        n_blocks = len(model.blocks)
    else:
        n_blocks = 0
        logger.warning("Could not determine number of blocks; layerwise LR decay disabled.")

    all_param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        decay_rate = get_vit_lr_decay_rate(name, lr_decay_rate, num_layers=n_blocks)

        d = {
            "name": name,
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
        }

        # No weight decay on biases, norm layers (ln1, ln2, ln_out, norm*),
        # learned tokens and positional embeddings — same rule as DINOv3.
        if (
            name.endswith("bias")
            or "norm" in name
            or "ln1" in name
            or "ln2" in name
            or "ln_out" in name
            or "cls_token" in name
            or "reg_tokens" in name
        ):
            d["wd_multiplier"] = 0.0

        # patch_embed gets reduced LR (DINOv3: patch_embed_lr_mult = 0.2)
        if "patch_embed" in name:
            d["lr_multiplier"] *= patch_embed_lr_mult

        all_param_groups.append(d)
        logger.debug(
            f"{name}: lr_multiplier={d['lr_multiplier']:.4f}, wd_multiplier={d['wd_multiplier']}"
        )

    return all_param_groups


def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer")):
    """Merge param dicts that share the same (lr_multiplier, wd_multiplier, is_last_layer) triple."""
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = "".join(k + str(d[k]) + "_" for k in keys)
        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])
    return fused_params_groups.values()


def apply_optim_scheduler(optimizer, lr, wd):
    """
    Apply learning rate and weight decay to all optimizer param groups.
    Called at every training iteration (not epoch) — same as DINOv3.

    Each group's effective lr  = lr  * group["lr_multiplier"]
    Each group's effective wd  = wd  * group["wd_multiplier"]
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group["lr_multiplier"]
        param_group["weight_decay"] = wd * param_group["wd_multiplier"]
