"""
train_simple.py — Training with DINOv3-style LR/scheduler for SWIFTNet.

DINOv3 strategy (from dinov3/dinov3/train/):
  1. LR scaling:  sqrt rule  → lr = base_lr * 4 * sqrt(total_bs / 1024)
  2. LR schedule: CosineScheduler — linear warmup from 0 → peak, then cosine → min_lr
  3. WD schedule: CosineScheduler — cosine from wd_start → wd_end  (both scheduled!)
  4. LLRD:  lr_multiplier = layer_decay ^ (num_blocks + 1 - layer_id)
            patch_embed → layer_id=0 (lowest LR)
            stages.S.B → layer_id = cumulative_block_offset + B + 1
            head/norm   → layer_id = num_blocks + 1 (highest LR)
  5. No weight decay on bias / norm / gamma / ls1 / ls2
  6. patch_embed gets additional patch_embed_lr_mult (default 0.2)
  7. apply_optim_scheduler called every step: sets lr & wd per param group
  8. Gradient clipping (default 3.0)

Usage:
    python train_simple.py
    python train_simple.py --amp --epochs 20 --batch_size 32
    python train_simple.py --layer_decay 0.9 --base_lr 1e-3
"""

import argparse
import math
import time
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

import model  # registers timm models
import timm


# ---------------------------------------------------------------------------
# DINOv3 CosineScheduler (from dinov3/dinov3/train/cosine_lr_scheduler.py)
# ---------------------------------------------------------------------------

class CosineScheduler:
    """
    Precomputed numpy schedule: freeze → linear warmup → cosine decay.
    Indexed by step: scheduler[it] returns the value at iteration `it`.
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
    ):
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule  = np.zeros(freeze_iters)
        warmup_schedule  = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        cosine_schedule  = (
            final_value
            + 0.5 * (base_value - final_value)
            * (1 + np.cos(np.pi * iters / len(iters)))
        )
        self.schedule = np.concatenate(
            [freeze_schedule, warmup_schedule, cosine_schedule]
        ).astype(np.float64)
        assert len(self.schedule) == total_iters

    def __getitem__(self, it: int) -> float:
        if it >= self.total_iters:
            return float(self.final_value)
        return float(self.schedule[it])


# ---------------------------------------------------------------------------
# DINOv3 param groups with LLRD
# (adapted from dinov3/dinov3/train/param_groups.py for SWIFTNet naming)
# ---------------------------------------------------------------------------

def _get_swiftnet_layer_id(name: str, num_blocks: int, stage_offsets: List[int]) -> int:
    """
    Map a SWIFTNet parameter name to a layer depth index.

    Layer IDs:
        0                  — patch_embed
        1 … num_blocks     — stages.S.B  (cumulative block index + 1)
        num_blocks + 1     — head, norm, mergers (everything else)
    """
    if "patch_embed" in name:
        return 0

    if "stages." in name:
        # e.g. "stages.2.3.ffn.w1.weight"
        parts = name.split(".")
        try:
            stage_idx = int(parts[parts.index("stages") + 1])
            block_idx = int(parts[parts.index("stages") + 2])
            return stage_offsets[stage_idx] + block_idx + 1
        except (ValueError, IndexError):
            pass

    # head, norm, mergers → highest layer id
    return num_blocks + 1


def get_params_groups(
    model: nn.Module,
    layer_decay: float = 0.9,
    patch_embed_lr_mult: float = 0.2,
    weight_decay: float = 0.04,
) -> List[dict]:
    """
    Build per-parameter groups exactly as DINOv3 does:
      - lr_multiplier  = layer_decay ^ (num_blocks + 1 - layer_id)
      - wd_multiplier  = 0 for bias / norm / gamma / layer-scale params
      - patch_embed gets extra lr_multiplier *= patch_embed_lr_mult

    The actual lr and wd values are set every step by apply_optim_scheduler.
    """
    # Compute cumulative block offsets per stage
    stages = getattr(model, "stages", None)
    if stages is not None:
        stage_depths = [len(s) for s in stages]
    else:
        stage_depths = []
    num_blocks = sum(stage_depths)
    stage_offsets = []
    running = 0
    for d in stage_depths:
        stage_offsets.append(running)
        running += d

    all_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id   = _get_swiftnet_layer_id(name, num_blocks, stage_offsets)
        lr_mult    = layer_decay ** (num_blocks + 1 - layer_id)

        # No WD on 1-D params: bias, norm weight/bias, layer scale (ls1/ls2/gamma)
        no_wd = (
            param.ndim == 1
            or name.endswith("bias")
            or "norm" in name
            or "gamma" in name
            or name.endswith("ls1")
            or name.endswith("ls2")
        )
        wd_mult = 0.0 if no_wd else 1.0

        if "patch_embed" in name:
            lr_mult *= patch_embed_lr_mult

        all_params.append({
            "name":         name,
            "params":       param,
            "lr_multiplier": lr_mult,
            "wd_multiplier": wd_mult,
            "is_last_layer": "last_layer" in name,
        })

    # Fuse into fewer groups by (lr_mult, wd_mult, is_last_layer)
    fused = defaultdict(lambda: {"params": []})
    for d in all_params:
        key = f"lr{d['lr_multiplier']:.6f}_wd{d['wd_multiplier']}_ll{d['is_last_layer']}"
        fused[key]["params"].append(d["params"])
        fused[key]["lr_multiplier"]  = d["lr_multiplier"]
        fused[key]["wd_multiplier"]  = d["wd_multiplier"]
        fused[key]["is_last_layer"]  = d["is_last_layer"]
        fused[key]["name"]           = key
        # Placeholder values; overwritten every step by apply_optim_scheduler
        fused[key]["lr"]             = 0.0
        fused[key]["weight_decay"]   = 0.0

    return list(fused.values())


# ---------------------------------------------------------------------------
# DINOv3 apply_optim_scheduler  (from dinov3/dinov3/train/train.py)
# ---------------------------------------------------------------------------

def apply_optim_scheduler(optimizer, lr: float, wd: float, last_layer_lr: float):
    """Set lr and wd on each param group every step."""
    for pg in optimizer.param_groups:
        lr_mult = pg["lr_multiplier"]
        wd_mult = pg["wd_multiplier"]
        pg["weight_decay"] = wd * wd_mult
        if pg["is_last_layer"]:
            pg["lr"] = last_layer_lr * lr_mult
        else:
            pg["lr"] = lr * lr_mult


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples: int = 8192, num_classes: int = 10,
                 img_size: int = 224, augment: bool = False):
        torch.manual_seed(42)
        self.images = torch.randn(num_samples, 3, img_size, img_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        # Basic augmentation to reduce overfitting
        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(img_size, padding=img_size // 8),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(net, loader, criterion, optimizer,
                    lr_schedule, wd_schedule, last_layer_lr_schedule,
                    step_offset: int, clip_grad: float,
                    device, amp, scaler):
    net.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        it = step_offset + batch_idx
        lr = lr_schedule[it]
        wd = wd_schedule[it]
        ll_lr = last_layer_lr_schedule[it]
        apply_optim_scheduler(optimizer, lr, wd, ll_lr)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = net(images)
            loss   = criterion(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(net, loader, criterion, device, amp):
    net.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = net(images)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",              default="swift_net_tiny", type=str)
    p.add_argument("--num_classes",        default=10,     type=int)
    p.add_argument("--img_size",           default=224,    type=int)
    p.add_argument("--train_size",         default=1000,   type=int,
                   help="More samples needed: 3.35M param model overfit 1024 easily")
    p.add_argument("--val_size",           default=100,   type=int)
    p.add_argument("--batch_size",         default=16,     type=int)
    p.add_argument("--epochs",             default=20,     type=int)
    # DINOv3 defaults from ssl_default_config.yaml
    p.add_argument("--base_lr",            default=1e-3,   type=float,
                   help="Base LR before scaling (DINOv3 default: 1e-3)")
    p.add_argument("--min_lr",             default=1e-6,   type=float)
    p.add_argument("--weight_decay",       default=0.05,   type=float,
                   help="Fixed WD for classification (DINOv3's 0.04→0.4 ramp is SSL-only)")
    p.add_argument("--weight_decay_end",   default=0.05,   type=float,
                   help="Set equal to weight_decay to keep WD fixed throughout training")
    p.add_argument("--warmup_epochs",      default=5,      type=int)
    p.add_argument("--layer_decay",        default=0.9,    type=float,
                   help="LLRD factor (DINOv3 default: 0.9)")
    p.add_argument("--patch_embed_lr_mult", default=0.2,   type=float,
                   help="Extra LR multiplier for patch_embed (DINOv3 default: 0.2)")
    p.add_argument("--clip_grad",          default=3.0,    type=float)
    p.add_argument("--freeze_last_layer_epochs", default=1, type=int)
    p.add_argument("--amp",                action="store_true", default=False)
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers",        default=2,      type=int)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = torch.device(args.device)

    # ── DINOv3 LR scaling: sqrt rule wrt 1024 ────────────────────────────
    # lr = base_lr * 4 * sqrt(total_batch_size / 1024)
    total_bs = args.batch_size  # single GPU; multiply by world_size for DDP
    lr_peak  = args.base_lr * 4 * math.sqrt(total_bs / 1024.0)
    lr_min   = args.min_lr  * 4 * math.sqrt(total_bs / 1024.0)
    print(f"Device: {device}  |  AMP: {args.amp}")
    print(f"LR scaling (sqrt wrt 1024): base={args.base_lr:.2e} → peak={lr_peak:.2e}, min={lr_min:.2e}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = SyntheticImageDataset(args.train_size, args.num_classes, args.img_size, augment=False)
    val_ds   = SyntheticImageDataset(args.val_size,   args.num_classes, args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = steps_per_epoch * args.warmup_epochs
    freeze_ll_steps = steps_per_epoch * args.freeze_last_layer_epochs
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | "
          f"{steps_per_epoch} steps/epoch | {warmup_steps} warmup steps")

    # ── Model (layer scale stays at 1e-5 as designed) ────────────────────
    net = timm.create_model(
        args.model, pretrained=False, num_classes=args.num_classes,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Params: {n_params/1e6:.2f}M")

    # ── DINOv3 precomputed schedules ──────────────────────────────────────
    lr_schedule = CosineScheduler(
        base_value=lr_peak,
        final_value=lr_min,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
        start_warmup_value=0.0,
    )
    wd_schedule = CosineScheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        total_iters=total_steps,
    )
    # last-layer LR: same as lr_schedule but frozen for first N steps
    last_layer_lr_schedule = CosineScheduler(
        base_value=lr_peak,
        final_value=lr_min,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
        start_warmup_value=0.0,
        freeze_iters=freeze_ll_steps,
    )

    # ── DINOv3 param groups with LLRD ────────────────────────────────────
    param_groups = get_params_groups(
        net,
        layer_decay=args.layer_decay,
        patch_embed_lr_mult=args.patch_embed_lr_mult,
        weight_decay=args.weight_decay,
    )
    print(f"Param groups: {len(param_groups)}")
    for g in sorted(param_groups, key=lambda x: -x["lr_multiplier"])[:5]:
        n = sum(p.numel() for p in g["params"])
        print(f"  lr_mult={g['lr_multiplier']:.4f}  wd_mult={g['wd_multiplier']}  "
              f"last_layer={g['is_last_layer']}  params={n/1e3:.1f}k")

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0          = time.time()
        step_offset = (epoch - 1) * steps_per_epoch

        train_stats = train_one_epoch(
            net, train_loader, criterion, optimizer,
            lr_schedule, wd_schedule, last_layer_lr_schedule,
            step_offset, args.clip_grad, device, args.amp, scaler,
        )
        val_stats = evaluate(net, val_loader, criterion, device, args.amp)

        # Current LR / WD from schedule (first step of next epoch)
        it = epoch * steps_per_epoch
        cur_lr = lr_schedule[min(it, total_steps - 1)]
        cur_wd = wd_schedule[min(it, total_steps - 1)]
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"train_loss={train_stats['loss']:.4f}  train_acc={train_stats['acc']:.4f}  "
            f"val_loss={val_stats['loss']:.4f}  val_acc={val_stats['acc']:.4f}  "
            f"lr={cur_lr:.2e}  wd={cur_wd:.4f}  time={elapsed:.1f}s"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save(net.state_dict(), "best_model.pth")

    print(f"\nDone. Best val acc: {best_val_acc:.4f}  |  Saved to best_model.pth")


if __name__ == "__main__":
    main()
