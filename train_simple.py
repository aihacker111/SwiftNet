"""
train.py — Dogs vs Cats training with LW-ViT + Albumentations + multi-GPU DDP.

Expected dataset layout:
    <data_dir>/
    ├── train/
    │   ├── cat/    (or any two class names)
    │   └── dog/
    └── val/
        ├── cat/
        └── dog/

Single GPU:
    python train.py --data-dir /path/to/dataset

Multi-GPU (torchrun, recommended):
    torchrun --nproc_per_node=4 train.py --data-dir /path/to/dataset

Multi-GPU (legacy torch.distributed.launch):
    python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
        --data-dir /path/to/dataset

Kaggle example:
    python train.py --data-dir /kaggle/working/cat-and-dog --epochs 50 --batch-size 64

Install deps:
    pip install albumentations
"""

import argparse
import os
import sys
import time
import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Make sure the local `model` package is importable ──────────────────────
_DIR = Path(__file__).parent.resolve()
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import model  # registers lw_vit_* models with timm
from timm.models import create_model
import utils  # project-level dist helpers (init_distributed_mode, save_on_master, …)
from train.param_groups import (
    get_params_groups_with_decay,
    fuse_params_groups,
    apply_optim_scheduler,
)
from train.cosine_lr_scheduler import CosineScheduler


# ───────────────────────────────────────────────────────────────────────────
# Albumentations transforms
# ───────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, is_train: bool) -> A.Compose:
    if is_train:
        return A.Compose([
            # Spatial
            A.RandomResizedCrop(height=img_size, width=img_size,
                                scale=(0.08, 1.0), ratio=(0.75, 1.33),
                                interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.4,
                               border_mode=cv2.BORDER_REFLECT_101),
            # Color & texture
            A.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=20,
                                 val_shift_limit=10, p=0.3),
            A.ToGray(p=0.02),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.Sharpen(alpha=(0.1, 0.3), p=0.2),
            # Noise & dropout
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
            A.CoarseDropout(
                max_holes=8, max_height=img_size // 8, max_width=img_size // 8,
                min_holes=1, min_height=img_size // 16, min_width=img_size // 16,
                fill_value=0, p=0.3,
            ),
            # Normalise + to tensor
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    # Validation: deterministic resize + crop only
    return A.Compose([
        A.Resize(height=int(img_size * 256 / 224),
                 width=int(img_size * 256 / 224),
                 interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ───────────────────────────────────────────────────────────────────────────
# Dataset wrapper: feeds numpy arrays to Albumentations
# ───────────────────────────────────────────────────────────────────────────

class AlbumentationsDataset(Dataset):
    """Wraps torchvision ImageFolder to use Albumentations transforms."""

    def __init__(self, root: str, transform: A.Compose):
        self.base    = ImageFolder(root)
        self.transform = transform
        self.classes        = self.base.classes
        self.class_to_idx   = self.base.class_to_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        # Load as RGB numpy array (H, W, 3) for Albumentations
        image = np.array(Image.open(path).convert('RGB'))
        augmented = self.transform(image=image)
        return augmented['image'], label


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count

    def synchronize(self):
        """All-reduce sum+count across DDP ranks, then recompute avg."""
        if not dist.is_available() or not dist.is_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64,
                         device='cuda')
        dist.barrier()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.sum, self.count = t[0].item(), t[1].item()
        self.avg = self.sum / max(self.count, 1)


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item() * 100.0


def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def is_main() -> bool:
    """True on rank-0 (or when not using distributed)."""
    return utils.is_main_process()


def print_main(*args, **kwargs):
    """Print only on rank-0."""
    if is_main():
        print(*args, **kwargs)


# ───────────────────────────────────────────────────────────────────────────
# One epoch
# ───────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch,
    clip_grad, lr_schedule, wd_schedule, start_iteration,
):
    """
    Per-iteration LR/WD schedule (DINOv3-style):
      - lr_schedule[i]  → base LR at global iteration i
      - wd_schedule[i]  → base WD at global iteration i
      - apply_optim_scheduler scales each param group by its own
        lr_multiplier / wd_multiplier (layer-wise LR decay).
    DDP: gradients are averaged across all ranks automatically by DDP.
         Metrics are all-reduced at epoch end for accurate reporting.
    """
    model.train()
    loss_m      = AverageMeter()
    acc_m       = AverageMeter()
    nan_skipped = 0
    t0 = time.time()

    for step, (images, targets) in enumerate(loader):
        global_iter = start_iteration + step

        # ── Per-iteration LR & WD (applied to all param groups) ───────────
        lr = lr_schedule[global_iter]
        wd = wd_schedule[global_iter]
        apply_optim_scheduler(optimizer, lr=lr, wd=wd)

        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss   = criterion(logits, targets)

        # ── NaN / Inf guard ────────────────────────────────────────────────
        if not torch.isfinite(loss):
            print_main(
                f'  [WARN] step {step+1}: non-finite loss ({loss.item()}), '
                f'skipping update.',
                flush=True,
            )
            nan_skipped += 1
            if nan_skipped >= 10:
                print_main(
                    '  [ERROR] 10 consecutive NaN/Inf losses — weights likely '
                    'corrupted. Lower --lr or --clip-grad and restart.',
                    flush=True,
                )
                return {'loss': float('nan'), 'acc': acc_m.avg,
                        'nan_skipped': nan_skipped}
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        # Track local metrics; all-reduce happens at epoch end
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(accuracy(logits.detach(), targets), images.size(0))

        if is_main() and ((step + 1) % 20 == 0 or (step + 1) == len(loader)):
            elapsed = time.time() - t0
            scale_str = (f'  scale={scaler.get_scale():.0f}'
                         if scaler is not None else '')
            print(
                f'  Epoch [{epoch}] step {step+1:>4}/{len(loader)} | '
                f'loss {loss_m.avg:.4f} | acc {acc_m.avg:.2f}% | '
                f'lr={lr:.2e}  wd={wd:.4f}{scale_str}',
                flush=True,
            )

    # ── Synchronize metrics across all ranks ──────────────────────────────
    loss_m.synchronize()
    acc_m.synchronize()

    if nan_skipped:
        print_main(f'  [WARN] {nan_skipped} batch(es) skipped due to NaN/Inf loss this epoch.')

    return {'loss': loss_m.avg, 'acc': acc_m.avg, 'nan_skipped': nan_skipped}


@torch.no_grad()
def evaluate(model, loader, criterion, device, sync_ddp: bool = True):
    """
    Evaluate on the val set.  When using a DistributedSampler the sampler
    may pad the last batch with duplicates; metrics are all-reduced across
    ranks so the returned values are the global average.
    """
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits  = model(images)
        loss    = criterion(logits, targets)
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(accuracy(logits, targets), images.size(0))

    # Synchronize across DDP ranks only when every rank participates.
    # If only rank-0 runs validation (dist_eval=False), syncing here would hang.
    if sync_ddp:
        loss_m.synchronize()
        acc_m.synchronize()

    return {'loss': loss_m.avg, 'acc': acc_m.avg}


# ───────────────────────────────────────────────────────────────────────────
# Reparameterization evaluation
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_reparam_compare(
    net,
    val_loader,
    criterion,
    device,
    img_size: int = 224,
    fuse_qkv: bool = True,
    speed_device: str = 'cpu',
    speed_batch: int = 1,
    speed_warmup: int = 30,
    speed_iters: int = 200,
    verify_samples: int = 8,
    verify_tol_abs: float = 1e-6,
    verify_tol_rel: float = 1e-6,
):
    """
    Compare the model before vs after reparameterize() across:
      - Val accuracy & loss
      - Parameter count (total + per-group breakdown)
      - GFLOPs
      - Latency (ms/image)
    """
    from copy import deepcopy

    if not hasattr(net, 'reparameterize'):
        print('[WARN] Model has no reparameterize() — skipping reparam compare.')
        return

    W   = 76
    sep = '─' * W
    bar = '═' * W

    print(f'\n{bar}')
    print('  Reparameterization Compare')
    print(f'  fuse_qkv={fuse_qkv}   speed_device={speed_device}   '
          f'batch={speed_batch}   warmup={speed_warmup}   iters={speed_iters}')
    print(bar)

    # ── Build before / after models ────────────────────────────────────────
    net_b = deepcopy(net).eval().to(device)   # before
    net_a = deepcopy(net).eval().to(device)   # after
    net_a.reparameterize(fuse_qkv=fuse_qkv, verbose=False)

    # ── Val accuracy ───────────────────────────────────────────────────────
    def _eval(m):
        m.eval()
        lm, am = AverageMeter(), AverageMeter()
        for imgs, tgts in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            out  = m(imgs)
            lm.update(criterion(out, tgts).item(), imgs.size(0))
            am.update(accuracy(out, tgts), imgs.size(0))
        return lm.avg, am.avg

    print(f'\n  Evaluating before reparam ...', flush=True)
    loss_b, acc_b = _eval(net_b)
    print(f'  Evaluating after  reparam ...', flush=True)
    loss_a, acc_a = _eval(net_a)

    # ── Parameter count ────────────────────────────────────────────────────
    total_b = sum(p.numel() for p in net_b.parameters())
    total_a = sum(p.numel() for p in net_a.parameters())

    # Per-group breakdown via backbone
    bb_b = getattr(net_b, 'backbone', net_b)
    bb_a = getattr(net_a, 'backbone', net_a)
    try:
        breakdown_b = _param_breakdown(bb_b)
        breakdown_a = _param_breakdown(bb_a)
        has_breakdown = True
    except Exception:
        has_breakdown = False

    # ── GFLOPs ─────────────────────────────────────────────────────────────
    try:
        gflops_b = _count_gflops(net_b, img_size, device=str(device))
        gflops_a = _count_gflops(net_a, img_size, device=str(device))
        has_gflops = True
    except Exception:
        has_gflops = False

    # ── Latency ────────────────────────────────────────────────────────────
    sp_net_b = deepcopy(net_b).to(speed_device)
    sp_net_a = deepcopy(net_a).to(speed_device)
    sample   = torch.randn(speed_batch, 3, img_size, img_size, device=speed_device)

    print(f'  Measuring latency on {speed_device} (bs={speed_batch}) ...', flush=True)
    lat_b = _measure_latency_ms(sp_net_b, sample, warmup=speed_warmup,
                                iters=speed_iters, device=speed_device)
    lat_a = _measure_latency_ms(sp_net_a, sample, warmup=speed_warmup,
                                iters=speed_iters, device=speed_device)
    del sp_net_b, sp_net_a

    # ── Print table ────────────────────────────────────────────────────────
    def _row(label, before, after, fmt='.4f', unit=''):
        b_s  = f'{before:{fmt}}{unit}'
        a_s  = f'{after:{fmt}}{unit}'
        diff = after - before
        sign = '+' if diff >= 0 else ''
        d_s  = f'{sign}{diff:{fmt}}{unit}'
        ok   = '  OK' if abs(diff) < 1e-3 * max(abs(before), 1) else ''
        print(f'  {label:<24}  {b_s:>14}  {a_s:>14}  {d_s:>14}{ok}')

    print(f'\n  {"Metric":<24}  {"Before":>14}  {"After":>14}  {"Delta":>14}')
    print(f'  {sep}')

    print(f'\n  -- Accuracy & Loss --')
    _row('Val Acc@1 (%)', acc_b, acc_a, fmt='.3f')
    _row('Val Loss',      loss_b, loss_a, fmt='.5f')

    print(f'\n  -- Parameters --')
    _row('Total (M)', total_b / 1e6, total_a / 1e6, fmt='.3f', unit='M')
    if has_breakdown:
        groups = [k for k in breakdown_b if k != 'total']
        for g in groups:
            vb = breakdown_b.get(g, 0) / 1e6
            va = breakdown_a.get(g, 0) / 1e6
            if vb > 0 or va > 0:
                _row(f'  {g}', vb, va, fmt='.3f', unit='M')

    if has_gflops:
        print(f'\n  -- Computation --')
        _row('GFLOPs', gflops_b, gflops_a, fmt='.4f', unit='G')

    print(f'\n  -- Latency ({speed_device}, bs={speed_batch}) --')
    _row('Latency (ms/img)', lat_b, lat_a, fmt='.3f', unit='ms')
    speedup = lat_b / max(lat_a, 1e-9)
    pct     = (lat_a - lat_b) / max(lat_b, 1e-9) * 100
    sign    = '+' if pct >= 0 else ''
    print(f'  {"Speedup":<24}  {"":>14}  {speedup:>13.4f}x  '
          f'({sign}{pct:.2f}%)')

    print(f'\n  {bar}')

    # ── Numerical correctness check ────────────────────────────────────────
    # CUDA softmax/reduction order can introduce small non-associative drift
    # after algebraic fusion. Check both absolute and relative errors.
    max_abs = 0.0
    max_rel = 0.0
    for _ in range(max(1, verify_samples)):
        x = torch.randn(1, 3, img_size, img_size, device=device)
        out_b = net_b(x)
        out_a = net_a(x)
        diff = (out_b - out_a).abs()
        rel  = diff / out_b.abs().clamp_min(1e-6)
        max_abs = max(max_abs, diff.max().item())
        max_rel = max(max_rel, rel.max().item())

    passed = (max_abs <= verify_tol_abs) or (max_rel <= verify_tol_rel)
    status = 'PASS' if passed else 'FAIL'
    print(
        f'  Numerical correctness: max_abs={max_abs:.2e} (tol={verify_tol_abs:.1e})  '
        f'max_rel={max_rel:.2e} (tol={verify_tol_rel:.1e})  [{status}]'
    )
    print(f'  {bar}\n')

    return dict(
        acc_before=acc_b, acc_after=acc_a,
        loss_before=loss_b, loss_after=loss_a,
        params_before=total_b, params_after=total_a,
        gflops_before=gflops_b if has_gflops else None,
        gflops_after=gflops_a  if has_gflops else None,
        latency_before_ms=lat_b, latency_after_ms=lat_a,
        speedup=speedup, max_abs_error=max_abs, max_rel_error=max_rel, passed=passed,
    )


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser('LW-ViT Dogs vs Cats trainer')

    # Data
    parser.add_argument('--data-dir', default='./data', type=str,
                        help='Root folder with train/ and val/ sub-directories')
    parser.add_argument('--img-size', default=224, type=int)
    parser.add_argument('--num-classes', default=2, type=int,
                        help='Number of output classes (2 for dogs vs cats)')

    # Model
    parser.add_argument('--model', default='swift_net_tiny', type=str,
                        help='Model name registered in timm (e.g. swift_net_tiny, '
                             'lw_vit_nano_cls, lw_vit_xs_swiglu_cls)')
    parser.add_argument('--pretrained-ckpt', default='', type=str,
                        help='Optional path to a pretrained checkpoint (ImageNet). '
                             'The classifier head will be re-initialised automatically.')

    # Training
    parser.add_argument('--epochs',     default=100, type=int)
    parser.add_argument('--batch-size', default=64,  type=int)

    # ── Learning rate (DINOv3-style) ──────────────────────────────────────
    # Effective LR = base_lr * 4 * sqrt(total_bs / 1024)   [sqrt_wrt_1024]
    # For bs=64 single GPU: effective = 3e-4 * 4 * sqrt(64/1024) = 3e-4
    parser.add_argument('--lr',         default=3e-4, type=float,
                        help='Base LR before batch-size scaling.')
    parser.add_argument('--lr-scaling', default='sqrt_wrt_1024',
                        choices=['sqrt_wrt_1024', 'linear_wrt_256', 'none'],
                        help='LR scaling rule. sqrt_wrt_1024 = DINOv3 default.')
    parser.add_argument('--min-lr',     default=1e-6, type=float,
                        help='Final LR after cosine decay.')
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        help='Linear LR warmup epochs. Longer warmup prevents '
                             'early explosion on small datasets.')

    # ── Weight decay cosine schedule ──────────────────────────────────────
    parser.add_argument('--weight-decay',     default=0.04,  type=float,
                        help='Initial WD (DINOv3 default: 0.04).')
    parser.add_argument('--weight-decay-end', default=0.40,  type=float,
                        help='Final WD after cosine schedule (DINOv3: 0.40).')

    # ── Layer-wise LR decay (LLRD) ────────────────────────────────────────
    # block i effective_lr = base_lr * layerwise_decay^(num_blocks - i)
    # last block → lr * 1.0 | patch_embed → lr * 0.9^N * 0.2
    parser.add_argument('--layerwise-decay',     default=0.9,  type=float,
                        help='LLRD rate per block (DINOv3 default: 0.9). '
                             '1.0 disables LLRD.')
    parser.add_argument('--patch-embed-lr-mult', default=0.2,  type=float,
                        help='Extra LR multiplier for patch_embed (DINOv3: 0.2).')

    parser.add_argument('--label-smoothing', default=0.1, type=float)
    parser.add_argument('--clip-grad',  default=1.0, type=float,
                        help='Gradient clip norm (≤1.0 recommended for ViT).')
    parser.add_argument('--num-workers', default=4, type=int)

    # Checkpointing
    parser.add_argument('--output-dir', default='./checkpoints_dogcat', type=str)
    parser.add_argument('--save-freq',  default=10, type=int,
                        help='Save a periodic checkpoint every N epochs (0 = disable)')
    parser.add_argument('--resume',     default='', type=str,
                        help='Resume from checkpoint path')

    # Reparameterization evaluation
    parser.add_argument('--eval-reparam', action='store_true', default=False,
                        help='After training (or with --resume --eval-only), '
                             'run before vs after reparameterize() comparison.')
    parser.add_argument('--eval-only',   action='store_true', default=False,
                        help='Skip training; only run --eval-reparam (requires --resume).')
    parser.add_argument('--fuse-qkv',   action='store_true',  default=True,
                        help='Fuse QKV projection during reparameterize() (default: on).')
    parser.add_argument('--no-fuse-qkv', action='store_false', dest='fuse_qkv')
    parser.add_argument('--speed-device', default='cuda', type=str,
                        help='Device for latency benchmark (default: cuda). '
                             'Use "cpu" for CPU latency.')
    parser.add_argument('--speed-batch',  default=1,   type=int,
                        help='Batch size for latency benchmark (default: 1).')
    parser.add_argument('--speed-warmup', default=30,  type=int)
    parser.add_argument('--speed-iters',  default=200, type=int)

    # Distributed training
    parser.add_argument('--world-size', default=1, type=int,
                        help='Total number of processes (set automatically by torchrun).')
    parser.add_argument('--dist-url',   default='env://', type=str,
                        help='URL for distributed init (default: env://, works with torchrun).')
    parser.add_argument('--dist-eval',  action='store_true', default=False,
                        help='Use DistributedSampler for validation (default: off — '
                             'rank-0 evaluates the full val set, which is simpler '
                             'and avoids padding artefacts).')

    # Misc
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed',   default=42, type=int)
    parser.add_argument('--amp',    action='store_true', default=False,
                        help='Use automatic mixed precision (default: off)')
    parser.add_argument('--no-amp', action='store_false', dest='amp')

    return parser.parse_args()


def main():
    args = get_args()

    # ── Distributed init ──────────────────────────────────────────────────
    # Reads RANK / WORLD_SIZE / LOCAL_RANK from env (set by torchrun).
    # Falls back to single-GPU when those vars are absent.
    utils.init_distributed_mode(args)
    num_tasks    = utils.get_world_size()   # total GPUs
    global_rank  = utils.get_rank()         # this process' rank

    # ── Device & reproducibility ───────────────────────────────────────────
    if args.distributed:
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Per-rank seed so each GPU gets different random augmentations
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        # Deterministic / strict numeric settings (helps before-vs-after
        # reparameterization comparisons be more stable on CUDA).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir   = os.path.join(args.data_dir, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f'Expected sub-folders "train/" and "val/" inside {args.data_dir}.\n'
            f'Please organise your data as:\n'
            f'  {args.data_dir}/train/cat/  {args.data_dir}/train/dog/\n'
            f'  {args.data_dir}/val/cat/    {args.data_dir}/val/dog/'
        )

    train_dataset = AlbumentationsDataset(train_dir, transform=build_transforms(args.img_size, True))
    val_dataset   = AlbumentationsDataset(val_dir,   transform=build_transforms(args.img_size, False))

    # Auto-detect number of classes from folder structure
    detected_classes = len(train_dataset.classes)
    if detected_classes != args.num_classes:
        print_main(
            f'[WARNING] --num-classes={args.num_classes} but detected '
            f'{detected_classes} classes: {train_dataset.classes}. '
            f'Using {detected_classes}.'
        )
        args.num_classes = detected_classes

    print_main(f'Classes : {train_dataset.classes}  ({args.num_classes} total)')
    print_main(f'Train   : {len(train_dataset):,} images')
    print_main(f'Val     : {len(val_dataset):,} images')
    print_main(f'GPUs    : {num_tasks}  (global_rank={global_rank})')

    # ── Samplers — DistributedSampler for train; optional for val ─────────
    if args.distributed:
        sampler_train = DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True,
        )
        sampler_val = (
            DistributedSampler(val_dataset, num_replicas=num_tasks,
                               rank=global_rank, shuffle=False)
            if args.dist_eval
            else None          # rank-0 evaluates the full val set
        )
    else:
        sampler_train = None   # DataLoader will use default random shuffle
        sampler_val   = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),   # shuffle only when no sampler
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size * 1.5),
        shuffle=False,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print_main(f'\nCreating model: {args.model}  (num_classes={args.num_classes})')
    net = create_model(args.model, num_classes=args.num_classes, pretrained=False)

    if args.pretrained_ckpt:
        print_main(f'Loading pretrained weights from {args.pretrained_ckpt}')
        ckpt  = torch.load(args.pretrained_ckpt, map_location='cpu')
        state = ckpt.get('model', ckpt)
        # Drop the classification head so it gets re-initialised
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        msg   = net.load_state_dict(state, strict=False)
        print_main(f'  load_state_dict: {msg}')

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print_main(f'Parameters: {n_params / 1e6:.3f} M')
    net = net.to(device)

    # ── Wrap with DDP ─────────────────────────────────────────────────────
    net_without_ddp = net          # keep a reference to the unwrapped model
    if args.distributed:
        net = DDP(net, device_ids=[args.gpu])
        net_without_ddp = net.module

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Optimizer with layer-wise LR decay (LLRD) ─────────────────────────
    # Build from net_without_ddp so parameter names match the underlying model
    # (DDP wraps names with "module." prefix which confuses the LLRD logic).
    all_param_groups = get_params_groups_with_decay(
        net_without_ddp,
        lr_decay_rate=args.layerwise_decay,
        patch_embed_lr_mult=args.patch_embed_lr_mult,
    )
    fused_groups = list(fuse_params_groups(all_param_groups))
    optimizer = optim.AdamW(
        fused_groups,
        lr=args.lr,          # overridden per-iteration by apply_optim_scheduler
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),   # beta2=0.95: standard for ViT (0.999 delays 2nd-moment warm-up)
        eps=1e-8,
    )

    # ── Per-iteration LR & WD cosine schedules (DINOv3-style) ────────────
    iters_per_epoch = len(train_loader)
    total_iters     = args.epochs * iters_per_epoch
    warmup_iters    = args.warmup_epochs * iters_per_epoch

    # Apply batch-size scaling to the base LR.
    # With DDP, total_batch_size = per_gpu_batch * num_gpus.
    import math
    total_batch_size = args.batch_size * num_tasks
    if args.lr_scaling == 'sqrt_wrt_1024':
        scaled_lr = args.lr * 4.0 * math.sqrt(total_batch_size / 1024.0)
    elif args.lr_scaling == 'linear_wrt_256':
        scaled_lr = args.lr * total_batch_size / 256.0
    else:
        scaled_lr = args.lr
    print_main(
        f'LR scaling ({args.lr_scaling}): {args.lr:.2e} → {scaled_lr:.2e} '
        f'(total_bs={total_batch_size}  gpus={num_tasks})'
    )

    lr_schedule = CosineScheduler(
        base_value=scaled_lr,
        final_value=args.min_lr,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
        start_warmup_value=0.0,
    )
    wd_schedule = CosineScheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        total_iters=total_iters,
    )

    print_main(
        f'Schedule: iters_per_epoch={iters_per_epoch}  '
        f'total_iters={total_iters}  warmup_iters={warmup_iters}\n'
        f'  LR: {scaled_lr:.2e} → {args.min_lr:.2e}  '
        f'WD: {args.weight_decay} → {args.weight_decay_end}'
    )
    if is_main():
        print('Layer-wise LR decay (LLRD):')
        _seen = set()
        for pg in fused_groups:
            mult = pg['lr_multiplier']
            eff  = scaled_lr * mult
            if mult not in _seen:
                print(f'  lr_multiplier={mult:.4f}  effective_lr={eff:.2e}')
                _seen.add(mult)

    # init_scale=2**10: avoids fp16 overflow on small models
    # (PyTorch default 2**16 is too aggressive for small-scale training)
    scaler = (
        torch.amp.GradScaler('cuda', init_scale=2**10, growth_interval=2000)
        if (args.amp and device.type == 'cuda') else None
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_acc     = 0.0
    output_dir   = Path(args.output_dir)
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        print_main(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        net_without_ddp.load_state_dict(ckpt['model'])
        # Optimizer state can be incompatible when param-group logic changed
        # between old and new train.py versions (e.g., different LLRD groups).
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except ValueError as e:
                print_main(
                    f'[WARN] Could not load optimizer state from checkpoint: {e}\n'
                    '       Continuing with freshly initialized optimizer.'
                )
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt.get('best_acc', 0.0)
        if scaler is not None and 'scaler' in ckpt:
            try:
                scaler.load_state_dict(ckpt['scaler'])
            except Exception as e:
                print_main(
                    f'[WARN] Could not load AMP scaler state: {e}\n'
                    '       Continuing with a fresh GradScaler.'
                )
        print_main(f'  Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%')

    # ── Eval-only (reparam compare without training) ──────────────────────
    if args.eval_only:
        if not args.resume:
            raise ValueError('--eval-only requires --resume /path/to/checkpoint.pth')
        if is_main():
            print('\n[eval-only] Skipping training.')
            eval_reparam_compare(
                net_without_ddp, val_loader, criterion, device,
                img_size=args.img_size,
                fuse_qkv=args.fuse_qkv,
                speed_device=args.speed_device,
                speed_batch=args.speed_batch,
                speed_warmup=args.speed_warmup,
                speed_iters=args.speed_iters,
            )
        return

    # ── Training loop ─────────────────────────────────────────────────────
    print_main(f'\nTraining for {args.epochs} epochs  |  device={device}  |  GPUs={num_tasks}')
    print_main('─' * 60)
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        ep_t0 = time.time()

        # Tell the DistributedSampler which epoch we're on so each GPU
        # gets a different random permutation of the training data.
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)

        sched_lr = lr_schedule[epoch * iters_per_epoch]
        sched_wd = wd_schedule[epoch * iters_per_epoch]
        print_main(f'\nEpoch {epoch+1}/{args.epochs}  base_lr={sched_lr:.2e}  wd={sched_wd:.4f}')

        start_iter  = epoch * iters_per_epoch
        train_stats = train_one_epoch(
            net, train_loader, criterion, optimizer, scaler,
            device, epoch + 1,
            clip_grad=args.clip_grad,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            start_iteration=start_iter,
        )

        # For distributed eval: if not using dist_eval only rank-0 evaluates
        if not args.dist_eval and args.distributed:
            if is_main():
                val_stats = evaluate(
                    net_without_ddp, val_loader, criterion, device, sync_ddp=False
                )
            else:
                val_stats = {'loss': 0.0, 'acc': 0.0}
        else:
            val_stats = evaluate(net, val_loader, criterion, device)

        # Abort if training has fully diverged (unrecoverable NaN)
        if not torch.isfinite(torch.tensor(train_stats['loss'])):
            print_main('\n[ERROR] Training diverged (NaN loss). Stopping early.')
            print_main('Suggestions:')
            print_main(f'  1. Lower --lr  (current: {args.lr:.0e})  → try {args.lr/3:.0e}')
            print_main(f'  2. Lower --clip-grad  (current: {args.clip_grad})  → try 0.5')
            print_main(f'  3. Increase --warmup-epochs  (current: {args.warmup_epochs})')
            break

        is_best = val_stats['acc'] > best_acc
        if is_best:
            best_acc = val_stats['acc']

        epoch_time = time.time() - ep_t0
        eta        = (args.epochs - epoch - 1) * epoch_time
        print_main(
            f'  train  loss={train_stats["loss"]:.4f}  acc={train_stats["acc"]:.2f}%\n'
            f'  val    loss={val_stats["loss"]:.4f}  acc={val_stats["acc"]:.2f}%  '
            f'(best={best_acc:.2f}%)\n'
            f'  time={format_time(epoch_time)}  ETA={format_time(eta)}'
        )

        # ── Checkpoint saving (rank-0 only via save_on_master) ───────────
        _ckpt = {
            'epoch':     epoch,
            'model':     net_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler':    scaler.state_dict() if scaler else None,
            'best_acc':  best_acc,
            'classes':   train_dataset.classes,
            'args':      vars(args),
        }
        if is_best:
            best_path = output_dir / 'checkpoint_best.pth'
            utils.save_on_master(_ckpt, best_path)
            print_main(f'  ** Saved best checkpoint -> {best_path}')

        if args.save_freq > 0 and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            ckpt_path = output_dir / f'checkpoint_{epoch}.pth'
            utils.save_on_master(_ckpt, ckpt_path)
            # Prune previous periodic checkpoint
            prev_path = output_dir / f'checkpoint_{epoch - args.save_freq}.pth'
            if is_main() and prev_path.exists():
                prev_path.unlink()
            print_main(f'  Saved periodic checkpoint -> {ckpt_path}')

    total_time = time.time() - t_start
    print_main(f'\nTraining complete in {format_time(total_time)}')
    print_main(f'Best val accuracy: {best_acc:.2f}%')
    print_main(f'Best checkpoint  : {output_dir / "checkpoint_best.pth"}')

    # ── Post-training reparameterization compare (rank-0 only) ────────────
    if args.eval_reparam and is_main():
        best_ckpt_path = output_dir / 'checkpoint_best.pth'
        if best_ckpt_path.exists():
            print_main(f'\nLoading best checkpoint for reparam eval: {best_ckpt_path}')
            best_ckpt = torch.load(best_ckpt_path, map_location='cpu')
            net_without_ddp.load_state_dict(best_ckpt['model'])
        eval_reparam_compare(
            net_without_ddp, val_loader, criterion, device,
            img_size=args.img_size,
            fuse_qkv=args.fuse_qkv,
            speed_device=args.speed_device,
            speed_batch=args.speed_batch,
            speed_warmup=args.speed_warmup,
            speed_iters=args.speed_iters,
        )


if __name__ == '__main__':
    main()
