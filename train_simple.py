"""
train_simple.py — Minimal training script with synthetic data.

Usage:
    python train_simple.py
    python train_simple.py --model swift_net_tiny --epochs 5 --batch_size 16
    python train_simple.py --amp  # mixed precision (fp16)
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import model  # registers timm models
import timm


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticImageDataset(Dataset):
    """Random RGB images with integer class labels."""

    def __init__(self, num_samples: int = 1024, num_classes: int = 10,
                 img_size: int = 224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        # Pre-generate fixed data so each epoch is identical
        torch.manual_seed(42)
        self.images = torch.randn(num_samples, 3, img_size, img_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="swift_net_tiny", type=str)
    p.add_argument("--num_classes",  default=10,   type=int)
    p.add_argument("--img_size",     default=224,  type=int)
    p.add_argument("--train_size",   default=1024, type=int)
    p.add_argument("--val_size",     default=256,  type=int)
    p.add_argument("--batch_size",   default=16,   type=int)
    p.add_argument("--epochs",       default=20,    type=int)
    p.add_argument("--lr",           default=1e-3, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--amp",          action="store_true", default=False,
                   help="Use mixed precision (fp16). NOTE: requires power-of-2 "
                        "spatial dims for cuFFT; the fix in ssm.py handles this.")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers",  default=2, type=int)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Device: {device}  |  AMP: {args.amp}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = SyntheticImageDataset(args.train_size, args.num_classes, args.img_size)
    val_ds   = SyntheticImageDataset(args.val_size,   args.num_classes, args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────
    net = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Params: {n_params/1e6:.2f}M")

    # ── Optimizer / loss / scaler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            net, train_loader, criterion, optimizer, scaler, device, args.amp
        )
        val_stats = evaluate(net, val_loader, criterion, device, args.amp)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={train_stats['loss']:.4f}  train_acc={train_stats['acc']:.4f}  "
            f"val_loss={val_stats['loss']:.4f}  val_acc={val_stats['acc']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.1f}s"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save(net.state_dict(), "best_model.pth")

    print(f"\nTraining done. Best val acc: {best_val_acc:.4f}")
    print("Checkpoint saved to best_model.pth")


if __name__ == "__main__":
    main()
