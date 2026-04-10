"""
train_example.py — Ví dụ sử dụng SWIFT-Net
===========================================
Chạy: python train_example.py
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from swiftnet import swift_net_tiny, swift_net_small, swift_net_base, SWIFTNet
from swiftnet.config import SWIFTNetConfig


# ---------------------------------------------------------------------------
# Helper: pretty print param counts
# ---------------------------------------------------------------------------

def print_model_summary(model: SWIFTNet, name: str) -> None:
    counts = model.count_parameters()
    total  = counts["total"]
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Total params:     {total:>12,}  ({total/1e6:.2f}M)")
    print(f"  Trainable params: {counts['trainable']:>12,}")
    print(f"  Patch embed:      {counts['patch_embed']:>12,}")
    for key, val in counts.items():
        if key.startswith("stage_"):
            print(f"  {key}:         {val:>12,}")
    print(f"  Head:             {counts['head']:>12,}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Helper: benchmark throughput
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark(
    model: nn.Module,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval().to(device)
    x = torch.randn(*input_size, device=device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(x)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms
    fps     = 1000.0 / elapsed * input_size[0]            # images/sec

    return {"latency_ms": round(elapsed, 2), "fps": round(fps, 1)}


# ---------------------------------------------------------------------------
# Test 1: Model instantiation và summary
# ---------------------------------------------------------------------------

def test_instantiation():
    print("\n[TEST 1] Model instantiation")

    for name, factory in [
        ("swift_net_tiny",  swift_net_tiny),
        ("swift_net_small", swift_net_small),
        ("swift_net_base",  swift_net_base),
    ]:
        model = factory(num_classes=1000)
        print_model_summary(model, name)

    print("\n[PASS] Tất cả models khởi tạo thành công")


# ---------------------------------------------------------------------------
# Test 2: Forward pass với input 224×224
# ---------------------------------------------------------------------------

def test_forward_pass():
    print("\n[TEST 2] Forward pass")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = swift_net_tiny(num_classes=10).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)

    # Classification output
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 10), f"Expected (2,10), got {logits.shape}"
    print(f"  Classification output: {logits.shape}  [OK]")

    # Feature maps (cho dense prediction)
    with torch.no_grad():
        feature_maps = model.get_feature_maps(x)
    print("  Feature map shapes (per stage):")
    for i, fm in enumerate(feature_maps):
        print(f"    Stage {i}: {tuple(fm.shape)}")

    print("\n[PASS] Forward pass OK")


# ---------------------------------------------------------------------------
# Test 3: Forward với nhiều kích thước input
# ---------------------------------------------------------------------------

def test_variable_input_sizes():
    print("\n[TEST 3] Variable input sizes")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = swift_net_small(num_classes=1000).to(device)
    model.eval()

    for h, w in [(224, 224), (256, 256), (384, 384), (128, 128)]:
        x = torch.randn(1, 3, h, w, device=device)
        with torch.no_grad():
            out = model(x)
        print(f"  Input ({h}×{w}) → output {tuple(out.shape)}  [OK]")

    print("\n[PASS] Variable input sizes OK")


# ---------------------------------------------------------------------------
# Test 4: Gradient flow (training step)
# ---------------------------------------------------------------------------

def test_gradient_flow():
    print("\n[TEST 4] Gradient flow")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model     = swift_net_tiny(num_classes=10).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    model.train()
    x = torch.randn(4, 3, 224, 224, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    # Forward + backward
    optimizer.zero_grad()
    logits = model(x)
    loss   = criterion(logits, y)
    loss.backward()
    optimizer.step()

    # Kiểm tra grad
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Params với grad: {len(grad_norms)}")
    print(f"  Grad norm (mean): {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"  Grad norm (max):  {max(grad_norms):.6f}")
    print("\n[PASS] Gradient flow OK")


# ---------------------------------------------------------------------------
# Test 5: Throughput benchmark
# ---------------------------------------------------------------------------

def test_throughput():
    print("\n[TEST 5] Throughput benchmark")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, factory in [
        ("Tiny  (3M)", swift_net_tiny),
        ("Small (8M)", swift_net_small),
    ]:
        model   = factory(num_classes=1000)
        results = benchmark(model, input_size=(1, 3, 224, 224), device=device)
        print(f"  {name}: {results['latency_ms']}ms/img  |  {results['fps']} FPS  [{device}]")

    print("\n[PASS] Throughput benchmark OK")


# ---------------------------------------------------------------------------
# Test 6: Custom config
# ---------------------------------------------------------------------------

def test_custom_config():
    print("\n[TEST 6] Custom config")

    # Ví dụ: model cho classification 5 classes, rất nhẹ
    config = SWIFTNetConfig(
        dims=[16, 32, 64, 128],
        depths=[1, 1, 2, 1],
        num_heads=[1, 1, 2, 4],
        d_state=8,
        kd_rank=4,
        num_rff=16,
        wavelet_levels=1,
        drop_path_rate=0.0,
        num_classes=5,
    )
    model = SWIFTNet(config)
    counts = model.count_parameters()
    print(f"  Custom model: {counts['total']/1e6:.2f}M params")

    x   = torch.randn(2, 3, 112, 112)
    out = model(x)
    assert out.shape == (2, 5)
    print(f"  Output shape: {out.shape}  [OK]")
    print("\n[PASS] Custom config OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("SWIFT-Net Test Suite")
    print("=" * 55)

    try:
        test_instantiation()
        test_forward_pass()
        test_variable_input_sizes()
        test_gradient_flow()
        test_throughput()
        test_custom_config()
        print("\n" + "=" * 55)
        print("  Tất cả tests PASSED!")
        print("=" * 55)
    except Exception as e:
        import traceback
        print(f"\n[FAIL] {e}")
        traceback.print_exc()