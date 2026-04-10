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

# Optional FLOP-counter backends (install one for GFLOPs support)
try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE = True
except ImportError:
    _FVCORE = False

try:
    from thop import profile as thop_profile
    _THOP = True
except ImportError:
    _THOP = False

from swiftnet import swift_net_tiny, swift_net_small, swift_net_base, SWIFTNet
from swiftnet.config import SWIFTNetConfig


# ---------------------------------------------------------------------------
# Helper: compute GFLOPs (fvcore preferred, thop fallback)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_gflops(
    model: nn.Module,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> float | None:
    """
    Return GFLOPs for one forward pass.

    Tries fvcore first, then thop. Returns None if neither is installed.
    Batch dimension is forced to 1 — GFLOPs are per-image.

    Install one of:
        pip install fvcore        # recommended
        pip install thop          # lightweight alternative
    """
    b, c, h, w = input_size
    x = torch.randn(1, c, h, w)   # always batch=1 for per-image cost
    model = model.cpu().eval()

    if _FVCORE:
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9  # → GFLOPs

    if _THOP:
        macs, _ = thop_profile(model, inputs=(x,), verbose=False)
        return macs / 1e9  # MACs ≈ FLOPs/2; report as GFLOPs (MACs)

    return None


# ---------------------------------------------------------------------------
# Helper: pretty print param counts
# ---------------------------------------------------------------------------

def print_model_summary(
    model: SWIFTNet,
    name: str,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> None:
    counts = model.count_parameters()
    total  = counts["total"]
    gflops = compute_gflops(model, input_size)
    gflops_str = f"{gflops:.2f} GFLOPs" if gflops is not None else "n/a (install fvcore or thop)"
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Total params:     {total:>12,}  ({total/1e6:.2f}M)")
    print(f"  Trainable params: {counts['trainable']:>12,}")
    print(f"  GFLOPs @ {input_size[2]}×{input_size[3]}:  {gflops_str}")
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
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = (1, 3, 224, 224)

    for name, factory in [
        ("Tiny  (3M)", swift_net_tiny),
        ("Small (8M)", swift_net_small),
    ]:
        model   = factory(num_classes=1000)
        gflops  = compute_gflops(model, input_size)
        results = benchmark(model, input_size=input_size, device=device)
        gflops_str = f"{gflops:.2f}G" if gflops is not None else "n/a"
        print(
            f"  {name}: {results['latency_ms']}ms/img  |  "
            f"{results['fps']} FPS  |  {gflops_str} FLOPs  [{device}]"
        )

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
# Test 7: GFLOPs at multiple input resolutions
# ---------------------------------------------------------------------------

def test_gflops():
    print("\n[TEST 7] GFLOPs")

    if not _FVCORE and not _THOP:
        print("  SKIP — install fvcore or thop:  pip install fvcore")
        return

    resolutions = [(224, 224), (256, 256), (384, 384)]

    for model_name, factory in [
        ("swift_net_tiny",  swift_net_tiny),
        ("swift_net_small", swift_net_small),
        ("swift_net_base",  swift_net_base),
    ]:
        print(f"\n  {model_name}")
        model = factory(num_classes=1000)
        for h, w in resolutions:
            gflops = compute_gflops(model, input_size=(1, 3, h, w))
            print(f"    {h}×{w}: {gflops:.3f} GFLOPs")

    print("\n[PASS] GFLOPs OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("SWIFT-Net Test Suite")
    print("=" * 55)

    test_instantiation()
    test_forward_pass()
    test_variable_input_sizes()
    test_gradient_flow()
    test_throughput()
    test_custom_config()
    test_gflops()
    print("\n" + "=" * 55)
    print("  Tất cả tests PASSED!")
    print("=" * 55)