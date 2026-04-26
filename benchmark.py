"""
Đo latency (ms) và throughput (images/s) của swift_net_tiny.
Chạy: python benchmark.py [--batch-size 1] [--device cuda] [--checkpoint path/to/ckpt.pth]
"""
import argparse
import torch
import model  # noqa: F401  — đăng ký các model vào timm registry
from timm.models import create_model

WARMUP_STEPS = 50
MEASURE_STEPS = 200


def load_model(model_name: str, checkpoint: str | None, device: str):
    net = create_model(model_name, num_classes=1000)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # hỗ trợ cả raw state_dict lẫn checkpoint có key 'model'
        state_dict = ckpt.get("model", ckpt)
        msg = net.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint: {checkpoint}")
        print(f"  {msg}")
    return net.to(device).eval()


@torch.no_grad()
def benchmark(model_name: str, batch_size: int, device: str, resolution: int = 224,
              checkpoint: str | None = None):
    dummy = torch.randn(batch_size, 3, resolution, resolution, device=device)

    net = load_model(model_name, checkpoint, device)

    n_params = sum(p.numel() for p in net.parameters()) / 1e6

    # warmup
    for _ in range(WARMUP_STEPS):
        net(dummy)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start_event = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_STEPS)]
        end_event   = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_STEPS)]

        for i in range(MEASURE_STEPS):
            start_event[i].record()
            net(dummy)
            end_event[i].record()

        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    else:
        import time
        times_ms = []
        for _ in range(MEASURE_STEPS):
            t0 = time.perf_counter()
            net(dummy)
            times_ms.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times_ms)
    latency_mean = t.mean().item()
    latency_std  = t.std().item()
    latency_p99  = t.quantile(0.99).item()
    throughput   = batch_size / (latency_mean / 1000)

    print(f"\n{'='*45}")
    print(f"  Model      : {model_name}")
    print(f"  Params     : {n_params:.1f}M")
    print(f"  Device     : {device}")
    print(f"  Batch size : {batch_size}")
    print(f"  Resolution : {resolution}x{resolution}")
    print(f"{'='*45}")
    print(f"  Latency mean : {latency_mean:.2f} ms")
    print(f"  Latency std  : {latency_std:.2f} ms")
    print(f"  Latency p99  : {latency_p99:.2f} ms")
    print(f"  Throughput   : {throughput:.1f} images/s")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="swift_net_tiny", type=str)
    parser.add_argument("--batch-size", default=1,   type=int)
    parser.add_argument("--device",     default="cuda", type=str)
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument("--checkpoint", default=None, type=str, help="path to .pth checkpoint")
    args = parser.parse_args()

    benchmark(args.model, args.batch_size, args.device, args.resolution, args.checkpoint)
