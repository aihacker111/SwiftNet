"""
Đếm Parameters và GFLOPs của model.
Chạy: python count_params.py [--model swift_net_tiny]
"""
import argparse
import torch
import model  # noqa: F401
from timm.models import create_model
from fvcore.nn import FlopCountAnalysis, parameter_count


def count(model_name: str, resolution: int = 224):
    net = create_model(model_name, num_classes=1000)
    net.eval()

    dummy = torch.randn(1, 3, resolution, resolution)

    params   = sum(p.numel() for p in net.parameters()) / 1e6
    flops    = FlopCountAnalysis(net, dummy)
    gflops   = flops.total() / 1e9

    print(f"\n{'='*40}")
    print(f"  Model      : {model_name}")
    print(f"  Resolution : {resolution}x{resolution}")
    print(f"{'='*40}")
    print(f"  Params     : {params:.2f} M")
    print(f"  GFLOPs     : {gflops:.2f} G")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="swift_net_base", type=str)
    parser.add_argument("--resolution", default=224, type=int)
    args = parser.parse_args()

    count(args.model, args.resolution)
