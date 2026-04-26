"""
export_onnx.py — Export SWIFTNet to ONNX.

Usage:
    # From checkpoint
    python export_onnx.py --checkpoint checkpoints/checkpoint_best.pth --output swift_net_tiny.onnx

    # Without checkpoint (random weights, for architecture check)
    python export_onnx.py --output swift_net_tiny.onnx

    # Different model / image size
    python export_onnx.py --model swift_net_small --img-size 224 --checkpoint best.pth

    # Verify exported model
    python export_onnx.py --checkpoint best.pth --verify
"""

import argparse
import torch
import timm
import model  # registers swift_net_* models


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="swift_net_tiny", type=str)
    p.add_argument("--checkpoint",  default="",  type=str, help="Path to .pth checkpoint")
    p.add_argument("--output",      default="swift_net.onnx", type=str)
    p.add_argument("--num-classes", default=1000, type=int)
    p.add_argument("--img-size",    default=224,  type=int)
    p.add_argument("--batch-size",  default=1,    type=int)
    p.add_argument("--opset",       default=17,   type=int)
    p.add_argument("--verify",      action="store_true", help="Verify ONNX output vs PyTorch")
    return p.parse_args()


def load_model(args):
    net = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # Support both raw state_dict and wrapped checkpoints
        state = ckpt.get("model", ckpt)
        msg = net.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}  {msg}")
    else:
        print("No checkpoint provided — using random weights")

    net.eval()
    return net


def export(net, args):
    dummy = torch.randn(args.batch_size, 3, args.img_size, args.img_size)

    torch.onnx.export(
        net,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images":  {0: "batch"},
            "logits":  {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"Exported → {args.output}  (opset {args.opset})")


def verify(net, args):
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("onnxruntime not installed — skipping verify (pip install onnxruntime)")
        return

    dummy = torch.randn(args.batch_size, 3, args.img_size, args.img_size)
    with torch.no_grad():
        pt_out = net(dummy).numpy()

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logits"], {"images": dummy.numpy()})[0]

    max_diff = abs(pt_out - ort_out).max()
    print(f"Max output diff PyTorch vs ONNX: {max_diff:.6f}", end="  ")
    print("✓ OK" if max_diff < 1e-4 else "⚠ Large diff — check model")


def main():
    args = get_args()
    net  = load_model(args)

    print(f"Model: {args.model}  |  "
          f"Params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M  |  "
          f"Input: {args.batch_size}×3×{args.img_size}×{args.img_size}")

    export(net, args)

    if args.verify:
        verify(net, args)


if __name__ == "__main__":
    main()
