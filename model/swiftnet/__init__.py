"""
SWIFT-Net: State-space Wavelet-Integrated Fast Transformer
==========================================================
Kiến trúc model nhẹ cho edge devices, kết hợp:
  - Wavelet Linear Attention + 2D RoPE  (O(n log n))
  - Kronecker-factored Depthwise Conv   (8-16× ít params hơn standard conv)
  - DPLR State Space Model              (long-range context, O(n log n))
  - SE-WHT Gate                         (không có phép nhân, INT8-friendly)

Package layout
--------------
rope.py         — 2D RoPE positional embedding (DINOv3 style)
attention.py    — WaveAttentionWithRoPE, WindowSelfAttentionWithRoPE
convolution.py  — KroneckerDepthwiseConv (KD-Conv)
ssm.py          — DiagonalPlusLowRankSSM (DPLR-SSM)
fusion.py       — GMSFusion, SEWHTGate
block.py        — SWIFTBlock (lắp ráp tất cả modules)
model.py        — SWIFTNet backbone (full model)
"""