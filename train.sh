# NCCL_P2P_DISABLE=1 torchrun \
#     --nproc_per_node=2 \
#     --master_port 12346 \
#     main.py \
#     --model swift_net_tiny \
#     --data-path /kaggle/input/datasets/mayurmadnani/imagenet-dataset \
#     --batch-size 64 \
#     --epochs 300 \
#     --lr 1e-3 \
#     --min-lr 1e-6 \
#     --warmup-epochs 10 \
#     --weight-decay 0.05 \
#     --layer-decay 0.9 \
#     --patch-embed-lr-mult 0.2 \
#     --freeze-last-layer-epochs 1 \
#     --clip-grad 3.0 \
#     --clip-mode norm \
#     --output_dir checkpoints \
#     --dist-eval \
#     --num_workers 4 \
#     --no_wandb
#     # --no-amp  # uncomment to disable mixed precision (fp32)

# --nproc_per_node=1 \
# --master_port 12346 \
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=3 python \
    main.py \
    --model swift_net_tiny \
    --data-path /kaggle/input/datasets/mayurmadnani/imagenet-dataset \
    --batch-size 256 \
    --epochs 300 \
    --lr 4e-4 \
    --lr-scaling none \
    --min-lr 1e-6 \
    --warmup-epochs 5 \
    --weight-decay 0.025 \
    --smoothing 0.1 \
    --aa rand-m1-mstd0.5-inc1 \
    --color-jitter 0.0 \
    --mixup 0.8 \
    --cutmix 0.2 \
    --mixup-prob 1.0 \
    --mixup-switch-prob 0.5 \
    --mixup-mode batch \
    --reprob 0.25 \
    --remode pixel \
    --recount 1 \
    --layer-decay 0.9 \
    --patch-embed-lr-mult 0.2 \
    --freeze-last-layer-epochs 1 \
    --clip-grad 3.0 \
    --clip-mode norm \
    --model-ema \
    --model-ema-decay 0.99996 \
    --distillation-type hard \
    --distillation-alpha 0.5 \
    --distillation-tau 1.0 \
    --teacher-model regnety_160 \
    --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth \
    --output_dir checkpoints \
    --dist-eval \
    --num_workers 4 \
    --no_wandb