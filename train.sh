NCCL_P2P_DISABLE=1 torchrun \
    --nproc_per_node=2 \
    --master_port 12346 \
    main.py \
    --model swift_net_tiny \
    --data-path /kaggle/input/datasets/mayurmadnani/imagenet-dataset \
    --batch-size 256 \
    --epochs 300 \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --warmup-epochs 10 \
    --weight-decay 0.05 \
    --layer-decay 0.9 \
    --patch-embed-lr-mult 0.2 \
    --freeze-last-layer-epochs 1 \
    --clip-grad 3.0 \
    --clip-mode norm \
    --output_dir checkpoints \
    --dist-eval \
    --num_workers 2 \
    --no_wandb
    # --no-amp  # uncomment to disable mixed precision (fp32)
