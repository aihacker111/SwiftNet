import argparse
import datetime
import numpy as np
import time
import math
import torch
import torch.backends.cudnn as cudnn
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma

from data.samplers import RASampler
from data.datasets import build_dataset
from data.threeaugment import new_data_aug_generator
from engine import train_one_epoch, evaluate
from losses import DistillationLoss

import model
import utils


# ---------------------------------------------------------------------------
# DINOv3 LR/WD schedule  (dinov3/dinov3/train/cosine_lr_scheduler.py)
# ---------------------------------------------------------------------------

class CosineScheduler:
    """
    Precomputed schedule: freeze → linear warmup → cosine decay.
    Indexed per step: scheduler[it].
    """
    def __init__(self, base_value, final_value, total_iters,
                 warmup_iters=0, start_warmup_value=0.0, freeze_iters=0):
        self.final_value = final_value
        self.total_iters = total_iters

        freeze   = np.zeros(freeze_iters)
        warmup   = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters    = np.arange(total_iters - warmup_iters - freeze_iters)
        cosine   = (final_value
                    + 0.5 * (base_value - final_value)
                    * (1 + np.cos(np.pi * iters / len(iters))))
        self.schedule = np.concatenate([freeze, warmup, cosine]).astype(np.float64)
        assert len(self.schedule) == total_iters

    def __getitem__(self, it):
        return float(self.schedule[it] if it < self.total_iters else self.final_value)


# ---------------------------------------------------------------------------
# DINOv3 param groups with LLRD  (dinov3/dinov3/train/param_groups.py)
# ---------------------------------------------------------------------------

def _get_swiftnet_layer_id(name: str, num_blocks: int, stage_offsets: List[int]) -> int:
    """
    patch_embed            → 0          (lowest LR)
    stages.S.B.*           → offset[S] + B + 1
    head / norm / mergers  → num_blocks + 1  (highest LR)
    """
    if "patch_embed" in name:
        return 0
    if "stages." in name:
        parts = name.split(".")
        try:
            si = parts.index("stages")
            stage_idx = int(parts[si + 1])
            block_idx = int(parts[si + 2])
            return stage_offsets[stage_idx] + block_idx + 1
        except (ValueError, IndexError):
            pass
    return num_blocks + 1


def get_params_groups(net, layer_decay=0.9, patch_embed_lr_mult=0.2, weight_decay=0.05):
    stages = getattr(net, "stages", None)
    stage_depths  = [len(s) for s in stages] if stages else []
    num_blocks    = sum(stage_depths)
    stage_offsets = []
    running = 0
    for d in stage_depths:
        stage_offsets.append(running)
        running += d

    all_params = []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _get_swiftnet_layer_id(name, num_blocks, stage_offsets)
        lr_mult  = layer_decay ** (num_blocks + 1 - layer_id)
        no_wd    = (param.ndim == 1 or "norm" in name or "gamma" in name
                    or name.endswith(("bias", "ls1", "ls2")))
        wd_mult  = 0.0 if no_wd else 1.0
        if "patch_embed" in name:
            lr_mult *= patch_embed_lr_mult
        all_params.append({
            "name": name, "params": param,
            "lr_multiplier": lr_mult, "wd_multiplier": wd_mult,
            "is_last_layer": "last_layer" in name,
        })

    fused = defaultdict(lambda: {"params": []})
    for d in all_params:
        key = f"lr{d['lr_multiplier']:.6f}_wd{d['wd_multiplier']}_ll{d['is_last_layer']}"
        fused[key]["params"].append(d["params"])
        fused[key].update({k: d[k] for k in ("lr_multiplier", "wd_multiplier", "is_last_layer")})
        fused[key].setdefault("lr", 0.0)
        fused[key].setdefault("weight_decay", 0.0)
    return list(fused.values())


def get_args_parser():
    parser = argparse.ArgumentParser(
        'RepNeXt training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='repnext_m1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument(
        '--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float,
                        default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--clip-grad', type=float, default=3.0, metavar='NORM',
                        help='Gradient clip norm (default: 3.0)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode (default: norm)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Fixed weight decay for classification (default: 0.05)')

    # DINOv3-style LR schedule
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Base LR before sqrt scaling (default: 1e-3)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='Minimum LR after cosine decay (default: 1e-6)')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='Warmup epochs (default: 10)')
    parser.add_argument('--freeze-last-layer-epochs', type=int, default=1,
                        help='Freeze last layer LR for N epochs (default: 1)')
    parser.add_argument('--layer-decay', type=float, default=0.9,
                        help='LLRD factor per layer (default: 0.9)')
    parser.add_argument('--patch-embed-lr-mult', type=float, default=0.2,
                        help='Extra LR multiplier for patch_embed (default: 0.2)')

    # Augmentation parameters
    parser.add_argument('--ThreeAugment', action='store_true')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str,
                        default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth')
    parser.add_argument('--distillation-type', default='hard',
                        choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha',
                        default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--set_bn_eval', action='store_true', default=False,
                        help='set BN layers to eval mode during finetuning.')

    # Dataset parameters
    parser.add_argument('--data-path', default='/root/FastBaseline/data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order',
                                 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--output_dir', default='checkpoints',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequency of model saving')
    
    parser.add_argument('--no-amp', action='store_true', default=False,
                        help='Disable automatic mixed precision (fp32 training)')
    parser.add_argument('--deploy', action='store_true', default=False)
    parser.add_argument('--project', default='repnext', type=str)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    return parser

import wandb

def main(args):
    
    utils.init_distributed_mode(args)

    if utils.is_main_process() and not args.eval and not args.no_wandb:
        wandb.init(project=args.project, config=args)
        wandb.run.log_code('model')
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError(
            "Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)
        
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        distillation=(args.distillation_type != 'none'),
        pretrained=False,
    )
    export_onnx(model, args.output_dir)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.finetune))
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.l.weight', 'head.l.bias',
                  'head_dist.l.weight', 'head_dist.l.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # ── DINOv3-style optimizer & schedules ───────────────────────────────
    # LR scaling: sqrt rule wrt 1024  (ssl_default_config.yaml: scaling_rule=sqrt_wrt_1024)
    total_bs = args.batch_size * utils.get_world_size()
    lr_peak  = args.lr     * 4 * math.sqrt(total_bs / 1024.0)
    lr_min   = args.min_lr * 4 * math.sqrt(total_bs / 1024.0)
    print(f"LR sqrt scaling: base={args.lr:.2e} → peak={lr_peak:.2e}, min={lr_min:.2e}")

    steps_per_epoch = len(data_loader_train)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = steps_per_epoch * args.warmup_epochs
    freeze_ll_steps = steps_per_epoch * args.freeze_last_layer_epochs

    lr_schedule = CosineScheduler(
        base_value=lr_peak, final_value=lr_min,
        total_iters=total_steps, warmup_iters=warmup_steps,
    )
    wd_schedule = CosineScheduler(
        base_value=args.weight_decay, final_value=args.weight_decay,
        total_iters=total_steps,
    )
    last_layer_lr_schedule = CosineScheduler(
        base_value=lr_peak, final_value=lr_min,
        total_iters=total_steps, warmup_iters=warmup_steps,
        freeze_iters=freeze_ll_steps,
    )

    param_groups = get_params_groups(
        model_without_ddp,
        layer_decay=args.layer_decay,
        patch_embed_lr_mult=args.patch_embed_lr_mult,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    loss_scaler = NativeScaler()
    lr_scheduler = None  # scheduling done per-step inside train_one_epoch

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is
    # 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
            print(str(model))
    if args.output_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")
            print(json.dumps(args.__dict__, indent=2) + "\n")
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    if args.eval:
        utils.replace_batchnorm(model) # Users may choose whether to merge Conv-BN layers during eval
        print(f"Evaluating model: {args.model}")
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        step_offset = (epoch - args.start_epoch) * steps_per_epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema, mixup_fn,
            set_training_mode=True,
            set_bn_eval=args.set_bn_eval,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            last_layer_lr_schedule=last_layer_lr_schedule,
            step_offset=step_offset,
            amp=not args.no_amp,
        )

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if args.output_dir:
            ckpt_path = os.path.join(output_dir, 'checkpoint_'+str(epoch)+'.pth')
            checkpoint_paths = [ckpt_path]
            print("Saving checkpoint to {}".format(ckpt_path))
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            remove_epoch = epoch - 3
            if remove_epoch >= 0 and utils.is_main_process():
                os.remove(os.path.join(output_dir, 'checkpoint_'+str(remove_epoch)+'.pth'))

        if max_accuracy < test_stats["acc1"]:
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, os.path.join(output_dir, 'checkpoint_best.pth'))
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        
        print(f'Max accuracy: {max_accuracy:.2f}%')
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        if utils.is_main_process() and not args.no_wandb:
            wandb.log({**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'max_accuracy': max_accuracy}, step=epoch)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process() and not args.no_wandb:
        wandb.finish()

def export_onnx(model, output_dir):
    # if utils.is_main_process():
    #     dummy_input = torch.randn(1, 3, 224, 224)
    #     torch.onnx.export(model, dummy_input, f"{output_dir}/model.onnx")
    #     wandb.save(f"{output_dir}/model.onnx")
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'RepNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.resume and not args.eval:
        args.output_dir = '/'.join(args.resume.split('/')[:-1])
    elif args.output_dir:
        args.output_dir = args.output_dir + f"/{args.model}/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        assert(False)
    main(args)
