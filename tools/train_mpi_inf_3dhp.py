# ------------------------------------------------------------------------------
# HRNet training entrypoint for MPI-INF-3DHP
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.function import train
from core.function import validate
from core.loss import JointsMSELoss
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint

import dataset
import models

try:
    wandb = __import__('wandb')
except ImportError:
    wandb = None


def _extract_mpjpe2d(name_values):
    if isinstance(name_values, list):
        for item in name_values:
            if isinstance(item, dict) and 'MPJPE2D' in item:
                return float(item['MPJPE2D'])
    elif isinstance(name_values, dict) and 'MPJPE2D' in name_values:
        return float(name_values['MPJPE2D'])
    return None


def _extract_pck005(name_values):
    if isinstance(name_values, list):
        for item in name_values:
            if isinstance(item, dict) and 'PCK@0.05' in item:
                return float(item['PCK@0.05'])
    elif isinstance(name_values, dict) and 'PCK@0.05' in name_values:
        return float(name_values['PCK@0.05'])
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Train HRNet on MPI-INF-3DHP')
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        type=str,
        default='experiments/3DHP/hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp.yaml'
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev model directory', type=str, default='')
    parser.add_argument('--use-wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb-project', type=str, default='HRNet_MPI_INF_3DHP', help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default='hrnet_w48_mpi_inf_3dhp', help='WandB run name')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    use_wandb = args.use_wandb and (wandb is not None)
    if args.use_wandb and wandb is None:
        logger.warning('WandB requested but package is not installed. Continuing without WandB logging.')

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'cfg_file': args.cfg,
                'model': cfg.MODEL.NAME,
                'num_joints': cfg.MODEL.NUM_JOINTS,
                'epochs': cfg.TRAIN.END_EPOCH,
                'batch_size_per_gpu': cfg.TRAIN.BATCH_SIZE_PER_GPU,
                'gpus': list(cfg.GPUS),
                'lr': cfg.TRAIN.LR,
                'dataset': cfg.DATASET.DATASET,
                'train_annotation_file': cfg.DATASET.TRAIN_ANNOTATION_FILE,
                'test_annotation_file': cfg.DATASET.TEST_ANNOTATION_FILE,
                'train_image_root': cfg.DATASET.TRAIN_IMAGE_ROOT,
                'test_image_root': cfg.DATASET.TEST_IMAGE_ROOT,
            },
            sync_tensorboard=True,
        )
        logger.info('WandB enabled: project=%s run=%s', args.wandb_project, args.wandb_run_name)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=True)

    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir
    )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model, (dump_input,))
    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    best_perf = float('inf')
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        perf_name = checkpoint.get('perf_name', '')
        if perf_name == 'MPJPE2D':
            best_perf = checkpoint['perf']
        else:
            logger.warning('Checkpoint perf_name is "%s"; resetting best_perf for MPJPE2D selection.', perf_name)
            best_perf = float('inf')
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        train(
            cfg,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            final_output_dir,
            tb_log_dir,
            writer_dict,
        )

        perf_indicator, name_values = validate(
            cfg,
            valid_loader,
            valid_dataset,
            model,
            criterion,
            final_output_dir,
            tb_log_dir,
            writer_dict,
            return_metrics=True,
        )

        mpjpe2d = _extract_mpjpe2d(name_values)
        pck005 = _extract_pck005(name_values)
        selection_perf = mpjpe2d if mpjpe2d is not None else perf_indicator

        logger.info(
            'Epoch %d validation metrics: PCK@0.05=%s MPJPE2D=%s selection(MPJPE2D)=%s best=%s',
            epoch + 1,
            '{:.6f}'.format(pck005) if pck005 is not None else 'N/A',
            '{:.6f}'.format(mpjpe2d) if mpjpe2d is not None else 'N/A',
            '{:.6f}'.format(selection_perf),
            '{:.6f}'.format(best_perf),
        )

        if selection_perf <= best_perf:
            best_perf = selection_perf
            best_model = True
        else:
            best_model = False

        if use_wandb:
            wandb.log(
                {
                    'epoch': epoch + 1,
                    'valid/perf_indicator': float(perf_indicator),
                    'valid/selection_perf_mpjpe2d': float(selection_perf),
                    'valid/best_perf': float(best_perf),
                    'train/lr': float(optimizer.param_groups[0]['lr']),
                },
                step=epoch + 1,
            )

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': selection_perf,
                'perf_name': 'MPJPE2D',
                'optimizer': optimizer.state_dict(),
            },
            best_model,
            final_output_dir,
        )

        lr_scheduler.step()

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

    if use_wandb:
        wandb.summary['final_model_state_file'] = final_model_state_file
        wandb.summary['best_perf'] = float(best_perf)
        wandb.finish()


if __name__ == '__main__':
    main()
