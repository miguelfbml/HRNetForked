# ------------------------------------------------------------------------------
# HRNet test entrypoint with YOLO-style 2D metrics for MPI-INF-3DHP
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pprint
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test HRNet on MPI-INF-3DHP with MPJPE/PCK/AUC metrics'
    )
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
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

    parser.add_argument('--fixed-pck-threshold', type=float, default=150.0,
                        help='Fixed pixel threshold for PCK metric')
    parser.add_argument('--auc-max-threshold', type=float, default=150.0,
                        help='Maximum threshold for AUC integration')
    parser.add_argument('--auc-num-thresholds', type=int, default=50,
                        help='Number of thresholds for AUC curve')
    parser.add_argument('--root-joint-idx', type=int, default=14,
                        help='Root joint index for root-relative evaluation')
    parser.add_argument('--save-json', action='store_true',
                        help='Save metrics JSON in final output dir')

    return parser.parse_args()


def calculate_torso_diameter_2d(poses_2d):
    if len(poses_2d.shape) != 3 or poses_2d.shape[1] != 17:
        return np.ones(poses_2d.shape[0], dtype=np.float32) * 100.0

    left_shoulder = poses_2d[:, 2, :]
    right_shoulder = poses_2d[:, 5, :]
    spine_shoulder = poses_2d[:, 1, :]
    sacrum = poses_2d[:, 14, :]
    left_hip = poses_2d[:, 8, :]
    right_hip = poses_2d[:, 11, :]

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    hip_width = np.linalg.norm(left_hip - right_hip, axis=1)
    torso_height = np.linalg.norm(spine_shoulder - sacrum, axis=1)

    torso_diameter = np.maximum.reduce([
        shoulder_width,
        hip_width,
        torso_height * 0.5,
    ])

    torso_diameter = np.clip(torso_diameter, 50.0, 300.0)
    invalid_mask = torso_diameter < 1e-6
    torso_diameter[invalid_mask] = 100.0
    return torso_diameter


def compute_pck_2d(pred_poses, gt_poses, torso_diameters, fixed_threshold=150.0):
    if pred_poses.shape != gt_poses.shape:
        return {
            'PCK@20%_torso': 0.0,
            'PCK@50%_torso': 0.0,
            'PCK@80%_torso': 0.0,
            'PCK@100%_torso': 0.0,
            'PCK@100%_{}px'.format(int(fixed_threshold)): 0.0,
        }

    joint_distances = np.linalg.norm(pred_poses - gt_poses, axis=2)

    results = {}
    for pct in [20, 50, 80, 100]:
        threshold = torso_diameters[:, np.newaxis] * (pct / 100.0)
        correct = joint_distances < threshold
        results['PCK@{}%_torso'.format(pct)] = float(np.mean(correct))

    correct_fixed = joint_distances < fixed_threshold
    results['PCK@100%_{}px'.format(int(fixed_threshold))] = float(np.mean(correct_fixed))
    return results


def compute_auc_2d(pred_poses, gt_poses, max_threshold=150.0, num_thresholds=50):
    if pred_poses.shape != gt_poses.shape:
        return 0.0

    joint_distances = np.linalg.norm(pred_poses - gt_poses, axis=2)
    thresholds = np.linspace(0.0, max_threshold, num_thresholds)

    pck_values = []
    for threshold in thresholds:
        correct = joint_distances < threshold
        pck_values.append(float(np.mean(correct)))

    auc = float(np.trapz(pck_values, thresholds) / max_threshold)
    return auc


def make_root_relative_2d_pixel(poses_2d_pixel, root_joint_idx=14):
    root_relative_poses = poses_2d_pixel.copy()
    for frame_idx in range(poses_2d_pixel.shape[0]):
        root_pos = poses_2d_pixel[frame_idx, root_joint_idx, :2]
        root_relative_poses[frame_idx, :, :2] -= root_pos
    return root_relative_poses


def compute_compare_metrics(gt_poses_2d, pred_poses_2d, fixed_pck_threshold=150.0,
                            auc_max_threshold=150.0, auc_num_thresholds=50):
    min_frames = min(len(gt_poses_2d), len(pred_poses_2d))
    gt_poses = gt_poses_2d[:min_frames]
    pred_poses = pred_poses_2d[:min_frames]

    valid_gt_list = []
    valid_pred_list = []
    frame_mpjpe = []

    for frame_idx in range(min_frames):
        gt_frame = gt_poses[frame_idx]
        pred_frame = pred_poses[frame_idx]

        gt_valid = not np.all(gt_frame == 0)
        pred_valid = not np.all(pred_frame == 0)

        if gt_valid and pred_valid:
            valid_gt_list.append(gt_frame)
            valid_pred_list.append(pred_frame)
            joint_diffs = np.linalg.norm(gt_frame - pred_frame, axis=1)
            frame_mpjpe.append(float(np.mean(joint_diffs)))
        else:
            frame_mpjpe.append(float('nan'))

    if not valid_gt_list:
        return None

    valid_gt = np.array(valid_gt_list, dtype=np.float32)
    valid_pred = np.array(valid_pred_list, dtype=np.float32)

    avg_mpjpe = float(np.mean([e for e in frame_mpjpe if not np.isnan(e)]))
    joint_errors = np.mean(np.linalg.norm(valid_gt - valid_pred, axis=2), axis=0)

    torso_diameters = calculate_torso_diameter_2d(valid_gt)
    pck_results = compute_pck_2d(
        valid_pred,
        valid_gt,
        torso_diameters,
        fixed_threshold=fixed_pck_threshold,
    )
    auc = compute_auc_2d(
        valid_pred,
        valid_gt,
        max_threshold=auc_max_threshold,
        num_thresholds=auc_num_thresholds,
    )

    return {
        'avg_mpjpe': avg_mpjpe,
        'frame_mpjpe': frame_mpjpe,
        'joint_errors': [float(x) for x in joint_errors],
        'pck_results': pck_results,
        'auc': float(auc),
        'valid_frames': int(len(valid_gt_list)),
        'total_frames': int(min_frames),
    }


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, _ = create_logger(cfg, args.cfg, 'valid_compare')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )

    num_samples = len(valid_dataset)
    all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
    all_gts = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
    all_vis = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS), dtype=np.float32)

    idx = 0
    total_inference_time = 0.0

    with torch.no_grad():
        for i, (input_data, _, _, meta) in enumerate(valid_loader):
            input_data = input_data.cuda(non_blocking=True)
            batch_size = input_data.size(0)

            start_time = time.perf_counter()
            outputs = model(input_data)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if cfg.TEST.FLIP_TEST:
                input_flipped = np.flip(input_data.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(), valid_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                if cfg.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            total_inference_time += (time.perf_counter() - start_time)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            preds, _ = get_final_preds(cfg, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + batch_size, :, :] = preds[:, :, 0:2]

            for j in range(batch_size):
                db_rec = valid_dataset.db[idx + j]
                all_gts[idx + j, :, :] = db_rec['joints_3d'][:, 0:2]
                all_vis[idx + j, :] = db_rec['joints_3d_vis'][:, 0]

            idx += batch_size

            if i % max(1, cfg.PRINT_FREQ) == 0:
                logger.info('Test: [{}/{}]'.format(i, len(valid_loader)))

    all_preds = all_preds[:idx]
    all_gts = all_gts[:idx]
    all_vis = all_vis[:idx]

    all_preds_root = make_root_relative_2d_pixel(all_preds, root_joint_idx=args.root_joint_idx)
    all_gts_root = make_root_relative_2d_pixel(all_gts, root_joint_idx=args.root_joint_idx)

    invisible_mask = all_vis <= 0
    all_preds_root[invisible_mask] = 0.0
    all_gts_root[invisible_mask] = 0.0

    metrics = compute_compare_metrics(
        all_gts_root,
        all_preds_root,
        fixed_pck_threshold=args.fixed_pck_threshold,
        auc_max_threshold=args.auc_max_threshold,
        auc_num_thresholds=args.auc_num_thresholds,
    )

    if metrics is None:
        logger.error('No valid frames found for metric computation.')
        return

    mean_inference_time = total_inference_time / max(1, idx)
    fps = idx / total_inference_time if total_inference_time > 0 else 0.0

    metrics['performance'] = {
        'total_inference_time': float(total_inference_time),
        'mean_inference_time': float(mean_inference_time),
        'fps': float(fps),
        'processed_frames': int(idx),
    }

    logger.info('============================================================')
    logger.info('HRNet vs Ground Truth (2D Root-Relative Pixel Metrics)')
    logger.info('Valid frames: %d/%d', metrics['valid_frames'], metrics['total_frames'])
    logger.info('Average MPJPE: %.4f px', metrics['avg_mpjpe'])
    logger.info('AUC: %.6f', metrics['auc'])
    for key, value in metrics['pck_results'].items():
        logger.info('%s: %.2f%%', key, value * 100.0)

    logger.info('Total inference time: %.4f s', metrics['performance']['total_inference_time'])
    logger.info('Mean inference time: %.4f ms', metrics['performance']['mean_inference_time'] * 1000.0)
    logger.info('FPS: %.2f', metrics['performance']['fps'])
    logger.info('============================================================')

    if args.save_json:
        out_path = os.path.join(final_output_dir, 'compare_metrics.json')
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info('Saved metrics JSON: %s', out_path)


if __name__ == '__main__':
    main()
