# ------------------------------------------------------------------------------
# HRNet train entrypoint with YOLO-style 2D metrics for MPI-INF-3DHP train split
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import json
import os
import pprint
import time

import cv2
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
from utils.transforms import flip_back, get_affine_transform
from utils.utils import create_logger

import dataset
import models


CONNECTIONS_2D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test HRNet on MPI-INF-3DHP train split with MPJPE/PCK/AUC metrics'
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
    parser.add_argument('--save-video', action='store_true',
                        help='Save side-by-side GT vs HRNet keypoint video')
    parser.add_argument('--video-sequence', type=str, default='',
                        help='Train stream for video export (e.g., S1_Seq1_cam0).')
    parser.add_argument('--video-num-frames', type=int, default=300,
                        help='Maximum number of frames to export in video')
    parser.add_argument('--video-fps', type=float, default=8.0,
                        help='Output video FPS')
    parser.add_argument('--video-output-dir', type=str, default='',
                        help='Directory to save generated video (default: final output dir)')

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


def extract_train_stream_name(db_rec):
    filename = str(db_rec.get('filename', ''))
    if '_cam' in filename and '_frame' in filename:
        return filename.split('_frame')[0]

    image_path = str(db_rec.get('image', ''))
    normalized = image_path.replace('\\', '/')
    parts = [p for p in normalized.split('/') if p]
    subject = next((p for p in parts if p.startswith('S') and p[1:].isdigit()), 'UNKNOWN')
    sequence = next((p for p in parts if p.lower().startswith('seq')), 'UNKNOWN')
    camera_part = 'UNKNOWN'
    for part in parts:
        if part.startswith('video_'):
            camera_part = 'cam{}'.format(part.replace('video_', ''))
            break
    return '{}_{}_{}'.format(subject, sequence, camera_part)


def print_sequence_results(logger, metrics):
    if not metrics:
        return

    logger.info('')
    logger.info('Results for %s:', metrics['sequence'])
    logger.info('  MPJPE: %.2f pixels', metrics['avg_mpjpe'])
    logger.info('  AUC: %.4f', metrics['auc'])
    logger.info('  Valid frames: %d/%d', metrics['valid_frames'], metrics['total_frames'])
    logger.info('  FPS: %.2f', metrics['performance']['fps'])
    logger.info('  Mean inference time: %.2f ms', metrics['performance']['mean_inference_time'] * 1000.0)
    logger.info('  PCK metrics:')
    for key, value in metrics['pck_results'].items():
        logger.info('    %s: %.2f%%', key, value * 100.0)


def print_summary_results(logger, all_metrics, model_name):
    if not all_metrics:
        logger.info('No results to summarize.')
        return

    valid_metrics = [m for m in all_metrics if m is not None]
    if not valid_metrics:
        logger.info('No valid metrics found.')
        return

    logger.info('')
    logger.info('================================================================================')
    logger.info('COMPREHENSIVE SUMMARY - HRNet vs Ground Truth (2D)')
    logger.info('Model: %s', model_name)
    logger.info('================================================================================')

    total_mpjpe_sum = 0.0
    total_weighted_frames = 0
    for metric in valid_metrics:
        weight = metric['valid_frames']
        total_mpjpe_sum += metric['avg_mpjpe'] * weight
        total_weighted_frames += weight

    overall_mpjpe = total_mpjpe_sum / total_weighted_frames if total_weighted_frames > 0 else 0.0

    avg_mpjpe = float(np.mean([m['avg_mpjpe'] for m in valid_metrics]))
    avg_auc = float(np.mean([m['auc'] for m in valid_metrics]))
    total_valid_frames = int(sum([m['valid_frames'] for m in valid_metrics]))
    total_frames = int(sum([m['total_frames'] for m in valid_metrics]))

    total_inference_time = float(sum([m['performance']['total_inference_time'] for m in valid_metrics]))
    total_processed_frames = int(sum([m['performance']['processed_frames'] for m in valid_metrics]))

    weighted_fps_sum = 0.0
    weighted_inference_time_sum = 0.0
    total_weight = 0
    for metric in valid_metrics:
        weight = metric['performance']['processed_frames']
        weighted_fps_sum += metric['performance']['fps'] * weight
        weighted_inference_time_sum += metric['performance']['mean_inference_time'] * weight
        total_weight += weight

    overall_fps = weighted_fps_sum / total_weight if total_weight > 0 else 0.0
    overall_mean_inference_time = weighted_inference_time_sum / total_weight if total_weight > 0 else 0.0

    logger.info('')
    logger.info('OVERALL METRICS:')
    logger.info('  Streams processed: %d', len(valid_metrics))
    logger.info('  Total valid frames: %d/%d', total_valid_frames, total_frames)
    logger.info('  Overall MPJPE (weighted): %.2f pixels', overall_mpjpe)
    logger.info('  Average MPJPE (per stream): %.2f pixels', avg_mpjpe)
    logger.info('  Average AUC: %.4f', avg_auc)

    logger.info('')
    logger.info('PERFORMACE METRICS:')
    logger.info('  Total inference time: %.2f seconds', total_inference_time)
    logger.info('  Total processed frames: %d', total_processed_frames)
    logger.info('  Overall FPS (weighted): %.2f', overall_fps)
    logger.info('  Overall mean inference time: %.2f ms', overall_mean_inference_time * 1000.0)

    pck_keys = valid_metrics[0]['pck_results'].keys()
    logger.info('')
    logger.info('PCK METRICS (averaged across streams):')
    for key in pck_keys:
        avg_pck = float(np.mean([m['pck_results'][key] for m in valid_metrics]))
        logger.info('  %s: %.2f%%', key, avg_pck * 100.0)

    logger.info('')
    logger.info('PER-STREAM BREAKDOWN:')
    logger.info('%-18s %-12s %-10s %-10s %-12s %-12s',
                'Stream', 'MPJPE', 'AUC', 'FPS', 'Time(ms)', 'Valid/Total')
    logger.info('%s', '-' * 86)

    for metric in valid_metrics:
        logger.info('%-18s %-12.2f %-10.4f %-10.2f %-12.2f %d/%d',
                    metric['sequence'],
                    metric['avg_mpjpe'],
                    metric['auc'],
                    metric['performance']['fps'],
                    metric['performance']['mean_inference_time'] * 1000.0,
                    metric['valid_frames'],
                    metric['total_frames'])

    logger.info('================================================================================')
    logger.info('FINAL OVERALL MPJPE: %.2f pixels', overall_mpjpe)
    logger.info('FINAL OVERALL FPS: %.2f', overall_fps)
    logger.info('FINAL MEAN INFERENCE TIME: %.2f ms', overall_mean_inference_time * 1000.0)
    logger.info('================================================================================')


def draw_pose(image, joints, visibility, line_color, point_color):
    for j1, j2 in CONNECTIONS_2D:
        if not (visibility[j1] and visibility[j2]):
            continue
        p1 = joints[j1]
        p2 = joints[j2]
        if np.allclose(p1, 0.0) or np.allclose(p2, 0.0):
            continue
        cv2.line(
            image,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            line_color,
            2,
            cv2.LINE_AA,
        )

    for joint_idx, joint_xy in enumerate(joints):
        if not visibility[joint_idx]:
            continue
        if np.allclose(joint_xy, 0.0):
            continue
        cv2.circle(
            image,
            (int(round(joint_xy[0])), int(round(joint_xy[1]))),
            4,
            point_color,
            -1,
            cv2.LINE_AA,
        )


def save_keypoint_comparison_video(logger, output_path, stream_name, sample_indices, image_paths,
                                   all_gts, all_preds, all_vis, frame_mpjpe, fps, max_frames):
    if not sample_indices:
        logger.warning('No indices found for stream %s. Video was not created.', stream_name)
        return

    if max_frames is not None and max_frames > 0:
        sample_indices = sample_indices[:max_frames]

    writer = None
    exported = 0

    for local_idx, global_idx in enumerate(sample_indices):
        image_path = image_paths[global_idx]
        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning('Could not read image for video: %s', image_path)
            continue

        gt_frame = frame.copy()
        pred_frame = frame.copy()

        visibility = all_vis[global_idx] > 0
        draw_pose(gt_frame, all_gts[global_idx], visibility, line_color=(255, 140, 0), point_color=(255, 80, 0))
        draw_pose(pred_frame, all_preds[global_idx], visibility, line_color=(0, 180, 255), point_color=(0, 255, 255))

        cv2.putText(gt_frame, 'Ground Truth', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pred_frame, 'HRNet Prediction', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        frame_err = float('nan')
        if local_idx < len(frame_mpjpe):
            frame_err = frame_mpjpe[local_idx]
        mpjpe_text = 'Frame MPJPE: N/A' if np.isnan(frame_err) else 'Frame MPJPE: {:.2f}px'.format(frame_err)
        cv2.putText(pred_frame, mpjpe_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        side_by_side = np.concatenate([gt_frame, pred_frame], axis=1)

        if writer is None:
            h, w = side_by_side.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                logger.error('Failed to open video writer: %s', output_path)
                return

        writer.write(side_by_side)
        exported += 1

    if writer is not None:
        writer.release()

    if exported == 0:
        logger.warning('No frames were exported to video for stream %s.', stream_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        return

    logger.info('Saved video for %s (%d frames): %s', stream_name, exported, output_path)


def safe_load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            return checkpoint['state_dict']
        if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            return checkpoint['model_state_dict']
    return checkpoint


def parse_train_stream_name(stream_name):
    if '_cam' not in stream_name:
        raise ValueError('Invalid stream format: {}. Expected Sx_Seqy_camZ'.format(stream_name))

    stream_part, cam_part = stream_name.rsplit('_cam', 1)
    pieces = stream_part.split('_', 1)
    if len(pieces) != 2:
        raise ValueError('Invalid stream format: {}. Expected Sx_Seqy_camZ'.format(stream_name))

    subject = pieces[0]
    sequence = pieces[1]
    camera = cam_part
    return subject, sequence, camera


def resolve_annotation_file(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(os.getcwd(), path_value)


def resolve_train_camera_folder(train_image_root, subject, sequence, camera_key):
    candidates = [
        os.path.join(train_image_root, subject, sequence, 'imageFrames', 'video_{}'.format(camera_key)),
    ]
    if str(camera_key).isdigit():
        candidates.append(os.path.join(train_image_root, subject, sequence, 'imageFrames', 'video_{}'.format(int(camera_key))))
        candidates.append(os.path.join(train_image_root, subject, sequence, 'imageFrames', 'video_{:02d}'.format(int(camera_key))))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def compute_center_scale(joints_xy, aspect_ratio, pixel_std=200.0):
    x_min = np.min(joints_xy[:, 0])
    y_min = np.min(joints_xy[:, 1])
    x_max = np.max(joints_xy[:, 0])
    y_max = np.max(joints_xy[:, 1])

    w = x_max - x_min
    h = y_max - y_min
    if w < 2 or h < 2:
        return None, None

    center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32) * 1.25
    return center, scale


def build_train_stream_db(cfg_obj, stream_name):
    subject, sequence, camera = parse_train_stream_name(stream_name)
    seq_key = '{} {}'.format(subject, sequence)

    annotation_file = resolve_annotation_file(cfg_obj.DATASET.TRAIN_ANNOTATION_FILE)
    if not os.path.exists(annotation_file):
        raise FileNotFoundError('Training annotation file not found: {}'.format(annotation_file))

    annotations = np.load(annotation_file, allow_pickle=True)['data'].item()
    if seq_key not in annotations:
        raise KeyError('Sequence not found in train annotations: {}'.format(seq_key))

    seq_data = annotations[seq_key]
    if not isinstance(seq_data, list) or len(seq_data) < 1:
        raise ValueError('Unexpected train annotation format for {}'.format(seq_key))
    camera_dict = seq_data[0]

    cam_candidates = [str(camera)]
    if str(camera).isdigit():
        cam_candidates.append(str(int(camera)))
        cam_candidates.append('{:02d}'.format(int(camera)))

    selected_cam = None
    for cam_key in cam_candidates:
        if cam_key in camera_dict and isinstance(camera_dict[cam_key], dict) and 'data_2d' in camera_dict[cam_key]:
            selected_cam = cam_key
            break
    if selected_cam is None:
        raise KeyError('Camera {} not found in sequence {}'.format(camera, seq_key))

    poses_2d = camera_dict[selected_cam]['data_2d']
    train_image_root = cfg_obj.DATASET.TRAIN_IMAGE_ROOT if cfg_obj.DATASET.TRAIN_IMAGE_ROOT else cfg_obj.DATASET.ROOT
    image_folder = resolve_train_camera_folder(train_image_root, subject, sequence, selected_cam)
    if image_folder is None:
        raise FileNotFoundError('Image folder not found for {}_{}_cam{}'.format(subject, sequence, selected_cam))

    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
    image_files.extend(glob.glob(os.path.join(image_folder, '*.JPG')))
    image_files.sort()
    if not image_files:
        raise FileNotFoundError('No images found in {}'.format(image_folder))

    max_frames = min(len(image_files), len(poses_2d))
    frame_stride = max(1, int(os.environ.get('MPI3DHP_TRAIN_FRAME_STRIDE', '1')))
    aspect_ratio = cfg_obj.MODEL.IMAGE_SIZE[0] * 1.0 / cfg_obj.MODEL.IMAGE_SIZE[1]

    db = []
    for frame_idx in range(0, max_frames, frame_stride):
        joints_xy = poses_2d[frame_idx].astype(np.float32)
        if joints_xy.shape[0] != 17:
            continue

        center, scale = compute_center_scale(joints_xy, aspect_ratio)
        if center is None:
            continue

        joints_3d = np.zeros((17, 3), dtype=np.float32)
        joints_3d_vis = np.ones((17, 3), dtype=np.float32)
        joints_3d[:, 0:2] = joints_xy[:, 0:2]

        db.append(
            {
                'image': image_files[frame_idx],
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '{}_{}_cam{}_frame{:06d}'.format(subject, sequence, selected_cam, frame_idx),
                'imgnum': frame_idx,
                'score': 1.0,
            }
        )

    return db


class TrainStreamVideoDataset(torch.utils.data.Dataset):
    def __init__(self, db, cfg_obj, transform=None):
        self.db = db
        self.color_rgb = cfg_obj.DATASET.COLOR_RGB
        self.image_size = np.array(cfg_obj.MODEL.IMAGE_SIZE)
        self.transform = transform

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = self.db[idx]
        image_file = db_rec['image']
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        c = db_rec['center']
        s = db_rec['scale']
        trans = get_affine_transform(c, s, 0, self.image_size)

        input_data = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR,
        )

        if self.transform:
            input_data = self.transform(input_data)

        meta = {
            'center': c.astype(np.float32),
            'scale': s.astype(np.float32),
            'db_index': np.int64(idx),
        }
        return input_data, meta


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, _ = create_logger(cfg, args.cfg, 'train_compare')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(extract_state_dict(safe_load_checkpoint(cfg.TEST.MODEL_FILE)), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(extract_state_dict(safe_load_checkpoint(model_state_file)), strict=False)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.save_video:
        if not args.video_sequence:
            logger.error('In video mode please provide --video-sequence (e.g., S1_Seq1_cam0).')
            return

        logger.info('Video mode: loading only stream %s', args.video_sequence)
        try:
            stream_db = build_train_stream_db(cfg, args.video_sequence)
        except Exception as e:
            logger.error('Failed to load requested stream %s: %s', args.video_sequence, str(e))
            return

        if not stream_db:
            logger.error('No frames found for stream %s', args.video_sequence)
            return

        video_dataset = TrainStreamVideoDataset(stream_db, cfg, transform=transform_pipeline)
        video_loader = torch.utils.data.DataLoader(
            video_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True,
        )

        num_samples = len(video_dataset)
        all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
        all_gts = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
        all_vis = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS), dtype=np.float32)
        all_image_paths = [''] * num_samples

        idx = 0
        with torch.no_grad():
            for i, (input_data, meta) in enumerate(video_loader):
                input_data = input_data.cuda(non_blocking=True)
                batch_size = input_data.size(0)

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

                    output_flipped = flip_back(output_flipped.cpu().numpy(), [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13]])
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                    if cfg.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                preds, _ = get_final_preds(cfg, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + batch_size, :, :] = preds[:, :, 0:2]

                for j in range(batch_size):
                    db_idx = int(meta['db_index'][j].item())
                    db_rec = video_dataset.db[db_idx]
                    all_gts[idx + j, :, :] = db_rec['joints_3d'][:, 0:2]
                    all_vis[idx + j, :] = db_rec['joints_3d_vis'][:, 0]
                    all_image_paths[idx + j] = str(db_rec.get('image', ''))

                idx += batch_size

                if i % max(1, cfg.PRINT_FREQ) == 0:
                    logger.info('Video stream: [{}/{}]'.format(i, len(video_loader)))

        all_preds = all_preds[:idx]
        all_gts = all_gts[:idx]
        all_vis = all_vis[:idx]
        all_image_paths = all_image_paths[:idx]

        all_preds_root = make_root_relative_2d_pixel(all_preds, root_joint_idx=args.root_joint_idx)
        all_gts_root = make_root_relative_2d_pixel(all_gts, root_joint_idx=args.root_joint_idx)

        invisible_mask = all_vis <= 0
        all_preds_root[invisible_mask] = 0.0
        all_gts_root[invisible_mask] = 0.0

        stream_metrics = compute_compare_metrics(
            all_gts_root,
            all_preds_root,
            fixed_pck_threshold=args.fixed_pck_threshold,
            auc_max_threshold=args.auc_max_threshold,
            auc_num_thresholds=args.auc_num_thresholds,
        )
        frame_mpjpe = stream_metrics['frame_mpjpe'] if stream_metrics is not None else []

        video_output_dir = args.video_output_dir if args.video_output_dir else final_output_dir
        os.makedirs(video_output_dir, exist_ok=True)

        model_name = os.path.basename(cfg.TEST.MODEL_FILE) if cfg.TEST.MODEL_FILE else 'final_state.pth'
        model_stem = os.path.splitext(model_name)[0]
        video_path = os.path.join(video_output_dir, '{}_hrnet_train_compare_{}.mp4'.format(args.video_sequence, model_stem))

        logger.info('Exporting keypoint video for stream %s ...', args.video_sequence)
        save_keypoint_comparison_video(
            logger=logger,
            output_path=video_path,
            stream_name=args.video_sequence,
            sample_indices=list(range(idx)),
            image_paths=all_image_paths,
            all_gts=all_gts,
            all_preds=all_preds,
            all_vis=all_vis,
            frame_mpjpe=frame_mpjpe,
            fps=max(1.0, args.video_fps),
            max_frames=args.video_num_frames,
        )

        if args.save_json:
            out_path = os.path.join(final_output_dir, 'compare_train_metrics.json')
            with open(out_path, 'w') as f:
                json.dump(
                    {
                        'video_stream': args.video_sequence,
                        'stream_metrics': stream_metrics,
                    },
                    f,
                    indent=2,
                )
            logger.info('Saved stream metrics JSON: %s', out_path)

        return

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_SET,
        True,
        transform_pipeline,
    )
    train_dataset.is_train = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )

    num_samples = len(train_dataset)
    all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
    all_gts = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
    all_vis = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS), dtype=np.float32)
    all_stream_names = ['UNKNOWN'] * num_samples
    all_image_paths = [''] * num_samples
    all_inference_times = np.zeros((num_samples,), dtype=np.float32)

    idx = 0
    total_inference_time = 0.0

    with torch.no_grad():
        for i, (input_data, _, _, meta) in enumerate(train_loader):
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

                output_flipped = flip_back(output_flipped.cpu().numpy(), train_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                if cfg.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            batch_inference_time = time.perf_counter() - start_time
            total_inference_time += batch_inference_time
            per_sample_time = batch_inference_time / max(1, batch_size)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            preds, _ = get_final_preds(cfg, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + batch_size, :, :] = preds[:, :, 0:2]

            for j in range(batch_size):
                db_rec = train_dataset.db[idx + j]
                all_gts[idx + j, :, :] = db_rec['joints_3d'][:, 0:2]
                all_vis[idx + j, :] = db_rec['joints_3d_vis'][:, 0]
                all_stream_names[idx + j] = extract_train_stream_name(db_rec)
                all_image_paths[idx + j] = str(db_rec.get('image', ''))
                all_inference_times[idx + j] = per_sample_time

            idx += batch_size

            if i % max(1, cfg.PRINT_FREQ) == 0:
                logger.info('Train: [{}/{}]'.format(i, len(train_loader)))

    all_preds = all_preds[:idx]
    all_gts = all_gts[:idx]
    all_vis = all_vis[:idx]
    all_stream_names = all_stream_names[:idx]
    all_image_paths = all_image_paths[:idx]
    all_inference_times = all_inference_times[:idx]

    all_preds_root = make_root_relative_2d_pixel(all_preds, root_joint_idx=args.root_joint_idx)
    all_gts_root = make_root_relative_2d_pixel(all_gts, root_joint_idx=args.root_joint_idx)

    invisible_mask = all_vis <= 0
    all_preds_root[invisible_mask] = 0.0
    all_gts_root[invisible_mask] = 0.0

    stream_indices_map = {}
    for stream_name in sorted(set(all_stream_names)):
        stream_indices = [i for i, seq in enumerate(all_stream_names) if seq == stream_name]
        if stream_indices:
            stream_indices_map[stream_name] = stream_indices

    if args.save_video:
        if not stream_indices_map:
            logger.warning('No stream data available for video export.')
            return

        target_stream = args.video_sequence if args.video_sequence else sorted(stream_indices_map.keys())[0]
        if target_stream not in stream_indices_map:
            logger.warning('Requested video stream %s not found. Using %s instead.',
                           target_stream, sorted(stream_indices_map.keys())[0])
            target_stream = sorted(stream_indices_map.keys())[0]

        target_indices = stream_indices_map[target_stream]
        target_metrics = compute_compare_metrics(
            all_gts_root[target_indices],
            all_preds_root[target_indices],
            fixed_pck_threshold=args.fixed_pck_threshold,
            auc_max_threshold=args.auc_max_threshold,
            auc_num_thresholds=args.auc_num_thresholds,
        )
        target_frame_mpjpe = target_metrics['frame_mpjpe'] if target_metrics is not None else []

        video_output_dir = args.video_output_dir if args.video_output_dir else final_output_dir
        os.makedirs(video_output_dir, exist_ok=True)

        model_name = os.path.basename(cfg.TEST.MODEL_FILE) if cfg.TEST.MODEL_FILE else 'final_state.pth'
        model_stem = os.path.splitext(model_name)[0]
        video_path = os.path.join(video_output_dir, '{}_hrnet_train_compare_{}.mp4'.format(target_stream, model_stem))

        logger.info('Exporting keypoint video for stream %s ...', target_stream)
        save_keypoint_comparison_video(
            logger=logger,
            output_path=video_path,
            stream_name=target_stream,
            sample_indices=target_indices,
            image_paths=all_image_paths,
            all_gts=all_gts,
            all_preds=all_preds,
            all_vis=all_vis,
            frame_mpjpe=target_frame_mpjpe,
            fps=max(1.0, args.video_fps),
            max_frames=args.video_num_frames,
        )

        if args.save_json:
            out_path = os.path.join(final_output_dir, 'compare_train_metrics.json')
            with open(out_path, 'w') as f:
                json.dump({
                    'video_stream': target_stream,
                    'stream_metrics': target_metrics,
                }, f, indent=2)
            logger.info('Saved stream metrics JSON: %s', out_path)

        return

    overall_metrics = compute_compare_metrics(
        all_gts_root,
        all_preds_root,
        fixed_pck_threshold=args.fixed_pck_threshold,
        auc_max_threshold=args.auc_max_threshold,
        auc_num_thresholds=args.auc_num_thresholds,
    )

    if overall_metrics is None:
        logger.error('No valid frames found for metric computation.')
        return

    mean_inference_time = total_inference_time / max(1, idx)
    fps = idx / total_inference_time if total_inference_time > 0 else 0.0

    overall_metrics['performance'] = {
        'total_inference_time': float(total_inference_time),
        'mean_inference_time': float(mean_inference_time),
        'fps': float(fps),
        'processed_frames': int(idx),
    }

    per_stream_metrics = []
    stream_frame_mpjpe_map = {}
    for stream_name, stream_indices in stream_indices_map.items():

        seq_preds = all_preds_root[stream_indices]
        seq_gts = all_gts_root[stream_indices]
        seq_times = all_inference_times[stream_indices]

        seq_metrics = compute_compare_metrics(
            seq_gts,
            seq_preds,
            fixed_pck_threshold=args.fixed_pck_threshold,
            auc_max_threshold=args.auc_max_threshold,
            auc_num_thresholds=args.auc_num_thresholds,
        )
        if seq_metrics is None:
            continue

        seq_mean_time = float(np.mean(seq_times)) if len(seq_times) > 0 else 0.0
        seq_fps = float(1.0 / seq_mean_time) if seq_mean_time > 0 else 0.0

        seq_metrics['sequence'] = stream_name
        seq_metrics['performance'] = {
            'total_inference_time': float(np.sum(seq_times)),
            'mean_inference_time': seq_mean_time,
            'fps': seq_fps,
            'processed_frames': int(len(seq_times)),
        }
        stream_frame_mpjpe_map[stream_name] = seq_metrics['frame_mpjpe']
        per_stream_metrics.append(seq_metrics)

    for seq_metric in per_stream_metrics:
        print_sequence_results(logger, seq_metric)

    model_name = os.path.basename(cfg.TEST.MODEL_FILE) if cfg.TEST.MODEL_FILE else 'final_state.pth'
    print_summary_results(logger, per_stream_metrics, model_name)

    logger.info('============================================================')
    logger.info('HRNet vs Ground Truth (2D Root-Relative Pixel Metrics)')
    logger.info('Valid frames: %d/%d', overall_metrics['valid_frames'], overall_metrics['total_frames'])
    logger.info('Average MPJPE: %.4f px', overall_metrics['avg_mpjpe'])
    logger.info('AUC: %.6f', overall_metrics['auc'])
    for key, value in overall_metrics['pck_results'].items():
        logger.info('%s: %.2f%%', key, value * 100.0)

    logger.info('Total inference time: %.4f s', overall_metrics['performance']['total_inference_time'])
    logger.info('Mean inference time: %.4f ms', overall_metrics['performance']['mean_inference_time'] * 1000.0)
    logger.info('FPS: %.2f', overall_metrics['performance']['fps'])
    logger.info('============================================================')

    if args.save_json:
        out_path = os.path.join(final_output_dir, 'compare_train_metrics.json')
        with open(out_path, 'w') as f:
            json.dump({
                'overall': overall_metrics,
                'per_stream': per_stream_metrics,
            }, f, indent=2)
        logger.info('Saved metrics JSON: %s', out_path)


if __name__ == '__main__':
    main()