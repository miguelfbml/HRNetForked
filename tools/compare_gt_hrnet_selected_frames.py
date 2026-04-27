# ------------------------------------------------------------------------------
# Export selected-frame GT vs HRNet comparisons for MPI-INF-3DHP
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import _init_paths
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import flip_back, get_affine_transform

import models


CONNECTIONS_2D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export selected-frame GT vs HRNet comparisons as images'
    )
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='experiments/3DHP/hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp.yaml', type=str)
    parser.add_argument('--sequence', type=str, required=True,
                        help='Sequence to export (TS1..TS6)')
    parser.add_argument('--frames', nargs='+', type=int, required=True,
                        help='Frame indices to export, e.g. --frames 600 1200 1800')
    parser.add_argument('--model-path', type=str, default='',
                        help='HRNet checkpoint path (.pth). Overrides TEST.MODEL_FILE when provided.')
    parser.add_argument('--output-dir', type=str, default='comparison_selected_frames',
                        help='Output root directory. Sequence subfolders are created inside it.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Inference batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference, e.g. cuda:0 or cpu')

    # Compatibility args (accepted but not used directly for HRNet image size).
    parser.add_argument('--img-size', type=int, default=640,
                        help='Compatibility arg; HRNet image size comes from cfg.MODEL.IMAGE_SIZE')

    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev model directory', type=str, default='')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser.parse_args()


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


def resolve_annotation_file(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(os.getcwd(), path_value)


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


def resolve_test_sequence_folder(test_image_root, sequence_name):
    candidate = os.path.join(test_image_root, sequence_name, 'imageSequence')
    if os.path.exists(candidate):
        return candidate
    return None


def build_selected_sequence_db(cfg_obj, sequence_name, frame_indices):
    annotation_file = resolve_annotation_file(cfg_obj.DATASET.TEST_ANNOTATION_FILE)
    if not os.path.exists(annotation_file):
        raise FileNotFoundError('Test annotation file not found: {}'.format(annotation_file))

    annotations = np.load(annotation_file, allow_pickle=True)['data'].item()
    if sequence_name not in annotations:
        raise KeyError('Sequence not found in test annotations: {}'.format(sequence_name))

    seq_data = annotations[sequence_name]
    if not isinstance(seq_data, dict) or 'data_2d' not in seq_data:
        raise ValueError('Unexpected test annotation format for {}'.format(sequence_name))

    test_image_root = cfg_obj.DATASET.TEST_IMAGE_ROOT if cfg_obj.DATASET.TEST_IMAGE_ROOT else os.path.join(cfg_obj.DATASET.ROOT, 'mpi_inf_3dhp_test_set')
    image_folder = resolve_test_sequence_folder(test_image_root, sequence_name)
    if image_folder is None:
        raise FileNotFoundError('Image folder not found for {}'.format(sequence_name))

    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
    image_files.extend(glob.glob(os.path.join(image_folder, '*.png')))
    image_files.sort()
    if not image_files:
        raise FileNotFoundError('No images found in {}'.format(image_folder))

    poses_2d = seq_data['data_2d']
    max_frames = min(len(image_files), len(poses_2d))
    aspect_ratio = cfg_obj.MODEL.IMAGE_SIZE[0] * 1.0 / cfg_obj.MODEL.IMAGE_SIZE[1]

    db = []
    unique_frames = []
    seen = set()
    for frame_idx in frame_indices:
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        unique_frames.append(frame_idx)

    for frame_idx in unique_frames:
        if frame_idx < 0 or frame_idx >= max_frames:
            print('Warning: frame {} is out of range for {} (0..{}), skipping.'.format(
                frame_idx, sequence_name, max_frames - 1
            ))
            continue

        joints_xy = poses_2d[frame_idx].astype(np.float32)
        if joints_xy.shape[0] != 17:
            print('Warning: frame {} has {} joints, expected 17, skipping.'.format(frame_idx, joints_xy.shape[0]))
            continue

        center, scale = compute_center_scale(joints_xy, aspect_ratio)
        if center is None:
            print('Warning: frame {} has invalid bbox from joints, skipping.'.format(frame_idx))
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
                'filename': '{}_frame{:06d}'.format(sequence_name, frame_idx),
                'imgnum': frame_idx,
                'score': 1.0,
            }
        )

    return db


class SelectedFramesDataset(torch.utils.data.Dataset):
    def __init__(self, db, cfg_obj, transform=None):
        self.db = db
        self.color_rgb = cfg_obj.DATASET.COLOR_RGB
        self.image_size = np.array(cfg_obj.MODEL.IMAGE_SIZE)
        self.flip_pairs = [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13]]
        self.transform = transform

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = self.db[idx]
        image_file = db_rec['image']
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            raise ValueError('Failed to read {}'.format(image_file))

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


def infer_selected_frames(model, loader, dataset_obj, cfg_obj, device):
    num_samples = len(dataset_obj)
    all_preds = np.zeros((num_samples, cfg_obj.MODEL.NUM_JOINTS, 2), dtype=np.float32)
    all_gts = np.zeros((num_samples, cfg_obj.MODEL.NUM_JOINTS, 2), dtype=np.float32)
    all_vis = np.zeros((num_samples, cfg_obj.MODEL.NUM_JOINTS), dtype=np.float32)
    all_image_paths = [''] * num_samples
    all_frame_numbers = np.zeros((num_samples,), dtype=np.int64)

    idx = 0
    with torch.no_grad():
        for input_data, meta in loader:
            input_data = input_data.to(device, non_blocking=True)
            batch_size = input_data.size(0)

            outputs = model(input_data)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if cfg_obj.TEST.FLIP_TEST:
                input_flipped = np.flip(input_data.detach().cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).to(device)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.detach().cpu().numpy(), dataset_obj.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).to(device)

                if cfg_obj.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            preds, _ = get_final_preds(cfg_obj, output.detach().cpu().numpy(), c, s)
            all_preds[idx:idx + batch_size, :, :] = preds[:, :, 0:2]

            for j in range(batch_size):
                db_idx = int(meta['db_index'][j].item())
                db_rec = dataset_obj.db[db_idx]
                all_gts[idx + j, :, :] = db_rec['joints_3d'][:, 0:2]
                all_vis[idx + j, :] = db_rec['joints_3d_vis'][:, 0]
                all_image_paths[idx + j] = str(db_rec.get('image', ''))
                all_frame_numbers[idx + j] = int(db_rec.get('imgnum', -1))

            idx += batch_size

    return {
        'preds': all_preds[:idx],
        'gts': all_gts[:idx],
        'vis': all_vis[:idx],
        'image_paths': all_image_paths[:idx],
        'frame_numbers': all_frame_numbers[:idx],
    }


def save_selected_frame_images(sequence_name, output_root, infer_data):
    sequence_dir = os.path.join(output_root, sequence_name)
    os.makedirs(sequence_dir, exist_ok=True)

    exported = 0
    n = len(infer_data['image_paths'])

    for i in range(n):
        image_path = infer_data['image_paths'][i]
        frame = cv2.imread(image_path)
        if frame is None:
            print('Warning: could not read image {}'.format(image_path))
            continue

        gt_frame = frame.copy()
        pred_frame = frame.copy()

        visibility = infer_data['vis'][i] > 0
        gt_joints = infer_data['gts'][i]
        pred_joints = infer_data['preds'][i]

        draw_pose(gt_frame, gt_joints, visibility, line_color=(255, 140, 0), point_color=(255, 80, 0))
        draw_pose(pred_frame, pred_joints, visibility, line_color=(0, 180, 255), point_color=(0, 255, 255))

        cv2.putText(pred_frame, 'HRNet Prediction', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(gt_frame, 'Ground Truth', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Prediction on the left, Ground Truth on the right (as requested).
        side_by_side = np.concatenate([pred_frame, gt_frame], axis=1)

        frame_number = int(infer_data['frame_numbers'][i])
        out_name = 'frame_{:06d}_gt_vs_hrnet.png'.format(frame_number)
        out_path = os.path.join(sequence_dir, out_name)
        ok = cv2.imwrite(out_path, side_by_side)
        if ok:
            exported += 1

    return sequence_dir, exported


def main():
    args = parse_args()
    update_config(cfg, args)

    if args.model_path:
        cfg.defrost()
        cfg.TEST.MODEL_FILE = args.model_path
        cfg.freeze()

    if not cfg.TEST.MODEL_FILE:
        raise ValueError('No HRNet checkpoint provided. Use --model-path or set TEST.MODEL_FILE in cfg.')

    checkpoint_path = cfg.TEST.MODEL_FILE
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_path))

    use_cuda = args.device.startswith('cuda') and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else 'cpu')

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    print('Loading HRNet checkpoint: {}'.format(checkpoint_path))
    model.load_state_dict(extract_state_dict(safe_load_checkpoint(checkpoint_path)), strict=False)
    model = model.to(device)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    db = build_selected_sequence_db(cfg, args.sequence, args.frames)
    if not db:
        print('No valid frames to process for {}.'.format(args.sequence))
        return

    dataset_obj = SelectedFramesDataset(db, cfg, transform=transform_pipeline)
    loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY and use_cuda,
    )

    infer_data = infer_selected_frames(model, loader, dataset_obj, cfg, device)

    output_root = args.output_dir
    if not os.path.isabs(output_root):
        output_root = os.path.join(os.getcwd(), output_root)
    os.makedirs(output_root, exist_ok=True)

    sequence_dir, exported = save_selected_frame_images(args.sequence, output_root, infer_data)
    print('Saved {} frame comparisons to {}'.format(exported, sequence_dir))


if __name__ == '__main__':
    main()
