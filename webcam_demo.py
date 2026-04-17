# ------------------------------------------------------------------------------
# HRNet webcam keypoint demo
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
TOOLS_DIR = os.path.join(REPO_ROOT, 'tools')
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import _init_paths
from config import cfg
from core.inference import get_final_preds
import models
from utils.transforms import get_affine_transform


COCO_SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


def parse_args():
    parser = argparse.ArgumentParser(description='HRNet webcam keypoint demo')
    parser.add_argument(
        '--cfg',
        type=str,
        default='experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml',
        help='Path to experiment config yaml',
    )
    parser.add_argument(
        '--model-file',
        type=str,
        default='models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth',
        help='Path to .pth model checkpoint',
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='OpenCV camera index',
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.25,
        help='Minimum keypoint confidence for drawing',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Inference device',
    )
    parser.add_argument(
        '--fps-only',
        action='store_true',
        help='Run only inference FPS benchmark on image files and exit',
    )
    parser.add_argument(
        '--fps-image-dir',
        type=str,
        default='data/coco/images/val2017',
        help='Image directory used when --fps-only is enabled',
    )
    parser.add_argument(
        '--fps-num-images',
        type=int,
        default=200,
        help='Number of images to benchmark when --fps-only is enabled',
    )
    parser.add_argument(
        '--fps-warmup',
        type=int,
        default=20,
        help='Number of warmup iterations before timed benchmark',
    )
    return parser.parse_args()


def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(REPO_ROOT, path_value))


def load_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    return cleaned


def get_center_scale(frame_w, frame_h, image_size):
    center = np.array([frame_w * 0.5, frame_h * 0.5], dtype=np.float32)
    aspect_ratio = float(image_size[0]) / float(image_size[1])

    box_w = float(frame_w)
    box_h = float(frame_h)
    if box_w > aspect_ratio * box_h:
        box_h = box_w / aspect_ratio
    elif box_w < aspect_ratio * box_h:
        box_w = box_h * aspect_ratio

    scale = np.array([box_w / 200.0, box_h / 200.0], dtype=np.float32)
    scale *= 1.25
    return center, scale


def preprocess_frame(frame_bgr, image_size, normalize):
    frame_h, frame_w = frame_bgr.shape[:2]
    center, scale = get_center_scale(frame_w, frame_h, image_size)

    trans = get_affine_transform(center, scale, 0, image_size)
    input_img = cv2.warpAffine(
        frame_bgr,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR,
    )
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    tensor = normalize(input_img).unsqueeze(0)
    return tensor, center, scale


def draw_pose(frame, joints, scores, score_thr):
    for j_idx, (x, y) in enumerate(joints):
        score = float(scores[j_idx])
        if score >= score_thr:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

    for j1, j2 in COCO_SKELETON:
        s1 = float(scores[j1])
        s2 = float(scores[j2])
        if s1 >= score_thr and s2 >= score_thr:
            p1 = (int(joints[j1][0]), int(joints[j1][1]))
            p2 = (int(joints[j2][0]), int(joints[j2][1]))
            cv2.line(frame, p1, p2, (50, 220, 50), 2)


def list_images(image_dir):
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    image_paths.sort()
    return image_paths


def load_image_unicode_safe(path):
    # cv2.imread can fail on some Windows builds when the file path has
    # non-ASCII characters. Decode from raw bytes as a fallback-safe path.
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def benchmark_fps(args, cfg, model, device, image_size, normalize):
    image_dir = resolve_path(args.fps_image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError('FPS image directory not found: {}'.format(image_dir))

    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError('No image files found in {}'.format(image_dir))

    num_images = min(max(1, args.fps_num_images), len(image_paths))
    warmup_iters = max(0, args.fps_warmup)
    selected_paths = image_paths[:num_images]

    def run_one(path):
        image = load_image_unicode_safe(path)
        if image is None:
            return False
        input_tensor, _, _ = preprocess_frame(image, image_size, normalize)
        _ = model(input_tensor.to(device, non_blocking=(device.type == 'cuda')))
        return True

    skipped = 0
    warmup_done = 0
    with torch.no_grad():
        idx = 0
        while warmup_done < warmup_iters and idx < (num_images + warmup_iters) * 3:
            ok = run_one(selected_paths[idx % num_images])
            if ok:
                warmup_done += 1
            else:
                skipped += 1
            idx += 1
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_t = time.perf_counter()
        processed = 0
        for path in selected_paths:
            ok = run_one(path)
            if ok:
                processed += 1
            else:
                skipped += 1
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_t = time.perf_counter() - start_t

    if processed == 0:
        raise RuntimeError(
            'Could not read any benchmark image from {}. '
            'Verify dataset files and path encoding support.'.format(image_dir)
        )

    avg_ms = (total_t / processed) * 1000.0
    fps = processed / max(total_t, 1e-12)

    print('FPS benchmark complete')
    print('Device: {}'.format(device.type))
    print('Image dir: {}'.format(image_dir))
    print('Requested images: {}'.format(num_images))
    print('Processed images: {}'.format(processed))
    print('Skipped unreadable images: {}'.format(skipped))
    print('Warmup iterations: {}'.format(warmup_iters))
    print('Total inference time: {:.4f} s'.format(total_t))
    print('Average inference time: {:.3f} ms/image'.format(avg_ms))
    print('Average FPS: {:.2f}'.format(fps))


def main():
    args = parse_args()

    cfg_path = resolve_path(args.cfg)
    model_path = resolve_path(args.model_file)

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError('Config file not found: {}'.format(cfg_path))
    if not os.path.isfile(model_path):
        raise FileNotFoundError('Model file not found: {}'.format(model_path))

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA was requested, but torch.cuda.is_available() is False. '
                'Install a CUDA-enabled PyTorch build or run with --device cpu.'
            )
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    cfg.defrost()
    cfg.merge_from_file(cfg_path)
    cfg.TEST.POST_PROCESS = True
    cfg.freeze()

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = load_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_size = np.array(cfg.MODEL.IMAGE_SIZE, dtype=np.int32)

    if args.fps_only:
        benchmark_fps(args, cfg, model, device, image_size, normalize)
        return

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError('Could not open camera id {}'.format(args.camera_id))

    win_name = 'HRNet Webcam Demo'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    prev_t = time.time()
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            input_tensor, center, scale = preprocess_frame(frame, image_size, normalize)
            output = model(input_tensor.to(device))
            if isinstance(output, (list, tuple)):
                output = output[-1]

            heatmaps = output.detach().cpu().numpy()
            preds, maxvals = get_final_preds(
                cfg,
                heatmaps,
                np.expand_dims(center, axis=0),
                np.expand_dims(scale, axis=0),
            )

            joints = preds[0]
            scores = maxvals[0, :, 0]
            draw_pose(frame, joints, scores, args.score_thr)

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            cv2.putText(
                frame,
                'FPS: {:.1f}'.format(fps),
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                'Device: {}'.format(device.type),
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
