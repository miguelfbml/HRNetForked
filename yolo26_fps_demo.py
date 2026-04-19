from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO26 FPS benchmark on COCO val images')
    parser.add_argument(
        '--model-file',
        type=str,
        default='yolo26l-pose.pt',
        help='Path to YOLO26 model weights',
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/coco/images/val2017',
        help='Directory with benchmark images',
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=200,
        help='Number of images to benchmark',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=20,
        help='Warmup iterations before timing',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Inference device',
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(REPO_ROOT, path_value))


def list_images(image_dir: str):
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    image_paths.sort()
    return image_paths


def load_image_unicode_safe(path: str):
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def build_device(device_arg: str) -> str:
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA was requested, but torch.cuda.is_available() is False. '
                'Install a CUDA-enabled PyTorch build or run with --device cpu.'
            )
        torch.backends.cudnn.benchmark = True
        return 'cuda:0'
    return 'cpu'


def main():
    args = parse_args()

    resolved_model_path = resolve_path(args.model_file)
    model_source = resolved_model_path if os.path.isfile(resolved_model_path) else args.model_file
    image_dir = resolve_path(args.image_dir)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError('Image directory not found: {}'.format(image_dir))

    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError('No images found in {}'.format(image_dir))

    num_images = min(max(1, args.num_images), len(image_paths))
    warmup = max(0, args.warmup)
    selected = image_paths[:num_images]

    device = build_device(args.device)
    model = YOLO(model_source)

    def run_one(path: str) -> bool:
        image = load_image_unicode_safe(path)
        if image is None:
            return False
        _ = model.predict(image, device=device, verbose=False)
        return True

    skipped = 0
    warmup_done = 0
    with torch.no_grad():
        idx = 0
        while warmup_done < warmup and idx < (num_images + warmup) * 3:
            ok = run_one(selected[idx % num_images])
            if ok:
                warmup_done += 1
            else:
                skipped += 1
            idx += 1

        if device.startswith('cuda'):
            torch.cuda.synchronize()

        start_t = time.perf_counter()
        processed = 0
        for path in selected:
            ok = run_one(path)
            if ok:
                processed += 1
            else:
                skipped += 1

        if device.startswith('cuda'):
            torch.cuda.synchronize()
        total_t = time.perf_counter() - start_t

    if processed == 0:
        raise RuntimeError(
            'Could not read any benchmark image from {}. '
            'Verify dataset files and path encoding support.'.format(image_dir)
        )

    avg_ms = (total_t / processed) * 1000.0
    fps = processed / max(total_t, 1e-12)

    print('YOLO26 FPS benchmark complete')
    print('Model source: {}'.format(model_source))
    print('Device: {}'.format(device))
    print('Image dir: {}'.format(image_dir))
    print('Requested images: {}'.format(num_images))
    print('Processed images: {}'.format(processed))
    print('Skipped unreadable images: {}'.format(skipped))
    print('Warmup iterations: {}'.format(warmup))
    print('Total inference time: {:.4f} s'.format(total_t))
    print('Average inference time: {:.3f} ms/image'.format(avg_ms))
    print('Average FPS: {:.2f}'.format(fps))


if __name__ == '__main__':
    main()
