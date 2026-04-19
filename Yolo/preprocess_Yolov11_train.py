import argparse
import os
import cv2
import numpy as np
import glob
import gc
import torch
from ultralytics import YOLO
from tqdm import tqdm

class YOLO2DPoseEstimator:
    def __init__(self, model_path, img_size=640, device='auto'):
        if not os.path.exists(model_path):
            print(f"ERROR: YOLO model not found: {model_path}")
            exit(1)
        self.model = YOLO(model_path)
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else device
        self.model.to(self.device)
        self.img_size = img_size

    def estimate_2d_pose_from_image(self, image, target_width, target_height):
        results = self.model.predict(image, verbose=False, imgsz=self.img_size, conf=0.3, device=self.device, max_det=1)
        # Only process the first detected person (if any)
        if (results and len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0):
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
            pose_2d = np.zeros((17, 3), dtype=np.float32)
            n_kpts = min(17, keypoints.shape[0])
            img_height, img_width = image.shape[:2]
            scaled_keypoints = keypoints[:n_kpts].copy()
            scaled_keypoints[:, 0] = keypoints[:n_kpts, 0] * (target_width / img_width)
            scaled_keypoints[:, 1] = keypoints[:n_kpts, 1] * (target_height / img_height)
            pose_2d[:n_kpts, :2] = scaled_keypoints
            pose_2d[:n_kpts, 2] = conf[:n_kpts] if len(conf) >= n_kpts else 0.5
            return pose_2d, conf
        else:
            return np.zeros((17, 3), dtype=np.float32), np.zeros(17, dtype=np.float32)

    def close(self):
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

def get_sequence_image_dimensions(sequence_name):
    return (2048, 2048)

def load_original_train_dataset(path):
    if not os.path.exists(path):
        print(f"ERROR: Train dataset not found: {path}")
        return None
    print(f"✓ Loading train dataset from: {os.path.abspath(path)}")
    data = np.load(path, allow_pickle=True)['data'].item()
    print(f"Top-level keys: {list(data.keys())}")
    for k in data:
        print(f"  {k}: type={type(data[k])}, len={len(data[k]) if hasattr(data[k],'__len__') else 'N/A'}")
    return data

def load_sequence_images(subject, seq, camera_num):
    """Load images for a specific camera"""
    base_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp',
        '../motion3d/mpi_inf_3dhp',
        '../../../motion3d/mpi_inf_3dhp',
        'mpi_inf_3dhp'
    ]
    for base_path in base_paths:
        image_folder = os.path.join(base_path, subject, seq, 'imageFrames', f'video_{camera_num}')
        print(f"  Looking for images in: {image_folder}")
        if os.path.exists(image_folder):
            image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png")))
            print(f"    Found {len(image_files)} images")
            if image_files:
                return image_files
            else:
                print(f"    No images found in {image_folder}")
    print(f"    No image folder found for {subject} {seq} camera {camera_num}")
    return None

def create_yolo_train_dataset(original_data, estimator, output_path):
    print("Creating YOLO version of train dataset... (cams 0-8)")
    yolo_data = {}

    subjects = [f'S{i}' for i in range(1, 9)]
    seqs = ['Seq1', 'Seq2']
    cam_keys = ['0', '1', '2', '4', '5', '6', '7', '8']

    for key in original_data:
        try:
            subj, seq = key.split()
        except Exception as e:
            print(f"Skipping key {key}: {e}")
            continue
        if subj not in subjects or seq not in seqs:
            print(f"Skipping {key}: not in subjects/sequences filter")
            continue
        cam_list = original_data[key]
        if not isinstance(cam_list, list) or len(cam_list) == 0:
            print(f"  {key}: cam_list not found (type={type(cam_list)})")
            continue

        yolo_cam_dict = {}
        for cam_key in cam_keys:
            if cam_key not in cam_list[0]:
                print(f"  {key}: camera {cam_key} not found, skipping.")
                continue
            cam_data = cam_list[0][cam_key]
            print(f"Processing {subj} {seq} camera {cam_key}")

            if 'data_3d' in cam_data and 'data_2d' in cam_data:
                num_frames = len(cam_data['data_2d'])
                valid_flag = np.ones(num_frames, dtype=bool)
                yolo_cam_data = {
                    'data_3d': cam_data['data_3d'].copy(),
                    'valid': valid_flag,
                    'camera': None,
                    'confidences': []  # Store confidences here
                }
                original_2d = cam_data['data_2d']
            else:
                print(f"ERROR: Missing expected keys in cam_data for {subj} {seq} camera {cam_key}: {list(cam_data.keys())}")
                continue

            orig_width, orig_height = get_sequence_image_dimensions(subj)
            # Pass camera number to load_sequence_images
            image_files = load_sequence_images(subj, seq, cam_key)
            if image_files is None:
                print(f"  Skipping {subj} {seq} camera {cam_key}: No images found")
                continue

            num_frames = len(original_2d)
            yolo_poses_2d = []
            detection_failures = 0

            for frame_idx in tqdm(range(num_frames), desc=f"{subj} {seq} cam{cam_key}"):
                if frame_idx < len(image_files):
                    image = cv2.imread(image_files[frame_idx])
                    if image is not None:
                        pose_2d_with_conf, conf = estimator.estimate_2d_pose_from_image(image, orig_width, orig_height)
                        if np.all(pose_2d_with_conf[:, :2] == 0):
                            yolo_cam_data['valid'][frame_idx] = False
                            detection_failures += 1
                        yolo_poses_2d.append(pose_2d_with_conf[:, :2])
                        yolo_cam_data['confidences'].append(conf.tolist())
                    else:
                        yolo_poses_2d.append(np.zeros((17, 2), dtype=np.float32))
                        yolo_cam_data['valid'][frame_idx] = False
                        yolo_cam_data['confidences'].append([0.0]*17)
                        detection_failures += 1
                else:
                    yolo_poses_2d.append(np.zeros((17, 2), dtype=np.float32))
                    yolo_cam_data['valid'][frame_idx] = False
                    yolo_cam_data['confidences'].append([0.0]*17)
                    detection_failures += 1

            yolo_cam_data['data_2d'] = np.array(yolo_poses_2d, dtype=np.float32)
            print(f"  ✓ YOLO 2D shape: {yolo_cam_data['data_2d'].shape}")
            print(f"  ✓ Detection failures: {detection_failures}/{num_frames} ({detection_failures/num_frames*100:.1f}%)")
            print(f"  ✓ Valid frames: {np.sum(yolo_cam_data['valid'])}/{num_frames}")

            yolo_cam_dict[cam_key] = yolo_cam_data

            del yolo_poses_2d
            gc.collect()
            if estimator.device.startswith('cuda'):
                torch.cuda.empty_cache()

        if yolo_cam_dict:
            yolo_data[key] = [yolo_cam_dict]

    print(f"\nSaving YOLO train dataset to: {output_path}")
    if not yolo_data:
        print("No data processed! Check your image paths and .npz structure.")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(output_path, data=yolo_data)
        print("✓ YOLO train dataset created successfully!")
    return yolo_data

def main():
    parser = argparse.ArgumentParser(description='Create YOLO version of MPI-INF-3DHP train dataset (S1-S8, Seq1/Seq2 all cams)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--train-data-path', type=str, default='./data_train_3dhp.npz', help='Path to original train dataset')
    parser.add_argument('--output-path', type=str, default='./data_train_3dhp_yolo_all.npz', help='Output path for YOLO train dataset')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size for YOLO inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, etc.)')
    args = parser.parse_args()

    print("Creating YOLO version of MPI-INF-3DHP train dataset (S1-S8, Seq1/Seq2, all cameras)")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Train data path: {args.train_data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")

    original_data = load_original_train_dataset(args.train_data_path)
    if original_data is None:
        return

    estimator = YOLO2DPoseEstimator(args.model_path, args.img_size, args.device)
    try:
        yolo_data = create_yolo_train_dataset(original_data, estimator, args.output_path)
        if yolo_data:
            print(f"\n✓ Success! YOLO train dataset saved to: {args.output_path}")
            print("To use in training, point your config to this file or rename it to replace the original.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        estimator.close()
        print("✓ YOLO estimator closed")

if __name__ == '__main__':
    main()