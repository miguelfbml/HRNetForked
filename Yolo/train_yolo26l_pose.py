"""
Train YOLO26l-pose on MPI-INF-3DHP dataset for 17 keypoint pose estimation
Trains from scratch (no pretrained weights loaded)

Usage:
python train_yolo26l_pose.py --epochs 100 --batch-size 4 --img-size 1280
"""


# --- BEGIN: Imports and Classes from train.py ---
import argparse
import os
import shutil
import glob
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
import yaml
import cv2
from tqdm import tqdm
import threading
import time
from statistics import mean
try:
    wandb = __import__("wandb")
except ImportError:
    wandb = None

# GPU Utilization Monitor
class GPUUtilizationMonitor:
    def __init__(self, device_idx=0):
        try:
            pynvml = __import__("pynvml")
            pynvml.nvmlInit()
            self.device = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self.pynvml = pynvml
            self.utilization_rates = []
            self.memory_usage = []
            self.running = False
            self.thread = None
            self.enabled = True
        except:
            self.pynvml = None
            self.enabled = False
            print("Warning: GPU monitoring not available")

    def start(self):
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        pynvml = self.pynvml
        if pynvml is None:
            return
        while self.running and self.enabled:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
                self.utilization_rates.append(util.gpu)
                self.memory_usage.append(memory_info.used / memory_info.total * 100)
                time.sleep(0.1)
            except:
                break

    def stop(self):
        if not self.enabled:
            return
        self.running = False
        if self.thread:
            self.thread.join()
        try:
            if self.pynvml is not None:
                self.pynvml.nvmlShutdown()
        except:
            pass

    def get_stats(self):
        if not self.enabled:
            return 0, 0
        avg_util = mean(self.utilization_rates) if self.utilization_rates else 0
        avg_mem = mean(self.memory_usage) if self.memory_usage else 0
        self.utilization_rates.clear()
        self.memory_usage.clear()
        return avg_util, avg_mem

# Enhanced Metrics Tracker
class YOLOMetricsTracker:
    def __init__(self, use_wandb=False, wandb_project="YOLO_MPI_Training"):
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.gpu_monitor = GPUUtilizationMonitor()
        self.epoch_metrics = {}
        self.training_start_time = None
        self.epoch_times = []
        if self.use_wandb and wandb is None:
            print("Warning: wandb is not installed; disabling WandB logging")
            self.use_wandb = False
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name="YOLO26l_MPI_3DHP_Training",
                tags=["YOLO26l", "MPI-INF-3DHP", "pose_estimation"]
            )

    def start_training(self):
        self.training_start_time = time.time()
        self.gpu_monitor.start()
        print(f"\n{'='*70}")
        print(f"ENHANCED TRAINING MONITORING STARTED")
        print(f"{'='*70}")

    def log_epoch_metrics(self, epoch, results_dict, model_path=None):
        epoch_time = time.time()
        gpu_util, gpu_mem = self.gpu_monitor.get_stats()
        train_loss = results_dict.get('train/loss', 0)
        val_loss = results_dict.get('val/loss', 0)
        train_pose_loss = results_dict.get('train/pose_loss', results_dict.get('train/kobj_loss', 0))
        train_kobj_loss = results_dict.get('train/kobj_loss', 0)
        val_pose_loss = results_dict.get('val/pose_loss', results_dict.get('val/kobj_loss', 0))
        val_kobj_loss = results_dict.get('val/kobj_loss', 0)
        precision = results_dict.get('metrics/precision', 0)
        recall = results_dict.get('metrics/recall', 0)
        map50 = results_dict.get('metrics/mAP50', 0)
        map50_95 = results_dict.get('metrics/mAP50-95', 0)
        pose_precision = results_dict.get('metrics/pose_precision', 0)
        pose_recall = results_dict.get('metrics/pose_recall', 0)
        pose_map50 = results_dict.get('metrics/pose_mAP50', 0)
        pose_map50_95 = results_dict.get('metrics/pose_mAP50-95', 0)
        flops_per_image = 0
        epoch_metrics = {
            'epoch': epoch,
            'epoch_time': epoch_time - (self.epoch_times[-1] if self.epoch_times else self.training_start_time),
            'gpu_utilization': gpu_util,
            'gpu_memory_usage': gpu_mem,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'train/pose_loss': train_pose_loss,
            'train/kobj_loss': train_kobj_loss,
            'val/pose_loss': val_pose_loss,
            'val/kobj_loss': val_kobj_loss,
            'metrics/precision': precision,
            'metrics/recall': recall,
            'metrics/mAP50': map50,
            'metrics/mAP50-95': map50_95,
            'metrics/pose_precision': pose_precision,
            'metrics/pose_recall': pose_recall,
            'metrics/pose_mAP50': pose_map50,
            'metrics/pose_mAP50-95': pose_map50_95,
            'flops_per_image': flops_per_image
        }
        self.epoch_metrics[epoch] = epoch_metrics
        self.epoch_times.append(epoch_time)
        self.print_epoch_summary(epoch, epoch_metrics)
        if self.use_wandb:
            try:
                wandb.log(epoch_metrics, step=epoch)
                print(f"✓ Logged metrics to WandB for epoch {epoch}")
            except Exception as e:
                print(f"⚠ Failed to log to WandB: {e}")

    def print_epoch_summary(self, epoch, metrics):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} COMPREHENSIVE METRICS")
        print(f"{'='*70}")
        print(f"Performance Metrics:")
        print(f"  Epoch Time: {metrics['epoch_time']:.2f} seconds")
        print(f"  GPU Utilization: {metrics['gpu_utilization']:.2f}%")
        print(f"  GPU Memory Usage: {metrics['gpu_memory_usage']:.2f}%")
        print(f"\nTraining Losses:")
        print(f"  Total Loss: {metrics['train/loss']:.4f}")
        print(f"  Pose Loss: {metrics['train/pose_loss']:.4f}")
        print(f"  Keypoint Obj Loss: {metrics['train/kobj_loss']:.4f}")
        print(f"\nValidation Losses:")
        print(f"  Total Loss: {metrics['val/loss']:.4f}")
        print(f"  Pose Loss: {metrics['val/pose_loss']:.4f}")
        print(f"  Keypoint Obj Loss: {metrics['val/kobj_loss']:.4f}")
        print(f"\nDetection Metrics:")
        print(f"  Precision: {metrics['metrics/precision']:.4f}")
        print(f"  Recall: {metrics['metrics/recall']:.4f}")
        print(f"  mAP@0.5: {metrics['metrics/mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['metrics/mAP50-95']:.4f}")
        print(f"\nPose Estimation Metrics:")
        print(f"  Pose Precision: {metrics['metrics/pose_precision']:.4f}")
        print(f"  Pose Recall: {metrics['metrics/pose_recall']:.4f}")
        print(f"  Pose mAP@0.5: {metrics['metrics/pose_mAP50']:.4f}")
        print(f"  Pose mAP@0.5:0.95: {metrics['metrics/pose_mAP50-95']:.4f}")
        print(f"{'='*70}")

    def finish_training(self):
        self.gpu_monitor.stop()
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            print(f"\n{'='*70}")
            print(f"ENHANCED TRAINING COMPLETED - FINAL SUMMARY")
            print(f"{'='*70}")
            print(f"Total Training Time: {total_time/3600:.2f} hours")
            print(f"Total Epochs: {len(self.epoch_metrics)}")
            print(f"Average Epoch Time: {np.mean([m['epoch_time'] for m in self.epoch_metrics.values()]):.2f} seconds")
            if self.epoch_metrics:
                best_epoch = min(self.epoch_metrics.keys(), 
                               key=lambda k: self.epoch_metrics[k]['val/loss'] if self.epoch_metrics[k]['val/loss'] > 0 else float('inf'))
                best_metrics = self.epoch_metrics[best_epoch]
                print(f"\nBest Epoch: {best_epoch}")
                print(f"  Best Val Loss: {best_metrics['val/loss']:.4f}")
                print(f"  Best Pose mAP@0.5: {best_metrics['metrics/pose_mAP50']:.4f}")
                print(f"  Best Pose mAP@0.5:0.95: {best_metrics['metrics/pose_mAP50-95']:.4f}")
            print(f"{'='*70}")
        if self.use_wandb:
            final_summary = {
                "final/total_training_time_hours": total_time/3600,
                "final/total_epochs": len(self.epoch_metrics),
                "final/avg_epoch_time": np.mean([m['epoch_time'] for m in self.epoch_metrics.values()]) if self.epoch_metrics else 0,
            }
            wandb.log(final_summary)
            print("✓ Final summary logged to WandB")
            wandb.finish()

# --- END: Imports and Classes from train.py ---

# --- Dataset Converter (minimal, as before) ---
class MPIDatasetConverter:
    def __init__(self, base_path, annotations_path, output_path):
        self.base_path = base_path
        self.annotations_path = annotations_path
        self.output_path = output_path
        self.train_images_path = os.path.join(output_path, 'images', 'train')
        self.val_images_path = os.path.join(output_path, 'images', 'val')
        self.train_labels_path = os.path.join(output_path, 'labels', 'train')
        self.val_labels_path = os.path.join(output_path, 'labels', 'val')
        os.makedirs(self.train_images_path, exist_ok=True)
        os.makedirs(self.val_images_path, exist_ok=True)
        os.makedirs(self.train_labels_path, exist_ok=True)
        os.makedirs(self.val_labels_path, exist_ok=True)

    def is_dataset_processed(self):
        yaml_path = os.path.join(self.output_path, 'mpi_dataset.yaml')
        if not os.path.exists(yaml_path):
            return False, "YAML config not found"
        train_images = list(Path(self.train_images_path).glob("*.jpg"))
        train_labels = list(Path(self.train_labels_path).glob("*.txt"))
        val_images = list(Path(self.val_images_path).glob("*.jpg"))
        val_labels = list(Path(self.val_labels_path).glob("*.txt"))
        if len(train_images) == 0 or len(train_labels) == 0 or len(val_images) == 0 or len(val_labels) == 0:
            return False, "No training images or labels found"
        if len(train_images) != len(train_labels):
            return False, "Mismatch in train images/labels count"
        if len(val_images) != len(val_labels):
            return False, "Mismatch in val images/labels count"
        return True, f"Dataset found: {len(train_images)} train images"

    def load_annotations(self):
        print(f"Loading annotations from: {self.annotations_path}")
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        data = np.load(self.annotations_path, allow_pickle=True)['data'].item()
        print(f"Loaded training annotations for {len(data)} sequences")
        return data

    def load_test_annotations(self):
        test_annotations_path = self.annotations_path.replace('data_train_3dhp.npz', 'data_test_3dhp.npz')
        if not os.path.exists(test_annotations_path):
            print(f"Test annotations not found: {test_annotations_path}")
            return {}
        data = np.load(test_annotations_path, allow_pickle=True)['data'].item()
        print(f"Loaded test annotations for {len(data)} sequences")
        return data

    def normalize_keypoints(self, keypoints_2d, img_width, img_height):
        normalized_kpts = keypoints_2d.copy()
        if np.max(keypoints_2d[:, :2]) <= 1.0:
            normalized_kpts[:, 0] = keypoints_2d[:, 0] * img_width
            normalized_kpts[:, 1] = keypoints_2d[:, 1] * img_height
        normalized_kpts[:, 0] = np.clip(normalized_kpts[:, 0] / img_width, 0, 1)
        normalized_kpts[:, 1] = np.clip(normalized_kpts[:, 1] / img_height, 0, 1)
        return normalized_kpts

    def create_yolo_annotation(self, keypoints_2d, img_width, img_height, confidence_threshold=0.1):
        norm_kpts = self.normalize_keypoints(keypoints_2d, img_width, img_height)
        visible_kpts = norm_kpts[norm_kpts[:, 2] > confidence_threshold] if norm_kpts.shape[1] > 2 else norm_kpts
        if len(visible_kpts) == 0:
            return None

        x_coords = visible_kpts[:, 0]
        y_coords = visible_kpts[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        padding = 0.10
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - padding * width)
        x_max = min(1, x_max + padding * width)
        y_min = max(0, y_min - padding * height)
        y_max = min(1, y_max + padding * height)

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        center_x = x_min + bbox_width / 2
        center_y = y_min + bbox_height / 2

        keypoint_str = ""
        for i in range(17):
            if i < len(norm_kpts):
                x, y = norm_kpts[i, 0], norm_kpts[i, 1]
                visibility = 2 if (norm_kpts.shape[1] > 2 and norm_kpts[i, 2] > confidence_threshold) else 0
                keypoint_str += f" {x:.6f} {y:.6f} {visibility}"
            else:
                keypoint_str += " 0.0 0.0 0"

        return f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}{keypoint_str}"

    def process_training_data(self, annotations):
        print("\nProcessing training data...")
        processed_count = 0
        skipped_count = 0

        for seq_name, seq_data in tqdm(annotations.items(), desc="Processing training sequences"):
            if not isinstance(seq_data, list) or len(seq_data) < 1:
                continue

            camera_dict = seq_data[0]
            parts = seq_name.split(' ')
            if len(parts) != 2:
                continue
            subject, sequence = parts

            camera_keys = sorted(
                [k for k in camera_dict.keys() if isinstance(camera_dict[k], dict) and 'data_2d' in camera_dict[k]],
                key=lambda x: int(x) if str(x).isdigit() else str(x)
            )

            for cam_key in camera_keys:
                camera_data = camera_dict[cam_key]
                poses_2d = camera_data['data_2d']
                image_folder = os.path.join(self.base_path, subject, sequence, 'imageFrames', f'video_{cam_key}')
                if not os.path.exists(image_folder):
                    continue

                image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
                image_files.extend(glob.glob(os.path.join(image_folder, "*.JPG")))
                image_files.sort()
                if not image_files:
                    continue

                max_frames = min(len(image_files), len(poses_2d))
                for frame_idx in range(max_frames):
                    try:
                        img_path = image_files[frame_idx]
                        image = cv2.imread(img_path)
                        if image is None:
                            skipped_count += 1
                            continue

                        img_height, img_width = image.shape[:2]
                        pose_2d = poses_2d[frame_idx]
                        if pose_2d.shape[1] == 2:
                            confidence = np.ones((pose_2d.shape[0], 1)) * 0.9
                            pose_2d = np.hstack([pose_2d, confidence])

                        annotation = self.create_yolo_annotation(pose_2d, img_width, img_height)
                        if annotation is None:
                            skipped_count += 1
                            continue

                        img_name = f"{subject}_{sequence}_cam{cam_key}_frame{frame_idx:06d}.jpg"
                        label_name = f"{subject}_{sequence}_cam{cam_key}_frame{frame_idx:06d}.txt"

                        cv2.imwrite(os.path.join(self.train_images_path, img_name), image)
                        with open(os.path.join(self.train_labels_path, label_name), 'w') as f:
                            f.write(annotation + '\n')

                        processed_count += 1
                    except Exception:
                        skipped_count += 1

        print(f"Training split: wrote {processed_count} samples, skipped {skipped_count}")

    def process_test_data(self, test_annotations):
        print("\nProcessing validation data from MPI test split...")
        if not test_annotations:
            print("No test annotations available, skipping validation creation")
            return

        processed_count = 0
        skipped_count = 0
        test_base_paths = [
            '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
            '../motion3d/mpi_inf_3dhp_test_set',
            '../../motion3d/mpi_inf_3dhp_test_set'
        ]

        for seq_name, seq_data in tqdm(test_annotations.items(), desc="Processing test sequences"):
            image_folder = None
            for base_path in test_base_paths:
                potential_path = os.path.join(base_path, seq_name, 'imageSequence')
                if os.path.exists(potential_path):
                    image_folder = potential_path
                    break

            if image_folder is None:
                continue

            image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
            image_files.sort()
            if not image_files:
                continue

            poses_2d = seq_data['data_2d']
            max_frames = min(len(image_files), len(poses_2d))

            for img_idx in range(max_frames):
                try:
                    image = cv2.imread(image_files[img_idx])
                    if image is None:
                        skipped_count += 1
                        continue

                    img_height, img_width = image.shape[:2]
                    pose_2d = poses_2d[img_idx]
                    if pose_2d.shape[1] == 2:
                        confidence = np.ones((pose_2d.shape[0], 1)) * 0.9
                        pose_2d = np.hstack([pose_2d, confidence])

                    annotation = self.create_yolo_annotation(pose_2d, img_width, img_height)
                    if annotation is None:
                        skipped_count += 1
                        continue

                    img_name = f"{seq_name}_frame{img_idx:06d}.jpg"
                    label_name = f"{seq_name}_frame{img_idx:06d}.txt"

                    cv2.imwrite(os.path.join(self.val_images_path, img_name), image)
                    with open(os.path.join(self.val_labels_path, label_name), 'w') as f:
                        f.write(annotation + '\n')

                    processed_count += 1
                except Exception:
                    skipped_count += 1

        print(f"Validation split: wrote {processed_count} samples, skipped {skipped_count}")

    def create_dataset_yaml(self):
        dataset_config = {
            'path': os.path.abspath(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['person'],
            'kpt_shape': [17, 3],
            'flip_idx': [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14, 15, 16]
        }
        yaml_path = os.path.join(self.output_path, 'mpi_dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        return yaml_path

    def convert_dataset(self, force_reprocess=False):
        is_processed, status_msg = self.is_dataset_processed()
        if is_processed and not force_reprocess:
            print(f"Using existing converted dataset: {status_msg}")
            return os.path.join(self.output_path, 'mpi_dataset.yaml')

        if force_reprocess:
            print("Force reprocess enabled; rebuilding train/val images and labels")

        train_annotations = self.load_annotations()
        test_annotations = self.load_test_annotations()
        self.process_training_data(train_annotations)
        self.process_test_data(test_annotations)
        return self.create_dataset_yaml()

# --- Training function with callbacks, metrics, wandb, GPU monitoring ---
def train_yolo26l_pose(dataset_yaml, args):
    print("\n==============================")
    print("STARTING YOLO26l-pose TRAINING (from scratch)")
    print("==============================")
    metrics_tracker = YOLOMetricsTracker(
        use_wandb=args.use_wandb,
        wandb_project=getattr(args, 'wandb_project', 'YOLO26l_MPI_3DHP_Training')
    )
    metrics_tracker.start_training()
    model = YOLO('yolo26l-pose.yaml')
    print(f"Training config: epochs={args.epochs}, batch={args.batch_size}, imgsz={args.img_size}, device={args.device}")
    training_config = {
        'data': dataset_yaml,
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'device': args.device,
        'workers': args.workers,
        'project': 'runs/pose',
        'name': 'mpi_yolo26l_pose_scratch',
        'patience': args.patience,
        'cache': args.cache,
        'pose': 17.0,
        'pretrained': False
    }
    if args.use_wandb:
        wandb.config.update({
            "model": "YOLO26l-pose",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "pose_loss_weight": 17.0,
            "device": args.device,
            "dataset": "MPI-INF-3DHP",
            "keypoints": 17,
            "cache": args.cache,
            "patience": args.patience,
        })
    def on_train_epoch_end(trainer):
        try:
            epoch = trainer.epoch + 1
            results_dict = {}
            try:
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    loss_items = trainer.loss_items
                    if isinstance(loss_items, (list, tuple)) and len(loss_items) >= 3:
                        results_dict['train/box_loss'] = float(loss_items[0])
                        results_dict['train/cls_loss'] = float(loss_items[1])
                        results_dict['train/kobj_loss'] = float(loss_items[2])
                        results_dict['train/loss'] = float(sum(loss_items))
                elif hasattr(trainer, 'loss') and trainer.loss is not None:
                    results_dict['train/loss'] = float(trainer.loss.item()) if hasattr(trainer.loss, 'item') else float(trainer.loss)
            except Exception as e:
                print(f"⚠ Could not extract training losses: {e}")
            try:
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    for key, value in trainer.metrics.items():
                        if isinstance(value, (int, float)):
                            results_dict[f'val/{key}'] = float(value)
                        elif hasattr(value, 'item'):
                            results_dict[f'val/{key}'] = float(value.item())
            except Exception as e:
                print(f"⚠ Could not extract validation metrics: {e}")
            try:
                metrics_tracker.log_epoch_metrics(epoch, results_dict)
            except Exception as e:
                print(f"⚠ Failed to log epoch metrics: {e}")
        except Exception as e:
            print(f"⚠ Error in epoch callback: {e}")
    def on_val_end(trainer):
        try:
            if hasattr(trainer, 'metrics') and trainer.metrics:
                val_metrics = {}
                for key, value in trainer.metrics.items():
                    if isinstance(value, (int, float)):
                        val_metrics[f'val/{key}'] = float(value)
                if args.use_wandb and val_metrics:
                    wandb.log(val_metrics)
        except Exception as e:
            print(f"⚠ Error in validation callback: {e}")
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    try:
        results = model.train(**training_config)
        print("\n✅ Training completed!")
        print("Model saved to:", 'runs/pose/mpi_yolo26l_pose_scratch/weights/')
        return results
    finally:
        metrics_tracker.finish_training()

# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description='YOLO26l-pose training from scratch on MPI-INF-3DHP')
    parser.add_argument('--base-path', type=str, default='/nas-ctm01/datasets/public/mpi_inf_3dhp', help='Base path to MPI-INF-3DHP dataset')
    parser.add_argument('--annotations-path', type=str, default='../../motion3d/data_train_3dhp.npz', help='Path to training annotations file')
    parser.add_argument('--output-path', type=str, default='/nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo', help='Output path for converted dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--device', type=str, default='0', help='Device to use for training (0, 1, 2, etc. or cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--cache', type=str, default='disk', help='Cache strategy: "ram", "disk", or False')
    parser.add_argument('--convert-only', action='store_true', help='Only convert dataset, do not train')
    parser.add_argument('--train-only', action='store_true', help='Only train (assume dataset already converted)')
    parser.add_argument('--force-reprocess', action='store_true', help='Force reprocessing even if dataset exists')
    parser.add_argument('--use-wandb', action='store_true', help='Enable WandB logging for monitoring')
    parser.add_argument('--wandb-project', type=str, default='YOLO26l_MPI_3DHP_Training', help='WandB project name')
    args = parser.parse_args()
    if not args.train_only:
        converter = MPIDatasetConverter(args.base_path, args.annotations_path, args.output_path)
        dataset_yaml = converter.convert_dataset(force_reprocess=args.force_reprocess)
    else:
        dataset_yaml = os.path.join(args.output_path, 'mpi_dataset.yaml')
        if not os.path.exists(dataset_yaml):
            print(f"ERROR: Dataset config not found: {dataset_yaml}")
            return
    if not args.convert_only:
        train_yolo26l_pose(dataset_yaml, args)
    print("\n🎉 Training process completed!")

if __name__ == '__main__':
    main()
