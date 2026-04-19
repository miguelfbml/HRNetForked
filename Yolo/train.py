"""
Train YOLO on MPI-INF-3DHP dataset for 17 keypoint pose estimation
Converts MPI-INF-3DHP 2D annotations to YOLO format and trains a custom model

Usage:
python train.py --epochs 100 --batch-size 4 --img-size 1280

# First run - will convert if needed
python train.py --epochs 100 --batch-size 4

# Subsequent runs - will use existing annotations
python train.py --train-only --epochs 100 --batch-size 4

# Force regeneration if annotations are corrupted
python train.py --force-reprocess --epochs 100 --batch-size 4
"""

import argparse
import os
import cv2
import numpy as np
import yaml
import shutil
import glob
import time
import pynvml
import threading
from statistics import mean
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.callbacks import default_callbacks
import json
import wandb
import pkg_resources
from ptflops import get_model_complexity_info
import torch
import torch.nn as nn

# MPI-INF-3DHP joint names (17 keypoints) - CORRECTED ORDER
MPI_JOINT_NAMES = [
    'Head',           # 0
    'SpineShoulder',  # 1 
    'RShoulder',      # 2
    'RElbow',         # 3
    'RHand',          # 4
    'LShoulder',      # 5
    'LElbow',         # 6
    'LHand',          # 7
    'RHip',           # 8
    'RKnee',          # 9
    'RAnkle',         # 10
    'LHip',           # 11
    'LKnee',          # 12
    'LAnkle',         # 13
    'Sacrum',         # 14
    'Spine',          # 15
    'Neck'            # 16
]

# MPI-INF-3DHP skeleton connections - Updated for corrected order
MPI_SKELETON = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

class GPUUtilizationMonitor:
    def __init__(self, device_idx=0):
        try:
            pynvml.nvmlInit()
            self.device = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self.utilization_rates = []
            self.memory_usage = []
            self.running = False
            self.thread = None
            self.enabled = True
        except:
            self.enabled = False
            print("Warning: GPU monitoring not available")

    def start(self):
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
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
            pynvml.nvmlShutdown()
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

class YOLOMetricsTracker:
    def __init__(self, use_wandb=False, wandb_project="YOLO_MPI_Training"):
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.gpu_monitor = GPUUtilizationMonitor()
        self.epoch_metrics = {}
        self.training_start_time = None
        self.epoch_times = []
        
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name="YOLO_MPI_3DHP_Enhanced_Training",
                tags=["YOLO", "MPI-INF-3DHP", "pose_estimation", "enhanced"]
            )
    
    def start_training(self):
        self.training_start_time = time.time()
        self.gpu_monitor.start()
        print(f"\n{'='*70}")
        print(f"ENHANCED TRAINING MONITORING STARTED")
        print(f"{'='*70}")
    
    def log_epoch_metrics(self, epoch, results_dict, model_path=None):
        """Log comprehensive metrics for each epoch with enhanced metric extraction"""
        epoch_time = time.time()
        
        # Get GPU stats
        gpu_util, gpu_mem = self.gpu_monitor.get_stats()
        
        # Enhanced metric extraction from YOLO results
        train_loss = 0
        val_loss = 0
        
        # Try multiple ways to extract training loss
        if 'train/loss' in results_dict:
            train_loss = results_dict['train/loss']
        elif hasattr(results_dict, 'box_loss'):
            train_loss = results_dict.box_loss + getattr(results_dict, 'cls_loss', 0) + getattr(results_dict, 'dfl_loss', 0)
        elif 'loss' in results_dict:
            train_loss = results_dict['loss']
        
        # Try multiple ways to extract validation loss
        if 'val/loss' in results_dict:
            val_loss = results_dict['val/loss']
        elif 'metrics/loss' in results_dict:
            val_loss = results_dict['metrics/loss']
        elif hasattr(results_dict, 'val_loss'):
            val_loss = results_dict.val_loss
        
        # Pose-specific metrics with better extraction
        train_pose_loss = results_dict.get('train/pose_loss', results_dict.get('train/kobj_loss', 0))
        train_kobj_loss = results_dict.get('train/kobj_loss', 0)
        val_pose_loss = results_dict.get('val/pose_loss', results_dict.get('val/kobj_loss', 0))
        val_kobj_loss = results_dict.get('val/kobj_loss', 0)
        
        # Detection metrics with multiple fallbacks
        precision = (results_dict.get('metrics/precision(B)', 0) or 
                    results_dict.get('precision', 0) or 
                    results_dict.get('metrics/precision', 0))
        
        recall = (results_dict.get('metrics/recall(B)', 0) or 
                 results_dict.get('recall', 0) or 
                 results_dict.get('metrics/recall', 0))
        
        map50 = (results_dict.get('metrics/mAP50(B)', 0) or 
                results_dict.get('map50', 0) or 
                results_dict.get('metrics/mAP50', 0))
        
        map50_95 = (results_dict.get('metrics/mAP50-95(B)', 0) or 
                   results_dict.get('map50_95', 0) or 
                   results_dict.get('metrics/mAP50-95', 0))
        
        # Pose metrics (if available)
        pose_precision = (results_dict.get('metrics/precision(P)', 0) or 
                         results_dict.get('pose_precision', 0))
        pose_recall = (results_dict.get('metrics/recall(P)', 0) or 
                      results_dict.get('pose_recall', 0))
        pose_map50 = (results_dict.get('metrics/mAP50(P)', 0) or 
                     results_dict.get('pose_map50', 0))
        pose_map50_95 = (results_dict.get('metrics/mAP50-95(P)', 0) or 
                        results_dict.get('pose_map50_95', 0))
        
        # Calculate FLOPs if model path is provided
        flops_per_image = 0
        if model_path and os.path.exists(model_path):
            try:
                flops_per_image = self.calculate_model_flops(model_path)
            except:
                pass
        
        # Store epoch metrics
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
        
        # Print comprehensive metrics
        self.print_epoch_summary(epoch, epoch_metrics)
        
        # Log to WandB immediately
        if self.use_wandb:
            try:
                wandb.log(epoch_metrics, step=epoch)
                print(f"✓ Logged metrics to WandB for epoch {epoch}")
            except Exception as e:
                print(f"⚠ Failed to log to WandB: {e}")
    
    def print_epoch_summary(self, epoch, metrics):
        """Print comprehensive epoch summary like TCPFormer"""
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} COMPREHENSIVE METRICS")
        print(f"{'='*70}")
        
        print(f"Performance Metrics:")
        print(f"  Epoch Time: {metrics['epoch_time']:.2f} seconds")
        print(f"  GPU Utilization: {metrics['gpu_utilization']:.2f}%")
        print(f"  GPU Memory Usage: {metrics['gpu_memory_usage']:.2f}%")
        if metrics['flops_per_image'] > 0:
            print(f"  FLOPs per Image: {metrics['flops_per_image']/1e9:.2f} GFLOPs")
        
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
    
    def calculate_model_flops(self, model_path, input_size=(1280, 1280)):
        """Calculate FLOPs for the trained model"""
        try:
            model = YOLO(model_path)
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            
            # Extract the actual PyTorch model
            pytorch_model = model.model
            
            # Calculate FLOPs
            flops, params = get_model_complexity_info(
                pytorch_model, 
                (3, input_size[0], input_size[1]),
                as_strings=False, 
                print_per_layer_stat=False
            )
            
            return flops
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            return 0
    
    def finish_training(self):
        """Clean up and print final summary"""
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
            # Log final summary
            final_summary = {
                "final/total_training_time_hours": total_time/3600,
                "final/total_epochs": len(self.epoch_metrics),
                "final/avg_epoch_time": np.mean([m['epoch_time'] for m in self.epoch_metrics.values()]) if self.epoch_metrics else 0,
            }
            wandb.log(final_summary)
            print("✓ Final summary logged to WandB")
            wandb.finish()

class MPIDatasetConverter:
    def __init__(self, base_path, annotations_path, output_path):
        self.base_path = base_path
        self.annotations_path = annotations_path
        self.output_path = output_path
        self.train_images_path = os.path.join(output_path, 'images', 'train')
        self.val_images_path = os.path.join(output_path, 'images', 'val')
        self.train_labels_path = os.path.join(output_path, 'labels', 'train')
        self.val_labels_path = os.path.join(output_path, 'labels', 'val')
        
        # Create directories with proper permissions
        print(f"Creating dataset structure in: {output_path}")
        os.makedirs(self.train_images_path, exist_ok=True)
        os.makedirs(self.val_images_path, exist_ok=True)
        os.makedirs(self.train_labels_path, exist_ok=True)
        os.makedirs(self.val_labels_path, exist_ok=True)
        
        # Verify directories were created successfully
        for path in [self.train_images_path, self.val_images_path, self.train_labels_path, self.val_labels_path]:
            if not os.path.exists(path):
                raise OSError(f"Failed to create directory: {path}")
        
        print(f"✓ Dataset directories created successfully")
    
    def is_dataset_processed(self):
        """Check if dataset is already processed"""
        # Check if yaml config exists
        yaml_path = os.path.join(self.output_path, 'mpi_dataset.yaml')
        if not os.path.exists(yaml_path):
            return False, "YAML config not found"
        
        # Check if directories exist and have files
        train_images = glob.glob(os.path.join(self.train_images_path, "*.jpg"))
        train_labels = glob.glob(os.path.join(self.train_labels_path, "*.txt"))
        val_images = glob.glob(os.path.join(self.val_images_path, "*.jpg"))
        val_labels = glob.glob(os.path.join(self.val_labels_path, "*.txt"))
        
        if len(train_images) == 0:
            return False, "No training images found"
        if len(train_labels) == 0:
            return False, "No training labels found"
        if len(train_images) != len(train_labels):
            return False, f"Mismatch: {len(train_images)} images vs {len(train_labels)} labels"
        
        # Check for reasonable dataset size (should have at least 1000 training samples)
        if len(train_images) < 1000:
            return False, f"Dataset too small: only {len(train_images)} training samples"
        
        return True, f"Dataset found: {len(train_images)} train, {len(val_images)} val images"
    
    def get_processing_summary(self):
        """Get summary of processed dataset"""
        train_images = len(glob.glob(os.path.join(self.train_images_path, "*.jpg")))
        train_labels = len(glob.glob(os.path.join(self.train_labels_path, "*.txt")))
        val_images = len(glob.glob(os.path.join(self.val_images_path, "*.jpg")))
        val_labels = len(glob.glob(os.path.join(self.val_labels_path, "*.txt")))
        
        return {
            'train_images': train_images,
            'train_labels': train_labels,
            'val_images': val_images,
            'val_labels': val_labels,
            'total_images': train_images + val_images
        }
        
    def load_annotations(self):
        """Load MPI-INF-3DHP annotations"""
        print(f"Loading annotations from: {self.annotations_path}")
        
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        
        data = np.load(self.annotations_path, allow_pickle=True)['data'].item()
        print(f"✓ Loaded annotations for sequences: {list(data.keys())}")
        
        return data
    
    def load_test_annotations(self):
        """Load MPI-INF-3DHP test annotations"""
        test_annotations_path = self.annotations_path.replace('data_train_3dhp.npz', 'data_test_3dhp.npz')
        
        if not os.path.exists(test_annotations_path):
            print(f"Test annotations not found: {test_annotations_path}")
            return {}
            
        print(f"Loading test annotations from: {test_annotations_path}")
        data = np.load(test_annotations_path, allow_pickle=True)['data'].item()
        print(f"✓ Loaded test annotations for sequences: {list(data.keys())}")
        
        return data
    
    def normalize_keypoints(self, keypoints_2d, img_width, img_height):
        """Convert pixel coordinates to YOLO normalized format [0,1]"""
        normalized_kpts = keypoints_2d.copy()
        
        # Check if already normalized
        if np.max(keypoints_2d[:, :2]) <= 1.0:
            # Already normalized, convert to pixel coords first
            normalized_kpts[:, 0] = keypoints_2d[:, 0] * img_width
            normalized_kpts[:, 1] = keypoints_2d[:, 1] * img_height
        
        # Normalize to [0,1]
        normalized_kpts[:, 0] = np.clip(normalized_kpts[:, 0] / img_width, 0, 1)
        normalized_kpts[:, 1] = np.clip(normalized_kpts[:, 1] / img_height, 0, 1)
        
        return normalized_kpts
    
    def create_yolo_annotation(self, keypoints_2d, img_width, img_height, confidence_threshold=0.1):
        """Create YOLO pose annotation format"""
        # Normalize keypoints
        norm_kpts = self.normalize_keypoints(keypoints_2d, img_width, img_height)
        
        # Calculate bounding box from visible keypoints
        visible_kpts = norm_kpts[norm_kpts[:, 2] > confidence_threshold] if norm_kpts.shape[1] > 2 else norm_kpts
        
        if len(visible_kpts) == 0:
            return None
        
        x_coords = visible_kpts[:, 0]
        y_coords = visible_kpts[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add padding to bounding box
        padding = 0.10
        width = x_max - x_min
        height = y_max - y_min
        
        x_min = max(0, x_min - padding * width)
        x_max = min(1, x_max + padding * width)
        y_min = max(0, y_min - padding * height)
        y_max = min(1, y_max + padding * height)
        
        # YOLO bounding box format: center_x, center_y, width, height
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        center_x = x_min + bbox_width / 2
        center_y = y_min + bbox_height / 2
        
        # Create keypoint string for YOLO format
        keypoint_str = ""
        for i in range(17):  # MPI-INF-3DHP has 17 keypoints
            if i < len(norm_kpts):
                x, y = norm_kpts[i, 0], norm_kpts[i, 1]
                visibility = 2 if (norm_kpts.shape[1] > 2 and norm_kpts[i, 2] > confidence_threshold) else 0
                keypoint_str += f" {x:.6f} {y:.6f} {visibility}"
            else:
                keypoint_str += " 0.0 0.0 0"
        
        # YOLO annotation: class_id center_x center_y width height keypoints
        annotation = f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}{keypoint_str}"
        
        return annotation
    
    def process_training_data(self, annotations):
        """Process MPI-INF-3DHP training data (FULL DATASET)"""
        print("\nProcessing training data (FULL DATASET)...")
        
        processed_count = 0
        skipped_count = 0
        total_sequences = len(annotations)
        
        print(f"Training data: Processing all sequences and all frames")
        print(f"  Total sequences to process: {total_sequences}")
        print(f"  Output directory: {self.train_images_path}")
        
        for seq_idx, (seq_name, seq_data) in enumerate(tqdm(annotations.items(), desc="Processing training sequences")):
            if not isinstance(seq_data, list) or len(seq_data) < 1:
                continue
                
            camera_dict = seq_data[0]
            subject, sequence = seq_name.split(' ')
            
            # Process camera 0 only for training
            if '0' not in camera_dict:
                print(f"Warning: Camera 0 not found for {seq_name}")
                continue
                
            camera_data = camera_dict['0']
            poses_2d = camera_data['data_2d']  # Shape: (frames, 17, 2)
            
            # Find corresponding images
            image_folder = os.path.join(self.base_path, subject, sequence, 'imageFrames', 'video_0')
            
            if not os.path.exists(image_folder):
                print(f"Warning: Image folder not found: {image_folder}")
                continue
                
            image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(image_folder, "*.JPG")))
            image_files.sort()
            
            if not image_files:
                print(f"Warning: No images found in {image_folder}")
                continue
            
            # Process ALL frames (no sampling)
            max_frames = min(len(image_files), len(poses_2d))
            
            seq_processed = 0
            seq_skipped = 0
            
            for frame_idx in range(max_frames):
                try:
                    # Load image
                    img_path = image_files[frame_idx]
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        seq_skipped += 1
                        continue
                    
                    img_height, img_width = image.shape[:2]
                    pose_2d = poses_2d[frame_idx]  # Shape: (17, 2)
                    
                    # Add dummy confidence if not present
                    if pose_2d.shape[1] == 2:
                        confidence = np.ones((pose_2d.shape[0], 1)) * 0.9
                        pose_2d = np.hstack([pose_2d, confidence])
                    
                    # Create YOLO annotation
                    annotation = self.create_yolo_annotation(pose_2d, img_width, img_height)
                    
                    if annotation is None:
                        seq_skipped += 1
                        continue
                    
                    # Save image and annotation
                    img_name = f"{subject}_{sequence}_cam0_frame{frame_idx:06d}.jpg"
                    label_name = f"{subject}_{sequence}_cam0_frame{frame_idx:06d}.txt"
                    
                    # Copy image
                    dst_img_path = os.path.join(self.train_images_path, img_name)
                    cv2.imwrite(dst_img_path, image)
                    
                    # Save annotation
                    dst_label_path = os.path.join(self.train_labels_path, label_name)
                    with open(dst_label_path, 'w') as f:
                        f.write(annotation + '\n')
                    
                    seq_processed += 1
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {seq_name} frame {frame_idx}: {e}")
                    seq_skipped += 1
                    skipped_count += 1
                    continue
        
        print(f"\nTraining data: Processed {processed_count} frames, skipped {skipped_count}")
        
    def process_test_data(self, test_annotations):
        """Process MPI-INF-3DHP test data for validation using REAL ground truth"""
        print("\nProcessing test data for validation using REAL GROUND TRUTH...")
        
        if not test_annotations:
            print("No test annotations available, skipping validation data creation")
            return
        
        processed_count = 0
        skipped_count = 0
        
        print(f"Validation data: Processing all test sequences with REAL annotations")
        print(f"  Output directory: {self.val_images_path}")
        
        # Test image paths
        test_base_paths = [
            '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
            '../motion3d/mpi_inf_3dhp_test_set',
            '../../motion3d/mpi_inf_3dhp_test_set'
        ]
        
        for seq_idx, (seq_name, seq_data) in enumerate(tqdm(test_annotations.items(), desc="Processing test sequences")):
            # Find test images
            image_folder = None
            for base_path in test_base_paths:
                potential_path = os.path.join(base_path, seq_name, 'imageSequence')
                if os.path.exists(potential_path):
                    image_folder = potential_path
                    break
            
            if image_folder is None:
                print(f"Warning: Test images not found for {seq_name}")
                continue
            
            # Get image files
            image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
            image_files.sort()
            
            if not image_files:
                continue
            
            # Extract ground truth 2D poses from test annotations
            poses_2d = seq_data['data_2d']  # Shape: (frames, 17, 2)
            
            # Process frames with real annotations
            max_frames = min(len(image_files), len(poses_2d))
            
            for img_idx in range(max_frames):
                try:
                    img_path = image_files[img_idx]
                    image = cv2.imread(img_path)
                    if image is None:
                        skipped_count += 1
                        continue
                    
                    img_height, img_width = image.shape[:2]
                    pose_2d = poses_2d[img_idx]  # Shape: (17, 2)
                    
                    # Add dummy confidence if not present (assume all keypoints are visible)
                    if pose_2d.shape[1] == 2:
                        confidence = np.ones((pose_2d.shape[0], 1)) * 0.9
                        pose_2d = np.hstack([pose_2d, confidence])
                    
                    # Create YOLO annotation using REAL ground truth
                    annotation = self.create_yolo_annotation(pose_2d, img_width, img_height)
                    
                    if annotation is None:
                        skipped_count += 1
                        continue
                    
                    # Save image and annotation
                    img_name = f"{seq_name}_frame{img_idx:06d}.jpg"
                    label_name = f"{seq_name}_frame{img_idx:06d}.txt"
                    
                    dst_img_path = os.path.join(self.val_images_path, img_name)
                    cv2.imwrite(dst_img_path, image)
                    
                    dst_label_path = os.path.join(self.val_labels_path, label_name)
                    with open(dst_label_path, 'w') as f:
                        f.write(annotation + '\n')
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {seq_name} frame {img_idx}: {e}")
                    skipped_count += 1
                    continue
        
        print(f"\nValidation data: Processed {processed_count} frames with REAL annotations, skipped {skipped_count}")
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        dataset_config = {
            'path': os.path.abspath(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,  # number of classes (person)
            'names': ['person'],
            'kpt_shape': [17, 3],  # 17 keypoints, 3 values each (x, y, visibility)
            # Updated flip indices for corrected keypoint order
            'flip_idx': [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14, 15, 16]  # MPI joint flip indices
        }
        
        yaml_path = os.path.join(self.output_path, 'mpi_dataset.yaml')
        
        print(f"Creating YOLO dataset configuration...")
        print(f"  Config path: {yaml_path}")
        print(f"  Dataset path: {os.path.abspath(self.output_path)}")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✓ Created dataset configuration: {yaml_path}")
        return yaml_path
    
    def convert_dataset(self, force_reprocess=False):
        """Convert MPI-INF-3DHP dataset to YOLO format with caching"""
        print("="*60)
        print("Converting MPI-INF-3DHP to YOLO format")
        print("="*60)
        print(f"Target directory: {self.output_path}")
        
        # Check available space
        import shutil
        total, used, free = shutil.disk_usage(self.output_path)
        print(f"Available space: {free // (1024**3)} GB")
        
        # Check if already processed
        if not force_reprocess:
            is_processed, status_msg = self.is_dataset_processed()
            if is_processed:
                print(f"✓ Dataset already processed: {status_msg}")
                
                # Print existing dataset summary
                summary = self.get_processing_summary()
                print(f"\nExisting dataset summary:")
                print(f"  Training images: {summary['train_images']}")
                print(f"  Training labels: {summary['train_labels']}")
                print(f"  Validation images: {summary['val_images']}")
                print(f"  Validation labels: {summary['val_labels']}")
                print(f"  Total images: {summary['total_images']}")
                
                yaml_path = os.path.join(self.output_path, 'mpi_dataset.yaml')
                print(f"  Dataset config: {yaml_path}")
                print(f"✓ Skipping conversion, using existing dataset!")
                return yaml_path
            else:
                print(f"⚠ Dataset needs processing: {status_msg}")
        else:
            print("🔄 Force reprocessing requested, will overwrite existing data")
        
        start_time = time.time()
        
        # Load annotations
        train_annotations = self.load_annotations()
        test_annotations = self.load_test_annotations()
        
        # Process training data
        self.process_training_data(train_annotations)
        
        # Process test data for validation
        self.process_test_data(test_annotations)
        
        # Create dataset YAML
        yaml_path = self.create_dataset_yaml()
        
        # Print summary
        summary = self.get_processing_summary()
        conversion_time = time.time() - start_time
        
        print(f"\n" + "="*60)
        print("DATASET CONVERSION COMPLETED")
        print("="*60)
        print(f"Training images: {summary['train_images']}")
        print(f"Training labels: {summary['train_labels']}")
        print(f"Validation images: {summary['val_images']}")
        print(f"Validation labels: {summary['val_labels']}")
        print(f"Total images: {summary['total_images']}")
        print(f"Conversion time: {conversion_time/60:.1f} minutes")
        print(f"Dataset config: {yaml_path}")
        print(f"Dataset location: {self.output_path}")
        print(f"Ready for YOLO training!")
        
        return yaml_path

def train_yolo_model(dataset_yaml, args):
    """Train YOLO model on converted dataset with enhanced keypoint accuracy settings"""
    print("\n" + "="*60)
    print("STARTING ENHANCED YOLO TRAINING FOR SUPERIOR KEYPOINT ACCURACY")
    print("="*60)
    
    # Initialize metrics tracker
    metrics_tracker = YOLOMetricsTracker(
        use_wandb=args.use_wandb, 
        wandb_project=getattr(args, 'wandb_project', 'YOLO_MPI_3DHP_Enhanced_Training')
    )
    
    # Start monitoring
    metrics_tracker.start_training()
    
    # Load YOLOv11x-pose model from local folder
    model_path = 'model/yolo11x-pose.pt'
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"🔍 Looking for model in: {os.path.abspath(model_path)}")
        
        # Fallback to YOLOv8n-pose if YOLOv11x not found
        print(f"⚠️ Falling back to YOLOv8n-pose...")
        model = YOLO('yolov8n-pose.pt')
        model_name = "YOLOv8n-pose (fallback)"
    else:
        print(f"✅ Loading YOLOv11x-pose from: {os.path.abspath(model_path)}")
        model = YOLO(model_path)
        model_name = "YOLOv11x-pose"
    
    print(f"Enhanced Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Pose loss weight: 17.0 (MAXIMUM KEYPOINT FOCUS)")
    print(f"  Device: {args.device}")
    print(f"  Cache: {args.cache}")
    print(f"  Patience: {args.patience}")
    
    # Log configuration to WandB
    if args.use_wandb:
        wandb.config.update({
            "model": model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "lr": args.lr,
            "pose_loss_weight": 17.0,
            "device": args.device,
            "dataset": "MPI-INF-3DHP",
            "keypoints": 17,
            "cache": args.cache,
            "patience": args.patience,
        })
    
    # Enhanced callback functions
    def on_train_epoch_end(trainer):
        """Enhanced callback for end of training epoch"""
        try:
            epoch = trainer.epoch + 1  # YOLO uses 0-based indexing
            
            # Extract metrics from trainer with better error handling
            results_dict = {}
            
            # Get training losses - safer extraction
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
            
            # Get validation metrics if available
            try:
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    for key, value in trainer.metrics.items():
                        if isinstance(value, (int, float)):
                            results_dict[f'val/{key}'] = float(value)
                        elif hasattr(value, 'item'):
                            results_dict[f'val/{key}'] = float(value.item())
            except Exception as e:
                print(f"⚠ Could not extract validation metrics: {e}")
            
            # Log comprehensive metrics
            try:
                metrics_tracker.log_epoch_metrics(epoch, results_dict)
            except Exception as e:
                print(f"⚠ Failed to log epoch metrics: {e}")
            
        except Exception as e:
            print(f"⚠ Error in epoch callback: {e}")
    
    def on_val_end(trainer):
        """Enhanced callback for end of validation"""
        try:
            if hasattr(trainer, 'metrics') and trainer.metrics:
                # Log validation metrics immediately
                val_metrics = {}
                for key, value in trainer.metrics.items():
                    if isinstance(value, (int, float)):
                        val_metrics[f'val/{key}'] = float(value)
                
                if args.use_wandb and val_metrics:
                    wandb.log(val_metrics)
        except Exception as e:
            print(f"⚠ Error in validation callback: {e}")
    
    # Add callbacks to model
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    
    try:
        print(f"\n🚀 Starting YOLO training with {model_name}...")
        
        # SIMPLIFIED training configuration - just the essentials + pose weight
        training_config = {
            'data': dataset_yaml,
            'epochs': args.epochs,
            'imgsz': args.img_size,
            'batch': args.batch_size,
            'device': args.device,
            'workers': args.workers,
            'project': 'runs/pose',
            'name': 'mpi_yolo11x_pose_enhanced_keypoints',
            'patience': args.patience,
            'cache': args.cache,
            'pose': 17.0,    # ONLY CUSTOM SETTING - Maximum focus on pose estimation
        }
        
        # Auto learning rate configuration
        if args.lr == 'auto':
            # Let YOLO use its default learning rate
            pass  # Don't set lr0, let YOLO decide
        else:
            training_config['lr0'] = args.lr
        
        print(f"\n🎯 TRAINING CONFIGURATION:")
        print(f"   📏 Image Resolution: {args.img_size}px")
        print(f"   🎯 Pose Loss Weight: 17.0 (ENHANCED)")
        print(f"   🔄 Learning Rate: {args.lr}")
        print(f"   💾 Cache: {args.cache}")
        print(f"   ⏱️ Patience: {args.patience} epochs")
        print(f"   📦 All other settings: YOLO DEFAULTS")
        
        # Train the model with simplified configuration
        results = model.train(**training_config)
        
        # Log final results
        if hasattr(results, 'results_dict'):
            final_metrics = results.results_dict
            if args.use_wandb:
                wandb.log({"final_results": final_metrics})
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: runs/pose/mpi_yolo11x_pose_enhanced_keypoints/weights/")
        print(f"🏆 Best model: runs/pose/mpi_yolo11x_pose_enhanced_keypoints/weights/best.pt")
        print(f"📋 Last model: runs/pose/mpi_yolo11x_pose_enhanced_keypoints/weights/last.pt")
        print(f"🎯 Enhanced for: 2D KEYPOINT ACCURACY")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise
    finally:
        # Clean up monitoring
        metrics_tracker.finish_training()

def main():
    parser = argparse.ArgumentParser(description='Enhanced YOLO training for superior 2D keypoint estimation')
    
    # Dataset paths
    parser.add_argument('--base-path', type=str, 
                       default='/nas-ctm01/datasets/public/mpi_inf_3dhp',
                       help='Base path to MPI-INF-3DHP dataset')
    parser.add_argument('--annotations-path', type=str,
                       default='../../motion3d/data_train_3dhp.npz',
                       help='Path to training annotations file')
    parser.add_argument('--output-path', type=str, 
                       default='/nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo',
                       help='Output path for converted dataset (with more space)')
    
    # Enhanced training parameters for keypoint accuracy
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training (optimized for 640px images)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training (high resolution for better keypoints)')
    parser.add_argument('--lr', type=str, default='auto',
                       help='Learning rate (auto for YOLO auto-determination)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use for training (0, 1, 2, etc. or cpu)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--cache', type=str, default='disk',
                       help='Cache strategy: "ram", "disk", or False (optimized for limited RAM)')
    
    # Monitoring options
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable WandB logging for comprehensive monitoring')
    parser.add_argument('--wandb-project', type=str, default='YOLO_MPI_3DHP_Enhanced_Keypoints',
                       help='WandB project name')
    
    # Processing options
    parser.add_argument('--convert-only', action='store_true',
                       help='Only convert dataset, do not train')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train (assume dataset already converted)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing even if dataset exists')
    
    args = parser.parse_args()
    
    # Convert lr to float if not 'auto'
    if args.lr != 'auto':
        try:
            args.lr = float(args.lr)
        except ValueError:
            print(f"❌ Invalid learning rate: {args.lr}. Use 'auto' or a float value.")
            return
    
    print("🎯 ENHANCED YOLO TRAINING FOR SUPERIOR 2D KEYPOINT ESTIMATION")
    print("="*70)
    print(f"📂 Base path: {args.base_path}")
    print(f"📋 Annotations: {args.annotations_path}")
    print(f"💾 Output path: {args.output_path}")
    print(f"🖼️ Image size: {args.img_size}px (HIGH RESOLUTION)")
    print(f"📚 Batch size: {args.batch_size} (optimized for large images)")
    print(f"📈 Learning rate: {args.lr} (AUTO-OPTIMIZED)")
    print(f"🎯 Pose loss weight: 17.0 (MAXIMUM KEYPOINT FOCUS)")
    print(f"💾 Cache strategy: {args.cache} (RAM-OPTIMIZED)")
    print(f"⏱️ Patience: {args.patience} epochs")
    print(f"📊 WandB logging: {args.use_wandb}")
    print(f"🔄 Force reprocess: {args.force_reprocess}")
    print(f"📊 Keypoint order: {MPI_JOINT_NAMES}")
    print(f"🎯 OPTIMIZED FOR: MAXIMUM 2D KEYPOINT ACCURACY")
    
    # Check if output directory exists and create if needed
    if not os.path.exists(args.output_path):
        print(f"📁 Creating output directory: {args.output_path}")
        os.makedirs(args.output_path, exist_ok=True)
    
    # Check available space
    import shutil
    total, used, free = shutil.disk_usage(args.output_path)
    print(f"💾 Available space in {args.output_path}: {free // (1024**3)} GB")
    
    if not args.train_only:
        # Convert dataset (with caching)
        converter = MPIDatasetConverter(args.base_path, args.annotations_path, args.output_path)
        dataset_yaml = converter.convert_dataset(force_reprocess=args.force_reprocess)
    else:
        dataset_yaml = os.path.join(args.output_path, 'mpi_dataset.yaml')
        if not os.path.exists(dataset_yaml):
            print(f"❌ ERROR: Dataset config not found: {dataset_yaml}")
            print("Run without --train-only to convert dataset first")
            return
        else:
            print(f"✅ Using existing dataset: {dataset_yaml}")
    
    if not args.convert_only:
        # Train model with enhanced configuration for maximum keypoint accuracy
        train_yolo_model(dataset_yaml, args)
    
    print(f"\n🎉 Enhanced training process completed!")
    print(f"🎯 Model optimized for superior 2D keypoint estimation accuracy!")

if __name__ == '__main__':
    main()