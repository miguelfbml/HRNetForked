"""
Compare Ground Truth and YOLO 2D poses with comprehensive metrics (MPJPE, PCK, AUC)
Creates side-by-side visualization showing GT vs YOLO predictions with frame-by-frame MPJPE

Usage:
python compare_gt_yolo_2d.py --sequence TS1 --model-path runs/pose/mpi_yolo_pose_full/weights/best.pt --num-frames 50 --save-video

python compare_gt_yolo_2d.py --sequence TS1 --model-path runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt --num-frames 50 --save-video

# Run on all sequences with all frames
python compare_gt_yolo_2d.py --all --model-path runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt

# Run on all sequences with limited frames
python compare_gt_yolo_2d.py --all --model-path runs/pose/mpi_yolo11x_pose_corrected/weights/best.pt --num-frames 50
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from ultralytics import YOLO
from tqdm import tqdm
import glob
import gc
import time

# GPU detection and setup
def check_gpu_availability():
    """Check GPU availability and print system information"""
    gpu_available = torch.cuda.is_available()
    
    print(f"\n{'='*60}")
    print(f"SYSTEM INFORMATION")
    print(f"{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {gpu_available}")
    
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"GPU count: {gpu_count}")
        print(f"Current GPU: {current_device}")
        print(f"GPU name: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f} GB")
        print(f"CUDA compute capability: {torch.cuda.get_device_capability(current_device)}")
        
        # Set memory fraction to avoid OOM
        #torch.cuda.set_per_process_memory_fraction(0.8)
        #print(f"GPU memory fraction set to: 80%")
        
        device = f'cuda:{current_device}'
    else:
        print(f"No GPU available, using CPU")
        device = 'cpu'
    
    print(f"Selected device: {device}")
    print(f"{'='*60}")
    
    return device, gpu_available

# Metric calculation functions
def calculate_torso_diameter_2d(poses_2d):
    """
    Calculate torso diameter from 2D poses using shoulder width and torso height
    
    Args:
        poses_2d: numpy array of shape (frames, 17, 2) containing 2D pose data
        
    Returns:
        numpy array of shape (frames,) containing torso diameter for each frame
    """
    if len(poses_2d.shape) != 3 or poses_2d.shape[1] != 17:
        print(f"Warning: Unexpected pose shape {poses_2d.shape}, using default torso diameter")
        return np.ones(poses_2d.shape[0]) * 100.0
    
    # Joint indices according to MPI-INF-3DHP format
    # 2: LShoulder, 5: RShoulder, 1: SpineShoulder, 14: Sacrum, 8: LHip, 11: RHip
    left_shoulder = poses_2d[:, 2, :]    # LShoulder
    right_shoulder = poses_2d[:, 5, :]   # RShoulder  
    spine_shoulder = poses_2d[:, 1, :]   # SpineShoulder
    sacrum = poses_2d[:, 14, :]          # Sacrum
    left_hip = poses_2d[:, 8, :]         # LHip
    right_hip = poses_2d[:, 11, :]       # RHip
    
    # Calculate shoulder width
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    
    # Calculate hip width  
    hip_width = np.linalg.norm(left_hip - right_hip, axis=1)
    
    # Calculate torso height (spine shoulder to sacrum)
    torso_height = np.linalg.norm(spine_shoulder - sacrum, axis=1)
    
    # Use maximum of shoulder width, hip width, and 50% of torso height as torso diameter
    torso_diameter = np.maximum.reduce([
        shoulder_width,
        hip_width, 
        torso_height * 0.5
    ])
    
    # Ensure minimum diameter of 50 pixels and maximum of 300 pixels for reasonable values
    torso_diameter = np.clip(torso_diameter, 50.0, 300.0)
    
    # Handle invalid cases (all zeros)
    invalid_mask = torso_diameter < 1e-6
    torso_diameter[invalid_mask] = 100.0  # Default value for invalid frames
    
    return torso_diameter

def compute_pck_2d(pred_poses, gt_poses, torso_diameters, fixed_threshold=150.0):
    """
    Compute PCK (Percentage of Correct Keypoints) for 2D poses
    
    Args:
        pred_poses: numpy array of shape (frames, 17, 2) - predicted poses
        gt_poses: numpy array of shape (frames, 17, 2) - ground truth poses
        torso_diameters: numpy array of shape (frames,) - torso diameter for each frame
        fixed_threshold: float - fixed threshold in pixels for PCK calculation
        
    Returns:
        dict containing PCK values for different thresholds
    """
    if pred_poses.shape != gt_poses.shape:
        print(f"Warning: Shape mismatch between predictions {pred_poses.shape} and GT {gt_poses.shape}")
        return {
            'PCK@20%_torso': 0.0,
            'PCK@50%_torso': 0.0, 
            'PCK@80%_torso': 0.0,
            'PCK@100%_torso': 0.0,
            f'PCK@100%_{int(fixed_threshold)}px': 0.0
        }
    
    n_frames, n_joints, _ = pred_poses.shape
    
    # Calculate Euclidean distance between predicted and ground truth keypoints
    joint_distances = np.linalg.norm(pred_poses - gt_poses, axis=2)  # Shape: (frames, joints)
    
    results = {}
    
    # PCK with different torso diameter percentages
    for pct in [20, 50, 80, 100]:
        # Threshold for each frame based on torso diameter percentage
        threshold = torso_diameters[:, np.newaxis] * (pct / 100.0)  # Shape: (frames, 1)
        
        # Check which joints are within threshold
        correct = joint_distances < threshold  # Shape: (frames, joints)
        
        # Calculate PCK as percentage of correct predictions
        pck = np.mean(correct)
        results[f'PCK@{pct}%_torso'] = pck
    
    # PCK with fixed pixel threshold
    correct_fixed = joint_distances < fixed_threshold
    pck_fixed = np.mean(correct_fixed)
    results[f'PCK@100%_{int(fixed_threshold)}px'] = pck_fixed
    
    return results

def compute_auc_2d(pred_poses, gt_poses, max_threshold=150.0, num_thresholds=50):
    """
    Compute AUC (Area Under Curve) for 2D poses using PCK curve
    
    Args:
        pred_poses: numpy array of shape (frames, 17, 2) - predicted poses
        gt_poses: numpy array of shape (frames, 17, 2) - ground truth poses  
        max_threshold: float - maximum threshold for AUC calculation
        num_thresholds: int - number of threshold points to evaluate
        
    Returns:
        float - AUC value normalized by max_threshold
    """
    if pred_poses.shape != gt_poses.shape:
        print(f"Warning: Shape mismatch for AUC calculation")
        return 0.0
    
    # Calculate joint distances
    joint_distances = np.linalg.norm(pred_poses - gt_poses, axis=2)  # Shape: (frames, joints)
    
    # Create threshold range from 0 to max_threshold
    thresholds = np.linspace(0, max_threshold, num_thresholds)
    pck_values = []
    
    # Calculate PCK for each threshold
    for threshold in thresholds:
        correct = joint_distances < threshold
        pck = np.mean(correct)
        pck_values.append(pck)
    
    # Calculate AUC using trapezoidal rule and normalize by max_threshold
    auc = np.trapz(pck_values, thresholds) / max_threshold
    
    return auc

# MPI-INF-3DHP joint names and connections
JOINT_NAMES = [
'Head',
'SpineShoulder', 
'LShoulder',
'LElbow',
'LHand',
'RShoulder',
'RElbow',
'RHand',
'LHip',
'LKnee',
'LAnkle',
'RHip',
'RKnee',
'RAnkle',
'Sacrum',
'Spine',
'Neck'
]

CONNECTIONS_2D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

# Available test sequences
TEST_SEQUENCES = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

def get_available_sequences():
    """Get list of available test sequences from data file"""
    test_data_paths = [
        '../../motion3d/data_test_3dhp.npz',
    ]
    
    for data_path in test_data_paths:
        if os.path.exists(data_path):
            print(f"✓ Found ground truth data at: {os.path.abspath(data_path)}")
            print(f"Data Path: {data_path}")
            data = np.load(data_path, allow_pickle=True)['data'].item()
            return list(data.keys())
    
    # Fallback to default sequences
    return TEST_SEQUENCES

def load_test_3d_data_from_dataset(sequence_name):
    """Load test data for a specific sequence"""
    test_data_paths = [
        '../../motion3d/data_test_3dhp.npz',
    ]
    
    for data_path in test_data_paths:
        if os.path.exists(data_path):
            print(f"✓ Loading ground truth from: {os.path.abspath(data_path)}")
            data = np.load(data_path, allow_pickle=True)['data'].item()
            
            if sequence_name in data:
                seq_data = data[sequence_name]
                poses_2d = seq_data['data_2d']  # Shape: (frames, 17, 2)
                poses_3d = seq_data['data_3d']  # Shape: (frames, 17, 3)
                
                return poses_2d, poses_3d, sequence_name
    
    return None, None, None

def load_test_frames_batch(sequence_name, start_frame=0, num_frames=None):
    """Load test frames for the sequence in batches"""
    test_image_paths = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
    ]
    
    for base_path in test_image_paths:
        image_folder = os.path.join(base_path, sequence_name, 'imageSequence')
        if os.path.exists(image_folder):
            print(f"✓ Found images at: {os.path.abspath(image_folder)}")
            print(f"test image path: {base_path}")
            image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))
            image_files.sort()
            
            if image_files:
                # Get the requested batch of frames
                end_frame = start_frame + num_frames if num_frames is not None else len(image_files)
                batch_files = image_files[start_frame:end_frame]
                
                frames = []
                for img_path in batch_files:
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        frames.append(frame)
                
                return frames, len(image_files)  # Return frames and total count
    
    return None, 0

def load_test_frames(sequence_name, num_frames=None):
    """Load test frames for the sequence (legacy function for compatibility)"""
    frames, total_frames = load_test_frames_batch(sequence_name, 0, num_frames)
    return frames

def estimate_yolo_poses_batch(model, frames, img_size=640, device='cpu'):
    """Estimate poses using YOLO model with memory management and GPU support, tracking execution time"""
    yolo_poses = []
    confidences = []
    inference_times = []
    
    # Process frames in smaller batches to avoid memory issues
    batch_size = 32 if device.startswith('cuda') else 16  # Larger batches for GPU
    
    print(f"  Processing {len(frames)} frames in batches of {batch_size} on {device}")
    
    for i in tqdm(range(0, len(frames), batch_size), desc="YOLO Inference"):
        batch_frames = frames[i:i+batch_size]
        
        for frame in batch_frames:
            try:
                # Track inference time
                start_time = time.time()
                
                # Run YOLO inference with device specification
                results = model.predict(frame, verbose=False, imgsz=img_size, conf=0.65, device=device)
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                if (results and len(results) > 0 and 
                    hasattr(results[0], 'keypoints') and 
                    results[0].keypoints is not None and 
                    len(results[0].keypoints.xy) > 0):
                    
                    # Get first detection's keypoints
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
                    conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
                    
                    # Ensure we have 17 keypoints
                    if keypoints.shape[0] == 17:
                        yolo_poses.append(keypoints)
                        confidences.append(conf)
                    else:
                        # Pad or truncate to 17 keypoints
                        padded_kpts = np.zeros((17, 2))
                        padded_conf = np.zeros(17)
                        
                        n_kpts = min(17, keypoints.shape[0])
                        padded_kpts[:n_kpts] = keypoints[:n_kpts]
                        padded_conf[:n_kpts] = conf[:n_kpts] if len(conf) > 0 else 0.5
                        
                        yolo_poses.append(padded_kpts)
                        confidences.append(padded_conf)
                else:
                    # No detection, create zero pose
                    yolo_poses.append(np.zeros((17, 2)))
                    confidences.append(np.zeros(17))
                    
            except Exception as e:
                print(f"Error in YOLO inference: {e}")
                yolo_poses.append(np.zeros((17, 2)))
                confidences.append(np.zeros(17))
                inference_times.append(0.0)  # Add 0 for failed inference
        
        # Clear memory after each batch
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()
    
    yolo_poses = np.array(yolo_poses)  # Shape: (frames, 17, 2)
    confidences = np.array(confidences)  # Shape: (frames, 17)
    
    # Calculate performance metrics
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if inference_times else 0.0
    fps = 1.0 / mean_inference_time if mean_inference_time > 0 else 0.0
    
    performance_metrics = {
        'total_inference_time': total_inference_time,
        'mean_inference_time': mean_inference_time,
        'fps': fps,
        'processed_frames': len(frames)
    }
    
    return yolo_poses, confidences, performance_metrics

def estimate_yolo_poses(model, frames, img_size=640, device='cpu'):
    """Legacy function for compatibility"""
    yolo_poses, confidences, performance_metrics = estimate_yolo_poses_batch(model, frames, img_size, device)
    return yolo_poses, confidences, performance_metrics

def convert_coordinates_to_pixels(poses_2d, frames):
    """Convert normalized coordinates to pixel coordinates"""
    if len(frames) == 0:
        return poses_2d
    
    # Get image dimensions from first frame
    img_height, img_width = frames[0].shape[:2]
    
    # Check if coordinates are already in pixel format
    if np.max(poses_2d[:, :, :2]) > 1.0:
        return poses_2d  # Already in pixel coordinates
    
    # Convert from normalized [0,1] to pixel coordinates
    poses_pixel = poses_2d.copy()
    poses_pixel[:, :, 0] *= img_width
    poses_pixel[:, :, 1] *= img_height
    
    return poses_pixel

def make_root_relative_2d_pixel(poses_2d_pixel, root_joint_idx=14):
    """Make poses root-relative in pixel domain"""
    root_relative_poses = poses_2d_pixel.copy()
    
    for frame_idx in range(poses_2d_pixel.shape[0]):
        root_pos = poses_2d_pixel[frame_idx, root_joint_idx, :2]
        root_relative_poses[frame_idx, :, :2] -= root_pos
    
    return root_relative_poses

def compute_mpjpe_2d(gt_poses_2d, yolo_poses_2d):
    """Compute comprehensive metrics including MPJPE, PCK, and AUC for 2D poses"""
    min_frames = min(len(gt_poses_2d), len(yolo_poses_2d))
    gt_poses = gt_poses_2d[:min_frames]
    yolo_poses = yolo_poses_2d[:min_frames]
    
    # Find valid frames (non-zero poses)
    valid_frames = []
    valid_gt_list = []
    valid_yolo_list = []
    frame_mpjpe = []
    
    for frame_idx in range(min_frames):
        gt_frame = gt_poses[frame_idx]
        yolo_frame = yolo_poses[frame_idx]
        
        gt_valid = not np.all(gt_frame == 0)
        yolo_valid = not np.all(yolo_frame == 0)
        
        if gt_valid and yolo_valid:
            valid_frames.append(frame_idx)
            valid_gt_list.append(gt_frame)
            valid_yolo_list.append(yolo_frame)
            
            # Calculate frame MPJPE
            joint_diffs = np.linalg.norm(gt_frame - yolo_frame, axis=1)
            frame_error = np.mean(joint_diffs)
            frame_mpjpe.append(frame_error)
        else:
            frame_mpjpe.append(np.nan)
    
    if len(valid_gt_list) == 0:
        return None
    
    # Convert to numpy arrays
    valid_gt = np.array(valid_gt_list)  # (V, 17, 2)
    valid_yolo = np.array(valid_yolo_list)  # (V, 17, 2)
    
    # Calculate MPJPE
    avg_mpjpe = np.mean([e for e in frame_mpjpe if not np.isnan(e)])
    
    # Calculate joint-wise errors
    joint_errors = np.mean(np.linalg.norm(valid_gt - valid_yolo, axis=2), axis=0)  # (17,)
    
    # Calculate torso diameters for PCK
    torso_diameters = calculate_torso_diameter_2d(valid_gt)
    
    # Compute PCK metrics
    pck_results = compute_pck_2d(valid_yolo, valid_gt, torso_diameters, fixed_threshold=150.0)
    
    # Compute AUC
    auc = compute_auc_2d(valid_yolo, valid_gt, max_threshold=150.0)
    
    return {
        'avg_mpjpe': float(avg_mpjpe),
        'frame_mpjpe': frame_mpjpe,
        'joint_errors': [float(x) for x in joint_errors],
        'pck_results': pck_results,
        'auc': float(auc),
        'valid_frames': len(valid_frames),
        'total_frames': min_frames,
        'torso_diameters': torso_diameters
    }

def process_single_sequence_batched(model, sequence_name, args, device='cpu'):
    """Process a single sequence in batches to avoid memory issues"""
    print(f"\n{'='*60}")
    print(f"Processing sequence: {sequence_name}")
    print(f"{'='*60}")
    
    # Load ground truth data
    gt_poses_2d, gt_poses_3d, seq_name = load_test_3d_data_from_dataset(sequence_name)
    
    if gt_poses_2d is None:
        print(f"❌ Failed to load ground truth data for {sequence_name}")
        return None
    
    total_gt_frames = len(gt_poses_2d)
    
    # Initialize performance tracking
    total_inference_time = 0.0
    total_processed_frames = 0
    all_inference_times = []
    
    # Determine processing strategy
    if args.all and args.num_frames is None:
        # Process all frames in batches for memory efficiency
        print(f"✓ Processing ALL {total_gt_frames} frames for sequence {sequence_name} (in batches)")
        batch_size = 300 if device.startswith('cuda') else 200  # Adjust batch size based on device
        
        all_frame_mpjpe = []
        all_valid_gt = []
        all_valid_yolo = []
        total_valid_frames = 0
        
        for start_idx in range(0, total_gt_frames, batch_size):
            end_idx = min(start_idx + batch_size, total_gt_frames)
            batch_frames_count = end_idx - start_idx
            
            print(f"  Processing batch {start_idx//batch_size + 1}/{(total_gt_frames + batch_size - 1)//batch_size}: frames {start_idx+1}-{end_idx}")
            
            # Load batch of ground truth and image frames
            gt_batch = gt_poses_2d[start_idx:end_idx]
            frames_batch, _ = load_test_frames_batch(sequence_name, start_idx, batch_frames_count)
            
            if frames_batch is None:
                print(f"❌ Failed to load frames for batch {start_idx}-{end_idx}")
                continue
            
            # Ensure matching frames
            min_frames = min(len(frames_batch), len(gt_batch))
            frames_batch = frames_batch[:min_frames]
            gt_batch = gt_batch[:min_frames]
            
            # Run YOLO inference on batch
            yolo_batch, _, batch_performance = estimate_yolo_poses_batch(model, frames_batch, args.img_size, device)
            
            # Accumulate performance metrics
            total_inference_time += batch_performance['total_inference_time']
            total_processed_frames += batch_performance['processed_frames']
            all_inference_times.extend([batch_performance['mean_inference_time']] * batch_performance['processed_frames'])
            
            # Convert coordinates
            gt_batch_pixel = convert_coordinates_to_pixels(gt_batch, frames_batch)
            gt_batch_root_rel = make_root_relative_2d_pixel(gt_batch_pixel, root_joint_idx=14)
            yolo_batch_root_rel = make_root_relative_2d_pixel(yolo_batch, root_joint_idx=14)
            
            # Process batch metrics
            for frame_idx in range(min_frames):
                gt_frame = gt_batch_root_rel[frame_idx]
                yolo_frame = yolo_batch_root_rel[frame_idx]
                
                gt_valid = not np.all(gt_frame == 0)
                yolo_valid = not np.all(yolo_frame == 0)
                
                if gt_valid and yolo_valid:
                    all_valid_gt.append(gt_frame)
                    all_valid_yolo.append(yolo_frame)
                    total_valid_frames += 1
                    
                    # Calculate frame MPJPE
                    joint_diffs = np.linalg.norm(gt_frame - yolo_frame, axis=1)
                    frame_error = np.mean(joint_diffs)
                    all_frame_mpjpe.append(frame_error)
                else:
                    all_frame_mpjpe.append(np.nan)
            
            # Clear memory
            del frames_batch, gt_batch, yolo_batch, gt_batch_pixel, gt_batch_root_rel, yolo_batch_root_rel
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            gc.collect()
        
        if len(all_valid_gt) == 0:
            print(f"❌ No valid frames found for {sequence_name}")
            return None
        
        # Calculate final metrics
        valid_gt = np.array(all_valid_gt)
        valid_yolo = np.array(all_valid_yolo)
        
        avg_mpjpe = np.mean([e for e in all_frame_mpjpe if not np.isnan(e)])
        joint_errors = np.mean(np.linalg.norm(valid_gt - valid_yolo, axis=2), axis=0)
        
        torso_diameters = calculate_torso_diameter_2d(valid_gt)
        pck_results = compute_pck_2d(valid_yolo, valid_gt, torso_diameters, fixed_threshold=150.0)
        auc = compute_auc_2d(valid_yolo, valid_gt, max_threshold=150.0)
        
        # Calculate overall performance metrics
        mean_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
        fps = 1.0 / mean_inference_time if mean_inference_time > 0 else 0.0
        
        metrics = {
            'avg_mpjpe': float(avg_mpjpe),
            'frame_mpjpe': all_frame_mpjpe,
            'joint_errors': [float(x) for x in joint_errors],
            'pck_results': pck_results,
            'auc': float(auc),
            'valid_frames': total_valid_frames,
            'total_frames': total_processed_frames,
            'torso_diameters': torso_diameters,
            'sequence': sequence_name,
            'performance': {
                'total_inference_time': total_inference_time,
                'mean_inference_time': mean_inference_time,
                'fps': fps,
                'processed_frames': total_processed_frames
            }
        }
        
    else:
        # Process limited frames (original method)
        num_frames_to_use = args.num_frames if args.num_frames is not None else 50
        gt_poses_2d_limited = gt_poses_2d[:num_frames_to_use]
        print(f"✓ Using {len(gt_poses_2d_limited)} frames for sequence {sequence_name}")
        
        # Load test frames
        frames = load_test_frames(sequence_name, num_frames_to_use)
        
        if frames is None:
            print(f"❌ Failed to load test frames for {sequence_name}")
            return None
        
        # Ensure matching number of frames
        min_frames = min(len(frames), len(gt_poses_2d_limited))
        frames = frames[:min_frames]
        gt_poses_2d_final = gt_poses_2d_limited[:min_frames]
        
        print(f"✓ Processing {min_frames} frames for comparison")
        
        # Run YOLO pose estimation
        print(f"🔍 Running YOLO pose estimation...")
        yolo_poses_2d, yolo_confidences, performance_metrics = estimate_yolo_poses(model, frames, args.img_size, device)
        
        # Convert coordinates to pixels
        gt_poses_2d_pixel = convert_coordinates_to_pixels(gt_poses_2d_final, frames)
        
        # Make both datasets root-relative
        gt_poses_2d_root_rel = make_root_relative_2d_pixel(gt_poses_2d_pixel, root_joint_idx=14)
        yolo_poses_2d_root_rel = make_root_relative_2d_pixel(yolo_poses_2d, root_joint_idx=14)
        
        # Compute comprehensive metrics
        print(f"📊 Computing comprehensive metrics...")
        metrics = compute_mpjpe_2d(gt_poses_2d_root_rel, yolo_poses_2d_root_rel)
        
        if metrics:
            metrics['sequence'] = sequence_name
            metrics['performance'] = performance_metrics
    
    if metrics:
        print(f"✓ Metrics computed for {sequence_name}")
        print(f"  MPJPE: {metrics['avg_mpjpe']:.2f} pixels")
        print(f"  Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
        print(f"  FPS: {metrics['performance']['fps']:.2f}")
        print(f"  Mean inference time: {metrics['performance']['mean_inference_time']*1000:.2f} ms")
    else:
        print(f"❌ Failed to compute metrics for {sequence_name}")
    
    return metrics

def process_single_sequence(model, sequence_name, args, device='cpu'):
    """Wrapper function to choose between batched and regular processing"""
    if args.all and args.num_frames is None:
        return process_single_sequence_batched(model, sequence_name, args, device)
    else:
        # Use original processing for limited frames
        return process_single_sequence_original(model, sequence_name, args, device)

def process_single_sequence_original(model, sequence_name, args, device='cpu'):
    """Original processing function for limited frames"""
    print(f"\n{'='*60}")
    print(f"Processing sequence: {sequence_name}")
    print(f"{'='*60}")
    
    # Load ground truth data
    gt_poses_2d, gt_poses_3d, seq_name = load_test_3d_data_from_dataset(sequence_name)
    
    if gt_poses_2d is None:
        print(f"❌ Failed to load ground truth data for {sequence_name}")
        return None
    
    # Use specified number of frames or default
    num_frames_to_use = args.num_frames if args.num_frames is not None else 50
    gt_poses_2d_limited = gt_poses_2d[:num_frames_to_use]
    print(f"✓ Using {len(gt_poses_2d_limited)} frames for sequence {sequence_name}")
    
    # Load test frames
    frames = load_test_frames(sequence_name, num_frames_to_use)
    
    if frames is None:
        print(f"❌ Failed to load test frames for {sequence_name}")
        return None
    
    # Ensure matching number of frames
    min_frames = min(len(frames), len(gt_poses_2d_limited))
    frames = frames[:min_frames]
    gt_poses_2d_final = gt_poses_2d_limited[:min_frames]
    
    print(f"✓ Processing {min_frames} frames for comparison")
    
    # Run YOLO pose estimation
    print(f"🔍 Running YOLO pose estimation...")
    yolo_poses_2d, yolo_confidences, performance_metrics = estimate_yolo_poses(model, frames, args.img_size, device)
    
    # Convert coordinates to pixels
    gt_poses_2d_pixel = convert_coordinates_to_pixels(gt_poses_2d_final, frames)
    
    # Make both datasets root-relative
    gt_poses_2d_root_rel = make_root_relative_2d_pixel(gt_poses_2d_pixel, root_joint_idx=14)
    yolo_poses_2d_root_rel = make_root_relative_2d_pixel(yolo_poses_2d, root_joint_idx=14)
    
    # Compute comprehensive metrics
    print(f"📊 Computing comprehensive metrics...")
    metrics = compute_mpjpe_2d(gt_poses_2d_root_rel, yolo_poses_2d_root_rel)
    
    if metrics:
        metrics['sequence'] = sequence_name
        metrics['performance'] = performance_metrics
        print(f"✓ Metrics computed for {sequence_name}")
        print(f"  MPJPE: {metrics['avg_mpjpe']:.2f} pixels")
        print(f"  Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
        print(f"  FPS: {metrics['performance']['fps']:.2f}")
        print(f"  Mean inference time: {metrics['performance']['mean_inference_time']*1000:.2f} ms")
    else:
        print(f"❌ Failed to compute metrics for {sequence_name}")
    
    return metrics

def create_comparison_visualization(gt_poses_2d, yolo_poses_2d, seq_name, metrics, args):
    """Create side-by-side comparison visualization with comprehensive metrics"""
    min_frames = min(len(gt_poses_2d), len(yolo_poses_2d), args.num_frames if args.num_frames else len(gt_poses_2d))
    
    if min_frames == 0:
        print("No frames to visualize!")
        return None, None, 0
    
    gt_poses = gt_poses_2d[:min_frames]
    yolo_poses = yolo_poses_2d[:min_frames]
    frame_mpjpe = metrics['frame_mpjpe'][:min_frames] if metrics else [np.nan] * min_frames
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Enhanced title with all metrics including performance
    if metrics:
        title = (f'2D Pose Comparison: GT vs YOLO - {seq_name} (Root-Relative, Pixels)\n'
                f'Avg MPJPE: {metrics["avg_mpjpe"]:.1f}px | '
                f'AUC: {metrics["auc"]:.4f} | '
                f'PCK@50%_150px: {metrics["pck_results"]["PCK@50%_torso"]*100:.1f}% | '
                f'FPS: {metrics["performance"]["fps"]:.2f}')
    else:
        title = f'2D Pose Comparison: GT vs YOLO - {seq_name} (Root-Relative, Pixels)'
    
    fig.suptitle(title, fontsize=14)
    
    # Calculate visualization bounds
    all_gt_non_root = gt_poses[:, :, :].reshape(-1, 2)
    all_yolo_non_root = yolo_poses[:, :, :].reshape(-1, 2)
    
    valid_gt = all_gt_non_root[~np.all(all_gt_non_root == 0, axis=1)]
    valid_yolo = all_yolo_non_root[~np.all(all_yolo_non_root == 0, axis=1)]
    
    if len(valid_gt) > 0 and len(valid_yolo) > 0:
        all_points = np.vstack([valid_gt, valid_yolo])
        x_range = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
        y_range = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
        
        # Add padding
        x_padding = max((x_range[1] - x_range[0]) * 0.1, 10)
        y_padding = max((y_range[1] - y_range[0]) * 0.1, 10)
        
        x_min, x_max = x_range[0] - x_padding, x_range[1] + x_padding
        y_min, y_max = y_range[0] - y_padding, y_range[1] + y_padding
    else:
        x_min, x_max = -500, 500
        y_min, y_max = -500, 500
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # Set common properties
        for ax in [ax1, ax2]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Plot 1: Ground Truth
        ax1.set_title(f'Ground Truth\nFrame {frame_idx+1}/{min_frames}', fontsize=14)
        gt_frame = gt_poses[frame_idx]
        
        gt_valid = not np.all(gt_frame == 0)
        if gt_valid:
            # Draw skeleton connections
            for connection in CONNECTIONS_2D:
                joint1, joint2 = connection
                if joint1 < len(gt_frame) and joint2 < len(gt_frame):
                    x1, y1 = gt_frame[joint1]
                    x2, y2 = gt_frame[joint2]
                    ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
            
            # Draw joints
            for joint_idx, (x, y) in enumerate(gt_frame):
                if joint_idx != 14:  # Skip root joint
                    ax1.scatter(x, y, c='blue', s=60, alpha=0.9, edgecolors='darkblue', linewidth=1)
                    ax1.text(x+15, y+15, str(joint_idx), fontsize=9, ha='left', va='bottom', 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        else:
            ax1.text((x_min+x_max)/2, (y_min+y_max)/2, 'No GT Data', ha='center', va='center', 
                    fontsize=16, color='red')
        
        # Enhanced frame title with frame-specific MPJPE
        frame_mpjpe_val = frame_mpjpe[frame_idx] if not np.isnan(frame_mpjpe[frame_idx]) else 0
        ax2.set_title(f'YOLO Estimation\nFrame {frame_idx+1}/{min_frames} | '
                     f'Frame MPJPE: {frame_mpjpe_val:.1f}px', fontsize=14)
        yolo_frame = yolo_poses[frame_idx]
        
        yolo_valid = not np.all(yolo_frame == 0)
        if yolo_valid:
            # Draw skeleton connections
            for connection in CONNECTIONS_2D:
                joint1, joint2 = connection
                if joint1 < len(yolo_frame) and joint2 < len(yolo_frame):
                    x1, y1 = yolo_frame[joint1]
                    x2, y2 = yolo_frame[joint2]
                    ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
            
            # Draw joints
            for joint_idx, (x, y) in enumerate(yolo_frame):
                if joint_idx != 14:  # Skip root joint
                    ax2.scatter(x, y, c='red', s=60, alpha=0.9, edgecolors='darkred', linewidth=1)
                    ax2.text(x+15, y+15, str(joint_idx), fontsize=9, ha='left', va='bottom', 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        else:
            ax2.text((x_min+x_max)/2, (y_min+y_max)/2, 'No YOLO Detection', ha='center', va='center', 
                    fontsize=16, color='red')
        
        # Highlight root joint (at origin)
        for ax in [ax1, ax2]:
            ax.scatter(0, 0, c='green', s=120, marker='*', alpha=1.0, 
                      edgecolors='darkgreen', linewidth=2, label='Root (Hip Center)')
        
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        ax2.set_xlabel('X (pixels)', fontsize=12)
        ax2.set_ylabel('Y (pixels)', fontsize=12)
        
        plt.tight_layout()
        
        return [ax1, ax2]
    
    return update, fig, min_frames

def print_sequence_results(metrics):
    """Print results for a single sequence"""
    if not metrics:
        return
    
    print(f"\nResults for {metrics['sequence']}:")
    print(f"  MPJPE: {metrics['avg_mpjpe']:.2f} pixels")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
    print(f"  FPS: {metrics['performance']['fps']:.2f}")
    print(f"  Mean inference time: {metrics['performance']['mean_inference_time']*1000:.2f} ms")
    
    print(f"  PCK metrics:")
    for key, value in metrics['pck_results'].items():
        print(f"    {key}: {value*100:.2f}%")

def print_summary_results(all_metrics, model_name):
    """Print summary results for all sequences with overall MPJPE and performance metrics"""
    if not all_metrics:
        print("No results to summarize.")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SUMMARY - YOLO vs Ground Truth (2D)")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    
    # Calculate overall statistics
    valid_metrics = [m for m in all_metrics if m is not None]
    
    if not valid_metrics:
        print("No valid metrics found.")
        return
    
    # Calculate weighted overall MPJPE (weighted by number of valid frames)
    total_mpjpe_sum = 0
    total_weighted_frames = 0
    
    for m in valid_metrics:
        weight = m['valid_frames']
        total_mpjpe_sum += m['avg_mpjpe'] * weight
        total_weighted_frames += weight
    
    overall_mpjpe = total_mpjpe_sum / total_weighted_frames if total_weighted_frames > 0 else 0
    
    # Calculate simple average MPJPE across sequences
    avg_mpjpe = np.mean([m['avg_mpjpe'] for m in valid_metrics])
    avg_auc = np.mean([m['auc'] for m in valid_metrics])
    total_valid_frames = sum([m['valid_frames'] for m in valid_metrics])
    total_frames = sum([m['total_frames'] for m in valid_metrics])
    
    # Calculate performance metrics
    total_inference_time = sum([m['performance']['total_inference_time'] for m in valid_metrics])
    total_processed_frames = sum([m['performance']['processed_frames'] for m in valid_metrics])
    
    # Calculate weighted average FPS and mean inference time
    weighted_fps_sum = 0
    weighted_inference_time_sum = 0
    total_weight = 0
    
    for m in valid_metrics:
        weight = m['performance']['processed_frames']
        weighted_fps_sum += m['performance']['fps'] * weight
        weighted_inference_time_sum += m['performance']['mean_inference_time'] * weight
        total_weight += weight
    
    overall_fps = weighted_fps_sum / total_weight if total_weight > 0 else 0
    overall_mean_inference_time = weighted_inference_time_sum / total_weight if total_weight > 0 else 0
    
    print(f"\nOVERALL METRICS:")
    print(f"  Sequences processed: {len(valid_metrics)}")
    print(f"  Total valid frames: {total_valid_frames}/{total_frames}")
    print(f"  Overall MPJPE (weighted): {overall_mpjpe:.2f} pixels")
    print(f"  Average MPJPE (per sequence): {avg_mpjpe:.2f} pixels")
    print(f"  Average AUC: {avg_auc:.4f}")
    
    print(f"\nPERFORMACE METRICS:")
    print(f"  Total inference time: {total_inference_time:.2f} seconds")
    print(f"  Total processed frames: {total_processed_frames}")
    print(f"  Overall FPS (weighted): {overall_fps:.2f}")
    print(f"  Overall mean inference time: {overall_mean_inference_time*1000:.2f} ms")
    
    # PCK metrics
    pck_keys = valid_metrics[0]['pck_results'].keys()
    print(f"\nPCK METRICS (averaged across sequences):")
    for key in pck_keys:
        avg_pck = np.mean([m['pck_results'][key] for m in valid_metrics])
        print(f"  {key}: {avg_pck*100:.2f}%")
    
    print(f"\nPER-SEQUENCE BREAKDOWN:")
    print(f"{'Sequence':<10} {'MPJPE':<12} {'AUC':<10} {'FPS':<10} {'Time(ms)':<12} {'Valid/Total':<12}")
    print(f"{'-'*70}")
    
    for metrics in valid_metrics:
        mpjpe = metrics['avg_mpjpe']
        auc = metrics['auc']
        fps = metrics['performance']['fps']
        mean_time = metrics['performance']['mean_inference_time'] * 1000
        valid_frames = metrics['valid_frames']
        total_frames = metrics['total_frames']
        sequence = metrics['sequence']
        
        print(f"{sequence:<10} {mpjpe:<12.2f} {auc:<10.4f} {fps:<10.2f} {mean_time:<12.2f} {valid_frames}/{total_frames:<12}")
    
    print(f"{'='*80}")
    print(f"🎯 FINAL OVERALL MPJPE: {overall_mpjpe:.2f} pixels")
    print(f"⚡ FINAL OVERALL FPS: {overall_fps:.2f}")
    print(f"⏱️  FINAL MEAN INFERENCE TIME: {overall_mean_inference_time*1000:.2f} ms")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Compare Ground Truth and YOLO 2D poses with comprehensive metrics')
    parser.add_argument('--sequence', type=str, default='TS1', 
                       help='Sequence to compare (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--num-frames', type=int, default=None,
                       help='Number of frames to process per sequence (if not specified with --all, uses all frames in batches)')
    parser.add_argument('--all', action='store_true',
                       help='Run evaluation on all available sequences with all frames (unless --num-frames specified)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save comparison as GIF (only for single sequence)')
    parser.add_argument('--output-dir', type=str, default='comparison_output',
                       help='Directory to save outputs')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size for YOLO inference')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    args = parser.parse_args()
    
    print("🎯 Ground Truth vs YOLO 2D Pose Comparison with Comprehensive Metrics")
    print("="*80)
    
    # Check GPU availability and setup device
    if args.device == 'auto':
        device, gpu_available = check_gpu_availability()
    else:
        device = args.device
        gpu_available = device.startswith('cuda') and torch.cuda.is_available()
    
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    
    # Update frame information display
    if args.all and args.num_frames is None:
        print(f"Mode: Process ALL FRAMES from ALL SEQUENCES (in memory-efficient batches)")
    elif args.all and args.num_frames is not None:
        print(f"Mode: Process {args.num_frames} frames from ALL SEQUENCES")
    else:
        frame_count = args.num_frames if args.num_frames is not None else 50
        print(f"Mode: Process {frame_count} frames from sequence {args.sequence}")
    
    print(f"Input size: {args.img_size}")
    print("Metrics: MPJPE, PCK (Percentage of Correct Keypoints), AUC (Area Under Curve), FPS, Inference Time")
    print("Coordinate system: Root-relative poses in pixel domain")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    # Load YOLO model with device specification
    print(f"🤖 Loading YOLO model from {args.model_path}...")
    try:
        model = YOLO(args.model_path)
        
        # Move model to GPU if available
        if gpu_available:
            print(f"📦 Moving model to {device}...")
            # YOLO handles device placement automatically, but we can ensure it
            model.to(device)
            
        print("✓ YOLO model loaded successfully")
        
        # Print model info
        print(f"Model device: {next(model.model.parameters()).device if hasattr(model, 'model') else 'Unknown'}")
        
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    model_name = os.path.basename(args.model_path)
    
    if args.all:
        # Process all sequences
        print(f"\n🔄 Processing all available sequences...")
        available_sequences = get_available_sequences()
        print(f"Available sequences: {available_sequences}")
        
        if args.num_frames is None:
            print("ℹ️  Processing ALL frames from ALL sequences using memory-efficient batching.")
            print("   This will process sequences in batches to avoid memory issues.")
        
        all_metrics = []
        
        for sequence in available_sequences:
            try:
                metrics = process_single_sequence(model, sequence, args, device)
                all_metrics.append(metrics)
                
                if metrics:
                    print_sequence_results(metrics)
                
                # Clear cache between sequences
                if gpu_available:
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error processing sequence {sequence}: {e}")
                all_metrics.append(None)
        
        # Print summary
        print_summary_results(all_metrics, model_name)
        
    else:
        # Process single sequence
        print(f"\n📂 Processing single sequence: {args.sequence}")
        
        # Set default frames for single sequence if not specified
        if args.num_frames is None:
            args.num_frames = 50
            print(f"Using default {args.num_frames} frames for single sequence")
        
        metrics = process_single_sequence(model, args.sequence, args, device)
        
        if not metrics:
            print("❌ Failed to process sequence")
            return
        
        # Print detailed results
        print(f"\n" + "="*60)
        print(f"DETAILED RESULTS FOR {args.sequence}")
        print(f"="*60)
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  FPS: {metrics['performance']['fps']:.2f}")
        print(f"  Mean inference time: {metrics['performance']['mean_inference_time']*1000:.2f} ms")
        print(f"  Total inference time: {metrics['performance']['total_inference_time']:.2f} seconds")
        print(f"  Processed frames: {metrics['performance']['processed_frames']}")
        
        print(f"\nMPJPE (Mean Per Joint Position Error):")
        print(f"  Average MPJPE: {metrics['avg_mpjpe']:.2f} pixels")
        
        print(f"\nPCK (Percentage of Correct Keypoints):")
        for key, value in metrics['pck_results'].items():
            print(f"  {key}: {value*100:.2f}%")
        
        print(f"\nAUC (Area Under Curve):")
        print(f"  AUC: {metrics['auc']:.4f}")
        
        print(f"\nJoint-wise errors (top 5 worst):")
        joint_error_pairs = [(i, metrics['joint_errors'][i], JOINT_NAMES[i]) for i in range(17)]
        joint_error_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (joint_idx, error, name) in enumerate(joint_error_pairs[:5]):
            print(f"  {name} (joint {joint_idx}): {error:.2f} pixels")
        
        print(f"="*60)
        
        # Create visualization only for single sequence
        if args.save_video or not args.all:
            # Load data again for visualization
            gt_poses_2d, _, _ = load_test_3d_data_from_dataset(args.sequence)
            frames = load_test_frames(args.sequence, args.num_frames)
            
            if gt_poses_2d is not None and frames is not None:
                # Process data for visualization
                min_frames = min(len(frames), len(gt_poses_2d), args.num_frames if args.num_frames else len(gt_poses_2d))
                frames = frames[:min_frames]
                gt_poses_2d = gt_poses_2d[:min_frames]
                
                yolo_poses_2d, _, _ = estimate_yolo_poses(model, frames, args.img_size, device)
                
                gt_poses_2d_pixel = convert_coordinates_to_pixels(gt_poses_2d, frames)
                gt_poses_2d_root_rel = make_root_relative_2d_pixel(gt_poses_2d_pixel, root_joint_idx=14)
                yolo_poses_2d_root_rel = make_root_relative_2d_pixel(yolo_poses_2d, root_joint_idx=14)
                
                print(f"\n🎬 Creating visualization...")
                try:
                    result = create_comparison_visualization(
                        gt_poses_2d_root_rel, yolo_poses_2d_root_rel, args.sequence, metrics, args)
                    
                    if result[0] is not None:
                        update_func, fig, min_frames = result
                        
                        if args.save_video:
                            print("Creating animation...")
                            ani = FuncAnimation(fig, update_func, frames=min_frames, 
                                              interval=300, repeat=True, blit=False)
                            
                            os.makedirs(args.output_dir, exist_ok=True)
                            model_name_clean = os.path.splitext(os.path.basename(args.model_path))[0]
                            output_path = os.path.join(args.output_dir, 
                                                     f'{args.sequence}_gt_vs_yolo_{model_name_clean}_comprehensive.gif')
                            ani.save(output_path, writer='pillow', fps=3, dpi=100)
                            print(f"✓ Animation saved to: {output_path}")
                            
                            update_func(0)
                            static_path = os.path.join(args.output_dir, 
                                                     f'{args.sequence}_gt_vs_yolo_{model_name_clean}_comprehensive.png')
                            plt.savefig(static_path, dpi=150, bbox_inches='tight')
                            print(f"✓ Static image saved to: {static_path}")
                            
                            plt.close(fig)
                        else:
                            print("Showing interactive visualization...")
                            update_func(0)
                            plt.show()
                            
                except Exception as e:
                    print(f"❌ Error creating visualization: {e}")

if __name__ == '__main__':
    main()