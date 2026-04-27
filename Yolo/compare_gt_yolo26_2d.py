"""
Compare Ground Truth and YOLO26l 2D poses with comprehensive metrics.

This is the YOLO26-specific entrypoint that reuses the existing comparison
pipeline from compare_gt_yolo_2d.py and defaults to the trained weights at
Yolo/weights/best.pt.

Usage:
python compare_gt_yolo26_2d.py --sequence TS1 --save-video
python compare_gt_yolo26_2d.py --all --num-frames 50
"""

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

from compare_gt_yolo_2d import (
    check_gpu_availability,
    convert_coordinates_to_pixels,
    create_comparison_visualization,
    estimate_yolo_poses,
    get_available_sequences,
    JOINT_NAMES,
    load_test_3d_data_from_dataset,
    load_test_frames,
    make_root_relative_2d_pixel,
    print_sequence_results,
    print_summary_results,
    process_single_sequence,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "comparison_output_yolo26"


def main():
    parser = argparse.ArgumentParser(
        description="Compare Ground Truth and YOLO26l 2D poses with comprehensive metrics"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="TS1",
        help="Sequence to compare (TS1, TS2, TS3, TS4, TS5, TS6)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained YOLO26l model (.pt file)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to process per sequence (if not specified with --all, uses all frames in batches)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run evaluation on all available sequences with all frames (unless --num-frames specified)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save comparison as GIF (only for single sequence)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size for YOLO inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0, etc.)",
    )
    args = parser.parse_args()

    print("YOLO26l Ground Truth vs 2D Pose Comparison with Comprehensive Metrics")
    print("=" * 80)

    if args.device == "auto":
        device, gpu_available = check_gpu_availability()
    else:
        device = args.device
        gpu_available = device.startswith("cuda") and torch.cuda.is_available()

    print(f"Model: {args.model_path}")
    print(f"Device: {device}")

    if args.all and args.num_frames is None:
        print("Mode: Process ALL FRAMES from ALL SEQUENCES (in memory-efficient batches)")
    elif args.all and args.num_frames is not None:
        print(f"Mode: Process {args.num_frames} frames from ALL SEQUENCES")
    else:
        frame_count = args.num_frames if args.num_frames is not None else 50
        print(f"Mode: Process {frame_count} frames from sequence {args.sequence}")

    print(f"Input size: {args.img_size}")
    print("Metrics: MPJPE, PCK (Percentage of Correct Keypoints), AUC (Area Under Curve), FPS, Inference Time")
    print("Coordinate system: Root-relative poses in pixel domain")
    print("=" * 80)

    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return

    print(f"🤖 Loading YOLO26l model from {args.model_path}...")
    try:
        model = YOLO(args.model_path)

        if gpu_available:
            print(f"📦 Moving model to {device}...")
            model.to(device)

        print("✓ YOLO26l model loaded successfully")
        print(f"Model device: {next(model.model.parameters()).device if hasattr(model, 'model') else 'Unknown'}")

    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return

    model_name = os.path.basename(args.model_path)

    if args.all:
        print("\n🔄 Processing all available sequences...")
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

                if gpu_available:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Error processing sequence {sequence}: {e}")
                all_metrics.append(None)

        print_summary_results(all_metrics, model_name)

    else:
        print(f"\n📂 Processing single sequence: {args.sequence}")

        if args.num_frames is None:
            args.num_frames = 50
            print(f"Using default {args.num_frames} frames for single sequence")

        metrics = process_single_sequence(model, args.sequence, args, device)

        if not metrics:
            print("❌ Failed to process sequence")
            return

        print("\n" + "=" * 60)
        print(f"DETAILED RESULTS FOR {args.sequence}")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Valid frames: {metrics['valid_frames']}/{metrics['total_frames']}")

        print("\nPERFORMANCE METRICS:")
        print(f"  FPS: {metrics['performance']['fps']:.2f}")
        print(f"  Mean inference time: {metrics['performance']['mean_inference_time']*1000:.2f} ms")
        print(f"  Total inference time: {metrics['performance']['total_inference_time']:.2f} seconds")
        print(f"  Processed frames: {metrics['performance']['processed_frames']}")

        print("\nMPJPE (Mean Per Joint Position Error):")
        print(f"  Average MPJPE: {metrics['avg_mpjpe']:.2f} pixels")

        print("\nPCK (Percentage of Correct Keypoints):")
        for key, value in metrics['pck_results'].items():
            print(f"  {key}: {value*100:.2f}%")

        print("\nAUC (Area Under Curve):")
        print(f"  AUC: {metrics['auc']:.4f}")

        print("\nJoint-wise errors (top 5 worst):")
        joint_error_pairs = [(i, metrics['joint_errors'][i], JOINT_NAMES[i]) for i in range(17)]
        joint_error_pairs.sort(key=lambda x: x[1], reverse=True)

        for joint_idx, error, name in joint_error_pairs[:5]:
            print(f"  {name} (joint {joint_idx}): {error:.2f} pixels")

        print("=" * 60)

        if args.save_video or not args.all:
            gt_poses_2d, _, _ = load_test_3d_data_from_dataset(args.sequence)
            frames = load_test_frames(args.sequence, args.num_frames)

            if gt_poses_2d is not None and frames is not None:
                min_frames = min(len(frames), len(gt_poses_2d), args.num_frames if args.num_frames else len(gt_poses_2d))
                frames = frames[:min_frames]
                gt_poses_2d = gt_poses_2d[:min_frames]

                yolo_poses_2d, _, _ = estimate_yolo_poses(model, frames, args.img_size, device)

                gt_poses_2d_pixel = convert_coordinates_to_pixels(gt_poses_2d, frames)
                gt_poses_2d_root_rel = make_root_relative_2d_pixel(gt_poses_2d_pixel, root_joint_idx=14)
                yolo_poses_2d_root_rel = make_root_relative_2d_pixel(yolo_poses_2d, root_joint_idx=14)

                print("\n🎬 Creating visualization...")
                try:
                    result = create_comparison_visualization(
                        gt_poses_2d_root_rel,
                        yolo_poses_2d_root_rel,
                        args.sequence,
                        metrics,
                        args,
                    )

                    if result[0] is not None:
                        update_func, fig, min_frames = result

                        if args.save_video:
                            print("Creating animation...")
                            from matplotlib.animation import FuncAnimation

                            ani = FuncAnimation(fig, update_func, frames=min_frames, interval=300, repeat=True, blit=False)

                            os.makedirs(args.output_dir, exist_ok=True)
                            model_name_clean = os.path.splitext(os.path.basename(args.model_path))[0]
                            output_path = os.path.join(
                                args.output_dir,
                                f"{args.sequence}_gt_vs_yolo26_{model_name_clean}_comprehensive.gif",
                            )
                            ani.save(output_path, writer="pillow", fps=3, dpi=100)
                            print(f"✓ Animation saved to: {output_path}")

                            update_func(0)
                            static_path = os.path.join(
                                args.output_dir,
                                f"{args.sequence}_gt_vs_yolo26_{model_name_clean}_comprehensive.png",
                            )
                            import matplotlib.pyplot as plt

                            plt.savefig(static_path, dpi=150, bbox_inches="tight")
                            print(f"✓ Static image saved to: {static_path}")

                            plt.close(fig)
                        else:
                            print("Showing interactive visualization...")
                            import matplotlib.pyplot as plt

                            update_func(0)
                            plt.show()

                except Exception as e:
                    print(f"❌ Error creating visualization: {e}")


if __name__ == "__main__":
    main()