#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=HRNet3DHP_TS6_Video
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Starting HRNet MPI-INF-3DHP test video export for TS6"

# Keep sampling profile consistent with your training/test setup.
export MPI3DHP_TRAIN_FRAME_STRIDE=10
export MPI3DHP_TEST_FRAME_STRIDE=1
export MPI3DHP_MAX_TRAIN_SAMPLES=0
export MPI3DHP_MAX_TEST_SAMPLES=0

echo "Train frame stride: ${MPI3DHP_TRAIN_FRAME_STRIDE}"
echo "Test frame stride: ${MPI3DHP_TEST_FRAME_STRIDE}"
echo "Max train samples: ${MPI3DHP_MAX_TRAIN_SAMPLES} (0 means unlimited)"
echo "Max test samples: ${MPI3DHP_MAX_TEST_SAMPLES} (0 means unlimited)"

# Update if your checkpoint is in a different directory.
MODEL_FILE="output/mpi_inf_3dhp/pose_hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp/model_best.pth"

python tools/test_mpi_inf_3dhp_compare_metrics.py \
  --cfg experiments/3DHP/hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp.yaml \
  --save-json \
  --save-video \
  --video-sequence TS6 \
  --video-num-frames 300 \
  --video-fps 8 \
  --video-output-dir comparison_output \
  TEST.MODEL_FILE "${MODEL_FILE}"
