#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=HRNet3DHP_Test
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

set -euo pipefail

echo "Starting HRNet evaluation on MPI-INF-3DHP"

# Use the same sampling profile as full evaluation.
export MPI3DHP_TRAIN_FRAME_STRIDE=10
export MPI3DHP_TEST_FRAME_STRIDE=1
export MPI3DHP_MAX_TRAIN_SAMPLES=0
export MPI3DHP_MAX_TEST_SAMPLES=0

echo "Train frame stride: ${MPI3DHP_TRAIN_FRAME_STRIDE}"
echo "Test frame stride: ${MPI3DHP_TEST_FRAME_STRIDE}"
echo "Max train samples: ${MPI3DHP_MAX_TRAIN_SAMPLES} (0 means unlimited)"
echo "Max test samples: ${MPI3DHP_MAX_TEST_SAMPLES} (0 means unlimited)"

# Update MODEL_FILE if your checkpoint is in a different location.
MODEL_FILE="output/mpi_inf_3dhp/pose_hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp/model_best.pth"

python tools/test.py \
  --cfg experiments/3DHP/hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp.yaml \
  TEST.MODEL_FILE "${MODEL_FILE}"
