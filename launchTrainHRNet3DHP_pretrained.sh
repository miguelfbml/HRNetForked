#!/bin/bash
#SBATCH --partition=gpu_min32gb     # Reserved partition
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=HRNet3DHP_Train_Pretrained
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err


echo "Starting HRNet training on MPI-INF-3DHP"

# Requested runtime sampling profile.
export MPI3DHP_TRAIN_FRAME_STRIDE=10
export MPI3DHP_TEST_FRAME_STRIDE=1
export MPI3DHP_MAX_TRAIN_SAMPLES=0
export MPI3DHP_MAX_TEST_SAMPLES=0

echo "Train frame stride: ${MPI3DHP_TRAIN_FRAME_STRIDE}"
echo "Test frame stride: ${MPI3DHP_TEST_FRAME_STRIDE}"
echo "Max train samples: ${MPI3DHP_MAX_TRAIN_SAMPLES} (0 means unlimited)"
echo "Max test samples: ${MPI3DHP_MAX_TEST_SAMPLES} (0 means unlimited)"


python tools/train_mpi_inf_3dhp.py \
  --modelDir output_pretrained \
  --logDir log_pretrained \
  --use-wandb \
  --wandb-project HRNet_MPI_INF_3DHP \
  --wandb-run-name hrnet_w48_mpi_inf_3dhp_pretrained_coco \
  --cfg experiments/3DHP/hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp.yaml \
  MODEL.PRETRAINED models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth

