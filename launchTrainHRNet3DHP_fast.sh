#!/bin/bash
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=HRNet3DHP_Fast
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

# Fast debug/prototyping run: downsample frames and train fewer epochs.
export MPI3DHP_TRAIN_FRAME_STRIDE=10
export MPI3DHP_TEST_FRAME_STRIDE=10
export MPI3DHP_MAX_TRAIN_SAMPLES=120000
export MPI3DHP_MAX_TEST_SAMPLES=5000

echo "Starting FAST HRNet training on MPI-INF-3DHP"
echo "Stride train/test: ${MPI3DHP_TRAIN_FRAME_STRIDE}/${MPI3DHP_TEST_FRAME_STRIDE}"
echo "Max samples train/test: ${MPI3DHP_MAX_TRAIN_SAMPLES}/${MPI3DHP_MAX_TEST_SAMPLES}"

python tools/train_mpi_inf_3dhp.py \
  --cfg experiments/3DHP/hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp.yaml \
  --use-wandb \
  --wandb-project HRNet_MPI_INF_3DHP \
  --wandb-run-name hrnet_w48_mpi_inf_3dhp_fast \
  MODEL.PRETRAINED '' \
  TRAIN.END_EPOCH 20 \
  TRAIN.LR_STEP [12,16] \
  PRINT_FREQ 50
