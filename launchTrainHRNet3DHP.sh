#!/bin/bash
#SBATCH --partition=gpu_min32gb     # Reserved partition
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=HRNet3DHP_Train
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err


echo "Starting HRNet training on MPI-INF-3DHP"


python tools/train_mpi_inf_3dhp.py \
  --cfg experiments/3DHP/hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp.yaml \
  --use-wandb \
  --wandb-project HRNet_MPI_INF_3DHP \
  --wandb-run-name hrnet_w48_mpi_inf_3dhp

