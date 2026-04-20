#!/bin/bash
#SBATCH --partition=gpu_min11gb   # Reserved partition
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=Yolo26Train_Enhanced
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Starting Enhanced YOLO training on MPI-INF-3DHP for superior keypoint accuracy"

cd Yolo

python train_yolo26l_pose.py \
    --base-path /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --annotations-path ../motion3d/data_train_3dhp.npz \
    --output-path /nas-ctm01/datasets/public/mpi_inf_3dhp_Yolo26 \
    --epochs 100 \
    --batch-size 32 \
    --img-size 640 \
    --device 0 \
    --workers 8 \
    --patience 20 \
    --cache disk \
    --use-wandb \
    --force-reprocess \
    --wandb-project YOLO_MPI_3DHP_26L