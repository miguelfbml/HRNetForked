#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=Yolo26Test_CompareMetrics
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Starting YOLO26l test with compare metrics (MPJPE/PCK/AUC)"

cd Yolo

python compare_gt_yolo26_2d.py \
  --model-path weights/best.pt \
  --output-dir comparison_output_yolo26 \
  --all