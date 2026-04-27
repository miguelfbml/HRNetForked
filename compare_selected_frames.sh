#!/bin/bash
#
#SBATCH --partition=gpu_min11gb     # Reserved partition
#SBATCH --qos=gpu_min11gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=compareSelected  # Job name
#SBATCH --output=slurm_%x.%j.out    # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err     # File containing STDERR output. If ommited, use STDOUT.


echo "Running selected-frame GT vs HRNet comparison"

cd tools

HRNET_CFG="../experiments/3DHP/hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp.yaml"
HRNET_MODEL="../output/mpi_inf_3dhp/pose_hrnet/w48_384x288_adam_lr1e-3_mpi_inf_3dhp/model_best.pth"

python3 compare_gt_hrnet_selected_frames.py \
    --cfg "${HRNET_CFG}" \
    --sequence "TS1" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "${HRNET_MODEL}" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"


python3 compare_gt_hrnet_selected_frames.py \
    --cfg "${HRNET_CFG}" \
    --sequence "TS2" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "${HRNET_MODEL}" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"



python3 compare_gt_hrnet_selected_frames.py \
    --cfg "${HRNET_CFG}" \
    --sequence "TS3" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 5800 \
    --model-path "${HRNET_MODEL}" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"



python3 compare_gt_hrnet_selected_frames.py \
    --cfg "${HRNET_CFG}" \
    --sequence "TS4" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "${HRNET_MODEL}" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"




python3 compare_gt_hrnet_selected_frames.py \
    --cfg "${HRNET_CFG}" \
    --sequence "TS5" \
    --frames 30 60 90 120 150 180 210 240 270 300 \
    --model-path "${HRNET_MODEL}" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"




python3 compare_gt_hrnet_selected_frames.py \
    --cfg "${HRNET_CFG}" \
    --sequence "TS6" \
    --frames 50 100 150 200 250 300 350 400 450 490 \
    --model-path "${HRNET_MODEL}" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"