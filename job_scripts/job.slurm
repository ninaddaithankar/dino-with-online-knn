#!/bin/bash

#SBATCH --account=bcsi-dtai-gh
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=0
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=FULL_INI1K_dino_ssv2+ego4d_k_1024
#SBATCH --output=logs/slurm/dino_recipe/FULL_INI1K_dino_ssv2+ego4d_k_1024%A-%a%A-%a.log
export RUN_NAME="FULL_INI1K_dino_ssv2+ego4d_k_1024"

mkdir -p logs/slurm/dino_recipe/
module purge
ulimit -n 65535

torchrun --nproc_per_node=2 main_dino.py \
--run_name ${RUN_NAME} \
--arch vit_base \
--batch_size_per_gpu 8 \
--epochs 100 \
--knn_freq 1 \
--out_dim 1024 \
--datasets "ego4d, ssv2" \
--data_paths "/work/hdd/beex/ndaithankar/datasets/ego4d/moment_clips, /work/hdd/bcsi/ndaithankar/datasets/ssv2"  \
--output_dir /work/hdd/bcsi/ndaithankar/playground/dino/output \