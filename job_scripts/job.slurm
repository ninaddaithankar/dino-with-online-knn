#!/bin/bash

#SBATCH --account=beig-dtai-gh
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=0
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=1-2
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

### LOG INFO ###
#SBATCH --job-name=HIGHER_BS_1024_dino_ego4d+ssv2_k_
#SBATCH --output=logs/slurm/dino_recipe/HIGHER_BS_1024_dino_ego4d+ssv2_k_%A-%a%A-%a.log
export RUN_NAME="HIGHER_BS_1024_dino_ego4d+ssv2_k_"

mkdir -p logs/slurm/dino_recipe/
module purge
ulimit -n 65535

k=(32768 8192 1024 256)

torchrun --nproc_per_node=4 main_dino.py \
--run_name ${RUN_NAME}${k[${SLURM_ARRAY_TASK_ID}]} \
--arch vit_base \
--batch_size_per_gpu 8 \
--acc_grad_steps 2 \
--epochs 100 \
--knn_freq 1 \
--out_dim ${k[${SLURM_ARRAY_TASK_ID}]} \
--data_path "/work/hdd/beex/ndaithankar/datasets/ego4d/moment_clips, /work/hdd/bcsi/ndaithankar/datasets/ssv2"  \
--output_dir /work/hdd/bcsi/ndaithankar/playground/dino/output \