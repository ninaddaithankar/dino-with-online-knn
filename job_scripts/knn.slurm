#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
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
#SBATCH --job-name=knn_eval
#SBATCH --output=logs/slurm/ini1k_eval/%x_%A-%a.log

export RUN_NAME="knn_eval"

mkdir -p logs/slurm/ini1k_eval/
module purge
ulimit -n 65535

PRETRAINED_WEIGHTS="/path/to/checkpoint.pth"  # set to your checkpoint

torchrun --nproc_per_node=2 eval_knn.py \
--run_name ${RUN_NAME} \
--project_name debug \
--arch vit_base \
--batch_size_per_gpu 128 \
--train_samples_per_class -1 \
--val_samples_per_class -1 \
--pretrained_weights "${PRETRAINED_WEIGHTS}"
