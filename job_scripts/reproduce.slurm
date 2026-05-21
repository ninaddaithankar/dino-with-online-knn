#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --mem-per-gpu=118G
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=dino_ds_pct_sweep
#SBATCH --output=logs/slurm/dino/%x_%A-%a.log

export RUN_NAME="dino_ds_pct_sweep_"
export MASTER_PORT=$((20000 + (${SLURM_ARRAY_JOB_ID:-0} % 9999) + ${SLURM_ARRAY_TASK_ID:-0}))

OUTPUT_BASE="./output"  # set to your output directory
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_"
mkdir -p "$OUTPUT_DIR"

module purge
ulimit -n 65535

NUM_GPUS=2

pct=(0.01 0.1 0.5)
val_epochs=(100 10 2)

# Set your dataset paths below
DATA_PATHS="/path/to/ssv2, /path/to/imagenet"  # comma-separated list

torchrun --master_port ${MASTER_PORT} --nproc_per_node=${NUM_GPUS} main_dino.py \
--run_name ${RUN_NAME}${pct[$SLURM_ARRAY_TASK_ID]} \
--project_name dino_recipe \
--minimal_augmentation False \
--use_ibot_masking False \
--data_pct ${pct[$SLURM_ARRAY_TASK_ID]} \
--arch vit_base \
--batch_size_per_gpu 8 \
--epochs 100 \
--knn_freq ${val_epochs[$SLURM_ARRAY_TASK_ID]} \
--out_dim 1024 \
--datasets "ssv2" \
--data_paths "${DATA_PATHS}" \
--output_dir "${OUTPUT_DIR}" \
--resume False \
