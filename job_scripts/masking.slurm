#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --mem-per-gpu=118G
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=1-4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=ibot_masking_sweep
#SBATCH --output=logs/slurm/masking/%x_%A-%a.log

export RUN_NAME="ibot_masking_sweep_"
export MASTER_PORT=$((20000 + ($SLURM_JOB_ID % 20000)))

OUTPUT_BASE="./output"  # set to your output directory

mkdir -p logs/slurm/masking/
module purge

mask_ratio=(0.1 0.2 0.3 0.4 0.5)
mask_ratio_high=(0.11 0.21 0.31 0.41 0.51)

OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}${mask_ratio[$SLURM_ARRAY_TASK_ID]}_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${OUTPUT_BASE}/${RUN_NAME}${mask_ratio[$SLURM_ARRAY_TASK_ID]}"

NUM_GPUS=2

# Set your dataset paths below
DATA_PATHS="/path/to/ssv2, /path/to/imagenet"  # comma-separated list

torchrun --nproc_per_node=${NUM_GPUS} --master_port ${MASTER_PORT} main_dino.py \
--run_name ${RUN_NAME}${mask_ratio[$SLURM_ARRAY_TASK_ID]} \
--project_name masking_sweep \
--local_crops_number 0 \
--minimal_augmentation False \
--num_teacher_views 2 \
--num_student_views 2 \
--imagenet_samples_per_class -1 \
--use_ibot_masking True \
--ibot_mask_ratio_min_max ${mask_ratio[$SLURM_ARRAY_TASK_ID]} ${mask_ratio_high[$SLURM_ARRAY_TASK_ID]} \
--ibot_mask_sample_probability 1.0 \
--arch vit_base \
--batch_size_per_gpu 128 \
--epochs 100 \
--knn_freq 1 \
--out_dim 1024 \
--datasets "ssv2" \
--data_paths "${DATA_PATHS}" \
--output_dir "${OUTPUT_DIR}" \
--checkpoint_pth "${CHECKPOINT_DIR}/checkpoint.pth" \
