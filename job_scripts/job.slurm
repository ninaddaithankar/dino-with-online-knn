#!/bin/bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --mem-per-gpu=118G
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=dino_training
#SBATCH --output=logs/slurm/dino/%x_%A-%a.log

export RUN_NAME="dino_training"

OUTPUT_BASE="./output"  # set to your output directory
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

module purge
ulimit -n 65535

NUM_GPUS=2

# Set your dataset paths below. Unused entries can be left as empty strings.
DATA_PATHS="/path/to/ssv2, /path/to/imagenet"  # comma-separated list

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29792 --nproc_per_node=${NUM_GPUS} main_dino.py \
--run_name ${RUN_NAME} \
--project_name dino_training \
--local_crops_number 0 \
--minimal_augmentation False \
--num_teacher_views 2 \
--num_student_views 2 \
--imagenet_samples_per_class 1 \
--mask_ratio 0 \
--use_ibot_masking False \
--ibot_mask_ratio_min_max 0.5 0.51 \
--ibot_mask_sample_probability 1.0 \
--arch vit_base \
--batch_size_per_gpu 128 \
--epochs 100 \
--knn_freq 1000 \
--out_dim 1024 \
--datasets "imagenet" \
--data_paths "${DATA_PATHS}" \
--output_dir "${OUTPUT_DIR}" \
