#!/bin/bash

#SBATCH --account=beig-dtai-gh
#SBATCH --partition=ghx4
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
#SBATCH --job-name=IBOT_INI1k_100pct_MASKED_all_aug_no_local_crop_imagenet_k_1024_mask_ratio_
#SBATCH --output=logs/slurm/philosophical_backing_experiment/IBOT_INI1k_100pct_MASKED_all_aug_no_local_crop_imagenet_k_1024_mask_ratio_%A-%a%A-%a.log
export RUN_NAME="IBOT_INI1k_100pct_MASKED_all_aug_no_local_crop_imagenet_k_1024_mask_ratio_"
export MASTER_PORT=$((20000 + ($SLURM_JOB_ID % 20000))) # To avoid port conflicts in case of multiple jobs

OUTPUT_BASE="/work/hdd/bcsi/ndaithankar/playground/dino/output"

mkdir -p "$OUTPUT_DIR"

mkdir -p logs/slurm/philosophical_backing_experiment/
module purge
# ulimit -n 65535

mask_ratio=(0.1 0.2 0.3 0.4 0.5)
mask_ratio_high=(0.11 0.21 0.31 0.41 0.51)

OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_${mask_ratio[$SLURM_ARRAY_TASK_ID]}_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${OUTPUT_BASE}/${RUN_NAME}_${mask_ratio[$SLURM_ARRAY_TASK_ID]}"

NUM_GPUS=2
torchrun --nproc_per_node=${NUM_GPUS} --master_port ${MASTER_PORT} main_dino.py \
--run_name ${RUN_NAME}${mask_ratio[$SLURM_ARRAY_TASK_ID]} \
--project_name philosophical_backing_experiment \
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
--datasets "imagenet" \
--data_paths "/work/hdd/bcsi/ndaithankar/datasets/imagenet-1k-hf, /shared/nas2/ninadd2/datasets/ssv2, /shared/nas/data/m1/shared-resource/ego4d/moment_clips, /work/hdd/beex/ndaithankar/datasets/ego4d/processed_fullscale, /work/hdd/bcsi/ndaithankar/datasets/new-kinetics/k400"  \
--output_dir "${OUTPUT_DIR}" \
--checkpoint_dir "${CHECKPOINT_DIR}" \