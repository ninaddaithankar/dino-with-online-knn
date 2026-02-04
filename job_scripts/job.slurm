#!/bin/bash

#SBATCH --account=beex-dtai-gh
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
#SBATCH --job-name=INI1k_1pct_MASKED_jitter+blur_imagenet_k_1024_mask_ratio_0.5
#SBATCH --output=logs/slurm/dino_recipe/INI1k_1pct_MASKED_jitter+blur_imagenet_k_1024_mask_ratio_0.5%A-%a%A-%a.log
export RUN_NAME="INI1k_1pct_MASKED_jitter+blur_imagenet_k_1024_mask_ratio_0.5"

OUTPUT_BASE="/shared/nas2/ninadd2/repos/dino-recipe/output"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

mkdir -p logs/slurm/dino_recipe/
module purge
ulimit -n 65535

NUM_GPUS=2

CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port=29790 --nproc_per_node=${NUM_GPUS} main_dino.py \
--run_name ${RUN_NAME} \
--project_name philosophical_backing_experiment \
--local_crops_number 0 \
--minimal_augmentation True \
--num_teacher_views 1 \
--num_student_views 2 \
--imagenet_samples_per_class 12 \
--mask_ratio 0.5 \
--arch vit_base \
--batch_size_per_gpu 128 \
--epochs 1000 \
--knn_freq 100 \
--out_dim 1024 \
--datasets "imagenet" \
--data_paths "/shared/nas2/ninadd2/datasets/ssv2, /shared/nas/data/m1/shared-resource/ego4d/moment_clips, /work/hdd/bcsi/ndaithankar/datasets/imagenet-1k-hf, /work/hdd/beex/ndaithankar/datasets/ego4d/processed_fullscale, /work/hdd/bcsi/ndaithankar/datasets/new-kinetics/k400"  \
--output_dir "${OUTPUT_DIR}" \