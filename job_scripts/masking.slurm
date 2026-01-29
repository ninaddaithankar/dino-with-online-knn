#!/bin/bash

#SBATCH --account=beex-dtai-gh
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=0
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=0-9
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=GRID_MASK_no_augs_imagenet_k_1024_mask_ratio_
#SBATCH --output=logs/slurm/dino_recipe/GRID_MASK_no_augs_imagenet_k_1024_mask_ratio_%A-%a%A-%a.log
export RUN_NAME="GRID_MASK_no_augs_imagenet_k_1024_mask_ratio_"

OUTPUT_BASE="/shared/nas2/ninadd2/repos/dino-recipe/output"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

mkdir -p logs/slurm/dino_recipe/
module purge
# ulimit -n 65535

mask_ratio=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

NUM_GPUS=2

torchrun --nproc_per_node=${NUM_GPUS} main_dino.py \
--run_name ${RUN_NAME}${mask_ratio[$SLURM_ARRAY_TASK_ID]} \
--project_name dino_recipe \
--local_crops_number 0 \
--minimal_augmentation True \
--num_teacher_views 1 \
--num_student_views 2 \
--mask_ratio ${mask_ratio[$SLURM_ARRAY_TASK_ID]} \
--arch vit_base \
--batch_size_per_gpu 128 \
--epochs 100 \
--knn_freq 1 \
--out_dim 1024 \
--datasets "imagenet" \
--data_paths "/work/hdd/bcsi/ndaithankar/datasets/imagenet-1k-hf, /shared/nas2/ninadd2/datasets/ssv2, /shared/nas/data/m1/shared-resource/ego4d/moment_clips, /work/hdd/beex/ndaithankar/datasets/ego4d/processed_fullscale, /work/hdd/bcsi/ndaithankar/datasets/new-kinetics/k400"  \
--output_dir "${OUTPUT_DIR}" \