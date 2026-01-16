#!/bin/bash

#SBATCH --account=bdta-dtai-gh
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
#SBATCH --job-name=CENTER_CROP_only_ssv2_k_1024
#SBATCH --output=logs/slurm/dino_recipe/CENTER_CROP_only_ssv2_k_1024%A-%a%A-%a.log
export RUN_NAME="CENTER_CROP_only_ssv2_k_1024"

OUTPUT_BASE="/shared/nas2/ninadd2/repos/dino-recipe/output"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

mkdir -p logs/slurm/dino_recipe/
module purge
# ulimit -n 65535

NUM_GPUS=2

CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port=29512 --nproc_per_node=${NUM_GPUS} main_dino.py \
--run_name ${RUN_NAME} \
--project_name dino_recipe \
--minimal_augmentation True \
--local_crops_number 0 \
--arch vit_base \
--batch_size_per_gpu 16 \
--epochs 100 \
--knn_freq 1 \
--out_dim 1024 \
--datasets "ssv2" \
--data_paths "/shared/nas2/ninadd2/datasets/ssv2, /shared/nas/data/m1/shared-resource/ego4d/moment_clips, /work/hdd/bcsi/ndaithankar/datasets/imagenet-1k-hf, /work/hdd/beex/ndaithankar/datasets/ego4d/processed_fullscale, /work/hdd/bcsi/ndaithankar/datasets/new-kinetics/k400"  \
--output_dir "${OUTPUT_DIR}" \