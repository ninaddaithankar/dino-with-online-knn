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
#SBATCH --gpus-per-node=4

### LOG INFO ###
#SBATCH --job-name=HIGH_BS_FULL_INI1K_dino_ssv2+ego4d_k_1024
#SBATCH --output=logs/slurm/dino_recipe/HIGH_BS_FULL_INI1K_dino_ssv2+ego4d_k_1024%A-%a%A-%a.log
export RUN_NAME="HIGH_BS_FULL_INI1K_dino_ssv2+ego4d_k_1024"

OUTPUT_BASE="/work/hdd/bcsi/ndaithankar/playground/dino/output"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

mkdir -p logs/slurm/dino_recipe/
module purge
ulimit -n 65535

NUM_GPUS=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=${NUM_GPUS} main_dino.py \
--run_name ${RUN_NAME} \
--project_name debug \
--arch vit_base \
--batch_size_per_gpu 8 \
--acc_grad_steps 2 \
--epochs 100 \
--knn_freq 1 \
--out_dim 1024 \
--datasets "ssv2, ego4d" \
--data_paths "/work/hdd/bcsi/ndaithankar/datasets/ssv2,  /work/hdd/beex/ndaithankar/datasets/ego4d/moment_clips, /work/hdd/bcsi/ndaithankar/datasets/imagenet1k-hf"  \
--output_dir "${OUTPUT_DIR}" \