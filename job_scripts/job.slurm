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
#SBATCH --job-name=VIT_H_dino_only_ssv2_k_1024
#SBATCH --output=logs/slurm/dino_recipe/VIT_H_dino_only_ssv2_k_1024%A-%a%A-%a.log
export RUN_NAME="VIT_H_dino_only_ssv2_k_1024"

OUTPUT_BASE="/work/hdd/bcsi/ndaithankar/playground/dino/output"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

mkdir -p logs/slurm/dino_recipe/
module purge
# ulimit -n 65535

NUM_GPUS=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=${NUM_GPUS} main_dino.py \
--run_name ${RUN_NAME} \
--project_name dino_recipe \
--arch vit_huge \
--batch_size_per_gpu 2 \
--acc_grad_steps 4 \
--epochs 100 \
--knn_freq 1 \
--out_dim 1024 \
--datasets "ssv2" \
--data_paths "/work/hdd/bcsi/ndaithankar/datasets/ssv2, /work/hdd/bcsi/ndaithankar/datasets/imagenet1k-hf, /work/hdd/beex/ndaithankar/datasets/ego4d/moment_clips"  \
--output_dir "${OUTPUT_DIR}" \