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
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=EGO4D+SSV2_dino_with_knn
#SBATCH --output=logs/slurm/dino_recipe/EGO4D+SSV2_dino_with_knn%A-%a.log
export RUN_NAME="EGO4D+SSV2_dino_with_knn"

mkdir -p logs/slurm/dino_recipe/
module purge
ulimit -n 65535

torchrun --nproc_per_node=2 main_dino.py \
--arch vit_base \
--batch_size_per_gpu 8 \
--epochs 20 \
--data_path "/work/hdd/beex/ndaithankar/datasets/ego4d/moment_clips, /work/hdd/bcsi/ndaithankar/datasets/ssv2"  \
--output_dir /work/hdd/bcsi/ndaithankar/playground/dino/output \
--run_name ${RUN_NAME} \