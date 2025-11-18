#!/bin/bash

#SBATCH --account=beex-dtai-gh
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=0
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=1-2
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

### LOG INFO ###
#SBATCH --job-name=CLS_DINO_ssv2+imagenet
#SBATCH --output=logs/slurm/ini1k_eval/CLS_DINO_ssv2+imagenet%A-%a%A-%a.log
export RUN_NAME="CLS_USING_DINO_CODE_DINO_ssv2+imagenet"

mkdir -p logs/slurm/ini1k_eval/
module purge
ulimit -n 65535

torchrun --nproc_per_node=2 eval_knn.py \
--run_name ${RUN_NAME} \
--project_name debug \
--arch vit_base \
--batch_size_per_gpu 128 \
--train_samples_per_class -1 \
--val_samples_per_class -1 \
--pretrained_weights "/work/hdd/bcsi/ndaithankar/playground/dino/output/FULL_INI1K_dino_ssv2+imagenet_k_1024_1387349/checkpoint0015.pth"\