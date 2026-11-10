#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=256g
#SBATCH -J "pretrain"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH -C "H100|H200|A100-80G|RTX6000B"
#SBATCH --output=%x_%j.log
set -e
source slurm/environment.sh
uv run --extra "$SPEAKUP_EXTRA" torchrun --standalone --nproc_per_node "$SLURM_GPUS_ON_NODE" -m training.pretrain "$CONFIG" model.attention_backend=$([ "$SPEAKUP_CAPABILITY" = 9 ] && echo flex_flash || echo flex)
