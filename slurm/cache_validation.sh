#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=96g
#SBATCH -J "cache_mvp_actions_validation"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "H100|H200|RTX6000B"
#SBATCH --output=%x_%j.log
set -e
source slurm/environment.sh
uv run --extra "$SPEAKUP_EXTRA" python -m data.latent_cache "${CONFIG:-configs/pretrain/mvp_large_actions.yaml}" --split validation data.validation_clips=512
