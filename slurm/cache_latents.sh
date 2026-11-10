#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=96g
#SBATCH -J "cache_mvp_actions"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "H100|H200|RTX6000B"
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.log
set -e
source slurm/environment.sh
uv run --extra "$SPEAKUP_EXTRA" python -m data.latent_cache "${CONFIG:-configs/pretrain/mvp_large_actions.yaml}" --start $((SLURM_ARRAY_TASK_ID * ${COUNT:-25000})) --count "${COUNT:-25000}"
