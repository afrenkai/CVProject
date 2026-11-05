#!/bin/bash -l
#SBATCH --job-name=cache-latents
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/environment.sh
uv sync --frozen --extra "$SPEAKUP_EXTRA"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

CONFIG="${CONFIG:-configs/pretrain/mvp.yaml}"
OUTPUT="${OUTPUT:?set OUTPUT to the latent cache directory, e.g. $SPEAKUP_LATENTS/mvp_v1}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m data.latent_cache "$CONFIG" "$OUTPUT" \
    --split "${SPLIT:-train}" --start "${START:-0}" --count "${COUNT:-25000}" "$@"
