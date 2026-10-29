#!/bin/bash -l
#SBATCH --job-name=speakup-pretrain-mix
#SBATCH --partition=short,long
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100|H200|RTX6000B
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/afrenk/speakup/runs/pretrain_mix_%x_%j.log

set -euo pipefail
module load cuda/12.9.0/b3yzg3f ffmpeg/6.1.1/cup2q2r
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(which ffmpeg)")")/lib:${LD_LIBRARY_PATH:-}"
cd "$HOME/School/CV/Project"
export UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONUNBUFFERED=1
SPEAKUP_DATA=/scratch/afrenk/speakup

nvidia-smi --query-gpu=name,memory.total --format=csv
"$HOME/.local/bin/uv" run --no-sync scripts/pretrain_mix.py \
    --catalog /archive/afrenk/Project-local/pretraining_pools/acquired_mix_20261028_v2 \
    --output "$SPEAKUP_DATA/${RUN_NAME:-pretrain_mix}/$MODE" \
    --mode "$MODE" --windows "${WINDOWS:-40000}" --epochs "${EPOCHS:-2}" --batch_size 32 --workers 12 --device cuda
