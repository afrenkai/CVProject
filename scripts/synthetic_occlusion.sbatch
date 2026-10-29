#!/bin/bash -l
#SBATCH --job-name=speakup-occlusion
#SBATCH --partition=short,long
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100|H200|RTX6000B
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/afrenk/speakup/runs/synthetic_occlusion_%j.log

set -euo pipefail
module load cuda/12.9.0/b3yzg3f
cd "$HOME/School/CV/Project"
export UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONUNBUFFERED=1
SPEAKUP_DATA=/scratch/afrenk/speakup
run() { "$HOME/.local/bin/uv" run --no-sync "$@"; }

nvidia-smi --query-gpu=name,memory.total --format=csv

[ -f "$SPEAKUP_DATA"/artifacts/synthetic_occlusion/train_seed1.pt ] || run scripts/cache_synthetic_occlusion.py --examples 4000 --seed 1 \
    --output "$SPEAKUP_DATA"/artifacts/synthetic_occlusion/train_seed1.pt --device cuda --batch_size 32 --workers 4
[ -f "$SPEAKUP_DATA"/artifacts/synthetic_occlusion/eval_seed2.pt ] || run scripts/cache_synthetic_occlusion.py --examples 1000 --seed 2 \
    --output "$SPEAKUP_DATA"/artifacts/synthetic_occlusion/eval_seed2.pt --device cuda --batch_size 32 --workers 4
run scripts/audio_contribution.py \
    --train_cache "$SPEAKUP_DATA"/artifacts/synthetic_occlusion/train_seed1.pt \
    --eval_cache "$SPEAKUP_DATA"/artifacts/synthetic_occlusion/eval_seed2.pt \
    --output "$SPEAKUP_DATA/runs/synthetic_occlusion_${SLURM_JOB_ID}.json" \
    --speech_quantile 0 --device cuda --steps 10000 --batch_size 32 --futures 8 --integration_steps 16 --log_every 1000
