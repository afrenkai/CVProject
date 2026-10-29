#!/bin/bash -l
#SBATCH --job-name=speakup-opencs2
#SBATCH --partition=short,long
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100|H200|RTX6000B
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/afrenk/speakup/runs/opencs2_eval_%j.log

set -euo pipefail
module load cuda/12.9.0/b3yzg3f ffmpeg/6.1.1/cup2q2r
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(which ffmpeg)")")/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/scratch/afrenk/huggingface
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONUNBUFFERED=1
cd "$HOME/School/CV/Project"
run() { "$HOME/.local/bin/uv" run --no-sync "$@"; }
DATA=/scratch/afrenk/speakup/opencs2
EVENTS="${EVENTS:-$DATA/turn_events_v1.parquet}"
TAG="$(basename "$EVENTS" .parquet)"
nvidia-smi --query-gpu=name,memory.total --format=csv

for split in train eval; do
    [ -f "$DATA/cache/${TAG}_${split}.pt" ] || run scripts/cache_opencs2.py --events "$EVENTS" --split "$split" \
        --output "$DATA/cache/${TAG}_${split}.pt" --device cuda --batch_size 16 --workers 12
done
run scripts/audio_contribution.py \
    --train_cache "$DATA/cache/${TAG}_train.pt" --eval_cache "$DATA/cache/${TAG}_eval.pt" \
    --output "/scratch/afrenk/speakup/runs/opencs2_${TAG}_${SLURM_JOB_ID}.json" \
    --speech_quantile 0 --probe_features change --likelihood_steps 0 --video_target change \
    --device cuda --steps 10000 --batch_size 32 --futures 8 --integration_steps 16 --log_every 1000
