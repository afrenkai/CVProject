#!/bin/bash -l
#SBATCH --job-name=speakup-audio
#SBATCH --partition=short,long
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100|H200|RTX6000B
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/afrenk/speakup/runs/audio_contribution_%j.log

set -euo pipefail
module load cuda/12.9.0/b3yzg3f ffmpeg/6.1.1/cup2q2r
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(which ffmpeg)")")/lib:${LD_LIBRARY_PATH:-}"
cd "$HOME/School/CV/Project"
export UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONUNBUFFERED=1
SPEAKUP_DATA=/scratch/afrenk/speakup
run() { "$HOME/.local/bin/uv" run --no-sync "$@"; }

nvidia-smi --query-gpu=name,memory.total --format=csv
run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"

[ -f "$SPEAKUP_DATA"/artifacts/ravdess_latents/eval_actors19-24.pt ] || run scripts/cache_ravdess_latents.py --first_actor 19 --last_actor 24 --videos_per_actor 10 \
    --output "$SPEAKUP_DATA"/artifacts/ravdess_latents/eval_actors19-24.pt --device cuda --batch_size 16 --workers 12
[ -f "$SPEAKUP_DATA"/artifacts/ravdess_latents/train_actors01-18.pt ] || run scripts/cache_ravdess_latents.py --first_actor 1 --last_actor 18 --videos_per_actor 20 \
    --output "$SPEAKUP_DATA"/artifacts/ravdess_latents/train_actors01-18.pt --device cuda --batch_size 16 --workers 12
run scripts/audio_contribution.py \
    --train_cache "$SPEAKUP_DATA"/artifacts/ravdess_latents/train_actors01-18.pt \
    --eval_cache "$SPEAKUP_DATA"/artifacts/ravdess_latents/eval_actors19-24.pt \
    --output "$SPEAKUP_DATA/runs/audio_contribution_${SLURM_JOB_ID}.json" \
    --device cuda --steps 10000 --batch_size 32 --futures 8 --integration_steps 16 --log_every 250
