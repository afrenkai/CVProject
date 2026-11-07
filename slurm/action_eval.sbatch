#!/bin/bash -l
#SBATCH --job-name=action-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/environment.sh
uv sync --frozen --extra "$SPEAKUP_EXTRA"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

RUN="${RUN:?set RUN to a run directory under $SPEAKUP_RUNS}"
CACHE="${CACHE:-$SPEAKUP_LATENTS/mvp_actions_v1/validation/d2e}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m "${MODULE:-evals.action_eval}" "$RUN" "$RUN/${CHECKPOINT:-checkpoint-4689}" "$CACHE" "$@"
