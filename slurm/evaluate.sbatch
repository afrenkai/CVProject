#!/bin/bash -l
#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/environment.sh
uv sync --frozen --extra "$SPEAKUP_EXTRA"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

RUN="${RUN:?set RUN to a run directory under $SPEAKUP_RUNS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
while [ "$(date +%s)" -lt "$(( SLURM_JOB_END_TIME - 3600 ))" ]; do
    for checkpoint in $(ls -d "$RUN"/${PATTERN:-checkpoint-*000} 2>/dev/null | sort -t- -k2 -n); do
        [ -e "$checkpoint/evaluated" ] && continue
        echo "Evaluating $checkpoint"
        torchrun --standalone --nproc_per_node 1 -m evals.evaluate "$RUN" "$checkpoint" "$@" \
            || echo "Evaluation of $checkpoint failed"
    done
    [ -n "${ONCE:-}" ] && break
    sleep 600
done
