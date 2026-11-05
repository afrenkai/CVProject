#!/bin/bash -l
#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/environment.sh
uv sync --frozen --extra "$SPEAKUP_EXTRA"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

CONFIG="${CONFIG:-configs/pretrain/stage1.yaml}"
LOCK="logs/.${LOCK_NAME:-$SLURM_JOB_NAME}"
NODES="${SLURM_NNODES:-1}"
GPUS=$(( NODES * SLURM_GPUS_ON_NODE ))
exec 9>"$LOCK.lock"
if ! flock -n 9; then
    HOLDER=$(cat "$LOCK.holder" 2>/dev/null || echo 0)
    [ "$GPUS" -gt "$HOLDER" ] || { echo "Run is held by a $HOLDER-GPU job; this job has $GPUS GPUs, not taking over"; exit 0; }
    echo "Asking the $HOLDER-GPU job to save and stop"
    touch "$LOCK.stop"
    flock -w 1800 9 || { echo "Lock still held after 30 min"; exit 1; }
fi
rm -f "$LOCK.stop"
echo "$GPUS" > "$LOCK.holder"
ATTENTION=$([ "$SPEAKUP_CAPABILITY" = 9 ] && echo flex_flash || echo flex)
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "$NODES node(s) x $SLURM_GPUS_ON_NODE GPU(s), head $HEAD, attention $ATTENTION, extra $SPEAKUP_EXTRA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks-per-node=1 torchrun \
    --nnodes "$NODES" --nproc_per_node "$SLURM_GPUS_ON_NODE" \
    --rdzv_backend c10d --rdzv_endpoint "$HEAD:29500" --rdzv_id "$SLURM_JOB_ID" \
    -m training.pretrain "$CONFIG" model.attention_backend="$ATTENTION" \
    trainer.stop_at="$(( SLURM_JOB_END_TIME - 900 ))" trainer.stop_file="$LOCK.stop" "$@"
