#!/bin/bash -l
#SBATCH --job-name=speakup-synthetic-eval
#SBATCH --partition=short,long
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100|H200|RTX6000B
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/afrenk/speakup/runs/synthetic_eval_%x_%j.log

set -euo pipefail
module load cuda/12.9.0/b3yzg3f
cd "$HOME/School/CV/Project"
export UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONUNBUFFERED=1
SPEAKUP_DATA=/scratch/afrenk/speakup
SEED="${SEED:-0}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-0}"
NORMALIZER="${NORMALIZER:-checkpoint}"
PRETRAIN_DIR="${PRETRAIN_DIR:-$SPEAKUP_DATA/pretrain_mix}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_epoch2.pt}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
EMA_DECAY="${EMA_DECAY:-0}"
KEY_VALUE_HEADS="${KEY_VALUE_HEADS:-0}"
ATTENTION_WINDOW="${ATTENTION_WINDOW:-0}"
PRETRAINED=(--init_audio_visual "$PRETRAIN_DIR/audio_visual/$CHECKPOINT_NAME" --init_video_only "$PRETRAIN_DIR/video_only/$CHECKPOINT_NAME" --normalizer_source "$NORMALIZER")
CHECKPOINTS=()
STEPS=10000
case "$KIND" in
    scratch) ;;
    zero_shot) STEPS=0; CHECKPOINTS=("${PRETRAINED[@]}") ;;
    fine_tune) CHECKPOINTS=("${PRETRAINED[@]}") ;;
    *) echo "Unknown KIND=$KIND"; exit 1 ;;
esac
TAG="${KIND}_seed${SEED}_n${TRAIN_EXAMPLES}_${NORMALIZER}_$(basename "$PRETRAIN_DIR")_${LR_SCHEDULE}_ema${EMA_DECAY}_kv${KEY_VALUE_HEADS}_window${ATTENTION_WINDOW}"
echo "Run $TAG"

nvidia-smi --query-gpu=name,memory.total --format=csv
"$HOME/.local/bin/uv" run --no-sync scripts/audio_contribution.py \
    --train_cache "$SPEAKUP_DATA/artifacts/synthetic_occlusion/train_seed1.pt" \
    --eval_cache "$SPEAKUP_DATA/artifacts/synthetic_occlusion/eval_seed2.pt" \
    --output "$SPEAKUP_DATA/runs/synthetic_eval_${TAG}_${SLURM_JOB_ID}.json" \
    --speech_quantile 0 --device cuda --steps "$STEPS" --batch_size 32 --futures 8 --integration_steps 16 \
    --likelihood_steps 0 --log_every 1000 --video_target change --seed "$SEED" --train_examples "$TRAIN_EXAMPLES" \
    --lr_schedule "$LR_SCHEDULE" --ema_decay "$EMA_DECAY" \
    --key_value_heads "$KEY_VALUE_HEADS" --head_key_value_heads "$KEY_VALUE_HEADS" --attention_window "$ATTENTION_WINDOW" \
    "${CHECKPOINTS[@]}"
