#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=192g
#SBATCH -J "pretrain"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "[H200|H100|A100-80G|RTX6000B|L40S]"
#SBATCH --output=%x_%j.log
set -e
srun bash -c 'source slurm/environment.sh && uv run --extra "$SPEAKUP_EXTRA" torchrun --nnodes "$SLURM_NNODES" --nproc_per_node 1 --rdzv_backend c10d --rdzv_id "$SLURM_JOB_ID" --rdzv_endpoint "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1):$((20000 + SLURM_JOB_ID % 20000))" -m training.pretrain "$CONFIG" model.attention_backend=$([ "$SPEAKUP_CAPABILITY" = 9 ] && echo flex_flash || echo flex)'
