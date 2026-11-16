#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -J "evaluate"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "H100|H200|A100-80G|RTX6000B"
#SBATCH --output=%x_%j.log
set -e
source slurm/environment.sh
uv run --extra "$SPEAKUP_EXTRA" python -m "evals.$MODULE" $ARGUMENTS
