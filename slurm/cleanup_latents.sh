#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=8g
#SBATCH -J "cleanup_latents"
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH --output=%x_%j.log
set -e
rm -rf /scratch/afrenk/speakup/latents/mvp_v1 /scratch/afrenk/speakup/latents/mvp_actions_v1 /scratch/afrenk/speakup/latents/mvp_audiomosaic_v1 /scratch/afrenk/speakup/latents/r6_actions_v1
