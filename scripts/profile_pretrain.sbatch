#!/bin/bash -l
#SBATCH --job-name=profile-pretrain
#SBATCH --partition=short,long
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/afrenk/speakup/runs/profile_pretrain_%j.log

set -euo pipefail
module load cuda/12.9.0/b3yzg3f ffmpeg/6.1.1/cup2q2r
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(which ffmpeg)")")/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/scratch/afrenk/huggingface
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONUNBUFFERED=1
cd "$HOME/School/CV/Project"
OUTPUT=/scratch/afrenk/speakup/profile/nsys
mkdir -p "$OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

for precision in bf16 fp32; do
    report="$OUTPUT/${precision}_${SLURM_JOB_ID}"
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o "$report" \
        "$HOME/.local/bin/uv" run --no-sync scripts/pretrain_mix.py \
        --catalog /archive/afrenk/Project-local/pretraining_pools/acquired_mix_20261028_v2 \
        --output "$OUTPUT/run_${precision}" --mode audio_visual --windows 512 --epochs 1 --normalizer_windows 64 \
        --workers 8 --precision "$precision" --max_steps 6 --cuda_profiler
    nsys stats --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum --format csv --force-export true \
        --output "$report" "$report.nsys-rep"
    echo "=== $precision top kernels"
    head -16 "${report}_cuda_gpu_kern_sum.csv" | cut -c1-220
done

for precision in fp32 bf16; do
    [ "${SKIP_TORCH_PROFILE:-0}" = 1 ] && break
    "$HOME/.local/bin/uv" run --no-sync scripts/pretrain_mix.py \
        --catalog /archive/afrenk/Project-local/pretraining_pools/acquired_mix_20261028_v2 \
        --output /scratch/afrenk/speakup/profile/torch_$precision --mode audio_visual --windows 4096 --epochs 1 \
        --normalizer_windows 128 --workers 12 --precision "$precision" --profile_steps 20
done
