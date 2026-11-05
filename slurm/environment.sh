module load cuda ffmpeg
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(which ffmpeg)")")/lib:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/bin:$PATH"
export SPEAKUP_ROOT=/scratch/afrenk/speakup
export SPEAKUP_CATALOG="$SPEAKUP_ROOT/catalog_stage1" SPEAKUP_RUNS="$SPEAKUP_ROOT/stage1" SPEAKUP_LATENTS="$SPEAKUP_ROOT/latents"
export HF_HOME=/scratch/afrenk/huggingface HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export UV_CACHE_DIR=/scratch/afrenk/uv_cache
export TRITON_CACHE_DIR="$SPEAKUP_ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$SPEAKUP_ROOT/cache/inductor"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
SPEAKUP_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
case "${SPEAKUP_EXTRA:-$SPEAKUP_CAPABILITY}" in
    8) SPEAKUP_EXTRA=ampere ;;
    9) SPEAKUP_EXTRA=hopper ;;
    10 | 12) SPEAKUP_EXTRA=ampere13 ;;
    ampere | ampere13 | hopper | cpu) ;;
    *) echo "No dependency extra for compute capability $SPEAKUP_CAPABILITY" >&2; exit 1 ;;
esac
export SPEAKUP_CAPABILITY SPEAKUP_EXTRA UV_PROJECT_ENVIRONMENT="$SPEAKUP_ROOT/venvs/$SPEAKUP_EXTRA"
set -a
source "${ENV_FILE:-.env}"
set +a
