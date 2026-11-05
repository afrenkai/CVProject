#!/bin/bash -l
#SBATCH --job-name=serve
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/environment.sh
export UV_PROJECT_ENVIRONMENT="$SPEAKUP_ROOT/venvs/serving" SGLANG_EXTERNAL_MODEL_PACKAGE=inference.plugin
export TORCHINDUCTOR_CACHE_DIR="$SPEAKUP_ROOT/cache/serving_inductor" TRITON_CACHE_DIR="$SPEAKUP_ROOT/cache/serving_triton"
uv sync --project serving --frozen
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
export CUDA_HOME="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"

RUN="${RUN:-$SPEAKUP_RUNS/mvp_v1}"
CHECKPOINT="${CHECKPOINT:-$RUN/checkpoint-4689}"
BUNDLE="${BUNDLE:-$SPEAKUP_ROOT/bundles/speakup-$(basename "$RUN")-$(basename "$CHECKPOINT")}"
CACHE="${CACHE:-$SPEAKUP_LATENTS/$(basename "$RUN")/validation}"
GPUS="${SLURM_GPUS_ON_NODE:-1}"
API_PORT=$(( 20000 + RANDOM % 10000 ))
WEB_PORT=$(( API_PORT + 500 ))

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -e "$BUNDLE/model_index.json" ] || python -m inference.export "$RUN" "$CHECKPOINT" "$BUNDLE"
echo "Serving $BUNDLE with $GPUS replica(s) on $(hostname):$API_PORT"
if [ "${PARALLEL:-replicas}" = ulysses ]; then
    LAYOUT=(--num-gpus "$GPUS" --dp-size 1 --sp-degree "$GPUS" --ulysses-degree "$GPUS" --ring-degree 1)
else
    LAYOUT=(--num-gpus "$GPUS" --dp-size "$GPUS")
fi
echo "Layout ${PARALLEL:-replicas}: ${LAYOUT[*]}"
sglang serve --model-type diffusion --model-path "$BUNDLE" "${LAYOUT[@]}" \
    --host 0.0.0.0 --port "$API_PORT" --warmup-mode off "$@" &
SERVER=$!
trap 'kill $(jobs -p) 2>/dev/null' EXIT
until curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null; do
    kill -0 "$SERVER" 2>/dev/null || { echo "Server exited before becoming healthy"; exit 1; }
    sleep 10
done
echo "Server healthy"
if [ -n "${SMOKE:-}" ]; then
    python -m inference.client --url "http://127.0.0.1:$API_PORT" \
        --cache "$CACHE/d2e" --window 0
    python -m inference.client --url "http://127.0.0.1:$API_PORT" \
        --cache "$CACHE/kinetics400" --window 30 --candidates 32 --samples 8
fi
if [ -n "${BENCHMARK:-}" ]; then
    python -m inference.client --url "http://127.0.0.1:$API_PORT" \
        --cache "$CACHE/d2e" --benchmark "$BENCHMARK" ${FIRST_ROW_ONLY:+--first-row-only}
fi
if [ -n "${WEB:-}" ]; then
    echo "Web app on $(hostname):$WEB_PORT, API on $(hostname):$API_PORT"
    python -m inference.webapp --api "http://127.0.0.1:$API_PORT" \
        --cache "$CACHE" --port "$WEB_PORT" &
    until curl -sf "http://127.0.0.1:$WEB_PORT/contract" >/dev/null; do sleep 5; done
    curl -sf "http://127.0.0.1:$WEB_PORT/forecast?dataset=d2e&window=0&samples=4" | head -c 600
    echo
    wait "$SERVER"
fi
