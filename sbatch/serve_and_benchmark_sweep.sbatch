#!/usr/bin/env bash
#SBATCH --job-name=gec_sweep
#SBATCH --gres=gpu:1                         # all reported numbers: 1x NVIDIA A100-SXM4-80GB
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/serve_sweep_%j.out
# Add --partition / --qos / --account lines for your cluster as needed.
#
# Serve one (model, method, K, min_ngram) configuration with vLLM and benchmark it on all
# three datasets (ua_gec, falko_merlin, writeandimprove2024) at all batch sizes, within a
# single job / server launch. Methods: autoregressive baseline, vLLM's stock N-gram SD,
# AD (code name `sad`), PANgram (code name `pa`), and draft-model SD (`standard_sd`).
#
# Usage (from the repository root):
#   sbatch sbatch/serve_and_benchmark_sweep.sbatch <model_key> <method> <k> [min_ngram] [ckpt_path] [batch_sizes]
#
# Datasets: set DATASETS when submitting to benchmark only some of them (keys from
# configs/datasets.yaml, space- or comma-separated; default all three; see
# sbatch/datasets.sh), e.g.
#   DATASETS=ua_gec sbatch sbatch/serve_and_benchmark_sweep.sbatch qwen3_8b pa 10 1
#
# <model_key>  gemma4_12b | qwen3_8b (must be a key in configs/models.yaml
#              and in MODEL_PATHS below, unless ckpt_path is given explicitly)
# <method>     autoregressive | ngram | sad | pa | standard_sd
#              - autoregressive: no --speculative-config; k and min_ngram are ignored
#                (pass 0 0 as placeholders).
#              - ngram: vLLM's stock CPU N-gram proposer (prompt_lookup_min=min_ngram,
#                prompt_lookup_max=5)
#              - sad:   AD, gec_sad_ngram.SADProposer (custom_class) -- has no min_ngram knob
#                (uniqueness determines match length); min_ngram arg is ignored
#              - pa:    PANgram, gec_pa_ngram.PANgramProposer (custom_class) --
#                GEC_NGRAM_MIN=min_ngram, GEC_NGRAM_MAX=5
#              - standard_sd: finetuned draft model proposes tokens, target verifies
#                (vLLM infers method="draft_model" from a plain HF-id/path "model" field --
#                see DRAFT_MODEL_PATHS below); min_ngram arg is ignored. Qwen3-8B only.
# <k>          num_speculative_tokens (ignored for autoregressive)
# [min_ngram]  prompt_lookup_min / GEC_NGRAM_MIN (ngram/pa only; ignored for sad,
#              standard_sd and autoregressive); default 2 if omitted
# [ckpt_path]  override MODEL_PATHS (HF repo id or local checkpoint directory). The
#              script checks the served name via /v1/models and fails on a mismatch.
# [batch_sizes] comma-separated concurrency levels passed to benchmark_serving.py
#              --batch-sizes; default "1,4,16,64,128" (the full sweep). Pass e.g. "128"
#              for a single-batch-size smoke test.
#
# Resumability: benchmark_serving.py skips any batch size whose serving_conc{N}.json
# already exists (see its --force flag), so a resubmitted job resumes instead of redoing
# finished work.

set -euo pipefail

MODEL_KEY="${1:?Usage: sbatch serve_and_benchmark_sweep.sbatch <model_key> <method> <k> [min_ngram] [ckpt_path]}"
METHOD="${2:?Usage: sbatch serve_and_benchmark_sweep.sbatch <model_key> <method> <k> [min_ngram] [ckpt_path]}"
K="${3:?Usage: sbatch serve_and_benchmark_sweep.sbatch <model_key> <method> <k> [min_ngram] [ckpt_path]}"
MIN_NGRAM="${4:-2}"
CKPT_OVERRIDE="${5:-}"
BATCH_SIZES="${6:-1,4,16,64,128}"

case "$METHOD" in
  autoregressive|ngram|sad|pa|standard_sd) ;;
  *) echo "Unknown method '$METHOD' -- must be one of autoregressive, ngram, sad, pa, standard_sd" >&2; exit 1 ;;
esac

# Finetuned target checkpoint per configs/models.yaml key (Hugging Face repo ids, see
# MODELS.md). Replace with local directories to serve from a local copy.
declare -A MODEL_PATHS=(
  [gemma4_12b]="rstaruch/gemma4-12b-gec"
  [qwen3_8b]="rstaruch/qwen3-8b-gec"
)

# Finetuned draft-model checkpoint per configs/models.yaml key -- only used by
# method=standard_sd.
declare -A DRAFT_MODEL_PATHS=(
  [qwen3_8b]="rstaruch/qwen3-0.6b-gec-draft"
)

if [[ -n "$CKPT_OVERRIDE" ]]; then
  MODEL_PATH="$CKPT_OVERRIDE"
else
  MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]:?Unknown model_key '$MODEL_KEY' -- add it to MODEL_PATHS or pass ckpt_path explicitly}"
fi

# Derived from SLURM_JOB_ID so concurrent jobs landing on the same multi-GPU node don't
# collide trying to bind the same port (HOST is always loopback).
JOB_ID="${SLURM_JOB_ID:-0}"
PORT=$((8100 + JOB_ID % 900))
HOST=127.0.0.1
NGRAM_MAX=5

# vLLM's internal torch.distributed rendezvous port (unrelated to the API --port above)
# is picked via a probe-then-bind pattern that can race when several `vllm serve`
# processes start on the same node at once (EADDRINUSE). VLLM_PORT makes it
# deterministic, so give each job its own base.
export VLLM_PORT=$((30000 + JOB_ID % 5000))

# TODO: point these at your checkout and at the environment built from requirements.txt.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VLLM_ENV="/path/to/venv"
VLLM_BIN="${VLLM_ENV}/bin/vllm"

cd "$REPO_DIR"
mkdir -p logs

# Datasets to benchmark (DATASETS from the environment, default all three). Checked
# before the server starts, so a missing dataset fails the job in seconds.
source sbatch/datasets.sh
check_datasets || exit 1

echo "Job ${SLURM_JOB_ID:-local}: model_key=$MODEL_KEY model_path=$MODEL_PATH method=$METHOD k=$K min_ngram=$MIN_NGRAM batch_sizes=$BATCH_SIZES datasets=\"$DATASETS\" node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Activate the serving environment (edit for conda / modules on your cluster).
source "$VLLM_ENV/bin/activate"

SERVER_LOG="logs/vllm_serve_${MODEL_KEY}_${METHOD}_k${K}_minn${MIN_NGRAM}_${SLURM_JOB_ID:-local}.log"

SERVE_ARGS=(
  "$MODEL_PATH"
  --port "$PORT"
  --max-model-len 10000
  --no-enable-prefix-caching
  --max-num-batched-tokens 10000
  --max-num-seqs 128
  --gpu-memory-utilization 0.8
)

# --language-model-only strips the vision/audio towers -- only applies to gemma4_12b
# (which is multimodal); qwen3_8b is text-only.
if [[ "$MODEL_KEY" == "gemma4_12b" ]]; then
  SERVE_ARGS+=(--language-model-only)
fi

# qwen3_8b is a hybrid thinking model: serve with thinking disabled (it was finetuned with
# enable_thinking=False) and the qwen3 reasoning parser, so /v1/chat/completions strips the
# (empty) <think></think> block out of message.content.
if [[ "$MODEL_KEY" == "qwen3_8b" ]]; then
  SERVE_ARGS+=(--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}')
fi

case "$METHOD" in
  autoregressive)
    ;;
  ngram)
    SERVE_ARGS+=(--speculative-config "{ \"method\": \"ngram\", \"num_speculative_tokens\": ${K}, \"prompt_lookup_min\": ${MIN_NGRAM}, \"prompt_lookup_max\": ${NGRAM_MAX} }")
    ;;
  sad)
    export PYTHONPATH="${REPO_DIR}/proposers"
    SERVE_ARGS+=(--speculative-config "{ \"method\": \"custom_class\", \"model\": \"gec_sad_ngram.SADProposer\", \"num_speculative_tokens\": ${K} }")
    ;;
  pa)
    export PYTHONPATH="${REPO_DIR}/proposers"
    export GEC_NGRAM_MIN="$MIN_NGRAM"
    export GEC_NGRAM_MAX="$NGRAM_MAX"
    SERVE_ARGS+=(--speculative-config "{ \"method\": \"custom_class\", \"model\": \"gec_pa_ngram.PANgramProposer\", \"num_speculative_tokens\": ${K} }")
    ;;
  standard_sd)
    DRAFT_MODEL_PATH="${DRAFT_MODEL_PATHS[$MODEL_KEY]:?Unknown model_key '$MODEL_KEY' for method=standard_sd -- add it to DRAFT_MODEL_PATHS}"
    SERVE_ARGS+=(--speculative-config "{ \"model\": \"${DRAFT_MODEL_PATH}\", \"num_speculative_tokens\": ${K} }")
    ;;
esac

"$VLLM_BIN" serve "${SERVE_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "Stopping vllm server (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vllm server to become healthy on ${HOST}:${PORT} ..."
for i in $(seq 1 180); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "Server is up after ${i}0s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vllm server process died during startup -- see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 10
done

if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "vllm server never became healthy within timeout -- see $SERVER_LOG" >&2
  exit 1
fi

# Confirm the served model name actually matches MODEL_PATH before trusting any result --
# a mismatch doesn't error loudly in benchmark_serving.py, it silently writes a
# 0-throughput "successful" result.
SERVED_NAME="$(curl -sf "http://${HOST}:${PORT}/v1/models" | "$VLLM_ENV/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
if [[ "$SERVED_NAME" != "$MODEL_PATH" ]]; then
  echo "Served model name '$SERVED_NAME' does not match expected MODEL_PATH '$MODEL_PATH' -- aborting." >&2
  exit 1
fi
echo "Confirmed served model name: $SERVED_NAME"

if [[ "$METHOD" == "autoregressive" ]]; then
  METHOD_ARGS=(--method autoregressive)
  TAG_ARGS=()
else
  METHOD_ARGS=(--method "$METHOD" --k "$K")
  if [[ "$METHOD" == "sad" || "$METHOD" == "standard_sd" ]]; then
    TAG_ARGS=()
  else
    TAG_ARGS=(--tag "minn${MIN_NGRAM}")
  fi
fi

for DATASET in $DATASETS; do
  echo "=== Benchmarking $MODEL_KEY | $METHOD | k=$K | min_ngram=$MIN_NGRAM | $DATASET (all batch sizes) ==="
  "$VLLM_ENV/bin/python" benchmarking/benchmark_serving.py \
    --model "$MODEL_KEY" \
    "${METHOD_ARGS[@]}" \
    "${TAG_ARGS[@]}" \
    --dataset "$DATASET" \
    --batch-sizes "$BATCH_SIZES" \
    --num-warmups 128 \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$MODEL_PATH" \
    --vllm-bin "$VLLM_BIN"
done

echo "Done."
