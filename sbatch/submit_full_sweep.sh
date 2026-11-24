#!/usr/bin/env bash
# Submits the full benchmarking sweep reported in the paper as independent SLURM jobs via
# sbatch/serve_and_benchmark_sweep.sbatch:
#
#   - Models:      gemma4_12b, qwen3_8b
#   - Methods:     ngram (min_ngram in {1,2,3}), pa = PANgram (min_ngram in {1,2,3}),
#                  sad = AD (no min_ngram knob)
#   - K:           5, 10, 20, 30
#   - Datasets:    ua_gec, falko_merlin, writeandimprove2024 (all benchmarked within each
#                  job by serve_and_benchmark_sweep.sbatch, not looped here). Set DATASETS
#                  to benchmark a subset, e.g. DATASETS=ua_gec (see sbatch/datasets.sh).
#   - Batch sizes: 1, 4, 16, 64, 128 (set inside serve_and_benchmark_sweep.sbatch)
#   - Autoregressive baseline: once per model (not per K/min_ngram)
#
# Job count: 2 models x 4 K x 7 (method, min_ngram) combos = 56, plus 2 autoregressive
# baseline jobs = 58 total.
#
# Usage:
#   sbatch/submit_full_sweep.sh [model_key ...]
#
# Defaults to both gemma4_12b and qwen3_8b if no model_key is given. Pass one or more
# model_keys to submit a subset, e.g.:
#   sbatch/submit_full_sweep.sh gemma4_12b
#
# Run from a login node (submission itself needs no GPU). Checkpoints are taken from
# MODEL_PATHS in serve_and_benchmark_sweep.sbatch.

set -euo pipefail

MODEL_KEYS=("$@")
if [[ ${#MODEL_KEYS[@]} -eq 0 ]]; then
  MODEL_KEYS=(gemma4_12b qwen3_8b)
fi

K_VALUES=(5 10 20 30)
MIN_NGRAM_VALUES=(1 2 3)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$REPO_DIR/sbatch/serve_and_benchmark_sweep.sbatch"

cd "$REPO_DIR"
mkdir -p logs

# Check the datasets now, so a missing one fails here instead of in every queued job.
source sbatch/datasets.sh
check_datasets || exit 1
echo "Datasets: $DATASETS"

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  echo "Submitting $MODEL_KEY autoregressive baseline (k=0)"
  sbatch "$SBATCH_SCRIPT" "$MODEL_KEY" autoregressive 0 0

  for K in "${K_VALUES[@]}"; do
    for MIN_NGRAM in "${MIN_NGRAM_VALUES[@]}"; do
      echo "Submitting $MODEL_KEY ngram k=$K min_ngram=$MIN_NGRAM"
      sbatch "$SBATCH_SCRIPT" "$MODEL_KEY" ngram "$K" "$MIN_NGRAM"

      echo "Submitting $MODEL_KEY pa k=$K min_ngram=$MIN_NGRAM"
      sbatch "$SBATCH_SCRIPT" "$MODEL_KEY" pa "$K" "$MIN_NGRAM"
    done

    echo "Submitting $MODEL_KEY sad k=$K"
    sbatch "$SBATCH_SCRIPT" "$MODEL_KEY" sad "$K"
  done
done
