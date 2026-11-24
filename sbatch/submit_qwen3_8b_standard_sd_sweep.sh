#!/usr/bin/env bash
# Submits the draft-model speculative decoding baseline (method=standard_sd) for qwen3_8b
# as independent SLURM jobs via sbatch/serve_and_benchmark_sweep.sbatch. The finetuned
# Qwen3-8B target is verified against the finetuned Qwen3-0.6B draft (DRAFT_MODEL_PATHS
# in that script). Same K / batch-size / dataset grid as submit_full_sweep.sh:
#
#   - K:           5, 10, 20, 30
#   - Batch sizes: 1, 4, 16, 64, 128 (serve_and_benchmark_sweep.sbatch's default)
#   - Datasets:    ua_gec, falko_merlin, writeandimprove2024 (all benchmarked within each
#                  job, not looped here). Set DATASETS to benchmark a subset, e.g.
#                  DATASETS=ua_gec (see sbatch/datasets.sh).
#   - No min_ngram sweep -- a draft model has no such knob, so one job per K.
#
# Job count: 4 K + 1 autoregressive baseline = 5 total. (The baseline is the same one
# submit_full_sweep.sh runs; benchmark_serving.py skips result files that already exist.)
#
# Usage:
#   sbatch/submit_qwen3_8b_standard_sd_sweep.sh
#
# Run from a login node (submission itself needs no GPU).

set -euo pipefail

MODEL_KEY="qwen3_8b"
K_VALUES=(5 10 20 30)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$REPO_DIR/sbatch/serve_and_benchmark_sweep.sbatch"

cd "$REPO_DIR"
mkdir -p logs

# Check the datasets now, so a missing one fails here instead of in every queued job.
source sbatch/datasets.sh
check_datasets || exit 1
echo "Datasets: $DATASETS"

echo "Submitting $MODEL_KEY autoregressive baseline (k=0)"
sbatch "$SBATCH_SCRIPT" "$MODEL_KEY" autoregressive 0 0

for K in "${K_VALUES[@]}"; do
  echo "Submitting $MODEL_KEY standard_sd k=$K"
  sbatch "$SBATCH_SCRIPT" "$MODEL_KEY" standard_sd "$K"
done
