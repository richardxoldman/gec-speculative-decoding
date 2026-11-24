# Dataset selection, shared by serve_and_benchmark_sweep.sbatch and the submit_*.sh
# scripts. Source it from the repository root.
#
# DATASETS: keys from configs/datasets.yaml to benchmark, space- or comma-separated.
# Defaults to the paper's three datasets. Set it when submitting to benchmark a subset;
# sbatch passes it on to the job (it exports the submitting shell's environment by
# default), e.g.
#
#   DATASETS=ua_gec sbatch/submit_full_sweep.sh
#   DATASETS="ua_gec falko_merlin" sbatch sbatch/serve_and_benchmark_sweep.sbatch qwen3_8b pa 10 1

DATASETS="${DATASETS:-ua_gec falko_merlin writeandimprove2024}"
DATASETS="${DATASETS//,/ }"
export DATASETS

# Prints an error and returns 1 if a key in DATASETS is not in configs/datasets.yaml, or
# if its benchmark file (built by data_prep/prepare_benchmark_prompts.py) is missing.
check_datasets() {
  local key file known failed=0
  known="$(awk '/^  [^ #]/ { sub(":", "", $1); printf "%s ", $1 }' configs/datasets.yaml)"
  if [[ -z "${DATASETS// /}" ]]; then
    echo "DATASETS is empty -- set it to one or more of: $known" >&2
    return 1
  fi
  for key in $DATASETS; do
    file="$(awk -v k="$key" '
      /^  [^ #]/ { cur = ($1 == k ":") }
      cur && $1 == "path:" { p = $2 }
      cur && $1 == "benchmark_file:" { b = $2 }
      END { if (p != "" && b != "") print p "/" b }' configs/datasets.yaml)"
    if [[ -z "$file" ]]; then
      echo "Unknown dataset '$key' in DATASETS -- keys in configs/datasets.yaml: $known" >&2
      failed=1
    elif [[ ! -f "$file" ]]; then
      echo "Dataset '$key': $file not found. Build it with data_prep/prepare_benchmark_prompts.py." >&2
      if [[ "$key" == "writeandimprove2024" ]]; then
        echo "  W&I must first be rebuilt from your own MultiGEC-2025 copy with" \
             "data_prep/rebuild_writeandimprove2024.py (see README.md)." >&2
      fi
      failed=1
    fi
  done
  if (( failed )); then
    echo "To benchmark only some datasets, set DATASETS, e.g. DATASETS=\"ua_gec falko_merlin\"." >&2
    return 1
  fi
}
