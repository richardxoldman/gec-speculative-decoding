"""
Online (serving) speed benchmarking via `vllm bench serve` against an already-running
vLLM OpenAI-compatible server (started separately, e.g. `vllm serve ...`).

Sweeps concurrency levels from experiments.yaml's batch_sizes, runs num_runs repeats per
level, and writes aggregated latency/throughput (and, for SD methods, acceptance-rate)
stats to results/{gpu}/{model}/{method}/k{k}/{dataset}/serving_conc{N}.json.
"""

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import yaml

AGGREGATE_FIELDS = [
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "mean_e2el_ms",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
]

SPEC_DECODE_FIELDS = [
    "spec_decode_acceptance_length",
    "spec_decode_acceptance_rate",
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_gpu_label() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        return out.replace(" ", "_").lower()
    except Exception:
        return "unknown_gpu"


def load_dataset(dataset_cfg: dict) -> list[str]:
    # benchmark_file (if set) already has the training-time instruction
    # prefix baked into "source" -- see data_prep/prepare_benchmark_prompts.py.
    # --skip-chat-template means this script sends "source" verbatim, so
    # falling back to data_file would silently drop that prefix.
    filename = dataset_cfg.get("benchmark_file", dataset_cfg["data_file"])
    data_path = Path(dataset_cfg["path"]) / filename
    with open(data_path) as f:
        examples = json.load(f)
    return [e["source"] for e in examples]


def write_prompts_jsonl(sources: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for source in sources:
            f.write(json.dumps({"prompt": source}) + "\n")


def run_bench_serve(
    *,
    vllm_bin: str,
    host: str,
    port: int,
    served_model_name: str,
    dataset_path: Path,
    max_new_tokens: int,
    num_prompts: int,
    max_concurrency: int,
    warmup_requests: int,
    seed: int,
    result_dir: Path,
    result_filename: str,
) -> dict:
    cmd = [
        vllm_bin, "bench", "serve",
        "--backend", "openai-chat",
        "--host", host,
        "--port", str(port),
        "--endpoint", "/v1/chat/completions",
        "--model", served_model_name,
        "--dataset-name", "custom",
        "--dataset-path", str(dataset_path),
        "--skip-chat-template",
        "--custom-output-len", str(max_new_tokens),
        "--num-prompts", str(num_prompts),
        "--no-oversample",
        "--max-concurrency", str(max_concurrency),
        "--num-warmups", str(warmup_requests),
        "--seed", str(seed),
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-dir", str(result_dir),
        "--result-filename", result_filename,
    ]
    subprocess.run(cmd, check=True)

    with open(result_dir / result_filename) as f:
        return json.load(f)


def aggregate_runs(runs: list[dict]) -> dict:
    aggregated = {}
    for field in AGGREGATE_FIELDS:
        values = [r[field] for r in runs]
        aggregated[f"{field}_mean"] = statistics.mean(values)
        aggregated[f"{field}_std"] = statistics.pstdev(values)

    for field in SPEC_DECODE_FIELDS:
        values = [r[field] for r in runs if field in r]
        if values:
            aggregated[f"{field}_mean"] = statistics.mean(values)
            aggregated[f"{field}_std"] = statistics.pstdev(values)

    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Key from configs/models.yaml")
    parser.add_argument("--method", required=True, choices=["autoregressive", "standard_sd", "ngram", "sad", "pa"])
    parser.add_argument("--dataset", required=True, help="Key from configs/datasets.yaml")
    parser.add_argument("--k", type=int, default=None, help="Speculative tokens the running server was launched with (ignored for autoregressive)")
    parser.add_argument("--host", default="127.0.0.1", help="Host of the already-running vLLM server")
    parser.add_argument("--port", type=int, default=8000, help="Port of the already-running vLLM server")
    parser.add_argument("--served-model-name", default=None, help="Model name the server reports at /v1/models (defaults to the model's finetuned `checkpoint` in configs/models.yaml)")
    parser.add_argument("--batch-sizes", default=None, help="Comma-separated concurrency levels to override experiments.yaml's benchmark.batch_sizes (e.g. '64')")
    parser.add_argument("--vllm-bin", default="vllm", help="Path to the vllm CLI to use for `vllm bench serve` (this is purely an HTTP client against the running server, so its version doesn't need to match the served model's vllm version)")
    parser.add_argument("--force", action="store_true", help="Re-run and overwrite serving_conc{N}.json files that already exist (default: skip batch sizes already benchmarked, so a preempted/requeued job resumes instead of redoing finished work)")
    parser.add_argument("--num-runs", type=int, default=None, help="Override experiments.yaml's benchmark.num_runs (e.g. for quicker ad-hoc sweeps)")
    parser.add_argument("--num-warmups", type=int, default=None, help="Override experiments.yaml's benchmark.warmup_requests (e.g. to avoid a concurrent duplicate-prompt warmup burst)")
    parser.add_argument("--tag", default=None, help="Suffix appended to the k{k} output directory (e.g. 'mnbt8192') to keep ad-hoc config-ablation runs from overwriting the canonical results")
    parser.add_argument("--gpu-label", default=None, help="Override the auto-detected GPU label (needed when benchmarking a remote server from a client node with no local GPU, e.g. nvidia-smi unavailable)")
    parser.add_argument("--models-cfg", default="configs/models.yaml")
    parser.add_argument("--datasets-cfg", default="configs/datasets.yaml")
    parser.add_argument("--experiments-cfg", default="configs/experiments.yaml")
    args = parser.parse_args()

    if args.method != "autoregressive" and args.k is None:
        parser.error("--k is required unless --method is autoregressive")

    models_cfg = load_config(args.models_cfg)
    datasets_cfg = load_config(args.datasets_cfg)
    exp_cfg = load_config(args.experiments_cfg)

    model_cfg = models_cfg["targets"][args.model]
    dataset_cfg = datasets_cfg["datasets"][args.dataset]
    bench_cfg = exp_cfg["benchmark"]

    served_model_name = args.served_model_name or model_cfg.get("checkpoint") or model_cfg["hf_id"]
    k_label = "none" if args.method == "autoregressive" else str(args.k)
    if args.tag:
        k_label = f"{k_label}_{args.tag}"

    gpu_label = args.gpu_label or get_gpu_label()
    out_dir = Path("results") / gpu_label / args.model / args.method / f"k{k_label}" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = (
        [int(b) for b in args.batch_sizes.split(",")]
        if args.batch_sizes is not None
        else bench_cfg["batch_sizes"]
    )
    num_runs = args.num_runs if args.num_runs is not None else bench_cfg["num_runs"]
    warmup_requests = args.num_warmups if args.num_warmups is not None else bench_cfg["warmup_requests"]

    sources = load_dataset(dataset_cfg)
    num_prompts = len(sources)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dataset_path = tmpdir / "prompts.jsonl"
        write_prompts_jsonl(sources, dataset_path)

        for batch_size in batch_sizes:
            out_path = out_dir / f"serving_conc{batch_size}.json"
            if out_path.exists() and not args.force:
                print(f"Skipping {args.model} | {args.method} | k={args.k} | concurrency={batch_size} (already exists at {out_path}, use --force to re-run)")
                continue

            print(f"Serving-benchmarking {args.model} | {args.method} | k={args.k} | concurrency={batch_size}")
            runs = []
            for run_idx in range(num_runs):
                result = run_bench_serve(
                    vllm_bin=args.vllm_bin,
                    host=args.host,
                    port=args.port,
                    served_model_name=served_model_name,
                    dataset_path=dataset_path,
                    max_new_tokens=models_cfg["common"]["max_new_tokens"],
                    num_prompts=num_prompts,
                    max_concurrency=batch_size,
                    warmup_requests=warmup_requests,
                    seed=models_cfg["common"]["seed"],
                    result_dir=tmpdir,
                    result_filename=f"run{run_idx}.json",
                )
                runs.append(result)

            aggregated = aggregate_runs(runs)
            aggregated.update({
                "model": args.model,
                "method": args.method,
                "dataset": args.dataset,
                "k": args.k,
                "gpu": gpu_label,
                "batch_size": batch_size,
                "num_runs": num_runs,
                "num_prompts": num_prompts,
            })

            with open(out_path, "w") as f:
                json.dump(aggregated, f, indent=2)
            print(f"  ttft={aggregated['mean_ttft_ms_mean']:.1f}ms  throughput={aggregated['output_throughput_mean']:.1f} tok/s  -> {out_path}")


if __name__ == "__main__":
    main()
