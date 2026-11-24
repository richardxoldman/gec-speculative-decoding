# Prompt-Based Speculative Decoding for Grammatical Error Correction

Code, data and models for the paper *Prompt-Based Speculative Decoding for Grammatical
Error Correction* (arXiv: TODO). This README assumes you have read the paper.

- `proposers/`: PAN-gram and AD as [vLLM](https://github.com/vllm-project/vllm)
  `custom_class` proposers. vLLM itself is not modified.
- `benchmarking/`, `sbatch/`, `configs/`: the serving benchmark and the sweep behind every
  speed number in the paper.
- `data/`, `data_prep/`: the dataset splits and how they were made.
- [`MODELS.md`](MODELS.md): the finetuned models on Hugging Face.

## Paper names in the code

| Paper | `--method` | Implementation |
|---|---|---|
| autoregressive decoding | `autoregressive` | no `--speculative-config` |
| N-gram | `ngram` | vLLM's built-in `ngram` proposer |
| PAN-gram | `pa` | `proposers/gec_pa_ngram.py`, class `PANgramProposer` |
| AD | `sad` | `proposers/gec_sad_ngram.py`, class `SADProposer` |
| standard speculative decoding | `standard_sd` | Qwen3-0.6B draft model (Qwen3-8B only) |

- **Models:** Gemma4-12B = `gemma4_12b`, Qwen3-8B = `qwen3_8b`.
- **Datasets:** UA-GEC = `ua_gec`, Falko-MERLIN = `falko_merlin`, W&I =
  `writeandimprove2024`.
- **Symbols:**
  - *k* = `num_speculative_tokens` (`--k`).
  - *n* = `prompt_lookup_min` for N-gram, `GEC_NGRAM_MIN` for PAN-gram (`--tag minn<n>`).
  - *b* = `--batch-sizes`, the client's concurrency cap.

## Installation

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins every package of the environment the paper's numbers were
measured in (Linux x86_64, CUDA 12.9). `requirements.txt` lists only the direct
dependencies, at the same versions, and lets pip resolve the rest. Use it to run the
proposers in another environment. vLLM ≥ 0.24 is required for `custom_class`.

## Data and models

UA-GEC and Falko-MERLIN are included in full. The evaluation files are all named
`dev_resplit_1000.json`, but Falko-MERLIN's has only 530 essays (UA-GEC's has 1000).
Write & Improve 2024 cannot be
redistributed, so only its essay IDs are included. Rebuild its text from your own
MultiGEC-2025 copy, then build the benchmark prompts:

```bash
python data_prep/rebuild_writeandimprove2024.py \
    --multigec-dir /path/to/multigec-2025/english/writeandimprove2024
python data_prep/prepare_benchmark_prompts.py
```

For file formats, licences and citations, see [`data/README.md`](data/README.md).

The finetuned models are on Hugging Face. vLLM downloads them on first use, and
[`MODELS.md`](MODELS.md) has the details:

- Gemma4-12B: [`rstaruch/gemma4-12b-gec`](https://huggingface.co/rstaruch/gemma4-12b-gec)
- Qwen3-8B: [`rstaruch/qwen3-8b-gec`](https://huggingface.co/rstaruch/qwen3-8b-gec)
- Qwen3-0.6B draft for standard speculative decoding:
  [`rstaruch/qwen3-0.6b-gec-draft`](https://huggingface.co/rstaruch/qwen3-0.6b-gec-draft)

## Serving PAN-gram or AD

```bash
# PAN-gram, k=10, n=1
PYTHONPATH=$PWD/proposers GEC_NGRAM_MIN=1 GEC_NGRAM_MAX=5 \
vllm serve rstaruch/gemma4-12b-gec --language-model-only --speculative-config \
  '{"method": "custom_class", "model": "gec_pa_ngram.PANgramProposer", "num_speculative_tokens": 10}'

# AD, k=10
PYTHONPATH=$PWD/proposers \
vllm serve rstaruch/gemma4-12b-gec --language-model-only --speculative-config \
  '{"method": "custom_class", "model": "gec_sad_ngram.SADProposer", "num_speculative_tokens": 10}'
```

For Qwen3-8B, serve `rstaruch/qwen3-8b-gec` and replace `--language-model-only` with
`--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'`.

vLLM ignores `prompt_lookup_min`/`prompt_lookup_max` for `custom_class` proposers. PAN-gram
therefore reads *n* and the maximum n-gram length from the `GEC_NGRAM_MIN`/`GEC_NGRAM_MAX`
environment variables of the server process. Other options are documented in each
proposer's module docstring.

Both proposers find the end of the prompt from the chat template's generation marker, so
send requests through `/v1/chat/completions`. A request without the marker gets no draft;
correctness is unaffected. PAN-gram finds where the source text starts from the
instruction prefix in the paper's appendix (`INSTRUCTION_PREFIX` in `gec_pa_ngram.py`).
Edit it if you prompt your model differently.

## Benchmarking

`benchmarking/benchmark_serving.py` runs `vllm bench serve` against an already running
server for each batch size. It needs no GPU.

```bash
# PAN-gram, k=10, n=1, on UA-GEC
python benchmarking/benchmark_serving.py \
  --model gemma4_12b --method pa --k 10 --tag minn1 --dataset ua_gec \
  --host <host> --port <port>
```

`--served-model-name` defaults to the model's Hugging Face repo id from
`configs/models.yaml`. That is what the server reports when it is started as above. If
you served a local copy, pass the id from `curl http://<host>:<port>/v1/models`
(`data[0].id`). A wrong `--served-model-name` does not raise an error. Instead, it writes a
zero-throughput result. `--method`, `--k` and `--tag` only label the output, so they must
match how the server was launched.

### Reproducing the paper's sweep (SLURM)

First edit `VLLM_ENV` and the `#SBATCH` lines in `sbatch/serve_and_benchmark_sweep.sbatch`.
To serve local copies instead of the Hugging Face repos, also edit `MODEL_PATHS` and
`DRAFT_MODEL_PATHS` there. Then:

```bash
sbatch/submit_full_sweep.sh                 # autoregressive, N-gram, PAN-gram, AD; Gemma4-12B + Qwen3-8B (58 jobs)
sbatch/submit_qwen3_8b_standard_sd_sweep.sh # standard speculative decoding, Qwen3-8B (5 jobs)
```

The settings match the paper's Appendix A. The values themselves are in
`configs/experiments.yaml` and the sbatch script.

Each job benchmarks all three datasets. To run only some of them, for example without
W&I, set `DATASETS` (space- or comma-separated keys from `configs/datasets.yaml`) when
submitting:

```bash
DATASETS="ua_gec falko_merlin" sbatch/submit_full_sweep.sh
```

The submit scripts check that every selected dataset's benchmark file exists before
submitting anything.

### Output

Each run writes
`results/<gpu>/<model>/<method>/k<k>[_minn<n>]/<dataset>/serving_conc<b>.json`. Runs that
already have a result file are skipped unless `--force` is given.

How the paper's metrics map to these files:

- Accepted length is `spec_decode_acceptance_length`. It includes vLLM's bonus token.
- Acceptance rate is `spec_decode_acceptance_rate`.
- Speedup is `output_throughput_mean` divided by the autoregressive value.

## Tests

```bash
python proposers/test_gec_pa_ngram.py
python proposers/test_gec_sad_ngram.py
```

The tests run on CPU only. They compare each proposer's numba kernel against a brute-force
reference on random inputs.

## License

The code is licensed under [Apache-2.0](LICENSE). The datasets keep their own licences
(see [`data/README.md`](data/README.md)). The finetuned models are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see [`MODELS.md`](MODELS.md)),
and their base models keep their own licences.

## Citation

```bibtex
@misc{TODO,
  title  = {Prompt-Based Speculative Decoding for Grammatical Error Correction},
  author = {TODO},
  year   = {2026},
  eprint = {TODO},
  archivePrefix = {arXiv}
}
```
