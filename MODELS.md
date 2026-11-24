# Models

The finetuned checkpoints benchmarked in the paper. All three are **full fine-tunes**
(not LoRA adapters) of the listed base model, saved as a single bf16 `model.safetensors`.
They were trained for 2 epochs at learning rate 2e-5 on the 3 × 400 essays of
`data/{ua_gec,falko_merlin,writeandimprove2024}/train_resplit.json`, with the user turn

```
Correct all language errors in the following text. Return only the corrected text. Text to correct:
{source}
```

rendered through each model's own chat template. Qwen3 models were trained and are
served with thinking disabled (`enable_thinking=False`).

| Key | Role | Base model | Hugging Face repo | Size |
|---|---|---|---|---|
| `gemma4_12b` | target | [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) | [`rstaruch/gemma4-12b-gec`](https://huggingface.co/rstaruch/gemma4-12b-gec) | 23 GB |
| `qwen3_8b` | target | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | [`rstaruch/qwen3-8b-gec`](https://huggingface.co/rstaruch/qwen3-8b-gec) | 16 GB |
| `qwen3_06b` | draft model for the standard-SD baseline (with `qwen3_8b`) | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | [`rstaruch/qwen3-0.6b-gec-draft`](https://huggingface.co/rstaruch/qwen3-0.6b-gec-draft) | 1.2 GB |

Notes:

- `gemma4_12b` keeps the multimodal `Gemma4UnifiedForConditionalGeneration` architecture
  of its base model. It is served with `--language-model-only`, which drops the vision and
  audio towers.
- The two Qwen checkpoints are `Qwen3ForCausalLM`, and share the same tokenizer.
- `generation_config.json` keeps the base models' sampling defaults. All benchmarks
  override them with `temperature=0`, `seed=42`.
- The chat template (`chat_template.jinja`) is part of each checkpoint and **must be kept**.
  PANgram and AD locate the prompt/generation boundary from the chat template's
  generation-prompt marker.

`configs/models.yaml` (`checkpoint`, `draft_checkpoint`) and `MODEL_PATHS` /
`DRAFT_MODEL_PATHS` in `sbatch/serve_and_benchmark_sweep.sbatch` point to these repos, so
vLLM downloads them on first use. To serve local copies instead, change those entries to
the local directories. Each repo's model card has a ready-to-run `vllm serve` command
and a request example.

## License

The finetuned models are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use, with
attribution). The base models keep their own licences (Apache-2.0).
