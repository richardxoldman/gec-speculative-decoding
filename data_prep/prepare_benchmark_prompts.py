"""
Build benchmark-ready dev files with the training-time instruction prefix
baked into "source", for use by benchmarking/benchmark_serving.py.

benchmark_serving.py calls `vllm bench serve` with --skip-chat-template, so
it sends whatever is in "source" verbatim as the user-turn content -- it
does not add the instruction prefix itself. The finetuned models were trained
on

    "Correct all language errors in the following text. Return only the
    corrected text. Text to correct:\n{source}"

as the user message (the server applies the model's chat template on top),
so benchmark prompts must carry that same prefix or the served model won't
behave as it does in the paper's measurements.

Reads data/{key}/dev_resplit_1000.json for ua_gec, falko_merlin and
writeandimprove2024 and writes data/{key}/dev_resplit_1000_prompted.json
alongside it (source-only change; id/reference are carried through
unmodified). For writeandimprove2024, run
data_prep/rebuild_writeandimprove2024.py first.

Run from the repository root.
"""

import json
from pathlib import Path

DATA_ROOT = Path("data")
DATASETS = ["ua_gec", "falko_merlin", "writeandimprove2024"]

INSTRUCTION_PREFIX = "Correct all language errors in the following text. Return only the corrected text. Text to correct:\n"


def build_prompted(in_path: Path, out_path: Path) -> None:
    examples = json.load(open(in_path))
    prompted = [
        {**e, "source": INSTRUCTION_PREFIX + e["source"]}
        for e in examples
    ]
    json.dump(prompted, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"{in_path} ({len(examples)}) -> {out_path}")


def main():
    for dataset in DATASETS:
        in_path = DATA_ROOT / dataset / "dev_resplit_1000.json"
        if not in_path.exists():
            print(f"skipping {dataset}: {in_path} not found"
                  + (" (run data_prep/rebuild_writeandimprove2024.py first)"
                     if dataset == "writeandimprove2024" else ""))
            continue
        build_prompted(in_path, DATA_ROOT / dataset / "dev_resplit_1000_prompted.json")


if __name__ == "__main__":
    main()
