"""
Shrink train splits to a fixed-size sample (for finetuning) and push the
leftover examples into dev (for inference/eval), so there's enough held-out
data for benchmarking without wasting the rest of the annotated data.

- ua_gec, falko_merlin, writeandimprove2024: sample `TRAIN_N` random examples from train.json as
  the new train split; the remaining train examples plus the original dev.json
  become the new dev split.
- ua_gec additionally drops any essay with len(source) >= MAX_CHARS_UA_GEC
  (long-tail essays up to ~19k chars) from both the sampling pool and the
  leftover dev pool, before sampling -- so neither split contains them.

Writes data/{key}/{train,dev}_resplit.json as new files -- does NOT touch the
original train.json/dev.json.
"""

import json
import random
from pathlib import Path

DATA_ROOT = Path("data")
SEED = 42
TRAIN_N = 400
MAX_CHARS_UA_GEC = 5000


def load(path: Path) -> list:
    return json.load(open(path))


def save(path: Path, examples: list) -> None:
    json.dump(examples, open(path, "w"), ensure_ascii=False, indent=2)


def resplit_from_train(dataset: str, max_chars: int | None = None) -> None:
    root = DATA_ROOT / dataset
    train = load(root / "train.json")
    dev = load(root / "dev.json")

    if max_chars is not None:
        dropped_train = sum(1 for e in train if len(e["source"]) >= max_chars)
        dropped_dev = sum(1 for e in dev if len(e["source"]) >= max_chars)
        train = [e for e in train if len(e["source"]) < max_chars]
        dev = [e for e in dev if len(e["source"]) < max_chars]
        print(f"{dataset}: dropped {dropped_train} train / {dropped_dev} dev essays >= {max_chars} chars")

    rng = random.Random(SEED)
    indices = list(range(len(train)))
    rng.shuffle(indices)
    train_idx = set(indices[:TRAIN_N])

    new_train = [train[i] for i in indices[:TRAIN_N]]
    new_dev = [ex for i, ex in enumerate(train) if i not in train_idx] + dev

    save(root / "train_resplit.json", new_train)
    save(root / "dev_resplit.json", new_dev)
    print(f"{dataset}: train {len(train)}->{len(new_train)}, dev {len(dev)}->{len(new_dev)}")


def main():
    resplit_from_train("ua_gec", max_chars=MAX_CHARS_UA_GEC)
    resplit_from_train("falko_merlin")
    resplit_from_train("writeandimprove2024")


if __name__ == "__main__":
    main()
