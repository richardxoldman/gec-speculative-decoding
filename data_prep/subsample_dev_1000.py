"""
Build a smaller, size-matched dev set for ua_gec and writeandimprove2024 by
randomly subsampling 1000 examples each from their existing dev_resplit.json
(itself already produced by resplit_small_train.py). falko_merlin's
dev_resplit.json already has only 530 examples, so it's copied through
unchanged to keep all three datasets available under the same filename.

Does NOT touch dev_resplit.json / train_resplit.json -- writes
data/{key}/dev_resplit_1000.json as new files.
"""

import json
import random
from pathlib import Path

DATA_ROOT = Path("data")
SEED = 42
DEV_N = 1000

SUBSAMPLE_KEYS = ["ua_gec", "writeandimprove2024"]
COPY_KEYS = ["falko_merlin"]


def load(path: Path) -> list:
    return json.load(open(path))


def save(path: Path, examples: list) -> None:
    json.dump(examples, open(path, "w"), ensure_ascii=False, indent=2)


def subsample(dataset: str) -> None:
    root = DATA_ROOT / dataset
    dev = load(root / "dev_resplit.json")

    rng = random.Random(SEED)
    indices = list(range(len(dev)))
    rng.shuffle(indices)
    new_dev = [dev[i] for i in indices[:DEV_N]]

    save(root / "dev_resplit_1000.json", new_dev)
    print(f"{dataset}: dev_resplit {len(dev)} -> dev_resplit_1000 {len(new_dev)}")


def copy_through(dataset: str) -> None:
    root = DATA_ROOT / dataset
    dev = load(root / "dev_resplit.json")
    save(root / "dev_resplit_1000.json", dev)
    print(f"{dataset}: dev_resplit {len(dev)} -> dev_resplit_1000 {len(dev)} (unchanged, copied through)")


def main():
    for key in SUBSAMPLE_KEYS:
        subsample(key)
    for key in COPY_KEYS:
        copy_through(key)


if __name__ == "__main__":
    main()
