"""
Rebuild the Write & Improve 2024 (English) splits used in the paper from your own
copy of the MultiGEC-2025 data.

The W&I 2024 text cannot be redistributed, so this repository ships only the essay
IDs of each split (data/writeandimprove2024/{train_resplit,dev_resplit_1000}_ids.json,
lists of {"id", "multigec_split"} in split order). `id` is the MultiGEC-2025
`essay_id` (from the "### essay_id = <id>" headers) and `multigec_split` says which
MultiGEC file (train or dev) the essay comes from.

Given the folder holding

    en-writeandimprove2024-orig-{train,dev}.md   (learner essays -> "source")
    en-writeandimprove2024-ref1-{train,dev}.md   (reference corrections -> "reference")

this writes data/writeandimprove2024/{train_resplit,dev_resplit_1000}.json as lists of
{"id", "source", "reference"} objects -- byte-identical to the files the paper's
English numbers were measured on.

Usage (from the repository root):
    python data_prep/rebuild_writeandimprove2024.py --multigec-dir /path/to/multigec-2025/writeandimprove2024
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path("data/writeandimprove2024")
PREFIX = "en-writeandimprove2024"
SPLITS = ["train_resplit", "dev_resplit_1000"]


def md_to_dict(md: str) -> dict[str, str]:
    # Same parser as data_prep/prepare_multigec_data.py.
    essay_dict = {}
    for essay in md.split("### essay_id = ")[1:]:
        essay_id, text = essay.split("\n", maxsplit=1)
        essay_dict[essay_id] = text.strip("\n")
    return essay_dict


def load_multigec_split(multigec_dir: Path, split: str) -> tuple[dict, dict]:
    orig = multigec_dir / f"{PREFIX}-orig-{split}.md"
    ref = multigec_dir / f"{PREFIX}-ref1-{split}.md"
    for path in (orig, ref):
        if not path.exists():
            sys.exit(f"error: {path} not found -- point --multigec-dir at the folder "
                     f"holding the MultiGEC-2025 Write & Improve 2024 .md files")
    return md_to_dict(orig.read_text()), md_to_dict(ref.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--multigec-dir", type=Path, required=True,
                        help="Folder with en-writeandimprove2024-{orig,ref1}-{train,dev}.md")
    args = parser.parse_args()

    corpus = {s: load_multigec_split(args.multigec_dir, s) for s in ("train", "dev")}

    for split in SPLITS:
        ids = json.load(open(DATA_DIR / f"{split}_ids.json"))
        examples, missing = [], []
        for entry in ids:
            sources, references = corpus[entry["multigec_split"]]
            essay_id = entry["id"]
            if essay_id not in sources or essay_id not in references:
                missing.append(f"{entry['multigec_split']}/{essay_id}")
                continue
            examples.append({"id": essay_id,
                             "source": sources[essay_id],
                             "reference": references[essay_id]})
        if missing:
            sys.exit(f"error: {len(missing)} essay(s) of {split} not found in "
                     f"{args.multigec_dir}, e.g. {missing[:5]}")

        out_path = DATA_DIR / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"{split}: {len(examples)} essays -> {out_path}")


if __name__ == "__main__":
    main()
