"""
Provenance only: step 1 of how the splits shipped in data/ were derived (followed by
detokenize_falko_merlin.py, resplit_small_train.py and subsample_dev_1000.py, all run
from the repository root). Not needed to run the benchmarks.

Convert raw MultiGEC-2025 essay files into the essay-level JSON format used by
benchmarking/benchmark_serving.py. Expected input layout (copied from the MultiGEC-2025
distribution):

    data/multigec-2025/ua-gec/uk-ua_gec-{orig,ref1}-{train,dev}.md
    data/multigec-2025/writeandimprove2024/en-writeandimprove2024-{orig,ref1}-{train,dev}.md
    data/falko-merlin/de-merlin-{orig,ref1}-{train,dev}.md

Each *.md file holds essays separated by "### essay_id = <id>" headers, with the essay
text (including internal newlines/paragraphs) following. orig-*.md holds the source
(uncorrected) essays, ref1-*.md holds the reference correction, both keyed by the same
essay_id. Essay text is kept whole (not sentence-split) since document-level newlines are
meaningful and splitting risked orig/ref1 sentence-count mismatches.

Writes data/{key}/{split}.json as a list of {"id", "source", "reference"} objects, aligned
by essay_id and sorted for reproducibility.
"""

import argparse
import json
from pathlib import Path

MULTIGEC_ROOT = Path("data/multigec-2025")
DATA_ROOT = Path("data")

# key -> (root, subdir, file prefix)
CORPORA = {
    "writeandimprove2024": (MULTIGEC_ROOT, "writeandimprove2024", "en-writeandimprove2024"),
    "ua_gec": (MULTIGEC_ROOT, "ua-gec", "uk-ua_gec"),
    "falko_merlin": (DATA_ROOT, "falko-merlin", "de-merlin"),
}

SPLITS = ["dev", "train"]


def md_to_dict(md: str) -> dict[str, str]:
    essay_dict = {}
    for essay in md.split("### essay_id = ")[1:]:
        essay_id, text = essay.split("\n", maxsplit=1)
        essay_dict[essay_id] = text.strip("\n")
    return essay_dict


def convert_split(root: Path, subdir: str, prefix: str, split: str) -> list[dict]:
    orig_path = root / subdir / f"{prefix}-orig-{split}.md"
    ref_path = root / subdir / f"{prefix}-ref1-{split}.md"

    sources = md_to_dict(orig_path.read_text())
    references = md_to_dict(ref_path.read_text())

    if sources.keys() != references.keys():
        missing_in_ref = sources.keys() - references.keys()
        missing_in_src = references.keys() - sources.keys()
        raise ValueError(
            f"essay_id mismatch between {orig_path} and {ref_path}: "
            f"missing_in_ref={missing_in_ref}, missing_in_src={missing_in_src}"
        )

    return [
        {"id": essay_id, "source": sources[essay_id], "reference": references[essay_id]}
        for essay_id in sorted(sources)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=list(CORPORA), default=None,
        help="Convert only this dataset key (default: all three)",
    )
    args = parser.parse_args()

    keys = [args.dataset] if args.dataset else list(CORPORA)

    for key in keys:
        root, subdir, prefix = CORPORA[key]
        out_dir = Path("data") / key
        out_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            examples = convert_split(root, subdir, prefix, split)
            out_path = out_dir / f"{split}.json"
            with open(out_path, "w") as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
            print(f"{key}/{split}: {len(examples)} essays -> {out_path}")


if __name__ == "__main__":
    main()
