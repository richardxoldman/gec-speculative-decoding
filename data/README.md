# Data

These are the splits used in the paper; Section 4.2 ("Datasets") describes how they were
built and gives their statistics. All three datasets come from
[MultiGEC-2025](https://spraakbanken.github.io/multigec-2025/). They are document-level and
use the first reference (`ref1`). The models were finetuned on `train_resplit.json`, and
every number in the paper is measured on `dev_resplit_1000.json`.

| Folder | Paper name | Contents | Licence |
|---|---|---|---|
| `ua_gec/` | UA-GEC | `train_resplit.json`, `dev_resplit_1000.json` (full text) | CC BY 4.0 |
| `falko_merlin/` | Falko-MERLIN | same files, detokenized; MERLIN essays only | CC BY-SA 4.0 |
| `writeandimprove2024/` | W&I | `train_resplit_ids.json`, `dev_resplit_1000_ids.json` (essay IDs only) | not redistributable, see `NOTICE` |

The full-text files are JSON lists of `{"id", "source", "reference"}` objects.

The ID files are JSON lists of `{"id", "multigec_split"}` objects, in split order:
- `id` is the MultiGEC `essay_id`;
- `multigec_split` is the MultiGEC file the essay comes from (`train` or `dev`).

## Rebuilding W&I

```bash
python data_prep/rebuild_writeandimprove2024.py \
    --multigec-dir /path/to/multigec-2025/english/writeandimprove2024
```

This writes `train_resplit.json` and `dev_resplit_1000.json` into
`writeandimprove2024/`. Both are byte-identical to the files used in the paper and are
ignored by git. Then run `data_prep/prepare_benchmark_prompts.py` (see the main README).

## Additional info

- `data_prep/` holds the scripts that produced the splits. They are included as a record
  only; you don't need them to run the benchmarks.
- Sampling used seed 42. UA-GEC essays of 5000 or more characters were dropped before
  sampling.
- MultiGEC ships Falko-MERLIN pre-tokenized (e.g. `gesucht .`). We detokenized it with
  `sacremoses` plus small fixes (`data_prep/detokenize_falko_merlin.py`), so the text
  differs from the raw corpus.
- One detokenization quirk is kept because the models were trained on it: when a closing
  quote follows sentence-final punctuation, a stray space remains before the punctuation
  (`gesucht ."`).

## Licences and attribution

- **UA-GEC** was created by Grammarly and is licensed under CC BY 4.0 (`ua_gec/LICENSE`).
  - Cite: Syvokon et al., UNLP 2023, https://aclanthology.org/2023.unlp-1.12/
- **Falko-MERLIN** is licensed under CC BY-SA 4.0 (`falko_merlin/LICENSE`). Our
  detokenized adaptation is released under the same licence.
  - Cite MERLIN: Boyd et al., LREC 2014, https://aclanthology.org/L14-1488/, and
    Wisniewski et al. 2013.
  - Cite the splits: Boyd 2018, https://aclanthology.org/W18-6111
  - MERLIN was funded by the EU Lifelong Learning Programme, project
    518989-LLP-1-2011-1-DE-KA2-KA2MP.
- **W&I 2024** is by Nicholls, Caines and Buttery (Cambridge University Press &
  Assessment, University of Cambridge). Obtain it under its own licence.
- **MultiGEC-2025**: please also cite Masciolini et al. 2025,
  https://doi.org/10.1075/ijlcr.24033.mas, and the shared task,
  https://aclanthology.org/2025.nlp4call-1.1/
  - Dataset and terms of use: https://lt3.ugent.be/resources/multigec-dataset
