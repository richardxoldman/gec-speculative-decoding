"""
Detokenize Falko-MERLIN (data/falko_merlin/{train,dev}.json): the raw
MultiGEC-2025 markdown for this corpus ships pre-tokenized (space before
punctuation, e.g. "Herren , ihre Anzeige ... gesucht ."), unlike the other
corpora in this project. This produces natural running text instead, using
sacremoses' German MosesDetokenizer plus a few corpus-specific fixes it gets
wrong or leaves untouched:

- Straight double quotes ("): Moses' quote handling is unconditional
  (always glues to one side) and gets multi-quote sentences backwards, e.g.
  "der " Berliner Morgenpost " gelesen" -> 'der"Berliner Morgenpost"gelesen'
  instead of 'der "Berliner Morgenpost" gelesen'. Fixed with a stack-based
  matcher over „ / " / " that treats „ as always-opening, " as
  always-closing (for a „ or a plain-quote opener), and a bare " as opening
  if no quote is currently open, closing otherwise.
- Emoticons (":)" ";)"): Moses treats the ":" as sentence punctuation and
  glues it to the preceding word ("getrunken auch:)"). Protected before
  detokenization and restored after with a single leading space.
- "€" immediately followed by a letter (e.g. "500 €als"): Moses drops the
  trailing space after "€" as if it were closing punctuation; re-inserted.
- "´ 12" (apostrophe'd two-digit year, e.g. "16.03. ´ 12"): Moses leaves the
  space before the digit; collapsed to "´12".

Not fixed (rare, left as-is): stray "´" used as a stylistic quote mark
("der ´ neuen ´ EU-Ländern") or mid-word apostrophe with inconsistent
spacing ("Montag ´ s" vs. the far more common already-glued "geht´s").
Original tokenized files are kept as data/falko_merlin/{split}_tokenized.json.
"""

import json
import re
import shutil
from pathlib import Path

from sacremoses import MosesDetokenizer

DATA_DIR = Path("data/falko_merlin")
SPLITS = ["train", "dev"]

_detokenizer = MosesDetokenizer(lang="de")


def fix_quotes(text: str) -> str:
    parts = re.split(r'([„“"])', text)
    stack = []
    out = [parts[0]]
    for i in range(1, len(parts), 2):
        q = parts[i]
        seg = parts[i + 1] if i + 1 < len(parts) else ""
        if q == "„":
            out.append(q)
            stack.append("„")
            seg = seg.lstrip()
        elif q == "“":
            if stack and stack[-1] in ("„", '"'):
                stack.pop()
                out[-1] = out[-1].rstrip()
                out.append(q)
            else:
                stack.append("“")
                seg = seg.lstrip()
        else:  # plain "
            if stack:
                stack.pop()
                out[-1] = out[-1].rstrip()
                out.append(q)
            else:
                stack.append('"')
                out.append(q)
                seg = seg.lstrip()
        out.append(seg)
    return "".join(out)


def protect_emoticons(text: str) -> str:
    # ":)"/";)" may be tokenized as one token or split across "colon space paren"
    return re.sub(
        r"([:;])\s*\)",
        lambda m: f' __EMOTICON_{"WINK" if m.group(1) == ";" else "SMILE"}__ ',
        text,
    )


def restore_emoticons(text: str) -> str:
    text = text.replace("__EMOTICON_SMILE__", ":)").replace("__EMOTICON_WINK__", ";)")
    return re.sub(r"\s+([:;]\))", r" \1", text)


def fix_euro_glue(text: str) -> str:
    return re.sub(r"€(?=[A-Za-zÄÖÜäöüß])", "€ ", text)


def fix_apostrophe_year(text: str) -> str:
    return re.sub(r"´\s+(\d)", r"´\1", text)


def detokenize(text: str) -> str:
    text = fix_quotes(text)
    text = protect_emoticons(text)
    tokens = text.split()
    text = _detokenizer.detokenize(tokens)
    text = restore_emoticons(text)
    text = fix_euro_glue(text)
    text = fix_apostrophe_year(text)
    return text


def main():
    for split in SPLITS:
        path = DATA_DIR / f"{split}.json"
        backup_path = DATA_DIR / f"{split}_tokenized.json"
        if not backup_path.exists():
            shutil.copy(path, backup_path)

        examples = json.load(open(backup_path))
        detok = [
            {
                "id": e["id"],
                "source": detokenize(e["source"]),
                "reference": detokenize(e["reference"]),
            }
            for e in examples
        ]
        with open(path, "w") as f:
            json.dump(detok, f, ensure_ascii=False, indent=2)
        print(f"falko_merlin/{split}: {len(detok)} essays detokenized -> {path} "
              f"(tokenized backup: {backup_path})")


if __name__ == "__main__":
    main()
