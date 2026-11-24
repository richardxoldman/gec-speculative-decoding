"""Correctness tests for gec_sad_ngram (pure CPU, no GPU/vLLM needed).

Differentially tests the numba kernels against naive brute-force references:

1. the unique-suffix (SAD) search vs. a brute-force "smallest q >= 1 whose
   suffix occurs exactly once inside the prompt" reference, including the
   no-draft outcomes (occurrences drop to 0, or still ambiguous at the
   GEC_SAD_MAX_SUFFIX cap);
2. marker detection (last occurrence, missing marker, marker at start/end);
3. the batch kernel incl. the no-marker path (proposes nothing, no fallback).

Run:  python proposers/test_gec_sad_ngram.py
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gec_sad_ngram import _batch_propose, _last_marker_end, _propose_one_sad


def brute_force_sad(tokens, prompt_len, max_suffix, max_model_len, k):
    """Smallest q >= 1 such that the context suffix of length q occurs exactly
    once in tokens[:prompt_len] (match end i <= prompt_len - 2, so at least
    one prompt token remains to draft). Unique -> draft k tokens after the
    match, capped at the prompt boundary; zero occurrences or still ambiguous
    at max_suffix -> no draft."""
    total = len(tokens)
    if total < 2 or prompt_len < 2:
        return []
    k = min(k, max_model_len - total)
    if k <= 0:
        return []
    for q in range(1, min(max_suffix, total) + 1):
        suffix = tokens[total - q:]
        matches = [
            i
            for i in range(q - 1, prompt_len - 1)
            if tokens[i - q + 1 : i + 1] == suffix
        ]
        if not matches:
            return []
        if len(matches) == 1:
            start = matches[0] + 1
            return tokens[start : start + min(k, prompt_len - start)]
    return []  # still ambiguous at the search cap


def run_kernel(tokens, prompt_len, max_suffix, max_model_len, k):
    out = np.zeros(k if k > 0 else 1, dtype=np.int32)
    cand = np.zeros(max(len(tokens), 1), dtype=np.int32)
    arr = np.asarray(tokens, dtype=np.int32)
    n = _propose_one_sad(arr, prompt_len, max_suffix, max_model_len, k, out, cand)
    return out[:n].tolist()


def test_differential(num_cases=20000, seed=42):
    rng = random.Random(seed)
    checked_nonempty = 0
    for case in range(num_cases):
        vocab = rng.choice([3, 5, 8, 30])  # small vocab => frequent matches
        total = rng.randint(1, 120)
        tokens = [rng.randrange(vocab) for _ in range(total)]
        prompt_len = rng.randint(0, total)
        max_suffix = rng.choice([1, 2, 4, 64])
        k = rng.choice([1, 4, 10])
        max_model_len = rng.choice([total + rng.randint(0, 20), 8192])

        expected = brute_force_sad(tokens, prompt_len, max_suffix, max_model_len, k)
        got = run_kernel(tokens, prompt_len, max_suffix, max_model_len, k)
        assert got == expected, (
            f"case {case}: tokens={tokens} prompt_len={prompt_len} "
            f"max_suffix={max_suffix} k={k} max_model_len={max_model_len}\n"
            f"expected={expected}\ngot     ={got}"
        )
        if expected:
            checked_nonempty += 1
    assert checked_nonempty > num_cases // 10, (
        f"only {checked_nonempty} non-empty cases -- test data too sparse"
    )
    print(f"differential: {num_cases} cases OK ({checked_nonempty} non-empty)")


def test_gec_shape_case():
    """A GEC-shaped case: output copies the prompt except one 'correction'.

    The 1-token suffix is ambiguous in the source; growing q makes it unique
    and the draft continues copying the source from there.
    """
    # prompt: instruction(9x7) + source + marker(90,91,92); token 6 appears
    # twice in the source so the q=1 suffix "6" is ambiguous.
    source = [1, 2, 3, 4, 5, 6, 7, 8, 6, 10, 11, 12]
    marker = [90, 91, 92]
    prompt = [7] * 9 + source + marker
    # generation so far: copy of source with token 5 "corrected" to 50,
    # currently positioned right after the first token 6
    generated = [1, 2, 3, 4, 50, 6]
    tokens = prompt + generated

    marker_arr = np.asarray(marker, dtype=np.int32)
    tokens_arr = np.asarray(tokens, dtype=np.int32)
    marker_end = _last_marker_end(tokens_arr, marker_arr)
    assert marker_end == len(prompt), (marker_end, len(prompt))

    got = run_kernel(tokens, marker_end, 64, 8192, 4)
    # q=1 suffix "6" matches the source twice (ambiguous); q=2 suffix "50,6"
    # matches nowhere (50 is the correction, absent from the source), so
    # occurrences drop from 2 straight to 0 -> no draft this step.
    assert got == [], got

    generated = [1, 2, 3, 4, 50, 6, 7, 8, 6]
    tokens = prompt + generated
    got = run_kernel(tokens, marker_end, 64, 8192, 4)
    # q=1 "6" ambiguous (two 6s in source); q=2 "8,6" unique at the second 6
    # -> draft continues the source: 10,11,12 then the marker's first token
    assert got == [10, 11, 12, 90], got
    print("gec-shaped case OK")


def test_marker():
    m = np.asarray([90, 91], dtype=np.int32)
    seq = lambda xs: np.asarray(xs, dtype=np.int32)
    assert _last_marker_end(seq([90, 91, 1, 2]), m) == 2
    assert _last_marker_end(seq([1, 90, 91]), m) == 3
    assert _last_marker_end(seq([90, 91, 5, 90, 91, 7]), m) == 5  # last one
    assert _last_marker_end(seq([1, 2, 3]), m) == -1
    assert _last_marker_end(seq([90]), m) == -1
    assert _last_marker_end(seq([1, 2]), np.zeros(0, dtype=np.int32)) == -1
    print("marker OK")


def test_batch_and_no_marker():
    max_model_len = 256
    k = 4
    marker = np.asarray([90, 91], dtype=np.int32)
    token_ids = np.zeros((3, max_model_len), dtype=np.int32)
    num_tokens = np.zeros(3, dtype=np.int32)

    # request 0: marker present, unique q=1 suffix -> draft from the prompt
    row0 = [1, 2, 3, 4, 5, 90, 91] + [3]
    token_ids[0, : len(row0)] = row0
    num_tokens[0] = len(row0)
    # request 1: no marker -> proposes nothing (SAD has no fallback)
    row1 = [1, 2, 3, 4, 1, 2, 3]
    token_ids[1, : len(row1)] = row1
    num_tokens[1] = len(row1)
    # request 2: marker present but suffix token absent from the prompt
    row2 = [1, 2, 3, 90, 91, 7]
    token_ids[2, : len(row2)] = row2
    num_tokens[2] = len(row2)

    draft_buf = np.zeros((3, k), dtype=np.int32)
    len_buf = np.zeros(3, dtype=np.int32)
    cand_buf = np.zeros(max_model_len, dtype=np.int32)
    num_no_marker = _batch_propose(
        np.asarray([0, 1, 2], dtype=np.int64), num_tokens, token_ids,
        marker, 64, max_model_len, k, draft_buf, len_buf, cand_buf,
    )
    assert num_no_marker == 1, num_no_marker
    # req 0: suffix "3" unique at prompt idx 2 -> draft 4,5,90,91
    assert len_buf[0] == 4 and draft_buf[0, :4].tolist() == [4, 5, 90, 91]
    # req 1: no marker -> nothing
    assert len_buf[1] == 0
    # req 2: suffix "7" not in prompt -> nothing
    assert len_buf[2] == 0
    print("batch/no-marker OK")


if __name__ == "__main__":
    test_marker()
    test_gec_shape_case()
    test_batch_and_no_marker()
    test_differential()
    print("all tests passed")
