"""Correctness tests for gec_pa_ngram (pure CPU, no GPU/vLLM needed).

Differentially tests the numba kernels against naive brute-force references:

1. marker detection: last occurrence (end-of-prompt) and first occurrence
   (instruction prefix), including first-vs-last divergence;
2. target-position math: pure-Python re-implementation of the frac/offset
   formula, including round-half-up (not banker's) behavior;
3. `_propose_one_pa` vs. a brute-force "longest suffix n-gram inside the
   prompt, nearest-to-target tie-break" reference over 20k randomized cases
   (with the earliest-position mode randomly mixed in);
4. byte-identical equivalence to gec_prompt_ngram._propose_one when the
   position-aware tie-break is disabled (test-only cross-import);
5. deterministic tie selection on a hand-built 3-way tie;
6. a GEC-shaped case where the selected draft flips with generation progress;
7. `_batch_propose` rows covering both markers, missing end marker, and
   missing prefix marker (fallback + counters).

Run:  python proposers/test_gec_pa_ngram.py
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gec_pa_ngram import (
    _batch_propose,
    _first_marker_end,
    _last_marker_end,
    _propose_one_pa,
)


def compute_target_pos(generated, source_len, prefix_end):
    """Pure-Python mirror of the target-position math in _batch_propose."""
    if source_len <= 0:
        return None
    frac = generated / source_len
    frac = min(max(frac, 0.0), 1.0)
    offset = int(frac * source_len + 0.5)  # round half up, NOT round()
    offset = min(offset, source_len)
    return prefix_end + offset


def brute_force_pa(tokens, prompt_len, min_n, max_n, max_model_len, k,
                   target_pos=None):
    """Longest suffix n-gram in [min_n, max_n] matched inside tokens[:prompt_len].
    Collects *all* tied longest-length draft starts; picks
    argmin(|start - target_pos|, start) when a target is given, else the
    earliest start. Drafts follow the match, capped at the prompt boundary."""
    total = len(tokens)
    if total < min_n or prompt_len < min_n + 1:
        return []
    k = min(k, max_model_len - total)
    if k <= 0:
        return []
    for length in range(min(max_n, total - 1), min_n - 1, -1):
        suffix = tokens[total - length:]
        # match at s covers [s, s+length); needs >= 1 draft token in prompt
        starts = [
            s + length
            for s in range(0, prompt_len - length)
            if tokens[s:s + length] == suffix
        ]
        if starts:
            if target_pos is None:
                start = min(starts)
            else:
                start = min(starts, key=lambda st: (abs(st - target_pos), st))
            return tokens[start:start + min(k, prompt_len - start)]
    return []


def run_kernel(tokens, prompt_len, min_n, max_n, max_model_len, k,
               target_pos=None):
    out = np.zeros(k if k > 0 else 1, dtype=np.int32)
    pos_buf = np.zeros(max(len(tokens), 1), dtype=np.int32)
    arr = np.asarray(tokens, dtype=np.int32)
    n = _propose_one_pa(
        arr, prompt_len, min_n, max_n, max_model_len, k,
        target_pos is not None, target_pos if target_pos is not None else 0,
        out, pos_buf,
    )
    return out[:n].tolist()


def test_marker():
    m = np.asarray([90, 91], dtype=np.int32)
    seq = lambda xs: np.asarray(xs, dtype=np.int32)
    # last occurrence (end-of-prompt marker)
    assert _last_marker_end(seq([90, 91, 1, 2]), m) == 2
    assert _last_marker_end(seq([1, 90, 91]), m) == 3
    assert _last_marker_end(seq([90, 91, 5, 90, 91, 7]), m) == 5  # last one
    assert _last_marker_end(seq([1, 2, 3]), m) == -1
    assert _last_marker_end(seq([90]), m) == -1
    assert _last_marker_end(seq([1, 2]), np.zeros(0, dtype=np.int32)) == -1
    # first occurrence (instruction-prefix marker)
    assert _first_marker_end(seq([90, 91, 1, 2]), m) == 2  # at index 0
    assert _first_marker_end(seq([1, 90, 91]), m) == 3
    assert _first_marker_end(seq([90, 91, 5, 90, 91, 7]), m) == 2  # first one
    assert _first_marker_end(seq([1, 2, 3]), m) == -1
    assert _first_marker_end(seq([90]), m) == -1  # marker longer than tokens
    assert _first_marker_end(seq([1, 2]), np.zeros(0, dtype=np.int32)) == -1
    # first-vs-last divergence on a twice-occurring marker
    twice = seq([5, 90, 91, 6, 90, 91, 7])
    assert _first_marker_end(twice, m) == 3
    assert _last_marker_end(twice, m) == 6
    print("marker OK")


def test_target_position_math():
    # generation not started / just started: near the source start
    assert compute_target_pos(0, 10, 3) == 3
    assert compute_target_pos(1, 10, 3) == 4
    # tracks generation 1:1 (frac * source_len == generated when in range)
    assert compute_target_pos(7, 10, 3) == 10
    assert compute_target_pos(1, 3, 0) == 1  # float roundoff (1/3*3) safe
    assert compute_target_pos(2, 3, 0) == 2
    # generated past the source: clamped to the end of the span
    assert compute_target_pos(15, 10, 3) == 13
    assert compute_target_pos(10, 10, 3) == 13
    # empty/invalid source span short-circuits
    assert compute_target_pos(5, 0, 3) is None
    assert compute_target_pos(5, -2, 3) is None
    # round-half-up, not banker's rounding: int(x + 0.5) rounds 2.5 -> 3,
    # whereas round() would give 2. The kernel and this reference must both
    # use int(x + 0.5) so exact .5 values (however they arise) round up.
    assert int(2.5 + 0.5) == 3 and round(2.5) == 2  # divergence exists
    assert int(3.5 + 0.5) == 4 and round(3.5) == 4  # agrees on odd halves
    print("target-position math OK")


def test_differential_pa(num_cases=20000, seed=42):
    rng = random.Random(seed)
    checked_nonempty = 0
    checked_pa = 0
    for case in range(num_cases):
        vocab = rng.choice([3, 5, 8, 30])  # small vocab => frequent matches
        total = rng.randint(1, 120)
        tokens = [rng.randrange(vocab) for _ in range(total)]
        prompt_len = rng.randint(0, total)
        min_n = rng.randint(1, 4)
        max_n = rng.randint(min_n, min_n + 4)
        k = rng.choice([1, 4, 10])
        max_model_len = rng.choice([total + rng.randint(0, 20), 8192])
        if rng.random() < 0.25:
            target_pos = None  # PA disabled: earliest-position tie-break
        else:
            # any target, including slightly out-of-range values, must work
            target_pos = rng.randint(-2, prompt_len + 2)

        expected = brute_force_pa(
            tokens, prompt_len, min_n, max_n, max_model_len, k, target_pos
        )
        got = run_kernel(
            tokens, prompt_len, min_n, max_n, max_model_len, k, target_pos
        )
        assert got == expected, (
            f"case {case}: tokens={tokens} prompt_len={prompt_len} "
            f"min_n={min_n} max_n={max_n} k={k} max_model_len={max_model_len} "
            f"target_pos={target_pos}\n"
            f"expected={expected}\ngot     ={got}"
        )
        if expected:
            checked_nonempty += 1
            if target_pos is not None:
                checked_pa += 1
    assert checked_nonempty > num_cases // 10, (
        f"only {checked_nonempty} non-empty cases -- test data too sparse"
    )
    assert checked_pa > num_cases // 20, (
        f"only {checked_pa} non-empty PA cases -- test data too sparse"
    )
    print(
        f"differential: {num_cases} cases OK "
        f"({checked_nonempty} non-empty, {checked_pa} with PA tie-break)"
    )


def test_pa_equals_prompt_ngram_when_disabled(num_cases=20000, seed=7):
    """With pa_enabled=False the kernel must be byte-identical to
    gec_prompt_ngram._propose_one (test-only cross-import; the shipped
    module never imports it)."""
    from gec_prompt_ngram import _propose_one

    rng = random.Random(seed)
    for case in range(num_cases):
        vocab = rng.choice([3, 5, 8, 30])
        total = rng.randint(1, 120)
        tokens = [rng.randrange(vocab) for _ in range(total)]
        prompt_len = rng.randint(0, total)
        if rng.random() < 0.25:
            prompt_len = total  # full-context path, for parity with its tests
        min_n = rng.randint(1, 4)
        max_n = rng.randint(min_n, min_n + 4)
        k = rng.choice([1, 4, 10])
        max_model_len = rng.choice([total + rng.randint(0, 20), 8192])

        arr = np.asarray(tokens, dtype=np.int32)
        out_ref = np.zeros(k, dtype=np.int32)
        n_ref = _propose_one(
            arr, prompt_len, min_n, max_n, max_model_len, k, out_ref
        )
        got = run_kernel(
            tokens, prompt_len, min_n, max_n, max_model_len, k, None
        )
        assert got == out_ref[:n_ref].tolist(), (
            f"case {case}: tokens={tokens} prompt_len={prompt_len} "
            f"min_n={min_n} max_n={max_n} k={k} max_model_len={max_model_len}\n"
            f"prompt_ngram={out_ref[:n_ref].tolist()}\npa          ={got}"
        )
    print(f"pa==prompt_ngram when disabled: {num_cases} cases OK")


def test_tie_selection_deterministic():
    """3 tied longest matches at known draft starts: 2, 6, 10."""
    prompt = [1, 2, 10, 0, 1, 2, 20, 0, 1, 2, 30, 0, 90, 91]
    generated = [1, 2]
    tokens = prompt + generated
    prompt_len = len(prompt)

    def draft(target_pos):
        return run_kernel(tokens, prompt_len, 2, 5, 8192, 1, target_pos)

    assert draft(2) == [10]
    assert draft(6) == [20]
    assert draft(10) == [30]
    assert draft(0) == [10]     # before all candidates -> nearest is first
    assert draft(100) == [30]   # past all candidates -> nearest is last
    assert draft(4) == [10]     # exact midpoint of 2 and 6 -> earliest wins
    assert draft(8) == [20]     # exact midpoint of 6 and 10 -> earliest wins
    assert draft(5) == [20]     # strictly nearer to 6
    # earliest-position mode ignores position entirely
    assert run_kernel(tokens, prompt_len, 2, 5, 8192, 1, None) == [10]
    print("tie selection OK")


def test_gec_shape_case_pa():
    """Realistic prompt shape: prefix marker + source + end marker. The same
    ambiguous bigram occurs twice in the source; the selected draft must flip
    from the early to the late occurrence as generation progresses."""
    prefix_marker = np.asarray([80, 81], dtype=np.int32)
    marker = np.asarray([90, 91, 92], dtype=np.int32)
    source = [1, 2, 50, 5, 6, 7, 8, 9, 1, 2, 60, 11]
    prompt = [80, 81] + source + [90, 91, 92]
    prompt_len = len(prompt)  # 17
    max_model_len = 256
    k = 2

    def propose(generated):
        tokens = prompt + generated
        token_ids = np.zeros((1, max_model_len), dtype=np.int32)
        token_ids[0, :len(tokens)] = tokens
        num_tokens = np.asarray([len(tokens)], dtype=np.int32)
        draft_buf = np.zeros((1, k), dtype=np.int32)
        len_buf = np.zeros(1, dtype=np.int32)
        pos_buf = np.zeros(max_model_len, dtype=np.int32)
        no_marker, no_span = _batch_propose(
            np.asarray([0], dtype=np.int64), num_tokens, token_ids,
            marker, prefix_marker, True, 2, 5, max_model_len, k,
            draft_buf, len_buf, pos_buf,
        )
        assert no_marker == 0 and no_span == 0, (no_marker, no_span)
        return draft_buf[0, :len_buf[0]].tolist()

    # Early in generation (2 tokens in): target sits near the source start,
    # so the tie [1,2]@start4 vs [1,2]@start12 resolves to the early one.
    assert propose([1, 2]) == [50, 5]
    # Later (9 tokens in): target has advanced past the middle of the span,
    # so the same tie now resolves to the late occurrence.
    assert propose([1, 2, 50, 5, 6, 7, 8, 1, 2]) == [60, 11]
    print("gec-shaped case OK")


def test_batch_and_fallback_pa():
    max_model_len = 256
    k = 2
    marker = np.asarray([90, 91], dtype=np.int32)
    prefix_marker = np.asarray([80, 81], dtype=np.int32)
    token_ids = np.zeros((3, max_model_len), dtype=np.int32)
    num_tokens = np.zeros(3, dtype=np.int32)

    # request 0: both markers, a 2-way tie, generation far along -> the
    # PA tie-break picks the *late* occurrence (earliest would pick 20).
    row0 = ([80, 81] + [1, 2, 20, 5, 6, 7, 8, 9, 1, 2, 30, 11] + [90, 91]
            + [1, 2, 20, 5, 6, 7, 8, 1, 2])
    token_ids[0, :len(row0)] = row0
    num_tokens[0] = len(row0)
    # request 1: no end-of-prompt marker -> no draft
    row1 = [80, 81, 1, 2, 3, 1, 2]
    token_ids[1, :len(row1)] = row1
    num_tokens[1] = len(row1)
    # request 2: end marker but no prefix marker -> earliest-position
    # fallback picks the *early* occurrence of the same shaped tie.
    row2 = ([1, 2, 20, 5, 6, 7, 8, 9, 1, 2, 30, 11] + [90, 91]
            + [1, 2, 20, 5, 6, 7, 8, 1, 2])
    token_ids[2, :len(row2)] = row2
    num_tokens[2] = len(row2)

    draft_buf = np.zeros((3, k), dtype=np.int32)
    len_buf = np.zeros(3, dtype=np.int32)
    pos_buf = np.zeros(max_model_len, dtype=np.int32)
    valid = np.asarray([0, 1, 2], dtype=np.int64)

    no_marker, no_span = _batch_propose(
        valid, num_tokens, token_ids, marker, prefix_marker, True,
        2, 5, max_model_len, k, draft_buf, len_buf, pos_buf,
    )
    assert no_marker == 1, no_marker
    assert no_span == 1, no_span
    assert len_buf[0] == 2 and draft_buf[0, :2].tolist() == [30, 11]
    assert len_buf[1] == 0
    assert len_buf[2] == 2 and draft_buf[2, :2].tolist() == [20, 5]

    # GEC_PA_TIEBREAK=0 equivalent: prefix search skipped entirely; output is
    # the earliest tie-break regardless of whether a prefix marker exists.
    no_marker, no_span = _batch_propose(
        valid, num_tokens, token_ids, marker, prefix_marker, False,
        2, 5, max_model_len, k, draft_buf, len_buf, pos_buf,
    )
    assert no_marker == 1 and no_span == 0, (no_marker, no_span)
    assert len_buf[0] == 2 and draft_buf[0, :2].tolist() == [20, 5]
    assert len_buf[1] == 0
    assert len_buf[2] == 2 and draft_buf[2, :2].tolist() == [20, 5]
    print("batch/fallback OK")


if __name__ == "__main__":
    test_marker()
    test_target_position_math()
    test_tie_selection_deterministic()
    test_gec_shape_case_pa()
    test_batch_and_fallback_pa()
    test_differential_pa()
    test_pa_equals_prompt_ngram_when_disabled()
    print("all tests passed")
