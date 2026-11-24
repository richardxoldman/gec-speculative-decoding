"""Prompt-only N-gram speculative-decoding proposer for GEC (vLLM custom_class).

A drop-in replacement for vLLM's CPU ``ngram`` proposer that restricts the
n-gram lookup to the *prompt* (the user turn containing the source text),
excluding everything the model has generated so far. Rationale: GEC output is
a near-copy of the source, so useful draft continuations come from the source
text in the prompt; matches inside the model's own output are usually
repeated corrections (e.g. the same inserted comma) whose continuation is
already-emitted text that the target model rejects, wasting the step's
speculation budget.

Loaded through vLLM's ``custom_class`` speculative method (vLLM >= 0.24), so
no site-packages patching is needed. The server process must be able to
import this module, i.e. this directory must be on PYTHONPATH:

    PYTHONPATH=/path/to/repo/proposers \
    vllm serve <checkpoint> ... \
      --speculative-config '{"method": "custom_class",
                             "model": "gec_prompt_ngram.PromptNgramProposer",
                             "num_speculative_tokens": 5}'

vLLM forces ``prompt_lookup_min/max`` to 0 for custom_class, so n-gram sizes
come from environment variables (set them in the serve environment):

    GEC_NGRAM_MIN        minimum n-gram length (default 2, matching the
                         project's serve_and_benchmark_ngram.sbatch setting)
    GEC_NGRAM_MAX        maximum n-gram length (default 5, ditto)
    GEC_PROMPT_ONLY      "1" (default) restricts search to the prompt;
                         "0" reproduces stock full-context ngram behavior
                         (useful as the A/B control with everything else,
                         including this plugin path, held identical)
    GEC_MARKER_TOKENS    comma-separated token IDs overriding the
                         auto-detected generation-prompt marker
    GEC_MARKER_MAX_LEN   truncate the auto-detected marker to at most this
                         many leading tokens (default 3; covers gemma's
                         "<start_of_turn>model\\n" and qwen's
                         "<|im_start|>assistant\\n" while staying robust to
                         thinking-block template variations after them)

Finding the prompt boundary: vLLM's custom_class ``propose()`` interface
passes neither prompt lengths nor request IDs (and batch slots are reordered
between steps), so the boundary is recovered from the tokens themselves.
Every chat-templated request ends its prompt with the same generation-prompt
marker (e.g. ``<start_of_turn>model\\n``), derived once at startup by
diffing ``apply_chat_template(..., add_generation_prompt=True/False)``. The
marker starts with a special token that user text can never tokenize into,
so its last occurrence unambiguously ends the prompt region. If the marker
is not found in a sequence (e.g. a raw /v1/completions request), that
request falls back to stock full-context search.

Verification is unchanged, so with temperature=0 the generated text is
bit-for-bit identical to stock ngram / autoregressive decoding; only
acceptance rate and drafting cost can differ.

Not one of the methods reported in the paper: included only as the differential
reference that proposers/test_gec_pa_ngram.py checks PANgram against.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:  # avoid importing vllm at module import time (keeps tests light)
    from vllm.config import VllmConfig

logger = logging.getLogger("gec_prompt_ngram")


@njit(cache=True)
def _last_marker_end(tokens: np.ndarray, marker: np.ndarray) -> int:
    """Index right after the last occurrence of ``marker`` in ``tokens``, or -1."""
    n = tokens.shape[0]
    m = marker.shape[0]
    if m == 0 or n < m:
        return -1
    for start in range(n - m, -1, -1):
        found = True
        for j in range(m):
            if tokens[start + j] != marker[j]:
                found = False
                break
        if found:
            return start + m
    return -1


@njit(cache=True)
def _propose_one(
    origin_tokens: np.ndarray,
    prompt_len: int,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
    out_row: np.ndarray,
) -> int:
    """Longest-suffix n-gram match restricted to ``origin_tokens[:prompt_len]``.

    Same algorithm as vLLM's ``_find_longest_matched_ngram_and_propose_tokens``
    (reversed-sequence capped KMP; longest match wins, ties go to the earliest
    original position), with two changes:

    - only matches lying entirely inside the prompt (and leaving at least one
      prompt token to draft) are eligible;
    - drafted tokens are capped at the prompt boundary, never crossing into
      the generated region.

    With ``prompt_len == len(origin_tokens)`` this reproduces the stock
    full-context behavior exactly.

    Writes up to ``k`` draft tokens into ``out_row`` and returns their count.
    """
    total = origin_tokens.shape[0]
    if total < min_ngram or prompt_len < min_ngram + 1:
        return 0
    if max_model_len - total < k:
        k = max_model_len - total
    if k <= 0:
        return 0

    # Reversed view: the goal becomes matching a prefix (the old suffix)
    # elsewhere in the sequence.
    tokens = origin_tokens[::-1]
    # The prompt occupies reversed indices [total - prompt_len, total).
    boundary = total - prompt_len

    pattern_len = max_ngram if max_ngram < total else total
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # Capped-KMP scan of the reversed sequence against its own prefix (the
    # current suffix), as in stock vLLM. Differences:
    #
    # - a match of length l ending at reversed position i is only eligible if
    #   its draft start lies inside the prompt: total - 1 - i + l < prompt_len,
    #   i.e. l <= i - boundary. When the maximal match at a position is too
    #   long to fit, a shorter prefix-suffix at the same position may still
    #   fit -- those are exactly the LPS chain of prev_lps, so walk it down
    #   (at most max_ngram steps) to the longest eligible length.
    # - once the pattern's LPS table is built (i == pattern_len), the rest of
    #   the generated region is skipped: every eligible match lies strictly
    #   inside the prompt region, so resetting the KMP state there is
    #   lossless. With prompt_len == total (boundary 0) nothing is skipped
    #   and no candidate is ever rejected: stock behavior.
    prev_lps = 0
    i = 1
    while i < total:
        if i == pattern_len and boundary > pattern_len:
            i = boundary
            prev_lps = 0
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            cap = i - boundary
            if cap > 0:
                length = prev_lps
                while length > cap:
                    length = lps[length - 1]
                if length >= longest_ngram and length >= 1:
                    longest_ngram = length
                    position = i
            if i < pattern_len:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return 0

    start = total - 1 - position + longest_ngram
    if prompt_len - start < k:
        k = prompt_len - start
    if k <= 0:
        return 0
    for j in range(k):
        out_row[j] = origin_tokens[start + j]
    return k


@njit(cache=True)
def _batch_propose(
    valid_indices: np.ndarray,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    marker: np.ndarray,
    prompt_only: bool,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
    draft_buf: np.ndarray,
    len_buf: np.ndarray,
) -> int:
    """Propose for every request in ``valid_indices``.

    Returns the number of requests where the marker was not found and the
    search fell back to the full context.
    """
    num_fallbacks = 0
    for j in range(valid_indices.shape[0]):
        idx = valid_indices[j]
        num_tokens = num_tokens_no_spec[idx]
        context = token_ids_cpu[idx, :num_tokens]
        prompt_len = num_tokens
        if prompt_only:
            marker_end = _last_marker_end(context, marker)
            if marker_end >= 0:
                prompt_len = marker_end
            else:
                num_fallbacks += 1
        len_buf[idx] = _propose_one(
            context,
            prompt_len,
            min_ngram,
            max_ngram,
            max_model_len,
            k,
            draft_buf[idx],
        )
    return num_fallbacks


class PromptNgramProposer:
    """vLLM custom_class speculative proposer. See module docstring."""

    def __init__(self, vllm_config: "VllmConfig"):
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        self.k = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        self.min_n = int(os.environ.get("GEC_NGRAM_MIN", "2"))
        self.max_n = int(os.environ.get("GEC_NGRAM_MAX", "5"))
        if not (1 <= self.min_n <= self.max_n):
            raise ValueError(
                f"Need 1 <= GEC_NGRAM_MIN <= GEC_NGRAM_MAX, "
                f"got {self.min_n}/{self.max_n}"
            )
        self.prompt_only = os.environ.get("GEC_PROMPT_ONLY", "1") != "0"

        self._marker = self._derive_marker(vllm_config)
        if self.prompt_only and self._marker.shape[0] == 0:
            logger.warning(
                "No generation-prompt marker available; every request will "
                "fall back to full-context search (stock ngram behavior)."
            )

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._draft_buf = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self._len_buf = np.zeros(max_num_seqs, dtype=np.int32)
        self._fallbacks_seen = 0
        self._fallback_logged = False

        # Trigger numba JIT compilation with dummy data of the real dtypes.
        self.propose(
            [[0]] * 4,
            np.full(4, 8, dtype=np.int32),
            np.zeros((4, self.max_model_len), dtype=np.int32),
        )
        self._fallbacks_seen = 0  # dummy data has no marker; don't count it

        logger.info(
            "PromptNgramProposer ready: k=%d, ngram=[%d,%d], prompt_only=%s, "
            "marker=%s",
            self.k,
            self.min_n,
            self.max_n,
            self.prompt_only,
            self._marker.tolist(),
        )

    def _derive_marker(self, vllm_config: "VllmConfig") -> np.ndarray:
        override = os.environ.get("GEC_MARKER_TOKENS")
        if override is not None:
            ids = [int(x) for x in override.split(",") if x.strip()]
            return np.asarray(ids, dtype=np.int32)
        if not self.prompt_only:
            return np.zeros(0, dtype=np.int32)

        try:
            from vllm.tokenizers import cached_tokenizer_from_config

            tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
            messages = [{"role": "user", "content": "x"}]
            kwargs = {"tokenize": True}
            try:
                with_gen = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    **kwargs,
                )
                without_gen = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    enable_thinking=False,
                    **kwargs,
                )
            except TypeError:
                # Template/tokenizer without enable_thinking support.
                with_gen = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, **kwargs
                )
                without_gen = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, **kwargs
                )
        except Exception:
            logger.exception(
                "Could not derive the generation-prompt marker from the "
                "tokenizer; set GEC_MARKER_TOKENS explicitly."
            )
            return np.zeros(0, dtype=np.int32)

        # Depending on the transformers version, tokenize=True returns either
        # a plain list[int] or a BatchEncoding.
        if hasattr(with_gen, "input_ids"):
            with_gen = with_gen.input_ids
        if hasattr(without_gen, "input_ids"):
            without_gen = without_gen.input_ids

        lcp = 0
        for a, b in zip(without_gen, with_gen):
            if a != b:
                break
            lcp += 1
        marker = list(with_gen[lcp:])
        max_len = int(os.environ.get("GEC_MARKER_MAX_LEN", "3"))
        marker = marker[:max_len]
        if marker:
            try:
                rendered = tokenizer.decode(marker)
            except Exception:
                rendered = "<decode failed>"
            logger.info(
                "Derived generation-prompt marker %s (%r)", marker, rendered
            )
        return np.asarray(marker, dtype=np.int32)

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings=None,  # unused, part of the custom_class interface
    ) -> list[list[int]]:
        num_requests = len(sampled_token_ids)
        valid: list[int] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                continue  # partial prefill / discarded request
            if num_tokens_no_spec[i] >= self.max_model_len:
                continue
            valid.append(i)

        draft_token_ids: list[list[int]] = [[] for _ in range(num_requests)]
        if not valid:
            return draft_token_ids

        valid_arr = np.asarray(valid, dtype=np.int64)
        num_fallbacks = _batch_propose(
            valid_arr,
            num_tokens_no_spec,
            token_ids_cpu,
            self._marker,
            self.prompt_only and self._marker.shape[0] > 0,
            self.min_n,
            self.max_n,
            self.max_model_len,
            self.k,
            self._draft_buf,
            self._len_buf,
        )
        if num_fallbacks:
            self._fallbacks_seen += num_fallbacks
            if not self._fallback_logged:
                self._fallback_logged = True
                logger.warning(
                    "Generation-prompt marker not found in %d request(s); "
                    "those fall back to full-context search. Expected for "
                    "non-chat requests, otherwise check GEC_MARKER_TOKENS.",
                    num_fallbacks,
                )

        for i in valid:
            count = self._len_buf[i]
            if count > 0:
                draft_token_ids[i] = self._draft_buf[i, :count].tolist()
        return draft_token_ids

    def load_model(self, *args, **kwargs):
        """No model to load."""
