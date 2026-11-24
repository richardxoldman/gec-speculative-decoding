"""Shallow-Aggressive-Decoding-style speculative proposer for GEC (vLLM custom_class).

Implements the drafting trigger of "Instantaneous Grammatical Error Correction
with Shallow Aggressive Decoding" (Sun et al., ACL 2021) as a vLLM speculative
proposer: at each decoding step, take the suffix ``o_{j-q..j}`` of the output
generated so far and try to find it in the input (source) text ``x``. Draft
only when the suffix match is *unique* in the input -- find the smallest
``q >= 1`` such that ``o_{j-q..j}`` occurs exactly once in ``x`` as
``x_{i-q..i}``, then propose ``x_{i+1..i+k}`` as the draft. If the occurrence
count drops from >=2 straight to 0 while growing ``q`` (or the search cap is
hit while still ambiguous), nothing is proposed for that step.

Deviations from the paper, forced by vLLM:

- **Fixed draft length k.** The paper drafts all the way to the end of the
  input; vLLM truncates every draft to the launch-time
  ``num_speculative_tokens``, so this proposer drafts up to ``k`` tokens per
  step (capped at the prompt boundary), exactly like ngram decoding. ``k`` is
  the only speculation-length knob and is configured normally via
  ``--speculative-config``.
- **No step-0 verification of the whole input.** vLLM only calls the proposer
  after the first token has been sampled, so the paper's initial "assume
  output == input" pass is replaced by ordinary drafting from the first step
  onward (costs one extra decoding step per request).

There are no n-gram length knobs (``prompt_lookup_min/max`` do not apply);
uniqueness determines the match length. Verification is unchanged, so with
temperature=0 the generated text is bit-for-bit identical to stock ngram /
autoregressive decoding; only acceptance rate and drafting cost can differ.

Loaded through vLLM's ``custom_class`` speculative method (vLLM >= 0.24), so
no site-packages patching is needed. The server process must be able to
import this module, i.e. this directory must be on PYTHONPATH:

    PYTHONPATH=/path/to/repo/proposers \
    vllm serve <checkpoint> ... \
      --speculative-config '{"method": "custom_class",
                             "model": "gec_sad_ngram.SADProposer",
                             "num_speculative_tokens": 5}'

Environment variables (set them in the serve environment):

    GEC_SAD_MAX_SUFFIX   cap on the uniqueness-search suffix length q
                         (default 64); guards the worst case on highly
                         repetitive text where many candidates survive long
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
request proposes nothing -- SAD is undefined without a prompt boundary.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:  # avoid importing vllm at module import time (keeps tests light)
    from vllm.config import VllmConfig

logger = logging.getLogger("gec_sad_ngram")


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
def _propose_one_sad(
    origin_tokens: np.ndarray,
    prompt_len: int,
    max_suffix: int,
    max_model_len: int,
    k: int,
    out_row: np.ndarray,
    cand_buf: np.ndarray,
) -> int:
    """Unique-suffix match of the context tail against ``origin_tokens[:prompt_len]``.

    Grows the context suffix one token at a time (q = 1, 2, ...) and keeps the
    prompt positions where it matches. Drafts only when exactly one candidate
    survives; returns 0 as soon as no candidate survives, or when ``q`` would
    exceed ``max_suffix`` (or the whole context) while still ambiguous.

    A candidate is the *end* index ``i`` of a match ``origin_tokens[i-q+1..i]``;
    only ``i < prompt_len - 1`` is eligible, so the match lies inside the
    prompt and at least one prompt token remains to draft. On a unique match,
    up to ``k`` tokens from ``origin_tokens[i+1:]`` are drafted, capped at the
    prompt boundary (never crossing into the generated region) and at
    ``max_model_len``.

    The suffix is allowed to grow past the start of the generated region into
    the prompt's trailing marker tokens; those are special tokens that cannot
    occur inside the source text, so all candidates die and 0 is returned --
    behaviorally identical to capping q at the generated length.

    ``cand_buf`` is caller-provided scratch (>= prompt_len entries). Writes up
    to ``k`` draft tokens into ``out_row`` and returns their count.
    """
    total = origin_tokens.shape[0]
    if total < 2 or prompt_len < 2:
        return 0
    if max_model_len - total < k:
        k = max_model_len - total
    if k <= 0:
        return 0

    # q = 1: every prompt position matching the last context token.
    last_tok = origin_tokens[total - 1]
    n_cand = 0
    for i in range(prompt_len - 1):
        if origin_tokens[i] == last_tok:
            cand_buf[n_cand] = i
            n_cand += 1

    q_cap = max_suffix if max_suffix < total else total
    q = 1
    while n_cand > 1:
        q += 1
        if q > q_cap:
            return 0  # still ambiguous at the search cap
        suffix_tok = origin_tokens[total - q]
        kept = 0
        for j in range(n_cand):
            i = cand_buf[j]
            if i - q + 1 >= 0 and origin_tokens[i - q + 1] == suffix_tok:
                cand_buf[kept] = i
                kept += 1
        n_cand = kept

    if n_cand == 0:
        return 0

    start = cand_buf[0] + 1  # unique match ends at cand_buf[0]
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
    max_suffix: int,
    max_model_len: int,
    k: int,
    draft_buf: np.ndarray,
    len_buf: np.ndarray,
    cand_buf: np.ndarray,
) -> int:
    """Propose for every request in ``valid_indices``.

    Returns the number of requests where the marker was not found (those
    propose nothing).
    """
    num_no_marker = 0
    for j in range(valid_indices.shape[0]):
        idx = valid_indices[j]
        num_tokens = num_tokens_no_spec[idx]
        context = token_ids_cpu[idx, :num_tokens]
        marker_end = _last_marker_end(context, marker)
        if marker_end < 0:
            num_no_marker += 1
            len_buf[idx] = 0
            continue
        len_buf[idx] = _propose_one_sad(
            context,
            marker_end,
            max_suffix,
            max_model_len,
            k,
            draft_buf[idx],
            cand_buf,
        )
    return num_no_marker


class SADProposer:
    """vLLM custom_class speculative proposer. See module docstring."""

    def __init__(self, vllm_config: "VllmConfig"):
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        self.k = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        self.max_suffix = int(os.environ.get("GEC_SAD_MAX_SUFFIX", "64"))
        if self.max_suffix < 1:
            raise ValueError(f"Need GEC_SAD_MAX_SUFFIX >= 1, got {self.max_suffix}")

        self._marker = self._derive_marker(vllm_config)
        if self._marker.shape[0] == 0:
            logger.warning(
                "No generation-prompt marker available; SAD needs the prompt "
                "boundary, so no request will ever get a draft. Set "
                "GEC_MARKER_TOKENS explicitly."
            )

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._draft_buf = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self._len_buf = np.zeros(max_num_seqs, dtype=np.int32)
        self._cand_buf = np.zeros(self.max_model_len, dtype=np.int32)
        self._no_marker_seen = 0
        self._no_marker_logged = False

        # Trigger numba JIT compilation with dummy data of the real dtypes.
        # The dummy rows contain the marker followed by a fake source with a
        # unique match, so _propose_one_sad's hot paths compile now too.
        self._warm_up()
        self._no_marker_seen = 0  # don't count warmup rows

        logger.info(
            "SADProposer ready: k=%d, max_suffix=%d, marker=%s",
            self.k,
            self.max_suffix,
            self._marker.tolist(),
        )

    def _warm_up(self) -> None:
        rows = 4
        dummy = np.zeros((rows, self.max_model_len), dtype=np.int32)
        marker = self._marker
        m = marker.shape[0]
        num_tokens = np.zeros(rows, dtype=np.int32)
        for r in range(rows):
            pos = 0
            if m:
                dummy[r, :m] = marker
                pos = m
            # fake source "5 6 7 6 8", marker, then generated "5 6" -> the
            # suffix "6" is ambiguous, "5 6" unique: exercises the q-growth
            # loop and the draft-copy path.
            source = [5, 6, 7, 6, 8]
            for t in source:
                dummy[r, pos] = t
                pos += 1
            if m:
                dummy[r, pos : pos + m] = marker
                pos += m
            for t in (5, 6):
                dummy[r, pos] = t
                pos += 1
            num_tokens[r] = pos
        self.propose([[0]] * rows, num_tokens, dummy)

    def _derive_marker(self, vllm_config: "VllmConfig") -> np.ndarray:
        override = os.environ.get("GEC_MARKER_TOKENS")
        if override is not None:
            ids = [int(x) for x in override.split(",") if x.strip()]
            return np.asarray(ids, dtype=np.int32)

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
        if not valid or self._marker.shape[0] == 0:
            return draft_token_ids

        valid_arr = np.asarray(valid, dtype=np.int64)
        num_no_marker = _batch_propose(
            valid_arr,
            num_tokens_no_spec,
            token_ids_cpu,
            self._marker,
            self.max_suffix,
            self.max_model_len,
            self.k,
            self._draft_buf,
            self._len_buf,
            self._cand_buf,
        )
        if num_no_marker:
            self._no_marker_seen += num_no_marker
            if not self._no_marker_logged:
                self._no_marker_logged = True
                logger.warning(
                    "Generation-prompt marker not found in %d request(s); "
                    "those get no drafts. Expected for non-chat requests, "
                    "otherwise check GEC_MARKER_TOKENS.",
                    num_no_marker,
                )

        for i in valid:
            count = self._len_buf[i]
            if count > 0:
                draft_token_ids[i] = self._draft_buf[i, :count].tolist()
        return draft_token_ids

    def load_model(self, *args, **kwargs):
        """No model to load."""
