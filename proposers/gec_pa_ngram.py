"""Position-aware prompt-only N-gram speculative proposer for GEC (vLLM custom_class).

A variant of the prompt-only n-gram proposer (``gec_prompt_ngram.py``) with a
different tie-break among equally long candidate matches. The underlying
search is unchanged -- the longest suffix n-gram match inside the prompt wins,
and a shorter match never beats a longer one -- but when several candidates
tie at the longest length, this proposer picks the one whose draft start is
*closest to where generation currently is*, instead of the earliest one.
Rationale: GEC output tracks the source roughly left-to-right at a pace
proportional to the source's own length, so when the same n-gram occurs in
several places, the occurrence near the current position fraction
(``generated_so_far / source_len``) is the most likely true continuation.

Loaded through vLLM's ``custom_class`` speculative method (vLLM >= 0.24), so
no site-packages patching is needed. The server process must be able to
import this module, i.e. this directory must be on PYTHONPATH:

    PYTHONPATH=/path/to/repo/proposers \\
    vllm serve <checkpoint> ... \\
      --speculative-config '{"method": "custom_class",
                             "model": "gec_pa_ngram.PANgramProposer",
                             "num_speculative_tokens": 5}'

vLLM forces ``prompt_lookup_min/max`` to 0 for custom_class, so all knobs
come from environment variables (set them in the serve environment):

    GEC_NGRAM_MIN               minimum n-gram length (default 2)
    GEC_NGRAM_MAX               maximum n-gram length (default 5)
    GEC_MARKER_TOKENS           comma-separated token IDs overriding the
                                auto-detected generation-prompt (end-of-prompt)
                                marker
    GEC_MARKER_MAX_LEN          truncate the auto-detected end-of-prompt
                                marker to at most this many leading tokens
                                (default 3; covers gemma's
                                "<start_of_turn>model\\n" and qwen's
                                "<|im_start|>assistant\\n" while staying
                                robust to thinking-block template variations)
    GEC_PREFIX_MARKER_TOKENS    comma-separated token IDs overriding the
                                auto-detected instruction-prefix (source-start)
                                marker
    GEC_PREFIX_MARKER_MAX_LEN   truncate the auto-detected prefix marker
                                (default 0 = uncapped -- deliberately different
                                from GEC_MARKER_MAX_LEN: the prefix marker is
                                ordinary text, and truncating it raises the
                                risk of a false-positive match against real
                                source content, whereas the end-of-prompt
                                marker starts with a special control token)
    GEC_PA_TIEBREAK             "1" (default) = position-aware tie-break;
                                "0" = earliest-position tie-break (the A/B
                                control -- byte-identical output to
                                gec_prompt_ngram.py's prompt-only mode, with
                                prefix-marker derivation/search skipped
                                entirely as a pure cost saving)

There is no GEC_PROMPT_ONLY: this proposer is always prompt-restricted
(position-awareness is meaningless without a source scope); use
``gec_prompt_ngram.py`` for a full-context baseline.

Two boundaries are recovered from the tokens themselves each step (vLLM's
custom_class ``propose()`` passes neither prompt lengths nor request IDs, and
batch slots are reordered between steps):

- **End-of-prompt marker** (*last* occurrence): the chat template's
  generation-prompt marker (e.g. ``<|im_start|>assistant\\n``), derived once
  at startup by diffing ``apply_chat_template(add_generation_prompt=
  True/False)``. It starts with a special token user text can never tokenize
  into, so its last occurrence unambiguously ends the prompt. Unlike
  ``gec_prompt_ngram.py`` there is **no full-context fallback** when it is
  missing -- an unbounded search would silently defeat this proposer's whole
  purpose -- such requests simply get no draft (same convention as
  ``gec_sad_ngram.py``).

- **Instruction-prefix marker** (*first* occurrence, searched only within the
  prompt): the tokens of the fixed training-time instruction prefix
  ("Correct all language errors in the following text. Return only the
  corrected text. Text to correct:\\n" -- see ``train_gec_g4.py``), locating
  where the source text starts. Derived once
  at startup by diffing chat-templated dummy prompts: with vs. without the
  prefix (finds where the prefix starts) and two different dummy sources
  (finds where source-dependent tokenization starts, guarding against the
  prefix's trailing newline merging into a source-specific token).
  First-occurrence is used (not last) because the prefix is ordinary text
  with no special-token guarantee, and the real boilerplate always sits at
  the very start of the user turn; a coincidental occurrence *inside* the
  source remains a known theoretical risk, mitigated by the full-length
  (uncapped) marker default. If this marker is missing for a request (or
  yields an empty source span), that request falls back to the
  earliest-position tie-break -- the prompt-restricted search is still
  well-defined, only the tie-break sophistication is lost.

Target position for the tie-break (per request, per step)::

    source_len   = prompt_len - prefix_end
    frac         = clamp((num_tokens - prompt_len) / source_len, 0, 1)
    offset       = int(frac * source_len + 0.5)      # round half up
    target_pos   = prefix_end + min(offset, source_len)

Verification is unchanged, so with temperature=0 the generated text is
bit-for-bit identical to stock ngram / autoregressive decoding; only
acceptance rate and drafting cost can differ.

Standalone module by design (no imports from gec_prompt_ngram / gec_sad_ngram)
so it can be open-sourced independently; the marker-detection logic is
deliberately duplicated, not shared.

Tested by proposers/test_gec_pa_ngram.py (pure CPU, no GPU needed).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:  # avoid importing vllm at module import time (keeps tests light)
    from vllm.config import VllmConfig

logger = logging.getLogger("gec_pa_ngram")

# The training-time instruction prefix prepended to the user turn -- must
# match train_gec_g4.py exactly.
INSTRUCTION_PREFIX = "Correct all language errors in the following text. Return only the corrected text. Text to correct:\n"


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
def _first_marker_end(tokens: np.ndarray, marker: np.ndarray) -> int:
    """Index right after the first occurrence of ``marker`` in ``tokens``, or -1."""
    n = tokens.shape[0]
    m = marker.shape[0]
    if m == 0 or n < m:
        return -1
    for start in range(0, n - m + 1):
        found = True
        for j in range(m):
            if tokens[start + j] != marker[j]:
                found = False
                break
        if found:
            return start + m
    return -1


@njit(cache=True)
def _propose_one_pa(
    origin_tokens: np.ndarray,
    prompt_len: int,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
    pa_enabled: bool,
    target_pos: int,
    out_row: np.ndarray,
    pos_buf: np.ndarray,
) -> int:
    """Longest-suffix n-gram match restricted to ``origin_tokens[:prompt_len]``,
    with a position-aware tie-break among equally long candidates.

    Same reversed-sequence capped-KMP scan as gec_prompt_ngram.py's
    ``_propose_one`` (longest match wins; only matches lying entirely inside
    the prompt and leaving at least one prompt token to draft are eligible;
    drafts never cross the prompt boundary), except that *all* positions tied
    at the current-best length are collected into ``pos_buf`` (reset whenever
    a strictly longer match is found) and the winner is selected after the
    scan ends:

    - ``pa_enabled``: argmin over ``(|start - target_pos|, start)`` -- nearest
      to the target position, earliest on an exact distance tie;
    - otherwise: earliest position (byte-identical to ``_propose_one``).

    ``pos_buf`` is caller-provided scratch (>= len(origin_tokens) entries),
    reused across the batch. Writes up to ``k`` draft tokens into ``out_row``
    and returns their count.
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
    n_tied = 0

    # A match of length l ending at reversed position i is only eligible if
    # its draft start lies inside the prompt (l <= i - boundary); when the
    # maximal match at a position is too long to fit, walk the LPS chain down
    # (at most max_ngram steps) to the longest eligible length. Once the
    # pattern's LPS table is built (i == pattern_len), the rest of the
    # generated region is skipped: every eligible match lies strictly inside
    # the prompt region, so resetting the KMP state there is lossless.
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
                if length >= 1:
                    if length > longest_ngram:
                        longest_ngram = length
                        pos_buf[0] = i
                        n_tied = 1
                    elif length == longest_ngram:
                        pos_buf[n_tied] = i
                        n_tied += 1
            if i < pattern_len:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram or n_tied == 0:
        return 0

    # Selection among the tied longest-length candidates. Distinct reversed
    # positions map to distinct draft starts (start = total-1-i+length), so
    # no dedup is needed.
    best_start = -1
    best_dist = 0
    for j in range(n_tied):
        start_j = total - 1 - pos_buf[j] + longest_ngram
        if pa_enabled:
            d = start_j - target_pos
            if d < 0:
                d = -d
            if (
                best_start < 0
                or d < best_dist
                or (d == best_dist and start_j < best_start)
            ):
                best_start = start_j
                best_dist = d
        else:
            if best_start < 0 or start_j < best_start:
                best_start = start_j

    start = best_start
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
    prefix_marker: np.ndarray,
    pa_tiebreak: bool,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
    draft_buf: np.ndarray,
    len_buf: np.ndarray,
    pos_buf: np.ndarray,
):
    """Propose for every request in ``valid_indices``.

    Returns ``(num_no_prompt_marker, num_no_source_span)``: requests where
    the end-of-prompt marker was not found (those propose nothing), and
    requests where the prefix marker was missing / gave an empty source span
    (those fall back to the earliest-position tie-break).
    """
    num_no_prompt_marker = 0
    num_no_source_span = 0
    for j in range(valid_indices.shape[0]):
        idx = valid_indices[j]
        num_tokens = num_tokens_no_spec[idx]
        context = token_ids_cpu[idx, :num_tokens]
        prompt_len = _last_marker_end(context, marker)
        if prompt_len < 0:
            num_no_prompt_marker += 1
            len_buf[idx] = 0
            continue
        pa_enabled = False
        target_pos = 0
        if pa_tiebreak:
            prefix_end = _first_marker_end(context[:prompt_len], prefix_marker)
            source_len = prompt_len - prefix_end
            if prefix_end < 0 or source_len <= 0:
                num_no_source_span += 1
            else:
                pa_enabled = True
                generated = num_tokens - prompt_len
                frac = generated / source_len
                if frac < 0.0:
                    frac = 0.0
                if frac > 1.0:
                    frac = 1.0
                # Round half up, not Python round() (banker's rounding).
                offset = int(frac * source_len + 0.5)
                if offset > source_len:
                    offset = source_len
                target_pos = prefix_end + offset
        len_buf[idx] = _propose_one_pa(
            context,
            prompt_len,
            min_ngram,
            max_ngram,
            max_model_len,
            k,
            pa_enabled,
            target_pos,
            draft_buf[idx],
            pos_buf,
        )
    return num_no_prompt_marker, num_no_source_span


class PANgramProposer:
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
        self.pa_tiebreak = os.environ.get("GEC_PA_TIEBREAK", "1") != "0"

        self._marker = self._derive_marker(vllm_config)
        if self._marker.shape[0] == 0:
            logger.warning(
                "No generation-prompt marker available; this proposer needs "
                "the prompt boundary, so no request will ever get a draft. "
                "Set GEC_MARKER_TOKENS explicitly."
            )
        self._prefix_marker = self._derive_prefix_marker(vllm_config)
        if self.pa_tiebreak and self._prefix_marker.shape[0] == 0:
            logger.warning(
                "No instruction-prefix marker available; every request will "
                "use the earliest-position tie-break instead of the "
                "position-aware one. Set GEC_PREFIX_MARKER_TOKENS explicitly."
            )
        self._pa_active = self.pa_tiebreak and self._prefix_marker.shape[0] > 0

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._draft_buf = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self._len_buf = np.zeros(max_num_seqs, dtype=np.int32)
        self._pos_buf = np.zeros(self.max_model_len, dtype=np.int32)
        self._no_prompt_marker_seen = 0
        self._no_prompt_marker_logged = False
        self._no_source_span_seen = 0
        self._no_source_span_logged = False

        # Trigger numba JIT compilation with dummy data of the real dtypes.
        self._warm_up()
        self._no_prompt_marker_seen = 0  # don't count warmup rows
        self._no_source_span_seen = 0

        logger.info(
            "PANgramProposer ready: k=%d, ngram=[%d,%d], pa_tiebreak=%s, "
            "marker=%s, prefix_marker=%s",
            self.k,
            self.min_n,
            self.max_n,
            self.pa_tiebreak,
            self._marker.tolist(),
            self._prefix_marker.tolist(),
        )

    def _warm_up(self) -> None:
        rows = 4
        dummy = np.zeros((rows, self.max_model_len), dtype=np.int32)
        marker = self._marker
        prefix_marker = self._prefix_marker
        num_tokens = np.zeros(rows, dtype=np.int32)
        for r in range(rows):
            pos = 0
            pm = prefix_marker.shape[0]
            if pm:
                dummy[r, :pm] = prefix_marker
                pos = pm
            # fake source "5 6 7 5 6 8", then marker, then generated "5 6":
            # the suffix "5 6" ties at two prompt positions, exercising the
            # tie-collection and both selection paths.
            for t in (5, 6, 7, 5, 6, 8):
                dummy[r, pos] = t
                pos += 1
            m = marker.shape[0]
            if m:
                dummy[r, pos : pos + m] = marker
                pos += m
            for t in (5, 6):
                dummy[r, pos] = t
                pos += 1
            num_tokens[r] = pos
        self.propose([[0]] * rows, num_tokens, dummy)

    def _apply_template(self, tokenizer, content: str) -> list[int]:
        messages = [{"role": "user", "content": content}]
        try:
            out = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
            )
        except TypeError:
            # Template/tokenizer without enable_thinking support.
            out = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
        # Depending on the transformers version, tokenize=True returns either
        # a plain list[int] or a BatchEncoding.
        if hasattr(out, "input_ids"):
            out = out.input_ids
        return list(out)

    @staticmethod
    def _lcp(a: list[int], b: list[int]) -> int:
        lcp = 0
        for x, y in zip(a, b):
            if x != y:
                break
            lcp += 1
        return lcp

    def _derive_marker(self, vllm_config: "VllmConfig") -> np.ndarray:
        override = os.environ.get("GEC_MARKER_TOKENS")
        if override is not None:
            ids = [int(x) for x in override.split(",") if x.strip()]
            return np.asarray(ids, dtype=np.int32)

        try:
            from vllm.tokenizers import cached_tokenizer_from_config

            tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
            messages = [{"role": "user", "content": "x"}]
            try:
                with_gen = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                )
                without_gen = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    enable_thinking=False,
                    tokenize=True,
                )
            except TypeError:
                # Template/tokenizer without enable_thinking support.
                with_gen = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                )
                without_gen = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=True
                )
        except Exception:
            logger.exception(
                "Could not derive the generation-prompt marker from the "
                "tokenizer; set GEC_MARKER_TOKENS explicitly."
            )
            return np.zeros(0, dtype=np.int32)

        if hasattr(with_gen, "input_ids"):
            with_gen = with_gen.input_ids
        if hasattr(without_gen, "input_ids"):
            without_gen = without_gen.input_ids

        lcp = self._lcp(list(without_gen), list(with_gen))
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

    def _derive_prefix_marker(self, vllm_config: "VllmConfig") -> np.ndarray:
        override = os.environ.get("GEC_PREFIX_MARKER_TOKENS")
        if override is not None:
            ids = [int(x) for x in override.split(",") if x.strip()]
            return np.asarray(ids, dtype=np.int32)
        if not self.pa_tiebreak:
            # Pure cost saving: the earliest-position control never needs it.
            return np.zeros(0, dtype=np.int32)

        try:
            from vllm.tokenizers import cached_tokenizer_from_config

            tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
            plain = self._apply_template(tokenizer, "x")
            pref_a = self._apply_template(tokenizer, INSTRUCTION_PREFIX + "x")
            pref_b = self._apply_template(tokenizer, INSTRUCTION_PREFIX + "9")
        except Exception:
            logger.exception(
                "Could not derive the instruction-prefix marker from the "
                "tokenizer; set GEC_PREFIX_MARKER_TOKENS explicitly."
            )
            return np.zeros(0, dtype=np.int32)

        # The marker spans from where the prefixed prompt diverges from the
        # unprefixed one (start of the instruction text) to where the two
        # prefixed prompts with different dummy sources diverge (start of
        # source-dependent tokenization) -- the second diff guards against the
        # prefix's trailing newline merging into a source-specific token.
        start = self._lcp(plain, pref_a)
        end = self._lcp(pref_a, pref_b)
        marker = pref_a[start:end]
        max_len = int(os.environ.get("GEC_PREFIX_MARKER_MAX_LEN", "0"))
        if max_len > 0:
            marker = marker[:max_len]
        if marker:
            try:
                rendered = tokenizer.decode(marker)
            except Exception:
                rendered = "<decode failed>"
            logger.info(
                "Derived instruction-prefix marker %s (%r)", marker, rendered
            )
        else:
            logger.warning(
                "Instruction-prefix marker derivation produced an empty "
                "marker (template diff start=%d end=%d).",
                start,
                end,
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
        num_no_prompt_marker, num_no_source_span = _batch_propose(
            valid_arr,
            num_tokens_no_spec,
            token_ids_cpu,
            self._marker,
            self._prefix_marker,
            self._pa_active,
            self.min_n,
            self.max_n,
            self.max_model_len,
            self.k,
            self._draft_buf,
            self._len_buf,
            self._pos_buf,
        )
        if num_no_prompt_marker:
            self._no_prompt_marker_seen += num_no_prompt_marker
            if not self._no_prompt_marker_logged:
                self._no_prompt_marker_logged = True
                logger.warning(
                    "Generation-prompt marker not found in %d request(s); "
                    "those get no drafts. Expected for non-chat requests, "
                    "otherwise check GEC_MARKER_TOKENS.",
                    num_no_prompt_marker,
                )
        if num_no_source_span:
            self._no_source_span_seen += num_no_source_span
            if not self._no_source_span_logged:
                self._no_source_span_logged = True
                logger.warning(
                    "Instruction-prefix marker not found (or empty source "
                    "span) in %d request(s); those use the earliest-position "
                    "tie-break. Expected for prompts without the training "
                    "instruction prefix, otherwise check "
                    "GEC_PREFIX_MARKER_TOKENS.",
                    num_no_source_span,
                )

        for i in valid:
            count = self._len_buf[i]
            if count > 0:
                draft_token_ids[i] = self._draft_buf[i, :count].tolist()
        return draft_token_ids

    def load_model(self, *args, **kwargs):
        """No model to load."""
