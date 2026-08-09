# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Locate ``<abstract>...</abstract>`` inside a rollout, on token ids.

The spans are resolved on the *generated* token ids, never by re-tokenizing a
decoded string: the abstraction tokens have to line up 1:1 with the tokens the
prior scorer sees, and a re-tokenization is not guaranteed to reproduce the
generated segmentation.

Offsets come from ``tokenizer.batch_decode([[t] for t in ids])`` -- one batched
call into the fast tokenizer. Concatenating single-token decodes can differ from
decoding the whole sequence when a multi-byte character straddles two tokens
(each half decodes to U+FFFD), but that is harmless here: the search happens
inside that same concatenation, so the char->token table stays self-consistent,
and the tags themselves are pure ASCII and never straddle a replacement char.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class AbstractSpan:
    """Token spans of one rollout, as absolute indices into the response.

    ``Y = [y_start, y_end)`` and ``A = [a_start, a_end)``. A token that straddles a
    tag boundary belongs to neither span (at most one token per boundary), which
    keeps the tags out of the compression-KL mask. ``y_start`` is 0 in the default
    "trailing" layout (abstract after the reasoning); "leading" layout (abstract
    first) instead has ``y_start = a_end`` and ``y_end = response_len``.
    """

    response_len: int
    y_end: int
    a_start: int
    a_end: int
    valid: bool
    missing_open: bool
    missing_close: bool
    reversed_tags: bool
    empty_abstract: bool
    abstract_text: str = ""
    y_start: int = 0
    # |A_i| > max_abstract_tokens (config.py). Distinct from empty/missing so
    # metrics can tell "wrote too much" apart from "wrote nothing" -- see plan
    # section 15 (the model learned to dump the whole solution into A when
    # nothing capped its length).
    too_long: bool = False

    @property
    def n_reasoning(self) -> int:
        return max(0, self.y_end - self.y_start)

    @property
    def n_abstract(self) -> int:
        return max(0, self.a_end - self.a_start)


def _valid_length(response_mask_row: torch.Tensor) -> int:
    """Number of response tokens, i.e. one past the last unmasked position."""
    nz = torch.nonzero(response_mask_row, as_tuple=False)
    return 0 if nz.numel() == 0 else int(nz[-1].item()) + 1


def parse_one(
    tokenizer,
    response_ids_row: torch.Tensor,
    response_mask_row: torch.Tensor,
    open_tag: str = "<abstract>",
    close_tag: str = "</abstract>",
    max_scan_tokens: int = 512,
    leading: bool = False,
    max_abstract_tokens: int = 0,
) -> AbstractSpan:
    span = _parse_one_raw(tokenizer, response_ids_row, response_mask_row, open_tag, close_tag, max_scan_tokens, leading)
    if max_abstract_tokens > 0 and span.n_abstract > max_abstract_tokens:
        return replace(span, valid=False, too_long=True)
    return span


def _parse_one_raw(
    tokenizer,
    response_ids_row: torch.Tensor,
    response_mask_row: torch.Tensor,
    open_tag: str,
    close_tag: str,
    max_scan_tokens: int,
    leading: bool,
) -> AbstractSpan:
    length = _valid_length(response_mask_row)
    if length == 0:
        return AbstractSpan(0, 0, 0, 0, False, True, True, False, True)

    if leading:
        return _parse_leading(tokenizer, response_ids_row, length, open_tag, close_tag, max_scan_tokens)

    # ponytail: suffix-only scan -- the abstract is always at the end of the
    # response. Raise max_scan_tokens if missing_open_tag_rate looks implausible.
    win_start = max(0, length - max_scan_tokens)
    ids = response_ids_row[win_start:length].tolist()
    pieces = tokenizer.batch_decode([[t] for t in ids])

    # offsets[j] is the char index at which token j starts; offsets[-1] == len(text)
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    text = "".join(pieces)

    def _span(y_end_c: int | None, a_lo_c: int | None, a_hi_c: int | None, **flags) -> AbstractSpan:
        # Y ends at the last token lying entirely before y_end_c, i.e. the number
        # of tokens j with offsets[j + 1] <= y_end_c.
        y_end = length if y_end_c is None else win_start + max(0, bisect_right(offsets, y_end_c) - 1)
        if a_lo_c is None or a_hi_c is None:
            a_start = a_end = y_end
        else:
            a_start = win_start + bisect_left(offsets, a_lo_c)
            a_end = win_start + max(0, bisect_right(offsets, a_hi_c) - 1)
            a_end = max(a_end, a_start)
        abstract_text = text[a_lo_c:a_hi_c] if a_lo_c is not None and a_hi_c is not None else ""
        empty = flags.pop("empty_abstract", a_end <= a_start)
        valid = not (flags.get("missing_open") or flags.get("missing_close") or empty)
        return AbstractSpan(
            response_len=length,
            y_end=min(y_end, length),
            a_start=min(a_start, length),
            a_end=min(a_end, length),
            valid=valid,
            empty_abstract=empty,
            abstract_text=abstract_text,
            **flags,
        )

    # Last well-formed pair wins: find the last close tag, then the last open tag
    # before it. Repeated or nested blocks therefore resolve to the outermost-last
    # opening that precedes the final closing tag.
    close_c = text.rfind(close_tag)
    open_c = text.rfind(open_tag, 0, close_c) if close_c >= 0 else text.rfind(open_tag)

    if open_c < 0 and close_c < 0:
        return _span(None, None, None, missing_open=True, missing_close=True, reversed_tags=False)

    if open_c < 0:
        # A close tag with no open before it -- we have no idea where the abstraction
        # was meant to start, so unlike the missing-close case below there is no safe
        # implicit boundary to fill in (guessing "everything before it" would swallow
        # the reasoning/boxed-answer into the abstract span). Stays invalid.
        later_open = text.find(open_tag, close_c)
        return _span(close_c, None, None, missing_open=True, missing_close=False, reversed_tags=later_open >= 0)

    if close_c < 0:
        # Open tag but no close (truncated by max_response_length, or the model just
        # never emitted it). The content after the open tag is real generated text;
        # treat end-of-response as an implicit close rather than discarding it.
        inner_lo = open_c + len(open_tag)
        return _span(open_c, inner_lo, len(text), missing_open=False, missing_close=False, reversed_tags=False)

    inner_lo = open_c + len(open_tag)
    if not text[inner_lo:close_c].strip():
        return _span(
            open_c, None, None, missing_open=False, missing_close=False, reversed_tags=False, empty_abstract=True
        )

    return _span(open_c, inner_lo, close_c, missing_open=False, missing_close=False, reversed_tags=False)


def _parse_leading(
    tokenizer,
    response_ids_row: torch.Tensor,
    length: int,
    open_tag: str,
    close_tag: str,
    max_scan_tokens: int,
) -> AbstractSpan:
    """Mirror of the trailing scan above, but abstract-first: prefix window, first
    open/close pair wins, and Y is the suffix that follows the abstract instead of
    the prefix that precedes it.
    """
    win_end = min(length, max_scan_tokens)
    ids = response_ids_row[:win_end].tolist()
    pieces = tokenizer.batch_decode([[t] for t in ids])
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    text = "".join(pieces)

    def _span(a_lo_c: int, a_hi_c: int, y_start_c: int, **flags) -> AbstractSpan:
        # a_end stops right before its tag (bisect_right - 1); y_start uses
        # bisect_left so it begins at the first token starting >= y_start_c, i.e.
        # right after that same tag -- the two must NOT share a boundary formula,
        # a_hi_c and y_start_c are different char positions (before vs after the tag).
        a_start = bisect_left(offsets, a_lo_c)
        a_end = max(a_start, max(0, bisect_right(offsets, a_hi_c) - 1))
        y_start = min(length, bisect_left(offsets, y_start_c))
        empty = flags.pop("empty_abstract", a_end <= a_start)
        valid = not (flags.get("missing_open") or flags.get("missing_close") or empty)
        return AbstractSpan(
            response_len=length,
            y_end=length,
            a_start=min(a_start, length),
            a_end=min(a_end, length),
            valid=valid,
            empty_abstract=empty,
            abstract_text=text[a_lo_c:a_hi_c],
            y_start=y_start,
            **flags,
        )

    # First pair wins (the abstract is meant to be first): first open, then the
    # first close after it.
    open_c = text.find(open_tag)
    if open_c < 0:
        # No start anchor -- same as the trailing case's "nothing found" fallback,
        # the whole response counts as Y.
        return AbstractSpan(length, length, 0, 0, False, True, True, False, True, y_start=0)

    inner_lo = open_c + len(open_tag)
    close_c = text.find(close_tag, inner_lo)
    if close_c < 0:
        # Open present but the window (or the response) ended before it closed --
        # real content, only the delimiter is missing; same leniency as the
        # trailing case's missing-close branch. Nothing follows in the window, so Y
        # is empty (y_start = y_end = length).
        return _span(inner_lo, len(text), len(text), missing_open=False, missing_close=False, reversed_tags=False)

    y_start_c = close_c + len(close_tag)
    if not text[inner_lo:close_c].strip():
        return _span(
            inner_lo,
            close_c,
            y_start_c,
            missing_open=False,
            missing_close=False,
            reversed_tags=False,
            empty_abstract=True,
        )

    return _span(inner_lo, close_c, y_start_c, missing_open=False, missing_close=False, reversed_tags=False)


def parse_batch(
    tokenizer,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    open_tag: str = "<abstract>",
    close_tag: str = "</abstract>",
    max_scan_tokens: int = 512,
    leading: bool = False,
    max_abstract_tokens: int = 0,
) -> list[AbstractSpan]:
    return [
        parse_one(
            tokenizer,
            response_ids[i],
            response_mask[i],
            open_tag,
            close_tag,
            max_scan_tokens,
            leading,
            max_abstract_tokens,
        )
        for i in range(response_ids.shape[0])
    ]


def build_masks(spans: list[AbstractSpan], response_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(reasoning_mask, abstract_mask)``, both shaped like ``response_mask``.

    The two are disjoint and both are subsets of ``response_mask``; the tag tokens
    (and any token straddling a tag boundary) are in neither. ``response_mask``
    itself -- Y u A u tags -- stays the GRPO mask and is not modified here.
    """
    reasoning = torch.zeros_like(response_mask, dtype=torch.bool)
    abstract = torch.zeros_like(response_mask, dtype=torch.bool)
    for i, span in enumerate(spans):
        if span.y_end > span.y_start:
            reasoning[i, span.y_start : span.y_end] = True
        if span.a_end > span.a_start:
            abstract[i, span.a_start : span.a_end] = True
    keep = response_mask.bool()
    return reasoning & keep, abstract & keep
