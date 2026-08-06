# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tag -> token-span parsing, masks, and the abstraction-blind verifier."""

import os

import pytest
import torch
from transformers import AutoTokenizer

from verl.experimental.abstract_rl.parsing import build_masks, parse_one
from verl.experimental.abstract_rl.reward_fn import strip_abstract

MODEL = os.environ.get("ABSTRACT_RL_TEST_MODEL", "/workspace/models/Qwen2.5-Math-1.5B-Instruct")


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(MODEL):
        pytest.skip(f"tokenizer not available at {MODEL}")
    return AutoTokenizer.from_pretrained(MODEL)


def _row(tokenizer, text, pad_to=None):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    mask = [1] * len(ids)
    if pad_to:
        ids = ids + [tokenizer.pad_token_id or 0] * (pad_to - len(ids))
        mask = mask + [0] * (pad_to - len(mask))
    return torch.tensor(ids), torch.tensor(mask)


def _parse(tokenizer, text, **kw):
    ids, mask = _row(tokenizer, text)
    return parse_one(tokenizer, ids, mask, **kw), ids


def _decode_span(tokenizer, ids, lo, hi):
    return tokenizer.decode(ids[lo:hi])


def test_valid_abstraction(tokenizer):
    text = "Step one.\nTherefore \\boxed{42}.\n<abstract>\nUse symmetry.\n</abstract>"
    span, ids = _parse(tokenizer, text)
    assert span.valid
    assert not (span.missing_open or span.missing_close or span.empty_abstract)
    assert "Use symmetry." in span.abstract_text
    # Y holds the answer, A does not
    assert "\\boxed{42}" in _decode_span(tokenizer, ids, 0, span.y_end)
    assert "boxed" not in _decode_span(tokenizer, ids, span.a_start, span.a_end)
    # tags are in neither span
    assert "<abstract>" not in _decode_span(tokenizer, ids, span.a_start, span.a_end)
    assert "</abstract>" not in _decode_span(tokenizer, ids, span.a_start, span.a_end)
    assert "<abstract>" not in _decode_span(tokenizer, ids, 0, span.y_end)


def test_missing_open_tag(tokenizer):
    span, _ = _parse(tokenizer, "Reasoning. \\boxed{1}.\nUse symmetry.\n</abstract>")
    assert span.missing_open and not span.valid and span.n_abstract == 0


def test_missing_close_tag(tokenizer):
    # An open tag with no close is treated as implicitly closed at end-of-response
    # (real generated content, only the delimiter is missing) -- unlike a missing
    # open tag, where there is no safe boundary to fill in.
    span, ids = _parse(tokenizer, "Reasoning. \\boxed{1}.\n<abstract>\nUse symmetry.")
    assert not span.missing_close and span.valid and span.abstract_text.strip() == "Use symmetry."
    # Y still stops before the opening tag
    assert "<abstract>" not in _decode_span(tokenizer, ids, 0, span.y_end)


def test_reversed_tags(tokenizer):
    span, _ = _parse(tokenizer, "Reasoning.\n</abstract>\nUse symmetry.\n<abstract>")
    assert span.reversed_tags and not span.valid


def test_empty_abstraction(tokenizer):
    for body in ("", "\n", "   \n  "):
        span, _ = _parse(tokenizer, f"Reasoning. \\boxed{{1}}.\n<abstract>{body}</abstract>")
        assert span.empty_abstract and not span.valid, body


def test_repeated_blocks_take_the_last_pair(tokenizer):
    text = "R. \\boxed{1}.\n<abstract>first</abstract>\nmore\n<abstract>second</abstract>"
    span, _ = _parse(tokenizer, text)
    assert span.valid and "second" in span.abstract_text and "first" not in span.abstract_text


def test_nested_tags_do_not_crash(tokenizer):
    text = "R. \\boxed{1}.\n<abstract>outer <abstract>inner</abstract> tail</abstract>"
    span, _ = _parse(tokenizer, text)
    assert span.valid and span.n_abstract > 0


def test_boxed_inside_abstract_never_reaches_the_verifier(tokenizer):
    text = "R. Therefore \\boxed{7}.\n<abstract>Compare with \\boxed{99} style problems.</abstract>"
    span, ids = _parse(tokenizer, text)
    assert span.valid
    y_text = _decode_span(tokenizer, ids, 0, span.y_end)
    assert "\\boxed{7}" in y_text and "99" not in y_text
    # the verifier wrapper cuts the same content
    assert strip_abstract(text).rstrip().endswith("\\boxed{7}.")


def test_strip_abstract_is_noop_without_tag():
    assert strip_abstract("plain \\boxed{3}") == "plain \\boxed{3}"


def test_truncated_abstraction_is_recovered(tokenizer):
    # generation hit max_response_length mid-abstraction -- still real content, so
    # it counts rather than being thrown away over the missing closing tag
    span, _ = _parse(tokenizer, "R. \\boxed{1}.\n<abstract>Use symme")
    assert not span.missing_close and span.valid and span.abstract_text == "Use symme"


def test_leading_valid_abstraction(tokenizer):
    text = "<abstract>\nUse symmetry.\n</abstract>\nStep one.\nTherefore \\boxed{42}."
    span, ids = _parse(tokenizer, text, leading=True)
    assert span.valid
    assert "Use symmetry." in span.abstract_text
    # Y (the suffix) holds the answer, A does not
    assert "\\boxed{42}" in _decode_span(tokenizer, ids, span.y_start, span.y_end)
    assert "boxed" not in _decode_span(tokenizer, ids, span.a_start, span.a_end)
    assert "<abstract>" not in _decode_span(tokenizer, ids, span.y_start, span.y_end)
    assert span.y_end == span.response_len


def test_leading_missing_open_is_invalid(tokenizer):
    # no start anchor at all -- the whole response counts as Y, same fallback as
    # the trailing case
    span, _ = _parse(tokenizer, "Step one. Therefore \\boxed{42}.", leading=True)
    assert not span.valid and span.missing_open and span.n_abstract == 0
    assert span.y_start == 0 and span.y_end == span.response_len


def test_leading_missing_close_is_recovered(tokenizer):
    # open present, no close before the response ends -- real content, recovered
    # valid, same leniency as the trailing case (and nothing follows in Y)
    span, _ = _parse(tokenizer, "<abstract>Use symmetry", leading=True)
    assert not span.missing_close and span.valid
    assert span.abstract_text == "Use symmetry"
    assert span.y_start == span.y_end == span.response_len


def test_leading_empty_abstraction_is_invalid(tokenizer):
    span, ids = _parse(tokenizer, "<abstract></abstract>\nStep one. \\boxed{1}.", leading=True)
    assert not span.valid and span.empty_abstract
    # Y still starts right after the closing tag, not swallowed into A
    assert "boxed" in _decode_span(tokenizer, ids, span.y_start, span.y_end)


def test_zero_length_response(tokenizer):
    ids = torch.zeros(8, dtype=torch.long)
    mask = torch.zeros(8, dtype=torch.long)
    span = parse_one(tokenizer, ids, mask)
    assert not span.valid and span.n_abstract == 0 and span.n_reasoning == 0


def test_padding_is_ignored(tokenizer):
    text = "R. \\boxed{1}.\n<abstract>Use symmetry.</abstract>"
    ids, mask = _row(tokenizer, text, pad_to=200)
    span = parse_one(tokenizer, ids, mask)
    assert span.valid and span.a_end <= span.response_len < 200


def test_scan_window_limits_lookback(tokenizer):
    text = "R. \\boxed{1}.\n<abstract>Use symmetry.</abstract>"
    ids, mask = _row(tokenizer, text)
    span = parse_one(tokenizer, ids, mask, max_scan_tokens=3)
    assert not span.valid  # the opening tag is outside the window -> treated as malformed


def test_masks_are_disjoint_and_within_response(tokenizer):
    texts = [
        "R1. \\boxed{1}.\n<abstract>alpha</abstract>",
        "R2 with no abstraction at all. \\boxed{2}.",
        "R3. \\boxed{3}.\n<abstract></abstract>",
    ]
    width = 96
    ids = torch.stack([_row(tokenizer, t, pad_to=width)[0] for t in texts])
    mask = torch.stack([_row(tokenizer, t, pad_to=width)[1] for t in texts])
    spans = [parse_one(tokenizer, ids[i], mask[i]) for i in range(len(texts))]
    reasoning, abstract = build_masks(spans, mask)

    assert not (reasoning & abstract).any(), "Y and A must be disjoint"
    assert not (reasoning & ~mask.bool()).any(), "Y must stay inside the response"
    assert not (abstract & ~mask.bool()).any(), "A must stay inside the response"
    assert abstract[0].sum() > 0 and abstract[1].sum() == 0 and abstract[2].sum() == 0
    # tags belong to neither mask -> the union is a strict subset of the response
    assert (reasoning[0] | abstract[0]).sum() < mask[0].sum()
